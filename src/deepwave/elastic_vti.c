/*
 * Elastic VTI (vertically transverse isotropic) wave equation propagator
 */

/*
 * This file contains the C implementation of the elastic wave equation
 * propagator in a VTI medium. It is templated from the isotropic elastic
 * propagator (elastic.c): the staggered grid, velocity update, PML,
 * improved-vacuum free surface, source/receiver handling, storage, and the
 * whole time-stepping/adjoint skeleton are UNCHANGED. Only the constitutive
 * (stress-strain) relation and the model set differ. It is compiled multiple
 * times with different options to generate a set of functions that can be
 * called from Python. The options are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 2-3
 *    (there is no 1-D VTI form -- with a single spatial dimension only c33
 *    survives, i.e. the isotropic 1-D elastic equation).
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * The propagator solves the elastic wave equation using a velocity-stress
 * formulation on a staggered grid. The free surface boundary condition is
 * implemented using the improved vacuum method of Zeng et al. (2012), and the
 * PML implementation is based on the C-PML method of Komatitsch and
 * Martin (2007).
 *
 * CONSTITUTIVE RELATION (VTI, symmetry axis = slowest dimension: z in 3-D,
 * y in 2-D). With the symmetry axis as the Voigt "3" direction and
 * d_i = (PML-corrected) dv_i/dx_i, the forward stress-rate updates are:
 *
 *   3-D (models c11, c12=c11-2*c66, c13, c33, c55 [zy,zx nodes], c66 [yx]):
 *     sigmaxx += dt * (c11 dvxdx + c12 dvydy + c13 dvzdz)
 *     sigmayy += dt * (c12 dvxdx + c11 dvydy + c13 dvzdz)
 *     sigmazz += dt * (c13 dvxdx + c13 dvydy + c33 dvzdz)
 *     sigmaxy += dt * c66_yx (dvxdy + dvydx)
 *     sigmaxz += dt * c55_zx (dvxdz + dvzdx)
 *     sigmayz += dt * c55_zy (dvydz + dvzdy)
 *
 *   2-D P-SV in the (y,x) plane, axis = y (models c11, c13, c33, c55 [yx]):
 *     sigmaxx += dt * (c11 dvxdx + c13 dvydy)
 *     sigmayy += dt * (c13 dvxdx + c33 dvydy)
 *     sigmaxy += dt * c55_yx (dvxdy + dvydx)
 *
 * The isotropic elastic propagator is recovered by c11=c33=lamb+2*mu,
 * c13=lamb, c55=c66=mu (then c12=c11-2*c66=lamb).
 *
 * ADJOINT / GRADIENTS. The stress update is a linear map with a symmetric
 * stiffness matrix, so its transpose reuses the same coefficients. In the
 * backward pass the adjoint velocity picks up (with s? = adjoint stresses):
 *     to dvxdx:  c11 sxx + c12 syy + c13 szz   (VX_X)
 *     to dvydy:  c12 sxx + c11 syy + c13 szz   (VY_Y)
 *     to dvzdz:  c13 sxx + c13 syy + c33 szz   (VZ_Z)
 * (2-D: VX_X = c11 sxx + c13 syy; VY_Y = c13 sxx + c33 syy), and each shear
 * adjoint uses its own c55/c66. Because sigma_new = sigma_old + dt * C(m):Dv,
 * the model gradients are the time-summed cross-correlations of the adjoint
 * stresses with the stored forward velocity derivatives S_a = dt*(Dv)_a:
 *     3-D: grad_c11 = sum (sxx S_xx + syy S_yy);  grad_c33 = sum szz S_zz;
 *          grad_c13 = sum (szz (S_xx+S_yy) + (sxx+syy) S_zz);
 *          grad_c12 = sum (sxx S_yy + syy S_xx);
 *          grad_c66_yx = sum sxy S_xy;  grad_c55_zx = sum sxz S_xz;
 *          grad_c55_zy = sum syz S_yz.
 *     2-D: grad_c11 = sum sxx S_xx;  grad_c33 = sum syy S_yy;
 *          grad_c13 = sum (sxx S_yy + syy S_xx);  grad_c55_yx = sum sxy S_xy.
 * (c12 is derived as c11-2*c66 in Python, so autograd folds grad_c12 back
 * into c11 and c66 there.) The velocity update, PML, staggering, and the
 * IVM free-surface averaging are identical to the isotropic propagator.
 *
 * The code is structured to maximise performance by using macros to
 * generate the code for each of the regions of the computational
 * domain (a central region, side regions, edge regions, and corner regions),
 * and by using OpenMP to parallelise the loops over shots.
 *
 * The staggered grid locations of the different wavefield components depends
 * on the number of dimensions. In 3D, the velocity components (vx, vy, vz)
 * are stored at half-integer locations in their respective dimension and
 * integer locations in the other dimensions. The normal stress components
 * (sigmaxx, sigmayy, sigmazz) are stored at integer locations. The shear
 * stress components (sigmaxy, sigmaxz, sigmayz) are stored at half-integer
 * locations in their two respective dimensions and integer locations in the
 * other. The normal-node stiffnesses (c11, c12, c13, c33) are at the same
 * locations as the normal stresses; each shear stiffness (c55/c66) is
 * averaged onto its shear-stress node exactly as mu is. The other dimensions
 * have analogous staggered grid locations.
 *
 * o--------->x
 * | SII=VX==|=SII=VX==|=SII vx
 * | VY  SXY | VY  SXY | VY  sxy
 * v [-------------------]------
 * y SII VX  | SII VX  | SII vx
 *   VY  SXY | VY  SXY | VY  sxy
 *   [-------------------]------
 *   SII=VX==|=SII=VX==|=SII vx
 *   vy  sxy | vy  sxy | vy  sxy
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "common_cpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  elastic_vti_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
/* Multiaxial-PML (M-PML) build variant. When DW_MPML is defined the exported
 * forward/backward symbols gain an ``_mpml`` suffix (e.g.
 * elastic_iso_2d_4_float_forward_mpml_cpu) and the PML profiles are read as
 * full-grid arrays (see the P_* accessors below); otherwise the default
 * per-axis 1-D profile behaviour is compiled unchanged. */
#ifdef DW_MPML
#define ELASTIC_VARIANT_I(name) name##_mpml
#define ELASTIC_VARIANT(name) ELASTIC_VARIANT_I(name)
#else
#define ELASTIC_VARIANT(name) name
#endif
#define FUNC(name) \
  CAT(ELASTIC_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* PML-region skip test. The default kernel skips a direction's PML update
 * outside that direction's PML region (where its 1-D profile is zero). Under
 * M-PML each direction's damping is nonzero wherever ANY axis is inside its
 * PML (the cross-axis term), so a direction's PML update may be skipped only
 * in the pure interior (interior along EVERY axis). Inside the strips the
 * full-grid M-PML profiles vanish identically wherever there is in fact no
 * damping (a = b = 0), so applying the PML path throughout the strip union is
 * exact; the pure interior keeps the fast non-PML path. The region flags
 * pml_z/pml_y/pml_x are the enclosing region-loop indices (1 = interior along
 * that axis), constant per region, so the compiler still specialises each
 * region's body. */
#ifdef DW_MPML
#if DW_NDIM == 3
#define DW_PML_SKIP(cond) (pml_z == 1 && pml_y == 1 && pml_x == 1)
#elif DW_NDIM == 2
#define DW_PML_SKIP(cond) (pml_y == 1 && pml_x == 1)
#else
#define DW_PML_SKIP(cond) (pml_x == 1)
#endif
#else
#define DW_PML_SKIP(cond) (cond)
#endif

/* ND_INDEX, the P_A(*)/P_B(*) PML-profile accessors, the VZ/VY/VX/SIGMA(*)/
 * C11/C12/C13/C33/C_XY/C55_ZX/C55_ZY/BUOYANCY_(*)/M_SIGMA(**)/M_V(**)
 * field-access macros, the VZ_Z/VY_Y/VX_X(+_PML)/SIGMAZZ_Z/.../SIGMAYZ_Y
 * (+_PML) adjoint-term macros, and the FUNC(forward_vz_cell)/.../
 * FUNC(backward_sigmayz_cell) per-grid-point stencil update functions (14
 * total: one per velocity axis/combined-normal-or-shear-stress component,
 * forward and backward) live in elastic_vti_kernels.h, shared verbatim with
 * elastic_vti.cu (see that header for the full explanation). *_shot/m_*_t/
 * m_*_shot below are this file's names for the (per-shot-offset)
 * arrays/values passed as the corresponding parameters of those functions;
 * the DW_PML_SKIP macro above (driver-side PML-region-skip scaffolding, not
 * per-grid-point math) is used to compute the plain `pml_z`/`pml_y`/`pml_x`
 * bools those functions expect. */
#include "elastic_vti_kernels.h"

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

/* --- boundary=["free"/"rigid"/"periodic",...] guard-ring image fill ------
 *
 * Ports the flat-face free/rigid/periodic guard-ring image method to
 * elastic_vti's stress tensor + velocity field set. This is a verbatim copy
 * of elastic.c's own FUNC(fill_p_bc_line)/FUNC(fill_elastic_onrow_velocity_
 * line)/FUNC(fill_elastic_offrow_velocity_line)/FUNC(fill_elastic_shear_
 * line)/FUNC(fill_elastic_stress_bc)/FUNC(fill_elastic_velocity_bc) --
 * not a re-derivation. The physics citations
 * (ROBERTSSON_FS_SUMMARY.md Sec 1.1-1.3) establish which field is odd/even/
 * zero-filled on which face; that assignment depends only on which fields
 * are stresses vs. velocities and which axis a field differentiates across
 * (both geometric/topological facts, unchanged between isotropic and VTI
 * media -- see docs/boundaries.rst's Policy section and
 * _boundaries.py's fill_elastic_stress_boundary/fill_elastic_velocity_
 * boundary module comment for the full derivation), NOT on the stiffness
 * *values* used to compute the stress-rate updates from these fields (VTI
 * replaces the isotropic lamb/mu coupling with the c11/c12/c13/c33/c55/c66
 * coupling documented in this file's own header, but every normal stress is
 * still driven only by axis-divergence terms and every shear stress only by
 * its own two cross-derivative terms, exactly as in isotropic elastic).
 * Per axis A, exactly six wavefields are read across that face: sigma_AA
 * (on-row), the two shear stresses sharing axis A (off-row), and every
 * velocity component; the other three fields need no guard fill on face A.
 * This file is a separate translation unit from elastic.c, so its ``static
 * inline`` definitions are not visible here -- mirrors elastic_vti's own
 * general templating comment at the top of this file ("only the
 * constitutive relation and the model set differ"): the guard-fill index
 * arithmetic is part of the UNCHANGED staggered-grid/PML/free-surface
 * skeleton, so it is copied, not rederived. Unlike elastic.c, no DW_NDIM==1
 * branch is needed: elastic_vti has no 1-D form (CMakeLists.txt only builds
 * elastic_vti for DW_NDIM 2 and 3 -- see the module docstring above).
 *
 * These four functions (forward FUNC(fill_elastic_stress_bc)/FUNC(fill_
 * elastic_velocity_bc) and their FUNC(*_adjoint) transposes) stay local
 * here rather than sharing elastic_bc.h's copies (used by elastic.c/
 * elastic_born.c): they are functionally identical but not byte-identical,
 * because elastic_bc.h's copies unconditionally cast away both nz_ and ny_
 * (needed only when elastic.c/elastic_born.c build at their DW_NDIM==1,
 * where neither is used) while these versions cast away only nz_ (ny_ is
 * always genuinely used here, since elastic_vti never builds at
 * DW_NDIM==1); sharing the header would add a harmless but real extra
 * statement, which the preprocessed-token-identity gate (this is provable,
 * mechanical code motion, not a rewrite) correctly rejects.
 * clone_manifest.json polices this pairing against elastic_bc.h.
 */

/* sigma_AA: on-row, lattice-coincident with the face -- identical to
 * acoustic's own pressure treatment (free: odd + on-surface node held to
 * zero; rigid: even about the half-cell wall; periodic: wrap). Lives in
 * line_bc.h, shared verbatim across every propagator on this staggered
 * layout (see that header). */
#include "line_bc.h"

/* FUNC(fill_elastic_onrow_velocity_line)/FUNC(fill_elastic_offrow_velocity_
 * line)/FUNC(fill_elastic_shear_line) live in elastic_line_bc.h (see that
 * header). */
#include "elastic_line_bc.h"

/* Apply the stress guard fill on every reflecting axis: sigma_AA (reusing
 * FUNC(fill_p_bc_line) verbatim) plus the two shear stresses sharing axis A
 * (FUNC(fill_elastic_shear_line)), for each axis with a non-PML face.
 * elastic_vti has no DW_NDIM==1 build (see the module docstring), so unlike
 * elastic.c's own version of this function there is no #else fallback. */
static void FUNC(fill_elastic_stress_bc)(
#if DW_NDIM == 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
    DW_DTYPE* __restrict const sigmaxx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_p_bc_line)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
          FUNC(fill_elastic_shear_line)(sigmaxz, base, nx_, 1, bc[4], bc[5]);
          FUNC(fill_elastic_shear_line)(sigmaxy, base, nx_, 1, bc[4], bc[5]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_p_bc_line)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
          FUNC(fill_elastic_shear_line)(sigmayz, base, ny_, nx_, bc[2], bc[3]);
          FUNC(fill_elastic_shear_line)(sigmaxy, base, ny_, nx_, bc[2], bc[3]);
        }
    if (bc[0] != 0 || bc[1] != 0) /* z-face */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_p_bc_line)(sigmazz, base, nz_, ny_ * nx_, bc[0], bc[1]);
          FUNC(fill_elastic_shear_line)(sigmayz, base, nz_, ny_ * nx_, bc[0],
                                        bc[1]);
          FUNC(fill_elastic_shear_line)(sigmaxz, base, nz_, ny_ * nx_, bc[0],
                                        bc[1]);
        }
  }
#else /* DW_NDIM == 2 */
  {
    int64_t yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_p_bc_line)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
        FUNC(fill_elastic_shear_line)(sigmaxy, base, nx_, 1, bc[4], bc[5]);
      }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_p_bc_line)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
        FUNC(fill_elastic_shear_line)(sigmaxy, base, ny_, nx_, bc[2], bc[3]);
      }
  }
#endif
}

/* Apply the velocity guard fill on every reflecting axis: unlike acoustic
 * (where only the boundary-normal velocity is imaged), *every* velocity
 * component is touched on each reflecting face here -- each normal
 * stress's own divergence term reads every velocity's ghost region along
 * that axis, and (for free) Robertsson's point 3 zeroes "particle
 * velocities" generically, not just the normal component. */
static void FUNC(fill_elastic_velocity_bc)(
#if DW_NDIM == 3
    DW_DTYPE* __restrict const vz,
#endif
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vx,
    int64_t const* __restrict const bc, int64_t const nz_, int64_t const ny_,
    int64_t const nx_) {
  (void)nz_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face: vx off-row, vz/vy on-row */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_elastic_offrow_velocity_line)(vx, base, nx_, 1, bc[4],
                                                  bc[5]);
          FUNC(fill_elastic_onrow_velocity_line)(vz, base, nx_, 1, bc[4],
                                                 bc[5]);
          FUNC(fill_elastic_onrow_velocity_line)(vy, base, nx_, 1, bc[4],
                                                 bc[5]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face: vy off-row, vz/vx on-row */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line)(vy, base, ny_, nx_, bc[2],
                                                  bc[3]);
          FUNC(fill_elastic_onrow_velocity_line)(vz, base, ny_, nx_, bc[2],
                                                 bc[3]);
          FUNC(fill_elastic_onrow_velocity_line)(vx, base, ny_, nx_, bc[2],
                                                 bc[3]);
        }
    if (bc[0] != 0 || bc[1] != 0) /* z-face: vz off-row, vy/vx on-row */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line)(vz, base, nz_, ny_ * nx_,
                                                  bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line)(vy, base, nz_, ny_ * nx_,
                                                 bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line)(vx, base, nz_, ny_ * nx_,
                                                 bc[0], bc[1]);
        }
  }
#else /* DW_NDIM == 2 */
  {
    int64_t yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face: vx off-row, vy on-row */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_elastic_offrow_velocity_line)(vx, base, nx_, 1, bc[4], bc[5]);
        FUNC(fill_elastic_onrow_velocity_line)(vy, base, nx_, 1, bc[4], bc[5]);
      }
    if (bc[2] != 0 || bc[3] != 0) /* y-face: vy off-row, vx on-row */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_elastic_offrow_velocity_line)(vy, base, ny_, nx_, bc[2],
                                                bc[3]);
        FUNC(fill_elastic_onrow_velocity_line)(vx, base, ny_, nx_, bc[2],
                                               bc[3]);
      }
  }
#endif
}

/* --- Adjoint (transpose) of the guard-ring image fill above --------------
 *
 * The compiled backward re-applies these _adjoint fills to the adjoint
 * (cotangent) stress/velocity fields in place of the forward fills above.
 * The forward fill is a linear map (a projection that overwrites some rows
 * with signed images of others, zeroes some rows outright, and leaves
 * interior rows untouched); unlike acoustic's single-field pressure fill
 * (genuinely self-adjoint -- reapplying the same fill to the adjoint field
 * is its own correct transpose), elastic_vti's multi-field fill mixes
 * distinct per-field parity/zero/slave rules across the SAME guard ring
 * (node-centred odd images with an on-surface zero for sigma_AA, half-cell
 * odd/even images with no surface node for sigma_AB/AC, ghost-zeroing plus
 * rigid on-node clamps for velocities, periodic wraps with an extra
 * high-side slaved node for every off-row/shear field), so reapplying the
 * forward fill verbatim is NOT its own transpose here. This is a verbatim
 * port of elastic.c's own post-B5-fix FUNC(fill_p_bc_line_adjoint)/FUNC(
 * fill_elastic_onrow_velocity_line_adjoint)/FUNC(fill_elastic_offrow_
 * velocity_line_adjoint)/FUNC(fill_elastic_shear_line_adjoint)/FUNC(fill_
 * elastic_stress_bc_adjoint)/FUNC(fill_elastic_velocity_bc_adjoint) -- see
 * this file's module comment above FUNC(fill_p_bc_line) for why the
 * guard-fill index arithmetic (unchanged between isotropic and VTI media)
 * is copied rather than rederived; the same holds for its transpose.
 *
 * Mechanical derivation (matches the autograd semantics of
 * _boundaries.py's fill_elastic_stress_boundary/fill_elastic_velocity_
 * boundary): every forward assignment `dest = sign * src` has adjoint
 * `adj[src] += sign * adj[dest]`; every forward assignment `dest = 0` (a
 * hard zero -- no source) has no accumulation step. After all
 * accumulations for one line's fill, every cell the forward fill wrote
 * (ghost cells, on-plane zeros, slaved nodes) is zeroed -- those cells are
 * not independent degrees of freedom, so no adjoint value may remain on
 * them (the ghost <-> interior image map is injective -- see
 * docs/boundaries.rst's mirror-plane conventions -- so this scatter-add
 * never has two ghost cells racing to accumulate into the same interior
 * cell within one line's fill).
 *
 * Two ordering requirements fall out of doing this faithfully (both
 * mirror how PyTorch's autograd would walk the forward graph backward
 * through the same sequence of tensor writes):
 *   - Within one line's fill, a later forward assignment may read a cell
 *     an earlier one wrote (the periodic high-side slaved node, written
 *     before the ghost wraps -- see fill_elastic_offrow_velocity_line's
 *     and fill_elastic_shear_line's own `bc_hi == 3` slave step); the
 *     adjoint below undoes assignments in the reverse of the forward
 *     order (high-face ghost wrap, then low-face ghost wrap, then the
 *     slaved-node/on-plane-zero step) so any such read is folded before
 *     its source is touched again.
 *   - Each stress/shear or velocity component is refilled once per
 *     reflecting axis it touches (e.g. sigma_xy on both an x-face and a
 *     y-face); the later face's forward fill overwrites the earlier
 *     face's corner ghost cells, so only the later face's fill is part of
 *     the true map at that corner. FUNC(fill_elastic_stress_bc_adjoint)/
 *     FUNC(fill_elastic_velocity_bc_adjoint) below process faces in the
 *     exact reverse of FUNC(fill_elastic_stress_bc)/FUNC(fill_elastic_
 *     velocity_bc)'s own face order, so the later (in forward time) face's
 *     adjoint runs first and zeros the corner; the earlier face's adjoint
 *     then finds that cell already zero and correctly contributes nothing
 *     there, without any explicit corner special-casing.
 */

/* FUNC(fill_p_bc_line_adjoint)/FUNC(fill_elastic_onrow_velocity_line_adjoint)/
 * FUNC(fill_elastic_offrow_velocity_line_adjoint)/FUNC(fill_elastic_shear_
 * line_adjoint) live in elastic_line_bc.h, already included above. */

/* Transpose of FUNC(fill_elastic_stress_bc): faces undone in the reverse of
 * that function's own face order (see the module comment above). */
static void FUNC(fill_elastic_stress_bc_adjoint)(
#if DW_NDIM == 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
    DW_DTYPE* __restrict const sigmaxx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[0] != 0 || bc[1] != 0) /* z-face */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_p_bc_line_adjoint)(sigmazz, base, nz_, ny_ * nx_, bc[0],
                                       bc[1]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmayz, base, nz_, ny_ * nx_,
                                                bc[0], bc[1]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxz, base, nz_, ny_ * nx_,
                                                bc[0], bc[1]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_p_bc_line_adjoint)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmayz, base, ny_, nx_, bc[2],
                                                bc[3]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, ny_, nx_, bc[2],
                                                bc[3]);
        }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_p_bc_line_adjoint)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxz, base, nx_, 1, bc[4],
                                                bc[5]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, nx_, 1, bc[4],
                                                bc[5]);
        }
  }
#else /* DW_NDIM == 2 */
  {
    int64_t yy, xx;
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_p_bc_line_adjoint)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
        FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, ny_, nx_, bc[2],
                                              bc[3]);
      }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_p_bc_line_adjoint)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
        FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, nx_, 1, bc[4],
                                              bc[5]);
      }
  }
#endif
}

/* Transpose of FUNC(fill_elastic_velocity_bc): faces undone in the reverse
 * of that function's own face order (see the module comment above). */
static void FUNC(fill_elastic_velocity_bc_adjoint)(
#if DW_NDIM == 3
    DW_DTYPE* __restrict const vz,
#endif
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vx,
    int64_t const* __restrict const bc, int64_t const nz_, int64_t const ny_,
    int64_t const nx_) {
  (void)nz_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[0] != 0 || bc[1] != 0) /* z-face */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line_adjoint)(
              vz, base, nz_, ny_ * nx_, bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(
              vy, base, nz_, ny_ * nx_, bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(
              vx, base, nz_, ny_ * nx_, bc[0], bc[1]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line_adjoint)(vy, base, ny_, nx_,
                                                          bc[2], bc[3]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vz, base, ny_, nx_,
                                                         bc[2], bc[3]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vx, base, ny_, nx_,
                                                         bc[2], bc[3]);
        }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_elastic_offrow_velocity_line_adjoint)(vx, base, nx_, 1,
                                                          bc[4], bc[5]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vz, base, nx_, 1,
                                                         bc[4], bc[5]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vy, base, nx_, 1,
                                                         bc[4], bc[5]);
        }
  }
#else /* DW_NDIM == 2 */
  {
    int64_t yy, xx;
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_elastic_offrow_velocity_line_adjoint)(vy, base, ny_, nx_,
                                                        bc[2], bc[3]);
        FUNC(fill_elastic_onrow_velocity_line_adjoint)(vx, base, ny_, nx_,
                                                       bc[2], bc[3]);
      }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_elastic_offrow_velocity_line_adjoint)(vx, base, nx_, 1, bc[4],
                                                        bc[5]);
        FUNC(fill_elastic_onrow_velocity_line_adjoint)(vy, base, nx_, 1, bc[4],
                                                       bc[5]);
      }
  }
#endif
}

/* add_pressure/record_pressure only: byte-identical across elastic.c/
 * elastic_born.c/elastic_vti.c, so shared via elastic_bc.h (see that
 * header for their derivation); the four guard-ring appliers above stay
 * local (see the module comment above), hence the opt-out macro. */
#define DW_ELASTIC_BC_NO_FILLS
#include "elastic_bc.h"
#undef DW_ELASTIC_BC_NO_FILLS

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(forward)(
    DW_DTYPE const* __restrict const c11,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12,
#endif
    DW_DTYPE const* __restrict const c13, DW_DTYPE const* __restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c55_zy,
    DW_DTYPE const* __restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    /* Shear stiffness at the sigma_xy node: c66_yx in 3-D, c55_yx in 2-D. */
    DW_DTYPE const* __restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const f_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const f_y,
#endif
    DW_DTYPE const* __restrict const f_x, DW_DTYPE const* __restrict const f_p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmaxz,
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const m_sigmayzz,
    DW_DTYPE* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vyy,
    DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmaxyy,
    DW_DTYPE* __restrict const m_sigmaxyx,
#endif
    DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
    DW_DTYPE* __restrict const dvzdbuoyancy_store_1b,
    void* __restrict const dvzdbuoyancy_store_2,
    void* __restrict const dvzdbuoyancy_store_3,
    char const* __restrict const* __restrict const dvzdbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvzdz_store_1,
    DW_DTYPE* __restrict const dvzdz_store_1b,
    void* __restrict const dvzdz_store_2, void* __restrict const dvzdz_store_3,
    char const* __restrict const* __restrict const dvzdz_filenames_ptr,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1b,
    void* __restrict const dvzdx_plus_dvxdz_store_2,
    void* __restrict const dvzdx_plus_dvxdz_store_3,
    char const* __restrict const* __restrict const dvzdx_plus_dvxdz_filenames_ptr,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1b,
    void* __restrict const dvzdy_plus_dvydz_store_2,
    void* __restrict const dvzdy_plus_dvydz_store_3,
    char const* __restrict const* __restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const dvydbuoyancy_store_1,
    DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
    void* __restrict const dvydbuoyancy_store_2,
    void* __restrict const dvydbuoyancy_store_3,
    char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvydy_store_1,
    DW_DTYPE* __restrict const dvydy_store_1b,
    void* __restrict const dvydy_store_2, void* __restrict const dvydy_store_3,
    char const* __restrict const* __restrict const dvydy_filenames_ptr,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1b,
    void* __restrict const dvydx_plus_dvxdy_store_2,
    void* __restrict const dvydx_plus_dvxdy_store_3,
    char const* __restrict const* __restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
    DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
    DW_DTYPE* __restrict const dvxdbuoyancy_store_1b,
    void* __restrict const dvxdbuoyancy_store_2,
    void* __restrict const dvxdbuoyancy_store_3,
    char const* __restrict const* __restrict const dvxdbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvxdx_store_1,
    DW_DTYPE* __restrict const dvxdx_store_1b,
    void* __restrict const dvxdx_store_2, void* __restrict const dvxdx_store_3,
    char const* __restrict const* __restrict const dvxdx_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const r_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const r_y,
#endif
    DW_DTYPE* __restrict const r_x, DW_DTYPE* __restrict const r_p,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_y_i,
#endif
    int64_t const* __restrict const sources_x_i,
    int64_t const* __restrict const sources_p_i,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_y_i,
#endif
    int64_t const* __restrict const receivers_x_i,
    int64_t const* __restrict const receivers_p_i,
#if DW_NDIM >= 3
    DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy,
#endif
    DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
    int64_t const n_shots,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const n_sources_z_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_y_per_shot,
#endif
    int64_t const n_sources_x_per_shot, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_z_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_y_per_shot,
#endif
    int64_t n_receivers_x_per_shot, int64_t const n_receivers_p_per_shot,
    int64_t const step_ratio, int64_t const storage_mode,
    size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
    bool const c11_requires_grad,
#if DW_NDIM >= 3
    bool const c12_requires_grad,
#endif
    bool const c13_requires_grad, bool const c33_requires_grad,
    bool const c55_requires_grad,
#if DW_NDIM >= 3
    bool const c66_requires_grad,
#endif
    bool const buoyancy_requires_grad, bool const c11_batched,
#if DW_NDIM >= 3
    bool const c12_batched,
#endif
    bool const c13_batched, bool const c33_batched, bool const c55_batched,
#if DW_NDIM >= 3
    bool const c66_batched,
#endif
    bool const buoyancy_batched, bool const storage_compression,
    int64_t start_t,
#if DW_NDIM >= 3
    int64_t const pml_z0,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y0,
#endif
    int64_t const pml_x0,
#if DW_NDIM >= 3
    int64_t const pml_z1,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y1,
#endif
    int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0,
#endif
    int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y1,
#endif
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;

#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0) && sources_z_i == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot > 0) && sources_y_i == NULL) return 1;
#endif
    if ((n_sources_x_per_shot > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_p_per_shot > 0) && sources_p_i == NULL) return 1;
  }
#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
#if DW_NDIM >= 2
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
#endif
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 2 */
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#endif
  int64_t shot;
  /* Intra-shot spatial parallelism: shots run serially and each time step's
   * velocity and stress stencils are spread across threads (see the omp
   * parallel regions below), so a single shot saturates all cores. */
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
    int64_t const n_grid_points = ny * nx;
#else
    int64_t const n_grid_points = nx;
#endif
    int64_t const si = shot * n_grid_points;
#if DW_NDIM >= 3
    int64_t const siz = shot * n_sources_z_per_shot;
    int64_t const riz = shot * n_receivers_z_per_shot;
    int64_t const* __restrict const sources_z_i_shot =
        sources_z_i != NULL ? (sources_z_i + siz) : NULL;
    int64_t const* __restrict const receivers_z_i_shot =
        receivers_z_i != NULL ? (receivers_z_i + riz) : NULL;
    DW_DTYPE const* __restrict const f_z_shot =
        f_z != NULL ? (f_z + siz) : NULL;
    DW_DTYPE* __restrict const r_z_shot = r_z != NULL ? (r_z + riz) : NULL;
    DW_DTYPE const* __restrict const c55_zy_shot =
        c55_batched ? c55_zy + si : c55_zy;
    DW_DTYPE const* __restrict const c55_zx_shot =
        c55_batched ? c55_zx + si : c55_zx;
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        buoyancy_batched ? buoyancy_z + si : buoyancy_z;
    DW_DTYPE* __restrict const vz_shot = vz + si;
    DW_DTYPE* __restrict const sigmazz_shot = sigmazz + si;
    DW_DTYPE* __restrict const sigmayz_shot = sigmayz + si;
    DW_DTYPE* __restrict const sigmaxz_shot = sigmaxz + si;
    DW_DTYPE* __restrict const m_vzz_shot = m_vzz + si;
    DW_DTYPE* __restrict const m_vzy_shot = m_vzy + si;
    DW_DTYPE* __restrict const m_vzx_shot = m_vzx + si;
    DW_DTYPE* __restrict const m_vyz_shot = m_vyz + si;
    DW_DTYPE* __restrict const m_vxz_shot = m_vxz + si;
    DW_DTYPE* __restrict const m_sigmazzz_shot = m_sigmazzz + si;
    DW_DTYPE* __restrict const m_sigmayzy_shot = m_sigmayzy + si;
    DW_DTYPE* __restrict const m_sigmaxzx_shot = m_sigmaxzx + si;
    DW_DTYPE* __restrict const m_sigmayzz_shot = m_sigmayzz + si;
    DW_DTYPE* __restrict const m_sigmaxzz_shot = m_sigmaxzz + si;
#endif
#if DW_NDIM >= 2
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const* __restrict const sources_y_i_shot =
        sources_y_i != NULL ? (sources_y_i + siy) : NULL;
    int64_t const* __restrict const receivers_y_i_shot =
        receivers_y_i != NULL ? (receivers_y_i + riy) : NULL;
    DW_DTYPE const* __restrict const f_y_shot =
        f_y != NULL ? (f_y + siy) : NULL;
    DW_DTYPE* __restrict const r_y_shot = r_y != NULL ? (r_y + riy) : NULL;
    DW_DTYPE const* __restrict const c_xy_shot =
#if DW_NDIM >= 3
        c66_batched ? c_xy + si : c_xy;
#else
        c55_batched ? c_xy + si : c_xy;
#endif
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE* __restrict const vy_shot = vy + si;
    DW_DTYPE* __restrict const sigmayy_shot = sigmayy + si;
    DW_DTYPE* __restrict const sigmaxy_shot = sigmaxy + si;
    DW_DTYPE* __restrict const m_vyy_shot = m_vyy + si;
    DW_DTYPE* __restrict const m_vyx_shot = m_vyx + si;
    DW_DTYPE* __restrict const m_vxy_shot = m_vxy + si;
    DW_DTYPE* __restrict const m_sigmayyy_shot = m_sigmayyy + si;
    DW_DTYPE* __restrict const m_sigmaxyy_shot = m_sigmaxyy + si;
    DW_DTYPE* __restrict const m_sigmaxyx_shot = m_sigmaxyx + si;
#endif
    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const sip = shot * n_sources_p_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const rip = shot * n_receivers_p_per_shot;
    int64_t const* __restrict const sources_x_i_shot =
        sources_x_i != NULL ? (sources_x_i + six) : NULL;
    int64_t const* __restrict const sources_p_i_shot =
        sources_p_i != NULL ? (sources_p_i + sip) : NULL;
    int64_t const* __restrict const receivers_x_i_shot =
        receivers_x_i != NULL ? (receivers_x_i + rix) : NULL;
    int64_t const* __restrict const receivers_p_i_shot =
        receivers_p_i != NULL ? (receivers_p_i + rip) : NULL;
    DW_DTYPE const* __restrict const f_x_shot =
        f_x != NULL ? (f_x + six) : NULL;
    DW_DTYPE const* __restrict const f_p_shot =
        f_p != NULL ? (f_p + sip) : NULL;
    DW_DTYPE* __restrict const r_x_shot = r_x != NULL ? (r_x + rix) : NULL;
    DW_DTYPE* __restrict const r_p_shot = r_p != NULL ? (r_p + rip) : NULL;
    DW_DTYPE const* __restrict const c11_shot = c11_batched ? c11 + si : c11;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12_shot = c12_batched ? c12 + si : c12;
#endif
    DW_DTYPE const* __restrict const c13_shot = c13_batched ? c13 + si : c13;
    DW_DTYPE const* __restrict const c33_shot = c33_batched ? c33 + si : c33;
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE* __restrict const vx_shot = vx + si;
    DW_DTYPE* __restrict const sigmaxx_shot = sigmaxx + si;
    DW_DTYPE* __restrict const m_vxx_shot = m_vxx + si;
    DW_DTYPE* __restrict const m_sigmaxxx_shot = m_sigmaxxx + si;
    int64_t t;

    // Aggregate requires-grad conditions gating the stored velocity
    // derivatives (must match the store-array sizing in Python's
    // storage_flags): any normal-node stiffness drives the dvidi stores;
    // the sigma_xy-node shear (c66 in 3-D, c55 in 2-D) drives its store.
#if DW_NDIM >= 3
    bool const cnormal_requires_grad = c11_requires_grad || c12_requires_grad ||
                                       c13_requires_grad || c33_requires_grad;
    bool const c_xy_requires_grad = c66_requires_grad;
#else
    bool const cnormal_requires_grad =
        c11_requires_grad || c13_requires_grad || c33_requires_grad;
    bool const c_xy_requires_grad = c55_requires_grad;
#endif
    int storage_err = 0;
#if DW_NDIM >= 3
    FILE* fp_dvzdbuoyancy = NULL;
    FILE* fp_dvzdz = NULL;
    FILE* fp_dvzdx_plus_dvxdz = NULL;
    FILE* fp_dvzdy_plus_dvydz = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_dvydbuoyancy = NULL;
    FILE* fp_dvydy = NULL;
    FILE* fp_dvydx_plus_dvxdy = NULL;
#endif
    FILE* fp_dvxdbuoyancy = NULL;
    FILE* fp_dvxdx = NULL;

#if DW_NDIM >= 3
    OPEN_FILE_WRITE(dvzdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvzdz, cnormal_requires_grad)
    OPEN_FILE_WRITE(dvzdx_plus_dvxdz, c55_requires_grad)
    OPEN_FILE_WRITE(dvzdy_plus_dvydz, c55_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_WRITE(dvydbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvydy, cnormal_requires_grad)
    OPEN_FILE_WRITE(dvydx_plus_dvxdy, c_xy_requires_grad)
#endif
    OPEN_FILE_WRITE(dvxdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvxdx, cnormal_requires_grad)

    for (t = start_t; t < start_t + nt; ++t) {
#if DW_NDIM >= 3
      if ((t % step_ratio) == 0 && (buoyancy_requires_grad) &&
          (dvzdbuoyancy_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
#endif
#if DW_NDIM >= 2
      if ((t % step_ratio) == 0 && (buoyancy_requires_grad) &&
          (dvydbuoyancy_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
#endif
      if ((t % step_ratio) == 0 && (buoyancy_requires_grad) &&
          (dvxdbuoyancy_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      int64_t const store_1_t =
          (shot + ((storage_mode == STORAGE_DEVICE && !storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
          n_grid_points;
      int64_t const store_2_t =
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
          (int64_t)shot_bytes_comp;

#if DW_NDIM >= 3
      DW_DTYPE* __restrict const dvzdbuoyancy_store_1_t =
          dvzdbuoyancy_store_1 != NULL ? (dvzdbuoyancy_store_1 + store_1_t)
                                       : NULL;
      void* __restrict const dvzdbuoyancy_store_2_t =
          dvzdbuoyancy_store_2 != NULL
              ? ((uint8_t*)dvzdbuoyancy_store_2 + store_2_t)
              : NULL;
      DW_DTYPE* __restrict const dvzdz_store_1_t =
          dvzdz_store_1 != NULL ? (dvzdz_store_1 + store_1_t) : NULL;
      void* __restrict const dvzdz_store_2_t =
          dvzdz_store_2 != NULL ? ((uint8_t*)dvzdz_store_2 + store_2_t) : NULL;
      DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1_t =
          dvzdx_plus_dvxdz_store_1 != NULL
              ? (dvzdx_plus_dvxdz_store_1 + store_1_t)
              : NULL;
      void* __restrict const dvzdx_plus_dvxdz_store_2_t =
          dvzdx_plus_dvxdz_store_2 != NULL
              ? ((uint8_t*)dvzdx_plus_dvxdz_store_2 + store_2_t)
              : NULL;
      DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1_t =
          dvzdy_plus_dvydz_store_1 != NULL
              ? (dvzdy_plus_dvydz_store_1 + store_1_t)
              : NULL;
      void* __restrict const dvzdy_plus_dvydz_store_2_t =
          dvzdy_plus_dvydz_store_2 != NULL
              ? ((uint8_t*)dvzdy_plus_dvydz_store_2 + store_2_t)
              : NULL;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const dvydbuoyancy_store_1_t =
          dvydbuoyancy_store_1 != NULL ? (dvydbuoyancy_store_1 + store_1_t)
                                       : NULL;
      void* __restrict const dvydbuoyancy_store_2_t =
          dvydbuoyancy_store_2 != NULL
              ? ((uint8_t*)dvydbuoyancy_store_2 + store_2_t)
              : NULL;
      DW_DTYPE* __restrict const dvydy_store_1_t =
          dvydy_store_1 != NULL ? (dvydy_store_1 + store_1_t) : NULL;
      void* __restrict const dvydy_store_2_t =
          dvydy_store_2 != NULL ? ((uint8_t*)dvydy_store_2 + store_2_t) : NULL;
      DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1_t =
          dvydx_plus_dvxdy_store_1 != NULL
              ? (dvydx_plus_dvxdy_store_1 + store_1_t)
              : NULL;
      void* __restrict const dvydx_plus_dvxdy_store_2_t =
          dvydx_plus_dvxdy_store_2 != NULL
              ? ((uint8_t*)dvydx_plus_dvxdy_store_2 + store_2_t)
              : NULL;
#endif
      DW_DTYPE* __restrict const dvxdbuoyancy_store_1_t =
          dvxdbuoyancy_store_1 != NULL ? (dvxdbuoyancy_store_1 + store_1_t)
                                       : NULL;
      void* __restrict const dvxdbuoyancy_store_2_t =
          dvxdbuoyancy_store_2 != NULL
              ? ((uint8_t*)dvxdbuoyancy_store_2 + store_2_t)
              : NULL;
      DW_DTYPE* __restrict const dvxdx_store_1_t =
          dvxdx_store_1 != NULL ? (dvxdx_store_1 + store_1_t) : NULL;
      void* __restrict const dvxdx_store_2_t =
          dvxdx_store_2 != NULL ? ((uint8_t*)dvxdx_store_2 + store_2_t) : NULL;
      bool const rg_step_t = ((t % step_ratio) == 0);
      // Any normal-node stiffness (c11/c12/c13/c33) drives the dvxdx/dvydy/
      // dvzdz stores; the shear stiffnesses each drive their own store.
      bool const cnormal_requires_grad_t =
          (c11_requires_grad || c13_requires_grad || c33_requires_grad
#if DW_NDIM >= 3
           || c12_requires_grad
#endif
           ) &&
          rg_step_t;
#if DW_NDIM >= 3
      bool const c55_requires_grad_t = c55_requires_grad && rg_step_t;
      bool const c_xy_requires_grad_t = c66_requires_grad && rg_step_t;
#else
      bool const c_xy_requires_grad_t = c55_requires_grad && rg_step_t;
#endif
      bool const buoyancy_requires_grad_t = buoyancy_requires_grad && rg_step_t;

      /* Refill the stress guard ring (read by the velocity update via
       * sigma_AA/the shared-axis shears' derivatives, and by the pressure
       * receiver record below); on a free face this also holds the
       * on-surface normal-stress node to zero. Mirrors elastic.c's own
       * placement (before receiver recording, before the velocity-update
       * region loops). */
      if (has_bc) {
        FUNC(fill_elastic_stress_bc)(
#if DW_NDIM == 3
            sigmazz_shot, sigmayz_shot, sigmaxz_shot,
#endif
            sigmayy_shot, sigmaxy_shot, sigmaxx_shot, bc,
#if DW_NDIM == 3
            nz,
#else
            0,
#endif
            ny, nx);
      }

#if DW_NDIM >= 3
      if (n_receivers_z_per_shot > 0) {
        record_from_wavefield(vz_shot, receivers_z_i_shot,
                              r_z_shot + t * n_shots * n_receivers_z_per_shot,
                              n_receivers_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_y_per_shot > 0) {
        record_from_wavefield(vy_shot, receivers_y_i_shot,
                              r_y_shot + t * n_shots * n_receivers_y_per_shot,
                              n_receivers_y_per_shot);
      }
#endif
      if (n_receivers_x_per_shot > 0) {
        record_from_wavefield(vx_shot, receivers_x_i_shot,
                              r_x_shot + t * n_shots * n_receivers_x_per_shot,
                              n_receivers_x_per_shot);
      }
      if (n_receivers_p_per_shot > 0) {
        record_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, receivers_p_i_shot,
            r_p_shot + t * n_shots * n_receivers_p_per_shot,
            n_receivers_p_per_shot);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
              DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

#if DW_NDIM >= 3

              // vz

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(forward_vz_cell)(
                        sigmazz_shot, sigmayz_shot, sigmaxz_shot, vz_shot,
                        m_sigmazzz_shot, m_sigmayzy_shot, m_sigmaxzx_shot,
                        dvzdbuoyancy_store_1_t, buoyancy_z_shot[i], azh, bzh,
                        ay, by, ax, bx, dt, buoyancy_requires_grad_t, i, ny, nx,
                        z, y, x, is_pml_z, is_pml_y, is_pml_x, rdz, rdy, rdx);
                  }
#endif

#if DW_NDIM >= 2

              // vy

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
#if DW_NDIM >= 3
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
#endif
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(forward_vy_cell)(
#if DW_NDIM >= 3
                        sigmayz_shot,
#endif
                        sigmayy_shot, sigmaxy_shot, vy_shot,
#if DW_NDIM >= 3
                        m_sigmayzz_shot,
#endif
                        m_sigmayyy_shot, m_sigmaxyx_shot,
                        dvydbuoyancy_store_1_t, buoyancy_y_shot[i],
#if DW_NDIM >= 3
                        az, bz,
#endif
                        ayh, byh, ax, bx, dt, buoyancy_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z, is_pml_z, rdz,
#endif
                        y, x, is_pml_y, is_pml_x, rdy, rdx);
                  }
#endif

              // vx

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
#if DW_NDIM >= 3
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
#endif
#if DW_NDIM >= 2
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
#endif
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(forward_vx_cell)(
#if DW_NDIM >= 3
                        sigmaxz_shot,
#endif
#if DW_NDIM >= 2
                        sigmaxy_shot,
#endif
                        sigmaxx_shot, vx_shot,
#if DW_NDIM >= 3
                        m_sigmaxzz_shot,
#endif
#if DW_NDIM >= 2
                        m_sigmaxyy_shot,
#endif
                        m_sigmaxxx_shot, dvxdbuoyancy_store_1_t,
                        buoyancy_x_shot[i],
#if DW_NDIM >= 3
                        az, bz,
#endif
#if DW_NDIM >= 2
                        ay, by,
#endif
                        axh, bxh, dt, buoyancy_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, is_pml_z, rdz,
#endif
#if DW_NDIM >= 2
                        y, is_pml_y, rdy,
#endif
                        x, is_pml_x, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

#if DW_NDIM >= 3
      if (n_sources_z_per_shot > 0) {
        add_to_wavefield(vz_shot, sources_z_i_shot,
                         f_z_shot + t * n_shots * n_sources_z_per_shot,
                         n_sources_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_y_per_shot > 0) {
        add_to_wavefield(vy_shot, sources_y_i_shot,
                         f_y_shot + t * n_shots * n_sources_y_per_shot,
                         n_sources_y_per_shot);
      }
#endif
      if (n_sources_x_per_shot > 0) {
        add_to_wavefield(vx_shot, sources_x_i_shot,
                         f_x_shot + t * n_shots * n_sources_x_per_shot,
                         n_sources_x_per_shot);
      }

      /* Refill the velocity guard ring (read by every normal stress's own
       * divergence term and by the shared-axis shears' updates); on a free
       * face every component's ghost region is zeroed. Mirrors elastic.c's
       * own placement (after velocity source injection, before the
       * stress-update region loops). */
      if (has_bc) {
        FUNC(fill_elastic_velocity_bc)(
#if DW_NDIM == 3
            vz_shot,
#endif
            vy_shot, vx_shot, bc,
#if DW_NDIM == 3
            nz,
#else
            0,
#endif
            ny, nx);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
              DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

              // sigmaii

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
#if DW_NDIM >= 3
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
#endif
#if DW_NDIM >= 2
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
#endif
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(forward_sigmaii_cell)(
#if DW_NDIM >= 3
                        vz_shot, sigmazz_shot, m_vzz_shot, dvzdz_store_1_t,
#endif
#if DW_NDIM >= 2
                        vy_shot, sigmayy_shot, m_vyy_shot, dvydy_store_1_t,
#endif
                        vx_shot, sigmaxx_shot, m_vxx_shot, dvxdx_store_1_t,
                        c11_shot[i],
#if DW_NDIM >= 3
                        c12_shot[i],
#endif
                        c13_shot[i],
#if DW_NDIM >= 2
                        c33_shot[i],
#endif
                        dt, cnormal_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, is_pml_z, az, bz, rdz,
#endif
#if DW_NDIM >= 2
                        y, is_pml_y, ay, by, rdy,
#endif
                        x, is_pml_x, ax, bx, rdx);
                  }

#if DW_NDIM >= 2

              // sigmaxy

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(forward_sigmaxy_cell)(
                        vx_shot, vy_shot, sigmaxy_shot, m_vxy_shot, m_vyx_shot,
                        dvydx_plus_dvxdy_store_1_t, c_xy_shot[i], dt,
                        c_xy_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny, z,
#endif
                        nx, y, x, is_pml_y, is_pml_x, ayh, byh, axh, bxh, rdy,
                        rdx);
                  }
#endif

#if DW_NDIM >= 3

              // sigmaxz

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(forward_sigmaxz_cell)(
                        vx_shot, vz_shot, sigmaxz_shot, m_vxz_shot, m_vzx_shot,
                        dvzdx_plus_dvxdz_store_1_t, c55_zx_shot[i], dt,
                        c55_requires_grad_t, i, ny, nx, z, y, x, is_pml_z,
                        is_pml_x, azh, bzh, axh, bxh, rdz, rdx);
                  }
#endif

#if DW_NDIM >= 3

              // sigmayz

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    FUNC(forward_sigmayz_cell)(
                        vy_shot, vz_shot, sigmayz_shot, m_vyz_shot, m_vzy_shot,
                        dvzdy_plus_dvydz_store_1_t, c55_zy_shot[i], dt,
                        c55_requires_grad_t, i, ny, nx, z, y, x, is_pml_z,
                        is_pml_y, azh, bzh, ayh, byh, rdz, rdy);
                  }
#endif
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      if (n_sources_p_per_shot > 0) {
        add_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, sources_p_i_shot,
            f_p_shot + t * n_shots * n_sources_p_per_shot,
            n_sources_p_per_shot);
      }

#define SAVE_SNAPSHOT(name, grad_cond)                                         \
  if (grad_cond) {                                                             \
    int64_t const step_idx = t / step_ratio;                                   \
    if (STORAGE_FUNC(save_snapshot_cpu)(                                       \
            name##_store_1_t, name##_store_2_t, fp_##name, storage_mode,       \
            storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp, \
            DIM_ARGS) != 0) {                                                  \
      storage_err = 1;                                                         \
      goto storage_cleanup;                                                    \
    }                                                                          \
  }

#if DW_NDIM >= 3
      SAVE_SNAPSHOT(dvzdbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvzdz, cnormal_requires_grad_t)
      SAVE_SNAPSHOT(dvzdx_plus_dvxdz, c55_requires_grad_t)
      SAVE_SNAPSHOT(dvzdy_plus_dvydz, c55_requires_grad_t)
#endif
#if DW_NDIM >= 2
      SAVE_SNAPSHOT(dvydbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvydy, cnormal_requires_grad_t)
      SAVE_SNAPSHOT(dvydx_plus_dvxdy, c_xy_requires_grad_t)
#endif
      SAVE_SNAPSHOT(dvxdbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvxdx, cnormal_requires_grad_t)
    }

    /* Leave the returned wavefields boundary-consistent (e.g. free-surface
     * stress nodes held to zero, guard rings imaged), matching
     * _elastic_vti_eager_core's own post-loop refill in elastic_vti.py --
     * without this, the last timestep's stress update runs after the last
     * in-loop stress fill, so the returned sigma guard ring/on-surface zero
     * reflect a stale (pre-update) state. Receiver traces and gradients are
     * unaffected (already recorded/consumed inside the loop before this
     * point); only the final returned wavefields change. */
    if (has_bc) {
      FUNC(fill_elastic_stress_bc)(
#if DW_NDIM == 3
          sigmazz_shot, sigmayz_shot, sigmaxz_shot,
#endif
          sigmayy_shot, sigmaxy_shot, sigmaxx_shot, bc,
#if DW_NDIM == 3
          nz,
#else
          0,
#endif
          ny, nx);
      FUNC(fill_elastic_velocity_bc)(
#if DW_NDIM == 3
          vz_shot,
#endif
          vy_shot, vx_shot, bc,
#if DW_NDIM == 3
          nz,
#else
          0,
#endif
          ny, nx);
    }

#define CLOSE_FILE(name) \
  if (dw_close_file(&fp_##name) != 0) storage_err = 1;

  storage_cleanup:
#if DW_NDIM >= 3
    CLOSE_FILE(dvzdbuoyancy)
    CLOSE_FILE(dvzdz)
    CLOSE_FILE(dvzdx_plus_dvxdz)
    CLOSE_FILE(dvzdy_plus_dvydz)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(dvydbuoyancy)
    CLOSE_FILE(dvydy)
    CLOSE_FILE(dvydx_plus_dvxdy)
#endif
    CLOSE_FILE(dvxdbuoyancy)
    CLOSE_FILE(dvxdx)
    if (storage_err) return 1;
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(backward)(
    DW_DTYPE const* __restrict const c11,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12,
#endif
    DW_DTYPE const* __restrict const c13, DW_DTYPE const* __restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c55_zy,
    DW_DTYPE const* __restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    /* Shear stiffness at the sigma_xy node: c66_yx in 3-D, c55_yx in 2-D. */
    DW_DTYPE const* __restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const grad_r_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const grad_r_y,
#endif
    DW_DTYPE const* __restrict const grad_r_x,
    DW_DTYPE const* __restrict const grad_r_p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmaxz,
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const m_sigmayzz,
    DW_DTYPE* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vyy,
    DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmaxyy,
    DW_DTYPE* __restrict const m_sigmaxyx,
#endif
    DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE* __restrict const m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE* __restrict const m_sigmaxyxn,
#endif
    DW_DTYPE* __restrict const m_sigmaxxxn,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
    DW_DTYPE* __restrict const dvzdbuoyancy_store_1b,
    void* __restrict const dvzdbuoyancy_store_2,
    void* __restrict const dvzdbuoyancy_store_3,
    char const* __restrict const* __restrict const dvzdbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvzdz_store_1,
    DW_DTYPE* __restrict const dvzdz_store_1b,
    void* __restrict const dvzdz_store_2, void* __restrict const dvzdz_store_3,
    char const* __restrict const* __restrict const dvzdz_filenames_ptr,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1b,
    void* __restrict const dvzdx_plus_dvxdz_store_2,
    void* __restrict const dvzdx_plus_dvxdz_store_3,
    char const* __restrict const* __restrict const dvzdx_plus_dvxdz_filenames_ptr,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1b,
    void* __restrict const dvzdy_plus_dvydz_store_2,
    void* __restrict const dvzdy_plus_dvydz_store_3,
    char const* __restrict const* __restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const dvydbuoyancy_store_1,
    DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
    void* __restrict const dvydbuoyancy_store_2,
    void* __restrict const dvydbuoyancy_store_3,
    char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvydy_store_1,
    DW_DTYPE* __restrict const dvydy_store_1b,
    void* __restrict const dvydy_store_2, void* __restrict const dvydy_store_3,
    char const* __restrict const* __restrict const dvydy_filenames_ptr,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1b,
    void* __restrict const dvydx_plus_dvxdy_store_2,
    void* __restrict const dvydx_plus_dvxdy_store_3,
    char const* __restrict const* __restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
    DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
    DW_DTYPE* __restrict const dvxdbuoyancy_store_1b,
    void* __restrict const dvxdbuoyancy_store_2,
    void* __restrict const dvxdbuoyancy_store_3,
    char const* __restrict const* __restrict const dvxdbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvxdx_store_1,
    DW_DTYPE* __restrict const dvxdx_store_1b,
    void* __restrict const dvxdx_store_2, void* __restrict const dvxdx_store_3,
    char const* __restrict const* __restrict const dvxdx_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_f_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_f_y,
#endif
    DW_DTYPE* __restrict const grad_f_x, DW_DTYPE* __restrict const grad_f_p,
    DW_DTYPE* __restrict const grad_c11,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_c12,
#endif
    DW_DTYPE* __restrict const grad_c13, DW_DTYPE* __restrict const grad_c33,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_c55_zy,
    DW_DTYPE* __restrict const grad_c55_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_buoyancy_y,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_x,
    DW_DTYPE* __restrict const grad_c11_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_c12_thread,
#endif
    DW_DTYPE* __restrict const grad_c13_thread,
    DW_DTYPE* __restrict const grad_c33_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_c55_zy_thread,
    DW_DTYPE* __restrict const grad_c55_zx_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_c_xy_thread,
#endif
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_buoyancy_y_thread,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_x_thread,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_y_i,
#endif
    int64_t const* __restrict const sources_x_i,
    int64_t const* __restrict const sources_p_i,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_y_i,
#endif
    int64_t const* __restrict const receivers_x_i,
    int64_t const* __restrict const receivers_p_i,
#if DW_NDIM >= 3
    DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy,
#endif
    DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
    int64_t const n_shots,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const n_sources_z_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_y_per_shot,
#endif
    int64_t const n_sources_x_per_shot, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_z_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_y_per_shot,
#endif
    int64_t n_receivers_x_per_shot, int64_t const n_receivers_p_per_shot,
    int64_t const step_ratio, int64_t const storage_mode,
    size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
    bool const c11_requires_grad,
#if DW_NDIM >= 3
    bool const c12_requires_grad,
#endif
    bool const c13_requires_grad, bool const c33_requires_grad,
    bool const c55_requires_grad,
#if DW_NDIM >= 3
    bool const c66_requires_grad,
#endif
    bool const buoyancy_requires_grad, bool const c11_batched,
#if DW_NDIM >= 3
    bool const c12_batched,
#endif
    bool const c13_batched, bool const c33_batched, bool const c55_batched,
#if DW_NDIM >= 3
    bool const c66_batched,
#endif
    bool const buoyancy_batched, bool const storage_compression,
    int64_t start_t,
#if DW_NDIM >= 3
    int64_t const pml_z0,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y0,
#endif
    int64_t const pml_x0,
#if DW_NDIM >= 3
    int64_t const pml_z1,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y1,
#endif
    int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0,
#endif
    int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y1,
#endif
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((c11_requires_grad) && (grad_c11 == NULL)) return 1;
#if DW_NDIM >= 3
    if ((c12_requires_grad) && (grad_c12 == NULL)) return 1;
#endif
    if ((c13_requires_grad) && (grad_c13 == NULL)) return 1;
    if ((c33_requires_grad) && (grad_c33 == NULL)) return 1;
#if DW_NDIM >= 3
    if ((buoyancy_requires_grad) && (grad_buoyancy_z == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((buoyancy_requires_grad) && (grad_buoyancy_y == NULL)) return 1;
#endif
    if ((buoyancy_requires_grad) && (grad_buoyancy_x == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0) && sources_z_i == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot > 0) && sources_y_i == NULL) return 1;
#endif
    if ((n_sources_x_per_shot > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_p_per_shot > 0) && sources_p_i == NULL) return 1;
  }
#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
#if DW_NDIM >= 2
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
#endif
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 2 */
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#endif
  int64_t shot;
  /* Shots run serially; each time step's adjoint velocity and stress
   * stencils are spread across threads (see the omp parallel regions below).
   * Model gradients accumulate directly into the shared gradient buffers --
   * each cell is written by exactly one thread, and time steps and shots are
   * barrier-separated, so there is no race. */
  (void)grad_c11_thread;
#if DW_NDIM >= 3
  (void)grad_c12_thread;
#endif
  (void)grad_c13_thread;
  (void)grad_c33_thread;
#if DW_NDIM >= 3
  (void)grad_c55_zy_thread;
  (void)grad_c55_zx_thread;
  (void)grad_buoyancy_z_thread;
#endif
#if DW_NDIM >= 2
  (void)grad_c_xy_thread;
  (void)grad_buoyancy_y_thread;
#endif
  (void)grad_buoyancy_x_thread;
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
    int64_t const n_grid_points = ny * nx;
#else
    int64_t const n_grid_points = nx;
#endif
    int64_t const si = shot * n_grid_points;
#if DW_NDIM >= 3
    int64_t const siz = shot * n_sources_z_per_shot;
    int64_t const riz = shot * n_receivers_z_per_shot;
    int64_t const* __restrict const sources_z_i_shot =
        sources_z_i != NULL ? (sources_z_i + siz) : NULL;
    int64_t const* __restrict const receivers_z_i_shot =
        receivers_z_i != NULL ? (receivers_z_i + riz) : NULL;
    DW_DTYPE* __restrict const grad_f_z_shot =
        grad_f_z != NULL ? (grad_f_z + siz) : NULL;
    DW_DTYPE const* __restrict const grad_r_z_shot =
        grad_r_z != NULL ? (grad_r_z + riz) : NULL;
    DW_DTYPE const* __restrict const c55_zy_shot =
        c55_batched ? c55_zy + si : c55_zy;
    DW_DTYPE const* __restrict const c55_zx_shot =
        c55_batched ? c55_zx + si : c55_zx;
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        buoyancy_batched ? buoyancy_z + si : buoyancy_z;
    DW_DTYPE* __restrict const grad_c55_zy_shot =
        grad_c55_zy != NULL ? (grad_c55_zy + (c55_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_c55_zx_shot =
        grad_c55_zx != NULL ? (grad_c55_zx + (c55_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_buoyancy_z_shot =
        grad_buoyancy_z != NULL
            ? (grad_buoyancy_z + (buoyancy_batched ? si : 0))
            : NULL;
    DW_DTYPE* __restrict const vz_shot = vz + si;
    DW_DTYPE* __restrict const sigmazz_shot = sigmazz + si;
    DW_DTYPE* __restrict const sigmayz_shot = sigmayz + si;
    DW_DTYPE* __restrict const sigmaxz_shot = sigmaxz + si;
    DW_DTYPE* __restrict const m_vzz_shot = m_vzz + si;
    DW_DTYPE* __restrict const m_vzy_shot = m_vzy + si;
    DW_DTYPE* __restrict const m_vzx_shot = m_vzx + si;
    DW_DTYPE* __restrict const m_vyz_shot = m_vyz + si;
    DW_DTYPE* __restrict const m_vxz_shot = m_vxz + si;
#endif
#if DW_NDIM >= 2
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const* __restrict const sources_y_i_shot =
        sources_y_i != NULL ? (sources_y_i + siy) : NULL;
    int64_t const* __restrict const receivers_y_i_shot =
        receivers_y_i != NULL ? (receivers_y_i + riy) : NULL;
    DW_DTYPE* __restrict const grad_f_y_shot =
        grad_f_y != NULL ? (grad_f_y + siy) : NULL;
    DW_DTYPE const* __restrict const grad_r_y_shot =
        grad_r_y != NULL ? (grad_r_y + riy) : NULL;
    DW_DTYPE const* __restrict const c_xy_shot =
#if DW_NDIM >= 3
        c66_batched ? c_xy + si : c_xy;
#else
        c55_batched ? c_xy + si : c_xy;
#endif
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE* __restrict const grad_c_xy_shot =
#if DW_NDIM >= 3
        grad_c_xy != NULL ? (grad_c_xy + (c66_batched ? si : 0)) : NULL;
#else
        grad_c_xy != NULL ? (grad_c_xy + (c55_batched ? si : 0)) : NULL;
#endif
    DW_DTYPE* __restrict const grad_buoyancy_y_shot =
        grad_buoyancy_y != NULL
            ? (grad_buoyancy_y + (buoyancy_batched ? si : 0))
            : NULL;
    DW_DTYPE* __restrict const vy_shot = vy + si;
    DW_DTYPE* __restrict const sigmayy_shot = sigmayy + si;
    DW_DTYPE* __restrict const sigmaxy_shot = sigmaxy + si;
    DW_DTYPE* __restrict const m_vyy_shot = m_vyy + si;
    DW_DTYPE* __restrict const m_vyx_shot = m_vyx + si;
    DW_DTYPE* __restrict const m_vxy_shot = m_vxy + si;
#endif
    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const sip = shot * n_sources_p_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const rip = shot * n_receivers_p_per_shot;
    int64_t const* __restrict const sources_x_i_shot =
        sources_x_i != NULL ? (sources_x_i + six) : NULL;
    int64_t const* __restrict const sources_p_i_shot =
        sources_p_i != NULL ? (sources_p_i + sip) : NULL;
    int64_t const* __restrict const receivers_x_i_shot =
        receivers_x_i != NULL ? (receivers_x_i + rix) : NULL;
    int64_t const* __restrict const receivers_p_i_shot =
        receivers_p_i != NULL ? (receivers_p_i + rip) : NULL;
    DW_DTYPE* __restrict const grad_f_x_shot =
        grad_f_x != NULL ? (grad_f_x + six) : NULL;
    DW_DTYPE* __restrict const grad_f_p_shot =
        grad_f_p != NULL ? (grad_f_p + sip) : NULL;
    DW_DTYPE const* __restrict const grad_r_x_shot =
        grad_r_x != NULL ? (grad_r_x + rix) : NULL;
    DW_DTYPE const* __restrict const grad_r_p_shot =
        grad_r_p != NULL ? (grad_r_p + rip) : NULL;

    DW_DTYPE* __restrict const grad_c11_shot =
        grad_c11 != NULL ? (grad_c11 + (c11_batched ? si : 0)) : NULL;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_c12_shot =
        grad_c12 != NULL ? (grad_c12 + (c12_batched ? si : 0)) : NULL;
#endif
    DW_DTYPE* __restrict const grad_c13_shot =
        grad_c13 != NULL ? (grad_c13 + (c13_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_c33_shot =
        grad_c33 != NULL ? (grad_c33 + (c33_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_buoyancy_x_shot =
        grad_buoyancy_x != NULL
            ? (grad_buoyancy_x + (buoyancy_batched ? si : 0))
            : NULL;
    DW_DTYPE const* __restrict const c11_shot = c11_batched ? c11 + si : c11;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12_shot = c12_batched ? c12 + si : c12;
#endif
    DW_DTYPE const* __restrict const c13_shot = c13_batched ? c13 + si : c13;
    DW_DTYPE const* __restrict const c33_shot = c33_batched ? c33 + si : c33;
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE* __restrict const vx_shot = vx + si;
    DW_DTYPE* __restrict const sigmaxx_shot = sigmaxx + si;
    DW_DTYPE* __restrict const m_vxx_shot = m_vxx + si;
    int64_t t;

    // See the forward pass: these aggregate the requires-grad conditions that
    // gate each stored velocity-derivative array (must match Python's
    // storage_flags sizing).
#if DW_NDIM >= 3
    bool const cnormal_requires_grad = c11_requires_grad || c12_requires_grad ||
                                       c13_requires_grad || c33_requires_grad;
    bool const c_xy_requires_grad = c66_requires_grad;
#else
    bool const cnormal_requires_grad =
        c11_requires_grad || c13_requires_grad || c33_requires_grad;
    bool const c_xy_requires_grad = c55_requires_grad;
#endif
    int storage_err = 0;
#if DW_NDIM >= 3
    FILE* fp_dvzdbuoyancy = NULL;
    FILE* fp_dvzdz = NULL;
    FILE* fp_dvzdx_plus_dvxdz = NULL;
    FILE* fp_dvzdy_plus_dvydz = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_dvydbuoyancy = NULL;
    FILE* fp_dvydy = NULL;
    FILE* fp_dvydx_plus_dvxdy = NULL;
#endif
    FILE* fp_dvxdbuoyancy = NULL;
    FILE* fp_dvxdx = NULL;

#if DW_NDIM >= 3
    OPEN_FILE_READ(dvzdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_READ(dvzdz, cnormal_requires_grad)
    OPEN_FILE_READ(dvzdx_plus_dvxdz, c55_requires_grad)
    OPEN_FILE_READ(dvzdy_plus_dvydz, c55_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_READ(dvydbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_READ(dvydy, cnormal_requires_grad)
    OPEN_FILE_READ(dvydx_plus_dvxdy, c_xy_requires_grad)
#endif
    OPEN_FILE_READ(dvxdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_READ(dvxdx, cnormal_requires_grad)

    for (t = start_t - 1; t >= start_t - nt; --t) {
#if DW_NDIM >= 3
      SETUP_STORE_LOAD(dvzdbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvzdz, cnormal_requires_grad)
      SETUP_STORE_LOAD(dvzdx_plus_dvxdz, c55_requires_grad)
      SETUP_STORE_LOAD(dvzdy_plus_dvydz, c55_requires_grad)
      DW_DTYPE* __restrict const m_sigmazzz_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmazzzn : m_sigmazzz) + si;
      DW_DTYPE* __restrict const m_sigmayzy_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmayzyn : m_sigmayzy) + si;
      DW_DTYPE* __restrict const m_sigmaxzx_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxzxn : m_sigmaxzx) + si;
      DW_DTYPE* __restrict const m_sigmayzz_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmayzzn : m_sigmayzz) + si;
      DW_DTYPE* __restrict const m_sigmaxzz_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxzzn : m_sigmaxzz) + si;
      DW_DTYPE* __restrict const m_sigmazzzn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmazzz : m_sigmazzzn) + si;
      DW_DTYPE* __restrict const m_sigmayzyn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmayzy : m_sigmayzyn) + si;
      DW_DTYPE* __restrict const m_sigmaxzxn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxzx : m_sigmaxzxn) + si;
      DW_DTYPE* __restrict const m_sigmayzzn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmayzz : m_sigmayzzn) + si;
      DW_DTYPE* __restrict const m_sigmaxzzn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxzz : m_sigmaxzzn) + si;
#endif
#if DW_NDIM >= 2
      SETUP_STORE_LOAD(dvydbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvydy, cnormal_requires_grad)
      SETUP_STORE_LOAD(dvydx_plus_dvxdy, c_xy_requires_grad)
      DW_DTYPE* __restrict const m_sigmayyy_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmayyyn : m_sigmayyy) + si;
      DW_DTYPE* __restrict const m_sigmaxyy_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxyyn : m_sigmaxyy) + si;
      DW_DTYPE* __restrict const m_sigmaxyx_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxyxn : m_sigmaxyx) + si;
      DW_DTYPE* __restrict const m_sigmayyyn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmayyy : m_sigmayyyn) + si;
      DW_DTYPE* __restrict const m_sigmaxyyn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxyy : m_sigmaxyyn) + si;
      DW_DTYPE* __restrict const m_sigmaxyxn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxyx : m_sigmaxyxn) + si;
#endif
      SETUP_STORE_LOAD(dvxdbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvxdx, cnormal_requires_grad)

      bool const rg_step_t = ((t % step_ratio) == 0);
      bool const c11_requires_grad_t = c11_requires_grad && rg_step_t;
#if DW_NDIM >= 3
      bool const c12_requires_grad_t = c12_requires_grad && rg_step_t;
#endif
      bool const c13_requires_grad_t = c13_requires_grad && rg_step_t;
      bool const c33_requires_grad_t = c33_requires_grad && rg_step_t;
#if DW_NDIM >= 3
      bool const c55_requires_grad_t = c55_requires_grad && rg_step_t;
      bool const c_xy_requires_grad_t = c66_requires_grad && rg_step_t;
#else
      bool const c_xy_requires_grad_t = c55_requires_grad && rg_step_t;
#endif
      bool const buoyancy_requires_grad_t = buoyancy_requires_grad && rg_step_t;
      DW_DTYPE* __restrict const m_sigmaxxx_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxxxn : m_sigmaxxx) + si;
      DW_DTYPE* __restrict const m_sigmaxxxn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigmaxxx : m_sigmaxxxn) + si;

      /* Fold the (adjoint) stress guard ring back into the interior before
       * the velocity-adjoint update loops read it and before the
       * source-gradient extraction just below (the adjoint counterpart of
       * forward's pressure receiver record). Unlike acoustic's pressure
       * fill, elastic_vti's multi-field stress fill is not self-adjoint
       * (see the module comment above FUNC(fill_elastic_stress_bc_adjoint)
       * in this file), so this applies that fill's true transpose rather
       * than reapplying the forward fill. */
      if (has_bc) {
        FUNC(fill_elastic_stress_bc_adjoint)(
#if DW_NDIM == 3
            sigmazz_shot, sigmayz_shot, sigmaxz_shot,
#endif
            sigmayy_shot, sigmaxy_shot, sigmaxx_shot, bc,
#if DW_NDIM == 3
            nz,
#else
            0,
#endif
            ny, nx);
      }

      if (n_sources_p_per_shot > 0) {
        record_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, sources_p_i_shot,
            grad_f_p_shot + t * n_shots * n_sources_p_per_shot,
            n_sources_p_per_shot);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
              DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

#if DW_NDIM >= 3

              // vz

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(backward_vz_cell)(
                        c13_shot, c33_shot, c55_zy_shot, c55_zx_shot,
                        sigmazz_shot, sigmayy_shot, sigmaxx_shot, sigmayz_shot,
                        sigmaxz_shot, vz_shot, m_vzz_shot, m_vzy_shot,
                        m_vzx_shot, m_sigmazzz_t, m_sigmazzzn_t, m_sigmayzy_t,
                        m_sigmayzyn_t, m_sigmaxzx_t, m_sigmaxzxn_t, az, azh, bz,
                        bzh, ay, ayh, by, byh, ax, axh, bx, bxh, dt, i, i, ny,
                        nx, z, y, x, is_pml_z, is_pml_y, is_pml_x, rdz, rdy,
                        rdx);
                  }
#endif

#if DW_NDIM >= 2

              // vy

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
#if DW_NDIM >= 3
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
#endif
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(backward_vy_cell)(
#if DW_NDIM >= 3
                        c55_zy_shot, c11_shot, c12_shot, c13_shot,
#else
                      c13_shot, c33_shot,
#endif
                        c_xy_shot,
#if DW_NDIM >= 3
                        sigmazz_shot,
#endif
                        sigmayy_shot, sigmaxx_shot,
#if DW_NDIM >= 3
                        sigmayz_shot,
#endif
                        sigmaxy_shot, vy_shot,
#if DW_NDIM >= 3
                        m_vyz_shot,
#endif
                        m_vyy_shot, m_vyx_shot,
#if DW_NDIM >= 3
                        m_sigmayzz_t, m_sigmayzzn_t,
#endif
                        m_sigmayyy_t, m_sigmayyyn_t, m_sigmaxyx_t,
                        m_sigmaxyxn_t,
#if DW_NDIM >= 3
                        az, azh, bz, bzh,
#endif
                        ay, ayh, by, byh, ax, axh, bx, bxh, dt, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z, is_pml_z, rdz,
#endif
                        y, x, is_pml_y, is_pml_x, rdy, rdx);
                  }

#endif

              // vx
              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
#if DW_NDIM >= 3
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
#endif
#if DW_NDIM >= 2
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
#endif
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(backward_vx_cell)(
#if DW_NDIM >= 3
                        c55_zx_shot,
#endif
#if DW_NDIM >= 2
                        c_xy_shot,
#endif
                        c11_shot,
#if DW_NDIM >= 2
                        c13_shot,
#endif
#if DW_NDIM >= 3
                        c12_shot,
#endif
#if DW_NDIM >= 3
                        sigmazz_shot,
#endif
#if DW_NDIM >= 2
                        sigmayy_shot,
#endif
                        sigmaxx_shot,
#if DW_NDIM >= 3
                        sigmaxz_shot,
#endif
#if DW_NDIM >= 2
                        sigmaxy_shot,
#endif
                        vx_shot,
#if DW_NDIM >= 3
                        m_vxz_shot,
#endif
#if DW_NDIM >= 2
                        m_vxy_shot,
#endif
                        m_vxx_shot,
#if DW_NDIM >= 3
                        m_sigmaxzz_t, m_sigmaxzzn_t,
#endif
#if DW_NDIM >= 2
                        m_sigmaxyy_t, m_sigmaxyyn_t,
#endif
                        m_sigmaxxx_t, m_sigmaxxxn_t,
#if DW_NDIM >= 3
                        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
                        ay, ayh, by, byh,
#endif
                        ax, axh, bx, bxh, dt, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, is_pml_z, rdz,
#endif
#if DW_NDIM >= 2
                        y, is_pml_y, rdy,
#endif
                        x, is_pml_x, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      /* Guard-band extension of the velocity-adjoint gathers (see the
       * "Guard-band gathers" module comment in elastic_vti_kernels.h,
       * above FUNC(backward_vz_cell_guard)): the interior loops just above
       * never visit a cell within FD_PAD of a non-PML face, so FUNC(fill_
       * elastic_velocity_bc_adjoint) below has nothing genuine to fold
       * there without this. Runs single-threaded, like the fold-fill
       * itself: a cell can be a simultaneous target of more than one
       * axis's extended range at once (a genuine corner when several
       * faces are non-PML), so parallelising this would race on it; the
       * work here is O(FD_PAD * grid-face-area), negligible next to the
       * interior loops above. */
      if (has_bc) {
#if DW_NDIM >= 3
        {
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], true);
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], false);
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], false);
          int64_t z, y, x;
          for (z = rz.target_lo; z < rz.target_hi; ++z)
            for (y = ry.target_lo; y < ry.target_hi; ++y)
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                bool const in_target = FUNC(dw_in_target)(rz, z) &&
                                       FUNC(dw_in_target)(ry, y) &&
                                       FUNC(dw_in_target)(rx, x);
                bool const all_std = FUNC(dw_in_std)(rz, z) &&
                                     FUNC(dw_in_std)(ry, y) &&
                                     FUNC(dw_in_std)(rx, x);
                if (!in_target || all_std) continue;
                SET_I;
                /* Guard-ring cells have no forward-pass counterpart at
                 * all (forward's own update loops never reach them --
                 * see the module comment above), so there is no "this
                 * cell's own PML formula" to transpose here. Always take
                 * the plain (non-PML) branch, avoiding invented recursive
                 * state with no forward counterpart. */
                bool const is_pml_z = false;
                bool const is_pml_y = false;
                bool const is_pml_x = false;
                FUNC(backward_vz_cell_guard)(
                    c13_shot, c33_shot, c55_zy_shot, c55_zx_shot, sigmazz_shot,
                    sigmayy_shot, sigmaxx_shot, sigmayz_shot, sigmaxz_shot,
                    vz_shot, m_vzz_shot, m_vzy_shot, m_vzx_shot, m_sigmazzz_t,
                    m_sigmazzzn_t, m_sigmayzy_t, m_sigmayzyn_t, m_sigmaxzx_t,
                    m_sigmaxzxn_t, buoyancy_z_shot[i], az, azh, bz, bzh, ay,
                    ayh, by, byh, ax, axh, bx, bxh, dt, i, i, nz, ny, nx, z, y,
                    x, is_pml_z, is_pml_y, is_pml_x, rdz, rdy, rdx);
              }
        }
#endif
#if DW_NDIM >= 2
        {
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], true);
#if DW_NDIM >= 3
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], false);
#endif
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], false);
#if DW_NDIM >= 3
          int64_t z;
#endif
          int64_t y, x;
#if DW_NDIM >= 3
          for (z = rz.target_lo; z < rz.target_hi; ++z)
#endif
            for (y = ry.target_lo; y < ry.target_hi; ++y)
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                bool const in_target = FUNC(dw_in_target)(ry, y) &&
                                       FUNC(dw_in_target)(rx, x)
#if DW_NDIM >= 3
                                       && FUNC(dw_in_target)(rz, z)
#endif
                    ;
                bool const all_std = FUNC(dw_in_std)(ry, y) &&
                                     FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 3
                                     && FUNC(dw_in_std)(rz, z)
#endif
                    ;
                if (!in_target || all_std) continue;
                SET_I;
                /* Guard cells have no forward-pass PML state -- see the
                 * module comment above the vz guard-band block. */
                bool const is_pml_y = false;
                bool const is_pml_x = false;
#if DW_NDIM >= 3
                bool const is_pml_z = false;
#endif
                FUNC(backward_vy_cell_guard)(
#if DW_NDIM >= 3
                    c55_zy_shot, c11_shot, c12_shot, c13_shot,
#else
                  c13_shot, c33_shot,
#endif
                    c_xy_shot,
#if DW_NDIM >= 3
                    sigmazz_shot,
#endif
                    sigmayy_shot, sigmaxx_shot,
#if DW_NDIM >= 3
                    sigmayz_shot,
#endif
                    sigmaxy_shot, vy_shot,
#if DW_NDIM >= 3
                    m_vyz_shot,
#endif
                    m_vyy_shot, m_vyx_shot,
#if DW_NDIM >= 3
                    m_sigmayzz_t, m_sigmayzzn_t,
#endif
                    m_sigmayyy_t, m_sigmayyyn_t, m_sigmaxyx_t, m_sigmaxyxn_t,
                    buoyancy_y_shot[i],
#if DW_NDIM >= 3
                    az, azh, bz, bzh,
#endif
                    ay, ayh, by, byh, ax, axh, bx, bxh, dt, i, i,
#if DW_NDIM >= 3
                    nz,
#endif
                    ny, nx,
#if DW_NDIM >= 3
                    z, is_pml_z, rdz,
#endif
                    y, x, is_pml_y, is_pml_x, rdy, rdx);
              }
        }
#endif
        {
#if DW_NDIM >= 3
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], false);
#endif
#if DW_NDIM >= 2
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], false);
#endif
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], true);
#if DW_NDIM >= 3
          int64_t z;
#endif
#if DW_NDIM >= 2
          int64_t y;
#endif
          int64_t x;
#if DW_NDIM >= 3
          for (z = rz.target_lo; z < rz.target_hi; ++z)
#endif
#if DW_NDIM >= 2
            for (y = ry.target_lo; y < ry.target_hi; ++y)
#endif
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                bool const in_target = FUNC(dw_in_target)(rx, x)
#if DW_NDIM >= 2
                                       && FUNC(dw_in_target)(ry, y)
#endif
#if DW_NDIM >= 3
                                       && FUNC(dw_in_target)(rz, z)
#endif
                    ;
                bool const all_std = FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 2
                                     && FUNC(dw_in_std)(ry, y)
#endif
#if DW_NDIM >= 3
                                     && FUNC(dw_in_std)(rz, z)
#endif
                    ;
                if (!in_target || all_std) continue;
                SET_I;
            /* Guard cells have no forward-pass PML state -- see the
             * module comment above the vz guard-band block. */
#if DW_NDIM >= 3
                bool const is_pml_z = false;
#endif
#if DW_NDIM >= 2
                bool const is_pml_y = false;
#endif
                bool const is_pml_x = false;
                FUNC(backward_vx_cell_guard)(
#if DW_NDIM >= 3
                    c55_zx_shot,
#endif
#if DW_NDIM >= 2
                    c_xy_shot,
#endif
                    c11_shot,
#if DW_NDIM >= 2
                    c13_shot,
#endif
#if DW_NDIM >= 3
                    c12_shot,
#endif
#if DW_NDIM >= 3
                    sigmazz_shot,
#endif
#if DW_NDIM >= 2
                    sigmayy_shot,
#endif
                    sigmaxx_shot,
#if DW_NDIM >= 3
                    sigmaxz_shot,
#endif
#if DW_NDIM >= 2
                    sigmaxy_shot,
#endif
                    vx_shot,
#if DW_NDIM >= 3
                    m_vxz_shot,
#endif
#if DW_NDIM >= 2
                    m_vxy_shot,
#endif
                    m_vxx_shot,
#if DW_NDIM >= 3
                    m_sigmaxzz_t, m_sigmaxzzn_t,
#endif
#if DW_NDIM >= 2
                    m_sigmaxyy_t, m_sigmaxyyn_t,
#endif
                    m_sigmaxxx_t, m_sigmaxxxn_t, buoyancy_x_shot[i],
#if DW_NDIM >= 3
                    az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
                    ay, ayh, by, byh,
#endif
                    ax, axh, bx, bxh, dt, i, i,
#if DW_NDIM >= 3
                    nz,
#endif
#if DW_NDIM >= 2
                    ny,
#endif
                    nx,
#if DW_NDIM >= 3
                    z, is_pml_z, rdz,
#endif
#if DW_NDIM >= 2
                    y, is_pml_y, rdy,
#endif
                    x, is_pml_x, rdx);
              }
        }
      }

      /* Fold the (adjoint) velocity guard ring back into the interior
       * before the stress-adjoint update loops just below read it. Unlike
       * acoustic's velocity fill, elastic_vti's multi-field velocity fill
       * is not self-adjoint (see the module comment above FUNC(fill_
       * elastic_velocity_bc_adjoint)), so this applies that fill's true
       * transpose rather than reapplying the forward fill. */
      if (has_bc) {
        FUNC(fill_elastic_velocity_bc_adjoint)(
#if DW_NDIM == 3
            vz_shot,
#endif
            vy_shot, vx_shot, bc,
#if DW_NDIM == 3
            nz,
#else
            0,
#endif
            ny, nx);
      }

      /* Velocity-type source-gradient extraction, the M_SIGMA** PML
       * memory-variable update (split out of FUNC(backward_v*_cell) --
       * see FUNC(backward_vz_cell_pml_mem)'s comment in elastic_vti_
       * kernels.h), and the buoyancy-gradient accumulation all read the
       * velocity adjoint at (near-face) interior cells the fold above may
       * have just changed, so all three must run after it: a pre-fold
       * read there is stale under the corrected architecture, exactly
       * like FUNC(fill_elastic_stress_bc_adjoint)'s placement before
       * record_pressure/the sigmaii gather above. Every one of these
       * three uses the SAME (unextended) interior/PML range FUNC(forward_
       * vz_cell) etc. used in the forward pass, so this stays a single
       * flat loop over that original range, not the guard-extended one;
       * the M_SIGMA** write must run unconditionally (every time step,
       * regardless of buoyancy_requires_grad) since FUNC(backward_sigma*_
       * cell) reads it next time step regardless of whether any gradient
       * is requested. */
#if DW_NDIM >= 3
      if (n_sources_z_per_shot > 0) {
        record_from_wavefield(
            vz_shot, sources_z_i_shot,
            grad_f_z_shot + t * n_shots * n_sources_z_per_shot,
            n_sources_z_per_shot);
      }
      {
        int64_t z, y, x;
        for (z = pml_bounds_zh[0]; z < pml_bounds_zh[3]; ++z)
          for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
            for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
              SET_I;
              bool const is_pml_z = FUNC(dw_is_pml_region)(
                  z, pml_z0, MAX(pml_z0, pml_z1 - 1), nz - FD_PAD);
              bool const is_pml_y =
                  FUNC(dw_is_pml_region)(y, pml_y0, pml_y1, ny - FD_PAD + 1);
              bool const is_pml_x =
                  FUNC(dw_is_pml_region)(x, pml_x0, pml_x1, nx - FD_PAD + 1);
              FUNC(backward_vz_cell_pml_mem)(
                  vz_shot, m_sigmazzz_t, m_sigmazzzn_t, m_sigmayzy_t,
                  m_sigmayzyn_t, m_sigmaxzx_t, m_sigmaxzxn_t,
                  buoyancy_z_shot[i], azh, ay, ax, dt, i, ny, nx, z, y, x,
                  is_pml_z, is_pml_y, is_pml_x);
              if (buoyancy_requires_grad_t) {
                grad_buoyancy_z_shot[i] += vz_shot[i] *
                                           dvzdbuoyancy_store_1_t[i] *
                                           (DW_DTYPE)step_ratio;
              }
            }
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_y_per_shot > 0) {
        record_from_wavefield(
            vy_shot, sources_y_i_shot,
            grad_f_y_shot + t * n_shots * n_sources_y_per_shot,
            n_sources_y_per_shot);
      }
      {
#if DW_NDIM >= 3
        int64_t z;
#endif
        int64_t y, x;
#if DW_NDIM >= 3
        for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
          for (y = pml_bounds_yh[0]; y < pml_bounds_yh[3]; ++y)
            for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
              SET_I;
              bool const is_pml_y = FUNC(dw_is_pml_region)(
                  y, pml_y0, MAX(pml_y0, pml_y1 - 1), ny - FD_PAD);
              bool const is_pml_x =
                  FUNC(dw_is_pml_region)(x, pml_x0, pml_x1, nx - FD_PAD + 1);
#if DW_NDIM >= 3
              bool const is_pml_z =
                  FUNC(dw_is_pml_region)(z, pml_z0, pml_z1, nz - FD_PAD + 1);
#endif
              FUNC(backward_vy_cell_pml_mem)(vy_shot,
#if DW_NDIM >= 3
                                             m_sigmayzz_t, m_sigmayzzn_t,
#endif
                                             m_sigmayyy_t, m_sigmayyyn_t,
                                             m_sigmaxyx_t, m_sigmaxyxn_t,
                                             buoyancy_y_shot[i],
#if DW_NDIM >= 3
                                             az,
#endif
                                             ayh, ax, dt, i,
#if DW_NDIM >= 3
                                             ny, z, is_pml_z,
#endif
                                             nx, y, x, is_pml_y, is_pml_x);
              if (buoyancy_requires_grad_t) {
                grad_buoyancy_y_shot[i] += vy_shot[i] *
                                           dvydbuoyancy_store_1_t[i] *
                                           (DW_DTYPE)step_ratio;
              }
            }
      }
#endif
      if (n_sources_x_per_shot > 0) {
        record_from_wavefield(
            vx_shot, sources_x_i_shot,
            grad_f_x_shot + t * n_shots * n_sources_x_per_shot,
            n_sources_x_per_shot);
      }
      {
#if DW_NDIM >= 3
        int64_t z;
#endif
#if DW_NDIM >= 2
        int64_t y;
#endif
        int64_t x;
#if DW_NDIM >= 3
        for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
#if DW_NDIM >= 2
          for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
#endif
            for (x = pml_bounds_xh[0]; x < pml_bounds_xh[3]; ++x) {
              SET_I;
#if DW_NDIM >= 3
              bool const is_pml_z =
                  FUNC(dw_is_pml_region)(z, pml_z0, pml_z1, nz - FD_PAD + 1);
#endif
#if DW_NDIM >= 2
              bool const is_pml_y =
                  FUNC(dw_is_pml_region)(y, pml_y0, pml_y1, ny - FD_PAD + 1);
#endif
              bool const is_pml_x = FUNC(dw_is_pml_region)(
                  x, pml_x0, MAX(pml_x0, pml_x1 - 1), nx - FD_PAD);
              FUNC(backward_vx_cell_pml_mem)(vx_shot,
#if DW_NDIM >= 3
                                             m_sigmaxzz_t, m_sigmaxzzn_t,
#endif
#if DW_NDIM >= 2
                                             m_sigmaxyy_t, m_sigmaxyyn_t,
#endif
                                             m_sigmaxxx_t, m_sigmaxxxn_t,
                                             buoyancy_x_shot[i],
#if DW_NDIM >= 3
                                             az,
#endif
#if DW_NDIM >= 2
                                             ay,
#endif
                                             axh, dt, i,
#if DW_NDIM >= 3
                                             ny, z, is_pml_z,
#endif
#if DW_NDIM >= 2
                                             nx, y, is_pml_y,
#endif
                                             x, is_pml_x);
              if (buoyancy_requires_grad_t) {
                grad_buoyancy_x_shot[i] += vx_shot[i] *
                                           dvxdbuoyancy_store_1_t[i] *
                                           (DW_DTYPE)step_ratio;
              }
            }
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
              DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

              // sigmaii

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
#if DW_NDIM >= 3
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
#endif
#if DW_NDIM >= 2
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
#endif
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(backward_sigmaii_cell)(
                        c11_shot,
#if DW_NDIM >= 3
                        c12_shot,
#endif
                        c13_shot, c33_shot,
#if DW_NDIM >= 3
                        buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
                        buoyancy_y_shot,
#endif
                        buoyancy_x_shot,
#if DW_NDIM >= 3
                        vz_shot, sigmazz_shot, m_vzz_shot, m_sigmazzz_t,
                        dvzdz_store_1_t,
#endif
#if DW_NDIM >= 2
                        vy_shot, sigmayy_shot, m_vyy_shot, m_sigmayyy_t,
                        dvydy_store_1_t,
#endif
                        vx_shot, sigmaxx_shot, m_vxx_shot, m_sigmaxxx_t,
                        dvxdx_store_1_t, grad_c11_shot,
#if DW_NDIM >= 3
                        grad_c12_shot,
#endif
                        grad_c13_shot, grad_c33_shot, dt, step_ratio,
                        c11_requires_grad_t,
#if DW_NDIM >= 3
                        c12_requires_grad_t,
#endif
                        c13_requires_grad_t, c33_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, is_pml_z, az, azh, bz, bzh, rdz,
#endif
#if DW_NDIM >= 2
                        y, is_pml_y, ay, ayh, by, byh, rdy,
#endif
                        x, is_pml_x, ax, axh, bx, bxh, rdx);
                  }

#if DW_NDIM >= 2

              // sigmaxy

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(backward_sigmaxy_cell)(
                        c_xy_shot, buoyancy_x_shot, buoyancy_y_shot, vx_shot,
                        vy_shot, sigmaxy_shot, m_vxy_shot, m_vyx_shot,
                        m_sigmaxyy_t, m_sigmaxyx_t, dvydx_plus_dvxdy_store_1_t,
                        grad_c_xy_shot, dt, step_ratio, c_xy_requires_grad_t, i,
                        i,
#if DW_NDIM >= 3
                        ny, z,
#endif
                        nx, y, x, is_pml_y, is_pml_x, ay, ayh, by, byh, ax, axh,
                        bx, bxh, rdy, rdx);
                  }

#endif

#if DW_NDIM >= 3

              // sigmaxz

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
                    bool const is_pml_x = !DW_PML_SKIP(pml_x == 1);
                    FUNC(backward_sigmaxz_cell)(
                        c55_zx_shot, buoyancy_x_shot, buoyancy_z_shot, vx_shot,
                        vz_shot, sigmaxz_shot, m_vxz_shot, m_vzx_shot,
                        m_sigmaxzz_t, m_sigmaxzx_t, dvzdx_plus_dvxdz_store_1_t,
                        grad_c55_zx_shot, dt, step_ratio, c55_requires_grad_t,
                        i, i, ny, nx, z, y, x, is_pml_z, is_pml_x, az, azh, bz,
                        bzh, ax, axh, bx, bxh, rdz, rdx);
                  }
#endif

#if DW_NDIM >= 3

              // sigmayz

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    bool const is_pml_z = !DW_PML_SKIP(pml_z == 1);
                    bool const is_pml_y = !DW_PML_SKIP(pml_y == 1);
                    FUNC(backward_sigmayz_cell)(
                        c55_zy_shot, buoyancy_y_shot, buoyancy_z_shot, vy_shot,
                        vz_shot, sigmayz_shot, m_vyz_shot, m_vzy_shot,
                        m_sigmayzz_t, m_sigmayzy_t, dvzdy_plus_dvydz_store_1_t,
                        grad_c55_zy_shot, dt, step_ratio, c55_requires_grad_t,
                        i, i, ny, nx, z, y, x, is_pml_z, is_pml_y, az, azh, bz,
                        bzh, ay, ayh, by, byh, rdz, rdy);
                  }

#endif
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      /* Guard-band extension of the stress-adjoint gathers (see the
       * "Guard-band gathers" module comment in elastic_vti_kernels.h): the
       * interior loops just above never visit a cell within FD_PAD of a
       * non-PML face, so the NEXT time step's FUNC(fill_elastic_stress_bc_
       * adjoint) call (at the top of this loop) has nothing genuine to
       * fold there without this. sigmazz/sigmayy/sigmaxx each extend on
       * only their own axis (their forward fill never touches the other
       * two -- FUNC(fill_p_bc_line)/FUNC(fill_p_bc_line_adjoint) is called
       * once per own-axis face only), so those three loops sweep a single
       * axis's new cells at the other axes' ordinary (unextended)
       * interior/PML positions; sigmaxy/sigmaxz/sigmayz each extend on
       * both of their own two axes (mirrors the velocity guard block
       * above). Single-threaded for the same corner-race reason as that
       * block. */
      if (has_bc) {
#if DW_NDIM >= 3
        {
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], false);
          int64_t z, y, x;
          for (z = rz.target_lo; z < rz.target_hi; ++z) {
            if (!FUNC(dw_in_target)(rz, z) || FUNC(dw_in_std)(rz, z)) continue;
            for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                SET_I;
                FUNC(backward_sigmazz_z_cell_guard)(buoyancy_z_shot, vz_shot,
                                                    sigmazz_shot, dt, i, i, nz,
                                                    ny, nx, z, rdz);
              }
          }
        }
#endif
#if DW_NDIM >= 2
        {
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], false);
#if DW_NDIM >= 3
          int64_t z;
#endif
          int64_t y, x;
          for (y = ry.target_lo; y < ry.target_hi; ++y) {
            if (!FUNC(dw_in_target)(ry, y) || FUNC(dw_in_std)(ry, y)) continue;
#if DW_NDIM >= 3
            for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                SET_I;
                FUNC(backward_sigmayy_y_cell_guard)(buoyancy_y_shot, vy_shot,
                                                    sigmayy_shot, dt, i, i, ny,
                                                    nx, y, rdy);
              }
          }
        }
#endif
        {
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], false);
#if DW_NDIM >= 3
          int64_t z;
#endif
#if DW_NDIM >= 2
          int64_t y;
#endif
          int64_t x;
          for (x = rx.target_lo; x < rx.target_hi; ++x) {
            if (!FUNC(dw_in_target)(rx, x) || FUNC(dw_in_std)(rx, x)) continue;
#if DW_NDIM >= 3
            for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
#endif
              {
                SET_I;
                FUNC(backward_sigmaxx_x_cell_guard)(buoyancy_x_shot, vx_shot,
                                                    sigmaxx_shot, dt, i, i,
#if DW_NDIM >= 3
                                                    ny,
#endif
                                                    nx, x, rdx);
              }
          }
        }
#if DW_NDIM >= 2
        {
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], true);
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], true);
#if DW_NDIM >= 3
          int64_t z;
#endif
          int64_t y, x;
#if DW_NDIM >= 3
          for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
            for (y = ry.target_lo; y < ry.target_hi; ++y)
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                bool const in_target =
                    FUNC(dw_in_target)(ry, y) && FUNC(dw_in_target)(rx, x);
                bool const all_std =
                    FUNC(dw_in_std)(ry, y) && FUNC(dw_in_std)(rx, x);
                if (!in_target || all_std) continue;
                SET_I;
                /* Guard cells have no forward-pass PML state -- see the
                 * module comment above the vz guard-band block. */
                bool const is_pml_y = false;
                bool const is_pml_x = false;
                FUNC(backward_sigmaxy_cell_guard)(
                    c_xy_shot, buoyancy_x_shot, buoyancy_y_shot, vx_shot,
                    vy_shot, sigmaxy_shot, m_vxy_shot, m_vyx_shot, m_sigmaxyy_t,
                    m_sigmaxyx_t, dt, i, i,
#if DW_NDIM >= 3
                    nz, z,
#endif
                    ny, nx, y, x, is_pml_y, is_pml_x, ay, ayh, by, byh, ax, axh,
                    bx, bxh, rdy, rdx);
              }
        }
#endif
#if DW_NDIM >= 3
        {
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], true);
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], true);
          int64_t z, y, x;
          for (z = rz.target_lo; z < rz.target_hi; ++z)
            for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                bool const in_target =
                    FUNC(dw_in_target)(rz, z) && FUNC(dw_in_target)(rx, x);
                bool const all_std =
                    FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(rx, x);
                if (!in_target || all_std) continue;
                SET_I;
                /* Guard cells have no forward-pass PML state -- see the
                 * module comment above the vz guard-band block. */
                bool const is_pml_z = false;
                bool const is_pml_x = false;
                FUNC(backward_sigmaxz_cell_guard)(
                    c55_zx_shot, buoyancy_x_shot, buoyancy_z_shot, vx_shot,
                    vz_shot, sigmaxz_shot, m_vxz_shot, m_vzx_shot, m_sigmaxzz_t,
                    m_sigmaxzx_t, dt, i, i, nz, ny, nx, z, y, x, is_pml_z,
                    is_pml_x, az, azh, bz, bzh, ax, axh, bx, bxh, rdz, rdx);
              }
        }
        {
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], true);
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], true);
          int64_t z, y, x;
          for (z = rz.target_lo; z < rz.target_hi; ++z)
            for (y = ry.target_lo; y < ry.target_hi; ++y)
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                bool const in_target =
                    FUNC(dw_in_target)(rz, z) && FUNC(dw_in_target)(ry, y);
                bool const all_std =
                    FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(ry, y);
                if (!in_target || all_std) continue;
                SET_I;
                /* Guard cells have no forward-pass PML state -- see the
                 * module comment above the vz guard-band block. */
                bool const is_pml_z = false;
                bool const is_pml_y = false;
                FUNC(backward_sigmayz_cell_guard)(
                    c55_zy_shot, buoyancy_y_shot, buoyancy_z_shot, vy_shot,
                    vz_shot, sigmayz_shot, m_vyz_shot, m_vzy_shot, m_sigmayzz_t,
                    m_sigmayzy_t, dt, i, i, nz, ny, nx, z, y, x, is_pml_z,
                    is_pml_y, az, azh, bz, bzh, ay, ayh, by, byh, rdz, rdy);
              }
        }
#endif
      }

#if DW_NDIM >= 3
      if (n_receivers_z_per_shot > 0) {
        add_to_wavefield(vz_shot, receivers_z_i_shot,
                         grad_r_z_shot + t * n_shots * n_receivers_z_per_shot,
                         n_receivers_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_y_per_shot > 0) {
        add_to_wavefield(vy_shot, receivers_y_i_shot,
                         grad_r_y_shot + t * n_shots * n_receivers_y_per_shot,
                         n_receivers_y_per_shot);
      }
#endif
      if (n_receivers_x_per_shot > 0) {
        add_to_wavefield(vx_shot, receivers_x_i_shot,
                         grad_r_x_shot + t * n_shots * n_receivers_x_per_shot,
                         n_receivers_x_per_shot);
      }
      if (n_receivers_p_per_shot > 0) {
        add_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, receivers_p_i_shot,
            grad_r_p_shot + t * n_shots * n_receivers_p_per_shot,
            n_receivers_p_per_shot);
      }
    }
  storage_cleanup:
#if DW_NDIM >= 3
    CLOSE_FILE(dvzdbuoyancy)
    CLOSE_FILE(dvzdz)
    CLOSE_FILE(dvzdx_plus_dvxdz)
    CLOSE_FILE(dvzdy_plus_dvydz)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(dvydbuoyancy)
    CLOSE_FILE(dvydy)
    CLOSE_FILE(dvydx_plus_dvxdy)
#endif
    CLOSE_FILE(dvxdbuoyancy)
    CLOSE_FILE(dvxdx)
    if (storage_err) return 1;
  }
  return 0;
}

/*
 * Pseudo-acoustic VTI (vertically transverse isotropic, V_S = 0) wave
 * equation propagator
 */

/*
 * This file contains the C implementation of the pseudo-acoustic VTI wave
 * equation propagator (Duveneck et al. 2008; Alkhalifah's V_S = 0 limit of
 * VTI elasticity). It is templated from the elastic VTI propagator
 * (elastic_vti.c, which itself derives from elastic.c) by REDUCTION to the
 * pseudo-acoustic field set: the staggered grid, C-PML, improved-vacuum free
 * surface, source/receiver handling, and storage machinery are UNCHANGED;
 * only the constitutive relation and the model/wavefield set are reduced
 * (all shear stresses -- identically zero when c55 = c66 = 0 -- and their
 * PML memories are dropped, and the two (3-D: three) normal stresses
 * collapse onto the two surviving pseudo-acoustic stresses sigma_h/sigma_v).
 * The full derivation, including the exact elastic_vti reduction map used to
 * derive this file and its adjoint, is in
 * ``docs/acoustic_vti_derivation.md``; this header
 * reproduces the load-bearing formulas (its Sections 3-5) and cites it for
 * the rest. It is compiled multiple times with different options to
 * generate a set of functions that can be called from Python. The options
 * are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 2-3
 *    (there is no 1-D form -- with a single spatial dimension only c33
 *    survives, i.e. the 1-D scalar/acoustic equation).
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * The propagator solves the pseudo-acoustic VTI wave equation using a
 * velocity-stress formulation on a staggered grid. The free surface
 * boundary condition is implemented using the improved vacuum method of
 * Zeng et al. (2012), and the PML implementation is based on the C-PML
 * method of Komatitsch and Martin (2007) -- identical machinery to
 * elastic.c/elastic_vti.c.
 *
 * CONTINUOUS SYSTEM (derivation doc Sec. 1). Symmetry axis = slowest
 * spatial dimension (z in 3-D, y in 2-D; "z" below denotes the axis
 * dimension, "x, y" the transverse ones -- drop y in 2-D). With
 * b = buoyancy = 1/rho:
 *
 *     d(sigma_h)/dt = c11 (dvx/dx + dvy/dy) + c13 dvz/dz + s_h(t)
 *     d(sigma_v)/dt = c13 (dvx/dx + dvy/dy) + c33 dvz/dz + s_v(t)
 *     d(vx)/dt      = b d(sigma_h)/dx
 *     d(vy)/dt      = b d(sigma_h)/dy
 *     d(vz)/dt      = b d(sigma_v)/dz
 *
 * sigma_h is the horizontal normal stress (equal in every transverse
 * direction: sigma_h = sigmaxx = sigmayy in 3-D, sigma_h = sigmaxx in 2-D);
 * sigma_v is the normal stress along the symmetry axis (sigma_v = sigmazz
 * in 3-D, sigma_v = sigmayy in 2-D). c11, c13, c33 are the true
 * (density-normalised-free) VTI stiffnesses, identical to elastic_vti's:
 * c33 = rho vp^2, c11 = c33 (1 + 2 eps), c13 = c33 sqrt(1 + 2 delta).
 * Unlike elastic_vti each velocity component is fed by exactly ONE stress
 * derivative (no shear cross-terms survive): dvz/dz couples only to
 * sigma_v, dvx/dx and dvy/dy only to sigma_h -- so, unlike elastic(_vti)'s
 * velocity update, no C-PML memory is shared across multiple stress-derivative
 * terms per velocity component.
 *
 * DISCRETE FORWARD SCHEME (derivation doc Sec. 3). Inherited wholesale from
 * elastic_vti: Virieux staggering (velocities at half-integer nodes in their
 * own dimension, sigma_h/sigma_v and c11/c13/c33 at integer nodes, buoyancy
 * averaged onto each velocity's staggered node exactly as in elastic),
 * leapfrog time stepping, C-PML, IVM free surface. With d_i the
 * (PML-corrected) dv_i/dx_i and the axis as dimension 0:
 *
 *   3-D: sigma_v += dt (c33 d_z + c13 (d_x + d_y))
 *        sigma_h += dt (c11 (d_x + d_y) + c13 d_z)
 *   2-D (axis = y): sigma_v += dt (c33 d_y + c13 d_x)
 *                   sigma_h += dt (c11 d_x + c13 d_y)
 *
 * This is the c66 = 0 (=> c12 = c11) reduction of elastic_vti's normal-stress
 * update with sigmaxx, sigmayy identified as the single sigma_h and sigmazz
 * as sigma_v (derivation doc Sec. 2); the isotropic elastic propagator's
 * pseudo-acoustic P-wave limit is recovered at c11 = c33 = c13 (elliptical).
 *
 * DISCRETE ADJOINT (derivation doc Sec. 4) -- implemented in this file's
 * backward() (task 7.H milestone H2; audited line-by-line against the
 * derivation doc by the task 7.J adversarial review, 2026-07-12). The
 * skeleton (time reversal, PML transpose, storage/decimation) is inherited
 * from elastic_vti verbatim; only the constitutive couplings reduce. With
 * s_h, s_v the adjoint stresses (after the time-t adjoint update, matching
 * elastic_vti.c's convention for which time level correlates) and the
 * symmetric stiffness matrix's transpose reusing the same Cij:
 *
 *     to dvx/dx and dvy/dy:  c11 s_h + c13 s_v
 *     to dvz/dz:             c13 s_h + c33 s_v
 *   (2-D: to dvx/dx: c11 s_h + c13 s_v; to dvy/dy [axis]: c13 s_h + c33 s_v.)
 *
 * The adjoint stress update transposes the velocity update with the same
 * staggered stencils transposed (identical structure to elastic's); the
 * THREE-DIMENSIONAL HAZARD (derivation doc Sec. 2/7a) is that sigma_h's
 * adjoint accumulates the SUM of what the full elastic_vti system spreads
 * over sigmaxx and sigmayy (s_h = s_xx + s_yy), not either alone.
 *
 * MODEL GRADIENTS (derivation doc Sec. 5) -- implemented in backward(). With
 * S_a := dt * (PML-corrected forward dv_a/da at the stress node, as stored
 * by forward for the backward pass) and s_h, s_v the adjoint stresses:
 *
 *     grad_c11 = sum_t  s_h * (S_x + S_y)          [2-D: s_h * S_x]
 *     grad_c13 = sum_t  (s_v * (S_x + S_y) + s_h * S_z)
 *                                                  [2-D: s_v S_x + s_h S_y]
 *     grad_c33 = sum_t  s_v * S_z                  [2-D: s_v * S_y]
 *     grad_b   : identical in structure to elastic's buoyancy gradient.
 *
 * Every accumulation above must be gated on ``(t % step_ratio) == 0`` AND
 * multiplied by ``(DW_DTYPE)step_ratio`` -- the pattern elastic.c implements
 * and that elastic_born.c once omitted (a real correctness bug, since
 * fixed -- see elastic_born.c's SETUP_STORE_LOAD-adjacent gradient
 * accumulation for the corrected pattern); H2 must verify at step_ratio in
 * {1, 2, 3}.
 *
 * See docs/acoustic_vti_derivation.md for the full derivation, the oracle
 * test protocol, and the hazard checklist (Sec. 6-7) that both this file and
 * its CUDA twin (acoustic_vti.cu) must satisfy.
 *
 * The code is structured to maximise performance by using macros to
 * generate the code for each of the regions of the computational domain (a
 * central region, side regions, edge regions, and corner regions), and by
 * using OpenMP to parallelise the loops over shots.
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
  acoustic_vti_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
/* Multiaxial-PML (M-PML) build variant -- see elastic.c for the rationale.
 * When DW_MPML is defined the exported forward/backward symbols gain an
 * ``_mpml`` suffix and the PML profiles are read as full-grid arrays;
 * otherwise the default per-axis 1-D profile behaviour is compiled
 * unchanged. Kept in lockstep with elastic_vti.c. */
#ifdef DW_MPML
#define ACOUSTIC_VTI_VARIANT_I(name) name##_mpml
#define ACOUSTIC_VTI_VARIANT(name) ACOUSTIC_VTI_VARIANT_I(name)
#else
#define ACOUSTIC_VTI_VARIANT(name) name
#endif
#define FUNC(name) \
  CAT(ACOUSTIC_VTI_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* PML-region skip test -- identical to elastic_vti.c's DW_PML_SKIP. Driver-
 * side scaffolding (depends on the outer region-loop indices pml_z/pml_y/
 * pml_x below), so it is NOT part of acoustic_vti_kernels.h -- see that
 * header's top-of-file note. */
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

/* ND_INDEX, the M-PML-or-1-D PML profile accessors P_AZ/P_BZ/.../P_AXH/
 * P_BXH, the field-access macros VZ/VY/VX/SIGMA_V/SIGMA_H, the model-access
 * macros C11/C13/C33/BUOYANCY_Z/Y/X, the M_VZZ/M_VYY/M_VXX/M_SIGMA_V_Z/
 * M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X PML-memory families, the adjoint-term
 * macros VZ_Z/VY_Y/VX_X/SIGMA_V_Z/SIGMA_H_Y/SIGMA_V_Y/SIGMA_H_X (+_PML), and
 * the FUNC(forward_vz_cell)/.../FUNC(backward_sigma_cell) per-grid-point
 * stencil update functions live in acoustic_vti_kernels.h, shared verbatim
 * with acoustic_vti.cu (see that header for the full explanation).
 * vz_shot/vy_shot/.../m_sigma_v_z_t/.../c11_shot/buoyancy_z_shot/... below
 * are this file's names for the (per-shot-offset, and for the ping-ponged
 * backward-only PML memories, per-reversed-timestep) arrays/values passed
 * as the vz/vy/.../m_sigma_v_z/.../c11_shot/buoyancy_z_shot/... parameters
 * of those functions; on this (CPU) driver the model arrays' `j` parameter
 * is always passed the same value as the wavefields' `i` parameter (see the
 * header's addressing-convention note). */
#include "acoustic_vti_kernels.h"

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

/* First-class reflecting/periodic boundary faces on the staggered acoustic_vti
 * grid (see docs/boundaries.rst). fill_p_bc_line/fill_v_bc_line/
 * fill_p_bc/fill_v_bc are ported verbatim from acoustic.c's (and
 * viscoacoustic.c's) copies -- the pressure/velocity image rule is identical
 * (the same staggered pressure-velocity layout; acoustic_vti's velocity
 * equation is byte-identical in form to acoustic's, one term per axis). ``bc``
 * codes: 0=pml (untouched), 1=free (pressure release), 2=rigid (hard wall),
 * 3=periodic.
 *
 * acoustic_vti has TWO "pressure-like" fields (sigma_v, sigma_h) instead of
 * acoustic's one. Per the CONTINUOUS SYSTEM equations in this file's header,
 * sigma_v (the axial normal stress) is fed only by the axis-derivative term
 * and feeds only the axis velocity back (FUNC(forward_vz_cell) reads only
 * sigma_v; FUNC(forward_vy_cell)'s 2-D axis branch likewise); sigma_h (the
 * transverse normal stress) is fed only by the transverse-derivative terms
 * and feeds only the transverse velocities back (FUNC(forward_vx_cell) and
 * FUNC(forward_vy_cell)'s 3-D transverse branch read only sigma_h). So
 * sigma_v's guard ring is never read by the forward kernel outside the axis
 * face, and sigma_h's is never read outside the transverse faces -- each is
 * therefore guard-filled with FUNC(fill_p_bc) using a bc array with the
 * OTHER field's faces forced to 0 (``bc_axis``/``bc_transverse`` below),
 * reusing FUNC(fill_p_bc) unchanged rather than deriving a new rule (mirrors
 * fill_acoustic_pressure_boundary's identical split in acoustic_vti.py --
 * see that module's ``_acoustic_vti_axis_boundary``/
 * ``_acoustic_vti_transverse_boundary`` for the full citation). This also
 * makes the two stresses' guard rings exactly zero everywhere the OTHER
 * field's ghost values would matter to the compiled adjoint (see
 * _boundaries.mirror_pad_acoustic_vti_models's docstring for why c13 then
 * needs no model mirror-pad).
 *
 * The velocity fill is unchanged from acoustic (each v_d imaged only along
 * its own axis d, via FUNC(fill_v_bc) applied to the full ``bc`` array). */

/* FUNC(fill_p_bc_line)/FUNC(fill_v_bc_line) live in line_bc.h, shared
 * verbatim across every propagator on this staggered layout -- see that
 * header for the full explanation. */
#include "line_bc.h"

/* FUNC(fill_p_bc) and FUNC(fill_v_bc) below are deliberate byte-identical
 * clones of staggered_bc.h's functions of the same names, not an oversight.
 * They are kept local because this propagator's adjoint boundary strategy is a
 * pair of direct transposes -- FUNC(fill_p_bc_adjoint) and
 * FUNC(fill_v_bc_adjoint) below, which undo the faces in the reverse of the
 * order these apply them -- rather than the fold/gather family staggered_bc.h
 * carries, and a transpose belongs on the same screen as the function it
 * transposes: that adjacency is what lets the pairing be checked by eye. The
 * cost of the duplication is paid instead by scripts/check_clone_drift.py,
 * whose manifest pins both blocks byte-identical to the shared header's, so
 * neither copy can be edited alone. The DW_NDIM == 1 branches are never
 * compiled here (this propagator is 2-D and 3-D only); they are present so the
 * blocks match exactly. */

/* Apply the pressure guard fill on every reflecting axis (``bc`` is the
 * face-order array [z0,z1,y0,y1,x0,x1]; only the trailing DW_NDIM pairs are
 * used). */
static void FUNC(fill_p_bc)(DW_DTYPE* __restrict const p,
                            int64_t const* __restrict const bc,
                            int64_t const nz_, int64_t const ny_,
                            int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_p_bc_line)(p, (zz * ny_ + yy) * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_p_bc_line)(p, zz * ny_ * nx_ + xx, ny_, nx_, bc[2], bc[3]);
    if (bc[0] != DW_BC_PML || bc[1] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_p_bc_line)(p, yy * nx_ + xx, nz_, ny_ * nx_, bc[0], bc[1]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_p_bc_line)(p, yy * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_p_bc_line)(p, xx, ny_, nx_, bc[2], bc[3]);
  }
#else
  FUNC(fill_p_bc_line)(p, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* Apply the boundary-normal velocity guard fill: only ``v[d]`` is imaged along
 * axis ``d`` (tangential velocities and corner cells are never read). */
static void FUNC(fill_v_bc)(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_v_bc_line)(vx, (zz * ny_ + yy) * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_v_bc_line)(vy, zz * ny_ * nx_ + xx, ny_, nx_, bc[2], bc[3]);
    if (bc[0] != DW_BC_PML || bc[1] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_v_bc_line)(vz, yy * nx_ + xx, nz_, ny_ * nx_, bc[0], bc[1]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_v_bc_line)(vx, yy * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_v_bc_line)(vy, xx, ny_, nx_, bc[2], bc[3]);
  }
#else
  FUNC(fill_v_bc_line)(vx, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* bc_axis: only the axis face's codes survive (dim 0 -- bc[0..1] in 3-D,
 * bc[2..3] in 2-D); bc_transverse: only the transverse faces' codes survive.
 * Built once per bc[] (see the forward()/backward() bodies below). */
#if DW_NDIM == 3
#define MAKE_BC_AXIS_TRANSVERSE()                        \
  int64_t const bc_axis[6] = {bc[0], bc[1], 0, 0, 0, 0}; \
  int64_t const bc_transverse[6] = {0, 0, bc[2], bc[3], bc[4], bc[5]}
#define FILL_SIGMA_V_BC() FUNC(fill_p_bc)(sigma_v_shot, bc_axis, nz, ny, nx)
#define FILL_SIGMA_H_BC() \
  FUNC(fill_p_bc)(sigma_h_shot, bc_transverse, nz, ny, nx)
#define FILL_V_BC() FUNC(fill_v_bc)(vz_shot, vy_shot, vx_shot, bc, nz, ny, nx)
#elif DW_NDIM == 2
#define MAKE_BC_AXIS_TRANSVERSE()                        \
  int64_t const bc_axis[6] = {0, 0, bc[2], bc[3], 0, 0}; \
  int64_t const bc_transverse[6] = {0, 0, 0, 0, bc[4], bc[5]}
#define FILL_SIGMA_V_BC() FUNC(fill_p_bc)(sigma_v_shot, bc_axis, 1, ny, nx)
#define FILL_SIGMA_H_BC() \
  FUNC(fill_p_bc)(sigma_h_shot, bc_transverse, 1, ny, nx)
#define FILL_V_BC() FUNC(fill_v_bc)(vy_shot, vx_shot, bc, 1, ny, nx)
#endif

/* --- Adjoint (transpose) of the guard-ring image fills above -------------
 *
 * The compiled backward folds these _adjoint fills into the adjoint
 * (cotangent) sigma_v/sigma_h/velocity fields in place of reapplying the
 * forward fills above. FUNC(fill_p_bc_line) is byte-identical to
 * elastic.c's/elastic_vti.c's own sigma_AA fill (node-centred odd image
 * with an on-surface zero for free, even image for rigid, wrap for
 * periodic), so FUNC(fill_p_bc_line_adjoint) below is that already-solved
 * transpose, ported unchanged (see elastic_vti.c's module comment above its
 * own FUNC(fill_p_bc_line_adjoint) for the full mechanical-derivation
 * writeup: `dest = sign * src` -> `adj[src] += sign * adj[dest]`, `dest = 0`
 * -> no accumulation, then zero every cell the forward fill wrote, all done
 * in the reverse of the forward assignment order since the periodic
 * high-side slave read in step 1 happens before the ghost wraps in steps
 * 2-3).
 *
 * FUNC(fill_v_bc_line), by contrast, is NOT a port of elastic's onrow/
 * offrow velocity fill: acoustic_vti's velocity fill mirrors acoustic's own
 * (see the module comment above FUNC(fill_p_bc_line)/FUNC(fill_v_bc_line)),
 * whose free branch MIRRORS the ghost (even image, matching acoustic's
 * Neumann-style free-surface velocity condition) rather than zeroing it as
 * elastic's off-row velocity fill does -- so FUNC(fill_v_bc_line_adjoint)
 * is re-derived here from FUNC(fill_v_bc_line)'s actual assignments (by the
 * same mechanical rule), not copied from elastic(_vti). Verified by
 * building the explicit per-(bc_lo, bc_hi) transfer matrix for both
 * functions and checking it equals the forward matrix's transpose to
 * machine precision (every code combination, several grid sizes).
 *
 * Two per-line ordering facts (both already true of the forward
 * function, and preserved by processing steps in reverse):
 *   - FUNC(fill_v_bc_line)'s own interior high-face node (bc_hi==2 zero /
 *     bc_hi==3 periodic slave, index n-P) is written FIRST, before the
 *     ghost-cell loops that (for bc_hi==3) read node P -- so a periodic
 *     high face's ghost wraps never read the pre-slave value of node P by
 *     construction. The adjoint therefore accumulates the ghost-loop
 *     contributions before touching node n-P.
 *   - Under free (bc_hi==1 / bc_lo==1) the interior high node n-P (resp.
 *     low node -- there is no separate low interior node here) is NEVER
 *     written by the forward fill at all -- it is a live, independent DOF
 *     -- so the adjoint's final zero of node n-P is gated on
 *     `bc_hi == 2 || bc_hi == 3` only, not on `bc_hi != 0` (unlike
 *     FUNC(fill_p_bc_line_adjoint)'s on-surface node, which the forward
 *     zeros under free too).
 *
 * Faces are undone in the reverse of FUNC(fill_p_bc)'s/FUNC(fill_v_bc)'s own
 * face order in the dispatchers below, for the same corner-overwrite reason
 * documented in elastic_vti.c -- though for FUNC(fill_v_bc), each velocity
 * component is imaged on exactly one axis (see the module comment above
 * FUNC(fill_v_bc)), so no two faces ever write the same cell of the same
 * component and the face order is immaterial there; it is kept reversed
 * anyway for uniformity with FUNC(fill_p_bc_adjoint), where sigma_h's two
 * transverse faces (3-D) really do share corner ghosts.
 */

/* FUNC(fill_p_bc_line_adjoint) lives in elastic_line_bc.h, shared with
 * elastic.c/elastic_born.c/elastic_vti.c (whose sigma_AA fill is
 * byte-identical). */
#include "elastic_line_bc.h"

/* Transpose of FUNC(fill_v_bc_line) -- re-derived, not ported (see the
 * module comment above); mind the free-branch zero-gate difference from
 * FUNC(fill_p_bc_line_adjoint) noted there. */
static inline void FUNC(fill_v_bc_line_adjoint)(
    DW_DTYPE* __restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j;
      } else if (bc_hi == 1) {
        src = P + m - 3 - j;
      } else {
        src = P + m - 2 - j;
        sign = -1;
      }
      v[base + src * stride] += sign * v[base + xh * stride];
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        v[base + (m - 1 + g) * stride] += v[base + g * stride];
      } else if (bc_lo == 1) {
        v[base + (2 * P - 1 - g) * stride] += v[base + g * stride];
      } else if (g != P - 1) {
        /* rigid, off the on-plane guard cell */
        v[base + (2 * P - 2 - g) * stride] += -v[base + g * stride];
      }
      /* rigid, g == P - 1: the forward ghost was a hard zero -- no
       * source to fold. */
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      v[base + (nax - P + 1 + j) * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) v[base + g * stride] = 0;
  }
  if (bc_hi == 3) {
    v[base + P * stride] += v[base + (nax - P) * stride];
  }
  if (bc_hi == 2 || bc_hi == 3) v[base + (nax - P) * stride] = 0;
}

/* Transpose of FUNC(fill_p_bc): faces undone in the reverse of that
 * function's own face order. */
static void FUNC(fill_p_bc_adjoint)(DW_DTYPE* __restrict const p,
                                    int64_t const* __restrict const bc,
                                    int64_t const nz_, int64_t const ny_,
                                    int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[0] != 0 || bc[1] != 0)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_p_bc_line_adjoint)(p, yy * nx_ + xx, nz_, ny_ * nx_, bc[0],
                                       bc[1]);
    if (bc[2] != 0 || bc[3] != 0)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_p_bc_line_adjoint)(p, zz * ny_ * nx_ + xx, ny_, nx_, bc[2],
                                       bc[3]);
    if (bc[4] != 0 || bc[5] != 0)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_p_bc_line_adjoint)(p, (zz * ny_ + yy) * nx_, nx_, 1, bc[4],
                                       bc[5]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[2] != 0 || bc[3] != 0)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_p_bc_line_adjoint)(p, xx, ny_, nx_, bc[2], bc[3]);
    if (bc[4] != 0 || bc[5] != 0)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_p_bc_line_adjoint)(p, yy * nx_, nx_, 1, bc[4], bc[5]);
  }
#endif
}

/* Transpose of FUNC(fill_v_bc): faces undone in the reverse of that
 * function's own face order (immaterial here -- see the module comment
 * above -- but kept for uniformity). */
static void FUNC(fill_v_bc_adjoint)(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vx,
    int64_t const* __restrict const bc, int64_t const nz_, int64_t const ny_,
    int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[0] != 0 || bc[1] != 0)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_v_bc_line_adjoint)(vz, yy * nx_ + xx, nz_, ny_ * nx_, bc[0],
                                       bc[1]);
    if (bc[2] != 0 || bc[3] != 0)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_v_bc_line_adjoint)(vy, zz * ny_ * nx_ + xx, ny_, nx_, bc[2],
                                       bc[3]);
    if (bc[4] != 0 || bc[5] != 0)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_v_bc_line_adjoint)(vx, (zz * ny_ + yy) * nx_, nx_, 1, bc[4],
                                       bc[5]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[2] != 0 || bc[3] != 0)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_v_bc_line_adjoint)(vy, xx, ny_, nx_, bc[2], bc[3]);
    if (bc[4] != 0 || bc[5] != 0)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_v_bc_line_adjoint)(vx, yy * nx_, nx_, 1, bc[4], bc[5]);
  }
#endif
}

#if DW_NDIM == 3
#define FILL_SIGMA_V_BC_ADJOINT() \
  FUNC(fill_p_bc_adjoint)(sigma_v_shot, bc_axis, nz, ny, nx)
#define FILL_SIGMA_H_BC_ADJOINT() \
  FUNC(fill_p_bc_adjoint)(sigma_h_shot, bc_transverse, nz, ny, nx)
#define FILL_V_BC_ADJOINT() \
  FUNC(fill_v_bc_adjoint)(vz_shot, vy_shot, vx_shot, bc, nz, ny, nx)
#elif DW_NDIM == 2
#define FILL_SIGMA_V_BC_ADJOINT() \
  FUNC(fill_p_bc_adjoint)(sigma_v_shot, bc_axis, 1, ny, nx)
#define FILL_SIGMA_H_BC_ADJOINT() \
  FUNC(fill_p_bc_adjoint)(sigma_h_shot, bc_transverse, 1, ny, nx)
#define FILL_V_BC_ADJOINT() \
  FUNC(fill_v_bc_adjoint)(vy_shot, vx_shot, bc, 1, ny, nx)
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(forward)(
    DW_DTYPE const* __restrict const c11, DW_DTYPE const* __restrict const c13,
    DW_DTYPE const* __restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const f_z,
#endif
    DW_DTYPE const* __restrict const f_y, DW_DTYPE const* __restrict const f_x,
    DW_DTYPE const* __restrict const f_sigma_h,
    DW_DTYPE const* __restrict const f_sigma_v,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vx,
    DW_DTYPE* __restrict const sigma_v, DW_DTYPE* __restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz,
#endif
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vxx,
#if DW_NDIM == 3
    DW_DTYPE* __restrict const m_sigma_v_z,
    DW_DTYPE* __restrict const m_sigma_h_y,
#else
    DW_DTYPE* __restrict const m_sigma_v_y,
#endif
    DW_DTYPE* __restrict const m_sigma_h_x,
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
#endif
    DW_DTYPE* __restrict const dvydbuoyancy_store_1,
    DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
    void* __restrict const dvydbuoyancy_store_2,
    void* __restrict const dvydbuoyancy_store_3,
    char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvydy_store_1,
    DW_DTYPE* __restrict const dvydy_store_1b,
    void* __restrict const dvydy_store_2, void* __restrict const dvydy_store_3,
    char const* __restrict const* __restrict const dvydy_filenames_ptr,
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
    DW_DTYPE* __restrict const r_y, DW_DTYPE* __restrict const r_x,
    DW_DTYPE* __restrict const r_sigma_h, DW_DTYPE* __restrict const r_sigma_v,
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
    int64_t const* __restrict const sources_y_i,
    int64_t const* __restrict const sources_x_i,
    int64_t const* __restrict const sources_sigma_h_i,
    int64_t const* __restrict const sources_sigma_v_i,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_z_i,
#endif
    int64_t const* __restrict const receivers_y_i,
    int64_t const* __restrict const receivers_x_i,
    int64_t const* __restrict const receivers_sigma_h_i,
    int64_t const* __restrict const receivers_sigma_v_i,
#if DW_NDIM >= 3
    DW_DTYPE const rdz,
#endif
    DW_DTYPE const rdy, DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
    int64_t const n_shots,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
    int64_t const ny, int64_t const nx,
#if DW_NDIM >= 3
    int64_t const n_sources_z_per_shot,
#endif
    int64_t const n_sources_y_per_shot, int64_t const n_sources_x_per_shot,
    int64_t const n_sources_sigma_h_per_shot,
    int64_t const n_sources_sigma_v_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_z_per_shot,
#endif
    int64_t n_receivers_y_per_shot, int64_t n_receivers_x_per_shot,
    int64_t const n_receivers_sigma_h_per_shot,
    int64_t const n_receivers_sigma_v_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const c11_requires_grad,
    bool const c13_requires_grad, bool const c33_requires_grad,
    bool const buoyancy_requires_grad, bool const c11_batched,
    bool const c13_batched, bool const c33_batched, bool const buoyancy_batched,
    bool const storage_compression, int64_t start_t,
#if DW_NDIM >= 3
    int64_t const pml_z0,
#endif
    int64_t const pml_y0, int64_t const pml_x0,
#if DW_NDIM >= 3
    int64_t const pml_z1,
#endif
    int64_t const pml_y1, int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
    int64_t const bc_y0, int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
    int64_t const bc_y1, int64_t const bc_x1, int64_t const n_threads,
    void* unused) {
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
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_sigma_h_per_shot > 0) && (receivers_sigma_h_i == NULL))
      return 1;
    if ((n_receivers_sigma_v_per_shot > 0) && (receivers_sigma_v_i == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0) && sources_z_i == NULL) return 1;
#endif
    if ((n_sources_y_per_shot > 0) && sources_y_i == NULL) return 1;
    if ((n_sources_x_per_shot > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_sigma_h_per_shot > 0) && sources_sigma_h_i == NULL) return 1;
    if ((n_sources_sigma_v_per_shot > 0) && sources_sigma_v_i == NULL) return 1;
  }
#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_z0 || bc_z1;
  int const has_bc_transverse = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_y0 || bc_y1;
  int const has_bc_transverse = bc_x0 || bc_x1;
#endif
  int const has_bc = has_bc_axis || has_bc_transverse;
  MAKE_BC_AXIS_TRANSVERSE();
  int64_t shot;
  /* Intra-shot spatial parallelism: shots run serially and each time step's
   * velocity and sigma_v/sigma_h stencils are spread across threads (see the
   * omp parallel regions below), so a single shot saturates all cores. */
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#else
    int64_t const n_grid_points = ny * nx;
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
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        buoyancy_batched ? buoyancy_z + si : buoyancy_z;
    DW_DTYPE* __restrict const vz_shot = vz + si;
    DW_DTYPE* __restrict const m_vzz_shot = m_vzz + si;
    DW_DTYPE* __restrict const m_sigma_v_z_shot = m_sigma_v_z + si;
    DW_DTYPE* __restrict const m_sigma_h_y_shot = m_sigma_h_y + si;
#else
    DW_DTYPE* __restrict const m_sigma_v_y_shot = m_sigma_v_y + si;
#endif
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const* __restrict const sources_y_i_shot =
        sources_y_i != NULL ? (sources_y_i + siy) : NULL;
    int64_t const* __restrict const receivers_y_i_shot =
        receivers_y_i != NULL ? (receivers_y_i + riy) : NULL;
    DW_DTYPE const* __restrict const f_y_shot =
        f_y != NULL ? (f_y + siy) : NULL;
    DW_DTYPE* __restrict const r_y_shot = r_y != NULL ? (r_y + riy) : NULL;
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE* __restrict const vy_shot = vy + si;
    DW_DTYPE* __restrict const m_vyy_shot = m_vyy + si;

    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const sish = shot * n_sources_sigma_h_per_shot;
    int64_t const rish = shot * n_receivers_sigma_h_per_shot;
    int64_t const sisv = shot * n_sources_sigma_v_per_shot;
    int64_t const risv = shot * n_receivers_sigma_v_per_shot;
    int64_t const* __restrict const sources_x_i_shot =
        sources_x_i != NULL ? (sources_x_i + six) : NULL;
    int64_t const* __restrict const receivers_x_i_shot =
        receivers_x_i != NULL ? (receivers_x_i + rix) : NULL;
    int64_t const* __restrict const sources_sigma_h_i_shot =
        sources_sigma_h_i != NULL ? (sources_sigma_h_i + sish) : NULL;
    int64_t const* __restrict const receivers_sigma_h_i_shot =
        receivers_sigma_h_i != NULL ? (receivers_sigma_h_i + rish) : NULL;
    int64_t const* __restrict const sources_sigma_v_i_shot =
        sources_sigma_v_i != NULL ? (sources_sigma_v_i + sisv) : NULL;
    int64_t const* __restrict const receivers_sigma_v_i_shot =
        receivers_sigma_v_i != NULL ? (receivers_sigma_v_i + risv) : NULL;
    DW_DTYPE const* __restrict const f_x_shot =
        f_x != NULL ? (f_x + six) : NULL;
    DW_DTYPE* __restrict const r_x_shot = r_x != NULL ? (r_x + rix) : NULL;
    DW_DTYPE const* __restrict const f_sigma_h_shot =
        f_sigma_h != NULL ? (f_sigma_h + sish) : NULL;
    DW_DTYPE* __restrict const r_sigma_h_shot =
        r_sigma_h != NULL ? (r_sigma_h + rish) : NULL;
    DW_DTYPE const* __restrict const f_sigma_v_shot =
        f_sigma_v != NULL ? (f_sigma_v + sisv) : NULL;
    DW_DTYPE* __restrict const r_sigma_v_shot =
        r_sigma_v != NULL ? (r_sigma_v + risv) : NULL;
    DW_DTYPE const* __restrict const c11_shot = c11_batched ? c11 + si : c11;
    DW_DTYPE const* __restrict const c13_shot = c13_batched ? c13 + si : c13;
    DW_DTYPE const* __restrict const c33_shot = c33_batched ? c33 + si : c33;
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE* __restrict const vx_shot = vx + si;
    DW_DTYPE* __restrict const sigma_v_shot = sigma_v + si;
    DW_DTYPE* __restrict const sigma_h_shot = sigma_h + si;
    DW_DTYPE* __restrict const m_vxx_shot = m_vxx + si;
    DW_DTYPE* __restrict const m_sigma_h_x_shot = m_sigma_h_x + si;
    int64_t t;

    // Aggregate requires-grad condition gating the stored velocity
    // derivatives (must match the store-array sizing in Python's
    // storage_flags): any normal-stress-node stiffness (c11/c13/c33)
    // drives ALL THREE dv?d? stores together (coarse gating, matching
    // elastic_vti's cnormal_requires_grad pattern).
    bool const cnormal_requires_grad =
        c11_requires_grad || c13_requires_grad || c33_requires_grad;
    int storage_err = 0;
#if DW_NDIM >= 3
    FILE* fp_dvzdbuoyancy = NULL;
    FILE* fp_dvzdz = NULL;
#endif
    FILE* fp_dvydbuoyancy = NULL;
    FILE* fp_dvydy = NULL;
    FILE* fp_dvxdbuoyancy = NULL;
    FILE* fp_dvxdx = NULL;

#if DW_NDIM >= 3
    OPEN_FILE_WRITE(dvzdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvzdz, cnormal_requires_grad)
#endif
    OPEN_FILE_WRITE(dvydbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvydy, cnormal_requires_grad)
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
      if ((t % step_ratio) == 0 && (buoyancy_requires_grad) &&
          (dvydbuoyancy_store_1 == NULL || dvxdbuoyancy_store_1 == NULL)) {
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
#endif
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
      bool const cnormal_requires_grad_t = cnormal_requires_grad && rg_step_t;
      bool const buoyancy_requires_grad_t = buoyancy_requires_grad && rg_step_t;

      // Refill the sigma_v/sigma_h guard rings (read by the velocity update
      // below); this also holds free-surface stress nodes to zero before
      // recording (no-op when every face is PML). sigma_v is imaged only on
      // the axis face, sigma_h only on the transverse faces (see the
      // FILL_SIGMA_V_BC/FILL_SIGMA_H_BC comment above).
      if (has_bc_axis) FILL_SIGMA_V_BC();
      if (has_bc_transverse) FILL_SIGMA_H_BC();

#if DW_NDIM >= 3
      if (n_receivers_z_per_shot > 0) {
        record_from_wavefield(vz_shot, receivers_z_i_shot,
                              r_z_shot + t * n_shots * n_receivers_z_per_shot,
                              n_receivers_z_per_shot);
      }
#endif
      if (n_receivers_y_per_shot > 0) {
        record_from_wavefield(vy_shot, receivers_y_i_shot,
                              r_y_shot + t * n_shots * n_receivers_y_per_shot,
                              n_receivers_y_per_shot);
      }
      if (n_receivers_x_per_shot > 0) {
        record_from_wavefield(vx_shot, receivers_x_i_shot,
                              r_x_shot + t * n_shots * n_receivers_x_per_shot,
                              n_receivers_x_per_shot);
      }
      if (n_receivers_sigma_h_per_shot > 0) {
        record_from_wavefield(
            sigma_h_shot, receivers_sigma_h_i_shot,
            r_sigma_h_shot + t * n_shots * n_receivers_sigma_h_per_shot,
            n_receivers_sigma_h_per_shot);
      }
      if (n_receivers_sigma_v_per_shot > 0) {
        record_from_wavefield(
            sigma_v_shot, receivers_sigma_v_i_shot,
            r_sigma_v_shot + t * n_shots * n_receivers_sigma_v_per_shot,
            n_receivers_sigma_v_per_shot);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
        int pml_y;
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y) DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
              int64_t y;
              int64_t x;

#if DW_NDIM >= 3

              // vz -- fed only by d(sigma_v)/dz (the axis derivative).

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_vz_cell)(
                        sigma_v_shot, vz_shot, m_sigma_v_z_shot,
                        dvzdbuoyancy_store_1_t, buoyancy_z_shot[i], azh, bzh,
                        dt, buoyancy_requires_grad_t, i, ny, nx, z, y, x,
                        !DW_PML_SKIP(pml_z == 1), rdz);
                  }
#endif

              // vy -- 3-D: transverse, fed only by d(sigma_h)/dy.
              //       2-D: axis, fed only by d(sigma_v)/dy.

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_vy_cell)(
#if DW_NDIM == 3
                        sigma_h_shot, m_sigma_h_y_shot,
#else
                    sigma_v_shot, m_sigma_v_y_shot,
#endif
                        vy_shot, dvydbuoyancy_store_1_t, buoyancy_y_shot[i],
                        ayh, byh, dt, buoyancy_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z,
#endif
                        y, x, !DW_PML_SKIP(pml_y == 1), rdy);
                  }

              // vx -- always transverse, fed only by d(sigma_h)/dx.

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_vx_cell)(
                        sigma_h_shot, vx_shot, m_sigma_h_x_shot,
                        dvxdbuoyancy_store_1_t, buoyancy_x_shot[i], axh, bxh,
                        dt, buoyancy_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z,
#endif
                        y, x, !DW_PML_SKIP(pml_x == 1), rdx);
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
      if (n_sources_y_per_shot > 0) {
        add_to_wavefield(vy_shot, sources_y_i_shot,
                         f_y_shot + t * n_shots * n_sources_y_per_shot,
                         n_sources_y_per_shot);
      }
      if (n_sources_x_per_shot > 0) {
        add_to_wavefield(vx_shot, sources_x_i_shot,
                         f_x_shot + t * n_shots * n_sources_x_per_shot,
                         n_sources_x_per_shot);
      }

      // Refill the boundary-normal velocity guard ring (read by the
      // sigma_v/sigma_h update below); holds rigid-wall velocity nodes to
      // zero (no-op for all-PML).
      if (has_bc) FILL_V_BC();

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
        int pml_y;
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y) DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
              int64_t y;
              int64_t x;

              // sigma_v, sigma_h -- VTI constitutive relation (c66 = 0
              // reduction of elastic_vti's normal-stress update; see the
              // header derivation).

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_sigma_cell)(
#if DW_NDIM >= 3
                        vz_shot,
#endif
                        vy_shot, vx_shot, sigma_v_shot, sigma_h_shot,
#if DW_NDIM >= 3
                        m_vzz_shot,
#endif
                        m_vyy_shot, m_vxx_shot,
#if DW_NDIM >= 3
                        dvzdz_store_1_t,
#endif
                        dvydy_store_1_t, dvxdx_store_1_t, c11_shot[i],
                        c13_shot[i], c33_shot[i], dt, cnormal_requires_grad_t,
                        i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z, !DW_PML_SKIP(pml_z == 1), az, bz, rdz,
#endif
                        y, !DW_PML_SKIP(pml_y == 1), ay, by, rdy, x,
                        !DW_PML_SKIP(pml_x == 1), ax, bx, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      if (n_sources_sigma_h_per_shot > 0) {
        add_to_wavefield(
            sigma_h_shot, sources_sigma_h_i_shot,
            f_sigma_h_shot + t * n_shots * n_sources_sigma_h_per_shot,
            n_sources_sigma_h_per_shot);
      }
      if (n_sources_sigma_v_per_shot > 0) {
        add_to_wavefield(
            sigma_v_shot, sources_sigma_v_i_shot,
            f_sigma_v_shot + t * n_shots * n_sources_sigma_v_per_shot,
            n_sources_sigma_v_per_shot);
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
#endif
      SAVE_SNAPSHOT(dvydbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvydy, cnormal_requires_grad_t)
      SAVE_SNAPSHOT(dvxdbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvxdx, cnormal_requires_grad_t)
    }

    /* Leave the returned wavefields boundary-consistent (e.g. free-surface
     * stress nodes held to zero, guard rings imaged), matching
     * _acoustic_vti_eager_core's own post-loop refill in acoustic_vti.py --
     * without this, the last time step's sigma_v/sigma_h update runs after
     * the last in-loop stress fill, so the returned guard ring/on-surface
     * zero reflect a stale (pre-update) state. Receiver traces and
     * gradients are unaffected (already recorded/consumed inside the loop
     * before this point); only the final returned wavefields change. */
    if (has_bc_axis) FILL_SIGMA_V_BC();
    if (has_bc_transverse) FILL_SIGMA_H_BC();
    if (has_bc) FILL_V_BC();

#define CLOSE_FILE(name) \
  if (dw_close_file(&fp_##name) != 0) storage_err = 1;

  storage_cleanup:
#if DW_NDIM >= 3
    CLOSE_FILE(dvzdbuoyancy)
    CLOSE_FILE(dvzdz)
#endif
    CLOSE_FILE(dvydbuoyancy)
    CLOSE_FILE(dvydy)
    CLOSE_FILE(dvxdbuoyancy)
    CLOSE_FILE(dvxdx)
    if (storage_err) return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Adjoint (task 7.H milestone H2). See the header derivation (Sec. 4-5,
// citing docs/acoustic_vti_derivation.md) for the formulas implemented below.
// ---------------------------------------------------------------------------

// The VZ_Z/VY_Y/VX_X(+_PML)/SIGMA_V_Z/SIGMA_H_Y/SIGMA_V_Y/SIGMA_H_X(+_PML)
// adjoint-access macros and the FUNC(backward_vz_cell)/vy_cell/vx_cell/
// sigma_cell per-grid-point stencil functions used below now live in
// acoustic_vti_kernels.h (included above), shared verbatim with
// acoustic_vti.cu.

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(backward)(
    DW_DTYPE const* __restrict const c11, DW_DTYPE const* __restrict const c13,
    DW_DTYPE const* __restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const grad_r_z,
#endif
    DW_DTYPE const* __restrict const grad_r_y,
    DW_DTYPE const* __restrict const grad_r_x,
    DW_DTYPE const* __restrict const grad_r_sigma_h,
    DW_DTYPE const* __restrict const grad_r_sigma_v,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vx,
    DW_DTYPE* __restrict const sigma_v, DW_DTYPE* __restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz,
#endif
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vxx,
#if DW_NDIM == 3
    DW_DTYPE* __restrict const m_sigma_v_z,
    DW_DTYPE* __restrict const m_sigma_h_y,
#else
    DW_DTYPE* __restrict const m_sigma_v_y,
#endif
    DW_DTYPE* __restrict const m_sigma_h_x,
#if DW_NDIM == 3
    DW_DTYPE* __restrict const m_sigma_v_zn,
    DW_DTYPE* __restrict const m_sigma_h_yn,
#else
    DW_DTYPE* __restrict const m_sigma_v_yn,
#endif
    DW_DTYPE* __restrict const m_sigma_h_xn,
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
#endif
    DW_DTYPE* __restrict const dvydbuoyancy_store_1,
    DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
    void* __restrict const dvydbuoyancy_store_2,
    void* __restrict const dvydbuoyancy_store_3,
    char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
    DW_DTYPE* __restrict const dvydy_store_1,
    DW_DTYPE* __restrict const dvydy_store_1b,
    void* __restrict const dvydy_store_2, void* __restrict const dvydy_store_3,
    char const* __restrict const* __restrict const dvydy_filenames_ptr,
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
    DW_DTYPE* __restrict const grad_f_y, DW_DTYPE* __restrict const grad_f_x,
    DW_DTYPE* __restrict const grad_f_sigma_h,
    DW_DTYPE* __restrict const grad_f_sigma_v,
    DW_DTYPE* __restrict const grad_c11, DW_DTYPE* __restrict const grad_c13,
    DW_DTYPE* __restrict const grad_c33,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_y,
    DW_DTYPE* __restrict const grad_buoyancy_x,
    DW_DTYPE* __restrict const grad_c11_thread,
    DW_DTYPE* __restrict const grad_c13_thread,
    DW_DTYPE* __restrict const grad_c33_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z_thread,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_y_thread,
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
    int64_t const* __restrict const sources_y_i,
    int64_t const* __restrict const sources_x_i,
    int64_t const* __restrict const sources_sigma_h_i,
    int64_t const* __restrict const sources_sigma_v_i,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_z_i,
#endif
    int64_t const* __restrict const receivers_y_i,
    int64_t const* __restrict const receivers_x_i,
    int64_t const* __restrict const receivers_sigma_h_i,
    int64_t const* __restrict const receivers_sigma_v_i,
#if DW_NDIM >= 3
    DW_DTYPE const rdz,
#endif
    DW_DTYPE const rdy, DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
    int64_t const n_shots,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
    int64_t const ny, int64_t const nx,
#if DW_NDIM >= 3
    int64_t const n_sources_z_per_shot,
#endif
    int64_t const n_sources_y_per_shot, int64_t const n_sources_x_per_shot,
    int64_t const n_sources_sigma_h_per_shot,
    int64_t const n_sources_sigma_v_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_z_per_shot,
#endif
    int64_t n_receivers_y_per_shot, int64_t n_receivers_x_per_shot,
    int64_t const n_receivers_sigma_h_per_shot,
    int64_t const n_receivers_sigma_v_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const c11_requires_grad,
    bool const c13_requires_grad, bool const c33_requires_grad,
    bool const buoyancy_requires_grad, bool const c11_batched,
    bool const c13_batched, bool const c33_batched, bool const buoyancy_batched,
    bool const storage_compression, int64_t start_t,
#if DW_NDIM >= 3
    int64_t const pml_z0,
#endif
    int64_t const pml_y0, int64_t const pml_x0,
#if DW_NDIM >= 3
    int64_t const pml_z1,
#endif
    int64_t const pml_y1, int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
    int64_t const bc_y0, int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
    int64_t const bc_y1, int64_t const bc_x1, int64_t const n_threads,
    void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((c11_requires_grad) && (grad_c11 == NULL)) return 1;
    if ((c13_requires_grad) && (grad_c13 == NULL)) return 1;
    if ((c33_requires_grad) && (grad_c33 == NULL)) return 1;
#if DW_NDIM >= 3
    if ((buoyancy_requires_grad) && (grad_buoyancy_z == NULL)) return 1;
#endif
    if ((buoyancy_requires_grad) &&
        (grad_buoyancy_y == NULL || grad_buoyancy_x == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot > 0) && (receivers_z_i == NULL)) return 1;
#endif
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_sigma_h_per_shot > 0) && (receivers_sigma_h_i == NULL))
      return 1;
    if ((n_receivers_sigma_v_per_shot > 0) && (receivers_sigma_v_i == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0) && sources_z_i == NULL) return 1;
#endif
    if ((n_sources_y_per_shot > 0) && sources_y_i == NULL) return 1;
    if ((n_sources_x_per_shot > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_sigma_h_per_shot > 0) && sources_sigma_h_i == NULL) return 1;
    if ((n_sources_sigma_v_per_shot > 0) && sources_sigma_v_i == NULL) return 1;
  }
#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_z0 || bc_z1;
  int const has_bc_transverse = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_y0 || bc_y1;
  int const has_bc_transverse = bc_x0 || bc_x1;
#endif
  int const has_bc = has_bc_axis || has_bc_transverse;
  MAKE_BC_AXIS_TRANSVERSE();
  int64_t shot;
  /* Shots run serially; each time step's adjoint velocity and sigma_v/
   * sigma_h stencils are spread across threads (see the omp parallel regions
   * below). Model gradients accumulate directly into the shared gradient
   * buffers -- each cell is written by exactly one thread, and time steps
   * and shots are barrier-separated, so there is no race. */
  (void)grad_c11_thread;
  (void)grad_c13_thread;
  (void)grad_c33_thread;
#if DW_NDIM >= 3
  (void)grad_buoyancy_z_thread;
#endif
  (void)grad_buoyancy_y_thread;
  (void)grad_buoyancy_x_thread;
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#else
    int64_t const n_grid_points = ny * nx;
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
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        buoyancy_batched ? buoyancy_z + si : buoyancy_z;
    DW_DTYPE* __restrict const grad_buoyancy_z_shot =
        grad_buoyancy_z != NULL
            ? (grad_buoyancy_z + (buoyancy_batched ? si : 0))
            : NULL;
    DW_DTYPE* __restrict const vz_shot = vz + si;
    DW_DTYPE* __restrict const m_vzz_shot = m_vzz + si;
#endif
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
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE* __restrict const grad_buoyancy_y_shot =
        grad_buoyancy_y != NULL
            ? (grad_buoyancy_y + (buoyancy_batched ? si : 0))
            : NULL;
    DW_DTYPE* __restrict const vy_shot = vy + si;
    DW_DTYPE* __restrict const m_vyy_shot = m_vyy + si;

    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const sish = shot * n_sources_sigma_h_per_shot;
    int64_t const rish = shot * n_receivers_sigma_h_per_shot;
    int64_t const sisv = shot * n_sources_sigma_v_per_shot;
    int64_t const risv = shot * n_receivers_sigma_v_per_shot;
    int64_t const* __restrict const sources_x_i_shot =
        sources_x_i != NULL ? (sources_x_i + six) : NULL;
    int64_t const* __restrict const receivers_x_i_shot =
        receivers_x_i != NULL ? (receivers_x_i + rix) : NULL;
    int64_t const* __restrict const sources_sigma_h_i_shot =
        sources_sigma_h_i != NULL ? (sources_sigma_h_i + sish) : NULL;
    int64_t const* __restrict const receivers_sigma_h_i_shot =
        receivers_sigma_h_i != NULL ? (receivers_sigma_h_i + rish) : NULL;
    int64_t const* __restrict const sources_sigma_v_i_shot =
        sources_sigma_v_i != NULL ? (sources_sigma_v_i + sisv) : NULL;
    int64_t const* __restrict const receivers_sigma_v_i_shot =
        receivers_sigma_v_i != NULL ? (receivers_sigma_v_i + risv) : NULL;
    DW_DTYPE* __restrict const grad_f_x_shot =
        grad_f_x != NULL ? (grad_f_x + six) : NULL;
    DW_DTYPE const* __restrict const grad_r_x_shot =
        grad_r_x != NULL ? (grad_r_x + rix) : NULL;
    DW_DTYPE* __restrict const grad_f_sigma_h_shot =
        grad_f_sigma_h != NULL ? (grad_f_sigma_h + sish) : NULL;
    DW_DTYPE const* __restrict const grad_r_sigma_h_shot =
        grad_r_sigma_h != NULL ? (grad_r_sigma_h + rish) : NULL;
    DW_DTYPE* __restrict const grad_f_sigma_v_shot =
        grad_f_sigma_v != NULL ? (grad_f_sigma_v + sisv) : NULL;
    DW_DTYPE const* __restrict const grad_r_sigma_v_shot =
        grad_r_sigma_v != NULL ? (grad_r_sigma_v + risv) : NULL;

    DW_DTYPE* __restrict const grad_c11_shot =
        grad_c11 != NULL ? (grad_c11 + (c11_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_c13_shot =
        grad_c13 != NULL ? (grad_c13 + (c13_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_c33_shot =
        grad_c33 != NULL ? (grad_c33 + (c33_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_buoyancy_x_shot =
        grad_buoyancy_x != NULL
            ? (grad_buoyancy_x + (buoyancy_batched ? si : 0))
            : NULL;
    DW_DTYPE const* __restrict const c11_shot = c11_batched ? c11 + si : c11;
    DW_DTYPE const* __restrict const c13_shot = c13_batched ? c13 + si : c13;
    DW_DTYPE const* __restrict const c33_shot = c33_batched ? c33 + si : c33;
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE* __restrict const vx_shot = vx + si;
    DW_DTYPE* __restrict const sigma_v_shot = sigma_v + si;
    DW_DTYPE* __restrict const sigma_h_shot = sigma_h + si;
    DW_DTYPE* __restrict const m_vxx_shot = m_vxx + si;
    int64_t t;

    bool const cnormal_requires_grad =
        c11_requires_grad || c13_requires_grad || c33_requires_grad;
    int storage_err = 0;
#if DW_NDIM >= 3
    FILE* fp_dvzdbuoyancy = NULL;
    FILE* fp_dvzdz = NULL;
#endif
    FILE* fp_dvydbuoyancy = NULL;
    FILE* fp_dvydy = NULL;
    FILE* fp_dvxdbuoyancy = NULL;
    FILE* fp_dvxdx = NULL;

#if DW_NDIM >= 3
    OPEN_FILE_READ(dvzdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_READ(dvzdz, cnormal_requires_grad)
#endif
    OPEN_FILE_READ(dvydbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_READ(dvydy, cnormal_requires_grad)
    OPEN_FILE_READ(dvxdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_READ(dvxdx, cnormal_requires_grad)

    for (t = start_t - 1; t >= start_t - nt; --t) {
#if DW_NDIM >= 3
      SETUP_STORE_LOAD(dvzdbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvzdz, cnormal_requires_grad)
      DW_DTYPE* __restrict const m_sigma_v_z_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_v_zn : m_sigma_v_z) +
          si;
      DW_DTYPE* __restrict const m_sigma_h_y_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_h_yn : m_sigma_h_y) +
          si;
      DW_DTYPE* __restrict const m_sigma_v_zn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_v_z : m_sigma_v_zn) +
          si;
      DW_DTYPE* __restrict const m_sigma_h_yn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_h_y : m_sigma_h_yn) +
          si;
#else
      DW_DTYPE* __restrict const m_sigma_v_y_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_v_yn : m_sigma_v_y) +
          si;
      DW_DTYPE* __restrict const m_sigma_v_yn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_v_y : m_sigma_v_yn) +
          si;
#endif
      SETUP_STORE_LOAD(dvydbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvydy, cnormal_requires_grad)
      SETUP_STORE_LOAD(dvxdbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvxdx, cnormal_requires_grad)
      DW_DTYPE* __restrict const m_sigma_h_x_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_h_xn : m_sigma_h_x) +
          si;
      DW_DTYPE* __restrict const m_sigma_h_xn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? m_sigma_h_x : m_sigma_h_xn) +
          si;

      bool const rg_step_t = ((t % step_ratio) == 0);
      bool const c11_requires_grad_t = c11_requires_grad && rg_step_t;
      bool const c13_requires_grad_t = c13_requires_grad && rg_step_t;
      bool const c33_requires_grad_t = c33_requires_grad && rg_step_t;
      bool const buoyancy_requires_grad_t = buoyancy_requires_grad && rg_step_t;

      // Fold the (adjoint) sigma_v/sigma_h guard rings back into the
      // interior before the adjoint velocity update below reads them (via
      // VZ_Z/VY_Y/VX_X -- see acoustic_vti_kernels.h) and before the
      // source-gradient extraction just below (the adjoint counterpart of
      // forward's stress receiver record). Unlike acoustic's single
      // pressure field (genuinely self-adjoint under a plain re-fill),
      // acoustic_vti's sigma_v/sigma_h feed back into DIFFERENT velocity
      // components than they are imaged from at a reflecting face -- e.g.
      // sigma_v is imaged only on the axis face, but the compiled
      // backward's per-cell gathers (FUNC(backward_vz_cell)/vy_cell/
      // vx_cell in acoustic_vti_kernels.h) only ever run over the grid
      // interior, so reapplying the forward fill here discards real
      // gradient contributions near a reflecting face -- so this applies
      // the fill's true transpose instead (see the module comment above
      // FUNC(fill_p_bc_line_adjoint)).
      if (has_bc_axis) FILL_SIGMA_V_BC_ADJOINT();
      if (has_bc_transverse) FILL_SIGMA_H_BC_ADJOINT();

      if (n_sources_sigma_h_per_shot > 0) {
        record_from_wavefield(
            sigma_h_shot, sources_sigma_h_i_shot,
            grad_f_sigma_h_shot + t * n_shots * n_sources_sigma_h_per_shot,
            n_sources_sigma_h_per_shot);
      }
      if (n_sources_sigma_v_per_shot > 0) {
        record_from_wavefield(
            sigma_v_shot, sources_sigma_v_i_shot,
            grad_f_sigma_v_shot + t * n_shots * n_sources_sigma_v_per_shot,
            n_sources_sigma_v_per_shot);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
        int pml_y;
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y) DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
              int64_t y;
              int64_t x;

#if DW_NDIM >= 3

              // vz -- adjoint update (transpose of the c33/c13 coupling into
              // dvzdz), then ping-pong the m_sigma_v_z/m_sigma_h_y memories
              // (read by the sigma-adjoint block below at a possibly-neighbour
              // offset, so the pre-this-block value must survive).

              DW_OMP_FOR_SIMD(nowait) for (z = pml_bounds_zh[pml_z];
                                           z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_vz_cell)(c13_shot, c33_shot, sigma_h_shot,
                                           sigma_v_shot, vz_shot, m_vzz_shot,
                                           bz, dt, i, i, ny, nx, z, y, x,
                                           !DW_PML_SKIP(pml_z == 1), rdz);
                  }
#endif

              // vy -- 3-D: transverse (fed by sigma_h); 2-D: axis (fed by
              // sigma_v). See acoustic_vti_kernels.h's VY_Y/VY_Y_PML macros.

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_vy_cell)(
#if DW_NDIM == 3
                        c11_shot, c13_shot,
#else
                    c13_shot, c33_shot,
#endif
                        sigma_h_shot, sigma_v_shot, vy_shot, m_vyy_shot, by, dt,
                        i, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z,
#endif
                        y, x, !DW_PML_SKIP(pml_y == 1), rdy);
                  }

              // vx -- always transverse (fed by sigma_h).

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_vx_cell)(c11_shot, c13_shot, sigma_h_shot,
                                           sigma_v_shot, vx_shot, m_vxx_shot,
                                           bx, dt, i, i,
#if DW_NDIM >= 3
                                           ny,
#endif
                                           nx,
#if DW_NDIM >= 3
                                           z,
#endif
                                           y, x, !DW_PML_SKIP(pml_x == 1), rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Guard-band extension of the velocity-adjoint gathers (see the
      // "Guard-band gathers" module comment in acoustic_vti_kernels.h): the
      // interior loops just above never visit a cell within FD_PAD of a
      // non-PML face, so FUNC(fill_v_bc_adjoint) below has nothing genuine
      // to fold there without this. Each velocity component is fed by
      // exactly one stress-derivative term along its own axis (see
      // acoustic_vti_kernels.h's module comment), so each block below
      // extends only that component's own axis, sweeping the other axes at
      // their ordinary (unextended) interior/PML range; unlike elastic(_
      // vti)'s velocity guard block, no cell here is ever the simultaneous
      // target of two different axes' extended ranges for the SAME
      // component (each component only has one extended axis), so no
      // corner special-casing or thread-race is possible within one
      // component's block, but the block still runs single-threaded (O(
      // FD_PAD * grid-face-area), negligible next to the interior loops
      // above) for consistency with the stress-adjoint guard block below,
      // which does have genuine multi-axis corners (sigma_h).
      if (has_bc) {
#if DW_NDIM >= 3
        if (bc[0] != 0 || bc[1] != 0) {
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], true);
          int64_t z, y, x;
          for (z = rz.target_lo; z < rz.target_hi; ++z) {
            if (!FUNC(dw_in_target)(rz, z) || FUNC(dw_in_std)(rz, z)) continue;
            for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                SET_I;
                FUNC(backward_vz_cell_guard)(c13_shot, c33_shot, sigma_h_shot,
                                             sigma_v_shot, vz_shot, dt, i, i,
                                             nz, ny, nx, z, y, x, rdz);
              }
          }
        }
#endif
        if (bc[2] != 0 || bc[3] != 0) {
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], true);
#if DW_NDIM >= 3
          int64_t z;
#endif
          int64_t y, x;
#if DW_NDIM >= 3
          for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
            for (y = ry.target_lo; y < ry.target_hi; ++y) {
              if (!FUNC(dw_in_target)(ry, y) || FUNC(dw_in_std)(ry, y))
                continue;
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                SET_I;
                FUNC(backward_vy_cell_guard)(
#if DW_NDIM == 3
                    c11_shot, c13_shot,
#else
                  c13_shot, c33_shot,
#endif
                    sigma_h_shot, sigma_v_shot, vy_shot, dt, i, i,
#if DW_NDIM >= 3
                    nz,
#endif
                    ny, nx,
#if DW_NDIM >= 3
                    z,
#endif
                    y, x, rdy);
              }
            }
        }
        if (bc[4] != 0 || bc[5] != 0) {
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], true);
#if DW_NDIM >= 3
          int64_t z;
#endif
          int64_t y, x;
#if DW_NDIM >= 3
          for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
            for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                if (!FUNC(dw_in_target)(rx, x) || FUNC(dw_in_std)(rx, x))
                  continue;
                SET_I;
                FUNC(backward_vx_cell_guard)(c11_shot, c13_shot, sigma_h_shot,
                                             sigma_v_shot, vx_shot, dt, i, i,
#if DW_NDIM >= 3
                                             nz,
#endif
                                             ny, nx,
#if DW_NDIM >= 3
                                             z,
#endif
                                             y, x, rdx);
              }
        }
      }

      // Fold the (adjoint) velocity guard ring back into the interior
      // before the sigma-adjoint update loops below read it. Unlike
      // acoustic's single pressure-velocity system (audited self-adjoint),
      // acoustic_vti's velocities feed back into a coupled sigma_v/sigma_h
      // pair, so this applies the true transpose of FUNC(fill_v_bc) rather
      // than reapplying the forward fill (see the module comment above
      // FUNC(fill_p_bc_line_adjoint)). No-op for all-PML.
      if (has_bc) FILL_V_BC_ADJOINT();

      // Velocity-type source-gradient extraction, the M_SIGMA_V_Z/
      // M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X PML memory-variable update
      // (split out of FUNC(backward_v*_cell) -- see FUNC(backward_vz_cell_
      // pml_mem)'s comment in acoustic_vti_kernels.h), and the
      // buoyancy-gradient accumulation all read the velocity adjoint at
      // (near-face) interior cells the fold above may have just changed,
      // so all three must run after it: a pre-fold read there is stale
      // under the corrected architecture. Every one of these three uses
      // the SAME (unextended) interior/PML range FUNC(forward_vz_cell)
      // etc. used in the forward pass, so this stays a flat loop over that
      // original range, not the guard-extended one; the M_SIGMA** write
      // must run unconditionally (every time step, regardless of
      // buoyancy_requires_grad) since FUNC(backward_sigma_cell) reads it
      // next time step regardless of whether any gradient is requested.
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
              FUNC(backward_vz_cell_pml_mem)(
                  vz_shot, m_sigma_v_z_t, m_sigma_v_zn_t, buoyancy_z_shot[i],
                  azh, dt, i, ny, nx, z, y, x, is_pml_z);
              if (buoyancy_requires_grad_t) {
                grad_buoyancy_z_shot[i] += vz_shot[i] *
                                           dvzdbuoyancy_store_1_t[i] *
                                           (DW_DTYPE)step_ratio;
              }
            }
      }
#endif
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
              FUNC(backward_vy_cell_pml_mem)(vy_shot,
#if DW_NDIM == 3
                                             m_sigma_h_y_t, m_sigma_h_yn_t,
#else
                                           m_sigma_v_y_t, m_sigma_v_yn_t,
#endif
                                             buoyancy_y_shot[i], ayh, dt, i,
#if DW_NDIM >= 3
                                             ny, z,
#endif
                                             nx, y, x, is_pml_y);
              if (buoyancy_requires_grad_t) {
                grad_buoyancy_y_shot[i] += vy_shot[i] *
                                           dvydbuoyancy_store_1_t[i] *
                                           (DW_DTYPE)step_ratio;
              }
            }
      }
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
        int64_t y, x;
#if DW_NDIM >= 3
        for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
          for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
            for (x = pml_bounds_xh[0]; x < pml_bounds_xh[3]; ++x) {
              SET_I;
              bool const is_pml_x = FUNC(dw_is_pml_region)(
                  x, pml_x0, MAX(pml_x0, pml_x1 - 1), nx - FD_PAD);
              FUNC(backward_vx_cell_pml_mem)(vx_shot, m_sigma_h_x_t,
                                             m_sigma_h_xn_t, buoyancy_x_shot[i],
                                             axh, dt, i,
#if DW_NDIM >= 3
                                             ny, z,
#endif
                                             nx, y, x, is_pml_x);
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
        int pml_y;
        int pml_x;
#if DW_NDIM >= 3
        DW_UNROLL_PML for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
            DW_UNROLL_PML
          for (pml_y = 0; pml_y < 3; ++pml_y) DW_UNROLL_PML
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
              int64_t y;
              int64_t x;

              // sigma_v, sigma_h -- model gradients (derivation doc Sec. 5):
              //   grad_c11 = sum s_h (S_x + S_y)          [2-D: s_h S_x]
              //   grad_c13 = sum (s_v (S_x+S_y) + s_h S_z) [2-D: s_v S_x+s_h
              //   S_y] grad_c33 = sum s_v S_z                   [2-D: s_v S_y]
              // gated on (t % step_ratio)==0 AND scaled by step_ratio (the
              // elastic_born-escalation-#7 pattern), then the adjoint PML
              // memory for m_vxx/m_vyy/m_vzz (single-buffer: written here,
              // read by the vz/vy/vx block ABOVE in the same reversed
              // timestep -- i.e. read-before-write within an iteration, so no
              // ping-pong needed), then the sigma_v/sigma_h adjoint update
              // itself (transpose of the velocity update's buoyancy coupling).

              DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_sigma_cell)(
#if DW_NDIM >= 3
                        vz_shot, buoyancy_z_shot,
#endif
                        vy_shot, buoyancy_y_shot, vx_shot, buoyancy_x_shot,
                        sigma_v_shot, sigma_h_shot,
#if DW_NDIM >= 3
                        m_vzz_shot,
#endif
                        m_vyy_shot, m_vxx_shot,
#if DW_NDIM == 3
                        m_sigma_v_z_t, m_sigma_h_y_t,
#else
                    m_sigma_v_y_t,
#endif
                        m_sigma_h_x_t,
#if DW_NDIM >= 3
                        dvzdz_store_1_t,
#endif
                        dvydy_store_1_t, dvxdx_store_1_t, c11_shot[i],
                        c13_shot[i], c33_shot[i], grad_c11_shot, grad_c13_shot,
                        grad_c33_shot, dt, step_ratio, c11_requires_grad_t,
                        c13_requires_grad_t, c33_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx,
#if DW_NDIM >= 3
                        z, !DW_PML_SKIP(pml_z == 1), az, bzh, rdz,
#endif
                        y, !DW_PML_SKIP(pml_y == 1), ay, byh, rdy, x,
                        !DW_PML_SKIP(pml_x == 1), ax, bxh, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Guard-band extension of the stress-adjoint gathers (see the
      // "Guard-band gathers" module comment in acoustic_vti_kernels.h): the
      // interior loop just above never visits a cell within FD_PAD of a
      // non-PML face, so the NEXT time step's FUNC(fill_p_bc_adjoint) calls
      // (at the top of this loop) have nothing genuine to fold there
      // without this. sigma_v extends only on its own axis (z in 3-D, y in
      // 2-D -- its forward fill never touches the transverse faces); sigma_h
      // extends on every transverse axis (y and x in 3-D, x in 2-D) -- see
      // acoustic_vti.c's module comment above FUNC(fill_p_bc_line). The
      // sigma_h y/x blocks (3-D) genuinely share corner cells, so -- like
      // the velocity guard block above -- this runs single-threaded.
      if (has_bc) {
#if DW_NDIM >= 3
        if (bc[0] != 0 || bc[1] != 0) {
          FUNC(dw_ext_range)
          const rz = FUNC(dw_compute_ext_range)(nz, bc[0], bc[1], false);
          int64_t z, y, x;
          for (z = rz.target_lo; z < rz.target_hi; ++z) {
            if (!FUNC(dw_in_target)(rz, z) || FUNC(dw_in_std)(rz, z)) continue;
            for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                SET_I;
                FUNC(backward_sigma_v_z_cell_guard)(buoyancy_z_shot, vz_shot,
                                                    sigma_v_shot, dt, i, i, nz,
                                                    ny, nx, z, rdz);
              }
          }
        }
        if (bc[2] != 0 || bc[3] != 0) {
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], false);
          int64_t z, y, x;
          for (y = ry.target_lo; y < ry.target_hi; ++y) {
            if (!FUNC(dw_in_target)(ry, y) || FUNC(dw_in_std)(ry, y)) continue;
            for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
              for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
                SET_I;
                FUNC(backward_sigma_h_y_cell_guard)(buoyancy_y_shot, vy_shot,
                                                    sigma_h_shot, dt, i, i, ny,
                                                    nx, y, rdy);
              }
          }
        }
#else
        if (bc[2] != 0 || bc[3] != 0) {
          FUNC(dw_ext_range)
          const ry = FUNC(dw_compute_ext_range)(ny, bc[2], bc[3], false);
          int64_t y, x;
          for (y = ry.target_lo; y < ry.target_hi; ++y) {
            if (!FUNC(dw_in_target)(ry, y) || FUNC(dw_in_std)(ry, y)) continue;
            for (x = pml_bounds_x[0]; x < pml_bounds_x[3]; ++x) {
              SET_I;
              FUNC(backward_sigma_v_y_cell_guard)(buoyancy_y_shot, vy_shot,
                                                  sigma_v_shot, dt, i, i, ny,
                                                  nx, y, rdy);
            }
          }
        }
#endif
        if (bc[4] != 0 || bc[5] != 0) {
          FUNC(dw_ext_range)
          const rx = FUNC(dw_compute_ext_range)(nx, bc[4], bc[5], false);
#if DW_NDIM >= 3
          int64_t z;
#endif
          int64_t y, x;
#if DW_NDIM >= 3
          for (z = pml_bounds_z[0]; z < pml_bounds_z[3]; ++z)
#endif
            for (y = pml_bounds_y[0]; y < pml_bounds_y[3]; ++y)
              for (x = rx.target_lo; x < rx.target_hi; ++x) {
                if (!FUNC(dw_in_target)(rx, x) || FUNC(dw_in_std)(rx, x))
                  continue;
                SET_I;
                FUNC(backward_sigma_h_x_cell_guard)(buoyancy_x_shot, vx_shot,
                                                    sigma_h_shot, dt, i, i,
#if DW_NDIM >= 3
                                                    ny,
#endif
                                                    nx, x, rdx);
              }
        }
      }

#if DW_NDIM >= 3
      if (n_receivers_z_per_shot > 0) {
        add_to_wavefield(vz_shot, receivers_z_i_shot,
                         grad_r_z_shot + t * n_shots * n_receivers_z_per_shot,
                         n_receivers_z_per_shot);
      }
#endif
      if (n_receivers_y_per_shot > 0) {
        add_to_wavefield(vy_shot, receivers_y_i_shot,
                         grad_r_y_shot + t * n_shots * n_receivers_y_per_shot,
                         n_receivers_y_per_shot);
      }
      if (n_receivers_x_per_shot > 0) {
        add_to_wavefield(vx_shot, receivers_x_i_shot,
                         grad_r_x_shot + t * n_shots * n_receivers_x_per_shot,
                         n_receivers_x_per_shot);
      }
      if (n_receivers_sigma_h_per_shot > 0) {
        add_to_wavefield(
            sigma_h_shot, receivers_sigma_h_i_shot,
            grad_r_sigma_h_shot + t * n_shots * n_receivers_sigma_h_per_shot,
            n_receivers_sigma_h_per_shot);
      }
      if (n_receivers_sigma_v_per_shot > 0) {
        add_to_wavefield(
            sigma_v_shot, receivers_sigma_v_i_shot,
            grad_r_sigma_v_shot + t * n_shots * n_receivers_sigma_v_per_shot,
            n_receivers_sigma_v_per_shot);
      }
    }
  storage_cleanup:
#if DW_NDIM >= 3
    CLOSE_FILE(dvzdbuoyancy)
    CLOSE_FILE(dvzdz)
#endif
    CLOSE_FILE(dvydbuoyancy)
    CLOSE_FILE(dvydy)
    CLOSE_FILE(dvxdbuoyancy)
    CLOSE_FILE(dvxdx)
    if (storage_err) return 1;
  }
  return 0;
}

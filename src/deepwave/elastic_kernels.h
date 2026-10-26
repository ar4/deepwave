#ifndef DW_ELASTIC_KERNELS_H
#define DW_ELASTIC_KERNELS_H

/*
 * elastic_kernels.h
 *
 * Single-source per-grid-point stencil math for the elastic (isotropic,
 * variable density, staggered grid, velocity-stress) wave equation
 * propagator (see elastic.c for the physics/PML/staggered-grid-layout
 * description). This header is included by BOTH elastic.c (compiled by gcc)
 * and elastic.cu (compiled by nvcc), after each has defined the FUNC()/CAT()
 * name-mangling macros (and, for elastic, the DW_MPML-dependent
 * ELASTIC_VARIANT()) and before either uses the ND_INDEX/P_A(*)/P_B(*)/VZ/VY/
 * VX/SIGMA(*)/LAMB/MU/M_(*)/... family below. Elastic has THREE velocity
 * components (vz/vy/vx) and SIX stress components (three normal --
 * sigmazz/sigmayy/sigmaxx, updated together as one combined equation because
 * they share the same lambda*div(v) term -- and three shear --
 * sigmaxy/sigmaxz/sigmayz, each independent), so this header has more
 * per-grid-point functions than acoustic_kernels.h/viscoacoustic_kernels.h:
 * elastic.c/elastic.cu already run/compute the velocity update as three
 * independent per-axis blocks (FUNC(forward_vz_cell)/vy_cell/vx_cell, exactly
 * as acoustic's do) and the stress update as FOUR independent blocks -- one
 * combined normal-stress block (FUNC(forward_sigmaii_cell), the direct
 * analogue of acoustic's combined FUNC(forward_p_cell)) plus three
 * independent shear-stress blocks (FUNC(forward_sigmaxy_cell)/sigmaxz_cell/
 * sigmayz_cell, one per shear plane) -- for 7 forward functions, and
 * correspondingly 7 backward (adjoint) functions of the same shape, 14 in
 * total. It contains:
 *
 *  * The per-axis addressing macros (ND_INDEX, the M-PML-or-1-D PML profile
 *    accessors P_AZ/P_BZ/P_AZH/P_BZH/.../P_AXH/P_BXH, and the field-access
 *    macros VZ/VY/VX, SIGMAZZ/SIGMAYY/SIGMAXX/SIGMAXY/SIGMAXZ/SIGMAYZ, LAMB/
 *    MU/MU_YX/MU_ZX/MU_ZY/BUOYANCY_Z/Y/X, the M_SIGMA** and M_V** PML memory
 *    families, LAMB_2MU, and the per-axis adjoint-term macros VZ_Z/VZ_Y/VZ_X
 *    (+ their _PML twins), VY_Z/VY_Y/VY_X(+_PML), VX_Z/VX_Y/VX_X(+_PML),
 *    SIGMAZZ_Z(+_PML), SIGMAYY_Y(+_PML), SIGMAXX_X(+_PML), SIGMAXY_Y/X
 *    (+_PML), SIGMAXZ_Z/X(+_PML), SIGMAYZ_Z/Y(+_PML)), lifted verbatim from
 *    elastic.cu's copy of these macros -- elastic.cu's naming/addressing
 *    convention (wavefields vz/vy/vx/sigma(*)/m_v(**)/m_sigma(**) addressed by
 *    `i`, model arrays lamb/mu/mu_(*)/buoyancy_(*) addressed by `j` via
 *    ND_INDEX, with a per-cell VALUE read directly by the caller wherever no
 *    neighbour offset is needed) is adopted as the one shared convention,
 *    exactly as acoustic_kernels.h/viscoacoustic_kernels.h already
 *    established.
 *    elastic.c's copy of the LAMB/MU/MU_YX/MU_ZX/MU_ZY/BUOYANCY_Z/Y/X macros
 *    was textually identical EXCEPT that it used `i` (not `j`) to address the
 *    model arrays -- a difference that disappears once the CPU driver passes
 *    `i` for `j` too at every call site below (it already is the shot-local
 *    index there, exactly as `j` is on the CUDA driver -- see the
 *    addressing-convention note below). The DW_MPML-dependent PML-region-skip
 *    decision (DW_PML_SKIP on the CPU driver, DW_PML_TEST on the CUDA driver)
 *    is NOT part of this header: it is driver-side scaffolding (it depends on
 *    the outer region-loop indices on the CPU driver and on the raw z/y/x
 *    position against the __constant__ pml_*0/pml_*1 bounds on the CUDA
 *    driver, two genuinely different parallelisation strategies), so both
 *    drivers keep computing their own `pml_z`/`pml_y`/`pml_x` bool and simply
 *    pass the resulting bool into the functions below (matching
 *    acoustic_kernels.h's `pml_z`/`pml_y`/`pml_x` parameters exactly).
 *  * FUNC(forward_vz_cell) / FUNC(forward_vy_cell) / FUNC(forward_vx_cell):
 *    the per-grid-point half-grid velocity update for each axis (div of the
 *    three stress components that act on that velocity component, plus that
 *    axis's three PML memory-variable recursions). Model buoyancy is read
 *    once, by value (no neighbour offset needed), exactly as
 *    acoustic_kernels.h's forward_v*_cell take buoyancy by value.
 *    FUNC(forward_sigmaii_cell): the combined normal-stress update (all
 *    present axes' dv/dn terms, weighted by lambda and (for the diagonal
 *    term) 2*mu, matching elastic.c's/elastic.cu's single combined
 *    "sigmaii" loop body -- the direct analogue of acoustic_kernels.h's
 *    FUNC(forward_p_cell)). FUNC(forward_sigmaxy_cell) / sigmaxz_cell /
 *    sigmayz_cell: the three independent shear-stress updates (each a sum of
 *    two velocity cross-derivatives weighted by the corresponding
 *    off-diagonal mu), matching elastic.c's/elastic.cu's three independent
 *    shear loop bodies.
 *  * FUNC(backward_vz_cell) / FUNC(backward_vy_cell) / FUNC(backward_vx_cell):
 *    the adjoint velocity update per axis -- structurally the transpose of
 *    FUNC(forward_sigma*_cell) (it differentiates the *stress* adjoint
 *    fields, weighted by lambda/mu, to update the velocity adjoint), plus
 *    that axis's M_SIGMA** PML memory-variable recursion (feeding
 *    FUNC(backward_sigma*_cell) at the *next* time step -- see the
 *    ping-pong-buffer note below) and the buoyancy gradient accumulation.
 *    Because this differentiates lambda/mu *with a neighbour offset* (the
 *    stencil in FUNC(backward_vz_cell) etc. touches lambda/mu at every
 *    offset the finite-difference stencil visits, not just the centre point),
 *    lamb_shot/mu_shot/mu_zy_shot/mu_zx_shot/mu_yx_shot are POINTERS here
 *    (addressed by `j`), unlike the forward functions where they are read
 *    only at the centre point.
 *    FUNC(backward_sigmaii_cell) / sigmaxy_cell / sigmaxz_cell / sigmayz_cell:
 *    the adjoint stress update -- structurally the transpose of
 *    FUNC(forward_v*_cell) (it differentiates the *velocity* adjoint fields,
 *    weighted by buoyancy, to update the stress adjoint), the (grad_lamb,
 *    grad_mu) / grad_mu_yx / grad_mu_zx / grad_mu_zy accumulation, and the
 *    M_V** PML memory-variable update (read by FUNC(backward_v*_cell) at the
 *    *next* time step). buoyancy_z_shot/buoyancy_y_shot/buoyancy_x_shot are
 *    POINTERS here (addressed by `j`, same reasoning as above); lamb_shot/
 *    mu_shot/mu_yx_shot/mu_zx_shot/mu_zy_shot are read only at the centre
 *    point (no finite-difference neighbour offset touches them here), but are
 *    passed as POINTERS too (dereferenced through the LAMB(0,0,0)-style
 *    macro), matching elastic.cu's existing convention exactly (elastic.c's
 *    copy of this same code read the identical value directly as
 *    `lamb_shot[i]`; both forms read the same memory location, since `i` and
 *    `j` coincide on the CPU driver, so unifying on the macro form changes no
 *    value computed).
 *
 * PML memory-variable buffering: the M_V** family (m_vzz/m_vzy/m_vzx/m_vyz/
 * m_vxz/m_vxx/m_vyy/m_vxy/m_vyx) is a SINGLE buffer per array, read (with a
 * finite-difference neighbour offset, via the M_VZZ/etc. macros) by
 * FUNC(backward_v*_cell) and written in place (centre point only) by
 * FUNC(backward_sigma*_cell) LATER in the same time step -- since the
 * backward driver always runs the v-adjoint region before the sigma-adjoint
 * region, this is safe without ping-pong (backward_v*_cell always sees the
 * value FUNC(backward_sigma*_cell) wrote at the *previous* time step). The
 * M_SIGMA** family (m_sigmazzz/m_sigmayzy/m_sigmaxzx/m_sigmayzz/m_sigmaxzz/
 * m_sigmayyy/m_sigmaxyy/m_sigmaxyx/m_sigmaxxx) is the reverse dependency
 * (written by FUNC(backward_v*_cell), read with a neighbour offset by
 * FUNC(backward_sigma*_cell)) but in the OPPOSITE order within a time step
 * (v-adjoint region runs first and would otherwise read the value it is
 * about to overwrite this same step), so it DOES need the old/new ping-pong
 * pair the driver already maintains (m_sigmazzz vs m_sigmazzzn, etc.); the
 * functions below simply take "old" (read, neighbour-offset via macro, and
 * also read at the centre point to seed the recursion) and "new" (write,
 * centre point only) pointers, matching the existing ping-pong parameter
 * names.
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest and PML-region triple loop in
 * elastic.c, the CUDA kernel launch configuration and per-thread grid-point
 * assignment (including the single-kernel-multiple-independently-gated-block
 * structure of forward_kernel_v/forward_kernel_sigma/backward_kernel_sigma/
 * backward_kernel_v) in elastic.cu -- stays in elastic.c/elastic.cu exactly
 * as before; only the math performed once the caller has identified "this"
 * grid point (and, for the backward stress functions, decided whether this
 * cell is within the relevant axis's PML region) lives here.
 *
 * Addressing convention (matches scalar_kernels.h/acoustic_kernels.h/
 * viscoacoustic_kernels.h exactly): vz/vy/vx/sigma(*)/m_v(**)/m_sigma(**)/
 * dv(*)d(*)_store_1/grad_buoyancy_(*)/... are addressed by `i` (the CPU
 * driver's per-shot-local flat index, which is also valid as the CUDA
 * driver's global
 * shot-spanning index since both drivers pass already-appropriately-offset
 * pointers); lamb_shot/mu_shot/mu_*_shot/buoyancy_*_shot are addressed by `j`
 * (the per-shot-local index) wherever a POINTER is taken (used only by the
 * backward functions, via the LAMB/MU/MU_YX/MU_ZX/MU_ZY/BUOYANCY_Z/Y/X
 * macros), because the CUDA driver's model pointers are shot-local (offset by
 * shot_idx*shot_numel, not `i`) while the wavefield arrays span all shots. On
 * the CPU driver, wavefields/model arrays are ALL already offset to the
 * current shot, so `i` and `j` are the same value there and the CPU driver
 * simply passes `i` for both. The forward functions need no `j`: they only
 * ever read the *centre* value of the relevant model field (no neighbour
 * offset), so the caller passes that value directly, by value, exactly as
 * scalar_kernels.h's forward_cell does for v_shot.
 *
 * DW_HOST_DEVICE marks these functions callable from either gcc (a plain
 * forced-inline static function) or nvcc (additionally callable from device
 * code).
 */

#ifdef __CUDACC__
#define DW_HOST_DEVICE static __host__ __device__ __forceinline__
#elif defined(_MSC_VER)
#define DW_HOST_DEVICE static __forceinline
#else
#define DW_HOST_DEVICE static inline __attribute__((always_inline))
#endif

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) ((i) + (dz) * ny * nx + (dy) * nx + (dx))
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) ((i) + (dy) * nx + (dx))
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) ((i) + (dx))
#endif

/* PML profile access. The standard (non-M-PML) profiles are 1-D per axis, so a
 * given axis's profile is indexed by that axis's own coordinate only
 * (e.g. by[y + dy]). The multiaxial (M-PML) profiles vary on every axis, so
 * they are indexed at the full-grid spatial cell. PML_IDX is the spatial-only
 * flattened index (no shot offset -- the profiles have no shot dimension) of
 * the cell offset (dz, dy, dx) from the current (z, y, x). Each P_* accessor
 * takes the single offset along that profile's own axis; under DW_MPML it
 * expands to a full-grid load, otherwise to the historical 1-D load (which is
 * textually equivalent to the previous code, so the default kernel is
 * unchanged). */
#if DW_NDIM == 3
#define PML_IDX(dz, dy, dx) \
  ((z + (dz)) * ny * nx + (y + (dy)) * nx + (x + (dx)))
#elif DW_NDIM == 2
#define PML_IDX(dz, dy, dx) ((y + (dy)) * nx + (x + (dx)))
#else
#define PML_IDX(dz, dy, dx) (x + (dx))
#endif

#ifdef DW_MPML
#if DW_NDIM >= 3
#define P_AZ(d) az[PML_IDX(d, 0, 0)]
#define P_BZ(d) bz[PML_IDX(d, 0, 0)]
#define P_AZH(d) azh[PML_IDX(d, 0, 0)]
#define P_BZH(d) bzh[PML_IDX(d, 0, 0)]
#endif
#if DW_NDIM >= 2
#define P_AY(d) ay[PML_IDX(0, d, 0)]
#define P_BY(d) by[PML_IDX(0, d, 0)]
#define P_AYH(d) ayh[PML_IDX(0, d, 0)]
#define P_BYH(d) byh[PML_IDX(0, d, 0)]
#endif
#define P_AX(d) ax[PML_IDX(0, 0, d)]
#define P_BX(d) bx[PML_IDX(0, 0, d)]
#define P_AXH(d) axh[PML_IDX(0, 0, d)]
#define P_BXH(d) bxh[PML_IDX(0, 0, d)]
#else /* standard 1-D per-axis profiles */
#if DW_NDIM >= 3
#define P_AZ(d) az[z + (d)]
#define P_BZ(d) bz[z + (d)]
#define P_AZH(d) azh[z + (d)]
#define P_BZH(d) bzh[z + (d)]
#endif
#if DW_NDIM >= 2
#define P_AY(d) ay[y + (d)]
#define P_BY(d) by[y + (d)]
#define P_AYH(d) ayh[y + (d)]
#define P_BYH(d) byh[y + (d)]
#endif
#define P_AX(d) ax[x + (d)]
#define P_BX(d) bx[x + (d)]
#define P_AXH(d) axh[x + (d)]
#define P_BXH(d) bxh[x + (d)]
#endif /* DW_MPML */

// Wavefields, addressed by i (see the addressing-convention note above).
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define SIGMAZZ(dz, dy, dx) sigmazz[ND_INDEX(i, dz, dy, dx)]
#elif DW_NDIM == 2
#define SIGMAZZ(dz, dy, dx) 0
#endif
#if DW_NDIM >= 2
#define SIGMAYY(dz, dy, dx) sigmayy[ND_INDEX(i, dz, dy, dx)]
#endif
#define SIGMAXX(dz, dy, dx) sigmaxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define SIGMAXY(dz, dy, dx) sigmaxy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define SIGMAXZ(dz, dy, dx) sigmaxz[ND_INDEX(i, dz, dy, dx)]
#define SIGMAYZ(dz, dy, dx) sigmayz[ND_INDEX(i, dz, dy, dx)]
#endif

// Models, addressed by the shot-local index j (see the addressing-convention
// note above) -- used only by the backward (adjoint) functions, via the
// LAMB_2MU/VZ_Z/.../SIGMAYZ_Y(_PML) macros below.
#define LAMB(dz, dy, dx) lamb_shot[ND_INDEX(j, dz, dy, dx)]
#define MU(dz, dy, dx) mu_shot[ND_INDEX(j, dz, dy, dx)]
#ifdef DW_PM
// Parameter-modified free-surface (Cao and Chen, 2018,
// https://doi.org/10.1190/geo2018-0098.1) per-component normal-stress
// stiffnesses, addressed by the shot-local index j (neighbour-offset capable,
// exactly like LAMB/MU). C[i][j] is the coefficient coupling the velocity
// derivative in direction j (0 = y, 1 = x) to normal-stress component i
// (0 = sigma_yy, 1 = sigma_xx): the normal-stress update is
// d sigma_i = dt * sum_j C[i][j] d v_j / d x_j. In the interior the diagonal
// (C00, C11) equals lamb + 2 mu and the off-diagonal (C01, C10) equals lamb,
// reproducing the isotropic relation; at a surface cell they take the
// stress-free-reduced values. 2-D only (DW_PM is built for DW_NDIM == 2).
#define C00(dz, dy, dx) c00_shot[ND_INDEX(j, dz, dy, dx)]
#define C01(dz, dy, dx) c01_shot[ND_INDEX(j, dz, dy, dx)]
#define C10(dz, dy, dx) c10_shot[ND_INDEX(j, dz, dy, dx)]
#define C11(dz, dy, dx) c11_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define MU_YX(dz, dy, dx) mu_yx_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define MU_ZX(dz, dy, dx) mu_zx_shot[ND_INDEX(j, dz, dy, dx)]
#define MU_ZY(dz, dy, dx) mu_zy_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// PML memory variables for the velocity equations (the "M_SIGMA**" family --
// see the ping-pong-buffering note above): written (centre point only) by
// FUNC(backward_v*_cell), read (with a neighbour offset, via these macros) by
// FUNC(backward_sigma*_cell).
#if DW_NDIM >= 2
#define M_SIGMAXYY(dz, dy, dx) m_sigmaxyy[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_SIGMAXXX(dz, dy, dx) m_sigmaxxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define M_SIGMAXZZ(dz, dy, dx) m_sigmaxzz[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 2
#define M_SIGMAXYX(dz, dy, dx) m_sigmaxyx[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYYY(dz, dy, dx) m_sigmayyy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_SIGMAYZZ(dz, dy, dx) m_sigmayzz[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 3
#define M_SIGMAXZX(dz, dy, dx) m_sigmaxzx[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYZY(dz, dy, dx) m_sigmayzy[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAZZZ(dz, dy, dx) m_sigmazzz[ND_INDEX(i, dz, dy, dx)]
#endif

// PML memory variables for the stress equations (the "M_V**" family -- see
// the ping-pong-buffering note above): read (with a neighbour offset, via
// these macros) by FUNC(backward_v*_cell), written (centre point only) by
// FUNC(backward_sigma*_cell).
#if DW_NDIM >= 2
#define M_VXY(dz, dy, dx) m_vxy[ND_INDEX(i, dz, dy, dx)]
#define M_VYX(dz, dy, dx) m_vyx[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_VXX(dz, dy, dx) m_vxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define M_VYY(dz, dy, dx) m_vyy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_VZZ(dz, dy, dx) m_vzz[ND_INDEX(i, dz, dy, dx)]
#define M_VXZ(dz, dy, dx) m_vxz[ND_INDEX(i, dz, dy, dx)]
#define M_VZX(dz, dy, dx) m_vzx[ND_INDEX(i, dz, dy, dx)]
#define M_VYZ(dz, dy, dx) m_vyz[ND_INDEX(i, dz, dy, dx)]
#define M_VZY(dz, dy, dx) m_vzy[ND_INDEX(i, dz, dy, dx)]
#endif

// Adjoint-term macros used by the backward (adjoint) velocity update
// (FUNC(backward_v*_cell)): the transpose of the forward stress equations.
#define LAMB_2MU(dz, dy, dx) (LAMB(dz, dy, dx) + 2 * MU(dz, dy, dx))

#if DW_NDIM >= 3
#define VZ_Z(dz, dy, dx)                          \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         LAMB_2MU(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#define VZ_Y(dz, dy, dx) (dt * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#define VZ_X(dz, dy, dx) (dt * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#define VZ_Z_PML(dz, dy, dx)                          \
  (dt * (1 + P_BZ(dz)) *                              \
       (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +      \
        LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +      \
        LAMB_2MU(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) + \
   P_BZ(dz) * M_VZZ(dz, dy, dx))
#define VZ_Y_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BYH(dy)) * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx) + \
   P_BYH(dy) * M_VZY(dz, dy, dx))
#define VZ_X_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BXH(dx)) * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx) + \
   P_BXH(dx) * M_VZX(dz, dy, dx))
#endif

#if DW_NDIM >= 2
#if DW_NDIM >= 3
#define VY_Z(dz, dy, dx) (dt * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y(dz, dy, dx)                          \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#else
#define VY_Y(dz, dy, dx)                              \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +     \
         LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#define VY_X(dz, dy, dx) (dt * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#if DW_NDIM >= 3
#define VY_Z_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BZH(dz)) * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx) + \
   P_BZH(dz) * M_VYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y_PML(dz, dy, dx)                          \
  (dt * (1 + P_BY(dy)) *                              \
       (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +      \
        LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx)) + \
   P_BY(dy) * M_VYY(dz, dy, dx))
#else
#define VY_Y_PML(dz, dy, dx)                         \
  (dt * (1 + P_BY(dy)) *                             \
       (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +     \
        LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
        LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) +    \
   P_BY(dy) * M_VYY(dz, dy, dx))
#endif
#define VY_X_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BXH(dx)) * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx) + \
   P_BXH(dx) * M_VYX(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define VX_Z(dz, dy, dx) (dt * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y(dz, dy, dx) (dt * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X(dz, dy, dx) (dt * LAMB_2MU(0, 0, dx) * SIGMAXX(0, 0, dx))
#else
#define VX_X(dz, dy, dx)                              \
  (dt * (LAMB_2MU(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +     \
         LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#if DW_NDIM >= 3
#define VX_Z_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BZH(dz)) * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx) + \
   P_BZH(dz) * M_VXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BYH(dy)) * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx) + \
   P_BYH(dy) * M_VXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X_PML(dz, dy, dx)                                        \
  (dt * (1 + P_BX(dx)) * (LAMB_2MU(0, 0, dx) * SIGMAXX(0, 0, dx)) + \
   P_BX(dx) * M_VXX(0, 0, dx))
#else
#define VX_X_PML(dz, dy, dx)                         \
  (dt * (1 + P_BX(dx)) *                             \
       (LAMB_2MU(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
        LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +     \
        LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) +    \
   P_BX(dx) * M_VXX(dz, dy, dx))
#endif

#ifdef DW_PM
// Parameter-modified transposes of the normal-stress line (2-D). The forward
// couples (d v_y/d y, d v_x/d x) into (sigma_yy, sigma_xx) by the matrix C;
// these adjoint terms couple the stress adjoints back into the velocity
// adjoints by C^T. VY_Y accumulates the sigma-adjoint contributions weighted
// by the y-direction column (C[*][0]: C00 to sigma_yy, C10 to sigma_xx); VX_X
// by the x-direction column (C[*][1]: C11 to sigma_xx, C01 to sigma_yy).
#define VY_Y_PM(dz, dy, dx)                      \
  (dt * (C10(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C00(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#define VY_Y_PM_PML(dz, dy, dx)                  \
  (dt * (1 + P_BY(dy)) *                         \
       (C10(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C00(dz, dy, dx) * SIGMAYY(dz, dy, dx)) + \
   P_BY(dy) * M_VYY(dz, dy, dx))
#define VX_X_PM(dz, dy, dx)                      \
  (dt * (C11(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C01(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#define VX_X_PM_PML(dz, dy, dx)                  \
  (dt * (1 + P_BX(dx)) *                         \
       (C11(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C01(dz, dy, dx) * SIGMAYY(dz, dy, dx)) + \
   P_BX(dx) * M_VXX(dz, dy, dx))
#endif

// Adjoint-term macros used by the backward (adjoint) stress update
// (FUNC(backward_sigma*_cell)): the transpose of the forward velocity
// equations.
#if DW_NDIM >= 3
#define SIGMAZZ_Z_PML(dz, dy, dx)                                   \
  ((1 + P_BZH(dz)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BZH(dz) * M_SIGMAZZZ(dz, dy, dx))
#define SIGMAZZ_Z(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define SIGMAYY_Y_PML(dz, dy, dx)                                   \
  ((1 + P_BYH(dy)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BYH(dy) * M_SIGMAYYY(dz, dy, dx))
#define SIGMAYY_Y(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define SIGMAXX_X_PML(dz, dy, dx)                                   \
  ((1 + P_BXH(dx)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BXH(dx) * M_SIGMAXXX(dz, dy, dx))
#define SIGMAXX_X(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

#if DW_NDIM >= 2
#define SIGMAXY_Y_PML(dz, dy, dx)                                  \
  ((1 + P_BY(dy)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BY(dy) * M_SIGMAXYY(dz, dy, dx))
#define SIGMAXY_Y(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define SIGMAXY_X_PML(dz, dy, dx)                                  \
  ((1 + P_BX(dx)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BX(dx) * M_SIGMAXYX(dz, dy, dx))
#define SIGMAXY_X(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define SIGMAXZ_Z_PML(dz, dy, dx)                                  \
  ((1 + P_BZ(dz)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BZ(dz) * M_SIGMAXZZ(dz, dy, dx))
#define SIGMAXZ_Z(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define SIGMAXZ_X_PML(dz, dy, dx)                                  \
  ((1 + P_BX(dx)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BX(dx) * M_SIGMAXZX(dz, dy, dx))
#define SIGMAXZ_X(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))

#define SIGMAYZ_Z_PML(dz, dy, dx)                                  \
  ((1 + P_BZ(dz)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BZ(dz) * M_SIGMAYZZ(dz, dy, dx))
#define SIGMAYZ_Z(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define SIGMAYZ_Y_PML(dz, dy, dx)                                  \
  ((1 + P_BY(dy)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BY(dy) * M_SIGMAYZY(dz, dy, dx))
#define SIGMAYZ_Y(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif

/* ============================= Forward ============================= */

#if DW_NDIM >= 3
/* vz += buoyancy_z * dt * (d(sigmazz)/dz + d(sigmayz)/dy + d(sigmaxz)/dx),
 * each term accumulating its axis's PML memory variable when inside that
 * axis's PML region. `i` addresses sigmazz/sigmayz/sigmaxz/vz/m_sigmazzz/
 * m_sigmayzy/m_sigmaxzx/dvzdbuoyancy_store_1 (per-shot-local on the CPU
 * driver, global on the CUDA driver); `buoyancy_z_shot` is the (already
 * shot-selected) buoyancy VALUE at this point (read once by the caller --
 * this update never needs a neighbour offset of it); `pml_z`/`pml_y`/`pml_x`
 * are true when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz, DW_DTYPE* __restrict const vz,
    DW_DTYPE* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const dt,
    bool const buoyancy_requires_grad, int64_t const i, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, bool const pml_x, DW_DTYPE const rdz,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  if (!pml_z) {
    w_sum += DIFFZH1(SIGMAZZ);
  } else {
    DW_DTYPE d3 = DIFFZH1(SIGMAZZ);
    m_sigmazzz[i] = P_AZH(0) * m_sigmazzz[i] + P_BZH(0) * d3;
    d3 += m_sigmazzz[i];
    w_sum += d3;
  }
  if (!pml_y) {
    w_sum += DIFFY1(SIGMAYZ);
  } else {
    DW_DTYPE d2 = DIFFY1(SIGMAYZ);
    m_sigmayzy[i] = P_AY(0) * m_sigmayzy[i] + P_BY(0) * d2;
    d2 += m_sigmayzy[i];
    w_sum += d2;
  }
  if (!pml_x) {
    w_sum += DIFFX1(SIGMAXZ);
  } else {
    DW_DTYPE d1 = DIFFX1(SIGMAXZ);
    m_sigmaxzx[i] = P_AX(0) * m_sigmaxzx[i] + P_BX(0) * d1;
    d1 += m_sigmaxzx[i];
    w_sum += d1;
  }

  if (buoyancy_requires_grad) {
    dvzdbuoyancy_store_1[i] = dt * w_sum;
  }
  vz[i] += buoyancy_z_shot * dt * w_sum;
}
#endif

#if DW_NDIM >= 2
/* vy += buoyancy_y * dt * ((d(sigmayz)/dz, if DW_NDIM>=3) + d(sigmayy)/dy +
 * d(sigmaxy)/dx) (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vy_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmayz,
#endif
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxy, DW_DTYPE* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmayzz,
#endif
    DW_DTYPE* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmaxyx,
    DW_DTYPE* __restrict const dvydbuoyancy_store_1,
    DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
#endif
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const dt, bool const buoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
    int64_t const y, int64_t const x, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ1(SIGMAYZ);
  } else {
    DW_DTYPE d3 = DIFFZ1(SIGMAYZ);
    m_sigmayzz[i] = P_AZ(0) * m_sigmayzz[i] + P_BZ(0) * d3;
    d3 += m_sigmayzz[i];
    w_sum += d3;
  }
#endif
  if (!pml_y) {
    w_sum += DIFFYH1(SIGMAYY);
  } else {
    DW_DTYPE d2 = DIFFYH1(SIGMAYY);
    m_sigmayyy[i] = P_AYH(0) * m_sigmayyy[i] + P_BYH(0) * d2;
    d2 += m_sigmayyy[i];
    w_sum += d2;
  }
  if (!pml_x) {
    w_sum += DIFFX1(SIGMAXY);
  } else {
    DW_DTYPE d1 = DIFFX1(SIGMAXY);
    m_sigmaxyx[i] = P_AX(0) * m_sigmaxyx[i] + P_BX(0) * d1;
    d1 += m_sigmaxyx[i];
    w_sum += d1;
  }

  if (buoyancy_requires_grad) {
    dvydbuoyancy_store_1[i] = dt * w_sum;
  }
  vy[i] += buoyancy_y_shot * dt * w_sum;
}
#endif

/* vx += buoyancy_x * dt * ((d(sigmaxz)/dz, if DW_NDIM>=3) +
 * (d(sigmaxy)/dy, if DW_NDIM>=2) + d(sigmaxx)/dx) (see
 * FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vx_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE const* __restrict const sigmaxx, DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmaxyy,
#endif
    DW_DTYPE* __restrict const m_sigmaxxx,
    DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
    DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
#endif
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, bool const buoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ1(SIGMAXZ);
  } else {
    DW_DTYPE d3 = DIFFZ1(SIGMAXZ);
    m_sigmaxzz[i] = P_AZ(0) * m_sigmaxzz[i] + P_BZ(0) * d3;
    d3 += m_sigmaxzz[i];
    w_sum += d3;
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY1(SIGMAXY);
  } else {
    DW_DTYPE d2 = DIFFY1(SIGMAXY);
    m_sigmaxyy[i] = P_AY(0) * m_sigmaxyy[i] + P_BY(0) * d2;
    d2 += m_sigmaxyy[i];
    w_sum += d2;
  }
#endif
  if (!pml_x) {
    w_sum += DIFFXH1(SIGMAXX);
  } else {
    DW_DTYPE d1 = DIFFXH1(SIGMAXX);
    m_sigmaxxx[i] = P_AXH(0) * m_sigmaxxx[i] + P_BXH(0) * d1;
    d1 += m_sigmaxxx[i];
    w_sum += d1;
  }

  if (buoyancy_requires_grad) {
    dvxdbuoyancy_store_1[i] = dt * w_sum;
  }
  vx[i] += buoyancy_x_shot * dt * w_sum;
}

/* sigmazz/sigmayy/sigmaxx += dt * (lambda * div(v) + (that axis's term) 2 *
 * mu * d(v_axis)/d(axis)), the combined normal-stress update (every present
 * axis's dv/dn term accumulates into `w_sum` = lambda * div(v), like
 * acoustic_kernels.h's forward_p_cell combines its axes into div_v), plus
 * that axis's PML memory-variable recursion and (if gradients are needed)
 * storing dt*(dv_axis/d(axis)) for the backward pass. `i` addresses vz/vy/vx/
 * sigmazz/sigmayy/sigmaxx/m_vzz/m_vyy/m_vxx/dv*d*_store_1 (per-shot-local on
 * the CPU driver, global on the CUDA driver); `lamb_shot`/`mu_shot` are the
 * (already shot-selected) model VALUEs at this point (read once by the
 * caller -- never need a neighbour offset here); `pml_z`/`pml_y`/`pml_x` are
 * true when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_sigmaii_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const dvzdz_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const dvydy_store_1,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const dvxdx_store_1,
#ifdef DW_PM
    DW_DTYPE const c00_val, DW_DTYPE const c01_val, DW_DTYPE const c10_val,
    DW_DTYPE const c11_val, DW_DTYPE const dt,
#else
    DW_DTYPE const lamb_shot, DW_DTYPE const mu_shot, DW_DTYPE const dt,
#endif
    bool const lamb_or_mu_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const by, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const rdx) {
#ifndef DW_PM
  DW_DTYPE w_sum = 0;
#endif

#if DW_NDIM >= 3
  DW_DTYPE dvzdz;
  if (pml_z) {
    dvzdz = DIFFZ1(VZ);
    m_vzz[i] = P_AZ(0) * m_vzz[i] + P_BZ(0) * dvzdz;
    dvzdz += m_vzz[i];
  } else {
    dvzdz = DIFFZ1(VZ);
  }
  w_sum += lamb_shot * dvzdz;
  if (lamb_or_mu_requires_grad) {
    dvzdz_store_1[i] = dt * dvzdz;
  }
#endif

#if DW_NDIM >= 2
  DW_DTYPE dvydy;
  if (pml_y) {
    dvydy = DIFFY1(VY);
    m_vyy[i] = P_AY(0) * m_vyy[i] + P_BY(0) * dvydy;
    dvydy += m_vyy[i];
  } else {
    dvydy = DIFFY1(VY);
  }
#ifndef DW_PM
  w_sum += lamb_shot * dvydy;
#endif
  if (lamb_or_mu_requires_grad) {
    dvydy_store_1[i] = dt * dvydy;
  }
#endif

  DW_DTYPE dvxdx;
  if (pml_x) {
    dvxdx = DIFFX1(VX);
    m_vxx[i] = P_AX(0) * m_vxx[i] + P_BX(0) * dvxdx;
    dvxdx += m_vxx[i];
  } else {
    dvxdx = DIFFX1(VX);
  }
#ifndef DW_PM
  w_sum += lamb_shot * dvxdx;
#endif
  if (lamb_or_mu_requires_grad) {
    dvxdx_store_1[i] = dt * dvxdx;
  }

#ifdef DW_PM
  // Per-component surface-modified normal-stress update (2-D):
  // sigma_yy += dt (C00 dvy/dy + C01 dvx/dx), sigma_xx += dt (C10 dvy/dy +
  // C11 dvx/dx). Reduces to the isotropic (lamb, mu) relation in the interior.
  sigmayy[i] += dt * (c00_val * dvydy + c01_val * dvxdx);
  sigmaxx[i] += dt * (c10_val * dvydy + c11_val * dvxdx);
#else
#if DW_NDIM >= 3
  sigmazz[i] += dt * (w_sum + 2 * mu_shot * dvzdz);
#endif
#if DW_NDIM >= 2
  sigmayy[i] += dt * (w_sum + 2 * mu_shot * dvydy);
#endif
  sigmaxx[i] += dt * (w_sum + 2 * mu_shot * dvxdx);
#endif
}

#if DW_NDIM >= 2
/* sigmaxy += dt * mu_yx * (d(vx)/dy + d(vy)/dx) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmaxy_cell)(
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE const mu_yx_shot, DW_DTYPE const dt, bool const mu_requires_grad,
    int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE dvxdy, dvydx;

  if (pml_y) {
    dvxdy = DIFFYH1(VX);
    m_vxy[i] = P_AYH(0) * m_vxy[i] + P_BYH(0) * dvxdy;
    dvxdy += m_vxy[i];
  } else {
    dvxdy = DIFFYH1(VX);
  }
  w_sum += dvxdy;

  if (pml_x) {
    dvydx = DIFFXH1(VY);
    m_vyx[i] = P_AXH(0) * m_vyx[i] + P_BXH(0) * dvydx;
    dvydx += m_vyx[i];
  } else {
    dvydx = DIFFXH1(VY);
  }
  w_sum += dvydx;

  if (mu_requires_grad) {
    dvydx_plus_dvxdy_store_1[i] = dt * w_sum;
  }
  sigmaxy[i] += dt * mu_yx_shot * w_sum;
}
#endif

#if DW_NDIM >= 3
/* sigmaxz += dt * mu_zx * (d(vx)/dz + d(vz)/dx) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmaxz_cell)(
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const m_vxz,
    DW_DTYPE* __restrict const m_vzx,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE const mu_zx_shot, DW_DTYPE const dt, bool const mu_requires_grad,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_x,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const rdz, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE dvxdz, dvzdx;

  if (pml_z) {
    dvxdz = DIFFZH1(VX);
    m_vxz[i] = P_AZH(0) * m_vxz[i] + P_BZH(0) * dvxdz;
    dvxdz += m_vxz[i];
  } else {
    dvxdz = DIFFZH1(VX);
  }
  w_sum += dvxdz;

  if (pml_x) {
    dvzdx = DIFFXH1(VZ);
    m_vzx[i] = P_AXH(0) * m_vzx[i] + P_BXH(0) * dvzdx;
    dvzdx += m_vzx[i];
  } else {
    dvzdx = DIFFXH1(VZ);
  }
  w_sum += dvzdx;

  if (mu_requires_grad) {
    dvzdx_plus_dvxdz_store_1[i] = dt * w_sum;
  }
  sigmaxz[i] += dt * mu_zx_shot * w_sum;
}

/* sigmayz += dt * mu_zy * (d(vy)/dz + d(vz)/dy) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmayz_cell)(
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE const mu_zy_shot, DW_DTYPE const dt, bool const mu_requires_grad,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const rdz, DW_DTYPE const rdy) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE dvydz, dvzdy;

  if (pml_z) {
    dvydz = DIFFZH1(VY);
    m_vyz[i] = P_AZH(0) * m_vyz[i] + P_BZH(0) * dvydz;
    dvydz += m_vyz[i];
  } else {
    dvydz = DIFFZH1(VY);
  }
  w_sum += dvydz;

  if (pml_y) {
    dvzdy = DIFFYH1(VZ);
    m_vzy[i] = P_AYH(0) * m_vzy[i] + P_BYH(0) * dvzdy;
    dvzdy += m_vzy[i];
  } else {
    dvzdy = DIFFYH1(VZ);
  }
  w_sum += dvzdy;

  if (mu_requires_grad) {
    dvzdy_plus_dvydz_store_1[i] = dt * w_sum;
  }
  sigmayz[i] += dt * mu_zy_shot * w_sum;
}
#endif

/* ============================= Backward ============================= */

#if DW_NDIM >= 3
/* vz += -(d(VZ_Z)/dz + d(VZ_Y)/dy + d(VZ_X)/dx) (the transpose of
 * FUNC(forward_sigmaii_cell)/sigmaxy_cell/sigmaxz_cell's use of vz), then
 * each axis's M_SIGMA** PML memory-variable recursion (feeding
 * FUNC(backward_sigma*_cell) at the *next* time step -- see the
 * ping-pong-buffering note above) and the buoyancy gradient accumulation.
 * `i` addresses sigma(*)/vz/m_sigmazzz(n)/m_sigmayzy(n)/m_sigmaxzx(n)/
 * dvzdbuoyancy_store_1/grad_buoyancy_z (per-shot-local on the CPU driver,
 * global on the CUDA driver); `j` addresses the already-shot-sliced
 * lamb_shot/mu_shot/mu_zy_shot/mu_zx_shot (equal to `i` on the CPU driver;
 * the local per-shot grid index on the CUDA driver, distinct from the global
 * `i`) -- see the addressing-convention note at the top of this file.
 * `pml_z`/`pml_y`/`pml_x` are true inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
    DW_DTYPE const* __restrict const mu_zy_shot,
    DW_DTYPE const* __restrict const mu_zx_shot,
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz, DW_DTYPE* __restrict const vz,
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmazzz_old,
    DW_DTYPE* __restrict const m_sigmazzz_new,
    DW_DTYPE const* __restrict const m_sigmayzy_old,
    DW_DTYPE* __restrict const m_sigmayzy_new,
    DW_DTYPE const* __restrict const m_sigmaxzx_old,
    DW_DTYPE* __restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, bool const pml_x, DW_DTYPE const rdz,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  w_sum += pml_z ? -DIFFZH1(VZ_Z_PML) : -DIFFZH1(VZ_Z);
  w_sum += pml_y ? -DIFFY1(VZ_Y_PML) : -DIFFY1(VZ_Y);
  w_sum += pml_x ? -DIFFX1(VZ_X_PML) : -DIFFX1(VZ_X);

  vz[i] += w_sum;
}

/* M_SIGMA** PML memory-variable update for FUNC(backward_vz_cell) -- split
 * out and moved to run in a separate pass AFTER FUNC(fill_elastic_velocity_
 * bc_adjoint) (see the driver-side comment at its new call site). vz[i]
 * here must be read post-fold: for a cell the fold accumulates into or
 * clamps to zero (e.g. a rigid on-row velocity node), the value FUNC(
 * backward_vz_cell) computed pre-fold is stale -- the fold changes it
 * immediately afterward -- so writing this memory variable from the
 * pre-fold value would seed the next time step's recursion with the wrong
 * number for exactly those cells. For every other cell (the fold never
 * touches it) vz[i] is identical pre- and post-fold, so this split changes
 * nothing there. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_pml_mem)(
    DW_DTYPE const* __restrict const vz,
    DW_DTYPE const* __restrict const m_sigmazzz_old,
    DW_DTYPE* __restrict const m_sigmazzz_new,
    DW_DTYPE const* __restrict const m_sigmayzy_old,
    DW_DTYPE* __restrict const m_sigmayzy_new,
    DW_DTYPE const* __restrict const m_sigmaxzx_old,
    DW_DTYPE* __restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const dt, int64_t const i,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_y, bool const pml_x) {
  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz_old[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy_old[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx_old[i];
  }
}
#endif

#if DW_NDIM >= 2
/* Adjoint velocity update, y axis (see FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vy_cell)(
#ifdef DW_PM
    DW_DTYPE const* __restrict const c00_shot,
    DW_DTYPE const* __restrict const c10_shot,
#else
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy_shot,
#endif
    DW_DTYPE const* __restrict const mu_yx_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmayz,
#endif
    DW_DTYPE const* __restrict const sigmaxy, DW_DTYPE* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vyz,
#endif
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmayzz_old,
    DW_DTYPE* __restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const* __restrict const m_sigmayyy_old,
    DW_DTYPE* __restrict const m_sigmayyy_new,
    DW_DTYPE const* __restrict const m_sigmaxyx_old,
    DW_DTYPE* __restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
    int64_t const y, int64_t const x, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DIFFZ1(VY_Z_PML) : -DIFFZ1(VY_Z);
#endif
#ifdef DW_PM
  w_sum += pml_y ? -DIFFYH1(VY_Y_PM_PML) : -DIFFYH1(VY_Y_PM);
#else
  w_sum += pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y);
#endif
  w_sum += pml_x ? -DIFFX1(VY_X_PML) : -DIFFX1(VY_X);

  vy[i] += w_sum;
}

/* M_SIGMA** PML memory-variable update for FUNC(backward_vy_cell) -- see
 * FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason: vy[i]
 * must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vy_cell_pml_mem)(
    DW_DTYPE const* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmayzz_old,
    DW_DTYPE* __restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const* __restrict const m_sigmayyy_old,
    DW_DTYPE* __restrict const m_sigmayyy_new,
    DW_DTYPE const* __restrict const m_sigmaxyx_old,
    DW_DTYPE* __restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az,
#endif
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const dt, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z, bool const pml_z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x) {
#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz_old[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy_old[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx_old[i];
  }
}
#endif

/* Adjoint velocity update, x axis (see FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell)(
#ifdef DW_PM
    DW_DTYPE const* __restrict const c01_shot,
    DW_DTYPE const* __restrict const c11_shot,
#else
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vxy,
#endif
    DW_DTYPE const* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz_old,
    DW_DTYPE* __restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy_old,
    DW_DTYPE* __restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx_old,
    DW_DTYPE* __restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DIFFZ1(VX_Z_PML) : -DIFFZ1(VX_Z);
#endif
#if DW_NDIM >= 2
  w_sum += pml_y ? -DIFFY1(VX_Y_PML) : -DIFFY1(VX_Y);
#endif
#ifdef DW_PM
  w_sum += pml_x ? -DIFFXH1(VX_X_PM_PML) : -DIFFXH1(VX_X_PM);
#else
  w_sum += pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X);
#endif

  vx[i] += w_sum;
}

/* M_SIGMA** PML memory-variable update for FUNC(backward_vx_cell) -- see
 * FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason: vx[i]
 * must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_pml_mem)(
    DW_DTYPE const* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz_old,
    DW_DTYPE* __restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy_old,
    DW_DTYPE* __restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx_old,
    DW_DTYPE* __restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay,
#endif
    DW_DTYPE const* __restrict const axh, DW_DTYPE const dt, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z, bool const pml_z,
#endif
#if DW_NDIM >= 2
    int64_t const nx, int64_t const y, bool const pml_y,
#endif
    int64_t const x, bool const pml_x) {
#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmaxzz_new[i] =
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz_old[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy_old[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx_old[i];
  }
}

/* grad_lamb/grad_mu accumulation, the M_V** PML memory-variable update (read
 * by FUNC(backward_v*_cell) at the *next* time step -- see the
 * ping-pong-buffering note above), and sigmazz/sigmayy/sigmaxx +=
 * -(d(SIGMAZZ_Z)/dz + d(SIGMAYY_Y)/dy + d(SIGMAXX_X)/dx) (the transpose of
 * FUNC(forward_vz_cell)/vy_cell/vx_cell's use of sigma*). `i`/`j` addressing
 * matches FUNC(backward_vz_cell) above. `lamb_shot`/`mu_shot` are read only
 * at the centre point here (no finite-difference neighbour offset touches
 * them in this function), but are passed as POINTERS (dereferenced through
 * the LAMB(0,0,0)-style macro) to match elastic.cu's existing convention --
 * see the header comment at the top of this file. `buoyancy_z_shot`/
 * `buoyancy_y_shot`/`buoyancy_x_shot` ARE read with a neighbour offset here
 * (via SIGMAZZ_Z(_PML) etc.), so are genuinely POINTERS.
 */
DW_HOST_DEVICE void FUNC(backward_sigmaii_cell)(
#ifdef DW_PM
    DW_DTYPE const* __restrict const c00_shot,
    DW_DTYPE const* __restrict const c01_shot,
    DW_DTYPE const* __restrict const c10_shot,
    DW_DTYPE const* __restrict const c11_shot,
#else
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y_shot,
#endif
    DW_DTYPE const* __restrict const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const dvzdz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const dvydy_store,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE* __restrict const m_vxx,
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE const* __restrict const dvxdx_store,
#ifdef DW_PM
    DW_DTYPE* __restrict const grad_c00, DW_DTYPE* __restrict const grad_c01,
    DW_DTYPE* __restrict const grad_c10, DW_DTYPE* __restrict const grad_c11,
#else
    DW_DTYPE* __restrict const grad_lamb, DW_DTYPE* __restrict const grad_mu,
#endif
    DW_DTYPE const dt, int64_t const step_ratio, bool const lamb_requires_grad,
    bool const mu_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdx) {
#ifdef DW_PM
  // grad_C[i][j] = sum_t (adjoint sigma_i) * dt * (dv_j/dx_j); the transpose
  // of the constant-coefficient normal-stress line. C requires grad whenever
  // lamb or mu does (C is a differentiable function of both), so the four
  // gradients are accumulated together under that combined condition. Their
  // chain rule back to lamb/mu is done by autograd in Python.
  if (lamb_requires_grad || mu_requires_grad) {
    grad_c00[i] += sigmayy[i] * dvydy_store[i] * (DW_DTYPE)step_ratio;
    grad_c01[i] += sigmayy[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
    grad_c10[i] += sigmaxx[i] * dvydy_store[i] * (DW_DTYPE)step_ratio;
    grad_c11[i] += sigmaxx[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
  }
#else
  if (lamb_requires_grad) {
#if DW_NDIM == 3
    grad_lamb[i] += (sigmaxx[i] + sigmayy[i] + sigmazz[i]) *
                    (dvxdx_store[i] + dvydy_store[i] + dvzdz_store[i]) *
                    (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_lamb[i] += (sigmaxx[i] + sigmayy[i]) *
                    (dvxdx_store[i] + dvydy_store[i]) * (DW_DTYPE)step_ratio;
#else
    grad_lamb[i] += sigmaxx[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
#endif
  }
  if (mu_requires_grad) {
#if DW_NDIM == 3
    grad_mu[i] += 2 *
                  (sigmaxx[i] * dvxdx_store[i] + sigmayy[i] * dvydy_store[i] +
                   sigmazz[i] * dvzdz_store[i]) *
                  (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_mu[i] += 2 *
                  (sigmaxx[i] * dvxdx_store[i] + sigmayy[i] * dvydy_store[i]) *
                  (DW_DTYPE)step_ratio;
#else
    grad_mu[i] += 2 * sigmaxx[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
#endif
  }
#endif

#if DW_NDIM >= 3
  if (pml_z) {
    m_vzz[i] = LAMB_2MU(0, 0, 0) * dt * P_AZ(0) * sigmazz[i] +
               LAMB(0, 0, 0) * dt * P_AZ(0) * sigmaxx[i] +
               LAMB(0, 0, 0) * dt * P_AZ(0) * sigmayy[i] + P_AZ(0) * m_vzz[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
#ifdef DW_PM
    m_vyy[i] = C00(0, 0, 0) * dt * P_AY(0) * sigmayy[i] +
               C10(0, 0, 0) * dt * P_AY(0) * sigmaxx[i] + P_AY(0) * m_vyy[i];
#else
    m_vyy[i] = LAMB_2MU(0, 0, 0) * dt * P_AY(0) * sigmayy[i] +
               LAMB(0, 0, 0) * dt * P_AY(0) * sigmaxx[i] +
#if DW_NDIM == 3
               LAMB(0, 0, 0) * dt * P_AY(0) * sigmazz[i] +
#endif
               P_AY(0) * m_vyy[i];
#endif
  }
#endif
  if (pml_x) {
#ifdef DW_PM
    m_vxx[i] = C11(0, 0, 0) * dt * P_AX(0) * sigmaxx[i] +
               C01(0, 0, 0) * dt * P_AX(0) * sigmayy[i] + P_AX(0) * m_vxx[i];
#else
    m_vxx[i] = LAMB_2MU(0, 0, 0) * dt * P_AX(0) * sigmaxx[i] +
#if DW_NDIM >= 2
               LAMB(0, 0, 0) * dt * P_AX(0) * sigmayy[i] +
#endif
#if DW_NDIM == 3
               LAMB(0, 0, 0) * dt * P_AX(0) * sigmazz[i] +
#endif
               P_AX(0) * m_vxx[i];
#endif
  }

#if DW_NDIM >= 3
  sigmazz[i] += pml_z ? -DIFFZ1(SIGMAZZ_Z_PML) : -DIFFZ1(SIGMAZZ_Z);
#endif
#if DW_NDIM >= 2
  sigmayy[i] += pml_y ? -DIFFY1(SIGMAYY_Y_PML) : -DIFFY1(SIGMAYY_Y);
#endif
  sigmaxx[i] += pml_x ? -DIFFX1(SIGMAXX_X_PML) : -DIFFX1(SIGMAXX_X);
}

#if DW_NDIM >= 2
/* grad_mu_yx accumulation, M_VXY/M_VYX PML memory update, and sigmaxy +=
 * -(d(SIGMAXY_Y)/dy + d(SIGMAXY_X)/dx) (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxy_cell)(
    DW_DTYPE const* __restrict const mu_yx_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE const* __restrict const dvydx_plus_dvxdy_store,
    DW_DTYPE* __restrict const grad_mu_yx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const mu_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (mu_requires_grad) {
    grad_mu_yx[i] +=
        sigmaxy[i] * dvydx_plus_dvxdy_store[i] * (DW_DTYPE)step_ratio;
  }

  if (pml_y) {
    m_vxy[i] =
        MU_YX(0, 0, 0) * dt * P_AYH(0) * sigmaxy[i] + P_AYH(0) * m_vxy[i];
  }
  if (pml_x) {
    m_vyx[i] =
        MU_YX(0, 0, 0) * dt * P_AXH(0) * sigmaxy[i] + P_AXH(0) * m_vyx[i];
  }

  sigmaxy[i] += (pml_y ? -DIFFYH1(SIGMAXY_Y_PML) : -DIFFYH1(SIGMAXY_Y)) +
                (pml_x ? -DIFFXH1(SIGMAXY_X_PML) : -DIFFXH1(SIGMAXY_X));
}
#endif

#if DW_NDIM >= 3
/* grad_mu_zx accumulation, M_VXZ/M_VZX PML memory update, and sigmaxz +=
 * -(d(SIGMAXZ_Z)/dz + d(SIGMAXZ_X)/dx) (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxz_cell)(
    DW_DTYPE const* __restrict const mu_zx_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const m_vxz,
    DW_DTYPE* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE* __restrict const grad_mu_zx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const mu_requires_grad, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_x,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const rdz, DW_DTYPE const rdx) {
  if (mu_requires_grad) {
    grad_mu_zx[i] +=
        sigmaxz[i] * dvzdx_plus_dvxdz_store[i] * (DW_DTYPE)step_ratio;
  }

  if (pml_z) {
    m_vxz[i] =
        MU_ZX(0, 0, 0) * dt * P_AZH(0) * sigmaxz[i] + P_AZH(0) * m_vxz[i];
  }
  if (pml_x) {
    m_vzx[i] =
        MU_ZX(0, 0, 0) * dt * P_AXH(0) * sigmaxz[i] + P_AXH(0) * m_vzx[i];
  }

  sigmaxz[i] += (pml_z ? -DIFFZH1(SIGMAXZ_Z_PML) : -DIFFZH1(SIGMAXZ_Z)) +
                (pml_x ? -DIFFXH1(SIGMAXZ_X_PML) : -DIFFXH1(SIGMAXZ_X));
}

/* grad_mu_zy accumulation, M_VYZ/M_VZY PML memory update, and sigmayz +=
 * -(d(SIGMAYZ_Z)/dz + d(SIGMAYZ_Y)/dy) (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmayz_cell)(
    DW_DTYPE const* __restrict const mu_zy_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const dvzdy_plus_dvydz_store,
    DW_DTYPE* __restrict const grad_mu_zy, DW_DTYPE const dt,
    int64_t const step_ratio, bool const mu_requires_grad, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const rdz, DW_DTYPE const rdy) {
  if (mu_requires_grad) {
    grad_mu_zy[i] +=
        sigmayz[i] * dvzdy_plus_dvydz_store[i] * (DW_DTYPE)step_ratio;
  }

  if (pml_z) {
    m_vyz[i] =
        MU_ZY(0, 0, 0) * dt * P_AZH(0) * sigmayz[i] + P_AZH(0) * m_vyz[i];
  }
  if (pml_y) {
    m_vzy[i] =
        MU_ZY(0, 0, 0) * dt * P_AYH(0) * sigmayz[i] + P_AYH(0) * m_vzy[i];
  }

  sigmayz[i] += (pml_z ? -DIFFZH1(SIGMAYZ_Z_PML) : -DIFFZH1(SIGMAYZ_Z)) +
                (pml_y ? -DIFFYH1(SIGMAYZ_Y_PML) : -DIFFYH1(SIGMAYZ_Y));
}
#endif

/* ========================= Guard-band gathers ========================= *
 *
 * The backward gather kernels above (FUNC(backward_v*_cell)/FUNC(backward_
 * sigma*_cell)) only ever run over the grid's strict interior/PML region
 * (elastic.c's/elastic.cu's pml_bounds_* / pml_bounds_*h arrays both start at
 * FD_PAD and stop FD_PAD or FD_PAD-1 short of the far edge), so they never
 * deposit an adjoint contribution into the guard ring next to a reflecting
 * (non-PML) face. FUNC(fill_elastic_stress_bc_adjoint)/FUNC(fill_elastic_
 * velocity_bc_adjoint) (see the module comment above those, elastic.c) then
 * fold that guard ring into the interior -- but with nothing genuine ever
 * written there, that fold silently discards real gradient contributions.
 * The functions below are guard-band-only companions to the interior
 * gathers: same physics (transpose of the corresponding FUNC(forward_*_
 * cell)), but evaluated with bounds-safe stencil taps (DW_DIFF*_SAFE, which
 * treat any tap landing outside [0, nz)/[0, ny)/[0, nx) as exactly zero --
 * see adjoint_fix_prototype/proto_adjoint.py's `d1_full`/`d1h_full`, the
 * executable spec this translates) so they may be called at guard-ring
 * cells, where a stencil can reach up to FD_PAD cells past the physical
 * array. They omit the grad_lamb/grad_mu/grad_mu_* / grad_buoyancy_*
 * accumulation the interior functions perform (guard cells are not
 * independent model-gradient locations -- the model itself is only
 * mirror-padded, never an independent DOF there) and, for the velocity
 * gathers, the buoyancy-gradient accumulation the interior functions used
 * to perform too (moved to a separate post-fold pass -- see elastic.c's/
 * elastic.cu's driver-side comment at its new call site -- since a pre-fold
 * read of a near-face interior cell is stale once the fold changes it).
 *
 * elastic.c's/elastic.cu's new guard-band driver loops call these over
 * exactly the *new* cells a field's boundary fill touches beyond the
 * existing interior/PML loop -- see FUNC(dw_compute_ext_range) below, which
 * mirrors adjoint_fix_prototype/proto_adjoint.py's `ext_slices()` `rng`/
 * `vec` helpers -- so double-accumulation against the interior loop's own
 * cells never occurs. Every axis argument here (including ones that happen
 * to be this call's "currently extended" axis, which is always non-PML by
 * construction) is treated fully generally: a transverse axis can
 * legitimately be PML-active even while another axis of the same cell is in
 * that axis's guard ring (e.g. boundary=["free","rigid","pml","pml"]: Y is
 * guard-extended, X has a real PML region the Y-guard-band loop still
 * sweeps through), so is_pml_z/is_pml_y/is_pml_x are ordinary parameters
 * here exactly as in the interior functions, and the PML memory-variable
 * read/write these functions perform is left completely intact (a sibling
 * field's OWN guard-band gather -- e.g. sigmaxy's, which also extends into
 * the same Y guard ring -- may need to read the M_SIGMA** / M_V** value a
 * velocity/stress guard gather wrote there this same time step).
 */

/* Guard-extension bookkeeping for one field's use of one axis. `n` is that
 * axis's full length (nz/ny/nx); `bc_lo`/`bc_hi` are that axis's two face
 * codes (0 = pml, 1 = free, 2 = rigid, 3 = periodic); `is_half` selects the
 * half-grid convention (velocity's own axis; a shear stress's two own axes)
 * vs full-grid (everything else -- see the grid-layout comment atop this
 * file). For an axis this field does not touch at all, callers pass
 * `bc_lo=bc_hi=0` regardless of that axis's real face codes, so `target`
 * degenerates to `std` and the new-cell test below is trivially false
 * everywhere on that axis.
 *
 * `target_lo/target_hi` is the field's full boundary-fill-aware target
 * range on this axis (mirrors `rng()` in the prototype); `std_lo/std_hi` is
 * the existing interior/PML loop's own (unextended) range on this axis
 * (matches pml_bounds_z/pml_bounds_zh etc.'s own [FD_PAD, n-FD_PAD+1) /
 * [FD_PAD, n-FD_PAD) span) -- cells with EVERY touched axis inside `std` are
 * already handled by the interior loop and must be skipped -- callers do
 * this with FUNC(dw_in_target) AND NOT-ALL-axes-FUNC(dw_in_std) below.
 * `has_exclude`/`exclude_idx` implements the
 * half-lattice "extra node" (`n - FD_PAD`) rule: under a half-grid axis's
 * free or rigid high face that node is excluded from the target set (a live
 * untouched DOF under free; a clamped, fold-zeroed DOF under rigid -- either
 * way not part of the new-cell set), included under periodic (the slaved
 * node the fold-fill's `bc_hi==3` step folds and zeros) -- see
 * `ext_slices()`'s own `vec()` docstring in the prototype.
 */
typedef struct {
  int64_t target_lo, target_hi;
  int64_t std_lo, std_hi;
  bool has_exclude;
  int64_t exclude_idx;
} FUNC(dw_ext_range);

DW_HOST_DEVICE FUNC(dw_ext_range)
    FUNC(dw_compute_ext_range)(int64_t const n, int64_t const bc_lo,
                               int64_t const bc_hi, bool const is_half) {
  FUNC(dw_ext_range) r;
  r.std_lo = FD_PAD;
  r.std_hi = is_half ? (n - FD_PAD) : (n - FD_PAD + 1);
  r.target_lo = bc_lo != 0 ? 0 : r.std_lo;
  r.target_hi = bc_hi != 0 ? n : r.std_hi;
  r.exclude_idx = n - FD_PAD;
  r.has_exclude = is_half && (bc_hi == 1 || bc_hi == 2);
  return r;
}

/* True iff coordinate `c` is in this field-axis's boundary-fill-aware
 * target set (see FUNC(dw_compute_ext_range)'s comment). */
DW_HOST_DEVICE bool FUNC(dw_in_target)(FUNC(dw_ext_range) const r,
                                       int64_t const c) {
  if (c < r.target_lo || c >= r.target_hi) return false;
  if (r.has_exclude && c == r.exclude_idx) return false;
  return true;
}

/* True iff coordinate `c` is in this field-axis's existing (unextended)
 * interior/PML loop range -- i.e. already handled by the interior gather. */
DW_HOST_DEVICE bool FUNC(dw_in_std)(FUNC(dw_ext_range) const r,
                                    int64_t const c) {
  return c >= r.std_lo && c < r.std_hi;
}

/* Whether spatial coordinate `c` on one axis is inside that axis's PML
 * sub-region, replicating elastic.c's/elastic.cu's own region split
 * (pml_bounds_* / pml_bounds_*h: region 0 = [FD_PAD, pml_lo), region 2 =
 * [pml_hi, std_hi)) so a guard-band gather agrees with the interior
 * gather's own treatment of any cell both loops cover -- in particular the
 * single boundary-adjacent cell that a genuinely reflecting face's
 * `pml_hi == std_hi - 1` (a reflecting face always has zero PML width, so
 * `pml_hi` sits exactly one cell inside the array edge -- see
 * `_common.py`'s `set_boundary`) still labels as region 2/is_pml, relying
 * on the PML profile arrays being trivial (a=1, b=0) there. Cells strictly
 * inside the guard ring (c < FD_PAD or c >= std_hi) always evaluate false
 * here, since `pml_lo`/`pml_hi` never reach past FD_PAD/std_hi on a
 * genuinely reflecting face -- so this never reads a PML profile array
 * out of bounds either. `pml_hi` must be pre-adjusted by the caller for a
 * half-grid axis (`MAX(pml_lo, pml_hi_raw - 1)`, matching pml_bounds_*h's
 * own construction) before being passed in here; `std_hi` should be the
 * same-role FUNC(dw_ext_range)'s `std_hi` (so the half/full adjustment
 * stays consistent between the two). */
DW_HOST_DEVICE bool FUNC(dw_is_pml_region)(int64_t const c,
                                           int64_t const pml_lo,
                                           int64_t const pml_hi,
                                           int64_t const std_hi) {
  return (c >= FD_PAD && c < pml_lo) || (c >= pml_hi && c < std_hi);
}

#if DW_ACCURACY == 2
#define DW_FD_WEIGHT0 ((DW_DTYPE)1.0)
#elif DW_ACCURACY == 4
#define DW_FD_WEIGHT0 ((DW_DTYPE)(9.0 / 8.0))
#define DW_FD_WEIGHT1 ((DW_DTYPE)(-1.0 / 24.0))
#elif DW_ACCURACY == 6
#define DW_FD_WEIGHT0 ((DW_DTYPE)(75.0 / 64.0))
#define DW_FD_WEIGHT1 ((DW_DTYPE)(-25.0 / 384.0))
#define DW_FD_WEIGHT2 ((DW_DTYPE)(3.0 / 640.0))
#elif DW_ACCURACY == 8
#define DW_FD_WEIGHT0 ((DW_DTYPE)(1225.0 / 1024.0))
#define DW_FD_WEIGHT1 ((DW_DTYPE)(-245.0 / 3072.0))
#define DW_FD_WEIGHT2 ((DW_DTYPE)(49.0 / 5120.0))
#define DW_FD_WEIGHT3 ((DW_DTYPE)(-5.0 / 7168.0))
#endif

/* Bounds-safe single-tap read of compound field-access macro `a` (e.g.
 * VZ_Z, SIGMAXY_X_PML) at this axis's offset `d` from the current cell: `a`
 * itself (which may read model/wavefield/PML-memory arrays at that offset,
 * all at once, via the ND_INDEX-based macros above) is only evaluated -- so
 * only ever dereferenced -- when the offset cell is inside the array; a
 * tap landing outside contributes exactly zero, matching the prototype's
 * zero-padded `d1_full`/`d1h_full`. */
#if DW_NDIM >= 3
#define DW_SAFE_TAP_Z(a, d) \
  ((z + (d) >= 0 && z + (d) < nz) ? a(d, 0, 0) : (DW_DTYPE)0)
#endif
#if DW_NDIM >= 2
#define DW_SAFE_TAP_Y(a, d) \
  ((y + (d) >= 0 && y + (d) < ny) ? a(0, d, 0) : (DW_DTYPE)0)
#endif
#define DW_SAFE_TAP_X(a, d) \
  ((x + (d) >= 0 && x + (d) < nx) ? a(0, 0, d) : (DW_DTYPE)0)

/* Bounds-safe weighted sum reproducing DIFF*1/DIFF*H1's own structure
 * (staggered_grid.h) via the general identity DIFF1(a) = rd * sum_k
 * w[k]*(a(k) - a(-k-1)), DIFFH1(a) = rd * sum_k w[k]*(a(k+1) - a(-k)), k =
 * 0..FD_PAD-1 (`s` is 0 for the *1 family, 1 for the *H1 family) -- the
 * Taylor coefficients above satisfy this identity by construction, so
 * DW_DIFF*_SAFE below agrees bit-for-bit with DIFF*1/DIFF*H1 whenever every
 * tap is in bounds. elastic_kernels.h may not modify staggered_grid.h
 * (out of scope), hence the duplication here. */
#if FD_PAD == 1
#define DW_SAFE_SUM(TAP, a, s) \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))))
#elif FD_PAD == 2
#define DW_SAFE_SUM(TAP, a, s)                                \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))) + \
   DW_FD_WEIGHT1 * (TAP(a, (1 + (s))) - TAP(a, (-2 + (s)))))
#elif FD_PAD == 3
#define DW_SAFE_SUM(TAP, a, s)                                \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))) + \
   DW_FD_WEIGHT1 * (TAP(a, (1 + (s))) - TAP(a, (-2 + (s)))) + \
   DW_FD_WEIGHT2 * (TAP(a, (2 + (s))) - TAP(a, (-3 + (s)))))
#elif FD_PAD == 4
#define DW_SAFE_SUM(TAP, a, s)                                \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))) + \
   DW_FD_WEIGHT1 * (TAP(a, (1 + (s))) - TAP(a, (-2 + (s)))) + \
   DW_FD_WEIGHT2 * (TAP(a, (2 + (s))) - TAP(a, (-3 + (s)))) + \
   DW_FD_WEIGHT3 * (TAP(a, (3 + (s))) - TAP(a, (-4 + (s)))))
#endif

#if DW_NDIM >= 3
#define DW_DIFFZ1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Z, a, 0) * rdz)
#define DW_DIFFZH1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Z, a, 1) * rdz)
#endif
#if DW_NDIM >= 2
#define DW_DIFFY1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Y, a, 0) * rdy)
#define DW_DIFFYH1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Y, a, 1) * rdy)
#endif
#define DW_DIFFX1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_X, a, 0) * rdx)
#define DW_DIFFXH1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_X, a, 1) * rdx)

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_vz_cell)'s gather (no buoyancy-grad
 * accumulation -- see the module comment above). `nz` is required here
 * (unlike FUNC(backward_vz_cell), which never needs it for flat indexing)
 * purely for DW_SAFE_TAP_Z's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_guard)(
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
    DW_DTYPE const* __restrict const mu_zy_shot,
    DW_DTYPE const* __restrict const mu_zx_shot,
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz, DW_DTYPE* __restrict const vz,
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmazzz_old,
    DW_DTYPE* __restrict const m_sigmazzz_new,
    DW_DTYPE const* __restrict const m_sigmayzy_old,
    DW_DTYPE* __restrict const m_sigmayzy_new,
    DW_DTYPE const* __restrict const m_sigmaxzx_old,
    DW_DTYPE* __restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdz, DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  w_sum += pml_z ? -DW_DIFFZH1_SAFE(VZ_Z_PML) : -DW_DIFFZH1_SAFE(VZ_Z);
  w_sum += pml_y ? -DW_DIFFY1_SAFE(VZ_Y_PML) : -DW_DIFFY1_SAFE(VZ_Y);
  w_sum += pml_x ? -DW_DIFFX1_SAFE(VZ_X_PML) : -DW_DIFFX1_SAFE(VZ_X);

  vz[i] += w_sum;

  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz_old[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy_old[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx_old[i];
  }
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_vy_cell)'s gather (see
 * FUNC(backward_vz_cell_guard)). `ny` is required unconditionally here
 * (unlike FUNC(backward_vy_cell), which only needs it for ND_INDEX when
 * DW_NDIM==3) purely for DW_SAFE_TAP_Y's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vy_cell_guard)(
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy_shot,
#endif
    DW_DTYPE const* __restrict const mu_yx_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmayz,
#endif
    DW_DTYPE const* __restrict const sigmaxy, DW_DTYPE* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vyz,
#endif
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmayzz_old,
    DW_DTYPE* __restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const* __restrict const m_sigmayyy_old,
    DW_DTYPE* __restrict const m_sigmayyy_new,
    DW_DTYPE const* __restrict const m_sigmaxyx_old,
    DW_DTYPE* __restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
    int64_t const ny, int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
    int64_t const y, int64_t const x, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DW_DIFFZ1_SAFE(VY_Z_PML) : -DW_DIFFZ1_SAFE(VY_Z);
#endif
  w_sum += pml_y ? -DW_DIFFYH1_SAFE(VY_Y_PML) : -DW_DIFFYH1_SAFE(VY_Y);
  w_sum += pml_x ? -DW_DIFFX1_SAFE(VY_X_PML) : -DW_DIFFX1_SAFE(VY_X);

  vy[i] += w_sum;

#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz_old[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy_old[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx_old[i];
  }
}
#endif

/* Guard-band companion to FUNC(backward_vx_cell)'s gather (see
 * FUNC(backward_vz_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_guard)(
    DW_DTYPE const* __restrict const lamb_shot,
    DW_DTYPE const* __restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vxy,
#endif
    DW_DTYPE const* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz_old,
    DW_DTYPE* __restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy_old,
    DW_DTYPE* __restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx_old,
    DW_DTYPE* __restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DW_DIFFZ1_SAFE(VX_Z_PML) : -DW_DIFFZ1_SAFE(VX_Z);
#endif
#if DW_NDIM >= 2
  w_sum += pml_y ? -DW_DIFFY1_SAFE(VX_Y_PML) : -DW_DIFFY1_SAFE(VX_Y);
#endif
  w_sum += pml_x ? -DW_DIFFXH1_SAFE(VX_X_PML) : -DW_DIFFXH1_SAFE(VX_X);

  vx[i] += w_sum;

#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmaxzz_new[i] =
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz_old[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy_old[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx_old[i];
  }
}

#if DW_NDIM >= 3
/* Guard-band companion to the sigmazz sub-update inside FUNC(backward_
 * sigmaii_cell): sigmazz's only term is its own (z) axis (SIGMAZZ_Z/
 * SIGMAZZ_Z_PML), so a guard cell here -- which by construction only
 * exists on an axis with a non-PML face -- is never inside a PML region on
 * that same axis; the M_VZZ PML-memory write the interior function
 * performs (gated by the identical, always-false-here condition) is
 * therefore unreachable and intentionally omitted, and grad_lamb/grad_mu
 * (also computed there, using sigmazz's value at the *old* cell, not the
 * new guard cell) is left entirely to the interior function -- see the
 * module comment above. */
DW_HOST_DEVICE void FUNC(backward_sigmazz_z_cell_guard)(
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, DW_DTYPE const rdz) {
  sigmazz[i] += -DW_DIFFZ1_SAFE(SIGMAZZ_Z);
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to the sigmayy sub-update inside FUNC(backward_
 * sigmaii_cell) (see FUNC(backward_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmayy_y_cell_guard)(
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const y, DW_DTYPE const rdy) {
  sigmayy[i] += -DW_DIFFY1_SAFE(SIGMAYY_Y);
}
#endif

/* Guard-band companion to the sigmaxx sub-update inside FUNC(backward_
 * sigmaii_cell) (see FUNC(backward_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxx_x_cell_guard)(
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const x, DW_DTYPE const rdx) {
  sigmaxx[i] += -DW_DIFFX1_SAFE(SIGMAXX_X);
}

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_sigmaxy_cell)'s gather (see
 * FUNC(backward_vz_cell_guard) -- like the velocity guards, this keeps the
 * PML memory-variable read/write intact since sigmaxy's guard ring may be
 * genuinely PML-active on its *other* axis). `ny` is required
 * unconditionally here (unlike FUNC(backward_sigmaxy_cell), which only
 * needs it for ND_INDEX when DW_NDIM==3) purely for DW_SAFE_TAP_Y's bounds
 * check. */
DW_HOST_DEVICE void FUNC(backward_sigmaxy_cell_guard)(
    DW_DTYPE const* __restrict const mu_yx_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx, DW_DTYPE const dt,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz, int64_t const z,
#endif
    int64_t const ny, int64_t const nx, int64_t const y, int64_t const x,
    bool const pml_y, bool const pml_x, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (pml_y) {
    m_vxy[i] =
        MU_YX(0, 0, 0) * dt * P_AYH(0) * sigmaxy[i] + P_AYH(0) * m_vxy[i];
  }
  if (pml_x) {
    m_vyx[i] =
        MU_YX(0, 0, 0) * dt * P_AXH(0) * sigmaxy[i] + P_AXH(0) * m_vyx[i];
  }

  sigmaxy[i] +=
      (pml_y ? -DW_DIFFYH1_SAFE(SIGMAXY_Y_PML) : -DW_DIFFYH1_SAFE(SIGMAXY_Y)) +
      (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXY_X_PML) : -DW_DIFFXH1_SAFE(SIGMAXY_X));
}
#endif

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_sigmaxz_cell)'s gather (see
 * FUNC(backward_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxz_cell_guard)(
    DW_DTYPE const* __restrict const mu_zx_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const m_vxz,
    DW_DTYPE* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzx, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_x, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdz,
    DW_DTYPE const rdx) {
  if (pml_z) {
    m_vxz[i] =
        MU_ZX(0, 0, 0) * dt * P_AZH(0) * sigmaxz[i] + P_AZH(0) * m_vxz[i];
  }
  if (pml_x) {
    m_vzx[i] =
        MU_ZX(0, 0, 0) * dt * P_AXH(0) * sigmaxz[i] + P_AXH(0) * m_vzx[i];
  }

  sigmaxz[i] +=
      (pml_z ? -DW_DIFFZH1_SAFE(SIGMAXZ_Z_PML) : -DW_DIFFZH1_SAFE(SIGMAXZ_Z)) +
      (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXZ_X_PML) : -DW_DIFFXH1_SAFE(SIGMAXZ_X));
}

/* Guard-band companion to FUNC(backward_sigmayz_cell)'s gather (see
 * FUNC(backward_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmayz_cell_guard)(
    DW_DTYPE const* __restrict const mu_zy_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzy, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const rdz,
    DW_DTYPE const rdy) {
  if (pml_z) {
    m_vyz[i] =
        MU_ZY(0, 0, 0) * dt * P_AZH(0) * sigmayz[i] + P_AZH(0) * m_vyz[i];
  }
  if (pml_y) {
    m_vzy[i] =
        MU_ZY(0, 0, 0) * dt * P_AYH(0) * sigmayz[i] + P_AYH(0) * m_vzy[i];
  }

  sigmayz[i] +=
      (pml_z ? -DW_DIFFZH1_SAFE(SIGMAYZ_Z_PML) : -DW_DIFFZH1_SAFE(SIGMAYZ_Z)) +
      (pml_y ? -DW_DIFFYH1_SAFE(SIGMAYZ_Y_PML) : -DW_DIFFYH1_SAFE(SIGMAYZ_Y));
}
#endif

#endif /* DW_ELASTIC_KERNELS_H */

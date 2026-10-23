#ifndef DW_ELASTIC_BORN_KERNELS_H
#define DW_ELASTIC_BORN_KERNELS_H

/*
 * elastic_born_kernels.h
 *
 * Single-source per-grid-point stencil math for the elastic Born
 * (linearised) velocity-stress propagator (see elastic_born.c for the
 * physics/PML/staggered-grid-layout description). This header is included by
 * BOTH elastic_born.c (compiled by gcc) and elastic_born.cu (compiled by
 * nvcc), after each has defined the FUNC()/CAT() name-mangling macros (and
 * the DW_MPML-dependent ELASTIC_VARIANT()) and before either uses the
 * ND_INDEX/VZ/VY/VX/SIGMA(*)/LAMB/MU/... family below.
 *
 * elastic_born combines the two characteristics already handled separately by
 * elastic_kernels.h (full velocity-stress: three independent per-axis
 * velocity updates plus one combined normal-stress update plus three
 * independent shear-stress updates) and acoustic_born_kernels.h /
 * scalar_born_kernels.h (a background field coupled to a scattered field,
 * with a background-only forward, scattered-only adjoint, and combined
 * two-field adjoint). Crossing those two axes gives 7 functions per
 * top-level entry point (3 per-axis velocity + 1 combined normal-stress + 3
 * independent shear-stress), times 3 entry points, for 21 functions total:
 *
 *  * FUNC(forward_vz_cell)/vy_cell/vx_cell, FUNC(forward_sigmaii_cell),
 *    FUNC(forward_sigmaxy_cell)/sigmaxz_cell/sigmayz_cell: advance BOTH the
 *    background and scattered fields for one axis (velocity) or component
 *    (stress) in one call, matching elastic_born.c's/.cu's single fused
 *    background-then-scattered loop body per axis/component (the scattered
 *    update at each point directly reads the just-computed background term
 *    at that SAME point -- see the doc comment in elastic_born.c).
 *  * FUNC(backward_sc_vz_cell)/vy_cell/vx_cell,
 *    FUNC(backward_sc_sigmaii_cell)/sigmaxy_cell/sigmaxz_cell/sigmayz_cell:
 *    the scattered-only adjoint (fixed background lamb/mu/mu_ij/buoyancy_i)
 *    -- the ORDINARY, unperturbed elastic adjoint recursion (textually the
 *    same shape as elastic_kernels.h's backward_*_cell family, applied to the
 *    scattered adjoint field) that also accumulates
 *    grad_dlamb/grad_dmu/grad_dmu_ij/grad_dbuoyancy_i by correlating the
 *    propagating scattered adjoint field with the checkpointed background
 *    terms. Used when only the scattering models (dlamb, dmu, dmu_ij,
 *    dbuoyancy_i) require a gradient, so the background adjoint field need
 *    not be propagated at all (the same optimisation
 *    acoustic_born_kernels.h's FUNC(backward_sc_*_cell) makes for its two
 *    fields).
 *  * FUNC(backward_vz_cell)/vy_cell/vx_cell,
 *    FUNC(backward_sigmaii_cell)/sigmaxy_cell/sigmaxz_cell/sigmayz_cell: the
 *    full two-field adjoint -- propagates the background adjoint field with
 *    the COUPLED adjoint (every lamb*P_adj becomes lamb*P_adj + dlamb*P_adj_sc,
 *    every buoyancy*V_adj becomes buoyancy*V_adj + dbuoyancy*V_adj_sc, etc.)
 *    AND the scattered adjoint field with the ordinary adjoint (as
 *    backward_sc), in the same call, accumulating
 *    grad_lamb/grad_mu/grad_mu_ij/grad_buoyancy_i (two correlation terms
 *    each) and grad_dlamb/grad_dmu/grad_dmu_ij/grad_dbuoyancy_i (one term
 *    each).
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest and PML-region triple loop in
 * elastic_born.c, the CUDA kernel launch configuration and per-thread
 * grid-point assignment in elastic_born.cu -- stays in elastic_born.c/
 * elastic_born.cu exactly as before (per the common_cpu.h/common_gpu.h
 * split); only the math performed once the caller has identified "this" grid
 * point lives here.
 *
 * Addressing convention (matches elastic_kernels.h/acoustic_born_kernels.h
 * exactly): the background/scattered/background-adjoint wavefields
 * (vz/vy/vx/sigma(*)/their _sc and _bg twins), the M_SIGMA** / M_V** PML memory
 * families (and their _bg twins), and the term stores (*_store_1 /
 * *_sc_store_1)
 * are addressed by `i` (the CPU driver's per-shot-local flat index, which is
 * also valid as the CUDA driver's global shot-spanning index since both
 * drivers pass already-appropriately-offset pointers); lamb_shot/mu_shot/
 * mu_*_shot/buoyancy_*_shot/dlamb_shot/dmu_shot/dmu_*_shot/dbuoyancy_*_shot
 * are addressed by `j` (the per-shot-local index) wherever a POINTER is
 * taken (used only by the backward functions), because the CUDA driver's
 * model pointers are shot-local (offset by shot_idx*shot_numel, not `i`)
 * while the wavefield arrays span all shots. On the CPU driver, wavefields
 * and model arrays are ALL already offset to the current shot, so `i` and
 * `j` are the same value there and the CPU driver simply passes `i` for
 * both. The forward functions and the VALUE arguments of the backward
 * functions need no `j`: they only ever read the centre value of
 * lamb/mu/buoyancy/dlamb/dmu/dbuoyancy (no neighbour offset), so the caller
 * passes that value directly, by value, exactly as elastic_kernels.h's
 * forward cells / acoustic_born_kernels.h's backward cells do.
 *
 * M-PML support: the P_A(*)/P_B(*) PML-profile accessor macros below are
 * lifted verbatim from elastic_kernels.h (see that header's comment for the
 * full DW_MPML / PML_IDX explanation); the driver-side DW_PML_SKIP
 * region-skip decision is NOT part of this header (driver-side scaffolding,
 * kept in elastic_born.c/elastic_born.cu exactly as elastic_kernels.h
 * established), so both drivers keep computing their own
 * `pml_z`/`pml_y`/`pml_x` bool and simply pass the resulting bool into the
 * functions below.
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
#define ND_INDEX(i, dz, dy, dx) (i + (dz) * ny * nx + (dy) * nx + (dx))
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy) * nx + (dx))
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#endif

/* PML profile access (see elastic_kernels.h for the full explanation). */
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

/* Background-forward / scattered-adjoint wavefields, addressed by i (see the
 * addressing-convention note above). Reused across entry points:
 * FUNC(forward_*) reads/writes these as the BACKGROUND forward field;
 * FUNC(backward_sc_*) and FUNC(backward_*) read/write these as the SCATTERED
 * adjoint field (matching elastic_born.c's/.cu's own reuse of the plain
 * `vz`/`sigmazz`/... buffers for both roles across the file, since they are
 * never needed simultaneously). */
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

/* Scattered-forward wavefields, addressed by i. Used only by FUNC(forward_*).
 */
#if DW_NDIM >= 3
#define VZ_SC(dz, dy, dx) vz_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY_SC(dz, dy, dx) vy_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX_SC(dz, dy, dx) vx_sc[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define SIGMAZZ_SC(dz, dy, dx) sigmazz_sc[ND_INDEX(i, dz, dy, dx)]
#elif DW_NDIM == 2
#define SIGMAZZ_SC(dz, dy, dx) 0
#endif
#if DW_NDIM >= 2
#define SIGMAYY_SC(dz, dy, dx) sigmayy_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#define SIGMAXX_SC(dz, dy, dx) sigmaxx_sc[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define SIGMAXY_SC(dz, dy, dx) sigmaxy_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define SIGMAXZ_SC(dz, dy, dx) sigmaxz_sc[ND_INDEX(i, dz, dy, dx)]
#define SIGMAYZ_SC(dz, dy, dx) sigmayz_sc[ND_INDEX(i, dz, dy, dx)]
#endif

/* Background-adjoint wavefields, addressed by i. Used only by FUNC(backward_*)
 * (the full two-field adjoint): the ordinary adjoint macros below (VZ_Z etc.)
 * operate on the plain (scattered-adjoint) buffers above; these _BG twins let
 * the FUNC(backward_*_cell) family also propagate the background adjoint
 * field, coupled via the DLAMB/DMU/DBUOYANCY_* perturbation terms. */
#if DW_NDIM >= 3
#define VZ_BG(dz, dy, dx) vz_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY_BG(dz, dy, dx) vy_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX_BG(dz, dy, dx) vx_bg[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define SIGMAZZ_BG(dz, dy, dx) sigmazz_bg[ND_INDEX(i, dz, dy, dx)]
#elif DW_NDIM == 2
#define SIGMAZZ_BG(dz, dy, dx) 0
#endif
#if DW_NDIM >= 2
#define SIGMAYY_BG(dz, dy, dx) sigmayy_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#define SIGMAXX_BG(dz, dy, dx) sigmaxx_bg[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define SIGMAXY_BG(dz, dy, dx) sigmaxy_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define SIGMAXZ_BG(dz, dy, dx) sigmaxz_bg[ND_INDEX(i, dz, dy, dx)]
#define SIGMAYZ_BG(dz, dy, dx) sigmayz_bg[ND_INDEX(i, dz, dy, dx)]
#endif

/* Models (background, addressed by the shot-local index j -- see the
 * addressing-convention note above). Used only by the backward functions. */
#define LAMB(dz, dy, dx) lamb_shot[ND_INDEX(j, dz, dy, dx)]
#define MU(dz, dy, dx) mu_shot[ND_INDEX(j, dz, dy, dx)]
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

/* Models (scattering perturbations, addressed by j). Used only by the full
 * two-field backward's BG-coupled macros below. */
#define DLAMB(dz, dy, dx) dlamb_shot[ND_INDEX(j, dz, dy, dx)]
#define DMU(dz, dy, dx) dmu_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 2
#define DMU_YX(dz, dy, dx) dmu_yx_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define DMU_ZX(dz, dy, dx) dmu_zx_shot[ND_INDEX(j, dz, dy, dx)]
#define DMU_ZY(dz, dy, dx) dmu_zy_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define DBUOYANCY_Z(dz, dy, dx) dbuoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define DBUOYANCY_Y(dz, dy, dx) dbuoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define DBUOYANCY_X(dz, dy, dx) dbuoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

/* PML memory variables for the velocity equations (the "M_SIGMA**" family --
 * see elastic_kernels.h's ping-pong-buffering note): written (centre point
 * only, via the _old/_new parameter pair) by FUNC(backward_sc_v*_cell)/
 * FUNC(backward_v*_cell), read (with a neighbour offset, via these macros,
 * through the plain "m_sigma**" parameter -- the OLD value) by
 * FUNC(backward_sc_sigma*_cell)/FUNC(backward_sigma*_cell). */
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

/* Background-adjoint twins of the M_SIGMA** family above (full two-field
 * backward only). */
#if DW_NDIM >= 2
#define M_SIGMAXYY_BG(dz, dy, dx) m_sigmaxyy_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_SIGMAXXX_BG(dz, dy, dx) m_sigmaxxx_bg[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define M_SIGMAXZZ_BG(dz, dy, dx) m_sigmaxzz_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define M_SIGMAXYX_BG(dz, dy, dx) m_sigmaxyx_bg[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYYY_BG(dz, dy, dx) m_sigmayyy_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_SIGMAYZZ_BG(dz, dy, dx) m_sigmayzz_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_SIGMAXZX_BG(dz, dy, dx) m_sigmaxzx_bg[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYZY_BG(dz, dy, dx) m_sigmayzy_bg[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAZZZ_BG(dz, dy, dx) m_sigmazzz_bg[ND_INDEX(i, dz, dy, dx)]
#endif

/* PML memory variables for the stress equations (the "M_V**" family -- read
 * with a neighbour offset by FUNC(backward_sc_v*_cell)/FUNC(backward_v*_cell),
 * written centre-point-only by FUNC(backward_sc_sigma*_cell)/
 * FUNC(backward_sigma*_cell)). */
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

/* Background-adjoint twins of the M_V** family above (full two-field backward
 * only). */
#if DW_NDIM >= 2
#define M_VXY_BG(dz, dy, dx) m_vxy_bg[ND_INDEX(i, dz, dy, dx)]
#define M_VYX_BG(dz, dy, dx) m_vyx_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_VXX_BG(dz, dy, dx) m_vxx_bg[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define M_VYY_BG(dz, dy, dx) m_vyy_bg[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_VZZ_BG(dz, dy, dx) m_vzz_bg[ND_INDEX(i, dz, dy, dx)]
#define M_VXZ_BG(dz, dy, dx) m_vxz_bg[ND_INDEX(i, dz, dy, dx)]
#define M_VZX_BG(dz, dy, dx) m_vzx_bg[ND_INDEX(i, dz, dy, dx)]
#define M_VYZ_BG(dz, dy, dx) m_vyz_bg[ND_INDEX(i, dz, dy, dx)]
#define M_VZY_BG(dz, dy, dx) m_vzy_bg[ND_INDEX(i, dz, dy, dx)]
#endif

/* Adjoint-term macros used by the ordinary (scattered-field) adjoint velocity
 * update (FUNC(backward_sc_v*_cell)/FUNC(backward_v*_cell)'s plain half): the
 * transpose of the forward stress equations, identical to elastic_kernels.h's
 * copy of the same macros. */
#define LAMB_2MU(dz, dy, dx) (LAMB(dz, dy, dx) + 2 * MU(dz, dy, dx))
#define DLAMB_2MU(dz, dy, dx) (DLAMB(dz, dy, dx) + 2 * DMU(dz, dy, dx))

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

/* Adjoint-term macros used by the ordinary (scattered-field) adjoint stress
 * update (FUNC(backward_sc_sigma*_cell)/FUNC(backward_sigma*_cell)'s plain
 * half): the transpose of the forward velocity equations. */
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

/* --- Background-adjoint (BG) coupled macros: the ordinary adjoint macros
 * above, operating on the BACKGROUND adjoint field (the _BG wavefields),
 * with the Born coupling term added (every lamb*P_adj/mu*P_adj becomes
 * lamb*P_adj + dlamb*P_adj_sc / mu*P_adj + dmu*P_adj_sc, every
 * buoyancy*V_adj becomes buoyancy*V_adj + dbuoyancy*V_adj_sc). The plain
 * (non-_BG) SIGMA* / VZ/VY/VX here are the SCATTERED adjoint field. Used only
 * by the full two-field FUNC(backward_*_cell) family. --- */
#if DW_NDIM >= 3
#define VZ_Z_BG(dz, dy, dx)                               \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) +      \
         LAMB(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx) +      \
         LAMB_2MU(dz, dy, dx) * SIGMAZZ_BG(dz, dy, dx)) + \
   dt * (DLAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +        \
         DLAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +        \
         DLAMB_2MU(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#define VZ_Y_BG(dz, dy, dx)                          \
  (dt * MU_ZY(dz, dy, dx) * SIGMAYZ_BG(dz, dy, dx) + \
   dt * DMU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#define VZ_X_BG(dz, dy, dx)                          \
  (dt * MU_ZX(dz, dy, dx) * SIGMAXZ_BG(dz, dy, dx) + \
   dt * DMU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#define VZ_Z_BG_PML(dz, dy, dx)                           \
  (dt * (1 + P_BZ(dz)) *                                  \
       ((LAMB(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) +      \
         LAMB(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx) +      \
         LAMB_2MU(dz, dy, dx) * SIGMAZZ_BG(dz, dy, dx)) + \
        (DLAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +        \
         DLAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +        \
         DLAMB_2MU(dz, dy, dx) * SIGMAZZ(dz, dy, dx))) +  \
   P_BZ(dz) * M_VZZ_BG(dz, dy, dx))
#define VZ_Y_BG_PML(dz, dy, dx)                      \
  (dt * (1 + P_BYH(dy)) *                            \
       (MU_ZY(dz, dy, dx) * SIGMAYZ_BG(dz, dy, dx) + \
        DMU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx)) +  \
   P_BYH(dy) * M_VZY_BG(dz, dy, dx))
#define VZ_X_BG_PML(dz, dy, dx)                      \
  (dt * (1 + P_BXH(dx)) *                            \
       (MU_ZX(dz, dy, dx) * SIGMAXZ_BG(dz, dy, dx) + \
        DMU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx)) +  \
   P_BXH(dx) * M_VZX_BG(dz, dy, dx))
#endif

#if DW_NDIM >= 2
#if DW_NDIM >= 3
#define VY_Z_BG(dz, dy, dx)                          \
  (dt * MU_ZY(dz, dy, dx) * SIGMAYZ_BG(dz, dy, dx) + \
   dt * DMU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y_BG(dz, dy, dx)                               \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) +      \
         LAMB_2MU(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx)) + \
   dt * (DLAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +        \
         DLAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#else
#define VY_Y_BG(dz, dy, dx)                              \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) +     \
         LAMB_2MU(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAZZ_BG(dz, dy, dx)) +    \
   dt * (DLAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +       \
         DLAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx) +   \
         DLAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#define VY_X_BG(dz, dy, dx)                          \
  (dt * MU_YX(dz, dy, dx) * SIGMAXY_BG(dz, dy, dx) + \
   dt * DMU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#if DW_NDIM >= 3
#define VY_Z_BG_PML(dz, dy, dx)                      \
  (dt * (1 + P_BZH(dz)) *                            \
       (MU_ZY(dz, dy, dx) * SIGMAYZ_BG(dz, dy, dx) + \
        DMU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx)) +  \
   P_BZH(dz) * M_VYZ_BG(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y_BG_PML(dz, dy, dx)                           \
  (dt * (1 + P_BY(dy)) *                                  \
       ((LAMB(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) +      \
         LAMB_2MU(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx)) + \
        (DLAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +        \
         DLAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx))) +  \
   P_BY(dy) * M_VYY_BG(dz, dy, dx))
#else
#define VY_Y_BG_PML(dz, dy, dx)                          \
  (dt * (1 + P_BY(dy)) *                                 \
       ((LAMB(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) +     \
         LAMB_2MU(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAZZ_BG(dz, dy, dx)) +    \
        (DLAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +       \
         DLAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx) +   \
         DLAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx))) +     \
   P_BY(dy) * M_VYY_BG(dz, dy, dx))
#endif
#define VY_X_BG_PML(dz, dy, dx)                      \
  (dt * (1 + P_BXH(dx)) *                            \
       (MU_YX(dz, dy, dx) * SIGMAXY_BG(dz, dy, dx) + \
        DMU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx)) +  \
   P_BXH(dx) * M_VYX_BG(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define VX_Z_BG(dz, dy, dx)                          \
  (dt * MU_ZX(dz, dy, dx) * SIGMAXZ_BG(dz, dy, dx) + \
   dt * DMU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y_BG(dz, dy, dx)                          \
  (dt * MU_YX(dz, dy, dx) * SIGMAXY_BG(dz, dy, dx) + \
   dt * DMU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X_BG(dz, dy, dx)                         \
  (dt * LAMB_2MU(0, 0, dx) * SIGMAXX_BG(0, 0, dx) + \
   dt * DLAMB_2MU(0, 0, dx) * SIGMAXX(0, 0, dx))
#else
#define VX_X_BG(dz, dy, dx)                              \
  (dt * (LAMB_2MU(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx) +     \
         LAMB(dz, dy, dx) * SIGMAZZ_BG(dz, dy, dx)) +    \
   dt * (DLAMB_2MU(dz, dy, dx) * SIGMAXX(dz, dy, dx) +   \
         DLAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +       \
         DLAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#if DW_NDIM >= 3
#define VX_Z_BG_PML(dz, dy, dx)                      \
  (dt * (1 + P_BZH(dz)) *                            \
       (MU_ZX(dz, dy, dx) * SIGMAXZ_BG(dz, dy, dx) + \
        DMU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx)) +  \
   P_BZH(dz) * M_VXZ_BG(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y_BG_PML(dz, dy, dx)                      \
  (dt * (1 + P_BYH(dy)) *                            \
       (MU_YX(dz, dy, dx) * SIGMAXY_BG(dz, dy, dx) + \
        DMU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx)) +  \
   P_BYH(dy) * M_VXY_BG(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X_BG_PML(dz, dy, dx)                     \
  (dt * (1 + P_BX(dx)) *                            \
       (LAMB_2MU(0, 0, dx) * SIGMAXX_BG(0, 0, dx) + \
        DLAMB_2MU(0, 0, dx) * SIGMAXX(0, 0, dx)) +  \
   P_BX(dx) * M_VXX_BG(0, 0, dx))
#else
#define VX_X_BG_PML(dz, dy, dx)                          \
  (dt * (1 + P_BX(dx)) *                                 \
       ((LAMB_2MU(dz, dy, dx) * SIGMAXX_BG(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAYY_BG(dz, dy, dx) +     \
         LAMB(dz, dy, dx) * SIGMAZZ_BG(dz, dy, dx)) +    \
        (DLAMB_2MU(dz, dy, dx) * SIGMAXX(dz, dy, dx) +   \
         DLAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +       \
         DLAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx))) +     \
   P_BX(dx) * M_VXX_BG(dz, dy, dx))
#endif

/* --- Background-adjoint (BG) coupled stress macros: V*_BG here are
 * background adjoint velocities; the plain V* are the scattered adjoint
 * (Q_sc). Used only by the full two-field FUNC(backward_sigma*_cell)
 * family. --- */
#if DW_NDIM >= 3
#define SIGMAZZ_Z_BG_PML(dz, dy, dx)                 \
  ((1 + P_BZH(dz)) * dt *                            \
       (BUOYANCY_Z(dz, dy, dx) * VZ_BG(dz, dy, dx) + \
        DBUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx)) +  \
   P_BZH(dz) * M_SIGMAZZZ_BG(dz, dy, dx))
#define SIGMAZZ_Z_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_Z(dz, dy, dx) * VZ_BG(dz, dy, dx) + \
         DBUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx)))
#endif
#if DW_NDIM >= 2
#define SIGMAYY_Y_BG_PML(dz, dy, dx)                 \
  ((1 + P_BYH(dy)) * dt *                            \
       (BUOYANCY_Y(dz, dy, dx) * VY_BG(dz, dy, dx) + \
        DBUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx)) +  \
   P_BYH(dy) * M_SIGMAYYY_BG(dz, dy, dx))
#define SIGMAYY_Y_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_Y(dz, dy, dx) * VY_BG(dz, dy, dx) + \
         DBUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx)))
#endif
#define SIGMAXX_X_BG_PML(dz, dy, dx)                 \
  ((1 + P_BXH(dx)) * dt *                            \
       (BUOYANCY_X(dz, dy, dx) * VX_BG(dz, dy, dx) + \
        DBUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx)) +  \
   P_BXH(dx) * M_SIGMAXXX_BG(dz, dy, dx))
#define SIGMAXX_X_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_X(dz, dy, dx) * VX_BG(dz, dy, dx) + \
         DBUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx)))

#if DW_NDIM >= 2
#define SIGMAXY_Y_BG_PML(dz, dy, dx)                 \
  ((1 + P_BY(dy)) * dt *                             \
       (BUOYANCY_X(dz, dy, dx) * VX_BG(dz, dy, dx) + \
        DBUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx)) +  \
   P_BY(dy) * M_SIGMAXYY_BG(dz, dy, dx))
#define SIGMAXY_Y_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_X(dz, dy, dx) * VX_BG(dz, dy, dx) + \
         DBUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx)))
#define SIGMAXY_X_BG_PML(dz, dy, dx)                 \
  ((1 + P_BX(dx)) * dt *                             \
       (BUOYANCY_Y(dz, dy, dx) * VY_BG(dz, dy, dx) + \
        DBUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx)) +  \
   P_BX(dx) * M_SIGMAXYX_BG(dz, dy, dx))
#define SIGMAXY_X_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_Y(dz, dy, dx) * VY_BG(dz, dy, dx) + \
         DBUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx)))
#endif

#if DW_NDIM >= 3
#define SIGMAXZ_Z_BG_PML(dz, dy, dx)                 \
  ((1 + P_BZ(dz)) * dt *                             \
       (BUOYANCY_X(dz, dy, dx) * VX_BG(dz, dy, dx) + \
        DBUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx)) +  \
   P_BZ(dz) * M_SIGMAXZZ_BG(dz, dy, dx))
#define SIGMAXZ_Z_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_X(dz, dy, dx) * VX_BG(dz, dy, dx) + \
         DBUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx)))
#define SIGMAXZ_X_BG_PML(dz, dy, dx)                 \
  ((1 + P_BX(dx)) * dt *                             \
       (BUOYANCY_Z(dz, dy, dx) * VZ_BG(dz, dy, dx) + \
        DBUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx)) +  \
   P_BX(dx) * M_SIGMAXZX_BG(dz, dy, dx))
#define SIGMAXZ_X_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_Z(dz, dy, dx) * VZ_BG(dz, dy, dx) + \
         DBUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx)))
#define SIGMAYZ_Z_BG_PML(dz, dy, dx)                 \
  ((1 + P_BZ(dz)) * dt *                             \
       (BUOYANCY_Y(dz, dy, dx) * VY_BG(dz, dy, dx) + \
        DBUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx)) +  \
   P_BZ(dz) * M_SIGMAYZZ_BG(dz, dy, dx))
#define SIGMAYZ_Z_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_Y(dz, dy, dx) * VY_BG(dz, dy, dx) + \
         DBUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx)))
#define SIGMAYZ_Y_BG_PML(dz, dy, dx)                 \
  ((1 + P_BY(dy)) * dt *                             \
       (BUOYANCY_Z(dz, dy, dx) * VZ_BG(dz, dy, dx) + \
        DBUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx)) +  \
   P_BY(dy) * M_SIGMAYZY_BG(dz, dy, dx))
#define SIGMAYZ_Y_BG(dz, dy, dx)                      \
  (dt * (BUOYANCY_Z(dz, dy, dx) * VZ_BG(dz, dy, dx) + \
         DBUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx)))
#endif

/* ============================= Forward ============================= */

#if DW_NDIM >= 3
/* vz += buoyancy_z*dt*(d(sigmazz)/dz + d(sigmayz)/dy + d(sigmaxz)/dx);
 * vz_sc += buoyancy_z*dt*w_sum_sc + dbuoyancy_z*dt*w_sum, background then
 * scattered (coupled), matching elastic_born.c's/.cu's single fused loop
 * body per axis. `i` addresses sigmazz/sigmayz/sigmaxz/vz/m_sigmazzz/
 * m_sigmayzy/m_sigmaxzx/dvzdbuoyancy_store_1 and their _sc twins
 * (per-shot-local on the CPU driver, global on the CUDA driver);
 * `buoyancy_z_shot`/`dbuoyancy_z_shot` are the (already shot-selected)
 * buoyancy/perturbation VALUES at this point (read once by the caller --
 * this update never needs a neighbour offset of either); `pml_z`/`pml_y`/
 * `pml_x` are true when this point is inside that axis's PML region. */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
    DW_DTYPE const *__restrict const sigmazz_sc,
    DW_DTYPE const *__restrict const sigmayz_sc,
    DW_DTYPE const *__restrict const sigmaxz_sc, DW_DTYPE *__restrict const vz,
    DW_DTYPE *__restrict const vz_sc, DW_DTYPE *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmazzz_sc,
    DW_DTYPE *__restrict const m_sigmayzy_sc,
    DW_DTYPE *__restrict const m_sigmaxzx_sc,
    DW_DTYPE *__restrict const dvzdbuoyancy_store_1,
    DW_DTYPE *__restrict const dvzdbuoyancy_sc_store_1,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    bool const pml_x, DW_DTYPE const rdz, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE w_sum_sc = 0;

  if (!pml_z) {
    w_sum += DIFFZH1(SIGMAZZ);
    w_sum_sc += DIFFZH1(SIGMAZZ_SC);
  } else {
    DW_DTYPE d3 = DIFFZH1(SIGMAZZ);
    m_sigmazzz[i] = P_AZH(0) * m_sigmazzz[i] + P_BZH(0) * d3;
    d3 += m_sigmazzz[i];
    w_sum += d3;
    DW_DTYPE d3s = DIFFZH1(SIGMAZZ_SC);
    m_sigmazzz_sc[i] = P_AZH(0) * m_sigmazzz_sc[i] + P_BZH(0) * d3s;
    d3s += m_sigmazzz_sc[i];
    w_sum_sc += d3s;
  }
  if (!pml_y) {
    w_sum += DIFFY1(SIGMAYZ);
    w_sum_sc += DIFFY1(SIGMAYZ_SC);
  } else {
    DW_DTYPE d2 = DIFFY1(SIGMAYZ);
    m_sigmayzy[i] = P_AY(0) * m_sigmayzy[i] + P_BY(0) * d2;
    d2 += m_sigmayzy[i];
    w_sum += d2;
    DW_DTYPE d2s = DIFFY1(SIGMAYZ_SC);
    m_sigmayzy_sc[i] = P_AY(0) * m_sigmayzy_sc[i] + P_BY(0) * d2s;
    d2s += m_sigmayzy_sc[i];
    w_sum_sc += d2s;
  }
  if (!pml_x) {
    w_sum += DIFFX1(SIGMAXZ);
    w_sum_sc += DIFFX1(SIGMAXZ_SC);
  } else {
    DW_DTYPE d1 = DIFFX1(SIGMAXZ);
    m_sigmaxzx[i] = P_AX(0) * m_sigmaxzx[i] + P_BX(0) * d1;
    d1 += m_sigmaxzx[i];
    w_sum += d1;
    DW_DTYPE d1s = DIFFX1(SIGMAXZ_SC);
    m_sigmaxzx_sc[i] = P_AX(0) * m_sigmaxzx_sc[i] + P_BX(0) * d1s;
    d1s += m_sigmaxzx_sc[i];
    w_sum_sc += d1s;
  }

  if (store_bg) dvzdbuoyancy_store_1[i] = dt * w_sum;
  if (store_sc) dvzdbuoyancy_sc_store_1[i] = dt * w_sum_sc;
  vz[i] += buoyancy_z_shot * dt * w_sum;
  vz_sc[i] += buoyancy_z_shot * dt * w_sum_sc + dbuoyancy_z_shot * dt * w_sum;
}
#endif

#if DW_NDIM >= 2
/* Forward velocity update, y axis (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vy_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmayz_sc,
#endif
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
    DW_DTYPE const *__restrict const sigmayy_sc,
    DW_DTYPE const *__restrict const sigmaxy_sc, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const vy_sc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmayzz_sc,
#endif
    DW_DTYPE *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmayyy_sc,
    DW_DTYPE *__restrict const m_sigmaxyx_sc,
    DW_DTYPE *__restrict const dvydbuoyancy_store_1,
    DW_DTYPE *__restrict const dvydbuoyancy_sc_store_1,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE const dbuoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
#endif
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i,
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
  DW_DTYPE w_sum_sc = 0;

#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ1(SIGMAYZ);
    w_sum_sc += DIFFZ1(SIGMAYZ_SC);
  } else {
    DW_DTYPE d3 = DIFFZ1(SIGMAYZ);
    m_sigmayzz[i] = P_AZ(0) * m_sigmayzz[i] + P_BZ(0) * d3;
    d3 += m_sigmayzz[i];
    w_sum += d3;
    DW_DTYPE d3s = DIFFZ1(SIGMAYZ_SC);
    m_sigmayzz_sc[i] = P_AZ(0) * m_sigmayzz_sc[i] + P_BZ(0) * d3s;
    d3s += m_sigmayzz_sc[i];
    w_sum_sc += d3s;
  }
#endif
  if (!pml_y) {
    w_sum += DIFFYH1(SIGMAYY);
    w_sum_sc += DIFFYH1(SIGMAYY_SC);
  } else {
    DW_DTYPE d2 = DIFFYH1(SIGMAYY);
    m_sigmayyy[i] = P_AYH(0) * m_sigmayyy[i] + P_BYH(0) * d2;
    d2 += m_sigmayyy[i];
    w_sum += d2;
    DW_DTYPE d2s = DIFFYH1(SIGMAYY_SC);
    m_sigmayyy_sc[i] = P_AYH(0) * m_sigmayyy_sc[i] + P_BYH(0) * d2s;
    d2s += m_sigmayyy_sc[i];
    w_sum_sc += d2s;
  }
  if (!pml_x) {
    w_sum += DIFFX1(SIGMAXY);
    w_sum_sc += DIFFX1(SIGMAXY_SC);
  } else {
    DW_DTYPE d1 = DIFFX1(SIGMAXY);
    m_sigmaxyx[i] = P_AX(0) * m_sigmaxyx[i] + P_BX(0) * d1;
    d1 += m_sigmaxyx[i];
    w_sum += d1;
    DW_DTYPE d1s = DIFFX1(SIGMAXY_SC);
    m_sigmaxyx_sc[i] = P_AX(0) * m_sigmaxyx_sc[i] + P_BX(0) * d1s;
    d1s += m_sigmaxyx_sc[i];
    w_sum_sc += d1s;
  }

  if (store_bg) dvydbuoyancy_store_1[i] = dt * w_sum;
  if (store_sc) dvydbuoyancy_sc_store_1[i] = dt * w_sum_sc;
  vy[i] += buoyancy_y_shot * dt * w_sum;
  vy_sc[i] += buoyancy_y_shot * dt * w_sum_sc + dbuoyancy_y_shot * dt * w_sum;
}
#endif

/* Forward velocity update, x axis (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vx_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz,
    DW_DTYPE const *__restrict const sigmaxz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy,
    DW_DTYPE const *__restrict const sigmaxy_sc,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const sigmaxx_sc, DW_DTYPE *__restrict const vx,
    DW_DTYPE *__restrict const vx_sc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_sigmaxzz,
    DW_DTYPE *__restrict const m_sigmaxzz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyy_sc,
#endif
    DW_DTYPE *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmaxxx_sc,
    DW_DTYPE *__restrict const dvxdbuoyancy_store_1,
    DW_DTYPE *__restrict const dvxdbuoyancy_sc_store_1,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE const dbuoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
#endif
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bxh,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i,
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
  DW_DTYPE w_sum_sc = 0;

#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ1(SIGMAXZ);
    w_sum_sc += DIFFZ1(SIGMAXZ_SC);
  } else {
    DW_DTYPE d3 = DIFFZ1(SIGMAXZ);
    m_sigmaxzz[i] = P_AZ(0) * m_sigmaxzz[i] + P_BZ(0) * d3;
    d3 += m_sigmaxzz[i];
    w_sum += d3;
    DW_DTYPE d3s = DIFFZ1(SIGMAXZ_SC);
    m_sigmaxzz_sc[i] = P_AZ(0) * m_sigmaxzz_sc[i] + P_BZ(0) * d3s;
    d3s += m_sigmaxzz_sc[i];
    w_sum_sc += d3s;
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY1(SIGMAXY);
    w_sum_sc += DIFFY1(SIGMAXY_SC);
  } else {
    DW_DTYPE d2 = DIFFY1(SIGMAXY);
    m_sigmaxyy[i] = P_AY(0) * m_sigmaxyy[i] + P_BY(0) * d2;
    d2 += m_sigmaxyy[i];
    w_sum += d2;
    DW_DTYPE d2s = DIFFY1(SIGMAXY_SC);
    m_sigmaxyy_sc[i] = P_AY(0) * m_sigmaxyy_sc[i] + P_BY(0) * d2s;
    d2s += m_sigmaxyy_sc[i];
    w_sum_sc += d2s;
  }
#endif
  if (!pml_x) {
    w_sum += DIFFXH1(SIGMAXX);
    w_sum_sc += DIFFXH1(SIGMAXX_SC);
  } else {
    DW_DTYPE d1 = DIFFXH1(SIGMAXX);
    m_sigmaxxx[i] = P_AXH(0) * m_sigmaxxx[i] + P_BXH(0) * d1;
    d1 += m_sigmaxxx[i];
    w_sum += d1;
    DW_DTYPE d1s = DIFFXH1(SIGMAXX_SC);
    m_sigmaxxx_sc[i] = P_AXH(0) * m_sigmaxxx_sc[i] + P_BXH(0) * d1s;
    d1s += m_sigmaxxx_sc[i];
    w_sum_sc += d1s;
  }

  if (store_bg) dvxdbuoyancy_store_1[i] = dt * w_sum;
  if (store_sc) dvxdbuoyancy_sc_store_1[i] = dt * w_sum_sc;
  vx[i] += buoyancy_x_shot * dt * w_sum;
  vx_sc[i] += buoyancy_x_shot * dt * w_sum_sc + dbuoyancy_x_shot * dt * w_sum;
}

/* sigmazz/sigmayy/sigmaxx += dt*(lamb*strain + 2*mu*dv_axis/d_axis); the _sc
 * twins += dt*(lamb*strain_sc + 2*mu*dv_axis_sc/d_axis) +
 * dt*(dlamb*strain + 2*dmu*dv_axis/d_axis), the combined normal-stress
 * update (every present axis's dv/dn term accumulates into `strain`/
 * `strain_sc`, like elastic_kernels.h's forward_sigmaii_cell combines its
 * axes into `w_sum`), background then scattered (coupled). `i` addresses
 * vz/vy/vx/sigmazz/sigmayy/sigmaxx/m_vzz/m_vyy/m_vxx/dv*d*_store_1 and their
 * _sc twins (per-shot-local on the CPU driver, global on the CUDA driver);
 * `lamb_shot`/`mu_shot`/`dlamb_shot`/`dmu_shot` are the (already
 * shot-selected) model VALUEs at this point (read once by the caller --
 * never need a neighbour offset here); `pml_z`/`pml_y`/`pml_x` are true when
 * this point is inside that axis's PML region. */
DW_HOST_DEVICE void FUNC(forward_sigmaii_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz, DW_DTYPE const *__restrict const vz_sc,
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmazz_sc,
    DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzz_sc,
    DW_DTYPE *__restrict const dvzdz_store_1,
    DW_DTYPE *__restrict const dvzdz_sc_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vy_sc,
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmayy_sc,
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyy_sc,
    DW_DTYPE *__restrict const dvydy_store_1,
    DW_DTYPE *__restrict const dvydy_sc_store_1,
#endif
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vx_sc,
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE *__restrict const sigmaxx_sc,
    DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict const m_vxx_sc,
    DW_DTYPE *__restrict const dvxdx_store_1,
    DW_DTYPE *__restrict const dvxdx_sc_store_1, DW_DTYPE const lamb_shot,
    DW_DTYPE const mu_shot, DW_DTYPE const dlamb_shot, DW_DTYPE const dmu_shot,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const by, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const rdx) {
#if DW_NDIM >= 3
  DW_DTYPE dvzdz, dvzdz_sc;
  if (!pml_z) {
    dvzdz = DIFFZ1(VZ);
    dvzdz_sc = DIFFZ1(VZ_SC);
  } else {
    dvzdz = DIFFZ1(VZ);
    m_vzz[i] = P_AZ(0) * m_vzz[i] + P_BZ(0) * dvzdz;
    dvzdz += m_vzz[i];
    dvzdz_sc = DIFFZ1(VZ_SC);
    m_vzz_sc[i] = P_AZ(0) * m_vzz_sc[i] + P_BZ(0) * dvzdz_sc;
    dvzdz_sc += m_vzz_sc[i];
  }
#endif
#if DW_NDIM >= 2
  DW_DTYPE dvydy, dvydy_sc;
  if (!pml_y) {
    dvydy = DIFFY1(VY);
    dvydy_sc = DIFFY1(VY_SC);
  } else {
    dvydy = DIFFY1(VY);
    m_vyy[i] = P_AY(0) * m_vyy[i] + P_BY(0) * dvydy;
    dvydy += m_vyy[i];
    dvydy_sc = DIFFY1(VY_SC);
    m_vyy_sc[i] = P_AY(0) * m_vyy_sc[i] + P_BY(0) * dvydy_sc;
    dvydy_sc += m_vyy_sc[i];
  }
#endif
  DW_DTYPE dvxdx, dvxdx_sc;
  if (!pml_x) {
    dvxdx = DIFFX1(VX);
    dvxdx_sc = DIFFX1(VX_SC);
  } else {
    dvxdx = DIFFX1(VX);
    m_vxx[i] = P_AX(0) * m_vxx[i] + P_BX(0) * dvxdx;
    dvxdx += m_vxx[i];
    dvxdx_sc = DIFFX1(VX_SC);
    m_vxx_sc[i] = P_AX(0) * m_vxx_sc[i] + P_BX(0) * dvxdx_sc;
    dvxdx_sc += m_vxx_sc[i];
  }

  DW_DTYPE const strain =
#if DW_NDIM >= 3
      dvzdz +
#endif
#if DW_NDIM >= 2
      dvydy +
#endif
      dvxdx;
  DW_DTYPE const strain_sc =
#if DW_NDIM >= 3
      dvzdz_sc +
#endif
#if DW_NDIM >= 2
      dvydy_sc +
#endif
      dvxdx_sc;

  if (store_bg) {
#if DW_NDIM >= 3
    dvzdz_store_1[i] = dt * dvzdz;
#endif
#if DW_NDIM >= 2
    dvydy_store_1[i] = dt * dvydy;
#endif
    dvxdx_store_1[i] = dt * dvxdx;
  }
  if (store_sc) {
#if DW_NDIM >= 3
    dvzdz_sc_store_1[i] = dt * dvzdz_sc;
#endif
#if DW_NDIM >= 2
    dvydy_sc_store_1[i] = dt * dvydy_sc;
#endif
    dvxdx_sc_store_1[i] = dt * dvxdx_sc;
  }

#if DW_NDIM >= 3
  sigmazz[i] += dt * (lamb_shot * strain + 2 * mu_shot * dvzdz);
  sigmazz_sc[i] += dt * (lamb_shot * strain_sc + 2 * mu_shot * dvzdz_sc) +
                   dt * (dlamb_shot * strain + 2 * dmu_shot * dvzdz);
#endif
#if DW_NDIM >= 2
  sigmayy[i] += dt * (lamb_shot * strain + 2 * mu_shot * dvydy);
  sigmayy_sc[i] += dt * (lamb_shot * strain_sc + 2 * mu_shot * dvydy_sc) +
                   dt * (dlamb_shot * strain + 2 * dmu_shot * dvydy);
#endif
  sigmaxx[i] += dt * (lamb_shot * strain + 2 * mu_shot * dvxdx);
  sigmaxx_sc[i] += dt * (lamb_shot * strain_sc + 2 * mu_shot * dvxdx_sc) +
                   dt * (dlamb_shot * strain + 2 * dmu_shot * dvxdx);
}

#if DW_NDIM >= 2
/* sigmaxy += dt*mu_yx*(d(vx)/dy + d(vy)/dx); sigmaxy_sc += dt*mu_yx*w_sum_sc
 * + dt*dmu_yx*w_sum (see FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmaxy_cell)(
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const vx_sc,
    DW_DTYPE const *__restrict const vy_sc, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxy_sc, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy_sc,
    DW_DTYPE *__restrict const m_vyx_sc,
    DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE *__restrict const dvydx_plus_dvxdy_sc_store_1,
    DW_DTYPE const mu_yx_shot, DW_DTYPE const dmu_yx_shot, DW_DTYPE const dt,
    bool const store_bg, bool const store_sc, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0, w_sum_sc = 0;
  DW_DTYPE dvxdy, dvxdy_sc, dvydx, dvydx_sc;

  if (!pml_y) {
    dvxdy = DIFFYH1(VX);
    dvxdy_sc = DIFFYH1(VX_SC);
  } else {
    dvxdy = DIFFYH1(VX);
    m_vxy[i] = P_AYH(0) * m_vxy[i] + P_BYH(0) * dvxdy;
    dvxdy += m_vxy[i];
    dvxdy_sc = DIFFYH1(VX_SC);
    m_vxy_sc[i] = P_AYH(0) * m_vxy_sc[i] + P_BYH(0) * dvxdy_sc;
    dvxdy_sc += m_vxy_sc[i];
  }
  w_sum += dvxdy;
  w_sum_sc += dvxdy_sc;

  if (!pml_x) {
    dvydx = DIFFXH1(VY);
    dvydx_sc = DIFFXH1(VY_SC);
  } else {
    dvydx = DIFFXH1(VY);
    m_vyx[i] = P_AXH(0) * m_vyx[i] + P_BXH(0) * dvydx;
    dvydx += m_vyx[i];
    dvydx_sc = DIFFXH1(VY_SC);
    m_vyx_sc[i] = P_AXH(0) * m_vyx_sc[i] + P_BXH(0) * dvydx_sc;
    dvydx_sc += m_vyx_sc[i];
  }
  w_sum += dvydx;
  w_sum_sc += dvydx_sc;

  if (store_bg) dvydx_plus_dvxdy_store_1[i] = dt * w_sum;
  if (store_sc) dvydx_plus_dvxdy_sc_store_1[i] = dt * w_sum_sc;
  sigmaxy[i] += dt * mu_yx_shot * w_sum;
  sigmaxy_sc[i] += dt * mu_yx_shot * w_sum_sc + dt * dmu_yx_shot * w_sum;
}
#endif

#if DW_NDIM >= 3
/* sigmaxz += dt*mu_zx*(d(vx)/dz + d(vz)/dx) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmaxz_cell)(
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const vx_sc,
    DW_DTYPE const *__restrict const vz_sc, DW_DTYPE *__restrict const sigmaxz,
    DW_DTYPE *__restrict const sigmaxz_sc, DW_DTYPE *__restrict const m_vxz,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vxz_sc,
    DW_DTYPE *__restrict const m_vzx_sc,
    DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE *__restrict const dvzdx_plus_dvxdz_sc_store_1,
    DW_DTYPE const mu_zx_shot, DW_DTYPE const dmu_zx_shot, DW_DTYPE const dt,
    bool const store_bg, bool const store_sc, int64_t const i, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_x, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdz,
    DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0, w_sum_sc = 0;
  DW_DTYPE dvxdz, dvxdz_sc, dvzdx, dvzdx_sc;

  if (!pml_z) {
    dvxdz = DIFFZH1(VX);
    dvxdz_sc = DIFFZH1(VX_SC);
  } else {
    dvxdz = DIFFZH1(VX);
    m_vxz[i] = P_AZH(0) * m_vxz[i] + P_BZH(0) * dvxdz;
    dvxdz += m_vxz[i];
    dvxdz_sc = DIFFZH1(VX_SC);
    m_vxz_sc[i] = P_AZH(0) * m_vxz_sc[i] + P_BZH(0) * dvxdz_sc;
    dvxdz_sc += m_vxz_sc[i];
  }
  w_sum += dvxdz;
  w_sum_sc += dvxdz_sc;

  if (!pml_x) {
    dvzdx = DIFFXH1(VZ);
    dvzdx_sc = DIFFXH1(VZ_SC);
  } else {
    dvzdx = DIFFXH1(VZ);
    m_vzx[i] = P_AXH(0) * m_vzx[i] + P_BXH(0) * dvzdx;
    dvzdx += m_vzx[i];
    dvzdx_sc = DIFFXH1(VZ_SC);
    m_vzx_sc[i] = P_AXH(0) * m_vzx_sc[i] + P_BXH(0) * dvzdx_sc;
    dvzdx_sc += m_vzx_sc[i];
  }
  w_sum += dvzdx;
  w_sum_sc += dvzdx_sc;

  if (store_bg) dvzdx_plus_dvxdz_store_1[i] = dt * w_sum;
  if (store_sc) dvzdx_plus_dvxdz_sc_store_1[i] = dt * w_sum_sc;
  sigmaxz[i] += dt * mu_zx_shot * w_sum;
  sigmaxz_sc[i] += dt * mu_zx_shot * w_sum_sc + dt * dmu_zx_shot * w_sum;
}

/* sigmayz += dt*mu_zy*(d(vy)/dz + d(vz)/dy) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmayz_cell)(
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const vy_sc,
    DW_DTYPE const *__restrict const vz_sc, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmayz_sc, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vzy, DW_DTYPE *__restrict const m_vyz_sc,
    DW_DTYPE *__restrict const m_vzy_sc,
    DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE *__restrict const dvzdy_plus_dvydz_sc_store_1,
    DW_DTYPE const mu_zy_shot, DW_DTYPE const dmu_zy_shot, DW_DTYPE const dt,
    bool const store_bg, bool const store_sc, int64_t const i, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdz,
    DW_DTYPE const rdy) {
  DW_DTYPE w_sum = 0, w_sum_sc = 0;
  DW_DTYPE dvydz, dvydz_sc, dvzdy, dvzdy_sc;

  if (!pml_z) {
    dvydz = DIFFZH1(VY);
    dvydz_sc = DIFFZH1(VY_SC);
  } else {
    dvydz = DIFFZH1(VY);
    m_vyz[i] = P_AZH(0) * m_vyz[i] + P_BZH(0) * dvydz;
    dvydz += m_vyz[i];
    dvydz_sc = DIFFZH1(VY_SC);
    m_vyz_sc[i] = P_AZH(0) * m_vyz_sc[i] + P_BZH(0) * dvydz_sc;
    dvydz_sc += m_vyz_sc[i];
  }
  w_sum += dvydz;
  w_sum_sc += dvydz_sc;

  if (!pml_y) {
    dvzdy = DIFFYH1(VZ);
    dvzdy_sc = DIFFYH1(VZ_SC);
  } else {
    dvzdy = DIFFYH1(VZ);
    m_vzy[i] = P_AYH(0) * m_vzy[i] + P_BYH(0) * dvzdy;
    dvzdy += m_vzy[i];
    dvzdy_sc = DIFFYH1(VZ_SC);
    m_vzy_sc[i] = P_AYH(0) * m_vzy_sc[i] + P_BYH(0) * dvzdy_sc;
    dvzdy_sc += m_vzy_sc[i];
  }
  w_sum += dvzdy;
  w_sum_sc += dvzdy_sc;

  if (store_bg) dvzdy_plus_dvydz_store_1[i] = dt * w_sum;
  if (store_sc) dvzdy_plus_dvydz_sc_store_1[i] = dt * w_sum_sc;
  sigmayz[i] += dt * mu_zy_shot * w_sum;
  sigmayz_sc[i] += dt * mu_zy_shot * w_sum_sc + dt * dmu_zy_shot * w_sum;
}
#endif

/* ==================== Backward (scattered-only) ==================== */

#if DW_NDIM >= 3
/* Scattered-only adjoint velocity update, z axis: the ORDINARY (unperturbed)
 * elastic adjoint recursion (identical in form to elastic_kernels.h's
 * FUNC(backward_vz_cell)) applied to the scattered adjoint field, using the
 * BACKGROUND models, plus the M_SIGMA** ping-pong recursion (feeding
 * FUNC(backward_sc_sigma*_cell) at the *next* time step) and grad_dbuoyancy_z
 * accumulation (correlating the scattered adjoint field with the BACKGROUND
 * term store `dvzdbuoyancy_store_1`). `i` addresses sigma(*)/vz/m_v(**)/
 * m_sigma***(n)/grad_dbuoyancy_z (per-shot-local on the CPU driver, global on
 * the CUDA driver); `j` addresses the already-shot-sliced lamb_shot/mu_shot/
 * mu_zy_shot/mu_zx_shot (equal to `i` on the CPU driver); `pml_z`/`pml_y`/
 * `pml_x` are true inside that axis's PML region. */
DW_HOST_DEVICE void FUNC(backward_sc_vz_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz, DW_DTYPE *__restrict const vz,
    DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const bxh,
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

/* M_SIGMA** PML memory-variable update for FUNC(backward_sc_vz_cell) and
 * grad_dbuoyancy_z accumulation, split out and moved to run in a separate
 * pass AFTER FUNC(fill_elastic_stress_bc_adjoint)/FUNC(fill_elastic_
 * velocity_bc_adjoint)'s velocity fold (see the driver-side comment at its
 * new call site in elastic_born.c/.cu): vz[i] must be read post-fold -- for
 * a cell the fold accumulates into or clamps to zero (e.g. a rigid on-row
 * velocity node), the value FUNC(backward_sc_vz_cell) computed pre-fold is
 * stale, so writing this memory variable / the gradient from the pre-fold
 * value would use the wrong number for exactly those cells. For every other
 * cell (the fold never touches it) vz[i] is identical pre- and post-fold,
 * so this split changes nothing there. */
DW_HOST_DEVICE void FUNC(backward_sc_vz_cell_pml_mem)(
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmazzz_new,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmayzy_new,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE *__restrict const grad_dbuoyancy_z,
    DW_DTYPE const *__restrict const dvzdbuoyancy_store_1,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const dt,
    int64_t const step_ratio, bool const dbuoyancy_requires_grad,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    bool const pml_x) {
  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx[i];
  }

  if (dbuoyancy_requires_grad) {
    grad_dbuoyancy_z[i] +=
        vz[i] * dvzdbuoyancy_store_1[i] * (DW_DTYPE)step_ratio;
  }
}
#endif

#if DW_NDIM >= 2
/* Scattered-only adjoint velocity update, y axis (see
 * FUNC(backward_sc_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_vy_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy_shot,
#endif
    DW_DTYPE const *__restrict const mu_yx_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz,
#endif
    DW_DTYPE const *__restrict const sigmaxy, DW_DTYPE *__restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vyz,
#endif
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bxh,
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
  w_sum += pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y);
  w_sum += pml_x ? -DIFFX1(VY_X_PML) : -DIFFX1(VY_X);

  vy[i] += w_sum;
}

/* See FUNC(backward_sc_vz_cell_pml_mem)'s comment (same split, same
 * reason: vy[i] must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_sc_vy_cell_pml_mem)(
    DW_DTYPE const *__restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmayyy_new,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
    DW_DTYPE *__restrict const grad_dbuoyancy_y,
    DW_DTYPE const *__restrict const dvydbuoyancy_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az,
#endif
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const dt, int64_t const step_ratio,
    bool const dbuoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z, bool const pml_z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x) {
#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx[i];
  }

  if (dbuoyancy_requires_grad) {
    grad_dbuoyancy_y[i] +=
        vy[i] * dvydbuoyancy_store_1[i] * (DW_DTYPE)step_ratio;
  }
}
#endif

/* Scattered-only adjoint velocity update, x axis (see
 * FUNC(backward_sc_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_vx_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const bx, DW_DTYPE const dt, int64_t const i,
    int64_t const j,
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
  w_sum += pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X);

  vx[i] += w_sum;
}

/* See FUNC(backward_sc_vz_cell_pml_mem)'s comment (same split, same
 * reason: vx[i] must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_sc_vx_cell_pml_mem)(
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE *__restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
    DW_DTYPE *__restrict const grad_dbuoyancy_x,
    DW_DTYPE const *__restrict const dvxdbuoyancy_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay,
#endif
    DW_DTYPE const *__restrict const axh, DW_DTYPE const dt,
    int64_t const step_ratio, bool const dbuoyancy_requires_grad,
    int64_t const i,
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
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx[i];
  }

  if (dbuoyancy_requires_grad) {
    grad_dbuoyancy_x[i] +=
        vx[i] * dvxdbuoyancy_store_1[i] * (DW_DTYPE)step_ratio;
  }
}

/* grad_dlamb/grad_dmu accumulation (correlating the scattered adjoint field
 * with the BACKGROUND term stores), the M_V** PML memory-variable update
 * (read by FUNC(backward_sc_v*_cell) at the *next* time step), and
 * sigmazz/sigmayy/sigmaxx += -(d(SIGMAZZ_Z)/dz + d(SIGMAYY_Y)/dy +
 * d(SIGMAXX_X)/dx) (the transpose of FUNC(forward_vz_cell)/vy_cell/vx_cell's
 * use of sigma*). `i`/`j` addressing matches FUNC(backward_sc_vz_cell)
 * above. */
DW_HOST_DEVICE void FUNC(backward_sc_sigmaii_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y_shot,
#endif
    DW_DTYPE const *__restrict const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz, DW_DTYPE *__restrict const sigmazz,
    DW_DTYPE *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const dvzdz_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
    DW_DTYPE *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const dvydy_store_1,
#endif
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE *__restrict const m_vxx,
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE const *__restrict const dvxdx_store_1,
    DW_DTYPE *__restrict const grad_dlamb, DW_DTYPE *__restrict const grad_dmu,
    DW_DTYPE const dt, int64_t const step_ratio, bool const dlamb_requires_grad,
    bool const dmu_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdx) {
  if (dlamb_requires_grad) {
#if DW_NDIM == 3
    grad_dlamb[i] += (sigmaxx[i] + sigmayy[i] + sigmazz[i]) *
                     (dvxdx_store_1[i] + dvydy_store_1[i] + dvzdz_store_1[i]) *
                     (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_dlamb[i] += (sigmaxx[i] + sigmayy[i]) *
                     (dvxdx_store_1[i] + dvydy_store_1[i]) *
                     (DW_DTYPE)step_ratio;
#else
    grad_dlamb[i] += sigmaxx[i] * dvxdx_store_1[i] * (DW_DTYPE)step_ratio;
#endif
  }
  if (dmu_requires_grad) {
#if DW_NDIM == 3
    grad_dmu[i] +=
        2 *
        (sigmaxx[i] * dvxdx_store_1[i] + sigmayy[i] * dvydy_store_1[i] +
         sigmazz[i] * dvzdz_store_1[i]) *
        (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_dmu[i] +=
        2 * (sigmaxx[i] * dvxdx_store_1[i] + sigmayy[i] * dvydy_store_1[i]) *
        (DW_DTYPE)step_ratio;
#else
    grad_dmu[i] += 2 * sigmaxx[i] * dvxdx_store_1[i] * (DW_DTYPE)step_ratio;
#endif
  }

#if DW_NDIM >= 3
  if (pml_z) {
    m_vzz[i] = LAMB_2MU(0, 0, 0) * dt * P_AZ(0) * sigmazz[i] +
               LAMB(0, 0, 0) * dt * P_AZ(0) * sigmaxx[i] +
               LAMB(0, 0, 0) * dt * P_AZ(0) * sigmayy[i] + P_AZ(0) * m_vzz[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_vyy[i] = LAMB_2MU(0, 0, 0) * dt * P_AY(0) * sigmayy[i] +
               LAMB(0, 0, 0) * dt * P_AY(0) * sigmaxx[i] +
#if DW_NDIM == 3
               LAMB(0, 0, 0) * dt * P_AY(0) * sigmazz[i] +
#endif
               P_AY(0) * m_vyy[i];
  }
#endif
  if (pml_x) {
    m_vxx[i] = LAMB_2MU(0, 0, 0) * dt * P_AX(0) * sigmaxx[i] +
#if DW_NDIM >= 2
               LAMB(0, 0, 0) * dt * P_AX(0) * sigmayy[i] +
#endif
#if DW_NDIM == 3
               LAMB(0, 0, 0) * dt * P_AX(0) * sigmazz[i] +
#endif
               P_AX(0) * m_vxx[i];
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
/* grad_dmu_yx accumulation, M_VXY/M_VYX PML memory update, and sigmaxy +=
 * -(d(SIGMAXY_Y)/dy + d(SIGMAXY_X)/dx) (see
 * FUNC(backward_sc_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmaxy_cell)(
    DW_DTYPE const *__restrict const mu_yx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vy,
    DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE const *__restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE *__restrict const grad_dmu_yx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const dmu_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (dmu_requires_grad) {
    grad_dmu_yx[i] +=
        sigmaxy[i] * dvydx_plus_dvxdy_store_1[i] * (DW_DTYPE)step_ratio;
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
/* grad_dmu_zx accumulation, M_VXZ/M_VZX PML memory update, and sigmaxz +=
 * -(d(SIGMAXZ_Z)/dz + d(SIGMAXZ_X)/dx) (see
 * FUNC(backward_sc_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmaxz_cell)(
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vz,
    DW_DTYPE *__restrict const sigmaxz, DW_DTYPE *__restrict const m_vxz,
    DW_DTYPE *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE *__restrict const grad_dmu_zx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const dmu_requires_grad, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_x,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    DW_DTYPE const rdz, DW_DTYPE const rdx) {
  if (dmu_requires_grad) {
    grad_dmu_zx[i] +=
        sigmaxz[i] * dvzdx_plus_dvxdz_store_1[i] * (DW_DTYPE)step_ratio;
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

/* grad_dmu_zy accumulation, M_VYZ/M_VZY PML memory update, and sigmayz +=
 * -(d(SIGMAYZ_Z)/dz + d(SIGMAYZ_Y)/dy) (see
 * FUNC(backward_sc_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmayz_cell)(
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vz,
    DW_DTYPE *__restrict const sigmayz, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE *__restrict const grad_dmu_zy, DW_DTYPE const dt,
    int64_t const step_ratio, bool const dmu_requires_grad, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const rdz, DW_DTYPE const rdy) {
  if (dmu_requires_grad) {
    grad_dmu_zy[i] +=
        sigmayz[i] * dvzdy_plus_dvydz_store_1[i] * (DW_DTYPE)step_ratio;
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

/* ==================== Backward (full two-field) ==================== */

#if DW_NDIM >= 3
/* Full two-field adjoint velocity update, z axis: the SCATTERED adjoint field
 * with the ordinary adjoint (VZ_Z/_PML, as FUNC(backward_sc_vz_cell)) AND the
 * BACKGROUND adjoint field with the COUPLED adjoint (VZ_Z_BG/_PML), in one
 * call. Accumulates grad_dbuoyancy_z (one term, as
 * FUNC(backward_sc_vz_cell)) and grad_buoyancy_z (two correlation terms:
 * background field with the background store, scattered field with the
 * scattered store). `i`/`j` addressing matches FUNC(backward_sc_vz_cell)
 * above. */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
    DW_DTYPE const *__restrict const dmu_zy_shot,
    DW_DTYPE const *__restrict const dmu_zx_shot,
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
    DW_DTYPE const *__restrict const sigmazz_bg,
    DW_DTYPE const *__restrict const sigmayy_bg,
    DW_DTYPE const *__restrict const sigmaxx_bg,
    DW_DTYPE const *__restrict const sigmayz_bg,
    DW_DTYPE const *__restrict const sigmaxz_bg, DW_DTYPE *__restrict const vz,
    DW_DTYPE *__restrict const vz_bg, DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_vzz_bg,
    DW_DTYPE const *__restrict const m_vzy_bg,
    DW_DTYPE const *__restrict const m_vzx_bg,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const dt, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    bool const pml_x, DW_DTYPE const rdz, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0, w_sum_bg = 0;
  w_sum += pml_z ? -DIFFZH1(VZ_Z_PML) : -DIFFZH1(VZ_Z);
  w_sum += pml_y ? -DIFFY1(VZ_Y_PML) : -DIFFY1(VZ_Y);
  w_sum += pml_x ? -DIFFX1(VZ_X_PML) : -DIFFX1(VZ_X);
  w_sum_bg += pml_z ? -DIFFZH1(VZ_Z_BG_PML) : -DIFFZH1(VZ_Z_BG);
  w_sum_bg += pml_y ? -DIFFY1(VZ_Y_BG_PML) : -DIFFY1(VZ_Y_BG);
  w_sum_bg += pml_x ? -DIFFX1(VZ_X_BG_PML) : -DIFFX1(VZ_X_BG);

  vz[i] += w_sum;
  vz_bg[i] += w_sum_bg;
}

/* M_SIGMA** / M_SIGMA**_bg PML memory-variable update for FUNC(backward_
 * vz_cell) and grad_buoyancy_z/grad_dbuoyancy_z accumulation, split out and
 * moved to run in a separate pass AFTER the velocity fold (see FUNC(
 * backward_sc_vz_cell_pml_mem)'s comment -- same reason: vz[i]/vz_bg[i]
 * must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vz_cell_pml_mem)(
    DW_DTYPE const *__restrict const vz, DW_DTYPE const *__restrict const vz_bg,
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmazzz_new,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmayzy_new,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmaxzx_new,
    DW_DTYPE const *__restrict const m_sigmazzz_bg,
    DW_DTYPE *__restrict const m_sigmazzz_bg_new,
    DW_DTYPE const *__restrict const m_sigmayzy_bg,
    DW_DTYPE *__restrict const m_sigmayzy_bg_new,
    DW_DTYPE const *__restrict const m_sigmaxzx_bg,
    DW_DTYPE *__restrict const m_sigmaxzx_bg_new,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const dbuoyancy_z_shot,
    DW_DTYPE *__restrict const grad_buoyancy_z,
    DW_DTYPE *__restrict const grad_dbuoyancy_z,
    DW_DTYPE const *__restrict const dvzdbuoyancy_store_1,
    DW_DTYPE const *__restrict const dvzdbuoyancy_sc_store_1,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const dt,
    int64_t const step_ratio, bool const buoyancy_requires_grad,
    bool const dbuoyancy_requires_grad, int64_t const i, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, bool const pml_x) {
  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz[i];
    m_sigmazzz_bg_new[i] =
        (buoyancy_z_shot * vz_bg[i] + dbuoyancy_z_shot * vz[i]) * dt *
            P_AZH(0) +
        P_AZH(0) * m_sigmazzz_bg[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy[i];
    m_sigmayzy_bg_new[i] =
        (buoyancy_z_shot * vz_bg[i] + dbuoyancy_z_shot * vz[i]) * dt * P_AY(0) +
        P_AY(0) * m_sigmayzy_bg[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx[i];
    m_sigmaxzx_bg_new[i] =
        (buoyancy_z_shot * vz_bg[i] + dbuoyancy_z_shot * vz[i]) * dt * P_AX(0) +
        P_AX(0) * m_sigmaxzx_bg[i];
  }

  if (dbuoyancy_requires_grad) {
    grad_dbuoyancy_z[i] +=
        vz[i] * dvzdbuoyancy_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (buoyancy_requires_grad) {
    grad_buoyancy_z[i] += (vz_bg[i] * dvzdbuoyancy_store_1[i] +
                           vz[i] * dvzdbuoyancy_sc_store_1[i]) *
                          (DW_DTYPE)step_ratio;
  }
}
#endif

#if DW_NDIM >= 2
/* Full two-field adjoint velocity update, y axis (see
 * FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vy_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy_shot,
#endif
    DW_DTYPE const *__restrict const mu_yx_shot,
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dmu_zy_shot,
#endif
    DW_DTYPE const *__restrict const dmu_yx_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz,
#endif
    DW_DTYPE const *__restrict const sigmaxy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz_bg,
#endif
    DW_DTYPE const *__restrict const sigmayy_bg,
    DW_DTYPE const *__restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz_bg,
#endif
    DW_DTYPE const *__restrict const sigmaxy_bg, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const vy_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vyz,
#endif
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vyz_bg,
#endif
    DW_DTYPE const *__restrict const m_vyy_bg,
    DW_DTYPE const *__restrict const m_vyx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bxh,
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
  DW_DTYPE w_sum = 0, w_sum_bg = 0;
#if DW_NDIM >= 3
  w_sum += pml_z ? -DIFFZ1(VY_Z_PML) : -DIFFZ1(VY_Z);
  w_sum_bg += pml_z ? -DIFFZ1(VY_Z_BG_PML) : -DIFFZ1(VY_Z_BG);
#endif
  w_sum += pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y);
  w_sum += pml_x ? -DIFFX1(VY_X_PML) : -DIFFX1(VY_X);
  w_sum_bg += pml_y ? -DIFFYH1(VY_Y_BG_PML) : -DIFFYH1(VY_Y_BG);
  w_sum_bg += pml_x ? -DIFFX1(VY_X_BG_PML) : -DIFFX1(VY_X_BG);

  vy[i] += w_sum;
  vy_bg[i] += w_sum_bg;
}

/* See FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason). */
DW_HOST_DEVICE void FUNC(backward_vy_cell_pml_mem)(
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vy_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmayzz_new,
    DW_DTYPE const *__restrict const m_sigmayzz_bg,
    DW_DTYPE *__restrict const m_sigmayzz_bg_new,
#endif
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmayyy_new,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxyx_new,
    DW_DTYPE const *__restrict const m_sigmayyy_bg,
    DW_DTYPE *__restrict const m_sigmayyy_bg_new,
    DW_DTYPE const *__restrict const m_sigmaxyx_bg,
    DW_DTYPE *__restrict const m_sigmaxyx_bg_new,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE const dbuoyancy_y_shot,
    DW_DTYPE *__restrict const grad_buoyancy_y,
    DW_DTYPE *__restrict const grad_dbuoyancy_y,
    DW_DTYPE const *__restrict const dvydbuoyancy_store_1,
    DW_DTYPE const *__restrict const dvydbuoyancy_sc_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az,
#endif
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const dt, int64_t const step_ratio,
    bool const buoyancy_requires_grad, bool const dbuoyancy_requires_grad,
    int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z, bool const pml_z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x) {
#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz[i];
    m_sigmayzz_bg_new[i] =
        (buoyancy_y_shot * vy_bg[i] + dbuoyancy_y_shot * vy[i]) * dt * P_AZ(0) +
        P_AZ(0) * m_sigmayzz_bg[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy[i];
    m_sigmayyy_bg_new[i] =
        (buoyancy_y_shot * vy_bg[i] + dbuoyancy_y_shot * vy[i]) * dt *
            P_AYH(0) +
        P_AYH(0) * m_sigmayyy_bg[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx[i];
    m_sigmaxyx_bg_new[i] =
        (buoyancy_y_shot * vy_bg[i] + dbuoyancy_y_shot * vy[i]) * dt * P_AX(0) +
        P_AX(0) * m_sigmaxyx_bg[i];
  }

  if (dbuoyancy_requires_grad) {
    grad_dbuoyancy_y[i] +=
        vy[i] * dvydbuoyancy_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (buoyancy_requires_grad) {
    grad_buoyancy_y[i] += (vy_bg[i] * dvydbuoyancy_store_1[i] +
                           vy[i] * dvydbuoyancy_sc_store_1[i]) *
                          (DW_DTYPE)step_ratio;
  }
}
#endif

/* Full two-field adjoint velocity update, x axis (see
 * FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx_shot,
#endif
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dmu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dmu_yx_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy_bg,
#endif
    DW_DTYPE const *__restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy_bg,
#endif
    DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vxy_bg,
#endif
    DW_DTYPE const *__restrict const m_vxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const bx, DW_DTYPE const dt, int64_t const i,
    int64_t const j,
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
  DW_DTYPE w_sum = 0, w_sum_bg = 0;
#if DW_NDIM >= 3
  w_sum += pml_z ? -DIFFZ1(VX_Z_PML) : -DIFFZ1(VX_Z);
  w_sum_bg += pml_z ? -DIFFZ1(VX_Z_BG_PML) : -DIFFZ1(VX_Z_BG);
#endif
#if DW_NDIM >= 2
  w_sum += pml_y ? -DIFFY1(VX_Y_PML) : -DIFFY1(VX_Y);
  w_sum_bg += pml_y ? -DIFFY1(VX_Y_BG_PML) : -DIFFY1(VX_Y_BG);
#endif
  w_sum += pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X);
  w_sum_bg += pml_x ? -DIFFXH1(VX_X_BG_PML) : -DIFFXH1(VX_X_BG);

  vx[i] += w_sum;
  vx_bg[i] += w_sum_bg;
}

/* See FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_pml_mem)(
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE *__restrict const m_sigmaxzz_new,
    DW_DTYPE const *__restrict const m_sigmaxzz_bg,
    DW_DTYPE *__restrict const m_sigmaxzz_bg_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyy_new,
    DW_DTYPE const *__restrict const m_sigmaxyy_bg,
    DW_DTYPE *__restrict const m_sigmaxyy_bg_new,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmaxxx_new,
    DW_DTYPE const *__restrict const m_sigmaxxx_bg,
    DW_DTYPE *__restrict const m_sigmaxxx_bg_new,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE const dbuoyancy_x_shot,
    DW_DTYPE *__restrict const grad_buoyancy_x,
    DW_DTYPE *__restrict const grad_dbuoyancy_x,
    DW_DTYPE const *__restrict const dvxdbuoyancy_store_1,
    DW_DTYPE const *__restrict const dvxdbuoyancy_sc_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay,
#endif
    DW_DTYPE const *__restrict const axh, DW_DTYPE const dt,
    int64_t const step_ratio, bool const buoyancy_requires_grad,
    bool const dbuoyancy_requires_grad, int64_t const i,
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
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz[i];
    m_sigmaxzz_bg_new[i] =
        (buoyancy_x_shot * vx_bg[i] + dbuoyancy_x_shot * vx[i]) * dt * P_AZ(0) +
        P_AZ(0) * m_sigmaxzz_bg[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy[i];
    m_sigmaxyy_bg_new[i] =
        (buoyancy_x_shot * vx_bg[i] + dbuoyancy_x_shot * vx[i]) * dt * P_AY(0) +
        P_AY(0) * m_sigmaxyy_bg[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx[i];
    m_sigmaxxx_bg_new[i] =
        (buoyancy_x_shot * vx_bg[i] + dbuoyancy_x_shot * vx[i]) * dt *
            P_AXH(0) +
        P_AXH(0) * m_sigmaxxx_bg[i];
  }

  if (dbuoyancy_requires_grad) {
    grad_dbuoyancy_x[i] +=
        vx[i] * dvxdbuoyancy_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (buoyancy_requires_grad) {
    grad_buoyancy_x[i] += (vx_bg[i] * dvxdbuoyancy_store_1[i] +
                           vx[i] * dvxdbuoyancy_sc_store_1[i]) *
                          (DW_DTYPE)step_ratio;
  }
}

/* grad_dlamb/grad_dmu accumulation (correlating the SCATTERED adjoint field
 * with the BACKGROUND term stores, as FUNC(backward_sc_sigmaii_cell)) AND
 * grad_lamb/grad_mu accumulation (two correlation terms: background field
 * with the background store, scattered field with the scattered store), the
 * M_V** / M_V**_BG PML memory-variable updates, and
 * sigmazz/sigmayy/sigmaxx += -(d(SIGMAZZ_Z)/dz + ...) (ordinary, scattered)
 * / sigmazz_bg/sigmayy_bg/sigmaxx_bg += -(d(SIGMAZZ_Z_BG)/dz + ...) (BG
 * coupled), in one call. `i`/`j` addressing matches
 * FUNC(backward_sc_sigmaii_cell) above. */
DW_HOST_DEVICE void FUNC(backward_sigmaii_cell)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const dbuoyancy_y_shot,
#endif
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const dbuoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz, DW_DTYPE *__restrict const sigmazz,
    DW_DTYPE const *__restrict const vz_bg,
    DW_DTYPE *__restrict const sigmazz_bg, DW_DTYPE *__restrict const m_vzz,
    DW_DTYPE *__restrict const m_vzz_bg,
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const m_sigmazzz_bg,
    DW_DTYPE const *__restrict const dvzdz_store_1,
    DW_DTYPE const *__restrict const dvzdz_sc_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
    DW_DTYPE const *__restrict const vy_bg,
    DW_DTYPE *__restrict const sigmayy_bg, DW_DTYPE *__restrict const m_vyy,
    DW_DTYPE *__restrict const m_vyy_bg,
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmayyy_bg,
    DW_DTYPE const *__restrict const dvydy_store_1,
    DW_DTYPE const *__restrict const dvydy_sc_store_1,
#endif
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const vx_bg,
    DW_DTYPE *__restrict const sigmaxx_bg, DW_DTYPE *__restrict const m_vxx,
    DW_DTYPE *__restrict const m_vxx_bg,
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE const *__restrict const m_sigmaxxx_bg,
    DW_DTYPE const *__restrict const dvxdx_store_1,
    DW_DTYPE const *__restrict const dvxdx_sc_store_1,
    DW_DTYPE *__restrict const grad_lamb, DW_DTYPE *__restrict const grad_mu,
    DW_DTYPE *__restrict const grad_dlamb, DW_DTYPE *__restrict const grad_dmu,
    DW_DTYPE const dt, int64_t const step_ratio, bool const lamb_requires_grad,
    bool const mu_requires_grad, bool const dlamb_requires_grad,
    bool const dmu_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdx) {
  if (dlamb_requires_grad) {
#if DW_NDIM == 3
    grad_dlamb[i] += (sigmaxx[i] + sigmayy[i] + sigmazz[i]) *
                     (dvxdx_store_1[i] + dvydy_store_1[i] + dvzdz_store_1[i]) *
                     (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_dlamb[i] += (sigmaxx[i] + sigmayy[i]) *
                     (dvxdx_store_1[i] + dvydy_store_1[i]) *
                     (DW_DTYPE)step_ratio;
#else
    grad_dlamb[i] += sigmaxx[i] * dvxdx_store_1[i] * (DW_DTYPE)step_ratio;
#endif
  }
  if (lamb_requires_grad) {
#if DW_NDIM == 3
    grad_lamb[i] +=
        ((sigmaxx_bg[i] + sigmayy_bg[i] + sigmazz_bg[i]) *
             (dvxdx_store_1[i] + dvydy_store_1[i] + dvzdz_store_1[i]) +
         (sigmaxx[i] + sigmayy[i] + sigmazz[i]) *
             (dvxdx_sc_store_1[i] + dvydy_sc_store_1[i] +
              dvzdz_sc_store_1[i])) *
        (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_lamb[i] += ((sigmaxx_bg[i] + sigmayy_bg[i]) *
                         (dvxdx_store_1[i] + dvydy_store_1[i]) +
                     (sigmaxx[i] + sigmayy[i]) *
                         (dvxdx_sc_store_1[i] + dvydy_sc_store_1[i])) *
                    (DW_DTYPE)step_ratio;
#else
    grad_lamb[i] +=
        (sigmaxx_bg[i] * dvxdx_store_1[i] + sigmaxx[i] * dvxdx_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
#endif
  }
  if (dmu_requires_grad) {
#if DW_NDIM == 3
    grad_dmu[i] +=
        2 *
        (sigmaxx[i] * dvxdx_store_1[i] + sigmayy[i] * dvydy_store_1[i] +
         sigmazz[i] * dvzdz_store_1[i]) *
        (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_dmu[i] +=
        2 * (sigmaxx[i] * dvxdx_store_1[i] + sigmayy[i] * dvydy_store_1[i]) *
        (DW_DTYPE)step_ratio;
#else
    grad_dmu[i] += 2 * sigmaxx[i] * dvxdx_store_1[i] * (DW_DTYPE)step_ratio;
#endif
  }
  if (mu_requires_grad) {
#if DW_NDIM == 3
    grad_mu[i] +=
        2 *
        (sigmaxx_bg[i] * dvxdx_store_1[i] + sigmayy_bg[i] * dvydy_store_1[i] +
         sigmazz_bg[i] * dvzdz_store_1[i] + sigmaxx[i] * dvxdx_sc_store_1[i] +
         sigmayy[i] * dvydy_sc_store_1[i] + sigmazz[i] * dvzdz_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_mu[i] +=
        2 *
        (sigmaxx_bg[i] * dvxdx_store_1[i] + sigmayy_bg[i] * dvydy_store_1[i] +
         sigmaxx[i] * dvxdx_sc_store_1[i] + sigmayy[i] * dvydy_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
#else
    grad_mu[i] +=
        2 *
        (sigmaxx_bg[i] * dvxdx_store_1[i] + sigmaxx[i] * dvxdx_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
#endif
  }

#if DW_NDIM >= 3
  if (pml_z) {
    m_vzz[i] = LAMB_2MU(0, 0, 0) * dt * P_AZ(0) * sigmazz[i] +
               LAMB(0, 0, 0) * dt * P_AZ(0) * sigmaxx[i] +
               LAMB(0, 0, 0) * dt * P_AZ(0) * sigmayy[i] + P_AZ(0) * m_vzz[i];
    m_vzz_bg[i] =
        ((LAMB_2MU(0, 0, 0))*sigmazz_bg[i] + LAMB(0, 0, 0) * sigmaxx_bg[i] +
         LAMB(0, 0, 0) * sigmayy_bg[i] + (DLAMB_2MU(0, 0, 0))*sigmazz[i] +
         DLAMB(0, 0, 0) * sigmaxx[i] + DLAMB(0, 0, 0) * sigmayy[i]) *
            dt * P_AZ(0) +
        P_AZ(0) * m_vzz_bg[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_vyy[i] = LAMB_2MU(0, 0, 0) * dt * P_AY(0) * sigmayy[i] +
               LAMB(0, 0, 0) * dt * P_AY(0) * sigmaxx[i] +
#if DW_NDIM == 3
               LAMB(0, 0, 0) * dt * P_AY(0) * sigmazz[i] +
#endif
               P_AY(0) * m_vyy[i];
    m_vyy_bg[i] =
        ((LAMB_2MU(0, 0, 0))*sigmayy_bg[i] + LAMB(0, 0, 0) * sigmaxx_bg[i] +
#if DW_NDIM == 3
         LAMB(0, 0, 0) * sigmazz_bg[i] + DLAMB(0, 0, 0) * sigmazz[i] +
#endif
         (DLAMB_2MU(0, 0, 0))*sigmayy[i] + DLAMB(0, 0, 0) * sigmaxx[i]) *
            dt * P_AY(0) +
        P_AY(0) * m_vyy_bg[i];
  }
#endif
  if (pml_x) {
    m_vxx[i] = LAMB_2MU(0, 0, 0) * dt * P_AX(0) * sigmaxx[i] +
#if DW_NDIM >= 2
               LAMB(0, 0, 0) * dt * P_AX(0) * sigmayy[i] +
#endif
#if DW_NDIM == 3
               LAMB(0, 0, 0) * dt * P_AX(0) * sigmazz[i] +
#endif
               P_AX(0) * m_vxx[i];
    m_vxx_bg[i] = ((LAMB_2MU(0, 0, 0))*sigmaxx_bg[i] +
#if DW_NDIM >= 2
                   LAMB(0, 0, 0) * sigmayy_bg[i] + DLAMB(0, 0, 0) * sigmayy[i] +
#endif
#if DW_NDIM == 3
                   LAMB(0, 0, 0) * sigmazz_bg[i] + DLAMB(0, 0, 0) * sigmazz[i] +
#endif
                   (DLAMB_2MU(0, 0, 0))*sigmaxx[i]) *
                      dt * P_AX(0) +
                  P_AX(0) * m_vxx_bg[i];
  }

#if DW_NDIM >= 3
  sigmazz[i] += pml_z ? -DIFFZ1(SIGMAZZ_Z_PML) : -DIFFZ1(SIGMAZZ_Z);
  sigmazz_bg[i] += pml_z ? -DIFFZ1(SIGMAZZ_Z_BG_PML) : -DIFFZ1(SIGMAZZ_Z_BG);
#endif
#if DW_NDIM >= 2
  sigmayy[i] += pml_y ? -DIFFY1(SIGMAYY_Y_PML) : -DIFFY1(SIGMAYY_Y);
  sigmayy_bg[i] += pml_y ? -DIFFY1(SIGMAYY_Y_BG_PML) : -DIFFY1(SIGMAYY_Y_BG);
#endif
  sigmaxx[i] += pml_x ? -DIFFX1(SIGMAXX_X_PML) : -DIFFX1(SIGMAXX_X);
  sigmaxx_bg[i] += pml_x ? -DIFFX1(SIGMAXX_X_BG_PML) : -DIFFX1(SIGMAXX_X_BG);
}

#if DW_NDIM >= 2
/* grad_dmu_yx and grad_mu_yx accumulation, M_VXY/M_VYX(+_BG) PML memory
 * update, and sigmaxy/sigmaxy_bg stencil update (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxy_cell)(
    DW_DTYPE const *__restrict const mu_yx_shot,
    DW_DTYPE const *__restrict const dmu_yx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const dbuoyancy_x_shot,
    DW_DTYPE const *__restrict const dbuoyancy_y_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const vx_bg,
    DW_DTYPE const *__restrict const vy_bg, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxy_bg, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy_bg,
    DW_DTYPE *__restrict const m_vyx_bg,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE const *__restrict const m_sigmaxyy_bg,
    DW_DTYPE const *__restrict const m_sigmaxyx_bg,
    DW_DTYPE const *__restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE const *__restrict const dvydx_plus_dvxdy_sc_store_1,
    DW_DTYPE *__restrict const grad_mu_yx,
    DW_DTYPE *__restrict const grad_dmu_yx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const mu_requires_grad,
    bool const dmu_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (dmu_requires_grad) {
    grad_dmu_yx[i] +=
        sigmaxy[i] * dvydx_plus_dvxdy_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (mu_requires_grad) {
    grad_mu_yx[i] += (sigmaxy_bg[i] * dvydx_plus_dvxdy_store_1[i] +
                      sigmaxy[i] * dvydx_plus_dvxdy_sc_store_1[i]) *
                     (DW_DTYPE)step_ratio;
  }

  if (pml_y) {
    m_vxy[i] =
        MU_YX(0, 0, 0) * dt * P_AYH(0) * sigmaxy[i] + P_AYH(0) * m_vxy[i];
    m_vxy_bg[i] =
        (MU_YX(0, 0, 0) * sigmaxy_bg[i] + DMU_YX(0, 0, 0) * sigmaxy[i]) * dt *
            P_AYH(0) +
        P_AYH(0) * m_vxy_bg[i];
  }
  if (pml_x) {
    m_vyx[i] =
        MU_YX(0, 0, 0) * dt * P_AXH(0) * sigmaxy[i] + P_AXH(0) * m_vyx[i];
    m_vyx_bg[i] =
        (MU_YX(0, 0, 0) * sigmaxy_bg[i] + DMU_YX(0, 0, 0) * sigmaxy[i]) * dt *
            P_AXH(0) +
        P_AXH(0) * m_vyx_bg[i];
  }

  sigmaxy[i] += (pml_y ? -DIFFYH1(SIGMAXY_Y_PML) : -DIFFYH1(SIGMAXY_Y)) +
                (pml_x ? -DIFFXH1(SIGMAXY_X_PML) : -DIFFXH1(SIGMAXY_X));
  sigmaxy_bg[i] +=
      (pml_y ? -DIFFYH1(SIGMAXY_Y_BG_PML) : -DIFFYH1(SIGMAXY_Y_BG)) +
      (pml_x ? -DIFFXH1(SIGMAXY_X_BG_PML) : -DIFFXH1(SIGMAXY_X_BG));
}
#endif

#if DW_NDIM >= 3
/* grad_dmu_zx and grad_mu_zx accumulation, M_VXZ/M_VZX(+_BG) PML memory
 * update, and sigmaxz/sigmaxz_bg stencil update (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxz_cell)(
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const dmu_zx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const dbuoyancy_x_shot,
    DW_DTYPE const *__restrict const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const vx_bg,
    DW_DTYPE const *__restrict const vz_bg, DW_DTYPE *__restrict const sigmaxz,
    DW_DTYPE *__restrict const sigmaxz_bg, DW_DTYPE *__restrict const m_vxz,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vxz_bg,
    DW_DTYPE *__restrict const m_vzx_bg,
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmaxzz_bg,
    DW_DTYPE const *__restrict const m_sigmaxzx_bg,
    DW_DTYPE const *__restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE const *__restrict const dvzdx_plus_dvxdz_sc_store_1,
    DW_DTYPE *__restrict const grad_mu_zx,
    DW_DTYPE *__restrict const grad_dmu_zx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const mu_requires_grad,
    bool const dmu_requires_grad, int64_t const i, int64_t const j,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_x,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    DW_DTYPE const rdz, DW_DTYPE const rdx) {
  if (dmu_requires_grad) {
    grad_dmu_zx[i] +=
        sigmaxz[i] * dvzdx_plus_dvxdz_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (mu_requires_grad) {
    grad_mu_zx[i] += (sigmaxz_bg[i] * dvzdx_plus_dvxdz_store_1[i] +
                      sigmaxz[i] * dvzdx_plus_dvxdz_sc_store_1[i]) *
                     (DW_DTYPE)step_ratio;
  }

  if (pml_z) {
    m_vxz[i] =
        MU_ZX(0, 0, 0) * dt * P_AZH(0) * sigmaxz[i] + P_AZH(0) * m_vxz[i];
    m_vxz_bg[i] =
        (MU_ZX(0, 0, 0) * sigmaxz_bg[i] + DMU_ZX(0, 0, 0) * sigmaxz[i]) * dt *
            P_AZH(0) +
        P_AZH(0) * m_vxz_bg[i];
  }
  if (pml_x) {
    m_vzx[i] =
        MU_ZX(0, 0, 0) * dt * P_AXH(0) * sigmaxz[i] + P_AXH(0) * m_vzx[i];
    m_vzx_bg[i] =
        (MU_ZX(0, 0, 0) * sigmaxz_bg[i] + DMU_ZX(0, 0, 0) * sigmaxz[i]) * dt *
            P_AXH(0) +
        P_AXH(0) * m_vzx_bg[i];
  }

  sigmaxz[i] += (pml_z ? -DIFFZH1(SIGMAXZ_Z_PML) : -DIFFZH1(SIGMAXZ_Z)) +
                (pml_x ? -DIFFXH1(SIGMAXZ_X_PML) : -DIFFXH1(SIGMAXZ_X));
  sigmaxz_bg[i] +=
      (pml_z ? -DIFFZH1(SIGMAXZ_Z_BG_PML) : -DIFFZH1(SIGMAXZ_Z_BG)) +
      (pml_x ? -DIFFXH1(SIGMAXZ_X_BG_PML) : -DIFFXH1(SIGMAXZ_X_BG));
}

/* grad_dmu_zy and grad_mu_zy accumulation, M_VYZ/M_VZY(+_BG) PML memory
 * update, and sigmayz/sigmayz_bg stencil update (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmayz_cell)(
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const dmu_zy_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const dbuoyancy_y_shot,
    DW_DTYPE const *__restrict const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const vy_bg,
    DW_DTYPE const *__restrict const vz_bg, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmayz_bg, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vzy, DW_DTYPE *__restrict const m_vyz_bg,
    DW_DTYPE *__restrict const m_vzy_bg,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmayzz_bg,
    DW_DTYPE const *__restrict const m_sigmayzy_bg,
    DW_DTYPE const *__restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE const *__restrict const dvzdy_plus_dvydz_sc_store_1,
    DW_DTYPE *__restrict const grad_mu_zy,
    DW_DTYPE *__restrict const grad_dmu_zy, DW_DTYPE const dt,
    int64_t const step_ratio, bool const mu_requires_grad,
    bool const dmu_requires_grad, int64_t const i, int64_t const j,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_y,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const rdz, DW_DTYPE const rdy) {
  if (dmu_requires_grad) {
    grad_dmu_zy[i] +=
        sigmayz[i] * dvzdy_plus_dvydz_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (mu_requires_grad) {
    grad_mu_zy[i] += (sigmayz_bg[i] * dvzdy_plus_dvydz_store_1[i] +
                      sigmayz[i] * dvzdy_plus_dvydz_sc_store_1[i]) *
                     (DW_DTYPE)step_ratio;
  }

  if (pml_z) {
    m_vyz[i] =
        MU_ZY(0, 0, 0) * dt * P_AZH(0) * sigmayz[i] + P_AZH(0) * m_vyz[i];
    m_vyz_bg[i] =
        (MU_ZY(0, 0, 0) * sigmayz_bg[i] + DMU_ZY(0, 0, 0) * sigmayz[i]) * dt *
            P_AZH(0) +
        P_AZH(0) * m_vyz_bg[i];
  }
  if (pml_y) {
    m_vzy[i] =
        MU_ZY(0, 0, 0) * dt * P_AYH(0) * sigmayz[i] + P_AYH(0) * m_vzy[i];
    m_vzy_bg[i] =
        (MU_ZY(0, 0, 0) * sigmayz_bg[i] + DMU_ZY(0, 0, 0) * sigmayz[i]) * dt *
            P_AYH(0) +
        P_AYH(0) * m_vzy_bg[i];
  }

  sigmayz[i] += (pml_z ? -DIFFZH1(SIGMAYZ_Z_PML) : -DIFFZH1(SIGMAYZ_Z)) +
                (pml_y ? -DIFFYH1(SIGMAYZ_Y_PML) : -DIFFYH1(SIGMAYZ_Y));
  sigmayz_bg[i] +=
      (pml_z ? -DIFFZH1(SIGMAYZ_Z_BG_PML) : -DIFFZH1(SIGMAYZ_Z_BG)) +
      (pml_y ? -DIFFYH1(SIGMAYZ_Y_BG_PML) : -DIFFYH1(SIGMAYZ_Y_BG));
}
#endif

/* ========================= Guard-band gathers ========================= *
 *
 * The backward gather kernels above (FUNC(backward_v*_cell)/FUNC(backward_
 * sigma*_cell)/FUNC(backward_sc_v*_cell)/FUNC(backward_sc_sigma*_cell)) only
 * ever run over the grid's strict interior/PML region (elastic_born.c's/
 * elastic_born.cu's pml_bounds_* / pml_bounds_*h arrays both start at
 * FD_PAD and stop FD_PAD or FD_PAD-1 short of the far edge), so they never
 * deposit an adjoint contribution into the guard ring next to a reflecting
 * (non-PML) face. FUNC(fill_elastic_stress_bc_adjoint)/FUNC(fill_elastic_
 * velocity_bc_adjoint) (see the module comment above those, elastic_born.c)
 * then fold that guard ring into the interior -- but with nothing genuine
 * ever written there, that fold silently discards real gradient
 * contributions. The functions below are guard-band-only companions to the
 * interior gathers: same physics (transpose of the corresponding FUNC(
 * forward_*_cell)), but evaluated with bounds-safe stencil taps (DW_DIFF*_
 * SAFE, which treat any tap landing outside [0, nz)/[0, ny)/[0, nx) as
 * exactly zero -- see elastic_kernels.h's identical mechanism, ported
 * unchanged here) so they may be called at guard-ring cells, where a
 * stencil can reach up to FD_PAD cells past the physical array. They omit
 * the grad_lamb/grad_mu/grad_mu_* /grad_dlamb/grad_dmu/grad_dmu_*
 * accumulation the interior functions perform (guard cells are not
 * independent model-gradient locations -- the model itself is only
 * mirror-padded, never an independent DOF there) and, for the velocity
 * gathers, the buoyancy-gradient/dbuoyancy-gradient accumulation the
 * interior functions used to perform too (moved to a separate post-fold
 * pass -- see elastic_born.c's/elastic_born.cu's driver-side comment at its
 * new call site -- since a pre-fold read of a near-face interior cell is
 * stale once the fold changes it).
 *
 * elastic_born.c's/elastic_born.cu's new guard-band driver loops call these
 * over exactly the *new* cells a field's boundary fill touches beyond the
 * existing interior/PML loop -- see FUNC(dw_compute_ext_range) below -- so
 * double-accumulation against the interior loop's own cells never occurs.
 * Every axis argument here (including ones that happen to be this call's
 * "currently extended" axis, which is always non-PML by construction) is
 * treated fully generally: a transverse axis can legitimately be PML-active
 * even while another axis of the same cell is in that axis's guard ring, so
 * is_pml_z/is_pml_y/is_pml_x are ordinary parameters here exactly as in the
 * interior functions -- elastic.c's landed driver, the reference this ports
 * from, hardcodes them false at every guard call site regardless
 * (comment: "Guard cells have no forward-pass PML state"); this file's
 * driver reproduces that same call-site choice rather than the more general
 * treatment these parameters would otherwise allow, matching the verified
 * reference exactly rather than re-deriving it.
 *
 * Two independent field sets need this treatment: the scattered-only
 * adjoint (FUNC(backward_sc_*_cell), driven by FUNC(backward_sc)) and the
 * full two-field adjoint (FUNC(backward_*_cell), driven by FUNC(backward)),
 * which fuses the background (_bg) and scattered (plain) adjoint updates
 * into one call per cell exactly as their non-guard interior counterparts
 * do.
 */

/* Guard-extension bookkeeping for one field's use of one axis. `n` is that
 * axis's full length (nz/ny/nx); `bc_lo`/`bc_hi` are that axis's two face
 * codes (0 = pml, 1 = free, 2 = rigid, 3 = periodic); `is_half` selects the
 * half-grid convention (velocity's own axis; a shear stress's two own axes)
 * vs full-grid (everything else). For an axis this field does not touch at
 * all, callers pass `bc_lo=bc_hi=0` regardless of that axis's real face
 * codes, so `target` degenerates to `std` and the new-cell test below is
 * trivially false everywhere on that axis.
 *
 * `target_lo/target_hi` is the field's full boundary-fill-aware target
 * range on this axis; `std_lo/std_hi` is the existing interior/PML loop's
 * own (unextended) range on this axis (matches pml_bounds_z/pml_bounds_zh
 * etc.'s own [FD_PAD, n-FD_PAD+1) / [FD_PAD, n-FD_PAD) span) -- cells with
 * EVERY touched axis inside `std` are already handled by the interior loop
 * and must be skipped -- callers do this with FUNC(dw_in_target) AND
 * NOT-ALL-axes-FUNC(dw_in_std) below. `has_exclude`/`exclude_idx`
 * implements the half-lattice "extra node" (`n - FD_PAD`) rule: under a
 * half-grid axis's free or rigid high face that node is excluded from the
 * target set (a live untouched DOF under free; a clamped, fold-zeroed DOF
 * under rigid -- either way not part of the new-cell set), included under
 * periodic (the slaved node the fold-fill's `bc_hi==3` step folds and
 * zeros).
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
 * sub-region, replicating elastic_born.c's/elastic_born.cu's own region
 * split (pml_bounds_* / pml_bounds_*h: region 0 = [FD_PAD, pml_lo), region
 * 2 = [pml_hi, std_hi)) so a guard-band gather agrees with the interior
 * gather's own treatment of any cell both loops cover. `pml_hi` must be
 * pre-adjusted by the caller for a half-grid axis (`MAX(pml_lo,
 * pml_hi_raw - 1)`, matching pml_bounds_*h's own construction) before being
 * passed in here; `std_hi` should be the same-role FUNC(dw_ext_range)'s
 * `std_hi`. Not currently called by this file's driver (which hardcodes
 * is_pml_* false at every guard call site -- see the module comment above)
 * but kept for parity with the elastic_kernels.h/elastic_vti_kernels.h
 * ports of this same infrastructure. */
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
 * tap landing outside contributes exactly zero. */
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
 * tap is in bounds. elastic_born_kernels.h may not modify staggered_grid.h
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

/* ---------------- Scattered-only adjoint (backward_sc) guards ---------- */

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_sc_vz_cell)'s gather (no
 * grad_dbuoyancy_z accumulation -- see the module comment above). `nz` is
 * required here (unlike FUNC(backward_sc_vz_cell), which never needs it for
 * flat indexing) purely for DW_SAFE_TAP_Z's bounds check. */
DW_HOST_DEVICE void FUNC(backward_sc_vz_cell_guard)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz, DW_DTYPE *__restrict const vz,
    DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmazzz_new,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmayzy_new,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
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
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx[i];
  }
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_sc_vy_cell)'s gather (see
 * FUNC(backward_sc_vz_cell_guard)). `ny` is required unconditionally here
 * purely for DW_SAFE_TAP_Y's bounds check. */
DW_HOST_DEVICE void FUNC(backward_sc_vy_cell_guard)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy_shot,
#endif
    DW_DTYPE const *__restrict const mu_yx_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz,
#endif
    DW_DTYPE const *__restrict const sigmaxy, DW_DTYPE *__restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vyz,
#endif
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmayyy_new,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
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
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx[i];
  }
}
#endif

/* Guard-band companion to FUNC(backward_sc_vx_cell)'s gather (see
 * FUNC(backward_sc_vz_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sc_vx_cell_guard)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE *__restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
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
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx[i];
  }
}

#if DW_NDIM >= 3
/* Guard-band companion to the sigmazz sub-update inside FUNC(backward_
 * sc_sigmaii_cell): sigmazz's only term is its own (z) axis, so a guard
 * cell here (which by construction only exists on an axis with a non-PML
 * face) is never inside a PML region on that same axis; the M_VZZ PML-
 * memory write and grad_dlamb/grad_dmu the interior function performs are
 * both left entirely to the interior function -- see the module comment
 * above. */
DW_HOST_DEVICE void FUNC(backward_sc_sigmazz_z_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const vz, DW_DTYPE *__restrict const sigmazz,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, DW_DTYPE const rdz) {
  sigmazz[i] += -DW_DIFFZ1_SAFE(SIGMAZZ_Z);
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to the sigmayy sub-update inside FUNC(backward_
 * sc_sigmaii_cell) (see FUNC(backward_sc_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmayy_y_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const y, DW_DTYPE const rdy) {
  sigmayy[i] += -DW_DIFFY1_SAFE(SIGMAYY_Y);
}
#endif

/* Guard-band companion to the sigmaxx sub-update inside FUNC(backward_
 * sc_sigmaii_cell) (see FUNC(backward_sc_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmaxx_x_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const x, DW_DTYPE const rdx) {
  sigmaxx[i] += -DW_DIFFX1_SAFE(SIGMAXX_X);
}

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_sc_sigmaxy_cell)'s gather (keeps
 * the PML memory-variable read/write intact since sigmaxy's guard ring may
 * be genuinely PML-active on its *other* axis). `ny` is required
 * unconditionally here purely for DW_SAFE_TAP_Y's bounds check. */
DW_HOST_DEVICE void FUNC(backward_sc_sigmaxy_cell_guard)(
    DW_DTYPE const *__restrict const mu_yx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vy,
    DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx, DW_DTYPE const dt,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz, int64_t const z,
#endif
    int64_t const ny, int64_t const nx, int64_t const y, int64_t const x,
    bool const pml_y, bool const pml_x, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdy,
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
/* Guard-band companion to FUNC(backward_sc_sigmaxz_cell)'s gather (see
 * FUNC(backward_sc_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmaxz_cell_guard)(
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vz,
    DW_DTYPE *__restrict const sigmaxz, DW_DTYPE *__restrict const m_vxz,
    DW_DTYPE *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE const *__restrict const m_sigmaxzx, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_x, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdz,
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

/* Guard-band companion to FUNC(backward_sc_sigmayz_cell)'s gather (see
 * FUNC(backward_sc_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sc_sigmayz_cell_guard)(
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vz,
    DW_DTYPE *__restrict const sigmayz, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmayzy, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdz,
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

/* ---------------- Full two-field adjoint (backward) guards ------------- */

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_vz_cell)'s gather, fused over both
 * the background (_bg) and scattered (plain) adjoint fields exactly as its
 * interior counterpart is (no grad_buoyancy_z/grad_dbuoyancy_z
 * accumulation -- see the module comment above). `nz` is required here
 * purely for DW_SAFE_TAP_Z's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_guard)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
    DW_DTYPE const *__restrict const dmu_zy_shot,
    DW_DTYPE const *__restrict const dmu_zx_shot,
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
    DW_DTYPE const *__restrict const sigmazz_bg,
    DW_DTYPE const *__restrict const sigmayy_bg,
    DW_DTYPE const *__restrict const sigmaxx_bg,
    DW_DTYPE const *__restrict const sigmayz_bg,
    DW_DTYPE const *__restrict const sigmaxz_bg, DW_DTYPE *__restrict const vz,
    DW_DTYPE *__restrict const vz_bg, DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_vzz_bg,
    DW_DTYPE const *__restrict const m_vzy_bg,
    DW_DTYPE const *__restrict const m_vzx_bg,
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmazzz_new,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmayzy_new,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmaxzx_new,
    DW_DTYPE const *__restrict const m_sigmazzz_bg,
    DW_DTYPE *__restrict const m_sigmazzz_bg_new,
    DW_DTYPE const *__restrict const m_sigmayzy_bg,
    DW_DTYPE *__restrict const m_sigmayzy_bg_new,
    DW_DTYPE const *__restrict const m_sigmaxzx_bg,
    DW_DTYPE *__restrict const m_sigmaxzx_bg_new,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdz, DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0, w_sum_bg = 0;
  w_sum += pml_z ? -DW_DIFFZH1_SAFE(VZ_Z_PML) : -DW_DIFFZH1_SAFE(VZ_Z);
  w_sum += pml_y ? -DW_DIFFY1_SAFE(VZ_Y_PML) : -DW_DIFFY1_SAFE(VZ_Y);
  w_sum += pml_x ? -DW_DIFFX1_SAFE(VZ_X_PML) : -DW_DIFFX1_SAFE(VZ_X);
  w_sum_bg += pml_z ? -DW_DIFFZH1_SAFE(VZ_Z_BG_PML) : -DW_DIFFZH1_SAFE(VZ_Z_BG);
  w_sum_bg += pml_y ? -DW_DIFFY1_SAFE(VZ_Y_BG_PML) : -DW_DIFFY1_SAFE(VZ_Y_BG);
  w_sum_bg += pml_x ? -DW_DIFFX1_SAFE(VZ_X_BG_PML) : -DW_DIFFX1_SAFE(VZ_X_BG);

  vz[i] += w_sum;
  vz_bg[i] += w_sum_bg;

  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz[i];
    m_sigmazzz_bg_new[i] =
        (buoyancy_z_shot * vz_bg[i] + dbuoyancy_z_shot * vz[i]) * dt *
            P_AZH(0) +
        P_AZH(0) * m_sigmazzz_bg[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy[i];
    m_sigmayzy_bg_new[i] =
        (buoyancy_z_shot * vz_bg[i] + dbuoyancy_z_shot * vz[i]) * dt * P_AY(0) +
        P_AY(0) * m_sigmayzy_bg[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx[i];
    m_sigmaxzx_bg_new[i] =
        (buoyancy_z_shot * vz_bg[i] + dbuoyancy_z_shot * vz[i]) * dt * P_AX(0) +
        P_AX(0) * m_sigmaxzx_bg[i];
  }
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_vy_cell)'s gather (see
 * FUNC(backward_vz_cell_guard)). `ny` is required unconditionally here
 * purely for DW_SAFE_TAP_Y's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vy_cell_guard)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy_shot,
#endif
    DW_DTYPE const *__restrict const mu_yx_shot,
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dmu_zy_shot,
#endif
    DW_DTYPE const *__restrict const dmu_yx_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz,
#endif
    DW_DTYPE const *__restrict const sigmaxy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz_bg,
#endif
    DW_DTYPE const *__restrict const sigmayy_bg,
    DW_DTYPE const *__restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmayz_bg,
#endif
    DW_DTYPE const *__restrict const sigmaxy_bg, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const vy_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vyz,
#endif
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vyz_bg,
#endif
    DW_DTYPE const *__restrict const m_vyy_bg,
    DW_DTYPE const *__restrict const m_vyx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmayyy_new,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxyx_new,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmayzz_bg,
    DW_DTYPE *__restrict const m_sigmayzz_bg_new,
#endif
    DW_DTYPE const *__restrict const m_sigmayyy_bg,
    DW_DTYPE *__restrict const m_sigmayyy_bg_new,
    DW_DTYPE const *__restrict const m_sigmaxyx_bg,
    DW_DTYPE *__restrict const m_sigmaxyx_bg_new,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE const dbuoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
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
  DW_DTYPE w_sum = 0, w_sum_bg = 0;
#if DW_NDIM >= 3
  w_sum += pml_z ? -DW_DIFFZ1_SAFE(VY_Z_PML) : -DW_DIFFZ1_SAFE(VY_Z);
  w_sum_bg += pml_z ? -DW_DIFFZ1_SAFE(VY_Z_BG_PML) : -DW_DIFFZ1_SAFE(VY_Z_BG);
#endif
  w_sum += pml_y ? -DW_DIFFYH1_SAFE(VY_Y_PML) : -DW_DIFFYH1_SAFE(VY_Y);
  w_sum += pml_x ? -DW_DIFFX1_SAFE(VY_X_PML) : -DW_DIFFX1_SAFE(VY_X);
  w_sum_bg += pml_y ? -DW_DIFFYH1_SAFE(VY_Y_BG_PML) : -DW_DIFFYH1_SAFE(VY_Y_BG);
  w_sum_bg += pml_x ? -DW_DIFFX1_SAFE(VY_X_BG_PML) : -DW_DIFFX1_SAFE(VY_X_BG);

  vy[i] += w_sum;
  vy_bg[i] += w_sum_bg;

#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz[i];
    m_sigmayzz_bg_new[i] =
        (buoyancy_y_shot * vy_bg[i] + dbuoyancy_y_shot * vy[i]) * dt * P_AZ(0) +
        P_AZ(0) * m_sigmayzz_bg[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy[i];
    m_sigmayyy_bg_new[i] =
        (buoyancy_y_shot * vy_bg[i] + dbuoyancy_y_shot * vy[i]) * dt *
            P_AYH(0) +
        P_AYH(0) * m_sigmayyy_bg[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx[i];
    m_sigmaxyx_bg_new[i] =
        (buoyancy_y_shot * vy_bg[i] + dbuoyancy_y_shot * vy[i]) * dt * P_AX(0) +
        P_AX(0) * m_sigmaxyx_bg[i];
  }
}
#endif

/* Guard-band companion to FUNC(backward_vx_cell)'s gather (see
 * FUNC(backward_vz_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_guard)(
    DW_DTYPE const *__restrict const lamb_shot,
    DW_DTYPE const *__restrict const mu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx_shot,
#endif
    DW_DTYPE const *__restrict const dlamb_shot,
    DW_DTYPE const *__restrict const dmu_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dmu_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dmu_yx_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy_bg,
#endif
    DW_DTYPE const *__restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmaxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmaxy_bg,
#endif
    DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vxy_bg,
#endif
    DW_DTYPE const *__restrict const m_vxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE *__restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmaxxx_new,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmaxzz_bg,
    DW_DTYPE *__restrict const m_sigmaxzz_bg_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmaxyy_bg,
    DW_DTYPE *__restrict const m_sigmaxyy_bg_new,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx_bg,
    DW_DTYPE *__restrict const m_sigmaxxx_bg_new,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE const dbuoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
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
  DW_DTYPE w_sum = 0, w_sum_bg = 0;
#if DW_NDIM >= 3
  w_sum += pml_z ? -DW_DIFFZ1_SAFE(VX_Z_PML) : -DW_DIFFZ1_SAFE(VX_Z);
  w_sum_bg += pml_z ? -DW_DIFFZ1_SAFE(VX_Z_BG_PML) : -DW_DIFFZ1_SAFE(VX_Z_BG);
#endif
#if DW_NDIM >= 2
  w_sum += pml_y ? -DW_DIFFY1_SAFE(VX_Y_PML) : -DW_DIFFY1_SAFE(VX_Y);
  w_sum_bg += pml_y ? -DW_DIFFY1_SAFE(VX_Y_BG_PML) : -DW_DIFFY1_SAFE(VX_Y_BG);
#endif
  w_sum += pml_x ? -DW_DIFFXH1_SAFE(VX_X_PML) : -DW_DIFFXH1_SAFE(VX_X);
  w_sum_bg += pml_x ? -DW_DIFFXH1_SAFE(VX_X_BG_PML) : -DW_DIFFXH1_SAFE(VX_X_BG);

  vx[i] += w_sum;
  vx_bg[i] += w_sum_bg;

#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmaxzz_new[i] =
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz[i];
    m_sigmaxzz_bg_new[i] =
        (buoyancy_x_shot * vx_bg[i] + dbuoyancy_x_shot * vx[i]) * dt * P_AZ(0) +
        P_AZ(0) * m_sigmaxzz_bg[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy[i];
    m_sigmaxyy_bg_new[i] =
        (buoyancy_x_shot * vx_bg[i] + dbuoyancy_x_shot * vx[i]) * dt * P_AY(0) +
        P_AY(0) * m_sigmaxyy_bg[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx[i];
    m_sigmaxxx_bg_new[i] =
        (buoyancy_x_shot * vx_bg[i] + dbuoyancy_x_shot * vx[i]) * dt *
            P_AXH(0) +
        P_AXH(0) * m_sigmaxxx_bg[i];
  }
}

#if DW_NDIM >= 3
/* Guard-band companion to the sigmazz/sigmazz_bg sub-update inside FUNC(
 * backward_sigmaii_cell): sigmazz's only term is its own (z) axis, so a
 * guard cell here is never inside a PML region on that same axis; the
 * M_VZZ/M_VZZ_BG PML-memory writes and grad_lamb/grad_mu/grad_dlamb/
 * grad_dmu the interior function performs are left entirely to it -- see
 * the module comment above. */
DW_HOST_DEVICE void FUNC(backward_sigmazz_z_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const vz, DW_DTYPE *__restrict const sigmazz,
    DW_DTYPE const *__restrict const vz_bg,
    DW_DTYPE *__restrict const sigmazz_bg, DW_DTYPE const dt, int64_t const i,
    int64_t const j, int64_t const nz, int64_t const ny, int64_t const nx,
    int64_t const z, DW_DTYPE const rdz) {
  sigmazz[i] += -DW_DIFFZ1_SAFE(SIGMAZZ_Z);
  sigmazz_bg[i] += -DW_DIFFZ1_SAFE(SIGMAZZ_Z_BG);
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to the sigmayy/sigmayy_bg sub-update inside FUNC(
 * backward_sigmaii_cell) (see FUNC(backward_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmayy_y_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const dbuoyancy_y_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
    DW_DTYPE const *__restrict const vy_bg,
    DW_DTYPE *__restrict const sigmayy_bg, DW_DTYPE const dt, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const y,
    DW_DTYPE const rdy) {
  sigmayy[i] += -DW_DIFFY1_SAFE(SIGMAYY_Y);
  sigmayy_bg[i] += -DW_DIFFY1_SAFE(SIGMAYY_Y_BG);
}
#endif

/* Guard-band companion to the sigmaxx/sigmaxx_bg sub-update inside FUNC(
 * backward_sigmaii_cell) (see FUNC(backward_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxx_x_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const dbuoyancy_x_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const vx_bg,
    DW_DTYPE *__restrict const sigmaxx_bg, DW_DTYPE const dt, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const x, DW_DTYPE const rdx) {
  sigmaxx[i] += -DW_DIFFX1_SAFE(SIGMAXX_X);
  sigmaxx_bg[i] += -DW_DIFFX1_SAFE(SIGMAXX_X_BG);
}

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_sigmaxy_cell)'s gather, fused over
 * both adjoint fields (keeps the M_V** / M_V**_BG PML memory-variable
 * read/write intact since sigmaxy's guard ring may be genuinely PML-active
 * on its *other* axis). `ny` is required unconditionally here purely for
 * DW_SAFE_TAP_Y's bounds check. */
DW_HOST_DEVICE void FUNC(backward_sigmaxy_cell_guard)(
    DW_DTYPE const *__restrict const mu_yx_shot,
    DW_DTYPE const *__restrict const dmu_yx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const dbuoyancy_x_shot,
    DW_DTYPE const *__restrict const dbuoyancy_y_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const vx_bg,
    DW_DTYPE const *__restrict const vy_bg, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxy_bg, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy_bg,
    DW_DTYPE *__restrict const m_vyx_bg,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE const *__restrict const m_sigmaxyy_bg,
    DW_DTYPE const *__restrict const m_sigmaxyx_bg, DW_DTYPE const dt,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz, int64_t const z,
#endif
    int64_t const ny, int64_t const nx, int64_t const y, int64_t const x,
    bool const pml_y, bool const pml_x, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (pml_y) {
    m_vxy[i] =
        MU_YX(0, 0, 0) * dt * P_AYH(0) * sigmaxy[i] + P_AYH(0) * m_vxy[i];
    m_vxy_bg[i] =
        (MU_YX(0, 0, 0) * sigmaxy_bg[i] + DMU_YX(0, 0, 0) * sigmaxy[i]) * dt *
            P_AYH(0) +
        P_AYH(0) * m_vxy_bg[i];
  }
  if (pml_x) {
    m_vyx[i] =
        MU_YX(0, 0, 0) * dt * P_AXH(0) * sigmaxy[i] + P_AXH(0) * m_vyx[i];
    m_vyx_bg[i] =
        (MU_YX(0, 0, 0) * sigmaxy_bg[i] + DMU_YX(0, 0, 0) * sigmaxy[i]) * dt *
            P_AXH(0) +
        P_AXH(0) * m_vyx_bg[i];
  }

  sigmaxy[i] +=
      (pml_y ? -DW_DIFFYH1_SAFE(SIGMAXY_Y_PML) : -DW_DIFFYH1_SAFE(SIGMAXY_Y)) +
      (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXY_X_PML) : -DW_DIFFXH1_SAFE(SIGMAXY_X));
  sigmaxy_bg[i] += (pml_y ? -DW_DIFFYH1_SAFE(SIGMAXY_Y_BG_PML)
                          : -DW_DIFFYH1_SAFE(SIGMAXY_Y_BG)) +
                   (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXY_X_BG_PML)
                          : -DW_DIFFXH1_SAFE(SIGMAXY_X_BG));
}
#endif

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_sigmaxz_cell)'s gather (see
 * FUNC(backward_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxz_cell_guard)(
    DW_DTYPE const *__restrict const mu_zx_shot,
    DW_DTYPE const *__restrict const dmu_zx_shot,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const dbuoyancy_x_shot,
    DW_DTYPE const *__restrict const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const vx_bg,
    DW_DTYPE const *__restrict const vz_bg, DW_DTYPE *__restrict const sigmaxz,
    DW_DTYPE *__restrict const sigmaxz_bg, DW_DTYPE *__restrict const m_vxz,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vxz_bg,
    DW_DTYPE *__restrict const m_vzx_bg,
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmaxzz_bg,
    DW_DTYPE const *__restrict const m_sigmaxzx_bg, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_x, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdz,
    DW_DTYPE const rdx) {
  if (pml_z) {
    m_vxz[i] =
        MU_ZX(0, 0, 0) * dt * P_AZH(0) * sigmaxz[i] + P_AZH(0) * m_vxz[i];
    m_vxz_bg[i] =
        (MU_ZX(0, 0, 0) * sigmaxz_bg[i] + DMU_ZX(0, 0, 0) * sigmaxz[i]) * dt *
            P_AZH(0) +
        P_AZH(0) * m_vxz_bg[i];
  }
  if (pml_x) {
    m_vzx[i] =
        MU_ZX(0, 0, 0) * dt * P_AXH(0) * sigmaxz[i] + P_AXH(0) * m_vzx[i];
    m_vzx_bg[i] =
        (MU_ZX(0, 0, 0) * sigmaxz_bg[i] + DMU_ZX(0, 0, 0) * sigmaxz[i]) * dt *
            P_AXH(0) +
        P_AXH(0) * m_vzx_bg[i];
  }

  sigmaxz[i] +=
      (pml_z ? -DW_DIFFZH1_SAFE(SIGMAXZ_Z_PML) : -DW_DIFFZH1_SAFE(SIGMAXZ_Z)) +
      (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXZ_X_PML) : -DW_DIFFXH1_SAFE(SIGMAXZ_X));
  sigmaxz_bg[i] += (pml_z ? -DW_DIFFZH1_SAFE(SIGMAXZ_Z_BG_PML)
                          : -DW_DIFFZH1_SAFE(SIGMAXZ_Z_BG)) +
                   (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXZ_X_BG_PML)
                          : -DW_DIFFXH1_SAFE(SIGMAXZ_X_BG));
}

/* Guard-band companion to FUNC(backward_sigmayz_cell)'s gather (see
 * FUNC(backward_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmayz_cell_guard)(
    DW_DTYPE const *__restrict const mu_zy_shot,
    DW_DTYPE const *__restrict const dmu_zy_shot,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const dbuoyancy_y_shot,
    DW_DTYPE const *__restrict const dbuoyancy_z_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const vy_bg,
    DW_DTYPE const *__restrict const vz_bg, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmayz_bg, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vzy, DW_DTYPE *__restrict const m_vyz_bg,
    DW_DTYPE *__restrict const m_vzy_bg,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmayzz_bg,
    DW_DTYPE const *__restrict const m_sigmayzy_bg, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdz,
    DW_DTYPE const rdy) {
  if (pml_z) {
    m_vyz[i] =
        MU_ZY(0, 0, 0) * dt * P_AZH(0) * sigmayz[i] + P_AZH(0) * m_vyz[i];
    m_vyz_bg[i] =
        (MU_ZY(0, 0, 0) * sigmayz_bg[i] + DMU_ZY(0, 0, 0) * sigmayz[i]) * dt *
            P_AZH(0) +
        P_AZH(0) * m_vyz_bg[i];
  }
  if (pml_y) {
    m_vzy[i] =
        MU_ZY(0, 0, 0) * dt * P_AYH(0) * sigmayz[i] + P_AYH(0) * m_vzy[i];
    m_vzy_bg[i] =
        (MU_ZY(0, 0, 0) * sigmayz_bg[i] + DMU_ZY(0, 0, 0) * sigmayz[i]) * dt *
            P_AYH(0) +
        P_AYH(0) * m_vzy_bg[i];
  }

  sigmayz[i] +=
      (pml_z ? -DW_DIFFZH1_SAFE(SIGMAYZ_Z_PML) : -DW_DIFFZH1_SAFE(SIGMAYZ_Z)) +
      (pml_y ? -DW_DIFFYH1_SAFE(SIGMAYZ_Y_PML) : -DW_DIFFYH1_SAFE(SIGMAYZ_Y));
  sigmayz_bg[i] += (pml_z ? -DW_DIFFZH1_SAFE(SIGMAYZ_Z_BG_PML)
                          : -DW_DIFFZH1_SAFE(SIGMAYZ_Z_BG)) +
                   (pml_y ? -DW_DIFFYH1_SAFE(SIGMAYZ_Y_BG_PML)
                          : -DW_DIFFYH1_SAFE(SIGMAYZ_Y_BG));
}
#endif

#endif /* DW_ELASTIC_BORN_KERNELS_H */

#ifndef DW_ACOUSTIC_BORN_KERNELS_H
#define DW_ACOUSTIC_BORN_KERNELS_H

/*
 * acoustic_born_kernels.h
 *
 * Single-source per-grid-point stencil math for the variable-density
 * acoustic Born (linearised) propagator (see acoustic_born.c for the
 * physics/PML/boundary description). This header is included by BOTH
 * acoustic_born.c (compiled by gcc) and acoustic_born.cu (compiled by nvcc),
 * after each has defined the FUNC()/CAT() name-mangling macros and before
 * either uses the K/P/VZ/... family below.
 *
 * acoustic_born combines the two characteristics already handled separately
 * by acoustic_kernels.h (velocity-stress: independent per-axis velocity
 * updates plus one combined pressure update) and scalar_born_kernels.h
 * (a background field coupled to a scattered field, with a background-only,
 * scattered-only, and combined-adjoint variant). Crossing those two axes
 * gives twelve functions -- four per top-level entry point (three per-axis
 * velocity cells + one combined pressure cell), times three entry points:
 *
 *  * FUNC(forward_vz_cell)/vy_cell/vx_cell, FUNC(forward_p_cell): advance
 *    BOTH the background and scattered fields for one axis (velocity) or
 *    combined (pressure) in one call, matching acoustic_born.c's/.cu's
 *    single fused background-then-scattered loop body per axis (there is no
 *    per-axis split between background and scattered here, unlike the
 *    top-level forward/backward_sc/backward split below, because the
 *    scattered update at each point directly reads the just-computed
 *    background term at that SAME point -- see the doc comment in
 *    acoustic_born.c).
 *  * FUNC(backward_sc_vz_cell)/vy_cell/vx_cell, FUNC(backward_sc_p_cell):
 *    the scattered-only adjoint (fixed background K, B_d) -- the ORDINARY,
 *    unperturbed acoustic adjoint recursion (textually the same shape as
 *    acoustic_kernels.h's backward_*_cell family, applied to the scattered
 *    adjoint field) that also accumulates grad_dK/grad_dB_d by correlating
 *    the propagating scattered adjoint field with the checkpointed
 *    background terms. Used when only the scattering models (dK, dB_d)
 *    require a gradient, so the background adjoint field need not be
 *    propagated at all (the same optimisation scalar_born_kernels.h's
 *    FUNC(backward_cell_sc) makes for its one field).
 *  * FUNC(backward_vz_cell)/vy_cell/vx_cell, FUNC(backward_p_cell): the full
 *    two-field adjoint -- propagates the background adjoint field with the
 *    COUPLED adjoint (every K*P_adj becomes K*P_adj + dK*P_adj_sc, every
 *    B_d*V_adj_d becomes B_d*V_adj_d + dB_d*V_adj_sc_d) AND the scattered
 *    adjoint field with the ordinary adjoint (as backward_sc), in the same
 *    call, accumulating grad_K/grad_B_d (two correlation terms each) and
 *    grad_dK/grad_dB_d (one term each).
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest and PML-region triple loop in
 * acoustic_born.c, the CUDA kernel launch configuration and per-thread
 * grid-point assignment in acoustic_born.cu -- stays in acoustic_born.c/
 * acoustic_born.cu exactly as before (per the common_cpu.h/common_gpu.h
 * split); only the math performed once the caller has identified "this" grid
 * point lives here.
 *
 * Addressing convention (matches acoustic_kernels.h/scalar_born_kernels.h
 * exactly): p/vz/vy/vx/psc/vzsc/vysc/vxsc, phi_z/y/x and their _sc twins,
 * psi_z/y/x and their _sc twins, the *_grad_store_1 / *_grad_sc_store_1
 * stores, and grad_k/grad_bz/.../grad_dk/grad_dbz/... are addressed by `i`
 * (the CPU driver's per-shot-local flat index, which is also valid as the
 * CUDA driver's global shot-spanning index since both drivers pass
 * already-appropriately-offset pointers); k_shot, buoyancy_z/y/x_shot,
 * dk_shot, and dbuoyancy_z/y/x_shot are addressed by `j` (the per-shot-local
 * index) wherever a neighbour offset is needed (via the K/BUOYANCY_Z/Y/X/DK/
 * DBUOYANCY_Z/Y/X macros, used only by the VZ_ADJOINT/VZ_ADJOINT_BG/
 * P_ADJOINT_Z/P_ADJOINT_Z_BG families in the backward functions), because
 * the CUDA driver's k_shot/buoyancy_z/y/x_shot/dk_shot/dbuoyancy_z/y/x_shot
 * pointers are shot-local (offset by shot_idx*shot_numel, not i) while the
 * wavefield arrays span all shots. On the CPU driver, wavefields, k_shot,
 * buoyancy_z/y/x_shot, dk_shot, and dbuoyancy_z/y/x_shot are ALL already
 * offset to the current shot, so `i` and `j` are the same value there and
 * the CPU driver simply passes `i` for both. The forward velocity/pressure
 * cells and the background-model VALUE arguments of the backward cells need
 * no `j`: they only ever read the centre value of k/buoyancy/dk/dbuoyancy
 * (no neighbour offsets), so the caller passes that value directly, by
 * value, exactly as acoustic_kernels.h's forward cell/backward cell
 * functions do.
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

// Models (background, addressed by the shot-local index j -- see the
// addressing-convention note above)
#define K(dz, dy, dx) k_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// Models (scattering perturbations, addressed by j)
#define DK(dz, dy, dx) dk_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define DBUOYANCY_Z(dz, dy, dx) dbuoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define DBUOYANCY_Y(dz, dy, dx) dbuoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define DBUOYANCY_X(dz, dy, dx) dbuoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// Wavefields (background, addressed by i)
#define P(dz, dy, dx) p[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx[ND_INDEX(i, dz, dy, dx)]

// Wavefields (scattered, addressed by i)
#define PSC(dz, dy, dx) psc[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define VZSC(dz, dy, dx) vzsc[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VYSC(dz, dy, dx) vysc[ND_INDEX(i, dz, dy, dx)]
#endif
#define VXSC(dz, dy, dx) vxsc[ND_INDEX(i, dz, dy, dx)]

// PML memory variables (background, addressed by i)
#if DW_NDIM >= 3
#define PHI_Z(dz, dy, dx) phi_z[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PHI_Y(dz, dy, dx) phi_y[ND_INDEX(i, dz, dy, dx)]
#endif
#define PHI_X(dz, dy, dx) phi_x[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define PSI_Z(dz, dy, dx) psi_z[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PSI_Y(dz, dy, dx) psi_y[ND_INDEX(i, dz, dy, dx)]
#endif
#define PSI_X(dz, dy, dx) psi_x[ND_INDEX(i, dz, dy, dx)]

// PML memory variables (scattered, addressed by i)
#if DW_NDIM >= 3
#define PHI_Z_SC(dz, dy, dx) phi_z_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PHI_Y_SC(dz, dy, dx) phi_y_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#define PHI_X_SC(dz, dy, dx) phi_x_sc[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define PSI_Z_SC(dz, dy, dx) psi_z_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PSI_Y_SC(dz, dy, dx) psi_y_sc[ND_INDEX(i, dz, dy, dx)]
#endif
#define PSI_X_SC(dz, dy, dx) psi_x_sc[ND_INDEX(i, dz, dy, dx)]

// Scattered-only (ordinary, unperturbed acoustic) adjoint transpose macros --
// identical in form to acoustic_kernels.h's VZ_ADJOINT/P_ADJOINT_X etc.,
// operating on the scattered adjoint field with the (fixed) background
// models K, B_d.
#if DW_NDIM >= 3
#define VZ_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * PSC(dz, dy, dx))
#define VZ_ADJOINT_PML(dz, dy, dx)       \
  (bz[z + (dz)] * PHI_Z_SC(dz, dy, dx) - \
   dt * (1 + bz[z + (dz)]) * K(dz, dy, dx) * PSC(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VY_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * PSC(dz, dy, dx))
#define VY_ADJOINT_PML(dz, dy, dx)       \
  (by[y + (dy)] * PHI_Y_SC(dz, dy, dx) - \
   dt * (1 + by[y + (dy)]) * K(dz, dy, dx) * PSC(dz, dy, dx))
#endif
#define VX_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * PSC(dz, dy, dx))
#define VX_ADJOINT_PML(dz, dy, dx)       \
  (bx[x + (dx)] * PHI_X_SC(dz, dy, dx) - \
   dt * (1 + bx[x + (dx)]) * K(dz, dy, dx) * PSC(dz, dy, dx))

#if DW_NDIM >= 3
#define P_ADJOINT_Z(dz, dy, dx) \
  (-dt * BUOYANCY_Z(dz, dy, dx) * VZSC(dz, dy, dx))
#define P_ADJOINT_Z_PML(dz, dy, dx)       \
  (bzh[z + (dz)] * PSI_Z_SC(dz, dy, dx) - \
   dt * (1 + bzh[z + (dz)]) * BUOYANCY_Z(dz, dy, dx) * VZSC(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define P_ADJOINT_Y(dz, dy, dx) \
  (-dt * BUOYANCY_Y(dz, dy, dx) * VYSC(dz, dy, dx))
#define P_ADJOINT_Y_PML(dz, dy, dx)       \
  (byh[y + (dy)] * PSI_Y_SC(dz, dy, dx) - \
   dt * (1 + byh[y + (dy)]) * BUOYANCY_Y(dz, dy, dx) * VYSC(dz, dy, dx))
#endif
#define P_ADJOINT_X(dz, dy, dx) \
  (-dt * BUOYANCY_X(dz, dy, dx) * VXSC(dz, dy, dx))
#define P_ADJOINT_X_PML(dz, dy, dx)       \
  (bxh[x + (dx)] * PSI_X_SC(dz, dy, dx) - \
   dt * (1 + bxh[x + (dx)]) * BUOYANCY_X(dz, dy, dx) * VXSC(dz, dy, dx))

// Background-coupled adjoint transpose macros for the full two-field
// backward: the ordinary adjoint macros above, operating on the BACKGROUND
// adjoint field (P, VX), with the Born coupling term added (every K*P_adj
// becomes K*P_adj + dK*P_adj_sc, every B_d*V_adj_d becomes
// B_d*V_adj_d + dB_d*V_adj_sc_d). PSC/VXSC here are the SCATTERED adjoint
// field.
#if DW_NDIM >= 3
#define VZ_ADJOINT_BG(dz, dy, dx) \
  (-dt * (K(dz, dy, dx) * P(dz, dy, dx) + DK(dz, dy, dx) * PSC(dz, dy, dx)))
#define VZ_ADJOINT_BG_PML(dz, dy, dx) \
  (bz[z + (dz)] * PHI_Z(dz, dy, dx) - \
   dt * (1 + bz[z + (dz)]) *          \
       (K(dz, dy, dx) * P(dz, dy, dx) + DK(dz, dy, dx) * PSC(dz, dy, dx)))
#endif
#if DW_NDIM >= 2
#define VY_ADJOINT_BG(dz, dy, dx) \
  (-dt * (K(dz, dy, dx) * P(dz, dy, dx) + DK(dz, dy, dx) * PSC(dz, dy, dx)))
#define VY_ADJOINT_BG_PML(dz, dy, dx) \
  (by[y + (dy)] * PHI_Y(dz, dy, dx) - \
   dt * (1 + by[y + (dy)]) *          \
       (K(dz, dy, dx) * P(dz, dy, dx) + DK(dz, dy, dx) * PSC(dz, dy, dx)))
#endif
#define VX_ADJOINT_BG(dz, dy, dx) \
  (-dt * (K(dz, dy, dx) * P(dz, dy, dx) + DK(dz, dy, dx) * PSC(dz, dy, dx)))
#define VX_ADJOINT_BG_PML(dz, dy, dx) \
  (bx[x + (dx)] * PHI_X(dz, dy, dx) - \
   dt * (1 + bx[x + (dx)]) *          \
       (K(dz, dy, dx) * P(dz, dy, dx) + DK(dz, dy, dx) * PSC(dz, dy, dx)))

#if DW_NDIM >= 3
#define P_ADJOINT_Z_BG(dz, dy, dx)                  \
  (-dt * (BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
          DBUOYANCY_Z(dz, dy, dx) * VZSC(dz, dy, dx)))
#define P_ADJOINT_Z_BG_PML(dz, dy, dx)            \
  (bzh[z + (dz)] * PSI_Z(dz, dy, dx) -            \
   dt * (1 + bzh[z + (dz)]) *                     \
       (BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
        DBUOYANCY_Z(dz, dy, dx) * VZSC(dz, dy, dx)))
#endif
#if DW_NDIM >= 2
#define P_ADJOINT_Y_BG(dz, dy, dx)                  \
  (-dt * (BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
          DBUOYANCY_Y(dz, dy, dx) * VYSC(dz, dy, dx)))
#define P_ADJOINT_Y_BG_PML(dz, dy, dx)            \
  (byh[y + (dy)] * PSI_Y(dz, dy, dx) -            \
   dt * (1 + byh[y + (dy)]) *                     \
       (BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
        DBUOYANCY_Y(dz, dy, dx) * VYSC(dz, dy, dx)))
#endif
#define P_ADJOINT_X_BG(dz, dy, dx)                  \
  (-dt * (BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
          DBUOYANCY_X(dz, dy, dx) * VXSC(dz, dy, dx)))
#define P_ADJOINT_X_BG_PML(dz, dy, dx)            \
  (bxh[x + (dx)] * PSI_X(dz, dy, dx) -            \
   dt * (1 + bxh[x + (dx)]) *                     \
       (BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
        DBUOYANCY_X(dz, dy, dx) * VXSC(dz, dy, dx)))

/* ==================== Forward ==================== */

#if DW_NDIM >= 3
/* v = v - dt*B*(grad(p) + psi); dv = dv - dt*B*(grad(dp) + psi_sc) -
 * dt*dB*term_bg, z axis, background then scattered (coupled). `i` addresses
 * p/psc/vz/vzsc/psi_z/psi_z_sc/bz_grad_store_1/bz_grad_sc_store_1
 * (per-shot-local on the CPU driver, global on the CUDA driver);
 * `buoyancy_z_shot`/`dbuoyancy_z_shot` are the (already shot-selected)
 * buoyancy/perturbation VALUES at this point (read once by the caller --
 * this update never needs a neighbour offset of either); `pml_z` is true
 * when this point is inside the z-axis PML region.
 */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const psc,
    DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const vzsc,
    DW_DTYPE* __restrict const psi_z, DW_DTYPE* __restrict const psi_z_sc,
    DW_DTYPE* __restrict const bz_grad_store_1,
    DW_DTYPE* __restrict const bz_grad_sc_store_1,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const dbuoyancy_z_shot,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    bool const pml_z, DW_DTYPE const rdz) {
  DW_DTYPE term_z = DIFFZH1(P);
  if (pml_z) {
    psi_z[i] = bzh[z] * term_z + azh[z] * psi_z[i];
    term_z += psi_z[i];
  }
  if (store_bg) {
    bz_grad_store_1[i] = term_z;
  }
  vz[i] -= dt * buoyancy_z_shot * term_z;

  DW_DTYPE term_z_sc = DIFFZH1(PSC);
  if (pml_z) {
    psi_z_sc[i] = bzh[z] * term_z_sc + azh[z] * psi_z_sc[i];
    term_z_sc += psi_z_sc[i];
  }
  if (store_sc) {
    bz_grad_sc_store_1[i] = term_z_sc;
  }
  vzsc[i] -= dt * buoyancy_z_shot * term_z_sc + dt * dbuoyancy_z_shot * term_z;
}
#endif

#if DW_NDIM >= 2
/* Forward velocity update, y axis (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vy_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const psc,
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vysc,
    DW_DTYPE* __restrict const psi_y, DW_DTYPE* __restrict const psi_y_sc,
    DW_DTYPE* __restrict const by_grad_store_1,
    DW_DTYPE* __restrict const by_grad_sc_store_1,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE const dbuoyancy_y_shot,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const y, bool const pml_y, DW_DTYPE const rdy) {
  DW_DTYPE term_y = DIFFYH1(P);
  if (pml_y) {
    psi_y[i] = byh[y] * term_y + ayh[y] * psi_y[i];
    term_y += psi_y[i];
  }
  if (store_bg) {
    by_grad_store_1[i] = term_y;
  }
  vy[i] -= dt * buoyancy_y_shot * term_y;

  DW_DTYPE term_y_sc = DIFFYH1(PSC);
  if (pml_y) {
    psi_y_sc[i] = byh[y] * term_y_sc + ayh[y] * psi_y_sc[i];
    term_y_sc += psi_y_sc[i];
  }
  if (store_sc) {
    by_grad_sc_store_1[i] = term_y_sc;
  }
  vysc[i] -= dt * buoyancy_y_shot * term_y_sc + dt * dbuoyancy_y_shot * term_y;
}
#endif

/* Forward velocity update, x axis (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vx_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const psc,
    DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const vxsc,
    DW_DTYPE* __restrict const psi_x, DW_DTYPE* __restrict const psi_x_sc,
    DW_DTYPE* __restrict const bx_grad_store_1,
    DW_DTYPE* __restrict const bx_grad_sc_store_1,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE const dbuoyancy_x_shot,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, bool const store_bg, bool const store_sc,
    int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE term_x = DIFFXH1(P);
  if (pml_x) {
    psi_x[i] = bxh[x] * term_x + axh[x] * psi_x[i];
    term_x += psi_x[i];
  }
  if (store_bg) {
    bx_grad_store_1[i] = term_x;
  }
  vx[i] -= dt * buoyancy_x_shot * term_x;

  DW_DTYPE term_x_sc = DIFFXH1(PSC);
  if (pml_x) {
    psi_x_sc[i] = bxh[x] * term_x_sc + axh[x] * psi_x_sc[i];
    term_x_sc += psi_x_sc[i];
  }
  if (store_sc) {
    bx_grad_sc_store_1[i] = term_x_sc;
  }
  vxsc[i] -= dt * buoyancy_x_shot * term_x_sc + dt * dbuoyancy_x_shot * term_x;
}

/* p = p - dt*K*(div(v) + phi); dp = dp - dt*K*(div(dv) + phi_sc) - dt*dK*
 * div_bg, combining every present axis, background then scattered (coupled).
 * `i` addresses p/psc/vz/vzsc/.../phi_z/phi_z_sc/.../k_grad_store_1/
 * k_grad_sc_store_1 (per-shot-local on the CPU driver, global on the CUDA
 * driver); `k_shot`/`dk_shot` are the (already shot-selected) bulk modulus/
 * perturbation VALUES at this point (read once by the caller -- never needs
 * a neighbour offset here); `pml_z`/`pml_y`/`pml_x` are true when this point
 * is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_p_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE const* __restrict const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vysc,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vxsc,
    DW_DTYPE* __restrict const p, DW_DTYPE* __restrict const psc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z, DW_DTYPE* __restrict const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y, DW_DTYPE* __restrict const phi_y_sc,
#endif
    DW_DTYPE* __restrict const phi_x, DW_DTYPE* __restrict const phi_x_sc,
    DW_DTYPE* __restrict const k_grad_store_1,
    DW_DTYPE* __restrict const k_grad_sc_store_1, DW_DTYPE const k_shot,
    DW_DTYPE const dk_shot, DW_DTYPE const dt, bool const store_bg,
    bool const store_sc, int64_t const i,
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
  DW_DTYPE div_v = 0;
#if DW_NDIM >= 3
  DW_DTYPE d_z = DIFFZ1(VZ);
  if (pml_z) {
    phi_z[i] = az[z] * phi_z[i] + bz[z] * d_z;
    d_z += phi_z[i];
  }
  div_v += d_z;
#endif
#if DW_NDIM >= 2
  DW_DTYPE d_y = DIFFY1(VY);
  if (pml_y) {
    phi_y[i] = ay[y] * phi_y[i] + by[y] * d_y;
    d_y += phi_y[i];
  }
  div_v += d_y;
#endif
  DW_DTYPE d_x = DIFFX1(VX);
  if (pml_x) {
    phi_x[i] = ax[x] * phi_x[i] + bx[x] * d_x;
    d_x += phi_x[i];
  }
  div_v += d_x;

  if (store_bg) {
    k_grad_store_1[i] = div_v;
  }
  p[i] -= dt * k_shot * div_v;

  DW_DTYPE div_v_sc = 0;
#if DW_NDIM >= 3
  DW_DTYPE d_z_sc = DIFFZ1(VZSC);
  if (pml_z) {
    phi_z_sc[i] = az[z] * phi_z_sc[i] + bz[z] * d_z_sc;
    d_z_sc += phi_z_sc[i];
  }
  div_v_sc += d_z_sc;
#endif
#if DW_NDIM >= 2
  DW_DTYPE d_y_sc = DIFFY1(VYSC);
  if (pml_y) {
    phi_y_sc[i] = ay[y] * phi_y_sc[i] + by[y] * d_y_sc;
    d_y_sc += phi_y_sc[i];
  }
  div_v_sc += d_y_sc;
#endif
  DW_DTYPE d_x_sc = DIFFX1(VXSC);
  if (pml_x) {
    phi_x_sc[i] = ax[x] * phi_x_sc[i] + bx[x] * d_x_sc;
    d_x_sc += phi_x_sc[i];
  }
  div_v_sc += d_x_sc;

  if (store_sc) {
    k_grad_sc_store_1[i] = div_v_sc;
  }
  psc[i] -= dt * k_shot * div_v_sc + dt * dk_shot * div_v;
}

/* ==================== Backward (scattered-only) ==================== */

#if DW_NDIM >= 3
/* Scattered-only adjoint velocity update, z axis: the ORDINARY (unperturbed)
 * acoustic adjoint recursion (identical in form to acoustic_kernels.h's
 * FUNC(backward_vz_cell)) applied to the scattered adjoint field, plus
 * grad_dB_d accumulation (correlating the scattered adjoint field with the
 * BACKGROUND term store `bz_grad_store_1`, per the coupling derivation).
 * `i`/`j` addressing matches acoustic_kernels.h's backward_vz_cell.
 */
DW_HOST_DEVICE void FUNC(backward_sc_vz_cell)(
    DW_DTYPE const* __restrict const psc,
    DW_DTYPE const* __restrict const k_shot, DW_DTYPE* __restrict const vzsc,
    DW_DTYPE const* __restrict const phi_z_sc,
    DW_DTYPE const* __restrict const psi_z_sc,
    DW_DTYPE* __restrict const psi_zn_sc, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE* __restrict const grad_dbz,
    DW_DTYPE const* __restrict const bz_grad_store_1,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const dt, int64_t const step_ratio, bool const db_requires_grad,
    int64_t const i, int64_t const j, int64_t const ny, int64_t const nx,
    int64_t const z, bool const pml_z, DW_DTYPE const rdz) {
  vzsc[i] += (pml_z ? -DIFFZH1(VZ_ADJOINT_PML) : -DIFFZH1(VZ_ADJOINT));
  psi_zn_sc[i] = azh[z] * psi_z_sc[i] - dt * buoyancy_z_shot * azh[z] * vzsc[i];
  if (db_requires_grad) {
    grad_dbz[i] -= dt * vzsc[i] * bz_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
}
#endif

#if DW_NDIM >= 2
/* Scattered-only adjoint velocity update, y axis (see
 * FUNC(backward_sc_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_vy_cell)(
    DW_DTYPE const* __restrict const psc,
    DW_DTYPE const* __restrict const k_shot, DW_DTYPE* __restrict const vysc,
    DW_DTYPE const* __restrict const phi_y_sc,
    DW_DTYPE const* __restrict const psi_y_sc,
    DW_DTYPE* __restrict const psi_yn_sc, DW_DTYPE const buoyancy_y_shot,
    DW_DTYPE* __restrict const grad_dby,
    DW_DTYPE const* __restrict const by_grad_store_1,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const dt, int64_t const step_ratio, bool const db_requires_grad,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const y, bool const pml_y, DW_DTYPE const rdy) {
  vysc[i] += (pml_y ? -DIFFYH1(VY_ADJOINT_PML) : -DIFFYH1(VY_ADJOINT));
  psi_yn_sc[i] = ayh[y] * psi_y_sc[i] - dt * buoyancy_y_shot * ayh[y] * vysc[i];
  if (db_requires_grad) {
    grad_dby[i] -= dt * vysc[i] * by_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
}
#endif

/* Scattered-only adjoint velocity update, x axis (see
 * FUNC(backward_sc_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_sc_vx_cell)(
    DW_DTYPE const* __restrict const psc,
    DW_DTYPE const* __restrict const k_shot, DW_DTYPE* __restrict const vxsc,
    DW_DTYPE const* __restrict const phi_x_sc,
    DW_DTYPE const* __restrict const psi_x_sc,
    DW_DTYPE* __restrict const psi_xn_sc, DW_DTYPE const buoyancy_x_shot,
    DW_DTYPE* __restrict const grad_dbx,
    DW_DTYPE const* __restrict const bx_grad_store_1,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const dt, int64_t const step_ratio, bool const db_requires_grad,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  vxsc[i] += (pml_x ? -DIFFXH1(VX_ADJOINT_PML) : -DIFFXH1(VX_ADJOINT));
  psi_xn_sc[i] = axh[x] * psi_x_sc[i] - dt * buoyancy_x_shot * axh[x] * vxsc[i];
  if (db_requires_grad) {
    grad_dbx[i] -= dt * vxsc[i] * bx_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
}

/* Scattered-only adjoint pressure/phi update, combining every present axis
 * (see FUNC(forward_p_cell)): the ORDINARY acoustic adjoint (identical in
 * form to acoustic_kernels.h's FUNC(backward_p_cell)) applied to the
 * scattered adjoint field, plus grad_dK accumulation (correlating the
 * scattered adjoint field with the BACKGROUND term store `k_grad_store_1`).
 * `i`/`j` addressing matches acoustic_kernels.h's backward_p_cell.
 */
DW_HOST_DEVICE void FUNC(backward_sc_p_cell)(
    DW_DTYPE* __restrict const psc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vzsc, DW_DTYPE* __restrict const phi_z_sc,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vysc, DW_DTYPE* __restrict const phi_y_sc,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const psi_y_sc,
#endif
    DW_DTYPE const* __restrict const vxsc, DW_DTYPE* __restrict const phi_x_sc,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const psi_x_sc, DW_DTYPE const k_shot,
    DW_DTYPE* __restrict const grad_dk,
    DW_DTYPE const* __restrict const k_grad_store_1, DW_DTYPE const dt,
    int64_t const step_ratio, bool const dk_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdx) {
#if DW_NDIM >= 3
  if (pml_z) phi_z_sc[i] = az[z] * phi_z_sc[i] - dt * k_shot * az[z] * psc[i];
#endif
#if DW_NDIM >= 2
  if (pml_y) phi_y_sc[i] = ay[y] * phi_y_sc[i] - dt * k_shot * ay[y] * psc[i];
#endif
  if (pml_x) phi_x_sc[i] = ax[x] * phi_x_sc[i] - dt * k_shot * ax[x] * psc[i];

  if (dk_requires_grad) {
    grad_dk[i] -= dt * psc[i] * k_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }

  DW_DTYPE div_term = 0;
#if DW_NDIM >= 3
  div_term += pml_z ? -DIFFZ1(P_ADJOINT_Z_PML) : -DIFFZ1(P_ADJOINT_Z);
#endif
#if DW_NDIM >= 2
  div_term += pml_y ? -DIFFY1(P_ADJOINT_Y_PML) : -DIFFY1(P_ADJOINT_Y);
#endif
  div_term += pml_x ? -DIFFX1(P_ADJOINT_X_PML) : -DIFFX1(P_ADJOINT_X);

  psc[i] += div_term;
}

/* ==================== Backward (full two-field) ==================== */

#if DW_NDIM >= 3
/* Full two-field adjoint velocity update, z axis: the background adjoint
 * field with the COUPLED adjoint (VZ_ADJOINT_BG/_PML) AND the scattered
 * adjoint field with the ordinary adjoint (VZ_ADJOINT/_PML, as
 * FUNC(backward_sc_vz_cell)), in one call. Accumulates grad_B_d (two
 * correlation terms: background field with the background store, scattered
 * field with the scattered store) and grad_dB_d (one term, as
 * FUNC(backward_sc_vz_cell)). `i`/`j` addressing matches
 * acoustic_kernels.h's backward_vz_cell; `psi_z`/`psi_z_sc` are the OLD
 * (pre-this-step) buffers (read only), `psi_zn`/`psi_zn_sc` are the NEW
 * buffers (written).
 */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const psc,
    DW_DTYPE const* __restrict const k_shot,
    DW_DTYPE const* __restrict const dk_shot, DW_DTYPE* __restrict const vz,
    DW_DTYPE* __restrict const vzsc, DW_DTYPE const* __restrict const phi_z,
    DW_DTYPE const* __restrict const phi_z_sc,
    DW_DTYPE const* __restrict const psi_z, DW_DTYPE* __restrict const psi_zn,
    DW_DTYPE const* __restrict const psi_z_sc,
    DW_DTYPE* __restrict const psi_zn_sc, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const dbuoyancy_z_shot, DW_DTYPE* __restrict const grad_bz,
    DW_DTYPE* __restrict const grad_dbz,
    DW_DTYPE const* __restrict const bz_grad_store_1,
    DW_DTYPE const* __restrict const bz_grad_sc_store_1,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    bool const db_requires_grad, int64_t const i, int64_t const j,
    int64_t const ny, int64_t const nx, int64_t const z, bool const pml_z,
    DW_DTYPE const rdz) {
  vz[i] += (pml_z ? -DIFFZH1(VZ_ADJOINT_BG_PML) : -DIFFZH1(VZ_ADJOINT_BG));
  vzsc[i] += (pml_z ? -DIFFZH1(VZ_ADJOINT_PML) : -DIFFZH1(VZ_ADJOINT));
  psi_zn[i] =
      azh[z] * psi_z[i] -
      dt * azh[z] * (buoyancy_z_shot * vz[i] + dbuoyancy_z_shot * vzsc[i]);
  psi_zn_sc[i] = azh[z] * psi_z_sc[i] - dt * buoyancy_z_shot * azh[z] * vzsc[i];
  if (db_requires_grad) {
    grad_dbz[i] -= dt * vzsc[i] * bz_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (b_requires_grad) {
    grad_bz[i] -=
        dt * (vz[i] * bz_grad_store_1[i] + vzsc[i] * bz_grad_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
  }
}
#endif

#if DW_NDIM >= 2
/* Full two-field adjoint velocity update, y axis (see
 * FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vy_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const psc,
    DW_DTYPE const* __restrict const k_shot,
    DW_DTYPE const* __restrict const dk_shot, DW_DTYPE* __restrict const vy,
    DW_DTYPE* __restrict const vysc, DW_DTYPE const* __restrict const phi_y,
    DW_DTYPE const* __restrict const phi_y_sc,
    DW_DTYPE const* __restrict const psi_y, DW_DTYPE* __restrict const psi_yn,
    DW_DTYPE const* __restrict const psi_y_sc,
    DW_DTYPE* __restrict const psi_yn_sc, DW_DTYPE const buoyancy_y_shot,
    DW_DTYPE const dbuoyancy_y_shot, DW_DTYPE* __restrict const grad_by,
    DW_DTYPE* __restrict const grad_dby,
    DW_DTYPE const* __restrict const by_grad_store_1,
    DW_DTYPE const* __restrict const by_grad_sc_store_1,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    bool const db_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const y, bool const pml_y, DW_DTYPE const rdy) {
  vy[i] += (pml_y ? -DIFFYH1(VY_ADJOINT_BG_PML) : -DIFFYH1(VY_ADJOINT_BG));
  vysc[i] += (pml_y ? -DIFFYH1(VY_ADJOINT_PML) : -DIFFYH1(VY_ADJOINT));
  psi_yn[i] =
      ayh[y] * psi_y[i] -
      dt * ayh[y] * (buoyancy_y_shot * vy[i] + dbuoyancy_y_shot * vysc[i]);
  psi_yn_sc[i] = ayh[y] * psi_y_sc[i] - dt * buoyancy_y_shot * ayh[y] * vysc[i];
  if (db_requires_grad) {
    grad_dby[i] -= dt * vysc[i] * by_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (b_requires_grad) {
    grad_by[i] -=
        dt * (vy[i] * by_grad_store_1[i] + vysc[i] * by_grad_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
  }
}
#endif

/* Full two-field adjoint velocity update, x axis (see
 * FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const psc,
    DW_DTYPE const* __restrict const k_shot,
    DW_DTYPE const* __restrict const dk_shot, DW_DTYPE* __restrict const vx,
    DW_DTYPE* __restrict const vxsc, DW_DTYPE const* __restrict const phi_x,
    DW_DTYPE const* __restrict const phi_x_sc,
    DW_DTYPE const* __restrict const psi_x, DW_DTYPE* __restrict const psi_xn,
    DW_DTYPE const* __restrict const psi_x_sc,
    DW_DTYPE* __restrict const psi_xn_sc, DW_DTYPE const buoyancy_x_shot,
    DW_DTYPE const dbuoyancy_x_shot, DW_DTYPE* __restrict const grad_bx,
    DW_DTYPE* __restrict const grad_dbx,
    DW_DTYPE const* __restrict const bx_grad_store_1,
    DW_DTYPE const* __restrict const bx_grad_sc_store_1,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    bool const db_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  vx[i] += (pml_x ? -DIFFXH1(VX_ADJOINT_BG_PML) : -DIFFXH1(VX_ADJOINT_BG));
  vxsc[i] += (pml_x ? -DIFFXH1(VX_ADJOINT_PML) : -DIFFXH1(VX_ADJOINT));
  psi_xn[i] =
      axh[x] * psi_x[i] -
      dt * axh[x] * (buoyancy_x_shot * vx[i] + dbuoyancy_x_shot * vxsc[i]);
  psi_xn_sc[i] = axh[x] * psi_x_sc[i] - dt * buoyancy_x_shot * axh[x] * vxsc[i];
  if (db_requires_grad) {
    grad_dbx[i] -= dt * vxsc[i] * bx_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (b_requires_grad) {
    grad_bx[i] -=
        dt * (vx[i] * bx_grad_store_1[i] + vxsc[i] * bx_grad_sc_store_1[i]) *
        (DW_DTYPE)step_ratio;
  }
}

/* Full two-field adjoint pressure/phi update, combining every present axis
 * (see FUNC(forward_p_cell)): the background adjoint field with the COUPLED
 * adjoint (the P_ADJOINT_Z_BG/P_ADJOINT_Z_BG_PML family) AND the scattered
 * adjoint field with the ordinary adjoint (the P_ADJOINT_Z/P_ADJOINT_Z_PML
 * family, as FUNC(backward_sc_p_cell)), in one call. Accumulates grad_K (two
 * correlation terms) and grad_dK (one term, as FUNC(backward_sc_p_cell)).
 * `i`/`j` addressing matches acoustic_kernels.h's backward_p_cell;
 * `psi_z`/`psi_z_sc` etc. are the OLD (pre-this-step) velocity-PML buffers,
 * read by the P_ADJOINT_Z_BG_PML/P_ADJOINT_Z_PML macros (not the buffers
 * FUNC(backward_vz_cell)/vy_cell/vx_cell just wrote -- matching the
 * transpose structure exactly, as in acoustic_kernels.h).
 */
DW_HOST_DEVICE void FUNC(backward_p_cell)(
    DW_DTYPE* __restrict const p, DW_DTYPE* __restrict const psc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE const* __restrict const vzsc,
    DW_DTYPE* __restrict const phi_z, DW_DTYPE* __restrict const phi_z_sc,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const dbuoyancy_z_shot,
    DW_DTYPE const* __restrict const psi_z,
    DW_DTYPE const* __restrict const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vysc,
    DW_DTYPE* __restrict const phi_y, DW_DTYPE* __restrict const phi_y_sc,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const dbuoyancy_y_shot,
    DW_DTYPE const* __restrict const psi_y,
    DW_DTYPE const* __restrict const psi_y_sc,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vxsc,
    DW_DTYPE* __restrict const phi_x, DW_DTYPE* __restrict const phi_x_sc,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const dbuoyancy_x_shot,
    DW_DTYPE const* __restrict const psi_x,
    DW_DTYPE const* __restrict const psi_x_sc, DW_DTYPE const k_shot,
    DW_DTYPE const dk_shot, DW_DTYPE* __restrict const grad_k,
    DW_DTYPE* __restrict const grad_dk,
    DW_DTYPE const* __restrict const k_grad_store_1,
    DW_DTYPE const* __restrict const k_grad_sc_store_1, DW_DTYPE const dt,
    int64_t const step_ratio, bool const k_requires_grad,
    bool const dk_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdx) {
#if DW_NDIM >= 3
  if (pml_z) {
    phi_z[i] =
        az[z] * phi_z[i] - dt * az[z] * (k_shot * p[i] + dk_shot * psc[i]);
    phi_z_sc[i] = az[z] * phi_z_sc[i] - dt * k_shot * az[z] * psc[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    phi_y[i] =
        ay[y] * phi_y[i] - dt * ay[y] * (k_shot * p[i] + dk_shot * psc[i]);
    phi_y_sc[i] = ay[y] * phi_y_sc[i] - dt * k_shot * ay[y] * psc[i];
  }
#endif
  if (pml_x) {
    phi_x[i] =
        ax[x] * phi_x[i] - dt * ax[x] * (k_shot * p[i] + dk_shot * psc[i]);
    phi_x_sc[i] = ax[x] * phi_x_sc[i] - dt * k_shot * ax[x] * psc[i];
  }

  if (dk_requires_grad) {
    grad_dk[i] -= dt * psc[i] * k_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (k_requires_grad) {
    grad_k[i] -= dt *
                 (p[i] * k_grad_store_1[i] + psc[i] * k_grad_sc_store_1[i]) *
                 (DW_DTYPE)step_ratio;
  }

  DW_DTYPE div_term = 0;
#if DW_NDIM >= 3
  div_term += pml_z ? -DIFFZ1(P_ADJOINT_Z_BG_PML) : -DIFFZ1(P_ADJOINT_Z_BG);
#endif
#if DW_NDIM >= 2
  div_term += pml_y ? -DIFFY1(P_ADJOINT_Y_BG_PML) : -DIFFY1(P_ADJOINT_Y_BG);
#endif
  div_term += pml_x ? -DIFFX1(P_ADJOINT_X_BG_PML) : -DIFFX1(P_ADJOINT_X_BG);
  p[i] += div_term;

  DW_DTYPE div_term_sc = 0;
#if DW_NDIM >= 3
  div_term_sc += pml_z ? -DIFFZ1(P_ADJOINT_Z_PML) : -DIFFZ1(P_ADJOINT_Z);
#endif
#if DW_NDIM >= 2
  div_term_sc += pml_y ? -DIFFY1(P_ADJOINT_Y_PML) : -DIFFY1(P_ADJOINT_Y);
#endif
  div_term_sc += pml_x ? -DIFFX1(P_ADJOINT_X_PML) : -DIFFX1(P_ADJOINT_X);
  psc[i] += div_term_sc;
}

/* ---- Backward: periodic-face adjoint guard maintenance ---- */
/*
 * Same fold/gather scheme as acoustic_kernels.h (see the comment there),
 * applied per adjoint field set. The gather functions take a single
 * (model, field) payload; the Born-coupled background gathers (payload
 * K*P_adj + dK*P_adj_sc, or B*V_adj + dB*V_adj_sc) are composed by the
 * drivers as two single-payload calls into the same target (the gather is
 * linear in its payload).
 */

/* FUNC(adj_fold_p_line)/FUNC(adj_fold_v_line)/FUNC(adj_gather_p_guard_line)
 * and their _fr counterparts live in adj_line_bc.h (see that header and
 * acoustic_kernels.h's copy of this same comment). FUNC(adj_gather_v_guard_
 * line)[_fr] (below) stays local: viscoacoustic's version couples in the
 * SLS attenuation terms. */
#include "adj_line_bc.h"

/* Adjoint-velocity guard gather for one periodic-axis line (see
 * acoustic_kernels.h's function of the same name). */
DW_HOST_DEVICE void FUNC(adj_gather_v_guard_line)(
    DW_DTYPE* __restrict const v, DW_DTYPE const* __restrict const p,
    DW_DTYPE const* __restrict const k_shot, int64_t const base,
    int64_t const jbase, int64_t const nax, int64_t const stride,
    DW_DTYPE const dt, DW_DTYPE const rd, DW_DTYPE const* __restrict const sw) {
  int64_t const P = FD_PAD;
  int64_t c, q;
  for (c = 0; c < 2 * P; ++c) {
    int64_t const y = (c < P) ? c : nax - 2 * P + c;
    DW_DTYPE acc = 0;
    for (q = 0; q < P; ++q) {
      int64_t const th = y + 1 + q;
      int64_t const tl = y - q;
      if (th < nax)
        acc += sw[q] * k_shot[jbase + th * stride] * p[base + th * stride];
      if (tl >= 0)
        acc -= sw[q] * k_shot[jbase + tl * stride] * p[base + tl * stride];
    }
    v[base + y * stride] += dt * rd * acc;
  }
}

/* ---- Backward: free/rigid-face adjoint guard maintenance ---- */
/*
 * Same exact-transpose fold/gather scheme as acoustic_kernels.h's _fr
 * functions (see the comment there), applied per adjoint field set: the
 * re-fill realises the free/rigid transpose only for the (free-low,
 * rigid-high) pairing; other combinations fold with the mirror/sign-flip
 * transposes of the fills, per face. The Born-coupled background gathers
 * are composed by the drivers as two single-payload calls, exactly like
 * the periodic gathers above.
 */

/* Free/rigid adjoint-velocity guard gather (per-face gated; the extra high
 * interior node n-P needs no gather -- see acoustic_kernels.h). */
DW_HOST_DEVICE void FUNC(adj_gather_v_guard_line_fr)(
    DW_DTYPE* __restrict const v, DW_DTYPE const* __restrict const p,
    DW_DTYPE const* __restrict const k_shot, int64_t const base,
    int64_t const jbase, int64_t const nax, int64_t const stride,
    DW_DTYPE const dt, DW_DTYPE const rd, DW_DTYPE const* __restrict const sw,
    int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t c, q;
  for (c = 0; c < 2 * P - 1; ++c) {
    int64_t y;
    if (c < P) {
      if (bc_lo == 0) continue;
      y = c;
    } else {
      if (bc_hi == 0) continue;
      y = nax - 2 * P + 1 + c;
    }
    {
      DW_DTYPE acc = 0;
      for (q = 0; q < P; ++q) {
        int64_t const th = y + 1 + q;
        int64_t const tl = y - q;
        if (th < nax)
          acc += sw[q] * k_shot[jbase + th * stride] * p[base + th * stride];
        if (tl >= 0)
          acc -= sw[q] * k_shot[jbase + tl * stride] * p[base + tl * stride];
      }
      v[base + y * stride] += dt * rd * acc;
    }
  }
}

#endif /* DW_ACOUSTIC_BORN_KERNELS_H */

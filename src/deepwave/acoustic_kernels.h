#ifndef DW_ACOUSTIC_KERNELS_H
#define DW_ACOUSTIC_KERNELS_H

/*
 * acoustic_kernels.h
 *
 * Single-source per-grid-point stencil math for the acoustic (variable
 * density, staggered grid, velocity-stress) wave equation propagator (see
 * acoustic.c for the physics/PML/boundary description). This header is
 * included by BOTH acoustic.c (compiled by gcc) and acoustic.cu (compiled by
 * nvcc), after each has defined the FUNC()/CAT() name-mangling macros and
 * before either uses the K/P/VZ/... family below. It contains:
 *
 *  * The per-axis addressing/adjoint-term macros (ND_INDEX, K, BUOYANCY_Z/Y/X,
 *    P, VZ/VY/VX, PHI_Z/Y/X, PSI_Z/Y/X, and the VZ_ADJOINT/VY_ADJOINT/
 *    VX_ADJOINT/P_ADJOINT_Z/Y/X family (+ their _PML twins)), lifted verbatim
 *    from acoustic.cu's copy of these macros -- acoustic.cu's naming/
 *    addressing convention (wavefields addressed by `i`, model arrays
 *    addressed by `j`; see below) is adopted as the one shared convention,
 *    since it is already exactly the convention this header's functions need
 *    for both drivers. acoustic.c's copy of the same macros was textually
 *    identical EXCEPT that it used `i` (not `j`) to address the model arrays
 *    K/BUOYANCY_Z/Y/X -- a difference that disappears once the CPU driver
 *    passes `i` for `j` too at every call site below (it already is the
 *    shot-local index there, exactly as `j` is on the CUDA driver -- see the
 *    addressing-convention note).
 *  * FUNC(forward_vz_cell) / FUNC(forward_vy_cell) / FUNC(forward_vx_cell):
 *    the per-grid-point half-grid velocity update for each axis (unlike
 *    scalar's single combined pressure-only update, acoustic's staggered
 *    grid keeps each velocity component's PML/update logic independent per
 *    axis -- acoustic.c already runs these as three separate OpenMP loops
 *    with independent bounds per axis, and acoustic.cu's forward_kernel_v
 *    already computes them as three independent, separately-gated blocks
 *    within one kernel -- so three per-axis functions is the natural
 *    single-source mapping of the existing structure, not a new design).
 *    FUNC(forward_p_cell): the combined pressure update (div(v) accumulated
 *    across all present axes, like scalar's forward_cell combines its axes),
 *    matching acoustic.c's/acoustic.cu's single combined pressure loop body.
 *  * FUNC(backward_vz_cell) / FUNC(backward_vy_cell) / FUNC(backward_vx_cell)
 *    / FUNC(backward_p_cell): the adjoint counterparts, one per velocity axis
 *    plus one combined pressure/phi adjoint update, mirroring the forward
 *    functions' structure (acoustic.c's/acoustic.cu's backward loops already
 *    have the same per-axis-velocity / combined-pressure shape as forward).
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest and PML-region triple loop in
 * acoustic.c, the CUDA kernel launch configuration and per-thread grid-point
 * assignment in acoustic.cu -- stays in acoustic.c/acoustic.cu exactly as
 * before (per the common_cpu.h/common_gpu.h split); only the math performed
 * once the caller has identified "this" grid point lives here. The boundary-
 * condition guard-fill helpers (fill_p_bc/fill_v_bc and their CUDA
 * equivalents) are NOT per-grid-point stencil math (they are line-oriented
 * boundary maintenance already implemented as two independent parallelisation
 * strategies, not textually-identical duplicated code) and are out of scope
 * for this refactor; they remain in acoustic.c/acoustic.cu unchanged.
 *
 * Addressing convention (matches scalar_kernels.h/scalar_born_kernels.h
 * exactly): p/vz/vy/vx/phi_z/.../psi_z/.../ *_grad_store_1/grad_bz/.../grad_k
 * are addressed by `i` (the CPU driver's per-shot-local flat index, which is
 * also valid as the CUDA driver's global shot-spanning index since both
 * drivers pass already-appropriately-offset pointers); k_shot/buoyancy_*_shot
 * are addressed by `j` (the per-shot-local index) wherever a neighbour offset
 * is needed (via the K/BUOYANCY_Z/Y/X macros, used only by the *_ADJOINT/
 * P_ADJOINT_* families in the backward functions), because the CUDA driver's
 * k_shot/buoyancy_*_shot pointers are shot-local (offset by
 * shot_idx*shot_numel, not i) while the wavefield arrays span all shots. On
 * the CPU driver, wavefields/k_shot/buoyancy_*_shot are ALL already offset to
 * the current shot, so `i` and `j` are the same value there and the CPU
 * driver simply passes `i` for both. The forward velocity/pressure cells need
 * no `j`: they only ever read the *centre* value of k/buoyancy (no neighbour
 * offsets), so the caller passes that value directly, by value, exactly as
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

// Models (addressed by the shot-local index j -- see the addressing-
// convention note above)
#define K(dz, dy, dx) k_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// Wavefields (addressed by i -- see the addressing-convention note above)
#define P(dz, dy, dx) p[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx[ND_INDEX(i, dz, dy, dx)]

// PML memory variables (Phi for pressure, Psi for velocity)
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

#if DW_NDIM >= 3
#define VZ_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VZ_ADJOINT_PML(dz, dy, dx)    \
  (bz[z + (dz)] * PHI_Z(dz, dy, dx) - \
   dt * (1 + bz[z + (dz)]) * K(dz, dy, dx) * P(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VY_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VY_ADJOINT_PML(dz, dy, dx)    \
  (by[y + (dy)] * PHI_Y(dz, dy, dx) - \
   dt * (1 + by[y + (dy)]) * K(dz, dy, dx) * P(dz, dy, dx))
#endif
#define VX_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VX_ADJOINT_PML(dz, dy, dx)    \
  (bx[x + (dx)] * PHI_X(dz, dy, dx) - \
   dt * (1 + bx[x + (dx)]) * K(dz, dy, dx) * P(dz, dy, dx))

#if DW_NDIM >= 3
#define P_ADJOINT_Z(dz, dy, dx) (-dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#define P_ADJOINT_Z_PML(dz, dy, dx)    \
  (bzh[z + (dz)] * PSI_Z(dz, dy, dx) - \
   dt * (1 + bzh[z + (dz)]) * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define P_ADJOINT_Y(dz, dy, dx) (-dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define P_ADJOINT_Y_PML(dz, dy, dx)    \
  (byh[y + (dy)] * PSI_Y(dz, dy, dx) - \
   dt * (1 + byh[y + (dy)]) * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define P_ADJOINT_X(dz, dy, dx) (-dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define P_ADJOINT_X_PML(dz, dy, dx)    \
  (bxh[x + (dx)] * PSI_X(dz, dy, dx) - \
   dt * (1 + bxh[x + (dx)]) * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

/* ---- Forward: per-axis half-grid velocity updates ---- */

#if DW_NDIM >= 3
/* v = v - dt * B * (grad(p) + psi), z axis. `i` addresses p/vz/psi_z/
 * bz_grad_store_1 (per-shot-local on the CPU driver, global on the CUDA
 * driver); `buoyancy_z_shot` is the (already shot-selected) buoyancy VALUE
 * at this point (read once by the caller -- this update never needs a
 * neighbour offset of it); `pml_z` is true when this point is inside the
 * z-axis PML region (false in the interior).
 */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE* __restrict const vz,
    DW_DTYPE* __restrict const psi_z,
    DW_DTYPE* __restrict const bz_grad_store_1, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const dt, bool const b_requires_grad, int64_t const i,
    int64_t const ny, int64_t const nx, int64_t const z, bool const pml_z,
    DW_DTYPE const rdz) {
  DW_DTYPE term_z = DIFFZH1(P);
  if (pml_z) {
    psi_z[i] = bzh[z] * term_z + azh[z] * psi_z[i];
    term_z += psi_z[i];
  }
  if (b_requires_grad) {
    bz_grad_store_1[i] = term_z;
  }
  vz[i] -= dt * buoyancy_z_shot * term_z;
}
#endif

#if DW_NDIM >= 2
/* v = v - dt * B * (grad(p) + psi), y axis (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vy_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE* __restrict const vy,
    DW_DTYPE* __restrict const psi_y,
    DW_DTYPE* __restrict const by_grad_store_1, DW_DTYPE const buoyancy_y_shot,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const dt, bool const b_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const y, bool const pml_y, DW_DTYPE const rdy) {
  DW_DTYPE term_y = DIFFYH1(P);
  if (pml_y) {
    psi_y[i] = byh[y] * term_y + ayh[y] * psi_y[i];
    term_y += psi_y[i];
  }
  if (b_requires_grad) {
    by_grad_store_1[i] = term_y;
  }
  vy[i] -= dt * buoyancy_y_shot * term_y;
}
#endif

/* v = v - dt * B * (grad(p) + psi), x axis (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vx_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE* __restrict const vx,
    DW_DTYPE* __restrict const psi_x,
    DW_DTYPE* __restrict const bx_grad_store_1, DW_DTYPE const buoyancy_x_shot,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, bool const b_requires_grad, int64_t const i,
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
  if (b_requires_grad) {
    bx_grad_store_1[i] = term_x;
  }
  vx[i] -= dt * buoyancy_x_shot * term_x;
}

/* ---- Forward: combined pressure update ---- */

/* p = p - dt * K * (div(v) + phi), combining every present axis (like
 * scalar_kernels.h's forward_cell combines its axes into one w_sum). `i`
 * addresses p/vz/vy/vx/phi_z/.../k_grad_store_1 (per-shot-local on the CPU
 * driver, global on the CUDA driver); `k_shot` is the (already
 * shot-selected) bulk modulus VALUE at this point (read once by the caller
 * -- never needs a neighbour offset here); `pml_z`/`pml_y`/`pml_x` are true
 * when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_p_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y,
#endif
    DW_DTYPE* __restrict const phi_x, DW_DTYPE* __restrict const k_grad_store_1,
    DW_DTYPE const k_shot, DW_DTYPE const dt, bool const k_requires_grad,
    int64_t const i,
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

  if (k_requires_grad) {
    k_grad_store_1[i] = div_v;
  }
  p[i] -= dt * k_shot * div_v;
}

/* ---- Backward: per-axis half-grid adjoint velocity updates ---- */

#if DW_NDIM >= 3
/* Adjoint of FUNC(forward_vz_cell)'s z-axis velocity update: the transpose of
 * -dt*B*DIFFZH1(P), which is +DIFFZ1 applied to -dt*K*Pbar (the VZ_ADJOINT
 * macro), plus the transposed CPML recursion. Verified numerically, as the
 * whole-operator transpose, by tests/test_adjoint_dot_product.py; derived in
 * acoustic.c's header comment (ADJOINT) and docs/acoustic.rst.
 *
 * v = v + dt*(grad((1+b)Kp) - grad(b*phi)); psi = a*psi - dt*B*a*v, z axis.
 * `i` addresses p/vz/phi_z/psi_z/psi_zn/bz_grad_store_1/grad_bz
 * (per-shot-local on the CPU driver, global on the CUDA driver); `j`
 * addresses the already-shot-sliced k_shot (equal to `i` on the CPU driver;
 * the local per-shot grid index on the CUDA driver, distinct from the global
 * `i`) -- see the addressing-convention note at the top of this file.
 * `psi_z` is the OLD (pre-this-step) buffer (read only); `psi_zn` is the
 * NEW buffer (written); `pml_z` is true inside the z-axis PML region.
 */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const k_shot,
    DW_DTYPE* __restrict const vz, DW_DTYPE const* __restrict const phi_z,
    DW_DTYPE const* __restrict const psi_z, DW_DTYPE* __restrict const psi_zn,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE* __restrict const grad_bz,
    DW_DTYPE const* __restrict const bz_grad_store_1,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    int64_t const i, int64_t const j, int64_t const ny, int64_t const nx,
    int64_t const z, bool const pml_z, DW_DTYPE const rdz) {
  vz[i] += (pml_z ? -DIFFZH1(VZ_ADJOINT_PML) : -DIFFZH1(VZ_ADJOINT));
  psi_zn[i] = azh[z] * psi_z[i] - dt * buoyancy_z_shot * azh[z] * vz[i];
  if (b_requires_grad) {
    grad_bz[i] -= dt * vz[i] * bz_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
}
#endif

#if DW_NDIM >= 2
/* Adjoint of FUNC(forward_vy_cell); y axis (see FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vy_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const k_shot,
    DW_DTYPE* __restrict const vy, DW_DTYPE const* __restrict const phi_y,
    DW_DTYPE const* __restrict const psi_y, DW_DTYPE* __restrict const psi_yn,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE* __restrict const grad_by,
    DW_DTYPE const* __restrict const by_grad_store_1,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const y, bool const pml_y, DW_DTYPE const rdy) {
  vy[i] += (pml_y ? -DIFFYH1(VY_ADJOINT_PML) : -DIFFYH1(VY_ADJOINT));
  psi_yn[i] = ayh[y] * psi_y[i] - dt * buoyancy_y_shot * ayh[y] * vy[i];
  if (b_requires_grad) {
    grad_by[i] -= dt * vy[i] * by_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
}
#endif

/* Adjoint of FUNC(forward_vx_cell); x axis (see FUNC(backward_vz_cell)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell)(
    DW_DTYPE const* __restrict const p, DW_DTYPE const* __restrict const k_shot,
    DW_DTYPE* __restrict const vx, DW_DTYPE const* __restrict const phi_x,
    DW_DTYPE const* __restrict const psi_x, DW_DTYPE* __restrict const psi_xn,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE* __restrict const grad_bx,
    DW_DTYPE const* __restrict const bx_grad_store_1,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  vx[i] += (pml_x ? -DIFFXH1(VX_ADJOINT_PML) : -DIFFXH1(VX_ADJOINT));
  psi_xn[i] = axh[x] * psi_x[i] - dt * buoyancy_x_shot * axh[x] * vx[i];
  if (b_requires_grad) {
    grad_bx[i] -= dt * vx[i] * bx_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }
}

/* ---- Backward: combined pressure/phi adjoint update ---- */

/* Adjoint of FUNC(forward_p_cell): the transpose of -dt*K*div_h(v) is
 * +grad_h applied to -dt*B*Vbar per axis (the P_ADJOINT_Z/Y/X macros), plus
 * the transposed CPML recursion and the grad_K accumulation
 * -dt*Pbar*div_h(v)*step_ratio against the divergence FUNC(forward_p_cell)
 * stored. Verified numerically by tests/test_adjoint_dot_product.py; derived
 * in acoustic.c's header comment (ADJOINT) and docs/acoustic.rst.
 *
 * phi = a*phi - dt*K*a*p; p = p + dt*(div(B(1+b)v) - div(b*psi)), combining
 * every present axis (see FUNC(forward_p_cell)). `i`/`j` addressing matches
 * FUNC(backward_vz_cell) above. `psi_z`/`psi_y`/`psi_x` are the OLD
 * (pre-this-step) velocity-PML buffers -- P_ADJOINT_Z_PML/Y_PML/X_PML read
 * the same buffer FUNC(backward_vz_cell)/vy_cell/vx_cell read (not the
 * buffer they just wrote), matching the transpose structure exactly.
 */
DW_HOST_DEVICE void FUNC(backward_p_cell)(
    DW_DTYPE* __restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const phi_z,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const psi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const phi_y,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const psi_y,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const phi_x,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const psi_x, DW_DTYPE const k_shot,
    DW_DTYPE* __restrict const grad_k,
    DW_DTYPE const* __restrict const k_grad_store_1, DW_DTYPE const dt,
    int64_t const step_ratio, bool const k_requires_grad, int64_t const i,
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
  if (pml_z) phi_z[i] = az[z] * phi_z[i] - dt * k_shot * az[z] * p[i];
#endif
#if DW_NDIM >= 2
  if (pml_y) phi_y[i] = ay[y] * phi_y[i] - dt * k_shot * ay[y] * p[i];
#endif
  if (pml_x) phi_x[i] = ax[x] * phi_x[i] - dt * k_shot * ax[x] * p[i];

  if (k_requires_grad) {
    grad_k[i] -= dt * p[i] * k_grad_store_1[i] * (DW_DTYPE)step_ratio;
  }

  DW_DTYPE div_term = 0;
#if DW_NDIM >= 3
  div_term += pml_z ? -DIFFZ1(P_ADJOINT_Z_PML) : -DIFFZ1(P_ADJOINT_Z);
#endif
#if DW_NDIM >= 2
  div_term += pml_y ? -DIFFY1(P_ADJOINT_Y_PML) : -DIFFY1(P_ADJOINT_Y);
#endif
  div_term += pml_x ? -DIFFX1(P_ADJOINT_X_PML) : -DIFFX1(P_ADJOINT_X);

  p[i] += div_term;
}

/* ---- Backward: periodic-face adjoint guard maintenance ---- */
/*
 * On a periodic axis the forward guard fill copies interior wrap images into
 * the guard cells (and slaves the extra high interior velocity node to node
 * FD_PAD), so its exact transpose is a FOLD: each guard cell's adjoint value
 * is added to the interior node the forward copied it from, then the guard
 * cell is zeroed. Unlike the free/rigid faces (where re-applying the forward
 * fill to the adjoint field realises the transpose through the mirror-padded
 * models), the fold writes into interior seam nodes, so the backward drivers
 * must (a) apply the pressure fold before the source-gradient read, (b)
 * accumulate the buoyancy gradient only after the velocity fold (the fused
 * accumulation in FUNC(backward_v*_cell) is suppressed on periodic axes),
 * and (c) first accumulate, via the *_guard_line gather functions below, the
 * adjoint-stencil contributions that fall on the guard cells themselves
 * (out-of-range taps are zero: the adjoint fields are zero outside their
 * interior nodes once folded). The line functions below are shared by the
 * CPU (acoustic.c) and CUDA (acoustic.cu) drivers, which supply their own
 * line enumeration, exactly like the *_cell functions above.
 */

/* FUNC(adj_fold_p_line)/FUNC(adj_fold_v_line)/FUNC(adj_gather_p_guard_line)
 * and their _fr (free/rigid) counterparts below live in adj_line_bc.h,
 * shared verbatim across every propagator on this staggered layout -- see
 * that header for the full explanation. FUNC(adj_gather_v_guard_line)[_fr]
 * (immediately below) stays local: viscoacoustic's version couples in the
 * SLS attenuation terms. */
#include "adj_line_bc.h"

/* Adjoint-velocity guard gather for one periodic-axis line: the guard cells
 * (and the slaved extra high interior node) receive the same
 * -DIFF*H1(-dt*K*P) contribution the interior receives in
 * FUNC(backward_v*_cell), with out-of-range taps dropped (the adjoint
 * pressure is zero there). No PML/psi/grad terms: a periodic axis has no
 * PML, and the buoyancy gradient is accumulated after the fold. `sw` is the
 * staggered stencil weight array (DW_STAG_TAYLOR_W). `base` addresses the
 * wavefields, `jbase` the (possibly shot-local) model array. */
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
 * Re-applying the forward guard fill to the adjoint fields (with
 * mirror-padded models) realises the transpose of the free/rigid fills ONLY
 * for the per-axis (free-low, rigid-high) pairing; every other free/rigid
 * combination needs the same exact fold/gather treatment as the periodic
 * faces above, with the mirror/sign-flip transposes of fill_p_bc_line /
 * fill_v_bc_line instead of the periodic wrap fold. The fold targets are the
 * forward fill's source indices with the forward's signs, applied in the
 * reverse of the forward's op order (high guards, then low guards, then the
 * transposes of the forward's leading on-surface/on-wall interior-node
 * writes). The gathers cover only the faces that are free/rigid; unlike the
 * periodic case the velocity's extra high interior node (n-P) needs no
 * gather: rigid-high zeroes its adjoint in the fold, and for a free/pml high
 * face it is dead (never updated by the forward, so nothing propagates
 * through it). The same ordering rules as periodic apply (pressure fold
 * before the source-gradient read; buoyancy gradient from the post-fold
 * adjoint velocity) because these folds also write interior nodes.
 *
 * FUNC(adj_fold_p_line_fr)/FUNC(adj_fold_v_line_fr) live in adj_line_bc.h
 * (see that header). */

/* Adjoint-velocity guard gather for one free/rigid-axis line (the guard
 * cells of the free/rigid faces only; see FUNC(adj_gather_v_guard_line) for
 * the payload, which is identical). */
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

#endif /* DW_ACOUSTIC_KERNELS_H */

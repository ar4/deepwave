#ifndef DW_VISCOACOUSTIC_KERNELS_H
#define DW_VISCOACOUSTIC_KERNELS_H

/*
 * viscoacoustic_kernels.h
 *
 * Single-source per-grid-point stencil math for the viscoacoustic (variable
 * density, single-mechanism SLS attenuation, staggered grid) wave equation
 * propagator (see viscoacoustic.c for the physics/PML/discretisation/adjoint
 * description). This header is included by BOTH viscoacoustic.c (compiled by
 * gcc) and viscoacoustic.cu (compiled by nvcc), after each has defined the
 * FUNC()/CAT() name-mangling macros and before either uses the M1/C2/AR/P/
 * VZ/... family below. It ports acoustic_kernels.h's recipe unchanged for the
 * velocity update (viscoacoustic's velocity equation IS acoustic's -- rho
 * dv/dt = -grad p, no attenuation term) and extends acoustic's combined
 * pressure update and backward (adjoint) functions with the SLS memory
 * variable `r`/`rmem` and its (m1, c2, ar) model fields, per the physics
 * comment at the top of viscoacoustic.c. It contains:
 *
 *  * The per-axis addressing/adjoint-term macros (ND_INDEX, M1, C2,
 *    BUOYANCY_Z/Y/X, P, RMEM, VZ/VY/VX, PHI_Z/Y/X, PSI_Z/Y/X, B_EFF, and the
 *    VZ_ADJOINT/VY_ADJOINT/VX_ADJOINT/P_ADJOINT_Z/Y/X family (+ their _PML
 *    twins)), lifted verbatim from viscoacoustic.cu's copy of these macros --
 *    viscoacoustic.cu's naming/addressing convention (wavefields p/rmem/vz/
 *    vy/vx/phi_z.../psi_z... addressed by `i`, model arrays m1/c2/buoyancy_*
 *    addressed by `j`) is adopted as the one shared convention, matching
 *    acoustic_kernels.h exactly. viscoacoustic.c's copy of the same macros
 *    was textually identical EXCEPT that it used `i` (not `j`) to address the
 *    model arrays m1_shot/c2_shot/buoyancy_*_shot -- a difference that
 *    disappears once the CPU driver passes `i` for `j` too at every call site
 *    below (it already is the shot-local index there, exactly as `j` is on
 *    the CUDA driver). The `ar_shot`/AR macro pair existed in both original
 *    files but AR(...) was never actually invoked anywhere (ar_shot is only
 *    ever read at the *centre* point, `ar_shot[i]`/`ar_shot[j]`, never via a
 *    neighbour-offset macro) -- it is dropped here as genuinely dead code,
 *    not a behaviour change.
 *  * FUNC(forward_vz_cell) / FUNC(forward_vy_cell) / FUNC(forward_vx_cell):
 *    the per-grid-point half-grid velocity update for each axis. This is
 *    IDENTICAL logic to acoustic_kernels.h's functions of the same name --
 *    viscoacoustic's velocity equation has no attenuation term -- so these
 *    three functions are the same per-axis mapping for the same reason
 *    acoustic's are (viscoacoustic.c already runs them as three separate
 *    OpenMP loops with independent bounds per axis; viscoacoustic.cu's
 *    forward_kernel_v already computes them as three independent,
 *    separately-gated blocks within one kernel).
 *    FUNC(forward_p_cell): the combined pressure AND SLS memory-variable
 *    update (div(v) accumulated across all present axes, like acoustic's
 *    forward_p_cell, plus the memory-variable Crank-Nicolson recursion
 *    r_new = ar*r_old - c2*theta and p -= dt*m1*theta + (dt/2)(r_old+r_new)),
 *    matching viscoacoustic.c's/viscoacoustic.cu's single combined pressure
 *    loop body.
 *  * FUNC(backward_vz_cell) / FUNC(backward_vy_cell) / FUNC(backward_vx_cell):
 *    the adjoint velocity update per axis. Structurally identical to
 *    acoustic's backward_v*_cell (same VZ_ADJOINT/PML formula shape,
 *    DIFFZH1/etc dispatch) but with acoustic's single `dt*K*p` term replaced
 *    by the effective pressure-adjoint field B_EFF = dt*(m1-c2/2)*Pbar +
 *    c2*Rbar (see viscoacoustic.c's header comment, ADJOINT section) -- so
 *    these take `m1_shot`/`c2_shot`/`rmem` pointers (neighbour-offset
 *    accessed via the B_EFF macro) where acoustic's took a single `k_shot`
 *    pointer.
 *    FUNC(backward_p_cell): the combined adjoint pressure/phi update AND the
 *    memory-variable adjoint recursion (Rbar_new = ar*Rbar - (dt/2)(1+ar)*Pbar)
 *    plus the (grad_m1, grad_c2, grad_ar) accumulation, mirroring
 *    acoustic's backward_p_cell's structure (div_term accumulated across all
 *    present axes via P_ADJOINT_Z/Y/X(_PML), unaffected by attenuation) with
 *    the SLS pieces added, matching viscoacoustic.c's/viscoacoustic.cu's
 *    single combined pressure-adjoint loop body exactly (same statement
 *    order: beff, then phi updates, then model gradients, then the rmem
 *    recursion, then div_term/p update).
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest and PML-region triple loop in
 * viscoacoustic.c, the CUDA kernel launch configuration and per-thread
 * grid-point assignment in viscoacoustic.cu -- stays in viscoacoustic.c/
 * viscoacoustic.cu exactly as before (per the common_cpu.h/common_gpu.h
 * split); only the math performed once the caller has identified "this" grid
 * point lives here.
 *
 * Addressing convention (matches scalar_kernels.h/scalar_born_kernels.h/
 * acoustic_kernels.h exactly): p/rmem/vz/vy/vx/phi_z/.../psi_z/.../
 * *_grad_store_1/grad_m1/grad_c2/grad_ar/grad_b* are addressed by `i` (the
 * CPU driver's per-shot-local flat index, which is also valid as the CUDA
 * driver's global shot-spanning index since both drivers pass
 * already-appropriately-offset pointers); m1_shot/c2_shot/buoyancy_*_shot are
 * addressed by `j` (the per-shot-local index) wherever a neighbour offset is
 * needed (via the M1/C2/BUOYANCY_Z/Y/X macros, used only by the *_ADJOINT/
 * P_ADJOINT_* families in the backward functions), because the CUDA driver's
 * m1_shot/c2_shot/buoyancy_*_shot pointers are shot-local (offset by
 * shot_idx*shot_numel, not i) while the wavefield arrays span all shots. On
 * the CPU driver, wavefields/m1_shot/c2_shot/buoyancy_*_shot are ALL already
 * offset to the current shot, so `i` and `j` are the same value there and the
 * CPU driver simply passes `i` for both. Cells that only ever read the
 * *centre* value of m1/c2/ar/buoyancy (no neighbour offset) take that value
 * directly, by value, exactly as scalar_kernels.h's forward_cell does for
 * v_shot: FUNC(forward_p_cell) takes m1_shot/c2_shot/ar_shot by value;
 * FUNC(backward_p_cell) likewise takes m1_shot/c2_shot/ar_shot by value
 * (its own div_term/phi/beff computation never needs a model-field
 * neighbour); only FUNC(backward_v*_cell) (via B_EFF's neighbour-offset
 * M1(dz,dy,dx)/C2(dz,dy,dx)) and FUNC(backward_p_cell)'s div_term (via
 * P_ADJOINT_*'s neighbour-offset BUOYANCY_Z/Y/X(dz,dy,dx)) need pointers.
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

// Models (addressed by the shot-local index j -- see the addressing-
// convention note above)
#define M1(dz, dy, dx) m1_shot[ND_INDEX(j, dz, dy, dx)]
#define C2(dz, dy, dx) c2_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// Wavefields (addressed by i -- see the addressing-convention note above)
#define P(dz, dy, dx) p[ND_INDEX(i, dz, dy, dx)]
// SLS memory variable (backward: adjoint of the memory variable).
#define RMEM(dz, dy, dx) rmem[ND_INDEX(i, dz, dy, dx)]
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

// Effective pressure-adjoint modulus field. Acoustic uses dt*K*p here; the
// SLS mechanism replaces it with dt*(m1-c2/2)*Pbar + c2*Rbar (see
// viscoacoustic.c's header comment, ADJOINT section). Under DW_GSLS
// (multi-mechanism GSLS variant) the single-mechanism c2*Rbar terms become
// per-mechanism sums, B_eff = dt*(m1 - sum_l c2_l/2)*Pbar + sum_l c2_l*
// Rbar_l, computed by FUNC(gsls_beff): `rmem` then holds the n_mech adjoint
// memory variables stacked with mechanism stride `r_lstride` (n_shots *
// shot_numel -- the (L, n_shots, spatial) stacked-buffer layout) and
// `c2_shot` the n_mech per-mechanism fields stacked with stride `m_lstride`
// (shot_numel; per-shot-batched models are rejected upstream for the GSLS
// compiled path). The callers (FUNC(backward_v*_cell) and the guard-gather
// lines) provide `n_mech`/`r_lstride`/`m_lstride` as parameters.
#ifdef DW_GSLS
DW_HOST_DEVICE DW_DTYPE FUNC(gsls_beff)(
    DW_DTYPE const *__restrict const p, DW_DTYPE const *__restrict const rmem,
    DW_DTYPE const *__restrict const m1_shot,
    DW_DTYPE const *__restrict const c2_shot, int64_t const wi,
    int64_t const mi, DW_DTYPE const dt, int64_t const n_mech,
    int64_t const r_lstride, int64_t const m_lstride) {
  DW_DTYPE c2sum = 0;
  DW_DTYPE c2rbar = 0;
  int64_t l;
  for (l = 0; l < n_mech; ++l) {
    DW_DTYPE const c2l = c2_shot[mi + l * m_lstride];
    c2sum += c2l;
    c2rbar += c2l * rmem[wi + l * r_lstride];
  }
  return dt * (m1_shot[mi] - (DW_DTYPE)0.5 * c2sum) * p[wi] + c2rbar;
}
#define B_EFF(dz, dy, dx)                                             \
  FUNC(gsls_beff)(p, rmem, m1_shot, c2_shot, ND_INDEX(i, dz, dy, dx), \
                  ND_INDEX(j, dz, dy, dx), dt, n_mech, r_lstride, m_lstride)
#else
#define B_EFF(dz, dy, dx)                                                   \
  (dt * (M1(dz, dy, dx) - (DW_DTYPE)0.5 * C2(dz, dy, dx)) * P(dz, dy, dx) + \
   C2(dz, dy, dx) * RMEM(dz, dy, dx))
#endif /* DW_GSLS */

#if DW_NDIM >= 3
#define VZ_ADJOINT(dz, dy, dx) (-B_EFF(dz, dy, dx))
#define VZ_ADJOINT_PML(dz, dy, dx) \
  (bz[z + dz] * PHI_Z(dz, dy, dx) - (1 + bz[z + dz]) * B_EFF(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VY_ADJOINT(dz, dy, dx) (-B_EFF(dz, dy, dx))
#define VY_ADJOINT_PML(dz, dy, dx) \
  (by[y + dy] * PHI_Y(dz, dy, dx) - (1 + by[y + dy]) * B_EFF(dz, dy, dx))
#endif
#define VX_ADJOINT(dz, dy, dx) (-B_EFF(dz, dy, dx))
#define VX_ADJOINT_PML(dz, dy, dx) \
  (bx[x + dx] * PHI_X(dz, dy, dx) - (1 + bx[x + dx]) * B_EFF(dz, dy, dx))

#if DW_NDIM >= 3
#define P_ADJOINT_Z(dz, dy, dx) (-dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#define P_ADJOINT_Z_PML(dz, dy, dx)  \
  (bzh[z + dz] * PSI_Z(dz, dy, dx) - \
   dt * (1 + bzh[z + dz]) * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define P_ADJOINT_Y(dz, dy, dx) (-dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define P_ADJOINT_Y_PML(dz, dy, dx)  \
  (byh[y + dy] * PSI_Y(dz, dy, dx) - \
   dt * (1 + byh[y + dy]) * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define P_ADJOINT_X(dz, dy, dx) (-dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define P_ADJOINT_X_PML(dz, dy, dx)  \
  (bxh[x + dx] * PSI_X(dz, dy, dx) - \
   dt * (1 + bxh[x + dx]) * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

/* ---- Forward: per-axis half-grid velocity updates (identical to acoustic,
 * no attenuation term on this equation) ---- */

#if DW_NDIM >= 3
/* v = v - dt * B * (grad(p) + psi), z axis. `i` addresses p/vz/psi_z/
 * bz_grad_store_1 (per-shot-local on the CPU driver, global on the CUDA
 * driver); `buoyancy_z_shot` is the (already shot-selected) buoyancy VALUE
 * at this point (read once by the caller -- this update never needs a
 * neighbour offset of it); `pml_z` is true when this point is inside the
 * z-axis PML region (false in the interior).
 */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const *__restrict const p, DW_DTYPE *__restrict const vz,
    DW_DTYPE *__restrict const psi_z,
    DW_DTYPE *__restrict const bz_grad_store_1, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bzh,
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
    DW_DTYPE const *__restrict const p, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const psi_y,
    DW_DTYPE *__restrict const by_grad_store_1, DW_DTYPE const buoyancy_y_shot,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const byh,
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
    DW_DTYPE const *__restrict const p, DW_DTYPE *__restrict const vx,
    DW_DTYPE *__restrict const psi_x,
    DW_DTYPE *__restrict const bx_grad_store_1, DW_DTYPE const buoyancy_x_shot,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bxh,
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

/* ---- Forward: combined pressure + SLS memory-variable update ---- */

/* theta = div(v) + phi (combining every present axis, like acoustic's
 * forward_p_cell combines its axes into div_v); r_new = ar*r_old - c2*theta
 * (Crank-Nicolson); p -= dt*m1*theta + (dt/2)*(r_old+r_new). `i` addresses
 * p/rmem/vz/vy/vx/phi_z/.../m1_grad_store_1/... (per-shot-local on the CPU
 * driver, global on the CUDA driver); `m1_shot`/`c2_shot`/`ar_shot` are the
 * (already shot-selected) model VALUEs at this point (read once by the
 * caller -- never need a neighbour offset here); `pml_z`/`pml_y`/`pml_x` are
 * true when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_p_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const p,
    DW_DTYPE *__restrict const rmem,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const phi_y,
#endif
    DW_DTYPE *__restrict const phi_x, DW_DTYPE const m1_shot,
#ifdef DW_GSLS
    /* Multi-mechanism GSLS: c2/ar are per-mechanism fields (stacked with
     * stride m_lstride, indexed from mj), rmem holds n_mech stacked memory
     * variables (stride r_lstride, indexed from i), and the r_old
     * checkpoint store is n_mech-times bigger (stride g_lstride, indexed
     * from gidx). */
    DW_DTYPE const *__restrict const c2_shot,
    DW_DTYPE const *__restrict const ar_shot, int64_t const mj,
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride,
    int64_t const gidx, int64_t const g_lstride,
#else
    DW_DTYPE const c2_shot, DW_DTYPE const ar_shot,
#endif
    DW_DTYPE *__restrict const m1_grad_store_1,
    DW_DTYPE *__restrict const c2_grad_store_1,
    DW_DTYPE *__restrict const ar_grad_store_1, DW_DTYPE const dt,
    bool const kmod_requires_grad, bool const ar_requires_grad, int64_t const i,
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

#ifdef DW_GSLS
  if (kmod_requires_grad) {
    // theta drives both grad_m1 and every mechanism's grad_c2_l.
    m1_grad_store_1[i] = div_v;
    c2_grad_store_1[i] = div_v;
  }
  {
    // Each mechanism's Crank-Nicolson recursion is independent; their
    // (r_old_l + r_new_l) contributions sum into the pressure update
    // (matching deepwave.viscoacoustic.update_pressure's n_mech > 1 branch).
    DW_DTYPE sum_r = 0;
    int64_t l;
    for (l = 0; l < n_mech; ++l) {
      DW_DTYPE const r_old = rmem[i + l * r_lstride];
      DW_DTYPE const r_new = ar_shot[mj + l * m_lstride] * r_old -
                             c2_shot[mj + l * m_lstride] * div_v;
      if (ar_requires_grad) {
        ar_grad_store_1[gidx + l * g_lstride] = r_old;
      }
      sum_r += r_old + r_new;
      rmem[i + l * r_lstride] = r_new;
    }
    p[i] -= dt * m1_shot * div_v + (DW_DTYPE)0.5 * dt * sum_r;
  }
#else
  DW_DTYPE const r_old = rmem[i];
  DW_DTYPE const r_new = ar_shot * r_old - c2_shot * div_v;

  if (kmod_requires_grad) {
    // theta drives both grad_m1 and grad_c2.
    m1_grad_store_1[i] = div_v;
    c2_grad_store_1[i] = div_v;
  }
  if (ar_requires_grad) {
    ar_grad_store_1[i] = r_old;
  }
  p[i] -= dt * m1_shot * div_v + (DW_DTYPE)0.5 * dt * (r_old + r_new);
  rmem[i] = r_new;
#endif /* DW_GSLS */
}

/* ---- Backward: per-axis half-grid adjoint velocity updates ---- */

#if DW_NDIM >= 3
/* Adjoint of FUNC(forward_vz_cell)'s z-axis velocity update. Structurally
 * acoustic_kernels.h's FUNC(backward_vz_cell) with the pressure-adjoint
 * payload dt*K*Pbar replaced by B_EFF = dt*(m1 - c2/2)*Pbar + c2*Rbar, which
 * is what the transpose of the COUPLED pressure/memory-variable update
 * contributes to thetabar (both p^{n+1} and r^{n+1} depend on the same
 * divergence theta, so both adjoints flow back through the same stencil).
 * Derived in viscoacoustic.c's header comment (ADJOINT) and
 * docs/viscoacoustic.rst; verified numerically by
 * tests/test_adjoint_dot_product.py.
 *
 * v = v + dt*(grad(B_eff) - grad(b*phi)); psi = a*psi - dt*B*a*v, z axis
 * (acoustic's dt*K*p replaced by B_eff = dt*(m1-c2/2)*Pbar + c2*Rbar -- see
 * viscoacoustic.c's header comment, ADJOINT section). `i` addresses
 * p/rmem/vz/phi_z/psi_z/psi_zn/bz_grad_store_1/grad_bz (per-shot-local on the
 * CPU driver, global on the CUDA driver); `j` addresses the already-shot-
 * sliced m1_shot/c2_shot (equal to `i` on the CPU driver; the local per-shot
 * grid index on the CUDA driver, distinct from the global `i`) -- see the
 * addressing-convention note at the top of this file. `psi_z` is the OLD
 * (pre-this-step) buffer (read only); `psi_zn` is the NEW buffer (written);
 * `pml_z` is true inside the z-axis PML region.
 */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const m1_shot,
    DW_DTYPE const *__restrict const c2_shot,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE *__restrict const vz,
    DW_DTYPE const *__restrict const phi_z,
    DW_DTYPE const *__restrict const psi_z, DW_DTYPE *__restrict const psi_zn,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE *__restrict const grad_bz,
    DW_DTYPE const *__restrict const bz_grad_store_1,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    int64_t const i, int64_t const j,
#ifdef DW_GSLS
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride,
#endif
    int64_t const ny, int64_t const nx, int64_t const z, bool const pml_z,
    DW_DTYPE const rdz) {
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
    DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const m1_shot,
    DW_DTYPE const *__restrict const c2_shot,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE *__restrict const vy,
    DW_DTYPE const *__restrict const phi_y,
    DW_DTYPE const *__restrict const psi_y, DW_DTYPE *__restrict const psi_yn,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE *__restrict const grad_by,
    DW_DTYPE const *__restrict const by_grad_store_1,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    int64_t const i, int64_t const j,
#ifdef DW_GSLS
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride,
#endif
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
    DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const m1_shot,
    DW_DTYPE const *__restrict const c2_shot,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE *__restrict const vx,
    DW_DTYPE const *__restrict const phi_x,
    DW_DTYPE const *__restrict const psi_x, DW_DTYPE *__restrict const psi_xn,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE *__restrict const grad_bx,
    DW_DTYPE const *__restrict const bx_grad_store_1,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const dt, int64_t const step_ratio, bool const b_requires_grad,
    int64_t const i, int64_t const j,
#ifdef DW_GSLS
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride,
#endif
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

/* ---- Backward: combined pressure/phi adjoint update + memory-variable
 * adjoint recursion ---- */

/* Adjoint of FUNC(forward_p_cell)'s coupled pressure/memory-variable step.
 * The transpose of r^{n+1} = ar*r^n - c2*theta and
 * p^{n+1} = p^n - dt*m1*theta - (dt/2)(r^n + r^{n+1}) gives the backward-in-
 * time memory recursion Rbar_new = ar*Rbar - (dt/2)(1+ar)*Pbar and the three
 * model gradients below; the wavefield part is acoustic's adjoint pressure
 * update with B_EFF in place of dt*K*Pbar. Derived in viscoacoustic.c's
 * header comment (ADJOINT) and docs/viscoacoustic.rst; verified numerically
 * by tests/test_adjoint_dot_product.py.
 *
 * beff = dt*(m1-c2/2)*Pbar + c2*Rbar; phi = a*phi - a*beff; grad_m1 +=
 * -dt*theta*Pbar; grad_c2 += theta*((dt/2)*Pbar - Rbar); grad_ar +=
 * r_old*(Rbar - (dt/2)*Pbar); Rbar_new = ar*Rbar - (dt/2)(1+ar)*Pbar
 * (SLS-like backward-in-time recursion); p += div(B(1+b)v) - div(b*psi),
 * combining every present axis (see FUNC(forward_p_cell) and acoustic's
 * backward_p_cell). `i`/`j` addressing matches FUNC(backward_vz_cell) above.
 * `psi_z`/`psi_y`/`psi_x` are the OLD (pre-this-step) velocity-PML buffers --
 * P_ADJOINT_Z_PML/Y_PML/X_PML read the same buffer FUNC(backward_vz_cell)/
 * vy_cell/vx_cell read (not the buffer they just wrote), matching the
 * transpose structure exactly. `m1_shot`/`c2_shot`/`ar_shot` are read only at
 * the centre point here (no neighbour offset), so are passed by value.
 */
DW_HOST_DEVICE void FUNC(backward_p_cell)(
    DW_DTYPE *__restrict const p, DW_DTYPE *__restrict const rmem,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz, DW_DTYPE *__restrict const phi_z,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const psi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const phi_y,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const psi_y,
#endif
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const phi_x,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const psi_x, DW_DTYPE const m1_shot,
#ifdef DW_GSLS
    /* Multi-mechanism GSLS: c2/ar are per-mechanism fields (stride
     * m_lstride from mj), rmem the n_mech stacked adjoint memory variables
     * (stride r_lstride from i), grad_c2/grad_ar and the r_old checkpoint
     * store n_mech-times bigger (stride g_lstride from gidx). */
    DW_DTYPE const *__restrict const c2_shot,
    DW_DTYPE const *__restrict const ar_shot, int64_t const mj,
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride,
    int64_t const gidx, int64_t const g_lstride,
#else
    DW_DTYPE const c2_shot, DW_DTYPE const ar_shot,
#endif
    DW_DTYPE *__restrict const grad_m1, DW_DTYPE *__restrict const grad_c2,
    DW_DTYPE *__restrict const grad_ar,
    DW_DTYPE const *__restrict const m1_grad_store_1,
    DW_DTYPE const *__restrict const c2_grad_store_1,
    DW_DTYPE const *__restrict const ar_grad_store_1, DW_DTYPE const dt,
    int64_t const step_ratio, bool const kmod_requires_grad,
    bool const ar_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdx) {
  // Incoming adjoints (before this step modifies them) and the effective
  // pressure-adjoint field B_eff.
  DW_DTYPE const pbar = p[i];
#ifdef DW_GSLS
  // B_eff = dt*(m1 - sum_l c2_l/2)*Pbar + sum_l c2_l*Rbar_l (the n_mech > 1
  // generalisation of the single-mechanism formula below; the same sum
  // FUNC(gsls_beff) computes, inlined here because this function receives
  // m1 as an already-loaded centre value).
  DW_DTYPE beff;
  {
    DW_DTYPE c2sum = 0;
    DW_DTYPE c2rbar = 0;
    int64_t l;
    for (l = 0; l < n_mech; ++l) {
      DW_DTYPE const c2l = c2_shot[mj + l * m_lstride];
      c2sum += c2l;
      c2rbar += c2l * rmem[i + l * r_lstride];
    }
    beff = dt * (m1_shot - (DW_DTYPE)0.5 * c2sum) * pbar + c2rbar;
  }
#else
  DW_DTYPE const rbar = rmem[i];
  DW_DTYPE const beff =
      dt * (m1_shot - (DW_DTYPE)0.5 * c2_shot) * pbar + c2_shot * rbar;
#endif

  // Update Phi (pressure PML adjoint): acoustic's dt*K*p -> B_eff.
#if DW_NDIM >= 3
  if (pml_z) phi_z[i] = az[z] * phi_z[i] - az[z] * beff;
#endif
#if DW_NDIM >= 2
  if (pml_y) phi_y[i] = ay[y] * phi_y[i] - ay[y] * beff;
#endif
  if (pml_x) phi_x[i] = ax[x] * phi_x[i] - ax[x] * beff;

#ifdef DW_GSLS
  // Model gradients (theta stored once for m1 and every grad_c2_l; r_old_l
  // stored per mechanism for grad_ar_l), then each mechanism's independent
  // adjoint recursion Rbar_new_l = ar_l*Rbar_l - (dt/2)(1+ar_l)*Pbar. The
  // gradient reads of Rbar_l precede the recursion's overwrite.
  if (kmod_requires_grad) {
    grad_m1[i] -= dt * pbar * m1_grad_store_1[i] * (DW_DTYPE)step_ratio;
    {
      int64_t l;
      for (l = 0; l < n_mech; ++l) {
        grad_c2[gidx + l * g_lstride] +=
            ((DW_DTYPE)0.5 * dt * pbar - rmem[i + l * r_lstride]) *
            c2_grad_store_1[i] * (DW_DTYPE)step_ratio;
      }
    }
  }
  if (ar_requires_grad) {
    int64_t l;
    for (l = 0; l < n_mech; ++l) {
      grad_ar[gidx + l * g_lstride] +=
          (rmem[i + l * r_lstride] - (DW_DTYPE)0.5 * dt * pbar) *
          ar_grad_store_1[gidx + l * g_lstride] * (DW_DTYPE)step_ratio;
    }
  }
  {
    int64_t l;
    for (l = 0; l < n_mech; ++l) {
      DW_DTYPE const arl = ar_shot[mj + l * m_lstride];
      rmem[i + l * r_lstride] =
          arl * rmem[i + l * r_lstride] - (DW_DTYPE)0.5 * dt * (1 + arl) * pbar;
    }
  }
#else
  // Model gradients. theta (=div_v) was stored for m1 & c2; r_old for ar.
  // grad_c2 and grad_ar share the factor (Rbar - (dt/2) Pbar).
  if (kmod_requires_grad) {
    grad_m1[i] -= dt * pbar * m1_grad_store_1[i] * (DW_DTYPE)step_ratio;
    grad_c2[i] += ((DW_DTYPE)0.5 * dt * pbar - rbar) * c2_grad_store_1[i] *
                  (DW_DTYPE)step_ratio;
  }
  if (ar_requires_grad) {
    grad_ar[i] += (rbar - (DW_DTYPE)0.5 * dt * pbar) * ar_grad_store_1[i] *
                  (DW_DTYPE)step_ratio;
  }

  // Memory-variable adjoint recursion (SLS-like, backward in time; per-point,
  // no PML, runs unconditionally as it feeds the velocity/pressure adjoints
  // via B_eff at the next step).
  rmem[i] = ar_shot * rbar - (DW_DTYPE)0.5 * dt * (1 + ar_shot) * pbar;
#endif /* DW_GSLS */

  // Update P
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
 * Same fold/gather scheme as acoustic_kernels.h (see the comment there): on
 * a periodic axis the exact transpose of the forward guard fill is a fold
 * (guard cells' adjoint values are returned to the interior nodes the
 * forward copied them from, then zeroed), preceded by a gather of the
 * adjoint-stencil contributions that fall on the guard cells. The adjoint
 * memory variable (rmem) gets the same fold as the adjoint pressure -- the
 * forward fills its guard ring identically -- but needs no gather (nothing
 * differentiates rmem: its ghost cells receive no adjoint-stencil
 * contribution). The velocity-guard gather's payload is B_EFF (the SLS
 * effective pressure-adjoint field) instead of acoustic's dt*K*P.
 */

/* FUNC(adj_fold_p_line)/FUNC(adj_fold_v_line)/FUNC(adj_gather_p_guard_line)
 * and their _fr counterparts live in adj_line_bc.h (see that header and
 * acoustic_kernels.h's copy of this same comment). */
#include "adj_line_bc.h"

/* Adjoint-velocity guard gather: the guard cells (and the slaved extra high
 * interior node) receive the same -DIFF*H1(-B_EFF) contribution the
 * interior receives in FUNC(backward_v*_cell), out-of-range taps dropped
 * (the adjoint pressure/memory fields are zero there once folded). */
DW_HOST_DEVICE void FUNC(adj_gather_v_guard_line)(
    DW_DTYPE *__restrict const v, DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const rmem,
    DW_DTYPE const *__restrict const m1_shot,
    DW_DTYPE const *__restrict const c2_shot, int64_t const base,
    int64_t const jbase, int64_t const nax, int64_t const stride,
    DW_DTYPE const dt, DW_DTYPE const rd, DW_DTYPE const *__restrict const sw
#ifdef DW_GSLS
    ,
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride
#endif
) {
  int64_t const P = FD_PAD;
  int64_t c, q;
  for (c = 0; c < 2 * P; ++c) {
    int64_t const y = (c < P) ? c : nax - 2 * P + c;
    DW_DTYPE acc = 0;
    for (q = 0; q < P; ++q) {
      int64_t const th = y + 1 + q;
      int64_t const tl = y - q;
      if (th < nax) {
        int64_t const wi = base + th * stride;
        int64_t const mi = jbase + th * stride;
#ifdef DW_GSLS
        acc += sw[q] * FUNC(gsls_beff)(p, rmem, m1_shot, c2_shot, wi, mi, dt,
                                       n_mech, r_lstride, m_lstride);
#else
        acc +=
            sw[q] * (dt * (m1_shot[mi] - (DW_DTYPE)0.5 * c2_shot[mi]) * p[wi] +
                     c2_shot[mi] * rmem[wi]);
#endif
      }
      if (tl >= 0) {
        int64_t const wi = base + tl * stride;
        int64_t const mi = jbase + tl * stride;
#ifdef DW_GSLS
        acc -= sw[q] * FUNC(gsls_beff)(p, rmem, m1_shot, c2_shot, wi, mi, dt,
                                       n_mech, r_lstride, m_lstride);
#else
        acc -=
            sw[q] * (dt * (m1_shot[mi] - (DW_DTYPE)0.5 * c2_shot[mi]) * p[wi] +
                     c2_shot[mi] * rmem[wi]);
#endif
      }
    }
    v[base + y * stride] += rd * acc;
  }
}

/* ---- Backward: free/rigid-face adjoint guard maintenance ---- */
/*
 * Same exact-transpose fold/gather scheme as acoustic_kernels.h's _fr
 * functions (see the comment there): the re-fill realises the free/rigid
 * transpose only for the (free-low, rigid-high) pairing; other combinations
 * fold with the mirror/sign-flip transposes of the fills, per face. The
 * adjoint memory variable (rmem) gets the same _fr pressure fold as the
 * adjoint pressure (the forward fills its guard ring identically) and needs
 * no gather; the velocity-guard gather's payload is B_EFF.
 */

/* Free/rigid adjoint-velocity guard gather (B_EFF payload, per-face gated;
 * the extra high interior node n-P needs no gather -- see
 * acoustic_kernels.h's FUNC(adj_gather_v_guard_line_fr)). */
DW_HOST_DEVICE void FUNC(adj_gather_v_guard_line_fr)(
    DW_DTYPE *__restrict const v, DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const rmem,
    DW_DTYPE const *__restrict const m1_shot,
    DW_DTYPE const *__restrict const c2_shot, int64_t const base,
    int64_t const jbase, int64_t const nax, int64_t const stride,
    DW_DTYPE const dt, DW_DTYPE const rd, DW_DTYPE const *__restrict const sw,
    int64_t const bc_lo, int64_t const bc_hi
#ifdef DW_GSLS
    ,
    int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride
#endif
) {
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
        if (th < nax) {
          int64_t const wi = base + th * stride;
          int64_t const mi = jbase + th * stride;
#ifdef DW_GSLS
          acc += sw[q] * FUNC(gsls_beff)(p, rmem, m1_shot, c2_shot, wi, mi, dt,
                                         n_mech, r_lstride, m_lstride);
#else
          acc += sw[q] *
                 (dt * (m1_shot[mi] - (DW_DTYPE)0.5 * c2_shot[mi]) * p[wi] +
                  c2_shot[mi] * rmem[wi]);
#endif
        }
        if (tl >= 0) {
          int64_t const wi = base + tl * stride;
          int64_t const mi = jbase + tl * stride;
#ifdef DW_GSLS
          acc -= sw[q] * FUNC(gsls_beff)(p, rmem, m1_shot, c2_shot, wi, mi, dt,
                                         n_mech, r_lstride, m_lstride);
#else
          acc -= sw[q] *
                 (dt * (m1_shot[mi] - (DW_DTYPE)0.5 * c2_shot[mi]) * p[wi] +
                  c2_shot[mi] * rmem[wi]);
#endif
        }
      }
      v[base + y * stride] += rd * acc;
    }
  }
}

#endif /* DW_VISCOACOUSTIC_KERNELS_H */

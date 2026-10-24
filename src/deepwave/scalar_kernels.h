#ifndef DW_SCALAR_KERNELS_H
#define DW_SCALAR_KERNELS_H

/*
 * scalar_kernels.h
 *
 * Single-source per-grid-point stencil math for the scalar wave equation
 * propagator (see scalar.c for the physics/PML description). This header is
 * included by BOTH scalar.c (compiled by gcc) and scalar.cu (compiled by
 * nvcc), after each has defined the FUNC()/CAT() name-mangling macros and
 * before either uses the WFC/PSIZ/... family below. It contains:
 *
 *  * The per-axis addressing/PML-term macros (ND_INDEX, WFC, V2DT2_WFC,
 *    PSIZ/ZETAZ/AZ_PSIZ (+Y, +X), and the UT_TERM/PSI_TERM family), copied
 *    verbatim from
 *    what used to be two independently-maintained (but already textually
 *    identical, modulo local variable names) copies in scalar.c and
 *    scalar.cu.
 *  * FUNC(forward_cell) / FUNC(backward_cell): the actual per-grid-point
 *    finite-difference/PML update, i.e. the stencil body. Everything about
 *    HOW the loop over grid points is launched and threaded -- the OpenMP
 *    parallel/simd/collapse loop nest in scalar.c, the CUDA kernel launch
 *    configuration and per-thread grid-point assignment in scalar.cu -- stays
 *    in scalar.c/scalar.cu exactly as before (per the common_cpu.h/
 *    common_gpu.h split); only the math performed once the caller has
 *    identified "this" grid point lives here.
 *
 * These functions take the grid point's coordinates/PML-region flags and all
 * the arrays/scalars they touch as ordinary parameters, so the exact same
 * function body works whether the caller is the CPU driver (where wfc/psiz/
 * zetaz/... are already offset to the current shot, so the "whole-array"
 * index i doubles as the per-shot local index) or the CUDA driver (where wfc/
 * psiz/zetaz/... span all shots, so i is shot_idx * shot_numel + j, and j is
 * the per-shot local index used only for the already-shot-sliced v2dt2_shot
 * in the backward pass). DW_HOST_DEVICE marks them callable from either
 * gcc (a plain forced-inline static function) or nvcc (additionally callable
 * from device code).
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

#define WFC(dz, dy, dx) wfc[ND_INDEX(i, dz, dy, dx)]
#define V2DT2_WFC(dz, dy, dx) \
  v2dt2_shot[ND_INDEX(j, dz, dy, dx)] * WFC(dz, dy, dx)

#if DW_NDIM >= 3
#define PSIZ(dz, dy, dx) psiz[ND_INDEX(i, dz, dy, dx)]
#define ZETAZ(dz, dy, dx) zetaz[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZ(dz, dy, dx) az[z + (dz)] * PSIZ(dz, dy, dx)
#endif

#if DW_NDIM >= 2
#define PSIY(dz, dy, dx) psiy[ND_INDEX(i, dz, dy, dx)]
#define ZETAY(dz, dy, dx) zetay[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIY(dz, dy, dx) ay[y + (dy)] * PSIY(dz, dy, dx)
#endif

#define PSIX(dz, dy, dx) psix[ND_INDEX(i, dz, dy, dx)]
#define ZETAX(dz, dy, dx) zetax[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIX(dz, dy, dx) ax[x + (dx)] * PSIX(dz, dy, dx)

#if DW_NDIM == 3
#define UT_TERMZ1(dz, dy, dx)                                    \
  (dbzdz[z + (dz)] * ((1 + bz[z + (dz)]) * V2DT2_WFC(dz, 0, 0) + \
                      bz[z + (dz)] * ZETAZ(dz, 0, 0)) +          \
   bz[z + (dz)] * PSIZ(dz, 0, 0))
#define UT_TERMZ2(dz, dy, dx)                                       \
  ((1 + bz[z + (dz)]) * ((1 + bz[z + (dz)]) * V2DT2_WFC(dz, 0, 0) + \
                         bz[z + (dz)] * ZETAZ(dz, 0, 0)))
#define PSIZ_TERM(dz, dy, dx) \
  ((1 + bz[z + (dz)]) * V2DT2_WFC(dz, 0, 0) + bz[z + (dz)] * ZETAZ(dz, 0, 0))
#endif

#if DW_NDIM >= 2
#define UT_TERMY1(dz, dy, dx)                                    \
  (dbydy[y + (dy)] * ((1 + by[y + (dy)]) * V2DT2_WFC(0, dy, 0) + \
                      by[y + (dy)] * ZETAY(0, dy, 0)) +          \
   by[y + (dy)] * PSIY(0, dy, 0))
#define UT_TERMY2(dz, dy, dx)                                       \
  ((1 + by[y + (dy)]) * ((1 + by[y + (dy)]) * V2DT2_WFC(0, dy, 0) + \
                         by[y + (dy)] * ZETAY(0, dy, 0)))
#define PSIY_TERM(dz, dy, dx) \
  ((1 + by[y + (dy)]) * V2DT2_WFC(0, dy, 0) + by[y + (dy)] * ZETAY(0, dy, 0))
#endif

#define UT_TERMX1(dz, dy, dx)                                    \
  (dbxdx[x + (dx)] * ((1 + bx[x + (dx)]) * V2DT2_WFC(0, 0, dx) + \
                      bx[x + (dx)] * ZETAX(0, 0, dx)) +          \
   bx[x + (dx)] * PSIX(0, 0, dx))
#define UT_TERMX2(dz, dy, dx)                                       \
  ((1 + bx[x + (dx)]) * ((1 + bx[x + (dx)]) * V2DT2_WFC(0, 0, dx) + \
                         bx[x + (dx)] * ZETAX(0, 0, dx)))
#define PSIX_TERM(dz, dy, dx) \
  ((1 + bx[x + (dx)]) * V2DT2_WFC(0, 0, dx) + bx[x + (dx)] * ZETAX(0, 0, dx))

/* Forward stencil update for one grid point. `i` is the flat index used to
 * address wfc/wfp/w_store_1/psi/zeta arrays (per-shot-local on the CPU driver,
 * global across all shots on the CUDA driver); `v_shot` is the (already
 * shot-selected) wavespeed at this point; `pml_z`/`pml_y`/`pml_x` are true
 * when this point is inside that axis's PML region (false in the interior).
 */
DW_HOST_DEVICE void FUNC(forward_cell)(
    DW_DTYPE const* __restrict const wfc, DW_DTYPE* __restrict const wfp,
    DW_DTYPE* __restrict const w_store_1, DW_DTYPE const v_shot,
    DW_DTYPE const dt2, bool const v_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const psiz,
    DW_DTYPE* __restrict const psizn, DW_DTYPE* __restrict const zetaz,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz, DW_DTYPE const rdz,
    DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const psiy,
    DW_DTYPE* __restrict const psiyn, DW_DTYPE* __restrict const zetay,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy, DW_DTYPE const rdy,
    DW_DTYPE const rdy2,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const psix,
    DW_DTYPE* __restrict const psixn, DW_DTYPE* __restrict const zetax,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx, DW_DTYPE const rdx,
    DW_DTYPE const rdx2) {
  DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ2(WFC);
  } else {
    DW_DTYPE dwfcdz = DIFFZ1(WFC);
    DW_DTYPE tmpz =
        ((1 + bz[z]) * DIFFZ2(WFC) + dbzdz[z] * dwfcdz + DIFFZ1(AZ_PSIZ));
    w_sum += (1 + bz[z]) * tmpz + az[z] * zetaz[i];
    psizn[i] = bz[z] * dwfcdz + az[z] * psiz[i];
    zetaz[i] = bz[z] * tmpz + az[z] * zetaz[i];
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY2(WFC);
  } else {
    DW_DTYPE dwfcdy = DIFFY1(WFC);
    DW_DTYPE tmpy =
        ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
    w_sum += (1 + by[y]) * tmpy + ay[y] * zetay[i];
    psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];
    zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
  }
#endif
  if (!pml_x) {
    w_sum += DIFFX2(WFC);
  } else {
    DW_DTYPE dwfcdx = DIFFX1(WFC);
    DW_DTYPE tmpx =
        ((1 + bx[x]) * DIFFX2(WFC) + dbxdx[x] * dwfcdx + DIFFX1(AX_PSIX));
    w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];
    psixn[i] = bx[x] * dwfcdx + ax[x] * psix[i];
    zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];
  }
  if (v_requires_grad) {
    w_store_1[i] = 2 * v_shot * dt2 * w_sum;
  }
  wfp[i] = v_shot * v_shot * dt2 * w_sum + 2 * wfc[i] - wfp[i];
}

/* Backward (adjoint) stencil update for one grid point. `i` addresses wfc/
 * wfp/grad_v_shot/psi/zeta/zetan arrays (per-shot-local on the CPU driver,
 * global on the CUDA driver); `j` addresses the already-shot-sliced v2dt2_shot
 * (equal to `i` on the CPU driver, where wfc/... are themselves already
 * shot-sliced so the two coincide; the local per-shot grid index on the CUDA
 * driver, distinct from the global `i`).
 */
DW_HOST_DEVICE void FUNC(backward_cell)(
    DW_DTYPE const* __restrict const wfc, DW_DTYPE* __restrict const wfp,
    DW_DTYPE const* __restrict const v2dt2_shot,
    DW_DTYPE const* __restrict const w_store_1,
    DW_DTYPE* __restrict const grad_v_shot, bool const v_requires_grad,
    int64_t const step_ratio, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const psiz,
    DW_DTYPE* __restrict const psizn, DW_DTYPE const* __restrict const zetaz,
    DW_DTYPE* __restrict const zetazn, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const dbzdz,
    DW_DTYPE const rdz, DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const psiy,
    DW_DTYPE* __restrict const psiyn, DW_DTYPE const* __restrict const zetay,
    DW_DTYPE* __restrict const zetayn, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const dbydy,
    DW_DTYPE const rdy, DW_DTYPE const rdy2,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const psix,
    DW_DTYPE* __restrict const psixn, DW_DTYPE const* __restrict const zetax,
    DW_DTYPE* __restrict const zetaxn, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const dbxdx,
    DW_DTYPE const rdx, DW_DTYPE const rdx2) {
  DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ2(V2DT2_WFC);
  } else {
    w_sum += -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2);
    psizn[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz[i];
    zetazn[i] = az[z] * V2DT2_WFC(0, 0, 0) + az[z] * zetaz[i];
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY2(V2DT2_WFC);
  } else {
    w_sum += -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2);
    psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
    zetayn[i] = ay[y] * V2DT2_WFC(0, 0, 0) + ay[y] * zetay[i];
  }
#endif
  if (!pml_x) {
    w_sum += DIFFX2(V2DT2_WFC);
  } else {
    w_sum += -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2);
    psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
    zetaxn[i] = ax[x] * V2DT2_WFC(0, 0, 0) + ax[x] * zetax[i];
  }
  wfp[i] = w_sum + 2 * wfc[i] - wfp[i];
  if (v_requires_grad) {
    grad_v_shot[i] += wfc[i] * w_store_1[i] * (DW_DTYPE)step_ratio;
  }
}

#endif /* DW_SCALAR_KERNELS_H */

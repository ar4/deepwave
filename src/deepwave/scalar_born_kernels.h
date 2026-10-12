#ifndef DW_SCALAR_BORN_KERNELS_H
#define DW_SCALAR_BORN_KERNELS_H

/*
 * scalar_born_kernels.h
 *
 * Single-source per-grid-point stencil math for the scalar Born wave
 * equation propagator (see scalar_born.c for the physics/PML description).
 * This header is included by BOTH scalar_born.c (compiled by gcc) and
 * scalar_born.cu (compiled by nvcc), after each has defined the
 * FUNC()/CAT() name-mangling macros and before either uses the WFC/PSIZ/...
 * family below. It contains:
 *
 *  * The per-axis addressing/PML-term macros (ND_INDEX, WFC/WFCSC,
 *    PSIZ/ZETAZ/AZ_PSIZ (+Y, +X) and their *SC scattered-field twins, V/VDT2/
 *    V2DT2/SCATTER, V2DT2_WFC/V2DT2_WFCSC, and the UT_TERM/UTSC_TERM/
 *    PSI*_TERM/PSI*SC_TERM family), copied verbatim from what used to be two
 *    independently-maintained (but already textually identical, modulo local
 *    variable/index names -- see below) copies in scalar_born.c and
 *    scalar_born.cu.
 *  * FUNC(forward_cell): the per-grid-point update of BOTH the background
 *    and scattered wavefields for one time step (scalar_born couples the two
 *    fields -- the scattered field's source term is the background
 *    wavefield times the scattering potential -- so, unlike plain scalar,
 *    there is exactly one combined forward per-cell function, matching the
 *    single combined loop body already present in both scalar_born.c's
 *    FUNC(forward) and scalar_born.cu's forward_kernel).
 *  * FUNC(backward_cell): the combined adjoint update of both fields,
 *    accumulating both grad_v and grad_scatter, matching scalar_born.c's
 *    FUNC(backward) and scalar_born.cu's backward_kernel.
 *  * FUNC(backward_cell_sc): the scattered-field-only adjoint update (no
 *    background field, no grad_v), matching scalar_born.c's
 *    FUNC(backward_sc) and scalar_born.cu's backward_kernel_sc. This third
 *    function exists because, when only the scattering potential requires a
 *    gradient (v does not), the background adjoint field need not be
 *    propagated at all -- an optimisation specific to this coupled
 *    propagator with no analogue in plain scalar's two-function (forward/
 *    backward) template.
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest in scalar_born.c, the CUDA
 * kernel launch configuration and per-thread grid-point assignment in
 * scalar_born.cu -- stays in scalar_born.c/scalar_born.cu exactly as before
 * (per the common_cpu.h/common_gpu.h split); only the math performed once
 * the caller has identified "this" grid point lives here.
 *
 * Addressing convention (matches scalar_kernels.h exactly): wfc/wfp/wfcsc/
 * wfpsc/psiz/.../zetaz/w_store/wsc_store/grad_v/grad_scatter are addressed by
 * `i` (the CPU driver's per-shot-local flat index, which is also valid as
 * the CUDA driver's global shot-spanning index since both drivers pass
 * already-appropriately-offset pointers); v_shot/scatter_shot are addressed
 * by `j` (the per-shot-local index) wherever a neighbour offset is needed
 * (via the V/VDT2/V2DT2/SCATTER macros), because the CUDA driver's v_shot/
 * scatter_shot pointers are shot-local (offset by shot_idx*shot_numel, not
 * i) while wfc/... span all shots. On the CPU driver, wfc/.../v_shot/
 * scatter_shot are ALL already offset to the current shot, so `i` and `j`
 * are the same value there and the CPU driver simply passes `i` for both.
 * forward_cell needs no `j`: it only ever reads the *centre* value of v/
 * scatter (no neighbour offsets), so the caller passes that value directly,
 * by value, exactly as scalar_kernels.h's forward_cell does for v_shot.
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

// Access the background wavefield at offset (dz, dy, dx) from i
#define WFC(dz, dy, dx) wfc[ND_INDEX(i, dz, dy, dx)]
// Access the scattered wavefield at offset (dz, dy, dx) from i
#define WFCSC(dz, dy, dx) wfcsc[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define PSIZ(dz, dy, dx) psiz[ND_INDEX(i, dz, dy, dx)]
#define ZETAZ(dz, dy, dx) zetaz[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZ(dz, dy, dx) az[z + (dz)] * PSIZ(dz, dy, dx)
#define PSIZSC(dz, dy, dx) psizsc[ND_INDEX(i, dz, dy, dx)]
#define ZETAZSC(dz, dy, dx) zetazsc[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZSC(dz, dy, dx) az[z + (dz)] * PSIZSC(dz, dy, dx)
#endif

#if DW_NDIM >= 2
#define PSIY(dz, dy, dx) psiy[ND_INDEX(i, dz, dy, dx)]
#define ZETAY(dz, dy, dx) zetay[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIY(dz, dy, dx) ay[y + (dy)] * PSIY(dz, dy, dx)
#define PSIYSC(dz, dy, dx) psiysc[ND_INDEX(i, dz, dy, dx)]
#define ZETAYSC(dz, dy, dx) zetaysc[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIYSC(dz, dy, dx) ay[y + (dy)] * PSIYSC(dz, dy, dx)
#endif

#define PSIX(dz, dy, dx) psix[ND_INDEX(i, dz, dy, dx)]
#define ZETAX(dz, dy, dx) zetax[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIX(dz, dy, dx) ax[x + (dx)] * PSIX(dz, dy, dx)
#define PSIXSC(dz, dy, dx) psixsc[ND_INDEX(i, dz, dy, dx)]
#define ZETAXSC(dz, dy, dx) zetaxsc[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIXSC(dz, dy, dx) ax[x + (dx)] * PSIXSC(dz, dy, dx)

// Access velocity at offset (dz, dy, dx) from j (see the addressing-
// convention note above: v_shot/scatter_shot are shot-local pointers)
#define V(dz, dy, dx) v_shot[ND_INDEX(j, dz, dy, dx)]
// v * dt^2 at offset
#define VDT2(dz, dy, dx) V(dz, dy, dx) * dt2
// v^2 * dt^2 at offset
#define V2DT2(dz, dy, dx) V(dz, dy, dx) * V(dz, dy, dx) * dt2
// Scattering potential at offset
#define SCATTER(dz, dy, dx) scatter_shot[ND_INDEX(j, dz, dy, dx)]
// Second derivative term in the backward background wavefield update
#define V2DT2_WFC(dz, dy, dx)            \
  (V2DT2(dz, dy, dx) * WFC(dz, dy, dx) + \
   2 * VDT2(dz, dy, dx) * SCATTER(dz, dy, dx) * WFCSC(dz, dy, dx))
// Second derivative term in the backward scattered wavefield update
#define V2DT2_WFCSC(dz, dy, dx) V2DT2(dz, dy, dx) * WFCSC(dz, dy, dx)

#if DW_NDIM >= 3
#define UT_TERMZ1(dz, dy, dx)                                \
  (dbzdz[z + dz] * ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + \
                    bz[z + dz] * ZETAZ(dz, 0, 0)) +          \
   bz[z + dz] * PSIZ(dz, 0, 0))
#define UT_TERMZ2(dz, dy, dx) \
  ((1 + bz[z + dz]) *         \
   ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + bz[z + dz] * ZETAZ(dz, 0, 0)))
#define PSIZ_TERM(dz, dy, dx) \
  ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + bz[z + dz] * ZETAZ(dz, 0, 0))
#define UTSC_TERMZ1(dz, dy, dx)                                \
  (dbzdz[z + dz] * ((1 + bz[z + dz]) * V2DT2_WFCSC(dz, 0, 0) + \
                    bz[z + dz] * ZETAZSC(dz, 0, 0)) +          \
   bz[z + dz] * PSIZSC(dz, 0, 0))
#define UTSC_TERMZ2(dz, dy, dx)                                   \
  ((1 + bz[z + dz]) * ((1 + bz[z + dz]) * V2DT2_WFCSC(dz, 0, 0) + \
                       bz[z + dz] * ZETAZSC(dz, 0, 0)))
#define PSIZSC_TERM(dz, dy, dx) \
  ((1 + bz[z + dz]) * V2DT2_WFCSC(dz, 0, 0) + bz[z + dz] * ZETAZSC(dz, 0, 0))
#endif

#if DW_NDIM >= 2
#define UT_TERMY1(dz, dy, dx)                                \
  (dbydy[y + dy] * ((1 + by[y + dy]) * V2DT2_WFC(0, dy, 0) + \
                    by[y + dy] * ZETAY(0, dy, 0)) +          \
   by[y + dy] * PSIY(0, dy, 0))
#define UT_TERMY2(dz, dy, dx) \
  ((1 + by[y + dy]) *         \
   ((1 + by[y + dy]) * V2DT2_WFC(0, dy, 0) + by[y + dy] * ZETAY(0, dy, 0)))
#define PSIY_TERM(dz, dy, dx) \
  ((1 + by[y + dy]) * V2DT2_WFC(0, dy, 0) + by[y + dy] * ZETAY(0, dy, 0))
#define UTSC_TERMY1(dz, dy, dx)                                \
  (dbydy[y + dy] * ((1 + by[y + dy]) * V2DT2_WFCSC(0, dy, 0) + \
                    by[y + dy] * ZETAYSC(0, dy, 0)) +          \
   by[y + dy] * PSIYSC(0, dy, 0))
#define UTSC_TERMY2(dz, dy, dx)                                   \
  ((1 + by[y + dy]) * ((1 + by[y + dy]) * V2DT2_WFCSC(0, dy, 0) + \
                       by[y + dy] * ZETAYSC(0, dy, 0)))
#define PSIYSC_TERM(dz, dy, dx) \
  ((1 + by[y + dy]) * V2DT2_WFCSC(0, dy, 0) + by[y + dy] * ZETAYSC(0, dy, 0))
#endif

#define UT_TERMX1(dz, dy, dx)                                \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) * V2DT2_WFC(0, 0, dx) + \
                    bx[x + dx] * ZETAX(0, 0, dx)) +          \
   bx[x + dx] * PSIX(0, 0, dx))
#define UT_TERMX2(dz, dy, dx) \
  ((1 + bx[x + dx]) *         \
   ((1 + bx[x + dx]) * V2DT2_WFC(0, 0, dx) + bx[x + dx] * ZETAX(0, 0, dx)))
#define PSIX_TERM(dz, dy, dx) \
  ((1 + bx[x + dx]) * V2DT2_WFC(0, 0, dx) + bx[x + dx] * ZETAX(0, 0, dx))
#define UTSC_TERMX1(dz, dy, dx)                                \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) * V2DT2_WFCSC(0, 0, dx) + \
                    bx[x + dx] * ZETAXSC(0, 0, dx)) +          \
   bx[x + dx] * PSIXSC(0, 0, dx))
#define UTSC_TERMX2(dz, dy, dx)                                   \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * V2DT2_WFCSC(0, 0, dx) + \
                       bx[x + dx] * ZETAXSC(0, 0, dx)))
#define PSIXSC_TERM(dz, dy, dx) \
  ((1 + bx[x + dx]) * V2DT2_WFCSC(0, 0, dx) + bx[x + dx] * ZETAXSC(0, 0, dx))

/* Combined forward stencil update for one grid point: advances both the
 * background wavefield (wfc/wfp) and the scattered wavefield (wfcsc/wfpsc)
 * for one time step. `i` is the flat index used to address wfc/wfp/wfcsc/
 * wfpsc/w_store/wsc_store/psi/zeta arrays (per-shot-local on the CPU driver,
 * global across all shots on the CUDA driver); `v_shot`/`scatter_shot` are
 * the (already shot-selected) wavespeed/scattering-potential VALUES at this
 * point (read once by the caller -- forward never needs a neighbour offset
 * of either); `pml_z`/`pml_y`/`pml_x` are true when this point is inside
 * that axis's PML region (false in the interior); `store_w`/`store_wsc`
 * select whether to save this step's w_sum/wsc_sum for the backward pass
 * (background: whenever v OR scatter requires a gradient; scattered: only
 * when v requires a gradient, since grad_scatter's derivation only needs
 * w_sum -- see FUNC(backward_cell)/FUNC(backward_cell_sc) below).
 */
DW_HOST_DEVICE void FUNC(forward_cell)(
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
    DW_DTYPE *__restrict const w_store, DW_DTYPE *__restrict const wsc_store,
    DW_DTYPE const v_shot, DW_DTYPE const scatter_shot, DW_DTYPE const dt2,
    bool const store_w, bool const store_wsc, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const psiz,
    DW_DTYPE *__restrict const psizn, DW_DTYPE *__restrict const zetaz,
    DW_DTYPE const *__restrict const psizsc, DW_DTYPE *__restrict const psiznsc,
    DW_DTYPE *__restrict const zetazsc, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const dbzdz,
    DW_DTYPE const rdz, DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const psiy,
    DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE const *__restrict const psiysc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE *__restrict const zetaysc, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const rdy, DW_DTYPE const rdy2,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const psix,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetax,
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psixnsc,
    DW_DTYPE *__restrict const zetaxsc, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbxdx,
    DW_DTYPE const rdx, DW_DTYPE const rdx2) {
  DW_DTYPE w_sum = 0, wsc_sum = 0;
#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ2(WFC);
    wsc_sum += DIFFZ2(WFCSC);
  } else {
    DW_DTYPE dwfcdz = DIFFZ1(WFC);
    DW_DTYPE tmpz =
        ((1 + bz[z]) * DIFFZ2(WFC) + dbzdz[z] * dwfcdz + DIFFZ1(AZ_PSIZ));
    DW_DTYPE dwfcscdz = DIFFZ1(WFCSC);
    DW_DTYPE tmpzsc =
        ((1 + bz[z]) * DIFFZ2(WFCSC) + dbzdz[z] * dwfcscdz + DIFFZ1(AZ_PSIZSC));
    w_sum += (1 + bz[z]) * tmpz + az[z] * zetaz[i];
    wsc_sum += (1 + bz[z]) * tmpzsc + az[z] * zetazsc[i];
    psizn[i] = bz[z] * dwfcdz + az[z] * psiz[i];
    zetaz[i] = bz[z] * tmpz + az[z] * zetaz[i];
    psiznsc[i] = bz[z] * dwfcscdz + az[z] * psizsc[i];
    zetazsc[i] = bz[z] * tmpzsc + az[z] * zetazsc[i];
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY2(WFC);
    wsc_sum += DIFFY2(WFCSC);
  } else {
    DW_DTYPE dwfcdy = DIFFY1(WFC);
    DW_DTYPE tmpy =
        ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
    DW_DTYPE dwfcscdy = DIFFY1(WFCSC);
    DW_DTYPE tmpysc =
        ((1 + by[y]) * DIFFY2(WFCSC) + dbydy[y] * dwfcscdy + DIFFY1(AY_PSIYSC));
    w_sum += (1 + by[y]) * tmpy + ay[y] * zetay[i];
    wsc_sum += (1 + by[y]) * tmpysc + ay[y] * zetaysc[i];
    psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];
    zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
    psiynsc[i] = by[y] * dwfcscdy + ay[y] * psiysc[i];
    zetaysc[i] = by[y] * tmpysc + ay[y] * zetaysc[i];
  }
#endif
  if (!pml_x) {
    w_sum += DIFFX2(WFC);
    wsc_sum += DIFFX2(WFCSC);
  } else {
    DW_DTYPE dwfcdx = DIFFX1(WFC);
    DW_DTYPE tmpx =
        ((1 + bx[x]) * DIFFX2(WFC) + dbxdx[x] * dwfcdx + DIFFX1(AX_PSIX));
    DW_DTYPE dwfcscdx = DIFFX1(WFCSC);
    DW_DTYPE tmpxsc =
        ((1 + bx[x]) * DIFFX2(WFCSC) + dbxdx[x] * dwfcscdx + DIFFX1(AX_PSIXSC));
    w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];
    wsc_sum += (1 + bx[x]) * tmpxsc + ax[x] * zetaxsc[i];
    psixn[i] = bx[x] * dwfcdx + ax[x] * psix[i];
    zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];
    psixnsc[i] = bx[x] * dwfcscdx + ax[x] * psixsc[i];
    zetaxsc[i] = bx[x] * tmpxsc + ax[x] * zetaxsc[i];
  }
  wfp[i] = v_shot * v_shot * dt2 * w_sum + 2 * wfc[i] - wfp[i];
  wfpsc[i] = v_shot * v_shot * dt2 * wsc_sum + 2 * wfcsc[i] - wfpsc[i] +
             2 * v_shot * scatter_shot * dt2 * w_sum;
  if (store_w) {
    w_store[i] = w_sum;
  }
  if (store_wsc) {
    wsc_store[i] = wsc_sum;
  }
}

/* Combined backward (adjoint) stencil update for one grid point: advances
 * both adjoint fields and accumulates both grad_v and grad_scatter. `i`
 * addresses wfc/wfp/wfcsc/wfpsc/w_store/wsc_store/grad_v/grad_scatter/psi/
 * zeta/zetan arrays (per-shot-local on the CPU driver, global on the CUDA
 * driver); `j` addresses the already-shot-sliced v_shot/scatter_shot
 * (equal to `i` on the CPU driver; the local per-shot grid index on the
 * CUDA driver, distinct from the global `i`) -- see the addressing-
 * convention note at the top of this file.
 *
 * Per-axis statement order (accumulate w_sum/wsc_sum for this axis and, in
 * the PML region, immediately write that axis's psi*n/zeta* -- for both
 * fields -- before moving to the next axis; write wfp/wfpsc last) follows
 * scalar_born.c's original CPU order exactly, so this refactor changes zero
 * CPU statements. scalar_born.cu's original order differed (accumulate all
 * three axes' w_sum/wsc_sum first, write wfp/wfpsc, THEN go back and write
 * psi*n/zeta* per axis) but computes bitwise-identical values, by the same
 * reasoning already used and verified for scalar.cu in scalar_kernels.h's
 * FUNC(backward_cell): none of the deferred psi/zeta writes are read by
 * anything computed earlier in the function, so adopting the CPU order
 * changes only CUDA's (unverifiable on this box) instruction schedule, not
 * either platform's results.
 */
DW_HOST_DEVICE void FUNC(backward_cell)(
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
    DW_DTYPE const *__restrict const v_shot,
    DW_DTYPE const *__restrict const scatter_shot,
    DW_DTYPE const *__restrict const w_store,
    DW_DTYPE const *__restrict const wsc_store,
    DW_DTYPE *__restrict const grad_v, DW_DTYPE *__restrict const grad_scatter,
    DW_DTYPE const dt2, int64_t const step_ratio, bool const v_requires_grad,
    bool const scatter_requires_grad, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const psiz,
    DW_DTYPE *__restrict const psizn, DW_DTYPE const *__restrict const zetaz,
    DW_DTYPE *__restrict const zetazn, DW_DTYPE const *__restrict const psizsc,
    DW_DTYPE *__restrict const psiznsc,
    DW_DTYPE const *__restrict const zetazsc,
    DW_DTYPE *__restrict const zetaznsc, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const dbzdz,
    DW_DTYPE const rdz, DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const psiy,
    DW_DTYPE *__restrict const psiyn, DW_DTYPE const *__restrict const zetay,
    DW_DTYPE *__restrict const zetayn, DW_DTYPE const *__restrict const psiysc,
    DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE const *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaynsc, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const rdy, DW_DTYPE const rdy2,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const psix,
    DW_DTYPE *__restrict const psixn, DW_DTYPE const *__restrict const zetax,
    DW_DTYPE *__restrict const zetaxn, DW_DTYPE const *__restrict const psixsc,
    DW_DTYPE *__restrict const psixnsc,
    DW_DTYPE const *__restrict const zetaxsc,
    DW_DTYPE *__restrict const zetaxnsc, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbxdx,
    DW_DTYPE const rdx, DW_DTYPE const rdx2) {
  DW_DTYPE w_sum = 0, wsc_sum = 0;
#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ2(V2DT2_WFC);
    wsc_sum += DIFFZ2(V2DT2_WFCSC);
  } else {
    w_sum += -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2);
    wsc_sum += -DIFFZ1(UTSC_TERMZ1) + DIFFZ2(UTSC_TERMZ2);
    psiznsc[i] = -az[z] * DIFFZ1(PSIZSC_TERM) + az[z] * psizsc[i];
    zetaznsc[i] = az[z] * V2DT2(0, 0, 0) * wfcsc[i] + az[z] * zetazsc[i];
    psizn[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz[i];
    zetazn[i] = az[z] * V2DT2(0, 0, 0) * wfc[i] +
                az[z] * 2 * VDT2(0, 0, 0) * SCATTER(0, 0, 0) * wfcsc[i] +
                az[z] * zetaz[i];
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY2(V2DT2_WFC);
    wsc_sum += DIFFY2(V2DT2_WFCSC);
  } else {
    w_sum += -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2);
    wsc_sum += -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2);
    psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];
    zetaynsc[i] = ay[y] * V2DT2(0, 0, 0) * wfcsc[i] + ay[y] * zetaysc[i];
    psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
    zetayn[i] = ay[y] * V2DT2(0, 0, 0) * wfc[i] +
                ay[y] * 2 * VDT2(0, 0, 0) * SCATTER(0, 0, 0) * wfcsc[i] +
                ay[y] * zetay[i];
  }
#endif
  if (!pml_x) {
    w_sum += DIFFX2(V2DT2_WFC);
    wsc_sum += DIFFX2(V2DT2_WFCSC);
  } else {
    w_sum += -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2);
    wsc_sum += -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2);
    psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];
    zetaxnsc[i] = ax[x] * V2DT2(0, 0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];
    psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
    zetaxn[i] = ax[x] * V2DT2(0, 0, 0) * wfc[i] +
                ax[x] * 2 * VDT2(0, 0, 0) * SCATTER(0, 0, 0) * wfcsc[i] +
                ax[x] * zetax[i];
  }
  wfp[i] = w_sum + 2 * wfc[i] - wfp[i];
  wfpsc[i] = wsc_sum + 2 * wfcsc[i] - wfpsc[i];
  if (v_requires_grad) {
    grad_v[i] +=
        wfc[i] * 2 * v_shot[j] * dt2 * w_store[i] * (DW_DTYPE)step_ratio +
        wfcsc[i] *
            (2 * dt2 * scatter_shot[j] * w_store[i] +
             2 * v_shot[j] * dt2 * wsc_store[i]) *
            (DW_DTYPE)step_ratio;
  }
  if (scatter_requires_grad) {
    grad_scatter[i] +=
        wfcsc[i] * 2 * v_shot[j] * dt2 * w_store[i] * (DW_DTYPE)step_ratio;
  }
}

/* Scattered-field-only backward (adjoint) stencil update for one grid
 * point: used instead of FUNC(backward_cell) when v does not require a
 * gradient (only grad_scatter is needed), so the background adjoint field
 * need not be propagated at all. `i`/`j` addressing matches
 * FUNC(backward_cell) above. Per-axis statement order again follows
 * scalar_born.c's original CPU order (see FUNC(backward_cell)'s comment).
 */
DW_HOST_DEVICE void FUNC(backward_cell_sc)(
    DW_DTYPE const *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
    DW_DTYPE const *__restrict const v_shot,
    DW_DTYPE const *__restrict const w_store,
    DW_DTYPE *__restrict const grad_scatter, DW_DTYPE const dt2,
    int64_t const step_ratio, bool const scatter_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const psizsc,
    DW_DTYPE *__restrict const psiznsc,
    DW_DTYPE const *__restrict const zetazsc,
    DW_DTYPE *__restrict const zetaznsc, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const dbzdz,
    DW_DTYPE const rdz, DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const psiysc,
    DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE const *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaynsc, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const rdy, DW_DTYPE const rdy2,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const *__restrict const psixsc,
    DW_DTYPE *__restrict const psixnsc,
    DW_DTYPE const *__restrict const zetaxsc,
    DW_DTYPE *__restrict const zetaxnsc, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbxdx,
    DW_DTYPE const rdx, DW_DTYPE const rdx2) {
  DW_DTYPE wsc_sum = 0;
#if DW_NDIM >= 3
  if (!pml_z) {
    wsc_sum += DIFFZ2(V2DT2_WFCSC);
  } else {
    wsc_sum += -DIFFZ1(UTSC_TERMZ1) + DIFFZ2(UTSC_TERMZ2);
    psiznsc[i] = -az[z] * DIFFZ1(PSIZSC_TERM) + az[z] * psizsc[i];
    zetaznsc[i] = az[z] * V2DT2(0, 0, 0) * wfcsc[i] + az[z] * zetazsc[i];
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    wsc_sum += DIFFY2(V2DT2_WFCSC);
  } else {
    wsc_sum += -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2);
    psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];
    zetaynsc[i] = ay[y] * V2DT2(0, 0, 0) * wfcsc[i] + ay[y] * zetaysc[i];
  }
#endif
  if (!pml_x) {
    wsc_sum += DIFFX2(V2DT2_WFCSC);
  } else {
    wsc_sum += -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2);
    psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];
    zetaxnsc[i] = ax[x] * V2DT2(0, 0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];
  }
  wfpsc[i] = wsc_sum + 2 * wfcsc[i] - wfpsc[i];
  if (scatter_requires_grad) {
    grad_scatter[i] +=
        wfcsc[i] * 2 * v_shot[j] * dt2 * w_store[i] * (DW_DTYPE)step_ratio;
  }
}

#endif /* DW_SCALAR_BORN_KERNELS_H */

/*
 * Scalar Born wave equation propagator
 */

/*
 * This file contains the C implementation of the scalar Born wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 1-3.
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * For a description of the method, see the C implementation of the scalar
 * propagator in scalar.c. This file implements the same functionality,
 * but for the scalar Born wave equation. This involves propagating two
 * wavefields simultaneously: the background wavefield and the scattered
 * wavefield. The scattered wavefield has a source term that is
 * proportional to the background wavefield multiplied by the scattering
 * potential.
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "common_cpu.h"
#include "regular_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  scalar_born_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz)*ny * nx + (dy)*nx + (dx))
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy)*nx + (dx))
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#define DIM_ARGS nx
#endif

// Access the background wavefield at offset (dz, dy, dx) from i
#define WFC(dz, dy, dx) wfc_t[ND_INDEX(i, dz, dy, dx)]
// Access the scattered wavefield at offset (dz, dy, dx) from i
#define WFCSC(dz, dy, dx) wfcsc_t[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define PSIZ(dz, dy, dx) psiz_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAZ(dz, dy, dx) zetaz_t[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZ(dz, dy, dx) az[z + (dz)] * PSIZ(dz, dy, dx)
#define PSIZSC(dz, dy, dx) psizsc_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAZSC(dz, dy, dx) zetazsc_t[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZSC(dz, dy, dx) az[z + (dz)] * PSIZSC(dz, dy, dx)
#endif

#if DW_NDIM >= 2
#define PSIY(dz, dy, dx) psiy_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAY(dz, dy, dx) zetay_t[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIY(dz, dy, dx) ay[y + (dy)] * PSIY(dz, dy, dx)
#define PSIYSC(dz, dy, dx) psiysc_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAYSC(dz, dy, dx) zetaysc_t[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIYSC(dz, dy, dx) ay[y + (dy)] * PSIYSC(dz, dy, dx)
#endif

#define PSIX(dz, dy, dx) psix_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAX(dz, dy, dx) zetax_t[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIX(dz, dy, dx) ax[x + (dx)] * PSIX(dz, dy, dx)
#define PSIXSC(dz, dy, dx) psixsc_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAXSC(dz, dy, dx) zetaxsc_t[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIXSC(dz, dy, dx) ax[x + (dx)] * PSIXSC(dz, dy, dx)

// Access velocity at offset (dz, dy, dx) from i
#define V(dz, dy, dx) v_shot[ND_INDEX(i, dz, dy, dx)]
// v * dt^2 at offset
#define VDT2(dz, dy, dx) V(dz, dy, dx) * dt2
// v^2 * dt^2 at offset
#define V2DT2(dz, dy, dx) V(dz, dy, dx) * V(dz, dy, dx) * dt2
// Scattering potential at offset
#define SCATTER(dz, dy, dx) scatter_shot[ND_INDEX(i, dz, dy, dx)]

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

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(forward)(
        DW_DTYPE const *__restrict const v,
        DW_DTYPE const *__restrict const scatter,
        DW_DTYPE const *__restrict const f,
        DW_DTYPE const *__restrict const fsc, DW_DTYPE *__restrict const wfc,
        DW_DTYPE *__restrict const wfp,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psiz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiy,
#endif
        DW_DTYPE *__restrict const psix,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psizn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiyn,
#endif
        DW_DTYPE *__restrict const psixn,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetaz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetay,
#endif
        DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const wfcsc,
        DW_DTYPE *__restrict const wfpsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psizsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiysc,
#endif
        DW_DTYPE *__restrict const psixsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psiznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiynsc,
#endif
        DW_DTYPE *__restrict const psixnsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetazsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetaysc,
#endif
        DW_DTYPE *__restrict const zetaxsc,
        DW_DTYPE *__restrict const w_store_1,
        DW_DTYPE *__restrict const w_store_1b, void *__restrict const w_store_2,
        void *__restrict const w_store_3,
        char const *__restrict const *__restrict const w_filenames_ptr,
        DW_DTYPE *__restrict const wsc_store_1,
        DW_DTYPE *__restrict const wsc_store_1b,
        void *__restrict const wsc_store_2, void *__restrict const wsc_store_3,
        char const *__restrict const *__restrict const wsc_filenames_ptr,
        DW_DTYPE *__restrict const r, DW_DTYPE *__restrict const rsc,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const az,
        DW_DTYPE const *__restrict const bz,
        DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const dbydy,
#endif
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receivers_i,
        int64_t const *__restrict const receiverssc_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy,
#endif
        DW_DTYPE const rdx,
#if DW_NDIM >= 3
        DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy2,
#endif
        DW_DTYPE const rdx2, DW_DTYPE const dt2, int64_t const nt,
        int64_t const n_shots,
#if DW_NDIM >= 3
        int64_t const nz,
#endif
#if DW_NDIM >= 2
        int64_t const ny,
#endif
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot,
        int64_t const n_receiverssc_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const scatter_requires_grad, bool const v_batched,
        bool const scatter_batched, bool const storage_compression,
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
        int64_t const pml_x1, int64_t const n_threads, void *unused) {
  int64_t shot;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
    int64_t const n_grid_points = ny * nx;
#else
    int64_t const n_grid_points = nx;
#endif
    int64_t const shot_i = shot * n_grid_points;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + shot_i : v;
    DW_DTYPE const *__restrict const scatter_shot =
        scatter_batched ? scatter + shot_i : scatter;
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const zetaz_t = zetaz + shot_i;
    DW_DTYPE *__restrict const zetazsc_t = zetazsc + shot_i;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const zetay_t = zetay + shot_i;
    DW_DTYPE *__restrict const zetaysc_t = zetaysc + shot_i;
#endif
    DW_DTYPE *__restrict const zetax_t = zetax + shot_i;
    DW_DTYPE *__restrict const zetaxsc_t = zetaxsc + shot_i;
    int64_t t;

    FILE *fp_w = NULL;
    FILE *fp_wsc = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad || scatter_requires_grad)
        fp_w = fopen(w_filenames_ptr[shot], "ab");
      if (v_requires_grad) fp_wsc = fopen(wsc_filenames_ptr[shot], "ab");
    }

    for (t = start_t; t < start_t + nt; ++t) {
      DW_DTYPE *__restrict const wfc_t =
          (((t - start_t) & 1) ? wfp : wfc) + shot_i;
      DW_DTYPE *__restrict const wfp_t =
          (((t - start_t) & 1) ? wfc : wfp) + shot_i;
      DW_DTYPE *__restrict const wfcsc_t =
          (((t - start_t) & 1) ? wfpsc : wfcsc) + shot_i;
      DW_DTYPE *__restrict const wfpsc_t =
          (((t - start_t) & 1) ? wfcsc : wfpsc) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE *__restrict const psiz_t =
          (((t - start_t) & 1) ? psizn : psiz) + shot_i;
      DW_DTYPE *__restrict const psizn_t =
          (((t - start_t) & 1) ? psiz : psizn) + shot_i;
      DW_DTYPE *__restrict const psizsc_t =
          (((t - start_t) & 1) ? psiznsc : psizsc) + shot_i;
      DW_DTYPE *__restrict const psiznsc_t =
          (((t - start_t) & 1) ? psizsc : psiznsc) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE *__restrict const psiy_t =
          (((t - start_t) & 1) ? psiyn : psiy) + shot_i;
      DW_DTYPE *__restrict const psiyn_t =
          (((t - start_t) & 1) ? psiy : psiyn) + shot_i;
      DW_DTYPE *__restrict const psiysc_t =
          (((t - start_t) & 1) ? psiynsc : psiysc) + shot_i;
      DW_DTYPE *__restrict const psiynsc_t =
          (((t - start_t) & 1) ? psiysc : psiynsc) + shot_i;
#endif
      DW_DTYPE *__restrict const psix_t =
          (((t - start_t) & 1) ? psixn : psix) + shot_i;
      DW_DTYPE *__restrict const psixn_t =
          (((t - start_t) & 1) ? psix : psixn) + shot_i;
      DW_DTYPE *__restrict const psixsc_t =
          (((t - start_t) & 1) ? psixnsc : psixsc) + shot_i;
      DW_DTYPE *__restrict const psixnsc_t =
          (((t - start_t) & 1) ? psixsc : psixnsc) + shot_i;

      DW_DTYPE *__restrict const w_store_1_t =
          w_store_1 + shot_i +
          ((storage_mode == STORAGE_DEVICE && !storage_compression)
               ? t / step_ratio * n_shots * n_grid_points
               : 0);
      DW_DTYPE *__restrict const wsc_store_1_t =
          wsc_store_1 + shot_i +
          ((storage_mode == STORAGE_DEVICE && !storage_compression)
               ? t / step_ratio * n_shots * n_grid_points
               : 0);
      void *__restrict const w_store_2_t =
          (uint8_t *)w_store_2 +
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
              (int64_t)shot_bytes_comp;
      void *__restrict const wsc_store_2_t =
          (uint8_t *)wsc_store_2 +
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
              (int64_t)shot_bytes_comp;

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);
      bool const scatter_requires_grad_t =
          scatter_requires_grad && ((t % step_ratio) == 0);
      bool const v_or_scatter_requires_grad_t =
          v_requires_grad_t || scatter_requires_grad_t;
#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = (pml_z == 0   ? FD_PAD
                      : pml_z == 1 ? pml_z0
                                   : pml_z1);
                 z < (pml_z == 0   ? pml_z0
                      : pml_z == 1 ? pml_z1
                                   : nz - FD_PAD);
                 z++)
#endif
#if DW_NDIM >= 2
              for (y = (pml_y == 0   ? FD_PAD
                        : pml_y == 1 ? pml_y0
                                     : pml_y1);
                   y < (pml_y == 0   ? pml_y0
                        : pml_y == 1 ? pml_y1
                                     : ny - FD_PAD);
                   y++)
#endif
                for (x = (pml_x == 0   ? FD_PAD
                          : pml_x == 1 ? pml_x0
                                       : pml_x1);
                     x < (pml_x == 0   ? pml_x0
                          : pml_x == 1 ? pml_x1
                                       : nx - FD_PAD);
                     x++) {
#if DW_NDIM == 3
                  int64_t const i = z * ny * nx + y * nx + x;
#elif DW_NDIM == 2
          int64_t const i = y * nx + x;
#else
          int64_t const i = x;
#endif
                  DW_DTYPE w_sum = 0, wsc_sum = 0;
#if DW_NDIM >= 3
                  if (pml_z == 1) {
                    w_sum += DIFFZ2(WFC);
                    wsc_sum += DIFFZ2(WFCSC);
                  } else {
                    DW_DTYPE dwfcdz = DIFFZ1(WFC);
                    DW_DTYPE tmpz = ((1 + bz[z]) * DIFFZ2(WFC) +
                                     dbzdz[z] * dwfcdz + DIFFZ1(AZ_PSIZ));
                    DW_DTYPE dwfcscdz = DIFFZ1(WFCSC);
                    DW_DTYPE tmpzsc = ((1 + bz[z]) * DIFFZ2(WFCSC) +
                                       dbzdz[z] * dwfcscdz + DIFFZ1(AZ_PSIZSC));
                    w_sum += (1 + bz[z]) * tmpz + az[z] * zetaz_t[i];
                    wsc_sum += (1 + bz[z]) * tmpzsc + az[z] * zetazsc_t[i];
                    psizn_t[i] = bz[z] * dwfcdz + az[z] * psiz_t[i];
                    zetaz_t[i] = bz[z] * tmpz + az[z] * zetaz_t[i];
                    psiznsc_t[i] = bz[z] * dwfcscdz + az[z] * psizsc_t[i];
                    zetazsc_t[i] = bz[z] * tmpzsc + az[z] * zetazsc_t[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y == 1) {
                    w_sum += DIFFY2(WFC);
                    wsc_sum += DIFFY2(WFCSC);
                  } else {
                    DW_DTYPE dwfcdy = DIFFY1(WFC);
                    DW_DTYPE tmpy = ((1 + by[y]) * DIFFY2(WFC) +
                                     dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
                    DW_DTYPE dwfcscdy = DIFFY1(WFCSC);
                    DW_DTYPE tmpysc = ((1 + by[y]) * DIFFY2(WFCSC) +
                                       dbydy[y] * dwfcscdy + DIFFY1(AY_PSIYSC));
                    w_sum += (1 + by[y]) * tmpy + ay[y] * zetay_t[i];
                    wsc_sum += (1 + by[y]) * tmpysc + ay[y] * zetaysc_t[i];
                    psiyn_t[i] = by[y] * dwfcdy + ay[y] * psiy_t[i];
                    zetay_t[i] = by[y] * tmpy + ay[y] * zetay_t[i];
                    psiynsc_t[i] = by[y] * dwfcscdy + ay[y] * psiysc_t[i];
                    zetaysc_t[i] = by[y] * tmpysc + ay[y] * zetaysc_t[i];
                  }
#endif
                  if (pml_x == 1) {
                    w_sum += DIFFX2(WFC);
                    wsc_sum += DIFFX2(WFCSC);
                  } else {
                    DW_DTYPE dwfcdx = DIFFX1(WFC);
                    DW_DTYPE tmpx = ((1 + bx[x]) * DIFFX2(WFC) +
                                     dbxdx[x] * dwfcdx + DIFFX1(AX_PSIX));
                    DW_DTYPE dwfcscdx = DIFFX1(WFCSC);
                    DW_DTYPE tmpxsc = ((1 + bx[x]) * DIFFX2(WFCSC) +
                                       dbxdx[x] * dwfcscdx + DIFFX1(AX_PSIXSC));
                    w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax_t[i];
                    wsc_sum += (1 + bx[x]) * tmpxsc + ax[x] * zetaxsc_t[i];
                    psixn_t[i] = bx[x] * dwfcdx + ax[x] * psix_t[i];
                    zetax_t[i] = bx[x] * tmpx + ax[x] * zetax_t[i];
                    psixnsc_t[i] = bx[x] * dwfcscdx + ax[x] * psixsc_t[i];
                    zetaxsc_t[i] = bx[x] * tmpxsc + ax[x] * zetaxsc_t[i];
                  }
                  wfp_t[i] = v_shot[i] * v_shot[i] * dt2 * w_sum +
                             2 * wfc_t[i] - wfp_t[i];
                  wfpsc_t[i] = v_shot[i] * v_shot[i] * dt2 * wsc_sum +
                               2 * wfcsc_t[i] - wfpsc_t[i] +
                               2 * v_shot[i] * scatter_shot[i] * dt2 * w_sum;
                  if (v_or_scatter_requires_grad_t) {
                    w_store_1_t[i] = w_sum;
                  }
                  if (v_requires_grad_t) {
                    wsc_store_1_t[i] = wsc_sum;
                  }
                }
          }

      // Save Snapshots
      if (v_or_scatter_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(save_snapshot_cpu)(
            w_store_1_t, w_store_2_t, fp_w, storage_mode, storage_compression,
            step_idx, shot_bytes_uncomp, shot_bytes_comp, DIM_ARGS);
      }
      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(save_snapshot_cpu)(
            wsc_store_1_t, wsc_store_2_t, fp_wsc, storage_mode,
            storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
            DIM_ARGS);
      }

      add_to_wavefield(wfp_t, sources_i + si,
                       f + t * n_shots * n_sources_per_shot + si,
                       n_sources_per_shot);
      add_to_wavefield(wfpsc_t, sources_i + si,
                       fsc + t * n_shots * n_sources_per_shot + si,
                       n_sources_per_shot);
      record_from_wavefield(wfc_t, receivers_i + ri,
                            r + t * n_shots * n_receivers_per_shot + ri,
                            n_receivers_per_shot);
      record_from_wavefield(wfcsc_t, receiverssc_i + risc,
                            rsc + t * n_shots * n_receiverssc_per_shot + risc,
                            n_receiverssc_per_shot);
    }

    if (fp_w) fclose(fp_w);
    if (fp_wsc) fclose(fp_wsc);
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(backward)(
        DW_DTYPE const *__restrict const v,
        DW_DTYPE const *__restrict const scatter,
        DW_DTYPE const *__restrict const grad_r,
        DW_DTYPE const *__restrict const grad_rsc,
        DW_DTYPE *__restrict const wfc, DW_DTYPE *__restrict const wfp,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psiz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiy,
#endif
        DW_DTYPE *__restrict const psix,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psizn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiyn,
#endif
        DW_DTYPE *__restrict const psixn,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetaz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetay,
#endif
        DW_DTYPE *__restrict const zetax,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetazn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetayn,
#endif
        DW_DTYPE *__restrict const zetaxn, DW_DTYPE *__restrict const wfcsc,
        DW_DTYPE *__restrict const wfpsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psizsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiysc,
#endif
        DW_DTYPE *__restrict const psixsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psiznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiynsc,
#endif
        DW_DTYPE *__restrict const psixnsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetazsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetaysc,
#endif
        DW_DTYPE *__restrict const zetaxsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetaynsc,
#endif
        DW_DTYPE *__restrict const zetaxnsc,
        DW_DTYPE *__restrict const w_store_1,
        DW_DTYPE *__restrict const w_store_1b, void *__restrict const w_store_2,
        void *__restrict const w_store_3,
        char const *__restrict const *__restrict const w_filenames_ptr,
        DW_DTYPE *__restrict const wsc_store_1,
        DW_DTYPE *__restrict const wsc_store_1b,
        void *__restrict const wsc_store_2, void *__restrict const wsc_store_3,
        char const *__restrict const *__restrict const wsc_filenames_ptr,
        DW_DTYPE *__restrict const grad_f, DW_DTYPE *__restrict const grad_fsc,
        DW_DTYPE *__restrict const grad_v,
        DW_DTYPE *__restrict const grad_scatter,
        DW_DTYPE *__restrict const grad_v_thread,
        DW_DTYPE *__restrict const grad_scatter_thread,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const az,
        DW_DTYPE const *__restrict const bz,
        DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const dbydy,
#endif
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receivers_i,
        int64_t const *__restrict const receiverssc_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy,
#endif
        DW_DTYPE const rdx,
#if DW_NDIM >= 3
        DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy2,
#endif
        DW_DTYPE const rdx2, DW_DTYPE const dt2, int64_t const nt,
        int64_t const n_shots,
#if DW_NDIM >= 3
        int64_t const nz,
#endif
#if DW_NDIM >= 2
        int64_t const ny,
#endif
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_sourcessc_per_shot, int64_t const n_receivers_per_shot,
        int64_t const n_receiverssc_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const scatter_requires_grad, bool const v_batched,
        bool const scatter_batched, bool const storage_compression,
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
        int64_t const pml_x1, int64_t const n_threads, void *unused) {
#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
#else
  int64_t const n_grid_points = nx;
#endif
  int64_t shot;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const shot_i = shot * n_grid_points;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const sisc = shot * n_sourcessc_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * n_grid_points;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    DW_DTYPE *__restrict const grad_v_shot =
        grad_v_thread + (v_batched ? shot_i : threadi);
    DW_DTYPE *__restrict const grad_scatter_shot =
        grad_scatter_thread + (scatter_batched ? shot_i : threadi);
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + shot_i : v;
    DW_DTYPE const *__restrict const scatter_shot =
        scatter_batched ? scatter + shot_i : scatter;
    int64_t t;

    FILE *fp_w = NULL;
    FILE *fp_wsc = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad || scatter_requires_grad)
        fp_w = fopen(w_filenames_ptr[shot], "rb");
      if (v_requires_grad) fp_wsc = fopen(wsc_filenames_ptr[shot], "rb");
    }

    for (t = start_t - 1; t >= start_t - nt; --t) {
      DW_DTYPE *__restrict const wfc_t =
          (((start_t - 1 - t) & 1) ? wfp : wfc) + shot_i;
      DW_DTYPE *__restrict const wfp_t =
          (((start_t - 1 - t) & 1) ? wfc : wfp) + shot_i;
      DW_DTYPE *__restrict const wfcsc_t =
          (((start_t - 1 - t) & 1) ? wfpsc : wfcsc) + shot_i;
      DW_DTYPE *__restrict const wfpsc_t =
          (((start_t - 1 - t) & 1) ? wfcsc : wfpsc) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE *__restrict const psiz_t =
          (((start_t - 1 - t) & 1) ? psizn : psiz) + shot_i;
      DW_DTYPE *__restrict const psizn_t =
          (((start_t - 1 - t) & 1) ? psiz : psizn) + shot_i;
      DW_DTYPE *__restrict const zetaz_t =
          (((start_t - 1 - t) & 1) ? zetazn : zetaz) + shot_i;
      DW_DTYPE *__restrict const zetazn_t =
          (((start_t - 1 - t) & 1) ? zetaz : zetazn) + shot_i;
      DW_DTYPE *__restrict const psizsc_t =
          (((start_t - 1 - t) & 1) ? psiznsc : psizsc) + shot_i;
      DW_DTYPE *__restrict const psiznsc_t =
          (((start_t - 1 - t) & 1) ? psizsc : psiznsc) + shot_i;
      DW_DTYPE *__restrict const zetazsc_t =
          (((start_t - 1 - t) & 1) ? zetaznsc : zetazsc) + shot_i;
      DW_DTYPE *__restrict const zetaznsc_t =
          (((start_t - 1 - t) & 1) ? zetazsc : zetaznsc) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE *__restrict const psiy_t =
          (((start_t - 1 - t) & 1) ? psiyn : psiy) + shot_i;
      DW_DTYPE *__restrict const psiyn_t =
          (((start_t - 1 - t) & 1) ? psiy : psiyn) + shot_i;
      DW_DTYPE *__restrict const zetay_t =
          (((start_t - 1 - t) & 1) ? zetayn : zetay) + shot_i;
      DW_DTYPE *__restrict const zetayn_t =
          (((start_t - 1 - t) & 1) ? zetay : zetayn) + shot_i;
      DW_DTYPE *__restrict const psiysc_t =
          (((start_t - 1 - t) & 1) ? psiynsc : psiysc) + shot_i;
      DW_DTYPE *__restrict const psiynsc_t =
          (((start_t - 1 - t) & 1) ? psiysc : psiynsc) + shot_i;
      DW_DTYPE *__restrict const zetaysc_t =
          (((start_t - 1 - t) & 1) ? zetaynsc : zetaysc) + shot_i;
      DW_DTYPE *__restrict const zetaynsc_t =
          (((start_t - 1 - t) & 1) ? zetaysc : zetaynsc) + shot_i;
#endif
      DW_DTYPE *__restrict const psix_t =
          (((start_t - 1 - t) & 1) ? psixn : psix) + shot_i;
      DW_DTYPE *__restrict const psixn_t =
          (((start_t - 1 - t) & 1) ? psix : psixn) + shot_i;
      DW_DTYPE *__restrict const zetax_t =
          (((start_t - 1 - t) & 1) ? zetaxn : zetax) + shot_i;
      DW_DTYPE *__restrict const zetaxn_t =
          (((start_t - 1 - t) & 1) ? zetax : zetaxn) + shot_i;
      DW_DTYPE *__restrict const psixsc_t =
          (((start_t - 1 - t) & 1) ? psixnsc : psixsc) + shot_i;
      DW_DTYPE *__restrict const psixnsc_t =
          (((start_t - 1 - t) & 1) ? psixsc : psixnsc) + shot_i;
      DW_DTYPE *__restrict const zetaxsc_t =
          (((start_t - 1 - t) & 1) ? zetaxnsc : zetaxsc) + shot_i;
      DW_DTYPE *__restrict const zetaxnsc_t =
          (((start_t - 1 - t) & 1) ? zetaxsc : zetaxnsc) + shot_i;

#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

      DW_DTYPE *__restrict const w_store_1_t =
          w_store_1 + shot_i +
          ((storage_mode == STORAGE_DEVICE && !storage_compression)
               ? t / step_ratio * n_shots * n_grid_points
               : 0);
      DW_DTYPE *__restrict const wsc_store_1_t =
          wsc_store_1 + shot_i +
          ((storage_mode == STORAGE_DEVICE && !storage_compression)
               ? t / step_ratio * n_shots * n_grid_points
               : 0);
      void *__restrict const w_store_2_t =
          (uint8_t *)w_store_2 +
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
              (int64_t)shot_bytes_comp;
      void *__restrict const wsc_store_2_t =
          (uint8_t *)wsc_store_2 +
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
              (int64_t)shot_bytes_comp;

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);
      bool const scatter_requires_grad_t =
          scatter_requires_grad && ((t % step_ratio) == 0);

      if (v_requires_grad_t || scatter_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(load_snapshot_cpu)(
            w_store_1_t, w_store_2_t, fp_w, storage_mode, storage_compression,
            step_idx, shot_bytes_uncomp, shot_bytes_comp, DIM_ARGS);
      }

      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(load_snapshot_cpu)(
            wsc_store_1_t, wsc_store_2_t, fp_wsc, storage_mode,
            storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
            DIM_ARGS);
      }

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;
#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = (pml_z == 0   ? FD_PAD
                      : pml_z == 1 ? pml_z0
                                   : pml_z1);
                 z < (pml_z == 0   ? pml_z0
                      : pml_z == 1 ? pml_z1
                                   : nz - FD_PAD);
                 z++)
#endif
#if DW_NDIM >= 2
              for (y = (pml_y == 0   ? FD_PAD
                        : pml_y == 1 ? pml_y0
                                     : pml_y1);
                   y < (pml_y == 0   ? pml_y0
                        : pml_y == 1 ? pml_y1
                                     : ny - FD_PAD);
                   y++)
#endif
                for (x = (pml_x == 0   ? FD_PAD
                          : pml_x == 1 ? pml_x0
                                       : pml_x1);
                     x < (pml_x == 0   ? pml_x0
                          : pml_x == 1 ? pml_x1
                                       : nx - FD_PAD);
                     x++) {
#if DW_NDIM == 3
                  int64_t const i = z * ny * nx + y * nx + x;
#elif DW_NDIM == 2
          int64_t const i = y * nx + x;
#else
          int64_t const i = x;
#endif
                  DW_DTYPE w_sum = 0, wsc_sum = 0;
#if DW_NDIM >= 3
                  if (pml_z == 1) {
                    w_sum += DIFFZ2(V2DT2_WFC);
                    wsc_sum += DIFFZ2(V2DT2_WFCSC);
                  } else {
                    w_sum += -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2);
                    wsc_sum += -DIFFZ1(UTSC_TERMZ1) + DIFFZ2(UTSC_TERMZ2);
                    psiznsc_t[i] =
                        -az[z] * DIFFZ1(PSIZSC_TERM) + az[z] * psizsc_t[i];
                    zetaznsc_t[i] = az[z] * V2DT2(0, 0, 0) * wfcsc_t[i] +
                                    az[z] * zetazsc_t[i];
                    psizn_t[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz_t[i];
                    zetazn_t[i] = az[z] * V2DT2(0, 0, 0) * wfc_t[i] +
                                  az[z] * 2 * VDT2(0, 0, 0) * scatter_shot[i] *
                                      wfcsc_t[i] +
                                  az[z] * zetaz_t[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y == 1) {
                    w_sum += DIFFY2(V2DT2_WFC);
                    wsc_sum += DIFFY2(V2DT2_WFCSC);
                  } else {
                    w_sum += -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2);
                    wsc_sum += -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2);
                    psiynsc_t[i] =
                        -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc_t[i];
                    zetaynsc_t[i] = ay[y] * V2DT2(0, 0, 0) * wfcsc_t[i] +
                                    ay[y] * zetaysc_t[i];
                    psiyn_t[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy_t[i];
                    zetayn_t[i] = ay[y] * V2DT2(0, 0, 0) * wfc_t[i] +
                                  ay[y] * 2 * VDT2(0, 0, 0) * scatter_shot[i] *
                                      wfcsc_t[i] +
                                  ay[y] * zetay_t[i];
                  }
#endif
                  if (pml_x == 1) {
                    w_sum += DIFFX2(V2DT2_WFC);
                    wsc_sum += DIFFX2(V2DT2_WFCSC);
                  } else {
                    w_sum += -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2);
                    wsc_sum += -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2);
                    psixnsc_t[i] =
                        -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc_t[i];
                    zetaxnsc_t[i] = ax[x] * V2DT2(0, 0, 0) * wfcsc_t[i] +
                                    ax[x] * zetaxsc_t[i];
                    psixn_t[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix_t[i];
                    zetaxn_t[i] = ax[x] * V2DT2(0, 0, 0) * wfc_t[i] +
                                  ax[x] * 2 * VDT2(0, 0, 0) * scatter_shot[i] *
                                      wfcsc_t[i] +
                                  ax[x] * zetax_t[i];
                  }
                  wfp_t[i] = w_sum + 2 * wfc_t[i] - wfp_t[i];
                  wfpsc_t[i] = wsc_sum + 2 * wfcsc_t[i] - wfpsc_t[i];
                  if (v_requires_grad_t) {
                    grad_v_shot[i] +=
                        wfc_t[i] * 2 * v_shot[i] * dt2 * w_store_1_t[i] *
                            (DW_DTYPE)step_ratio +
                        wfcsc_t[i] *
                            (2 * dt2 * scatter_shot[i] * w_store_1_t[i] +
                             2 * v_shot[i] * dt2 * wsc_store_1_t[i]) *
                            (DW_DTYPE)step_ratio;
                  }
                  if (scatter_requires_grad_t) {
                    grad_scatter_shot[i] += wfcsc_t[i] * 2 * v_shot[i] * dt2 *
                                            w_store_1_t[i] *
                                            (DW_DTYPE)step_ratio;
                  }
                }
          }

      add_to_wavefield(wfp_t, receivers_i + ri,
                       grad_r + t * n_shots * n_receivers_per_shot + ri,
                       n_receivers_per_shot);
      add_to_wavefield(wfpsc_t, receiverssc_i + risc,
                       grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                       n_receiverssc_per_shot);
      record_from_wavefield(wfc_t, sources_i + si,
                            grad_f + t * n_shots * n_sources_per_shot + si,
                            n_sources_per_shot);
      record_from_wavefield(
          wfcsc_t, sources_i + sisc,
          grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
          n_sourcessc_per_shot);
    }

    if (fp_w) fclose(fp_w);
    if (fp_wsc) fclose(fp_wsc);
  }
#ifdef _OPENMP
  if (v_requires_grad && !v_batched && n_threads > 1) {
    combine_grad(grad_v, grad_v_thread, n_threads, n_grid_points);
  }
  if (scatter_requires_grad && !scatter_batched && n_threads > 1) {
    combine_grad(grad_scatter, grad_scatter_thread, n_threads, n_grid_points);
  }
#endif /* _OPENMP */
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(backward_sc)(
        DW_DTYPE const *__restrict const v,
        DW_DTYPE const *__restrict const grad_rsc,
        DW_DTYPE *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psizsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiysc,
#endif
        DW_DTYPE *__restrict const psixsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const psiznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const psiynsc,
#endif
        DW_DTYPE *__restrict const psixnsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetazsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetaysc,
#endif
        DW_DTYPE *__restrict const zetaxsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetaynsc,
#endif
        DW_DTYPE *__restrict const zetaxnsc,
        DW_DTYPE *__restrict const w_store_1,
        DW_DTYPE *__restrict const w_store_1b, void *__restrict const w_store_2,
        void *__restrict const w_store_3,
        char const *__restrict const *__restrict const w_filenames_ptr,
        DW_DTYPE *__restrict const grad_fsc,
        DW_DTYPE *__restrict const grad_scatter,
        DW_DTYPE *__restrict const grad_scatter_thread,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const az,
        DW_DTYPE const *__restrict const bz,
        DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const dbydy,
#endif
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receiverssc_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy,
#endif
        DW_DTYPE const rdx,
#if DW_NDIM >= 3
        DW_DTYPE const rdz2,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy2,
#endif
        DW_DTYPE const rdx2, DW_DTYPE const dt2, int64_t const nt,
        int64_t const n_shots,
#if DW_NDIM >= 3
        int64_t const nz,
#endif
#if DW_NDIM >= 2
        int64_t const ny,
#endif
        int64_t const nx, int64_t const n_sourcessc_per_shot,
        int64_t const n_receiverssc_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const scatter_requires_grad,
        bool const v_batched, bool const scatter_batched,
        bool const storage_compression, int64_t const start_t,
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
        int64_t const pml_x1, int64_t const n_threads) {
#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
#else
  int64_t const n_grid_points = nx;
#endif
  int64_t shot;

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const shot_i = shot * n_grid_points;
    int64_t const sisc = shot * n_sourcessc_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * n_grid_points;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    DW_DTYPE *__restrict const grad_scatter_shot =
        grad_scatter_thread + (scatter_batched ? shot_i : threadi);
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + shot_i : v;
    int64_t t;

    FILE *fp_w = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (scatter_requires_grad) fp_w = fopen(w_filenames_ptr[shot], "rb");
    }

    for (t = start_t - 1; t >= start_t - nt; --t) {
      DW_DTYPE *__restrict const wfcsc_t =
          (((start_t - 1 - t) & 1) ? wfpsc : wfcsc) + shot_i;
      DW_DTYPE *__restrict const wfpsc_t =
          (((start_t - 1 - t) & 1) ? wfcsc : wfpsc) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE *__restrict const psizsc_t =
          (((start_t - 1 - t) & 1) ? psiznsc : psizsc) + shot_i;
      DW_DTYPE *__restrict const psiznsc_t =
          (((start_t - 1 - t) & 1) ? psizsc : psiznsc) + shot_i;
      DW_DTYPE *__restrict const zetazsc_t =
          (((start_t - 1 - t) & 1) ? zetaznsc : zetazsc) + shot_i;
      DW_DTYPE *__restrict const zetaznsc_t =
          (((start_t - 1 - t) & 1) ? zetazsc : zetaznsc) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE *__restrict const psiysc_t =
          (((start_t - 1 - t) & 1) ? psiynsc : psiysc) + shot_i;
      DW_DTYPE *__restrict const psiynsc_t =
          (((start_t - 1 - t) & 1) ? psiysc : psiynsc) + shot_i;
      DW_DTYPE *__restrict const zetaysc_t =
          (((start_t - 1 - t) & 1) ? zetaynsc : zetaysc) + shot_i;
      DW_DTYPE *__restrict const zetaynsc_t =
          (((start_t - 1 - t) & 1) ? zetaysc : zetaynsc) + shot_i;
#endif
      DW_DTYPE *__restrict const psixsc_t =
          (((start_t - 1 - t) & 1) ? psixnsc : psixsc) + shot_i;
      DW_DTYPE *__restrict const psixnsc_t =
          (((start_t - 1 - t) & 1) ? psixsc : psixnsc) + shot_i;
      DW_DTYPE *__restrict const zetaxsc_t =
          (((start_t - 1 - t) & 1) ? zetaxnsc : zetaxsc) + shot_i;
      DW_DTYPE *__restrict const zetaxnsc_t =
          (((start_t - 1 - t) & 1) ? zetaxsc : zetaxnsc) + shot_i;

#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

      DW_DTYPE *__restrict const w_store_1_t =
          w_store_1 + shot_i +
          ((storage_mode == STORAGE_DEVICE && !storage_compression)
               ? t / step_ratio * n_shots * n_grid_points
               : 0);
      void *__restrict const w_store_2_t =
          (uint8_t *)w_store_2 +
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
              (int64_t)shot_bytes_comp;

      bool const scatter_requires_grad_t =
          scatter_requires_grad && ((t % step_ratio) == 0);

      if (scatter_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(load_snapshot_cpu)(
            w_store_1_t, w_store_2_t, fp_w, storage_mode, storage_compression,
            step_idx, shot_bytes_uncomp, shot_bytes_comp, DIM_ARGS);
      }

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;
#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = (pml_z == 0   ? FD_PAD
                      : pml_z == 1 ? pml_z0
                                   : pml_z1);
                 z < (pml_z == 0   ? pml_z0
                      : pml_z == 1 ? pml_z1
                                   : nz - FD_PAD);
                 z++)
#endif
#if DW_NDIM >= 2
              for (y = (pml_y == 0   ? FD_PAD
                        : pml_y == 1 ? pml_y0
                                     : pml_y1);
                   y < (pml_y == 0   ? pml_y0
                        : pml_y == 1 ? pml_y1
                                     : ny - FD_PAD);
                   y++)
#endif
                for (x = (pml_x == 0   ? FD_PAD
                          : pml_x == 1 ? pml_x0
                                       : pml_x1);
                     x < (pml_x == 0   ? pml_x0
                          : pml_x == 1 ? pml_x1
                                       : nx - FD_PAD);
                     x++) {
#if DW_NDIM == 3
                  int64_t const i = z * ny * nx + y * nx + x;
#elif DW_NDIM == 2
          int64_t const i = y * nx + x;
#else
          int64_t const i = x;
#endif
                  DW_DTYPE wsc_sum = 0;
#if DW_NDIM >= 3
                  if (pml_z == 1) {
                    wsc_sum += DIFFZ2(V2DT2_WFCSC);
                  } else {
                    wsc_sum += -DIFFZ1(UTSC_TERMZ1) + DIFFZ2(UTSC_TERMZ2);
                    psiznsc_t[i] =
                        -az[z] * DIFFZ1(PSIZSC_TERM) + az[z] * psizsc_t[i];
                    zetaznsc_t[i] = az[z] * V2DT2(0, 0, 0) * wfcsc_t[i] +
                                    az[z] * zetazsc_t[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y == 1) {
                    wsc_sum += DIFFY2(V2DT2_WFCSC);
                  } else {
                    wsc_sum += -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2);
                    psiynsc_t[i] =
                        -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc_t[i];
                    zetaynsc_t[i] = ay[y] * V2DT2(0, 0, 0) * wfcsc_t[i] +
                                    ay[y] * zetaysc_t[i];
                  }
#endif
                  if (pml_x == 1) {
                    wsc_sum += DIFFX2(V2DT2_WFCSC);
                  } else {
                    wsc_sum += -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2);
                    psixnsc_t[i] =
                        -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc_t[i];
                    zetaxnsc_t[i] = ax[x] * V2DT2(0, 0, 0) * wfcsc_t[i] +
                                    ax[x] * zetaxsc_t[i];
                  }
                  wfpsc_t[i] = wsc_sum + 2 * wfcsc_t[i] - wfpsc_t[i];
                  if (scatter_requires_grad_t) {
                    grad_scatter_shot[i] += wfcsc_t[i] * 2 * v_shot[i] * dt2 *
                                            w_store_1_t[i] *
                                            (DW_DTYPE)step_ratio;
                  }
                }
          }

      add_to_wavefield(wfpsc_t, receiverssc_i + risc,
                       grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                       n_receiverssc_per_shot);
      record_from_wavefield(
          wfcsc_t, sources_i + sisc,
          grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
          n_sourcessc_per_shot);
    }

    if (fp_w) fclose(fp_w);
  }
#ifdef _OPENMP
  if (scatter_requires_grad && !scatter_batched && n_threads > 1) {
    combine_grad(grad_scatter, grad_scatter_thread, n_threads, n_grid_points);
  }
#endif /* _OPENMP */
  return 0;
}

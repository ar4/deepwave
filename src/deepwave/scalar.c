/*
 * Scalar wave equation propagator
 */

/*
 * This file contains the C implementation of the scalar wave equation
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
 * The propagator solves the scalar wave equation:
 * d^2u/dt^2 = v^2 * laplacian(u) + v^2 * f
 * where u is the wavefield, t is time, v is the wavespeed, and f is the
 * source. The Laplacian is applied to the spatial dimensions.
 *
 * The propagator uses a finite difference scheme that is second-order
 * accurate in time and has a user-specifiable order of accuracy in space.
 * To prevent reflections from the boundaries of the computational domain,
 * a Perfectly Matched Layer (PML) is used. The PML implementation is based
 * on the Convolutional PML (C-PML) method of Pasalic and McGarry (2010).
 * This requires the use of auxiliary wavefields (psi and zeta).
 *
 * To improve performance, the computational domain is divided into
 * regions, and only the necessary calculations are performed in each
 * region. The regions are:
 *  * The central region, where no PML calculations are needed.
 *  * Side regions, where PML calculations are needed in one dimension.
 *  * Edge regions, where PML calculations are needed in two dimensions.
 *  * Corner regions, where PML calculations are needed in all dimensions.
 *
 * The code is structured to maximize performance by using macros to
 * generate the code for each of the regions, and by using OpenMP
 * to parallelize the loops over shots.
 */

/*
 * Assumptions:
 *  * The first and last accuracy/2 elements in each spatial dimension
 *    are zero in all wavefields (forward and backward). This is to avoid
 *    the need for special handling of the boundaries, which would prevent
 *    vectorisation.
 *  * Elements of a and b PML profiles are zero except for the first and last
 *    accuracy/2 + pml_width elements.
 *  * Elements of dbdx are zero except for the first and last
 *    accuracy + pml_width elements. These are the spatial derivatives of
 *    the PML profiles.
 *  * Elements of psi and zeta are zero except for the first and last
 *    accuracy + pml_width elements in the corresponding dimension for
 *    forward propagation, and 3 * accuracy / 2 + pml_width elements in the
 *    corresponding dimension for backward propagation. These are the PML
 *    auxiliary/memory wavefields.
 *  * The values in wfp (the wavefield at the previous time step) are
 *    multiplied by -1 before and after calls to the backward propagator.
 *    This is to simplify the backpropagation calculations.
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "common_cpu.h"
#include "regular_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  scalar_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
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

#define WFC(dz, dy, dx) wfc_t[ND_INDEX(i, dz, dy, dx)]
#define V2DT2_WFC(dz, dy, dx) \
  v2dt2_shot[ND_INDEX(i, dz, dy, dx)] * WFC(dz, dy, dx)

#if DW_NDIM >= 3
#define PSIZ(dz, dy, dx) psiz_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAZ(dz, dy, dx) zetaz_t[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZ(dz, dy, dx) az[z + (dz)] * PSIZ(dz, dy, dx)
#endif

#if DW_NDIM >= 2
#define PSIY(dz, dy, dx) psiy_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAY(dz, dy, dx) zetay_t[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIY(dz, dy, dx) ay[y + (dy)] * PSIY(dz, dy, dx)
#endif

#define PSIX(dz, dy, dx) psix_t[ND_INDEX(i, dz, dy, dx)]
#define ZETAX(dz, dy, dx) zetax_t[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIX(dz, dy, dx) ax[x + (dx)] * PSIX(dz, dy, dx)

#if DW_NDIM == 3
#define UT_TERMZ1(dz, dy, dx)                                \
  (dbzdz[z + dz] * ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + \
                    bz[z + dz] * ZETAZ(dz, 0, 0)) +          \
   bz[z + dz] * PSIZ(dz, 0, 0))
#define UT_TERMZ2(dz, dy, dx) \
  ((1 + bz[z + dz]) *         \
   ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + bz[z + dz] * ZETAZ(dz, 0, 0)))
#define PSIZ_TERM(dz, dy, dx) \
  ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + bz[z + dz] * ZETAZ(dz, 0, 0))
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

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(forward)(
        DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const f,
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
        DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const w_store_1,
        DW_DTYPE *__restrict const w_store_1b, void *__restrict const w_store_2,
        void *__restrict const w_store_3,
        char const *__restrict const *__restrict const w_filenames_ptr,
        DW_DTYPE *__restrict const r,
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
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const v_batched, bool const storage_compression,
        int64_t const start_t,
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
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + shot_i : v;
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const zetaz_t = zetaz + shot_i;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const zetay_t = zetay + shot_i;
#endif
    DW_DTYPE *__restrict const zetax_t = zetax + shot_i;
    int64_t t;

    FILE *fp_w = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad) fp_w = fopen(w_filenames_ptr[shot], "ab");
    }

    for (t = start_t; t < start_t + nt; ++t) {
      DW_DTYPE *__restrict const wfc_t =
          (((t - start_t) & 1) ? wfp : wfc) + shot_i;
      DW_DTYPE *__restrict const wfp_t =
          (((t - start_t) & 1) ? wfc : wfp) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE *__restrict const psiz_t =
          (((t - start_t) & 1) ? psizn : psiz) + shot_i;
      DW_DTYPE *__restrict const psizn_t =
          (((t - start_t) & 1) ? psiz : psizn) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE *__restrict const psiy_t =
          (((t - start_t) & 1) ? psiyn : psiy) + shot_i;
      DW_DTYPE *__restrict const psiyn_t =
          (((t - start_t) & 1) ? psiy : psiyn) + shot_i;
#endif
      DW_DTYPE *__restrict const psix_t =
          (((t - start_t) & 1) ? psixn : psix) + shot_i;
      DW_DTYPE *__restrict const psixn_t =
          (((t - start_t) & 1) ? psix : psixn) + shot_i;
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

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);
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
                  DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
                  if (pml_z == 1) {
                    w_sum += DIFFZ2(WFC);
                  } else {
                    DW_DTYPE dwfcdz = DIFFZ1(WFC);
                    DW_DTYPE tmpz = ((1 + bz[z]) * DIFFZ2(WFC) +
                                     dbzdz[z] * dwfcdz + DIFFZ1(AZ_PSIZ));
                    w_sum += (1 + bz[z]) * tmpz + az[z] * zetaz_t[i];
                    psizn_t[i] = bz[z] * dwfcdz + az[z] * psiz_t[i];
                    zetaz_t[i] = bz[z] * tmpz + az[z] * zetaz_t[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y == 1) {
                    w_sum += DIFFY2(WFC);
                  } else {
                    DW_DTYPE dwfcdy = DIFFY1(WFC);
                    DW_DTYPE tmpy = ((1 + by[y]) * DIFFY2(WFC) +
                                     dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
                    w_sum += (1 + by[y]) * tmpy + ay[y] * zetay_t[i];
                    psiyn_t[i] = by[y] * dwfcdy + ay[y] * psiy_t[i];
                    zetay_t[i] = by[y] * tmpy + ay[y] * zetay_t[i];
                  }
#endif
                  if (pml_x == 1) {
                    w_sum += DIFFX2(WFC);
                  } else {
                    DW_DTYPE dwfcdx = DIFFX1(WFC);
                    DW_DTYPE tmpx = ((1 + bx[x]) * DIFFX2(WFC) +
                                     dbxdx[x] * dwfcdx + DIFFX1(AX_PSIX));
                    w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax_t[i];
                    psixn_t[i] = bx[x] * dwfcdx + ax[x] * psix_t[i];
                    zetax_t[i] = bx[x] * tmpx + ax[x] * zetax_t[i];
                  }
                  wfp_t[i] = v_shot[i] * v_shot[i] * dt2 * w_sum +
                             2 * wfc_t[i] - wfp_t[i];
                  if (v_requires_grad_t) {
                    w_store_1_t[i] = 2 * v_shot[i] * dt2 * w_sum;
                  }
                }
          }

      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(save_snapshot_cpu)(
            w_store_1_t, w_store_2_t, fp_w, storage_mode, storage_compression,
            step_idx, shot_bytes_uncomp, shot_bytes_comp, DIM_ARGS);
      }

      add_to_wavefield(wfp_t, sources_i + si,
                       f + t * n_shots * n_sources_per_shot + si,
                       n_sources_per_shot);
      record_from_wavefield(wfc_t, receivers_i + ri,
                            r + t * n_shots * n_receivers_per_shot + ri,
                            n_receivers_per_shot);
    }
    if (fp_w) fclose(fp_w);
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(backward)(
        DW_DTYPE const *__restrict const v2dt2,
        DW_DTYPE const *__restrict const grad_r, DW_DTYPE *__restrict const wfc,
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
        DW_DTYPE *__restrict const zetax,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetazn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetayn,
#endif
        DW_DTYPE *__restrict const zetaxn, DW_DTYPE *__restrict const w_store_1,
        DW_DTYPE *__restrict const w_store_1b, void *__restrict const w_store_2,
        void *__restrict const w_store_3,
        char const *__restrict const *__restrict const w_filenames_ptr,
        DW_DTYPE *__restrict const grad_f, DW_DTYPE *__restrict const grad_v,
        DW_DTYPE *__restrict const grad_v_thread,
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
        DW_DTYPE const rdx2, int64_t const nt, int64_t const n_shots,
#if DW_NDIM >= 3
        int64_t const nz,
#endif
#if DW_NDIM >= 2
        int64_t const ny,
#endif
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const v_batched, bool const storage_compression, int64_t start_t,
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
    int64_t const ri = shot * n_receivers_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * n_grid_points;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    DW_DTYPE *__restrict const grad_v_shot =
        grad_v_thread + (v_batched ? shot_i : threadi);
    DW_DTYPE const *__restrict const v2dt2_shot =
        v_batched ? v2dt2 + shot_i : v2dt2;
    int64_t t;

    FILE *fp_w = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad) fp_w = fopen(w_filenames_ptr[shot], "rb");
    }

    for (t = start_t - 1; t >= start_t - nt; --t) {
      DW_DTYPE *__restrict const wfc_t =
          (((start_t - 1 - t) & 1) ? wfp : wfc) + shot_i;
      DW_DTYPE *__restrict const wfp_t =
          (((start_t - 1 - t) & 1) ? wfc : wfp) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE *__restrict const psiz_t =
          (((start_t - 1 - t) & 1) ? psizn : psiz) + shot_i;
      DW_DTYPE *__restrict const psizn_t =
          (((start_t - 1 - t) & 1) ? psiz : psizn) + shot_i;
      DW_DTYPE *__restrict const zetaz_t =
          (((start_t - 1 - t) & 1) ? zetazn : zetaz) + shot_i;
      DW_DTYPE *__restrict const zetazn_t =
          (((start_t - 1 - t) & 1) ? zetaz : zetazn) + shot_i;
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
#endif
      DW_DTYPE *__restrict const psix_t =
          (((start_t - 1 - t) & 1) ? psixn : psix) + shot_i;
      DW_DTYPE *__restrict const psixn_t =
          (((start_t - 1 - t) & 1) ? psix : psixn) + shot_i;
      DW_DTYPE *__restrict const zetax_t =
          (((start_t - 1 - t) & 1) ? zetaxn : zetax) + shot_i;
      DW_DTYPE *__restrict const zetaxn_t =
          (((start_t - 1 - t) & 1) ? zetax : zetaxn) + shot_i;
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

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);

      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        STORAGE_FUNC(load_snapshot_cpu)(
            w_store_1_t, w_store_2_t, fp_w, storage_mode, storage_compression,
            step_idx, shot_bytes_uncomp, shot_bytes_comp, DIM_ARGS);
      }

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
                  DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
                  if (pml_z == 1) {
                    w_sum += DIFFZ2(V2DT2_WFC);
                  } else {
                    w_sum += -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2);
                    psizn_t[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz_t[i];
                    zetazn_t[i] =
                        az[z] * V2DT2_WFC(0, 0, 0) + az[z] * zetaz_t[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y == 1) {
                    w_sum += DIFFY2(V2DT2_WFC);
                  } else {
                    w_sum += -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2);
                    psiyn_t[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy_t[i];
                    zetayn_t[i] =
                        ay[y] * V2DT2_WFC(0, 0, 0) + ay[y] * zetay_t[i];
                  }
#endif
                  if (pml_x == 1) {
                    w_sum += DIFFX2(V2DT2_WFC);
                  } else {
                    w_sum += -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2);
                    psixn_t[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix_t[i];
                    zetaxn_t[i] =
                        ax[x] * V2DT2_WFC(0, 0, 0) + ax[x] * zetax_t[i];
                  }

                  wfp_t[i] = w_sum + 2 * wfc_t[i] - wfp_t[i];

                  if (v_requires_grad_t) {
                    grad_v_shot[i] +=
                        wfc_t[i] * w_store_1_t[i] * (DW_DTYPE)step_ratio;
                  }
                }
          }

      add_to_wavefield(wfp_t, receivers_i + ri,
                       grad_r + t * n_shots * n_receivers_per_shot + ri,
                       n_receivers_per_shot);
      record_from_wavefield(wfc_t, sources_i + si,
                            grad_f + t * n_shots * n_sources_per_shot + si,
                            n_sources_per_shot);
    }
    if (fp_w) fclose(fp_w);
  }
#ifdef _OPENMP
  if (v_requires_grad && !v_batched && n_threads > 1) {
    combine_grad(grad_v, grad_v_thread, n_threads, n_grid_points);
  }
#endif /* _OPENMP */
  return 0;
}

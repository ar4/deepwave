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
#define SCALAR_BORN_VARIANT_I(name) name
#define SCALAR_BORN_VARIANT(name) SCALAR_BORN_VARIANT_I(name)
#define FUNC(name) \
  CAT(SCALAR_BORN_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

/* ND_INDEX, WFC/WFCSC, PSIZ/ZETAZ/... (+Y, +X) and their *SC scattered-field
 * twins, V/VDT2/V2DT2/SCATTER, V2DT2_WFC/V2DT2_WFCSC, the UT_TERM/UTSC_TERM/
 * PSI*_TERM/PSI*SC_TERM family, and the FUNC(forward_cell)/FUNC(backward_cell)/
 * FUNC(backward_cell_sc) per-grid-point stencil update functions live in
 * scalar_born_kernels.h, shared verbatim with scalar_born.cu (see that
 * header for the full explanation). wfc_t/wfp_t/psiz_t/.../zetaz_t below are
 * this file's names for the (per-shot-offset) arrays passed as the wfc/wfp/
 * psiz/.../zetaz parameters of those functions. */
#include "scalar_born_kernels.h"

/* First-class reflecting/periodic boundary faces (see docs/boundaries.rst and
 * scalar.c, which this mirrors exactly -- scalar_born is a regular-grid nodal
 * propagator like scalar). ``bc`` codes: 0=pml (absorbing, guard left
 * untouched), 1=free (pressure release, field odd about the outer node P-1,
 * R=-1), 2=rigid (field even about the half-cell plane P-1/2, R=+1),
 * 3=periodic (wrap with period n-2P). The nodal-grid interior along an axis of
 * length ``n`` is ``[P, n-P)``.
 *
 * ``fill_bc`` refills a wavefield's guard ring with the exact image every
 * outer step, so the interior stencil reads the correct ghost values. The
 * same fill is applied to the forward and the (adjoint) backward pass, and to
 * BOTH the background and the scattered field: mirroring is linear, so the
 * Born linearisation of the mirrored operator gives the scattered field the
 * same odd/even/wrap images as the background field (validated directly via
 * the R=+-1 analytic reflection-coefficient gate for the scattered receiver,
 * not merely assumed -- see tests/test_boundary.py
 * test_scalar_born_c_{free,rigid}_reflection_coefficient). ``is_model``
 * selects the model variant (unused here: scalar_born's v/scatter guard rings
 * are mirror-padded on fresh Python-side clones before the call, since --
 * unlike scalar's fresh-per-call v2dt2 temporary -- v and scatter are read
 * directly and may alias the user's saved tensors, so an in-place C-side mirror
 * would be unsafe; see mirror_pad_scalar_born_models in boundaries.py). */
/* Fill the guard cells of a single line along one axis (base offset, element
 * stride ``stride``, axis length ``nax``) with the boundary image. */
static inline void FUNC(fill_bc_line)(DW_DTYPE* __restrict const wf,
                                      int64_t const base, int64_t const nax,
                                      int64_t const stride, int64_t const bc_lo,
                                      int64_t const bc_hi, int const is_model) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P; /* interior node count */
  int64_t g;
  if (bc_lo != 0) {
    for (g = 0; g < P; ++g) {
      int64_t src = 0;
      DW_DTYPE sign = 1;
      int zero = 0;
      if (bc_lo == 3) {
        src = g + m; /* periodic wrap */
      } else if (bc_lo == 2) {
        src = 2 * P - 1 - g; /* rigid: even about P-1/2 */
      } else {               /* free: odd about node P-1 */
        if (g == P - 1) {
          if (is_model) continue; /* model even-mirror leaves this node */
          zero = 1;
        } else {
          src = 2 * P - 2 - g;
          if (!is_model) sign = -1;
        }
      }
      wf[base + g * stride] =
          zero ? (DW_DTYPE)0 : sign * wf[base + src * stride];
    }
  }
  if (bc_hi != 0) {
    int64_t j;
    for (j = 0; j < P; ++j) {
      int64_t const x = nax - P + j;
      int64_t src = 0;
      DW_DTYPE sign = 1;
      int zero = 0;
      if (bc_hi == 3) {
        src = P + j; /* periodic wrap */
      } else if (bc_hi == 2) {
        src = nax - P - 1 - j; /* rigid: even about (n-P)-1/2 */
      } else {                 /* free: odd about node n-P */
        if (j == 0) {
          if (is_model) continue;
          zero = 1;
        } else {
          src = nax - P - j;
          if (!is_model) sign = -1;
        }
      }
      wf[base + x * stride] =
          zero ? (DW_DTYPE)0 : sign * wf[base + src * stride];
    }
  }
}

/* Apply the guard fill on every reflecting axis of one shot's field. ``bc`` is
 * the per-face code array [z0,z1,y0,y1,x0,x1] (only the trailing DW_NDIM pairs
 * are used). ``is_model`` selects the even/wrap (no-sign, no-zero) model
 * variant (unused by scalar_born; kept for signature parity with scalar.c). */
static void FUNC(fill_bc)(DW_DTYPE* __restrict const wf, int const is_model,
                          int64_t const* __restrict const bc, int64_t const nz_,
                          int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != 0 || bc[5] != 0)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_bc_line)(wf, (zz * ny_ + yy) * nx_, nx_, 1, bc[4], bc[5],
                             is_model);
    if (bc[2] != 0 || bc[3] != 0)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_bc_line)(wf, zz * ny_ * nx_ + xx, ny_, nx_, bc[2], bc[3],
                             is_model);
    if (bc[0] != 0 || bc[1] != 0)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_bc_line)(wf, yy * nx_ + xx, nz_, ny_ * nx_, bc[0], bc[1],
                             is_model);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != 0 || bc[5] != 0)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_bc_line)(wf, yy * nx_, nx_, 1, bc[4], bc[5], is_model);
    if (bc[2] != 0 || bc[3] != 0)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_bc_line)(wf, xx, ny_, nx_, bc[2], bc[3], is_model);
  }
#else
  FUNC(fill_bc_line)(wf, 0, nx_, 1, bc[4], bc[5], is_model);
#endif
}

#if DW_NDIM == 3
#define FILL_BC(wf, ismodel) FUNC(fill_bc)(wf, ismodel, bc, nz, ny, nx)
#elif DW_NDIM == 2
#define FILL_BC(wf, ismodel) FUNC(fill_bc)(wf, ismodel, bc, 1, ny, nx)
#else
#define FILL_BC(wf, ismodel) FUNC(fill_bc)(wf, ismodel, bc, 1, 1, nx)
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(forward)(
    DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const scatter,
    DW_DTYPE const* __restrict const f, DW_DTYPE const* __restrict const fsc,
    DW_DTYPE* __restrict const wfc, DW_DTYPE* __restrict const wfp,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psiz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiy,
#endif
    DW_DTYPE* __restrict const psix,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psizn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiyn,
#endif
    DW_DTYPE* __restrict const psixn,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetaz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetay,
#endif
    DW_DTYPE* __restrict const zetax, DW_DTYPE* __restrict const wfcsc,
    DW_DTYPE* __restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psizsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiysc,
#endif
    DW_DTYPE* __restrict const psixsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psiznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiynsc,
#endif
    DW_DTYPE* __restrict const psixnsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetazsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetaysc,
#endif
    DW_DTYPE* __restrict const zetaxsc, DW_DTYPE* __restrict const w_store_1,
    DW_DTYPE* __restrict const w_store_1b, void* __restrict const w_store_2,
    void* __restrict const w_store_3,
    char const* __restrict const* __restrict const w_filenames_ptr,
    DW_DTYPE* __restrict const wsc_store_1,
    DW_DTYPE* __restrict const wsc_store_1b, void* __restrict const wsc_store_2,
    void* __restrict const wsc_store_3,
    char const* __restrict const* __restrict const wsc_filenames_ptr,
    DW_DTYPE* __restrict const r, DW_DTYPE* __restrict const rsc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx,
    int64_t const* __restrict const sources_i,
    int64_t const* __restrict const receivers_i,
    int64_t const* __restrict const receiverssc_i,
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
    int64_t const n_receivers_per_shot, int64_t const n_receiverssc_per_shot,
    int64_t const step_ratio, int64_t const storage_mode,
    size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
    bool const v_requires_grad, bool const scatter_requires_grad,
    bool const v_batched, bool const scatter_batched,
    bool const storage_compression, int64_t start_t,
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
    int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0,
#endif
    int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y1,
#endif
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;

    if ((n_receivers_per_shot > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot > 0) && sources_i == NULL) return 1;
    if (n_receiverssc_per_shot > 0 && receiverssc_i == NULL) return 1;
  }
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  int64_t shot;

  /* Intra-shot spatial parallelism: shots run serially and each time step's
   * combined background+scattered stencil is spread across threads (see the
   * omp parallel region below), so a single shot saturates all cores. */
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
    DW_DTYPE const* __restrict const v_shot = v_batched ? v + shot_i : v;
    DW_DTYPE const* __restrict const scatter_shot =
        scatter_batched ? scatter + shot_i : scatter;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetaz_t = zetaz + shot_i;
    DW_DTYPE* __restrict const zetazsc_t = zetazsc + shot_i;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetay_t = zetay + shot_i;
    DW_DTYPE* __restrict const zetaysc_t = zetaysc + shot_i;
#endif
    DW_DTYPE* __restrict const zetax_t = zetax + shot_i;
    DW_DTYPE* __restrict const zetaxsc_t = zetaxsc + shot_i;
    int64_t t;

    int storage_err = 0;
    FILE* fp_w = NULL;
    FILE* fp_wsc = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad || scatter_requires_grad) {
        fp_w = dw_open_file_utf8(w_filenames_ptr[shot], "ab");
        if (fp_w == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
      if (v_requires_grad) {
        fp_wsc = dw_open_file_utf8(wsc_filenames_ptr[shot], "ab");
        if (fp_wsc == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    for (t = start_t; t < start_t + nt; ++t) {
      if ((t % step_ratio) == 0 && (v_requires_grad || scatter_requires_grad) &&
          (w_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      if ((t % step_ratio) == 0 && (v_requires_grad) && (wsc_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      DW_DTYPE* __restrict const wfc_t =
          (((uint64_t)(t - start_t) & 1u) ? wfp : wfc) + shot_i;
      DW_DTYPE* __restrict const wfp_t =
          (((uint64_t)(t - start_t) & 1u) ? wfc : wfp) + shot_i;
      DW_DTYPE* __restrict const wfcsc_t =
          (((uint64_t)(t - start_t) & 1u) ? wfpsc : wfcsc) + shot_i;
      DW_DTYPE* __restrict const wfpsc_t =
          (((uint64_t)(t - start_t) & 1u) ? wfcsc : wfpsc) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE* __restrict const psiz_t =
          (((uint64_t)(t - start_t) & 1u) ? psizn : psiz) + shot_i;
      DW_DTYPE* __restrict const psizn_t =
          (((uint64_t)(t - start_t) & 1u) ? psiz : psizn) + shot_i;
      DW_DTYPE* __restrict const psizsc_t =
          (((uint64_t)(t - start_t) & 1u) ? psiznsc : psizsc) + shot_i;
      DW_DTYPE* __restrict const psiznsc_t =
          (((uint64_t)(t - start_t) & 1u) ? psizsc : psiznsc) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const psiy_t =
          (((uint64_t)(t - start_t) & 1u) ? psiyn : psiy) + shot_i;
      DW_DTYPE* __restrict const psiyn_t =
          (((uint64_t)(t - start_t) & 1u) ? psiy : psiyn) + shot_i;
      DW_DTYPE* __restrict const psiysc_t =
          (((uint64_t)(t - start_t) & 1u) ? psiynsc : psiysc) + shot_i;
      DW_DTYPE* __restrict const psiynsc_t =
          (((uint64_t)(t - start_t) & 1u) ? psiysc : psiynsc) + shot_i;
#endif
      DW_DTYPE* __restrict const psix_t =
          (((uint64_t)(t - start_t) & 1u) ? psixn : psix) + shot_i;
      DW_DTYPE* __restrict const psixn_t =
          (((uint64_t)(t - start_t) & 1u) ? psix : psixn) + shot_i;
      DW_DTYPE* __restrict const psixsc_t =
          (((uint64_t)(t - start_t) & 1u) ? psixnsc : psixsc) + shot_i;
      DW_DTYPE* __restrict const psixnsc_t =
          (((uint64_t)(t - start_t) & 1u) ? psixsc : psixnsc) + shot_i;

      /* Refill the guard ring of both the background and the scattered field
       * with the boundary image before the interior stencil reads it (no-op
       * when every face is PML). Mirroring is linear, so the scattered field
       * uses the same image as the background (see the module-level comment
       * above FUNC(fill_bc_line)). */
      if (has_bc) {
        FILL_BC(wfc_t, 0);
        FILL_BC(wfcsc_t, 0);
      }

      DW_DTYPE* __restrict const w_store_1_t =
          w_store_1 != NULL
              ? (w_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      DW_DTYPE* __restrict const wsc_store_1_t =
          wsc_store_1 != NULL
              ? (wsc_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      void* __restrict const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t*)w_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;
      void* __restrict const wsc_store_2_t =
          wsc_store_2 != NULL
              ? ((uint8_t*)wsc_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);
      bool const scatter_requires_grad_t =
          scatter_requires_grad && ((t % step_ratio) == 0);
      bool const v_or_scatter_requires_grad_t =
          v_requires_grad_t || scatter_requires_grad_t;
#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
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

#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
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
                    FUNC(forward_cell)(
                        wfc_t, wfp_t, wfcsc_t, wfpsc_t, w_store_1_t,
                        wsc_store_1_t, v_shot[i], scatter_shot[i], dt2,
                        v_or_scatter_requires_grad_t, v_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != 1, psiz_t, psizn_t, zetaz_t, psizsc_t,
                        psiznsc_t, zetazsc_t, az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                        y, pml_y != 1, psiy_t, psiyn_t, zetay_t, psiysc_t,
                        psiynsc_t, zetaysc_t, ay, by, dbydy, rdy, rdy2,
#endif
                        x, pml_x != 1, psix_t, psixn_t, zetax_t, psixsc_t,
                        psixnsc_t, zetaxsc_t, ax, bx, dbxdx, rdx, rdx2);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Save Snapshots
      if (v_or_scatter_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(save_snapshot_cpu)(w_store_1_t, w_store_2_t, fp_w,
                                            storage_mode, storage_compression,
                                            step_idx, shot_bytes_uncomp,
                                            shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(save_snapshot_cpu)(
                wsc_store_1_t, wsc_store_2_t, fp_wsc, storage_mode,
                storage_compression, step_idx, shot_bytes_uncomp,
                shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }

      if (n_sources_per_shot > 0) {
        add_to_wavefield(wfp_t, sources_i + si,
                         f + t * n_shots * n_sources_per_shot + si,
                         n_sources_per_shot);
      }
      if (n_sources_per_shot > 0) {
        add_to_wavefield(wfpsc_t, sources_i + si,
                         fsc + t * n_shots * n_sources_per_shot + si,
                         n_sources_per_shot);
      }
      if (n_receivers_per_shot > 0) {
        record_from_wavefield(wfc_t, receivers_i + ri,
                              r + t * n_shots * n_receivers_per_shot + ri,
                              n_receivers_per_shot);
      }
      if (n_receiverssc_per_shot > 0) {
        record_from_wavefield(wfcsc_t, receiverssc_i + risc,
                              rsc + t * n_shots * n_receiverssc_per_shot + risc,
                              n_receiverssc_per_shot);
      }
    }

  storage_cleanup:
    if (dw_close_file(&fp_w) != 0) storage_err = 1;
    if (dw_close_file(&fp_wsc) != 0) storage_err = 1;
    if (storage_err) return 1;
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(backward)(
    DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const scatter,
    DW_DTYPE const* __restrict const grad_r,
    DW_DTYPE const* __restrict const grad_rsc, DW_DTYPE* __restrict const wfc,
    DW_DTYPE* __restrict const wfp,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psiz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiy,
#endif
    DW_DTYPE* __restrict const psix,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psizn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiyn,
#endif
    DW_DTYPE* __restrict const psixn,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetaz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetay,
#endif
    DW_DTYPE* __restrict const zetax,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetazn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetayn,
#endif
    DW_DTYPE* __restrict const zetaxn, DW_DTYPE* __restrict const wfcsc,
    DW_DTYPE* __restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psizsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiysc,
#endif
    DW_DTYPE* __restrict const psixsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psiznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiynsc,
#endif
    DW_DTYPE* __restrict const psixnsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetazsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetaysc,
#endif
    DW_DTYPE* __restrict const zetaxsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetaynsc,
#endif
    DW_DTYPE* __restrict const zetaxnsc, DW_DTYPE* __restrict const w_store_1,
    DW_DTYPE* __restrict const w_store_1b, void* __restrict const w_store_2,
    void* __restrict const w_store_3,
    char const* __restrict const* __restrict const w_filenames_ptr,
    DW_DTYPE* __restrict const wsc_store_1,
    DW_DTYPE* __restrict const wsc_store_1b, void* __restrict const wsc_store_2,
    void* __restrict const wsc_store_3,
    char const* __restrict const* __restrict const wsc_filenames_ptr,
    DW_DTYPE* __restrict const grad_f, DW_DTYPE* __restrict const grad_fsc,
    DW_DTYPE* __restrict const grad_v, DW_DTYPE* __restrict const grad_scatter,
    DW_DTYPE* __restrict const grad_v_thread,
    DW_DTYPE* __restrict const grad_scatter_thread,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx,
    int64_t const* __restrict const sources_i,
    int64_t const* __restrict const receivers_i,
    int64_t const* __restrict const receiverssc_i,
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
    bool const scatter_batched, bool const storage_compression, int64_t start_t,
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
    int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0,
#endif
    int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y1,
#endif
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;

    if ((scatter_requires_grad) && (grad_scatter == NULL)) return 1;
    if ((n_receivers_per_shot > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot > 0 || n_sourcessc_per_shot > 0) &&
        sources_i == NULL)
      return 1;
    if (n_receiverssc_per_shot > 0 && receiverssc_i == NULL) return 1;
  }
#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const n_grid_points = nx;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  int64_t shot;

  (void)grad_v_thread; /* per-thread scratch unused under intra-shot spatial */
  (void)grad_scatter_thread;
  /* Shots run serially; each time step's adjoint stencil is spread across
   * threads (see the omp parallel region below). Model gradients accumulate
   * directly into the shared gradient buffers (self-cell writes, disjoint
   * across threads, barrier-separated across time steps and shots). */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const shot_i = shot * n_grid_points;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const sisc = shot * n_sourcessc_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
    DW_DTYPE* __restrict const grad_v_shot =
        grad_v != NULL ? (grad_v + (v_batched ? shot_i : 0)) : NULL;
    DW_DTYPE* __restrict const grad_scatter_shot =
        grad_scatter != NULL ? (grad_scatter + (scatter_batched ? shot_i : 0))
                             : NULL;
    DW_DTYPE const* __restrict const v_shot = v_batched ? v + shot_i : v;
    DW_DTYPE const* __restrict const scatter_shot =
        scatter_batched ? scatter + shot_i : scatter;
    int64_t t;

    int storage_err = 0;
    FILE* fp_w = NULL;
    FILE* fp_wsc = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad || scatter_requires_grad) {
        fp_w = dw_open_file_utf8(w_filenames_ptr[shot], "rb");
        if (fp_w == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
      if (v_requires_grad) {
        fp_wsc = dw_open_file_utf8(wsc_filenames_ptr[shot], "rb");
        if (fp_wsc == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    for (t = start_t - 1; t >= start_t - nt; --t) {
      if ((t % step_ratio) == 0 && (v_requires_grad || scatter_requires_grad) &&
          (w_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      if ((t % step_ratio) == 0 && (v_requires_grad) &&
          (wsc_store_1 == NULL || grad_v == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      DW_DTYPE* __restrict const wfc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? wfp : wfc) + shot_i;
      DW_DTYPE* __restrict const wfp_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? wfc : wfp) + shot_i;
      DW_DTYPE* __restrict const wfcsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? wfpsc : wfcsc) + shot_i;
      DW_DTYPE* __restrict const wfpsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? wfcsc : wfpsc) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE* __restrict const psiz_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psizn : psiz) + shot_i;
      DW_DTYPE* __restrict const psizn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiz : psizn) + shot_i;
      DW_DTYPE* __restrict const zetaz_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetazn : zetaz) + shot_i;
      DW_DTYPE* __restrict const zetazn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaz : zetazn) + shot_i;
      DW_DTYPE* __restrict const psizsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiznsc : psizsc) + shot_i;
      DW_DTYPE* __restrict const psiznsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psizsc : psiznsc) + shot_i;
      DW_DTYPE* __restrict const zetazsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaznsc : zetazsc) + shot_i;
      DW_DTYPE* __restrict const zetaznsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetazsc : zetaznsc) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const psiy_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiyn : psiy) + shot_i;
      DW_DTYPE* __restrict const psiyn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiy : psiyn) + shot_i;
      DW_DTYPE* __restrict const zetay_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetayn : zetay) + shot_i;
      DW_DTYPE* __restrict const zetayn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetay : zetayn) + shot_i;
      DW_DTYPE* __restrict const psiysc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiynsc : psiysc) + shot_i;
      DW_DTYPE* __restrict const psiynsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiysc : psiynsc) + shot_i;
      DW_DTYPE* __restrict const zetaysc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaynsc : zetaysc) + shot_i;
      DW_DTYPE* __restrict const zetaynsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaysc : zetaynsc) + shot_i;
#endif
      DW_DTYPE* __restrict const psix_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psixn : psix) + shot_i;
      DW_DTYPE* __restrict const psixn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psix : psixn) + shot_i;
      DW_DTYPE* __restrict const zetax_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaxn : zetax) + shot_i;
      DW_DTYPE* __restrict const zetaxn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetax : zetaxn) + shot_i;
      DW_DTYPE* __restrict const psixsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psixnsc : psixsc) + shot_i;
      DW_DTYPE* __restrict const psixnsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psixsc : psixnsc) + shot_i;
      DW_DTYPE* __restrict const zetaxsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaxnsc : zetaxsc) + shot_i;
      DW_DTYPE* __restrict const zetaxnsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaxsc : zetaxnsc) + shot_i;

      /* Adjoint guard fill: same mirror as the forward, applied to both
       * adjoint fields (self-adjoint on a reflecting, pml_width=0 face; see
       * the module-level comment above FUNC(fill_bc_line)). */
      if (has_bc) {
        FILL_BC(wfc_t, 0);
        FILL_BC(wfcsc_t, 0);
      }

      DW_DTYPE* __restrict const w_store_1_t =
          w_store_1 != NULL
              ? (w_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      DW_DTYPE* __restrict const wsc_store_1_t =
          wsc_store_1 != NULL
              ? (wsc_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      void* __restrict const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t*)w_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;
      void* __restrict const wsc_store_2_t =
          wsc_store_2 != NULL
              ? ((uint8_t*)wsc_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);
      bool const scatter_requires_grad_t =
          scatter_requires_grad && ((t % step_ratio) == 0);

      if (v_requires_grad_t || scatter_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(load_snapshot_cpu)(w_store_1_t, w_store_2_t, fp_w,
                                            storage_mode, storage_compression,
                                            step_idx, shot_bytes_uncomp,
                                            shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }

      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(load_snapshot_cpu)(
                wsc_store_1_t, wsc_store_2_t, fp_wsc, storage_mode,
                storage_compression, step_idx, shot_bytes_uncomp,
                shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
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
#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
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
                    FUNC(backward_cell)(
                        wfc_t, wfp_t, wfcsc_t, wfpsc_t, v_shot, scatter_shot,
                        w_store_1_t, wsc_store_1_t, grad_v_shot,
                        grad_scatter_shot, dt2, step_ratio, v_requires_grad_t,
                        scatter_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != 1, psiz_t, psizn_t, zetaz_t, zetazn_t,
                        psizsc_t, psiznsc_t, zetazsc_t, zetaznsc_t, az, bz,
                        dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                        y, pml_y != 1, psiy_t, psiyn_t, zetay_t, zetayn_t,
                        psiysc_t, psiynsc_t, zetaysc_t, zetaynsc_t, ay, by,
                        dbydy, rdy, rdy2,
#endif
                        x, pml_x != 1, psix_t, psixn_t, zetax_t, zetaxn_t,
                        psixsc_t, psixnsc_t, zetaxsc_t, zetaxnsc_t, ax, bx,
                        dbxdx, rdx, rdx2);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      if (n_receivers_per_shot > 0) {
        add_to_wavefield(wfp_t, receivers_i + ri,
                         grad_r + t * n_shots * n_receivers_per_shot + ri,
                         n_receivers_per_shot);
      }
      if (n_receiverssc_per_shot > 0) {
        add_to_wavefield(wfpsc_t, receiverssc_i + risc,
                         grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                         n_receiverssc_per_shot);
      }
      if (n_sources_per_shot > 0) {
        record_from_wavefield(wfc_t, sources_i + si,
                              grad_f + t * n_shots * n_sources_per_shot + si,
                              n_sources_per_shot);
      }
      if (n_sourcessc_per_shot > 0) {
        record_from_wavefield(
            wfcsc_t, sources_i + sisc,
            grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
            n_sourcessc_per_shot);
      }
    }

  storage_cleanup:
    if (dw_close_file(&fp_w) != 0) storage_err = 1;
    if (dw_close_file(&fp_wsc) != 0) storage_err = 1;
    if (storage_err) return 1;
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(backward_sc)(
    DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const grad_rsc, DW_DTYPE* __restrict const wfcsc,
    DW_DTYPE* __restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psizsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiysc,
#endif
    DW_DTYPE* __restrict const psixsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psiznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psiynsc,
#endif
    DW_DTYPE* __restrict const psixnsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetazsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetaysc,
#endif
    DW_DTYPE* __restrict const zetaxsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const zetaynsc,
#endif
    DW_DTYPE* __restrict const zetaxnsc, DW_DTYPE* __restrict const w_store_1,
    DW_DTYPE* __restrict const w_store_1b, void* __restrict const w_store_2,
    void* __restrict const w_store_3,
    char const* __restrict const* __restrict const w_filenames_ptr,
    DW_DTYPE* __restrict const grad_fsc,
    DW_DTYPE* __restrict const grad_scatter,
    DW_DTYPE* __restrict const grad_scatter_thread,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx,
    int64_t const* __restrict const sources_i,
    int64_t const* __restrict const receiverssc_i,
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
    int64_t const pml_x1,
#if DW_NDIM >= 3
    int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0,
#endif
    int64_t const bc_x0,
#if DW_NDIM >= 3
    int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y1,
#endif
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((n_sourcessc_per_shot > 0) && sources_i == NULL) return 1;
    if (n_receiverssc_per_shot > 0 && receiverssc_i == NULL) return 1;
  }
#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const n_grid_points = nx;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  int64_t shot;

  (void)grad_scatter_thread; /* per-thread scratch unused under spatial parallel
                              */
  /* Shots run serially; each time step's scattered adjoint stencil is spread
   * across threads (see the omp parallel region below). The scatter gradient
   * accumulates directly into the shared buffer (self-cell, disjoint across
   * threads, barrier-separated across time steps and shots). */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const shot_i = shot * n_grid_points;
    int64_t const sisc = shot * n_sourcessc_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
    DW_DTYPE* __restrict const grad_scatter_shot =
        grad_scatter != NULL ? (grad_scatter + (scatter_batched ? shot_i : 0))
                             : NULL;
    DW_DTYPE const* __restrict const v_shot = v_batched ? v + shot_i : v;
    int64_t t;

    int storage_err = 0;
    FILE* fp_w = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (scatter_requires_grad) {
        fp_w = dw_open_file_utf8(w_filenames_ptr[shot], "rb");
        if (fp_w == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    for (t = start_t - 1; t >= start_t - nt; --t) {
      if ((t % step_ratio) == 0 && (scatter_requires_grad) &&
          (w_store_1 == NULL || grad_scatter == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      DW_DTYPE* __restrict const wfcsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? wfpsc : wfcsc) + shot_i;
      DW_DTYPE* __restrict const wfpsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? wfcsc : wfpsc) + shot_i;
#if DW_NDIM >= 3
      DW_DTYPE* __restrict const psizsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiznsc : psizsc) + shot_i;
      DW_DTYPE* __restrict const psiznsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psizsc : psiznsc) + shot_i;
      DW_DTYPE* __restrict const zetazsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaznsc : zetazsc) + shot_i;
      DW_DTYPE* __restrict const zetaznsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetazsc : zetaznsc) + shot_i;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const psiysc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiynsc : psiysc) + shot_i;
      DW_DTYPE* __restrict const psiynsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psiysc : psiynsc) + shot_i;
      DW_DTYPE* __restrict const zetaysc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaynsc : zetaysc) + shot_i;
      DW_DTYPE* __restrict const zetaynsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaysc : zetaynsc) + shot_i;
#endif
      DW_DTYPE* __restrict const psixsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psixnsc : psixsc) + shot_i;
      DW_DTYPE* __restrict const psixnsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psixsc : psixnsc) + shot_i;
      DW_DTYPE* __restrict const zetaxsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaxnsc : zetaxsc) + shot_i;
      DW_DTYPE* __restrict const zetaxnsc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? zetaxsc : zetaxnsc) + shot_i;

      /* Adjoint guard fill: same mirror as the forward (self-adjoint on a
       * reflecting, pml_width=0 face). This scattered-only pass has no
       * background adjoint field to fill. */
      if (has_bc) FILL_BC(wfcsc_t, 0);

      DW_DTYPE* __restrict const w_store_1_t =
          w_store_1 != NULL
              ? (w_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      void* __restrict const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t*)w_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;

      bool const scatter_requires_grad_t =
          scatter_requires_grad && ((t % step_ratio) == 0);

      if (scatter_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(load_snapshot_cpu)(w_store_1_t, w_store_2_t, fp_w,
                                            storage_mode, storage_compression,
                                            step_idx, shot_bytes_uncomp,
                                            shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
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
#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
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
                    FUNC(backward_cell_sc)(
                        wfcsc_t, wfpsc_t, v_shot, w_store_1_t,
                        grad_scatter_shot, dt2, step_ratio,
                        scatter_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != 1, psizsc_t, psiznsc_t, zetazsc_t,
                        zetaznsc_t, az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                        y, pml_y != 1, psiysc_t, psiynsc_t, zetaysc_t,
                        zetaynsc_t, ay, by, dbydy, rdy, rdy2,
#endif
                        x, pml_x != 1, psixsc_t, psixnsc_t, zetaxsc_t,
                        zetaxnsc_t, ax, bx, dbxdx, rdx, rdx2);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      if (n_receiverssc_per_shot > 0) {
        add_to_wavefield(wfpsc_t, receiverssc_i + risc,
                         grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                         n_receiverssc_per_shot);
      }
      if (n_sourcessc_per_shot > 0) {
        record_from_wavefield(
            wfcsc_t, sources_i + sisc,
            grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
            n_sourcessc_per_shot);
      }
    }

  storage_cleanup:
    if (dw_close_file(&fp_w) != 0) storage_err = 1;
    if (storage_err) return 1;
  }
  return 0;
}

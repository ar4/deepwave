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
 * d^2u/dt^2 = v^2 * laplacian(u) - v^2 * f
 * (equivalently laplacian(u) - (1/v^2) d^2u/dt^2 = f)
 * where u is the wavefield, t is time, v is the wavespeed, and f is the
 * source. The Laplacian is applied to the spatial dimensions. The caller
 * (scalar.py) supplies the source amplitudes already multiplied by
 * -v^2 * dt^2, which is what add_to_wavefield adds to wfp below.
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
 * The code is structured to maximise performance by using macros to
 * generate the code for each of the regions, and by using OpenMP
 * to parallelise the loops over shots.
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
#include <string.h>

#include "common_cpu.h"
#include "regular_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  scalar_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define SCALAR_VARIANT_I(name) name
#define SCALAR_VARIANT(name) SCALAR_VARIANT_I(name)
#define FUNC(name) \
  CAT(SCALAR_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

/* ND_INDEX, WFC/V2DT2_WFC, PSIZ/ZETAZ/... (+Y, +X), the UT_TERM and PSI_TERM
 * family, and the FUNC(forward_cell)/FUNC(backward_cell) per-grid-point stencil
 * update functions live in scalar_kernels.h, shared verbatim with scalar.cu
 * (see that header for the full explanation). wfc_t/wfp_t/psiz_t/.../zetaz_t
 * below are this file's names for the (per-shot-offset) arrays passed as the
 * wfc/wfp/psiz/.../zetaz parameters of those functions. */
#include "scalar_kernels.h"

/* First-class reflecting/periodic boundary faces (see docs/boundaries.rst).
 * ``bc`` codes: 0=pml (absorbing, guard left untouched), 1=free (pressure
 * release, field odd about the outer node P-1, R=-1), 2=rigid (field even about
 * the half-cell plane P-1/2, R=+1), 3=periodic (wrap with period n-2P). The
 * nodal-grid interior along an axis of length ``n`` is ``[P, n-P)``.
 *
 * ``fill_bc`` refills a wavefield's guard ring with the exact image every outer
 * step, so the interior stencil reads the correct ghost values. The same fill
 * is applied in the forward and the (adjoint) backward pass: on a reflecting
 * face there is no PML, the spatial operator is the plain symmetric FD
 * operator, and the mirror-filled operator is self-adjoint, so re-applying the
 * fill (to the adjoint field) realises the exact transpose. ``is_model``
 * selects the model variant (used to mirror-pad v^2 into the guard so the
 * adjoint's coefficient reads match): the even/wrap image with no sign flip and
 * no on-plane zeroing. */
/* Fill the guard cells of a single line along one axis (base offset, element
 * stride ``stride``, axis length ``nax``) with the boundary image. */
static inline void FUNC(fill_bc_line)(DW_DTYPE *__restrict const wf,
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
 * variant used to mirror-pad the coefficients for the adjoint. */
static void FUNC(fill_bc)(DW_DTYPE *__restrict const wf, int const is_model,
                          int64_t const *__restrict const bc, int64_t const nz_,
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
DW_TARGET_CLONES int
FUNC(forward)(
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
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const dbydy,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
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
    bool const v_batched, bool const storage_compression, int64_t const start_t,
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
    int64_t const bc_x1, DW_DTYPE *__restrict const raw_store_1a,
    DW_DTYPE *__restrict const raw_store_1b, void *__restrict const raw_store_2,
    void *__restrict const raw_store_3,
    char const *__restrict const *__restrict const raw_filenames_ptr,
    bool const store_raw_forward, int64_t const raw_storage_mode,
    size_t const raw_shot_bytes_uncomp, size_t const raw_shot_bytes_comp,
    bool const raw_storage_compression, int64_t const n_threads, void *unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;

    if ((n_receivers_per_shot > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot > 0) && sources_i == NULL) return 1;
  }
  int64_t shot;
  /* raw_store_1b/raw_store_3 are unused on the CPU forward path: the
   * raw-forward checkpoint (opt-in, see CallbackState role="forward") never
   * double-buffers (unlike the primary checkpoint's GPU async pipeline), and
   * "cpu" storage mode on a CPU compute path is downgraded to "device" by
   * resolve_storage_mode (see scalar.py) -- so raw_store_3 (the CPU-pinned
   * multi-step staging buffer needed only when the compute and storage
   * devices differ) is never populated here either. raw_store_1a/
   * raw_store_2/raw_filenames_ptr ARE used below, for disk storage and/or
   * compression. */
  (void)raw_store_1b;
  (void)raw_store_3;
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
  /* Intra-shot spatial parallelism: shots run serially and the per-time-step
   * interior stencil is spread across threads (see the omp parallel region
   * below), so a single shot saturates all cores. */
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

    int storage_err = 0;
    FILE *fp_w = NULL;
    FILE *fp_raw = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad) {
        fp_w = dw_open_file_utf8(w_filenames_ptr[shot], "ab");
        if (fp_w == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    if (raw_storage_mode == STORAGE_DISK) {
      if (store_raw_forward) {
        fp_raw = dw_open_file_utf8(raw_filenames_ptr[shot], "ab");
        if (fp_raw == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    for (t = start_t; t < start_t + nt; ++t) {
      if ((t % step_ratio) == 0 && (v_requires_grad) && (w_store_1 == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      if ((t % step_ratio) == 0 &&
          (store_raw_forward && raw_storage_mode != STORAGE_NONE) &&
          (raw_store_1a == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      DW_DTYPE *__restrict const wfc_t =
          (((t - start_t) & 1) ? wfp : wfc) + shot_i;
      DW_DTYPE *__restrict const wfp_t =
          (((t - start_t) & 1) ? wfc : wfp) + shot_i;
      /* Opt-in raw-forward checkpoint (CallbackState role="forward"): copy
       * the field as it stands entering this outer step -- before the guard-
       * ring fill below and before this step's update -- so it is
       * time-coincident with the adjoint field the backward_callback sees at
       * the same step. Uses the SAME storage configuration as the primary
       * checkpoint (see scalar.py): the DEVICE-uncompressed fast path below
       * is a plain write into its permanent per-step slot, unchanged from
       * before this buffer honoured storage_mode. Every other mode/
       * compression combination instead stages the step into a single-step
       * buffer and persists it exactly like the primary checkpoint's own
       * per-step save (compression and/or a disk write); FUNC(backward)
       * below reloads it immediately before each chunk's backward_callback
       * fires. storage_mode="none" allocates raw_store_1a/1b/2 as
       * zero-size placeholders (StorageManager.allocate is a no-op in that
       * mode, matching the primary checkpoint), so it must be excluded here
       * too -- writing into it would be an out-of-bounds write, not merely
       * a wrong value. Python's forward_wavefields_for_step already returns
       * None for that mode's steps rather than reading it back. */
      if (store_raw_forward && ((t % step_ratio) == 0) &&
          raw_storage_mode != STORAGE_NONE) {
        int64_t const raw_step_idx = t / step_ratio;
        if (raw_storage_mode == STORAGE_DEVICE && !raw_storage_compression) {
          memcpy(raw_store_1a + shot_i + raw_step_idx * n_shots * n_grid_points,
                 wfc_t, (size_t)n_grid_points * sizeof(DW_DTYPE));
        } else {
          DW_DTYPE *__restrict const raw_store_1_t =
              raw_store_1a != NULL ? (raw_store_1a + shot_i) : NULL;
          void *__restrict const raw_store_2_t =
              raw_store_2 != NULL ? ((uint8_t *)raw_store_2 +
                                     (shot + (raw_storage_mode == STORAGE_DEVICE
                                                  ? raw_step_idx * n_shots
                                                  : 0)) *
                                         (int64_t)raw_shot_bytes_comp)
                                  : NULL;
          memcpy(raw_store_1_t, wfc_t,
                 (size_t)n_grid_points * sizeof(DW_DTYPE));
          if (STORAGE_FUNC(save_snapshot_cpu)(
                  raw_store_1_t, raw_store_2_t, fp_raw, raw_storage_mode,
                  raw_storage_compression, raw_step_idx, raw_shot_bytes_uncomp,
                  raw_shot_bytes_comp, DIM_ARGS) != 0) {
            storage_err = 1;
            goto storage_cleanup;
          }
        }
      }
      /* Refill the guard ring with the boundary image before the interior
       * stencil reads it (no-op when every face is PML). */
      if (has_bc) FILL_BC(wfc_t, 0);
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
          w_store_1 != NULL
              ? (w_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      void *__restrict const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t *)w_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
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
                    FUNC(forward_cell)(wfc_t, wfp_t, w_store_1_t, v_shot[i],
                                       dt2, v_requires_grad_t, i,
#if DW_NDIM >= 3
                                       ny,
#endif
#if DW_NDIM >= 2
                                       nx,
#endif
#if DW_NDIM >= 3
                                       z, pml_z != 1, psiz_t, psizn_t, zetaz_t,
                                       az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                                       y, pml_y != 1, psiy_t, psiyn_t, zetay_t,
                                       ay, by, dbydy, rdy, rdy2,
#endif
                                       x, pml_x != 1, psix_t, psixn_t, zetax_t,
                                       ax, bx, dbxdx, rdx, rdx2);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(save_snapshot_cpu)(w_store_1_t, w_store_2_t, fp_w,
                                            storage_mode, storage_compression,
                                            step_idx, shot_bytes_uncomp,
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
      if (n_receivers_per_shot > 0) {
        record_from_wavefield(wfc_t, receivers_i + ri,
                              r + t * n_shots * n_receivers_per_shot + ri,
                              n_receivers_per_shot);
      }
    }
  storage_cleanup:
    if (dw_close_file(&fp_w) != 0) storage_err = 1;
    if (dw_close_file(&fp_raw) != 0) storage_err = 1;
    if (storage_err) return 1;
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int
FUNC(backward)(
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
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const dbydy,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
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
    int64_t const bc_x1, DW_DTYPE *__restrict const raw_store_1a,
    DW_DTYPE *__restrict const raw_store_1b, void *__restrict const raw_store_2,
    void *__restrict const raw_store_3,
    char const *__restrict const *__restrict const raw_filenames_ptr,
    bool const store_raw_forward, int64_t const raw_storage_mode,
    size_t const raw_shot_bytes_uncomp, size_t const raw_shot_bytes_comp,
    bool const raw_storage_compression, int64_t const n_threads, void *unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;

    if ((n_receivers_per_shot > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot > 0) && sources_i == NULL) return 1;
  }
  /* The raw-forward checkpoint (CallbackState role="forward") is written by
   * FUNC(forward). For its DEVICE-uncompressed fast path it is read back
   * directly from Python by step index (it is DEVICE-resident and outlives
   * this call), so this backward pass still never touches raw_store_1a in
   * that case; raw_store_1b/raw_store_3 are unused here for the same reason
   * FUNC(forward) leaves them unused (see its comment). For every other
   * mode/compression combination, raw_store_1a/raw_store_2/
   * raw_filenames_ptr ARE used below to reload (and, if compressed,
   * decompress) the single-step staging buffer the backward_callback needs,
   * mirroring the primary checkpoint's own per-step load. */
  (void)raw_store_1b;
  (void)raw_store_3;
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
  /* Mirror-pad the model coefficients into the guard ring on reflecting faces
   * so the adjoint's V2DT2_WFC neighbour reads reproduce the exact transpose of
   * the forward's mirror fill (the mirror-filled operator is self-adjoint). For
   * a shared (non-batched) model this is done once, before the parallel region;
   * v2dt2 is a fresh per-call temporary (v**2 * dt**2), so this in-place write
   * of its guard cells is safe. */
  if (has_bc && !v_batched) FILL_BC((DW_DTYPE *)v2dt2, 1);
  (void)grad_v_thread; /* per-thread scratch unused: intra-shot spatial
                        * parallelism accumulates directly into grad_v (each
                        * cell written by exactly one thread, time steps and
                        * shots separated by barriers). */
  /* Shots run serially; the per-time-step interior stencil is spread across
   * threads (see the omp parallel region below). */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const shot_i = shot * n_grid_points;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    DW_DTYPE *__restrict const grad_v_shot =
        grad_v != NULL ? (grad_v + (v_batched ? shot_i : 0)) : NULL;
    DW_DTYPE const *__restrict const v2dt2_shot =
        v_batched ? v2dt2 + shot_i : v2dt2;
    int64_t t;
    /* For a batched (per-shot) model, mirror-pad this shot's coefficients. */
    if (has_bc && v_batched) FILL_BC((DW_DTYPE *)v2dt2_shot, 1);

    int storage_err = 0;
    FILE *fp_w = NULL;
    FILE *fp_raw = NULL;
    if (storage_mode == STORAGE_DISK) {
      if (v_requires_grad) {
        fp_w = dw_open_file_utf8(w_filenames_ptr[shot], "rb");
        if (fp_w == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    if (raw_storage_mode == STORAGE_DISK) {
      if (store_raw_forward) {
        fp_raw = dw_open_file_utf8(raw_filenames_ptr[shot], "rb");
        if (fp_raw == NULL) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
    }

    for (t = start_t - 1; t >= start_t - nt; --t) {
      if ((t % step_ratio) == 0 && (v_requires_grad) &&
          (w_store_1 == NULL || grad_v == NULL)) {
        storage_err = 1;
        goto storage_cleanup;
      }
      DW_DTYPE *__restrict const wfc_t =
          (((start_t - 1 - t) & 1) ? wfp : wfc) + shot_i;
      DW_DTYPE *__restrict const wfp_t =
          (((start_t - 1 - t) & 1) ? wfc : wfp) + shot_i;
      /* Adjoint guard fill: same mirror as the forward (self-adjoint). */
      if (has_bc) FILL_BC(wfc_t, 0);
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
          w_store_1 != NULL
              ? (w_store_1 + shot_i +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? t / step_ratio * n_shots * n_grid_points
                      : 0))
              : NULL;
      void *__restrict const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t *)w_store_2 +
                 (shot +
                  ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
                     (int64_t)shot_bytes_comp)
              : NULL;

      bool const v_requires_grad_t = v_requires_grad && ((t % step_ratio) == 0);

      if (v_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(load_snapshot_cpu)(w_store_1_t, w_store_2_t, fp_w,
                                            storage_mode, storage_compression,
                                            step_idx, shot_bytes_uncomp,
                                            shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }

      if (store_raw_forward && ((t % step_ratio) == 0) &&
          raw_storage_mode != STORAGE_NONE &&
          !(raw_storage_mode == STORAGE_DEVICE && !raw_storage_compression)) {
        /* CPU/disk-mode (or compressed DEVICE-mode) raw-forward checkpoint:
         * reload (and, if compressed, decompress) the step this chunk's
         * backward_callback is about to fire for into raw_store_1a's
         * single-step staging slot -- see FUNC(forward) above (which wrote
         * it) and scalar.py's forward_wavefields_for_step (which reads it
         * back once this call returns). Reloaded at every step boundary in
         * the chunk regardless of v_requires_grad_t (this checkpoint exists
         * purely for the callback, independent of whether the model needs a
         * gradient), so the buffer holds the correct value regardless of
         * callback_frequency; only the LAST (lowest t) load in a chunk is
         * ever actually read. storage_mode="none" is excluded (nothing was
         * ever persisted; see FUNC(forward) above and
         * forward_wavefields_for_step in scalar.py, which returns None for
         * its steps instead of reading this buffer). */
        int64_t const raw_step_idx = t / step_ratio;
        DW_DTYPE *__restrict const raw_store_1_t =
            raw_store_1a != NULL ? (raw_store_1a + shot_i) : NULL;
        void *__restrict const raw_store_2_t =
            raw_store_2 != NULL ? ((uint8_t *)raw_store_2 +
                                   (shot + (raw_storage_mode == STORAGE_DEVICE
                                                ? raw_step_idx * n_shots
                                                : 0)) *
                                       (int64_t)raw_shot_bytes_comp)
                                : NULL;
        if (STORAGE_FUNC(load_snapshot_cpu)(
                raw_store_1_t, raw_store_2_t, fp_raw, raw_storage_mode,
                raw_storage_compression, raw_step_idx, raw_shot_bytes_uncomp,
                raw_shot_bytes_comp, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
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
                    FUNC(backward_cell)(wfc_t, wfp_t, v2dt2_shot, w_store_1_t,
                                        grad_v_shot, v_requires_grad_t,
                                        step_ratio, i, i,
#if DW_NDIM >= 3
                                        ny,
#endif
#if DW_NDIM >= 2
                                        nx,
#endif
#if DW_NDIM >= 3
                                        z, pml_z != 1, psiz_t, psizn_t, zetaz_t,
                                        zetazn_t, az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                                        y, pml_y != 1, psiy_t, psiyn_t, zetay_t,
                                        zetayn_t, ay, by, dbydy, rdy, rdy2,
#endif
                                        x, pml_x != 1, psix_t, psixn_t, zetax_t,
                                        zetaxn_t, ax, bx, dbxdx, rdx, rdx2);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      if (n_receivers_per_shot > 0) {
        add_to_wavefield(wfp_t, receivers_i + ri,
                         grad_r + t * n_shots * n_receivers_per_shot + ri,
                         n_receivers_per_shot);
      }
      if (n_sources_per_shot > 0) {
        record_from_wavefield(wfc_t, sources_i + si,
                              grad_f + t * n_shots * n_sources_per_shot + si,
                              n_sources_per_shot);
      }
    }
  storage_cleanup:
    if (dw_close_file(&fp_w) != 0) storage_err = 1;
    if (dw_close_file(&fp_raw) != 0) storage_err = 1;
    if (storage_err) return 1;
  }
  return 0;
}

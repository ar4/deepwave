/*
 * Scalar Born wave equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the scalar Born wave equation
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
 * For a description of the method, see the C implementation in
 * scalar_born.c and scalar.c. This file implements the same functionality,
 * but for execution on a GPU using CUDA.
 */
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>

#include "common_gpu.h"
#include "regular_grid.h"
#include "storage_utils.h"

// Macro to concatenate function names with accuracy, dtype, and device for
// Python bindings
#define CAT_I(name, ndim, accuracy, dtype, device) \
  scalar_born_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define SCALAR_BORN_VARIANT_I(name) name
#define SCALAR_BORN_VARIANT(name) SCALAR_BORN_VARIANT_I(name)
#define FUNC(name) \
  CAT(SCALAR_BORN_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx_h
#endif

/* ND_INDEX, WFC/WFCSC, PSIZ/ZETAZ/... (+Y, +X) and their *SC scattered-field
 * twins, V/VDT2/V2DT2/SCATTER, V2DT2_WFC/V2DT2_WFCSC, the UT_TERM/UTSC_TERM/
 * PSI*_TERM/PSI*SC_TERM family, and the FUNC(forward_cell)/FUNC(backward_cell)/
 * FUNC(backward_cell_sc) per-grid-point stencil update functions live in
 * scalar_born_kernels.h, shared verbatim with scalar_born.c (see that header
 * for the full explanation). wfc/wfp/psiz/.../zetaz below (the arrays
 * spanning all shots) are this file's names for the arrays passed as the
 * wfc/wfp/psiz/.../zetaz parameters of those functions. */
#include "scalar_born_kernels.h"

// --- Device constants for configuration ---
// These are copied to constant memory for fast access by all kernels
namespace {
__constant__ DW_DTYPE dt2;  // Time step squared
#if DW_NDIM >= 3
__constant__ DW_DTYPE rdz;
__constant__ DW_DTYPE rdz2;
__constant__ int64_t nz;
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
#endif
#if DW_NDIM >= 2
__constant__ DW_DTYPE rdy;
__constant__ DW_DTYPE rdy2;
__constant__ int64_t ny;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
#endif
__constant__ DW_DTYPE rdx;
__constant__ DW_DTYPE rdx2;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_shots;
__constant__ int64_t n_sources_per_shot, n_sourcessc_per_shot;
__constant__ int64_t n_receivers_per_shot, n_receiverssc_per_shot;
__constant__ int64_t
    step_ratio;  // Number of steps between gradient contributions
__constant__ int64_t pml_x0, pml_x1;  // PML region bounds
__constant__ bool v_batched,
    scatter_batched;  // Whether v/scatter are shared or per shot

/* First-class reflecting/periodic boundary faces (see docs/boundaries.rst and
 * scalar.c/scalar.cu, which this mirrors exactly). ``bc`` codes: 0=pml (guard
 * untouched), 1=free (odd about node P-1), 2=rigid (even about half-cell
 * P-1/2), 3=periodic (wrap, period n-2P). ``is_model`` selects the even/wrap
 * no-sign no-zero variant (unused here: scalar_born's v/scatter guard rings
 * are mirror-padded on fresh Python-side clones before the call, since --
 * unlike scalar's fresh-per-call v2dt2 temporary -- v and scatter are read
 * directly and may alias the user's saved tensors, so an in-place CUDA-side
 * mirror would be unsafe; see mirror_pad_scalar_born_models in
 * boundaries.py). Kept for signature parity with scalar.cu. */
__device__ inline void fill_bc_line_dev(DW_DTYPE* __restrict const wf,
                                        int64_t const base, int64_t const nax,
                                        int64_t const stride,
                                        int64_t const bc_lo,
                                        int64_t const bc_hi,
                                        int const is_model) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P; /* interior node count */
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
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
    for (int64_t j = 0; j < P; ++j) {
      int64_t const xh = nax - P + j;
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
      wf[base + xh * stride] =
          zero ? (DW_DTYPE)0 : sign * wf[base + src * stride];
    }
  }
}

/* One thread per line along the chosen axis. ``n_lines`` = total_numel / nax.
 * For a contiguous array the base offset of line ``L`` is
 * ``(L / stride) * (nax * stride) + (L % stride)`` (the axis stride equals the
 * product of all lower dims, so the flat line index splits cleanly into the
 * indices above and below the axis). */
__global__ void fill_bc_kernel(DW_DTYPE* __restrict const wf,
                               int64_t const n_lines, int64_t const nax,
                               int64_t const stride, int64_t const bc_lo,
                               int64_t const bc_hi, int const is_model) {
  int64_t const L = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  fill_bc_line_dev(wf, base, nax, stride, bc_lo, bc_hi, is_model);
}

/* Launch the guard fill on every reflecting axis, innermost stride first (the
 * order the C reference uses, so corner cells are filled identically: a later
 * axis reads the already-filled guard of the earlier axes). ``bc`` is the
 * face-order array [z0,z1,y0,y1,x0,x1]. Kernels serialise on ``stream``. */
static void launch_fill_bc(DW_DTYPE* __restrict const wf, int const is_model,
                           int64_t const bc[6], int64_t const total_numel,
                           int64_t const nz_, int64_t const ny_,
                           int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const n_lines = total_numel / nx_;
    fill_bc_kernel<<<ceil_div(n_lines, tpb), tpb, 0, stream>>>(
        wf, n_lines, nx_, 1, bc[4], bc[5], is_model);
  }
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const n_lines = total_numel / ny_;
    fill_bc_kernel<<<ceil_div(n_lines, tpb), tpb, 0, stream>>>(
        wf, n_lines, ny_, nx_, bc[2], bc[3], is_model);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const n_lines = total_numel / nz_;
    fill_bc_kernel<<<ceil_div(n_lines, tpb), tpb, 0, stream>>>(
        wf, n_lines, nz_, ny_ * nx_, bc[0], bc[1], is_model);
  }
#endif
}

/* Convenience wrapper expanded inside the host launchers (bc, nz_h/ny_h/nx_h,
 * and stream_compute come from the enclosing scope). */
#if DW_NDIM == 3
#define LAUNCH_FILL_BC(wf, ismodel, total) \
  launch_fill_bc(wf, ismodel, bc, total, nz_h, ny_h, nx_h, stream_compute)
#elif DW_NDIM == 2
#define LAUNCH_FILL_BC(wf, ismodel, total) \
  launch_fill_bc(wf, ismodel, bc, total, 1, ny_h, nx_h, stream_compute)
#else
#define LAUNCH_FILL_BC(wf, ismodel, total) \
  launch_fill_bc(wf, ismodel, bc, total, 1, 1, nx_h, stream_compute)
#endif

__launch_bounds__(32) __global__
    void add_sources_both(DW_DTYPE* __restrict const wf,
                          DW_DTYPE* __restrict const wfsc,
                          DW_DTYPE const* __restrict const f,
                          DW_DTYPE const* __restrict const fsc,
                          int64_t const* __restrict const sources_i) {
  int64_t const source_idx = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    if (0 <= sources_i[k]) {
      wf[shot_idx * shot_numel + sources_i[k]] += f[k];
      wfsc[shot_idx * shot_numel + sources_i[k]] += fsc[k];
    }
  }
}

__launch_bounds__(32) __global__
    void add_adjoint_sources(DW_DTYPE* __restrict const wf,
                             DW_DTYPE const* __restrict const f,
                             int64_t const* __restrict const sources_i) {
  int64_t const source_idx = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (source_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void add_adjoint_sourcessc(DW_DTYPE* __restrict const wf,
                               DW_DTYPE const* __restrict const f,
                               int64_t const* __restrict const sources_i) {
  int64_t const source_idx = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (source_idx < n_receiverssc_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receiverssc_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void record_receivers(DW_DTYPE* __restrict const r,
                          DW_DTYPE const* __restrict const wf,
                          int64_t const* __restrict receivers_i) {
  int64_t const receiver_idx =
      dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

// Record scattered wavefield amplitudes at receiver locations for all shots.
__launch_bounds__(32) __global__
    void record_receiverssc(DW_DTYPE* __restrict const r,
                            DW_DTYPE const* __restrict const wf,
                            int64_t const* __restrict receivers_i) {
  int64_t const receiver_idx =
      dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (receiver_idx < n_receiverssc_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receiverssc_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

// Record adjoint wavefield amplitudes at source locations for all shots (used
// in adjoint computations).
__launch_bounds__(32) __global__
    void record_adjoint_receivers(DW_DTYPE* __restrict const r,
                                  DW_DTYPE const* __restrict const wf,
                                  int64_t const* __restrict receivers_i) {
  int64_t const receiver_idx =
      dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (receiver_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__
    void record_adjoint_receiverssc(DW_DTYPE* __restrict const rsc,
                                    DW_DTYPE const* __restrict const wfsc,
                                    int64_t const* __restrict receivers_i) {
  int64_t const receiver_idx =
      dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (receiver_idx < n_sourcessc_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sourcessc_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
      rsc[k] = wfsc[shot_idx * shot_numel + receivers_i[k]];
  }
}

// Combine per-shot gradients into a single gradient array (for v or scatter).
__launch_bounds__(128) __global__
    void combine_grad(DW_DTYPE* __restrict const grad,
                      DW_DTYPE const* __restrict const grad_shot) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  if (x < nx - FD_PAD) {
    int64_t const i = x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
#endif
}

// Main CUDA kernel for forward propagation of both background and scattered
// wavefields. Computes the next time step for both wavefields, including PML
// and auxiliary fields. Arguments:
//   v, scatter: velocity and scattering potential (can be batched per shot)
//   wfc, wfcsc: current background and scattered wavefields
//   wfp, wfpsc: previous (input) and next (output) time step wavefields
//   psiz, psizn, zetaz: PML auxiliary fields (background z-dim)
//   psiy, psiy n, zetay: PML auxiliary fields (background y-dim)
//   psix, psixn, zetax: PML auxiliary fields (background x-dim)
//   psizsc, psiznsc, zetazsc: PML auxiliary fields (scattered z-dim)
//   psiysc, psiynsc, zetaysc: PML auxiliary fields (scattered y-dim)
//   psixsc, psixnsc, zetaxsc: PML auxiliary fields (scattered x-dim)
//   w_store, wsc_store: storage for gradient calculation snapshots
//   az, bz, dbzdz: PML profiles and derivatives for z-dim
//   ay, ax, by, bx, dbydy, dbxdx: PML profiles and derivatives for y-dim and
//   x-dim v_requires_grad, scatter_requires_grad: whether to store values for
//   gradient computation
/* Thread-block shape and occupancy target for the three stencil kernels
 * below, mirroring scalar.cu's 2026-08-10 tuning (see that file for the full
 * rationale). scalar_born carries both a background and a scattered
 * wavefield through the same stencil, so its 3-D kernels are more
 * register-hungry than scalar's: a 128-thread bound already pins
 * forward_kernel/backward_kernel at up to 255 registers/thread at accuracy
 * >= 4, capping SM occupancy. A 32x4x4 block with a matching 512-thread
 * bound caps the allocation at 128 registers, the same lever that won for
 * scalar. Accuracy 2 keeps the narrow default (see the exclusion boundary
 * recorded at this file's launcher sites). */
#if DW_NDIM == 3 && DW_ACCURACY >= 4
#define DW_STENCIL_BLOCK_3D 32, 4, 4
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(512)
#else
#define DW_STENCIL_BLOCK_3D 32, 4, 1
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(128)
#endif

DW_STENCIL_LAUNCH_BOUNDS __global__ void forward_kernel(
    DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const scatter,
    DW_DTYPE const* __restrict const wfc, DW_DTYPE* __restrict const wfp,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const psiz, DW_DTYPE* __restrict const psizn,
    DW_DTYPE* __restrict const zetaz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const psiy, DW_DTYPE* __restrict const psiyn,
    DW_DTYPE* __restrict const zetay,
#endif
    DW_DTYPE const* __restrict const psix, DW_DTYPE* __restrict const psixn,
    DW_DTYPE* __restrict const zetax, DW_DTYPE const* __restrict const wfcsc,
    DW_DTYPE* __restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const psizsc, DW_DTYPE* __restrict const psiznsc,
    DW_DTYPE* __restrict const zetazsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const psiysc, DW_DTYPE* __restrict const psiynsc,
    DW_DTYPE* __restrict const zetaysc,
#endif
    DW_DTYPE const* __restrict const psixsc, DW_DTYPE* __restrict const psixnsc,
    DW_DTYPE* __restrict const zetaxsc, DW_DTYPE* __restrict const w_store,
    DW_DTYPE* __restrict const wsc_store,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx, bool const store_w,
    bool const store_wsc) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity and scatter arrays for this shot (batched or shared)
      DW_DTYPE const* __restrict const v_shot =
          v_batched ? v + shot_idx * shot_numel : v;
      DW_DTYPE const* __restrict const scatter_shot =
          scatter_batched ? scatter + shot_idx * shot_numel : scatter;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      FUNC(forward_cell)(wfc, wfp, wfcsc, wfpsc, w_store, wsc_store, v_shot[j],
                         scatter_shot[j], dt2, store_w, store_wsc, i,
#if DW_NDIM >= 3
                         ny,
#endif
#if DW_NDIM >= 2
                         nx,
#endif
#if DW_NDIM >= 3
                         z, pml_z, psiz, psizn, zetaz, psizsc, psiznsc, zetazsc,
                         az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                         y, pml_y, psiy, psiyn, zetay, psiysc, psiynsc, zetaysc,
                         ay, by, dbydy, rdy, rdy2,
#endif
                         x, pml_x, psix, psixn, zetax, psixsc, psixnsc, zetaxsc,
                         ax, bx, dbxdx, rdx, rdx2);
#if DW_NDIM == 3
    }
#endif
  }
}

// Main CUDA kernel for backward (adjoint) propagation for both background and
// scattered wavefields. Computes gradients with respect to velocity and scatter
// if requested. Arguments are analogous to forward_kernel, with additional
// gradient outputs.
DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel(
    DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const scatter,
    DW_DTYPE const* __restrict const wfc, DW_DTYPE* __restrict const wfp,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const psiz, DW_DTYPE* __restrict const psizn,
    DW_DTYPE* __restrict const zetaz, DW_DTYPE* __restrict const zetazn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const psiy, DW_DTYPE* __restrict const psiyn,
    DW_DTYPE const* __restrict const zetay, DW_DTYPE* __restrict const zetayn,
#endif
    DW_DTYPE const* __restrict const psix, DW_DTYPE* __restrict const psixn,
    DW_DTYPE const* __restrict const zetax, DW_DTYPE* __restrict const zetaxn,
    DW_DTYPE const* __restrict const wfcsc, DW_DTYPE* __restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const psizsc, DW_DTYPE* __restrict const psiznsc,
    DW_DTYPE* __restrict const zetazsc, DW_DTYPE* __restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const psiysc, DW_DTYPE* __restrict const psiynsc,
    DW_DTYPE const* __restrict const zetaysc,
    DW_DTYPE* __restrict const zetaynsc,
#endif
    DW_DTYPE const* __restrict const psixsc, DW_DTYPE* __restrict const psixnsc,
    DW_DTYPE const* __restrict const zetaxsc,
    DW_DTYPE* __restrict const zetaxnsc,
    DW_DTYPE const* __restrict const w_store,
    DW_DTYPE const* __restrict const wsc_store,
    DW_DTYPE* __restrict const grad_v, DW_DTYPE* __restrict const grad_scatter,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx, bool const v_requires_grad,
    bool const scatter_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity and scatter arrays for this shot (batched or shared)
      DW_DTYPE const* __restrict const v_shot =
          v_batched ? v + shot_idx * shot_numel : v;
      DW_DTYPE const* __restrict const scatter_shot =
          scatter_batched ? scatter + shot_idx * shot_numel : scatter;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      FUNC(backward_cell)(wfc, wfp, wfcsc, wfpsc, v_shot, scatter_shot, w_store,
                          wsc_store, grad_v, grad_scatter, dt2, step_ratio,
                          v_requires_grad, scatter_requires_grad, i, j,
#if DW_NDIM >= 3
                          ny,
#endif
#if DW_NDIM >= 2
                          nx,
#endif
#if DW_NDIM >= 3
                          z, pml_z, psiz, psizn, zetaz, zetazn, psizsc, psiznsc,
                          zetazsc, zetaznsc, az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                          y, pml_y, psiy, psiyn, zetay, zetayn, psiysc, psiynsc,
                          zetaysc, zetaynsc, ay, by, dbydy, rdy, rdy2,
#endif
                          x, pml_x, psix, psixn, zetax, zetaxn, psixsc, psixnsc,
                          zetaxsc, zetaxnsc, ax, bx, dbxdx, rdx, rdx2);
#if DW_NDIM == 3
    }
#endif
  }
}

// Specialised backward kernel for computing only the scatter gradient.
DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel_sc(
    DW_DTYPE const* __restrict const v, DW_DTYPE const* __restrict const wfcsc,
    DW_DTYPE* __restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const psizsc, DW_DTYPE* __restrict const psiznsc,
    DW_DTYPE const* __restrict const zetazsc,
    DW_DTYPE* __restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const psiysc, DW_DTYPE* __restrict const psiynsc,
    DW_DTYPE const* __restrict const zetaysc,
    DW_DTYPE* __restrict const zetaynsc,
#endif
    DW_DTYPE const* __restrict const psixsc, DW_DTYPE* __restrict const psixnsc,
    DW_DTYPE const* __restrict const zetaxsc,
    DW_DTYPE* __restrict const zetaxnsc,
    DW_DTYPE const* __restrict const w_store,
    DW_DTYPE* __restrict const grad_scatter,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const dbydy,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const dbxdx, bool const scatter_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity array for this shot (batched or shared)
      DW_DTYPE const* __restrict const v_shot =
          v_batched ? v + shot_idx * shot_numel : v;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      FUNC(backward_cell_sc)(wfcsc, wfpsc, v_shot, w_store, grad_scatter, dt2,
                             step_ratio, scatter_requires_grad, i, j,
#if DW_NDIM >= 3
                             ny,
#endif
#if DW_NDIM >= 2
                             nx,
#endif
#if DW_NDIM >= 3
                             z, pml_z, psizsc, psiznsc, zetazsc, zetaznsc, az,
                             bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                             y, pml_y, psiysc, psiynsc, zetaysc, zetaynsc, ay,
                             by, dbydy, rdy, rdy2,
#endif
                             x, pml_x, psixsc, psixnsc, zetaxsc, zetaxnsc, ax,
                             bx, dbxdx, rdx, rdx2);
#if DW_NDIM == 3
    }
#endif
  }
}

// Copy configuration parameters from host to device constant memory
int set_config(
#if DW_NDIM >= 3
    DW_DTYPE const rdz_h, DW_DTYPE const rdz2_h, int64_t const nz_h,
    int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_h, DW_DTYPE const rdy2_h, int64_t const ny_h,
    int64_t const pml_y0_h, int64_t const pml_y1_h,
#endif
    DW_DTYPE const rdx_h, DW_DTYPE const rdx2_h, int64_t const nx_h,
    DW_DTYPE const dt2_h, int64_t const n_shots_h,
    int64_t const n_sources_per_shot_h, int64_t const n_sourcessc_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
    int64_t const pml_x0_h, int64_t const pml_x1_h, bool const v_batched_h,
    bool const scatter_batched_h) {
#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif
  gpuErrchk(cudaMemcpyToSymbol(dt2, &dt2_h, sizeof(DW_DTYPE)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdz2, &rdz2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy2, &rdy2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx2, &rdx2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sourcessc_per_shot, &n_sourcessc_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receiverssc_per_shot,
                               &n_receiverssc_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(v_batched, &v_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(scatter_batched, &scatter_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int FUNC(forward)(
        DW_DTYPE const* __restrict const v,
        DW_DTYPE const* __restrict const scatter,
        DW_DTYPE const* __restrict const f,
        DW_DTYPE const* __restrict const fsc, DW_DTYPE* __restrict wfc,
        DW_DTYPE* __restrict wfp,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psiz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiy,
#endif
        DW_DTYPE* __restrict psix,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psizn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiyn,
#endif
        DW_DTYPE* __restrict psixn,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const zetaz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const zetay,
#endif
        DW_DTYPE* __restrict const zetax, DW_DTYPE* __restrict wfcsc,
        DW_DTYPE* __restrict wfpsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psizsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiysc,
#endif
        DW_DTYPE* __restrict psixsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psiznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiynsc,
#endif
        DW_DTYPE* __restrict psixnsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const zetazsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const zetaysc,
#endif
        DW_DTYPE* __restrict const zetaxsc,
        DW_DTYPE* __restrict const w_store_1a,
        DW_DTYPE* __restrict const w_store_1b, void* __restrict const w_store_2,
        void* __restrict const w_store_3,
        char const* __restrict const* __restrict const w_filenames_ptr,
        DW_DTYPE* __restrict const wsc_store_1a,
        DW_DTYPE* __restrict const wsc_store_1b,
        void* __restrict const wsc_store_2, void* __restrict const wsc_store_3,
        char const* __restrict const* __restrict const wsc_filenames_ptr,
        DW_DTYPE* __restrict const r, DW_DTYPE* __restrict const rsc,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const az,
        DW_DTYPE const* __restrict const bz,
        DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const ay,
        DW_DTYPE const* __restrict const by,
        DW_DTYPE const* __restrict const dbydy,
#endif
        DW_DTYPE const* __restrict const ax,
        DW_DTYPE const* __restrict const bx,
        DW_DTYPE const* __restrict const dbxdx,
        int64_t const* __restrict const sources_i,
        int64_t const* __restrict const receivers_i,
        int64_t const* __restrict const receiverssc_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy_h,
#endif
        DW_DTYPE const rdx_h,
#if DW_NDIM >= 3
        DW_DTYPE const rdz2_h,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy2_h,
#endif
        DW_DTYPE const rdx2_h, DW_DTYPE const dt2_h, int64_t const nt,
        int64_t const n_shots_h,
#if DW_NDIM >= 3
        int64_t const nz_h,
#endif
#if DW_NDIM >= 2
        int64_t const ny_h,
#endif
        int64_t const nx_h, int64_t const n_sources_per_shot_h,
        int64_t const n_receivers_per_shot_h,
        int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const scatter_requires_grad, bool const v_batched_h,
        bool const scatter_batched_h, bool const storage_compression,
        int64_t const start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0_h,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y0_h,
#endif
        int64_t const pml_x0_h,
#if DW_NDIM >= 3
        int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y1_h,
#endif
        int64_t const pml_x1_h,
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
        int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;

    if ((n_receivers_per_shot_h > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot_h > 0) && sources_i == NULL) return 1;
    if (n_receiverssc_per_shot_h > 0 && receiverssc_i == NULL) return 1;
    bool const dw_has_sample = (start_t % step_ratio_h == 0 ||
                                nt > step_ratio_h - start_t % step_ratio_h);
    if (dw_has_sample && (v_requires_grad || scatter_requires_grad) &&
        (w_store_1a == NULL))
      return 1;
    if (dw_has_sample && (v_requires_grad) && (wsc_store_1a == NULL)) return 1;
  }
  // --- CUDA kernel launch configuration ---
#if DW_NDIM == 3
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2LL * FD_PAD, dimBlock.z);
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);

  // For source and receiver operations
  dim3 const dimBlock_sources(32, 1, 1);
  unsigned int const gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int const gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int const gridz_sources = 1;
  dim3 const dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 const dimBlock_receivers(32, 1, 1);
  unsigned int const gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int const gridy_receivers =
      ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int const gridz_receivers = 1;
  dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers,
                               gridz_receivers);
  dim3 const dimBlock_receiverssc(32, 1, 1);
  unsigned int const gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int const gridy_receiverssc =
      ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int const gridz_receiverssc = 1;
  dim3 const dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                                 gridz_receiverssc);

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

  int64_t t;
  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device_guard.selected_device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, rdz2_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, rdy2_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, rdx2_h, nx_h, dt2_h, n_shots_h, n_sources_per_shot_h,
        n_sources_per_shot_h, n_receivers_per_shot_h, n_receiverssc_per_shot_h,
        step_ratio_h, pml_x0_h, pml_x1_h, v_batched_h, scatter_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (v_requires_grad || scatter_requires_grad);

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  cudaEvent_t event_storage_done;
  if (use_double_buffering) {
    gpuErrchk(
        cudaStreamCreateWithFlags(&stream_storage, cudaStreamNonBlocking));
    gpuErrchk(cudaEventCreate(&event_storage_done_a));
    gpuErrchk(cudaEventCreate(&event_storage_done_b));
    gpuErrchk(cudaEventCreate(&event_compute_done_a));
    gpuErrchk(cudaEventCreate(&event_compute_done_b));
    gpuErrchk(cudaEventRecord(event_storage_done_a, stream_storage));
    gpuErrchk(cudaEventRecord(event_storage_done_b, stream_storage));
    event_storage_done = event_storage_done_a;
  }

  ScopedFile fp_w;
  file_group.add(fp_w);
  ScopedFile fp_wsc;
  file_group.add(fp_wsc);
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad && !fp_w.open(w_filenames_ptr[0], "ab")) return 1;
  }
  if (storage_mode == STORAGE_DISK) {
    if ((v_requires_grad || scatter_requires_grad) &&
        !fp_w.open(w_filenames_ptr[0], "ab"))
      return 1;
    if (v_requires_grad && !fp_wsc.open(wsc_filenames_ptr[0], "ab")) return 1;
  }

  // --- Time-stepping loop for forward propagation ---
  // Alternates between wfc/wfp and wfp/wfc for memory efficiency
  for (t = start_t; t < start_t + nt; ++t) {
    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_w =
        store_step && (v_requires_grad || scatter_requires_grad);
    bool const store_wsc = store_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE* __restrict w_store_1_t =
        w_store_1a != NULL
            ? (w_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                    : 0))
            : NULL;
    DW_DTYPE* __restrict wsc_store_1_t =
        wsc_store_1a != NULL
            ? (wsc_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                    : 0))
            : NULL;
    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;

    if ((store_w || store_wsc) && use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }
    if (store_w && use_double_buffering) {
      if (step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
      }
    }
    if (store_wsc && use_double_buffering) {
      if (step_idx % 2 != 0) {
        wsc_store_1_t = wsc_store_1b;
      }
    }
    void* __restrict const w_store_2_t =
        w_store_2 != NULL
            ? ((uint8_t*)w_store_2 +
               (storage_mode == STORAGE_DEVICE && storage_compression
                    ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const wsc_store_2_t =
        wsc_store_2 != NULL
            ? ((uint8_t*)wsc_store_2 +
               (storage_mode == STORAGE_DEVICE && storage_compression
                    ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const w_store_3_t =
        w_store_3 != NULL
            ? ((uint8_t*)w_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) *
                          (storage_compression ? shot_bytes_comp
                                               : shot_bytes_uncomp) *
                          n_shots_h
                    : 0))
            : NULL;
    void* __restrict const wsc_store_3_t =
        wsc_store_3 != NULL
            ? ((uint8_t*)wsc_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) *
                          (storage_compression ? shot_bytes_comp
                                               : shot_bytes_uncomp) *
                          n_shots_h
                    : 0))
            : NULL;

    /* Refill the guard ring of both the background and the scattered field
     * with the boundary image before the interior stencil reads it (no-op
     * when every face is PML). */
    if (has_bc) {
      LAUNCH_FILL_BC(wfc, 0, n_shots_h * shot_numel_h);
      LAUNCH_FILL_BC(wfcsc, 0, n_shots_h * shot_numel_h);
    }

    forward_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, scatter, wfc, wfp,
#if DW_NDIM >= 3
        psiz, psizn, zetaz,
#endif
#if DW_NDIM >= 2
        psiy, psiyn, zetay,
#endif
        psix, psixn, zetax, wfcsc, wfpsc,
#if DW_NDIM >= 3
        psizsc, psiznsc, zetazsc,
#endif
#if DW_NDIM >= 2
        psiysc, psiynsc, zetaysc,
#endif
        psixsc, psixnsc, zetaxsc, w_store_1_t, wsc_store_1_t,
#if DW_NDIM >= 3
        az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        ay, by, dbydy,
#endif
        ax, bx, dbxdx, store_w, store_wsc);
    CHECK_KERNEL_ERROR

    if ((store_w || store_wsc) && use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    if (store_w) {
      if (STORAGE_FUNC(save_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }
    if (store_wsc) {
      if (STORAGE_FUNC(save_snapshot_gpu)(
              wsc_store_1_t, wsc_store_2_t, wsc_store_3_t, fp_wsc, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }

    if ((store_w || store_wsc) && use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    // Add sources to both background and scattered wavefields
    if (n_sources_per_shot_h > 0) {
      add_sources_both<<<dimGrid_sources, dimBlock_sources, 0,
                         stream_compute>>>(
          wfp, wfpsc, f + t * n_shots_h * n_sources_per_shot_h,
          fsc + t * n_shots_h * n_sources_per_shot_h, sources_i);
      CHECK_KERNEL_ERROR
    }
    // Record background wavefield at receiver locations
    if (n_receivers_per_shot_h > 0) {
      record_receivers<<<dimGrid_receivers, dimBlock_receivers, 0,
                         stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, wfc, receivers_i);
      CHECK_KERNEL_ERROR
    }
    // Record scattered wavefield at receiver locations
    if (n_receiverssc_per_shot_h > 0) {
      record_receiverssc<<<dimGrid_receiverssc, dimBlock_receiverssc, 0,
                           stream_compute>>>(
          rsc + t * n_shots_h * n_receiverssc_per_shot_h, wfcsc, receiverssc_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psiz, psizn);
    std::swap(psizsc, psiznsc);
#endif
#if DW_NDIM >= 2
    std::swap(psiy, psiyn);
    std::swap(psiysc, psiynsc);
#endif
    std::swap(wfc, wfp);
    std::swap(psix, psixn);
    std::swap(wfcsc, wfpsc);
    std::swap(psixsc, psixnsc);
  }

  if (use_double_buffering)
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

//
// Backward (adjoint) propagation for the Born model.
// This function launches the backward CUDA kernels in a time-reversed
// loop, handling both the background and scattered wavefields, and accumulates
// gradients for velocity and scatter.
//
// Arguments:
//   v, scatter: velocity and scattering potential arrays
//   grad_r, grad_rsc: receiver gradients for background and scattered fields
//   wfc, wfp, ...: wavefield and auxiliary arrays (see forward for details)
//   grad_v, grad_scatter: output gradients for velocity and scatter
//   grad_v_shot, grad_scatter_shot: per-shot gradients (for reduction)
//   ay, ax, by, bx, dbydy, dbxdx: PML profiles and derivatives
//   sources_i, receivers_i, receiverssc_i: source/receiver indices
//   ... and various grid, time, and configuration parameters
//
extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int FUNC(backward)(
        DW_DTYPE const* __restrict const v,
        DW_DTYPE const* __restrict const scatter,
        DW_DTYPE const* __restrict const grad_r,
        DW_DTYPE const* __restrict const grad_rsc, DW_DTYPE* __restrict wfc,
        DW_DTYPE* __restrict wfp,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psiz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiy,
#endif
        DW_DTYPE* __restrict psix,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psizn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiyn,
#endif
        DW_DTYPE* __restrict psixn,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict zetaz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict zetay,
#endif
        DW_DTYPE* __restrict zetax,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict zetazn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict zetayn,
#endif
        DW_DTYPE* __restrict zetaxn, DW_DTYPE* __restrict wfcsc,
        DW_DTYPE* __restrict wfpsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psizsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiysc,
#endif
        DW_DTYPE* __restrict psixsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psiznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiynsc,
#endif
        DW_DTYPE* __restrict psixnsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict zetazsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict zetaysc,
#endif
        DW_DTYPE* __restrict zetaxsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict zetaznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict zetaynsc,
#endif
        DW_DTYPE* __restrict zetaxnsc, DW_DTYPE* __restrict const w_store_1a,
        DW_DTYPE* __restrict const w_store_1b, void* __restrict const w_store_2,
        void* __restrict const w_store_3,
        char const* __restrict const* __restrict const w_filenames_ptr,
        DW_DTYPE* __restrict const wsc_store_1a,
        DW_DTYPE* __restrict const wsc_store_1b,
        void* __restrict const wsc_store_2, void* __restrict const wsc_store_3,
        char const* __restrict const* __restrict const wsc_filenames_ptr,
        DW_DTYPE* __restrict const grad_f, DW_DTYPE* __restrict const grad_fsc,
        DW_DTYPE* __restrict const grad_v,
        DW_DTYPE* __restrict const grad_scatter,
        DW_DTYPE* __restrict const grad_v_shot,
        DW_DTYPE* __restrict const grad_scatter_shot,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const az,
        DW_DTYPE const* __restrict const bz,
        DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const ay,
        DW_DTYPE const* __restrict const by,
        DW_DTYPE const* __restrict const dbydy,
#endif
        DW_DTYPE const* __restrict const ax,
        DW_DTYPE const* __restrict const bx,
        DW_DTYPE const* __restrict const dbxdx,
        int64_t const* __restrict const sources_i,
        int64_t const* __restrict const receivers_i,
        int64_t const* __restrict const receiverssc_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy_h,
#endif
        DW_DTYPE const rdx_h,
#if DW_NDIM >= 3
        DW_DTYPE const rdz2_h,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy2_h,
#endif
        DW_DTYPE const rdx2_h, DW_DTYPE const dt2_h, int64_t const nt,
        int64_t const n_shots_h,
#if DW_NDIM >= 3
        int64_t const nz_h,
#endif
#if DW_NDIM >= 2
        int64_t const ny_h,
#endif
        int64_t const nx_h, int64_t const n_sources_per_shot_h,
        int64_t const n_sourcessc_per_shot_h,
        int64_t const n_receivers_per_shot_h,
        int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const scatter_requires_grad, bool const v_batched_h,
        bool const scatter_batched_h, bool const storage_compression,
        int64_t const start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0_h,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y0_h,
#endif
        int64_t const pml_x0_h,
#if DW_NDIM >= 3
        int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y1_h,
#endif
        int64_t const pml_x1_h,
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
        int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;

    if ((scatter_requires_grad) && (grad_scatter == NULL)) return 1;
    if ((n_receivers_per_shot_h > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot_h > 0 || n_sourcessc_per_shot_h > 0) &&
        sources_i == NULL)
      return 1;
    if (n_receiverssc_per_shot_h > 0 && receiverssc_i == NULL) return 1;
    bool const dw_has_sample = ((start_t - 1) % step_ratio_h < nt);
    if (dw_has_sample && (v_requires_grad || scatter_requires_grad) &&
        (w_store_1a == NULL))
      return 1;
    if (dw_has_sample && (v_requires_grad) &&
        (wsc_store_1a == NULL || grad_v == NULL))
      return 1;
  }
#if DW_NDIM == 3
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2LL * FD_PAD, dimBlock.z);
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);

  // For source and receiver operations
  dim3 const dimBlock_sources(32, 1, 1);
  unsigned int const gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int const gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int const gridz_sources = 1;
  dim3 const dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 const dimBlock_sourcessc(32, 1, 1);
  unsigned int const gridx_sourcessc =
      ceil_div(n_sourcessc_per_shot_h, dimBlock_sourcessc.x);
  unsigned int const gridy_sourcessc =
      ceil_div(n_shots_h, dimBlock_sourcessc.y);
  unsigned int const gridz_sourcessc = 1;
  dim3 const dimGrid_sourcessc(gridx_sourcessc, gridy_sourcessc,
                               gridz_sourcessc);
  dim3 const dimBlock_receivers(32, 1, 1);
  unsigned int const gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int const gridy_receivers =
      ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int const gridz_receivers = 1;
  dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers,
                               gridz_receivers);
  dim3 const dimBlock_receiverssc(32, 1, 1);
  unsigned int const gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int const gridy_receiverssc =
      ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int const gridz_receiverssc = 1;
  dim3 const dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                                 gridz_receiverssc);

#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2LL * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2LL * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine =
      ceil_div(nz_h - 2LL * FD_PAD, dimBlock_combine.z);
#elif DW_NDIM == 2
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2LL * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2LL * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock_combine(128, 1, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2LL * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine = 1;
  unsigned int const gridz_combine = 1;
#endif
  dim3 const dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

  int64_t t;
  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device_guard.selected_device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, rdz2_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, rdy2_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, rdx2_h, nx_h, dt2_h, n_shots_h, n_sources_per_shot_h,
        n_sourcessc_per_shot_h, n_receivers_per_shot_h,
        n_receiverssc_per_shot_h, step_ratio_h, pml_x0_h, pml_x1_h, v_batched_h,
        scatter_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (v_requires_grad || scatter_requires_grad);

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  if (use_double_buffering) {
    gpuErrchk(
        cudaStreamCreateWithFlags(&stream_storage, cudaStreamNonBlocking));
    gpuErrchk(cudaEventCreate(&event_storage_done_a));
    gpuErrchk(cudaEventCreate(&event_storage_done_b));
    gpuErrchk(cudaEventCreate(&event_compute_done_a));
    gpuErrchk(cudaEventCreate(&event_compute_done_b));
    gpuErrchk(cudaEventRecord(event_compute_done_a, stream_compute));
    gpuErrchk(cudaEventRecord(event_compute_done_b, stream_compute));
  }

  ScopedFile fp_w;
  file_group.add(fp_w);
  ScopedFile fp_wsc;
  file_group.add(fp_wsc);
  if (storage_mode == STORAGE_DISK) {
    if ((v_requires_grad || scatter_requires_grad) &&
        !fp_w.open(w_filenames_ptr[0], "rb"))
      return 1;
    if (v_requires_grad && !fp_wsc.open(wsc_filenames_ptr[0], "rb")) return 1;
  }

  // --- Time-reversed loop for adjoint propagation ---
  // Alternates between wfc/wfp and wfp/wfc for memory efficiency
  for (t = start_t - 1; t >= start_t - nt; --t) {
    bool const load_step = (t % step_ratio_h) == 0;
    bool const load_w = load_step && (v_requires_grad || scatter_requires_grad);
    bool const load_wsc = load_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE* __restrict w_store_1_t =
        w_store_1a != NULL
            ? (w_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                    : 0))
            : NULL;
    DW_DTYPE* __restrict wsc_store_1_t =
        wsc_store_1a != NULL
            ? (wsc_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const w_store_2_t =
        w_store_2 != NULL
            ? ((uint8_t*)w_store_2 +
               (storage_mode == STORAGE_DEVICE && storage_compression
                    ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const wsc_store_2_t =
        wsc_store_2 != NULL
            ? ((uint8_t*)wsc_store_2 +
               (storage_mode == STORAGE_DEVICE && storage_compression
                    ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const w_store_3_t =
        w_store_3 != NULL
            ? ((uint8_t*)w_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) *
                          (storage_compression ? shot_bytes_comp
                                               : shot_bytes_uncomp) *
                          n_shots_h
                    : 0))
            : NULL;
    void* __restrict const wsc_store_3_t =
        wsc_store_3 != NULL
            ? ((uint8_t*)wsc_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) *
                          (storage_compression ? shot_bytes_comp
                                               : shot_bytes_uncomp) *
                          n_shots_h
                    : 0))
            : NULL;
    cudaEvent_t event_storage_done = event_storage_done_a;
    cudaEvent_t event_compute_done = event_compute_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_storage_done = event_storage_done_b;
        event_compute_done = event_compute_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    if (load_w) {
      if (use_double_buffering && step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
      }
      if (STORAGE_FUNC(load_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }

    if (load_wsc) {
      if (use_double_buffering && step_idx % 2 != 0) {
        wsc_store_1_t = wsc_store_1b;
      }
      if (STORAGE_FUNC(load_snapshot_gpu)(
              wsc_store_1_t, wsc_store_2_t, wsc_store_3_t, fp_wsc, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }
    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    // Record source gradients for background and scattered fields
    if (n_sources_per_shot_h > 0) {
      record_adjoint_receivers<<<dimGrid_sources, dimBlock_sources, 0,
                                 stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, wfc, sources_i);
      CHECK_KERNEL_ERROR
    }
    if (n_sourcessc_per_shot_h > 0) {
      record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc, 0,
                                   stream_compute>>>(
          grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfcsc, sources_i);
      CHECK_KERNEL_ERROR
    }

    /* Adjoint guard fill: same mirror as the forward, applied to both
     * adjoint fields (self-adjoint on a reflecting, pml_width=0 face). */
    if (has_bc) {
      LAUNCH_FILL_BC(wfc, 0, n_shots_h * shot_numel_h);
      LAUNCH_FILL_BC(wfcsc, 0, n_shots_h * shot_numel_h);
    }

    // Launch backward kernel for this time step
    backward_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, scatter, wfc, wfp,
#if DW_NDIM >= 3
        psiz, psizn, zetaz, zetazn,
#endif
#if DW_NDIM >= 2
        psiy, psiyn, zetay, zetayn,
#endif
        psix, psixn, zetax, zetaxn, wfcsc, wfpsc,
#if DW_NDIM >= 3
        psizsc, psiznsc, zetazsc, zetaznsc,
#endif
#if DW_NDIM >= 2
        psiysc, psiynsc, zetaysc, zetaynsc,
#endif
        psixsc, psixnsc, zetaxsc, zetaxnsc, w_store_1_t, wsc_store_1_t,
        grad_v_shot, grad_scatter_shot,
#if DW_NDIM >= 3
        az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        ay, by, dbydy,
#endif
        ax, bx, dbxdx, load_w && v_requires_grad,
        load_w && scatter_requires_grad);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Add receiver gradients as adjoint sources
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources<<<dimGrid_receivers, dimBlock_receivers, 0,
                            stream_compute>>>(
          wfp, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receiverssc_per_shot_h > 0) {
      add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc, 0,
                              stream_compute>>>(
          wfpsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
          receiverssc_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psiz, psizn);
    std::swap(zetaz, zetazn);
    std::swap(psizsc, psiznsc);
    std::swap(zetazsc, zetaznsc);
#endif
#if DW_NDIM >= 2
    std::swap(psiy, psiyn);
    std::swap(zetay, zetayn);
    std::swap(psiysc, psiynsc);
    std::swap(zetaysc, zetaynsc);
#endif
    std::swap(wfc, wfp);
    std::swap(psix, psixn);
    std::swap(zetax, zetaxn);
    std::swap(wfcsc, wfpsc);
    std::swap(psixsc, psixnsc);
    std::swap(zetaxsc, zetaxnsc);
  }
  // Reduce per-shot gradients to a single array if needed
  if (v_requires_grad && !v_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_v, grad_v_shot);
    CHECK_KERNEL_ERROR
  }
  if (scatter_requires_grad && !scatter_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_scatter, grad_scatter_shot);
    CHECK_KERNEL_ERROR
  }

  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

//
// Specialised entry point for the CUDA backward (adjoint) propagation for the
// scattered field only. This is an optimisation used when only the scatter
// gradient is needed.
//
// Arguments are analogous to FUNC(backward), but only for the scattered field
// and its gradient.
//
extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int FUNC(backward_sc)(
        DW_DTYPE const* __restrict const v,
        DW_DTYPE const* __restrict const grad_rsc, DW_DTYPE* __restrict wfcsc,
        DW_DTYPE* __restrict wfpsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psizsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiysc,
#endif
        DW_DTYPE* __restrict psixsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict psiznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict psiynsc,
#endif
        DW_DTYPE* __restrict psixnsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict zetazsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict zetaysc,
#endif
        DW_DTYPE* __restrict zetaxsc,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict zetaznsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict zetaynsc,
#endif
        DW_DTYPE* __restrict zetaxnsc, DW_DTYPE* __restrict const w_store_1a,
        DW_DTYPE* __restrict const w_store_1b, void* __restrict const w_store_2,
        void* __restrict const w_store_3,
        char const* __restrict const* __restrict const w_filenames_ptr,
        DW_DTYPE* __restrict const grad_fsc,
        DW_DTYPE* __restrict const grad_scatter,
        DW_DTYPE* __restrict const grad_scatter_shot,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const az,
        DW_DTYPE const* __restrict const bz,
        DW_DTYPE const* __restrict const dbzdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const ay,
        DW_DTYPE const* __restrict const by,
        DW_DTYPE const* __restrict const dbydy,
#endif
        DW_DTYPE const* __restrict const ax,
        DW_DTYPE const* __restrict const bx,
        DW_DTYPE const* __restrict const dbxdx,
        int64_t const* __restrict const sources_i,
        int64_t const* __restrict const receiverssc_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy_h,
#endif
        DW_DTYPE const rdx_h,
#if DW_NDIM >= 3
        DW_DTYPE const rdz2_h,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy2_h,
#endif
        DW_DTYPE const rdx2_h, DW_DTYPE const dt2_h, int64_t const nt,
        int64_t const n_shots_h,
#if DW_NDIM >= 3
        int64_t const nz_h,
#endif
#if DW_NDIM >= 2
        int64_t const ny_h,
#endif
        int64_t const nx_h, int64_t const n_sourcessc_per_shot_h,
        int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const scatter_requires_grad,
        bool const v_batched_h, bool const scatter_batched_h,
        bool const storage_compression, int64_t const start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0_h,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y0_h,
#endif
        int64_t const pml_x0_h,
#if DW_NDIM >= 3
        int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y1_h,
#endif
        int64_t const pml_x1_h,
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
        int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;
    if ((n_sourcessc_per_shot_h > 0) && sources_i == NULL) return 1;
    if (n_receiverssc_per_shot_h > 0 && receiverssc_i == NULL) return 1;
    bool const dw_has_sample = ((start_t - 1) % step_ratio_h < nt);
    if (dw_has_sample && (scatter_requires_grad) &&
        (w_store_1a == NULL || grad_scatter == NULL))
      return 1;
  }
#if DW_NDIM == 3
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2LL * FD_PAD, dimBlock.z);
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);

  dim3 const dimBlock_sourcessc(32, 1, 1);
  unsigned int const gridx_sourcessc =
      ceil_div(n_sourcessc_per_shot_h, dimBlock_sourcessc.x);
  unsigned int const gridy_sourcessc =
      ceil_div(n_shots_h, dimBlock_sourcessc.y);
  unsigned int const gridz_sourcessc = 1;
  dim3 const dimGrid_sourcessc(gridx_sourcessc, gridy_sourcessc,
                               gridz_sourcessc);
  dim3 const dimBlock_receiverssc(32, 1, 1);
  unsigned int const gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int const gridy_receiverssc =
      ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int const gridz_receiverssc = 1;
  dim3 const dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                                 gridz_receiverssc);

#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2LL * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2LL * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine =
      ceil_div(nz_h - 2LL * FD_PAD, dimBlock_combine.z);
#elif DW_NDIM == 2
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2LL * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2LL * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock_combine(128, 1, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2LL * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine = 1;
  unsigned int const gridz_combine = 1;
#endif
  dim3 const dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

  int64_t t;
  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device_guard.selected_device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, rdz2_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, rdy2_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, rdx2_h, nx_h, dt2_h, n_shots_h, n_sourcessc_per_shot_h,
        n_sourcessc_per_shot_h, n_receiverssc_per_shot_h,
        n_receiverssc_per_shot_h, step_ratio_h, pml_x0_h, pml_x1_h, v_batched_h,
        scatter_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      scatter_requires_grad;

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  if (use_double_buffering) {
    gpuErrchk(
        cudaStreamCreateWithFlags(&stream_storage, cudaStreamNonBlocking));
    gpuErrchk(cudaEventCreate(&event_storage_done_a));
    gpuErrchk(cudaEventCreate(&event_storage_done_b));
    gpuErrchk(cudaEventCreate(&event_compute_done_a));
    gpuErrchk(cudaEventCreate(&event_compute_done_b));
    gpuErrchk(cudaEventRecord(event_compute_done_a, stream_compute));
    gpuErrchk(cudaEventRecord(event_compute_done_b, stream_compute));
  }

  ScopedFile fp_w;
  file_group.add(fp_w);
  if (storage_mode == STORAGE_DISK) {
    if (scatter_requires_grad && !fp_w.open(w_filenames_ptr[0], "rb")) return 1;
  }

  // --- Time-reversed loop for adjoint propagation (scattered field only)
  // ---
  for (t = start_t - 1; t >= start_t - nt; --t) {
    bool const load_step = ((t % step_ratio_h) == 0);
    bool const load_w = load_step && scatter_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE* __restrict w_store_1_t =
        w_store_1a != NULL
            ? (w_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const w_store_2_t =
        w_store_2 != NULL
            ? ((uint8_t*)w_store_2 +
               (storage_mode == STORAGE_DEVICE && storage_compression
                    ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
                    : 0))
            : NULL;
    void* __restrict const w_store_3_t =
        w_store_3 != NULL
            ? ((uint8_t*)w_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) *
                          (storage_compression ? shot_bytes_comp
                                               : shot_bytes_uncomp) *
                          n_shots_h
                    : 0))
            : NULL;

    cudaEvent_t event_storage_done = event_storage_done_a;
    cudaEvent_t event_compute_done = event_compute_done_a;

    if (load_w && use_double_buffering) {
      if (step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
        event_storage_done = event_storage_done_b;
        event_compute_done = event_compute_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    // Record source gradients for scattered field
    if (n_sourcessc_per_shot_h > 0) {
      record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc, 0,
                                   stream_compute>>>(
          grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfcsc, sources_i);
      CHECK_KERNEL_ERROR
    }

    if (load_w) {
      if (STORAGE_FUNC(load_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
      if (use_double_buffering) {
        gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
        gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
      }
    }

    /* Adjoint guard fill: same mirror as the forward (self-adjoint on a
     * reflecting, pml_width=0 face). This scattered-only pass has no
     * background adjoint field to fill. */
    if (has_bc) LAUNCH_FILL_BC(wfcsc, 0, n_shots_h * shot_numel_h);

    // Launch backward kernel for this time step (scattered field only)
    backward_kernel_sc<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, wfcsc, wfpsc,
#if DW_NDIM >= 3
        psizsc, psiznsc, zetazsc, zetaznsc,
#endif
#if DW_NDIM >= 2
        psiysc, psiynsc, zetaysc, zetaynsc,
#endif
        psixsc, psixnsc, zetaxsc, zetaxnsc, w_store_1_t, grad_scatter_shot,
#if DW_NDIM >= 3
        az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        ay, by, dbydy,
#endif
        ax, bx, dbxdx, load_w);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Add receiver gradients as adjoint sources
    if (n_receiverssc_per_shot_h > 0) {
      add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc, 0,
                              stream_compute>>>(
          wfpsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
          receiverssc_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psizsc, psiznsc);
    std::swap(zetazsc, zetaznsc);
#endif
#if DW_NDIM >= 2
    std::swap(psiysc, psiynsc);
    std::swap(zetaysc, zetaynsc);
#endif
    std::swap(wfcsc, wfpsc);
    std::swap(psixsc, psixnsc);
    std::swap(zetaxsc, zetaxnsc);
  }
  // Reduce per-shot gradients to a single array if needed
  if (scatter_requires_grad && !scatter_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_scatter, grad_scatter_shot);
    CHECK_KERNEL_ERROR
  }

  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

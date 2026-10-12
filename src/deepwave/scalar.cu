/*
 * Scalar wave equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the scalar wave equation
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
 * For a description of the method, see the C implementation in scalar.c.
 * This file implements the same functionality, but for execution on a GPU
 * using CUDA.
 *
 * The CUDA kernels are launched with a grid of threads that is appropriate
 * for the number of dimensions. For 1D and 2D, one of the grid dimensions
 * is used for shots, allowing multiple shots to be propagated simultaneously.
 * For 3D, the three grid dimensions are used for the spatial dimensions,
 * and the kernel loops over shots.
 *
 * Constant memory is used to store the configuration parameters of the
 * propagator, which are the same for all threads. This is more efficient
 * than passing them as arguments to the kernels.
 */

#include <stdio.h>

#include <cstdint>

#include "common_gpu.h"
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
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx_h
#endif

/* ND_INDEX, WFC/V2DT2_WFC, PSIZ/ZETAZ/... (+Y, +X), the UT_TERM and PSI_TERM
 * family, and the FUNC(forward_cell)/FUNC(backward_cell) per-grid-point
 * stencil update functions live in scalar_kernels.h, shared verbatim with
 * scalar.c (see that header for the full explanation). wfc/wfp/psiz/.../
 * zetaz below (the arrays spanning all shots) are this file's names for the
 * arrays passed as the wfc/wfp/psiz/.../zetaz parameters of those
 * functions. */
#include "scalar_kernels.h"

namespace {
__constant__ DW_DTYPE dt2;
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
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;
__constant__ int64_t step_ratio;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool v_batched;

/* First-class reflecting/periodic boundary faces (see docs/boundaries.rst and
 * the scalar.c reference implementation, which this mirrors exactly). ``bc``
 * codes: 0=pml (guard untouched), 1=free (odd about node P-1), 2=rigid (even
 * about half-cell P-1/2), 3=periodic (wrap, period n-2P). ``is_model`` selects
 * the even/wrap no-sign no-zero variant used to mirror-pad v^2 into the guard
 * so the (self-adjoint) adjoint's coefficient reads match. */
__device__ inline void fill_bc_line_dev(DW_DTYPE *__restrict const wf,
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
__global__ void fill_bc_kernel(DW_DTYPE *__restrict const wf,
                               int64_t const n_lines, int64_t const nax,
                               int64_t const stride, int64_t const bc_lo,
                               int64_t const bc_hi, int const is_model) {
  int64_t const L = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  fill_bc_line_dev(wf, base, nax, stride, bc_lo, bc_hi, is_model);
}

/* Launch the guard fill on every reflecting axis, innermost stride first (the
 * order the C reference uses, so corner cells are filled identically: a later
 * axis reads the already-filled guard of the earlier axes). ``bc`` is the
 * face-order array [z0,z1,y0,y1,x0,x1]. Kernels serialise on ``stream``. */
static void launch_fill_bc(DW_DTYPE *__restrict const wf, int const is_model,
                           int64_t const bc[6], int64_t const total_numel,
                           int64_t const nz_, int64_t const ny_,
                           int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const n_lines = total_numel / nx_;
    fill_bc_kernel<<<ceil_div((unsigned int)n_lines, tpb), tpb, 0, stream>>>(
        wf, n_lines, nx_, 1, bc[4], bc[5], is_model);
  }
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const n_lines = total_numel / ny_;
    fill_bc_kernel<<<ceil_div((unsigned int)n_lines, tpb), tpb, 0, stream>>>(
        wf, n_lines, ny_, nx_, bc[2], bc[3], is_model);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const n_lines = total_numel / nz_;
    fill_bc_kernel<<<ceil_div((unsigned int)n_lines, tpb), tpb, 0, stream>>>(
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
    void add_sources(DW_DTYPE *__restrict const wf,
                     DW_DTYPE const *__restrict const f,
                     int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void record_receivers(DW_DTYPE *__restrict const r,
                          DW_DTYPE const *__restrict const wf,
                          int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

#if DW_NDIM == 3
__launch_bounds__(128) __global__
    void combine_grad_v(DW_DTYPE *__restrict const grad_v,
                        DW_DTYPE const *__restrict const grad_v_shot) {
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad_v[i] += grad_v_shot[shot_idx * shot_numel + i];
    }
  }
}
#elif DW_NDIM == 2
__launch_bounds__(128) __global__
    void combine_grad_v(DW_DTYPE *__restrict const grad_v,
                        DW_DTYPE const *__restrict const grad_v_shot) {
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad_v[i] += grad_v_shot[shot_idx * shot_numel + i];
    }
  }
}
#else /* DW_NDIM == 1 */
__launch_bounds__(128) __global__
    void combine_grad_v(DW_DTYPE *__restrict const grad_v,
                        DW_DTYPE const *__restrict const grad_v_shot) {
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  if (x < nx - FD_PAD) {
    int64_t const i = x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad_v[i] += grad_v_shot[shot_idx * shot_numel + i];
    }
  }
}
#endif

/* Thread-block shape and occupancy target for the two stencil kernels below.
 *
 * In 1-D and 2-D the defaults are already right: ptxas allocates 27-86
 * registers per thread across the whole accuracy/dtype matrix there, so warp
 * slots rather than registers bound occupancy, and a launch-geometry sweep on
 * an RTX A4000 (sm_86) and an RTX 4000 Ada (sm_89) measured no gain from
 * changing either the block shape or the bound.
 *
 * The 3-D kernels are different. They carry a set of z taps on top of the y
 * and x ones, and a 128-thread bound leaves ptxas free to spend up to 255
 * registers per thread: at accuracy 4 and above it takes 94-128 for
 * forward_kernel and 164-168 for backward_kernel, which limits an sm_86 SM to
 * 25-42% of its warp slots (12-20 of 48 resident warps). These
 * kernels are memory-latency bound -- one interior grid point issues 13
 * (accuracy 4) to 25 (accuracy 8) wavefield loads -- so too few resident warps
 * cannot keep enough of those loads in flight to cover a DRAM round trip. A
 * 32x4x4 block with a matching 512-thread bound caps the allocation at 128
 * registers (ptxas then settles on 40-104) and additionally gives the four z
 * planes of a block a shared L1 working set. Measured on 256^3 float32, PML
 * width 10, one shot, median of 3 reps on an otherwise idle GPU, kernel times
 * from Nsight Systems (before -> after):
 *
 *                            RTX A4000            RTX 4000 Ada
 *   accuracy 4  forward    1900 -> 1502 us      1885 -> 1586 us
 *               backward   4107 -> 2698 us      3502 -> 2781 us
 *   accuracy 6  forward    2525 -> 1910 us      2423 -> 2024 us
 *               backward   5209 -> 3052 us      3571 -> 3035 us
 *   accuracy 8  forward    2369 -> 1856 us      2063 -> 1802 us
 *               backward   6185 -> 4064 us      3867 -> 3713 us
 *
 * Accuracy 2 keeps the 1-D/2-D configuration: its 3-D forward kernel already
 * fits in 72 registers at the default bound, and the wider block measured 45%
 * slower there -- that stencil is too small to pay for the coarser block. */
#if DW_NDIM == 3 && DW_ACCURACY >= 4
#define DW_STENCIL_BLOCK_3D 32, 4, 4
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(512)
#else
#define DW_STENCIL_BLOCK_3D 32, 4, 1
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(128)
#endif

DW_STENCIL_LAUNCH_BOUNDS __global__ void forward_kernel(
    DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const wfc,
    DW_DTYPE *__restrict const wfp, DW_DTYPE *__restrict const dwdv,
#if DW_NDIM >= 3
    /* z-dimension */
    DW_DTYPE const *__restrict const psiz, DW_DTYPE *__restrict const psizn,
    DW_DTYPE *__restrict const zetaz, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    /* y-dimension */
    DW_DTYPE const *__restrict const psiy, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const zetay, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const dbydy,
#endif
    /* x-dimension */
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psixn,
    DW_DTYPE *__restrict const zetax, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbxdx,
    bool const v_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity for this shot/grid point
      DW_DTYPE const v_shot = v_batched ? v[i] : v[j];
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      FUNC(forward_cell)(wfc, wfp, dwdv, v_shot, dt2, v_requires_grad, i,
#if DW_NDIM >= 3
                         ny,
#endif
#if DW_NDIM >= 2
                         nx,
#endif
#if DW_NDIM >= 3
                         z, pml_z, psiz, psizn, zetaz, az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
                         y, pml_y, psiy, psiyn, zetay, ay, by, dbydy, rdy, rdy2,
#endif
                         x, pml_x, psix, psixn, zetax, ax, bx, dbxdx, rdx,
                         rdx2);
#if DW_NDIM == 3
    }
#endif
  }
}

DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel(
    DW_DTYPE const *__restrict const v2dt2,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const dwdv, DW_DTYPE *__restrict const grad_v,
#if DW_NDIM >= 3
    /* z-dimension */
    DW_DTYPE const *__restrict const psiz, DW_DTYPE *__restrict const psizn,
    DW_DTYPE *__restrict const zetaz, DW_DTYPE *__restrict const zetazn,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    /* y-dimension */
    DW_DTYPE const *__restrict const psiy, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetayn,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const dbydy,
#endif
    /* x-dimension */
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psixn,
    DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const zetaxn,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbxdx, bool const v_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select v^2*dt^2 for this shot/grid point
      DW_DTYPE const *__restrict const v2dt2_shot =
          v_batched ? v2dt2 + shot_idx * shot_numel : v2dt2;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      FUNC(backward_cell)(
          wfc, wfp, v2dt2_shot, dwdv, grad_v, v_requires_grad, step_ratio, i, j,
#if DW_NDIM >= 3
          ny,
#endif
#if DW_NDIM >= 2
          nx,
#endif
#if DW_NDIM >= 3
          z, pml_z, psiz, psizn, zetaz, zetazn, az, bz, dbzdz, rdz, rdz2,
#endif
#if DW_NDIM >= 2
          y, pml_y, psiy, psiyn, zetay, zetayn, ay, by, dbydy, rdy, rdy2,
#endif
          x, pml_x, psix, psixn, zetax, zetaxn, ax, bx, dbxdx, rdx, rdx2);
#if DW_NDIM == 3
    }
#endif
  }
}

int set_config(
#if DW_NDIM >= 3
    /* z-dimension */
    DW_DTYPE const rdz_h, DW_DTYPE const rdz2_h, int64_t const nz_h,
    int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
    /* y-dimension */
    DW_DTYPE const rdy_h, DW_DTYPE const rdy2_h, int64_t const ny_h,
    int64_t const pml_y0_h, int64_t const pml_y1_h,
#endif
    /* x-dimension */
    DW_DTYPE const rdx_h, DW_DTYPE const rdx2_h, int64_t const nx_h,
    int64_t const shot_numel_h, int64_t const pml_x0_h, int64_t const pml_x1_h,
    /* other */
    DW_DTYPE const dt2_h, int64_t const n_shots_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h, bool const v_batched_h) {
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
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(v_batched, &v_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(forward)(
        DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const f,
        DW_DTYPE *__restrict wfc, DW_DTYPE *__restrict wfp,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict psiz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict psiy,
#endif
        DW_DTYPE *__restrict psix,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict psizn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict psiyn,
#endif
        DW_DTYPE *__restrict psixn,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const zetaz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const zetay,
#endif
        DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const w_store_1a,
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
        int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const v_batched_h, bool const storage_compression,
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
        int64_t const bc_x1, DW_DTYPE *__restrict const raw_store_1a,
        DW_DTYPE *__restrict const raw_store_1b,
        void *__restrict const raw_store_2, void *__restrict const raw_store_3,
        char const *__restrict const *__restrict const raw_filenames_ptr,
        bool const store_raw_forward, int64_t const raw_storage_mode,
        size_t const raw_shot_bytes_uncomp, size_t const raw_shot_bytes_comp,
        bool const raw_storage_compression, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;

    if ((n_receivers_per_shot_h > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot_h > 0) && sources_i == NULL) return 1;
    bool const dw_has_sample = (start_t % step_ratio_h == 0 ||
                                nt > step_ratio_h - start_t % step_ratio_h);
    if (dw_has_sample && (v_requires_grad) && (w_store_1a == NULL)) return 1;
    if (dw_has_sample &&
        (store_raw_forward && raw_storage_mode != STORAGE_NONE) &&
        (raw_store_1a == NULL))
      return 1;
  }
  /* raw_store_1b is unused on the GPU forward path: the raw-forward
   * checkpoint (opt-in, see CallbackState role="forward") never
   * double-buffers across steps (unlike the primary checkpoint's async
   * overlap pipeline below) -- correctness over throughput for this
   * opt-in, callback-only buffer. raw_store_2/raw_store_3/
   * raw_filenames_ptr ARE used below, for compression and/or CPU/disk
   * storage. */
  (void)raw_store_1b;
#if DW_NDIM == 3
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
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
  ScopedCudaConfig config_guard(device);
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
        rdx_h, rdx2_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, dt2_h, n_shots_h,
        n_sources_per_shot_h, n_receivers_per_shot_h, step_ratio_h,
        v_batched_h);
    if (err != 0) return err;
  }
  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      v_requires_grad;

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
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad && !fp_w.open(w_filenames_ptr[0], "ab")) return 1;
  }

  ScopedFile fp_raw;
  file_group.add(fp_raw);
  if (raw_storage_mode == STORAGE_DISK) {
    if (store_raw_forward && !fp_raw.open(raw_filenames_ptr[0], "ab")) return 1;
  }
  bool const raw_is_full_history =
      raw_storage_mode == STORAGE_DEVICE && !raw_storage_compression;

  for (t = start_t; t < start_t + nt; ++t) {
    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_w = store_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE *w_store_1_t =
        w_store_1a != NULL
            ? (w_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? step_idx * shot_numel_h * n_shots_h
                    : 0))
            : NULL;
    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;

    if (store_w && use_double_buffering) {
      if (step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    /* Opt-in raw-forward checkpoint (CallbackState role="forward"): copy the
     * field as it stands entering this outer step -- before the guard-ring
     * fill below and before this step's update -- so it is time-coincident
     * with the adjoint field the backward_callback sees at the same step.
     * Uses the SAME storage configuration as the primary checkpoint (see
     * scalar.py): the DEVICE-uncompressed fast path below is a plain
     * device-to-device copy into its permanent per-step slot, unchanged
     * from before this buffer honoured storage_mode. Every other mode/
     * compression combination instead stages the step into a single-step
     * buffer (never double-buffered, unlike the primary checkpoint above --
     * correctness over throughput for this opt-in, callback-only buffer) and
     * persists it via the same STORAGE_FUNC helper the primary checkpoint
     * uses; FUNC(backward) below reloads it immediately before each chunk's
     * backward_callback fires. All shots are contiguous, so one copy covers
     * every shot either way. storage_mode="none" allocates raw_store_1a/1b/2
     * as zero-size placeholders (StorageManager.allocate is a no-op in that
     * mode, matching the primary checkpoint), so it must be excluded here
     * too -- writing into it would be an out-of-bounds write, not merely a
     * wrong value. Python's forward_wavefields_for_step already returns
     * None for that mode's steps rather than reading it back. */
    if (store_raw_forward && store_step && raw_storage_mode != STORAGE_NONE) {
      if (raw_is_full_history) {
        gpuErrchk(cudaMemcpyAsync(
            raw_store_1a + step_idx * n_shots_h * shot_numel_h, wfc,
            (size_t)n_shots_h * shot_numel_h * sizeof(DW_DTYPE),
            cudaMemcpyDeviceToDevice, stream_compute));
      } else {
        void *const raw_store_2_t =
            raw_store_2 != NULL
                ? ((uint8_t *)raw_store_2 +
                   (raw_storage_mode == STORAGE_DEVICE
                        ? step_idx * raw_shot_bytes_comp * n_shots_h
                        : 0))
                : NULL;
        void *const raw_store_3_t =
            raw_store_3 != NULL ? ((uint8_t *)raw_store_3 +
                                   (raw_storage_mode == STORAGE_CPU
                                        ? step_idx *
                                              (raw_storage_compression
                                                   ? raw_shot_bytes_comp
                                                   : raw_shot_bytes_uncomp) *
                                              n_shots_h
                                        : 0))
                                : NULL;
        gpuErrchk(
            cudaMemcpyAsync(raw_store_1a, wfc,
                            (size_t)n_shots_h * shot_numel_h * sizeof(DW_DTYPE),
                            cudaMemcpyDeviceToDevice, stream_compute));
        if (STORAGE_FUNC(save_snapshot_gpu)(
                raw_store_1a, raw_store_2_t, raw_store_3_t, fp_raw,
                raw_storage_mode, raw_storage_compression, step_idx,
                raw_shot_bytes_uncomp, raw_shot_bytes_comp, n_shots_h, DIM_ARGS,
                stream_compute) != 0)
          return 1;
      }
    }

    /* Refill the guard ring with the boundary image before the interior
     * stencil reads it (no-op when every face is PML). */
    if (has_bc) LAUNCH_FILL_BC(wfc, 0, n_shots_h * shot_numel_h);

    forward_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, wfc, wfp, w_store_1_t,
#if DW_NDIM >= 3
        psiz, psizn, zetaz, az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        psiy, psiyn, zetay, ay, by, dbydy,
#endif
        psix, psixn, zetax, ax, bx, dbxdx, store_w);
    CHECK_KERNEL_ERROR

    if (store_w) {
      void *const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t *)w_store_2 +
                 (storage_mode == STORAGE_DEVICE && storage_compression
                      ? step_idx * shot_bytes_comp * n_shots_h
                      : 0))
              : NULL;
      void *const w_store_3_t =
          w_store_3 != NULL
              ? ((uint8_t *)w_store_3 +
                 (storage_mode == STORAGE_CPU
                      ? step_idx *
                            (storage_compression ? shot_bytes_comp
                                                 : shot_bytes_uncomp) *
                            n_shots_h
                      : 0))
              : NULL;
      if (use_double_buffering) {
        gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
        gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
      }
      if (STORAGE_FUNC(save_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
      if (use_double_buffering) {
        gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      }
    }

    if (n_sources_per_shot_h > 0) {
      add_sources<<<dimGrid_sources, dimBlock_sources, 0, stream_compute>>>(
          wfp, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers<<<dimGrid_receivers, dimBlock_receivers, 0,
                         stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, wfc, receivers_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psiz, psizn);
#endif
#if DW_NDIM >= 2
    std::swap(psiy, psiyn);
#endif
    std::swap(wfc, wfp);
    std::swap(psix, psixn);
  }
  if (use_double_buffering) {
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
  }

  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(backward)(
        DW_DTYPE const *__restrict const v2dt2,
        DW_DTYPE const *__restrict const grad_r, DW_DTYPE *__restrict wfc,
        DW_DTYPE *__restrict wfp,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict psiz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict psiy,
#endif
        DW_DTYPE *__restrict psix,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict psizn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict psiyn,
#endif
        DW_DTYPE *__restrict psixn,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict zetaz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict zetay,
#endif
        DW_DTYPE *__restrict zetax,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict zetazn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict zetayn,
#endif
        DW_DTYPE *__restrict zetaxn, DW_DTYPE *__restrict const w_store_1a,
        DW_DTYPE *__restrict const w_store_1b, void *__restrict const w_store_2,
        void *__restrict const w_store_3,
        char const *__restrict const *__restrict const w_filenames_ptr,
        DW_DTYPE *__restrict const grad_f, DW_DTYPE *__restrict const grad_v,
        DW_DTYPE *__restrict const grad_v_shot,
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
        DW_DTYPE const rdx2_h, int64_t const nt, int64_t const n_shots_h,
#if DW_NDIM >= 3
        int64_t const nz_h,
#endif
#if DW_NDIM >= 2
        int64_t const ny_h,
#endif
        int64_t const nx_h, int64_t const n_sources_per_shot_h,
        int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const v_requires_grad,
        bool const v_batched_h, bool const storage_compression,
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
        int64_t const bc_x1, DW_DTYPE *__restrict const raw_store_1a,
        DW_DTYPE *__restrict const raw_store_1b,
        void *__restrict const raw_store_2, void *__restrict const raw_store_3,
        char const *__restrict const *__restrict const raw_filenames_ptr,
        bool const store_raw_forward, int64_t const raw_storage_mode,
        size_t const raw_shot_bytes_uncomp, size_t const raw_shot_bytes_comp,
        bool const raw_storage_compression, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;

    if ((n_receivers_per_shot_h > 0) && (receivers_i == NULL)) return 1;
    if ((n_sources_per_shot_h > 0) && sources_i == NULL) return 1;
    bool const dw_has_sample = ((start_t - 1) % step_ratio_h < nt);
    if (dw_has_sample && (v_requires_grad) &&
        (w_store_1a == NULL || grad_v == NULL))
      return 1;
  }
  /* The raw-forward checkpoint (CallbackState role="forward") is written by
   * FUNC(forward). For its DEVICE-uncompressed fast path it is read back
   * directly from Python by step index (it is DEVICE-resident and outlives
   * this call), so this backward pass still never touches raw_store_1a in
   * that case; raw_store_1b is unused here for the same reason FUNC(forward)
   * leaves it unused (see its comment: this buffer never double-buffers).
   * For every other mode/compression combination, raw_store_1a/raw_store_2/
   * raw_store_3/raw_filenames_ptr ARE used below to reload (and, if
   * compressed, decompress) the single-step staging buffer the
   * backward_callback needs, mirroring the primary checkpoint's own
   * per-step load. */
  (void)raw_store_1b;
#if DW_NDIM == 3
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
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
#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine =
      ceil_div(nz_h - 2 * FD_PAD, dimBlock_combine.z);
#elif DW_NDIM == 2
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock_combine(128, 1, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
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
  ScopedCudaConfig config_guard(device);
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
        rdx_h, rdx2_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, 0, n_shots_h,
        n_receivers_per_shot_h, n_sources_per_shot_h, step_ratio_h,
        v_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      v_requires_grad;

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
    if (v_requires_grad && !fp_w.open(w_filenames_ptr[0], "rb")) return 1;
  }

  ScopedFile fp_raw;
  file_group.add(fp_raw);
  if (raw_storage_mode == STORAGE_DISK) {
    if (store_raw_forward && !fp_raw.open(raw_filenames_ptr[0], "rb")) return 1;
  }
  bool const raw_is_full_history =
      raw_storage_mode == STORAGE_DEVICE && !raw_storage_compression;

  /* Mirror-pad the model coefficients (v^2 * dt^2) into the guard ring on
   * reflecting faces so the adjoint's V2DT2_WFC neighbour reads reproduce the
   * exact transpose of the forward's mirror fill (the mirror-filled operator is
   * self-adjoint). v2dt2 is a fresh per-call temporary, so writing its guard
   * cells in place is safe. Done once (constant across time steps). */
  if (has_bc)
    LAUNCH_FILL_BC((DW_DTYPE *)v2dt2, 1,
                   (v_batched_h ? n_shots_h : 1) * shot_numel_h);

  for (t = start_t - 1; t >= start_t - nt; --t) {
    bool const load_step = (t % step_ratio_h) == 0;
    bool const load_w = load_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE *__restrict w_store_1_t =
        w_store_1a != NULL
            ? (w_store_1a +
               (storage_mode == STORAGE_DEVICE && !storage_compression
                    ? step_idx * shot_numel_h * n_shots_h
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

    if (n_sources_per_shot_h > 0) {
      record_receivers<<<dimGrid_sources, dimBlock_sources, 0,
                         stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, wfc, sources_i);
      CHECK_KERNEL_ERROR
    }

    if (load_w) {
      void *const w_store_2_t =
          w_store_2 != NULL
              ? ((uint8_t *)w_store_2 +
                 (storage_mode == STORAGE_DEVICE && storage_compression
                      ? step_idx * shot_bytes_comp * n_shots_h
                      : 0))
              : NULL;
      void *const w_store_3_t =
          w_store_3 != NULL
              ? ((uint8_t *)w_store_3 +
                 (storage_mode == STORAGE_CPU
                      ? step_idx *
                            (storage_compression ? shot_bytes_comp
                                                 : shot_bytes_uncomp) *
                            n_shots_h
                      : 0))
              : NULL;
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

    if (store_raw_forward && load_step && !raw_is_full_history &&
        raw_storage_mode != STORAGE_NONE) {
      /* CPU/disk-mode (or compressed DEVICE-mode) raw-forward checkpoint:
       * reload (and, if compressed, decompress) the step this chunk's
       * backward_callback is about to fire for into raw_store_1a's
       * single-step staging slot -- see FUNC(forward) above (which wrote
       * it) and scalar.py's forward_wavefields_for_step (which reads it
       * back once this call returns). Reloaded at every step boundary in
       * the chunk regardless of v_requires_grad (this checkpoint exists
       * purely for the callback, independent of whether the model needs a
       * gradient), so the buffer holds the correct value regardless of
       * callback_frequency; only the LAST (lowest t) load in a chunk is
       * ever actually read. Always on stream_compute (never double-
       * buffered), so subsequent same-stream reads -- including from
       * Python once this call returns -- are correctly ordered after it
       * without an explicit host sync. storage_mode="none" is excluded
       * (nothing was ever persisted; see FUNC(forward) above and
       * forward_wavefields_for_step in scalar.py, which returns None for
       * its steps instead of reading this buffer). */
      void *const raw_store_2_t =
          raw_store_2 != NULL
              ? ((uint8_t *)raw_store_2 +
                 (raw_storage_mode == STORAGE_DEVICE
                      ? step_idx * raw_shot_bytes_comp * n_shots_h
                      : 0))
              : NULL;
      void *const raw_store_3_t =
          raw_store_3 != NULL
              ? ((uint8_t *)raw_store_3 +
                 (raw_storage_mode == STORAGE_CPU
                      ? step_idx *
                            (raw_storage_compression ? raw_shot_bytes_comp
                                                     : raw_shot_bytes_uncomp) *
                            n_shots_h
                      : 0))
              : NULL;
      if (STORAGE_FUNC(load_snapshot_gpu)(
              raw_store_1a, raw_store_2_t, raw_store_3_t, fp_raw,
              raw_storage_mode, raw_storage_compression, step_idx,
              raw_shot_bytes_uncomp, raw_shot_bytes_comp, n_shots_h, DIM_ARGS,
              stream_compute) != 0)
        return 1;
    }

    /* Adjoint guard fill: same mirror as the forward (self-adjoint). */
    if (has_bc) LAUNCH_FILL_BC(wfc, 0, n_shots_h * shot_numel_h);

    backward_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v2dt2, wfc, wfp, w_store_1_t, grad_v_shot,
#if DW_NDIM >= 3
        psiz, psizn, zetaz, zetazn, az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        psiy, psiyn, zetay, zetayn, ay, by, dbydy,
#endif
        psix, psixn, zetax, zetaxn, ax, bx, dbxdx, load_w);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    if (n_receivers_per_shot_h > 0) {
      add_sources<<<dimGrid_receivers, dimBlock_receivers, 0, stream_compute>>>(
          wfp, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psiz, psizn);
    std::swap(zetaz, zetazn);
#endif
#if DW_NDIM >= 2
    std::swap(psiy, psiyn);
    std::swap(zetay, zetayn);
#endif
    std::swap(wfc, wfp);
    std::swap(psix, psixn);
    std::swap(zetax, zetaxn);
  }
  if (v_requires_grad && !v_batched_h && n_shots_h > 1) {
    combine_grad_v<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_v, grad_v_shot);
    CHECK_KERNEL_ERROR
  }
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

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
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz)*ny * nx + (dy)*nx + (dx))
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy)*nx + (dx))
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#define DIM_ARGS nx_h
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

__launch_bounds__(128) __global__ void forward_kernel(
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
      DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
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
      bool const pml_y = y < pml_y0 || y >= pml_y1;
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
      bool const pml_x = x < pml_x0 || x >= pml_x1;
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
      // If gradients are required, store snapshot for backward
      if (v_requires_grad) {
        dwdv[i] = 2 * v_shot * dt2 * w_sum;
      }
      // Main wavefield update: finite difference time stepping
      wfp[i] = v_shot * v_shot * dt2 * w_sum + 2 * wfc[i] - wfp[i];
#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel(
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
      DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      w_sum +=
          pml_z ? -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2) : DIFFZ2(V2DT2_WFC);
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
      w_sum +=
          pml_y ? -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2) : DIFFY2(V2DT2_WFC);
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      w_sum +=
          pml_x ? -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2) : DIFFX2(V2DT2_WFC);
      // Main adjoint wavefield update
      wfp[i] = w_sum + 2 * wfc[i] - wfp[i];
#if DW_NDIM >= 3
      if (pml_z) {
        psizn[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz[i];
        zetazn[i] = az[z] * V2DT2_WFC(0, 0, 0) + az[z] * zetaz[i];
      }
#endif
#if DW_NDIM >= 2
      if (pml_y) {
        psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
        zetayn[i] = ay[y] * V2DT2_WFC(0, 0, 0) + ay[y] * zetay[i];
      }
#endif
      if (pml_x) {
        psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
        zetaxn[i] = ax[x] * V2DT2_WFC(0, 0, 0) + ax[x] * zetax[i];
      }
      // If gradients are required, accumulate gradient for velocity
      if (v_requires_grad) {
        grad_v[i] += wfc[i] * dwdv[i] * step_ratio;
      }
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
        int FUNC(forward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const f, DW_DTYPE *__restrict wfc,
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
            DW_DTYPE *__restrict const zetaz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const zetay,
#endif
            DW_DTYPE *__restrict const zetax,
            DW_DTYPE *__restrict const w_store_1a,
            DW_DTYPE *__restrict const w_store_1b,
            void *__restrict const w_store_2, void *__restrict const w_store_3,
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
            int64_t const pml_x1_h, int64_t const device,
            cudaStream_t stream_compute) {
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
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
  gpuErrchk(cudaSetDevice(device));
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
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad) fp_w.open(w_filenames_ptr[0], "ab");
  }

  for (t = start_t; t < start_t + nt; ++t) {
    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_w = store_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE *w_store_1_t =
        w_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                          ? step_idx * shot_numel_h * n_shots_h
                          : 0);
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
          (uint8_t *)w_store_2 +
          (storage_mode == STORAGE_DEVICE && storage_compression
               ? step_idx * shot_bytes_comp * n_shots_h
               : 0);
      void *const w_store_3_t =
          (uint8_t *)w_store_3 +
          (storage_mode == STORAGE_CPU
               ? step_idx *
                     (storage_compression ? shot_bytes_comp
                                          : shot_bytes_uncomp) *
                     n_shots_h
               : 0);
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

  return 0;
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        int FUNC(backward)(
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
            DW_DTYPE *__restrict const w_store_1b,
            void *__restrict const w_store_2, void *__restrict const w_store_3,
            char const *__restrict const *__restrict const w_filenames_ptr,
            DW_DTYPE *__restrict const grad_f,
            DW_DTYPE *__restrict const grad_v,
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
            int64_t const pml_x1_h, int64_t const device,
            cudaStream_t stream_compute) {
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
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
  gpuErrchk(cudaSetDevice(device));
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
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad) fp_w.open(w_filenames_ptr[0], "rb");
  }

  for (t = start_t - 1; t >= start_t - nt; --t) {
    bool const load_step = (t % step_ratio_h) == 0;
    bool const load_w = load_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE *__restrict w_store_1_t =
        w_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                          ? step_idx * shot_numel_h * n_shots_h
                          : 0);
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
          (uint8_t *)w_store_2 +
          (storage_mode == STORAGE_DEVICE && storage_compression
               ? step_idx * shot_bytes_comp * n_shots_h
               : 0);
      void *const w_store_3_t =
          (uint8_t *)w_store_3 +
          (storage_mode == STORAGE_CPU
               ? step_idx *
                     (storage_compression ? shot_bytes_comp
                                          : shot_bytes_uncomp) *
                     n_shots_h
               : 0);
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
  return 0;
}

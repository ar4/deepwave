/*
 * Scalar wave equation propagator
 *
 * See scalar.c
 */

#include <stdio.h>

#include <cstdint>

#include "common.h"

#define CAT_I(name, accuracy, dtype) scalar_iso_##accuracy##_##dtype##_##name
#define CAT(name, accuracy, dtype) CAT_I(name, accuracy, dtype)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE)

#define WFC(dy, dx) wfc[i + dy * nx + dx]
#define AY_PSIY(dy, dx) ay[y + dy] * psiy[i + dy * nx + dx]
#define AX_PSIX(dy, dx) ax[x + dx] * psix[i + dy * nx + dx]
#define V2DT2_WFC(dy, dx) v2dt2[j + dy * nx + dx] * wfc[i + dy * nx + dx]
#define UT_TERMY1(dy, dx)                                           \
  ((dbydy[y + dy] *                                                 \
        ((1 + by[y + dy]) * v2dt2[j + dy * nx] * wfc[i + dy * nx] + \
         by[y + dy] * zetay[i + dy * nx]) +                         \
    by[y + dy] * psiy[i + dy * nx]))
#define UT_TERMX1(dy, dx)                                             \
  ((dbxdx[x + dx] * ((1 + bx[x + dx]) * v2dt2[j + dx] * wfc[i + dx] + \
                     bx[x + dx] * zetax[i + dx]) +                    \
    bx[x + dx] * psix[i + dx]))
#define UT_TERMY2(dy, dx)                                      \
  ((1 + by[y + dy]) *                                          \
   ((1 + by[y + dy]) * v2dt2[j + dy * nx] * wfc[i + dy * nx] + \
    by[y + dy] * zetay[i + dy * nx]))
#define UT_TERMX2(dy, dx)                                               \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * v2dt2[j + dx] * wfc[i + dx] + \
                       bx[x + dx] * zetax[i + dx]))
#define PSIY_TERM(dy, dx)                                     \
  ((1 + by[y + dy]) * v2dt2[j + dy * nx] * wfc[i + dy * nx] + \
   by[y + dy] * zetay[i + dy * nx])
#define PSIX_TERM(dy, dx) \
  ((1 + bx[x + dx]) * v2dt2[j + dx] * wfc[i + dx] + bx[x + dx] * zetax[i + dx])

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

namespace {
__constant__ DW_DTYPE dt2;
__constant__ DW_DTYPE rdy;
__constant__ DW_DTYPE rdx;
__constant__ DW_DTYPE rdy2;
__constant__ DW_DTYPE rdx2;
__constant__ int64_t n_shots;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t nynx;
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;
__constant__ int64_t step_ratio;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;

__global__ void add_sources(DW_DTYPE *__restrict const wf,
                            DW_DTYPE const *__restrict const f,
                            int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void record_receivers(DW_DTYPE *__restrict const r,
                                 DW_DTYPE const *__restrict const wf,
                                 int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

__global__ void combine_grad_v(DW_DTYPE *__restrict const grad_v,
                               DW_DTYPE const *__restrict const grad_v_shot) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t i = y * nx + x;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx;
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad_v[i] += grad_v_shot[shot_idx * nynx + i];
    }
  }
}

__global__ void forward_kernel(
    DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const wfc,
    DW_DTYPE *__restrict const wfp, DW_DTYPE const *__restrict const psiy,
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const dwdv,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, bool const v_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    DW_DTYPE w_sum;
    if (!pml_y) {
      w_sum = DIFFY2(WFC);
    } else {
      DW_DTYPE dwfcdy = DIFFY1(WFC);
      DW_DTYPE tmpy =
          ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
      w_sum = (1 + by[y]) * tmpy + ay[y] * zetay[i];
      psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];
      zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
    }
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
      dwdv[i] = 2 * v[j] * dt2 * w_sum;
    }
    wfp[i] = v[j] * v[j] * dt2 * w_sum + 2 * wfc[i] - wfp[i];
  }
}

__global__ void backward_kernel(
    DW_DTYPE const *__restrict const v2dt2,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const psiy,
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const zetayn,
    DW_DTYPE *__restrict const zetaxn, DW_DTYPE const *__restrict const dwdv,
    DW_DTYPE *__restrict const grad_v, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, bool const v_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    wfp[i] =
        (pml_y ? -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2) : DIFFY2(V2DT2_WFC)) +
        (pml_x ? -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2) : DIFFX2(V2DT2_WFC)) +
        2 * wfc[i] - wfp[i];
    if (pml_y) {
      psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
      zetayn[i] = ay[y] * v2dt2[j] * wfc[i] + ay[y] * zetay[i];
    }
    if (pml_x) {
      psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
      zetaxn[i] = ax[x] * v2dt2[j] * wfc[i] + ax[x] * zetax[i];
    }
    if (v_requires_grad) {
      grad_v[i] += wfc[i] * dwdv[i] * step_ratio;
    }
  }
}

inline void gpuAssert(cudaError_t code, const char *file, bool line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

inline unsigned int ceil_div(unsigned int numerator, unsigned int denominator) {
  return (numerator + denominator - 1) / denominator;
}

void set_config(DW_DTYPE const dt2_h, DW_DTYPE const rdy_h,
                DW_DTYPE const rdx_h, DW_DTYPE const rdy2_h,
                DW_DTYPE const rdx2_h, int64_t const n_shots_h,
                int64_t const ny_h, int64_t const nx_h,
                int64_t const n_sources_per_shot_h,
                int64_t const n_receivers_per_shot_h,
                int64_t const step_ratio_h, int64_t const pml_y0_h,
                int64_t const pml_y1_h, int64_t const pml_x0_h,
                int64_t const pml_x1_h) {
  int64_t const nynx_h = ny_h * nx_h;
  gpuErrchk(cudaMemcpyToSymbol(dt2, &dt2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy2, &rdy2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx2, &rdx2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nynx, &nynx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(forward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const f, DW_DTYPE *__restrict const wfc,
            DW_DTYPE *__restrict const wfp, DW_DTYPE *__restrict const psiy,
            DW_DTYPE *__restrict const psix, DW_DTYPE *__restrict const psiyn,
            DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
            DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const dwdv,
            DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbydy,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i, DW_DTYPE const rdy_h,
            DW_DTYPE const rdx_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            DW_DTYPE const dt2_h, int64_t const nt, int64_t const n_shots_h,
            int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_per_shot_h,
            int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
            bool const v_requires_grad, int64_t const pml_y0_h,
            int64_t const pml_y1_h, int64_t const pml_x0_h,
            int64_t const pml_x1_h, int64_t const device) {
  dim3 dimBlock(32, 32, 1);
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);
  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt2_h, rdy_h, rdx_h, rdy2_h, rdx2_h, n_shots_h, ny_h, nx_h,
             n_sources_per_shot_h, n_receivers_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h);
  for (t = 0; t < nt; ++t) {
    if (t & 1) {
      forward_kernel<<<dimGrid, dimBlock>>>(
          v, wfp, wfc, psiyn, psixn, psiy, psix, zetay, zetax,
          dwdv + (t / step_ratio_h) * ny_h * nx_h * n_shots_h, ay, ax, by, bx,
          dbydy, dbxdx, v_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      if (n_sources_per_shot_h > 0) {
        add_sources<<<dimGrid_sources, dimBlock_sources>>>(
            wfc, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
        CHECK_KERNEL_ERROR
      }
      if (n_receivers_per_shot_h > 0) {
        record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
            r + t * n_shots_h * n_receivers_per_shot_h, wfp, receivers_i);
        CHECK_KERNEL_ERROR
      }
    } else {
      forward_kernel<<<dimGrid, dimBlock>>>(
          v, wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax,
          dwdv + (t / step_ratio_h) * ny_h * nx_h * n_shots_h, ay, ax, by, bx,
          dbydy, dbxdx, v_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      if (n_sources_per_shot_h > 0) {
        add_sources<<<dimGrid_sources, dimBlock_sources>>>(
            wfp, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
        CHECK_KERNEL_ERROR
      }
      if (n_receivers_per_shot_h > 0) {
        record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
            r + t * n_shots_h * n_receivers_per_shot_h, wfc, receivers_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(backward)(
            DW_DTYPE const *__restrict const v2dt2,
            DW_DTYPE const *__restrict const grad_r,
            DW_DTYPE *__restrict const wfc, DW_DTYPE *__restrict const wfp,
            DW_DTYPE *__restrict const psiy, DW_DTYPE *__restrict const psix,
            DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn,
            DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax,
            DW_DTYPE *__restrict const zetayn,
            DW_DTYPE *__restrict const zetaxn,
            DW_DTYPE const *__restrict const dwdv,
            DW_DTYPE *__restrict const grad_f,
            DW_DTYPE *__restrict const grad_v,
            DW_DTYPE *__restrict const grad_v_shot,
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbydy,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i, DW_DTYPE const rdy_h,
            DW_DTYPE const rdx_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            int64_t const nt, int64_t const n_shots_h, int64_t const ny_h,
            int64_t const nx_h, int64_t const n_sources_per_shot_h,
            int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
            bool const v_requires_grad, int64_t const pml_y0_h,
            int64_t const pml_y1_h, int64_t const pml_x0_h,
            int64_t const pml_x1_h, int64_t const device) {
  dim3 dimBlock(32, 16, 1);
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);
  dim3 dimBlock_combine(32, 32, 1);
  unsigned int gridx_combine = ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int gridy_combine = ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int gridz_combine = 1;
  dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(0, rdy_h, rdx_h, rdy2_h, rdx2_h, n_shots_h, ny_h, nx_h,
             n_receivers_per_shot_h, n_sources_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h);
  for (t = nt - 1; t >= 0; --t) {
    if ((nt - 1 - t) & 1) {
      if (n_sources_per_shot_h > 0) {
        record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
            grad_f + t * n_shots_h * n_sources_per_shot_h, wfp, sources_i);
        CHECK_KERNEL_ERROR
      }
      backward_kernel<<<dimGrid, dimBlock>>>(
          v2dt2, wfp, wfc, psiyn, psixn, psiy, psix, zetayn, zetaxn, zetay,
          zetax, dwdv + (t / step_ratio_h) * n_shots_h * ny_h * nx_h,
          grad_v_shot, ay, ax, by, bx, dbydy, dbxdx,
          v_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      if (n_receivers_per_shot_h > 0) {
        add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
            wfc, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
        CHECK_KERNEL_ERROR
      }
    } else {
      if (n_sources_per_shot_h > 0) {
        record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
            grad_f + t * n_shots_h * n_sources_per_shot_h, wfc, sources_i);
        CHECK_KERNEL_ERROR
      }
      backward_kernel<<<dimGrid, dimBlock>>>(
          v2dt2, wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, zetayn,
          zetaxn, dwdv + (t / step_ratio_h) * n_shots_h * ny_h * nx_h,
          grad_v_shot, ay, ax, by, bx, dbydy, dbxdx,
          v_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      if (n_receivers_per_shot_h > 0) {
        add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
            wfp, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
  if (v_requires_grad && n_shots_h > 1) {
    combine_grad_v<<<dimGrid_combine, dimBlock_combine>>>(grad_v, grad_v_shot);
    CHECK_KERNEL_ERROR
  }
}

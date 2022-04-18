#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

#include <iostream>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line,
                             bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

namespace {
template <typename T, int A>
__device__ __forceinline__ T diffx(T const *__restrict a, T one_over_dx,
                                   int64_t ny) {
  if (A == 2) {
    return static_cast<T>(1.0 / 2.0) * (a[ny] - a[-ny]) * one_over_dx;
  } else if (A == 4) {
    return (static_cast<T>(8.0 / 12.0) * (a[ny] - a[-ny]) -
            static_cast<T>(1.0 / 12.0) * (a[2 * ny] - a[-2 * ny])) *
           one_over_dx;
  } else if (A == 6) {
    return (static_cast<T>(3.0 / 4.0) * (a[ny] - a[-ny]) -
            static_cast<T>(3.0 / 20.0) * (a[2 * ny] - a[-2 * ny]) +
            static_cast<T>(1.0 / 60.0) * (a[3 * ny] - a[-3 * ny])) *
           one_over_dx;
  } else {
    return (static_cast<T>(4.0 / 5.0) * (a[ny] - a[-ny]) -
            static_cast<T>(1.0 / 5.0) * (a[2 * ny] - a[-2 * ny]) +
            static_cast<T>(4.0 / 105.0) * (a[3 * ny] - a[-3 * ny]) -
            static_cast<T>(1.0 / 280.0) * (a[4 * ny] - a[-4 * ny])) *
           one_over_dx;
  }
}

template <typename T, int A>
__device__ __forceinline__ T diffx2(T const *__restrict a, T one_over_dx2,
                                    int64_t ny) {
  if (A == 2) {
    return (-static_cast<T>(2.0) * a[0] + a[ny] + a[-ny]) * one_over_dx2;
  } else if (A == 4) {
    return (-static_cast<T>(5.0 / 2.0) * a[0] +
            static_cast<T>(4.0 / 3.0) * (a[ny] + a[-ny]) -
            static_cast<T>(1.0 / 12.0) * (a[2 * ny] + a[-2 * ny])) *
           one_over_dx2;
  } else if (A == 6) {
    return (-static_cast<T>(49.0 / 18.0) * a[0] +
            static_cast<T>(3.0 / 2.0) * (a[ny] + a[-ny]) -
            static_cast<T>(3.0 / 20.0) * (a[2 * ny] + a[-2 * ny]) +
            static_cast<T>(1.0 / 90.0) * (a[3 * ny] + a[-3 * ny])) *
           one_over_dx2;
  } else {
    return (-static_cast<T>(205.0 / 72.0) * a[0] +
            static_cast<T>(8.0 / 5.0) * (a[ny] + a[-ny]) -
            static_cast<T>(1.0 / 5.0) * (a[2 * ny] + a[-2 * ny]) +
            static_cast<T>(8.0 / 315.0) * (a[3 * ny] + a[-3 * ny]) -
            static_cast<T>(1.0 / 560.0) * (a[4 * ny] + a[-4 * ny])) *
           one_over_dx2;
  }
}

template <typename T, int A>
__device__ __forceinline__ T diffy(T const *__restrict a, T one_over_dy) {
  if (A == 2) {
    return static_cast<T>(1.0 / 2.0) * (a[1] - a[-1]) * one_over_dy;
  } else if (A == 4) {
    return (static_cast<T>(8.0 / 12.0) * (a[1] - a[-1]) -
            static_cast<T>(1.0 / 12.0) * (a[2] - a[-2])) *
           one_over_dy;
  } else if (A == 6) {
    return (static_cast<T>(3.0 / 4.0) * (a[1] - a[-1]) -
            static_cast<T>(3.0 / 20.0) * (a[2] - a[-2]) +
            static_cast<T>(1.0 / 60.0) * (a[3] - a[-3])) *
           one_over_dy;
  } else {
    return (static_cast<T>(4.0 / 5.0) * (a[1] - a[-1]) -
            static_cast<T>(1.0 / 5.0) * (a[2] - a[-2]) +
            static_cast<T>(4.0 / 105.0) * (a[3] - a[-3]) -
            static_cast<T>(1.0 / 280.0) * (a[4] - a[-4])) *
           one_over_dy;
  }
}

template <typename T, int A>
__device__ __forceinline__ T diffy2(T const *__restrict a, T one_over_dy2) {
  if (A == 2) {
    return (-static_cast<T>(2.0) * a[0] + a[1] + a[-1]) * one_over_dy2;
  } else if (A == 4) {
    return (-static_cast<T>(5.0 / 2.0) * a[0] +
            static_cast<T>(4.0 / 3.0) * (a[1] + a[-1]) -
            static_cast<T>(1.0 / 12.0) * (a[2] + a[-2])) *
           one_over_dy2;
  } else if (A == 6) {
    return (-static_cast<T>(49.0 / 18.0) * a[0] +
            static_cast<T>(3.0 / 2.0) * (a[1] + a[-1]) -
            static_cast<T>(3.0 / 20.0) * (a[2] + a[-2]) +
            static_cast<T>(1.0 / 90.0) * (a[3] + a[-3])) *
           one_over_dy2;
  } else {
    return (-static_cast<T>(205.0 / 72.0) * a[0] +
            static_cast<T>(8.0 / 5.0) * (a[1] + a[-1]) -
            static_cast<T>(1.0 / 5.0) * (a[2] + a[-2]) +
            static_cast<T>(8.0 / 315.0) * (a[3] + a[-3]) -
            static_cast<T>(1.0 / 560.0) * (a[4] + a[-4])) *
           one_over_dy2;
  }
}

torch::Tensor create_or_pad(torch::Tensor const &tensor, int64_t fd_pad,
                            at::TensorOptions const &options,
                            std::array<int64_t, 3> const &size) {
  if (tensor.numel() == 0) {
    return at::zeros(size, options);
  } else {
    return at::constant_pad_nd(tensor, {fd_pad, fd_pad, fd_pad, fd_pad});
  }
}

template <typename T>
T ceil_div(T numerator, T denominator) {
  return (numerator + denominator - static_cast<T>(1)) / denominator;
}

template <typename T, int A>
__global__ void forward_kernel_1(
    T const *__restrict wfc, T *__restrict psix, T *__restrict psiy,
    T const *__restrict ax, T const *__restrict ay, T const *__restrict bx,
    T const *__restrict by, int64_t nx, int64_t ny, T one_over_dx,
    T one_over_dy, int64_t interior1_xstart, int64_t interior1_xend,
    int64_t interior1_ystart, int64_t interior1_yend) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad or
      (x >= interior1_xstart and y >= interior1_ystart and
       x < interior1_xend and y < interior1_yend))
    return;
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nx * ny + x * ny + y};
  psix[i] = bx[x] * diffx<T, A>(wfc + i, one_over_dx, ny) + ax[x] * psix[i];
  psiy[i] = by[y] * diffy<T, A>(wfc + i, one_over_dy) + ay[y] * psiy[i];
}

template <typename T, int A, bool v_requires_grad>
__global__ void forward_kernel_2(
    T const *__restrict wfc, T *__restrict wfp, T const *__restrict psix,
    T const *__restrict psiy, T *__restrict zetax, T *__restrict zetay,
    T *__restrict dwdv, T const *__restrict v2dt2, T const *__restrict two_vdt2,
    T const *__restrict ax, T const *__restrict ay, T const *__restrict bx,
    T const *__restrict by, int64_t nx, int64_t ny, T one_over_dx,
    T one_over_dy, T one_over_dx2, T one_over_dy2, int64_t interior2_xstart,
    int64_t interior2_xend, int64_t interior2_ystart, int64_t interior2_yend) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad) return;
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{x * ny + y};
  auto i{batch * nx * ny + j};
  T d2wdx2{diffx2<T, A>(wfc + i, one_over_dx2, ny)};
  T d2wdy2{diffy2<T, A>(wfc + i, one_over_dy2)};
  T wf_sum;
  if (x < interior2_xstart or y < interior2_ystart or x >= interior2_xend or
      y >= interior2_yend) {
    T psix_x{diffx<T, A>(psix + i, one_over_dx, ny)};
    T psiy_y{diffy<T, A>(psiy + i, one_over_dy)};
    zetax[i] = ax[x] * zetax[i] + bx[x] * (d2wdx2 + psix_x);
    zetay[i] = ay[y] * zetay[i] + by[y] * (d2wdy2 + psiy_y);
    wf_sum = d2wdx2 + d2wdy2 + psix_x + psiy_y + zetax[i] + zetay[i];
  } else {
    wf_sum = d2wdx2 + d2wdy2;
  }
  if (v_requires_grad) {
    dwdv[i] = two_vdt2[j] * wf_sum;
  }
  wfp[i] = v2dt2[j] * wf_sum + 2 * wfc[i] - wfp[i];
}

template <typename T>
__global__ void add_sources(T *__restrict wf, T const *__restrict f,
                            int64_t const *__restrict sources_i,
                            int64_t n_sources_per_shot, int64_t n_shots,
                            int64_t nx, int64_t ny) {
  auto source_idx{blockIdx.x * blockDim.x + threadIdx.x};
  auto shot_idx{blockIdx.y * blockDim.y + threadIdx.y};
  if (source_idx >= n_sources_per_shot or shot_idx >= n_shots) return;
  auto k{shot_idx * n_sources_per_shot + source_idx};
  wf[shot_idx * nx * ny + sources_i[k]] += f[k];
}

template <typename T>
__global__ void record_receivers(T *__restrict r, T const *__restrict wf,
                                 int64_t const *__restrict receivers_i,
                                 int64_t n_receivers_per_shot, int64_t n_shots,
                                 int64_t nx, int64_t ny) {
  auto receiver_idx{blockIdx.x * blockDim.x + threadIdx.x};
  auto shot_idx{blockIdx.y * blockDim.y + threadIdx.y};
  if (receiver_idx >= n_receivers_per_shot or shot_idx >= n_shots) return;
  auto k{shot_idx * n_receivers_per_shot + receiver_idx};
  r[k] = wf[shot_idx * nx * ny + receivers_i[k]];
}

template <typename T, int A>
__global__ void add_to_grad_v(T *__restrict grad_v, T const *__restrict wfc,
                              T const *__restrict dwdv, int64_t step_ratio,
                              int64_t nx, int64_t ny) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad) return;
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nx * ny + x * ny + y};
  grad_v[i] += wfc[i] * dwdv[i] * step_ratio;
}

template <typename T, int A>
__global__ void backward_kernel_1(
    T *__restrict zetax, T *__restrict zetay, T *__restrict b_a_zetax,
    T *__restrict b_a_zetay, T *__restrict v2dt2_wfc, T const *__restrict wfc,
    T const *__restrict v2dt2, T const *__restrict ax, T const *__restrict ay,
    T const *__restrict bx_over_ax, T const *__restrict by_over_ay, int64_t nx,
    int64_t ny, int64_t interior1_xstart, int64_t interior1_xend,
    int64_t interior1_ystart, int64_t interior1_yend) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad) return;
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{x * ny + y};
  auto i{batch * nx * ny + j};
  if (x < interior1_xstart or y < interior1_ystart or x >= interior1_xend or
      y >= interior1_yend) {
    zetax[i] = ax[x] * v2dt2[j] * wfc[i] + ax[x] * zetax[i];
    zetay[i] = ay[y] * v2dt2[j] * wfc[i] + ay[y] * zetay[i];
    b_a_zetax[i] = bx_over_ax[x] * zetax[i];
    b_a_zetay[i] = by_over_ay[y] * zetay[i];
  }
  v2dt2_wfc[i] = v2dt2[j] * wfc[i];
}

template <typename T, int A>
__global__ void backward_kernel_2(
    T *__restrict psix, T *__restrict psiy, T one_over_dx, T one_over_dy,
    int64_t nx, int64_t ny, T const *ax, T const *ay, T *__restrict b_a_psix,
    T *__restrict b_a_psiy, T const *__restrict b_a_zetax,
    T const *__restrict b_a_zetay, T const *__restrict v2dt2_wfc,
    T const *__restrict bx_over_ax, T const *__restrict by_over_ay,
    int64_t interior1_xstart, int64_t interior1_xend, int64_t interior1_ystart,
    int64_t interior1_yend) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad or
      (x >= interior1_xstart and y >= interior1_ystart and
       x < interior1_xend and y < interior1_yend))
    return;
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nx * ny + x * ny + y};
  psix[i] = -ax[x] * diffx<T, A>(b_a_zetax + i, one_over_dx, ny) -
            ax[x] * diffx<T, A>(v2dt2_wfc + i, one_over_dx, ny) +
            ax[x] * psix[i];
  psiy[i] = -ay[y] * diffy<T, A>(b_a_zetay + i, one_over_dy) -
            ay[y] * diffy<T, A>(v2dt2_wfc + i, one_over_dy) + ay[y] * psiy[i];
  b_a_psix[i] = bx_over_ax[x] * psix[i];
  b_a_psiy[i] = by_over_ay[y] * psiy[i];
}

template <typename T, int A>
__global__ void backward_kernel_3(
    T *__restrict wfp, T *__restrict wfc, T one_over_dx, T one_over_dy,
    T one_over_dx2, T one_over_dy2, int64_t nx, int64_t ny,
    T const *__restrict b_a_psix, T const *__restrict b_a_psiy,
    T const *__restrict b_a_zetax, T const *__restrict b_a_zetay,
    T const *__restrict v2dt2_wfc, int64_t interior2_xstart,
    int64_t interior2_xend, int64_t interior2_ystart, int64_t interior2_yend) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad) return;
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nx * ny + x * ny + y};
  if (x < interior2_xstart or y < interior2_ystart or x >= interior2_xend or
      y >= interior2_yend) {
    wfp[i] = diffx2<T, A>(v2dt2_wfc + i, one_over_dx2, ny) +
             diffy2<T, A>(v2dt2_wfc + i, one_over_dy2) + 2 * wfc[i] + wfp[i] +
             diffx2<T, A>(b_a_zetax + i, one_over_dx2, ny) +
             diffy2<T, A>(b_a_zetay + i, one_over_dy2) -
             diffx<T, A>(b_a_psix + i, one_over_dx, ny) -
             diffy<T, A>(b_a_psiy + i, one_over_dy);
  } else {
    wfp[i] = diffx2<T, A>(v2dt2_wfc + i, one_over_dx2, ny) +
             diffy2<T, A>(v2dt2_wfc + i, one_over_dy2) + 2 * wfc[i] + wfp[i];
  }
  wfc[i] *= -1;
}

template <typename T, int A>
void forward_batch(T *wfc, T *wfp, T *psix, T *psiy, T *zetax, T *zetay,
                   int64_t const *sources_i, int64_t const *receivers_i,
                   T *dwdv, T const *v2dt2, T const *two_vdt2, T const *f, T *r,
                   T const *ax, T const *ay, T const *bx, T const *by,
                   T one_over_dx, T one_over_dy, T one_over_dx2, T one_over_dy2,
                   int64_t n_sources_per_shot, int64_t n_receivers_per_shot,
                   int64_t nx, int64_t ny, int64_t nt, int64_t step_ratio,
                   bool v_requires_grad, int64_t n_batch,
                   int64_t const pml_width[4]) {
  constexpr int fd_pad{A / 2};
  dim3 dimBlock(32, 32, 1);
  auto gridx{ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(dimBlock.x))};
  auto gridy{ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(dimBlock.y))};
  auto gridz{ceil_div(n_batch, static_cast<int64_t>(dimBlock.z))};
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  auto gridx_sources{
      ceil_div(n_sources_per_shot, static_cast<int64_t>(dimBlock_sources.x))};
  auto gridy_sources{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_sources.y))};
  auto gridz_sources{1};
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  auto gridx_receivers{ceil_div(n_receivers_per_shot,
                                static_cast<int64_t>(dimBlock_receivers.x))};
  auto gridy_receivers{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_receivers.y))};
  auto gridz_receivers{1};
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);
  int64_t const interior1[4]{pml_width[0] + fd_pad, nx - pml_width[1] - fd_pad,
                             pml_width[2] + fd_pad, ny - pml_width[3] - fd_pad};
  int64_t const interior2[4]{
      pml_width[0] + 2 * fd_pad, nx - pml_width[1] - 2 * fd_pad,
      pml_width[2] + 2 * fd_pad, ny - pml_width[3] - 2 * fd_pad};
  for (int64_t t{}; t < nt; ++t) {
    forward_kernel_1<T, A><<<dimGrid, dimBlock>>>(
        wfc, psix, psiy, ax, ay, bx, by, nx, ny, one_over_dx, one_over_dy,
        interior1[0], interior1[1], interior1[2], interior1[3]);
    gpuErrchk(cudaPeekAtLastError());
    if (t % step_ratio == 0 and v_requires_grad) {
      forward_kernel_2<T, A, true><<<dimGrid, dimBlock>>>(
          wfc, wfp, psix, psiy, zetax, zetay,
          dwdv + (t / step_ratio) * n_batch * nx * ny, v2dt2, two_vdt2, ax, ay,
          bx, by, nx, ny, one_over_dx, one_over_dy, one_over_dx2, one_over_dy2,
          interior2[0], interior2[1], interior2[2], interior2[3]);
    } else {
      forward_kernel_2<T, A, false><<<dimGrid, dimBlock>>>(
          wfc, wfp, psix, psiy, zetax, zetay, nullptr, v2dt2, two_vdt2, ax, ay,
          bx, by, nx, ny, one_over_dx, one_over_dy, one_over_dx2, one_over_dy2,
          interior2[0], interior2[1], interior2[2], interior2[3]);
    }
    gpuErrchk(cudaPeekAtLastError());
    if (n_sources_per_shot > 0) {
      add_sources<<<dimGrid_sources, dimBlock_sources>>>(
          wfp, f + t * n_batch * n_sources_per_shot, sources_i,
          n_sources_per_shot, n_batch, nx, ny);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_receivers_per_shot > 0) {
      record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_batch * n_receivers_per_shot, wfc, receivers_i,
          n_receivers_per_shot, n_batch, nx, ny);
      gpuErrchk(cudaPeekAtLastError());
    }
    std::swap(wfp, wfc);
  }
}

template <typename T, int A>
__global__ void combine_grad_v(T *__restrict grad_v,
                               T const *__restrict grad_v_batch,
                               int64_t n_batch, int64_t nx, int64_t ny) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  if (x >= nx - fd_pad or y >= ny - fd_pad) return;
  auto i{x * ny + y};
  int64_t nxny{nx * ny};
  for (int64_t batch{}; batch < n_batch; ++batch) {
    grad_v[i] += grad_v_batch[batch * nxny + i];
  }
}

template <typename T, int A>
void backward_batch(T *wfc, T *wfp, T *psix, T *psiy, T *zetax, T *zetay,
                    int64_t const *sources_i, int64_t const *receivers_i,
                    T *dwdv, T const *v2dt2, T *f, T const *r, T *grad_v,
                    T const *ax, T const *ay, T const *bx_over_ax,
                    T const *by_over_ay, T *b_a_psix, T *b_a_psiy, T *b_a_zetax,
                    T *b_a_zetay, T *v2dt2_wfc, T one_over_dx, T one_over_dy,
                    T one_over_dx2, T one_over_dy2, int64_t n_sources_per_shot,
                    int64_t n_receivers_per_shot, int64_t nx, int64_t ny,
                    int64_t nt, int64_t step_ratio, bool v_requires_grad,
                    int64_t n_batch, int64_t const pml_width[4]) {
  constexpr int fd_pad{A / 2};
  dim3 dimBlock(32, 32, 1);
  auto gridx{ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(dimBlock.x))};
  auto gridy{ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(dimBlock.y))};
  auto gridz{ceil_div(n_batch, static_cast<int64_t>(dimBlock.z))};
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  auto gridx_sources{
      ceil_div(n_sources_per_shot, static_cast<int64_t>(dimBlock_sources.x))};
  auto gridy_sources{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_sources.y))};
  auto gridz_sources{1};
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  auto gridx_receivers{ceil_div(n_receivers_per_shot,
                                static_cast<int64_t>(dimBlock_receivers.x))};
  auto gridy_receivers{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_receivers.y))};
  auto gridz_receivers{1};
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);
  int64_t interior1[4]{pml_width[0] + fd_pad, nx - pml_width[1] - fd_pad,
                       pml_width[2] + fd_pad, ny - pml_width[3] - fd_pad};
  int64_t interior2[4]{
      pml_width[0] + 2 * fd_pad, nx - pml_width[1] - 2 * fd_pad,
      pml_width[2] + 2 * fd_pad, ny - pml_width[3] - 2 * fd_pad};
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (n_receivers_per_shot > 0) {
      add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
          wfp, r + t * n_batch * n_receivers_per_shot, receivers_i,
          n_receivers_per_shot, n_batch, nx, ny);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_sources_per_shot > 0) {
      record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
          f + t * n_batch * n_sources_per_shot, wfc, sources_i,
          n_sources_per_shot, n_batch, nx, ny);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 and v_requires_grad) {
      add_to_grad_v<T, A><<<dimGrid, dimBlock>>>(
          grad_v, wfc, dwdv + (t / step_ratio) * n_batch * nx * ny, step_ratio,
          nx, ny);
      gpuErrchk(cudaPeekAtLastError());
    }
    backward_kernel_1<T, A><<<dimGrid, dimBlock>>>(
        zetax, zetay, b_a_zetax, b_a_zetay, v2dt2_wfc, wfc, v2dt2, ax, ay,
        bx_over_ax, by_over_ay, nx, ny, interior1[0], interior1[1],
        interior1[2], interior1[3]);
    gpuErrchk(cudaPeekAtLastError());
    backward_kernel_2<T, A><<<dimGrid, dimBlock>>>(
        psix, psiy, one_over_dx, one_over_dy, nx, ny, ax, ay, b_a_psix,
        b_a_psiy, b_a_zetax, b_a_zetay, v2dt2_wfc, bx_over_ax, by_over_ay,
        interior1[0], interior1[1], interior1[2], interior1[3]);
    gpuErrchk(cudaPeekAtLastError());
    backward_kernel_3<T, A><<<dimGrid, dimBlock>>>(
        wfp, wfc, one_over_dx, one_over_dy, one_over_dx2, one_over_dy2, nx, ny,
        b_a_psix, b_a_psiy, b_a_zetax, b_a_zetay, v2dt2_wfc, interior2[0],
        interior2[1], interior2[2], interior2[3]);
    gpuErrchk(cudaPeekAtLastError());
    std::swap(wfp, wfc);
  }
}

void zero_interior(torch::Tensor tensor, int64_t nx, int64_t ny, int fd_pad,
                   int64_t const pml_width[4]) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  at::indexing::TensorIndex slicex{torch::indexing::Slice(
      fd_pad + pml_width[0], nx - pml_width[1] - fd_pad)};
  at::indexing::TensorIndex slicey{torch::indexing::Slice(
      fd_pad + pml_width[2], ny - pml_width[3] - fd_pad)};
  tensor.index_put_({all_slice, slicex, slicey}, 0);
}

}  // namespace

class ScalarCUDAFunction
    : public torch::autograd::Function<ScalarCUDAFunction> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext *ctx, torch::Tensor const &v,
      torch::Tensor const &f, torch::Tensor const &wfc0,
      torch::Tensor const &wfp0, torch::Tensor const &psix0,
      torch::Tensor const &psiy0, torch::Tensor const &zetax0,
      torch::Tensor const &zetay0, torch::Tensor const &ax,
      torch::Tensor const &ay, torch::Tensor const &bx, torch::Tensor const &by,
      torch::Tensor const &sources_i, torch::Tensor const &receivers_i,
      double dx, double dy, double dt, int64_t nt, int64_t n_batch,
      int64_t step_ratio, int64_t accuracy, int64_t pml_width0,
      int64_t pml_width1, int64_t pml_width2, int64_t pml_width3) {
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(v.device());

    auto options{at::device(v.device()).dtype(v.scalar_type())};
    auto nx{v.size(0)};
    auto ny{v.size(1)};
    std::array<int64_t, 3> size_with_batch{n_batch, nx, ny};
    auto fd_pad{accuracy / 2};
    int64_t const pml_width[4] = {pml_width0, pml_width1, pml_width2,
                                  pml_width3};
    int64_t n_sources_per_shot{};
    if (sources_i.numel() > 0) {
      n_sources_per_shot = sources_i.size(1);
    }
    int64_t n_receivers_per_shot{};
    if (receivers_i.numel() > 0) {
      n_receivers_per_shot = receivers_i.size(1);
    }
    auto wfc{create_or_pad(wfc0, fd_pad, options, size_with_batch)};
    auto wfp{create_or_pad(wfp0, fd_pad, options, size_with_batch)};
    auto psix{create_or_pad(psix0, fd_pad, options, size_with_batch)};
    auto psiy{create_or_pad(psiy0, fd_pad, options, size_with_batch)};
    auto zetax{create_or_pad(zetax0, fd_pad, options, size_with_batch)};
    auto zetay{create_or_pad(zetay0, fd_pad, options, size_with_batch)};
    auto r{at::empty({nt, n_batch, n_receivers_per_shot}, options)};
    torch::Tensor dwdv;
    if (v.requires_grad()) {
      dwdv = at::empty({(nt + step_ratio - 1) / step_ratio, n_batch, nx, ny},
                       options);
    }

    zero_interior(psix, nx, ny, fd_pad, pml_width);
    zero_interior(psiy, nx, ny, fd_pad, pml_width);
    zero_interior(zetax, nx, ny, fd_pad, pml_width);
    zero_interior(zetay, nx, ny, fd_pad, pml_width);

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cuda_forward", ([&] {
          scalar_t dt2_a = dt * dt;
          scalar_t one_over_dx_a = 1.0 / dx;
          scalar_t one_over_dy_a = 1.0 / dy;
          scalar_t one_over_dx2_a = 1.0 / (dx * dx);
          scalar_t one_over_dy2_a = 1.0 / (dy * dy);
          auto v2dt2{v * v * dt2_a};
          auto v2dt2_a{v2dt2.data_ptr<scalar_t>()};
          auto two_vdt2{static_cast<scalar_t>(2.0) * v * dt2_a};
          auto two_vdt2_a{two_vdt2.data_ptr<scalar_t>()};
          auto f_a{f.data_ptr<scalar_t>()};
          auto r_a{r.data_ptr<scalar_t>()};
          auto wfc_a{wfc.data_ptr<scalar_t>()};
          auto wfp_a{wfp.data_ptr<scalar_t>()};
          auto psix_a{psix.data_ptr<scalar_t>()};
          auto psiy_a{psiy.data_ptr<scalar_t>()};
          auto zetax_a{zetax.data_ptr<scalar_t>()};
          auto zetay_a{zetay.data_ptr<scalar_t>()};
          auto ax_a{ax.data_ptr<scalar_t>()};
          auto ay_a{ay.data_ptr<scalar_t>()};
          auto bx_a{bx.data_ptr<scalar_t>()};
          auto by_a{by.data_ptr<scalar_t>()};
          auto sources_i_a{sources_i.data_ptr<int64_t>()};
          auto receivers_i_a{receivers_i.data_ptr<int64_t>()};
          scalar_t *dwdv_a{};
          if (v.requires_grad()) {
            dwdv_a = dwdv.data_ptr<scalar_t>();
          }
          constexpr void (*forward_batches[])(
              scalar_t * wfc, scalar_t * wfp, scalar_t * psix, scalar_t * psiy,
              scalar_t * zetax, scalar_t * zetay, int64_t const *sources_i,
              int64_t const *receivers_i, scalar_t *dwdv, scalar_t const *v2dt2,
              scalar_t const *two_vdt2, scalar_t const *f, scalar_t *r,
              scalar_t const *ax, scalar_t const *ay, scalar_t const *bx,
              scalar_t const *by, scalar_t one_over_dx, scalar_t one_over_dy,
              scalar_t one_over_dx2, scalar_t one_over_dy2,
              int64_t n_sources_per_shot, int64_t n_receivers_per_shot,
              int64_t nx, int64_t ny, int64_t nt, int64_t step_ratio,
              bool v_requires_grad, int64_t n_batch,
              int64_t const pml_width[4]){
              forward_batch<scalar_t, 2>, forward_batch<scalar_t, 4>,
              forward_batch<scalar_t, 6>, forward_batch<scalar_t, 8>};
          forward_batches[accuracy / 2 - 1](
              wfc_a, wfp_a, psix_a, psiy_a, zetax_a, zetay_a, sources_i_a,
              receivers_i_a, dwdv_a, v2dt2_a, two_vdt2_a, f_a, r_a, ax_a, ay_a,
              bx_a, by_a, one_over_dx_a, one_over_dy_a, one_over_dx2_a,
              one_over_dy2_a, n_sources_per_shot, n_receivers_per_shot, nx, ny,
              nt, step_ratio, v.requires_grad(), n_batch, pml_width);
        }));
    if (v.requires_grad() or f.requires_grad() or wfc0.requires_grad() or
        wfp0.requires_grad() or psix0.requires_grad() or
        psiy0.requires_grad() or zetax0.requires_grad() or
        zetay0.requires_grad()) {
      ctx->save_for_backward({v, ax, ay, bx, by, sources_i, receivers_i});
      ctx->saved_data["dwdv"] = dwdv;
      ctx->saved_data["dx"] = dx;
      ctx->saved_data["dy"] = dy;
      ctx->saved_data["dt"] = dt;
      ctx->saved_data["nt"] = nt;
      ctx->saved_data["n_batch"] = n_batch;
      ctx->saved_data["step_ratio"] = step_ratio;
      ctx->saved_data["accuracy"] = accuracy;
      ctx->saved_data["pml_width0"] = pml_width[0];
      ctx->saved_data["pml_width1"] = pml_width[1];
      ctx->saved_data["pml_width2"] = pml_width[2];
      ctx->saved_data["pml_width3"] = pml_width[3];
    }
    at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
    at::indexing::TensorIndex slicex{
        torch::indexing::Slice(fd_pad, nx - fd_pad)};
    at::indexing::TensorIndex slicey{
        torch::indexing::Slice(fd_pad, ny - fd_pad)};
    if (nt & 1) {
      return {wfp.index({all_slice, slicex, slicey}),
              wfc.index({all_slice, slicex, slicey}),
              psix.index({all_slice, slicex, slicey}),
              psiy.index({all_slice, slicex, slicey}),
              zetax.index({all_slice, slicex, slicey}),
              zetay.index({all_slice, slicex, slicey}),
              r};
    }
    return {wfc.index({all_slice, slicex, slicey}),
            wfp.index({all_slice, slicex, slicey}),
            psix.index({all_slice, slicex, slicey}),
            psiy.index({all_slice, slicex, slicey}),
            zetax.index({all_slice, slicex, slicey}),
            zetay.index({all_slice, slicex, slicey}),
            r};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list const &grad_outputs) {
    auto saved{ctx->get_saved_variables()};
    auto const &v{saved[0]};
    auto const &ax{saved[1]};
    auto const &ay{saved[2]};
    auto const &bx{saved[3]};
    auto const &by{saved[4]};
    auto const &sources_i{saved[5]};
    auto const &receivers_i{saved[6]};
    auto const &dwdv{ctx->saved_data["dwdv"].toTensor()};
    auto dx{ctx->saved_data["dx"].toDouble()};
    auto dy{ctx->saved_data["dy"].toDouble()};
    auto dt{ctx->saved_data["dt"].toDouble()};
    auto nt{ctx->saved_data["nt"].toInt()};
    auto n_batch{ctx->saved_data["n_batch"].toInt()};
    auto step_ratio{ctx->saved_data["step_ratio"].toInt()};
    auto accuracy{ctx->saved_data["accuracy"].toInt()};
    int64_t const pml_width[4] = {ctx->saved_data["pml_width0"].toInt(),
                                  ctx->saved_data["pml_width1"].toInt(),
                                  ctx->saved_data["pml_width2"].toInt(),
                                  ctx->saved_data["pml_width3"].toInt()};
    auto fd_pad{accuracy / 2};

    auto wfc =
        at::constant_pad_nd(grad_outputs[0], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto wfp =
        at::constant_pad_nd(grad_outputs[1], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psix =
        at::constant_pad_nd(grad_outputs[2], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psiy =
        at::constant_pad_nd(grad_outputs[3], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetax =
        at::constant_pad_nd(grad_outputs[4], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetay =
        at::constant_pad_nd(grad_outputs[5], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto grad_r{grad_outputs[6].contiguous()};
    auto options{at::device(v.device()).dtype(v.scalar_type())};
    auto nx{v.size(0)};
    auto ny{v.size(1)};
    int64_t n_sources_per_shot{};
    if (sources_i.numel() > 0) {
      n_sources_per_shot = sources_i.size(1);
    }
    int64_t n_receivers_per_shot{};
    if (receivers_i.numel() > 0) {
      n_receivers_per_shot = receivers_i.size(1);
    }

    auto grad_v{at::zeros({nx, ny}, options)};
    auto grad_v_batch{n_batch > 1 ? at::zeros({n_batch, nx, ny}, options)
                                  : at::empty(0)};
    auto grad_f{at::empty({nt, n_batch, n_sources_per_shot}, options)};

    auto bx_over_ax{bx / ax};
    auto by_over_ay{by / ay};
    auto b_a_psix{torch::zeros_like(psix)};
    auto b_a_psiy{torch::zeros_like(psiy)};
    auto b_a_zetax{torch::zeros_like(zetax)};
    auto b_a_zetay{torch::zeros_like(zetay)};
    auto v2dt2_wfc{torch::zeros_like(wfc)};

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cuda_backward", ([&] {
          scalar_t dt2_a = dt * dt;
          scalar_t one_over_dx_a = 1.0 / dx;
          scalar_t one_over_dy_a = 1.0 / dy;
          scalar_t one_over_dx2_a = 1.0 / (dx * dx);
          scalar_t one_over_dy2_a = 1.0 / (dy * dy);
          auto v2dt2{v * v * dt2_a};
          auto v2dt2_a{v2dt2.data_ptr<scalar_t>()};
          auto grad_v_a{grad_v.data_ptr<scalar_t>()};
          auto grad_v_batch_a{n_batch > 1 ? grad_v_batch.data_ptr<scalar_t>()
                                          : grad_v_a};
          auto grad_r_a{grad_r.data_ptr<scalar_t>()};
          auto grad_f_a{grad_f.data_ptr<scalar_t>()};
          auto wfc_a{wfc.data_ptr<scalar_t>()};
          auto wfp_a{wfp.data_ptr<scalar_t>()};
          auto psix_a{psix.data_ptr<scalar_t>()};
          auto psiy_a{psiy.data_ptr<scalar_t>()};
          auto zetax_a{zetax.data_ptr<scalar_t>()};
          auto zetay_a{zetay.data_ptr<scalar_t>()};
          auto ax_a{ax.data_ptr<scalar_t>()};
          auto ay_a{ay.data_ptr<scalar_t>()};
          auto bx_a{bx.data_ptr<scalar_t>()};
          auto by_a{by.data_ptr<scalar_t>()};
          auto bx_over_ax_a{bx_over_ax.data_ptr<scalar_t>()};
          auto by_over_ay_a{by_over_ay.data_ptr<scalar_t>()};
          auto b_a_psix_a{b_a_psix.data_ptr<scalar_t>()};
          auto b_a_psiy_a{b_a_psiy.data_ptr<scalar_t>()};
          auto b_a_zetax_a{b_a_zetax.data_ptr<scalar_t>()};
          auto b_a_zetay_a{b_a_zetay.data_ptr<scalar_t>()};
          auto sources_i_a{sources_i.data_ptr<int64_t>()};
          auto receivers_i_a{receivers_i.data_ptr<int64_t>()};
          auto v2dt2_wfc_a{v2dt2_wfc.data_ptr<scalar_t>()};
          scalar_t *dwdv_a;
          if (v.requires_grad()) {
            dwdv_a = dwdv.data_ptr<scalar_t>();
          }
          constexpr void (*backward_batches[])(
              scalar_t * wfc, scalar_t * wfp, scalar_t * psix, scalar_t * psiy,
              scalar_t * zetax, scalar_t * zetay, int64_t const *sources_i,
              int64_t const *receivers_i, scalar_t *dwdv, scalar_t const *v2dt2,
              scalar_t *f, scalar_t const *r, scalar_t *grad_v,
              scalar_t const *ax, scalar_t const *ay,
              scalar_t const *bx_over_ax, scalar_t const *by_over_ay,
              scalar_t *b_a_psix, scalar_t *b_a_psiy, scalar_t *b_a_zetax,
              scalar_t *b_a_zetay, scalar_t *v2dt2_wfc, scalar_t one_over_dx,
              scalar_t one_over_dy, scalar_t one_over_dx2,
              scalar_t one_over_dy2, int64_t n_sources_per_shot,
              int64_t n_receivers_per_shot, int64_t nx, int64_t ny, int64_t nt,
              int64_t step_ratio, bool v_requires_grad, int64_t n_batch,
              int64_t const pml_width[4]){
              backward_batch<scalar_t, 2>, backward_batch<scalar_t, 4>,
              backward_batch<scalar_t, 6>, backward_batch<scalar_t, 8>};
          backward_batches[accuracy / 2 - 1](
              wfc_a, wfp_a, psix_a, psiy_a, zetax_a, zetay_a, sources_i_a,
              receivers_i_a, dwdv_a, v2dt2_a, grad_f_a, grad_r_a,
              grad_v_batch_a, ax_a, ay_a, bx_over_ax_a, by_over_ay_a,
              b_a_psix_a, b_a_psiy_a, b_a_zetax_a, b_a_zetay_a, v2dt2_wfc_a,
              one_over_dx_a, one_over_dy_a, one_over_dx2_a, one_over_dy2_a,
              n_sources_per_shot, n_receivers_per_shot, nx, ny, nt, step_ratio,
              v.requires_grad(), n_batch, pml_width);
          if (v.requires_grad() and n_batch > 1) {
            dim3 dimBlock_combine(32, 32, 1);
            auto gridx_combine{ceil_div(
                nx - 2 * fd_pad, static_cast<int64_t>(dimBlock_combine.x))};
            auto gridy_combine{ceil_div(
                ny - 2 * fd_pad, static_cast<int64_t>(dimBlock_combine.y))};
            auto gridz_combine{1};
            dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
            if (accuracy == 2) {
              combine_grad_v<scalar_t, 2>
                  <<<dimGrid_combine, dimBlock_combine>>>(
                      grad_v_a, grad_v_batch_a, n_batch, nx, ny);
            } else if (accuracy == 4) {
              combine_grad_v<scalar_t, 4>
                  <<<dimGrid_combine, dimBlock_combine>>>(
                      grad_v_a, grad_v_batch_a, n_batch, nx, ny);
            } else if (accuracy == 6) {
              combine_grad_v<scalar_t, 6>
                  <<<dimGrid_combine, dimBlock_combine>>>(
                      grad_v_a, grad_v_batch_a, n_batch, nx, ny);
            } else {
              combine_grad_v<scalar_t, 8>
                  <<<dimGrid_combine, dimBlock_combine>>>(
                      grad_v_a, grad_v_batch_a, n_batch, nx, ny);
            }
            gpuErrchk(cudaPeekAtLastError());
          }
        }));

    zero_interior(psix, nx, ny, fd_pad, pml_width);
    zero_interior(psiy, nx, ny, fd_pad, pml_width);
    zero_interior(zetax, nx, ny, fd_pad, pml_width);
    zero_interior(zetay, nx, ny, fd_pad, pml_width);

    at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
    at::indexing::TensorIndex slicex{
        torch::indexing::Slice(fd_pad, nx - fd_pad)};
    at::indexing::TensorIndex slicey{
        torch::indexing::Slice(fd_pad, ny - fd_pad)};
    if (nt & 1) {
      return {grad_v,
              grad_f,
              wfp.index({all_slice, slicex, slicey}),
              wfc.index({all_slice, slicex, slicey}),
              psix.index({all_slice, slicex, slicey}),
              psiy.index({all_slice, slicex, slicey}),
              zetax.index({all_slice, slicex, slicey}),
              zetay.index({all_slice, slicex, slicey}),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor(),
              torch::Tensor()};
    }
    return {grad_v,
            grad_f,
            wfc.index({all_slice, slicex, slicey}),
            wfp.index({all_slice, slicex, slicey}),
            psix.index({all_slice, slicex, slicey}),
            psiy.index({all_slice, slicex, slicey}),
            zetax.index({all_slice, slicex, slicey}),
            zetay.index({all_slice, slicex, slicey}),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

std::vector<torch::Tensor> scalar_cuda_autograd(
    torch::Tensor const &v, torch::Tensor const &f, torch::Tensor const &wfc0,
    torch::Tensor const &wfp0, torch::Tensor const &psix0,
    torch::Tensor const &psiy0, torch::Tensor const &zetax0,
    torch::Tensor const &zetay0, torch::Tensor const &ax,
    torch::Tensor const &ay, torch::Tensor const &bx, torch::Tensor const &by,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i, double dx,
    double dy, double dt, int64_t nt, int64_t n_batch, int64_t step_ratio,
    int64_t accuracy, int64_t pml_width0, int64_t pml_width1,
    int64_t pml_width2, int64_t pml_width3) {
  return ScalarCUDAFunction::apply(
      v, f, wfc0, wfp0, psix0, psiy0, zetax0, zetay0, ax, ay, bx, by, sources_i,
      receivers_i, dx, dy, dt, nt, n_batch, step_ratio, accuracy, pml_width0,
      pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCUDA, m) {
  m.impl("scalar", scalar_cuda_autograd);
}

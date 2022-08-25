#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

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

__constant__ char fd_coeffs1yc[4 * sizeof(double)];
__constant__ char fd_coeffs1xc[4 * sizeof(double)];
__constant__ char fd_coeffs2yc[5 * sizeof(double)];
__constant__ char fd_coeffs2xc[5 * sizeof(double)];
__constant__ int64_t pml_regionsyc[4];
__constant__ int64_t pml_regionsxc[4];
__constant__ int64_t nxc;
__constant__ int64_t nynxc;
__constant__ char dt2c[sizeof(double)];

namespace {

template <typename T>
__device__ __inline__ T fd_coeffs1y(int64_t i) {
  return ((T *)fd_coeffs1yc)[i];
}

template <typename T>
__device__ __inline__ T fd_coeffs1x(int64_t i) {
  return ((T *)fd_coeffs1xc)[i];
}

template <typename T>
__device__ __inline__ T fd_coeffs2y(int64_t i) {
  return ((T *)fd_coeffs2yc)[i];
}

template <typename T>
__device__ __inline__ T fd_coeffs2x(int64_t i) {
  return ((T *)fd_coeffs2xc)[i];
}

template <typename T>
__device__ __inline__ T dt2() {
  return *((T *)dt2c);
}

template <typename T, int A>
__global__ void diffy(T const *__restrict a, T *__restrict da) {
  constexpr int fd_pad{A / 2};
  int64_t y = blockIdx.x * blockDim.x + threadIdx.x + fd_pad;
  if (y < pml_regionsyc[3]) {
    if (A == 2) {
      da[y] = fd_coeffs1y<T>(0) * (a[y + 1] - a[y - 1]);
    } else if (A == 4) {
      da[y] = (fd_coeffs1y<T>(0) * (a[y + 1] - a[y - 1]) +
               fd_coeffs1y<T>(1) * (a[y + 2] - a[y - 2]));
    } else if (A == 6) {
      da[y] = (fd_coeffs1y<T>(0) * (a[y + 1] - a[y - 1]) +
               fd_coeffs1y<T>(1) * (a[y + 2] - a[y - 2]) +
               fd_coeffs1y<T>(2) * (a[y + 3] - a[y - 3]));
    } else {
      da[y] = (fd_coeffs1y<T>(0) * (a[y + 1] - a[y - 1]) +
               fd_coeffs1y<T>(1) * (a[y + 2] - a[y - 2]) +
               fd_coeffs1y<T>(2) * (a[y + 3] - a[y - 3]) +
               fd_coeffs1y<T>(3) * (a[y + 4] - a[y - 4]));
    }
  }
}

template <typename T, int A>
__global__ void diffx(T const *__restrict a, T *__restrict da) {
  constexpr int fd_pad{A / 2};
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + fd_pad;
  if (x < pml_regionsxc[3]) {
    if (A == 2) {
      da[x] = fd_coeffs1x<T>(0) * (a[x + 1] - a[x - 1]);
    } else if (A == 4) {
      da[x] = (fd_coeffs1x<T>(0) * (a[x + 1] - a[x - 1]) +
               fd_coeffs1x<T>(1) * (a[x + 2] - a[x - 2]));
    } else if (A == 6) {
      da[x] = (fd_coeffs1x<T>(0) * (a[x + 1] - a[x - 1]) +
               fd_coeffs1x<T>(1) * (a[x + 2] - a[x - 2]) +
               fd_coeffs1x<T>(2) * (a[x + 3] - a[x - 3]));
    } else {
      da[x] = (fd_coeffs1x<T>(0) * (a[x + 1] - a[x - 1]) +
               fd_coeffs1x<T>(1) * (a[x + 2] - a[x - 2]) +
               fd_coeffs1x<T>(2) * (a[x + 3] - a[x - 3]) +
               fd_coeffs1x<T>(3) * (a[x + 4] - a[x - 4]));
    }
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

template <typename T, int A, bool v_requires_grad>
__global__ void forward_kernel(
    T const *__restrict wfc, T *__restrict wfp, T const *__restrict psiy,
    T const *__restrict psix, T *__restrict psiyn, T *__restrict psixn,
    T *__restrict zetay, T *__restrict zetax, T *__restrict dwdv,
    T const *__restrict v, T const *__restrict ay, T const *__restrict ax,
    T const *__restrict by, T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{y * nxc + x};
  auto i{batch * nynxc + j};
  if (y < pml_regionsyc[3] and x < pml_regionsxc[3]) {
    T d2wdy2;
    T d2wdx2;
#define D2WDY20 fd_coeffs2y<T>(0) * wfc[i]
#define D2WDY2(t) fd_coeffs2y<T>(t) * (wfc[i + t * nxc] + wfc[i - t * nxc])
#define D2WDX20 fd_coeffs2x<T>(0) * wfc[i]
#define D2WDX2(t) fd_coeffs2x<T>(t) * (wfc[i + t] + wfc[i - t])
    if (A == 2) {
      d2wdy2 = D2WDY20 + D2WDY2(1);
      d2wdx2 = D2WDX20 + D2WDX2(1);
    } else if (A == 4) {
      d2wdy2 = D2WDY20 + D2WDY2(1) + D2WDY2(2);
      d2wdx2 = D2WDX20 + D2WDX2(1) + D2WDX2(2);
    } else if (A == 6) {
      d2wdy2 = D2WDY20 + D2WDY2(1) + D2WDY2(2) + D2WDY2(3);
      d2wdx2 = D2WDX20 + D2WDX2(1) + D2WDX2(2) + D2WDX2(3);
    } else {
      d2wdy2 = D2WDY20 + D2WDY2(1) + D2WDY2(2) + D2WDY2(3) + D2WDY2(4);
      d2wdx2 = D2WDX20 + D2WDX2(1) + D2WDX2(2) + D2WDX2(3) + D2WDX2(4);
    }

    T w_sum{};
    if (y < pml_regionsyc[1] or y >= pml_regionsyc[2]) {
      T dwdy;
      T dpsiydy;
#define DDY(a, t) \
  fd_coeffs1y<T>(t) * (a[i + (t + 1) * nxc] - a[i - (t + 1) * nxc])
      if (A == 2) {
        dwdy = DDY(wfc, 0);
        dpsiydy = DDY(psiy, 0);
      } else if (A == 4) {
        dwdy = DDY(wfc, 0) + DDY(wfc, 1);
        dpsiydy = DDY(psiy, 0) + DDY(psiy, 1);
      } else if (A == 6) {
        dwdy = DDY(wfc, 0) + DDY(wfc, 1) + DDY(wfc, 2);
        dpsiydy = DDY(psiy, 0) + DDY(psiy, 1) + DDY(psiy, 2);
      } else {
        dwdy = DDY(wfc, 0) + DDY(wfc, 1) + DDY(wfc, 2) + DDY(wfc, 3);
        dpsiydy = DDY(psiy, 0) + DDY(psiy, 1) + DDY(psiy, 2) + DDY(psiy, 3);
      }
      T tmpy{(1 + by[y]) * d2wdy2 + dbydy[y] * dwdy + daydy[y] * psiy[i] +
             ay[y] * dpsiydy};
      w_sum += (1 + by[y]) * tmpy + ay[y] * zetay[i];
      psiyn[i] = by[y] * dwdy + ay[y] * psiy[i];
      zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
    } else {
      w_sum += d2wdy2;
    }
    if (x < pml_regionsxc[1] or x >= pml_regionsxc[2]) {
      T dwdx;
      T dpsixdx;
#define DDX(a, t) fd_coeffs1x<T>(t) * (a[i + (t + 1)] - a[i - (t + 1)])
      if (A == 2) {
        dwdx = DDX(wfc, 0);
        dpsixdx = DDX(psix, 0);
      } else if (A == 4) {
        dwdx = DDX(wfc, 0) + DDX(wfc, 1);
        dpsixdx = DDX(psix, 0) + DDX(psix, 1);
      } else if (A == 6) {
        dwdx = DDX(wfc, 0) + DDX(wfc, 1) + DDX(wfc, 2);
        dpsixdx = DDX(psix, 0) + DDX(psix, 1) + DDX(psix, 2);
      } else {
        dwdx = DDX(wfc, 0) + DDX(wfc, 1) + DDX(wfc, 2) + DDX(wfc, 3);
        dpsixdx = DDX(psix, 0) + DDX(psix, 1) + DDX(psix, 2) + DDX(psix, 3);
      }
      T tmpx{(1 + bx[x]) * d2wdx2 + dbxdx[x] * dwdx + daxdx[x] * psix[i] +
             ax[x] * dpsixdx};
      w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];
      psixn[i] = bx[x] * dwdx + ax[x] * psix[i];
      zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];
    } else {
      w_sum += d2wdx2;
    }
    if (v_requires_grad) {
      dwdv[i] = 2 * v[j] * dt2<T>() * w_sum;
    }
    wfp[i] = v[j] * v[j] * dt2<T>() * w_sum + 2 * wfc[i] - wfp[i];
  }
}

template <typename T>
__global__ void add_sources(T *__restrict wf, T const *__restrict f,
                            int64_t const *__restrict sources_i,
                            int64_t n_sources_per_shot, int64_t n_shots) {
  auto source_idx{blockIdx.x * blockDim.x + threadIdx.x};
  auto shot_idx{blockIdx.y * blockDim.y + threadIdx.y};
  if (source_idx >= n_sources_per_shot or shot_idx >= n_shots) return;
  auto k{shot_idx * n_sources_per_shot + source_idx};
  wf[shot_idx * nynxc + sources_i[k]] += f[k];
}

template <typename T>
__global__ void record_receivers(T *__restrict r, T const *__restrict wf,
                                 int64_t const *__restrict receivers_i,
                                 int64_t n_receivers_per_shot,
                                 int64_t n_shots) {
  auto receiver_idx{blockIdx.x * blockDim.x + threadIdx.x};
  auto shot_idx{blockIdx.y * blockDim.y + threadIdx.y};
  if (receiver_idx >= n_receivers_per_shot or shot_idx >= n_shots) return;
  auto k{shot_idx * n_receivers_per_shot + receiver_idx};
  r[k] = wf[shot_idx * nynxc + receivers_i[k]];
}

template <typename T, int A>
__global__ void add_to_grad_v(T *__restrict grad_v, T const *__restrict wfc,
                              T const *__restrict dwdv, int64_t step_ratio) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nynxc + y * nxc + x};
  if (y < pml_regionsyc[3] and x < pml_regionsxc[3]) {
    grad_v[i] += wfc[i] * dwdv[i] * step_ratio;
  }
}

template <typename T, int A>
__global__ void backward_kernel(
    T const *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
    T const *__restrict psiy, T const *__restrict psix, T *__restrict psiyn,
    T *__restrict psixn, T const *__restrict zetay, T const *__restrict zetax,
    T *__restrict zetayn, T *__restrict zetaxn, T const *__restrict v2dt2,
    T const *__restrict ay, T const *__restrict ax, T const *__restrict by,
    T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{y * nxc + x};
  auto i{batch * nynxc + j};
  if (y < pml_regionsyc[3] and x < pml_regionsxc[3]) {
    bool pml_y{y < pml_regionsyc[1] or y >= pml_regionsyc[2]};
    bool pml_x{x < pml_regionsxc[1] or x >= pml_regionsxc[2]};
    T wfp_y_term;
    T wfp_x_term;
#define WFPY0 fd_coeffs2y<T>(0) * v2dt2[j] * wfc[i]
#define WFPY(t)                                                \
  fd_coeffs2y<T>(t) * (v2dt2[j + t * nxc] * wfc[i + t * nxc] + \
                       v2dt2[j - t * nxc] * wfc[i - t * nxc])
#define WFPYPML0      \
  fd_coeffs2y<T>(0) * \
      ((1 + by[y]) * ((1 + by[y]) * v2dt2[j] * wfc[i] + by[y] * zetay[i]))
#define WFPYPML(t)                                                      \
  fd_coeffs2y<T>(t) *                                                   \
          (((1 + by[y + t]) *                                           \
            ((1 + by[y + t]) * v2dt2[j + t * nxc] * wfc[i + t * nxc] +  \
             by[y + t] * zetay[i + t * nxc])) +                         \
           ((1 + by[y - t]) *                                           \
            ((1 + by[y - t]) * v2dt2[j - t * nxc] * wfc[i - t * nxc] +  \
             by[y - t] * zetay[i - t * nxc]))) -                        \
      (fd_coeffs1y<T>(t - 1) *                                          \
       ((dbydy[y + t] *                                                 \
             ((1 + by[y + t]) * v2dt2[j + t * nxc] * wfc[i + t * nxc] + \
              by[y + t] * zetay[i + t * nxc]) +                         \
         by[y + t] * psiy[i + t * nxc]) -                               \
        (dbydy[y - t] *                                                 \
             ((1 + by[y - t]) * v2dt2[j - t * nxc] * wfc[i - t * nxc] + \
              by[y - t] * zetay[i - t * nxc]) +                         \
         by[y - t] * psiy[i - t * nxc])))
    if (A == 2) {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1);
      } else {
        wfp_y_term = WFPYPML0 + WFPYPML(1);
      }
    } else if (A == 4) {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1) + WFPY(2);
      } else {
        wfp_y_term = WFPYPML0 + WFPYPML(1) + WFPYPML(2);
      }
    } else if (A == 6) {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1) + WFPY(2) + WFPY(3);
      } else {
        wfp_y_term = WFPYPML0 + WFPYPML(1) + WFPYPML(2) + WFPYPML(3);
      }
    } else {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1) + WFPY(2) + WFPY(3) + WFPY(4);
      } else {
        wfp_y_term =
            WFPYPML0 + WFPYPML(1) + WFPYPML(2) + WFPYPML(3) + WFPYPML(4);
      }
    }
#define WFPX0 fd_coeffs2x<T>(0) * v2dt2[j] * wfc[i]
#define WFPX(t) \
  fd_coeffs2x<T>(t) * (v2dt2[j + t] * wfc[i + t] + v2dt2[j - t] * wfc[i - t])
#define WFPXPML0      \
  fd_coeffs2x<T>(0) * \
      ((1 + bx[x]) * ((1 + bx[x]) * v2dt2[j] * wfc[i] + bx[x] * zetax[i]))
#define WFPXPML(t)                                                           \
  fd_coeffs2x<T>(t) *                                                        \
          (((1 + bx[x + t]) * ((1 + bx[x + t]) * v2dt2[j + t] * wfc[i + t] + \
                               bx[x + t] * zetax[i + t])) +                  \
           ((1 + bx[x - t]) * ((1 + bx[x - t]) * v2dt2[j - t] * wfc[i - t] + \
                               bx[x - t] * zetax[i - t]))) -                 \
      (fd_coeffs1x<T>(t - 1) *                                               \
       ((dbxdx[x + t] * ((1 + bx[x + t]) * v2dt2[j + t] * wfc[i + t] +       \
                         bx[x + t] * zetax[i + t]) +                         \
         bx[x + t] * psix[i + t]) -                                          \
        (dbxdx[x - t] * ((1 + bx[x - t]) * v2dt2[j - t] * wfc[i - t] +       \
                         bx[x - t] * zetax[i - t]) +                         \
         bx[x - t] * psix[i - t])))
    if (A == 2) {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1);
      } else {
        wfp_x_term = WFPXPML0 + WFPXPML(1);
      }
    } else if (A == 4) {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1) + WFPX(2);
      } else {
        wfp_x_term = WFPXPML0 + WFPXPML(1) + WFPXPML(2);
      }
    } else if (A == 6) {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1) + WFPX(2) + WFPX(3);
      } else {
        wfp_x_term = WFPXPML0 + WFPXPML(1) + WFPXPML(2) + WFPXPML(3);
      }
    } else {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1) + WFPX(2) + WFPX(3) + WFPX(4);
      } else {
        wfp_x_term =
            WFPXPML0 + WFPXPML(1) + WFPXPML(2) + WFPXPML(3) + WFPXPML(4);
      }
    }

    wfp[i] = wfp_y_term + wfp_x_term + 2 * wfc[i] + wfp[i];
    wfcn[i] = -wfc[i];
    if (pml_y) {
      T tmp;
#define PSIY(t)                                                                \
  fd_coeffs1y<T>(t - 1) *                                                      \
      ((ay[y + t] * ((1 + by[y + t]) * v2dt2[j + t * nxc] * wfc[i + t * nxc] + \
                     by[y + t] * zetay[i + t * nxc])) -                        \
       (ay[y - t] * ((1 + by[y - t]) * v2dt2[j - t * nxc] * wfc[i - t * nxc] + \
                     by[y - t] * zetay[i - t * nxc])))
      if (A == 2) {
        tmp = -(PSIY(1));
      } else if (A == 4) {
        tmp = -(PSIY(1) + PSIY(2));
      } else if (A == 6) {
        tmp = -(PSIY(1) + PSIY(2) + PSIY(3));
      } else {
        tmp = -(PSIY(1) + PSIY(2) + PSIY(3) + PSIY(4));
      }
      psiyn[i] =
          tmp +
          daydy[y] * ((1 + by[y]) * v2dt2[j] * wfc[i] + by[y] * zetay[i]) +
          ay[y] * psiy[i];
      zetayn[i] = ay[y] * v2dt2[j] * wfc[i] + ay[y] * zetay[i];
    }
    if (pml_x) {
      T tmp;
#define PSIX(t)                                                    \
  fd_coeffs1x<T>(t - 1) *                                          \
      ((ax[x + t] * ((1 + bx[x + t]) * v2dt2[j + t] * wfc[i + t] + \
                     bx[x + t] * zetax[i + t])) -                  \
       (ax[x - t] * ((1 + bx[x - t]) * v2dt2[j - t] * wfc[i - t] + \
                     bx[x - t] * zetax[i - t])))
      if (A == 2) {
        tmp = -(PSIX(1));
      } else if (A == 4) {
        tmp = -(PSIX(1) + PSIX(2));
      } else if (A == 6) {
        tmp = -(PSIX(1) + PSIX(2) + PSIX(3));
      } else {
        tmp = -(PSIX(1) + PSIX(2) + PSIX(3) + PSIX(4));
      }
      psixn[i] =
          tmp +
          daxdx[x] * ((1 + bx[x]) * v2dt2[j] * wfc[i] + bx[x] * zetax[i]) +
          ax[x] * psix[i];
      zetaxn[i] = ax[x] * v2dt2[j] * wfc[i] + ax[x] * zetax[i];
    }
  }
}

template <typename T, int A>
void forward_batch(T *__restrict wfc, T *__restrict wfp, T *__restrict psiy,
                   T *__restrict psix, T *__restrict psiyn, T *__restrict psixn,
                   T *__restrict zetay, T *__restrict zetax,
                   int64_t const *__restrict sources_i,
                   int64_t const *__restrict receivers_i, T *__restrict dwdv,
                   T const *__restrict v, T const *__restrict f,
                   T *__restrict r, T const *__restrict ay,
                   T const *__restrict ax, T const *__restrict by,
                   T const *__restrict bx, T const *__restrict daydy,
                   T const *__restrict daxdx, T const *__restrict dbydy,
                   T const *__restrict dbxdx, int64_t n_sources_per_shot,
                   int64_t n_receivers_per_shot, int64_t ny, int64_t nx,
                   int64_t nt, int64_t step_ratio, bool v_requires_grad,
                   int64_t n_batch) {
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
  for (int64_t t{}; t < nt; ++t) {
    if (t % step_ratio == 0 and v_requires_grad) {
      forward_kernel<T, A, true><<<dimGrid, dimBlock>>>(
          wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax,
          dwdv + (t / step_ratio) * n_batch * ny * nx, v, ay, ax, by, bx, daydy,
          daxdx, dbydy, dbxdx);
    } else {
      forward_kernel<T, A, false><<<dimGrid, dimBlock>>>(
          wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, nullptr, v, ay, ax,
          by, bx, daydy, daxdx, dbydy, dbxdx);
    }
    gpuErrchk(cudaPeekAtLastError());
    if (n_sources_per_shot > 0) {
      add_sources<<<dimGrid_sources, dimBlock_sources>>>(
          wfp, f + t * n_batch * n_sources_per_shot, sources_i,
          n_sources_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_receivers_per_shot > 0) {
      record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_batch * n_receivers_per_shot, wfc, receivers_i,
          n_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    std::swap(wfp, wfc);
    std::swap(psiyn, psiy);
    std::swap(psixn, psix);
  }
}

template <typename T, int A>
__global__ void combine_grad_v(T *__restrict grad_v,
                               T const *__restrict grad_v_batch,
                               int64_t n_batch) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto i{y * nxc + x};
  if (y < pml_regionsyc[3] and x < pml_regionsxc[3]) {
    for (int64_t batch{}; batch < n_batch; ++batch) {
      grad_v[i] += grad_v_batch[batch * nynxc + i];
    }
  }
}

template <typename T, int A>
void backward_batch(T *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
                    T *__restrict psiy, T *__restrict psix, T *__restrict psiyn,
                    T *__restrict psixn, T *__restrict zetay,
                    T *__restrict zetax, T *__restrict zetayn,
                    T *__restrict zetaxn, int64_t const *__restrict sources_i,
                    int64_t const *__restrict receivers_i,
                    T const *__restrict dwdv, T const *__restrict v2dt2,
                    T *__restrict f, T const *__restrict r,
                    T *__restrict grad_v, T const *__restrict ay,
                    T const *__restrict ax, T const *__restrict by,
                    T const *__restrict bx, T const *__restrict daydy,
                    T const *__restrict daxdx, T const *__restrict dbydy,
                    T const *__restrict dbxdx, int64_t n_sources_per_shot,
                    int64_t n_receivers_per_shot, int64_t ny, int64_t nx,
                    int64_t nt, int64_t step_ratio, bool v_requires_grad,
                    int64_t n_batch) {
  constexpr int fd_pad{A / 2};
  dim3 dimBlock(32, 16, 1);
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
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (n_receivers_per_shot > 0) {
      add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
          wfp, r + t * n_batch * n_receivers_per_shot, receivers_i,
          n_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_sources_per_shot > 0) {
      record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
          f + t * n_batch * n_sources_per_shot, wfc, sources_i,
          n_sources_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 and v_requires_grad) {
      add_to_grad_v<T, A><<<dimGrid, dimBlock>>>(
          grad_v, wfc, dwdv + (t / step_ratio) * n_batch * ny * nx, step_ratio);
      gpuErrchk(cudaPeekAtLastError());
    }
    backward_kernel<T, A><<<dimGrid, dimBlock>>>(
        wfc, wfp, wfcn, psiy, psix, psiyn, psixn, zetay, zetax, zetayn, zetaxn,
        v2dt2, ay, ax, by, bx, daydy, daxdx, dbydy, dbxdx);
    gpuErrchk(cudaPeekAtLastError());
    T *tmp{wfc};
    wfc = wfp;
    wfp = wfcn;
    wfcn = tmp;
    std::swap(psiyn, psiy);
    std::swap(psixn, psix);
    std::swap(zetayn, zetay);
    std::swap(zetaxn, zetax);
  }
}

template <bool y>
void zero_interior(torch::Tensor tensor, int64_t ny, int64_t nx, int fd_pad,
                   int64_t const pml_width[4]) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  if (y) {
    at::indexing::TensorIndex slicey{torch::indexing::Slice(
        fd_pad + pml_width[0], ny - pml_width[1] - fd_pad)};
    tensor.index_put_({all_slice, slicey, all_slice}, 0);
  } else {
    at::indexing::TensorIndex slicex{torch::indexing::Slice(
        fd_pad + pml_width[2], nx - pml_width[3] - fd_pad)};
    tensor.index_put_({all_slice, all_slice, slicex}, 0);
  }
}

template <typename T>
void set_fd_coeffs(T *fd_coeffs1, T *fd_coeffs2, int64_t accuracy, T dx) {
  if (accuracy == 2) {
    fd_coeffs1[0] = static_cast<T>(1.0 / 2.0) / dx;
    fd_coeffs2[0] = -static_cast<T>(2.0) / dx / dx;
    fd_coeffs2[1] = static_cast<T>(1.0) / dx / dx;
  } else if (accuracy == 4) {
    fd_coeffs1[0] = static_cast<T>(8.0 / 12.0) / dx;
    fd_coeffs1[1] = -static_cast<T>(1.0 / 12.0) / dx;
    fd_coeffs2[0] = -static_cast<T>(5.0 / 2.0) / dx / dx;
    fd_coeffs2[1] = static_cast<T>(4.0 / 3.0) / dx / dx;
    fd_coeffs2[2] = -static_cast<T>(1.0 / 12.0) / dx / dx;
  } else if (accuracy == 6) {
    fd_coeffs1[0] = static_cast<T>(3.0 / 4.0) / dx;
    fd_coeffs1[1] = -static_cast<T>(3.0 / 20.0) / dx;
    fd_coeffs1[2] = static_cast<T>(1.0 / 60.0) / dx;
    fd_coeffs2[0] = -static_cast<T>(49.0 / 18.0) / dx / dx;
    fd_coeffs2[1] = static_cast<T>(3.0 / 2.0) / dx / dx;
    fd_coeffs2[2] = -static_cast<T>(3.0 / 20.0) / dx / dx;
    fd_coeffs2[3] = static_cast<T>(1.0 / 90.0) / dx / dx;
  } else {
    fd_coeffs1[0] = static_cast<T>(4.0 / 5.0) / dx;
    fd_coeffs1[1] = -static_cast<T>(1.0 / 5.0) / dx;
    fd_coeffs1[2] = static_cast<T>(4.0 / 105.0) / dx;
    fd_coeffs1[3] = -static_cast<T>(1.0 / 280.0) / dx;
    fd_coeffs2[0] = -static_cast<T>(205.0 / 72.0) / dx / dx;
    fd_coeffs2[1] = static_cast<T>(8.0 / 5.0) / dx / dx;
    fd_coeffs2[2] = -static_cast<T>(1.0 / 5.0) / dx / dx;
    fd_coeffs2[3] = static_cast<T>(8.0 / 315.0) / dx / dx;
    fd_coeffs2[4] = -static_cast<T>(1.0 / 560.0) / dx / dx;
  }
}

}  // namespace

class ScalarCUDAFunction
    : public torch::autograd::Function<ScalarCUDAFunction> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext *ctx, torch::Tensor const &v,
      torch::Tensor const &f, torch::Tensor const &wfc0,
      torch::Tensor const &wfp0, torch::Tensor const &psiy0,
      torch::Tensor const &psix0, torch::Tensor const &zetay0,
      torch::Tensor const &zetax0, torch::Tensor const &ay,
      torch::Tensor const &ax, torch::Tensor const &by, torch::Tensor const &bx,
      torch::Tensor const &sources_i, torch::Tensor const &receivers_i,
      double dy, double dx, double dt, int64_t nt, int64_t n_batch,
      int64_t step_ratio, int64_t accuracy, int64_t pml_width0,
      int64_t pml_width1, int64_t pml_width2, int64_t pml_width3) {
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(v.device());
    at::AutoDispatchBelowADInplaceOrView g;
    auto options{at::device(v.device()).dtype(v.scalar_type())};
    auto ny{v.size(0)};
    auto nx{v.size(1)};
    std::array<int64_t, 3> size_with_batch{n_batch, ny, nx};
    auto fd_pad{accuracy / 2};
    int64_t const pml_width[4] = {pml_width0, pml_width1, pml_width2,
                                  pml_width3};

    int64_t pml_regionsy0{fd_pad};
    int64_t pml_regionsy1{std::min(pml_width[0] + 2 * fd_pad, ny - fd_pad)};
    int64_t pml_regionsy2{
        std::max(pml_regionsy1, ny - pml_width[1] - 2 * fd_pad)};
    int64_t pml_regionsy3{ny - fd_pad};
    int64_t pml_regionsy[]{pml_regionsy0, pml_regionsy1, pml_regionsy2,
                           pml_regionsy3};
    int64_t pml_regionsx0{fd_pad};
    int64_t pml_regionsx1{std::min(pml_width[2] + 2 * fd_pad, nx - fd_pad)};
    int64_t pml_regionsx2{
        std::max(pml_regionsx1, nx - pml_width[3] - 2 * fd_pad)};
    int64_t pml_regionsx3{nx - fd_pad};
    int64_t pml_regionsx[]{pml_regionsx0, pml_regionsx1, pml_regionsx2,
                           pml_regionsx3};
    cudaMemcpyToSymbol(pml_regionsyc, pml_regionsy, sizeof(int64_t) * 4);
    cudaMemcpyToSymbol(pml_regionsxc, pml_regionsx, sizeof(int64_t) * 4);

    int64_t nynx{ny * nx};
    cudaMemcpyToSymbol(nxc, &nx, sizeof(int64_t));
    cudaMemcpyToSymbol(nynxc, &nynx, sizeof(int64_t));

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
    auto psiy{create_or_pad(psiy0, fd_pad, options, size_with_batch)};
    auto psix{create_or_pad(psix0, fd_pad, options, size_with_batch)};
    auto zetay{create_or_pad(zetay0, fd_pad, options, size_with_batch)};
    auto zetax{create_or_pad(zetax0, fd_pad, options, size_with_batch)};
    auto psiyn{at::zeros_like(psiy)};
    auto psixn{at::zeros_like(psix)};
    auto r{at::empty({nt, n_batch, n_receivers_per_shot}, options)};
    torch::Tensor dwdv;
    auto daydy{at::zeros_like(ay)};
    auto daxdx{at::zeros_like(ax)};
    auto dbydy{at::zeros_like(by)};
    auto dbxdx{at::zeros_like(bx)};
    if (v.requires_grad()) {
      dwdv = at::empty({(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx},
                       options);
    }

    zero_interior<true>(psiy, ny, nx, fd_pad, pml_width);
    zero_interior<false>(psix, ny, nx, fd_pad, pml_width);
    zero_interior<true>(zetay, ny, nx, fd_pad, pml_width);
    zero_interior<false>(zetax, ny, nx, fd_pad, pml_width);

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cuda_forward", ([&] {
          scalar_t dt2_a = dt * dt;
          cudaMemcpyToSymbol(dt2c, &dt2_a, sizeof(scalar_t));
          scalar_t const *__restrict v_a{v.data_ptr<scalar_t>()};
          scalar_t const *__restrict f_a{f.data_ptr<scalar_t>()};
          scalar_t *__restrict r_a{r.data_ptr<scalar_t>()};
          scalar_t *__restrict wfc_a{wfc.data_ptr<scalar_t>()};
          scalar_t *__restrict wfp_a{wfp.data_ptr<scalar_t>()};
          scalar_t *__restrict psiy_a{psiy.data_ptr<scalar_t>()};
          scalar_t *__restrict psix_a{psix.data_ptr<scalar_t>()};
          scalar_t *__restrict psiyn_a{psiyn.data_ptr<scalar_t>()};
          scalar_t *__restrict psixn_a{psixn.data_ptr<scalar_t>()};
          scalar_t *__restrict zetay_a{zetay.data_ptr<scalar_t>()};
          scalar_t *__restrict zetax_a{zetax.data_ptr<scalar_t>()};
          scalar_t const *__restrict ay_a{ay.data_ptr<scalar_t>()};
          scalar_t const *__restrict ax_a{ax.data_ptr<scalar_t>()};
          scalar_t const *__restrict by_a{by.data_ptr<scalar_t>()};
          scalar_t const *__restrict bx_a{bx.data_ptr<scalar_t>()};
          scalar_t *__restrict daydy_a{daydy.data_ptr<scalar_t>()};
          scalar_t *__restrict daxdx_a{daxdx.data_ptr<scalar_t>()};
          scalar_t *__restrict dbydy_a{dbydy.data_ptr<scalar_t>()};
          scalar_t *__restrict dbxdx_a{dbxdx.data_ptr<scalar_t>()};
          int64_t const *__restrict sources_i_a{sources_i.data_ptr<int64_t>()};
          int64_t const *__restrict receivers_i_a{
              receivers_i.data_ptr<int64_t>()};
          scalar_t *__restrict dwdv_a{};
          if (v.requires_grad()) {
            dwdv_a = dwdv.data_ptr<scalar_t>();
          }
          scalar_t fd_coeffs1h[4];
          scalar_t fd_coeffs2h[5];
          set_fd_coeffs<scalar_t>(fd_coeffs1h, fd_coeffs2h, accuracy,
                                  static_cast<scalar_t>(dy));
          cudaMemcpyToSymbol(fd_coeffs1yc, fd_coeffs1h, sizeof(scalar_t) * 4);
          cudaMemcpyToSymbol(fd_coeffs2yc, fd_coeffs2h, sizeof(scalar_t) * 5);
          set_fd_coeffs<scalar_t>(fd_coeffs1h, fd_coeffs2h, accuracy,
                                  static_cast<scalar_t>(dx));
          cudaMemcpyToSymbol(fd_coeffs1xc, fd_coeffs1h, sizeof(scalar_t) * 4);
          cudaMemcpyToSymbol(fd_coeffs2xc, fd_coeffs2h, sizeof(scalar_t) * 5);
          gpuErrchk(cudaPeekAtLastError());
          decltype(&diffy<scalar_t, 4>) diffys[]{
              diffy<scalar_t, 2>, diffy<scalar_t, 4>, diffy<scalar_t, 6>,
              diffy<scalar_t, 8>};
          auto diffyi{diffys[accuracy / 2 - 1]};
          decltype(&diffx<scalar_t, 4>) diffxs[]{
              diffx<scalar_t, 2>, diffx<scalar_t, 4>, diffx<scalar_t, 6>,
              diffx<scalar_t, 8>};
          auto diffxi{diffxs[accuracy / 2 - 1]};
          diffyi<<<32, ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              ay_a, daydy_a);
          diffxi<<<32, ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              ax_a, daxdx_a);
          diffyi<<<32, ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              by_a, dbydy_a);
          diffxi<<<32, ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              bx_a, dbxdx_a);
          decltype(&forward_batch<scalar_t, 4>) forward_batches[]{
              forward_batch<scalar_t, 2>, forward_batch<scalar_t, 4>,
              forward_batch<scalar_t, 6>, forward_batch<scalar_t, 8>};
          forward_batches[accuracy / 2 - 1](
              wfc_a, wfp_a, psiy_a, psix_a, psiyn_a, psixn_a, zetay_a, zetax_a,
              sources_i_a, receivers_i_a, dwdv_a, v_a, f_a, r_a, ay_a, ax_a,
              by_a, bx_a, daydy_a, daxdx_a, dbydy_a, dbxdx_a,
              n_sources_per_shot, n_receivers_per_shot, ny, nx, nt, step_ratio,
              v.requires_grad(), n_batch);
        }));
    if (v.requires_grad() or f.requires_grad() or wfc0.requires_grad() or
        wfp0.requires_grad() or psiy0.requires_grad() or
        psix0.requires_grad() or zetay0.requires_grad() or
        zetax0.requires_grad()) {
      ctx->save_for_backward({v, ay, ax, by, bx, sources_i, receivers_i});
      ctx->saved_data["dwdv"] = dwdv;
      ctx->saved_data["dy"] = dy;
      ctx->saved_data["dx"] = dx;
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
    at::indexing::TensorIndex slicey{
        torch::indexing::Slice(fd_pad, ny - fd_pad)};
    at::indexing::TensorIndex slicex{
        torch::indexing::Slice(fd_pad, nx - fd_pad)};
    if (nt & 1) {
      return {wfp.index({all_slice, slicey, slicex}),
              wfc.index({all_slice, slicey, slicex}),
              psiyn.index({all_slice, slicey, slicex}),
              psixn.index({all_slice, slicey, slicex}),
              zetay.index({all_slice, slicey, slicex}),
              zetax.index({all_slice, slicey, slicex}),
              r};
    }
    return {wfc.index({all_slice, slicey, slicex}),
            wfp.index({all_slice, slicey, slicex}),
            psiy.index({all_slice, slicey, slicex}),
            psix.index({all_slice, slicey, slicex}),
            zetay.index({all_slice, slicey, slicex}),
            zetax.index({all_slice, slicey, slicex}),
            r};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list const &grad_outputs) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto saved{ctx->get_saved_variables()};
    auto const &v{saved[0]};
    auto const &ay{saved[1]};
    auto const &ax{saved[2]};
    auto const &by{saved[3]};
    auto const &bx{saved[4]};
    auto const &sources_i{saved[5]};
    auto const &receivers_i{saved[6]};
    auto const &dwdv{ctx->saved_data["dwdv"].toTensor()};
    auto dy{ctx->saved_data["dy"].toDouble()};
    auto dx{ctx->saved_data["dx"].toDouble()};
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
    auto ny{v.size(0)};
    auto nx{v.size(1)};
    int64_t pml_regionsy0{fd_pad};
    int64_t pml_regionsy1{std::min(pml_width[0] + 3 * fd_pad, ny - fd_pad)};
    int64_t pml_regionsy2{
        std::max(pml_regionsy1, ny - pml_width[1] - 3 * fd_pad)};
    int64_t pml_regionsy3{ny - fd_pad};
    int64_t pml_regionsy[]{pml_regionsy0, pml_regionsy1, pml_regionsy2,
                           pml_regionsy3};
    int64_t pml_regionsx0{fd_pad};
    int64_t pml_regionsx1{std::min(pml_width[2] + 3 * fd_pad, nx - fd_pad)};
    int64_t pml_regionsx2{
        std::max(pml_regionsx1, nx - pml_width[3] - 3 * fd_pad)};
    int64_t pml_regionsx3{nx - fd_pad};
    int64_t pml_regionsx[]{pml_regionsx0, pml_regionsx1, pml_regionsx2,
                           pml_regionsx3};
    cudaMemcpyToSymbol(pml_regionsyc, pml_regionsy, sizeof(int64_t) * 4);
    cudaMemcpyToSymbol(pml_regionsxc, pml_regionsx, sizeof(int64_t) * 4);

    int64_t nynx{ny * nx};
    cudaMemcpyToSymbol(nxc, &nx, sizeof(int64_t));
    cudaMemcpyToSymbol(nynxc, &nynx, sizeof(int64_t));

    auto wfc =
        at::constant_pad_nd(grad_outputs[0], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto wfp =
        at::constant_pad_nd(grad_outputs[1], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psiy =
        at::constant_pad_nd(grad_outputs[2], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psix =
        at::constant_pad_nd(grad_outputs[3], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetay =
        at::constant_pad_nd(grad_outputs[4], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetax =
        at::constant_pad_nd(grad_outputs[5], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto wfcn{at::zeros_like(wfc)};
    auto psiyn{at::zeros_like(psiy)};
    auto psixn{at::zeros_like(psix)};
    auto zetayn{at::zeros_like(zetay)};
    auto zetaxn{at::zeros_like(zetax)};
    auto grad_r{grad_outputs[6].contiguous()};
    auto daydy{at::zeros_like(ay)};
    auto daxdx{at::zeros_like(ax)};
    auto dbydy{at::zeros_like(by)};
    auto dbxdx{at::zeros_like(bx)};
    auto options{at::device(v.device()).dtype(v.scalar_type())};
    int64_t n_sources_per_shot{};
    if (sources_i.numel() > 0) {
      n_sources_per_shot = sources_i.size(1);
    }
    int64_t n_receivers_per_shot{};
    if (receivers_i.numel() > 0) {
      n_receivers_per_shot = receivers_i.size(1);
    }

    auto grad_v{at::zeros({ny, nx}, options)};
    auto grad_v_batch{n_batch > 1 ? at::zeros({n_batch, ny, nx}, options)
                                  : at::empty(0)};
    auto grad_f{at::empty({nt, n_batch, n_sources_per_shot}, options)};

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cuda_backward", ([&] {
          scalar_t dt2_a = dt * dt;
          cudaMemcpyToSymbol(dt2c, &dt2_a, sizeof(scalar_t));
          auto v2dt2{v * v * dt2_a};
          scalar_t const *__restrict v2dt2_a{v2dt2.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_v_a{grad_v.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_v_batch_a{
              n_batch > 1 ? grad_v_batch.data_ptr<scalar_t>() : grad_v_a};
          scalar_t const *__restrict grad_r_a{grad_r.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_f_a{grad_f.data_ptr<scalar_t>()};
          scalar_t *__restrict wfc_a{wfc.data_ptr<scalar_t>()};
          scalar_t *__restrict wfp_a{wfp.data_ptr<scalar_t>()};
          scalar_t *__restrict wfcn_a{wfcn.data_ptr<scalar_t>()};
          scalar_t *__restrict psiy_a{psiy.data_ptr<scalar_t>()};
          scalar_t *__restrict psix_a{psix.data_ptr<scalar_t>()};
          scalar_t *__restrict psiyn_a{psiyn.data_ptr<scalar_t>()};
          scalar_t *__restrict psixn_a{psixn.data_ptr<scalar_t>()};
          scalar_t *__restrict zetay_a{zetay.data_ptr<scalar_t>()};
          scalar_t *__restrict zetax_a{zetax.data_ptr<scalar_t>()};
          scalar_t *__restrict zetayn_a{zetayn.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaxn_a{zetaxn.data_ptr<scalar_t>()};
          scalar_t const *__restrict ay_a{ay.data_ptr<scalar_t>()};
          scalar_t const *__restrict ax_a{ax.data_ptr<scalar_t>()};
          scalar_t const *__restrict by_a{by.data_ptr<scalar_t>()};
          scalar_t const *__restrict bx_a{bx.data_ptr<scalar_t>()};
          scalar_t *__restrict daydy_a{daydy.data_ptr<scalar_t>()};
          scalar_t *__restrict daxdx_a{daxdx.data_ptr<scalar_t>()};
          scalar_t *__restrict dbydy_a{dbydy.data_ptr<scalar_t>()};
          scalar_t *__restrict dbxdx_a{dbxdx.data_ptr<scalar_t>()};
          int64_t const *__restrict sources_i_a{sources_i.data_ptr<int64_t>()};
          int64_t const *__restrict receivers_i_a{
              receivers_i.data_ptr<int64_t>()};
          scalar_t const *__restrict dwdv_a{};
          if (v.requires_grad()) {
            dwdv_a = dwdv.data_ptr<scalar_t>();
          }
          scalar_t fd_coeffs1h[4];
          scalar_t fd_coeffs2h[5];
          set_fd_coeffs<scalar_t>(fd_coeffs1h, fd_coeffs2h, accuracy,
                                  static_cast<scalar_t>(dy));
          cudaMemcpyToSymbol(fd_coeffs1yc, fd_coeffs1h, sizeof(scalar_t) * 4);
          cudaMemcpyToSymbol(fd_coeffs2yc, fd_coeffs2h, sizeof(scalar_t) * 5);
          set_fd_coeffs<scalar_t>(fd_coeffs1h, fd_coeffs2h, accuracy,
                                  static_cast<scalar_t>(dx));
          cudaMemcpyToSymbol(fd_coeffs1xc, fd_coeffs1h, sizeof(scalar_t) * 4);
          cudaMemcpyToSymbol(fd_coeffs2xc, fd_coeffs2h, sizeof(scalar_t) * 5);
          gpuErrchk(cudaPeekAtLastError());
          decltype(&diffy<scalar_t, 4>) diffys[]{
              diffy<scalar_t, 2>, diffy<scalar_t, 4>, diffy<scalar_t, 6>,
              diffy<scalar_t, 8>};
          auto diffyi{diffys[accuracy / 2 - 1]};
          decltype(&diffx<scalar_t, 4>) diffxs[]{
              diffx<scalar_t, 2>, diffx<scalar_t, 4>, diffx<scalar_t, 6>,
              diffx<scalar_t, 8>};
          auto diffxi{diffxs[accuracy / 2 - 1]};
          diffyi<<<32, ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              ay_a, daydy_a);
          diffxi<<<32, ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              ax_a, daxdx_a);
          diffyi<<<32, ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              by_a, dbydy_a);
          diffxi<<<32, ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(32))>>>(
              bx_a, dbxdx_a);
          decltype(&backward_batch<scalar_t, 4>) backward_batches[]{
              backward_batch<scalar_t, 2>, backward_batch<scalar_t, 4>,
              backward_batch<scalar_t, 6>, backward_batch<scalar_t, 8>};
          backward_batches[accuracy / 2 - 1](
              wfc_a, wfp_a, wfcn_a, psiy_a, psix_a, psiyn_a, psixn_a, zetay_a,
              zetax_a, zetayn_a, zetaxn_a, sources_i_a, receivers_i_a, dwdv_a,
              v2dt2_a, grad_f_a, grad_r_a, grad_v_batch_a, ay_a, ax_a, by_a,
              bx_a, daydy_a, daxdx_a, dbydy_a, dbxdx_a, n_sources_per_shot,
              n_receivers_per_shot, ny, nx, nt, step_ratio, v.requires_grad(),
              n_batch);
          if (v.requires_grad() and n_batch > 1) {
            dim3 dimBlock_combine(32, 32, 1);
            auto gridx_combine{ceil_div(
                nx - 2 * fd_pad, static_cast<int64_t>(dimBlock_combine.x))};
            auto gridy_combine{ceil_div(
                ny - 2 * fd_pad, static_cast<int64_t>(dimBlock_combine.y))};
            auto gridz_combine{1};
            dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
            decltype(&combine_grad_v<scalar_t, 4>) combine_grad_vs[]{
                combine_grad_v<scalar_t, 2>, combine_grad_v<scalar_t, 4>,
                combine_grad_v<scalar_t, 6>, combine_grad_v<scalar_t, 8>};
            auto combine_grad_vi{combine_grad_vs[accuracy / 2 - 1]};
            combine_grad_vi<<<dimGrid_combine, dimBlock_combine>>>(
                grad_v_a, grad_v_batch_a, n_batch);
            gpuErrchk(cudaPeekAtLastError());
          }
        }));

    torch::Tensor *wfnt;
    torch::Tensor *wfntm1;
    torch::Tensor *psiyntm1{nt & 1 ? &psiyn : &psiy};
    torch::Tensor *psixntm1{nt & 1 ? &psixn : &psix};
    torch::Tensor *zetayntm1{nt & 1 ? &zetayn : &zetay};
    torch::Tensor *zetaxntm1{nt & 1 ? &zetaxn : &zetax};
    if (nt % 3 == 1) {
      wfnt = &wfp;
      wfntm1 = &wfcn;
    } else if (nt % 3 == 2) {
      wfnt = &wfcn;
      wfntm1 = &wfc;
    } else {
      wfnt = &wfc;
      wfntm1 = &wfp;
    }

    zero_interior<true>(*psiyntm1, ny, nx, fd_pad, pml_width);
    zero_interior<false>(*psixntm1, ny, nx, fd_pad, pml_width);
    zero_interior<true>(*zetayntm1, ny, nx, fd_pad, pml_width);
    zero_interior<false>(*zetaxntm1, ny, nx, fd_pad, pml_width);

    at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
    at::indexing::TensorIndex slicey{
        torch::indexing::Slice(fd_pad, ny - fd_pad)};
    at::indexing::TensorIndex slicex{
        torch::indexing::Slice(fd_pad, nx - fd_pad)};
    return {grad_v,
            grad_f,
            wfnt->index({all_slice, slicey, slicex}),
            wfntm1->index({all_slice, slicey, slicex}),
            psiyntm1->index({all_slice, slicey, slicex}),
            psixntm1->index({all_slice, slicey, slicex}),
            zetayntm1->index({all_slice, slicey, slicex}),
            zetaxntm1->index({all_slice, slicey, slicex}),
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
    torch::Tensor const &wfp0, torch::Tensor const &psiy0,
    torch::Tensor const &psix0, torch::Tensor const &zetay0,
    torch::Tensor const &zetax0, torch::Tensor const &ay,
    torch::Tensor const &ax, torch::Tensor const &by, torch::Tensor const &bx,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i, double dy,
    double dx, double dt, int64_t nt, int64_t n_batch, int64_t step_ratio,
    int64_t accuracy, int64_t pml_width0, int64_t pml_width1,
    int64_t pml_width2, int64_t pml_width3) {
  return ScalarCUDAFunction::apply(
      v, f, wfc0, wfp0, psiy0, psix0, zetay0, zetax0, ay, ax, by, bx, sources_i,
      receivers_i, dy, dx, dt, nt, n_batch, step_ratio, accuracy, pml_width0,
      pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCUDA, m) {
  m.impl("scalar", scalar_cuda_autograd);
}

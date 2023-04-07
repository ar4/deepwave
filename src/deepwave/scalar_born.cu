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

template <typename T, int A, bool v_requires_grad, bool scatter_requires_grad>
__global__ void forward_kernel(
    T const *__restrict wfc, T *__restrict wfp, T const *__restrict psiy,
    T const *__restrict psix, T *__restrict psiyn, T *__restrict psixn,
    T *__restrict zetay, T *__restrict zetax, T const *__restrict wfcsc,
    T *__restrict wfpsc, T const *__restrict psiysc, T const *__restrict psixsc,
    T *__restrict psiynsc, T *__restrict psixnsc, T *__restrict zetaysc,
    T *__restrict zetaxsc, T *__restrict w_store, T *__restrict wsc_store,
    T const *__restrict v, T const *__restrict scatter, T const *__restrict ay,
    T const *__restrict ax, T const *__restrict by, T const *__restrict bx,
    T const *__restrict daydy, T const *__restrict daxdx,
    T const *__restrict dbydy, T const *__restrict dbxdx) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{y * nxc + x};
  auto i{batch * nynxc + j};
  if (y < pml_regionsyc[3] && x < pml_regionsxc[3]) {
    T d2wdy2;
    T d2wdx2;
    T d2wscdy2;
    T d2wscdx2;
#define D2WDY20 fd_coeffs2y<T>(0) * wfc[i]
#define D2WDY2(t) fd_coeffs2y<T>(t) * (wfc[i + t * nxc] + wfc[i - t * nxc])
#define D2WDX20 fd_coeffs2x<T>(0) * wfc[i]
#define D2WDX2(t) fd_coeffs2x<T>(t) * (wfc[i + t] + wfc[i - t])
#define D2WSCDY20 fd_coeffs2y<T>(0) * wfcsc[i]
#define D2WSCDY2(t) \
  fd_coeffs2y<T>(t) * (wfcsc[i + t * nxc] + wfcsc[i - t * nxc])
#define D2WSCDX20 fd_coeffs2x<T>(0) * wfcsc[i]
#define D2WSCDX2(t) fd_coeffs2x<T>(t) * (wfcsc[i + t] + wfcsc[i - t])
    if (A == 2) {
      d2wdy2 = D2WDY20 + D2WDY2(1);
      d2wdx2 = D2WDX20 + D2WDX2(1);
      d2wscdy2 = D2WSCDY20 + D2WSCDY2(1);
      d2wscdx2 = D2WSCDX20 + D2WSCDX2(1);
    } else if (A == 4) {
      d2wdy2 = D2WDY20 + D2WDY2(1) + D2WDY2(2);
      d2wdx2 = D2WDX20 + D2WDX2(1) + D2WDX2(2);
      d2wscdy2 = D2WSCDY20 + D2WSCDY2(1) + D2WSCDY2(2);
      d2wscdx2 = D2WSCDX20 + D2WSCDX2(1) + D2WSCDX2(2);
    } else if (A == 6) {
      d2wdy2 = D2WDY20 + D2WDY2(1) + D2WDY2(2) + D2WDY2(3);
      d2wdx2 = D2WDX20 + D2WDX2(1) + D2WDX2(2) + D2WDX2(3);
      d2wscdy2 = D2WSCDY20 + D2WSCDY2(1) + D2WSCDY2(2) + D2WSCDY2(3);
      d2wscdx2 = D2WSCDX20 + D2WSCDX2(1) + D2WSCDX2(2) + D2WSCDX2(3);
    } else {
      d2wdy2 = D2WDY20 + D2WDY2(1) + D2WDY2(2) + D2WDY2(3) + D2WDY2(4);
      d2wdx2 = D2WDX20 + D2WDX2(1) + D2WDX2(2) + D2WDX2(3) + D2WDX2(4);
      d2wscdy2 =
          D2WSCDY20 + D2WSCDY2(1) + D2WSCDY2(2) + D2WSCDY2(3) + D2WSCDY2(4);
      d2wscdx2 =
          D2WSCDX20 + D2WSCDX2(1) + D2WSCDX2(2) + D2WSCDX2(3) + D2WSCDX2(4);
    }

    T w_sum{};
    T wsc_sum{};
    if (y < pml_regionsyc[1] || y >= pml_regionsyc[2]) {
      T dwdy;
      T dpsiydy;
      T dwscdy;
      T dpsiyscdy;
#define DDY(a, t) \
  fd_coeffs1y<T>(t) * (a[i + (t + 1) * nxc] - a[i - (t + 1) * nxc])
      if (A == 2) {
        dwdy = DDY(wfc, 0);
        dpsiydy = DDY(psiy, 0);
        dwscdy = DDY(wfcsc, 0);
        dpsiyscdy = DDY(psiysc, 0);
      } else if (A == 4) {
        dwdy = DDY(wfc, 0) + DDY(wfc, 1);
        dpsiydy = DDY(psiy, 0) + DDY(psiy, 1);
        dwscdy = DDY(wfcsc, 0) + DDY(wfcsc, 1);
        dpsiyscdy = DDY(psiysc, 0) + DDY(psiysc, 1);
      } else if (A == 6) {
        dwdy = DDY(wfc, 0) + DDY(wfc, 1) + DDY(wfc, 2);
        dpsiydy = DDY(psiy, 0) + DDY(psiy, 1) + DDY(psiy, 2);
        dwscdy = DDY(wfcsc, 0) + DDY(wfcsc, 1) + DDY(wfcsc, 2);
        dpsiyscdy = DDY(psiysc, 0) + DDY(psiysc, 1) + DDY(psiysc, 2);
      } else {
        dwdy = DDY(wfc, 0) + DDY(wfc, 1) + DDY(wfc, 2) + DDY(wfc, 3);
        dpsiydy = DDY(psiy, 0) + DDY(psiy, 1) + DDY(psiy, 2) + DDY(psiy, 3);
        dwscdy = DDY(wfcsc, 0) + DDY(wfcsc, 1) + DDY(wfcsc, 2) + DDY(wfcsc, 3);
        dpsiyscdy =
            DDY(psiysc, 0) + DDY(psiysc, 1) + DDY(psiysc, 2) + DDY(psiysc, 3);
      }
      T tmpy{(1 + by[y]) * d2wdy2 + dbydy[y] * dwdy + daydy[y] * psiy[i] +
             ay[y] * dpsiydy};
      T tmpysc{(1 + by[y]) * d2wscdy2 + dbydy[y] * dwscdy +
               daydy[y] * psiysc[i] + ay[y] * dpsiyscdy};
      w_sum += (1 + by[y]) * tmpy + ay[y] * zetay[i];
      wsc_sum += (1 + by[y]) * tmpysc + ay[y] * zetaysc[i];
      psiyn[i] = by[y] * dwdy + ay[y] * psiy[i];
      zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
      psiynsc[i] = by[y] * dwscdy + ay[y] * psiysc[i];
      zetaysc[i] = by[y] * tmpysc + ay[y] * zetaysc[i];
    } else {
      w_sum += d2wdy2;
      wsc_sum += d2wscdy2;
    }
    if (x < pml_regionsxc[1] || x >= pml_regionsxc[2]) {
      T dwdx;
      T dpsixdx;
      T dwscdx;
      T dpsixscdx;
#define DDX(a, t) fd_coeffs1x<T>(t) * (a[i + (t + 1)] - a[i - (t + 1)])
      if (A == 2) {
        dwdx = DDX(wfc, 0);
        dpsixdx = DDX(psix, 0);
        dwscdx = DDX(wfcsc, 0);
        dpsixscdx = DDX(psixsc, 0);
      } else if (A == 4) {
        dwdx = DDX(wfc, 0) + DDX(wfc, 1);
        dpsixdx = DDX(psix, 0) + DDX(psix, 1);
        dwscdx = DDX(wfcsc, 0) + DDX(wfcsc, 1);
        dpsixscdx = DDX(psixsc, 0) + DDX(psixsc, 1);
      } else if (A == 6) {
        dwdx = DDX(wfc, 0) + DDX(wfc, 1) + DDX(wfc, 2);
        dpsixdx = DDX(psix, 0) + DDX(psix, 1) + DDX(psix, 2);
        dwscdx = DDX(wfcsc, 0) + DDX(wfcsc, 1) + DDX(wfcsc, 2);
        dpsixscdx = DDX(psixsc, 0) + DDX(psixsc, 1) + DDX(psixsc, 2);
      } else {
        dwdx = DDX(wfc, 0) + DDX(wfc, 1) + DDX(wfc, 2) + DDX(wfc, 3);
        dpsixdx = DDX(psix, 0) + DDX(psix, 1) + DDX(psix, 2) + DDX(psix, 3);
        dwscdx = DDX(wfcsc, 0) + DDX(wfcsc, 1) + DDX(wfcsc, 2) + DDX(wfcsc, 3);
        dpsixscdx =
            DDX(psixsc, 0) + DDX(psixsc, 1) + DDX(psixsc, 2) + DDX(psixsc, 3);
      }
      T tmpx{(1 + bx[x]) * d2wdx2 + dbxdx[x] * dwdx + daxdx[x] * psix[i] +
             ax[x] * dpsixdx};
      T tmpxsc{(1 + bx[x]) * d2wscdx2 + dbxdx[x] * dwscdx +
               daxdx[x] * psixsc[i] + ax[x] * dpsixscdx};
      w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];
      wsc_sum += (1 + bx[x]) * tmpxsc + ax[x] * zetaxsc[i];
      psixn[i] = bx[x] * dwdx + ax[x] * psix[i];
      zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];
      psixnsc[i] = bx[x] * dwscdx + ax[x] * psixsc[i];
      zetaxsc[i] = bx[x] * tmpxsc + ax[x] * zetaxsc[i];
    } else {
      w_sum += d2wdx2;
      wsc_sum += d2wscdx2;
    }
    if (v_requires_grad || scatter_requires_grad) {
      w_store[i] = w_sum;
    }
    if (v_requires_grad) {
      wsc_store[i] = wsc_sum;
    }
    wfp[i] = v[j] * v[j] * dt2<T>() * w_sum + 2 * wfc[i] - wfp[i];
    wfpsc[i] = v[j] * v[j] * dt2<T>() * wsc_sum +
               2 * scatter[j] * v[j] * dt2<T>() * w_sum + 2 * wfcsc[i] -
               wfpsc[i];
  }
}

template <typename T>
__global__ void add_sources(T *__restrict wf, T const *__restrict f,
                            int64_t const *__restrict sources_i,
                            int64_t n_sources_per_shot, int64_t n_shots) {
  auto source_idx{blockIdx.x * blockDim.x + threadIdx.x};
  auto shot_idx{blockIdx.y * blockDim.y + threadIdx.y};
  if (source_idx >= n_sources_per_shot || shot_idx >= n_shots) return;
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
  if (receiver_idx >= n_receivers_per_shot || shot_idx >= n_shots) return;
  auto k{shot_idx * n_receivers_per_shot + receiver_idx};
  r[k] = wf[shot_idx * nynxc + receivers_i[k]];
}

template <typename T, int A>
__global__ void add_to_grad_v(T *__restrict grad_v, T const *__restrict wfc,
                              T const *__restrict wfcsc,
                              T const *__restrict w_store,
                              T const *__restrict wsc_store,
                              T const *__restrict v,
                              T const *__restrict scatter, int64_t step_ratio) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{y * nxc + x};
  auto i{batch * nynxc + j};
  if (y < pml_regionsyc[3] && x < pml_regionsxc[3]) {
    grad_v[i] += wfc[i] * 2 * v[j] * dt2<T>() * w_store[i] * step_ratio +
                 wfcsc[i] *
                     (2 * dt2<T>() * scatter[j] * w_store[i] +
                      2 * v[j] * dt2<T>() * wsc_store[i]) *
                     step_ratio;
  }
}

template <typename T, int A>
__global__ void add_to_grad_scatter(T *__restrict grad_scatter,
                                    T const *__restrict wfcsc,
                                    T const *__restrict w_store,
                                    T const *__restrict v, int64_t step_ratio) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{y * nxc + x};
  auto i{batch * nynxc + j};
  if (y < pml_regionsyc[3] && x < pml_regionsxc[3]) {
    grad_scatter[i] += wfcsc[i] * 2 * v[j] * dt2<T>() * w_store[i] * step_ratio;
  }
}

template <typename T, int A>
__global__ void backward_kernel(
    T const *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
    T const *__restrict psiy, T const *__restrict psix, T *__restrict psiyn,
    T *__restrict psixn, T const *__restrict zetay, T const *__restrict zetax,
    T *__restrict zetayn, T *__restrict zetaxn, T const *__restrict wfcsc,
    T *__restrict wfpsc, T *__restrict wfcnsc, T const *__restrict psiysc,
    T const *__restrict psixsc, T *__restrict psiynsc, T *__restrict psixnsc,
    T const *__restrict zetaysc, T const *__restrict zetaxsc,
    T *__restrict zetaynsc, T *__restrict zetaxnsc, T const *__restrict v,
    T const *__restrict scatter, T const *__restrict ay, T const *__restrict ax,
    T const *__restrict by, T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto j{y * nxc + x};
  auto i{batch * nynxc + j};
  if (y < pml_regionsyc[3] && x < pml_regionsxc[3]) {
    bool pml_y{y < pml_regionsyc[1] || y >= pml_regionsyc[2]};
    bool pml_x{x < pml_regionsxc[1] || x >= pml_regionsxc[2]};
    T wfp_y_term;
    T wfp_x_term;
    T wfpsc_y_term;
    T wfpsc_x_term;

#define WFPY0                                            \
  fd_coeffs2y<T>(0) * (v[j] * v[j] * dt2<T>() * wfc[i] + \
                       2 * v[j] * dt2<T>() * scatter[j] * wfcsc[i])
#define WFPY(t)                                                        \
  fd_coeffs2y<T>(t) *                                                  \
      (v[j + t * nxc] * v[j + t * nxc] * dt2<T>() * wfc[i + t * nxc] + \
       2 * v[j + t * nxc] * dt2<T>() * scatter[j + t * nxc] *          \
           wfcsc[i + t * nxc] +                                        \
       v[j - t * nxc] * v[j - t * nxc] * dt2<T>() * wfc[i - t * nxc] + \
       2 * v[j - t * nxc] * dt2<T>() * scatter[j - t * nxc] *          \
           wfcsc[i - t * nxc])
#define WFPYPML0                                                      \
  fd_coeffs2y<T>(0) *                                                 \
      ((1 + by[y]) * ((1 + by[y]) * v[j] * v[j] * dt2<T>() * wfc[i] + \
                      by[y] * zetay[i]) +                             \
       (1 + by[y]) * (1 + by[y]) * 2 * v[j] * dt2<T>() * scatter[j] * \
           wfcsc[i])
#define WFPYPML(t)                                                             \
  fd_coeffs2y<T>(t) *                                                          \
          (((1 + by[y + t]) *                                                  \
            ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * dt2<T>() *    \
                 wfc[i + t * nxc] +                                            \
             by[y + t] * zetay[i + t * nxc])) +                                \
           (1 + by[y + t]) * (1 + by[y + t]) * 2 * v[j + t * nxc] * dt2<T>() * \
               scatter[j + t * nxc] * wfcsc[i + t * nxc] +                     \
           ((1 + by[y - t]) *                                                  \
            ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * dt2<T>() *    \
                 wfc[i - t * nxc] +                                            \
             by[y - t] * zetay[i - t * nxc])) +                                \
           (1 + by[y - t]) * (1 + by[y - t]) * 2 * v[j - t * nxc] * dt2<T>() * \
               scatter[j - t * nxc] * wfcsc[i - t * nxc]) -                    \
      (fd_coeffs1y<T>(t - 1) *                                                 \
       ((dbydy[y + t] * ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] *   \
                             dt2<T>() * wfc[i + t * nxc] +                     \
                         by[y + t] * zetay[i + t * nxc] +                      \
                         (1 + by[y + t]) * 2 * v[j + t * nxc] * dt2<T>() *     \
                             scatter[j + t * nxc] * wfcsc[i + t * nxc]) +      \
         by[y + t] * psiy[i + t * nxc]) -                                      \
        (dbydy[y - t] * ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] *   \
                             dt2<T>() * wfc[i - t * nxc] +                     \
                         by[y - t] * zetay[i - t * nxc] +                      \
                         (1 + by[y - t]) * 2 * v[j - t * nxc] * dt2<T>() *     \
                             scatter[j - t * nxc] * wfcsc[i - t * nxc]) +      \
         by[y - t] * psiy[i - t * nxc])))

#define WFPSCY0 fd_coeffs2y<T>(0) * v[j] * v[j] * dt2<T>() * wfcsc[i]
#define WFPSCY(t)                                                        \
  fd_coeffs2y<T>(t) *                                                    \
      (v[j + t * nxc] * v[j + t * nxc] * dt2<T>() * wfcsc[i + t * nxc] + \
       v[j - t * nxc] * v[j - t * nxc] * dt2<T>() * wfcsc[i - t * nxc])
#define WFPSCYPML0    \
  fd_coeffs2y<T>(0) * \
      ((1 + by[y]) *  \
       ((1 + by[y]) * v[j] * v[j] * dt2<T>() * wfcsc[i] + by[y] * zetaysc[i]))
#define WFPSCYPML(t)                                                         \
  fd_coeffs2y<T>(t) * (((1 + by[y + t]) *                                    \
                        ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                             dt2<T>() * wfcsc[i + t * nxc] +                 \
                         by[y + t] * zetaysc[i + t * nxc])) +                \
                       ((1 + by[y - t]) *                                    \
                        ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                             dt2<T>() * wfcsc[i - t * nxc] +                 \
                         by[y - t] * zetaysc[i - t * nxc]))) -               \
      (fd_coeffs1y<T>(t - 1) *                                               \
       ((dbydy[y + t] * ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                             dt2<T>() * wfcsc[i + t * nxc] +                 \
                         by[y + t] * zetaysc[i + t * nxc]) +                 \
         by[y + t] * psiysc[i + t * nxc]) -                                  \
        (dbydy[y - t] * ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                             dt2<T>() * wfcsc[i - t * nxc] +                 \
                         by[y - t] * zetaysc[i - t * nxc]) +                 \
         by[y - t] * psiysc[i - t * nxc])))

    if (A == 2) {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1);
        wfpsc_y_term = WFPSCY0 + WFPSCY(1);
      } else {
        wfp_y_term = WFPYPML0 + WFPYPML(1);
        wfpsc_y_term = WFPSCYPML0 + WFPSCYPML(1);
      }
    } else if (A == 4) {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1) + WFPY(2);
        wfpsc_y_term = WFPSCY0 + WFPSCY(1) + WFPSCY(2);
      } else {
        wfp_y_term = WFPYPML0 + WFPYPML(1) + WFPYPML(2);
        wfpsc_y_term = WFPSCYPML0 + WFPSCYPML(1) + WFPSCYPML(2);
      }
    } else if (A == 6) {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1) + WFPY(2) + WFPY(3);
        wfpsc_y_term = WFPSCY0 + WFPSCY(1) + WFPSCY(2) + WFPSCY(3);
      } else {
        wfp_y_term = WFPYPML0 + WFPYPML(1) + WFPYPML(2) + WFPYPML(3);
        wfpsc_y_term = WFPSCYPML0 + WFPSCYPML(1) + WFPSCYPML(2) + WFPSCYPML(3);
      }
    } else {
      if (not pml_y) {
        wfp_y_term = WFPY0 + WFPY(1) + WFPY(2) + WFPY(3) + WFPY(4);
        wfpsc_y_term = WFPSCY0 + WFPSCY(1) + WFPSCY(2) + WFPSCY(3) + WFPSCY(4);
      } else {
        wfp_y_term =
            WFPYPML0 + WFPYPML(1) + WFPYPML(2) + WFPYPML(3) + WFPYPML(4);
        wfpsc_y_term = WFPSCYPML0 + WFPSCYPML(1) + WFPSCYPML(2) + WFPSCYPML(3) +
                       WFPSCYPML(4);
      }
    }

#define WFPX0                                            \
  fd_coeffs2x<T>(0) * (v[j] * v[j] * dt2<T>() * wfc[i] + \
                       2 * v[j] * dt2<T>() * scatter[j] * wfcsc[i])
#define WFPX(t)                                                  \
  fd_coeffs2x<T>(t) *                                            \
      (v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] +             \
       2 * v[j + t] * dt2<T>() * scatter[j + t] * wfcsc[i + t] + \
       v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] +             \
       2 * v[j - t] * dt2<T>() * scatter[j - t] * wfcsc[i - t])
#define WFPXPML0                                                      \
  fd_coeffs2x<T>(0) *                                                 \
      ((1 + bx[x]) * ((1 + bx[x]) * v[j] * v[j] * dt2<T>() * wfc[i] + \
                      bx[x] * zetax[i]) +                             \
       (1 + bx[x]) * (1 + bx[x]) * 2 * v[j] * dt2<T>() * scatter[j] * \
           wfcsc[i])
#define WFPXPML(t)                                                            \
  fd_coeffs2x<T>(t) *                                                         \
          (((1 + bx[x + t]) *                                                 \
            ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] +  \
             bx[x + t] * zetax[i + t])) +                                     \
           (1 + bx[x + t]) * (1 + bx[x + t]) * 2 * v[j + t] * dt2<T>() *      \
               scatter[j + t] * wfcsc[i + t] +                                \
           ((1 + bx[x - t]) *                                                 \
            ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] +  \
             bx[x - t] * zetax[i - t])) +                                     \
           (1 + bx[x - t]) * (1 + bx[x - t]) * 2 * v[j - t] * dt2<T>() *      \
               scatter[j - t] * wfcsc[i - t]) -                               \
      (fd_coeffs1x<T>(t - 1) *                                                \
       ((dbxdx[x + t] *                                                       \
             ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] + \
              bx[x + t] * zetax[i + t] +                                      \
              (1 + bx[x + t]) * 2 * v[j + t] * dt2<T>() * scatter[j + t] *    \
                  wfcsc[i + t]) +                                             \
         bx[x + t] * psix[i + t]) -                                           \
        (dbxdx[x - t] *                                                       \
             ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] + \
              bx[x - t] * zetax[i - t] +                                      \
              (1 + bx[x - t]) * 2 * v[j - t] * dt2<T>() * scatter[j - t] *    \
                  wfcsc[i - t]) +                                             \
         bx[x - t] * psix[i - t])))

#define WFPSCX0 fd_coeffs2x<T>(0) * v[j] * v[j] * dt2<T>() * wfcsc[i]
#define WFPSCX(t)                                                      \
  fd_coeffs2x<T>(t) * (v[j + t] * v[j + t] * dt2<T>() * wfcsc[i + t] + \
                       v[j - t] * v[j - t] * dt2<T>() * wfcsc[i - t])
#define WFPSCXPML0    \
  fd_coeffs2x<T>(0) * \
      ((1 + bx[x]) *  \
       ((1 + bx[x]) * v[j] * v[j] * dt2<T>() * wfcsc[i] + bx[x] * zetaxsc[i]))
#define WFPSCXPML(t)                                                           \
  fd_coeffs2x<T>(t) *                                                          \
          (((1 + bx[x + t]) *                                                  \
            ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfcsc[i + t] + \
             bx[x + t] * zetaxsc[i + t])) +                                    \
           ((1 + bx[x - t]) *                                                  \
            ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfcsc[i - t] + \
             bx[x - t] * zetaxsc[i - t]))) -                                   \
      (fd_coeffs1x<T>(t - 1) *                                                 \
       ((dbxdx[x + t] * ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() *    \
                             wfcsc[i + t] +                                    \
                         bx[x + t] * zetaxsc[i + t]) +                         \
         bx[x + t] * psixsc[i + t]) -                                          \
        (dbxdx[x - t] * ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() *    \
                             wfcsc[i - t] +                                    \
                         bx[x - t] * zetaxsc[i - t]) +                         \
         bx[x - t] * psixsc[i - t])))

    if (A == 2) {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1);
        wfpsc_x_term = WFPSCX0 + WFPSCX(1);
      } else {
        wfp_x_term = WFPXPML0 + WFPXPML(1);
        wfpsc_x_term = WFPSCXPML0 + WFPSCXPML(1);
      }
    } else if (A == 4) {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1) + WFPX(2);
        wfpsc_x_term = WFPSCX0 + WFPSCX(1) + WFPSCX(2);
      } else {
        wfp_x_term = WFPXPML0 + WFPXPML(1) + WFPXPML(2);
        wfpsc_x_term = WFPSCXPML0 + WFPSCXPML(1) + WFPSCXPML(2);
      }
    } else if (A == 6) {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1) + WFPX(2) + WFPX(3);
        wfpsc_x_term = WFPSCX0 + WFPSCX(1) + WFPSCX(2) + WFPSCX(3);
      } else {
        wfp_x_term = WFPXPML0 + WFPXPML(1) + WFPXPML(2) + WFPXPML(3);
        wfpsc_x_term = WFPSCXPML0 + WFPSCXPML(1) + WFPSCXPML(2) + WFPSCXPML(3);
      }
    } else {
      if (not pml_x) {
        wfp_x_term = WFPX0 + WFPX(1) + WFPX(2) + WFPX(3) + WFPX(4);
        wfpsc_x_term = WFPSCX0 + WFPSCX(1) + WFPSCX(2) + WFPSCX(3) + WFPSCX(4);
      } else {
        wfp_x_term =
            WFPXPML0 + WFPXPML(1) + WFPXPML(2) + WFPXPML(3) + WFPXPML(4);
        wfpsc_x_term = WFPSCXPML0 + WFPSCXPML(1) + WFPSCXPML(2) + WFPSCXPML(3) +
                       WFPSCXPML(4);
      }
    }

    wfp[i] = wfp_y_term + wfp_x_term + 2 * wfc[i] + wfp[i];
    wfcn[i] = -wfc[i];
    wfpsc[i] = wfpsc_y_term + wfpsc_x_term + 2 * wfcsc[i] + wfpsc[i];
    wfcnsc[i] = -wfcsc[i];
    if (pml_y) {
      T tmp;
      T tmpsc;
#define PSIY(t)                                                          \
  fd_coeffs1y<T>(t - 1) *                                                \
      ((ay[y + t] * ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                         dt2<T>() * wfc[i + t * nxc] +                   \
                     by[y + t] * zetay[i + t * nxc] +                    \
                     (1 + by[y + t]) * 2 * v[j + t * nxc] * dt2<T>() *   \
                         scatter[j + t * nxc] * wfcsc[i + t * nxc])) -   \
       (ay[y - t] * ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                         dt2<T>() * wfc[i - t * nxc] +                   \
                     by[y - t] * zetay[i - t * nxc] +                    \
                     (1 + by[y - t]) * 2 * v[j - t * nxc] * dt2<T>() *   \
                         scatter[j - t * nxc] * wfcsc[i - t * nxc])))
#define PSIYSC(t)                                                        \
  fd_coeffs1y<T>(t - 1) *                                                \
      ((ay[y + t] * ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                         dt2<T>() * wfcsc[i + t * nxc] +                 \
                     by[y + t] * zetaysc[i + t * nxc])) -                \
       (ay[y - t] * ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                         dt2<T>() * wfcsc[i - t * nxc] +                 \
                     by[y - t] * zetaysc[i - t * nxc])))

      if (A == 2) {
        tmp = -(PSIY(1));
        tmpsc = -(PSIYSC(1));
      } else if (A == 4) {
        tmp = -(PSIY(1) + PSIY(2));
        tmpsc = -(PSIYSC(1) + PSIYSC(2));
      } else if (A == 6) {
        tmp = -(PSIY(1) + PSIY(2) + PSIY(3));
        tmpsc = -(PSIYSC(1) + PSIYSC(2) + PSIYSC(3));
      } else {
        tmp = -(PSIY(1) + PSIY(2) + PSIY(3) + PSIY(4));
        tmpsc = -(PSIYSC(1) + PSIYSC(2) + PSIYSC(3) + PSIYSC(4));
      }
      psiyn[i] = tmp +
                 daydy[y] * ((1 + by[y]) * v[j] * v[j] * dt2<T>() * wfc[i] +
                             by[y] * zetay[i] +
                             (1 + by[y]) * 2 * v[j] * dt2<T>() * scatter[j] *
                                 wfcsc[i]) +
                 ay[y] * psiy[i];
      zetayn[i] = ay[y] * v[j] * v[j] * dt2<T>() * wfc[i] + ay[y] * zetay[i] +
                  ay[y] * 2 * v[j] * dt2<T>() * scatter[j] * wfcsc[i];
      psiynsc[i] = tmpsc +
                   daydy[y] * ((1 + by[y]) * v[j] * v[j] * dt2<T>() * wfcsc[i] +
                               by[y] * zetaysc[i]) +
                   ay[y] * psiysc[i];
      zetaynsc[i] =
          ay[y] * v[j] * v[j] * dt2<T>() * wfcsc[i] + ay[y] * zetaysc[i];
    }
    if (pml_x) {
      T tmp;
      T tmpsc;
#define PSIX(t)                                                          \
  fd_coeffs1x<T>(t - 1) *                                                \
      ((ax[x + t] *                                                      \
        ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] + \
         bx[x + t] * zetax[i + t] +                                      \
         (1 + bx[x + t]) * 2 * v[j + t] * dt2<T>() * scatter[j + t] *    \
             wfcsc[i + t])) -                                            \
       (ax[x - t] *                                                      \
        ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] + \
         bx[x - t] * zetax[i - t] +                                      \
         (1 + bx[x - t]) * 2 * v[j - t] * dt2<T>() * scatter[j - t] *    \
             wfcsc[i - t])))
#define PSIXSC(t)                                                          \
  fd_coeffs1x<T>(t - 1) *                                                  \
      ((ax[x + t] *                                                        \
        ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfcsc[i + t] + \
         bx[x + t] * zetaxsc[i + t])) -                                    \
       (ax[x - t] *                                                        \
        ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfcsc[i - t] + \
         bx[x - t] * zetaxsc[i - t])))

      if (A == 2) {
        tmp = -(PSIX(1));
        tmpsc = -(PSIXSC(1));
      } else if (A == 4) {
        tmp = -(PSIX(1) + PSIX(2));
        tmpsc = -(PSIXSC(1) + PSIXSC(2));
      } else if (A == 6) {
        tmp = -(PSIX(1) + PSIX(2) + PSIX(3));
        tmpsc = -(PSIXSC(1) + PSIXSC(2) + PSIXSC(3));
      } else {
        tmp = -(PSIX(1) + PSIX(2) + PSIX(3) + PSIX(4));
        tmpsc = -(PSIXSC(1) + PSIXSC(2) + PSIXSC(3) + PSIXSC(4));
      }
      psixn[i] = tmp +
                 daxdx[x] * ((1 + bx[x]) * v[j] * v[j] * dt2<T>() * wfc[i] +
                             bx[x] * zetax[i] +
                             (1 + bx[x]) * 2 * v[j] * dt2<T>() * scatter[j] *
                                 wfcsc[i]) +
                 ax[x] * psix[i];
      zetaxn[i] = ax[x] * v[j] * v[j] * dt2<T>() * wfc[i] + ax[x] * zetax[i] +
                  ax[x] * 2 * v[j] * dt2<T>() * scatter[j] * wfcsc[i];
      psixnsc[i] = tmpsc +
                   daxdx[x] * ((1 + bx[x]) * v[j] * v[j] * dt2<T>() * wfcsc[i] +
                               bx[x] * zetaxsc[i]) +
                   ax[x] * psixsc[i];
      zetaxnsc[i] =
          ax[x] * v[j] * v[j] * dt2<T>() * wfcsc[i] + ax[x] * zetaxsc[i];
    }
  }
}

template <typename T, int A>
__global__ void backward_kernel_sc(
    T const *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
    T const *__restrict psiy, T const *__restrict psix, T *__restrict psiyn,
    T *__restrict psixn, T const *__restrict zetay, T const *__restrict zetax,
    T *__restrict zetayn, T *__restrict zetaxn, T const *__restrict v,
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
  if (y < pml_regionsyc[3] && x < pml_regionsxc[3]) {
    bool pml_y{y < pml_regionsyc[1] || y >= pml_regionsyc[2]};
    bool pml_x{x < pml_regionsxc[1] || x >= pml_regionsxc[2]};
    T wfp_y_term;
    T wfp_x_term;
#define WFPY0_SC fd_coeffs2y<T>(0) * v[j] * v[j] * dt2<T>() * wfc[i]
#define WFPY_SC(t)                                                     \
  fd_coeffs2y<T>(t) *                                                  \
      (v[j + t * nxc] * v[j + t * nxc] * dt2<T>() * wfc[i + t * nxc] + \
       v[j - t * nxc] * v[j - t * nxc] * dt2<T>() * wfc[i - t * nxc])
#define WFPYPML0_SC   \
  fd_coeffs2y<T>(0) * \
      ((1 + by[y]) *  \
       ((1 + by[y]) * v[j] * v[j] * dt2<T>() * wfc[i] + by[y] * zetay[i]))
#define WFPYPML_SC(t)                                                        \
  fd_coeffs2y<T>(t) * (((1 + by[y + t]) *                                    \
                        ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                             dt2<T>() * wfc[i + t * nxc] +                   \
                         by[y + t] * zetay[i + t * nxc])) +                  \
                       ((1 + by[y - t]) *                                    \
                        ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                             dt2<T>() * wfc[i - t * nxc] +                   \
                         by[y - t] * zetay[i - t * nxc]))) -                 \
      (fd_coeffs1y<T>(t - 1) *                                               \
       ((dbydy[y + t] * ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                             dt2<T>() * wfc[i + t * nxc] +                   \
                         by[y + t] * zetay[i + t * nxc]) +                   \
         by[y + t] * psiy[i + t * nxc]) -                                    \
        (dbydy[y - t] * ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                             dt2<T>() * wfc[i - t * nxc] +                   \
                         by[y - t] * zetay[i - t * nxc]) +                   \
         by[y - t] * psiy[i - t * nxc])))
    if (A == 2) {
      if (not pml_y) {
        wfp_y_term = WFPY0_SC + WFPY_SC(1);
      } else {
        wfp_y_term = WFPYPML0_SC + WFPYPML_SC(1);
      }
    } else if (A == 4) {
      if (not pml_y) {
        wfp_y_term = WFPY0_SC + WFPY_SC(1) + WFPY_SC(2);
      } else {
        wfp_y_term = WFPYPML0_SC + WFPYPML_SC(1) + WFPYPML_SC(2);
      }
    } else if (A == 6) {
      if (not pml_y) {
        wfp_y_term = WFPY0_SC + WFPY_SC(1) + WFPY_SC(2) + WFPY_SC(3);
      } else {
        wfp_y_term =
            WFPYPML0_SC + WFPYPML_SC(1) + WFPYPML_SC(2) + WFPYPML_SC(3);
      }
    } else {
      if (not pml_y) {
        wfp_y_term =
            WFPY0_SC + WFPY_SC(1) + WFPY_SC(2) + WFPY_SC(3) + WFPY_SC(4);
      } else {
        wfp_y_term = WFPYPML0_SC + WFPYPML_SC(1) + WFPYPML_SC(2) +
                     WFPYPML_SC(3) + WFPYPML_SC(4);
      }
    }
#define WFPX0_SC fd_coeffs2x<T>(0) * v[j] * v[j] * dt2<T>() * wfc[i]
#define WFPX_SC(t)                                                   \
  fd_coeffs2x<T>(t) * (v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] + \
                       v[j - t] * v[j - t] * dt2<T>() * wfc[i - t])
#define WFPXPML0_SC   \
  fd_coeffs2x<T>(0) * \
      ((1 + bx[x]) *  \
       ((1 + bx[x]) * v[j] * v[j] * dt2<T>() * wfc[i] + bx[x] * zetax[i]))
#define WFPXPML_SC(t)                                                         \
  fd_coeffs2x<T>(t) *                                                         \
          (((1 + bx[x + t]) *                                                 \
            ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] +  \
             bx[x + t] * zetax[i + t])) +                                     \
           ((1 + bx[x - t]) *                                                 \
            ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] +  \
             bx[x - t] * zetax[i - t]))) -                                    \
      (fd_coeffs1x<T>(t - 1) *                                                \
       ((dbxdx[x + t] *                                                       \
             ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] + \
              bx[x + t] * zetax[i + t]) +                                     \
         bx[x + t] * psix[i + t]) -                                           \
        (dbxdx[x - t] *                                                       \
             ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] + \
              bx[x - t] * zetax[i - t]) +                                     \
         bx[x - t] * psix[i - t])))
    if (A == 2) {
      if (not pml_x) {
        wfp_x_term = WFPX0_SC + WFPX_SC(1);
      } else {
        wfp_x_term = WFPXPML0_SC + WFPXPML_SC(1);
      }
    } else if (A == 4) {
      if (not pml_x) {
        wfp_x_term = WFPX0_SC + WFPX_SC(1) + WFPX_SC(2);
      } else {
        wfp_x_term = WFPXPML0_SC + WFPXPML_SC(1) + WFPXPML_SC(2);
      }
    } else if (A == 6) {
      if (not pml_x) {
        wfp_x_term = WFPX0_SC + WFPX_SC(1) + WFPX_SC(2) + WFPX_SC(3);
      } else {
        wfp_x_term =
            WFPXPML0_SC + WFPXPML_SC(1) + WFPXPML_SC(2) + WFPXPML_SC(3);
      }
    } else {
      if (not pml_x) {
        wfp_x_term =
            WFPX0_SC + WFPX_SC(1) + WFPX_SC(2) + WFPX_SC(3) + WFPX_SC(4);
      } else {
        wfp_x_term = WFPXPML0_SC + WFPXPML_SC(1) + WFPXPML_SC(2) +
                     WFPXPML_SC(3) + WFPXPML_SC(4);
      }
    }

    wfp[i] = wfp_y_term + wfp_x_term + 2 * wfc[i] + wfp[i];
    wfcn[i] = -wfc[i];
    if (pml_y) {
      T tmp;
#define PSIY_SC(t)                                                       \
  fd_coeffs1y<T>(t - 1) *                                                \
      ((ay[y + t] * ((1 + by[y + t]) * v[j + t * nxc] * v[j + t * nxc] * \
                         dt2<T>() * wfc[i + t * nxc] +                   \
                     by[y + t] * zetay[i + t * nxc])) -                  \
       (ay[y - t] * ((1 + by[y - t]) * v[j - t * nxc] * v[j - t * nxc] * \
                         dt2<T>() * wfc[i - t * nxc] +                   \
                     by[y - t] * zetay[i - t * nxc])))
      if (A == 2) {
        tmp = -(PSIY_SC(1));
      } else if (A == 4) {
        tmp = -(PSIY_SC(1) + PSIY_SC(2));
      } else if (A == 6) {
        tmp = -(PSIY_SC(1) + PSIY_SC(2) + PSIY_SC(3));
      } else {
        tmp = -(PSIY_SC(1) + PSIY_SC(2) + PSIY_SC(3) + PSIY_SC(4));
      }
      psiyn[i] = tmp +
                 daydy[y] * ((1 + by[y]) * v[j] * v[j] * dt2<T>() * wfc[i] +
                             by[y] * zetay[i]) +
                 ay[y] * psiy[i];
      zetayn[i] = ay[y] * v[j] * v[j] * dt2<T>() * wfc[i] + ay[y] * zetay[i];
    }
    if (pml_x) {
      T tmp;
#define PSIX_SC(t)                                                       \
  fd_coeffs1x<T>(t - 1) *                                                \
      ((ax[x + t] *                                                      \
        ((1 + bx[x + t]) * v[j + t] * v[j + t] * dt2<T>() * wfc[i + t] + \
         bx[x + t] * zetax[i + t])) -                                    \
       (ax[x - t] *                                                      \
        ((1 + bx[x - t]) * v[j - t] * v[j - t] * dt2<T>() * wfc[i - t] + \
         bx[x - t] * zetax[i - t])))
      if (A == 2) {
        tmp = -(PSIX_SC(1));
      } else if (A == 4) {
        tmp = -(PSIX_SC(1) + PSIX_SC(2));
      } else if (A == 6) {
        tmp = -(PSIX_SC(1) + PSIX_SC(2) + PSIX_SC(3));
      } else {
        tmp = -(PSIX_SC(1) + PSIX_SC(2) + PSIX_SC(3) + PSIX_SC(4));
      }
      psixn[i] = tmp +
                 daxdx[x] * ((1 + bx[x]) * v[j] * v[j] * dt2<T>() * wfc[i] +
                             bx[x] * zetax[i]) +
                 ax[x] * psix[i];
      zetaxn[i] = ax[x] * v[j] * v[j] * dt2<T>() * wfc[i] + ax[x] * zetax[i];
    }
  }
}

template <typename T, int A>
void forward_batch(
    T *__restrict wfc, T *__restrict wfp, T *__restrict psiy,
    T *__restrict psix, T *__restrict psiyn, T *__restrict psixn,
    T *__restrict zetay, T *__restrict zetax, T *__restrict wfcsc,
    T *__restrict wfpsc, T *__restrict psiysc, T *__restrict psixsc,
    T *__restrict psiynsc, T *__restrict psixnsc, T *__restrict zetaysc,
    T *__restrict zetaxsc, int64_t const *__restrict sources_i,
    int64_t const *__restrict receivers_i,
    int64_t const *__restrict bg_receivers_i, T *__restrict w_store,
    T *__restrict wsc_store, T const *__restrict v, T const *scatter,
    T const *__restrict f, T *__restrict r, T *__restrict bg_r,
    T const *__restrict ay, T const *__restrict ax, T const *__restrict by,
    T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx, int64_t n_sources_per_shot,
    int64_t n_receivers_per_shot, int64_t n_bg_receivers_per_shot, int64_t ny,
    int64_t nx, int64_t nt, int64_t step_ratio, bool v_requires_grad,
    bool scatter_requires_grad, int64_t n_batch) {
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
  auto gridx_bg_receivers{ceil_div(n_bg_receivers_per_shot,
                                   static_cast<int64_t>(dimBlock_receivers.x))};
  dim3 dimGrid_bg_receivers(gridx_bg_receivers, gridy_receivers,
                            gridz_receivers);

  for (int64_t t{}; t < nt; ++t) {
    if (t % step_ratio == 0 && (v_requires_grad || scatter_requires_grad)) {
      if (v_requires_grad && scatter_requires_grad) {
        forward_kernel<T, A, true, true><<<dimGrid, dimBlock>>>(
            wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, wfcsc, wfpsc,
            psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc,
            w_store + (t / step_ratio) * n_batch * ny * nx,
            wsc_store + (t / step_ratio) * n_batch * ny * nx, v, scatter, ay,
            ax, by, bx, daydy, daxdx, dbydy, dbxdx);
      } else if (v_requires_grad) {
        forward_kernel<T, A, true, false><<<dimGrid, dimBlock>>>(
            wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, wfcsc, wfpsc,
            psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc,
            w_store + (t / step_ratio) * n_batch * ny * nx,
            wsc_store + (t / step_ratio) * n_batch * ny * nx, v, scatter, ay,
            ax, by, bx, daydy, daxdx, dbydy, dbxdx);
      } else {
        forward_kernel<T, A, false, true><<<dimGrid, dimBlock>>>(
            wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, wfcsc, wfpsc,
            psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc,
            w_store + (t / step_ratio) * n_batch * ny * nx, nullptr, v, scatter,
            ay, ax, by, bx, daydy, daxdx, dbydy, dbxdx);
      }
    } else {
      forward_kernel<T, A, false, false><<<dimGrid, dimBlock>>>(
          wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, wfcsc, wfpsc,
          psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc, nullptr, nullptr,
          v, scatter, ay, ax, by, bx, daydy, daxdx, dbydy, dbxdx);
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
          r + t * n_batch * n_receivers_per_shot, wfcsc, receivers_i,
          n_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_bg_receivers_per_shot > 0) {
      record_receivers<<<dimGrid_bg_receivers, dimBlock_receivers>>>(
          bg_r + t * n_batch * n_bg_receivers_per_shot, wfc, bg_receivers_i,
          n_bg_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    std::swap(wfp, wfc);
    std::swap(psiyn, psiy);
    std::swap(psixn, psix);
    std::swap(wfpsc, wfcsc);
    std::swap(psiynsc, psiysc);
    std::swap(psixnsc, psixsc);
  }
}

template <typename T, int A>
__global__ void combine_grad_model(T *__restrict grad_v,
                                   T const *__restrict grad_v_batch,
                                   int64_t n_batch) {
  constexpr int fd_pad{A / 2};
  auto x{blockIdx.x * blockDim.x + threadIdx.x + fd_pad};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + fd_pad};
  auto i{y * nxc + x};
  if (y < pml_regionsyc[3] && x < pml_regionsxc[3]) {
    for (int64_t batch{}; batch < n_batch; ++batch) {
      grad_v[i] += grad_v_batch[batch * nynxc + i];
    }
  }
}

template <typename T, int A>
void backward_batch(
    T *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
    T *__restrict psiy, T *__restrict psix, T *__restrict psiyn,
    T *__restrict psixn, T *__restrict zetay, T *__restrict zetax,
    T *__restrict zetayn, T *__restrict zetaxn, T *__restrict wfcsc,
    T *__restrict wfpsc, T *__restrict wfcnsc, T *__restrict psiysc,
    T *__restrict psixsc, T *__restrict psiynsc, T *__restrict psixnsc,
    T *__restrict zetaysc, T *__restrict zetaxsc, T *__restrict zetaynsc,
    T *__restrict zetaxnsc, int64_t const *__restrict sources_i,
    int64_t const *__restrict receivers_i,
    int64_t const *__restrict bg_receivers_i, T const *__restrict w_store,
    T const *__restrict wsc_store, T const *__restrict v,
    T const *__restrict scatter, T *__restrict f, T const *__restrict r,
    T const *__restrict bg_r, T *__restrict grad_v, T *__restrict grad_scatter,
    T const *__restrict ay, T const *__restrict ax, T const *__restrict by,
    T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx, int64_t n_sources_per_shot,
    int64_t n_receivers_per_shot, int64_t n_bg_receivers_per_shot, int64_t ny,
    int64_t nx, int64_t nt, int64_t step_ratio, bool v_requires_grad,
    bool scatter_requires_grad, int64_t n_batch) {
  constexpr int fd_pad{A / 2};
  dim3 dimBlock(32, 8, 1);
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
  auto gridx_bg_receivers{ceil_div(n_bg_receivers_per_shot,
                                   static_cast<int64_t>(dimBlock_receivers.x))};
  dim3 dimGrid_bg_receivers(gridx_bg_receivers, gridy_receivers,
                            gridz_receivers);
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (n_receivers_per_shot > 0) {
      add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
          wfpsc, r + t * n_batch * n_receivers_per_shot, receivers_i,
          n_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_bg_receivers_per_shot > 0) {
      add_sources<<<dimGrid_bg_receivers, dimBlock_receivers>>>(
          wfp, bg_r + t * n_batch * n_bg_receivers_per_shot, bg_receivers_i,
          n_bg_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_sources_per_shot > 0) {
      record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
          f + t * n_batch * n_sources_per_shot, wfc, sources_i,
          n_sources_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 && v_requires_grad) {
      add_to_grad_v<T, A><<<dimGrid, dimBlock>>>(
          grad_v, wfc, wfcsc, w_store + (t / step_ratio) * n_batch * ny * nx,
          wsc_store + (t / step_ratio) * n_batch * ny * nx, v, scatter,
          step_ratio);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 && scatter_requires_grad) {
      add_to_grad_scatter<T, A><<<dimGrid, dimBlock>>>(
          grad_scatter, wfcsc, w_store + (t / step_ratio) * n_batch * nx * ny,
          v, step_ratio);
    }
    backward_kernel<T, A><<<dimGrid, dimBlock>>>(
        wfc, wfp, wfcn, psiy, psix, psiyn, psixn, zetay, zetax, zetayn, zetaxn,
        wfcsc, wfpsc, wfcnsc, psiysc, psixsc, psiynsc, psixnsc, zetaysc,
        zetaxsc, zetaynsc, zetaxnsc, v, scatter, ay, ax, by, bx, daydy, daxdx,
        dbydy, dbxdx);
    gpuErrchk(cudaPeekAtLastError());
    T *tmp{wfc};
    wfc = wfp;
    wfp = wfcn;
    wfcn = tmp;
    std::swap(psiyn, psiy);
    std::swap(psixn, psix);
    std::swap(zetayn, zetay);
    std::swap(zetaxn, zetax);
    T *tmpsc{wfcsc};
    wfcsc = wfpsc;
    wfpsc = wfcnsc;
    wfcnsc = tmpsc;
    std::swap(psiynsc, psiysc);
    std::swap(psixnsc, psixsc);
    std::swap(zetaynsc, zetaysc);
    std::swap(zetaxnsc, zetaxsc);
  }
}

template <typename T, int A>
void backward_batch_sc(
    T *__restrict wfcsc, T *__restrict wfpsc, T *__restrict wfcnsc,
    T *__restrict psiysc, T *__restrict psixsc, T *__restrict psiynsc,
    T *__restrict psixnsc, T *__restrict zetaysc, T *__restrict zetaxsc,
    T *__restrict zetaynsc, T *__restrict zetaxnsc,
    int64_t const *__restrict receivers_i, T const *__restrict w_store,
    T const *__restrict v, T const *__restrict r, T *__restrict grad_scatter,
    T const *__restrict ay, T const *__restrict ax, T const *__restrict by,
    T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx, int64_t n_receivers_per_shot, int64_t ny,
    int64_t nx, int64_t nt, int64_t step_ratio, bool scatter_requires_grad,
    int64_t n_batch) {
  constexpr int fd_pad{A / 2};
  dim3 dimBlock(32, 8, 1);
  auto gridx{ceil_div(nx - 2 * fd_pad, static_cast<int64_t>(dimBlock.x))};
  auto gridy{ceil_div(ny - 2 * fd_pad, static_cast<int64_t>(dimBlock.y))};
  auto gridz{ceil_div(n_batch, static_cast<int64_t>(dimBlock.z))};
  dim3 dimGrid(gridx, gridy, gridz);
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
          wfpsc, r + t * n_batch * n_receivers_per_shot, receivers_i,
          n_receivers_per_shot, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 && scatter_requires_grad) {
      add_to_grad_scatter<T, A><<<dimGrid, dimBlock>>>(
          grad_scatter, wfcsc, w_store + (t / step_ratio) * n_batch * ny * nx,
          v, step_ratio);
      gpuErrchk(cudaPeekAtLastError());
    }
    backward_kernel_sc<T, A>
        <<<dimGrid, dimBlock>>>(wfcsc, wfpsc, wfcnsc, psiysc, psixsc, psiynsc,
                                psixnsc, zetaysc, zetaxsc, zetaynsc, zetaxnsc,
                                v, ay, ax, by, bx, daydy, daxdx, dbydy, dbxdx);
    gpuErrchk(cudaPeekAtLastError());
    T *tmpsc{wfcsc};
    wfcsc = wfpsc;
    wfpsc = wfcnsc;
    wfcnsc = tmpsc;
    std::swap(psiynsc, psiysc);
    std::swap(psixnsc, psixsc);
    std::swap(zetaynsc, zetaysc);
    std::swap(zetaxnsc, zetaxsc);
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
void set_fd_coeffs(T fd_coeffs1[], T fd_coeffs2[], int64_t accuracy, T dx) {
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

class ScalarBornCUDAFunction
    : public torch::autograd::Function<ScalarBornCUDAFunction> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext *ctx, torch::Tensor const &v,
      torch::Tensor const &scatter, torch::Tensor const &f,
      torch::Tensor const &wfc0, torch::Tensor const &wfp0,
      torch::Tensor const &psiy0, torch::Tensor const &psix0,
      torch::Tensor const &zetay0, torch::Tensor const &zetax0,
      torch::Tensor const &wfcsc0, torch::Tensor const &wfpsc0,
      torch::Tensor const &psiysc0, torch::Tensor const &psixsc0,
      torch::Tensor const &zetaysc0, torch::Tensor const &zetaxsc0,
      torch::Tensor const &ay, torch::Tensor const &ax, torch::Tensor const &by,
      torch::Tensor const &bx, torch::Tensor const &sources_i,
      torch::Tensor const &receivers_i, torch::Tensor const &bg_receivers_i,
      double dy, double dx, double dt, int64_t nt, int64_t n_batch,
      int64_t step_ratio, int64_t accuracy, int64_t pml_width0,
      int64_t pml_width1, int64_t pml_width2, int64_t pml_width3) {
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    at::cuda::CUDAGuard device_guard(v.device());
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
    int64_t n_bg_receivers_per_shot{};
    if (bg_receivers_i.numel() > 0) {
      n_bg_receivers_per_shot = bg_receivers_i.size(1);
    }
    auto wfc{create_or_pad(wfc0, fd_pad, options, size_with_batch)};
    auto wfp{create_or_pad(wfp0, fd_pad, options, size_with_batch)};
    auto psiy{create_or_pad(psiy0, fd_pad, options, size_with_batch)};
    auto psix{create_or_pad(psix0, fd_pad, options, size_with_batch)};
    auto zetay{create_or_pad(zetay0, fd_pad, options, size_with_batch)};
    auto zetax{create_or_pad(zetax0, fd_pad, options, size_with_batch)};
    auto psiyn{at::zeros_like(psiy)};
    auto psixn{at::zeros_like(psix)};
    auto wfcsc{create_or_pad(wfcsc0, fd_pad, options, size_with_batch)};
    auto wfpsc{create_or_pad(wfpsc0, fd_pad, options, size_with_batch)};
    auto psiysc{create_or_pad(psiysc0, fd_pad, options, size_with_batch)};
    auto psixsc{create_or_pad(psixsc0, fd_pad, options, size_with_batch)};
    auto zetaysc{create_or_pad(zetaysc0, fd_pad, options, size_with_batch)};
    auto zetaxsc{create_or_pad(zetaxsc0, fd_pad, options, size_with_batch)};
    auto psiynsc{at::zeros_like(psiysc)};
    auto psixnsc{at::zeros_like(psixsc)};
    auto r{at::empty({nt, n_batch, n_receivers_per_shot}, options)};
    auto bg_r{at::empty({nt, n_batch, n_bg_receivers_per_shot}, options)};
    auto daydy{at::zeros_like(ay)};
    auto daxdx{at::zeros_like(ax)};
    auto dbydy{at::zeros_like(by)};
    auto dbxdx{at::zeros_like(bx)};
    torch::Tensor w_store;
    torch::Tensor wsc_store;
    if (v.requires_grad() || scatter.requires_grad()) {
      w_store = at::empty({(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx},
                          options);
    }
    if (v.requires_grad()) {
      wsc_store = at::empty(
          {(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx}, options);
    }

    bool non_sc{v.requires_grad() || f.requires_grad() ||
                wfc0.requires_grad() || wfp0.requires_grad() ||
                psiy0.requires_grad() || psix0.requires_grad() ||
                zetay0.requires_grad() || zetax0.requires_grad()};

    zero_interior<true>(psiy, ny, nx, fd_pad, pml_width);
    zero_interior<false>(psix, ny, nx, fd_pad, pml_width);
    zero_interior<true>(zetay, ny, nx, fd_pad, pml_width);
    zero_interior<false>(zetax, ny, nx, fd_pad, pml_width);
    zero_interior<true>(psiysc, ny, nx, fd_pad, pml_width);
    zero_interior<false>(psixsc, ny, nx, fd_pad, pml_width);
    zero_interior<true>(zetaysc, ny, nx, fd_pad, pml_width);
    zero_interior<false>(zetaxsc, ny, nx, fd_pad, pml_width);

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_born_cuda_forward", ([&] {
          scalar_t dt2_a = dt * dt;
          cudaMemcpyToSymbol(dt2c, &dt2_a, sizeof(scalar_t));
          scalar_t const *__restrict v_a{v.data_ptr<scalar_t>()};
          scalar_t const *__restrict scatter_a{scatter.data_ptr<scalar_t>()};
          scalar_t const *__restrict f_a{f.data_ptr<scalar_t>()};
          scalar_t *__restrict r_a{r.data_ptr<scalar_t>()};
          scalar_t *__restrict bg_r_a{bg_r.data_ptr<scalar_t>()};
          scalar_t *__restrict wfc_a{wfc.data_ptr<scalar_t>()};
          scalar_t *__restrict wfp_a{wfp.data_ptr<scalar_t>()};
          scalar_t *__restrict psiy_a{psiy.data_ptr<scalar_t>()};
          scalar_t *__restrict psix_a{psix.data_ptr<scalar_t>()};
          scalar_t *__restrict psiyn_a{psiyn.data_ptr<scalar_t>()};
          scalar_t *__restrict psixn_a{psixn.data_ptr<scalar_t>()};
          scalar_t *__restrict zetay_a{zetay.data_ptr<scalar_t>()};
          scalar_t *__restrict zetax_a{zetax.data_ptr<scalar_t>()};
          scalar_t *__restrict wfcsc_a{wfcsc.data_ptr<scalar_t>()};
          scalar_t *__restrict wfpsc_a{wfpsc.data_ptr<scalar_t>()};
          scalar_t *__restrict psiysc_a{psiysc.data_ptr<scalar_t>()};
          scalar_t *__restrict psixsc_a{psixsc.data_ptr<scalar_t>()};
          scalar_t *__restrict psiynsc_a{psiynsc.data_ptr<scalar_t>()};
          scalar_t *__restrict psixnsc_a{psixnsc.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaysc_a{zetaysc.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaxsc_a{zetaxsc.data_ptr<scalar_t>()};
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
          int64_t const *__restrict bg_receivers_i_a{
              bg_receivers_i.data_ptr<int64_t>()};
          scalar_t *__restrict w_store_a{};
          scalar_t *__restrict wsc_store_a{};
          if (v.requires_grad() || scatter.requires_grad()) {
            w_store_a = w_store.data_ptr<scalar_t>();
          }
          if (v.requires_grad()) {
            wsc_store_a = wsc_store.data_ptr<scalar_t>();
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
              wfcsc_a, wfpsc_a, psiysc_a, psixsc_a, psiynsc_a, psixnsc_a,
              zetaysc_a, zetaxsc_a, sources_i_a, receivers_i_a,
              bg_receivers_i_a, w_store_a, wsc_store_a, v_a, scatter_a, f_a,
              r_a, bg_r_a, ay_a, ax_a, by_a, bx_a, daydy_a, daxdx_a, dbydy_a,
              dbxdx_a, n_sources_per_shot, n_receivers_per_shot,
              n_bg_receivers_per_shot, ny, nx, nt, step_ratio,
              v.requires_grad(), scatter.requires_grad(), n_batch);
        }));
    if (v.requires_grad() || scatter.requires_grad() || f.requires_grad() ||
        wfc0.requires_grad() || wfp0.requires_grad() || psiy0.requires_grad() ||
        psix0.requires_grad() || zetay0.requires_grad() ||
        zetax0.requires_grad() || wfcsc0.requires_grad() ||
        wfpsc0.requires_grad() || psiysc0.requires_grad() ||
        psixsc0.requires_grad() || zetaysc0.requires_grad() ||
        zetaxsc0.requires_grad()) {
      ctx->save_for_backward(
          {v, scatter, ay, ax, by, bx, sources_i, receivers_i, bg_receivers_i});
      ctx->saved_data["w_store"] = w_store;
      ctx->saved_data["wsc_store"] = wsc_store;
      ctx->saved_data["dy"] = dy;
      ctx->saved_data["dx"] = dx;
      ctx->saved_data["dt"] = dt;
      ctx->saved_data["nt"] = nt;
      ctx->saved_data["n_batch"] = n_batch;
      ctx->saved_data["step_ratio"] = step_ratio;
      ctx->saved_data["accuracy"] = accuracy;
      ctx->saved_data["non_sc"] = non_sc;
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
              wfpsc.index({all_slice, slicey, slicex}),
              wfcsc.index({all_slice, slicey, slicex}),
              psiynsc.index({all_slice, slicey, slicex}),
              psixnsc.index({all_slice, slicey, slicex}),
              zetaysc.index({all_slice, slicey, slicex}),
              zetaxsc.index({all_slice, slicey, slicex}),
              bg_r,
              r};
    }
    return {wfc.index({all_slice, slicey, slicex}),
            wfp.index({all_slice, slicey, slicex}),
            psiy.index({all_slice, slicey, slicex}),
            psix.index({all_slice, slicey, slicex}),
            zetay.index({all_slice, slicey, slicex}),
            zetax.index({all_slice, slicey, slicex}),
            wfcsc.index({all_slice, slicey, slicex}),
            wfpsc.index({all_slice, slicey, slicex}),
            psiysc.index({all_slice, slicey, slicex}),
            psixsc.index({all_slice, slicey, slicex}),
            zetaysc.index({all_slice, slicey, slicex}),
            zetaxsc.index({all_slice, slicey, slicex}),
            bg_r,
            r};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list const &grad_outputs) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto saved{ctx->get_saved_variables()};
    auto const &v{saved[0]};
    auto const &scatter{saved[1]};
    auto const &ay{saved[2]};
    auto const &ax{saved[3]};
    auto const &by{saved[4]};
    auto const &bx{saved[5]};
    auto const &sources_i{saved[6]};
    auto const &receivers_i{saved[7]};
    auto const &bg_receivers_i{saved[8]};
    auto const &w_store{ctx->saved_data["w_store"].toTensor()};
    auto const &wsc_store{ctx->saved_data["wsc_store"].toTensor()};
    auto dy{ctx->saved_data["dy"].toDouble()};
    auto dx{ctx->saved_data["dx"].toDouble()};
    auto dt{ctx->saved_data["dt"].toDouble()};
    auto nt{ctx->saved_data["nt"].toInt()};
    auto n_batch{ctx->saved_data["n_batch"].toInt()};
    auto step_ratio{ctx->saved_data["step_ratio"].toInt()};
    auto accuracy{ctx->saved_data["accuracy"].toInt()};
    auto non_sc{ctx->saved_data["non_sc"].toBool()};
    int64_t const pml_width[4] = {ctx->saved_data["pml_width0"].toInt(),
                                  ctx->saved_data["pml_width1"].toInt(),
                                  ctx->saved_data["pml_width2"].toInt(),
                                  ctx->saved_data["pml_width3"].toInt()};
    auto fd_pad{accuracy / 2};

    auto options{at::device(v.device()).dtype(v.scalar_type())};
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

    auto wfc = non_sc ? at::constant_pad_nd(grad_outputs[0],
                                            {fd_pad, fd_pad, fd_pad, fd_pad})
                      : at::empty(0, options);
    auto wfp = non_sc ? at::constant_pad_nd(grad_outputs[1],
                                            {fd_pad, fd_pad, fd_pad, fd_pad})
                      : at::empty(0, options);
    auto psiy = non_sc ? at::constant_pad_nd(grad_outputs[2],
                                             {fd_pad, fd_pad, fd_pad, fd_pad})
                       : at::empty(0, options);
    auto psix = non_sc ? at::constant_pad_nd(grad_outputs[3],
                                             {fd_pad, fd_pad, fd_pad, fd_pad})
                       : at::empty(0, options);
    auto zetay = non_sc ? at::constant_pad_nd(grad_outputs[4],
                                              {fd_pad, fd_pad, fd_pad, fd_pad})
                        : at::empty(0, options);
    auto zetax = non_sc ? at::constant_pad_nd(grad_outputs[5],
                                              {fd_pad, fd_pad, fd_pad, fd_pad})
                        : at::empty(0, options);
    auto wfcn{at::zeros_like(wfc)};
    auto psiyn{at::zeros_like(psiy)};
    auto psixn{at::zeros_like(psix)};
    auto zetayn{at::zeros_like(zetay)};
    auto zetaxn{at::zeros_like(zetax)};
    auto wfcsc =
        at::constant_pad_nd(grad_outputs[6], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto wfpsc =
        at::constant_pad_nd(grad_outputs[7], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psiysc =
        at::constant_pad_nd(grad_outputs[8], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psixsc =
        at::constant_pad_nd(grad_outputs[9], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetaysc =
        at::constant_pad_nd(grad_outputs[10], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetaxsc =
        at::constant_pad_nd(grad_outputs[11], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto wfcnsc{at::zeros_like(wfcsc)};
    auto psiynsc{at::zeros_like(psiysc)};
    auto psixnsc{at::zeros_like(psixsc)};
    auto zetaynsc{at::zeros_like(zetaysc)};
    auto zetaxnsc{at::zeros_like(zetaxsc)};
    auto grad_bg_r{grad_outputs[12].contiguous()};
    auto grad_r{grad_outputs[13].contiguous()};
    auto daydy{at::zeros_like(ay)};
    auto daxdx{at::zeros_like(ax)};
    auto dbydy{at::zeros_like(by)};
    auto dbxdx{at::zeros_like(bx)};
    int64_t n_sources_per_shot{};
    if (sources_i.numel() > 0) {
      n_sources_per_shot = sources_i.size(1);
    }
    int64_t n_receivers_per_shot{};
    if (receivers_i.numel() > 0) {
      n_receivers_per_shot = receivers_i.size(1);
    }
    int64_t n_bg_receivers_per_shot{};
    if (bg_receivers_i.numel() > 0) {
      n_bg_receivers_per_shot = bg_receivers_i.size(1);
    }

    auto grad_v{non_sc ? at::zeros_like(v) : at::empty(0, options)};
    auto grad_v_batch{non_sc && n_batch > 1
                          ? at::zeros({n_batch, ny, nx}, options)
                          : at::empty(0, options)};
    auto grad_scatter{at::zeros_like(scatter)};
    auto grad_scatter_batch{n_batch > 1 ? at::zeros({n_batch, ny, nx}, options)
                                        : at::empty(0)};
    auto grad_f{non_sc ? at::empty({nt, n_batch, n_sources_per_shot}, options)
                       : at::empty(0, options)};

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_born_cuda_backward", ([&] {
          scalar_t dt2_a = dt * dt;
          cudaMemcpyToSymbol(dt2c, &dt2_a, sizeof(scalar_t));
          scalar_t const *__restrict v_a{v.data_ptr<scalar_t>()};
          scalar_t const *__restrict scatter_a{scatter.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_v_a{grad_v.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_v_batch_a{
              n_batch > 1 ? grad_v_batch.data_ptr<scalar_t>() : grad_v_a};
          scalar_t *__restrict grad_scatter_a{
              grad_scatter.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_scatter_batch_a{
              n_batch > 1 ? grad_scatter_batch.data_ptr<scalar_t>()
                          : grad_scatter_a};
          scalar_t *__restrict grad_r_a{grad_r.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_bg_r_a{grad_bg_r.data_ptr<scalar_t>()};
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
          scalar_t *__restrict wfcsc_a{wfcsc.data_ptr<scalar_t>()};
          scalar_t *__restrict wfpsc_a{wfpsc.data_ptr<scalar_t>()};
          scalar_t *__restrict wfcnsc_a{wfcnsc.data_ptr<scalar_t>()};
          scalar_t *__restrict psiysc_a{psiysc.data_ptr<scalar_t>()};
          scalar_t *__restrict psixsc_a{psixsc.data_ptr<scalar_t>()};
          scalar_t *__restrict psiynsc_a{psiynsc.data_ptr<scalar_t>()};
          scalar_t *__restrict psixnsc_a{psixnsc.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaysc_a{zetaysc.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaxsc_a{zetaxsc.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaynsc_a{zetaynsc.data_ptr<scalar_t>()};
          scalar_t *__restrict zetaxnsc_a{zetaxnsc.data_ptr<scalar_t>()};
          scalar_t const *__restrict ay_a{ay.data_ptr<scalar_t>()};
          scalar_t const *__restrict ax_a{ax.data_ptr<scalar_t>()};
          scalar_t const *__restrict by_a{by.data_ptr<scalar_t>()};
          scalar_t const *__restrict bx_a{bx.data_ptr<scalar_t>()};
          scalar_t *__restrict daydy_a{daydy.data_ptr<scalar_t>()};
          scalar_t *__restrict daxdx_a{daxdx.data_ptr<scalar_t>()};
          scalar_t *__restrict dbydy_a{dbydy.data_ptr<scalar_t>()};
          scalar_t *__restrict dbxdx_a{dbxdx.data_ptr<scalar_t>()};
          int64_t *__restrict sources_i_a{sources_i.data_ptr<int64_t>()};
          int64_t *__restrict receivers_i_a{receivers_i.data_ptr<int64_t>()};
          int64_t *__restrict bg_receivers_i_a{
              bg_receivers_i.data_ptr<int64_t>()};
          scalar_t const *__restrict w_store_a{};
          scalar_t const *__restrict wsc_store_a{};
          if (v.requires_grad() || scatter.requires_grad()) {
            w_store_a = w_store.data_ptr<scalar_t>();
          }
          if (v.requires_grad()) {
            wsc_store_a = wsc_store.data_ptr<scalar_t>();
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
          decltype(&backward_batch_sc<scalar_t, 4>) backward_batch_scs[]{
              backward_batch_sc<scalar_t, 2>, backward_batch_sc<scalar_t, 4>,
              backward_batch_sc<scalar_t, 6>, backward_batch_sc<scalar_t, 8>};
          if (non_sc) {
            backward_batches[accuracy / 2 - 1](
                wfc_a, wfp_a, wfcn_a, psiy_a, psix_a, psiyn_a, psixn_a, zetay_a,
                zetax_a, zetayn_a, zetaxn_a, wfcsc_a, wfpsc_a, wfcnsc_a,
                psiysc_a, psixsc_a, psiynsc_a, psixnsc_a, zetaysc_a, zetaxsc_a,
                zetaynsc_a, zetaxnsc_a, sources_i_a, receivers_i_a,
                bg_receivers_i_a, w_store_a, wsc_store_a, v_a, scatter_a,
                grad_f_a, grad_r_a, grad_bg_r_a, grad_v_batch_a,
                grad_scatter_batch_a, ay_a, ax_a, by_a, bx_a, daydy_a, daxdx_a,
                dbydy_a, dbxdx_a, n_sources_per_shot, n_receivers_per_shot,
                n_bg_receivers_per_shot, ny, nx, nt, step_ratio,
                v.requires_grad(), scatter.requires_grad(), n_batch);
          } else {
            backward_batch_scs[accuracy / 2 - 1](
                wfcsc_a, wfpsc_a, wfcnsc_a, psiysc_a, psixsc_a, psiynsc_a,
                psixnsc_a, zetaysc_a, zetaxsc_a, zetaynsc_a, zetaxnsc_a,
                receivers_i_a, w_store_a, v_a, grad_r_a, grad_scatter_batch_a,
                ay_a, ax_a, by_a, bx_a, daydy_a, daxdx_a, dbydy_a, dbxdx_a,
                n_receivers_per_shot, ny, nx, nt, step_ratio,
                scatter.requires_grad(), n_batch);
          }
          dim3 dimBlock_combine(32, 32, 1);
          auto gridx_combine{ceil_div(
              nx - 2 * fd_pad, static_cast<int64_t>(dimBlock_combine.x))};
          auto gridy_combine{ceil_div(
              ny - 2 * fd_pad, static_cast<int64_t>(dimBlock_combine.y))};
          auto gridz_combine{1};
          dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
          decltype(&combine_grad_model<scalar_t, 4>) combine_grad_models[]{
              combine_grad_model<scalar_t, 2>, combine_grad_model<scalar_t, 4>,
              combine_grad_model<scalar_t, 6>, combine_grad_model<scalar_t, 8>};
          auto combine_grad_modeli{combine_grad_models[accuracy / 2 - 1]};

          if (v.requires_grad() && n_batch > 1) {
            combine_grad_modeli<<<dimGrid_combine, dimBlock_combine>>>(
                grad_v_a, grad_v_batch_a, n_batch);
            gpuErrchk(cudaPeekAtLastError());
          }
          if (scatter.requires_grad() && n_batch > 1) {
            combine_grad_modeli<<<dimGrid_combine, dimBlock_combine>>>(
                grad_scatter_a, grad_scatter_batch_a, n_batch);
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
    torch::Tensor *wfscnt;
    torch::Tensor *wfscntm1;
    torch::Tensor *psiyscntm1{nt & 1 ? &psiynsc : &psiysc};
    torch::Tensor *psixscntm1{nt & 1 ? &psixnsc : &psixsc};
    torch::Tensor *zetayscntm1{nt & 1 ? &zetaynsc : &zetaysc};
    torch::Tensor *zetaxscntm1{nt & 1 ? &zetaxnsc : &zetaxsc};
    if (nt % 3 == 1) {
      wfscnt = &wfpsc;
      wfscntm1 = &wfcnsc;
    } else if (nt % 3 == 2) {
      wfscnt = &wfcnsc;
      wfscntm1 = &wfcsc;
    } else {
      wfscnt = &wfcsc;
      wfscntm1 = &wfpsc;
    }

    if (non_sc) {
      zero_interior<true>(*psiyntm1, ny, nx, fd_pad, pml_width);
      zero_interior<false>(*psixntm1, ny, nx, fd_pad, pml_width);
      zero_interior<true>(*zetayntm1, ny, nx, fd_pad, pml_width);
      zero_interior<false>(*zetaxntm1, ny, nx, fd_pad, pml_width);
    }
    zero_interior<true>(*psiyscntm1, ny, nx, fd_pad, pml_width);
    zero_interior<false>(*psixscntm1, ny, nx, fd_pad, pml_width);
    zero_interior<true>(*zetayscntm1, ny, nx, fd_pad, pml_width);
    zero_interior<false>(*zetaxscntm1, ny, nx, fd_pad, pml_width);

    at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
    at::indexing::TensorIndex slicey{
        torch::indexing::Slice(fd_pad, ny - fd_pad)};
    at::indexing::TensorIndex slicex{
        torch::indexing::Slice(fd_pad, nx - fd_pad)};
    return {
        non_sc ? grad_v : torch::Tensor(),
        grad_scatter,
        non_sc ? grad_f : torch::Tensor(),
        non_sc ? wfnt->index({all_slice, slicey, slicex}) : torch::Tensor(),
        non_sc ? wfntm1->index({all_slice, slicey, slicex}) : torch::Tensor(),
        non_sc ? psiyntm1->index({all_slice, slicey, slicex}) : torch::Tensor(),
        non_sc ? psixntm1->index({all_slice, slicey, slicex}) : torch::Tensor(),
        non_sc ? zetayntm1->index({all_slice, slicey, slicex})
               : torch::Tensor(),
        non_sc ? zetaxntm1->index({all_slice, slicey, slicex})
               : torch::Tensor(),
        wfscnt->index({all_slice, slicey, slicex}),
        wfscntm1->index({all_slice, slicey, slicex}),
        psiyscntm1->index({all_slice, slicey, slicex}),
        psixscntm1->index({all_slice, slicey, slicex}),
        zetayscntm1->index({all_slice, slicey, slicex}),
        zetaxscntm1->index({all_slice, slicey, slicex}),
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

std::vector<torch::Tensor> scalar_born_cuda_autograd(
    torch::Tensor const &v, torch::Tensor const &scatter,
    torch::Tensor const &f, torch::Tensor const &wfc0,
    torch::Tensor const &wfp0, torch::Tensor const &psiy0,
    torch::Tensor const &psix0, torch::Tensor const &zetay0,
    torch::Tensor const &zetax0, torch::Tensor const &wfcsc0,
    torch::Tensor const &wfpsc0, torch::Tensor const &psiysc0,
    torch::Tensor const &psixsc0, torch::Tensor const &zetaysc0,
    torch::Tensor const &zetaxsc0, torch::Tensor const &ay,
    torch::Tensor const &ax, torch::Tensor const &by, torch::Tensor const &bx,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i,
    torch::Tensor const &bg_receivers_i, double dy, double dx, double dt,
    int64_t nt, int64_t n_batch, int64_t step_ratio, int64_t accuracy,
    int64_t pml_width0, int64_t pml_width1, int64_t pml_width2,
    int64_t pml_width3) {
  return ScalarBornCUDAFunction::apply(
      v, scatter, f, wfc0, wfp0, psiy0, psix0, zetay0, zetax0, wfcsc0, wfpsc0,
      psiysc0, psixsc0, zetaysc0, zetaxsc0, ay, ax, by, bx, sources_i,
      receivers_i, bg_receivers_i, dy, dx, dt, nt, n_batch, step_ratio,
      accuracy, pml_width0, pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCUDA, m) {
  m.impl("scalar_born", scalar_born_cuda_autograd);
}

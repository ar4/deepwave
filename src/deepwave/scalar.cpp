#include <torch/script.h>
#include <torch/torch.h>

TORCH_LIBRARY_FRAGMENT(deepwave, m) {
  m.def(
      "scalar(Tensor v, Tensor f, Tensor wfc0, Tensor wfp0, Tensor psiy0, "
      "Tensor psix0, Tensor zetay0, Tensor zetax0, Tensor ay, Tensor ax, "
      "Tensor by, Tensor bx, Tensor sources_i, Tensor receivers_i, float dy, "
      "float dx, float dt, int nt, int n_batch, int step_ratio, int accuracy, "
      "int pml_width0, int pml_width1, int pml_width2, int pml_width3) "
      "-> Tensor[]");
}

std::vector<torch::Tensor> scalar(
    torch::Tensor const &v, torch::Tensor const &f, torch::Tensor const &wfc0,
    torch::Tensor const &wfp0, torch::Tensor const &psiy0,
    torch::Tensor const &psix0, torch::Tensor const &zetay0,
    torch::Tensor const &zetax0, torch::Tensor const &ay,
    torch::Tensor const &ax, torch::Tensor const &by, torch::Tensor const &bx,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i, double dy,
    double dx, double dt, int64_t nt, int64_t n_batch, int64_t step_ratio,
    int64_t accuracy, int64_t pml_width0, int64_t pml_width1,
    int64_t pml_width2, int64_t pml_width3) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("deepwave::scalar", "")
                       .typed<decltype(scalar)>();
  return op.call(v, f, wfc0, wfp0, psiy0, psix0, zetay0, zetax0, ay, ax, by, bx,
                 sources_i, receivers_i, dy, dx, dt, nt, n_batch, step_ratio,
                 accuracy, pml_width0, pml_width1, pml_width2, pml_width3);
}

namespace {

template <typename T, int A>
inline T diffx(T const *__restrict a, T const *__restrict fd_coeffs) {
  if (A == 2) {
    return fd_coeffs[0] * (a[1] - a[-1]);
  } else if (A == 4) {
    return (fd_coeffs[0] * (a[1] - a[-1]) + fd_coeffs[1] * (a[2] - a[-2]));
  } else if (A == 6) {
    return (fd_coeffs[0] * (a[1] - a[-1]) + fd_coeffs[1] * (a[2] - a[-2]) +
            fd_coeffs[2] * (a[3] - a[-3]));
  } else {
    return (fd_coeffs[0] * (a[1] - a[-1]) + fd_coeffs[1] * (a[2] - a[-2]) +
            fd_coeffs[2] * (a[3] - a[-3]) + fd_coeffs[3] * (a[4] - a[-4]));
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

template <typename T, int A, bool v_requires_grad, int pml_y, int pml_x>
inline void forward_kernel(
    T const *__restrict wfc, T *__restrict wfp, T const *__restrict psiy,
    T const *__restrict psix, T *__restrict psiyn, T *__restrict psixn,
    T *__restrict zetay, T *__restrict zetax, T *__restrict dwdv,
    T const *__restrict v, T const *__restrict ay, T const *__restrict ax,
    T const *__restrict by, T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx, int64_t nx, T dt2,
    T const *__restrict fd_coeffs1y, T const *__restrict fd_coeffs1x,
    T const *__restrict fd_coeffs2y, T const *__restrict fd_coeffs2x,
    int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
  int64_t ybegin;
  int64_t yend;
  int64_t xbegin;
  int64_t xend;
  if (pml_y == 0) {
    ybegin = pml_regionsy[0];
    yend = pml_regionsy[1];
  } else if (pml_y == 2) {
    ybegin = pml_regionsy[2];
    yend = pml_regionsy[3];
  } else {
    ybegin = pml_regionsy[1];
    yend = pml_regionsy[2];
  }
  if (pml_x == 0) {
    xbegin = pml_regionsx[0];
    xend = pml_regionsx[1];
  } else if (pml_x == 2) {
    xbegin = pml_regionsx[2];
    xend = pml_regionsx[3];
  } else {
    xbegin = pml_regionsx[1];
    xend = pml_regionsx[2];
  }

  for (int64_t y = ybegin; y < yend; ++y) {
    int64_t yi{y * nx};
    auto ayi{ay[y]};
    auto byi{by[y]};
    auto daydyi{daydy[y]};
    auto dbydyi{dbydy[y]};
    for (int64_t x = xbegin; x < xend; ++x) {
      int64_t i{yi + x};
      T d2wdy2;
      T d2wdx2;
#define D2WDY20 fd_coeffs2y[0] * wfc[i]
#define D2WDY2(t) fd_coeffs2y[t] * (wfc[i + t * nx] + wfc[i - t * nx])
#define D2WDX20 fd_coeffs2x[0] * wfc[i]
#define D2WDX2(t) fd_coeffs2x[t] * (wfc[i + t] + wfc[i - t])
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
      if (pml_y == 0 || pml_y == 2) {
        T dwdy;
        T dpsiydy;
#define DDY(a, t) fd_coeffs1y[t] * (a[i + (t + 1) * nx] - a[i - (t + 1) * nx])
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
        T tmpy{(1 + byi) * d2wdy2 + dbydyi * dwdy + daydyi * psiy[i] +
               ayi * dpsiydy};
        w_sum += (1 + byi) * tmpy + ayi * zetay[i];
        psiyn[i] = byi * dwdy + ayi * psiy[i];
        zetay[i] = byi * tmpy + ayi * zetay[i];
      } else {
        w_sum += d2wdy2;
      }
      if (pml_x == 0 || pml_x == 2) {
        T dwdx;
        T dpsixdx;
#define DDX(a, t) fd_coeffs1x[t] * (a[i + (t + 1)] - a[i - (t + 1)])
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
        dwdv[i] = 2 * v[i] * dt2 * w_sum;
      }
      wfp[i] = v[i] * v[i] * dt2 * w_sum + 2 * wfc[i] - wfp[i];
    }
  }
}

template <typename T>
void add_sources(T *__restrict wf, T const *__restrict f,
                 int64_t const *__restrict sources_i,
                 int64_t n_sources_per_shot) {
  for (int64_t source_idx{}; source_idx < n_sources_per_shot; ++source_idx) {
    wf[sources_i[source_idx]] += f[source_idx];
  }
}

template <typename T>
void record_receivers(T *__restrict r, T const *__restrict wf,
                      int64_t const *__restrict receivers_i,
                      int64_t n_receivers_per_shot) {
  for (int64_t receiver_idx{}; receiver_idx < n_receivers_per_shot;
       ++receiver_idx) {
    r[receiver_idx] = wf[receivers_i[receiver_idx]];
  }
}

template <typename T, int A>
void add_to_grad_v(T *__restrict grad_v, T const *__restrict wfc,
                   T const *__restrict dwdv, int64_t step_ratio, int64_t ny,
                   int64_t nx) {
  constexpr int fd_pad{A / 2};
  for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
      int64_t i{yi + x};
      grad_v[i] += wfc[i] * dwdv[i] * step_ratio;
    }
  }
}

template <typename T, int A, int pml_y, int pml_x>
inline void backward_kernel(
    T const *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
    T const *__restrict psiy, T const *__restrict psix, T *__restrict psiyn,
    T *__restrict psixn, T const *__restrict zetay, T const *__restrict zetax,
    T *__restrict zetayn, T *__restrict zetaxn, T const *__restrict v2dt2,
    T const *__restrict ay, T const *__restrict ax, T const *__restrict by,
    T const *__restrict bx, T const *__restrict daydy,
    T const *__restrict daxdx, T const *__restrict dbydy,
    T const *__restrict dbxdx, int64_t nx, T const *__restrict fd_coeffs1y,
    T const *__restrict fd_coeffs1x, T const *__restrict fd_coeffs2y,
    T const *__restrict fd_coeffs2x, int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
  int64_t ybegin;
  int64_t yend;
  int64_t xbegin;
  int64_t xend;
  if (pml_y == 0) {
    ybegin = pml_regionsy[0];
    yend = pml_regionsy[1];
  } else if (pml_y == 2) {
    ybegin = pml_regionsy[2];
    yend = pml_regionsy[3];
  } else {
    ybegin = pml_regionsy[1];
    yend = pml_regionsy[2];
  }
  if (pml_x == 0) {
    xbegin = pml_regionsx[0];
    xend = pml_regionsx[1];
  } else if (pml_x == 2) {
    xbegin = pml_regionsx[2];
    xend = pml_regionsx[3];
  } else {
    xbegin = pml_regionsx[1];
    xend = pml_regionsx[2];
  }
  for (int64_t y = ybegin; y < yend; ++y) {
    int64_t yi{y * nx};
    auto ayi{ay[y]};
    auto byi{by[y]};
    auto daydyi{daydy[y]};
    for (int64_t x = xbegin; x < xend; ++x) {
      int64_t i{yi + x};
      T wfp_y_term;
      T wfp_x_term;
#define WFPY0 fd_coeffs2y[0] * v2dt2[i] * wfc[i]
#define WFPY(t)                                           \
  fd_coeffs2y[t] * (v2dt2[i + t * nx] * wfc[i + t * nx] + \
                    v2dt2[i - t * nx] * wfc[i - t * nx])
#define WFPYPML0   \
  fd_coeffs2y[0] * \
      ((1 + by[y]) * ((1 + by[y]) * v2dt2[i] * wfc[i] + by[y] * zetay[i]))
#define WFPYPML(t)                                                            \
  fd_coeffs2y[t] * (((1 + by[y + t]) *                                        \
                     ((1 + by[y + t]) * v2dt2[i + t * nx] * wfc[i + t * nx] + \
                      by[y + t] * zetay[i + t * nx])) +                       \
                    ((1 + by[y - t]) *                                        \
                     ((1 + by[y - t]) * v2dt2[i - t * nx] * wfc[i - t * nx] + \
                      by[y - t] * zetay[i - t * nx]))) -                      \
      (fd_coeffs1y[t - 1] *                                                   \
       ((dbydy[y + t] *                                                       \
             ((1 + by[y + t]) * v2dt2[i + t * nx] * wfc[i + t * nx] +         \
              by[y + t] * zetay[i + t * nx]) +                                \
         by[y + t] * psiy[i + t * nx]) -                                      \
        (dbydy[y - t] *                                                       \
             ((1 + by[y - t]) * v2dt2[i - t * nx] * wfc[i - t * nx] +         \
              by[y - t] * zetay[i - t * nx]) +                                \
         by[y - t] * psiy[i - t * nx])))
      if (A == 2) {
        if (pml_y == 1) {
          wfp_y_term = WFPY0 + WFPY(1);
        } else {
          wfp_y_term = WFPYPML0 + WFPYPML(1);
        }
      } else if (A == 4) {
        if (pml_y == 1) {
          wfp_y_term = WFPY0 + WFPY(1) + WFPY(2);
        } else {
          wfp_y_term = WFPYPML0 + WFPYPML(1) + WFPYPML(2);
        }
      } else if (A == 6) {
        if (pml_y == 1) {
          wfp_y_term = WFPY0 + WFPY(1) + WFPY(2) + WFPY(3);
        } else {
          wfp_y_term = WFPYPML0 + WFPYPML(1) + WFPYPML(2) + WFPYPML(3);
        }
      } else {
        if (pml_y == 1) {
          wfp_y_term = WFPY0 + WFPY(1) + WFPY(2) + WFPY(3) + WFPY(4);
        } else {
          wfp_y_term =
              WFPYPML0 + WFPYPML(1) + WFPYPML(2) + WFPYPML(3) + WFPYPML(4);
        }
      }
#define WFPX0 fd_coeffs2x[0] * v2dt2[i] * wfc[i]
#define WFPX(t) \
  fd_coeffs2x[t] * (v2dt2[i + t] * wfc[i + t] + v2dt2[i - t] * wfc[i - t])
#define WFPXPML0   \
  fd_coeffs2x[0] * \
      ((1 + bx[x]) * ((1 + bx[x]) * v2dt2[i] * wfc[i] + bx[x] * zetax[i]))
#define WFPXPML(t)                                                           \
  fd_coeffs2x[t] *                                                           \
          (((1 + bx[x + t]) * ((1 + bx[x + t]) * v2dt2[i + t] * wfc[i + t] + \
                               bx[x + t] * zetax[i + t])) +                  \
           ((1 + bx[x - t]) * ((1 + bx[x - t]) * v2dt2[i - t] * wfc[i - t] + \
                               bx[x - t] * zetax[i - t]))) -                 \
      (fd_coeffs1x[t - 1] *                                                  \
       ((dbxdx[x + t] * ((1 + bx[x + t]) * v2dt2[i + t] * wfc[i + t] +       \
                         bx[x + t] * zetax[i + t]) +                         \
         bx[x + t] * psix[i + t]) -                                          \
        (dbxdx[x - t] * ((1 + bx[x - t]) * v2dt2[i - t] * wfc[i - t] +       \
                         bx[x - t] * zetax[i - t]) +                         \
         bx[x - t] * psix[i - t])))
      if (A == 2) {
        if (pml_x == 1) {
          wfp_x_term = WFPX0 + WFPX(1);
        } else {
          wfp_x_term = WFPXPML0 + WFPXPML(1);
        }
      } else if (A == 4) {
        if (pml_x == 1) {
          wfp_x_term = WFPX0 + WFPX(1) + WFPX(2);
        } else {
          wfp_x_term = WFPXPML0 + WFPXPML(1) + WFPXPML(2);
        }
      } else if (A == 6) {
        if (pml_x == 1) {
          wfp_x_term = WFPX0 + WFPX(1) + WFPX(2) + WFPX(3);
        } else {
          wfp_x_term = WFPXPML0 + WFPXPML(1) + WFPXPML(2) + WFPXPML(3);
        }
      } else {
        if (pml_x == 1) {
          wfp_x_term = WFPX0 + WFPX(1) + WFPX(2) + WFPX(3) + WFPX(4);
        } else {
          wfp_x_term =
              WFPXPML0 + WFPXPML(1) + WFPXPML(2) + WFPXPML(3) + WFPXPML(4);
        }
      }

      wfp[i] = wfp_y_term + wfp_x_term + 2 * wfc[i] + wfp[i];
      wfcn[i] = -wfc[i];
      if (pml_y == 0 || pml_y == 2) {
        T tmp;
#define PSIY(t)                                                              \
  fd_coeffs1y[t - 1] *                                                       \
      ((ay[y + t] * ((1 + by[y + t]) * v2dt2[i + t * nx] * wfc[i + t * nx] + \
                     by[y + t] * zetay[i + t * nx])) -                       \
       (ay[y - t] * ((1 + by[y - t]) * v2dt2[i - t * nx] * wfc[i - t * nx] + \
                     by[y - t] * zetay[i - t * nx])))
        if (A == 2) {
          tmp = -(PSIY(1));
        } else if (A == 4) {
          tmp = -(PSIY(1) + PSIY(2));
        } else if (A == 6) {
          tmp = -(PSIY(1) + PSIY(2) + PSIY(3));
        } else {
          tmp = -(PSIY(1) + PSIY(2) + PSIY(3) + PSIY(4));
        }
        psiyn[i] = tmp +
                   daydyi * ((1 + byi) * v2dt2[i] * wfc[i] + byi * zetay[i]) +
                   ayi * psiy[i];
        zetayn[i] = ayi * v2dt2[i] * wfc[i] + ayi * zetay[i];
      }
      if (pml_x == 0 || pml_x == 2) {
        T tmp;
#define PSIX(t)                                                    \
  fd_coeffs1x[t - 1] *                                             \
      ((ax[x + t] * ((1 + bx[x + t]) * v2dt2[i + t] * wfc[i + t] + \
                     bx[x + t] * zetax[i + t])) -                  \
       (ax[x - t] * ((1 + bx[x - t]) * v2dt2[i - t] * wfc[i - t] + \
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
            daxdx[x] * ((1 + bx[x]) * v2dt2[i] * wfc[i] + bx[x] * zetax[i]) +
            ax[x] * psix[i];
        zetaxn[i] = ax[x] * v2dt2[i] * wfc[i] + ax[x] * zetax[i];
      }
    }
  }
}

template <typename T, int A>
void forward_shot(T *__restrict wfc, T *__restrict wfp, T *__restrict psiy,
                  T *__restrict psix, T *__restrict psiyn, T *__restrict psixn,
                  T *__restrict zetay, T *__restrict zetax,
                  int64_t const *__restrict sources_i,
                  int64_t const *__restrict receivers_i, T *__restrict dwdv,
                  T const *__restrict v, T const *__restrict f, T *__restrict r,
                  T const *__restrict ay, T const *__restrict ax,
                  T const *__restrict by, T const *__restrict bx,
                  T const *__restrict daydy, T const *__restrict daxdx,
                  T const *__restrict dbydy, T const *__restrict dbxdx, T dt2,
                  T const *__restrict fd_coeffs1y,
                  T const *__restrict fd_coeffs1x,
                  T const *__restrict fd_coeffs2y,
                  T const *__restrict fd_coeffs2x, int64_t n_sources_per_shot,
                  int64_t n_receivers_per_shot, int64_t ny, int64_t nx,
                  int64_t nt, int64_t step_ratio, bool v_requires_grad,
                  int64_t const *__restrict pml_regionsy,
                  int64_t const *__restrict pml_regionsx) {
#define FORWARD_KERNELGRAD(pml_y, pml_x)                                  \
  forward_kernel<T, A, true, pml_y, pml_x>(                               \
      wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax,                   \
      dwdv + (t / step_ratio) * ny * nx, v, ay, ax, by, bx, daydy, daxdx, \
      dbydy, dbxdx, nx, dt2, fd_coeffs1y, fd_coeffs1x, fd_coeffs2y,       \
      fd_coeffs2x, pml_regionsy, pml_regionsx)
#define FORWARD_KERNELNOGRAD(pml_y, pml_x)                                   \
  forward_kernel<T, A, false, pml_y, pml_x>(                                 \
      wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, nullptr, v, ay, ax,  \
      by, bx, daydy, daxdx, dbydy, dbxdx, nx, dt2, fd_coeffs1y, fd_coeffs1x, \
      fd_coeffs2y, fd_coeffs2x, pml_regionsy, pml_regionsx)
  for (int64_t t{}; t < nt; ++t) {
    if (t % step_ratio == 0 && v_requires_grad) {
      FORWARD_KERNELGRAD(0, 0);
      FORWARD_KERNELGRAD(0, 1);
      FORWARD_KERNELGRAD(0, 2);
      FORWARD_KERNELGRAD(1, 0);
      FORWARD_KERNELGRAD(1, 1);
      FORWARD_KERNELGRAD(1, 2);
      FORWARD_KERNELGRAD(2, 0);
      FORWARD_KERNELGRAD(2, 1);
      FORWARD_KERNELGRAD(2, 2);
    } else {
      FORWARD_KERNELNOGRAD(0, 0);
      FORWARD_KERNELNOGRAD(0, 1);
      FORWARD_KERNELNOGRAD(0, 2);
      FORWARD_KERNELNOGRAD(1, 0);
      FORWARD_KERNELNOGRAD(1, 1);
      FORWARD_KERNELNOGRAD(1, 2);
      FORWARD_KERNELNOGRAD(2, 0);
      FORWARD_KERNELNOGRAD(2, 1);
      FORWARD_KERNELNOGRAD(2, 2);
    }

    if (n_sources_per_shot > 0) {
      add_sources(wfp, f + t * n_sources_per_shot, sources_i,
                  n_sources_per_shot);
    }
    if (n_receivers_per_shot > 0) {
      record_receivers(r + t * n_receivers_per_shot, wfc, receivers_i,
                       n_receivers_per_shot);
    }
    std::swap(wfp, wfc);
    std::swap(psiyn, psiy);
    std::swap(psixn, psix);
  }
}

template <typename T, int A>
void combine_grad_v(T *__restrict grad_v, T const *__restrict grad_v_batch,
                    int64_t n_parallel, int64_t ny, int64_t nx) {
  constexpr int fd_pad{A / 2};
  int64_t nynx{ny * nx};
  for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
      int64_t i{yi + x};
      for (int64_t batch{}; batch < n_parallel; ++batch) {
        grad_v[i] += grad_v_batch[batch * nynx + i];
      }
    }
  }
}

template <typename T, int A>
void backward_shot(
    T *__restrict wfc, T *__restrict wfp, T *__restrict wfcn,
    T *__restrict psiy, T *__restrict psix, T *__restrict psiyn,
    T *__restrict psixn, T *__restrict zetay, T *__restrict zetax,
    T *__restrict zetayn, T *__restrict zetaxn,
    int64_t const *__restrict sources_i, int64_t const *__restrict receivers_i,
    T const *__restrict dwdv, T const *__restrict v2dt2, T *__restrict f,
    T const *__restrict r, T *__restrict grad_v, T const *__restrict ay,
    T const *__restrict ax, T const *__restrict by, T const *__restrict bx,
    T const *__restrict daydy, T const *__restrict daxdx,
    T const *__restrict dbydy, T const *__restrict dbxdx,
    T const *__restrict fd_coeffs1y, T const *__restrict fd_coeffs1x,
    T const *__restrict fd_coeffs2y, T const *__restrict fd_coeffs2x,
    int64_t n_sources_per_shot, int64_t n_receivers_per_shot, int64_t ny,
    int64_t nx, int64_t nt, int64_t step_ratio, bool v_requires_grad,
    int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
#define BACKWARD_KERNEL(pml_y, pml_x)                                         \
  backward_kernel<T, A, pml_y, pml_x>(                                        \
      wfc, wfp, wfcn, psiy, psix, psiyn, psixn, zetay, zetax, zetayn, zetaxn, \
      v2dt2, ay, ax, by, bx, daydy, daxdx, dbydy, dbxdx, nx, fd_coeffs1y,     \
      fd_coeffs1x, fd_coeffs2y, fd_coeffs2x, pml_regionsy, pml_regionsx)
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (n_receivers_per_shot > 0) {
      add_sources(wfp, r + t * n_receivers_per_shot, receivers_i,
                  n_receivers_per_shot);
    }
    if (n_sources_per_shot > 0) {
      record_receivers(f + t * n_sources_per_shot, wfc, sources_i,
                       n_sources_per_shot);
    }
    if (t % step_ratio == 0 && v_requires_grad) {
      add_to_grad_v<T, A>(grad_v, wfc, dwdv + (t / step_ratio) * ny * nx,
                          step_ratio, ny, nx);
    }
    BACKWARD_KERNEL(0, 0);
    BACKWARD_KERNEL(0, 1);
    BACKWARD_KERNEL(0, 2);
    BACKWARD_KERNEL(1, 0);
    BACKWARD_KERNEL(1, 1);
    BACKWARD_KERNEL(1, 2);
    BACKWARD_KERNEL(2, 0);
    BACKWARD_KERNEL(2, 1);
    BACKWARD_KERNEL(2, 2);
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

class ScalarCPUFunction : public torch::autograd::Function<ScalarCPUFunction> {
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
    auto r{at::empty({n_batch, nt, n_receivers_per_shot}, options)};
    auto daydy{at::zeros_like(ay)};
    auto daxdx{at::zeros_like(ax)};
    auto dbydy{at::zeros_like(by)};
    auto dbxdx{at::zeros_like(bx)};
    torch::Tensor dwdv;
    if (v.requires_grad()) {
      dwdv = at::empty({n_batch, (nt + step_ratio - 1) / step_ratio, ny, nx},
                       options);
    }

    zero_interior<true>(psiy, ny, nx, fd_pad, pml_width);
    zero_interior<false>(psix, ny, nx, fd_pad, pml_width);
    zero_interior<true>(zetay, ny, nx, fd_pad, pml_width);
    zero_interior<false>(zetax, ny, nx, fd_pad, pml_width);

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cpu_forward", ([&] {
          scalar_t dt2_a = dt * dt;
          scalar_t fd_coeffs1y[4];
          scalar_t fd_coeffs1x[4];
          scalar_t fd_coeffs2y[5];
          scalar_t fd_coeffs2x[5];
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
          set_fd_coeffs(fd_coeffs1y, fd_coeffs2y, accuracy,
                        static_cast<scalar_t>(dy));
          set_fd_coeffs(fd_coeffs1x, fd_coeffs2x, accuracy,
                        static_cast<scalar_t>(dx));
          decltype(&diffx<scalar_t, 4>) diffxs[]{
              diffx<scalar_t, 2>, diffx<scalar_t, 4>, diffx<scalar_t, 6>,
              diffx<scalar_t, 8>};
          auto diffxi{diffxs[accuracy / 2 - 1]};
          for (int64_t y{fd_pad}; y < ny - fd_pad; ++y) {
            daydy_a[y] = diffxi(ay_a + y, fd_coeffs1y);
          }
          for (int64_t x{fd_pad}; x < nx - fd_pad; ++x) {
            daxdx_a[x] = diffxi(ax_a + x, fd_coeffs1x);
          }
          for (int64_t y{fd_pad}; y < ny - fd_pad; ++y) {
            dbydy_a[y] = diffxi(by_a + y, fd_coeffs1y);
          }
          for (int64_t x{fd_pad}; x < nx - fd_pad; ++x) {
            dbxdx_a[x] = diffxi(bx_a + x, fd_coeffs1x);
          }
          decltype(&forward_shot<scalar_t, 4>) forward_shots[]{
              forward_shot<scalar_t, 2>, forward_shot<scalar_t, 4>,
              forward_shot<scalar_t, 6>, forward_shot<scalar_t, 8>};
          at::parallel_for(0, n_batch, 0, [&](int64_t bstart, int64_t bend) {
            for (int64_t shot = bstart; shot < bend; ++shot) {
              auto i{shot * ny * nx};
              auto si{shot * n_sources_per_shot};
              auto ri{shot * n_receivers_per_shot};
              forward_shots[accuracy / 2 - 1](
                  wfc_a + i, wfp_a + i, psiy_a + i, psix_a + i, psiyn_a + i,
                  psixn_a + i, zetay_a + i, zetax_a + i, sources_i_a + si,
                  receivers_i_a + ri, dwdv_a + i * (nt / step_ratio), v_a,
                  f_a + si * nt, r_a + ri * nt, ay_a, ax_a, by_a, bx_a, daydy_a,
                  daxdx_a, dbydy_a, dbxdx_a, dt2_a, fd_coeffs1y, fd_coeffs1x,
                  fd_coeffs2y, fd_coeffs2x, n_sources_per_shot,
                  n_receivers_per_shot, ny, nx, nt, step_ratio,
                  v.requires_grad(), pml_regionsy, pml_regionsx);
            }
          });
        }));

    if (v.requires_grad() || f.requires_grad() || wfc0.requires_grad() ||
        wfp0.requires_grad() || psiy0.requires_grad() ||
        psix0.requires_grad() || zetay0.requires_grad() ||
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

    int64_t n_parallel{
        std::min(static_cast<int>(n_batch), at::get_num_threads())};
    auto grad_v{at::zeros({ny, nx}, options)};
    auto grad_v_batch{n_parallel > 1 ? at::zeros({n_parallel, ny, nx}, options)
                                     : at::empty(0)};
    auto grad_f{at::empty({n_batch, nt, n_sources_per_shot}, options)};

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cpu_backward", ([&] {
          scalar_t dt2_a = dt * dt;
          scalar_t fd_coeffs1y[4];
          scalar_t fd_coeffs1x[4];
          scalar_t fd_coeffs2y[5];
          scalar_t fd_coeffs2x[5];
          auto v2dt2{v * v * dt2_a};
          scalar_t const *__restrict v2dt2_a{v2dt2.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_v_a{grad_v.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_v_batch_a{
              n_parallel > 1 ? grad_v_batch.data_ptr<scalar_t>() : grad_v_a};
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
          set_fd_coeffs(fd_coeffs1y, fd_coeffs2y, accuracy,
                        static_cast<scalar_t>(dy));
          set_fd_coeffs(fd_coeffs1x, fd_coeffs2x, accuracy,
                        static_cast<scalar_t>(dx));
          decltype(&diffx<scalar_t, 4>) diffxs[]{
              diffx<scalar_t, 2>, diffx<scalar_t, 4>, diffx<scalar_t, 6>,
              diffx<scalar_t, 8>};
          auto diffxi{diffxs[accuracy / 2 - 1]};
          for (int64_t y{fd_pad}; y < ny - fd_pad; ++y) {
            daydy_a[y] = diffxi(ay_a + y, fd_coeffs1y);
          }
          for (int64_t x{fd_pad}; x < nx - fd_pad; ++x) {
            daxdx_a[x] = diffxi(ax_a + x, fd_coeffs1x);
          }
          for (int64_t y{fd_pad}; y < ny - fd_pad; ++y) {
            dbydy_a[y] = diffxi(by_a + y, fd_coeffs1y);
          }
          for (int64_t x{fd_pad}; x < nx - fd_pad; ++x) {
            dbxdx_a[x] = diffxi(bx_a + x, fd_coeffs1x);
          }
          decltype(&backward_shot<scalar_t, 4>) backward_shots[]{
              backward_shot<scalar_t, 2>, backward_shot<scalar_t, 4>,
              backward_shot<scalar_t, 6>, backward_shot<scalar_t, 8>};
          at::parallel_for(0, n_batch, 0, [&](int64_t bstart, int64_t bend) {
            for (int64_t shot = bstart; shot < bend; ++shot) {
              auto i{shot * ny * nx};
              auto si{shot * n_sources_per_shot};
              auto ri{shot * n_receivers_per_shot};
              backward_shots[accuracy / 2 - 1](
                  wfc_a + i, wfp_a + i, wfcn_a + i, psiy_a + i, psix_a + i,
                  psiyn_a + i, psixn_a + i, zetay_a + i, zetax_a + i,
                  zetayn_a + i, zetaxn_a + i, sources_i_a + si,
                  receivers_i_a + ri, dwdv_a + i * (nt / step_ratio), v2dt2_a,
                  grad_f_a + si * nt, grad_r_a + ri * nt,
                  grad_v_batch_a + at::get_thread_num() * ny * nx, ay_a, ax_a,
                  by_a, bx_a, daydy_a, daxdx_a, dbydy_a, dbxdx_a, fd_coeffs1y,
                  fd_coeffs1x, fd_coeffs2y, fd_coeffs2x, n_sources_per_shot,
                  n_receivers_per_shot, ny, nx, nt, step_ratio,
                  v.requires_grad(), pml_regionsy, pml_regionsx);
            }
          });
          if (v.requires_grad() && n_parallel > 1) {
            decltype(&combine_grad_v<scalar_t, 4>) combine_grad_vs[]{
                combine_grad_v<scalar_t, 2>, combine_grad_v<scalar_t, 4>,
                combine_grad_v<scalar_t, 6>, combine_grad_v<scalar_t, 8>};
            auto combine_grad_vi{combine_grad_vs[accuracy / 2 - 1]};
            combine_grad_vi(grad_v_a, grad_v_batch_a, n_parallel, ny, nx);
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

std::vector<torch::Tensor> scalar_cpu_autograd(
    torch::Tensor const &v, torch::Tensor const &f, torch::Tensor const &wfc0,
    torch::Tensor const &wfp0, torch::Tensor const &psiy0,
    torch::Tensor const &psix0, torch::Tensor const &zetay0,
    torch::Tensor const &zetax0, torch::Tensor const &ay,
    torch::Tensor const &ax, torch::Tensor const &by, torch::Tensor const &bx,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i, double dy,
    double dx, double dt, int64_t nt, int64_t n_batch, int64_t step_ratio,
    int64_t accuracy, int64_t pml_width0, int64_t pml_width1,
    int64_t pml_width2, int64_t pml_width3) {
  return ScalarCPUFunction::apply(
      v, f, wfc0, wfp0, psiy0, psix0, zetay0, zetax0, ay, ax, by, bx, sources_i,
      receivers_i, dy, dx, dt, nt, n_batch, step_ratio, accuracy, pml_width0,
      pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCPU, m) {
  m.impl("scalar", scalar_cpu_autograd);
}

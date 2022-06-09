#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>

TORCH_LIBRARY_FRAGMENT(deepwave, m) {
  m.def(
      "scalar_born(Tensor v, Tensor scatter, Tensor f, Tensor wfc0, Tensor "
      "wfp0, Tensor psix0, Tensor psiy0, Tensor zetax0, Tensor zetay0, Tensor "
      "wfcsc0, Tensor wfpsc0, Tensor psixsc0, Tensor psiysc0, Tensor zetaxsc0, "
      "Tensor zetaysc0, Tensor ax, Tensor ay, Tensor bx, Tensor by, Tensor "
      "sources_i, Tensor receivers_i, float dx, float dy, float dt, int nt, "
      "int n_batch, int step_ratio, int accuracy, "
      "int pml_width0, int pml_width1, int pml_width2, int pml_width3) "
      "-> Tensor[]");
}

std::vector<torch::Tensor> scalar_born(
    torch::Tensor const &v, torch::Tensor const &scatter,
    torch::Tensor const &f, torch::Tensor const &wfc0,
    torch::Tensor const &wfp0, torch::Tensor const &psix0,
    torch::Tensor const &psiy0, torch::Tensor const &zetax0,
    torch::Tensor const &zetay0, torch::Tensor const &wfcsc0,
    torch::Tensor const &wfpsc0, torch::Tensor const &psixsc0,
    torch::Tensor const &psiysc0, torch::Tensor const &zetaxsc0,
    torch::Tensor const &zetaysc0, torch::Tensor const &ax,
    torch::Tensor const &ay, torch::Tensor const &bx, torch::Tensor const &by,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i, double dx,
    double dy, double dt, int64_t nt, int64_t n_batch, int64_t step_ratio,
    int64_t accuracy, int64_t pml_width0, int64_t pml_width1,
    int64_t pml_width2, int64_t pml_width3) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("deepwave::scalar_born", "")
                       .typed<decltype(scalar_born)>();
  return op.call(v, scatter, f, wfc0, wfp0, psix0, psiy0, zetax0, zetay0,
                 wfcsc0, wfpsc0, psixsc0, psiysc0, zetaxsc0, zetaysc0, ax, ay,
                 bx, by, sources_i, receivers_i, dx, dy, dt, nt, n_batch,
                 step_ratio, accuracy, pml_width0, pml_width1, pml_width2,
                 pml_width3);
}

namespace {
template <typename T, int A>
inline T diffx(T const *__restrict a, T one_over_dx, int64_t ny) {
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
inline T diffx2(T const *__restrict a, T one_over_dx2, int64_t ny) {
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
inline T diffy(T const *__restrict a, T one_over_dy) {
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
inline T diffy2(T const *__restrict a, T one_over_dy2) {
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

template <typename T, int A>
void forward_kernel_1(T const *__restrict wfc, T *__restrict psix,
                      T *__restrict psiy, T const *__restrict wfcsc,
                      T *__restrict psixsc, T *__restrict psiysc,
                      T const *__restrict ax, T const *__restrict ay,
                      T const *__restrict bx, T const *__restrict by,
                      int64_t nx, int64_t ny, T one_over_dx, T one_over_dy) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    auto axi{ax[x]};
    auto bxi{bx[x]};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      psix[i] = bxi * diffx<T, A>(wfc + i, one_over_dx, ny) + axi * psix[i];
      psiy[i] = by[y] * diffy<T, A>(wfc + i, one_over_dy) + ay[y] * psiy[i];
      psixsc[i] =
          bxi * diffx<T, A>(wfcsc + i, one_over_dx, ny) + axi * psixsc[i];
      psiysc[i] =
          by[y] * diffy<T, A>(wfcsc + i, one_over_dy) + ay[y] * psiysc[i];
    }
  }
}

template <typename T, int A, bool v_requires_grad, bool scatter_requires_grad>
void forward_kernel_2(
    T const *__restrict wfc, T *__restrict wfp, T const *__restrict psix,
    T const *__restrict psiy, T *__restrict zetax, T *__restrict zetay,
    T const *__restrict wfcsc, T *__restrict wfpsc, T const *__restrict psixsc,
    T const *__restrict psiysc, T *__restrict zetaxsc, T *__restrict zetaysc,
    T *__restrict w_store, T *__restrict wsc_store, T const *__restrict v2dt2,
    T const *__restrict two_vdt2, T const *__restrict scatter,
    T const *__restrict ax, T const *__restrict ay, T const *__restrict bx,
    T const *__restrict by, int64_t nx, int64_t ny, T one_over_dx,
    T one_over_dy, T one_over_dx2, T one_over_dy2) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    auto axi{ax[x]};
    auto bxi{bx[x]};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      T d2wdx2{diffx2<T, A>(wfc + i, one_over_dx2, ny)};
      T d2wdy2{diffy2<T, A>(wfc + i, one_over_dy2)};
      T psix_x{diffx<T, A>(psix + i, one_over_dx, ny)};
      T psiy_y{diffy<T, A>(psiy + i, one_over_dy)};
      zetax[i] = axi * zetax[i] + bxi * (d2wdx2 + psix_x);
      zetay[i] = ay[y] * zetay[i] + by[y] * (d2wdy2 + psiy_y);
      T d2wdx2sc{diffx2<T, A>(wfcsc + i, one_over_dx2, ny)};
      T d2wdy2sc{diffy2<T, A>(wfcsc + i, one_over_dy2)};
      T psixsc_x{diffx<T, A>(psixsc + i, one_over_dx, ny)};
      T psiysc_y{diffy<T, A>(psiysc + i, one_over_dy)};
      zetaxsc[i] = axi * zetaxsc[i] + bxi * (d2wdx2sc + psixsc_x);
      zetaysc[i] = ay[y] * zetaysc[i] + by[y] * (d2wdy2sc + psiysc_y);
      T wf_sum{d2wdx2 + d2wdy2 + psix_x + psiy_y + zetax[i] + zetay[i]};
      T wfsc_sum{d2wdx2sc + d2wdy2sc + psixsc_x + psiysc_y + zetaxsc[i] +
                 zetaysc[i]};
      if (v_requires_grad or scatter_requires_grad) {
        w_store[i] = wf_sum;
      }
      if (v_requires_grad) {
        wsc_store[i] = wfsc_sum;
      }
      wfp[i] = v2dt2[i] * wf_sum + 2 * wfc[i] - wfp[i];
      wfpsc[i] = v2dt2[i] * wfsc_sum + 2 * wfcsc[i] - wfpsc[i] +
                 two_vdt2[i] * scatter[i] * wf_sum;
    }
  }
}

template <typename T, bool requires_grad>
void add_sources(T *__restrict wf, T *__restrict wfsc, T *__restrict w_store,
                 T const *__restrict f, T const *__restrict v2dt2,
                 T const *__restrict two_vdt2, T const *__restrict scatter,
                 int64_t const *__restrict sources_i,
                 int64_t n_sources_per_shot) {
  for (int64_t source_idx{}; source_idx < n_sources_per_shot; ++source_idx) {
    auto i{sources_i[source_idx]};
    wf[i] -= f[source_idx] * v2dt2[i];
    wfsc[i] -= f[source_idx] * scatter[i] * two_vdt2[i];
    if (requires_grad) w_store[i] -= f[source_idx];
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

template <typename T>
void add_adjoint_sources(T *__restrict wf, T const *__restrict f,
                         int64_t const *__restrict sources_i,
                         int64_t n_sources_per_shot) {
  for (int64_t source_idx{}; source_idx < n_sources_per_shot; ++source_idx) {
    wf[sources_i[source_idx]] += f[source_idx];
  }
}

template <typename T>
void record_adjoint_receivers(T *__restrict r, T const *__restrict wf,
                              T const *__restrict wfsc,
                              T const *__restrict v2dt2,
                              T const *__restrict two_vdt2,
                              T const *__restrict scatter,
                              int64_t const *__restrict receivers_i,
                              int64_t n_receivers_per_shot) {
  for (int64_t receiver_idx{}; receiver_idx < n_receivers_per_shot;
       ++receiver_idx) {
    auto i{receivers_i[receiver_idx]};
    r[receiver_idx] = -wf[i] * v2dt2[i] - wfsc[i] * two_vdt2[i] * scatter[i];
  }
}

template <typename T, int A>
void add_to_grad_v(T *__restrict grad_v, T const *__restrict wfc,
                   T const *__restrict wfcsc, T const *__restrict w_store,
                   T const *__restrict wsc_store, T const *__restrict two_vdt2,
                   T const *__restrict scatter, T dt2, int64_t step_ratio,
                   int64_t nx, int64_t ny) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      grad_v[i] += wfc[i] * two_vdt2[i] * w_store[i] * step_ratio +
                   wfcsc[i] *
                       (static_cast<T>(2.0) * dt2 * scatter[i] * w_store[i] +
                        two_vdt2[i] * wsc_store[i]) *
                       step_ratio;
    }
  }
}

template <typename T, int A>
void add_to_grad_scatter(T *__restrict grad_scatter, T const *__restrict wfcsc,
                         T const *__restrict w_store,
                         T const *__restrict two_vdt2, int64_t step_ratio,
                         int64_t nx, int64_t ny) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      grad_scatter[i] += wfcsc[i] * two_vdt2[i] * w_store[i] * step_ratio;
    }
  }
}

template <typename T, int A>
void backward_kernel_1(T *__restrict zetaxsc, T *__restrict zetaysc,
                       T *__restrict b_a_zetaxsc, T *__restrict b_a_zetaysc,
                       T *__restrict v2dt2_wfcsc,
                       T *__restrict scatter_v2dt2_wfcsc,
                       T const *__restrict wfcsc, T *__restrict zetax,
                       T *__restrict zetay, T *__restrict b_a_zetax,
                       T *__restrict b_a_zetay, T *__restrict v2dt2_wfc,
                       T const *__restrict wfc, T const *__restrict v2dt2,
                       T const *__restrict two_vdt2,
                       T const *__restrict scatter, T const *__restrict ax,
                       T const *__restrict ay, T const *__restrict bx_over_ax,
                       T const *__restrict by_over_ay, int64_t nx, int64_t ny) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    auto axi{ax[x]};
    auto bx_over_axi{bx_over_ax[x]};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      zetaxsc[i] = axi * v2dt2[i] * wfcsc[i] + axi * zetaxsc[i];
      zetaysc[i] = ay[y] * v2dt2[i] * wfcsc[i] + ay[y] * zetaysc[i];
      b_a_zetaxsc[i] = bx_over_axi * zetaxsc[i];
      b_a_zetaysc[i] = by_over_ay[y] * zetaysc[i];
      v2dt2_wfcsc[i] = v2dt2[i] * wfcsc[i];
      scatter_v2dt2_wfcsc[i] = scatter[i] * two_vdt2[i] * wfcsc[i];
      zetax[i] = axi * v2dt2[i] * wfc[i] + axi * zetax[i] +
                 axi * scatter_v2dt2_wfcsc[i];
      zetay[i] = ay[y] * v2dt2[i] * wfc[i] + ay[y] * zetay[i] +
                 ay[y] * scatter_v2dt2_wfcsc[i];
      b_a_zetax[i] = bx_over_axi * zetax[i];
      b_a_zetay[i] = by_over_ay[y] * zetay[i];
      v2dt2_wfc[i] = v2dt2[i] * wfc[i];
    }
  }
}

template <typename T, int A>
void backward_kernel_2(
    T *__restrict psixsc, T *__restrict psiysc, T *__restrict psix,
    T *__restrict psiy, T one_over_dx, T one_over_dy, int64_t nx, int64_t ny,
    T const *ax, T const *ay, T *__restrict b_a_psixsc,
    T *__restrict b_a_psiysc, T const *__restrict b_a_zetaxsc,
    T const *__restrict b_a_zetaysc, T const *__restrict v2dt2_wfcsc,
    T const *__restrict scatter_v2dt2_wfcsc, T *__restrict b_a_psix,
    T *__restrict b_a_psiy, T const *__restrict b_a_zetax,
    T const *__restrict b_a_zetay, T const *__restrict v2dt2_wfc,
    T const *__restrict bx_over_ax, T const *__restrict by_over_ay) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    auto axi{ax[x]};
    auto bx_over_axi{bx_over_ax[x]};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      psixsc[i] = -axi * diffx<T, A>(b_a_zetaxsc + i, one_over_dx, ny) -
                  axi * diffx<T, A>(v2dt2_wfcsc + i, one_over_dx, ny) +
                  axi * psixsc[i];
      psiysc[i] = -ay[y] * diffy<T, A>(b_a_zetaysc + i, one_over_dy) -
                  ay[y] * diffy<T, A>(v2dt2_wfcsc + i, one_over_dy) +
                  ay[y] * psiysc[i];
      b_a_psixsc[i] = bx_over_axi * psixsc[i];
      b_a_psiysc[i] = by_over_ay[y] * psiysc[i];
      psix[i] = -axi * diffx<T, A>(b_a_zetax + i, one_over_dx, ny) -
                axi * diffx<T, A>(v2dt2_wfc + i, one_over_dx, ny) +
                axi * psix[i] -
                axi * diffx<T, A>(scatter_v2dt2_wfcsc + i, one_over_dx, ny);
      psiy[i] = -ay[y] * diffy<T, A>(b_a_zetay + i, one_over_dy) -
                ay[y] * diffy<T, A>(v2dt2_wfc + i, one_over_dy) +
                ay[y] * psiy[i] -
                ay[y] * diffy<T, A>(scatter_v2dt2_wfcsc + i, one_over_dy);
      b_a_psix[i] = bx_over_axi * psix[i];
      b_a_psiy[i] = by_over_ay[y] * psiy[i];
    }
  }
}

template <typename T, int A>
void backward_kernel_3(
    T *__restrict wfpsc, T *__restrict wfcsc, T *__restrict wfp,
    T *__restrict wfc, T one_over_dx, T one_over_dy, T one_over_dx2,
    T one_over_dy2, int64_t nx, int64_t ny, T const *__restrict b_a_psixsc,
    T const *__restrict b_a_psiysc, T const *__restrict b_a_zetaxsc,
    T const *__restrict b_a_zetaysc, T const *__restrict v2dt2_wfcsc,
    T const *__restrict scatter_v2dt2_wfcsc, T const *__restrict b_a_psix,
    T const *__restrict b_a_psiy, T const *__restrict b_a_zetax,
    T const *__restrict b_a_zetay, T const *__restrict v2dt2_wfc) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      wfpsc[i] = diffx2<T, A>(v2dt2_wfcsc + i, one_over_dx2, ny) +
                 diffy2<T, A>(v2dt2_wfcsc + i, one_over_dy2) + 2 * wfcsc[i] +
                 wfpsc[i] + diffx2<T, A>(b_a_zetaxsc + i, one_over_dx2, ny) +
                 diffy2<T, A>(b_a_zetaysc + i, one_over_dy2) -
                 diffx<T, A>(b_a_psixsc + i, one_over_dx, ny) -
                 diffy<T, A>(b_a_psiysc + i, one_over_dy);
      wfcsc[i] *= -1;
      wfp[i] = diffx2<T, A>(v2dt2_wfc + i, one_over_dx2, ny) +
               diffy2<T, A>(v2dt2_wfc + i, one_over_dy2) + 2 * wfc[i] + wfp[i] +
               diffx2<T, A>(b_a_zetax + i, one_over_dx2, ny) +
               diffy2<T, A>(b_a_zetay + i, one_over_dy2) -
               diffx<T, A>(b_a_psix + i, one_over_dx, ny) -
               diffy<T, A>(b_a_psiy + i, one_over_dy) +
               diffx2<T, A>(scatter_v2dt2_wfcsc + i, one_over_dx2, ny) +
               diffy2<T, A>(scatter_v2dt2_wfcsc + i, one_over_dy2);
      wfc[i] *= -1;
    }
  }
}

template <typename T, int A>
void backward_kernel_1_sc(T *__restrict zetaxsc, T *__restrict zetaysc,
                          T *__restrict b_a_zetaxsc, T *__restrict b_a_zetaysc,
                          T *__restrict v2dt2_wfcsc, T const *__restrict wfcsc,
                          T const *__restrict v2dt2, T const *__restrict ax,
                          T const *__restrict ay,
                          T const *__restrict bx_over_ax,
                          T const *__restrict by_over_ay, int64_t nx,
                          int64_t ny) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    auto axi{ax[x]};
    auto bx_over_axi{bx_over_ax[x]};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      zetaxsc[i] = axi * v2dt2[i] * wfcsc[i] + axi * zetaxsc[i];
      zetaysc[i] = ay[y] * v2dt2[i] * wfcsc[i] + ay[y] * zetaysc[i];
      b_a_zetaxsc[i] = bx_over_axi * zetaxsc[i];
      b_a_zetaysc[i] = by_over_ay[y] * zetaysc[i];
      v2dt2_wfcsc[i] = v2dt2[i] * wfcsc[i];
    }
  }
}

template <typename T, int A>
void backward_kernel_2_sc(
    T *__restrict psixsc, T *__restrict psiysc, T one_over_dx, T one_over_dy,
    int64_t nx, int64_t ny, T const *ax, T const *ay, T *__restrict b_a_psixsc,
    T *__restrict b_a_psiysc, T const *__restrict b_a_zetaxsc,
    T const *__restrict b_a_zetaysc, T const *__restrict v2dt2_wfcsc,
    T const *__restrict bx_over_ax, T const *__restrict by_over_ay) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    auto axi{ax[x]};
    auto bx_over_axi{bx_over_ax[x]};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      psixsc[i] = -axi * diffx<T, A>(b_a_zetaxsc + i, one_over_dx, ny) -
                  axi * diffx<T, A>(v2dt2_wfcsc + i, one_over_dx, ny) +
                  axi * psixsc[i];
      psiysc[i] = -ay[y] * diffy<T, A>(b_a_zetaysc + i, one_over_dy) -
                  ay[y] * diffy<T, A>(v2dt2_wfcsc + i, one_over_dy) +
                  ay[y] * psiysc[i];
      b_a_psixsc[i] = bx_over_axi * psixsc[i];
      b_a_psiysc[i] = by_over_ay[y] * psiysc[i];
    }
  }
}

template <typename T, int A>
void backward_kernel_3_sc(T *__restrict wfpsc, T *__restrict wfcsc,
                          T one_over_dx, T one_over_dy, T one_over_dx2,
                          T one_over_dy2, int64_t nx, int64_t ny,
                          T const *__restrict b_a_psixsc,
                          T const *__restrict b_a_psiysc,
                          T const *__restrict b_a_zetaxsc,
                          T const *__restrict b_a_zetaysc,
                          T const *__restrict v2dt2_wfcsc) {
  constexpr int fd_pad{A / 2};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      wfpsc[i] = diffx2<T, A>(v2dt2_wfcsc + i, one_over_dx2, ny) +
                 diffy2<T, A>(v2dt2_wfcsc + i, one_over_dy2) + 2 * wfcsc[i] +
                 wfpsc[i] + diffx2<T, A>(b_a_zetaxsc + i, one_over_dx2, ny) +
                 diffy2<T, A>(b_a_zetaysc + i, one_over_dy2) -
                 diffx<T, A>(b_a_psixsc + i, one_over_dx, ny) -
                 diffy<T, A>(b_a_psiysc + i, one_over_dy);
      wfcsc[i] *= -1;
    }
  }
}

template <typename T, int A>
void forward_shot(T *wfc, T *wfp, T *psix, T *psiy, T *zetax, T *zetay,
                  T *wfcsc, T *wfpsc, T *psixsc, T *psiysc, T *zetaxsc,
                  T *zetaysc, int64_t const *sources_i,
                  int64_t const *receivers_i, T *w_store, T *wsc_store,
                  T const *v2dt2, T const *two_vdt2, T const *scatter,
                  T const *f, T *r, T const *ax, T const *ay, T const *bx,
                  T const *by, T one_over_dx, T one_over_dy, T one_over_dx2,
                  T one_over_dy2, int64_t n_sources_per_shot,
                  int64_t n_receivers_per_shot, int64_t nx, int64_t ny,
                  int64_t nt, int64_t step_ratio, bool v_requires_grad,
                  bool scatter_requires_grad) {
  constexpr void (*forward_kernel_2s[])(
      T const *__restrict wfc, T *__restrict wfp, T const *__restrict psix,
      T const *__restrict psiy, T *__restrict zetax, T *__restrict zetay,
      T const *__restrict wfcsc, T *__restrict wfpsc,
      T const *__restrict psixsc, T const *__restrict psiysc,
      T *__restrict zetaxsc, T *__restrict zetaysc, T *__restrict w_store,
      T *__restrict wsc_store, T const *__restrict v2dt2,
      T const *__restrict two_vdt2, T const *__restrict scatter,
      T const *__restrict ax, T const *__restrict ay, T const *__restrict bx,
      T const *__restrict by, int64_t nx, int64_t ny, T one_over_dx,
      T one_over_dy, T one_over_dx2, T one_over_dy2){
      forward_kernel_2<T, A, false, false>, forward_kernel_2<T, A, true, false>,
      forward_kernel_2<T, A, false, true>, forward_kernel_2<T, A, true, true>};

  for (int64_t t{}; t < nt; ++t) {
    forward_kernel_1<T, A>(wfc, psix, psiy, wfcsc, psixsc, psiysc, ax, ay, bx,
                           by, nx, ny, one_over_dx, one_over_dy);
    if (t % step_ratio == 0 and (v_requires_grad or scatter_requires_grad)) {
      forward_kernel_2s[v_requires_grad + 2 * scatter_requires_grad](
          wfc, wfp, psix, psiy, zetax, zetay, wfcsc, wfpsc, psixsc, psiysc,
          zetaxsc, zetaysc, w_store + (t / step_ratio) * nx * ny,
          wsc_store + (t / step_ratio) * nx * ny, v2dt2, two_vdt2, scatter, ax,
          ay, bx, by, nx, ny, one_over_dx, one_over_dy, one_over_dx2,
          one_over_dy2);
    } else {
      forward_kernel_2<T, A, false, false>(
          wfc, wfp, psix, psiy, zetax, zetay, wfcsc, wfpsc, psixsc, psiysc,
          zetaxsc, zetaysc, nullptr, nullptr, v2dt2, two_vdt2, scatter, ax, ay,
          bx, by, nx, ny, one_over_dx, one_over_dy, one_over_dx2, one_over_dy2);
    }
    if (n_sources_per_shot > 0) {
      if (t % step_ratio == 0 and (v_requires_grad or scatter_requires_grad)) {
        add_sources<T, true>(wfp, wfpsc, w_store + (t / step_ratio) * nx * ny,
                             f + t * n_sources_per_shot, v2dt2, two_vdt2,
                             scatter, sources_i, n_sources_per_shot);
      } else {
        add_sources<T, false>(wfp, wfpsc, nullptr, f + t * n_sources_per_shot,
                              v2dt2, two_vdt2, scatter, sources_i,
                              n_sources_per_shot);
      }
    }
    if (n_receivers_per_shot > 0) {
      record_receivers(r + t * n_receivers_per_shot, wfcsc, receivers_i,
                       n_receivers_per_shot);
    }
    std::swap(wfp, wfc);
    std::swap(wfpsc, wfcsc);
  }
}

template <typename T, int A>
void combine_grad_model(T *__restrict grad, T const *__restrict grad_batch,
                        int64_t n_parallel, int64_t nx, int64_t ny) {
  constexpr int fd_pad{A / 2};
  int64_t nxny{nx * ny};
  for (int64_t x = fd_pad; x < nx - fd_pad; ++x) {
    int64_t xi{x * ny};
    for (int64_t y = fd_pad; y < ny - fd_pad; ++y) {
      int64_t i{xi + y};
      for (int64_t batch{}; batch < n_parallel; ++batch) {
        grad[i] += grad_batch[batch * nxny + i];
      }
    }
  }
}

template <typename T, int A>
void backward_shot(
    T *wfc, T *wfp, T *psix, T *psiy, T *zetax, T *zetay, T *wfcsc, T *wfpsc,
    T *psixsc, T *psiysc, T *zetaxsc, T *zetaysc, int64_t const *sources_i,
    int64_t const *receivers_i, T *w_store, T *wsc_store, T const *v2dt2,
    T const *two_vdt2, T const *scatter, T *f, T const *r, T *grad_v,
    T *grad_scatter, T const *ax, T const *ay, T const *bx_over_ax,
    T const *by_over_ay, T *b_a_psix, T *b_a_psiy, T *b_a_zetax, T *b_a_zetay,
    T *v2dt2_wfc, T *b_a_psixsc, T *b_a_psiysc, T *b_a_zetaxsc, T *b_a_zetaysc,
    T *v2dt2_wfcsc, T *scatter_v2dt2_wfcsc, T one_over_dx, T one_over_dy,
    T one_over_dx2, T one_over_dy2, T dt2, int64_t n_sources_per_shot,
    int64_t n_receivers_per_shot, int64_t nx, int64_t ny, int64_t nt,
    int64_t step_ratio, bool v_requires_grad, bool scatter_requires_grad) {
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (n_receivers_per_shot > 0) {
      add_adjoint_sources(wfpsc, r + t * n_receivers_per_shot, receivers_i,
                          n_receivers_per_shot);
    }
    if (n_sources_per_shot > 0) {
      record_adjoint_receivers(f + t * n_sources_per_shot, wfc, wfcsc, v2dt2,
                               two_vdt2, scatter, sources_i,
                               n_sources_per_shot);
    }
    if (t % step_ratio == 0 and v_requires_grad) {
      add_to_grad_v<T, A>(grad_v, wfc, wfcsc,
                          w_store + (t / step_ratio) * nx * ny,
                          wsc_store + (t / step_ratio) * nx * ny, two_vdt2,
                          scatter, dt2, step_ratio, nx, ny);
    }
    if (t % step_ratio == 0 and scatter_requires_grad) {
      add_to_grad_scatter<T, A>(grad_scatter, wfcsc,
                                w_store + (t / step_ratio) * nx * ny, two_vdt2,
                                step_ratio, nx, ny);
    }
    backward_kernel_1<T, A>(zetaxsc, zetaysc, b_a_zetaxsc, b_a_zetaysc,
                            v2dt2_wfcsc, scatter_v2dt2_wfcsc, wfcsc, zetax,
                            zetay, b_a_zetax, b_a_zetay, v2dt2_wfc, wfc, v2dt2,
                            two_vdt2, scatter, ax, ay, bx_over_ax, by_over_ay,
                            nx, ny);
    backward_kernel_2<T, A>(psixsc, psiysc, psix, psiy, one_over_dx,
                            one_over_dy, nx, ny, ax, ay, b_a_psixsc, b_a_psiysc,
                            b_a_zetaxsc, b_a_zetaysc, v2dt2_wfcsc,
                            scatter_v2dt2_wfcsc, b_a_psix, b_a_psiy, b_a_zetax,
                            b_a_zetay, v2dt2_wfc, bx_over_ax, by_over_ay);
    backward_kernel_3<T, A>(wfpsc, wfcsc, wfp, wfc, one_over_dx, one_over_dy,
                            one_over_dx2, one_over_dy2, nx, ny, b_a_psixsc,
                            b_a_psiysc, b_a_zetaxsc, b_a_zetaysc, v2dt2_wfcsc,
                            scatter_v2dt2_wfcsc, b_a_psix, b_a_psiy, b_a_zetax,
                            b_a_zetay, v2dt2_wfc);
    std::swap(wfp, wfc);
    std::swap(wfpsc, wfcsc);
  }
}

template <typename T, int A>
void backward_shot_sc(
    T *wfcsc, T *wfpsc, T *psixsc, T *psiysc, T *zetaxsc, T *zetaysc,
    int64_t const *receivers_i, T *w_store, T const *v2dt2, T const *two_vdt2,
    T const *r, T *grad_scatter, T const *ax, T const *ay, T const *bx_over_ax,
    T const *by_over_ay, T *b_a_psixsc, T *b_a_psiysc, T *b_a_zetaxsc,
    T *b_a_zetaysc, T *v2dt2_wfcsc, T one_over_dx, T one_over_dy,
    T one_over_dx2, T one_over_dy2, int64_t n_receivers_per_shot, int64_t nx,
    int64_t ny, int64_t nt, int64_t step_ratio, bool scatter_requires_grad) {
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (n_receivers_per_shot > 0) {
      add_adjoint_sources(wfpsc, r + t * n_receivers_per_shot, receivers_i,
                          n_receivers_per_shot);
    }
    if (t % step_ratio == 0 and scatter_requires_grad) {
      add_to_grad_scatter<T, A>(grad_scatter, wfcsc,
                                w_store + (t / step_ratio) * nx * ny, two_vdt2,
                                step_ratio, nx, ny);
    }
    backward_kernel_1_sc<T, A>(zetaxsc, zetaysc, b_a_zetaxsc, b_a_zetaysc,
                               v2dt2_wfcsc, wfcsc, v2dt2, ax, ay, bx_over_ax,
                               by_over_ay, nx, ny);
    backward_kernel_2_sc<T, A>(psixsc, psiysc, one_over_dx, one_over_dy, nx, ny,
                               ax, ay, b_a_psixsc, b_a_psiysc, b_a_zetaxsc,
                               b_a_zetaysc, v2dt2_wfcsc, bx_over_ax,
                               by_over_ay);
    backward_kernel_3_sc<T, A>(
        wfpsc, wfcsc, one_over_dx, one_over_dy, one_over_dx2, one_over_dy2, nx,
        ny, b_a_psixsc, b_a_psiysc, b_a_zetaxsc, b_a_zetaysc, v2dt2_wfcsc);
    std::swap(wfpsc, wfcsc);
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

class ScalarBornCPUFunction
    : public torch::autograd::Function<ScalarBornCPUFunction> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext *ctx, torch::Tensor const &v,
      torch::Tensor const &scatter, torch::Tensor const &f,
      torch::Tensor const &wfc0, torch::Tensor const &wfp0,
      torch::Tensor const &psix0, torch::Tensor const &psiy0,
      torch::Tensor const &zetax0, torch::Tensor const &zetay0,
      torch::Tensor const &wfcsc0, torch::Tensor const &wfpsc0,
      torch::Tensor const &psixsc0, torch::Tensor const &psiysc0,
      torch::Tensor const &zetaxsc0, torch::Tensor const &zetaysc0,
      torch::Tensor const &ax, torch::Tensor const &ay, torch::Tensor const &bx,
      torch::Tensor const &by, torch::Tensor const &sources_i,
      torch::Tensor const &receivers_i, double dx, double dy, double dt,
      int64_t nt, int64_t n_batch, int64_t step_ratio, int64_t accuracy,
      int64_t pml_width0, int64_t pml_width1, int64_t pml_width2,
      int64_t pml_width3) {
    at::AutoDispatchBelowADInplaceOrView g;
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
    auto wfcsc{create_or_pad(wfcsc0, fd_pad, options, size_with_batch)};
    auto wfpsc{create_or_pad(wfpsc0, fd_pad, options, size_with_batch)};
    auto psixsc{create_or_pad(psixsc0, fd_pad, options, size_with_batch)};
    auto psiysc{create_or_pad(psiysc0, fd_pad, options, size_with_batch)};
    auto zetaxsc{create_or_pad(zetaxsc0, fd_pad, options, size_with_batch)};
    auto zetaysc{create_or_pad(zetaysc0, fd_pad, options, size_with_batch)};
    auto r{at::empty({n_batch, nt, n_receivers_per_shot}, options)};
    torch::Tensor w_store;
    torch::Tensor wsc_store;
    if (v.requires_grad() or scatter.requires_grad()) {
      w_store = at::empty({n_batch, (nt + step_ratio - 1) / step_ratio, nx, ny},
                          options);
    }
    if (v.requires_grad()) {
      wsc_store = at::empty(
          {n_batch, (nt + step_ratio - 1) / step_ratio, nx, ny}, options);
    }

    bool non_sc{v.requires_grad() or f.requires_grad() or
                wfc0.requires_grad() or wfp0.requires_grad() or
                psix0.requires_grad() or psiy0.requires_grad() or
                zetax0.requires_grad() or zetay0.requires_grad()};

    if (non_sc) {
      zero_interior(psix, nx, ny, fd_pad, pml_width);
      zero_interior(psiy, nx, ny, fd_pad, pml_width);
      zero_interior(zetax, nx, ny, fd_pad, pml_width);
      zero_interior(zetay, nx, ny, fd_pad, pml_width);
    }
    zero_interior(psixsc, nx, ny, fd_pad, pml_width);
    zero_interior(psiysc, nx, ny, fd_pad, pml_width);
    zero_interior(zetaxsc, nx, ny, fd_pad, pml_width);
    zero_interior(zetaysc, nx, ny, fd_pad, pml_width);

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_born_cpu_forward", ([&] {
          scalar_t dt2_a = dt * dt;
          scalar_t one_over_dx_a = 1.0 / dx;
          scalar_t one_over_dy_a = 1.0 / dy;
          scalar_t one_over_dx2_a = 1.0 / (dx * dx);
          scalar_t one_over_dy2_a = 1.0 / (dy * dy);
          auto v2dt2{v * v * dt2_a};
          auto v2dt2_a{v2dt2.data_ptr<scalar_t>()};
          auto two_vdt2{static_cast<scalar_t>(2.0) * v * dt2_a};
          auto two_vdt2_a{two_vdt2.data_ptr<scalar_t>()};
          auto scatter_a{scatter.data_ptr<scalar_t>()};
          auto f_a{f.data_ptr<scalar_t>()};
          auto r_a{r.data_ptr<scalar_t>()};
          auto wfc_a{wfc.data_ptr<scalar_t>()};
          auto wfp_a{wfp.data_ptr<scalar_t>()};
          auto psix_a{psix.data_ptr<scalar_t>()};
          auto psiy_a{psiy.data_ptr<scalar_t>()};
          auto zetax_a{zetax.data_ptr<scalar_t>()};
          auto zetay_a{zetay.data_ptr<scalar_t>()};
          auto wfcsc_a{wfcsc.data_ptr<scalar_t>()};
          auto wfpsc_a{wfpsc.data_ptr<scalar_t>()};
          auto psixsc_a{psixsc.data_ptr<scalar_t>()};
          auto psiysc_a{psiysc.data_ptr<scalar_t>()};
          auto zetaxsc_a{zetaxsc.data_ptr<scalar_t>()};
          auto zetaysc_a{zetaysc.data_ptr<scalar_t>()};
          auto ax_a{ax.data_ptr<scalar_t>()};
          auto ay_a{ay.data_ptr<scalar_t>()};
          auto bx_a{bx.data_ptr<scalar_t>()};
          auto by_a{by.data_ptr<scalar_t>()};
          auto sources_i_a{sources_i.data_ptr<int64_t>()};
          auto receivers_i_a{receivers_i.data_ptr<int64_t>()};
          scalar_t *w_store_a{};
          scalar_t *wsc_store_a{};
          if (v.requires_grad() or scatter.requires_grad()) {
            w_store_a = w_store.data_ptr<scalar_t>();
          }
          if (v.requires_grad()) {
            wsc_store_a = wsc_store.data_ptr<scalar_t>();
          }
          constexpr void (*forward_shots[])(
              scalar_t * wfc, scalar_t * wfp, scalar_t * psix, scalar_t * psiy,
              scalar_t * zetax, scalar_t * zetay, scalar_t * wfcsc,
              scalar_t * wfpsc, scalar_t * psixsc, scalar_t * psiysc,
              scalar_t * zetaxsc, scalar_t * zetaysc, int64_t const *sources_i,
              int64_t const *receivers_i, scalar_t *w_store,
              scalar_t *wsc_store, scalar_t const *v2dt2,
              scalar_t const *two_vdt2, scalar_t const *scatter,
              scalar_t const *f, scalar_t *r, scalar_t const *ax,
              scalar_t const *ay, scalar_t const *bx, scalar_t const *by,
              scalar_t one_over_dx, scalar_t one_over_dy, scalar_t one_over_dx2,
              scalar_t one_over_dy2, int64_t n_sources_per_shot,
              int64_t n_receivers_per_shot, int64_t nx, int64_t ny, int64_t nt,
              int64_t step_ratio, bool v_requires_grad,
              bool scatter_requires_grad){
              forward_shot<scalar_t, 2>, forward_shot<scalar_t, 4>,
              forward_shot<scalar_t, 6>, forward_shot<scalar_t, 8>};
          at::parallel_for(0, n_batch, 0, [&](int64_t bstart, int64_t bend) {
            for (int64_t shot = bstart; shot < bend; ++shot) {
              auto i{shot * nx * ny};
              auto si{shot * n_sources_per_shot};
              auto ri{shot * n_receivers_per_shot};
              forward_shots[accuracy / 2 - 1](
                  wfc_a + i, wfp_a + i, psix_a + i, psiy_a + i, zetax_a + i,
                  zetay_a + i, wfcsc_a + i, wfpsc_a + i, psixsc_a + i,
                  psiysc_a + i, zetaxsc_a + i, zetaysc_a + i, sources_i_a + si,
                  receivers_i_a + ri, w_store_a + i * (nt / step_ratio),
                  wsc_store_a + i * (nt / step_ratio), v2dt2_a, two_vdt2_a,
                  scatter_a, f_a + si * nt, r_a + ri * nt, ax_a, ay_a, bx_a,
                  by_a, one_over_dx_a, one_over_dy_a, one_over_dx2_a,
                  one_over_dy2_a, n_sources_per_shot, n_receivers_per_shot, nx,
                  ny, nt, step_ratio, v.requires_grad(),
                  scatter.requires_grad());
            }
          });
        }));
    if (v.requires_grad() or scatter.requires_grad() or f.requires_grad() or
        wfc0.requires_grad() or wfp0.requires_grad() or psix0.requires_grad() or
        psiy0.requires_grad() or zetax0.requires_grad() or
        zetay0.requires_grad() or wfcsc0.requires_grad() or
        wfpsc0.requires_grad() or psixsc0.requires_grad() or
        psiysc0.requires_grad() or zetaxsc0.requires_grad() or
        zetaysc0.requires_grad()) {
      ctx->save_for_backward(
          {v, scatter, ax, ay, bx, by, sources_i, receivers_i});
      ctx->saved_data["w_store"] = w_store;
      ctx->saved_data["wsc_store"] = wsc_store;
      ctx->saved_data["dx"] = dx;
      ctx->saved_data["dy"] = dy;
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
    auto all_slice{torch::indexing::Slice()};
    auto slicex{torch::indexing::Slice(fd_pad, nx - fd_pad)};
    auto slicey{torch::indexing::Slice(fd_pad, ny - fd_pad)};
    if (nt & 1) {
      return {wfp.index({all_slice, slicex, slicey}),
              wfc.index({all_slice, slicex, slicey}),
              psix.index({all_slice, slicex, slicey}),
              psiy.index({all_slice, slicex, slicey}),
              zetax.index({all_slice, slicex, slicey}),
              zetay.index({all_slice, slicex, slicey}),
              wfpsc.index({all_slice, slicex, slicey}),
              wfcsc.index({all_slice, slicex, slicey}),
              psixsc.index({all_slice, slicex, slicey}),
              psiysc.index({all_slice, slicex, slicey}),
              zetaxsc.index({all_slice, slicex, slicey}),
              zetaysc.index({all_slice, slicex, slicey}),
              r};
    }
    return {wfc.index({all_slice, slicex, slicey}),
            wfp.index({all_slice, slicex, slicey}),
            psix.index({all_slice, slicex, slicey}),
            psiy.index({all_slice, slicex, slicey}),
            zetax.index({all_slice, slicex, slicey}),
            zetay.index({all_slice, slicex, slicey}),
            wfcsc.index({all_slice, slicex, slicey}),
            wfpsc.index({all_slice, slicex, slicey}),
            psixsc.index({all_slice, slicex, slicey}),
            psiysc.index({all_slice, slicex, slicey}),
            zetaxsc.index({all_slice, slicex, slicey}),
            zetaysc.index({all_slice, slicex, slicey}),
            r};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list const &grad_outputs) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto saved{ctx->get_saved_variables()};
    auto const &v{saved[0]};
    auto const &scatter{saved[1]};
    auto const &ax{saved[2]};
    auto const &ay{saved[3]};
    auto const &bx{saved[4]};
    auto const &by{saved[5]};
    auto const &sources_i{saved[6]};
    auto const &receivers_i{saved[7]};
    auto const &w_store{ctx->saved_data["w_store"].toTensor()};
    auto const &wsc_store{ctx->saved_data["wsc_store"].toTensor()};
    auto dx{ctx->saved_data["dx"].toDouble()};
    auto dy{ctx->saved_data["dy"].toDouble()};
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
    auto nx{v.size(0)};
    auto ny{v.size(1)};
    auto wfc = non_sc ? at::constant_pad_nd(grad_outputs[0],
                                            {fd_pad, fd_pad, fd_pad, fd_pad})
                      : at::empty(0, options);
    auto wfp = non_sc ? at::constant_pad_nd(grad_outputs[1],
                                            {fd_pad, fd_pad, fd_pad, fd_pad})
                      : at::empty(0, options);
    auto psix = non_sc ? at::constant_pad_nd(grad_outputs[2],
                                             {fd_pad, fd_pad, fd_pad, fd_pad})
                       : at::empty(0, options);
    auto psiy = non_sc ? at::constant_pad_nd(grad_outputs[3],
                                             {fd_pad, fd_pad, fd_pad, fd_pad})
                       : at::empty(0, options);
    auto zetax = non_sc ? at::constant_pad_nd(grad_outputs[4],
                                              {fd_pad, fd_pad, fd_pad, fd_pad})
                        : at::empty(0, options);
    auto zetay = non_sc ? at::constant_pad_nd(grad_outputs[5],
                                              {fd_pad, fd_pad, fd_pad, fd_pad})
                        : at::empty(0, options);
    auto wfcsc =
        at::constant_pad_nd(grad_outputs[6], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto wfpsc =
        at::constant_pad_nd(grad_outputs[7], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psixsc =
        at::constant_pad_nd(grad_outputs[8], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto psiysc =
        at::constant_pad_nd(grad_outputs[9], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetaxsc =
        at::constant_pad_nd(grad_outputs[10], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto zetaysc =
        at::constant_pad_nd(grad_outputs[11], {fd_pad, fd_pad, fd_pad, fd_pad});
    auto grad_r{grad_outputs[12].contiguous()};
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
    auto grad_v{non_sc ? at::zeros_like(v) : at::empty(0, options)};
    auto grad_v_batch{non_sc and n_parallel > 1
                          ? at::zeros({n_parallel, nx, ny}, options)
                          : at::empty(0, options)};
    auto grad_scatter{at::zeros_like(scatter)};
    auto grad_scatter_batch{n_parallel > 1
                                ? at::zeros({n_parallel, nx, ny}, options)
                                : at::empty(0)};
    auto grad_f{non_sc ? at::empty({n_batch, nt, n_sources_per_shot}, options)
                       : at::empty(0, options)};

    auto bx_over_ax{bx / ax};
    auto by_over_ay{by / ay};
    auto b_a_psix{non_sc ? torch::zeros_like(psix) : at::empty(0, options)};
    auto b_a_psiy{non_sc ? torch::zeros_like(psiy) : at::empty(0, options)};
    auto b_a_zetax{non_sc ? torch::zeros_like(zetax) : at::empty(0, options)};
    auto b_a_zetay{non_sc ? torch::zeros_like(zetay) : at::empty(0, options)};
    auto v2dt2_wfc{non_sc ? torch::zeros_like(wfc) : at::empty(0, options)};
    auto b_a_psixsc{torch::zeros_like(psixsc)};
    auto b_a_psiysc{torch::zeros_like(psiysc)};
    auto b_a_zetaxsc{torch::zeros_like(zetaxsc)};
    auto b_a_zetaysc{torch::zeros_like(zetaysc)};
    auto v2dt2_wfcsc{torch::zeros_like(wfcsc)};
    auto scatter_v2dt2_wfcsc{non_sc ? torch::zeros_like(wfcsc)
                                    : at::empty(0, options)};

    AT_DISPATCH_FLOATING_TYPES(
        v.scalar_type(), "scalar_cpu_backward", ([&] {
          scalar_t dt2_a = dt * dt;
          scalar_t one_over_dx_a = 1.0 / dx;
          scalar_t one_over_dy_a = 1.0 / dy;
          scalar_t one_over_dx2_a = 1.0 / (dx * dx);
          scalar_t one_over_dy2_a = 1.0 / (dy * dy);
          auto v2dt2{v * v * dt2_a};
          auto v2dt2_a{v2dt2.data_ptr<scalar_t>()};
          auto two_vdt2{static_cast<scalar_t>(2.0) * v * dt2_a};
          auto two_vdt2_a{two_vdt2.data_ptr<scalar_t>()};
          auto scatter_a{scatter.data_ptr<scalar_t>()};
          auto grad_v_a{grad_v.data_ptr<scalar_t>()};
          auto grad_v_batch_a{n_parallel > 1 ? grad_v_batch.data_ptr<scalar_t>()
                                             : grad_v_a};
          auto grad_scatter_a{grad_scatter.data_ptr<scalar_t>()};
          auto grad_scatter_batch_a{
              n_parallel > 1 ? grad_scatter_batch.data_ptr<scalar_t>()
                             : grad_scatter_a};
          auto grad_r_a{grad_r.data_ptr<scalar_t>()};
          auto grad_f_a{grad_f.data_ptr<scalar_t>()};
          auto wfc_a{wfc.data_ptr<scalar_t>()};
          auto wfp_a{wfp.data_ptr<scalar_t>()};
          auto psix_a{psix.data_ptr<scalar_t>()};
          auto psiy_a{psiy.data_ptr<scalar_t>()};
          auto zetax_a{zetax.data_ptr<scalar_t>()};
          auto zetay_a{zetay.data_ptr<scalar_t>()};
          auto wfcsc_a{wfcsc.data_ptr<scalar_t>()};
          auto wfpsc_a{wfpsc.data_ptr<scalar_t>()};
          auto psixsc_a{psixsc.data_ptr<scalar_t>()};
          auto psiysc_a{psiysc.data_ptr<scalar_t>()};
          auto zetaxsc_a{zetaxsc.data_ptr<scalar_t>()};
          auto zetaysc_a{zetaysc.data_ptr<scalar_t>()};
          auto ax_a{ax.data_ptr<scalar_t>()};
          auto ay_a{ay.data_ptr<scalar_t>()};
          auto bx_over_ax_a{bx_over_ax.data_ptr<scalar_t>()};
          auto by_over_ay_a{by_over_ay.data_ptr<scalar_t>()};
          auto b_a_psix_a{b_a_psix.data_ptr<scalar_t>()};
          auto b_a_psiy_a{b_a_psiy.data_ptr<scalar_t>()};
          auto b_a_zetax_a{b_a_zetax.data_ptr<scalar_t>()};
          auto b_a_zetay_a{b_a_zetay.data_ptr<scalar_t>()};
          auto b_a_psixsc_a{b_a_psixsc.data_ptr<scalar_t>()};
          auto b_a_psiysc_a{b_a_psiysc.data_ptr<scalar_t>()};
          auto b_a_zetaxsc_a{b_a_zetaxsc.data_ptr<scalar_t>()};
          auto b_a_zetaysc_a{b_a_zetaysc.data_ptr<scalar_t>()};
          auto sources_i_a{sources_i.data_ptr<int64_t>()};
          auto receivers_i_a{receivers_i.data_ptr<int64_t>()};
          auto v2dt2_wfc_a{v2dt2_wfc.data_ptr<scalar_t>()};
          auto v2dt2_wfcsc_a{v2dt2_wfcsc.data_ptr<scalar_t>()};
          auto scatter_v2dt2_wfcsc_a{scatter_v2dt2_wfcsc.data_ptr<scalar_t>()};
          scalar_t *w_store_a{};
          scalar_t *wsc_store_a{};
          if (v.requires_grad() or scatter.requires_grad()) {
            w_store_a = w_store.data_ptr<scalar_t>();
          }
          if (v.requires_grad()) {
            wsc_store_a = wsc_store.data_ptr<scalar_t>();
          }

          constexpr void (*backward_shots[])(
              scalar_t * wfc, scalar_t * wfp, scalar_t * psix, scalar_t * psiy,
              scalar_t * zetax, scalar_t * zetay, scalar_t * wfcsc,
              scalar_t * wfpsc, scalar_t * psixsc, scalar_t * psiysc,
              scalar_t * zetaxsc, scalar_t * zetaysc, int64_t const *sources_i,
              int64_t const *receivers_i, scalar_t *w_store,
              scalar_t *wsc_store, scalar_t const *v2dt2,
              scalar_t const *two_vdt2, scalar_t const *scatter, scalar_t *f,
              scalar_t const *r, scalar_t *grad_v, scalar_t *grad_scatter,
              scalar_t const *ax, scalar_t const *ay,
              scalar_t const *bx_over_ax, scalar_t const *by_over_ay,
              scalar_t *b_a_psix, scalar_t *b_a_psiy, scalar_t *b_a_zetax,
              scalar_t *b_a_zetay, scalar_t *v2dt2_wfc, scalar_t *b_a_psixsc,
              scalar_t *b_a_psiysc, scalar_t *b_a_zetaxsc,
              scalar_t *b_a_zetaysc, scalar_t *v2dt2_wfcsc,
              scalar_t *scatter_v2dt2_wfcsc, scalar_t one_over_dx,
              scalar_t one_over_dy, scalar_t one_over_dx2,
              scalar_t one_over_dy2, scalar_t dt2, int64_t n_sources_per_shot,
              int64_t n_receivers_per_shot, int64_t nx, int64_t ny, int64_t nt,
              int64_t step_ratio, bool v_requires_grad,
              bool scatter_requires_grad){
              backward_shot<scalar_t, 2>, backward_shot<scalar_t, 4>,
              backward_shot<scalar_t, 6>, backward_shot<scalar_t, 8>};
          constexpr void (*backward_shot_scs[])(
              scalar_t * wfcsc, scalar_t * wfpsc, scalar_t * psixsc,
              scalar_t * psiysc, scalar_t * zetaxsc, scalar_t * zetaysc,
              int64_t const *receivers_i, scalar_t *w_store,
              scalar_t const *v2dt2, scalar_t const *two_vdt2,
              scalar_t const *r, scalar_t *grad_scatter, scalar_t const *ax,
              scalar_t const *ay, scalar_t const *bx_over_ax,
              scalar_t const *by_over_ay, scalar_t *b_a_psixsc,
              scalar_t *b_a_psiysc, scalar_t *b_a_zetaxsc,
              scalar_t *b_a_zetaysc, scalar_t *v2dt2_wfcsc,
              scalar_t one_over_dx, scalar_t one_over_dy, scalar_t one_over_dx2,
              scalar_t one_over_dy2, int64_t n_receivers_per_shot, int64_t nx,
              int64_t ny, int64_t nt, int64_t step_ratio,
              bool scatter_requires_grad){
              backward_shot_sc<scalar_t, 2>, backward_shot_sc<scalar_t, 4>,
              backward_shot_sc<scalar_t, 6>, backward_shot_sc<scalar_t, 8>};
          at::parallel_for(0, n_batch, 0, [&](int64_t bstart, int64_t bend) {
            if (non_sc) {
              for (int64_t shot = bstart; shot < bend; ++shot) {
                auto i{shot * nx * ny};
                auto si{shot * n_sources_per_shot};
                auto ri{shot * n_receivers_per_shot};
                backward_shots[accuracy / 2 - 1](
                    wfc_a + i, wfp_a + i, psix_a + i, psiy_a + i, zetax_a + i,
                    zetay_a + i, wfcsc_a + i, wfpsc_a + i, psixsc_a + i,
                    psiysc_a + i, zetaxsc_a + i, zetaysc_a + i,
                    sources_i_a + si, receivers_i_a + ri,
                    w_store_a + i * (nt / step_ratio),
                    wsc_store_a + i * (nt / step_ratio), v2dt2_a, two_vdt2_a,
                    scatter_a, grad_f_a + si * nt, grad_r_a + ri * nt,
                    grad_v_batch_a + at::get_thread_num() * nx * ny,
                    grad_scatter_batch_a + at::get_thread_num() * nx * ny, ax_a,
                    ay_a, bx_over_ax_a, by_over_ay_a, b_a_psix_a + i,
                    b_a_psiy_a + i, b_a_zetax_a + i, b_a_zetay_a + i,
                    v2dt2_wfc_a + i, b_a_psixsc_a + i, b_a_psiysc_a + i,
                    b_a_zetaxsc_a + i, b_a_zetaysc_a + i, v2dt2_wfcsc_a + i,
                    scatter_v2dt2_wfcsc_a + i, one_over_dx_a, one_over_dy_a,
                    one_over_dx2_a, one_over_dy2_a, dt2_a, n_sources_per_shot,
                    n_receivers_per_shot, nx, ny, nt, step_ratio,
                    v.requires_grad(), scatter.requires_grad());
              }
            } else {
              for (int64_t shot = bstart; shot < bend; ++shot) {
                auto i{shot * nx * ny};
                auto ri{shot * n_receivers_per_shot};
                backward_shot_scs[accuracy / 2 - 1](
                    wfcsc_a + i, wfpsc_a + i, psixsc_a + i, psiysc_a + i,
                    zetaxsc_a + i, zetaysc_a + i, receivers_i_a + ri,
                    w_store_a + i * (nt / step_ratio), v2dt2_a, two_vdt2_a,
                    grad_r_a + ri * nt,
                    grad_scatter_batch_a + at::get_thread_num() * nx * ny, ax_a,
                    ay_a, bx_over_ax_a, by_over_ay_a, b_a_psixsc_a + i,
                    b_a_psiysc_a + i, b_a_zetaxsc_a + i, b_a_zetaysc_a + i,
                    v2dt2_wfcsc_a + i, one_over_dx_a, one_over_dy_a,
                    one_over_dx2_a, one_over_dy2_a, n_receivers_per_shot, nx,
                    ny, nt, step_ratio, scatter.requires_grad());
              }
            }
          });
          if (v.requires_grad() and n_parallel > 1) {
            if (accuracy == 2) {
              combine_grad_model<scalar_t, 2>(grad_v_a, grad_v_batch_a,
                                              n_parallel, nx, ny);
            } else if (accuracy == 4) {
              combine_grad_model<scalar_t, 4>(grad_v_a, grad_v_batch_a,
                                              n_parallel, nx, ny);
            } else if (accuracy == 6) {
              combine_grad_model<scalar_t, 6>(grad_v_a, grad_v_batch_a,
                                              n_parallel, nx, ny);
            } else {
              combine_grad_model<scalar_t, 8>(grad_v_a, grad_v_batch_a,
                                              n_parallel, nx, ny);
            }
          }
          if (scatter.requires_grad() and n_parallel > 1) {
            if (accuracy == 2) {
              combine_grad_model<scalar_t, 2>(
                  grad_scatter_a, grad_scatter_batch_a, n_parallel, nx, ny);
            } else if (accuracy == 4) {
              combine_grad_model<scalar_t, 4>(
                  grad_scatter_a, grad_scatter_batch_a, n_parallel, nx, ny);
            } else if (accuracy == 6) {
              combine_grad_model<scalar_t, 6>(
                  grad_scatter_a, grad_scatter_batch_a, n_parallel, nx, ny);
            } else {
              combine_grad_model<scalar_t, 8>(
                  grad_scatter_a, grad_scatter_batch_a, n_parallel, nx, ny);
            }
          }
        }));

    if (non_sc) {
      zero_interior(psix, nx, ny, fd_pad, pml_width);
      zero_interior(psiy, nx, ny, fd_pad, pml_width);
      zero_interior(zetax, nx, ny, fd_pad, pml_width);
      zero_interior(zetay, nx, ny, fd_pad, pml_width);
    }
    zero_interior(psixsc, nx, ny, fd_pad, pml_width);
    zero_interior(psiysc, nx, ny, fd_pad, pml_width);
    zero_interior(zetaxsc, nx, ny, fd_pad, pml_width);
    zero_interior(zetaysc, nx, ny, fd_pad, pml_width);

    auto all_slice{torch::indexing::Slice()};
    auto slicex{torch::indexing::Slice(fd_pad, nx - fd_pad)};
    auto slicey{torch::indexing::Slice(fd_pad, ny - fd_pad)};
    if (nt & 1) {
      return {
          non_sc ? grad_v : torch::Tensor(),
          grad_scatter,
          non_sc ? grad_f : torch::Tensor(),
          non_sc ? wfp.index({all_slice, slicex, slicey}) : torch::Tensor(),
          non_sc ? wfc.index({all_slice, slicex, slicey}) : torch::Tensor(),
          non_sc ? psix.index({all_slice, slicex, slicey}) : torch::Tensor(),
          non_sc ? psiy.index({all_slice, slicex, slicey}) : torch::Tensor(),
          non_sc ? zetax.index({all_slice, slicex, slicey}) : torch::Tensor(),
          non_sc ? zetay.index({all_slice, slicex, slicey}) : torch::Tensor(),
          wfpsc.index({all_slice, slicex, slicey}),
          wfcsc.index({all_slice, slicex, slicey}),
          psixsc.index({all_slice, slicex, slicey}),
          psiysc.index({all_slice, slicex, slicey}),
          zetaxsc.index({all_slice, slicex, slicey}),
          zetaysc.index({all_slice, slicex, slicey}),
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
    return {non_sc ? grad_v : torch::Tensor(),
            grad_scatter,
            non_sc ? grad_f : torch::Tensor(),
            non_sc ? wfc.index({all_slice, slicex, slicey}) : torch::Tensor(),
            non_sc ? wfp.index({all_slice, slicex, slicey}) : torch::Tensor(),
            non_sc ? psix.index({all_slice, slicex, slicey}) : torch::Tensor(),
            non_sc ? psiy.index({all_slice, slicex, slicey}) : torch::Tensor(),
            non_sc ? zetax.index({all_slice, slicex, slicey}) : torch::Tensor(),
            non_sc ? zetay.index({all_slice, slicex, slicey}) : torch::Tensor(),
            wfcsc.index({all_slice, slicex, slicey}),
            wfpsc.index({all_slice, slicex, slicey}),
            psixsc.index({all_slice, slicex, slicey}),
            psiysc.index({all_slice, slicex, slicey}),
            zetaxsc.index({all_slice, slicex, slicey}),
            zetaysc.index({all_slice, slicex, slicey}),
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

std::vector<torch::Tensor> scalar_born_cpu_autograd(
    torch::Tensor const &v, torch::Tensor const &scatter,
    torch::Tensor const &f, torch::Tensor const &wfc0,
    torch::Tensor const &wfp0, torch::Tensor const &psix0,
    torch::Tensor const &psiy0, torch::Tensor const &zetax0,
    torch::Tensor const &zetay0, torch::Tensor const &wfcsc0,
    torch::Tensor const &wfpsc0, torch::Tensor const &psixsc0,
    torch::Tensor const &psiysc0, torch::Tensor const &zetaxsc0,
    torch::Tensor const &zetaysc0, torch::Tensor const &ax,
    torch::Tensor const &ay, torch::Tensor const &bx, torch::Tensor const &by,
    torch::Tensor const &sources_i, torch::Tensor const &receivers_i, double dx,
    double dy, double dt, int64_t nt, int64_t n_batch, int64_t step_ratio,
    int64_t accuracy, int64_t pml_width0, int64_t pml_width1,
    int64_t pml_width2, int64_t pml_width3) {
  return ScalarBornCPUFunction::apply(
      v, scatter, f, wfc0, wfp0, psix0, psiy0, zetax0, zetay0, wfcsc0, wfpsc0,
      psixsc0, psiysc0, zetaxsc0, zetaysc0, ax, ay, bx, by, sources_i,
      receivers_i, dx, dy, dt, nt, n_batch, step_ratio, accuracy, pml_width0,
      pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCPU, m) {
  m.impl("scalar_born", scalar_born_cpu_autograd);
}

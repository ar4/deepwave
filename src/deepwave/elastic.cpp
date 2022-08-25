#include <torch/script.h>
#include <torch/torch.h>

TORCH_LIBRARY_FRAGMENT(deepwave, m) {
  m.def(
      "elastic(Tensor lamb, Tensor mu, Tensor buoyancy, Tensor f_y, Tensor "
      "f_x, "
      "Tensor vy0, Tensor vx0, Tensor sigmayy0, "
      "Tensor sigmaxy0, Tensor sigmaxx0, Tensor m_vyy0, Tensor m_vyx0, Tensor "
      "m_vxy0, Tensor m_vxx0, Tensor m_sigmayyy0, Tensor m_sigmaxyy0, Tensor "
      "m_sigmaxyx0, Tensor m_sigmaxxx0, Tensor ay, Tensor ayh, Tensor ax, "
      "Tensor axh, "
      "Tensor by, Tensor byh, Tensor bx, Tensor bxh, Tensor sources_y_i, "
      "Tensor sources_x_i, Tensor receivers_y_i, Tensor receivers_x_i, float "
      "dy, "
      "float dx, float dt, int nt, int n_batch, int step_ratio, int accuracy, "
      "int pml_width0, int pml_width1, int pml_width2, int pml_width3) "
      "-> Tensor[]");
}

std::vector<torch::Tensor> elastic(
    torch::Tensor const &lamb, torch::Tensor const &mu,
    torch::Tensor const &buoyancy, torch::Tensor const &f_y,
    torch::Tensor const &f_x, torch::Tensor const &vy0,
    torch::Tensor const &vx0, torch::Tensor const &sigmayy0,
    torch::Tensor const &sigmaxy0, torch::Tensor const &sigmaxx0,
    torch::Tensor const &m_vyy0, torch::Tensor const &m_vyx0,
    torch::Tensor const &m_vxy0, torch::Tensor const &m_vxx0,
    torch::Tensor const &m_sigmayyy0, torch::Tensor const &m_sigmaxyy0,
    torch::Tensor const &m_sigmaxyx0, torch::Tensor const &m_sigmaxxx0,
    torch::Tensor const &ay, torch::Tensor const &ayh, torch::Tensor const &ax,
    torch::Tensor const &axh, torch::Tensor const &by, torch::Tensor const &byh,
    torch::Tensor const &bx, torch::Tensor const &bxh,
    torch::Tensor const &sources_y_i, torch::Tensor const &sources_x_i,
    torch::Tensor const &receivers_y_i, torch::Tensor const &receivers_x_i,
    double dy, double dx, double dt, int64_t nt, int64_t n_batch,
    int64_t step_ratio, int64_t accuracy, int64_t pml_width0,
    int64_t pml_width1, int64_t pml_width2, int64_t pml_width3) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("deepwave::elastic", "")
                       .typed<decltype(elastic)>();
  return op.call(lamb, mu, buoyancy, f_y, f_x, vy0, vx0, sigmayy0, sigmaxy0,
                 sigmaxx0, m_vyy0, m_vyx0, m_vxy0, m_vxx0, m_sigmayyy0,
                 m_sigmaxyy0, m_sigmaxyx0, m_sigmaxxx0, ay, ayh, ax, axh, by,
                 byh, bx, bxh, sources_y_i, sources_x_i, receivers_y_i,
                 receivers_x_i, dy, dx, dt, nt, n_batch, step_ratio, accuracy,
                 pml_width0, pml_width1, pml_width2, pml_width3);
}

namespace {

torch::Tensor create_or_clone(torch::Tensor const &tensor,
                              at::TensorOptions const &options,
                              std::array<int64_t, 3> const &size) {
  if (tensor.numel() == 0) {
    return at::zeros(size, options);
  } else {
    return at::clone(tensor);
  }
}

template <typename T, int A, bool buoyancy_requires_grad, int pml_y, int pml_x>
inline void forward_kernel_v(
    T *__restrict vy, T *__restrict vx, T const *__restrict sigmayy,
    T const *__restrict sigmaxy, T const *__restrict sigmaxx,
    T *__restrict m_sigmayyy, T *__restrict m_sigmaxyy,
    T *__restrict m_sigmaxyx, T *__restrict m_sigmaxxx,
    T *__restrict dvydbuoyancy, T *__restrict dvxdbuoyancy,
    T const *__restrict buoyancy, T const *__restrict ay,
    T const *__restrict ayh, T const *__restrict ax, T const *__restrict axh,
    T const *__restrict by, T const *__restrict byh, T const *__restrict bx,
    T const *__restrict bxh, int64_t ny, int64_t nx, T dt,
    T const *__restrict fd_coeffsy, T const fd_coeffs1y[][5],
    T const *__restrict fd_coeffs2y, T const *__restrict fd_coeffsx,
    T const fd_coeffs1x[][5], T const *__restrict fd_coeffs2x,
    int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
  int64_t ybegin_y;
  int64_t yend_y;
  int64_t xbegin_y;
  int64_t xend_y;
  int64_t ybegin_x;
  int64_t yend_x;
  int64_t xbegin_x;
  int64_t xend_x;

  if (pml_y == 0) {
    ybegin_y = 0;
    yend_y = pml_regionsy[0];
    ybegin_x = 1;
    yend_x = pml_regionsy[0] + 1;
  } else if (pml_y == 2) {
    ybegin_y = pml_regionsy[1];
    yend_y = ny;
    ybegin_x = pml_regionsy[1];
    yend_x = ny;
  } else {
    ybegin_y = pml_regionsy[0];
    yend_y = pml_regionsy[1];
    ybegin_x = pml_regionsy[0] + 1;
    yend_x = pml_regionsy[1];
  }
  if (pml_x == 0) {
    xbegin_y = 0;
    xend_y = pml_regionsx[0];
    xbegin_x = 0;
    xend_x = pml_regionsx[0];
  } else if (pml_x == 2) {
    xbegin_y = pml_regionsx[1] - 1;
    xend_y = nx - 1;
    xbegin_x = pml_regionsx[1];
    xend_x = nx;
  } else {
    xbegin_y = pml_regionsx[0];
    xend_y = pml_regionsx[1] - 1;
    xbegin_x = pml_regionsx[0];
    xend_x = pml_regionsx[1];
  }

  for (int64_t y = ybegin_y; y < yend_y; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_y; x < xend_y; ++x) {
      int64_t i{yi + x};
      T dsigmayydy{};
      T dsigmaxydx{};

      // dsigmaxydx
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_x == 0 and x == j) {
          for (int k{}; k < A; ++k) {
            dsigmaxydx += fd_coeffs2x[1 + k] * sigmaxy[i - j + 1 + k];
          }
        } else if (pml_x == 2 and x == nx - 2 - j) {
          for (int k{}; k < A; ++k) {
            dsigmaxydx -= fd_coeffs2x[1 + k] * sigmaxy[i + j - k];
          }
        }
      }
      if (pml_x == 1 or (x > A / 2 - 2 and x < nx - 2 - A / 2 + 2)) {
        for (int k{}; k < A / 2; ++k) {
          dsigmaxydx += fd_coeffsx[k] * (sigmaxy[i + 1 + k] - sigmaxy[i - k]);
        }
      }

      // dsigmayydy
      for (int j{}; j < A / 2; ++j) {
        if (pml_y == 0 and y == j) {
          for (int k{}; k < A; ++k) {
            dsigmayydy += fd_coeffs1y[j][1 + k] * sigmayy[i + (1 - j + k) * nx];
          }
        } else if (pml_y == 2 and y == ny - 1 - j) {
          for (int k{}; k < A; ++k) {
            dsigmayydy -= fd_coeffs1y[j][1 + k] * sigmayy[i + (j - k) * nx];
          }
        }
      }
      if (pml_y == 1 or (y > A / 2 - 1 and y < ny - 1 - A / 2 + 1)) {
        for (int k{}; k < A / 2; ++k) {
          dsigmayydy +=
              fd_coeffsy[k] * (sigmayy[i + (1 + k) * nx] - sigmayy[i - k * nx]);
        }
      }

      if (pml_y == 0 or pml_y == 2) {
        m_sigmayyy[i] = ayh[y] * m_sigmayyy[i] + byh[y] * dsigmayydy;
        dsigmayydy += m_sigmayyy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_sigmaxyx[i] = axh[x] * m_sigmaxyx[i] + bxh[x] * dsigmaxydx;
        dsigmaxydx += m_sigmaxyx[i];
      }
      T buoyancyyhxh;
      if (pml_y == 2 and y == ny - 1) {
        buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1]) / 2;
      } else {
        buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1] + buoyancy[i + nx] +
                        buoyancy[i + nx + 1]) /
                       4;
      }
      vy[i] += buoyancyyhxh * dt * (dsigmayydy + dsigmaxydx);
      if (buoyancy_requires_grad) {
        dvydbuoyancy[i] = dt * (dsigmayydy + dsigmaxydx);
      }
    }
  }

  for (int64_t y = ybegin_x; y < yend_x; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_x; x < xend_x; ++x) {
      int64_t i{yi + x};
      T dsigmaxydy{};
      T dsigmaxxdx{};

      // dsigmaxydy
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_y == 0 and y == 1 + j) {
          for (int k{}; k < A; ++k) {
            dsigmaxydy += fd_coeffs2y[1 + k] * sigmaxy[i + (-j + k) * nx];
          }
        } else if (pml_y == 2 and y == ny - 1 - j) {
          for (int k{}; k < A; ++k) {
            dsigmaxydy -= fd_coeffs2y[1 + k] * sigmaxy[i + (j - k - 1) * nx];
          }
        }
      }
      if (pml_y == 1 or (y > 1 + A / 2 - 2 and y < ny - 1 - A / 2 + 2)) {
        for (int k{}; k < A / 2; ++k) {
          dsigmaxydy +=
              fd_coeffsy[k] * (sigmaxy[i + k * nx] - sigmaxy[i - (k + 1) * nx]);
        }
      }

      // dsigmaxxdx
      for (int j{}; j < A / 2; ++j) {
        if (pml_x == 0 and x == j) {
          for (int k{}; k < A; ++k) {
            dsigmaxxdx += fd_coeffs1x[j][1 + k] * sigmaxx[i - j + k];
          }
        } else if (pml_x == 2 and x == nx - 1 - j) {
          for (int k{}; k < A; ++k) {
            dsigmaxxdx -= fd_coeffs1x[j][1 + k] * sigmaxx[i + (j - k - 1)];
          }
        }
      }
      if (pml_x == 1 or (x > A / 2 - 1 and x < nx - 1 - A / 2 + 1)) {
        for (int k{}; k < A / 2; ++k) {
          dsigmaxxdx += fd_coeffsx[k] * (sigmaxx[i + k] - sigmaxx[i - (1 + k)]);
        }
      }

      if (pml_y == 0 or pml_y == 2) {
        m_sigmaxyy[i] = ay[y] * m_sigmaxyy[i] + by[y] * dsigmaxydy;
        dsigmaxydy += m_sigmaxyy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_sigmaxxx[i] = ax[x] * m_sigmaxxx[i] + bx[x] * dsigmaxxdx;
        dsigmaxxdx += m_sigmaxxx[i];
      }
      vx[i] += buoyancy[i] * dt * (dsigmaxxdx + dsigmaxydy);
      if (buoyancy_requires_grad) {
        dvxdbuoyancy[i] = dt * (dsigmaxxdx + dsigmaxydy);
      }
    }
  }
}

template <typename T, int A, bool lamb_requires_grad, bool mu_requires_grad,
          int pml_y, int pml_x>
inline void forward_kernel_sigma(
    T const *__restrict vy, T const *__restrict vx, T *__restrict sigmayy,
    T *__restrict sigmaxy, T *__restrict sigmaxx, T *__restrict m_vyy,
    T *__restrict m_vyx, T *__restrict m_vxy, T *__restrict m_vxx,
    T *__restrict dvydy_store, T *__restrict dvxdx_store,
    T *__restrict dvydxdvxdy_store, T const *__restrict lamb,
    T const *__restrict mu, T const *__restrict ay, T const *__restrict ayh,
    T const *__restrict ax, T const *__restrict axh, T const *__restrict by,
    T const *__restrict byh, T const *__restrict bx, T const *__restrict bxh,
    int64_t ny, int64_t nx, T dt, T const *__restrict fd_coeffsy,
    T const *__restrict fd_coeffs2y, T const *__restrict fd_coeffs3y,
    T const *__restrict fd_coeffsx, T const *__restrict fd_coeffs2x,
    T const *__restrict fd_coeffs3x, int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
  int64_t ybegin_ii;
  int64_t yend_ii;
  int64_t xbegin_ii;
  int64_t xend_ii;
  int64_t ybegin_xy;
  int64_t yend_xy;
  int64_t xbegin_xy;
  int64_t xend_xy;

  if (pml_y == 0) {
    ybegin_ii = 1;
    yend_ii = pml_regionsy[0] + 1;
    ybegin_xy = 1;
    yend_xy = pml_regionsy[0] + 1;
  } else if (pml_y == 2) {
    ybegin_ii = pml_regionsy[1];
    yend_ii = ny;
    ybegin_xy = pml_regionsy[1] - 1;
    yend_xy = ny - 1;
  } else {
    ybegin_ii = pml_regionsy[0] + 1;
    yend_ii = pml_regionsy[1];
    ybegin_xy = pml_regionsy[0] + 1;
    yend_xy = pml_regionsy[1] - 1;
  }
  if (pml_x == 0) {
    xbegin_ii = 0;
    xend_ii = pml_regionsx[0];
    xbegin_xy = 1;
    xend_xy = pml_regionsx[0] + 1;
  } else if (pml_x == 2) {
    xbegin_ii = pml_regionsx[1] - 1;
    xend_ii = nx - 1;
    xbegin_xy = pml_regionsx[1] - 1;
    xend_xy = nx - 1;
  } else {
    xbegin_ii = pml_regionsx[0];
    xend_ii = pml_regionsx[1] - 1;
    xbegin_xy = pml_regionsx[0] + 1;
    xend_xy = pml_regionsx[1] - 1;
  }

  for (int64_t y = ybegin_ii; y < yend_ii; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_ii; x < xend_ii; ++x) {
      int64_t i{yi + x};
      T dvydy{};
      T dvxdx{};

      // dvydy
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_y == 0 and y == 1 + j) {
          for (int k{}; k <= A; ++k) {
            dvydy += fd_coeffs2y[k] * vy[i + (-j + k - 1) * nx];
          }
        } else if (pml_y == 2 and y == ny - 1 - j) {
          for (int k{}; k <= A; ++k) {
            dvydy -= fd_coeffs2y[k] * vy[i + (j - k) * nx];
          }
        }
      }
      if (pml_y == 1 or (y > 1 + A / 2 - 2 and y < ny - 1 - A / 2 + 2)) {
        for (int k{}; k < A / 2; ++k) {
          dvydy += fd_coeffsy[k] * (vy[i + k * nx] - vy[i - (k + 1) * nx]);
        }
      }

      // dvxdx
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_x == 0 and x == j) {
          for (int k{}; k <= A; ++k) {
            dvxdx += fd_coeffs2x[k] * vx[i - j + k];
          }
        } else if (pml_x == 2 and x == nx - 2 - j) {
          for (int k{}; k <= A; ++k) {
            dvxdx -= fd_coeffs2x[k] * vx[i + j - k + 1];
          }
        }
      }
      if (pml_x == 1 or (x > A / 2 - 2 and x < nx - 2 - A / 2 + 2)) {
        for (int k{}; k < A / 2; ++k) {
          dvxdx += fd_coeffsx[k] * (vx[i + 1 + k] - vx[i - k]);
        }
      }

      if (pml_y == 0 or pml_y == 2) {
        m_vyy[i] = ay[y] * m_vyy[i] + by[y] * dvydy;
        dvydy += m_vyy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_vxx[i] = axh[x] * m_vxx[i] + bxh[x] * dvxdx;
        dvxdx += m_vxx[i];
      }
      T lambyxh{(lamb[i] + lamb[i + 1]) / 2};
      T muyxh{(mu[i] + mu[i + 1]) / 2};
      sigmayy[i] += dt * ((lambyxh + 2 * muyxh) * dvydy + lambyxh * dvxdx);
      sigmaxx[i] += dt * ((lambyxh + 2 * muyxh) * dvxdx + lambyxh * dvydy);
      if (lamb_requires_grad or mu_requires_grad) {
        dvydy_store[i] = dt * dvydy;
        dvxdx_store[i] = dt * dvxdx;
      }
    }
  }

  for (int64_t y = ybegin_xy; y < yend_xy; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_xy; x < xend_xy; ++x) {
      int64_t i{yi + x};
      T dvydx{};
      T dvxdy{};

      // dvxdy
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_y == 0 and y == 1 + j) {
          T dvydxp{};
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (pml_x == 0 and x == 1 + jp) {
              for (int k{}; k <= A; ++k) {
                dvydxp += fd_coeffs2x[k] * vy[i - (j + 1) * nx - (jp + 1) + k];
              }
            } else if (pml_x == 2 and x == nx - 2 - jp) {
              for (int k{}; k <= A; ++k) {
                dvydxp -= fd_coeffs2x[k] * vy[i - (j + 1) * nx + jp - k];
              }
            }
          }
          if (pml_x == 1 or (x > 1 + A / 2 - 2 and x < nx - 2 - A / 2 + 2)) {
            for (int k{}; k < A / 2; ++k) {
              dvydxp += fd_coeffsx[k] * (vy[i - (j + 1) * nx + k] -
                                         vy[i - (j + 1) * nx - k - 1]);
            }
          }
          dvxdy = -fd_coeffs3y[0] * dvydxp;
          for (int k{1}; k <= A + 1; ++k) {
            dvxdy += fd_coeffs3y[k] * vx[i + (-j + k - 1) * nx];
          }
        } else if (pml_y == 2 and y == ny - 2 - j) {
          T dvydxp{};
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (pml_x == 0 and x == 1 + jp) {
              for (int k{}; k <= A; ++k) {
                dvydxp += fd_coeffs2x[k] * vy[i + (j + 1) * nx - (jp + 1) + k];
              }
            } else if (pml_x == 2 and x == nx - 2 - jp) {
              for (int k{}; k <= A; ++k) {
                dvydxp -= fd_coeffs2x[k] * vy[i + (j + 1) * nx + jp - k];
              }
            }
          }
          if (pml_x == 1 or (x > 1 + A / 2 - 2 and x < nx - 2 - A / 2 + 2)) {
            for (int k{}; k < A / 2; ++k) {
              dvydxp += fd_coeffsx[k] * (vy[i + (j + 1) * nx + k] -
                                         vy[i + (j + 1) * nx - k - 1]);
            }
          }
          dvxdy = fd_coeffs3y[0] * dvydxp;
          for (int k{1}; k <= A + 1; ++k) {
            dvxdy -= fd_coeffs3y[k] * vx[i + (j - k + 2) * nx];
          }
        }
      }
      if (pml_y == 1 or (y > 1 + A / 2 - 2 and y < ny - 2 - A / 2 + 2)) {
        for (int k{}; k < A / 2; ++k) {
          dvxdy += fd_coeffsy[k] * (vx[i + (k + 1) * nx] - vx[i - k * nx]);
        }
      }

      // dvydx
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_x == 0 and x == 1 + j) {
          T dvxdyp{};
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (pml_y == 0 and y == 1 + jp) {
              for (int k{}; k <= A; ++k) {
                dvxdyp += fd_coeffs2y[k] * vx[i - (j + 1) + (-jp + k) * nx];
              }
            } else if (pml_y == 2 and y == ny - 2 - jp) {
              for (int k{}; k <= A; ++k) {
                dvxdyp -=
                    fd_coeffs2y[k] * vx[i - (j + 1) + ((jp + 1) - k) * nx];
              }
            }
          }
          if (pml_y == 1 or (y > 1 + A / 2 - 2 and y < ny - 2 - A / 2 + 2)) {
            for (int k{}; k < A / 2; ++k) {
              dvxdyp += fd_coeffsy[k] * (vx[i - (j + 1) + (k + 1) * nx] -
                                         vx[i - (j + 1) - k * nx]);
            }
          }
          dvydx = -fd_coeffs3x[0] * dvxdyp;
          for (int k{1}; k <= A + 1; ++k) {
            dvydx += fd_coeffs3x[k] * vy[i + (-j + k - 2)];
          }
        } else if (pml_x == 2 and x == nx - 2 - j) {
          T dvxdyp{};
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (pml_y == 0 and y == 1 + jp) {
              for (int k{}; k <= A; ++k) {
                dvxdyp += fd_coeffs2y[k] * vx[i + (j + 1) + (-jp + k) * nx];
              }
            } else if (pml_y == 2 and y == ny - 2 - jp) {
              for (int k{}; k <= A; ++k) {
                dvxdyp -= fd_coeffs2y[k] * vx[i + (j + 1) + (jp - k + 1) * nx];
              }
            }
          }
          if (pml_y == 1 or (y > 1 + A / 2 - 2 and y < ny - 2 - A / 2 + 2)) {
            for (int k{}; k < A / 2; ++k) {
              dvxdyp += fd_coeffsy[k] * (vx[i + (j + 1) + (k + 1) * nx] -
                                         vx[i + (j + 1) + (-k) * nx]);
            }
          }
          dvydx = fd_coeffs3x[0] * dvxdyp;
          for (int k{1}; k <= A + 1; ++k) {
            dvydx -= fd_coeffs3x[k] * vy[i + (j - k + 1)];
          }
        }
      }
      if (pml_x == 1 or (x > 1 + A / 2 - 2 and x < nx - 2 - A / 2 + 2)) {
        for (int k{}; k < A / 2; ++k) {
          dvydx += fd_coeffsx[k] * (vy[i + k] - vy[i - k - 1]);
        }
      }

      if (pml_y == 0 or pml_y == 2) {
        m_vxy[i] = ayh[y] * m_vxy[i] + byh[y] * dvxdy;
        dvxdy += m_vxy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_vyx[i] = ax[x] * m_vyx[i] + bx[x] * dvydx;
        dvydx += m_vyx[i];
      }
      T muyhx{(mu[i] + mu[i + nx]) / 2};
      sigmaxy[i] += dt * muyhx * (dvydx + dvxdy);
      if (mu_requires_grad) {
        dvydxdvxdy_store[i] = dt * (dvydx + dvxdy);
      }
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

template <typename T>
void add_to_grad_lamb(T *__restrict grad_lamb, T const *__restrict sigmayy,
                      T const *__restrict sigmaxx,
                      T const *__restrict dvydy_store,
                      T const *__restrict dvxdx_store, int64_t step_ratio,
                      int64_t ny, int64_t nx) {
  for (int64_t y = 1; y < ny; ++y) {
    int64_t yi{y * nx};
    {
      int64_t i{yi};
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2) *
          step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2 +
           (sigmayy[i - 1] + sigmaxx[i - 1]) *
               (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
          step_ratio;
    }
    {
      int64_t i{yi + nx - 1};
      grad_lamb[i] += ((sigmayy[i - 1] + sigmaxx[i - 1]) *
                       (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
                      step_ratio;
    }
  }
}

template <typename T>
void add_to_grad_mu(T *__restrict grad_mu, T const *__restrict sigmayy,
                    T const *__restrict sigmaxy, T const *__restrict sigmaxx,
                    T const *__restrict dvydy_store,
                    T const *__restrict dvxdx_store,
                    T const *__restrict dvydxdvxdy_store, int64_t step_ratio,
                    int64_t ny, int64_t nx) {
  {
    int64_t y{1};
    int64_t yi{y * nx};
    {
      int64_t i{yi};
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2) *
          step_ratio;
    }
    {
      int64_t i{yi + nx - 1};
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    step_ratio;
    }
  }
  for (int64_t y = 2; y < ny - 1; ++y) {
    int64_t yi{y * nx};
    {
      int64_t i{yi};
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2 +
           sigmaxy[i - nx] * dvydxdvxdy_store[i - nx] / 2) *
          step_ratio;
    }
    {
      int64_t i{yi + nx - 1};
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    step_ratio;
    }
  }
  {
    int64_t y{ny - 1};
    int64_t yi{y * nx};
    {
      int64_t i{yi};
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i - nx] * dvydxdvxdy_store[i - nx] / 2) *
          step_ratio;
    }
    {
      int64_t i{yi + nx - 1};
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    step_ratio;
    }
  }
}

template <typename T>
void add_to_grad_buoyancy(T *__restrict grad_buoyancy, T const *__restrict vy,
                          T const *__restrict vx,
                          T const *__restrict dvydbuoyancy,
                          T const *__restrict dvxdbuoyancy, int64_t step_ratio,
                          int64_t ny, int64_t nx) {
  {
    int64_t y = 0;
    int64_t yi{y * nx};
    {
      int64_t x{0};
      int64_t i{yi + x};
      grad_buoyancy[i] += (vy[i] * dvydbuoyancy[i] / 4) * step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4) *
          step_ratio;
    }
    {
      int64_t x{nx - 1};
      int64_t i{yi + x};
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 4) * step_ratio;
    }
  }
  for (int64_t y = 1; y < ny - 1; ++y) {
    int64_t yi{y * nx};
    {
      int64_t x{0};
      int64_t i{yi + x};
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
           vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          step_ratio;
    }
    {
      int64_t x{nx - 1};
      int64_t i{yi + x};
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
                           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          step_ratio;
    }
  }
  {
    int64_t y{ny - 1};
    int64_t yi{y * nx};
    {
      int64_t x{0};
      int64_t i{yi + x};
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 2 + vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          step_ratio;
    }
    for (int64_t x = 1; x < nx - 1; ++x) {
      int64_t i{yi + x};
      grad_buoyancy[i] +=
          (vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
           vy[i] * dvydbuoyancy[i] / 2 + vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
           vx[i] * dvxdbuoyancy[i]) *
          step_ratio;
    }
    {
      int64_t x{nx - 1};
      int64_t i{yi + x};
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
                           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          step_ratio;
    }
  }
}

template <typename T, int A, int pml_y, int pml_x>
inline void backward_kernel_sigma(
    T *__restrict vy, T *__restrict vx, T const *__restrict sigmayy,
    T const *__restrict sigmaxy, T const *__restrict sigmaxx,
    T const *__restrict m_vyy, T const *__restrict m_vyx,
    T const *__restrict m_vxy, T const *__restrict m_vxx,
    T const *__restrict m_sigmayyy, T const *__restrict m_sigmaxyy,
    T const *__restrict m_sigmaxyx, T const *__restrict m_sigmaxxx,
    T *__restrict m_sigmayyyn, T *__restrict m_sigmaxyyn,
    T *__restrict m_sigmaxyxn, T *__restrict m_sigmaxxxn,
    T const *__restrict lamb, T const *__restrict mu,
    T const *__restrict buoyancy, T const *__restrict ay,
    T const *__restrict ayh, T const *__restrict ax, T const *__restrict axh,
    T const *__restrict by, T const *__restrict byh, T const *__restrict bx,
    T const *__restrict bxh, int64_t ny, int64_t nx, T dt,
    T const *__restrict fd_coeffsy, T const *__restrict fd_coeffs2y,
    T const *__restrict fd_coeffs3y, T const *__restrict fd_coeffsx,
    T const *__restrict fd_coeffs2x, T const *__restrict fd_coeffs3x,
    int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
  int64_t ybegin_y;
  int64_t yend_y;
  int64_t xbegin_y;
  int64_t xend_y;
  int64_t ybegin_x;
  int64_t yend_x;
  int64_t xbegin_x;
  int64_t xend_x;

  if (pml_y == 0) {
    ybegin_y = 0;
    yend_y = pml_regionsy[0];
    ybegin_x = 1;
    yend_x = pml_regionsy[0] + 1;
  } else if (pml_y == 2) {
    ybegin_y = pml_regionsy[1];
    yend_y = ny;
    ybegin_x = pml_regionsy[1];
    yend_x = ny;
  } else {
    ybegin_y = pml_regionsy[0];
    yend_y = pml_regionsy[1];
    ybegin_x = pml_regionsy[0] + 1;
    yend_x = pml_regionsy[1];
  }
  if (pml_x == 0) {
    xbegin_y = 0;
    xend_y = pml_regionsx[0];
    xbegin_x = 0;
    xend_x = pml_regionsx[0];
  } else if (pml_x == 2) {
    xbegin_y = pml_regionsx[1] - 1;
    xend_y = nx - 1;
    xbegin_x = pml_regionsx[1];
    xend_x = nx;
  } else {
    xbegin_y = pml_regionsx[0];
    xend_y = pml_regionsx[1] - 1;
    xbegin_x = pml_regionsx[0];
    xend_x = pml_regionsx[1];
  }

  for (int64_t y = ybegin_y; y < yend_y; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_y; x < xend_y; ++x) {
      int64_t i{yi + x};

      // from sigmayy/sigmaxx edges
      for (int k{}; k <= A; ++k) {
        for (int j{}; j < A / 2 - 1; ++j) {
          if (pml_y == 0 and y == 1 + j + (-j + k - 1)) {
            T lambyxh{(lamb[i - (-j + k - 1) * nx] +
                       lamb[i + 1 - (-j + k - 1) * nx]) /
                      2};
            T muyxh{
                (mu[i - (-j + k - 1) * nx] + mu[i + 1 - (-j + k - 1) * nx]) /
                2};
            vy[i] +=
                fd_coeffs2y[k] *
                (dt * (1 + by[y - (-j + k - 1)]) *
                     ((lambyxh + 2 * muyxh) * sigmayy[i - (-j + k - 1) * nx] +
                      lambyxh * sigmaxx[i - (-j + k - 1) * nx]) +
                 by[y - (-j + k - 1)] * m_vyy[i - (-j + k - 1) * nx]);
          } else if (pml_y == 2 and y == ny - 1 - j + (j - k)) {
            T lambyxh{(lamb[i - (j - k) * nx] + lamb[i + 1 - (j - k) * nx]) /
                      2};
            T muyxh{(mu[i - (j - k) * nx] + mu[i + 1 - (j - k) * nx]) / 2};
            vy[i] -= fd_coeffs2y[k] *
                     (dt * (1 + by[y - (j - k)]) *
                          ((lambyxh + 2 * muyxh) * sigmayy[i - (j - k) * nx] +
                           lambyxh * sigmaxx[i - (j - k) * nx]) +
                      by[y - (j - k)] * m_vyy[i - (j - k) * nx]);
          }
        }
      }

      // from sigmayy/sigmaxx centre
      for (int k{}; k < A / 2; ++k) {
        if (pml_y == 1 or
            (y > 1 + A / 2 - 2 + k and y < ny - 1 - A / 2 + 2 + k)) {
          T lambyxh{(lamb[i - k * nx] + lamb[i + 1 - k * nx]) / 2};
          T muyxh{(mu[i - k * nx] + mu[i + 1 - k * nx]) / 2};
          vy[i] += fd_coeffsy[k] *
                   (dt * (1 + by[y - k]) *
                        ((lambyxh + 2 * muyxh) * sigmayy[i - k * nx] +
                         lambyxh * sigmaxx[i - k * nx]) +
                    by[y - k] * m_vyy[i - k * nx]);
        }
        if (pml_y == 1 or (y > 1 + A / 2 - 2 - (k + 1) and
                           y < ny - 1 - A / 2 + 2 - (k + 1))) {
          T lambyxh{(lamb[i + (k + 1) * nx] + lamb[i + 1 + (k + 1) * nx]) / 2};
          T muyxh{(mu[i + (k + 1) * nx] + mu[i + 1 + (k + 1) * nx]) / 2};
          vy[i] -= fd_coeffsy[k] *
                   (dt * (1 + by[y + k + 1]) *
                        ((lambyxh + 2 * muyxh) * sigmayy[i + (k + 1) * nx] +
                         lambyxh * sigmaxx[i + (k + 1) * nx]) +
                    by[y + k + 1] * m_vyy[i + (k + 1) * nx]);
        }
      }

      // from sigmaxy dvxdy
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_y == 0 and y == 1 + j - (j + 1)) {
          int64_t y2{y + (j + 1)};
          for (int k{}; k <= A; ++k) {
            for (int jp{}; jp < A / 2 - 1; ++jp) {
              if (pml_x == 0 and x == 1 + jp - (jp + 1) + k) {
                int64_t i2{i - (-(j + 1) * nx - (jp + 1) + k)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vy[i] += fd_coeffs2x[k] * (-fd_coeffs3y[0]) *
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                          byh[y2] * m_vxy[i2]);
              } else if (pml_x == 2 and x == nx - 2 - jp + jp - k) {
                int64_t i2{i - (-(j + 1) * nx + jp - k)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vy[i] -= fd_coeffs2x[k] * (-fd_coeffs3y[0]) *
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                          byh[y2] * m_vxy[i2]);
              }
            }
          }
          for (int k{}; k < A / 2; ++k) {
            if (pml_x == 1 or
                (x > 1 + A / 2 - 2 + k and x < nx - 2 - A / 2 + 2 + k)) {
              int64_t i2{i - (-(j + 1) * nx + k)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vy[i] += fd_coeffsx[k] * (-fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
            if (pml_x == 1 or (x > 1 + A / 2 - 2 - k - 1 and
                               x < nx - 2 - A / 2 + 2 - k - 1)) {
              int64_t i2{i - (-(j + 1) * nx - k - 1)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vy[i] -= fd_coeffsx[k] * (-fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
          }

        } else if (pml_y == 2 and y == ny - 2 - j + (j + 1)) {
          int64_t y2{y - (j + 1)};
          for (int k{}; k <= A; ++k) {
            for (int jp{}; jp < A / 2 - 1; ++jp) {
              if (pml_x == 0 and x == 1 + jp - (jp + 1) + k) {
                int64_t i2{i - ((j + 1) * nx - (jp + 1) + k)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vy[i] += fd_coeffs2x[k] * (fd_coeffs3y[0]) *
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                          byh[y2] * m_vxy[i2]);
              } else if (pml_x == 2 and x == nx - 2 - jp + jp - k) {
                int64_t i2{i - ((j + 1) * nx + jp - k)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vy[i] -= fd_coeffs2x[k] * (fd_coeffs3y[0]) *
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                          byh[y2] * m_vxy[i2]);
              }
            }
          }
          for (int k{}; k < A / 2; ++k) {
            if (pml_x == 1 or
                (x > 1 + A / 2 - 2 + k and x < nx - 2 - A / 2 + 2 + k)) {
              int64_t i2{i - ((j + 1) * nx + k)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vy[i] += fd_coeffsx[k] * (fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
            if (pml_x == 1 or (x > 1 + A / 2 - 2 - k - 1 and
                               x < nx - 2 - A / 2 + 2 - k - 1)) {
              int64_t i2{i - ((j + 1) * nx - k - 1)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vy[i] -= fd_coeffsx[k] * (fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
          }
        }
      }

      // from sigmaxy dvydx
      if (y > 0 and y < ny - 1) {
        for (int k{1}; k <= A + 1; ++k) {
          for (int j{}; j < A / 2 - 1; ++j) {
            if (pml_x == 0 and x == 1 + j + (-j + k - 2)) {
              int64_t x2{x - (-j + k - 2)};
              int64_t i2{i - (-j + k - 2)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vy[i] +=
                  fd_coeffs3x[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                    bx[x2] * m_vyx[i2]);
            } else if (pml_x == 2 and x == nx - 2 - j + (j - k + 1)) {
              int64_t x2{x - (j - k + 1)};
              int64_t i2{i - (j - k + 1)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vy[i] -=
                  fd_coeffs3x[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                    bx[x2] * m_vyx[i2]);
            }
          }
        }
        for (int k{}; k < A / 2; ++k) {
          if (pml_x == 1 or
              (x > 1 + A / 2 - 2 + k and x < nx - 2 - A / 2 + 2 + k)) {
            int64_t x2{x - k};
            int64_t i2{i - k};
            T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
            vy[i] += fd_coeffsx[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                      bx[x2] * m_vyx[i2]);
          }
          if (pml_x == 1 or
              (x > 1 + A / 2 - 2 - k - 1 and x < nx - 2 - A / 2 + 2 - k - 1)) {
            int64_t x2{x + k + 1};
            int64_t i2{i + k + 1};
            T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
            vy[i] -= fd_coeffsx[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                      bx[x2] * m_vyx[i2]);
          }
        }
      }

      T buoyancyyhxh;
      if (pml_y == 2 and y == ny - 1) {
        buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1]) / 2;
      } else {
        buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1] + buoyancy[i + nx] +
                        buoyancy[i + nx + 1]) /
                       4;
      }

      if (pml_y == 0 or pml_y == 2) {
        m_sigmayyyn[i] =
            buoyancyyhxh * dt * ayh[y] * vy[i] + ayh[y] * m_sigmayyy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_sigmaxyxn[i] =
            buoyancyyhxh * dt * axh[x] * vy[i] + axh[x] * m_sigmaxyx[i];
      }
    }
  }

  for (int64_t y = ybegin_x; y < yend_x; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_x; x < xend_x; ++x) {
      int64_t i{yi + x};

      // from sigmayy/sigmaxx edges
      for (int k{}; k <= A; ++k) {
        for (int j{}; j < A / 2 - 1; ++j) {
          if (pml_x == 0 and x == j + (-j + k)) {
            int64_t i2{i - (-j + k)};
            int64_t x2{x - (-j + k)};
            T lambyxh{(lamb[i2] + lamb[i2 + 1]) / 2};
            T muyxh{(mu[i2] + mu[i2 + 1]) / 2};
            vx[i] +=
                fd_coeffs2x[k] * (dt * (1 + bxh[x2]) *
                                      ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                       lambyxh * sigmayy[i2]) +
                                  bxh[x2] * m_vxx[i2]);
          } else if (pml_x == 2 and x == nx - 2 - j + (j - k + 1)) {
            int64_t i2{i - (j - k + 1)};
            int64_t x2{x - (j - k + 1)};
            T lambyxh{(lamb[i2] + lamb[i2 + 1]) / 2};
            T muyxh{(mu[i2] + mu[i2 + 1]) / 2};
            vx[i] -=
                fd_coeffs2x[k] * (dt * (1 + bxh[x2]) *
                                      ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                       lambyxh * sigmayy[i2]) +
                                  bxh[x2] * m_vxx[i2]);
          }
        }
      }

      // from sigmayy/sigmaxx centre
      for (int k{}; k < A / 2; ++k) {
        if (pml_x == 1 or
            (x > A / 2 - 2 + 1 + k and x < nx - 2 - A / 2 + 2 + 1 + k)) {
          int64_t i2{i - (1 + k)};
          int64_t x2{x - (1 + k)};
          T lambyxh{(lamb[i2] + lamb[i2 + 1]) / 2};
          T muyxh{(mu[i2] + mu[i2 + 1]) / 2};
          vx[i] += fd_coeffsx[k] * (dt * (1 + bxh[x2]) *
                                        ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                         lambyxh * sigmayy[i2]) +
                                    bxh[x2] * m_vxx[i2]);
        }
        if (pml_x == 1 or (x > A / 2 - 2 - k and x < nx - 2 - A / 2 + 2 - k)) {
          int64_t i2{i + k};
          int64_t x2{x + k};
          T lambyxh{(lamb[i2] + lamb[i2 + 1]) / 2};
          T muyxh{(mu[i2] + mu[i2 + 1]) / 2};
          vx[i] -= fd_coeffsx[k] * (dt * (1 + bxh[x2]) *
                                        ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                         lambyxh * sigmayy[i2]) +
                                    bxh[x2] * m_vxx[i2]);
        }
      }

      // from sigmaxy dvydx
      for (int j{}; j < A / 2 - 1; ++j) {
        if (pml_x == 0 and x == 1 + j - (j + 1)) {
          int64_t x2{x + (j + 1)};
          for (int k{}; k <= A; ++k) {
            for (int jp{}; jp < A / 2 - 1; ++jp) {
              if (pml_y == 0 and y == 1 + jp - jp + k) {
                int64_t i2{i - (-(j + 1) + (-jp + k) * nx)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vx[i] += fd_coeffs2y[k] * (-fd_coeffs3x[0]) *
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                          bx[x2] * m_vyx[i2]);
              } else if (pml_y == 2 and y == ny - 2 - jp + (jp + 1) - k) {
                int64_t i2{i - (-(j + 1) + ((jp + 1) - k) * nx)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vx[i] -= fd_coeffs2y[k] * (-fd_coeffs3x[0]) *
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                          bx[x2] * m_vyx[i2]);
              }
            }
          }
          for (int k{}; k < A / 2; ++k) {
            if (pml_y == 1 or (y > 1 + A / 2 - 2 + k + 1 and
                               y < ny - 2 - A / 2 + 2 + k + 1)) {
              int64_t i2{i - (-(j + 1) + (k + 1) * nx)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vx[i] += fd_coeffsy[k] * (-fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
            if (pml_y == 1 or
                (y > 1 + A / 2 - 2 - k and y < ny - 2 - A / 2 + 2 - k)) {
              int64_t i2{i - (-(j + 1) - k * nx)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vx[i] -= fd_coeffsy[k] * (-fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
          }

        } else if (pml_x == 2 and x == nx - 2 - j + (j + 1)) {
          int64_t x2{x - (j + 1)};
          for (int k{}; k <= A; ++k) {
            for (int jp{}; jp < A / 2 - 1; ++jp) {
              if (pml_y == 0 and y == 1 + jp - jp + k) {
                int64_t i2{i - ((j + 1) + (-jp + k) * nx)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vx[i] += fd_coeffs2y[k] * (fd_coeffs3x[0]) *
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                          bx[x2] * m_vyx[i2]);
              } else if (pml_y == 2 and y == ny - 2 - jp + jp - k + 1) {
                int64_t i2{i - ((j + 1) + (jp - k + 1) * nx)};
                T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
                vx[i] -= fd_coeffs2y[k] * (fd_coeffs3x[0]) *
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                          bx[x2] * m_vyx[i2]);
              }
            }
          }
          for (int k{}; k < A / 2; ++k) {
            if (pml_y == 1 or (y > 1 + A / 2 - 2 + k + 1 and
                               y < ny - 2 - A / 2 + 2 + k + 1)) {
              int64_t i2{i - ((j + 1) + (k + 1) * nx)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vx[i] += fd_coeffsy[k] * (fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
            if (pml_y == 1 or
                (y > 1 + A / 2 - 2 - k and y < ny - 2 - A / 2 + 2 - k)) {
              int64_t i2{i - ((j + 1) + (-k) * nx)};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vx[i] -= fd_coeffsy[k] * (fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
          }
        }
      }

      // from sigmaxy dvxdy
      if (x > 0 and x < nx - 1) {
        for (int k{1}; k <= A + 1; ++k) {
          for (int j{}; j < A / 2 - 1; ++j) {
            if (pml_y == 0 and y == 1 + j + (-j + k - 1)) {
              int64_t y2{y - (-j + k - 1)};
              int64_t i2{i - (-j + k - 1) * nx};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vx[i] +=
                  fd_coeffs3y[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                    byh[y2] * m_vxy[i2]);
            } else if (pml_y == 2 and y == ny - 2 - j + (j - k + 2)) {
              int64_t y2{y - (j - k + 2)};
              int64_t i2{i - (j - k + 2) * nx};
              T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
              vx[i] -=
                  fd_coeffs3y[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                    byh[y2] * m_vxy[i2]);
            }
          }
        }
        for (int k{}; k < A / 2; ++k) {
          if (pml_y == 1 or
              (y > 1 + A / 2 - 2 + k + 1 and y < ny - 2 - A / 2 + 2 + k + 1)) {
            int64_t y2{y - (k + 1)};
            int64_t i2{i - (k + 1) * nx};
            T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
            vx[i] += fd_coeffsy[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                      byh[y2] * m_vxy[i2]);
          }
          if (pml_y == 1 or
              (y > 1 + A / 2 - 2 - k and y < ny - 2 - A / 2 + 2 - k)) {
            int64_t y2{y + k};
            int64_t i2{i + k * nx};
            T muyhx{(mu[i2] + mu[i2 + nx]) / 2};
            vx[i] -= fd_coeffsy[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                      byh[y2] * m_vxy[i2]);
          }
        }
      }

      if (pml_y == 0 or pml_y == 2) {
        m_sigmaxyyn[i] =
            buoyancy[i] * dt * ay[y] * vx[i] + ay[y] * m_sigmaxyy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_sigmaxxxn[i] =
            buoyancy[i] * dt * ax[x] * vx[i] + ax[x] * m_sigmaxxx[i];
      }
    }
  }
}

template <typename T, int A, int pml_y, int pml_x>
inline void backward_kernel_v(
    T const *__restrict vy, T const *__restrict vx, T *__restrict sigmayy,
    T *__restrict sigmaxy, T *__restrict sigmaxx, T *__restrict m_vyy,
    T *__restrict m_vyx, T *__restrict m_vxy, T *__restrict m_vxx,
    T const *__restrict m_sigmayyy, T const *__restrict m_sigmaxyy,
    T const *__restrict m_sigmaxyx, T const *__restrict m_sigmaxxx,
    T const *__restrict lamb, T const *__restrict mu,
    T const *__restrict buoyancy, T const *__restrict ay,
    T const *__restrict ayh, T const *__restrict ax, T const *__restrict axh,
    T const *__restrict by, T const *__restrict byh, T const *__restrict bx,
    T const *__restrict bxh, int64_t ny, int64_t nx, T dt,
    T const *__restrict fd_coeffsy, T const fd_coeffs1y[][5],
    T const *__restrict fd_coeffs2y, T const *__restrict fd_coeffsx,
    T const fd_coeffs1x[][5], T const *__restrict fd_coeffs2x,
    int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
  int64_t ybegin_ii;
  int64_t yend_ii;
  int64_t xbegin_ii;
  int64_t xend_ii;
  int64_t ybegin_xy;
  int64_t yend_xy;
  int64_t xbegin_xy;
  int64_t xend_xy;

  if (pml_y == 0) {
    ybegin_ii = 1;
    yend_ii = pml_regionsy[0] + 1;
    ybegin_xy = 1;
    yend_xy = pml_regionsy[0] + 1;
  } else if (pml_y == 2) {
    ybegin_ii = pml_regionsy[1];
    yend_ii = ny;
    ybegin_xy = pml_regionsy[1] - 1;
    yend_xy = ny - 1;
  } else {
    ybegin_ii = pml_regionsy[0] + 1;
    yend_ii = pml_regionsy[1];
    ybegin_xy = pml_regionsy[0] + 1;
    yend_xy = pml_regionsy[1] - 1;
  }
  if (pml_x == 0) {
    xbegin_ii = 0;
    xend_ii = pml_regionsx[0];
    xbegin_xy = 1;
    xend_xy = pml_regionsx[0] + 1;
  } else if (pml_x == 2) {
    xbegin_ii = pml_regionsx[1] - 1;
    xend_ii = nx - 1;
    xbegin_xy = pml_regionsx[1] - 1;
    xend_xy = nx - 1;
  } else {
    xbegin_ii = pml_regionsx[0];
    xend_ii = pml_regionsx[1] - 1;
    xbegin_xy = pml_regionsx[0] + 1;
    xend_xy = pml_regionsx[1] - 1;
  }

  for (int64_t y = ybegin_ii; y < yend_ii; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_ii; x < xend_ii; ++x) {
      int64_t i{yi + x};
      T lambyxh{(lamb[i] + lamb[i + 1]) / 2};
      T muyxh{(mu[i] + mu[i + 1]) / 2};

      if (pml_y == 0 or pml_y == 2) {
        m_vyy[i] = (lambyxh + 2 * muyxh) * dt * ay[y] * sigmayy[i] +
                   lambyxh * dt * ay[y] * sigmaxx[i] + ay[y] * m_vyy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_vxx[i] = (lambyxh + 2 * muyxh) * dt * axh[x] * sigmaxx[i] +
                   lambyxh * dt * axh[x] * sigmayy[i] + axh[x] * m_vxx[i];
      }

      // dsigmayydy
      for (int k{}; k < A; ++k) {
        for (int j{}; j < A / 2; ++j) {
          if (pml_y == 0 and y == j + (1 - j + k)) {
            int64_t i2{i - (1 - j + k) * nx};
            int64_t y2{y - (1 - j + k)};
            T buoyancyyhxh;
            if (pml_y == 2 and y2 == ny - 1) {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
            } else {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                             4;
            }
            sigmayy[i] += fd_coeffs1y[j][1 + k] *
                          (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                           byh[y2] * m_sigmayyy[i2]);
          } else if (pml_y == 2 and y == ny - 1 - j + (j - k)) {
            int64_t i2{i - (j - k) * nx};
            int64_t y2{y - (j - k)};
            T buoyancyyhxh;
            if (pml_y == 2 and y2 == ny - 1) {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
            } else {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                             4;
            }
            sigmayy[i] -= fd_coeffs1y[j][1 + k] *
                          (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                           byh[y2] * m_sigmayyy[i2]);
          }
        }
      }
      for (int k{}; k < A / 2; ++k) {
        if (pml_y == 1 or
            (y > A / 2 - 1 + (1 + k) and y < ny - 1 - A / 2 + 1 + (1 + k))) {
          int64_t i2{i - (1 + k) * nx};
          int64_t y2{y - (1 + k)};
          T buoyancyyhxh;
          if (pml_y == 2 and y2 == ny - 1) {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
          } else {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                            buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                           4;
          }
          sigmayy[i] +=
              fd_coeffsy[k] * (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                               byh[y2] * m_sigmayyy[i2]);
        }
        if (pml_y == 1 or (y > A / 2 - 1 - k and y < ny - 1 - A / 2 + 1 - k)) {
          int64_t i2{i - (-k) * nx};
          int64_t y2{y - (-k)};
          T buoyancyyhxh;
          if (pml_y == 2 and y2 == ny - 1) {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
          } else {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                            buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                           4;
          }
          sigmayy[i] -=
              fd_coeffsy[k] * (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                               byh[y2] * m_sigmayyy[i2]);
        }
      }

      // dsigmaxxdx
      for (int k{}; k < A; ++k) {
        for (int j{}; j < A / 2; ++j) {
          if (pml_x == 0 and x == j + (-j + k)) {
            int64_t i2{i - (-j + k)};
            int64_t x2{x - (-j + k)};
            sigmaxx[i] += fd_coeffs1x[j][1 + k] *
                          (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +
                           bx[x2] * m_sigmaxxx[i2]);
          } else if (pml_x == 2 and x == nx - 1 - j + (j - k - 1)) {
            int64_t i2{i - (j - k - 1)};
            int64_t x2{x - (j - k - 1)};
            sigmaxx[i] -= fd_coeffs1x[j][1 + k] *
                          (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +
                           bx[x2] * m_sigmaxxx[i2]);
          }
        }
      }
      for (int k{}; k < A / 2; ++k) {
        if (pml_x == 1 or
            (x > A / 2 - 1 + (k) and x < nx - 1 - A / 2 + 1 + (k))) {
          int64_t i2{i - (k)};
          int64_t x2{x - (k)};
          sigmaxx[i] +=
              fd_coeffsx[k] * (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +
                               bx[x2] * m_sigmaxxx[i2]);
        }
        if (pml_x == 1 or
            (x > A / 2 - 1 - (1 + k) and x < nx - 1 - A / 2 + 1 - (1 + k))) {
          int64_t i2{i + (1 + k)};
          int64_t x2{x + (1 + k)};
          sigmaxx[i] -=
              fd_coeffsx[k] * (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +
                               bx[x2] * m_sigmaxxx[i2]);
        }
      }
    }
  }

  for (int64_t y = ybegin_xy; y < yend_xy; ++y) {
    int64_t yi{y * nx};
    for (int64_t x = xbegin_xy; x < xend_xy; ++x) {
      int64_t i{yi + x};
      T muyhx{(mu[i] + mu[i + nx]) / 2};

      if (pml_y == 0 or pml_y == 2) {
        m_vxy[i] = muyhx * dt * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i];
      }
      if (pml_x == 0 or pml_x == 2) {
        m_vyx[i] = muyhx * dt * ax[x] * sigmaxy[i] + ax[x] * m_vyx[i];
      }

      // dsigmaxydx
      for (int k{}; k < A; ++k) {
        for (int j{}; j < A / 2 - 1; ++j) {
          if (pml_x == 0 and x == j - j + 1 + k) {
            int64_t i2{i - (-j + 1 + k)};
            int64_t x2{x - (-j + 1 + k)};
            T buoyancyyhxh;
            if (pml_y == 2 and y == ny - 1) {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
            } else {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                             4;
            }
            sigmaxy[i] += fd_coeffs2x[1 + k] *
                          (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                           bxh[x2] * m_sigmaxyx[i2]);
          } else if (pml_x == 2 and x == nx - 2 - j + j - k) {
            int64_t i2{i - (j - k)};
            int64_t x2{x - (j - k)};
            T buoyancyyhxh;
            if (pml_y == 2 and y == ny - 1) {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
            } else {
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                             4;
            }
            sigmaxy[i] -= fd_coeffs2x[1 + k] *
                          (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                           bxh[x2] * m_sigmaxyx[i2]);
          }
        }
      }
      for (int k{}; k < A / 2; ++k) {
        if (pml_x == 1 or
            (x > A / 2 - 2 + 1 + k and x < nx - 2 - A / 2 + 2 + 1 + k)) {
          int64_t i2{i - (1 + k)};
          int64_t x2{x - (1 + k)};
          T buoyancyyhxh;
          if (pml_y == 2 and y == ny - 1) {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
          } else {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                            buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                           4;
          }
          sigmaxy[i] +=
              fd_coeffsx[k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                               bxh[x2] * m_sigmaxyx[i2]);
        }
        if (pml_x == 1 or (x > A / 2 - 2 - k and x < nx - 2 - A / 2 + 2 - k)) {
          int64_t i2{i - (-k)};
          int64_t x2{x - (-k)};
          T buoyancyyhxh;
          if (pml_y == 2 and y == ny - 1) {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;
          } else {
            buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +
                            buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /
                           4;
          }
          sigmaxy[i] -=
              fd_coeffsx[k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                               bxh[x2] * m_sigmaxyx[i2]);
        }
      }

      // dsigmaxydy
      for (int k{}; k < A; ++k) {
        for (int j{}; j < A / 2 - 1; ++j) {
          if (pml_y == 0 and y == 1 + j - j + k) {
            int64_t i2{i - (-j + k) * nx};
            int64_t y2{y - (-j + k)};
            sigmaxy[i] += fd_coeffs2y[1 + k] *
                          (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +
                           by[y2] * m_sigmaxyy[i2]);
          } else if (pml_y == 2 and y == ny - 1 - j + j - k - 1) {
            int64_t i2{i - (j - k - 1) * nx};
            int64_t y2{y - (j - k - 1)};
            sigmaxy[i] -= fd_coeffs2y[1 + k] *
                          (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +
                           by[y2] * m_sigmaxyy[i2]);
          }
        }
      }
      for (int k{}; k < A / 2; ++k) {
        if (pml_y == 1 or
            (y > 1 + A / 2 - 2 + k and y < ny - 1 - A / 2 + 2 + k)) {
          int64_t i2{i - (k)*nx};
          int64_t y2{y - (k)};
          sigmaxy[i] +=
              fd_coeffsy[k] * (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +
                               by[y2] * m_sigmaxyy[i2]);
        }
        if (pml_y == 1 or (y > 1 + A / 2 - 2 - (k + 1) and
                           y < ny - 1 - A / 2 + 2 - (k + 1))) {
          int64_t i2{i - (-(k + 1)) * nx};
          int64_t y2{y - (-(k + 1))};
          sigmaxy[i] -=
              fd_coeffsy[k] * (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +
                               by[y2] * m_sigmaxyy[i2]);
        }
      }
    }
  }
}

template <typename T, int A>
void forward_shot(
    T *__restrict vy, T *__restrict vx, T *__restrict sigmayy,
    T *__restrict sigmaxy, T *__restrict sigmaxx, T *__restrict m_vyy,
    T *__restrict m_vyx, T *__restrict m_vxy, T *__restrict m_vxx,
    T *__restrict m_sigmayyy, T *__restrict m_sigmaxyy,
    T *__restrict m_sigmaxyx, T *__restrict m_sigmaxxx,
    int64_t const *__restrict sources_y_i,
    int64_t const *__restrict sources_x_i,
    int64_t const *__restrict receivers_y_i,
    int64_t const *__restrict receivers_x_i, T *__restrict dvydbuoyancy,
    T *__restrict dvxdbuoyancy, T *__restrict dvydy_store,
    T *__restrict dvxdx_store, T *__restrict dvydxdvxdy_store,
    T const *__restrict lamb, T const *__restrict mu,
    T const *__restrict buoyancy, T const *__restrict f_y,
    T const *__restrict f_x, T *__restrict r_y, T *__restrict r_x,
    T const *__restrict ay, T const *__restrict ayh, T const *__restrict ax,
    T const *__restrict axh, T const *__restrict by, T const *__restrict byh,
    T const *__restrict bx, T const *__restrict bxh, T dt,
    T const *__restrict fd_coeffsy, T const fd_coeffs1y[][5],
    T const *__restrict fd_coeffs2y, T const *__restrict fd_coeffs3y,
    T const *__restrict fd_coeffsx, T const fd_coeffs1x[][5],
    T const *__restrict fd_coeffs2x, T const *__restrict fd_coeffs3x,
    int64_t n_sources_per_shot_y, int64_t n_sources_per_shot_x,
    int64_t n_receivers_per_shot_y, int64_t n_receivers_per_shot_x, int64_t ny,
    int64_t nx, int64_t nt, int64_t step_ratio, bool lamb_requires_grad,
    bool mu_requires_grad, bool buoyancy_requires_grad,
    int64_t const *__restrict pml_regionsy,
    int64_t const *__restrict pml_regionsx) {
#define FORWARD_KERNEL_VGRAD(pml_y, pml_x)                                   \
  forward_kernel_v<T, A, true, pml_y, pml_x>(                                \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx, \
      m_sigmaxxx, dvydbuoyancy + (t / step_ratio) * ny * nx,                 \
      dvxdbuoyancy + (t / step_ratio) * ny * nx, buoyancy, ay, ayh, ax, axh, \
      by, byh, bx, bxh, ny, nx, dt, fd_coeffsy, fd_coeffs1y, fd_coeffs2y,    \
      fd_coeffsx, fd_coeffs1x, fd_coeffs2x, pml_regionsy, pml_regionsx)
#define FORWARD_KERNEL_VNOGRAD(pml_y, pml_x)                                 \
  forward_kernel_v<T, A, false, pml_y, pml_x>(                               \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx, \
      m_sigmaxxx, nullptr, nullptr, buoyancy, ay, ayh, ax, axh, by, byh, bx, \
      bxh, ny, nx, dt, fd_coeffsy, fd_coeffs1y, fd_coeffs2y, fd_coeffsx,     \
      fd_coeffs1x, fd_coeffs2x, pml_regionsy, pml_regionsx)
#define FORWARD_KERNEL_SIGMALAMBGRAD(pml_y, pml_x)                          \
  forward_kernel_sigma<T, A, true, false, pml_y, pml_x>(                    \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,        \
      dvydy_store + (t / step_ratio) * ny * nx,                             \
      dvxdx_store + (t / step_ratio) * ny * nx, nullptr, lamb, mu, ay, ayh, \
      ax, axh, by, byh, bx, bxh, ny, nx, dt, fd_coeffsy, fd_coeffs2y,       \
      fd_coeffs3y, fd_coeffsx, fd_coeffs2x, fd_coeffs3x, pml_regionsy,      \
      pml_regionsx);
#define FORWARD_KERNEL_SIGMAMUGRAD(pml_y, pml_x)                               \
  forward_kernel_sigma<T, A, false, true, pml_y, pml_x>(                       \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,           \
      dvydy_store + (t / step_ratio) * ny * nx,                                \
      dvxdx_store + (t / step_ratio) * ny * nx,                                \
      dvydxdvxdy_store + (t / step_ratio) * ny * nx, lamb, mu, ay, ayh, ax,    \
      axh, by, byh, bx, bxh, ny, nx, dt, fd_coeffsy, fd_coeffs2y, fd_coeffs3y, \
      fd_coeffsx, fd_coeffs2x, fd_coeffs3x, pml_regionsy, pml_regionsx);
#define FORWARD_KERNEL_SIGMALAMBMUGRAD(pml_y, pml_x)                           \
  forward_kernel_sigma<T, A, true, true, pml_y, pml_x>(                        \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,           \
      dvydy_store + (t / step_ratio) * ny * nx,                                \
      dvxdx_store + (t / step_ratio) * ny * nx,                                \
      dvydxdvxdy_store + (t / step_ratio) * ny * nx, lamb, mu, ay, ayh, ax,    \
      axh, by, byh, bx, bxh, ny, nx, dt, fd_coeffsy, fd_coeffs2y, fd_coeffs3y, \
      fd_coeffsx, fd_coeffs2x, fd_coeffs3x, pml_regionsy, pml_regionsx);
#define FORWARD_KERNEL_SIGMANOGRAD(pml_y, pml_x)                              \
  forward_kernel_sigma<T, A, false, false, pml_y, pml_x>(                     \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx, nullptr, \
      nullptr, nullptr, lamb, mu, ay, ayh, ax, axh, by, byh, bx, bxh, ny, nx, \
      dt, fd_coeffsy, fd_coeffs2y, fd_coeffs3y, fd_coeffsx, fd_coeffs2x,      \
      fd_coeffs3x, pml_regionsy, pml_regionsx);
  for (int64_t t{}; t < nt; ++t) {
    if (t % step_ratio == 0 and buoyancy_requires_grad) {
      FORWARD_KERNEL_VGRAD(0, 0);
      FORWARD_KERNEL_VGRAD(0, 1);
      FORWARD_KERNEL_VGRAD(0, 2);
      FORWARD_KERNEL_VGRAD(1, 0);
      FORWARD_KERNEL_VGRAD(1, 1);
      FORWARD_KERNEL_VGRAD(1, 2);
      FORWARD_KERNEL_VGRAD(2, 0);
      FORWARD_KERNEL_VGRAD(2, 1);
      FORWARD_KERNEL_VGRAD(2, 2);
    } else {
      FORWARD_KERNEL_VNOGRAD(0, 0);
      FORWARD_KERNEL_VNOGRAD(0, 1);
      FORWARD_KERNEL_VNOGRAD(0, 2);
      FORWARD_KERNEL_VNOGRAD(1, 0);
      FORWARD_KERNEL_VNOGRAD(1, 1);
      FORWARD_KERNEL_VNOGRAD(1, 2);
      FORWARD_KERNEL_VNOGRAD(2, 0);
      FORWARD_KERNEL_VNOGRAD(2, 1);
      FORWARD_KERNEL_VNOGRAD(2, 2);
    }
    if (n_sources_per_shot_y > 0) {
      add_sources(vy, f_y + t * n_sources_per_shot_y, sources_y_i,
                  n_sources_per_shot_y);
    }
    if (n_sources_per_shot_x > 0) {
      add_sources(vx, f_x + t * n_sources_per_shot_x, sources_x_i,
                  n_sources_per_shot_x);
    }
    if (n_receivers_per_shot_y > 0) {
      record_receivers(r_y + t * n_receivers_per_shot_y, vy, receivers_y_i,
                       n_receivers_per_shot_y);
    }
    if (n_receivers_per_shot_x > 0) {
      record_receivers(r_x + t * n_receivers_per_shot_x, vx, receivers_x_i,
                       n_receivers_per_shot_x);
    }
    if (t % step_ratio == 0 and (lamb_requires_grad or mu_requires_grad)) {
      if (lamb_requires_grad and mu_requires_grad) {
        FORWARD_KERNEL_SIGMALAMBMUGRAD(0, 0);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(0, 1);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(0, 2);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(1, 0);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(1, 1);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(1, 2);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(2, 0);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(2, 1);
        FORWARD_KERNEL_SIGMALAMBMUGRAD(2, 2);
      } else if (lamb_requires_grad) {
        FORWARD_KERNEL_SIGMALAMBGRAD(0, 0);
        FORWARD_KERNEL_SIGMALAMBGRAD(0, 1);
        FORWARD_KERNEL_SIGMALAMBGRAD(0, 2);
        FORWARD_KERNEL_SIGMALAMBGRAD(1, 0);
        FORWARD_KERNEL_SIGMALAMBGRAD(1, 1);
        FORWARD_KERNEL_SIGMALAMBGRAD(1, 2);
        FORWARD_KERNEL_SIGMALAMBGRAD(2, 0);
        FORWARD_KERNEL_SIGMALAMBGRAD(2, 1);
        FORWARD_KERNEL_SIGMALAMBGRAD(2, 2);
      } else {
        FORWARD_KERNEL_SIGMAMUGRAD(0, 0);
        FORWARD_KERNEL_SIGMAMUGRAD(0, 1);
        FORWARD_KERNEL_SIGMAMUGRAD(0, 2);
        FORWARD_KERNEL_SIGMAMUGRAD(1, 0);
        FORWARD_KERNEL_SIGMAMUGRAD(1, 1);
        FORWARD_KERNEL_SIGMAMUGRAD(1, 2);
        FORWARD_KERNEL_SIGMAMUGRAD(2, 0);
        FORWARD_KERNEL_SIGMAMUGRAD(2, 1);
        FORWARD_KERNEL_SIGMAMUGRAD(2, 2);
      }
    } else {
      FORWARD_KERNEL_SIGMANOGRAD(0, 0);
      FORWARD_KERNEL_SIGMANOGRAD(0, 1);
      FORWARD_KERNEL_SIGMANOGRAD(0, 2);
      FORWARD_KERNEL_SIGMANOGRAD(1, 0);
      FORWARD_KERNEL_SIGMANOGRAD(1, 1);
      FORWARD_KERNEL_SIGMANOGRAD(1, 2);
      FORWARD_KERNEL_SIGMANOGRAD(2, 0);
      FORWARD_KERNEL_SIGMANOGRAD(2, 1);
      FORWARD_KERNEL_SIGMANOGRAD(2, 2);
    }
  }
}

template <typename T, int A>
void combine_grad_model(T *__restrict grad, T const *__restrict grad_batch,
                        int64_t n_parallel, int64_t ny, int64_t nx) {
  int64_t nynx{ny * nx};
  for (int64_t i = 0; i < nynx; ++i) {
    for (int64_t batch{}; batch < n_parallel; ++batch) {
      grad[i] += grad_batch[batch * nynx + i];
    }
  }
}

template <typename T, int A>
void backward_shot(
    T *__restrict vy, T *__restrict vx, T *__restrict sigmayy,
    T *__restrict sigmaxy, T *__restrict sigmaxx, T *__restrict m_vyy,
    T *__restrict m_vyx, T *__restrict m_vxy, T *__restrict m_vxx,
    T *__restrict m_sigmayyy, T *__restrict m_sigmaxyy,
    T *__restrict m_sigmaxyx, T *__restrict m_sigmaxxx,
    T *__restrict m_sigmayyyn, T *__restrict m_sigmaxyyn,
    T *__restrict m_sigmaxyxn, T *__restrict m_sigmaxxxn,
    int64_t const *__restrict sources_y_i,
    int64_t const *__restrict sources_x_i,
    int64_t const *__restrict receivers_y_i,
    int64_t const *__restrict receivers_x_i, T const *__restrict dvydbuoyancy,
    T const *__restrict dvxdbuoyancy, T const *__restrict dvydy_store,
    T const *__restrict dvxdx_store, T const *__restrict dvydxdvxdy_store,
    T const *__restrict lamb, T const *__restrict mu,
    T const *__restrict buoyancy, T *__restrict f_y, T *__restrict f_x,
    T const *__restrict r_y, T const *__restrict r_x, T *__restrict grad_lamb,
    T *__restrict grad_mu, T *__restrict grad_buoyancy, T const *__restrict ay,
    T const *__restrict ayh, T const *__restrict ax, T const *__restrict axh,
    T const *__restrict by, T const *__restrict byh, T const *__restrict bx,
    T const *__restrict bxh, T dt, T const *__restrict fd_coeffsy,
    T const fd_coeffs1y[][5], T const *__restrict fd_coeffs2y,
    T const *__restrict fd_coeffs3y, T const *__restrict fd_coeffsx,
    T const fd_coeffs1x[][5], T const *__restrict fd_coeffs2x,
    T const *__restrict fd_coeffs3x, int64_t n_sources_per_shot_y,
    int64_t n_sources_per_shot_x, int64_t n_receivers_per_shot_y,
    int64_t n_receivers_per_shot_x, int64_t ny, int64_t nx, int64_t nt,
    int64_t step_ratio, bool lamb_requires_grad, bool mu_requires_grad,
    bool buoyancy_requires_grad, int64_t const *__restrict spml_regionsy,
    int64_t const *__restrict spml_regionsx,
    int64_t const *__restrict vpml_regionsy,
    int64_t const *__restrict vpml_regionsx) {
#define BACKWARD_KERNEL_SIGMA(pml_y, pml_x)                                    \
  backward_kernel_sigma<T, A, pml_y, pml_x>(                                   \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,           \
      m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, m_sigmayyyn,             \
      m_sigmaxyyn, m_sigmaxyxn, m_sigmaxxxn, lamb, mu, buoyancy, ay, ayh, ax,  \
      axh, by, byh, bx, bxh, ny, nx, dt, fd_coeffsy, fd_coeffs2y, fd_coeffs3y, \
      fd_coeffsx, fd_coeffs2x, fd_coeffs3x, spml_regionsy, spml_regionsx)
#define BACKWARD_KERNEL_V(pml_y, pml_x)                                       \
  backward_kernel_v<T, A, pml_y, pml_x>(                                      \
      vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,          \
      m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, lamb, mu, buoyancy, ay, \
      ayh, ax, axh, by, byh, bx, bxh, ny, nx, dt, fd_coeffsy, fd_coeffs1y,    \
      fd_coeffs2y, fd_coeffsx, fd_coeffs1x, fd_coeffs2x, vpml_regionsy,       \
      vpml_regionsx)

  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (t % step_ratio == 0 and lamb_requires_grad) {
      add_to_grad_lamb<T>(
          grad_lamb, sigmayy, sigmaxx, dvydy_store + (t / step_ratio) * ny * nx,
          dvxdx_store + (t / step_ratio) * ny * nx, step_ratio, ny, nx);
    }
    if (t % step_ratio == 0 and mu_requires_grad) {
      add_to_grad_mu<T>(grad_mu, sigmayy, sigmaxy, sigmaxx,
                        dvydy_store + (t / step_ratio) * ny * nx,
                        dvxdx_store + (t / step_ratio) * ny * nx,
                        dvydxdvxdy_store + (t / step_ratio) * ny * nx,
                        step_ratio, ny, nx);
    }

    if (n_receivers_per_shot_y > 0) {
      add_sources(vy, r_y + t * n_receivers_per_shot_y, receivers_y_i,
                  n_receivers_per_shot_y);
    }
    if (n_receivers_per_shot_x > 0) {
      add_sources(vx, r_x + t * n_receivers_per_shot_x, receivers_x_i,
                  n_receivers_per_shot_x);
    }
    BACKWARD_KERNEL_SIGMA(0, 0);
    BACKWARD_KERNEL_SIGMA(0, 1);
    BACKWARD_KERNEL_SIGMA(0, 2);
    BACKWARD_KERNEL_SIGMA(1, 0);
    BACKWARD_KERNEL_SIGMA(1, 1);
    BACKWARD_KERNEL_SIGMA(1, 2);
    BACKWARD_KERNEL_SIGMA(2, 0);
    BACKWARD_KERNEL_SIGMA(2, 1);
    BACKWARD_KERNEL_SIGMA(2, 2);
    if (n_sources_per_shot_y > 0) {
      record_receivers(f_y + t * n_sources_per_shot_y, vy, sources_y_i,
                       n_sources_per_shot_y);
    }
    if (n_sources_per_shot_x > 0) {
      record_receivers(f_x + t * n_sources_per_shot_x, vx, sources_x_i,
                       n_sources_per_shot_x);
    }
    if (t % step_ratio == 0 and buoyancy_requires_grad) {
      add_to_grad_buoyancy<T>(
          grad_buoyancy, vy, vx, dvydbuoyancy + (t / step_ratio) * ny * nx,
          dvxdbuoyancy + (t / step_ratio) * ny * nx, step_ratio, ny, nx);
    }

    BACKWARD_KERNEL_V(0, 0);
    BACKWARD_KERNEL_V(0, 1);
    BACKWARD_KERNEL_V(0, 2);
    BACKWARD_KERNEL_V(1, 0);
    BACKWARD_KERNEL_V(1, 1);
    BACKWARD_KERNEL_V(1, 2);
    BACKWARD_KERNEL_V(2, 0);
    BACKWARD_KERNEL_V(2, 1);
    BACKWARD_KERNEL_V(2, 2);
    std::swap(m_sigmayyyn, m_sigmayyy);
    std::swap(m_sigmaxyyn, m_sigmaxyy);
    std::swap(m_sigmaxyxn, m_sigmaxyx);
    std::swap(m_sigmaxxxn, m_sigmaxxx);
  }
}

void zero_edge_all(torch::Tensor tensor, int64_t ny, int64_t nx) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  tensor.index_put_({all_slice, ny - 1, all_slice}, 0);
  tensor.index_put_({all_slice, all_slice, nx - 1}, 0);
  tensor.index_put_({all_slice, 0, all_slice}, 0);
  tensor.index_put_({all_slice, all_slice, 0}, 0);
}

void zero_edge_top(torch::Tensor tensor) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  tensor.index_put_({all_slice, 0, all_slice}, 0);
}

void zero_edge_bottom(torch::Tensor tensor, int ny) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  tensor.index_put_({all_slice, ny - 1, all_slice}, 0);
}

void zero_edge_left(torch::Tensor tensor) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  tensor.index_put_({all_slice, all_slice, 0}, 0);
}

void zero_edge_right(torch::Tensor tensor, int nx) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  tensor.index_put_({all_slice, all_slice, nx - 1}, 0);
}

void zero_interior(torch::Tensor tensor, int64_t ybegin, int64_t yend,
                   int64_t xbegin, int64_t xend) {
  at::indexing::TensorIndex all_slice{torch::indexing::Slice()};
  at::indexing::TensorIndex slicey{torch::indexing::Slice(ybegin, yend)};
  at::indexing::TensorIndex slicex{torch::indexing::Slice(xbegin, xend)};
  tensor.index_put_({all_slice, slicey, slicex}, 0);
}

template <typename T>
void set_fd_coeffs(T fd_coeffs[], T fd_coeffs1[][5], T fd_coeffs2[5],
                   T fd_coeffs3[6], int64_t accuracy, T dx) {
  if (accuracy == 2) {
    fd_coeffs[0] = static_cast<T>(1.0 / 1.0) / dx;
    fd_coeffs1[0][0] = static_cast<T>(-8.0 / 3.0) / dx;
    fd_coeffs1[0][1] = static_cast<T>(3.0 / 1.0) / dx;
    fd_coeffs1[0][2] = static_cast<T>(-1.0 / 3.0) / dx;
  } else {
    fd_coeffs[0] = static_cast<T>(9.0 / 8.0) / dx;
    fd_coeffs[1] = static_cast<T>(-1.0 / 24.0) / dx;
    fd_coeffs1[0][0] = static_cast<T>(-352.0 / 105.0) / dx;
    fd_coeffs1[0][1] = static_cast<T>(35.0 / 8.0) / dx;
    fd_coeffs1[0][2] = static_cast<T>(-35.0 / 24.0) / dx;
    fd_coeffs1[0][3] = static_cast<T>(21.0 / 40.0) / dx;
    fd_coeffs1[0][4] = static_cast<T>(-5.0 / 56.0) / dx;
    fd_coeffs1[1][0] = static_cast<T>(16.0 / 105.0) / dx;
    fd_coeffs1[1][1] = static_cast<T>(-31.0 / 24.0) / dx;
    fd_coeffs1[1][2] = static_cast<T>(29.0 / 24.0) / dx;
    fd_coeffs1[1][3] = static_cast<T>(-3.0 / 40.0) / dx;
    fd_coeffs1[1][4] = static_cast<T>(1.0 / 168.0) / dx;
    fd_coeffs2[0] = static_cast<T>(-11.0 / 12.0) / dx;
    fd_coeffs2[1] = static_cast<T>(17.0 / 24.0) / dx;
    fd_coeffs2[2] = static_cast<T>(3.0 / 8.0) / dx;
    fd_coeffs2[3] = static_cast<T>(-5.0 / 24.0) / dx;
    fd_coeffs2[4] = static_cast<T>(1.0 / 24.0) / dx;
    fd_coeffs3[0] = static_cast<T>(-71.0 / 1689.0);
    fd_coeffs3[1] = static_cast<T>(-14587.0 / 13512.0) / dx;
    fd_coeffs3[2] = static_cast<T>(11243.0 / 10134.0) / dx;
    fd_coeffs3[3] = static_cast<T>(-43.0 / 2252.0) / dx;
    fd_coeffs3[4] = static_cast<T>(-47.0 / 3378.0) / dx;
    fd_coeffs3[5] = static_cast<T>(127.0 / 40536.0) / dx;
  }
}

}  // namespace

class ElasticCPUFunction
    : public torch::autograd::Function<ElasticCPUFunction> {
 public:
  static std::vector<torch::Tensor> forward(
      torch::autograd::AutogradContext *ctx, torch::Tensor const &lamb,
      torch::Tensor const &mu, torch::Tensor const &buoyancy,
      torch::Tensor const &f_y, torch::Tensor const &f_x,
      torch::Tensor const &vy0, torch::Tensor const &vx0,
      torch::Tensor const &sigmayy0, torch::Tensor const &sigmaxy0,
      torch::Tensor const &sigmaxx0, torch::Tensor const &m_vyy0,
      torch::Tensor const &m_vyx0, torch::Tensor const &m_vxy0,
      torch::Tensor const &m_vxx0, torch::Tensor const &m_sigmayyy0,
      torch::Tensor const &m_sigmaxyy0, torch::Tensor const &m_sigmaxyx0,
      torch::Tensor const &m_sigmaxxx0, torch::Tensor const &ay,
      torch::Tensor const &ayh, torch::Tensor const &ax,
      torch::Tensor const &axh, torch::Tensor const &by,
      torch::Tensor const &byh, torch::Tensor const &bx,
      torch::Tensor const &bxh, torch::Tensor const &sources_y_i,
      torch::Tensor const &sources_x_i, torch::Tensor const &receivers_y_i,
      torch::Tensor const &receivers_x_i, double dy, double dx, double dt,
      int64_t nt, int64_t n_batch, int64_t step_ratio, int64_t accuracy,
      int64_t pml_width0, int64_t pml_width1, int64_t pml_width2,
      int64_t pml_width3) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto options{at::device(lamb.device()).dtype(lamb.scalar_type())};
    auto ny{lamb.size(0)};
    auto nx{lamb.size(1)};
    std::array<int64_t, 3> size_with_batch{n_batch, ny, nx};
    int64_t const pml_width[4] = {pml_width0, pml_width1, pml_width2,
                                  pml_width3};
    int64_t pml_regionsy1{std::max(pml_width[0], accuracy / 2)};
    int64_t pml_regionsy2{std::min(ny - pml_width[1], ny - accuracy / 2)};
    int64_t pml_regionsy[]{pml_regionsy1, pml_regionsy2};
    int64_t pml_regionsx1{std::max(pml_width[2], accuracy / 2)};
    int64_t pml_regionsx2{std::min(nx - pml_width[3], nx - accuracy / 2)};
    int64_t pml_regionsx[]{pml_regionsx1, pml_regionsx2};

    int64_t n_sources_per_shot_y{};
    int64_t n_sources_per_shot_x{};
    if (sources_y_i.numel() > 0) {
      n_sources_per_shot_y = sources_y_i.size(1);
    }
    if (sources_x_i.numel() > 0) {
      n_sources_per_shot_x = sources_x_i.size(1);
    }
    int64_t n_receivers_per_shot_y{};
    int64_t n_receivers_per_shot_x{};
    if (receivers_y_i.numel() > 0) {
      n_receivers_per_shot_y = receivers_y_i.size(1);
    }
    if (receivers_x_i.numel() > 0) {
      n_receivers_per_shot_x = receivers_x_i.size(1);
    }
    auto vy{create_or_clone(vy0, options, size_with_batch)};
    auto vx{create_or_clone(vx0, options, size_with_batch)};
    auto sigmayy{create_or_clone(sigmayy0, options, size_with_batch)};
    auto sigmaxy{create_or_clone(sigmaxy0, options, size_with_batch)};
    auto sigmaxx{create_or_clone(sigmaxx0, options, size_with_batch)};
    auto m_vyy{create_or_clone(m_vyy0, options, size_with_batch)};
    auto m_vyx{create_or_clone(m_vyx0, options, size_with_batch)};
    auto m_vxy{create_or_clone(m_vxy0, options, size_with_batch)};
    auto m_vxx{create_or_clone(m_vxx0, options, size_with_batch)};
    auto m_sigmayyy{create_or_clone(m_sigmayyy0, options, size_with_batch)};
    auto m_sigmaxyy{create_or_clone(m_sigmaxyy0, options, size_with_batch)};
    auto m_sigmaxyx{create_or_clone(m_sigmaxyx0, options, size_with_batch)};
    auto m_sigmaxxx{create_or_clone(m_sigmaxxx0, options, size_with_batch)};
    auto r_y{at::empty({n_batch, nt, n_receivers_per_shot_y}, options)};
    auto r_x{at::empty({n_batch, nt, n_receivers_per_shot_x}, options)};
    torch::Tensor dvydbuoyancy;
    torch::Tensor dvxdbuoyancy;
    torch::Tensor dvydy_store;
    torch::Tensor dvxdx_store;
    torch::Tensor dvydxdvxdy_store;
    if (buoyancy.requires_grad()) {
      dvydbuoyancy = at::empty(
          {n_batch, (nt + step_ratio - 1) / step_ratio, ny, nx}, options);
      dvxdbuoyancy = at::empty(
          {n_batch, (nt + step_ratio - 1) / step_ratio, ny, nx}, options);
    }
    if (lamb.requires_grad() or mu.requires_grad()) {
      dvydy_store = at::empty(
          {n_batch, (nt + step_ratio - 1) / step_ratio, ny, nx}, options);
      dvxdx_store = at::empty(
          {n_batch, (nt + step_ratio - 1) / step_ratio, ny, nx}, options);
    }
    if (mu.requires_grad()) {
      dvydxdvxdy_store = at::empty(
          {n_batch, (nt + step_ratio - 1) / step_ratio, ny, nx}, options);
    }

    zero_edge_all(sigmaxy, ny, nx);
    zero_edge_all(m_vxy, ny, nx);
    zero_edge_all(m_vyx, ny, nx);
    zero_edge_right(vy, nx);
    zero_edge_right(m_sigmayyy, nx);
    zero_edge_right(m_sigmaxyx, nx);
    zero_edge_top(vx);
    zero_edge_top(m_sigmaxyy);
    zero_edge_top(m_sigmaxxx);
    zero_edge_top(sigmaxx);
    zero_edge_top(m_vxx);
    zero_edge_top(sigmayy);
    zero_edge_top(m_vyy);
    zero_edge_right(sigmaxx, nx);
    zero_edge_right(m_vxx, nx);
    zero_edge_right(sigmayy, nx);
    zero_edge_right(m_vyy, nx);
    zero_interior(m_sigmayyy, pml_regionsy[0], pml_regionsy[1], 0, nx);
    zero_interior(m_sigmaxyx, 0, ny, pml_regionsx[0], pml_regionsx[1] - 1);
    zero_interior(m_sigmaxyy, pml_regionsy[0] + 1, pml_regionsy[1], 0, nx);
    zero_interior(m_sigmaxxx, 0, ny, pml_regionsx[0], pml_regionsx[1]);
    zero_interior(m_vyy, pml_regionsy[0] + 1, pml_regionsy[1], 0, nx);
    zero_interior(m_vxx, 0, ny, pml_regionsx[0], pml_regionsx[1] - 1);
    zero_interior(m_vyx, 0, ny, pml_regionsx[0] + 1, pml_regionsx[1] - 1);
    zero_interior(m_vxy, pml_regionsy[0] + 1, pml_regionsy[1] - 1, 0, nx);

    AT_DISPATCH_FLOATING_TYPES(
        lamb.scalar_type(), "elastic_cpu_forward", ([&] {
          scalar_t dt_a = dt;
          scalar_t fd_coeffsy[2];
          scalar_t fd_coeffsx[2];
          scalar_t fd_coeffs1y[2][5];
          scalar_t fd_coeffs1x[2][5];
          scalar_t fd_coeffs2y[5];
          scalar_t fd_coeffs2x[5];
          scalar_t fd_coeffs3y[6];
          scalar_t fd_coeffs3x[6];

          scalar_t const *__restrict lamb_a{lamb.data_ptr<scalar_t>()};
          scalar_t const *__restrict mu_a{mu.data_ptr<scalar_t>()};
          scalar_t const *__restrict buoyancy_a{buoyancy.data_ptr<scalar_t>()};
          scalar_t const *__restrict f_y_a{f_y.data_ptr<scalar_t>()};
          scalar_t const *__restrict f_x_a{f_x.data_ptr<scalar_t>()};
          scalar_t *__restrict r_y_a{r_y.data_ptr<scalar_t>()};
          scalar_t *__restrict r_x_a{r_x.data_ptr<scalar_t>()};
          scalar_t *__restrict vy_a{vy.data_ptr<scalar_t>()};
          scalar_t *__restrict vx_a{vx.data_ptr<scalar_t>()};
          scalar_t *__restrict sigmayy_a{sigmayy.data_ptr<scalar_t>()};
          scalar_t *__restrict sigmaxy_a{sigmaxy.data_ptr<scalar_t>()};
          scalar_t *__restrict sigmaxx_a{sigmaxx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vyy_a{m_vyy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vyx_a{m_vyx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vxy_a{m_vxy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vxx_a{m_vxx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmayyy_a{m_sigmayyy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxyy_a{m_sigmaxyy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxyx_a{m_sigmaxyx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxxx_a{m_sigmaxxx.data_ptr<scalar_t>()};
          scalar_t const *__restrict ay_a{ay.data_ptr<scalar_t>()};
          scalar_t const *__restrict ayh_a{ayh.data_ptr<scalar_t>()};
          scalar_t const *__restrict ax_a{ax.data_ptr<scalar_t>()};
          scalar_t const *__restrict axh_a{axh.data_ptr<scalar_t>()};
          scalar_t const *__restrict by_a{by.data_ptr<scalar_t>()};
          scalar_t const *__restrict byh_a{byh.data_ptr<scalar_t>()};
          scalar_t const *__restrict bx_a{bx.data_ptr<scalar_t>()};
          scalar_t const *__restrict bxh_a{bxh.data_ptr<scalar_t>()};
          int64_t const *__restrict sources_y_i_a{
              sources_y_i.data_ptr<int64_t>()};
          int64_t const *__restrict sources_x_i_a{
              sources_x_i.data_ptr<int64_t>()};
          int64_t const *__restrict receivers_y_i_a{
              receivers_y_i.data_ptr<int64_t>()};
          int64_t const *__restrict receivers_x_i_a{
              receivers_x_i.data_ptr<int64_t>()};
          scalar_t *__restrict dvydbuoyancy_a{};
          scalar_t *__restrict dvxdbuoyancy_a{};
          scalar_t *__restrict dvydy_store_a{};
          scalar_t *__restrict dvxdx_store_a{};
          scalar_t *__restrict dvydxdvxdy_store_a{};
          if (buoyancy.requires_grad()) {
            dvydbuoyancy_a = dvydbuoyancy.data_ptr<scalar_t>();
            dvxdbuoyancy_a = dvxdbuoyancy.data_ptr<scalar_t>();
          }
          if (lamb.requires_grad() or mu.requires_grad()) {
            dvydy_store_a = dvydy_store.data_ptr<scalar_t>();
            dvxdx_store_a = dvxdx_store.data_ptr<scalar_t>();
          }
          if (mu.requires_grad()) {
            dvydxdvxdy_store_a = dvydxdvxdy_store.data_ptr<scalar_t>();
          }
          set_fd_coeffs(fd_coeffsy, fd_coeffs1y, fd_coeffs2y, fd_coeffs3y,
                        accuracy, static_cast<scalar_t>(dy));
          set_fd_coeffs(fd_coeffsx, fd_coeffs1x, fd_coeffs2x, fd_coeffs3x,
                        accuracy, static_cast<scalar_t>(dx));
          decltype(&forward_shot<scalar_t, 4>) forward_shots[]{
              forward_shot<scalar_t, 2>, forward_shot<scalar_t, 4>};
          at::parallel_for(0, n_batch, 0, [&](int64_t bstart, int64_t bend) {
            for (int64_t shot = bstart; shot < bend; ++shot) {
              auto i{shot * ny * nx};
              auto siy{shot * n_sources_per_shot_y};
              auto six{shot * n_sources_per_shot_x};
              auto riy{shot * n_receivers_per_shot_y};
              auto rix{shot * n_receivers_per_shot_x};
              forward_shots[accuracy / 2 - 1](
                  vy_a + i, vx_a + i, sigmayy_a + i, sigmaxy_a + i,
                  sigmaxx_a + i, m_vyy_a + i, m_vyx_a + i, m_vxy_a + i,
                  m_vxx_a + i, m_sigmayyy_a + i, m_sigmaxyy_a + i,
                  m_sigmaxyx_a + i, m_sigmaxxx_a + i, sources_y_i_a + siy,
                  sources_x_i_a + six, receivers_y_i_a + riy,
                  receivers_x_i_a + rix, dvydbuoyancy_a + i * (nt / step_ratio),
                  dvxdbuoyancy_a + i * (nt / step_ratio),
                  dvydy_store_a + i * (nt / step_ratio),
                  dvxdx_store_a + i * (nt / step_ratio),
                  dvydxdvxdy_store_a + i * (nt / step_ratio), lamb_a, mu_a,
                  buoyancy_a, f_y_a + siy * nt, f_x_a + six * nt,
                  r_y_a + riy * nt, r_x_a + rix * nt, ay_a, ayh_a, ax_a, axh_a,
                  by_a, byh_a, bx_a, bxh_a, dt_a, fd_coeffsy, fd_coeffs1y,
                  fd_coeffs2y, fd_coeffs3y, fd_coeffsx, fd_coeffs1x,
                  fd_coeffs2x, fd_coeffs3x, n_sources_per_shot_y,
                  n_sources_per_shot_x, n_receivers_per_shot_y,
                  n_receivers_per_shot_x, ny, nx, nt, step_ratio,
                  lamb.requires_grad(), mu.requires_grad(),
                  buoyancy.requires_grad(), pml_regionsy, pml_regionsx);
            }
          });
        }));

    if (lamb.requires_grad() or mu.requires_grad() or
        buoyancy.requires_grad() or f_y.requires_grad() or
        f_x.requires_grad() or vy0.requires_grad() or vx0.requires_grad() or
        sigmayy0.requires_grad() or sigmaxy0.requires_grad() or
        sigmaxx0.requires_grad() or m_vyy0.requires_grad() or
        m_vyx0.requires_grad() or m_vxy0.requires_grad() or
        m_vxx0.requires_grad() or m_sigmayyy0.requires_grad() or
        m_sigmaxyy0.requires_grad() or m_sigmaxyx0.requires_grad() or
        m_sigmaxxx0.requires_grad()) {
      ctx->save_for_backward({lamb, mu, buoyancy, ay, ayh, ax, axh, by, byh, bx,
                              bxh, sources_y_i, sources_x_i, receivers_y_i,
                              receivers_x_i});
      ctx->saved_data["dvydbuoyancy"] = dvydbuoyancy;
      ctx->saved_data["dvxdbuoyancy"] = dvxdbuoyancy;
      ctx->saved_data["dvydy_store"] = dvydy_store;
      ctx->saved_data["dvxdx_store"] = dvxdx_store;
      ctx->saved_data["dvydxdvxdy_store"] = dvydxdvxdy_store;
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
    return {vy,         vx,         sigmayy,    sigmaxy, sigmaxx,
            m_vyy,      m_vyx,      m_vxy,      m_vxx,   m_sigmayyy,
            m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, r_y,     r_x};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::tensor_list const &grad_outputs) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto saved{ctx->get_saved_variables()};
    auto const &lamb{saved[0]};
    auto const &mu{saved[1]};
    auto const &buoyancy{saved[2]};
    auto const &ay{saved[3]};
    auto const &ayh{saved[4]};
    auto const &ax{saved[5]};
    auto const &axh{saved[6]};
    auto const &by{saved[7]};
    auto const &byh{saved[8]};
    auto const &bx{saved[9]};
    auto const &bxh{saved[10]};
    auto const &sources_y_i{saved[11]};
    auto const &sources_x_i{saved[12]};
    auto const &receivers_y_i{saved[13]};
    auto const &receivers_x_i{saved[14]};
    auto const &dvydbuoyancy{ctx->saved_data["dvydbuoyancy"].toTensor()};
    auto const &dvxdbuoyancy{ctx->saved_data["dvxdbuoyancy"].toTensor()};
    auto const &dvydy_store{ctx->saved_data["dvydy_store"].toTensor()};
    auto const &dvxdx_store{ctx->saved_data["dvxdx_store"].toTensor()};
    auto const &dvydxdvxdy_store{
        ctx->saved_data["dvydxdvxdy_store"].toTensor()};
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
    auto ny{lamb.size(0)};
    auto nx{lamb.size(1)};
    int64_t pml_regionsy1{std::max(pml_width[0], accuracy / 2)};
    int64_t pml_regionsy2{std::min(ny - pml_width[1], ny - accuracy / 2)};
    int64_t pml_regionsy[]{pml_regionsy1, pml_regionsy2};
    int64_t pml_regionsx1{std::max(pml_width[2], accuracy / 2)};
    int64_t pml_regionsx2{std::min(nx - pml_width[3], nx - accuracy / 2)};
    int64_t pml_regionsx[]{pml_regionsx1, pml_regionsx2};
    int64_t spml_regions1y{std::max(pml_width[0], accuracy + 1)};
    int64_t spml_regions2y{std::min(ny - pml_width[1], ny - (accuracy + 1))};
    int64_t spml_regions1x{std::max(pml_width[2], accuracy + 1)};
    int64_t spml_regions2x{std::min(nx - pml_width[3], nx - (accuracy + 1))};
    int64_t spml_regionsy[2]{spml_regions1y, spml_regions2y};
    int64_t spml_regionsx[2]{spml_regions1x, spml_regions2x};
    int64_t vpml_regions1y{std::max(pml_width[0], accuracy)};
    int64_t vpml_regions2y{std::min(ny - pml_width[1], ny - accuracy)};
    int64_t vpml_regions1x{std::max(pml_width[2], accuracy)};
    int64_t vpml_regions2x{std::min(nx - pml_width[3], nx - accuracy)};
    int64_t vpml_regionsy[2]{vpml_regions1y, vpml_regions2y};
    int64_t vpml_regionsx[2]{vpml_regions1x, vpml_regions2x};
    auto vy{at::clone(grad_outputs[0])};
    auto vx{at::clone(grad_outputs[1])};
    auto sigmayy{at::clone(grad_outputs[2])};
    auto sigmaxy{at::clone(grad_outputs[3])};
    auto sigmaxx{at::clone(grad_outputs[4])};
    auto m_vyy{at::clone(grad_outputs[5])};
    auto m_vyx{at::clone(grad_outputs[6])};
    auto m_vxy{at::clone(grad_outputs[7])};
    auto m_vxx{at::clone(grad_outputs[8])};
    auto m_sigmayyy{at::clone(grad_outputs[9])};
    auto m_sigmaxyy{at::clone(grad_outputs[10])};
    auto m_sigmaxyx{at::clone(grad_outputs[11])};
    auto m_sigmaxxx{at::clone(grad_outputs[12])};
    auto m_sigmayyyn{at::zeros_like(m_sigmayyy)};
    auto m_sigmaxyyn{at::zeros_like(m_sigmaxyy)};
    auto m_sigmaxyxn{at::zeros_like(m_sigmaxyx)};
    auto m_sigmaxxxn{at::zeros_like(m_sigmaxxx)};
    auto grad_r_y{grad_outputs[13].contiguous()};
    auto grad_r_x{grad_outputs[14].contiguous()};
    auto options{at::device(vy.device()).dtype(vy.scalar_type())};
    int64_t n_sources_per_shot_y{};
    int64_t n_sources_per_shot_x{};
    if (sources_y_i.numel() > 0) {
      n_sources_per_shot_y = sources_y_i.size(1);
    }
    if (sources_x_i.numel() > 0) {
      n_sources_per_shot_x = sources_x_i.size(1);
    }
    int64_t n_receivers_per_shot_y{};
    int64_t n_receivers_per_shot_x{};
    if (receivers_y_i.numel() > 0) {
      n_receivers_per_shot_y = receivers_y_i.size(1);
    }
    if (receivers_x_i.numel() > 0) {
      n_receivers_per_shot_x = receivers_x_i.size(1);
    }

    int64_t n_parallel{
        std::min(static_cast<int>(n_batch), at::get_num_threads())};
    auto grad_lamb{at::zeros({ny, nx}, options)};
    auto grad_mu{at::zeros({ny, nx}, options)};
    auto grad_buoyancy{at::zeros({ny, nx}, options)};
    auto grad_lamb_batch{n_parallel > 1
                             ? at::zeros({n_parallel, ny, nx}, options)
                             : at::empty(0)};
    auto grad_mu_batch{n_parallel > 1 ? at::zeros({n_parallel, ny, nx}, options)
                                      : at::empty(0)};
    auto grad_buoyancy_batch{n_parallel > 1
                                 ? at::zeros({n_parallel, ny, nx}, options)
                                 : at::empty(0)};
    auto grad_f_y{at::empty({n_batch, nt, n_sources_per_shot_y}, options)};
    auto grad_f_x{at::empty({n_batch, nt, n_sources_per_shot_x}, options)};

    zero_edge_all(sigmaxy, ny, nx);
    zero_edge_all(m_vxy, ny, nx);
    zero_edge_all(m_vyx, ny, nx);
    zero_edge_right(vy, nx);
    zero_edge_right(m_sigmayyy, nx);
    zero_edge_right(m_sigmaxyx, nx);
    zero_edge_top(vx);
    zero_edge_top(m_sigmaxyy);
    zero_edge_top(m_sigmaxxx);
    zero_edge_top(sigmaxx);
    zero_edge_top(m_vxx);
    zero_edge_top(sigmayy);
    zero_edge_top(m_vyy);
    zero_edge_right(sigmaxx, nx);
    zero_edge_right(m_vxx, nx);
    zero_edge_right(sigmayy, nx);
    zero_edge_right(m_vyy, nx);
    zero_interior(m_sigmayyy, pml_regionsy[0], pml_regionsy[1], 0, nx);
    zero_interior(m_sigmaxyx, 0, ny, pml_regionsx[0], pml_regionsx[1] - 1);
    zero_interior(m_sigmaxyy, pml_regionsy[0] + 1, pml_regionsy[1], 0, nx);
    zero_interior(m_sigmaxxx, 0, ny, pml_regionsx[0], pml_regionsx[1]);
    zero_interior(m_vyy, pml_regionsy[0] + 1, pml_regionsy[1], 0, nx);
    zero_interior(m_vxx, 0, ny, pml_regionsx[0], pml_regionsx[1] - 1);
    zero_interior(m_vyx, 0, ny, pml_regionsx[0] + 1, pml_regionsx[1] - 1);
    zero_interior(m_vxy, pml_regionsy[0] + 1, pml_regionsy[1] - 1, 0, nx);

    AT_DISPATCH_FLOATING_TYPES(
        vy.scalar_type(), "elastic_cpu_backward", ([&] {
          scalar_t dt_a = dt;
          scalar_t fd_coeffsy[2];
          scalar_t fd_coeffsx[2];
          scalar_t fd_coeffs1y[2][5];
          scalar_t fd_coeffs1x[2][5];
          scalar_t fd_coeffs2y[5];
          scalar_t fd_coeffs2x[5];
          scalar_t fd_coeffs3y[6];
          scalar_t fd_coeffs3x[6];
          scalar_t const *__restrict lamb_a{lamb.data_ptr<scalar_t>()};
          scalar_t const *__restrict mu_a{mu.data_ptr<scalar_t>()};
          scalar_t const *__restrict buoyancy_a{buoyancy.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_lamb_a{grad_lamb.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_mu_a{grad_mu.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_buoyancy_a{
              grad_buoyancy.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_lamb_batch_a{
              n_parallel > 1 ? grad_lamb_batch.data_ptr<scalar_t>()
                             : grad_lamb_a};
          scalar_t *__restrict grad_mu_batch_a{
              n_parallel > 1 ? grad_mu_batch.data_ptr<scalar_t>() : grad_mu_a};
          scalar_t *__restrict grad_buoyancy_batch_a{
              n_parallel > 1 ? grad_buoyancy_batch.data_ptr<scalar_t>()
                             : grad_buoyancy_a};
          scalar_t const *__restrict grad_r_y_a{grad_r_y.data_ptr<scalar_t>()};
          scalar_t const *__restrict grad_r_x_a{grad_r_x.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_f_y_a{grad_f_y.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_f_x_a{grad_f_x.data_ptr<scalar_t>()};
          scalar_t *__restrict vy_a{vy.data_ptr<scalar_t>()};
          scalar_t *__restrict vx_a{vx.data_ptr<scalar_t>()};
          scalar_t *__restrict sigmayy_a{sigmayy.data_ptr<scalar_t>()};
          scalar_t *__restrict sigmaxy_a{sigmaxy.data_ptr<scalar_t>()};
          scalar_t *__restrict sigmaxx_a{sigmaxx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vyy_a{m_vyy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vyx_a{m_vyx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vxy_a{m_vxy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_vxx_a{m_vxx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmayyy_a{m_sigmayyy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxyy_a{m_sigmaxyy.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxyx_a{m_sigmaxyx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxxx_a{m_sigmaxxx.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmayyyn_a{m_sigmayyyn.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxyyn_a{m_sigmaxyyn.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxyxn_a{m_sigmaxyxn.data_ptr<scalar_t>()};
          scalar_t *__restrict m_sigmaxxxn_a{m_sigmaxxxn.data_ptr<scalar_t>()};
          scalar_t const *__restrict ay_a{ay.data_ptr<scalar_t>()};
          scalar_t const *__restrict ayh_a{ayh.data_ptr<scalar_t>()};
          scalar_t const *__restrict ax_a{ax.data_ptr<scalar_t>()};
          scalar_t const *__restrict axh_a{axh.data_ptr<scalar_t>()};
          scalar_t const *__restrict by_a{by.data_ptr<scalar_t>()};
          scalar_t const *__restrict byh_a{byh.data_ptr<scalar_t>()};
          scalar_t const *__restrict bx_a{bx.data_ptr<scalar_t>()};
          scalar_t const *__restrict bxh_a{bxh.data_ptr<scalar_t>()};
          int64_t const *__restrict sources_y_i_a{
              sources_y_i.data_ptr<int64_t>()};
          int64_t const *__restrict sources_x_i_a{
              sources_x_i.data_ptr<int64_t>()};
          int64_t const *__restrict receivers_y_i_a{
              receivers_y_i.data_ptr<int64_t>()};
          int64_t const *__restrict receivers_x_i_a{
              receivers_x_i.data_ptr<int64_t>()};
          scalar_t const *__restrict dvydbuoyancy_a{};
          scalar_t const *__restrict dvxdbuoyancy_a{};
          scalar_t const *__restrict dvydy_store_a{};
          scalar_t const *__restrict dvxdx_store_a{};
          scalar_t const *__restrict dvydxdvxdy_store_a{};
          if (lamb.requires_grad() or mu.requires_grad()) {
            dvydy_store_a = dvydy_store.data_ptr<scalar_t>();
            dvxdx_store_a = dvxdx_store.data_ptr<scalar_t>();
          }
          if (mu.requires_grad()) {
            dvydxdvxdy_store_a = dvydxdvxdy_store.data_ptr<scalar_t>();
          }
          if (buoyancy.requires_grad()) {
            dvydbuoyancy_a = dvydbuoyancy.data_ptr<scalar_t>();
            dvxdbuoyancy_a = dvxdbuoyancy.data_ptr<scalar_t>();
          }
          set_fd_coeffs(fd_coeffsy, fd_coeffs1y, fd_coeffs2y, fd_coeffs3y,
                        accuracy, static_cast<scalar_t>(dy));
          set_fd_coeffs(fd_coeffsx, fd_coeffs1x, fd_coeffs2x, fd_coeffs3x,
                        accuracy, static_cast<scalar_t>(dx));
          decltype(&backward_shot<scalar_t, 4>) backward_shots[]{
              backward_shot<scalar_t, 2>, backward_shot<scalar_t, 4>};
          at::parallel_for(0, n_batch, 0, [&](int64_t bstart, int64_t bend) {
            for (int64_t shot = bstart; shot < bend; ++shot) {
              auto i{shot * ny * nx};
              auto siy{shot * n_sources_per_shot_y};
              auto six{shot * n_sources_per_shot_x};
              auto riy{shot * n_receivers_per_shot_y};
              auto rix{shot * n_receivers_per_shot_x};
              backward_shots[accuracy / 2 - 1](
                  vy_a + i, vx_a + i, sigmayy_a + i, sigmaxy_a + i,
                  sigmaxx_a + i, m_vyy_a + i, m_vyx_a + i, m_vxy_a + i,
                  m_vxx_a + i, m_sigmayyy_a + i, m_sigmaxyy_a + i,
                  m_sigmaxyx_a + i, m_sigmaxxx_a + i, m_sigmayyyn_a + i,
                  m_sigmaxyyn_a + i, m_sigmaxyxn_a + i, m_sigmaxxxn_a + i,
                  sources_y_i_a + siy, sources_x_i_a + six,
                  receivers_y_i_a + riy, receivers_x_i_a + rix,
                  dvydbuoyancy_a + i * (nt / step_ratio),
                  dvxdbuoyancy_a + i * (nt / step_ratio),
                  dvydy_store_a + i * (nt / step_ratio),
                  dvxdx_store_a + i * (nt / step_ratio),
                  dvydxdvxdy_store_a + i * (nt / step_ratio), lamb_a, mu_a,
                  buoyancy_a, grad_f_y_a + siy * nt, grad_f_x_a + six * nt,
                  grad_r_y_a + riy * nt, grad_r_x_a + rix * nt,
                  grad_lamb_batch_a + at::get_thread_num() * ny * nx,
                  grad_mu_batch_a + at::get_thread_num() * ny * nx,
                  grad_buoyancy_batch_a + at::get_thread_num() * ny * nx, ay_a,
                  ayh_a, ax_a, axh_a, by_a, byh_a, bx_a, bxh_a, dt_a,
                  fd_coeffsy, fd_coeffs1y, fd_coeffs2y, fd_coeffs3y, fd_coeffsx,
                  fd_coeffs1x, fd_coeffs2x, fd_coeffs3x, n_sources_per_shot_y,
                  n_sources_per_shot_x, n_receivers_per_shot_y,
                  n_receivers_per_shot_x, ny, nx, nt, step_ratio,
                  lamb.requires_grad(), mu.requires_grad(),
                  buoyancy.requires_grad(), spml_regionsy, spml_regionsx,
                  vpml_regionsy, vpml_regionsx);
            }
          });
          decltype(&combine_grad_model<scalar_t, 4>) combine_grad_models[]{
              combine_grad_model<scalar_t, 2>, combine_grad_model<scalar_t, 4>};
          auto combine_grad_modeli{combine_grad_models[accuracy / 2 - 1]};
          if (lamb.requires_grad() and n_parallel > 1) {
            combine_grad_modeli(grad_lamb_a, grad_lamb_batch_a, n_parallel, ny,
                                nx);
          }
          if (mu.requires_grad() and n_parallel > 1) {
            combine_grad_modeli(grad_mu_a, grad_mu_batch_a, n_parallel, ny, nx);
          }
          if (buoyancy.requires_grad() and n_parallel > 1) {
            combine_grad_modeli(grad_buoyancy_a, grad_buoyancy_batch_a,
                                n_parallel, ny, nx);
          }
        }));

    zero_interior(((nt & 1) ? m_sigmayyyn : m_sigmayyy), pml_regionsy[0],
                  pml_regionsy[1], 0, nx);
    zero_interior(((nt & 1) ? m_sigmaxyxn : m_sigmaxyx), 0, ny, pml_regionsx[0],
                  pml_regionsx[1] - 1);
    zero_interior(((nt & 1) ? m_sigmaxyyn : m_sigmaxyy), pml_regionsy[0] + 1,
                  pml_regionsy[1], 0, nx);
    zero_interior(((nt & 1) ? m_sigmaxxxn : m_sigmaxxx), 0, ny, pml_regionsx[0],
                  pml_regionsx[1]);
    zero_interior(m_vyy, pml_regionsy[0] + 1, pml_regionsy[1], 0, nx);
    zero_interior(m_vxx, 0, ny, pml_regionsx[0], pml_regionsx[1] - 1);
    zero_interior(m_vyx, 0, ny, pml_regionsx[0] + 1, pml_regionsx[1] - 1);
    zero_interior(m_vxy, pml_regionsy[0] + 1, pml_regionsy[1] - 1, 0, nx);

    return {grad_lamb,
            grad_mu,
            grad_buoyancy,
            grad_f_y,
            grad_f_x,
            vy,
            vx,
            sigmayy,
            sigmaxy,
            sigmaxx,
            m_vyy,
            m_vyx,
            m_vxy,
            m_vxx,
            ((nt & 1) ? m_sigmayyyn : m_sigmayyy),
            ((nt & 1) ? m_sigmaxyyn : m_sigmaxyy),
            ((nt & 1) ? m_sigmaxyxn : m_sigmaxyx),
            ((nt & 1) ? m_sigmaxxxn : m_sigmaxxx),
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
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

std::vector<torch::Tensor> elastic_cpu_autograd(
    torch::Tensor const &lamb, torch::Tensor const &mu,
    torch::Tensor const &buoyancy, torch::Tensor const &f_y,
    torch::Tensor const &f_x, torch::Tensor const &vy0,
    torch::Tensor const &vx0, torch::Tensor const &sigmayy0,
    torch::Tensor const &sigmaxy0, torch::Tensor const &sigmaxx0,
    torch::Tensor const &m_vyy0, torch::Tensor const &m_vyx0,
    torch::Tensor const &m_vxy0, torch::Tensor const &m_vxx0,
    torch::Tensor const &m_sigmayyy0, torch::Tensor const &m_sigmaxyy0,
    torch::Tensor const &m_sigmaxyx0, torch::Tensor const &m_sigmaxxx0,
    torch::Tensor const &ay, torch::Tensor const &ayh, torch::Tensor const &ax,
    torch::Tensor const &axh, torch::Tensor const &by, torch::Tensor const &byh,
    torch::Tensor const &bx, torch::Tensor const &bxh,
    torch::Tensor const &sources_y_i, torch::Tensor const &sources_x_i,
    torch::Tensor const &receivers_y_i, torch::Tensor const &receivers_x_i,
    double dy, double dx, double dt, int64_t nt, int64_t n_batch,
    int64_t step_ratio, int64_t accuracy, int64_t pml_width0,
    int64_t pml_width1, int64_t pml_width2, int64_t pml_width3) {
  return ElasticCPUFunction::apply(
      lamb, mu, buoyancy, f_y, f_x, vy0, vx0, sigmayy0, sigmaxy0, sigmaxx0,
      m_vyy0, m_vyx0, m_vxy0, m_vxx0, m_sigmayyy0, m_sigmaxyy0, m_sigmaxyx0,
      m_sigmaxxx0, ay, ayh, ax, axh, by, byh, bx, bxh, sources_y_i, sources_x_i,
      receivers_y_i, receivers_x_i, dy, dx, dt, nt, n_batch, step_ratio,
      accuracy, pml_width0, pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCPU, m) {
  m.impl("elastic", elastic_cpu_autograd);
}

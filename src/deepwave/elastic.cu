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

__constant__ char fd_coeffsyc[2 * sizeof(double)];
__constant__ char fd_coeffsxc[2 * sizeof(double)];
__constant__ char fd_coeffs1yc[2 * 5 * sizeof(double)];
__constant__ char fd_coeffs1xc[2 * 5 * sizeof(double)];
__constant__ char fd_coeffs2yc[5 * sizeof(double)];
__constant__ char fd_coeffs2xc[5 * sizeof(double)];
__constant__ char fd_coeffs3yc[6 * sizeof(double)];
__constant__ char fd_coeffs3xc[6 * sizeof(double)];
__constant__ int64_t pml_regionsyc[2];
__constant__ int64_t pml_regionsxc[2];
__constant__ int64_t spml_regionsyc[2];
__constant__ int64_t spml_regionsxc[2];
__constant__ int64_t vpml_regionsyc[2];
__constant__ int64_t vpml_regionsxc[2];
__constant__ int64_t nyc;
__constant__ int64_t nxc;
__constant__ int64_t nynxc;
__constant__ int64_t step_ratioc;
__constant__ char dt_const[sizeof(double)];

namespace {

template <typename T>
__device__ __inline__ T fd_coeffsy(int64_t i) {
  return ((T *)fd_coeffsyc)[i];
}

template <typename T>
__device__ __inline__ T fd_coeffsx(int64_t i) {
  return ((T *)fd_coeffsxc)[i];
}

template <typename T>
__device__ __inline__ T fd_coeffs1y(int64_t i, int64_t j) {
  return ((T *)fd_coeffs1yc)[i * 5 + j];
}

template <typename T>
__device__ __inline__ T fd_coeffs1x(int64_t i, int64_t j) {
  return ((T *)fd_coeffs1xc)[i * 5 + j];
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
__device__ __inline__ T fd_coeffs3y(int64_t i) {
  return ((T *)fd_coeffs3yc)[i];
}

template <typename T>
__device__ __inline__ T fd_coeffs3x(int64_t i) {
  return ((T *)fd_coeffs3xc)[i];
}

template <typename T>
__device__ __inline__ T dtc() {
  return *((T *)dt_const);
}

torch::Tensor create_or_clone(torch::Tensor const &tensor,
                              at::TensorOptions const &options,
                              std::array<int64_t, 3> const &size) {
  if (tensor.numel() == 0) {
    return at::zeros(size, options);
  } else {
    return at::clone(tensor);
  }
}

template <typename T>
T ceil_div(T numerator, T denominator) {
  return (numerator + denominator - static_cast<T>(1)) / denominator;
}

template <typename T, int A, bool buoyancy_requires_grad>
__global__ void forward_kernel_v(
    T *__restrict vy, T *__restrict vx, T const *__restrict sigmayy,
    T const *__restrict sigmaxy, T const *__restrict sigmaxx,
    T *__restrict m_sigmayyy, T *__restrict m_sigmaxyy,
    T *__restrict m_sigmaxyx, T *__restrict m_sigmaxxx,
    T *__restrict dvydbuoyancy, T *__restrict dvxdbuoyancy,
    T const *__restrict buoyancy, T const *__restrict ay,
    T const *__restrict ayh, T const *__restrict ax, T const *__restrict axh,
    T const *__restrict by, T const *__restrict byh, T const *__restrict bx,
    T const *__restrict bxh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  auto i_nobatch{y * nxc + x};
  auto i{batch * nynxc + i_nobatch};

  if (y < nyc and x < nxc - 1) {
    bool pml_y{y < pml_regionsyc[0] or y >= pml_regionsyc[1]};
    bool pml_x{x < pml_regionsxc[0] or x >= pml_regionsxc[1] - 1};
    T dsigmayydy{};
    T dsigmaxydx{};

    // dsigmaxydx
    for (int j{}; j < A / 2 - 1; ++j) {
      if (x == j) {
        for (int k{}; k < A; ++k) {
          dsigmaxydx += fd_coeffs2x<T>(1 + k) * sigmaxy[i - j + 1 + k];
        }
      } else if (x == nxc - 2 - j) {
        for (int k{}; k < A; ++k) {
          dsigmaxydx -= fd_coeffs2x<T>(1 + k) * sigmaxy[i + j - k];
        }
      }
    }
    if (x > A / 2 - 2 and x < nxc - 2 - A / 2 + 2) {
      for (int k{}; k < A / 2; ++k) {
        dsigmaxydx += fd_coeffsx<T>(k) * (sigmaxy[i + 1 + k] - sigmaxy[i - k]);
      }
    }

    // dsigmayydy
    for (int j{}; j < A / 2; ++j) {
      if (y == j) {
        for (int k{}; k < A; ++k) {
          dsigmayydy +=
              fd_coeffs1y<T>(j, 1 + k) * sigmayy[i + (1 - j + k) * nxc];
        }
      } else if (y == nyc - 1 - j) {
        for (int k{}; k < A; ++k) {
          dsigmayydy -= fd_coeffs1y<T>(j, 1 + k) * sigmayy[i + (j - k) * nxc];
        }
      }
    }
    if (y > A / 2 - 1 and y < nyc - 1 - A / 2 + 1) {
      for (int k{}; k < A / 2; ++k) {
        dsigmayydy += fd_coeffsy<T>(k) *
                      (sigmayy[i + (1 + k) * nxc] - sigmayy[i - k * nxc]);
      }
    }

    if (pml_y) {
      m_sigmayyy[i] = ayh[y] * m_sigmayyy[i] + byh[y] * dsigmayydy;
      dsigmayydy += m_sigmayyy[i];
    }
    if (pml_x) {
      m_sigmaxyx[i] = axh[x] * m_sigmaxyx[i] + bxh[x] * dsigmaxydx;
      dsigmaxydx += m_sigmaxyx[i];
    }
    T buoyancyyhxh;
    if (y == nyc - 1) {
      buoyancyyhxh = (buoyancy[i_nobatch] + buoyancy[i_nobatch + 1]) / 2;
    } else {
      buoyancyyhxh =
          (buoyancy[i_nobatch] + buoyancy[i_nobatch + 1] +
           buoyancy[i_nobatch + nxc] + buoyancy[i_nobatch + nxc + 1]) /
          4;
    }
    vy[i] += buoyancyyhxh * dtc<T>() * (dsigmayydy + dsigmaxydx);
    if (buoyancy_requires_grad) {
      dvydbuoyancy[i] = dtc<T>() * (dsigmayydy + dsigmaxydx);
    }
  }

  if (y >= 1 and y < nyc and x < nxc) {
    bool pml_y{y < pml_regionsyc[0] + 1 or y >= pml_regionsyc[1]};
    bool pml_x{x < pml_regionsxc[0] or x >= pml_regionsxc[1]};
    T dsigmaxydy{};
    T dsigmaxxdx{};

    // dsigmaxydy
    for (int j{}; j < A / 2 - 1; ++j) {
      if (y == 1 + j) {
        for (int k{}; k < A; ++k) {
          dsigmaxydy += fd_coeffs2y<T>(1 + k) * sigmaxy[i + (-j + k) * nxc];
        }
      } else if (y == nyc - 1 - j) {
        for (int k{}; k < A; ++k) {
          dsigmaxydy -= fd_coeffs2y<T>(1 + k) * sigmaxy[i + (j - k - 1) * nxc];
        }
      }
    }
    if (y > 1 + A / 2 - 2 and y < nyc - 1 - A / 2 + 2) {
      for (int k{}; k < A / 2; ++k) {
        dsigmaxydy += fd_coeffsy<T>(k) *
                      (sigmaxy[i + k * nxc] - sigmaxy[i - (k + 1) * nxc]);
      }
    }

    // dsigmaxxdx
    for (int j{}; j < A / 2; ++j) {
      if (x == j) {
        for (int k{}; k < A; ++k) {
          dsigmaxxdx += fd_coeffs1x<T>(j, 1 + k) * sigmaxx[i - j + k];
        }
      } else if (x == nxc - 1 - j) {
        for (int k{}; k < A; ++k) {
          dsigmaxxdx -= fd_coeffs1x<T>(j, 1 + k) * sigmaxx[i + (j - k - 1)];
        }
      }
    }
    if (x > A / 2 - 1 and x < nxc - 1 - A / 2 + 1) {
      for (int k{}; k < A / 2; ++k) {
        dsigmaxxdx +=
            fd_coeffsx<T>(k) * (sigmaxx[i + k] - sigmaxx[i - (1 + k)]);
      }
    }

    if (pml_y) {
      m_sigmaxyy[i] = ay[y] * m_sigmaxyy[i] + by[y] * dsigmaxydy;
      dsigmaxydy += m_sigmaxyy[i];
    }
    if (pml_x) {
      m_sigmaxxx[i] = ax[x] * m_sigmaxxx[i] + bx[x] * dsigmaxxdx;
      dsigmaxxdx += m_sigmaxxx[i];
    }
    vx[i] += buoyancy[i_nobatch] * dtc<T>() * (dsigmaxxdx + dsigmaxydy);
    if (buoyancy_requires_grad) {
      dvxdbuoyancy[i] = dtc<T>() * (dsigmaxxdx + dsigmaxydy);
    }
  }
}

template <typename T, int A, bool lamb_requires_grad, bool mu_requires_grad>
__global__ void forward_kernel_sigma(
    T const *__restrict vy, T const *__restrict vx, T *__restrict sigmayy,
    T *__restrict sigmaxy, T *__restrict sigmaxx, T *__restrict m_vyy,
    T *__restrict m_vyx, T *__restrict m_vxy, T *__restrict m_vxx,
    T *__restrict dvydy_store, T *__restrict dvxdx_store,
    T *__restrict dvydxdvxdy_store, T const *__restrict lamb,
    T const *__restrict mu, T const *__restrict ay, T const *__restrict ayh,
    T const *__restrict ax, T const *__restrict axh, T const *__restrict by,
    T const *__restrict byh, T const *__restrict bx, T const *__restrict bxh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  auto i_nobatch{y * nxc + x};
  auto i{batch * nynxc + i_nobatch};

  if (y < nyc and x < nxc - 1) {
    bool pml_y{y < pml_regionsyc[0] + 1 or y >= pml_regionsyc[1]};
    bool pml_x{x < pml_regionsxc[0] or x >= pml_regionsxc[1] - 1};
    T dvydy{};
    T dvxdx{};

    // dvydy
    for (int j{}; j < A / 2 - 1; ++j) {
      if (y == 1 + j) {
        for (int k{}; k <= A; ++k) {
          dvydy += fd_coeffs2y<T>(k) * vy[i + (-j + k - 1) * nxc];
        }
      } else if (y == nyc - 1 - j) {
        for (int k{}; k <= A; ++k) {
          dvydy -= fd_coeffs2y<T>(k) * vy[i + (j - k) * nxc];
        }
      }
    }
    if (y > 1 + A / 2 - 2 and y < nyc - 1 - A / 2 + 2) {
      for (int k{}; k < A / 2; ++k) {
        dvydy += fd_coeffsy<T>(k) * (vy[i + k * nxc] - vy[i - (k + 1) * nxc]);
      }
    }

    // dvxdx
    for (int j{}; j < A / 2 - 1; ++j) {
      if (x == j) {
        for (int k{}; k <= A; ++k) {
          dvxdx += fd_coeffs2x<T>(k) * vx[i - j + k];
        }
      } else if (x == nxc - 2 - j) {
        for (int k{}; k <= A; ++k) {
          dvxdx -= fd_coeffs2x<T>(k) * vx[i + j - k + 1];
        }
      }
    }
    if (x > A / 2 - 2 and x < nxc - 2 - A / 2 + 2) {
      for (int k{}; k < A / 2; ++k) {
        dvxdx += fd_coeffsx<T>(k) * (vx[i + 1 + k] - vx[i - k]);
      }
    }

    if (pml_y) {
      m_vyy[i] = ay[y] * m_vyy[i] + by[y] * dvydy;
      dvydy += m_vyy[i];
    }
    if (pml_x) {
      m_vxx[i] = axh[x] * m_vxx[i] + bxh[x] * dvxdx;
      dvxdx += m_vxx[i];
    }
    T lambyxh{(lamb[i_nobatch] + lamb[i_nobatch + 1]) / 2};
    T muyxh{(mu[i_nobatch] + mu[i_nobatch + 1]) / 2};
    sigmayy[i] += dtc<T>() * ((lambyxh + 2 * muyxh) * dvydy + lambyxh * dvxdx);
    sigmaxx[i] += dtc<T>() * ((lambyxh + 2 * muyxh) * dvxdx + lambyxh * dvydy);
    if (lamb_requires_grad or mu_requires_grad) {
      dvydy_store[i] = dtc<T>() * dvydy;
      dvxdx_store[i] = dtc<T>() * dvxdx;
    }
  }

  if (y < nyc - 1 and x >= 1 and x < nxc - 1) {
    bool pml_y{y < pml_regionsyc[0] + 1 or y >= pml_regionsyc[1] - 1};
    bool pml_x{x < pml_regionsxc[0] + 1 or x >= pml_regionsxc[1] - 1};
    T dvydx{};
    T dvxdy{};

    // dvxdy
    for (int j{}; j < A / 2 - 1; ++j) {
      if (y == 1 + j) {
        T dvydxp{};
        for (int jp{}; jp < A / 2 - 1; ++jp) {
          if (x == 1 + jp) {
            for (int k{}; k <= A; ++k) {
              dvydxp +=
                  fd_coeffs2x<T>(k) * vy[i - (j + 1) * nxc - (jp + 1) + k];
            }
          } else if (x == nxc - 2 - jp) {
            for (int k{}; k <= A; ++k) {
              dvydxp -= fd_coeffs2x<T>(k) * vy[i - (j + 1) * nxc + jp - k];
            }
          }
        }
        if (x > 1 + A / 2 - 2 and x < nxc - 2 - A / 2 + 2) {
          for (int k{}; k < A / 2; ++k) {
            dvydxp += fd_coeffsx<T>(k) * (vy[i - (j + 1) * nxc + k] -
                                          vy[i - (j + 1) * nxc - k - 1]);
          }
        }
        dvxdy = -fd_coeffs3y<T>(0) * dvydxp;
        for (int k{1}; k <= A + 1; ++k) {
          dvxdy += fd_coeffs3y<T>(k) * vx[i + (-j + k - 1) * nxc];
        }
      } else if (y == nyc - 2 - j) {
        T dvydxp{};
        for (int jp{}; jp < A / 2 - 1; ++jp) {
          if (x == 1 + jp) {
            for (int k{}; k <= A; ++k) {
              dvydxp +=
                  fd_coeffs2x<T>(k) * vy[i + (j + 1) * nxc - (jp + 1) + k];
            }
          } else if (x == nxc - 2 - jp) {
            for (int k{}; k <= A; ++k) {
              dvydxp -= fd_coeffs2x<T>(k) * vy[i + (j + 1) * nxc + jp - k];
            }
          }
        }
        if (x > 1 + A / 2 - 2 and x < nxc - 2 - A / 2 + 2) {
          for (int k{}; k < A / 2; ++k) {
            dvydxp += fd_coeffsx<T>(k) * (vy[i + (j + 1) * nxc + k] -
                                          vy[i + (j + 1) * nxc - k - 1]);
          }
        }
        dvxdy = fd_coeffs3y<T>(0) * dvydxp;
        for (int k{1}; k <= A + 1; ++k) {
          dvxdy -= fd_coeffs3y<T>(k) * vx[i + (j - k + 2) * nxc];
        }
      }
    }
    if (y > 1 + A / 2 - 2 and y < nyc - 2 - A / 2 + 2) {
      for (int k{}; k < A / 2; ++k) {
        dvxdy += fd_coeffsy<T>(k) * (vx[i + (k + 1) * nxc] - vx[i - k * nxc]);
      }
    }

    // dvydx
    for (int j{}; j < A / 2 - 1; ++j) {
      if (x == 1 + j) {
        T dvxdyp{};
        for (int jp{}; jp < A / 2 - 1; ++jp) {
          if (y == 1 + jp) {
            for (int k{}; k <= A; ++k) {
              dvxdyp += fd_coeffs2y<T>(k) * vx[i - (j + 1) + (-jp + k) * nxc];
            }
          } else if (y == nyc - 2 - jp) {
            for (int k{}; k <= A; ++k) {
              dvxdyp -=
                  fd_coeffs2y<T>(k) * vx[i - (j + 1) + ((jp + 1) - k) * nxc];
            }
          }
        }
        if (y > 1 + A / 2 - 2 and y < nyc - 2 - A / 2 + 2) {
          for (int k{}; k < A / 2; ++k) {
            dvxdyp += fd_coeffsy<T>(k) * (vx[i - (j + 1) + (k + 1) * nxc] -
                                          vx[i - (j + 1) - k * nxc]);
          }
        }
        dvydx = -fd_coeffs3x<T>(0) * dvxdyp;
        for (int k{1}; k <= A + 1; ++k) {
          dvydx += fd_coeffs3x<T>(k) * vy[i + (-j + k - 2)];
        }
      } else if (x == nxc - 2 - j) {
        T dvxdyp{};
        for (int jp{}; jp < A / 2 - 1; ++jp) {
          if (y == 1 + jp) {
            for (int k{}; k <= A; ++k) {
              dvxdyp += fd_coeffs2y<T>(k) * vx[i + (j + 1) + (-jp + k) * nxc];
            }
          } else if (y == nyc - 2 - jp) {
            for (int k{}; k <= A; ++k) {
              dvxdyp -=
                  fd_coeffs2y<T>(k) * vx[i + (j + 1) + (jp - k + 1) * nxc];
            }
          }
        }
        if (y > 1 + A / 2 - 2 and y < nyc - 2 - A / 2 + 2) {
          for (int k{}; k < A / 2; ++k) {
            dvxdyp += fd_coeffsy<T>(k) * (vx[i + (j + 1) + (k + 1) * nxc] -
                                          vx[i + (j + 1) + (-k) * nxc]);
          }
        }
        dvydx = fd_coeffs3x<T>(0) * dvxdyp;
        for (int k{1}; k <= A + 1; ++k) {
          dvydx -= fd_coeffs3x<T>(k) * vy[i + (j - k + 1)];
        }
      }
    }
    if (x > 1 + A / 2 - 2 and x < nxc - 2 - A / 2 + 2) {
      for (int k{}; k < A / 2; ++k) {
        dvydx += fd_coeffsx<T>(k) * (vy[i + k] - vy[i - k - 1]);
      }
    }

    if (pml_y) {
      m_vxy[i] = ayh[y] * m_vxy[i] + byh[y] * dvxdy;
      dvxdy += m_vxy[i];
    }
    if (pml_x) {
      m_vyx[i] = ax[x] * m_vyx[i] + bx[x] * dvydx;
      dvydx += m_vyx[i];
    }
    T muyhx{(mu[i_nobatch] + mu[i_nobatch + nxc]) / 2};
    sigmaxy[i] += dtc<T>() * muyhx * (dvydx + dvxdy);
    if (mu_requires_grad) {
      dvydxdvxdy_store[i] = dtc<T>() * (dvydx + dvxdy);
    }
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

template <typename T>
__global__ void add_to_grad_lamb(T *__restrict grad_lamb,
                                 T const *__restrict sigmayy,
                                 T const *__restrict sigmaxx,
                                 T const *__restrict dvydy_store,
                                 T const *__restrict dvxdx_store) {
  auto x{blockIdx.x * blockDim.x + threadIdx.x};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + 1};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nynxc + y * nxc + x};
  if (y < nyc) {
    if (x == 0) {
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2) *
          step_ratioc;
    } else if (x < nxc - 1) {
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2 +
           (sigmayy[i - 1] + sigmaxx[i - 1]) *
               (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_lamb[i] += ((sigmayy[i - 1] + sigmaxx[i - 1]) *
                       (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
                      step_ratioc;
    }
  }
}

template <typename T>
__global__ void add_to_grad_mu(T *__restrict grad_mu,
                               T const *__restrict sigmayy,
                               T const *__restrict sigmaxy,
                               T const *__restrict sigmaxx,
                               T const *__restrict dvydy_store,
                               T const *__restrict dvxdx_store,
                               T const *__restrict dvydxdvxdy_store) {
  auto x{blockIdx.x * blockDim.x + threadIdx.x};
  auto y{blockIdx.y * blockDim.y + threadIdx.y + 1};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nynxc + y * nxc + x};
  if (y == 1) {
    if (x == 0) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          step_ratioc;
    } else if (x < nxc - 1) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    step_ratioc;
    }
  } else if (y < nyc - 1) {
    if (x == 0) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          step_ratioc;
    } else if (x < nxc - 1) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2 +
           sigmaxy[i - nxc] * dvydxdvxdy_store[i - nxc] / 2) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    step_ratioc;
    }
  } else if (y == nyc - 1) {
    if (x == 0) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          step_ratioc;
    } else if (x < nxc - 1) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i - nxc] * dvydxdvxdy_store[i - nxc] / 2) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    step_ratioc;
    }
  }
}

template <typename T>
__global__ void add_to_grad_buoyancy(T *__restrict grad_buoyancy,
                                     T const *__restrict vy,
                                     T const *__restrict vx,
                                     T const *__restrict dvydbuoyancy,
                                     T const *__restrict dvxdbuoyancy) {
  auto x{blockIdx.x * blockDim.x + threadIdx.x};
  auto y{blockIdx.y * blockDim.y + threadIdx.y};
  auto batch{blockIdx.z * blockDim.z + threadIdx.z};
  auto i{batch * nynxc + y * nxc + x};
  if (y == 0) {
    if (x == 0) {
      grad_buoyancy[i] += (vy[i] * dvydbuoyancy[i] / 4) * step_ratioc;
    } else if (x < nxc - 1) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 4) * step_ratioc;
    }
  } else if (y < nyc - 1) {
    if (x == 0) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 +
           vy[i - nxc] * dvydbuoyancy[i - nxc] / 4 + vx[i] * dvxdbuoyancy[i]) *
          step_ratioc;
    } else if (x < nxc - 1) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
           vy[i - nxc] * dvydbuoyancy[i - nxc] / 4 +
           vy[i - nxc - 1] * dvydbuoyancy[i - nxc - 1] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
                           vy[i - nxc - 1] * dvydbuoyancy[i - nxc - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          step_ratioc;
    }
  } else if (y == nyc - 1) {
    if (x == 0) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 2 +
           vy[i - nxc] * dvydbuoyancy[i - nxc] / 4 + vx[i] * dvxdbuoyancy[i]) *
          step_ratioc;
    } else if (x < nxc - 1) {
      grad_buoyancy[i] +=
          (vy[i - nxc] * dvydbuoyancy[i - nxc] / 4 +
           vy[i - nxc - 1] * dvydbuoyancy[i - nxc - 1] / 4 +
           vy[i] * dvydbuoyancy[i] / 2 + vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
           vx[i] * dvxdbuoyancy[i]) *
          step_ratioc;
    } else if (x == nxc - 1) {
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
                           vy[i - nxc - 1] * dvydbuoyancy[i - nxc - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          step_ratioc;
    }
  }
}

template <typename T, int A>
__global__ void backward_kernel_sigma(
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
    T const *__restrict bxh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  auto i_nobatch{y * nxc + x};
  auto i{batch * nynxc + i_nobatch};
  if (y < nyc and x < nxc - 1) {
    bool pml_y{y < spml_regionsyc[0] or y >= spml_regionsyc[1]};
    bool pml_x{x < spml_regionsxc[0] or x >= spml_regionsxc[1] - 1};
    // from sigmayy/sigmaxx edges
    for (int k{}; k <= A; ++k) {
      for (int j{}; j < A / 2 - 1; ++j) {
        if (y == 1 + j + (-j + k - 1)) {
          T lambyxh{(lamb[i_nobatch - (-j + k - 1) * nxc] +
                     lamb[i_nobatch + 1 - (-j + k - 1) * nxc]) /
                    2};
          T muyxh{(mu[i_nobatch - (-j + k - 1) * nxc] +
                   mu[i_nobatch + 1 - (-j + k - 1) * nxc]) /
                  2};
          vy[i] +=
              fd_coeffs2y<T>(k) *
              (dtc<T>() * (1 + by[y - (-j + k - 1)]) *
                   ((lambyxh + 2 * muyxh) * sigmayy[i - (-j + k - 1) * nxc] +
                    lambyxh * sigmaxx[i - (-j + k - 1) * nxc]) +
               by[y - (-j + k - 1)] * m_vyy[i - (-j + k - 1) * nxc]);
        } else if (y == nyc - 1 - j + (j - k)) {
          T lambyxh{(lamb[i_nobatch - (j - k) * nxc] +
                     lamb[i_nobatch + 1 - (j - k) * nxc]) /
                    2};
          T muyxh{(mu[i_nobatch - (j - k) * nxc] +
                   mu[i_nobatch + 1 - (j - k) * nxc]) /
                  2};
          vy[i] -= fd_coeffs2y<T>(k) *
                   (dtc<T>() * (1 + by[y - (j - k)]) *
                        ((lambyxh + 2 * muyxh) * sigmayy[i - (j - k) * nxc] +
                         lambyxh * sigmaxx[i - (j - k) * nxc]) +
                    by[y - (j - k)] * m_vyy[i - (j - k) * nxc]);
        }
      }
    }

    // from sigmayy/sigmaxx centre
    for (int k{}; k < A / 2; ++k) {
      if (y > 1 + A / 2 - 2 + k and y < nyc - 1 - A / 2 + 2 + k) {
        T lambyxh{(lamb[i_nobatch - k * nxc] + lamb[i_nobatch + 1 - k * nxc]) /
                  2};
        T muyxh{(mu[i_nobatch - k * nxc] + mu[i_nobatch + 1 - k * nxc]) / 2};
        vy[i] += fd_coeffsy<T>(k) *
                 (dtc<T>() * (1 + by[y - k]) *
                      ((lambyxh + 2 * muyxh) * sigmayy[i - k * nxc] +
                       lambyxh * sigmaxx[i - k * nxc]) +
                  by[y - k] * m_vyy[i - k * nxc]);
      }
      if (y > 1 + A / 2 - 2 - (k + 1) and y < nyc - 1 - A / 2 + 2 - (k + 1)) {
        T lambyxh{(lamb[i_nobatch + (k + 1) * nxc] +
                   lamb[i_nobatch + 1 + (k + 1) * nxc]) /
                  2};
        T muyxh{(mu[i_nobatch + (k + 1) * nxc] +
                 mu[i_nobatch + 1 + (k + 1) * nxc]) /
                2};
        vy[i] -= fd_coeffsy<T>(k) *
                 (dtc<T>() * (1 + by[y + k + 1]) *
                      ((lambyxh + 2 * muyxh) * sigmayy[i + (k + 1) * nxc] +
                       lambyxh * sigmaxx[i + (k + 1) * nxc]) +
                  by[y + k + 1] * m_vyy[i + (k + 1) * nxc]);
      }
    }

    // from sigmaxy dvxdy
    for (int j{}; j < A / 2 - 1; ++j) {
      if (y == 1 + j - (j + 1)) {
        int64_t y2{y + (j + 1)};
        for (int k{}; k <= A; ++k) {
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (x == 1 + jp - (jp + 1) + k) {
              int64_t i2{i - (-(j + 1) * nxc - (jp + 1) + k)};
              int64_t i2_nobatch{i_nobatch - (-(j + 1) * nxc - (jp + 1) + k)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vy[i] += fd_coeffs2x<T>(k) * (-fd_coeffs3y<T>(0)) *
                       (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            } else if (x == nxc - 2 - jp + jp - k) {
              int64_t i2{i - (-(j + 1) * nxc + jp - k)};
              int64_t i2_nobatch{i_nobatch - (-(j + 1) * nxc + jp - k)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vy[i] -= fd_coeffs2x<T>(k) * (-fd_coeffs3y<T>(0)) *
                       (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
          }
        }
        for (int k{}; k < A / 2; ++k) {
          if (x > 1 + A / 2 - 2 + k and x < nxc - 2 - A / 2 + 2 + k) {
            int64_t i2{i - (-(j + 1) * nxc + k)};
            int64_t i2_nobatch{i_nobatch - (-(j + 1) * nxc + k)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vy[i] += fd_coeffsx<T>(k) * (-fd_coeffs3y<T>(0)) *
                     (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
          if (x > 1 + A / 2 - 2 - k - 1 and x < nxc - 2 - A / 2 + 2 - k - 1) {
            int64_t i2{i - (-(j + 1) * nxc - k - 1)};
            int64_t i2_nobatch{i_nobatch - (-(j + 1) * nxc - k - 1)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vy[i] -= fd_coeffsx<T>(k) * (-fd_coeffs3y<T>(0)) *
                     (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
        }

      } else if (y == nyc - 2 - j + (j + 1)) {
        int64_t y2{y - (j + 1)};
        for (int k{}; k <= A; ++k) {
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (x == 1 + jp - (jp + 1) + k) {
              int64_t i2{i - ((j + 1) * nxc - (jp + 1) + k)};
              int64_t i2_nobatch{i_nobatch - ((j + 1) * nxc - (jp + 1) + k)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vy[i] += fd_coeffs2x<T>(k) * (fd_coeffs3y<T>(0)) *
                       (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            } else if (x == nxc - 2 - jp + jp - k) {
              int64_t i2{i - ((j + 1) * nxc + jp - k)};
              int64_t i2_nobatch{i_nobatch - ((j + 1) * nxc + jp - k)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vy[i] -= fd_coeffs2x<T>(k) * (fd_coeffs3y<T>(0)) *
                       (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
          }
        }
        for (int k{}; k < A / 2; ++k) {
          if (x > 1 + A / 2 - 2 + k and x < nxc - 2 - A / 2 + 2 + k) {
            int64_t i2{i - ((j + 1) * nxc + k)};
            int64_t i2_nobatch{i_nobatch - ((j + 1) * nxc + k)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vy[i] += fd_coeffsx<T>(k) * (fd_coeffs3y<T>(0)) *
                     (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
          if (x > 1 + A / 2 - 2 - k - 1 and x < nxc - 2 - A / 2 + 2 - k - 1) {
            int64_t i2{i - ((j + 1) * nxc - k - 1)};
            int64_t i2_nobatch{i_nobatch - ((j + 1) * nxc - k - 1)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vy[i] -= fd_coeffsx<T>(k) * (fd_coeffs3y<T>(0)) *
                     (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
        }
      }
    }

    // from sigmaxy dvydx
    if (y > 0 and y < nyc - 1) {
      for (int k{1}; k <= A + 1; ++k) {
        for (int j{}; j < A / 2 - 1; ++j) {
          if (x == 1 + j + (-j + k - 2)) {
            int64_t x2{x - (-j + k - 2)};
            int64_t i2{i - (-j + k - 2)};
            int64_t i2_nobatch{i_nobatch - (-j + k - 2)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vy[i] += fd_coeffs3x<T>(k) *
                     (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                      bx[x2] * m_vyx[i2]);
          } else if (x == nxc - 2 - j + (j - k + 1)) {
            int64_t x2{x - (j - k + 1)};
            int64_t i2{i - (j - k + 1)};
            int64_t i2_nobatch{i_nobatch - (j - k + 1)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vy[i] -= fd_coeffs3x<T>(k) *
                     (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                      bx[x2] * m_vyx[i2]);
          }
        }
      }
      for (int k{}; k < A / 2; ++k) {
        if (x > 1 + A / 2 - 2 + k and x < nxc - 2 - A / 2 + 2 + k) {
          int64_t x2{x - k};
          int64_t i2{i - k};
          int64_t i2_nobatch{i_nobatch - k};
          T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
          vy[i] += fd_coeffsx<T>(k) *
                   (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                    bx[x2] * m_vyx[i2]);
        }
        if (x > 1 + A / 2 - 2 - k - 1 and x < nxc - 2 - A / 2 + 2 - k - 1) {
          int64_t x2{x + k + 1};
          int64_t i2{i + k + 1};
          int64_t i2_nobatch{i_nobatch + k + 1};
          T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
          vy[i] -= fd_coeffsx<T>(k) *
                   (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                    bx[x2] * m_vyx[i2]);
        }
      }
    }

    T buoyancyyhxh;
    if (y == nyc - 1) {
      buoyancyyhxh = (buoyancy[i_nobatch] + buoyancy[i_nobatch + 1]) / 2;
    } else {
      buoyancyyhxh =
          (buoyancy[i_nobatch] + buoyancy[i_nobatch + 1] +
           buoyancy[i_nobatch + nxc] + buoyancy[i_nobatch + nxc + 1]) /
          4;
    }

    if (pml_y) {
      m_sigmayyyn[i] =
          buoyancyyhxh * dtc<T>() * ayh[y] * vy[i] + ayh[y] * m_sigmayyy[i];
    }
    if (pml_x) {
      m_sigmaxyxn[i] =
          buoyancyyhxh * dtc<T>() * axh[x] * vy[i] + axh[x] * m_sigmaxyx[i];
    }
  }

  if (y >= 1 and y < nyc and x < nxc) {
    bool pml_y{y < spml_regionsyc[0] + 1 or y >= spml_regionsyc[1]};
    bool pml_x{x < spml_regionsxc[0] or x >= spml_regionsxc[1]};
    // from sigmayy/sigmaxx edges
    for (int k{}; k <= A; ++k) {
      for (int j{}; j < A / 2 - 1; ++j) {
        if (x == j + (-j + k)) {
          int64_t i2{i - (-j + k)};
          int64_t i2_nobatch{i_nobatch - (-j + k)};
          int64_t x2{x - (-j + k)};
          T lambyxh{(lamb[i2_nobatch] + lamb[i2_nobatch + 1]) / 2};
          T muyxh{(mu[i2_nobatch] + mu[i2_nobatch + 1]) / 2};
          vx[i] +=
              fd_coeffs2x<T>(k) * (dtc<T>() * (1 + bxh[x2]) *
                                       ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                        lambyxh * sigmayy[i2]) +
                                   bxh[x2] * m_vxx[i2]);
        } else if (x == nxc - 2 - j + (j - k + 1)) {
          int64_t i2{i - (j - k + 1)};
          int64_t i2_nobatch{i_nobatch - (j - k + 1)};
          int64_t x2{x - (j - k + 1)};
          T lambyxh{(lamb[i2_nobatch] + lamb[i2_nobatch + 1]) / 2};
          T muyxh{(mu[i2_nobatch] + mu[i2_nobatch + 1]) / 2};
          vx[i] -=
              fd_coeffs2x<T>(k) * (dtc<T>() * (1 + bxh[x2]) *
                                       ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                        lambyxh * sigmayy[i2]) +
                                   bxh[x2] * m_vxx[i2]);
        }
      }
    }

    // from sigmayy/sigmaxx centre
    for (int k{}; k < A / 2; ++k) {
      if (x > A / 2 - 2 + 1 + k and x < nxc - 2 - A / 2 + 2 + 1 + k) {
        int64_t i2{i - (1 + k)};
        int64_t i2_nobatch{i_nobatch - (1 + k)};
        int64_t x2{x - (1 + k)};
        T lambyxh{(lamb[i2_nobatch] + lamb[i2_nobatch + 1]) / 2};
        T muyxh{(mu[i2_nobatch] + mu[i2_nobatch + 1]) / 2};
        vx[i] +=
            fd_coeffsx<T>(k) *
            (dtc<T>() * (1 + bxh[x2]) *
                 ((lambyxh + 2 * muyxh) * sigmaxx[i2] + lambyxh * sigmayy[i2]) +
             bxh[x2] * m_vxx[i2]);
      }
      if (x > A / 2 - 2 - k and x < nxc - 2 - A / 2 + 2 - k) {
        int64_t i2{i + k};
        int64_t i2_nobatch{i_nobatch + k};
        int64_t x2{x + k};
        T lambyxh{(lamb[i2_nobatch] + lamb[i2_nobatch + 1]) / 2};
        T muyxh{(mu[i2_nobatch] + mu[i2_nobatch + 1]) / 2};
        vx[i] -=
            fd_coeffsx<T>(k) *
            (dtc<T>() * (1 + bxh[x2]) *
                 ((lambyxh + 2 * muyxh) * sigmaxx[i2] + lambyxh * sigmayy[i2]) +
             bxh[x2] * m_vxx[i2]);
      }
    }

    // from sigmaxy dvydx
    for (int j{}; j < A / 2 - 1; ++j) {
      if (x == 1 + j - (j + 1)) {
        int64_t x2{x + (j + 1)};
        for (int k{}; k <= A; ++k) {
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (y == 1 + jp - jp + k) {
              int64_t i2{i - (-(j + 1) + (-jp + k) * nxc)};
              int64_t i2_nobatch{i_nobatch - (-(j + 1) + (-jp + k) * nxc)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vx[i] += fd_coeffs2y<T>(k) * (-fd_coeffs3x<T>(0)) *
                       (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            } else if (y == nyc - 2 - jp + (jp + 1) - k) {
              int64_t i2{i - (-(j + 1) + ((jp + 1) - k) * nxc)};
              int64_t i2_nobatch{i_nobatch - (-(j + 1) + ((jp + 1) - k) * nxc)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vx[i] -= fd_coeffs2y<T>(k) * (-fd_coeffs3x<T>(0)) *
                       (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
          }
        }
        for (int k{}; k < A / 2; ++k) {
          if (y > 1 + A / 2 - 2 + k + 1 and y < nyc - 2 - A / 2 + 2 + k + 1) {
            int64_t i2{i - (-(j + 1) + (k + 1) * nxc)};
            int64_t i2_nobatch{i_nobatch - (-(j + 1) + (k + 1) * nxc)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vx[i] += fd_coeffsy<T>(k) * (-fd_coeffs3x<T>(0)) *
                     (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                      bx[x2] * m_vyx[i2]);
          }
          if (y > 1 + A / 2 - 2 - k and y < nyc - 2 - A / 2 + 2 - k) {
            int64_t i2{i - (-(j + 1) - k * nxc)};
            int64_t i2_nobatch{i_nobatch - (-(j + 1) - k * nxc)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vx[i] -= fd_coeffsy<T>(k) * (-fd_coeffs3x<T>(0)) *
                     (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                      bx[x2] * m_vyx[i2]);
          }
        }

      } else if (x == nxc - 2 - j + (j + 1)) {
        int64_t x2{x - (j + 1)};
        for (int k{}; k <= A; ++k) {
          for (int jp{}; jp < A / 2 - 1; ++jp) {
            if (y == 1 + jp - jp + k) {
              int64_t i2{i - ((j + 1) + (-jp + k) * nxc)};
              int64_t i2_nobatch{i_nobatch - ((j + 1) + (-jp + k) * nxc)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vx[i] += fd_coeffs2y<T>(k) * (fd_coeffs3x<T>(0)) *
                       (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            } else if (y == nyc - 2 - jp + jp - k + 1) {
              int64_t i2{i - ((j + 1) + (jp - k + 1) * nxc)};
              int64_t i2_nobatch{i_nobatch - ((j + 1) + (jp - k + 1) * nxc)};
              T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
              vx[i] -= fd_coeffs2y<T>(k) * (fd_coeffs3x<T>(0)) *
                       (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
          }
        }
        for (int k{}; k < A / 2; ++k) {
          if (y > 1 + A / 2 - 2 + k + 1 and y < nyc - 2 - A / 2 + 2 + k + 1) {
            int64_t i2{i - ((j + 1) + (k + 1) * nxc)};
            int64_t i2_nobatch{i_nobatch - ((j + 1) + (k + 1) * nxc)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vx[i] += fd_coeffsy<T>(k) * (fd_coeffs3x<T>(0)) *
                     (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                      bx[x2] * m_vyx[i2]);
          }
          if (y > 1 + A / 2 - 2 - k and y < nyc - 2 - A / 2 + 2 - k) {
            int64_t i2{i - ((j + 1) + (-k) * nxc)};
            int64_t i2_nobatch{i_nobatch - ((j + 1) + (-k) * nxc)};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vx[i] -= fd_coeffsy<T>(k) * (fd_coeffs3x<T>(0)) *
                     (muyhx * dtc<T>() * (1 + bx[x2]) * sigmaxy[i2] +
                      bx[x2] * m_vyx[i2]);
          }
        }
      }
    }

    // from sigmaxy dvxdy
    if (x > 0 and x < nxc - 1) {
      for (int k{1}; k <= A + 1; ++k) {
        for (int j{}; j < A / 2 - 1; ++j) {
          if (y == 1 + j + (-j + k - 1)) {
            int64_t y2{y - (-j + k - 1)};
            int64_t i2{i - (-j + k - 1) * nxc};
            int64_t i2_nobatch{i_nobatch - (-j + k - 1) * nxc};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vx[i] += fd_coeffs3y<T>(k) *
                     (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          } else if (y == nyc - 2 - j + (j - k + 2)) {
            int64_t y2{y - (j - k + 2)};
            int64_t i2{i - (j - k + 2) * nxc};
            int64_t i2_nobatch{i_nobatch - (j - k + 2) * nxc};
            T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
            vx[i] -= fd_coeffs3y<T>(k) *
                     (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
        }
      }
      for (int k{}; k < A / 2; ++k) {
        if (y > 1 + A / 2 - 2 + k + 1 and y < nyc - 2 - A / 2 + 2 + k + 1) {
          int64_t y2{y - (k + 1)};
          int64_t i2{i - (k + 1) * nxc};
          int64_t i2_nobatch{i_nobatch - (k + 1) * nxc};
          T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
          vx[i] += fd_coeffsy<T>(k) *
                   (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                    byh[y2] * m_vxy[i2]);
        }
        if (y > 1 + A / 2 - 2 - k and y < nyc - 2 - A / 2 + 2 - k) {
          int64_t y2{y + k};
          int64_t i2{i + k * nxc};
          int64_t i2_nobatch{i_nobatch + k * nxc};
          T muyhx{(mu[i2_nobatch] + mu[i2_nobatch + nxc]) / 2};
          vx[i] -= fd_coeffsy<T>(k) *
                   (muyhx * dtc<T>() * (1 + byh[y2]) * sigmaxy[i2] +
                    byh[y2] * m_vxy[i2]);
        }
      }
    }

    if (pml_y) {
      m_sigmaxyyn[i] = buoyancy[i_nobatch] * dtc<T>() * ay[y] * vx[i] +
                       ay[y] * m_sigmaxyy[i];
    }
    if (pml_x) {
      m_sigmaxxxn[i] = buoyancy[i_nobatch] * dtc<T>() * ax[x] * vx[i] +
                       ax[x] * m_sigmaxxx[i];
    }
  }
}

template <typename T, int A>
__global__ void backward_kernel_v(
    T const *__restrict vy, T const *__restrict vx, T *__restrict sigmayy,
    T *__restrict sigmaxy, T *__restrict sigmaxx, T *__restrict m_vyy,
    T *__restrict m_vyx, T *__restrict m_vxy, T *__restrict m_vxx,
    T const *__restrict m_sigmayyy, T const *__restrict m_sigmaxyy,
    T const *__restrict m_sigmaxyx, T const *__restrict m_sigmaxxx,
    T const *__restrict lamb, T const *__restrict mu,
    T const *__restrict buoyancy, T const *__restrict ay,
    T const *__restrict ayh, T const *__restrict ax, T const *__restrict axh,
    T const *__restrict by, T const *__restrict byh, T const *__restrict bx,
    T const *__restrict bxh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  auto i_nobatch{y * nxc + x};
  auto i{batch * nynxc + i_nobatch};
  if (y < nyc and x < nxc - 1) {
    bool pml_y{y < vpml_regionsyc[0] + 1 or y >= vpml_regionsyc[1]};
    bool pml_x{x < vpml_regionsxc[0] or x >= vpml_regionsxc[1] - 1};
    T lambyxh{(lamb[i_nobatch] + lamb[i_nobatch + 1]) / 2};
    T muyxh{(mu[i_nobatch] + mu[i_nobatch + 1]) / 2};

    if (pml_y) {
      m_vyy[i] = (lambyxh + 2 * muyxh) * dtc<T>() * ay[y] * sigmayy[i] +
                 lambyxh * dtc<T>() * ay[y] * sigmaxx[i] + ay[y] * m_vyy[i];
    }
    if (pml_x) {
      m_vxx[i] = (lambyxh + 2 * muyxh) * dtc<T>() * axh[x] * sigmaxx[i] +
                 lambyxh * dtc<T>() * axh[x] * sigmayy[i] + axh[x] * m_vxx[i];
    }

    // dsigmayydy
    for (int k{}; k < A; ++k) {
      for (int j{}; j < A / 2; ++j) {
        if (y == j + (1 - j + k)) {
          int64_t i2{i - (1 - j + k) * nxc};
          int64_t i2_nobatch{i_nobatch - (1 - j + k) * nxc};
          int64_t y2{y - (1 - j + k)};
          T buoyancyyhxh;
          if (y2 == nyc - 1) {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
                 buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
                4;
          }
          sigmayy[i] += fd_coeffs1y<T>(j, 1 + k) *
                        (buoyancyyhxh * dtc<T>() * (1 + byh[y2]) * vy[i2] +
                         byh[y2] * m_sigmayyy[i2]);
        } else if (y == nyc - 1 - j + (j - k)) {
          int64_t i2{i - (j - k) * nxc};
          int64_t i2_nobatch{i_nobatch - (j - k) * nxc};
          int64_t y2{y - (j - k)};
          T buoyancyyhxh;
          if (y2 == nyc - 1) {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
                 buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
                4;
          }
          sigmayy[i] -= fd_coeffs1y<T>(j, 1 + k) *
                        (buoyancyyhxh * dtc<T>() * (1 + byh[y2]) * vy[i2] +
                         byh[y2] * m_sigmayyy[i2]);
        }
      }
    }
    for (int k{}; k < A / 2; ++k) {
      if (y > A / 2 - 1 + (1 + k) and y < nyc - 1 - A / 2 + 1 + (1 + k)) {
        int64_t i2{i - (1 + k) * nxc};
        int64_t i2_nobatch{i_nobatch - (1 + k) * nxc};
        int64_t y2{y - (1 + k)};
        T buoyancyyhxh;
        if (y2 == nyc - 1) {
          buoyancyyhxh = (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
               buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
              4;
        }
        sigmayy[i] += fd_coeffsy<T>(k) *
                      (buoyancyyhxh * dtc<T>() * (1 + byh[y2]) * vy[i2] +
                       byh[y2] * m_sigmayyy[i2]);
      }
      if (y > A / 2 - 1 - k and y < nyc - 1 - A / 2 + 1 - k) {
        int64_t i2{i - (-k) * nxc};
        int64_t i2_nobatch{i_nobatch - (-k) * nxc};
        int64_t y2{y - (-k)};
        T buoyancyyhxh;
        if (y2 == nyc - 1) {
          buoyancyyhxh = (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
               buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
              4;
        }
        sigmayy[i] -= fd_coeffsy<T>(k) *
                      (buoyancyyhxh * dtc<T>() * (1 + byh[y2]) * vy[i2] +
                       byh[y2] * m_sigmayyy[i2]);
      }
    }

    // dsigmaxxdx
    for (int k{}; k < A; ++k) {
      for (int j{}; j < A / 2; ++j) {
        if (x == j + (-j + k)) {
          int64_t i2{i - (-j + k)};
          int64_t i2_nobatch{i_nobatch - (-j + k)};
          int64_t x2{x - (-j + k)};
          sigmaxx[i] +=
              fd_coeffs1x<T>(j, 1 + k) *
              (buoyancy[i2_nobatch] * dtc<T>() * (1 + bx[x2]) * vx[i2] +
               bx[x2] * m_sigmaxxx[i2]);
        } else if (x == nxc - 1 - j + (j - k - 1)) {
          int64_t i2{i - (j - k - 1)};
          int64_t i2_nobatch{i_nobatch - (j - k - 1)};
          int64_t x2{x - (j - k - 1)};
          sigmaxx[i] -=
              fd_coeffs1x<T>(j, 1 + k) *
              (buoyancy[i2_nobatch] * dtc<T>() * (1 + bx[x2]) * vx[i2] +
               bx[x2] * m_sigmaxxx[i2]);
        }
      }
    }
    for (int k{}; k < A / 2; ++k) {
      if (x > A / 2 - 1 + (k) and x < nxc - 1 - A / 2 + 1 + (k)) {
        int64_t i2{i - (k)};
        int64_t i2_nobatch{i_nobatch - (k)};
        int64_t x2{x - (k)};
        sigmaxx[i] += fd_coeffsx<T>(k) *
                      (buoyancy[i2_nobatch] * dtc<T>() * (1 + bx[x2]) * vx[i2] +
                       bx[x2] * m_sigmaxxx[i2]);
      }
      if (x > A / 2 - 1 - (1 + k) and x < nxc - 1 - A / 2 + 1 - (1 + k)) {
        int64_t i2{i + (1 + k)};
        int64_t i2_nobatch{i_nobatch + (1 + k)};
        int64_t x2{x + (1 + k)};
        sigmaxx[i] -= fd_coeffsx<T>(k) *
                      (buoyancy[i2_nobatch] * dtc<T>() * (1 + bx[x2]) * vx[i2] +
                       bx[x2] * m_sigmaxxx[i2]);
      }
    }
  }

  if (y < nyc - 1 and x >= 1 and x < nxc - 1) {
    bool pml_y{y < vpml_regionsyc[0] + 1 or y >= vpml_regionsyc[1] - 1};
    bool pml_x{x < vpml_regionsxc[0] + 1 or x >= vpml_regionsxc[1] - 1};

    T muyhx{(mu[i_nobatch] + mu[i_nobatch + nxc]) / 2};

    if (pml_y) {
      m_vxy[i] = muyhx * dtc<T>() * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i];
    }
    if (pml_x) {
      m_vyx[i] = muyhx * dtc<T>() * ax[x] * sigmaxy[i] + ax[x] * m_vyx[i];
    }

    // dsigmaxydx
    for (int k{}; k < A; ++k) {
      for (int j{}; j < A / 2 - 1; ++j) {
        if (x == j - j + 1 + k) {
          int64_t i2{i - (-j + 1 + k)};
          int64_t i2_nobatch{i_nobatch - (-j + 1 + k)};
          int64_t x2{x - (-j + 1 + k)};
          T buoyancyyhxh;
          if (y == nyc - 1) {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
                 buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
                4;
          }
          sigmaxy[i] += fd_coeffs2x<T>(1 + k) *
                        (buoyancyyhxh * dtc<T>() * (1 + bxh[x2]) * vy[i2] +
                         bxh[x2] * m_sigmaxyx[i2]);
        } else if (x == nxc - 2 - j + j - k) {
          int64_t i2{i - (j - k)};
          int64_t i2_nobatch{i_nobatch - (j - k)};
          int64_t x2{x - (j - k)};
          T buoyancyyhxh;
          if (y == nyc - 1) {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
                 buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
                4;
          }
          sigmaxy[i] -= fd_coeffs2x<T>(1 + k) *
                        (buoyancyyhxh * dtc<T>() * (1 + bxh[x2]) * vy[i2] +
                         bxh[x2] * m_sigmaxyx[i2]);
        }
      }
    }
    for (int k{}; k < A / 2; ++k) {
      if (x > A / 2 - 2 + 1 + k and x < nxc - 2 - A / 2 + 2 + 1 + k) {
        int64_t i2{i - (1 + k)};
        int64_t i2_nobatch{i_nobatch - (1 + k)};
        int64_t x2{x - (1 + k)};
        T buoyancyyhxh;
        if (y == nyc - 1) {
          buoyancyyhxh = (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
               buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
              4;
        }
        sigmaxy[i] += fd_coeffsx<T>(k) *
                      (buoyancyyhxh * dtc<T>() * (1 + bxh[x2]) * vy[i2] +
                       bxh[x2] * m_sigmaxyx[i2]);
      }
      if (x > A / 2 - 2 - k and x < nxc - 2 - A / 2 + 2 - k) {
        int64_t i2{i - (-k)};
        int64_t i2_nobatch{i_nobatch - (-k)};
        int64_t x2{x - (-k)};
        T buoyancyyhxh;
        if (y == nyc - 1) {
          buoyancyyhxh = (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy[i2_nobatch] + buoyancy[i2_nobatch + 1] +
               buoyancy[i2_nobatch + nxc] + buoyancy[i2_nobatch + nxc + 1]) /
              4;
        }
        sigmaxy[i] -= fd_coeffsx<T>(k) *
                      (buoyancyyhxh * dtc<T>() * (1 + bxh[x2]) * vy[i2] +
                       bxh[x2] * m_sigmaxyx[i2]);
      }
    }

    // dsigmaxydy
    for (int k{}; k < A; ++k) {
      for (int j{}; j < A / 2 - 1; ++j) {
        if (y == 1 + j - j + k) {
          int64_t i2{i - (-j + k) * nxc};
          int64_t i2_nobatch{i_nobatch - (-j + k) * nxc};
          int64_t y2{y - (-j + k)};
          sigmaxy[i] +=
              fd_coeffs2y<T>(1 + k) *
              (buoyancy[i2_nobatch] * dtc<T>() * (1 + by[y2]) * vx[i2] +
               by[y2] * m_sigmaxyy[i2]);
        } else if (y == nyc - 1 - j + j - k - 1) {
          int64_t i2{i - (j - k - 1) * nxc};
          int64_t i2_nobatch{i_nobatch - (j - k - 1) * nxc};
          int64_t y2{y - (j - k - 1)};
          sigmaxy[i] -=
              fd_coeffs2y<T>(1 + k) *
              (buoyancy[i2_nobatch] * dtc<T>() * (1 + by[y2]) * vx[i2] +
               by[y2] * m_sigmaxyy[i2]);
        }
      }
    }
    for (int k{}; k < A / 2; ++k) {
      if (y > 1 + A / 2 - 2 + k and y < nyc - 1 - A / 2 + 2 + k) {
        int64_t i2{i - (k)*nxc};
        int64_t i2_nobatch{i_nobatch - (k)*nxc};
        int64_t y2{y - (k)};
        sigmaxy[i] += fd_coeffsy<T>(k) *
                      (buoyancy[i2_nobatch] * dtc<T>() * (1 + by[y2]) * vx[i2] +
                       by[y2] * m_sigmaxyy[i2]);
      }
      if (y > 1 + A / 2 - 2 - (k + 1) and y < nyc - 1 - A / 2 + 2 - (k + 1)) {
        int64_t i2{i - (-(k + 1)) * nxc};
        int64_t i2_nobatch{i_nobatch - (-(k + 1)) * nxc};
        int64_t y2{y - (-(k + 1))};
        sigmaxy[i] -= fd_coeffsy<T>(k) *
                      (buoyancy[i2_nobatch] * dtc<T>() * (1 + by[y2]) * vx[i2] +
                       by[y2] * m_sigmaxyy[i2]);
      }
    }
  }
}

template <typename T, int A>
void forward_batch(
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
    T const *__restrict bx, T const *__restrict bxh,
    int64_t n_sources_per_shot_y, int64_t n_sources_per_shot_x,
    int64_t n_receivers_per_shot_y, int64_t n_receivers_per_shot_x, int64_t ny,
    int64_t nx, int64_t nt, int64_t step_ratio, bool lamb_requires_grad,
    bool mu_requires_grad, bool buoyancy_requires_grad, int64_t n_batch) {
  dim3 dimBlock(32, 32, 1);
  auto gridx{ceil_div(nx, static_cast<int64_t>(dimBlock.x))};
  auto gridy{ceil_div(ny, static_cast<int64_t>(dimBlock.y))};
  auto gridz{ceil_div(n_batch, static_cast<int64_t>(dimBlock.z))};
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  auto gridx_sources_y{
      ceil_div(n_sources_per_shot_y, static_cast<int64_t>(dimBlock_sources.x))};
  auto gridx_sources_x{
      ceil_div(n_sources_per_shot_x, static_cast<int64_t>(dimBlock_sources.x))};
  auto gridy_sources{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_sources.y))};
  auto gridz_sources{1};
  dim3 dimGrid_sources_y(gridx_sources_y, gridy_sources, gridz_sources);
  dim3 dimGrid_sources_x(gridx_sources_x, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  auto gridx_receivers_y{ceil_div(n_receivers_per_shot_y,
                                  static_cast<int64_t>(dimBlock_receivers.x))};
  auto gridx_receivers_x{ceil_div(n_receivers_per_shot_x,
                                  static_cast<int64_t>(dimBlock_receivers.x))};
  auto gridy_receivers{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_receivers.y))};
  auto gridz_receivers{1};
  dim3 dimGrid_receivers_y(gridx_receivers_y, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_x(gridx_receivers_x, gridy_receivers, gridz_receivers);

  for (int64_t t{}; t < nt; ++t) {
    if (t % step_ratio == 0 and buoyancy_requires_grad) {
      forward_kernel_v<T, A, true><<<dimGrid, dimBlock>>>(
          vy, vx, sigmayy, sigmaxy, sigmaxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
          m_sigmaxxx, dvydbuoyancy + (t / step_ratio) * n_batch * ny * nx,
          dvxdbuoyancy + (t / step_ratio) * n_batch * ny * nx, buoyancy, ay,
          ayh, ax, axh, by, byh, bx, bxh);
    } else {
      forward_kernel_v<T, A, false><<<dimGrid, dimBlock>>>(
          vy, vx, sigmayy, sigmaxy, sigmaxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
          m_sigmaxxx, nullptr, nullptr, buoyancy, ay, ayh, ax, axh, by, byh, bx,
          bxh);
    }
    gpuErrchk(cudaPeekAtLastError());
    if (n_sources_per_shot_y > 0) {
      add_sources<<<dimGrid_sources_y, dimBlock_sources>>>(
          vy, f_y + t * n_batch * n_sources_per_shot_y, sources_y_i,
          n_sources_per_shot_y, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_sources_per_shot_x > 0) {
      add_sources<<<dimGrid_sources_x, dimBlock_sources>>>(
          vx, f_x + t * n_batch * n_sources_per_shot_x, sources_x_i,
          n_sources_per_shot_x, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_receivers_per_shot_y > 0) {
      record_receivers<<<dimGrid_receivers_y, dimBlock_receivers>>>(
          r_y + t * n_batch * n_receivers_per_shot_y, vy, receivers_y_i,
          n_receivers_per_shot_y, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_receivers_per_shot_x > 0) {
      record_receivers<<<dimGrid_receivers_x, dimBlock_receivers>>>(
          r_x + t * n_batch * n_receivers_per_shot_x, vx, receivers_x_i,
          n_receivers_per_shot_x, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 and (lamb_requires_grad or mu_requires_grad)) {
      if (lamb_requires_grad and mu_requires_grad) {
        forward_kernel_sigma<T, A, true, true><<<dimGrid, dimBlock>>>(
            vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
            dvydy_store + (t / step_ratio) * n_batch * ny * nx,
            dvxdx_store + (t / step_ratio) * n_batch * ny * nx,
            dvydxdvxdy_store + (t / step_ratio) * n_batch * ny * nx, lamb, mu,
            ay, ayh, ax, axh, by, byh, bx, bxh);
      } else if (lamb_requires_grad) {
        forward_kernel_sigma<T, A, true, false><<<dimGrid, dimBlock>>>(
            vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
            dvydy_store + (t / step_ratio) * n_batch * ny * nx,
            dvxdx_store + (t / step_ratio) * n_batch * ny * nx, nullptr, lamb,
            mu, ay, ayh, ax, axh, by, byh, bx, bxh);
      } else {
        forward_kernel_sigma<T, A, false, true><<<dimGrid, dimBlock>>>(
            vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
            dvydy_store + (t / step_ratio) * n_batch * ny * nx,
            dvxdx_store + (t / step_ratio) * n_batch * ny * nx,
            dvydxdvxdy_store + (t / step_ratio) * n_batch * ny * nx, lamb, mu,
            ay, ayh, ax, axh, by, byh, bx, bxh);
      }
    } else {
      forward_kernel_sigma<T, A, false, false><<<dimGrid, dimBlock>>>(
          vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
          nullptr, nullptr, nullptr, lamb, mu, ay, ayh, ax, axh, by, byh, bx,
          bxh);
    }
    gpuErrchk(cudaPeekAtLastError());
  }
}

template <typename T, int A>
__global__ void combine_grad_model(T *__restrict grad,
                                   T const *__restrict grad_batch,
                                   int64_t n_batch) {
  auto x{blockIdx.x * blockDim.x + threadIdx.x};
  auto y{blockIdx.y * blockDim.y + threadIdx.y};
  auto i{y * nxc + x};
  if (y < nyc and x < nxc) {
    for (int64_t batch{}; batch < n_batch; ++batch) {
      grad[i] += grad_batch[batch * nynxc + i];
    }
  }
}

template <typename T, int A>
void backward_batch(
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
    T const *__restrict bxh, int64_t n_sources_per_shot_y,
    int64_t n_sources_per_shot_x, int64_t n_receivers_per_shot_y,
    int64_t n_receivers_per_shot_x, int64_t ny, int64_t nx, int64_t nt,
    int64_t step_ratio, bool lamb_requires_grad, bool mu_requires_grad,
    bool buoyancy_requires_grad, int64_t n_batch) {
  dim3 dimBlock(32, 16, 1);
  auto gridx{ceil_div(nx, static_cast<int64_t>(dimBlock.x))};
  auto gridy{ceil_div(ny, static_cast<int64_t>(dimBlock.y))};
  auto gridz{ceil_div(n_batch, static_cast<int64_t>(dimBlock.z))};
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  auto gridx_sources_y{
      ceil_div(n_sources_per_shot_y, static_cast<int64_t>(dimBlock_sources.x))};
  auto gridx_sources_x{
      ceil_div(n_sources_per_shot_x, static_cast<int64_t>(dimBlock_sources.x))};
  auto gridy_sources{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_sources.y))};
  auto gridz_sources{1};
  dim3 dimGrid_sources_y(gridx_sources_y, gridy_sources, gridz_sources);
  dim3 dimGrid_sources_x(gridx_sources_x, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  auto gridx_receivers_y{ceil_div(n_receivers_per_shot_y,
                                  static_cast<int64_t>(dimBlock_receivers.x))};
  auto gridx_receivers_x{ceil_div(n_receivers_per_shot_x,
                                  static_cast<int64_t>(dimBlock_receivers.x))};
  auto gridy_receivers{
      ceil_div(n_batch, static_cast<int64_t>(dimBlock_receivers.y))};
  auto gridz_receivers{1};
  dim3 dimGrid_receivers_y(gridx_receivers_y, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_x(gridx_receivers_x, gridy_receivers, gridz_receivers);
  for (int64_t t{nt - 1}; t >= 0; --t) {
    if (t % step_ratio == 0 and lamb_requires_grad) {
      add_to_grad_lamb<T><<<dimGrid, dimBlock>>>(
          grad_lamb, sigmayy, sigmaxx,
          dvydy_store + (t / step_ratio) * n_batch * ny * nx,
          dvxdx_store + (t / step_ratio) * n_batch * ny * nx);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 and mu_requires_grad) {
      add_to_grad_mu<T><<<dimGrid, dimBlock>>>(
          grad_mu, sigmayy, sigmaxy, sigmaxx,
          dvydy_store + (t / step_ratio) * n_batch * ny * nx,
          dvxdx_store + (t / step_ratio) * n_batch * ny * nx,
          dvydxdvxdy_store + (t / step_ratio) * n_batch * ny * nx);
      gpuErrchk(cudaPeekAtLastError());
    }

    if (n_receivers_per_shot_y > 0) {
      add_sources<<<dimGrid_receivers_y, dimBlock_receivers>>>(
          vy, r_y + t * n_batch * n_receivers_per_shot_y, receivers_y_i,
          n_receivers_per_shot_y, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_receivers_per_shot_x > 0) {
      add_sources<<<dimGrid_receivers_x, dimBlock_receivers>>>(
          vx, r_x + t * n_batch * n_receivers_per_shot_x, receivers_x_i,
          n_receivers_per_shot_x, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    backward_kernel_sigma<T, A><<<dimGrid, dimBlock>>>(
        vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
        m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, m_sigmayyyn,
        m_sigmaxyyn, m_sigmaxyxn, m_sigmaxxxn, lamb, mu, buoyancy, ay, ayh, ax,
        axh, by, byh, bx, bxh);
    gpuErrchk(cudaPeekAtLastError());
    if (n_sources_per_shot_y > 0) {
      record_receivers<<<dimGrid_sources_y, dimBlock_sources>>>(
          f_y + t * n_batch * n_sources_per_shot_y, vy, sources_y_i,
          n_sources_per_shot_y, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (n_sources_per_shot_x > 0) {
      record_receivers<<<dimGrid_sources_x, dimBlock_sources>>>(
          f_x + t * n_batch * n_sources_per_shot_x, vx, sources_x_i,
          n_sources_per_shot_x, n_batch);
      gpuErrchk(cudaPeekAtLastError());
    }
    if (t % step_ratio == 0 and buoyancy_requires_grad) {
      add_to_grad_buoyancy<T><<<dimGrid, dimBlock>>>(
          grad_buoyancy, vy, vx,
          dvydbuoyancy + (t / step_ratio) * n_batch * ny * nx,
          dvxdbuoyancy + (t / step_ratio) * n_batch * ny * nx);
      gpuErrchk(cudaPeekAtLastError());
    }

    backward_kernel_v<T, A><<<dimGrid, dimBlock>>>(
        vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
        m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, lamb, mu, buoyancy, ay,
        ayh, ax, axh, by, byh, bx, bxh);
    gpuErrchk(cudaPeekAtLastError());
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

class ElasticCUDAFunction
    : public torch::autograd::Function<ElasticCUDAFunction> {
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
    TORCH_CHECK(lamb.is_cuda(), "lamb must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(lamb.device());
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
    cudaMemcpyToSymbol(pml_regionsyc, pml_regionsy, sizeof(int64_t) * 2);
    cudaMemcpyToSymbol(pml_regionsxc, pml_regionsx, sizeof(int64_t) * 2);

    int64_t nynx{ny * nx};
    cudaMemcpyToSymbol(nyc, &ny, sizeof(int64_t));
    cudaMemcpyToSymbol(nxc, &nx, sizeof(int64_t));
    cudaMemcpyToSymbol(nynxc, &nynx, sizeof(int64_t));
    cudaMemcpyToSymbol(step_ratioc, &step_ratio, sizeof(int64_t));

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
    auto r_y{at::empty({nt, n_batch, n_receivers_per_shot_y}, options)};
    auto r_x{at::empty({nt, n_batch, n_receivers_per_shot_x}, options)};
    torch::Tensor dvydbuoyancy;
    torch::Tensor dvxdbuoyancy;
    torch::Tensor dvydy_store;
    torch::Tensor dvxdx_store;
    torch::Tensor dvydxdvxdy_store;
    if (buoyancy.requires_grad()) {
      dvydbuoyancy = at::empty(
          {(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx}, options);
      dvxdbuoyancy = at::empty(
          {(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx}, options);
    }
    if (lamb.requires_grad() or mu.requires_grad()) {
      dvydy_store = at::empty(
          {(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx}, options);
      dvxdx_store = at::empty(
          {(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx}, options);
    }
    if (mu.requires_grad()) {
      dvydxdvxdy_store = at::empty(
          {(nt + step_ratio - 1) / step_ratio, n_batch, ny, nx}, options);
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
        lamb.scalar_type(), "elastic_cuda_forward", ([&] {
          scalar_t dt_a = dt;
          cudaMemcpyToSymbol(dt_const, &dt_a, sizeof(scalar_t));
          scalar_t fd_coeffsyh[2];
          scalar_t fd_coeffsxh[2];
          scalar_t fd_coeffs1yh[2][5];
          scalar_t fd_coeffs1xh[2][5];
          scalar_t fd_coeffs2yh[5];
          scalar_t fd_coeffs2xh[5];
          scalar_t fd_coeffs3yh[6];
          scalar_t fd_coeffs3xh[6];

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
          set_fd_coeffs(fd_coeffsyh, fd_coeffs1yh, fd_coeffs2yh, fd_coeffs3yh,
                        accuracy, static_cast<scalar_t>(dy));
          cudaMemcpyToSymbol(fd_coeffsyc, fd_coeffsyh, sizeof(scalar_t) * 2);
          cudaMemcpyToSymbol(fd_coeffs1yc, fd_coeffs1yh[0],
                             sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs1yc, fd_coeffs1yh[1],
                             sizeof(scalar_t) * 5, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs2yc, fd_coeffs2yh, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs3yc, fd_coeffs3yh, sizeof(scalar_t) * 6);
          set_fd_coeffs(fd_coeffsxh, fd_coeffs1xh, fd_coeffs2xh, fd_coeffs3xh,
                        accuracy, static_cast<scalar_t>(dx));
          cudaMemcpyToSymbol(fd_coeffsxc, fd_coeffsxh, sizeof(scalar_t) * 2);
          cudaMemcpyToSymbol(fd_coeffs1xc, fd_coeffs1xh[0],
                             sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs1xc, fd_coeffs1xh[1],
                             sizeof(scalar_t) * 5, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs2xc, fd_coeffs2xh, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs3xc, fd_coeffs3xh, sizeof(scalar_t) * 6);
          decltype(&forward_batch<scalar_t, 4>) forward_batches[]{
              forward_batch<scalar_t, 2>, forward_batch<scalar_t, 4>};
          forward_batches[accuracy / 2 - 1](
              vy_a, vx_a, sigmayy_a, sigmaxy_a, sigmaxx_a, m_vyy_a, m_vyx_a,
              m_vxy_a, m_vxx_a, m_sigmayyy_a, m_sigmaxyy_a, m_sigmaxyx_a,
              m_sigmaxxx_a, sources_y_i_a, sources_x_i_a, receivers_y_i_a,
              receivers_x_i_a, dvydbuoyancy_a, dvxdbuoyancy_a, dvydy_store_a,
              dvxdx_store_a, dvydxdvxdy_store_a, lamb_a, mu_a, buoyancy_a,
              f_y_a, f_x_a, r_y_a, r_x_a, ay_a, ayh_a, ax_a, axh_a, by_a, byh_a,
              bx_a, bxh_a, n_sources_per_shot_y, n_sources_per_shot_x,
              n_receivers_per_shot_y, n_receivers_per_shot_x, ny, nx, nt,
              step_ratio, lamb.requires_grad(), mu.requires_grad(),
              buoyancy.requires_grad(), n_batch);
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
    cudaMemcpyToSymbol(spml_regionsyc, spml_regionsy, sizeof(int64_t) * 2);
    cudaMemcpyToSymbol(spml_regionsxc, spml_regionsx, sizeof(int64_t) * 2);
    cudaMemcpyToSymbol(vpml_regionsyc, vpml_regionsy, sizeof(int64_t) * 2);
    cudaMemcpyToSymbol(vpml_regionsxc, vpml_regionsx, sizeof(int64_t) * 2);

    int64_t nynx{ny * nx};
    cudaMemcpyToSymbol(nyc, &ny, sizeof(int64_t));
    cudaMemcpyToSymbol(nxc, &nx, sizeof(int64_t));
    cudaMemcpyToSymbol(nynxc, &nynx, sizeof(int64_t));
    cudaMemcpyToSymbol(step_ratioc, &step_ratio, sizeof(int64_t));

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

    auto grad_lamb{at::zeros({ny, nx}, options)};
    auto grad_mu{at::zeros({ny, nx}, options)};
    auto grad_buoyancy{at::zeros({ny, nx}, options)};
    auto grad_lamb_batch{n_batch > 1 ? at::zeros({n_batch, ny, nx}, options)
                                     : at::empty(0)};
    auto grad_mu_batch{n_batch > 1 ? at::zeros({n_batch, ny, nx}, options)
                                   : at::empty(0)};
    auto grad_buoyancy_batch{n_batch > 1 ? at::zeros({n_batch, ny, nx}, options)
                                         : at::empty(0)};
    auto grad_f_y{at::empty({nt, n_batch, n_sources_per_shot_y}, options)};
    auto grad_f_x{at::empty({nt, n_batch, n_sources_per_shot_x}, options)};

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
        vy.scalar_type(), "elastic_cuda_backward", ([&] {
          scalar_t dt_a = dt;
          cudaMemcpyToSymbol(dt_const, &dt_a, sizeof(scalar_t));
          scalar_t fd_coeffsyh[2];
          scalar_t fd_coeffsxh[2];
          scalar_t fd_coeffs1yh[2][5];
          scalar_t fd_coeffs1xh[2][5];
          scalar_t fd_coeffs2yh[5];
          scalar_t fd_coeffs2xh[5];
          scalar_t fd_coeffs3yh[6];
          scalar_t fd_coeffs3xh[6];
          scalar_t const *__restrict lamb_a{lamb.data_ptr<scalar_t>()};
          scalar_t const *__restrict mu_a{mu.data_ptr<scalar_t>()};
          scalar_t const *__restrict buoyancy_a{buoyancy.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_lamb_a{grad_lamb.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_mu_a{grad_mu.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_buoyancy_a{
              grad_buoyancy.data_ptr<scalar_t>()};
          scalar_t *__restrict grad_lamb_batch_a{
              n_batch > 1 ? grad_lamb_batch.data_ptr<scalar_t>() : grad_lamb_a};
          scalar_t *__restrict grad_mu_batch_a{
              n_batch > 1 ? grad_mu_batch.data_ptr<scalar_t>() : grad_mu_a};
          scalar_t *__restrict grad_buoyancy_batch_a{
              n_batch > 1 ? grad_buoyancy_batch.data_ptr<scalar_t>()
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
          set_fd_coeffs(fd_coeffsyh, fd_coeffs1yh, fd_coeffs2yh, fd_coeffs3yh,
                        accuracy, static_cast<scalar_t>(dy));
          cudaMemcpyToSymbol(fd_coeffsyc, fd_coeffsyh, sizeof(scalar_t) * 2);
          cudaMemcpyToSymbol(fd_coeffs1yc, fd_coeffs1yh[0],
                             sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs1yc, fd_coeffs1yh[1],
                             sizeof(scalar_t) * 5, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs2yc, fd_coeffs2yh, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs3yc, fd_coeffs3yh, sizeof(scalar_t) * 6);
          set_fd_coeffs(fd_coeffsxh, fd_coeffs1xh, fd_coeffs2xh, fd_coeffs3xh,
                        accuracy, static_cast<scalar_t>(dx));
          cudaMemcpyToSymbol(fd_coeffsxc, fd_coeffsxh, sizeof(scalar_t) * 2);
          cudaMemcpyToSymbol(fd_coeffs1xc, fd_coeffs1xh[0],
                             sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs1xc, fd_coeffs1xh[1],
                             sizeof(scalar_t) * 5, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs2xc, fd_coeffs2xh, sizeof(scalar_t) * 5);
          cudaMemcpyToSymbol(fd_coeffs3xc, fd_coeffs3xh, sizeof(scalar_t) * 6);
          decltype(&backward_batch<scalar_t, 4>) backward_batches[]{
              backward_batch<scalar_t, 2>, backward_batch<scalar_t, 4>};
          backward_batches[accuracy / 2 - 1](
              vy_a, vx_a, sigmayy_a, sigmaxy_a, sigmaxx_a, m_vyy_a, m_vyx_a,
              m_vxy_a, m_vxx_a, m_sigmayyy_a, m_sigmaxyy_a, m_sigmaxyx_a,
              m_sigmaxxx_a, m_sigmayyyn_a, m_sigmaxyyn_a, m_sigmaxyxn_a,
              m_sigmaxxxn_a, sources_y_i_a, sources_x_i_a, receivers_y_i_a,
              receivers_x_i_a, dvydbuoyancy_a, dvxdbuoyancy_a, dvydy_store_a,
              dvxdx_store_a, dvydxdvxdy_store_a, lamb_a, mu_a, buoyancy_a,
              grad_f_y_a, grad_f_x_a, grad_r_y_a, grad_r_x_a, grad_lamb_batch_a,
              grad_mu_batch_a, grad_buoyancy_batch_a, ay_a, ayh_a, ax_a, axh_a,
              by_a, byh_a, bx_a, bxh_a, n_sources_per_shot_y,
              n_sources_per_shot_x, n_receivers_per_shot_y,
              n_receivers_per_shot_x, ny, nx, nt, step_ratio,
              lamb.requires_grad(), mu.requires_grad(),
              buoyancy.requires_grad(), n_batch);
          decltype(&combine_grad_model<scalar_t, 4>) combine_grad_models[]{
              combine_grad_model<scalar_t, 2>, combine_grad_model<scalar_t, 4>};
          dim3 dimBlock_combine(32, 32, 1);
          auto gridx_combine{
              ceil_div(nx, static_cast<int64_t>(dimBlock_combine.x))};
          auto gridy_combine{
              ceil_div(ny, static_cast<int64_t>(dimBlock_combine.y))};
          auto gridz_combine{1};
          dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
          auto combine_grad_modeli{combine_grad_models[accuracy / 2 - 1]};
          if (lamb.requires_grad() and n_batch > 1) {
            combine_grad_modeli<<<dimGrid_combine, dimBlock_combine>>>(
                grad_lamb_a, grad_lamb_batch_a, n_batch);
            gpuErrchk(cudaPeekAtLastError());
          }
          if (mu.requires_grad() and n_batch > 1) {
            combine_grad_modeli<<<dimGrid_combine, dimBlock_combine>>>(
                grad_mu_a, grad_mu_batch_a, n_batch);
            gpuErrchk(cudaPeekAtLastError());
          }
          if (buoyancy.requires_grad() and n_batch > 1) {
            combine_grad_modeli<<<dimGrid_combine, dimBlock_combine>>>(
                grad_buoyancy_a, grad_buoyancy_batch_a, n_batch);
            gpuErrchk(cudaPeekAtLastError());
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

std::vector<torch::Tensor> elastic_cuda_autograd(
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
  return ElasticCUDAFunction::apply(
      lamb, mu, buoyancy, f_y, f_x, vy0, vx0, sigmayy0, sigmaxy0, sigmaxx0,
      m_vyy0, m_vyx0, m_vxy0, m_vxx0, m_sigmayyy0, m_sigmaxyy0, m_sigmaxyx0,
      m_sigmaxxx0, ay, ayh, ax, axh, by, byh, bx, bxh, sources_y_i, sources_x_i,
      receivers_y_i, receivers_x_i, dy, dx, dt, nt, n_batch, step_ratio,
      accuracy, pml_width0, pml_width1, pml_width2, pml_width3);
}

TORCH_LIBRARY_IMPL(deepwave, AutogradCUDA, m) {
  m.impl("elastic", elastic_cuda_autograd);
}

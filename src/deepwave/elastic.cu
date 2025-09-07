/*
 * Elastic wave equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the elastic wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2 and 4.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * For a description of the method, see the C implementation in elastic.c.
 * This file implements the same functionality, but for execution on a GPU
 * using CUDA.
 */

#include <stdio.h>

#include <cstdint>

#include "common.h"

#define CAT_I(name, accuracy, dtype, device) \
  elastic_iso_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_ACCURACY == 2
#elif DW_ACCURACY == 4
#else
#error DW_ACCURACY must be specified and either 2 or 4
#endif /* DW_ACCURACY */

#define A DW_ACCURACY  // Macro for finite difference accuracy order

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

namespace {
__constant__ DW_DTYPE dt;
__constant__ DW_DTYPE fd_coeffsy[2];
__constant__ DW_DTYPE fd_coeffsx[2];
__constant__ DW_DTYPE fd_coeffs1y[2][5];
__constant__ DW_DTYPE fd_coeffs1x[2][5];
__constant__ DW_DTYPE fd_coeffs2y[5];
__constant__ DW_DTYPE fd_coeffs2x[5];
__constant__ DW_DTYPE fd_coeffs3y[6];
__constant__ DW_DTYPE fd_coeffs3x[6];
__constant__ int64_t n_shots;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t nynx;
__constant__ int64_t n_sources_y_per_shot;
__constant__ int64_t n_sources_x_per_shot;
__constant__ int64_t n_receivers_y_per_shot;
__constant__ int64_t n_receivers_x_per_shot;
__constant__ int64_t n_receivers_p_per_shot;
__constant__ int64_t step_ratio;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ int64_t spml_y0;
__constant__ int64_t spml_y1;
__constant__ int64_t spml_x0;
__constant__ int64_t spml_x1;
__constant__ int64_t vpml_y0;
__constant__ int64_t vpml_y1;
__constant__ int64_t vpml_x0;
__constant__ int64_t vpml_x1;
__constant__ bool lamb_batched;
__constant__ bool mu_batched;
__constant__ bool buoyancy_batched;

__global__ void add_sources_y(DW_DTYPE *__restrict const wf,
                              DW_DTYPE const *__restrict const f,
                              int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_y_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_y_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void add_sources_x(DW_DTYPE *__restrict const wf,
                              DW_DTYPE const *__restrict const f,
                              int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_x_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_x_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void add_adjoint_sources_y(
    DW_DTYPE *__restrict const wf, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_y_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_y_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void add_adjoint_sources_x(
    DW_DTYPE *__restrict const wf, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_x_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_x_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void add_adjoint_pressure_sources(
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_p_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_p_per_shot + source_idx;
    if (0 <= sources_i[k]) {
      sigmayy[shot_idx * nynx + sources_i[k]] -= f[k] / (DW_DTYPE)2;
      sigmaxx[shot_idx * nynx + sources_i[k]] -= f[k] / (DW_DTYPE)2;
    }
  }
}

__global__ void record_receivers_y(DW_DTYPE *__restrict const r,
                                   DW_DTYPE const *__restrict const wf,
                                   int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_y_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_y_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

__global__ void record_receivers_x(DW_DTYPE *__restrict const r,
                                   DW_DTYPE const *__restrict const wf,
                                   int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_x_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_x_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

__global__ void record_pressure_receivers(
    DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxx,
    int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_p_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_p_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
      r[k] = -(sigmayy[shot_idx * nynx + receivers_i[k]] +
               sigmaxx[shot_idx * nynx + receivers_i[k]]) /
             (DW_DTYPE)2;
  }
}

__global__ void record_adjoint_receivers_y(
    DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const wf,
    int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_y_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_y_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

__global__ void record_adjoint_receivers_x(
    DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const wf,
    int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_x_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_x_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

__global__ void add_to_grad_lamb(DW_DTYPE *__restrict const grad_lamb,
                                 DW_DTYPE const *__restrict const sigmayy,
                                 DW_DTYPE const *__restrict const sigmaxx,
                                 DW_DTYPE const *__restrict const dvydy_store,
                                 DW_DTYPE const *__restrict const dvxdx_store) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i = batch * nynx + y * nx + x;
  if (y < ny) {
    if (x == 0) {
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2) *
          (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2 +
           (sigmayy[i - 1] + sigmaxx[i - 1]) *
               (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_lamb[i] += ((sigmayy[i - 1] + sigmaxx[i - 1]) *
                       (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
                      (DW_DTYPE)step_ratio;
    }
  }
}

__global__ void add_to_grad_mu(
    DW_DTYPE *__restrict const grad_mu,
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const dvydy_store,
    DW_DTYPE const *__restrict const dvxdx_store,
    DW_DTYPE const *__restrict const dvydxdvxdy_store) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i = batch * nynx + y * nx + x;
  if (y == 1) {
    if (x == 0) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    (DW_DTYPE)step_ratio;
    }
  } else if (y < ny - 1) {
    if (x == 0) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2 +
           sigmaxy[i - nx] * dvydxdvxdy_store[i - nx] / 2) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    (DW_DTYPE)step_ratio;
    }
  } else if (y == ny - 1) {
    if (x == 0) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i - nx] * dvydxdvxdy_store[i - nx] / 2) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    (DW_DTYPE)step_ratio;
    }
  }
}

__global__ void add_to_grad_buoyancy(
    DW_DTYPE *__restrict const grad_buoyancy,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const dvydbuoyancy,
    DW_DTYPE const *__restrict const dvxdbuoyancy) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i = batch * nynx + y * nx + x;
  if (y == 0) {
    if (x == 0) {
      grad_buoyancy[i] += (vy[i] * dvydbuoyancy[i] / 4) * (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_buoyancy[i] +=
          (vy[i - 1] * dvydbuoyancy[i - 1] / 4) * (DW_DTYPE)step_ratio;
    }
  } else if (y < ny - 1) {
    if (x == 0) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
           vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
                           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          (DW_DTYPE)step_ratio;
    }
  } else if (y == ny - 1) {
    if (x == 0) {
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 2 + vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    } else if (x < nx - 1) {
      grad_buoyancy[i] +=
          (vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
           vy[i] * dvydbuoyancy[i] / 2 + vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    } else if (x == nx - 1) {
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
                           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          (DW_DTYPE)step_ratio;
    }
  }
}

__global__ void combine_grad(DW_DTYPE *__restrict const grad,
                             DW_DTYPE const *__restrict const grad_shot) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t i = y * nx + x;
  if (y < ny && x < nx) {
    int64_t shot_idx;
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * nynx + i];
    }
  }
}

__global__ void forward_kernel_v(
    DW_DTYPE const *__restrict const buoyancy, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const vx, DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const dvydbuoyancy,
    DW_DTYPE *__restrict const dvxdbuoyancy,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i_noshot = y * nx + x;
  int64_t i = batch * nynx + i_noshot;
  int64_t j, k;
  DW_DTYPE const *__restrict const buoyancy_shot =
      buoyancy_batched ? buoyancy + batch * nynx : buoyancy;

  if (y < ny && x < nx - 1) {
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1 - 1;
    DW_DTYPE dsigmayydy = 0;
    DW_DTYPE dsigmaxydx = 0;

    // dsigmaxydx
    for (j = 0; j < A / 2 - 1; ++j) {
      if (x == j) {
        for (k = 0; k < A; ++k) {
          dsigmaxydx += fd_coeffs2x[1 + k] * sigmaxy[i - j + 1 + k];
        }
      } else if (x == nx - 2 - j) {
        for (k = 0; k < A; ++k) {
          dsigmaxydx -= fd_coeffs2x[1 + k] * sigmaxy[i + j - k];
        }
      }
    }
    if (x > A / 2 - 2 && x < nx - 2 - A / 2 + 2) {
      for (k = 0; k < A / 2; ++k) {
        dsigmaxydx += fd_coeffsx[k] * (sigmaxy[i + 1 + k] - sigmaxy[i - k]);
      }
    }

    // dsigmayydy
    for (j = 0; j < A / 2; ++j) {
      if (y == j) {
        for (k = 0; k < A; ++k) {
          dsigmayydy += fd_coeffs1y[j][1 + k] * sigmayy[i + (1 - j + k) * nx];
        }
      } else if (y == ny - 1 - j) {
        for (k = 0; k < A; ++k) {
          dsigmayydy -= fd_coeffs1y[j][1 + k] * sigmayy[i + (j - k) * nx];
        }
      }
    }
    if (y > A / 2 - 1 && y < ny - 1 - A / 2 + 1) {
      for (k = 0; k < A / 2; ++k) {
        dsigmayydy +=
            fd_coeffsy[k] * (sigmayy[i + (1 + k) * nx] - sigmayy[i - k * nx]);
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
    {
      DW_DTYPE buoyancyyhxh;
      if (y == ny - 1) {
        buoyancyyhxh =
            (buoyancy_shot[i_noshot] + buoyancy_shot[i_noshot + 1]) / 2;
      } else {
        buoyancyyhxh =
            (buoyancy_shot[i_noshot] + buoyancy_shot[i_noshot + 1] +
             buoyancy_shot[i_noshot + nx] + buoyancy_shot[i_noshot + nx + 1]) /
            4;
      }
      vy[i] += buoyancyyhxh * dt * (dsigmayydy + dsigmaxydx);
      if (buoyancy_requires_grad) {
        dvydbuoyancy[i] = dt * (dsigmayydy + dsigmaxydx);
      }
    }
  }

  if (y >= 1 && y < ny && x < nx) {
    bool pml_y = y < pml_y0 + 1 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    DW_DTYPE dsigmaxydy = 0;
    DW_DTYPE dsigmaxxdx = 0;

    // dsigmaxydy
    for (j = 0; j < A / 2 - 1; ++j) {
      if (y == 1 + j) {
        for (k = 0; k < A; ++k) {
          dsigmaxydy += fd_coeffs2y[1 + k] * sigmaxy[i + (-j + k) * nx];
        }
      } else if (y == ny - 1 - j) {
        for (k = 0; k < A; ++k) {
          dsigmaxydy -= fd_coeffs2y[1 + k] * sigmaxy[i + (j - k - 1) * nx];
        }
      }
    }
    if (y > 1 + A / 2 - 2 && y < ny - 1 - A / 2 + 2) {
      for (k = 0; k < A / 2; ++k) {
        dsigmaxydy +=
            fd_coeffsy[k] * (sigmaxy[i + k * nx] - sigmaxy[i - (k + 1) * nx]);
      }
    }

    // dsigmaxxdx
    for (j = 0; j < A / 2; ++j) {
      if (x == j) {
        for (k = 0; k < A; ++k) {
          dsigmaxxdx += fd_coeffs1x[j][1 + k] * sigmaxx[i - j + k];
        }
      } else if (x == nx - 1 - j) {
        for (k = 0; k < A; ++k) {
          dsigmaxxdx -= fd_coeffs1x[j][1 + k] * sigmaxx[i + (j - k - 1)];
        }
      }
    }
    if (x > A / 2 - 1 && x < nx - 1 - A / 2 + 1) {
      for (k = 0; k < A / 2; ++k) {
        dsigmaxxdx += fd_coeffsx[k] * (sigmaxx[i + k] - sigmaxx[i - (1 + k)]);
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
    vx[i] += buoyancy_shot[i_noshot] * dt * (dsigmaxxdx + dsigmaxydy);
    if (buoyancy_requires_grad) {
      dvxdbuoyancy[i] = dt * (dsigmaxxdx + dsigmaxydy);
    }
  }
}

__global__ void forward_kernel_sigma(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE *__restrict const m_vyy,
    DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict const dvydy_store,
    DW_DTYPE *__restrict const dvxdx_store,
    DW_DTYPE *__restrict const dvydxdvxdy_store,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const lamb_requires_grad, bool const mu_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i_noshot = y * nx + x;
  int64_t i = batch * nynx + i_noshot;
  int64_t j, jp, k;
  DW_DTYPE const *__restrict const lamb_shot =
      lamb_batched ? lamb + batch * nynx : lamb;
  DW_DTYPE const *__restrict const mu_shot =
      mu_batched ? mu + batch * nynx : mu;

  if (y < ny && x < nx - 1) {
    bool pml_y = y < pml_y0 + 1 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1 - 1;
    DW_DTYPE dvydy = 0;
    DW_DTYPE dvxdx = 0;

    // dvydy
    for (j = 0; j < A / 2 - 1; ++j) {
      if (y == 1 + j) {
        for (k = 0; k <= A; ++k) {
          dvydy += fd_coeffs2y[k] * vy[i + (-j + k - 1) * nx];
        }
      } else if (y == ny - 1 - j) {
        for (k = 0; k <= A; ++k) {
          dvydy -= fd_coeffs2y[k] * vy[i + (j - k) * nx];
        }
      }
    }
    if (y > 1 + A / 2 - 2 && y < ny - 1 - A / 2 + 2) {
      for (k = 0; k < A / 2; ++k) {
        dvydy += fd_coeffsy[k] * (vy[i + k * nx] - vy[i - (k + 1) * nx]);
      }
    }

    // dvxdx
    for (j = 0; j < A / 2 - 1; ++j) {
      if (x == j) {
        for (k = 0; k <= A; ++k) {
          dvxdx += fd_coeffs2x[k] * vx[i - j + k];
        }
      } else if (x == nx - 2 - j) {
        for (k = 0; k <= A; ++k) {
          dvxdx -= fd_coeffs2x[k] * vx[i + j - k + 1];
        }
      }
    }
    if (x > A / 2 - 2 && x < nx - 2 - A / 2 + 2) {
      for (k = 0; k < A / 2; ++k) {
        dvxdx += fd_coeffsx[k] * (vx[i + 1 + k] - vx[i - k]);
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
    {
      DW_DTYPE lambyxh = (lamb_shot[i_noshot] + lamb_shot[i_noshot + 1]) / 2;
      DW_DTYPE muyxh = (mu_shot[i_noshot] + mu_shot[i_noshot + 1]) / 2;
      sigmayy[i] += dt * ((lambyxh + 2 * muyxh) * dvydy + lambyxh * dvxdx);
      sigmaxx[i] += dt * ((lambyxh + 2 * muyxh) * dvxdx + lambyxh * dvydy);
      if (lamb_requires_grad || mu_requires_grad) {
        dvydy_store[i] = dt * dvydy;
        dvxdx_store[i] = dt * dvxdx;
      }
    }
  }

  if (y < ny - 1 && x >= 1 && x < nx - 1) {
    bool pml_y = y < pml_y0 + 1 || y >= pml_y1 - 1;
    bool pml_x = x < pml_x0 + 1 || x >= pml_x1 - 1;
    DW_DTYPE dvydx = 0;
    DW_DTYPE dvxdy = 0;

    // dvxdy
    for (j = 0; j < A / 2 - 1; ++j) {
      if (y == 1 + j) {
        DW_DTYPE dvydxp = 0;
        for (jp = 0; jp < A / 2 - 1; ++jp) {
          if (x == 1 + jp) {
            for (k = 0; k <= A; ++k) {
              dvydxp += fd_coeffs2x[k] * vy[i - (j + 1) * nx - (jp + 1) + k];
            }
          } else if (x == nx - 2 - jp) {
            for (k = 0; k <= A; ++k) {
              dvydxp -= fd_coeffs2x[k] * vy[i - (j + 1) * nx + jp - k];
            }
          }
        }
        if (x > 1 + A / 2 - 2 && x < nx - 2 - A / 2 + 2) {
          for (k = 0; k < A / 2; ++k) {
            dvydxp += fd_coeffsx[k] *
                      (vy[i - (j + 1) * nx + k] - vy[i - (j + 1) * nx - k - 1]);
          }
        }
        dvxdy = -fd_coeffs3y[0] * dvydxp;
        for (k = 1; k <= A + 1; ++k) {
          dvxdy += fd_coeffs3y[k] * vx[i + (-j + k - 1) * nx];
        }
      } else if (y == ny - 2 - j) {
        DW_DTYPE dvydxp = 0;
        for (jp = 0; jp < A / 2 - 1; ++jp) {
          if (x == 1 + jp) {
            for (k = 0; k <= A; ++k) {
              dvydxp += fd_coeffs2x[k] * vy[i + (j + 1) * nx - (jp + 1) + k];
            }
          } else if (x == nx - 2 - jp) {
            for (k = 0; k <= A; ++k) {
              dvydxp -= fd_coeffs2x[k] * vy[i + (j + 1) * nx + jp - k];
            }
          }
        }
        if (x > 1 + A / 2 - 2 && x < nx - 2 - A / 2 + 2) {
          for (k = 0; k < A / 2; ++k) {
            dvydxp += fd_coeffsx[k] *
                      (vy[i + (j + 1) * nx + k] - vy[i + (j + 1) * nx - k - 1]);
          }
        }
        dvxdy = fd_coeffs3y[0] * dvydxp;
        for (k = 1; k <= A + 1; ++k) {
          dvxdy -= fd_coeffs3y[k] * vx[i + (j - k + 2) * nx];
        }
      }
    }
    if (y > 1 + A / 2 - 2 && y < ny - 2 - A / 2 + 2) {
      for (k = 0; k < A / 2; ++k) {
        dvxdy += fd_coeffsy[k] * (vx[i + (k + 1) * nx] - vx[i - k * nx]);
      }
    }

    // dvydx
    for (j = 0; j < A / 2 - 1; ++j) {
      if (x == 1 + j) {
        DW_DTYPE dvxdyp = 0;
        for (jp = 0; jp < A / 2 - 1; ++jp) {
          if (y == 1 + jp) {
            for (k = 0; k <= A; ++k) {
              dvxdyp += fd_coeffs2y[k] * vx[i - (j + 1) + (-jp + k) * nx];
            }
          } else if (y == ny - 2 - jp) {
            for (k = 0; k <= A; ++k) {
              dvxdyp -= fd_coeffs2y[k] * vx[i - (j + 1) + ((jp + 1) - k) * nx];
            }
          }
        }
        if (y > 1 + A / 2 - 2 && y < ny - 2 - A / 2 + 2) {
          for (k = 0; k < A / 2; ++k) {
            dvxdyp += fd_coeffsy[k] * (vx[i - (j + 1) + (k + 1) * nx] -
                                       vx[i - (j + 1) - k * nx]);
          }
        }
        dvydx = -fd_coeffs3x[0] * dvxdyp;
        for (k = 1; k <= A + 1; ++k) {
          dvydx += fd_coeffs3x[k] * vy[i + (-j + k - 2)];
        }
      } else if (x == nx - 2 - j) {
        DW_DTYPE dvxdyp = 0;
        for (jp = 0; jp < A / 2 - 1; ++jp) {
          if (y == 1 + jp) {
            for (k = 0; k <= A; ++k) {
              dvxdyp += fd_coeffs2y[k] * vx[i + (j + 1) + (-jp + k) * nx];
            }
          } else if (y == ny - 2 - jp) {
            for (k = 0; k <= A; ++k) {
              dvxdyp -= fd_coeffs2y[k] * vx[i + (j + 1) + (jp - k + 1) * nx];
            }
          }
        }
        if (y > 1 + A / 2 - 2 && y < ny - 2 - A / 2 + 2) {
          for (k = 0; k < A / 2; ++k) {
            dvxdyp += fd_coeffsy[k] * (vx[i + (j + 1) + (k + 1) * nx] -
                                       vx[i + (j + 1) + (-k) * nx]);
          }
        }
        dvydx = fd_coeffs3x[0] * dvxdyp;
        for (k = 1; k <= A + 1; ++k) {
          dvydx -= fd_coeffs3x[k] * vy[i + (j - k + 1)];
        }
      }
    }
    if (x > 1 + A / 2 - 2 && x < nx - 2 - A / 2 + 2) {
      for (k = 0; k < A / 2; ++k) {
        dvydx += fd_coeffsx[k] * (vy[i + k] - vy[i - k - 1]);
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
    {
      DW_DTYPE muyhx = (mu_shot[i_noshot] + mu_shot[i_noshot + nx]) / 2;
      sigmaxy[i] += dt * muyhx * (dvydx + dvxdy);
      if (mu_requires_grad) {
        dvydxdvxdy_store[i] = dt * (dvydx + dvxdy);
      }
    }
  }
}

__global__ void backward_kernel_sigma(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const buoyancy, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const vx, DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
    DW_DTYPE const *__restrict const sigmaxx,
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
    DW_DTYPE const *__restrict const m_vxy,
    DW_DTYPE const *__restrict const m_vxx,
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmayyyn,
    DW_DTYPE *__restrict const m_sigmaxyyn,
    DW_DTYPE *__restrict const m_sigmaxyxn,
    DW_DTYPE *__restrict const m_sigmaxxxn, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i_noshot = y * nx + x;
  int64_t i = batch * nynx + i_noshot;
  int64_t j, jp, k;
  DW_DTYPE const *__restrict const lamb_shot =
      lamb_batched ? lamb + batch * nynx : lamb;
  DW_DTYPE const *__restrict const mu_shot =
      mu_batched ? mu + batch * nynx : mu;
  DW_DTYPE const *__restrict const buoyancy_shot =
      buoyancy_batched ? buoyancy + batch * nynx : buoyancy;
  if (y < ny && x < nx - 1) {
    bool pml_y = y < spml_y0 || y >= spml_y1;
    bool pml_x = x < spml_x0 || x >= spml_x1 - 1;
    // from sigmayy/sigmaxx edges
    for (k = 0; k <= A; ++k) {
      for (j = 0; j < A / 2 - 1; ++j) {
        if (y == 1 + j + (-j + k - 1)) {
          DW_DTYPE lambyxh = (lamb_shot[i_noshot - (-j + k - 1) * nx] +
                              lamb_shot[i_noshot + 1 - (-j + k - 1) * nx]) /
                             2;
          DW_DTYPE muyxh = (mu_shot[i_noshot - (-j + k - 1) * nx] +
                            mu_shot[i_noshot + 1 - (-j + k - 1) * nx]) /
                           2;
          vy[i] +=
              fd_coeffs2y[k] *
              (dt * (1 + by[y - (-j + k - 1)]) *
                   ((lambyxh + 2 * muyxh) * sigmayy[i - (-j + k - 1) * nx] +
                    lambyxh * sigmaxx[i - (-j + k - 1) * nx]) +
               by[y - (-j + k - 1)] * m_vyy[i - (-j + k - 1) * nx]);
        } else if (y == ny - 1 - j + (j - k)) {
          DW_DTYPE lambyxh = (lamb_shot[i_noshot - (j - k) * nx] +
                              lamb_shot[i_noshot + 1 - (j - k) * nx]) /
                             2;
          DW_DTYPE muyxh = (mu_shot[i_noshot - (j - k) * nx] +
                            mu_shot[i_noshot + 1 - (j - k) * nx]) /
                           2;
          vy[i] -= fd_coeffs2y[k] *
                   (dt * (1 + by[y - (j - k)]) *
                        ((lambyxh + 2 * muyxh) * sigmayy[i - (j - k) * nx] +
                         lambyxh * sigmaxx[i - (j - k) * nx]) +
                    by[y - (j - k)] * m_vyy[i - (j - k) * nx]);
        }
      }
    }

    // from sigmayy/sigmaxx centre
    for (k = 0; k < A / 2; ++k) {
      if (y > 1 + A / 2 - 2 + k && y < ny - 1 - A / 2 + 2 + k) {
        DW_DTYPE lambyxh =
            (lamb_shot[i_noshot - k * nx] + lamb_shot[i_noshot + 1 - k * nx]) /
            2;
        DW_DTYPE muyxh =
            (mu_shot[i_noshot - k * nx] + mu_shot[i_noshot + 1 - k * nx]) / 2;
        vy[i] +=
            fd_coeffsy[k] * (dt * (1 + by[y - k]) *
                                 ((lambyxh + 2 * muyxh) * sigmayy[i - k * nx] +
                                  lambyxh * sigmaxx[i - k * nx]) +
                             by[y - k] * m_vyy[i - k * nx]);
      }
      if (y > 1 + A / 2 - 2 - (k + 1) && y < ny - 1 - A / 2 + 2 - (k + 1)) {
        DW_DTYPE lambyxh = (lamb_shot[i_noshot + (k + 1) * nx] +
                            lamb_shot[i_noshot + 1 + (k + 1) * nx]) /
                           2;
        DW_DTYPE muyxh = (mu_shot[i_noshot + (k + 1) * nx] +
                          mu_shot[i_noshot + 1 + (k + 1) * nx]) /
                         2;
        vy[i] -= fd_coeffsy[k] *
                 (dt * (1 + by[y + k + 1]) *
                      ((lambyxh + 2 * muyxh) * sigmayy[i + (k + 1) * nx] +
                       lambyxh * sigmaxx[i + (k + 1) * nx]) +
                  by[y + k + 1] * m_vyy[i + (k + 1) * nx]);
      }
    }

    // from sigmaxy dvxdy
    for (j = 0; j < A / 2 - 1; ++j) {
      if (y == 1 + j - (j + 1)) {
        int64_t y2 = y + (j + 1);
        for (k = 0; k <= A; ++k) {
          for (jp = 0; jp < A / 2 - 1; ++jp) {
            if (x == 1 + jp - (jp + 1) + k) {
              int64_t i2 = i - (-(j + 1) * nx - (jp + 1) + k);
              int64_t i2_noshot = i_noshot - (-(j + 1) * nx - (jp + 1) + k);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vy[i] += fd_coeffs2x[k] * (-fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            } else if (x == nx - 2 - jp + jp - k) {
              int64_t i2 = i - (-(j + 1) * nx + jp - k);
              int64_t i2_noshot = i_noshot - (-(j + 1) * nx + jp - k);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vy[i] -= fd_coeffs2x[k] * (-fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
          }
        }
        for (k = 0; k < A / 2; ++k) {
          if (x > 1 + A / 2 - 2 + k && x < nx - 2 - A / 2 + 2 + k) {
            int64_t i2 = i - (-(j + 1) * nx + k);
            int64_t i2_noshot = i_noshot - (-(j + 1) * nx + k);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vy[i] += fd_coeffsx[k] * (-fd_coeffs3y[0]) *
                     (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
          if (x > 1 + A / 2 - 2 - k - 1 && x < nx - 2 - A / 2 + 2 - k - 1) {
            int64_t i2 = i - (-(j + 1) * nx - k - 1);
            int64_t i2_noshot = i_noshot - (-(j + 1) * nx - k - 1);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vy[i] -= fd_coeffsx[k] * (-fd_coeffs3y[0]) *
                     (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
        }

      } else if (y == ny - 2 - j + (j + 1)) {
        int64_t y2 = y - (j + 1);
        for (k = 0; k <= A; ++k) {
          for (jp = 0; jp < A / 2 - 1; ++jp) {
            if (x == 1 + jp - (jp + 1) + k) {
              int64_t i2 = i - ((j + 1) * nx - (jp + 1) + k);
              int64_t i2_noshot = i_noshot - ((j + 1) * nx - (jp + 1) + k);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vy[i] += fd_coeffs2x[k] * (fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            } else if (x == nx - 2 - jp + jp - k) {
              int64_t i2 = i - ((j + 1) * nx + jp - k);
              int64_t i2_noshot = i_noshot - ((j + 1) * nx + jp - k);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vy[i] -= fd_coeffs2x[k] * (fd_coeffs3y[0]) *
                       (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                        byh[y2] * m_vxy[i2]);
            }
          }
        }
        for (k = 0; k < A / 2; ++k) {
          if (x > 1 + A / 2 - 2 + k && x < nx - 2 - A / 2 + 2 + k) {
            int64_t i2 = i - ((j + 1) * nx + k);
            int64_t i2_noshot = i_noshot - ((j + 1) * nx + k);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vy[i] += fd_coeffsx[k] * (fd_coeffs3y[0]) *
                     (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
          if (x > 1 + A / 2 - 2 - k - 1 && x < nx - 2 - A / 2 + 2 - k - 1) {
            int64_t i2 = i - ((j + 1) * nx - k - 1);
            int64_t i2_noshot = i_noshot - ((j + 1) * nx - k - 1);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vy[i] -= fd_coeffsx[k] * (fd_coeffs3y[0]) *
                     (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                      byh[y2] * m_vxy[i2]);
          }
        }
      }
    }

    // from sigmaxy dvydx
    if (y > 0 && y < ny - 1) {
      for (k = 1; k <= A + 1; ++k) {
        for (j = 0; j < A / 2 - 1; ++j) {
          if (x == 1 + j + (-j + k - 2)) {
            int64_t x2 = x - (-j + k - 2);
            int64_t i2 = i - (-j + k - 2);
            int64_t i2_noshot = i_noshot - (-j + k - 2);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vy[i] += fd_coeffs3x[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                       bx[x2] * m_vyx[i2]);
          } else if (x == nx - 2 - j + (j - k + 1)) {
            int64_t x2 = x - (j - k + 1);
            int64_t i2 = i - (j - k + 1);
            int64_t i2_noshot = i_noshot - (j - k + 1);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vy[i] -= fd_coeffs3x[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                       bx[x2] * m_vyx[i2]);
          }
        }
      }
      for (k = 0; k < A / 2; ++k) {
        if (x > 1 + A / 2 - 2 + k && x < nx - 2 - A / 2 + 2 + k) {
          int64_t x2 = x - k;
          int64_t i2 = i - k;
          int64_t i2_noshot = i_noshot - k;
          DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
          vy[i] += fd_coeffsx[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                    bx[x2] * m_vyx[i2]);
        }
        if (x > 1 + A / 2 - 2 - k - 1 && x < nx - 2 - A / 2 + 2 - k - 1) {
          int64_t x2 = x + k + 1;
          int64_t i2 = i + k + 1;
          int64_t i2_noshot = i_noshot + k + 1;
          DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
          vy[i] -= fd_coeffsx[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                                    bx[x2] * m_vyx[i2]);
        }
      }
    }

    {
      DW_DTYPE buoyancyyhxh;
      if (y == ny - 1) {
        buoyancyyhxh =
            (buoyancy_shot[i_noshot] + buoyancy_shot[i_noshot + 1]) / 2;
      } else {
        buoyancyyhxh =
            (buoyancy_shot[i_noshot] + buoyancy_shot[i_noshot + 1] +
             buoyancy_shot[i_noshot + nx] + buoyancy_shot[i_noshot + nx + 1]) /
            4;
      }

      if (pml_y) {
        m_sigmayyyn[i] =
            buoyancyyhxh * dt * ayh[y] * vy[i] + ayh[y] * m_sigmayyy[i];
      }
      if (pml_x) {
        m_sigmaxyxn[i] =
            buoyancyyhxh * dt * axh[x] * vy[i] + axh[x] * m_sigmaxyx[i];
      }
    }
  }

  if (y >= 1 && y < ny && x < nx) {
    bool pml_y = y < spml_y0 + 1 || y >= spml_y1;
    bool pml_x = x < spml_x0 || x >= spml_x1;
    // from sigmayy/sigmaxx edges
    for (k = 0; k <= A; ++k) {
      for (j = 0; j < A / 2 - 1; ++j) {
        if (x == j + (-j + k)) {
          int64_t i2 = i - (-j + k);
          int64_t i2_noshot = i_noshot - (-j + k);
          int64_t x2 = x - (-j + k);
          DW_DTYPE lambyxh =
              (lamb_shot[i2_noshot] + lamb_shot[i2_noshot + 1]) / 2;
          DW_DTYPE muyxh = (mu_shot[i2_noshot] + mu_shot[i2_noshot + 1]) / 2;
          vx[i] += fd_coeffs2x[k] * (dt * (1 + bxh[x2]) *
                                         ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                          lambyxh * sigmayy[i2]) +
                                     bxh[x2] * m_vxx[i2]);
        } else if (x == nx - 2 - j + (j - k + 1)) {
          int64_t i2 = i - (j - k + 1);
          int64_t i2_noshot = i_noshot - (j - k + 1);
          int64_t x2 = x - (j - k + 1);
          DW_DTYPE lambyxh =
              (lamb_shot[i2_noshot] + lamb_shot[i2_noshot + 1]) / 2;
          DW_DTYPE muyxh = (mu_shot[i2_noshot] + mu_shot[i2_noshot + 1]) / 2;
          vx[i] -= fd_coeffs2x[k] * (dt * (1 + bxh[x2]) *
                                         ((lambyxh + 2 * muyxh) * sigmaxx[i2] +
                                          lambyxh * sigmayy[i2]) +
                                     bxh[x2] * m_vxx[i2]);
        }
      }
    }

    // from sigmayy/sigmaxx centre
    for (k = 0; k < A / 2; ++k) {
      if (x > A / 2 - 2 + 1 + k && x < nx - 2 - A / 2 + 2 + 1 + k) {
        int64_t i2 = i - (1 + k);
        int64_t i2_noshot = i_noshot - (1 + k);
        int64_t x2 = x - (1 + k);
        DW_DTYPE lambyxh =
            (lamb_shot[i2_noshot] + lamb_shot[i2_noshot + 1]) / 2;
        DW_DTYPE muyxh = (mu_shot[i2_noshot] + mu_shot[i2_noshot + 1]) / 2;
        vx[i] +=
            fd_coeffsx[k] *
            (dt * (1 + bxh[x2]) *
                 ((lambyxh + 2 * muyxh) * sigmaxx[i2] + lambyxh * sigmayy[i2]) +
             bxh[x2] * m_vxx[i2]);
      }
      if (x > A / 2 - 2 - k && x < nx - 2 - A / 2 + 2 - k) {
        int64_t i2 = i + k;
        int64_t i2_noshot = i_noshot + k;
        int64_t x2 = x + k;
        DW_DTYPE lambyxh =
            (lamb_shot[i2_noshot] + lamb_shot[i2_noshot + 1]) / 2;
        DW_DTYPE muyxh = (mu_shot[i2_noshot] + mu_shot[i2_noshot + 1]) / 2;
        vx[i] -=
            fd_coeffsx[k] *
            (dt * (1 + bxh[x2]) *
                 ((lambyxh + 2 * muyxh) * sigmaxx[i2] + lambyxh * sigmayy[i2]) +
             bxh[x2] * m_vxx[i2]);
      }
    }

    // from sigmaxy dvydx
    for (j = 0; j < A / 2 - 1; ++j) {
      if (x == 1 + j - (j + 1)) {
        int64_t x2 = x + (j + 1);
        for (k = 0; k <= A; ++k) {
          for (jp = 0; jp < A / 2 - 1; ++jp) {
            if (y == 1 + jp - jp + k) {
              int64_t i2 = i - (-(j + 1) + (-jp + k) * nx);
              int64_t i2_noshot = i_noshot - (-(j + 1) + (-jp + k) * nx);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vx[i] += fd_coeffs2y[k] * (-fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            } else if (y == ny - 2 - jp + (jp + 1) - k) {
              int64_t i2 = i - (-(j + 1) + ((jp + 1) - k) * nx);
              int64_t i2_noshot = i_noshot - (-(j + 1) + ((jp + 1) - k) * nx);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vx[i] -= fd_coeffs2y[k] * (-fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
          }
        }
        for (k = 0; k < A / 2; ++k) {
          if (y > 1 + A / 2 - 2 + k + 1 && y < ny - 2 - A / 2 + 2 + k + 1) {
            int64_t i2 = i - (-(j + 1) + (k + 1) * nx);
            int64_t i2_noshot = i_noshot - (-(j + 1) + (k + 1) * nx);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vx[i] +=
                fd_coeffsy[k] * (-fd_coeffs3x[0]) *
                (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] + bx[x2] * m_vyx[i2]);
          }
          if (y > 1 + A / 2 - 2 - k && y < ny - 2 - A / 2 + 2 - k) {
            int64_t i2 = i - (-(j + 1) - k * nx);
            int64_t i2_noshot = i_noshot - (-(j + 1) - k * nx);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vx[i] -=
                fd_coeffsy[k] * (-fd_coeffs3x[0]) *
                (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] + bx[x2] * m_vyx[i2]);
          }
        }

      } else if (x == nx - 2 - j + (j + 1)) {
        int64_t x2 = x - (j + 1);
        for (k = 0; k <= A; ++k) {
          for (jp = 0; jp < A / 2 - 1; ++jp) {
            if (y == 1 + jp - jp + k) {
              int64_t i2 = i - ((j + 1) + (-jp + k) * nx);
              int64_t i2_noshot = i_noshot - ((j + 1) + (-jp + k) * nx);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vx[i] += fd_coeffs2y[k] * (fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            } else if (y == ny - 2 - jp + jp - k + 1) {
              int64_t i2 = i - ((j + 1) + (jp - k + 1) * nx);
              int64_t i2_noshot = i_noshot - ((j + 1) + (jp - k + 1) * nx);
              DW_DTYPE muyhx =
                  (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
              vx[i] -= fd_coeffs2y[k] * (fd_coeffs3x[0]) *
                       (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +
                        bx[x2] * m_vyx[i2]);
            }
          }
        }
        for (k = 0; k < A / 2; ++k) {
          if (y > 1 + A / 2 - 2 + k + 1 && y < ny - 2 - A / 2 + 2 + k + 1) {
            int64_t i2 = i - ((j + 1) + (k + 1) * nx);
            int64_t i2_noshot = i_noshot - ((j + 1) + (k + 1) * nx);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vx[i] +=
                fd_coeffsy[k] * (fd_coeffs3x[0]) *
                (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] + bx[x2] * m_vyx[i2]);
          }
          if (y > 1 + A / 2 - 2 - k && y < ny - 2 - A / 2 + 2 - k) {
            int64_t i2 = i - ((j + 1) + (-k) * nx);
            int64_t i2_noshot = i_noshot - ((j + 1) + (-k) * nx);
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vx[i] -=
                fd_coeffsy[k] * (fd_coeffs3x[0]) *
                (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] + bx[x2] * m_vyx[i2]);
          }
        }
      }
    }

    // from sigmaxy dvxdy
    if (x > 0 && x < nx - 1) {
      for (k = 1; k <= A + 1; ++k) {
        for (j = 0; j < A / 2 - 1; ++j) {
          if (y == 1 + j + (-j + k - 1)) {
            int64_t y2 = y - (-j + k - 1);
            int64_t i2 = i - (-j + k - 1) * nx;
            int64_t i2_noshot = i_noshot - (-j + k - 1) * nx;
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vx[i] +=
                fd_coeffs3y[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                  byh[y2] * m_vxy[i2]);
          } else if (y == ny - 2 - j + (j - k + 2)) {
            int64_t y2 = y - (j - k + 2);
            int64_t i2 = i - (j - k + 2) * nx;
            int64_t i2_noshot = i_noshot - (j - k + 2) * nx;
            DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
            vx[i] -=
                fd_coeffs3y[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                  byh[y2] * m_vxy[i2]);
          }
        }
      }
      for (k = 0; k < A / 2; ++k) {
        if (y > 1 + A / 2 - 2 + k + 1 && y < ny - 2 - A / 2 + 2 + k + 1) {
          int64_t y2 = y - (k + 1);
          int64_t i2 = i - (k + 1) * nx;
          int64_t i2_noshot = i_noshot - (k + 1) * nx;
          DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
          vx[i] += fd_coeffsy[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                    byh[y2] * m_vxy[i2]);
        }
        if (y > 1 + A / 2 - 2 - k && y < ny - 2 - A / 2 + 2 - k) {
          int64_t y2 = y + k;
          int64_t i2 = i + k * nx;
          int64_t i2_noshot = i_noshot + k * nx;
          DW_DTYPE muyhx = (mu_shot[i2_noshot] + mu_shot[i2_noshot + nx]) / 2;
          vx[i] -= fd_coeffsy[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +
                                    byh[y2] * m_vxy[i2]);
        }
      }
    }

    if (pml_y) {
      m_sigmaxyyn[i] =
          buoyancy_shot[i_noshot] * dt * ay[y] * vx[i] + ay[y] * m_sigmaxyy[i];
    }
    if (pml_x) {
      m_sigmaxxxn[i] =
          buoyancy_shot[i_noshot] * dt * ax[x] * vx[i] + ax[x] * m_sigmaxxx[i];
    }
  }
}

__global__ void backward_kernel_v(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const buoyancy,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE *__restrict const m_vyy,
    DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vxx,
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int64_t batch = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t i_noshot = y * nx + x;
  int64_t i = batch * nynx + i_noshot;
  int64_t j, k;
  DW_DTYPE const *__restrict const lamb_shot =
      lamb_batched ? lamb + batch * nynx : lamb;
  DW_DTYPE const *__restrict const mu_shot =
      mu_batched ? mu + batch * nynx : mu;
  DW_DTYPE const *__restrict const buoyancy_shot =
      buoyancy_batched ? buoyancy + batch * nynx : buoyancy;
  if (y < ny && x < nx - 1) {
    bool pml_y = y < vpml_y0 + 1 || y >= vpml_y1;
    bool pml_x = x < vpml_x0 || x >= vpml_x1 - 1;
    DW_DTYPE lambyxh = (lamb_shot[i_noshot] + lamb_shot[i_noshot + 1]) / 2;
    DW_DTYPE muyxh = (mu_shot[i_noshot] + mu_shot[i_noshot + 1]) / 2;

    if (pml_y) {
      m_vyy[i] = (lambyxh + 2 * muyxh) * dt * ay[y] * sigmayy[i] +
                 lambyxh * dt * ay[y] * sigmaxx[i] + ay[y] * m_vyy[i];
    }
    if (pml_x) {
      m_vxx[i] = (lambyxh + 2 * muyxh) * dt * axh[x] * sigmaxx[i] +
                 lambyxh * dt * axh[x] * sigmayy[i] + axh[x] * m_vxx[i];
    }

    // dsigmayydy
    for (k = 0; k < A; ++k) {
      for (j = 0; j < A / 2; ++j) {
        if (y == j + (1 - j + k)) {
          int64_t i2 = i - (1 - j + k) * nx;
          int64_t i2_noshot = i_noshot - (1 - j + k) * nx;
          int64_t y2 = y - (1 - j + k);
          DW_DTYPE buoyancyyhxh;
          if (y2 == ny - 1) {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
                 buoyancy_shot[i2_noshot + nx] +
                 buoyancy_shot[i2_noshot + nx + 1]) /
                4;
          }
          sigmayy[i] += fd_coeffs1y[j][1 + k] *
                        (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                         byh[y2] * m_sigmayyy[i2]);
        } else if (y == ny - 1 - j + (j - k)) {
          int64_t i2 = i - (j - k) * nx;
          int64_t i2_noshot = i_noshot - (j - k) * nx;
          int64_t y2 = y - (j - k);
          DW_DTYPE buoyancyyhxh;
          if (y2 == ny - 1) {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
                 buoyancy_shot[i2_noshot + nx] +
                 buoyancy_shot[i2_noshot + nx + 1]) /
                4;
          }
          sigmayy[i] -= fd_coeffs1y[j][1 + k] *
                        (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                         byh[y2] * m_sigmayyy[i2]);
        }
      }
    }
    for (k = 0; k < A / 2; ++k) {
      if (y > A / 2 - 1 + (1 + k) && y < ny - 1 - A / 2 + 1 + (1 + k)) {
        int64_t i2 = i - (1 + k) * nx;
        int64_t i2_noshot = i_noshot - (1 + k) * nx;
        int64_t y2 = y - (1 + k);
        DW_DTYPE buoyancyyhxh;
        if (y2 == ny - 1) {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
               buoyancy_shot[i2_noshot + nx] +
               buoyancy_shot[i2_noshot + nx + 1]) /
              4;
        }
        sigmayy[i] +=
            fd_coeffsy[k] * (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                             byh[y2] * m_sigmayyy[i2]);
      }
      if (y > A / 2 - 1 - k && y < ny - 1 - A / 2 + 1 - k) {
        int64_t i2 = i - (-k) * nx;
        int64_t i2_noshot = i_noshot - (-k) * nx;
        int64_t y2 = y - (-k);
        DW_DTYPE buoyancyyhxh;
        if (y2 == ny - 1) {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
               buoyancy_shot[i2_noshot + nx] +
               buoyancy_shot[i2_noshot + nx + 1]) /
              4;
        }
        sigmayy[i] -=
            fd_coeffsy[k] * (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +
                             byh[y2] * m_sigmayyy[i2]);
      }
    }

    // dsigmaxxdx
    for (k = 0; k < A; ++k) {
      for (j = 0; j < A / 2; ++j) {
        if (x == j + (-j + k)) {
          int64_t i2 = i - (-j + k);
          int64_t i2_noshot = i_noshot - (-j + k);
          int64_t x2 = x - (-j + k);
          sigmaxx[i] += fd_coeffs1x[j][1 + k] *
                        (buoyancy_shot[i2_noshot] * dt * (1 + bx[x2]) * vx[i2] +
                         bx[x2] * m_sigmaxxx[i2]);
        } else if (x == nx - 1 - j + (j - k - 1)) {
          int64_t i2 = i - (j - k - 1);
          int64_t i2_noshot = i_noshot - (j - k - 1);
          int64_t x2 = x - (j - k - 1);
          sigmaxx[i] -= fd_coeffs1x[j][1 + k] *
                        (buoyancy_shot[i2_noshot] * dt * (1 + bx[x2]) * vx[i2] +
                         bx[x2] * m_sigmaxxx[i2]);
        }
      }
    }
    for (k = 0; k < A / 2; ++k) {
      if (x > A / 2 - 1 + (k) && x < nx - 1 - A / 2 + 1 + (k)) {
        int64_t i2 = i - (k);
        int64_t i2_noshot = i_noshot - (k);
        int64_t x2 = x - (k);
        sigmaxx[i] += fd_coeffsx[k] *
                      (buoyancy_shot[i2_noshot] * dt * (1 + bx[x2]) * vx[i2] +
                       bx[x2] * m_sigmaxxx[i2]);
      }
      if (x > A / 2 - 1 - (1 + k) && x < nx - 1 - A / 2 + 1 - (1 + k)) {
        int64_t i2 = i + (1 + k);
        int64_t i2_noshot = i_noshot + (1 + k);
        int64_t x2 = x + (1 + k);
        sigmaxx[i] -= fd_coeffsx[k] *
                      (buoyancy_shot[i2_noshot] * dt * (1 + bx[x2]) * vx[i2] +
                       bx[x2] * m_sigmaxxx[i2]);
      }
    }
  }

  if (y < ny - 1 && x >= 1 && x < nx - 1) {
    bool pml_y = y < vpml_y0 + 1 || y >= vpml_y1 - 1;
    bool pml_x = x < vpml_x0 + 1 || x >= vpml_x1 - 1;

    DW_DTYPE muyhx = (mu_shot[i_noshot] + mu_shot[i_noshot + nx]) / 2;

    if (pml_y) {
      m_vxy[i] = muyhx * dt * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i];
    }
    if (pml_x) {
      m_vyx[i] = muyhx * dt * ax[x] * sigmaxy[i] + ax[x] * m_vyx[i];
    }

    // dsigmaxydx
    for (k = 0; k < A; ++k) {
      for (j = 0; j < A / 2 - 1; ++j) {
        if (x == j - j + 1 + k) {
          int64_t i2 = i - (-j + 1 + k);
          int64_t i2_noshot = i_noshot - (-j + 1 + k);
          int64_t x2 = x - (-j + 1 + k);
          DW_DTYPE buoyancyyhxh;
          if (y == ny - 1) {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
                 buoyancy_shot[i2_noshot + nx] +
                 buoyancy_shot[i2_noshot + nx + 1]) /
                4;
          }
          sigmaxy[i] +=
              fd_coeffs2x[1 + k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                                    bxh[x2] * m_sigmaxyx[i2]);
        } else if (x == nx - 2 - j + j - k) {
          int64_t i2 = i - (j - k);
          int64_t i2_noshot = i_noshot - (j - k);
          int64_t x2 = x - (j - k);
          DW_DTYPE buoyancyyhxh;
          if (y == ny - 1) {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
          } else {
            buoyancyyhxh =
                (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
                 buoyancy_shot[i2_noshot + nx] +
                 buoyancy_shot[i2_noshot + nx + 1]) /
                4;
          }
          sigmaxy[i] -=
              fd_coeffs2x[1 + k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                                    bxh[x2] * m_sigmaxyx[i2]);
        }
      }
    }
    for (k = 0; k < A / 2; ++k) {
      if (x > A / 2 - 2 + 1 + k && x < nx - 2 - A / 2 + 2 + 1 + k) {
        int64_t i2 = i - (1 + k);
        int64_t i2_noshot = i_noshot - (1 + k);
        int64_t x2 = x - (1 + k);
        DW_DTYPE buoyancyyhxh;
        if (y == ny - 1) {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
               buoyancy_shot[i2_noshot + nx] +
               buoyancy_shot[i2_noshot + nx + 1]) /
              4;
        }
        sigmaxy[i] +=
            fd_coeffsx[k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                             bxh[x2] * m_sigmaxyx[i2]);
      }
      if (x > A / 2 - 2 - k && x < nx - 2 - A / 2 + 2 - k) {
        int64_t i2 = i - (-k);
        int64_t i2_noshot = i_noshot - (-k);
        int64_t x2 = x - (-k);
        DW_DTYPE buoyancyyhxh;
        if (y == ny - 1) {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1]) / 2;
        } else {
          buoyancyyhxh =
              (buoyancy_shot[i2_noshot] + buoyancy_shot[i2_noshot + 1] +
               buoyancy_shot[i2_noshot + nx] +
               buoyancy_shot[i2_noshot + nx + 1]) /
              4;
        }
        sigmaxy[i] -=
            fd_coeffsx[k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +
                             bxh[x2] * m_sigmaxyx[i2]);
      }
    }

    // dsigmaxydy
    for (k = 0; k < A; ++k) {
      for (j = 0; j < A / 2 - 1; ++j) {
        if (y == 1 + j - j + k) {
          int64_t i2 = i - (-j + k) * nx;
          int64_t i2_noshot = i_noshot - (-j + k) * nx;
          int64_t y2 = y - (-j + k);
          sigmaxy[i] += fd_coeffs2y[1 + k] *
                        (buoyancy_shot[i2_noshot] * dt * (1 + by[y2]) * vx[i2] +
                         by[y2] * m_sigmaxyy[i2]);
        } else if (y == ny - 1 - j + j - k - 1) {
          int64_t i2 = i - (j - k - 1) * nx;
          int64_t i2_noshot = i_noshot - (j - k - 1) * nx;
          int64_t y2 = y - (j - k - 1);
          sigmaxy[i] -= fd_coeffs2y[1 + k] *
                        (buoyancy_shot[i2_noshot] * dt * (1 + by[y2]) * vx[i2] +
                         by[y2] * m_sigmaxyy[i2]);
        }
      }
    }
    for (k = 0; k < A / 2; ++k) {
      if (y > 1 + A / 2 - 2 + k && y < ny - 1 - A / 2 + 2 + k) {
        int64_t i2 = i - (k)*nx;
        int64_t i2_noshot = i_noshot - (k)*nx;
        int64_t y2 = y - (k);
        sigmaxy[i] += fd_coeffsy[k] *
                      (buoyancy_shot[i2_noshot] * dt * (1 + by[y2]) * vx[i2] +
                       by[y2] * m_sigmaxyy[i2]);
      }
      if (y > 1 + A / 2 - 2 - (k + 1) && y < ny - 1 - A / 2 + 2 - (k + 1)) {
        int64_t i2 = i - (-(k + 1)) * nx;
        int64_t i2_noshot = i_noshot - (-(k + 1)) * nx;
        int64_t y2 = y - (-(k + 1));
        sigmaxy[i] -= fd_coeffsy[k] *
                      (buoyancy_shot[i2_noshot] * dt * (1 + by[y2]) * vx[i2] +
                       by[y2] * m_sigmaxyy[i2]);
      }
    }
  }
}

inline void gpuAssert(cudaError_t code, const char *file, int line,
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

static void set_fd_coeffs(DW_DTYPE fd_coeffs_h[2], DW_DTYPE fd_coeffs1_h[2][5],
                          DW_DTYPE fd_coeffs2_h[5], DW_DTYPE fd_coeffs3_h[6],
                          DW_DTYPE const dx) {
  if (DW_ACCURACY == 2) {
    fd_coeffs_h[0] = (DW_DTYPE)(1.0 / 1.0) / dx;
    fd_coeffs1_h[0][0] = (DW_DTYPE)(-8.0 / 3.0) / dx;
    fd_coeffs1_h[0][1] = (DW_DTYPE)(3.0 / 1.0) / dx;
    fd_coeffs1_h[0][2] = (DW_DTYPE)(-1.0 / 3.0) / dx;
  } else {
    fd_coeffs_h[0] = (DW_DTYPE)(9.0 / 8.0) / dx;
    fd_coeffs_h[1] = (DW_DTYPE)(-1.0 / 24.0) / dx;
    fd_coeffs1_h[0][0] = (DW_DTYPE)(-352.0 / 105.0) / dx;
    fd_coeffs1_h[0][1] = (DW_DTYPE)(35.0 / 8.0) / dx;
    fd_coeffs1_h[0][2] = (DW_DTYPE)(-35.0 / 24.0) / dx;
    fd_coeffs1_h[0][3] = (DW_DTYPE)(21.0 / 40.0) / dx;
    fd_coeffs1_h[0][4] = (DW_DTYPE)(-5.0 / 56.0) / dx;
    fd_coeffs1_h[1][0] = (DW_DTYPE)(16.0 / 105.0) / dx;
    fd_coeffs1_h[1][1] = (DW_DTYPE)(-31.0 / 24.0) / dx;
    fd_coeffs1_h[1][2] = (DW_DTYPE)(29.0 / 24.0) / dx;
    fd_coeffs1_h[1][3] = (DW_DTYPE)(-3.0 / 40.0) / dx;
    fd_coeffs1_h[1][4] = (DW_DTYPE)(1.0 / 168.0) / dx;
    fd_coeffs2_h[0] = (DW_DTYPE)(-11.0 / 12.0) / dx;
    fd_coeffs2_h[1] = (DW_DTYPE)(17.0 / 24.0) / dx;
    fd_coeffs2_h[2] = (DW_DTYPE)(3.0 / 8.0) / dx;
    fd_coeffs2_h[3] = (DW_DTYPE)(-5.0 / 24.0) / dx;
    fd_coeffs2_h[4] = (DW_DTYPE)(1.0 / 24.0) / dx;
    fd_coeffs3_h[0] = (DW_DTYPE)(-71.0 / 1689.0);
    fd_coeffs3_h[1] = (DW_DTYPE)(-14587.0 / 13512.0) / dx;
    fd_coeffs3_h[2] = (DW_DTYPE)(11243.0 / 10134.0) / dx;
    fd_coeffs3_h[3] = (DW_DTYPE)(-43.0 / 2252.0) / dx;
    fd_coeffs3_h[4] = (DW_DTYPE)(-47.0 / 3378.0) / dx;
    fd_coeffs3_h[5] = (DW_DTYPE)(127.0 / 40536.0) / dx;
  }
}

void set_config(DW_DTYPE const dt_h, DW_DTYPE const dy, DW_DTYPE const dx,
                int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
                int64_t const n_sources_y_per_shot_h,
                int64_t const n_sources_x_per_shot_h,
                int64_t const n_receivers_y_per_shot_h,
                int64_t const n_receivers_x_per_shot_h,
                int64_t const n_receivers_p_per_shot_h,
                int64_t const step_ratio_h, int64_t const pml_y0_h,
                int64_t const pml_y1_h, int64_t const pml_x0_h,
                int64_t const pml_x1_h, int64_t const spml_y0_h,
                int64_t const spml_y1_h, int64_t const spml_x0_h,
                int64_t const spml_x1_h, int64_t const vpml_y0_h,
                int64_t const vpml_y1_h, int64_t const vpml_x0_h,
                int64_t const vpml_x1_h, bool const lamb_batched_h,
                bool const mu_batched_h, bool const buoyancy_batched_h) {
  int64_t const nynx_h = ny_h * nx_h;
  DW_DTYPE fd_coeffsy_h[2], fd_coeffsx_h[2], fd_coeffs1y_h[2][5],
      fd_coeffs1x_h[2][5], fd_coeffs2y_h[5], fd_coeffs2x_h[5], fd_coeffs3y_h[6],
      fd_coeffs3x_h[6];

  set_fd_coeffs(fd_coeffsy_h, fd_coeffs1y_h, fd_coeffs2y_h, fd_coeffs3y_h, dy);
  gpuErrchk(cudaMemcpyToSymbol(fd_coeffsy, fd_coeffsy_h, sizeof(DW_DTYPE) * 2));
  gpuErrchk(
      cudaMemcpyToSymbol(fd_coeffs1y, fd_coeffs1y_h[0], sizeof(DW_DTYPE) * 5));
  gpuErrchk(cudaMemcpyToSymbol(fd_coeffs1y, fd_coeffs1y_h[1],
                               sizeof(DW_DTYPE) * 5, sizeof(DW_DTYPE) * 5));
  gpuErrchk(
      cudaMemcpyToSymbol(fd_coeffs2y, fd_coeffs2y_h, sizeof(DW_DTYPE) * 5));
  gpuErrchk(
      cudaMemcpyToSymbol(fd_coeffs3y, fd_coeffs3y_h, sizeof(DW_DTYPE) * 6));
  set_fd_coeffs(fd_coeffsx_h, fd_coeffs1x_h, fd_coeffs2x_h, fd_coeffs3x_h, dx);
  gpuErrchk(cudaMemcpyToSymbol(fd_coeffsx, fd_coeffsx_h, sizeof(DW_DTYPE) * 2));
  gpuErrchk(
      cudaMemcpyToSymbol(fd_coeffs1x, fd_coeffs1x_h[0], sizeof(DW_DTYPE) * 5));
  gpuErrchk(cudaMemcpyToSymbol(fd_coeffs1x, fd_coeffs1x_h[1],
                               sizeof(DW_DTYPE) * 5, sizeof(DW_DTYPE) * 5));
  gpuErrchk(
      cudaMemcpyToSymbol(fd_coeffs2x, fd_coeffs2x_h, sizeof(DW_DTYPE) * 5));
  gpuErrchk(
      cudaMemcpyToSymbol(fd_coeffs3x, fd_coeffs3x_h, sizeof(DW_DTYPE) * 6));

  gpuErrchk(cudaMemcpyToSymbol(dt, &dt_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nynx, &nynx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_y_per_shot, &n_sources_y_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_x_per_shot, &n_sources_x_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_y_per_shot,
                               &n_receivers_y_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_x_per_shot,
                               &n_receivers_x_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_p_per_shot,
                               &n_receivers_p_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(spml_y0, &spml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(spml_y1, &spml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(spml_x0, &spml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(spml_x1, &spml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(vpml_y0, &vpml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(vpml_y1, &vpml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(vpml_x0, &vpml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(vpml_x1, &vpml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(lamb_batched, &lamb_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(mu_batched, &mu_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(buoyancy_batched, &buoyancy_batched_h, sizeof(bool)));
}

void backward_batch(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const buoyancy,
    DW_DTYPE const *__restrict const grad_r_y,
    DW_DTYPE const *__restrict const grad_r_x,
    DW_DTYPE const *__restrict const grad_r_p, DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmayy,
    DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy, DW_DTYPE *__restrict const m_vxx,
    DW_DTYPE *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmayyyn,
    DW_DTYPE *__restrict const m_sigmaxyyn,
    DW_DTYPE *__restrict const m_sigmaxyxn,
    DW_DTYPE *__restrict const m_sigmaxxxn,
    DW_DTYPE const *__restrict const dvydbuoyancy,
    DW_DTYPE const *__restrict const dvxdbuoyancy,
    DW_DTYPE const *__restrict const dvydy_store,
    DW_DTYPE const *__restrict const dvxdx_store,
    DW_DTYPE const *__restrict const dvydxdvxdy_store,
    DW_DTYPE *__restrict const grad_f_y, DW_DTYPE *__restrict const grad_f_x,
    DW_DTYPE *__restrict const grad_lamb, DW_DTYPE *__restrict const grad_mu,
    DW_DTYPE *__restrict const grad_buoyancy,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    int64_t const *__restrict const sources_y_i,
    int64_t const *__restrict const sources_x_i,
    int64_t const *__restrict const receivers_y_i,
    int64_t const *__restrict const receivers_x_i,
    int64_t const *__restrict const receivers_p_i, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_y_per_shot_h, int64_t const n_sources_x_per_shot_h,
    int64_t const n_receivers_y_per_shot_h, int64_t n_receivers_x_per_shot_h,
    int64_t const n_receivers_p_per_shot_h, bool const lamb_requires_grad,
    bool const mu_requires_grad, bool const buoyancy_requires_grad) {
  dim3 dimBlock(32, 8, 1);
  unsigned int gridx = ceil_div(nx_h, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources_y =
      ceil_div(n_sources_y_per_shot_h, dimBlock_sources.x);
  unsigned int gridx_sources_x =
      ceil_div(n_sources_x_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources_y(gridx_sources_y, gridy_sources, gridz_sources);
  dim3 dimGrid_sources_x(gridx_sources_x, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers_y =
      ceil_div(n_receivers_y_per_shot_h, dimBlock_receivers.x);
  unsigned int gridx_receivers_x =
      ceil_div(n_receivers_x_per_shot_h, dimBlock_receivers.x);
  unsigned int gridx_receivers_p =
      ceil_div(n_receivers_p_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers_y(gridx_receivers_y, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_x(gridx_receivers_x, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_p(gridx_receivers_p, gridy_receivers, gridz_receivers);
  if (lamb_requires_grad) {
    add_to_grad_lamb<<<dimGrid, dimBlock>>>(grad_lamb, sigmayy, sigmaxx,
                                            dvydy_store, dvxdx_store);
    CHECK_KERNEL_ERROR
  }
  if (mu_requires_grad) {
    add_to_grad_mu<<<dimGrid, dimBlock>>>(grad_mu, sigmayy, sigmaxy, sigmaxx,
                                          dvydy_store, dvxdx_store,
                                          dvydxdvxdy_store);
    CHECK_KERNEL_ERROR
  }
  backward_kernel_sigma<<<dimGrid, dimBlock>>>(
      lamb, mu, buoyancy, vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx,
      m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, m_sigmayyyn,
      m_sigmaxyyn, m_sigmaxyxn, m_sigmaxxxn, ay, ayh, ax, axh, by, byh, bx,
      bxh);
  CHECK_KERNEL_ERROR
  if (n_sources_y_per_shot_h > 0) {
    record_adjoint_receivers_y<<<dimGrid_sources_y, dimBlock_sources>>>(
        grad_f_y, vy, sources_y_i);
    CHECK_KERNEL_ERROR
  }
  if (n_sources_x_per_shot_h > 0) {
    record_adjoint_receivers_x<<<dimGrid_sources_x, dimBlock_sources>>>(
        grad_f_x, vx, sources_x_i);
    CHECK_KERNEL_ERROR
  }
  if (buoyancy_requires_grad) {
    add_to_grad_buoyancy<<<dimGrid, dimBlock>>>(grad_buoyancy, vy, vx,
                                                dvydbuoyancy, dvxdbuoyancy);
    CHECK_KERNEL_ERROR
  }
  backward_kernel_v<<<dimGrid, dimBlock>>>(
      lamb, mu, buoyancy, vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx,
      m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, ay, ayh, ax,
      axh, by, byh, bx, bxh);
  CHECK_KERNEL_ERROR
  if (n_receivers_y_per_shot_h > 0) {
    add_adjoint_sources_y<<<dimGrid_receivers_y, dimBlock_receivers>>>(
        vy, grad_r_y, receivers_y_i);
    CHECK_KERNEL_ERROR
  }
  if (n_receivers_x_per_shot_h > 0) {
    add_adjoint_sources_x<<<dimGrid_receivers_x, dimBlock_receivers>>>(
        vx, grad_r_x, receivers_x_i);
    CHECK_KERNEL_ERROR
  }
  if (n_receivers_p_per_shot_h > 0) {
    add_adjoint_pressure_sources<<<dimGrid_receivers_p, dimBlock_receivers>>>(
        sigmayy, sigmaxx, grad_r_p, receivers_p_i);
    CHECK_KERNEL_ERROR
  }
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(forward)(
            DW_DTYPE const *__restrict const lamb,
            DW_DTYPE const *__restrict const mu,
            DW_DTYPE const *__restrict const buoyancy,
            DW_DTYPE const *__restrict const f_y,
            DW_DTYPE const *__restrict const f_x, DW_DTYPE *__restrict const vy,
            DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmayy,
            DW_DTYPE *__restrict const sigmaxy,
            DW_DTYPE *__restrict const sigmaxx,
            DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
            DW_DTYPE *__restrict const m_vxy, DW_DTYPE *__restrict const m_vxx,
            DW_DTYPE *__restrict const m_sigmayyy,
            DW_DTYPE *__restrict const m_sigmaxyy,
            DW_DTYPE *__restrict const m_sigmaxyx,
            DW_DTYPE *__restrict const m_sigmaxxx,
            DW_DTYPE *__restrict const dvydbuoyancy,
            DW_DTYPE *__restrict const dvxdbuoyancy,
            DW_DTYPE *__restrict const dvydy_store,
            DW_DTYPE *__restrict const dvxdx_store,
            DW_DTYPE *__restrict const dvydxdvxdy_store,
            DW_DTYPE *__restrict const r_y, DW_DTYPE *__restrict const r_x,
            DW_DTYPE *__restrict const r_p, DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ayh,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const axh,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const byh,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const bxh,
            int64_t const *__restrict const sources_y_i,
            int64_t const *__restrict const sources_x_i,
            int64_t const *__restrict const receivers_y_i,
            int64_t const *__restrict const receivers_x_i,
            int64_t const *__restrict const receivers_p_i, DW_DTYPE const dy,
            DW_DTYPE const dx, DW_DTYPE const dt_h, int64_t const nt,
            int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_y_per_shot_h,
            int64_t const n_sources_x_per_shot_h,
            int64_t const n_receivers_y_per_shot_h,
            int64_t n_receivers_x_per_shot_h,
            int64_t const n_receivers_p_per_shot_h, int64_t const step_ratio_h,
            bool const lamb_requires_grad, bool const mu_requires_grad,
            bool const buoyancy_requires_grad, bool const lamb_batched_h,
            bool const mu_batched_h, bool const buoyancy_batched_h,
            int64_t const start_t, int64_t const pml_y0_h,
            int64_t const pml_y1_h, int64_t const pml_x0_h,
            int64_t const pml_x1_h, int64_t const device) {

  dim3 dimBlock(32, 8, 1);
  unsigned int gridx = ceil_div(nx_h, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources_y =
      ceil_div(n_sources_y_per_shot_h, dimBlock_sources.x);
  unsigned int gridx_sources_x =
      ceil_div(n_sources_x_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources_y(gridx_sources_y, gridy_sources, gridz_sources);
  dim3 dimGrid_sources_x(gridx_sources_x, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers_y =
      ceil_div(n_receivers_y_per_shot_h, dimBlock_receivers.x);
  unsigned int gridx_receivers_x =
      ceil_div(n_receivers_x_per_shot_h, dimBlock_receivers.x);
  unsigned int gridx_receivers_p =
      ceil_div(n_receivers_p_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers_y(gridx_receivers_y, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_x(gridx_receivers_x, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_p(gridx_receivers_p, gridy_receivers, gridz_receivers);

  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt_h, dy, dx, n_shots_h, ny_h, nx_h, n_sources_y_per_shot_h,
             n_sources_x_per_shot_h, n_receivers_y_per_shot_h,
             n_receivers_x_per_shot_h, n_receivers_p_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, 0, 0, 0, 0, 0, 0, 0, 0,
             lamb_batched_h, mu_batched_h, buoyancy_batched_h);

  for (t = start_t; t < start_t + nt; ++t) {
    if (n_receivers_y_per_shot_h > 0) {
      record_receivers_y<<<dimGrid_receivers_y, dimBlock_receivers>>>(
          r_y + t * n_shots_h * n_receivers_y_per_shot_h, vy, receivers_y_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_x_per_shot_h > 0) {
      record_receivers_x<<<dimGrid_receivers_x, dimBlock_receivers>>>(
          r_x + t * n_shots_h * n_receivers_x_per_shot_h, vx, receivers_x_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_p_per_shot_h > 0) {
      record_pressure_receivers<<<dimGrid_receivers_p, dimBlock_receivers>>>(
          r_p + t * n_shots_h * n_receivers_p_per_shot_h, sigmayy, sigmaxx,
          receivers_p_i);
      CHECK_KERNEL_ERROR
    }
    forward_kernel_v<<<dimGrid, dimBlock>>>(
        buoyancy, vy, vx, sigmayy, sigmaxy, sigmaxx, m_sigmayyy, m_sigmaxyy,
        m_sigmaxyx, m_sigmaxxx,
        dvydbuoyancy + (t / step_ratio_h) * ny_h * nx_h * n_shots_h,
        dvxdbuoyancy + (t / step_ratio_h) * ny_h * nx_h * n_shots_h, ay, ayh,
        ax, axh, by, byh, bx, bxh,
        buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR
    if (n_sources_y_per_shot_h > 0) {
      add_sources_y<<<dimGrid_sources_y, dimBlock_sources>>>(
          vy, f_y + t * n_shots_h * n_sources_y_per_shot_h, sources_y_i);
      CHECK_KERNEL_ERROR
    }
    if (n_sources_x_per_shot_h > 0) {
      add_sources_x<<<dimGrid_sources_x, dimBlock_sources>>>(
          vx, f_x + t * n_shots_h * n_sources_x_per_shot_h, sources_x_i);
      CHECK_KERNEL_ERROR
    }
    forward_kernel_sigma<<<dimGrid, dimBlock>>>(
        lamb, mu, vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
        dvydy_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h,
        dvxdx_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h,
        dvydxdvxdy_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h, ay,
        ayh, ax, axh, by, byh, bx, bxh,
        lamb_requires_grad && ((t % step_ratio_h) == 0),
        mu_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR
  }
  if (n_receivers_y_per_shot_h > 0) {
    record_receivers_y<<<dimGrid_receivers_y, dimBlock_receivers>>>(
        r_y + t * n_shots_h * n_receivers_y_per_shot_h, vy, receivers_y_i);
    CHECK_KERNEL_ERROR
  }
  if (n_receivers_x_per_shot_h > 0) {
    record_receivers_x<<<dimGrid_receivers_x, dimBlock_receivers>>>(
        r_x + t * n_shots_h * n_receivers_x_per_shot_h, vx, receivers_x_i);
    CHECK_KERNEL_ERROR
  }
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(backward)(
            DW_DTYPE const *__restrict const lamb,
            DW_DTYPE const *__restrict const mu,
            DW_DTYPE const *__restrict const buoyancy,
            DW_DTYPE const *__restrict const grad_r_y,
            DW_DTYPE const *__restrict const grad_r_x,
            DW_DTYPE const *__restrict const grad_r_p,
            DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
            DW_DTYPE *__restrict const sigmayy,
            DW_DTYPE *__restrict const sigmaxy,
            DW_DTYPE *__restrict const sigmaxx,
            DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
            DW_DTYPE *__restrict const m_vxy, DW_DTYPE *__restrict const m_vxx,
            DW_DTYPE *__restrict const m_sigmayyy,
            DW_DTYPE *__restrict const m_sigmaxyy,
            DW_DTYPE *__restrict const m_sigmaxyx,
            DW_DTYPE *__restrict const m_sigmaxxx,
            DW_DTYPE *__restrict const m_sigmayyyn,
            DW_DTYPE *__restrict const m_sigmaxyyn,
            DW_DTYPE *__restrict const m_sigmaxyxn,
            DW_DTYPE *__restrict const m_sigmaxxxn,
            DW_DTYPE const *__restrict const dvydbuoyancy,
            DW_DTYPE const *__restrict const dvxdbuoyancy,
            DW_DTYPE const *__restrict const dvydy_store,
            DW_DTYPE const *__restrict const dvxdx_store,
            DW_DTYPE const *__restrict const dvydxdvxdy_store,
            DW_DTYPE *__restrict const grad_f_y,
            DW_DTYPE *__restrict const grad_f_x,
            DW_DTYPE *__restrict const grad_lamb,
            DW_DTYPE *__restrict const grad_lamb_shot,
            DW_DTYPE *__restrict const grad_mu,
            DW_DTYPE *__restrict const grad_mu_shot,
            DW_DTYPE *__restrict const grad_buoyancy,
            DW_DTYPE *__restrict const grad_buoyancy_shot,
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ayh,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const axh,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const byh,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const bxh,
            int64_t const *__restrict const sources_y_i,
            int64_t const *__restrict const sources_x_i,
            int64_t const *__restrict const receivers_y_i,
            int64_t const *__restrict const receivers_x_i,
            int64_t const *__restrict const receivers_p_i, DW_DTYPE const dy,
            DW_DTYPE const dx, DW_DTYPE const dt_h, int64_t const nt,
            int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_y_per_shot_h,
            int64_t const n_sources_x_per_shot_h,
            int64_t const n_receivers_y_per_shot_h,
            int64_t n_receivers_x_per_shot_h,
            int64_t const n_receivers_p_per_shot_h, int64_t const step_ratio_h,
            bool const lamb_requires_grad, bool const mu_requires_grad,
            bool const buoyancy_requires_grad, bool const lamb_batched_h,
            bool const mu_batched_h, bool const buoyancy_batched_h,
            int64_t const start_t, int64_t const spml_y0_h,
            int64_t const spml_y1_h, int64_t const spml_x0_h,
            int64_t const spml_x1_h, int64_t const vpml_y0_h,
            int64_t const vpml_y1_h, int64_t const vpml_x0_h,
            int64_t const vpml_x1_h, int64_t const device) {
  dim3 dimBlock_combine(32, 32, 1);
  unsigned int gridx_combine = ceil_div(nx_h, dimBlock_combine.x);
  unsigned int gridy_combine = ceil_div(ny_h, dimBlock_combine.y);
  unsigned int gridz_combine = 1;
  dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers_y =
      ceil_div(n_receivers_y_per_shot_h, dimBlock_receivers.x);
  unsigned int gridx_receivers_x =
      ceil_div(n_receivers_x_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers_y(gridx_receivers_y, gridy_receivers, gridz_receivers);
  dim3 dimGrid_receivers_x(gridx_receivers_x, gridy_receivers, gridz_receivers);
  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt_h, dy, dx, n_shots_h, ny_h, nx_h, n_sources_y_per_shot_h,
             n_sources_x_per_shot_h, n_receivers_y_per_shot_h,
             n_receivers_x_per_shot_h, n_receivers_p_per_shot_h, step_ratio_h,
             0, 0, 0, 0, spml_y0_h, spml_y1_h, spml_x0_h, spml_x1_h, vpml_y0_h,
             vpml_y1_h, vpml_x0_h, vpml_x1_h, lamb_batched_h, mu_batched_h,
             buoyancy_batched_h);
  if (n_receivers_y_per_shot_h > 0) {
    add_adjoint_sources_y<<<dimGrid_receivers_y, dimBlock_receivers>>>(
        vy, grad_r_y + nt * n_shots_h * n_receivers_y_per_shot_h,
        receivers_y_i);
    CHECK_KERNEL_ERROR
  }
  if (n_receivers_x_per_shot_h > 0) {
    add_adjoint_sources_x<<<dimGrid_receivers_x, dimBlock_receivers>>>(
        vx, grad_r_x + nt * n_shots_h * n_receivers_x_per_shot_h,
        receivers_x_i);
    CHECK_KERNEL_ERROR
  }
  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t store_i = (t / step_ratio_h) * n_shots_h * ny_h * nx_h;
    if ((start_t - 1 - t) & 1) {
      backward_batch(
          lamb, mu, buoyancy,
          grad_r_y + t * n_shots_h * n_receivers_y_per_shot_h,
          grad_r_x + t * n_shots_h * n_receivers_x_per_shot_h,
          grad_r_p + t * n_shots_h * n_receivers_p_per_shot_h, vy, vx, sigmayy,
          sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyyn,
          m_sigmaxyyn, m_sigmaxyxn, m_sigmaxxxn, m_sigmayyy, m_sigmaxyy,
          m_sigmaxyx, m_sigmaxxx, dvydbuoyancy + store_i,
          dvxdbuoyancy + store_i, dvydy_store + store_i, dvxdx_store + store_i,
          dvydxdvxdy_store + store_i,
          grad_f_y + t * n_shots_h * n_sources_y_per_shot_h,
          grad_f_x + t * n_shots_h * n_sources_x_per_shot_h, grad_lamb_shot,
          grad_mu_shot, grad_buoyancy_shot, ay, ayh, ax, axh, by, byh, bx, bxh,
          sources_y_i, sources_x_i, receivers_y_i, receivers_x_i, receivers_p_i,
          n_shots_h, ny_h, nx_h, n_sources_y_per_shot_h, n_sources_x_per_shot_h,
          n_receivers_y_per_shot_h, n_receivers_x_per_shot_h,
          n_receivers_p_per_shot_h,
          lamb_requires_grad && ((t % step_ratio_h) == 0),
          mu_requires_grad && ((t % step_ratio_h) == 0),
          buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    } else {
      backward_batch(
          lamb, mu, buoyancy,
          grad_r_y + t * n_shots_h * n_receivers_y_per_shot_h,
          grad_r_x + t * n_shots_h * n_receivers_x_per_shot_h,
          grad_r_p + t * n_shots_h * n_receivers_p_per_shot_h, vy, vx, sigmayy,
          sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy,
          m_sigmaxyx, m_sigmaxxx, m_sigmayyyn, m_sigmaxyyn, m_sigmaxyxn,
          m_sigmaxxxn, dvydbuoyancy + store_i, dvxdbuoyancy + store_i,
          dvydy_store + store_i, dvxdx_store + store_i,
          dvydxdvxdy_store + store_i,
          grad_f_y + t * n_shots_h * n_sources_y_per_shot_h,
          grad_f_x + t * n_shots_h * n_sources_x_per_shot_h, grad_lamb_shot,
          grad_mu_shot, grad_buoyancy_shot, ay, ayh, ax, axh, by, byh, bx, bxh,
          sources_y_i, sources_x_i, receivers_y_i, receivers_x_i, receivers_p_i,
          n_shots_h, ny_h, nx_h, n_sources_y_per_shot_h, n_sources_x_per_shot_h,
          n_receivers_y_per_shot_h, n_receivers_x_per_shot_h,
          n_receivers_p_per_shot_h,
          lamb_requires_grad && ((t % step_ratio_h) == 0),
          mu_requires_grad && ((t % step_ratio_h) == 0),
          buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    }
  }
  if (lamb_requires_grad && !lamb_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_lamb,
                                                        grad_lamb_shot);
    CHECK_KERNEL_ERROR
  }
  if (mu_requires_grad && !mu_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_mu, grad_mu_shot);
    CHECK_KERNEL_ERROR
  }
  if (buoyancy_requires_grad && !buoyancy_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_buoyancy,
                                                        grad_buoyancy_shot);
    CHECK_KERNEL_ERROR
  }
}

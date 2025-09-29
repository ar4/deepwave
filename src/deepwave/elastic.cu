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

#include "common_gpu.h"
#include "staggered_grid.h"

#define CAT_I(name, accuracy, dtype, device) \
  elastic_iso_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

#define VY(dy, dx) vy[i + dy * nx + dx]
#define VX(dy, dx) vx[i + dy * nx + dx]
#define SIGMAYY(dy, dx) sigmayy[i + dy * nx + dx]
#define SIGMAXX(dy, dx) sigmaxx[i + dy * nx + dx]
#define SIGMAXY(dy, dx) sigmaxy[i + dy * nx + dx]
#define LAMB(dy, dx) lamb_shot[j + dy * nx + dx]
#define MU(dy, dx) mu_shot[j + dy * nx + dx]
#define MU_YX(dy, dx) mu_yx_shot[j + dy * nx + dx]
#define BUOYANCY_Y(dy, dx) buoyancy_y_shot[j + dy * nx + dx]
#define BUOYANCY_X(dy, dx) buoyancy_x_shot[j + dy * nx + dx]
#define M_VYY(dy, dx) m_vyy[i + dy * nx + dx]
#define M_VYX(dy, dx) m_vyx[i + dy * nx + dx]
#define M_VXY(dy, dx) m_vxy[i + dy * nx + dx]
#define M_VXX(dy, dx) m_vxx[i + dy * nx + dx]
#define M_SIGMAYYY(dy, dx) m_sigmayyy[i + dy * nx + dx]
#define M_SIGMAXYY(dy, dx) m_sigmaxyy[i + dy * nx + dx]
#define M_SIGMAXYX(dy, dx) m_sigmaxyx[i + dy * nx + dx]
#define M_SIGMAXXX(dy, dx) m_sigmaxxx[i + dy * nx + dx]

// Access terms used in the backward pass
#define LAMB_2MU(dy, dx) (LAMB(dy, dx) + 2 * MU(dy, dx))
#define VY_Y(dy, dx) \
  (dt * (LAMB_2MU(dy, dx) * SIGMAYY(dy, dx) + LAMB(dy, dx) * SIGMAXX(dy, dx)))
#define VY_X(dy, dx) (dt * MU_YX(dy, dx) * SIGMAXY(dy, dx))
#define VY_Y_PML(dy, dx)                                                       \
  (dt * (1 + by[y + dy]) *                                                     \
       (LAMB_2MU(dy, dx) * SIGMAYY(dy, dx) + LAMB(dy, dx) * SIGMAXX(dy, dx)) + \
   by[y + dy] * M_VYY(dy, dx))
#define VY_X_PML(dy, dx)                                      \
  (dt * (1 + bxh[x + dx]) * MU_YX(dy, dx) * SIGMAXY(dy, dx) + \
   bxh[x + dx] * M_VYX(dy, dx))

#define VX_Y(dy, dx) (dt * MU_YX(dy, dx) * SIGMAXY(dy, dx))
#define VX_X(dy, dx) \
  (dt * (LAMB(dy, dx) * SIGMAYY(dy, dx) + LAMB_2MU(dy, dx) * SIGMAXX(dy, dx)))
#define VX_Y_PML(dy, dx)                                      \
  (dt * (1 + byh[y + dy]) * MU_YX(dy, dx) * SIGMAXY(dy, dx) + \
   byh[y + dy] * M_VXY(dy, dx))
#define VX_X_PML(dy, dx)                                                       \
  (dt * (1 + bx[x + dx]) *                                                     \
       (LAMB(dy, dx) * SIGMAYY(dy, dx) + LAMB_2MU(dy, dx) * SIGMAXX(dy, dx)) + \
   bx[x + dx] * M_VXX(dy, dx))
#define SIGMAYY_Y_PML(dy, dx)                                 \
  ((1 + byh[y + dy]) * dt * BUOYANCY_Y(dy, dx) * VY(dy, dx) + \
   byh[y + dy] * M_SIGMAYYY(dy, dx))
#define SIGMAYY_Y(dy, dx) (dt * BUOYANCY_Y(dy, dx) * VY(dy, dx))
#define SIGMAXX_X_PML(dy, dx)                                 \
  ((1 + bxh[x + dx]) * dt * BUOYANCY_X(dy, dx) * VX(dy, dx) + \
   bxh[x + dx] * M_SIGMAXXX(dy, dx))
#define SIGMAXX_X(dy, dx) (dt * BUOYANCY_X(dy, dx) * VX(dy, dx))
#define SIGMAXY_Y_PML(dy, dx)                                \
  ((1 + by[y + dy]) * dt * BUOYANCY_X(dy, dx) * VX(dy, dx) + \
   by[y + dy] * M_SIGMAXYY(dy, dx))
#define SIGMAXY_Y(dy, dx) (dt * BUOYANCY_X(dy, dx) * VX(dy, dx))
#define SIGMAXY_X_PML(dy, dx)                                \
  ((1 + bx[x + dx]) * dt * BUOYANCY_Y(dy, dx) * VY(dy, dx) + \
   bx[x + dx] * M_SIGMAXYX(dy, dx))
#define SIGMAXY_X(dy, dx) (dt * BUOYANCY_Y(dy, dx) * VY(dy, dx))

#define MAX(a, b) (a > b ? a : b)

namespace {
__constant__ DW_DTYPE dt;
__constant__ DW_DTYPE rdy;
__constant__ DW_DTYPE rdx;
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
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x, DW_DTYPE *__restrict const vy,
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
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD && x < nx - FD_PAD + 1) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    DW_DTYPE const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y[i] : buoyancy_y[j];
    bool pml_y = y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1);
    bool pml_x = x < pml_x0 || x >= pml_x1;
    DW_DTYPE dsigmayydy = DIFFYH1(SIGMAYY);
    DW_DTYPE dsigmaxydx = DIFFX1(SIGMAXY);
    if (pml_y) {
      m_sigmayyy[i] = ayh[y] * m_sigmayyy[i] + byh[y] * dsigmayydy;
      dsigmayydy += m_sigmayyy[i];
    }
    if (pml_x) {
      m_sigmaxyx[i] = ax[x] * m_sigmaxyx[i] + bx[x] * dsigmaxydx;
      dsigmaxydx += m_sigmaxyx[i];
    }
    vy[i] += buoyancy_y_shot * dt * (dsigmayydy + dsigmaxydx);
    if (buoyancy_requires_grad) {
      dvydbuoyancy[i] = dt * (dsigmayydy + dsigmaxydx);
    }
  }
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    DW_DTYPE const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x[i] : buoyancy_x[j];
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1);
    DW_DTYPE dsigmaxydy = DIFFY1(SIGMAXY);
    DW_DTYPE dsigmaxxdx = DIFFXH1(SIGMAXX);
    if (pml_y) {
      m_sigmaxyy[i] = ay[y] * m_sigmaxyy[i] + by[y] * dsigmaxydy;
      dsigmaxydy += m_sigmaxyy[i];
    }
    if (pml_x) {
      m_sigmaxxx[i] = axh[x] * m_sigmaxxx[i] + bxh[x] * dsigmaxxdx;
      dsigmaxxdx += m_sigmaxxx[i];
    }
    vx[i] += buoyancy_x_shot * dt * (dsigmaxxdx + dsigmaxydy);
    if (buoyancy_requires_grad) {
      dvxdbuoyancy[i] = dt * (dsigmaxxdx + dsigmaxydy);
    }
  }
}

__global__ void forward_kernel_sigma(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const mu_yx, DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigmayy,
    DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const sigmaxx,
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy, DW_DTYPE *__restrict const m_vxx,
    DW_DTYPE *__restrict const dvydy_store,
    DW_DTYPE *__restrict const dvxdx_store,
    DW_DTYPE *__restrict const dvydxdvxdy_store,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const lamb_requires_grad, bool const mu_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    DW_DTYPE const lamb_shot = lamb_batched ? lamb[i] : lamb[j];
    DW_DTYPE const mu_shot = mu_batched ? mu[i] : mu[j];
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    DW_DTYPE dvydy = DIFFY1(VY);
    DW_DTYPE dvxdx = DIFFX1(VX);
    if (pml_y) {
      m_vyy[i] = ay[y] * m_vyy[i] + by[y] * dvydy;
      dvydy += m_vyy[i];
    }
    if (pml_x) {
      m_vxx[i] = ax[x] * m_vxx[i] + bx[x] * dvxdx;
      dvxdx += m_vxx[i];
    }
    sigmayy[i] += dt * ((lamb_shot + 2 * mu_shot) * dvydy + lamb_shot * dvxdx);
    sigmaxx[i] += dt * ((lamb_shot + 2 * mu_shot) * dvxdx + lamb_shot * dvydy);
    if (lamb_requires_grad || mu_requires_grad) {
      dvydy_store[i] = dt * dvydy;
      dvxdx_store[i] = dt * dvxdx;
    }
  }
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    DW_DTYPE const mu_yx_shot = mu_batched ? mu_yx[i] : mu_yx[j];
    bool pml_y = y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1);
    bool pml_x = x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1);
    DW_DTYPE dvydx = DIFFXH1(VY);
    DW_DTYPE dvxdy = DIFFYH1(VX);
    if (pml_y) {
      m_vxy[i] = ayh[y] * m_vxy[i] + byh[y] * dvxdy;
      dvxdy += m_vxy[i];
    }
    if (pml_x) {
      m_vyx[i] = axh[x] * m_vyx[i] + bxh[x] * dvydx;
      dvydx += m_vyx[i];
    }
    sigmaxy[i] += dt * mu_yx_shot * (dvydx + dvxdy);
    if (mu_requires_grad) {
      dvydxdvxdy_store[i] = dt * (dvydx + dvxdy);
    }
  }
}

__global__ void backward_kernel_sigma(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const mu_yx,
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x, DW_DTYPE *__restrict const vy,
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
    DW_DTYPE *__restrict const m_sigmaxxxn,
    DW_DTYPE const *__restrict const dvydbuoyancy,
    DW_DTYPE const *__restrict const dvxdbuoyancy,
    DW_DTYPE *__restrict const grad_buoyancy_y,
    DW_DTYPE *__restrict const grad_buoyancy_x,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  DW_DTYPE const *__restrict const lamb_shot =
      lamb_batched ? lamb + shot_idx * nynx : lamb;
  DW_DTYPE const *__restrict const mu_shot =
      mu_batched ? mu + shot_idx * nynx : mu;
  DW_DTYPE const *__restrict const mu_yx_shot =
      mu_batched ? mu_yx + shot_idx * nynx : mu_yx;
  DW_DTYPE const *__restrict const buoyancy_y_shot =
      buoyancy_batched ? buoyancy_y + shot_idx * nynx : buoyancy_y;
  DW_DTYPE const *__restrict const buoyancy_x_shot =
      buoyancy_batched ? buoyancy_x + shot_idx * nynx : buoyancy_x;
  if (y < ny - FD_PAD && x < nx - FD_PAD + 1) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    bool pml_y = y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1);
    bool pml_x = x < pml_x0 || x >= pml_x1;
    vy[i] += ((pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y)) +
              (pml_x ? -DIFFX1(VY_X_PML) : -DIFFX1(VY_X)));
    if (pml_y) {
      m_sigmayyyn[i] =
          buoyancy_y_shot[j] * dt * ayh[y] * vy[i] + ayh[y] * m_sigmayyy[i];
    }
    if (pml_x) {
      m_sigmaxyxn[i] =
          buoyancy_y_shot[j] * dt * ax[x] * vy[i] + ax[x] * m_sigmaxyx[i];
    }
    if (buoyancy_requires_grad) {
      grad_buoyancy_y[i] += vy[i] * dvydbuoyancy[i] * (DW_DTYPE)step_ratio;
    }
  }
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1);
    vx[i] += ((pml_y ? -DIFFY1(VX_Y_PML) : -DIFFY1(VX_Y)) +
              (pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X)));
    if (pml_y) {
      m_sigmaxyyn[i] =
          buoyancy_x_shot[j] * dt * ay[y] * vx[i] + ay[y] * m_sigmaxyy[i];
    }
    if (pml_x) {
      m_sigmaxxxn[i] =
          buoyancy_x_shot[j] * dt * axh[x] * vx[i] + axh[x] * m_sigmaxxx[i];
    }
    if (buoyancy_requires_grad) {
      grad_buoyancy_x[i] += vx[i] * dvxdbuoyancy[i] * (DW_DTYPE)step_ratio;
    }
  }
}

__global__ void backward_kernel_v(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const mu_yx,
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x,
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE *__restrict const m_vyy,
    DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
    DW_DTYPE *__restrict const m_vxx,
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE const *__restrict const dvydy_store,
    DW_DTYPE const *__restrict const dvxdx_store,
    DW_DTYPE const *__restrict const dvydxdvxdy_store,
    DW_DTYPE *__restrict const grad_lamb, DW_DTYPE *__restrict const grad_mu,
    DW_DTYPE *__restrict const grad_mu_yx, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const bxh, bool const lamb_requires_grad,
    bool const mu_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  DW_DTYPE const *__restrict const lamb_shot =
      lamb_batched ? lamb + shot_idx * nynx : lamb;
  DW_DTYPE const *__restrict const mu_shot =
      mu_batched ? mu + shot_idx * nynx : mu;
  DW_DTYPE const *__restrict const mu_yx_shot =
      mu_batched ? mu_yx + shot_idx * nynx : mu_yx;
  DW_DTYPE const *__restrict const buoyancy_y_shot =
      buoyancy_batched ? buoyancy_y + shot_idx * nynx : buoyancy_y;
  DW_DTYPE const *__restrict const buoyancy_x_shot =
      buoyancy_batched ? buoyancy_x + shot_idx * nynx : buoyancy_x;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    if (lamb_requires_grad) {
      grad_lamb[i] += (sigmayy[i] + sigmaxx[i]) *
                      (dvydy_store[i] + dvxdx_store[i]) * (DW_DTYPE)step_ratio;
    }
    if (mu_requires_grad) {
      grad_mu[i] +=
          2 * (sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) *
          (DW_DTYPE)step_ratio;
    }
    if (pml_y) {
      m_vyy[i] = (lamb_shot[j] + 2 * mu_shot[j]) * dt * ay[y] * sigmayy[i] +
                 lamb_shot[j] * dt * ay[y] * sigmaxx[i] + ay[y] * m_vyy[i];
    }
    if (pml_x) {
      m_vxx[i] = (lamb_shot[j] + 2 * mu_shot[j]) * dt * ax[x] * sigmaxx[i] +
                 lamb_shot[j] * dt * ax[x] * sigmayy[i] + ax[x] * m_vxx[i];
    }
    sigmayy[i] += (pml_y ? -DIFFY1(SIGMAYY_Y_PML) : -DIFFY1(SIGMAYY_Y));
    sigmaxx[i] += (pml_x ? -DIFFX1(SIGMAXX_X_PML) : -DIFFX1(SIGMAXX_X));
  }
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    bool pml_y = y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1);
    bool pml_x = x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1);
    if (mu_requires_grad) {
      grad_mu_yx[i] += sigmaxy[i] * dvydxdvxdy_store[i] * (DW_DTYPE)step_ratio;
    }
    if (pml_y) {
      m_vxy[i] = mu_yx_shot[j] * dt * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i];
    }
    if (pml_x) {
      m_vyx[i] = mu_yx_shot[j] * dt * axh[x] * sigmaxy[i] + axh[x] * m_vyx[i];
    }
    sigmaxy[i] += ((pml_y ? -DIFFYH1(SIGMAXY_Y_PML) : -DIFFYH1(SIGMAXY_Y)) +
                   (pml_x ? -DIFFXH1(SIGMAXY_X_PML) : -DIFFXH1(SIGMAXY_X)));
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

void set_config(DW_DTYPE const dt_h, DW_DTYPE const rdy_h, DW_DTYPE const rdx_h,
                int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
                int64_t const n_sources_y_per_shot_h,
                int64_t const n_sources_x_per_shot_h,
                int64_t const n_receivers_y_per_shot_h,
                int64_t const n_receivers_x_per_shot_h,
                int64_t const n_receivers_p_per_shot_h,
                int64_t const step_ratio_h, int64_t const pml_y0_h,
                int64_t const pml_y1_h, int64_t const pml_x0_h,
                int64_t const pml_x1_h, bool const lamb_batched_h,
                bool const mu_batched_h, bool const buoyancy_batched_h) {
  int64_t const nynx_h = ny_h * nx_h;

  gpuErrchk(cudaMemcpyToSymbol(dt, &dt_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
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
  gpuErrchk(cudaMemcpyToSymbol(lamb_batched, &lamb_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(mu_batched, &mu_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(buoyancy_batched, &buoyancy_batched_h, sizeof(bool)));
}

void backward_batch(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
    DW_DTYPE const *__restrict const mu_yx,
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x,
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
    DW_DTYPE *__restrict const grad_mu_yx,
    DW_DTYPE *__restrict const grad_buoyancy_y,
    DW_DTYPE *__restrict const grad_buoyancy_x,
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
  backward_kernel_sigma<<<dimGrid, dimBlock>>>(
      lamb, mu, mu_yx, buoyancy_y, buoyancy_x, vy, vx, sigmayy, sigmaxy,
      sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
      m_sigmaxxx, m_sigmayyyn, m_sigmaxyyn, m_sigmaxyxn, m_sigmaxxxn,
      dvydbuoyancy, dvxdbuoyancy, grad_buoyancy_y, grad_buoyancy_x, ay, ayh, ax,
      axh, by, byh, bx, bxh, buoyancy_requires_grad);
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
  backward_kernel_v<<<dimGrid, dimBlock>>>(
      lamb, mu, mu_yx, buoyancy_y, buoyancy_x, vy, vx, sigmayy, sigmaxy,
      sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
      m_sigmaxxx, dvydy_store, dvxdx_store, dvydxdvxdy_store, grad_lamb,
      grad_mu, grad_mu_yx, ay, ayh, ax, axh, by, byh, bx, bxh,
      lamb_requires_grad, mu_requires_grad);
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
            DW_DTYPE const *__restrict const mu_yx,
            DW_DTYPE const *__restrict const buoyancy_y,
            DW_DTYPE const *__restrict const buoyancy_x,
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
            int64_t const *__restrict const receivers_p_i, DW_DTYPE const rdy,
            DW_DTYPE const rdx, DW_DTYPE const dt_h, int64_t const nt,
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
  set_config(dt_h, rdy, rdx, n_shots_h, ny_h, nx_h, n_sources_y_per_shot_h,
             n_sources_x_per_shot_h, n_receivers_y_per_shot_h,
             n_receivers_x_per_shot_h, n_receivers_p_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, lamb_batched_h,
             mu_batched_h, buoyancy_batched_h);

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
        buoyancy_y, buoyancy_x, vy, vx, sigmayy, sigmaxy, sigmaxx, m_sigmayyy,
        m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
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
        lamb, mu, mu_yx, vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy,
        m_vxx, dvydy_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h,
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
            DW_DTYPE const *__restrict const mu_yx,
            DW_DTYPE const *__restrict const buoyancy_y,
            DW_DTYPE const *__restrict const buoyancy_x,
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
            DW_DTYPE *__restrict const grad_mu_yx,
            DW_DTYPE *__restrict const grad_mu_yx_shot,
            DW_DTYPE *__restrict const grad_buoyancy_y,
            DW_DTYPE *__restrict const grad_buoyancy_y_shot,
            DW_DTYPE *__restrict const grad_buoyancy_x,
            DW_DTYPE *__restrict const grad_buoyancy_x_shot,
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
            int64_t const *__restrict const receivers_p_i, DW_DTYPE const rdy,
            DW_DTYPE const rdx, DW_DTYPE const dt_h, int64_t const nt,
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
  set_config(dt_h, rdy, rdx, n_shots_h, ny_h, nx_h, n_sources_y_per_shot_h,
             n_sources_x_per_shot_h, n_receivers_y_per_shot_h,
             n_receivers_x_per_shot_h, n_receivers_p_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, lamb_batched_h,
             mu_batched_h, buoyancy_batched_h);
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
          lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
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
          grad_mu_shot, grad_mu_yx_shot, grad_buoyancy_y_shot,
          grad_buoyancy_x_shot, ay, ayh, ax, axh, by, byh, bx, bxh, sources_y_i,
          sources_x_i, receivers_y_i, receivers_x_i, receivers_p_i, n_shots_h,
          ny_h, nx_h, n_sources_y_per_shot_h, n_sources_x_per_shot_h,
          n_receivers_y_per_shot_h, n_receivers_x_per_shot_h,
          n_receivers_p_per_shot_h,
          lamb_requires_grad && ((t % step_ratio_h) == 0),
          mu_requires_grad && ((t % step_ratio_h) == 0),
          buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    } else {
      backward_batch(
          lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
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
          grad_mu_shot, grad_mu_yx_shot, grad_buoyancy_y_shot,
          grad_buoyancy_x_shot, ay, ayh, ax, axh, by, byh, bx, bxh, sources_y_i,
          sources_x_i, receivers_y_i, receivers_x_i, receivers_p_i, n_shots_h,
          ny_h, nx_h, n_sources_y_per_shot_h, n_sources_x_per_shot_h,
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
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_mu_yx,
                                                        grad_mu_yx_shot);
    CHECK_KERNEL_ERROR
  }
  if (buoyancy_requires_grad && !buoyancy_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_buoyancy_y,
                                                        grad_buoyancy_y_shot);
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_buoyancy_x,
                                                        grad_buoyancy_x_shot);
    CHECK_KERNEL_ERROR
  }
}

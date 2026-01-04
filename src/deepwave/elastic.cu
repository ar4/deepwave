/*
 * Elastic wave equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the elastic wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 1-3.
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
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  elastic_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz)*ny * nx + (dy)*nx + (dx))
#define ND_INDEX_J(j, dz, dy, dx) (j + (dz)*ny * nx + (dy)*nx + (dx))
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy)*nx + (dx))
#define ND_INDEX_J(j, dz, dy, dx) (j + (dy)*nx + (dx))
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#define ND_INDEX_J(j, dz, dy, dx) (j + (dx))
#define DIM_ARGS nx_h
#endif

#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define SIGMAZZ(dz, dy, dx) sigmazz[ND_INDEX(i, dz, dy, dx)]
#elif DW_NDIM == 2
#define SIGMAZZ(dz, dy, dx) 0
#endif
#if DW_NDIM >= 2
#define SIGMAYY(dz, dy, dx) sigmayy[ND_INDEX(i, dz, dy, dx)]
#endif
#define SIGMAXX(dz, dy, dx) sigmaxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define SIGMAXY(dz, dy, dx) sigmaxy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define SIGMAXZ(dz, dy, dx) sigmaxz[ND_INDEX(i, dz, dy, dx)]
#define SIGMAYZ(dz, dy, dx) sigmayz[ND_INDEX(i, dz, dy, dx)]
#endif
#define LAMB(dz, dy, dx) lamb_shot[ND_INDEX_J(j, dz, dy, dx)]
#define MU(dz, dy, dx) mu_shot[ND_INDEX_J(j, dz, dy, dx)]
#if DW_NDIM >= 2
#define MU_YX(dz, dy, dx) mu_yx_shot[ND_INDEX_J(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define MU_ZX(dz, dy, dx) mu_zx_shot[ND_INDEX_J(j, dz, dy, dx)]
#define MU_ZY(dz, dy, dx) mu_zy_shot[ND_INDEX_J(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX_J(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX_J(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX_J(j, dz, dy, dx)]

// PML memory variables for velocity equations
#if DW_NDIM >= 2
#define M_SIGMAXYY(dz, dy, dx) m_sigmaxyy[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_SIGMAXXX(dz, dy, dx) m_sigmaxxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define M_SIGMAXZZ(dz, dy, dx) m_sigmaxzz[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 2
#define M_SIGMAXYX(dz, dy, dx) m_sigmaxyx[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYYY(dz, dy, dx) m_sigmayyy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_SIGMAYZZ(dz, dy, dx) m_sigmayzz[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 3
#define M_SIGMAXZX(dz, dy, dx) m_sigmaxzx[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYZY(dz, dy, dx) m_sigmayzy[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAZZZ(dz, dy, dx) m_sigmazzz[ND_INDEX(i, dz, dy, dx)]
#endif

// PML memory variables for stress equations
#if DW_NDIM >= 2
#define M_VXY(dz, dy, dx) m_vxy[ND_INDEX(i, dz, dy, dx)]
#define M_VYX(dz, dy, dx) m_vyx[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_VXX(dz, dy, dx) m_vxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define M_VYY(dz, dy, dx) m_vyy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_VZZ(dz, dy, dx) m_vzz[ND_INDEX(i, dz, dy, dx)]
#define M_VXZ(dz, dy, dx) m_vxz[ND_INDEX(i, dz, dy, dx)]
#define M_VZX(dz, dy, dx) m_vzx[ND_INDEX(i, dz, dy, dx)]
#define M_VYZ(dz, dy, dx) m_vyz[ND_INDEX(i, dz, dy, dx)]
#define M_VZY(dz, dy, dx) m_vzy[ND_INDEX(i, dz, dy, dx)]
#endif

// Access terms used in the backward pass
#define LAMB_2MU(dz, dy, dx) (LAMB(dz, dy, dx) + 2 * MU(dz, dy, dx))

#if DW_NDIM >= 3
#define VZ_Z(dz, dy, dx)                          \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         LAMB_2MU(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#define VZ_Y(dz, dy, dx) (dt * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#define VZ_X(dz, dy, dx) (dt * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#define VZ_Z_PML(dz, dy, dx)                          \
  (dt * (1 + bz[z + dz]) *                            \
       (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +      \
        LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +      \
        LAMB_2MU(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) + \
   bz[z + dz] * M_VZZ(dz, dy, dx))
#define VZ_Y_PML(dz, dy, dx)                                          \
  (dt * (1 + byh[y + dy]) * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx) + \
   byh[y + dy] * M_VZY(dz, dy, dx))
#define VZ_X_PML(dz, dy, dx)                                          \
  (dt * (1 + bxh[x + dx]) * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx) + \
   bxh[x + dx] * M_VZX(dz, dy, dx))
#endif

#if DW_NDIM >= 2
#if DW_NDIM >= 3
#define VY_Z(dz, dy, dx) (dt * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y(dz, dy, dx)                          \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#else
#define VY_Y(dz, dy, dx)                              \
  (dt * (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +     \
         LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#define VY_X(dz, dy, dx) (dt * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#if DW_NDIM >= 3
#define VY_Z_PML(dz, dy, dx)                                          \
  (dt * (1 + bzh[z + dz]) * MU_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx) + \
   bzh[z + dz] * M_VYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y_PML(dz, dy, dx)                          \
  (dt * (1 + by[y + dy]) *                            \
       (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +      \
        LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx)) + \
   by[y + dy] * M_VYY(dz, dy, dx))
#else
#define VY_Y_PML(dz, dy, dx)                         \
  (dt * (1 + by[y + dy]) *                           \
       (LAMB(dz, dy, dx) * SIGMAXX(dz, dy, dx) +     \
        LAMB_2MU(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
        LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) +    \
   by[y + dy] * M_VYY(dz, dy, dx))
#endif
#define VY_X_PML(dz, dy, dx)                                          \
  (dt * (1 + bxh[x + dx]) * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx) + \
   bxh[x + dx] * M_VYX(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define VX_Z(dz, dy, dx) (dt * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y(dz, dy, dx) (dt * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X(dz, dy, dx) (dt * LAMB_2MU(0, 0, dx) * SIGMAXX(0, 0, dx))
#else
#define VX_X(dz, dy, dx)                              \
  (dt * (LAMB_2MU(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +     \
         LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#if DW_NDIM >= 3
#define VX_Z_PML(dz, dy, dx)                                          \
  (dt * (1 + bzh[z + dz]) * MU_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx) + \
   bzh[z + dz] * M_VXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y_PML(dz, dy, dx)                                          \
  (dt * (1 + byh[y + dy]) * MU_YX(dz, dy, dx) * SIGMAXY(dz, dy, dx) + \
   byh[y + dy] * M_VXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X_PML(dz, dy, dx)                                          \
  (dt * (1 + bx[x + dx]) * (LAMB_2MU(0, 0, dx) * SIGMAXX(0, 0, dx)) + \
   bx[x + dx] * M_VXX(0, 0, dx))
#else
#define VX_X_PML(dz, dy, dx)                         \
  (dt * (1 + bx[x + dx]) *                           \
       (LAMB_2MU(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
        LAMB(dz, dy, dx) * SIGMAYY(dz, dy, dx) +     \
        LAMB(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) +    \
   bx[x + dx] * M_VXX(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define SIGMAZZ_Z_PML(dz, dy, dx)                                     \
  ((1 + bzh[z + dz]) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   bzh[z + dz] * M_SIGMAZZZ(dz, dy, dx))
#define SIGMAZZ_Z(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define SIGMAYY_Y_PML(dz, dy, dx)                                     \
  ((1 + byh[y + dy]) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   byh[y + dy] * M_SIGMAYYY(dz, dy, dx))
#define SIGMAYY_Y(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define SIGMAXX_X_PML(dz, dy, dx)                                     \
  ((1 + bxh[x + dx]) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   bxh[x + dx] * M_SIGMAXXX(dz, dy, dx))
#define SIGMAXX_X(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

#if DW_NDIM >= 2
#define SIGMAXY_Y_PML(dz, dy, dx)                                    \
  ((1 + by[y + dy]) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   by[y + dy] * M_SIGMAXYY(dz, dy, dx))
#define SIGMAXY_Y(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define SIGMAXY_X_PML(dz, dy, dx)                                    \
  ((1 + bx[x + dx]) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   bx[x + dx] * M_SIGMAXYX(dz, dy, dx))
#define SIGMAXY_X(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define SIGMAXZ_Z_PML(dz, dy, dx)                                    \
  ((1 + bz[z + dz]) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   bz[z + dz] * M_SIGMAXZZ(dz, dy, dx))
#define SIGMAXZ_Z(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define SIGMAXZ_X_PML(dz, dy, dx)                                    \
  ((1 + bx[x + dx]) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   bx[x + dx] * M_SIGMAXZX(dz, dy, dx))
#define SIGMAXZ_X(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))

#define SIGMAYZ_Z_PML(dz, dy, dx)                                    \
  ((1 + bz[z + dz]) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   bz[z + dz] * M_SIGMAYZZ(dz, dy, dx))
#define SIGMAYZ_Z(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define SIGMAYZ_Y_PML(dz, dy, dx)                                    \
  ((1 + by[y + dy]) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   by[y + dy] * M_SIGMAYZY(dz, dy, dx))
#define SIGMAYZ_Y(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif

#define MAX(a, b) (a > b ? a : b)

namespace {
__constant__ DW_DTYPE dt;
#if DW_NDIM >= 3
__constant__ DW_DTYPE rdz;
#endif
#if DW_NDIM >= 2
__constant__ DW_DTYPE rdy;
#endif
__constant__ DW_DTYPE rdx;
__constant__ int64_t n_shots;
#if DW_NDIM >= 3
__constant__ int64_t nz;
#endif
#if DW_NDIM >= 2
__constant__ int64_t ny;
#endif
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
#if DW_NDIM >= 3
__constant__ int64_t n_sources_z_per_shot;
#endif
#if DW_NDIM >= 2
__constant__ int64_t n_sources_y_per_shot;
#endif
__constant__ int64_t n_sources_x_per_shot;
__constant__ int64_t n_sources_p_per_shot;
#if DW_NDIM >= 3
__constant__ int64_t n_receivers_z_per_shot;
#endif
#if DW_NDIM >= 2
__constant__ int64_t n_receivers_y_per_shot;
#endif
__constant__ int64_t n_receivers_x_per_shot;
__constant__ int64_t n_receivers_p_per_shot;
__constant__ int64_t step_ratio;
#if DW_NDIM >= 3
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
#endif
#if DW_NDIM >= 2
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
#endif
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool lamb_batched;
__constant__ bool mu_batched;
__constant__ bool buoyancy_batched;

#if DW_NDIM >= 3
__launch_bounds__(32) __global__
    void add_sources_z(DW_DTYPE *__restrict const wf,
                       DW_DTYPE const *__restrict const f,
                       int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_z_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_z_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}
#endif

#if DW_NDIM >= 2
__launch_bounds__(32) __global__
    void add_sources_y(DW_DTYPE *__restrict const wf,
                       DW_DTYPE const *__restrict const f,
                       int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_y_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_y_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}
#endif

__launch_bounds__(32) __global__
    void add_sources_x(DW_DTYPE *__restrict const wf,
                       DW_DTYPE const *__restrict const f,
                       int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_x_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_x_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__ void add_sources_p(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy,
#endif
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_p_per_shot + source_idx;
    if (0 <= sources_i[k]) {
#if DW_NDIM >= 3
      sigmazz[shot_idx * shot_numel + sources_i[k]] += f[k];
#endif
#if DW_NDIM >= 2
      sigmayy[shot_idx * shot_numel + sources_i[k]] += f[k];
#endif
      sigmaxx[shot_idx * shot_numel + sources_i[k]] += f[k];
    }
  }
}

#if DW_NDIM >= 3
__launch_bounds__(32) __global__
    void add_adjoint_sources_z(DW_DTYPE *__restrict const wf,
                               DW_DTYPE const *__restrict const f,
                               int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_z_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_z_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}
#endif

#if DW_NDIM >= 2
__launch_bounds__(32) __global__
    void add_adjoint_sources_y(DW_DTYPE *__restrict const wf,
                               DW_DTYPE const *__restrict const f,
                               int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_y_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_y_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}
#endif

__launch_bounds__(32) __global__
    void add_adjoint_sources_x(DW_DTYPE *__restrict const wf,
                               DW_DTYPE const *__restrict const f,
                               int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_x_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_x_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__ void add_adjoint_pressure_sources(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy,
#endif
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_p_per_shot + source_idx;
    if (0 <= sources_i[k]) {
#if DW_NDIM >= 3
      sigmazz[shot_idx * shot_numel + sources_i[k]] += f[k];
#endif
#if DW_NDIM >= 2
      sigmayy[shot_idx * shot_numel + sources_i[k]] += f[k];
#endif
      sigmaxx[shot_idx * shot_numel + sources_i[k]] += f[k];
    }
  }
}

#if DW_NDIM >= 3
__launch_bounds__(32) __global__
    void record_receivers_z(DW_DTYPE *__restrict const r,
                            DW_DTYPE const *__restrict const wf,
                            int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_z_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_z_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}
#endif

#if DW_NDIM >= 2
__launch_bounds__(32) __global__
    void record_receivers_y(DW_DTYPE *__restrict const r,
                            DW_DTYPE const *__restrict const wf,
                            int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_y_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_y_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}
#endif

__launch_bounds__(32) __global__
    void record_receivers_x(DW_DTYPE *__restrict const r,
                            DW_DTYPE const *__restrict const wf,
                            int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_x_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_x_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__
    void record_pressure_receivers(DW_DTYPE *__restrict const r,
#if DW_NDIM >= 3
                                   DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
                                   DW_DTYPE const *__restrict const sigmayy,
#endif
                                   DW_DTYPE const *__restrict const sigmaxx,
                                   int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_p_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
#if DW_NDIM == 3
      r[k] = (sigmazz[shot_idx * shot_numel + receivers_i[k]] +
              sigmayy[shot_idx * shot_numel + receivers_i[k]] +
              sigmaxx[shot_idx * shot_numel + receivers_i[k]]);
#elif DW_NDIM == 2
      r[k] = (sigmayy[shot_idx * shot_numel + receivers_i[k]] +
              sigmaxx[shot_idx * shot_numel + receivers_i[k]]);
#else
      r[k] = sigmaxx[shot_idx * shot_numel + receivers_i[k]];
#endif
  }
}

#if DW_NDIM >= 3
__launch_bounds__(32) __global__
    void record_adjoint_receivers_z(DW_DTYPE *__restrict const r,
                                    DW_DTYPE const *__restrict const wf,
                                    int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_z_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_z_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}
#endif

#if DW_NDIM >= 2
__launch_bounds__(32) __global__
    void record_adjoint_receivers_y(DW_DTYPE *__restrict const r,
                                    DW_DTYPE const *__restrict const wf,
                                    int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_y_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_y_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}
#endif

__launch_bounds__(32) __global__
    void record_adjoint_receivers_x(DW_DTYPE *__restrict const r,
                                    DW_DTYPE const *__restrict const wf,
                                    int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_x_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_x_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__ void record_adjoint_pressure_receivers(
    DW_DTYPE *__restrict const r,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
    int64_t const *__restrict const receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_p_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
#if DW_NDIM == 3
      r[k] = (sigmazz[shot_idx * shot_numel + receivers_i[k]] +
              sigmayy[shot_idx * shot_numel + receivers_i[k]] +
              sigmaxx[shot_idx * shot_numel + receivers_i[k]]);
#elif DW_NDIM == 2
      r[k] = (sigmayy[shot_idx * shot_numel + receivers_i[k]] +
              sigmaxx[shot_idx * shot_numel + receivers_i[k]]);
#else
      r[k] = sigmaxx[shot_idx * shot_numel + receivers_i[k]];
#endif
  }
}

__launch_bounds__(128) __global__
    void combine_grad(DW_DTYPE *__restrict const grad,
                      DW_DTYPE const *__restrict const grad_shot) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z < nz && y < ny && x < nx) {
    int64_t const i = z * ny * nx + y * nx + x;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < ny && x < nx) {
    int64_t const i = y * nx + x;
#else
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < nx) {
    int64_t const i = x;
#endif
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
}

__launch_bounds__(128) __global__ void forward_kernel_v(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const dvydbuoyancy,
#endif
    DW_DTYPE *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif

#if DW_NDIM >= 3
      int64_t const pml_z0h = pml_z0;
      int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
#endif
#if DW_NDIM >= 2
      int64_t const pml_y0h = pml_y0;
      int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
#endif
      int64_t const pml_x0h = pml_x0;
      int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

      int64_t const i = shot_idx * shot_numel + j;
#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const buoyancy_z_shot =
            buoyancy_batched ? buoyancy_z[i] : buoyancy_z[j];
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        bool const pml_y = y < pml_y0 || y >= pml_y1;
        bool const pml_x = x < pml_x0 || x >= pml_x1;

        DW_DTYPE dsigmazzdz = DIFFZH1(SIGMAZZ);
        DW_DTYPE dsigmayzdy = DIFFY1(SIGMAYZ);
        DW_DTYPE dsigmaxzdx = DIFFX1(SIGMAXZ);

        if (pml_z) {
          m_sigmazzz[i] = azh[z] * m_sigmazzz[i] + bzh[z] * dsigmazzdz;
          dsigmazzdz += m_sigmazzz[i];
        }
        if (pml_y) {
          m_sigmayzy[i] = ay[y] * m_sigmayzy[i] + by[y] * dsigmayzdy;
          dsigmayzdy += m_sigmayzy[i];
        }
        if (pml_x) {
          m_sigmaxzx[i] = ax[x] * m_sigmaxzx[i] + bx[x] * dsigmaxzdx;
          dsigmaxzdx += m_sigmaxzx[i];
        }

        DW_DTYPE w_sum = dsigmazzdz + dsigmayzdy + dsigmaxzdx;
        vz[i] += buoyancy_z_shot * dt * w_sum;

        if (buoyancy_requires_grad) {
          dvzdbuoyancy[i] = dt * w_sum;
        }
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const buoyancy_y_shot =
            buoyancy_batched ? buoyancy_y[i] : buoyancy_y[j];
#if DW_NDIM == 3
        bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        bool const pml_x = x < pml_x0 || x >= pml_x1;

#if DW_NDIM >= 3
        DW_DTYPE dsigmayzdz = DIFFZ1(SIGMAYZ);
#endif
        DW_DTYPE dsigmayydy = DIFFYH1(SIGMAYY);
        DW_DTYPE dsigmaxydx = DIFFX1(SIGMAXY);

#if DW_NDIM >= 3
        if (pml_z) {
          m_sigmayzz[i] = az[z] * m_sigmayzz[i] + bz[z] * dsigmayzdz;
          dsigmayzdz += m_sigmayzz[i];
        }
#endif
        if (pml_y) {
          m_sigmayyy[i] = ayh[y] * m_sigmayyy[i] + byh[y] * dsigmayydy;
          dsigmayydy += m_sigmayyy[i];
        }
        if (pml_x) {
          m_sigmaxyx[i] = ax[x] * m_sigmaxyx[i] + bx[x] * dsigmaxydx;
          dsigmaxydx += m_sigmaxyx[i];
        }

        DW_DTYPE w_sum =
#if DW_NDIM >= 3
            dsigmayzdz +
#endif
            dsigmayydy + dsigmaxydx;
        vy[i] += buoyancy_y_shot * dt * w_sum;
        if (buoyancy_requires_grad) {
          dvydbuoyancy[i] = dt * w_sum;
        }
      }
#endif
      if (x < nx - FD_PAD) {
        DW_DTYPE const buoyancy_x_shot =
            buoyancy_batched ? buoyancy_x[i] : buoyancy_x[j];
#if DW_NDIM >= 3
        bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
        bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
        bool const pml_x = x < pml_x0h || x >= pml_x1h;

#if DW_NDIM >= 3
        DW_DTYPE dsigmaxzdz = DIFFZ1(SIGMAXZ);
#endif
#if DW_NDIM >= 2
        DW_DTYPE dsigmaxydy = DIFFY1(SIGMAXY);
#endif
        DW_DTYPE dsigmaxxdx = DIFFXH1(SIGMAXX);

#if DW_NDIM >= 3
        if (pml_z) {
          m_sigmaxzz[i] = az[z] * m_sigmaxzz[i] + bz[z] * dsigmaxzdz;
          dsigmaxzdz += m_sigmaxzz[i];
        }
#endif
#if DW_NDIM >= 2
        if (pml_y) {
          m_sigmaxyy[i] = ay[y] * m_sigmaxyy[i] + by[y] * dsigmaxydy;
          dsigmaxydy += m_sigmaxyy[i];
        }
#endif
        if (pml_x) {
          m_sigmaxxx[i] = axh[x] * m_sigmaxxx[i] + bxh[x] * dsigmaxxdx;
          dsigmaxxdx += m_sigmaxxx[i];
        }

        DW_DTYPE w_sum =
#if DW_NDIM >= 3
            dsigmaxzdz +
#endif
#if DW_NDIM >= 2
            dsigmaxydy +
#endif
            dsigmaxxdx;
        vx[i] += buoyancy_x_shot * dt * w_sum;
        if (buoyancy_requires_grad) {
          dvxdbuoyancy[i] = dt * w_sum;
        }
      }
#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void forward_kernel_sigma(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy,
    DW_DTYPE const *__restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy,
#endif
    DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdz_store,
    DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE *__restrict const dvzdy_plus_dvydz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const dvydy_store,
    DW_DTYPE *__restrict const dvydx_plus_dvxdy_store,
#endif
    DW_DTYPE *__restrict const dvxdx_store,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const lamb_requires_grad, bool const mu_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
#if DW_NDIM == 3
      int64_t const pml_z0h = pml_z0;
      int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
#endif
#if DW_NDIM >= 2
      int64_t const pml_y0h = pml_y0;
      int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
      int64_t const pml_x0h = pml_x0;
      int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);
#endif

      int64_t const i = shot_idx * shot_numel + j;
      DW_DTYPE const lamb_shot_i = lamb_batched ? lamb[i] : lamb[j];
      DW_DTYPE const mu_shot_i = mu_batched ? mu[i] : mu[j];

#if DW_NDIM >= 3
      DW_DTYPE dvzdz = DIFFZ1(VZ);
      if (z < pml_z0 || z >= pml_z1) {
        m_vzz[i] = az[z] * m_vzz[i] + bz[z] * dvzdz;
        dvzdz += m_vzz[i];
      }
#endif
#if DW_NDIM >= 2
      DW_DTYPE dvydy = DIFFY1(VY);
      if (y < pml_y0 || y >= pml_y1) {
        m_vyy[i] = ay[y] * m_vyy[i] + by[y] * dvydy;
        dvydy += m_vyy[i];
      }
#endif
      DW_DTYPE dvxdx = DIFFX1(VX);
      if (x < pml_x0 || x >= pml_x1) {
        m_vxx[i] = ax[x] * m_vxx[i] + bx[x] * dvxdx;
        dvxdx += m_vxx[i];
      }

      DW_DTYPE w_sum = lamb_shot_i * (
#if DW_NDIM >= 3
                                         dvzdz +
#endif
#if DW_NDIM >= 2
                                         dvydy +
#endif
                                         dvxdx);
#if DW_NDIM >= 3
      sigmazz[i] += dt * (w_sum + 2 * mu_shot_i * dvzdz);
#endif
#if DW_NDIM >= 2
      sigmayy[i] += dt * (w_sum + 2 * mu_shot_i * dvydy);
#endif
      sigmaxx[i] += dt * (w_sum + 2 * mu_shot_i * dvxdx);
      if (lamb_requires_grad || mu_requires_grad) {
#if DW_NDIM >= 3
        dvzdz_store[i] = dt * dvzdz;
#endif
#if DW_NDIM >= 2
        dvydy_store[i] = dt * dvydy;
#endif
        dvxdx_store[i] = dt * dvxdx;
      }
#if DW_NDIM >= 2
      if (y < ny - FD_PAD && x < nx - FD_PAD) {
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        DW_DTYPE const mu_yx_shot_i = mu_batched ? mu_yx[i] : mu_yx[j];
        DW_DTYPE dvxdy = DIFFYH1(VX);
        DW_DTYPE dvydx = DIFFXH1(VY);
        if (pml_y) {
          m_vxy[i] = ayh[y] * m_vxy[i] + byh[y] * dvxdy;
          dvxdy += m_vxy[i];
        }
        if (pml_x) {
          m_vyx[i] = axh[x] * m_vyx[i] + bxh[x] * dvydx;
          dvydx += m_vyx[i];
        }
        w_sum = dvxdy + dvydx;
        sigmaxy[i] += dt * mu_yx_shot_i * w_sum;
        if (mu_requires_grad) {
          dvydx_plus_dvxdy_store[i] = dt * w_sum;
        }
      }
#endif
#if DW_NDIM >= 3
      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        DW_DTYPE const mu_zx_shot_i = mu_batched ? mu_zx[i] : mu_zx[j];
        DW_DTYPE dvxdz = DIFFZH1(VX);
        DW_DTYPE dvzdx = DIFFXH1(VZ);
        if (pml_z) {
          m_vxz[i] = azh[z] * m_vxz[i] + bzh[z] * dvxdz;
          dvxdz += m_vxz[i];
        }
        if (pml_x) {
          m_vzx[i] = axh[x] * m_vzx[i] + bxh[x] * dvzdx;
          dvzdx += m_vzx[i];
        }
        w_sum = dvxdz + dvzdx;
        sigmaxz[i] += dt * mu_zx_shot_i * w_sum;
        if (mu_requires_grad) {
          dvzdx_plus_dvxdz_store[i] = dt * w_sum;
        }
      }
      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        DW_DTYPE const mu_zy_shot_i = mu_batched ? mu_zy[i] : mu_zy[j];
        DW_DTYPE dvydz = DIFFZH1(VY);
        DW_DTYPE dvzdy = DIFFYH1(VZ);
        if (pml_z) {
          m_vyz[i] = azh[z] * m_vyz[i] + bzh[z] * dvydz;
          dvydz += m_vyz[i];
        }
        if (pml_y) {
          m_vzy[i] = ayh[y] * m_vzy[i] + byh[y] * dvzdy;
          dvzdy += m_vzy[i];
        }
        w_sum = dvydz + dvzdy;
        sigmayz[i] += dt * mu_zy_shot_i * w_sum;
        if (mu_requires_grad) {
          dvzdy_plus_dvydz_store[i] = dt * w_sum;
        }
      }
#endif

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel_sigma(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy,
    DW_DTYPE const *__restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_vyz,
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_sigmazzzn,
    DW_DTYPE *__restrict const m_sigmayzyn,
    DW_DTYPE *__restrict const m_sigmaxzxn,
    DW_DTYPE *__restrict const m_sigmayzzn,
    DW_DTYPE *__restrict const m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_sigmayyyn,
    DW_DTYPE *__restrict const m_sigmaxyyn,
    DW_DTYPE *__restrict const m_sigmaxyxn,
#endif
    DW_DTYPE *__restrict const m_sigmaxxxn,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dvydbuoyancy,
#endif
    DW_DTYPE const *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_buoyancy_y,
#endif
    DW_DTYPE *__restrict const grad_buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      DW_DTYPE const *__restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const *__restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const *__restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
#endif

#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

#if DW_NDIM >= 3
      int64_t const pml_z0h = pml_z0;
      int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
#endif
#if DW_NDIM >= 2
      int64_t const pml_y0h = pml_y0;
      int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
#endif
      int64_t const pml_x0h = pml_x0;
      int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

      int64_t i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        bool const pml_y = y < pml_y0 || y >= pml_y1;
        bool const pml_x = x < pml_x0 || x >= pml_x1;
        DW_DTYPE w_sum = 0;
        w_sum += pml_z ? -DIFFZH1(VZ_Z_PML) : -DIFFZH1(VZ_Z);
        w_sum += pml_y ? -DIFFY1(VZ_Y_PML) : -DIFFY1(VZ_Y);
        w_sum += pml_x ? -DIFFX1(VZ_X_PML) : -DIFFX1(VZ_X);
        vz[i] += w_sum;
        if (pml_z) {
          m_sigmazzzn[i] = BUOYANCY_Z(0, 0, 0) * dt * azh[z] * vz[i] +
                           azh[z] * m_sigmazzz[i];
        }
        if (pml_y) {
          m_sigmayzyn[i] =
              BUOYANCY_Z(0, 0, 0) * dt * ay[y] * vz[i] + ay[y] * m_sigmayzy[i];
        }
        if (pml_x) {
          m_sigmaxzxn[i] =
              BUOYANCY_Z(0, 0, 0) * dt * ax[x] * vz[i] + ax[x] * m_sigmaxzx[i];
        }
        if (buoyancy_requires_grad) {
          grad_buoyancy_z[i] += vz[i] * dvzdbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        bool const pml_x = x < pml_x0 || x >= pml_x1;
        DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
        w_sum += pml_z ? -DIFFZ1(VY_Z_PML) : -DIFFZ1(VY_Z);
#endif
        w_sum += pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y);
        w_sum += pml_x ? -DIFFX1(VY_X_PML) : -DIFFX1(VY_X);

        vy[i] += w_sum;

#if DW_NDIM >= 3
        if (pml_z) {
          m_sigmayzzn[i] =
              BUOYANCY_Y(0, 0, 0) * dt * az[z] * vy[i] + az[z] * m_sigmayzz[i];
        }
#endif
        if (pml_y) {
          m_sigmayyyn[i] = BUOYANCY_Y(0, 0, 0) * dt * ayh[y] * vy[i] +
                           ayh[y] * m_sigmayyy[i];
        }
        if (pml_x) {
          m_sigmaxyxn[i] =
              BUOYANCY_Y(0, 0, 0) * dt * ax[x] * vy[i] + ax[x] * m_sigmaxyx[i];
        }
        if (buoyancy_requires_grad) {
          grad_buoyancy_y[i] += vy[i] * dvydbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif

      if (x < nx - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
        bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        DW_DTYPE w_sum = 0;
#if DW_NDIM >= 3
        w_sum += pml_z ? -DIFFZ1(VX_Z_PML) : -DIFFZ1(VX_Z);
#endif
#if DW_NDIM >= 2
        w_sum += pml_y ? -DIFFY1(VX_Y_PML) : -DIFFY1(VX_Y);
#endif
        w_sum += pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X);
        vx[i] += w_sum;

#if DW_NDIM >= 3
        if (pml_z) {
          m_sigmaxzzn[i] =
              BUOYANCY_X(0, 0, 0) * dt * az[z] * vx[i] + az[z] * m_sigmaxzz[i];
        }
#endif
#if DW_NDIM >= 2
        if (pml_y) {
          m_sigmaxyyn[i] =
              BUOYANCY_X(0, 0, 0) * dt * ay[y] * vx[i] + ay[y] * m_sigmaxyy[i];
        }
#endif
        if (pml_x) {
          m_sigmaxxxn[i] = BUOYANCY_X(0, 0, 0) * dt * axh[x] * vx[i] +
                           axh[x] * m_sigmaxxx[i];
        }

        if (buoyancy_requires_grad) {
          grad_buoyancy_x[i] += vx[i] * dvxdbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel_v(
    DW_DTYPE const *__restrict const lamb, DW_DTYPE const *__restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const mu_zy,
    DW_DTYPE const *__restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy,
#endif
    DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdz_store,
    DW_DTYPE const *__restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE const *__restrict const dvzdy_plus_dvydz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dvydy_store,
    DW_DTYPE const *__restrict const dvydx_plus_dvxdy_store,
#endif
    DW_DTYPE const *__restrict const dvxdx_store,
    DW_DTYPE *__restrict const grad_lamb, DW_DTYPE *__restrict const grad_mu,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_mu_zy,
    DW_DTYPE *__restrict const grad_mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const lamb_requires_grad, bool const mu_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      DW_DTYPE const *__restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const *__restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const *__restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

#if DW_NDIM >= 3
      int64_t const pml_z0h = pml_z0;
      int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
#endif
#if DW_NDIM >= 2
      int64_t const pml_y0h = pml_y0;
      int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
      int64_t const pml_x0h = pml_x0;
      int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);
#endif

      int64_t const i = shot_idx * shot_numel + j;

      if (lamb_requires_grad) {
#if DW_NDIM == 3
        grad_lamb[i] += (sigmaxx[i] + sigmayy[i] + sigmazz[i]) *
                        (dvxdx_store[i] + dvydy_store[i] + dvzdz_store[i]) *
                        (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
      grad_lamb[i] += (sigmaxx[i] + sigmayy[i]) *
                      (dvxdx_store[i] + dvydy_store[i]) * (DW_DTYPE)step_ratio;
#else
      grad_lamb[i] += sigmaxx[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
#endif
      }
      if (mu_requires_grad) {
#if DW_NDIM == 3
        grad_mu[i] +=
            2 *
            (sigmaxx[i] * dvxdx_store[i] + sigmayy[i] * dvydy_store[i] +
             sigmazz[i] * dvzdz_store[i]) *
            (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
      grad_mu[i] +=
          2 * (sigmaxx[i] * dvxdx_store[i] + sigmayy[i] * dvydy_store[i]) *
          (DW_DTYPE)step_ratio;
#else
      grad_mu[i] += 2 * sigmaxx[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
#endif
      }

      {
#if DW_NDIM >= 3
        bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
        bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
        bool const pml_x = x < pml_x0 || x >= pml_x1;

#if DW_NDIM >= 3
        if (pml_z) {
          m_vzz[i] = (LAMB_2MU(0, 0, 0)) * dt * az[z] * sigmazz[i] +
                     LAMB(0, 0, 0) * dt * az[z] * sigmaxx[i] +
                     LAMB(0, 0, 0) * dt * az[z] * sigmayy[i] + az[z] * m_vzz[i];
        }
#endif
#if DW_NDIM >= 2
        if (pml_y) {
          m_vyy[i] = (LAMB_2MU(0, 0, 0)) * dt * ay[y] * sigmayy[i] +
                     LAMB(0, 0, 0) * dt * ay[y] * sigmaxx[i] +
#if DW_NDIM == 3
                     LAMB(0, 0, 0) * dt * ay[y] * sigmazz[i] +
#endif
                     ay[y] * m_vyy[i];
        }
#endif
        if (pml_x) {
          m_vxx[i] = (LAMB_2MU(0, 0, 0)) * dt * ax[x] * sigmaxx[i] +
#if DW_NDIM >= 2
                     LAMB(0, 0, 0) * dt * ax[x] * sigmayy[i] +
#endif
#if DW_NDIM == 3
                     LAMB(0, 0, 0) * dt * ax[x] * sigmazz[i] +
#endif
                     ax[x] * m_vxx[i];
        }

#if DW_NDIM >= 3
        sigmazz[i] += pml_z ? -DIFFZ1(SIGMAZZ_Z_PML) : -DIFFZ1(SIGMAZZ_Z);
#endif
#if DW_NDIM >= 2
        sigmayy[i] += pml_y ? -DIFFY1(SIGMAYY_Y_PML) : -DIFFY1(SIGMAYY_Y);
#endif
        sigmaxx[i] += pml_x ? -DIFFX1(SIGMAXX_X_PML) : -DIFFX1(SIGMAXX_X);
      }

#if DW_NDIM >= 2
      if (y < ny - FD_PAD && x < nx - FD_PAD) {
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        if (mu_requires_grad) {
          grad_mu_yx[i] +=
              sigmaxy[i] * dvydx_plus_dvxdy_store[i] * (DW_DTYPE)step_ratio;
        }

        if (pml_y) {
          m_vxy[i] =
              MU_YX(0, 0, 0) * dt * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i];
        }
        if (pml_x) {
          m_vyx[i] =
              MU_YX(0, 0, 0) * dt * axh[x] * sigmaxy[i] + axh[x] * m_vyx[i];
        }

        sigmaxy[i] += (pml_y ? -DIFFYH1(SIGMAXY_Y_PML) : -DIFFYH1(SIGMAXY_Y)) +
                      (pml_x ? -DIFFXH1(SIGMAXY_X_PML) : -DIFFXH1(SIGMAXY_X));
      }
#endif

#if DW_NDIM >= 3
      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        if (mu_requires_grad) {
          grad_mu_zx[i] +=
              sigmaxz[i] * dvzdx_plus_dvxdz_store[i] * (DW_DTYPE)step_ratio;
        }

        if (pml_z) {
          m_vxz[i] =
              MU_ZX(0, 0, 0) * dt * azh[z] * sigmaxz[i] + azh[z] * m_vxz[i];
        }
        if (pml_x) {
          m_vzx[i] =
              MU_ZX(0, 0, 0) * dt * axh[x] * sigmaxz[i] + axh[x] * m_vzx[i];
        }

        sigmaxz[i] += (pml_z ? -DIFFZH1(SIGMAXZ_Z_PML) : -DIFFZH1(SIGMAXZ_Z)) +
                      (pml_x ? -DIFFXH1(SIGMAXZ_X_PML) : -DIFFXH1(SIGMAXZ_X));
      }

      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        if (mu_requires_grad) {
          grad_mu_zy[i] +=
              sigmayz[i] * dvzdy_plus_dvydz_store[i] * (DW_DTYPE)step_ratio;
        }

        if (pml_z) {
          m_vyz[i] =
              MU_ZY(0, 0, 0) * dt * azh[z] * sigmayz[i] + azh[z] * m_vyz[i];
        }
        if (pml_y) {
          m_vzy[i] =
              MU_ZY(0, 0, 0) * dt * ayh[y] * sigmayz[i] + ayh[y] * m_vzy[i];
        }

        sigmayz[i] += (pml_z ? -DIFFZH1(SIGMAYZ_Z_PML) : -DIFFZH1(SIGMAYZ_Z)) +
                      (pml_y ? -DIFFYH1(SIGMAYZ_Y_PML) : -DIFFYH1(SIGMAYZ_Y));
      }
#endif

#if DW_NDIM == 3
    }
#endif
  }
}

int set_config(DW_DTYPE const dt_h,
#if DW_NDIM >= 3
               DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
               DW_DTYPE const rdy_h,
#endif
               DW_DTYPE const rdx_h, int64_t const n_shots_h,
#if DW_NDIM >= 3
               int64_t const nz_h,
#endif
#if DW_NDIM >= 2
               int64_t const ny_h,
#endif
               int64_t const nx_h,
#if DW_NDIM >= 3
               int64_t const n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
               int64_t const n_sources_y_per_shot_h,
#endif
               int64_t const n_sources_x_per_shot_h,
               int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
               int64_t const n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
               int64_t const n_receivers_y_per_shot_h,
#endif
               int64_t const n_receivers_x_per_shot_h,
               int64_t const n_receivers_p_per_shot_h,
               int64_t const step_ratio_h,
#if DW_NDIM >= 3
               int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
               int64_t const pml_y0_h, int64_t const pml_y1_h,
#endif
               int64_t const pml_x0_h, int64_t const pml_x1_h,
               bool const lamb_batched_h, bool const mu_batched_h,
               bool const buoyancy_batched_h) {
#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

  gpuErrchk(cudaMemcpyToSymbol(dt, &dt_h, sizeof(DW_DTYPE)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(n_sources_z_per_shot, &n_sources_z_per_shot_h,
                               sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(n_sources_y_per_shot, &n_sources_y_per_shot_h,
                               sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(n_sources_x_per_shot, &n_sources_x_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_p_per_shot, &n_sources_p_per_shot_h,
                               sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_z_per_shot,
                               &n_receivers_z_per_shot_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_y_per_shot,
                               &n_receivers_y_per_shot_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_x_per_shot,
                               &n_receivers_x_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_p_per_shot,
                               &n_receivers_p_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(lamb_batched, &lamb_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(mu_batched, &mu_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(buoyancy_batched, &buoyancy_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        int FUNC(forward)(
            DW_DTYPE const *__restrict const lamb,
            DW_DTYPE const *__restrict const mu,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const mu_zy,
            DW_DTYPE const *__restrict const mu_zx,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const mu_yx,
#endif
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const buoyancy_y,
#endif
            DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const f_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const f_y,
#endif
            DW_DTYPE const *__restrict const f_x,
            DW_DTYPE const *__restrict const f_p,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const vz, DW_DTYPE *__restrict const sigmazz,
            DW_DTYPE *__restrict const sigmayz,
            DW_DTYPE *__restrict const sigmaxz,
            DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
            DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
            DW_DTYPE *__restrict const m_vxz,
            DW_DTYPE *__restrict const m_sigmazzz,
            DW_DTYPE *__restrict const m_sigmayzy,
            DW_DTYPE *__restrict const m_sigmaxzx,
            DW_DTYPE *__restrict const m_sigmayzz,
            DW_DTYPE *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
            DW_DTYPE *__restrict const sigmaxy,
            DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
            DW_DTYPE *__restrict const m_vxy,
            DW_DTYPE *__restrict const m_sigmayyy,
            DW_DTYPE *__restrict const m_sigmaxyy,
            DW_DTYPE *__restrict const m_sigmaxyx,
#endif
            DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
            DW_DTYPE *__restrict const m_vxx,
            DW_DTYPE *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const dvzdbuoyancy_store_1a,
            DW_DTYPE *__restrict const dvzdbuoyancy_store_1b,
            void *__restrict const dvzdbuoyancy_store_2,
            void *__restrict const dvzdbuoyancy_store_3,
            char const *__restrict const
                *__restrict const dvzdbuoyancy_filenames_ptr,
            DW_DTYPE *__restrict const dvzdz_store_1a,
            DW_DTYPE *__restrict const dvzdz_store_1b,
            void *__restrict const dvzdz_store_2,
            void *__restrict const dvzdz_store_3,
            char const *__restrict const *__restrict const dvzdz_filenames_ptr,
            DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1a,
            DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1b,
            void *__restrict const dvzdx_plus_dvxdz_store_2,
            void *__restrict const dvzdx_plus_dvxdz_store_3,
            char const *__restrict const
                *__restrict const dvzdx_plus_dvxdz_filenames_ptr,
            DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1a,
            DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1b,
            void *__restrict const dvzdy_plus_dvydz_store_2,
            void *__restrict const dvzdy_plus_dvydz_store_3,
            char const *__restrict const
                *__restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const dvydbuoyancy_store_1a,
            DW_DTYPE *__restrict const dvydbuoyancy_store_1b,
            void *__restrict const dvydbuoyancy_store_2,
            void *__restrict const dvydbuoyancy_store_3,
            char const *__restrict const
                *__restrict const dvydbuoyancy_filenames_ptr,
            DW_DTYPE *__restrict const dvydy_store_1a,
            DW_DTYPE *__restrict const dvydy_store_1b,
            void *__restrict const dvydy_store_2,
            void *__restrict const dvydy_store_3,
            char const *__restrict const *__restrict const dvydy_filenames_ptr,
            DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1a,
            DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1b,
            void *__restrict const dvydx_plus_dvxdy_store_2,
            void *__restrict const dvydx_plus_dvxdy_store_3,
            char const *__restrict const
                *__restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
            DW_DTYPE *__restrict const dvxdbuoyancy_store_1a,
            DW_DTYPE *__restrict const dvxdbuoyancy_store_1b,
            void *__restrict const dvxdbuoyancy_store_2,
            void *__restrict const dvxdbuoyancy_store_3,
            char const *__restrict const
                *__restrict const dvxdbuoyancy_filenames_ptr,
            DW_DTYPE *__restrict const dvxdx_store_1a,
            DW_DTYPE *__restrict const dvxdx_store_1b,
            void *__restrict const dvxdx_store_2,
            void *__restrict const dvxdx_store_3,
            char const *__restrict const *__restrict const dvxdx_filenames_ptr,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const r_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const r_y,
#endif
            DW_DTYPE *__restrict const r_x, DW_DTYPE *__restrict const r_p,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const az,
            DW_DTYPE const *__restrict const bz,
            DW_DTYPE const *__restrict const azh,
            DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const ayh,
            DW_DTYPE const *__restrict const byh,
#endif
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const axh,
            DW_DTYPE const *__restrict const bxh,
#if DW_NDIM >= 3
            int64_t const *__restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict const sources_y_i,
#endif
            int64_t const *__restrict const sources_x_i,
            int64_t const *__restrict const sources_p_i,
#if DW_NDIM >= 3
            int64_t const *__restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict const receivers_y_i,
#endif
            int64_t const *__restrict const receivers_x_i,
            int64_t const *__restrict const receivers_p_i,
#if DW_NDIM >= 3
            DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy_h,
#endif
            DW_DTYPE const rdx_h, DW_DTYPE const dt_h, int64_t const nt,
            int64_t const n_shots_h,
#if DW_NDIM >= 3
            int64_t const nz_h,
#endif
#if DW_NDIM >= 2
            int64_t const ny_h,
#endif
            int64_t const nx_h,
#if DW_NDIM >= 3
            int64_t const n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_sources_y_per_shot_h,
#endif
            int64_t const n_sources_x_per_shot_h,
            int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
            int64_t const n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_receivers_y_per_shot_h,
#endif
            int64_t n_receivers_x_per_shot_h,
            int64_t const n_receivers_p_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const lamb_requires_grad,
            bool const mu_requires_grad, bool const buoyancy_requires_grad,
            bool const lamb_batched_h, bool const mu_batched_h,
            bool const buoyancy_batched_h, bool const storage_compression,
            int64_t const start_t,
#if DW_NDIM >= 3
            int64_t const pml_z0_h,
#endif
#if DW_NDIM >= 2
            int64_t const pml_y0_h,
#endif
            int64_t const pml_x0_h,
#if DW_NDIM >= 3
            int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
            int64_t const pml_y1_h,
#endif
            int64_t const pml_x1_h, int64_t const device,
            cudaStream_t stream_compute) {
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD + 1, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
  dim3 const dimBlock_src_rcv(32, 1, 1);

  gpuErrchk(cudaSetDevice(device));
  set_config(dt_h,
#if DW_NDIM >= 3
             rdz_h,
#endif
#if DW_NDIM >= 2
             rdy_h,
#endif
             rdx_h, n_shots_h,
#if DW_NDIM >= 3
             nz_h,
#endif
#if DW_NDIM >= 2
             ny_h,
#endif
             nx_h,
#if DW_NDIM >= 3
             n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
             n_sources_y_per_shot_h,
#endif
             n_sources_x_per_shot_h, n_sources_p_per_shot_h,
#if DW_NDIM >= 3
             n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
             n_receivers_y_per_shot_h,
#endif
             n_receivers_x_per_shot_h, n_receivers_p_per_shot_h, step_ratio_h,
#if DW_NDIM >= 3
             pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
             pml_y0_h, pml_y1_h,
#endif
             pml_x0_h, pml_x1_h, lamb_batched_h, mu_batched_h,
             buoyancy_batched_h);

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel = ny_h * nx_h;
#else
  int64_t const shot_numel = nx_h;
#endif

#define OPEN_FILE(name, grad_cond)                   \
  ScopedFile fp_##name;                              \
  if (storage_mode == STORAGE_DISK && (grad_cond)) { \
    fp_##name.open(name##_filenames_ptr[0], "ab");   \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (lamb_requires_grad || mu_requires_grad || buoyancy_requires_grad);

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  cudaEvent_t event_storage_done;
  if (use_double_buffering) {
    gpuErrchk(
        cudaStreamCreateWithFlags(&stream_storage, cudaStreamNonBlocking));
    gpuErrchk(cudaEventCreate(&event_storage_done_a));
    gpuErrchk(cudaEventCreate(&event_storage_done_b));
    gpuErrchk(cudaEventCreate(&event_compute_done_a));
    gpuErrchk(cudaEventCreate(&event_compute_done_b));
    gpuErrchk(cudaEventRecord(event_storage_done_a, stream_storage));
    gpuErrchk(cudaEventRecord(event_storage_done_b, stream_storage));
    event_storage_done = event_storage_done_a;
  }

#if DW_NDIM >= 3
  OPEN_FILE(dvzdbuoyancy, buoyancy_requires_grad)
  OPEN_FILE(dvzdz, lamb_requires_grad || mu_requires_grad)
  OPEN_FILE(dvzdx_plus_dvxdz, mu_requires_grad)
  OPEN_FILE(dvzdy_plus_dvydz, mu_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE(dvydbuoyancy, buoyancy_requires_grad)
  OPEN_FILE(dvydy, lamb_requires_grad || mu_requires_grad)
  OPEN_FILE(dvydx_plus_dvxdy, mu_requires_grad)
#endif
  OPEN_FILE(dvxdbuoyancy, buoyancy_requires_grad)
  OPEN_FILE(dvxdx, lamb_requires_grad || mu_requires_grad)

  for (t = start_t; t < start_t + nt; ++t) {
    int64_t const step_idx = t / step_ratio_h;
#define SETUP_STORE(name, grad_cond)                                         \
  DW_DTYPE *__restrict name##_store_1_t =                                    \
      name##_store_1a +                                                      \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)              \
           ? (t / step_ratio_h) * n_shots_h * shot_numel                     \
           : 0);                                                             \
  void *__restrict const name##_store_2_t =                                  \
      (uint8_t *)name##_store_2 +                                            \
      (storage_mode == STORAGE_DEVICE && storage_compression                 \
           ? (t / step_ratio_h) * n_shots_h * shot_bytes_comp                \
           : 0);                                                             \
  void *__restrict const name##_store_3_t =                                  \
      (uint8_t *)name##_store_3 +                                            \
      (storage_mode == STORAGE_CPU                                           \
           ? (t / step_ratio_h) * n_shots_h *                                \
                 (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) \
           : 0);                                                             \
  bool const name##_cond = (grad_cond) && ((t % step_ratio_h) == 0);         \
  if ((grad_cond) && use_double_buffering) {                                 \
    if (step_idx % 2 != 0) {                                                 \
      name##_store_1_t = name##_store_1b;                                    \
    }                                                                        \
  }

#if DW_NDIM >= 3
    SETUP_STORE(dvzdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvzdz, lamb_requires_grad || mu_requires_grad)
    SETUP_STORE(dvzdx_plus_dvxdz, mu_requires_grad)
    SETUP_STORE(dvzdy_plus_dvydz, mu_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE(dvydbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvydy, lamb_requires_grad || mu_requires_grad)
    SETUP_STORE(dvydx_plus_dvxdy, mu_requires_grad)
#endif
    SETUP_STORE(dvxdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvxdx, lamb_requires_grad || mu_requires_grad)

    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

#if DW_NDIM >= 3
    if (n_receivers_z_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_z_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers_z<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                           stream_compute>>>(
          r_z + t * n_shots_h * n_receivers_z_per_shot_h, vz, receivers_z_i);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_receivers_y_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_y_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers_y<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                           stream_compute>>>(
          r_y + t * n_shots_h * n_receivers_y_per_shot_h, vy, receivers_y_i);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_receivers_x_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_x_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers_x<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                           stream_compute>>>(
          r_x + t * n_shots_h * n_receivers_x_per_shot_h, vx, receivers_x_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_pressure_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                  stream_compute>>>(
          r_p + t * n_shots_h * n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, receivers_p_i);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        sigmazz, sigmayz, sigmaxz,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmaxy,
#endif
        sigmaxx,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmayzz, m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
#endif
        m_sigmaxxx,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t,
#endif
        dvxdbuoyancy_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

#if DW_NDIM >= 3
    if (n_sources_z_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_z_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_z<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vz, f_z + t * n_shots_h * n_sources_z_per_shot_h, sources_z_i);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_sources_y_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_y_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_y<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vy, f_y + t * n_shots_h * n_sources_y_per_shot_h, sources_y_i);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_sources_x_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_x_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_x<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vx, f_x + t * n_shots_h * n_sources_x_per_shot_h, sources_x_i);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        lamb, mu,
#if DW_NDIM >= 3
        mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
        mu_yx,
#endif
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        sigmazz, sigmayz, sigmaxz,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmaxy,
#endif
        sigmaxx,
#if DW_NDIM >= 3
        m_vzz, m_vzy, m_vzx, m_vyz, m_vxz,
#endif
#if DW_NDIM >= 2
        m_vyy, m_vyx, m_vxy,
#endif
        m_vxx,
#if DW_NDIM >= 3
        dvzdz_store_1_t, dvzdx_plus_dvxdz_store_1_t, dvzdy_plus_dvydz_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydy_store_1_t, dvydx_plus_dvxdy_store_1_t,
#endif
        dvxdx_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, lamb_requires_grad && ((t % step_ratio_h) == 0),
        mu_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

#define SAVE_SNAPSHOT(name)                                                  \
  if (name##_cond) {                                                         \
    if (STORAGE_FUNC(save_snapshot_gpu)(                                     \
            name##_store_1_t, name##_store_2_t, name##_store_3_t, fp_##name, \
            storage_mode, storage_compression, step_idx, shot_bytes_uncomp,  \
            shot_bytes_comp, n_shots_h, DIM_ARGS,                            \
            use_double_buffering ? stream_storage : stream_compute) != 0)    \
      return 1;                                                              \
  }

#if DW_NDIM >= 3
    SAVE_SNAPSHOT(dvzdbuoyancy)
    SAVE_SNAPSHOT(dvzdz)
    SAVE_SNAPSHOT(dvzdx_plus_dvxdz)
    SAVE_SNAPSHOT(dvzdy_plus_dvydz)
#endif
#if DW_NDIM >= 2
    SAVE_SNAPSHOT(dvydbuoyancy)
    SAVE_SNAPSHOT(dvydy)
    SAVE_SNAPSHOT(dvydx_plus_dvxdy)
#endif
    SAVE_SNAPSHOT(dvxdbuoyancy)
    SAVE_SNAPSHOT(dvxdx)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_p<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, f_p + t * n_shots_h * n_sources_p_per_shot_h, sources_p_i);
      CHECK_KERNEL_ERROR
    }
  }

  if (use_double_buffering)
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
  return 0;
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        int FUNC(backward)(
            DW_DTYPE const *__restrict const lamb,
            DW_DTYPE const *__restrict const mu,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const mu_zy,
            DW_DTYPE const *__restrict const mu_zx,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const mu_yx,
#endif
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const buoyancy_y,
#endif
            DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const grad_r_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const grad_r_y,
#endif
            DW_DTYPE const *__restrict const grad_r_x,
            DW_DTYPE const *__restrict const grad_r_p,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const vz, DW_DTYPE *__restrict const sigmazz,
            DW_DTYPE *__restrict const sigmayz,
            DW_DTYPE *__restrict const sigmaxz,
            DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
            DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
            DW_DTYPE *__restrict const m_vxz, DW_DTYPE *__restrict m_sigmazzz,
            DW_DTYPE *__restrict m_sigmayzy, DW_DTYPE *__restrict m_sigmaxzx,
            DW_DTYPE *__restrict m_sigmayzz, DW_DTYPE *__restrict m_sigmaxzz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
            DW_DTYPE *__restrict const sigmaxy,
            DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
            DW_DTYPE *__restrict const m_vxy, DW_DTYPE *__restrict m_sigmayyy,
            DW_DTYPE *__restrict m_sigmaxyy, DW_DTYPE *__restrict m_sigmaxyx,
#endif
            DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
            DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict m_sigmaxxx,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict m_sigmazzzn, DW_DTYPE *__restrict m_sigmayzyn,
            DW_DTYPE *__restrict m_sigmaxzxn, DW_DTYPE *__restrict m_sigmayzzn,
            DW_DTYPE *__restrict m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict m_sigmayyyn, DW_DTYPE *__restrict m_sigmaxyyn,
            DW_DTYPE *__restrict m_sigmaxyxn,
#endif
            DW_DTYPE *__restrict m_sigmaxxxn,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const dvzdbuoyancy_store_1a,
            DW_DTYPE *__restrict const dvzdbuoyancy_store_1b,
            void *__restrict const dvzdbuoyancy_store_2,
            void *__restrict const dvzdbuoyancy_store_3,
            char const *__restrict const
                *__restrict const dvzdbuoyancy_filenames_ptr,
            DW_DTYPE *__restrict const dvzdz_store_1a,
            DW_DTYPE *__restrict const dvzdz_store_1b,
            void *__restrict const dvzdz_store_2,
            void *__restrict const dvzdz_store_3,
            char const *__restrict const *__restrict const dvzdz_filenames_ptr,
            DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1a,
            DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1b,
            void *__restrict const dvzdx_plus_dvxdz_store_2,
            void *__restrict const dvzdx_plus_dvxdz_store_3,
            char const *__restrict const
                *__restrict const dvzdx_plus_dvxdz_filenames_ptr,
            DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1a,
            DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1b,
            void *__restrict const dvzdy_plus_dvydz_store_2,
            void *__restrict const dvzdy_plus_dvydz_store_3,
            char const *__restrict const
                *__restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const dvydbuoyancy_store_1a,
            DW_DTYPE *__restrict const dvydbuoyancy_store_1b,
            void *__restrict const dvydbuoyancy_store_2,
            void *__restrict const dvydbuoyancy_store_3,
            char const *__restrict const
                *__restrict const dvydbuoyancy_filenames_ptr,
            DW_DTYPE *__restrict const dvydy_store_1a,
            DW_DTYPE *__restrict const dvydy_store_1b,
            void *__restrict const dvydy_store_2,
            void *__restrict const dvydy_store_3,
            char const *__restrict const *__restrict const dvydy_filenames_ptr,
            DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1a,
            DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1b,
            void *__restrict const dvydx_plus_dvxdy_store_2,
            void *__restrict const dvydx_plus_dvxdy_store_3,
            char const *__restrict const
                *__restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
            DW_DTYPE *__restrict const dvxdbuoyancy_store_1a,
            DW_DTYPE *__restrict const dvxdbuoyancy_store_1b,
            void *__restrict const dvxdbuoyancy_store_2,
            void *__restrict const dvxdbuoyancy_store_3,
            char const *__restrict const
                *__restrict const dvxdbuoyancy_filenames_ptr,
            DW_DTYPE *__restrict const dvxdx_store_1a,
            DW_DTYPE *__restrict const dvxdx_store_1b,
            void *__restrict const dvxdx_store_2,
            void *__restrict const dvxdx_store_3,
            char const *__restrict const *__restrict const dvxdx_filenames_ptr,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const grad_f_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const grad_f_y,
#endif
            DW_DTYPE *__restrict const grad_f_x,
            DW_DTYPE *__restrict const grad_f_p,
            DW_DTYPE *__restrict const grad_lamb,
            DW_DTYPE *__restrict const grad_mu,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const grad_mu_zy,
            DW_DTYPE *__restrict const grad_mu_zx,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const grad_mu_yx,
#endif
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const grad_buoyancy_y,
#endif
            DW_DTYPE *__restrict const grad_buoyancy_x,
            DW_DTYPE *__restrict const grad_lamb_shot,
            DW_DTYPE *__restrict const grad_mu_shot,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const grad_mu_zy_shot,
            DW_DTYPE *__restrict const grad_mu_zx_shot,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const grad_mu_yx_shot,
#endif
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const grad_buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const grad_buoyancy_y_shot,
#endif
            DW_DTYPE *__restrict const grad_buoyancy_x_shot,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const az,
            DW_DTYPE const *__restrict const bz,
            DW_DTYPE const *__restrict const azh,
            DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const ayh,
            DW_DTYPE const *__restrict const byh,
#endif
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const axh,
            DW_DTYPE const *__restrict const bxh,
#if DW_NDIM >= 3
            int64_t const *__restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict const sources_y_i,
#endif
            int64_t const *__restrict const sources_x_i,
            int64_t const *__restrict const sources_p_i,
#if DW_NDIM >= 3
            int64_t const *__restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict const receivers_y_i,
#endif
            int64_t const *__restrict const receivers_x_i,
            int64_t const *__restrict const receivers_p_i,
#if DW_NDIM >= 3
            DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy_h,
#endif
            DW_DTYPE const rdx_h, DW_DTYPE const dt_h, int64_t const nt,
            int64_t const n_shots_h,
#if DW_NDIM >= 3
            int64_t const nz_h,
#endif
#if DW_NDIM >= 2
            int64_t const ny_h,
#endif
            int64_t const nx_h,
#if DW_NDIM >= 3
            int64_t const n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_sources_y_per_shot_h,
#endif
            int64_t const n_sources_x_per_shot_h,
            int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
            int64_t const n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_receivers_y_per_shot_h,
#endif
            int64_t n_receivers_x_per_shot_h,
            int64_t const n_receivers_p_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const lamb_requires_grad,
            bool const mu_requires_grad, bool const buoyancy_requires_grad,
            bool const lamb_batched_h, bool const mu_batched_h,
            bool const buoyancy_batched_h, bool const storage_compression,
            int64_t const start_t,
#if DW_NDIM >= 3
            int64_t const pml_z0_h,
#endif
#if DW_NDIM >= 2
            int64_t const pml_y0_h,
#endif
            int64_t const pml_x0_h,
#if DW_NDIM >= 3
            int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
            int64_t const pml_y1_h,
#endif
            int64_t const pml_x1_h, int64_t const device,
            cudaStream_t stream_compute) {
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD + 1, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
  dim3 const dimBlock_src_rcv(32, 1, 1);

  gpuErrchk(cudaSetDevice(device));
  set_config(dt_h,
#if DW_NDIM >= 3
             rdz_h,
#endif
#if DW_NDIM >= 2
             rdy_h,
#endif
             rdx_h, n_shots_h,
#if DW_NDIM >= 3
             nz_h,
#endif
#if DW_NDIM >= 2
             ny_h,
#endif
             nx_h,
#if DW_NDIM >= 3
             n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
             n_sources_y_per_shot_h,
#endif
             n_sources_x_per_shot_h, n_sources_p_per_shot_h,
#if DW_NDIM >= 3
             n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
             n_receivers_y_per_shot_h,
#endif
             n_receivers_x_per_shot_h, n_receivers_p_per_shot_h, step_ratio_h,
#if DW_NDIM >= 3
             pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
             pml_y0_h, pml_y1_h,
#endif
             pml_x0_h, pml_x1_h, lamb_batched_h, mu_batched_h,
             buoyancy_batched_h);

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel = ny_h * nx_h;
#else
  int64_t const shot_numel = nx_h;
#endif

#define OPEN_FILE_READ(name, grad_cond)              \
  ScopedFile fp_##name;                              \
  if (storage_mode == STORAGE_DISK && (grad_cond)) { \
    fp_##name.open(name##_filenames_ptr[0], "rb");   \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (lamb_requires_grad || mu_requires_grad || buoyancy_requires_grad);

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  if (use_double_buffering) {
    gpuErrchk(
        cudaStreamCreateWithFlags(&stream_storage, cudaStreamNonBlocking));
    gpuErrchk(cudaEventCreate(&event_storage_done_a));
    gpuErrchk(cudaEventCreate(&event_storage_done_b));
    gpuErrchk(cudaEventCreate(&event_compute_done_a));
    gpuErrchk(cudaEventCreate(&event_compute_done_b));
    gpuErrchk(cudaEventRecord(event_compute_done_a, stream_compute));
    gpuErrchk(cudaEventRecord(event_compute_done_b, stream_compute));
  }

#if DW_NDIM >= 3
  OPEN_FILE_READ(dvzdbuoyancy, buoyancy_requires_grad)
  OPEN_FILE_READ(dvzdz, lamb_requires_grad || mu_requires_grad)
  OPEN_FILE_READ(dvzdx_plus_dvxdz, mu_requires_grad)
  OPEN_FILE_READ(dvzdy_plus_dvydz, mu_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(dvydbuoyancy, buoyancy_requires_grad)
  OPEN_FILE_READ(dvydy, lamb_requires_grad || mu_requires_grad)
  OPEN_FILE_READ(dvydx_plus_dvxdy, mu_requires_grad)
#endif
  OPEN_FILE_READ(dvxdbuoyancy, buoyancy_requires_grad)
  OPEN_FILE_READ(dvxdx, lamb_requires_grad || mu_requires_grad)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio_h;

#define SETUP_STORE_LOAD(name, grad_cond)                                    \
  DW_DTYPE const *__restrict name##_store_1_t =                              \
      name##_store_1a +                                                      \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)              \
           ? (t / step_ratio_h) * n_shots_h * shot_numel                     \
           : 0);                                                             \
  void *__restrict const name##_store_2_t =                                  \
      (uint8_t *)name##_store_2 +                                            \
      (storage_mode == STORAGE_DEVICE && storage_compression                 \
           ? (t / step_ratio_h) * n_shots_h * shot_bytes_comp                \
           : 0);                                                             \
  void *__restrict const name##_store_3_t =                                  \
      (uint8_t *)name##_store_3 +                                            \
      (storage_mode == STORAGE_CPU                                           \
           ? (t / step_ratio_h) * n_shots_h *                                \
                 (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) \
           : 0);                                                             \
  bool const name##_cond = (grad_cond) && ((t % step_ratio_h) == 0);         \
  if (name##_cond) {                                                         \
    if (use_double_buffering && step_idx % 2 != 0) {                         \
      name##_store_1_t = name##_store_1b;                                    \
    }                                                                        \
    if (STORAGE_FUNC(load_snapshot_gpu)(                                     \
            (void *)name##_store_1_t, name##_store_2_t, name##_store_3_t,    \
            fp_##name, storage_mode, storage_compression, step_idx,          \
            shot_bytes_uncomp, shot_bytes_comp, n_shots_h, DIM_ARGS,         \
            use_double_buffering ? stream_storage : stream_compute) != 0)    \
      return 1;                                                              \
  }

    cudaEvent_t event_storage_done = event_storage_done_a;
    cudaEvent_t event_compute_done = event_compute_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_storage_done = event_storage_done_b;
        event_compute_done = event_compute_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

#if DW_NDIM >= 3
    SETUP_STORE_LOAD(dvzdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvzdz, lamb_requires_grad || mu_requires_grad)
    SETUP_STORE_LOAD(dvzdx_plus_dvxdz, mu_requires_grad)
    SETUP_STORE_LOAD(dvzdy_plus_dvydz, mu_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(dvydbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvydy, lamb_requires_grad || mu_requires_grad)
    SETUP_STORE_LOAD(dvydx_plus_dvxdy, mu_requires_grad)
#endif
    SETUP_STORE_LOAD(dvxdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvxdx, lamb_requires_grad || mu_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_pressure_receivers<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                          stream_compute>>>(
          grad_f_p + t * n_shots_h * n_sources_p_per_shot_h,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, sources_p_i);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        lamb, mu,
#if DW_NDIM >= 3
        mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
        mu_yx,
#endif
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        sigmazz, sigmayz, sigmaxz,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmaxy,
#endif
        sigmaxx,
#if DW_NDIM >= 3
        m_vzz, m_vzy, m_vzx, m_vyz, m_vxz,
#endif
#if DW_NDIM >= 2
        m_vyy, m_vyx, m_vxy,
#endif
        m_vxx,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmayzz, m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
#endif
        m_sigmaxxx,
#if DW_NDIM >= 3
        m_sigmazzzn, m_sigmayzyn, m_sigmaxzxn, m_sigmayzzn, m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
        m_sigmayyyn, m_sigmaxyyn, m_sigmaxyxn,
#endif
        m_sigmaxxxn,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t,
#endif
        dvxdbuoyancy_store_1_t,
#if DW_NDIM >= 3
        grad_buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        grad_buoyancy_y_shot,
#endif
        grad_buoyancy_x_shot,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

#if DW_NDIM >= 3
    if (n_sources_z_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_z_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_receivers_z<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                   stream_compute>>>(
          grad_f_z + t * n_shots_h * n_sources_z_per_shot_h, vz, sources_z_i);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_sources_y_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_y_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_receivers_y<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                   stream_compute>>>(
          grad_f_y + t * n_shots_h * n_sources_y_per_shot_h, vy, sources_y_i);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_sources_x_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_x_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_receivers_x<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                   stream_compute>>>(
          grad_f_x + t * n_shots_h * n_sources_x_per_shot_h, vx, sources_x_i);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        lamb, mu,
#if DW_NDIM >= 3
        mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
        mu_yx,
#endif
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        sigmazz, sigmayz, sigmaxz,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmaxy,
#endif
        sigmaxx,
#if DW_NDIM >= 3
        m_vzz, m_vzy, m_vzx, m_vyz, m_vxz,
#endif
#if DW_NDIM >= 2
        m_vyy, m_vyx, m_vxy,
#endif
        m_vxx,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmayzz, m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
#endif
        m_sigmaxxx,
#if DW_NDIM >= 3
        dvzdz_store_1_t, dvzdx_plus_dvxdz_store_1_t, dvzdy_plus_dvydz_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydy_store_1_t, dvydx_plus_dvxdy_store_1_t,
#endif
        dvxdx_store_1_t, grad_lamb_shot, grad_mu_shot,
#if DW_NDIM >= 3
        grad_mu_zy_shot, grad_mu_zx_shot,
#endif
#if DW_NDIM >= 2
        grad_mu_yx_shot,
#endif
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, lamb_requires_grad && ((t % step_ratio_h) == 0),
        mu_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }
#if DW_NDIM >= 3
    if (n_receivers_z_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_z_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_sources_z<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                              stream_compute>>>(
          vz, grad_r_z + t * n_shots_h * n_receivers_z_per_shot_h,
          receivers_z_i);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_receivers_y_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_y_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_sources_y<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                              stream_compute>>>(
          vy, grad_r_y + t * n_shots_h * n_receivers_y_per_shot_h,
          receivers_y_i);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_receivers_x_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_x_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_sources_x<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                              stream_compute>>>(
          vx, grad_r_x + t * n_shots_h * n_receivers_x_per_shot_h,
          receivers_x_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_pressure_sources<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                     stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, grad_r_p + t * n_shots_h * n_receivers_p_per_shot_h,
          receivers_p_i);
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM >= 3
    std::swap(m_sigmazzz, m_sigmazzzn);
    std::swap(m_sigmayzy, m_sigmayzyn);
    std::swap(m_sigmaxzx, m_sigmaxzxn);
    std::swap(m_sigmayzz, m_sigmayzzn);
    std::swap(m_sigmaxzz, m_sigmaxzzn);
#endif
#if DW_NDIM >= 2
    std::swap(m_sigmayyy, m_sigmayyyn);
    std::swap(m_sigmaxyy, m_sigmaxyyn);
    std::swap(m_sigmaxyx, m_sigmaxyxn);
#endif
    std::swap(m_sigmaxxx, m_sigmaxxxn);
  }

#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine = ceil_div(nx_h, dimBlock_combine.x);
  unsigned int const gridy_combine = ceil_div(ny_h, dimBlock_combine.y);
  unsigned int const gridz_combine = ceil_div(nz_h, dimBlock_combine.z);
#elif DW_NDIM == 2
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine = ceil_div(nx_h, dimBlock_combine.x);
  unsigned int const gridy_combine = ceil_div(ny_h, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock_combine(128, 1, 1);
  unsigned int const gridx_combine = ceil_div(nx_h, dimBlock_combine.x);
  unsigned int const gridy_combine = 1;
  unsigned int const gridz_combine = 1;
#endif
  dim3 const dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

  if (lamb_requires_grad && !lamb_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_lamb, grad_lamb_shot);
    CHECK_KERNEL_ERROR
  }
  if (mu_requires_grad && !mu_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_mu, grad_mu_shot);
    CHECK_KERNEL_ERROR
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_mu_zy, grad_mu_zy_shot);
    CHECK_KERNEL_ERROR
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_mu_zx, grad_mu_zx_shot);
    CHECK_KERNEL_ERROR
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_mu_yx, grad_mu_yx_shot);
    CHECK_KERNEL_ERROR
#endif
  }
  if (buoyancy_requires_grad && !buoyancy_batched_h && n_shots_h > 1) {
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_buoyancy_z, grad_buoyancy_z_shot);
    CHECK_KERNEL_ERROR
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_buoyancy_y, grad_buoyancy_y_shot);
    CHECK_KERNEL_ERROR
#endif
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_buoyancy_x, grad_buoyancy_x_shot);
    CHECK_KERNEL_ERROR
  }
  return 0;
}

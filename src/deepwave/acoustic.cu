/*
 * Acoustic wave equation propagator (CUDA implementation)
 */

#include <stdio.h>

#include <algorithm>  // for std::swap
#include <cstdint>

#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  acoustic_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz)*ny * nx + (dy)*nx + (dx))
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy)*nx + (dx))
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#define DIM_ARGS nx_h
#endif

#define P(dz, dy, dx) p[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx[ND_INDEX(i, dz, dy, dx)]

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define K(dz, dy, dx) k_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

#if DW_NDIM >= 3
#define PHI_Z(dz, dy, dx) phi_z[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PHI_Y(dz, dy, dx) phi_y[ND_INDEX(i, dz, dy, dx)]
#endif
#define PHI_X(dz, dy, dx) phi_x[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define PSI_Z(dz, dy, dx) psi_z[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PSI_Y(dz, dy, dx) psi_y[ND_INDEX(i, dz, dy, dx)]
#endif
#define PSI_X(dz, dy, dx) psi_x[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define VZ_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VZ_ADJOINT_PML(dz, dy, dx)  \
  (bz[z + dz] * PHI_Z(dz, dy, dx) - \
   dt * (1 + bz[z + dz]) * K(dz, dy, dx) * P(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VY_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VY_ADJOINT_PML(dz, dy, dx)  \
  (by[y + dy] * PHI_Y(dz, dy, dx) - \
   dt * (1 + by[y + dy]) * K(dz, dy, dx) * P(dz, dy, dx))
#endif
#define VX_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VX_ADJOINT_PML(dz, dy, dx)  \
  (bx[x + dx] * PHI_X(dz, dy, dx) - \
   dt * (1 + bx[x + dx]) * K(dz, dy, dx) * P(dz, dy, dx))

#if DW_NDIM >= 3
#define P_ADJOINT_Z(dz, dy, dx) (-dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#define P_ADJOINT_Z_PML(dz, dy, dx)  \
  (bzh[z + dz] * PSI_Z(dz, dy, dx) - \
   dt * (1 + bzh[z + dz]) * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define P_ADJOINT_Y(dz, dy, dx) (-dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define P_ADJOINT_Y_PML(dz, dy, dx)  \
  (byh[y + dy] * PSI_Y(dz, dy, dx) - \
   dt * (1 + byh[y + dy]) * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define P_ADJOINT_X(dz, dy, dx) (-dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define P_ADJOINT_X_PML(dz, dy, dx)  \
  (bxh[x + dx] * PSI_X(dz, dy, dx) - \
   dt * (1 + bxh[x + dx]) * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

namespace {
__constant__ DW_DTYPE dt;
#if DW_NDIM >= 3
__constant__ DW_DTYPE rdz;
__constant__ int64_t nz;
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
#endif
#if DW_NDIM >= 2
__constant__ DW_DTYPE rdy;
__constant__ int64_t ny;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
#endif
__constant__ DW_DTYPE rdx;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_shots;
__constant__ int64_t step_ratio;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool k_batched;
__constant__ bool b_batched;

__launch_bounds__(32) __global__
    void add_sources(DW_DTYPE *__restrict__ const wf,
                     DW_DTYPE const *__restrict__ const f,
                     int64_t const *__restrict__ const sources_i,
                     int64_t const n_sources_per_shot) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void record_receivers(DW_DTYPE *__restrict__ const r,
                          DW_DTYPE const *__restrict__ const wf,
                          int64_t const *__restrict__ receivers_i,
                          int64_t const n_receivers_per_shot) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
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

__launch_bounds__(128) __global__
    void forward_kernel_v(DW_DTYPE const *__restrict__ const p,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const vy,
#endif
                          DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const psi_y,
#endif
                          DW_DTYPE *__restrict__ const psi_x,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                          DW_DTYPE const *__restrict__ const buoyancy_x,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const bz_grad_store_1,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const by_grad_store_1,
#endif
                          DW_DTYPE *__restrict__ const bx_grad_store_1,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const azh,
                          DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const ayh,
                          DW_DTYPE const *__restrict__ const byh,
#endif
                          DW_DTYPE const *__restrict__ const axh,
                          DW_DTYPE const *__restrict__ const bxh,
                          bool const b_requires_grad) {

#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

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

      DW_DTYPE const *__restrict__ const buoyancy_x_shot =
          b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        DW_DTYPE term_z = DIFFZH1(P);
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        if (pml_z) {
          psi_z[i] = bzh[z] * term_z + azh[z] * psi_z[i];
          term_z += psi_z[i];
        }
        if (b_requires_grad) {
          bz_grad_store_1[i] = term_z;
        }
        vz[i] -= dt * buoyancy_z_shot[j] * term_z;
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        DW_DTYPE term_y = DIFFYH1(P);
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        if (pml_y) {
          psi_y[i] = byh[y] * term_y + ayh[y] * psi_y[i];
          term_y += psi_y[i];
        }
        if (b_requires_grad) {
          by_grad_store_1[i] = term_y;
        }
        vy[i] -= dt * buoyancy_y_shot[j] * term_y;
      }
#endif

      if (x < nx - FD_PAD) {
        DW_DTYPE term_x = DIFFXH1(P);
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        if (pml_x) {
          psi_x[i] = bxh[x] * term_x + axh[x] * psi_x[i];
          term_x += psi_x[i];
        }
        if (b_requires_grad) {
          bx_grad_store_1[i] = term_x;
        }
        vx[i] -= dt * buoyancy_x_shot[j] * term_x;
      }

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__
    void forward_kernel_p(DW_DTYPE *__restrict__ const p,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const vy,
#endif
                          DW_DTYPE const *__restrict__ const vx,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const phi_y,
#endif
                          DW_DTYPE *__restrict__ const phi_x,
                          DW_DTYPE const *__restrict__ const k,
                          DW_DTYPE *__restrict__ const k_grad_store_1,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const az,
                          DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const ay,
                          DW_DTYPE const *__restrict__ const by,
#endif
                          DW_DTYPE const *__restrict__ const ax,
                          DW_DTYPE const *__restrict__ const bx,
                          bool const k_requires_grad) {

#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;

      DW_DTYPE div_v = 0;
#if DW_NDIM >= 3
      DW_DTYPE d_z = DIFFZ1(VZ);
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      if (pml_z) {
        phi_z[i] = az[z] * phi_z[i] + bz[z] * d_z;
        d_z += phi_z[i];
      }
      div_v += d_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE d_y = DIFFY1(VY);
      bool const pml_y = y < pml_y0 || y >= pml_y1;
      if (pml_y) {
        phi_y[i] = ay[y] * phi_y[i] + by[y] * d_y;
        d_y += phi_y[i];
      }
      div_v += d_y;
#endif
      DW_DTYPE d_x = DIFFX1(VX);
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      if (pml_x) {
        phi_x[i] = ax[x] * phi_x[i] + bx[x] * d_x;
        d_x += phi_x[i];
      }
      div_v += d_x;

      if (k_requires_grad) {
        k_grad_store_1[i] = div_v;
      }
      p[i] -= dt * k_shot[j] * div_v;

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__
    void backward_kernel_v(DW_DTYPE const *__restrict__ const p,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const vy,
#endif
                           DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const phi_y,
#endif
                           DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const psi_y,
#endif
                           DW_DTYPE *__restrict__ const psi_x,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const psi_zn,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const psi_yn,
#endif
                           DW_DTYPE *__restrict__ const psi_xn,
                           DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
                           DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                           DW_DTYPE const *__restrict__ const buoyancy_x,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const grad_bz_shot,
                           DW_DTYPE const *__restrict__ const bz_grad_store_1,
                           DW_DTYPE const *__restrict__ const azh,
                           DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const grad_by_shot,
                           DW_DTYPE const *__restrict__ const by_grad_store_1,
                           DW_DTYPE const *__restrict__ const ayh,
                           DW_DTYPE const *__restrict__ const by,
#endif
                           DW_DTYPE *__restrict__ const grad_bx_shot,
                           DW_DTYPE const *__restrict__ const bx_grad_store_1,
                           DW_DTYPE const *__restrict__ const axh,
                           DW_DTYPE const *__restrict__ const bx,
                           bool const b_requires_grad) {

#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;

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

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        vz[i] += (pml_z ? -DIFFZH1(VZ_ADJOINT_PML) : -DIFFZH1(VZ_ADJOINT));
        psi_zn[i] =
            azh[z] * psi_z[i] - dt * buoyancy_z_shot[j] * azh[z] * vz[i];

        if (b_requires_grad) {
          grad_bz_shot[i] -=
              dt * vz[i] * bz_grad_store_1[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        vy[i] += (pml_y ? -DIFFYH1(VY_ADJOINT_PML) : -DIFFYH1(VY_ADJOINT));
        psi_yn[i] =
            ayh[y] * psi_y[i] - dt * buoyancy_y_shot[j] * ayh[y] * vy[i];

        if (b_requires_grad) {
          grad_by_shot[i] -=
              dt * vy[i] * by_grad_store_1[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif

      if (x < nx - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_x_shot =
            b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        vx[i] += (pml_x ? -DIFFXH1(VX_ADJOINT_PML) : -DIFFXH1(VX_ADJOINT));
        psi_xn[i] =
            axh[x] * psi_x[i] - dt * buoyancy_x_shot[j] * axh[x] * vx[i];

        if (b_requires_grad) {
          grad_bx_shot[i] -=
              dt * vx[i] * bx_grad_store_1[i] * (DW_DTYPE)step_ratio;
        }
      }

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__
    void backward_kernel_p(DW_DTYPE *__restrict__ const p,
#if DW_NDIM >= 3
                           DW_DTYPE const *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE const *__restrict__ const vy,
#endif
                           DW_DTYPE const *__restrict__ const vx,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const phi_y,
#endif
                           DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const psi_y,
#endif
                           DW_DTYPE *__restrict__ const psi_x,
                           DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
                           DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                           DW_DTYPE const *__restrict__ const buoyancy_x,
                           DW_DTYPE *__restrict__ const grad_k_shot,
                           DW_DTYPE const *__restrict__ const k_grad_store_1,
#if DW_NDIM >= 3
                           DW_DTYPE const *__restrict__ const az,
                           DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE const *__restrict__ const ay,
                           DW_DTYPE const *__restrict__ const byh,
#endif
                           DW_DTYPE const *__restrict__ const ax,
                           DW_DTYPE const *__restrict__ const bxh,
                           bool const k_requires_grad) {

#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;

      // Update Phi
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      if (pml_z) phi_z[i] = az[z] * phi_z[i] - dt * k_shot[j] * az[z] * p[i];
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
      if (pml_y) phi_y[i] = ay[y] * phi_y[i] - dt * k_shot[j] * ay[y] * p[i];
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      if (pml_x) phi_x[i] = ax[x] * phi_x[i] - dt * k_shot[j] * ax[x] * p[i];

      if (k_requires_grad) {
        grad_k_shot[i] -= dt * p[i] * k_grad_store_1[i] * (DW_DTYPE)step_ratio;
      }

      // Update P
      DW_DTYPE div_term = 0;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict__ const buoyancy_z_shot =
          b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
      div_term += (pml_z ? -DIFFZ1(P_ADJOINT_Z_PML) : -DIFFZ1(P_ADJOINT_Z));
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict__ const buoyancy_y_shot =
          b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      div_term += (pml_y ? -DIFFY1(P_ADJOINT_Y_PML) : -DIFFY1(P_ADJOINT_Y));
#endif
      DW_DTYPE const *__restrict__ const buoyancy_x_shot =
          b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      div_term += (pml_x ? -DIFFX1(P_ADJOINT_X_PML) : -DIFFX1(P_ADJOINT_X));

      p[i] += div_term;

#if DW_NDIM == 3
    }
#endif
  }
}

int set_config(
#if DW_NDIM >= 3
    /* z-dimension */
    DW_DTYPE const rdz_h, int64_t const nz_h, int64_t const pml_z0_h,
    int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
    /* y-dimension */
    DW_DTYPE const rdy_h, int64_t const ny_h, int64_t const pml_y0_h,
    int64_t const pml_y1_h,
#endif
    /* x-dimension */
    DW_DTYPE const rdx_h, int64_t const nx_h, int64_t const shot_numel_h,
    int64_t const pml_x0_h, int64_t const pml_x1_h,
    /* other */
    DW_DTYPE const dt_h, int64_t const n_shots_h, int64_t const step_ratio_h,
    bool const k_batched_h, bool const b_batched_h) {
  gpuErrchk(cudaMemcpyToSymbol(dt, &dt_h, sizeof(DW_DTYPE)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(k_batched, &k_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(b_batched, &b_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        int FUNC(forward)(
            DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
            DW_DTYPE const *__restrict__ const buoyancy_x,
            DW_DTYPE const *__restrict__ const f_p,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict__ const f_vz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict__ const f_vy,
#endif
            DW_DTYPE const *__restrict__ const f_vx,
            DW_DTYPE *__restrict__ const p,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const vy,
#endif
            DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const phi_y,
#endif
            DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const psi_y,
#endif
            DW_DTYPE *__restrict__ const psi_x,
            DW_DTYPE *__restrict__ const k_grad_store_1a,
            DW_DTYPE *__restrict__ const k_grad_store_1b,
            void *__restrict__ const k_grad_store_2,
            void *__restrict__ const k_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const k_grad_filenames_ptr,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const bz_grad_store_1a,
            DW_DTYPE *__restrict__ const bz_grad_store_1b,
            void *__restrict__ const bz_grad_store_2,
            void *__restrict__ const bz_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const by_grad_store_1a,
            DW_DTYPE *__restrict__ const by_grad_store_1b,
            void *__restrict__ const by_grad_store_2,
            void *__restrict__ const by_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const by_grad_filenames_ptr,
#endif
            DW_DTYPE *__restrict__ const bx_grad_store_1a,
            DW_DTYPE *__restrict__ const bx_grad_store_1b,
            void *__restrict__ const bx_grad_store_2,
            void *__restrict__ const bx_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const bx_grad_filenames_ptr,
            DW_DTYPE *__restrict__ const r_p,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const r_vz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const r_vy,
#endif
            DW_DTYPE *__restrict__ const r_vx,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict__ const az,
            DW_DTYPE const *__restrict__ const bz,
            DW_DTYPE const *__restrict__ const azh,
            DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict__ const ay,
            DW_DTYPE const *__restrict__ const by,
            DW_DTYPE const *__restrict__ const ayh,
            DW_DTYPE const *__restrict__ const byh,
#endif
            DW_DTYPE const *__restrict__ const ax,
            DW_DTYPE const *__restrict__ const bx,
            DW_DTYPE const *__restrict__ const axh,
            DW_DTYPE const *__restrict__ const bxh,
            int64_t const *__restrict__ const sources_i_p,
#if DW_NDIM >= 3
            int64_t const *__restrict__ const sources_i_vz,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict__ const sources_i_vy,
#endif
            int64_t const *__restrict__ const sources_i_vx,
            int64_t const *__restrict__ const receivers_i_p,
#if DW_NDIM >= 3
            int64_t const *__restrict__ const receivers_i_vz,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict__ const receivers_i_vy,
#endif
            int64_t const *__restrict__ const receivers_i_vx,
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
            int64_t const nx_h, int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
            int64_t const n_sources_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_sources_vy_per_shot_h,
#endif
            int64_t const n_sources_vx_per_shot_h,
            int64_t const n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
            int64_t const n_receivers_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_receivers_vy_per_shot_h,
#endif
            int64_t const n_receivers_vx_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const k_requires_grad,
            bool const b_requires_grad, bool const k_batched_h,
            bool const b_batched_h, bool const storage_compression,
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

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, dt_h, n_shots_h,
        step_ratio_h, k_batched_h, b_batched_h);
    if (err != 0) return err;
  }

#define OPEN_FILE_WRITE(name, grad_cond)             \
  ScopedFile fp_##name;                              \
  if (storage_mode == STORAGE_DISK && (grad_cond)) { \
    fp_##name.open(name##_filenames_ptr[0], "ab");   \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (k_requires_grad || b_requires_grad);

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

  OPEN_FILE_WRITE(k_grad, k_requires_grad)
#if DW_NDIM >= 3
  OPEN_FILE_WRITE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_WRITE(by_grad, b_requires_grad)
#endif
  OPEN_FILE_WRITE(bx_grad, b_requires_grad)

  for (t = start_t; t < start_t + nt; ++t) {
    int64_t const step_idx = t / step_ratio_h;
#define SETUP_STORE_SAVE(name, grad_cond)                                    \
  DW_DTYPE *__restrict name##_store_1_t =                                    \
      name##_store_1a +                                                      \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)              \
           ? (t / step_ratio_h) * n_shots_h * shot_numel_h                   \
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

    SETUP_STORE_SAVE(k_grad, k_requires_grad)
#if DW_NDIM >= 3
    SETUP_STORE_SAVE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_SAVE(by_grad, b_requires_grad)
#endif
    SETUP_STORE_SAVE(bx_grad, b_requires_grad)

    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_p + t * n_shots_h * n_receivers_p_per_shot_h, p, receivers_i_p,
          n_receivers_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    if (n_receivers_vz_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_vz + t * n_shots_h * n_receivers_vz_per_shot_h, vz, receivers_i_vz,
          n_receivers_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_receivers_vy_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_vy + t * n_shots_h * n_receivers_vy_per_shot_h, vy, receivers_i_vy,
          n_receivers_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_receivers_vx_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_vx + t * n_shots_h * n_receivers_vx_per_shot_h, vx, receivers_i_vx,
          n_receivers_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        bz_grad_store_1_t,
#endif
#if DW_NDIM >= 2
        by_grad_store_1_t,
#endif
        bx_grad_store_1_t,
#if DW_NDIM >= 3
        azh, bzh,
#endif
#if DW_NDIM >= 2
        ayh, byh,
#endif
        axh, bxh, b_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

#if DW_NDIM >= 3
    if (n_sources_vz_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vz, f_vz + t * n_shots_h * n_sources_vz_per_shot_h, sources_i_vz,
          n_sources_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_sources_vy_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vy, f_vy + t * n_shots_h * n_sources_vy_per_shot_h, sources_i_vy,
          n_sources_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_sources_vx_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vx, f_vx + t * n_shots_h * n_sources_vx_per_shot_h, sources_i_vx,
          n_sources_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x, k, k_grad_store_1_t,
#if DW_NDIM >= 3
        az, bz,
#endif
#if DW_NDIM >= 2
        ay, by,
#endif
        ax, bx, k_grad_cond);
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

    SAVE_SNAPSHOT(k_grad)
#if DW_NDIM >= 3
    SAVE_SNAPSHOT(bz_grad)
#endif
#if DW_NDIM >= 2
    SAVE_SNAPSHOT(by_grad)
#endif
    SAVE_SNAPSHOT(bx_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          p, f_p + t * n_shots_h * n_sources_p_per_shot_h, sources_i_p,
          n_sources_p_per_shot_h);
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
            DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
            DW_DTYPE const *__restrict__ const buoyancy_x,
            DW_DTYPE const *__restrict__ const grad_r_p,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict__ const grad_r_vz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict__ const grad_r_vy,
#endif
            DW_DTYPE const *__restrict__ const grad_r_vx,
            DW_DTYPE *__restrict__ const p,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const vy,
#endif
            DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const phi_y,
#endif
            DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ psi_z,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ psi_y,
#endif
            DW_DTYPE *__restrict__ psi_x,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ psi_zn,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ psi_yn,
#endif
            DW_DTYPE *__restrict__ psi_xn,

            DW_DTYPE *__restrict__ const k_grad_store_1a,
            DW_DTYPE *__restrict__ const k_grad_store_1b,
            void *__restrict__ const k_grad_store_2,
            void *__restrict__ const k_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const k_grad_filenames_ptr,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const bz_grad_store_1a,
            DW_DTYPE *__restrict__ const bz_grad_store_1b,
            void *__restrict__ const bz_grad_store_2,
            void *__restrict__ const bz_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const by_grad_store_1a,
            DW_DTYPE *__restrict__ const by_grad_store_1b,
            void *__restrict__ const by_grad_store_2,
            void *__restrict__ const by_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const by_grad_filenames_ptr,
#endif
            DW_DTYPE *__restrict__ const bx_grad_store_1a,
            DW_DTYPE *__restrict__ const bx_grad_store_1b,
            void *__restrict__ const bx_grad_store_2,
            void *__restrict__ const bx_grad_store_3,
            char const *__restrict__ const
                *__restrict__ const bx_grad_filenames_ptr,
            DW_DTYPE *__restrict__ const grad_f_p,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const grad_f_vz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const grad_f_vy,
#endif
            DW_DTYPE *__restrict__ const grad_f_vx,
            DW_DTYPE *__restrict__ const grad_k,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const grad_bz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const grad_by,
#endif
            DW_DTYPE *__restrict__ const grad_bx,
            DW_DTYPE *__restrict__ const grad_k_shot,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict__ const grad_bz_shot,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict__ const grad_by_shot,
#endif
            DW_DTYPE *__restrict__ const grad_bx_shot,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict__ const az,
            DW_DTYPE const *__restrict__ const bz,
            DW_DTYPE const *__restrict__ const azh,
            DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict__ const ay,
            DW_DTYPE const *__restrict__ const by,
            DW_DTYPE const *__restrict__ const ayh,
            DW_DTYPE const *__restrict__ const byh,
#endif
            DW_DTYPE const *__restrict__ const ax,
            DW_DTYPE const *__restrict__ const bx,
            DW_DTYPE const *__restrict__ const axh,
            DW_DTYPE const *__restrict__ const bxh,
            int64_t const *__restrict__ const sources_i_p,
#if DW_NDIM >= 3
            int64_t const *__restrict__ const sources_i_vz,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict__ const sources_i_vy,
#endif
            int64_t const *__restrict__ const sources_i_vx,
            int64_t const *__restrict__ const receivers_i_p,
#if DW_NDIM >= 3
            int64_t const *__restrict__ const receivers_i_vz,
#endif
#if DW_NDIM >= 2
            int64_t const *__restrict__ const receivers_i_vy,
#endif
            int64_t const *__restrict__ const receivers_i_vx,
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
            int64_t const nx_h, int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
            int64_t const n_sources_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_sources_vy_per_shot_h,
#endif
            int64_t const n_sources_vx_per_shot_h,
            int64_t const n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
            int64_t const n_receivers_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
            int64_t const n_receivers_vy_per_shot_h,
#endif
            int64_t const n_receivers_vx_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const k_requires_grad,
            bool const b_requires_grad, bool const k_batched_h,
            bool const b_batched_h, bool const storage_compression,
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

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

  gpuErrchk(cudaSetDevice(device));
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, dt_h, n_shots_h,
        step_ratio_h, k_batched_h, b_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (k_requires_grad || b_requires_grad);

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

#define OPEN_FILE_READ(name, grad_cond)              \
  ScopedFile fp_##name;                              \
  if (storage_mode == STORAGE_DISK && (grad_cond)) { \
    fp_##name.open(name##_filenames_ptr[0], "rb");   \
  }

  OPEN_FILE_READ(k_grad, k_requires_grad)
#if DW_NDIM >= 3
  OPEN_FILE_READ(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(by_grad, b_requires_grad)
#endif
  OPEN_FILE_READ(bx_grad, b_requires_grad)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio_h;
#define SETUP_STORE_LOAD(name, grad_cond)                                    \
  DW_DTYPE *__restrict__ name##_store_1_t =                                  \
      name##_store_1a +                                                      \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)              \
           ? (t / step_ratio_h) * n_shots_h * shot_numel_h                   \
           : 0);                                                             \
  void *__restrict__ const name##_store_2_t =                                \
      (uint8_t *)name##_store_2 +                                            \
      ((storage_mode == STORAGE_DEVICE && storage_compression)               \
           ? (t / step_ratio_h) * n_shots_h * shot_bytes_comp                \
           : 0);                                                             \
  void *__restrict__ const name##_store_3_t =                                \
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

    SETUP_STORE_LOAD(k_grad, k_requires_grad)
#if DW_NDIM >= 3
    SETUP_STORE_LOAD(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(by_grad, b_requires_grad)
#endif
    SETUP_STORE_LOAD(bx_grad, b_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_p + t * n_shots_h * n_sources_p_per_shot_h, p, sources_i_p,
          n_sources_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x,
#if DW_NDIM >= 3
        psi_zn,
#endif
#if DW_NDIM >= 2
        psi_yn,
#endif
        psi_xn, k,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        grad_bz_shot, bz_grad_store_1_t, azh, bz,
#endif
#if DW_NDIM >= 2
        grad_by_shot, by_grad_store_1_t, ayh, by,
#endif
        grad_bx_shot, bx_grad_store_1_t, axh, bx,
        b_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

#if DW_NDIM >= 3
    if (n_sources_vz_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_vz + t * n_shots_h * n_sources_vz_per_shot_h, vz, sources_i_vz,
          n_sources_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_sources_vy_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_vy + t * n_shots_h * n_sources_vy_per_shot_h, vy, sources_i_vy,
          n_sources_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_sources_vx_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_vx + t * n_shots_h * n_sources_vx_per_shot_h, vx, sources_i_vx,
          n_sources_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x, k,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x, grad_k_shot, k_grad_store_1_t,
#if DW_NDIM >= 3
        az, bzh,
#endif
#if DW_NDIM >= 2
        ay, byh,
#endif
        ax, bxh, k_grad_cond);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Inject adjoint source (grad_r) into P
    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          p, grad_r_p + t * n_shots_h * n_receivers_p_per_shot_h, receivers_i_p,
          n_receivers_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    if (n_receivers_vz_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          vz, grad_r_vz + t * n_shots_h * n_receivers_vz_per_shot_h,
          receivers_i_vz, n_receivers_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_receivers_vy_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          vy, grad_r_vy + t * n_shots_h * n_receivers_vy_per_shot_h,
          receivers_i_vy, n_receivers_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_receivers_vx_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          vx, grad_r_vx + t * n_shots_h * n_receivers_vx_per_shot_h,
          receivers_i_vx, n_receivers_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM >= 3
    std::swap(psi_z, psi_zn);
#endif
#if DW_NDIM >= 2
    std::swap(psi_y, psi_yn);
#endif
    std::swap(psi_x, psi_xn);
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

  if (k_requires_grad && !k_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_k, grad_k_shot);
    CHECK_KERNEL_ERROR
  }
#if DW_NDIM >= 3
  if (b_requires_grad && !b_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_bz, grad_bz_shot);
    CHECK_KERNEL_ERROR
  }
#endif
#if DW_NDIM >= 2
  if (b_requires_grad && !b_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_by, grad_by_shot);
    CHECK_KERNEL_ERROR
  }
#endif
  if (b_requires_grad && !b_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_bx, grad_bx_shot);
    CHECK_KERNEL_ERROR
  }

  return 0;
}

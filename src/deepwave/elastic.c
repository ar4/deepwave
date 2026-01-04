/*
 * Elastic wave equation propagator
 */

/*
 * This file contains the C implementation of the elastic wave equation
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
 * The propagator solves the elastic wave equation using a velocity-stress
 * formulation on a staggered grid. The free surface boundary condition is
 * implemented using the improved vacuum method of Zeng et al. (2012), and the
 * PML implementation is based on the C-PML method of Komatitsch and
 * Martin (2007).
 *
 * The code is structured to maximize performance by using macros to
 * generate the code for each of the regions of the computational
 * domain (a central region, side regions, edge regions, and corner regions),
 * and by using OpenMP to parallelize the loops over shots.
 *
 * The staggered grid locations of the different wavefield components depends
 * on the number of dimensions. In 3D, the velocity components (vx, vy, vz)
 * are stored at half-integer locations in their respective dimension and
 * integer locations in the other dimensions. The normal stress components
 * (sigmaxx, sigmayy, sigmazz) are stored at integer locations. The shear
 * stress components (sigmaxy, sigmaxz, sigmayz) are stored at half-integer
 * locations in their two respective dimensions and integer locations in the
 * other. The model parameters (lambda, mu, and buoyancy) are at the same
 * locations as the normal stresses. The other dimensions have analogous
 * staggered grid locations.
 *
 * o--------->x
 * | SII=VX==|=SII=VX==|=SII vx
 * | VY  SXY | VY  SXY | VY  sxy
 * v [-------------------]------
 * y SII VX  | SII VX  | SII vx
 *   VY  SXY | VY  SXY | VY  sxy
 *   [-------------------]------
 *   SII=VX==|=SII=VX==|=SII vx
 *   vy  sxy | vy  sxy | vy  sxy
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "common_cpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  elastic_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz)*ny * nx + (dy)*nx + (dx))
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy)*nx + (dx))
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Access the wavefield at offset (dz, dy, dx) from the current index i
// Note: arrays are row-major with flattened index i. For batched
// shots the base pointer is shot_offset = shot * num_grid_pts, so callers pass
// pointers already offset into the shot's memory region.
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx_shot[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define SIGMAZZ(dz, dy, dx) sigmazz_shot[ND_INDEX(i, dz, dy, dx)]
#elif DW_NDIM == 2
#define SIGMAZZ(dz, dy, dx) 0
#endif
#if DW_NDIM >= 2
#define SIGMAYY(dz, dy, dx) sigmayy_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define SIGMAXX(dz, dy, dx) sigmaxx_shot[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define SIGMAXY(dz, dy, dx) sigmaxy_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define SIGMAXZ(dz, dy, dx) sigmaxz_shot[ND_INDEX(i, dz, dy, dx)]
#define SIGMAYZ(dz, dy, dx) sigmayz_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define LAMB(dz, dy, dx) lamb_shot[ND_INDEX(i, dz, dy, dx)]
#define MU(dz, dy, dx) mu_shot[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define MU_YX(dz, dy, dx) mu_yx_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define MU_ZX(dz, dy, dx) mu_zx_shot[ND_INDEX(i, dz, dy, dx)]
#define MU_ZY(dz, dy, dx) mu_zy_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(i, dz, dy, dx)]

// PML memory variables for velocity equations
#if DW_NDIM >= 2
#define M_SIGMAXYY(dz, dy, dx) m_sigmaxyy_t[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_SIGMAXXX(dz, dy, dx) m_sigmaxxx_t[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define M_SIGMAXZZ(dz, dy, dx) m_sigmaxzz_t[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 2
#define M_SIGMAXYX(dz, dy, dx) m_sigmaxyx_t[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYYY(dz, dy, dx) m_sigmayyy_t[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_SIGMAYZZ(dz, dy, dx) m_sigmayzz_t[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 3
#define M_SIGMAXZX(dz, dy, dx) m_sigmaxzx_t[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYZY(dz, dy, dx) m_sigmayzy_t[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAZZZ(dz, dy, dx) m_sigmazzz_t[ND_INDEX(i, dz, dy, dx)]
#endif

// PML memory variables for stress equations
#if DW_NDIM >= 2
#define M_VXY(dz, dy, dx) m_vxy_shot[ND_INDEX(i, dz, dy, dx)]
#define M_VYX(dz, dy, dx) m_vyx_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_VXX(dz, dy, dx) m_vxx_shot[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define M_VYY(dz, dy, dx) m_vyy_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_VZZ(dz, dy, dx) m_vzz_shot[ND_INDEX(i, dz, dy, dx)]
#define M_VXZ(dz, dy, dx) m_vxz_shot[ND_INDEX(i, dz, dy, dx)]
#define M_VZX(dz, dy, dx) m_vzx_shot[ND_INDEX(i, dz, dy, dx)]
#define M_VYZ(dz, dy, dx) m_vyz_shot[ND_INDEX(i, dz, dy, dx)]
#define M_VZY(dz, dy, dx) m_vzy_shot[ND_INDEX(i, dz, dy, dx)]
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

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

static inline void add_pressure(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy,
#endif
    DW_DTYPE *__restrict const sigmaxx,
    int64_t const *__restrict const sources_i,
    DW_DTYPE const *__restrict const f, int64_t const n_sources_per_shot) {
  int64_t source_idx;
#pragma omp simd
  for (source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
    if (sources_i[source_idx] < 0) continue;
#if DW_NDIM >= 3
    sigmazz[sources_i[source_idx]] += f[source_idx];
#endif
#if DW_NDIM >= 2
    sigmayy[sources_i[source_idx]] += f[source_idx];
#endif
    sigmaxx[sources_i[source_idx]] += f[source_idx];
  }
}

static inline void record_pressure(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
    int64_t const *__restrict const locations,
    DW_DTYPE *__restrict const amplitudes, int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i])
#if DW_NDIM == 3
      amplitudes[i] = (sigmazz[locations[i]] + sigmayy[locations[i]] +
                       sigmaxx[locations[i]]);
#elif DW_NDIM == 2
      amplitudes[i] = (sigmayy[locations[i]] + sigmaxx[locations[i]]);
#else
      amplitudes[i] = sigmaxx[locations[i]];
#endif
  }
}

static inline void combine_grad_elastic(
    DW_DTYPE *__restrict const grad,
    DW_DTYPE const *__restrict const grad_thread, int64_t const n_threads,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx) {
#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
#else
  int64_t const n_grid_points = nx;
#endif
#if DW_NDIM >= 3
  int64_t z;
#endif
#if DW_NDIM >= 2
  int64_t y;
#endif
  int64_t x;
#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
  for (z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
#endif
#if DW_NDIM >= 2
    for (y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
#endif
      for (x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
#if DW_NDIM == 3
        int64_t const i = z * ny * nx + y * nx + x;
#elif DW_NDIM == 2
    int64_t const i = y * nx + x;
#else
    int64_t const i = x;
#endif
        int64_t threadidx;
        for (threadidx = 0; threadidx < n_threads; ++threadidx) {
          grad[i] += grad_thread[threadidx * n_grid_points + i];
        }
      }
#if DW_NDIM >= 3
    }
#endif
#if DW_NDIM >= 2
  }
#endif
}

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
        DW_DTYPE *__restrict const sigmayz, DW_DTYPE *__restrict const sigmaxz,
        DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
        DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
        DW_DTYPE *__restrict const m_vxz, DW_DTYPE *__restrict const m_sigmazzz,
        DW_DTYPE *__restrict const m_sigmayzy,
        DW_DTYPE *__restrict const m_sigmaxzx,
        DW_DTYPE *__restrict const m_sigmayzz,
        DW_DTYPE *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
        DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const m_vyy,
        DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
        DW_DTYPE *__restrict const m_sigmayyy,
        DW_DTYPE *__restrict const m_sigmaxyy,
        DW_DTYPE *__restrict const m_sigmaxyx,
#endif
        DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
        DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const dvzdbuoyancy_store_1,
        DW_DTYPE *__restrict const dvzdbuoyancy_store_1b,
        void *__restrict const dvzdbuoyancy_store_2,
        void *__restrict const dvzdbuoyancy_store_3,
        char const *__restrict const
            *__restrict const dvzdbuoyancy_filenames_ptr,
        DW_DTYPE *__restrict const dvzdz_store_1,
        DW_DTYPE *__restrict const dvzdz_store_1b,
        void *__restrict const dvzdz_store_2,
        void *__restrict const dvzdz_store_3,
        char const *__restrict const *__restrict const dvzdz_filenames_ptr,
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1,
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1b,
        void *__restrict const dvzdx_plus_dvxdz_store_2,
        void *__restrict const dvzdx_plus_dvxdz_store_3,
        char const *__restrict const
            *__restrict const dvzdx_plus_dvxdz_filenames_ptr,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1b,
        void *__restrict const dvzdy_plus_dvydz_store_2,
        void *__restrict const dvzdy_plus_dvydz_store_3,
        char const *__restrict const
            *__restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const dvydbuoyancy_store_1,
        DW_DTYPE *__restrict const dvydbuoyancy_store_1b,
        void *__restrict const dvydbuoyancy_store_2,
        void *__restrict const dvydbuoyancy_store_3,
        char const *__restrict const
            *__restrict const dvydbuoyancy_filenames_ptr,
        DW_DTYPE *__restrict const dvydy_store_1,
        DW_DTYPE *__restrict const dvydy_store_1b,
        void *__restrict const dvydy_store_2,
        void *__restrict const dvydy_store_3,
        char const *__restrict const *__restrict const dvydy_filenames_ptr,
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1,
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1b,
        void *__restrict const dvydx_plus_dvxdy_store_2,
        void *__restrict const dvydx_plus_dvxdy_store_3,
        char const *__restrict const
            *__restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
        DW_DTYPE *__restrict const dvxdbuoyancy_store_1,
        DW_DTYPE *__restrict const dvxdbuoyancy_store_1b,
        void *__restrict const dvxdbuoyancy_store_2,
        void *__restrict const dvxdbuoyancy_store_3,
        char const *__restrict const
            *__restrict const dvxdbuoyancy_filenames_ptr,
        DW_DTYPE *__restrict const dvxdx_store_1,
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
        DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy,
#endif
        DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
        int64_t const n_shots,
#if DW_NDIM >= 3
        int64_t const nz,
#endif
#if DW_NDIM >= 2
        int64_t const ny,
#endif
        int64_t const nx,
#if DW_NDIM >= 3
        int64_t const n_sources_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_per_shot,
#endif
        int64_t const n_sources_x_per_shot, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_per_shot,
#endif
        int64_t n_receivers_x_per_shot, int64_t const n_receivers_p_per_shot,
        int64_t const step_ratio, int64_t const storage_mode,
        size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
        bool const lamb_requires_grad, bool const mu_requires_grad,
        bool const buoyancy_requires_grad, bool const lamb_batched,
        bool const mu_batched, bool const buoyancy_batched,
        bool const storage_compression, int64_t start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y0,
#endif
        int64_t const pml_x0,
#if DW_NDIM >= 3
        int64_t const pml_z1,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y1,
#endif
        int64_t const pml_x1, int64_t const n_threads, void *unused) {
#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
#if DW_NDIM >= 2
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
#endif
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
    int64_t const n_grid_points = ny * nx;
#else
    int64_t const n_grid_points = nx;
#endif
    int64_t const si = shot * n_grid_points;
#if DW_NDIM >= 3
    int64_t const siz = shot * n_sources_z_per_shot;
    int64_t const riz = shot * n_receivers_z_per_shot;
    int64_t const *__restrict const sources_z_i_shot = sources_z_i + siz;
    int64_t const *__restrict const receivers_z_i_shot = receivers_z_i + riz;
    DW_DTYPE const *__restrict const f_z_shot = f_z + siz;
    DW_DTYPE *__restrict const r_z_shot = r_z + riz;
    DW_DTYPE const *__restrict const mu_zy_shot =
        mu_batched ? mu_zy + si : mu_zy;
    DW_DTYPE const *__restrict const mu_zx_shot =
        mu_batched ? mu_zx + si : mu_zx;
    DW_DTYPE const *__restrict const buoyancy_z_shot =
        buoyancy_batched ? buoyancy_z + si : buoyancy_z;
    DW_DTYPE *__restrict const vz_shot = vz + si;
    DW_DTYPE *__restrict const sigmazz_shot = sigmazz + si;
    DW_DTYPE *__restrict const sigmayz_shot = sigmayz + si;
    DW_DTYPE *__restrict const sigmaxz_shot = sigmaxz + si;
    DW_DTYPE *__restrict const m_vzz_shot = m_vzz + si;
    DW_DTYPE *__restrict const m_vzy_shot = m_vzy + si;
    DW_DTYPE *__restrict const m_vzx_shot = m_vzx + si;
    DW_DTYPE *__restrict const m_vyz_shot = m_vyz + si;
    DW_DTYPE *__restrict const m_vxz_shot = m_vxz + si;
    DW_DTYPE *__restrict const m_sigmazzz_shot = m_sigmazzz + si;
    DW_DTYPE *__restrict const m_sigmayzy_shot = m_sigmayzy + si;
    DW_DTYPE *__restrict const m_sigmaxzx_shot = m_sigmaxzx + si;
    DW_DTYPE *__restrict const m_sigmayzz_shot = m_sigmayzz + si;
    DW_DTYPE *__restrict const m_sigmaxzz_shot = m_sigmaxzz + si;
#endif
#if DW_NDIM >= 2
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const *__restrict const sources_y_i_shot = sources_y_i + siy;
    int64_t const *__restrict const receivers_y_i_shot = receivers_y_i + riy;
    DW_DTYPE const *__restrict const f_y_shot = f_y + siy;
    DW_DTYPE *__restrict const r_y_shot = r_y + riy;
    DW_DTYPE const *__restrict const mu_yx_shot =
        mu_batched ? mu_yx + si : mu_yx;
    DW_DTYPE const *__restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE *__restrict const vy_shot = vy + si;
    DW_DTYPE *__restrict const sigmayy_shot = sigmayy + si;
    DW_DTYPE *__restrict const sigmaxy_shot = sigmaxy + si;
    DW_DTYPE *__restrict const m_vyy_shot = m_vyy + si;
    DW_DTYPE *__restrict const m_vyx_shot = m_vyx + si;
    DW_DTYPE *__restrict const m_vxy_shot = m_vxy + si;
    DW_DTYPE *__restrict const m_sigmayyy_shot = m_sigmayyy + si;
    DW_DTYPE *__restrict const m_sigmaxyy_shot = m_sigmaxyy + si;
    DW_DTYPE *__restrict const m_sigmaxyx_shot = m_sigmaxyx + si;
#endif
    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const sip = shot * n_sources_p_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const rip = shot * n_receivers_p_per_shot;
    int64_t const *__restrict const sources_x_i_shot = sources_x_i + six;
    int64_t const *__restrict const sources_p_i_shot = sources_p_i + sip;
    int64_t const *__restrict const receivers_x_i_shot = receivers_x_i + rix;
    int64_t const *__restrict const receivers_p_i_shot = receivers_p_i + rip;
    DW_DTYPE const *__restrict const f_x_shot = f_x + six;
    DW_DTYPE const *__restrict const f_p_shot = f_p + sip;
    DW_DTYPE *__restrict const r_x_shot = r_x + rix;
    DW_DTYPE *__restrict const r_p_shot = r_p + rip;
    DW_DTYPE const *__restrict const lamb_shot =
        lamb_batched ? lamb + si : lamb;
    DW_DTYPE const *__restrict const mu_shot = mu_batched ? mu + si : mu;
    DW_DTYPE const *__restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE *__restrict const vx_shot = vx + si;
    DW_DTYPE *__restrict const sigmaxx_shot = sigmaxx + si;
    DW_DTYPE *__restrict const m_vxx_shot = m_vxx + si;
    DW_DTYPE *__restrict const m_sigmaxxx_shot = m_sigmaxxx + si;
    int64_t t;

#define OPEN_FILE_WRITE(name, grad_cond)                 \
  FILE *fp_##name = NULL;                                \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {     \
    fp_##name = fopen(name##_filenames_ptr[shot], "ab"); \
  }

#if DW_NDIM >= 3
    OPEN_FILE_WRITE(dvzdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvzdz, lamb_requires_grad || mu_requires_grad)
    OPEN_FILE_WRITE(dvzdx_plus_dvxdz, mu_requires_grad)
    OPEN_FILE_WRITE(dvzdy_plus_dvydz, mu_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_WRITE(dvydbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvydy, lamb_requires_grad || mu_requires_grad)
    OPEN_FILE_WRITE(dvydx_plus_dvxdy, mu_requires_grad)
#endif
    OPEN_FILE_WRITE(dvxdbuoyancy, buoyancy_requires_grad)
    OPEN_FILE_WRITE(dvxdx, lamb_requires_grad || mu_requires_grad)

    for (t = start_t; t < start_t + nt; ++t) {
      int64_t const store_1_t =
          (shot + ((storage_mode == STORAGE_DEVICE && !storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
          n_grid_points;
      int64_t const store_2_t =
          (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)
                       ? t / step_ratio * n_shots
                       : 0)) *
          (int64_t)shot_bytes_comp;

#if DW_NDIM >= 3
      DW_DTYPE *__restrict const dvzdbuoyancy_store_1_t =
          dvzdbuoyancy_store_1 + store_1_t;
      void *__restrict const dvzdbuoyancy_store_2_t =
          (uint8_t *)dvzdbuoyancy_store_2 + store_2_t;
      DW_DTYPE *__restrict const dvzdz_store_1_t = dvzdz_store_1 + store_1_t;
      void *__restrict const dvzdz_store_2_t =
          (uint8_t *)dvzdz_store_2 + store_2_t;
      DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1_t =
          dvzdx_plus_dvxdz_store_1 + store_1_t;
      void *__restrict const dvzdx_plus_dvxdz_store_2_t =
          (uint8_t *)dvzdx_plus_dvxdz_store_2 + store_2_t;
      DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1_t =
          dvzdy_plus_dvydz_store_1 + store_1_t;
      void *__restrict const dvzdy_plus_dvydz_store_2_t =
          (uint8_t *)dvzdy_plus_dvydz_store_2 + store_2_t;
#endif
#if DW_NDIM >= 2
      DW_DTYPE *__restrict const dvydbuoyancy_store_1_t =
          dvydbuoyancy_store_1 + store_1_t;
      void *__restrict const dvydbuoyancy_store_2_t =
          (uint8_t *)dvydbuoyancy_store_2 + store_2_t;
      DW_DTYPE *__restrict const dvydy_store_1_t = dvydy_store_1 + store_1_t;
      void *__restrict const dvydy_store_2_t =
          (uint8_t *)dvydy_store_2 + store_2_t;
      DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1_t =
          dvydx_plus_dvxdy_store_1 + store_1_t;
      void *__restrict const dvydx_plus_dvxdy_store_2_t =
          (uint8_t *)dvydx_plus_dvxdy_store_2 + store_2_t;
#endif
      DW_DTYPE *__restrict const dvxdbuoyancy_store_1_t =
          dvxdbuoyancy_store_1 + store_1_t;
      void *__restrict const dvxdbuoyancy_store_2_t =
          (uint8_t *)dvxdbuoyancy_store_2 + store_2_t;
      DW_DTYPE *__restrict const dvxdx_store_1_t = dvxdx_store_1 + store_1_t;
      void *__restrict const dvxdx_store_2_t =
          (uint8_t *)dvxdx_store_2 + store_2_t;
      bool const lamb_requires_grad_t =
          lamb_requires_grad && ((t % step_ratio) == 0);
      bool const mu_requires_grad_t =
          mu_requires_grad && ((t % step_ratio) == 0);
      bool const lamb_or_mu_requires_grad_t =
          lamb_requires_grad_t || mu_requires_grad_t;
      bool const buoyancy_requires_grad_t =
          buoyancy_requires_grad && ((t % step_ratio) == 0);
#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

#if DW_NDIM >= 3
      if (n_receivers_z_per_shot > 0) {
        record_from_wavefield(vz_shot, receivers_z_i_shot,
                              r_z_shot + t * n_shots * n_receivers_z_per_shot,
                              n_receivers_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_y_per_shot > 0) {
        record_from_wavefield(vy_shot, receivers_y_i_shot,
                              r_y_shot + t * n_shots * n_receivers_y_per_shot,
                              n_receivers_y_per_shot);
      }
#endif
      if (n_receivers_x_per_shot > 0) {
        record_from_wavefield(vx_shot, receivers_x_i_shot,
                              r_x_shot + t * n_shots * n_receivers_x_per_shot,
                              n_receivers_x_per_shot);
      }
      if (n_receivers_p_per_shot > 0) {
        record_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, receivers_p_i_shot,
            r_p_shot + t * n_shots * n_receivers_p_per_shot,
            n_receivers_p_per_shot);
      }

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;

#if DW_NDIM >= 3

            // vz

#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  DW_DTYPE w_sum = 0;

                  if (pml_z == 1) {
                    w_sum += DIFFZH1(SIGMAZZ);
                  } else {
                    DW_DTYPE d3 = DIFFZH1(SIGMAZZ);
                    m_sigmazzz_shot[i] =
                        azh[z] * m_sigmazzz_shot[i] + bzh[z] * d3;
                    d3 += m_sigmazzz_shot[i];
                    w_sum += d3;
                  }
                  if (pml_y == 1) {
                    w_sum += DIFFY1(SIGMAYZ);
                  } else {
                    DW_DTYPE d2 = DIFFY1(SIGMAYZ);
                    m_sigmayzy_shot[i] =
                        ay[y] * m_sigmayzy_shot[i] + by[y] * d2;
                    d2 += m_sigmayzy_shot[i];
                    w_sum += d2;
                  }
                  if (pml_x == 1) {
                    w_sum += DIFFX1(SIGMAXZ);
                  } else {
                    DW_DTYPE d1 = DIFFX1(SIGMAXZ);
                    m_sigmaxzx_shot[i] =
                        ax[x] * m_sigmaxzx_shot[i] + bx[x] * d1;
                    d1 += m_sigmaxzx_shot[i];
                    w_sum += d1;
                  }

                  if (buoyancy_requires_grad_t) {
                    dvzdbuoyancy_store_1_t[i] = dt * w_sum;
                  }
                  vz_shot[i] += buoyancy_z_shot[i] * dt * w_sum;
                }
#endif

#if DW_NDIM >= 2

                // vy

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE w_sum = 0;

#if DW_NDIM == 3
                  if (pml_z == 1) {
                    w_sum += DIFFZ1(SIGMAYZ);
                  } else {
                    DW_DTYPE d3 = DIFFZ1(SIGMAYZ);
                    m_sigmayzz_shot[i] =
                        az[z] * m_sigmayzz_shot[i] + bz[z] * d3;
                    d3 += m_sigmayzz_shot[i];
                    w_sum += d3;
                  }
#endif
                  if (pml_y == 1) {
                    w_sum += DIFFYH1(SIGMAYY);
                  } else {
                    DW_DTYPE d2 = DIFFYH1(SIGMAYY);
                    m_sigmayyy_shot[i] =
                        ayh[y] * m_sigmayyy_shot[i] + byh[y] * d2;
                    d2 += m_sigmayyy_shot[i];
                    w_sum += d2;
                  }
                  if (pml_x == 1) {
                    w_sum += DIFFX1(SIGMAXY);
                  } else {
                    DW_DTYPE d1 = DIFFX1(SIGMAXY);
                    m_sigmaxyx_shot[i] =
                        ax[x] * m_sigmaxyx_shot[i] + bx[x] * d1;
                    d1 += m_sigmaxyx_shot[i];
                    w_sum += d1;
                  }

                  if (buoyancy_requires_grad_t) {
                    dvydbuoyancy_store_1_t[i] = dt * w_sum;
                  }
                  vy_shot[i] += buoyancy_y_shot[i] * dt * w_sum;
                }
#endif

                // vx

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE w_sum = 0;

#if DW_NDIM == 3
                  if (pml_z == 1) {
                    w_sum += DIFFZ1(SIGMAXZ);
                  } else {
                    DW_DTYPE d3 = DIFFZ1(SIGMAXZ);
                    m_sigmaxzz_shot[i] =
                        az[z] * m_sigmaxzz_shot[i] + bz[z] * d3;
                    d3 += m_sigmaxzz_shot[i];
                    w_sum += d3;
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y == 1) {
                    w_sum += DIFFY1(SIGMAXY);
                  } else {
                    DW_DTYPE d2 = DIFFY1(SIGMAXY);
                    m_sigmaxyy_shot[i] =
                        ay[y] * m_sigmaxyy_shot[i] + by[y] * d2;
                    d2 += m_sigmaxyy_shot[i];
                    w_sum += d2;
                  }
#endif
                  if (pml_x == 1) {
                    w_sum += DIFFXH1(SIGMAXX);
                  } else {
                    DW_DTYPE d1 = DIFFXH1(SIGMAXX);
                    m_sigmaxxx_shot[i] =
                        axh[x] * m_sigmaxxx_shot[i] + bxh[x] * d1;
                    d1 += m_sigmaxxx_shot[i];
                    w_sum += d1;
                  }

                  if (buoyancy_requires_grad_t) {
                    dvxdbuoyancy_store_1_t[i] = dt * w_sum;
                  }
                  vx_shot[i] += buoyancy_x_shot[i] * dt * w_sum;
                }
          }

#if DW_NDIM >= 3
      if (n_sources_z_per_shot > 0) {
        add_to_wavefield(vz_shot, sources_z_i_shot,
                         f_z_shot + t * n_shots * n_sources_z_per_shot,
                         n_sources_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_y_per_shot > 0) {
        add_to_wavefield(vy_shot, sources_y_i_shot,
                         f_y_shot + t * n_shots * n_sources_y_per_shot,
                         n_sources_y_per_shot);
      }
#endif
      if (n_sources_x_per_shot > 0) {
        add_to_wavefield(vx_shot, sources_x_i_shot,
                         f_x_shot + t * n_shots * n_sources_x_per_shot,
                         n_sources_x_per_shot);
      }

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;

            // sigmaii

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
                  DW_DTYPE dvzdz;
                  if (pml_z != 1) {
                    dvzdz = DIFFZ1(VZ);
                    m_vzz_shot[i] = az[z] * m_vzz_shot[i] + bz[z] * dvzdz;
                    dvzdz += m_vzz_shot[i];
                  } else {
                    dvzdz = DIFFZ1(VZ);
                  }
                  w_sum += lamb_shot[i] * dvzdz;
                  if (lamb_or_mu_requires_grad_t) {
                    dvzdz_store_1_t[i] = dt * dvzdz;
                  }
#endif

#if DW_NDIM >= 2
                  DW_DTYPE dvydy;
                  if (pml_y != 1) {
                    dvydy = DIFFY1(VY);
                    m_vyy_shot[i] = ay[y] * m_vyy_shot[i] + by[y] * dvydy;
                    dvydy += m_vyy_shot[i];
                  } else {
                    dvydy = DIFFY1(VY);
                  }
                  w_sum += lamb_shot[i] * dvydy;
                  if (lamb_or_mu_requires_grad_t) {
                    dvydy_store_1_t[i] = dt * dvydy;
                  }
#endif

                  DW_DTYPE dvxdx;
                  if (pml_x != 1) {
                    dvxdx = DIFFX1(VX);
                    m_vxx_shot[i] = ax[x] * m_vxx_shot[i] + bx[x] * dvxdx;
                    dvxdx += m_vxx_shot[i];
                  } else {
                    dvxdx = DIFFX1(VX);
                  }
                  w_sum += lamb_shot[i] * dvxdx;
                  if (lamb_or_mu_requires_grad_t) {
                    dvxdx_store_1_t[i] = dt * dvxdx;
                  }

#if DW_NDIM >= 3
                  sigmazz_shot[i] += dt * (w_sum + 2 * mu_shot[i] * dvzdz);
#endif
#if DW_NDIM >= 2
                  sigmayy_shot[i] += dt * (w_sum + 2 * mu_shot[i] * dvydy);
#endif
                  sigmaxx_shot[i] += dt * (w_sum + 2 * mu_shot[i] * dvxdx);
                }

#if DW_NDIM >= 2

                // sigmaxy

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;

                  DW_DTYPE w_sum = 0;
                  DW_DTYPE dvxdy, dvydx;

                  if (pml_y != 1) {
                    dvxdy = DIFFYH1(VX);
                    m_vxy_shot[i] = ayh[y] * m_vxy_shot[i] + byh[y] * dvxdy;
                    dvxdy += m_vxy_shot[i];
                  } else {
                    dvxdy = DIFFYH1(VX);
                  }
                  w_sum += dvxdy;

                  if (pml_x != 1) {
                    dvydx = DIFFXH1(VY);
                    m_vyx_shot[i] = axh[x] * m_vyx_shot[i] + bxh[x] * dvydx;
                    dvydx += m_vyx_shot[i];
                  } else {
                    dvydx = DIFFXH1(VY);
                  }
                  w_sum += dvydx;

                  if (mu_requires_grad_t) {
                    dvydx_plus_dvxdy_store_1_t[i] = dt * w_sum;
                  }
                  sigmaxy_shot[i] += dt * mu_yx_shot[i] * w_sum;
                }
#endif

#if DW_NDIM >= 3

                // sigmaxz

#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;

                  DW_DTYPE w_sum = 0;
                  DW_DTYPE dvxdz, dvzdx;

                  if (pml_z != 1) {
                    dvxdz = DIFFZH1(VX);
                    m_vxz_shot[i] = azh[z] * m_vxz_shot[i] + bzh[z] * dvxdz;
                    dvxdz += m_vxz_shot[i];
                  } else {
                    dvxdz = DIFFZH1(VX);
                  }
                  w_sum += dvxdz;

                  if (pml_x != 1) {
                    dvzdx = DIFFXH1(VZ);
                    m_vzx_shot[i] = axh[x] * m_vzx_shot[i] + bxh[x] * dvzdx;
                    dvzdx += m_vzx_shot[i];
                  } else {
                    dvzdx = DIFFXH1(VZ);
                  }
                  w_sum += dvzdx;

                  if (mu_requires_grad_t) {
                    dvzdx_plus_dvxdz_store_1_t[i] = dt * w_sum;
                  }
                  sigmaxz_shot[i] += dt * mu_zx_shot[i] * w_sum;
                }
#endif

#if DW_NDIM >= 3

                // sigmayz

#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE w_sum = 0;
                  DW_DTYPE dvydz, dvzdy;

                  if (pml_z != 1) {
                    dvydz = DIFFZH1(VY);
                    m_vyz_shot[i] = azh[z] * m_vyz_shot[i] + bzh[z] * dvydz;
                    dvydz += m_vyz_shot[i];
                  } else {
                    dvydz = DIFFZH1(VY);
                  }
                  w_sum += dvydz;

                  if (pml_y != 1) {
                    dvzdy = DIFFYH1(VZ);
                    m_vzy_shot[i] = ayh[y] * m_vzy_shot[i] + byh[y] * dvzdy;
                    dvzdy += m_vzy_shot[i];
                  } else {
                    dvzdy = DIFFYH1(VZ);
                  }
                  w_sum += dvzdy;

                  if (mu_requires_grad_t) {
                    dvzdy_plus_dvydz_store_1_t[i] = dt * w_sum;
                  }
                  sigmayz_shot[i] += dt * mu_zy_shot[i] * w_sum;
                }
#endif
          }

      if (n_sources_p_per_shot > 0) {
        add_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, sources_p_i_shot,
            f_p_shot + t * n_shots * n_sources_p_per_shot,
            n_sources_p_per_shot);
      }

#define SAVE_SNAPSHOT(name, grad_cond)                                     \
  if (grad_cond) {                                                         \
    int64_t const step_idx = t / step_ratio;                               \
    STORAGE_FUNC(save_snapshot_cpu)(                                       \
        name##_store_1_t, name##_store_2_t, fp_##name, storage_mode,       \
        storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp, \
        DIM_ARGS);                                                         \
  }

#if DW_NDIM >= 3
      SAVE_SNAPSHOT(dvzdbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvzdz, lamb_or_mu_requires_grad_t)
      SAVE_SNAPSHOT(dvzdx_plus_dvxdz, mu_requires_grad_t)
      SAVE_SNAPSHOT(dvzdy_plus_dvydz, mu_requires_grad_t)
#endif
#if DW_NDIM >= 2
      SAVE_SNAPSHOT(dvydbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvydy, lamb_or_mu_requires_grad_t)
      SAVE_SNAPSHOT(dvydx_plus_dvxdy, mu_requires_grad_t)
#endif
      SAVE_SNAPSHOT(dvxdbuoyancy, buoyancy_requires_grad_t)
      SAVE_SNAPSHOT(dvxdx, lamb_or_mu_requires_grad_t)
    }

#define CLOSE_FILE(name) \
  if (fp_##name) fclose(fp_##name);

#if DW_NDIM >= 3
    CLOSE_FILE(dvzdbuoyancy)
    CLOSE_FILE(dvzdz)
    CLOSE_FILE(dvzdx_plus_dvxdz)
    CLOSE_FILE(dvzdy_plus_dvydz)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(dvydbuoyancy)
    CLOSE_FILE(dvydy)
    CLOSE_FILE(dvydx_plus_dvxdy)
#endif
    CLOSE_FILE(dvxdbuoyancy)
    CLOSE_FILE(dvxdx)
  }
  return 0;
}

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
        DW_DTYPE *__restrict const sigmayz, DW_DTYPE *__restrict const sigmaxz,
        DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
        DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
        DW_DTYPE *__restrict const m_vxz, DW_DTYPE *__restrict const m_sigmazzz,
        DW_DTYPE *__restrict const m_sigmayzy,
        DW_DTYPE *__restrict const m_sigmaxzx,
        DW_DTYPE *__restrict const m_sigmayzz,
        DW_DTYPE *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
        DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const m_vyy,
        DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
        DW_DTYPE *__restrict const m_sigmayyy,
        DW_DTYPE *__restrict const m_sigmaxyy,
        DW_DTYPE *__restrict const m_sigmaxyx,
#endif
        DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
        DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict const m_sigmaxxx,
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
        DW_DTYPE *__restrict const dvzdbuoyancy_store_1,
        DW_DTYPE *__restrict const dvzdbuoyancy_store_1b,
        void *__restrict const dvzdbuoyancy_store_2,
        void *__restrict const dvzdbuoyancy_store_3,
        char const *__restrict const
            *__restrict const dvzdbuoyancy_filenames_ptr,
        DW_DTYPE *__restrict const dvzdz_store_1,
        DW_DTYPE *__restrict const dvzdz_store_1b,
        void *__restrict const dvzdz_store_2,
        void *__restrict const dvzdz_store_3,
        char const *__restrict const *__restrict const dvzdz_filenames_ptr,
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1,
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1b,
        void *__restrict const dvzdx_plus_dvxdz_store_2,
        void *__restrict const dvzdx_plus_dvxdz_store_3,
        char const *__restrict const
            *__restrict const dvzdx_plus_dvxdz_filenames_ptr,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1b,
        void *__restrict const dvzdy_plus_dvydz_store_2,
        void *__restrict const dvzdy_plus_dvydz_store_3,
        char const *__restrict const
            *__restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const dvydbuoyancy_store_1,
        DW_DTYPE *__restrict const dvydbuoyancy_store_1b,
        void *__restrict const dvydbuoyancy_store_2,
        void *__restrict const dvydbuoyancy_store_3,
        char const *__restrict const
            *__restrict const dvydbuoyancy_filenames_ptr,
        DW_DTYPE *__restrict const dvydy_store_1,
        DW_DTYPE *__restrict const dvydy_store_1b,
        void *__restrict const dvydy_store_2,
        void *__restrict const dvydy_store_3,
        char const *__restrict const *__restrict const dvydy_filenames_ptr,
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1,
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1b,
        void *__restrict const dvydx_plus_dvxdy_store_2,
        void *__restrict const dvydx_plus_dvxdy_store_3,
        char const *__restrict const
            *__restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
        DW_DTYPE *__restrict const dvxdbuoyancy_store_1,
        DW_DTYPE *__restrict const dvxdbuoyancy_store_1b,
        void *__restrict const dvxdbuoyancy_store_2,
        void *__restrict const dvxdbuoyancy_store_3,
        char const *__restrict const
            *__restrict const dvxdbuoyancy_filenames_ptr,
        DW_DTYPE *__restrict const dvxdx_store_1,
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
        DW_DTYPE *__restrict const grad_lamb_thread,
        DW_DTYPE *__restrict const grad_mu_thread,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_mu_zy_thread,
        DW_DTYPE *__restrict const grad_mu_zx_thread,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_mu_yx_thread,
#endif
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_buoyancy_z_thread,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_buoyancy_y_thread,
#endif
        DW_DTYPE *__restrict const grad_buoyancy_x_thread,
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
        DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const rdy,
#endif
        DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
        int64_t const n_shots,
#if DW_NDIM >= 3
        int64_t const nz,
#endif
#if DW_NDIM >= 2
        int64_t const ny,
#endif
        int64_t const nx,
#if DW_NDIM >= 3
        int64_t const n_sources_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_per_shot,
#endif
        int64_t const n_sources_x_per_shot, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_per_shot,
#endif
        int64_t n_receivers_x_per_shot, int64_t const n_receivers_p_per_shot,
        int64_t const step_ratio, int64_t const storage_mode,
        size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
        bool const lamb_requires_grad, bool const mu_requires_grad,
        bool const buoyancy_requires_grad, bool const lamb_batched,
        bool const mu_batched, bool const buoyancy_batched,
        bool const storage_compression, int64_t start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y0,
#endif
        int64_t const pml_x0,
#if DW_NDIM >= 3
        int64_t const pml_z1,
#endif
#if DW_NDIM >= 2
        int64_t const pml_y1,
#endif
        int64_t const pml_x1, int64_t const n_threads, void *unused) {
#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
#if DW_NDIM >= 2
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
#endif
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
#if DW_NDIM == 3
    int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
    int64_t const n_grid_points = ny * nx;
#else
    int64_t const n_grid_points = nx;
#endif
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * n_grid_points;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    int64_t const si = shot * n_grid_points;
#if DW_NDIM >= 3
    int64_t const siz = shot * n_sources_z_per_shot;
    int64_t const riz = shot * n_receivers_z_per_shot;
    int64_t const *__restrict const sources_z_i_shot = sources_z_i + siz;
    int64_t const *__restrict const receivers_z_i_shot = receivers_z_i + riz;
    DW_DTYPE *__restrict const grad_f_z_shot = grad_f_z + siz;
    DW_DTYPE const *__restrict const grad_r_z_shot = grad_r_z + riz;
    DW_DTYPE const *__restrict const mu_zy_shot =
        mu_batched ? mu_zy + si : mu_zy;
    DW_DTYPE const *__restrict const mu_zx_shot =
        mu_batched ? mu_zx + si : mu_zx;
    DW_DTYPE const *__restrict const buoyancy_z_shot =
        buoyancy_batched ? buoyancy_z + si : buoyancy_z;
    DW_DTYPE *__restrict const grad_mu_zy_shot =
        grad_mu_zy_thread + (mu_batched ? si : threadi);
    DW_DTYPE *__restrict const grad_mu_zx_shot =
        grad_mu_zx_thread + (mu_batched ? si : threadi);
    DW_DTYPE *__restrict const grad_buoyancy_z_shot =
        grad_buoyancy_z_thread + (buoyancy_batched ? si : threadi);
    DW_DTYPE *__restrict const vz_shot = vz + si;
    DW_DTYPE *__restrict const sigmazz_shot = sigmazz + si;
    DW_DTYPE *__restrict const sigmayz_shot = sigmayz + si;
    DW_DTYPE *__restrict const sigmaxz_shot = sigmaxz + si;
    DW_DTYPE *__restrict const m_vzz_shot = m_vzz + si;
    DW_DTYPE *__restrict const m_vzy_shot = m_vzy + si;
    DW_DTYPE *__restrict const m_vzx_shot = m_vzx + si;
    DW_DTYPE *__restrict const m_vyz_shot = m_vyz + si;
    DW_DTYPE *__restrict const m_vxz_shot = m_vxz + si;
#endif
#if DW_NDIM >= 2
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const *__restrict const sources_y_i_shot = sources_y_i + siy;
    int64_t const *__restrict const receivers_y_i_shot = receivers_y_i + riy;
    DW_DTYPE *__restrict const grad_f_y_shot = grad_f_y + siy;
    DW_DTYPE const *__restrict const grad_r_y_shot = grad_r_y + riy;
    DW_DTYPE const *__restrict const mu_yx_shot =
        mu_batched ? mu_yx + si : mu_yx;
    DW_DTYPE const *__restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE *__restrict const grad_mu_yx_shot =
        grad_mu_yx_thread + (mu_batched ? si : threadi);
    DW_DTYPE *__restrict const grad_buoyancy_y_shot =
        grad_buoyancy_y_thread + (buoyancy_batched ? si : threadi);
    DW_DTYPE *__restrict const vy_shot = vy + si;
    DW_DTYPE *__restrict const sigmayy_shot = sigmayy + si;
    DW_DTYPE *__restrict const sigmaxy_shot = sigmaxy + si;
    DW_DTYPE *__restrict const m_vyy_shot = m_vyy + si;
    DW_DTYPE *__restrict const m_vyx_shot = m_vyx + si;
    DW_DTYPE *__restrict const m_vxy_shot = m_vxy + si;
#endif
    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const sip = shot * n_sources_p_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const rip = shot * n_receivers_p_per_shot;
    int64_t const *__restrict const sources_x_i_shot = sources_x_i + six;
    int64_t const *__restrict const sources_p_i_shot = sources_p_i + sip;
    int64_t const *__restrict const receivers_x_i_shot = receivers_x_i + rix;
    int64_t const *__restrict const receivers_p_i_shot = receivers_p_i + rip;
    DW_DTYPE *__restrict const grad_f_x_shot = grad_f_x + six;
    DW_DTYPE *__restrict const grad_f_p_shot = grad_f_p + sip;
    DW_DTYPE const *__restrict const grad_r_x_shot = grad_r_x + rix;
    DW_DTYPE const *__restrict const grad_r_p_shot = grad_r_p + rip;

    DW_DTYPE *__restrict const grad_lamb_shot =
        grad_lamb_thread + (lamb_batched ? si : threadi);
    DW_DTYPE *__restrict const grad_mu_shot =
        grad_mu_thread + (mu_batched ? si : threadi);
    DW_DTYPE *__restrict const grad_buoyancy_x_shot =
        grad_buoyancy_x_thread + (buoyancy_batched ? si : threadi);
    DW_DTYPE const *__restrict const lamb_shot =
        lamb_batched ? lamb + si : lamb;
    DW_DTYPE const *__restrict const mu_shot = mu_batched ? mu + si : mu;
    DW_DTYPE const *__restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE *__restrict const vx_shot = vx + si;
    DW_DTYPE *__restrict const sigmaxx_shot = sigmaxx + si;
    DW_DTYPE *__restrict const m_vxx_shot = m_vxx + si;
    int64_t t;

#define OPEN_FILE_READ(name, grad_cond)                  \
  FILE *fp_##name = NULL;                                \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {     \
    fp_##name = fopen(name##_filenames_ptr[shot], "rb"); \
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
#define SETUP_STORE_LOAD(name, grad_cond)                                    \
  DW_DTYPE *__restrict const name##_store_1_t =                              \
      name##_store_1 + si +                                                  \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)              \
           ? (t / step_ratio) * n_shots * n_grid_points                      \
           : 0);                                                             \
  void *__restrict const name##_store_2_t =                                  \
      (uint8_t *)name##_store_2 +                                            \
      (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)       \
                   ? t / step_ratio * n_shots                                \
                   : 0)) *                                                   \
          (int64_t)shot_bytes_comp;                                          \
  if ((grad_cond) && ((t % step_ratio) == 0)) {                              \
    int64_t const step_idx = t / step_ratio;                                 \
    STORAGE_FUNC(load_snapshot_cpu)(                                         \
        (void *)name##_store_1_t, name##_store_2_t, fp_##name, storage_mode, \
        storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,   \
        DIM_ARGS);                                                           \
  }

#if DW_NDIM >= 3
      SETUP_STORE_LOAD(dvzdbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvzdz, lamb_requires_grad || mu_requires_grad)
      SETUP_STORE_LOAD(dvzdx_plus_dvxdz, mu_requires_grad)
      SETUP_STORE_LOAD(dvzdy_plus_dvydz, mu_requires_grad)
      DW_DTYPE *__restrict const m_sigmazzz_t =
          (((start_t - 1 - t) & 1) ? m_sigmazzzn : m_sigmazzz) + si;
      DW_DTYPE *__restrict const m_sigmayzy_t =
          (((start_t - 1 - t) & 1) ? m_sigmayzyn : m_sigmayzy) + si;
      DW_DTYPE *__restrict const m_sigmaxzx_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxzxn : m_sigmaxzx) + si;
      DW_DTYPE *__restrict const m_sigmayzz_t =
          (((start_t - 1 - t) & 1) ? m_sigmayzzn : m_sigmayzz) + si;
      DW_DTYPE *__restrict const m_sigmaxzz_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxzzn : m_sigmaxzz) + si;
      DW_DTYPE *__restrict const m_sigmazzzn_t =
          (((start_t - 1 - t) & 1) ? m_sigmazzz : m_sigmazzzn) + si;
      DW_DTYPE *__restrict const m_sigmayzyn_t =
          (((start_t - 1 - t) & 1) ? m_sigmayzy : m_sigmayzyn) + si;
      DW_DTYPE *__restrict const m_sigmaxzxn_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxzx : m_sigmaxzxn) + si;
      DW_DTYPE *__restrict const m_sigmayzzn_t =
          (((start_t - 1 - t) & 1) ? m_sigmayzz : m_sigmayzzn) + si;
      DW_DTYPE *__restrict const m_sigmaxzzn_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxzz : m_sigmaxzzn) + si;
#endif
#if DW_NDIM >= 2
      SETUP_STORE_LOAD(dvydbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvydy, lamb_requires_grad || mu_requires_grad)
      SETUP_STORE_LOAD(dvydx_plus_dvxdy, mu_requires_grad)
      DW_DTYPE *__restrict const m_sigmayyy_t =
          (((start_t - 1 - t) & 1) ? m_sigmayyyn : m_sigmayyy) + si;
      DW_DTYPE *__restrict const m_sigmaxyy_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxyyn : m_sigmaxyy) + si;
      DW_DTYPE *__restrict const m_sigmaxyx_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxyxn : m_sigmaxyx) + si;
      DW_DTYPE *__restrict const m_sigmayyyn_t =
          (((start_t - 1 - t) & 1) ? m_sigmayyy : m_sigmayyyn) + si;
      DW_DTYPE *__restrict const m_sigmaxyyn_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxyy : m_sigmaxyyn) + si;
      DW_DTYPE *__restrict const m_sigmaxyxn_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxyx : m_sigmaxyxn) + si;
#endif
      SETUP_STORE_LOAD(dvxdbuoyancy, buoyancy_requires_grad)
      SETUP_STORE_LOAD(dvxdx, lamb_requires_grad || mu_requires_grad)

      bool const lamb_requires_grad_t =
          lamb_requires_grad && ((t % step_ratio) == 0);
      bool const mu_requires_grad_t =
          mu_requires_grad && ((t % step_ratio) == 0);
      bool const buoyancy_requires_grad_t =
          buoyancy_requires_grad && ((t % step_ratio) == 0);
      DW_DTYPE *__restrict const m_sigmaxxx_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxxxn : m_sigmaxxx) + si;
      DW_DTYPE *__restrict const m_sigmaxxxn_t =
          (((start_t - 1 - t) & 1) ? m_sigmaxxx : m_sigmaxxxn) + si;

#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

      if (n_sources_p_per_shot > 0) {
        record_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, sources_p_i_shot,
            grad_f_p_shot + t * n_shots * n_sources_p_per_shot,
            n_sources_p_per_shot);
      }

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;

#if DW_NDIM >= 3

            // vz

#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE w_sum = 0;
                  w_sum += pml_z == 1 ? -DIFFZH1(VZ_Z) : -DIFFZH1(VZ_Z_PML);
                  w_sum += pml_y == 1 ? -DIFFY1(VZ_Y) : -DIFFY1(VZ_Y_PML);
                  w_sum += pml_x == 1 ? -DIFFX1(VZ_X) : -DIFFX1(VZ_X_PML);

                  vz_shot[i] += w_sum;

                  if (pml_z != 1) {
                    m_sigmazzzn_t[i] =
                        buoyancy_z_shot[i] * dt * azh[z] * vz_shot[i] +
                        azh[z] * m_sigmazzz_t[i];
                  }
                  if (pml_y != 1) {
                    m_sigmayzyn_t[i] =
                        buoyancy_z_shot[i] * dt * ay[y] * vz_shot[i] +
                        ay[y] * m_sigmayzy_t[i];
                  }
                  if (pml_x != 1) {
                    m_sigmaxzxn_t[i] =
                        buoyancy_z_shot[i] * dt * ax[x] * vz_shot[i] +
                        ax[x] * m_sigmaxzx_t[i];
                  }

                  if (buoyancy_requires_grad_t) {
                    grad_buoyancy_z_shot[i] += vz_shot[i] *
                                               dvzdbuoyancy_store_1_t[i] *
                                               (DW_DTYPE)step_ratio;
                  }
                }
#endif

#if DW_NDIM >= 2

                // vy

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
                  w_sum += pml_z == 1 ? -DIFFZ1(VY_Z) : -DIFFZ1(VY_Z_PML);
#endif
                  w_sum += pml_y == 1 ? -DIFFYH1(VY_Y) : -DIFFYH1(VY_Y_PML);
                  w_sum += pml_x == 1 ? -DIFFX1(VY_X) : -DIFFX1(VY_X_PML);

                  vy_shot[i] += w_sum;

#if DW_NDIM >= 3
                  if (pml_z != 1) {
                    m_sigmayzzn_t[i] =
                        buoyancy_y_shot[i] * dt * az[z] * vy_shot[i] +
                        az[z] * m_sigmayzz_t[i];
                  }
#endif
                  if (pml_y != 1) {
                    m_sigmayyyn_t[i] =
                        buoyancy_y_shot[i] * dt * ayh[y] * vy_shot[i] +
                        ayh[y] * m_sigmayyy_t[i];
                  }
                  if (pml_x != 1) {
                    m_sigmaxyxn_t[i] =
                        buoyancy_y_shot[i] * dt * ax[x] * vy_shot[i] +
                        ax[x] * m_sigmaxyx_t[i];
                  }

                  if (buoyancy_requires_grad_t) {
                    grad_buoyancy_y_shot[i] += vy_shot[i] *
                                               dvydbuoyancy_store_1_t[i] *
                                               (DW_DTYPE)step_ratio;
                  }
                }

#endif

                // vx
#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
                  w_sum += pml_z == 1 ? -DIFFZ1(VX_Z) : -DIFFZ1(VX_Z_PML);
#endif
#if DW_NDIM >= 2
                  w_sum += pml_y == 1 ? -DIFFY1(VX_Y) : -DIFFY1(VX_Y_PML);
#endif
                  w_sum += pml_x == 1 ? -DIFFXH1(VX_X) : -DIFFXH1(VX_X_PML);

                  vx_shot[i] += w_sum;

#if DW_NDIM >= 3
                  if (pml_z != 1) {
                    m_sigmaxzzn_t[i] =
                        buoyancy_x_shot[i] * dt * az[z] * vx_shot[i] +
                        az[z] * m_sigmaxzz_t[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y != 1) {
                    m_sigmaxyyn_t[i] =
                        buoyancy_x_shot[i] * dt * ay[y] * vx_shot[i] +
                        ay[y] * m_sigmaxyy_t[i];
                  }
#endif
                  if (pml_x != 1) {
                    m_sigmaxxxn_t[i] =
                        buoyancy_x_shot[i] * dt * axh[x] * vx_shot[i] +
                        axh[x] * m_sigmaxxx_t[i];
                  }

                  if (buoyancy_requires_grad_t) {
                    grad_buoyancy_x_shot[i] += vx_shot[i] *
                                               dvxdbuoyancy_store_1_t[i] *
                                               (DW_DTYPE)step_ratio;
                  }
                }
          }

#if DW_NDIM >= 3
      if (n_sources_z_per_shot > 0) {
        record_from_wavefield(
            vz_shot, sources_z_i_shot,
            grad_f_z_shot + t * n_shots * n_sources_z_per_shot,
            n_sources_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_y_per_shot > 0) {
        record_from_wavefield(
            vy_shot, sources_y_i_shot,
            grad_f_y_shot + t * n_shots * n_sources_y_per_shot,
            n_sources_y_per_shot);
      }
#endif
      if (n_sources_x_per_shot > 0) {
        record_from_wavefield(
            vx_shot, sources_x_i_shot,
            grad_f_x_shot + t * n_shots * n_sources_x_per_shot,
            n_sources_x_per_shot);
      }

#if DW_NDIM >= 3
#pragma GCC unroll 3
      for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
        for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
          for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
            int64_t z;
#endif
#if DW_NDIM >= 2
            int64_t y;
#endif
            int64_t x;

            // sigmaii

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  if (lamb_requires_grad_t) {
#if DW_NDIM == 3
                    grad_lamb_shot[i] +=
                        (sigmaxx_shot[i] + sigmayy_shot[i] + sigmazz_shot[i]) *
                        (dvxdx_store_1_t[i] + dvydy_store_1_t[i] +
                         dvzdz_store_1_t[i]) *
                        (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
            grad_lamb_shot[i] += (sigmaxx_shot[i] + sigmayy_shot[i]) *
                                 (dvxdx_store_1_t[i] + dvydy_store_1_t[i]) *
                                 (DW_DTYPE)step_ratio;
#else
            grad_lamb_shot[i] +=
                sigmaxx_shot[i] * dvxdx_store_1_t[i] * (DW_DTYPE)step_ratio;
#endif
                  }
                  if (mu_requires_grad_t) {
#if DW_NDIM == 3
                    grad_mu_shot[i] += 2 *
                                       (sigmaxx_shot[i] * dvxdx_store_1_t[i] +
                                        sigmayy_shot[i] * dvydy_store_1_t[i] +
                                        sigmazz_shot[i] * dvzdz_store_1_t[i]) *
                                       (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
            grad_mu_shot[i] += 2 *
                               (sigmaxx_shot[i] * dvxdx_store_1_t[i] +
                                sigmayy_shot[i] * dvydy_store_1_t[i]) *
                               (DW_DTYPE)step_ratio;
#else
            grad_mu_shot[i] +=
                2 * sigmaxx_shot[i] * dvxdx_store_1_t[i] * (DW_DTYPE)step_ratio;
#endif
                  }

#if DW_NDIM >= 3
                  if (pml_z != 1) {
                    m_vzz_shot[i] =
                        (lamb_shot[i] + 2 * mu_shot[i]) * dt * az[z] *
                            sigmazz_shot[i] +
                        lamb_shot[i] * dt * az[z] * sigmaxx_shot[i] +
                        lamb_shot[i] * dt * az[z] * sigmayy_shot[i] +
                        az[z] * m_vzz_shot[i];
                  }
#endif
#if DW_NDIM >= 2
                  if (pml_y != 1) {
                    m_vyy_shot[i] =
                        (lamb_shot[i] + 2 * mu_shot[i]) * dt * ay[y] *
                            sigmayy_shot[i] +
                        lamb_shot[i] * dt * ay[y] * sigmaxx_shot[i] +
#if DW_NDIM == 3
                        lamb_shot[i] * dt * ay[y] * sigmazz_shot[i] +
#endif
                        ay[y] * m_vyy_shot[i];
                  }
#endif
                  if (pml_x != 1) {
                    m_vxx_shot[i] =
                        (lamb_shot[i] + 2 * mu_shot[i]) * dt * ax[x] *
                            sigmaxx_shot[i] +
#if DW_NDIM >= 2
                        lamb_shot[i] * dt * ax[x] * sigmayy_shot[i] +
#endif
#if DW_NDIM == 3
                        lamb_shot[i] * dt * ax[x] * sigmazz_shot[i] +
#endif
                        ax[x] * m_vxx_shot[i];
                  }

#if DW_NDIM >= 3
                  sigmazz_shot[i] +=
                      pml_z == 1 ? -DIFFZ1(SIGMAZZ_Z) : -DIFFZ1(SIGMAZZ_Z_PML);
#endif
#if DW_NDIM >= 2
                  sigmayy_shot[i] +=
                      pml_y == 1 ? -DIFFY1(SIGMAYY_Y) : -DIFFY1(SIGMAYY_Y_PML);
#endif
                  sigmaxx_shot[i] +=
                      pml_x == 1 ? -DIFFX1(SIGMAXX_X) : -DIFFX1(SIGMAXX_X_PML);
                }

#if DW_NDIM >= 2

                // sigmaxy

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;

                  if (mu_requires_grad_t) {
                    grad_mu_yx_shot[i] += sigmaxy_shot[i] *
                                          dvydx_plus_dvxdy_store_1_t[i] *
                                          (DW_DTYPE)step_ratio;
                  }

                  if (pml_y != 1) {
                    m_vxy_shot[i] =
                        mu_yx_shot[i] * dt * ayh[y] * sigmaxy_shot[i] +
                        ayh[y] * m_vxy_shot[i];
                  }
                  if (pml_x != 1) {
                    m_vyx_shot[i] =
                        mu_yx_shot[i] * dt * axh[x] * sigmaxy_shot[i] +
                        axh[x] * m_vyx_shot[i];
                  }

                  sigmaxy_shot[i] += pml_y == 1 ? -DIFFYH1(SIGMAXY_Y)
                                                : -DIFFYH1(SIGMAXY_Y_PML);
                  sigmaxy_shot[i] += pml_x == 1 ? -DIFFXH1(SIGMAXY_X)
                                                : -DIFFXH1(SIGMAXY_X_PML);
                }

#endif

#if DW_NDIM >= 3

                // sigmaxz

#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;

                  if (mu_requires_grad_t) {
                    grad_mu_zx_shot[i] += sigmaxz_shot[i] *
                                          dvzdx_plus_dvxdz_store_1_t[i] *
                                          (DW_DTYPE)step_ratio;
                  }

                  if (pml_z != 1) {
                    m_vxz_shot[i] =
                        mu_zx_shot[i] * dt * azh[z] * sigmaxz_shot[i] +
                        azh[z] * m_vxz_shot[i];
                  }
                  if (pml_x != 1) {
                    m_vzx_shot[i] =
                        mu_zx_shot[i] * dt * axh[x] * sigmaxz_shot[i] +
                        axh[x] * m_vzx_shot[i];
                  }

                  sigmaxz_shot[i] += pml_z == 1 ? -DIFFZH1(SIGMAXZ_Z)
                                                : -DIFFZH1(SIGMAXZ_Z_PML);
                  sigmaxz_shot[i] += pml_x == 1 ? -DIFFXH1(SIGMAXZ_X)
                                                : -DIFFXH1(SIGMAXZ_X_PML);
                }
#endif

#if DW_NDIM >= 3

                // sigmayz

#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  if (mu_requires_grad_t) {
                    grad_mu_zy_shot[i] += sigmayz_shot[i] *
                                          dvzdy_plus_dvydz_store_1_t[i] *
                                          (DW_DTYPE)step_ratio;
                  }

                  if (pml_z != 1) {
                    m_vyz_shot[i] =
                        mu_zy_shot[i] * dt * azh[z] * sigmayz_shot[i] +
                        azh[z] * m_vyz_shot[i];
                  }
                  if (pml_y != 1) {
                    m_vzy_shot[i] =
                        mu_zy_shot[i] * dt * ayh[y] * sigmayz_shot[i] +
                        ayh[y] * m_vzy_shot[i];
                  }

                  sigmayz_shot[i] += pml_z == 1 ? -DIFFZH1(SIGMAYZ_Z)
                                                : -DIFFZH1(SIGMAYZ_Z_PML);
                  sigmayz_shot[i] += pml_y == 1 ? -DIFFYH1(SIGMAYZ_Y)
                                                : -DIFFYH1(SIGMAYZ_Y_PML);
                }

#endif
          }

#if DW_NDIM >= 3
      if (n_receivers_z_per_shot > 0) {
        add_to_wavefield(vz_shot, receivers_z_i_shot,
                         grad_r_z_shot + t * n_shots * n_receivers_z_per_shot,
                         n_receivers_z_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_y_per_shot > 0) {
        add_to_wavefield(vy_shot, receivers_y_i_shot,
                         grad_r_y_shot + t * n_shots * n_receivers_y_per_shot,
                         n_receivers_y_per_shot);
      }
#endif
      if (n_receivers_x_per_shot > 0) {
        add_to_wavefield(vx_shot, receivers_x_i_shot,
                         grad_r_x_shot + t * n_shots * n_receivers_x_per_shot,
                         n_receivers_x_per_shot);
      }
      if (n_receivers_p_per_shot > 0) {
        add_pressure(
#if DW_NDIM >= 3
            sigmazz_shot,
#endif
#if DW_NDIM >= 2
            sigmayy_shot,
#endif
            sigmaxx_shot, receivers_p_i_shot,
            grad_r_p_shot + t * n_shots * n_receivers_p_per_shot,
            n_receivers_p_per_shot);
      }
    }
#if DW_NDIM >= 3
    CLOSE_FILE(dvzdbuoyancy)
    CLOSE_FILE(dvzdz)
    CLOSE_FILE(dvzdx_plus_dvxdz)
    CLOSE_FILE(dvzdy_plus_dvydz)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(dvydbuoyancy)
    CLOSE_FILE(dvydy)
    CLOSE_FILE(dvydx_plus_dvxdy)
#endif
    CLOSE_FILE(dvxdbuoyancy)
    CLOSE_FILE(dvxdx)
  }
#ifdef _OPENMP
  if (lamb_requires_grad && !lamb_batched && n_threads > 1) {
    combine_grad_elastic(grad_lamb, grad_lamb_thread, n_threads,
#if DW_NDIM >= 3
                         nz,
#endif
#if DW_NDIM >= 2
                         ny,
#endif
                         nx);
  }
  if (mu_requires_grad && !mu_batched && n_threads > 1) {
    combine_grad_elastic(grad_mu, grad_mu_thread, n_threads,
#if DW_NDIM >= 3
                         nz,
#endif
#if DW_NDIM >= 2
                         ny,
#endif
                         nx);
#if DW_NDIM >= 3
    combine_grad_elastic(grad_mu_zy, grad_mu_zy_thread, n_threads, nz, ny, nx);
    combine_grad_elastic(grad_mu_zx, grad_mu_zx_thread, n_threads, nz, ny, nx);
#endif
#if DW_NDIM >= 2
    combine_grad_elastic(grad_mu_yx, grad_mu_yx_thread, n_threads,
#if DW_NDIM >= 3
                         nz,
#endif
                         ny, nx);
#endif
  }
  if (buoyancy_requires_grad && !buoyancy_batched && n_threads > 1) {
#if DW_NDIM >= 3
    combine_grad_elastic(grad_buoyancy_z, grad_buoyancy_z_thread, n_threads, nz,
                         ny, nx);
#endif
#if DW_NDIM >= 2
    combine_grad_elastic(grad_buoyancy_y, grad_buoyancy_y_thread, n_threads,
#if DW_NDIM >= 3
                         nz,
#endif
                         ny, nx);
#endif
    combine_grad_elastic(grad_buoyancy_x, grad_buoyancy_x_thread, n_threads,
#if DW_NDIM >= 3
                         nz,
#endif
#if DW_NDIM >= 2
                         ny,
#endif
                         nx);
  }
#endif /* _OPENMP */
  return 0;
}

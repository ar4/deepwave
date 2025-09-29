/*
 * Elastic wave equation propagator
 */

/*
 * This file contains the C implementation of the elastic wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2 and 4.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * The propagator solves the 2D elastic wave equation using a velocity-stress
 * formulation on a staggered grid. The free surface boundary condition is
 * implemented using the improved vacuum method of Zeng et al. (2012), and the
 * PML implementation is based on the C-PML method of Komatitsch and
 * Martin (2007).
 *
 * The code is structured to maximize performance by using macros to
 * generate the code for each of the nine regions of the computational
 * domain (a central region, four edge regions, and four corner regions),
 * and by using OpenMP to parallelize the loops over shots.
 */

/*
 * The staggered grid used in this propagator is shown below.
 * Lowercase letters indicate that the component is not used.
 * The model parameters (lambda, mu, and buoyancy) are at the same
 * locations as vx.
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

#include "common_cpu.h"
#include "staggered_grid.h"

#define CAT_I(name, accuracy, dtype, device) \
  elastic_iso_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

// Access the wavefield at offset (dy, dx) from the current index i
// Note: arrays are row-major with flattened index i = y*nx + x. For batched
// shots the base pointer is shot_offset = shot * ny * nx, so callers pass
// pointers already offset into the shot's memory region.
#define VY(dy, dx) vy[i + dy * nx + dx]
#define VX(dy, dx) vx[i + dy * nx + dx]
#define SIGMAYY(dy, dx) sigmayy[i + dy * nx + dx]
#define SIGMAXX(dy, dx) sigmaxx[i + dy * nx + dx]
#define SIGMAXY(dy, dx) sigmaxy[i + dy * nx + dx]
#define LAMB(dy, dx) lamb[i + dy * nx + dx]
#define MU(dy, dx) mu[i + dy * nx + dx]
#define MU_YX(dy, dx) mu_yx[i + dy * nx + dx]
#define BUOYANCY_Y(dy, dx) buoyancy_y[i + dy * nx + dx]
#define BUOYANCY_X(dy, dx) buoyancy_x[i + dy * nx + dx]
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

#define SET_RANGE_V(pml_y, pml_x)          \
  {                                        \
    if (pml_y == 0) {                      \
      y_begin_y = FD_PAD;                  \
      y_end_y = pml_y0;                    \
      y_begin_x = FD_PAD;                  \
      y_end_x = pml_y0;                    \
    } else if (pml_y == 2) {               \
      y_begin_y = MAX(pml_y0, pml_y1 - 1); \
      y_end_y = ny - FD_PAD;               \
      y_begin_x = pml_y1;                  \
      y_end_x = ny - FD_PAD + 1;           \
    } else {                               \
      y_begin_y = pml_y0;                  \
      y_end_y = MAX(pml_y0, pml_y1 - 1);   \
      y_begin_x = pml_y0;                  \
      y_end_x = pml_y1;                    \
    }                                      \
    if (pml_x == 0) {                      \
      x_begin_y = FD_PAD;                  \
      x_end_y = pml_x0;                    \
      x_begin_x = FD_PAD;                  \
      x_end_x = pml_x0;                    \
    } else if (pml_x == 2) {               \
      x_begin_y = pml_x1;                  \
      x_end_y = nx - FD_PAD + 1;           \
      x_begin_x = MAX(pml_x0, pml_x1 - 1); \
      x_end_x = nx - FD_PAD;               \
    } else {                               \
      x_begin_y = pml_x0;                  \
      x_end_y = pml_x1;                    \
      x_begin_x = pml_x0;                  \
      x_end_x = MAX(pml_x0, pml_x1 - 1);   \
    }                                      \
  }

#define SET_RANGE_SIGMA(pml_y, pml_x)       \
  {                                         \
    if (pml_y == 0) {                       \
      y_begin_ii = FD_PAD;                  \
      y_end_ii = pml_y0;                    \
      y_begin_xy = FD_PAD;                  \
      y_end_xy = pml_y0;                    \
    } else if (pml_y == 2) {                \
      y_begin_ii = pml_y1;                  \
      y_end_ii = ny - FD_PAD + 1;           \
      y_begin_xy = MAX(pml_y0, pml_y1 - 1); \
      y_end_xy = ny - FD_PAD;               \
    } else {                                \
      y_begin_ii = pml_y0;                  \
      y_end_ii = pml_y1;                    \
      y_begin_xy = pml_y0;                  \
      y_end_xy = MAX(pml_y0, pml_y1 - 1);   \
    }                                       \
    if (pml_x == 0) {                       \
      x_begin_ii = FD_PAD;                  \
      x_end_ii = pml_x0;                    \
      x_begin_xy = FD_PAD;                  \
      x_end_xy = pml_x0;                    \
    } else if (pml_x == 2) {                \
      x_begin_ii = pml_x1;                  \
      x_end_ii = nx - FD_PAD + 1;           \
      x_begin_xy = MAX(pml_x0, pml_x1 - 1); \
      x_end_xy = nx - FD_PAD;               \
    } else {                                \
      x_begin_ii = pml_x0;                  \
      x_end_ii = pml_x1;                    \
      x_begin_xy = pml_x0;                  \
      x_end_xy = MAX(pml_x0, pml_x1 - 1);   \
    }                                       \
  }

#define FORWARD_KERNEL_V(pml_y, pml_x, buoyancy_requires_grad)              \
  {                                                                         \
    SET_RANGE_V(pml_y, pml_x)                                               \
    _Pragma("omp simd collapse(2)") for (y = y_begin_y; y < y_end_y; ++y) { \
      for (x = x_begin_y; x < x_end_y; ++x) {                               \
        int64_t i = y * nx + x;                                             \
        DW_DTYPE dsigmayydy = DIFFYH1(SIGMAYY);                             \
        DW_DTYPE dsigmaxydx = DIFFX1(SIGMAXY);                              \
                                                                            \
        if (pml_y == 0 || pml_y == 2) {                                     \
          m_sigmayyy[i] = ayh[y] * m_sigmayyy[i] + byh[y] * dsigmayydy;     \
          dsigmayydy += m_sigmayyy[i];                                      \
        }                                                                   \
        if (pml_x == 0 || pml_x == 2) {                                     \
          m_sigmaxyx[i] = ax[x] * m_sigmaxyx[i] + bx[x] * dsigmaxydx;       \
          dsigmaxydx += m_sigmaxyx[i];                                      \
        }                                                                   \
        vy[i] += buoyancy_y[i] * dt * (dsigmayydy + dsigmaxydx);            \
        if (buoyancy_requires_grad) {                                       \
          dvydbuoyancy[i] = dt * (dsigmayydy + dsigmaxydx);                 \
        }                                                                   \
      }                                                                     \
    }                                                                       \
                                                                            \
    _Pragma("omp simd collapse(2)") for (y = y_begin_x; y < y_end_x; ++y) { \
      for (x = x_begin_x; x < x_end_x; ++x) {                               \
        int64_t i = y * nx + x;                                             \
        DW_DTYPE dsigmaxydy = DIFFY1(SIGMAXY);                              \
        DW_DTYPE dsigmaxxdx = DIFFXH1(SIGMAXX);                             \
                                                                            \
        if (pml_y == 0 || pml_y == 2) {                                     \
          m_sigmaxyy[i] = ay[y] * m_sigmaxyy[i] + by[y] * dsigmaxydy;       \
          dsigmaxydy += m_sigmaxyy[i];                                      \
        }                                                                   \
        if (pml_x == 0 || pml_x == 2) {                                     \
          m_sigmaxxx[i] = axh[x] * m_sigmaxxx[i] + bxh[x] * dsigmaxxdx;     \
          dsigmaxxdx += m_sigmaxxx[i];                                      \
        }                                                                   \
        vx[i] += buoyancy_x[i] * dt * (dsigmaxxdx + dsigmaxydy);            \
        if (buoyancy_requires_grad) {                                       \
          dvxdbuoyancy[i] = dt * (dsigmaxxdx + dsigmaxydy);                 \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }

#define FORWARD_KERNEL_SIGMA(pml_y, pml_x, lamb_requires_grad,                \
                             mu_requires_grad)                                \
  {                                                                           \
    SET_RANGE_SIGMA(pml_y, pml_x)                                             \
    _Pragma("omp simd collapse(2)") for (y = y_begin_ii; y < y_end_ii; ++y) { \
      for (x = x_begin_ii; x < x_end_ii; ++x) {                               \
        int64_t i = y * nx + x;                                               \
        DW_DTYPE dvydy = DIFFY1(VY);                                          \
        DW_DTYPE dvxdx = DIFFX1(VX);                                          \
                                                                              \
        if (pml_y == 0 || pml_y == 2) {                                       \
          m_vyy[i] = ay[y] * m_vyy[i] + by[y] * dvydy;                        \
          dvydy += m_vyy[i];                                                  \
        }                                                                     \
        if (pml_x == 0 || pml_x == 2) {                                       \
          m_vxx[i] = ax[x] * m_vxx[i] + bx[x] * dvxdx;                        \
          dvxdx += m_vxx[i];                                                  \
        }                                                                     \
        sigmayy[i] += dt * ((lamb[i] + 2 * mu[i]) * dvydy + lamb[i] * dvxdx); \
        sigmaxx[i] += dt * ((lamb[i] + 2 * mu[i]) * dvxdx + lamb[i] * dvydy); \
        if (lamb_requires_grad || mu_requires_grad) {                         \
          dvydy_store[i] = dt * dvydy;                                        \
          dvxdx_store[i] = dt * dvxdx;                                        \
        }                                                                     \
      }                                                                       \
    }                                                                         \
                                                                              \
    _Pragma("omp simd collapse(2)") for (y = y_begin_xy; y < y_end_xy; ++y) { \
      for (x = x_begin_xy; x < x_end_xy; ++x) {                               \
        int64_t i = y * nx + x;                                               \
        DW_DTYPE dvydx = DIFFXH1(VY);                                         \
        DW_DTYPE dvxdy = DIFFYH1(VX);                                         \
                                                                              \
        if (pml_y == 0 || pml_y == 2) {                                       \
          m_vxy[i] = ayh[y] * m_vxy[i] + byh[y] * dvxdy;                      \
          dvxdy += m_vxy[i];                                                  \
        }                                                                     \
        if (pml_x == 0 || pml_x == 2) {                                       \
          m_vyx[i] = axh[x] * m_vyx[i] + bxh[x] * dvydx;                      \
          dvydx += m_vyx[i];                                                  \
        }                                                                     \
        {                                                                     \
          sigmaxy[i] += dt * mu_yx[i] * (dvydx + dvxdy);                      \
          if (mu_requires_grad) {                                             \
            dvydxdvxdy_store[i] = dt * (dvydx + dvxdy);                       \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

#define BACKWARD_KERNEL_SIGMA(pml_y, pml_x, buoyancy_requires_grad)         \
  {                                                                         \
    SET_RANGE_V(pml_y, pml_x)                                               \
    _Pragma("omp simd collapse(2)") for (y = y_begin_y; y < y_end_y; ++y) { \
      for (x = x_begin_y; x < x_end_y; ++x) {                               \
        int64_t i = y * nx + x;                                             \
        vy[i] += ((pml_y == 1 ? -DIFFYH1(VY_Y) : -DIFFYH1(VY_Y_PML)) +      \
                  (pml_x == 1 ? -DIFFX1(VY_X) : -DIFFX1(VY_X_PML)));        \
        if (pml_y == 0 || pml_y == 2) {                                     \
          m_sigmayyyn[i] =                                                  \
              buoyancy_y[i] * dt * ayh[y] * vy[i] + ayh[y] * m_sigmayyy[i]; \
        }                                                                   \
        if (pml_x == 0 || pml_x == 2) {                                     \
          m_sigmaxyxn[i] =                                                  \
              buoyancy_y[i] * dt * ax[x] * vy[i] + ax[x] * m_sigmaxyx[i];   \
        }                                                                   \
        if (buoyancy_requires_grad) {                                       \
          grad_buoyancy_y[i] +=                                             \
              vy[i] * dvydbuoyancy[i] * (DW_DTYPE)step_ratio;               \
        }                                                                   \
      }                                                                     \
    }                                                                       \
                                                                            \
    _Pragma("omp simd collapse(2)") for (y = y_begin_x; y < y_end_x; ++y) { \
      for (x = x_begin_x; x < x_end_x; ++x) {                               \
        int64_t i = y * nx + x;                                             \
        vx[i] += ((pml_y == 1 ? -DIFFY1(VX_Y) : -DIFFY1(VX_Y_PML)) +        \
                  (pml_x == 1 ? -DIFFXH1(VX_X) : -DIFFXH1(VX_X_PML)));      \
        if (pml_y == 0 || pml_y == 2) {                                     \
          m_sigmaxyyn[i] =                                                  \
              buoyancy_x[i] * dt * ay[y] * vx[i] + ay[y] * m_sigmaxyy[i];   \
        }                                                                   \
        if (pml_x == 0 || pml_x == 2) {                                     \
          m_sigmaxxxn[i] =                                                  \
              buoyancy_x[i] * dt * axh[x] * vx[i] + axh[x] * m_sigmaxxx[i]; \
        }                                                                   \
        if (buoyancy_requires_grad) {                                       \
          grad_buoyancy_x[i] +=                                             \
              vx[i] * dvxdbuoyancy[i] * (DW_DTYPE)step_ratio;               \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }

#define BACKWARD_KERNEL_V(pml_y, pml_x, lamb_requires_grad, mu_requires_grad) \
  {                                                                           \
    SET_RANGE_SIGMA(pml_y, pml_x)                                             \
    _Pragma("omp simd collapse(2)") for (y = y_begin_ii; y < y_end_ii; ++y) { \
      for (x = x_begin_ii; x < x_end_ii; ++x) {                               \
        int64_t i = y * nx + x;                                               \
        if (lamb_requires_grad) {                                             \
          grad_lamb[i] += (sigmayy[i] + sigmaxx[i]) *                         \
                          (dvydy_store[i] + dvxdx_store[i]) *                 \
                          (DW_DTYPE)step_ratio;                               \
        }                                                                     \
        if (mu_requires_grad) {                                               \
          grad_mu[i] +=                                                       \
              2 *                                                             \
              (sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) *   \
              (DW_DTYPE)step_ratio;                                           \
        }                                                                     \
        if (pml_y == 0 || pml_y == 2) {                                       \
          m_vyy[i] = (lamb[i] + 2 * mu[i]) * dt * ay[y] * sigmayy[i] +        \
                     lamb[i] * dt * ay[y] * sigmaxx[i] + ay[y] * m_vyy[i];    \
        }                                                                     \
        if (pml_x == 0 || pml_x == 2) {                                       \
          m_vxx[i] = (lamb[i] + 2 * mu[i]) * dt * ax[x] * sigmaxx[i] +        \
                     lamb[i] * dt * ax[x] * sigmayy[i] + ax[x] * m_vxx[i];    \
        }                                                                     \
        sigmayy[i] +=                                                         \
            (pml_y == 1 ? -DIFFY1(SIGMAYY_Y) : -DIFFY1(SIGMAYY_Y_PML));       \
        sigmaxx[i] +=                                                         \
            (pml_x == 1 ? -DIFFX1(SIGMAXX_X) : -DIFFX1(SIGMAXX_X_PML));       \
      }                                                                       \
    }                                                                         \
                                                                              \
    _Pragma("omp simd collapse(2)") for (y = y_begin_xy; y < y_end_xy; ++y) { \
      for (x = x_begin_xy; x < x_end_xy; ++x) {                               \
        int64_t i = y * nx + x;                                               \
        if (mu_requires_grad) {                                               \
          grad_mu_yx[i] +=                                                    \
              sigmaxy[i] * dvydxdvxdy_store[i] * (DW_DTYPE)step_ratio;        \
        }                                                                     \
        if (pml_y == 0 || pml_y == 2) {                                       \
          m_vxy[i] = mu_yx[i] * dt * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i]; \
        }                                                                     \
        if (pml_x == 0 || pml_x == 2) {                                       \
          m_vyx[i] = mu_yx[i] * dt * axh[x] * sigmaxy[i] + axh[x] * m_vyx[i]; \
        }                                                                     \
        sigmaxy[i] +=                                                         \
            ((pml_y == 1 ? -DIFFYH1(SIGMAXY_Y) : -DIFFYH1(SIGMAXY_Y_PML)) +   \
             (pml_x == 1 ? -DIFFXH1(SIGMAXY_X) : -DIFFXH1(SIGMAXY_X_PML)));   \
      }                                                                       \
    }                                                                         \
  }

static inline void add_pressure(DW_DTYPE *__restrict const sigmayy,
                         DW_DTYPE *__restrict const sigmaxx,
                         int64_t const *__restrict const sources_i,
                         DW_DTYPE const *__restrict const f,
                         int64_t const n_sources_per_shot) {
  int64_t source_idx;
#pragma omp simd
  for (source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
    if (sources_i[source_idx] < 0) continue;
    sigmayy[sources_i[source_idx]] -= f[source_idx] / (DW_DTYPE)2;
    sigmaxx[sources_i[source_idx]] -= f[source_idx] / (DW_DTYPE)2;
  }
}

static inline void record_pressure(DW_DTYPE const *__restrict sigmayy,
                                   DW_DTYPE const *__restrict sigmaxx,
                                   int64_t const *__restrict locations,
                                   DW_DTYPE *__restrict amplitudes, int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i])
      amplitudes[i] =
          -(sigmayy[locations[i]] + sigmaxx[locations[i]]) / (DW_DTYPE)2;
  }
}

static inline void combine_grad_elastic(DW_DTYPE *__restrict const grad,
                                 DW_DTYPE const *__restrict const grad_thread,
                                 int64_t const n_threads, int64_t const ny,
                                 int64_t const nx) {
  int64_t y, x, threadidx;
#pragma omp simd collapse(2)
  for (y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
    for (x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
      int64_t const i = y * nx + x;
      for (threadidx = 0; threadidx < n_threads; ++threadidx) {
        grad[i] += grad_thread[threadidx * ny * nx + i];
      }
    }
  }
}

    static inline void forward_shot_v(DW_DTYPE const *__restrict const buoyancy_y,
                               DW_DTYPE const *__restrict const buoyancy_x,
                               DW_DTYPE *__restrict const vy,
                               DW_DTYPE *__restrict const vx,
                               DW_DTYPE const *__restrict const sigmayy,
                               DW_DTYPE const *__restrict const sigmaxy,
                               DW_DTYPE const *__restrict const sigmaxx,
                               DW_DTYPE *__restrict const m_sigmayyy,
                               DW_DTYPE *__restrict const m_sigmaxyy,
                               DW_DTYPE *__restrict const m_sigmaxyx,
                               DW_DTYPE *__restrict const m_sigmaxxx,
                               DW_DTYPE *__restrict const dvydbuoyancy,
                               DW_DTYPE *__restrict const dvxdbuoyancy,
                               DW_DTYPE const *__restrict const ay,
                               DW_DTYPE const *__restrict const ayh,
                               DW_DTYPE const *__restrict const ax,
                               DW_DTYPE const *__restrict const axh,
                               DW_DTYPE const *__restrict const by,
                               DW_DTYPE const *__restrict const byh,
                               DW_DTYPE const *__restrict const bx,
                               DW_DTYPE const *__restrict const bxh,
                               DW_DTYPE const rdy, DW_DTYPE const rdx,
                               DW_DTYPE const dt, int64_t const ny,
                               int64_t const nx,
                               bool const buoyancy_requires_grad,
                               int64_t const pml_y0, int64_t const pml_y1,
                               int64_t const pml_x0, int64_t const pml_x1) {
  int64_t y, x, y_begin_y, y_end_y, x_begin_y, x_end_y, y_begin_x, y_end_x,
      x_begin_x, x_end_x;
  // Check if gradients are required for buoyancy (density)
  if (buoyancy_requires_grad) {
    // Execute forward kernel for all PML configurations with gradients
    FORWARD_KERNEL_V(0, 0, 1)
    FORWARD_KERNEL_V(0, 1, 1)
    FORWARD_KERNEL_V(0, 2, 1)
    FORWARD_KERNEL_V(1, 0, 1)
    FORWARD_KERNEL_V(1, 1, 1)
    FORWARD_KERNEL_V(1, 2, 1)
    FORWARD_KERNEL_V(2, 0, 1)
    FORWARD_KERNEL_V(2, 1, 1)
    FORWARD_KERNEL_V(2, 2, 1)
  } else {
    // Execute forward kernel for all PML configurations without gradients
    FORWARD_KERNEL_V(0, 0, 0)
    FORWARD_KERNEL_V(0, 1, 0)
    FORWARD_KERNEL_V(0, 2, 0)
    FORWARD_KERNEL_V(1, 0, 0)
    FORWARD_KERNEL_V(1, 1, 0)
    FORWARD_KERNEL_V(1, 2, 0)
    FORWARD_KERNEL_V(2, 0, 0)
    FORWARD_KERNEL_V(2, 1, 0)
    FORWARD_KERNEL_V(2, 2, 0)
  }
}

    static inline void forward_shot_sigma(
        DW_DTYPE const *__restrict const lamb,
        DW_DTYPE const *__restrict const mu,
        DW_DTYPE const *__restrict const mu_yx,
        DW_DTYPE const *__restrict const vy,
        DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigmayy,
        DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const sigmaxx,
        DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
        DW_DTYPE *__restrict const m_vxy, DW_DTYPE *__restrict const m_vxx,
        DW_DTYPE *__restrict const dvydy_store,
        DW_DTYPE *__restrict const dvxdx_store,
        DW_DTYPE *__restrict const dvydxdvxdy_store,
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const ayh,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const axh,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const byh,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdy,
        DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const ny,
        int64_t const nx, bool const lamb_requires_grad,
        bool const mu_requires_grad, int64_t const pml_y0, int64_t const pml_y1,
        int64_t const pml_x0, int64_t const pml_x1) {
  int64_t y, x, y_begin_ii, y_end_ii, x_begin_ii, x_end_ii, y_begin_xy,
      y_end_xy, x_begin_xy, x_end_xy;
  if (lamb_requires_grad && mu_requires_grad) {
    FORWARD_KERNEL_SIGMA(0, 0, 1, 1)
    FORWARD_KERNEL_SIGMA(0, 1, 1, 1)
    FORWARD_KERNEL_SIGMA(0, 2, 1, 1)
    FORWARD_KERNEL_SIGMA(1, 0, 1, 1)
    FORWARD_KERNEL_SIGMA(1, 1, 1, 1)
    FORWARD_KERNEL_SIGMA(1, 2, 1, 1)
    FORWARD_KERNEL_SIGMA(2, 0, 1, 1)
    FORWARD_KERNEL_SIGMA(2, 1, 1, 1)
    FORWARD_KERNEL_SIGMA(2, 2, 1, 1)
  } else if (lamb_requires_grad) {
    FORWARD_KERNEL_SIGMA(0, 0, 1, 0)
    FORWARD_KERNEL_SIGMA(0, 1, 1, 0)
    FORWARD_KERNEL_SIGMA(0, 2, 1, 0)
    FORWARD_KERNEL_SIGMA(1, 0, 1, 0)
    FORWARD_KERNEL_SIGMA(1, 1, 1, 0)
    FORWARD_KERNEL_SIGMA(1, 2, 1, 0)
    FORWARD_KERNEL_SIGMA(2, 0, 1, 0)
    FORWARD_KERNEL_SIGMA(2, 1, 1, 0)
    FORWARD_KERNEL_SIGMA(2, 2, 1, 0)
  } else if (mu_requires_grad) {
    FORWARD_KERNEL_SIGMA(0, 0, 0, 1)
    FORWARD_KERNEL_SIGMA(0, 1, 0, 1)
    FORWARD_KERNEL_SIGMA(0, 2, 0, 1)
    FORWARD_KERNEL_SIGMA(1, 0, 0, 1)
    FORWARD_KERNEL_SIGMA(1, 1, 0, 1)
    FORWARD_KERNEL_SIGMA(1, 2, 0, 1)
    FORWARD_KERNEL_SIGMA(2, 0, 0, 1)
    FORWARD_KERNEL_SIGMA(2, 1, 0, 1)
    FORWARD_KERNEL_SIGMA(2, 2, 0, 1)
  } else {
    FORWARD_KERNEL_SIGMA(0, 0, 0, 0)
    FORWARD_KERNEL_SIGMA(0, 1, 0, 0)
    FORWARD_KERNEL_SIGMA(0, 2, 0, 0)
    FORWARD_KERNEL_SIGMA(1, 0, 0, 0)
    FORWARD_KERNEL_SIGMA(1, 1, 0, 0)
    FORWARD_KERNEL_SIGMA(1, 2, 0, 0)
    FORWARD_KERNEL_SIGMA(2, 0, 0, 0)
    FORWARD_KERNEL_SIGMA(2, 1, 0, 0)
    FORWARD_KERNEL_SIGMA(2, 2, 0, 0)
  }
}

static inline void backward_shot(
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
    int64_t const *__restrict const receivers_p_i, DW_DTYPE const rdy,
    DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const ny, int64_t const nx,
    int64_t const n_sources_y_per_shot, int64_t const n_sources_x_per_shot,
    int64_t const n_receivers_y_per_shot, int64_t n_receivers_x_per_shot,
    int64_t const n_receivers_p_per_shot, int64_t const step_ratio,
    bool const lamb_requires_grad, bool const mu_requires_grad,
    bool const buoyancy_requires_grad, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1) {
  int64_t y, x, y_begin_y, y_end_y, x_begin_y, x_end_y, y_begin_x, y_end_x,
      x_begin_x, x_end_x, y_begin_ii, y_end_ii, x_begin_ii, x_end_ii,
      y_begin_xy, y_end_xy, x_begin_xy, x_end_xy;
  if (buoyancy_requires_grad) {
    BACKWARD_KERNEL_SIGMA(0, 0, 1)
    BACKWARD_KERNEL_SIGMA(0, 1, 1)
    BACKWARD_KERNEL_SIGMA(0, 2, 1)
    BACKWARD_KERNEL_SIGMA(1, 0, 1)
    BACKWARD_KERNEL_SIGMA(1, 1, 1)
    BACKWARD_KERNEL_SIGMA(1, 2, 1)
    BACKWARD_KERNEL_SIGMA(2, 0, 1)
    BACKWARD_KERNEL_SIGMA(2, 1, 1)
    BACKWARD_KERNEL_SIGMA(2, 2, 1)
  } else {
    BACKWARD_KERNEL_SIGMA(0, 0, 0)
    BACKWARD_KERNEL_SIGMA(0, 1, 0)
    BACKWARD_KERNEL_SIGMA(0, 2, 0)
    BACKWARD_KERNEL_SIGMA(1, 0, 0)
    BACKWARD_KERNEL_SIGMA(1, 1, 0)
    BACKWARD_KERNEL_SIGMA(1, 2, 0)
    BACKWARD_KERNEL_SIGMA(2, 0, 0)
    BACKWARD_KERNEL_SIGMA(2, 1, 0)
    BACKWARD_KERNEL_SIGMA(2, 2, 0)
  }
  if (n_sources_y_per_shot > 0) {
    record_from_wavefield(vy, sources_y_i, grad_f_y, n_sources_y_per_shot);
  }
  if (n_sources_x_per_shot > 0) {
    record_from_wavefield(vx, sources_x_i, grad_f_x, n_sources_x_per_shot);
  }
  if (lamb_requires_grad && mu_requires_grad) {
    BACKWARD_KERNEL_V(0, 0, 1, 1)
    BACKWARD_KERNEL_V(0, 1, 1, 1)
    BACKWARD_KERNEL_V(0, 2, 1, 1)
    BACKWARD_KERNEL_V(1, 0, 1, 1)
    BACKWARD_KERNEL_V(1, 1, 1, 1)
    BACKWARD_KERNEL_V(1, 2, 1, 1)
    BACKWARD_KERNEL_V(2, 0, 1, 1)
    BACKWARD_KERNEL_V(2, 1, 1, 1)
    BACKWARD_KERNEL_V(2, 2, 1, 1)
  } else if (lamb_requires_grad) {
    BACKWARD_KERNEL_V(0, 0, 1, 0)
    BACKWARD_KERNEL_V(0, 1, 1, 0)
    BACKWARD_KERNEL_V(0, 2, 1, 0)
    BACKWARD_KERNEL_V(1, 0, 1, 0)
    BACKWARD_KERNEL_V(1, 1, 1, 0)
    BACKWARD_KERNEL_V(1, 2, 1, 0)
    BACKWARD_KERNEL_V(2, 0, 1, 0)
    BACKWARD_KERNEL_V(2, 1, 1, 0)
    BACKWARD_KERNEL_V(2, 2, 1, 0)
  } else if (mu_requires_grad) {
    BACKWARD_KERNEL_V(0, 0, 0, 1)
    BACKWARD_KERNEL_V(0, 1, 0, 1)
    BACKWARD_KERNEL_V(0, 2, 0, 1)
    BACKWARD_KERNEL_V(1, 0, 0, 1)
    BACKWARD_KERNEL_V(1, 1, 0, 1)
    BACKWARD_KERNEL_V(1, 2, 0, 1)
    BACKWARD_KERNEL_V(2, 0, 0, 1)
    BACKWARD_KERNEL_V(2, 1, 0, 1)
    BACKWARD_KERNEL_V(2, 2, 0, 1)
  } else {
    BACKWARD_KERNEL_V(0, 0, 0, 0)
    BACKWARD_KERNEL_V(0, 1, 0, 0)
    BACKWARD_KERNEL_V(0, 2, 0, 0)
    BACKWARD_KERNEL_V(1, 0, 0, 0)
    BACKWARD_KERNEL_V(1, 1, 0, 0)
    BACKWARD_KERNEL_V(1, 2, 0, 0)
    BACKWARD_KERNEL_V(2, 0, 0, 0)
    BACKWARD_KERNEL_V(2, 1, 0, 0)
    BACKWARD_KERNEL_V(2, 2, 0, 0)
  }
  if (n_receivers_y_per_shot > 0) {
    add_to_wavefield(vy, receivers_y_i, grad_r_y, n_receivers_y_per_shot);
  }
  if (n_receivers_x_per_shot > 0) {
    add_to_wavefield(vx, receivers_x_i, grad_r_x, n_receivers_x_per_shot);
  }
  if (n_receivers_p_per_shot > 0) {
    add_pressure(sigmayy, sigmaxx, receivers_p_i, grad_r_p,
                 n_receivers_p_per_shot);
  }
}

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
        DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const sigmaxx,
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
        DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
        int64_t const n_shots, int64_t const ny, int64_t const nx,
        int64_t const n_sources_y_per_shot, int64_t const n_sources_x_per_shot,
        int64_t const n_receivers_y_per_shot, int64_t n_receivers_x_per_shot,
        int64_t const n_receivers_p_per_shot, int64_t const step_ratio,
        bool const lamb_requires_grad, bool const mu_requires_grad,
        bool const buoyancy_requires_grad, bool const lamb_batched,
        bool const mu_batched, bool const buoyancy_batched, int64_t start_t,
        int64_t const pml_y0, int64_t const pml_y1, int64_t const pml_x0,
        int64_t const pml_x1, int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const si = shot * ny * nx;
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const rip = shot * n_receivers_p_per_shot;
    DW_DTYPE const *__restrict const lamb_shot =
        lamb_batched ? lamb + si : lamb;
    DW_DTYPE const *__restrict const mu_shot = mu_batched ? mu + si : mu;
    DW_DTYPE const *__restrict const mu_yx_shot =
        mu_batched ? mu_yx + si : mu_yx;
    DW_DTYPE const *__restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE const *__restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    int64_t t;

    for (t = start_t; t < start_t + nt; ++t) {
      int64_t store_i = (t / step_ratio) * n_shots * ny * nx + shot * ny * nx;
      if (n_receivers_y_per_shot > 0) {
        record_from_wavefield(vy + si, receivers_y_i + riy,
                              r_y + t * n_shots * n_receivers_y_per_shot + riy,
                              n_receivers_y_per_shot);
      }
      if (n_receivers_x_per_shot > 0) {
        record_from_wavefield(vx + si, receivers_x_i + rix,
                              r_x + t * n_shots * n_receivers_x_per_shot + rix,
                              n_receivers_x_per_shot);
      }
      if (n_receivers_p_per_shot > 0) {
        record_pressure(sigmayy + si, sigmaxx + si, receivers_p_i + rip,
                        r_p + t * n_shots * n_receivers_p_per_shot + rip,
                        n_receivers_p_per_shot);
      }
      forward_shot_v(buoyancy_y_shot, buoyancy_x_shot, vy + si, vx + si,
                     sigmayy + si, sigmaxy + si, sigmaxx + si, m_sigmayyy + si,
                     m_sigmaxyy + si, m_sigmaxyx + si, m_sigmaxxx + si,
                     dvydbuoyancy + store_i, dvxdbuoyancy + store_i, ay, ayh,
                     ax, axh, by, byh, bx, bxh, rdy, rdx, dt, ny, nx,
                     buoyancy_requires_grad && ((t % step_ratio) == 0), pml_y0,
                     pml_y1, pml_x0, pml_x1);

      if (n_sources_y_per_shot > 0) {
        add_to_wavefield(vy + si, sources_y_i + siy,
                         f_y + t * n_shots * n_sources_y_per_shot + siy,
                         n_sources_y_per_shot);
      }
      if (n_sources_x_per_shot > 0) {
        add_to_wavefield(vx + si, sources_x_i + six,
                         f_x + t * n_shots * n_sources_x_per_shot + six,
                         n_sources_x_per_shot);
      }
      forward_shot_sigma(
          lamb_shot, mu_shot, mu_yx_shot, vy + si, vx + si, sigmayy + si,
          sigmaxy + si, sigmaxx + si, m_vyy + si, m_vyx + si, m_vxy + si,
          m_vxx + si, dvydy_store + store_i, dvxdx_store + store_i,
          dvydxdvxdy_store + store_i, ay, ayh, ax, axh, by, byh, bx, bxh, rdy,
          rdx, dt, ny, nx, lamb_requires_grad && ((t % step_ratio) == 0),
          mu_requires_grad && ((t % step_ratio) == 0), pml_y0, pml_y1, pml_x0,
          pml_x1);
    }
    if (n_receivers_y_per_shot > 0) {
      record_from_wavefield(vy + si, receivers_y_i + riy,
                            r_y + t * n_shots * n_receivers_y_per_shot + riy,
                            n_receivers_y_per_shot);
    }
    if (n_receivers_x_per_shot > 0) {
      record_from_wavefield(vx + si, receivers_x_i + rix,
                            r_x + t * n_shots * n_receivers_x_per_shot + rix,
                            n_receivers_x_per_shot);
    }
  }
}

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
        DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
        DW_DTYPE *__restrict const sigmaxx, DW_DTYPE *__restrict const m_vyy,
        DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
        DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict const m_sigmayyy,
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
        DW_DTYPE *__restrict const grad_lamb_thread,
        DW_DTYPE *__restrict const grad_mu,
        DW_DTYPE *__restrict const grad_mu_thread,
        DW_DTYPE *__restrict const grad_mu_yx,
        DW_DTYPE *__restrict const grad_mu_yx_thread,
        DW_DTYPE *__restrict const grad_buoyancy_y,
        DW_DTYPE *__restrict const grad_buoyancy_y_thread,
        DW_DTYPE *__restrict const grad_buoyancy_x,
        DW_DTYPE *__restrict const grad_buoyancy_x_thread,
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
        DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
        int64_t const n_shots, int64_t const ny, int64_t const nx,
        int64_t const n_sources_y_per_shot, int64_t const n_sources_x_per_shot,
        int64_t const n_receivers_y_per_shot, int64_t n_receivers_x_per_shot,
        int64_t const n_receivers_p_per_shot, int64_t const step_ratio,
        bool const lamb_requires_grad, bool const mu_requires_grad,
        bool const buoyancy_requires_grad, bool const lamb_batched,
        bool const mu_batched, bool const buoyancy_batched, int64_t start_t,
        int64_t const pml_y0, int64_t const pml_y1, int64_t const pml_x0,
        int64_t const pml_x1, int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const si = shot * ny * nx;
    int64_t const siy = shot * n_sources_y_per_shot;
    int64_t const six = shot * n_sources_x_per_shot;
    int64_t const riy = shot * n_receivers_y_per_shot;
    int64_t const rix = shot * n_receivers_x_per_shot;
    int64_t const rip = shot * n_receivers_p_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * ny * nx;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    int64_t const grad_lamb_i = lamb_batched ? si : threadi;
    int64_t const grad_mu_i = mu_batched ? si : threadi;
    int64_t const grad_buoyancy_i = buoyancy_batched ? si : threadi;
    DW_DTYPE const *__restrict const lamb_shot =
        lamb_batched ? lamb + si : lamb;
    DW_DTYPE const *__restrict const mu_shot = mu_batched ? mu + si : mu;
    DW_DTYPE const *__restrict const mu_yx_shot =
        mu_batched ? mu_yx + si : mu_yx;
    DW_DTYPE const *__restrict const buoyancy_y_shot =
        buoyancy_batched ? buoyancy_y + si : buoyancy_y;
    DW_DTYPE const *__restrict const buoyancy_x_shot =
        buoyancy_batched ? buoyancy_x + si : buoyancy_x;
    int64_t t = nt;

    if (n_receivers_y_per_shot > 0 && nt > 0) {
      add_to_wavefield(vy + si, receivers_y_i + riy,
                       grad_r_y + t * n_shots * n_receivers_y_per_shot + riy,
                       n_receivers_y_per_shot);
    }
    if (n_receivers_x_per_shot > 0 && nt > 0) {
      add_to_wavefield(vx + si, receivers_x_i + rix,
                       grad_r_x + t * n_shots * n_receivers_x_per_shot + rix,
                       n_receivers_x_per_shot);
    }
    for (t = start_t - 1; t >= start_t - nt; --t) {
      int64_t store_i = (t / step_ratio) * n_shots * ny * nx + shot * ny * nx;
      if ((start_t - 1 - t) & 1) {
        backward_shot(
            lamb_shot, mu_shot, mu_yx_shot, buoyancy_y_shot, buoyancy_x_shot,
            grad_r_y + t * n_shots * n_receivers_y_per_shot + riy,
            grad_r_x + t * n_shots * n_receivers_x_per_shot + rix,
            grad_r_p + t * n_shots * n_receivers_p_per_shot + rip, vy + si,
            vx + si, sigmayy + si, sigmaxy + si, sigmaxx + si, m_vyy + si,
            m_vyx + si, m_vxy + si, m_vxx + si, m_sigmayyyn + si,
            m_sigmaxyyn + si, m_sigmaxyxn + si, m_sigmaxxxn + si,
            m_sigmayyy + si, m_sigmaxyy + si, m_sigmaxyx + si, m_sigmaxxx + si,
            dvydbuoyancy + store_i, dvxdbuoyancy + store_i,
            dvydy_store + store_i, dvxdx_store + store_i,
            dvydxdvxdy_store + store_i,
            grad_f_y + t * n_shots * n_sources_y_per_shot + siy,
            grad_f_x + t * n_shots * n_sources_x_per_shot + six,
            grad_lamb_thread + grad_lamb_i, grad_mu_thread + grad_mu_i,
            grad_mu_yx_thread + grad_mu_i,
            grad_buoyancy_y_thread + grad_buoyancy_i,
            grad_buoyancy_x_thread + grad_buoyancy_i, ay, ayh, ax, axh, by, byh,
            bx, bxh, sources_y_i + siy, sources_x_i + six, receivers_y_i + riy,
            receivers_x_i + rix, receivers_p_i + rip, rdy, rdx, dt, ny, nx,
            n_sources_y_per_shot, n_sources_x_per_shot, n_receivers_y_per_shot,
            n_receivers_x_per_shot, n_receivers_p_per_shot, step_ratio,
            lamb_requires_grad && ((t % step_ratio) == 0),
            mu_requires_grad && ((t % step_ratio) == 0),
            buoyancy_requires_grad && ((t % step_ratio) == 0), pml_y0, pml_y1,
            pml_x0, pml_x1);
      } else {
        backward_shot(
            lamb_shot, mu_shot, mu_yx_shot, buoyancy_y_shot, buoyancy_x_shot,
            grad_r_y + t * n_shots * n_receivers_y_per_shot + riy,
            grad_r_x + t * n_shots * n_receivers_x_per_shot + rix,
            grad_r_p + t * n_shots * n_receivers_p_per_shot + rip, vy + si,
            vx + si, sigmayy + si, sigmaxy + si, sigmaxx + si, m_vyy + si,
            m_vyx + si, m_vxy + si, m_vxx + si, m_sigmayyy + si,
            m_sigmaxyy + si, m_sigmaxyx + si, m_sigmaxxx + si, m_sigmayyyn + si,
            m_sigmaxyyn + si, m_sigmaxyxn + si, m_sigmaxxxn + si,
            dvydbuoyancy + store_i, dvxdbuoyancy + store_i,
            dvydy_store + store_i, dvxdx_store + store_i,
            dvydxdvxdy_store + store_i,
            grad_f_y + t * n_shots * n_sources_y_per_shot + siy,
            grad_f_x + t * n_shots * n_sources_x_per_shot + six,
            grad_lamb_thread + grad_lamb_i, grad_mu_thread + grad_mu_i,
            grad_mu_yx_thread + grad_mu_i,
            grad_buoyancy_y_thread + grad_buoyancy_i,
            grad_buoyancy_x_thread + grad_buoyancy_i, ay, ayh, ax, axh, by, byh,
            bx, bxh, sources_y_i + siy, sources_x_i + six, receivers_y_i + riy,
            receivers_x_i + rix, receivers_p_i + rip, rdy, rdx, dt, ny, nx,
            n_sources_y_per_shot, n_sources_x_per_shot, n_receivers_y_per_shot,
            n_receivers_x_per_shot, n_receivers_p_per_shot, step_ratio,
            lamb_requires_grad && ((t % step_ratio) == 0),
            mu_requires_grad && ((t % step_ratio) == 0),
            buoyancy_requires_grad && ((t % step_ratio) == 0), pml_y0, pml_y1,
            pml_x0, pml_x1);
      }
    }
  }
#ifdef _OPENMP
  if (lamb_requires_grad && !lamb_batched && n_threads > 1) {
    combine_grad_elastic(grad_lamb, grad_lamb_thread, n_threads, ny, nx);
  }
  if (mu_requires_grad && !mu_batched && n_threads > 1) {
    combine_grad_elastic(grad_mu, grad_mu_thread, n_threads, ny, nx);
    combine_grad_elastic(grad_mu_yx, grad_mu_yx_thread, n_threads, ny, nx);
  }
  if (buoyancy_requires_grad && !buoyancy_batched && n_threads > 1) {
    combine_grad_elastic(grad_buoyancy_y, grad_buoyancy_y_thread, n_threads, ny,
                         nx);
    combine_grad_elastic(grad_buoyancy_x, grad_buoyancy_x_thread, n_threads, ny,
                         nx);
  }
#endif /* _OPENMP */
}

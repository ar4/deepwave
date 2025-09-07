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
 * implemented using the W-AFDA method of Kristek et al. (2002), and the
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
 * | vx  sii | vx  sii | vx  sii
 * | SXY=VY==|=SXY=VY==|=SXY vy
 * v [-------------------]------
 * y VX  SII | VX  SII | VX  sii
 *   SXY VY  | SXY VY  | SXY vy
 *   [-------------------]------
 *   VX  SII | VX  SII | VX  sii
 *   SXY=VY==|=SXY=VY==|=SXY vy
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>

#include "common.h"
#include "common_cpu.h"

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

#define FORWARD_KERNEL_V(pml_y, pml_x, buoyancy_requires_grad)                 \
  {                                                                            \
    if (pml_y == 0) {                                                          \
      y_begin_y = 0;                                                           \
      y_end_y = pml_y0;                                                        \
      y_begin_x = 1;                                                           \
      y_end_x = pml_y0 + 1;                                                    \
    } else if (pml_y == 2) {                                                   \
      y_begin_y = pml_y1;                                                      \
      y_end_y = ny;                                                            \
      y_begin_x = pml_y1;                                                      \
      y_end_x = ny;                                                            \
    } else {                                                                   \
      y_begin_y = pml_y0;                                                      \
      y_end_y = pml_y1;                                                        \
      y_begin_x = pml_y0 + 1;                                                  \
      y_end_x = pml_y1;                                                        \
    }                                                                          \
    if (pml_x == 0) {                                                          \
      x_begin_y = 0;                                                           \
      x_end_y = pml_x0;                                                        \
      x_begin_x = 0;                                                           \
      x_end_x = pml_x0;                                                        \
    } else if (pml_x == 2) {                                                   \
      x_begin_y = pml_x1 - 1;                                                  \
      x_end_y = nx - 1;                                                        \
      x_begin_x = pml_x1;                                                      \
      x_end_x = nx;                                                            \
    } else {                                                                   \
      x_begin_y = pml_x0;                                                      \
      x_end_y = pml_x1 - 1;                                                    \
      x_begin_x = pml_x0;                                                      \
      x_end_x = pml_x1;                                                        \
    }                                                                          \
    for (y = y_begin_y; y < y_end_y; ++y) {                                    \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_y; x < x_end_y; ++x) {                                  \
        int64_t i = yi + x, j, k;                                              \
        DW_DTYPE dsigmayydy = 0;                                               \
        DW_DTYPE dsigmaxydx = 0;                                               \
                                                                               \
        /* dsigmaxydx */                                                       \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_x == 0 && x == j) {                                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmaxydx += fd_coeffs2x[1 + k] * sigmaxy[i - j + 1 + k];       \
            }                                                                  \
          } else if (pml_x == 2 && x == nx - 2 - j) {                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmaxydx -= fd_coeffs2x[1 + k] * sigmaxy[i + j - k];           \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_x == 1 || (x > A / 2 - 2 && x < nx - 2 - A / 2 + 2)) {         \
          for (k = 0; k < A / 2; ++k) {                                        \
            dsigmaxydx +=                                                      \
                fd_coeffsx[k] * (sigmaxy[i + 1 + k] - sigmaxy[i - k]);         \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* dsigmayydy */                                                       \
        for (j = 0; j < A / 2; ++j) {                                          \
          if (pml_y == 0 && y == j) {                                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmayydy +=                                                    \
                  fd_coeffs1y[j][1 + k] * sigmayy[i + (1 - j + k) * nx];       \
            }                                                                  \
          } else if (pml_y == 2 && y == ny - 1 - j) {                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmayydy -= fd_coeffs1y[j][1 + k] * sigmayy[i + (j - k) * nx]; \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_y == 1 || (y > A / 2 - 1 && y < ny - 1 - A / 2 + 1)) {         \
          for (k = 0; k < A / 2; ++k) {                                        \
            dsigmayydy += fd_coeffsy[k] *                                      \
                          (sigmayy[i + (1 + k) * nx] - sigmayy[i - k * nx]);   \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_sigmayyy[i] = ayh[y] * m_sigmayyy[i] + byh[y] * dsigmayydy;        \
          dsigmayydy += m_sigmayyy[i];                                         \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_sigmaxyx[i] = axh[x] * m_sigmaxyx[i] + bxh[x] * dsigmaxydx;        \
          dsigmaxydx += m_sigmaxyx[i];                                         \
        }                                                                      \
        {                                                                      \
          DW_DTYPE buoyancyyhxh;                                               \
          if (pml_y == 2 && y == ny - 1) {                                     \
            buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1]) / 2;                \
          } else {                                                             \
            buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1] + buoyancy[i + nx] + \
                            buoyancy[i + nx + 1]) /                            \
                           4;                                                  \
          }                                                                    \
          vy[i] += buoyancyyhxh * dt * (dsigmayydy + dsigmaxydx);              \
          if (buoyancy_requires_grad) {                                        \
            dvydbuoyancy[i] = dt * (dsigmayydy + dsigmaxydx);                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (y = y_begin_x; y < y_end_x; ++y) {                                    \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_x; x < x_end_x; ++x) {                                  \
        int64_t i = yi + x, j, k;                                              \
        DW_DTYPE dsigmaxydy = 0;                                               \
        DW_DTYPE dsigmaxxdx = 0;                                               \
                                                                               \
        /* dsigmaxydy */                                                       \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_y == 0 && y == 1 + j) {                                      \
            for (k = 0; k < A; ++k) {                                          \
              dsigmaxydy += fd_coeffs2y[1 + k] * sigmaxy[i + (-j + k) * nx];   \
            }                                                                  \
          } else if (pml_y == 2 && y == ny - 1 - j) {                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmaxydy -=                                                    \
                  fd_coeffs2y[1 + k] * sigmaxy[i + (j - k - 1) * nx];          \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_y == 1 || (y > 1 + A / 2 - 2 && y < ny - 1 - A / 2 + 2)) {     \
          for (k = 0; k < A / 2; ++k) {                                        \
            dsigmaxydy += fd_coeffsy[k] *                                      \
                          (sigmaxy[i + k * nx] - sigmaxy[i - (k + 1) * nx]);   \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* dsigmaxxdx */                                                       \
        for (j = 0; j < A / 2; ++j) {                                          \
          if (pml_x == 0 && x == j) {                                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmaxxdx += fd_coeffs1x[j][1 + k] * sigmaxx[i - j + k];        \
            }                                                                  \
          } else if (pml_x == 2 && x == nx - 1 - j) {                          \
            for (k = 0; k < A; ++k) {                                          \
              dsigmaxxdx -= fd_coeffs1x[j][1 + k] * sigmaxx[i + (j - k - 1)];  \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_x == 1 || (x > A / 2 - 1 && x < nx - 1 - A / 2 + 1)) {         \
          for (k = 0; k < A / 2; ++k) {                                        \
            dsigmaxxdx +=                                                      \
                fd_coeffsx[k] * (sigmaxx[i + k] - sigmaxx[i - (1 + k)]);       \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_sigmaxyy[i] = ay[y] * m_sigmaxyy[i] + by[y] * dsigmaxydy;          \
          dsigmaxydy += m_sigmaxyy[i];                                         \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_sigmaxxx[i] = ax[x] * m_sigmaxxx[i] + bx[x] * dsigmaxxdx;          \
          dsigmaxxdx += m_sigmaxxx[i];                                         \
        }                                                                      \
        vx[i] += buoyancy[i] * dt * (dsigmaxxdx + dsigmaxydy);                 \
        if (buoyancy_requires_grad) {                                          \
          dvxdbuoyancy[i] = dt * (dsigmaxxdx + dsigmaxydy);                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define FORWARD_KERNEL_SIGMA(pml_y, pml_x, lamb_requires_grad,                 \
                             mu_requires_grad)                                 \
  {                                                                            \
    if (pml_y == 0) {                                                          \
      y_begin_ii = 1;                                                          \
      y_end_ii = pml_y0 + 1;                                                   \
      y_begin_xy = 1;                                                          \
      y_end_xy = pml_y0 + 1;                                                   \
    } else if (pml_y == 2) {                                                   \
      y_begin_ii = pml_y1;                                                     \
      y_end_ii = ny;                                                           \
      y_begin_xy = pml_y1 - 1;                                                 \
      y_end_xy = ny - 1;                                                       \
    } else {                                                                   \
      y_begin_ii = pml_y0 + 1;                                                 \
      y_end_ii = pml_y1;                                                       \
      y_begin_xy = pml_y0 + 1;                                                 \
      y_end_xy = pml_y1 - 1;                                                   \
    }                                                                          \
    if (pml_x == 0) {                                                          \
      x_begin_ii = 0;                                                          \
      x_end_ii = pml_x0;                                                       \
      x_begin_xy = 1;                                                          \
      x_end_xy = pml_x0 + 1;                                                   \
    } else if (pml_x == 2) {                                                   \
      x_begin_ii = pml_x1 - 1;                                                 \
      x_end_ii = nx - 1;                                                       \
      x_begin_xy = pml_x1 - 1;                                                 \
      x_end_xy = nx - 1;                                                       \
    } else {                                                                   \
      x_begin_ii = pml_x0;                                                     \
      x_end_ii = pml_x1 - 1;                                                   \
      x_begin_xy = pml_x0 + 1;                                                 \
      x_end_xy = pml_x1 - 1;                                                   \
    }                                                                          \
                                                                               \
    for (y = y_begin_ii; y < y_end_ii; ++y) {                                  \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_ii; x < x_end_ii; ++x) {                                \
        int64_t i = yi + x, j, k;                                              \
        DW_DTYPE dvydy = 0;                                                    \
        DW_DTYPE dvxdx = 0;                                                    \
                                                                               \
        /* dvydy */                                                            \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_y == 0 && y == 1 + j) {                                      \
            for (k = 0; k <= A; ++k) {                                         \
              dvydy += fd_coeffs2y[k] * vy[i + (-j + k - 1) * nx];             \
            }                                                                  \
          } else if (pml_y == 2 && y == ny - 1 - j) {                          \
            for (k = 0; k <= A; ++k) {                                         \
              dvydy -= fd_coeffs2y[k] * vy[i + (j - k) * nx];                  \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_y == 1 || (y > 1 + A / 2 - 2 && y < ny - 1 - A / 2 + 2)) {     \
          for (k = 0; k < A / 2; ++k) {                                        \
            dvydy += fd_coeffsy[k] * (vy[i + k * nx] - vy[i - (k + 1) * nx]);  \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* dvxdx */                                                            \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_x == 0 && x == j) {                                          \
            for (k = 0; k <= A; ++k) {                                         \
              dvxdx += fd_coeffs2x[k] * vx[i - j + k];                         \
            }                                                                  \
          } else if (pml_x == 2 && x == nx - 2 - j) {                          \
            for (k = 0; k <= A; ++k) {                                         \
              dvxdx -= fd_coeffs2x[k] * vx[i + j - k + 1];                     \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_x == 1 || (x > A / 2 - 2 && x < nx - 2 - A / 2 + 2)) {         \
          for (k = 0; k < A / 2; ++k) {                                        \
            dvxdx += fd_coeffsx[k] * (vx[i + 1 + k] - vx[i - k]);              \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_vyy[i] = ay[y] * m_vyy[i] + by[y] * dvydy;                         \
          dvydy += m_vyy[i];                                                   \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_vxx[i] = axh[x] * m_vxx[i] + bxh[x] * dvxdx;                       \
          dvxdx += m_vxx[i];                                                   \
        }                                                                      \
        {                                                                      \
          DW_DTYPE lambyxh = (lamb[i] + lamb[i + 1]) / 2;                      \
          DW_DTYPE muyxh = (mu[i] + mu[i + 1]) / 2;                            \
          sigmayy[i] +=                                                        \
              dt * ((lambyxh + 2 * muyxh) * dvydy + lambyxh * dvxdx);          \
          sigmaxx[i] +=                                                        \
              dt * ((lambyxh + 2 * muyxh) * dvxdx + lambyxh * dvydy);          \
          if (lamb_requires_grad || mu_requires_grad) {                        \
            dvydy_store[i] = dt * dvydy;                                       \
            dvxdx_store[i] = dt * dvxdx;                                       \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (y = y_begin_xy; y < y_end_xy; ++y) {                                  \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_xy; x < x_end_xy; ++x) {                                \
        int64_t i = yi + x, j, jp, k;                                          \
        DW_DTYPE dvydx = 0;                                                    \
        DW_DTYPE dvxdy = 0;                                                    \
                                                                               \
        /* dvxdy */                                                            \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_y == 0 && y == 1 + j) {                                      \
            DW_DTYPE dvydxp = 0;                                               \
            for (jp = 0; jp < A / 2 - 1; ++jp) {                               \
              if (pml_x == 0 && x == 1 + jp) {                                 \
                for (k = 0; k <= A; ++k) {                                     \
                  dvydxp +=                                                    \
                      fd_coeffs2x[k] * vy[i - (j + 1) * nx - (jp + 1) + k];    \
                }                                                              \
              } else if (pml_x == 2 && x == nx - 2 - jp) {                     \
                for (k = 0; k <= A; ++k) {                                     \
                  dvydxp -= fd_coeffs2x[k] * vy[i - (j + 1) * nx + jp - k];    \
                }                                                              \
              }                                                                \
            }                                                                  \
            if (pml_x == 1 || (x > 1 + A / 2 - 2 && x < nx - 2 - A / 2 + 2)) { \
              for (k = 0; k < A / 2; ++k) {                                    \
                dvydxp += fd_coeffsx[k] * (vy[i - (j + 1) * nx + k] -          \
                                           vy[i - (j + 1) * nx - k - 1]);      \
              }                                                                \
            }                                                                  \
            dvxdy = -fd_coeffs3y[0] * dvydxp;                                  \
            for (k = 1; k <= A + 1; ++k) {                                     \
              dvxdy += fd_coeffs3y[k] * vx[i + (-j + k - 1) * nx];             \
            }                                                                  \
          } else if (pml_y == 2 && y == ny - 2 - j) {                          \
            DW_DTYPE dvydxp = 0;                                               \
            for (jp = 0; jp < A / 2 - 1; ++jp) {                               \
              if (pml_x == 0 && x == 1 + jp) {                                 \
                for (k = 0; k <= A; ++k) {                                     \
                  dvydxp +=                                                    \
                      fd_coeffs2x[k] * vy[i + (j + 1) * nx - (jp + 1) + k];    \
                }                                                              \
              } else if (pml_x == 2 && x == nx - 2 - jp) {                     \
                for (k = 0; k <= A; ++k) {                                     \
                  dvydxp -= fd_coeffs2x[k] * vy[i + (j + 1) * nx + jp - k];    \
                }                                                              \
              }                                                                \
            }                                                                  \
            if (pml_x == 1 || (x > 1 + A / 2 - 2 && x < nx - 2 - A / 2 + 2)) { \
              for (k = 0; k < A / 2; ++k) {                                    \
                dvydxp += fd_coeffsx[k] * (vy[i + (j + 1) * nx + k] -          \
                                           vy[i + (j + 1) * nx - k - 1]);      \
              }                                                                \
            }                                                                  \
            dvxdy = fd_coeffs3y[0] * dvydxp;                                   \
            for (k = 1; k <= A + 1; ++k) {                                     \
              dvxdy -= fd_coeffs3y[k] * vx[i + (j - k + 2) * nx];              \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_y == 1 || (y > 1 + A / 2 - 2 && y < ny - 2 - A / 2 + 2)) {     \
          for (k = 0; k < A / 2; ++k) {                                        \
            dvxdy += fd_coeffsy[k] * (vx[i + (k + 1) * nx] - vx[i - k * nx]);  \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* dvydx */                                                            \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_x == 0 && x == 1 + j) {                                      \
            DW_DTYPE dvxdyp = 0;                                               \
            for (jp = 0; jp < A / 2 - 1; ++jp) {                               \
              if (pml_y == 0 && y == 1 + jp) {                                 \
                for (k = 0; k <= A; ++k) {                                     \
                  dvxdyp += fd_coeffs2y[k] * vx[i - (j + 1) + (-jp + k) * nx]; \
                }                                                              \
              } else if (pml_y == 2 && y == ny - 2 - jp) {                     \
                for (k = 0; k <= A; ++k) {                                     \
                  dvxdyp -=                                                    \
                      fd_coeffs2y[k] * vx[i - (j + 1) + ((jp + 1) - k) * nx];  \
                }                                                              \
              }                                                                \
            }                                                                  \
            if (pml_y == 1 || (y > 1 + A / 2 - 2 && y < ny - 2 - A / 2 + 2)) { \
              for (k = 0; k < A / 2; ++k) {                                    \
                dvxdyp += fd_coeffsy[k] * (vx[i - (j + 1) + (k + 1) * nx] -    \
                                           vx[i - (j + 1) - k * nx]);          \
              }                                                                \
            }                                                                  \
            dvydx = -fd_coeffs3x[0] * dvxdyp;                                  \
            for (k = 1; k <= A + 1; ++k) {                                     \
              dvydx += fd_coeffs3x[k] * vy[i + (-j + k - 2)];                  \
            }                                                                  \
          } else if (pml_x == 2 && x == nx - 2 - j) {                          \
            DW_DTYPE dvxdyp = 0;                                               \
            for (jp = 0; jp < A / 2 - 1; ++jp) {                               \
              if (pml_y == 0 && y == 1 + jp) {                                 \
                for (k = 0; k <= A; ++k) {                                     \
                  dvxdyp += fd_coeffs2y[k] * vx[i + (j + 1) + (-jp + k) * nx]; \
                }                                                              \
              } else if (pml_y == 2 && y == ny - 2 - jp) {                     \
                for (k = 0; k <= A; ++k) {                                     \
                  dvxdyp -=                                                    \
                      fd_coeffs2y[k] * vx[i + (j + 1) + (jp - k + 1) * nx];    \
                }                                                              \
              }                                                                \
            }                                                                  \
            if (pml_y == 1 || (y > 1 + A / 2 - 2 && y < ny - 2 - A / 2 + 2)) { \
              for (k = 0; k < A / 2; ++k) {                                    \
                dvxdyp += fd_coeffsy[k] * (vx[i + (j + 1) + (k + 1) * nx] -    \
                                           vx[i + (j + 1) + (-k) * nx]);       \
              }                                                                \
            }                                                                  \
            dvydx = fd_coeffs3x[0] * dvxdyp;                                   \
            for (k = 1; k <= A + 1; ++k) {                                     \
              dvydx -= fd_coeffs3x[k] * vy[i + (j - k + 1)];                   \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        if (pml_x == 1 || (x > 1 + A / 2 - 2 && x < nx - 2 - A / 2 + 2)) {     \
          for (k = 0; k < A / 2; ++k) {                                        \
            dvydx += fd_coeffsx[k] * (vy[i + k] - vy[i - k - 1]);              \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_vxy[i] = ayh[y] * m_vxy[i] + byh[y] * dvxdy;                       \
          dvxdy += m_vxy[i];                                                   \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_vyx[i] = ax[x] * m_vyx[i] + bx[x] * dvydx;                         \
          dvydx += m_vyx[i];                                                   \
        }                                                                      \
        {                                                                      \
          DW_DTYPE muyhx = (mu[i] + mu[i + nx]) / 2;                           \
          sigmaxy[i] += dt * muyhx * (dvydx + dvxdy);                          \
          if (mu_requires_grad) {                                              \
            dvydxdvxdy_store[i] = dt * (dvydx + dvxdy);                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define BACKWARD_KERNEL_SIGMA(pml_y, pml_x)                                    \
  {                                                                            \
    if (pml_y == 0) {                                                          \
      y_begin_y = 0;                                                           \
      y_end_y = spml_y0;                                                       \
      y_begin_x = 1;                                                           \
      y_end_x = spml_y0 + 1;                                                   \
    } else if (pml_y == 2) {                                                   \
      y_begin_y = spml_y1;                                                     \
      y_end_y = ny;                                                            \
      y_begin_x = spml_y1;                                                     \
      y_end_x = ny;                                                            \
    } else {                                                                   \
      y_begin_y = spml_y0;                                                     \
      y_end_y = spml_y1;                                                       \
      y_begin_x = spml_y0 + 1;                                                 \
      y_end_x = spml_y1;                                                       \
    }                                                                          \
    if (pml_x == 0) {                                                          \
      x_begin_y = 0;                                                           \
      x_end_y = spml_x0;                                                       \
      x_begin_x = 0;                                                           \
      x_end_x = spml_x0;                                                       \
    } else if (pml_x == 2) {                                                   \
      x_begin_y = spml_x1 - 1;                                                 \
      x_end_y = nx - 1;                                                        \
      x_begin_x = spml_x1;                                                     \
      x_end_x = nx;                                                            \
    } else {                                                                   \
      x_begin_y = spml_x0;                                                     \
      x_end_y = spml_x1 - 1;                                                   \
      x_begin_x = spml_x0;                                                     \
      x_end_x = spml_x1;                                                       \
    }                                                                          \
                                                                               \
    for (y = y_begin_y; y < y_end_y; ++y) {                                    \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_y; x < x_end_y; ++x) {                                  \
        int64_t i = yi + x, j, jp, k;                                          \
                                                                               \
        /* from sigmayy/sigmaxx edges */                                       \
        for (k = 0; k <= A; ++k) {                                             \
          for (j = 0; j < A / 2 - 1; ++j) {                                    \
            if (pml_y == 0 && y == 1 + j + (-j + k - 1)) {                     \
              DW_DTYPE lambyxh = (lamb[i - (-j + k - 1) * nx] +                \
                                  lamb[i + 1 - (-j + k - 1) * nx]) /           \
                                 2;                                            \
              DW_DTYPE muyxh = (mu[i - (-j + k - 1) * nx] +                    \
                                mu[i + 1 - (-j + k - 1) * nx]) /               \
                               2;                                              \
              vy[i] += fd_coeffs2y[k] *                                        \
                       (dt * (1 + by[y - (-j + k - 1)]) *                      \
                            ((lambyxh + 2 * muyxh) *                           \
                                 sigmayy[i - (-j + k - 1) * nx] +              \
                             lambyxh * sigmaxx[i - (-j + k - 1) * nx]) +       \
                        by[y - (-j + k - 1)] * m_vyy[i - (-j + k - 1) * nx]);  \
            } else if (pml_y == 2 && y == ny - 1 - j + (j - k)) {              \
              DW_DTYPE lambyxh =                                               \
                  (lamb[i - (j - k) * nx] + lamb[i + 1 - (j - k) * nx]) / 2;   \
              DW_DTYPE muyxh =                                                 \
                  (mu[i - (j - k) * nx] + mu[i + 1 - (j - k) * nx]) / 2;       \
              vy[i] -=                                                         \
                  fd_coeffs2y[k] *                                             \
                  (dt * (1 + by[y - (j - k)]) *                                \
                       ((lambyxh + 2 * muyxh) * sigmayy[i - (j - k) * nx] +    \
                        lambyxh * sigmaxx[i - (j - k) * nx]) +                 \
                   by[y - (j - k)] * m_vyy[i - (j - k) * nx]);                 \
            }                                                                  \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* from sigmayy/sigmaxx centre */                                      \
        for (k = 0; k < A / 2; ++k) {                                          \
          if (pml_y == 1 ||                                                    \
              (y > 1 + A / 2 - 2 + k && y < ny - 1 - A / 2 + 2 + k)) {         \
            DW_DTYPE lambyxh = (lamb[i - k * nx] + lamb[i + 1 - k * nx]) / 2;  \
            DW_DTYPE muyxh = (mu[i - k * nx] + mu[i + 1 - k * nx]) / 2;        \
            vy[i] += fd_coeffsy[k] *                                           \
                     (dt * (1 + by[y - k]) *                                   \
                          ((lambyxh + 2 * muyxh) * sigmayy[i - k * nx] +       \
                           lambyxh * sigmaxx[i - k * nx]) +                    \
                      by[y - k] * m_vyy[i - k * nx]);                          \
          }                                                                    \
          if (pml_y == 1 || (y > 1 + A / 2 - 2 - (k + 1) &&                    \
                             y < ny - 1 - A / 2 + 2 - (k + 1))) {              \
            DW_DTYPE lambyxh =                                                 \
                (lamb[i + (k + 1) * nx] + lamb[i + 1 + (k + 1) * nx]) / 2;     \
            DW_DTYPE muyxh =                                                   \
                (mu[i + (k + 1) * nx] + mu[i + 1 + (k + 1) * nx]) / 2;         \
            vy[i] -= fd_coeffsy[k] *                                           \
                     (dt * (1 + by[y + k + 1]) *                               \
                          ((lambyxh + 2 * muyxh) * sigmayy[i + (k + 1) * nx] + \
                           lambyxh * sigmaxx[i + (k + 1) * nx]) +              \
                      by[y + k + 1] * m_vyy[i + (k + 1) * nx]);                \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* from sigmaxy dvxdy */                                               \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_y == 0 && y == 1 + j - (j + 1)) {                            \
            int64_t y2 = y + (j + 1);                                          \
            for (k = 0; k <= A; ++k) {                                         \
              for (jp = 0; jp < A / 2 - 1; ++jp) {                             \
                if (pml_x == 0 && x == 1 + jp - (jp + 1) + k) {                \
                  int64_t i2 = i - (-(j + 1) * nx - (jp + 1) + k);             \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vy[i] += fd_coeffs2x[k] * (-fd_coeffs3y[0]) *                \
                           (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +         \
                            byh[y2] * m_vxy[i2]);                              \
                } else if (pml_x == 2 && x == nx - 2 - jp + jp - k) {          \
                  int64_t i2 = i - (-(j + 1) * nx + jp - k);                   \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vy[i] -= fd_coeffs2x[k] * (-fd_coeffs3y[0]) *                \
                           (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +         \
                            byh[y2] * m_vxy[i2]);                              \
                }                                                              \
              }                                                                \
            }                                                                  \
            for (k = 0; k < A / 2; ++k) {                                      \
              if (pml_x == 1 ||                                                \
                  (x > 1 + A / 2 - 2 + k && x < nx - 2 - A / 2 + 2 + k)) {     \
                int64_t i2 = i - (-(j + 1) * nx + k);                          \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vy[i] += fd_coeffsx[k] * (-fd_coeffs3y[0]) *                   \
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +           \
                          byh[y2] * m_vxy[i2]);                                \
              }                                                                \
              if (pml_x == 1 || (x > 1 + A / 2 - 2 - k - 1 &&                  \
                                 x < nx - 2 - A / 2 + 2 - k - 1)) {            \
                int64_t i2 = i - (-(j + 1) * nx - k - 1);                      \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vy[i] -= fd_coeffsx[k] * (-fd_coeffs3y[0]) *                   \
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +           \
                          byh[y2] * m_vxy[i2]);                                \
              }                                                                \
            }                                                                  \
                                                                               \
          } else if (pml_y == 2 && y == ny - 2 - j + (j + 1)) {                \
            int64_t y2 = y - (j + 1);                                          \
            for (k = 0; k <= A; ++k) {                                         \
              for (jp = 0; jp < A / 2 - 1; ++jp) {                             \
                if (pml_x == 0 && x == 1 + jp - (jp + 1) + k) {                \
                  int64_t i2 = i - ((j + 1) * nx - (jp + 1) + k);              \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vy[i] += fd_coeffs2x[k] * (fd_coeffs3y[0]) *                 \
                           (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +         \
                            byh[y2] * m_vxy[i2]);                              \
                } else if (pml_x == 2 && x == nx - 2 - jp + jp - k) {          \
                  int64_t i2 = i - ((j + 1) * nx + jp - k);                    \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vy[i] -= fd_coeffs2x[k] * (fd_coeffs3y[0]) *                 \
                           (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +         \
                            byh[y2] * m_vxy[i2]);                              \
                }                                                              \
              }                                                                \
            }                                                                  \
            for (k = 0; k < A / 2; ++k) {                                      \
              if (pml_x == 1 ||                                                \
                  (x > 1 + A / 2 - 2 + k && x < nx - 2 - A / 2 + 2 + k)) {     \
                int64_t i2 = i - ((j + 1) * nx + k);                           \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vy[i] += fd_coeffsx[k] * (fd_coeffs3y[0]) *                    \
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +           \
                          byh[y2] * m_vxy[i2]);                                \
              }                                                                \
              if (pml_x == 1 || (x > 1 + A / 2 - 2 - k - 1 &&                  \
                                 x < nx - 2 - A / 2 + 2 - k - 1)) {            \
                int64_t i2 = i - ((j + 1) * nx - k - 1);                       \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vy[i] -= fd_coeffsx[k] * (fd_coeffs3y[0]) *                    \
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +           \
                          byh[y2] * m_vxy[i2]);                                \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* from sigmaxy dvydx */                                               \
        if (y > 0 && y < ny - 1) {                                             \
          for (k = 1; k <= A + 1; ++k) {                                       \
            for (j = 0; j < A / 2 - 1; ++j) {                                  \
              if (pml_x == 0 && x == 1 + j + (-j + k - 2)) {                   \
                int64_t x2 = x - (-j + k - 2);                                 \
                int64_t i2 = i - (-j + k - 2);                                 \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vy[i] += fd_coeffs3x[k] *                                      \
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +            \
                          bx[x2] * m_vyx[i2]);                                 \
              } else if (pml_x == 2 && x == nx - 2 - j + (j - k + 1)) {        \
                int64_t x2 = x - (j - k + 1);                                  \
                int64_t i2 = i - (j - k + 1);                                  \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vy[i] -= fd_coeffs3x[k] *                                      \
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +            \
                          bx[x2] * m_vyx[i2]);                                 \
              }                                                                \
            }                                                                  \
          }                                                                    \
          for (k = 0; k < A / 2; ++k) {                                        \
            if (pml_x == 1 ||                                                  \
                (x > 1 + A / 2 - 2 + k && x < nx - 2 - A / 2 + 2 + k)) {       \
              int64_t x2 = x - k;                                              \
              int64_t i2 = i - k;                                              \
              DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                     \
              vy[i] +=                                                         \
                  fd_coeffsx[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +   \
                                   bx[x2] * m_vyx[i2]);                        \
            }                                                                  \
            if (pml_x == 1 || (x > 1 + A / 2 - 2 - k - 1 &&                    \
                               x < nx - 2 - A / 2 + 2 - k - 1)) {              \
              int64_t x2 = x + k + 1;                                          \
              int64_t i2 = i + k + 1;                                          \
              DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                     \
              vy[i] -=                                                         \
                  fd_coeffsx[k] * (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +   \
                                   bx[x2] * m_vyx[i2]);                        \
            }                                                                  \
          }                                                                    \
        }                                                                      \
                                                                               \
        {                                                                      \
          DW_DTYPE buoyancyyhxh;                                               \
          if (pml_y == 2 && y == ny - 1) {                                     \
            buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1]) / 2;                \
          } else {                                                             \
            buoyancyyhxh = (buoyancy[i] + buoyancy[i + 1] + buoyancy[i + nx] + \
                            buoyancy[i + nx + 1]) /                            \
                           4;                                                  \
          }                                                                    \
                                                                               \
          if (pml_y == 0 || pml_y == 2) {                                      \
            m_sigmayyyn[i] =                                                   \
                buoyancyyhxh * dt * ayh[y] * vy[i] + ayh[y] * m_sigmayyy[i];   \
          }                                                                    \
          if (pml_x == 0 || pml_x == 2) {                                      \
            m_sigmaxyxn[i] =                                                   \
                buoyancyyhxh * dt * axh[x] * vy[i] + axh[x] * m_sigmaxyx[i];   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (y = y_begin_x; y < y_end_x; ++y) {                                    \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_x; x < x_end_x; ++x) {                                  \
        int64_t i = yi + x, j, k;                                              \
                                                                               \
        /* from sigmayy/sigmaxx edges */                                       \
        for (k = 0; k <= A; ++k) {                                             \
          for (j = 0; j < A / 2 - 1; ++j) {                                    \
            if (pml_x == 0 && x == j + (-j + k)) {                             \
              int64_t i2 = i - (-j + k);                                       \
              int64_t x2 = x - (-j + k);                                       \
              DW_DTYPE lambyxh = (lamb[i2] + lamb[i2 + 1]) / 2;                \
              DW_DTYPE muyxh = (mu[i2] + mu[i2 + 1]) / 2;                      \
              vx[i] +=                                                         \
                  fd_coeffs2x[k] * (dt * (1 + bxh[x2]) *                       \
                                        ((lambyxh + 2 * muyxh) * sigmaxx[i2] + \
                                         lambyxh * sigmayy[i2]) +              \
                                    bxh[x2] * m_vxx[i2]);                      \
            } else if (pml_x == 2 && x == nx - 2 - j + (j - k + 1)) {          \
              int64_t i2 = i - (j - k + 1);                                    \
              int64_t x2 = x - (j - k + 1);                                    \
              DW_DTYPE lambyxh = (lamb[i2] + lamb[i2 + 1]) / 2;                \
              DW_DTYPE muyxh = (mu[i2] + mu[i2 + 1]) / 2;                      \
              vx[i] -=                                                         \
                  fd_coeffs2x[k] * (dt * (1 + bxh[x2]) *                       \
                                        ((lambyxh + 2 * muyxh) * sigmaxx[i2] + \
                                         lambyxh * sigmayy[i2]) +              \
                                    bxh[x2] * m_vxx[i2]);                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* from sigmayy/sigmaxx centre */                                      \
        for (k = 0; k < A / 2; ++k) {                                          \
          if (pml_x == 1 ||                                                    \
              (x > A / 2 - 2 + 1 + k && x < nx - 2 - A / 2 + 2 + 1 + k)) {     \
            int64_t i2 = i - (1 + k);                                          \
            int64_t x2 = x - (1 + k);                                          \
            DW_DTYPE lambyxh = (lamb[i2] + lamb[i2 + 1]) / 2;                  \
            DW_DTYPE muyxh = (mu[i2] + mu[i2 + 1]) / 2;                        \
            vx[i] +=                                                           \
                fd_coeffsx[k] * (dt * (1 + bxh[x2]) *                          \
                                     ((lambyxh + 2 * muyxh) * sigmaxx[i2] +    \
                                      lambyxh * sigmayy[i2]) +                 \
                                 bxh[x2] * m_vxx[i2]);                         \
          }                                                                    \
          if (pml_x == 1 ||                                                    \
              (x > A / 2 - 2 - k && x < nx - 2 - A / 2 + 2 - k)) {             \
            int64_t i2 = i + k;                                                \
            int64_t x2 = x + k;                                                \
            DW_DTYPE lambyxh = (lamb[i2] + lamb[i2 + 1]) / 2;                  \
            DW_DTYPE muyxh = (mu[i2] + mu[i2 + 1]) / 2;                        \
            vx[i] -=                                                           \
                fd_coeffsx[k] * (dt * (1 + bxh[x2]) *                          \
                                     ((lambyxh + 2 * muyxh) * sigmaxx[i2] +    \
                                      lambyxh * sigmayy[i2]) +                 \
                                 bxh[x2] * m_vxx[i2]);                         \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* from sigmaxy dvydx */                                               \
        for (j = 0; j < A / 2 - 1; ++j) {                                      \
          if (pml_x == 0 && x == 1 + j - (j + 1)) {                            \
            int64_t x2 = x + (j + 1), jp;                                      \
            for (k = 0; k <= A; ++k) {                                         \
              for (jp = 0; jp < A / 2 - 1; ++jp) {                             \
                if (pml_y == 0 && y == 1 + jp - jp + k) {                      \
                  int64_t i2 = i - (-(j + 1) + (-jp + k) * nx);                \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vx[i] += fd_coeffs2y[k] * (-fd_coeffs3x[0]) *                \
                           (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +          \
                            bx[x2] * m_vyx[i2]);                               \
                } else if (pml_y == 2 && y == ny - 2 - jp + (jp + 1) - k) {    \
                  int64_t i2 = i - (-(j + 1) + ((jp + 1) - k) * nx);           \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vx[i] -= fd_coeffs2y[k] * (-fd_coeffs3x[0]) *                \
                           (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +          \
                            bx[x2] * m_vyx[i2]);                               \
                }                                                              \
              }                                                                \
            }                                                                  \
            for (k = 0; k < A / 2; ++k) {                                      \
              if (pml_y == 1 || (y > 1 + A / 2 - 2 + k + 1 &&                  \
                                 y < ny - 2 - A / 2 + 2 + k + 1)) {            \
                int64_t i2 = i - (-(j + 1) + (k + 1) * nx);                    \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vx[i] += fd_coeffsy[k] * (-fd_coeffs3x[0]) *                   \
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +            \
                          bx[x2] * m_vyx[i2]);                                 \
              }                                                                \
              if (pml_y == 1 ||                                                \
                  (y > 1 + A / 2 - 2 - k && y < ny - 2 - A / 2 + 2 - k)) {     \
                int64_t i2 = i - (-(j + 1) - k * nx);                          \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vx[i] -= fd_coeffsy[k] * (-fd_coeffs3x[0]) *                   \
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +            \
                          bx[x2] * m_vyx[i2]);                                 \
              }                                                                \
            }                                                                  \
                                                                               \
          } else if (pml_x == 2 && x == nx - 2 - j + (j + 1)) {                \
            int64_t x2 = x - (j + 1), jp;                                      \
            for (k = 0; k <= A; ++k) {                                         \
              for (jp = 0; jp < A / 2 - 1; ++jp) {                             \
                if (pml_y == 0 && y == 1 + jp - jp + k) {                      \
                  int64_t i2 = i - ((j + 1) + (-jp + k) * nx);                 \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vx[i] += fd_coeffs2y[k] * (fd_coeffs3x[0]) *                 \
                           (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +          \
                            bx[x2] * m_vyx[i2]);                               \
                } else if (pml_y == 2 && y == ny - 2 - jp + jp - k + 1) {      \
                  int64_t i2 = i - ((j + 1) + (jp - k + 1) * nx);              \
                  DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                 \
                  vx[i] -= fd_coeffs2y[k] * (fd_coeffs3x[0]) *                 \
                           (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +          \
                            bx[x2] * m_vyx[i2]);                               \
                }                                                              \
              }                                                                \
            }                                                                  \
            for (k = 0; k < A / 2; ++k) {                                      \
              if (pml_y == 1 || (y > 1 + A / 2 - 2 + k + 1 &&                  \
                                 y < ny - 2 - A / 2 + 2 + k + 1)) {            \
                int64_t i2 = i - ((j + 1) + (k + 1) * nx);                     \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vx[i] += fd_coeffsy[k] * (fd_coeffs3x[0]) *                    \
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +            \
                          bx[x2] * m_vyx[i2]);                                 \
              }                                                                \
              if (pml_y == 1 ||                                                \
                  (y > 1 + A / 2 - 2 - k && y < ny - 2 - A / 2 + 2 - k)) {     \
                int64_t i2 = i - ((j + 1) + (-k) * nx);                        \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vx[i] -= fd_coeffsy[k] * (fd_coeffs3x[0]) *                    \
                         (muyhx * dt * (1 + bx[x2]) * sigmaxy[i2] +            \
                          bx[x2] * m_vyx[i2]);                                 \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* from sigmaxy dvxdy */                                               \
        if (x > 0 && x < nx - 1) {                                             \
          for (k = 1; k <= A + 1; ++k) {                                       \
            for (j = 0; j < A / 2 - 1; ++j) {                                  \
              if (pml_y == 0 && y == 1 + j + (-j + k - 1)) {                   \
                int64_t y2 = y - (-j + k - 1);                                 \
                int64_t i2 = i - (-j + k - 1) * nx;                            \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vx[i] += fd_coeffs3y[k] *                                      \
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +           \
                          byh[y2] * m_vxy[i2]);                                \
              } else if (pml_y == 2 && y == ny - 2 - j + (j - k + 2)) {        \
                int64_t y2 = y - (j - k + 2);                                  \
                int64_t i2 = i - (j - k + 2) * nx;                             \
                DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                   \
                vx[i] -= fd_coeffs3y[k] *                                      \
                         (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +           \
                          byh[y2] * m_vxy[i2]);                                \
              }                                                                \
            }                                                                  \
          }                                                                    \
          for (k = 0; k < A / 2; ++k) {                                        \
            if (pml_y == 1 || (y > 1 + A / 2 - 2 + k + 1 &&                    \
                               y < ny - 2 - A / 2 + 2 + k + 1)) {              \
              int64_t y2 = y - (k + 1);                                        \
              int64_t i2 = i - (k + 1) * nx;                                   \
              DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                     \
              vx[i] +=                                                         \
                  fd_coeffsy[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +  \
                                   byh[y2] * m_vxy[i2]);                       \
            }                                                                  \
            if (pml_y == 1 ||                                                  \
                (y > 1 + A / 2 - 2 - k && y < ny - 2 - A / 2 + 2 - k)) {       \
              int64_t y2 = y + k;                                              \
              int64_t i2 = i + k * nx;                                         \
              DW_DTYPE muyhx = (mu[i2] + mu[i2 + nx]) / 2;                     \
              vx[i] -=                                                         \
                  fd_coeffsy[k] * (muyhx * dt * (1 + byh[y2]) * sigmaxy[i2] +  \
                                   byh[y2] * m_vxy[i2]);                       \
            }                                                                  \
          }                                                                    \
        }                                                                      \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_sigmaxyyn[i] =                                                     \
              buoyancy[i] * dt * ay[y] * vx[i] + ay[y] * m_sigmaxyy[i];        \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_sigmaxxxn[i] =                                                     \
              buoyancy[i] * dt * ax[x] * vx[i] + ax[x] * m_sigmaxxx[i];        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define BACKWARD_KERNEL_V(pml_y, pml_x)                                        \
  {                                                                            \
    if (pml_y == 0) {                                                          \
      y_begin_ii = 1;                                                          \
      y_end_ii = vpml_y0 + 1;                                                  \
      y_begin_xy = 1;                                                          \
      y_end_xy = vpml_y0 + 1;                                                  \
    } else if (pml_y == 2) {                                                   \
      y_begin_ii = vpml_y1;                                                    \
      y_end_ii = ny;                                                           \
      y_begin_xy = vpml_y1 - 1;                                                \
      y_end_xy = ny - 1;                                                       \
    } else {                                                                   \
      y_begin_ii = vpml_y0 + 1;                                                \
      y_end_ii = vpml_y1;                                                      \
      y_begin_xy = vpml_y0 + 1;                                                \
      y_end_xy = vpml_y1 - 1;                                                  \
    }                                                                          \
    if (pml_x == 0) {                                                          \
      x_begin_ii = 0;                                                          \
      x_end_ii = vpml_x0;                                                      \
      x_begin_xy = 1;                                                          \
      x_end_xy = vpml_x0 + 1;                                                  \
    } else if (pml_x == 2) {                                                   \
      x_begin_ii = vpml_x1 - 1;                                                \
      x_end_ii = nx - 1;                                                       \
      x_begin_xy = vpml_x1 - 1;                                                \
      x_end_xy = nx - 1;                                                       \
    } else {                                                                   \
      x_begin_ii = vpml_x0;                                                    \
      x_end_ii = vpml_x1 - 1;                                                  \
      x_begin_xy = vpml_x0 + 1;                                                \
      x_end_xy = vpml_x1 - 1;                                                  \
    }                                                                          \
    for (y = y_begin_ii; y < y_end_ii; ++y) {                                  \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_ii; x < x_end_ii; ++x) {                                \
        int64_t i = yi + x, k, j;                                              \
        DW_DTYPE lambyxh = (lamb[i] + lamb[i + 1]) / 2;                        \
        DW_DTYPE muyxh = (mu[i] + mu[i + 1]) / 2;                              \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_vyy[i] = (lambyxh + 2 * muyxh) * dt * ay[y] * sigmayy[i] +         \
                     lambyxh * dt * ay[y] * sigmaxx[i] + ay[y] * m_vyy[i];     \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_vxx[i] = (lambyxh + 2 * muyxh) * dt * axh[x] * sigmaxx[i] +        \
                     lambyxh * dt * axh[x] * sigmayy[i] + axh[x] * m_vxx[i];   \
        }                                                                      \
                                                                               \
        /* dsigmayydy */                                                       \
        for (k = 0; k < A; ++k) {                                              \
          for (j = 0; j < A / 2; ++j) {                                        \
            if (pml_y == 0 && y == j + (1 - j + k)) {                          \
              int64_t i2 = i - (1 - j + k) * nx;                               \
              int64_t y2 = y - (1 - j + k);                                    \
              DW_DTYPE buoyancyyhxh;                                           \
              if (pml_y == 2 && y2 == ny - 1) {                                \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;          \
              } else {                                                         \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +              \
                                buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /   \
                               4;                                              \
              }                                                                \
              sigmayy[i] += fd_coeffs1y[j][1 + k] *                            \
                            (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +      \
                             byh[y2] * m_sigmayyy[i2]);                        \
            } else if (pml_y == 2 && y == ny - 1 - j + (j - k)) {              \
              int64_t i2 = i - (j - k) * nx;                                   \
              int64_t y2 = y - (j - k);                                        \
              DW_DTYPE buoyancyyhxh;                                           \
              if (pml_y == 2 && y2 == ny - 1) {                                \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;          \
              } else {                                                         \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +              \
                                buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /   \
                               4;                                              \
              }                                                                \
              sigmayy[i] -= fd_coeffs1y[j][1 + k] *                            \
                            (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +      \
                             byh[y2] * m_sigmayyy[i2]);                        \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        for (k = 0; k < A / 2; ++k) {                                          \
          if (pml_y == 1 ||                                                    \
              (y > A / 2 - 1 + (1 + k) && y < ny - 1 - A / 2 + 1 + (1 + k))) { \
            int64_t i2 = i - (1 + k) * nx;                                     \
            int64_t y2 = y - (1 + k);                                          \
            DW_DTYPE buoyancyyhxh;                                             \
            if (pml_y == 2 && y2 == ny - 1) {                                  \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;            \
            } else {                                                           \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +                \
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /     \
                             4;                                                \
            }                                                                  \
            sigmayy[i] +=                                                      \
                fd_coeffsy[k] * (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +  \
                                 byh[y2] * m_sigmayyy[i2]);                    \
          }                                                                    \
          if (pml_y == 1 ||                                                    \
              (y > A / 2 - 1 - k && y < ny - 1 - A / 2 + 1 - k)) {             \
            int64_t i2 = i - (-k) * nx;                                        \
            int64_t y2 = y - (-k);                                             \
            DW_DTYPE buoyancyyhxh;                                             \
            if (pml_y == 2 && y2 == ny - 1) {                                  \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;            \
            } else {                                                           \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +                \
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /     \
                             4;                                                \
            }                                                                  \
            sigmayy[i] -=                                                      \
                fd_coeffsy[k] * (buoyancyyhxh * dt * (1 + byh[y2]) * vy[i2] +  \
                                 byh[y2] * m_sigmayyy[i2]);                    \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* dsigmaxxdx */                                                       \
        for (k = 0; k < A; ++k) {                                              \
          for (j = 0; j < A / 2; ++j) {                                        \
            if (pml_x == 0 && x == j + (-j + k)) {                             \
              int64_t i2 = i - (-j + k);                                       \
              int64_t x2 = x - (-j + k);                                       \
              sigmaxx[i] += fd_coeffs1x[j][1 + k] *                            \
                            (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +       \
                             bx[x2] * m_sigmaxxx[i2]);                         \
            } else if (pml_x == 2 && x == nx - 1 - j + (j - k - 1)) {          \
              int64_t i2 = i - (j - k - 1);                                    \
              int64_t x2 = x - (j - k - 1);                                    \
              sigmaxx[i] -= fd_coeffs1x[j][1 + k] *                            \
                            (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +       \
                             bx[x2] * m_sigmaxxx[i2]);                         \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        for (k = 0; k < A / 2; ++k) {                                          \
          if (pml_x == 1 ||                                                    \
              (x > A / 2 - 1 + (k) && x < nx - 1 - A / 2 + 1 + (k))) {         \
            int64_t i2 = i - (k);                                              \
            int64_t x2 = x - (k);                                              \
            sigmaxx[i] +=                                                      \
                fd_coeffsx[k] * (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +   \
                                 bx[x2] * m_sigmaxxx[i2]);                     \
          }                                                                    \
          if (pml_x == 1 ||                                                    \
              (x > A / 2 - 1 - (1 + k) && x < nx - 1 - A / 2 + 1 - (1 + k))) { \
            int64_t i2 = i + (1 + k);                                          \
            int64_t x2 = x + (1 + k);                                          \
            sigmaxx[i] -=                                                      \
                fd_coeffsx[k] * (buoyancy[i2] * dt * (1 + bx[x2]) * vx[i2] +   \
                                 bx[x2] * m_sigmaxxx[i2]);                     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (y = y_begin_xy; y < y_end_xy; ++y) {                                  \
      int64_t yi = y * nx;                                                     \
      for (x = x_begin_xy; x < x_end_xy; ++x) {                                \
        int64_t i = yi + x, k, j;                                              \
        DW_DTYPE muyhx = (mu[i] + mu[i + nx]) / 2;                             \
                                                                               \
        if (pml_y == 0 || pml_y == 2) {                                        \
          m_vxy[i] = muyhx * dt * ayh[y] * sigmaxy[i] + ayh[y] * m_vxy[i];     \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          m_vyx[i] = muyhx * dt * ax[x] * sigmaxy[i] + ax[x] * m_vyx[i];       \
        }                                                                      \
                                                                               \
        /* dsigmaxydx */                                                       \
        for (k = 0; k < A; ++k) {                                              \
          for (j = 0; j < A / 2 - 1; ++j) {                                    \
            if (pml_x == 0 && x == j - j + 1 + k) {                            \
              int64_t i2 = i - (-j + 1 + k);                                   \
              int64_t x2 = x - (-j + 1 + k);                                   \
              DW_DTYPE buoyancyyhxh;                                           \
              if (pml_y == 2 && y == ny - 1) {                                 \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;          \
              } else {                                                         \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +              \
                                buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /   \
                               4;                                              \
              }                                                                \
              sigmaxy[i] += fd_coeffs2x[1 + k] *                               \
                            (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +      \
                             bxh[x2] * m_sigmaxyx[i2]);                        \
            } else if (pml_x == 2 && x == nx - 2 - j + j - k) {                \
              int64_t i2 = i - (j - k);                                        \
              int64_t x2 = x - (j - k);                                        \
              DW_DTYPE buoyancyyhxh;                                           \
              if (pml_y == 2 && y == ny - 1) {                                 \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;          \
              } else {                                                         \
                buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +              \
                                buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /   \
                               4;                                              \
              }                                                                \
              sigmaxy[i] -= fd_coeffs2x[1 + k] *                               \
                            (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +      \
                             bxh[x2] * m_sigmaxyx[i2]);                        \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        for (k = 0; k < A / 2; ++k) {                                          \
          if (pml_x == 1 ||                                                    \
              (x > A / 2 - 2 + 1 + k && x < nx - 2 - A / 2 + 2 + 1 + k)) {     \
            int64_t i2 = i - (1 + k);                                          \
            int64_t x2 = x - (1 + k);                                          \
            DW_DTYPE buoyancyyhxh;                                             \
            if (pml_y == 2 && y == ny - 1) {                                   \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;            \
            } else {                                                           \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +                \
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /     \
                             4;                                                \
            }                                                                  \
            sigmaxy[i] +=                                                      \
                fd_coeffsx[k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +  \
                                 bxh[x2] * m_sigmaxyx[i2]);                    \
          }                                                                    \
          if (pml_x == 1 ||                                                    \
              (x > A / 2 - 2 - k && x < nx - 2 - A / 2 + 2 - k)) {             \
            int64_t i2 = i - (-k);                                             \
            int64_t x2 = x - (-k);                                             \
            DW_DTYPE buoyancyyhxh;                                             \
            if (pml_y == 2 && y == ny - 1) {                                   \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1]) / 2;            \
            } else {                                                           \
              buoyancyyhxh = (buoyancy[i2] + buoyancy[i2 + 1] +                \
                              buoyancy[i2 + nx] + buoyancy[i2 + nx + 1]) /     \
                             4;                                                \
            }                                                                  \
            sigmaxy[i] -=                                                      \
                fd_coeffsx[k] * (buoyancyyhxh * dt * (1 + bxh[x2]) * vy[i2] +  \
                                 bxh[x2] * m_sigmaxyx[i2]);                    \
          }                                                                    \
        }                                                                      \
                                                                               \
        /* dsigmaxydy */                                                       \
        for (k = 0; k < A; ++k) {                                              \
          for (j = 0; j < A / 2 - 1; ++j) {                                    \
            if (pml_y == 0 && y == 1 + j - j + k) {                            \
              int64_t i2 = i - (-j + k) * nx;                                  \
              int64_t y2 = y - (-j + k);                                       \
              sigmaxy[i] += fd_coeffs2y[1 + k] *                               \
                            (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +       \
                             by[y2] * m_sigmaxyy[i2]);                         \
            } else if (pml_y == 2 && y == ny - 1 - j + j - k - 1) {            \
              int64_t i2 = i - (j - k - 1) * nx;                               \
              int64_t y2 = y - (j - k - 1);                                    \
              sigmaxy[i] -= fd_coeffs2y[1 + k] *                               \
                            (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +       \
                             by[y2] * m_sigmaxyy[i2]);                         \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        for (k = 0; k < A / 2; ++k) {                                          \
          if (pml_y == 1 ||                                                    \
              (y > 1 + A / 2 - 2 + k && y < ny - 1 - A / 2 + 2 + k)) {         \
            int64_t i2 = i - (k)*nx;                                           \
            int64_t y2 = y - (k);                                              \
            sigmaxy[i] +=                                                      \
                fd_coeffsy[k] * (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +   \
                                 by[y2] * m_sigmaxyy[i2]);                     \
          }                                                                    \
          if (pml_y == 1 || (y > 1 + A / 2 - 2 - (k + 1) &&                    \
                             y < ny - 1 - A / 2 + 2 - (k + 1))) {              \
            int64_t i2 = i - (-(k + 1)) * nx;                                  \
            int64_t y2 = y - (-(k + 1));                                       \
            sigmaxy[i] -=                                                      \
                fd_coeffsy[k] * (buoyancy[i2] * dt * (1 + by[y2]) * vx[i2] +   \
                                 by[y2] * m_sigmaxyy[i2]);                     \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

static DW_DTYPE fd_coeffsy[2], fd_coeffsx[2], fd_coeffs1y[2][5],
    fd_coeffs1x[2][5], fd_coeffs2y[5], fd_coeffs2x[5], fd_coeffs3y[6],
    fd_coeffs3x[6];

static void add_pressure(DW_DTYPE *__restrict const sigmayy,
                         DW_DTYPE *__restrict const sigmaxx,
                         int64_t const *__restrict const sources_i,
                         DW_DTYPE const *__restrict const f,
                         int64_t const n_sources_per_shot) {
  int64_t source_idx;
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

static void add_to_grad_lamb(DW_DTYPE *__restrict const grad_lamb,
                             DW_DTYPE const *__restrict const sigmayy,
                             DW_DTYPE const *__restrict const sigmaxx,
                             DW_DTYPE const *__restrict const dvydy_store,
                             DW_DTYPE const *__restrict const dvxdx_store,
                             int64_t const step_ratio, int64_t const ny,
                             int64_t const nx) {
  int64_t y;
  for (y = 1; y < ny; ++y) {
    int64_t x;
    int64_t yi = y * nx;
    {
      int64_t i = yi;
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2) *
          (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_lamb[i] +=
          ((sigmayy[i] + sigmaxx[i]) * (dvydy_store[i] + dvxdx_store[i]) / 2 +
           (sigmayy[i - 1] + sigmaxx[i - 1]) *
               (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i = yi + nx - 1;
      grad_lamb[i] += ((sigmayy[i - 1] + sigmaxx[i - 1]) *
                       (dvydy_store[i - 1] + dvxdx_store[i - 1]) / 2) *
                      (DW_DTYPE)step_ratio;
    }
  }
}

static void add_to_grad_mu(DW_DTYPE *__restrict const grad_mu,
                           DW_DTYPE const *__restrict const sigmayy,
                           DW_DTYPE const *__restrict const sigmaxy,
                           DW_DTYPE const *__restrict const sigmaxx,
                           DW_DTYPE const *__restrict const dvydy_store,
                           DW_DTYPE const *__restrict const dvxdx_store,
                           DW_DTYPE const *__restrict const dvydxdvxdy_store,
                           int64_t const step_ratio, int64_t const ny,
                           int64_t const nx) {
  int64_t y;
  {
    int64_t x, yi;
    y = 1;
    yi = y * nx;
    {
      int64_t i = yi;
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i = yi + nx - 1;
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    (DW_DTYPE)step_ratio;
    }
  }
  for (y = 2; y < ny - 1; ++y) {
    int64_t x, yi = y * nx;
    {
      int64_t i = yi;
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i] * dvydxdvxdy_store[i] / 2 +
           sigmaxy[i - nx] * dvydxdvxdy_store[i - nx] / 2) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i = yi + nx - 1;
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    (DW_DTYPE)step_ratio;
    }
  }
  {
    int64_t x, yi;
    y = ny - 1;
    yi = y * nx;
    {
      int64_t i = yi;
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i])) *
          (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_mu[i] +=
          ((sigmayy[i] * dvydy_store[i] + sigmaxx[i] * dvxdx_store[i]) +
           (sigmayy[i - 1] * dvydy_store[i - 1] +
            sigmaxx[i - 1] * dvxdx_store[i - 1]) +
           sigmaxy[i - nx] * dvydxdvxdy_store[i - nx] / 2) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i = yi + nx - 1;
      grad_mu[i] += ((sigmayy[i - 1] * dvydy_store[i - 1] +
                      sigmaxx[i - 1] * dvxdx_store[i - 1])) *
                    (DW_DTYPE)step_ratio;
    }
  }
}

static void add_to_grad_buoyancy(DW_DTYPE *__restrict const grad_buoyancy,
                                 DW_DTYPE const *__restrict const vy,
                                 DW_DTYPE const *__restrict const vx,
                                 DW_DTYPE const *__restrict const dvydbuoyancy,
                                 DW_DTYPE const *__restrict const dvxdbuoyancy,
                                 int64_t const step_ratio, int64_t const ny,
                                 int64_t const nx) {
  int64_t y, x;
  {
    int64_t yi;
    y = 0;
    yi = y * nx;
    {
      int64_t i;
      x = 0;
      i = yi + x;
      grad_buoyancy[i] += (vy[i] * dvydbuoyancy[i] / 4) * (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i;
      x = nx - 1;
      i = yi + x;
      grad_buoyancy[i] +=
          (vy[i - 1] * dvydbuoyancy[i - 1] / 4) * (DW_DTYPE)step_ratio;
    }
  }
  for (y = 1; y < ny - 1; ++y) {
    int64_t yi = y * nx;
    {
      int64_t i;
      x = 0;
      i = yi + x;
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 4 + vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
           vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i;
      x = nx - 1;
      i = yi + x;
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 4 +
                           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          (DW_DTYPE)step_ratio;
    }
  }
  {
    int64_t yi;
    y = ny - 1;
    yi = y * nx;
    {
      int64_t i;
      x = 0;
      i = yi + x;
      grad_buoyancy[i] +=
          (vy[i] * dvydbuoyancy[i] / 2 + vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    }
    for (x = 1; x < nx - 1; ++x) {
      int64_t i = yi + x;
      grad_buoyancy[i] +=
          (vy[i - nx] * dvydbuoyancy[i - nx] / 4 +
           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
           vy[i] * dvydbuoyancy[i] / 2 + vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
           vx[i] * dvxdbuoyancy[i]) *
          (DW_DTYPE)step_ratio;
    }
    {
      int64_t i;
      x = nx - 1;
      i = yi + x;
      grad_buoyancy[i] += (vy[i - 1] * dvydbuoyancy[i - 1] / 2 +
                           vy[i - nx - 1] * dvydbuoyancy[i - nx - 1] / 4 +
                           vx[i] * dvxdbuoyancy[i]) *
                          (DW_DTYPE)step_ratio;
    }
  }
}

static void combine_grad_elastic(DW_DTYPE *__restrict const grad,
                                 DW_DTYPE const *__restrict const grad_thread,
                                 int64_t const n_threads, int64_t const ny,
                                 int64_t const nx) {
  int64_t y, x, threadidx;
  for (y = 0; y < ny; ++y) {
    for (x = 0; x < nx; ++x) {
      int64_t const i = y * nx + x;
      for (threadidx = 0; threadidx < n_threads; ++threadidx) {
        grad[i] += grad_thread[threadidx * ny * nx + i];
      }
    }
  }
}

#ifdef _WIN32
__declspec(noinline)
#else
__attribute__ ((noinline))
#endif
    static void forward_shot_v(DW_DTYPE const *__restrict const buoyancy,
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

#ifdef _WIN32
__declspec(noinline)
#else
__attribute__ ((noinline))
#endif
    static void forward_shot_sigma(
        DW_DTYPE const *__restrict const lamb,
        DW_DTYPE const *__restrict const mu,
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
        DW_DTYPE const *__restrict const bxh, DW_DTYPE const dt,
        int64_t const ny, int64_t const nx, bool const lamb_requires_grad,
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

static void backward_shot(
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
    int64_t const *__restrict const receivers_p_i, DW_DTYPE const dt,
    int64_t const ny, int64_t const nx, int64_t const n_sources_y_per_shot,
    int64_t const n_sources_x_per_shot, int64_t const n_receivers_y_per_shot,
    int64_t n_receivers_x_per_shot, int64_t const n_receivers_p_per_shot,
    int64_t const step_ratio, bool const lamb_requires_grad,
    bool const mu_requires_grad, bool const buoyancy_requires_grad,
    int64_t const spml_y0, int64_t const spml_y1, int64_t const spml_x0,
    int64_t const spml_x1, int64_t const vpml_y0, int64_t const vpml_y1,
    int64_t const vpml_x0, int64_t const vpml_x1) {
  int64_t y, x, y_begin_y, y_end_y, x_begin_y, x_end_y, y_begin_x, y_end_x,
      x_begin_x, x_end_x, y_begin_ii, y_end_ii, x_begin_ii, x_end_ii,
      y_begin_xy, y_end_xy, x_begin_xy, x_end_xy;
  /* For efficiency, the order of operations is different from what one
     might expect for the adjoint, from looking at the forward.

     Sometime before backward_v, we use sigma^{t+1} to add to the
     lamb and mu gradients.

     In backward_sigma we backpropagate sigma^{t+1} and m_v^{t+1/2}
     into v^{t+1/2} and backpropagate m_sigma^t and the newly
     calculated v^{t+1/2} into m_sigma^{t-1} (stored as m_sigma_n).

     Sometime after backward_sigma,
     we record the source gradients from v^{t+1/2}, and use
     v^{t+1/2} to add to the buoyancy gradient.

     In backward_v we first backpropagate sigma^{t+1} and m_v^{t+1/2}
     into m_v^{t-1/2}. We then backpropagate v^{t+1/2} and
     m_sigma^t into sigma^t.

     After this we add the adjoint velocity
     sources from time step t-1/2 to v^{t-1/2}.

     Adding the adjoint pressure sources from time step t to
     sigma^t completes its calculation.
  */
  if (lamb_requires_grad) {
    add_to_grad_lamb(grad_lamb, sigmayy, sigmaxx, dvydy_store, dvxdx_store,
                     step_ratio, ny, nx);
  }
  if (mu_requires_grad) {
    add_to_grad_mu(grad_mu, sigmayy, sigmaxy, sigmaxx, dvydy_store, dvxdx_store,
                   dvydxdvxdy_store, step_ratio, ny, nx);
  }

  BACKWARD_KERNEL_SIGMA(0, 0)
  BACKWARD_KERNEL_SIGMA(0, 1)
  BACKWARD_KERNEL_SIGMA(0, 2)
  BACKWARD_KERNEL_SIGMA(1, 0)
  BACKWARD_KERNEL_SIGMA(1, 1)
  BACKWARD_KERNEL_SIGMA(1, 2)
  BACKWARD_KERNEL_SIGMA(2, 0)
  BACKWARD_KERNEL_SIGMA(2, 1)
  BACKWARD_KERNEL_SIGMA(2, 2)
  if (n_sources_y_per_shot > 0) {
    record_from_wavefield(vy, sources_y_i, grad_f_y, n_sources_y_per_shot);
  }
  if (n_sources_x_per_shot > 0) {
    record_from_wavefield(vx, sources_x_i, grad_f_x, n_sources_x_per_shot);
  }
  if (buoyancy_requires_grad) {
    add_to_grad_buoyancy(grad_buoyancy, vy, vx, dvydbuoyancy, dvxdbuoyancy,
                         step_ratio, ny, nx);
  }
  BACKWARD_KERNEL_V(0, 0)
  BACKWARD_KERNEL_V(0, 1)
  BACKWARD_KERNEL_V(0, 2)
  BACKWARD_KERNEL_V(1, 0)
  BACKWARD_KERNEL_V(1, 1)
  BACKWARD_KERNEL_V(1, 2)
  BACKWARD_KERNEL_V(2, 0)
  BACKWARD_KERNEL_V(2, 1)
  BACKWARD_KERNEL_V(2, 2)
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

static void set_fd_coeffs_y(DW_DTYPE const dy) {
  if (DW_ACCURACY == 2) {
    fd_coeffsy[0] = (DW_DTYPE)(1.0 / 1.0) / dy;
    fd_coeffs1y[0][0] = (DW_DTYPE)(-8.0 / 3.0) / dy;
    fd_coeffs1y[0][1] = (DW_DTYPE)(3.0 / 1.0) / dy;
    fd_coeffs1y[0][2] = (DW_DTYPE)(-1.0 / 3.0) / dy;
  } else {
    fd_coeffsy[0] = (DW_DTYPE)(9.0 / 8.0) / dy;
    fd_coeffsy[1] = (DW_DTYPE)(-1.0 / 24.0) / dy;
    fd_coeffs1y[0][0] = (DW_DTYPE)(-352.0 / 105.0) / dy;
    fd_coeffs1y[0][1] = (DW_DTYPE)(35.0 / 8.0) / dy;
    fd_coeffs1y[0][2] = (DW_DTYPE)(-35.0 / 24.0) / dy;
    fd_coeffs1y[0][3] = (DW_DTYPE)(21.0 / 40.0) / dy;
    fd_coeffs1y[0][4] = (DW_DTYPE)(-5.0 / 56.0) / dy;
    fd_coeffs1y[1][0] = (DW_DTYPE)(16.0 / 105.0) / dy;
    fd_coeffs1y[1][1] = (DW_DTYPE)(-31.0 / 24.0) / dy;
    fd_coeffs1y[1][2] = (DW_DTYPE)(29.0 / 24.0) / dy;
    fd_coeffs1y[1][3] = (DW_DTYPE)(-3.0 / 40.0) / dy;
    fd_coeffs1y[1][4] = (DW_DTYPE)(1.0 / 168.0) / dy;
    fd_coeffs2y[0] = (DW_DTYPE)(-11.0 / 12.0) / dy;
    fd_coeffs2y[1] = (DW_DTYPE)(17.0 / 24.0) / dy;
    fd_coeffs2y[2] = (DW_DTYPE)(3.0 / 8.0) / dy;
    fd_coeffs2y[3] = (DW_DTYPE)(-5.0 / 24.0) / dy;
    fd_coeffs2y[4] = (DW_DTYPE)(1.0 / 24.0) / dy;
    fd_coeffs3y[0] = (DW_DTYPE)(-71.0 / 1689.0);
    fd_coeffs3y[1] = (DW_DTYPE)(-14587.0 / 13512.0) / dy;
    fd_coeffs3y[2] = (DW_DTYPE)(11243.0 / 10134.0) / dy;
    fd_coeffs3y[3] = (DW_DTYPE)(-43.0 / 2252.0) / dy;
    fd_coeffs3y[4] = (DW_DTYPE)(-47.0 / 3378.0) / dy;
    fd_coeffs3y[5] = (DW_DTYPE)(127.0 / 40536.0) / dy;
  }
}

static void set_fd_coeffs_x(DW_DTYPE const dx) {
  if (DW_ACCURACY == 2) {
    fd_coeffsx[0] = (DW_DTYPE)(1.0 / 1.0) / dx;
    fd_coeffs1x[0][0] = (DW_DTYPE)(-8.0 / 3.0) / dx;
    fd_coeffs1x[0][1] = (DW_DTYPE)(3.0 / 1.0) / dx;
    fd_coeffs1x[0][2] = (DW_DTYPE)(-1.0 / 3.0) / dx;
  } else {
    fd_coeffsx[0] = (DW_DTYPE)(9.0 / 8.0) / dx;
    fd_coeffsx[1] = (DW_DTYPE)(-1.0 / 24.0) / dx;
    fd_coeffs1x[0][0] = (DW_DTYPE)(-352.0 / 105.0) / dx;
    fd_coeffs1x[0][1] = (DW_DTYPE)(35.0 / 8.0) / dx;
    fd_coeffs1x[0][2] = (DW_DTYPE)(-35.0 / 24.0) / dx;
    fd_coeffs1x[0][3] = (DW_DTYPE)(21.0 / 40.0) / dx;
    fd_coeffs1x[0][4] = (DW_DTYPE)(-5.0 / 56.0) / dx;
    fd_coeffs1x[1][0] = (DW_DTYPE)(16.0 / 105.0) / dx;
    fd_coeffs1x[1][1] = (DW_DTYPE)(-31.0 / 24.0) / dx;
    fd_coeffs1x[1][2] = (DW_DTYPE)(29.0 / 24.0) / dx;
    fd_coeffs1x[1][3] = (DW_DTYPE)(-3.0 / 40.0) / dx;
    fd_coeffs1x[1][4] = (DW_DTYPE)(1.0 / 168.0) / dx;
    fd_coeffs2x[0] = (DW_DTYPE)(-11.0 / 12.0) / dx;
    fd_coeffs2x[1] = (DW_DTYPE)(17.0 / 24.0) / dx;
    fd_coeffs2x[2] = (DW_DTYPE)(3.0 / 8.0) / dx;
    fd_coeffs2x[3] = (DW_DTYPE)(-5.0 / 24.0) / dx;
    fd_coeffs2x[4] = (DW_DTYPE)(1.0 / 24.0) / dx;
    fd_coeffs3x[0] = (DW_DTYPE)(-71.0 / 1689.0);
    fd_coeffs3x[1] = (DW_DTYPE)(-14587.0 / 13512.0) / dx;
    fd_coeffs3x[2] = (DW_DTYPE)(11243.0 / 10134.0) / dx;
    fd_coeffs3x[3] = (DW_DTYPE)(-43.0 / 2252.0) / dx;
    fd_coeffs3x[4] = (DW_DTYPE)(-47.0 / 3378.0) / dx;
    fd_coeffs3x[5] = (DW_DTYPE)(127.0 / 40536.0) / dx;
  }
}

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
        int64_t const *__restrict const receivers_p_i, DW_DTYPE const dy,
        DW_DTYPE const dx, DW_DTYPE const dt, int64_t const nt,
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
  set_fd_coeffs_y(dy);
  set_fd_coeffs_x(dx);
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
    DW_DTYPE const *__restrict const buoyancy_shot =
        buoyancy_batched ? buoyancy + si : buoyancy;
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
      forward_shot_v(
          buoyancy_shot, vy + si, vx + si, sigmayy + si, sigmaxy + si,
          sigmaxx + si, m_sigmayyy + si, m_sigmaxyy + si, m_sigmaxyx + si,
          m_sigmaxxx + si, dvydbuoyancy + store_i, dvxdbuoyancy + store_i, ay,
          ayh, ax, axh, by, byh, bx, bxh, dt, ny, nx,
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
          lamb_shot, mu_shot, vy + si, vx + si, sigmayy + si, sigmaxy + si,
          sigmaxx + si, m_vyy + si, m_vyx + si, m_vxy + si, m_vxx + si,
          dvydy_store + store_i, dvxdx_store + store_i,
          dvydxdvxdy_store + store_i, ay, ayh, ax, axh, by, byh, bx, bxh, dt,
          ny, nx, lamb_requires_grad && ((t % step_ratio) == 0),
          mu_requires_grad && ((t % step_ratio) == 0), pml_y0,
          pml_y1, pml_x0, pml_x1);
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
        DW_DTYPE const *__restrict const buoyancy,
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
        DW_DTYPE *__restrict const grad_buoyancy,
        DW_DTYPE *__restrict const grad_buoyancy_thread,
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
        DW_DTYPE const dx, DW_DTYPE const dt, int64_t const nt,
        int64_t const n_shots, int64_t const ny, int64_t const nx,
        int64_t const n_sources_y_per_shot, int64_t const n_sources_x_per_shot,
        int64_t const n_receivers_y_per_shot, int64_t n_receivers_x_per_shot,
        int64_t const n_receivers_p_per_shot, int64_t const step_ratio,
        bool const lamb_requires_grad, bool const mu_requires_grad,
        bool const buoyancy_requires_grad, bool const lamb_batched,
        bool const mu_batched, bool const buoyancy_batched, int64_t start_t,
        int64_t const spml_y0, int64_t const spml_y1, int64_t const spml_x0,
        int64_t const spml_x1, int64_t const vpml_y0, int64_t const vpml_y1,
        int64_t const vpml_x0, int64_t const vpml_x1, int64_t const n_threads) {
  int64_t shot;
  set_fd_coeffs_y(dy);
  set_fd_coeffs_x(dx);
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
    DW_DTYPE const *__restrict const buoyancy_shot =
        buoyancy_batched ? buoyancy + si : buoyancy;
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
            lamb_shot, mu_shot, buoyancy_shot,
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
            grad_buoyancy_thread + grad_buoyancy_i, ay, ayh, ax, axh, by, byh,
            bx, bxh, sources_y_i + siy, sources_x_i + six, receivers_y_i + riy,
            receivers_x_i + rix, receivers_p_i + rip, dt, ny, nx,
            n_sources_y_per_shot, n_sources_x_per_shot, n_receivers_y_per_shot,
            n_receivers_x_per_shot, n_receivers_p_per_shot, step_ratio,
            lamb_requires_grad && ((t % step_ratio) == 0),
            mu_requires_grad && ((t % step_ratio) == 0),
            buoyancy_requires_grad && ((t % step_ratio) == 0),
            spml_y0, spml_y1, spml_x0, spml_x1, vpml_y0, vpml_y1, vpml_x0,
            vpml_x1);
      } else {
        backward_shot(
            lamb_shot, mu_shot, buoyancy_shot,
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
            grad_buoyancy_thread + grad_buoyancy_i, ay, ayh, ax, axh, by, byh,
            bx, bxh, sources_y_i + siy, sources_x_i + six, receivers_y_i + riy,
            receivers_x_i + rix, receivers_p_i + rip, dt, ny, nx,
            n_sources_y_per_shot, n_sources_x_per_shot, n_receivers_y_per_shot,
            n_receivers_x_per_shot, n_receivers_p_per_shot, step_ratio,
            lamb_requires_grad && ((t % step_ratio) == 0),
            mu_requires_grad && ((t % step_ratio) == 0),
            buoyancy_requires_grad && ((t % step_ratio) == 0),
            spml_y0, spml_y1, spml_x0, spml_x1, vpml_y0, vpml_y1, vpml_x0,
            vpml_x1);
      }
    }
  }
#ifdef _OPENMP
  if (lamb_requires_grad && !lamb_batched && n_threads > 1) {
    combine_grad_elastic(grad_lamb, grad_lamb_thread, n_threads, ny, nx);
  }
  if (mu_requires_grad && !mu_batched && n_threads > 1) {
    combine_grad_elastic(grad_mu, grad_mu_thread, n_threads, ny, nx);
  }
  if (buoyancy_requires_grad && !buoyancy_batched && n_threads > 1) {
    combine_grad_elastic(grad_buoyancy, grad_buoyancy_thread, n_threads, ny,
                         nx);
  }
#endif /* _OPENMP */
}

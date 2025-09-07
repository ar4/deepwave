/*
 * Scalar Born wave equation propagator
 */

/*
 * This file contains the C implementation of the scalar Born wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * For a description of the method, see the C implementation of the scalar
 * propagator in scalar.c. This file implements the same functionality,
 * but for the scalar Born wave equation. This involves propagating two
 * wavefields simultaneously: the background wavefield and the scattered
 * wavefield. The scattered wavefield has a source term that is
 * proportional to the background wavefield multiplied by the scattering
 * potential.
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>

#include "common.h"
#include "common_cpu.h"

#define CAT_I(name, accuracy, dtype, device) \
  scalar_born_iso_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

// Access the background wavefield at offset (dy, dx) from i
#define WFC(dy, dx) wfc[i + dy * nx + dx]
// Access the scattered wavefield at offset (dy, dx) from i
#define WFCSC(dy, dx) wfcsc[i + dy * nx + dx]
// PML profile ay times auxiliary field psiy (background)
#define AY_PSIY(dy, dx) ay[y + dy] * psiy[i + dy * nx + dx]
// PML profile ax times auxiliary field psix (background)
#define AX_PSIX(dy, dx) ax[x + dx] * psix[i + dy * nx + dx]
// PML profile ay times auxiliary field psiysc (scattered)
#define AY_PSIYSC(dy, dx) ay[y + dy] * psiysc[i + dy * nx + dx]
// PML profile ax times auxiliary field psixsc (scattered)
#define AX_PSIXSC(dy, dx) ax[x + dx] * psixsc[i + dy * nx + dx]
// Access velocity at offset (dy, dx) from i
#define V(dy, dx) v[i + dy * nx + dx]
// v * dt^2 at offset
#define VDT2(dy, dx) V(dy, dx) * dt2
// v^2 * dt^2 at offset
#define V2DT2(dy, dx) V(dy, dx) * V(dy, dx) * dt2
// Scattering potential at offset
#define SCATTER(dy, dx) scatter[i + dy * nx + dx]
// Second derivative term in the backward background wavefield update
#define V2DT2_WFC(dy, dx)        \
  (V2DT2(dy, dx) * WFC(dy, dx) + \
   2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx))
// Second derivative term in the backward scattered wavefield update
#define V2DT2_WFCSC(dy, dx) V2DT2(dy, dx) * WFCSC(dy, dx)
/* Update term for the y-derivative of the wavefield in the PML region, used
 * in the backward pass. This corresponds to the first derivative term in the
 * update equation.
 */
#define UT_TERMY1(dy, dx)                                                      \
  (dbydy[y + dy] * ((1 + by[y + dy]) *                                         \
                        (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                         2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
                    by[y + dy] * zetay[i + dy * nx]) +                         \
   by[y + dy] * psiy[i + dy * nx])
/* Update term for the x-derivative of the wavefield in the PML region, used
 * in the backward pass. This corresponds to the first derivative term in the
 * update equation.
 */
#define UT_TERMX1(dy, dx)                                                      \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) *                                         \
                        (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                         2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
                    bx[x + dx] * zetax[i + dx]) +                              \
   bx[x + dx] * psix[i + dx])
/* Update term for the y-derivative of the wavefield in the PML region, used
 * in the backward pass. This corresponds to the second derivative term in the
 * update equation.
 */
#define UT_TERMY2(dy, dx)                                                     \
  ((1 + by[y + dy]) *                                                         \
   ((1 + by[y + dy]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                        2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
    by[y + dy] * zetay[i + dy * nx]))
/* Update term for the x-derivative of the wavefield in the PML region, used
 * in the backward pass. This corresponds to the second derivative term in the
 * update equation.
 */
#define UT_TERMX2(dy, dx)                                                     \
  ((1 + bx[x + dx]) *                                                         \
   ((1 + bx[x + dx]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                        2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
    bx[x + dx] * zetax[i + dx]))
#define PSIY_TERM(dy, dx)                                                    \
  ((1 + by[y + dy]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                       2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
   by[y + dy] * zetay[i + dy * nx])
#define PSIX_TERM(dy, dx)                                                    \
  ((1 + bx[x + dx]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                       2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
   bx[x + dx] * zetax[i + dx])
#define UTSC_TERMY1(dy, dx)                                             \
  ((dbydy[y + dy] * ((1 + by[y + dy]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                     by[y + dy] * zetaysc[i + dy * nx]) +               \
    by[y + dy] * psiysc[i + dy * nx]))
#define UTSC_TERMX1(dy, dx)                                             \
  ((dbxdx[x + dx] * ((1 + bx[x + dx]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                     bx[x + dx] * zetaxsc[i + dx]) +                    \
    bx[x + dx] * psixsc[i + dx]))
#define UTSC_TERMY2(dy, dx)                                               \
  ((1 + by[y + dy]) * ((1 + by[y + dy]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                       by[y + dy] * zetaysc[i + dy * nx]))
#define UTSC_TERMX2(dy, dx)                                               \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                       bx[x + dx] * zetaxsc[i + dx]))
#define PSIYSC_TERM(dy, dx)                           \
  ((1 + by[y + dy]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
   by[y + dy] * zetaysc[i + dy * nx])
#define PSIXSC_TERM(dy, dx)                           \
  ((1 + bx[x + dx]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
   bx[x + dx] * zetaxsc[i + dx])

#define SET_RANGE(pml_y, pml_x) \
  {                             \
    if (pml_y == 0) {           \
      y_begin = FD_PAD;         \
      y_end = pml_y0;           \
    } else if (pml_y == 1) {    \
      y_begin = pml_y0;         \
      y_end = pml_y1;           \
    } else {                    \
      y_begin = pml_y1;         \
      y_end = ny - FD_PAD;      \
    }                           \
    if (pml_x == 0) {           \
      x_begin = FD_PAD;         \
      x_end = pml_x0;           \
    } else if (pml_x == 1) {    \
      x_begin = pml_x0;         \
      x_end = pml_x1;           \
    } else {                    \
      x_begin = pml_x1;         \
      x_end = nx - FD_PAD;      \
    }                           \
  }

#define FORWARD_KERNEL(pml_y, pml_x, v_requires_grad, scatter_requires_grad) \
  {                                                                          \
    SET_RANGE(pml_y, pml_x)                                                  \
    _Pragma("omp simd collapse(2)") for (y = y_begin; y < y_end; ++y) {      \
      for (x = x_begin; x < x_end; ++x) {                                    \
        int64_t const i = y * nx + x;                                        \
        if (pml_y == 1) {                                                    \
          w_sum = DIFFY2(WFC);                                               \
          wsc_sum = DIFFY2(WFCSC);                                           \
        } else {                                                             \
          DW_DTYPE dwfcdy = DIFFY1(WFC);                                     \
          DW_DTYPE tmpy = ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy +   \
                           DIFFY1(AY_PSIY));                                 \
          DW_DTYPE dwfcscdy = DIFFY1(WFCSC);                                 \
          DW_DTYPE tmpysc = ((1 + by[y]) * DIFFY2(WFCSC) +                   \
                             dbydy[y] * dwfcscdy + DIFFY1(AY_PSIYSC));       \
          w_sum = (1 + by[y]) * tmpy + ay[y] * zetay[i];                     \
          wsc_sum = (1 + by[y]) * tmpysc + ay[y] * zetaysc[i];               \
          psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];                       \
          zetay[i] = by[y] * tmpy + ay[y] * zetay[i];                        \
          psiynsc[i] = by[y] * dwfcscdy + ay[y] * psiysc[i];                 \
          zetaysc[i] = by[y] * tmpysc + ay[y] * zetaysc[i];                  \
        }                                                                    \
        if (pml_x == 1) {                                                    \
          w_sum += DIFFX2(WFC);                                              \
          wsc_sum += DIFFX2(WFCSC);                                          \
        } else {                                                             \
          DW_DTYPE dwfcdx = DIFFX1(WFC);                                     \
          DW_DTYPE tmpx = ((1 + bx[x]) * DIFFX2(WFC) + dbxdx[x] * dwfcdx +   \
                           DIFFX1(AX_PSIX));                                 \
          DW_DTYPE dwfcscdx = DIFFX1(WFCSC);                                 \
          DW_DTYPE tmpxsc = ((1 + bx[x]) * DIFFX2(WFCSC) +                   \
                             dbxdx[x] * dwfcscdx + DIFFX1(AX_PSIXSC));       \
          w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];                    \
          wsc_sum += (1 + bx[x]) * tmpxsc + ax[x] * zetaxsc[i];              \
          psixn[i] = bx[x] * dwfcdx + ax[x] * psix[i];                       \
          zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];                        \
          psixnsc[i] = bx[x] * dwfcscdx + ax[x] * psixsc[i];                 \
          zetaxsc[i] = bx[x] * tmpxsc + ax[x] * zetaxsc[i];                  \
        }                                                                    \
        wfp[i] = v[i] * v[i] * dt2 * w_sum + 2 * wfc[i] - wfp[i];            \
        wfpsc[i] = v[i] * v[i] * dt2 * wsc_sum + 2 * wfcsc[i] - wfpsc[i] +   \
                   2 * v[i] * scatter[i] * dt2 * w_sum;                      \
        if (v_requires_grad || scatter_requires_grad) {                      \
          w_store[i] = w_sum;                                                \
        }                                                                    \
        if (v_requires_grad) {                                               \
          wsc_store[i] = wsc_sum;                                            \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  }

#define BACKWARD_KERNEL(pml_y, pml_x, v_requires_grad, scatter_requires_grad)  \
  {                                                                            \
    SET_RANGE(pml_y, pml_x)                                                    \
    _Pragma("omp simd collapse(2)") for (y = y_begin; y < y_end; ++y) {        \
      for (x = x_begin; x < x_end; ++x) {                                      \
        int64_t const i = y * nx + x;                                          \
        wfp[i] = (pml_y == 1 ? DIFFY2(V2DT2_WFC)                               \
                             : -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2)) +       \
                 (pml_x == 1 ? DIFFX2(V2DT2_WFC)                               \
                             : -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2)) +       \
                 2 * wfc[i] - wfp[i];                                          \
        wfpsc[i] = (pml_y == 1 ? DIFFY2(V2DT2_WFCSC)                           \
                               : -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2)) + \
                   (pml_x == 1 ? DIFFX2(V2DT2_WFCSC)                           \
                               : -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2)) + \
                   2 * wfcsc[i] - wfpsc[i];                                    \
        if (pml_y == 0 || pml_y == 2) {                                        \
          psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];       \
          zetaynsc[i] = ay[y] * V2DT2(0, 0) * wfcsc[i] + ay[y] * zetaysc[i];   \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];       \
          zetaxnsc[i] = ax[x] * V2DT2(0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];   \
        }                                                                      \
        if (pml_y == 0 || pml_y == 2) {                                        \
          psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];             \
          zetayn[i] = ay[y] * V2DT2(0, 0) * wfc[i] +                           \
                      ay[y] * 2 * VDT2(0, 0) * SCATTER(0, 0) * wfcsc[i] +      \
                      ay[y] * zetay[i];                                        \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];             \
          zetaxn[i] = ax[x] * V2DT2(0, 0) * wfc[i] +                           \
                      ax[x] * 2 * VDT2(0, 0) * SCATTER(0, 0) * wfcsc[i] +      \
                      ax[x] * zetax[i];                                        \
        }                                                                      \
        if (v_requires_grad) {                                                 \
          grad_v[i] +=                                                         \
              wfc[i] * 2 * v[i] * dt2 * w_store[i] * (DW_DTYPE)step_ratio +    \
              wfcsc[i] *                                                       \
                  (2 * dt2 * scatter[i] * w_store[i] +                         \
                   2 * v[i] * dt2 * wsc_store[i]) *                            \
                  (DW_DTYPE)step_ratio;                                        \
        }                                                                      \
        if (scatter_requires_grad) {                                           \
          grad_scatter[i] +=                                                   \
              wfcsc[i] * 2 * v[i] * dt2 * w_store[i] * (DW_DTYPE)step_ratio;   \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#define BACKWARD_KERNEL_SC(pml_y, pml_x, scatter_requires_grad)                \
  {                                                                            \
    SET_RANGE(pml_y, pml_x)                                                    \
    _Pragma("omp simd collapse(2)") for (y = y_begin; y < y_end; ++y) {        \
      for (x = x_begin; x < x_end; ++x) {                                      \
        int64_t const i = y * nx + x;                                          \
        wfpsc[i] = (pml_y == 1 ? DIFFY2(V2DT2_WFCSC)                           \
                               : -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2)) + \
                   (pml_x == 1 ? DIFFX2(V2DT2_WFCSC)                           \
                               : -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2)) + \
                   2 * wfcsc[i] - wfpsc[i];                                    \
        if (pml_y == 0 || pml_y == 2) {                                        \
          psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];       \
          zetaynsc[i] = ay[y] * V2DT2(0, 0) * wfcsc[i] + ay[y] * zetaysc[i];   \
        }                                                                      \
        if (pml_x == 0 || pml_x == 2) {                                        \
          psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];       \
          zetaxnsc[i] = ax[x] * V2DT2(0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];   \
        }                                                                      \
        if (scatter_requires_grad) {                                           \
          grad_scatter[i] +=                                                   \
              wfcsc[i] * 2 * v[i] * dt2 * w_store[i] * (DW_DTYPE)step_ratio;   \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

static void forward_kernel(
    DW_DTYPE const *__restrict const v,
    DW_DTYPE const *__restrict const scatter,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const psiy,
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE *__restrict const zetax, DW_DTYPE const *__restrict const wfcsc,
    DW_DTYPE *__restrict const wfpsc, DW_DTYPE const *__restrict const psiysc,
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE *__restrict const psixnsc, DW_DTYPE *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaxsc, DW_DTYPE *__restrict const w_store,
    DW_DTYPE *__restrict const wsc_store, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, DW_DTYPE const rdy,
    DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
    DW_DTYPE const dt2, int64_t const ny, int64_t const nx,
    bool const v_requires_grad, bool const scatter_requires_grad,
    int64_t const pml_y0, int64_t const pml_y1, int64_t const pml_x0,
    int64_t const pml_x1) {
  int64_t y, x, y_begin, y_end, x_begin, x_end;
  DW_DTYPE w_sum, wsc_sum;
  // Dispatch to the correct kernel macro for all PML regions and grad configs
  if (v_requires_grad) {
    if (scatter_requires_grad) {
      FORWARD_KERNEL(0, 0, 1, 1)
      FORWARD_KERNEL(0, 1, 1, 1)
      FORWARD_KERNEL(0, 2, 1, 1)
      FORWARD_KERNEL(1, 0, 1, 1)
      FORWARD_KERNEL(1, 1, 1, 1)
      FORWARD_KERNEL(1, 2, 1, 1)
      FORWARD_KERNEL(2, 0, 1, 1)
      FORWARD_KERNEL(2, 1, 1, 1)
      FORWARD_KERNEL(2, 2, 1, 1)
    } else {
      FORWARD_KERNEL(0, 0, 1, 0)
      FORWARD_KERNEL(0, 1, 1, 0)
      FORWARD_KERNEL(0, 2, 1, 0)
      FORWARD_KERNEL(1, 0, 1, 0)
      FORWARD_KERNEL(1, 1, 1, 0)
      FORWARD_KERNEL(1, 2, 1, 0)
      FORWARD_KERNEL(2, 0, 1, 0)
      FORWARD_KERNEL(2, 1, 1, 0)
      FORWARD_KERNEL(2, 2, 1, 0)
    }
  } else {
    if (scatter_requires_grad) {
      FORWARD_KERNEL(0, 0, 0, 1)
      FORWARD_KERNEL(0, 1, 0, 1)
      FORWARD_KERNEL(0, 2, 0, 1)
      FORWARD_KERNEL(1, 0, 0, 1)
      FORWARD_KERNEL(1, 1, 0, 1)
      FORWARD_KERNEL(1, 2, 0, 1)
      FORWARD_KERNEL(2, 0, 0, 1)
      FORWARD_KERNEL(2, 1, 0, 1)
      FORWARD_KERNEL(2, 2, 0, 1)
    } else {
      FORWARD_KERNEL(0, 0, 0, 0)
      FORWARD_KERNEL(0, 1, 0, 0)
      FORWARD_KERNEL(0, 2, 0, 0)
      FORWARD_KERNEL(1, 0, 0, 0)
      FORWARD_KERNEL(1, 1, 0, 0)
      FORWARD_KERNEL(1, 2, 0, 0)
      FORWARD_KERNEL(2, 0, 0, 0)
      FORWARD_KERNEL(2, 1, 0, 0)
      FORWARD_KERNEL(2, 2, 0, 0)
    }
  }
}

static void backward_kernel(
    DW_DTYPE const *__restrict const v,
    DW_DTYPE const *__restrict const scatter,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const psiy,
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const zetayn,
    DW_DTYPE *__restrict const zetaxn, DW_DTYPE const *__restrict const wfcsc,
    DW_DTYPE *__restrict const wfpsc, DW_DTYPE const *__restrict const psiysc,
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE *__restrict const psixnsc, DW_DTYPE *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaxsc, DW_DTYPE *__restrict const zetaynsc,
    DW_DTYPE *__restrict const zetaxnsc,
    DW_DTYPE const *__restrict const w_store,
    DW_DTYPE const *__restrict const wsc_store,
    DW_DTYPE *__restrict const grad_v, DW_DTYPE *__restrict const grad_scatter,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, DW_DTYPE const rdy,
    DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
    DW_DTYPE const dt2, int64_t const ny, int64_t const nx,
    int64_t const step_ratio, bool const v_requires_grad,
    bool const scatter_requires_grad, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1) {
  int64_t y, x, y_begin, y_end, x_begin, x_end;
  // Dispatch to the correct kernel macro for all PML regions and grad configs
  if (v_requires_grad) {
    if (scatter_requires_grad) {
      BACKWARD_KERNEL(0, 0, 1, 1)
      BACKWARD_KERNEL(0, 1, 1, 1)
      BACKWARD_KERNEL(0, 2, 1, 1)
      BACKWARD_KERNEL(1, 0, 1, 1)
      BACKWARD_KERNEL(1, 1, 1, 1)
      BACKWARD_KERNEL(1, 2, 1, 1)
      BACKWARD_KERNEL(2, 0, 1, 1)
      BACKWARD_KERNEL(2, 1, 1, 1)
      BACKWARD_KERNEL(2, 2, 1, 1)
    } else {
      BACKWARD_KERNEL(0, 0, 1, 0)
      BACKWARD_KERNEL(0, 1, 1, 0)
      BACKWARD_KERNEL(0, 2, 1, 0)
      BACKWARD_KERNEL(1, 0, 1, 0)
      BACKWARD_KERNEL(1, 1, 1, 0)
      BACKWARD_KERNEL(1, 2, 1, 0)
      BACKWARD_KERNEL(2, 0, 1, 0)
      BACKWARD_KERNEL(2, 1, 1, 0)
      BACKWARD_KERNEL(2, 2, 1, 0)
    }
  } else {
    if (scatter_requires_grad) {
      BACKWARD_KERNEL(0, 0, 0, 1)
      BACKWARD_KERNEL(0, 1, 0, 1)
      BACKWARD_KERNEL(0, 2, 0, 1)
      BACKWARD_KERNEL(1, 0, 0, 1)
      BACKWARD_KERNEL(1, 1, 0, 1)
      BACKWARD_KERNEL(1, 2, 0, 1)
      BACKWARD_KERNEL(2, 0, 0, 1)
      BACKWARD_KERNEL(2, 1, 0, 1)
      BACKWARD_KERNEL(2, 2, 0, 1)
    } else {
      BACKWARD_KERNEL(0, 0, 0, 0)
      BACKWARD_KERNEL(0, 1, 0, 0)
      BACKWARD_KERNEL(0, 2, 0, 0)
      BACKWARD_KERNEL(1, 0, 0, 0)
      BACKWARD_KERNEL(1, 1, 0, 0)
      BACKWARD_KERNEL(1, 2, 0, 0)
      BACKWARD_KERNEL(2, 0, 0, 0)
      BACKWARD_KERNEL(2, 1, 0, 0)
      BACKWARD_KERNEL(2, 2, 0, 0)
    }
  }
}

static void backward_kernel_sc(
    DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const wfcsc,
    DW_DTYPE *__restrict const wfpsc, DW_DTYPE const *__restrict const psiysc,
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE *__restrict const psixnsc, DW_DTYPE *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaxsc, DW_DTYPE *__restrict const zetaynsc,
    DW_DTYPE *__restrict const zetaxnsc,
    DW_DTYPE const *__restrict const w_store,
    DW_DTYPE *__restrict const grad_scatter,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, DW_DTYPE const rdy,
    DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
    DW_DTYPE const dt2, int64_t const ny, int64_t const nx,
    int64_t const step_ratio, bool const scatter_requires_grad,
    int64_t const pml_y0, int64_t const pml_y1, int64_t const pml_x0,
    int64_t const pml_x1) {
  int64_t y, x, y_begin, y_end, x_begin, x_end;
  if (scatter_requires_grad) {
    BACKWARD_KERNEL_SC(0, 0, 1)
    BACKWARD_KERNEL_SC(0, 1, 1)
    BACKWARD_KERNEL_SC(0, 2, 1)
    BACKWARD_KERNEL_SC(1, 0, 1)
    BACKWARD_KERNEL_SC(1, 1, 1)
    BACKWARD_KERNEL_SC(1, 2, 1)
    BACKWARD_KERNEL_SC(2, 0, 1)
    BACKWARD_KERNEL_SC(2, 1, 1)
    BACKWARD_KERNEL_SC(2, 2, 1)
  } else {
    BACKWARD_KERNEL_SC(0, 0, 0)
    BACKWARD_KERNEL_SC(0, 1, 0)
    BACKWARD_KERNEL_SC(0, 2, 0)
    BACKWARD_KERNEL_SC(1, 0, 0)
    BACKWARD_KERNEL_SC(1, 1, 0)
    BACKWARD_KERNEL_SC(1, 2, 0)
    BACKWARD_KERNEL_SC(2, 0, 0)
    BACKWARD_KERNEL_SC(2, 1, 0)
    BACKWARD_KERNEL_SC(2, 2, 0)
  }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FUNC(forward)(
        DW_DTYPE const *__restrict const v,
        DW_DTYPE const *__restrict const scatter,
        DW_DTYPE const *__restrict const f,
        DW_DTYPE const *__restrict const fsc, DW_DTYPE *__restrict const wfc,
        DW_DTYPE *__restrict const wfp, DW_DTYPE *__restrict const psiy,
        DW_DTYPE *__restrict const psix, DW_DTYPE *__restrict const psiyn,
        DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
        DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const wfcsc,
        DW_DTYPE *__restrict const wfpsc, DW_DTYPE *__restrict const psiysc,
        DW_DTYPE *__restrict const psixsc, DW_DTYPE *__restrict const psiynsc,
        DW_DTYPE *__restrict const psixnsc, DW_DTYPE *__restrict const zetaysc,
        DW_DTYPE *__restrict const zetaxsc, DW_DTYPE *__restrict const w_store,
        DW_DTYPE *__restrict const wsc_store, DW_DTYPE *__restrict const r,
        DW_DTYPE *__restrict const rsc, DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbydy,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receivers_i,
        int64_t const *__restrict const receiverssc_i, DW_DTYPE const rdy,
        DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
        DW_DTYPE const dt2, int64_t const nt, int64_t const n_shots,
        int64_t const ny, int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot,
        int64_t const n_receiverssc_per_shot, int64_t const step_ratio,
        bool const v_requires_grad, bool const scatter_requires_grad,
        bool const v_batched, bool const scatter_batched, int64_t start_t,
        int64_t const pml_y0, int64_t const pml_y1, int64_t const pml_x0,
        int64_t const pml_x1, int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const i = shot * ny * nx;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + i : v;
    DW_DTYPE const *__restrict const scatter_shot =
        scatter_batched ? scatter + i : scatter;
    int64_t t;
    for (t = start_t; t < start_t + nt; ++t) {
      if ((t - start_t) & 1) {
        forward_kernel(
            v_shot, scatter_shot, wfp + i, wfc + i, psiyn + i, psixn + i,
            psiy + i, psix + i, zetay + i, zetax + i, wfpsc + i, wfcsc + i,
            psiynsc + i, psixnsc + i, psiysc + i, psixsc + i, zetaysc + i,
            zetaxsc + i,
            w_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            wsc_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            ay, ax, by, bx, dbydy, dbxdx, rdy, rdx, rdy2, rdx2, dt2, ny, nx,
            v_requires_grad && ((t % step_ratio) == 0),
            scatter_requires_grad && ((t % step_ratio) == 0),
            pml_y0, pml_y1, pml_x0, pml_x1);
        add_to_wavefield(wfc + i, sources_i + si,
                         f + t * n_shots * n_sources_per_shot + si,
                         n_sources_per_shot);
        add_to_wavefield(wfcsc + i, sources_i + si,
                         fsc + t * n_shots * n_sources_per_shot + si,
                         n_sources_per_shot);
        record_from_wavefield(wfp + i, receivers_i + ri,
                              r + t * n_shots * n_receivers_per_shot + ri,
                              n_receivers_per_shot);
        record_from_wavefield(wfpsc + i, receiverssc_i + risc,
                              rsc + t * n_shots * n_receiverssc_per_shot + risc,
                              n_receiverssc_per_shot);
      } else {
        forward_kernel(
            v_shot, scatter_shot, wfc + i, wfp + i, psiy + i, psix + i,
            psiyn + i, psixn + i, zetay + i, zetax + i, wfcsc + i, wfpsc + i,
            psiysc + i, psixsc + i, psiynsc + i, psixnsc + i, zetaysc + i,
            zetaxsc + i,
            w_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            wsc_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            ay, ax, by, bx, dbydy, dbxdx, rdy, rdx, rdy2, rdx2, dt2, ny, nx,
            v_requires_grad && ((t % step_ratio) == 0),
            scatter_requires_grad && ((t % step_ratio) == 0),
            pml_y0, pml_y1, pml_x0, pml_x1);
        add_to_wavefield(wfp + i, sources_i + si,
                         f + t * n_shots * n_sources_per_shot + si,
                         n_sources_per_shot);
        add_to_wavefield(wfpsc + i, sources_i + si,
                         fsc + t * n_shots * n_sources_per_shot + si,
                         n_sources_per_shot);
        record_from_wavefield(wfc + i, receivers_i + ri,
                              r + t * n_shots * n_receivers_per_shot + ri,
                              n_receivers_per_shot);
        record_from_wavefield(wfcsc + i, receiverssc_i + risc,
                              rsc + t * n_shots * n_receiverssc_per_shot + risc,
                              n_receiverssc_per_shot);
      }
    }
  }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FUNC(backward)(
        DW_DTYPE const *__restrict const v,
        DW_DTYPE const *__restrict const scatter,
        DW_DTYPE const *__restrict const grad_r,
        DW_DTYPE const *__restrict const grad_rsc,
        DW_DTYPE *__restrict const wfc, DW_DTYPE *__restrict const wfp,
        DW_DTYPE *__restrict const psiy, DW_DTYPE *__restrict const psix,
        DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn,
        DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax,
        DW_DTYPE *__restrict const zetayn, DW_DTYPE *__restrict const zetaxn,
        DW_DTYPE *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
        DW_DTYPE *__restrict const psiysc, DW_DTYPE *__restrict const psixsc,
        DW_DTYPE *__restrict const psiynsc, DW_DTYPE *__restrict const psixnsc,
        DW_DTYPE *__restrict const zetaysc, DW_DTYPE *__restrict const zetaxsc,
        DW_DTYPE *__restrict const zetaynsc,
        DW_DTYPE *__restrict const zetaxnsc,
        DW_DTYPE const *__restrict const w_store,
        DW_DTYPE const *__restrict const wsc_store,
        DW_DTYPE *__restrict const grad_f, DW_DTYPE *__restrict const grad_fsc,
        DW_DTYPE *__restrict const grad_v,
        DW_DTYPE *__restrict const grad_scatter,
        DW_DTYPE *__restrict const grad_v_thread,
        DW_DTYPE *__restrict const grad_scatter_thread,
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbydy,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receivers_i,
        int64_t const *__restrict const receiverssc_i, DW_DTYPE const rdy,
        DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
        DW_DTYPE const dt2, int64_t const nt, int64_t const n_shots,
        int64_t const ny, int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_sourcessc_per_shot, int64_t const n_receivers_per_shot,
        int64_t const n_receiverssc_per_shot, int64_t const step_ratio,
        bool const v_requires_grad, bool const scatter_requires_grad,
        bool const v_batched, bool const scatter_batched, int64_t start_t,
        int64_t const pml_y0, int64_t const pml_y1, int64_t const pml_x0,
        int64_t const pml_x1, int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const i = shot * ny * nx;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const sisc = shot * n_sourcessc_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * ny * nx;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    int64_t const grad_v_i = v_batched ? i : threadi;
    int64_t const grad_scatter_i = scatter_batched ? i : threadi;
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + i : v;
    DW_DTYPE const *__restrict const scatter_shot =
        scatter_batched ? scatter + i : scatter;
    int64_t t;
    for (t = start_t - 1; t >= start_t - nt; --t) {
      if ((start_t - 1 - t) & 1) {
        record_from_wavefield(wfp + i, sources_i + si,
                              grad_f + t * n_shots * n_sources_per_shot + si,
                              n_sources_per_shot);
        record_from_wavefield(
            wfpsc + i, sources_i + sisc,
            grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
            n_sourcessc_per_shot);
        backward_kernel(
            v_shot, scatter_shot, wfp + i, wfc + i, psiyn + i, psixn + i,
            psiy + i, psix + i, zetayn + i, zetaxn + i, zetay + i, zetax + i,
            wfpsc + i, wfcsc + i, psiynsc + i, psixnsc + i, psiysc + i,
            psixsc + i, zetaynsc + i, zetaxnsc + i, zetaysc + i, zetaxsc + i,
            w_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            wsc_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            grad_v_thread + grad_v_i, grad_scatter_thread + grad_scatter_i, ay,
            ax, by, bx, dbydy, dbxdx, rdy, rdx, rdy2, rdx2, dt2, ny, nx,
            step_ratio, v_requires_grad && ((t % step_ratio) == 0),
            scatter_requires_grad && ((t % step_ratio) == 0),
            pml_y0, pml_y1, pml_x0, pml_x1);
        add_to_wavefield(wfc + i, receivers_i + ri,
                         grad_r + t * n_shots * n_receivers_per_shot + ri,
                         n_receivers_per_shot);
        add_to_wavefield(wfcsc + i, receiverssc_i + risc,
                         grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                         n_receiverssc_per_shot);
      } else {
        record_from_wavefield(wfc + i, sources_i + si,
                              grad_f + t * n_shots * n_sources_per_shot + si,
                              n_sources_per_shot);
        record_from_wavefield(
            wfcsc + i, sources_i + sisc,
            grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
            n_sourcessc_per_shot);
        backward_kernel(
            v_shot, scatter_shot, wfc + i, wfp + i, psiy + i, psix + i,
            psiyn + i, psixn + i, zetay + i, zetax + i, zetayn + i, zetaxn + i,
            wfcsc + i, wfpsc + i, psiysc + i, psixsc + i, psiynsc + i,
            psixnsc + i, zetaysc + i, zetaxsc + i, zetaynsc + i, zetaxnsc + i,
            w_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            wsc_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            grad_v_thread + grad_v_i, grad_scatter_thread + grad_scatter_i, ay,
            ax, by, bx, dbydy, dbxdx, rdy, rdx, rdy2, rdx2, dt2, ny, nx,
            step_ratio, v_requires_grad && ((t % step_ratio) == 0),
            scatter_requires_grad && ((t % step_ratio) == 0),
            pml_y0, pml_y1, pml_x0, pml_x1);
        add_to_wavefield(wfp + i, receivers_i + ri,
                         grad_r + t * n_shots * n_receivers_per_shot + ri,
                         n_receivers_per_shot);
        add_to_wavefield(wfpsc + i, receiverssc_i + risc,
                         grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                         n_receiverssc_per_shot);
      }
    }
  }
#ifdef _OPENMP
  if (v_requires_grad && !v_batched && n_threads > 1) {
    combine_grad(grad_v, grad_v_thread, n_threads, ny, nx);
  }
  if (scatter_requires_grad && !scatter_batched && n_threads > 1) {
    combine_grad(grad_scatter, grad_scatter_thread, n_threads, ny, nx);
  }
#endif /* _OPENMP */
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FUNC(backward_sc)(
        DW_DTYPE const *__restrict const v,
        DW_DTYPE const *__restrict const grad_rsc,
        DW_DTYPE *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
        DW_DTYPE *__restrict const psiysc, DW_DTYPE *__restrict const psixsc,
        DW_DTYPE *__restrict const psiynsc, DW_DTYPE *__restrict const psixnsc,
        DW_DTYPE *__restrict const zetaysc, DW_DTYPE *__restrict const zetaxsc,
        DW_DTYPE *__restrict const zetaynsc,
        DW_DTYPE *__restrict const zetaxnsc,
        DW_DTYPE const *__restrict const w_store,
        DW_DTYPE *__restrict const grad_fsc,
        DW_DTYPE *__restrict const grad_scatter,
        DW_DTYPE *__restrict const grad_scatter_thread,
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbydy,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receiverssc_i, DW_DTYPE const rdy,
        DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
        DW_DTYPE const dt2, int64_t const nt, int64_t const n_shots,
        int64_t const ny, int64_t const nx, int64_t const n_sourcessc_per_shot,
        int64_t const n_receiverssc_per_shot, int64_t const step_ratio,
        bool const scatter_requires_grad, bool const v_batched,
        bool const scatter_batched, int64_t start_t, int64_t const pml_y0,
        int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
        int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const i = shot * ny * nx;
    int64_t const sisc = shot * n_sourcessc_per_shot;
    int64_t const risc = shot * n_receiverssc_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * ny * nx;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    int64_t const grad_scatter_i = scatter_batched ? i : threadi;
    DW_DTYPE const *__restrict const v_shot = v_batched ? v + i : v;
    int64_t t;
    for (t = start_t - 1; t >= start_t - nt; --t) {
      if ((start_t - 1 - t) & 1) {
        record_from_wavefield(
            wfpsc + i, sources_i + sisc,
            grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
            n_sourcessc_per_shot);
        backward_kernel_sc(
            v_shot, wfpsc + i, wfcsc + i, psiynsc + i, psixnsc + i, psiysc + i,
            psixsc + i, zetaynsc + i, zetaxnsc + i, zetaysc + i, zetaxsc + i,
            w_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            grad_scatter_thread + grad_scatter_i, ay, ax, by, bx, dbydy, dbxdx,
            rdy, rdx, rdy2, rdx2, dt2, ny, nx, step_ratio,
            scatter_requires_grad && ((t % step_ratio) == 0),
            pml_y0, pml_y1, pml_x0, pml_x1);
        add_to_wavefield(wfcsc + i, receiverssc_i + risc,
                         grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                         n_receiverssc_per_shot);
      } else {
        record_from_wavefield(
            wfcsc + i, sources_i + sisc,
            grad_fsc + t * n_shots * n_sourcessc_per_shot + sisc,
            n_sourcessc_per_shot);
        backward_kernel_sc(
            v_shot, wfcsc + i, wfpsc + i, psiysc + i, psixsc + i, psiynsc + i,
            psixnsc + i, zetaysc + i, zetaxsc + i, zetaynsc + i, zetaxnsc + i,
            w_store + (t / step_ratio) * n_shots * ny * nx + shot * ny * nx,
            grad_scatter_thread + grad_scatter_i, ay, ax, by, bx, dbydy, dbxdx,
            rdy, rdx, rdy2, rdx2, dt2, ny, nx, step_ratio,
            scatter_requires_grad && ((t % step_ratio) == 0),
            pml_y0, pml_y1, pml_x0, pml_x1);
        add_to_wavefield(wfpsc + i, receiverssc_i + risc,
                         grad_rsc + t * n_shots * n_receiverssc_per_shot + risc,
                         n_receiverssc_per_shot);
      }
    }
  }
#ifdef _OPENMP
  if (scatter_requires_grad && !scatter_batched && n_threads > 1) {
    combine_grad(grad_scatter, grad_scatter_thread, n_threads, ny, nx);
  }
#endif /* _OPENMP */
}

/*
 * Scalar wave equation propagator
 *
 * Assumptions:
 *  * The first and last accuracy/2 elements in each spatial dimension
 *    are zero in all wavefields (forward and backward).
 *  * Elements of ay, ax, by, bx are zero except for the first and last
 *    accuracy/2 + pml_width elements.
 *  * Elements of dbydx and dbxdx are zero except for the first and last
 *    accuracy + pml_width elements.
 *  * Elements of psiy and zetay are zero except for the first and last
 *    accuracy + pml_width elements in the y dimension for forward,
 *    and 3 * accuracy / 2 + pml_width elements in the y dimension for
 *    backward.
 *  * The previous assumption applies to psix and zetax in the x dimension.
 *  * The values in wfp are multiplied by -1 before and after calls to
 *    backward.
 */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>

#include "common.h"
#include "common_cpu.h"

#define CAT_I(name, accuracy, dtype) scalar_iso_##accuracy##_##dtype##_##name
#define CAT(name, accuracy, dtype) CAT_I(name, accuracy, dtype)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE)

#define WFC(dy, dx) wfc[i + dy * nx + dx]
#define AY_PSIY(dy, dx) ay[y + dy] * psiy[i + dy * nx + dx]
#define AX_PSIX(dy, dx) ax[x + dx] * psix[i + dy * nx + dx]
#define V2DT2_WFC(dy, dx) v2dt2[i + dy * nx + dx] * wfc[i + dy * nx + dx]
#define UT_TERMY1(dy, dx)                                           \
  ((dbydy[y + dy] *                                                 \
        ((1 + by[y + dy]) * v2dt2[i + dy * nx] * wfc[i + dy * nx] + \
         by[y + dy] * zetay[i + dy * nx]) +                         \
    by[y + dy] * psiy[i + dy * nx]))
#define UT_TERMX1(dy, dx)                                             \
  ((dbxdx[x + dx] * ((1 + bx[x + dx]) * v2dt2[i + dx] * wfc[i + dx] + \
                     bx[x + dx] * zetax[i + dx]) +                    \
    bx[x + dx] * psix[i + dx]))
#define UT_TERMY2(dy, dx)                                      \
  ((1 + by[y + dy]) *                                          \
   ((1 + by[y + dy]) * v2dt2[i + dy * nx] * wfc[i + dy * nx] + \
    by[y + dy] * zetay[i + dy * nx]))
#define UT_TERMX2(dy, dx)                                               \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * v2dt2[i + dx] * wfc[i + dx] + \
                       bx[x + dx] * zetax[i + dx]))
#define PSIY_TERM(dy, dx)                                     \
  ((1 + by[y + dy]) * v2dt2[i + dy * nx] * wfc[i + dy * nx] + \
   by[y + dy] * zetay[i + dy * nx])
#define PSIX_TERM(dy, dx) \
  ((1 + bx[x + dx]) * v2dt2[i + dx] * wfc[i + dx] + bx[x + dx] * zetax[i + dx])

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

#define FORWARD_KERNEL(pml_y, pml_x, v_requires_grad)                      \
  {                                                                        \
    SET_RANGE(pml_y, pml_x)                                                \
    for (y = y_begin; y < y_end; ++y) {                                    \
      for (x = x_begin; x < x_end; ++x) {                                  \
        int64_t const i = y * nx + x;                                      \
        if (pml_y == 1) {                                                  \
          w_sum = DIFFY2(WFC);                                             \
        } else {                                                           \
          DW_DTYPE dwfcdy = DIFFY1(WFC);                                   \
          DW_DTYPE tmpy = ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + \
                           DIFFY1(AY_PSIY));                               \
          w_sum = (1 + by[y]) * tmpy + ay[y] * zetay[i];                   \
          psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];                     \
          zetay[i] = by[y] * tmpy + ay[y] * zetay[i];                      \
        }                                                                  \
        if (pml_x == 1) {                                                  \
          w_sum += DIFFX2(WFC);                                            \
        } else {                                                           \
          DW_DTYPE dwfcdx = DIFFX1(WFC);                                   \
          DW_DTYPE tmpx = ((1 + bx[x]) * DIFFX2(WFC) + dbxdx[x] * dwfcdx + \
                           DIFFX1(AX_PSIX));                               \
          w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];                  \
          psixn[i] = bx[x] * dwfcdx + ax[x] * psix[i];                     \
          zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];                      \
        }                                                                  \
        if (v_requires_grad) {                                             \
          dwdv[i] = 2 * v[i] * dt2 * w_sum;                                \
        }                                                                  \
        wfp[i] = v[i] * v[i] * dt2 * w_sum + 2 * wfc[i] - wfp[i];          \
      }                                                                    \
    }                                                                      \
  }

#define BACKWARD_KERNEL(pml_y, pml_x, v_requires_grad)                   \
  {                                                                      \
    SET_RANGE(pml_y, pml_x)                                              \
    for (y = y_begin; y < y_end; ++y) {                                  \
      for (x = x_begin; x < x_end; ++x) {                                \
        int64_t const i = y * nx + x;                                    \
        wfp[i] = (pml_y == 1 ? DIFFY2(V2DT2_WFC)                         \
                             : -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2)) + \
                 (pml_x == 1 ? DIFFX2(V2DT2_WFC)                         \
                             : -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2)) + \
                 2 * wfc[i] - wfp[i];                                    \
        if (pml_y == 0 || pml_y == 2) {                                  \
          psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];       \
          zetayn[i] = ay[y] * v2dt2[i] * wfc[i] + ay[y] * zetay[i];      \
        }                                                                \
        if (pml_x == 0 || pml_x == 2) {                                  \
          psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];       \
          zetaxn[i] = ax[x] * v2dt2[i] * wfc[i] + ax[x] * zetax[i];      \
        }                                                                \
        if (v_requires_grad) {                                           \
          grad_v[i] += wfc[i] * dwdv[i] * (DW_DTYPE)step_ratio;          \
        }                                                                \
      }                                                                  \
    }                                                                    \
  }

static void forward_kernel(
    DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const wfc,
    DW_DTYPE *__restrict const wfp, DW_DTYPE const *__restrict const psiy,
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const dwdv,
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, DW_DTYPE const rdy,
    DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
    DW_DTYPE const dt2, int64_t const ny, int64_t const nx,
    bool const v_requires_grad, int64_t const pml_y0, int64_t const pml_y1,
    int64_t const pml_x0, int64_t const pml_x1) {
  int64_t y, x, y_begin, y_end, x_begin, x_end;
  DW_DTYPE w_sum;
  if (v_requires_grad) {
    FORWARD_KERNEL(0, 0, 1)
    FORWARD_KERNEL(0, 1, 1)
    FORWARD_KERNEL(0, 2, 1)
    FORWARD_KERNEL(1, 0, 1)
    FORWARD_KERNEL(1, 1, 1)
    FORWARD_KERNEL(1, 2, 1)
    FORWARD_KERNEL(2, 0, 1)
    FORWARD_KERNEL(2, 1, 1)
    FORWARD_KERNEL(2, 2, 1)
  } else {
    FORWARD_KERNEL(0, 0, 0)
    FORWARD_KERNEL(0, 1, 0)
    FORWARD_KERNEL(0, 2, 0)
    FORWARD_KERNEL(1, 0, 0)
    FORWARD_KERNEL(1, 1, 0)
    FORWARD_KERNEL(1, 2, 0)
    FORWARD_KERNEL(2, 0, 0)
    FORWARD_KERNEL(2, 1, 0)
    FORWARD_KERNEL(2, 2, 0)
  }
}

static void backward_kernel(
    DW_DTYPE const *__restrict const v2dt2,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE const *__restrict const psiy,
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
    DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const zetayn,
    DW_DTYPE *__restrict const zetaxn, DW_DTYPE const *__restrict const dwdv,
    DW_DTYPE *__restrict const grad_v, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const dbydy,
    DW_DTYPE const *__restrict const dbxdx, DW_DTYPE const rdy,
    DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
    int64_t const ny, int64_t const nx, int64_t const step_ratio,
    bool const v_requires_grad, int64_t const pml_y0, int64_t const pml_y1,
    int64_t const pml_x0, int64_t const pml_x1) {
  int64_t y, x, y_begin, y_end, x_begin, x_end;
  if (v_requires_grad) {
    BACKWARD_KERNEL(0, 0, 1)
    BACKWARD_KERNEL(0, 1, 1)
    BACKWARD_KERNEL(0, 2, 1)
    BACKWARD_KERNEL(1, 0, 1)
    BACKWARD_KERNEL(1, 1, 1)
    BACKWARD_KERNEL(1, 2, 1)
    BACKWARD_KERNEL(2, 0, 1)
    BACKWARD_KERNEL(2, 1, 1)
    BACKWARD_KERNEL(2, 2, 1)
  } else {
    BACKWARD_KERNEL(0, 0, 0)
    BACKWARD_KERNEL(0, 1, 0)
    BACKWARD_KERNEL(0, 2, 0)
    BACKWARD_KERNEL(1, 0, 0)
    BACKWARD_KERNEL(1, 1, 0)
    BACKWARD_KERNEL(1, 2, 0)
    BACKWARD_KERNEL(2, 0, 0)
    BACKWARD_KERNEL(2, 1, 0)
    BACKWARD_KERNEL(2, 2, 0)
  }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FUNC(forward)(
        DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const f,
        DW_DTYPE *__restrict const wfc, DW_DTYPE *__restrict const wfp,
        DW_DTYPE *__restrict const psiy, DW_DTYPE *__restrict const psix,
        DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn,
        DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax,
        DW_DTYPE *__restrict const dwdv, DW_DTYPE *__restrict const r,
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbydy,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receivers_i, DW_DTYPE const rdy,
        DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
        DW_DTYPE const dt2, int64_t const nt, int64_t const n_shots,
        int64_t const ny, int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        bool const v_requires_grad, int64_t const pml_y0, int64_t const pml_y1,
        int64_t const pml_x0, int64_t const pml_x1, int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const i = shot * ny * nx;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
    int64_t t;
    for (t = 0; t < nt; ++t) {
      if (t & 1) {
        forward_kernel(v, wfp + i, wfc + i, psiyn + i, psixn + i, psiy + i,
                       psix + i, zetay + i, zetax + i,
                       dwdv + shot * (nt / step_ratio) * ny * nx +
                           (t / step_ratio) * ny * nx,
                       ay, ax, by, bx, dbydy, dbxdx, rdy, rdx, rdy2, rdx2, dt2,
                       ny, nx, v_requires_grad && ((t % step_ratio) == 0),
                       pml_y0, pml_y1, pml_x0, pml_x1);
        add_sources(wfc + i, f + si * nt + t * n_sources_per_shot,
                    sources_i + si, n_sources_per_shot);
        record_receivers(r + ri * nt + t * n_receivers_per_shot, wfp + i,
                         receivers_i + ri, n_receivers_per_shot);
      } else {
        forward_kernel(v, wfc + i, wfp + i, psiy + i, psix + i, psiyn + i,
                       psixn + i, zetay + i, zetax + i,
                       dwdv + shot * (nt / step_ratio) * ny * nx +
                           (t / step_ratio) * ny * nx,
                       ay, ax, by, bx, dbydy, dbxdx, rdy, rdx, rdy2, rdx2, dt2,
                       ny, nx, v_requires_grad && ((t % step_ratio) == 0),
                       pml_y0, pml_y1, pml_x0, pml_x1);
        add_sources(wfp + i, f + si * nt + t * n_sources_per_shot,
                    sources_i + si, n_sources_per_shot);
        record_receivers(r + ri * nt + t * n_receivers_per_shot, wfc + i,
                         receivers_i + ri, n_receivers_per_shot);
      }
    }
  }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FUNC(backward)(
        DW_DTYPE const *__restrict const v2dt2,
        DW_DTYPE const *__restrict const grad_r, DW_DTYPE *__restrict const wfc,
        DW_DTYPE *__restrict const wfp, DW_DTYPE *__restrict const psiy,
        DW_DTYPE *__restrict const psix, DW_DTYPE *__restrict const psiyn,
        DW_DTYPE *__restrict const psixn, DW_DTYPE *__restrict const zetay,
        DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict const zetayn,
        DW_DTYPE *__restrict const zetaxn,
        DW_DTYPE const *__restrict const dwdv,
        DW_DTYPE *__restrict const grad_f, DW_DTYPE *__restrict const grad_v,
        DW_DTYPE *__restrict const grad_v_thread,
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const dbydy,
        DW_DTYPE const *__restrict const dbxdx,
        int64_t const *__restrict const sources_i,
        int64_t const *__restrict const receivers_i, DW_DTYPE const rdy,
        DW_DTYPE const rdx, DW_DTYPE const rdy2, DW_DTYPE const rdx2,
        int64_t const nt, int64_t const n_shots, int64_t const ny,
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        bool const v_requires_grad, int64_t const pml_y0, int64_t const pml_y1,
        int64_t const pml_x0, int64_t const pml_x1, int64_t const n_threads) {
  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const i = shot * ny * nx;
    int64_t const si = shot * n_sources_per_shot;
    int64_t const ri = shot * n_receivers_per_shot;
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * ny * nx;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */
    int64_t t;
    for (t = nt - 1; t >= 0; --t) {
      if ((nt - 1 - t) & 1) {
        record_receivers(grad_f + si * nt + t * n_sources_per_shot, wfp + i,
                         sources_i + si, n_sources_per_shot);
        backward_kernel(v2dt2, wfp + i, wfc + i, psiyn + i, psixn + i, psiy + i,
                        psix + i, zetayn + i, zetaxn + i, zetay + i, zetax + i,
                        dwdv + shot * (nt / step_ratio) * ny * nx +
                            (t / step_ratio) * ny * nx,
                        grad_v_thread + threadi, ay, ax, by, bx, dbydy, dbxdx,
                        rdy, rdx, rdy2, rdx2, ny, nx, step_ratio,
                        v_requires_grad && ((t % step_ratio) == 0), pml_y0,
                        pml_y1, pml_x0, pml_x1);
        add_sources(wfc + i, grad_r + ri * nt + t * n_receivers_per_shot,
                    receivers_i + ri, n_receivers_per_shot);
      } else {
        record_receivers(grad_f + si * nt + t * n_sources_per_shot, wfc + i,
                         sources_i + si, n_sources_per_shot);
        backward_kernel(v2dt2, wfc + i, wfp + i, psiy + i, psix + i, psiyn + i,
                        psixn + i, zetay + i, zetax + i, zetayn + i, zetaxn + i,
                        dwdv + shot * (nt / step_ratio) * ny * nx +
                            (t / step_ratio) * ny * nx,
                        grad_v_thread + threadi, ay, ax, by, bx, dbydy, dbxdx,
                        rdy, rdx, rdy2, rdx2, ny, nx, step_ratio,
                        v_requires_grad && ((t % step_ratio) == 0), pml_y0,
                        pml_y1, pml_x0, pml_x1);
        add_sources(wfp + i, grad_r + ri * nt + t * n_receivers_per_shot,
                    receivers_i + ri, n_receivers_per_shot);
      }
    }
  }
#ifdef _OPENMP
  if (v_requires_grad && n_threads > 1) {
    combine_grad(grad_v, grad_v_thread, n_threads, ny, nx);
  }
#endif /* _OPENMP */
}

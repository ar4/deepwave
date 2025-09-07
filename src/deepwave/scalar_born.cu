/*
 * Scalar Born wave equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the scalar Born wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * For a description of the method, see the C implementation in
 * scalar_born.c and scalar.c. This file implements the same functionality,
 * but for execution on a GPU using CUDA.
 */

#include <stdio.h>

#include <cstdint>

#include "common.h"

// Macro to concatenate function names with accuracy, dtype, and device for
// Python bindings
#define CAT_I(name, accuracy, dtype, device) \
  scalar_born_iso_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

// --- Memory access and finite difference macros ---
// All wavefields are stored in (shot, y, x) order, flattened as shot * (ny*nx)
// + y*nx + x Macros below provide access at offsets from the current location,
// which is useful for derivative calculations.
#define WFC(dy, dx) \
  wfc[i + dy * nx + dx]  // Background wavefield at (y+dy, x+dx)
#define WFCSC(dy, dx) \
  wfcsc[i + dy * nx + dx]  // Scattered wavefield at (y+dy, x+dx)
#define AY_PSIY(dy, dx)  \
  ay[y + dy] *           \
      psiy[i + dy * nx + \
           dx]  // PML profile ay times auxiliary field psiy (background)
#define AX_PSIX(dy, dx)  \
  ax[x + dx] *           \
      psix[i + dy * nx + \
           dx]  // PML profile ax times auxiliary field psix (background)
#define AY_PSIYSC(dy, dx)  \
  ay[y + dy] *             \
      psiysc[i + dy * nx + \
             dx]  // PML profile ay times auxiliary field psiysc (scattered)
#define AX_PSIXSC(dy, dx)  \
  ax[x + dx] *             \
      psixsc[i + dy * nx + \
             dx]  // PML profile ax times auxiliary field psixsc (scattered)
#define V(dy, dx) v_shot[j + dy * nx + dx]         // Velocity at (y+dy, x+dx)
#define VDT2(dy, dx) V(dy, dx) * dt2               // v * dt^2 at (y+dy, x+dx)
#define V2DT2(dy, dx) V(dy, dx) * V(dy, dx) * dt2  // v^2 * dt^2 at (y+dy, x+dx)
#define SCATTER(dy, dx) \
  scatter_shot[j + dy * nx + dx]  // Scattering potential at (y+dy, x+dx)
// Second derivative term in the backward background wavefield update
#define V2DT2_WFC(dy, dx)        \
  (V2DT2(dy, dx) * WFC(dy, dx) + \
   2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx))
// Second derivative term in the backward scattered wavefield update
#define V2DT2_WFCSC(dy, dx) V2DT2(dy, dx) * WFCSC(dy, dx)

// --- Macros for PML and auxiliary field updates ---
// Each macro below implements a specific term in the backward update equations
// for the PML (Perfectly Matched Layer) and auxiliary fields for both the
// background and scattered wavefields. These are used in the CUDA kernels.

// First derivative update term for y-derivative in PML region (background)
#define UT_TERMY1(dy, dx)                                                      \
  (dbydy[y + dy] * ((1 + by[y + dy]) *                                         \
                        (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                         2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
                    by[y + dy] * zetay[i + dy * nx]) +                         \
   by[y + dy] * psiy[i + dy * nx])

// First derivative update term for x-derivative in PML region (background)
#define UT_TERMX1(dy, dx)                                                      \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) *                                         \
                        (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                         2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
                    bx[x + dx] * zetax[i + dx]) +                              \
   bx[x + dx] * psix[i + dx])

// Second derivative update term for y-derivative in PML region (background)
#define UT_TERMY2(dy, dx)                                                     \
  ((1 + by[y + dy]) *                                                         \
   ((1 + by[y + dy]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                        2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
    by[y + dy] * zetay[i + dy * nx]))

// Second derivative update term for x-derivative in PML region (background)
#define UT_TERMX2(dy, dx)                                                     \
  ((1 + bx[x + dx]) *                                                         \
   ((1 + bx[x + dx]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                        2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
    bx[x + dx] * zetax[i + dx]))

// Term for y-derivative auxiliary field update (background)
#define PSIY_TERM(dy, dx)                                                    \
  ((1 + by[y + dy]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                       2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
   by[y + dy] * zetay[i + dy * nx])

// Term for x-derivative auxiliary field update (background)
#define PSIX_TERM(dy, dx)                                                    \
  ((1 + bx[x + dx]) * (V2DT2(dy, dx) * WFC(dy, dx) +                         \
                       2 * VDT2(dy, dx) * SCATTER(dy, dx) * WFCSC(dy, dx)) + \
   bx[x + dx] * zetax[i + dx])

// First derivative update term for y-derivative in PML region (scattered)
#define UTSC_TERMY1(dy, dx)                                             \
  ((dbydy[y + dy] * ((1 + by[y + dy]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                     by[y + dy] * zetaysc[i + dy * nx]) +               \
    by[y + dy] * psiysc[i + dy * nx]))

// First derivative update term for x-derivative in PML region (scattered)
#define UTSC_TERMX1(dy, dx)                                             \
  ((dbxdx[x + dx] * ((1 + bx[x + dx]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                     bx[x + dx] * zetaxsc[i + dx]) +                    \
    bx[x + dx] * psixsc[i + dx]))

// Second derivative update term for y-derivative in PML region (scattered)
#define UTSC_TERMY2(dy, dx)                                               \
  ((1 + by[y + dy]) * ((1 + by[y + dy]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                       by[y + dy] * zetaysc[i + dy * nx]))

// Second derivative update term for x-derivative in PML region (scattered)
#define UTSC_TERMX2(dy, dx)                                               \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
                       bx[x + dx] * zetaxsc[i + dx]))

// Term for y-derivative auxiliary field update (scattered)
#define PSIYSC_TERM(dy, dx)                           \
  ((1 + by[y + dy]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
   by[y + dy] * zetaysc[i + dy * nx])

// Term for x-derivative auxiliary field update (scattered)
#define PSIXSC_TERM(dy, dx)                           \
  ((1 + bx[x + dx]) * V2DT2(dy, dx) * WFCSC(dy, dx) + \
   bx[x + dx] * zetaxsc[i + dx])

// Macro for CUDA error checking
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

// --- Device constants for configuration ---
// These are copied to constant memory for fast access by all kernels
namespace {
__constant__ DW_DTYPE dt2;  // Time step squared
__constant__ DW_DTYPE rdy, rdx, rdy2,
    rdx2;  // Reciprocal grid spacings and their squares
__constant__ int64_t n_shots, ny, nx, nynx;  // Model and batch sizes
__constant__ int64_t n_sources_per_shot, n_sourcessc_per_shot;
__constant__ int64_t n_receivers_per_shot, n_receiverssc_per_shot;
__constant__ int64_t
    step_ratio;  // Number of steps between gradient contributions
__constant__ int64_t pml_y0, pml_y1, pml_x0, pml_x1;  // PML region bounds
__constant__ bool v_batched,
    scatter_batched;  // Whether v/scatter are shared or per shot

__global__ void add_sources_both(DW_DTYPE *__restrict const wf,
                                 DW_DTYPE *__restrict const wfsc,
                                 DW_DTYPE const *__restrict const f,
                                 DW_DTYPE const *__restrict const fsc,
                                 int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    if (0 <= sources_i[k]) {
      wf[shot_idx * nynx + sources_i[k]] += f[k];
      wfsc[shot_idx * nynx + sources_i[k]] += fsc[k];
    }
  }
}

__global__ void add_adjoint_sources(DW_DTYPE *__restrict const wf,
                                    DW_DTYPE const *__restrict const f,
                                    int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void add_adjoint_sourcessc(
    DW_DTYPE *__restrict const wf, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receiverssc_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receiverssc_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * nynx + sources_i[k]] += f[k];
  }
}

__global__ void record_receivers(DW_DTYPE *__restrict const r,
                                 DW_DTYPE const *__restrict const wf,
                                 int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

// Record scattered wavefield amplitudes at receiver locations for all shots.
__global__ void record_receiverssc(DW_DTYPE *__restrict const r,
                                   DW_DTYPE const *__restrict const wf,
                                   int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receiverssc_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receiverssc_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

// Record adjoint wavefield amplitudes at source locations for all shots (used
// in adjoint computations).
__global__ void record_adjoint_receivers(
    DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const wf,
    int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * nynx + receivers_i[k]];
  }
}

__global__ void record_adjoint_receiverssc(
    DW_DTYPE *__restrict const rsc, DW_DTYPE const *__restrict const wfsc,
    int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sourcessc_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sourcessc_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) rsc[k] = wfsc[shot_idx * nynx + receivers_i[k]];
  }
}

// Combine per-shot gradients into a single gradient array (for v or scatter).
__global__ void combine_grad(DW_DTYPE *__restrict const grad,
                             DW_DTYPE const *__restrict const grad_shot) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t i = y * nx + x;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx;
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * nynx + i];
    }
  }
}

// Main CUDA kernel for forward propagation of both background and scattered
// wavefields. Computes the next time step for both wavefields, including PML
// and auxiliary fields. Arguments:
//   v, scatter: velocity and scattering potential (can be batched per shot)
//   wfc, wfcsc: current background and scattered wavefields
//   wfp, wfpsc: previous (input) and next (output) time step wavefields
//   psiy, psix, psiyn, psixn, zetay, zetax: PML auxiliary fields (background)
//   psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc: PML auxiliary fields
//   (scattered) w_store, wsc_store: storage for gradient calculation snapshots
//   ay, ax, by, bx, dbydy, dbxdx: PML profiles and derivatives
//   v_requires_grad, scatter_requires_grad: whether to store values for
//   gradient computation
__global__ void forward_kernel(
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
    DW_DTYPE const *__restrict const dbxdx, bool const v_requires_grad,
    bool const scatter_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    // Select velocity and scatter arrays for this shot (batched or shared)
    DW_DTYPE const *__restrict const v_shot =
        v_batched ? v + shot_idx * nynx : v;
    DW_DTYPE const *__restrict const scatter_shot =
        scatter_batched ? scatter + shot_idx * nynx : scatter;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    DW_DTYPE w_sum, wsc_sum;
    if (!pml_y) {
      // Central region: standard finite difference for y-derivative
      w_sum = DIFFY2(WFC);
      wsc_sum = DIFFY2(WFCSC);
    } else {
      // PML region: use auxiliary fields and PML profiles for y-derivative
      DW_DTYPE dwfcdy = DIFFY1(WFC);
      DW_DTYPE tmpy =
          ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
      DW_DTYPE dwfcscdy = DIFFY1(WFCSC);
      DW_DTYPE tmpysc = ((1 + by[y]) * DIFFY2(WFCSC) + dbydy[y] * dwfcscdy +
                         DIFFY1(AY_PSIYSC));
      w_sum = (1 + by[y]) * tmpy + ay[y] * zetay[i];
      wsc_sum = (1 + by[y]) * tmpysc + ay[y] * zetaysc[i];
      psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];
      zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
      psiynsc[i] = by[y] * dwfcscdy + ay[y] * psiysc[i];
      zetaysc[i] = by[y] * tmpysc + ay[y] * zetaysc[i];
    }
    if (!pml_x) {
      // Central region: standard finite difference for x-derivative
      w_sum += DIFFX2(WFC);
      wsc_sum += DIFFX2(WFCSC);
    } else {
      // PML region: use auxiliary fields and PML profiles for x-derivative
      DW_DTYPE dwfcdx = DIFFX1(WFC);
      DW_DTYPE tmpx =
          ((1 + bx[x]) * DIFFX2(WFC) + dbxdx[x] * dwfcdx + DIFFX1(AX_PSIX));
      DW_DTYPE dwfcscdx = DIFFX1(WFCSC);
      DW_DTYPE tmpxsc = ((1 + bx[x]) * DIFFX2(WFCSC) + dbxdx[x] * dwfcscdx +
                         DIFFX1(AX_PSIXSC));
      w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];
      wsc_sum += (1 + bx[x]) * tmpxsc + ax[x] * zetaxsc[i];
      psixn[i] = bx[x] * dwfcdx + ax[x] * psix[i];
      zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];
      psixnsc[i] = bx[x] * dwfcscdx + ax[x] * psixsc[i];
      zetaxsc[i] = bx[x] * tmpxsc + ax[x] * zetaxsc[i];
    }
    // Update background and scattered wavefields for next time step
    wfp[i] = v_shot[j] * v_shot[j] * dt2 * w_sum + 2 * wfc[i] - wfp[i];
    wfpsc[i] = v_shot[j] * v_shot[j] * dt2 * wsc_sum + 2 * wfcsc[i] - wfpsc[i] +
               2 * v_shot[j] * scatter_shot[j] * dt2 * w_sum;
    // Store values for gradient calculation if needed
    if (v_requires_grad || scatter_requires_grad) {
      w_store[i] = w_sum;
    }
    if (v_requires_grad) {
      wsc_store[i] = wsc_sum;
    }
  }
}

// Main CUDA kernel for backward (adjoint) propagation for both background and
// scattered wavefields. Computes gradients with respect to velocity and scatter
// if requested. Arguments are analogous to forward_kernel, with additional
// gradient outputs.
__global__ void backward_kernel(
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
    DW_DTYPE const *__restrict const dbxdx, bool const v_requires_grad,
    bool const scatter_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    // Select velocity and scatter arrays for this shot (batched or shared)
    DW_DTYPE const *__restrict const v_shot =
        v_batched ? v + shot_idx * nynx : v;
    DW_DTYPE const *__restrict const scatter_shot =
        scatter_batched ? scatter + shot_idx * nynx : scatter;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    // Update background and scattered wavefields for next time step
    // (adjoint/backward)
    wfp[i] =
        (pml_y ? -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2) : DIFFY2(V2DT2_WFC)) +
        (pml_x ? -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2) : DIFFX2(V2DT2_WFC)) +
        2 * wfc[i] - wfp[i];
    wfpsc[i] = (pml_y ? -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2)
                      : DIFFY2(V2DT2_WFCSC)) +
               (pml_x ? -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2)
                      : DIFFX2(V2DT2_WFCSC)) +
               2 * wfcsc[i] - wfpsc[i];
    // Update PML auxiliary fields for y/x directions (background and scattered)
    if (pml_y) {
      psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];
      zetaynsc[i] = ay[y] * V2DT2(0, 0) * wfcsc[i] + ay[y] * zetaysc[i];
      psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
      zetayn[i] = ay[y] * V2DT2(0, 0) * wfc[i] +
                  ay[y] * 2 * VDT2(0, 0) * SCATTER(0, 0) * wfcsc[i] +
                  ay[y] * zetay[i];
    }
    if (pml_x) {
      psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];
      zetaxnsc[i] = ax[x] * V2DT2(0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];
      psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
      zetaxn[i] = ax[x] * V2DT2(0, 0) * wfc[i] +
                  ax[x] * 2 * VDT2(0, 0) * SCATTER(0, 0) * wfcsc[i] +
                  ax[x] * zetax[i];
    }
    // Accumulate gradients for velocity and scatter if required
    if (v_requires_grad) {
      grad_v[i] += wfc[i] * 2 * v_shot[j] * dt2 * w_store[i] * step_ratio +
                   wfcsc[i] *
                       (2 * dt2 * scatter_shot[j] * w_store[i] +
                        2 * v_shot[j] * dt2 * wsc_store[i]) *
                       step_ratio;
    }
    if (scatter_requires_grad) {
      grad_scatter[i] +=
          wfcsc[i] * 2 * v_shot[j] * dt2 * w_store[i] * step_ratio;
    }
  }
}

// Specialized backward kernel for computing only the scatter gradient.
__global__ void backward_kernel_sc(
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
    DW_DTYPE const *__restrict const dbxdx, bool const scatter_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * nynx + j;
    // Select velocity array for this shot (batched or shared)
    DW_DTYPE const *__restrict const v_shot =
        v_batched ? v + shot_idx * nynx : v;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    // Update scattered wavefield for next time step (adjoint/backward)
    wfpsc[i] = (pml_y ? -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2)
                      : DIFFY2(V2DT2_WFCSC)) +
               (pml_x ? -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2)
                      : DIFFX2(V2DT2_WFCSC)) +
               2 * wfcsc[i] - wfpsc[i];
    // Update PML auxiliary fields for y/x directions (scattered)
    if (pml_y) {
      psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];
      zetaynsc[i] = ay[y] * V2DT2(0, 0) * wfcsc[i] + ay[y] * zetaysc[i];
    }
    if (pml_x) {
      psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];
      zetaxnsc[i] = ax[x] * V2DT2(0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];
    }
    // Accumulate gradient for scatter if required
    if (scatter_requires_grad) {
      grad_scatter[i] +=
          wfcsc[i] * 2 * v_shot[j] * dt2 * w_store[i] * step_ratio;
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

// Copy configuration parameters from host to device constant memory
void set_config(
    DW_DTYPE const dt2_h, DW_DTYPE const rdy_h, DW_DTYPE const rdx_h,
    DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_sourcessc_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
    int64_t const pml_y0_h, int64_t const pml_y1_h, int64_t const pml_x0_h,
    int64_t const pml_x1_h, bool const v_batched_h,
    bool const scatter_batched_h) {
  int64_t const nynx_h = ny_h * nx_h;
  gpuErrchk(cudaMemcpyToSymbol(dt2, &dt2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy2, &rdy2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx2, &rdx2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nynx, &nynx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sourcessc_per_shot, &n_sourcessc_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receiverssc_per_shot,
                               &n_receiverssc_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(v_batched, &v_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(scatter_batched, &scatter_batched_h, sizeof(bool)));
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(forward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const scatter,
            DW_DTYPE const *__restrict const f,
            DW_DTYPE const *__restrict const fsc,
            DW_DTYPE *__restrict const wfc, DW_DTYPE *__restrict const wfp,
            DW_DTYPE *__restrict const psiy, DW_DTYPE *__restrict const psix,
            DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn,
            DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax,
            DW_DTYPE *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
            DW_DTYPE *__restrict const psiysc,
            DW_DTYPE *__restrict const psixsc,
            DW_DTYPE *__restrict const psiynsc,
            DW_DTYPE *__restrict const psixnsc,
            DW_DTYPE *__restrict const zetaysc,
            DW_DTYPE *__restrict const zetaxsc,
            DW_DTYPE *__restrict const w_store,
            DW_DTYPE *__restrict const wsc_store, DW_DTYPE *__restrict const r,
            DW_DTYPE *__restrict const rsc, DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbydy,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i,
            int64_t const *__restrict const receiverssc_i, DW_DTYPE const rdy_h,
            DW_DTYPE const rdx_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            DW_DTYPE const dt2_h, int64_t const nt, int64_t const n_shots_h,
            int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_per_shot_h,
            int64_t const n_receivers_per_shot_h,
            int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
            bool const v_requires_grad, bool const scatter_requires_grad,
            bool const v_batched_h, bool const scatter_batched_h,
            int64_t const start_t, int64_t const pml_y0_h,
            int64_t const pml_y1_h, int64_t const pml_x0_h,
            int64_t const pml_x1_h, int64_t const device) {

  // --- CUDA kernel launch configuration ---
  dim3 dimBlock(32, 32, 1);  // For wavefield update
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  // For source and receiver operations
  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);
  dim3 dimBlock_receiverssc(32, 1, 1);
  unsigned int gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int gridy_receiverssc = ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int gridz_receiverssc = 1;
  dim3 dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                           gridz_receiverssc);

  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt2_h, rdy_h, rdx_h, rdy2_h, rdx2_h, n_shots_h, ny_h, nx_h,
             n_sources_per_shot_h, n_sources_per_shot_h, n_receivers_per_shot_h,
             n_receiverssc_per_shot_h, step_ratio_h, pml_y0_h, pml_y1_h,
             pml_x0_h, pml_x1_h, v_batched_h, scatter_batched_h);
  // --- Time-stepping loop for forward propagation ---
  // Alternates between wfc/wfp and wfp/wfc for memory efficiency
  for (t = start_t; t < start_t + nt; ++t) {
    if ((t - start_t) & 1) {
      // Launch CUDA kernel for forward propagation (odd/even time step)
      forward_kernel<<<dimGrid, dimBlock>>>(
          v, scatter, wfp, wfc, psiyn, psixn, psiy, psix, zetay, zetax, wfpsc,
          wfcsc, psiynsc, psixnsc, psiysc, psixsc, zetaysc, zetaxsc,
          w_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h,
          wsc_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h, ay, ax, by,
          bx, dbydy, dbxdx,
          v_requires_grad && ((t % step_ratio_h) == 0),
          scatter_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      // Add sources to both background and scattered wavefields
      if (n_sources_per_shot_h > 0) {
        add_sources_both<<<dimGrid_sources, dimBlock_sources>>>(
            wfc, wfcsc, f + t * n_shots_h * n_sources_per_shot_h,
            fsc + t * n_shots_h * n_sources_per_shot_h, sources_i);
        CHECK_KERNEL_ERROR
      }
      // Record background wavefield at receiver locations
      if (n_receivers_per_shot_h > 0) {
        record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
            r + t * n_shots_h * n_receivers_per_shot_h, wfp, receivers_i);
        CHECK_KERNEL_ERROR
      }
      // Record scattered wavefield at receiver locations
      if (n_receiverssc_per_shot_h > 0) {
        record_receiverssc<<<dimGrid_receiverssc, dimBlock_receiverssc>>>(
            rsc + t * n_shots_h * n_receiverssc_per_shot_h, wfpsc,
            receiverssc_i);
        CHECK_KERNEL_ERROR
      }

    } else {
      // Launch CUDA kernel for forward propagation (even/odd time step)
      forward_kernel<<<dimGrid, dimBlock>>>(
          v, scatter, wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, wfcsc,
          wfpsc, psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc,
          w_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h,
          wsc_store + (t / step_ratio_h) * ny_h * nx_h * n_shots_h, ay, ax, by,
          bx, dbydy, dbxdx,
          v_requires_grad && ((t % step_ratio_h) == 0),
          scatter_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      // Add sources to both background and scattered wavefields
      if (n_sources_per_shot_h > 0) {
        add_sources_both<<<dimGrid_sources, dimBlock_sources>>>(
            wfp, wfpsc, f + t * n_shots_h * n_sources_per_shot_h,
            fsc + t * n_shots_h * n_sources_per_shot_h, sources_i);
        CHECK_KERNEL_ERROR
      }
      // Record background wavefield at receiver locations
      if (n_receivers_per_shot_h > 0) {
        record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
            r + t * n_shots_h * n_receivers_per_shot_h, wfc, receivers_i);
        CHECK_KERNEL_ERROR
      }
      // Record scattered wavefield at receiver locations
      if (n_receiverssc_per_shot_h > 0) {
        record_receiverssc<<<dimGrid_receiverssc, dimBlock_receiverssc>>>(
            rsc + t * n_shots_h * n_receiverssc_per_shot_h, wfcsc,
            receiverssc_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
}

//
// Backward (adjoint) propagation for the Born model.
// This function launches the backward CUDA kernels in a time-reversed
// loop, handling both the background and scattered wavefields, and accumulates
// gradients for velocity and scatter.
//
// Arguments:
//   v, scatter: velocity and scattering potential arrays
//   grad_r, grad_rsc: receiver gradients for background and scattered fields
//   wfc, wfp, ...: wavefield and auxiliary arrays (see forward for details)
//   grad_v, grad_scatter: output gradients for velocity and scatter
//   grad_v_shot, grad_scatter_shot: per-shot gradients (for reduction)
//   ay, ax, by, bx, dbydy, dbxdx: PML profiles and derivatives
//   sources_i, receivers_i, receiverssc_i: source/receiver indices
//   ... and various grid, time, and configuration parameters
//
extern "C"
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
            DW_DTYPE *__restrict const zetayn,
            DW_DTYPE *__restrict const zetaxn, DW_DTYPE *__restrict const wfcsc,
            DW_DTYPE *__restrict const wfpsc, DW_DTYPE *__restrict const psiysc,
            DW_DTYPE *__restrict const psixsc,
            DW_DTYPE *__restrict const psiynsc,
            DW_DTYPE *__restrict const psixnsc,
            DW_DTYPE *__restrict const zetaysc,
            DW_DTYPE *__restrict const zetaxsc,
            DW_DTYPE *__restrict const zetaynsc,
            DW_DTYPE *__restrict const zetaxnsc,
            DW_DTYPE const *__restrict const w_store,
            DW_DTYPE const *__restrict const wsc_store,
            DW_DTYPE *__restrict const grad_f,
            DW_DTYPE *__restrict const grad_fsc,
            DW_DTYPE *__restrict const grad_v,
            DW_DTYPE *__restrict const grad_scatter,
            DW_DTYPE *__restrict const grad_v_shot,
            DW_DTYPE *__restrict const grad_scatter_shot,
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbydy,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i,
            int64_t const *__restrict const receiverssc_i, DW_DTYPE const rdy_h,
            DW_DTYPE const rdx_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            DW_DTYPE const dt2_h, int64_t const nt, int64_t const n_shots_h,
            int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_per_shot_h,
            int64_t const n_sourcessc_per_shot_h,
            int64_t const n_receivers_per_shot_h,
            int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
            bool const v_requires_grad, bool const scatter_requires_grad,
            bool const v_batched_h, bool const scatter_batched_h,
            int64_t const start_t, int64_t const pml_y0_h,
            int64_t const pml_y1_h, int64_t const pml_x0_h,
            int64_t const pml_x1_h, int64_t const device) {

  dim3 dimBlock(32, 4, 1);
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 dimBlock_sourcessc(32, 1, 1);
  unsigned int gridx_sourcessc =
      ceil_div(n_sourcessc_per_shot_h, dimBlock_sourcessc.x);
  unsigned int gridy_sourcessc = ceil_div(n_shots_h, dimBlock_sourcessc.y);
  unsigned int gridz_sourcessc = 1;
  dim3 dimGrid_sourcessc(gridx_sourcessc, gridy_sourcessc, gridz_sourcessc);
  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);
  dim3 dimBlock_receiverssc(32, 1, 1);
  unsigned int gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int gridy_receiverssc = ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int gridz_receiverssc = 1;
  dim3 dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                           gridz_receiverssc);
  dim3 dimBlock_combine(32, 32, 1);
  unsigned int gridx_combine = ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int gridy_combine = ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int gridz_combine = 1;
  dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt2_h, rdy_h, rdx_h, rdy2_h, rdx2_h, n_shots_h, ny_h, nx_h,
             n_sources_per_shot_h, n_sourcessc_per_shot_h,
             n_receivers_per_shot_h, n_receiverssc_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, v_batched_h,
             scatter_batched_h);
  // --- Time-reversed loop for adjoint propagation ---
  // Alternates between wfc/wfp and wfp/wfc for memory efficiency
  for (t = start_t - 1; t >= start_t - nt; --t) {
    // Odd/even time step logic for ping-ponging wavefield arrays
    if ((start_t - 1 - t) & 1) {
      // Record adjoint sources for background and scattered fields
      if (n_sources_per_shot_h > 0) {
        record_adjoint_receivers<<<dimGrid_sources, dimBlock_sources>>>(
            grad_f + t * n_shots_h * n_sources_per_shot_h, wfp, sources_i);
        CHECK_KERNEL_ERROR
      }
      if (n_sourcessc_per_shot_h > 0) {
        record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc>>>(
            grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfpsc,
            sources_i);
        CHECK_KERNEL_ERROR
      }
      // Launch backward kernel for this time step
      backward_kernel<<<dimGrid, dimBlock>>>(
          v, scatter, wfp, wfc, psiyn, psixn, psiy, psix, zetayn, zetaxn, zetay,
          zetax, wfpsc, wfcsc, psiynsc, psixnsc, psiysc, psixsc, zetaynsc,
          zetaxnsc, zetaysc, zetaxsc,
          w_store + (t / step_ratio_h) * n_shots_h * ny_h * nx_h,
          wsc_store + (t / step_ratio_h) * n_shots_h * ny_h * nx_h, grad_v_shot,
          grad_scatter_shot, ay, ax, by, bx, dbydy, dbxdx,
          v_requires_grad && ((t % step_ratio_h) == 0),
          scatter_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      // Add receiver gradients as adjoint sources
      if (n_receivers_per_shot_h > 0) {
        add_adjoint_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
            wfc, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
        CHECK_KERNEL_ERROR
      }
      if (n_receiverssc_per_shot_h > 0) {
        add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc>>>(
            wfcsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
            receiverssc_i);
        CHECK_KERNEL_ERROR
      }
    } else {
      // Record source gradients for background and scattered fields
      if (n_sources_per_shot_h > 0) {
        record_adjoint_receivers<<<dimGrid_sources, dimBlock_sources>>>(
            grad_f + t * n_shots_h * n_sources_per_shot_h, wfc, sources_i);
        CHECK_KERNEL_ERROR
      }
      if (n_sourcessc_per_shot_h > 0) {
        record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc>>>(
            grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfcsc,
            sources_i);
        CHECK_KERNEL_ERROR
      }
      // Launch backward kernel for this time step
      backward_kernel<<<dimGrid, dimBlock>>>(
          v, scatter, wfc, wfp, psiy, psix, psiyn, psixn, zetay, zetax, zetayn,
          zetaxn, wfcsc, wfpsc, psiysc, psixsc, psiynsc, psixnsc, zetaysc,
          zetaxsc, zetaynsc, zetaxnsc,
          w_store + (t / step_ratio_h) * n_shots_h * ny_h * nx_h,
          wsc_store + (t / step_ratio_h) * n_shots_h * ny_h * nx_h, grad_v_shot,
          grad_scatter_shot, ay, ax, by, bx, dbydy, dbxdx,
          v_requires_grad && ((t % step_ratio_h) == 0),
          scatter_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      // Add receiver gradients as adjoint sources
      if (n_receivers_per_shot_h > 0) {
        add_adjoint_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
            wfp, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
        CHECK_KERNEL_ERROR
      }
      if (n_receiverssc_per_shot_h > 0) {
        add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc>>>(
            wfpsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
            receiverssc_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
  // Reduce per-shot gradients to a single array if needed
  if (v_requires_grad && !v_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_v, grad_v_shot);
    CHECK_KERNEL_ERROR
  }
  if (scatter_requires_grad && !scatter_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_scatter,
                                                        grad_scatter_shot);
    CHECK_KERNEL_ERROR
  }
}

//
// Specialized entry point for the CUDA backward (adjoint) propagation for the
// scattered field only. This is an optimisation used when only the scatter
// gradient is needed.
//
// Arguments are analogous to FUNC(backward), but only for the scattered field
// and its gradient.
//
extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(backward_sc)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const grad_rsc,
            DW_DTYPE *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
            DW_DTYPE *__restrict const psiysc,
            DW_DTYPE *__restrict const psixsc,
            DW_DTYPE *__restrict const psiynsc,
            DW_DTYPE *__restrict const psixnsc,
            DW_DTYPE *__restrict const zetaysc,
            DW_DTYPE *__restrict const zetaxsc,
            DW_DTYPE *__restrict const zetaynsc,
            DW_DTYPE *__restrict const zetaxnsc,
            DW_DTYPE const *__restrict const w_store,
            DW_DTYPE *__restrict const grad_fsc,
            DW_DTYPE *__restrict const grad_scatter,
            DW_DTYPE *__restrict const grad_scatter_shot,
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbydy,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receiverssc_i, DW_DTYPE const rdy_h,
            DW_DTYPE const rdx_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            DW_DTYPE const dt2_h, int64_t const nt, int64_t const n_shots_h,
            int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sourcessc_per_shot_h,
            int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
            bool const scatter_requires_grad, bool const v_batched_h,
            bool const scatter_batched_h, int64_t const start_t,
            int64_t const pml_y0_h, int64_t const pml_y1_h,
            int64_t const pml_x0_h, int64_t const pml_x1_h,
            int64_t const device) {

  dim3 dimBlock(32, 8, 1);
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(n_shots_h, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);
  dim3 dimBlock_sourcessc(32, 1, 1);
  unsigned int gridx_sourcessc =
      ceil_div(n_sourcessc_per_shot_h, dimBlock_sourcessc.x);
  unsigned int gridy_sourcessc = ceil_div(n_shots_h, dimBlock_sourcessc.y);
  unsigned int gridz_sourcessc = 1;
  dim3 dimGrid_sourcessc(gridx_sourcessc, gridy_sourcessc, gridz_sourcessc);
  dim3 dimBlock_receiverssc(32, 1, 1);
  unsigned int gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int gridy_receiverssc = ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int gridz_receiverssc = 1;
  dim3 dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                           gridz_receiverssc);
  dim3 dimBlock_combine(32, 32, 1);
  unsigned int gridx_combine = ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int gridy_combine = ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int gridz_combine = 1;
  dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);
  int64_t t;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt2_h, rdy_h, rdx_h, rdy2_h, rdx2_h, n_shots_h, ny_h, nx_h,
             n_sourcessc_per_shot_h, n_sourcessc_per_shot_h,
             n_receiverssc_per_shot_h, n_receiverssc_per_shot_h, step_ratio_h,
             pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, v_batched_h,
             scatter_batched_h);
  // --- Time-reversed loop for adjoint propagation (scattered field only)
  // ---
  for (t = start_t - 1; t >= start_t - nt; --t) {
    // Odd/even time step logic for ping-ponging wavefield arrays
    if ((start_t - 1 - t) & 1) {
      // Record source gradients for scattered field
      if (n_sourcessc_per_shot_h > 0) {
        record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc>>>(
            grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfpsc,
            sources_i);
        CHECK_KERNEL_ERROR
      }
      // Launch backward kernel for this time step (scattered field only)
      backward_kernel_sc<<<dimGrid, dimBlock>>>(
          v, wfpsc, wfcsc, psiynsc, psixnsc, psiysc, psixsc, zetaynsc, zetaxnsc,
          zetaysc, zetaxsc,
          w_store + (t / step_ratio_h) * n_shots_h * ny_h * nx_h,
          grad_scatter_shot, ay, ax, by, bx, dbydy, dbxdx,
          scatter_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      // Add receiver gradients as adjoint sources
      if (n_receiverssc_per_shot_h > 0) {
        add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc>>>(
            wfcsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
            receiverssc_i);
        CHECK_KERNEL_ERROR
      }
    } else {
      // Record source gradients for scattered field
      if (n_sourcessc_per_shot_h > 0) {
        record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc>>>(
            grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfcsc,
            sources_i);
        CHECK_KERNEL_ERROR
      }
      // Launch backward kernel for this time step (scattered field only)
      backward_kernel_sc<<<dimGrid, dimBlock>>>(
          v, wfcsc, wfpsc, psiysc, psixsc, psiynsc, psixnsc, zetaysc, zetaxsc,
          zetaynsc, zetaxnsc,
          w_store + (t / step_ratio_h) * n_shots_h * ny_h * nx_h,
          grad_scatter_shot, ay, ax, by, bx, dbydy, dbxdx,
          scatter_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
      // Add receiver gradients as adjoint sources
      if (n_receiverssc_per_shot_h > 0) {
        add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc>>>(
            wfpsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
            receiverssc_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
  // Reduce per-shot gradients to a single array if needed
  if (scatter_requires_grad && !scatter_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_scatter,
                                                        grad_scatter_shot);
    CHECK_KERNEL_ERROR
  }
}

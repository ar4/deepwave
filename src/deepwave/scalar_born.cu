/*
 * Scalar Born wave equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the scalar Born wave equation
 * propagator. It is compiled multiple times with different options
 * to generate a set of functions that can be called from Python.
 * The options are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 1-3.
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
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>

#include "common_gpu.h"
#include "regular_grid.h"
#include "storage_utils.h"

// Macro to concatenate function names with accuracy, dtype, and device for
// Python bindings
#define CAT_I(name, ndim, accuracy, dtype, device) \
  scalar_born_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

// --- Memory access and finite difference macros ---
// All wavefields are stored in (shot, z, y, x) order, flattened as shot *
// shot_numel + z*ny*nx + y*nx + x Macros below provide access at offsets from
// the current location, which is useful for derivative calculations.

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

// Access the background wavefield at offset (dz, dy, dx) from i
#define WFC(dz, dy, dx) wfc[ND_INDEX(i, dz, dy, dx)]
// Access the scattered wavefield at offset (dz, dy, dx) from i
#define WFCSC(dz, dy, dx) wfcsc[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define PSIZ(dz, dy, dx) psiz[ND_INDEX(i, dz, dy, dx)]
#define ZETAZ(dz, dy, dx) zetaz[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZ(dz, dy, dx) az[z + (dz)] * PSIZ(dz, dy, dx)
#define PSIZSC(dz, dy, dx) psizsc[ND_INDEX(i, dz, dy, dx)]
#define ZETAZSC(dz, dy, dx) zetazsc[ND_INDEX(i, dz, dy, dx)]
#define AZ_PSIZSC(dz, dy, dx) az[z + (dz)] * PSIZSC(dz, dy, dx)
#endif

#if DW_NDIM >= 2
#define PSIY(dz, dy, dx) psiy[ND_INDEX(i, dz, dy, dx)]
#define ZETAY(dz, dy, dx) zetay[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIY(dz, dy, dx) ay[y + (dy)] * PSIY(dz, dy, dx)
#define PSIYSC(dz, dy, dx) psiysc[ND_INDEX(i, dz, dy, dx)]
#define ZETAYSC(dz, dy, dx) zetaysc[ND_INDEX(i, dz, dy, dx)]
#define AY_PSIYSC(dz, dy, dx) ay[y + (dy)] * PSIYSC(dz, dy, dx)
#endif

#define PSIX(dz, dy, dx) psix[ND_INDEX(i, dz, dy, dx)]
#define ZETAX(dz, dy, dx) zetax[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIX(dz, dy, dx) ax[x + (dx)] * PSIX(dz, dy, dx)
#define PSIXSC(dz, dy, dx) psixsc[ND_INDEX(i, dz, dy, dx)]
#define ZETAXSC(dz, dy, dx) zetaxsc[ND_INDEX(i, dz, dy, dx)]
#define AX_PSIXSC(dz, dy, dx) ax[x + (dx)] * PSIXSC(dz, dy, dx)

// Access velocity at offset (dz, dy, dx) from i
#define V(dz, dy, dx) v_shot[ND_INDEX(j, dz, dy, dx)]
// v * dt^2 at offset
#define VDT2(dz, dy, dx) V(dz, dy, dx) * dt2
// v^2 * dt^2 at offset
#define V2DT2(dz, dy, dx) V(dz, dy, dx) * V(dz, dy, dx) * dt2
// Scattering potential at offset
#define SCATTER(dz, dy, dx) scatter_shot[ND_INDEX(j, dz, dy, dx)]
// Second derivative term in the backward background wavefield update
#define V2DT2_WFC(dz, dy, dx)            \
  (V2DT2(dz, dy, dx) * WFC(dz, dy, dx) + \
   2 * VDT2(dz, dy, dx) * SCATTER(dz, dy, dx) * WFCSC(dz, dy, dx))
// Second derivative term in the backward scattered wavefield update
#define V2DT2_WFCSC(dz, dy, dx) V2DT2(dz, dy, dx) * WFCSC(dz, dy, dx)

// --- Macros for PML and auxiliary field updates ---
// Each macro below implements a specific term in the backward update equations
// for the PML (Perfectly Matched Layer) and auxiliary fields for both the
// background and scattered wavefields. These are used in the CUDA kernels.

#if DW_NDIM >= 3
#define UT_TERMZ1(dz, dy, dx)                                \
  (dbzdz[z + dz] * ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + \
                    bz[z + dz] * ZETAZ(dz, 0, 0)) +          \
   bz[z + dz] * PSIZ(dz, 0, 0))
#define UT_TERMZ2(dz, dy, dx) \
  ((1 + bz[z + dz]) *         \
   ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + bz[z + dz] * ZETAZ(dz, 0, 0)))
#define PSIZ_TERM(dz, dy, dx) \
  ((1 + bz[z + dz]) * V2DT2_WFC(dz, 0, 0) + bz[z + dz] * ZETAZ(dz, 0, 0))
#define UTSC_TERMZ1(dz, dy, dx)                                \
  (dbzdz[z + dz] * ((1 + bz[z + dz]) * V2DT2_WFCSC(dz, 0, 0) + \
                    bz[z + dz] * ZETAZSC(dz, 0, 0)) +          \
   bz[z + dz] * PSIZSC(dz, 0, 0))
#define UTSC_TERMZ2(dz, dy, dx)                                   \
  ((1 + bz[z + dz]) * ((1 + bz[z + dz]) * V2DT2_WFCSC(dz, 0, 0) + \
                       bz[z + dz] * ZETAZSC(dz, 0, 0)))
#define PSIZSC_TERM(dz, dy, dx) \
  ((1 + bz[z + dz]) * V2DT2_WFCSC(dz, 0, 0) + bz[z + dz] * ZETAZSC(dz, 0, 0))
#endif

#if DW_NDIM >= 2
// First derivative update term for y-derivative in PML region (background)
#define UT_TERMY1(dz, dy, dx)                                \
  (dbydy[y + dy] * ((1 + by[y + dy]) * V2DT2_WFC(0, dy, 0) + \
                    by[y + dy] * ZETAY(0, dy, 0)) +          \
   by[y + dy] * PSIY(0, dy, 0))

// Second derivative update term for y-derivative in PML region (background)
#define UT_TERMY2(dz, dy, dx) \
  ((1 + by[y + dy]) *         \
   ((1 + by[y + dy]) * V2DT2_WFC(0, dy, 0) + by[y + dy] * ZETAY(0, dy, 0)))

// Term for y-derivative auxiliary field update (background)
#define PSIY_TERM(dz, dy, dx) \
  ((1 + by[y + dy]) * V2DT2_WFC(0, dy, 0) + by[y + dy] * ZETAY(0, dy, 0))

// First derivative update term for y-derivative in PML region (scattered)
#define UTSC_TERMY1(dz, dy, dx)                                \
  (dbydy[y + dy] * ((1 + by[y + dy]) * V2DT2_WFCSC(0, dy, 0) + \
                    by[y + dy] * ZETAYSC(0, dy, 0)) +          \
   by[y + dy] * PSIYSC(0, dy, 0))

// Second derivative update term for y-derivative in PML region (scattered)
#define UTSC_TERMY2(dz, dy, dx)                                   \
  ((1 + by[y + dy]) * ((1 + by[y + dy]) * V2DT2_WFCSC(0, dy, 0) + \
                       by[y + dy] * ZETAYSC(0, dy, 0)))

// Term for y-derivative auxiliary field update (scattered)
#define PSIYSC_TERM(dz, dy, dx) \
  ((1 + by[y + dy]) * V2DT2_WFCSC(0, dy, 0) + by[y + dy] * ZETAYSC(0, dy, 0))

#endif

// First derivative update term for x-derivative in PML region (background)
#define UT_TERMX1(dz, dy, dx)                                \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) * V2DT2_WFC(0, 0, dx) + \
                    bx[x + dx] * ZETAX(0, 0, dx)) +          \
   bx[x + dx] * PSIX(0, 0, dx))

// Second derivative update term for x-derivative in PML region (background)
#define UT_TERMX2(dz, dy, dx) \
  ((1 + bx[x + dx]) *         \
   ((1 + bx[x + dx]) * V2DT2_WFC(0, 0, dx) + bx[x + dx] * ZETAX(0, 0, dx)))

// Term for x-derivative auxiliary field update (background)
#define PSIX_TERM(dz, dy, dx) \
  ((1 + bx[x + dx]) * V2DT2_WFC(0, 0, dx) + bx[x + dx] * ZETAX(0, 0, dx))

// First derivative update term for x-derivative in PML region (scattered)
#define UTSC_TERMX1(dz, dy, dx)                                \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) * V2DT2_WFCSC(0, 0, dx) + \
                    bx[x + dx] * ZETAXSC(0, 0, dx)) +          \
   bx[x + dx] * PSIXSC(0, 0, dx))

// Second derivative update term for x-derivative in PML region (scattered)
#define UTSC_TERMX2(dz, dy, dx)                                   \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * V2DT2_WFCSC(0, 0, dx) + \
                       bx[x + dx] * ZETAXSC(0, 0, dx)))

// Term for x-derivative auxiliary field update (scattered)
#define PSIXSC_TERM(dz, dy, dx) \
  ((1 + bx[x + dx]) * V2DT2_WFCSC(0, 0, dx) + bx[x + dx] * ZETAXSC(0, 0, dx))

// --- Device constants for configuration ---
// These are copied to constant memory for fast access by all kernels
namespace {
__constant__ DW_DTYPE dt2;  // Time step squared
#if DW_NDIM >= 3
__constant__ DW_DTYPE rdz;
__constant__ DW_DTYPE rdz2;
__constant__ int64_t nz;
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
#endif
#if DW_NDIM >= 2
__constant__ DW_DTYPE rdy;
__constant__ DW_DTYPE rdy2;
__constant__ int64_t ny;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
#endif
__constant__ DW_DTYPE rdx;
__constant__ DW_DTYPE rdx2;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_shots;
__constant__ int64_t n_sources_per_shot, n_sourcessc_per_shot;
__constant__ int64_t n_receivers_per_shot, n_receiverssc_per_shot;
__constant__ int64_t
    step_ratio;  // Number of steps between gradient contributions
__constant__ int64_t pml_x0, pml_x1;  // PML region bounds
__constant__ bool v_batched,
    scatter_batched;  // Whether v/scatter are shared or per shot

__launch_bounds__(32) __global__
    void add_sources_both(DW_DTYPE *__restrict const wf,
                          DW_DTYPE *__restrict const wfsc,
                          DW_DTYPE const *__restrict const f,
                          DW_DTYPE const *__restrict const fsc,
                          int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    if (0 <= sources_i[k]) {
      wf[shot_idx * shot_numel + sources_i[k]] += f[k];
      wfsc[shot_idx * shot_numel + sources_i[k]] += fsc[k];
    }
  }
}

__launch_bounds__(32) __global__
    void add_adjoint_sources(DW_DTYPE *__restrict const wf,
                             DW_DTYPE const *__restrict const f,
                             int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void add_adjoint_sourcessc(DW_DTYPE *__restrict const wf,
                               DW_DTYPE const *__restrict const f,
                               int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receiverssc_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receiverssc_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void record_receivers(DW_DTYPE *__restrict const r,
                          DW_DTYPE const *__restrict const wf,
                          int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

// Record scattered wavefield amplitudes at receiver locations for all shots.
__launch_bounds__(32) __global__
    void record_receiverssc(DW_DTYPE *__restrict const r,
                            DW_DTYPE const *__restrict const wf,
                            int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receiverssc_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receiverssc_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

// Record adjoint wavefield amplitudes at source locations for all shots (used
// in adjoint computations).
__launch_bounds__(32) __global__
    void record_adjoint_receivers(DW_DTYPE *__restrict const r,
                                  DW_DTYPE const *__restrict const wf,
                                  int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__
    void record_adjoint_receiverssc(DW_DTYPE *__restrict const rsc,
                                    DW_DTYPE const *__restrict const wfsc,
                                    int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sourcessc_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sourcessc_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
      rsc[k] = wfsc[shot_idx * shot_numel + receivers_i[k]];
  }
}

// Combine per-shot gradients into a single gradient array (for v or scatter).
__launch_bounds__(128) __global__
    void combine_grad(DW_DTYPE *__restrict const grad,
                      DW_DTYPE const *__restrict const grad_shot) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  if (x < nx - FD_PAD) {
    int64_t const i = x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
#endif
}

// Main CUDA kernel for forward propagation of both background and scattered
// wavefields. Computes the next time step for both wavefields, including PML
// and auxiliary fields. Arguments:
//   v, scatter: velocity and scattering potential (can be batched per shot)
//   wfc, wfcsc: current background and scattered wavefields
//   wfp, wfpsc: previous (input) and next (output) time step wavefields
//   psiz, psizn, zetaz: PML auxiliary fields (background z-dim)
//   psiy, psiy n, zetay: PML auxiliary fields (background y-dim)
//   psix, psixn, zetax: PML auxiliary fields (background x-dim)
//   psizsc, psiznsc, zetazsc: PML auxiliary fields (scattered z-dim)
//   psiysc, psiynsc, zetaysc: PML auxiliary fields (scattered y-dim)
//   psixsc, psixnsc, zetaxsc: PML auxiliary fields (scattered x-dim)
//   w_store, wsc_store: storage for gradient calculation snapshots
//   az, bz, dbzdz: PML profiles and derivatives for z-dim
//   ay, ax, by, bx, dbydy, dbxdx: PML profiles and derivatives for y-dim and
//   x-dim v_requires_grad, scatter_requires_grad: whether to store values for
//   gradient computation
__launch_bounds__(128) __global__ void forward_kernel(
    DW_DTYPE const *__restrict const v,
    DW_DTYPE const *__restrict const scatter,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const psiz, DW_DTYPE *__restrict const psizn,
    DW_DTYPE *__restrict const zetaz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const psiy, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE *__restrict const zetay,
#endif
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psixn,
    DW_DTYPE *__restrict const zetax, DW_DTYPE const *__restrict const wfcsc,
    DW_DTYPE *__restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const psizsc, DW_DTYPE *__restrict const psiznsc,
    DW_DTYPE *__restrict const zetazsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const psiysc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE *__restrict const zetaysc,
#endif
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psixnsc,
    DW_DTYPE *__restrict const zetaxsc, DW_DTYPE *__restrict const w_store,
    DW_DTYPE *__restrict const wsc_store,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const dbydy,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbxdx, bool const store_w,
    bool const store_wsc) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity and scatter arrays for this shot (batched or shared)
      DW_DTYPE const *__restrict const v_shot =
          v_batched ? v + shot_idx * shot_numel : v;
      DW_DTYPE const *__restrict const scatter_shot =
          scatter_batched ? scatter + shot_idx * shot_numel : scatter;
      DW_DTYPE w_sum = 0, wsc_sum = 0;

#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      if (!pml_z) {
        w_sum += DIFFZ2(WFC);
        wsc_sum += DIFFZ2(WFCSC);
      } else {
        DW_DTYPE dwfcdz = DIFFZ1(WFC);
        DW_DTYPE tmpz =
            ((1 + bz[z]) * DIFFZ2(WFC) + dbzdz[z] * dwfcdz + DIFFZ1(AZ_PSIZ));
        DW_DTYPE dwfcscdz = DIFFZ1(WFCSC);
        DW_DTYPE tmpzsc = ((1 + bz[z]) * DIFFZ2(WFCSC) + dbzdz[z] * dwfcscdz +
                           DIFFZ1(AZ_PSIZSC));
        w_sum += (1 + bz[z]) * tmpz + az[z] * zetaz[i];
        wsc_sum += (1 + bz[z]) * tmpzsc + az[z] * zetazsc[i];
        psizn[i] = bz[z] * dwfcdz + az[z] * psiz[i];
        zetaz[i] = bz[z] * tmpz + az[z] * zetaz[i];
        psiznsc[i] = bz[z] * dwfcscdz + az[z] * psizsc[i];
        zetazsc[i] = bz[z] * tmpzsc + az[z] * zetazsc[i];
      }
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
      if (!pml_y) {
        // Central region: standard finite difference for y-derivative
        w_sum += DIFFY2(WFC);
        wsc_sum += DIFFY2(WFCSC);
      } else {
        // PML region: use auxiliary fields and PML profiles for y-derivative
        DW_DTYPE dwfcdy = DIFFY1(WFC);
        DW_DTYPE tmpy =
            ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
        DW_DTYPE dwfcscdy = DIFFY1(WFCSC);
        DW_DTYPE tmpysc = ((1 + by[y]) * DIFFY2(WFCSC) + dbydy[y] * dwfcscdy +
                           DIFFY1(AY_PSIYSC));
        w_sum += (1 + by[y]) * tmpy + ay[y] * zetay[i];
        wsc_sum += (1 + by[y]) * tmpysc + ay[y] * zetaysc[i];
        psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];
        zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
        psiynsc[i] = by[y] * dwfcscdy + ay[y] * psiysc[i];
        zetaysc[i] = by[y] * tmpysc + ay[y] * zetaysc[i];
      }
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
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
      wfpsc[i] = v_shot[j] * v_shot[j] * dt2 * wsc_sum + 2 * wfcsc[i] -
                 wfpsc[i] + 2 * v_shot[j] * scatter_shot[j] * dt2 * w_sum;
      // Store values for gradient calculation if needed
      if (store_w) {
        w_store[i] = w_sum;
      }
      if (store_wsc) {
        wsc_store[i] = wsc_sum;
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// Main CUDA kernel for backward (adjoint) propagation for both background and
// scattered wavefields. Computes gradients with respect to velocity and scatter
// if requested. Arguments are analogous to forward_kernel, with additional
// gradient outputs.
__launch_bounds__(128) __global__ void backward_kernel(
    DW_DTYPE const *__restrict const v,
    DW_DTYPE const *__restrict const scatter,
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const psiz, DW_DTYPE *__restrict const psizn,
    DW_DTYPE *__restrict const zetaz, DW_DTYPE *__restrict const zetazn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const psiy, DW_DTYPE *__restrict const psiyn,
    DW_DTYPE const *__restrict const zetay, DW_DTYPE *__restrict const zetayn,
#endif
    DW_DTYPE const *__restrict const psix, DW_DTYPE *__restrict const psixn,
    DW_DTYPE const *__restrict const zetax, DW_DTYPE *__restrict const zetaxn,
    DW_DTYPE const *__restrict const wfcsc, DW_DTYPE *__restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const psizsc, DW_DTYPE *__restrict const psiznsc,
    DW_DTYPE *__restrict const zetazsc, DW_DTYPE *__restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const psiysc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE const *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaynsc,
#endif
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psixnsc,
    DW_DTYPE const *__restrict const zetaxsc,
    DW_DTYPE *__restrict const zetaxnsc,
    DW_DTYPE const *__restrict const w_store,
    DW_DTYPE const *__restrict const wsc_store,
    DW_DTYPE *__restrict const grad_v, DW_DTYPE *__restrict const grad_scatter,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const dbydy,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbxdx, bool const v_requires_grad,
    bool const scatter_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity and scatter arrays for this shot (batched or shared)
      DW_DTYPE const *__restrict const v_shot =
          v_batched ? v + shot_idx * shot_numel : v;
      DW_DTYPE const *__restrict const scatter_shot =
          scatter_batched ? scatter + shot_idx * shot_numel : scatter;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      // Update background and scattered wavefields for next time step
      // (adjoint/backward)
      DW_DTYPE w_sum = 0, wsc_sum = 0;
#if DW_NDIM >= 3
      w_sum +=
          pml_z ? -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2) : DIFFZ2(V2DT2_WFC);
      wsc_sum += pml_z ? -DIFFZ1(UTSC_TERMZ1) + DIFFZ2(UTSC_TERMZ2)
                       : DIFFZ2(V2DT2_WFCSC);
#endif
#if DW_NDIM >= 2
      w_sum +=
          pml_y ? -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2) : DIFFY2(V2DT2_WFC);
      wsc_sum += pml_y ? -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2)
                       : DIFFY2(V2DT2_WFCSC);
#endif
      w_sum +=
          pml_x ? -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2) : DIFFX2(V2DT2_WFC);
      wsc_sum += pml_x ? -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2)
                       : DIFFX2(V2DT2_WFCSC);

      wfp[i] = w_sum + 2 * wfc[i] - wfp[i];
      wfpsc[i] = wsc_sum + 2 * wfcsc[i] - wfpsc[i];

      // Update PML auxiliary fields for z/y/x directions (background and
      // scattered)
#if DW_NDIM >= 3
      if (pml_z) {
        psiznsc[i] = -az[z] * DIFFZ1(PSIZSC_TERM) + az[z] * psizsc[i];
        zetaznsc[i] = az[z] * V2DT2(0, 0, 0) * wfcsc[i] + az[z] * zetazsc[i];
        psizn[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz[i];
        zetazn[i] = az[z] * V2DT2(0, 0, 0) * wfc[i] +
                    az[z] * 2 * VDT2(0, 0, 0) * SCATTER(0, 0, 0) * wfcsc[i] +
                    az[z] * zetaz[i];
      }
#endif
#if DW_NDIM >= 2
      if (pml_y) {
        psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];
        zetaynsc[i] = ay[y] * V2DT2(0, 0, 0) * wfcsc[i] + ay[y] * zetaysc[i];
        psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
        zetayn[i] = ay[y] * V2DT2(0, 0, 0) * wfc[i] +
                    ay[y] * 2 * VDT2(0, 0, 0) * SCATTER(0, 0, 0) * wfcsc[i] +
                    ay[y] * zetay[i];
      }
#endif
      if (pml_x) {
        psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];
        zetaxnsc[i] = ax[x] * V2DT2(0, 0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];
        psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
        zetaxn[i] = ax[x] * V2DT2(0, 0, 0) * wfc[i] +
                    ax[x] * 2 * VDT2(0, 0, 0) * SCATTER(0, 0, 0) * wfcsc[i] +
                    ax[x] * zetax[i];
      }
      // Accumulate gradients for velocity and scatter if required
      if (v_requires_grad) {
        grad_v[i] +=
            wfc[i] * 2 * v_shot[j] * dt2 * w_store[i] * (DW_DTYPE)step_ratio +
            wfcsc[i] *
                (2 * dt2 * scatter_shot[j] * w_store[i] +
                 2 * v_shot[j] * dt2 * wsc_store[i]) *
                (DW_DTYPE)step_ratio;
      }
      if (scatter_requires_grad) {
        grad_scatter[i] +=
            wfcsc[i] * 2 * v_shot[j] * dt2 * w_store[i] * (DW_DTYPE)step_ratio;
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// Specialized backward kernel for computing only the scatter gradient.
__launch_bounds__(128) __global__ void backward_kernel_sc(
    DW_DTYPE const *__restrict const v, DW_DTYPE const *__restrict const wfcsc,
    DW_DTYPE *__restrict const wfpsc,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const psizsc, DW_DTYPE *__restrict const psiznsc,
    DW_DTYPE const *__restrict const zetazsc,
    DW_DTYPE *__restrict const zetaznsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const psiysc, DW_DTYPE *__restrict const psiynsc,
    DW_DTYPE const *__restrict const zetaysc,
    DW_DTYPE *__restrict const zetaynsc,
#endif
    DW_DTYPE const *__restrict const psixsc, DW_DTYPE *__restrict const psixnsc,
    DW_DTYPE const *__restrict const zetaxsc,
    DW_DTYPE *__restrict const zetaxnsc,
    DW_DTYPE const *__restrict const w_store,
    DW_DTYPE *__restrict const grad_scatter,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const dbydy,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbxdx, bool const scatter_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD && x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      // Select velocity array for this shot (batched or shared)
      DW_DTYPE const *__restrict const v_shot =
          v_batched ? v + shot_idx * shot_numel : v;
#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      // Update scattered wavefield for next time step (adjoint/backward)
      DW_DTYPE wsc_sum = 0;
#if DW_NDIM >= 3
      wsc_sum += pml_z ? -DIFFZ1(UTSC_TERMZ1) + DIFFZ2(UTSC_TERMZ2)
                       : DIFFZ2(V2DT2_WFCSC);
#endif
#if DW_NDIM >= 2
      wsc_sum += pml_y ? -DIFFY1(UTSC_TERMY1) + DIFFY2(UTSC_TERMY2)
                       : DIFFY2(V2DT2_WFCSC);
#endif
      wsc_sum += pml_x ? -DIFFX1(UTSC_TERMX1) + DIFFX2(UTSC_TERMX2)
                       : DIFFX2(V2DT2_WFCSC);

      wfpsc[i] = wsc_sum + 2 * wfcsc[i] - wfpsc[i];
      // Update PML auxiliary fields for z/y/x directions (scattered)
#if DW_NDIM >= 3
      if (pml_z) {
        psiznsc[i] = -az[z] * DIFFZ1(PSIZSC_TERM) + az[z] * psizsc[i];
        zetaznsc[i] = az[z] * V2DT2(0, 0, 0) * wfcsc[i] + az[z] * zetazsc[i];
      }
#endif
#if DW_NDIM >= 2
      if (pml_y) {
        psiynsc[i] = -ay[y] * DIFFY1(PSIYSC_TERM) + ay[y] * psiysc[i];
        zetaynsc[i] = ay[y] * V2DT2(0, 0, 0) * wfcsc[i] + ay[y] * zetaysc[i];
      }
#endif
      if (pml_x) {
        psixnsc[i] = -ax[x] * DIFFX1(PSIXSC_TERM) + ax[x] * psixsc[i];
        zetaxnsc[i] = ax[x] * V2DT2(0, 0, 0) * wfcsc[i] + ax[x] * zetaxsc[i];
      }
      // Accumulate gradient for scatter if required
      if (scatter_requires_grad) {
        grad_scatter[i] +=
            wfcsc[i] * 2 * v_shot[j] * dt2 * w_store[i] * (DW_DTYPE)step_ratio;
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// Copy configuration parameters from host to device constant memory
int set_config(
#if DW_NDIM >= 3
    DW_DTYPE const rdz_h, DW_DTYPE const rdz2_h, int64_t const nz_h,
    int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_h, DW_DTYPE const rdy2_h, int64_t const ny_h,
    int64_t const pml_y0_h, int64_t const pml_y1_h,
#endif
    DW_DTYPE const rdx_h, DW_DTYPE const rdx2_h, int64_t const nx_h,
    DW_DTYPE const dt2_h, int64_t const n_shots_h,
    int64_t const n_sources_per_shot_h, int64_t const n_sourcessc_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
    int64_t const pml_x0_h, int64_t const pml_x1_h, bool const v_batched_h,
    bool const scatter_batched_h) {
#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif
  gpuErrchk(cudaMemcpyToSymbol(dt2, &dt2_h, sizeof(DW_DTYPE)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdz2, &rdz2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy2, &rdy2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx2, &rdx2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sourcessc_per_shot, &n_sourcessc_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receiverssc_per_shot,
                               &n_receiverssc_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(v_batched, &v_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(scatter_batched, &scatter_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        int FUNC(forward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const scatter,
            DW_DTYPE const *__restrict const f,
            DW_DTYPE const *__restrict const fsc, DW_DTYPE *__restrict wfc,
            DW_DTYPE *__restrict wfp,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psiz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiy,
#endif
            DW_DTYPE *__restrict psix,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psizn,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiyn,
#endif
            DW_DTYPE *__restrict psixn,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const zetaz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const zetay,
#endif
            DW_DTYPE *__restrict const zetax, DW_DTYPE *__restrict wfcsc,
            DW_DTYPE *__restrict wfpsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psizsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiysc,
#endif
            DW_DTYPE *__restrict psixsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psiznsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiynsc,
#endif
            DW_DTYPE *__restrict psixnsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict const zetazsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict const zetaysc,
#endif
            DW_DTYPE *__restrict const zetaxsc,
            DW_DTYPE *__restrict const w_store_1a,
            DW_DTYPE *__restrict const w_store_1b,
            void *__restrict const w_store_2, void *__restrict const w_store_3,
            char const *__restrict const *__restrict const w_filenames_ptr,
            DW_DTYPE *__restrict const wsc_store_1a,
            DW_DTYPE *__restrict const wsc_store_1b,
            void *__restrict const wsc_store_2,
            void *__restrict const wsc_store_3,
            char const *__restrict const *__restrict const wsc_filenames_ptr,
            DW_DTYPE *__restrict const r, DW_DTYPE *__restrict const rsc,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const az,
            DW_DTYPE const *__restrict const bz,
            DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const dbydy,
#endif
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i,
            int64_t const *__restrict const receiverssc_i,
#if DW_NDIM >= 3
            DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy_h,
#endif
            DW_DTYPE const rdx_h,
#if DW_NDIM >= 3
            DW_DTYPE const rdz2_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy2_h,
#endif
            DW_DTYPE const rdx2_h, DW_DTYPE const dt2_h, int64_t const nt,
            int64_t const n_shots_h,
#if DW_NDIM >= 3
            int64_t const nz_h,
#endif
#if DW_NDIM >= 2
            int64_t const ny_h,
#endif
            int64_t const nx_h, int64_t const n_sources_per_shot_h,
            int64_t const n_receivers_per_shot_h,
            int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const v_requires_grad,
            bool const scatter_requires_grad, bool const v_batched_h,
            bool const scatter_batched_h, bool const storage_compression,
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
  // --- CUDA kernel launch configuration ---
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);

  // For source and receiver operations
  dim3 const dimBlock_sources(32, 1, 1);
  unsigned int const gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int const gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int const gridz_sources = 1;
  dim3 const dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 const dimBlock_receivers(32, 1, 1);
  unsigned int const gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int const gridy_receivers =
      ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int const gridz_receivers = 1;
  dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers,
                               gridz_receivers);
  dim3 const dimBlock_receiverssc(32, 1, 1);
  unsigned int const gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int const gridy_receiverssc =
      ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int const gridz_receiverssc = 1;
  dim3 const dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                                 gridz_receiverssc);

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
        rdz_h, rdz2_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, rdy2_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, rdx2_h, nx_h, dt2_h, n_shots_h, n_sources_per_shot_h,
        n_sources_per_shot_h, n_receivers_per_shot_h, n_receiverssc_per_shot_h,
        step_ratio_h, pml_x0_h, pml_x1_h, v_batched_h, scatter_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (v_requires_grad || scatter_requires_grad);

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

  ScopedFile fp_w;
  ScopedFile fp_wsc;
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad) fp_w.open(w_filenames_ptr[0], "ab");
  }
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad || scatter_requires_grad)
      fp_w.open(w_filenames_ptr[0], "ab");
    if (v_requires_grad) fp_wsc.open(wsc_filenames_ptr[0], "ab");
  }

  // --- Time-stepping loop for forward propagation ---
  // Alternates between wfc/wfp and wfp/wfc for memory efficiency
  for (t = start_t; t < start_t + nt; ++t) {
    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_w =
        store_step && (v_requires_grad || scatter_requires_grad);
    bool const store_wsc = store_step && v_requires_grad;
    int64_t const step_idx = t / step_ratio_h;
    DW_DTYPE *__restrict w_store_1_t =
        w_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                          ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                          : 0);
    DW_DTYPE *__restrict wsc_store_1_t =
        wsc_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                            ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                            : 0);
    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;

    if ((store_w || store_wsc) && use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }
    if (store_w && use_double_buffering) {
      if (step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
      }
    }
    if (store_wsc && use_double_buffering) {
      if (step_idx % 2 != 0) {
        wsc_store_1_t = wsc_store_1b;
      }
    }
    void *__restrict const w_store_2_t =
        (uint8_t *)w_store_2 +
        (storage_mode == STORAGE_DEVICE && storage_compression
             ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
             : 0);
    void *__restrict const wsc_store_2_t =
        (uint8_t *)wsc_store_2 +
        (storage_mode == STORAGE_DEVICE && storage_compression
             ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
             : 0);
    void *__restrict const w_store_3_t =
        (uint8_t *)w_store_3 +
        (storage_mode == STORAGE_CPU
             ? (t / step_ratio_h) *
                   (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) *
                   n_shots_h
             : 0);
    void *__restrict const wsc_store_3_t =
        (uint8_t *)wsc_store_3 +
        (storage_mode == STORAGE_CPU
             ? (t / step_ratio_h) *
                   (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) *
                   n_shots_h
             : 0);

    forward_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, scatter, wfc, wfp,
#if DW_NDIM >= 3
        psiz, psizn, zetaz,
#endif
#if DW_NDIM >= 2
        psiy, psiyn, zetay,
#endif
        psix, psixn, zetax, wfcsc, wfpsc,
#if DW_NDIM >= 3
        psizsc, psiznsc, zetazsc,
#endif
#if DW_NDIM >= 2
        psiysc, psiynsc, zetaysc,
#endif
        psixsc, psixnsc, zetaxsc, w_store_1_t, wsc_store_1_t,
#if DW_NDIM >= 3
        az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        ay, by, dbydy,
#endif
        ax, bx, dbxdx, store_w, store_wsc);
    CHECK_KERNEL_ERROR

    if ((store_w || store_wsc) && use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    if (store_w) {
      if (STORAGE_FUNC(save_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }
    if (store_wsc) {
      if (STORAGE_FUNC(save_snapshot_gpu)(
              wsc_store_1_t, wsc_store_2_t, wsc_store_3_t, fp_wsc, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }

    if ((store_w || store_wsc) && use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    // Add sources to both background and scattered wavefields
    if (n_sources_per_shot_h > 0) {
      add_sources_both<<<dimGrid_sources, dimBlock_sources, 0,
                         stream_compute>>>(
          wfp, wfpsc, f + t * n_shots_h * n_sources_per_shot_h,
          fsc + t * n_shots_h * n_sources_per_shot_h, sources_i);
      CHECK_KERNEL_ERROR
    }
    // Record background wavefield at receiver locations
    if (n_receivers_per_shot_h > 0) {
      record_receivers<<<dimGrid_receivers, dimBlock_receivers, 0,
                         stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, wfc, receivers_i);
      CHECK_KERNEL_ERROR
    }
    // Record scattered wavefield at receiver locations
    if (n_receiverssc_per_shot_h > 0) {
      record_receiverssc<<<dimGrid_receiverssc, dimBlock_receiverssc, 0,
                           stream_compute>>>(
          rsc + t * n_shots_h * n_receiverssc_per_shot_h, wfcsc, receiverssc_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psiz, psizn);
    std::swap(psizsc, psiznsc);
#endif
#if DW_NDIM >= 2
    std::swap(psiy, psiyn);
    std::swap(psiysc, psiynsc);
#endif
    std::swap(wfc, wfp);
    std::swap(psix, psixn);
    std::swap(wfcsc, wfpsc);
    std::swap(psixsc, psixnsc);
  }

  if (use_double_buffering)
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
  return 0;
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
        int FUNC(backward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const scatter,
            DW_DTYPE const *__restrict const grad_r,
            DW_DTYPE const *__restrict const grad_rsc, DW_DTYPE *__restrict wfc,
            DW_DTYPE *__restrict wfp,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psiz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiy,
#endif
            DW_DTYPE *__restrict psix,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psizn,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiyn,
#endif
            DW_DTYPE *__restrict psixn,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict zetaz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict zetay,
#endif
            DW_DTYPE *__restrict zetax,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict zetazn,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict zetayn,
#endif
            DW_DTYPE *__restrict zetaxn, DW_DTYPE *__restrict wfcsc,
            DW_DTYPE *__restrict wfpsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psizsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiysc,
#endif
            DW_DTYPE *__restrict psixsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psiznsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiynsc,
#endif
            DW_DTYPE *__restrict psixnsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict zetazsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict zetaysc,
#endif
            DW_DTYPE *__restrict zetaxsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict zetaznsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict zetaynsc,
#endif
            DW_DTYPE *__restrict zetaxnsc,
            DW_DTYPE *__restrict const w_store_1a,
            DW_DTYPE *__restrict const w_store_1b,
            void *__restrict const w_store_2, void *__restrict const w_store_3,
            char const *__restrict const *__restrict const w_filenames_ptr,
            DW_DTYPE *__restrict const wsc_store_1a,
            DW_DTYPE *__restrict const wsc_store_1b,
            void *__restrict const wsc_store_2,
            void *__restrict const wsc_store_3,
            char const *__restrict const *__restrict const wsc_filenames_ptr,
            DW_DTYPE *__restrict const grad_f,
            DW_DTYPE *__restrict const grad_fsc,
            DW_DTYPE *__restrict const grad_v,
            DW_DTYPE *__restrict const grad_scatter,
            DW_DTYPE *__restrict const grad_v_shot,
            DW_DTYPE *__restrict const grad_scatter_shot,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const az,
            DW_DTYPE const *__restrict const bz,
            DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const dbydy,
#endif
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i,
            int64_t const *__restrict const receiverssc_i,
#if DW_NDIM >= 3
            DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy_h,
#endif
            DW_DTYPE const rdx_h,
#if DW_NDIM >= 3
            DW_DTYPE const rdz2_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy2_h,
#endif
            DW_DTYPE const rdx2_h, DW_DTYPE const dt2_h, int64_t const nt,
            int64_t const n_shots_h,
#if DW_NDIM >= 3
            int64_t const nz_h,
#endif
#if DW_NDIM >= 2
            int64_t const ny_h,
#endif
            int64_t const nx_h, int64_t const n_sources_per_shot_h,
            int64_t const n_sourcessc_per_shot_h,
            int64_t const n_receivers_per_shot_h,
            int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const v_requires_grad,
            bool const scatter_requires_grad, bool const v_batched_h,
            bool const scatter_batched_h, bool const storage_compression,
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
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);

  // For source and receiver operations
  dim3 const dimBlock_sources(32, 1, 1);
  unsigned int const gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int const gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int const gridz_sources = 1;
  dim3 const dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);
  dim3 const dimBlock_sourcessc(32, 1, 1);
  unsigned int const gridx_sourcessc =
      ceil_div(n_sourcessc_per_shot_h, dimBlock_sourcessc.x);
  unsigned int const gridy_sourcessc =
      ceil_div(n_shots_h, dimBlock_sourcessc.y);
  unsigned int const gridz_sourcessc = 1;
  dim3 const dimGrid_sourcessc(gridx_sourcessc, gridy_sourcessc,
                               gridz_sourcessc);
  dim3 const dimBlock_receivers(32, 1, 1);
  unsigned int const gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int const gridy_receivers =
      ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int const gridz_receivers = 1;
  dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers,
                               gridz_receivers);
  dim3 const dimBlock_receiverssc(32, 1, 1);
  unsigned int const gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int const gridy_receiverssc =
      ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int const gridz_receiverssc = 1;
  dim3 const dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                                 gridz_receiverssc);

#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine =
      ceil_div(nz_h - 2 * FD_PAD, dimBlock_combine.z);
#elif DW_NDIM == 2
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock_combine(128, 1, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine = 1;
  unsigned int const gridz_combine = 1;
#endif
  dim3 const dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

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
        rdz_h, rdz2_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, rdy2_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, rdx2_h, nx_h, dt2_h, n_shots_h, n_sources_per_shot_h,
        n_sourcessc_per_shot_h, n_receivers_per_shot_h,
        n_receiverssc_per_shot_h, step_ratio_h, pml_x0_h, pml_x1_h, v_batched_h,
        scatter_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (v_requires_grad || scatter_requires_grad);

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

  ScopedFile fp_w;
  ScopedFile fp_wsc;
  if (storage_mode == STORAGE_DISK) {
    if (v_requires_grad || scatter_requires_grad)
      fp_w.open(w_filenames_ptr[0], "rb");
    if (v_requires_grad) fp_wsc.open(wsc_filenames_ptr[0], "rb");
  }

  // --- Time-reversed loop for adjoint propagation ---
  // Alternates between wfc/wfp and wfp/wfc for memory efficiency
  for (t = start_t - 1; t >= start_t - nt; --t) {
    bool const load_step = (t % step_ratio_h) == 0;
    bool const load_w = load_step && (v_requires_grad || scatter_requires_grad);
    bool const load_wsc = load_step && v_requires_grad;
    int const step_idx = t / step_ratio_h;
    DW_DTYPE *__restrict w_store_1_t =
        w_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                          ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                          : 0);
    DW_DTYPE *__restrict wsc_store_1_t =
        wsc_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                            ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                            : 0);
    void *__restrict const w_store_2_t =
        (uint8_t *)w_store_2 +
        (storage_mode == STORAGE_DEVICE && storage_compression
             ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
             : 0);
    void *__restrict const wsc_store_2_t =
        (uint8_t *)wsc_store_2 +
        (storage_mode == STORAGE_DEVICE && storage_compression
             ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
             : 0);
    void *__restrict const w_store_3_t =
        (uint8_t *)w_store_3 +
        (storage_mode == STORAGE_CPU
             ? (t / step_ratio_h) *
                   (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) *
                   n_shots_h
             : 0);
    void *__restrict const wsc_store_3_t =
        (uint8_t *)wsc_store_3 +
        (storage_mode == STORAGE_CPU
             ? (t / step_ratio_h) *
                   (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) *
                   n_shots_h
             : 0);
    cudaEvent_t event_storage_done = event_storage_done_a;
    cudaEvent_t event_compute_done = event_compute_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_storage_done = event_storage_done_b;
        event_compute_done = event_compute_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    if (load_w) {
      if (use_double_buffering && step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
      }
      if (STORAGE_FUNC(load_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }

    if (load_wsc) {
      if (use_double_buffering && step_idx % 2 != 0) {
        wsc_store_1_t = wsc_store_1b;
      }
      if (STORAGE_FUNC(load_snapshot_gpu)(
              wsc_store_1_t, wsc_store_2_t, wsc_store_3_t, fp_wsc, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }
    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    // Record source gradients for background and scattered fields
    if (n_sources_per_shot_h > 0) {
      record_adjoint_receivers<<<dimGrid_sources, dimBlock_sources, 0,
                                 stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, wfc, sources_i);
      CHECK_KERNEL_ERROR
    }
    if (n_sourcessc_per_shot_h > 0) {
      record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc, 0,
                                   stream_compute>>>(
          grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfcsc, sources_i);
      CHECK_KERNEL_ERROR
    }
    // Launch backward kernel for this time step
    backward_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, scatter, wfc, wfp,
#if DW_NDIM >= 3
        psiz, psizn, zetaz, zetazn,
#endif
#if DW_NDIM >= 2
        psiy, psiyn, zetay, zetayn,
#endif
        psix, psixn, zetax, zetaxn, wfcsc, wfpsc,
#if DW_NDIM >= 3
        psizsc, psiznsc, zetazsc, zetaznsc,
#endif
#if DW_NDIM >= 2
        psiysc, psiynsc, zetaysc, zetaynsc,
#endif
        psixsc, psixnsc, zetaxsc, zetaxnsc, w_store_1_t, wsc_store_1_t,
        grad_v_shot, grad_scatter_shot,
#if DW_NDIM >= 3
        az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        ay, by, dbydy,
#endif
        ax, bx, dbxdx, load_w && v_requires_grad,
        load_w && scatter_requires_grad);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Add receiver gradients as adjoint sources
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources<<<dimGrid_receivers, dimBlock_receivers, 0,
                            stream_compute>>>(
          wfp, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receiverssc_per_shot_h > 0) {
      add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc, 0,
                              stream_compute>>>(
          wfpsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
          receiverssc_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psiz, psizn);
    std::swap(zetaz, zetazn);
    std::swap(psizsc, psiznsc);
    std::swap(zetazsc, zetaznsc);
#endif
#if DW_NDIM >= 2
    std::swap(psiy, psiyn);
    std::swap(zetay, zetayn);
    std::swap(psiysc, psiynsc);
    std::swap(zetaysc, zetaynsc);
#endif
    std::swap(wfc, wfp);
    std::swap(psix, psixn);
    std::swap(zetax, zetaxn);
    std::swap(wfcsc, wfpsc);
    std::swap(psixsc, psixnsc);
    std::swap(zetaxsc, zetaxnsc);
  }
  // Reduce per-shot gradients to a single array if needed
  if (v_requires_grad && !v_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_v, grad_v_shot);
    CHECK_KERNEL_ERROR
  }
  if (scatter_requires_grad && !scatter_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_scatter, grad_scatter_shot);
    CHECK_KERNEL_ERROR
  }

  return 0;
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
        int FUNC(backward_sc)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const grad_rsc,
            DW_DTYPE *__restrict wfcsc, DW_DTYPE *__restrict wfpsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psizsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiysc,
#endif
            DW_DTYPE *__restrict psixsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict psiznsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict psiynsc,
#endif
            DW_DTYPE *__restrict psixnsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict zetazsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict zetaysc,
#endif
            DW_DTYPE *__restrict zetaxsc,
#if DW_NDIM >= 3
            DW_DTYPE *__restrict zetaznsc,
#endif
#if DW_NDIM >= 2
            DW_DTYPE *__restrict zetaynsc,
#endif
            DW_DTYPE *__restrict zetaxnsc,
            DW_DTYPE *__restrict const w_store_1a,
            DW_DTYPE *__restrict const w_store_1b,
            void *__restrict const w_store_2, void *__restrict const w_store_3,
            char const *__restrict const *__restrict const w_filenames_ptr,
            DW_DTYPE *__restrict const grad_fsc,
            DW_DTYPE *__restrict const grad_scatter,
            DW_DTYPE *__restrict const grad_scatter_shot,
#if DW_NDIM >= 3
            DW_DTYPE const *__restrict const az,
            DW_DTYPE const *__restrict const bz,
            DW_DTYPE const *__restrict const dbzdz,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const *__restrict const ay,
            DW_DTYPE const *__restrict const by,
            DW_DTYPE const *__restrict const dbydy,
#endif
            DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receiverssc_i,
#if DW_NDIM >= 3
            DW_DTYPE const rdz_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy_h,
#endif
            DW_DTYPE const rdx_h,
#if DW_NDIM >= 3
            DW_DTYPE const rdz2_h,
#endif
#if DW_NDIM >= 2
            DW_DTYPE const rdy2_h,
#endif
            DW_DTYPE const rdx2_h, DW_DTYPE const dt2_h, int64_t const nt,
            int64_t const n_shots_h,
#if DW_NDIM >= 3
            int64_t const nz_h,
#endif
#if DW_NDIM >= 2
            int64_t const ny_h,
#endif
            int64_t const nx_h, int64_t const n_sourcessc_per_shot_h,
            int64_t const n_receiverssc_per_shot_h, int64_t const step_ratio_h,
            int64_t const storage_mode, size_t const shot_bytes_uncomp,
            size_t const shot_bytes_comp, bool const scatter_requires_grad,
            bool const v_batched_h, bool const scatter_batched_h,
            bool const storage_compression, int64_t const start_t,
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
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);

  dim3 const dimBlock_sourcessc(32, 1, 1);
  unsigned int const gridx_sourcessc =
      ceil_div(n_sourcessc_per_shot_h, dimBlock_sourcessc.x);
  unsigned int const gridy_sourcessc =
      ceil_div(n_shots_h, dimBlock_sourcessc.y);
  unsigned int const gridz_sourcessc = 1;
  dim3 const dimGrid_sourcessc(gridx_sourcessc, gridy_sourcessc,
                               gridz_sourcessc);
  dim3 const dimBlock_receiverssc(32, 1, 1);
  unsigned int const gridx_receiverssc =
      ceil_div(n_receiverssc_per_shot_h, dimBlock_receiverssc.x);
  unsigned int const gridy_receiverssc =
      ceil_div(n_shots_h, dimBlock_receiverssc.y);
  unsigned int const gridz_receiverssc = 1;
  dim3 const dimGrid_receiverssc(gridx_receiverssc, gridy_receiverssc,
                                 gridz_receiverssc);

#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine =
      ceil_div(nz_h - 2 * FD_PAD, dimBlock_combine.z);
#elif DW_NDIM == 2
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine =
      ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#else /* DW_NDIM == 1 */
  dim3 const dimBlock_combine(128, 1, 1);
  unsigned int const gridx_combine =
      ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int const gridy_combine = 1;
  unsigned int const gridz_combine = 1;
#endif
  dim3 const dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

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
        rdz_h, rdz2_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, rdy2_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, rdx2_h, nx_h, dt2_h, n_shots_h, n_sourcessc_per_shot_h,
        n_sourcessc_per_shot_h, n_receiverssc_per_shot_h,
        n_receiverssc_per_shot_h, step_ratio_h, pml_x0_h, pml_x1_h, v_batched_h,
        scatter_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      scatter_requires_grad;

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

  ScopedFile fp_w;
  if (storage_mode == STORAGE_DISK) {
    if (scatter_requires_grad) fp_w.open(w_filenames_ptr[0], "rb");
  }

  // --- Time-reversed loop for adjoint propagation (scattered field only)
  // ---
  for (t = start_t - 1; t >= start_t - nt; --t) {
    bool const load_step = ((t % step_ratio_h) == 0);
    bool const load_w = load_step && scatter_requires_grad;
    int const step_idx = t / step_ratio_h;
    DW_DTYPE *__restrict w_store_1_t =
        w_store_1a + (storage_mode == STORAGE_DEVICE && !storage_compression
                          ? (t / step_ratio_h) * shot_numel_h * n_shots_h
                          : 0);
    void *__restrict const w_store_2_t =
        (uint8_t *)w_store_2 +
        (storage_mode == STORAGE_DEVICE && storage_compression
             ? (t / step_ratio_h) * shot_bytes_comp * n_shots_h
             : 0);
    void *__restrict const w_store_3_t =
        (uint8_t *)w_store_3 +
        (storage_mode == STORAGE_CPU
             ? (t / step_ratio_h) *
                   (storage_compression ? shot_bytes_comp : shot_bytes_uncomp) *
                   n_shots_h
             : 0);

    cudaEvent_t event_storage_done = event_storage_done_a;
    cudaEvent_t event_compute_done = event_compute_done_a;

    if (load_w && use_double_buffering) {
      if (step_idx % 2 != 0) {
        w_store_1_t = w_store_1b;
        event_storage_done = event_storage_done_b;
        event_compute_done = event_compute_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    // Record source gradients for scattered field
    if (n_sourcessc_per_shot_h > 0) {
      record_adjoint_receiverssc<<<dimGrid_sourcessc, dimBlock_sourcessc, 0,
                                   stream_compute>>>(
          grad_fsc + t * n_shots_h * n_sourcessc_per_shot_h, wfcsc, sources_i);
      CHECK_KERNEL_ERROR
    }

    if (load_w) {
      if (STORAGE_FUNC(load_snapshot_gpu)(
              w_store_1_t, w_store_2_t, w_store_3_t, fp_w, storage_mode,
              storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,
              n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
      if (use_double_buffering) {
        gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
        gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
      }
    }

    // Launch backward kernel for this time step (scattered field only)
    backward_kernel_sc<<<dimGrid, dimBlock, 0, stream_compute>>>(
        v, wfcsc, wfpsc,
#if DW_NDIM >= 3
        psizsc, psiznsc, zetazsc, zetaznsc,
#endif
#if DW_NDIM >= 2
        psiysc, psiynsc, zetaysc, zetaynsc,
#endif
        psixsc, psixnsc, zetaxsc, zetaxnsc, w_store_1_t, grad_scatter_shot,
#if DW_NDIM >= 3
        az, bz, dbzdz,
#endif
#if DW_NDIM >= 2
        ay, by, dbydy,
#endif
        ax, bx, dbxdx, load_w);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Add receiver gradients as adjoint sources
    if (n_receiverssc_per_shot_h > 0) {
      add_adjoint_sourcessc<<<dimGrid_receiverssc, dimBlock_receiverssc, 0,
                              stream_compute>>>(
          wfpsc, grad_rsc + t * n_shots_h * n_receiverssc_per_shot_h,
          receiverssc_i);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    std::swap(psizsc, psiznsc);
    std::swap(zetazsc, zetaznsc);
#endif
#if DW_NDIM >= 2
    std::swap(psiysc, psiynsc);
    std::swap(zetaysc, zetaynsc);
#endif
    std::swap(wfcsc, wfpsc);
    std::swap(psixsc, psixnsc);
    std::swap(zetaxsc, zetaxnsc);
  }
  // Reduce per-shot gradients to a single array if needed
  if (scatter_requires_grad && !scatter_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_scatter, grad_scatter_shot);
    CHECK_KERNEL_ERROR
  }

  return 0;
}

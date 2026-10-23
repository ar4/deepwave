/*
 * Pseudo-acoustic VTI (vertically transverse isotropic, V_S = 0) wave
 * equation propagator (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the pseudo-acoustic VTI wave
 * equation propagator. It is templated from the elastic VTI CUDA propagator
 * (elastic_vti.cu) by REDUCTION to the pseudo-acoustic field set, exactly as
 * acoustic_vti.c reduces elastic_vti.c -- see that file's header for the full
 * physics derivation (also reproduced in docs/acoustic_vti_derivation.md,
 * Sec. 1-5) and the CPU implementation this file mirrors kernel-for-kernel.
 * The options are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 2-3
 *    (there is no 1-D form).
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 *
 * As in acoustic_vti.c, this milestone (task 7.H, H1) implements forward()
 * only; backward() (the hand-derived adjoint) lands in H2.
 *
 * CONSTITUTIVE RELATION (see acoustic_vti.c for the full derivation). With
 * the symmetry axis as the slowest dimension (z in 3-D, y in 2-D) and d_i
 * the (PML-corrected) dv_i/dx_i:
 *
 *   3-D: sigma_v += dt (c33 d_z + c13 (d_x + d_y))
 *        sigma_h += dt (c11 (d_x + d_y) + c13 d_z)
 *   2-D (axis = y): sigma_v += dt (c33 d_y + c13 d_x)
 *                   sigma_h += dt (c11 d_x + c13 d_y)
 *
 * Each velocity component is fed by exactly ONE stress derivative (vz/vy
 * axis by sigma_v, transverse vy/vx by sigma_h) -- no shear cross terms
 * survive the V_S = 0 reduction.
 */

#include <stdio.h>

#include <cstdint>

#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  acoustic_vti_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
/* Multiaxial-PML (M-PML) build variant -- see the matching comment in
 * acoustic_vti.c for the full explanation. Must mirror acoustic_vti.c
 * exactly, since the exported symbol names (and thus the ctypes ABI) must
 * agree between the CPU and CUDA builds. */
#ifdef DW_MPML
#define ACOUSTIC_VTI_VARIANT_I(name) name##_mpml
#define ACOUSTIC_VTI_VARIANT(name) ACOUSTIC_VTI_VARIANT_I(name)
#else
#define ACOUSTIC_VTI_VARIANT(name) name
#endif
#define FUNC(name) \
  CAT(ACOUSTIC_VTI_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx_h
#endif

#define MAX(a, b) (a > b ? a : b)

/* PML-region test wrapper -- kept in lockstep with elastic_vti.cu. Driver-
 * side scaffolding (depends on the raw z/y/x position against the
 * __constant__ pml_*0/pml_*1 bounds below), so it is NOT part of
 * acoustic_vti_kernels.h -- see that header's top-of-file note. */
#ifdef DW_MPML
#if DW_NDIM == 3
#define DW_PML_TEST(cond)                                             \
  ((false && (cond)) || z < pml_z0 || z >= MAX(pml_z0, pml_z1 - 1) || \
   y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1) || x < pml_x0 ||        \
   x >= MAX(pml_x0, pml_x1 - 1))
#elif DW_NDIM == 2
#define DW_PML_TEST(cond)                                             \
  ((false && (cond)) || y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1) || \
   x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1))
#else
#define DW_PML_TEST(cond) \
  ((false && (cond)) || x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1))
#endif
#else
#define DW_PML_TEST(cond) (cond)
#endif

/* ND_INDEX, the M-PML-or-1-D PML profile accessors P_AZ/P_BZ/.../P_AXH/
 * P_BXH, the field-access macros VZ/VY/VX/SIGMA_V/SIGMA_H, the model-access
 * macros C11/C13/C33/BUOYANCY_Z/Y/X, the M_VZZ/M_VYY/M_VXX/M_SIGMA_V_Z/
 * M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X PML-memory families, the adjoint-term
 * macros VZ_Z/VY_Y/VX_X/SIGMA_V_Z/SIGMA_H_Y/SIGMA_V_Y/SIGMA_H_X (+_PML), and
 * the FUNC(forward_vz_cell)/.../FUNC(backward_sigma_cell) per-grid-point
 * stencil update functions live in acoustic_vti_kernels.h, shared verbatim
 * with acoustic_vti.c (see that header for the full explanation). On this
 * (CUDA) driver the model arrays' `j` parameter is the shot-local index
 * (distinct from the global `i` -- see the header's addressing-convention
 * note); this file's old separate ND_INDEX_J macro is subsumed by the
 * header's single ND_INDEX macro, which is invoked with either `i` or `j` as
 * its first (textual) argument at each call site below. */
#include "acoustic_vti_kernels.h"

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
__constant__ int64_t n_sources_sigma_h_per_shot;
__constant__ int64_t n_sources_sigma_v_per_shot;
#if DW_NDIM >= 3
__constant__ int64_t n_receivers_z_per_shot;
#endif
#if DW_NDIM >= 2
__constant__ int64_t n_receivers_y_per_shot;
#endif
__constant__ int64_t n_receivers_x_per_shot;
__constant__ int64_t n_receivers_sigma_h_per_shot;
__constant__ int64_t n_receivers_sigma_v_per_shot;
__constant__ int64_t step_ratio;
#if DW_NDIM >= 3
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
#endif
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool c11_batched;
__constant__ bool c13_batched;
__constant__ bool c33_batched;
__constant__ bool buoyancy_batched;

/* First-class reflecting/periodic boundary faces on the staggered acoustic_vti
 * grid (see docs/boundaries.rst and acoustic_vti.c's matching comment, which
 * these device functions mirror exactly). fill_p_bc_line_dev/
 * fill_v_bc_line_dev/fill_bc_kernel live in line_bc_dev.h, shared verbatim
 * across every propagator's .cu file on this staggered layout -- see that
 * header. sigma_v is guard-filled only on the axis face and sigma_h only on
 * the transverse faces (bc_axis/bc_transverse below, built once per bc[]) --
 * see acoustic_vti.c's long comment above its own FILL_SIGMA_V_BC/
 * FILL_SIGMA_H_BC macros for the full citation; the velocity fill is
 * unchanged from acoustic (each v[d] imaged only along its own axis d). */
#include "line_bc_dev.h"

/* Pressure-like (sigma_v or sigma_h) guard fill on every reflecting axis. */
static void launch_fill_p_bc(DW_DTYPE *__restrict const p, int64_t const bc[6],
                             int64_t const total_numel, int64_t const nz_,
                             int64_t const ny_, int64_t const nx_,
                             cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    fill_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nx_, 1, bc[4], bc[5], 0);
  }
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    fill_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, ny_, nx_, bc[2], bc[3], 0);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    fill_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nz_, ny_ * nx_, bc[0], bc[1], 0);
  }
#endif
}

/* Boundary-normal velocity guard fill: v[d] imaged along axis d only. */
static void launch_fill_v_bc(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
    DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
    int64_t const bc[6], int64_t const total_numel, int64_t const nz_,
    int64_t const ny_, int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    fill_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vx, nl, nx_, 1, bc[4], bc[5], 1);
  }
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    fill_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vy, nl, ny_, nx_, bc[2], bc[3], 1);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    fill_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
  }
#endif
}

/* --- Adjoint (transpose) of the guard-ring image fills above -------------
 *
 * See acoustic_vti.c's identical module comment above its own FUNC(fill_
 * p_bc_line_adjoint)/FUNC(fill_v_bc_line_adjoint) for the full derivation
 * (fill_p_bc_line_adjoint_dev is a straight port of elastic_vti.c's/
 * elastic_vti.cu's sigma_AA transpose; fill_v_bc_line_adjoint_dev is
 * re-derived from fill_v_bc_line_dev's own free-branch mirror rule, NOT
 * copied from elastic's off-row velocity fill). One thread per line
 * (fill_bc_kernel_adjoint), exactly like fill_bc_kernel above; launch_fill_
 * p_bc_adjoint/launch_fill_v_bc_adjoint launch the per-face kernels in the
 * REVERSE of launch_fill_p_bc's/launch_fill_v_bc's own face order, all on
 * one stream (stream_compute) -- CUDA guarantees kernels launched on one
 * stream execute in launch order, so this reproduces the same corner
 * behaviour as the CPU driver's reversed serial loop without extra
 * synchronisation. */
/* fill_p_bc_line_adjoint_dev lives in elastic_line_bc_dev.h (see that
 * header). */
#include "elastic_line_bc_dev.h"

/* Transpose of fill_v_bc_line_dev -- re-derived, not ported (see
 * acoustic_vti.c's identical module comment for the free-branch zero-gate
 * difference from fill_p_bc_line_adjoint_dev). */
__device__ inline void fill_v_bc_line_adjoint_dev(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j;
      } else if (bc_hi == 1) {
        src = P + m - 3 - j;
      } else {
        src = P + m - 2 - j;
        sign = -1;
      }
      v[base + src * stride] += sign * v[base + xh * stride];
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        v[base + (m - 1 + g) * stride] += v[base + g * stride];
      } else if (bc_lo == 1) {
        v[base + (2 * P - 1 - g) * stride] += v[base + g * stride];
      } else if (g != P - 1) {
        v[base + (2 * P - 2 - g) * stride] += -v[base + g * stride];
      }
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      v[base + (nax - P + 1 + j) * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) v[base + g * stride] = 0;
  }
  if (bc_hi == 3) {
    v[base + P * stride] += v[base + (nax - P) * stride];
  }
  if (bc_hi == 2 || bc_hi == 3) v[base + (nax - P) * stride] = 0;
}

/* One thread per line, matching fill_bc_kernel exactly. ``kind`` 0 =
 * pressure (sigma_v/sigma_h), 1 = velocity. */
__global__ void fill_bc_kernel_adjoint(DW_DTYPE *__restrict const wf,
                                       int64_t const n_lines, int64_t const nax,
                                       int64_t const stride,
                                       int64_t const bc_lo, int64_t const bc_hi,
                                       int const kind) {
  int64_t const L = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  if (kind == 0)
    fill_p_bc_line_adjoint_dev(wf, base, nax, stride, bc_lo, bc_hi);
  else
    fill_v_bc_line_adjoint_dev(wf, base, nax, stride, bc_lo, bc_hi);
}

/* Transpose of launch_fill_p_bc: faces launched in the reverse of that
 * function's own face order (z, then y, then x for 3-D; y, then x for
 * 2-D). */
static void launch_fill_p_bc_adjoint(DW_DTYPE *__restrict const p,
                                     int64_t const bc[6],
                                     int64_t const total_numel,
                                     int64_t const nz_, int64_t const ny_,
                                     int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    fill_bc_kernel_adjoint<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nz_, ny_ * nx_, bc[0], bc[1], 0);
  }
#endif
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    fill_bc_kernel_adjoint<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, ny_, nx_, bc[2], bc[3], 0);
  }
#endif
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    fill_bc_kernel_adjoint<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nx_, 1, bc[4], bc[5], 0);
  }
}

/* Transpose of launch_fill_v_bc: faces launched in the reverse of that
 * function's own face order (immaterial here -- each velocity component is
 * imaged on exactly one axis -- but kept for uniformity). */
static void launch_fill_v_bc_adjoint(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
    DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
    int64_t const bc[6], int64_t const total_numel, int64_t const nz_,
    int64_t const ny_, int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    fill_bc_kernel_adjoint<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
  }
#endif
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    fill_bc_kernel_adjoint<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vy, nl, ny_, nx_, bc[2], bc[3], 1);
  }
#endif
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    fill_bc_kernel_adjoint<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vx, nl, nx_, 1, bc[4], bc[5], 1);
  }
}

#if DW_NDIM == 3
#define MAKE_BC_AXIS_TRANSVERSE()                        \
  int64_t const bc_axis[6] = {bc[0], bc[1], 0, 0, 0, 0}; \
  int64_t const bc_transverse[6] = {0, 0, bc[2], bc[3], bc[4], bc[5]}
#define LAUNCH_FILL_SIGMA_V_BC()                                              \
  launch_fill_p_bc(sigma_v, bc_axis, n_shots_h *shot_numel, nz_h, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_SIGMA_H_BC()                                              \
  launch_fill_p_bc(sigma_h, bc_transverse, n_shots_h *shot_numel, nz_h, ny_h, \
                   nx_h, stream_compute)
#define LAUNCH_FILL_V_BC()                                                  \
  launch_fill_v_bc(vz, vy, vx, bc, n_shots_h *shot_numel, nz_h, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_SIGMA_V_BC_ADJOINT()                                  \
  launch_fill_p_bc_adjoint(sigma_v, bc_axis, n_shots_h *shot_numel, nz_h, \
                           ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_SIGMA_H_BC_ADJOINT()                                  \
  launch_fill_p_bc_adjoint(sigma_h, bc_transverse, n_shots_h *shot_numel, \
                           nz_h, ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_ADJOINT()                                            \
  launch_fill_v_bc_adjoint(vz, vy, vx, bc, n_shots_h *shot_numel, nz_h, ny_h, \
                           nx_h, stream_compute)
#elif DW_NDIM == 2
#define MAKE_BC_AXIS_TRANSVERSE()                        \
  int64_t const bc_axis[6] = {0, 0, bc[2], bc[3], 0, 0}; \
  int64_t const bc_transverse[6] = {0, 0, 0, 0, bc[4], bc[5]}
#define LAUNCH_FILL_SIGMA_V_BC()                                           \
  launch_fill_p_bc(sigma_v, bc_axis, n_shots_h *shot_numel, 1, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_SIGMA_H_BC()                                           \
  launch_fill_p_bc(sigma_h, bc_transverse, n_shots_h *shot_numel, 1, ny_h, \
                   nx_h, stream_compute)
#define LAUNCH_FILL_V_BC()                                           \
  launch_fill_v_bc(vy, vx, bc, n_shots_h *shot_numel, 1, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_SIGMA_V_BC_ADJOINT()                                     \
  launch_fill_p_bc_adjoint(sigma_v, bc_axis, n_shots_h *shot_numel, 1, ny_h, \
                           nx_h, stream_compute)
#define LAUNCH_FILL_SIGMA_H_BC_ADJOINT()                                     \
  launch_fill_p_bc_adjoint(sigma_h, bc_transverse, n_shots_h *shot_numel, 1, \
                           ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_ADJOINT()                                           \
  launch_fill_v_bc_adjoint(vy, vx, bc, n_shots_h *shot_numel, 1, ny_h, nx_h, \
                           stream_compute)
#endif

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

__launch_bounds__(32) __global__
    void add_sources_sigma_h(DW_DTYPE *__restrict const sigma_h,
                             DW_DTYPE const *__restrict const f,
                             int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_sigma_h_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_sigma_h_per_shot + source_idx;
    if (0 <= sources_i[k])
      sigma_h[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void add_sources_sigma_v(DW_DTYPE *__restrict const sigma_v,
                             DW_DTYPE const *__restrict const f,
                             int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_sigma_v_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_sigma_v_per_shot + source_idx;
    if (0 <= sources_i[k])
      sigma_v[shot_idx * shot_numel + sources_i[k]] += f[k];
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
    void record_receivers_sigma_h(DW_DTYPE *__restrict const r,
                                  DW_DTYPE const *__restrict const sigma_h,
                                  int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_sigma_h_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_sigma_h_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
      r[k] = sigma_h[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__
    void record_receivers_sigma_v(DW_DTYPE *__restrict const r,
                                  DW_DTYPE const *__restrict const sigma_v,
                                  int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_sigma_v_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_sigma_v_per_shot + receiver_idx;
    if (0 <= receivers_i[k])
      r[k] = sigma_v[shot_idx * shot_numel + receivers_i[k]];
  }
}

// vz/vy/vx update: fed by exactly ONE stress derivative each (no shear
// cross terms survive the V_S = 0 reduction) -- see acoustic_vti.c.
__launch_bounds__(128) __global__ void forward_kernel_v(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
    DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
    DW_DTYPE const *__restrict const sigma_v,
    DW_DTYPE const *__restrict const sigma_h,
#if DW_NDIM == 3
    DW_DTYPE *__restrict const m_sigma_v_z,
    DW_DTYPE *__restrict const m_sigma_h_y,
#else
    DW_DTYPE *__restrict const m_sigma_v_y,
#endif
    DW_DTYPE *__restrict const m_sigma_h_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdbuoyancy,
#endif
    DW_DTYPE *__restrict const dvydbuoyancy,
    DW_DTYPE *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif

#if DW_NDIM >= 3
      int64_t const pml_z0h = pml_z0;
      int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
#endif
      int64_t const pml_y0h = pml_y0;
      int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
      int64_t const pml_x0h = pml_x0;
      int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      // vz -- fed only by d(sigma_v)/dz (the axis derivative).
      if (z < nz - FD_PAD) {
        DW_DTYPE const buoyancy_z_shot =
            buoyancy_batched ? buoyancy_z[i] : buoyancy_z[j];
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        FUNC(forward_vz_cell)(
            sigma_v, vz, m_sigma_v_z, dvzdbuoyancy, buoyancy_z_shot, azh, bzh,
            dt, buoyancy_requires_grad, i, ny, nx, z, y, x, pml_z, rdz);
      }
#endif
      // vy -- 3-D: transverse, fed only by d(sigma_h)/dy.
      //       2-D: axis, fed only by d(sigma_v)/dy.
      if (y < ny - FD_PAD) {
        DW_DTYPE const buoyancy_y_shot =
            buoyancy_batched ? buoyancy_y[i] : buoyancy_y[j];
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        FUNC(forward_vy_cell)(
#if DW_NDIM == 3
            sigma_h, m_sigma_h_y,
#else
          sigma_v, m_sigma_v_y,
#endif
            vy, dvydbuoyancy, buoyancy_y_shot, ayh, byh, dt,
            buoyancy_requires_grad, i,
#if DW_NDIM >= 3
            ny,
#endif
            nx,
#if DW_NDIM >= 3
            z,
#endif
            y, x, pml_y, rdy);
      }
      // vx -- always transverse, fed only by d(sigma_h)/dx.
      if (x < nx - FD_PAD) {
        DW_DTYPE const buoyancy_x_shot =
            buoyancy_batched ? buoyancy_x[i] : buoyancy_x[j];
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(forward_vx_cell)(sigma_h, vx, m_sigma_h_x, dvxdbuoyancy,
                              buoyancy_x_shot, axh, bxh, dt,
                              buoyancy_requires_grad, i,
#if DW_NDIM >= 3
                              ny,
#endif
                              nx,
#if DW_NDIM >= 3
                              z,
#endif
                              y, x, pml_x, rdx);
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// sigma_v/sigma_h update: the c66 = 0 reduction of elastic_vti's
// normal-stress update -- see acoustic_vti.c.
__launch_bounds__(128) __global__ void forward_kernel_sigma(
    DW_DTYPE const *__restrict const c11, DW_DTYPE const *__restrict const c13,
    DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz,
#endif
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdz_store,
#endif
    DW_DTYPE *__restrict const dvydy_store,
    DW_DTYPE *__restrict const dvxdx_store,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const cnormal_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif

      int64_t const i = shot_idx * shot_numel + j;
      DW_DTYPE const c11_shot_i = c11_batched ? c11[i] : c11[j];
      DW_DTYPE const c13_shot_i = c13_batched ? c13[i] : c13[j];
      DW_DTYPE const c33_shot_i = c33_batched ? c33[i] : c33[j];

      FUNC(forward_sigma_cell)(
#if DW_NDIM >= 3
          vz,
#endif
          vy, vx, sigma_v, sigma_h,
#if DW_NDIM >= 3
          m_vzz,
#endif
          m_vyy, m_vxx,
#if DW_NDIM >= 3
          dvzdz_store,
#endif
          dvydy_store, dvxdx_store, c11_shot_i, c13_shot_i, c33_shot_i, dt,
          cnormal_requires_grad, i,
#if DW_NDIM >= 3
          ny,
#endif
          nx,
#if DW_NDIM >= 3
          z, DW_PML_TEST(z < pml_z0 || z >= pml_z1), az, bz, rdz,
#endif
          y, DW_PML_TEST(y < pml_y0 || y >= pml_y1), ay, by, rdy, x,
          DW_PML_TEST(x < pml_x0 || x >= pml_x1), ax, bx, rdx);
#if DW_NDIM == 3
    }
#endif
  }
}

int set_config(
    DW_DTYPE const dt_h,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_h,
#endif
    DW_DTYPE const rdy_h, DW_DTYPE const rdx_h, int64_t const n_shots_h,
#if DW_NDIM >= 3
    int64_t const nz_h,
#endif
    int64_t const ny_h, int64_t const nx_h,
#if DW_NDIM >= 3
    int64_t const n_sources_z_per_shot_h,
#endif
    int64_t const n_sources_y_per_shot_h, int64_t const n_sources_x_per_shot_h,
    int64_t const n_sources_sigma_h_per_shot_h,
    int64_t const n_sources_sigma_v_per_shot_h,
#if DW_NDIM >= 3
    int64_t const n_receivers_z_per_shot_h,
#endif
    int64_t const n_receivers_y_per_shot_h,
    int64_t const n_receivers_x_per_shot_h,
    int64_t const n_receivers_sigma_h_per_shot_h,
    int64_t const n_receivers_sigma_v_per_shot_h, int64_t const step_ratio_h,
#if DW_NDIM >= 3
    int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
    int64_t const pml_y0_h, int64_t const pml_y1_h, int64_t const pml_x0_h,
    int64_t const pml_x1_h, bool const c11_batched_h, bool const c13_batched_h,
    bool const c33_batched_h, bool const buoyancy_batched_h) {
#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#else
  int64_t const shot_numel_h = ny_h * nx_h;
#endif

  gpuErrchk(cudaMemcpyToSymbol(dt, &dt_h, sizeof(DW_DTYPE)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(n_sources_z_per_shot, &n_sources_z_per_shot_h,
                               sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(n_sources_y_per_shot, &n_sources_y_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_x_per_shot, &n_sources_x_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_sigma_h_per_shot,
                               &n_sources_sigma_h_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_sigma_v_per_shot,
                               &n_sources_sigma_v_per_shot_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_z_per_shot,
                               &n_receivers_z_per_shot_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_y_per_shot,
                               &n_receivers_y_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_x_per_shot,
                               &n_receivers_x_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_sigma_h_per_shot,
                               &n_receivers_sigma_h_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_sigma_v_per_shot,
                               &n_receivers_sigma_v_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(c11_batched, &c11_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(c13_batched, &c13_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(c33_batched, &c33_batched_h, sizeof(bool)));
  gpuErrchk(
      cudaMemcpyToSymbol(buoyancy_batched, &buoyancy_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(forward)(
        DW_DTYPE const *__restrict const c11,
        DW_DTYPE const *__restrict const c13,
        DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const buoyancy_z,
#endif
        DW_DTYPE const *__restrict const buoyancy_y,
        DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const f_z,
#endif
        DW_DTYPE const *__restrict const f_y,
        DW_DTYPE const *__restrict const f_x,
        DW_DTYPE const *__restrict const f_sigma_h,
        DW_DTYPE const *__restrict const f_sigma_v,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const vz,
#endif
        DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
        DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const m_vzz,
#endif
        DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM == 3
        DW_DTYPE *__restrict const m_sigma_v_z,
        DW_DTYPE *__restrict const m_sigma_h_y,
#else
        DW_DTYPE *__restrict const m_sigma_v_y,
#endif
        DW_DTYPE *__restrict const m_sigma_h_x,
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
#endif
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
        DW_DTYPE *__restrict const r_y, DW_DTYPE *__restrict const r_x,
        DW_DTYPE *__restrict const r_sigma_h,
        DW_DTYPE *__restrict const r_sigma_v,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const az,
        DW_DTYPE const *__restrict const bz,
        DW_DTYPE const *__restrict const azh,
        DW_DTYPE const *__restrict const bzh,
#endif
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const ayh,
        DW_DTYPE const *__restrict const byh,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const axh,
        DW_DTYPE const *__restrict const bxh,
#if DW_NDIM >= 3
        int64_t const *__restrict const sources_z_i,
#endif
        int64_t const *__restrict const sources_y_i,
        int64_t const *__restrict const sources_x_i,
        int64_t const *__restrict const sources_sigma_h_i,
        int64_t const *__restrict const sources_sigma_v_i,
#if DW_NDIM >= 3
        int64_t const *__restrict const receivers_z_i,
#endif
        int64_t const *__restrict const receivers_y_i,
        int64_t const *__restrict const receivers_x_i,
        int64_t const *__restrict const receivers_sigma_h_i,
        int64_t const *__restrict const receivers_sigma_v_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz_h,
#endif
        DW_DTYPE const rdy_h, DW_DTYPE const rdx_h, DW_DTYPE const dt_h,
        int64_t const nt, int64_t const n_shots_h,
#if DW_NDIM >= 3
        int64_t const nz_h,
#endif
        int64_t const ny_h, int64_t const nx_h,
#if DW_NDIM >= 3
        int64_t const n_sources_z_per_shot_h,
#endif
        int64_t const n_sources_y_per_shot_h,
        int64_t const n_sources_x_per_shot_h,
        int64_t const n_sources_sigma_h_per_shot_h,
        int64_t const n_sources_sigma_v_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot_h,
#endif
        int64_t n_receivers_y_per_shot_h, int64_t n_receivers_x_per_shot_h,
        int64_t const n_receivers_sigma_h_per_shot_h,
        int64_t const n_receivers_sigma_v_per_shot_h,
        int64_t const step_ratio_h, int64_t const storage_mode,
        size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
        bool const c11_requires_grad, bool const c13_requires_grad,
        bool const c33_requires_grad, bool const buoyancy_requires_grad,
        bool const c11_batched_h, bool const c13_batched_h,
        bool const c33_batched_h, bool const buoyancy_batched_h,
        bool const storage_compression, int64_t const start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0_h,
#endif
        int64_t const pml_y0_h, int64_t const pml_x0_h,
#if DW_NDIM >= 3
        int64_t const pml_z1_h,
#endif
        int64_t const pml_y1_h, int64_t const pml_x1_h,
#if DW_NDIM >= 3
        int64_t const bc_z0,
#endif
        int64_t const bc_y0, int64_t const bc_x0,
#if DW_NDIM >= 3
        int64_t const bc_z1,
#endif
        int64_t const bc_y1, int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;

#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot_h > 0) && (receivers_z_i == NULL)) return 1;
#endif
    if ((n_receivers_y_per_shot_h > 0) && (receivers_y_i == NULL)) return 1;
    if ((n_receivers_x_per_shot_h > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_sigma_h_per_shot_h > 0) && (receivers_sigma_h_i == NULL))
      return 1;
    if ((n_receivers_sigma_v_per_shot_h > 0) && (receivers_sigma_v_i == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot_h > 0) && sources_z_i == NULL) return 1;
#endif
    if ((n_sources_y_per_shot_h > 0) && sources_y_i == NULL) return 1;
    if ((n_sources_x_per_shot_h > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_sigma_h_per_shot_h > 0) && sources_sigma_h_i == NULL)
      return 1;
    if ((n_sources_sigma_v_per_shot_h > 0) && sources_sigma_v_i == NULL)
      return 1;
    bool const dw_has_sample = (start_t % step_ratio_h == 0 ||
                                nt > step_ratio_h - start_t % step_ratio_h);
#if DW_NDIM >= 3
    if (dw_has_sample && (buoyancy_requires_grad) &&
        (dvzdbuoyancy_store_1a == NULL))
      return 1;
#endif
    if (dw_has_sample && (buoyancy_requires_grad) &&
        (dvydbuoyancy_store_1a == NULL || dvxdbuoyancy_store_1a == NULL))
      return 1;
  }
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD + 1, dimBlock.z);
#else /* DW_NDIM == 2 */
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
  dim3 const dimBlock_src_rcv(32, 1, 1);

#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_z0 || bc_z1;
  int const has_bc_transverse = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_y0 || bc_y1;
  int const has_bc_transverse = bc_x0 || bc_x1;
#endif
  int const has_bc = has_bc_axis || has_bc_transverse;
  MAKE_BC_AXIS_TRANSVERSE();

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err =
        set_config(dt_h,
#if DW_NDIM >= 3
                   rdz_h,
#endif
                   rdy_h, rdx_h, n_shots_h,
#if DW_NDIM >= 3
                   nz_h,
#endif
                   ny_h, nx_h,
#if DW_NDIM >= 3
                   n_sources_z_per_shot_h,
#endif
                   n_sources_y_per_shot_h, n_sources_x_per_shot_h,
                   n_sources_sigma_h_per_shot_h, n_sources_sigma_v_per_shot_h,
#if DW_NDIM >= 3
                   n_receivers_z_per_shot_h,
#endif
                   n_receivers_y_per_shot_h, n_receivers_x_per_shot_h,
                   n_receivers_sigma_h_per_shot_h,
                   n_receivers_sigma_v_per_shot_h, step_ratio_h,
#if DW_NDIM >= 3
                   pml_z0_h, pml_z1_h,
#endif
                   pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, c11_batched_h,
                   c13_batched_h, c33_batched_h, buoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel = nz_h * ny_h * nx_h;
#else
  int64_t const shot_numel = ny_h * nx_h;
#endif

  bool const cnormal_requires_grad =
      c11_requires_grad || c13_requires_grad || c33_requires_grad;

#define OPEN_FILE(name, grad_cond)                                \
  ScopedFile fp_##name;                                           \
  file_group.add(fp_##name);                                      \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {              \
    if (!fp_##name.open(name##_filenames_ptr[0], "ab")) return 1; \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (c11_requires_grad || c13_requires_grad || c33_requires_grad ||
       buoyancy_requires_grad);

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  cudaEvent_t event_storage_done = event_storage_done_a;
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
  OPEN_FILE(dvzdz, cnormal_requires_grad)
#endif
  OPEN_FILE(dvydbuoyancy, buoyancy_requires_grad)
  OPEN_FILE(dvydy, cnormal_requires_grad)
  OPEN_FILE(dvxdbuoyancy, buoyancy_requires_grad)
  OPEN_FILE(dvxdx, cnormal_requires_grad)

  for (t = start_t; t < start_t + nt; ++t) {
    int64_t const step_idx = t / step_ratio_h;
#define SETUP_STORE(name, grad_cond)                                   \
  DW_DTYPE *__restrict name##_store_1_t =                              \
      name##_store_1a != NULL                                          \
          ? (name##_store_1a +                                         \
             ((storage_mode == STORAGE_DEVICE && !storage_compression) \
                  ? (t / step_ratio_h) * n_shots_h * shot_numel        \
                  : 0))                                                \
          : NULL;                                                      \
  void *__restrict const name##_store_2_t =                            \
      name##_store_2 != NULL                                           \
          ? ((uint8_t *)name##_store_2 +                               \
             (storage_mode == STORAGE_DEVICE && storage_compression    \
                  ? (t / step_ratio_h) * n_shots_h * shot_bytes_comp   \
                  : 0))                                                \
          : NULL;                                                      \
  void *__restrict const name##_store_3_t =                            \
      name##_store_3 != NULL                                           \
          ? ((uint8_t *)name##_store_3 +                               \
             (storage_mode == STORAGE_CPU                              \
                  ? (t / step_ratio_h) * n_shots_h *                   \
                        (storage_compression ? shot_bytes_comp         \
                                             : shot_bytes_uncomp)      \
                  : 0))                                                \
          : NULL;                                                      \
  bool const name##_cond = (grad_cond) && ((t % step_ratio_h) == 0);   \
  if ((grad_cond) && use_double_buffering) {                           \
    if (step_idx % 2 != 0) {                                           \
      name##_store_1_t = name##_store_1b;                              \
    }                                                                  \
  }

#if DW_NDIM >= 3
    SETUP_STORE(dvzdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvzdz, cnormal_requires_grad)
#endif
    SETUP_STORE(dvydbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvydy, cnormal_requires_grad)
    SETUP_STORE(dvxdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvxdx, cnormal_requires_grad)

    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    // Refill the sigma_v/sigma_h guard rings (read by forward_kernel_v
    // below); this also holds free-surface stress nodes to zero before
    // recording (no-op when every face is PML). sigma_v is imaged only on
    // the axis face, sigma_h only on the transverse faces.
    if (has_bc_axis) LAUNCH_FILL_SIGMA_V_BC();
    if (has_bc_transverse) LAUNCH_FILL_SIGMA_H_BC();

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
    if (n_receivers_sigma_h_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_sigma_h_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers_sigma_h<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                 stream_compute>>>(
          r_sigma_h + t * n_shots_h * n_receivers_sigma_h_per_shot_h, sigma_h,
          receivers_sigma_h_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_sigma_v_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_sigma_v_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers_sigma_v<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                 stream_compute>>>(
          r_sigma_v + t * n_shots_h * n_receivers_sigma_v_per_shot_h, sigma_v,
          receivers_sigma_v_i);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
        buoyancy_z,
#endif
        buoyancy_y, buoyancy_x,
#if DW_NDIM >= 3
        vz,
#endif
        vy, vx, sigma_v, sigma_h,
#if DW_NDIM == 3
        m_sigma_v_z, m_sigma_h_y,
#else
        m_sigma_v_y,
#endif
        m_sigma_h_x,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t,
#endif
        dvydbuoyancy_store_1_t, dvxdbuoyancy_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
        ay, ayh, by, byh, ax, axh, bx, bxh,
        buoyancy_requires_grad && ((t % step_ratio_h) == 0));
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

    // Refill the boundary-normal velocity guard ring (read by
    // forward_kernel_sigma below); holds rigid-wall velocity nodes to zero
    // (no-op for all-PML).
    if (has_bc) LAUNCH_FILL_V_BC();

    forward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        c11, c13, c33,
#if DW_NDIM >= 3
        vz,
#endif
        vy, vx, sigma_v, sigma_h,
#if DW_NDIM >= 3
        m_vzz,
#endif
        m_vyy, m_vxx,
#if DW_NDIM >= 3
        dvzdz_store_1_t,
#endif
        dvydy_store_1_t, dvxdx_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
        ay, ayh, by, byh, ax, axh, bx, bxh,
        cnormal_requires_grad && ((t % step_ratio_h) == 0));
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
#endif
    SAVE_SNAPSHOT(dvydbuoyancy)
    SAVE_SNAPSHOT(dvydy)
    SAVE_SNAPSHOT(dvxdbuoyancy)
    SAVE_SNAPSHOT(dvxdx)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    if (n_sources_sigma_h_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_sigma_h_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_sigma_h<<<dimGrid_sources, dimBlock_src_rcv, 0,
                            stream_compute>>>(
          sigma_h, f_sigma_h + t * n_shots_h * n_sources_sigma_h_per_shot_h,
          sources_sigma_h_i);
      CHECK_KERNEL_ERROR
    }
    if (n_sources_sigma_v_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_sigma_v_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_sigma_v<<<dimGrid_sources, dimBlock_src_rcv, 0,
                            stream_compute>>>(
          sigma_v, f_sigma_v + t * n_shots_h * n_sources_sigma_v_per_shot_h,
          sources_sigma_v_i);
      CHECK_KERNEL_ERROR
    }
  }

  /* Leave the returned wavefields boundary-consistent (e.g. free-surface
   * stress nodes held to zero, guard rings imaged), matching
   * _acoustic_vti_eager_core's own post-loop refill in acoustic_vti.py and
   * acoustic_vti.c's own matching fix -- without this, the last time
   * step's sigma_v/sigma_h update runs after the last in-loop stress
   * fill, so the returned guard ring/on-surface zero reflect a stale
   * (pre-update) state. Receiver traces and gradients are unaffected
   * (already recorded/consumed inside the loop before this point); only
   * the final returned wavefields change. */
  if (has_bc_axis) LAUNCH_FILL_SIGMA_V_BC();
  CHECK_KERNEL_ERROR
  if (has_bc_transverse) LAUNCH_FILL_SIGMA_H_BC();
  CHECK_KERNEL_ERROR
  if (has_bc) LAUNCH_FILL_V_BC();
  CHECK_KERNEL_ERROR

  if (use_double_buffering)
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

// ---------------------------------------------------------------------------
// Adjoint (task 7.H milestone H2). See acoustic_vti.c for the full
// derivation (citing docs/acoustic_vti_derivation.md Sec. 4-5) that this
// file mirrors kernel-for-kernel, exactly as its forward half does.
// ---------------------------------------------------------------------------

namespace {

// The C11/C13/C33/BUOYANCY_Z/Y/X (offset-read) macros and the VZ_Z/VY_Y/
// VX_X/SIGMA_V_Z/SIGMA_H_Y/SIGMA_V_Y/SIGMA_H_X (+_PML) adjoint-access macros
// used by FUNC(backward_vz_cell)/vy_cell/vx_cell/sigma_cell below now live in
// acoustic_vti_kernels.h (included near the top of this file), shared
// verbatim with acoustic_vti.c.

__launch_bounds__(32) __global__ void record_adjoint_receivers_sigma_h(
    DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const wf,
    int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_sigma_h_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_sigma_h_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__ void record_adjoint_receivers_sigma_v(
    DW_DTYPE *__restrict const r, DW_DTYPE const *__restrict const wf,
    int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_sigma_v_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_sigma_v_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
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

__launch_bounds__(32) __global__ void add_adjoint_sources_sigma_h(
    DW_DTYPE *__restrict const sigma_h, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_sigma_h_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_sigma_h_per_shot + source_idx;
    if (0 <= sources_i[k])
      sigma_h[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__ void add_adjoint_sources_sigma_v(
    DW_DTYPE *__restrict const sigma_v, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_sigma_v_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_sigma_v_per_shot + source_idx;
    if (0 <= sources_i[k])
      sigma_v[shot_idx * shot_numel + sources_i[k]] += f[k];
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
#else
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < ny && x < nx) {
    int64_t const i = y * nx + x;
#endif
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
}

// backward_kernel_sigma: adjoint of forward_kernel_sigma -- updates the
// VELOCITY adjoints (transpose of the stress-into-velocity coupling), reads
// (unping-ponged, single-buffer) m_vzz/m_vyy/m_vxx. The M_SIGMA_V_Z/
// M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X ping-ponged stress-derivative C-PML
// memory update and the grad_buoyancy accumulation are split out into
// backward_kernel_v_pml_mem below and run in a separate pass AFTER the
// velocity guard-ring fold (see that kernel's comment and the driver-side
// comment at its call site): v*[i] must be read post-fold, since the fold
// changes it for cells it touches (e.g. a rigid on-row velocity clamp
// node).
__launch_bounds__(128) __global__ void backward_kernel_sigma(
    DW_DTYPE const *__restrict const c11, DW_DTYPE const *__restrict const c13,
    DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
    DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
    DW_DTYPE const *__restrict const sigma_v,
    DW_DTYPE const *__restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vzz,
#endif
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const bz,
#endif
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif
      DW_DTYPE const *__restrict const c11_shot =
          c11_batched ? c11 + shot_idx * shot_numel : c11;
      DW_DTYPE const *__restrict const c13_shot =
          c13_batched ? c13 + shot_idx * shot_numel : c13;
      DW_DTYPE const *__restrict const c33_shot =
          c33_batched ? c33 + shot_idx * shot_numel : c33;

#if DW_NDIM >= 3
      int64_t const pml_z0h = pml_z0;
      int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
#endif
      int64_t const pml_y0h = pml_y0;
      int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
      int64_t const pml_x0h = pml_x0;
      int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        FUNC(backward_vz_cell)(c13_shot, c33_shot, sigma_h, sigma_v, vz, m_vzz,
                               bz, dt, i, j, ny, nx, z, y, x, pml_z, rdz);
      }
#endif
      if (y < ny - FD_PAD) {
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        FUNC(backward_vy_cell)(
#if DW_NDIM == 3
            c11_shot, c13_shot,
#else
          c13_shot, c33_shot,
#endif
            sigma_h, sigma_v, vy, m_vyy, by, dt, i, j,
#if DW_NDIM >= 3
            ny,
#endif
            nx,
#if DW_NDIM >= 3
            z,
#endif
            y, x, pml_y, rdy);
      }
      if (x < nx - FD_PAD) {
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_vx_cell)(c11_shot, c13_shot, sigma_h, sigma_v, vx, m_vxx,
                               bx, dt, i, j,
#if DW_NDIM >= 3
                               ny,
#endif
                               nx,
#if DW_NDIM >= 3
                               z,
#endif
                               y, x, pml_x, rdx);
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// backward_kernel_v: adjoint of forward_kernel_v -- accumulates
// grad_c11/c13/c33, updates the (unping-ponged, single-buffer)
// velocity-derivative C-PML memories m_vzz/m_vyy/m_vxx (written here, having
// been read by backward_kernel_sigma above in the same reversed timestep),
// and updates the STRESS adjoints (transpose of the velocity-into-stress
// buoyancy coupling), reading the SAME (pre-swap) stress-derivative C-PML
// memories backward_kernel_sigma read (not the "n" ones it just wrote).
__launch_bounds__(128) __global__ void backward_kernel_v(
    DW_DTYPE const *__restrict const c11, DW_DTYPE const *__restrict const c13,
    DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz,
#endif
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const m_sigma_v_z,
    DW_DTYPE const *__restrict const m_sigma_h_y,
#else
    DW_DTYPE const *__restrict const m_sigma_v_y,
#endif
    DW_DTYPE const *__restrict const m_sigma_h_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdz_store,
#endif
    DW_DTYPE const *__restrict const dvydy_store,
    DW_DTYPE const *__restrict const dvxdx_store,
    DW_DTYPE *__restrict const grad_c11, DW_DTYPE *__restrict const grad_c13,
    DW_DTYPE *__restrict const grad_c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const c11_requires_grad, bool const c13_requires_grad,
    bool const c33_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif
      DW_DTYPE const *__restrict const c11_shot =
          c11_batched ? c11 + shot_idx * shot_numel : c11;
      DW_DTYPE const *__restrict const c13_shot =
          c13_batched ? c13 + shot_idx * shot_numel : c13;
      DW_DTYPE const *__restrict const c33_shot =
          c33_batched ? c33 + shot_idx * shot_numel : c33;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

      int64_t const i = shot_idx * shot_numel + j;

      FUNC(backward_sigma_cell)(
#if DW_NDIM >= 3
          vz, buoyancy_z_shot,
#endif
          vy, buoyancy_y_shot, vx, buoyancy_x_shot, sigma_v, sigma_h,
#if DW_NDIM >= 3
          m_vzz,
#endif
          m_vyy, m_vxx,
#if DW_NDIM == 3
          m_sigma_v_z, m_sigma_h_y,
#else
        m_sigma_v_y,
#endif
          m_sigma_h_x,
#if DW_NDIM >= 3
          dvzdz_store,
#endif
          dvydy_store, dvxdx_store, c11_shot[j], c13_shot[j], c33_shot[j],
          grad_c11, grad_c13, grad_c33, dt, step_ratio, c11_requires_grad,
          c13_requires_grad, c33_requires_grad, i, j,
#if DW_NDIM >= 3
          ny,
#endif
          nx,
#if DW_NDIM >= 3
          z, DW_PML_TEST(z < pml_z0 || z >= pml_z1), az, bzh, rdz,
#endif
          y, DW_PML_TEST(y < pml_y0 || y >= pml_y1), ay, byh, rdy, x,
          DW_PML_TEST(x < pml_x0 || x >= pml_x1), ax, bxh, rdx);
#if DW_NDIM == 3
    }
#endif
  }
}

/* ========================= Guard-band gathers ========================= *
 *
 * CUDA counterpart of the guard-band-only companion functions in acoustic_
 * vti_kernels.h (FUNC(backward_vz_cell_guard) etc -- see that file's
 * "Guard-band gathers" module comment for the full rationale). The three
 * kernels below are guard-band-only companions to backward_kernel_sigma/
 * backward_kernel_v above: same fields, same per-cell math (via the
 * *_guard functions), but launched over the WHOLE domain (grid indices
 * start at 0, not FD_PAD) since a guard cell by definition lives inside
 * the band backward_kernel_sigma/backward_kernel_v exclude. Each thread
 * computes at most one update per field at its own, unique (shot, z, y, x)
 * -- no two threads ever accumulate into the same output index -- so these
 * are race-free without atomics purely from CUDA's per-thread-independent-
 * output property; ordering versus every other kernel here is guaranteed
 * by the single compute stream (stream_compute) every launch in this file
 * uses, so placement (immediately after the interior kernel, before the
 * corresponding fold-fill launch -- see FUNC(backward)'s driver body
 * below) is the only thing that matters, not concurrency.
 * backward_kernel_v_pml_mem is the post-fold M_SIGMA_V_Z/M_SIGMA_H_Y/
 * M_SIGMA_V_Y/M_SIGMA_H_X memory-variable update and buoyancy-gradient pass
 * acoustic_vti_kernels.h's own module comment describes (both moved out of
 * FUNC(backward_v*_cell) into a separate pass so they read v*[i] after the
 * velocity fold changes near-face interior cells); it is launched
 * unconditionally (the PML-memory write matters every time step regardless
 * of buoyancy_requires_grad) over (approximately) the same unextended
 * range FUNC(forward_vz_cell) etc. used to write dv*dbuoyancy in the
 * forward pass.
 */
__launch_bounds__(128) __global__
    void backward_kernel_sigma_guard(DW_DTYPE const *__restrict const c11,
                                     DW_DTYPE const *__restrict const c13,
                                     DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
                                     DW_DTYPE *__restrict const vz,
#endif
                                     DW_DTYPE *__restrict const vy,
                                     DW_DTYPE *__restrict const vx,
                                     DW_DTYPE const *__restrict const sigma_v,
                                     DW_DTYPE const *__restrict const sigma_h,
#if DW_NDIM >= 3
                                     int64_t const bc_z0, int64_t const bc_z1,
#endif
                                     int64_t const bc_y0, int64_t const bc_y1,
                                     int64_t const bc_x0, int64_t const bc_x1) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif
      DW_DTYPE const *__restrict const c11_shot =
          c11_batched ? c11 + shot_idx * shot_numel : c11;
      DW_DTYPE const *__restrict const c13_shot =
          c13_batched ? c13 + shot_idx * shot_numel : c13;
      DW_DTYPE const *__restrict const c33_shot =
          c33_batched ? c33 + shot_idx * shot_numel : c33;

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        if (FUNC(dw_in_target)(rz, z) && !FUNC(dw_in_std)(rz, z) &&
            y >= FD_PAD && y < ny - FD_PAD + 1 && x >= FD_PAD &&
            x < nx - FD_PAD + 1) {
          FUNC(backward_vz_cell_guard)(c13_shot, c33_shot, sigma_h, sigma_v, vz,
                                       dt, i, j, nz, ny, nx, z, y, x, rdz);
        }
      }
#endif
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
        if (FUNC(dw_in_target)(ry, y) && !FUNC(dw_in_std)(ry, y) &&
            x >= FD_PAD && x < nx - FD_PAD + 1
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          FUNC(backward_vy_cell_guard)(
#if DW_NDIM == 3
              c11_shot, c13_shot,
#else
            c13_shot, c33_shot,
#endif
              sigma_h, sigma_v, vy, dt, i, j,
#if DW_NDIM >= 3
              nz,
#endif
              ny, nx,
#if DW_NDIM >= 3
              z,
#endif
              y, x, rdy);
        }
      }
      {
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        if (FUNC(dw_in_target)(rx, x) && !FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
            && y >= FD_PAD && y < ny - FD_PAD + 1) {
          FUNC(backward_vx_cell_guard)(c11_shot, c13_shot, sigma_h, sigma_v, vx,
                                       dt, i, j,
#if DW_NDIM >= 3
                                       nz,
#endif
                                       ny, nx,
#if DW_NDIM >= 3
                                       z,
#endif
                                       y, x, rdx);
        }
      }
#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel_v_pml_mem(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const m_sigma_v_z,
    DW_DTYPE *__restrict const m_sigma_v_zn,
    DW_DTYPE const *__restrict const m_sigma_h_y,
    DW_DTYPE *__restrict const m_sigma_h_yn,
#else
    DW_DTYPE const *__restrict const m_sigma_v_y,
    DW_DTYPE *__restrict const m_sigma_v_yn,
#endif
    DW_DTYPE const *__restrict const m_sigma_h_x,
    DW_DTYPE *__restrict const m_sigma_h_xn,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdbuoyancy,
#endif
    DW_DTYPE const *__restrict const dvydbuoyancy,
    DW_DTYPE const *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
    DW_DTYPE *__restrict const grad_buoyancy_y,
    DW_DTYPE *__restrict const grad_buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const azh,
#endif
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const axh,
    bool const buoyancy_requires_grad_t) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      /* Same (unextended) range FUNC(forward_vz_cell) etc. used in the
       * forward pass -- matches the CPU driver's pml_bounds_*h[0]/[3]
       * loops exactly. */
#if DW_NDIM >= 3
      if (z >= FD_PAD && z < nz - FD_PAD && y >= FD_PAD &&
          y < ny - FD_PAD + 1 && x >= FD_PAD && x < nx - FD_PAD + 1) {
        bool const is_pml_z =
            DW_PML_TEST(z < pml_z0 || z >= MAX(pml_z0, pml_z1 - 1));
        FUNC(backward_vz_cell_pml_mem)(vz, m_sigma_v_z, m_sigma_v_zn,
                                       buoyancy_z_shot[j], azh, dt, i, ny, nx,
                                       z, y, x, is_pml_z);
        if (buoyancy_requires_grad_t) {
          grad_buoyancy_z[i] += vz[i] * dvzdbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif
      if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD && x < nx - FD_PAD + 1
#if DW_NDIM >= 3
          && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
      ) {
        bool const is_pml_y =
            DW_PML_TEST(y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1));
        FUNC(backward_vy_cell_pml_mem)(vy,
#if DW_NDIM == 3
                                       m_sigma_h_y, m_sigma_h_yn,
#else
                                     m_sigma_v_y, m_sigma_v_yn,
#endif
                                       buoyancy_y_shot[j], ayh, dt, i,
#if DW_NDIM >= 3
                                       ny, z,
#endif
                                       nx, y, x, is_pml_y);
        if (buoyancy_requires_grad_t) {
          grad_buoyancy_y[i] += vy[i] * dvydbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
      if (x >= FD_PAD && x < nx - FD_PAD
#if DW_NDIM >= 3
          && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
          && y >= FD_PAD && y < ny - FD_PAD + 1) {
        bool const is_pml_x =
            DW_PML_TEST(x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1));
        FUNC(backward_vx_cell_pml_mem)(vx, m_sigma_h_x, m_sigma_h_xn,
                                       buoyancy_x_shot[j], axh, dt, i,
#if DW_NDIM >= 3
                                       ny, z,
#endif
                                       nx, y, x, is_pml_x);
        if (buoyancy_requires_grad_t) {
          grad_buoyancy_x[i] += vx[i] * dvxdbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel_v_guard(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
    DW_DTYPE const *__restrict const buoyancy_y,
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
    int64_t const bc_y0, int64_t const bc_y1, int64_t const bc_x0,
    int64_t const bc_x1) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 2 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#else /* DW_NDIM == 2 */
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
        if (FUNC(dw_in_target)(rz, z) && !FUNC(dw_in_std)(rz, z) &&
            y >= FD_PAD && y < ny - FD_PAD + 1 && x >= FD_PAD &&
            x < nx - FD_PAD + 1) {
          FUNC(backward_sigma_v_z_cell_guard)(buoyancy_z_shot, vz, sigma_v, dt,
                                              i, j, nz, ny, nx, z, rdz);
        }
      }
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
        if (FUNC(dw_in_target)(ry, y) && !FUNC(dw_in_std)(ry, y) &&
            z >= FD_PAD && z < nz - FD_PAD + 1 && x >= FD_PAD &&
            x < nx - FD_PAD + 1) {
          FUNC(backward_sigma_h_y_cell_guard)(buoyancy_y_shot, vy, sigma_h, dt,
                                              i, j, ny, nx, y, rdy);
        }
      }
#else /* DW_NDIM == 2 */
    {
      FUNC(dw_ext_range)
      const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
      if (FUNC(dw_in_target)(ry, y) && !FUNC(dw_in_std)(ry, y) && x >= FD_PAD &&
          x < nx - FD_PAD + 1) {
        FUNC(backward_sigma_v_y_cell_guard)(buoyancy_y_shot, vy, sigma_v, dt, i,
                                            j, ny, nx, y, rdy);
      }
    }
#endif
      {
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        if (FUNC(dw_in_target)(rx, x) && !FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
            && y >= FD_PAD && y < ny - FD_PAD + 1) {
          FUNC(backward_sigma_h_x_cell_guard)(buoyancy_x_shot, vx, sigma_h, dt,
                                              i, j,
#if DW_NDIM >= 3
                                              ny,
#endif
                                              nx, x, rdx);
        }
      }
#if DW_NDIM == 3
    }
#endif
  }
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(backward)(
        DW_DTYPE const *__restrict const c11,
        DW_DTYPE const *__restrict const c13,
        DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const buoyancy_z,
#endif
        DW_DTYPE const *__restrict const buoyancy_y,
        DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const grad_r_z,
#endif
        DW_DTYPE const *__restrict const grad_r_y,
        DW_DTYPE const *__restrict const grad_r_x,
        DW_DTYPE const *__restrict const grad_r_sigma_h,
        DW_DTYPE const *__restrict const grad_r_sigma_v,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const vz,
#endif
        DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
        DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const m_vzz,
#endif
        DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM == 3
        DW_DTYPE *__restrict m_sigma_v_z, DW_DTYPE *__restrict m_sigma_h_y,
#else
        DW_DTYPE *__restrict m_sigma_v_y,
#endif
        DW_DTYPE *__restrict m_sigma_h_x,
#if DW_NDIM == 3
        DW_DTYPE *__restrict m_sigma_v_zn, DW_DTYPE *__restrict m_sigma_h_yn,
#else
        DW_DTYPE *__restrict m_sigma_v_yn,
#endif
        DW_DTYPE *__restrict m_sigma_h_xn,
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
#endif
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
        DW_DTYPE *__restrict const grad_f_y,
        DW_DTYPE *__restrict const grad_f_x,
        DW_DTYPE *__restrict const grad_f_sigma_h,
        DW_DTYPE *__restrict const grad_f_sigma_v,
        DW_DTYPE *__restrict const grad_c11,
        DW_DTYPE *__restrict const grad_c13,
        DW_DTYPE *__restrict const grad_c33,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
        DW_DTYPE *__restrict const grad_buoyancy_y,
        DW_DTYPE *__restrict const grad_buoyancy_x,
        DW_DTYPE *__restrict const grad_c11_shot,
        DW_DTYPE *__restrict const grad_c13_shot,
        DW_DTYPE *__restrict const grad_c33_shot,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_buoyancy_z_shot,
#endif
        DW_DTYPE *__restrict const grad_buoyancy_y_shot,
        DW_DTYPE *__restrict const grad_buoyancy_x_shot,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const az,
        DW_DTYPE const *__restrict const bz,
        DW_DTYPE const *__restrict const azh,
        DW_DTYPE const *__restrict const bzh,
#endif
        DW_DTYPE const *__restrict const ay,
        DW_DTYPE const *__restrict const by,
        DW_DTYPE const *__restrict const ayh,
        DW_DTYPE const *__restrict const byh,
        DW_DTYPE const *__restrict const ax,
        DW_DTYPE const *__restrict const bx,
        DW_DTYPE const *__restrict const axh,
        DW_DTYPE const *__restrict const bxh,
#if DW_NDIM >= 3
        int64_t const *__restrict const sources_z_i,
#endif
        int64_t const *__restrict const sources_y_i,
        int64_t const *__restrict const sources_x_i,
        int64_t const *__restrict const sources_sigma_h_i,
        int64_t const *__restrict const sources_sigma_v_i,
#if DW_NDIM >= 3
        int64_t const *__restrict const receivers_z_i,
#endif
        int64_t const *__restrict const receivers_y_i,
        int64_t const *__restrict const receivers_x_i,
        int64_t const *__restrict const receivers_sigma_h_i,
        int64_t const *__restrict const receivers_sigma_v_i,
#if DW_NDIM >= 3
        DW_DTYPE const rdz_h,
#endif
        DW_DTYPE const rdy_h, DW_DTYPE const rdx_h, DW_DTYPE const dt_h,
        int64_t const nt, int64_t const n_shots_h,
#if DW_NDIM >= 3
        int64_t const nz_h,
#endif
        int64_t const ny_h, int64_t const nx_h,
#if DW_NDIM >= 3
        int64_t const n_sources_z_per_shot_h,
#endif
        int64_t const n_sources_y_per_shot_h,
        int64_t const n_sources_x_per_shot_h,
        int64_t const n_sources_sigma_h_per_shot_h,
        int64_t const n_sources_sigma_v_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot_h,
#endif
        int64_t n_receivers_y_per_shot_h, int64_t n_receivers_x_per_shot_h,
        int64_t const n_receivers_sigma_h_per_shot_h,
        int64_t const n_receivers_sigma_v_per_shot_h,
        int64_t const step_ratio_h, int64_t const storage_mode,
        size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
        bool const c11_requires_grad, bool const c13_requires_grad,
        bool const c33_requires_grad, bool const buoyancy_requires_grad,
        bool const c11_batched_h, bool const c13_batched_h,
        bool const c33_batched_h, bool const buoyancy_batched_h,
        bool const storage_compression, int64_t const start_t,
#if DW_NDIM >= 3
        int64_t const pml_z0_h,
#endif
        int64_t const pml_y0_h, int64_t const pml_x0_h,
#if DW_NDIM >= 3
        int64_t const pml_z1_h,
#endif
        int64_t const pml_y1_h, int64_t const pml_x1_h,
#if DW_NDIM >= 3
        int64_t const bc_z0,
#endif
        int64_t const bc_y0, int64_t const bc_x0,
#if DW_NDIM >= 3
        int64_t const bc_z1,
#endif
        int64_t const bc_y1, int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;
    if ((c11_requires_grad) && (grad_c11 == NULL)) return 1;
    if ((c13_requires_grad) && (grad_c13 == NULL)) return 1;
    if ((c33_requires_grad) && (grad_c33 == NULL)) return 1;
#if DW_NDIM >= 3
    if ((buoyancy_requires_grad) && (grad_buoyancy_z == NULL)) return 1;
#endif
    if ((buoyancy_requires_grad) &&
        (grad_buoyancy_y == NULL || grad_buoyancy_x == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot_h > 0) && (receivers_z_i == NULL)) return 1;
#endif
    if ((n_receivers_y_per_shot_h > 0) && (receivers_y_i == NULL)) return 1;
    if ((n_receivers_x_per_shot_h > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_sigma_h_per_shot_h > 0) && (receivers_sigma_h_i == NULL))
      return 1;
    if ((n_receivers_sigma_v_per_shot_h > 0) && (receivers_sigma_v_i == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot_h > 0) && sources_z_i == NULL) return 1;
#endif
    if ((n_sources_y_per_shot_h > 0) && sources_y_i == NULL) return 1;
    if ((n_sources_x_per_shot_h > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_sigma_h_per_shot_h > 0) && sources_sigma_h_i == NULL)
      return 1;
    if ((n_sources_sigma_v_per_shot_h > 0) && sources_sigma_v_i == NULL)
      return 1;
  }
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2 * FD_PAD + 1, dimBlock.z);
#else /* DW_NDIM == 2 */
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2 * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2 * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
  /* Guard-band kernel launch grid: covers the WHOLE domain (indices 0..n-1
   * on every axis), unlike dimGrid above (which starts at FD_PAD and stops
   * short of the far edge) -- see the "Guard-band gathers" module comment
   * above backward_kernel_sigma_guard. */
#if DW_NDIM == 3
  unsigned int const gridx_g = ceil_div(nx_h, dimBlock.x);
  unsigned int const gridy_g = ceil_div(ny_h, dimBlock.y);
  unsigned int const gridz_g = ceil_div(nz_h, dimBlock.z);
#else /* DW_NDIM == 2 */
  unsigned int const gridx_g = ceil_div(nx_h, dimBlock.x);
  unsigned int const gridy_g = ceil_div(ny_h, dimBlock.y);
  unsigned int const gridz_g = ceil_div(n_shots_h, dimBlock.z);
#endif
  dim3 const dimGrid_guard(gridx_g, gridy_g, gridz_g);
  dim3 const dimBlock_src_rcv(32, 1, 1);

#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_z0 || bc_z1;
  int const has_bc_transverse = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc_axis = bc_y0 || bc_y1;
  int const has_bc_transverse = bc_x0 || bc_x1;
#endif
  int const has_bc = has_bc_axis || has_bc_transverse;
  MAKE_BC_AXIS_TRANSVERSE();

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err =
        set_config(dt_h,
#if DW_NDIM >= 3
                   rdz_h,
#endif
                   rdy_h, rdx_h, n_shots_h,
#if DW_NDIM >= 3
                   nz_h,
#endif
                   ny_h, nx_h,
#if DW_NDIM >= 3
                   n_sources_z_per_shot_h,
#endif
                   n_sources_y_per_shot_h, n_sources_x_per_shot_h,
                   n_sources_sigma_h_per_shot_h, n_sources_sigma_v_per_shot_h,
#if DW_NDIM >= 3
                   n_receivers_z_per_shot_h,
#endif
                   n_receivers_y_per_shot_h, n_receivers_x_per_shot_h,
                   n_receivers_sigma_h_per_shot_h,
                   n_receivers_sigma_v_per_shot_h, step_ratio_h,
#if DW_NDIM >= 3
                   pml_z0_h, pml_z1_h,
#endif
                   pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h, c11_batched_h,
                   c13_batched_h, c33_batched_h, buoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel = nz_h * ny_h * nx_h;
#else
  int64_t const shot_numel = ny_h * nx_h;
#endif

  bool const cnormal_requires_grad =
      c11_requires_grad || c13_requires_grad || c33_requires_grad;

#define OPEN_FILE_READ(name, grad_cond)                           \
  ScopedFile fp_##name;                                           \
  file_group.add(fp_##name);                                      \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {              \
    if (!fp_##name.open(name##_filenames_ptr[0], "rb")) return 1; \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (c11_requires_grad || c13_requires_grad || c33_requires_grad ||
       buoyancy_requires_grad);

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
  OPEN_FILE_READ(dvzdz, cnormal_requires_grad)
#endif
  OPEN_FILE_READ(dvydbuoyancy, buoyancy_requires_grad)
  OPEN_FILE_READ(dvydy, cnormal_requires_grad)
  OPEN_FILE_READ(dvxdbuoyancy, buoyancy_requires_grad)
  OPEN_FILE_READ(dvxdx, cnormal_requires_grad)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio_h;

#define SETUP_STORE_LOAD(name, grad_cond)                                 \
  DW_DTYPE const *__restrict name##_store_1_t =                           \
      name##_store_1a != NULL                                             \
          ? (name##_store_1a +                                            \
             ((storage_mode == STORAGE_DEVICE && !storage_compression)    \
                  ? (t / step_ratio_h) * n_shots_h * shot_numel           \
                  : 0))                                                   \
          : NULL;                                                         \
  void *__restrict const name##_store_2_t =                               \
      name##_store_2 != NULL                                              \
          ? ((uint8_t *)name##_store_2 +                                  \
             (storage_mode == STORAGE_DEVICE && storage_compression       \
                  ? (t / step_ratio_h) * n_shots_h * shot_bytes_comp      \
                  : 0))                                                   \
          : NULL;                                                         \
  void *__restrict const name##_store_3_t =                               \
      name##_store_3 != NULL                                              \
          ? ((uint8_t *)name##_store_3 +                                  \
             (storage_mode == STORAGE_CPU                                 \
                  ? (t / step_ratio_h) * n_shots_h *                      \
                        (storage_compression ? shot_bytes_comp            \
                                             : shot_bytes_uncomp)         \
                  : 0))                                                   \
          : NULL;                                                         \
  bool const name##_cond = (grad_cond) && ((t % step_ratio_h) == 0);      \
  if (name##_cond) {                                                      \
    if (use_double_buffering && step_idx % 2 != 0) {                      \
      name##_store_1_t = name##_store_1b;                                 \
    }                                                                     \
    if (STORAGE_FUNC(load_snapshot_gpu)(                                  \
            (void *)name##_store_1_t, name##_store_2_t, name##_store_3_t, \
            fp_##name, storage_mode, storage_compression, step_idx,       \
            shot_bytes_uncomp, shot_bytes_comp, n_shots_h, DIM_ARGS,      \
            use_double_buffering ? stream_storage : stream_compute) != 0) \
      return 1;                                                           \
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
    SETUP_STORE_LOAD(dvzdz, cnormal_requires_grad)
#endif
    SETUP_STORE_LOAD(dvydbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvydy, cnormal_requires_grad)
    SETUP_STORE_LOAD(dvxdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvxdx, cnormal_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    // Fold the (adjoint) sigma_v/sigma_h guard rings back into the interior
    // before backward_kernel_sigma below reads them and before the
    // source-gradient extraction just below. Unlike acoustic's single
    // pressure field (genuinely self-adjoint under a plain re-fill),
    // acoustic_vti's sigma_v/sigma_h feed back into DIFFERENT velocity
    // components than they are imaged from at a reflecting face, and the
    // per-cell gathers only ever run over the grid interior, so this
    // applies the fill's true transpose instead of reapplying the forward
    // fill (see the module comment above fill_p_bc_line_adjoint_dev).
    if (has_bc_axis) LAUNCH_FILL_SIGMA_V_BC_ADJOINT();
    if (has_bc_transverse) LAUNCH_FILL_SIGMA_H_BC_ADJOINT();

    if (n_sources_sigma_h_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_sigma_h_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_receivers_sigma_h<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                         stream_compute>>>(
          grad_f_sigma_h + t * n_shots_h * n_sources_sigma_h_per_shot_h,
          sigma_h, sources_sigma_h_i);
      CHECK_KERNEL_ERROR
    }
    if (n_sources_sigma_v_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_sigma_v_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_receivers_sigma_v<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                         stream_compute>>>(
          grad_f_sigma_v + t * n_shots_h * n_sources_sigma_v_per_shot_h,
          sigma_v, sources_sigma_v_i);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        c11, c13, c33,
#if DW_NDIM >= 3
        vz,
#endif
        vy, vx, sigma_v, sigma_h,
#if DW_NDIM >= 3
        m_vzz,
#endif
        m_vyy, m_vxx,
#if DW_NDIM >= 3
        bz,
#endif
        by, bx);
    CHECK_KERNEL_ERROR

    // Guard-band extension of the velocity-adjoint gathers (see the
    // "Guard-band gathers" module comment above backward_kernel_sigma_
    // guard). Must run (and complete, via the shared stream_compute
    // ordering) before LAUNCH_FILL_V_BC_ADJOINT below, which folds this
    // guard-band deposit into the interior.
    if (has_bc) {
      backward_kernel_sigma_guard<<<dimGrid_guard, dimBlock, 0,
                                    stream_compute>>>(c11, c13, c33,
#if DW_NDIM >= 3
                                                      vz,
#endif
                                                      vy, vx, sigma_v, sigma_h,
#if DW_NDIM >= 3
                                                      bc_z0, bc_z1,
#endif
                                                      bc_y0, bc_y1, bc_x0,
                                                      bc_x1);
      CHECK_KERNEL_ERROR
    }

    // Fold the (adjoint) velocity guard ring back into the interior before
    // backward_kernel_v below reads it. Unlike acoustic's single
    // pressure-velocity system, this applies the true transpose of the
    // velocity fill rather than reapplying the forward fill (see the
    // module comment above fill_v_bc_line_adjoint_dev).
    if (has_bc) LAUNCH_FILL_V_BC_ADJOINT();

    // The M_SIGMA_V_Z/M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X PML memory-
    // variable update (split out of FUNC(backward_v*_cell) -- see FUNC(
    // backward_vz_cell_pml_mem)'s comment in acoustic_vti_kernels.h) and
    // the buoyancy-gradient accumulation both read the velocity adjoint at
    // (near-face) interior cells the fold above may have just changed, so
    // both must run after it.
    backward_kernel_v_pml_mem<<<dimGrid_guard, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
        buoyancy_z,
#endif
        buoyancy_y, buoyancy_x,
#if DW_NDIM >= 3
        vz,
#endif
        vy, vx,
#if DW_NDIM == 3
        m_sigma_v_z, m_sigma_v_zn, m_sigma_h_y, m_sigma_h_yn,
#else
        m_sigma_v_y, m_sigma_v_yn,
#endif
        m_sigma_h_x, m_sigma_h_xn,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t,
#endif
        dvydbuoyancy_store_1_t, dvxdbuoyancy_store_1_t,
#if DW_NDIM >= 3
        grad_buoyancy_z_shot,
#endif
        grad_buoyancy_y_shot, grad_buoyancy_x_shot,
#if DW_NDIM >= 3
        azh,
#endif
        ayh, axh, buoyancy_requires_grad && ((t % step_ratio_h) == 0));
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
        c11, c13, c33,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
        buoyancy_y, buoyancy_x,
#if DW_NDIM >= 3
        vz,
#endif
        vy, vx, sigma_v, sigma_h,
#if DW_NDIM >= 3
        m_vzz,
#endif
        m_vyy, m_vxx,
#if DW_NDIM == 3
        m_sigma_v_z, m_sigma_h_y,
#else
        m_sigma_v_y,
#endif
        m_sigma_h_x,
#if DW_NDIM >= 3
        dvzdz_store_1_t,
#endif
        dvydy_store_1_t, dvxdx_store_1_t, grad_c11_shot, grad_c13_shot,
        grad_c33_shot,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
        ay, ayh, by, byh, ax, axh, bx, bxh,
        c11_requires_grad && ((t % step_ratio_h) == 0),
        c13_requires_grad && ((t % step_ratio_h) == 0),
        c33_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

    // Guard-band extension of the stress-adjoint gathers (see the
    // "Guard-band gathers" module comment above backward_kernel_sigma_
    // guard). Must run (and complete) before the NEXT time step's
    // LAUNCH_FILL_SIGMA_*_BC_ADJOINT calls (at the top of this loop),
    // which fold this guard-band deposit into the interior; the shared
    // stream_compute ordering guarantees that regardless of where in this
    // iteration the launch sits.
    if (has_bc) {
      backward_kernel_v_guard<<<dimGrid_guard, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
          buoyancy_z,
#endif
          buoyancy_y, buoyancy_x,
#if DW_NDIM >= 3
          vz,
#endif
          vy, vx, sigma_v, sigma_h,
#if DW_NDIM >= 3
          bc_z0, bc_z1,
#endif
          bc_y0, bc_y1, bc_x0, bc_x1);
      CHECK_KERNEL_ERROR
    }

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
    if (n_receivers_sigma_h_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_sigma_h_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_sources_sigma_h<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                    stream_compute>>>(
          sigma_h,
          grad_r_sigma_h + t * n_shots_h * n_receivers_sigma_h_per_shot_h,
          receivers_sigma_h_i);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_sigma_v_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_sigma_v_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_sources_sigma_v<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                    stream_compute>>>(
          sigma_v,
          grad_r_sigma_v + t * n_shots_h * n_receivers_sigma_v_per_shot_h,
          receivers_sigma_v_i);
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM == 3
    std::swap(m_sigma_v_z, m_sigma_v_zn);
    std::swap(m_sigma_h_y, m_sigma_h_yn);
#else
    std::swap(m_sigma_v_y, m_sigma_v_yn);
#endif
    std::swap(m_sigma_h_x, m_sigma_h_xn);
  }

#if DW_NDIM == 3
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine = ceil_div(nx_h, dimBlock_combine.x);
  unsigned int const gridy_combine = ceil_div(ny_h, dimBlock_combine.y);
  unsigned int const gridz_combine = ceil_div(nz_h, dimBlock_combine.z);
#else
  dim3 const dimBlock_combine(32, 4, 1);
  unsigned int const gridx_combine = ceil_div(nx_h, dimBlock_combine.x);
  unsigned int const gridy_combine = ceil_div(ny_h, dimBlock_combine.y);
  unsigned int const gridz_combine = 1;
#endif
  dim3 const dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

  if (c11_requires_grad && !c11_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c11, grad_c11_shot);
    CHECK_KERNEL_ERROR
  }
  if (c13_requires_grad && !c13_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c13, grad_c13_shot);
    CHECK_KERNEL_ERROR
  }
  if (c33_requires_grad && !c33_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c33, grad_c33_shot);
    CHECK_KERNEL_ERROR
  }
  if (buoyancy_requires_grad && !buoyancy_batched_h && n_shots_h > 1) {
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_buoyancy_z, grad_buoyancy_z_shot);
    CHECK_KERNEL_ERROR
#endif
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_buoyancy_y, grad_buoyancy_y_shot);
    CHECK_KERNEL_ERROR
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_buoyancy_x, grad_buoyancy_x_shot);
    CHECK_KERNEL_ERROR
  }

  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

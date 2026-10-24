/*
 * Elastic Born (linearised) wave equation propagator -- compiled forward and
 * adjoint passes (CUDA implementation, kept in lockstep with elastic_born.c).
 *
 * Propagates a background elastic velocity-stress field and a scattered field
 * together, reproducing elastic.cu's forward physics for the background field
 * and driving the scattered field with the derivation SS2.5 virtual sources
 * (see elastic_born.c for the full description). Implements all three entry
 * points (forward, backward_sc, backward) with the same engine signature as
 * their C twins (start_t/step_ratio + storage-slot + storage-mode params),
 * including first-class `boundary=` guard-fill support (bc[2*ndim]) on all
 * three, executed and diffed against CPU/eager on a real GPU.
 */

#include <stdio.h>

#include <algorithm>
#include <cstdint>

#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  elastic_born_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#ifdef DW_MPML
#define ELASTIC_VARIANT_I(name) name##_mpml
#define ELASTIC_VARIANT(name) ELASTIC_VARIANT_I(name)
#else
#define ELASTIC_VARIANT(name) name
#endif
#define FUNC(name) \
  CAT(ELASTIC_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx_h
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* PML-region test wrapper -- kept in lockstep with elastic.cu. */
#ifdef DW_MPML
#if DW_NDIM == 3
#define DW_PML_TEST(cond)                                      \
  (z < pml_z0 || z >= MAX(pml_z0, pml_z1 - 1) || y < pml_y0 || \
   y >= MAX(pml_y0, pml_y1 - 1) || x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1))
#elif DW_NDIM == 2
#define DW_PML_TEST(cond)                                      \
  (y < pml_y0 || y >= MAX(pml_y0, pml_y1 - 1) || x < pml_x0 || \
   x >= MAX(pml_x0, pml_x1 - 1))
#else
#define DW_PML_TEST(cond) (x < pml_x0 || x >= MAX(pml_x0, pml_x1 - 1))
#endif
#else
#define DW_PML_TEST(cond) (cond)
#endif

/* ND_INDEX, the P_A(*)/P_B(*) PML-profile accessors, the VZ/VY/VX/SIGMA(*)/
 * their _sc and _bg twins, LAMB/MU/MU_(*)/BUOYANCY_(*)/DLAMB/DMU/DMU_(*)/
 * DBUOYANCY_(*), the M_SIGMA** / M_V** (and _BG) PML memory families,
 * LAMB_2MU/DLAMB_2MU, and the FUNC(forward_vz_cell)/.../
 * FUNC(backward_sigmayz_cell) per-grid-point stencil update functions (21
 * total) live in elastic_born_kernels.h, shared verbatim with
 * elastic_born.c (see that header for the full explanation). DW_PML_TEST
 * above is CUDA-driver-side scaffolding (not per-grid-point math), used to
 * compute the plain `pml_z`/`pml_y`/`pml_x` bools those functions expect. */
#include "elastic_born_kernels.h"

namespace {

/* --- boundary=["free"/"rigid"/"periodic",...] guard-ring image fill ------
 *
 * CUDA mirror of elastic_born.c's own fill functions (see that file's
 * matching comment for the full derivation/citations, and elastic.cu's own
 * fill_p_bc_line_dev/fill_elastic_*_dev/fill_elastic_bc_kernel for the
 * precedent this is a direct, unchanged-arithmetic port of -- one generic
 * per-line ``__device__`` kernel dispatches (by a ``kind`` parameter) among
 * the four line-fill arithmetic variants, and a host-side ``launch_*`` wrapper
 * computes the per-axis line count and launches one kernel per (field,
 * face)). ``kind``: 0 = on-row stress (sigma_AA), 1 = on-row velocity, 2 =
 * off-row velocity, 3 = shared-axis shear stress. The launch wrappers and
 * LAUNCH_FILL_ELASTIC_*_BC macros below take explicit pointer arguments
 * (unlike elastic.cu's own zero-arg macros) so the same launch call can be
 * applied independently to the background and scattered field pointers at
 * each call site (see elastic_born.c's own bg/sc-doubled call sites for the
 * analogous CPU pattern). */

__device__ inline void fill_p_bc_line_dev(DW_DTYPE* __restrict const p,
                                          int64_t const base, int64_t const nax,
                                          int64_t const stride,
                                          int64_t const bc_lo,
                                          int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_lo == 1) p[base + P * stride] = 0;
  if (bc_hi == 1) p[base + (nax - P) * stride] = 0;
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_lo == 3) {
        src = m - 1 + g;
      } else if (bc_lo == 2) {
        src = 2 * P - 1 - g;
      } else {
        src = 2 * P - g;
        sign = -1;
      }
      p[base + g * stride] = sign * p[base + src * stride];
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j;
      } else if (bc_hi == 2) {
        src = P + m - 1 - j;
      } else {
        src = P + m - 2 - j;
        sign = -1;
      }
      p[base + xh * stride] = sign * p[base + src * stride];
    }
  }
}

/* fill_elastic_onrow_velocity_line_dev/fill_elastic_offrow_velocity_line_
 * dev/fill_elastic_shear_line_dev live in elastic_line_bc_dev.h (see that
 * header). */
#include "elastic_line_bc_dev.h"

/* One thread per line along the chosen axis, matching elastic.cu's own
 * fill_elastic_bc_kernel exactly. ``kind`` selects which of the four line
 * arithmetic variants above to apply. */
__global__ void fill_elastic_born_bc_kernel(
    DW_DTYPE* __restrict const wf, int64_t const n_lines, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi,
    int const kind) {
  int64_t const L = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  if (kind == 0)
    fill_p_bc_line_dev(wf, base, nax, stride, bc_lo, bc_hi);
  else if (kind == 1)
    fill_elastic_onrow_velocity_line_dev(wf, base, nax, stride, bc_lo, bc_hi);
  else if (kind == 2)
    fill_elastic_offrow_velocity_line_dev(wf, base, nax, stride, bc_lo, bc_hi);
  else
    fill_elastic_shear_line_dev(wf, base, nax, stride, bc_lo, bc_hi);
}

/* Stress guard fill on every reflecting axis for ONE field (background or
 * scattered, whichever pointers the caller passes): sigma_AA (kind 0) plus
 * the two shear stresses sharing axis A (kind 3), mirroring elastic_born.c's
 * FUNC(fill_elastic_stress_bc) but as CUDA kernel launches. */
static void launch_fill_elastic_born_stress_bc(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const sigmaxx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxx, nl, nx_, 1,
                                                          bc[4], bc[5], 0);
#if DW_NDIM >= 3
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxz, nl, nx_, 1,
                                                          bc[4], bc[5], 3);
#endif
#if DW_NDIM >= 2
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxy, nl, nx_, 1,
                                                          bc[4], bc[5], 3);
#endif
  }
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(sigmayy, nl, ny_, nx_,
                                                          bc[2], bc[3], 0);
#if DW_NDIM >= 3
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(sigmayz, nl, ny_, nx_,
                                                          bc[2], bc[3], 3);
#endif
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxy, nl, ny_, nx_,
                                                          bc[2], bc[3], 3);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(
        sigmazz, nl, nz_, ny_ * nx_, bc[0], bc[1], 0);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(
        sigmayz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
  }
#endif
}

/* Velocity guard fill on every reflecting axis for ONE field: v_A off-row
 * (kind 2), the other two velocities on-row (kind 1), mirroring
 * elastic_born.c's FUNC(fill_elastic_velocity_bc). */
static void launch_fill_elastic_born_velocity_bc(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(vx, nl, nx_, 1, bc[4],
                                                          bc[5], 2);
#if DW_NDIM >= 3
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(vz, nl, nx_, 1, bc[4],
                                                          bc[5], 1);
#endif
#if DW_NDIM >= 2
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(vy, nl, nx_, 1, bc[4],
                                                          bc[5], 1);
#endif
  }
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(vy, nl, ny_, nx_,
                                                          bc[2], bc[3], 2);
#if DW_NDIM >= 3
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(vz, nl, ny_, nx_,
                                                          bc[2], bc[3], 1);
#endif
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(vx, nl, ny_, nx_,
                                                          bc[2], bc[3], 1);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, bc[0], bc[1], 2);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(
        vy, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
    fill_elastic_born_bc_kernel<<<grid, tpb, 0, stream>>>(
        vx, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
  }
#endif
}

#if DW_NDIM == 3
#define LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(szz, syz, sxz, syy, sxy, sxx)  \
  launch_fill_elastic_born_stress_bc(szz, syz, sxz, syy, sxy, sxx, bc,    \
                                     n_shots_h* shot_numel_h, nz_h, ny_h, \
                                     nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz_, vy_, vx_)                 \
  launch_fill_elastic_born_velocity_bc(vz_, vy_, vx_, bc,                   \
                                       n_shots_h* shot_numel_h, nz_h, ny_h, \
                                       nx_h, stream_compute)
#elif DW_NDIM == 2
#define LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(szz, syz, sxz, syy, sxy, sxx)     \
  launch_fill_elastic_born_stress_bc(syy, sxy, sxx, bc,                      \
                                     n_shots_h* shot_numel_h, 1, ny_h, nx_h, \
                                     stream_compute)
#define LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz_, vy_, vx_)                   \
  launch_fill_elastic_born_velocity_bc(vy_, vx_, bc, n_shots_h* shot_numel_h, \
                                       1, ny_h, nx_h, stream_compute)
#else
#define LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(szz, syz, sxz, syy, sxy, sxx)     \
  launch_fill_elastic_born_stress_bc(sxx, bc, n_shots_h* shot_numel_h, 1, 1, \
                                     nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz_, vy_, vx_)                    \
  launch_fill_elastic_born_velocity_bc(vx_, bc, n_shots_h* shot_numel_h, 1, 1, \
                                       nx_h, stream_compute)
#endif

/* --- Adjoint (transpose) of the boundary-face guard-ring fill ------------
 *
 * CUDA mirror of elastic_born.c's own FUNC(fill_*_adjoint) functions (see
 * that file's matching comment, above FUNC(fill_p_bc_line_adjoint), for the
 * full transpose derivation) -- same per-line arithmetic, same "kind"
 * dispatch convention as fill_elastic_born_bc_kernel above, ported with
 * unchanged arithmetic to CUDA. */

/* fill_p_bc_line_adjoint_dev/fill_elastic_onrow_velocity_line_adjoint_dev/
 * fill_elastic_offrow_velocity_line_adjoint_dev/fill_elastic_shear_line_
 * adjoint_dev live in elastic_line_bc_dev.h, already included above. */

/* One thread per line, dispatched by ``kind`` exactly as fill_elastic_born_
 * bc_kernel above; the difference is entirely in the four line arithmetic
 * variants (their transposes) called. */
__global__ void fill_elastic_born_bc_adjoint_kernel(
    DW_DTYPE* __restrict const wf, int64_t const n_lines, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi,
    int const kind) {
  int64_t const L = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  if (kind == 0)
    fill_p_bc_line_adjoint_dev(wf, base, nax, stride, bc_lo, bc_hi);
  else if (kind == 1)
    fill_elastic_onrow_velocity_line_adjoint_dev(wf, base, nax, stride, bc_lo,
                                                 bc_hi);
  else if (kind == 2)
    fill_elastic_offrow_velocity_line_adjoint_dev(wf, base, nax, stride, bc_lo,
                                                  bc_hi);
  else
    fill_elastic_shear_line_adjoint_dev(wf, base, nax, stride, bc_lo, bc_hi);
}

/* Transpose of launch_fill_elastic_born_stress_bc: faces launched in the
 * exact reverse of that function's own face order (Z, then Y, then X --
 * elastic_born.c's FUNC(fill_elastic_stress_bc_adjoint) module comment
 * explains why; CUDA's same-stream in-order execution guarantee makes this
 * host-side launch order the actual execution order on device). */
static void launch_fill_elastic_born_stress_bc_adjoint(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const sigmaxx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmazz, nl, nz_, ny_ * nx_, bc[0], bc[1], 0);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmayz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
  }
#endif
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmayy, nl, ny_, nx_, bc[2], bc[3], 0);
#if DW_NDIM >= 3
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmayz, nl, ny_, nx_, bc[2], bc[3], 3);
#endif
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxy, nl, ny_, nx_, bc[2], bc[3], 3);
  }
#endif
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxx, nl, nx_, 1, bc[4], bc[5], 0);
#if DW_NDIM >= 3
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxz, nl, nx_, 1, bc[4], bc[5], 3);
#endif
#if DW_NDIM >= 2
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxy, nl, nx_, 1, bc[4], bc[5], 3);
#endif
  }
}

/* Transpose of launch_fill_elastic_born_velocity_bc: faces launched in the
 * exact reverse of that function's own face order (see the comment above
 * launch_fill_elastic_born_stress_bc_adjoint). */
static void launch_fill_elastic_born_velocity_bc_adjoint(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, bc[0], bc[1], 2);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vy, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vx, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
  }
#endif
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vy, nl, ny_, nx_, bc[2], bc[3], 2);
#if DW_NDIM >= 3
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vz, nl, ny_, nx_, bc[2], bc[3], 1);
#endif
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vx, nl, ny_, nx_, bc[2], bc[3], 1);
  }
#endif
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div(nl, tpb);
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vx, nl, nx_, 1, bc[4], bc[5], 2);
#if DW_NDIM >= 3
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vz, nl, nx_, 1, bc[4], bc[5], 1);
#endif
#if DW_NDIM >= 2
    fill_elastic_born_bc_adjoint_kernel<<<grid, tpb, 0, stream>>>(
        vy, nl, nx_, 1, bc[4], bc[5], 1);
#endif
  }
}

#if DW_NDIM == 3
#define LAUNCH_FILL_ELASTIC_BORN_STRESS_BC_ADJOINT(szz, syz, sxz, syy, sxy,    \
                                                   sxx)                        \
  launch_fill_elastic_born_stress_bc_adjoint(szz, syz, sxz, syy, sxy, sxx, bc, \
                                             n_shots_h* shot_numel_h, nz_h,    \
                                             ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC_ADJOINT(vz_, vy_, vx_)           \
  launch_fill_elastic_born_velocity_bc_adjoint(vz_, vy_, vx_, bc,             \
                                               n_shots_h* shot_numel_h, nz_h, \
                                               ny_h, nx_h, stream_compute)
#elif DW_NDIM == 2
#define LAUNCH_FILL_ELASTIC_BORN_STRESS_BC_ADJOINT(szz, syz, sxz, syy, sxy,    \
                                                   sxx)                        \
  launch_fill_elastic_born_stress_bc_adjoint(syy, sxy, sxx, bc,                \
                                             n_shots_h* shot_numel_h, 1, ny_h, \
                                             nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC_ADJOINT(vz_, vy_, vx_) \
  launch_fill_elastic_born_velocity_bc_adjoint(                     \
      vy_, vx_, bc, n_shots_h* shot_numel_h, 1, ny_h, nx_h, stream_compute)
#else
#define LAUNCH_FILL_ELASTIC_BORN_STRESS_BC_ADJOINT(szz, syz, sxz, syy, sxy,    \
                                                   sxx)                        \
  launch_fill_elastic_born_stress_bc_adjoint(sxx, bc, n_shots_h* shot_numel_h, \
                                             1, 1, nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC_ADJOINT(vz_, vy_, vx_) \
  launch_fill_elastic_born_velocity_bc_adjoint(                     \
      vx_, bc, n_shots_h* shot_numel_h, 1, 1, nx_h, stream_compute)
#endif

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
__constant__ bool dlamb_batched;
__constant__ bool dmu_batched;
__constant__ bool dbuoyancy_batched;

/* --- source injection / receiver recording (count passed as an argument so
 * the background and scattered blocks share one kernel) --- */

__launch_bounds__(32) __global__
    void add_sources(DW_DTYPE* __restrict const wf,
                     DW_DTYPE const* __restrict const f,
                     int64_t const* __restrict const sources_i,
                     int64_t const count) {
  int64_t const source_idx = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (source_idx < count && shot_idx < n_shots) {
    int64_t const k = shot_idx * count + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__ void add_sources_p(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy,
#endif
    DW_DTYPE* __restrict const sigmaxx, DW_DTYPE const* __restrict const f,
    int64_t const* __restrict const sources_i, int64_t const count) {
  int64_t const source_idx = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (source_idx < count && shot_idx < n_shots) {
    int64_t const k = shot_idx * count + source_idx;
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

__launch_bounds__(32) __global__
    void record_receivers(DW_DTYPE* __restrict const r,
                          DW_DTYPE const* __restrict const wf,
                          int64_t const* __restrict receivers_i,
                          int64_t const count) {
  int64_t const receiver_idx =
      dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (receiver_idx < count && shot_idx < n_shots) {
    int64_t const k = shot_idx * count + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
  }
}

__launch_bounds__(32) __global__
    void record_pressure_receivers(DW_DTYPE* __restrict const r,
#if DW_NDIM >= 3
                                   DW_DTYPE const* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
                                   DW_DTYPE const* __restrict const sigmayy,
#endif
                                   DW_DTYPE const* __restrict const sigmaxx,
                                   int64_t const* __restrict receivers_i,
                                   int64_t const count) {
  int64_t const receiver_idx =
      dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (receiver_idx < count && shot_idx < n_shots) {
    int64_t const k = shot_idx * count + receiver_idx;
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

/* --- coupled background + scattered velocity update --- */
__launch_bounds__(128) __global__ void forward_kernel_v(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vy_sc,
#endif
    DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const vx_sc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmazz_sc,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmayz_sc,
    DW_DTYPE const* __restrict const sigmaxz,
    DW_DTYPE const* __restrict const sigmaxz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmayy_sc,
    DW_DTYPE const* __restrict const sigmaxy,
    DW_DTYPE const* __restrict const sigmaxy_sc,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmaxx_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmazzz_sc,
    DW_DTYPE* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmayzy_sc,
    DW_DTYPE* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const m_sigmaxzx_sc,
    DW_DTYPE* __restrict const m_sigmayzz,
    DW_DTYPE* __restrict const m_sigmayzz_sc,
    DW_DTYPE* __restrict const m_sigmaxzz,
    DW_DTYPE* __restrict const m_sigmaxzz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmayyy_sc,
    DW_DTYPE* __restrict const m_sigmaxyy,
    DW_DTYPE* __restrict const m_sigmaxyy_sc,
    DW_DTYPE* __restrict const m_sigmaxyx,
    DW_DTYPE* __restrict const m_sigmaxyx_sc,
#endif
    DW_DTYPE* __restrict const m_sigmaxxx,
    DW_DTYPE* __restrict const m_sigmaxxx_sc,
/* per-quantity gradient stores (background then scattered): the
 * strain/stress-divergence terms elastic.cu stores for the buoyancy
 * gradients, doubled for the coupled Born field. Written only when the
 * corresponding store flag is set (t is a snapshot step and a model needs
 * grad); read back by the adjoint kernels via the StorageManager. */
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const dvzdbuoyancy_store,
    DW_DTYPE* __restrict const dvzdbuoyancy_sc_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const dvydbuoyancy_store,
    DW_DTYPE* __restrict const dvydbuoyancy_sc_store,
#endif
    DW_DTYPE* __restrict const dvxdbuoyancy_store,
    DW_DTYPE* __restrict const dvxdbuoyancy_sc_store,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    bool const store_bg, bool const store_sc) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
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

#ifndef DW_MPML
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
#endif /* !DW_MPML */

      int64_t const i = shot_idx * shot_numel + j;
#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const buoyancy_z_shot =
            buoyancy_batched ? buoyancy_z[i] : buoyancy_z[j];
        DW_DTYPE const dbuoyancy_z_shot =
            dbuoyancy_batched ? dbuoyancy_z[i] : dbuoyancy_z[j];
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

        FUNC(forward_vz_cell)(
            sigmazz, sigmayz, sigmaxz, sigmazz_sc, sigmayz_sc, sigmaxz_sc, vz,
            vz_sc, m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmazzz_sc,
            m_sigmayzy_sc, m_sigmaxzx_sc, dvzdbuoyancy_store,
            dvzdbuoyancy_sc_store, buoyancy_z_shot, dbuoyancy_z_shot, azh, bzh,
            ay, by, ax, bx, dt, store_bg, store_sc, i, ny, nx, z, y, x, pml_z,
            pml_y, pml_x, rdz, rdy, rdx);
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const buoyancy_y_shot =
            buoyancy_batched ? buoyancy_y[i] : buoyancy_y[j];
        DW_DTYPE const dbuoyancy_y_shot =
            dbuoyancy_batched ? dbuoyancy_y[i] : dbuoyancy_y[j];
#if DW_NDIM == 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

        FUNC(forward_vy_cell)(
#if DW_NDIM >= 3
            sigmayz, sigmayz_sc,
#endif
            sigmayy, sigmaxy, sigmayy_sc, sigmaxy_sc, vy, vy_sc,
#if DW_NDIM >= 3
            m_sigmayzz, m_sigmayzz_sc,
#endif
            m_sigmayyy, m_sigmaxyx, m_sigmayyy_sc, m_sigmaxyx_sc,
            dvydbuoyancy_store, dvydbuoyancy_sc_store, buoyancy_y_shot,
            dbuoyancy_y_shot,
#if DW_NDIM >= 3
            az, bz,
#endif
            ayh, byh, ax, bx, dt, store_bg, store_sc, i,
#if DW_NDIM >= 3
            ny,
#endif
            nx,
#if DW_NDIM >= 3
            z, pml_z, rdz,
#endif
            y, x, pml_y, pml_x, rdy, rdx);
      }
#endif
      if (x < nx - FD_PAD) {
        DW_DTYPE const buoyancy_x_shot =
            buoyancy_batched ? buoyancy_x[i] : buoyancy_x[j];
        DW_DTYPE const dbuoyancy_x_shot =
            dbuoyancy_batched ? dbuoyancy_x[i] : dbuoyancy_x[j];
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);

        FUNC(forward_vx_cell)(
#if DW_NDIM >= 3
            sigmaxz, sigmaxz_sc,
#endif
#if DW_NDIM >= 2
            sigmaxy, sigmaxy_sc,
#endif
            sigmaxx, sigmaxx_sc, vx, vx_sc,
#if DW_NDIM >= 3
            m_sigmaxzz, m_sigmaxzz_sc,
#endif
#if DW_NDIM >= 2
            m_sigmaxyy, m_sigmaxyy_sc,
#endif
            m_sigmaxxx, m_sigmaxxx_sc, dvxdbuoyancy_store,
            dvxdbuoyancy_sc_store, buoyancy_x_shot, dbuoyancy_x_shot,
#if DW_NDIM >= 3
            az, bz,
#endif
#if DW_NDIM >= 2
            ay, by,
#endif
            axh, bxh, dt, store_bg, store_sc, i,
#if DW_NDIM >= 3
            ny,
#endif
#if DW_NDIM >= 2
            nx,
#endif
#if DW_NDIM >= 3
            z, pml_z, rdz,
#endif
#if DW_NDIM >= 2
            y, pml_y, rdy,
#endif
            x, pml_x, rdx);
      }
#if DW_NDIM == 3
    }
#endif
  }
}

/* --- coupled background + scattered stress update --- */
__launch_bounds__(128) __global__ void forward_kernel_sigma(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
    DW_DTYPE const* __restrict const dlamb,
    DW_DTYPE const* __restrict const dmu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const dmu_zy,
    DW_DTYPE const* __restrict const mu_zx,
    DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
    DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE const* __restrict const vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vy_sc,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vx_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmazz_sc,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmayz_sc,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const sigmaxz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmayy_sc,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const sigmaxy_sc,
#endif
    DW_DTYPE* __restrict const sigmaxx, DW_DTYPE* __restrict const sigmaxx_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzz_sc,
    DW_DTYPE* __restrict const m_vzy, DW_DTYPE* __restrict const m_vzy_sc,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vzx_sc,
    DW_DTYPE* __restrict const m_vyz, DW_DTYPE* __restrict const m_vyz_sc,
    DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict const m_vxz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vyy_sc,
    DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vyx_sc,
    DW_DTYPE* __restrict const m_vxy, DW_DTYPE* __restrict const m_vxy_sc,
#endif
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const m_vxx_sc,
/* per-quantity gradient stores (background then scattered): the normal
 * strains and shear-strain sums elastic.cu stores for the lamb/mu
 * gradients, doubled for the coupled Born field. Written only when the
 * corresponding store flag is set. */
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const dvzdz_store,
    DW_DTYPE* __restrict const dvzdz_sc_store,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_sc_store,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_sc_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const dvydy_store,
    DW_DTYPE* __restrict const dvydy_sc_store,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_sc_store,
#endif
    DW_DTYPE* __restrict const dvxdx_store,
    DW_DTYPE* __restrict const dvxdx_sc_store,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    bool const store_bg, bool const store_sc) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
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
#ifndef DW_MPML
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
#endif /* !DW_MPML */

      int64_t const i = shot_idx * shot_numel + j;
      DW_DTYPE const lamb_shot_i = lamb_batched ? lamb[i] : lamb[j];
      DW_DTYPE const mu_shot_i = mu_batched ? mu[i] : mu[j];
      DW_DTYPE const dlamb_shot_i = dlamb_batched ? dlamb[i] : dlamb[j];
      DW_DTYPE const dmu_shot_i = dmu_batched ? dmu[i] : dmu[j];

#if DW_NDIM >= 3
      bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
      bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
      bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

      FUNC(forward_sigmaii_cell)(
#if DW_NDIM >= 3
          vz, vz_sc, sigmazz, sigmazz_sc, m_vzz, m_vzz_sc, dvzdz_store,
          dvzdz_sc_store,
#endif
#if DW_NDIM >= 2
          vy, vy_sc, sigmayy, sigmayy_sc, m_vyy, m_vyy_sc, dvydy_store,
          dvydy_sc_store,
#endif
          vx, vx_sc, sigmaxx, sigmaxx_sc, m_vxx, m_vxx_sc, dvxdx_store,
          dvxdx_sc_store, lamb_shot_i, mu_shot_i, dlamb_shot_i, dmu_shot_i, dt,
          store_bg, store_sc, i,
#if DW_NDIM >= 3
          ny,
#endif
#if DW_NDIM >= 2
          nx,
#endif
#if DW_NDIM >= 3
          z, pml_z, az, bz, rdz,
#endif
#if DW_NDIM >= 2
          y, pml_y, ay, by, rdy,
#endif
          x, pml_x, ax, bx, rdx);

#if DW_NDIM >= 2
      if (y < ny - FD_PAD && x < nx - FD_PAD) {
        bool const pml_yh = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_xh = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        DW_DTYPE const mu_yx_shot_i = mu_batched ? mu_yx[i] : mu_yx[j];
        DW_DTYPE const dmu_yx_shot_i = dmu_batched ? dmu_yx[i] : dmu_yx[j];
        FUNC(forward_sigmaxy_cell)(
            vx, vy, vx_sc, vy_sc, sigmaxy, sigmaxy_sc, m_vxy, m_vyx, m_vxy_sc,
            m_vyx_sc, dvydx_plus_dvxdy_store, dvydx_plus_dvxdy_sc_store,
            mu_yx_shot_i, dmu_yx_shot_i, dt, store_bg, store_sc, i,
#if DW_NDIM >= 3
            ny, z,
#endif
            nx, y, x, pml_yh, pml_xh, ayh, byh, axh, bxh, rdy, rdx);
      }
#endif
#if DW_NDIM >= 3
      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        bool const pml_zh = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_xh = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        DW_DTYPE const mu_zx_shot_i = mu_batched ? mu_zx[i] : mu_zx[j];
        DW_DTYPE const dmu_zx_shot_i = dmu_batched ? dmu_zx[i] : dmu_zx[j];
        FUNC(forward_sigmaxz_cell)(
            vx, vz, vx_sc, vz_sc, sigmaxz, sigmaxz_sc, m_vxz, m_vzx, m_vxz_sc,
            m_vzx_sc, dvzdx_plus_dvxdz_store, dvzdx_plus_dvxdz_sc_store,
            mu_zx_shot_i, dmu_zx_shot_i, dt, store_bg, store_sc, i, ny, nx, z,
            y, x, pml_zh, pml_xh, azh, bzh, axh, bxh, rdz, rdx);
      }
      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_zh = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_yh = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        DW_DTYPE const mu_zy_shot_i = mu_batched ? mu_zy[i] : mu_zy[j];
        DW_DTYPE const dmu_zy_shot_i = dmu_batched ? dmu_zy[i] : dmu_zy[j];
        FUNC(forward_sigmayz_cell)(
            vy, vz, vy_sc, vz_sc, sigmayz, sigmayz_sc, m_vyz, m_vzy, m_vyz_sc,
            m_vzy_sc, dvzdy_plus_dvydz_store, dvzdy_plus_dvydz_sc_store,
            mu_zy_shot_i, dmu_zy_shot_i, dt, store_bg, store_sc, i, ny, nx, z,
            y, x, pml_zh, pml_yh, azh, bzh, ayh, byh, rdz, rdy);
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
               int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
               int64_t const pml_y0_h, int64_t const pml_y1_h,
#endif
               int64_t const pml_x0_h, int64_t const pml_x1_h,
               bool const lamb_batched_h, bool const mu_batched_h,
               bool const buoyancy_batched_h, bool const dlamb_batched_h,
               bool const dmu_batched_h, bool const dbuoyancy_batched_h) {
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
  gpuErrchk(cudaMemcpyToSymbol(dlamb_batched, &dlamb_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(dmu_batched, &dmu_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(dbuoyancy_batched, &dbuoyancy_batched_h,
                               sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int FUNC(forward)(
        /* background models */
        DW_DTYPE const* __restrict const lamb,
        DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const mu_zy,
        DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const buoyancy_y,
#endif
        DW_DTYPE const* __restrict const buoyancy_x,
        /* scattering models */
        DW_DTYPE const* __restrict const dlamb,
        DW_DTYPE const* __restrict const dmu,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const dmu_zy,
        DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
        DW_DTYPE const* __restrict const dbuoyancy_x,
/* background sources */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const f_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const f_y,
#endif
        DW_DTYPE const* __restrict const f_x,
        DW_DTYPE const* __restrict const f_p,
/* scattered sources */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const f_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const f_y_sc,
#endif
        DW_DTYPE const* __restrict const f_x_sc,
        DW_DTYPE const* __restrict const f_p_sc,
/* background wavefields */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
        DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmaxz,
        DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
        DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
        DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict const m_sigmazzz,
        DW_DTYPE* __restrict const m_sigmayzy,
        DW_DTYPE* __restrict const m_sigmaxzx,
        DW_DTYPE* __restrict const m_sigmayzz,
        DW_DTYPE* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
        DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vyy,
        DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vxy,
        DW_DTYPE* __restrict const m_sigmayyy,
        DW_DTYPE* __restrict const m_sigmaxyy,
        DW_DTYPE* __restrict const m_sigmaxyx,
#endif
        DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
        DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const m_sigmaxxx,
/* scattered wavefields */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const vz_sc, DW_DTYPE* __restrict const sigmazz_sc,
        DW_DTYPE* __restrict const sigmayz_sc,
        DW_DTYPE* __restrict const sigmaxz_sc,
        DW_DTYPE* __restrict const m_vzz_sc,
        DW_DTYPE* __restrict const m_vzy_sc,
        DW_DTYPE* __restrict const m_vzx_sc,
        DW_DTYPE* __restrict const m_vyz_sc,
        DW_DTYPE* __restrict const m_vxz_sc,
        DW_DTYPE* __restrict const m_sigmazzz_sc,
        DW_DTYPE* __restrict const m_sigmayzy_sc,
        DW_DTYPE* __restrict const m_sigmaxzx_sc,
        DW_DTYPE* __restrict const m_sigmayzz_sc,
        DW_DTYPE* __restrict const m_sigmaxzz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const vy_sc, DW_DTYPE* __restrict const sigmayy_sc,
        DW_DTYPE* __restrict const sigmaxy_sc,
        DW_DTYPE* __restrict const m_vyy_sc,
        DW_DTYPE* __restrict const m_vyx_sc,
        DW_DTYPE* __restrict const m_vxy_sc,
        DW_DTYPE* __restrict const m_sigmayyy_sc,
        DW_DTYPE* __restrict const m_sigmaxyy_sc,
        DW_DTYPE* __restrict const m_sigmaxyx_sc,
#endif
        DW_DTYPE* __restrict const vx_sc, DW_DTYPE* __restrict const sigmaxx_sc,
        DW_DTYPE* __restrict const m_vxx_sc,
        DW_DTYPE* __restrict const m_sigmaxxx_sc,
/* background receivers */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const r_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const r_y,
#endif
        DW_DTYPE* __restrict const r_x, DW_DTYPE* __restrict const r_p,
/* scattered receivers */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const r_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const r_y_sc,
#endif
        DW_DTYPE* __restrict const r_x_sc, DW_DTYPE* __restrict const r_p_sc,
/* background gradient stores: engine-style per-quantity
 * 5-param slot groups, mirroring the CPU forward's signature
 * exactly (elastic_born.c) so the CPU/CUDA ABI stays
 * identical (test_cpu_cuda_argtypes_identical). GPU store
 * WRITE implementation is deferred to task N alongside the
 * GPU backward kernel -- every one of these parameters is
 * accepted and ignored ((void)-cast below) on this path;
 * store_bg/store_sc are always false on the currently-wired
 * GPU path. */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
        DW_DTYPE* __restrict const dvzdbuoyancy_store_1b,
        void* __restrict const dvzdbuoyancy_store_2,
        void* __restrict const dvzdbuoyancy_store_3,
        char const* __restrict const* __restrict const dvzdbuoyancy_filenames_ptr,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdz_store_1,
        DW_DTYPE* __restrict const dvzdz_store_1b,
        void* __restrict const dvzdz_store_2,
        void* __restrict const dvzdz_store_3,
        char const* __restrict const* __restrict const dvzdz_filenames_ptr,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1b,
        void* __restrict const dvzdx_plus_dvxdz_store_2,
        void* __restrict const dvzdx_plus_dvxdz_store_3,
        char const* __restrict const* __restrict const dvzdx_plus_dvxdz_filenames_ptr,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1b,
        void* __restrict const dvzdy_plus_dvydz_store_2,
        void* __restrict const dvzdy_plus_dvydz_store_3,
        char const* __restrict const* __restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydbuoyancy_store_1,
        DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
        void* __restrict const dvydbuoyancy_store_2,
        void* __restrict const dvydbuoyancy_store_3,
        char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydy_store_1,
        DW_DTYPE* __restrict const dvydy_store_1b,
        void* __restrict const dvydy_store_2,
        void* __restrict const dvydy_store_3,
        char const* __restrict const* __restrict const dvydy_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1b,
        void* __restrict const dvydx_plus_dvxdy_store_2,
        void* __restrict const dvydx_plus_dvxdy_store_3,
        char const* __restrict const* __restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
        DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
        DW_DTYPE* __restrict const dvxdbuoyancy_store_1b,
        void* __restrict const dvxdbuoyancy_store_2,
        void* __restrict const dvxdbuoyancy_store_3,
        char const* __restrict const* __restrict const dvxdbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvxdx_store_1,
        DW_DTYPE* __restrict const dvxdx_store_1b,
        void* __restrict const dvxdx_store_2,
        void* __restrict const dvxdx_store_3,
        char const* __restrict const* __restrict const dvxdx_filenames_ptr,
/* scattered gradient stores: same per-quantity 5-param slot
 * groups as above, accepted and ignored for the same reason.
 * */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdbuoyancy_sc_store_1,
        DW_DTYPE* __restrict const dvzdbuoyancy_sc_store_1b,
        void* __restrict const dvzdbuoyancy_sc_store_2,
        void* __restrict const dvzdbuoyancy_sc_store_3,
        char const* __restrict const* __restrict const dvzdbuoyancy_sc_filenames_ptr,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdz_sc_store_1,
        DW_DTYPE* __restrict const dvzdz_sc_store_1b,
        void* __restrict const dvzdz_sc_store_2,
        void* __restrict const dvzdz_sc_store_3,
        char const* __restrict const* __restrict const dvzdz_sc_filenames_ptr,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_sc_store_1,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_sc_store_1b,
        void* __restrict const dvzdx_plus_dvxdz_sc_store_2,
        void* __restrict const dvzdx_plus_dvxdz_sc_store_3,
        char const* __restrict const* __restrict const dvzdx_plus_dvxdz_sc_filenames_ptr,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_sc_store_1,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_sc_store_1b,
        void* __restrict const dvzdy_plus_dvydz_sc_store_2,
        void* __restrict const dvzdy_plus_dvydz_sc_store_3,
        char const* __restrict const* __restrict const dvzdy_plus_dvydz_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydbuoyancy_sc_store_1,
        DW_DTYPE* __restrict const dvydbuoyancy_sc_store_1b,
        void* __restrict const dvydbuoyancy_sc_store_2,
        void* __restrict const dvydbuoyancy_sc_store_3,
        char const* __restrict const* __restrict const dvydbuoyancy_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydy_sc_store_1,
        DW_DTYPE* __restrict const dvydy_sc_store_1b,
        void* __restrict const dvydy_sc_store_2,
        void* __restrict const dvydy_sc_store_3,
        char const* __restrict const* __restrict const dvydy_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_sc_store_1,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_sc_store_1b,
        void* __restrict const dvydx_plus_dvxdy_sc_store_2,
        void* __restrict const dvydx_plus_dvxdy_sc_store_3,
        char const* __restrict const* __restrict const dvydx_plus_dvxdy_sc_filenames_ptr,
#endif
        DW_DTYPE* __restrict const dvxdbuoyancy_sc_store_1,
        DW_DTYPE* __restrict const dvxdbuoyancy_sc_store_1b,
        void* __restrict const dvxdbuoyancy_sc_store_2,
        void* __restrict const dvxdbuoyancy_sc_store_3,
        char const* __restrict const* __restrict const dvxdbuoyancy_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvxdx_sc_store_1,
        DW_DTYPE* __restrict const dvxdx_sc_store_1b,
        void* __restrict const dvxdx_sc_store_2,
        void* __restrict const dvxdx_sc_store_3,
        char const* __restrict const* __restrict const dvxdx_sc_filenames_ptr,

/* PML profiles */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const az,
        DW_DTYPE const* __restrict const bz,
        DW_DTYPE const* __restrict const azh,
        DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const ay,
        DW_DTYPE const* __restrict const by,
        DW_DTYPE const* __restrict const ayh,
        DW_DTYPE const* __restrict const byh,
#endif
        DW_DTYPE const* __restrict const ax,
        DW_DTYPE const* __restrict const bx,
        DW_DTYPE const* __restrict const axh,
        DW_DTYPE const* __restrict const bxh,
/* shared source indices */
#if DW_NDIM >= 3
        int64_t const* __restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const sources_y_i,
#endif
        int64_t const* __restrict const sources_x_i,
        int64_t const* __restrict const sources_p_i,
/* background receiver indices */
#if DW_NDIM >= 3
        int64_t const* __restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const receivers_y_i,
#endif
        int64_t const* __restrict const receivers_x_i,
        int64_t const* __restrict const receivers_p_i,
/* scattered receiver indices */
#if DW_NDIM >= 3
        int64_t const* __restrict const receivers_z_sc_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const receivers_y_sc_i,
#endif
        int64_t const* __restrict const receivers_x_sc_i,
        int64_t const* __restrict const receivers_p_sc_i,
/* scalars */
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
/* shared source counts */
#if DW_NDIM >= 3
        int64_t const n_sources_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_per_shot,
#endif
        int64_t const n_sources_x_per_shot, int64_t const n_sources_p_per_shot,
/* scattered source counts */
#if DW_NDIM >= 3
        int64_t const n_sources_z_sc_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_sc_per_shot,
#endif
        int64_t const n_sources_x_sc_per_shot,
        int64_t const n_sources_p_sc_per_shot,
/* background receiver counts */
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_per_shot,
#endif
        int64_t const n_receivers_x_per_shot,
        int64_t const n_receivers_p_per_shot,
/* scattered receiver counts */
#if DW_NDIM >= 3
        int64_t const n_receivers_z_sc_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_sc_per_shot,
#endif
        int64_t const n_receivers_x_sc_per_shot,
        int64_t const n_receivers_p_sc_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const lamb_batched_h,
        bool const mu_batched_h, bool const buoyancy_batched_h,
        bool const dlamb_batched_h, bool const dmu_batched_h,
        bool const dbuoyancy_batched_h, bool const store_bg,
        bool const store_sc, bool const storage_compression, int64_t start_t,
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
        int64_t const pml_x1_h,
#if DW_NDIM >= 3
        int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
        int64_t const bc_y0,
#endif
        int64_t const bc_x0,
#if DW_NDIM >= 3
        int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
        int64_t const bc_y1,
#endif
        int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio <= 0) return 1;

#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_sc_per_shot > 0) && (receivers_z_sc_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_sc_per_shot > 0) && (receivers_y_sc_i == NULL)) return 1;
#endif
    if ((n_receivers_x_sc_per_shot > 0) && (receivers_x_sc_i == NULL)) return 1;
    if ((n_receivers_p_sc_per_shot > 0) && (receivers_p_sc_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0 || n_sources_z_sc_per_shot > 0) &&
        sources_z_i == NULL)
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot > 0 || n_sources_y_sc_per_shot > 0) &&
        sources_y_i == NULL)
      return 1;
#endif
    if ((n_sources_x_per_shot > 0 || n_sources_x_sc_per_shot > 0) &&
        sources_x_i == NULL)
      return 1;
    if ((n_sources_p_per_shot > 0 || n_sources_p_sc_per_shot > 0) &&
        sources_p_i == NULL)
      return 1;
    bool const dw_has_sample =
        (start_t % step_ratio == 0 || nt > step_ratio - start_t % step_ratio);
#if DW_NDIM >= 3
    if (dw_has_sample && (store_bg) &&
        (dvzdbuoyancy_store_1 == NULL || dvzdz_store_1 == NULL ||
         dvzdx_plus_dvxdz_store_1 == NULL || dvzdy_plus_dvydz_store_1 == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if (dw_has_sample && (store_bg) &&
        (dvydbuoyancy_store_1 == NULL || dvydy_store_1 == NULL ||
         dvydx_plus_dvxdy_store_1 == NULL))
      return 1;
#endif
    if (dw_has_sample && (store_bg) &&
        (dvxdbuoyancy_store_1 == NULL || dvxdx_store_1 == NULL))
      return 1;
#if DW_NDIM >= 3
    if (dw_has_sample && (store_sc) &&
        (dvzdbuoyancy_sc_store_1 == NULL || dvzdz_sc_store_1 == NULL ||
         dvzdx_plus_dvxdz_sc_store_1 == NULL ||
         dvzdy_plus_dvydz_sc_store_1 == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if (dw_has_sample && (store_sc) &&
        (dvydbuoyancy_sc_store_1 == NULL || dvydy_sc_store_1 == NULL ||
         dvydx_plus_dvxdy_sc_store_1 == NULL))
      return 1;
#endif
    if (dw_has_sample && (store_sc) &&
        (dvxdbuoyancy_sc_store_1 == NULL || dvxdx_sc_store_1 == NULL))
      return 1;
  }
#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const shot_numel_h = nx_h;
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif
  /* A background/scattered store is needed whenever any scatter/background
   * (bg) or any background (sc) model requires a gradient; the Python driver
   * has already collapsed that into the store_bg / store_sc flags. */
  bool const store_needed_h = store_bg;
  bool const store_sc_needed_h = store_sc;
#if DW_NDIM == 3
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2LL * FD_PAD + 1, dimBlock.z);
#elif DW_NDIM == 2
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
#endif
  dim3 const dimGrid(gridx, gridy, gridz);
  dim3 const dimBlock_src_rcv(32, 1, 1);

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device_guard.selected_device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(dt_h,
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
                         pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
                         pml_y0_h, pml_y1_h,
#endif
                         pml_x0_h, pml_x1_h, lamb_batched_h, mu_batched_h,
                         buoyancy_batched_h, dlamb_batched_h, dmu_batched_h,
                         dbuoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#define LAUNCH_RECV(r_arr, wf, idx, count)                        \
  if ((count) > 0) {                                              \
    dim3 const g(ceil_div((count), dimBlock_src_rcv.x),           \
                 ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);     \
    record_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>( \
        (r_arr) + t * n_shots_h * (count), wf, idx, count);       \
    CHECK_KERNEL_ERROR                                            \
  }

#define LAUNCH_SRC(wf, f_arr, idx, count)                     \
  if ((count) > 0) {                                          \
    dim3 const g(ceil_div((count), dimBlock_src_rcv.x),       \
                 ceil_div(n_shots_h, dimBlock_src_rcv.y), 1); \
    add_sources<<<g, dimBlock_src_rcv, 0, stream_compute>>>(  \
        wf, (f_arr) + t * n_shots_h * (count), idx, count);   \
    CHECK_KERNEL_ERROR                                        \
  }

#define OPEN_FILE_WRITE(name, grad_cond)                          \
  ScopedFile fp_##name;                                           \
  file_group.add(fp_##name);                                      \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {              \
    if (!fp_##name.open(name##_filenames_ptr[0], "ab")) return 1; \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (store_needed_h || store_sc_needed_h);

  ScopedStream stream_storage;
  ScopedEvent event_storage_done_a, event_storage_done_b, event_compute_done_a,
      event_compute_done_b;
  cudaEvent_t event_storage_done = nullptr;
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

  /* background gradient store files (opened once, appended per snapshot) */
#if DW_NDIM >= 3
  OPEN_FILE_WRITE(dvzdbuoyancy, store_needed_h)
  OPEN_FILE_WRITE(dvzdz, store_needed_h)
  OPEN_FILE_WRITE(dvzdx_plus_dvxdz, store_needed_h)
  OPEN_FILE_WRITE(dvzdy_plus_dvydz, store_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_WRITE(dvydbuoyancy, store_needed_h)
  OPEN_FILE_WRITE(dvydy, store_needed_h)
  OPEN_FILE_WRITE(dvydx_plus_dvxdy, store_needed_h)
#endif
  OPEN_FILE_WRITE(dvxdbuoyancy, store_needed_h)
  OPEN_FILE_WRITE(dvxdx, store_needed_h)
  /* scattered gradient store files */
#if DW_NDIM >= 3
  OPEN_FILE_WRITE(dvzdbuoyancy_sc, store_sc_needed_h)
  OPEN_FILE_WRITE(dvzdz_sc, store_sc_needed_h)
  OPEN_FILE_WRITE(dvzdx_plus_dvxdz_sc, store_sc_needed_h)
  OPEN_FILE_WRITE(dvzdy_plus_dvydz_sc, store_sc_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_WRITE(dvydbuoyancy_sc, store_sc_needed_h)
  OPEN_FILE_WRITE(dvydy_sc, store_sc_needed_h)
  OPEN_FILE_WRITE(dvydx_plus_dvxdy_sc, store_sc_needed_h)
#endif
  OPEN_FILE_WRITE(dvxdbuoyancy_sc, store_sc_needed_h)
  OPEN_FILE_WRITE(dvxdx_sc, store_sc_needed_h)

  for (t = start_t; t < start_t + nt; ++t) {
    int64_t const step_idx = t / step_ratio;
    /* Per-quantity store snapshot destinations for this step (a-buffer,
     * ping-ponged to the b-buffer on odd snapshot steps when double
     * buffering). Mirrors elastic.cu / acoustic_born.cu, doubled for the
     * background and scattered store sets. */
#define SETUP_STORE_SAVE(name, grad_cond)                              \
  DW_DTYPE* __restrict name##_store_1_t =                              \
      name##_store_1 != NULL                                           \
          ? (name##_store_1 +                                          \
             ((storage_mode == STORAGE_DEVICE && !storage_compression) \
                  ? step_idx * n_shots_h * shot_numel_h                \
                  : 0))                                                \
          : NULL;                                                      \
  void* __restrict const name##_store_2_t =                            \
      name##_store_2 != NULL                                           \
          ? ((uint8_t*)name##_store_2 +                                \
             (storage_mode == STORAGE_DEVICE && storage_compression    \
                  ? step_idx * n_shots_h * shot_bytes_comp             \
                  : 0))                                                \
          : NULL;                                                      \
  void* __restrict const name##_store_3_t =                            \
      name##_store_3 != NULL                                           \
          ? ((uint8_t*)name##_store_3 +                                \
             (storage_mode == STORAGE_CPU                              \
                  ? step_idx * n_shots_h *                             \
                        (storage_compression ? shot_bytes_comp         \
                                             : shot_bytes_uncomp)      \
                  : 0))                                                \
          : NULL;                                                      \
  bool const name##_cond = (grad_cond) && ((t % step_ratio) == 0);     \
  if ((grad_cond) && use_double_buffering && (step_idx % 2 != 0)) {    \
    name##_store_1_t = name##_store_1b;                                \
  }

#if DW_NDIM >= 3
    SETUP_STORE_SAVE(dvzdbuoyancy, store_needed_h)
    SETUP_STORE_SAVE(dvzdz, store_needed_h)
    SETUP_STORE_SAVE(dvzdx_plus_dvxdz, store_needed_h)
    SETUP_STORE_SAVE(dvzdy_plus_dvydz, store_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_SAVE(dvydbuoyancy, store_needed_h)
    SETUP_STORE_SAVE(dvydy, store_needed_h)
    SETUP_STORE_SAVE(dvydx_plus_dvxdy, store_needed_h)
#endif
    SETUP_STORE_SAVE(dvxdbuoyancy, store_needed_h)
    SETUP_STORE_SAVE(dvxdx, store_needed_h)
#if DW_NDIM >= 3
    SETUP_STORE_SAVE(dvzdbuoyancy_sc, store_sc_needed_h)
    SETUP_STORE_SAVE(dvzdz_sc, store_sc_needed_h)
    SETUP_STORE_SAVE(dvzdx_plus_dvxdz_sc, store_sc_needed_h)
    SETUP_STORE_SAVE(dvzdy_plus_dvydz_sc, store_sc_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_SAVE(dvydbuoyancy_sc, store_sc_needed_h)
    SETUP_STORE_SAVE(dvydy_sc, store_sc_needed_h)
    SETUP_STORE_SAVE(dvydx_plus_dvxdy_sc, store_sc_needed_h)
#endif
    SETUP_STORE_SAVE(dvxdbuoyancy_sc, store_sc_needed_h)
    SETUP_STORE_SAVE(dvxdx_sc, store_sc_needed_h)

    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    bool const store_bg_t = store_needed_h && ((t % step_ratio) == 0);
    bool const store_sc_t = store_sc_needed_h && ((t % step_ratio) == 0);

    /* Refill the stress guard ring of each field independently (read by the
     * velocity-update kernel via sigma_AA/the shared-axis shears'
     * derivatives, and by the pressure receiver record below); mirrors
     * elastic.cu's own LAUNCH_FILL_ELASTIC_STRESS_BC placement (before
     * receiver recording, before the velocity-update kernel launch) and
     * elastic_born.c's own forward stress-fill placement. The macro's
     * formal parameter list is fixed at 6 slots (see its definition above);
     * unconditionally passing all 6 identifiers is safe even when some
     * name (e.g. sigmazz at DW_NDIM<3) does not exist as a declared
     * variable, because the DW_NDIM<3 macro body never references that
     * slot -- the token is discarded during expansion, never emitted into
     * the translation unit. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(sigmazz, sigmayz, sigmaxz, sigmayy,
                                         sigmaxy, sigmaxx);
      CHECK_KERNEL_ERROR
      LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(sigmazz_sc, sigmayz_sc, sigmaxz_sc,
                                         sigmayy_sc, sigmaxy_sc, sigmaxx_sc);
      CHECK_KERNEL_ERROR
    }

    /* record receivers (background then scattered) */
#if DW_NDIM >= 3
    LAUNCH_RECV(r_z, vz, receivers_z_i, n_receivers_z_per_shot)
    LAUNCH_RECV(r_z_sc, vz_sc, receivers_z_sc_i, n_receivers_z_sc_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_RECV(r_y, vy, receivers_y_i, n_receivers_y_per_shot)
    LAUNCH_RECV(r_y_sc, vy_sc, receivers_y_sc_i, n_receivers_y_sc_per_shot)
#endif
    LAUNCH_RECV(r_x, vx, receivers_x_i, n_receivers_x_per_shot)
    LAUNCH_RECV(r_x_sc, vx_sc, receivers_x_sc_i, n_receivers_x_sc_per_shot)
    if (n_receivers_p_per_shot > 0) {
      dim3 const g(ceil_div(n_receivers_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      record_pressure_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
          r_p + t * n_shots_h * n_receivers_p_per_shot,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, receivers_p_i, n_receivers_p_per_shot);
      CHECK_KERNEL_ERROR
    }
    if (n_receivers_p_sc_per_shot > 0) {
      dim3 const g(ceil_div(n_receivers_p_sc_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      record_pressure_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
          r_p_sc + t * n_shots_h * n_receivers_p_sc_per_shot,
#if DW_NDIM >= 3
          sigmazz_sc,
#endif
#if DW_NDIM >= 2
          sigmayy_sc,
#endif
          sigmaxx_sc, receivers_p_sc_i, n_receivers_p_sc_per_shot);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
        buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y, dbuoyancy_y,
#endif
        buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
        vz, vz_sc,
#endif
#if DW_NDIM >= 2
        vy, vy_sc,
#endif
        vx, vx_sc,
#if DW_NDIM >= 3
        sigmazz, sigmazz_sc, sigmayz, sigmayz_sc, sigmaxz, sigmaxz_sc,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmayy_sc, sigmaxy, sigmaxy_sc,
#endif
        sigmaxx, sigmaxx_sc,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmazzz_sc, m_sigmayzy, m_sigmayzy_sc, m_sigmaxzx,
        m_sigmaxzx_sc, m_sigmayzz, m_sigmayzz_sc, m_sigmaxzz, m_sigmaxzz_sc,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmayyy_sc, m_sigmaxyy, m_sigmaxyy_sc, m_sigmaxyx,
        m_sigmaxyx_sc,
#endif
        m_sigmaxxx, m_sigmaxxx_sc,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t, dvzdbuoyancy_sc_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t, dvydbuoyancy_sc_store_1_t,
#endif
        dvxdbuoyancy_store_1_t, dvxdbuoyancy_sc_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, store_bg_t, store_sc_t);
    CHECK_KERNEL_ERROR

    /* velocity/force sources (background then scattered) */
#if DW_NDIM >= 3
    LAUNCH_SRC(vz, f_z, sources_z_i, n_sources_z_per_shot)
    LAUNCH_SRC(vz_sc, f_z_sc, sources_z_i, n_sources_z_sc_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_SRC(vy, f_y, sources_y_i, n_sources_y_per_shot)
    LAUNCH_SRC(vy_sc, f_y_sc, sources_y_i, n_sources_y_sc_per_shot)
#endif
    LAUNCH_SRC(vx, f_x, sources_x_i, n_sources_x_per_shot)
    LAUNCH_SRC(vx_sc, f_x_sc, sources_x_i, n_sources_x_sc_per_shot)

    /* Refill the velocity guard ring of each field independently (read by
     * every normal stress's own divergence term and by the shared-axis
     * shears' updates); mirrors elastic.cu's own
     * LAUNCH_FILL_ELASTIC_VELOCITY_BC placement (after velocity source
     * injection, before the stress-update kernel launch) and
     * elastic_born.c's own forward velocity-fill placement. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz, vy, vx);
      CHECK_KERNEL_ERROR
      LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz_sc, vy_sc, vx_sc);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        lamb, mu, dlamb, dmu,
#if DW_NDIM >= 3
        mu_zy, dmu_zy, mu_zx, dmu_zx,
#endif
#if DW_NDIM >= 2
        mu_yx, dmu_yx,
#endif
#if DW_NDIM >= 3
        vz, vz_sc,
#endif
#if DW_NDIM >= 2
        vy, vy_sc,
#endif
        vx, vx_sc,
#if DW_NDIM >= 3
        sigmazz, sigmazz_sc, sigmayz, sigmayz_sc, sigmaxz, sigmaxz_sc,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmayy_sc, sigmaxy, sigmaxy_sc,
#endif
        sigmaxx, sigmaxx_sc,
#if DW_NDIM >= 3
        m_vzz, m_vzz_sc, m_vzy, m_vzy_sc, m_vzx, m_vzx_sc, m_vyz, m_vyz_sc,
        m_vxz, m_vxz_sc,
#endif
#if DW_NDIM >= 2
        m_vyy, m_vyy_sc, m_vyx, m_vyx_sc, m_vxy, m_vxy_sc,
#endif
        m_vxx, m_vxx_sc,
#if DW_NDIM >= 3
        dvzdz_store_1_t, dvzdz_sc_store_1_t, dvzdx_plus_dvxdz_store_1_t,
        dvzdx_plus_dvxdz_sc_store_1_t, dvzdy_plus_dvydz_store_1_t,
        dvzdy_plus_dvydz_sc_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydy_store_1_t, dvydy_sc_store_1_t, dvydx_plus_dvxdy_store_1_t,
        dvydx_plus_dvxdy_sc_store_1_t,
#endif
        dvxdx_store_1_t, dvxdx_sc_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, store_bg_t, store_sc_t);
    CHECK_KERNEL_ERROR

    /* forward physics for this step is done; hand the freshly-written store
     * buffers to the storage stream (double buffering) and snapshot them. */
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
#if DW_NDIM >= 3
    SAVE_SNAPSHOT(dvzdbuoyancy_sc)
    SAVE_SNAPSHOT(dvzdz_sc)
    SAVE_SNAPSHOT(dvzdx_plus_dvxdz_sc)
    SAVE_SNAPSHOT(dvzdy_plus_dvydz_sc)
#endif
#if DW_NDIM >= 2
    SAVE_SNAPSHOT(dvydbuoyancy_sc)
    SAVE_SNAPSHOT(dvydy_sc)
    SAVE_SNAPSHOT(dvydx_plus_dvxdy_sc)
#endif
    SAVE_SNAPSHOT(dvxdbuoyancy_sc)
    SAVE_SNAPSHOT(dvxdx_sc)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }
#undef SETUP_STORE_SAVE
#undef SAVE_SNAPSHOT

    /* pressure sources (background then scattered) */
    if (n_sources_p_per_shot > 0) {
      dim3 const g(ceil_div(n_sources_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      add_sources_p<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, f_p + t * n_shots_h * n_sources_p_per_shot, sources_p_i,
          n_sources_p_per_shot);
      CHECK_KERNEL_ERROR
    }
    if (n_sources_p_sc_per_shot > 0) {
      dim3 const g(ceil_div(n_sources_p_sc_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      add_sources_p<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz_sc,
#endif
#if DW_NDIM >= 2
          sigmayy_sc,
#endif
          sigmaxx_sc, f_p_sc + t * n_shots_h * n_sources_p_sc_per_shot,
          sources_p_i, n_sources_p_sc_per_shot);
      CHECK_KERNEL_ERROR
    }
  }

  /* Leave the returned wavefields boundary-consistent, matching
   * elastic_born_eager_core's own post-loop refill in elastic_born.py and
   * elastic_born.c's own FUNC(forward) (see that file's matching comment)
   * -- without this, the last timestep's stress update runs after the last
   * in-loop stress fill, so the returned sigma guard ring/on-surface zero
   * reflect a stale (pre-update) state. Receiver traces and gradients are
   * unaffected (already recorded/consumed inside the loop before this
   * point); only the final returned wavefields change. Applied to the
   * background field then the scattered field independently, as the
   * in-loop fills above already do. */
  if (has_bc) {
    LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(sigmazz, sigmayz, sigmaxz, sigmayy,
                                       sigmaxy, sigmaxx);
    CHECK_KERNEL_ERROR
    LAUNCH_FILL_ELASTIC_BORN_STRESS_BC(sigmazz_sc, sigmayz_sc, sigmaxz_sc,
                                       sigmayy_sc, sigmaxy_sc, sigmaxx_sc);
    CHECK_KERNEL_ERROR
    LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz, vy, vx);
    CHECK_KERNEL_ERROR
    LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC(vz_sc, vy_sc, vx_sc);
    CHECK_KERNEL_ERROR
  }

  if (use_double_buffering)
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
#undef OPEN_FILE_WRITE
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

/* ===========================================================================
 * Scattered-only elastic-Born adjoint (backward_sc)
 *
 * Physically the plain single-field elastic adjoint operator (elastic.cu
 * FUNC(backward)) run on the SCATTERED adjoint field with the BACKGROUND
 * models, loading the background gradient stores and accumulating into the
 * SCATTERING-model gradient buffers. Kept in lockstep with elastic_born.c's
 * FUNC(backward_sc) (ABI/ordering) and elastic.cu's FUNC(backward) (physics).
 * ======================================================================== */

/* Model/adjoint access + backward differencing macros (LAMB/MU/MU_ (yx/zx/zy)/
 * BUOYANCY_*, M_SIGMA** / M_V**, LAMB_2MU, VZ_Z/VZ_Y/VZ_X/.../SIGMAYZ_Y
 * (+ _PML variants)) now live in elastic_born_kernels.h, included above,
 * shared verbatim with elastic_born.c. */

namespace {
__launch_bounds__(128) __global__
    void combine_grad(DW_DTYPE* __restrict const grad,
                      DW_DTYPE const* __restrict const grad_shot) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
  if (z < nz && y < ny && x < nx) {
    int64_t const i = z * ny * nx + y * nx + x;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  if (y < ny && x < nx) {
    int64_t const i = y * nx + x;
#else
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (x < nx) {
    int64_t const i = x;
#endif
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
}

__launch_bounds__(128) __global__ void backward_kernel_sigma(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_vyz,
    DW_DTYPE const* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_vxy,
#endif
    DW_DTYPE const* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE* __restrict const m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE* __restrict const m_sigmaxyxn,
#endif
    DW_DTYPE* __restrict const m_sigmaxxxn,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dvydbuoyancy,
#endif
    DW_DTYPE const* __restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_buoyancy_y,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    bool const buoyancy_requires_grad, int64_t const step_ratio) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
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
      DW_DTYPE const* __restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const* __restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
#endif

#ifndef DW_MPML
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
#endif /* !DW_MPML */

      int64_t i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_sc_vz_cell)(
            lamb_shot, mu_shot, mu_zy_shot, mu_zx_shot, sigmazz, sigmayy,
            sigmaxx, sigmayz, sigmaxz, vz, m_vzz, m_vzy, m_vzx, bz, byh, bxh,
            dt, i, j, ny, nx, z, y, x, pml_z, pml_y, pml_x, rdz, rdy, rdx);
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_sc_vy_cell)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                                  mu_zy_shot,
#endif
                                  mu_yx_shot,
#if DW_NDIM >= 3
                                  sigmazz,
#endif
                                  sigmayy, sigmaxx,
#if DW_NDIM >= 3
                                  sigmayz,
#endif
                                  sigmaxy, vy,
#if DW_NDIM >= 3
                                  m_vyz,
#endif
                                  m_vyy, m_vyx,
#if DW_NDIM >= 3
                                  bzh,
#endif
                                  by, bxh, dt, i, j,
#if DW_NDIM >= 3
                                  ny,
#endif
                                  nx,
#if DW_NDIM >= 3
                                  z, pml_z, rdz,
#endif
                                  y, x, pml_y, pml_x, rdy, rdx);
      }
#endif

      if (x < nx - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_sc_vx_cell)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                                  mu_zx_shot,
#endif
#if DW_NDIM >= 2
                                  mu_yx_shot,
#endif
#if DW_NDIM >= 3
                                  sigmazz,
#endif
#if DW_NDIM >= 2
                                  sigmayy,
#endif
                                  sigmaxx,
#if DW_NDIM >= 3
                                  sigmaxz,
#endif
#if DW_NDIM >= 2
                                  sigmaxy,
#endif
                                  vx,
#if DW_NDIM >= 3
                                  m_vxz,
#endif
#if DW_NDIM >= 2
                                  m_vxy,
#endif
                                  m_vxx,
#if DW_NDIM >= 3
                                  bzh,
#endif
#if DW_NDIM >= 2
                                  byh,
#endif
                                  bx, dt, i, j,
#if DW_NDIM >= 3
                                  ny,
#endif
#if DW_NDIM >= 2
                                  nx,
#endif
#if DW_NDIM >= 3
                                  z, pml_z, rdz,
#endif
#if DW_NDIM >= 2
                                  y, pml_y, rdy,
#endif
                                  x, pml_x, rdx);
      }

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel_v(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy,
#endif
    DW_DTYPE const* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE* __restrict const m_vxy,
#endif
    DW_DTYPE* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dvzdz_store,
    DW_DTYPE const* __restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE const* __restrict const dvzdy_plus_dvydz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dvydy_store,
    DW_DTYPE const* __restrict const dvydx_plus_dvxdy_store,
#endif
    DW_DTYPE const* __restrict const dvxdx_store,
    DW_DTYPE* __restrict const grad_lamb, DW_DTYPE* __restrict const grad_mu,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_mu_zy,
    DW_DTYPE* __restrict const grad_mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    bool const lamb_requires_grad, bool const mu_requires_grad,
    int64_t const step_ratio) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
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
      DW_DTYPE const* __restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const* __restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

#ifndef DW_MPML
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
#endif /* !DW_MPML */

      int64_t const i = shot_idx * shot_numel + j;

      {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

        FUNC(backward_sc_sigmaii_cell)(
            lamb_shot, mu_shot,
#if DW_NDIM >= 3
            buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
            buoyancy_y_shot,
#endif
            buoyancy_x_shot,
#if DW_NDIM >= 3
            vz, sigmazz, m_vzz, m_sigmazzz, dvzdz_store,
#endif
#if DW_NDIM >= 2
            vy, sigmayy, m_vyy, m_sigmayyy, dvydy_store,
#endif
            vx, sigmaxx, m_vxx, m_sigmaxxx, dvxdx_store, grad_lamb, grad_mu, dt,
            step_ratio, lamb_requires_grad, mu_requires_grad, i, j,
#if DW_NDIM >= 3
            ny,
#endif
#if DW_NDIM >= 2
            nx,
#endif
#if DW_NDIM >= 3
            z, pml_z, az, azh, bz, bzh, rdz,
#endif
#if DW_NDIM >= 2
            y, pml_y, ay, ayh, by, byh, rdy,
#endif
            x, pml_x, ax, axh, bx, bxh, rdx);
      }

#if DW_NDIM >= 2
      if (y < ny - FD_PAD && x < nx - FD_PAD) {
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_sc_sigmaxy_cell)(
            mu_yx_shot, buoyancy_x_shot, buoyancy_y_shot, vx, vy, sigmaxy,
            m_vxy, m_vyx, m_sigmaxyy, m_sigmaxyx, dvydx_plus_dvxdy_store,
            grad_mu_yx, dt, step_ratio, mu_requires_grad, i, j,
#if DW_NDIM >= 3
            ny, z,
#endif
            nx, y, x, pml_y, pml_x, ay, ayh, by, byh, ax, axh, bx, bxh, rdy,
            rdx);
      }
#endif

#if DW_NDIM >= 3
      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_sc_sigmaxz_cell)(
            mu_zx_shot, buoyancy_x_shot, buoyancy_z_shot, vx, vz, sigmaxz,
            m_vxz, m_vzx, m_sigmaxzz, m_sigmaxzx, dvzdx_plus_dvxdz_store,
            grad_mu_zx, dt, step_ratio, mu_requires_grad, i, j, ny, nx, z, y, x,
            pml_z, pml_x, az, azh, bz, bzh, ax, axh, bx, bxh, rdz, rdx);
      }

      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        FUNC(backward_sc_sigmayz_cell)(
            mu_zy_shot, buoyancy_y_shot, buoyancy_z_shot, vy, vz, sigmayz,
            m_vyz, m_vzy, m_sigmayzz, m_sigmayzy, dvzdy_plus_dvydz_store,
            grad_mu_zy, dt, step_ratio, mu_requires_grad, i, j, ny, nx, z, y, x,
            pml_z, pml_y, az, azh, bz, bzh, ay, ayh, by, byh, rdz, rdy);
      }
#endif

#if DW_NDIM == 3
    }
#endif
  }
}

/* ========================= Guard-band gathers ========================= *
 *
 * CUDA counterpart of the guard-band-only companion functions in elastic_
 * kernels.h (FUNC(backward_sc_vz_cell_guard) etc -- see that file's "Guard-band
 * gathers" module comment for the full rationale). The two kernels below
 * (backward_kernel_sigma_guard, backward_kernel_v_guard) are guard-band-only
 * companions to backward_kernel_sigma/backward_kernel_v above: same fields,
 * same per-cell math (via the *_guard functions), but launched over the
 * WHOLE domain (grid indices start at 0, not FD_PAD) since a guard cell by
 * definition lives inside the band backward_kernel_sigma/backward_kernel_v
 * exclude. Each thread computes at most one update per field at its own,
 * unique (shot, z, y, x) -- no two threads ever accumulate into the same
 * output index -- so these are race-free without atomics purely from CUDA's
 * per-thread-independent-output property; ordering versus every other
 * kernel here is guaranteed by the single compute stream (stream_compute)
 * every launch in this file uses, so placement (immediately after the
 * interior kernel, before the corresponding fold-fill launch -- see
 * FUNC(backward)'s driver body below) is the only thing that matters, not
 * concurrency. backward_kernel_v_pml_mem is the post-fold M_SIGMA**
 * PML-memory-variable update and buoyancy-gradient pass elastic_kernels.h's
 * own module comment describes (both moved out of FUNC(backward_v*_cell)
 * into a separate pass so they read v*[i] after the velocity fold changes
 * near-face interior cells, e.g. a rigid on-row velocity clamp node); it is
 * launched unconditionally (the PML-memory write matters every time step
 * regardless of buoyancy_requires_grad) over (approximately) the same
 * unextended range FUNC(forward_vz_cell) etc. used to write dv*dbuoyancy in
 * the forward pass.
 */
__launch_bounds__(128) __global__ void backward_kernel_sigma_guard(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_vyz,
    DW_DTYPE const* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_vxy,
#endif
    DW_DTYPE const* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE* __restrict const m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE* __restrict const m_sigmaxyxn,
#endif
    DW_DTYPE* __restrict const m_sigmaxxxn,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0, int64_t const bc_y1,
#endif
    int64_t const bc_x0, int64_t const bc_x1) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      DW_DTYPE const* __restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const* __restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        bool const in_target = FUNC(dw_in_target)(rz, z) &&
                               FUNC(dw_in_target)(ry, y) &&
                               FUNC(dw_in_target)(rx, x);
        bool const all_std = FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(ry, y) &&
                             FUNC(dw_in_std)(rx, x);
        if (in_target && !all_std) {
          /* Guard-ring cells have no forward-pass counterpart at all (see
           * the "Guard-band gathers" module comment in elastic_kernels.h),
           * so there is no PML-memory recursion state to read/write here;
           * always take the plain (non-PML) branch. */
          bool const pml_z = false;
          bool const pml_y = false;
          bool const pml_x = false;
          FUNC(backward_sc_vz_cell_guard)(
              lamb_shot, mu_shot, mu_zy_shot, mu_zx_shot, sigmazz, sigmayy,
              sigmaxx, sigmayz, sigmaxz, vz, m_vzz, m_vzy, m_vzx, m_sigmazzz,
              m_sigmazzzn, m_sigmayzy, m_sigmayzyn, m_sigmaxzx, m_sigmaxzxn,
              buoyancy_z_shot[j], az, azh, bz, bzh, ay, ayh, by, byh, ax, axh,
              bx, bxh, dt, i, j, nz, ny, nx, z, y, x, pml_z, pml_y, pml_x, rdz,
              rdy, rdx);
        }
      }
#endif
#if DW_NDIM >= 2
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
#if DW_NDIM >= 3
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
#endif
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        bool const in_target = FUNC(dw_in_target)(ry, y) &&
                               FUNC(dw_in_target)(rx, x)
#if DW_NDIM >= 3
                               && FUNC(dw_in_target)(rz, z)
#endif
            ;
        bool const all_std = FUNC(dw_in_std)(ry, y) && FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 3
                             && FUNC(dw_in_std)(rz, z)
#endif
            ;
        if (in_target && !all_std) {
          /* Guard cells have no forward-pass PML state -- see the module
           * comment above the vz guard-band block. */
          bool const pml_y = false;
          bool const pml_x = false;
#if DW_NDIM >= 3
          bool const pml_z = false;
#endif
          FUNC(backward_sc_vy_cell_guard)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                                          mu_zy_shot,
#endif
                                          mu_yx_shot,
#if DW_NDIM >= 3
                                          sigmazz,
#endif
                                          sigmayy, sigmaxx,
#if DW_NDIM >= 3
                                          sigmayz,
#endif
                                          sigmaxy, vy,
#if DW_NDIM >= 3
                                          m_vyz,
#endif
                                          m_vyy, m_vyx,
#if DW_NDIM >= 3
                                          m_sigmayzz, m_sigmayzzn,
#endif
                                          m_sigmayyy, m_sigmayyyn, m_sigmaxyx,
                                          m_sigmaxyxn, buoyancy_y_shot[j],
#if DW_NDIM >= 3
                                          az, azh, bz, bzh,
#endif
                                          ay, ayh, by, byh, ax, axh, bx, bxh,
                                          dt, i, j,
#if DW_NDIM >= 3
                                          nz,
#endif
                                          ny, nx,
#if DW_NDIM >= 3
                                          z, pml_z, rdz,
#endif
                                          y, x, pml_y, pml_x, rdy, rdx);
        }
      }
#endif
      {
#if DW_NDIM >= 3
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
#endif
#if DW_NDIM >= 2
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
#endif
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        bool const in_target = FUNC(dw_in_target)(rx, x)
#if DW_NDIM >= 2
                               && FUNC(dw_in_target)(ry, y)
#endif
#if DW_NDIM >= 3
                               && FUNC(dw_in_target)(rz, z)
#endif
            ;
        bool const all_std = FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 2
                             && FUNC(dw_in_std)(ry, y)
#endif
#if DW_NDIM >= 3
                             && FUNC(dw_in_std)(rz, z)
#endif
            ;
        if (in_target && !all_std) {
          /* Guard cells have no forward-pass PML state -- see the module
           * comment above the vz guard-band block. */
#if DW_NDIM >= 3
          bool const pml_z = false;
#endif
#if DW_NDIM >= 2
          bool const pml_y = false;
#endif
          bool const pml_x = false;
          FUNC(backward_sc_vx_cell_guard)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                                          mu_zx_shot,
#endif
#if DW_NDIM >= 2
                                          mu_yx_shot,
#endif
#if DW_NDIM >= 3
                                          sigmazz,
#endif
#if DW_NDIM >= 2
                                          sigmayy,
#endif
                                          sigmaxx,
#if DW_NDIM >= 3
                                          sigmaxz,
#endif
#if DW_NDIM >= 2
                                          sigmaxy,
#endif
                                          vx,
#if DW_NDIM >= 3
                                          m_vxz,
#endif
#if DW_NDIM >= 2
                                          m_vxy,
#endif
                                          m_vxx,
#if DW_NDIM >= 3
                                          m_sigmaxzz, m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
                                          m_sigmaxyy, m_sigmaxyyn,
#endif
                                          m_sigmaxxx, m_sigmaxxxn,
                                          buoyancy_x_shot[j],
#if DW_NDIM >= 3
                                          az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
                                          ay, ayh, by, byh,
#endif
                                          ax, axh, bx, bxh, dt, i, j,
#if DW_NDIM >= 3
                                          nz,
#endif
#if DW_NDIM >= 2
                                          ny,
#endif
                                          nx,
#if DW_NDIM >= 3
                                          z, pml_z, rdz,
#endif
#if DW_NDIM >= 2
                                          y, pml_y, rdy,
#endif
                                          x, pml_x, rdx);
        }
      }

#if DW_NDIM == 3
    }
#endif
  }
}

/* Post-fold buoyancy-gradient pass -- see the module comment above and
 * elastic_kernels.h's own comment above FUNC(backward_vz_cell) (the
 * buoyancy-gradient accumulation that used to live inside that function).
 * Launched over the same range FUNC(forward_vz_cell)/vy_cell/vx_cell wrote
 * dv*dbuoyancy in the forward pass (own axis half-grid, other axes
 * full-grid), so every cell it reads has a real (non-garbage) stored
 * value. */
__launch_bounds__(128) __global__ void backward_kernel_v_pml_mem(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy,
#endif
    DW_DTYPE const* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE* __restrict const m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE* __restrict const m_sigmaxyxn,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE* __restrict const m_sigmaxxxn,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dvydbuoyancy,
#endif
    DW_DTYPE const* __restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_buoyancy_y,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    bool const buoyancy_requires_grad_t, int64_t const step_ratio) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      /* M_SIGMA** PML memory-variable update (split out of FUNC(backward_
       * v*_cell) -- see that split's comment in elastic_kernels.h) and the
       * buoyancy-gradient accumulation both need v*[i] read post-fold (the
       * fold kernels ran just before this one on the same stream), and
       * both use the SAME (unextended) interior/PML range FUNC(forward_
       * vz_cell) etc. used in the forward pass -- see elastic.c's own
       * comment at its equivalent call site. The PML memory-variable write
       * runs unconditionally; the gradient accumulation is gated on
       * buoyancy_requires_grad_t. */
#if DW_NDIM >= 3
      if (z >= FD_PAD && z < nz - FD_PAD && y >= FD_PAD &&
          y < ny - FD_PAD + 1 && x >= FD_PAD && x < nx - FD_PAD + 1) {
        int64_t const pml_z1h = pml_z0 > pml_z1 - 1 ? pml_z0 : pml_z1 - 1;
        bool const is_pml_z =
            (z >= FD_PAD && z < pml_z0) || (z >= pml_z1h && z < nz - FD_PAD);
        bool const is_pml_y =
            (y >= FD_PAD && y < pml_y0) || (y >= pml_y1 && y < ny - FD_PAD + 1);
        bool const is_pml_x =
            (x >= FD_PAD && x < pml_x0) || (x >= pml_x1 && x < nx - FD_PAD + 1);
        if (is_pml_z) {
          m_sigmazzzn[i] = buoyancy_z_shot[j] * dt * P_AZH(0) * vz[i] +
                           P_AZH(0) * m_sigmazzz[i];
        }
        if (is_pml_y) {
          m_sigmayzyn[i] = buoyancy_z_shot[j] * dt * P_AY(0) * vz[i] +
                           P_AY(0) * m_sigmayzy[i];
        }
        if (is_pml_x) {
          m_sigmaxzxn[i] = buoyancy_z_shot[j] * dt * P_AX(0) * vz[i] +
                           P_AX(0) * m_sigmaxzx[i];
        }
        if (buoyancy_requires_grad_t) {
          grad_buoyancy_z[i] += vz[i] * dvzdbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif
#if DW_NDIM >= 2
      if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD && x < nx - FD_PAD + 1
#if DW_NDIM >= 3
          && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
      ) {
        int64_t const pml_y1h = pml_y0 > pml_y1 - 1 ? pml_y0 : pml_y1 - 1;
        bool const is_pml_y =
            (y >= FD_PAD && y < pml_y0) || (y >= pml_y1h && y < ny - FD_PAD);
        bool const is_pml_x =
            (x >= FD_PAD && x < pml_x0) || (x >= pml_x1 && x < nx - FD_PAD + 1);
#if DW_NDIM >= 3
        bool const is_pml_z =
            (z >= FD_PAD && z < pml_z0) || (z >= pml_z1 && z < nz - FD_PAD + 1);
        if (is_pml_z) {
          m_sigmayzzn[i] = buoyancy_y_shot[j] * dt * P_AZ(0) * vy[i] +
                           P_AZ(0) * m_sigmayzz[i];
        }
#endif
        if (is_pml_y) {
          m_sigmayyyn[i] = buoyancy_y_shot[j] * dt * P_AYH(0) * vy[i] +
                           P_AYH(0) * m_sigmayyy[i];
        }
        if (is_pml_x) {
          m_sigmaxyxn[i] = buoyancy_y_shot[j] * dt * P_AX(0) * vy[i] +
                           P_AX(0) * m_sigmaxyx[i];
        }
        if (buoyancy_requires_grad_t) {
          grad_buoyancy_y[i] += vy[i] * dvydbuoyancy[i] * (DW_DTYPE)step_ratio;
        }
      }
#endif
      if (x >= FD_PAD && x < nx - FD_PAD
#if DW_NDIM >= 2
          && y >= FD_PAD && y < ny - FD_PAD + 1
#endif
#if DW_NDIM >= 3
          && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
      ) {
#if DW_NDIM >= 3
        bool const is_pml_z =
            (z >= FD_PAD && z < pml_z0) || (z >= pml_z1 && z < nz - FD_PAD + 1);
        if (is_pml_z) {
          m_sigmaxzzn[i] = buoyancy_x_shot[j] * dt * P_AZ(0) * vx[i] +
                           P_AZ(0) * m_sigmaxzz[i];
        }
#endif
#if DW_NDIM >= 2
        bool const is_pml_y =
            (y >= FD_PAD && y < pml_y0) || (y >= pml_y1 && y < ny - FD_PAD + 1);
        if (is_pml_y) {
          m_sigmaxyyn[i] = buoyancy_x_shot[j] * dt * P_AY(0) * vx[i] +
                           P_AY(0) * m_sigmaxyy[i];
        }
#endif
        int64_t const pml_x1h = pml_x0 > pml_x1 - 1 ? pml_x0 : pml_x1 - 1;
        bool const is_pml_x =
            (x >= FD_PAD && x < pml_x0) || (x >= pml_x1h && x < nx - FD_PAD);
        if (is_pml_x) {
          m_sigmaxxxn[i] = buoyancy_x_shot[j] * dt * P_AXH(0) * vx[i] +
                           P_AXH(0) * m_sigmaxxx[i];
        }
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
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy,
#endif
    DW_DTYPE const* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE* __restrict const m_vxy,
#endif
    DW_DTYPE* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0, int64_t const bc_y1,
#endif
    int64_t const bc_x0, int64_t const bc_x1) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
#endif

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
        if (FUNC(dw_in_target)(rz, z) && !FUNC(dw_in_std)(rz, z) &&
            y >= FD_PAD && y < ny - FD_PAD + 1 && x >= FD_PAD &&
            x < nx - FD_PAD + 1) {
          FUNC(backward_sc_sigmazz_z_cell_guard)(buoyancy_z_shot, vz, sigmazz,
                                                 dt, i, j, nz, ny, nx, z, rdz);
        }
      }
#endif
#if DW_NDIM >= 2
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
        if (FUNC(dw_in_target)(ry, y) && !FUNC(dw_in_std)(ry, y) &&
            x >= FD_PAD && x < nx - FD_PAD + 1
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          FUNC(backward_sc_sigmayy_y_cell_guard)(buoyancy_y_shot, vy, sigmayy,
                                                 dt, i, j, ny, nx, y, rdy);
        }
      }
#endif
      {
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        if (FUNC(dw_in_target)(rx, x) && !FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 2
            && y >= FD_PAD && y < ny - FD_PAD + 1
#endif
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          FUNC(backward_sc_sigmaxx_x_cell_guard)(buoyancy_x_shot, vx, sigmaxx,
                                                 dt, i, j,
#if DW_NDIM >= 3
                                                 ny,
#endif
                                                 nx, x, rdx);
        }
      }
#if DW_NDIM >= 2
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        bool const in_target =
            FUNC(dw_in_target)(ry, y) && FUNC(dw_in_target)(rx, x);
        bool const all_std = FUNC(dw_in_std)(ry, y) && FUNC(dw_in_std)(rx, x);
        if (in_target && !all_std
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          /* Guard cells have no forward-pass PML state -- see the module
           * comment above the vz guard-band block. */
          bool const pml_y = false;
          bool const pml_x = false;
          FUNC(backward_sc_sigmaxy_cell_guard)(
              mu_yx_shot, buoyancy_x_shot, buoyancy_y_shot, vx, vy, sigmaxy,
              m_vxy, m_vyx, m_sigmaxyy, m_sigmaxyx, dt, i, j,
#if DW_NDIM >= 3
              nz, z,
#endif
              ny, nx, y, x, pml_y, pml_x, ay, ayh, by, byh, ax, axh, bx, bxh,
              rdy, rdx);
        }
      }
#endif
#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        bool const in_target =
            FUNC(dw_in_target)(rz, z) && FUNC(dw_in_target)(rx, x);
        bool const all_std = FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(rx, x);
        if (in_target && !all_std && y >= FD_PAD && y < ny - FD_PAD + 1) {
          /* Guard cells have no forward-pass PML state -- see the module
           * comment above the vz guard-band block. */
          bool const pml_z = false;
          bool const pml_x = false;
          FUNC(backward_sc_sigmaxz_cell_guard)(
              mu_zx_shot, buoyancy_x_shot, buoyancy_z_shot, vx, vz, sigmaxz,
              m_vxz, m_vzx, m_sigmaxzz, m_sigmaxzx, dt, i, j, nz, ny, nx, z, y,
              x, pml_z, pml_x, az, azh, bz, bzh, ax, axh, bx, bxh, rdz, rdx);
        }
      }
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
        bool const in_target =
            FUNC(dw_in_target)(rz, z) && FUNC(dw_in_target)(ry, y);
        bool const all_std = FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(ry, y);
        if (in_target && !all_std && x >= FD_PAD && x < nx - FD_PAD + 1) {
          /* Guard cells have no forward-pass PML state -- see the module
           * comment above the vz guard-band block. */
          bool const pml_z = false;
          bool const pml_y = false;
          FUNC(backward_sc_sigmayz_cell_guard)(
              mu_zy_shot, buoyancy_y_shot, buoyancy_z_shot, vy, vz, sigmayz,
              m_vyz, m_vzy, m_sigmayzz, m_sigmayzy, dt, i, j, nz, ny, nx, z, y,
              x, pml_z, pml_y, az, azh, bz, bzh, ay, ayh, by, byh, rdz, rdy);
        }
      }
#endif

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
    int FUNC(backward_sc)(
        /* --- background models (read-only) --- */
        DW_DTYPE const* __restrict const lamb,
        DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const mu_zy,
        DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const buoyancy_y,
#endif
        DW_DTYPE const* __restrict const buoyancy_x,
/* --- scattered data residuals (adjoint sources at receivers) --- */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const grad_r_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const grad_r_y,
#endif
        DW_DTYPE const* __restrict const grad_r_x,
        DW_DTYPE const* __restrict const grad_r_p,
/* --- scattered adjoint wavefields (in/out) --- */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
        DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmaxz,
        DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
        DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
        DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict m_sigmazzz,
        DW_DTYPE* __restrict m_sigmayzy, DW_DTYPE* __restrict m_sigmaxzx,
        DW_DTYPE* __restrict m_sigmayzz, DW_DTYPE* __restrict m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
        DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vyy,
        DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vxy,
        DW_DTYPE* __restrict m_sigmayyy, DW_DTYPE* __restrict m_sigmaxyy,
        DW_DTYPE* __restrict m_sigmaxyx,
#endif
        DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
        DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict m_sigmaxxx,
/* --- m_sigma* adjoint double-buffer scratch ("n" buffers) --- */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict m_sigmazzzn, DW_DTYPE* __restrict m_sigmayzyn,
        DW_DTYPE* __restrict m_sigmaxzxn, DW_DTYPE* __restrict m_sigmayzzn,
        DW_DTYPE* __restrict m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict m_sigmayyyn, DW_DTYPE* __restrict m_sigmaxyyn,
        DW_DTYPE* __restrict m_sigmaxyxn,
#endif
        DW_DTYPE* __restrict m_sigmaxxxn,
/* --- background gradient stores (loaded via SETUP_STORE_LOAD) --- */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
        DW_DTYPE* __restrict const dvzdbuoyancy_store_1b,
        void* __restrict const dvzdbuoyancy_store_2,
        void* __restrict const dvzdbuoyancy_store_3,
        char const* __restrict const* __restrict const dvzdbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvzdz_store_1,
        DW_DTYPE* __restrict const dvzdz_store_1b,
        void* __restrict const dvzdz_store_2,
        void* __restrict const dvzdz_store_3,
        char const* __restrict const* __restrict const dvzdz_filenames_ptr,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1b,
        void* __restrict const dvzdx_plus_dvxdz_store_2,
        void* __restrict const dvzdx_plus_dvxdz_store_3,
        char const* __restrict const* __restrict const dvzdx_plus_dvxdz_filenames_ptr,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1b,
        void* __restrict const dvzdy_plus_dvydz_store_2,
        void* __restrict const dvzdy_plus_dvydz_store_3,
        char const* __restrict const* __restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydbuoyancy_store_1,
        DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
        void* __restrict const dvydbuoyancy_store_2,
        void* __restrict const dvydbuoyancy_store_3,
        char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvydy_store_1,
        DW_DTYPE* __restrict const dvydy_store_1b,
        void* __restrict const dvydy_store_2,
        void* __restrict const dvydy_store_3,
        char const* __restrict const* __restrict const dvydy_filenames_ptr,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1b,
        void* __restrict const dvydx_plus_dvxdy_store_2,
        void* __restrict const dvydx_plus_dvxdy_store_3,
        char const* __restrict const* __restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
        DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
        DW_DTYPE* __restrict const dvxdbuoyancy_store_1b,
        void* __restrict const dvxdbuoyancy_store_2,
        void* __restrict const dvxdbuoyancy_store_3,
        char const* __restrict const* __restrict const dvxdbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvxdx_store_1,
        DW_DTYPE* __restrict const dvxdx_store_1b,
        void* __restrict const dvxdx_store_2,
        void* __restrict const dvxdx_store_3,
        char const* __restrict const* __restrict const dvxdx_filenames_ptr,
        /* --- scattering-model gradients (combine target) --- */
        DW_DTYPE* __restrict const grad_dlamb,
        DW_DTYPE* __restrict const grad_dmu,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dmu_zy,
        DW_DTYPE* __restrict const grad_dmu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dmu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dbuoyancy_y,
#endif
        DW_DTYPE* __restrict const grad_dbuoyancy_x,
        /* --- scattering-model gradient per-shot reduction buffers --- */
        DW_DTYPE* __restrict const grad_dlamb_shot,
        DW_DTYPE* __restrict const grad_dmu_shot,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dmu_zy_shot,
        DW_DTYPE* __restrict const grad_dmu_zx_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dmu_yx_shot,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dbuoyancy_y_shot,
#endif
        DW_DTYPE* __restrict const grad_dbuoyancy_x_shot,
/* --- scattered source-amplitude gradients --- */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_f_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_f_y,
#endif
        DW_DTYPE* __restrict const grad_f_x,
        DW_DTYPE* __restrict const grad_f_p,
/* --- PML profiles (az,bz,azh,bzh order) --- */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const az,
        DW_DTYPE const* __restrict const bz,
        DW_DTYPE const* __restrict const azh,
        DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const ay,
        DW_DTYPE const* __restrict const by,
        DW_DTYPE const* __restrict const ayh,
        DW_DTYPE const* __restrict const byh,
#endif
        DW_DTYPE const* __restrict const ax,
        DW_DTYPE const* __restrict const bx,
        DW_DTYPE const* __restrict const axh,
        DW_DTYPE const* __restrict const bxh,
/* --- shared source indices --- */
#if DW_NDIM >= 3
        int64_t const* __restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const sources_y_i,
#endif
        int64_t const* __restrict const sources_x_i,
        int64_t const* __restrict const sources_p_i,
/* --- scattered receiver indices --- */
#if DW_NDIM >= 3
        int64_t const* __restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const receivers_y_i,
#endif
        int64_t const* __restrict const receivers_x_i,
        int64_t const* __restrict const receivers_p_i,
/* --- scalars --- */
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
        int64_t const n_receivers_x_per_shot,
        int64_t const n_receivers_p_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const lamb_batched_h,
        bool const mu_batched_h, bool const buoyancy_batched_h,
        bool const dlamb_requires_grad, bool const dmu_requires_grad,
        bool const dbuoyancy_requires_grad, bool const dlamb_batched_h,
        bool const dmu_batched_h, bool const dbuoyancy_batched_h,
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
        int64_t const pml_x1_h,
#if DW_NDIM >= 3
        int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
        int64_t const bc_y0,
#endif
        int64_t const bc_x0,
#if DW_NDIM >= 3
        int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
        int64_t const bc_y1,
#endif
        int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio <= 0) return 1;
    if ((dlamb_requires_grad) && (grad_dlamb == NULL)) return 1;
    if ((dmu_requires_grad) && (grad_dmu == NULL)) return 1;
#if DW_NDIM >= 3
    if ((dmu_requires_grad) && (grad_dmu_zy == NULL || grad_dmu_zx == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((dmu_requires_grad) && (grad_dmu_yx == NULL)) return 1;
#endif
#if DW_NDIM >= 3
    if ((dbuoyancy_requires_grad) && (grad_dbuoyancy_z == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((dbuoyancy_requires_grad) && (grad_dbuoyancy_y == NULL)) return 1;
#endif
    if ((dbuoyancy_requires_grad) && (grad_dbuoyancy_x == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0) && sources_z_i == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot > 0) && sources_y_i == NULL) return 1;
#endif
    if ((n_sources_x_per_shot > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_p_per_shot > 0) && sources_p_i == NULL) return 1;
  }
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2LL * FD_PAD + 1, dimBlock.z);
#elif DW_NDIM == 2
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
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
#elif DW_NDIM == 2
  unsigned int const gridx_g = ceil_div(nx_h, dimBlock.x);
  unsigned int const gridy_g = ceil_div(ny_h, dimBlock.y);
  unsigned int const gridz_g = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  unsigned int const gridx_g = ceil_div(nx_h, dimBlock.x);
  unsigned int const gridy_g = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz_g = 1;
#endif
  dim3 const dimGrid_guard(gridx_g, gridy_g, gridz_g);
  dim3 const dimBlock_src_rcv(32, 1, 1);

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device_guard.selected_device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(dt_h,
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
                         pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
                         pml_y0_h, pml_y1_h,
#endif
                         pml_x0_h, pml_x1_h, lamb_batched_h, mu_batched_h,
                         buoyancy_batched_h, dlamb_batched_h, dmu_batched_h,
                         dbuoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

#define OPEN_FILE_READ(name, grad_cond)                           \
  ScopedFile fp_##name;                                           \
  file_group.add(fp_##name);                                      \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {              \
    if (!fp_##name.open(name##_filenames_ptr[0], "rb")) return 1; \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (dlamb_requires_grad || dmu_requires_grad || dbuoyancy_requires_grad);

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
  OPEN_FILE_READ(dvzdbuoyancy, dbuoyancy_requires_grad)
  OPEN_FILE_READ(dvzdz, dlamb_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvzdx_plus_dvxdz, dmu_requires_grad)
  OPEN_FILE_READ(dvzdy_plus_dvydz, dmu_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(dvydbuoyancy, dbuoyancy_requires_grad)
  OPEN_FILE_READ(dvydy, dlamb_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvydx_plus_dvxdy, dmu_requires_grad)
#endif
  OPEN_FILE_READ(dvxdbuoyancy, dbuoyancy_requires_grad)
  OPEN_FILE_READ(dvxdx, dlamb_requires_grad || dmu_requires_grad)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio;

#define SETUP_STORE_LOAD(name, grad_cond)                                 \
  DW_DTYPE const* __restrict name##_store_1_t =                           \
      name##_store_1 != NULL                                              \
          ? (name##_store_1 +                                             \
             ((storage_mode == STORAGE_DEVICE && !storage_compression)    \
                  ? (t / step_ratio) * n_shots_h * shot_numel_h           \
                  : 0))                                                   \
          : NULL;                                                         \
  void* __restrict const name##_store_2_t =                               \
      name##_store_2 != NULL                                              \
          ? ((uint8_t*)name##_store_2 +                                   \
             (storage_mode == STORAGE_DEVICE && storage_compression       \
                  ? (t / step_ratio) * n_shots_h * shot_bytes_comp        \
                  : 0))                                                   \
          : NULL;                                                         \
  void* __restrict const name##_store_3_t =                               \
      name##_store_3 != NULL                                              \
          ? ((uint8_t*)name##_store_3 +                                   \
             (storage_mode == STORAGE_CPU                                 \
                  ? (t / step_ratio) * n_shots_h *                        \
                        (storage_compression ? shot_bytes_comp            \
                                             : shot_bytes_uncomp)         \
                  : 0))                                                   \
          : NULL;                                                         \
  bool const name##_cond = (grad_cond) && ((t % step_ratio) == 0);        \
  if (name##_cond) {                                                      \
    if (use_double_buffering && step_idx % 2 != 0) {                      \
      name##_store_1_t = name##_store_1b;                                 \
    }                                                                     \
    if (STORAGE_FUNC(load_snapshot_gpu)(                                  \
            (void*)name##_store_1_t, name##_store_2_t, name##_store_3_t,  \
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
    SETUP_STORE_LOAD(dvzdbuoyancy, dbuoyancy_requires_grad)
    SETUP_STORE_LOAD(dvzdz, dlamb_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvzdx_plus_dvxdz, dmu_requires_grad)
    SETUP_STORE_LOAD(dvzdy_plus_dvydz, dmu_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(dvydbuoyancy, dbuoyancy_requires_grad)
    SETUP_STORE_LOAD(dvydy, dlamb_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvydx_plus_dvxdy, dmu_requires_grad)
#endif
    SETUP_STORE_LOAD(dvxdbuoyancy, dbuoyancy_requires_grad)
    SETUP_STORE_LOAD(dvxdx, dlamb_requires_grad || dmu_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    /* Refill the (adjoint) stress guard ring of the scattered adjoint field
     * -- the discrete adjoint of forward's velocity-update kernel (which
     * read the stress ghost) reads it here too, before the adjoint
     * velocity update and before the source-gradient extraction just below
     * (the adjoint counterpart of forward's pressure receiver record);
     * mirrors elastic_born.c's own backward_sc stress-fill placement and
     * elastic.cu's own backward FILL_P_BC placement (self-adjointness of
     * the mirror-fill operator). Only the scattered adjoint field is
     * guard-filled here -- backward_sc has no background adjoint field
     * (the background models are read-only). Unlike acoustic's pressure
     * fill, elastic_born's multi-field stress fill is not self-adjoint (see
     * the module comment above launch_fill_elastic_born_stress_bc_adjoint),
     * so this applies that fill's true transpose rather than reapplying the
     * forward fill; mirrors elastic_born.c's own backward_sc fix. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_BORN_STRESS_BC_ADJOINT(sigmazz, sigmayz, sigmaxz,
                                                 sigmayy, sigmaxy, sigmaxx);
      CHECK_KERNEL_ERROR
    }

    /* pressure adjoint-source record: grad_f_p from sigma at source_p sites */
    if (n_sources_p_per_shot > 0) {
      dim3 const g(ceil_div(n_sources_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      record_pressure_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_p + t * n_shots_h * n_sources_p_per_shot,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, sources_p_i, n_sources_p_per_shot);
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
        grad_dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        grad_dbuoyancy_y_shot,
#endif
        grad_dbuoyancy_x_shot,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, dbuoyancy_requires_grad && ((t % step_ratio) == 0),
        step_ratio);
    CHECK_KERNEL_ERROR

    /* Guard-band extension of the velocity-adjoint gathers -- see the
     * "Guard-band gathers" module comment above backward_kernel_sigma_
     * guard. Must run (and complete, via the shared stream_compute
     * ordering) before FUNC(fill_elastic_born_velocity_bc_adjoint)'s
     * kernels just below, which fold this guard-band deposit into the
     * interior. */
    if (has_bc) {
      backward_kernel_sigma_guard<<<dimGrid_guard, dimBlock, 0,
                                    stream_compute>>>(
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
          az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
          ay, ayh, by, byh,
#endif
          ax, axh, bx, bxh,
#if DW_NDIM >= 3
          bc_z0, bc_z1,
#endif
#if DW_NDIM >= 2
          bc_y0, bc_y1,
#endif
          bc_x0, bc_x1);
      CHECK_KERNEL_ERROR
    }

    /* Refill the (adjoint) velocity guard ring of the scattered adjoint
     * field -- the discrete adjoint of forward's stress-update kernel
     * (which read the velocity ghost) reads it here too, before the
     * stress-adjoint update kernel just below; mirrors elastic_born.c's own
     * backward_sc velocity-fill placement and elastic.cu's own backward
     * FILL_V_BC placement. Only the scattered adjoint field is guard-filled
     * here (see the analogous note above). Unlike acoustic's velocity fill,
     * elastic_born's multi-field velocity fill is not self-adjoint (see the
     * module comment above launch_fill_elastic_born_velocity_bc_adjoint),
     * so this applies that fill's true transpose rather than reapplying the
     * forward fill. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC_ADJOINT(vz, vy, vx);
      CHECK_KERNEL_ERROR
    }

    /* Velocity-type source-gradient extraction and the M_SIGMA** PML
     * memory-variable update / grad_dbuoyancy_* accumulation all read the
     * velocity adjoint at (near-face) interior cells the fold above may
     * have just changed, so all must run after it -- see elastic_born.c's
     * own driver-side comment at the equivalent point. */
#if DW_NDIM >= 3
    LAUNCH_RECV(grad_f_z, vz, sources_z_i, n_sources_z_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_RECV(grad_f_y, vy, sources_y_i, n_sources_y_per_shot)
#endif
    LAUNCH_RECV(grad_f_x, vx, sources_x_i, n_sources_x_per_shot)

    backward_kernel_v_pml_mem<<<dimGrid_guard, dimBlock, 0, stream_compute>>>(
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
        m_sigmazzz, m_sigmazzzn, m_sigmayzy, m_sigmayzyn, m_sigmaxzx,
        m_sigmaxzxn, m_sigmayzz, m_sigmayzzn, m_sigmaxzz, m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmayyyn, m_sigmaxyy, m_sigmaxyyn, m_sigmaxyx,
        m_sigmaxyxn,
#endif
        m_sigmaxxx, m_sigmaxxxn,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t,
#endif
        dvxdbuoyancy_store_1_t,
#if DW_NDIM >= 3
        grad_dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        grad_dbuoyancy_y_shot,
#endif
        grad_dbuoyancy_x_shot,
#if DW_NDIM >= 3
        az, azh,
#endif
#if DW_NDIM >= 2
        ay, ayh,
#endif
        ax, axh, dbuoyancy_requires_grad && ((t % step_ratio) == 0),
        step_ratio);
    CHECK_KERNEL_ERROR

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
        dvxdx_store_1_t, grad_dlamb_shot, grad_dmu_shot,
#if DW_NDIM >= 3
        grad_dmu_zy_shot, grad_dmu_zx_shot,
#endif
#if DW_NDIM >= 2
        grad_dmu_yx_shot,
#endif
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, dlamb_requires_grad && ((t % step_ratio) == 0),
        dmu_requires_grad && ((t % step_ratio) == 0), step_ratio);
    CHECK_KERNEL_ERROR

    /* Guard-band extension of the stress-adjoint gathers -- see the
     * "Guard-band gathers" module comment above backward_kernel_sigma_
     * guard. Must complete before the NEXT (reverse time) iteration's
     * FUNC(fill_elastic_born_stress_bc_adjoint), which folds this
     * guard-band deposit into the interior. */
    if (has_bc) {
      backward_kernel_v_guard<<<dimGrid_guard, dimBlock, 0, stream_compute>>>(
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
          m_sigmaxzz, m_sigmaxzx, m_sigmayzz, m_sigmayzy,
#endif
#if DW_NDIM >= 2
          m_sigmaxyy, m_sigmaxyx,
#endif
#if DW_NDIM >= 3
          az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
          ay, ayh, by, byh,
#endif
          ax, axh, bx, bxh,
#if DW_NDIM >= 3
          bc_z0, bc_z1,
#endif
#if DW_NDIM >= 2
          bc_y0, bc_y1,
#endif
          bc_x0, bc_x1);
      CHECK_KERNEL_ERROR
    }

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    /* adjoint-source injection: grad_r into v/sigma at receiver sites */
#if DW_NDIM >= 3
    LAUNCH_SRC(vz, grad_r_z, receivers_z_i, n_receivers_z_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_SRC(vy, grad_r_y, receivers_y_i, n_receivers_y_per_shot)
#endif
    LAUNCH_SRC(vx, grad_r_x, receivers_x_i, n_receivers_x_per_shot)
    if (n_receivers_p_per_shot > 0) {
      dim3 const g(ceil_div(n_receivers_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      add_sources_p<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, grad_r_p + t * n_shots_h * n_receivers_p_per_shot,
          receivers_p_i, n_receivers_p_per_shot);
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
#undef SETUP_STORE_LOAD
#undef OPEN_FILE_READ

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

  if (dlamb_requires_grad && !dlamb_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dlamb, grad_dlamb_shot);
    CHECK_KERNEL_ERROR
  }
  if (dmu_requires_grad && !dmu_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu, grad_dmu_shot);
    CHECK_KERNEL_ERROR
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu_zy, grad_dmu_zy_shot);
    CHECK_KERNEL_ERROR
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu_zx, grad_dmu_zx_shot);
    CHECK_KERNEL_ERROR
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu_yx, grad_dmu_yx_shot);
    CHECK_KERNEL_ERROR
#endif
  }
  if (dbuoyancy_requires_grad && !dbuoyancy_batched_h && n_shots_h > 1) {
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbuoyancy_z, grad_dbuoyancy_z_shot);
    CHECK_KERNEL_ERROR
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbuoyancy_y, grad_dbuoyancy_y_shot);
    CHECK_KERNEL_ERROR
#endif
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbuoyancy_x, grad_dbuoyancy_x_shot);
    CHECK_KERNEL_ERROR
  }
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

/* ===========================================================================
 * Full two-field elastic-Born adjoint (backward)
 *
 * Reproduces elastic_born.c's FUNC(backward) (lines ~3325-4955): two coupled
 * adjoint fields -- a BACKGROUND adjoint field (suffix _bg) and a SCATTERED
 * adjoint field (plain names, the same field backward_sc evolves). The
 * scattered half is the plain elastic adjoint (reusing the backward_sc access
 * macros already defined above); the background half runs the same elastic
 * adjoint stencil PLUS the SS4.2 coupling driven by the scattering models
 * (dlamb/dmu/dmu_ij/dbuoyancy) and the scattered adjoint field. Per-field
 * physics == elastic.cu / backward_sc; coupling == elastic_born.c.
 * ======================================================================== */

/* Scattering-model access, background adjoint wavefield/PML-memory access,
 * and the coupled background-adjoint differencing macros (DLAMB/DMU/DMU_
 * (yx/zx/zy)/DBUOYANCY_*, VZ_BG/.../SIGMAYZ_Y_BG (+ _PML variants)) now live
 * in elastic_born_kernels.h, included above, shared verbatim with
 * elastic_born.c (see that header ~3090-3320 for the physics). */

namespace {

/* Thread-block shape and occupancy target for the FUNC(backward) (the
 * "_born" kernels below, computing both background and Born-scatter
 * gradients together) 3-D staggered-grid stencil kernels, mirroring
 * scalar.cu's 2026-08-10 tuning (see that file for the full rationale). A
 * 128-thread bound lets ptxas spend up to 255 registers/thread on these
 * kernels at accuracy >= 4, capping SM occupancy; a 32x4x4 block with a
 * matching 512-thread bound caps the allocation at 128 registers. Measured
 * on 80^3/nt=50/float32, this launcher is a net win (backward
 * accuracy 6/8: 4-20% faster on both cards; accuracy 4: roughly flat,
 * -6.7% to -0.6%, within the regression gate). The plain (non-"_born")
 * FUNC(forward) and FUNC(backward_sc) launchers above did NOT clear the
 * gate this way (forward: a strong accuracy-4 win but -4% to -10% at
 * accuracy 6/8; FUNC(backward_sc), used when only the scatter parameters
 * need gradients, was never itself benchmarked) and keep the narrow
 * default unconditionally. Accuracy 2 keeps the narrow default here too. */
#if DW_NDIM == 3 && DW_ACCURACY >= 4
#define DW_STENCIL_BLOCK_3D 32, 4, 4
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(512)
#else
#define DW_STENCIL_BLOCK_3D 32, 4, 1
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(128)
#endif

DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel_sigma_born(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
    DW_DTYPE const* __restrict const dlamb,
    DW_DTYPE const* __restrict const dmu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dmu_zy,
    DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const vz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vy_bg,
#endif
    DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmazz_bg,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmayz_bg,
    DW_DTYPE const* __restrict const sigmaxz,
    DW_DTYPE const* __restrict const sigmaxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmayy_bg,
    DW_DTYPE const* __restrict const sigmaxy,
    DW_DTYPE const* __restrict const sigmaxy_bg,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzz_bg,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzy_bg,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_vzx_bg,
    DW_DTYPE const* __restrict const m_vyz,
    DW_DTYPE const* __restrict const m_vyz_bg,
    DW_DTYPE const* __restrict const m_vxz,
    DW_DTYPE const* __restrict const m_vxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyy_bg,
    DW_DTYPE const* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_vyx_bg,
    DW_DTYPE const* __restrict const m_vxy,
    DW_DTYPE const* __restrict const m_vxy_bg,
#endif
    DW_DTYPE const* __restrict const m_vxx,
    DW_DTYPE const* __restrict const m_vxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const m_sigmazzz_bg,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const m_sigmayzy_bg,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmaxzx_bg,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzz_bg,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const m_sigmayyy_bg,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyy_bg,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE const* __restrict const m_sigmaxyx_bg,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE const* __restrict const m_sigmaxxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE* __restrict const m_sigmazzzn_bg,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE* __restrict const m_sigmayzyn_bg,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE* __restrict const m_sigmaxzxn_bg,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE* __restrict const m_sigmayzzn_bg,
    DW_DTYPE* __restrict const m_sigmaxzzn,
    DW_DTYPE* __restrict const m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE* __restrict const m_sigmayyyn_bg,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE* __restrict const m_sigmaxyyn_bg,
    DW_DTYPE* __restrict const m_sigmaxyxn,
    DW_DTYPE* __restrict const m_sigmaxyxn_bg,
#endif
    DW_DTYPE* __restrict const m_sigmaxxxn,
    DW_DTYPE* __restrict const m_sigmaxxxn_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dvzdbuoyancy,
    DW_DTYPE const* __restrict const dvzdbuoyancy_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dvydbuoyancy,
    DW_DTYPE const* __restrict const dvydbuoyancy_sc,
#endif
    DW_DTYPE const* __restrict const dvxdbuoyancy,
    DW_DTYPE const* __restrict const dvxdbuoyancy_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z,
    DW_DTYPE* __restrict const grad_dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_buoyancy_y,
    DW_DTYPE* __restrict const grad_dbuoyancy_y,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_x,
    DW_DTYPE* __restrict const grad_dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    bool const buoyancy_requires_grad, bool const dbuoyancy_requires_grad,
    int64_t const step_ratio) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
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
      DW_DTYPE const* __restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const* __restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
      DW_DTYPE const* __restrict const dlamb_shot =
          dlamb_batched ? dlamb + shot_idx * shot_numel : dlamb;
      DW_DTYPE const* __restrict const dmu_shot =
          dmu_batched ? dmu + shot_idx * shot_numel : dmu;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
      DW_DTYPE const* __restrict const dmu_zy_shot =
          dmu_batched ? dmu_zy + shot_idx * shot_numel : dmu_zy;
      DW_DTYPE const* __restrict const dmu_zx_shot =
          dmu_batched ? dmu_zx + shot_idx * shot_numel : dmu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
      DW_DTYPE const* __restrict const dmu_yx_shot =
          dmu_batched ? dmu_yx + shot_idx * shot_numel : dmu_yx;
#endif

#ifndef DW_MPML
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
#endif /* !DW_MPML */

      int64_t i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_vz_cell)(
            lamb_shot, mu_shot, mu_zy_shot, mu_zx_shot, dlamb_shot, dmu_shot,
            dmu_zy_shot, dmu_zx_shot, sigmazz, sigmayy, sigmaxx, sigmayz,
            sigmaxz, sigmazz_bg, sigmayy_bg, sigmaxx_bg, sigmayz_bg, sigmaxz_bg,
            vz, vz_bg, m_vzz, m_vzy, m_vzx, m_vzz_bg, m_vzy_bg, m_vzx_bg, bz,
            byh, bxh, dt, i, j, ny, nx, z, y, x, pml_z, pml_y, pml_x, rdz, rdy,
            rdx);
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_vy_cell)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                               mu_zy_shot,
#endif
                               mu_yx_shot, dlamb_shot, dmu_shot,
#if DW_NDIM >= 3
                               dmu_zy_shot,
#endif
                               dmu_yx_shot,
#if DW_NDIM >= 3
                               sigmazz,
#endif
                               sigmayy, sigmaxx,
#if DW_NDIM >= 3
                               sigmayz,
#endif
                               sigmaxy,
#if DW_NDIM >= 3
                               sigmazz_bg,
#endif
                               sigmayy_bg, sigmaxx_bg,
#if DW_NDIM >= 3
                               sigmayz_bg,
#endif
                               sigmaxy_bg, vy, vy_bg,
#if DW_NDIM >= 3
                               m_vyz,
#endif
                               m_vyy, m_vyx,
#if DW_NDIM >= 3
                               m_vyz_bg,
#endif
                               m_vyy_bg, m_vyx_bg,
#if DW_NDIM >= 3
                               bzh,
#endif
                               by, bxh, dt, i, j,
#if DW_NDIM >= 3
                               ny,
#endif
                               nx,
#if DW_NDIM >= 3
                               z, pml_z, rdz,
#endif
                               y, x, pml_y, pml_x, rdy, rdx);
      }
#endif

      if (x < nx - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_vx_cell)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                               mu_zx_shot,
#endif
#if DW_NDIM >= 2
                               mu_yx_shot,
#endif
                               dlamb_shot, dmu_shot,
#if DW_NDIM >= 3
                               dmu_zx_shot,
#endif
#if DW_NDIM >= 2
                               dmu_yx_shot,
#endif
#if DW_NDIM >= 3
                               sigmazz,
#endif
#if DW_NDIM >= 2
                               sigmayy,
#endif
                               sigmaxx,
#if DW_NDIM >= 3
                               sigmaxz,
#endif
#if DW_NDIM >= 2
                               sigmaxy,
#endif
#if DW_NDIM >= 3
                               sigmazz_bg,
#endif
#if DW_NDIM >= 2
                               sigmayy_bg,
#endif
                               sigmaxx_bg,
#if DW_NDIM >= 3
                               sigmaxz_bg,
#endif
#if DW_NDIM >= 2
                               sigmaxy_bg,
#endif
                               vx, vx_bg,
#if DW_NDIM >= 3
                               m_vxz,
#endif
#if DW_NDIM >= 2
                               m_vxy,
#endif
                               m_vxx,
#if DW_NDIM >= 3
                               m_vxz_bg,
#endif
#if DW_NDIM >= 2
                               m_vxy_bg,
#endif
                               m_vxx_bg,
#if DW_NDIM >= 3
                               bzh,
#endif
#if DW_NDIM >= 2
                               byh,
#endif
                               bx, dt, i, j,
#if DW_NDIM >= 3
                               ny,
#endif
#if DW_NDIM >= 2
                               nx,
#endif
#if DW_NDIM >= 3
                               z, pml_z, rdz,
#endif
#if DW_NDIM >= 2
                               y, pml_y, rdy,
#endif
                               x, pml_x, rdx);
      }

#if DW_NDIM == 3
    }
#endif
  }
}

DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel_v_born(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
    DW_DTYPE const* __restrict const dlamb,
    DW_DTYPE const* __restrict const dmu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dmu_zy,
    DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE const* __restrict const vz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vy_bg,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmazz_bg,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmayz_bg,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const sigmaxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmayy_bg,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const sigmaxy_bg,
#endif
    DW_DTYPE* __restrict const sigmaxx, DW_DTYPE* __restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzz_bg,
    DW_DTYPE* __restrict const m_vzy, DW_DTYPE* __restrict const m_vzy_bg,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vzx_bg,
    DW_DTYPE* __restrict const m_vyz, DW_DTYPE* __restrict const m_vyz_bg,
    DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict const m_vxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vyy_bg,
    DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vyx_bg,
    DW_DTYPE* __restrict const m_vxy, DW_DTYPE* __restrict const m_vxy_bg,
#endif
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const m_vxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const m_sigmazzz_bg,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const m_sigmayzy_bg,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmaxzx_bg,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzz_bg,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const m_sigmayyy_bg,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyy_bg,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE const* __restrict const m_sigmaxyx_bg,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE const* __restrict const m_sigmaxxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dvzdz_store,
    DW_DTYPE const* __restrict const dvzdz_sc_store,
    DW_DTYPE const* __restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE const* __restrict const dvzdx_plus_dvxdz_sc_store,
    DW_DTYPE const* __restrict const dvzdy_plus_dvydz_store,
    DW_DTYPE const* __restrict const dvzdy_plus_dvydz_sc_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dvydy_store,
    DW_DTYPE const* __restrict const dvydy_sc_store,
    DW_DTYPE const* __restrict const dvydx_plus_dvxdy_store,
    DW_DTYPE const* __restrict const dvydx_plus_dvxdy_sc_store,
#endif
    DW_DTYPE const* __restrict const dvxdx_store,
    DW_DTYPE const* __restrict const dvxdx_sc_store,
    DW_DTYPE* __restrict const grad_lamb, DW_DTYPE* __restrict const grad_dlamb,
    DW_DTYPE* __restrict const grad_mu, DW_DTYPE* __restrict const grad_dmu,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_mu_zy,
    DW_DTYPE* __restrict const grad_dmu_zy,
    DW_DTYPE* __restrict const grad_mu_zx,
    DW_DTYPE* __restrict const grad_dmu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_mu_yx,
    DW_DTYPE* __restrict const grad_dmu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    bool const lamb_requires_grad, bool const dlamb_requires_grad,
    bool const mu_requires_grad, bool const dmu_requires_grad,
    int64_t const step_ratio) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z) + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x) + FD_PAD;
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
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
      DW_DTYPE const* __restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const* __restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
      DW_DTYPE const* __restrict const dlamb_shot =
          dlamb_batched ? dlamb + shot_idx * shot_numel : dlamb;
      DW_DTYPE const* __restrict const dmu_shot =
          dmu_batched ? dmu + shot_idx * shot_numel : dmu;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
      DW_DTYPE const* __restrict const dmu_zy_shot =
          dmu_batched ? dmu_zy + shot_idx * shot_numel : dmu_zy;
      DW_DTYPE const* __restrict const dmu_zx_shot =
          dmu_batched ? dmu_zx + shot_idx * shot_numel : dmu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
      DW_DTYPE const* __restrict const dmu_yx_shot =
          dmu_batched ? dmu_yx + shot_idx * shot_numel : dmu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
      DW_DTYPE const* __restrict const dbuoyancy_z_shot =
          dbuoyancy_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const* __restrict const dbuoyancy_y_shot =
          dbuoyancy_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      DW_DTYPE const* __restrict const dbuoyancy_x_shot =
          dbuoyancy_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;

#ifndef DW_MPML
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
#endif /* !DW_MPML */

      int64_t const i = shot_idx * shot_numel + j;

      {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

        FUNC(backward_sigmaii_cell)(
            lamb_shot, mu_shot, dlamb_shot, dmu_shot,
#if DW_NDIM >= 3
            buoyancy_z_shot, dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
            buoyancy_y_shot, dbuoyancy_y_shot,
#endif
            buoyancy_x_shot, dbuoyancy_x_shot,
#if DW_NDIM >= 3
            vz, sigmazz, vz_bg, sigmazz_bg, m_vzz, m_vzz_bg, m_sigmazzz,
            m_sigmazzz_bg, dvzdz_store, dvzdz_sc_store,
#endif
#if DW_NDIM >= 2
            vy, sigmayy, vy_bg, sigmayy_bg, m_vyy, m_vyy_bg, m_sigmayyy,
            m_sigmayyy_bg, dvydy_store, dvydy_sc_store,
#endif
            vx, sigmaxx, vx_bg, sigmaxx_bg, m_vxx, m_vxx_bg, m_sigmaxxx,
            m_sigmaxxx_bg, dvxdx_store, dvxdx_sc_store, grad_lamb, grad_mu,
            grad_dlamb, grad_dmu, dt, step_ratio, lamb_requires_grad,
            mu_requires_grad, dlamb_requires_grad, dmu_requires_grad, i, j,
#if DW_NDIM >= 3
            ny,
#endif
#if DW_NDIM >= 2
            nx,
#endif
#if DW_NDIM >= 3
            z, pml_z, az, azh, bz, bzh, rdz,
#endif
#if DW_NDIM >= 2
            y, pml_y, ay, ayh, by, byh, rdy,
#endif
            x, pml_x, ax, axh, bx, bxh, rdx);
      }

#if DW_NDIM >= 2
      if (y < ny - FD_PAD && x < nx - FD_PAD) {
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_sigmaxy_cell)(
            mu_yx_shot, dmu_yx_shot, buoyancy_x_shot, buoyancy_y_shot,
            dbuoyancy_x_shot, dbuoyancy_y_shot, vx, vy, vx_bg, vy_bg, sigmaxy,
            sigmaxy_bg, m_vxy, m_vyx, m_vxy_bg, m_vyx_bg, m_sigmaxyy,
            m_sigmaxyx, m_sigmaxyy_bg, m_sigmaxyx_bg, dvydx_plus_dvxdy_store,
            dvydx_plus_dvxdy_sc_store, grad_mu_yx, grad_dmu_yx, dt, step_ratio,
            mu_requires_grad, dmu_requires_grad, i, j,
#if DW_NDIM >= 3
            ny, z,
#endif
            nx, y, x, pml_y, pml_x, ay, ayh, by, byh, ax, axh, bx, bxh, rdy,
            rdx);
      }
#endif

#if DW_NDIM >= 3
      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        FUNC(backward_sigmaxz_cell)(
            mu_zx_shot, dmu_zx_shot, buoyancy_x_shot, buoyancy_z_shot,
            dbuoyancy_x_shot, dbuoyancy_z_shot, vx, vz, vx_bg, vz_bg, sigmaxz,
            sigmaxz_bg, m_vxz, m_vzx, m_vxz_bg, m_vzx_bg, m_sigmaxzz,
            m_sigmaxzx, m_sigmaxzz_bg, m_sigmaxzx_bg, dvzdx_plus_dvxdz_store,
            dvzdx_plus_dvxdz_sc_store, grad_mu_zx, grad_dmu_zx, dt, step_ratio,
            mu_requires_grad, dmu_requires_grad, i, j, ny, nx, z, y, x, pml_z,
            pml_x, az, azh, bz, bzh, ax, axh, bx, bxh, rdz, rdx);
      }

      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        FUNC(backward_sigmayz_cell)(
            mu_zy_shot, dmu_zy_shot, buoyancy_y_shot, buoyancy_z_shot,
            dbuoyancy_y_shot, dbuoyancy_z_shot, vy, vz, vy_bg, vz_bg, sigmayz,
            sigmayz_bg, m_vyz, m_vzy, m_vyz_bg, m_vzy_bg, m_sigmayzz,
            m_sigmayzy, m_sigmayzz_bg, m_sigmayzy_bg, dvzdy_plus_dvydz_store,
            dvzdy_plus_dvydz_sc_store, grad_mu_zy, grad_dmu_zy, dt, step_ratio,
            mu_requires_grad, dmu_requires_grad, i, j, ny, nx, z, y, x, pml_z,
            pml_y, az, azh, bz, bzh, ay, ayh, by, byh, rdz, rdy);
      }
#endif

#if DW_NDIM == 3
    }
#endif
  }
}

/* CUDA guard-band gather kernels for the full two-field (background +
 * scattered) adjoint, mirroring backward_kernel_sigma_guard/backward_
 * kernel_v_pml_mem/backward_kernel_v_guard above (the scattered-only
 * family) but calling the fused FUNC(backward_v*_cell_guard)/FUNC(
 * backward_sigma*_cell_guard)/FUNC(backward_v*_cell_pml_mem) functions in
 * elastic_born_kernels.h, which update both adjoint fields (background
 * then scattered) in one call per cell exactly as their non-guard interior
 * counterparts (backward_kernel_sigma_born/backward_kernel_v_born above)
 * do. See the "Guard-band gathers" module comment above backward_kernel_
 * sigma_guard for the full rationale; this is that same mechanism, doubled
 * for the fused field set. */
DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel_sigma_born_guard(
    DW_DTYPE const* __restrict const lamb, DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
    DW_DTYPE const* __restrict const dlamb,
    DW_DTYPE const* __restrict const dmu,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dmu_zy,
    DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const vz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const vy_bg,
#endif
    DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmazz_bg,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmayz_bg,
    DW_DTYPE const* __restrict const sigmaxz,
    DW_DTYPE const* __restrict const sigmaxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmayy_bg,
    DW_DTYPE const* __restrict const sigmaxy,
    DW_DTYPE const* __restrict const sigmaxy_bg,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzz_bg,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzy_bg,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_vzx_bg,
    DW_DTYPE const* __restrict const m_vyz,
    DW_DTYPE const* __restrict const m_vyz_bg,
    DW_DTYPE const* __restrict const m_vxz,
    DW_DTYPE const* __restrict const m_vxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyy_bg,
    DW_DTYPE const* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_vyx_bg,
    DW_DTYPE const* __restrict const m_vxy,
    DW_DTYPE const* __restrict const m_vxy_bg,
#endif
    DW_DTYPE const* __restrict const m_vxx,
    DW_DTYPE const* __restrict const m_vxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE const* __restrict const m_sigmazzz_bg,
    DW_DTYPE* __restrict const m_sigmazzzn_bg,
    DW_DTYPE const* __restrict const m_sigmayzy_bg,
    DW_DTYPE* __restrict const m_sigmayzyn_bg,
    DW_DTYPE const* __restrict const m_sigmaxzx_bg,
    DW_DTYPE* __restrict const m_sigmaxzxn_bg,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE* __restrict const m_sigmaxzzn,
    DW_DTYPE const* __restrict const m_sigmayzz_bg,
    DW_DTYPE* __restrict const m_sigmayzzn_bg,
    DW_DTYPE const* __restrict const m_sigmaxzz_bg,
    DW_DTYPE* __restrict const m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE* __restrict const m_sigmaxyxn,
    DW_DTYPE const* __restrict const m_sigmayyy_bg,
    DW_DTYPE* __restrict const m_sigmayyyn_bg,
    DW_DTYPE const* __restrict const m_sigmaxyy_bg,
    DW_DTYPE* __restrict const m_sigmaxyyn_bg,
    DW_DTYPE const* __restrict const m_sigmaxyx_bg,
    DW_DTYPE* __restrict const m_sigmaxyxn_bg,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE* __restrict const m_sigmaxxxn,
    DW_DTYPE const* __restrict const m_sigmaxxx_bg,
    DW_DTYPE* __restrict const m_sigmaxxxn_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0, int64_t const bc_y1,
#endif
    int64_t const bc_x0, int64_t const bc_x1) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      DW_DTYPE const* __restrict const lamb_shot =
          lamb_batched ? lamb + shot_idx * shot_numel : lamb;
      DW_DTYPE const* __restrict const mu_shot =
          mu_batched ? mu + shot_idx * shot_numel : mu;
      DW_DTYPE const* __restrict const dlamb_shot =
          dlamb_batched ? dlamb + shot_idx * shot_numel : dlamb;
      DW_DTYPE const* __restrict const dmu_shot =
          dmu_batched ? dmu + shot_idx * shot_numel : dmu;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
      DW_DTYPE const* __restrict const dmu_zy_shot =
          dmu_batched ? dmu_zy + shot_idx * shot_numel : dmu_zy;
      DW_DTYPE const* __restrict const dmu_zx_shot =
          dmu_batched ? dmu_zx + shot_idx * shot_numel : dmu_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
      DW_DTYPE const* __restrict const dmu_yx_shot =
          dmu_batched ? dmu_yx + shot_idx * shot_numel : dmu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
      DW_DTYPE const* __restrict const dbuoyancy_z_shot =
          dbuoyancy_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const* __restrict const dbuoyancy_y_shot =
          dbuoyancy_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      DW_DTYPE const* __restrict const dbuoyancy_x_shot =
          dbuoyancy_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        bool const in_target = FUNC(dw_in_target)(rz, z) &&
                               FUNC(dw_in_target)(ry, y) &&
                               FUNC(dw_in_target)(rx, x);
        bool const all_std = FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(ry, y) &&
                             FUNC(dw_in_std)(rx, x);
        if (in_target && !all_std) {
          bool const pml_z = false;
          bool const pml_y = false;
          bool const pml_x = false;
          FUNC(backward_vz_cell_guard)(
              lamb_shot, mu_shot, mu_zy_shot, mu_zx_shot, dlamb_shot, dmu_shot,
              dmu_zy_shot, dmu_zx_shot, sigmazz, sigmayy, sigmaxx, sigmayz,
              sigmaxz, sigmazz_bg, sigmayy_bg, sigmaxx_bg, sigmayz_bg,
              sigmaxz_bg, vz, vz_bg, m_vzz, m_vzy, m_vzx, m_vzz_bg, m_vzy_bg,
              m_vzx_bg, m_sigmazzz, m_sigmazzzn, m_sigmayzy, m_sigmayzyn,
              m_sigmaxzx, m_sigmaxzxn, m_sigmazzz_bg, m_sigmazzzn_bg,
              m_sigmayzy_bg, m_sigmayzyn_bg, m_sigmaxzx_bg, m_sigmaxzxn_bg,
              buoyancy_z_shot[j], dbuoyancy_z_shot[j], az, azh, bz, bzh, ay,
              ayh, by, byh, ax, axh, bx, bxh, dt, i, j, nz, ny, nx, z, y, x,
              pml_z, pml_y, pml_x, rdz, rdy, rdx);
        }
      }
#endif
#if DW_NDIM >= 2
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
#if DW_NDIM >= 3
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
#endif
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        bool const in_target = FUNC(dw_in_target)(ry, y) &&
                               FUNC(dw_in_target)(rx, x)
#if DW_NDIM >= 3
                               && FUNC(dw_in_target)(rz, z)
#endif
            ;
        bool const all_std = FUNC(dw_in_std)(ry, y) && FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 3
                             && FUNC(dw_in_std)(rz, z)
#endif
            ;
        if (in_target && !all_std) {
          bool const pml_y = false;
          bool const pml_x = false;
#if DW_NDIM >= 3
          bool const pml_z = false;
#endif
          FUNC(backward_vy_cell_guard)(
              lamb_shot, mu_shot,
#if DW_NDIM >= 3
              mu_zy_shot,
#endif
              mu_yx_shot, dlamb_shot, dmu_shot,
#if DW_NDIM >= 3
              dmu_zy_shot,
#endif
              dmu_yx_shot,
#if DW_NDIM >= 3
              sigmazz,
#endif
              sigmayy, sigmaxx,
#if DW_NDIM >= 3
              sigmayz,
#endif
              sigmaxy,
#if DW_NDIM >= 3
              sigmazz_bg,
#endif
              sigmayy_bg, sigmaxx_bg,
#if DW_NDIM >= 3
              sigmayz_bg,
#endif
              sigmaxy_bg, vy, vy_bg,
#if DW_NDIM >= 3
              m_vyz,
#endif
              m_vyy, m_vyx,
#if DW_NDIM >= 3
              m_vyz_bg,
#endif
              m_vyy_bg, m_vyx_bg,
#if DW_NDIM >= 3
              m_sigmayzz, m_sigmayzzn,
#endif
              m_sigmayyy, m_sigmayyyn, m_sigmaxyx, m_sigmaxyxn,
#if DW_NDIM >= 3
              m_sigmayzz_bg, m_sigmayzzn_bg,
#endif
              m_sigmayyy_bg, m_sigmayyyn_bg, m_sigmaxyx_bg, m_sigmaxyxn_bg,
              buoyancy_y_shot[j], dbuoyancy_y_shot[j],
#if DW_NDIM >= 3
              az, azh, bz, bzh,
#endif
              ay, ayh, by, byh, ax, axh, bx, bxh, dt, i, j,
#if DW_NDIM >= 3
              nz,
#endif
              ny, nx,
#if DW_NDIM >= 3
              z, pml_z, rdz,
#endif
              y, x, pml_y, pml_x, rdy, rdx);
        }
      }
#endif
      {
#if DW_NDIM >= 3
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
#endif
#if DW_NDIM >= 2
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
#endif
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        bool const in_target = FUNC(dw_in_target)(rx, x)
#if DW_NDIM >= 2
                               && FUNC(dw_in_target)(ry, y)
#endif
#if DW_NDIM >= 3
                               && FUNC(dw_in_target)(rz, z)
#endif
            ;
        bool const all_std = FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 2
                             && FUNC(dw_in_std)(ry, y)
#endif
#if DW_NDIM >= 3
                             && FUNC(dw_in_std)(rz, z)
#endif
            ;
        if (in_target && !all_std) {
#if DW_NDIM >= 3
          bool const pml_z = false;
#endif
#if DW_NDIM >= 2
          bool const pml_y = false;
#endif
          bool const pml_x = false;
          FUNC(backward_vx_cell_guard)(lamb_shot, mu_shot,
#if DW_NDIM >= 3
                                       mu_zx_shot,
#endif
#if DW_NDIM >= 2
                                       mu_yx_shot,
#endif
                                       dlamb_shot, dmu_shot,
#if DW_NDIM >= 3
                                       dmu_zx_shot,
#endif
#if DW_NDIM >= 2
                                       dmu_yx_shot,
#endif
#if DW_NDIM >= 3
                                       sigmazz,
#endif
#if DW_NDIM >= 2
                                       sigmayy,
#endif
                                       sigmaxx,
#if DW_NDIM >= 3
                                       sigmaxz,
#endif
#if DW_NDIM >= 2
                                       sigmaxy,
#endif
#if DW_NDIM >= 3
                                       sigmazz_bg,
#endif
#if DW_NDIM >= 2
                                       sigmayy_bg,
#endif
                                       sigmaxx_bg,
#if DW_NDIM >= 3
                                       sigmaxz_bg,
#endif
#if DW_NDIM >= 2
                                       sigmaxy_bg,
#endif
                                       vx, vx_bg,
#if DW_NDIM >= 3
                                       m_vxz,
#endif
#if DW_NDIM >= 2
                                       m_vxy,
#endif
                                       m_vxx,
#if DW_NDIM >= 3
                                       m_vxz_bg,
#endif
#if DW_NDIM >= 2
                                       m_vxy_bg,
#endif
                                       m_vxx_bg,
#if DW_NDIM >= 3
                                       m_sigmaxzz, m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
                                       m_sigmaxyy, m_sigmaxyyn,
#endif
                                       m_sigmaxxx, m_sigmaxxxn,
#if DW_NDIM >= 3
                                       m_sigmaxzz_bg, m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
                                       m_sigmaxyy_bg, m_sigmaxyyn_bg,
#endif
                                       m_sigmaxxx_bg, m_sigmaxxxn_bg,
                                       buoyancy_x_shot[j], dbuoyancy_x_shot[j],
#if DW_NDIM >= 3
                                       az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
                                       ay, ayh, by, byh,
#endif
                                       ax, axh, bx, bxh, dt, i, j,
#if DW_NDIM >= 3
                                       nz,
#endif
#if DW_NDIM >= 2
                                       ny,
#endif
                                       nx,
#if DW_NDIM >= 3
                                       z, pml_z, rdz,
#endif
#if DW_NDIM >= 2
                                       y, pml_y, rdy,
#endif
                                       x, pml_x, rdx);
        }
      }

#if DW_NDIM == 3
    }
#endif
  }
}

/* Post-fold M_SIGMA** / M_SIGMA**_bg PML memory-variable update and
 * grad_buoyancy_* / grad_dbuoyancy_* accumulation for the full two-field
 * adjoint -- see the module comment above backward_kernel_v_pml_mem (the
 * scattered-only family) and FUNC(backward_vz_cell_pml_mem)'s comment in
 * elastic_born_kernels.h. Launched over the same (unextended) range FUNC(
 * forward_vz_cell) etc. used to write dv*dbuoyancy/dv*dbuoyancy_sc in the
 * forward pass. */
DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel_v_pml_mem_born(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE const* __restrict const vz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vy_bg,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmazzzn,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmayzyn,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const m_sigmaxzxn,
    DW_DTYPE const* __restrict const m_sigmazzz_bg,
    DW_DTYPE* __restrict const m_sigmazzzn_bg,
    DW_DTYPE const* __restrict const m_sigmayzy_bg,
    DW_DTYPE* __restrict const m_sigmayzyn_bg,
    DW_DTYPE const* __restrict const m_sigmaxzx_bg,
    DW_DTYPE* __restrict const m_sigmaxzxn_bg,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE* __restrict const m_sigmayzzn,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE* __restrict const m_sigmaxzzn,
    DW_DTYPE const* __restrict const m_sigmayzz_bg,
    DW_DTYPE* __restrict const m_sigmayzzn_bg,
    DW_DTYPE const* __restrict const m_sigmaxzz_bg,
    DW_DTYPE* __restrict const m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmayyyn,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE* __restrict const m_sigmaxyyn,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE* __restrict const m_sigmaxyxn,
    DW_DTYPE const* __restrict const m_sigmayyy_bg,
    DW_DTYPE* __restrict const m_sigmayyyn_bg,
    DW_DTYPE const* __restrict const m_sigmaxyy_bg,
    DW_DTYPE* __restrict const m_sigmaxyyn_bg,
    DW_DTYPE const* __restrict const m_sigmaxyx_bg,
    DW_DTYPE* __restrict const m_sigmaxyxn_bg,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE* __restrict const m_sigmaxxxn,
    DW_DTYPE const* __restrict const m_sigmaxxx_bg,
    DW_DTYPE* __restrict const m_sigmaxxxn_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dvzdbuoyancy,
    DW_DTYPE const* __restrict const dvzdbuoyancy_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dvydbuoyancy,
    DW_DTYPE const* __restrict const dvydbuoyancy_sc,
#endif
    DW_DTYPE const* __restrict const dvxdbuoyancy,
    DW_DTYPE const* __restrict const dvxdbuoyancy_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_buoyancy_z,
    DW_DTYPE* __restrict const grad_dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_buoyancy_y,
    DW_DTYPE* __restrict const grad_dbuoyancy_y,
#endif
    DW_DTYPE* __restrict const grad_buoyancy_x,
    DW_DTYPE* __restrict const grad_dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    bool const buoyancy_requires_grad_t, bool const dbuoyancy_requires_grad_t,
    int64_t const step_ratio_t) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
      DW_DTYPE const* __restrict const dbuoyancy_z_shot =
          dbuoyancy_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const* __restrict const dbuoyancy_y_shot =
          dbuoyancy_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      DW_DTYPE const* __restrict const dbuoyancy_x_shot =
          dbuoyancy_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;

#if DW_NDIM >= 3
      if (z >= FD_PAD && z < nz - FD_PAD && y >= FD_PAD &&
          y < ny - FD_PAD + 1 && x >= FD_PAD && x < nx - FD_PAD + 1) {
        bool const is_pml_z = FUNC(dw_is_pml_region)(
            z, pml_z0, MAX(pml_z0, pml_z1 - 1), nz - FD_PAD);
        bool const is_pml_y =
            FUNC(dw_is_pml_region)(y, pml_y0, pml_y1, ny - FD_PAD + 1);
        bool const is_pml_x =
            FUNC(dw_is_pml_region)(x, pml_x0, pml_x1, nx - FD_PAD + 1);
        FUNC(backward_vz_cell_pml_mem)(
            vz, vz_bg, m_sigmazzz, m_sigmazzzn, m_sigmayzy, m_sigmayzyn,
            m_sigmaxzx, m_sigmaxzxn, m_sigmazzz_bg, m_sigmazzzn_bg,
            m_sigmayzy_bg, m_sigmayzyn_bg, m_sigmaxzx_bg, m_sigmaxzxn_bg,
            buoyancy_z_shot[j], dbuoyancy_z_shot[j], grad_buoyancy_z,
            grad_dbuoyancy_z, dvzdbuoyancy, dvzdbuoyancy_sc, azh, ay, ax, dt,
            step_ratio_t, buoyancy_requires_grad_t, dbuoyancy_requires_grad_t,
            i, ny, nx, z, y, x, is_pml_z, is_pml_y, is_pml_x);
      }
#endif
#if DW_NDIM >= 2
      if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD && x < nx - FD_PAD + 1
#if DW_NDIM >= 3
          && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
      ) {
        bool const is_pml_y = FUNC(dw_is_pml_region)(
            y, pml_y0, MAX(pml_y0, pml_y1 - 1), ny - FD_PAD);
        bool const is_pml_x =
            FUNC(dw_is_pml_region)(x, pml_x0, pml_x1, nx - FD_PAD + 1);
#if DW_NDIM >= 3
        bool const is_pml_z =
            FUNC(dw_is_pml_region)(z, pml_z0, pml_z1, nz - FD_PAD + 1);
#endif
        FUNC(backward_vy_cell_pml_mem)(
            vy, vy_bg,
#if DW_NDIM >= 3
            m_sigmayzz, m_sigmayzzn, m_sigmayzz_bg, m_sigmayzzn_bg,
#endif
            m_sigmayyy, m_sigmayyyn, m_sigmaxyx, m_sigmaxyxn, m_sigmayyy_bg,
            m_sigmayyyn_bg, m_sigmaxyx_bg, m_sigmaxyxn_bg, buoyancy_y_shot[j],
            dbuoyancy_y_shot[j], grad_buoyancy_y, grad_dbuoyancy_y,
            dvydbuoyancy, dvydbuoyancy_sc,
#if DW_NDIM >= 3
            az,
#endif
            ayh, ax, dt, step_ratio_t, buoyancy_requires_grad_t,
            dbuoyancy_requires_grad_t, i,
#if DW_NDIM >= 3
            ny, z, is_pml_z,
#endif
            nx, y, x, is_pml_y, is_pml_x);
      }
#endif
      if (x >= FD_PAD && x < nx - FD_PAD
#if DW_NDIM >= 2
          && y >= FD_PAD && y < ny - FD_PAD + 1
#endif
#if DW_NDIM >= 3
          && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
      ) {
#if DW_NDIM >= 3
        bool const is_pml_z =
            FUNC(dw_is_pml_region)(z, pml_z0, pml_z1, nz - FD_PAD + 1);
#endif
#if DW_NDIM >= 2
        bool const is_pml_y =
            FUNC(dw_is_pml_region)(y, pml_y0, pml_y1, ny - FD_PAD + 1);
#endif
        bool const is_pml_x = FUNC(dw_is_pml_region)(
            x, pml_x0, MAX(pml_x0, pml_x1 - 1), nx - FD_PAD);
        FUNC(backward_vx_cell_pml_mem)(
            vx, vx_bg,
#if DW_NDIM >= 3
            m_sigmaxzz, m_sigmaxzzn, m_sigmaxzz_bg, m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
            m_sigmaxyy, m_sigmaxyyn, m_sigmaxyy_bg, m_sigmaxyyn_bg,
#endif
            m_sigmaxxx, m_sigmaxxxn, m_sigmaxxx_bg, m_sigmaxxxn_bg,
            buoyancy_x_shot[j], dbuoyancy_x_shot[j], grad_buoyancy_x,
            grad_dbuoyancy_x, dvxdbuoyancy, dvxdbuoyancy_sc,
#if DW_NDIM >= 3
            az,
#endif
#if DW_NDIM >= 2
            ay,
#endif
            axh, dt, step_ratio_t, buoyancy_requires_grad_t,
            dbuoyancy_requires_grad_t, i,
#if DW_NDIM >= 3
            ny, z, is_pml_z,
#endif
#if DW_NDIM >= 2
            nx, y, is_pml_y,
#endif
            x, is_pml_x);
      }
#if DW_NDIM == 3
    }
#endif
  }
}

/* Guard-band stress-adjoint gathers for the full two-field adjoint,
 * mirroring backward_kernel_v_guard above (the scattered-only family) but
 * calling the fused FUNC(backward_sigma*_cell_guard) functions, doubled
 * for both adjoint fields exactly as backward_kernel_v_born's own interior
 * gather is. See the module comment above backward_kernel_sigma_guard. */
DW_STENCIL_LAUNCH_BOUNDS __global__ void backward_kernel_v_born_guard(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const mu_zy,
    DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dmu_zy,
    DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE const* __restrict const vz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vy_bg,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vx_bg,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmazz_bg,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmayz_bg,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const sigmaxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmayy_bg,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const sigmaxy_bg,
#endif
    DW_DTYPE* __restrict const sigmaxx, DW_DTYPE* __restrict const sigmaxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzz_bg,
    DW_DTYPE* __restrict const m_vzy, DW_DTYPE* __restrict const m_vzy_bg,
    DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vzx_bg,
    DW_DTYPE* __restrict const m_vyz, DW_DTYPE* __restrict const m_vyz_bg,
    DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict const m_vxz_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const m_vyy_bg,
    DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vyx_bg,
    DW_DTYPE* __restrict const m_vxy, DW_DTYPE* __restrict const m_vxy_bg,
#endif
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const m_vxx_bg,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzz_bg,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const m_sigmaxzx_bg,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzz_bg,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const m_sigmayzy_bg,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyy_bg,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE const* __restrict const m_sigmaxyx_bg,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0, int64_t const bc_y1,
#endif
    int64_t const bc_x0, int64_t const bc_x1) {
#if DW_NDIM == 3
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const z = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#elif DW_NDIM == 2
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const y = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
  int64_t const shot_idx = dw_grid_index(blockIdx.z, blockDim.z, threadIdx.z);
#else /* DW_NDIM == 1 */
  int64_t const x = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  int64_t const shot_idx = dw_grid_index(blockIdx.y, blockDim.y, threadIdx.y);
#endif

#if DW_NDIM == 3
  if (z < nz && y < ny && x < nx) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; shot_idx++) {
#elif DW_NDIM == 2
  if (y < ny && x < nx && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  if (x < nx && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      DW_DTYPE const* __restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      DW_DTYPE const* __restrict const dbuoyancy_x_shot =
          dbuoyancy_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
      DW_DTYPE const* __restrict const dbuoyancy_z_shot =
          dbuoyancy_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const* __restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const* __restrict const dbuoyancy_y_shot =
          dbuoyancy_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
      DW_DTYPE const* __restrict const mu_yx_shot =
          mu_batched ? mu_yx + shot_idx * shot_numel : mu_yx;
      DW_DTYPE const* __restrict const dmu_yx_shot =
          dmu_batched ? dmu_yx + shot_idx * shot_numel : dmu_yx;
#endif
#if DW_NDIM >= 3
      DW_DTYPE const* __restrict const mu_zx_shot =
          mu_batched ? mu_zx + shot_idx * shot_numel : mu_zx;
      DW_DTYPE const* __restrict const dmu_zx_shot =
          dmu_batched ? dmu_zx + shot_idx * shot_numel : dmu_zx;
      DW_DTYPE const* __restrict const mu_zy_shot =
          mu_batched ? mu_zy + shot_idx * shot_numel : mu_zy;
      DW_DTYPE const* __restrict const dmu_zy_shot =
          dmu_batched ? dmu_zy + shot_idx * shot_numel : dmu_zy;
#endif

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
        if (FUNC(dw_in_target)(rz, z) && !FUNC(dw_in_std)(rz, z) &&
            y >= FD_PAD && y < ny - FD_PAD + 1 && x >= FD_PAD &&
            x < nx - FD_PAD + 1) {
          FUNC(backward_sigmazz_z_cell_guard)(buoyancy_z_shot, dbuoyancy_z_shot,
                                              vz, sigmazz, vz_bg, sigmazz_bg,
                                              dt, i, j, nz, ny, nx, z, rdz);
        }
      }
#endif
#if DW_NDIM >= 2
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, false);
        if (FUNC(dw_in_target)(ry, y) && !FUNC(dw_in_std)(ry, y) &&
            x >= FD_PAD && x < nx - FD_PAD + 1
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          FUNC(backward_sigmayy_y_cell_guard)(buoyancy_y_shot, dbuoyancy_y_shot,
                                              vy, sigmayy, vy_bg, sigmayy_bg,
                                              dt, i, j, ny, nx, y, rdy);
        }
      }
#endif
      {
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, false);
        if (FUNC(dw_in_target)(rx, x) && !FUNC(dw_in_std)(rx, x)
#if DW_NDIM >= 2
            && y >= FD_PAD && y < ny - FD_PAD + 1
#endif
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          FUNC(backward_sigmaxx_x_cell_guard)(buoyancy_x_shot, dbuoyancy_x_shot,
                                              vx, sigmaxx, vx_bg, sigmaxx_bg,
                                              dt, i, j,
#if DW_NDIM >= 3
                                              ny,
#endif
                                              nx, x, rdx);
        }
      }
#if DW_NDIM >= 2
      {
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        bool const in_target =
            FUNC(dw_in_target)(ry, y) && FUNC(dw_in_target)(rx, x);
        bool const all_std = FUNC(dw_in_std)(ry, y) && FUNC(dw_in_std)(rx, x);
        if (in_target && !all_std
#if DW_NDIM >= 3
            && z >= FD_PAD && z < nz - FD_PAD + 1
#endif
        ) {
          bool const pml_y = false;
          bool const pml_x = false;
          FUNC(backward_sigmaxy_cell_guard)(
              mu_yx_shot, dmu_yx_shot, buoyancy_x_shot, buoyancy_y_shot,
              dbuoyancy_x_shot, dbuoyancy_y_shot, vx, vy, vx_bg, vy_bg, sigmaxy,
              sigmaxy_bg, m_vxy, m_vyx, m_vxy_bg, m_vyx_bg, m_sigmaxyy,
              m_sigmaxyx, m_sigmaxyy_bg, m_sigmaxyx_bg, dt, i, j,
#if DW_NDIM >= 3
              nz, z,
#endif
              ny, nx, y, x, pml_y, pml_x, ay, ayh, by, byh, ax, axh, bx, bxh,
              rdy, rdx);
        }
      }
#endif
#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        FUNC(dw_ext_range)
        const rx = FUNC(dw_compute_ext_range)(nx, bc_x0, bc_x1, true);
        bool const in_target =
            FUNC(dw_in_target)(rz, z) && FUNC(dw_in_target)(rx, x);
        bool const all_std = FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(rx, x);
        if (in_target && !all_std && y >= FD_PAD && y < ny - FD_PAD + 1) {
          bool const pml_z = false;
          bool const pml_x = false;
          FUNC(backward_sigmaxz_cell_guard)(
              mu_zx_shot, dmu_zx_shot, buoyancy_x_shot, buoyancy_z_shot,
              dbuoyancy_x_shot, dbuoyancy_z_shot, vx, vz, vx_bg, vz_bg, sigmaxz,
              sigmaxz_bg, m_vxz, m_vzx, m_vxz_bg, m_vzx_bg, m_sigmaxzz,
              m_sigmaxzx, m_sigmaxzz_bg, m_sigmaxzx_bg, dt, i, j, nz, ny, nx, z,
              y, x, pml_z, pml_x, az, azh, bz, bzh, ax, axh, bx, bxh, rdz, rdx);
        }
      }
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, true);
        FUNC(dw_ext_range)
        const ry = FUNC(dw_compute_ext_range)(ny, bc_y0, bc_y1, true);
        bool const in_target =
            FUNC(dw_in_target)(rz, z) && FUNC(dw_in_target)(ry, y);
        bool const all_std = FUNC(dw_in_std)(rz, z) && FUNC(dw_in_std)(ry, y);
        if (in_target && !all_std && x >= FD_PAD && x < nx - FD_PAD + 1) {
          bool const pml_z = false;
          bool const pml_y = false;
          FUNC(backward_sigmayz_cell_guard)(
              mu_zy_shot, dmu_zy_shot, buoyancy_y_shot, buoyancy_z_shot,
              dbuoyancy_y_shot, dbuoyancy_z_shot, vy, vz, vy_bg, vz_bg, sigmayz,
              sigmayz_bg, m_vyz, m_vzy, m_vyz_bg, m_vzy_bg, m_sigmayzz,
              m_sigmayzy, m_sigmayzz_bg, m_sigmayzy_bg, dt, i, j, nz, ny, nx, z,
              y, x, pml_z, pml_y, az, azh, bz, bzh, ay, ayh, by, byh, rdz, rdy);
        }
      }
#endif

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
    int FUNC(backward)(
        /* background models */
        DW_DTYPE const* __restrict const lamb,
        DW_DTYPE const* __restrict const mu,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const mu_zy,
        DW_DTYPE const* __restrict const mu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const mu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const buoyancy_y,
#endif
        DW_DTYPE const* __restrict const buoyancy_x,
        /* scattering models */
        DW_DTYPE const* __restrict const dlamb,
        DW_DTYPE const* __restrict const dmu,
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const dmu_zy,
        DW_DTYPE const* __restrict const dmu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const dmu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
        DW_DTYPE const* __restrict const dbuoyancy_x,
/* background data residuals */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const grad_r_bg_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const grad_r_bg_y,
#endif
        DW_DTYPE const* __restrict const grad_r_bg_x,
        DW_DTYPE const* __restrict const grad_r_bg_p,
/* scattered data residuals */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const grad_r_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const grad_r_y,
#endif
        DW_DTYPE const* __restrict const grad_r_x,
        DW_DTYPE const* __restrict const grad_r_p,
/* background adjoint wavefields */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const vz_bg, DW_DTYPE* __restrict const sigmazz_bg,
        DW_DTYPE* __restrict const sigmayz_bg,
        DW_DTYPE* __restrict const sigmaxz_bg,
        DW_DTYPE* __restrict const m_vzz_bg,
        DW_DTYPE* __restrict const m_vzy_bg,
        DW_DTYPE* __restrict const m_vzx_bg,
        DW_DTYPE* __restrict const m_vyz_bg,
        DW_DTYPE* __restrict const m_vxz_bg, DW_DTYPE* __restrict m_sigmazzz_bg,
        DW_DTYPE* __restrict m_sigmayzy_bg, DW_DTYPE* __restrict m_sigmaxzx_bg,
        DW_DTYPE* __restrict m_sigmayzz_bg, DW_DTYPE* __restrict m_sigmaxzz_bg,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const vy_bg, DW_DTYPE* __restrict const sigmayy_bg,
        DW_DTYPE* __restrict const sigmaxy_bg,
        DW_DTYPE* __restrict const m_vyy_bg,
        DW_DTYPE* __restrict const m_vyx_bg,
        DW_DTYPE* __restrict const m_vxy_bg, DW_DTYPE* __restrict m_sigmayyy_bg,
        DW_DTYPE* __restrict m_sigmaxyy_bg, DW_DTYPE* __restrict m_sigmaxyx_bg,
#endif
        DW_DTYPE* __restrict const vx_bg, DW_DTYPE* __restrict const sigmaxx_bg,
        DW_DTYPE* __restrict const m_vxx_bg, DW_DTYPE* __restrict m_sigmaxxx_bg,
/* scattered adjoint wavefields */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
        DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const sigmaxz,
        DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const m_vzy,
        DW_DTYPE* __restrict const m_vzx, DW_DTYPE* __restrict const m_vyz,
        DW_DTYPE* __restrict const m_vxz, DW_DTYPE* __restrict m_sigmazzz,
        DW_DTYPE* __restrict m_sigmayzy, DW_DTYPE* __restrict m_sigmaxzx,
        DW_DTYPE* __restrict m_sigmayzz, DW_DTYPE* __restrict m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
        DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vyy,
        DW_DTYPE* __restrict const m_vyx, DW_DTYPE* __restrict const m_vxy,
        DW_DTYPE* __restrict m_sigmayyy, DW_DTYPE* __restrict m_sigmaxyy,
        DW_DTYPE* __restrict m_sigmaxyx,
#endif
        DW_DTYPE* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
        DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict m_sigmaxxx,
/* background m_sigma* aux ("n") scratch */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict m_sigmazzzn_bg,
        DW_DTYPE* __restrict m_sigmayzyn_bg,
        DW_DTYPE* __restrict m_sigmaxzxn_bg,
        DW_DTYPE* __restrict m_sigmayzzn_bg,
        DW_DTYPE* __restrict m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict m_sigmayyyn_bg,
        DW_DTYPE* __restrict m_sigmaxyyn_bg,
        DW_DTYPE* __restrict m_sigmaxyxn_bg,
#endif
        DW_DTYPE* __restrict m_sigmaxxxn_bg,
/* scattered m_sigma* aux ("n") scratch */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict m_sigmazzzn, DW_DTYPE* __restrict m_sigmayzyn,
        DW_DTYPE* __restrict m_sigmaxzxn, DW_DTYPE* __restrict m_sigmayzzn,
        DW_DTYPE* __restrict m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict m_sigmayyyn, DW_DTYPE* __restrict m_sigmaxyyn,
        DW_DTYPE* __restrict m_sigmaxyxn,
#endif
        DW_DTYPE* __restrict m_sigmaxxxn,
/* background gradient stores (5-param groups) */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
        DW_DTYPE* __restrict const dvzdbuoyancy_store_1b,
        void* __restrict const dvzdbuoyancy_store_2,
        void* __restrict const dvzdbuoyancy_store_3,
        char const* __restrict const* __restrict const dvzdbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvzdz_store_1,
        DW_DTYPE* __restrict const dvzdz_store_1b,
        void* __restrict const dvzdz_store_2,
        void* __restrict const dvzdz_store_3,
        char const* __restrict const* __restrict const dvzdz_filenames_ptr,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1b,
        void* __restrict const dvzdx_plus_dvxdz_store_2,
        void* __restrict const dvzdx_plus_dvxdz_store_3,
        char const* __restrict const* __restrict const dvzdx_plus_dvxdz_filenames_ptr,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1b,
        void* __restrict const dvzdy_plus_dvydz_store_2,
        void* __restrict const dvzdy_plus_dvydz_store_3,
        char const* __restrict const* __restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydbuoyancy_store_1,
        DW_DTYPE* __restrict const dvydbuoyancy_store_1b,
        void* __restrict const dvydbuoyancy_store_2,
        void* __restrict const dvydbuoyancy_store_3,
        char const* __restrict const* __restrict const dvydbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvydy_store_1,
        DW_DTYPE* __restrict const dvydy_store_1b,
        void* __restrict const dvydy_store_2,
        void* __restrict const dvydy_store_3,
        char const* __restrict const* __restrict const dvydy_filenames_ptr,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1b,
        void* __restrict const dvydx_plus_dvxdy_store_2,
        void* __restrict const dvydx_plus_dvxdy_store_3,
        char const* __restrict const* __restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
        DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
        DW_DTYPE* __restrict const dvxdbuoyancy_store_1b,
        void* __restrict const dvxdbuoyancy_store_2,
        void* __restrict const dvxdbuoyancy_store_3,
        char const* __restrict const* __restrict const dvxdbuoyancy_filenames_ptr,
        DW_DTYPE* __restrict const dvxdx_store_1,
        DW_DTYPE* __restrict const dvxdx_store_1b,
        void* __restrict const dvxdx_store_2,
        void* __restrict const dvxdx_store_3,
        char const* __restrict const* __restrict const dvxdx_filenames_ptr,
/* scattered gradient stores (5-param groups) */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const dvzdbuoyancy_sc_store_1,
        DW_DTYPE* __restrict const dvzdbuoyancy_sc_store_1b,
        void* __restrict const dvzdbuoyancy_sc_store_2,
        void* __restrict const dvzdbuoyancy_sc_store_3,
        char const* __restrict const* __restrict const dvzdbuoyancy_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvzdz_sc_store_1,
        DW_DTYPE* __restrict const dvzdz_sc_store_1b,
        void* __restrict const dvzdz_sc_store_2,
        void* __restrict const dvzdz_sc_store_3,
        char const* __restrict const* __restrict const dvzdz_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_sc_store_1,
        DW_DTYPE* __restrict const dvzdx_plus_dvxdz_sc_store_1b,
        void* __restrict const dvzdx_plus_dvxdz_sc_store_2,
        void* __restrict const dvzdx_plus_dvxdz_sc_store_3,
        char const* __restrict const* __restrict const dvzdx_plus_dvxdz_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_sc_store_1,
        DW_DTYPE* __restrict const dvzdy_plus_dvydz_sc_store_1b,
        void* __restrict const dvzdy_plus_dvydz_sc_store_2,
        void* __restrict const dvzdy_plus_dvydz_sc_store_3,
        char const* __restrict const* __restrict const dvzdy_plus_dvydz_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const dvydbuoyancy_sc_store_1,
        DW_DTYPE* __restrict const dvydbuoyancy_sc_store_1b,
        void* __restrict const dvydbuoyancy_sc_store_2,
        void* __restrict const dvydbuoyancy_sc_store_3,
        char const* __restrict const* __restrict const dvydbuoyancy_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvydy_sc_store_1,
        DW_DTYPE* __restrict const dvydy_sc_store_1b,
        void* __restrict const dvydy_sc_store_2,
        void* __restrict const dvydy_sc_store_3,
        char const* __restrict const* __restrict const dvydy_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_sc_store_1,
        DW_DTYPE* __restrict const dvydx_plus_dvxdy_sc_store_1b,
        void* __restrict const dvydx_plus_dvxdy_sc_store_2,
        void* __restrict const dvydx_plus_dvxdy_sc_store_3,
        char const* __restrict const* __restrict const dvydx_plus_dvxdy_sc_filenames_ptr,
#endif
        DW_DTYPE* __restrict const dvxdbuoyancy_sc_store_1,
        DW_DTYPE* __restrict const dvxdbuoyancy_sc_store_1b,
        void* __restrict const dvxdbuoyancy_sc_store_2,
        void* __restrict const dvxdbuoyancy_sc_store_3,
        char const* __restrict const* __restrict const dvxdbuoyancy_sc_filenames_ptr,
        DW_DTYPE* __restrict const dvxdx_sc_store_1,
        DW_DTYPE* __restrict const dvxdx_sc_store_1b,
        void* __restrict const dvxdx_sc_store_2,
        void* __restrict const dvxdx_sc_store_3,
        char const* __restrict const* __restrict const dvxdx_sc_filenames_ptr,
        /* background model gradients (combine target) */
        DW_DTYPE* __restrict const grad_lamb,
        DW_DTYPE* __restrict const grad_mu,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_mu_zy,
        DW_DTYPE* __restrict const grad_mu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_mu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_buoyancy_y,
#endif
        DW_DTYPE* __restrict const grad_buoyancy_x,
        /* scattering model gradients (combine target) */
        DW_DTYPE* __restrict const grad_dlamb,
        DW_DTYPE* __restrict const grad_dmu,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dmu_zy,
        DW_DTYPE* __restrict const grad_dmu_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dmu_yx,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dbuoyancy_y,
#endif
        DW_DTYPE* __restrict const grad_dbuoyancy_x,
        /* background model gradient per-shot reduction buffers */
        DW_DTYPE* __restrict const grad_lamb_shot,
        DW_DTYPE* __restrict const grad_mu_shot,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_mu_zy_shot,
        DW_DTYPE* __restrict const grad_mu_zx_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_mu_yx_shot,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_buoyancy_y_shot,
#endif
        DW_DTYPE* __restrict const grad_buoyancy_x_shot,
        /* scattering model gradient per-shot reduction buffers */
        DW_DTYPE* __restrict const grad_dlamb_shot,
        DW_DTYPE* __restrict const grad_dmu_shot,
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dmu_zy_shot,
        DW_DTYPE* __restrict const grad_dmu_zx_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dmu_yx_shot,
#endif
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_dbuoyancy_y_shot,
#endif
        DW_DTYPE* __restrict const grad_dbuoyancy_x_shot,
/* background source-amplitude gradients */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_f_bg_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_f_bg_y,
#endif
        DW_DTYPE* __restrict const grad_f_bg_x,
        DW_DTYPE* __restrict const grad_f_bg_p,
/* scattered source-amplitude gradients */
#if DW_NDIM >= 3
        DW_DTYPE* __restrict const grad_f_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE* __restrict const grad_f_y,
#endif
        DW_DTYPE* __restrict const grad_f_x,
        DW_DTYPE* __restrict const grad_f_p,
/* PML profiles */
#if DW_NDIM >= 3
        DW_DTYPE const* __restrict const az,
        DW_DTYPE const* __restrict const bz,
        DW_DTYPE const* __restrict const azh,
        DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const* __restrict const ay,
        DW_DTYPE const* __restrict const by,
        DW_DTYPE const* __restrict const ayh,
        DW_DTYPE const* __restrict const byh,
#endif
        DW_DTYPE const* __restrict const ax,
        DW_DTYPE const* __restrict const bx,
        DW_DTYPE const* __restrict const axh,
        DW_DTYPE const* __restrict const bxh,
/* shared source indices */
#if DW_NDIM >= 3
        int64_t const* __restrict const sources_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const sources_y_i,
#endif
        int64_t const* __restrict const sources_x_i,
        int64_t const* __restrict const sources_p_i,
/* background receiver indices */
#if DW_NDIM >= 3
        int64_t const* __restrict const receivers_bg_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const receivers_bg_y_i,
#endif
        int64_t const* __restrict const receivers_bg_x_i,
        int64_t const* __restrict const receivers_bg_p_i,
/* scattered receiver indices */
#if DW_NDIM >= 3
        int64_t const* __restrict const receivers_z_i,
#endif
#if DW_NDIM >= 2
        int64_t const* __restrict const receivers_y_i,
#endif
        int64_t const* __restrict const receivers_x_i,
        int64_t const* __restrict const receivers_p_i,
/* scalars */
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
        int64_t const n_sources_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_per_shot,
#endif
        int64_t const n_sources_x_per_shot, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
        int64_t const n_sources_z_sc_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_sc_per_shot,
#endif
        int64_t const n_sources_x_sc_per_shot,
        int64_t const n_sources_p_sc_per_shot,
#if DW_NDIM >= 3
        int64_t const n_receivers_bg_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_bg_y_per_shot,
#endif
        int64_t const n_receivers_bg_x_per_shot,
        int64_t const n_receivers_bg_p_per_shot,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_per_shot,
#endif
        int64_t const n_receivers_x_per_shot,
        int64_t const n_receivers_p_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const lamb_batched_h,
        bool const mu_batched_h, bool const buoyancy_batched_h,
        bool const lamb_requires_grad, bool const mu_requires_grad,
        bool const buoyancy_requires_grad, bool const dlamb_requires_grad,
        bool const dmu_requires_grad, bool const dbuoyancy_requires_grad,
        bool const dlamb_batched_h, bool const dmu_batched_h,
        bool const dbuoyancy_batched_h, bool const storage_compression,
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
        int64_t const pml_x1_h,
#if DW_NDIM >= 3
        int64_t const bc_z0,
#endif
#if DW_NDIM >= 2
        int64_t const bc_y0,
#endif
        int64_t const bc_x0,
#if DW_NDIM >= 3
        int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
        int64_t const bc_y1,
#endif
        int64_t const bc_x1, int64_t const device,
        cudaStream_t stream_compute) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio <= 0) return 1;
    if ((lamb_requires_grad) && (grad_lamb == NULL)) return 1;
    if ((mu_requires_grad) && (grad_mu == NULL)) return 1;
#if DW_NDIM >= 3
    if ((mu_requires_grad) && (grad_mu_zy == NULL || grad_mu_zx == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((mu_requires_grad) && (grad_mu_yx == NULL)) return 1;
#endif
#if DW_NDIM >= 3
    if ((buoyancy_requires_grad) && (grad_buoyancy_z == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((buoyancy_requires_grad) && (grad_buoyancy_y == NULL)) return 1;
#endif
    if ((buoyancy_requires_grad) && (grad_buoyancy_x == NULL)) return 1;
    if ((dlamb_requires_grad) && (grad_dlamb == NULL)) return 1;
    if ((dmu_requires_grad) && (grad_dmu == NULL)) return 1;
#if DW_NDIM >= 3
    if ((dmu_requires_grad) && (grad_dmu_zy == NULL || grad_dmu_zx == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((dmu_requires_grad) && (grad_dmu_yx == NULL)) return 1;
#endif
#if DW_NDIM >= 3
    if ((dbuoyancy_requires_grad) && (grad_dbuoyancy_z == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((dbuoyancy_requires_grad) && (grad_dbuoyancy_y == NULL)) return 1;
#endif
    if ((dbuoyancy_requires_grad) && (grad_dbuoyancy_x == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_bg_z_per_shot > 0) && (receivers_bg_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_bg_y_per_shot > 0) && (receivers_bg_y_i == NULL)) return 1;
#endif
    if ((n_receivers_bg_x_per_shot > 0) && (receivers_bg_x_i == NULL)) return 1;
    if ((n_receivers_bg_p_per_shot > 0) && (receivers_bg_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot > 0 || n_sources_z_sc_per_shot > 0) &&
        sources_z_i == NULL)
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot > 0 || n_sources_y_sc_per_shot > 0) &&
        sources_y_i == NULL)
      return 1;
#endif
    if ((n_sources_x_per_shot > 0 || n_sources_x_sc_per_shot > 0) &&
        sources_x_i == NULL)
      return 1;
    if ((n_sources_p_per_shot > 0 || n_sources_p_sc_per_shot > 0) &&
        sources_p_i == NULL)
      return 1;
  }

#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(nz_h - 2LL * FD_PAD + 1, dimBlock.z);
#elif DW_NDIM == 2
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
  dim3 const dimBlock(32, 4, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(ny_h - 2LL * FD_PAD + 1, dimBlock.y);
  unsigned int const gridz = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
  dim3 const dimBlock(128, 1, 1);
  unsigned int const gridx = ceil_div(nx_h - 2LL * FD_PAD + 1, dimBlock.x);
  unsigned int const gridy = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz = 1;
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
#elif DW_NDIM == 2
  unsigned int const gridx_g = ceil_div(nx_h, dimBlock.x);
  unsigned int const gridy_g = ceil_div(ny_h, dimBlock.y);
  unsigned int const gridz_g = ceil_div(n_shots_h, dimBlock.z);
#else /* DW_NDIM == 1 */
  unsigned int const gridx_g = ceil_div(nx_h, dimBlock.x);
  unsigned int const gridy_g = ceil_div(n_shots_h, dimBlock.y);
  unsigned int const gridz_g = 1;
#endif
  dim3 const dimGrid_guard(gridx_g, gridy_g, gridz_g);
  dim3 const dimBlock_src_rcv(32, 1, 1);

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device_guard.selected_device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(dt_h,
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
                         pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
                         pml_y0_h, pml_y1_h,
#endif
                         pml_x0_h, pml_x1_h, lamb_batched_h, mu_batched_h,
                         buoyancy_batched_h, dlamb_batched_h, dmu_batched_h,
                         dbuoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

#define OPEN_FILE_READ(name, grad_cond)                           \
  ScopedFile fp_##name;                                           \
  file_group.add(fp_##name);                                      \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {              \
    if (!fp_##name.open(name##_filenames_ptr[0], "rb")) return 1; \
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (lamb_requires_grad || mu_requires_grad || buoyancy_requires_grad ||
       dlamb_requires_grad || dmu_requires_grad || dbuoyancy_requires_grad);

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
  OPEN_FILE_READ(dvzdbuoyancy,
                 buoyancy_requires_grad || dbuoyancy_requires_grad)
  OPEN_FILE_READ(dvzdbuoyancy_sc, buoyancy_requires_grad)
  OPEN_FILE_READ(dvzdz, lamb_requires_grad || mu_requires_grad ||
                            dlamb_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvzdz_sc, lamb_requires_grad || mu_requires_grad)
  OPEN_FILE_READ(dvzdx_plus_dvxdz, mu_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvzdx_plus_dvxdz_sc, mu_requires_grad)
  OPEN_FILE_READ(dvzdy_plus_dvydz, mu_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvzdy_plus_dvydz_sc, mu_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(dvydbuoyancy,
                 buoyancy_requires_grad || dbuoyancy_requires_grad)
  OPEN_FILE_READ(dvydbuoyancy_sc, buoyancy_requires_grad)
  OPEN_FILE_READ(dvydy, lamb_requires_grad || mu_requires_grad ||
                            dlamb_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvydy_sc, lamb_requires_grad || mu_requires_grad)
  OPEN_FILE_READ(dvydx_plus_dvxdy, mu_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvydx_plus_dvxdy_sc, mu_requires_grad)
#endif
  OPEN_FILE_READ(dvxdbuoyancy,
                 buoyancy_requires_grad || dbuoyancy_requires_grad)
  OPEN_FILE_READ(dvxdbuoyancy_sc, buoyancy_requires_grad)
  OPEN_FILE_READ(dvxdx, lamb_requires_grad || mu_requires_grad ||
                            dlamb_requires_grad || dmu_requires_grad)
  OPEN_FILE_READ(dvxdx_sc, lamb_requires_grad || mu_requires_grad)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio;

#define SETUP_STORE_LOAD(name, grad_cond)                                 \
  DW_DTYPE const* __restrict name##_store_1_t =                           \
      name##_store_1 != NULL                                              \
          ? (name##_store_1 +                                             \
             ((storage_mode == STORAGE_DEVICE && !storage_compression)    \
                  ? (t / step_ratio) * n_shots_h * shot_numel_h           \
                  : 0))                                                   \
          : NULL;                                                         \
  void* __restrict const name##_store_2_t =                               \
      name##_store_2 != NULL                                              \
          ? ((uint8_t*)name##_store_2 +                                   \
             (storage_mode == STORAGE_DEVICE && storage_compression       \
                  ? (t / step_ratio) * n_shots_h * shot_bytes_comp        \
                  : 0))                                                   \
          : NULL;                                                         \
  void* __restrict const name##_store_3_t =                               \
      name##_store_3 != NULL                                              \
          ? ((uint8_t*)name##_store_3 +                                   \
             (storage_mode == STORAGE_CPU                                 \
                  ? (t / step_ratio) * n_shots_h *                        \
                        (storage_compression ? shot_bytes_comp            \
                                             : shot_bytes_uncomp)         \
                  : 0))                                                   \
          : NULL;                                                         \
  bool const name##_cond = (grad_cond) && ((t % step_ratio) == 0);        \
  if (name##_cond) {                                                      \
    if (use_double_buffering && step_idx % 2 != 0) {                      \
      name##_store_1_t = name##_store_1b;                                 \
    }                                                                     \
    if (STORAGE_FUNC(load_snapshot_gpu)(                                  \
            (void*)name##_store_1_t, name##_store_2_t, name##_store_3_t,  \
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
    SETUP_STORE_LOAD(dvzdbuoyancy,
                     buoyancy_requires_grad || dbuoyancy_requires_grad)
    SETUP_STORE_LOAD(dvzdbuoyancy_sc, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvzdz, lamb_requires_grad || mu_requires_grad ||
                                dlamb_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvzdz_sc, lamb_requires_grad || mu_requires_grad)
    SETUP_STORE_LOAD(dvzdx_plus_dvxdz, mu_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvzdx_plus_dvxdz_sc, mu_requires_grad)
    SETUP_STORE_LOAD(dvzdy_plus_dvydz, mu_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvzdy_plus_dvydz_sc, mu_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(dvydbuoyancy,
                     buoyancy_requires_grad || dbuoyancy_requires_grad)
    SETUP_STORE_LOAD(dvydbuoyancy_sc, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvydy, lamb_requires_grad || mu_requires_grad ||
                                dlamb_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvydy_sc, lamb_requires_grad || mu_requires_grad)
    SETUP_STORE_LOAD(dvydx_plus_dvxdy, mu_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvydx_plus_dvxdy_sc, mu_requires_grad)
#endif
    SETUP_STORE_LOAD(dvxdbuoyancy,
                     buoyancy_requires_grad || dbuoyancy_requires_grad)
    SETUP_STORE_LOAD(dvxdbuoyancy_sc, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvxdx, lamb_requires_grad || mu_requires_grad ||
                                dlamb_requires_grad || dmu_requires_grad)
    SETUP_STORE_LOAD(dvxdx_sc, lamb_requires_grad || mu_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    /* Refill the (adjoint) stress guard ring of each adjoint field
     * independently (background then scattered) -- the discrete adjoint of
     * forward's velocity-update kernel (which read the stress ghost) reads
     * it here too, before the adjoint velocity update and before the
     * source-gradient extraction just below (the adjoint counterpart of
     * forward's pressure receiver record); mirrors elastic_born.c's own
     * backward stress-fill placement and elastic.cu's own backward
     * FILL_P_BC placement (self-adjointness of the mirror-fill operator,
     * applied here to both the background (_bg) and scattered (plain)
     * adjoint fields). Unlike acoustic's pressure fill, elastic_born's
     * multi-field stress fill is not self-adjoint (see the module comment
     * above launch_fill_elastic_born_stress_bc_adjoint), so this applies
     * that fill's true transpose rather than reapplying the forward fill;
     * mirrors elastic_born.c's own backward fix. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_BORN_STRESS_BC_ADJOINT(sigmazz_bg, sigmayz_bg,
                                                 sigmaxz_bg, sigmayy_bg,
                                                 sigmaxy_bg, sigmaxx_bg);
      CHECK_KERNEL_ERROR
      LAUNCH_FILL_ELASTIC_BORN_STRESS_BC_ADJOINT(sigmazz, sigmayz, sigmaxz,
                                                 sigmayy, sigmaxy, sigmaxx);
      CHECK_KERNEL_ERROR
    }

    /* pressure adjoint-source record (background then scattered) */
    if (n_sources_p_per_shot > 0) {
      dim3 const g(ceil_div(n_sources_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      record_pressure_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_bg_p + t * n_shots_h * n_sources_p_per_shot,
#if DW_NDIM >= 3
          sigmazz_bg,
#endif
#if DW_NDIM >= 2
          sigmayy_bg,
#endif
          sigmaxx_bg, sources_p_i, n_sources_p_per_shot);
      CHECK_KERNEL_ERROR
    }
    if (n_sources_p_sc_per_shot > 0) {
      dim3 const g(ceil_div(n_sources_p_sc_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      record_pressure_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_p + t * n_shots_h * n_sources_p_sc_per_shot,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, sources_p_i, n_sources_p_sc_per_shot);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_sigma_born<<<dimGrid, dimBlock, 0, stream_compute>>>(
        lamb, mu,
#if DW_NDIM >= 3
        mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
        mu_yx,
#endif
        dlamb, dmu,
#if DW_NDIM >= 3
        dmu_zy, dmu_zx,
#endif
#if DW_NDIM >= 2
        dmu_yx,
#endif
#if DW_NDIM >= 3
        buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y, dbuoyancy_y,
#endif
        buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
        vz, vz_bg,
#endif
#if DW_NDIM >= 2
        vy, vy_bg,
#endif
        vx, vx_bg,
#if DW_NDIM >= 3
        sigmazz, sigmazz_bg, sigmayz, sigmayz_bg, sigmaxz, sigmaxz_bg,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmayy_bg, sigmaxy, sigmaxy_bg,
#endif
        sigmaxx, sigmaxx_bg,
#if DW_NDIM >= 3
        m_vzz, m_vzz_bg, m_vzy, m_vzy_bg, m_vzx, m_vzx_bg, m_vyz, m_vyz_bg,
        m_vxz, m_vxz_bg,
#endif
#if DW_NDIM >= 2
        m_vyy, m_vyy_bg, m_vyx, m_vyx_bg, m_vxy, m_vxy_bg,
#endif
        m_vxx, m_vxx_bg,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmazzz_bg, m_sigmayzy, m_sigmayzy_bg, m_sigmaxzx,
        m_sigmaxzx_bg, m_sigmayzz, m_sigmayzz_bg, m_sigmaxzz, m_sigmaxzz_bg,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmayyy_bg, m_sigmaxyy, m_sigmaxyy_bg, m_sigmaxyx,
        m_sigmaxyx_bg,
#endif
        m_sigmaxxx, m_sigmaxxx_bg,
#if DW_NDIM >= 3
        m_sigmazzzn, m_sigmazzzn_bg, m_sigmayzyn, m_sigmayzyn_bg, m_sigmaxzxn,
        m_sigmaxzxn_bg, m_sigmayzzn, m_sigmayzzn_bg, m_sigmaxzzn,
        m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
        m_sigmayyyn, m_sigmayyyn_bg, m_sigmaxyyn, m_sigmaxyyn_bg, m_sigmaxyxn,
        m_sigmaxyxn_bg,
#endif
        m_sigmaxxxn, m_sigmaxxxn_bg,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t, dvzdbuoyancy_sc_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t, dvydbuoyancy_sc_store_1_t,
#endif
        dvxdbuoyancy_store_1_t, dvxdbuoyancy_sc_store_1_t,
#if DW_NDIM >= 3
        grad_buoyancy_z_shot, grad_dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        grad_buoyancy_y_shot, grad_dbuoyancy_y_shot,
#endif
        grad_buoyancy_x_shot, grad_dbuoyancy_x_shot,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, buoyancy_requires_grad && ((t % step_ratio) == 0),
        dbuoyancy_requires_grad && ((t % step_ratio) == 0), step_ratio);
    CHECK_KERNEL_ERROR

    /* Guard-band extension of the velocity-adjoint gathers, fused over both
     * adjoint fields -- see the "Guard-band gathers" module comment above
     * backward_kernel_sigma_guard. Must run (and complete, via the shared
     * stream_compute ordering) before FUNC(fill_elastic_born_velocity_bc_
     * adjoint)'s kernels just below, which fold this guard-band deposit
     * into the interior. */
    if (has_bc) {
      backward_kernel_sigma_born_guard<<<dimGrid_guard, dimBlock, 0,
                                         stream_compute>>>(
          lamb, mu,
#if DW_NDIM >= 3
          mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
          mu_yx,
#endif
          dlamb, dmu,
#if DW_NDIM >= 3
          dmu_zy, dmu_zx,
#endif
#if DW_NDIM >= 2
          dmu_yx,
#endif
#if DW_NDIM >= 3
          buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
          buoyancy_y, dbuoyancy_y,
#endif
          buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
          vz, vz_bg,
#endif
#if DW_NDIM >= 2
          vy, vy_bg,
#endif
          vx, vx_bg,
#if DW_NDIM >= 3
          sigmazz, sigmazz_bg, sigmayz, sigmayz_bg, sigmaxz, sigmaxz_bg,
#endif
#if DW_NDIM >= 2
          sigmayy, sigmayy_bg, sigmaxy, sigmaxy_bg,
#endif
          sigmaxx, sigmaxx_bg,
#if DW_NDIM >= 3
          m_vzz, m_vzz_bg, m_vzy, m_vzy_bg, m_vzx, m_vzx_bg, m_vyz, m_vyz_bg,
          m_vxz, m_vxz_bg,
#endif
#if DW_NDIM >= 2
          m_vyy, m_vyy_bg, m_vyx, m_vyx_bg, m_vxy, m_vxy_bg,
#endif
          m_vxx, m_vxx_bg,
#if DW_NDIM >= 3
          m_sigmazzz, m_sigmazzzn, m_sigmayzy, m_sigmayzyn, m_sigmaxzx,
          m_sigmaxzxn, m_sigmazzz_bg, m_sigmazzzn_bg, m_sigmayzy_bg,
          m_sigmayzyn_bg, m_sigmaxzx_bg, m_sigmaxzxn_bg, m_sigmayzz,
          m_sigmayzzn, m_sigmaxzz, m_sigmaxzzn, m_sigmayzz_bg, m_sigmayzzn_bg,
          m_sigmaxzz_bg, m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
          m_sigmayyy, m_sigmayyyn, m_sigmaxyy, m_sigmaxyyn, m_sigmaxyx,
          m_sigmaxyxn, m_sigmayyy_bg, m_sigmayyyn_bg, m_sigmaxyy_bg,
          m_sigmaxyyn_bg, m_sigmaxyx_bg, m_sigmaxyxn_bg,
#endif
          m_sigmaxxx, m_sigmaxxxn, m_sigmaxxx_bg, m_sigmaxxxn_bg,
#if DW_NDIM >= 3
          az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
          ay, ayh, by, byh,
#endif
          ax, axh, bx, bxh,
#if DW_NDIM >= 3
          bc_z0, bc_z1,
#endif
#if DW_NDIM >= 2
          bc_y0, bc_y1,
#endif
          bc_x0, bc_x1);
      CHECK_KERNEL_ERROR
    }

    /* Refill the (adjoint) velocity guard ring of each adjoint field
     * independently (background then scattered) -- the discrete adjoint of
     * forward's stress-update kernel (which read the velocity ghost) reads
     * it here too, before the stress-adjoint update kernel just below;
     * mirrors elastic_born.c's own backward velocity-fill placement and
     * elastic.cu's own backward FILL_V_BC placement. Unlike acoustic's
     * velocity fill, elastic_born's multi-field velocity fill is not
     * self-adjoint (see the module comment above launch_fill_elastic_born_
     * velocity_bc_adjoint), so this applies that fill's true transpose
     * rather than reapplying the forward fill. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC_ADJOINT(vz_bg, vy_bg, vx_bg);
      CHECK_KERNEL_ERROR
      LAUNCH_FILL_ELASTIC_BORN_VELOCITY_BC_ADJOINT(vz, vy, vx);
      CHECK_KERNEL_ERROR
    }

    /* Velocity-type source-gradient extraction (background then scattered)
     * and the M_SIGMA** / M_SIGMA**_bg PML memory-variable update /
     * grad_buoyancy_* / grad_dbuoyancy_* accumulation all read the velocity
     * adjoint at (near-face) interior cells the fold above may have just
     * changed, so all must run after it -- see elastic_born.c's own
     * driver-side comment at the equivalent point. */
#if DW_NDIM >= 3
    LAUNCH_RECV(grad_f_bg_z, vz_bg, sources_z_i, n_sources_z_per_shot)
    LAUNCH_RECV(grad_f_z, vz, sources_z_i, n_sources_z_sc_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_RECV(grad_f_bg_y, vy_bg, sources_y_i, n_sources_y_per_shot)
    LAUNCH_RECV(grad_f_y, vy, sources_y_i, n_sources_y_sc_per_shot)
#endif
    LAUNCH_RECV(grad_f_bg_x, vx_bg, sources_x_i, n_sources_x_per_shot)
    LAUNCH_RECV(grad_f_x, vx, sources_x_i, n_sources_x_sc_per_shot)

    backward_kernel_v_pml_mem_born<<<dimGrid_guard, dimBlock, 0,
                                     stream_compute>>>(
#if DW_NDIM >= 3
        buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y, dbuoyancy_y,
#endif
        buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
        vz, vz_bg,
#endif
#if DW_NDIM >= 2
        vy, vy_bg,
#endif
        vx, vx_bg,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmazzzn, m_sigmayzy, m_sigmayzyn, m_sigmaxzx,
        m_sigmaxzxn, m_sigmazzz_bg, m_sigmazzzn_bg, m_sigmayzy_bg,
        m_sigmayzyn_bg, m_sigmaxzx_bg, m_sigmaxzxn_bg, m_sigmayzz, m_sigmayzzn,
        m_sigmaxzz, m_sigmaxzzn, m_sigmayzz_bg, m_sigmayzzn_bg, m_sigmaxzz_bg,
        m_sigmaxzzn_bg,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmayyyn, m_sigmaxyy, m_sigmaxyyn, m_sigmaxyx,
        m_sigmaxyxn, m_sigmayyy_bg, m_sigmayyyn_bg, m_sigmaxyy_bg,
        m_sigmaxyyn_bg, m_sigmaxyx_bg, m_sigmaxyxn_bg,
#endif
        m_sigmaxxx, m_sigmaxxxn, m_sigmaxxx_bg, m_sigmaxxxn_bg,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t, dvzdbuoyancy_sc_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t, dvydbuoyancy_sc_store_1_t,
#endif
        dvxdbuoyancy_store_1_t, dvxdbuoyancy_sc_store_1_t,
#if DW_NDIM >= 3
        grad_buoyancy_z_shot, grad_dbuoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        grad_buoyancy_y_shot, grad_dbuoyancy_y_shot,
#endif
        grad_buoyancy_x_shot, grad_dbuoyancy_x_shot,
#if DW_NDIM >= 3
        az, azh,
#endif
#if DW_NDIM >= 2
        ay, ayh,
#endif
        ax, axh, buoyancy_requires_grad && ((t % step_ratio) == 0),
        dbuoyancy_requires_grad && ((t % step_ratio) == 0), step_ratio);
    CHECK_KERNEL_ERROR

    backward_kernel_v_born<<<dimGrid, dimBlock, 0, stream_compute>>>(
        lamb, mu,
#if DW_NDIM >= 3
        mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
        mu_yx,
#endif
        dlamb, dmu,
#if DW_NDIM >= 3
        dmu_zy, dmu_zx,
#endif
#if DW_NDIM >= 2
        dmu_yx,
#endif
#if DW_NDIM >= 3
        buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y, dbuoyancy_y,
#endif
        buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
        vz, vz_bg,
#endif
#if DW_NDIM >= 2
        vy, vy_bg,
#endif
        vx, vx_bg,
#if DW_NDIM >= 3
        sigmazz, sigmazz_bg, sigmayz, sigmayz_bg, sigmaxz, sigmaxz_bg,
#endif
#if DW_NDIM >= 2
        sigmayy, sigmayy_bg, sigmaxy, sigmaxy_bg,
#endif
        sigmaxx, sigmaxx_bg,
#if DW_NDIM >= 3
        m_vzz, m_vzz_bg, m_vzy, m_vzy_bg, m_vzx, m_vzx_bg, m_vyz, m_vyz_bg,
        m_vxz, m_vxz_bg,
#endif
#if DW_NDIM >= 2
        m_vyy, m_vyy_bg, m_vyx, m_vyx_bg, m_vxy, m_vxy_bg,
#endif
        m_vxx, m_vxx_bg,
#if DW_NDIM >= 3
        m_sigmazzz, m_sigmazzz_bg, m_sigmayzy, m_sigmayzy_bg, m_sigmaxzx,
        m_sigmaxzx_bg, m_sigmayzz, m_sigmayzz_bg, m_sigmaxzz, m_sigmaxzz_bg,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmayyy_bg, m_sigmaxyy, m_sigmaxyy_bg, m_sigmaxyx,
        m_sigmaxyx_bg,
#endif
        m_sigmaxxx, m_sigmaxxx_bg,
#if DW_NDIM >= 3
        dvzdz_store_1_t, dvzdz_sc_store_1_t, dvzdx_plus_dvxdz_store_1_t,
        dvzdx_plus_dvxdz_sc_store_1_t, dvzdy_plus_dvydz_store_1_t,
        dvzdy_plus_dvydz_sc_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydy_store_1_t, dvydy_sc_store_1_t, dvydx_plus_dvxdy_store_1_t,
        dvydx_plus_dvxdy_sc_store_1_t,
#endif
        dvxdx_store_1_t, dvxdx_sc_store_1_t, grad_lamb_shot, grad_dlamb_shot,
        grad_mu_shot, grad_dmu_shot,
#if DW_NDIM >= 3
        grad_mu_zy_shot, grad_dmu_zy_shot, grad_mu_zx_shot, grad_dmu_zx_shot,
#endif
#if DW_NDIM >= 2
        grad_mu_yx_shot, grad_dmu_yx_shot,
#endif
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, lamb_requires_grad && ((t % step_ratio) == 0),
        dlamb_requires_grad && ((t % step_ratio) == 0),
        mu_requires_grad && ((t % step_ratio) == 0),
        dmu_requires_grad && ((t % step_ratio) == 0), step_ratio);
    CHECK_KERNEL_ERROR

    /* Guard-band extension of the stress-adjoint gathers, fused over both
     * adjoint fields -- see the "Guard-band gathers" module comment above
     * backward_kernel_sigma_guard. Must complete before the NEXT (reverse
     * time) iteration's FUNC(fill_elastic_born_stress_bc_adjoint), which
     * folds this guard-band deposit into the interior. */
    if (has_bc) {
      backward_kernel_v_born_guard<<<dimGrid_guard, dimBlock, 0,
                                     stream_compute>>>(
#if DW_NDIM >= 3
          mu_zy, mu_zx,
#endif
#if DW_NDIM >= 2
          mu_yx,
#endif
#if DW_NDIM >= 3
          dmu_zy, dmu_zx,
#endif
#if DW_NDIM >= 2
          dmu_yx,
#endif
#if DW_NDIM >= 3
          buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
          buoyancy_y, dbuoyancy_y,
#endif
          buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
          vz, vz_bg,
#endif
#if DW_NDIM >= 2
          vy, vy_bg,
#endif
          vx, vx_bg,
#if DW_NDIM >= 3
          sigmazz, sigmazz_bg, sigmayz, sigmayz_bg, sigmaxz, sigmaxz_bg,
#endif
#if DW_NDIM >= 2
          sigmayy, sigmayy_bg, sigmaxy, sigmaxy_bg,
#endif
          sigmaxx, sigmaxx_bg,
#if DW_NDIM >= 3
          m_vzz, m_vzz_bg, m_vzy, m_vzy_bg, m_vzx, m_vzx_bg, m_vyz, m_vyz_bg,
          m_vxz, m_vxz_bg,
#endif
#if DW_NDIM >= 2
          m_vyy, m_vyy_bg, m_vyx, m_vyx_bg, m_vxy, m_vxy_bg,
#endif
          m_vxx, m_vxx_bg,
#if DW_NDIM >= 3
          m_sigmaxzz, m_sigmaxzz_bg, m_sigmaxzx, m_sigmaxzx_bg, m_sigmayzz,
          m_sigmayzz_bg, m_sigmayzy, m_sigmayzy_bg,
#endif
#if DW_NDIM >= 2
          m_sigmaxyy, m_sigmaxyy_bg, m_sigmaxyx, m_sigmaxyx_bg,
#endif
#if DW_NDIM >= 3
          az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
          ay, ayh, by, byh,
#endif
          ax, axh, bx, bxh,
#if DW_NDIM >= 3
          bc_z0, bc_z1,
#endif
#if DW_NDIM >= 2
          bc_y0, bc_y1,
#endif
          bc_x0, bc_x1);
      CHECK_KERNEL_ERROR
    }

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    /* inject adjoint sources (background then scattered) */
#if DW_NDIM >= 3
    LAUNCH_SRC(vz_bg, grad_r_bg_z, receivers_bg_z_i, n_receivers_bg_z_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_SRC(vy_bg, grad_r_bg_y, receivers_bg_y_i, n_receivers_bg_y_per_shot)
#endif
    LAUNCH_SRC(vx_bg, grad_r_bg_x, receivers_bg_x_i, n_receivers_bg_x_per_shot)
    if (n_receivers_bg_p_per_shot > 0) {
      dim3 const g(ceil_div(n_receivers_bg_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      add_sources_p<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz_bg,
#endif
#if DW_NDIM >= 2
          sigmayy_bg,
#endif
          sigmaxx_bg, grad_r_bg_p + t * n_shots_h * n_receivers_bg_p_per_shot,
          receivers_bg_p_i, n_receivers_bg_p_per_shot);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    LAUNCH_SRC(vz, grad_r_z, receivers_z_i, n_receivers_z_per_shot)
#endif
#if DW_NDIM >= 2
    LAUNCH_SRC(vy, grad_r_y, receivers_y_i, n_receivers_y_per_shot)
#endif
    LAUNCH_SRC(vx, grad_r_x, receivers_x_i, n_receivers_x_per_shot)
    if (n_receivers_p_per_shot > 0) {
      dim3 const g(ceil_div(n_receivers_p_per_shot, dimBlock_src_rcv.x),
                   ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);
      add_sources_p<<<g, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, grad_r_p + t * n_shots_h * n_receivers_p_per_shot,
          receivers_p_i, n_receivers_p_per_shot);
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM >= 3
    std::swap(m_sigmazzz, m_sigmazzzn);
    std::swap(m_sigmayzy, m_sigmayzyn);
    std::swap(m_sigmaxzx, m_sigmaxzxn);
    std::swap(m_sigmayzz, m_sigmayzzn);
    std::swap(m_sigmaxzz, m_sigmaxzzn);
    std::swap(m_sigmazzz_bg, m_sigmazzzn_bg);
    std::swap(m_sigmayzy_bg, m_sigmayzyn_bg);
    std::swap(m_sigmaxzx_bg, m_sigmaxzxn_bg);
    std::swap(m_sigmayzz_bg, m_sigmayzzn_bg);
    std::swap(m_sigmaxzz_bg, m_sigmaxzzn_bg);
#endif
#if DW_NDIM >= 2
    std::swap(m_sigmayyy, m_sigmayyyn);
    std::swap(m_sigmaxyy, m_sigmaxyyn);
    std::swap(m_sigmaxyx, m_sigmaxyxn);
    std::swap(m_sigmayyy_bg, m_sigmayyyn_bg);
    std::swap(m_sigmaxyy_bg, m_sigmaxyyn_bg);
    std::swap(m_sigmaxyx_bg, m_sigmaxyxn_bg);
#endif
    std::swap(m_sigmaxxx, m_sigmaxxxn);
    std::swap(m_sigmaxxx_bg, m_sigmaxxxn_bg);
  }
#undef SETUP_STORE_LOAD
#undef OPEN_FILE_READ

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
  if (dlamb_requires_grad && !dlamb_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dlamb, grad_dlamb_shot);
    CHECK_KERNEL_ERROR
  }
  if (dmu_requires_grad && !dmu_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu, grad_dmu_shot);
    CHECK_KERNEL_ERROR
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu_zy, grad_dmu_zy_shot);
    CHECK_KERNEL_ERROR
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu_zx, grad_dmu_zx_shot);
    CHECK_KERNEL_ERROR
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dmu_yx, grad_dmu_yx_shot);
    CHECK_KERNEL_ERROR
#endif
  }
  if (dbuoyancy_requires_grad && !dbuoyancy_batched_h && n_shots_h > 1) {
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbuoyancy_z, grad_dbuoyancy_z_shot);
    CHECK_KERNEL_ERROR
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbuoyancy_y, grad_dbuoyancy_y_shot);
    CHECK_KERNEL_ERROR
#endif
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbuoyancy_x, grad_dbuoyancy_x_shot);
    CHECK_KERNEL_ERROR
  }
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

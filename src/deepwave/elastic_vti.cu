/*
 * Elastic VTI (vertically transverse isotropic) wave equation propagator
 * (CUDA implementation)
 */

/*
 * This file contains the CUDA implementation of the elastic VTI wave equation
 * propagator. It is templated from the isotropic elastic CUDA propagator
 * (elastic.cu): the staggered grid, velocity update, PML, improved-vacuum
 * free surface, source/receiver handling, storage, and the whole
 * time-stepping/adjoint skeleton are UNCHANGED. Only the constitutive
 * (stress-strain) relation and the model set differ. It is compiled multiple
 * times with different options to generate a set of functions that can be
 * called from Python. The options are specified by the following macros:
 *  * DW_NDIM: The number of spatial dimensions. Possible values are 2-3
 *    (there is no 1-D VTI form).
 *  * DW_ACCURACY: The order of accuracy of the spatial finite difference
 *    stencil. Possible values are 2, 4, 6, and 8.
 *  * DW_DTYPE: The floating point type to use for calculations. Possible
 *    values are float and double.
 */

/*
 * For a full description of the method, physics, and adjoint derivation, see
 * the CPU implementation in elastic_vti.c. This file implements the same
 * functionality, but for execution on a GPU using CUDA.
 *
 * CONSTITUTIVE RELATION (VTI, symmetry axis = slowest dimension: z in 3-D,
 * y in 2-D). With the symmetry axis as the Voigt "3" direction and
 * d_i = (PML-corrected) dv_i/dx_i, the forward stress-rate updates are:
 *
 *   3-D (models c11, c12=c11-2*c66, c13, c33, c55 [zy,zx nodes], c66 [yx]):
 *     sigmaxx += dt * (c11 dvxdx + c12 dvydy + c13 dvzdz)
 *     sigmayy += dt * (c12 dvxdx + c11 dvydy + c13 dvzdz)
 *     sigmazz += dt * (c13 dvxdx + c13 dvydy + c33 dvzdz)
 *     sigmaxy += dt * c66_yx (dvxdy + dvydx)
 *     sigmaxz += dt * c55_zx (dvxdz + dvzdx)
 *     sigmayz += dt * c55_zy (dvydz + dvzdy)
 *
 *   2-D P-SV in the (y,x) plane, axis = y (models c11, c13, c33, c55 [yx]):
 *     sigmaxx += dt * (c11 dvxdx + c13 dvydy)
 *     sigmayy += dt * (c13 dvxdx + c33 dvydy)
 *     sigmaxy += dt * c55_yx (dvxdy + dvydx)
 *
 * The isotropic elastic propagator is recovered by c11=c33=lamb+2*mu,
 * c13=lamb, c55=c66=mu (then c12=c11-2*c66=lamb).
 */

#include <stdio.h>

#include <cstdint>

#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  elastic_vti_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
/* Multiaxial-PML (M-PML) build variant -- see elastic.c for the rationale. When
 * DW_MPML is defined the exported forward/backward symbols gain an ``_mpml``
 * suffix and the PML profiles are read as full-grid arrays (P_* accessors);
 * otherwise the default per-axis 1-D behaviour is compiled unchanged. Kept in
 * lockstep with elastic.c. */
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

/* PML-region test wrapper -- kept in lockstep with elastic.c's DW_PML_SKIP.
 * The default kernel applies a direction's PML update only inside that
 * direction's PML region (where its 1-D profile is nonzero). Under M-PML each
 * direction's damping is nonzero wherever ANY axis is inside its PML (the
 * cross-axis term), so the M-PML variant applies the PML path throughout the
 * union of all axes' PML strips and keeps the fast path only in the pure
 * interior. The union below uses each axis's widest (half-grid) extent
 * (`end - 1`); that is a superset of every field's true strip, which is exact
 * because the full-grid M-PML profiles vanish identically wherever there is in
 * fact no damping (a = b = 0). The pml_*0/pml_*1 bounds are __constant__
 * device symbols, in scope in every kernel. */
/* The dead `false && (cond)` keeps the per-kernel bound variables that only
 * appear in `cond` referenced (no unused-variable warnings); the compiler
 * folds it away. */
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

/* ND_INDEX, the P_A(*)/P_B(*) PML-profile accessors, the VZ/VY/VX/SIGMA(*)/
 * C11/C12/C13/C33/C_XY/C55_ZX/C55_ZY/BUOYANCY_(*)/M_SIGMA(**)/M_V(**)
 * field-access macros, the VZ_Z/VY_Y/VX_X(+_PML)/SIGMAZZ_Z/.../SIGMAYZ_Y
 * (+_PML) adjoint-term macros, and the FUNC(forward_vz_cell)/.../
 * FUNC(backward_sigmayz_cell) per-grid-point stencil update functions (14
 * total: one per velocity axis/combined-normal-or-shear-stress component,
 * forward and backward) live in elastic_vti_kernels.h, shared verbatim with
 * elastic_vti.c (see that header for the full explanation). DW_PML_TEST above
 * (CUDA-driver-side PML-region test, not per-grid-point math) is used to
 * compute the plain `pml_z`/`pml_y`/`pml_x` bools those functions expect. */
#include "elastic_vti_kernels.h"

#define MAX(a, b) (a > b ? a : b)

namespace {

/* --- boundary=["free"/"rigid"/"periodic",...] guard-ring image fill ------
 *
 * CUDA mirror of elastic_vti.c's own fill functions (see that file's
 * matching comment for the full derivation/citations, and elastic.cu's own
 * version of this block for the acoustic.cu precedent this follows) -- a
 * direct, unchanged-arithmetic port. ``kind``: 0 = on-row stress (sigma_AA),
 * 1 = on-row velocity, 2 = off-row velocity, 3 = shared-axis shear stress.
 * elastic_vti has no DW_NDIM==1 build (see the module docstring above), so
 * unlike elastic.cu's own version of this block there is no DW_NDIM==1
 * branch anywhere below. */

__device__ inline void fill_p_bc_line_dev(DW_DTYPE *__restrict const p,
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
 * dev/fill_elastic_shear_line_dev, fill_p_bc_line_adjoint_dev, and their
 * three adjoint counterparts live in elastic_line_bc_dev.h (see that
 * header). */
#include "elastic_line_bc_dev.h"

/* One thread per line, exactly like fill_elastic_bc_kernel above, but
 * dispatching to the adjoint line arithmetic variants. */
__global__ void fill_elastic_bc_kernel_adjoint(
    DW_DTYPE *__restrict const wf, int64_t const n_lines, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi,
    int const kind) {
  int64_t const L = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
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

/* Transpose of launch_fill_elastic_stress_bc: faces launched in the
 * reverse of that function's own face order (z, then y, then x for 3D; y,
 * then x for 2D) -- see the module comment above for why this ordering
 * (all on one CUDA stream) is both necessary and sufficient for corner
 * cells, with no extra synchronisation required. */
static void launch_fill_elastic_stress_bc_adjoint(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const sigmaxx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmazz, nl, nz_, ny_ * nx_, bc[0], bc[1], 0);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmayz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmaxz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
  }
#endif
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmayy, nl, ny_, nx_, bc[2], bc[3], 0);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmayz, nl, ny_, nx_, bc[2], bc[3], 3);
#endif
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmaxy, nl, ny_, nx_, bc[2], bc[3], 3);
  }
#endif
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmaxx, nl, nx_, 1, bc[4], bc[5], 0);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmaxz, nl, nx_, 1, bc[4], bc[5], 3);
#endif
#if DW_NDIM >= 2
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        sigmaxy, nl, nx_, 1, bc[4], bc[5], 3);
#endif
  }
}

/* Transpose of launch_fill_elastic_velocity_bc: faces launched in the
 * reverse of that function's own face order (see the module comment
 * above). */
static void launch_fill_elastic_velocity_bc_adjoint(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, bc[0], bc[1], 2);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        vy, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(
        vx, nl, nz_, ny_ * nx_, bc[0], bc[1], 1);
  }
#endif
#if DW_NDIM >= 2
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(vy, nl, ny_, nx_,
                                                             bc[2], bc[3], 2);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(vz, nl, ny_, nx_,
                                                             bc[2], bc[3], 1);
#endif
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(vx, nl, ny_, nx_,
                                                             bc[2], bc[3], 1);
  }
#endif
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(vx, nl, nx_, 1,
                                                             bc[4], bc[5], 2);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(vz, nl, nx_, 1,
                                                             bc[4], bc[5], 1);
#endif
#if DW_NDIM >= 2
    fill_elastic_bc_kernel_adjoint<<<grid, tpb, 0, stream>>>(vy, nl, nx_, 1,
                                                             bc[4], bc[5], 1);
#endif
  }
}

/* One thread per line along the chosen axis, matching elastic.cu's own
 * fill_elastic_bc_kernel exactly. ``kind`` selects which of the four line
 * arithmetic variants above to apply. */
__global__ void fill_elastic_bc_kernel(DW_DTYPE *__restrict const wf,
                                       int64_t const n_lines, int64_t const nax,
                                       int64_t const stride,
                                       int64_t const bc_lo, int64_t const bc_hi,
                                       int const kind) {
  int64_t const L = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
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

/* Stress guard fill on every reflecting axis: sigma_AA (kind 0) plus the two
 * shear stresses sharing axis A (kind 3), mirroring elastic_vti.c's
 * FUNC(fill_elastic_stress_bc) but as CUDA kernel launches (one per
 * (field, face), matching elastic.cu's own launch_fill_elastic_stress_bc
 * structure). Only DW_NDIM 2/3 are reachable for elastic_vti. */
static void launch_fill_elastic_stress_bc(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
    DW_DTYPE *__restrict const sigmaxx, int64_t const bc[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxx, nl, nx_, 1, bc[4],
                                                     bc[5], 0);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxz, nl, nx_, 1, bc[4],
                                                     bc[5], 3);
#endif
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxy, nl, nx_, 1, bc[4],
                                                     bc[5], 3);
  }
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(sigmayy, nl, ny_, nx_,
                                                     bc[2], bc[3], 0);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(sigmayz, nl, ny_, nx_,
                                                     bc[2], bc[3], 3);
#endif
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(sigmaxy, nl, ny_, nx_,
                                                     bc[2], bc[3], 3);
  }
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(
        sigmazz, nl, nz_, ny_ * nx_, bc[0], bc[1], 0);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(
        sigmayz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(
        sigmaxz, nl, nz_, ny_ * nx_, bc[0], bc[1], 3);
  }
#endif
}

/* Velocity guard fill on every reflecting axis: v_A off-row (kind 2), the
 * other velocities on-row (kind 1), mirroring elastic_vti.c's
 * FUNC(fill_elastic_velocity_bc). */
static void launch_fill_elastic_velocity_bc(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
    DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const vx,
    int64_t const bc[6], int64_t const total_numel, int64_t const nz_,
    int64_t const ny_, int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  if (bc[4] != 0 || bc[5] != 0) {
    int64_t const nl = total_numel / nx_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vx, nl, nx_, 1, bc[4],
                                                     bc[5], 2);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vz, nl, nx_, 1, bc[4],
                                                     bc[5], 1);
#endif
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vy, nl, nx_, 1, bc[4],
                                                     bc[5], 1);
  }
  if (bc[2] != 0 || bc[3] != 0) {
    int64_t const nl = total_numel / ny_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vy, nl, ny_, nx_, bc[2],
                                                     bc[3], 2);
#if DW_NDIM >= 3
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vz, nl, ny_, nx_, bc[2],
                                                     bc[3], 1);
#endif
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vx, nl, ny_, nx_, bc[2],
                                                     bc[3], 1);
  }
#if DW_NDIM >= 3
  if (bc[0] != 0 || bc[1] != 0) {
    int64_t const nl = total_numel / nz_;
    unsigned int const grid = ceil_div((unsigned int)nl, tpb);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vz, nl, nz_, ny_ * nx_,
                                                     bc[0], bc[1], 2);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vy, nl, nz_, ny_ * nx_,
                                                     bc[0], bc[1], 1);
    fill_elastic_bc_kernel<<<grid, tpb, 0, stream>>>(vx, nl, nz_, ny_ * nx_,
                                                     bc[0], bc[1], 1);
  }
#endif
}

#if DW_NDIM == 3
#define LAUNCH_FILL_ELASTIC_STRESS_BC()                                      \
  launch_fill_elastic_stress_bc(sigmazz, sigmayz, sigmaxz, sigmayy, sigmaxy, \
                                sigmaxx, bc, n_shots_h *shot_numel, nz_h,    \
                                ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_VELOCITY_BC()                                      \
  launch_fill_elastic_velocity_bc(vz, vy, vx, bc, n_shots_h *shot_numel, nz_h, \
                                  ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_STRESS_BC_ADJOINT()                 \
  launch_fill_elastic_stress_bc_adjoint(                        \
      sigmazz, sigmayz, sigmaxz, sigmayy, sigmaxy, sigmaxx, bc, \
      n_shots_h *shot_numel, nz_h, ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_VELOCITY_BC_ADJOINT() \
  launch_fill_elastic_velocity_bc_adjoint(        \
      vz, vy, vx, bc, n_shots_h *shot_numel, nz_h, ny_h, nx_h, stream_compute)
#else /* DW_NDIM == 2 */
#define LAUNCH_FILL_ELASTIC_STRESS_BC()                               \
  launch_fill_elastic_stress_bc(sigmayy, sigmaxy, sigmaxx, bc,        \
                                n_shots_h *shot_numel, 1, ny_h, nx_h, \
                                stream_compute)
#define LAUNCH_FILL_ELASTIC_VELOCITY_BC()                                     \
  launch_fill_elastic_velocity_bc(vy, vx, bc, n_shots_h *shot_numel, 1, ny_h, \
                                  nx_h, stream_compute)
#define LAUNCH_FILL_ELASTIC_STRESS_BC_ADJOINT()                               \
  launch_fill_elastic_stress_bc_adjoint(sigmayy, sigmaxy, sigmaxx, bc,        \
                                        n_shots_h *shot_numel, 1, ny_h, nx_h, \
                                        stream_compute)
#define LAUNCH_FILL_ELASTIC_VELOCITY_BC_ADJOINT()                            \
  launch_fill_elastic_velocity_bc_adjoint(vy, vx, bc, n_shots_h *shot_numel, \
                                          1, ny_h, nx_h, stream_compute)
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
__constant__ int64_t n_sources_z_per_shot;
#endif
#if DW_NDIM >= 2
__constant__ int64_t n_sources_y_per_shot;
#endif
__constant__ int64_t n_sources_x_per_shot;
__constant__ int64_t n_sources_p_per_shot;
#if DW_NDIM >= 3
__constant__ int64_t n_receivers_z_per_shot;
#endif
#if DW_NDIM >= 2
__constant__ int64_t n_receivers_y_per_shot;
#endif
__constant__ int64_t n_receivers_x_per_shot;
__constant__ int64_t n_receivers_p_per_shot;
__constant__ int64_t step_ratio;
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
__constant__ bool c11_batched;
#if DW_NDIM >= 3
__constant__ bool c12_batched;
#endif
__constant__ bool c13_batched;
__constant__ bool c33_batched;
__constant__ bool c55_batched;
#if DW_NDIM >= 3
__constant__ bool c66_batched;
#endif
__constant__ bool buoyancy_batched;

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

#if DW_NDIM >= 2
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
#endif

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

__launch_bounds__(32) __global__ void add_sources_p(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy,
#endif
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_p_per_shot + source_idx;
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

#if DW_NDIM >= 2
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
#endif

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

__launch_bounds__(32) __global__ void add_adjoint_pressure_sources(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy,
#endif
    DW_DTYPE *__restrict const sigmaxx, DW_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_receivers_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_p_per_shot + source_idx;
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

#if DW_NDIM >= 2
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
#endif

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
    void record_pressure_receivers(DW_DTYPE *__restrict const r,
#if DW_NDIM >= 3
                                   DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
                                   DW_DTYPE const *__restrict const sigmayy,
#endif
                                   DW_DTYPE const *__restrict const sigmaxx,
                                   int64_t const *__restrict receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_p_per_shot + receiver_idx;
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

#if DW_NDIM >= 2
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
#endif

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

__launch_bounds__(32) __global__ void record_adjoint_pressure_receivers(
    DW_DTYPE *__restrict const r,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
    int64_t const *__restrict const receivers_i) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_sources_p_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_p_per_shot + receiver_idx;
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

/* Thread-block shape and occupancy target for the forward-pass 3-D
 * staggered-grid stencil kernels below, mirroring scalar.cu's 2026-08-10
 * tuning (see that file for the full rationale). A 128-thread bound lets
 * ptxas spend up to 255 registers/thread on these kernels at accuracy >= 4,
 * capping SM occupancy; a 32x4x4 block with a matching 512-thread bound
 * caps the allocation at 128 registers. Only the forward launcher (this
 * macro) uses the wider block: measured on 80^3/nt=50/float32 (same
 * methodology as elastic.cu, which shares this file's kernel structure),
 * the same change applied to the backward launcher was a net regression
 * there (10-19% slower at accuracy 6 and 8 on both cards) -- the backward
 * kernels keep the narrow default unconditionally. Accuracy 2 keeps the
 * narrow default here too. */
#if DW_NDIM == 3 && DW_ACCURACY >= 4
#define DW_STENCIL_BLOCK_3D 32, 4, 4
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(512)
#else
#define DW_STENCIL_BLOCK_3D 32, 4, 1
#define DW_STENCIL_LAUNCH_BOUNDS __launch_bounds__(128)
#endif

DW_STENCIL_LAUNCH_BOUNDS __global__ void forward_kernel_v(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const dvydbuoyancy,
#endif
    DW_DTYPE *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
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

      int64_t const i = shot_idx * shot_numel + j;
#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const buoyancy_z_shot =
            buoyancy_batched ? buoyancy_z[i] : buoyancy_z[j];
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

        FUNC(forward_vz_cell)(sigmazz, sigmayz, sigmaxz, vz, m_sigmazzz,
                              m_sigmayzy, m_sigmaxzx, dvzdbuoyancy,
                              buoyancy_z_shot, azh, bzh, ay, by, ax, bx, dt,
                              buoyancy_requires_grad, i, ny, nx, z, y, x, pml_z,
                              pml_y, pml_x, rdz, rdy, rdx);
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const buoyancy_y_shot =
            buoyancy_batched ? buoyancy_y[i] : buoyancy_y[j];
#if DW_NDIM == 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);

        FUNC(forward_vy_cell)(
#if DW_NDIM >= 3
            sigmayz,
#endif
            sigmayy, sigmaxy, vy,
#if DW_NDIM >= 3
            m_sigmayzz,
#endif
            m_sigmayyy, m_sigmaxyx, dvydbuoyancy, buoyancy_y_shot,
#if DW_NDIM >= 3
            az, bz,
#endif
            ayh, byh, ax, bx, dt, buoyancy_requires_grad, i,
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
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);

        FUNC(forward_vx_cell)(
#if DW_NDIM >= 3
            sigmaxz,
#endif
#if DW_NDIM >= 2
            sigmaxy,
#endif
            sigmaxx, vx,
#if DW_NDIM >= 3
            m_sigmaxzz,
#endif
#if DW_NDIM >= 2
            m_sigmaxyy,
#endif
            m_sigmaxxx, dvxdbuoyancy, buoyancy_x_shot,
#if DW_NDIM >= 3
            az, bz,
#endif
#if DW_NDIM >= 2
            ay, by,
#endif
            axh, bxh, dt, buoyancy_requires_grad, i,
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

DW_STENCIL_LAUNCH_BOUNDS __global__ void forward_kernel_sigma(
    DW_DTYPE const *__restrict const c11,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c12,
#endif
    DW_DTYPE const *__restrict const c13, DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c55_zy,
    DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    /* Shear stiffness at the sigma_xy node: c66_yx in 3-D, c55_yx in 2-D. */
    DW_DTYPE const *__restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy,
#endif
    DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdz_store,
    DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE *__restrict const dvzdy_plus_dvydz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const dvydy_store,
    DW_DTYPE *__restrict const dvydx_plus_dvxdy_store,
#endif
    DW_DTYPE *__restrict const dvxdx_store,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const c11_requires_grad,
#if DW_NDIM >= 3
    bool const c12_requires_grad,
#endif
    bool const c13_requires_grad, bool const c33_requires_grad,
#if DW_NDIM >= 3
    bool const c55_requires_grad, bool const c66_requires_grad
#else
    bool const c55_requires_grad
#endif
) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
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

      int64_t const i = shot_idx * shot_numel + j;
      // Per-cell VTI normal-node stiffnesses.
      DW_DTYPE const c11_shot_i = c11_batched ? c11[i] : c11[j];
#if DW_NDIM >= 2
      DW_DTYPE const c13_shot_i = c13_batched ? c13[i] : c13[j];
      DW_DTYPE const c33_shot_i = c33_batched ? c33[i] : c33[j];
#endif
#if DW_NDIM >= 3
      DW_DTYPE const c12_shot_i = c12_batched ? c12[i] : c12[j];
#endif
      // Aggregate requires-grad conditions gating the stored velocity
      // derivatives (must match Python's storage_flags sizing): any normal-node
      // stiffness drives the dvidi stores; the sigma_xy-node shear (c66 in 3-D,
      // c55 in 2-D) drives its store.
#if DW_NDIM >= 3
      bool const cnormal_requires_grad = c11_requires_grad ||
                                         c12_requires_grad ||
                                         c13_requires_grad || c33_requires_grad;
      bool const c_xy_requires_grad = c66_requires_grad;
#elif DW_NDIM == 2
    bool const cnormal_requires_grad =
        c11_requires_grad || c13_requires_grad || c33_requires_grad;
    bool const c_xy_requires_grad = c55_requires_grad;
#else
    bool const cnormal_requires_grad = c11_requires_grad;
#endif

#if DW_NDIM >= 3
      bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
      bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
      bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
      FUNC(forward_sigmaii_cell)(
#if DW_NDIM >= 3
          vz, sigmazz, m_vzz, dvzdz_store,
#endif
#if DW_NDIM >= 2
          vy, sigmayy, m_vyy, dvydy_store,
#endif
          vx, sigmaxx, m_vxx, dvxdx_store, c11_shot_i,
#if DW_NDIM >= 3
          c12_shot_i,
#endif
          c13_shot_i,
#if DW_NDIM >= 2
          c33_shot_i,
#endif
          dt, cnormal_requires_grad, i,
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
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        DW_DTYPE const c_xy_shot_i =
#if DW_NDIM >= 3
            c66_batched ? c_xy[i] : c_xy[j];
#else
            c55_batched ? c_xy[i] : c_xy[j];
#endif
        FUNC(forward_sigmaxy_cell)(
            vx, vy, sigmaxy, m_vxy, m_vyx, dvydx_plus_dvxdy_store, c_xy_shot_i,
            dt, c_xy_requires_grad, i,
#if DW_NDIM >= 3
            ny, z,
#endif
            nx, y, x, pml_y, pml_x, ayh, byh, axh, bxh, rdy, rdx);
      }
#endif
#if DW_NDIM >= 3
      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0h || x >= pml_x1h);
        DW_DTYPE const c55_zx_shot_i = c55_batched ? c55_zx[i] : c55_zx[j];
        FUNC(forward_sigmaxz_cell)(vx, vz, sigmaxz, m_vxz, m_vzx,
                                   dvzdx_plus_dvxdz_store, c55_zx_shot_i, dt,
                                   c55_requires_grad, i, ny, nx, z, y, x, pml_z,
                                   pml_x, azh, bzh, axh, bxh, rdz, rdx);
      }
      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        DW_DTYPE const c55_zy_shot_i = c55_batched ? c55_zy[i] : c55_zy[j];
        FUNC(forward_sigmayz_cell)(vy, vz, sigmayz, m_vyz, m_vzy,
                                   dvzdy_plus_dvydz_store, c55_zy_shot_i, dt,
                                   c55_requires_grad, i, ny, nx, z, y, x, pml_z,
                                   pml_y, azh, bzh, ayh, byh, rdz, rdy);
      }
#endif

#if DW_NDIM == 3
    }
#endif
  }
}

__launch_bounds__(128) __global__ void backward_kernel_sigma(
    DW_DTYPE const *__restrict const c11,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c12,
#endif
    DW_DTYPE const *__restrict const c13, DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c55_zy,
    DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    /* Shear stiffness at the sigma_xy node: c66_yx in 3-D, c55_yx in 2-D. */
    DW_DTYPE const *__restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_vyz,
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
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
    DW_DTYPE const *__restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dvydbuoyancy,
#endif
    DW_DTYPE const *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_buoyancy_y,
#endif
    DW_DTYPE *__restrict const grad_buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const buoyancy_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
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
      DW_DTYPE const *__restrict const c11_shot =
          c11_batched ? c11 + shot_idx * shot_numel : c11;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c12_shot =
          c12_batched ? c12 + shot_idx * shot_numel : c12;
#endif
      DW_DTYPE const *__restrict const c13_shot =
          c13_batched ? c13 + shot_idx * shot_numel : c13;
      DW_DTYPE const *__restrict const c33_shot =
          c33_batched ? c33 + shot_idx * shot_numel : c33;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c55_zy_shot =
          c55_batched ? c55_zy + shot_idx * shot_numel : c55_zy;
      DW_DTYPE const *__restrict const c55_zx_shot =
          c55_batched ? c55_zx + shot_idx * shot_numel : c55_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const c_xy_shot =
#if DW_NDIM >= 3
          c66_batched ? c_xy + shot_idx * shot_numel : c_xy;
#else
          c55_batched ? c_xy + shot_idx * shot_numel : c_xy;
#endif
#endif

#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

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

      int64_t i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_vz_cell)(
            c13_shot, c33_shot, c55_zy_shot, c55_zx_shot, sigmazz, sigmayy,
            sigmaxx, sigmayz, sigmaxz, vz, m_vzz, m_vzy, m_vzx, m_sigmazzz,
            m_sigmazzzn, m_sigmayzy, m_sigmayzyn, m_sigmaxzx, m_sigmaxzxn, az,
            azh, bz, bzh, ay, ayh, by, byh, ax, axh, bx, bxh, dt, i, j, ny, nx,
            z, y, x, pml_z, pml_y, pml_x, rdz, rdy, rdx);
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_vy_cell)(
#if DW_NDIM >= 3
            c55_zy_shot, c11_shot, c12_shot, c13_shot,
#else
            c13_shot, c33_shot,
#endif
            c_xy_shot,
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
            m_sigmayyy, m_sigmayyyn, m_sigmaxyx, m_sigmaxyxn,
#if DW_NDIM >= 3
            az, azh, bz, bzh,
#endif
            ay, ayh, by, byh, ax, axh, bx, bxh, dt, i, j,
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
        FUNC(backward_vx_cell)(
#if DW_NDIM >= 3
            c55_zx_shot,
#endif
#if DW_NDIM >= 2
            c_xy_shot,
#endif
            c11_shot,
#if DW_NDIM >= 2
            c13_shot,
#endif
#if DW_NDIM >= 3
            c12_shot,
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
#if DW_NDIM >= 3
            az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
            ay, ayh, by, byh,
#endif
            ax, axh, bx, bxh, dt, i, j,
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
    DW_DTYPE const *__restrict const c11,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c12,
#endif
    DW_DTYPE const *__restrict const c13, DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c55_zy,
    DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    /* Shear stiffness at the sigma_xy node: c66_yx in 3-D, c55_yx in 2-D. */
    DW_DTYPE const *__restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy,
#endif
    DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdz_store,
    DW_DTYPE const *__restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE const *__restrict const dvzdy_plus_dvydz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dvydy_store,
    DW_DTYPE const *__restrict const dvydx_plus_dvxdy_store,
#endif
    DW_DTYPE const *__restrict const dvxdx_store,
    DW_DTYPE *__restrict const grad_c11,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_c12,
#endif
    DW_DTYPE *__restrict const grad_c13, DW_DTYPE *__restrict const grad_c33,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_c55_zy,
    DW_DTYPE *__restrict const grad_c55_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
    bool const c11_requires_grad,
#if DW_NDIM >= 3
    bool const c12_requires_grad,
#endif
    bool const c13_requires_grad, bool const c33_requires_grad,
#if DW_NDIM >= 3
    bool const c55_requires_grad, bool const c66_requires_grad
#else
    bool const c55_requires_grad
#endif
) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
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
      DW_DTYPE const *__restrict const c11_shot =
          c11_batched ? c11 + shot_idx * shot_numel : c11;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c12_shot =
          c12_batched ? c12 + shot_idx * shot_numel : c12;
#endif
      DW_DTYPE const *__restrict const c13_shot =
          c13_batched ? c13 + shot_idx * shot_numel : c13;
      DW_DTYPE const *__restrict const c33_shot =
          c33_batched ? c33 + shot_idx * shot_numel : c33;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c55_zy_shot =
          c55_batched ? c55_zy + shot_idx * shot_numel : c55_zy;
      DW_DTYPE const *__restrict const c55_zx_shot =
          c55_batched ? c55_zx + shot_idx * shot_numel : c55_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const c_xy_shot =
#if DW_NDIM >= 3
          c66_batched ? c_xy + shot_idx * shot_numel : c_xy;
#else
          c55_batched ? c_xy + shot_idx * shot_numel : c_xy;
#endif
#endif
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

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

      int64_t const i = shot_idx * shot_numel + j;

      // Shear stiffness at the sigma_xy node: c66 in 3-D, c55 in 2-D.
#if DW_NDIM >= 3
      bool const c_xy_requires_grad = c66_requires_grad;
#else
    bool const c_xy_requires_grad = c55_requires_grad;
#endif

      {
#if DW_NDIM >= 3
        bool const pml_z = DW_PML_TEST(z < pml_z0 || z >= pml_z1);
#endif
#if DW_NDIM >= 2
        bool const pml_y = DW_PML_TEST(y < pml_y0 || y >= pml_y1);
#endif
        bool const pml_x = DW_PML_TEST(x < pml_x0 || x >= pml_x1);
        FUNC(backward_sigmaii_cell)(
            c11_shot,
#if DW_NDIM >= 3
            c12_shot,
#endif
            c13_shot, c33_shot,
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
            vx, sigmaxx, m_vxx, m_sigmaxxx, dvxdx_store, grad_c11,
#if DW_NDIM >= 3
            grad_c12,
#endif
            grad_c13, grad_c33, dt, step_ratio, c11_requires_grad,
#if DW_NDIM >= 3
            c12_requires_grad,
#endif
            c13_requires_grad, c33_requires_grad, i, j,
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
            c_xy_shot, buoyancy_x_shot, buoyancy_y_shot, vx, vy, sigmaxy, m_vxy,
            m_vyx, m_sigmaxyy, m_sigmaxyx, dvydx_plus_dvxdy_store, grad_c_xy,
            dt, step_ratio, c_xy_requires_grad, i, j,
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
            c55_zx_shot, buoyancy_x_shot, buoyancy_z_shot, vx, vz, sigmaxz,
            m_vxz, m_vzx, m_sigmaxzz, m_sigmaxzx, dvzdx_plus_dvxdz_store,
            grad_c55_zx, dt, step_ratio, c55_requires_grad, i, j, ny, nx, z, y,
            x, pml_z, pml_x, az, azh, bz, bzh, ax, axh, bx, bxh, rdz, rdx);
      }

      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        bool const pml_z = DW_PML_TEST(z < pml_z0h || z >= pml_z1h);
        bool const pml_y = DW_PML_TEST(y < pml_y0h || y >= pml_y1h);
        FUNC(backward_sigmayz_cell)(
            c55_zy_shot, buoyancy_y_shot, buoyancy_z_shot, vy, vz, sigmayz,
            m_vyz, m_vzy, m_sigmayzz, m_sigmayzy, dvzdy_plus_dvydz_store,
            grad_c55_zy, dt, step_ratio, c55_requires_grad, i, j, ny, nx, z, y,
            x, pml_z, pml_y, az, azh, bz, bzh, ay, ayh, by, byh, rdz, rdy);
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
 * vti_kernels.h (FUNC(backward_vz_cell_guard) etc -- see that file's
 * "Guard-band gathers" module comment for the full rationale). The two
 * kernels below (backward_kernel_sigma_guard, backward_kernel_v_guard) are
 * guard-band-only companions to backward_kernel_sigma/backward_kernel_v
 * above: same fields, same per-cell math (via the *_guard functions), but
 * launched over the WHOLE domain (grid indices start at 0, not FD_PAD)
 * since a guard cell by definition lives inside the band backward_kernel_
 * sigma/backward_kernel_v exclude. Each thread computes at most one update
 * per field at its own, unique (shot, z, y, x) -- no two threads ever
 * accumulate into the same output index -- so these are race-free without
 * atomics purely from CUDA's per-thread-independent-output property;
 * ordering versus every other kernel here is guaranteed by the single
 * compute stream (stream_compute) every launch in this file uses, so
 * placement (immediately after the interior kernel, before the
 * corresponding fold-fill launch -- see FUNC(backward)'s driver body below)
 * is the only thing that matters, not concurrency. backward_kernel_v_pml_
 * mem is the post-fold M_SIGMA** PML-memory-variable update and buoyancy-
 * gradient pass elastic_vti_kernels.h's own module comment describes (both
 * moved out of FUNC(backward_v*_cell) into a separate pass so they read
 * v*[i] after the velocity fold changes near-face interior cells, e.g. a
 * rigid on-row velocity clamp node); it is launched unconditionally (the
 * PML-memory write matters every time step regardless of buoyancy_
 * requires_grad) over (approximately) the same unextended range FUNC(
 * forward_vz_cell) etc. used to write dv*dbuoyancy in the forward pass.
 */
__launch_bounds__(128) __global__ void backward_kernel_sigma_guard(
    DW_DTYPE const *__restrict const c11,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c12,
#endif
    DW_DTYPE const *__restrict const c13, DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c55_zy,
    DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const sigmazz,
    DW_DTYPE const *__restrict const sigmayz,
    DW_DTYPE const *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const sigmayy,
    DW_DTYPE const *__restrict const sigmaxy,
#endif
    DW_DTYPE const *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_vzz,
    DW_DTYPE const *__restrict const m_vzy,
    DW_DTYPE const *__restrict const m_vzx,
    DW_DTYPE const *__restrict const m_vyz,
    DW_DTYPE const *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_vyy,
    DW_DTYPE const *__restrict const m_vyx,
    DW_DTYPE const *__restrict const m_vxy,
#endif
    DW_DTYPE const *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
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
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0, int64_t const bc_y1,
#endif
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
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c12_shot =
          c12_batched ? c12 + shot_idx * shot_numel : c12;
#endif
      DW_DTYPE const *__restrict const c13_shot =
          c13_batched ? c13 + shot_idx * shot_numel : c13;
      DW_DTYPE const *__restrict const c33_shot =
          c33_batched ? c33 + shot_idx * shot_numel : c33;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c55_zy_shot =
          c55_batched ? c55_zy + shot_idx * shot_numel : c55_zy;
      DW_DTYPE const *__restrict const c55_zx_shot =
          c55_batched ? c55_zx + shot_idx * shot_numel : c55_zx;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const c_xy_shot =
#if DW_NDIM >= 3
          c66_batched ? c_xy + shot_idx * shot_numel : c_xy;
#else
          c55_batched ? c_xy + shot_idx * shot_numel : c_xy;
#endif
#endif
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const *__restrict const buoyancy_x_shot =
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
           * the "Guard-band gathers" module comment in elastic_vti_
           * kernels.h), so there is no PML-memory recursion state to
           * read/write here; always take the plain (non-PML) branch. */
          bool const pml_z = false;
          bool const pml_y = false;
          bool const pml_x = false;
          FUNC(backward_vz_cell_guard)(
              c13_shot, c33_shot, c55_zy_shot, c55_zx_shot, sigmazz, sigmayy,
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
          FUNC(backward_vy_cell_guard)(
#if DW_NDIM >= 3
              c55_zy_shot, c11_shot, c12_shot, c13_shot,
#else
              c13_shot, c33_shot,
#endif
              c_xy_shot,
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
              m_sigmayyy, m_sigmayyyn, m_sigmaxyx, m_sigmaxyxn,
              buoyancy_y_shot[j],
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
          /* Guard cells have no forward-pass PML state -- see the module
           * comment above the vz guard-band block. */
#if DW_NDIM >= 3
          bool const pml_z = false;
#endif
#if DW_NDIM >= 2
          bool const pml_y = false;
#endif
          bool const pml_x = false;
          FUNC(backward_vx_cell_guard)(
#if DW_NDIM >= 3
              c55_zx_shot,
#endif
#if DW_NDIM >= 2
              c_xy_shot,
#endif
              c11_shot,
#if DW_NDIM >= 2
              c13_shot,
#endif
#if DW_NDIM >= 3
              c12_shot,
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
              m_sigmaxxx, m_sigmaxxxn, buoyancy_x_shot[j],
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

__launch_bounds__(128) __global__ void backward_kernel_v_pml_mem(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmazzz,
    DW_DTYPE *__restrict const m_sigmazzzn,
    DW_DTYPE const *__restrict const m_sigmayzy,
    DW_DTYPE *__restrict const m_sigmayzyn,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE *__restrict const m_sigmaxzxn,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE *__restrict const m_sigmayzzn,
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE *__restrict const m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmayyy,
    DW_DTYPE *__restrict const m_sigmayyyn,
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE *__restrict const m_sigmaxyyn,
    DW_DTYPE const *__restrict const m_sigmaxyx,
    DW_DTYPE *__restrict const m_sigmaxyxn,
#endif
    DW_DTYPE const *__restrict const m_sigmaxxx,
    DW_DTYPE *__restrict const m_sigmaxxxn,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdbuoyancy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const dvydbuoyancy,
#endif
    DW_DTYPE const *__restrict const dvxdbuoyancy,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_buoyancy_y,
#endif
    DW_DTYPE *__restrict const grad_buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
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
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      /* M_SIGMA** PML memory-variable update (split out of FUNC(backward_
       * v*_cell) -- see that split's comment in elastic_vti_kernels.h) and
       * the buoyancy-gradient accumulation both need v*[i] read post-fold
       * (the fold kernels ran just before this one on the same stream),
       * and both use the SAME (unextended) interior/PML range FUNC(
       * forward_vz_cell) etc. used in the forward pass -- see elastic_
       * vti.c's own comment at its equivalent call site. The PML
       * memory-variable write runs unconditionally; the gradient
       * accumulation is gated on buoyancy_requires_grad_t. */
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
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const c55_zy,
    DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const c_xy,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
#endif
    DW_DTYPE const *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const sigmazz, DW_DTYPE *__restrict const sigmayz,
    DW_DTYPE *__restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const sigmayy, DW_DTYPE *__restrict const sigmaxy,
#endif
    DW_DTYPE *__restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz, DW_DTYPE *__restrict const m_vzy,
    DW_DTYPE *__restrict const m_vzx, DW_DTYPE *__restrict const m_vyz,
    DW_DTYPE *__restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vyx,
    DW_DTYPE *__restrict const m_vxy,
#endif
    DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const m_sigmaxzz,
    DW_DTYPE const *__restrict const m_sigmaxzx,
    DW_DTYPE const *__restrict const m_sigmayzz,
    DW_DTYPE const *__restrict const m_sigmayzy,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const m_sigmaxyy,
    DW_DTYPE const *__restrict const m_sigmaxyx,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const *__restrict const bxh,
#if DW_NDIM >= 3
    int64_t const bc_z0, int64_t const bc_z1,
#endif
#if DW_NDIM >= 2
    int64_t const bc_y0, int64_t const bc_y1,
#endif
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
      DW_DTYPE const *__restrict const buoyancy_x_shot =
          buoyancy_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const buoyancy_z_shot =
          buoyancy_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const buoyancy_y_shot =
          buoyancy_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
#if DW_NDIM >= 2
      DW_DTYPE const *__restrict const c_xy_shot =
#if DW_NDIM >= 3
          c66_batched ? c_xy + shot_idx * shot_numel : c_xy;
#else
          c55_batched ? c_xy + shot_idx * shot_numel : c_xy;
#endif
#endif
#if DW_NDIM >= 3
      DW_DTYPE const *__restrict const c55_zx_shot =
          c55_batched ? c55_zx + shot_idx * shot_numel : c55_zx;
      DW_DTYPE const *__restrict const c55_zy_shot =
          c55_batched ? c55_zy + shot_idx * shot_numel : c55_zy;
#endif

      int64_t const i = shot_idx * shot_numel + j;

#if DW_NDIM >= 3
      {
        FUNC(dw_ext_range)
        const rz = FUNC(dw_compute_ext_range)(nz, bc_z0, bc_z1, false);
        if (FUNC(dw_in_target)(rz, z) && !FUNC(dw_in_std)(rz, z) &&
            y >= FD_PAD && y < ny - FD_PAD + 1 && x >= FD_PAD &&
            x < nx - FD_PAD + 1) {
          FUNC(backward_sigmazz_z_cell_guard)(buoyancy_z_shot, vz, sigmazz, dt,
                                              i, j, nz, ny, nx, z, rdz);
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
          FUNC(backward_sigmayy_y_cell_guard)(buoyancy_y_shot, vy, sigmayy, dt,
                                              i, j, ny, nx, y, rdy);
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
          FUNC(backward_sigmaxx_x_cell_guard)(buoyancy_x_shot, vx, sigmaxx, dt,
                                              i, j,
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
          FUNC(backward_sigmaxy_cell_guard)(
              c_xy_shot, buoyancy_x_shot, buoyancy_y_shot, vx, vy, sigmaxy,
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
          FUNC(backward_sigmaxz_cell_guard)(
              c55_zx_shot, buoyancy_x_shot, buoyancy_z_shot, vx, vz, sigmaxz,
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
          FUNC(backward_sigmayz_cell_guard)(
              c55_zy_shot, buoyancy_y_shot, buoyancy_z_shot, vy, vz, sigmayz,
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
               int64_t const n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
               int64_t const n_sources_y_per_shot_h,
#endif
               int64_t const n_sources_x_per_shot_h,
               int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
               int64_t const n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
               int64_t const n_receivers_y_per_shot_h,
#endif
               int64_t const n_receivers_x_per_shot_h,
               int64_t const n_receivers_p_per_shot_h,
               int64_t const step_ratio_h,
#if DW_NDIM >= 3
               int64_t const pml_z0_h, int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
               int64_t const pml_y0_h, int64_t const pml_y1_h,
#endif
               int64_t const pml_x0_h, int64_t const pml_x1_h,
               bool const c11_batched_h,
#if DW_NDIM >= 3
               bool const c12_batched_h,
#endif
               bool const c13_batched_h, bool const c33_batched_h,
               bool const c55_batched_h,
#if DW_NDIM >= 3
               bool const c66_batched_h,
#endif
               bool const buoyancy_batched_h) {
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
  gpuErrchk(cudaMemcpyToSymbol(n_sources_z_per_shot, &n_sources_z_per_shot_h,
                               sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(n_sources_y_per_shot, &n_sources_y_per_shot_h,
                               sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(n_sources_x_per_shot, &n_sources_x_per_shot_h,
                               sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_p_per_shot, &n_sources_p_per_shot_h,
                               sizeof(int64_t)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_z_per_shot,
                               &n_receivers_z_per_shot_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_y_per_shot,
                               &n_receivers_y_per_shot_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_x_per_shot,
                               &n_receivers_x_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_p_per_shot,
                               &n_receivers_p_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
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
  gpuErrchk(cudaMemcpyToSymbol(c11_batched, &c11_batched_h, sizeof(bool)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(c12_batched, &c12_batched_h, sizeof(bool)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(c13_batched, &c13_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(c33_batched, &c33_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(c55_batched, &c55_batched_h, sizeof(bool)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(c66_batched, &c66_batched_h, sizeof(bool)));
#endif
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
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const c12,
#endif
        DW_DTYPE const *__restrict const c13,
        DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const c55_zy,
        DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict const c_xy,
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
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1a,
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1b,
        void *__restrict const dvzdx_plus_dvxdz_store_2,
        void *__restrict const dvzdx_plus_dvxdz_store_3,
        char const *__restrict const
            *__restrict const dvzdx_plus_dvxdz_filenames_ptr,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1a,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1b,
        void *__restrict const dvzdy_plus_dvydz_store_2,
        void *__restrict const dvzdy_plus_dvydz_store_3,
        char const *__restrict const
            *__restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
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
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1a,
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1b,
        void *__restrict const dvydx_plus_dvxdy_store_2,
        void *__restrict const dvydx_plus_dvxdy_store_3,
        char const *__restrict const
            *__restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
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
        int64_t const n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_per_shot_h,
#endif
        int64_t const n_sources_x_per_shot_h,
        int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_per_shot_h,
#endif
        int64_t n_receivers_x_per_shot_h,
        int64_t const n_receivers_p_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const c11_requires_grad,
#if DW_NDIM >= 3
        bool const c12_requires_grad,
#endif
        bool const c13_requires_grad, bool const c33_requires_grad,
        bool const c55_requires_grad,
#if DW_NDIM >= 3
        bool const c66_requires_grad,
#endif
        bool const buoyancy_requires_grad, bool const c11_batched_h,
#if DW_NDIM >= 3
        bool const c12_batched_h,
#endif
        bool const c13_batched_h, bool const c33_batched_h,
        bool const c55_batched_h,
#if DW_NDIM >= 3
        bool const c66_batched_h,
#endif
        bool const buoyancy_batched_h, bool const storage_compression,
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
    if (step_ratio_h <= 0) return 1;

#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot_h > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot_h > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot_h > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot_h > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot_h > 0) && sources_z_i == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot_h > 0) && sources_y_i == NULL) return 1;
#endif
    if ((n_sources_x_per_shot_h > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_p_per_shot_h > 0) && sources_p_i == NULL) return 1;
    bool const dw_has_sample = (start_t % step_ratio_h == 0 ||
                                nt > step_ratio_h - start_t % step_ratio_h);
#if DW_NDIM >= 3
    if (dw_has_sample && (buoyancy_requires_grad) &&
        (dvzdbuoyancy_store_1a == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if (dw_has_sample && (buoyancy_requires_grad) &&
        (dvydbuoyancy_store_1a == NULL))
      return 1;
#endif
    if (dw_has_sample && (buoyancy_requires_grad) &&
        (dvxdbuoyancy_store_1a == NULL))
      return 1;
  }
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 2 */
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#endif
#if DW_NDIM == 3
  dim3 const dimBlock(DW_STENCIL_BLOCK_3D);
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

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
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
                         n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
                         n_sources_y_per_shot_h,
#endif
                         n_sources_x_per_shot_h, n_sources_p_per_shot_h,
#if DW_NDIM >= 3
                         n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
                         n_receivers_y_per_shot_h,
#endif
                         n_receivers_x_per_shot_h, n_receivers_p_per_shot_h,
                         step_ratio_h,
#if DW_NDIM >= 3
                         pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
                         pml_y0_h, pml_y1_h,
#endif
                         pml_x0_h, pml_x1_h, c11_batched_h,
#if DW_NDIM >= 3
                         c12_batched_h,
#endif
                         c13_batched_h, c33_batched_h, c55_batched_h,
#if DW_NDIM >= 3
                         c66_batched_h,
#endif
                         buoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel = ny_h * nx_h;
#else
  int64_t const shot_numel = nx_h;
#endif

  // Aggregate requires-grad conditions gating the stored velocity derivatives
  // (must match Python's storage_flags sizing): any normal-node stiffness
  // drives the dvidi stores; the sigma_xy-node shear (c66 in 3-D, c55 in 2-D)
  // drives its store.
#if DW_NDIM >= 3
  bool const cnormal_requires_grad = c11_requires_grad || c12_requires_grad ||
                                     c13_requires_grad || c33_requires_grad;
  bool const c_xy_requires_grad = c66_requires_grad;
#else
  bool const cnormal_requires_grad =
      c11_requires_grad || c13_requires_grad || c33_requires_grad;
  bool const c_xy_requires_grad = c55_requires_grad;
#endif

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
       c55_requires_grad ||
#if DW_NDIM >= 3
       c12_requires_grad || c66_requires_grad ||
#endif
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
  OPEN_FILE(dvzdx_plus_dvxdz, c55_requires_grad)
  OPEN_FILE(dvzdy_plus_dvydz, c55_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE(dvydbuoyancy, buoyancy_requires_grad)
  OPEN_FILE(dvydy, cnormal_requires_grad)
  OPEN_FILE(dvydx_plus_dvxdy, c_xy_requires_grad)
#endif
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
    SETUP_STORE(dvzdx_plus_dvxdz, c55_requires_grad)
    SETUP_STORE(dvzdy_plus_dvydz, c55_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE(dvydbuoyancy, buoyancy_requires_grad)
    SETUP_STORE(dvydy, cnormal_requires_grad)
    SETUP_STORE(dvydx_plus_dvxdy, c_xy_requires_grad)
#endif
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

    /* Refill the stress guard ring (read by the velocity-update kernel via
     * sigma_AA/the shared-axis shears' derivatives, and by the pressure
     * receiver record below); mirrors elastic.cu's own
     * LAUNCH_FILL_ELASTIC_STRESS_BC placement (before receiver recording,
     * before the velocity-update kernel launch) and elastic_vti.c's own
     * forward stress-fill placement. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_STRESS_BC();
      CHECK_KERNEL_ERROR
    }

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
#if DW_NDIM >= 2
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
#endif
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
    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_pressure_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                  stream_compute>>>(
          r_p + t * n_shots_h * n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, receivers_p_i);
      CHECK_KERNEL_ERROR
    }

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
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
        m_sigmazzz, m_sigmayzy, m_sigmaxzx, m_sigmayzz, m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        m_sigmayyy, m_sigmaxyy, m_sigmaxyx,
#endif
        m_sigmaxxx,
#if DW_NDIM >= 3
        dvzdbuoyancy_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydbuoyancy_store_1_t,
#endif
        dvxdbuoyancy_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, buoyancy_requires_grad && ((t % step_ratio_h) == 0));
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
#if DW_NDIM >= 2
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
#endif
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

    /* Refill the velocity guard ring (read by every normal stress's own
     * divergence term and by the shared-axis shears' updates); mirrors
     * elastic.cu's own LAUNCH_FILL_ELASTIC_VELOCITY_BC placement (after
     * velocity source injection, before the stress-update kernel launch)
     * and elastic_vti.c's own forward velocity-fill placement. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_VELOCITY_BC();
      CHECK_KERNEL_ERROR
    }

    forward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        c11,
#if DW_NDIM >= 3
        c12,
#endif
        c13, c33,
#if DW_NDIM >= 3
        c55_zy, c55_zx,
#endif
#if DW_NDIM >= 2
        c_xy,
#endif
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
        dvzdz_store_1_t, dvzdx_plus_dvxdz_store_1_t, dvzdy_plus_dvydz_store_1_t,
#endif
#if DW_NDIM >= 2
        dvydy_store_1_t, dvydx_plus_dvxdy_store_1_t,
#endif
        dvxdx_store_1_t,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, c11_requires_grad && ((t % step_ratio_h) == 0),
#if DW_NDIM >= 3
        c12_requires_grad && ((t % step_ratio_h) == 0),
#endif
        c13_requires_grad && ((t % step_ratio_h) == 0),
        c33_requires_grad && ((t % step_ratio_h) == 0),
#if DW_NDIM >= 3
        c55_requires_grad && ((t % step_ratio_h) == 0),
        c66_requires_grad && ((t % step_ratio_h) == 0)
#else
        c55_requires_grad && ((t % step_ratio_h) == 0)
#endif
    );
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

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources_p<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, f_p + t * n_shots_h * n_sources_p_per_shot_h, sources_p_i);
      CHECK_KERNEL_ERROR
    }
  }

  /* Leave the returned wavefields boundary-consistent (e.g. free-surface
   * stress nodes held to zero, guard rings imaged), matching
   * _elastic_vti_eager_core's own post-loop refill in elastic_vti.py and
   * elastic_vti.c's own matching fix -- without this, the last timestep's
   * stress update runs after the last in-loop stress fill, so the returned
   * sigma guard ring/on-surface zero reflect a stale (pre-update) state.
   * Receiver traces and gradients are unaffected (already recorded/
   * consumed inside the loop before this point); only the final returned
   * wavefields change. */
  if (has_bc) {
    LAUNCH_FILL_ELASTIC_STRESS_BC();
    CHECK_KERNEL_ERROR
    LAUNCH_FILL_ELASTIC_VELOCITY_BC();
    CHECK_KERNEL_ERROR
  }

  if (use_double_buffering)
    gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(backward)(
        DW_DTYPE const *__restrict const c11,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const c12,
#endif
        DW_DTYPE const *__restrict const c13,
        DW_DTYPE const *__restrict const c33,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict const c55_zy,
        DW_DTYPE const *__restrict const c55_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict const c_xy,
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
        DW_DTYPE *__restrict const m_vxz, DW_DTYPE *__restrict m_sigmazzz,
        DW_DTYPE *__restrict m_sigmayzy, DW_DTYPE *__restrict m_sigmaxzx,
        DW_DTYPE *__restrict m_sigmayzz, DW_DTYPE *__restrict m_sigmaxzz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const vy, DW_DTYPE *__restrict const sigmayy,
        DW_DTYPE *__restrict const sigmaxy, DW_DTYPE *__restrict const m_vyy,
        DW_DTYPE *__restrict const m_vyx, DW_DTYPE *__restrict const m_vxy,
        DW_DTYPE *__restrict m_sigmayyy, DW_DTYPE *__restrict m_sigmaxyy,
        DW_DTYPE *__restrict m_sigmaxyx,
#endif
        DW_DTYPE *__restrict const vx, DW_DTYPE *__restrict const sigmaxx,
        DW_DTYPE *__restrict const m_vxx, DW_DTYPE *__restrict m_sigmaxxx,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict m_sigmazzzn, DW_DTYPE *__restrict m_sigmayzyn,
        DW_DTYPE *__restrict m_sigmaxzxn, DW_DTYPE *__restrict m_sigmayzzn,
        DW_DTYPE *__restrict m_sigmaxzzn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict m_sigmayyyn, DW_DTYPE *__restrict m_sigmaxyyn,
        DW_DTYPE *__restrict m_sigmaxyxn,
#endif
        DW_DTYPE *__restrict m_sigmaxxxn,
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
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1a,
        DW_DTYPE *__restrict const dvzdx_plus_dvxdz_store_1b,
        void *__restrict const dvzdx_plus_dvxdz_store_2,
        void *__restrict const dvzdx_plus_dvxdz_store_3,
        char const *__restrict const
            *__restrict const dvzdx_plus_dvxdz_filenames_ptr,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1a,
        DW_DTYPE *__restrict const dvzdy_plus_dvydz_store_1b,
        void *__restrict const dvzdy_plus_dvydz_store_2,
        void *__restrict const dvzdy_plus_dvydz_store_3,
        char const *__restrict const
            *__restrict const dvzdy_plus_dvydz_filenames_ptr,
#endif
#if DW_NDIM >= 2
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
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1a,
        DW_DTYPE *__restrict const dvydx_plus_dvxdy_store_1b,
        void *__restrict const dvydx_plus_dvxdy_store_2,
        void *__restrict const dvydx_plus_dvxdy_store_3,
        char const *__restrict const
            *__restrict const dvydx_plus_dvxdy_filenames_ptr,
#endif
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
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_f_y,
#endif
        DW_DTYPE *__restrict const grad_f_x,
        DW_DTYPE *__restrict const grad_f_p,
        DW_DTYPE *__restrict const grad_c11,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_c12,
#endif
        DW_DTYPE *__restrict const grad_c13,
        DW_DTYPE *__restrict const grad_c33,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_c55_zy,
        DW_DTYPE *__restrict const grad_c55_zx,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_c_xy,
#endif
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_buoyancy_y,
#endif
        DW_DTYPE *__restrict const grad_buoyancy_x,
        DW_DTYPE *__restrict const grad_c11_shot,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_c12_shot,
#endif
        DW_DTYPE *__restrict const grad_c13_shot,
        DW_DTYPE *__restrict const grad_c33_shot,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_c55_zy_shot,
        DW_DTYPE *__restrict const grad_c55_zx_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_c_xy_shot,
#endif
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_buoyancy_y_shot,
#endif
        DW_DTYPE *__restrict const grad_buoyancy_x_shot,
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
        int64_t const n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_y_per_shot_h,
#endif
        int64_t const n_sources_x_per_shot_h,
        int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_y_per_shot_h,
#endif
        int64_t n_receivers_x_per_shot_h,
        int64_t const n_receivers_p_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const c11_requires_grad,
#if DW_NDIM >= 3
        bool const c12_requires_grad,
#endif
        bool const c13_requires_grad, bool const c33_requires_grad,
        bool const c55_requires_grad,
#if DW_NDIM >= 3
        bool const c66_requires_grad,
#endif
        bool const buoyancy_requires_grad, bool const c11_batched_h,
#if DW_NDIM >= 3
        bool const c12_batched_h,
#endif
        bool const c13_batched_h, bool const c33_batched_h,
        bool const c55_batched_h,
#if DW_NDIM >= 3
        bool const c66_batched_h,
#endif
        bool const buoyancy_batched_h, bool const storage_compression,
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
    if (step_ratio_h <= 0) return 1;
    if ((c11_requires_grad) && (grad_c11 == NULL)) return 1;
#if DW_NDIM >= 3
    if ((c12_requires_grad) && (grad_c12 == NULL)) return 1;
#endif
    if ((c13_requires_grad) && (grad_c13 == NULL)) return 1;
    if ((c33_requires_grad) && (grad_c33 == NULL)) return 1;
#if DW_NDIM >= 3
    if ((buoyancy_requires_grad) && (grad_buoyancy_z == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((buoyancy_requires_grad) && (grad_buoyancy_y == NULL)) return 1;
#endif
    if ((buoyancy_requires_grad) && (grad_buoyancy_x == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_z_per_shot_h > 0) && (receivers_z_i == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_y_per_shot_h > 0) && (receivers_y_i == NULL)) return 1;
#endif
    if ((n_receivers_x_per_shot_h > 0) && (receivers_x_i == NULL)) return 1;
    if ((n_receivers_p_per_shot_h > 0) && (receivers_p_i == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_sources_z_per_shot_h > 0) && sources_z_i == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_y_per_shot_h > 0) && sources_y_i == NULL) return 1;
#endif
    if ((n_sources_x_per_shot_h > 0) && sources_x_i == NULL) return 1;
    if ((n_sources_p_per_shot_h > 0) && sources_p_i == NULL) return 1;
  }
#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else /* DW_NDIM == 2 */
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#endif
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

  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
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
                         n_sources_z_per_shot_h,
#endif
#if DW_NDIM >= 2
                         n_sources_y_per_shot_h,
#endif
                         n_sources_x_per_shot_h, n_sources_p_per_shot_h,
#if DW_NDIM >= 3
                         n_receivers_z_per_shot_h,
#endif
#if DW_NDIM >= 2
                         n_receivers_y_per_shot_h,
#endif
                         n_receivers_x_per_shot_h, n_receivers_p_per_shot_h,
                         step_ratio_h,
#if DW_NDIM >= 3
                         pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
                         pml_y0_h, pml_y1_h,
#endif
                         pml_x0_h, pml_x1_h, c11_batched_h,
#if DW_NDIM >= 3
                         c12_batched_h,
#endif
                         c13_batched_h, c33_batched_h, c55_batched_h,
#if DW_NDIM >= 3
                         c66_batched_h,
#endif
                         buoyancy_batched_h);
    if (err != 0) return err;
  }

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel = ny_h * nx_h;
#else
  int64_t const shot_numel = nx_h;
#endif

  // Aggregate requires-grad conditions gating the stored velocity derivatives
  // (must match Python's storage_flags sizing): any normal-node stiffness
  // drives the dvidi stores; the sigma_xy-node shear (c66 in 3-D, c55 in 2-D)
  // drives its store.
#if DW_NDIM >= 3
  bool const cnormal_requires_grad = c11_requires_grad || c12_requires_grad ||
                                     c13_requires_grad || c33_requires_grad;
  bool const c_xy_requires_grad = c66_requires_grad;
#else
  bool const cnormal_requires_grad =
      c11_requires_grad || c13_requires_grad || c33_requires_grad;
  bool const c_xy_requires_grad = c55_requires_grad;
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
      (c11_requires_grad || c13_requires_grad || c33_requires_grad ||
       c55_requires_grad ||
#if DW_NDIM >= 3
       c12_requires_grad || c66_requires_grad ||
#endif
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
  OPEN_FILE_READ(dvzdx_plus_dvxdz, c55_requires_grad)
  OPEN_FILE_READ(dvzdy_plus_dvydz, c55_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(dvydbuoyancy, buoyancy_requires_grad)
  OPEN_FILE_READ(dvydy, cnormal_requires_grad)
  OPEN_FILE_READ(dvydx_plus_dvxdy, c_xy_requires_grad)
#endif
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
    SETUP_STORE_LOAD(dvzdx_plus_dvxdz, c55_requires_grad)
    SETUP_STORE_LOAD(dvzdy_plus_dvydz, c55_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(dvydbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvydy, cnormal_requires_grad)
    SETUP_STORE_LOAD(dvydx_plus_dvxdy, c_xy_requires_grad)
#endif
    SETUP_STORE_LOAD(dvxdbuoyancy, buoyancy_requires_grad)
    SETUP_STORE_LOAD(dvxdx, cnormal_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    /* Fold the (adjoint) stress guard ring back into the interior before
     * backward_kernel_sigma (the adjoint velocity update) reads it and
     * before the source-gradient extraction just below (the adjoint
     * counterpart of forward's pressure receiver record). Unlike
     * acoustic's pressure fill, elastic_vti's multi-field stress fill is
     * not self-adjoint (see the module comment above launch_fill_elastic_
     * stress_bc_adjoint), so this applies that fill's true transpose
     * rather than reapplying the forward fill; mirrors elastic_vti.c's own
     * backward stress-fill fix. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_STRESS_BC_ADJOINT();
      CHECK_KERNEL_ERROR
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      record_adjoint_pressure_receivers<<<dimGrid_sources, dimBlock_src_rcv, 0,
                                          stream_compute>>>(
          grad_f_p + t * n_shots_h * n_sources_p_per_shot_h,
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, sources_p_i);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_sigma<<<dimGrid, dimBlock, 0, stream_compute>>>(
        c11,
#if DW_NDIM >= 3
        c12,
#endif
        c13, c33,
#if DW_NDIM >= 3
        c55_zy, c55_zx,
#endif
#if DW_NDIM >= 2
        c_xy,
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
        grad_buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
        grad_buoyancy_y_shot,
#endif
        grad_buoyancy_x_shot,
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, buoyancy_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

    /* Guard-band extension of the velocity-adjoint gathers -- see the
     * "Guard-band gathers" module comment above backward_kernel_sigma_
     * guard. Must run (and complete, via the shared stream_compute
     * ordering) before FUNC(fill_elastic_velocity_bc_adjoint)'s kernels
     * just below, which fold this guard-band deposit into the interior. */
    if (has_bc) {
      backward_kernel_sigma_guard<<<dimGrid_guard, dimBlock, 0,
                                    stream_compute>>>(
          c11,
#if DW_NDIM >= 3
          c12,
#endif
          c13, c33,
#if DW_NDIM >= 3
          c55_zy, c55_zx,
#endif
#if DW_NDIM >= 2
          c_xy,
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

    /* Fold the (adjoint) velocity guard ring back into the interior before
     * backward_kernel_v (the adjoint stress update) reads it. Unlike
     * acoustic's velocity fill, elastic_vti's multi-field velocity fill is
     * not self-adjoint (see the module comment above launch_fill_elastic_
     * velocity_bc_adjoint), so this applies that fill's true transpose
     * rather than reapplying the forward fill; mirrors elastic_vti.c's own
     * backward velocity-fill fix. */
    if (has_bc) {
      LAUNCH_FILL_ELASTIC_VELOCITY_BC_ADJOINT();
      CHECK_KERNEL_ERROR
    }

    /* Velocity-type source-gradient extraction and the buoyancy-gradient
     * accumulation both read the velocity adjoint at (near-face) interior
     * cells the fold above may have just changed, so both must run after
     * it -- see elastic_vti_kernels.h's "Guard-band gathers" module comment
     * and elastic_vti.c's own driver-side comment at the equivalent
     * point. */

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
#if DW_NDIM >= 2
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
#endif
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

    /* M_SIGMA** PML memory-variable update (split out of FUNC(backward_
     * v*_cell) -- see that split's comment in elastic_vti_kernels.h) and
     * the buoyancy-gradient accumulation, run over the same (unextended)
     * range FUNC(forward_vz_cell) etc. used in the forward pass. Both must
     * run after the velocity fold above (a pre-fold read of a near-face
     * interior cell -- e.g. a rigid on-row velocity clamp node -- is
     * stale); mirrors elastic_vti.c's own backward placement. */
    {
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
          grad_buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
          grad_buoyancy_y_shot,
#endif
          grad_buoyancy_x_shot,
#if DW_NDIM >= 3
          az, azh,
#endif
#if DW_NDIM >= 2
          ay, ayh,
#endif
          ax, axh, buoyancy_requires_grad && ((t % step_ratio_h) == 0));
      CHECK_KERNEL_ERROR
    }

    backward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        c11,
#if DW_NDIM >= 3
        c12,
#endif
        c13, c33,
#if DW_NDIM >= 3
        c55_zy, c55_zx,
#endif
#if DW_NDIM >= 2
        c_xy,
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
        dvxdx_store_1_t, grad_c11_shot,
#if DW_NDIM >= 3
        grad_c12_shot,
#endif
        grad_c13_shot, grad_c33_shot,
#if DW_NDIM >= 3
        grad_c55_zy_shot, grad_c55_zx_shot,
#endif
#if DW_NDIM >= 2
        grad_c_xy_shot,
#endif
#if DW_NDIM >= 3
        az, azh, bz, bzh,
#endif
#if DW_NDIM >= 2
        ay, ayh, by, byh,
#endif
        ax, axh, bx, bxh, c11_requires_grad && ((t % step_ratio_h) == 0),
#if DW_NDIM >= 3
        c12_requires_grad && ((t % step_ratio_h) == 0),
#endif
        c13_requires_grad && ((t % step_ratio_h) == 0),
        c33_requires_grad && ((t % step_ratio_h) == 0),
#if DW_NDIM >= 3
        c55_requires_grad && ((t % step_ratio_h) == 0),
        c66_requires_grad && ((t % step_ratio_h) == 0)
#else
        c55_requires_grad && ((t % step_ratio_h) == 0)
#endif
    );
    CHECK_KERNEL_ERROR

    /* Guard-band extension of the stress-adjoint gathers -- see the
     * "Guard-band gathers" module comment above backward_kernel_sigma_
     * guard. Must run (and complete) before the NEXT time step's
     * FUNC(fill_elastic_stress_bc_adjoint) call (at the top of this loop),
     * which folds this guard-band deposit into the interior; the shared
     * stream_compute ordering guarantees that regardless of where in this
     * iteration the launch sits. */
    if (has_bc) {
      backward_kernel_v_guard<<<dimGrid_guard, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
          c55_zy, c55_zx,
#endif
#if DW_NDIM >= 2
          c_xy,
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
#if DW_NDIM >= 2
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
#endif
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
    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      add_adjoint_pressure_sources<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                                     stream_compute>>>(
#if DW_NDIM >= 3
          sigmazz,
#endif
#if DW_NDIM >= 2
          sigmayy,
#endif
          sigmaxx, grad_r_p + t * n_shots_h * n_receivers_p_per_shot_h,
          receivers_p_i);
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

  if (c11_requires_grad && !c11_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c11, grad_c11_shot);
    CHECK_KERNEL_ERROR
  }
#if DW_NDIM >= 3
  if (c12_requires_grad && !c12_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c12, grad_c12_shot);
    CHECK_KERNEL_ERROR
  }
#endif
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
  if (c55_requires_grad && !c55_batched_h && n_shots_h > 1) {
#if DW_NDIM >= 3
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c55_zy, grad_c55_zy_shot);
    CHECK_KERNEL_ERROR
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c55_zx, grad_c55_zx_shot);
    CHECK_KERNEL_ERROR
#endif
  }
#if DW_NDIM >= 3
  if (c66_requires_grad && !c66_batched_h && n_shots_h > 1) {
#else
  if (c55_requires_grad && !c55_batched_h && n_shots_h > 1) {
#endif
#if DW_NDIM >= 2
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c_xy, grad_c_xy_shot);
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
  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

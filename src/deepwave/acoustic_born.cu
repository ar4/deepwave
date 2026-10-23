/*
 * Variable-density acoustic Born (linearised) propagator (CUDA implementation).
 *
 * CUDA mirror of acoustic_born.c. Propagates a background velocity-pressure
 * field (p, v_d) and a scattered field (dp, dv_d) simultaneously, for
 * perturbations of the constitutive models: bulk modulus dK (pressure node)
 * and buoyancy dB_d (v_d node). The scattered field satisfies the SAME
 * acoustic system with the background (K, B_d), driven by two virtual sources
 * (first-order Born):
 *
 *     scattered velocity:  dv_d -= dt*B_d*(grad_d(dp) + psi_sc_d)
 *                                  + dt*dB_d*term_bg_d
 *     scattered pressure:  dp   -= dt*K*(div(dv) + phi_sc)
 *                                  + dt*dK*div_bg
 *
 * Structure mirrors acoustic.cu (thread-per-grid-point kernels, psi/phi PML
 * ping-pong, storage snapshot save/load, per-shot gradient buffers combined
 * with combine_grad), extended with the scattered field and Born coupling.
 *
 * Exports: acoustic_born_iso_<ndim>d_<accuracy>_<dtype>_<forward|backward|
 * backward_sc>_cuda.
 */

#include <stdio.h>

#include <algorithm>  // for std::swap
#include <cstdint>

#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  acoustic_born_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define ACOUSTIC_BORN_VARIANT_I(name) name
#define ACOUSTIC_BORN_VARIANT(name) ACOUSTIC_BORN_VARIANT_I(name)
#define FUNC(name) \
  CAT(ACOUSTIC_BORN_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx_h
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* ND_INDEX, K/BUOYANCY_Z/Y/X/DK/DBUOYANCY_Z/Y/X, P/VZ/VY/VX/PSC/VZSC/VYSC/
 * VXSC, PHI_Z/Y/X/PSI_Z/Y/X (+_SC scattered twins), the VZ_ADJOINT/.../
 * P_ADJOINT_Z/Y/X (+_PML, +_BG, +_BG_PML) family, and the twelve
 * FUNC(forward_vz_cell)/.../FUNC(backward_p_cell) per-grid-point stencil
 * update functions live in acoustic_born_kernels.h, shared verbatim with
 * acoustic_born.c (see that header for the full explanation). p/vz/.../psc/
 * .../psi_z/... below are this file's (kernel-parameter or shot-offset-
 * local) names for the arrays/values passed as the p/vz/.../psc/.../psi_z/
 * .../k_shot/buoyancy_z_shot/... parameters of those functions. */
#include "acoustic_born_kernels.h"

namespace {
__constant__ DW_DTYPE dt;
#if DW_NDIM >= 3
__constant__ DW_DTYPE rdz;
__constant__ int64_t nz;
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
#endif
#if DW_NDIM >= 2
__constant__ DW_DTYPE rdy;
__constant__ int64_t ny;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
#endif
__constant__ DW_DTYPE rdx;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_shots;
__constant__ int64_t step_ratio;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool k_batched;
__constant__ bool b_batched;

/* First-class reflecting/periodic boundary faces on the staggered acoustic
 * Born grid (see docs/boundaries.rst and the validated acoustic.c/acoustic.cu
 * reference, which these device functions mirror exactly). Applied
 * independently to the background and scattered fields (both are an
 * ordinary acoustic (p, v_d) field on the same grid; see acoustic_born.py's
 * module docstring), exactly mirroring the eager Python reference and
 * acoustic_born.c's own CPU kernel. Only guard cells (and, for the
 * documented on-surface/on-wall cases, specific interior nodes) are touched;
 * the adjoint's model mirror-pad is applied separately in Python.
 * fill_p_bc_line_dev/fill_v_bc_line_dev/fill_bc_kernel live in
 * line_bc_dev.h, shared verbatim across every propagator's .cu file on this
 * staggered layout (including this one -- the historical
 * acoustic_born_-prefixed names were a same-project naming-collision
 * precaution, not a technical necessity: each propagator's CUDA sources
 * build into their own separate object library, so there is no actual
 * link-time collision risk from reusing the canonical unprefixed names). */
#include "line_bc_dev.h"

/* Pressure guard fill on every reflecting axis (innermost stride first). */
static void launch_acoustic_born_fill_p_bc(DW_DTYPE *__restrict const p,
                                           int64_t const bc[6],
                                           int64_t const total_numel,
                                           int64_t const nz_, int64_t const ny_,
                                           int64_t const nx_,
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
static void launch_acoustic_born_fill_v_bc(
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

#if DW_NDIM == 3
#define LAUNCH_FILL_P_BC(p)                                                  \
  launch_acoustic_born_fill_p_bc(p, bc, n_shots_h *shot_numel_h, nz_h, ny_h, \
                                 nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_BG()                                             \
  launch_acoustic_born_fill_v_bc(vz, vy, vx, bc, n_shots_h *shot_numel_h, \
                                 nz_h, ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_SC()                                               \
  launch_acoustic_born_fill_v_bc(vzsc, vysc, vxsc, bc,                      \
                                 n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                                 stream_compute)
#elif DW_NDIM == 2
#define LAUNCH_FILL_P_BC(p)                                               \
  launch_acoustic_born_fill_p_bc(p, bc, n_shots_h *shot_numel_h, 1, ny_h, \
                                 nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_BG()                                                  \
  launch_acoustic_born_fill_v_bc(vy, vx, bc, n_shots_h *shot_numel_h, 1, ny_h, \
                                 nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_SC()                                                \
  launch_acoustic_born_fill_v_bc(vysc, vxsc, bc, n_shots_h *shot_numel_h, 1, \
                                 ny_h, nx_h, stream_compute)
#else
#define LAUNCH_FILL_P_BC(p)                                                  \
  launch_acoustic_born_fill_p_bc(p, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                                 stream_compute)
#define LAUNCH_FILL_V_BC_BG()                                                 \
  launch_acoustic_born_fill_v_bc(vx, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                                 stream_compute)
#define LAUNCH_FILL_V_BC_SC()                                             \
  launch_acoustic_born_fill_v_bc(vxsc, bc, n_shots_h *shot_numel_h, 1, 1, \
                                 nx_h, stream_compute)
#endif

/* Periodic-face adjoint guard maintenance (backward/backward_sc only):
 * fold/gather line functions from acoustic_born_kernels.h, one thread per
 * line, per adjoint field set. The folds run over every line (like the
 * fills); the gathers run only over the other axes' interior node ranges
 * [FD_PAD, n - FD_PAD + 1). Mirrors acoustic.cu; the Born-coupled
 * background payloads are composed as two single-payload gather launches.
 * The LAUNCH_* wrappers are defined after the launchers below. */

__constant__ DW_DTYPE stag_w_dev[FD_PAD] = DW_STAG_TAYLOR_W;

/* adj_fold_bc_kernel/adj_fold_fr_bc_kernel live in adj_fold_bc_kernel_dev.h
 * (see that header). */
#include "adj_fold_bc_kernel_dev.h"

static void launch_adj_fold_p_bc(DW_DTYPE *__restrict const p,
                                 int64_t const bc[6], int64_t const total_numel,
                                 int64_t const nz_, int64_t const ny_,
                                 int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bc[4] == 3) {
    int64_t const nl = total_numel / nx_;
    adj_fold_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nx_, 1, 0);
  }
#if DW_NDIM >= 2
  if (bc[2] == 3) {
    int64_t const nl = total_numel / ny_;
    adj_fold_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, ny_, nx_, 0);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] == 3) {
    int64_t const nl = total_numel / nz_;
    adj_fold_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nz_, ny_ * nx_, 0);
  }
#endif
}

static void launch_adj_fold_v_bc(
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
  if (bc[4] == 3) {
    int64_t const nl = total_numel / nx_;
    adj_fold_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vx, nl, nx_, 1, 1);
  }
#if DW_NDIM >= 2
  if (bc[2] == 3) {
    int64_t const nl = total_numel / ny_;
    adj_fold_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vy, nl, ny_, nx_, 1);
  }
#endif
#if DW_NDIM >= 3
  if (bc[0] == 3) {
    int64_t const nl = total_numel / nz_;
    adj_fold_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, 1);
  }
#endif
}

/* One thread per (shot, other-axis interior position) line. `kind` 0 =
 * pressure guards, 1 = velocity guards (see acoustic.cu's kernel of the
 * same name). */
__global__ void adj_gather_guard_kernel(
    DW_DTYPE *__restrict const fld, DW_DTYPE const *__restrict const src_fld,
    DW_DTYPE const *__restrict const model, bool const model_batched,
    int const kind, int64_t const n_lines, int64_t const nax,
    int64_t const stride, int64_t const n_a, int64_t const start_a,
    int64_t const stride_a, int64_t const n_b, int64_t const start_b,
    int64_t const stride_b, DW_DTYPE const rd) {
  int64_t const L = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (L >= n_lines) return;
  DW_DTYPE const *const sw = stag_w_dev;
  int64_t const per_shot = n_a * n_b;
  int64_t const shot_idx = L / per_shot;
  int64_t const r = L % per_shot;
  int64_t const a = start_a + r / n_b;
  int64_t const b = start_b + r % n_b;
  int64_t const jbase = a * stride_a + b * stride_b;
  int64_t const base = shot_idx * shot_numel + jbase;
  DW_DTYPE const *const model_shot =
      model_batched ? model + shot_idx * shot_numel : model;
  if (kind == 0)
    FUNC(adj_gather_p_guard_line)(fld, src_fld, model_shot, base, jbase, nax,
                                  stride, dt, rd, sw);
  else
    FUNC(adj_gather_v_guard_line)(fld, src_fld, model_shot, base, jbase, nax,
                                  stride, dt, rd, sw);
}

#define ADJ_GATHER_FD_ARG

static void launch_adj_gather_v_guards(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy_,
#endif
    DW_DTYPE *__restrict const vx_, DW_DTYPE const *__restrict const p_,
    DW_DTYPE const *__restrict const model, bool const model_batched,
    int64_t const bc[6], int64_t const n_shots_, int64_t const nz_,
    int64_t const ny_, int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  int64_t const mz = nz_ - 2 * FD_PAD + 1;
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * mz * my;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(vx_, p_, model, model_batched, 1, nl,
                                        nx_, 1, mz, FD_PAD, ny_ * nx_, my,
                                        FD_PAD, nx_, rdx_ ADJ_GATHER_FD_ARG);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(vy_, p_, model, model_batched, 1, nl,
                                        ny_, nx_, mz, FD_PAD, ny_ * nx_, mx,
                                        FD_PAD, 1, rdy_ ADJ_GATHER_FD_ARG);
  }
  if (bc[0] == 3) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(vz_, p_, model, model_batched, 1, nl,
                                        nz_, ny_ * nx_, my, FD_PAD, nx_, mx,
                                        FD_PAD, 1, rdz_ ADJ_GATHER_FD_ARG);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * my;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(vx_, p_, model, model_batched, 1, nl,
                                        nx_, 1, 1, 0, 0, my, FD_PAD, nx_,
                                        rdx_ ADJ_GATHER_FD_ARG);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(vy_, p_, model, model_batched, 1, nl,
                                        ny_, nx_, 1, 0, 0, mx, FD_PAD, 1,
                                        rdy_ ADJ_GATHER_FD_ARG);
  }
#else
  if (bc[4] == 3) {
    int64_t const nl = n_shots_;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(vx_, p_, model, model_batched, 1, nl,
                                        nx_, 1, 1, 0, 0, 1, 0, 0,
                                        rdx_ ADJ_GATHER_FD_ARG);
  }
#endif
}

static void launch_adj_gather_p_guards(
    DW_DTYPE *__restrict const p_,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz_,
    DW_DTYPE const *__restrict const model_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy_,
    DW_DTYPE const *__restrict const model_y,
#endif
    DW_DTYPE const *__restrict const vx_,
    DW_DTYPE const *__restrict const model_x, bool const model_batched,
    int64_t const bc[6], int64_t const n_shots_, int64_t const nz_,
    int64_t const ny_, int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  int64_t const mz = nz_ - 2 * FD_PAD + 1;
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * mz * my;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(p_, vx_, model_x, model_batched, 0, nl,
                                        nx_, 1, mz, FD_PAD, ny_ * nx_, my,
                                        FD_PAD, nx_, rdx_ ADJ_GATHER_FD_ARG);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(p_, vy_, model_y, model_batched, 0, nl,
                                        ny_, nx_, mz, FD_PAD, ny_ * nx_, mx,
                                        FD_PAD, 1, rdy_ ADJ_GATHER_FD_ARG);
  }
  if (bc[0] == 3) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(p_, vz_, model_z, model_batched, 0, nl,
                                        nz_, ny_ * nx_, my, FD_PAD, nx_, mx,
                                        FD_PAD, 1, rdz_ ADJ_GATHER_FD_ARG);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * my;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(p_, vx_, model_x, model_batched, 0, nl,
                                        nx_, 1, 1, 0, 0, my, FD_PAD, nx_,
                                        rdx_ ADJ_GATHER_FD_ARG);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(p_, vy_, model_y, model_batched, 0, nl,
                                        ny_, nx_, 1, 0, 0, mx, FD_PAD, 1,
                                        rdy_ ADJ_GATHER_FD_ARG);
  }
#else
  if (bc[4] == 3) {
    int64_t const nl = n_shots_;
    adj_gather_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                              stream>>>(p_, vx_, model_x, model_batched, 0, nl,
                                        nx_, 1, 1, 0, 0, 1, 0, 0,
                                        rdx_ ADJ_GATHER_FD_ARG);
  }
#endif
}

/* Free/rigid-face exact-adjoint guard maintenance (backward/backward_sc
 * only), for the free/rigid axes other than the (free-low, rigid-high)
 * re-fill pairing: same launch structure as the periodic fold/gather
 * kernels above, calling the _fr line functions (mirror/sign-flip
 * transposes, per-face gated). */

static void launch_adj_fold_p_bc_frx(DW_DTYPE *__restrict const p,
                                     int64_t const bcx[6],
                                     int64_t const total_numel,
                                     int64_t const nz_, int64_t const ny_,
                                     int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = total_numel / nx_;
    adj_fold_fr_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nx_, 1, bcx[4], bcx[5], 0);
  }
#if DW_NDIM >= 2
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = total_numel / ny_;
    adj_fold_fr_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, ny_, nx_, bcx[2], bcx[3], 0);
  }
#endif
#if DW_NDIM >= 3
  if (bcx[0] != 0 || bcx[1] != 0) {
    int64_t const nl = total_numel / nz_;
    adj_fold_fr_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        p, nl, nz_, ny_ * nx_, bcx[0], bcx[1], 0);
  }
#endif
}

static void launch_adj_fold_v_bc_frx(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, int64_t const bcx[6],
    int64_t const total_numel, int64_t const nz_, int64_t const ny_,
    int64_t const nx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = total_numel / nx_;
    adj_fold_fr_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vx, nl, nx_, 1, bcx[4], bcx[5], 1);
  }
#if DW_NDIM >= 2
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = total_numel / ny_;
    adj_fold_fr_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vy, nl, ny_, nx_, bcx[2], bcx[3], 1);
  }
#endif
#if DW_NDIM >= 3
  if (bcx[0] != 0 || bcx[1] != 0) {
    int64_t const nl = total_numel / nz_;
    adj_fold_fr_bc_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0, stream>>>(
        vz, nl, nz_, ny_ * nx_, bcx[0], bcx[1], 1);
  }
#endif
}

__global__ void adj_gather_fr_guard_kernel(
    DW_DTYPE *__restrict const fld, DW_DTYPE const *__restrict const src_fld,
    DW_DTYPE const *__restrict const model, bool const model_batched,
    int const kind, int64_t const n_lines, int64_t const nax,
    int64_t const stride, int64_t const n_a, int64_t const start_a,
    int64_t const stride_a, int64_t const n_b, int64_t const start_b,
    int64_t const stride_b, DW_DTYPE const rd, int64_t const bc_lo,
    int64_t const bc_hi) {
  int64_t const L = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (L >= n_lines) return;
  DW_DTYPE const *const sw = stag_w_dev;
  int64_t const per_shot = n_a * n_b;
  int64_t const shot_idx = L / per_shot;
  int64_t const r = L % per_shot;
  int64_t const a = start_a + r / n_b;
  int64_t const b = start_b + r % n_b;
  int64_t const jbase = a * stride_a + b * stride_b;
  int64_t const base = shot_idx * shot_numel + jbase;
  DW_DTYPE const *const model_shot =
      model_batched ? model + shot_idx * shot_numel : model;
  if (kind == 0)
    FUNC(adj_gather_p_guard_line_fr)(fld, src_fld, model_shot, base, jbase, nax,
                                     stride, dt, rd, sw, bc_lo, bc_hi);
  else
    FUNC(adj_gather_v_guard_line_fr)(fld, src_fld, model_shot, base, jbase, nax,
                                     stride, dt, rd, sw, bc_lo, bc_hi);
}

static void launch_adj_gather_v_guards_frx(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy_,
#endif
    DW_DTYPE *__restrict const vx_, DW_DTYPE const *__restrict const p_,
    DW_DTYPE const *__restrict const model, bool const model_batched,
    int64_t const bcx[6], int64_t const n_shots_, int64_t const nz_,
    int64_t const ny_, int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  int64_t const mz = nz_ - 2 * FD_PAD + 1;
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * mz * my;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        vx_, p_, model, model_batched, 1, nl, nx_, 1, mz, FD_PAD, ny_ * nx_, my,
        FD_PAD, nx_, rdx_, bcx[4], bcx[5] ADJ_GATHER_FD_ARG);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        vy_, p_, model, model_batched, 1, nl, ny_, nx_, mz, FD_PAD, ny_ * nx_,
        mx, FD_PAD, 1, rdy_, bcx[2], bcx[3] ADJ_GATHER_FD_ARG);
  }
  if (bcx[0] != 0 || bcx[1] != 0) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        vz_, p_, model, model_batched, 1, nl, nz_, ny_ * nx_, my, FD_PAD, nx_,
        mx, FD_PAD, 1, rdz_, bcx[0], bcx[1] ADJ_GATHER_FD_ARG);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * my;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        vx_, p_, model, model_batched, 1, nl, nx_, 1, 1, 0, 0, my, FD_PAD, nx_,
        rdx_, bcx[4], bcx[5] ADJ_GATHER_FD_ARG);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        vy_, p_, model, model_batched, 1, nl, ny_, nx_, 1, 0, 0, mx, FD_PAD, 1,
        rdy_, bcx[2], bcx[3] ADJ_GATHER_FD_ARG);
  }
#else
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(vx_, p_, model, model_batched, 1, nl,
                                           nx_, 1, 1, 0, 0, 1, 0, 0, rdx_,
                                           bcx[4], bcx[5] ADJ_GATHER_FD_ARG);
  }
#endif
}

static void launch_adj_gather_p_guards_frx(
    DW_DTYPE *__restrict const p_,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz_,
    DW_DTYPE const *__restrict const model_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy_,
    DW_DTYPE const *__restrict const model_y,
#endif
    DW_DTYPE const *__restrict const vx_,
    DW_DTYPE const *__restrict const model_x, bool const model_batched,
    int64_t const bcx[6], int64_t const n_shots_, int64_t const nz_,
    int64_t const ny_, int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, cudaStream_t stream) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  int64_t const mz = nz_ - 2 * FD_PAD + 1;
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * mz * my;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        p_, vx_, model_x, model_batched, 0, nl, nx_, 1, mz, FD_PAD, ny_ * nx_,
        my, FD_PAD, nx_, rdx_, bcx[4], bcx[5] ADJ_GATHER_FD_ARG);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        p_, vy_, model_y, model_batched, 0, nl, ny_, nx_, mz, FD_PAD, ny_ * nx_,
        mx, FD_PAD, 1, rdy_, bcx[2], bcx[3] ADJ_GATHER_FD_ARG);
  }
  if (bcx[0] != 0 || bcx[1] != 0) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        p_, vz_, model_z, model_batched, 0, nl, nz_, ny_ * nx_, my, FD_PAD, nx_,
        mx, FD_PAD, 1, rdz_, bcx[0], bcx[1] ADJ_GATHER_FD_ARG);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * my;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        p_, vx_, model_x, model_batched, 0, nl, nx_, 1, 1, 0, 0, my, FD_PAD,
        nx_, rdx_, bcx[4], bcx[5] ADJ_GATHER_FD_ARG);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(
        p_, vy_, model_y, model_batched, 0, nl, ny_, nx_, 1, 0, 0, mx, FD_PAD,
        1, rdy_, bcx[2], bcx[3] ADJ_GATHER_FD_ARG);
  }
#else
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_;
    adj_gather_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                 stream>>>(p_, vx_, model_x, model_batched, 0,
                                           nl, nx_, 1, 1, 0, 0, 1, 0, 0, rdx_,
                                           bcx[4], bcx[5] ADJ_GATHER_FD_ARG);
  }
#endif
}

/* Deferred (perturbation-)buoyancy-gradient accumulation, used when any
 * axis is periodic (see acoustic.cu's backward_grad_b_deferred_kernel for
 * the rationale): the fused accumulations in backward_kernel_v /
 * backward_sc_kernel_v are suppressed for every axis and re-done here from
 * the post-fold adjoint velocities (identical values for non-periodic
 * axes). Same thread geometry as backward_kernel_v. For backward_sc pass
 * null background pointers and do_b = false. */
__launch_bounds__(128) __global__ void backward_grad_b_deferred_kernel(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const vz,
    DW_DTYPE const *__restrict__ const vzsc,
    DW_DTYPE *__restrict__ const grad_bz_shot,
    DW_DTYPE *__restrict__ const grad_dbz_shot,
    DW_DTYPE const *__restrict__ const bz_grad_store_1,
    DW_DTYPE const *__restrict__ const bz_grad_sc_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const vy,
    DW_DTYPE const *__restrict__ const vysc,
    DW_DTYPE *__restrict__ const grad_by_shot,
    DW_DTYPE *__restrict__ const grad_dby_shot,
    DW_DTYPE const *__restrict__ const by_grad_store_1,
    DW_DTYPE const *__restrict__ const by_grad_sc_store_1,
#endif
    DW_DTYPE const *__restrict__ const vx,
    DW_DTYPE const *__restrict__ const vxsc,
    DW_DTYPE *__restrict__ const grad_bx_shot,
    DW_DTYPE *__restrict__ const grad_dbx_shot,
    DW_DTYPE const *__restrict__ const bx_grad_store_1,
    DW_DTYPE const *__restrict__ const bx_grad_sc_store_1, bool const do_b,
    bool const do_db) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;
      (void)j;
#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        if (do_db)
          grad_dbz_shot[i] -=
              dt * vzsc[i] * bz_grad_store_1[i] * (DW_DTYPE)step_ratio;
        if (do_b)
          grad_bz_shot[i] -=
              dt *
              (vz[i] * bz_grad_store_1[i] + vzsc[i] * bz_grad_sc_store_1[i]) *
              (DW_DTYPE)step_ratio;
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        if (do_db)
          grad_dby_shot[i] -=
              dt * vysc[i] * by_grad_store_1[i] * (DW_DTYPE)step_ratio;
        if (do_b)
          grad_by_shot[i] -=
              dt *
              (vy[i] * by_grad_store_1[i] + vysc[i] * by_grad_sc_store_1[i]) *
              (DW_DTYPE)step_ratio;
      }
#endif
      if (x < nx - FD_PAD) {
        if (do_db)
          grad_dbx_shot[i] -=
              dt * vxsc[i] * bx_grad_store_1[i] * (DW_DTYPE)step_ratio;
        if (do_b)
          grad_bx_shot[i] -=
              dt *
              (vx[i] * bx_grad_store_1[i] + vxsc[i] * bx_grad_sc_store_1[i]) *
              (DW_DTYPE)step_ratio;
      }
#if DW_NDIM == 3
    }
#endif
  }
}

#if DW_NDIM == 3
#define LAUNCH_FILL_P_BC_FR(p)                                            \
  launch_acoustic_born_fill_p_bc(p, bc_fr, n_shots_h *shot_numel_h, nz_h, \
                                 ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_BG_FR()                                             \
  launch_acoustic_born_fill_v_bc(vz, vy, vx, bc_fr, n_shots_h *shot_numel_h, \
                                 nz_h, ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_SC_FR()                                            \
  launch_acoustic_born_fill_v_bc(vzsc, vysc, vxsc, bc_fr,                   \
                                 n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                                 stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC(p)                                          \
  launch_adj_fold_p_bc(p, bc, n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_BG()                                           \
  launch_adj_fold_v_bc(vz, vy, vx, bc, n_shots_h *shot_numel_h, nz_h, ny_h, \
                       nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_SC()                                           \
  launch_adj_fold_v_bc(vzsc, vysc, vxsc, bc, n_shots_h *shot_numel_h, nz_h, \
                       ny_h, nx_h, stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_SET(vset_z, vset_y, vset_x, srcp, model,    \
                                       batched)                                \
  launch_adj_gather_v_guards(vset_z, vset_y, vset_x, srcp, model, batched, bc, \
                             n_shots_h, nz_h, ny_h, nx_h, rdz_h, rdy_h, rdx_h, \
                             stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_SET(target, vset_z, mz_, vset_y, my_,      \
                                       vset_x, mx_, batched)                  \
  launch_adj_gather_p_guards(target, vset_z, mz_, vset_y, my_, vset_x, mx_,   \
                             batched, bc, n_shots_h, nz_h, ny_h, nx_h, rdz_h, \
                             rdy_h, rdx_h, stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_FOLD_P_BC_FRX(p)                                        \
  launch_adj_fold_p_bc_frx(p, bc_frx, n_shots_h *shot_numel_h, nz_h, ny_h, \
                           nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_BG_FRX()                                         \
  launch_adj_fold_v_bc_frx(vz, vy, vx, bc_frx, n_shots_h *shot_numel_h, nz_h, \
                           ny_h, nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_SC_FRX()                                         \
  launch_adj_fold_v_bc_frx(vzsc, vysc, vxsc, bc_frx, n_shots_h *shot_numel_h, \
                           nz_h, ny_h, nx_h, stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vset_z, vset_y, vset_x, srcp,     \
                                           model, batched)                   \
  launch_adj_gather_v_guards_frx(                                            \
      vset_z, vset_y, vset_x, srcp, model, batched, bc_frx, n_shots_h, nz_h, \
      ny_h, nx_h, rdz_h, rdy_h, rdx_h, stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(target, vset_z, mz_, vset_y, my_,  \
                                           vset_x, mx_, batched)              \
  launch_adj_gather_p_guards_frx(target, vset_z, mz_, vset_y, my_, vset_x,    \
                                 mx_, batched, bc_frx, n_shots_h, nz_h, ny_h, \
                                 nx_h, rdz_h, rdy_h, rdx_h,                   \
                                 stream_compute ADJ_GATHER_FD_ARG)
#elif DW_NDIM == 2
#define LAUNCH_FILL_P_BC_FR(p)                                               \
  launch_acoustic_born_fill_p_bc(p, bc_fr, n_shots_h *shot_numel_h, 1, ny_h, \
                                 nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_BG_FR()                                            \
  launch_acoustic_born_fill_v_bc(vy, vx, bc_fr, n_shots_h *shot_numel_h, 1, \
                                 ny_h, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_SC_FR()                                             \
  launch_acoustic_born_fill_v_bc(vysc, vxsc, bc_fr, n_shots_h *shot_numel_h, \
                                 1, ny_h, nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC(p)                                       \
  launch_adj_fold_p_bc(p, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_BG()                                          \
  launch_adj_fold_v_bc(vy, vx, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_SC()                                              \
  launch_adj_fold_v_bc(vysc, vxsc, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_SET(vset_y, vset_x, srcp, model, batched) \
  launch_adj_gather_v_guards(vset_y, vset_x, srcp, model, batched, bc,       \
                             n_shots_h, 1, ny_h, nx_h, rdy_h, rdx_h,         \
                             stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_SET(target, vset_y, my_, vset_x, mx_,    \
                                       batched)                             \
  launch_adj_gather_p_guards(target, vset_y, my_, vset_x, mx_, batched, bc, \
                             n_shots_h, 1, ny_h, nx_h, rdy_h, rdx_h,        \
                             stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_FOLD_P_BC_FRX(p)                                           \
  launch_adj_fold_p_bc_frx(p, bc_frx, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_BG_FRX()                                        \
  launch_adj_fold_v_bc_frx(vy, vx, bc_frx, n_shots_h *shot_numel_h, 1, ny_h, \
                           nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_SC_FRX()                                      \
  launch_adj_fold_v_bc_frx(vysc, vxsc, bc_frx, n_shots_h *shot_numel_h, 1, \
                           ny_h, nx_h, stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vset_y, vset_x, srcp, model,        \
                                           batched)                            \
  launch_adj_gather_v_guards_frx(vset_y, vset_x, srcp, model, batched, bc_frx, \
                                 n_shots_h, 1, ny_h, nx_h, rdy_h, rdx_h,       \
                                 stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(target, vset_y, my_, vset_x, mx_, \
                                           batched)                          \
  launch_adj_gather_p_guards_frx(target, vset_y, my_, vset_x, mx_, batched,  \
                                 bc_frx, n_shots_h, 1, ny_h, nx_h, rdy_h,    \
                                 rdx_h, stream_compute ADJ_GATHER_FD_ARG)
#else
#define LAUNCH_FILL_P_BC_FR(p)                                            \
  launch_acoustic_born_fill_p_bc(p, bc_fr, n_shots_h *shot_numel_h, 1, 1, \
                                 nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_BG_FR()                                           \
  launch_acoustic_born_fill_v_bc(vx, bc_fr, n_shots_h *shot_numel_h, 1, 1, \
                                 nx_h, stream_compute)
#define LAUNCH_FILL_V_BC_SC_FR()                                             \
  launch_acoustic_born_fill_v_bc(vxsc, bc_fr, n_shots_h *shot_numel_h, 1, 1, \
                                 nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC(p)                                    \
  launch_adj_fold_p_bc(p, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_BG()                                   \
  launch_adj_fold_v_bc(vx, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_SC()                                     \
  launch_adj_fold_v_bc(vxsc, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_SET(vset_x, srcp, model, batched)         \
  launch_adj_gather_v_guards(vset_x, srcp, model, batched, bc, n_shots_h, 1, \
                             1, nx_h, rdx_h, stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_SET(target, vset_x, mx_, batched)         \
  launch_adj_gather_p_guards(target, vset_x, mx_, batched, bc, n_shots_h, 1, \
                             1, nx_h, rdx_h, stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_FOLD_P_BC_FRX(p)                                        \
  launch_adj_fold_p_bc_frx(p, bc_frx, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_BG_FRX()                                       \
  launch_adj_fold_v_bc_frx(vx, bc_frx, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_SC_FRX()                                         \
  launch_adj_fold_v_bc_frx(vxsc, bc_frx, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vset_x, srcp, model, batched) \
  launch_adj_gather_v_guards_frx(vset_x, srcp, model, batched, bc_frx,   \
                                 n_shots_h, 1, 1, nx_h, rdx_h,           \
                                 stream_compute ADJ_GATHER_FD_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(target, vset_x, mx_, batched) \
  launch_adj_gather_p_guards_frx(target, vset_x, mx_, batched, bc_frx,   \
                                 n_shots_h, 1, 1, nx_h, rdx_h,           \
                                 stream_compute ADJ_GATHER_FD_ARG)
#endif

__launch_bounds__(32) __global__
    void add_sources(DW_DTYPE *__restrict__ const wf,
                     DW_DTYPE const *__restrict__ const f,
                     int64_t const *__restrict__ const sources_i,
                     int64_t const n_sources_per_shot) {
  int64_t const source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    if (0 <= sources_i[k]) wf[shot_idx * shot_numel + sources_i[k]] += f[k];
  }
}

__launch_bounds__(32) __global__
    void record_receivers(DW_DTYPE *__restrict__ const r,
                          DW_DTYPE const *__restrict__ const wf,
                          int64_t const *__restrict__ receivers_i,
                          int64_t const n_receivers_per_shot) {
  int64_t const receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    if (0 <= receivers_i[k]) r[k] = wf[shot_idx * shot_numel + receivers_i[k]];
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

// Forward velocity update: background then scattered (coupled). Stores the
// background term term_bg_d (b*_grad_store) when store_bg, and the scattered
// term term_sc_d (b*_grad_sc_store) when store_sc.
__launch_bounds__(128) __global__
    void forward_kernel_v(DW_DTYPE const *__restrict__ const p,
                          DW_DTYPE const *__restrict__ const psc,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const vy,
#endif
                          DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const vysc,
#endif
                          DW_DTYPE *__restrict__ const vxsc,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const psi_y,
#endif
                          DW_DTYPE *__restrict__ const psi_x,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const psi_z_sc,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const psi_y_sc,
#endif
                          DW_DTYPE *__restrict__ const psi_x_sc,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                          DW_DTYPE const *__restrict__ const buoyancy_x,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const dbuoyancy_y,
#endif
                          DW_DTYPE const *__restrict__ const dbuoyancy_x,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const bz_grad_store_1,
                          DW_DTYPE *__restrict__ const bz_grad_sc_store_1,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const by_grad_store_1,
                          DW_DTYPE *__restrict__ const by_grad_sc_store_1,
#endif
                          DW_DTYPE *__restrict__ const bx_grad_store_1,
                          DW_DTYPE *__restrict__ const bx_grad_sc_store_1,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const azh,
                          DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const ayh,
                          DW_DTYPE const *__restrict__ const byh,
#endif
                          DW_DTYPE const *__restrict__ const axh,
                          DW_DTYPE const *__restrict__ const bxh,
                          bool const store_bg, bool const store_sc) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

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

      DW_DTYPE const *__restrict__ const buoyancy_x_shot =
          b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      DW_DTYPE const *__restrict__ const dbuoyancy_x_shot =
          b_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        DW_DTYPE const *__restrict__ const dbuoyancy_z_shot =
            b_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        FUNC(forward_vz_cell)(p, psc, vz, vzsc, psi_z, psi_z_sc,
                              bz_grad_store_1, bz_grad_sc_store_1,
                              buoyancy_z_shot[j], dbuoyancy_z_shot[j], azh, bzh,
                              dt, store_bg, store_sc, i, ny, nx, z, pml_z, rdz);
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        DW_DTYPE const *__restrict__ const dbuoyancy_y_shot =
            b_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        FUNC(forward_vy_cell)(p, psc, vy, vysc, psi_y, psi_y_sc,
                              by_grad_store_1, by_grad_sc_store_1,
                              buoyancy_y_shot[j], dbuoyancy_y_shot[j], ayh, byh,
                              dt, store_bg, store_sc, i,
#if DW_NDIM >= 3
                              ny,
#endif
                              nx, y, pml_y, rdy);
      }
#endif

      if (x < nx - FD_PAD) {
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        FUNC(forward_vx_cell)(p, psc, vx, vxsc, psi_x, psi_x_sc,
                              bx_grad_store_1, bx_grad_sc_store_1,
                              buoyancy_x_shot[j], dbuoyancy_x_shot[j], axh, bxh,
                              dt, store_bg, store_sc, i,
#if DW_NDIM >= 3
                              ny,
#endif
#if DW_NDIM >= 2
                              nx,
#endif
                              x, pml_x, rdx);
      }

#if DW_NDIM == 3
    }
#endif
  }
}

// Forward pressure update: background then scattered (coupled). Stores div_bg
// (k_grad_store) when store_bg, div_sc (k_grad_sc_store) when store_sc.
__launch_bounds__(128) __global__
    void forward_kernel_p(DW_DTYPE *__restrict__ const p,
                          DW_DTYPE *__restrict__ const psc,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const vy,
#endif
                          DW_DTYPE const *__restrict__ const vx,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const vysc,
#endif
                          DW_DTYPE const *__restrict__ const vxsc,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const phi_y,
#endif
                          DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const phi_y_sc,
#endif
                          DW_DTYPE *__restrict__ const phi_x_sc,
                          DW_DTYPE const *__restrict__ const k,
                          DW_DTYPE const *__restrict__ const dk,
                          DW_DTYPE *__restrict__ const k_grad_store_1,
                          DW_DTYPE *__restrict__ const k_grad_sc_store_1,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const az,
                          DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const ay,
                          DW_DTYPE const *__restrict__ const by,
#endif
                          DW_DTYPE const *__restrict__ const ax,
                          DW_DTYPE const *__restrict__ const bx,
                          bool const store_bg, bool const store_sc) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;
      DW_DTYPE const *__restrict__ const dk_shot =
          k_batched ? dk + shot_idx * shot_numel : dk;

#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;

      FUNC(forward_p_cell)(
#if DW_NDIM >= 3
          vz, vzsc,
#endif
#if DW_NDIM >= 2
          vy, vysc,
#endif
          vx, vxsc, p, psc,
#if DW_NDIM >= 3
          phi_z, phi_z_sc,
#endif
#if DW_NDIM >= 2
          phi_y, phi_y_sc,
#endif
          phi_x, phi_x_sc, k_grad_store_1, k_grad_sc_store_1, k_shot[j],
          dk_shot[j], dt, store_bg, store_sc, i,
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

#if DW_NDIM == 3
    }
#endif
  }
}

// Scattered-only adjoint velocity kernel (ordinary acoustic adjoint on the
// scattered adjoint field). Accumulates grad_dB_d.
__launch_bounds__(128) __global__ void backward_sc_kernel_v(
    DW_DTYPE const *__restrict__ const psc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const vysc,
#endif
    DW_DTYPE *__restrict__ const vxsc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const phi_y_sc,
#endif
    DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const psi_y_sc,
#endif
    DW_DTYPE *__restrict__ const psi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const psi_zn_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const psi_yn_sc,
#endif
    DW_DTYPE *__restrict__ const psi_xn_sc,
    DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
    DW_DTYPE const *__restrict__ const buoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const grad_dbz_shot,
    DW_DTYPE const *__restrict__ const bz_grad_store_1,
    DW_DTYPE const *__restrict__ const azh,
    DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const grad_dby_shot,
    DW_DTYPE const *__restrict__ const by_grad_store_1,
    DW_DTYPE const *__restrict__ const ayh,
    DW_DTYPE const *__restrict__ const by,
#endif
    DW_DTYPE *__restrict__ const grad_dbx_shot,
    DW_DTYPE const *__restrict__ const bx_grad_store_1,
    DW_DTYPE const *__restrict__ const axh,
    DW_DTYPE const *__restrict__ const bx, bool const db_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;

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

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        FUNC(backward_sc_vz_cell)(
            psc, k_shot, vzsc, phi_z_sc, psi_z_sc, psi_zn_sc,
            buoyancy_z_shot[j], grad_dbz_shot, bz_grad_store_1, azh, bz, dt,
            step_ratio, db_requires_grad, i, j, ny, nx, z, pml_z, rdz);
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        FUNC(backward_sc_vy_cell)(psc, k_shot, vysc, phi_y_sc, psi_y_sc,
                                  psi_yn_sc, buoyancy_y_shot[j], grad_dby_shot,
                                  by_grad_store_1, ayh, by, dt, step_ratio,
                                  db_requires_grad, i, j,
#if DW_NDIM >= 3
                                  ny,
#endif
                                  nx, y, pml_y, rdy);
      }
#endif
      if (x < nx - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_x_shot =
            b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        FUNC(backward_sc_vx_cell)(psc, k_shot, vxsc, phi_x_sc, psi_x_sc,
                                  psi_xn_sc, buoyancy_x_shot[j], grad_dbx_shot,
                                  bx_grad_store_1, axh, bx, dt, step_ratio,
                                  db_requires_grad, i, j,
#if DW_NDIM >= 3
                                  ny,
#endif
#if DW_NDIM >= 2
                                  nx,
#endif
                                  x, pml_x, rdx);
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// Scattered-only adjoint pressure kernel. Accumulates grad_dK.
__launch_bounds__(128) __global__
    void backward_sc_kernel_p(DW_DTYPE *__restrict__ const psc,
#if DW_NDIM >= 3
                              DW_DTYPE const *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
                              DW_DTYPE const *__restrict__ const vysc,
#endif
                              DW_DTYPE const *__restrict__ const vxsc,
#if DW_NDIM >= 3
                              DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
                              DW_DTYPE *__restrict__ const phi_y_sc,
#endif
                              DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
                              DW_DTYPE *__restrict__ const psi_z_sc,
#endif
#if DW_NDIM >= 2
                              DW_DTYPE *__restrict__ const psi_y_sc,
#endif
                              DW_DTYPE *__restrict__ const psi_x_sc,
                              DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
                              DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                              DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                              DW_DTYPE const *__restrict__ const buoyancy_x,
                              DW_DTYPE *__restrict__ const grad_dk_shot,
                              DW_DTYPE const *__restrict__ const k_grad_store_1,
#if DW_NDIM >= 3
                              DW_DTYPE const *__restrict__ const az,
                              DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
                              DW_DTYPE const *__restrict__ const ay,
                              DW_DTYPE const *__restrict__ const byh,
#endif
                              DW_DTYPE const *__restrict__ const ax,
                              DW_DTYPE const *__restrict__ const bxh,
                              bool const dk_requires_grad) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;

#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      DW_DTYPE const *__restrict__ const buoyancy_z_shot =
          b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
      DW_DTYPE const *__restrict__ const buoyancy_y_shot =
          b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      DW_DTYPE const *__restrict__ const buoyancy_x_shot =
          b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;

      FUNC(backward_sc_p_cell)(psc,
#if DW_NDIM >= 3
                               vzsc, phi_z_sc, buoyancy_z_shot, psi_z_sc,
#endif
#if DW_NDIM >= 2
                               vysc, phi_y_sc, buoyancy_y_shot, psi_y_sc,
#endif
                               vxsc, phi_x_sc, buoyancy_x_shot, psi_x_sc,
                               k_shot[j], grad_dk_shot, k_grad_store_1, dt,
                               step_ratio, dk_requires_grad, i, j,
#if DW_NDIM >= 3
                               ny,
#endif
#if DW_NDIM >= 2
                               nx,
#endif
#if DW_NDIM >= 3
                               z, pml_z, az, bzh, rdz,
#endif
#if DW_NDIM >= 2
                               y, pml_y, ay, byh, rdy,
#endif
                               x, pml_x, ax, bxh, rdx);

#if DW_NDIM == 3
    }
#endif
  }
}

// Full backward adjoint velocity kernel: background (coupled) + scattered
// (ordinary). Accumulates grad_B_d (two-term) and grad_dB_d.
__launch_bounds__(128) __global__ void backward_kernel_v(
    DW_DTYPE const *__restrict__ const p,
    DW_DTYPE const *__restrict__ const psc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const vy,
#endif
    DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const vysc,
#endif
    DW_DTYPE *__restrict__ const vxsc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const phi_z, DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const phi_y, DW_DTYPE *__restrict__ const phi_y_sc,
#endif
    DW_DTYPE *__restrict__ const phi_x, DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const psi_z, DW_DTYPE *__restrict__ const psi_z_sc,
    DW_DTYPE *__restrict__ const psi_zn, DW_DTYPE *__restrict__ const psi_zn_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const psi_y, DW_DTYPE *__restrict__ const psi_y_sc,
    DW_DTYPE *__restrict__ const psi_yn, DW_DTYPE *__restrict__ const psi_yn_sc,
#endif
    DW_DTYPE *__restrict__ const psi_x, DW_DTYPE *__restrict__ const psi_x_sc,
    DW_DTYPE *__restrict__ const psi_xn, DW_DTYPE *__restrict__ const psi_xn_sc,
    DW_DTYPE const *__restrict__ const k, DW_DTYPE const *__restrict__ const dk,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const buoyancy_z,
    DW_DTYPE const *__restrict__ const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const buoyancy_y,
    DW_DTYPE const *__restrict__ const dbuoyancy_y,
#endif
    DW_DTYPE const *__restrict__ const buoyancy_x,
    DW_DTYPE const *__restrict__ const dbuoyancy_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const grad_bz_shot,
    DW_DTYPE *__restrict__ const grad_dbz_shot,
    DW_DTYPE const *__restrict__ const bz_grad_store_1,
    DW_DTYPE const *__restrict__ const bz_grad_sc_store_1,
    DW_DTYPE const *__restrict__ const azh,
    DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const grad_by_shot,
    DW_DTYPE *__restrict__ const grad_dby_shot,
    DW_DTYPE const *__restrict__ const by_grad_store_1,
    DW_DTYPE const *__restrict__ const by_grad_sc_store_1,
    DW_DTYPE const *__restrict__ const ayh,
    DW_DTYPE const *__restrict__ const by,
#endif
    DW_DTYPE *__restrict__ const grad_bx_shot,
    DW_DTYPE *__restrict__ const grad_dbx_shot,
    DW_DTYPE const *__restrict__ const bx_grad_store_1,
    DW_DTYPE const *__restrict__ const bx_grad_sc_store_1,
    DW_DTYPE const *__restrict__ const axh,
    DW_DTYPE const *__restrict__ const bx, bool const b_requires_grad,
    bool const db_requires_grad, bool const dk_batched, bool const db_batched) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;
      DW_DTYPE const *__restrict__ const dk_shot =
          dk_batched ? dk + shot_idx * shot_numel : dk;

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

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        DW_DTYPE const *__restrict__ const dbuoyancy_z_shot =
            db_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        FUNC(backward_vz_cell)(
            p, psc, k_shot, dk_shot, vz, vzsc, phi_z, phi_z_sc, psi_z, psi_zn,
            psi_z_sc, psi_zn_sc, buoyancy_z_shot[j], dbuoyancy_z_shot[j],
            grad_bz_shot, grad_dbz_shot, bz_grad_store_1, bz_grad_sc_store_1,
            azh, bz, dt, step_ratio, b_requires_grad, db_requires_grad, i, j,
            ny, nx, z, pml_z, rdz);
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        DW_DTYPE const *__restrict__ const dbuoyancy_y_shot =
            db_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        FUNC(backward_vy_cell)(
            p, psc, k_shot, dk_shot, vy, vysc, phi_y, phi_y_sc, psi_y, psi_yn,
            psi_y_sc, psi_yn_sc, buoyancy_y_shot[j], dbuoyancy_y_shot[j],
            grad_by_shot, grad_dby_shot, by_grad_store_1, by_grad_sc_store_1,
            ayh, by, dt, step_ratio, b_requires_grad, db_requires_grad, i, j,
#if DW_NDIM >= 3
            ny,
#endif
            nx, y, pml_y, rdy);
      }
#endif
      if (x < nx - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_x_shot =
            b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
        DW_DTYPE const *__restrict__ const dbuoyancy_x_shot =
            db_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        FUNC(backward_vx_cell)(
            p, psc, k_shot, dk_shot, vx, vxsc, phi_x, phi_x_sc, psi_x, psi_xn,
            psi_x_sc, psi_xn_sc, buoyancy_x_shot[j], dbuoyancy_x_shot[j],
            grad_bx_shot, grad_dbx_shot, bx_grad_store_1, bx_grad_sc_store_1,
            axh, bx, dt, step_ratio, b_requires_grad, db_requires_grad, i, j,
#if DW_NDIM >= 3
            ny,
#endif
#if DW_NDIM >= 2
            nx,
#endif
            x, pml_x, rdx);
      }
#if DW_NDIM == 3
    }
#endif
  }
}

// Full backward adjoint pressure kernel: background (coupled) + scattered
// (ordinary). Accumulates grad_K (two-term) and grad_dK.
__launch_bounds__(128) __global__ void backward_kernel_p(
    DW_DTYPE *__restrict__ const p, DW_DTYPE *__restrict__ const psc,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const vy,
#endif
    DW_DTYPE const *__restrict__ const vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const vysc,
#endif
    DW_DTYPE const *__restrict__ const vxsc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const phi_z, DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const phi_y, DW_DTYPE *__restrict__ const phi_y_sc,
#endif
    DW_DTYPE *__restrict__ const phi_x, DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const psi_z, DW_DTYPE *__restrict__ const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const psi_y, DW_DTYPE *__restrict__ const psi_y_sc,
#endif
    DW_DTYPE *__restrict__ const psi_x, DW_DTYPE *__restrict__ const psi_x_sc,
    DW_DTYPE const *__restrict__ const k, DW_DTYPE const *__restrict__ const dk,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const buoyancy_z,
    DW_DTYPE const *__restrict__ const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const buoyancy_y,
    DW_DTYPE const *__restrict__ const dbuoyancy_y,
#endif
    DW_DTYPE const *__restrict__ const buoyancy_x,
    DW_DTYPE const *__restrict__ const dbuoyancy_x,
    DW_DTYPE *__restrict__ const grad_k_shot,
    DW_DTYPE *__restrict__ const grad_dk_shot,
    DW_DTYPE const *__restrict__ const k_grad_store_1,
    DW_DTYPE const *__restrict__ const k_grad_sc_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const az,
    DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const ay,
    DW_DTYPE const *__restrict__ const byh,
#endif
    DW_DTYPE const *__restrict__ const ax,
    DW_DTYPE const *__restrict__ const bxh, bool const k_requires_grad,
    bool const dk_requires_grad, bool const dk_batched, bool const db_batched) {
#if DW_NDIM == 3
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  if (z < nz - FD_PAD + 1 && y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1) {
    int64_t const j = z * ny * nx + y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
#elif DW_NDIM == 2
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t const shot_idx = blockIdx.z * blockDim.z + threadIdx.z;
  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = y * nx + x;
#else /* DW_NDIM == 1 */
  int64_t const x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t const shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const j = x;
#endif
      int64_t const i = shot_idx * shot_numel + j;

      DW_DTYPE const *__restrict__ const k_shot =
          k_batched ? k + shot_idx * shot_numel : k;
      DW_DTYPE const *__restrict__ const dk_shot =
          dk_batched ? dk + shot_idx * shot_numel : dk;

#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      DW_DTYPE const *__restrict__ const buoyancy_z_shot =
          b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
      DW_DTYPE const *__restrict__ const dbuoyancy_z_shot =
          db_batched ? dbuoyancy_z + shot_idx * shot_numel : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
      DW_DTYPE const *__restrict__ const buoyancy_y_shot =
          b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
      DW_DTYPE const *__restrict__ const dbuoyancy_y_shot =
          db_batched ? dbuoyancy_y + shot_idx * shot_numel : dbuoyancy_y;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;
      DW_DTYPE const *__restrict__ const buoyancy_x_shot =
          b_batched ? buoyancy_x + shot_idx * shot_numel : buoyancy_x;
      DW_DTYPE const *__restrict__ const dbuoyancy_x_shot =
          db_batched ? dbuoyancy_x + shot_idx * shot_numel : dbuoyancy_x;

      FUNC(backward_p_cell)(p, psc,
#if DW_NDIM >= 3
                            vz, vzsc, phi_z, phi_z_sc, buoyancy_z_shot,
                            dbuoyancy_z_shot, psi_z, psi_z_sc,
#endif
#if DW_NDIM >= 2
                            vy, vysc, phi_y, phi_y_sc, buoyancy_y_shot,
                            dbuoyancy_y_shot, psi_y, psi_y_sc,
#endif
                            vx, vxsc, phi_x, phi_x_sc, buoyancy_x_shot,
                            dbuoyancy_x_shot, psi_x, psi_x_sc, k_shot[j],
                            dk_shot[j], grad_k_shot, grad_dk_shot,
                            k_grad_store_1, k_grad_sc_store_1, dt, step_ratio,
                            k_requires_grad, dk_requires_grad, i, j,
#if DW_NDIM >= 3
                            ny,
#endif
#if DW_NDIM >= 2
                            nx,
#endif
#if DW_NDIM >= 3
                            z, pml_z, az, bzh, rdz,
#endif
#if DW_NDIM >= 2
                            y, pml_y, ay, byh, rdy,
#endif
                            x, pml_x, ax, bxh, rdx);

#if DW_NDIM == 3
    }
#endif
  }
}

int set_config(
#if DW_NDIM >= 3
    DW_DTYPE const rdz_h, int64_t const nz_h, int64_t const pml_z0_h,
    int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_h, int64_t const ny_h, int64_t const pml_y0_h,
    int64_t const pml_y1_h,
#endif
    DW_DTYPE const rdx_h, int64_t const nx_h, int64_t const shot_numel_h,
    int64_t const pml_x0_h, int64_t const pml_x1_h, DW_DTYPE const dt_h,
    int64_t const n_shots_h, int64_t const step_ratio_h, bool const k_batched_h,
    bool const b_batched_h) {
  gpuErrchk(cudaMemcpyToSymbol(dt, &dt_h, sizeof(DW_DTYPE)));
#if DW_NDIM >= 3
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
#endif
#if DW_NDIM >= 2
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
#endif
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(k_batched, &k_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(b_batched, &b_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(forward)(
        DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
        DW_DTYPE const *__restrict__ const buoyancy_x,
        DW_DTYPE const *__restrict__ const dk,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const dbuoyancy_y,
#endif
        DW_DTYPE const *__restrict__ const dbuoyancy_x,
        DW_DTYPE const *__restrict__ const f_p,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const f_vz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const f_vy,
#endif
        DW_DTYPE const *__restrict__ const f_vx,
        DW_DTYPE const *__restrict__ const f_p_sc,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const f_vz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const f_vy_sc,
#endif
        DW_DTYPE const *__restrict__ const f_vx_sc,
        DW_DTYPE *__restrict__ const p,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const vy,
#endif
        DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const phi_y,
#endif
        DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const psi_y,
#endif
        DW_DTYPE *__restrict__ const psi_x, DW_DTYPE *__restrict__ const psc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const vysc,
#endif
        DW_DTYPE *__restrict__ const vxsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const phi_y_sc,
#endif
        DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const psi_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const psi_y_sc,
#endif
        DW_DTYPE *__restrict__ const psi_x_sc,
        DW_DTYPE *__restrict__ const k_grad_store_1a,
        DW_DTYPE *__restrict__ const k_grad_store_1b,
        void *__restrict__ const k_grad_store_2,
        void *__restrict__ const k_grad_store_3,
        char const *__restrict__ const *__restrict__ const k_grad_filenames_ptr,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const bz_grad_store_1a,
        DW_DTYPE *__restrict__ const bz_grad_store_1b,
        void *__restrict__ const bz_grad_store_2,
        void *__restrict__ const bz_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const by_grad_store_1a,
        DW_DTYPE *__restrict__ const by_grad_store_1b,
        void *__restrict__ const by_grad_store_2,
        void *__restrict__ const by_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const by_grad_filenames_ptr,
#endif
        DW_DTYPE *__restrict__ const bx_grad_store_1a,
        DW_DTYPE *__restrict__ const bx_grad_store_1b,
        void *__restrict__ const bx_grad_store_2,
        void *__restrict__ const bx_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const bx_grad_filenames_ptr,
        DW_DTYPE *__restrict__ const k_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const k_grad_sc_store_1b,
        void *__restrict__ const k_grad_sc_store_2,
        void *__restrict__ const k_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const k_grad_sc_filenames_ptr,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const bz_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const bz_grad_sc_store_1b,
        void *__restrict__ const bz_grad_sc_store_2,
        void *__restrict__ const bz_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const bz_grad_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const by_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const by_grad_sc_store_1b,
        void *__restrict__ const by_grad_sc_store_2,
        void *__restrict__ const by_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const by_grad_sc_filenames_ptr,
#endif
        DW_DTYPE *__restrict__ const bx_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const bx_grad_sc_store_1b,
        void *__restrict__ const bx_grad_sc_store_2,
        void *__restrict__ const bx_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const bx_grad_sc_filenames_ptr,
        DW_DTYPE *__restrict__ const r_p,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const r_vz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const r_vy,
#endif
        DW_DTYPE *__restrict__ const r_vx, DW_DTYPE *__restrict__ const r_p_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const r_vz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const r_vy_sc,
#endif
        DW_DTYPE *__restrict__ const r_vx_sc,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const az,
        DW_DTYPE const *__restrict__ const bz,
        DW_DTYPE const *__restrict__ const azh,
        DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const ay,
        DW_DTYPE const *__restrict__ const by,
        DW_DTYPE const *__restrict__ const ayh,
        DW_DTYPE const *__restrict__ const byh,
#endif
        DW_DTYPE const *__restrict__ const ax,
        DW_DTYPE const *__restrict__ const bx,
        DW_DTYPE const *__restrict__ const axh,
        DW_DTYPE const *__restrict__ const bxh,
        int64_t const *__restrict__ const sources_i_p,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const sources_i_vz,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const sources_i_vy,
#endif
        int64_t const *__restrict__ const sources_i_vx,
        int64_t const *__restrict__ const sources_i_p_sc,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const sources_i_vz_sc,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const sources_i_vy_sc,
#endif
        int64_t const *__restrict__ const sources_i_vx_sc,
        int64_t const *__restrict__ const receivers_i_p,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const receivers_i_vz,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const receivers_i_vy,
#endif
        int64_t const *__restrict__ const receivers_i_vx,
        int64_t const *__restrict__ const receivers_i_p_sc,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const receivers_i_vz_sc,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const receivers_i_vy_sc,
#endif
        int64_t const *__restrict__ const receivers_i_vx_sc,
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
        int64_t const nx_h, int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_sources_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_vy_per_shot_h,
#endif
        int64_t const n_sources_vx_per_shot_h,
        int64_t const n_sources_p_sc_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_sources_vz_sc_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_vy_sc_per_shot_h,
#endif
        int64_t const n_sources_vx_sc_per_shot_h,
        int64_t const n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_vy_per_shot_h,
#endif
        int64_t const n_receivers_vx_per_shot_h,
        int64_t const n_receivers_p_sc_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_vz_sc_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_vy_sc_per_shot_h,
#endif
        int64_t const n_receivers_vx_sc_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const k_requires_grad,
        bool const b_requires_grad, bool const dk_requires_grad,
        bool const db_requires_grad, bool const k_batched_h,
        bool const b_batched_h, bool const storage_compression,
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
    if ((n_receivers_p_per_shot_h > 0) && (receivers_i_p == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_per_shot_h > 0) && (receivers_i_vz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_per_shot_h > 0) && (receivers_i_vy == NULL)) return 1;
#endif
    if ((n_receivers_vx_per_shot_h > 0) && (receivers_i_vx == NULL)) return 1;
    if ((n_receivers_p_sc_per_shot_h > 0) && (receivers_i_p_sc == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_sc_per_shot_h > 0) && (receivers_i_vz_sc == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_sc_per_shot_h > 0) && (receivers_i_vy_sc == NULL))
      return 1;
#endif
    if ((n_receivers_vx_sc_per_shot_h > 0) && (receivers_i_vx_sc == NULL))
      return 1;
    if ((n_sources_p_per_shot_h > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot_h > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot_h > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot_h > 0) && sources_i_vx == NULL) return 1;
    if ((n_sources_p_sc_per_shot_h > 0) && sources_i_p_sc == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_sc_per_shot_h > 0) && sources_i_vz_sc == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_sc_per_shot_h > 0) && sources_i_vy_sc == NULL) return 1;
#endif
    if ((n_sources_vx_sc_per_shot_h > 0) && sources_i_vx_sc == NULL) return 1;
  }

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

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif

  int64_t t;
  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, dt_h, n_shots_h,
        step_ratio_h, k_batched_h, b_batched_h);
    if (err != 0) return err;
  }

  bool const store_needed_h = k_requires_grad || b_requires_grad ||
                              dk_requires_grad || db_requires_grad;
  bool const store_sc_needed_h = k_requires_grad || b_requires_grad;

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

  OPEN_FILE_WRITE(k_grad, store_needed_h)
#if DW_NDIM >= 3
  OPEN_FILE_WRITE(bz_grad, store_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_WRITE(by_grad, store_needed_h)
#endif
  OPEN_FILE_WRITE(bx_grad, store_needed_h)
  OPEN_FILE_WRITE(k_grad_sc, store_sc_needed_h)
#if DW_NDIM >= 3
  OPEN_FILE_WRITE(bz_grad_sc, store_sc_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_WRITE(by_grad_sc, store_sc_needed_h)
#endif
  OPEN_FILE_WRITE(bx_grad_sc, store_sc_needed_h)

  for (t = start_t; t < start_t + nt; ++t) {
    int64_t const step_idx = t / step_ratio_h;
#define SETUP_STORE_SAVE(name, grad_cond)                              \
  DW_DTYPE *__restrict name##_store_1_t =                              \
      name##_store_1a != NULL                                          \
          ? (name##_store_1a +                                         \
             ((storage_mode == STORAGE_DEVICE && !storage_compression) \
                  ? (t / step_ratio_h) * n_shots_h * shot_numel_h      \
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

    SETUP_STORE_SAVE(k_grad, store_needed_h)
#if DW_NDIM >= 3
    SETUP_STORE_SAVE(bz_grad, store_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_SAVE(by_grad, store_needed_h)
#endif
    SETUP_STORE_SAVE(bx_grad, store_needed_h)
    SETUP_STORE_SAVE(k_grad_sc, store_sc_needed_h)
#if DW_NDIM >= 3
    SETUP_STORE_SAVE(bz_grad_sc, store_sc_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_SAVE(by_grad_sc, store_sc_needed_h)
#endif
    SETUP_STORE_SAVE(bx_grad_sc, store_sc_needed_h)

    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    bool const store_bg = store_needed_h && ((t % step_ratio_h) == 0);
    bool const store_sc = store_sc_needed_h && ((t % step_ratio_h) == 0);

#define RECORD_RCV(rname, wf, ridx, ncount)                       \
  if (ncount > 0) {                                               \
    dim3 const g(ceil_div(ncount, dimBlock_src_rcv.x),            \
                 ceil_div(n_shots_h, dimBlock_src_rcv.y), 1);     \
    record_receivers<<<g, dimBlock_src_rcv, 0, stream_compute>>>( \
        rname + t * n_shots_h * ncount, wf, ridx, ncount);        \
    CHECK_KERNEL_ERROR                                            \
  }

    // Refill the pressure guard ring (read by the velocity update), on both
    // fields independently; no-op for all-PML.
    if (has_bc) {
      LAUNCH_FILL_P_BC(p);
      LAUNCH_FILL_P_BC(psc);
    }

    RECORD_RCV(r_p, p, receivers_i_p, n_receivers_p_per_shot_h)
#if DW_NDIM >= 3
    RECORD_RCV(r_vz, vz, receivers_i_vz, n_receivers_vz_per_shot_h)
#endif
#if DW_NDIM >= 2
    RECORD_RCV(r_vy, vy, receivers_i_vy, n_receivers_vy_per_shot_h)
#endif
    RECORD_RCV(r_vx, vx, receivers_i_vx, n_receivers_vx_per_shot_h)
    RECORD_RCV(r_p_sc, psc, receivers_i_p_sc, n_receivers_p_sc_per_shot_h)
#if DW_NDIM >= 3
    RECORD_RCV(r_vz_sc, vzsc, receivers_i_vz_sc, n_receivers_vz_sc_per_shot_h)
#endif
#if DW_NDIM >= 2
    RECORD_RCV(r_vy_sc, vysc, receivers_i_vy_sc, n_receivers_vy_sc_per_shot_h)
#endif
    RECORD_RCV(r_vx_sc, vxsc, receivers_i_vx_sc, n_receivers_vx_sc_per_shot_h)

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, psc,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        vzsc,
#endif
#if DW_NDIM >= 2
        vysc,
#endif
        vxsc,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x,
#if DW_NDIM >= 3
        psi_z_sc,
#endif
#if DW_NDIM >= 2
        psi_y_sc,
#endif
        psi_x_sc,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        dbuoyancy_y,
#endif
        dbuoyancy_x,
#if DW_NDIM >= 3
        bz_grad_store_1_t, bz_grad_sc_store_1_t,
#endif
#if DW_NDIM >= 2
        by_grad_store_1_t, by_grad_sc_store_1_t,
#endif
        bx_grad_store_1_t, bx_grad_sc_store_1_t,
#if DW_NDIM >= 3
        azh, bzh,
#endif
#if DW_NDIM >= 2
        ayh, byh,
#endif
        axh, bxh, store_bg, store_sc);
    CHECK_KERNEL_ERROR

#define ADD_SRC(wf, f, sidx, ncount)                          \
  if (ncount > 0) {                                           \
    dim3 const g(ceil_div(ncount, dimBlock_src_rcv.x),        \
                 ceil_div(n_shots_h, dimBlock_src_rcv.y), 1); \
    add_sources<<<g, dimBlock_src_rcv, 0, stream_compute>>>(  \
        wf, f + t * n_shots_h * ncount, sidx, ncount);        \
    CHECK_KERNEL_ERROR                                        \
  }

#if DW_NDIM >= 3
    ADD_SRC(vz, f_vz, sources_i_vz, n_sources_vz_per_shot_h)
    ADD_SRC(vzsc, f_vz_sc, sources_i_vz_sc, n_sources_vz_sc_per_shot_h)
#endif
#if DW_NDIM >= 2
    ADD_SRC(vy, f_vy, sources_i_vy, n_sources_vy_per_shot_h)
    ADD_SRC(vysc, f_vy_sc, sources_i_vy_sc, n_sources_vy_sc_per_shot_h)
#endif
    ADD_SRC(vx, f_vx, sources_i_vx, n_sources_vx_per_shot_h)
    ADD_SRC(vxsc, f_vx_sc, sources_i_vx_sc, n_sources_vx_sc_per_shot_h)

    // Refill the boundary-normal velocity guard ring (read by the pressure
    // update), on both fields independently; no-op for all-PML.
    if (has_bc) {
      LAUNCH_FILL_V_BC_BG();
      LAUNCH_FILL_V_BC_SC();
    }

    forward_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, psc,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        vzsc,
#endif
#if DW_NDIM >= 2
        vysc,
#endif
        vxsc,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x,
#if DW_NDIM >= 3
        phi_z_sc,
#endif
#if DW_NDIM >= 2
        phi_y_sc,
#endif
        phi_x_sc, k, dk, k_grad_store_1_t, k_grad_sc_store_1_t,
#if DW_NDIM >= 3
        az, bz,
#endif
#if DW_NDIM >= 2
        ay, by,
#endif
        ax, bx, store_bg, store_sc);
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

    SAVE_SNAPSHOT(k_grad)
#if DW_NDIM >= 3
    SAVE_SNAPSHOT(bz_grad)
#endif
#if DW_NDIM >= 2
    SAVE_SNAPSHOT(by_grad)
#endif
    SAVE_SNAPSHOT(bx_grad)
    SAVE_SNAPSHOT(k_grad_sc)
#if DW_NDIM >= 3
    SAVE_SNAPSHOT(bz_grad_sc)
#endif
#if DW_NDIM >= 2
    SAVE_SNAPSHOT(by_grad_sc)
#endif
    SAVE_SNAPSHOT(bx_grad_sc)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    ADD_SRC(p, f_p, sources_i_p, n_sources_p_per_shot_h)
    ADD_SRC(psc, f_p_sc, sources_i_p_sc, n_sources_p_sc_per_shot_h)
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
    FUNC(backward_sc)(
        DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
        DW_DTYPE const *__restrict__ const buoyancy_x,
        DW_DTYPE const *__restrict__ const grad_r_p_sc,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const grad_r_vz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const grad_r_vy_sc,
#endif
        DW_DTYPE const *__restrict__ const grad_r_vx_sc,
        DW_DTYPE *__restrict__ const psc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const vysc,
#endif
        DW_DTYPE *__restrict__ const vxsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const phi_y_sc,
#endif
        DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ psi_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ psi_y_sc,
#endif
        DW_DTYPE *__restrict__ psi_x_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ psi_zn_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ psi_yn_sc,
#endif
        DW_DTYPE *__restrict__ psi_xn_sc,
        DW_DTYPE *__restrict__ const k_grad_store_1a,
        DW_DTYPE *__restrict__ const k_grad_store_1b,
        void *__restrict__ const k_grad_store_2,
        void *__restrict__ const k_grad_store_3,
        char const *__restrict__ const *__restrict__ const k_grad_filenames_ptr,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const bz_grad_store_1a,
        DW_DTYPE *__restrict__ const bz_grad_store_1b,
        void *__restrict__ const bz_grad_store_2,
        void *__restrict__ const bz_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const by_grad_store_1a,
        DW_DTYPE *__restrict__ const by_grad_store_1b,
        void *__restrict__ const by_grad_store_2,
        void *__restrict__ const by_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const by_grad_filenames_ptr,
#endif
        DW_DTYPE *__restrict__ const bx_grad_store_1a,
        DW_DTYPE *__restrict__ const bx_grad_store_1b,
        void *__restrict__ const bx_grad_store_2,
        void *__restrict__ const bx_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const bx_grad_filenames_ptr,
        DW_DTYPE *__restrict__ const grad_f_p_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_f_vz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_f_vy_sc,
#endif
        DW_DTYPE *__restrict__ const grad_f_vx_sc,
        DW_DTYPE *__restrict__ const grad_dk,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_dbz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_dby,
#endif
        DW_DTYPE *__restrict__ const grad_dbx,
        DW_DTYPE *__restrict__ const grad_dk_shot,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_dbz_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_dby_shot,
#endif
        DW_DTYPE *__restrict__ const grad_dbx_shot,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const az,
        DW_DTYPE const *__restrict__ const bz,
        DW_DTYPE const *__restrict__ const azh,
        DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const ay,
        DW_DTYPE const *__restrict__ const by,
        DW_DTYPE const *__restrict__ const ayh,
        DW_DTYPE const *__restrict__ const byh,
#endif
        DW_DTYPE const *__restrict__ const ax,
        DW_DTYPE const *__restrict__ const bx,
        DW_DTYPE const *__restrict__ const axh,
        DW_DTYPE const *__restrict__ const bxh,
        int64_t const *__restrict__ const sources_i_p_sc,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const sources_i_vz_sc,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const sources_i_vy_sc,
#endif
        int64_t const *__restrict__ const sources_i_vx_sc,
        int64_t const *__restrict__ const receivers_i_p_sc,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const receivers_i_vz_sc,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const receivers_i_vy_sc,
#endif
        int64_t const *__restrict__ const receivers_i_vx_sc,
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
        int64_t const nx_h, int64_t const n_sources_p_sc_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_sources_vz_sc_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_vy_sc_per_shot_h,
#endif
        int64_t const n_sources_vx_sc_per_shot_h,
        int64_t const n_receivers_p_sc_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_vz_sc_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_vy_sc_per_shot_h,
#endif
        int64_t const n_receivers_vx_sc_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const dk_requires_grad,
        bool const db_requires_grad, bool const k_batched_h,
        bool const b_batched_h, bool const dk_batched_h,
        bool const db_batched_h, bool const storage_compression,
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
    if ((dk_requires_grad) && (grad_dk == NULL)) return 1;
    if ((n_receivers_p_sc_per_shot_h > 0) && (receivers_i_p_sc == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_sc_per_shot_h > 0) && (receivers_i_vz_sc == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_sc_per_shot_h > 0) && (receivers_i_vy_sc == NULL))
      return 1;
#endif
    if ((n_receivers_vx_sc_per_shot_h > 0) && (receivers_i_vx_sc == NULL))
      return 1;
    if ((n_sources_p_sc_per_shot_h > 0) && sources_i_p_sc == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_sc_per_shot_h > 0) && sources_i_vz_sc == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_sc_per_shot_h > 0) && sources_i_vy_sc == NULL) return 1;
#endif
    if ((n_sources_vx_sc_per_shot_h > 0) && sources_i_vx_sc == NULL) return 1;
  }
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

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif

  /* Periodic faces need the adjoint fold/gather treatment (see
   * acoustic_born_kernels.h); free/rigid faces keep the forward re-fill.
   * bc_fr is bc with the periodic axes suppressed. */
#ifndef DW_AXIS_IS_REFILL
/* An axis keeps the historical adjoint re-fill only for the (free-low,
 * rigid-high) face pairing; every other free/rigid combination is
 * fold/gather-handled (see acoustic_born_kernels.h's _fr functions). */
#define DW_AXIS_IS_REFILL(lo, hi) ((lo) == 1 && (hi) == 2)
#endif
  int64_t const bc_fr[6] = {DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[0] : 0,
                            DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[1] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[2] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[3] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[4] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[5] : 0};
  int64_t const bc_frx[6] = {
      bc[0] == 3 || bc_fr[0] ? 0 : bc[0], bc[1] == 3 || bc_fr[1] ? 0 : bc[1],
      bc[2] == 3 || bc_fr[2] ? 0 : bc[2], bc[3] == 3 || bc_fr[3] ? 0 : bc[3],
      bc[4] == 3 || bc_fr[4] ? 0 : bc[4], bc[5] == 3 || bc_fr[5] ? 0 : bc[5]};
  int const has_frx = bc_frx[0] || bc_frx[1] || bc_frx[2] || bc_frx[3] ||
                      bc_frx[4] || bc_frx[5];
  int const has_fr =
      bc_fr[0] || bc_fr[1] || bc_fr[2] || bc_fr[3] || bc_fr[4] || bc_fr[5];
#if DW_NDIM == 3
  int const has_per = bc[0] == 3 || bc[2] == 3 || bc[4] == 3;
#elif DW_NDIM == 2
  int const has_per = bc[2] == 3 || bc[4] == 3;
#else
  int const has_per = bc[4] == 3;
#endif
  /* Any fold-handled axis (periodic or exact-fr) defers the
   * (perturbation-)buoyancy gradient to the post-fold deferred kernel. */
  int const has_defer = has_per || has_frx;
  (void)has_bc;

  int64_t t;
  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, dt_h, n_shots_h,
        step_ratio_h, k_batched_h, b_batched_h);
    if (err != 0) return err;
  }

  bool const store_needed_h = dk_requires_grad || db_requires_grad;

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      store_needed_h;

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

#define OPEN_FILE_READ(name, grad_cond)                           \
  ScopedFile fp_##name;                                           \
  file_group.add(fp_##name);                                      \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {              \
    if (!fp_##name.open(name##_filenames_ptr[0], "rb")) return 1; \
  }

  OPEN_FILE_READ(k_grad, store_needed_h)
#if DW_NDIM >= 3
  OPEN_FILE_READ(bz_grad, store_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(by_grad, store_needed_h)
#endif
  OPEN_FILE_READ(bx_grad, store_needed_h)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio_h;
#define SETUP_STORE_LOAD(name, grad_cond)                                 \
  DW_DTYPE *__restrict__ name##_store_1_t =                               \
      name##_store_1a != NULL                                             \
          ? (name##_store_1a +                                            \
             ((storage_mode == STORAGE_DEVICE && !storage_compression)    \
                  ? (t / step_ratio_h) * n_shots_h * shot_numel_h         \
                  : 0))                                                   \
          : NULL;                                                         \
  void *__restrict__ const name##_store_2_t =                             \
      name##_store_2 != NULL                                              \
          ? ((uint8_t *)name##_store_2 +                                  \
             ((storage_mode == STORAGE_DEVICE && storage_compression)     \
                  ? (t / step_ratio_h) * n_shots_h * shot_bytes_comp      \
                  : 0))                                                   \
          : NULL;                                                         \
  void *__restrict__ const name##_store_3_t =                             \
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

    SETUP_STORE_LOAD(k_grad, store_needed_h)
#if DW_NDIM >= 3
    SETUP_STORE_LOAD(bz_grad, store_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(by_grad, store_needed_h)
#endif
    SETUP_STORE_LOAD(bx_grad, store_needed_h)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    bool const dk_grad_t = dk_requires_grad && ((t % step_ratio_h) == 0);
    bool const db_grad_t = db_requires_grad && ((t % step_ratio_h) == 0);

    // Periodic faces: fold the scattered adjoint pressure's guard cells
    // back into the interior nodes the forward fill copied them from (the
    // exact transpose of the fill). The fold modifies interior seam nodes,
    // so it must precede the source-gradient read below.
    if (has_per) {
      LAUNCH_ADJ_FOLD_P_BC(psc);
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
      LAUNCH_ADJ_FOLD_P_BC_FRX(psc);
      CHECK_KERNEL_ERROR
    }

    // Record scattered source gradients from adjoint pressure.
    RECORD_RCV(grad_f_p_sc, psc, sources_i_p_sc, n_sources_p_sc_per_shot_h)

    // Adjoint guard fill of the scattered pressure field (read by the
    // adjoint velocity update) on the free/rigid faces: the same image the
    // forward applies, which realises the exact transpose on a reflecting
    // face. No-op for all-PML.
    if (has_fr) LAUNCH_FILL_P_BC_FR(psc);

    backward_sc_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        psc,
#if DW_NDIM >= 3
        vzsc,
#endif
#if DW_NDIM >= 2
        vysc,
#endif
        vxsc,
#if DW_NDIM >= 3
        phi_z_sc,
#endif
#if DW_NDIM >= 2
        phi_y_sc,
#endif
        phi_x_sc,
#if DW_NDIM >= 3
        psi_z_sc,
#endif
#if DW_NDIM >= 2
        psi_y_sc,
#endif
        psi_x_sc,
#if DW_NDIM >= 3
        psi_zn_sc,
#endif
#if DW_NDIM >= 2
        psi_yn_sc,
#endif
        psi_xn_sc, k,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        grad_dbz_shot, bz_grad_store_1_t, azh, bz,
#endif
#if DW_NDIM >= 2
        grad_dby_shot, by_grad_store_1_t, ayh, by,
#endif
        grad_dbx_shot, bx_grad_store_1_t, axh, bx, db_grad_t && !has_defer);
    CHECK_KERNEL_ERROR

    // Periodic faces: accumulate the adjoint-stencil contributions that fall
    // on the scattered adjoint velocity's guard cells (and the slaved extra
    // high interior node), then fold them back into the interior (the exact
    // transpose of the forward velocity fill). The fold modifies interior
    // seam nodes, so it must precede the source-gradient reads below.
    if (has_per) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vzsc, vysc, vxsc, psc, k, k_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vysc, vxsc, psc, k, k_batched_h);
#else
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vxsc, psc, k, k_batched_h);
#endif
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_SC();
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vzsc, vysc, vxsc, psc, k, k_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vysc, vxsc, psc, k, k_batched_h);
#else
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vxsc, psc, k, k_batched_h);
#endif
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_SC_FRX();
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM >= 3
    RECORD_RCV(grad_f_vz_sc, vzsc, sources_i_vz_sc, n_sources_vz_sc_per_shot_h)
#endif
#if DW_NDIM >= 2
    RECORD_RCV(grad_f_vy_sc, vysc, sources_i_vy_sc, n_sources_vy_sc_per_shot_h)
#endif
    RECORD_RCV(grad_f_vx_sc, vxsc, sources_i_vx_sc, n_sources_vx_sc_per_shot_h)

    // Adjoint guard fill of the scattered velocity field (read by the
    // adjoint pressure update) on the free/rigid faces: the
    // transpose-realising image, matching the forward. No-op for all-PML.
    if (has_fr) LAUNCH_FILL_V_BC_SC_FR();

    // Deferred perturbation-buoyancy-gradient accumulation (see
    // backward_grad_b_deferred_kernel): the periodic axes' accumulation
    // must use the post-fold scattered adjoint velocity. The background
    // pointers are unused (do_b is false).
    if (has_defer && db_grad_t) {
      backward_grad_b_deferred_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
          NULL, vzsc, NULL, grad_dbz_shot, bz_grad_store_1_t, NULL,
#endif
#if DW_NDIM >= 2
          NULL, vysc, NULL, grad_dby_shot, by_grad_store_1_t, NULL,
#endif
          NULL, vxsc, NULL, grad_dbx_shot, bx_grad_store_1_t, NULL, false,
          true);
      CHECK_KERNEL_ERROR
    }

    backward_sc_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        psc,
#if DW_NDIM >= 3
        vzsc,
#endif
#if DW_NDIM >= 2
        vysc,
#endif
        vxsc,
#if DW_NDIM >= 3
        phi_z_sc,
#endif
#if DW_NDIM >= 2
        phi_y_sc,
#endif
        phi_x_sc,
#if DW_NDIM >= 3
        psi_z_sc,
#endif
#if DW_NDIM >= 2
        psi_y_sc,
#endif
        psi_x_sc, k,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x, grad_dk_shot, k_grad_store_1_t,
#if DW_NDIM >= 3
        az, bzh,
#endif
#if DW_NDIM >= 2
        ay, byh,
#endif
        ax, bxh, dk_grad_t);
    CHECK_KERNEL_ERROR

    // Periodic faces: accumulate the adjoint-stencil contributions that fall
    // on the scattered adjoint pressure's guard cells (folded into the
    // interior at the top of the next iteration).
    if (has_per) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(psc, vzsc, buoyancy_z, vysc, buoyancy_y,
                                     vxsc, buoyancy_x, b_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(psc, vysc, buoyancy_y, vxsc, buoyancy_x,
                                     b_batched_h);
#else
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(psc, vxsc, buoyancy_x, b_batched_h);
#endif
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(psc, vzsc, buoyancy_z, vysc,
                                         buoyancy_y, vxsc, buoyancy_x,
                                         b_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(psc, vysc, buoyancy_y, vxsc,
                                         buoyancy_x, b_batched_h);
#else
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(psc, vxsc, buoyancy_x, b_batched_h);
#endif
      CHECK_KERNEL_ERROR
    }

    // Inject scattered adjoint sources (data residuals).
    ADD_SRC(psc, grad_r_p_sc, receivers_i_p_sc, n_receivers_p_sc_per_shot_h)
#if DW_NDIM >= 3
    ADD_SRC(vzsc, grad_r_vz_sc, receivers_i_vz_sc, n_receivers_vz_sc_per_shot_h)
#endif
#if DW_NDIM >= 2
    ADD_SRC(vysc, grad_r_vy_sc, receivers_i_vy_sc, n_receivers_vy_sc_per_shot_h)
#endif
    ADD_SRC(vxsc, grad_r_vx_sc, receivers_i_vx_sc, n_receivers_vx_sc_per_shot_h)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

#if DW_NDIM >= 3
    std::swap(psi_z_sc, psi_zn_sc);
#endif
#if DW_NDIM >= 2
    std::swap(psi_y_sc, psi_yn_sc);
#endif
    std::swap(psi_x_sc, psi_xn_sc);
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

  if (dk_requires_grad && !dk_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dk, grad_dk_shot);
    CHECK_KERNEL_ERROR
  }
#if DW_NDIM >= 3
  if (db_requires_grad && !db_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbz, grad_dbz_shot);
    CHECK_KERNEL_ERROR
  }
#endif
#if DW_NDIM >= 2
  if (db_requires_grad && !db_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dby, grad_dby_shot);
    CHECK_KERNEL_ERROR
  }
#endif
  if (db_requires_grad && !db_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbx, grad_dbx_shot);
    CHECK_KERNEL_ERROR
  }

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
        DW_DTYPE const *__restrict__ const k,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
        DW_DTYPE const *__restrict__ const buoyancy_x,
        DW_DTYPE const *__restrict__ const dk,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const dbuoyancy_y,
#endif
        DW_DTYPE const *__restrict__ const dbuoyancy_x,
        DW_DTYPE const *__restrict__ const grad_r_p,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const grad_r_vz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const grad_r_vy,
#endif
        DW_DTYPE const *__restrict__ const grad_r_vx,
        DW_DTYPE const *__restrict__ const grad_r_p_sc,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const grad_r_vz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const grad_r_vy_sc,
#endif
        DW_DTYPE const *__restrict__ const grad_r_vx_sc,
        DW_DTYPE *__restrict__ const p,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const vy,
#endif
        DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const phi_y,
#endif
        DW_DTYPE *__restrict__ const phi_x,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ psi_z,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ psi_y,
#endif
        DW_DTYPE *__restrict__ psi_x,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ psi_zn,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ psi_yn,
#endif
        DW_DTYPE *__restrict__ psi_xn, DW_DTYPE *__restrict__ const psc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const vzsc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const vysc,
#endif
        DW_DTYPE *__restrict__ const vxsc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const phi_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const phi_y_sc,
#endif
        DW_DTYPE *__restrict__ const phi_x_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ psi_z_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ psi_y_sc,
#endif
        DW_DTYPE *__restrict__ psi_x_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ psi_zn_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ psi_yn_sc,
#endif
        DW_DTYPE *__restrict__ psi_xn_sc,
        DW_DTYPE *__restrict__ const k_grad_store_1a,
        DW_DTYPE *__restrict__ const k_grad_store_1b,
        void *__restrict__ const k_grad_store_2,
        void *__restrict__ const k_grad_store_3,
        char const *__restrict__ const *__restrict__ const k_grad_filenames_ptr,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const bz_grad_store_1a,
        DW_DTYPE *__restrict__ const bz_grad_store_1b,
        void *__restrict__ const bz_grad_store_2,
        void *__restrict__ const bz_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const by_grad_store_1a,
        DW_DTYPE *__restrict__ const by_grad_store_1b,
        void *__restrict__ const by_grad_store_2,
        void *__restrict__ const by_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const by_grad_filenames_ptr,
#endif
        DW_DTYPE *__restrict__ const bx_grad_store_1a,
        DW_DTYPE *__restrict__ const bx_grad_store_1b,
        void *__restrict__ const bx_grad_store_2,
        void *__restrict__ const bx_grad_store_3,
        char const *__restrict__ const
            *__restrict__ const bx_grad_filenames_ptr,
        DW_DTYPE *__restrict__ const k_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const k_grad_sc_store_1b,
        void *__restrict__ const k_grad_sc_store_2,
        void *__restrict__ const k_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const k_grad_sc_filenames_ptr,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const bz_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const bz_grad_sc_store_1b,
        void *__restrict__ const bz_grad_sc_store_2,
        void *__restrict__ const bz_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const bz_grad_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const by_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const by_grad_sc_store_1b,
        void *__restrict__ const by_grad_sc_store_2,
        void *__restrict__ const by_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const by_grad_sc_filenames_ptr,
#endif
        DW_DTYPE *__restrict__ const bx_grad_sc_store_1a,
        DW_DTYPE *__restrict__ const bx_grad_sc_store_1b,
        void *__restrict__ const bx_grad_sc_store_2,
        void *__restrict__ const bx_grad_sc_store_3,
        char const *__restrict__ const
            *__restrict__ const bx_grad_sc_filenames_ptr,
        DW_DTYPE *__restrict__ const grad_f_p,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_f_vz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_f_vy,
#endif
        DW_DTYPE *__restrict__ const grad_f_vx,
        DW_DTYPE *__restrict__ const grad_f_p_sc,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_f_vz_sc,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_f_vy_sc,
#endif
        DW_DTYPE *__restrict__ const grad_f_vx_sc,
        DW_DTYPE *__restrict__ const grad_k,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_bz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_by,
#endif
        DW_DTYPE *__restrict__ const grad_bx,
        DW_DTYPE *__restrict__ const grad_dk,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_dbz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_dby,
#endif
        DW_DTYPE *__restrict__ const grad_dbx,
        DW_DTYPE *__restrict__ const grad_k_shot,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_bz_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_by_shot,
#endif
        DW_DTYPE *__restrict__ const grad_bx_shot,
        DW_DTYPE *__restrict__ const grad_dk_shot,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict__ const grad_dbz_shot,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict__ const grad_dby_shot,
#endif
        DW_DTYPE *__restrict__ const grad_dbx_shot,
#if DW_NDIM >= 3
        DW_DTYPE const *__restrict__ const az,
        DW_DTYPE const *__restrict__ const bz,
        DW_DTYPE const *__restrict__ const azh,
        DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
        DW_DTYPE const *__restrict__ const ay,
        DW_DTYPE const *__restrict__ const by,
        DW_DTYPE const *__restrict__ const ayh,
        DW_DTYPE const *__restrict__ const byh,
#endif
        DW_DTYPE const *__restrict__ const ax,
        DW_DTYPE const *__restrict__ const bx,
        DW_DTYPE const *__restrict__ const axh,
        DW_DTYPE const *__restrict__ const bxh,
        int64_t const *__restrict__ const sources_i_p,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const sources_i_vz,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const sources_i_vy,
#endif
        int64_t const *__restrict__ const sources_i_vx,
        int64_t const *__restrict__ const sources_i_p_sc,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const sources_i_vz_sc,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const sources_i_vy_sc,
#endif
        int64_t const *__restrict__ const sources_i_vx_sc,
        int64_t const *__restrict__ const receivers_i_p,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const receivers_i_vz,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const receivers_i_vy,
#endif
        int64_t const *__restrict__ const receivers_i_vx,
        int64_t const *__restrict__ const receivers_i_p_sc,
#if DW_NDIM >= 3
        int64_t const *__restrict__ const receivers_i_vz_sc,
#endif
#if DW_NDIM >= 2
        int64_t const *__restrict__ const receivers_i_vy_sc,
#endif
        int64_t const *__restrict__ const receivers_i_vx_sc,
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
        int64_t const nx_h, int64_t const n_sources_p_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_sources_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_vy_per_shot_h,
#endif
        int64_t const n_sources_vx_per_shot_h,
        int64_t const n_sources_p_sc_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_sources_vz_sc_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_sources_vy_sc_per_shot_h,
#endif
        int64_t const n_sources_vx_sc_per_shot_h,
        int64_t const n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_vy_per_shot_h,
#endif
        int64_t const n_receivers_vx_per_shot_h,
        int64_t const n_receivers_p_sc_per_shot_h,
#if DW_NDIM >= 3
        int64_t const n_receivers_vz_sc_per_shot_h,
#endif
#if DW_NDIM >= 2
        int64_t const n_receivers_vy_sc_per_shot_h,
#endif
        int64_t const n_receivers_vx_sc_per_shot_h, int64_t const step_ratio_h,
        int64_t const storage_mode, size_t const shot_bytes_uncomp,
        size_t const shot_bytes_comp, bool const k_requires_grad,
        bool const b_requires_grad, bool const dk_requires_grad,
        bool const db_requires_grad, bool const k_batched_h,
        bool const b_batched_h, bool const dk_batched_h,
        bool const db_batched_h, bool const storage_compression,
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
    if ((k_requires_grad) && (grad_k == NULL)) return 1;
#if DW_NDIM >= 3
    if ((b_requires_grad) && (grad_bz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((b_requires_grad) && (grad_by == NULL)) return 1;
#endif
    if ((b_requires_grad) && (grad_bx == NULL)) return 1;
    if ((dk_requires_grad) && (grad_dk == NULL)) return 1;
    if ((n_receivers_p_per_shot_h > 0) && (receivers_i_p == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_per_shot_h > 0) && (receivers_i_vz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_per_shot_h > 0) && (receivers_i_vy == NULL)) return 1;
#endif
    if ((n_receivers_vx_per_shot_h > 0) && (receivers_i_vx == NULL)) return 1;
    if ((n_receivers_p_sc_per_shot_h > 0) && (receivers_i_p_sc == NULL))
      return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_sc_per_shot_h > 0) && (receivers_i_vz_sc == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_sc_per_shot_h > 0) && (receivers_i_vy_sc == NULL))
      return 1;
#endif
    if ((n_receivers_vx_sc_per_shot_h > 0) && (receivers_i_vx_sc == NULL))
      return 1;
    if ((n_sources_p_per_shot_h > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot_h > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot_h > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot_h > 0) && sources_i_vx == NULL) return 1;
    if ((n_sources_p_sc_per_shot_h > 0) && sources_i_p_sc == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_sc_per_shot_h > 0) && sources_i_vz_sc == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_sc_per_shot_h > 0) && sources_i_vy_sc == NULL) return 1;
#endif
    if ((n_sources_vx_sc_per_shot_h > 0) && sources_i_vx_sc == NULL) return 1;
  }
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

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif

#if DW_NDIM == 3
  int64_t const bc[6] = {bc_z0, bc_z1, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_z0 || bc_z1 || bc_y0 || bc_y1 || bc_x0 || bc_x1;
#elif DW_NDIM == 2
  int64_t const bc[6] = {0, 0, bc_y0, bc_y1, bc_x0, bc_x1};
  int const has_bc = bc_y0 || bc_y1 || bc_x0 || bc_x1;
#else
  int64_t const bc[6] = {0, 0, 0, 0, bc_x0, bc_x1};
  int const has_bc = bc_x0 || bc_x1;
#endif

  /* Periodic faces need the adjoint fold/gather treatment (see
   * acoustic_born_kernels.h); free/rigid faces keep the forward re-fill.
   * bc_fr is bc with the periodic axes suppressed. */
#ifndef DW_AXIS_IS_REFILL
/* An axis keeps the historical adjoint re-fill only for the (free-low,
 * rigid-high) face pairing; every other free/rigid combination is
 * fold/gather-handled (see acoustic_born_kernels.h's _fr functions). */
#define DW_AXIS_IS_REFILL(lo, hi) ((lo) == 1 && (hi) == 2)
#endif
  int64_t const bc_fr[6] = {DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[0] : 0,
                            DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[1] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[2] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[3] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[4] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[5] : 0};
  int64_t const bc_frx[6] = {
      bc[0] == 3 || bc_fr[0] ? 0 : bc[0], bc[1] == 3 || bc_fr[1] ? 0 : bc[1],
      bc[2] == 3 || bc_fr[2] ? 0 : bc[2], bc[3] == 3 || bc_fr[3] ? 0 : bc[3],
      bc[4] == 3 || bc_fr[4] ? 0 : bc[4], bc[5] == 3 || bc_fr[5] ? 0 : bc[5]};
  int const has_frx = bc_frx[0] || bc_frx[1] || bc_frx[2] || bc_frx[3] ||
                      bc_frx[4] || bc_frx[5];
  int const has_fr =
      bc_fr[0] || bc_fr[1] || bc_fr[2] || bc_fr[3] || bc_fr[4] || bc_fr[5];
#if DW_NDIM == 3
  int const has_per = bc[0] == 3 || bc[2] == 3 || bc[4] == 3;
#elif DW_NDIM == 2
  int const has_per = bc[2] == 3 || bc[4] == 3;
#else
  int const has_per = bc[4] == 3;
#endif
  /* Any fold-handled axis (periodic or exact-fr) defers the
   * (perturbation-)buoyancy gradient to the post-fold deferred kernel. */
  int const has_defer = has_per || has_frx;
  (void)has_bc;

  int64_t t;
  ScopedCudaDevice device_guard(device);
  if (device_guard.init_error != 0) return device_guard.init_error;
  ScopedCudaConfig config_guard(device);
  if (config_guard.init_error != 0) return config_guard.init_error;
  ScopedFileGroup file_group;
  {
    int err = set_config(
#if DW_NDIM >= 3
        rdz_h, nz_h, pml_z0_h, pml_z1_h,
#endif
#if DW_NDIM >= 2
        rdy_h, ny_h, pml_y0_h, pml_y1_h,
#endif
        rdx_h, nx_h, shot_numel_h, pml_x0_h, pml_x1_h, dt_h, n_shots_h,
        step_ratio_h, k_batched_h, b_batched_h);
    if (err != 0) return err;
  }

  bool const store_needed_h = k_requires_grad || b_requires_grad ||
                              dk_requires_grad || db_requires_grad;
  bool const store_sc_needed_h = k_requires_grad || b_requires_grad;

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (store_needed_h || store_sc_needed_h);

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

  OPEN_FILE_READ(k_grad, store_needed_h)
#if DW_NDIM >= 3
  OPEN_FILE_READ(bz_grad, store_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(by_grad, store_needed_h)
#endif
  OPEN_FILE_READ(bx_grad, store_needed_h)
  OPEN_FILE_READ(k_grad_sc, store_sc_needed_h)
#if DW_NDIM >= 3
  OPEN_FILE_READ(bz_grad_sc, store_sc_needed_h)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(by_grad_sc, store_sc_needed_h)
#endif
  OPEN_FILE_READ(bx_grad_sc, store_sc_needed_h)

  for (t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const step_idx = t / step_ratio_h;
    cudaEvent_t event_storage_done = event_storage_done_a;
    cudaEvent_t event_compute_done = event_compute_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_storage_done = event_storage_done_b;
        event_compute_done = event_compute_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_storage, event_compute_done, 0));
    }

    SETUP_STORE_LOAD(k_grad, store_needed_h)
#if DW_NDIM >= 3
    SETUP_STORE_LOAD(bz_grad, store_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(by_grad, store_needed_h)
#endif
    SETUP_STORE_LOAD(bx_grad, store_needed_h)
    SETUP_STORE_LOAD(k_grad_sc, store_sc_needed_h)
#if DW_NDIM >= 3
    SETUP_STORE_LOAD(bz_grad_sc, store_sc_needed_h)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(by_grad_sc, store_sc_needed_h)
#endif
    SETUP_STORE_LOAD(bx_grad_sc, store_sc_needed_h)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    bool const k_grad_t = k_requires_grad && ((t % step_ratio_h) == 0);
    bool const b_grad_t = b_requires_grad && ((t % step_ratio_h) == 0);
    bool const dk_grad_t = dk_requires_grad && ((t % step_ratio_h) == 0);
    bool const db_grad_t = db_requires_grad && ((t % step_ratio_h) == 0);

    // Periodic faces: fold both adjoint pressure fields' guard cells back
    // into the interior nodes the forward fill copied them from (the exact
    // transpose of the fill). The fold modifies interior seam nodes, so it
    // must precede the source-gradient reads below.
    if (has_per) {
      LAUNCH_ADJ_FOLD_P_BC(p);
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_P_BC(psc);
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
      LAUNCH_ADJ_FOLD_P_BC_FRX(p);
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_P_BC_FRX(psc);
      CHECK_KERNEL_ERROR
    }

    // Record source gradients from the adjoint pressure (background then
    // scattered).
    RECORD_RCV(grad_f_p, p, sources_i_p, n_sources_p_per_shot_h)
    RECORD_RCV(grad_f_p_sc, psc, sources_i_p_sc, n_sources_p_sc_per_shot_h)

    // Adjoint guard fill of the pressure field (read by the adjoint velocity
    // update), on both fields independently, on the free/rigid faces: the
    // same image the forward applies, which realises the exact transpose on
    // a reflecting face. No-op for all-PML.
    if (has_fr) {
      LAUNCH_FILL_P_BC_FR(p);
      LAUNCH_FILL_P_BC_FR(psc);
    }

    backward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, psc,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        vzsc,
#endif
#if DW_NDIM >= 2
        vysc,
#endif
        vxsc,
#if DW_NDIM >= 3
        phi_z, phi_z_sc,
#endif
#if DW_NDIM >= 2
        phi_y, phi_y_sc,
#endif
        phi_x, phi_x_sc,
#if DW_NDIM >= 3
        psi_z, psi_z_sc, psi_zn, psi_zn_sc,
#endif
#if DW_NDIM >= 2
        psi_y, psi_y_sc, psi_yn, psi_yn_sc,
#endif
        psi_x, psi_x_sc, psi_xn, psi_xn_sc, k, dk,
#if DW_NDIM >= 3
        buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y, dbuoyancy_y,
#endif
        buoyancy_x, dbuoyancy_x,
#if DW_NDIM >= 3
        grad_bz_shot, grad_dbz_shot, bz_grad_store_1_t, bz_grad_sc_store_1_t,
        azh, bz,
#endif
#if DW_NDIM >= 2
        grad_by_shot, grad_dby_shot, by_grad_store_1_t, by_grad_sc_store_1_t,
        ayh, by,
#endif
        grad_bx_shot, grad_dbx_shot, bx_grad_store_1_t, bx_grad_sc_store_1_t,
        axh, bx, b_grad_t && !has_defer, db_grad_t && !has_defer, dk_batched_h,
        db_batched_h);
    CHECK_KERNEL_ERROR

    // Periodic faces: accumulate the adjoint-stencil contributions that fall
    // on both adjoint velocity sets' guard cells (and the slaved extra high
    // interior nodes), then fold them back into the interior (the exact
    // transpose of the forward velocity fill). The background set's
    // Born-coupled payload (K*P_adj + dK*P_adj_sc) is composed as two
    // single-payload gather launches. The folds modify interior seam nodes,
    // so they must precede the source-gradient reads below.
    if (has_per) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vz, vy, vx, p, k, k_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vz, vy, vx, psc, dk, dk_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vzsc, vysc, vxsc, psc, k, k_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vy, vx, p, k, k_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vy, vx, psc, dk, dk_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vysc, vxsc, psc, k, k_batched_h);
#else
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vx, p, k, k_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vx, psc, dk, dk_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET(vxsc, psc, k, k_batched_h);
#endif
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_BG();
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_SC();
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vz, vy, vx, p, k, k_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vz, vy, vx, psc, dk, dk_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vzsc, vysc, vxsc, psc, k, k_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vy, vx, p, k, k_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vy, vx, psc, dk, dk_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vysc, vxsc, psc, k, k_batched_h);
#else
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vx, p, k, k_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vx, psc, dk, dk_batched_h);
      LAUNCH_ADJ_GATHER_V_GUARDS_SET_FRX(vxsc, psc, k, k_batched_h);
#endif
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_BG_FRX();
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_SC_FRX();
      CHECK_KERNEL_ERROR
    }

    // Record velocity-component source gradients (background then scattered).
#if DW_NDIM >= 3
    RECORD_RCV(grad_f_vz, vz, sources_i_vz, n_sources_vz_per_shot_h)
    RECORD_RCV(grad_f_vz_sc, vzsc, sources_i_vz_sc, n_sources_vz_sc_per_shot_h)
#endif
#if DW_NDIM >= 2
    RECORD_RCV(grad_f_vy, vy, sources_i_vy, n_sources_vy_per_shot_h)
    RECORD_RCV(grad_f_vy_sc, vysc, sources_i_vy_sc, n_sources_vy_sc_per_shot_h)
#endif
    RECORD_RCV(grad_f_vx, vx, sources_i_vx, n_sources_vx_per_shot_h)
    RECORD_RCV(grad_f_vx_sc, vxsc, sources_i_vx_sc, n_sources_vx_sc_per_shot_h)

    // Adjoint guard fill of the velocity field (read by the adjoint pressure
    // update), on both fields independently, on the free/rigid faces: the
    // transpose-realising image, matching the forward. No-op for all-PML.
    if (has_fr) {
      LAUNCH_FILL_V_BC_BG_FR();
      LAUNCH_FILL_V_BC_SC_FR();
    }

    // Deferred (perturbation-)buoyancy-gradient accumulation (see
    // backward_grad_b_deferred_kernel): the periodic axes' accumulation
    // must use the post-fold adjoint velocities.
    if (has_defer && (b_grad_t || db_grad_t)) {
      backward_grad_b_deferred_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
          vz, vzsc, grad_bz_shot, grad_dbz_shot, bz_grad_store_1_t,
          bz_grad_sc_store_1_t,
#endif
#if DW_NDIM >= 2
          vy, vysc, grad_by_shot, grad_dby_shot, by_grad_store_1_t,
          by_grad_sc_store_1_t,
#endif
          vx, vxsc, grad_bx_shot, grad_dbx_shot, bx_grad_store_1_t,
          bx_grad_sc_store_1_t, b_grad_t, db_grad_t);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, psc,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        vzsc,
#endif
#if DW_NDIM >= 2
        vysc,
#endif
        vxsc,
#if DW_NDIM >= 3
        phi_z, phi_z_sc,
#endif
#if DW_NDIM >= 2
        phi_y, phi_y_sc,
#endif
        phi_x, phi_x_sc,
#if DW_NDIM >= 3
        psi_z, psi_z_sc,
#endif
#if DW_NDIM >= 2
        psi_y, psi_y_sc,
#endif
        psi_x, psi_x_sc, k, dk,
#if DW_NDIM >= 3
        buoyancy_z, dbuoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y, dbuoyancy_y,
#endif
        buoyancy_x, dbuoyancy_x, grad_k_shot, grad_dk_shot, k_grad_store_1_t,
        k_grad_sc_store_1_t,
#if DW_NDIM >= 3
        az, bzh,
#endif
#if DW_NDIM >= 2
        ay, byh,
#endif
        ax, bxh, k_grad_t, dk_grad_t, dk_batched_h, db_batched_h);
    CHECK_KERNEL_ERROR

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Periodic faces: accumulate the adjoint-stencil contributions that fall
    // on both adjoint pressure fields' guard cells (folded into the interior
    // at the top of the next iteration). The background field's Born-coupled
    // payload (B*V_adj + dB*V_adj_sc) is composed as two single-payload
    // gather launches.
    if (has_per) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(p, vz, buoyancy_z, vy, buoyancy_y, vx,
                                     buoyancy_x, b_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(p, vzsc, dbuoyancy_z, vysc, dbuoyancy_y,
                                     vxsc, dbuoyancy_x, db_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(psc, vzsc, buoyancy_z, vysc, buoyancy_y,
                                     vxsc, buoyancy_x, b_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(p, vy, buoyancy_y, vx, buoyancy_x,
                                     b_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(p, vysc, dbuoyancy_y, vxsc, dbuoyancy_x,
                                     db_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(psc, vysc, buoyancy_y, vxsc, buoyancy_x,
                                     b_batched_h);
#else
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(p, vx, buoyancy_x, b_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(p, vxsc, dbuoyancy_x, db_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET(psc, vxsc, buoyancy_x, b_batched_h);
#endif
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
#if DW_NDIM == 3
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(p, vz, buoyancy_z, vy, buoyancy_y, vx,
                                         buoyancy_x, b_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(p, vzsc, dbuoyancy_z, vysc,
                                         dbuoyancy_y, vxsc, dbuoyancy_x,
                                         db_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(psc, vzsc, buoyancy_z, vysc,
                                         buoyancy_y, vxsc, buoyancy_x,
                                         b_batched_h);
#elif DW_NDIM == 2
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(p, vy, buoyancy_y, vx, buoyancy_x,
                                         b_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(p, vysc, dbuoyancy_y, vxsc,
                                         dbuoyancy_x, db_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(psc, vysc, buoyancy_y, vxsc,
                                         buoyancy_x, b_batched_h);
#else
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(p, vx, buoyancy_x, b_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(p, vxsc, dbuoyancy_x, db_batched_h);
      LAUNCH_ADJ_GATHER_P_GUARDS_SET_FRX(psc, vxsc, buoyancy_x, b_batched_h);
#endif
      CHECK_KERNEL_ERROR
    }

    // Inject adjoint sources (data residuals): background then scattered.
    ADD_SRC(p, grad_r_p, receivers_i_p, n_receivers_p_per_shot_h)
#if DW_NDIM >= 3
    ADD_SRC(vz, grad_r_vz, receivers_i_vz, n_receivers_vz_per_shot_h)
#endif
#if DW_NDIM >= 2
    ADD_SRC(vy, grad_r_vy, receivers_i_vy, n_receivers_vy_per_shot_h)
#endif
    ADD_SRC(vx, grad_r_vx, receivers_i_vx, n_receivers_vx_per_shot_h)
    ADD_SRC(psc, grad_r_p_sc, receivers_i_p_sc, n_receivers_p_sc_per_shot_h)
#if DW_NDIM >= 3
    ADD_SRC(vzsc, grad_r_vz_sc, receivers_i_vz_sc, n_receivers_vz_sc_per_shot_h)
#endif
#if DW_NDIM >= 2
    ADD_SRC(vysc, grad_r_vy_sc, receivers_i_vy_sc, n_receivers_vy_sc_per_shot_h)
#endif
    ADD_SRC(vxsc, grad_r_vx_sc, receivers_i_vx_sc, n_receivers_vx_sc_per_shot_h)

#if DW_NDIM >= 3
    std::swap(psi_z, psi_zn);
    std::swap(psi_z_sc, psi_zn_sc);
#endif
#if DW_NDIM >= 2
    std::swap(psi_y, psi_yn);
    std::swap(psi_y_sc, psi_yn_sc);
#endif
    std::swap(psi_x, psi_xn);
    std::swap(psi_x_sc, psi_xn_sc);
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

  if (k_requires_grad && !k_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_k, grad_k_shot);
    CHECK_KERNEL_ERROR
  }
#if DW_NDIM >= 3
  if (b_requires_grad && !b_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_bz, grad_bz_shot);
    CHECK_KERNEL_ERROR
  }
#endif
#if DW_NDIM >= 2
  if (b_requires_grad && !b_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_by, grad_by_shot);
    CHECK_KERNEL_ERROR
  }
#endif
  if (b_requires_grad && !b_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_bx, grad_bx_shot);
    CHECK_KERNEL_ERROR
  }
  if (dk_requires_grad && !dk_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dk, grad_dk_shot);
    CHECK_KERNEL_ERROR
  }
#if DW_NDIM >= 3
  if (db_requires_grad && !db_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbz, grad_dbz_shot);
    CHECK_KERNEL_ERROR
  }
#endif
#if DW_NDIM >= 2
  if (db_requires_grad && !db_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dby, grad_dby_shot);
    CHECK_KERNEL_ERROR
  }
#endif
  if (db_requires_grad && !db_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dbx, grad_dbx_shot);
    CHECK_KERNEL_ERROR
  }

  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

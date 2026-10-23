/*
 * Viscoacoustic wave equation propagator (variable density, single-mechanism
 * standard-linear-solid attenuation, staggered grid). CUDA implementation.
 *
 * PHYSICS
 * -------
 * Variable-density acoustic (template: acoustic.c) plus one standard-linear-
 * solid (SLS) relaxation mechanism on the pressure equation. With B = 1/rho,
 * K = rho v^2 the RELAXED (zero-frequency) bulk modulus, and r the SLS memory
 * variable at the pressure nodes:
 *
 *     rho dv/dt = -grad p                               (velocity, as acoustic)
 *     dp/dt = -K(1+tau) div(v) - r
 *     dr/dt = -(1/tausig)( r + K tau div(v) )
 *
 * tau = taueps/tausig - 1 (relaxation defect). The user supplies a quality
 * factor Q at reference angular frequency omega0 = 2*pi*q_freq; Python maps it
 * to (taueps, tausig) via the constant-Q-at-omega0 single-SLS relations
 *     tausig = (sqrt(1+1/Q^2) - 1/Q)/omega0,  taueps = 1/(omega0^2 tausig),
 * so the SLS quality factor Q(omega) = (Q/2)(omega/omega0 + omega0/omega)
 * equals Q exactly at omega0. The complex modulus is
 *     M(omega) = K (1 + i omega taueps)/(1 + i omega tausig),
 * giving phase velocity c(omega) = 1/Re(sqrt(rho/M)) and, for Q>>1, amplitude
 * decay exp(-omega x/(2 c Q)). As Q->inf, tau->0 and this reduces exactly to
 * variable-density acoustic.
 *
 * DISCRETISATION (staggered leapfrog; p, r, K at integer time n at the
 * pressure nodes; v at n+1/2 at the velocity nodes). With theta =
 * div(v^{n+1/2}) (the PML-augmented divergence, identical to acoustic's div_v),
 * the memory variable uses an unconditionally-stable Crank-Nicolson
 * (trapezoidal) update centred at n+1/2, and the pressure uses r^{n+1/2} = (r^n
 * + r^{n+1})/2:
 *
 *     r^{n+1} = ar r^n - c2 theta                        ar=(1-h)/(1+h),
 * h=dt/(2 tausig) p^{n+1} = p^n - dt*m1*theta - (dt/2)(r^n + r^{n+1})
 *
 * where the Python side supplies the per-node model fields
 *     m1 = K(1+tau),  c2 = K tau b_r  (b_r=(dt/tausig)/(1+h)),  ar,
 * plus the buoyancies (identical to acoustic). All are differentiable
 * functions of (v, rho, Q); the kernel returns grad_m1/grad_c2/grad_ar/grad_b
 * and Python chains them to v, rho, Q by autograd.
 *
 * ADJOINT (discrete transpose of one step; Pbar, Rbar, Vbar are the adjoints of
 * p^{n+1}, r^{n+1}, v^{n+1/2}). Writing the effective pressure-adjoint field
 *     B_eff = dt*(m1 - c2/2)*Pbar + c2*Rbar     (acoustic uses just dt*K*Pbar),
 * so thetabar = -B_eff, the velocity-adjoint and PML recursions are IDENTICAL
 * to acoustic with dt*K*p replaced by B_eff. The genuinely new pieces are the
 * memory-variable adjoint recursion (an SLS-like relaxation run BACKWARD in
 * time) and the model-gradient accumulation:
 *
 *     Rbar_new (dL/dr^n) = ar*Rbar - (dt/2)(1+ar)*Pbar
 *     grad_m1 += -dt*theta*Pbar
 *     grad_c2 +=  theta*((dt/2)*Pbar - Rbar)
 *     grad_ar +=  r^n*(Rbar - (dt/2)*Pbar)
 *
 * The forward stores theta (for grad_m1 and grad_c2) and r^n (for grad_ar).
 * See docs/viscoacoustic.rst for the full derivation.
 */

#include <stdio.h>

#include <algorithm>  // for std::swap
#include <cstdint>

#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  viscoacoustic_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
/* Multi-mechanism GSLS (DW_GSLS) build variant -- see viscoacoustic.c's
 * matching comment for the full explanation. Must mirror viscoacoustic.c
 * exactly, since the exported symbol names (and thus the ctypes ABI) must
 * agree between the CPU and CUDA builds. */
#ifdef DW_GSLS
#define VISCO_VARIANT_I(name) name##_gsls
#define VISCO_VARIANT(name) VISCO_VARIANT_I(name)
#else
#define VISCO_VARIANT(name) name
#endif
#define FUNC(name) \
  CAT(VISCO_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

/* Threading of the GSLS mechanism count into the gather/stencil kernels
 * (the per-mechanism strides are derived in-kernel from the n_shots/
 * shot_numel constants). Expansion sites must have ``n_mech`` in scope. */
#ifdef DW_GSLS
#define DW_GSLS_NMECH_PARAM , int64_t const n_mech
#define DW_GSLS_NMECH_ARG , n_mech
#else
#define DW_GSLS_NMECH_PARAM
#define DW_GSLS_NMECH_ARG
#endif

#if DW_NDIM == 3
#define DIM_ARGS nz_h, ny_h, nx_h
#elif DW_NDIM == 2
#define DIM_ARGS ny_h, nx_h
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx_h
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* ND_INDEX, M1/C2/BUOYANCY_Z/Y/X, P/RMEM/VZ/VY/VX, PHI_Z/Y/X/PSI_Z/Y/X, B_EFF,
 * the VZ_ADJOINT/VY_ADJOINT/VX_ADJOINT/P_ADJOINT_Z/Y/X (+_PML) family, and the
 * FUNC(forward_vz_cell)/.../FUNC(backward_p_cell) per-grid-point stencil
 * update functions live in viscoacoustic_kernels.h, shared verbatim with
 * viscoacoustic.c (see that header for the full explanation). p/rmem/vz/.../
 * psi_z/... below are this file's (kernel-parameter or shot-offset-local)
 * names for the arrays/values passed as the p/rmem/vz/.../psi_z/.../
 * m1_shot/c2_shot/ar_shot/buoyancy_z_shot/... parameters of those
 * functions. */
#include "viscoacoustic_kernels.h"

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
__constant__ bool kmod_batched;
__constant__ bool ar_batched;
__constant__ bool b_batched;

/* First-class reflecting/periodic boundary faces on the staggered
 * viscoacoustic grid (see docs/boundaries.rst and viscoacoustic.c's matching
 * comment, which these device functions mirror exactly -- including using
 * the SAME fill for the pressure field ``p`` and the SLS memory variable
 * ``rmem``, per the ROBERTSSON_FS_SUMMARY.md Sec 1.3 citation there).
 * fill_p_bc_line_dev/fill_v_bc_line_dev/fill_bc_kernel live in
 * line_bc_dev.h, shared verbatim across every propagator's .cu file on this
 * staggered layout -- see that header. */
#include "line_bc_dev.h"

/* Pressure (or SLS memory-variable) guard fill on every reflecting axis
 * (innermost stride first). */
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

/* Periodic-face adjoint guard maintenance (backward only): fold/gather line
 * functions from viscoacoustic_kernels.h, one thread per line. The folds
 * run over every line (like the fills); the gathers run only over the other
 * axes' interior node ranges [FD_PAD, n - FD_PAD + 1). Mirrors acoustic.cu,
 * except the velocity-guard gather's payload is B_EFF (m1/c2/adjoint-p/
 * adjoint-rmem). */

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

/* One thread per (shot, other-axis interior position) line. */
__global__ void adj_gather_v_guard_kernel(
    DW_DTYPE *__restrict const v, DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE const *__restrict const m1,
    DW_DTYPE const *__restrict const c2, int64_t const n_lines,
    int64_t const nax, int64_t const stride, int64_t const n_a,
    int64_t const start_a, int64_t const stride_a, int64_t const n_b,
    int64_t const start_b, int64_t const stride_b,
    DW_DTYPE const rd DW_GSLS_NMECH_PARAM) {
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
  DW_DTYPE const *const m1_shot =
      kmod_batched ? m1 + shot_idx * shot_numel : m1;
  DW_DTYPE const *const c2_shot =
      kmod_batched ? c2 + shot_idx * shot_numel : c2;
#ifdef DW_GSLS
  int64_t const r_lstride = n_shots * shot_numel;
  int64_t const m_lstride = shot_numel;
#endif
  FUNC(adj_gather_v_guard_line)(v, p, rmem, m1_shot, c2_shot, base, jbase, nax,
                                stride, dt, rd, sw
#ifdef DW_GSLS
                                ,
                                n_mech, r_lstride, m_lstride
#endif
  );
}

__global__ void adj_gather_p_guard_kernel(
    DW_DTYPE *__restrict const p, DW_DTYPE const *__restrict const v,
    DW_DTYPE const *__restrict const buoyancy, int64_t const n_lines,
    int64_t const nax, int64_t const stride, int64_t const n_a,
    int64_t const start_a, int64_t const stride_a, int64_t const n_b,
    int64_t const start_b, int64_t const stride_b, DW_DTYPE const rd) {
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
  DW_DTYPE const *const b_shot =
      b_batched ? buoyancy + shot_idx * shot_numel : buoyancy;
  FUNC(adj_gather_p_guard_line)(p, v, b_shot, base, jbase, nax, stride, dt, rd,
                                sw);
}

static void launch_adj_gather_v_guards(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE const *__restrict const m1,
    DW_DTYPE const *__restrict const c2, int64_t const bc[6],
    int64_t const n_shots_, int64_t const nz_, int64_t const ny_,
    int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, cudaStream_t stream DW_GSLS_NMECH_PARAM) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  int64_t const mz = nz_ - 2 * FD_PAD + 1;
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * mz * my;
    adj_gather_v_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(vx, p, rmem, m1, c2, nl, nx_, 1, mz,
                                          FD_PAD, ny_ * nx_, my, FD_PAD, nx_,
                                          rdx_ DW_GSLS_NMECH_ARG);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_v_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(vy, p, rmem, m1, c2, nl, ny_, nx_, mz,
                                          FD_PAD, ny_ * nx_, mx, FD_PAD, 1,
                                          rdy_ DW_GSLS_NMECH_ARG);
  }
  if (bc[0] == 3) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_v_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(vz, p, rmem, m1, c2, nl, nz_,
                                          ny_ * nx_, my, FD_PAD, nx_, mx,
                                          FD_PAD, 1, rdz_ DW_GSLS_NMECH_ARG);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * my;
    adj_gather_v_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(vx, p, rmem, m1, c2, nl, nx_, 1, 1, 0,
                                          0, my, FD_PAD, nx_,
                                          rdx_ DW_GSLS_NMECH_ARG);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_v_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(vy, p, rmem, m1, c2, nl, ny_, nx_, 1,
                                          0, 0, mx, FD_PAD, 1,
                                          rdy_ DW_GSLS_NMECH_ARG);
  }
#else
  if (bc[4] == 3) {
    int64_t const nl = n_shots_;
    adj_gather_v_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(vx, p, rmem, m1, c2, nl, nx_, 1, 1, 0,
                                          0, 1, 0, 0, rdx_ DW_GSLS_NMECH_ARG);
  }
#endif
}

static void launch_adj_gather_p_guards(
    DW_DTYPE *__restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const buoyancy_x, int64_t const bc[6],
    int64_t const n_shots_, int64_t const nz_, int64_t const ny_,
    int64_t const nx_,
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
    adj_gather_p_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(p, vx, buoyancy_x, nl, nx_, 1, mz,
                                          FD_PAD, ny_ * nx_, my, FD_PAD, nx_,
                                          rdx_);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_p_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(p, vy, buoyancy_y, nl, ny_, nx_, mz,
                                          FD_PAD, ny_ * nx_, mx, FD_PAD, 1,
                                          rdy_);
  }
  if (bc[0] == 3) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_p_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(p, vz, buoyancy_z, nl, nz_, ny_ * nx_,
                                          my, FD_PAD, nx_, mx, FD_PAD, 1, rdz_);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bc[4] == 3) {
    int64_t const nl = n_shots_ * my;
    adj_gather_p_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(p, vx, buoyancy_x, nl, nx_, 1, 1, 0,
                                          0, my, FD_PAD, nx_, rdx_);
  }
  if (bc[2] == 3) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_p_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(p, vy, buoyancy_y, nl, ny_, nx_, 1, 0,
                                          0, mx, FD_PAD, 1, rdy_);
  }
#else
  if (bc[4] == 3) {
    int64_t const nl = n_shots_;
    adj_gather_p_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                stream>>>(p, vx, buoyancy_x, nl, nx_, 1, 1, 0,
                                          0, 1, 0, 0, rdx_);
  }
#endif
}

/* Free/rigid-face exact-adjoint guard maintenance (backward only), for the
 * free/rigid axes other than the (free-low, rigid-high) re-fill pairing:
 * same launch structure as the periodic fold/gather kernels above, calling
 * the _fr line functions (mirror/sign-flip transposes, per-face gated). */

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

__global__ void adj_gather_v_fr_guard_kernel(
    DW_DTYPE *__restrict const v, DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE const *__restrict const m1,
    DW_DTYPE const *__restrict const c2, int64_t const n_lines,
    int64_t const nax, int64_t const stride, int64_t const n_a,
    int64_t const start_a, int64_t const stride_a, int64_t const n_b,
    int64_t const start_b, int64_t const stride_b, DW_DTYPE const rd,
    int64_t const bc_lo, int64_t const bc_hi DW_GSLS_NMECH_PARAM) {
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
  DW_DTYPE const *const m1_shot =
      kmod_batched ? m1 + shot_idx * shot_numel : m1;
  DW_DTYPE const *const c2_shot =
      kmod_batched ? c2 + shot_idx * shot_numel : c2;
#ifdef DW_GSLS
  int64_t const r_lstride = n_shots * shot_numel;
  int64_t const m_lstride = shot_numel;
#endif
  FUNC(adj_gather_v_guard_line_fr)(v, p, rmem, m1_shot, c2_shot, base, jbase,
                                   nax, stride, dt, rd, sw, bc_lo, bc_hi
#ifdef DW_GSLS
                                   ,
                                   n_mech, r_lstride, m_lstride
#endif
  );
}

__global__ void adj_gather_p_fr_guard_kernel(
    DW_DTYPE *__restrict const p, DW_DTYPE const *__restrict const v,
    DW_DTYPE const *__restrict const buoyancy, int64_t const n_lines,
    int64_t const nax, int64_t const stride, int64_t const n_a,
    int64_t const start_a, int64_t const stride_a, int64_t const n_b,
    int64_t const start_b, int64_t const stride_b, DW_DTYPE const rd,
    int64_t const bc_lo, int64_t const bc_hi) {
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
  DW_DTYPE const *const b_shot =
      b_batched ? buoyancy + shot_idx * shot_numel : buoyancy;
  FUNC(adj_gather_p_guard_line_fr)(p, v, b_shot, base, jbase, nax, stride, dt,
                                   rd, sw, bc_lo, bc_hi);
}

static void launch_adj_gather_v_guards_frx(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, DW_DTYPE const *__restrict const p,
    DW_DTYPE const *__restrict const rmem, DW_DTYPE const *__restrict const m1,
    DW_DTYPE const *__restrict const c2, int64_t const bcx[6],
    int64_t const n_shots_, int64_t const nz_, int64_t const ny_,
    int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, cudaStream_t stream DW_GSLS_NMECH_PARAM) {
  unsigned int const tpb = 128;
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  int64_t const mz = nz_ - 2 * FD_PAD + 1;
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * mz * my;
    adj_gather_v_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(
        vx, p, rmem, m1, c2, nl, nx_, 1, mz, FD_PAD, ny_ * nx_, my, FD_PAD, nx_,
        rdx_, bcx[4], bcx[5] DW_GSLS_NMECH_ARG);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_v_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(
        vy, p, rmem, m1, c2, nl, ny_, nx_, mz, FD_PAD, ny_ * nx_, mx, FD_PAD, 1,
        rdy_, bcx[2], bcx[3] DW_GSLS_NMECH_ARG);
  }
  if (bcx[0] != 0 || bcx[1] != 0) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_v_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(
        vz, p, rmem, m1, c2, nl, nz_, ny_ * nx_, my, FD_PAD, nx_, mx, FD_PAD, 1,
        rdz_, bcx[0], bcx[1] DW_GSLS_NMECH_ARG);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * my;
    adj_gather_v_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(vx, p, rmem, m1, c2, nl, nx_, 1, 1,
                                             0, 0, my, FD_PAD, nx_, rdx_,
                                             bcx[4], bcx[5] DW_GSLS_NMECH_ARG);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_v_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(vy, p, rmem, m1, c2, nl, ny_, nx_,
                                             1, 0, 0, mx, FD_PAD, 1, rdy_,
                                             bcx[2], bcx[3] DW_GSLS_NMECH_ARG);
  }
#else
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_;
    adj_gather_v_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(vx, p, rmem, m1, c2, nl, nx_, 1, 1,
                                             0, 0, 1, 0, 0, rdx_, bcx[4],
                                             bcx[5] DW_GSLS_NMECH_ARG);
  }
#endif
}

static void launch_adj_gather_p_guards_frx(
    DW_DTYPE *__restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const buoyancy_x, int64_t const bcx[6],
    int64_t const n_shots_, int64_t const nz_, int64_t const ny_,
    int64_t const nx_,
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
    adj_gather_p_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(p, vx, buoyancy_x, nl, nx_, 1, mz,
                                             FD_PAD, ny_ * nx_, my, FD_PAD, nx_,
                                             rdx_, bcx[4], bcx[5]);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mz * mx;
    adj_gather_p_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(p, vy, buoyancy_y, nl, ny_, nx_,
                                             mz, FD_PAD, ny_ * nx_, mx, FD_PAD,
                                             1, rdy_, bcx[2], bcx[3]);
  }
  if (bcx[0] != 0 || bcx[1] != 0) {
    int64_t const nl = n_shots_ * my * mx;
    adj_gather_p_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(p, vz, buoyancy_z, nl, nz_,
                                             ny_ * nx_, my, FD_PAD, nx_, mx,
                                             FD_PAD, 1, rdz_, bcx[0], bcx[1]);
  }
#elif DW_NDIM == 2
  int64_t const my = ny_ - 2 * FD_PAD + 1;
  int64_t const mx = nx_ - 2 * FD_PAD + 1;
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_ * my;
    adj_gather_p_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(p, vx, buoyancy_x, nl, nx_, 1, 1,
                                             0, 0, my, FD_PAD, nx_, rdx_,
                                             bcx[4], bcx[5]);
  }
  if (bcx[2] != 0 || bcx[3] != 0) {
    int64_t const nl = n_shots_ * mx;
    adj_gather_p_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(p, vy, buoyancy_y, nl, ny_, nx_, 1,
                                             0, 0, mx, FD_PAD, 1, rdy_, bcx[2],
                                             bcx[3]);
  }
#else
  if (bcx[4] != 0 || bcx[5] != 0) {
    int64_t const nl = n_shots_;
    adj_gather_p_fr_guard_kernel<<<ceil_div((unsigned int)nl, tpb), tpb, 0,
                                   stream>>>(
        p, vx, buoyancy_x, nl, nx_, 1, 1, 0, 0, 1, 0, 0, rdx_, bcx[4], bcx[5]);
  }
#endif
}

/* Deferred buoyancy-gradient accumulation, used when any axis is periodic
 * (see acoustic.cu's kernel of the same name for the full rationale): the
 * periodic axes' accumulation must use the post-fold adjoint velocity, and
 * backward_kernel_v has a single b_requires_grad flag for every axis, so
 * when this kernel runs the fused accumulation is suppressed for all axes
 * and every axis is accumulated here instead (identical values for
 * non-periodic axes). Same thread geometry as backward_kernel_v. */
__launch_bounds__(128) __global__ void backward_grad_b_deferred_kernel(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const vz,
    DW_DTYPE *__restrict__ const grad_bz_shot,
    DW_DTYPE const *__restrict__ const bz_grad_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const vy,
    DW_DTYPE *__restrict__ const grad_by_shot,
    DW_DTYPE const *__restrict__ const by_grad_store_1,
#endif
    DW_DTYPE const *__restrict__ const vx,
    DW_DTYPE *__restrict__ const grad_bx_shot,
    DW_DTYPE const *__restrict__ const bx_grad_store_1) {
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
        grad_bz_shot[i] -=
            dt * vz[i] * bz_grad_store_1[i] * (DW_DTYPE)step_ratio;
      }
#endif
#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        grad_by_shot[i] -=
            dt * vy[i] * by_grad_store_1[i] * (DW_DTYPE)step_ratio;
      }
#endif
      if (x < nx - FD_PAD) {
        grad_bx_shot[i] -=
            dt * vx[i] * bx_grad_store_1[i] * (DW_DTYPE)step_ratio;
      }
#if DW_NDIM == 3
    }
#endif
  }
}

#if DW_NDIM == 3
#define LAUNCH_FILL_P_BC(p)                                          \
  launch_fill_p_bc(p, bc, n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_V_BC()                                                    \
  launch_fill_v_bc(vz, vy, vx, bc, n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_P_BC_FR(p)                                          \
  launch_fill_p_bc(p, bc_fr, n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_V_BC_FR()                                              \
  launch_fill_v_bc(vz, vy, vx, bc_fr, n_shots_h *shot_numel_h, nz_h, ny_h, \
                   nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC(p)                                          \
  launch_adj_fold_p_bc(p, bc, n_shots_h *shot_numel_h, nz_h, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC()                                              \
  launch_adj_fold_v_bc(vz, vy, vx, bc, n_shots_h *shot_numel_h, nz_h, ny_h, \
                       nx_h, stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS()                                           \
  launch_adj_gather_v_guards(vz, vy, vx, p, rmem, m1, c2, bc, n_shots_h, nz_h, \
                             ny_h, nx_h, rdz_h, rdy_h, rdx_h,                  \
                             stream_compute DW_GSLS_NMECH_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS()                                      \
  launch_adj_gather_p_guards(p, vz, buoyancy_z, vy, buoyancy_y, vx,       \
                             buoyancy_x, bc, n_shots_h, nz_h, ny_h, nx_h, \
                             rdz_h, rdy_h, rdx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC_FRX(p)                                        \
  launch_adj_fold_p_bc_frx(p, bc_frx, n_shots_h *shot_numel_h, nz_h, ny_h, \
                           nx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_FRX()                                            \
  launch_adj_fold_v_bc_frx(vz, vy, vx, bc_frx, n_shots_h *shot_numel_h, nz_h, \
                           ny_h, nx_h, stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_FRX()                                    \
  launch_adj_gather_v_guards_frx(vz, vy, vx, p, rmem, m1, c2, bc_frx,       \
                                 n_shots_h, nz_h, ny_h, nx_h, rdz_h, rdy_h, \
                                 rdx_h, stream_compute DW_GSLS_NMECH_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_FRX()                                    \
  launch_adj_gather_p_guards_frx(p, vz, buoyancy_z, vy, buoyancy_y, vx,     \
                                 buoyancy_x, bc_frx, n_shots_h, nz_h, ny_h, \
                                 nx_h, rdz_h, rdy_h, rdx_h, stream_compute)
#elif DW_NDIM == 2
#define LAUNCH_FILL_P_BC(p)                                       \
  launch_fill_p_bc(p, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_V_BC()                                             \
  launch_fill_v_bc(vy, vx, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_P_BC_FR(p)                                       \
  launch_fill_p_bc(p, bc_fr, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_V_BC_FR()                                             \
  launch_fill_v_bc(vy, vx, bc_fr, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                   stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC(p)                                       \
  launch_adj_fold_p_bc(p, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC()                                             \
  launch_adj_fold_v_bc(vy, vx, bc, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS()                                          \
  launch_adj_gather_v_guards(vy, vx, p, rmem, m1, c2, bc, n_shots_h, 1, ny_h, \
                             nx_h, rdy_h, rdx_h,                              \
                             stream_compute DW_GSLS_NMECH_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS()                                           \
  launch_adj_gather_p_guards(p, vy, buoyancy_y, vx, buoyancy_x, bc, n_shots_h, \
                             1, ny_h, nx_h, rdy_h, rdx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC_FRX(p)                                           \
  launch_adj_fold_p_bc_frx(p, bc_frx, n_shots_h *shot_numel_h, 1, ny_h, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_FRX()                                           \
  launch_adj_fold_v_bc_frx(vy, vx, bc_frx, n_shots_h *shot_numel_h, 1, ny_h, \
                           nx_h, stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_FRX()                                     \
  launch_adj_gather_v_guards_frx(vy, vx, p, rmem, m1, c2, bc_frx, n_shots_h, \
                                 1, ny_h, nx_h, rdy_h, rdx_h,                \
                                 stream_compute DW_GSLS_NMECH_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_FRX()                                    \
  launch_adj_gather_p_guards_frx(p, vy, buoyancy_y, vx, buoyancy_x, bc_frx, \
                                 n_shots_h, 1, ny_h, nx_h, rdy_h, rdx_h,    \
                                 stream_compute)
#else
#define LAUNCH_FILL_P_BC(p) \
  launch_fill_p_bc(p, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, stream_compute)
#define LAUNCH_FILL_V_BC() \
  launch_fill_v_bc(vx, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, stream_compute)
#define LAUNCH_FILL_P_BC_FR(p)                                    \
  launch_fill_p_bc(p, bc_fr, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                   stream_compute)
#define LAUNCH_FILL_V_BC_FR()                                      \
  launch_fill_v_bc(vx, bc_fr, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                   stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC(p)                                    \
  launch_adj_fold_p_bc(p, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC()                                      \
  launch_adj_fold_v_bc(vx, bc, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                       stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS()                                         \
  launch_adj_gather_v_guards(vx, p, rmem, m1, c2, bc, n_shots_h, 1, 1, nx_h, \
                             rdx_h, stream_compute DW_GSLS_NMECH_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS()                                       \
  launch_adj_gather_p_guards(p, vx, buoyancy_x, bc, n_shots_h, 1, 1, nx_h, \
                             rdx_h, stream_compute)
#define LAUNCH_ADJ_FOLD_P_BC_FRX(p)                                        \
  launch_adj_fold_p_bc_frx(p, bc_frx, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_FOLD_V_BC_FRX()                                          \
  launch_adj_fold_v_bc_frx(vx, bc_frx, n_shots_h *shot_numel_h, 1, 1, nx_h, \
                           stream_compute)
#define LAUNCH_ADJ_GATHER_V_GUARDS_FRX()                                       \
  launch_adj_gather_v_guards_frx(vx, p, rmem, m1, c2, bc_frx, n_shots_h, 1, 1, \
                                 nx_h, rdx_h,                                  \
                                 stream_compute DW_GSLS_NMECH_ARG)
#define LAUNCH_ADJ_GATHER_P_GUARDS_FRX()                                     \
  launch_adj_gather_p_guards_frx(p, vx, buoyancy_x, bc_frx, n_shots_h, 1, 1, \
                                 nx_h, rdx_h, stream_compute)
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

#ifdef DW_GSLS
/* Per-mechanism shot combine for the n_mech-times-bigger grad_c2/grad_ar:
 * launched once per mechanism with the grad/grad_shot pointers offset to
 * that mechanism's slice; the per-shot scratch layout is
 * (n_shots, n_mech, spatial), so the shot stride is n_mech * shot_numel. */
__launch_bounds__(128) __global__
    void combine_grad_gsls(DW_DTYPE *__restrict const grad,
                           DW_DTYPE const *__restrict const grad_shot,
                           int64_t const shot_stride) {
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
      grad[i] += grad_shot[shot_idx * shot_stride + i];
    }
  }
}
#endif /* DW_GSLS */

__launch_bounds__(128) __global__
    void forward_kernel_v(DW_DTYPE const *__restrict__ const p,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const vy,
#endif
                          DW_DTYPE *__restrict__ const vx,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const psi_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const psi_y,
#endif
                          DW_DTYPE *__restrict__ const psi_x,
#if DW_NDIM >= 3
                          DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                          DW_DTYPE const *__restrict__ const buoyancy_x,
#if DW_NDIM >= 3
                          DW_DTYPE *__restrict__ const bz_grad_store_1,
#endif
#if DW_NDIM >= 2
                          DW_DTYPE *__restrict__ const by_grad_store_1,
#endif
                          DW_DTYPE *__restrict__ const bx_grad_store_1,
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
                          bool const b_requires_grad) {

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

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        FUNC(forward_vz_cell)(p, vz, psi_z, bz_grad_store_1, buoyancy_z_shot[j],
                              azh, bzh, dt, b_requires_grad, i, ny, nx, z,
                              pml_z, rdz);
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        FUNC(forward_vy_cell)(p, vy, psi_y, by_grad_store_1, buoyancy_y_shot[j],
                              ayh, byh, dt, b_requires_grad, i,
#if DW_NDIM >= 3
                              ny,
#endif
                              nx, y, pml_y, rdy);
      }
#endif

      if (x < nx - FD_PAD) {
        bool const pml_x = x < pml_x0h || x >= pml_x1h;
        FUNC(forward_vx_cell)(p, vx, psi_x, bx_grad_store_1, buoyancy_x_shot[j],
                              axh, bxh, dt, b_requires_grad, i,
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

__launch_bounds__(128) __global__ void forward_kernel_p(
    DW_DTYPE *__restrict__ const p, DW_DTYPE *__restrict__ const rmem,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const vy,
#endif
    DW_DTYPE const *__restrict__ const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict__ const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict__ const phi_y,
#endif
    DW_DTYPE *__restrict__ const phi_x, DW_DTYPE const *__restrict__ const m1,
    DW_DTYPE const *__restrict__ const c2,
    DW_DTYPE const *__restrict__ const ar,
    DW_DTYPE *__restrict__ const m1_grad_store_1,
    DW_DTYPE *__restrict__ const c2_grad_store_1,
    DW_DTYPE *__restrict__ const ar_grad_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const az,
    DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const ay,
    DW_DTYPE const *__restrict__ const by,
#endif
    DW_DTYPE const *__restrict__ const ax,
    DW_DTYPE const *__restrict__ const bx, bool const kmod_requires_grad,
    bool const ar_requires_grad DW_GSLS_NMECH_PARAM) {

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

      DW_DTYPE const *__restrict__ const m1_shot =
          kmod_batched ? m1 + shot_idx * shot_numel : m1;
      DW_DTYPE const *__restrict__ const c2_shot =
          kmod_batched ? c2 + shot_idx * shot_numel : c2;
      DW_DTYPE const *__restrict__ const ar_shot =
          ar_batched ? ar + shot_idx * shot_numel : ar;

#if DW_NDIM >= 3
      bool const pml_z = z < pml_z0 || z >= pml_z1;
#endif
#if DW_NDIM >= 2
      bool const pml_y = y < pml_y0 || y >= pml_y1;
#endif
      bool const pml_x = x < pml_x0 || x >= pml_x1;

      FUNC(forward_p_cell)(
#if DW_NDIM >= 3
          vz,
#endif
#if DW_NDIM >= 2
          vy,
#endif
          vx, p, rmem,
#if DW_NDIM >= 3
          phi_z,
#endif
#if DW_NDIM >= 2
          phi_y,
#endif
          phi_x, m1_shot[j],
#ifdef DW_GSLS
          c2_shot, ar_shot, j, n_mech, n_shots * shot_numel, shot_numel,
          shot_idx * n_mech * shot_numel + j, shot_numel, m1_grad_store_1,
          c2_grad_store_1, ar_grad_store_1,
#else
        c2_shot[j], ar_shot[j], m1_grad_store_1, c2_grad_store_1,
        ar_grad_store_1,
#endif
          dt, kmod_requires_grad, ar_requires_grad, i,
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

__launch_bounds__(128) __global__
    void backward_kernel_v(DW_DTYPE const *__restrict__ const p,
                           DW_DTYPE const *__restrict__ const rmem,
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
                           DW_DTYPE *__restrict__ const psi_x,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const psi_zn,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const psi_yn,
#endif
                           DW_DTYPE *__restrict__ const psi_xn,
                           DW_DTYPE const *__restrict__ const m1,
                           DW_DTYPE const *__restrict__ const c2,
#if DW_NDIM >= 3
                           DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                           DW_DTYPE const *__restrict__ const buoyancy_x,
#if DW_NDIM >= 3
                           DW_DTYPE *__restrict__ const grad_bz_shot,
                           DW_DTYPE const *__restrict__ const bz_grad_store_1,
                           DW_DTYPE const *__restrict__ const azh,
                           DW_DTYPE const *__restrict__ const bz,
#endif
#if DW_NDIM >= 2
                           DW_DTYPE *__restrict__ const grad_by_shot,
                           DW_DTYPE const *__restrict__ const by_grad_store_1,
                           DW_DTYPE const *__restrict__ const ayh,
                           DW_DTYPE const *__restrict__ const by,
#endif
                           DW_DTYPE *__restrict__ const grad_bx_shot,
                           DW_DTYPE const *__restrict__ const bx_grad_store_1,
                           DW_DTYPE const *__restrict__ const axh,
                           DW_DTYPE const *__restrict__ const bx,
                           bool const b_requires_grad DW_GSLS_NMECH_PARAM) {

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

      DW_DTYPE const *__restrict__ const m1_shot =
          kmod_batched ? m1 + shot_idx * shot_numel : m1;
      DW_DTYPE const *__restrict__ const c2_shot =
          kmod_batched ? c2 + shot_idx * shot_numel : c2;

#ifdef DW_GSLS
      int64_t const r_lstride = n_shots * shot_numel;
      int64_t const m_lstride = shot_numel;
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

#if DW_NDIM >= 3
      if (z < nz - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_z_shot =
            b_batched ? buoyancy_z + shot_idx * shot_numel : buoyancy_z;
        bool const pml_z = z < pml_z0h || z >= pml_z1h;
        FUNC(backward_vz_cell)(p, m1_shot, c2_shot, rmem, vz, phi_z, psi_z,
                               psi_zn, buoyancy_z_shot[j], grad_bz_shot,
                               bz_grad_store_1, azh, bz, dt, step_ratio,
                               b_requires_grad, i, j,
#ifdef DW_GSLS
                               n_mech, r_lstride, m_lstride,
#endif
                               ny, nx, z, pml_z, rdz);
      }
#endif

#if DW_NDIM >= 2
      if (y < ny - FD_PAD) {
        DW_DTYPE const *__restrict__ const buoyancy_y_shot =
            b_batched ? buoyancy_y + shot_idx * shot_numel : buoyancy_y;
        bool const pml_y = y < pml_y0h || y >= pml_y1h;
        FUNC(backward_vy_cell)(p, m1_shot, c2_shot, rmem, vy, phi_y, psi_y,
                               psi_yn, buoyancy_y_shot[j], grad_by_shot,
                               by_grad_store_1, ayh, by, dt, step_ratio,
                               b_requires_grad, i, j,
#ifdef DW_GSLS
                               n_mech, r_lstride, m_lstride,
#endif
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
        FUNC(backward_vx_cell)(p, m1_shot, c2_shot, rmem, vx, phi_x, psi_x,
                               psi_xn, buoyancy_x_shot[j], grad_bx_shot,
                               bx_grad_store_1, axh, bx, dt, step_ratio,
                               b_requires_grad, i, j,
#ifdef DW_GSLS
                               n_mech, r_lstride, m_lstride,
#endif
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

__launch_bounds__(128) __global__ void backward_kernel_p(
    DW_DTYPE *__restrict__ const p, DW_DTYPE *__restrict__ const rmem,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const vy,
#endif
    DW_DTYPE const *__restrict__ const vx,
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
    DW_DTYPE *__restrict__ const psi_x, DW_DTYPE const *__restrict__ const m1,
    DW_DTYPE const *__restrict__ const c2,
    DW_DTYPE const *__restrict__ const ar,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
    DW_DTYPE const *__restrict__ const buoyancy_x,
    DW_DTYPE *__restrict__ const grad_m1_shot,
    DW_DTYPE *__restrict__ const grad_c2_shot,
    DW_DTYPE *__restrict__ const grad_ar_shot,
    DW_DTYPE const *__restrict__ const m1_grad_store_1,
    DW_DTYPE const *__restrict__ const c2_grad_store_1,
    DW_DTYPE const *__restrict__ const ar_grad_store_1,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict__ const az,
    DW_DTYPE const *__restrict__ const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict__ const ay,
    DW_DTYPE const *__restrict__ const byh,
#endif
    DW_DTYPE const *__restrict__ const ax,
    DW_DTYPE const *__restrict__ const bxh, bool const kmod_requires_grad,
    bool const ar_requires_grad DW_GSLS_NMECH_PARAM) {

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

      DW_DTYPE const *__restrict__ const m1_shot =
          kmod_batched ? m1 + shot_idx * shot_numel : m1;
      DW_DTYPE const *__restrict__ const c2_shot =
          kmod_batched ? c2 + shot_idx * shot_numel : c2;
      DW_DTYPE const *__restrict__ const ar_shot =
          ar_batched ? ar + shot_idx * shot_numel : ar;

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

      FUNC(backward_p_cell)(
          p, rmem,
#if DW_NDIM >= 3
          vz, phi_z, buoyancy_z_shot, psi_z,
#endif
#if DW_NDIM >= 2
          vy, phi_y, buoyancy_y_shot, psi_y,
#endif
          vx, phi_x, buoyancy_x_shot, psi_x, m1_shot[j],
#ifdef DW_GSLS
          c2_shot, ar_shot, j, n_mech, n_shots * shot_numel, shot_numel,
          shot_idx * n_mech * shot_numel + j, shot_numel, grad_m1_shot,
          grad_c2_shot, grad_ar_shot, m1_grad_store_1, c2_grad_store_1,
          ar_grad_store_1,
#else
        c2_shot[j], ar_shot[j], grad_m1_shot, grad_c2_shot, grad_ar_shot,
        m1_grad_store_1, c2_grad_store_1, ar_grad_store_1,
#endif
          dt, step_ratio, kmod_requires_grad, ar_requires_grad, i, j,
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
    /* z-dimension */
    DW_DTYPE const rdz_h, int64_t const nz_h, int64_t const pml_z0_h,
    int64_t const pml_z1_h,
#endif
#if DW_NDIM >= 2
    /* y-dimension */
    DW_DTYPE const rdy_h, int64_t const ny_h, int64_t const pml_y0_h,
    int64_t const pml_y1_h,
#endif
    /* x-dimension */
    DW_DTYPE const rdx_h, int64_t const nx_h, int64_t const shot_numel_h,
    int64_t const pml_x0_h, int64_t const pml_x1_h,
    /* other */
    DW_DTYPE const dt_h, int64_t const n_shots_h, int64_t const step_ratio_h,
    bool const kmod_batched_h, bool const ar_batched_h,
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
  gpuErrchk(cudaMemcpyToSymbol(kmod_batched, &kmod_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(ar_batched, &ar_batched_h, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(b_batched, &b_batched_h, sizeof(bool)));
  return 0;
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
    int
    FUNC(forward)(DW_DTYPE const *__restrict__ const m1,
                  DW_DTYPE const *__restrict__ const c2,
                  DW_DTYPE const *__restrict__ const ar,
#if DW_NDIM >= 3
                  DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                  DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                  DW_DTYPE const *__restrict__ const buoyancy_x,
                  DW_DTYPE const *__restrict__ const f_p,
#if DW_NDIM >= 3
                  DW_DTYPE const *__restrict__ const f_vz,
#endif
#if DW_NDIM >= 2
                  DW_DTYPE const *__restrict__ const f_vy,
#endif
                  DW_DTYPE const *__restrict__ const f_vx,
                  DW_DTYPE *__restrict__ const p,
                  DW_DTYPE *__restrict__ const rmem,
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
                  DW_DTYPE *__restrict__ const psi_x,
                  DW_DTYPE *__restrict__ const m1_grad_store_1a,
                  DW_DTYPE *__restrict__ const m1_grad_store_1b,
                  void *__restrict__ const m1_grad_store_2,
                  void *__restrict__ const m1_grad_store_3,
                  char const *__restrict__ const
                      *__restrict__ const m1_grad_filenames_ptr,
                  DW_DTYPE *__restrict__ const c2_grad_store_1a,
                  DW_DTYPE *__restrict__ const c2_grad_store_1b,
                  void *__restrict__ const c2_grad_store_2,
                  void *__restrict__ const c2_grad_store_3,
                  char const *__restrict__ const
                      *__restrict__ const c2_grad_filenames_ptr,
                  DW_DTYPE *__restrict__ const ar_grad_store_1a,
                  DW_DTYPE *__restrict__ const ar_grad_store_1b,
                  void *__restrict__ const ar_grad_store_2,
                  void *__restrict__ const ar_grad_store_3,
                  char const *__restrict__ const
                      *__restrict__ const ar_grad_filenames_ptr,
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
                  DW_DTYPE *__restrict__ const r_p,
#if DW_NDIM >= 3
                  DW_DTYPE *__restrict__ const r_vz,
#endif
#if DW_NDIM >= 2
                  DW_DTYPE *__restrict__ const r_vy,
#endif
                  DW_DTYPE *__restrict__ const r_vx,
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
                  int64_t const *__restrict__ const receivers_i_p,
#if DW_NDIM >= 3
                  int64_t const *__restrict__ const receivers_i_vz,
#endif
#if DW_NDIM >= 2
                  int64_t const *__restrict__ const receivers_i_vy,
#endif
                  int64_t const *__restrict__ const receivers_i_vx,
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
                  int64_t const n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
                  int64_t const n_receivers_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
                  int64_t const n_receivers_vy_per_shot_h,
#endif
                  int64_t const n_receivers_vx_per_shot_h,
                  int64_t const step_ratio_h, int64_t const storage_mode,
                  size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
                  bool const kmod_requires_grad, bool const ar_requires_grad,
                  bool const b_requires_grad, bool const kmod_batched_h,
                  bool const ar_batched_h, bool const b_batched_h,
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
                  cudaStream_t stream_compute
#ifdef DW_GSLS
                  ,
                  int64_t const n_mechanisms
#endif
    ) {
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
    if ((n_sources_p_per_shot_h > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot_h > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot_h > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot_h > 0) && sources_i_vx == NULL) return 1;
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
#ifdef DW_GSLS
  int64_t const n_mech = n_mechanisms;
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
        step_ratio_h, kmod_batched_h, ar_batched_h, b_batched_h);
    if (err != 0) return err;
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
      (kmod_requires_grad || ar_requires_grad || b_requires_grad);

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

  OPEN_FILE_WRITE(m1_grad, kmod_requires_grad)
  OPEN_FILE_WRITE(c2_grad, kmod_requires_grad)
  OPEN_FILE_WRITE(ar_grad, ar_requires_grad)
#if DW_NDIM >= 3
  OPEN_FILE_WRITE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_WRITE(by_grad, b_requires_grad)
#endif
  OPEN_FILE_WRITE(bx_grad, b_requires_grad)

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

    SETUP_STORE_SAVE(m1_grad, kmod_requires_grad)
    SETUP_STORE_SAVE(c2_grad, kmod_requires_grad)
    SETUP_STORE_SAVE(ar_grad, ar_requires_grad)
#if DW_NDIM >= 3
    SETUP_STORE_SAVE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_SAVE(by_grad, b_requires_grad)
#endif
    SETUP_STORE_SAVE(bx_grad, b_requires_grad)

#ifdef DW_GSLS
    /* The ar (r_old) checkpoint store is n_mech-times bigger, per-(step,
     * shot) layout (n_mech, spatial); the generic SETUP_STORE_SAVE-computed
     * ar_grad_store_1_t/store_3_t offsets do not apply (compression is
     * rejected upstream for this variant). Double-buffering swaps between
     * the same (n_shots, n_mech, spatial)-sized 1a/1b scratches. */
    (void)ar_grad_store_1_t;
    DW_DTYPE *__restrict ar_gsls_store_1_t =
        ar_grad_store_1a != NULL
            ? (ar_grad_store_1a +
               ((storage_mode == STORAGE_DEVICE && !storage_compression)
                    ? (t / step_ratio_h) * n_shots_h * n_mech * shot_numel_h
                    : 0))
            : NULL;
    void *__restrict const ar_gsls_store_3_t =
        ar_grad_store_3 != NULL
            ? ((uint8_t *)ar_grad_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) * n_shots_h *
                          (int64_t)(storage_compression
                                        ? shot_bytes_comp * (size_t)n_mech
                                        : shot_bytes_uncomp * (size_t)n_mech)
                    : 0))
            : NULL;
    if (ar_requires_grad && use_double_buffering && (step_idx % 2 != 0)) {
      ar_gsls_store_1_t = ar_grad_store_1b;
    }
#endif

    cudaEvent_t event_compute_done = event_compute_done_a;
    event_storage_done = event_storage_done_a;
    if (use_double_buffering) {
      if (step_idx % 2 != 0) {
        event_compute_done = event_compute_done_b;
        event_storage_done = event_storage_done_b;
      }
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_p + t * n_shots_h * n_receivers_p_per_shot_h, p, receivers_i_p,
          n_receivers_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    if (n_receivers_vz_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_vz + t * n_shots_h * n_receivers_vz_per_shot_h, vz, receivers_i_vz,
          n_receivers_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_receivers_vy_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_vy + t * n_shots_h * n_receivers_vy_per_shot_h, vy, receivers_i_vy,
          n_receivers_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_receivers_vx_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_receivers_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_receivers(gridx_receivers, gridy_receivers, 1);
      record_receivers<<<dimGrid_receivers, dimBlock_src_rcv, 0,
                         stream_compute>>>(
          r_vx + t * n_shots_h * n_receivers_vx_per_shot_h, vx, receivers_i_vx,
          n_receivers_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    // Refill the pressure and SLS memory-variable guard rings (read by the
    // velocity update); no-op for all-PML.
    if (has_bc) {
      LAUNCH_FILL_P_BC(p);
#ifdef DW_GSLS
      for (int64_t l = 0; l < n_mech; ++l) {
        LAUNCH_FILL_P_BC(rmem + l * n_shots_h * shot_numel_h);
      }
#else
      LAUNCH_FILL_P_BC(rmem);
#endif
    }

    forward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        bz_grad_store_1_t,
#endif
#if DW_NDIM >= 2
        by_grad_store_1_t,
#endif
        bx_grad_store_1_t,
#if DW_NDIM >= 3
        azh, bzh,
#endif
#if DW_NDIM >= 2
        ayh, byh,
#endif
        axh, bxh, b_requires_grad && ((t % step_ratio_h) == 0));
    CHECK_KERNEL_ERROR

#if DW_NDIM >= 3
    if (n_sources_vz_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vz, f_vz + t * n_shots_h * n_sources_vz_per_shot_h, sources_i_vz,
          n_sources_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_sources_vy_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vy, f_vy + t * n_shots_h * n_sources_vy_per_shot_h, sources_i_vy,
          n_sources_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_sources_vx_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          vx, f_vx + t * n_shots_h * n_sources_vx_per_shot_h, sources_i_vx,
          n_sources_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    // Refill the boundary-normal velocity guard ring (read by the pressure
    // update); no-op for all-PML.
    if (has_bc) LAUNCH_FILL_V_BC();

    forward_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, rmem,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x, m1, c2, ar, m1_grad_store_1_t, c2_grad_store_1_t,
#ifdef DW_GSLS
        ar_gsls_store_1_t,
#else
        ar_grad_store_1_t,
#endif
#if DW_NDIM >= 3
        az, bz,
#endif
#if DW_NDIM >= 2
        ay, by,
#endif
        ax, bx, m1_grad_cond, ar_grad_cond DW_GSLS_NMECH_ARG);
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

    SAVE_SNAPSHOT(m1_grad)
    SAVE_SNAPSHOT(c2_grad)
#ifdef DW_GSLS
    /* One opaque n_mech-times-bigger blob per (step, shot); byte copy /
     * seek only (compression rejected upstream). */
    if (ar_grad_cond) {
      if (STORAGE_FUNC(save_snapshot_gpu)(
              ar_gsls_store_1_t, ar_grad_store_2_t, ar_gsls_store_3_t,
              fp_ar_grad, storage_mode, storage_compression, step_idx,
              shot_bytes_uncomp * (size_t)n_mech,
              shot_bytes_comp * (size_t)n_mech, n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }
#else
    SAVE_SNAPSHOT(ar_grad)
#endif
#if DW_NDIM >= 3
    SAVE_SNAPSHOT(bz_grad)
#endif
#if DW_NDIM >= 2
    SAVE_SNAPSHOT(by_grad)
#endif
    SAVE_SNAPSHOT(bx_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      dim3 const dimGrid_sources(gridx_sources, gridy_sources, 1);
      add_sources<<<dimGrid_sources, dimBlock_src_rcv, 0, stream_compute>>>(
          p, f_p + t * n_shots_h * n_sources_p_per_shot_h, sources_i_p,
          n_sources_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }
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
    FUNC(backward)(DW_DTYPE const *__restrict__ const m1,
                   DW_DTYPE const *__restrict__ const c2,
                   DW_DTYPE const *__restrict__ const ar,
#if DW_NDIM >= 3
                   DW_DTYPE const *__restrict__ const buoyancy_z,
#endif
#if DW_NDIM >= 2
                   DW_DTYPE const *__restrict__ const buoyancy_y,
#endif
                   DW_DTYPE const *__restrict__ const buoyancy_x,
                   DW_DTYPE const *__restrict__ const grad_r_p,
#if DW_NDIM >= 3
                   DW_DTYPE const *__restrict__ const grad_r_vz,
#endif
#if DW_NDIM >= 2
                   DW_DTYPE const *__restrict__ const grad_r_vy,
#endif
                   DW_DTYPE const *__restrict__ const grad_r_vx,
                   DW_DTYPE *__restrict__ const p,
                   DW_DTYPE *__restrict__ const rmem,
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
                   DW_DTYPE *__restrict__ psi_xn,

                   DW_DTYPE *__restrict__ const m1_grad_store_1a,
                   DW_DTYPE *__restrict__ const m1_grad_store_1b,
                   void *__restrict__ const m1_grad_store_2,
                   void *__restrict__ const m1_grad_store_3,
                   char const *__restrict__ const
                       *__restrict__ const m1_grad_filenames_ptr,
                   DW_DTYPE *__restrict__ const c2_grad_store_1a,
                   DW_DTYPE *__restrict__ const c2_grad_store_1b,
                   void *__restrict__ const c2_grad_store_2,
                   void *__restrict__ const c2_grad_store_3,
                   char const *__restrict__ const
                       *__restrict__ const c2_grad_filenames_ptr,
                   DW_DTYPE *__restrict__ const ar_grad_store_1a,
                   DW_DTYPE *__restrict__ const ar_grad_store_1b,
                   void *__restrict__ const ar_grad_store_2,
                   void *__restrict__ const ar_grad_store_3,
                   char const *__restrict__ const
                       *__restrict__ const ar_grad_filenames_ptr,
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
                   DW_DTYPE *__restrict__ const grad_f_p,
#if DW_NDIM >= 3
                   DW_DTYPE *__restrict__ const grad_f_vz,
#endif
#if DW_NDIM >= 2
                   DW_DTYPE *__restrict__ const grad_f_vy,
#endif
                   DW_DTYPE *__restrict__ const grad_f_vx,
                   DW_DTYPE *__restrict__ const grad_m1,
                   DW_DTYPE *__restrict__ const grad_c2,
                   DW_DTYPE *__restrict__ const grad_ar,
#if DW_NDIM >= 3
                   DW_DTYPE *__restrict__ const grad_bz,
#endif
#if DW_NDIM >= 2
                   DW_DTYPE *__restrict__ const grad_by,
#endif
                   DW_DTYPE *__restrict__ const grad_bx,
                   DW_DTYPE *__restrict__ const grad_m1_shot,
                   DW_DTYPE *__restrict__ const grad_c2_shot,
                   DW_DTYPE *__restrict__ const grad_ar_shot,
#if DW_NDIM >= 3
                   DW_DTYPE *__restrict__ const grad_bz_shot,
#endif
#if DW_NDIM >= 2
                   DW_DTYPE *__restrict__ const grad_by_shot,
#endif
                   DW_DTYPE *__restrict__ const grad_bx_shot,
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
                   int64_t const *__restrict__ const receivers_i_p,
#if DW_NDIM >= 3
                   int64_t const *__restrict__ const receivers_i_vz,
#endif
#if DW_NDIM >= 2
                   int64_t const *__restrict__ const receivers_i_vy,
#endif
                   int64_t const *__restrict__ const receivers_i_vx,
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
                   int64_t const n_receivers_p_per_shot_h,
#if DW_NDIM >= 3
                   int64_t const n_receivers_vz_per_shot_h,
#endif
#if DW_NDIM >= 2
                   int64_t const n_receivers_vy_per_shot_h,
#endif
                   int64_t const n_receivers_vx_per_shot_h,
                   int64_t const step_ratio_h, int64_t const storage_mode,
                   size_t const shot_bytes_uncomp, size_t const shot_bytes_comp,
                   bool const kmod_requires_grad, bool const ar_requires_grad,
                   bool const b_requires_grad, bool const kmod_batched_h,
                   bool const ar_batched_h, bool const b_batched_h,
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
                   cudaStream_t stream_compute
#ifdef DW_GSLS
                   ,
                   int64_t const n_mechanisms
#endif
    ) {
  if (nt < 0 || n_shots_h < 0) return 1;
  if (nt == 0 || n_shots_h == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots_h > 0) {
    if (step_ratio_h <= 0) return 1;
    if ((kmod_requires_grad) && (grad_m1 == NULL || grad_c2 == NULL)) return 1;
    if ((ar_requires_grad) && (grad_ar == NULL)) return 1;
#if DW_NDIM >= 3
    if ((b_requires_grad) && (grad_bz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((b_requires_grad) && (grad_by == NULL)) return 1;
#endif
    if ((b_requires_grad) && (grad_bx == NULL)) return 1;
    if ((n_receivers_p_per_shot_h > 0) && (receivers_i_p == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_per_shot_h > 0) && (receivers_i_vz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_per_shot_h > 0) && (receivers_i_vy == NULL)) return 1;
#endif
    if ((n_receivers_vx_per_shot_h > 0) && (receivers_i_vx == NULL)) return 1;
    if ((n_sources_p_per_shot_h > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot_h > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot_h > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot_h > 0) && sources_i_vx == NULL) return 1;
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

  int64_t t;

#if DW_NDIM == 3
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
#elif DW_NDIM == 2
  int64_t const shot_numel_h = ny_h * nx_h;
#else
  int64_t const shot_numel_h = nx_h;
#endif
#ifdef DW_GSLS
  int64_t const n_mech = n_mechanisms;
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
   * viscoacoustic_kernels.h); free/rigid faces keep the forward re-fill.
   * bc_fr is bc with the periodic axes suppressed. */
#define DW_AXIS_IS_REFILL(lo, hi) ((lo) == 1 && (hi) == 2)
  int64_t const bc_fr[6] = {DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[0] : 0,
                            DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[1] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[2] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[3] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[4] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[5] : 0};
  int const has_fr =
      bc_fr[0] || bc_fr[1] || bc_fr[2] || bc_fr[3] || bc_fr[4] || bc_fr[5];
  int64_t const bc_frx[6] = {
      bc[0] == 3 || bc_fr[0] ? 0 : bc[0], bc[1] == 3 || bc_fr[1] ? 0 : bc[1],
      bc[2] == 3 || bc_fr[2] ? 0 : bc[2], bc[3] == 3 || bc_fr[3] ? 0 : bc[3],
      bc[4] == 3 || bc_fr[4] ? 0 : bc[4], bc[5] == 3 || bc_fr[5] ? 0 : bc[5]};
  int const has_frx = bc_frx[0] || bc_frx[1] || bc_frx[2] || bc_frx[3] ||
                      bc_frx[4] || bc_frx[5];
#if DW_NDIM == 3
  int const has_per = bc[0] == 3 || bc[2] == 3 || bc[4] == 3;
#elif DW_NDIM == 2
  int const has_per = bc[2] == 3 || bc[4] == 3;
#else
  int const has_per = bc[4] == 3;
#endif
  /* Any fold-handled axis (periodic or exact-fr) defers the buoyancy
   * gradient to the post-fold deferred kernel. */
  int const has_defer = has_per || has_frx;
  (void)has_bc;

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
        step_ratio_h, kmod_batched_h, ar_batched_h, b_batched_h);
    if (err != 0) return err;
  }

  bool const use_double_buffering =
      ((storage_mode == STORAGE_DEVICE && storage_compression) ||
       storage_mode == STORAGE_CPU) &&
      (kmod_requires_grad || ar_requires_grad || b_requires_grad);

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

  OPEN_FILE_READ(m1_grad, kmod_requires_grad)
  OPEN_FILE_READ(c2_grad, kmod_requires_grad)
  OPEN_FILE_READ(ar_grad, ar_requires_grad)
#if DW_NDIM >= 3
  OPEN_FILE_READ(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
  OPEN_FILE_READ(by_grad, b_requires_grad)
#endif
  OPEN_FILE_READ(bx_grad, b_requires_grad)

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

    SETUP_STORE_LOAD(m1_grad, kmod_requires_grad)
    SETUP_STORE_LOAD(c2_grad, kmod_requires_grad)
#ifdef DW_GSLS
    /* The generic macro's pointer arithmetic and load sizes do not apply to
     * the n_mech-times-bigger ar (r_old) store, so it is invoked with the
     * load suppressed (declares the store_2/store_3 names only; its
     * always-false ar_grad_cond is replaced by ar_gsls_cond below,
     * including at the backward_kernel_p launch) and the gsls-layout
     * pointer and byte-scaled load are done explicitly. */
    SETUP_STORE_LOAD(ar_grad, false)
    (void)ar_grad_store_1_t;
    DW_DTYPE *__restrict ar_gsls_store_1_t =
        ar_grad_store_1a != NULL
            ? (ar_grad_store_1a +
               ((storage_mode == STORAGE_DEVICE && !storage_compression)
                    ? (t / step_ratio_h) * n_shots_h * n_mech * shot_numel_h
                    : 0))
            : NULL;
    void *__restrict const ar_gsls_store_3_t =
        ar_grad_store_3 != NULL
            ? ((uint8_t *)ar_grad_store_3 +
               (storage_mode == STORAGE_CPU
                    ? (t / step_ratio_h) * n_shots_h *
                          (int64_t)(storage_compression
                                        ? shot_bytes_comp * (size_t)n_mech
                                        : shot_bytes_uncomp * (size_t)n_mech)
                    : 0))
            : NULL;
    bool const ar_gsls_cond = ar_requires_grad && ((t % step_ratio_h) == 0);
    if (ar_gsls_cond) {
      if (use_double_buffering && step_idx % 2 != 0) {
        ar_gsls_store_1_t = ar_grad_store_1b;
      }
      if (STORAGE_FUNC(load_snapshot_gpu)(
              (void *)ar_gsls_store_1_t, ar_grad_store_2_t, ar_gsls_store_3_t,
              fp_ar_grad, storage_mode, storage_compression, step_idx,
              shot_bytes_uncomp * (size_t)n_mech,
              shot_bytes_comp * (size_t)n_mech, n_shots_h, DIM_ARGS,
              use_double_buffering ? stream_storage : stream_compute) != 0)
        return 1;
    }
#else
    SETUP_STORE_LOAD(ar_grad, ar_requires_grad)
#endif
#if DW_NDIM >= 3
    SETUP_STORE_LOAD(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    SETUP_STORE_LOAD(by_grad, b_requires_grad)
#endif
    SETUP_STORE_LOAD(bx_grad, b_requires_grad)

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_storage_done, stream_storage));
      gpuErrchk(cudaStreamWaitEvent(stream_compute, event_storage_done, 0));
    }

    // Periodic faces: fold the adjoint pressure's and adjoint memory
    // variable's guard cells back into the interior nodes the forward fill
    // copied them from (the exact transpose of the fill). The fold modifies
    // interior seam nodes, so it must precede the source-gradient read
    // below.
    if (has_per) {
      LAUNCH_ADJ_FOLD_P_BC(p);
      CHECK_KERNEL_ERROR
#ifdef DW_GSLS
      for (int64_t l = 0; l < n_mech; ++l) {
        LAUNCH_ADJ_FOLD_P_BC(rmem + l * n_shots_h * shot_numel_h);
        CHECK_KERNEL_ERROR
      }
#else
      LAUNCH_ADJ_FOLD_P_BC(rmem);
      CHECK_KERNEL_ERROR
#endif
    }
    if (has_frx) {
      LAUNCH_ADJ_FOLD_P_BC_FRX(p);
      CHECK_KERNEL_ERROR
#ifdef DW_GSLS
      for (int64_t l = 0; l < n_mech; ++l) {
        LAUNCH_ADJ_FOLD_P_BC_FRX(rmem + l * n_shots_h * shot_numel_h);
        CHECK_KERNEL_ERROR
      }
#else
      LAUNCH_ADJ_FOLD_P_BC_FRX(rmem);
      CHECK_KERNEL_ERROR
#endif
    }

    if (n_sources_p_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_p + t * n_shots_h * n_sources_p_per_shot_h, p, sources_i_p,
          n_sources_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    // Refill the adjoint pressure/memory-variable guard rings (read by the
    // adjoint velocity update via B_EFF) on the free/rigid faces; no-op for
    // all-PML.
    if (has_fr) {
      LAUNCH_FILL_P_BC_FR(p);
#ifdef DW_GSLS
      for (int64_t l = 0; l < n_mech; ++l) {
        LAUNCH_FILL_P_BC_FR(rmem + l * n_shots_h * shot_numel_h);
      }
#else
      LAUNCH_FILL_P_BC_FR(rmem);
#endif
    }

    backward_kernel_v<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, rmem,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x,
#if DW_NDIM >= 3
        psi_zn,
#endif
#if DW_NDIM >= 2
        psi_yn,
#endif
        psi_xn, m1, c2,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x,
#if DW_NDIM >= 3
        grad_bz_shot, bz_grad_store_1_t, azh, bz,
#endif
#if DW_NDIM >= 2
        grad_by_shot, by_grad_store_1_t, ayh, by,
#endif
        grad_bx_shot, bx_grad_store_1_t, axh, bx,
        b_requires_grad && ((t % step_ratio_h) == 0) &&
            !has_defer DW_GSLS_NMECH_ARG);
    CHECK_KERNEL_ERROR

    // Periodic faces: accumulate the adjoint-stencil contributions that fall
    // on the adjoint velocity's guard cells (and the slaved extra high
    // interior node), then fold them back into the interior (the exact
    // transpose of the forward velocity fill). The fold modifies interior
    // seam nodes, so it must precede the source-gradient reads below.
    if (has_per) {
      LAUNCH_ADJ_GATHER_V_GUARDS();
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC();
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
      LAUNCH_ADJ_GATHER_V_GUARDS_FRX();
      CHECK_KERNEL_ERROR
      LAUNCH_ADJ_FOLD_V_BC_FRX();
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM >= 3
    if (n_sources_vz_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_vz + t * n_shots_h * n_sources_vz_per_shot_h, vz, sources_i_vz,
          n_sources_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_sources_vy_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_vy + t * n_shots_h * n_sources_vy_per_shot_h, vy, sources_i_vy,
          n_sources_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_sources_vx_per_shot_h > 0) {
      unsigned int const gridx_receivers =
          ceil_div(n_sources_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_receivers =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      record_receivers<<<dim3(gridx_receivers, gridy_receivers, 1),
                         dimBlock_src_rcv, 0, stream_compute>>>(
          grad_f_vx + t * n_shots_h * n_sources_vx_per_shot_h, vx, sources_i_vx,
          n_sources_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

    // Refill the adjoint boundary-normal velocity guard ring (read by the
    // adjoint pressure update via P_ADJOINT_Z/Y/X) on the free/rigid faces;
    // no-op for all-PML.
    if (has_fr) LAUNCH_FILL_V_BC_FR();

    // Deferred buoyancy-gradient accumulation (see
    // backward_grad_b_deferred_kernel): the periodic axes' accumulation must
    // use the post-fold adjoint velocity.
    if (has_defer && b_requires_grad && ((t % step_ratio_h) == 0)) {
      backward_grad_b_deferred_kernel<<<dimGrid, dimBlock, 0, stream_compute>>>(
#if DW_NDIM >= 3
          vz, grad_bz_shot, bz_grad_store_1_t,
#endif
#if DW_NDIM >= 2
          vy, grad_by_shot, by_grad_store_1_t,
#endif
          vx, grad_bx_shot, bx_grad_store_1_t);
      CHECK_KERNEL_ERROR
    }

    backward_kernel_p<<<dimGrid, dimBlock, 0, stream_compute>>>(
        p, rmem,
#if DW_NDIM >= 3
        vz,
#endif
#if DW_NDIM >= 2
        vy,
#endif
        vx,
#if DW_NDIM >= 3
        phi_z,
#endif
#if DW_NDIM >= 2
        phi_y,
#endif
        phi_x,
#if DW_NDIM >= 3
        psi_z,
#endif
#if DW_NDIM >= 2
        psi_y,
#endif
        psi_x, m1, c2, ar,
#if DW_NDIM >= 3
        buoyancy_z,
#endif
#if DW_NDIM >= 2
        buoyancy_y,
#endif
        buoyancy_x, grad_m1_shot, grad_c2_shot, grad_ar_shot, m1_grad_store_1_t,
        c2_grad_store_1_t,
#ifdef DW_GSLS
        ar_gsls_store_1_t,
#else
        ar_grad_store_1_t,
#endif
#if DW_NDIM >= 3
        az, bzh,
#endif
#if DW_NDIM >= 2
        ay, byh,
#endif
        ax, bxh, m1_grad_cond,
#ifdef DW_GSLS
        ar_gsls_cond
#else
        ar_grad_cond
#endif
            DW_GSLS_NMECH_ARG);
    CHECK_KERNEL_ERROR

    // Periodic faces: accumulate the adjoint-stencil contributions that fall
    // on the adjoint pressure's guard cells (folded into the interior at the
    // top of the next iteration).
    if (has_per) {
      LAUNCH_ADJ_GATHER_P_GUARDS();
      CHECK_KERNEL_ERROR
    }
    if (has_frx) {
      LAUNCH_ADJ_GATHER_P_GUARDS_FRX();
      CHECK_KERNEL_ERROR
    }

    if (use_double_buffering) {
      gpuErrchk(cudaEventRecord(event_compute_done, stream_compute));
    }

    // Inject adjoint source (grad_r) into P
    if (n_receivers_p_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_p_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          p, grad_r_p + t * n_shots_h * n_receivers_p_per_shot_h, receivers_i_p,
          n_receivers_p_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#if DW_NDIM >= 3
    if (n_receivers_vz_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_vz_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          vz, grad_r_vz + t * n_shots_h * n_receivers_vz_per_shot_h,
          receivers_i_vz, n_receivers_vz_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
#if DW_NDIM >= 2
    if (n_receivers_vy_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_vy_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          vy, grad_r_vy + t * n_shots_h * n_receivers_vy_per_shot_h,
          receivers_i_vy, n_receivers_vy_per_shot_h);
      CHECK_KERNEL_ERROR
    }
#endif
    if (n_receivers_vx_per_shot_h > 0) {
      unsigned int const gridx_sources =
          ceil_div(n_receivers_vx_per_shot_h, dimBlock_src_rcv.x);
      unsigned int const gridy_sources =
          ceil_div(n_shots_h, dimBlock_src_rcv.y);
      add_sources<<<dim3(gridx_sources, gridy_sources, 1), dimBlock_src_rcv, 0,
                    stream_compute>>>(
          vx, grad_r_vx + t * n_shots_h * n_receivers_vx_per_shot_h,
          receivers_i_vx, n_receivers_vx_per_shot_h);
      CHECK_KERNEL_ERROR
    }

#if DW_NDIM >= 3
    std::swap(psi_z, psi_zn);
#endif
#if DW_NDIM >= 2
    std::swap(psi_y, psi_yn);
#endif
    std::swap(psi_x, psi_xn);
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

  if (kmod_requires_grad && !kmod_batched_h && n_shots_h > 1) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_m1, grad_m1_shot);
    CHECK_KERNEL_ERROR
#ifdef DW_GSLS
    for (int64_t l = 0; l < n_mech; ++l) {
      combine_grad_gsls<<<dimGrid_combine, dimBlock_combine, 0,
                          stream_compute>>>(grad_c2 + l * shot_numel_h,
                                            grad_c2_shot + l * shot_numel_h,
                                            n_mech * shot_numel_h);
      CHECK_KERNEL_ERROR
    }
#else
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_c2, grad_c2_shot);
    CHECK_KERNEL_ERROR
#endif
  }
  if (ar_requires_grad && !ar_batched_h && n_shots_h > 1) {
#ifdef DW_GSLS
    for (int64_t l = 0; l < n_mech; ++l) {
      combine_grad_gsls<<<dimGrid_combine, dimBlock_combine, 0,
                          stream_compute>>>(grad_ar + l * shot_numel_h,
                                            grad_ar_shot + l * shot_numel_h,
                                            n_mech * shot_numel_h);
      CHECK_KERNEL_ERROR
    }
#else
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_ar, grad_ar_shot);
    CHECK_KERNEL_ERROR
#endif
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

  int status = config_guard.finish();
  if (file_group.close_all() != 0 && status == 0) status = 1;
  return status;
}

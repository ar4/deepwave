/*
 * Viscoacoustic wave equation propagator (variable density, single-mechanism
 * standard-linear-solid attenuation, staggered grid).
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

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "common_cpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, ndim, accuracy, dtype, device) \
  viscoacoustic_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
/* Multi-mechanism GSLS (DW_GSLS) build variant: the exported forward/backward
 * symbols gain a ``_gsls`` suffix and one trailing int64 parameter, the
 * runtime mechanism count ``n_mechanisms``. The memory variable ``rmem``
 * then points at ``n_mechanisms`` stacked (L, n_shots, spatial) fields
 * through the same single pointer, ``c2``/``ar`` at per-mechanism
 * (L, spatial) model fields, and ``ar``'s per-time-step checkpoint store is
 * ``n_mechanisms``-times bigger. The default (no macro) build is completely
 * untouched -- byte-identical to before this variant existed. */
#ifdef DW_GSLS
#define VISCO_VARIANT_I(name) name##_gsls
#define VISCO_VARIANT(name) VISCO_VARIANT_I(name)
#else
#define VISCO_VARIANT(name) name
#endif
#define FUNC(name) \
  CAT(VISCO_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

/* Threading of the GSLS per-mechanism geometry (mechanism count and the
 * rmem/model mechanism strides) through the guard-fill applier functions to
 * the per-line kernels in viscoacoustic_kernels.h. Expansion sites must have
 * ``n_mech``/``r_lstride``/``m_lstride`` in scope. */
#ifdef DW_GSLS
#define DW_GSLS_EXTRA_PARAMS \
  , int64_t const n_mech, int64_t const r_lstride, int64_t const m_lstride
#define DW_GSLS_EXTRA_ARGS , n_mech, r_lstride, m_lstride
#else
#define DW_GSLS_EXTRA_PARAMS
#define DW_GSLS_EXTRA_ARGS
#endif

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* ND_INDEX, M1/C2/BUOYANCY_Z/Y/X, P/RMEM/VZ/VY/VX, PHI_Z/Y/X/PSI_Z/Y/X, B_EFF,
 * the VZ_ADJOINT/VY_ADJOINT/VX_ADJOINT/P_ADJOINT_Z/Y/X (+_PML) family, and the
 * FUNC(forward_vz_cell)/.../FUNC(backward_p_cell) per-grid-point stencil
 * update functions live in viscoacoustic_kernels.h, shared verbatim with
 * viscoacoustic.cu (see that header for the full explanation). p_shot/
 * rmem_shot/vz_shot/.../psi_z_t/.../m1_shot/c2_shot/ar_shot/buoyancy_z_shot/
 * ... below are this file's names for the (per-shot-offset) arrays/values
 * passed as the p/rmem/vz/.../psi_z/.../m1_shot/c2_shot/ar_shot/
 * buoyancy_z_shot/... parameters of those functions. */
#include "viscoacoustic_kernels.h"

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

/* First-class reflecting/periodic boundary faces on the staggered viscoacoustic
 * grid (see docs/boundaries.rst). Ported verbatim from
 * acoustic.c's FUNC(fill_p_bc_line)/FUNC(fill_v_bc_line)/FUNC(fill_p_bc)/
 * FUNC(fill_v_bc) -- the pressure/velocity image rule is identical (both
 * propagators share the same staggered pressure-velocity layout and the
 * velocity equation is byte-identical). ``bc`` codes: 0=pml (untouched),
 * 1=free (pressure release), 2=rigid (hard wall), 3=periodic.
 *
 * The SLS memory variable ``r`` uses the SAME fill (FUNC(fill_p_bc) is called
 * on ``rmem_shot`` below, wherever it is called on ``p_shot``) rather than a
 * separately-derived rule. This is not a new physics decision: per
 * ROBERTSSON_FS_SUMMARY.md Sec 1.3 point 1, the memory variables that
 * accompany a stress component which must vanish at the surface are grouped
 * with -- and imaged the same way as -- that stress component ("the
 * components that need to be defined above the surface for the FD stencil are
 * sigma_zz and sigma_xz (and r_zz, r_xz) ... imaged as ODD ... about the
 * surface"), and point 4 confirms the memory variable never needs a stencil
 * of its OWN (no spatial derivative of a memory variable appears in the
 * update equations) -- it is simply carried along with the field it
 * accompanies. In viscoacoustic's forward kernel this makes filling
 * ``rmem``'s guard ring a pure no-op in effect (FUNC(forward_p_cell) reads
 * ``rmem[i]`` only at the centre point, never at a neighbour offset -- see
 * viscoacoustic_kernels.h); the fill is still applied there for symmetry and
 * so the returned/continued wavefield stays boundary-consistent. It is NOT a
 * no-op in the backward (adjoint) kernel: FUNC(backward_vz_cell)/vy_cell/
 * vx_cell's B_EFF macro (viscoacoustic_kernels.h) combines P(dz,dy,dx) with
 * RMEM(dz,dy,dx) pointwise before differencing, i.e. it reads ``rmem``'s
 * guard ring at exactly the same offsets the forward velocity update reads
 * ``p``'s -- because the discrete adjoint of the coupled pressure/memory
 * update distributes both the pressure-adjoint and the memory-variable-
 * adjoint through the same DIFFZH1/DIFFYH1/DIFFXH1 stencil that reads the
 * pressure-grid field. So the same image rule that keeps ``p``'s adjoint
 * correct on a reflecting face (see acoustic.c's "Adjoint" note in
 * docs/boundaries.rst: the mirror-filled operator is self-adjoint, so
 * reapplying the same fill to the adjoint field supplies the correct
 * transpose) is exactly what ``rmem``'s adjoint (Rbar) needs too -- hence "the
 * SAME image rule as pressure, not a separate derivation", as required. */

/* FUNC(fill_p_bc_line) (also used, unchanged, to refill the SLS
 * memory-variable ``rmem`` guard ring, per the note above) and
 * FUNC(fill_v_bc_line) live in line_bc.h, shared verbatim across every
 * propagator on this staggered layout -- see that header for the full
 * explanation. */
#include "line_bc.h"

/* The whole-face boundary appliers (FUNC(fill_p_bc) through
 * FUNC(adj_gather_p_guards_frx)) and the FILL_/ADJ_ dispatch macros that
 * drive them live in staggered_bc.h, shared verbatim with acoustic.c -- see
 * that header for the face-enumeration rules and the include contract. The
 * appliers are propagator-independent because the fills themselves are:
 * FUNC(fill_p_bc) is applied to the SLS memory variable exactly as it is to
 * the pressure (per the image-rule note above), and the velocity equation is
 * acoustic's.
 *
 * The one propagator-specific part is the payload of the adjoint-velocity
 * guard gather. Where acoustic's per-line kernel differences dt*K*Pbar, this
 * propagator's differences B_EFF = dt*(m1 - c2/2)*Pbar + c2*Rbar (see the
 * ADJOINT section of the header comment above and viscoacoustic_kernels.h),
 * so the payload is (rmem, m1, c2) rather than a single bulk modulus, and the
 * multi-mechanism GSLS build additionally threads its mechanism geometry
 * through as the trailing extras. */
#define DW_VGUARD_MODEL_PARAMS                  \
  DW_DTYPE const *__restrict const rmem,        \
      DW_DTYPE const *__restrict const m1_shot, \
      DW_DTYPE const *__restrict const c2_shot
#define DW_VGUARD_MODEL_ARGS rmem, m1_shot, c2_shot
#define DW_VGUARD_MODEL_SHOT_ARGS rmem_shot, m1_shot, c2_shot
#define DW_VGUARD_EXTRA_PARAMS DW_GSLS_EXTRA_PARAMS
#define DW_VGUARD_EXTRA_ARGS DW_GSLS_EXTRA_ARGS
#include "staggered_bc.h"

/* Close one gradient-checkpoint file if it was opened (both FUNC(forward) and
 * FUNC(backward) reach their storage_cleanup label on the success path as well
 * as on an I/O error). ``fp_<name>`` must be in scope. */
#define CLOSE_FILE(name) \
  if (dw_close_file(&fp_##name) != 0) storage_err = 1;

/* The trailing "void *unused" parameter is dead: it is part of the frozen
 * ctypes ABI every propagator's forward/backward entry point shares (see
 * tests/test_abi_stability.py) and must keep its slot. */
#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int
FUNC(forward)(
    DW_DTYPE const *__restrict const m1, DW_DTYPE const *__restrict const c2,
    DW_DTYPE const *__restrict const ar,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
    DW_DTYPE const *__restrict const f_p,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const f_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const f_vy,
#endif
    DW_DTYPE const *__restrict const f_vx, DW_DTYPE *__restrict const p,
    DW_DTYPE *__restrict const rmem,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const phi_y,
#endif
    DW_DTYPE *__restrict const phi_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const psi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const psi_y,
#endif
    DW_DTYPE *__restrict const psi_x,
    DW_DTYPE *__restrict const m1_grad_store_1,
    DW_DTYPE *__restrict const m1_grad_store_1b,
    void *__restrict const m1_grad_store_2,
    void *__restrict const m1_grad_store_3,
    char const *__restrict const *__restrict const m1_grad_filenames_ptr,
    DW_DTYPE *__restrict const c2_grad_store_1,
    DW_DTYPE *__restrict const c2_grad_store_1b,
    void *__restrict const c2_grad_store_2,
    void *__restrict const c2_grad_store_3,
    char const *__restrict const *__restrict const c2_grad_filenames_ptr,
    DW_DTYPE *__restrict const ar_grad_store_1,
    DW_DTYPE *__restrict const ar_grad_store_1b,
    void *__restrict const ar_grad_store_2,
    void *__restrict const ar_grad_store_3,
    char const *__restrict const *__restrict const ar_grad_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const bz_grad_store_1,
    DW_DTYPE *__restrict const bz_grad_store_1b,
    void *__restrict const bz_grad_store_2,
    void *__restrict const bz_grad_store_3,
    char const *__restrict const *__restrict const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const by_grad_store_1,
    DW_DTYPE *__restrict const by_grad_store_1b,
    void *__restrict const by_grad_store_2,
    void *__restrict const by_grad_store_3,
    char const *__restrict const *__restrict const by_grad_filenames_ptr,
#endif
    DW_DTYPE *__restrict const bx_grad_store_1,
    DW_DTYPE *__restrict const bx_grad_store_1b,
    void *__restrict const bx_grad_store_2,
    void *__restrict const bx_grad_store_3,
    char const *__restrict const *__restrict const bx_grad_filenames_ptr,
    DW_DTYPE *__restrict const r_p,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const r_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const r_vy,
#endif
    DW_DTYPE *__restrict const r_vx,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bxh,
    int64_t const *__restrict const sources_i_p,
#if DW_NDIM >= 3
    int64_t const *__restrict const sources_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const *__restrict const sources_i_vy,
#endif
    int64_t const *__restrict const sources_i_vx,
    int64_t const *__restrict const receivers_i_p,
#if DW_NDIM >= 3
    int64_t const *__restrict const receivers_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const *__restrict const receivers_i_vy,
#endif
    int64_t const *__restrict const receivers_i_vx,
#if DW_NDIM >= 3
    DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy,
#endif
    DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
    int64_t const n_shots,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_sources_vz_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_vy_per_shot,
#endif
    int64_t const n_sources_vx_per_shot, int64_t const n_receivers_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_per_shot,
#endif
    int64_t const n_receivers_vx_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const kmod_requires_grad,
    bool const ar_requires_grad, bool const b_requires_grad,
    bool const kmod_batched, bool const ar_batched, bool const b_batched,
    bool const storage_compression, int64_t const start_t,
#if DW_NDIM >= 3
    int64_t const pml_z0,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y0,
#endif
    int64_t const pml_x0,
#if DW_NDIM >= 3
    int64_t const pml_z1,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y1,
#endif
    int64_t const pml_x1,
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
    int64_t const bc_x1, int64_t const n_threads, void *unused
#ifdef DW_GSLS
    ,
    int64_t const n_mechanisms
#endif
) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_i_p == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_per_shot > 0) && (receivers_i_vz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_per_shot > 0) && (receivers_i_vy == NULL)) return 1;
#endif
    if ((n_receivers_vx_per_shot > 0) && (receivers_i_vx == NULL)) return 1;
    if ((n_sources_p_per_shot > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot > 0) && sources_i_vx == NULL) return 1;
  }

#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
#if DW_NDIM >= 2
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
#endif
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};

#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
#else
  int64_t const n_grid_points = nx;
#endif

#ifdef DW_GSLS
  /* Per-mechanism strides: rmem is the (n_mechanisms, n_shots, spatial)
   * stacked buffer; c2/ar are (n_mechanisms, spatial) model fields
   * (per-shot-batched models are rejected upstream for this variant); the
   * ar checkpoint store and grad_c2/grad_ar are (.., n_mechanisms, spatial)
   * with mechanism stride n_grid_points. */
  int64_t const n_mech = n_mechanisms;
  int64_t const r_lstride = n_shots * n_grid_points;
  int64_t const m_lstride = n_grid_points;
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

  int64_t shot;
  /* Intra-shot spatial parallelism: shots run serially and each time step's
   * velocity and pressure/memory-variable stencils are spread across threads
   * (see the omp parallel regions below), so a single shot saturates all
   * cores. */
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const si = shot * n_grid_points;
    int64_t const src_i_p = shot * n_sources_p_per_shot;
#if DW_NDIM >= 3
    int64_t const src_i_vz = shot * n_sources_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const src_i_vy = shot * n_sources_vy_per_shot;
#endif
    int64_t const src_i_vx = shot * n_sources_vx_per_shot;
    int64_t const rcv_i_p = shot * n_receivers_p_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz = shot * n_receivers_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy = shot * n_receivers_vy_per_shot;
#endif
    int64_t const rcv_i_vx = shot * n_receivers_vx_per_shot;

    DW_DTYPE const *__restrict const m1_shot = kmod_batched ? m1 + si : m1;
    DW_DTYPE const *__restrict const c2_shot = kmod_batched ? c2 + si : c2;
    DW_DTYPE const *__restrict const ar_shot = ar_batched ? ar + si : ar;
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z_shot =
        b_batched ? buoyancy_z + si : buoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y_shot =
        b_batched ? buoyancy_y + si : buoyancy_y;
#endif
    DW_DTYPE const *__restrict const buoyancy_x_shot =
        b_batched ? buoyancy_x + si : buoyancy_x;

    DW_DTYPE *__restrict const p_shot = p + si;
    DW_DTYPE *__restrict const rmem_shot = rmem + si;
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz_shot = vz + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy_shot = vy + si;
#endif
    DW_DTYPE *__restrict const vx_shot = vx + si;

#if DW_NDIM >= 3
    DW_DTYPE *__restrict const phi_z_shot = phi_z + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const phi_y_shot = phi_y + si;
#endif
    DW_DTYPE *__restrict const phi_x_shot = phi_x + si;

#if DW_NDIM >= 3
    DW_DTYPE *__restrict const psi_z_t = psi_z + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const psi_y_t = psi_y + si;
#endif
    DW_DTYPE *__restrict const psi_x_t = psi_x + si;

    int64_t t;

    int storage_err = 0;
    FILE *fp_m1_grad = NULL;
    FILE *fp_c2_grad = NULL;
    FILE *fp_ar_grad = NULL;
#if DW_NDIM >= 3
    FILE *fp_bz_grad = NULL;
#endif
#if DW_NDIM >= 2
    FILE *fp_by_grad = NULL;
#endif
    FILE *fp_bx_grad = NULL;

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
      /* The ar (r_old) checkpoint store is n_mechanisms-times bigger, with
       * per-(step, shot) layout (n_mechanisms, spatial): shot stride
       * n_mechanisms*n_grid_points, per-step stride
       * n_shots*n_mechanisms*n_grid_points (device-uncompressed history);
       * for the scratch modes only the shot offset applies. Replaces the
       * generic SETUP_STORE_SAVE-computed ar_grad_store_1_t (still declared
       * above, unused here). Compression is rejected upstream for this
       * variant. */
      (void)ar_grad_store_1_t;
      DW_DTYPE *__restrict const ar_gsls_store_1_t =
          ar_grad_store_1 != NULL
              ? (ar_grad_store_1 + si * n_mech +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? (t / step_ratio) * n_shots * n_mech * n_grid_points
                      : 0))
              : NULL;
#endif

      bool const kmod_requires_grad_t =
          kmod_requires_grad && ((t % step_ratio) == 0);
      bool const ar_requires_grad_t =
          ar_requires_grad && ((t % step_ratio) == 0);
      bool const b_requires_grad_t = b_requires_grad && ((t % step_ratio) == 0);

      // Refill the pressure and SLS memory-variable guard rings (the pressure
      // ring is read by the velocity update); this also holds free-surface
      // pressure/memory-variable nodes to zero before recording (no-op when
      // every face is PML). See the FUNC(fill_p_bc_line) comment above for why
      // rmem uses the same fill as p (per mechanism under DW_GSLS).
      if (has_bc) {
        FILL_P_BC(p_shot);
#ifdef DW_GSLS
        {
          int64_t l;
          for (l = 0; l < n_mech; ++l) FILL_P_BC(rmem_shot + l * r_lstride);
        }
#else
        FILL_P_BC(rmem_shot);
#endif
      }

      // Record Receivers
      if (n_receivers_p_per_shot > 0) {
        record_from_wavefield(
            p_shot, receivers_i_p + rcv_i_p,
            r_p + t * n_shots * n_receivers_p_per_shot + rcv_i_p,
            n_receivers_p_per_shot);
      }
#if DW_NDIM >= 3
      if (n_receivers_vz_per_shot > 0) {
        record_from_wavefield(
            vz_shot, receivers_i_vz + rcv_i_vz,
            r_vz + t * n_shots * n_receivers_vz_per_shot + rcv_i_vz,
            n_receivers_vz_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_vy_per_shot > 0) {
        record_from_wavefield(
            vy_shot, receivers_i_vy + rcv_i_vy,
            r_vy + t * n_shots * n_receivers_vy_per_shot + rcv_i_vy,
            n_receivers_vy_per_shot);
      }
#endif
      if (n_receivers_vx_per_shot > 0) {
        record_from_wavefield(
            vx_shot, receivers_i_vx + rcv_i_vx,
            r_vx + t * n_shots * n_receivers_vx_per_shot + rcv_i_vx,
            n_receivers_vx_per_shot);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
#pragma GCC unroll 3
        for (pml_z = 0; pml_z < DW_PML_REGIONS; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < DW_PML_REGIONS; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < DW_PML_REGIONS; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

          // Update Velocities (V)
          // v = v - dt * B * (grad(p) + psi)

#if DW_NDIM >= 3
#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
              for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_vz_cell)(
                        p_shot, vz_shot, psi_z_t, bz_grad_store_1_t,
                        buoyancy_z_shot[i], azh, bzh, dt, b_requires_grad_t, i,
                        ny, nx, z, pml_z != DW_PML_INTERIOR_REGION, rdz);
                  }
#endif

#if DW_NDIM >= 2
#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_vy_cell)(
                        p_shot, vy_shot, psi_y_t, by_grad_store_1_t,
                        buoyancy_y_shot[i], ayh, byh, dt, b_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx, y, pml_y != DW_PML_INTERIOR_REGION, rdy);
                  }
#endif

#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_vx_cell)(
                        p_shot, vx_shot, psi_x_t, bx_grad_store_1_t,
                        buoyancy_x_shot[i], axh, bxh, dt, b_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
                        x, pml_x != DW_PML_INTERIOR_REGION, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

#if DW_NDIM >= 3
      if (n_sources_vz_per_shot > 0) {
        add_to_wavefield(vz_shot, sources_i_vz + src_i_vz,
                         f_vz + t * n_shots * n_sources_vz_per_shot + src_i_vz,
                         n_sources_vz_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_vy_per_shot > 0) {
        add_to_wavefield(vy_shot, sources_i_vy + src_i_vy,
                         f_vy + t * n_shots * n_sources_vy_per_shot + src_i_vy,
                         n_sources_vy_per_shot);
      }
#endif
      if (n_sources_vx_per_shot > 0) {
        add_to_wavefield(vx_shot, sources_i_vx + src_i_vx,
                         f_vx + t * n_shots * n_sources_vx_per_shot + src_i_vx,
                         n_sources_vx_per_shot);
      }

      // Refill the boundary-normal velocity guard ring (read by the pressure
      // update); holds rigid-wall velocity nodes to zero (no-op for all-PML).
      if (has_bc) FILL_V_BC();

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
#pragma GCC unroll 3
        for (pml_z = 0; pml_z < DW_PML_REGIONS; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < DW_PML_REGIONS; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < DW_PML_REGIONS; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

          // Update Pressure (P) and SLS memory variable (r).
          // theta  = div(v) + phi (PML-augmented divergence)
          // r_new  = ar * r - c2 * theta                 (Crank-Nicolson)
          // p     -= dt*m1*theta + (dt/2)*(r_old + r_new)

#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(forward_p_cell)(
#if DW_NDIM >= 3
                        vz_shot,
#endif
#if DW_NDIM >= 2
                        vy_shot,
#endif
                        vx_shot, p_shot, rmem_shot,
#if DW_NDIM >= 3
                        phi_z_shot,
#endif
#if DW_NDIM >= 2
                        phi_y_shot,
#endif
                        phi_x_shot, m1_shot[i],
#ifdef DW_GSLS
                        c2_shot, ar_shot, i, n_mech, r_lstride, m_lstride, i,
                        n_grid_points, m1_grad_store_1_t, c2_grad_store_1_t,
                        ar_gsls_store_1_t,
#else
                c2_shot[i], ar_shot[i], m1_grad_store_1_t, c2_grad_store_1_t,
                ar_grad_store_1_t,
#endif
                        dt, kmod_requires_grad_t, ar_requires_grad_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != DW_PML_INTERIOR_REGION, az, bz, rdz,
#endif
#if DW_NDIM >= 2
                        y, pml_y != DW_PML_INTERIOR_REGION, ay, by, rdy,
#endif
                        x, pml_x != DW_PML_INTERIOR_REGION, ax, bx, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Inject Sources
      if (n_sources_p_per_shot > 0) {
        add_to_wavefield(p_shot, sources_i_p + src_i_p,
                         f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
                         n_sources_p_per_shot);
      }

      /* Checkpoint this step's stored forward quantities for the backward
       * pass (decimated to every step_ratio'th step by grad_cond). Defined
       * here, next to its only uses, and undefined again below the time
       * loop. */
#define SAVE_SNAPSHOT(name, grad_cond)                                         \
  if (grad_cond) {                                                             \
    int64_t const step_idx = t / step_ratio;                                   \
    if (STORAGE_FUNC(save_snapshot_cpu)(                                       \
            name##_store_1_t, name##_store_2_t, fp_##name, storage_mode,       \
            storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp, \
            DIM_ARGS) != 0) {                                                  \
      storage_err = 1;                                                         \
      goto storage_cleanup;                                                    \
    }                                                                          \
  }

      SAVE_SNAPSHOT(m1_grad, kmod_requires_grad_t)
      SAVE_SNAPSHOT(c2_grad, kmod_requires_grad_t)
#ifdef DW_GSLS
      /* The ar snapshot is one opaque n_mechanisms-times-bigger blob per
       * (step, shot) (compression rejected upstream, so the save is a plain
       * byte copy / seek-and-write). */
      if (ar_requires_grad_t) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(save_snapshot_cpu)(
                ar_gsls_store_1_t, ar_grad_store_2_t, fp_ar_grad, storage_mode,
                storage_compression, step_idx,
                shot_bytes_uncomp * (size_t)n_mech,
                shot_bytes_comp * (size_t)n_mech, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
      }
#else
      SAVE_SNAPSHOT(ar_grad, ar_requires_grad_t)
#endif
#if DW_NDIM >= 3
      SAVE_SNAPSHOT(bz_grad, b_requires_grad_t)
#endif
#if DW_NDIM >= 2
      SAVE_SNAPSHOT(by_grad, b_requires_grad_t)
#endif
      SAVE_SNAPSHOT(bx_grad, b_requires_grad_t)
    }

#undef SAVE_SNAPSHOT

  storage_cleanup:
    CLOSE_FILE(m1_grad)
    CLOSE_FILE(c2_grad)
    CLOSE_FILE(ar_grad)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad)
#endif
    CLOSE_FILE(bx_grad)
    if (storage_err) return 1;
  }
  return 0;
}

/* The trailing "void *unused" parameter is dead: it is part of the frozen
 * ctypes ABI every propagator's forward/backward entry point shares (see
 * tests/test_abi_stability.py) and must keep its slot. */
#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int
FUNC(backward)(
    DW_DTYPE const *__restrict const m1, DW_DTYPE const *__restrict const c2,
    DW_DTYPE const *__restrict const ar,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y,
#endif
    DW_DTYPE const *__restrict const buoyancy_x,
    DW_DTYPE const *__restrict const grad_r_p,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const grad_r_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const grad_r_vy,
#endif
    DW_DTYPE const *__restrict const grad_r_vx, DW_DTYPE *__restrict const p,
    DW_DTYPE *__restrict const rmem,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const phi_y,
#endif
    DW_DTYPE *__restrict const phi_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const psi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const psi_y,
#endif
    DW_DTYPE *__restrict const psi_x,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const psi_zn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const psi_yn,
#endif
    DW_DTYPE *__restrict const psi_xn,

    DW_DTYPE *__restrict const m1_grad_store_1,
    DW_DTYPE *__restrict const m1_grad_store_1b,
    void *__restrict const m1_grad_store_2,
    void *__restrict const m1_grad_store_3,
    char const *__restrict const *__restrict const m1_grad_filenames_ptr,
    DW_DTYPE *__restrict const c2_grad_store_1,
    DW_DTYPE *__restrict const c2_grad_store_1b,
    void *__restrict const c2_grad_store_2,
    void *__restrict const c2_grad_store_3,
    char const *__restrict const *__restrict const c2_grad_filenames_ptr,
    DW_DTYPE *__restrict const ar_grad_store_1,
    DW_DTYPE *__restrict const ar_grad_store_1b,
    void *__restrict const ar_grad_store_2,
    void *__restrict const ar_grad_store_3,
    char const *__restrict const *__restrict const ar_grad_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const bz_grad_store_1,
    DW_DTYPE *__restrict const bz_grad_store_1b,
    void *__restrict const bz_grad_store_2,
    void *__restrict const bz_grad_store_3,
    char const *__restrict const *__restrict const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const by_grad_store_1,
    DW_DTYPE *__restrict const by_grad_store_1b,
    void *__restrict const by_grad_store_2,
    void *__restrict const by_grad_store_3,
    char const *__restrict const *__restrict const by_grad_filenames_ptr,
#endif
    DW_DTYPE *__restrict const bx_grad_store_1,
    DW_DTYPE *__restrict const bx_grad_store_1b,
    void *__restrict const bx_grad_store_2,
    void *__restrict const bx_grad_store_3,
    char const *__restrict const *__restrict const bx_grad_filenames_ptr,
    DW_DTYPE *__restrict const grad_f_p,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_f_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_f_vy,
#endif
    DW_DTYPE *__restrict const grad_f_vx, DW_DTYPE *__restrict const grad_m1,
    DW_DTYPE *__restrict const grad_c2, DW_DTYPE *__restrict const grad_ar,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_by,
#endif
    DW_DTYPE *__restrict const grad_bx,
    DW_DTYPE *__restrict const grad_m1_thread,
    DW_DTYPE *__restrict const grad_c2_thread,
    DW_DTYPE *__restrict const grad_ar_thread,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_bz_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_by_thread,
#endif
    DW_DTYPE *__restrict const grad_bx_thread,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const *__restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const by,
    DW_DTYPE const *__restrict const ayh, DW_DTYPE const *__restrict const byh,
#endif
    DW_DTYPE const *__restrict const ax, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const *__restrict const bxh,
    int64_t const *__restrict const sources_i_p,
#if DW_NDIM >= 3
    int64_t const *__restrict const sources_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const *__restrict const sources_i_vy,
#endif
    int64_t const *__restrict const sources_i_vx,
    int64_t const *__restrict const receivers_i_p,
#if DW_NDIM >= 3
    int64_t const *__restrict const receivers_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const *__restrict const receivers_i_vy,
#endif
    int64_t const *__restrict const receivers_i_vx,
#if DW_NDIM >= 3
    DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy,
#endif
    DW_DTYPE const rdx, DW_DTYPE const dt, int64_t const nt,
    int64_t const n_shots,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx, int64_t const n_sources_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_sources_vz_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_vy_per_shot,
#endif
    int64_t const n_sources_vx_per_shot, int64_t const n_receivers_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_per_shot,
#endif
    int64_t const n_receivers_vx_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const kmod_requires_grad,
    bool const ar_requires_grad, bool const b_requires_grad,
    bool const kmod_batched, bool const ar_batched, bool const b_batched,
    bool const storage_compression, int64_t const start_t,
#if DW_NDIM >= 3
    int64_t const pml_z0,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y0,
#endif
    int64_t const pml_x0,
#if DW_NDIM >= 3
    int64_t const pml_z1,
#endif
#if DW_NDIM >= 2
    int64_t const pml_y1,
#endif
    int64_t const pml_x1,
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
    int64_t const bc_x1, int64_t const n_threads, void *unused
#ifdef DW_GSLS
    ,
    int64_t const n_mechanisms
#endif
) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((kmod_requires_grad) && (grad_m1 == NULL || grad_c2 == NULL)) return 1;
    if ((ar_requires_grad) && (grad_ar == NULL)) return 1;
#if DW_NDIM >= 3
    if ((b_requires_grad) && (grad_bz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((b_requires_grad) && (grad_by == NULL)) return 1;
#endif
    if ((b_requires_grad) && (grad_bx == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_i_p == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_per_shot > 0) && (receivers_i_vz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_per_shot > 0) && (receivers_i_vy == NULL)) return 1;
#endif
    if ((n_receivers_vx_per_shot > 0) && (receivers_i_vx == NULL)) return 1;
    if ((n_sources_p_per_shot > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot > 0) && sources_i_vx == NULL) return 1;
  }

#if DW_NDIM >= 3
  int64_t const pml_bounds_z[] = {FD_PAD, pml_z0, pml_z1, nz - FD_PAD + 1};
  int64_t const pml_bounds_zh[] = {FD_PAD, pml_z0, MAX(pml_z0, pml_z1 - 1),
                                   nz - FD_PAD};
#endif
#if DW_NDIM >= 2
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0, MAX(pml_y0, pml_y1 - 1),
                                   ny - FD_PAD};
#endif
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0, MAX(pml_x0, pml_x1 - 1),
                                   nx - FD_PAD};

#if DW_NDIM == 3
  int64_t const n_grid_points = nz * ny * nx;
#elif DW_NDIM == 2
  int64_t const n_grid_points = ny * nx;
#else
  int64_t const n_grid_points = nx;
#endif

#ifdef DW_GSLS
  /* Per-mechanism strides -- see FUNC(forward)'s matching comment. */
  int64_t const n_mech = n_mechanisms;
  int64_t const r_lstride = n_shots * n_grid_points;
  int64_t const m_lstride = n_grid_points;
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
   * viscoacoustic_kernels.h). Free/rigid faces: re-applying the forward
   * fill to the adjoint fields realises the transpose (through the
   * mirror-padded models) only for the per-axis (free-low, rigid-high)
   * pairing, which keeps that historical re-fill path (bc_fr); every other
   * free/rigid combination gets the exact fr fold/gather treatment
   * (bc_frx). */
#define DW_AXIS_IS_REFILL(lo, hi) ((lo) == DW_BC_FREE && (hi) == DW_BC_RIGID)
  int64_t const bc_fr[6] = {DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[0] : 0,
                            DW_AXIS_IS_REFILL(bc[0], bc[1]) ? bc[1] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[2] : 0,
                            DW_AXIS_IS_REFILL(bc[2], bc[3]) ? bc[3] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[4] : 0,
                            DW_AXIS_IS_REFILL(bc[4], bc[5]) ? bc[5] : 0};
  int const has_fr =
      bc_fr[0] || bc_fr[1] || bc_fr[2] || bc_fr[3] || bc_fr[4] || bc_fr[5];
  int64_t const bc_frx[6] = {
      bc[0] == DW_BC_PERIODIC || bc_fr[0] ? DW_BC_PML : bc[0],
      bc[1] == DW_BC_PERIODIC || bc_fr[1] ? DW_BC_PML : bc[1],
      bc[2] == DW_BC_PERIODIC || bc_fr[2] ? DW_BC_PML : bc[2],
      bc[3] == DW_BC_PERIODIC || bc_fr[3] ? DW_BC_PML : bc[3],
      bc[4] == DW_BC_PERIODIC || bc_fr[4] ? DW_BC_PML : bc[4],
      bc[5] == DW_BC_PERIODIC || bc_fr[5] ? DW_BC_PML : bc[5]};
  int const has_frx = bc_frx[0] || bc_frx[1] || bc_frx[2] || bc_frx[3] ||
                      bc_frx[4] || bc_frx[5];
#if DW_NDIM >= 3
  int const per_z = bc[0] == DW_BC_PERIODIC;
  /* Axes whose fold writes interior nodes defer the buoyancy-gradient
   * accumulation until after the fold. */
  int const defer_z = per_z || bc_frx[0] || bc_frx[1];
#endif
#if DW_NDIM >= 2
  int const per_y = bc[2] == DW_BC_PERIODIC;
  int const defer_y = per_y || bc_frx[2] || bc_frx[3];
#endif
  int const per_x = bc[4] == DW_BC_PERIODIC;
  int const defer_x = per_x || bc_frx[4] || bc_frx[5];
#if DW_NDIM == 3
  int const has_per = per_z || per_y || per_x;
#elif DW_NDIM == 2
  int const has_per = per_y || per_x;
#else
  int const has_per = per_x;
#endif
  DW_DTYPE const stag_w_arr[FD_PAD] = DW_STAG_TAYLOR_W;
  DW_DTYPE const *const stag_w = stag_w_arr;
  /* has_bc is computed (identically to FUNC(forward)) but unused here: the
     backward pass drives the guard rings through has_fr/has_per/has_frx
     instead, which partition the same faces. */
  (void)has_bc;

  int64_t shot;
  /* Shots run serially; each time step's adjoint velocity and pressure/
   * memory-variable stencils are spread across threads (see the omp parallel
   * regions below). Model gradients accumulate directly into the shared
   * gradient buffers -- each cell is written by exactly one thread, and time
   * steps and shots are barrier-separated, so there is no race. That makes the
   * grad_*_thread per-thread reduction buffers a dead ABI slot; they stay in
   * the signature because the ABI is frozen. */
  (void)grad_m1_thread;
  (void)grad_c2_thread;
  (void)grad_ar_thread;
#if DW_NDIM >= 3
  (void)grad_bz_thread;
#endif
#if DW_NDIM >= 2
  (void)grad_by_thread;
#endif
  (void)grad_bx_thread;
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const si = shot * n_grid_points;
    int64_t const src_i_p = shot * n_sources_p_per_shot;
#if DW_NDIM >= 3
    int64_t const src_i_vz = shot * n_sources_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const src_i_vy = shot * n_sources_vy_per_shot;
#endif
    int64_t const src_i_vx = shot * n_sources_vx_per_shot;
    int64_t const rcv_i_p = shot * n_receivers_p_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz = shot * n_receivers_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy = shot * n_receivers_vy_per_shot;
#endif
    int64_t const rcv_i_vx = shot * n_receivers_vx_per_shot;

    DW_DTYPE const *__restrict const m1_shot = kmod_batched ? m1 + si : m1;
    DW_DTYPE const *__restrict const c2_shot = kmod_batched ? c2 + si : c2;
    DW_DTYPE const *__restrict const ar_shot = ar_batched ? ar + si : ar;
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const buoyancy_z_shot =
        b_batched ? buoyancy_z + si : buoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const buoyancy_y_shot =
        b_batched ? buoyancy_y + si : buoyancy_y;
#endif
    DW_DTYPE const *__restrict const buoyancy_x_shot =
        b_batched ? buoyancy_x + si : buoyancy_x;

    // In backward pass, p is the adjoint pressure field (Pbar), rmem is the
    // adjoint memory variable (Rbar), vz/vy/vx are adjoint velocity fields.
    DW_DTYPE *__restrict const p_shot = p + si;
    DW_DTYPE *__restrict const rmem_shot = rmem + si;
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz_shot = vz + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy_shot = vy + si;
#endif
    DW_DTYPE *__restrict const vx_shot = vx + si;

#if DW_NDIM >= 3
    DW_DTYPE *__restrict const phi_z_shot = phi_z + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const phi_y_shot = phi_y + si;
#endif
    DW_DTYPE *__restrict const phi_x_shot = phi_x + si;

    // Gradients: accumulated directly into the shared buffers (see the
    // race-freedom comment above the shot loop).
    DW_DTYPE *__restrict const grad_m1_shot =
        grad_m1 != NULL ? (grad_m1 + (kmod_batched ? si : 0)) : NULL;
    DW_DTYPE *__restrict const grad_c2_shot =
        grad_c2 != NULL ? (grad_c2 + (kmod_batched ? si : 0)) : NULL;
    DW_DTYPE *__restrict const grad_ar_shot =
        grad_ar != NULL ? (grad_ar + (ar_batched ? si : 0)) : NULL;
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_bz_shot =
        grad_bz != NULL ? (grad_bz + (b_batched ? si : 0)) : NULL;
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_by_shot =
        grad_by != NULL ? (grad_by + (b_batched ? si : 0)) : NULL;
#endif
    DW_DTYPE *__restrict const grad_bx_shot =
        grad_bx != NULL ? (grad_bx + (b_batched ? si : 0)) : NULL;

    int64_t t;

    int storage_err = 0;
    FILE *fp_m1_grad = NULL;
    FILE *fp_c2_grad = NULL;
    FILE *fp_ar_grad = NULL;
#if DW_NDIM >= 3
    FILE *fp_bz_grad = NULL;
#endif
#if DW_NDIM >= 2
    FILE *fp_by_grad = NULL;
#endif
    FILE *fp_bx_grad = NULL;

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
      SETUP_STORE_LOAD(m1_grad, kmod_requires_grad)
      SETUP_STORE_LOAD(c2_grad, kmod_requires_grad)
#ifdef DW_GSLS
      /* The ar (r_old) checkpoint store is n_mechanisms-times bigger (see
       * FUNC(forward)'s matching comment); the generic macro's pointer
       * arithmetic and load sizes do not apply, so it is invoked with the
       * load suppressed (declares the fp/store_2 names only) and the
       * gsls-layout pointer and byte-scaled load are done explicitly. */
      SETUP_STORE_LOAD(ar_grad, false)
      (void)ar_grad_store_1_t;
      DW_DTYPE *__restrict const ar_gsls_store_1_t =
          ar_grad_store_1 != NULL
              ? (ar_grad_store_1 + si * n_mech +
                 ((storage_mode == STORAGE_DEVICE && !storage_compression)
                      ? (t / step_ratio) * n_shots * n_mech * n_grid_points
                      : 0))
              : NULL;
      if (ar_requires_grad && ((t % step_ratio) == 0)) {
        int64_t const step_idx = t / step_ratio;
        if (STORAGE_FUNC(load_snapshot_cpu)(
                (void *)ar_gsls_store_1_t, ar_grad_store_2_t, fp_ar_grad,
                storage_mode, storage_compression, step_idx,
                shot_bytes_uncomp * (size_t)n_mech,
                shot_bytes_comp * (size_t)n_mech, DIM_ARGS) != 0) {
          storage_err = 1;
          goto storage_cleanup;
        }
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

#if DW_NDIM >= 3
      DW_DTYPE *__restrict const psi_z_t =
          (((start_t - 1 - t) & 1) ? psi_zn : psi_z) + si;
      DW_DTYPE *__restrict const psi_zn_t =
          (((start_t - 1 - t) & 1) ? psi_z : psi_zn) + si;
#endif
#if DW_NDIM >= 2
      DW_DTYPE *__restrict const psi_y_t =
          (((start_t - 1 - t) & 1) ? psi_yn : psi_y) + si;
      DW_DTYPE *__restrict const psi_yn_t =
          (((start_t - 1 - t) & 1) ? psi_y : psi_yn) + si;
#endif
      DW_DTYPE *__restrict const psi_x_t =
          (((start_t - 1 - t) & 1) ? psi_xn : psi_x) + si;
      DW_DTYPE *__restrict const psi_xn_t =
          (((start_t - 1 - t) & 1) ? psi_x : psi_xn) + si;

      bool const kmod_requires_grad_t =
          kmod_requires_grad && ((t % step_ratio) == 0);
      bool const ar_requires_grad_t =
          ar_requires_grad && ((t % step_ratio) == 0);
      bool const b_requires_grad_t = b_requires_grad && ((t % step_ratio) == 0);

      // Periodic faces: fold the adjoint pressure's and adjoint memory
      // variable's guard cells back into the interior nodes the forward fill
      // copied them from (the exact transpose of the fill). The fold
      // modifies interior seam nodes, so it must precede the source-gradient
      // read below. Free/rigid faces other than the (free-low, rigid-high)
      // pairing get their own exact fold at the same position.
      if (has_per) {
        ADJ_FOLD_P_BC(p_shot);
#ifdef DW_GSLS
        {
          int64_t l;
          for (l = 0; l < n_mech; ++l) ADJ_FOLD_P_BC(rmem_shot + l * r_lstride);
        }
#else
        ADJ_FOLD_P_BC(rmem_shot);
#endif
      }
      if (has_frx) {
        ADJ_FOLD_P_BC_FRX(p_shot);
#ifdef DW_GSLS
        {
          int64_t l;
          for (l = 0; l < n_mech; ++l)
            ADJ_FOLD_P_BC_FRX(rmem_shot + l * r_lstride);
        }
#else
        ADJ_FOLD_P_BC_FRX(rmem_shot);
#endif
      }

      if (n_sources_p_per_shot > 0) {
        record_from_wavefield(
            p_shot, sources_i_p + src_i_p,
            grad_f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
            n_sources_p_per_shot);
      }

      // Refill the adjoint pressure/memory-variable guard rings (read by the
      // adjoint velocity update via B_EFF -- see viscoacoustic_kernels.h and
      // the FUNC(fill_p_bc_line) comment above) on the free/rigid faces;
      // no-op when every face is PML.
      if (has_fr) {
        FILL_P_BC_FR(p_shot);
#ifdef DW_GSLS
        {
          int64_t l;
          for (l = 0; l < n_mech; ++l) FILL_P_BC_FR(rmem_shot + l * r_lstride);
        }
#else
        FILL_P_BC_FR(rmem_shot);
#endif
      }

      // Update Adjoint Velocities (V)
      // v = v + dt * (grad((1+b)Kp) - grad(b*phi))
      // psi = a*psi - dt*B*a*v

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
#pragma GCC unroll 3
        for (pml_z = 0; pml_z < DW_PML_REGIONS; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < DW_PML_REGIONS; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < DW_PML_REGIONS; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

#if DW_NDIM >= 3
#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
              for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_vz_cell)(
                        p_shot, m1_shot, c2_shot, rmem_shot, vz_shot,
                        phi_z_shot, psi_z_t, psi_zn_t, buoyancy_z_shot[i],
                        grad_bz_shot, bz_grad_store_1_t, azh, bz, dt,
                        step_ratio, b_requires_grad_t && !defer_z, i, i,
#ifdef DW_GSLS
                        n_mech, r_lstride, m_lstride,
#endif
                        ny, nx, z, pml_z != DW_PML_INTERIOR_REGION, rdz);
                  }
#endif

#if DW_NDIM >= 2
#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
                for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1];
                     y++)
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_vy_cell)(
                        p_shot, m1_shot, c2_shot, rmem_shot, vy_shot,
                        phi_y_shot, psi_y_t, psi_yn_t, buoyancy_y_shot[i],
                        grad_by_shot, by_grad_store_1_t, ayh, by, dt,
                        step_ratio, b_requires_grad_t && !defer_y, i, i,
#ifdef DW_GSLS
                        n_mech, r_lstride, m_lstride,
#endif
#if DW_NDIM >= 3
                        ny,
#endif
                        nx, y, pml_y != DW_PML_INTERIOR_REGION, rdy);
                  }
#endif

#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_vx_cell)(
                        p_shot, m1_shot, c2_shot, rmem_shot, vx_shot,
                        phi_x_shot, psi_x_t, psi_xn_t, buoyancy_x_shot[i],
                        grad_bx_shot, bx_grad_store_1_t, axh, bx, dt,
                        step_ratio, b_requires_grad_t && !defer_x, i, i,
#ifdef DW_GSLS
                        n_mech, r_lstride, m_lstride,
#endif
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
                        x, pml_x != DW_PML_INTERIOR_REGION, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Periodic faces: accumulate the adjoint-stencil contributions that
      // fall on the adjoint velocity's guard cells (and the slaved extra
      // high interior node), then fold them back into the interior (the
      // exact transpose of the forward velocity fill). The fold modifies
      // interior seam nodes, so it must precede the source-gradient reads
      // below, and the buoyancy-gradient accumulation for periodic axes
      // (suppressed in the backward_v*_cell calls above) runs only after it.
      if (has_per) {
        ADJ_GATHER_V_GUARDS();
        ADJ_FOLD_V_BC();
      }
      if (has_frx) {
        ADJ_GATHER_V_GUARDS_FRX();
        ADJ_FOLD_V_BC_FRX();
      }

#if DW_NDIM >= 3
      if (n_sources_vz_per_shot > 0) {
        record_from_wavefield(
            vz_shot, sources_i_vz + src_i_vz,
            grad_f_vz + t * n_shots * n_sources_vz_per_shot + src_i_vz,
            n_sources_vz_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_vy_per_shot > 0) {
        record_from_wavefield(
            vy_shot, sources_i_vy + src_i_vy,
            grad_f_vy + t * n_shots * n_sources_vy_per_shot + src_i_vy,
            n_sources_vy_per_shot);
      }
#endif
      if (n_sources_vx_per_shot > 0) {
        record_from_wavefield(
            vx_shot, sources_i_vx + src_i_vx,
            grad_f_vx + t * n_shots * n_sources_vx_per_shot + src_i_vx,
            n_sources_vx_per_shot);
      }

      // Refill the adjoint boundary-normal velocity guard ring (read by the
      // adjoint pressure update via P_ADJOINT_Z/Y/X) on the free/rigid
      // faces; no-op for all-PML.
      if (has_fr) FILL_V_BC_FR();

      // Periodic axes: the buoyancy gradient must use the post-fold adjoint
      // velocity (the fold writes into interior seam nodes), so it is
      // accumulated here instead of inside backward_v*_cell.
      if (b_requires_grad_t && (has_per || has_frx)) {
#if DW_NDIM == 3
        int64_t gz, gy, gx;
        if (defer_z)
          for (gz = FD_PAD; gz < nz - FD_PAD; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                grad_bz_shot[gi] -= dt * vz_shot[gi] * bz_grad_store_1_t[gi] *
                                    (DW_DTYPE)step_ratio;
              }
        if (defer_y)
          for (gz = FD_PAD; gz < nz - FD_PAD + 1; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                grad_by_shot[gi] -= dt * vy_shot[gi] * by_grad_store_1_t[gi] *
                                    (DW_DTYPE)step_ratio;
              }
        if (defer_x)
          for (gz = FD_PAD; gz < nz - FD_PAD + 1; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                grad_bx_shot[gi] -= dt * vx_shot[gi] * bx_grad_store_1_t[gi] *
                                    (DW_DTYPE)step_ratio;
              }
#elif DW_NDIM == 2
        int64_t gy, gx;
        if (defer_y)
          for (gy = FD_PAD; gy < ny - FD_PAD; ++gy)
            for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
              int64_t const gi = gy * nx + gx;
              grad_by_shot[gi] -= dt * vy_shot[gi] * by_grad_store_1_t[gi] *
                                  (DW_DTYPE)step_ratio;
            }
        if (defer_x)
          for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
            for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
              int64_t const gi = gy * nx + gx;
              grad_bx_shot[gi] -= dt * vx_shot[gi] * bx_grad_store_1_t[gi] *
                                  (DW_DTYPE)step_ratio;
            }
#else
        int64_t gx;
        if (defer_x)
          for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
            grad_bx_shot[gx] -=
                dt * vx_shot[gx] * bx_grad_store_1_t[gx] * (DW_DTYPE)step_ratio;
          }
#endif
      }

      // Update Adjoint Pressure (P) and memory-variable adjoint (Rbar).
      // p = p + dt * (div(B(1+b)v) - div(b*psi))
      // phi = a*phi - a*B_eff        (B_eff = dt*(m1-c2/2)*Pbar + c2*Rbar)
      // Rbar_new = ar*Rbar - (dt/2)(1+ar)*Pbar   (SLS-like backward recursion)

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif /* _OPENMP */
      {
        unsigned int const dw_ftz_daz_prev_state = dw_ftz_daz_enter();
#if DW_NDIM >= 3
        int pml_z;
#endif
#if DW_NDIM >= 2
        int pml_y;
#endif
        int pml_x;
#if DW_NDIM >= 3
#pragma GCC unroll 3
        for (pml_z = 0; pml_z < DW_PML_REGIONS; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < DW_PML_REGIONS; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < DW_PML_REGIONS; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

#ifdef _OPENMP
#pragma omp for simd collapse(DW_NDIM) nowait
#else
#pragma omp simd collapse(DW_NDIM)
#endif
#if DW_NDIM >= 3
              for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
                for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                  for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                       x++) {
                    SET_I;
                    FUNC(backward_p_cell)(
                        p_shot, rmem_shot,
#if DW_NDIM >= 3
                        vz_shot, phi_z_shot, buoyancy_z_shot, psi_z_t,
#endif
#if DW_NDIM >= 2
                        vy_shot, phi_y_shot, buoyancy_y_shot, psi_y_t,
#endif
                        vx_shot, phi_x_shot, buoyancy_x_shot, psi_x_t,
                        m1_shot[i],
#ifdef DW_GSLS
                        c2_shot, ar_shot, i, n_mech, r_lstride, m_lstride, i,
                        n_grid_points, grad_m1_shot, grad_c2_shot, grad_ar_shot,
                        m1_grad_store_1_t, c2_grad_store_1_t, ar_gsls_store_1_t,
#else
                c2_shot[i], ar_shot[i], grad_m1_shot, grad_c2_shot,
                grad_ar_shot, m1_grad_store_1_t, c2_grad_store_1_t,
                ar_grad_store_1_t,
#endif
                        dt, step_ratio, kmod_requires_grad_t,
                        ar_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != DW_PML_INTERIOR_REGION, az, bzh, rdz,
#endif
#if DW_NDIM >= 2
                        y, pml_y != DW_PML_INTERIOR_REGION, ay, byh, rdy,
#endif
                        x, pml_x != DW_PML_INTERIOR_REGION, ax, bxh, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Periodic faces: accumulate the adjoint-stencil contributions that
      // fall on the adjoint pressure's guard cells (folded into the interior
      // at the top of the next iteration).
      if (has_per) ADJ_GATHER_P_GUARDS();
      if (has_frx) ADJ_GATHER_P_GUARDS_FRX();

      // Inject adjoint source (grad_r) into P
      if (n_receivers_p_per_shot > 0) {
        add_to_wavefield(
            p_shot, receivers_i_p + rcv_i_p,
            grad_r_p + t * n_shots * n_receivers_p_per_shot + rcv_i_p,
            n_receivers_p_per_shot);
      }
#if DW_NDIM >= 3
      if (n_receivers_vz_per_shot > 0) {
        add_to_wavefield(
            vz_shot, receivers_i_vz + rcv_i_vz,
            grad_r_vz + t * n_shots * n_receivers_vz_per_shot + rcv_i_vz,
            n_receivers_vz_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_vy_per_shot > 0) {
        add_to_wavefield(
            vy_shot, receivers_i_vy + rcv_i_vy,
            grad_r_vy + t * n_shots * n_receivers_vy_per_shot + rcv_i_vy,
            n_receivers_vy_per_shot);
      }
#endif
      if (n_receivers_vx_per_shot > 0) {
        add_to_wavefield(
            vx_shot, receivers_i_vx + rcv_i_vx,
            grad_r_vx + t * n_shots * n_receivers_vx_per_shot + rcv_i_vx,
            n_receivers_vx_per_shot);
      }
    }

  storage_cleanup:
    CLOSE_FILE(m1_grad)
    CLOSE_FILE(c2_grad)
    CLOSE_FILE(ar_grad)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad)
#endif
    CLOSE_FILE(bx_grad)
    if (storage_err) return 1;
  }

  return 0;
}

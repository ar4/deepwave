/*
 * Variable-density acoustic Born (linearised) propagator.
 *
 * Propagates a background velocity-pressure field (p, v_d) and a scattered
 * field (dp, dv_d) simultaneously on a staggered grid, for perturbations of
 * the constitutive models: bulk modulus dK (at the pressure node) and
 * buoyancy dB_d (at the v_d node). The scattered field satisfies the SAME
 * acoustic system with the background (K, B_d), driven by two virtual
 * sources (first-order Born):
 *
 *     scattered velocity:  dv_d -= dt*B_d*(grad_d(dp) + psi_sc_d)
 *                                  + dt*dB_d*term_bg_d
 *     scattered pressure:  dp   -= dt*K*(div(dv) + phi_sc)
 *                                  + dt*dK*div_bg
 *
 * where term_bg_d = grad_d(p) + psi_d  and  div_bg = sum_d(div_d(v) + phi_d)
 * are the background quantities also used for the (unperturbed) model
 * gradient in acoustic.c (bx_grad_store_1 / k_grad_store_1). This is the
 * acoustic analogue of scalar_born.c's 2*v*scatter*dt^2*w_sum coupling.
 *
 * The v/rho -> K,B and dv,drho -> dK,dB chain rules live in differentiable
 * Python (see docs/acoustic_born_derivation.md); this kernel scatters only
 * in (dK, dB).
 *
 * This file currently implements:
 *   forward:      coupled background + scattered forward pass, checkpointing
 *                 the background terms div_bg / term_bg_d whenever the
 *                 scattering models (dK, dB_d) require a gradient.
 *   backward_sc:  scattered-only adjoint (fixed background K, B_d):
 *                 propagates only the scattered adjoint field with the
 *                 ORDINARY (unperturbed) acoustic adjoint recursion --
 *                 exactly acoustic.c's backward() applied to the scattered
 *                 field -- and accumulates grad_dK, grad_dB_d by
 *                 correlating the propagating scattered adjoint field with
 *                 the checkpointed background terms. No coupling term is
 *                 needed here because Born's forward coupling
 *                 (-dt*dK*div_bg, -dt*dB_d*term_bg_d) has no dependence on
 *                 the scattered *state*, so it does not appear in the
 *                 transpose.
 *
 * The full two-field adjoint (`backward`), which additionally propagates a
 * background adjoint field carrying a coupling term read from the scattered
 * adjoint field (needed for gradients w.r.t. the background K/B_d and for
 * the acoustic Hessian-vector product), is NOT YET implemented -- see
 * docs/acoustic_born_derivation.md section 4.3/5 for the derived equations;
 * this is tracked as a follow-up.
 *
 * Symbols: acoustic_born_iso_<ndim>d_<accuracy>_<dtype>_<forward|
 * backward_sc>_<device>.
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
  acoustic_born_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define ACOUSTIC_BORN_VARIANT_I(name) name
#define ACOUSTIC_BORN_VARIANT(name) ACOUSTIC_BORN_VARIANT_I(name)
#define FUNC(name) \
  CAT(ACOUSTIC_BORN_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* ND_INDEX, K/BUOYANCY_Z/Y/X/DK/DBUOYANCY_Z/Y/X, P/VZ/VY/VX/PSC/VZSC/VYSC/
 * VXSC, PHI_Z/Y/X/PSI_Z/Y/X (+_SC scattered twins), the VZ_ADJOINT/.../
 * P_ADJOINT_Z/Y/X (+_PML, +_BG, +_BG_PML) family, and the twelve
 * FUNC(forward_vz_cell)/.../FUNC(backward_p_cell) per-grid-point stencil
 * update functions live in acoustic_born_kernels.h, shared verbatim with
 * acoustic_born.cu (see that header for the full explanation). p_shot/
 * vz_shot/.../psc_shot/.../psi_z_t/.../k_shot/buoyancy_z_shot/... below are
 * this file's names for the (per-shot-offset) arrays/values passed as the
 * p/vz/.../psc/.../psi_z/.../k_shot/buoyancy_z_shot/... parameters of those
 * functions. */
#include "acoustic_born_kernels.h"

/* An axis keeps the historical adjoint re-fill only for the (free-low,
 * rigid-high) face pairing; every other free/rigid combination is
 * fold/gather-handled (see acoustic_born_kernels.h's _fr functions). */
#define DW_AXIS_IS_REFILL(lo, hi) ((lo) == 1 && (hi) == 2)

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

/* FUNC(fill_p_bc_line)/FUNC(fill_v_bc_line) (first-class reflecting/periodic
 * boundary faces on the staggered acoustic Born grid) live in line_bc.h,
 * shared verbatim across every propagator on this staggered layout -- see
 * that header for the full explanation. The eager Python reference
 * (``fill_acoustic_pressure_boundary``/``fill_acoustic_velocity_boundary``
 * in ``_boundaries.py``) is applied independently to the background and
 * scattered fields, and FUNC(fill_p_bc)/FUNC(fill_v_bc) below mirror that by
 * calling the SAME per-line fill on the background field's arrays and,
 * separately, on the scattered field's arrays -- both fields share the
 * acoustic staggered-grid layout, so the same per-field image parity applies
 * to each of them independently (background and scattered are each an
 * ordinary acoustic (p, v_d) field on the same grid; see acoustic_born.py's
 * module docstring). */
#include "line_bc.h"

/* The whole-face boundary appliers (FUNC(fill_p_bc) through
 * FUNC(adj_gather_p_guards_frx)) live in staggered_bc.h, shared verbatim with
 * acoustic.c and viscoacoustic.c -- see that header for the face-enumeration
 * rules and the include contract. The appliers are per-field-set, and this
 * propagator has two field sets, so the FILL_/ADJ_ dispatch macros the header
 * would otherwise supply are declined here (DW_STAGGERED_BC_CUSTOM_DISPATCH)
 * in favour of the set-parameterised ones defined below: each applier is
 * driven once over the background (p, v_d) field and once over the scattered
 * one, and the Born-coupled background payload (K*P_adj + dK*P_adj_sc, and
 * likewise for the buoyancies) is composed as two single-payload gather calls
 * accumulating into the same target.
 *
 * The payload of the adjoint-velocity guard gather is the plain dt*K*Pbar of
 * FUNC(backward_v*_cell), exactly as in acoustic.c, so the bulk modulus of
 * whichever set is being driven is the whole payload and there are no
 * trailing extras. */
#define DW_VGUARD_MODEL_PARAMS DW_DTYPE const* __restrict const k_shot
#define DW_VGUARD_MODEL_ARGS k_shot
#define DW_VGUARD_EXTRA_PARAMS
#define DW_VGUARD_EXTRA_ARGS
#define DW_STAGGERED_BC_CUSTOM_DISPATCH
#include "staggered_bc.h"

#if DW_NDIM == 3
#define FILL_P_BC(p) FUNC(fill_p_bc)(p, bc, nz, ny, nx)
#define FILL_V_BC_BG() \
  FUNC(fill_v_bc)(vz_shot, vy_shot, vx_shot, bc, nz, ny, nx)
#define FILL_V_BC_SC() \
  FUNC(fill_v_bc)(vzsc_shot, vysc_shot, vxsc_shot, bc, nz, ny, nx)
#define FILL_P_BC_FR(p) FUNC(fill_p_bc)(p, bc_fr, nz, ny, nx)
#define FILL_V_BC_BG_FR() \
  FUNC(fill_v_bc)(vz_shot, vy_shot, vx_shot, bc_fr, nz, ny, nx)
#define FILL_V_BC_SC_FR() \
  FUNC(fill_v_bc)(vzsc_shot, vysc_shot, vxsc_shot, bc_fr, nz, ny, nx)
#define ADJ_FOLD_P_BC(p) FUNC(adj_fold_p_bc)(p, bc, nz, ny, nx)
#define ADJ_FOLD_V_BC_BG() \
  FUNC(adj_fold_v_bc)(vz_shot, vy_shot, vx_shot, bc, nz, ny, nx)
#define ADJ_FOLD_V_BC_SC() \
  FUNC(adj_fold_v_bc)(vzsc_shot, vysc_shot, vxsc_shot, bc, nz, ny, nx)
#define ADJ_GATHER_V_GUARDS_SET(vset_z, vset_y, vset_x, srcp, model)         \
  FUNC(adj_gather_v_guards)(vset_z, vset_y, vset_x, srcp, model, bc, nz, ny, \
                            nx, rdz, rdy, rdx, dt, stag_w)
#define ADJ_GATHER_P_GUARDS_SET(target, vset_z, bz_, vset_y, by_, vset_x, bx_) \
  FUNC(adj_gather_p_guards)(target, vset_z, bz_, vset_y, by_, vset_x, bx_, bc, \
                            nz, ny, nx, rdz, rdy, rdx, dt, stag_w)
#define ADJ_FOLD_P_BC_FRX(p) FUNC(adj_fold_p_bc_frx)(p, bc_frx, nz, ny, nx)
#define ADJ_FOLD_V_BC_BG_FRX() \
  FUNC(adj_fold_v_bc_frx)(vz_shot, vy_shot, vx_shot, bc_frx, nz, ny, nx)
#define ADJ_FOLD_V_BC_SC_FRX() \
  FUNC(adj_fold_v_bc_frx)(vzsc_shot, vysc_shot, vxsc_shot, bc_frx, nz, ny, nx)
#define ADJ_GATHER_V_GUARDS_SET_FRX(vset_z, vset_y, vset_x, srcp, model)     \
  FUNC(adj_gather_v_guards_frx)(vset_z, vset_y, vset_x, srcp, model, bc_frx, \
                                nz, ny, nx, rdz, rdy, rdx, dt, stag_w)
#define ADJ_GATHER_P_GUARDS_SET_FRX(target, vset_z, bz_, vset_y, by_, vset_x,  \
                                    bx_)                                       \
  FUNC(adj_gather_p_guards_frx)(target, vset_z, bz_, vset_y, by_, vset_x, bx_, \
                                bc_frx, nz, ny, nx, rdz, rdy, rdx, dt, stag_w)
#elif DW_NDIM == 2
#define FILL_P_BC(p) FUNC(fill_p_bc)(p, bc, 1, ny, nx)
#define FILL_V_BC_BG() FUNC(fill_v_bc)(vy_shot, vx_shot, bc, 1, ny, nx)
#define FILL_V_BC_SC() FUNC(fill_v_bc)(vysc_shot, vxsc_shot, bc, 1, ny, nx)
#define FILL_P_BC_FR(p) FUNC(fill_p_bc)(p, bc_fr, 1, ny, nx)
#define FILL_V_BC_BG_FR() FUNC(fill_v_bc)(vy_shot, vx_shot, bc_fr, 1, ny, nx)
#define FILL_V_BC_SC_FR() \
  FUNC(fill_v_bc)(vysc_shot, vxsc_shot, bc_fr, 1, ny, nx)
#define ADJ_FOLD_P_BC(p) FUNC(adj_fold_p_bc)(p, bc, 1, ny, nx)
#define ADJ_FOLD_V_BC_BG() FUNC(adj_fold_v_bc)(vy_shot, vx_shot, bc, 1, ny, nx)
#define ADJ_FOLD_V_BC_SC() \
  FUNC(adj_fold_v_bc)(vysc_shot, vxsc_shot, bc, 1, ny, nx)
#define ADJ_GATHER_V_GUARDS_SET(vset_y, vset_x, srcp, model)                 \
  FUNC(adj_gather_v_guards)(vset_y, vset_x, srcp, model, bc, 1, ny, nx, rdy, \
                            rdx, dt, stag_w)
#define ADJ_GATHER_P_GUARDS_SET(target, vset_y, by_, vset_x, bx_)            \
  FUNC(adj_gather_p_guards)(target, vset_y, by_, vset_x, bx_, bc, 1, ny, nx, \
                            rdy, rdx, dt, stag_w)
#define ADJ_FOLD_P_BC_FRX(p) FUNC(adj_fold_p_bc_frx)(p, bc_frx, 1, ny, nx)
#define ADJ_FOLD_V_BC_BG_FRX() \
  FUNC(adj_fold_v_bc_frx)(vy_shot, vx_shot, bc_frx, 1, ny, nx)
#define ADJ_FOLD_V_BC_SC_FRX() \
  FUNC(adj_fold_v_bc_frx)(vysc_shot, vxsc_shot, bc_frx, 1, ny, nx)
#define ADJ_GATHER_V_GUARDS_SET_FRX(vset_y, vset_x, srcp, model)            \
  FUNC(adj_gather_v_guards_frx)(vset_y, vset_x, srcp, model, bc_frx, 1, ny, \
                                nx, rdy, rdx, dt, stag_w)
#define ADJ_GATHER_P_GUARDS_SET_FRX(target, vset_y, by_, vset_x, bx_)        \
  FUNC(adj_gather_p_guards_frx)(target, vset_y, by_, vset_x, bx_, bc_frx, 1, \
                                ny, nx, rdy, rdx, dt, stag_w)
#else
#define FILL_P_BC(p) FUNC(fill_p_bc)(p, bc, 1, 1, nx)
#define FILL_V_BC_BG() FUNC(fill_v_bc)(vx_shot, bc, 1, 1, nx)
#define FILL_V_BC_SC() FUNC(fill_v_bc)(vxsc_shot, bc, 1, 1, nx)
#define FILL_P_BC_FR(p) FUNC(fill_p_bc)(p, bc_fr, 1, 1, nx)
#define FILL_V_BC_BG_FR() FUNC(fill_v_bc)(vx_shot, bc_fr, 1, 1, nx)
#define FILL_V_BC_SC_FR() FUNC(fill_v_bc)(vxsc_shot, bc_fr, 1, 1, nx)
#define ADJ_FOLD_P_BC(p) FUNC(adj_fold_p_bc)(p, bc, 1, 1, nx)
#define ADJ_FOLD_V_BC_BG() FUNC(adj_fold_v_bc)(vx_shot, bc, 1, 1, nx)
#define ADJ_FOLD_V_BC_SC() FUNC(adj_fold_v_bc)(vxsc_shot, bc, 1, 1, nx)
#define ADJ_GATHER_V_GUARDS_SET(vset_x, srcp, model) \
  FUNC(adj_gather_v_guards)(vset_x, srcp, model, bc, 1, 1, nx, rdx, dt, stag_w)
#define ADJ_GATHER_P_GUARDS_SET(target, vset_x, bx_) \
  FUNC(adj_gather_p_guards)(target, vset_x, bx_, bc, 1, 1, nx, rdx, dt, stag_w)
#define ADJ_FOLD_P_BC_FRX(p) FUNC(adj_fold_p_bc_frx)(p, bc_frx, 1, 1, nx)
#define ADJ_FOLD_V_BC_BG_FRX() \
  FUNC(adj_fold_v_bc_frx)(vx_shot, bc_frx, 1, 1, nx)
#define ADJ_FOLD_V_BC_SC_FRX() \
  FUNC(adj_fold_v_bc_frx)(vxsc_shot, bc_frx, 1, 1, nx)
#define ADJ_GATHER_V_GUARDS_SET_FRX(vset_x, srcp, model)                    \
  FUNC(adj_gather_v_guards_frx)(vset_x, srcp, model, bc_frx, 1, 1, nx, rdx, \
                                dt, stag_w)
#define ADJ_GATHER_P_GUARDS_SET_FRX(target, vset_x, bx_)                    \
  FUNC(adj_gather_p_guards_frx)(target, vset_x, bx_, bc_frx, 1, 1, nx, rdx, \
                                dt, stag_w)
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(forward)(
    DW_DTYPE const* __restrict const k,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dk,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const dbuoyancy_x,
    DW_DTYPE const* __restrict const f_p,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const f_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const f_vy,
#endif
    DW_DTYPE const* __restrict const f_vx,
    DW_DTYPE const* __restrict const f_p_sc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const f_vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const f_vy_sc,
#endif
    DW_DTYPE const* __restrict const f_vx_sc, DW_DTYPE* __restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y,
#endif
    DW_DTYPE* __restrict const phi_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y,
#endif
    DW_DTYPE* __restrict const psi_x, DW_DTYPE* __restrict const psc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vysc,
#endif
    DW_DTYPE* __restrict const vxsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_sc,
#endif
    DW_DTYPE* __restrict const phi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y_sc,
#endif
    DW_DTYPE* __restrict const psi_x_sc,
    DW_DTYPE* __restrict const k_grad_store_1,
    DW_DTYPE* __restrict const k_grad_store_1b,
    void* __restrict const k_grad_store_2,
    void* __restrict const k_grad_store_3,
    char const* __restrict const* __restrict const k_grad_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const bz_grad_store_1,
    DW_DTYPE* __restrict const bz_grad_store_1b,
    void* __restrict const bz_grad_store_2,
    void* __restrict const bz_grad_store_3,
    char const* __restrict const* __restrict const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const by_grad_store_1,
    DW_DTYPE* __restrict const by_grad_store_1b,
    void* __restrict const by_grad_store_2,
    void* __restrict const by_grad_store_3,
    char const* __restrict const* __restrict const by_grad_filenames_ptr,
#endif
    DW_DTYPE* __restrict const bx_grad_store_1,
    DW_DTYPE* __restrict const bx_grad_store_1b,
    void* __restrict const bx_grad_store_2,
    void* __restrict const bx_grad_store_3,
    char const* __restrict const* __restrict const bx_grad_filenames_ptr,
    /* Scattered-term stores (div_sc / term_sc_d), written only when the
     * background models require grad -- consumed by the full two-field
     * backward's grad_K / grad_B_d cross-term (see section 4.3). */
    DW_DTYPE* __restrict const k_grad_sc_store_1,
    DW_DTYPE* __restrict const k_grad_sc_store_1b,
    void* __restrict const k_grad_sc_store_2,
    void* __restrict const k_grad_sc_store_3,
    char const* __restrict const* __restrict const k_grad_sc_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const bz_grad_sc_store_1,
    DW_DTYPE* __restrict const bz_grad_sc_store_1b,
    void* __restrict const bz_grad_sc_store_2,
    void* __restrict const bz_grad_sc_store_3,
    char const* __restrict const* __restrict const bz_grad_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const by_grad_sc_store_1,
    DW_DTYPE* __restrict const by_grad_sc_store_1b,
    void* __restrict const by_grad_sc_store_2,
    void* __restrict const by_grad_sc_store_3,
    char const* __restrict const* __restrict const by_grad_sc_filenames_ptr,
#endif
    DW_DTYPE* __restrict const bx_grad_sc_store_1,
    DW_DTYPE* __restrict const bx_grad_sc_store_1b,
    void* __restrict const bx_grad_sc_store_2,
    void* __restrict const bx_grad_sc_store_3,
    char const* __restrict const* __restrict const bx_grad_sc_filenames_ptr,
    DW_DTYPE* __restrict const r_p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const r_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const r_vy,
#endif
    DW_DTYPE* __restrict const r_vx, DW_DTYPE* __restrict const r_p_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const r_vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const r_vy_sc,
#endif
    DW_DTYPE* __restrict const r_vx_sc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    int64_t const* __restrict const sources_i_p,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_i_vy,
#endif
    int64_t const* __restrict const sources_i_vx,
    int64_t const* __restrict const sources_i_p_sc,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_i_vz_sc,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_i_vy_sc,
#endif
    int64_t const* __restrict const sources_i_vx_sc,
    int64_t const* __restrict const receivers_i_p,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy,
#endif
    int64_t const* __restrict const receivers_i_vx,
    int64_t const* __restrict const receivers_i_p_sc,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz_sc,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy_sc,
#endif
    int64_t const* __restrict const receivers_i_vx_sc,
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
    int64_t const n_sources_vx_per_shot, int64_t const n_sources_p_sc_per_shot,
#if DW_NDIM >= 3
    int64_t const n_sources_vz_sc_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_vy_sc_per_shot,
#endif
    int64_t const n_sources_vx_sc_per_shot,
    int64_t const n_receivers_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_per_shot,
#endif
    int64_t const n_receivers_vx_per_shot,
    int64_t const n_receivers_p_sc_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_sc_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_sc_per_shot,
#endif
    int64_t const n_receivers_vx_sc_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const k_requires_grad,
    bool const b_requires_grad, bool const dk_requires_grad,
    bool const db_requires_grad, bool const k_batched, bool const b_batched,
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
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
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
    if ((n_receivers_p_sc_per_shot > 0) && (receivers_i_p_sc == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_sc_per_shot > 0) && (receivers_i_vz_sc == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_sc_per_shot > 0) && (receivers_i_vy_sc == NULL))
      return 1;
#endif
    if ((n_receivers_vx_sc_per_shot > 0) && (receivers_i_vx_sc == NULL))
      return 1;
    if ((n_sources_p_per_shot > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot > 0) && sources_i_vx == NULL) return 1;
    if ((n_sources_p_sc_per_shot > 0) && sources_i_p_sc == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_sc_per_shot > 0) && sources_i_vz_sc == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_sc_per_shot > 0) && sources_i_vy_sc == NULL) return 1;
#endif
    if ((n_sources_vx_sc_per_shot > 0) && sources_i_vx_sc == NULL) return 1;
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

  /* Background-term stores (div_bg / term_bg_d) are needed for any model
   * gradient: grad_dK/grad_dB_d (scattered) AND grad_K/grad_B_d (background)
   * all correlate an adjoint field with them. The scattered-term stores
   * (div_sc / term_sc_d) are needed only for the background model gradient. */
  bool const store_needed = k_requires_grad || b_requires_grad ||
                            dk_requires_grad || db_requires_grad;
  bool const store_sc_needed = k_requires_grad || b_requires_grad;

  int64_t shot;
  /* Intra-shot spatial parallelism: shots run serially and each time step's
   * velocity and pressure stencils are spread across threads (see the omp
   * parallel regions below), so a single shot saturates all cores. */
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
    int64_t const src_i_p_sc = shot * n_sources_p_sc_per_shot;
#if DW_NDIM >= 3
    int64_t const src_i_vz_sc = shot * n_sources_vz_sc_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const src_i_vy_sc = shot * n_sources_vy_sc_per_shot;
#endif
    int64_t const src_i_vx_sc = shot * n_sources_vx_sc_per_shot;
    int64_t const rcv_i_p = shot * n_receivers_p_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz = shot * n_receivers_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy = shot * n_receivers_vy_per_shot;
#endif
    int64_t const rcv_i_vx = shot * n_receivers_vx_per_shot;
    int64_t const rcv_i_p_sc = shot * n_receivers_p_sc_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz_sc = shot * n_receivers_vz_sc_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy_sc = shot * n_receivers_vy_sc_per_shot;
#endif
    int64_t const rcv_i_vx_sc = shot * n_receivers_vx_sc_per_shot;

    DW_DTYPE const* __restrict const k_shot = k_batched ? k + si : k;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        b_batched ? buoyancy_z + si : buoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        b_batched ? buoyancy_y + si : buoyancy_y;
#endif
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        b_batched ? buoyancy_x + si : buoyancy_x;

    DW_DTYPE const* __restrict const dk_shot = k_batched ? dk + si : dk;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dbuoyancy_z_shot =
        b_batched ? dbuoyancy_z + si : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dbuoyancy_y_shot =
        b_batched ? dbuoyancy_y + si : dbuoyancy_y;
#endif
    DW_DTYPE const* __restrict const dbuoyancy_x_shot =
        b_batched ? dbuoyancy_x + si : dbuoyancy_x;

    DW_DTYPE* __restrict const p_shot = p + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz_shot = vz + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy_shot = vy + si;
#endif
    DW_DTYPE* __restrict const vx_shot = vx + si;

    DW_DTYPE* __restrict const psc_shot = psc + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vzsc_shot = vzsc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vysc_shot = vysc + si;
#endif
    DW_DTYPE* __restrict const vxsc_shot = vxsc + si;

#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_shot = phi_z + si;
    DW_DTYPE* __restrict const phi_z_sc_shot = phi_z_sc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_shot = phi_y + si;
    DW_DTYPE* __restrict const phi_y_sc_shot = phi_y_sc + si;
#endif
    DW_DTYPE* __restrict const phi_x_shot = phi_x + si;
    DW_DTYPE* __restrict const phi_x_sc_shot = phi_x_sc + si;

#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z_t = psi_z + si;
    DW_DTYPE* __restrict const psi_z_sc_t = psi_z_sc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y_t = psi_y + si;
    DW_DTYPE* __restrict const psi_y_sc_t = psi_y_sc + si;
#endif
    DW_DTYPE* __restrict const psi_x_t = psi_x + si;
    DW_DTYPE* __restrict const psi_x_sc_t = psi_x_sc + si;

    int64_t t;

    int storage_err = 0;
    FILE* fp_k_grad = NULL;
#if DW_NDIM >= 3
    FILE* fp_bz_grad = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_by_grad = NULL;
#endif
    FILE* fp_bx_grad = NULL;
    FILE* fp_k_grad_sc = NULL;
#if DW_NDIM >= 3
    FILE* fp_bz_grad_sc = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_by_grad_sc = NULL;
#endif
    FILE* fp_bx_grad_sc = NULL;

    OPEN_FILE_WRITE(k_grad, store_needed)
#if DW_NDIM >= 3
    OPEN_FILE_WRITE(bz_grad, store_needed)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_WRITE(by_grad, store_needed)
#endif
    OPEN_FILE_WRITE(bx_grad, store_needed)
    OPEN_FILE_WRITE(k_grad_sc, store_sc_needed)
#if DW_NDIM >= 3
    OPEN_FILE_WRITE(bz_grad_sc, store_sc_needed)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_WRITE(by_grad_sc, store_sc_needed)
#endif
    OPEN_FILE_WRITE(bx_grad_sc, store_sc_needed)

    for (t = start_t; t < start_t + nt; ++t) {
      SETUP_STORE_SAVE(k_grad, store_needed)
#if DW_NDIM >= 3
      SETUP_STORE_SAVE(bz_grad, store_needed)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_SAVE(by_grad, store_needed)
#endif
      SETUP_STORE_SAVE(bx_grad, store_needed)
      SETUP_STORE_SAVE(k_grad_sc, store_sc_needed)
#if DW_NDIM >= 3
      SETUP_STORE_SAVE(bz_grad_sc, store_sc_needed)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_SAVE(by_grad_sc, store_sc_needed)
#endif
      SETUP_STORE_SAVE(bx_grad_sc, store_sc_needed)

      bool const store_needed_t = store_needed && ((t % step_ratio) == 0);
      bool const store_sc_needed_t = store_sc_needed && ((t % step_ratio) == 0);

      // Refill the pressure guard ring (read by the velocity update), on both
      // fields independently; this also holds free-surface pressure nodes to
      // zero before recording (no-op when every face is PML).
      if (has_bc) {
        FILL_P_BC(p_shot);
        FILL_P_BC(psc_shot);
      }

      // Record Receivers (background then scattered)
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
      if (n_receivers_p_sc_per_shot > 0) {
        record_from_wavefield(
            psc_shot, receivers_i_p_sc + rcv_i_p_sc,
            r_p_sc + t * n_shots * n_receivers_p_sc_per_shot + rcv_i_p_sc,
            n_receivers_p_sc_per_shot);
      }
#if DW_NDIM >= 3
      if (n_receivers_vz_sc_per_shot > 0) {
        record_from_wavefield(
            vzsc_shot, receivers_i_vz_sc + rcv_i_vz_sc,
            r_vz_sc + t * n_shots * n_receivers_vz_sc_per_shot + rcv_i_vz_sc,
            n_receivers_vz_sc_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_vy_sc_per_shot > 0) {
        record_from_wavefield(
            vysc_shot, receivers_i_vy_sc + rcv_i_vy_sc,
            r_vy_sc + t * n_shots * n_receivers_vy_sc_per_shot + rcv_i_vy_sc,
            n_receivers_vy_sc_per_shot);
      }
#endif
      if (n_receivers_vx_sc_per_shot > 0) {
        record_from_wavefield(
            vxsc_shot, receivers_i_vx_sc + rcv_i_vx_sc,
            r_vx_sc + t * n_shots * n_receivers_vx_sc_per_shot + rcv_i_vx_sc,
            n_receivers_vx_sc_per_shot);
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
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
        for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

          // Update Velocities (background then scattered, coupled)
          // v      -= dt * B  * (grad(p)  + psi)
          // dv      -= dt * B  * (grad(dp) + psi_sc)  +  dt * dB * term_bg

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
                        p_shot, psc_shot, vz_shot, vzsc_shot, psi_z_t,
                        psi_z_sc_t, bz_grad_store_1_t, bz_grad_sc_store_1_t,
                        buoyancy_z_shot[i], dbuoyancy_z_shot[i], azh, bzh, dt,
                        store_needed_t, store_sc_needed_t, i, ny, nx, z,
                        pml_z != 1, rdz);
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
                        p_shot, psc_shot, vy_shot, vysc_shot, psi_y_t,
                        psi_y_sc_t, by_grad_store_1_t, by_grad_sc_store_1_t,
                        buoyancy_y_shot[i], dbuoyancy_y_shot[i], ayh, byh, dt,
                        store_needed_t, store_sc_needed_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx, y, pml_y != 1, rdy);
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
                        p_shot, psc_shot, vx_shot, vxsc_shot, psi_x_t,
                        psi_x_sc_t, bx_grad_store_1_t, bx_grad_sc_store_1_t,
                        buoyancy_x_shot[i], dbuoyancy_x_shot[i], axh, bxh, dt,
                        store_needed_t, store_sc_needed_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
                        x, pml_x != 1, rdx);
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
      if (n_sources_vz_sc_per_shot > 0) {
        add_to_wavefield(
            vzsc_shot, sources_i_vz_sc + src_i_vz_sc,
            f_vz_sc + t * n_shots * n_sources_vz_sc_per_shot + src_i_vz_sc,
            n_sources_vz_sc_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_vy_per_shot > 0) {
        add_to_wavefield(vy_shot, sources_i_vy + src_i_vy,
                         f_vy + t * n_shots * n_sources_vy_per_shot + src_i_vy,
                         n_sources_vy_per_shot);
      }
      if (n_sources_vy_sc_per_shot > 0) {
        add_to_wavefield(
            vysc_shot, sources_i_vy_sc + src_i_vy_sc,
            f_vy_sc + t * n_shots * n_sources_vy_sc_per_shot + src_i_vy_sc,
            n_sources_vy_sc_per_shot);
      }
#endif
      if (n_sources_vx_per_shot > 0) {
        add_to_wavefield(vx_shot, sources_i_vx + src_i_vx,
                         f_vx + t * n_shots * n_sources_vx_per_shot + src_i_vx,
                         n_sources_vx_per_shot);
      }
      if (n_sources_vx_sc_per_shot > 0) {
        add_to_wavefield(
            vxsc_shot, sources_i_vx_sc + src_i_vx_sc,
            f_vx_sc + t * n_shots * n_sources_vx_sc_per_shot + src_i_vx_sc,
            n_sources_vx_sc_per_shot);
      }

      // Refill the boundary-normal velocity guard ring (read by the pressure
      // update), on both fields independently; holds rigid-wall velocity
      // nodes to zero (no-op for all-PML).
      if (has_bc) {
        FILL_V_BC_BG();
        FILL_V_BC_SC();
      }

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
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
        for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < 3; ++pml_x) {
#if DW_NDIM >= 3
              int64_t z;
#endif
#if DW_NDIM >= 2
              int64_t y;
#endif
              int64_t x;

          // Update Pressure (background then scattered, coupled)
          // p  -= dt * K * (div(v)  + phi)
          // dp -= dt * K * (div(dv) + phi_sc)  +  dt * dK * div_bg

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
                        vz_shot, vzsc_shot,
#endif
#if DW_NDIM >= 2
                        vy_shot, vysc_shot,
#endif
                        vx_shot, vxsc_shot, p_shot, psc_shot,
#if DW_NDIM >= 3
                        phi_z_shot, phi_z_sc_shot,
#endif
#if DW_NDIM >= 2
                        phi_y_shot, phi_y_sc_shot,
#endif
                        phi_x_shot, phi_x_sc_shot, k_grad_store_1_t,
                        k_grad_sc_store_1_t, k_shot[i], dk_shot[i], dt,
                        store_needed_t, store_sc_needed_t, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != 1, az, bz, rdz,
#endif
#if DW_NDIM >= 2
                        y, pml_y != 1, ay, by, rdy,
#endif
                        x, pml_x != 1, ax, bx, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Inject Sources (background then scattered)
      if (n_sources_p_per_shot > 0) {
        add_to_wavefield(p_shot, sources_i_p + src_i_p,
                         f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
                         n_sources_p_per_shot);
      }
      if (n_sources_p_sc_per_shot > 0) {
        add_to_wavefield(
            psc_shot, sources_i_p_sc + src_i_p_sc,
            f_p_sc + t * n_shots * n_sources_p_sc_per_shot + src_i_p_sc,
            n_sources_p_sc_per_shot);
      }

      // Save snapshots (background terms only; consumed by backward_sc)
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

      SAVE_SNAPSHOT(k_grad, store_needed_t)
#if DW_NDIM >= 3
      SAVE_SNAPSHOT(bz_grad, store_needed_t)
#endif
#if DW_NDIM >= 2
      SAVE_SNAPSHOT(by_grad, store_needed_t)
#endif
      SAVE_SNAPSHOT(bx_grad, store_needed_t)
      SAVE_SNAPSHOT(k_grad_sc, store_sc_needed_t)
#if DW_NDIM >= 3
      SAVE_SNAPSHOT(bz_grad_sc, store_sc_needed_t)
#endif
#if DW_NDIM >= 2
      SAVE_SNAPSHOT(by_grad_sc, store_sc_needed_t)
#endif
      SAVE_SNAPSHOT(bx_grad_sc, store_sc_needed_t)
    }

#define CLOSE_FILE(name) \
  if (dw_close_file(&fp_##name) != 0) storage_err = 1;

  storage_cleanup:
    CLOSE_FILE(k_grad)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad)
#endif
    CLOSE_FILE(bx_grad)
    CLOSE_FILE(k_grad_sc)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad_sc)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad_sc)
#endif
    CLOSE_FILE(bx_grad_sc)
    if (storage_err) return 1;
  }
  return 0;
}

// backward_sc: scattered-only adjoint (fixed background K, B_d). Propagates
// only the scattered adjoint field with the ORDINARY (unperturbed) acoustic
// adjoint recursion (identical in form to acoustic.c's backward()), driven
// by the scattered-data residuals, and accumulates grad_dK / grad_dB_d by
// correlating the propagating scattered adjoint field with the checkpointed
// background terms saved by `forward`.
#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(backward_sc)(
    DW_DTYPE const* __restrict const k,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const grad_r_p_sc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const grad_r_vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const grad_r_vy_sc,
#endif
    DW_DTYPE const* __restrict const grad_r_vx_sc,
    DW_DTYPE* __restrict const psc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vysc,
#endif
    DW_DTYPE* __restrict const vxsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_sc,
#endif
    DW_DTYPE* __restrict const phi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y_sc,
#endif
    DW_DTYPE* __restrict const psi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_zn_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_yn_sc,
#endif
    DW_DTYPE* __restrict const psi_xn_sc,

    DW_DTYPE* __restrict const k_grad_store_1,
    DW_DTYPE* __restrict const k_grad_store_1b,
    void* __restrict const k_grad_store_2,
    void* __restrict const k_grad_store_3,
    char const* __restrict const* __restrict const k_grad_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const bz_grad_store_1,
    DW_DTYPE* __restrict const bz_grad_store_1b,
    void* __restrict const bz_grad_store_2,
    void* __restrict const bz_grad_store_3,
    char const* __restrict const* __restrict const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const by_grad_store_1,
    DW_DTYPE* __restrict const by_grad_store_1b,
    void* __restrict const by_grad_store_2,
    void* __restrict const by_grad_store_3,
    char const* __restrict const* __restrict const by_grad_filenames_ptr,
#endif
    DW_DTYPE* __restrict const bx_grad_store_1,
    DW_DTYPE* __restrict const bx_grad_store_1b,
    void* __restrict const bx_grad_store_2,
    void* __restrict const bx_grad_store_3,
    char const* __restrict const* __restrict const bx_grad_filenames_ptr,
    DW_DTYPE* __restrict const grad_f_p_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_f_vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_f_vy_sc,
#endif
    DW_DTYPE* __restrict const grad_f_vx_sc, DW_DTYPE* __restrict const grad_dk,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_dbz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_dby,
#endif
    DW_DTYPE* __restrict const grad_dbx,
    DW_DTYPE* __restrict const grad_dk_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_dbz_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_dby_thread,
#endif
    DW_DTYPE* __restrict const grad_dbx_thread,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    int64_t const* __restrict const sources_i_p_sc,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_i_vz_sc,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_i_vy_sc,
#endif
    int64_t const* __restrict const sources_i_vx_sc,
    int64_t const* __restrict const receivers_i_p_sc,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz_sc,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy_sc,
#endif
    int64_t const* __restrict const receivers_i_vx_sc,
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
    int64_t const nx, int64_t const n_sources_p_sc_per_shot,
#if DW_NDIM >= 3
    int64_t const n_sources_vz_sc_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_vy_sc_per_shot,
#endif
    int64_t const n_sources_vx_sc_per_shot,
    int64_t const n_receivers_p_sc_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_sc_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_sc_per_shot,
#endif
    int64_t const n_receivers_vx_sc_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const dk_requires_grad,
    bool const db_requires_grad, bool const k_batched, bool const b_batched,
    bool const dk_batched, bool const db_batched,
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
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((dk_requires_grad) && (grad_dk == NULL)) return 1;
    if ((n_receivers_p_sc_per_shot > 0) && (receivers_i_p_sc == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_sc_per_shot > 0) && (receivers_i_vz_sc == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_sc_per_shot > 0) && (receivers_i_vy_sc == NULL))
      return 1;
#endif
    if ((n_receivers_vx_sc_per_shot > 0) && (receivers_i_vx_sc == NULL))
      return 1;
    if ((n_sources_p_sc_per_shot > 0) && sources_i_p_sc == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_sc_per_shot > 0) && sources_i_vz_sc == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_sc_per_shot > 0) && sources_i_vy_sc == NULL) return 1;
#endif
    if ((n_sources_vx_sc_per_shot > 0) && sources_i_vx_sc == NULL) return 1;
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
#if DW_NDIM >= 3
  int const per_z = bc[0] == 3;
  /* Axes whose fold writes interior nodes defer the (perturbation-)
   * buoyancy-gradient accumulation until after the fold. */
  int const defer_z = per_z || bc_frx[0] || bc_frx[1];
#endif
#if DW_NDIM >= 2
  int const per_y = bc[2] == 3;
  int const defer_y = per_y || bc_frx[2] || bc_frx[3];
#endif
  int const per_x = bc[4] == 3;
  int const defer_x = per_x || bc_frx[4] || bc_frx[5];
#if DW_NDIM == 3
  int const has_per = per_z || per_y || per_x;
#elif DW_NDIM == 2
  int const has_per = per_y || per_x;
#else
  int const has_per = per_x;
#endif
  DW_DTYPE const stag_w_arr[FD_PAD] = DW_STAG_TAYLOR_W;
  DW_DTYPE const* const stag_w = stag_w_arr;
  (void)has_bc;

  bool const store_needed = dk_requires_grad || db_requires_grad;

  int64_t shot;
  /* Shots run serially; each time step's adjoint velocity and pressure
   * stencils are spread across threads (see the omp parallel regions below).
   * Model gradients accumulate directly into the shared gradient buffers --
   * each cell is written by exactly one thread, and time steps and shots are
   * barrier-separated, so there is no race. */
  (void)grad_dk_thread;
#if DW_NDIM >= 3
  (void)grad_dbz_thread;
#endif
#if DW_NDIM >= 2
  (void)grad_dby_thread;
#endif
  (void)grad_dbx_thread;
  for (shot = 0; shot < n_shots; ++shot) {
    int64_t const si = shot * n_grid_points;
    int64_t const src_i_p_sc = shot * n_sources_p_sc_per_shot;
#if DW_NDIM >= 3
    int64_t const src_i_vz_sc = shot * n_sources_vz_sc_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const src_i_vy_sc = shot * n_sources_vy_sc_per_shot;
#endif
    int64_t const src_i_vx_sc = shot * n_sources_vx_sc_per_shot;
    int64_t const rcv_i_p_sc = shot * n_receivers_p_sc_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz_sc = shot * n_receivers_vz_sc_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy_sc = shot * n_receivers_vy_sc_per_shot;
#endif
    int64_t const rcv_i_vx_sc = shot * n_receivers_vx_sc_per_shot;

    DW_DTYPE const* __restrict const k_shot = k_batched ? k + si : k;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        b_batched ? buoyancy_z + si : buoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        b_batched ? buoyancy_y + si : buoyancy_y;
#endif
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        b_batched ? buoyancy_x + si : buoyancy_x;

    // p, vz, vy, vx (scattered) hold the scattered adjoint field.
    DW_DTYPE* __restrict const psc_shot = psc + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vzsc_shot = vzsc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vysc_shot = vysc + si;
#endif
    DW_DTYPE* __restrict const vxsc_shot = vxsc + si;

#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_sc_shot = phi_z_sc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_sc_shot = phi_y_sc + si;
#endif
    DW_DTYPE* __restrict const phi_x_sc_shot = phi_x_sc + si;

    // Gradients: accumulated directly into the shared buffers (see the
    // race-freedom comment above the shot loop).
    DW_DTYPE* __restrict const grad_dk_shot =
        grad_dk != NULL ? (grad_dk + (dk_batched ? si : 0)) : NULL;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_dbz_shot =
        grad_dbz != NULL ? (grad_dbz + (db_batched ? si : 0)) : NULL;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_dby_shot =
        grad_dby != NULL ? (grad_dby + (db_batched ? si : 0)) : NULL;
#endif
    DW_DTYPE* __restrict const grad_dbx_shot =
        grad_dbx != NULL ? (grad_dbx + (db_batched ? si : 0)) : NULL;

    int64_t t;

    int storage_err = 0;
    FILE* fp_k_grad = NULL;
#if DW_NDIM >= 3
    FILE* fp_bz_grad = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_by_grad = NULL;
#endif
    FILE* fp_bx_grad = NULL;

    OPEN_FILE_READ(k_grad, store_needed)
#if DW_NDIM >= 3
    OPEN_FILE_READ(bz_grad, store_needed)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_READ(by_grad, store_needed)
#endif
    OPEN_FILE_READ(bx_grad, store_needed)

    for (t = start_t - 1; t >= start_t - nt; --t) {
      SETUP_STORE_LOAD(k_grad, store_needed)
#if DW_NDIM >= 3
      SETUP_STORE_LOAD(bz_grad, store_needed)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_LOAD(by_grad, store_needed)
#endif
      SETUP_STORE_LOAD(bx_grad, store_needed)

#if DW_NDIM >= 3
      DW_DTYPE* __restrict const psi_z_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_zn_sc : psi_z_sc) + si;
      DW_DTYPE* __restrict const psi_zn_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_z_sc : psi_zn_sc) + si;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const psi_y_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_yn_sc : psi_y_sc) + si;
      DW_DTYPE* __restrict const psi_yn_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_y_sc : psi_yn_sc) + si;
#endif
      DW_DTYPE* __restrict const psi_x_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_xn_sc : psi_x_sc) + si;
      DW_DTYPE* __restrict const psi_xn_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_x_sc : psi_xn_sc) + si;

      bool const dk_requires_grad_t =
          dk_requires_grad && ((t % step_ratio) == 0);
      bool const db_requires_grad_t =
          db_requires_grad && ((t % step_ratio) == 0);

      // Periodic faces: fold the scattered adjoint pressure's guard cells
      // back into the interior nodes the forward fill copied them from (the
      // exact transpose of the fill). The fold modifies interior seam nodes,
      // so it must precede the source-gradient read below.
      if (has_per) ADJ_FOLD_P_BC(psc_shot);
      if (has_frx) ADJ_FOLD_P_BC_FRX(psc_shot);

      if (n_sources_p_sc_per_shot > 0) {
        record_from_wavefield(
            psc_shot, sources_i_p_sc + src_i_p_sc,
            grad_f_p_sc + t * n_shots * n_sources_p_sc_per_shot + src_i_p_sc,
            n_sources_p_sc_per_shot);
      }

      // Adjoint guard fill of the scattered pressure field (read by the
      // adjoint velocity update) on the free/rigid faces: the same image the
      // forward applies, which realises the exact transpose on a reflecting
      // face. No-op for all-PML.
      if (has_fr) FILL_P_BC_FR(psc_shot);

      // Update Adjoint Velocities (scattered), ordinary acoustic adjoint.

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
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
        for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < 3; ++pml_x) {
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
                    FUNC(backward_sc_vz_cell)(
                        psc_shot, k_shot, vzsc_shot, phi_z_sc_shot, psi_z_sc_t,
                        psi_zn_sc_t, buoyancy_z_shot[i], grad_dbz_shot,
                        bz_grad_store_1_t, azh, bz, dt, step_ratio,
                        db_requires_grad_t && !defer_z, i, i, ny, nx, z,
                        pml_z != 1, rdz);
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
                    FUNC(backward_sc_vy_cell)(
                        psc_shot, k_shot, vysc_shot, phi_y_sc_shot, psi_y_sc_t,
                        psi_yn_sc_t, buoyancy_y_shot[i], grad_dby_shot,
                        by_grad_store_1_t, ayh, by, dt, step_ratio,
                        db_requires_grad_t && !defer_y, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx, y, pml_y != 1, rdy);
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
                    FUNC(backward_sc_vx_cell)(
                        psc_shot, k_shot, vxsc_shot, phi_x_sc_shot, psi_x_sc_t,
                        psi_xn_sc_t, buoyancy_x_shot[i], grad_dbx_shot,
                        bx_grad_store_1_t, axh, bx, dt, step_ratio,
                        db_requires_grad_t && !defer_x, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
                        x, pml_x != 1, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Periodic faces: accumulate the adjoint-stencil contributions that
      // fall on the scattered adjoint velocity's guard cells (and the slaved
      // extra high interior node), then fold them back into the interior
      // (the exact transpose of the forward velocity fill). The fold
      // modifies interior seam nodes, so it must precede the
      // source-gradient reads below, and the perturbation-buoyancy-gradient
      // accumulation for periodic axes (suppressed in the
      // backward_sc_v*_cell calls above) runs only after it.
      if (has_per) {
#if DW_NDIM == 3
        ADJ_GATHER_V_GUARDS_SET(vzsc_shot, vysc_shot, vxsc_shot, psc_shot,
                                k_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_V_GUARDS_SET(vysc_shot, vxsc_shot, psc_shot, k_shot);
#else
        ADJ_GATHER_V_GUARDS_SET(vxsc_shot, psc_shot, k_shot);
#endif
        ADJ_FOLD_V_BC_SC();
      }
      if (has_frx) {
#if DW_NDIM == 3
        ADJ_GATHER_V_GUARDS_SET_FRX(vzsc_shot, vysc_shot, vxsc_shot, psc_shot,
                                    k_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_V_GUARDS_SET_FRX(vysc_shot, vxsc_shot, psc_shot, k_shot);
#else
        ADJ_GATHER_V_GUARDS_SET_FRX(vxsc_shot, psc_shot, k_shot);
#endif
        ADJ_FOLD_V_BC_SC_FRX();
      }

#if DW_NDIM >= 3
      if (n_sources_vz_sc_per_shot > 0) {
        record_from_wavefield(
            vzsc_shot, sources_i_vz_sc + src_i_vz_sc,
            grad_f_vz_sc + t * n_shots * n_sources_vz_sc_per_shot + src_i_vz_sc,
            n_sources_vz_sc_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_vy_sc_per_shot > 0) {
        record_from_wavefield(
            vysc_shot, sources_i_vy_sc + src_i_vy_sc,
            grad_f_vy_sc + t * n_shots * n_sources_vy_sc_per_shot + src_i_vy_sc,
            n_sources_vy_sc_per_shot);
      }
#endif
      if (n_sources_vx_sc_per_shot > 0) {
        record_from_wavefield(
            vxsc_shot, sources_i_vx_sc + src_i_vx_sc,
            grad_f_vx_sc + t * n_shots * n_sources_vx_sc_per_shot + src_i_vx_sc,
            n_sources_vx_sc_per_shot);
      }

      // Adjoint guard fill of the scattered velocity field (read by the
      // adjoint pressure update) on the free/rigid faces: the
      // transpose-realising image, matching the forward. No-op for all-PML.
      if (has_fr) FILL_V_BC_SC_FR();

      // Periodic axes: the perturbation-buoyancy gradient must use the
      // post-fold scattered adjoint velocity (the fold writes into interior
      // seam nodes), so it is accumulated here instead of inside
      // backward_sc_v*_cell.
      if (db_requires_grad_t && (has_per || has_frx)) {
#if DW_NDIM == 3
        int64_t gz, gy, gx;
        if (defer_z)
          for (gz = FD_PAD; gz < nz - FD_PAD; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                grad_dbz_shot[gi] -= dt * vzsc_shot[gi] *
                                     bz_grad_store_1_t[gi] *
                                     (DW_DTYPE)step_ratio;
              }
        if (defer_y)
          for (gz = FD_PAD; gz < nz - FD_PAD + 1; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                grad_dby_shot[gi] -= dt * vysc_shot[gi] *
                                     by_grad_store_1_t[gi] *
                                     (DW_DTYPE)step_ratio;
              }
        if (defer_x)
          for (gz = FD_PAD; gz < nz - FD_PAD + 1; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                grad_dbx_shot[gi] -= dt * vxsc_shot[gi] *
                                     bx_grad_store_1_t[gi] *
                                     (DW_DTYPE)step_ratio;
              }
#elif DW_NDIM == 2
        int64_t gy, gx;
        if (defer_y)
          for (gy = FD_PAD; gy < ny - FD_PAD; ++gy)
            for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
              int64_t const gi = gy * nx + gx;
              grad_dby_shot[gi] -= dt * vysc_shot[gi] * by_grad_store_1_t[gi] *
                                   (DW_DTYPE)step_ratio;
            }
        if (defer_x)
          for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
            for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
              int64_t const gi = gy * nx + gx;
              grad_dbx_shot[gi] -= dt * vxsc_shot[gi] * bx_grad_store_1_t[gi] *
                                   (DW_DTYPE)step_ratio;
            }
#else
        int64_t gx;
        if (defer_x)
          for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
            grad_dbx_shot[gx] -= dt * vxsc_shot[gx] * bx_grad_store_1_t[gx] *
                                 (DW_DTYPE)step_ratio;
          }
#endif
      }

      // Update Adjoint Pressure (scattered), ordinary acoustic adjoint.

#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
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
        for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < 3; ++pml_x) {
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
                    FUNC(backward_sc_p_cell)(
                        psc_shot,
#if DW_NDIM >= 3
                        vzsc_shot, phi_z_sc_shot, buoyancy_z_shot, psi_z_sc_t,
#endif
#if DW_NDIM >= 2
                        vysc_shot, phi_y_sc_shot, buoyancy_y_shot, psi_y_sc_t,
#endif
                        vxsc_shot, phi_x_sc_shot, buoyancy_x_shot, psi_x_sc_t,
                        k_shot[i], grad_dk_shot, k_grad_store_1_t, dt,
                        step_ratio, dk_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != 1, az, bzh, rdz,
#endif
#if DW_NDIM >= 2
                        y, pml_y != 1, ay, byh, rdy,
#endif
                        x, pml_x != 1, ax, bxh, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Periodic faces: accumulate the adjoint-stencil contributions that
      // fall on the scattered adjoint pressure's guard cells (folded into
      // the interior at the top of the next iteration).
      if (has_per) {
#if DW_NDIM == 3
        ADJ_GATHER_P_GUARDS_SET(psc_shot, vzsc_shot, buoyancy_z_shot, vysc_shot,
                                buoyancy_y_shot, vxsc_shot, buoyancy_x_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_P_GUARDS_SET(psc_shot, vysc_shot, buoyancy_y_shot, vxsc_shot,
                                buoyancy_x_shot);
#else
        ADJ_GATHER_P_GUARDS_SET(psc_shot, vxsc_shot, buoyancy_x_shot);
#endif
      }
      if (has_frx) {
#if DW_NDIM == 3
        ADJ_GATHER_P_GUARDS_SET_FRX(psc_shot, vzsc_shot, buoyancy_z_shot,
                                    vysc_shot, buoyancy_y_shot, vxsc_shot,
                                    buoyancy_x_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_P_GUARDS_SET_FRX(psc_shot, vysc_shot, buoyancy_y_shot,
                                    vxsc_shot, buoyancy_x_shot);
#else
        ADJ_GATHER_P_GUARDS_SET_FRX(psc_shot, vxsc_shot, buoyancy_x_shot);
#endif
      }

      if (n_receivers_p_sc_per_shot > 0) {
        add_to_wavefield(
            psc_shot, receivers_i_p_sc + rcv_i_p_sc,
            grad_r_p_sc + t * n_shots * n_receivers_p_sc_per_shot + rcv_i_p_sc,
            n_receivers_p_sc_per_shot);
      }
#if DW_NDIM >= 3
      if (n_receivers_vz_sc_per_shot > 0) {
        add_to_wavefield(vzsc_shot, receivers_i_vz_sc + rcv_i_vz_sc,
                         grad_r_vz_sc +
                             t * n_shots * n_receivers_vz_sc_per_shot +
                             rcv_i_vz_sc,
                         n_receivers_vz_sc_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_vy_sc_per_shot > 0) {
        add_to_wavefield(vysc_shot, receivers_i_vy_sc + rcv_i_vy_sc,
                         grad_r_vy_sc +
                             t * n_shots * n_receivers_vy_sc_per_shot +
                             rcv_i_vy_sc,
                         n_receivers_vy_sc_per_shot);
      }
#endif
      if (n_receivers_vx_sc_per_shot > 0) {
        add_to_wavefield(vxsc_shot, receivers_i_vx_sc + rcv_i_vx_sc,
                         grad_r_vx_sc +
                             t * n_shots * n_receivers_vx_sc_per_shot +
                             rcv_i_vx_sc,
                         n_receivers_vx_sc_per_shot);
      }
    }

#define CLOSE_FILE(name) \
  if (dw_close_file(&fp_##name) != 0) storage_err = 1;

  storage_cleanup:
    CLOSE_FILE(k_grad)
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

// backward: full two-field adjoint (background AND scattered). Propagates the
// scattered adjoint field with the ORDINARY acoustic adjoint (as backward_sc)
// AND the background adjoint field with the coupled adjoint (VX_ADJOINT_BG /
// P_ADJOINT_X_BG etc.: every K*P_adj -> K*P_adj + dK*P_adj_sc, every B*V_adj ->
// B*V_adj + dB*V_adj_sc, in both plain and PML memory-variable forms). Both
// adjoint fields are seeded directly from the output-wavefield gradients (no
// grad_wfp negation -- acoustic's first-order scheme has none); only the
// psi/psin auxiliary is ping-ponged. Accumulates grad_dK/grad_dB_d (scattered
// adjoint field correlated with the background store) AND grad_K/grad_B_d
// (background adjoint field correlated with the background store PLUS scattered
// adjoint field correlated with the scattered store; two terms each, section
// 4.3). Also records grad_f (background) and grad_f_sc (scattered) source
// gradients. Powers the acoustic HVP and the background-model / source /
// background-initial-wavefield gradients of acoustic_born.
#ifdef _WIN32
__declspec(dllexport)
#endif
DW_TARGET_CLONES int FUNC(backward)(
    DW_DTYPE const* __restrict const k,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const dk,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dbuoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dbuoyancy_y,
#endif
    DW_DTYPE const* __restrict const dbuoyancy_x,
    DW_DTYPE const* __restrict const grad_r_p,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const grad_r_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const grad_r_vy,
#endif
    DW_DTYPE const* __restrict const grad_r_vx,
    DW_DTYPE const* __restrict const grad_r_p_sc,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const grad_r_vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const grad_r_vy_sc,
#endif
    DW_DTYPE const* __restrict const grad_r_vx_sc, DW_DTYPE* __restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y,
#endif
    DW_DTYPE* __restrict const phi_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y,
#endif
    DW_DTYPE* __restrict const psi_x,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_zn,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_yn,
#endif
    DW_DTYPE* __restrict const psi_xn, DW_DTYPE* __restrict const psc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vzsc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vysc,
#endif
    DW_DTYPE* __restrict const vxsc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_sc,
#endif
    DW_DTYPE* __restrict const phi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y_sc,
#endif
    DW_DTYPE* __restrict const psi_x_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_zn_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_yn_sc,
#endif
    DW_DTYPE* __restrict const psi_xn_sc,
    DW_DTYPE* __restrict const k_grad_store_1,
    DW_DTYPE* __restrict const k_grad_store_1b,
    void* __restrict const k_grad_store_2,
    void* __restrict const k_grad_store_3,
    char const* __restrict const* __restrict const k_grad_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const bz_grad_store_1,
    DW_DTYPE* __restrict const bz_grad_store_1b,
    void* __restrict const bz_grad_store_2,
    void* __restrict const bz_grad_store_3,
    char const* __restrict const* __restrict const bz_grad_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const by_grad_store_1,
    DW_DTYPE* __restrict const by_grad_store_1b,
    void* __restrict const by_grad_store_2,
    void* __restrict const by_grad_store_3,
    char const* __restrict const* __restrict const by_grad_filenames_ptr,
#endif
    DW_DTYPE* __restrict const bx_grad_store_1,
    DW_DTYPE* __restrict const bx_grad_store_1b,
    void* __restrict const bx_grad_store_2,
    void* __restrict const bx_grad_store_3,
    char const* __restrict const* __restrict const bx_grad_filenames_ptr,
    DW_DTYPE* __restrict const k_grad_sc_store_1,
    DW_DTYPE* __restrict const k_grad_sc_store_1b,
    void* __restrict const k_grad_sc_store_2,
    void* __restrict const k_grad_sc_store_3,
    char const* __restrict const* __restrict const k_grad_sc_filenames_ptr,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const bz_grad_sc_store_1,
    DW_DTYPE* __restrict const bz_grad_sc_store_1b,
    void* __restrict const bz_grad_sc_store_2,
    void* __restrict const bz_grad_sc_store_3,
    char const* __restrict const* __restrict const bz_grad_sc_filenames_ptr,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const by_grad_sc_store_1,
    DW_DTYPE* __restrict const by_grad_sc_store_1b,
    void* __restrict const by_grad_sc_store_2,
    void* __restrict const by_grad_sc_store_3,
    char const* __restrict const* __restrict const by_grad_sc_filenames_ptr,
#endif
    DW_DTYPE* __restrict const bx_grad_sc_store_1,
    DW_DTYPE* __restrict const bx_grad_sc_store_1b,
    void* __restrict const bx_grad_sc_store_2,
    void* __restrict const bx_grad_sc_store_3,
    char const* __restrict const* __restrict const bx_grad_sc_filenames_ptr,
    DW_DTYPE* __restrict const grad_f_p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_f_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_f_vy,
#endif
    DW_DTYPE* __restrict const grad_f_vx,
    DW_DTYPE* __restrict const grad_f_p_sc,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_f_vz_sc,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_f_vy_sc,
#endif
    DW_DTYPE* __restrict const grad_f_vx_sc, DW_DTYPE* __restrict const grad_k,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_by,
#endif
    DW_DTYPE* __restrict const grad_bx, DW_DTYPE* __restrict const grad_dk,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_dbz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_dby,
#endif
    DW_DTYPE* __restrict const grad_dbx,
    DW_DTYPE* __restrict const grad_k_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_bz_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_by_thread,
#endif
    DW_DTYPE* __restrict const grad_bx_thread,
    DW_DTYPE* __restrict const grad_dk_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_dbz_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_dby_thread,
#endif
    DW_DTYPE* __restrict const grad_dbx_thread,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    int64_t const* __restrict const sources_i_p,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_i_vy,
#endif
    int64_t const* __restrict const sources_i_vx,
    int64_t const* __restrict const sources_i_p_sc,
#if DW_NDIM >= 3
    int64_t const* __restrict const sources_i_vz_sc,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const sources_i_vy_sc,
#endif
    int64_t const* __restrict const sources_i_vx_sc,
    int64_t const* __restrict const receivers_i_p,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy,
#endif
    int64_t const* __restrict const receivers_i_vx,
    int64_t const* __restrict const receivers_i_p_sc,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz_sc,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy_sc,
#endif
    int64_t const* __restrict const receivers_i_vx_sc,
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
    int64_t const n_sources_vx_per_shot, int64_t const n_sources_p_sc_per_shot,
#if DW_NDIM >= 3
    int64_t const n_sources_vz_sc_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_sources_vy_sc_per_shot,
#endif
    int64_t const n_sources_vx_sc_per_shot,
    int64_t const n_receivers_p_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_per_shot,
#endif
    int64_t const n_receivers_vx_per_shot,
    int64_t const n_receivers_p_sc_per_shot,
#if DW_NDIM >= 3
    int64_t const n_receivers_vz_sc_per_shot,
#endif
#if DW_NDIM >= 2
    int64_t const n_receivers_vy_sc_per_shot,
#endif
    int64_t const n_receivers_vx_sc_per_shot, int64_t const step_ratio,
    int64_t const storage_mode, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, bool const k_requires_grad,
    bool const b_requires_grad, bool const dk_requires_grad,
    bool const db_requires_grad, bool const k_batched, bool const b_batched,
    bool const dk_batched, bool const db_batched,
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
    int64_t const bc_x1, int64_t const n_threads, void* unused) {
  if (nt < 0 || n_shots < 0) return 1;
  if (nt == 0 || n_shots == 0) return 0;
  if (n_threads == 0 || !dw_fits_control_int(n_threads)) return 1;
  /* Optional buffers are required only for active work requesting them.
   * Reject inconsistent native inputs before opening storage files. */
  if (nt > 0 && n_shots > 0) {
    if (step_ratio <= 0) return 1;
    if ((k_requires_grad) && (grad_k == NULL)) return 1;
#if DW_NDIM >= 3
    if ((b_requires_grad) && (grad_bz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((b_requires_grad) && (grad_by == NULL)) return 1;
#endif
    if ((b_requires_grad) && (grad_bx == NULL)) return 1;
    if ((dk_requires_grad) && (grad_dk == NULL)) return 1;
    if ((n_receivers_p_per_shot > 0) && (receivers_i_p == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_per_shot > 0) && (receivers_i_vz == NULL)) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_per_shot > 0) && (receivers_i_vy == NULL)) return 1;
#endif
    if ((n_receivers_vx_per_shot > 0) && (receivers_i_vx == NULL)) return 1;
    if ((n_receivers_p_sc_per_shot > 0) && (receivers_i_p_sc == NULL)) return 1;
#if DW_NDIM >= 3
    if ((n_receivers_vz_sc_per_shot > 0) && (receivers_i_vz_sc == NULL))
      return 1;
#endif
#if DW_NDIM >= 2
    if ((n_receivers_vy_sc_per_shot > 0) && (receivers_i_vy_sc == NULL))
      return 1;
#endif
    if ((n_receivers_vx_sc_per_shot > 0) && (receivers_i_vx_sc == NULL))
      return 1;
    if ((n_sources_p_per_shot > 0) && sources_i_p == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_per_shot > 0) && sources_i_vz == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_per_shot > 0) && sources_i_vy == NULL) return 1;
#endif
    if ((n_sources_vx_per_shot > 0) && sources_i_vx == NULL) return 1;
    if ((n_sources_p_sc_per_shot > 0) && sources_i_p_sc == NULL) return 1;
#if DW_NDIM >= 3
    if ((n_sources_vz_sc_per_shot > 0) && sources_i_vz_sc == NULL) return 1;
#endif
#if DW_NDIM >= 2
    if ((n_sources_vy_sc_per_shot > 0) && sources_i_vy_sc == NULL) return 1;
#endif
    if ((n_sources_vx_sc_per_shot > 0) && sources_i_vx_sc == NULL) return 1;
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
#if DW_NDIM >= 3
  int const per_z = bc[0] == 3;
  /* Axes whose fold writes interior nodes defer the (perturbation-)
   * buoyancy-gradient accumulation until after the fold. */
  int const defer_z = per_z || bc_frx[0] || bc_frx[1];
#endif
#if DW_NDIM >= 2
  int const per_y = bc[2] == 3;
  int const defer_y = per_y || bc_frx[2] || bc_frx[3];
#endif
  int const per_x = bc[4] == 3;
  int const defer_x = per_x || bc_frx[4] || bc_frx[5];
#if DW_NDIM == 3
  int const has_per = per_z || per_y || per_x;
#elif DW_NDIM == 2
  int const has_per = per_y || per_x;
#else
  int const has_per = per_x;
#endif
  DW_DTYPE const stag_w_arr[FD_PAD] = DW_STAG_TAYLOR_W;
  DW_DTYPE const* const stag_w = stag_w_arr;
  (void)has_bc;

  bool const store_needed = k_requires_grad || b_requires_grad ||
                            dk_requires_grad || db_requires_grad;
  bool const store_sc_needed = k_requires_grad || b_requires_grad;

  int64_t shot;
  /* Shots run serially; each time step's adjoint velocity and pressure
   * stencils are spread across threads (see the omp parallel regions below).
   * Model gradients accumulate directly into the shared gradient buffers --
   * each cell is written by exactly one thread, and time steps and shots are
   * barrier-separated, so there is no race. */
  (void)grad_k_thread;
#if DW_NDIM >= 3
  (void)grad_bz_thread;
#endif
#if DW_NDIM >= 2
  (void)grad_by_thread;
#endif
  (void)grad_bx_thread;
  (void)grad_dk_thread;
#if DW_NDIM >= 3
  (void)grad_dbz_thread;
#endif
#if DW_NDIM >= 2
  (void)grad_dby_thread;
#endif
  (void)grad_dbx_thread;
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
    int64_t const src_i_p_sc = shot * n_sources_p_sc_per_shot;
#if DW_NDIM >= 3
    int64_t const src_i_vz_sc = shot * n_sources_vz_sc_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const src_i_vy_sc = shot * n_sources_vy_sc_per_shot;
#endif
    int64_t const src_i_vx_sc = shot * n_sources_vx_sc_per_shot;
    int64_t const rcv_i_p = shot * n_receivers_p_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz = shot * n_receivers_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy = shot * n_receivers_vy_per_shot;
#endif
    int64_t const rcv_i_vx = shot * n_receivers_vx_per_shot;
    int64_t const rcv_i_p_sc = shot * n_receivers_p_sc_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz_sc = shot * n_receivers_vz_sc_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy_sc = shot * n_receivers_vy_sc_per_shot;
#endif
    int64_t const rcv_i_vx_sc = shot * n_receivers_vx_sc_per_shot;

    DW_DTYPE const* __restrict const k_shot = k_batched ? k + si : k;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z_shot =
        b_batched ? buoyancy_z + si : buoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y_shot =
        b_batched ? buoyancy_y + si : buoyancy_y;
#endif
    DW_DTYPE const* __restrict const buoyancy_x_shot =
        b_batched ? buoyancy_x + si : buoyancy_x;
    DW_DTYPE const* __restrict const dk_shot = dk_batched ? dk + si : dk;
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const dbuoyancy_z_shot =
        db_batched ? dbuoyancy_z + si : dbuoyancy_z;
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const dbuoyancy_y_shot =
        db_batched ? dbuoyancy_y + si : dbuoyancy_y;
#endif
    DW_DTYPE const* __restrict const dbuoyancy_x_shot =
        db_batched ? dbuoyancy_x + si : dbuoyancy_x;

    // Background adjoint field.
    DW_DTYPE* __restrict const p_shot = p + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz_shot = vz + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy_shot = vy + si;
#endif
    DW_DTYPE* __restrict const vx_shot = vx + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_shot = phi_z + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_shot = phi_y + si;
#endif
    DW_DTYPE* __restrict const phi_x_shot = phi_x + si;

    // Scattered adjoint field.
    DW_DTYPE* __restrict const psc_shot = psc + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vzsc_shot = vzsc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vysc_shot = vysc + si;
#endif
    DW_DTYPE* __restrict const vxsc_shot = vxsc + si;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const phi_z_sc_shot = phi_z_sc + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const phi_y_sc_shot = phi_y_sc + si;
#endif
    DW_DTYPE* __restrict const phi_x_sc_shot = phi_x_sc + si;

    // Gradients: accumulated directly into the shared buffers (see the
    // race-freedom comment above the shot loop).
    DW_DTYPE* __restrict const grad_k_shot =
        grad_k != NULL ? (grad_k + (k_batched ? si : 0)) : NULL;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_bz_shot =
        grad_bz != NULL ? (grad_bz + (b_batched ? si : 0)) : NULL;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_by_shot =
        grad_by != NULL ? (grad_by + (b_batched ? si : 0)) : NULL;
#endif
    DW_DTYPE* __restrict const grad_bx_shot =
        grad_bx != NULL ? (grad_bx + (b_batched ? si : 0)) : NULL;
    DW_DTYPE* __restrict const grad_dk_shot =
        grad_dk != NULL ? (grad_dk + (dk_batched ? si : 0)) : NULL;
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_dbz_shot =
        grad_dbz != NULL ? (grad_dbz + (db_batched ? si : 0)) : NULL;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_dby_shot =
        grad_dby != NULL ? (grad_dby + (db_batched ? si : 0)) : NULL;
#endif
    DW_DTYPE* __restrict const grad_dbx_shot =
        grad_dbx != NULL ? (grad_dbx + (db_batched ? si : 0)) : NULL;

    int64_t t;

    int storage_err = 0;
    FILE* fp_k_grad = NULL;
#if DW_NDIM >= 3
    FILE* fp_bz_grad = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_by_grad = NULL;
#endif
    FILE* fp_bx_grad = NULL;
    FILE* fp_k_grad_sc = NULL;
#if DW_NDIM >= 3
    FILE* fp_bz_grad_sc = NULL;
#endif
#if DW_NDIM >= 2
    FILE* fp_by_grad_sc = NULL;
#endif
    FILE* fp_bx_grad_sc = NULL;

    OPEN_FILE_READ(k_grad, store_needed)
#if DW_NDIM >= 3
    OPEN_FILE_READ(bz_grad, store_needed)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_READ(by_grad, store_needed)
#endif
    OPEN_FILE_READ(bx_grad, store_needed)
    OPEN_FILE_READ(k_grad_sc, store_sc_needed)
#if DW_NDIM >= 3
    OPEN_FILE_READ(bz_grad_sc, store_sc_needed)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_READ(by_grad_sc, store_sc_needed)
#endif
    OPEN_FILE_READ(bx_grad_sc, store_sc_needed)

    for (t = start_t - 1; t >= start_t - nt; --t) {
      SETUP_STORE_LOAD(k_grad, store_needed)
#if DW_NDIM >= 3
      SETUP_STORE_LOAD(bz_grad, store_needed)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_LOAD(by_grad, store_needed)
#endif
      SETUP_STORE_LOAD(bx_grad, store_needed)
      SETUP_STORE_LOAD(k_grad_sc, store_sc_needed)
#if DW_NDIM >= 3
      SETUP_STORE_LOAD(bz_grad_sc, store_sc_needed)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_LOAD(by_grad_sc, store_sc_needed)
#endif
      SETUP_STORE_LOAD(bx_grad_sc, store_sc_needed)

#if DW_NDIM >= 3
      DW_DTYPE* __restrict const psi_z_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_zn : psi_z) + si;
      DW_DTYPE* __restrict const psi_zn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_z : psi_zn) + si;
      DW_DTYPE* __restrict const psi_z_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_zn_sc : psi_z_sc) + si;
      DW_DTYPE* __restrict const psi_zn_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_z_sc : psi_zn_sc) + si;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const psi_y_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_yn : psi_y) + si;
      DW_DTYPE* __restrict const psi_yn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_y : psi_yn) + si;
      DW_DTYPE* __restrict const psi_y_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_yn_sc : psi_y_sc) + si;
      DW_DTYPE* __restrict const psi_yn_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_y_sc : psi_yn_sc) + si;
#endif
      DW_DTYPE* __restrict const psi_x_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_xn : psi_x) + si;
      DW_DTYPE* __restrict const psi_xn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_x : psi_xn) + si;
      DW_DTYPE* __restrict const psi_x_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_xn_sc : psi_x_sc) + si;
      DW_DTYPE* __restrict const psi_xn_sc_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_x_sc : psi_xn_sc) + si;

      bool const k_requires_grad_t = k_requires_grad && ((t % step_ratio) == 0);
      bool const b_requires_grad_t = b_requires_grad && ((t % step_ratio) == 0);
      bool const dk_requires_grad_t =
          dk_requires_grad && ((t % step_ratio) == 0);
      bool const db_requires_grad_t =
          db_requires_grad && ((t % step_ratio) == 0);

      // Periodic faces: fold both adjoint pressure fields' guard cells back
      // into the interior nodes the forward fill copied them from (the exact
      // transpose of the fill). The fold modifies interior seam nodes, so it
      // must precede the source-gradient reads below.
      if (has_per) {
        ADJ_FOLD_P_BC(p_shot);
        ADJ_FOLD_P_BC(psc_shot);
      }
      if (has_frx) {
        ADJ_FOLD_P_BC_FRX(p_shot);
        ADJ_FOLD_P_BC_FRX(psc_shot);
      }

      // Record source gradients from the adjoint pressure (background then
      // scattered) at source locations.
      if (n_sources_p_per_shot > 0) {
        record_from_wavefield(
            p_shot, sources_i_p + src_i_p,
            grad_f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
            n_sources_p_per_shot);
      }
      if (n_sources_p_sc_per_shot > 0) {
        record_from_wavefield(
            psc_shot, sources_i_p_sc + src_i_p_sc,
            grad_f_p_sc + t * n_shots * n_sources_p_sc_per_shot + src_i_p_sc,
            n_sources_p_sc_per_shot);
      }

      // Adjoint guard fill of the pressure field (read by the adjoint
      // velocity update), on both fields independently, on the free/rigid
      // faces: the same image the forward applies, which realises the exact
      // transpose on a reflecting face. No-op for all-PML.
      if (has_fr) {
        FILL_P_BC_FR(p_shot);
        FILL_P_BC_FR(psc_shot);
      }

      // Update Adjoint Velocities (background coupled + scattered ordinary),
      // and accumulate grad_B_d / grad_dB_d.
#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
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
        for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < 3; ++pml_x) {
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
                        p_shot, psc_shot, k_shot, dk_shot, vz_shot, vzsc_shot,
                        phi_z_shot, phi_z_sc_shot, psi_z_t, psi_zn_t,
                        psi_z_sc_t, psi_zn_sc_t, buoyancy_z_shot[i],
                        dbuoyancy_z_shot[i], grad_bz_shot, grad_dbz_shot,
                        bz_grad_store_1_t, bz_grad_sc_store_1_t, azh, bz, dt,
                        step_ratio, b_requires_grad_t && !defer_z,
                        db_requires_grad_t && !defer_z, i, i, ny, nx, z,
                        pml_z != 1, rdz);
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
                        p_shot, psc_shot, k_shot, dk_shot, vy_shot, vysc_shot,
                        phi_y_shot, phi_y_sc_shot, psi_y_t, psi_yn_t,
                        psi_y_sc_t, psi_yn_sc_t, buoyancy_y_shot[i],
                        dbuoyancy_y_shot[i], grad_by_shot, grad_dby_shot,
                        by_grad_store_1_t, by_grad_sc_store_1_t, ayh, by, dt,
                        step_ratio, b_requires_grad_t && !defer_y,
                        db_requires_grad_t && !defer_y, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
                        nx, y, pml_y != 1, rdy);
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
                        p_shot, psc_shot, k_shot, dk_shot, vx_shot, vxsc_shot,
                        phi_x_shot, phi_x_sc_shot, psi_x_t, psi_xn_t,
                        psi_x_sc_t, psi_xn_sc_t, buoyancy_x_shot[i],
                        dbuoyancy_x_shot[i], grad_bx_shot, grad_dbx_shot,
                        bx_grad_store_1_t, bx_grad_sc_store_1_t, axh, bx, dt,
                        step_ratio, b_requires_grad_t && !defer_x,
                        db_requires_grad_t && !defer_x, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
                        x, pml_x != 1, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Periodic faces: accumulate the adjoint-stencil contributions that
      // fall on both adjoint velocity sets' guard cells (and the slaved
      // extra high interior nodes), then fold them back into the interior
      // (the exact transpose of the forward velocity fill). The background
      // set's Born-coupled payload (K*P_adj + dK*P_adj_sc) is composed as
      // two single-payload gather calls. The folds modify interior seam
      // nodes, so they must precede the source-gradient reads below, and
      // the (perturbation-)buoyancy-gradient accumulations for periodic
      // axes (suppressed in the backward_v*_cell calls above) run only
      // after them.
      if (has_per) {
#if DW_NDIM == 3
        ADJ_GATHER_V_GUARDS_SET(vz_shot, vy_shot, vx_shot, p_shot, k_shot);
        ADJ_GATHER_V_GUARDS_SET(vz_shot, vy_shot, vx_shot, psc_shot, dk_shot);
        ADJ_GATHER_V_GUARDS_SET(vzsc_shot, vysc_shot, vxsc_shot, psc_shot,
                                k_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_V_GUARDS_SET(vy_shot, vx_shot, p_shot, k_shot);
        ADJ_GATHER_V_GUARDS_SET(vy_shot, vx_shot, psc_shot, dk_shot);
        ADJ_GATHER_V_GUARDS_SET(vysc_shot, vxsc_shot, psc_shot, k_shot);
#else
        ADJ_GATHER_V_GUARDS_SET(vx_shot, p_shot, k_shot);
        ADJ_GATHER_V_GUARDS_SET(vx_shot, psc_shot, dk_shot);
        ADJ_GATHER_V_GUARDS_SET(vxsc_shot, psc_shot, k_shot);
#endif
        ADJ_FOLD_V_BC_BG();
        ADJ_FOLD_V_BC_SC();
      }
      if (has_frx) {
#if DW_NDIM == 3
        ADJ_GATHER_V_GUARDS_SET_FRX(vz_shot, vy_shot, vx_shot, p_shot, k_shot);
        ADJ_GATHER_V_GUARDS_SET_FRX(vz_shot, vy_shot, vx_shot, psc_shot,
                                    dk_shot);
        ADJ_GATHER_V_GUARDS_SET_FRX(vzsc_shot, vysc_shot, vxsc_shot, psc_shot,
                                    k_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_V_GUARDS_SET_FRX(vy_shot, vx_shot, p_shot, k_shot);
        ADJ_GATHER_V_GUARDS_SET_FRX(vy_shot, vx_shot, psc_shot, dk_shot);
        ADJ_GATHER_V_GUARDS_SET_FRX(vysc_shot, vxsc_shot, psc_shot, k_shot);
#else
        ADJ_GATHER_V_GUARDS_SET_FRX(vx_shot, p_shot, k_shot);
        ADJ_GATHER_V_GUARDS_SET_FRX(vx_shot, psc_shot, dk_shot);
        ADJ_GATHER_V_GUARDS_SET_FRX(vxsc_shot, psc_shot, k_shot);
#endif
        ADJ_FOLD_V_BC_BG_FRX();
        ADJ_FOLD_V_BC_SC_FRX();
      }

      // Record velocity-component source gradients (background then scattered).
#if DW_NDIM >= 3
      if (n_sources_vz_per_shot > 0) {
        record_from_wavefield(
            vz_shot, sources_i_vz + src_i_vz,
            grad_f_vz + t * n_shots * n_sources_vz_per_shot + src_i_vz,
            n_sources_vz_per_shot);
      }
      if (n_sources_vz_sc_per_shot > 0) {
        record_from_wavefield(
            vzsc_shot, sources_i_vz_sc + src_i_vz_sc,
            grad_f_vz_sc + t * n_shots * n_sources_vz_sc_per_shot + src_i_vz_sc,
            n_sources_vz_sc_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_sources_vy_per_shot > 0) {
        record_from_wavefield(
            vy_shot, sources_i_vy + src_i_vy,
            grad_f_vy + t * n_shots * n_sources_vy_per_shot + src_i_vy,
            n_sources_vy_per_shot);
      }
      if (n_sources_vy_sc_per_shot > 0) {
        record_from_wavefield(
            vysc_shot, sources_i_vy_sc + src_i_vy_sc,
            grad_f_vy_sc + t * n_shots * n_sources_vy_sc_per_shot + src_i_vy_sc,
            n_sources_vy_sc_per_shot);
      }
#endif
      if (n_sources_vx_per_shot > 0) {
        record_from_wavefield(
            vx_shot, sources_i_vx + src_i_vx,
            grad_f_vx + t * n_shots * n_sources_vx_per_shot + src_i_vx,
            n_sources_vx_per_shot);
      }
      if (n_sources_vx_sc_per_shot > 0) {
        record_from_wavefield(
            vxsc_shot, sources_i_vx_sc + src_i_vx_sc,
            grad_f_vx_sc + t * n_shots * n_sources_vx_sc_per_shot + src_i_vx_sc,
            n_sources_vx_sc_per_shot);
      }

      // Adjoint guard fill of the velocity field (read by the adjoint
      // pressure update), on both fields independently, on the free/rigid
      // faces: the transpose-realising image, matching the forward. No-op
      // for all-PML.
      if (has_fr) {
        FILL_V_BC_BG_FR();
        FILL_V_BC_SC_FR();
      }

      // Periodic axes: the buoyancy and perturbation-buoyancy gradients
      // must use the post-fold adjoint velocities (the folds write into
      // interior seam nodes), so they are accumulated here instead of
      // inside backward_v*_cell.
      if ((b_requires_grad_t || db_requires_grad_t) && (has_per || has_frx)) {
#if DW_NDIM == 3
        int64_t gz, gy, gx;
        if (defer_z)
          for (gz = FD_PAD; gz < nz - FD_PAD; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                if (db_requires_grad_t)
                  grad_dbz_shot[gi] -= dt * vzsc_shot[gi] *
                                       bz_grad_store_1_t[gi] *
                                       (DW_DTYPE)step_ratio;
                if (b_requires_grad_t)
                  grad_bz_shot[gi] -=
                      dt *
                      (vz_shot[gi] * bz_grad_store_1_t[gi] +
                       vzsc_shot[gi] * bz_grad_sc_store_1_t[gi]) *
                      (DW_DTYPE)step_ratio;
              }
        if (defer_y)
          for (gz = FD_PAD; gz < nz - FD_PAD + 1; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                if (db_requires_grad_t)
                  grad_dby_shot[gi] -= dt * vysc_shot[gi] *
                                       by_grad_store_1_t[gi] *
                                       (DW_DTYPE)step_ratio;
                if (b_requires_grad_t)
                  grad_by_shot[gi] -=
                      dt *
                      (vy_shot[gi] * by_grad_store_1_t[gi] +
                       vysc_shot[gi] * by_grad_sc_store_1_t[gi]) *
                      (DW_DTYPE)step_ratio;
              }
        if (defer_x)
          for (gz = FD_PAD; gz < nz - FD_PAD + 1; ++gz)
            for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
              for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
                int64_t const gi = (gz * ny + gy) * nx + gx;
                if (db_requires_grad_t)
                  grad_dbx_shot[gi] -= dt * vxsc_shot[gi] *
                                       bx_grad_store_1_t[gi] *
                                       (DW_DTYPE)step_ratio;
                if (b_requires_grad_t)
                  grad_bx_shot[gi] -=
                      dt *
                      (vx_shot[gi] * bx_grad_store_1_t[gi] +
                       vxsc_shot[gi] * bx_grad_sc_store_1_t[gi]) *
                      (DW_DTYPE)step_ratio;
              }
#elif DW_NDIM == 2
        int64_t gy, gx;
        if (defer_y)
          for (gy = FD_PAD; gy < ny - FD_PAD; ++gy)
            for (gx = FD_PAD; gx < nx - FD_PAD + 1; ++gx) {
              int64_t const gi = gy * nx + gx;
              if (db_requires_grad_t)
                grad_dby_shot[gi] -= dt * vysc_shot[gi] *
                                     by_grad_store_1_t[gi] *
                                     (DW_DTYPE)step_ratio;
              if (b_requires_grad_t)
                grad_by_shot[gi] -= dt *
                                    (vy_shot[gi] * by_grad_store_1_t[gi] +
                                     vysc_shot[gi] * by_grad_sc_store_1_t[gi]) *
                                    (DW_DTYPE)step_ratio;
            }
        if (defer_x)
          for (gy = FD_PAD; gy < ny - FD_PAD + 1; ++gy)
            for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
              int64_t const gi = gy * nx + gx;
              if (db_requires_grad_t)
                grad_dbx_shot[gi] -= dt * vxsc_shot[gi] *
                                     bx_grad_store_1_t[gi] *
                                     (DW_DTYPE)step_ratio;
              if (b_requires_grad_t)
                grad_bx_shot[gi] -= dt *
                                    (vx_shot[gi] * bx_grad_store_1_t[gi] +
                                     vxsc_shot[gi] * bx_grad_sc_store_1_t[gi]) *
                                    (DW_DTYPE)step_ratio;
            }
#else
        int64_t gx;
        if (defer_x)
          for (gx = FD_PAD; gx < nx - FD_PAD; ++gx) {
            if (db_requires_grad_t)
              grad_dbx_shot[gx] -= dt * vxsc_shot[gx] * bx_grad_store_1_t[gx] *
                                   (DW_DTYPE)step_ratio;
            if (b_requires_grad_t)
              grad_bx_shot[gx] -= dt *
                                  (vx_shot[gx] * bx_grad_store_1_t[gx] +
                                   vxsc_shot[gx] * bx_grad_sc_store_1_t[gx]) *
                                  (DW_DTYPE)step_ratio;
          }
#endif
      }

      // Update Adjoint Pressure (background coupled + scattered ordinary),
      // accumulate grad_K / grad_dK.
#ifdef _OPENMP
#pragma omp parallel num_threads((int)n_threads)
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
        for (pml_z = 0; pml_z < 3; ++pml_z)
#endif
#if DW_NDIM >= 2
#pragma GCC unroll 3
          for (pml_y = 0; pml_y < 3; ++pml_y)
#endif
#pragma GCC unroll 3
            for (pml_x = 0; pml_x < 3; ++pml_x) {
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
                        p_shot, psc_shot,
#if DW_NDIM >= 3
                        vz_shot, vzsc_shot, phi_z_shot, phi_z_sc_shot,
                        buoyancy_z_shot, dbuoyancy_z_shot, psi_z_t, psi_z_sc_t,
#endif
#if DW_NDIM >= 2
                        vy_shot, vysc_shot, phi_y_shot, phi_y_sc_shot,
                        buoyancy_y_shot, dbuoyancy_y_shot, psi_y_t, psi_y_sc_t,
#endif
                        vx_shot, vxsc_shot, phi_x_shot, phi_x_sc_shot,
                        buoyancy_x_shot, dbuoyancy_x_shot, psi_x_t, psi_x_sc_t,
                        k_shot[i], dk_shot[i], grad_k_shot, grad_dk_shot,
                        k_grad_store_1_t, k_grad_sc_store_1_t, dt, step_ratio,
                        k_requires_grad_t, dk_requires_grad_t, i, i,
#if DW_NDIM >= 3
                        ny,
#endif
#if DW_NDIM >= 2
                        nx,
#endif
#if DW_NDIM >= 3
                        z, pml_z != 1, az, bzh, rdz,
#endif
#if DW_NDIM >= 2
                        y, pml_y != 1, ay, byh, rdy,
#endif
                        x, pml_x != 1, ax, bxh, rdx);
                  }
            }
        dw_ftz_daz_exit(dw_ftz_daz_prev_state);
      }

      // Periodic faces: accumulate the adjoint-stencil contributions that
      // fall on both adjoint pressure fields' guard cells (folded into the
      // interior at the top of the next iteration). The background field's
      // Born-coupled payload (B*V_adj + dB*V_adj_sc) is composed as two
      // single-payload gather calls.
      if (has_per) {
#if DW_NDIM == 3
        ADJ_GATHER_P_GUARDS_SET(p_shot, vz_shot, buoyancy_z_shot, vy_shot,
                                buoyancy_y_shot, vx_shot, buoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET(p_shot, vzsc_shot, dbuoyancy_z_shot, vysc_shot,
                                dbuoyancy_y_shot, vxsc_shot, dbuoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET(psc_shot, vzsc_shot, buoyancy_z_shot, vysc_shot,
                                buoyancy_y_shot, vxsc_shot, buoyancy_x_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_P_GUARDS_SET(p_shot, vy_shot, buoyancy_y_shot, vx_shot,
                                buoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET(p_shot, vysc_shot, dbuoyancy_y_shot, vxsc_shot,
                                dbuoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET(psc_shot, vysc_shot, buoyancy_y_shot, vxsc_shot,
                                buoyancy_x_shot);
#else
        ADJ_GATHER_P_GUARDS_SET(p_shot, vx_shot, buoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET(p_shot, vxsc_shot, dbuoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET(psc_shot, vxsc_shot, buoyancy_x_shot);
#endif
      }
      if (has_frx) {
#if DW_NDIM == 3
        ADJ_GATHER_P_GUARDS_SET_FRX(p_shot, vz_shot, buoyancy_z_shot, vy_shot,
                                    buoyancy_y_shot, vx_shot, buoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET_FRX(p_shot, vzsc_shot, dbuoyancy_z_shot,
                                    vysc_shot, dbuoyancy_y_shot, vxsc_shot,
                                    dbuoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET_FRX(psc_shot, vzsc_shot, buoyancy_z_shot,
                                    vysc_shot, buoyancy_y_shot, vxsc_shot,
                                    buoyancy_x_shot);
#elif DW_NDIM == 2
        ADJ_GATHER_P_GUARDS_SET_FRX(p_shot, vy_shot, buoyancy_y_shot, vx_shot,
                                    buoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET_FRX(p_shot, vysc_shot, dbuoyancy_y_shot,
                                    vxsc_shot, dbuoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET_FRX(psc_shot, vysc_shot, buoyancy_y_shot,
                                    vxsc_shot, buoyancy_x_shot);
#else
        ADJ_GATHER_P_GUARDS_SET_FRX(p_shot, vx_shot, buoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET_FRX(p_shot, vxsc_shot, dbuoyancy_x_shot);
        ADJ_GATHER_P_GUARDS_SET_FRX(psc_shot, vxsc_shot, buoyancy_x_shot);
#endif
      }

      // Inject adjoint sources (data residuals): background then scattered.
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
      if (n_receivers_p_sc_per_shot > 0) {
        add_to_wavefield(
            psc_shot, receivers_i_p_sc + rcv_i_p_sc,
            grad_r_p_sc + t * n_shots * n_receivers_p_sc_per_shot + rcv_i_p_sc,
            n_receivers_p_sc_per_shot);
      }
#if DW_NDIM >= 3
      if (n_receivers_vz_sc_per_shot > 0) {
        add_to_wavefield(vzsc_shot, receivers_i_vz_sc + rcv_i_vz_sc,
                         grad_r_vz_sc +
                             t * n_shots * n_receivers_vz_sc_per_shot +
                             rcv_i_vz_sc,
                         n_receivers_vz_sc_per_shot);
      }
#endif
#if DW_NDIM >= 2
      if (n_receivers_vy_sc_per_shot > 0) {
        add_to_wavefield(vysc_shot, receivers_i_vy_sc + rcv_i_vy_sc,
                         grad_r_vy_sc +
                             t * n_shots * n_receivers_vy_sc_per_shot +
                             rcv_i_vy_sc,
                         n_receivers_vy_sc_per_shot);
      }
#endif
      if (n_receivers_vx_sc_per_shot > 0) {
        add_to_wavefield(vxsc_shot, receivers_i_vx_sc + rcv_i_vx_sc,
                         grad_r_vx_sc +
                             t * n_shots * n_receivers_vx_sc_per_shot +
                             rcv_i_vx_sc,
                         n_receivers_vx_sc_per_shot);
      }
    }

  storage_cleanup:
    CLOSE_FILE(k_grad)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad)
#endif
    CLOSE_FILE(bx_grad)
    CLOSE_FILE(k_grad_sc)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad_sc)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad_sc)
#endif
    CLOSE_FILE(bx_grad_sc)
    if (storage_err) return 1;
  }

  return 0;
}

/*
 * Acoustic wave equation propagator (variable density, staggered grid).
 *
 * PHYSICS
 * -------
 * The first-order velocity-pressure form of the acoustic wave equation in a
 * medium of variable density rho and adiabatic bulk modulus K = rho c^2, with
 * B = 1/rho the buoyancy:
 *
 *     rho dv/dt = -grad p          (momentum conservation)
 *     dp/dt = -K div(v)            (mass conservation / continuity)
 *
 * Eliminating v recovers the second-order form rho div(B grad p) = (1/c^2)
 * d2p/dt2, so with constant density this reduces to the scalar propagator's
 * equation; the first-order form is kept because it takes the density
 * variation exactly, without the grad(rho) term the second-order form would
 * need. See docs/acoustic.rst for the full derivation.
 *
 * DISCRETISATION (staggered leapfrog). p and K live at the integer grid nodes
 * at integer time n; each velocity component and its buoyancy live half a cell
 * along that component's own axis and half a step in time, at n+1/2. One step
 * is therefore
 *
 *     v^{n+1/2} = v^{n-1/2} - dt * B * grad_h(p^n)
 *     p^{n+1}   = p^n       - dt * K * div_h(v^{n+1/2})
 *
 * with grad_h/div_h the staggered FD operators of staggered_grid.h (DIFF*H1
 * and DIFF*1 respectively, to the requested DW_ACCURACY). Absorbing faces add
 * a CPML memory variable per axis -- psi on the velocity nodes, phi on the
 * pressure nodes -- each obeying the standard recursion psi = b*d + a*psi with
 * the a/b profile arrays supplied by Python, and each added to its axis's
 * derivative before it is used; the whole PML apparatus is inert (b = 0) in
 * the interior, which is why the grid is swept as 3^DW_NDIM separately-bounded
 * regions (see FUNC(forward)) rather than branching per cell. Reflecting and
 * periodic faces are handled instead by guard-ring image fills; see
 * staggered_bc.h and docs/boundaries.rst.
 *
 * ADJOINT (discrete transpose of one step; Pbar and Vbar are the adjoints of
 * p^{n+1} and v^{n+1/2}). Transposing the two updates above, and using that
 * the transpose of a staggered forward difference is minus the corresponding
 * staggered backward difference, gives
 *
 *     Vbar += -grad_h^T(dt*K*Pbar) = +div_h(dt*K*Pbar)
 *     Pbar += -div_h^T(dt*B*Vbar)  = +grad_h(dt*B*Vbar)
 *
 * i.e. the same stencils with the model factor moved inside the derivative --
 * which is exactly what the VZ_ADJOINT/P_ADJOINT_* macros in
 * acoustic_kernels.h form, with their _PML twins carrying the transposed CPML
 * recursions. The model gradients are the products of each field's adjoint
 * with the corresponding forward derivative, which the forward pass stores
 * every step_ratio steps:
 *
 *     grad_K += -dt * Pbar * div_h(v)      (stored by FUNC(forward_p_cell))
 *     grad_B += -dt * Vbar * grad_h(p)     (stored by FUNC(forward_v*_cell))
 *
 * Both carry a step_ratio factor to account for that decimation. The whole
 * transpose is checked numerically by tests/test_adjoint_dot_product.py.
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
  acoustic_iso_##ndim##d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, ndim, accuracy, dtype, device) \
  CAT_I(name, ndim, accuracy, dtype, device)
#define ACOUSTIC_VARIANT_I(name) name
#define ACOUSTIC_VARIANT(name) ACOUSTIC_VARIANT_I(name)
#define FUNC(name) \
  CAT(ACOUSTIC_VARIANT(name), DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* ND_INDEX, K/BUOYANCY_Z/Y/X, P/VZ/VY/VX, PHI_Z/Y/X/PSI_Z/Y/X, the
 * VZ_ADJOINT/VY_ADJOINT/VX_ADJOINT/P_ADJOINT_Z/Y/X (+_PML) family, and the
 * FUNC(forward_vz_cell)/.../FUNC(backward_p_cell) per-grid-point stencil
 * update functions live in acoustic_kernels.h, shared verbatim with
 * acoustic.cu (see that header for the full explanation). p_shot/vz_shot/
 * .../psi_z_t/.../k_shot/buoyancy_z_shot/... below are this file's names for
 * the (per-shot-offset) arrays/values passed as the p/vz/.../psi_z/.../
 * k_shot/buoyancy_z_shot/... parameters of those functions. */
#include "acoustic_kernels.h"

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

/* fill_p_bc_line/fill_v_bc_line (first-class reflecting/periodic boundary
 * faces on the staggered acoustic grid) live in line_bc.h, shared verbatim
 * across every propagator on this staggered layout -- see that header for
 * the full explanation. */
#include "line_bc.h"

/* The whole-face boundary appliers (FUNC(fill_p_bc) through
 * FUNC(adj_gather_p_guards_frx)) and the FILL_/ADJ_ dispatch macros that
 * drive them live in staggered_bc.h, shared verbatim with viscoacoustic.c --
 * see that header for the face-enumeration rules and the include contract.
 * The only propagator-specific part is the payload of the adjoint-velocity
 * guard gather; for acoustic that is the plain dt*K*Pbar of
 * FUNC(backward_v*_cell), so the bulk modulus is the whole payload and there
 * are no trailing extras. */
#define DW_VGUARD_MODEL_PARAMS DW_DTYPE const* __restrict const k_shot
#define DW_VGUARD_MODEL_ARGS k_shot
#define DW_VGUARD_MODEL_SHOT_ARGS k_shot
#define DW_VGUARD_EXTRA_PARAMS
#define DW_VGUARD_EXTRA_ARGS
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
DW_TARGET_CLONES int FUNC(forward)(
    DW_DTYPE const* __restrict const k,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y,
#endif
    DW_DTYPE const* __restrict const buoyancy_x,
    DW_DTYPE const* __restrict const f_p,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const f_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const f_vy,
#endif
    DW_DTYPE const* __restrict const f_vx, DW_DTYPE* __restrict const p,
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
    DW_DTYPE* __restrict const psi_x, DW_DTYPE* __restrict const k_grad_store_1,
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
    DW_DTYPE* __restrict const r_p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const r_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const r_vy,
#endif
    DW_DTYPE* __restrict const r_vx,
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
    int64_t const* __restrict const receivers_i_p,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy,
#endif
    int64_t const* __restrict const receivers_i_vx,
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
    size_t const shot_bytes_comp, bool const k_requires_grad,
    bool const b_requires_grad, bool const k_batched, bool const b_batched,
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
    int64_t const rcv_i_p = shot * n_receivers_p_per_shot;
#if DW_NDIM >= 3
    int64_t const rcv_i_vz = shot * n_receivers_vz_per_shot;
#endif
#if DW_NDIM >= 2
    int64_t const rcv_i_vy = shot * n_receivers_vy_per_shot;
#endif
    int64_t const rcv_i_vx = shot * n_receivers_vx_per_shot;

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

#if DW_NDIM >= 3
    DW_DTYPE* __restrict const psi_z_t = psi_z + si;
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const psi_y_t = psi_y + si;
#endif
    DW_DTYPE* __restrict const psi_x_t = psi_x + si;

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

    OPEN_FILE_WRITE(k_grad, k_requires_grad)
#if DW_NDIM >= 3
    OPEN_FILE_WRITE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_WRITE(by_grad, b_requires_grad)
#endif
    OPEN_FILE_WRITE(bx_grad, b_requires_grad)

    for (t = start_t; t < start_t + nt; ++t) {
      SETUP_STORE_SAVE(k_grad, k_requires_grad)
#if DW_NDIM >= 3
      SETUP_STORE_SAVE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_SAVE(by_grad, b_requires_grad)
#endif
      SETUP_STORE_SAVE(bx_grad, b_requires_grad)

      bool const k_requires_grad_t = k_requires_grad && ((t % step_ratio) == 0);
      bool const b_requires_grad_t = b_requires_grad && ((t % step_ratio) == 0);

      // Refill the pressure guard ring (read by the velocity update); this also
      // holds free-surface pressure nodes to zero before recording (no-op when
      // every face is PML).
      if (has_bc) FILL_P_BC(p_shot);

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

          // Update Pressure (P)
          // p = p - dt * K * (div(v) + phi)

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
                        vx_shot, p_shot,
#if DW_NDIM >= 3
                        phi_z_shot,
#endif
#if DW_NDIM >= 2
                        phi_y_shot,
#endif
                        phi_x_shot, k_grad_store_1_t, k_shot[i], dt,
                        k_requires_grad_t, i,
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

      SAVE_SNAPSHOT(k_grad, k_requires_grad_t)
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

/* The trailing "void *unused" parameter is dead: it is part of the frozen
 * ctypes ABI every propagator's forward/backward entry point shares (see
 * tests/test_abi_stability.py) and must keep its slot. */
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
    DW_DTYPE const* __restrict const grad_r_p,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const grad_r_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const grad_r_vy,
#endif
    DW_DTYPE const* __restrict const grad_r_vx, DW_DTYPE* __restrict const p,
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
    DW_DTYPE* __restrict const psi_xn,

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
    DW_DTYPE* __restrict const grad_f_p,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_f_vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_f_vy,
#endif
    DW_DTYPE* __restrict const grad_f_vx, DW_DTYPE* __restrict const grad_k,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_by,
#endif
    DW_DTYPE* __restrict const grad_bx,
    DW_DTYPE* __restrict const grad_k_thread,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_bz_thread,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const grad_by_thread,
#endif
    DW_DTYPE* __restrict const grad_bx_thread,
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
    int64_t const* __restrict const receivers_i_p,
#if DW_NDIM >= 3
    int64_t const* __restrict const receivers_i_vz,
#endif
#if DW_NDIM >= 2
    int64_t const* __restrict const receivers_i_vy,
#endif
    int64_t const* __restrict const receivers_i_vx,
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
    size_t const shot_bytes_comp, bool const k_requires_grad,
    bool const b_requires_grad, bool const k_batched, bool const b_batched,
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
   * acoustic_kernels.h). Free/rigid faces: re-applying the forward fill to
   * the adjoint fields realises the transpose (through the mirror-padded
   * models) only for the per-axis (free-low, rigid-high) pairing, which
   * keeps that historical re-fill path (bc_fr); every other free/rigid
   * combination gets the exact fr fold/gather treatment (bc_frx). */
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
  DW_DTYPE const* const stag_w = stag_w_arr;
  /* has_bc is computed (identically to FUNC(forward)) but unused here: the
     backward pass drives the guard rings through has_fr/has_per/has_frx
     instead, which partition the same faces. */
  (void)has_bc;

  int64_t shot;
  /* The grad_*_thread scratch buffers are a dead ABI slot: the parallelism is
     intra-shot spatial, so each gradient cell is written by exactly one
     thread and the per-thread reduction buffers they were sized for are never
     needed. They stay in the signature because the ABI is frozen. */
  (void)grad_k_thread;
#if DW_NDIM >= 3
  (void)grad_bz_thread;
#endif
#if DW_NDIM >= 2
  (void)grad_by_thread;
#endif
  (void)grad_bx_thread;
  /* Shots run serially; each time step's adjoint stencils are spread across
   * threads (see the omp parallel regions below). Model gradients accumulate
   * directly into the shared gradient buffers -- each cell is written by
   * exactly one thread, and time steps/shots are barrier-separated. */
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

    // In backward pass, p is the adjoint pressure field (q)
    // vz, vy, vx are adjoint velocity fields
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

    // Gradients (accumulated directly into the shared gradient buffers)
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

    OPEN_FILE_READ(k_grad, k_requires_grad)
#if DW_NDIM >= 3
    OPEN_FILE_READ(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_READ(by_grad, b_requires_grad)
#endif
    OPEN_FILE_READ(bx_grad, b_requires_grad)

    for (t = start_t - 1; t >= start_t - nt; --t) {
      SETUP_STORE_LOAD(k_grad, k_requires_grad)
#if DW_NDIM >= 3
      SETUP_STORE_LOAD(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
      SETUP_STORE_LOAD(by_grad, b_requires_grad)
#endif
      SETUP_STORE_LOAD(bx_grad, b_requires_grad)

#if DW_NDIM >= 3
      DW_DTYPE* __restrict const psi_z_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_zn : psi_z) + si;
      DW_DTYPE* __restrict const psi_zn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_z : psi_zn) + si;
#endif
#if DW_NDIM >= 2
      DW_DTYPE* __restrict const psi_y_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_yn : psi_y) + si;
      DW_DTYPE* __restrict const psi_yn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_y : psi_yn) + si;
#endif
      DW_DTYPE* __restrict const psi_x_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_xn : psi_x) + si;
      DW_DTYPE* __restrict const psi_xn_t =
          (((uint64_t)(start_t - 1 - t) & 1u) ? psi_x : psi_xn) + si;

      bool const k_requires_grad_t = k_requires_grad && ((t % step_ratio) == 0);
      bool const b_requires_grad_t = b_requires_grad && ((t % step_ratio) == 0);

      // Periodic faces: fold the adjoint pressure's guard cells back into
      // the interior nodes the forward fill copied them from (the exact
      // transpose of the fill). The fold modifies interior seam nodes, so it
      // must precede the source-gradient read below. Free/rigid faces other
      // than the (free-low, rigid-high) pairing get their own exact fold
      // (mirror/sign-flip transpose) at the same position.
      if (has_per) ADJ_FOLD_P_BC(p_shot);
      if (has_frx) ADJ_FOLD_P_BC_FRX(p_shot);

      if (n_sources_p_per_shot > 0) {
        record_from_wavefield(
            p_shot, sources_i_p + src_i_p,
            grad_f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
            n_sources_p_per_shot);
      }

      // Adjoint guard fill of the pressure field (read by the adjoint velocity
      // update) on (free-low, rigid-high) axes: the same image the forward
      // applies, which realises the exact transpose on such a reflecting
      // (pml_width=0) axis through the mirror-padded models. No-op for
      // all-PML and for the fold/gather-handled axes.
      if (has_fr) FILL_P_BC_FR(p_shot);

      // Update Adjoint Velocities (V)
      // v = v + dt * (grad((1+b)Kp) - grad(b*phi))
      // psi = a*psi - dt*B*a*v

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
                        p_shot, k_shot, vz_shot, phi_z_shot, psi_z_t, psi_zn_t,
                        buoyancy_z_shot[i], grad_bz_shot, bz_grad_store_1_t,
                        azh, bz, dt, step_ratio, b_requires_grad_t && !defer_z,
                        i, i, ny, nx, z, pml_z != DW_PML_INTERIOR_REGION, rdz);
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
                        p_shot, k_shot, vy_shot, phi_y_shot, psi_y_t, psi_yn_t,
                        buoyancy_y_shot[i], grad_by_shot, by_grad_store_1_t,
                        ayh, by, dt, step_ratio, b_requires_grad_t && !defer_y,
                        i, i,
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
                        p_shot, k_shot, vx_shot, phi_x_shot, psi_x_t, psi_xn_t,
                        buoyancy_x_shot[i], grad_bx_shot, bx_grad_store_1_t,
                        axh, bx, dt, step_ratio, b_requires_grad_t && !defer_x,
                        i, i,
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

      // Adjoint guard fill of the velocity field (read by the adjoint pressure
      // update) on the free/rigid faces: the transpose-realising image,
      // matching the forward. No-op for all-PML.
      if (has_fr) FILL_V_BC_FR();

      // Fold-handled axes (periodic or exact-fr): the buoyancy gradient must
      // use the post-fold adjoint velocity (the fold writes into interior
      // nodes), so it is accumulated here instead of inside
      // backward_v*_cell.
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

      // Update Adjoint Pressure (P)
      // p = p + dt * (div(B(1+b)v) - div(b*psi))
      // phi = a*phi - dt*K*a*p

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
                        p_shot,
#if DW_NDIM >= 3
                        vz_shot, phi_z_shot, buoyancy_z_shot, psi_z_t,
#endif
#if DW_NDIM >= 2
                        vy_shot, phi_y_shot, buoyancy_y_shot, psi_y_t,
#endif
                        vx_shot, phi_x_shot, buoyancy_x_shot, psi_x_t,
                        k_shot[i], grad_k_shot, k_grad_store_1_t, dt,
                        step_ratio, k_requires_grad_t, i, i,
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

      // Fold-handled faces: accumulate the adjoint-stencil contributions
      // that fall on the adjoint pressure's guard cells (folded into the
      // interior at the top of the next iteration).
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

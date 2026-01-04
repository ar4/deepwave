/*
 * Acoustic wave equation propagator (variable density, staggered grid)
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
#define FUNC(name) CAT(name, DW_NDIM, DW_ACCURACY, DW_DTYPE, DW_DEVICE)

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz)*ny * nx + (dy)*nx + (dx))
#define DIM_ARGS nz, ny, nx
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy)*nx + (dx))
#define DIM_ARGS ny, nx
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#define DIM_ARGS nx
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Models
#define K(dz, dy, dx) k_shot[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(i, dz, dy, dx)]

// Wavefields
#define P(dz, dy, dx) p_shot[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx_shot[ND_INDEX(i, dz, dy, dx)]

// PML Memory Variables (Phi for Pressure, Psi for Velocity)
#if DW_NDIM >= 3
#define PHI_Z(dz, dy, dx) phi_z_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PHI_Y(dz, dy, dx) phi_y_shot[ND_INDEX(i, dz, dy, dx)]
#endif
#define PHI_X(dz, dy, dx) phi_x_shot[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define PSI_Z(dz, dy, dx) psi_z_t[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define PSI_Y(dz, dy, dx) psi_y_t[ND_INDEX(i, dz, dy, dx)]
#endif
#define PSI_X(dz, dy, dx) psi_x_t[ND_INDEX(i, dz, dy, dx)]

#if DW_NDIM >= 3
#define VZ_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VZ_ADJOINT_PML(dz, dy, dx)  \
  (bz[z + dz] * PHI_Z(dz, dy, dx) - \
   dt * (1 + bz[z + dz]) * K(dz, dy, dx) * P(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VY_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VY_ADJOINT_PML(dz, dy, dx)  \
  (by[y + dy] * PHI_Y(dz, dy, dx) - \
   dt * (1 + by[y + dy]) * K(dz, dy, dx) * P(dz, dy, dx))
#endif
#define VX_ADJOINT(dz, dy, dx) (-dt * K(dz, dy, dx) * P(dz, dy, dx))
#define VX_ADJOINT_PML(dz, dy, dx)  \
  (bx[x + dx] * PHI_X(dz, dy, dx) - \
   dt * (1 + bx[x + dx]) * K(dz, dy, dx) * P(dz, dy, dx))

#if DW_NDIM >= 3
#define P_ADJOINT_Z(dz, dy, dx) (-dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#define P_ADJOINT_Z_PML(dz, dy, dx)  \
  (bzh[z + dz] * PSI_Z(dz, dy, dx) - \
   dt * (1 + bzh[z + dz]) * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define P_ADJOINT_Y(dz, dy, dx) (-dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define P_ADJOINT_Y_PML(dz, dy, dx)  \
  (byh[y + dy] * PSI_Y(dz, dy, dx) - \
   dt * (1 + byh[y + dy]) * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define P_ADJOINT_X(dz, dy, dx) (-dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define P_ADJOINT_X_PML(dz, dy, dx)  \
  (bxh[x + dx] * PSI_X(dz, dy, dx) - \
   dt * (1 + bxh[x + dx]) * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

#if DW_NDIM == 3
#define SET_I int64_t const i = z * ny * nx + y * nx + x
#elif DW_NDIM == 2
#define SET_I int64_t const i = y * nx + x
#else
#define SET_I int64_t const i = x
#endif

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(forward)(
        DW_DTYPE const *__restrict const k,
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
        DW_DTYPE *__restrict const k_grad_store_1,
        DW_DTYPE *__restrict const k_grad_store_1b,
        void *__restrict const k_grad_store_2,
        void *__restrict const k_grad_store_3,
        char const *__restrict const *__restrict const k_grad_filenames_ptr,
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
        int64_t const n_sources_vx_per_shot,
        int64_t const n_receivers_p_per_shot,
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
        int64_t const pml_x1, int64_t const n_threads, void *unused) {

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

  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
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

    DW_DTYPE const *__restrict const k_shot = k_batched ? k + si : k;
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

#define OPEN_FILE_WRITE(name, grad_cond)                 \
  FILE *fp_##name = NULL;                                \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {     \
    fp_##name = fopen(name##_filenames_ptr[shot], "ab"); \
  }

    OPEN_FILE_WRITE(k_grad, k_requires_grad)
#if DW_NDIM >= 3
    OPEN_FILE_WRITE(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_WRITE(by_grad, b_requires_grad)
#endif
    OPEN_FILE_WRITE(bx_grad, b_requires_grad)

    for (t = start_t; t < start_t + nt; ++t) {
#define SETUP_STORE_SAVE(name, grad_cond)                              \
  DW_DTYPE *__restrict const name##_store_1_t =                        \
      name##_store_1 + si +                                            \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)        \
           ? (t / step_ratio) * n_shots * n_grid_points                \
           : 0);                                                       \
  void *__restrict const name##_store_2_t =                            \
      (uint8_t *)name##_store_2 +                                      \
      (shot + ((storage_mode == STORAGE_DEVICE && storage_compression) \
                   ? t / step_ratio * n_shots                          \
                   : 0)) *                                             \
          (int64_t)shot_bytes_comp;

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

#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

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

            // Update Velocities (V)
            // v = v - dt * B * (grad(p) + psi)

#if DW_NDIM >= 3
#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE term_z = DIFFZH1(P);

                  if (pml_z != 1) {
                    psi_z_t[i] = bzh[z] * term_z + azh[z] * psi_z_t[i];
                    term_z += psi_z_t[i];
                  }

                  if (b_requires_grad_t) {
                    bz_grad_store_1_t[i] = term_z;
                  }
                  vz_shot[i] -= dt * buoyancy_z_shot[i] * term_z;
                }
#endif

#if DW_NDIM >= 2
#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE term_y = DIFFYH1(P);

                  if (pml_y != 1) {
                    psi_y_t[i] = byh[y] * term_y + ayh[y] * psi_y_t[i];
                    term_y += psi_y_t[i];
                  }

                  if (b_requires_grad_t) {
                    by_grad_store_1_t[i] = term_y;
                  }
                  vy_shot[i] -= dt * buoyancy_y_shot[i] * term_y;
                }
#endif

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;
                  DW_DTYPE term_x = DIFFXH1(P);

                  if (pml_x != 1) {
                    psi_x_t[i] = bxh[x] * term_x + axh[x] * psi_x_t[i];
                    term_x += psi_x_t[i];
                  }

                  if (b_requires_grad_t) {
                    bx_grad_store_1_t[i] = term_x;
                  }
                  vx_shot[i] -= dt * buoyancy_x_shot[i] * term_x;
                }
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

            // Update Pressure (P)
            // p = p - dt * K * (div(v) + phi)

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  DW_DTYPE div_v = 0;
#if DW_NDIM >= 3
                  DW_DTYPE d_z = DIFFZ1(VZ);
                  if (pml_z != 1) {
                    phi_z_shot[i] = az[z] * phi_z_shot[i] + bz[z] * d_z;
                    d_z += phi_z_shot[i];
                  }
                  div_v += d_z;
#endif
#if DW_NDIM >= 2
                  DW_DTYPE d_y = DIFFY1(VY);
                  if (pml_y != 1) {
                    phi_y_shot[i] = ay[y] * phi_y_shot[i] + by[y] * d_y;
                    d_y += phi_y_shot[i];
                  }
                  div_v += d_y;
#endif
                  DW_DTYPE d_x = DIFFX1(VX);
                  if (pml_x != 1) {
                    phi_x_shot[i] = ax[x] * phi_x_shot[i] + bx[x] * d_x;
                    d_x += phi_x_shot[i];
                  }
                  div_v += d_x;

                  if (k_requires_grad_t) {
                    k_grad_store_1_t[i] = div_v;
                  }
                  p_shot[i] -= dt * k_shot[i] * div_v;
                }
          }

      // Inject Sources
      if (n_sources_p_per_shot > 0) {
        add_to_wavefield(p_shot, sources_i_p + src_i_p,
                         f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
                         n_sources_p_per_shot);
      }

      // Save snapshots
#define SAVE_SNAPSHOT(name, grad_cond)                                     \
  if (grad_cond) {                                                         \
    int64_t const step_idx = t / step_ratio;                               \
    STORAGE_FUNC(save_snapshot_cpu)(                                       \
        name##_store_1_t, name##_store_2_t, fp_##name, storage_mode,       \
        storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp, \
        DIM_ARGS);                                                         \
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

#define CLOSE_FILE(name) \
  if (fp_##name) fclose(fp_##name);

    CLOSE_FILE(k_grad)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad)
#endif
    CLOSE_FILE(bx_grad)
  }
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FUNC(backward)(
        DW_DTYPE const *__restrict const k,
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
        DW_DTYPE const *__restrict const grad_r_vx,
        DW_DTYPE *__restrict const p,
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

        DW_DTYPE *__restrict const k_grad_store_1,
        DW_DTYPE *__restrict const k_grad_store_1b,
        void *__restrict const k_grad_store_2,
        void *__restrict const k_grad_store_3,
        char const *__restrict const *__restrict const k_grad_filenames_ptr,
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
        DW_DTYPE *__restrict const grad_f_vx, DW_DTYPE *__restrict const grad_k,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_bz,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_by,
#endif
        DW_DTYPE *__restrict const grad_bx,
        DW_DTYPE *__restrict const grad_k_thread,
#if DW_NDIM >= 3
        DW_DTYPE *__restrict const grad_bz_thread,
#endif
#if DW_NDIM >= 2
        DW_DTYPE *__restrict const grad_by_thread,
#endif
        DW_DTYPE *__restrict const grad_bx_thread,
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
        int64_t const n_sources_vx_per_shot,
        int64_t const n_receivers_p_per_shot,
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
        int64_t const pml_x1, int64_t const n_threads, void *unused) {

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

  int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif /* _OPENMP */
  for (shot = 0; shot < n_shots; ++shot) {
#ifdef _OPENMP
    int64_t const threadi = omp_get_thread_num() * n_grid_points;
#else
    int64_t const threadi = 0;
#endif /* _OPENMP */

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

    DW_DTYPE const *__restrict const k_shot = k_batched ? k + si : k;
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

    // In backward pass, p is the adjoint pressure field (q)
    // vz, vy, vx are adjoint velocity fields
    DW_DTYPE *__restrict const p_shot = p + si;
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

    // Gradients
    DW_DTYPE *__restrict const grad_k_shot =
        grad_k_thread + (k_batched ? si : threadi);
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const grad_bz_shot =
        grad_bz_thread + (b_batched ? si : threadi);
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const grad_by_shot =
        grad_by_thread + (b_batched ? si : threadi);
#endif
    DW_DTYPE *__restrict const grad_bx_shot =
        grad_bx_thread + (b_batched ? si : threadi);

    int64_t t;

#define OPEN_FILE_READ(name, grad_cond)                  \
  FILE *fp_##name = NULL;                                \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {     \
    fp_##name = fopen(name##_filenames_ptr[shot], "rb"); \
  }

    OPEN_FILE_READ(k_grad, k_requires_grad)
#if DW_NDIM >= 3
    OPEN_FILE_READ(bz_grad, b_requires_grad)
#endif
#if DW_NDIM >= 2
    OPEN_FILE_READ(by_grad, b_requires_grad)
#endif
    OPEN_FILE_READ(bx_grad, b_requires_grad)

    for (t = start_t - 1; t >= start_t - nt; --t) {
#define SETUP_STORE_LOAD(name, grad_cond)                                    \
  DW_DTYPE *__restrict const name##_store_1_t =                              \
      name##_store_1 + si +                                                  \
      ((storage_mode == STORAGE_DEVICE && !storage_compression)              \
           ? (t / step_ratio) * n_shots * n_grid_points                      \
           : 0);                                                             \
  void *__restrict const name##_store_2_t =                                  \
      (uint8_t *)name##_store_2 +                                            \
      (shot + ((storage_mode == STORAGE_DEVICE && storage_compression)       \
                   ? t / step_ratio * n_shots                                \
                   : 0)) *                                                   \
          (int64_t)shot_bytes_comp;                                          \
  if ((grad_cond) && ((t % step_ratio) == 0)) {                              \
    int64_t const step_idx = t / step_ratio;                                 \
    STORAGE_FUNC(load_snapshot_cpu)(                                         \
        (void *)name##_store_1_t, name##_store_2_t, fp_##name, storage_mode, \
        storage_compression, step_idx, shot_bytes_uncomp, shot_bytes_comp,   \
        DIM_ARGS);                                                           \
  }

      SETUP_STORE_LOAD(k_grad, k_requires_grad)
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

      bool const k_requires_grad_t = k_requires_grad && ((t % step_ratio) == 0);
      bool const b_requires_grad_t = b_requires_grad && ((t % step_ratio) == 0);

#if DW_NDIM >= 3
      int pml_z;
#endif
#if DW_NDIM >= 2
      int pml_y;
#endif
      int pml_x;

      if (n_sources_p_per_shot > 0) {
        record_from_wavefield(
            p_shot, sources_i_p + src_i_p,
            grad_f_p + t * n_shots * n_sources_p_per_shot + src_i_p,
            n_sources_p_per_shot);
      }

      // Update Adjoint Velocities (V)
      // v = v + dt * (grad((1+b)Kp) - grad(b*phi))
      // psi = a*psi - dt*B*a*v

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
#pragma omp simd collapse(DW_NDIM)
            for (z = pml_bounds_zh[pml_z]; z < pml_bounds_zh[pml_z + 1]; z++)
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  vz_shot[i] += (pml_z == 1 ? -DIFFZH1(VZ_ADJOINT)
                                            : -DIFFZH1(VZ_ADJOINT_PML));
                  psi_zn_t[i] = azh[z] * psi_z_t[i] -
                                dt * buoyancy_z_shot[i] * azh[z] * vz_shot[i];

                  if (b_requires_grad_t) {
                    grad_bz_shot[i] -= dt * vz_shot[i] * bz_grad_store_1_t[i] *
                                       (DW_DTYPE)step_ratio;
                  }
                }
#endif

#if DW_NDIM >= 2
#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
              for (y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; y++)
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  vy_shot[i] += (pml_y == 1 ? -DIFFYH1(VY_ADJOINT)
                                            : -DIFFYH1(VY_ADJOINT_PML));
                  psi_yn_t[i] = ayh[y] * psi_y_t[i] -
                                dt * buoyancy_y_shot[i] * ayh[y] * vy_shot[i];

                  if (b_requires_grad_t) {
                    grad_by_shot[i] -= dt * vy_shot[i] * by_grad_store_1_t[i] *
                                       (DW_DTYPE)step_ratio;
                  }
                }
#endif

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1];
                     x++) {
                  SET_I;

                  vx_shot[i] += (pml_x == 1 ? -DIFFXH1(VX_ADJOINT)
                                            : -DIFFXH1(VX_ADJOINT_PML));
                  psi_xn_t[i] = axh[x] * psi_x_t[i] -
                                dt * buoyancy_x_shot[i] * axh[x] * vx_shot[i];

                  if (b_requires_grad_t) {
                    grad_bx_shot[i] -= dt * vx_shot[i] * bx_grad_store_1_t[i] *
                                       (DW_DTYPE)step_ratio;
                  }
                }
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

      // Update Adjoint Pressure (P)
      // p = p + dt * (div(B(1+b)v) - div(b*psi))
      // phi = a*phi - dt*K*a*p

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

#pragma omp simd collapse(DW_NDIM)
#if DW_NDIM >= 3
            for (z = pml_bounds_z[pml_z]; z < pml_bounds_z[pml_z + 1]; z++)
#endif
#if DW_NDIM >= 2
              for (y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; y++)
#endif
                for (x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1];
                     x++) {
                  SET_I;

                  // Update Phi
#if DW_NDIM >= 3
                  if (pml_z != 1)
                    phi_z_shot[i] = az[z] * phi_z_shot[i] -
                                    dt * k_shot[i] * az[z] * p_shot[i];
#endif
#if DW_NDIM >= 2
                  if (pml_y != 1)
                    phi_y_shot[i] = ay[y] * phi_y_shot[i] -
                                    dt * k_shot[i] * ay[y] * p_shot[i];
#endif
                  if (pml_x != 1)
                    phi_x_shot[i] = ax[x] * phi_x_shot[i] -
                                    dt * k_shot[i] * ax[x] * p_shot[i];

                  if (k_requires_grad_t) {
                    grad_k_shot[i] -= dt * p_shot[i] * k_grad_store_1_t[i] *
                                      (DW_DTYPE)step_ratio;
                  }

                  // Update P
                  DW_DTYPE div_term = 0;
#if DW_NDIM >= 3
                  div_term += pml_z == 1 ? -DIFFZ1(P_ADJOINT_Z)
                                         : -DIFFZ1(P_ADJOINT_Z_PML);
#endif
#if DW_NDIM >= 2
                  div_term += pml_y == 1 ? -DIFFY1(P_ADJOINT_Y)
                                         : -DIFFY1(P_ADJOINT_Y_PML);
#endif
                  div_term += pml_x == 1 ? -DIFFX1(P_ADJOINT_X)
                                         : -DIFFX1(P_ADJOINT_X_PML);

                  p_shot[i] += div_term;
                }
          }

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

#define CLOSE_FILE(name) \
  if (fp_##name) fclose(fp_##name);

    CLOSE_FILE(k_grad)
#if DW_NDIM >= 3
    CLOSE_FILE(bz_grad)
#endif
#if DW_NDIM >= 2
    CLOSE_FILE(by_grad)
#endif
    CLOSE_FILE(bx_grad)
  }

#ifdef _OPENMP
  if (k_requires_grad && !k_batched && n_threads > 1) {
    combine_grad(grad_k, grad_k_thread, n_threads, n_grid_points);
  }
#if DW_NDIM >= 3
  if (b_requires_grad && !b_batched && n_threads > 1) {
    combine_grad(grad_bz, grad_bz_thread, n_threads, n_grid_points);
  }
#endif
#if DW_NDIM >= 2
  if (b_requires_grad && !b_batched && n_threads > 1) {
    combine_grad(grad_by, grad_by_thread, n_threads, n_grid_points);
  }
#endif
  if (b_requires_grad && !b_batched && n_threads > 1) {
    combine_grad(grad_bx, grad_bx_thread, n_threads, n_grid_points);
  }
#endif /* _OPENMP */

  return 0;
}

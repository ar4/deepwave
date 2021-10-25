#include "scalar_cpu.h"

#include <stddef.h>
#include <string.h>

#include "scalar.h"

static inline ptrdiff_t location_index(
    const ptrdiff_t * const arr,
    const ptrdiff_t * const shape, const ptrdiff_t index);
#if DEEPWAVE_DIM == 1
static inline DEEPWAVE_TYPE laplacian_1d(const DEEPWAVE_TYPE * const arr,
                                const DEEPWAVE_TYPE * const fd2,
                                const ptrdiff_t si);
#endif
#if DEEPWAVE_DIM == 2
static inline DEEPWAVE_TYPE laplacian_2d(const DEEPWAVE_TYPE * const arr,
                                const DEEPWAVE_TYPE * const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x);
#endif
#if DEEPWAVE_DIM == 3
static inline DEEPWAVE_TYPE laplacian_3d(const DEEPWAVE_TYPE * const arr,
                                const DEEPWAVE_TYPE * const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x,
                                const ptrdiff_t size_xy);
#endif
static inline DEEPWAVE_TYPE z_deriv(const DEEPWAVE_TYPE * const arr,
                           const DEEPWAVE_TYPE * const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_xy);
static inline DEEPWAVE_TYPE y_deriv(const DEEPWAVE_TYPE * const arr,
                           const DEEPWAVE_TYPE * const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_x);
static inline DEEPWAVE_TYPE x_deriv(const DEEPWAVE_TYPE * const arr,
                           const DEEPWAVE_TYPE * const fd1,
                           const ptrdiff_t si);

void setup(const DEEPWAVE_TYPE * const /*fd1*/,
           const DEEPWAVE_TYPE * const /*fd2*/) {}

#if DEEPWAVE_DIM == 1
void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next auxiliary */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current auxiliary */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1, /* 1st difference coeffs */
               const DEEPWAVE_TYPE * const fd2, /* 2nd difference coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t numel_shot = shape[0];
  DEEPWAVE_TYPE * const phizn = auxn;
  const DEEPWAVE_TYPE * const phizc = auxc;
  const DEEPWAVE_TYPE * const sigmaz = sigma;

#pragma omp parallel for collapse(2)
  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      const ptrdiff_t i = z;
      const ptrdiff_t si = shot * numel_shot + i;

      /* Spatial finite differences */
      const DEEPWAVE_TYPE lap = laplacian_1d(wfc, fd2, si);
      const DEEPWAVE_TYPE wfc_z = z_deriv(wfc, fd1, si, 1);
      const DEEPWAVE_TYPE phizc_z = z_deriv(phizc, fd1, si, 1);

      /* Update wavefield */
      wfn[si] = 1 / (1 + dt * sigmaz[z] / 2) *
                (model[i] * (lap + phizc_z) + dt * sigmaz[z] * wfp[si] / 2 +
                 (2 * wfc[si] - wfp[si]));

      /* Update phi */
      phizn[si] = phizc[si] - dt * sigmaz[z] * (wfc_z + phizc[si]);
    }
  }
}

void imaging_condition(DEEPWAVE_TYPE * const model_grad,
                       const DEEPWAVE_TYPE * const current_wavefield,
                       const DEEPWAVE_TYPE * const saved_wavefield,
                       const DEEPWAVE_TYPE * const saved_wavefield_t,
                       const DEEPWAVE_TYPE * const saved_wavefield_tt,
                       const DEEPWAVE_TYPE * const sigma,
                       const ptrdiff_t * const shape,
                       const ptrdiff_t * const pml_width,
                       const ptrdiff_t num_shots) {
  if (model_grad == nullptr) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0];
  const DEEPWAVE_TYPE * const sigmaz = sigma;

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      const ptrdiff_t i = z;
      const ptrdiff_t si = shot * numel_shot + i;

      model_grad[i] +=
          current_wavefield[si] *
          (saved_wavefield_tt[si] + sigmaz[z] * saved_wavefield_t[si]);
    }
  }
}

static inline ptrdiff_t location_index(
    const ptrdiff_t * const arr,
    const ptrdiff_t * const shape, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index];

  return z;
}

#elif DEEPWAVE_DIM == 2

void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next auxiliary */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current auxiliary */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1, /* 1st difference coeffs */
               const DEEPWAVE_TYPE * const fd2, /* 2nd difference coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t numel_shot = shape[0] * shape[1];
  const ptrdiff_t size_xy = shape[1];
  DEEPWAVE_TYPE * const phizn = auxn;
  DEEPWAVE_TYPE * const phiyn = auxn + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const phizc = auxc;
  const DEEPWAVE_TYPE * const phiyc = auxc + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const sigmaz = sigma;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];

#pragma omp parallel for collapse(2)
  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        const ptrdiff_t i = z * size_xy + y;
        const ptrdiff_t si = shot * numel_shot + i;

        /* Spatial finite differences */
        const DEEPWAVE_TYPE lap = laplacian_2d(wfc, fd2, si, size_xy);
        const DEEPWAVE_TYPE wfc_z = z_deriv(wfc, fd1, si, size_xy);
        const DEEPWAVE_TYPE wfc_y = y_deriv(wfc, fd1, si, 1);
        const DEEPWAVE_TYPE phizc_z = z_deriv(phizc, fd1, si, size_xy);
        const DEEPWAVE_TYPE phiyc_y = y_deriv(phiyc, fd1, si, 1);

        /* Update wavefield */
        wfn[si] = 1 / (1 + dt * (sigmaz[z] + sigmay[y]) / 2) *
                  (model[i] * (lap + phizc_z + phiyc_y) +
                   dt * (sigmaz[z] + sigmay[y]) * wfp[si] / 2 +
                   (2 * wfc[si] - wfp[si]) -
                   dt * dt * sigmaz[z] * sigmay[y] * wfc[si]);

        /* Update phi */
        phizn[si] = phizc[si] - dt * (sigmaz[z] * phizc[si] +
                                      (sigmaz[z] - sigmay[y]) * wfc_z);
        phiyn[si] = phiyc[si] - dt * (sigmay[y] * phiyc[si] +
                                      (sigmay[y] - sigmaz[z]) * wfc_y);
      }
    }
  }
}

void imaging_condition(DEEPWAVE_TYPE * const model_grad,
                       const DEEPWAVE_TYPE * const current_wavefield,
                       const DEEPWAVE_TYPE * const saved_wavefield,
                       const DEEPWAVE_TYPE * const saved_wavefield_t,
                       const DEEPWAVE_TYPE * const saved_wavefield_tt,
                       const DEEPWAVE_TYPE * const sigma,
                       const ptrdiff_t * const shape,
                       const ptrdiff_t * const pml_width,
                       const ptrdiff_t num_shots) {
  if (model_grad == nullptr) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0] * shape[1];
  const ptrdiff_t size_xy = shape[1];
  const DEEPWAVE_TYPE * const sigmaz = sigma;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        const ptrdiff_t i = z * size_xy + y;
        const ptrdiff_t si = shot * numel_shot + i;

        model_grad[i] += current_wavefield[si] *
                         (saved_wavefield_tt[si] +
                          (sigmaz[z] + sigmay[y]) * saved_wavefield_t[si] +
                          sigmaz[z] * sigmay[y] * saved_wavefield[si]);
      }
    }
  }
}

static inline ptrdiff_t location_index(
    const ptrdiff_t * const arr,
    const ptrdiff_t * const shape, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index * 2];
  const ptrdiff_t y = arr[index * 2 + 1];

  return z * shape[1] + y;
}

#elif DEEPWAVE_DIM == 3

void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next auxiliary */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current auxiliary */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1, /* 1st difference coeffs */
               const DEEPWAVE_TYPE * const fd2, /* 2nd difference coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
  const ptrdiff_t size_x = shape[2];
  const ptrdiff_t size_xy = shape[1] * shape[2];
  DEEPWAVE_TYPE * const phizn = auxn;
  DEEPWAVE_TYPE * const phiyn = auxn + num_shots * numel_shot;
  DEEPWAVE_TYPE * const phixn = auxn + 2 * num_shots * numel_shot;
  DEEPWAVE_TYPE * const psin = auxn + 3 * num_shots * numel_shot;
  const DEEPWAVE_TYPE * const phizc = auxc;
  const DEEPWAVE_TYPE * const phiyc = auxc + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const phixc = auxc + 2 * num_shots * numel_shot;
  const DEEPWAVE_TYPE * const psic = auxc + 3 * num_shots * numel_shot;
  const DEEPWAVE_TYPE * const sigmaz = sigma;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];
  const DEEPWAVE_TYPE * const sigmax = sigma + shape[0] + shape[1];

#pragma omp parallel for collapse(2)
  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        for (ptrdiff_t x = XPAD; x < shape[2] - XPAD; x++) {
          const ptrdiff_t i = z * size_xy + y * size_x + x;
          const ptrdiff_t si = shot * numel_shot + i;

          /* Spatial finite differences */
          const DEEPWAVE_TYPE lap = laplacian_3d(wfc, fd2, si, size_x, size_xy);
          const DEEPWAVE_TYPE wfc_z = z_deriv(wfc, fd1, si, size_xy);
          const DEEPWAVE_TYPE wfc_y = y_deriv(wfc, fd1, si, size_x);
          const DEEPWAVE_TYPE wfc_x = x_deriv(wfc, fd1, si);
          const DEEPWAVE_TYPE phizc_z = z_deriv(phizc, fd1, si, size_xy);
          const DEEPWAVE_TYPE phiyc_y = y_deriv(phiyc, fd1, si, size_x);
          const DEEPWAVE_TYPE phixc_x = x_deriv(phixc, fd1, si);
          const DEEPWAVE_TYPE psic_z = z_deriv(psic, fd1, si, size_xy);
          const DEEPWAVE_TYPE psic_y = y_deriv(psic, fd1, si, size_x);
          const DEEPWAVE_TYPE psic_x = x_deriv(psic, fd1, si);

          /* Update wavefield */
          wfn[si] = 1 / (1 + dt * (sigmaz[z] + sigmay[y] + sigmax[x]) / 2) *
                    (model[i] * lap +
                     dt * dt *
                         (phizc_z + phiyc_y + phixc_x -
                          sigmaz[z] * sigmay[y] * sigmax[x] * psic[si]) +
                     dt * (sigmaz[z] + sigmay[y] + sigmax[x]) * wfp[si] / 2 +
                     (2 * wfc[si] - wfp[si]) -
                     dt * dt * wfc[si] *
                         (sigmax[x] * sigmay[y] + sigmay[y] * sigmaz[z] +
                          sigmax[x] * sigmaz[z]));

          /* Update phi */
          phizn[si] = phizc[si] - dt * sigmaz[z] * phizc[si] +
                      model[i] / dt * (sigmay[y] + sigmax[x]) * wfc_z +
                      dt * sigmax[x] * sigmay[y] * psic_z;
          phiyn[si] = phiyc[si] - dt * sigmay[y] * phiyc[si] +
                      model[i] / dt * (sigmaz[z] + sigmax[x]) * wfc_y +
                      dt * sigmax[x] * sigmaz[z] * psic_y;
          phixn[si] = phixc[si] - dt * sigmax[x] * phixc[si] +
                      model[i] / dt * (sigmaz[z] + sigmay[y]) * wfc_x +
                      dt * sigmaz[z] * sigmay[y] * psic_x;

          /* Update psi */
          psin[si] = psic[si] + dt * wfc[si];
        }
      }
    }
  }
}

void imaging_condition(DEEPWAVE_TYPE * const model_grad,
                       const DEEPWAVE_TYPE * const current_wavefield,
                       const DEEPWAVE_TYPE * const saved_wavefield,
                       const DEEPWAVE_TYPE * const saved_wavefield_t,
                       const DEEPWAVE_TYPE * const saved_wavefield_tt,
                       const DEEPWAVE_TYPE * const sigma,
                       const ptrdiff_t * const shape,
                       const ptrdiff_t * const pml_width,
                       const ptrdiff_t num_shots) {
  if (model_grad == nullptr) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
  const ptrdiff_t size_x = shape[2];
  const ptrdiff_t size_xy = shape[1] * shape[2];
  const DEEPWAVE_TYPE * const sigmaz = sigma;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];
  const DEEPWAVE_TYPE * const sigmax = sigma + shape[0] + shape[1];

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        for (ptrdiff_t x = XPAD; x < shape[2] - XPAD; x++) {
          const ptrdiff_t i = z * size_xy + y * size_x + x;
          const ptrdiff_t si = shot * numel_shot + i;

          /* NOTE: There should be an additional term here (involving spatial
           * derivative of phi, sigma, and psi), but it is neglected due to
           * the additional computational cost it would cause. */
          model_grad[i] +=
              current_wavefield[si] *
              (saved_wavefield_tt[si] +
               (sigmaz[z] + sigmay[y] + sigmax[x]) * saved_wavefield_t[si] +
               (sigmax[x] * sigmay[y] + sigmay[y] * sigmaz[z] +
                sigmax[x] * sigmaz[z]) *
                   saved_wavefield[si]);
        }
      }
    }
  }
}

static inline ptrdiff_t location_index(
    const ptrdiff_t * const arr,
    const ptrdiff_t * const shape, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index * 3];
  const ptrdiff_t y = arr[index * 3 + 1];
  const ptrdiff_t x = arr[index * 3 + 2];

  return z * shape[1] * shape[2] + y * shape[2] + x;
}

#else
#error "Must specify the dimension, e.g. -D DEEPWAVE_DIM=1"
#endif /* DEEPWAVE_DIM */

void add_sources(DEEPWAVE_TYPE * const next_wavefield,
                 const DEEPWAVE_TYPE * const model,
                 const DEEPWAVE_TYPE * const source_amplitudes,
                 const ptrdiff_t * const source_locations,
                 const ptrdiff_t * const shape,
                 const ptrdiff_t num_shots,
                 const ptrdiff_t num_sources_per_shot) {
  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t source = 0; source < num_sources_per_shot; source++) {
      const ptrdiff_t s = shot * num_sources_per_shot + source;
      const ptrdiff_t i = location_index(source_locations, shape, s);
      const ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] + i;
      next_wavefield[si] += source_amplitudes[s] * model[i];
    }
  }
}

void add_scattering(DEEPWAVE_TYPE * const next_scattered_wavefield,
                    const DEEPWAVE_TYPE * const next_wavefield,
                    const DEEPWAVE_TYPE * const current_wavefield,
                    const DEEPWAVE_TYPE * const previous_wavefield,
                    const DEEPWAVE_TYPE * const scatter,
                    const ptrdiff_t * const shape,
                    const ptrdiff_t num_shots) {
  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
  const ptrdiff_t size_x = shape[2];
  const ptrdiff_t size_xy = shape[1] * shape[2];

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
      for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
        for (ptrdiff_t x = XPAD; x < shape[2] - XPAD; x++) {
          const ptrdiff_t i = z * size_xy + y * size_x + x;
          const ptrdiff_t si = shot * numel_shot + i;
          const DEEPWAVE_TYPE current_wavefield_tt =
              (next_wavefield[si] - 2 * current_wavefield[si] +
               previous_wavefield[si]); /* no dt^2 because of cancellation */
          next_scattered_wavefield[si] += current_wavefield_tt * scatter[i];
        }
      }
    }
  }
}

void record_receivers(DEEPWAVE_TYPE * const receiver_amplitudes,
                      const DEEPWAVE_TYPE * const current_wavefield,
                      const ptrdiff_t * const receiver_locations,
                      const ptrdiff_t * const shape,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_receivers_per_shot) {
  if (receiver_amplitudes == nullptr) return; /* no source inversion */

  for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
    for (ptrdiff_t receiver = 0; receiver < num_receivers_per_shot;
         receiver++) {
      const ptrdiff_t r = shot * num_receivers_per_shot + receiver;
      const ptrdiff_t si = shot * shape[0] * shape[1] * shape[2] +
                           location_index(receiver_locations, shape, r);
      receiver_amplitudes[r] = current_wavefield[si];
    }
  }
}

void save_wavefields(DEEPWAVE_TYPE * const current_saved_wavefield,
                     DEEPWAVE_TYPE * const current_saved_wavefield_t,
                     DEEPWAVE_TYPE * const current_saved_wavefield_tt,
                     const DEEPWAVE_TYPE * const next_wavefield,
                     const DEEPWAVE_TYPE * const current_wavefield,
                     const DEEPWAVE_TYPE * const previous_wavefield,
                     const ptrdiff_t * const shape,
                     const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt,
                     const enum wavefield_save_strategy save_strategy) {
  if (save_strategy == STRATEGY_COPY) {
    const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
    const ptrdiff_t size_x = shape[2];
    const ptrdiff_t size_xy = shape[1] * shape[2];
    memcpy(current_saved_wavefield, current_wavefield,
           static_cast<size_t>(num_shots * shape[0] * shape[1] * shape[2])
           * sizeof(DEEPWAVE_TYPE));

    for (ptrdiff_t shot = 0; shot < num_shots; shot++) {
      for (ptrdiff_t z = ZPAD; z < shape[0] - ZPAD; z++) {
        for (ptrdiff_t y = YPAD; y < shape[1] - YPAD; y++) {
          for (ptrdiff_t x = XPAD; x < shape[2] - XPAD; x++) {
            const ptrdiff_t i = z * size_xy + y * size_x + x;
            const ptrdiff_t si = shot * numel_shot + i;
            current_saved_wavefield_t[si] =
                (next_wavefield[si] - previous_wavefield[si]) / 2 / dt;
            current_saved_wavefield_tt[si] =
                (next_wavefield[si] - 2 * current_wavefield[si] +
                 previous_wavefield[si]) /
                dt / dt;
          }
        }
      }
    }
  }
}

void model_grad_scaling(DEEPWAVE_TYPE * const model_grad,
                        const DEEPWAVE_TYPE * const scaling,
                        const ptrdiff_t * const shape,
                        const ptrdiff_t * const pml_width) {
  if (model_grad == nullptr) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];

  for (ptrdiff_t i = 0; i < numel_shot; i++) {
    model_grad[i] *= scaling[i];
  }
}

#if DEEPWAVE_DIM == 1
static inline DEEPWAVE_TYPE laplacian_1d(const DEEPWAVE_TYPE * const arr,
                                const DEEPWAVE_TYPE * const fd2,
                                const ptrdiff_t si) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + 1] + arr[si - 1]) +
         fd2[2] * (arr[si + 2] + arr[si - 2]);
}
#endif

#if DEEPWAVE_DIM == 2
static inline DEEPWAVE_TYPE laplacian_2d(const DEEPWAVE_TYPE * const arr,
                                const DEEPWAVE_TYPE * const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + size_x] + arr[si - size_x]) +
         fd2[2] * (arr[si + 2 * size_x] + arr[si - 2 * size_x]) +
         +fd2[3] * (arr[si + 1] + arr[si - 1]) +
         fd2[4] * (arr[si + 2] + arr[si - 2]);
}
#endif

#if DEEPWAVE_DIM == 3
static inline DEEPWAVE_TYPE laplacian_3d(const DEEPWAVE_TYPE * const arr,
                                const DEEPWAVE_TYPE * const fd2,
                                const ptrdiff_t si, const ptrdiff_t size_x,
                                const ptrdiff_t size_xy) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + size_xy] + arr[si - size_xy]) +
         fd2[2] * (arr[si + 2 * size_xy] + arr[si - 2 * size_xy]) +
         +fd2[3] * (arr[si + size_x] + arr[si - size_x]) +
         fd2[4] * (arr[si + 2 * size_x] + arr[si - 2 * size_x]) +
         fd2[5] * (arr[si + 1] + arr[si - 1]) +
         fd2[6] * (arr[si + 2] + arr[si - 2]);
}
#endif

static inline DEEPWAVE_TYPE z_deriv(const DEEPWAVE_TYPE * const arr,
                           const DEEPWAVE_TYPE * const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_xy) {
  return fd1[0] * (arr[si + size_xy] - arr[si - size_xy]) +
         fd1[1] * (arr[si + 2 * size_xy] - arr[si - 2 * size_xy]);
}

static inline DEEPWAVE_TYPE y_deriv(const DEEPWAVE_TYPE * const arr,
                           const DEEPWAVE_TYPE * const fd1,
                           const ptrdiff_t si, const ptrdiff_t size_x) {
  return fd1[0] * (arr[si + size_x] - arr[si - size_x]) +
         fd1[1] * (arr[si + 2 * size_x] - arr[si - 2 * size_x]);
}

static inline DEEPWAVE_TYPE x_deriv(const DEEPWAVE_TYPE * const arr,
                           const DEEPWAVE_TYPE * const fd1,
                           const ptrdiff_t si) {
  return fd1[0] * (arr[si + 1] - arr[si - 1]) +
         fd1[1] * (arr[si + 2] - arr[si - 2]);
}

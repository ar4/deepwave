#include "scalar_gpu.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "scalar.h"

static __constant__ DEEPWAVE_TYPE fd1[2 * DEEPWAVE_DIM];
static __constant__ DEEPWAVE_TYPE fd2[2 * DEEPWAVE_DIM + 1];

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line,
                             bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

static inline __device__ DEEPWAVE_TYPE laplacian_1d(const DEEPWAVE_TYPE * const arr,
                                           const ptrdiff_t si);
static inline __device__ DEEPWAVE_TYPE laplacian_2d(const DEEPWAVE_TYPE * const arr,
                                           const ptrdiff_t si,
                                           const ptrdiff_t size_x);
static inline __device__ DEEPWAVE_TYPE laplacian_3d(const DEEPWAVE_TYPE * const arr,
                                           const ptrdiff_t si,
                                           const ptrdiff_t size_x,
                                           const ptrdiff_t size_xy);
static inline __device__ ptrdiff_t location_index(
    const ptrdiff_t * const arr,
    const ptrdiff_t * const shape, const ptrdiff_t index);
static inline __device__ DEEPWAVE_TYPE z_deriv(const DEEPWAVE_TYPE * const arr,
                                      const ptrdiff_t si,
                                      const ptrdiff_t size_xy);
static inline __device__ DEEPWAVE_TYPE y_deriv(const DEEPWAVE_TYPE * const arr,
                                      const ptrdiff_t si,
                                      const ptrdiff_t size_x);
static inline __device__ DEEPWAVE_TYPE x_deriv(const DEEPWAVE_TYPE * const arr,
                                      const ptrdiff_t si);

#if DEEPWAVE_DIM == 1
__global__ void propagate_kernel(
    DEEPWAVE_TYPE * const wfn, DEEPWAVE_TYPE * const phizn,
    const DEEPWAVE_TYPE * const wfc, const DEEPWAVE_TYPE * const wfp,
    const DEEPWAVE_TYPE * const phizc, const DEEPWAVE_TYPE * const sigmaz,
    const DEEPWAVE_TYPE * const model, const ptrdiff_t shape_z,
    const ptrdiff_t numel_shot, const ptrdiff_t num_shots,
    const ptrdiff_t pmlz0, const ptrdiff_t pmlz1, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
  const ptrdiff_t z = blockIdx.x * blockDim.x + threadIdx.x + ZPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD)) {
    const ptrdiff_t i = z;
    const ptrdiff_t si = shot * numel_shot + i;

    const DEEPWAVE_TYPE lap = laplacian_1d(wfc, si);

    if ((z >= pmlz0 + 2 * ZPAD) && (z < shape_z - pmlz1 - 2 * ZPAD)) {
      /* Update wavefield */
      wfn[si] = model[i] * lap + 2 * wfc[si] - wfp[si];
    } else {
      /* Inside PML region */

      const DEEPWAVE_TYPE wfc_z = z_deriv(wfc, si, 1);
      const DEEPWAVE_TYPE phizc_z = z_deriv(phizc, si, 1);

      /* Update wavefield */
      wfn[si] = 1 / (1 + dt * sigmaz[z] / 2) *
                (model[i] * (lap + phizc_z) + dt * sigmaz[z] * wfp[si] / 2 +
                 (2 * wfc[si] - wfp[si]));

      /* Update phi */
      phizn[si] = phizc[si] - dt * sigmaz[z] * (wfc_z + phizc[si]);
    }
  }
}

void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next auxiliary */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current auxiliary */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1_d, /* 1st diff coeffs */
               const DEEPWAVE_TYPE * const fd2_d, /* 2nd diff coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t numel_shot = shape[0];
  DEEPWAVE_TYPE * const phizn = auxn;
  const DEEPWAVE_TYPE * const phizc = auxc;
  const DEEPWAVE_TYPE * const sigmaz = sigma;

  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[0] - (2 * ZPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
  const int gridz = 1;
  const dim3 dimGrid(gridx, gridy, gridz);
  propagate_kernel<<<dimGrid, dimBlock>>>(
      wfn, phizn, wfc, wfp, phizc, sigmaz, model, shape[0], numel_shot,
      num_shots, pml_width[0], pml_width[1], dt);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ imaging_condition_kernel(
    DEEPWAVE_TYPE * const model_grad,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield_t,
    const DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const sigmaz, const ptrdiff_t shape_z,
    const ptrdiff_t num_shots) {
  const ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
  const ptrdiff_t z = blockIdx.x * blockDim.x + threadIdx.x + ZPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD)) {
    const ptrdiff_t i = z;
    const ptrdiff_t si = shot * shape_z + i;

    atomicAdd(model_grad + i, current_wavefield[si] *
                                  (current_saved_wavefield_tt[si] +
                                   sigmaz[z] * current_saved_wavefield_t[si]));
  }
}

void imaging_condition(
    DEEPWAVE_TYPE * const model_grad,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield_t,
    const DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const sigma,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t num_shots) {
  if (model_grad == NULL) return; /* Not doing model inversion */

  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[0] - (2 * ZPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
  const int gridz = 1;

  const dim3 dimGrid(gridx, gridy, gridz);
  const DEEPWAVE_TYPE * const sigmaz = sigma;

  imaging_condition_kernel<<<dimGrid, dimBlock>>>(
      model_grad, current_wavefield, current_saved_wavefield_t,
      current_saved_wavefield_tt, sigmaz, shape[0], num_shots);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ add_scattering_kernel(
    DEEPWAVE_TYPE * const next_scattered_wavefield,
    const DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const previous_wavefield,
    const DEEPWAVE_TYPE * const scatter,
    const ptrdiff_t shape_z,
    const ptrdiff_t num_shots) {
  const ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
  const ptrdiff_t z = blockIdx.x * blockDim.x + threadIdx.x + ZPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD)) {
    const ptrdiff_t i = z;
    const ptrdiff_t si = shot * shape_z + i;

    const DEEPWAVE_TYPE current_wavefield_tt =
        (next_wavefield[si] - 2 * current_wavefield[si] +
         previous_wavefield[si]); /* no dt^2 because of cancellation */
    next_scattered_wavefield[si] += current_wavefield_tt * scatter[i];
  }
}

void add_scattering(
                     DEEPWAVE_TYPE * const next_scattered_wavefield,
                     const DEEPWAVE_TYPE * const next_wavefield,
                     const DEEPWAVE_TYPE * const current_wavefield,
                     const DEEPWAVE_TYPE * const previous_wavefield,
                     const DEEPWAVE_TYPE * const scatter,
                     const ptrdiff_t * const shape,
                     const ptrdiff_t num_shots) {
  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[0] - (2 * ZPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
  const int gridz = 1;

  const dim3 dimGrid(gridx, gridy, gridz);
  add_scattering_kernel<<<dimGrid, dimBlock>>>(
      next_scattered_wavefield, next_wavefield,
      current_wavefield, previous_wavefield, scatter, shape[0], num_shots);
  gpuErrchk(cudaPeekAtLastError());
}

void __global__ save_wavefields_kernel(
    DEEPWAVE_TYPE * const current_saved_wavefield_t,
    DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const previous_wavefield, const ptrdiff_t shape_z,
    const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
  const ptrdiff_t z = blockIdx.x * blockDim.x + threadIdx.x + ZPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD)) {
    const ptrdiff_t i = z;
    const ptrdiff_t si = shot * shape_z + i;
    current_saved_wavefield_t[si] =
        (current_wavefield[si] - previous_wavefield[si]) / dt;

    current_saved_wavefield_tt[si] =
        (next_wavefield[si] - 2 * current_wavefield[si] +
         previous_wavefield[si]) /
        dt / dt;
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
    gpuErrchk(
        cudaMemcpy(current_saved_wavefield, current_wavefield,
                   num_shots * shape[0] * shape[1] * shape[2] * sizeof(DEEPWAVE_TYPE),
                   cudaMemcpyDeviceToDevice));

    const dim3 dimBlock(32, 32, 1);
    const int gridx = (shape[0] - (2 * ZPAD) + dimBlock.x - 1) / dimBlock.x;
    const int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
    const int gridz = 1;

    const dim3 dimGrid(gridx, gridy, gridz);
    save_wavefields_kernel<<<dimGrid, dimBlock>>>(
        current_saved_wavefield_t, current_saved_wavefield_tt, next_wavefield,
        current_wavefield, previous_wavefield, shape[0], num_shots, dt);
    gpuErrchk(cudaPeekAtLastError());
  }
}

static inline __device__ ptrdiff_t
location_index(const ptrdiff_t * const arr, const ptrdiff_t shape_y,
               const ptrdiff_t shape_x, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index];

  return z;
}

static inline __device__ DEEPWAVE_TYPE laplacian_1d(const DEEPWAVE_TYPE * const arr,
                                           const ptrdiff_t si) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + 1] + arr[si - 1]) +
         fd2[2] * (arr[si + 2] + arr[si - 2]);
}

#elif DEEPWAVE_DIM == 2

__global__ void propagate_kernel(
    DEEPWAVE_TYPE * const wfn, DEEPWAVE_TYPE * const phizn,
    DEEPWAVE_TYPE * const phiyn, const DEEPWAVE_TYPE * const wfc,
    const DEEPWAVE_TYPE * const wfp, const DEEPWAVE_TYPE * const phizc,
    const DEEPWAVE_TYPE * const sigmaz, const DEEPWAVE_TYPE * const phiyc,
    const DEEPWAVE_TYPE * const sigmay, const DEEPWAVE_TYPE * const model,
    const ptrdiff_t shape_z, const ptrdiff_t shape_y,
    const ptrdiff_t numel_shot, const ptrdiff_t num_shots,
    const ptrdiff_t pmlz0, const ptrdiff_t pmlz1, const ptrdiff_t pmly0,
    const ptrdiff_t pmly1, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t shot = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t z = blockIdx.y * blockDim.y + threadIdx.y + ZPAD;
  const ptrdiff_t y = blockIdx.x * blockDim.x + threadIdx.x + YPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD)) {
    const ptrdiff_t i = z * shape_y + y;
    const ptrdiff_t si = shot * numel_shot + i;

    const DEEPWAVE_TYPE lap = laplacian_2d(wfc, si, shape_y);

    if ((z >= pmlz0 + 2 * ZPAD) && (z < shape_z - pmlz1 - 2 * ZPAD) &&
        (y >= pmly0 + 2 * YPAD) && (y < shape_y - pmly1 - 2 * YPAD)) {
      /* Update wavefield */
      wfn[si] = model[i] * lap + 2 * wfc[si] - wfp[si];
    } else {
      /* Inside PML region */

      const DEEPWAVE_TYPE wfc_z = z_deriv(wfc, si, shape_y);
      const DEEPWAVE_TYPE phizc_z = z_deriv(phizc, si, shape_y);
      const DEEPWAVE_TYPE wfc_y = y_deriv(wfc, si, 1);
      const DEEPWAVE_TYPE phiyc_y = y_deriv(phiyc, si, 1);

      /* Update wavefield */
      wfn[si] =
          1 / (1 + dt * (sigmaz[z] + sigmay[y]) / 2) *
          (model[i] * (lap + phizc_z + phiyc_y) +
           dt * (sigmaz[z] + sigmay[y]) * wfp[si] / 2 +
           (2 * wfc[si] - wfp[si]) - dt * dt * sigmaz[z] * sigmay[y] * wfc[si]);

      /* Update phi */
      phizn[si] = phizc[si] - dt * (sigmaz[z] * phizc[si] +
                                    (sigmaz[z] - sigmay[y]) * wfc_z);
      phiyn[si] = phiyc[si] - dt * (sigmay[y] * phiyc[si] +
                                    (sigmay[y] - sigmaz[z]) * wfc_y);
    }
  }
}

void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next auxiliary */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current auxiliary */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1_d, /* 1st diff coeffs */
               const DEEPWAVE_TYPE * const fd2_d, /* 2nd diff coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t numel_shot = shape[0] * shape[1];
  DEEPWAVE_TYPE * const phizn = auxn;
  const DEEPWAVE_TYPE * const phizc = auxc;
  const DEEPWAVE_TYPE * const sigmaz = sigma;

  DEEPWAVE_TYPE * const phiyn = auxn + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const phiyc = auxc + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];

  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[1] - (2 * YPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (shape[0] - (2 * ZPAD) + dimBlock.y - 1) / dimBlock.y;
  const int gridz = (num_shots + dimBlock.z - 1) / dimBlock.z;
  const dim3 dimGrid(gridx, gridy, gridz);
  propagate_kernel<<<dimGrid, dimBlock>>>(
      wfn, phizn, phiyn, wfc, wfp, phizc, sigmaz, phiyc, sigmay, model,
      shape[0], shape[1], numel_shot, num_shots, pml_width[0], pml_width[1],
      pml_width[2], pml_width[3], dt);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ imaging_condition_kernel(
    DEEPWAVE_TYPE * const model_grad,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield_t,
    const DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const sigmaz,
    const DEEPWAVE_TYPE * const sigmay, const ptrdiff_t shape_z,
    const ptrdiff_t shape_y, const ptrdiff_t numel_shot,
    const ptrdiff_t num_shots) {
  const ptrdiff_t shot = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t z = blockIdx.y * blockDim.y + threadIdx.y + ZPAD;
  const ptrdiff_t y = blockIdx.x * blockDim.x + threadIdx.x + YPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD)) {
    const ptrdiff_t i = z * shape_y + y;
    const ptrdiff_t si = shot * numel_shot + i;

    atomicAdd(model_grad + i,
              current_wavefield[si] *
                  (current_saved_wavefield_tt[si] +
                   (sigmaz[z] + sigmay[y]) * current_saved_wavefield_t[si] +
                   sigmaz[z] * sigmay[y] * current_saved_wavefield[si]));
  }
}

void imaging_condition(
    DEEPWAVE_TYPE * const model_grad,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield_t,
    const DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const sigma,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t num_shots) {
  if (model_grad == NULL) return; /* Not doing model inversion */

  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[1] - (2 * YPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (shape[0] - (2 * ZPAD) + dimBlock.y - 1) / dimBlock.y;
  const int gridz = (num_shots + dimBlock.z - 1) / dimBlock.z;

  const dim3 dimGrid(gridx, gridy, gridz);
  const DEEPWAVE_TYPE * const sigmaz = sigma;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];

  imaging_condition_kernel<<<dimGrid, dimBlock>>>(
      model_grad, current_wavefield, current_saved_wavefield,
      current_saved_wavefield_t, current_saved_wavefield_tt, sigmaz, sigmay,
      shape[0], shape[1], shape[0] * shape[1], num_shots);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ add_scattering_kernel(
    DEEPWAVE_TYPE * const next_scattered_wavefield,
    const DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const previous_wavefield,
    const DEEPWAVE_TYPE * const scatter,
    const ptrdiff_t shape_z,
    const ptrdiff_t shape_y, const ptrdiff_t numel_shot,
    const ptrdiff_t num_shots) {
  const ptrdiff_t shot = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t z = blockIdx.y * blockDim.y + threadIdx.y + ZPAD;
  const ptrdiff_t y = blockIdx.x * blockDim.x + threadIdx.x + YPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD)) {
    const ptrdiff_t i = z * shape_y + y;
    const ptrdiff_t si = shot * numel_shot + i;

    const DEEPWAVE_TYPE current_wavefield_tt =
        (next_wavefield[si] - 2 * current_wavefield[si] +
         previous_wavefield[si]); /* no dt^2 because of cancellation */
    next_scattered_wavefield[si] += current_wavefield_tt * scatter[i];
  }
}

void add_scattering(
                     DEEPWAVE_TYPE * const next_scattered_wavefield,
                     const DEEPWAVE_TYPE * const next_wavefield,
                     const DEEPWAVE_TYPE * const current_wavefield,
                     const DEEPWAVE_TYPE * const previous_wavefield,
                     const DEEPWAVE_TYPE * const scatter,
                     const ptrdiff_t * const shape,
                     const ptrdiff_t num_shots) {
  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[1] - (2 * YPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (shape[0] - (2 * ZPAD) + dimBlock.y - 1) / dimBlock.y;
  const int gridz = (num_shots + dimBlock.z - 1) / dimBlock.z;

  const dim3 dimGrid(gridx, gridy, gridz);
  add_scattering_kernel<<<dimGrid, dimBlock>>>(
      next_scattered_wavefield, next_wavefield,
      current_wavefield, previous_wavefield, scatter, shape[0], shape[1],
      shape[0] * shape[1], num_shots);
  gpuErrchk(cudaPeekAtLastError());
}

void __global__ save_wavefields_kernel(
    DEEPWAVE_TYPE * const current_saved_wavefield_t,
    DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const previous_wavefield, const ptrdiff_t shape_z,
    const ptrdiff_t shape_y, const ptrdiff_t numel_shot,
    const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t shot = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t z = blockIdx.y * blockDim.y + threadIdx.y + ZPAD;
  const ptrdiff_t y = blockIdx.x * blockDim.x + threadIdx.x + YPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD)) {
    const ptrdiff_t i = z * shape_y + y;
    const ptrdiff_t si = shot * numel_shot + i;

    current_saved_wavefield_t[si] =
        (current_wavefield[si] - previous_wavefield[si]) / dt;

    current_saved_wavefield_tt[si] =
        (next_wavefield[si] - 2 * current_wavefield[si] +
         previous_wavefield[si]) /
        dt / dt;
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
    gpuErrchk(
        cudaMemcpy(current_saved_wavefield, current_wavefield,
                   num_shots * shape[0] * shape[1] * shape[2] * sizeof(DEEPWAVE_TYPE),
                   cudaMemcpyDeviceToDevice));

    const dim3 dimBlock(32, 32, 1);
    const int gridx = (shape[1] - (2 * YPAD) + dimBlock.x - 1) / dimBlock.x;
    const int gridy = (shape[0] - (2 * ZPAD) + dimBlock.y - 1) / dimBlock.y;
    const int gridz = (num_shots + dimBlock.z - 1) / dimBlock.z;

    const dim3 dimGrid(gridx, gridy, gridz);
    save_wavefields_kernel<<<dimGrid, dimBlock>>>(
        current_saved_wavefield_t, current_saved_wavefield_tt, next_wavefield,
        current_wavefield, previous_wavefield, shape[0], shape[1],
        shape[0] * shape[1], num_shots, dt);
    gpuErrchk(cudaPeekAtLastError());
  }
}

static inline __device__ ptrdiff_t
location_index(const ptrdiff_t * const arr, const ptrdiff_t shape_y,
               const ptrdiff_t shape_x, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index * 2];
  const ptrdiff_t y = arr[index * 2 + 1];

  return z * shape_y + y;
}

static inline __device__ DEEPWAVE_TYPE laplacian_2d(const DEEPWAVE_TYPE * const arr,
                                           const ptrdiff_t si,
                                           const ptrdiff_t size_x) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + size_x] + arr[si - size_x]) +
         fd2[2] * (arr[si + 2 * size_x] + arr[si - 2 * size_x]) +
         +fd2[3] * (arr[si + 1] + arr[si - 1]) +
         fd2[4] * (arr[si + 2] + arr[si - 2]);
}

#elif DEEPWAVE_DIM == 3
__global__ void propagate_kernel(
    DEEPWAVE_TYPE * const wfn, DEEPWAVE_TYPE * const phizn,
    DEEPWAVE_TYPE * const phiyn, DEEPWAVE_TYPE * const phixn,
    DEEPWAVE_TYPE * const psin, const DEEPWAVE_TYPE * const wfc,
    const DEEPWAVE_TYPE * const wfp, const DEEPWAVE_TYPE * const phizc,
    const DEEPWAVE_TYPE * const sigmaz, const DEEPWAVE_TYPE * const phiyc,
    const DEEPWAVE_TYPE * const sigmay, const DEEPWAVE_TYPE * const phixc,
    const DEEPWAVE_TYPE * const sigmax, const DEEPWAVE_TYPE * const psic,
    const DEEPWAVE_TYPE * const model, const ptrdiff_t shape_z,
    const ptrdiff_t shape_y, const ptrdiff_t shape_x,
    const ptrdiff_t numel_shot, const ptrdiff_t size_xy,
    const ptrdiff_t num_shots, const ptrdiff_t pmlz0, const ptrdiff_t pmlz1,
    const ptrdiff_t pmly0, const ptrdiff_t pmly1, const ptrdiff_t pmlx0,
    const ptrdiff_t pmlx1, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t threadz = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t shot = threadz / (shape_z - ZPAD - ZPAD + 1);
  const ptrdiff_t z = threadz % (shape_z - ZPAD - ZPAD + 1) + ZPAD;
  const ptrdiff_t y = blockIdx.y * blockDim.y + threadIdx.y + YPAD;
  const ptrdiff_t x = blockIdx.x * blockDim.x + threadIdx.x + XPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD) &&
      (x < shape_x - XPAD)) {
    const ptrdiff_t i = z * size_xy + y * shape_x + x;
    const ptrdiff_t si = shot * numel_shot + i;

    const DEEPWAVE_TYPE lap = laplacian_3d(wfc, si, size_xy, shape_x);

    if ((z >= pmlz0 + 2 * ZPAD) && (z < shape_z - pmlz1 - 2 * ZPAD) &&
        (y >= pmly0 + 2 * YPAD) && (y < shape_y - pmly1 - 2 * YPAD) &&
        (x >= pmlx0 + 2 * XPAD) && (x < shape_x - pmlx1 - 2 * XPAD)) {
      /* Update wavefield */
      wfn[si] = model[i] * lap + 2 * wfc[si] - wfp[si];
    } else {
      /* Inside PML region */

      const DEEPWAVE_TYPE wfc_z = z_deriv(wfc, si, size_xy);
      const DEEPWAVE_TYPE wfc_y = y_deriv(wfc, si, shape_x);
      const DEEPWAVE_TYPE wfc_x = x_deriv(wfc, si);
      const DEEPWAVE_TYPE phizc_z = z_deriv(phizc, si, size_xy);
      const DEEPWAVE_TYPE phiyc_y = y_deriv(phiyc, si, shape_x);
      const DEEPWAVE_TYPE phixc_x = x_deriv(phixc, si);
      const DEEPWAVE_TYPE psic_z = z_deriv(psic, si, size_xy);
      const DEEPWAVE_TYPE psic_y = y_deriv(psic, si, shape_x);
      const DEEPWAVE_TYPE psic_x = x_deriv(psic, si);

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

void propagate(DEEPWAVE_TYPE * const wfn,        /* next wavefield */
               DEEPWAVE_TYPE * const auxn,       /* next auxiliary */
               const DEEPWAVE_TYPE * const wfc,  /* current wavefield */
               const DEEPWAVE_TYPE * const wfp,  /* previous wavefield */
               const DEEPWAVE_TYPE * const auxc, /* current auxiliary */
               const DEEPWAVE_TYPE * const sigma,
               const DEEPWAVE_TYPE * const model,
               const DEEPWAVE_TYPE * const fd1_d, /* 1st diff coeffs */
               const DEEPWAVE_TYPE * const fd2_d, /* 2nd diff coeffs */
               const ptrdiff_t * const shape,
               const ptrdiff_t * const pml_width,
               const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];
  const ptrdiff_t size_xy = shape[1] * shape[2];
  DEEPWAVE_TYPE * const phizn = auxn;
  const DEEPWAVE_TYPE * const phizc = auxc;
  const DEEPWAVE_TYPE * const sigmaz = sigma;

  DEEPWAVE_TYPE * const phiyn = auxn + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const phiyc = auxc + num_shots * numel_shot;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];

  DEEPWAVE_TYPE * const phixn = auxn + 2 * num_shots * numel_shot;
  DEEPWAVE_TYPE * const psin = auxn + 3 * num_shots * numel_shot;
  const DEEPWAVE_TYPE * const phixc = auxc + 2 * num_shots * numel_shot;
  const DEEPWAVE_TYPE * const psic = auxc + 3 * num_shots * numel_shot;
  const DEEPWAVE_TYPE * const sigmax = sigma + shape[0] + shape[1];

  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[2] - (2 * XPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (shape[1] - (2 * YPAD) + dimBlock.y - 1) / dimBlock.y;
  const int gridz =
      (num_shots * (shape[0] - (2 * ZPAD)) + dimBlock.z - 1) / dimBlock.z;
  const dim3 dimGrid(gridx, gridy, gridz);
  propagate_kernel<<<dimGrid, dimBlock>>>(
      wfn, phizn, phiyn, phixn, psin, wfc, wfp, phizc, sigmaz, phiyc, sigmay,
      phixc, sigmax, psic, model, shape[0], shape[1], shape[2], numel_shot,
      size_xy, num_shots, pml_width[0], pml_width[1], pml_width[2],
      pml_width[3], pml_width[4], pml_width[5], dt);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ imaging_condition_kernel(
    DEEPWAVE_TYPE * const model_grad,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield_t,
    const DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const sigmaz,
    const DEEPWAVE_TYPE * const sigmay,
    const DEEPWAVE_TYPE * const sigmax, const ptrdiff_t shape_z,
    const ptrdiff_t shape_y, const ptrdiff_t shape_x,
    const ptrdiff_t numel_shot, const ptrdiff_t size_xy,
    const ptrdiff_t num_shots) {
  const ptrdiff_t threadz = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t shot = threadz / (shape_z - ZPAD - ZPAD + 1);
  const ptrdiff_t z = threadz % (shape_z - ZPAD - ZPAD + 1) + ZPAD;
  const ptrdiff_t y = blockIdx.y * blockDim.y + threadIdx.y + YPAD;
  const ptrdiff_t x = blockIdx.x * blockDim.x + threadIdx.x + XPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD) &&
      (x < shape_x - XPAD)) {
    const ptrdiff_t i = z * size_xy + y * shape_x + x;
    const ptrdiff_t si = shot * numel_shot + i;

    /* NOTE: There should be an additional term here (involving spatial
     * derivative of phi, sigma, and psi), but it is neglected due to
     * the additional computational cost it would cause. */
    atomicAdd(model_grad + i,
              current_wavefield[si] *
                  (current_saved_wavefield_tt[si] +
                   (sigmaz[z] + sigmay[y] + sigmax[x]) *
                       current_saved_wavefield_t[si] +
                   (sigmax[x] * sigmay[y] + sigmay[y] * sigmaz[z] +
                    sigmax[x] * sigmaz[z]) *
                       current_saved_wavefield[si]));
  }
}

void imaging_condition(
    DEEPWAVE_TYPE * const model_grad,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield,
    const DEEPWAVE_TYPE * const current_saved_wavefield_t,
    const DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const sigma,
    const ptrdiff_t * const shape,
    const ptrdiff_t * const pml_width, const ptrdiff_t num_shots) {
  if (model_grad == NULL) return; /* Not doing model inversion */

  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[2] - (2 * XPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (shape[1] - (2 * YPAD) + dimBlock.y - 1) / dimBlock.y;
  const int gridz =
      (num_shots * (shape[0] - (2 * ZPAD)) + dimBlock.z - 1) / dimBlock.z;

  const dim3 dimGrid(gridx, gridy, gridz);
  const DEEPWAVE_TYPE * const sigmaz = sigma;
  const DEEPWAVE_TYPE * const sigmay = sigma + shape[0];
  const DEEPWAVE_TYPE * const sigmax = sigma + shape[0] + shape[1];

  imaging_condition_kernel<<<dimGrid, dimBlock>>>(
      model_grad, current_wavefield, current_saved_wavefield,
      current_saved_wavefield_t, current_saved_wavefield_tt, sigmaz, sigmay,
      sigmax, shape[0], shape[1], shape[2], shape[0] * shape[1] * shape[2],
      shape[1] * shape[2], num_shots);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ add_scattering_kernel(
    DEEPWAVE_TYPE * const next_scattered_wavefield,
    const DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const previous_wavefield,
    const DEEPWAVE_TYPE * const scatter,
    const ptrdiff_t shape_z,
    const ptrdiff_t shape_y,
    const ptrdiff_t shape_x,
    const ptrdiff_t numel_shot, const ptrdiff_t size_xy,
    const ptrdiff_t num_shots) {
  const ptrdiff_t threadz = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t shot = threadz / (shape_z - ZPAD - ZPAD + 1);
  const ptrdiff_t z = threadz % (shape_z - ZPAD - ZPAD + 1) + ZPAD;
  const ptrdiff_t y = blockIdx.y * blockDim.y + threadIdx.y + YPAD;
  const ptrdiff_t x = blockIdx.x * blockDim.x + threadIdx.x + XPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD) &&
      (x < shape_x - XPAD)) {
    const ptrdiff_t i = z * size_xy + y * shape_x + x;
    const ptrdiff_t si = shot * numel_shot + i;

    const DEEPWAVE_TYPE current_wavefield_tt =
        (next_wavefield[si] - 2 * current_wavefield[si] +
         previous_wavefield[si]); /* no dt^2 because of cancellation */
    next_scattered_wavefield[si] += current_wavefield_tt * scatter[i];
  }
}

void add_scattering(
                     DEEPWAVE_TYPE * const next_scattered_wavefield,
                     const DEEPWAVE_TYPE * const next_wavefield,
                     const DEEPWAVE_TYPE * const current_wavefield,
                     const DEEPWAVE_TYPE * const previous_wavefield,
                     const DEEPWAVE_TYPE * const scatter,
                     const ptrdiff_t * const shape,
                     const ptrdiff_t num_shots) {
  const dim3 dimBlock(32, 32, 1);
  const int gridx = (shape[2] - (2 * XPAD) + dimBlock.x - 1) / dimBlock.x;
  const int gridy = (shape[1] - (2 * YPAD) + dimBlock.y - 1) / dimBlock.y;
  const int gridz =
      (num_shots * (shape[0] - (2 * ZPAD)) + dimBlock.z - 1) / dimBlock.z;

  const dim3 dimGrid(gridx, gridy, gridz);
  add_scattering_kernel<<<dimGrid, dimBlock>>>(
      next_scattered_wavefield, next_wavefield,
      current_wavefield, previous_wavefield, scatter, shape[0], shape[1],
      shape[2], shape[0] * shape[1] * shape[2], shape[1] * shape[2], num_shots);
  gpuErrchk(cudaPeekAtLastError());
}

void __global__ save_wavefields_kernel(
    DEEPWAVE_TYPE * const current_saved_wavefield_t,
    DEEPWAVE_TYPE * const current_saved_wavefield_tt,
    const DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const current_wavefield,
    const DEEPWAVE_TYPE * const previous_wavefield, const ptrdiff_t shape_z,
    const ptrdiff_t shape_y, const ptrdiff_t shape_x,
    const ptrdiff_t numel_shot, const ptrdiff_t size_xy,
    const ptrdiff_t num_shots, const DEEPWAVE_TYPE dt) {
  const ptrdiff_t threadz = blockIdx.z * blockDim.z + threadIdx.z;
  const ptrdiff_t shot = threadz / (shape_z - ZPAD - ZPAD + 1);
  const ptrdiff_t z = threadz % (shape_z - ZPAD - ZPAD + 1) + ZPAD;
  const ptrdiff_t y = blockIdx.y * blockDim.y + threadIdx.y + YPAD;
  const ptrdiff_t x = blockIdx.x * blockDim.x + threadIdx.x + XPAD;
  if ((shot < num_shots) && (z < shape_z - ZPAD) && (y < shape_y - YPAD) &&
      (x < shape_x - XPAD)) {
    const ptrdiff_t i = z * size_xy + y * shape_x + x;
    const ptrdiff_t si = shot * numel_shot + i;

    current_saved_wavefield_t[si] =
        (next_wavefield[si] - previous_wavefield[si]) / 2 / dt;

    current_saved_wavefield_tt[si] =
        (next_wavefield[si] - 2 * current_wavefield[si] +
         previous_wavefield[si]) /
        dt / dt;
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
    gpuErrchk(
        cudaMemcpy(current_saved_wavefield, current_wavefield,
                   num_shots * shape[0] * shape[1] * shape[2] * sizeof(DEEPWAVE_TYPE),
                   cudaMemcpyDeviceToDevice));
    const dim3 dimBlock(32, 32, 1);
    const int gridx = (shape[2] - (2 * XPAD) + dimBlock.x - 1) / dimBlock.x;
    const int gridy = (shape[1] - (2 * YPAD) + dimBlock.y - 1) / dimBlock.y;
    const int gridz =
        (num_shots * (shape[0] - (2 * ZPAD)) + dimBlock.z - 1) / dimBlock.z;

    const dim3 dimGrid(gridx, gridy, gridz);
    save_wavefields_kernel<<<dimGrid, dimBlock>>>(
        current_saved_wavefield_t, current_saved_wavefield_tt, next_wavefield,
        current_wavefield, previous_wavefield, shape[0], shape[1], shape[2],
        shape[0] * shape[1] * shape[2], shape[1] * shape[2], num_shots, dt);
    gpuErrchk(cudaPeekAtLastError());
  }
}

static inline __device__ ptrdiff_t
location_index(const ptrdiff_t * const arr, const ptrdiff_t shape_y,
               const ptrdiff_t shape_x, const ptrdiff_t index) {
  const ptrdiff_t z = arr[index * 3];
  const ptrdiff_t y = arr[index * 3 + 1];
  const ptrdiff_t x = arr[index * 3 + 2];

  return z * shape_y * shape_x + y * shape_x + x;
}

static inline __device__ DEEPWAVE_TYPE laplacian_3d(const DEEPWAVE_TYPE * const arr,
                                           const ptrdiff_t si,
                                           const ptrdiff_t size_x,
                                           const ptrdiff_t size_xy) {
  return fd2[0] * arr[si] + fd2[1] * (arr[si + size_xy] + arr[si - size_xy]) +
         fd2[2] * (arr[si + 2 * size_xy] + arr[si - 2 * size_xy]) +
         +fd2[3] * (arr[si + size_x] + arr[si - size_x]) +
         fd2[4] * (arr[si + 2 * size_x] + arr[si - 2 * size_x]) +
         fd2[5] * (arr[si + 1] + arr[si - 1]) +
         fd2[6] * (arr[si + 2] + arr[si - 2]);
}

#else
#error "Must specify the dimension, e.g. -D DEEPWAVE_DIM=1"
#endif /* DEEPWAVE_DIM */

void setup(const DEEPWAVE_TYPE * const fd1_d,
           const DEEPWAVE_TYPE * const fd2_d) {
  gpuErrchk(cudaMemcpyToSymbol(fd1, fd1_d, 2 * DEEPWAVE_DIM * sizeof(DEEPWAVE_TYPE)));
  gpuErrchk(cudaMemcpyToSymbol(fd2, fd2_d, (2 * DEEPWAVE_DIM + 1) * sizeof(DEEPWAVE_TYPE)));
}

void __global__ add_sources_kernel(
    DEEPWAVE_TYPE * const next_wavefield,
    const DEEPWAVE_TYPE * const model,
    const DEEPWAVE_TYPE * const source_amplitudes,
    const ptrdiff_t * const source_locations,
    const ptrdiff_t shape_z, const ptrdiff_t shape_y, const ptrdiff_t shape_x,
    const ptrdiff_t num_shots, const ptrdiff_t num_sources_per_shot) {
  ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
  ptrdiff_t source = blockIdx.x * blockDim.x + threadIdx.x;

  if ((shot < num_shots) && (source < num_sources_per_shot)) {
    ptrdiff_t s = shot * num_sources_per_shot + source;
    ptrdiff_t i = location_index(source_locations, shape_y, shape_x, s);
    ptrdiff_t si = shot * shape_z * shape_y * shape_x + i;
    atomicAdd(next_wavefield + si, source_amplitudes[s] * model[i]);
  }
}

void add_sources(DEEPWAVE_TYPE * const next_wavefield,
                 const DEEPWAVE_TYPE * const model,
                 const DEEPWAVE_TYPE * const source_amplitudes,
                 const ptrdiff_t * const source_locations,
                 const ptrdiff_t * const shape,
                 const ptrdiff_t num_shots,
                 const ptrdiff_t num_sources_per_shot) {
  dim3 dimBlock(32, 1, 1);
  int gridx = (num_sources_per_shot + dimBlock.x - 1) / dimBlock.x;
  int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
  int gridz = 1;
  dim3 dimGrid(gridx, gridy, gridz);

  add_sources_kernel<<<dimGrid, dimBlock>>>(
      next_wavefield, model, source_amplitudes, source_locations, shape[0],
      shape[1], shape[2], num_shots, num_sources_per_shot);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ record_receivers_kernel(
    DEEPWAVE_TYPE * const receiver_amplitudes,
    const DEEPWAVE_TYPE * const current_wavefield,
    const ptrdiff_t * const receiver_locations,
    const ptrdiff_t shape_z, const ptrdiff_t shape_y, const ptrdiff_t shape_x,
    const ptrdiff_t num_shots, const ptrdiff_t num_receivers_per_shot) {
  ptrdiff_t shot = blockIdx.y * blockDim.y + threadIdx.y;
  ptrdiff_t receiver = blockIdx.x * blockDim.x + threadIdx.x;

  if ((shot < num_shots) && (receiver < num_receivers_per_shot)) {
    ptrdiff_t r = shot * num_receivers_per_shot + receiver;
    ptrdiff_t si = shot * shape_z * shape_y * shape_x +
                   location_index(receiver_locations, shape_y, shape_x, r);
    receiver_amplitudes[r] = current_wavefield[si];
  }
}

void record_receivers(DEEPWAVE_TYPE * const receiver_amplitudes,
                      const DEEPWAVE_TYPE * const current_wavefield,
                      const ptrdiff_t * const receiver_locations,
                      const ptrdiff_t * const shape,
                      const ptrdiff_t num_shots,
                      const ptrdiff_t num_receivers_per_shot) {
  if (receiver_amplitudes == NULL) return; /* no source inversion */

  dim3 dimBlock(32, 1, 1);
  int gridx = (num_receivers_per_shot + dimBlock.x - 1) / dimBlock.x;
  int gridy = (num_shots + dimBlock.y - 1) / dimBlock.y;
  int gridz = 1;
  dim3 dimGrid(gridx, gridy, gridz);

  record_receivers_kernel<<<dimGrid, dimBlock>>>(
      receiver_amplitudes, current_wavefield, receiver_locations, shape[0],
      shape[1], shape[2], num_shots, num_receivers_per_shot);

  gpuErrchk(cudaPeekAtLastError());
}

void __global__ model_grad_scaling_kernel(
    DEEPWAVE_TYPE * const model_grad, const DEEPWAVE_TYPE * const scaling,
    const ptrdiff_t numel_shot) {
  ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numel_shot) {
    model_grad[i] *= scaling[i];
  }
}

void model_grad_scaling(DEEPWAVE_TYPE * const model_grad,
                        const DEEPWAVE_TYPE * const scaling,
                        const ptrdiff_t * const shape,
                        const ptrdiff_t * const pml_width) {
  if (model_grad == NULL) return; /* Not doing model inversion */

  const ptrdiff_t numel_shot = shape[0] * shape[1] * shape[2];

  dim3 dimBlock(32, 1, 1);
  int gridx = (numel_shot + dimBlock.x - 1) / dimBlock.x;
  int gridy = 1;
  int gridz = 1;
  dim3 dimGrid(gridx, gridy, gridz);

  model_grad_scaling_kernel<<<dimGrid, dimBlock>>>(model_grad, scaling,
                                                   numel_shot);

  gpuErrchk(cudaPeekAtLastError());
}

static inline __device__ DEEPWAVE_TYPE z_deriv(const DEEPWAVE_TYPE * const arr,
                                      const ptrdiff_t si,
                                      const ptrdiff_t size_xy) {
  return fd1[0] * (arr[si + size_xy] - arr[si - size_xy]) +
         fd1[1] * (arr[si + 2 * size_xy] - arr[si - 2 * size_xy]);
}

static inline __device__ DEEPWAVE_TYPE y_deriv(const DEEPWAVE_TYPE * const arr,
                                      const ptrdiff_t si,
                                      const ptrdiff_t size_x) {
  return fd1[0] * (arr[si + size_x] - arr[si - size_x]) +
         fd1[1] * (arr[si + 2 * size_x] - arr[si - 2 * size_x]);
}

static inline __device__ DEEPWAVE_TYPE x_deriv(const DEEPWAVE_TYPE * const arr,
                                      const ptrdiff_t si) {
  return fd1[0] * (arr[si + 1] - arr[si - 1]) +
         fd1[1] * (arr[si + 2] - arr[si - 2]);
}

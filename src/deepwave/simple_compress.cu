#include <stdint.h>
#include <string.h>

#include "common_gpu.h"
#include "simple_compress.h"

/* CUDA kernel for finding min/max per field */
template <typename T>
__launch_bounds__(256) __global__
    void find_minmax_kernel(T const *const input, T *const minmax,
                            size_t const n_elements_per_field,
                            size_t const n_batch) {
  size_t const b = blockIdx.x;
  if (b >= n_batch) return;

  T const *const field = input + b * n_elements_per_field;

  /* Shared memory for reduction */
  extern __shared__ char shared_mem[];
  T *const s_min = (T *)shared_mem;
  T *const s_max = (T *)(shared_mem + blockDim.x * sizeof(T));

  /* Each thread finds local min/max */
  T local_min, local_max;
  if (threadIdx.x < n_elements_per_field) {
    local_min = field[threadIdx.x];
    local_max = field[threadIdx.x];
  } else if (n_elements_per_field > 0) {
    local_min = field[0];
    local_max = field[0];
  } else {
    local_min = 0;
    local_max = 0;
  }

  for (size_t i = threadIdx.x + blockDim.x; i < n_elements_per_field;
       i += blockDim.x) {
    T val = field[i];
    if (val < local_min) local_min = val;
    if (val > local_max) local_max = val;
  }

  s_min[threadIdx.x] = local_min;
  s_max[threadIdx.x] = local_max;
  __syncthreads();

  /* Reduction in shared memory */
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (s_min[threadIdx.x + s] < s_min[threadIdx.x])
        s_min[threadIdx.x] = s_min[threadIdx.x + s];
      if (s_max[threadIdx.x + s] > s_max[threadIdx.x])
        s_max[threadIdx.x] = s_max[threadIdx.x + s];
    }
    __syncthreads();
  }

  /* Write result */
  if (threadIdx.x == 0) {
    minmax[2 * b] = s_min[0];
    minmax[2 * b + 1] = s_max[0];
  }
}

/* CUDA kernel for compression */
template <typename T>
__launch_bounds__(256) __global__
    void compress_kernel(T const *const input, uint8_t *const output,
                         T const *const minmax,
                         size_t const n_elements_per_field,
                         size_t const n_batch) {
  size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const total = n_batch * n_elements_per_field;

  if (idx >= total) return;

  size_t const b = idx / n_elements_per_field;
  T min_val = minmax[2 * b];
  T max_val = minmax[2 * b + 1];
  T range = max_val - min_val;
  T scale = (range > 0) ? (T(255) / range) : T(0);

  T normalized = (input[idx] - min_val) * scale;
  output[idx] = (uint8_t)(normalized + T(0.5));
}

/* CUDA kernel for decompression */
template <typename T>
__launch_bounds__(256) __global__
    void decompress_kernel(uint8_t const *const input, T *const output,
                           const T *const minmax,
                           size_t const n_elements_per_field,
                           size_t const n_batch) {
  size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const total = n_batch * n_elements_per_field;

  if (idx >= total) return;

  size_t const b = idx / n_elements_per_field;
  T min_val = minmax[2 * b];
  T max_val = minmax[2 * b + 1];
  T range = max_val - min_val;
  T scale = range / T(255);

  output[idx] = min_val + input[idx] * scale;
}

/* CUDA wrapper functions */
extern "C" {

static int simple_compress_cuda_float(float const *const input,
                                      uint8_t *const output,
                                      size_t const n_batch,
                                      size_t const n_elements_per_field,
                                      cudaStream_t stream) {
  float *const minmax = (float *)output;
  uint8_t *const compressed = output + 2 * n_batch * sizeof(float);

  /* Find min/max for each field */
  size_t const threads = 256;
  size_t const shared_mem = 2 * threads * sizeof(float);
  find_minmax_kernel<<<n_batch, threads, shared_mem, stream>>>(
      input, minmax, n_elements_per_field, n_batch);

  /* Compress */
  size_t const total = n_batch * n_elements_per_field;
  size_t const blocks = (total + threads - 1) / threads;
  compress_kernel<<<blocks, threads, 0, stream>>>(
      input, compressed, minmax, n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

static int simple_compress_cuda_double(double const *const input,
                                       uint8_t *const output,
                                       size_t const n_batch,
                                       size_t const n_elements_per_field,
                                       cudaStream_t stream) {
  double *const minmax = (double *)output;
  uint8_t *const compressed = output + 2 * n_batch * sizeof(double);

  /* Find min/max for each field */
  size_t const threads = 256;
  size_t const shared_mem = 2 * threads * sizeof(double);
  find_minmax_kernel<<<n_batch, threads, shared_mem, stream>>>(
      input, minmax, n_elements_per_field, n_batch);

  /* Compress */
  size_t const total = n_batch * n_elements_per_field;
  size_t const blocks = (total + threads - 1) / threads;
  compress_kernel<<<blocks, threads, 0, stream>>>(
      input, compressed, minmax, n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

static int simple_decompress_cuda_float(uint8_t const *const input,
                                        float *const output,
                                        size_t const n_batch,
                                        size_t const n_elements_per_field,
                                        cudaStream_t stream) {
  float const *const minmax = (const float *)input;
  uint8_t const *const compressed = input + 2 * n_batch * sizeof(float);

  size_t const threads = 256;
  size_t const total = n_batch * n_elements_per_field;
  size_t const blocks = (total + threads - 1) / threads;

  decompress_kernel<<<blocks, threads, 0, stream>>>(
      compressed, output, minmax, n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

static int simple_decompress_cuda_double(uint8_t const *const input,
                                         double *const output,
                                         size_t const n_batch,
                                         size_t const n_elements_per_field,
                                         cudaStream_t stream) {
  double const *const minmax = (const double *)input;
  uint8_t const *const compressed = input + 2 * n_batch * sizeof(double);

  size_t const threads = 256;
  size_t const total = n_batch * n_elements_per_field;
  size_t const blocks = (total + threads - 1) / threads;

  decompress_kernel<<<blocks, threads, 0, stream>>>(
      compressed, output, minmax, n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

/* Update the main compress/decompress functions to call CUDA versions */
int simple_compress_cuda(void const *const input, void *const output,
                         size_t const n_batch,
                         size_t const n_elements_per_field, int const is_double,
                         void *const stream) {
  if (is_double) {
    return simple_compress_cuda_double((double const *)input, (uint8_t *)output,
                                       n_batch, n_elements_per_field,
                                       (cudaStream_t)stream);
  } else {
    return simple_compress_cuda_float((float const *)input, (uint8_t *)output,
                                      n_batch, n_elements_per_field,
                                      (cudaStream_t)stream);
  }
}

int simple_decompress_cuda(void const *const input, void *const output,
                           size_t const n_batch,
                           size_t const n_elements_per_field,
                           int const is_double, void *const stream) {
  if (is_double) {
    return simple_decompress_cuda_double(
        (uint8_t const *)input, (double *)output, n_batch, n_elements_per_field,
        (cudaStream_t)stream);
  } else {
    return simple_decompress_cuda_float((uint8_t const *)input, (float *)output,
                                        n_batch, n_elements_per_field,
                                        (cudaStream_t)stream);
  }
}

} /* extern "C" */

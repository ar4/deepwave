#include <stdint.h>
#include <string.h>

#include "common_gpu.h"
#include "simple_compress.h"

/* CUDA kernel for finding min/max per field */
template <typename T>
__launch_bounds__(256) __global__
    void find_minmax_kernel(const T *input, T *minmax,
                            size_t n_elements_per_field, size_t n_batch) {
  size_t b = blockIdx.x;
  if (b >= n_batch) return;

  const T *field = input + b * n_elements_per_field;

  /* Shared memory for reduction */
  extern __shared__ char shared_mem[];
  T *s_min = (T *)shared_mem;
  T *s_max = (T *)(shared_mem + blockDim.x * sizeof(T));

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
    void compress_kernel(const T *input, uint8_t *output, const T *minmax,
                         size_t n_elements_per_field, size_t n_batch) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = n_batch * n_elements_per_field;

  if (idx >= total) return;

  size_t b = idx / n_elements_per_field;
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
    void decompress_kernel(const uint8_t *input, T *output, const T *minmax,
                           size_t n_elements_per_field, size_t n_batch) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = n_batch * n_elements_per_field;

  if (idx >= total) return;

  size_t b = idx / n_elements_per_field;
  T min_val = minmax[2 * b];
  T max_val = minmax[2 * b + 1];
  T range = max_val - min_val;
  T scale = range / T(255);

  output[idx] = min_val + input[idx] * scale;
}

/* CUDA wrapper functions */
extern "C" {

static int simple_compress_cuda_float(const float *input, uint8_t *output,
                                      size_t n_batch,
                                      size_t n_elements_per_field) {
  float *minmax = (float *)output;
  uint8_t *compressed = output + 2 * n_batch * sizeof(float);

  /* Find min/max for each field */
  size_t threads = 256;
  size_t shared_mem = 2 * threads * sizeof(float);
  find_minmax_kernel<<<n_batch, threads, shared_mem>>>(
      input, minmax, n_elements_per_field, n_batch);

  /* Compress */
  size_t total = n_batch * n_elements_per_field;
  size_t blocks = (total + threads - 1) / threads;
  compress_kernel<<<blocks, threads>>>(input, compressed, minmax,
                                       n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

static int simple_compress_cuda_double(const double *input, uint8_t *output,
                                       size_t n_batch,
                                       size_t n_elements_per_field) {
  double *minmax = (double *)output;
  uint8_t *compressed = output + 2 * n_batch * sizeof(double);

  /* Find min/max for each field */
  size_t threads = 256;
  size_t shared_mem = 2 * threads * sizeof(double);
  find_minmax_kernel<<<n_batch, threads, shared_mem>>>(
      input, minmax, n_elements_per_field, n_batch);

  /* Compress */
  size_t total = n_batch * n_elements_per_field;
  size_t blocks = (total + threads - 1) / threads;
  compress_kernel<<<blocks, threads>>>(input, compressed, minmax,
                                       n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

static int simple_decompress_cuda_float(const uint8_t *input, float *output,
                                        size_t n_batch,
                                        size_t n_elements_per_field) {
  const float *minmax = (const float *)input;
  const uint8_t *compressed = input + 2 * n_batch * sizeof(float);

  size_t threads = 256;
  size_t total = n_batch * n_elements_per_field;
  size_t blocks = (total + threads - 1) / threads;

  decompress_kernel<<<blocks, threads>>>(compressed, output, minmax,
                                         n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

static int simple_decompress_cuda_double(const uint8_t *input, double *output,
                                         size_t n_batch,
                                         size_t n_elements_per_field) {
  const double *minmax = (const double *)input;
  const uint8_t *compressed = input + 2 * n_batch * sizeof(double);

  size_t threads = 256;
  size_t total = n_batch * n_elements_per_field;
  size_t blocks = (total + threads - 1) / threads;

  decompress_kernel<<<blocks, threads>>>(compressed, output, minmax,
                                         n_elements_per_field, n_batch);
  CHECK_KERNEL_ERROR
  return 0;
}

/* Update the main compress/decompress functions to call CUDA versions */
int simple_compress_cuda(const void *input, void *output, size_t n_batch,
                         size_t n_elements_per_field, int is_double) {
  if (is_double) {
    return simple_compress_cuda_double((const double *)input, (uint8_t *)output,
                                       n_batch, n_elements_per_field);
  } else {
    return simple_compress_cuda_float((const float *)input, (uint8_t *)output,
                                      n_batch, n_elements_per_field);
  }
}

int simple_decompress_cuda(const void *input, void *output, size_t n_batch,
                           size_t n_elements_per_field, int is_double) {
  if (is_double) {
    return simple_decompress_cuda_double((const uint8_t *)input,
                                         (double *)output, n_batch,
                                         n_elements_per_field);
  } else {
    return simple_decompress_cuda_float((const uint8_t *)input, (float *)output,
                                        n_batch, n_elements_per_field);
  }
}

} /* extern "C" */

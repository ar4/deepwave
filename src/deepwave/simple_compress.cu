#include <float.h>
#include <stdint.h>
#include <string.h>

#include "common_gpu.h"
#include "simple_compress.h"

#define BLOCK_SIZE 8
#if DW_NDIM == 1
#define THREADS_PER_BLOCK BLOCK_SIZE
#elif DW_NDIM == 2
#define THREADS_PER_BLOCK (BLOCK_SIZE * BLOCK_SIZE)
#else
#define THREADS_PER_BLOCK (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE)
#endif

#if defined(DW_NDIM) && defined(DW_DTYPE)

// Fast-math flushes some float subnormal conversions/comparisons to zero.
// Recover their exact value from the IEEE-754 significand for the rare tiny
// compression path. Double conversions do not have this float FTZ behaviour.
__device__ inline double dw_wide_value(float value) {
  uint32_t const bits = __float_as_uint(value);
  if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0) {
    double const magnitude = (double)(bits & 0x007fffffu) * 0x1p-149;
    return (bits & 0x80000000u) ? -magnitude : magnitude;
  }
  return (double)value;
}

__device__ inline double dw_wide_value(double value) { return value; }

__device__ inline float dw_abs_value(float value) {
  return __uint_as_float(__float_as_uint(value) & 0x7fffffffu);
}

__device__ inline double dw_abs_value(double value) {
  return value >= 0 ? value : -value;
}

__device__ inline bool dw_larger_abs(float left, float right) {
  // Both operands are nonnegative, so IEEE-754 bit order matches value order.
  // Integer comparison also preserves subnormals under CUDA fast-math.
  return __float_as_uint(left) > __float_as_uint(right);
}

__device__ inline bool dw_larger_abs(double left, double right) {
  return left > right;
}

__device__ inline bool dw_in_tiny_range(float value, float limit) {
  uint32_t const bits = __float_as_uint(value);
  return bits != 0 && bits <= __float_as_uint(limit);
}

__device__ inline bool dw_in_tiny_range(double value, double limit) {
  return value > 0 && value <= limit;
}

__device__ inline bool dw_in_huge_range(float value, float limit) {
  return __float_as_uint(value) > __float_as_uint(limit);
}

__device__ inline bool dw_in_huge_range(double value, double limit) {
  return value > limit;
}

template <typename T>
__device__ inline T dw_narrow_value(double value);

template <>
__device__ inline float dw_narrow_value<float>(double value) {
  double const magnitude = value >= 0 ? value : -value;
  if (magnitude > 0 && magnitude < (double)FLT_MIN) {
    uint32_t const significand = __double2uint_rn(magnitude * 0x1p149);
    uint32_t const sign = value < 0 ? 0x80000000u : 0;
    return __uint_as_float(sign | significand);
  }
  return (float)value;
}

template <>
__device__ inline double dw_narrow_value<double>(double value) {
  return value;
}

/* CUDA kernel for compression */
__launch_bounds__(THREADS_PER_BLOCK) __global__
    void compress_kernel(DW_DTYPE const* const input, uint8_t* const output,
#if DW_NDIM >= 3
                         size_t const nz,
#endif
#if DW_NDIM >= 2
                         size_t const ny,
#endif
                         size_t const nx,
#if DW_NDIM >= 3
                         size_t const nbz,
#endif
#if DW_NDIM >= 2
                         size_t const nby,
#endif
                         size_t const nbx, size_t const n_blocks_per_shot) {
  size_t const global_block_idx = blockIdx.x;
  size_t const batch_idx = global_block_idx / n_blocks_per_shot;
  size_t const block_idx_in_shot = global_block_idx % n_blocks_per_shot;

  size_t const bx = block_idx_in_shot % nbx;
#if DW_NDIM >= 2
  size_t const tmp = block_idx_in_shot / nbx;
  size_t const by = tmp % nby;
#if DW_NDIM >= 3
  size_t const bz = tmp / nby;
#endif
#endif

  size_t const tid = threadIdx.x;
  size_t const tx = tid % BLOCK_SIZE;
#if DW_NDIM >= 2
  size_t const tmp_t = tid / BLOCK_SIZE;
  size_t const ty = tmp_t % BLOCK_SIZE;
#if DW_NDIM >= 3
  size_t const tz = tmp_t / BLOCK_SIZE;
#endif
#endif

  size_t const x = bx * BLOCK_SIZE + tx;
#if DW_NDIM >= 2
  size_t const y = by * BLOCK_SIZE + ty;
#endif
#if DW_NDIM >= 3
  size_t const z = bz * BLOCK_SIZE + tz;
#endif

  DW_DTYPE val = 0;
  bool in_bounds = (x < nx
#if DW_NDIM >= 2
                    && y < ny
#endif
#if DW_NDIM >= 3
                    && z < nz
#endif
  );
  size_t idx = 0;

  if (in_bounds) {
    idx = batch_idx * (
#if DW_NDIM >= 3
                          nz *
#endif
#if DW_NDIM >= 2
                          ny *
#endif
                          nx) +
#if DW_NDIM >= 3
          z * ny * nx +
#endif
#if DW_NDIM >= 2
          y * nx +
#endif
          x;
    val = input[idx];
  }

  /* Shared memory for reduction */
  extern __shared__ char shared_mem[];
  DW_DTYPE* const s_max_abs = (DW_DTYPE*)shared_mem;

  // Compare in double precision: CUDA fast-math may flush float subnormals
  // to zero in float comparisons, even though the input bits are nonzero.
  s_max_abs[tid] = dw_abs_value(val);
  __syncthreads();

  /* Reduction to find max abs in block */
  for (unsigned int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1U) {
    if (tid < s) {
      if (dw_larger_abs(s_max_abs[tid + s], s_max_abs[tid])) {
        s_max_abs[tid] = s_max_abs[tid + s];
      }
    }
    __syncthreads();
  }

  DW_DTYPE max_abs = s_max_abs[0];

  /* Store max_abs */
  DW_DTYPE* const max_abs_vals = (DW_DTYPE*)output;
  if (tid == 0) {
    max_abs_vals[global_block_idx] = max_abs;
  }

  /* Quantise and store */
  uint8_t* const compressed_data_start = output + gridDim.x * sizeof(DW_DTYPE);

  if (in_bounds) {
    DW_DTYPE const tiny_limit = sizeof(DW_DTYPE) == sizeof(float)
                                    ? (DW_DTYPE)(127.0 / FLT_MAX)
                                    : (DW_DTYPE)(127.0 / DBL_MAX);
    DW_DTYPE const huge_limit = sizeof(DW_DTYPE) == sizeof(float)
                                    ? (DW_DTYPE)(FLT_MAX / 8.0)
                                    : (DW_DTYPE)(DBL_MAX / 8.0);
    bool const tiny = dw_in_tiny_range(max_abs, tiny_limit);
    // Fast-math may form a reciprocal before multiplying by 127. Near the
    // dtype maximum that reciprocal flushes to zero (float) or loses range.
    bool const huge = dw_in_huge_range(max_abs, huge_limit);
    DW_DTYPE const scale = max_abs > 0 && !tiny && !huge
                               ? (DW_DTYPE)(127.0) / max_abs
                               : (DW_DTYPE)(0);
    DW_DTYPE const normalised =
        tiny || huge
            ? (DW_DTYPE)(__ddiv_rn(dw_wide_value(val), dw_wide_value(max_abs)) *
                         127.0)
            : val * scale;
    compressed_data_start[idx] =
        (uint8_t)((DW_DTYPE)(128.0) + normalised + (DW_DTYPE)(0.5));
  }
}

/* CUDA kernel for decompression */
__launch_bounds__(THREADS_PER_BLOCK) __global__
    void decompress_kernel(uint8_t const* const input, DW_DTYPE* const output,
#if DW_NDIM >= 3
                           size_t const nz,
#endif
#if DW_NDIM >= 2
                           size_t const ny,
#endif
                           size_t const nx,
#if DW_NDIM >= 3
                           size_t const nbz,
#endif
#if DW_NDIM >= 2
                           size_t const nby,
#endif
                           size_t const nbx, size_t const n_blocks_per_shot) {
  size_t const global_block_idx = blockIdx.x;
  size_t const batch_idx = global_block_idx / n_blocks_per_shot;
  size_t const block_idx_in_shot = global_block_idx % n_blocks_per_shot;

  size_t const bx = block_idx_in_shot % nbx;
#if DW_NDIM >= 2
  size_t const tmp = block_idx_in_shot / nbx;
  size_t const by = tmp % nby;
#if DW_NDIM >= 3
  size_t const bz = tmp / nby;
#endif
#endif

  size_t const tid = threadIdx.x;
  size_t const tx = tid % BLOCK_SIZE;
#if DW_NDIM >= 2
  size_t const tmp_t = tid / BLOCK_SIZE;
  size_t const ty = tmp_t % BLOCK_SIZE;
#if DW_NDIM >= 3
  size_t const tz = tmp_t / BLOCK_SIZE;
#endif
#endif

  size_t const x = bx * BLOCK_SIZE + tx;
#if DW_NDIM >= 2
  size_t const y = by * BLOCK_SIZE + ty;
#endif
#if DW_NDIM >= 3
  size_t const z = bz * BLOCK_SIZE + tz;
#endif

  if (x >= nx
#if DW_NDIM >= 2
      || y >= ny
#endif
#if DW_NDIM >= 3
      || z >= nz
#endif
  )
    return;

  DW_DTYPE const* const max_abs_vals = (DW_DTYPE const*)input;
  uint8_t const* const compressed_data_start =
      input + gridDim.x * sizeof(DW_DTYPE);

  DW_DTYPE max_abs = max_abs_vals[global_block_idx];
  DW_DTYPE scale = max_abs / (DW_DTYPE)(127.0);
  // max_abs / 127 becomes subnormal below this threshold. CUDA fast-math
  // flushes a float scale in that range to zero, even when compression itself
  // needed no wider arithmetic.
  DW_DTYPE const tiny_limit = sizeof(DW_DTYPE) == sizeof(float)
                                  ? (DW_DTYPE)(127.0 * FLT_MIN)
                                  : (DW_DTYPE)(127.0 * DBL_MIN);
  DW_DTYPE const huge_limit = sizeof(DW_DTYPE) == sizeof(float)
                                  ? (DW_DTYPE)(FLT_MAX / 8.0)
                                  : (DW_DTYPE)(DBL_MAX / 8.0);

  size_t const idx = batch_idx * (
#if DW_NDIM >= 3
                                     nz *
#endif
#if DW_NDIM >= 2
                                     ny *
#endif
                                     nx) +
#if DW_NDIM >= 3
                     z * ny * nx +
#endif
#if DW_NDIM >= 2
                     y * nx +
#endif
                     x;
  uint8_t val_u8 = compressed_data_start[idx];

  DW_DTYPE const level = (DW_DTYPE)(val_u8) - (DW_DTYPE)(128.0);
  output[idx] =
      dw_in_tiny_range(max_abs, tiny_limit)
          ? dw_narrow_value<DW_DTYPE>(__ddiv_rn(
                __dmul_rn((double)level, dw_wide_value(max_abs)), 127.0))
      : dw_in_huge_range(max_abs, huge_limit)
          ? dw_narrow_value<DW_DTYPE>(__dmul_rn(__ddiv_rn((double)level, 127.0),
                                                dw_wide_value(max_abs)))
          : level * scale;
}

/* CUDA wrapper functions */
extern "C" {

int SC_FUNC(compress_cuda)(void const* const input, void* const output,
                           size_t const n_batch,
#if DW_NDIM >= 3
                           size_t const nz,
#endif
#if DW_NDIM >= 2
                           size_t const ny,
#endif
                           size_t const nx, void* const stream) {

  size_t const nbx = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DW_NDIM >= 2
  size_t const nby = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
#else
  size_t const nby = 1;
#endif
#if DW_NDIM >= 3
  size_t const nbz = (nz + BLOCK_SIZE - 1) / BLOCK_SIZE;
#else
  size_t const nbz = 1;
#endif
  size_t const n_blocks_per_shot = nbx * nby * nbz;
  size_t const total_blocks = n_batch * n_blocks_per_shot;
  unsigned int const grid_blocks = dw_grid_dimension(total_blocks);
  if (grid_blocks == 0)
    return total_blocks == 0 ? 0 : (int)cudaErrorInvalidConfiguration;

  size_t const shared_mem =
      static_cast<size_t>(THREADS_PER_BLOCK) * sizeof(DW_DTYPE);

  compress_kernel<<<grid_blocks, THREADS_PER_BLOCK, shared_mem,
                    (cudaStream_t)stream>>>((DW_DTYPE const*)input,
                                            (uint8_t*)output,
#if DW_NDIM >= 3
                                            nz,
#endif
#if DW_NDIM >= 2
                                            ny,
#endif
                                            nx,
#if DW_NDIM >= 3
                                            nbz,
#endif
#if DW_NDIM >= 2
                                            nby,
#endif
                                            nbx, n_blocks_per_shot);
  CHECK_KERNEL_ERROR
  return 0;
}

int SC_FUNC(decompress_cuda)(void const* const input, void* const output,
                             size_t const n_batch,
#if DW_NDIM >= 3
                             size_t const nz,
#endif
#if DW_NDIM >= 2
                             size_t const ny,
#endif
                             size_t const nx, void* const stream) {
  size_t const nbx = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
#if DW_NDIM >= 2
  size_t const nby = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
#else
  size_t const nby = 1;
#endif
#if DW_NDIM >= 3
  size_t const nbz = (nz + BLOCK_SIZE - 1) / BLOCK_SIZE;
#else
  size_t const nbz = 1;
#endif
  size_t const n_blocks_per_shot = nbx * nby * nbz;
  size_t const total_blocks = n_batch * n_blocks_per_shot;
  unsigned int const grid_blocks = dw_grid_dimension(total_blocks);
  if (grid_blocks == 0)
    return total_blocks == 0 ? 0 : (int)cudaErrorInvalidConfiguration;

  decompress_kernel<<<grid_blocks, THREADS_PER_BLOCK, 0,
                      (cudaStream_t)stream>>>((uint8_t const*)input,
                                              (DW_DTYPE*)output,
#if DW_NDIM >= 3
                                              nz,
#endif
#if DW_NDIM >= 2
                                              ny,
#endif
                                              nx,
#if DW_NDIM >= 3
                                              nbz,
#endif
#if DW_NDIM >= 2
                                              nby,
#endif
                                              nbx, n_blocks_per_shot);
  CHECK_KERNEL_ERROR
  return 0;
}

} /* extern "C" */

#endif

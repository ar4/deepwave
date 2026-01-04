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

/* CUDA kernel for compression */
__launch_bounds__(THREADS_PER_BLOCK) __global__
    void compress_kernel(DW_DTYPE const *const input, uint8_t *const output,
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
                         size_t const nbx,
                         size_t const n_blocks_per_shot) {
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
  DW_DTYPE *const s_max_abs = (DW_DTYPE *)shared_mem;

  DW_DTYPE abs_val = (val >= 0) ? val : -val;
  s_max_abs[tid] = abs_val;
  __syncthreads();

  /* Reduction to find max abs in block */
  for (unsigned int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_max_abs[tid + s] > s_max_abs[tid]) {
        s_max_abs[tid] = s_max_abs[tid + s];
      }
    }
    __syncthreads();
  }

  DW_DTYPE max_abs = s_max_abs[0];

  /* Store max_abs */
  DW_DTYPE *const max_abs_vals = (DW_DTYPE *)output;
  if (tid == 0) {
    max_abs_vals[global_block_idx] = max_abs;
  }

  /* Quantize and store */
  uint8_t *const compressed_data_start = output + gridDim.x * sizeof(DW_DTYPE);

  if (in_bounds) {
    DW_DTYPE scale = (max_abs > 0) ? ((DW_DTYPE)(127.0) / max_abs) : (DW_DTYPE)(0);
    DW_DTYPE normalized = val * scale;
    compressed_data_start[idx] = (uint8_t)((DW_DTYPE)(128.0) + normalized + (DW_DTYPE)(0.5));
  }
}

/* CUDA kernel for decompression */
__launch_bounds__(THREADS_PER_BLOCK) __global__
    void decompress_kernel(uint8_t const *const input, DW_DTYPE *const output,
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
                           size_t const nbx,
                           size_t const n_blocks_per_shot) {
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
  ) return;

  DW_DTYPE const *const max_abs_vals = (DW_DTYPE const *)input;
  uint8_t const *const compressed_data_start = input + gridDim.x * sizeof(DW_DTYPE);

  DW_DTYPE max_abs = max_abs_vals[global_block_idx];
  DW_DTYPE scale = max_abs / (DW_DTYPE)(127.0);

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
  
  output[idx] = ((DW_DTYPE)(val_u8) - (DW_DTYPE)(128.0)) * scale;
}

/* CUDA wrapper functions */
extern "C" {

int SC_FUNC(compress_cuda)(void const *const input, void *const output,
                           size_t const n_batch,
#if DW_NDIM >= 3
                           size_t const nz,
#endif
#if DW_NDIM >= 2
                           size_t const ny,
#endif
                           size_t const nx,
                           void *const stream) {

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

  size_t const shared_mem = THREADS_PER_BLOCK * sizeof(DW_DTYPE);

  compress_kernel<<<total_blocks, THREADS_PER_BLOCK, shared_mem, (cudaStream_t)stream>>>(
      (DW_DTYPE const *)input, (uint8_t *)output, 
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

int SC_FUNC(decompress_cuda)(void const *const input, void *const output,
                             size_t const n_batch,
#if DW_NDIM >= 3
                             size_t const nz,
#endif
#if DW_NDIM >= 2
                             size_t const ny,
#endif
                             size_t const nx,
                             void *const stream) {
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

  decompress_kernel<<<total_blocks, THREADS_PER_BLOCK, 0, (cudaStream_t)stream>>>(
      (uint8_t const *)input, (DW_DTYPE *)output, 
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

#ifndef DW_NATIVE_TYPES_H
#define DW_NATIVE_TYPES_H

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

/* The ctypes ABI transports sizes and control values in int64_t. Tensor
 * offsets retain that range; CUDA device ordinals and OpenMP thread counts
 * must fit the int arguments required by those APIs. */
static inline bool dw_fits_control_int(int64_t value) {
  return value >= 0 && value <= INT_MAX;
}

/* CUDA block/thread coordinates are unsigned 32-bit values. Widen BEFORE
 * multiplying: assignment to a 64-bit result cannot undo 32-bit wraparound.
 * CUDA's block-size and grid limits bound this product well below INT64_MAX.
 * This host/device helper is also checked without allocating a large grid. */
#ifdef __CUDACC__
__host__ __device__
#endif
    static inline int64_t dw_grid_index(unsigned int block_index,
                                        unsigned int block_size,
                                        unsigned int thread_index) {
  return (int64_t)block_index * block_size + thread_index;
}

/* A zero launch dimension is rejected by CUDA. Return it for unrepresentable
 * dimensions rather than silently wrapping into a smaller, successful launch.
 * CUDA checks each device's additional hardware limits at launch time. */
static inline unsigned int dw_grid_dimension(uint64_t blocks) {
  return blocks <= UINT_MAX ? (unsigned int)blocks : 0;
}

static inline unsigned int ceil_div(int64_t numerator,
                                    unsigned int denominator) {
  if (numerator < 0 || denominator == 0) return 0;
  uint64_t const count = (uint64_t)numerator;
  return dw_grid_dimension(count / denominator + (count % denominator != 0));
}

#endif /* DW_NATIVE_TYPES_H */

#include "simple_compress.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define BLOCK_SIZE 8

#if defined(DW_NDIM) && defined(DW_DTYPE)

static inline DW_DTYPE dw_abs(DW_DTYPE val) { return (val >= 0) ? val : -val; }

/* CPU implementation */
void SC_FUNC(compress_cpu)(void const* const input, void* const output,
#if DW_NDIM >= 3
                           size_t const nz,
#endif
#if DW_NDIM >= 2
                           size_t const ny,
#endif
                           size_t const nx) {

  DW_DTYPE const* const in_ptr = (DW_DTYPE const*)input;
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
  size_t const n_blocks = nbz * nby * nbx;

  DW_DTYPE* const max_abs_vals = (DW_DTYPE*)output;
  uint8_t* const compressed = (uint8_t*)output + n_blocks * sizeof(DW_DTYPE);

#if DW_NDIM >= 3
  for (size_t bz = 0; bz < nbz; ++bz) {
    size_t const z_start = bz * BLOCK_SIZE;
    size_t const z_end =
        (z_start + BLOCK_SIZE < nz) ? z_start + BLOCK_SIZE : nz;
#else
  size_t const bz = 0;
#endif

#if DW_NDIM >= 2
    for (size_t by = 0; by < nby; ++by) {
      size_t const y_start = by * BLOCK_SIZE;
      size_t const y_end =
          (y_start + BLOCK_SIZE < ny) ? y_start + BLOCK_SIZE : ny;
#else
  size_t const by = 0;
#endif

      for (size_t bx = 0; bx < nbx; ++bx) {
        /* Find max abs in block */
        DW_DTYPE max_abs = 0;
        size_t const x_start = bx * BLOCK_SIZE;
        size_t const x_end =
            (x_start + BLOCK_SIZE < nx) ? x_start + BLOCK_SIZE : nx;

#if DW_NDIM >= 3
        for (size_t z = z_start; z < z_end; ++z) {
#endif
#if DW_NDIM >= 2
          for (size_t y = y_start; y < y_end; ++y) {
#endif
            for (size_t x = x_start; x < x_end; ++x) {
              size_t const idx =
#if DW_NDIM >= 3
                  z * ny * nx +
#endif
#if DW_NDIM >= 2
                  y * nx +
#endif
                  x;
              DW_DTYPE const val = dw_abs(in_ptr[idx]);
              if (val > max_abs) max_abs = val;
            }
#if DW_NDIM >= 2
          }
#endif
#if DW_NDIM >= 3
        }
#endif

        size_t const block_idx = bz * nby * nbx + by * nbx + bx;
        max_abs_vals[block_idx] = max_abs;

        /* A float scale overflows for tiny nonzero blocks. Use the wider
         * intermediate only in that rare range; it also preserves subnormals.
         */
        DW_DTYPE const tiny_limit = sizeof(DW_DTYPE) == sizeof(float)
                                        ? (DW_DTYPE)(127.0 / FLT_MAX)
                                        : (DW_DTYPE)(127.0 / DBL_MAX);
        bool const tiny = max_abs > 0 && max_abs <= tiny_limit;
        DW_DTYPE const scale =
            (max_abs > 0 && !tiny) ? ((DW_DTYPE)127.0 / max_abs) : 0;

        /* Quantise */
#if DW_NDIM >= 3
        for (size_t z = z_start; z < z_end; ++z) {
#endif
#if DW_NDIM >= 2
          for (size_t y = y_start; y < y_end; ++y) {
#endif
            for (size_t x = x_start; x < x_end; ++x) {
              size_t const idx =
#if DW_NDIM >= 3
                  z * ny * nx +
#endif
#if DW_NDIM >= 2
                  y * nx +
#endif
                  x;
              DW_DTYPE const normalised =
                  tiny ? (DW_DTYPE)(((long double)in_ptr[idx] /
                                     (long double)max_abs) *
                                    127.0L)
                       : in_ptr[idx] * scale;
              compressed[idx] =
                  (uint8_t)((DW_DTYPE)128.0 + normalised + (DW_DTYPE)0.5);
            }
#if DW_NDIM >= 2
          }
#endif
#if DW_NDIM >= 3
        }
#endif
      }

#if DW_NDIM >= 2
    }
#endif

#if DW_NDIM >= 3
  }
#endif
}

/* CPU implementation for decompression */
void SC_FUNC(decompress_cpu)(void const* const input, void* const output,
#if DW_NDIM >= 3
                             size_t const nz,
#endif
#if DW_NDIM >= 2
                             size_t const ny,
#endif
                             size_t const nx) {
  DW_DTYPE* const out_ptr = (DW_DTYPE*)output;
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
  size_t const n_blocks = nbz * nby * nbx;

  DW_DTYPE const* const max_abs_vals = (const DW_DTYPE*)input;
  uint8_t const* const compressed =
      (const uint8_t*)input + n_blocks * sizeof(DW_DTYPE);

#if DW_NDIM >= 3
  for (size_t bz = 0; bz < nbz; ++bz) {
    size_t const z_start = bz * BLOCK_SIZE;
    size_t const z_end =
        (z_start + BLOCK_SIZE < nz) ? z_start + BLOCK_SIZE : nz;
#else
  size_t const bz = 0;
#endif

#if DW_NDIM >= 2
    for (size_t by = 0; by < nby; ++by) {
      size_t const y_start = by * BLOCK_SIZE;
      size_t const y_end =
          (y_start + BLOCK_SIZE < ny) ? y_start + BLOCK_SIZE : ny;
#else
  size_t const by = 0;
#endif

      for (size_t bx = 0; bx < nbx; ++bx) {
        size_t const block_idx = bz * nby * nbx + by * nbx + bx;
        DW_DTYPE const max_abs = max_abs_vals[block_idx];
        DW_DTYPE const scale = max_abs / (DW_DTYPE)127.0;
        /* max_abs / 127 underflows to a subnormal below this limit;
         * fast-math may flush that scale to zero. This threshold differs
         * from the compression reciprocal-overflow threshold. */
        DW_DTYPE const tiny_limit = sizeof(DW_DTYPE) == sizeof(float)
                                        ? (DW_DTYPE)(127.0 * FLT_MIN)
                                        : (DW_DTYPE)(127.0 * DBL_MIN);
        DW_DTYPE const huge_limit = sizeof(DW_DTYPE) == sizeof(float)
                                        ? (DW_DTYPE)(FLT_MAX / 8.0)
                                        : (DW_DTYPE)(DBL_MAX / 8.0);

        size_t const x_start = bx * BLOCK_SIZE;
        size_t const x_end =
            (x_start + BLOCK_SIZE < nx) ? x_start + BLOCK_SIZE : nx;

#if DW_NDIM >= 3
        for (size_t z = z_start; z < z_end; ++z) {
#endif
#if DW_NDIM >= 2
          for (size_t y = y_start; y < y_end; ++y) {
#endif
            for (size_t x = x_start; x < x_end; ++x) {
              size_t const idx =
#if DW_NDIM >= 3
                  z * ny * nx +
#endif
#if DW_NDIM >= 2
                  y * nx +
#endif
                  x;
              DW_DTYPE const level =
                  (DW_DTYPE)compressed[idx] - (DW_DTYPE)128.0;
              out_ptr[idx] =
                  max_abs > 0 && max_abs <= tiny_limit
                      ? (DW_DTYPE)(((long double)level * (long double)max_abs) /
                                   127.0L)
                  /* A rounded scale times 127 can overflow at the dtype
                   * maximum. Divide the bounded level first in this range;
                   * this also works when long double has double's range. */
                  : max_abs > huge_limit
                      ? (DW_DTYPE)(((long double)level / 127.0L) *
                                   (long double)max_abs)
                      : level * scale;
            }
#if DW_NDIM >= 2
          }
#endif
#if DW_NDIM >= 3
        }
#endif
      }
#if DW_NDIM >= 2
    }
#endif
#if DW_NDIM >= 3
  }
#endif
}

#endif

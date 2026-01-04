#include "simple_compress.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define BLOCK_SIZE 8

#if defined(DW_NDIM) && defined(DW_DTYPE)

static inline DW_DTYPE dw_abs(DW_DTYPE val) { return (val >= 0) ? val : -val; }

/* CPU implementation */
void SC_FUNC(compress_cpu)(void const *const input, void *const output,
#if DW_NDIM >= 3
                           size_t const nz,
#endif
#if DW_NDIM >= 2
                           size_t const ny,
#endif
                           size_t const nx) {

  DW_DTYPE const *const in_ptr = (DW_DTYPE const *)input;
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

  DW_DTYPE *const max_abs_vals = (DW_DTYPE *)output;
  uint8_t *const compressed = (uint8_t *)output + n_blocks * sizeof(DW_DTYPE);

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

        DW_DTYPE const scale =
            (max_abs > 0) ? ((DW_DTYPE)127.0 / max_abs) : 0;

        /* Quantize */
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
              compressed[idx] =
                  (uint8_t)((DW_DTYPE)128.0 + in_ptr[idx] * scale + (DW_DTYPE)0.5);
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
void SC_FUNC(decompress_cpu)(void const *const input, void *const output,
#if DW_NDIM >= 3
                             size_t const nz,
#endif
#if DW_NDIM >= 2
                             size_t const ny,
#endif
                             size_t const nx) {
  DW_DTYPE *const out_ptr = (DW_DTYPE *)output;
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

  DW_DTYPE const *const max_abs_vals = (const DW_DTYPE *)input;
  uint8_t const *const compressed =
      (const uint8_t *)input + n_blocks * sizeof(DW_DTYPE);

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
              out_ptr[idx] = ((DW_DTYPE)compressed[idx] - (DW_DTYPE)128.0) * scale;
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

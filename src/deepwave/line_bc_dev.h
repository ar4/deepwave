#ifndef LINE_BC_DEV_H
#define LINE_BC_DEV_H

/* CUDA device-function/kernel counterparts of line_bc.h's fill_p_bc_line/
 * fill_v_bc_line (see that header for the full explanation of the boundary
 * fill itself). Byte-identical (modulo signature line-wrap) across every
 * propagator's .cu file on this staggered layout -- acoustic.cu,
 * acoustic_vti.cu, viscoacoustic.cu, acoustic_born.cu. `inline` (not
 * `static`) is deliberate: each of these device functions/kernels is
 * name-identical, byte-identical machine code across every including .cu
 * translation unit, and each propagator's CUDA sources build into their own
 * separate object library (see CMakeLists.txt's per-propagator CUDA
 * object-library macros), so there is no link-time collision to avoid by
 * making them internal-linkage. */

__device__ inline void fill_p_bc_line_dev(DW_DTYPE* __restrict const p,
                                          int64_t const base, int64_t const nax,
                                          int64_t const stride,
                                          int64_t const bc_lo,
                                          int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_lo == 1) p[base + P * stride] = 0;
  if (bc_hi == 1) p[base + (nax - P) * stride] = 0;
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_lo == 3) {
        src = m - 1 + g;
      } else if (bc_lo == 2) {
        src = 2 * P - 1 - g;
      } else {
        src = 2 * P - g;
        sign = -1;
      }
      p[base + g * stride] = sign * p[base + src * stride];
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j;
      } else if (bc_hi == 2) {
        src = P + m - 1 - j;
      } else {
        src = P + m - 2 - j;
        sign = -1;
      }
      p[base + xh * stride] = sign * p[base + src * stride];
    }
  }
}

__device__ inline void fill_v_bc_line_dev(DW_DTYPE* __restrict const v,
                                          int64_t const base, int64_t const nax,
                                          int64_t const stride,
                                          int64_t const bc_lo,
                                          int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi == 2) {
    v[base + (nax - P) * stride] = 0;
  } else if (bc_hi == 3) {
    v[base + (nax - P) * stride] = v[base + P * stride];
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      int64_t src;
      DW_DTYPE sign = 1;
      int zero = 0;
      if (bc_lo == 3) {
        src = m - 1 + g;
      } else if (bc_lo == 1) {
        src = 2 * P - 1 - g;
      } else {
        if (g == P - 1) {
          zero = 1;
          src = 0;
        } else {
          src = 2 * P - 2 - g;
          sign = -1;
        }
      }
      v[base + g * stride] = zero ? (DW_DTYPE)0 : sign * v[base + src * stride];
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j;
      } else if (bc_hi == 1) {
        src = P + m - 3 - j;
      } else {
        src = P + m - 2 - j;
        sign = -1;
      }
      v[base + xh * stride] = sign * v[base + src * stride];
    }
  }
}

/* One thread per line along the chosen axis (base = (L/stride)*(nax*stride) +
 * (L%stride) for a contiguous array). ``kind`` 0 = pressure, 1 = velocity. */
__global__ void fill_bc_kernel(DW_DTYPE* __restrict const wf,
                               int64_t const n_lines, int64_t const nax,
                               int64_t const stride, int64_t const bc_lo,
                               int64_t const bc_hi, int const kind) {
  int64_t const L = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  if (kind == 0)
    fill_p_bc_line_dev(wf, base, nax, stride, bc_lo, bc_hi);
  else
    fill_v_bc_line_dev(wf, base, nax, stride, bc_lo, bc_hi);
}

#endif /* LINE_BC_DEV_H */

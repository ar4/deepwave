#ifndef ADJ_FOLD_BC_KERNEL_DEV_H
#define ADJ_FOLD_BC_KERNEL_DEV_H

/* CUDA kernel wrappers around FUNC(adj_fold_p_line)/FUNC(adj_fold_v_line)
 * and their _fr counterparts (adj_line_bc.h) -- one thread per line,
 * dispatching on ``kind`` (0 = pressure, 1 = velocity). Byte-identical
 * across acoustic.cu, acoustic_born.cu, viscoacoustic.cu. */

__global__ void adj_fold_bc_kernel(DW_DTYPE* __restrict const wf,
                                   int64_t const n_lines, int64_t const nax,
                                   int64_t const stride, int const kind) {
  int64_t const L = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  if (kind == 0)
    FUNC(adj_fold_p_line)(wf, base, nax, stride);
  else
    FUNC(adj_fold_v_line)(wf, base, nax, stride);
}

__global__ void adj_fold_fr_bc_kernel(DW_DTYPE* __restrict const wf,
                                      int64_t const n_lines, int64_t const nax,
                                      int64_t const stride, int64_t const bc_lo,
                                      int64_t const bc_hi, int const kind) {
  int64_t const L = dw_grid_index(blockIdx.x, blockDim.x, threadIdx.x);
  if (L >= n_lines) return;
  int64_t const base = (L / stride) * (nax * stride) + (L % stride);
  if (kind == 0)
    FUNC(adj_fold_p_line_fr)(wf, base, nax, stride, bc_lo, bc_hi);
  else
    FUNC(adj_fold_v_line_fr)(wf, base, nax, stride, bc_lo, bc_hi);
}

#endif /* ADJ_FOLD_BC_KERNEL_DEV_H */

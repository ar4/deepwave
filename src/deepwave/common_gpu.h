#ifndef DW_COMMON_GPU_H
#define DW_COMMON_GPU_H

#ifdef DW_DEBUG
#define CHECK_KERNEL_ERROR              \
  {                                     \
    gpuErrchk(cudaPeekAtLastError());   \
    gpuErrchk(cudaDeviceSynchronize()); \
  }
#else
#define CHECK_KERNEL_ERROR gpuErrchk(cudaPeekAtLastError());
#endif /* DW_DEBUG */

#endif /* DW_COMMON_GPU_H */

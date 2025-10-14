#ifndef DW_COMMON_GPU_H
#define DW_COMMON_GPU_H

#ifdef DEBUG
#define CHECK_KERNEL_ERROR              \
  {                                     \
    gpuErrchk(cudaPeekAtLastError());   \
    gpuErrchk(cudaDeviceSynchronize()); \
  }
#else
#define CHECK_KERNEL_ERROR gpuErrchk(cudaPeekAtLastError());
#endif /* DEBUG */

#endif /* DW_COMMON_GPU_H */

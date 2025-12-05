#ifndef DW_COMMON_GPU_H
#define DW_COMMON_GPU_H

#include <stdio.h>

#define gpuErrchk(ans)                              \
  {                                                 \
    int err = gpuAssert((ans), __FILE__, __LINE__); \
    if (err != 0) return err;                       \
  }

inline int gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    return (int)code;
  }
  return 0;
}

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

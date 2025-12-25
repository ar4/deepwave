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

inline unsigned int ceil_div(unsigned int numerator, unsigned int denominator) {
  return (numerator + denominator - 1) / denominator;
}

struct ScopedStream {
  cudaStream_t stream;
  ScopedStream() : stream(nullptr) {}
  ~ScopedStream() {
    if (stream) cudaStreamDestroy(stream);
  }
  operator cudaStream_t() const { return stream; }
  cudaStream_t *operator&() { return &stream; }
};

struct ScopedEvent {
  cudaEvent_t event;
  ScopedEvent() : event(nullptr) {}
  ~ScopedEvent() {
    if (event) cudaEventDestroy(event);
  }
  operator cudaEvent_t() const { return event; }
  cudaEvent_t *operator&() { return &event; }
};

struct ScopedFile {
  FILE *fp;
  ScopedFile() : fp(nullptr) {}
  ~ScopedFile() {
    if (fp) fclose(fp);
  }
  operator FILE *() const { return fp; }
  void open(const char *filename, const char *mode) {
    if (fp) fclose(fp);
    fp = fopen(filename, mode);
  }
};

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

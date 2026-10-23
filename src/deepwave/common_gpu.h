#ifndef DW_COMMON_GPU_H
#define DW_COMMON_GPU_H

#include <stdio.h>

#include <mutex>

#include "storage_utils.h"

/* Every CUDA propagator writes translation-unit __constant__ configuration
 * before launching kernels. The lock is shared across all variants in the
 * library, and keyed by CUDA device so independent GPUs remain independent.
 * It must remain held until all compute and storage-stream work has finished:
 * locking only cudaMemcpyToSymbol would still let a later call overwrite the
 * configuration while an earlier kernel is running. */
std::mutex &dw_cuda_config_mutex(int device) noexcept;

struct ScopedCudaDevice {
  int prior_device;
  int init_error;
  bool have_prior;
  explicit ScopedCudaDevice(int device) noexcept
      : prior_device(0), init_error(0), have_prior(false) {
    cudaError_t error = cudaGetDevice(&prior_device);
    if (error != cudaSuccess) {
      init_error = static_cast<int>(error);
      return;
    }
    have_prior = true;
    error = cudaSetDevice(device);
    if (error != cudaSuccess) init_error = static_cast<int>(error);
  }
  ~ScopedCudaDevice() {
    if (!have_prior) return;
    cudaError_t const error = cudaSetDevice(prior_device);
    if (error != cudaSuccess)
      fprintf(stderr, "GPUassert: %s restoring CUDA device\n",
              cudaGetErrorString(error));
  }
  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;
};

struct ScopedCudaConfig {
  std::unique_lock<std::mutex> lock;
  int init_error;
  bool finished;
  explicit ScopedCudaConfig(int device) noexcept
      : init_error(0), finished(false) {
    try {
      lock = std::unique_lock<std::mutex>(dw_cuda_config_mutex(device));
    } catch (...) {
      init_error = static_cast<int>(cudaErrorUnknown);
    }
  }
  int finish() {
    cudaError_t const error = cudaDeviceSynchronize();
    finished = true;
    if (error != cudaSuccess)
      fprintf(stderr, "GPUassert: %s during CUDA configuration release\n",
              cudaGetErrorString(error));
    return static_cast<int>(error);
  }
  ~ScopedCudaConfig() {
    if (lock.owns_lock() && !finished) (void)finish();
  }
  ScopedCudaConfig(const ScopedCudaConfig &) = delete;
  ScopedCudaConfig &operator=(const ScopedCudaConfig &) = delete;
};

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
  // Launch dimensions may approach UINT_MAX; adding before division wraps.
  return numerator / denominator + (numerator % denominator != 0);
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
  ScopedFile *next;
  ScopedFile() : fp(nullptr), next(nullptr) {}
  ~ScopedFile() { (void)close(); }
  operator FILE *() const { return fp; }
  int close() { return dw_close_file(&fp); }
  bool open(const char *filename, const char *mode) {
    if (close() != 0) return false;
    fp = dw_open_file_utf8(filename, mode);
    return fp != nullptr;
  }
  ScopedFile(const ScopedFile &) = delete;
  ScopedFile &operator=(const ScopedFile &) = delete;
};

struct ScopedFileGroup {
  ScopedFile *head;
  ScopedFileGroup() : head(nullptr) {}
  void add(ScopedFile &file) {
    file.next = head;
    head = &file;
  }
  int close_all() {
    int status = 0;
    for (ScopedFile *file = head; file != nullptr; file = file->next)
      if (file->close() != 0) status = 1;
    return status;
  }
  ScopedFileGroup(const ScopedFileGroup &) = delete;
  ScopedFileGroup &operator=(const ScopedFileGroup &) = delete;
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

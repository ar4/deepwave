/* CUDA driver-API shim (declarations).
 *
 * The generated GPU host driver reaches CUDA through libcuda.so.1, resolved
 * with dlopen at first use -- no libcudart, no libtorch, no CUDA headers.
 * Device code is a fatbin (several SM architectures plus PTX) loaded with
 * cuModuleLoadData; modules and function handles are cached per device
 * ordinal.  The caller supplies the stream (torch's current stream) as an
 * opaque pointer.  Definitions live in dw_cuda.c, compiled once.
 */
#ifndef DW_CUDA_H
#define DW_CUDA_H

#include <stdint.h>
#include <stddef.h>

typedef int dw_curesult;
typedef void *dw_cumodule;
typedef void *dw_cufunction;
typedef void *dw_custream;
typedef void *dw_cucontext;

extern dw_curesult (*dw_cuLaunchKernel)(dw_cufunction, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, unsigned,
                                        unsigned, dw_custream, void **, void **);

int dw_cuda_init(const char *fatbin_path);
dw_cufunction dw_cuda_fn(const char *name);
int dw_cuda_launch(const char *name, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz, dw_custream stream,
                   void *params);
int dw_cuda_memcpy_dtod(void *dst, const void *src, size_t n, dw_custream s);

static inline unsigned dw_ceil_div(int64_t n, int64_t d) {
  return (unsigned)(n <= 0 ? 0 : (n + d - 1) / d);
}

#endif /* DW_CUDA_H */

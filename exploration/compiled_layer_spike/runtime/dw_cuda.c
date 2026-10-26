/* CUDA driver-API shim.
 *
 * The generated GPU host driver reaches CUDA through libcuda.so.1, resolved
 * with dlopen at first use -- no libcudart, no libtorch, no CUDA headers.  The
 * device code is a fatbin (several SM architectures plus PTX) loaded with
 * cuModuleLoadData.  Modules and function handles are cached per device
 * ordinal.  The caller supplies the stream (torch's current stream) as an
 * opaque pointer.
 */
#include "dw_cuda.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define DW_CU_MAX_DEV 8
#define DW_CU_MAX_FN 256

static void *dw_cu_lib = NULL;
static dw_curesult (*dw_cuInit)(unsigned);
static dw_curesult (*dw_cuModuleLoadData)(dw_cumodule *, const void *);
static dw_curesult (*dw_cuModuleGetFunction)(dw_cufunction *, dw_cumodule,
                                             const char *);
dw_curesult (*dw_cuLaunchKernel)(dw_cufunction, unsigned, unsigned,
                                        unsigned, unsigned, unsigned, unsigned,
                                        unsigned, dw_custream, void **, void **);
static dw_curesult (*dw_cuCtxGetCurrent)(dw_cucontext *);
static dw_curesult (*dw_cuCtxGetDevice)(int *);
static dw_curesult (*dw_cuDevicePrimaryCtxRetain)(dw_cucontext *, int);
static dw_curesult (*dw_cuCtxSetCurrent)(dw_cucontext);

#define DW_CU_MAX_IMG 16
static void *dw_cu_image[DW_CU_MAX_IMG];
static int dw_cu_nimg = 0;
static dw_cumodule dw_cu_mod[DW_CU_MAX_DEV][DW_CU_MAX_IMG];
static dw_curesult (*dw_cuMemcpyDtoDAsync)(void *, void *, size_t, dw_custream);
static struct {
  int dev;
  const char *name;
  dw_cufunction fn;
} dw_cu_cache[DW_CU_MAX_FN];
static int dw_cu_ncache = 0;

#define DW_SYM(n)                                    \
  do {                                               \
    *(void **)(&dw_##n) = dlsym(dw_cu_lib, #n);      \
    if (!dw_##n) {                                   \
      *(void **)(&dw_##n) = dlsym(dw_cu_lib, #n "_v2"); \
    }                                                \
    if (!dw_##n) return -1;                          \
  } while (0)

/* Load libcuda and the fatbin image.  Called once from Python. */
static int dw_cuda_init_impl(const char *fatbin_path) {
  if (!dw_cu_lib) {
    dw_cu_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!dw_cu_lib) dw_cu_lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dw_cu_lib) return -1;
    DW_SYM(cuInit);
    DW_SYM(cuModuleLoadData);
    DW_SYM(cuModuleGetFunction);
    DW_SYM(cuLaunchKernel);
    DW_SYM(cuCtxGetCurrent);
    DW_SYM(cuCtxGetDevice);
    DW_SYM(cuDevicePrimaryCtxRetain);
    DW_SYM(cuCtxSetCurrent);
    DW_SYM(cuMemcpyDtoDAsync);
    dw_cuInit(0);
  }
  if (dw_cu_nimg >= DW_CU_MAX_IMG) return -4;
  FILE *f = fopen(fatbin_path, "rb");
  if (!f) return -2;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *img = malloc((size_t)n);
  if (fread(img, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    return -3;
  }
  fclose(f);
  dw_cu_image[dw_cu_nimg++] = img;
  return 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int dw_cuda_init(const char *fatbin_path) {
  return dw_cuda_init_impl(fatbin_path);
}

dw_cufunction dw_cuda_fn(const char *name) {
  dw_cucontext ctx = NULL;
  int dev = 0;
  dw_cuCtxGetCurrent(&ctx);
  if (!ctx) {
    /* torch normally leaves the primary context current on this thread; if
     * not, retain and bind device 0's. */
    if (dw_cuDevicePrimaryCtxRetain(&ctx, 0) != 0) return NULL;
    dw_cuCtxSetCurrent(ctx);
  }
  dw_cuCtxGetDevice(&dev);
  if (dev < 0 || dev >= DW_CU_MAX_DEV) return NULL;
  for (int i = 0; i < dw_cu_ncache; ++i) {
    if (dw_cu_cache[i].dev == dev && dw_cu_cache[i].name == name)
      return dw_cu_cache[i].fn;
  }
  dw_cufunction fn = NULL;
  for (int im = 0; im < dw_cu_nimg; ++im) {
    if (!dw_cu_mod[dev][im]) {
      if (dw_cuModuleLoadData(&dw_cu_mod[dev][im], dw_cu_image[im]) != 0) continue;
    }
    if (dw_cuModuleGetFunction(&fn, dw_cu_mod[dev][im], name) == 0 && fn) break;
    fn = NULL;
  }
  if (!fn) return NULL;
  if (dw_cu_ncache < DW_CU_MAX_FN) {
    dw_cu_cache[dw_cu_ncache].dev = dev;
    dw_cu_cache[dw_cu_ncache].name = name;
    dw_cu_cache[dw_cu_ncache].fn = fn;
    dw_cu_ncache++;
  }
  return fn;
}

int dw_cuda_launch(const char *name, unsigned gx, unsigned gy,
                          unsigned gz, unsigned bx, unsigned by, unsigned bz,
                          dw_custream stream, void *params) {
  if (gx == 0 || gy == 0 || gz == 0) return 0;
  dw_cufunction fn = dw_cuda_fn(name);
  if (!fn) return -1;
  void *args[1];
  args[0] = params;
  return dw_cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, 0, stream, args, NULL);
}

int dw_cuda_memcpy_dtod(void *dst, const void *src, size_t n,
                               dw_custream s) {
  return dw_cuMemcpyDtoDAsync(dst, (void *)src, n, s);
}



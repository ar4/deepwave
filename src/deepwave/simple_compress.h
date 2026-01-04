#ifndef SIMPLE_COMPRESS_H
#define SIMPLE_COMPRESS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(DW_NDIM) && defined(DW_DTYPE)
#define SC_CAT_I(name, ndim, dtype) simple_compress_##name##_##ndim##d_##dtype
#define SC_CAT(name, ndim, dtype) SC_CAT_I(name, ndim, dtype)
#define SC_FUNC(name) SC_CAT(name, DW_NDIM, DW_DTYPE)

void SC_FUNC(compress_cpu)(void const *input, void *output,
#if DW_NDIM >= 3
                           size_t nz,
#endif
#if DW_NDIM >= 2
                           size_t ny,
#endif
                           size_t nx);
int SC_FUNC(compress_cuda)(void const *input, void *output, size_t n_batch,
#if DW_NDIM >= 3
                           size_t nz,
#endif
#if DW_NDIM >= 2
                           size_t ny,
#endif
                           size_t nx, void *stream);

void SC_FUNC(decompress_cpu)(void const *input, void *output,
#if DW_NDIM >= 3
                             size_t nz,
#endif
#if DW_NDIM >= 2
                             size_t ny,
#endif
                             size_t nx);
int SC_FUNC(decompress_cuda)(void const *input, void *output, size_t n_batch,
#if DW_NDIM >= 3
                             size_t nz,
#endif
#if DW_NDIM >= 2
                             size_t ny,
#endif
                             size_t nx, void *stream);
#endif /* DW_NDIM && DW_DTYPE */

#ifdef __cplusplus
}
#endif

#endif /* SIMPLE_COMPRESS_H */

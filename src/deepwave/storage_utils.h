#ifndef STORAGE_UTILS_H
#define STORAGE_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define STORAGE_DEVICE 0
#define STORAGE_CPU 1
#define STORAGE_DISK 2
#define STORAGE_NONE 3

#ifdef __cplusplus
extern "C" {
#endif

#if defined(DW_NDIM) && defined(DW_DTYPE)
#define SS_CAT_I(name, ndim, dtype) storage_##name##_##ndim##d_##dtype
#define SS_CAT(name, ndim, dtype) SS_CAT_I(name, ndim, dtype)
#define STORAGE_FUNC(name) SS_CAT(name, DW_NDIM, DW_DTYPE)

void STORAGE_FUNC(save_snapshot_cpu)(void const* store_1, void* store_2,
                                     FILE* fp, int64_t storage_mode,
                                     bool use_compression, int64_t step_idx,
                                     size_t step_bytes_uncomp,
                                     size_t step_bytes_comp,
#if DW_NDIM >= 3
                                     size_t nz,
#endif
#if DW_NDIM >= 2
                                     size_t ny,
#endif
                                     size_t nx);

int STORAGE_FUNC(save_snapshot_gpu)(void const* store_1, void* store_2,
                                    void* store_3, FILE* fp,
                                    int64_t storage_mode, bool use_compression,
                                    int64_t step_idx, size_t shot_bytes_uncomp,
                                    size_t shot_bytes_comp, size_t n_shots,
#if DW_NDIM >= 3
                                    size_t nz,
#endif
#if DW_NDIM >= 2
                                    size_t ny,
#endif
                                    size_t nx, void* stream);

void STORAGE_FUNC(load_snapshot_cpu)(void* store_1, void* store_2, FILE* fp,
                                     int64_t storage_mode, bool use_compression,
                                     int64_t step_idx, size_t step_bytes_uncomp,
                                     size_t step_bytes_comp,
#if DW_NDIM >= 3
                                     size_t nz,
#endif
#if DW_NDIM >= 2
                                     size_t ny,
#endif
                                     size_t nx);

int STORAGE_FUNC(load_snapshot_gpu)(void* store_1, void* store_2, void* store_3,
                                    FILE* fp, int64_t storage_mode,
                                    bool use_compression, int step_idx,
                                    size_t shot_bytes_uncomp,
                                    size_t shot_bytes_comp, size_t n_shots,
#if DW_NDIM >= 3
                                    size_t nz,
#endif
#if DW_NDIM >= 2
                                    size_t ny,
#endif
                                    size_t nx, void* stream);

#endif /* DW_NDIM && DW_DTYPE */

#ifdef __cplusplus
}
#endif

#endif

#ifndef STORAGE_UTILS_H
#define STORAGE_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#include <stdlib.h>
#include <windows.h>
#endif
#ifndef _WIN32
#include <sys/types.h>
#endif

#define STORAGE_DEVICE 0
#define STORAGE_CPU 1
#define STORAGE_DISK 2
#define STORAGE_NONE 3

/* Filenames crossing the Python/native boundary are UTF-8. The Windows C
 * runtime's narrow fopen uses the process code page instead. */
static inline FILE* dw_open_file_utf8(const char* filename, const char* mode) {
#ifdef _WIN32
  if (filename == NULL || mode == NULL) return NULL;
  int const path_len =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, NULL, 0);
  int const mode_len =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, mode, -1, NULL, 0);
  if (path_len == 0 || mode_len == 0) return NULL;
  wchar_t* const path = (wchar_t*)malloc((size_t)path_len * sizeof(wchar_t));
  wchar_t* const wide_mode =
      (wchar_t*)malloc((size_t)mode_len * sizeof(wchar_t));
  if (path == NULL || wide_mode == NULL) {
    free(path);
    free(wide_mode);
    return NULL;
  }
  FILE* fp = NULL;
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, filename, -1, path,
                          path_len) != 0 &&
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, mode, -1, wide_mode,
                          mode_len) != 0)
    fp = _wfopen(path, wide_mode);
  free(path);
  free(wide_mode);
  return fp;
#else
  return fopen(filename, mode);
#endif
}

/* Clear ownership before fclose: a failed close still consumes the handle. */
static inline int dw_close_file(FILE** handle) {
  if (handle == NULL || *handle == NULL) return 0;
  FILE* const fp = *handle;
  *handle = NULL;
  return fclose(fp) != 0;
}

/* Seek to a snapshot without narrowing the byte offset to Windows' 32-bit
 * long. Reject overflow before either multiplication or conversion. */
static inline int dw_seek_snapshot(FILE* fp, int64_t step_idx,
                                   size_t step_bytes) {
  if (fp == NULL || step_idx < 0 || step_bytes > INT64_MAX ||
      (step_bytes != 0 && step_idx > INT64_MAX / (int64_t)step_bytes))
    return 1;
  int64_t const offset = step_idx * (int64_t)step_bytes;
#ifdef _WIN32
  return _fseeki64(fp, offset, SEEK_SET) != 0;
#else
  if (sizeof(off_t) < sizeof(int64_t)) return 1;
  return fseeko(fp, (off_t)offset, SEEK_SET) != 0;
#endif
}

static inline int dw_checked_shot_bytes(size_t bytes_per_shot, size_t n_shots,
                                        size_t* total) {
  if (n_shots != 0 && bytes_per_shot > SIZE_MAX / n_shots) return 1;
  *total = bytes_per_shot * n_shots;
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

#if defined(DW_NDIM) && defined(DW_DTYPE)
#define SS_CAT_I(name, ndim, dtype) storage_##name##_##ndim##d_##dtype
#define SS_CAT(name, ndim, dtype) SS_CAT_I(name, ndim, dtype)
#define STORAGE_FUNC(name) SS_CAT(name, DW_NDIM, DW_DTYPE)

/* Shared per-gradient storage-file/pointer setup, used identically (byte-
 * for-byte, md5-confirmed) across every propagator with checkpointed
 * gradient storage. Kept as macros, not functions, because each declares
 * new local variables (``name##_store_1_t`` etc.) that the calling code
 * uses afterward -- a function could not do that in the caller's scope.
 * Callers must have ``storage_mode``, ``storage_compression``, ``t``,
 * ``step_ratio``, ``n_shots``, ``n_grid_points``, ``si``, ``shot``,
 * ``shot_bytes_comp``, ``shot_bytes_uncomp`` in scope (as every propagator's
 * forward/backward loop body already does), and, for
 * ``SETUP_STORE_LOAD``, ``DIM_ARGS`` defined per DW_NDIM (as every
 * propagator's own file already does before this macro's first use).
 *
 * ``OPEN_FILE_WRITE``/``OPEN_FILE_READ``/``SETUP_STORE_LOAD`` all report
 * failure via ``goto storage_cleanup`` rather than a bare ``return 1``, so
 * that a later failure in a sequence of these calls -- or a failed
 * fwrite/fread reached from deep inside the time-stepping loop -- still
 * closes every ``FILE*`` this call's caller has already opened, instead of
 * leaking them past an early function exit. Callers must therefore, before
 * using any of these three macros: (1) pre-declare every ``FILE* fp_name``
 * this call sequence will use, each initialised to ``NULL`` (the macros no
 * longer declare it themselves, so that a ``goto`` from an earlier call in
 * the same sequence can never jump over a later one's declaration -- doing
 * so would leave it holding an indeterminate value); (2) declare
 * ``int storage_err = 0;``; and (3) place a ``storage_cleanup:`` label
 * immediately before the block that closes every one of those handles
 * (mirroring the existing ``CLOSE_FILE`` calls), followed by
 * ``if (storage_err) return 1;``. */
#define OPEN_FILE_WRITE(name, grad_cond)                             \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {                 \
    fp_##name = dw_open_file_utf8(name##_filenames_ptr[shot], "ab"); \
    if (fp_##name == NULL) {                                         \
      storage_err = 1;                                               \
      goto storage_cleanup;                                          \
    }                                                                \
  }

#define OPEN_FILE_READ(name, grad_cond)                              \
  if (storage_mode == STORAGE_DISK && (grad_cond)) {                 \
    fp_##name = dw_open_file_utf8(name##_filenames_ptr[shot], "rb"); \
    if (fp_##name == NULL) {                                         \
      storage_err = 1;                                               \
      goto storage_cleanup;                                          \
    }                                                                \
  }

#define SETUP_STORE_SAVE(name, grad_cond)                                     \
  if ((grad_cond) && ((t % step_ratio) == 0) && name##_store_1 == NULL) {     \
    storage_err = 1;                                                          \
    goto storage_cleanup;                                                     \
  }                                                                           \
  DW_DTYPE* __restrict const name##_store_1_t =                               \
      name##_store_1 != NULL                                                  \
          ? (name##_store_1 + si +                                            \
             ((storage_mode == STORAGE_DEVICE && !storage_compression)        \
                  ? (t / step_ratio) * n_shots * n_grid_points                \
                  : 0))                                                       \
          : NULL;                                                             \
  void* __restrict const name##_store_2_t =                                   \
      name##_store_2 != NULL                                                  \
          ? ((uint8_t*)name##_store_2 +                                       \
             (shot + ((storage_mode == STORAGE_DEVICE && storage_compression) \
                          ? t / step_ratio * n_shots                          \
                          : 0)) *                                             \
                 (int64_t)shot_bytes_comp)                                    \
          : NULL;

#define SETUP_STORE_LOAD(name, grad_cond)                                     \
  if ((grad_cond) && ((t % step_ratio) == 0) && name##_store_1 == NULL) {     \
    storage_err = 1;                                                          \
    goto storage_cleanup;                                                     \
  }                                                                           \
  DW_DTYPE* __restrict const name##_store_1_t =                               \
      name##_store_1 != NULL                                                  \
          ? (name##_store_1 + si +                                            \
             ((storage_mode == STORAGE_DEVICE && !storage_compression)        \
                  ? (t / step_ratio) * n_shots * n_grid_points                \
                  : 0))                                                       \
          : NULL;                                                             \
  void* __restrict const name##_store_2_t =                                   \
      name##_store_2 != NULL                                                  \
          ? ((uint8_t*)name##_store_2 +                                       \
             (shot + ((storage_mode == STORAGE_DEVICE && storage_compression) \
                          ? t / step_ratio * n_shots                          \
                          : 0)) *                                             \
                 (int64_t)shot_bytes_comp)                                    \
          : NULL;                                                             \
  if ((grad_cond) && ((t % step_ratio) == 0)) {                               \
    int64_t const step_idx = t / step_ratio;                                  \
    if (STORAGE_FUNC(load_snapshot_cpu)(                                      \
            (void*)name##_store_1_t, name##_store_2_t, fp_##name,             \
            storage_mode, storage_compression, step_idx, shot_bytes_uncomp,   \
            shot_bytes_comp, DIM_ARGS) != 0) {                                \
      storage_err = 1;                                                        \
      goto storage_cleanup;                                                   \
    }                                                                         \
  }

int STORAGE_FUNC(save_snapshot_cpu)(void const* store_1, void* store_2,
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

int STORAGE_FUNC(load_snapshot_cpu)(void* store_1, void* store_2, FILE* fp,
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
                                    bool use_compression, int64_t step_idx,
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

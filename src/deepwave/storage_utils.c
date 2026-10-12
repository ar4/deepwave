#include "storage_utils.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "simple_compress.h"

#if defined(DW_NDIM) && defined(DW_DTYPE)

int STORAGE_FUNC(save_snapshot_cpu)(void const* const store_1,
                                    void* const store_2, FILE* const fp,
                                    int64_t const storage_mode,
                                    bool const use_compression,
                                    int64_t const step_idx,
                                    size_t const step_bytes_uncomp,
                                    size_t const step_bytes_comp,
#if DW_NDIM >= 3
                                    size_t nz,
#endif
#if DW_NDIM >= 2
                                    size_t ny,
#endif
                                    size_t nx) {
  void const* source = store_1;
  size_t size_to_write = step_bytes_uncomp;

  if (storage_mode == STORAGE_NONE) return 0;

  if (use_compression) {
    SC_FUNC(compress_cpu)(store_1, store_2,
#if DW_NDIM >= 3
                          nz,
#endif
#if DW_NDIM >= 2
                          ny,
#endif
                          nx);
    size_to_write = step_bytes_comp;
    source = (void const*)store_2;
  }

  if (storage_mode == STORAGE_DISK) {
    // Append to file
    if (dw_seek_snapshot(fp, step_idx, size_to_write) != 0) return 1;
    if (fwrite(source, 1, size_to_write, fp) != size_to_write) return 1;
  }
  return 0;
}

int STORAGE_FUNC(load_snapshot_cpu)(void* const store_1, void* const store_2,
                                    FILE* const fp, int64_t const storage_mode,
                                    bool const use_compression,
                                    int64_t const step_idx,
                                    size_t const step_bytes_uncomp,
                                    size_t const step_bytes_comp,
#if DW_NDIM >= 3
                                    size_t nz,
#endif
#if DW_NDIM >= 2
                                    size_t ny,
#endif
                                    size_t nx) {
  if (storage_mode == STORAGE_NONE) return 0;

  if (storage_mode == STORAGE_DISK) {
    void* const read_dst = use_compression ? store_2 : store_1;
    size_t const size_to_read =
        use_compression ? step_bytes_comp : step_bytes_uncomp;
    if (dw_seek_snapshot(fp, step_idx, size_to_read) != 0) return 1;
    if (fread(read_dst, 1, size_to_read, fp) != size_to_read) return 1;
  }

  if (use_compression) {
    SC_FUNC(decompress_cpu)(store_2, store_1,
#if DW_NDIM >= 3
                            nz,
#endif
#if DW_NDIM >= 2
                            ny,
#endif
                            nx);
  }
  return 0;
}

#endif

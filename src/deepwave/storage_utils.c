#include "storage_utils.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "simple_compress.h"

void storage_save_snapshot_cpu(void* store_1, void* store_2, FILE* fp,
                               int64_t storage_mode, bool use_compression,
                               int64_t step_idx, size_t step_bytes_uncomp,
                               size_t step_bytes_comp, size_t n_elements,
                               int is_double) {
  void* source = store_1;
  size_t size_to_write = step_bytes_uncomp;

  if (storage_mode == STORAGE_NONE) return;

  if (use_compression) {
    simple_compress_cpu(store_1, store_2, n_elements, is_double);
    size_to_write = step_bytes_comp;
    source = store_2;
  }

  if (storage_mode == STORAGE_DISK) {
    // Append to file
    int64_t offset = step_idx * (int64_t)size_to_write;
    fseek(fp, offset, SEEK_SET);
    fwrite(source, 1, size_to_write, fp);
  }
}

void storage_load_snapshot_cpu(void* store_1, void* store_2, FILE* fp,
                               int64_t storage_mode, bool use_compression,
                               int64_t step_idx, size_t step_bytes_uncomp,
                               size_t step_bytes_comp, size_t n_elements,
                               int is_double) {
  if (storage_mode == STORAGE_NONE) return;

  if (storage_mode == STORAGE_DISK) {
    void* read_dst = use_compression ? store_2 : store_1;
    size_t size_to_read = use_compression ? step_bytes_comp : step_bytes_uncomp;
    int64_t offset = step_idx * (int64_t)size_to_read;
    fseek(fp, offset, SEEK_SET);
    fread(read_dst, 1, size_to_read, fp);
  }

  if (use_compression) {
    simple_decompress_cpu(store_2, store_1, n_elements, is_double);
  }
}

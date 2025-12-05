#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "common_gpu.h"
#include "simple_compress.h"
#include "storage_utils.h"

extern "C" {

int storage_save_snapshot_gpu(
    void* store_1, void* store_2, void* store_3, FILE* fp, int64_t storage_mode,
    bool use_compression, int64_t step_idx,
    size_t shot_bytes_uncomp,  // Bytes per shot (uncompressed)
    size_t shot_bytes_comp,    // Bytes per shot (compressed)
    size_t n_shots, size_t n_elements_per_shot, int is_double) {
  // Calculate total bytes for this step across all shots
  size_t total_uncomp = shot_bytes_uncomp * n_shots;
  size_t total_comp = shot_bytes_comp * n_shots;

  void* data_to_store = store_1;
  size_t bytes_to_store = total_uncomp;

  if (storage_mode == STORAGE_NONE) return 0;

  if (use_compression) {
    if (simple_compress_cuda(store_1, store_2, n_shots, n_elements_per_shot,
                             is_double) != 0)
      return 1;
    data_to_store = store_2;
    bytes_to_store = total_comp;
  }

  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    // Copy to Host
    cudaMemcpy(store_3, data_to_store, bytes_to_store, cudaMemcpyDeviceToHost);
    CHECK_KERNEL_ERROR
  }
  if (storage_mode == STORAGE_DISK) {
    int64_t offset =
        step_idx * n_shots *
        (int64_t)(use_compression ? shot_bytes_comp : shot_bytes_uncomp);
    fseek(fp, offset, SEEK_SET);
    fwrite(store_3, 1, bytes_to_store, fp);
  }
  return 0;
}

int storage_load_snapshot_gpu(void* store_1, void* store_2, void* store_3,
                              FILE* fp, int64_t storage_mode,
                              bool use_compression, int step_idx,
                              size_t shot_bytes_uncomp, size_t shot_bytes_comp,
                              size_t n_shots, size_t n_elements_per_shot,
                              int is_double) {
  size_t total_uncomp = shot_bytes_uncomp * n_shots;
  size_t total_comp = shot_bytes_comp * n_shots;

  size_t bytes_to_load = use_compression ? total_comp : total_uncomp;

  if (storage_mode == STORAGE_NONE) return 0;

  if (storage_mode == STORAGE_DISK) {
    // Load from disk to Host
    int64_t offset = step_idx * (int64_t)bytes_to_load;
    fseek(fp, offset, SEEK_SET);
    fread(store_3, 1, bytes_to_load, fp);
  }

  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    cudaMemcpy(use_compression ? store_2 : store_1, store_3, bytes_to_load,
               cudaMemcpyHostToDevice);
    CHECK_KERNEL_ERROR
  }

  if (use_compression) {
    // Decompress from store_2 to store_1
    if (simple_decompress_cuda(store_2, store_1, n_shots, n_elements_per_shot,
                               is_double) != 0)
      return 1;
  }
  return 0;
}
}

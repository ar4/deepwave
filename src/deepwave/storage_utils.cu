#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "common_gpu.h"
#include "simple_compress.h"
#include "storage_utils.h"

extern "C" {

int storage_save_snapshot_gpu(
    void const* const store_1, void* const store_2, void* const store_3,
    FILE* const fp, int64_t const storage_mode, bool const use_compression,
    int64_t const step_idx,
    size_t const shot_bytes_uncomp,  // Bytes per shot (uncompressed)
    size_t const shot_bytes_comp,    // Bytes per shot (compressed)
    size_t const n_shots, size_t const n_elements_per_shot, int const is_double,
    void* const stream) {
  // Calculate total bytes for this step across all shots
  size_t const total_uncomp = shot_bytes_uncomp * n_shots;
  size_t const total_comp = shot_bytes_comp * n_shots;

  void const* data_to_store = store_1;
  size_t bytes_to_store = total_uncomp;

  if (storage_mode == STORAGE_NONE) return 0;

  if (use_compression) {
    if (simple_compress_cuda(store_1, store_2, n_shots, n_elements_per_shot,
                             is_double, stream) != 0)
      return 1;
    data_to_store = (void const*)store_2;
    bytes_to_store = total_comp;
  }

  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    // Copy to Host
    cudaMemcpyAsync(store_3, data_to_store, bytes_to_store,
                    cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    CHECK_KERNEL_ERROR
  }
  if (storage_mode == STORAGE_DISK) {
    cudaStreamSynchronize((cudaStream_t)stream);
    int64_t const offset = step_idx * (int64_t)bytes_to_store;
    fseek(fp, offset, SEEK_SET);
    fwrite(store_3, 1, bytes_to_store, fp);
  }
  return 0;
}

int storage_load_snapshot_gpu(
    void* const store_1, void* const store_2, void* const store_3,
    FILE* const fp, int64_t const storage_mode, bool const use_compression,
    int const step_idx, size_t const shot_bytes_uncomp,
    size_t const shot_bytes_comp, size_t const n_shots,
    size_t const n_elements_per_shot, int const is_double, void* const stream) {
  size_t const total_uncomp = shot_bytes_uncomp * n_shots;
  size_t const total_comp = shot_bytes_comp * n_shots;

  size_t const bytes_to_load = use_compression ? total_comp : total_uncomp;

  if (storage_mode == STORAGE_NONE) return 0;

  if (storage_mode == STORAGE_DISK) {
    // Load from disk to Host
    cudaStreamSynchronize((cudaStream_t)stream);
    int64_t const offset = step_idx * (int64_t)bytes_to_load;
    fseek(fp, offset, SEEK_SET);
    size_t const count = fread(store_3, 1, bytes_to_load, fp);
    if (count != bytes_to_load) return 1;
  }

  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    cudaMemcpyAsync(use_compression ? store_2 : store_1, store_3, bytes_to_load,
                    cudaMemcpyHostToDevice, (cudaStream_t)stream);
    CHECK_KERNEL_ERROR
  }

  if (use_compression) {
    // Decompress from store_2 to store_1
    if (simple_decompress_cuda(store_2, store_1, n_shots, n_elements_per_shot,
                               is_double, stream) != 0)
      return 1;
  }
  return 0;
}
}

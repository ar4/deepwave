#include <mutex>

std::mutex &dw_cuda_config_mutex(int device) noexcept {
  // CUDA ordinals are dense from zero. The overflow lock preserves safety if
  // an unusually large system exceeds this table's capacity.
  static std::mutex locks[4096];
  static std::mutex overflow_lock;
  return device >= 0 && device < 4096 ? locks[device] : overflow_lock;
}

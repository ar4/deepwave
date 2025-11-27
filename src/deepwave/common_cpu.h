#ifndef DW_COMMON_CPU_H
#define DW_COMMON_CPU_H

// Record amplitudes from the wavefield at specified locations.
static inline void record_from_wavefield(
    DW_DTYPE const* __restrict const wavefield,
    int64_t const* __restrict const locations,
    DW_DTYPE* __restrict const amplitudes, int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i]) amplitudes[i] = wavefield[locations[i]];
  }
}

// Add amplitudes to the wavefield at specified locations.
static inline void add_to_wavefield(DW_DTYPE* __restrict const wavefield,
                                    int64_t const* __restrict const locations,
                                    DW_DTYPE const* __restrict const amplitudes,
                                    int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i]) wavefield[locations[i]] += amplitudes[i];
  }
}

// Combine gradients from multiple threads into a single gradient array.
// grad_thread is expected to be laid out as n_threads blocks each of size
// n_grid_points (i.e. threadidx * n_grid_points + i). This helper sums those
// per-thread buffers into `grad` for the interior grid.
static inline void combine_grad(DW_DTYPE* __restrict const grad,
                                DW_DTYPE const* __restrict const grad_thread,
                                int64_t const n_threads,
                                int64_t const n_grid_points) {
  int64_t i, threadidx;
#pragma omp simd
  for (i = 0; i < n_grid_points; ++i) {
    for (threadidx = 0; threadidx < n_threads; ++threadidx) {
      grad[i] += grad_thread[threadidx * n_grid_points + i];
    }
  }
}
#endif /* DW_COMMON_CPU_H */

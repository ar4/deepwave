#ifndef DW_COMMON_CPU_H
#define DW_COMMON_CPU_H


inline void record_from_wavefield(DW_DTYPE const* __restrict const wavefield,
                                  int64_t const* __restrict const locations,
                                  DW_DTYPE* __restrict const amplitudes, int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i]) amplitudes[i] = wavefield[locations[i]];
  }
}

inline void add_to_wavefield(DW_DTYPE* __restrict const wavefield,
                             int64_t const* __restrict const locations,
                             DW_DTYPE const* __restrict const amplitudes, int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i]) wavefield[locations[i]] += amplitudes[i];
  }
}

static void combine_grad(DW_DTYPE *__restrict const grad,
                         DW_DTYPE const *__restrict const grad_thread,
                         int64_t const n_threads, int64_t const ny, int64_t const nx) {
  int64_t y, x, threadidx;
  for (y = FD_PAD; y < ny - FD_PAD; ++y) {
    for (x = FD_PAD; x < nx - FD_PAD; ++x) {
      int64_t const i = y * nx + x;
      for (threadidx = 0; threadidx < n_threads; ++threadidx) {
        grad[i] += grad_thread[threadidx * ny * nx + i];
      }
    }
  }
}
#endif /* DW_COMMON_CPU_H */

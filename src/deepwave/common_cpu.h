#ifndef DW_COMMON_CPU_H
#define DW_COMMON_CPU_H

static void add_sources(DW_DTYPE *__restrict const wf,
                        DW_DTYPE const *__restrict const f,
                        int64_t const *__restrict const sources_i,
                        int64_t const n_sources_per_shot) {
  int64_t source_idx;
  for (source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
    wf[sources_i[source_idx]] += f[source_idx];
  }
}

static void record_receivers(DW_DTYPE *__restrict const r,
                             DW_DTYPE const *__restrict const wf,
                             int64_t const *__restrict const receivers_i,
                             int64_t const n_receivers_per_shot) {
  int64_t receiver_idx;
  for (receiver_idx = 0; receiver_idx < n_receivers_per_shot; ++receiver_idx) {
    r[receiver_idx] = wf[receivers_i[receiver_idx]];
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

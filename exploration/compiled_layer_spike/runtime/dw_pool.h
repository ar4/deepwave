/* Minimal persistent thread pool (pthreads, no OpenMP).
 *
 * The generated kernels call dw_pool_run once per time step; the calling
 * thread participates as worker 0, so a pool of N threads creates N-1
 * pthreads.  Dispatch is a generation counter that workers spin on (with a
 * short back-off to a nanosleep so an idle pool does not burn cores), which
 * keeps per-step dispatch in the low microseconds without a libgomp
 * dependency.  n_threads <= 1 creates no threads at all.
 */
#ifndef DW_POOL_H
#define DW_POOL_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef void (*dw_task_fn)(void *ctx, int64_t tid, int64_t nthreads);

typedef struct {
  int64_t nthreads;
  pthread_t *threads;
  atomic_llong gen;
  atomic_llong done;
  atomic_int stop;
  dw_task_fn fn;
  void *ctx;
} dw_pool;

typedef struct {
  dw_pool *pool;
  int64_t tid;
} dw_worker_arg;

static void dw_backoff(int64_t *spins) {
  if (*spins < 4000) {
    (*spins)++;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  } else {
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 20000;
    nanosleep(&ts, NULL);
  }
}

static void *dw_worker(void *arg_) {
  dw_worker_arg *arg = (dw_worker_arg *)arg_;
  dw_pool *p = arg->pool;
  int64_t tid = arg->tid;
  long long seen = 0;
  for (;;) {
    int64_t spins = 0;
    while (atomic_load_explicit(&p->gen, memory_order_acquire) == seen) {
      if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
      dw_backoff(&spins);
    }
    seen = atomic_load_explicit(&p->gen, memory_order_acquire);
    if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
    p->fn(p->ctx, tid, p->nthreads);
    atomic_fetch_add_explicit(&p->done, 1, memory_order_release);
  }
}

static dw_pool *dw_pool_create(int64_t n) {
  dw_pool *p = (dw_pool *)calloc(1, sizeof(dw_pool));
  if (!p) return NULL;
  p->nthreads = n;
  atomic_store(&p->gen, 0);
  atomic_store(&p->done, 0);
  atomic_store(&p->stop, 0);
  if (n > 1) {
    p->threads = (pthread_t *)calloc((size_t)(n - 1), sizeof(pthread_t));
    for (int64_t i = 1; i < n; ++i) {
      dw_worker_arg *a = (dw_worker_arg *)malloc(sizeof(dw_worker_arg));
      a->pool = p;
      a->tid = i;
      pthread_create(&p->threads[i - 1], NULL, dw_worker, a);
    }
  }
  return p;
}

/* One cached pool per process; resized (torn down and rebuilt) if the caller
 * asks for a different thread count. */
static dw_pool *dw_pool_cached = NULL;
static pthread_mutex_t dw_pool_lock = PTHREAD_MUTEX_INITIALIZER;

static dw_pool *dw_pool_get(int64_t n) {
  if (n < 1) n = 1;
  pthread_mutex_lock(&dw_pool_lock);
  if (dw_pool_cached && dw_pool_cached->nthreads != n) {
    dw_pool *old = dw_pool_cached;
    atomic_store(&old->stop, 1);
    atomic_fetch_add(&old->gen, 1);
    for (int64_t i = 1; i < old->nthreads; ++i) pthread_join(old->threads[i - 1], NULL);
    free(old->threads);
    free(old);
    dw_pool_cached = NULL;
  }
  if (!dw_pool_cached) dw_pool_cached = dw_pool_create(n);
  dw_pool *p = dw_pool_cached;
  pthread_mutex_unlock(&dw_pool_lock);
  return p;
}

static void dw_pool_run(dw_pool *p, dw_task_fn fn, void *ctx) {
  if (p->nthreads <= 1) {
    fn(ctx, 0, 1);
    return;
  }
  p->fn = fn;
  p->ctx = ctx;
  atomic_store_explicit(&p->done, 0, memory_order_release);
  atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
  fn(ctx, 0, p->nthreads);
  int64_t spins = 0;
  while (atomic_load_explicit(&p->done, memory_order_acquire) < p->nthreads - 1) {
    dw_backoff(&spins);
  }
}

/* Static row-band partition of [lo, hi) across nthreads.  Returned by value
 * rather than through out-pointers: address-taken loop bounds stop GCC
 * treating the bound as loop-invariant, which silently disables
 * vectorisation of the interior stencil. */
static inline int64_t dw_band_lo(int64_t lo, int64_t hi, int64_t tid,
                                 int64_t nthreads) {
  int64_t n = hi - lo;
  if (n < 0) n = 0;
  int64_t const q = n / nthreads;
  int64_t const r = n % nthreads;
  return lo + tid * q + (tid < r ? tid : r);
}

static inline int64_t dw_band_hi(int64_t lo, int64_t hi, int64_t tid,
                                 int64_t nthreads) {
  int64_t n = hi - lo;
  if (n < 0) n = 0;
  int64_t const q = n / nthreads;
  int64_t const r = n % nthreads;
  return dw_band_lo(lo, hi, tid, nthreads) + q + (tid < r ? 1 : 0);
}

#endif /* DW_POOL_H */

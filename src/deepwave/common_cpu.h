#ifndef DW_COMMON_CPU_H
#define DW_COMMON_CPU_H

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#define DW_FTZ_DAZ_X86 1
#include <pmmintrin.h>
#include <string.h>
#include <xmmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

/* Runtime ISA dispatch (x86): expands to a target_clones attribute so the
 * exported forward/backward functions -- into which the stencil loops are
 * inlined -- get default/AVX2/AVX-512 clones selected at runtime via an ifunc
 * resolver. Enabled by DW_X86_MULTIVERSION (set in CMakeLists.txt only for x86
 * kernel builds with a supporting compiler). A no-op everywhere else, including
 * aarch64 and the CUDA host launchers, so it changes no exported symbol name or
 * signature (the ctypes ABI is unaffected). Every propagator driver picks this
 * up from here; the #ifndef simply lets a translation unit that needs a
 * different dispatch policy define its own before including this header. */
#ifndef DW_TARGET_CLONES
#if defined(DW_X86_MULTIVERSION)
#define DW_TARGET_CLONES __attribute__((target_clones("default,avx2,avx512f")))
#else
#define DW_TARGET_CLONES
#endif
#endif

// Enable flush-to-zero (FTZ) and denormals-are-zero (DAZ) for the calling
// thread only, returning the previous MXCSR state so dw_ftz_daz_exit can
// restore it. Call at the start of a #pragma omp parallel region (one call
// per worker thread) and restore with dw_ftz_daz_exit before the region
// ends -- this is deliberately scoped per-thread-per-region, not a
// process-wide linker constructor (the old -Ofast-on-executables
// behaviour), so it cannot silently change the numerics of any other code
// that later runs on the same OS thread outside deepwave's own kernels
// (e.g. other torch/numpy operations in the same process). Measured
// benefit: a same-commit A/B (LD_PRELOAD constructor proxy for this exact
// MXCSR state, meso-tier + many-shots CPU sweeps, reps>=5) found decaying
// PML wave tails do reach the float32 subnormal range in practice and
// that avoiding the resulting slow denormal arithmetic is a material,
// consistently positive effect under this project's default -Ofast build
// (meso tier: many cases -20% to -70%; many-shots sweep, forward and
// forward_backward alike: -5.8% to -99.5%; zero regressions found at
// reps=11). See docs/accuracy.rst for the full record.
//
// This guard only activates in a fast-math build (DW_FAST_MATH_BUILD, set by
// CMake when DW_FAST_MATH=ON): a DW_FAST_MATH=OFF build is meant to be
// IEEE-precise, and FTZ/DAZ is itself an IEEE 754 deviation (subnormal
// results are flushed to zero instead of computed exactly), so silently
// applying it there would defeat the point of the precise build. With
// DW_FAST_MATH_BUILD undefined, dw_ftz_daz_enter returns 0 without touching
// MXCSR and dw_ftz_daz_exit is a no-op.
//
// DAZ is optional on early SSE2 processors. An unsupported MXCSR bit is
// reserved and setting it raises #GP. FXSAVE reports the supported-bit mask;
// Intel specifies 0xFFBF (without DAZ) when that mask is zero.
#if defined(DW_FTZ_DAZ_X86) && defined(DW_FAST_MATH_BUILD)
static inline unsigned int dw_supported_mxcsr_mask(void) {
#if defined(_MSC_VER)
  __declspec(align(16)) unsigned char state[512];
  _fxsave(state);
#else
  unsigned char state[512] __attribute__((aligned(16)));
  __asm__ volatile("fxsave %0" : "=m"(state));
#endif
  unsigned int mask;
  memcpy(&mask, state + 28, sizeof(mask));
  return mask;
}

static inline unsigned int dw_ftz_daz_mode(unsigned int current,
                                           unsigned int supported_mask) {
  unsigned int const mask = supported_mask == 0 ? 0xFFBFu : supported_mask;
  return (current | _MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON) & mask;
}
#endif

static inline unsigned int dw_ftz_daz_enter(void) {
#if defined(DW_FTZ_DAZ_X86) && defined(DW_FAST_MATH_BUILD)
  unsigned int const prev = _mm_getcsr();
  unsigned int const mask = dw_supported_mxcsr_mask();
  _mm_setcsr(dw_ftz_daz_mode(prev, mask));
  return prev;
#else
  return 0;
#endif
}

// Restore the MXCSR state captured by a prior dw_ftz_daz_enter() call.
static inline void dw_ftz_daz_exit(unsigned int prev) {
#if defined(DW_FTZ_DAZ_X86) && defined(DW_FAST_MATH_BUILD)
  _mm_setcsr(prev);
#else
  (void)prev;
#endif
}

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

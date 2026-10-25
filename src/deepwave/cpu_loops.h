#ifndef DW_CPU_LOOPS_H
#define DW_CPU_LOOPS_H

/* MSVC /openmp:experimental accepts standalone simd, but rejects the
 * combined for simd directive (C3002). Its legacy OpenMP mode also lacks
 * work-sharing collapse. Distribute the outer loop on MSVC and let /O2
 * vectorise the inner loops; GCC/Clang distribute the flattened loop nest.
 * Keep scheduling and nowait clauses in both parallel paths. */
#if defined(_MSC_VER) && !defined(__clang__)
#ifdef _OPENMP
#define DW_OMP_FOR_SIMD(...) __pragma(omp for __VA_ARGS__)
#else
#define DW_OMP_FOR_SIMD(...) __pragma(omp simd)
#endif
#else
#define DW_PRAGMA_IMPL(x) _Pragma(#x)
#define DW_PRAGMA(x) DW_PRAGMA_IMPL(x)
#ifdef _OPENMP
#define DW_OMP_FOR_SIMD(...) \
  DW_PRAGMA(omp for simd collapse(DW_NDIM) __VA_ARGS__)
#else
#define DW_OMP_FOR_SIMD(...) DW_PRAGMA(omp simd collapse(DW_NDIM))
#endif
#endif

/* The three PML regions are an optimisation hint, not a loop requirement. */
#if defined(__GNUC__) || defined(__clang__)
#define DW_UNROLL_PML _Pragma("GCC unroll 3")
#else
#define DW_UNROLL_PML
#endif

#endif /* DW_CPU_LOOPS_H */

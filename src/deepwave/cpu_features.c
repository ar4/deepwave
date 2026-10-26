/*
 * CPU capability probe for Deepwave's ctypes backend.
 *
 * This translation unit is deliberately tiny and is ALWAYS compiled for the
 * baseline ISA only (never -mavx2 / -march=native; see CMakeLists.txt, which
 * gives this file no ISA-raising flags). That guarantee matters: the whole
 * point of these helpers is to let the Python layer detect, *before the first
 * kernel call*, whether the running CPU can execute the compiled kernels. If
 * the probe itself contained AVX instructions it could SIGILL before it ever
 * got the chance to report the problem.
 *
 * Exported symbols (loaded by name via ctypes in backend_utils.py):
 *
 *   int deepwave_cpu_supports(const char *feature)
 *       Returns  1 if the running CPU supports the named x86 feature,
 *                0 if it does not,
 *               -1 if it cannot be determined (non-x86 build, or a compiler
 *                  without the query builtin, or an unrecognised name).
 *       Recognised names: "sse2", "avx", "avx2", "avx512f".
 *
 *   const char *deepwave_cpu_required_baseline(void)
 *       Returns the ISA that the compiled kernels require as an absolute
 *       baseline -- i.e. the ISA below which calling a kernel would execute an
 *       illegal instruction. With target_clones runtime dispatch enabled the
 *       kernels' baseline clone runs on any x86-64 CPU, so this is the empty
 *       string. It is non-empty ("avx2") only for a fixed-ISA build (kernels
 *       compiled with -mavx2 and no runtime dispatch), which is the case the
 *       Python guard uses to turn a would-be SIGILL into a clear error.
 *
 * These symbols are unrelated to the propagator ABI frozen by
 * tests/test_abi_stability.py, so adding them does not affect those tests.
 */

#include <string.h>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

int deepwave_cpu_supports(const char* feature) {
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
     defined(_M_IX86))
#if defined(__GNUC__) || defined(__clang__)
  if (feature == NULL) {
    return -1;
  }
  /* Safe to call on any CPU; initialises the internal CPUID cache. */
  __builtin_cpu_init();
  if (strcmp(feature, "sse2") == 0) {
    return __builtin_cpu_supports("sse2") ? 1 : 0;
  }
  if (strcmp(feature, "avx") == 0) {
    return __builtin_cpu_supports("avx") ? 1 : 0;
  }
  if (strcmp(feature, "avx2") == 0) {
    return __builtin_cpu_supports("avx2") ? 1 : 0;
  }
  if (strcmp(feature, "avx512f") == 0) {
    return __builtin_cpu_supports("avx512f") ? 1 : 0;
  }
  return -1;
#elif defined(_MSC_VER)
  if (feature == NULL) return -1;
  int info[4];
  __cpuid(info, 0);
  int const max_leaf = info[0];
  if (max_leaf < 1) return -1;
  __cpuid(info, 1);
  int const sse2 = (info[3] & (1 << 26)) != 0;
  int const avx_cpu = (info[2] & (1 << 28)) != 0;
  int const osxsave = (info[2] & (1 << 27)) != 0;
  unsigned __int64 const xcr0 = osxsave ? _xgetbv(0) : 0;
  int const avx = avx_cpu && (xcr0 & 0x6) == 0x6;
  if (strcmp(feature, "sse2") == 0) return sse2;
  if (strcmp(feature, "avx") == 0) return avx;
  if (max_leaf < 7) {
    if (strcmp(feature, "avx2") == 0 || strcmp(feature, "avx512f") == 0)
      return 0;
    return -1;
  }
  __cpuidex(info, 7, 0);
  if (strcmp(feature, "avx2") == 0) return avx && (info[1] & (1 << 5)) != 0;
  if (strcmp(feature, "avx512f") == 0)
    return avx && (xcr0 & 0xe0) == 0xe0 && (info[1] & (1 << 16)) != 0;
  return -1;
#else
  (void)feature;
  return -1;
#endif
#else
  /* Non-x86 architecture: the x86 feature question does not apply. */
  (void)feature;
  return -1;
#endif
}

const char* deepwave_cpu_required_baseline(void) {
#if defined(DW_BASELINE_ISA)
  return DW_BASELINE_ISA;
#else
  return "";
#endif
}

"""Check serial, collapsed, and MSVC outer-loop work sharing visits each cell."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("msvc_path", [False, True])
def test_cpu_loop_partition(tmp_path, ndim, parallel, msvc_path):
    """Include empty PML regions, nowait, and the scalar adjoint's schedule."""
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("GCC required for the native and emulated MSVC pragma paths")
    source = tmp_path / "loops.c"
    source.write_text(
        r"""
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
/* Translate only MSVC's pragma spelling; GCC executes its work-sharing path. */
#ifdef TEST_MSVC_PATH
#define _MSC_VER 1944
#define __pragma(...) _Pragma(#__VA_ARGS__)
#endif
#include "cpu_loops.h"

int main(void) {
  int visits[4][2][114] = {{{0}}};
  int const bounds[4] = {0, 0, 4, 19};
#ifdef _OPENMP
  omp_set_dynamic(0);
#pragma omp parallel num_threads(4)
#endif
  {
    int thread = 0;
    int64_t x;
#if DW_NDIM >= 2
    int64_t y;
#endif
#if DW_NDIM == 3
    int64_t z;
#endif
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    DW_UNROLL_PML
    for (int region = 0; region < 3; ++region) {
      DW_OMP_FOR_SIMD(nowait)
#if DW_NDIM == 3
      for (z = 0; z < 2; ++z)
#endif
#if DW_NDIM >= 2
        for (y = 0; y < 3; ++y)
#endif
          for (x = bounds[region]; x < bounds[region + 1]; ++x) {
#if DW_NDIM < 3
            int64_t z = 0;
#endif
#if DW_NDIM < 2
            int64_t y = 0;
#endif
            visits[thread][0][(z * 3 + y) * 19 + x]++;
          }
      DW_OMP_FOR_SIMD(schedule(static, 256) nowait)
#if DW_NDIM == 3
      for (z = 0; z < 2; ++z)
#endif
#if DW_NDIM >= 2
        for (y = 0; y < 3; ++y)
#endif
          for (x = bounds[region]; x < bounds[region + 1]; ++x) {
#if DW_NDIM < 3
            int64_t z = 0;
#endif
#if DW_NDIM < 2
            int64_t y = 0;
#endif
            visits[thread][1][(z * 3 + y) * 19 + x]++;
          }
    }
  }
  int const cells = DW_NDIM == 3 ? 114 : (DW_NDIM == 2 ? 57 : 19);
  for (int loop = 0; loop < 2; ++loop)
    for (int cell = 0; cell < cells; ++cell) {
      int count = 0;
      for (int thread = 0; thread < 4; ++thread)
        count += visits[thread][loop][cell];
      if (count != 1) return 1;
    }
  return 0;
}
"""
    )
    executable = tmp_path / "loops"
    include = Path(__file__).resolve().parents[1] / "src/deepwave"
    subprocess.run(  # noqa: S603 - fixed local compiler and generated source
        [
            compiler,
            "-O2",
            "-Wall",
            "-Werror",
            "-fopenmp" if parallel else "-fopenmp-simd",
            f"-DDW_NDIM={ndim}",
            *(["-DTEST_MSVC_PATH"] if msvc_path else []),
            "-I",
            str(include),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=30)  # noqa: S603

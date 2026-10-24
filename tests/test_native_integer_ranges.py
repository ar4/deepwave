"""Native integer arithmetic must preserve tensor ranges before narrowing."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("compiler", ["gcc", "g++", "clang", "clang++"])
def test_native_integer_ranges(compiler: str, tmp_path: Path) -> None:
    """Check C and C++ at 32/64-bit boundaries without allocating large arrays."""
    executable = shutil.which(compiler)
    if executable is None:
        pytest.skip(f"{compiler} unavailable")
    cpp = compiler.endswith("++")
    source = tmp_path / ("ranges.cpp" if cpp else "ranges.c")
    source.write_text(
        '#include "native_types.h"\n'
        "#include <assert.h>\n"
        "int main(void) {\n"
        "  assert(dw_fits_control_int(0));\n"
        "  assert(dw_fits_control_int(INT_MAX));\n"
        "  assert(!dw_fits_control_int(-1));\n"
        "  assert(!dw_fits_control_int(INT64_MIN));\n"
        "  assert(!dw_fits_control_int((int64_t)INT_MAX + 1));\n"
        "  assert(!dw_fits_control_int(INT64_MAX));\n"
        "  assert(dw_grid_index(0, 128, 127) == 127);\n"
        "  assert(dw_grid_index(16777216, 128, 1) == INT64_C(2147483649));\n"
        "  assert(dw_grid_index(33554432, 128, 7) == INT64_C(4294967303));\n"
        "  assert(dw_grid_index(UINT_MAX, 1024, 1023) == INT64_C(4398046511103));\n"
        "  assert(ceil_div(0, 128) == 0);\n"
        "  assert(ceil_div(1, 128) == 1);\n"
        "  assert(ceil_div(128, 128) == 1);\n"
        "  assert(ceil_div(129, 128) == 2);\n"
        "  assert(ceil_div(INT64_C(4294967296), 128) == 33554432);\n"
        "  assert(ceil_div(INT64_C(4294967297), 128) == 33554433);\n"
        "  assert(ceil_div(UINT_MAX, 1) == UINT_MAX);\n"
        "  assert(ceil_div(INT64_C(4294967296), 1) == 0);\n"
        "  assert(ceil_div(INT64_MAX, 128) == 0);\n"
        "  assert(ceil_div(-1, 128) == 0);\n"
        "  assert(ceil_div(128, 0) == 0);\n"
        "  assert(dw_grid_dimension(UINT_MAX) == UINT_MAX);\n"
        "  assert(dw_grid_dimension(UINT64_C(4294967296)) == 0);\n"
        "  assert(dw_grid_dimension(UINT64_MAX) == 0);\n"
        "  return 0;\n"
        "}\n"
    )
    binary = tmp_path / "ranges"
    include = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    subprocess.run(  # noqa: S603 - fixed local probe and discovered compiler
        [
            executable,
            "-std=c++11" if cpp else "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(include),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(  # noqa: S603 - execute only the locally built probe
        [str(binary)], check=True, capture_output=True, text=True
    )


def test_cuda_compression_rejects_truncated_launches() -> None:
    """CUDA wheels reject oversized launches on CPU hosts, before touching data."""
    import sys

    from deepwave._backend_utils import dll

    if not hasattr(dll, "simple_compress_compress_cuda_1d_float"):
        pytest.skip("CUDA symbols are not built")
    # Isolate the deliberately invalid metadata: a regression must not poison
    # another test's CUDA context. No large arrays or GPU allocation are needed.
    probe = """
import ctypes
import sys
import deepwave  # Establish the wheel's runtime DLL search paths on Windows.
lib = ctypes.CDLL(sys.argv[1])
for ndim in (1, 2, 3):
    for dtype in ("float", "double"):
        for direction in ("compress", "decompress"):
            fn = getattr(lib, f"simple_compress_{direction}_cuda_{ndim}d_{dtype}")
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            fn.argtypes += [ctypes.c_size_t] * ndim + [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            shape = [8] * ndim
            assert fn(None, None, 0, *shape, None) == 0
            # cudaErrorInvalidConfiguration is 9 in the CUDA runtime ABI.
            assert fn(None, None, 2**32 + 1, *shape, None) == 9
"""
    subprocess.run(  # noqa: S603 - fixed probe and the already loaded native library
        [sys.executable, "-c", probe, str(dll._name)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

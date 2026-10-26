"""Direct checks of native history storage at numerical and offset limits."""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import deepwave
from deepwave._backend_utils import dll


def _standalone_ubsan_env() -> dict[str, str]:
    """Keep Clang's standalone runtime independent of the pytest preload."""
    env = os.environ.copy()
    # The installed extension may require GCC or Clang ASan in pytest. Mixing
    # that preload with Clang's statically linked UBSan can recurse at startup.
    # These harnesses retain their own explicit UBSan instrumentation.
    env.pop("LD_PRELOAD", None)
    return env


def test_compiled_disk_gradient_uses_unicode_storage_path(tmp_path: Path) -> None:
    """Native write/read and cleanup work through a UTF-8 disk-storage path."""
    storage_path = tmp_path / "履歴_é_😀"
    storage_path.mkdir()
    speed = torch.full((24, 24), 1500.0, dtype=torch.float64, requires_grad=True)
    source = torch.zeros(1, 1, 16, dtype=torch.float64)
    source[0, 0, 2] = 1
    source.requires_grad_()
    output = deepwave.scalar(
        speed,
        5.0,
        0.001,
        source_amplitudes=source,
        source_locations=torch.tensor([[[12, 12]]]),
        receiver_locations=torch.tensor([[[12, 14]]]),
        pml_width=2,
        pml_freq=25.0,
        max_vel=1500.0,
        accuracy=2,
        backend="compiled",
        storage_mode="disk",
        storage_path=storage_path,
    )
    output.receiver_amplitudes.square().sum().backward()
    assert speed.grad is not None
    assert speed.grad.abs().sum() > 0
    assert source.grad is not None
    assert source.grad.abs().sum() > 0
    # The output retains its backward context; files remain available for a
    # retained/repeated backward until that graph is released.
    del output
    gc.collect()
    assert not any(storage_path.iterdir())


@pytest.mark.parametrize(
    ("dtype", "peak"),
    [
        (ctypes.c_float, 0.0),
        (ctypes.c_float, 1.0),
        (ctypes.c_float, 1e30),
        (ctypes.c_float, 1e-37),
        (ctypes.c_float, 4e-37),
        (ctypes.c_float, 1e-36),
        (ctypes.c_float, 1e-44),
        (ctypes.c_float, torch.finfo(torch.float32).max),
        (ctypes.c_double, 1e-306),
        (ctypes.c_double, 1e300),
        (ctypes.c_double, 1e-320),
        (ctypes.c_double, torch.finfo(torch.float64).max),
    ],
)
@pytest.mark.parametrize("shape", [(8,), (3, 9), (2, 3, 9)])
def test_native_compression_tiny_blocks(dtype, peak, shape):
    """Quantization error stays within one half level, including partial blocks."""
    _check_compression_roundtrip(dll, dtype, peak, shape)


def _check_compression_roundtrip(library, dtype, peak, shape):
    """Check the numerical bound against an explicitly selected native library."""
    name = "float" if dtype is ctypes.c_float else "double"
    ndim = len(shape)
    n = 1
    blocks = 1
    for width in shape:
        n *= width
        blocks *= (width + 7) // 8
    values = [peak, -peak, peak / 2, -peak / 2, 0, 0, 0, 0]
    original = (dtype * n)(*(values[i % 8] for i in range(n)))
    packed = (ctypes.c_byte * (blocks * ctypes.sizeof(dtype) + n))()
    recovered = (dtype * n)()
    args = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * ndim
    compress = getattr(library, f"simple_compress_compress_cpu_{ndim}d_{name}")
    decompress = getattr(library, f"simple_compress_decompress_cpu_{ndim}d_{name}")
    compress.argtypes = decompress.argtypes = args
    compress.restype = decompress.restype = None
    compress(original, packed, *shape)
    decompress(packed, recovered, *shape)

    # The code stores one block maximum and 8-bit signed levels. Each input
    # should be within half a quantization step plus cast/underflow rounding.
    actual_peak = max(abs(x) for x in original)
    errors = [abs(a - b) for a, b in zip(original, recovered)]
    min_subnormal = 1.4e-45 if dtype is ctypes.c_float else 5e-324
    assert max(errors) <= actual_peak / 254 + actual_peak * 1e-6 + 2 * min_subnormal
    assert all(b == 0 for a, b in zip(original, recovered) if a == 0)


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
def test_compression_with_64_bit_long_double(compiler, tmp_path):
    """Exercise Apple Silicon's long-double range using the release build flags."""
    cc = shutil.which(compiler)
    cmake = shutil.which("cmake")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        pytest.skip("-mlong-double-64 regression harness requires Linux x86_64")
    if cc is None or cmake is None:
        pytest.skip("CMake and the requested C compiler are required")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(  # noqa: S603 - trusted compiler and local source
        [
            cmake,
            "-S",
            str(root),
            "-B",
            str(tmp_path),
            "-DDW_SKIP_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_COMPILER={cc}",
            "-DCMAKE_C_FLAGS=-mlong-double-64",
        ],
        check=True,
        text=True,
    )
    target = "simple_compress_1_2_double_obj"
    subprocess.run(  # noqa: S603 - trusted compiler and local source
        [cmake, "--build", str(tmp_path), "--target", target],
        check=True,
        text=True,
    )
    obj = next((tmp_path / "CMakeFiles" / f"{target}.dir").rglob("*.o"))
    library = tmp_path / "compression.so"
    subprocess.run(  # noqa: S603 - trusted compiler and local object
        [cc, "-shared", str(obj), "-o", str(library)], check=True
    )
    for peak in (0.0, 1e-320, 1e-306, 1.0, torch.finfo(torch.float64).max):
        _check_compression_roundtrip(
            ctypes.CDLL(str(library)), ctypes.c_double, peak, (8,)
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("torch_dtype", "peak"),
    [
        (torch.float32, 0.0),
        (torch.float32, 1.0),
        (torch.float32, 1e30),
        (torch.float32, 1e-37),
        (torch.float32, 4e-37),
        (torch.float32, 1e-36),
        (torch.float32, 1e-44),
        (torch.float32, torch.finfo(torch.float32).max),
        (torch.float64, 1e-306),
        (torch.float64, 1e300),
        (torch.float64, 1e-320),
        (torch.float64, torch.finfo(torch.float64).max),
    ],
)
@pytest.mark.parametrize("shape", [(8,), (3, 9), (2, 3, 9)])
def test_cuda_compression_tiny_blocks(torch_dtype, peak, shape):
    """The device quantizer obeys the same bound across batched dimensions."""
    name = "float" if torch_dtype is torch.float32 else "double"
    ndim = len(shape)
    n = 1
    blocks = 1
    for width in shape:
        n *= width
        blocks *= (width + 7) // 8
    batch = 2
    values = [peak, -peak, peak / 2, -peak / 2, 0, 0, 0, 0]
    original = torch.tensor(
        [values[i % 8] for i in range(n * batch)],
        dtype=torch_dtype,
        device="cuda",
    )
    packed = torch.empty(
        blocks * batch * original.element_size() + n * batch,
        dtype=torch.uint8,
        device="cuda",
    )
    recovered = torch.empty_like(original)
    args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    args += [ctypes.c_size_t] * ndim + [ctypes.c_void_p]
    compress = getattr(dll, f"simple_compress_compress_cuda_{ndim}d_{name}")
    decompress = getattr(dll, f"simple_compress_decompress_cuda_{ndim}d_{name}")
    compress.argtypes = decompress.argtypes = args
    compress.restype = decompress.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    assert compress(original.data_ptr(), packed.data_ptr(), batch, *shape, stream) == 0
    assert (
        decompress(packed.data_ptr(), recovered.data_ptr(), batch, *shape, stream) == 0
    )
    torch.cuda.synchronize()
    inputs = original.cpu().tolist()
    outputs = recovered.cpu().tolist()
    actual_peak = max(abs(x) for x in inputs)
    min_subnormal = 1.4e-45 if torch_dtype is torch.float32 else 5e-324
    bound = actual_peak / 254 + actual_peak * 1e-6 + 2 * min_subnormal
    assert max(abs(a - b) for a, b in zip(inputs, outputs)) <= bound
    assert all(b == 0 for a, b in zip(inputs, outputs) if a == 0)


def test_storage_seek_accepts_offsets_beyond_4_gib_without_allocating_file(tmp_path):
    """Compile the platform's actual 64-bit seek branch and inspect positions."""
    compiler = shutil.which("cmake") if os.name == "nt" else shutil.which("cc")
    if compiler is None:
        if os.name == "nt":
            pytest.fail("Windows wheel tests require CMake and Visual Studio")
        pytest.skip("C compiler unavailable for native seek helper check")
    header_dir = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    source = tmp_path / "seek_check.c"
    source.write_text(
        '#include "storage_utils.h"\n'
        "int main(void) {\n"
        "  FILE *fp = tmpfile();\n"
        "  if (!fp) return 1;\n"
        "  if (dw_seek_snapshot(fp, (1LL << 30) + 3, 4)) return 2;\n"
        "#ifdef _WIN32\n"
        "  if (_ftelli64(fp) != (1LL << 32) + 12) return 3;\n"
        "#else\n"
        "  if (ftello(fp) != (1LL << 32) + 12) return 3;\n"
        "  if (fputc(0x5a, fp) == EOF) return 8;\n"
        "  if (dw_seek_snapshot(fp, (1LL << 30) + 3, 4)) return 9;\n"
        "  if (fgetc(fp) != 0x5a) return 10;\n"
        "#endif\n"
        "  if (dw_seek_snapshot(fp, (1LL << 29) + 5, 4)) return 4;\n"
        "#ifdef _WIN32\n"
        "  if (_ftelli64(fp) != (1LL << 31) + 20) return 5;\n"
        "#else\n"
        "  if (ftello(fp) != (1LL << 31) + 20) return 5;\n"
        "#endif\n"
        "  if (!dw_seek_snapshot(fp, INT64_MAX, 2)) return 6;\n"
        "  if (!dw_seek_snapshot(fp, -1, 4)) return 7;\n"
        "  fclose(fp);\n"
        "  return 0;\n"
        "}\n"
    )
    executable = tmp_path / ("seek_check.exe" if os.name == "nt" else "seek_check")
    if os.name == "nt":
        shutil.copy2(header_dir / "storage_utils.h", tmp_path / "storage_utils.h")
        (tmp_path / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(seek_check C)\n"
            "add_executable(seek_check seek_check.c)\n"
        )
        build_dir = tmp_path / "build"
        subprocess.run(  # noqa: S603 - local CMake and generated source
            [
                compiler,
                "-S",
                str(tmp_path),
                "-B",
                str(build_dir),
                "-G",
                "Visual Studio 17 2022",
                "-A",
                "x64",
            ],
            check=True,
            cwd=tmp_path,
            text=True,
        )
        subprocess.run(  # noqa: S603 - local CMake and generated source
            [compiler, "--build", str(build_dir), "--config", "Release"],
            check=True,
            cwd=tmp_path,
            text=True,
        )
        executable = build_dir / "Release" / "seek_check.exe"
    else:
        command = [
            compiler,
            "-std=gnu11",
            "-Wall",
            "-Werror",
            "-I",
            str(header_dir),
            str(source),
            "-o",
            str(executable),
        ]
        subprocess.run(  # noqa: S603 - local compiler and generated source
            command, check=True, cwd=tmp_path, text=True
        )
    subprocess.run(  # noqa: S603 - executable built from generated source
        [str(executable)], check=True, cwd=tmp_path, text=True
    )


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("clang") is None,
    reason="requires a POSIX Clang UBSan toolchain",
)
def test_unused_native_storage_pointers_do_not_perform_arithmetic(
    tmp_path: Path,
) -> None:
    """Absent histories stay null, including nonzero shot/time offsets."""
    compiler = shutil.which("clang")
    assert compiler is not None
    header_dir = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    source = tmp_path / "null_storage.c"
    source.write_text(
        r"""
#define DW_NDIM 1
#define DW_DTYPE float
#define DIM_ARGS 8
#include <assert.h>
#include "storage_utils.h"

int main(void) {
  int64_t t=1, step_ratio=1, n_shots=2, n_grid_points=8, si=8, shot=1;
  size_t shot_bytes_comp=8, shot_bytes_uncomp=32;
  float buffer[64];
  unsigned char packed[64];
  int storage_err=0;
  FILE *fp_foo=NULL;
  for (int storage_mode=STORAGE_DEVICE; storage_mode<=STORAGE_NONE; ++storage_mode) {
    for (int compression=0; compression<2; ++compression) {
      bool storage_compression=compression != 0;
      for (int present=0; present<2; ++present) {
        float *foo_store_1=present ? buffer : NULL;
        void *foo_store_2=present ? packed : NULL;
        float *expected_1=present ? buffer + si +
            ((storage_mode==STORAGE_DEVICE && !storage_compression) ? 16 : 0) : NULL;
        void *expected_2=present ? packed +
            ((storage_mode==STORAGE_DEVICE && storage_compression) ? 24 : 8) : NULL;
        {
          SETUP_STORE_SAVE(foo, false)
          assert(foo_store_1_t == expected_1);
          assert(foo_store_2_t == expected_2);
        }
        {
          SETUP_STORE_LOAD(foo, false)
          assert(foo_store_1_t == expected_1);
          assert(foo_store_2_t == expected_2);
        }
      }
    }
  }
storage_cleanup:
  return storage_err;
}
"""
    )
    executable = tmp_path / "null_storage"
    subprocess.run(  # noqa: S603 - fixed local sanitizer harness
        [
            compiler,
            "-std=gnu11",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-I",
            str(header_dir),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        text=True,
        env=_standalone_ubsan_env(),
        timeout=120,
    )
    subprocess.run(  # noqa: S603 - executable built from fixed local harness
        [str(executable)],
        check=True,
        text=True,
        env=_standalone_ubsan_env(),
        timeout=10,
    )


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("clang") is None,
    reason="requires a POSIX Clang UBSan toolchain",
)
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("empty_survey", 0),
        ("missing_history", 1),
        ("missing_raw_history", 1),
        ("missing_receiver_indices", 1),
        ("missing_source_indices", 1),
        ("unsampled_missing_history", 0),
        ("invalid_stride", 1),
        ("zero_shots", 0),
        ("zero_steps", 0),
        ("negative_shots", 1),
        ("negative_steps", 1),
    ],
)
def test_native_scalar_empty_survey_has_no_null_arithmetic(
    tmp_path: Path, case: str, expected: int
) -> None:
    """Real kernels reject missing required buffers and accept unused ones."""
    compiler = shutil.which("clang")
    assert compiler is not None
    root = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    source = tmp_path / "empty_survey.c"
    source.write_text(
        r"""
#define DW_ACCURACY 2
#define DW_DEVICE cpu
#include "scalar.c"
int main(void) {
  double v[16]={0}, w[32]={0}, wp[32]={0};
  double psi[32]={0}, psin[32]={0}, zeta[32]={0};
  double a[16]={0},b[16]={0},db[16]={0};
  double history[96]={0};
  unsigned char packed[1024]={0};
  double *store=history;
  int64_t nt=3, shots=2, sources=0, receivers=0, stride=1, start=0;
  bool model_grad=false, raw=false;
  CASE_SETUP
  int status=FUNC(forward)(v,NULL,w,wp,psi,psin,zeta,
      store,NULL,packed,NULL,NULL,NULL,
      a,b,db,NULL,NULL,1,1,.01,nt,shots,16,sources,receivers,stride,
      STORAGE_DEVICE,128,0,model_grad,false,false,start,1,15,0,0,
      NULL,NULL,NULL,NULL,NULL,raw,STORAGE_DEVICE,128,0,false,1,NULL);
  return status == EXPECTED_STATUS ? 0 : 1;
}
""".replace(
            "CASE_SETUP",
            {
                "empty_survey": "",
                "missing_history": "store=NULL; model_grad=true;",
                "missing_raw_history": "raw=true;",
                "missing_receiver_indices": "receivers=1;",
                "missing_source_indices": "sources=1;",
                "unsampled_missing_history": (
                    "store=NULL; model_grad=true; nt=1; start=1; stride=2;"
                ),
                "invalid_stride": "stride=0;",
                "zero_shots": "shots=0; stride=0;",
                "zero_steps": "nt=0; stride=0;",
                "negative_shots": "shots=-1;",
                "negative_steps": "nt=-1;",
            }[case],
        ).replace("EXPECTED_STATUS", str(expected))
    )
    executable = tmp_path / "empty_survey"
    subprocess.run(  # noqa: S603 - fixed local real-kernel sanitizer harness
        [
            compiler,
            "-std=gnu11",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-DDW_NDIM=1",
            "-DDW_DTYPE=double",
            "-I",
            str(root),
            str(source),
            str(root / "storage_utils.c"),
            str(root / "simple_compress.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        text=True,
        env=_standalone_ubsan_env(),
        timeout=120,
    )
    subprocess.run(  # noqa: S603 - executable built from fixed local harness
        [str(executable)],
        check=True,
        text=True,
        env=_standalone_ubsan_env(),
        timeout=10,
    )

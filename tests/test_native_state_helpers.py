"""Fault injection for native CUDA state and disk-history helpers."""

# ruff: noqa: S603 - compile and execute fixed, locally generated test programs

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ unavailable")
def test_native_cuda_storage_failures_and_cleanup(tmp_path: Path) -> None:
    """Host API errors stop disk I/O; all handles close and device is restored."""
    if os.name == "nt":
        pytest.skip("requires POSIX cookie streams")
    compiler = shutil.which("g++")
    assert compiler is not None
    root = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    (tmp_path / "cuda_runtime.h").write_text(
        """
#pragma once
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 30,
       cudaMemcpyDeviceToHost = 1, cudaMemcpyHostToDevice = 2 };
extern cudaError_t cudaGetDevice(int*);
extern cudaError_t cudaSetDevice(int);
extern cudaError_t cudaDeviceSynchronize(void);
extern cudaError_t cudaStreamSynchronize(cudaStream_t);
extern cudaError_t cudaMemcpyAsync(void*, const void*, unsigned long, int,
                                   cudaStream_t);
extern cudaError_t cudaPeekAtLastError(void);
extern cudaError_t cudaStreamDestroy(cudaStream_t);
extern cudaError_t cudaEventDestroy(cudaEvent_t);
extern const char* cudaGetErrorString(cudaError_t);
"""
    )
    (tmp_path / "check.cpp").write_text(
        """
#define DW_NDIM 1
#define DW_DTYPE float
#include "cuda_runtime.h"
#include <cassert>
#include <cstring>
#include <mutex>
#include "storage_utils.cu"

static int device = 0, fail_get = 0, fail_set = 0;
static int fail_copy = 0, fail_sync = 0, copies = 0, syncs = 0;
static int decompresses = 0, closes = 0;
cudaError_t cudaGetDevice(int* value) {
  if (fail_get) return 19;
  *value = device; return 0;
}
cudaError_t cudaSetDevice(int value) {
  if (fail_set && value == 1) return 20;
  device = value; return 0;
}
cudaError_t cudaDeviceSynchronize() { return 0; }
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++syncs; return fail_sync ? 17 : 0;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, unsigned long size,
                            int, cudaStream_t) {
  ++copies;
  if (fail_copy) return 18;
  memcpy(dst, src, size);
  return 0;
}
cudaError_t cudaPeekAtLastError() { return 0; }
cudaError_t cudaStreamDestroy(cudaStream_t) { return 0; }
cudaError_t cudaEventDestroy(cudaEvent_t) { return 0; }
const char* cudaGetErrorString(cudaError_t) { return "injected error"; }
std::mutex& dw_cuda_config_mutex(int) noexcept {
  static std::mutex mutex; return mutex;
}
extern "C" int simple_compress_compress_cuda_1d_float(
    const void*, void*, size_t, size_t, void*) { return 0; }
extern "C" int simple_compress_decompress_cuda_1d_float(
    const void*, void*, size_t, size_t, void*) { ++decompresses; return 0; }

static ssize_t write_cookie(void*, const char*, size_t) { return -1; }
static int close_cookie(void*) { ++closes; return 0; }

int main() {
  {
    ScopedCudaDevice guard(1);
    assert(guard.init_error == 0 && device == 1);
    ScopedCudaConfig config(1);
    assert(config.init_error == 0);
    ScopedFileGroup files;
    ScopedFile first, second;
    files.add(first); files.add(second);
    cookie_io_functions_t io = {};
    io.write = write_cookie;
    io.close = close_cookie;
    first.fp = fopencookie(nullptr, "w", io);
    second.fp = fopencookie(nullptr, "w", io);
    assert(first.fp && second.fp);
    assert(fwrite("x", 1, 1, first.fp) == 1);
    assert(fwrite("y", 1, 1, second.fp) == 1);
    assert(config.finish() == 0);
    assert(files.close_all() != 0);
    assert(closes == 2 && !first.fp && !second.fp);
  }
  assert(device == 0 && closes == 2);
  fail_get = 1;
  { ScopedCudaDevice guard(1); assert(guard.init_error == 19); }
  assert(device == 0);
  fail_get = 0; fail_set = 1;
  { ScopedCudaDevice guard(1); assert(guard.init_error == 20); }
  assert(device == 0);
  fail_set = 0;

  float device_data = 7.0f, host_data = 0.0f;
  FILE* file = tmpfile();
  assert(file);
  fail_copy = 1;
  int result = storage_save_snapshot_gpu_1d_float(
      &device_data, nullptr, &host_data, file, STORAGE_DISK, false,
      0, sizeof(float), sizeof(float), 1, 1, nullptr);
  assert(result == 18 && ftello(file) == 0 && syncs == 0);
  fail_copy = 0; fail_sync = 1;
  result = storage_save_snapshot_gpu_1d_float(
      &device_data, nullptr, &host_data, file, STORAGE_DISK, false,
      0, sizeof(float), sizeof(float), 1, 1, nullptr);
  assert(result == 17 && ftello(file) == 0);
  fail_sync = 0;
  assert(fwrite(&device_data, sizeof(float), 1, file) == 1);
  fflush(file);
  rewind(file);
  fail_sync = 1;
  result = storage_load_snapshot_gpu_1d_float(
      &device_data, nullptr, &host_data, file, STORAGE_DISK, false,
      0, sizeof(float), sizeof(float), 1, 1, nullptr);
  assert(result == 17 && ftello(file) == 0);
  fail_sync = 0; fail_copy = 1;
  result = storage_load_snapshot_gpu_1d_float(
      &device_data, nullptr, &host_data, file, STORAGE_DISK, false,
      0, sizeof(float), sizeof(float), 1, 1, nullptr);
  assert(result == 18 && decompresses == 0);
  assert(dw_close_file(&file) == 0 && !file);
  return 0;
}
"""
    )
    executable = tmp_path / "check"
    subprocess.run(
        [
            compiler,
            "-std=gnu++17",
            "-Wall",
            "-Werror",
            "-I",
            str(tmp_path),
            "-I",
            str(root),
            str(tmp_path / "check.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_native_utf8_disk_history_path(tmp_path: Path) -> None:
    """The shared opener interprets a Python UTF-8 path on either platform."""
    filename = "history-é-波.bin"
    literal = ", ".join(f"0x{byte:02x}" for byte in filename.encode("utf-8"))
    root = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    source = tmp_path / "path_check.c"
    source.write_text(
        '#include "storage_utils.h"\n'
        "#ifdef _WIN32\n#include <mbctype.h>\n#endif\n"
        f"static const char name[] = {{{literal}, 0}};\n"
        "int main(void) {\n"
        "#ifdef _WIN32\n"
        "  if (_setmbcp(1252) != 0) return 1;\n"
        "#endif\n"
        '  FILE *fp = dw_open_file_utf8(name, "wb");\n'
        "  if (!fp) return 2;\n"
        '  if (fwrite("ok", 1, 2, fp) != 2) return 3;\n'
        "  if (dw_close_file(&fp) != 0 || fp != NULL) return 4;\n"
        '  fp = dw_open_file_utf8(name, "rb");\n'
        "  if (!fp) return 5;\n"
        "  char result[2];\n"
        "  if (fread(result, 1, 2, fp) != 2) return 6;\n"
        "  if (dw_close_file(&fp) != 0 || fp != NULL) return 7;\n"
        "  return result[0] == 'o' && result[1] == 'k' ? 0 : 8;\n"
        "}\n"
    )
    if os.name == "nt":
        cmake = shutil.which("cmake")
        if cmake is None:
            pytest.fail("Windows native path test requires CMake")
        (tmp_path / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(path_check C)\n"
            "add_executable(path_check path_check.c)\n"
            f'target_include_directories(path_check PRIVATE "{root.as_posix()}")\n'
        )
        build = tmp_path / "build"
        subprocess.run(
            [
                cmake,
                "-S",
                str(tmp_path),
                "-B",
                str(build),
                "-G",
                "Visual Studio 17 2022",
                "-A",
                "x64",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [cmake, "--build", str(build), "--config", "Release"],
            check=True,
            capture_output=True,
            text=True,
        )
        executable = build / "Release" / "path_check.exe"
    else:
        cc = shutil.which("cc")
        if cc is None:
            pytest.skip("C compiler unavailable")
        executable = tmp_path / "path_check"
        subprocess.run(
            [
                cc,
                "-std=gnu11",
                "-Wall",
                "-Werror",
                "-I",
                str(root),
                str(source),
                "-o",
                str(executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [str(executable)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / filename).read_bytes() == b"ok"

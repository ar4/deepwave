#!/usr/bin/env python3
"""Build a focused CPU library and run pytest against this source checkout.

The focused CMake mode builds every CPU propagator for 2D, float32/float64,
and all supported stencil accuracies. It avoids compiling unrelated 1D/3D
variants and CUDA while still loading the current Python sources and native
code together.

Usage::

    python scripts/focused_native_test.py -- \
        tests/test_native_boundary_initial_gradients.py

Pytest arguments can be passed after ``--``. The build lives under the system
temporary directory unless ``DEEPWAVE_FOCUSED_BUILD_DIR`` is set.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(
    os.environ.get(
        "DEEPWAVE_FOCUSED_BUILD_DIR",
        str(Path(tempfile.gettempdir()) / "deepwave-focused-native"),
    )
).expanduser()
LIBRARY_NAME = {
    "Linux": "libdeepwave_C.so",
    "Darwin": "libdeepwave_C.dylib",
    "Windows": "libdeepwave_C.dll",
}[platform.system()]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Print and run one checked subprocess in the repository root."""
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)  # noqa: S603


def main() -> int:
    """Build the focused library and execute the requested pytest selection."""
    args = sys.argv[1:]
    if "--" in args:
        sep = args.index("--")
        cmake_args, pytest_args = args[:sep], args[sep + 1 :]
    else:
        cmake_args, pytest_args = [], args
    if not pytest_args:
        pytest_args = ["tests/test_native_boundary_initial_gradients.py"]

    run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(BUILD_DIR),
            "-DDW_FOCUSED_TEST_BUILD=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            *cmake_args,
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            "deepwave_C",
            "--parallel",
            "4",
        ]
    )

    env = os.environ.copy()
    env["DEEPWAVE_NATIVE_LIBRARY"] = str(BUILD_DIR / LIBRARY_NAME)
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    run([sys.executable, "-m", "pytest", *pytest_args], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

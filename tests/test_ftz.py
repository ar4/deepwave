"""Tests that deepwave never leaves the process in FTZ/DAZ mode.

Flush-to-zero/denormals-are-zero (FTZ/DAZ). The C kernels' fast-math build
sets FTZ/DAZ for the duration of each OpenMP-parallel region only (see
``dw_ftz_daz_enter``/``dw_ftz_daz_exit`` in ``common_cpu.h``), restoring the
prior MXCSR state before returning control to the caller. Historically a
build could also leak FTZ/DAZ process-wide by
linking ``-Ofast`` on the *link* line, which on GCC before 13 pulls in
``crtfastmath.o``'s constructor and sets FTZ/DAZ for the whole process the
moment the shared library is loaded, regardless of any per-thread guard.
Both mechanisms are checked here: a still-denormal float32 immediately after
``import deepwave`` rules out the link-time constructor, and a still-denormal
float32 after running a propagation rules out the per-thread guard leaking.
"""

import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from helpers import c_backend_available

from deepwave import scalar
from deepwave.wavelets import ricker


def test_cpu_guard_respects_supported_mxcsr_bits(tmp_path: Path) -> None:
    """Reserved DAZ stays clear; the real processor mode is restored exactly."""
    compiler = shutil.which("cc")
    if compiler is None or platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("requires an x86-64 C compiler")
    source = tmp_path / "mxcsr.c"
    source.write_text(
        '#include <stdint.h>\n#include "common_cpu.h"\n'
        "int main(void) {\n"
        "  unsigned int prior = _mm_getcsr();\n"
        "  unsigned int mask = dw_supported_mxcsr_mask();\n"
        "  if (dw_ftz_daz_mode(0x1F80u, 0) != 0x9F80u) return 1;\n"
        "  if (dw_ftz_daz_mode(0x1F80u, 0xFFBFu) != 0x9F80u) return 2;\n"
        "  if (dw_ftz_daz_mode(0x1F80u, 0xFFFFu) != 0x9FC0u) return 3;\n"
        "  unsigned int saved = dw_ftz_daz_enter();\n"
        "  unsigned int inside = _mm_getcsr();\n"
        "  dw_ftz_daz_exit(saved);\n"
        "  if (saved != prior || _mm_getcsr() != prior) return 4;\n"
        "  if (inside != dw_ftz_daz_mode(prior, mask)) return 5;\n"
        "  return 0;\n}\n"
    )
    executable = tmp_path / "mxcsr"
    include_dir = Path(__file__).resolve().parents[1] / "src" / "deepwave"
    subprocess.run(  # noqa: S603 - fixed compiler and local probe paths
        [
            compiler,
            "-DDW_FAST_MATH_BUILD",
            "-DDW_DTYPE=float",
            "-I",
            str(include_dir),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True)  # noqa: S603 - local compiled probe


def _subnormal_survives() -> bool:
    """Whether a float32 subnormal multiply is still nonzero (not flushed)."""
    return bool((torch.tensor([1e-40], dtype=torch.float32) * 0.5).item() != 0.0)


def _run_small_scalar_propagation(device: torch.device) -> None:
    """Run a small compiled-backend scalar propagation, discarding the result."""
    torch.manual_seed(0)
    shape = (18, 18)
    dx, dt, nt = 5.0, 0.0007, 60
    freq = 15.0
    sa = ricker(freq, nt, dt, 1.4 / freq).reshape(1, 1, nt).to(device)
    v = (torch.ones(shape) * 1500.0).to(device)
    source_locations = torch.tensor([[[13, 4]]]).to(device)
    receiver_locations = torch.tensor([[[15, 3]]]).to(device)
    scalar(
        v,
        dx,
        dt,
        source_amplitudes=sa,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=4,
        pml_freq=freq,
        backend="compiled",
    )


def test_importing_deepwave_does_not_change_process_float_behaviour() -> None:
    """A subnormal float32 flush must not depend on whether deepwave is imported.

    Run in a fresh subprocess: the "before import" check must observe the
    interpreter's native (non-flushing) behaviour, which is only guaranteed
    if nothing has already imported deepwave (or anything else that could
    have set process-wide FTZ/DAZ) earlier in this process.
    """
    script = textwrap.dedent("""
        import torch
        before = torch.tensor([1e-40], dtype=torch.float32) * 0.5
        assert before.item() != 0.0, "subnormal flushed to zero before import"
        import deepwave  # noqa: F401
        after = torch.tensor([1e-40], dtype=torch.float32) * 0.5
        assert after.item() != 0.0, "subnormal flushed to zero after import"
        print("OK")
        """)
    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter, an inline script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip() == "OK"


@pytest.mark.skipif(
    not c_backend_available("scalar", device=torch.device("cpu")),
    reason="requires the compiled CPU backend",
)
def test_cpu_propagation_does_not_change_process_float_behaviour() -> None:
    """The per-thread FTZ/DAZ guard must be fully restored after a CPU run.

    A still-nonzero subnormal after the propagation shows the
    ``dw_ftz_daz_enter``/``dw_ftz_daz_exit`` pair around each OpenMP region
    actually restores the calling thread's MXCSR state rather than leaking
    FTZ/DAZ into subsequent computation on the same thread.
    """
    assert _subnormal_survives()

    _run_small_scalar_propagation(torch.device("cpu"))

    assert _subnormal_survives()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_propagation_does_not_change_process_float_behaviour() -> None:
    """CUDA's ``--use_fast_math`` is device-side only and must not affect the host.

    Mirrors the CPU check above for the CUDA backend: the host process's
    float32 behaviour (checked with a plain CPU tensor) must be unaffected
    by running a compiled CUDA propagation.
    """
    assert _subnormal_survives()

    _run_small_scalar_propagation(torch.device("cuda"))

    assert _subnormal_survives()

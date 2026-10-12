"""Timing/measurement primitives used by the per-case subprocess runner.

Nothing in this module spawns subprocesses for *cases* -- that isolation is
done one level up, in `run.py` (one `python -m benchmarks._case_runner`
subprocess per case, so `/proc/self/status`'s VmHWM is a per-case peak and
`OMP_NUM_THREADS` can be pinned per case before the child's `torch` import).
This module is imported *inside* that child process.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import deepwave
from deepwave._version import __version__ as deepwave_version

from . import models

if TYPE_CHECKING:
    from .cases import Case

_REPO_ROOT = Path(__file__).resolve().parent.parent


def median_mad(values: list[float]) -> tuple[float, float]:
    """Median and median absolute deviation (MAD) of a list of floats."""
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return med, mad


def read_vmhwm_bytes() -> int | None:
    """Peak resident set size (high-water mark) of the *current* process.

    Caveat: this is a whole-process, monotonically-increasing high-water
    mark (from /proc/self/status), not a value that resets between reps.
    Because `run.py` spawns one subprocess per case, it is nonetheless a
    reasonably clean per-case number (baseline Python+torch+deepwave
    import overhead aside) -- that is the entire reason for the
    subprocess-per-case design.
    """
    try:
        with Path("/proc/self/status").open(encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    # e.g. "VmHWM:	123456 kB"
                    return int(parts[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_dir(follow_symlinks=False):
                    total += _dir_size_bytes(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
            except OSError:  # noqa: PERF203 - keep scanning readable entries
                continue
    except OSError:
        return total
    return total


class DiskUsagePoller:
    """Polls a directory's total size in a background thread.

    Deepwave's disk `storage_mode` creates and removes a temporary
    subdirectory per call, so the only reliable way to catch its size with
    plain Python (no hooks into the C backend) is to sample repeatedly
    while the propagator runs and keep the maximum observed. This is a
    best-effort measurement: fast, small cases may under-sample. Non-disk
    storage modes report `None` (bytes not measurable this way; see
    README).
    """

    def __init__(self, path: Path, interval: float = 0.01) -> None:
        """Poll `path`'s total size every `interval` seconds once started."""
        self.path = path
        self.interval = interval
        self._max_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._max_bytes = max(self._max_bytes, _dir_size_bytes(self.path))
            time.sleep(self.interval)
        # One last sample in case the directory was still growing.
        self._max_bytes = max(self._max_bytes, _dir_size_bytes(self.path))

    def start(self) -> None:
        """Start the background polling thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop polling and return the maximum directory size observed."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self._max_bytes


def get_cpu_model() -> str:
    """Return the CPU model name from /proc/cpuinfo, or a platform fallback."""
    try:
        with Path("/proc/cpuinfo").open(encoding="ascii", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def get_gpu_models() -> list[str]:
    """Return `torch.cuda.get_device_name` for each visible CUDA device.

    Empty when CUDA is not available, so a CPU-only benchmark run's
    environment block does not claim a GPU it did not use.
    """
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def get_deepwave_commit() -> str:
    """Return `git rev-parse HEAD` for the repo containing this file."""
    git_exe = shutil.which("git") or "git"
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, read-only git command
            [git_exe, "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def get_environment(threads: int) -> dict[str, Any]:
    """Collect machine/software context to embed alongside benchmark results.

    `threads` is the value `run.py` pins as `OMP_NUM_THREADS` (and
    `torch.set_num_threads`) in *each per-case subprocess* -- this call
    itself runs in the parent process, which does not import torch or set
    thread counts, so there is no separate "actual" OMP thread count to
    read back here.
    """
    return {
        "cpu_model": get_cpu_model(),
        "gpu_models": get_gpu_models(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        # `deepwave_commit` is the commit of *this benchmarks/ checkout*, not
        # necessarily the commit of the `deepwave` package actually imported
        # below -- they diverge whenever `deepwave` is installed from a
        # different worktree/build (e.g. benchmarking a `master` install
        # against this branch's harness). `deepwave_package_version`/
        # `deepwave_package_file` record the ground truth of what was
        # actually imported and timed.
        "deepwave_commit": get_deepwave_commit(),
        # The private version module exists in both the current and the
        # released reference package, regardless of package-level exports.
        "deepwave_package_version": deepwave_version,
        "deepwave_package_file": str(Path(deepwave.__file__).resolve()),
        "threads": threads,
    }


def run_case(
    case: Case,
    reps: int = 5,
    warmup: int = 1,
    threads: int = 1,
    storage_scratch_dir: str | None = None,
) -> dict[str, Any]:
    """Run one case: warmup, then `reps` timed repetitions.

    Returns a JSON-serialisable result dict. Any exception during the run
    is caught and reported in the `"error"` field rather than raised, so a
    single bad case does not abort a whole `run.py` invocation (the parent
    process invokes this once per subprocess anyway, so a hard crash there
    is also handled, but in-process errors -- e.g. an invalid case
    configuration -- are cheaper to report this way).
    """
    result: dict[str, Any] = {
        "case": {
            "name": case.name,
            "propagator": case.propagator,
            "ndim": case.ndim,
            "accuracy": case.accuracy,
            "dtype": case.dtype,
            "mode": case.mode,
            "backend": case.backend,
            "tier": case.tier,
            "storage_mode": case.storage_mode,
            "storage_compression": case.storage_compression,
            "n_shots": case.n_shots,
            "size_per_dim": case.size_per_dim,
            "nt": case.nt,
            "device": case.device,
        },
        "threads": threads,
        "error": None,
    }

    disk_poller: DiskUsagePoller | None = None
    try:
        built = models.build_inputs(case)
        result["numerical_setup"] = {
            key: built.kwargs.get(key, default)
            for key, default in (
                ("dt", None),
                ("grid_spacing", None),
                ("max_vel", None),
                ("pml_freq", None),
                ("model_gradient_sampling_interval", 1),
                ("q_n_mechanisms", 1),
                ("q_band", None),
            )
        }
        if case.storage_mode == "disk":
            scratch = Path(storage_scratch_dir) if storage_scratch_dir else _REPO_ROOT
            built.kwargs["storage_path"] = str(scratch)
            disk_poller = DiskUsagePoller(scratch)
            disk_poller.start()

        fwd_times: list[float] = []
        bwd_times: list[float] = []
        total_times: list[float] = []

        is_cuda = case.device == "cuda"

        def one_run() -> None:
            for t in built.grad_tensors:
                if t.grad is not None:
                    t.grad = None
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = built.fn(**built.kwargs)
            if is_cuda:
                torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            info = getattr(out, "info", None)
            if info is not None:
                built.step_ratio = info.step_ratio
                result["numerical_setup"].update(
                    internal_dt=info.internal_dt,
                    model_gradient_stride=info.model_gradient_stride,
                )
            if case.mode == "forward_backward":
                # Both versions expose receiver tensors with this layout.
                # Current outputs carry PML memories opaquely whereas the
                # released tuple includes them. A receiver-only objective
                # avoids changing the differentiated experiment with that
                # representation change.
                receiver_shapes = {
                    (*value.shape[:2], built.nt)
                    for key, value in built.kwargs.items()
                    if "receiver_locations" in key
                }
                out_tensors = [
                    t
                    for t in out
                    if isinstance(t, torch.Tensor) and tuple(t.shape) in receiver_shapes
                ]
                if not out_tensors:
                    raise RuntimeError("benchmark has no requested receiver traces")
                loss = torch.stack([t.sum() for t in out_tensors]).sum()
                loss.backward()
            if is_cuda:
                torch.cuda.synchronize()
            t_bwd = time.perf_counter()
            fwd_times.append(t_fwd - t0)
            if case.mode == "forward_backward":
                bwd_times.append(t_bwd - t_fwd)
            total_times.append(t_bwd - t0)

        for _ in range(warmup):
            one_run()
        fwd_times.clear()
        bwd_times.clear()
        total_times.clear()

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        for _ in range(reps):
            one_run()

        total_median, total_mad = median_mad(total_times)
        fwd_median, fwd_mad = median_mad(fwd_times)
        mcells_per_s = (
            built.interior_cells * built.nt_effective / total_median / 1e6
            if total_median > 0
            else None
        )

        result["timing"] = {
            "total_median_s": total_median,
            "total_mad_s": total_mad,
            "total_reps_s": total_times,
            "forward_median_s": fwd_median,
            "forward_mad_s": fwd_mad,
        }
        result["interior_cells"] = built.interior_cells
        result["nt_effective"] = built.nt_effective
        result["step_ratio"] = built.step_ratio
        result["objective"] = "sum of requested receiver traces"
        result["source_peak_time_s"] = min(
            1.5 / models.SOURCE_FREQ, built.nt * built.kwargs["dt"] / 2
        )
        if case.mode == "forward_backward":
            gradients = [tensor.grad for tensor in built.grad_tensors]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("benchmark left a requested material gradient unset")
            result["gradient_norms"] = [
                float(gradient.norm()) for gradient in gradients
            ]
            if not all(
                math.isfinite(norm) for norm in result["gradient_norms"]
            ) or not any(result["gradient_norms"]):
                raise RuntimeError(
                    "benchmark produced nonfinite or all-zero material gradients"
                )
        result["mcells_per_s"] = mcells_per_s

        if case.mode == "forward_backward":
            bwd_median, bwd_mad = median_mad(bwd_times)
            result["timing"]["backward_median_s"] = bwd_median
            result["timing"]["backward_mad_s"] = bwd_mad
            result["backward_forward_ratio"] = (
                bwd_median / fwd_median if fwd_median > 0 else None
            )
        else:
            result["backward_forward_ratio"] = None

        if disk_poller is not None:
            result["disk_bytes"] = disk_poller.stop()
            disk_poller = None
        else:
            result["disk_bytes"] = "N/A"

        result["cuda_max_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if is_cuda else None
        )

    except Exception as exc:  # noqa: BLE001 - reported, not raised, by design
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if disk_poller is not None:
            disk_poller.stop()

    result["vmhwm_bytes"] = read_vmhwm_bytes()
    return result

#!/usr/bin/env python
"""Reentrant versus non-reentrant segment checkpointing: gradients and cost.

``deepwave.checkpointing.run_segments(checkpoint=True)`` -- the machinery the
propagators' own ``checkpoint_segments=`` argument delegates to -- wraps every
segment but the last in ``torch.utils.checkpoint.checkpoint``. PyTorch
implements that function two ways, selected by its ``use_reentrant`` argument:

* the **reentrant** implementation replays the segment inside a nested
  ``torch.autograd.backward`` call. It re-attaches only the ``Tensor``
  elements at the top level of what the wrapped function returns, refuses
  ``torch.autograd.grad`` (and an ``inputs=`` argument to ``.backward()``),
  and needs at least one input marked ``requires_grad``;
* the **non-reentrant** implementation records the recomputation as ordinary
  autograd nodes. It has none of those restrictions.

This script measures the two things that decide which of them should be the
default: whether they compute the same gradients, and what the non-reentrant
one costs.

``gradients``
    For every propagator, in float64, on each available device, under
    ``storage_mode`` ``"device"``, ``"cpu"`` and ``"disk"``, with one shot and
    with several: run a checkpointed segmented propagation twice, once in each
    mode, and compare every gradient (model gradients, Born-perturbation
    gradients and source-amplitude gradients). Also compares the propagators'
    own ``checkpoint_segments=`` lever, which always checkpoints reentrantly,
    against the non-reentrant driver -- the comparison that says whether
    changing the default would change what those callers get. The largest
    relative difference over all cases is printed, and compared against the
    ``--tolerance`` (1e-12 by default). The two modes recompute the same
    arithmetic in the same order, so the expectation is exact equality; the
    tolerance exists to make "exact" a measurement rather than an assumption.

``cost``
    Wall-clock time and peak memory of a checkpointed forward and backward
    pass, in each mode, for one scalar-family and one elastic-family
    propagator, on each available device, at two segment counts. Each cell
    runs in its own subprocess (so a CPU peak read from ``/proc`` is that
    cell's own high-water mark, not the accumulated one of the whole process)
    and repeats the measurement ``--reps`` times, reporting the median and the
    spread. Peak memory is ``torch.cuda.max_memory_allocated`` on CUDA and the
    growth in resident set size (``VmHWM`` minus the resident size before the
    run) on the CPU.

Timings are only meaningful on an idle machine, and only comparable within one
invocation on one machine.

Examples::

    python benchmarks/checkpoint_reentrancy.py gradients
    python benchmarks/checkpoint_reentrancy.py cost --reps 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch

import deepwave
from deepwave.checkpointing import run_segments
from deepwave.wavelets import ricker

#: Every propagator, in the order the capability table uses.
PROPAGATORS = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)

_DTYPE = torch.float64
_FREQ = 15.0
_DT = 0.001
_DX = 5.0


def _model_values(shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """A homogeneous, physically sensible value for every model name."""
    full = {
        "v": 1500.0,
        "rho": 1000.0,
        "q": 30.0,
        "qp": 30.0,
        "qs": 20.0,
        "lamb": 4.0e9,
        "mu": 2.0e9,
        "buoyancy": 1.0e-3,
        "c11": 9.0e9,
        "c13": 3.0e9,
        "c33": 9.0e9,
        "c55": 2.0e9,
        "c66": 2.0e9,
    }
    return {
        name: torch.full(shape, value, dtype=_DTYPE) for name, value in full.items()
    }


#: The model names that are keyword arguments rather than positional ones (a
#: propagator's ``model_names`` lists both). The Born perturbations, and
#: ``elastic_vti``'s 3-D ``c66``.
_KEYWORD_MODELS = frozenset({"dv", "drho", "dlamb", "dmu", "dbuoyancy", "tilt", "c66"})


def _perturbations(prop: str, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """The Born propagators' perturbation models (small, not zero).

    A zero perturbation is a valid call but a degenerate gradient test: the
    scattered field vanishes, so a difference in how the scattered pass is
    recomputed could not show up.
    """
    values = {
        "scalar_born": {"dv": 10.0},
        "acoustic_born": {"dv": 10.0, "drho": 5.0},
        "elastic_born": {"dlamb": 1.0e7, "dmu": 5.0e6, "dbuoyancy": 1.0e-5},
    }.get(prop, {})
    return {
        name: torch.full(shape, value, dtype=_DTYPE) for name, value in values.items()
    }


def _problem(
    prop: str,
    ndim: int,
    n: int,
    nt: int,
    n_shots: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Models, per-channel source amplitudes and keyword arguments.

    The survey is built from the propagator's own declared channels and
    keyword spellings (:func:`deepwave.spec`), so it needs no per-propagator
    table: one source on the first source channel, and receivers on every
    receiver channel.
    """
    spec = deepwave.spec(prop)
    shape = (n,) * ndim
    values = _model_values(shape)
    models = [
        values[name].clone().to(device)
        for name in spec.model_names(ndim)
        if name not in _KEYWORD_MODELS
    ]
    source = (
        ricker(_FREQ, nt, _DT, min(1.5 / _FREQ, (nt - 1) * _DT / 2))
        .to(_DTYPE)
        .reshape(1, 1, -1)
        .repeat(n_shots, 1, 1)
        .to(device)
    )
    channel = spec.source_channels(ndim)[0]
    middle = [n // 2] * ndim
    source_locations = torch.tensor([[middle]]).repeat(n_shots, 1, 1)
    receiver_locations = torch.tensor([[[min(n // 2 + 2, n - 1)] * ndim]]).repeat(
        n_shots, 1, 1
    )
    kwargs: dict[str, Any] = {
        spec.source_locations_kwarg(channel): source_locations,
        "pml_freq": _FREQ,
        "max_vel": 4000.0,
        "accuracy": 2,
        "pml_width": 3,
    }
    for receiver_channel in spec.receiver_channels(ndim):
        kwargs[spec.receiver_locations_kwarg(receiver_channel)] = receiver_locations
    if spec.has_bg_receivers:
        for bg_channel in spec.bg_receiver_channels(ndim):
            kwargs[spec.bg_receiver_locations_kwarg(bg_channel)] = receiver_locations
    kwargs.update(
        {
            name: tensor.to(device)
            for name, tensor in _perturbations(prop, shape).items()
        }
    )
    if prop == "viscoacoustic":
        kwargs.update(q_n_mechanisms=3, q_band=(_FREQ / 2, 2 * _FREQ))
        # Fixed reference bulk preserves pressure-rate forcing while every
        # physical model remains an independent differentiation input.
        source = source * (1000.0 * 1500.0**2)
    if prop == "elastic_vti" and ndim == 3:
        kwargs["c66"] = values["c66"].clone().to(device)
    return models, {spec.source_amplitudes_kwarg(channel): source}, kwargs


def _leaves(
    models: list[torch.Tensor],
    sources: dict[str, torch.Tensor],
    kwargs: dict[str, Any],
    perturbation_names: tuple[str, ...],
) -> list[torch.Tensor]:
    """Marks the models, perturbations and source amplitudes as leaves.

    Every gradient this script compares is one of these: the model gradients
    (the expensive ones, accumulated through the whole time loop), the Born
    perturbation gradients, and the source-amplitude gradients.
    """
    leaves = []
    for tensor in list(models) + list(sources.values()):
        tensor.requires_grad_(True)
        leaves.append(tensor)
    for name in perturbation_names:
        kwargs[name].requires_grad_(True)
        leaves.append(kwargs[name])
    return leaves


def _loss(out: deepwave.PropagatorOutput) -> torch.Tensor:
    """A scalar every output contributes to, auxiliary state included.

    The absorbing boundary's auxiliary state is part of it deliberately: the
    reentrant implementation re-attaches only the top-level ``Tensor``
    elements of what the checkpointed function returns, so a gradient path
    that runs through the auxiliary state is exactly where the two modes
    could disagree.
    """
    total = torch.zeros((), dtype=_DTYPE, device=out[-1].device)
    tensors = []
    for value in out.as_dict().values():
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, deepwave.AuxiliaryState):
            tensors.extend(value.as_dict().values())
    for tensor in tensors:
        if tensor.is_floating_point():
            total = total + tensor.pow(2).sum()
    return total


def _gradients(
    prop: str,
    ndim: int,
    n: int,
    nt: int,
    n_shots: int,
    n_segments: int,
    device: torch.device,
    storage_mode: str,
    use_reentrant: bool | None,
    storage_path: str | None,
    extra: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    """One checkpointed run's gradients.

    ``use_reentrant=None`` selects the propagator's own
    ``checkpoint_segments=`` argument instead of calling
    :func:`deepwave.checkpointing.run_segments` directly, which is how a user
    reaches this machinery; it always checkpoints reentrantly.
    """
    models, sources, kwargs = _problem(prop, ndim, n, nt, n_shots, device)
    perturbation_names = tuple(_perturbations(prop, (n,) * ndim))
    leaves = _leaves(models, sources, kwargs, perturbation_names)
    kwargs.update(extra or {})
    storage_kwargs: dict[str, Any] = {"storage_mode": storage_mode}
    if storage_mode == "disk":
        storage_kwargs["storage_path"] = storage_path
    if use_reentrant is None:
        out = getattr(deepwave, prop)(
            *models,
            _DX,
            _DT,
            **sources,
            **kwargs,
            **storage_kwargs,
            checkpoint_segments=n_segments,
        )
    else:
        # A differentiable optional model must be threaded as an
        # ``optional_model_args`` entry: a keyword argument is closure-captured
        # by the checkpointed segment and would receive no gradient, and
        # ``run_segments`` refuses one rather than returning a zero.
        optional_models = {
            name: kwargs.pop(name) for name in perturbation_names if name in kwargs
        }
        out = run_segments(
            getattr(deepwave, prop),
            models,
            _DX,
            _DT,
            sources,
            n_segments=n_segments,
            checkpoint=True,
            use_reentrant=use_reentrant,
            optional_model_args=optional_models or None,
            **kwargs,
            **storage_kwargs,
        )
    _loss(out).backward()  # type: ignore[no-untyped-call]
    grads = []
    for leaf in leaves:
        if leaf.grad is None:
            raise RuntimeError("a leaf received no gradient")
        gradient = leaf.grad.detach().clone()
        if not bool(torch.isfinite(gradient).all()) or not bool(
            torch.count_nonzero(gradient)
        ):
            raise RuntimeError(f"{prop}: a leaf received a nonfinite or zero gradient")
        grads.append(gradient)
    return grads


def _relative_difference(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    """The largest relative difference between two gradient tuples.

    Each tensor is compared on its own scale (its largest absolute value), so
    a small gradient is not excused by a large one elsewhere in the tuple.
    """
    if len(left) != len(right):
        raise ValueError("gradient tuples differ in length")
    worst = 0.0
    for a, b in zip(left, right):
        scale = a.abs().max().item()
        difference = (a - b).abs().max().item()
        worst = max(worst, difference if scale == 0 else difference / scale)
    return worst


def _devices(requested: str) -> list[torch.device]:
    """The devices to measure on."""
    if requested != "all":
        return [torch.device(requested)]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def run_gradients(arguments: argparse.Namespace) -> int:
    """Compare the two modes' gradients over every case; 0 if all agree."""
    worst_overall = 0.0
    failures = []
    with tempfile.TemporaryDirectory() as storage_path:
        for prop in arguments.propagators:
            spec = deepwave.spec(prop)
            ndim = 2 if 2 in spec.ndims else spec.ndims[0]
            for device in _devices(arguments.device):
                for storage_mode in arguments.storage_modes:
                    for n_shots in (1, 3):
                        cases: list[tuple[str, dict[str, Any] | None]] = [
                            ("run_segments", None)
                        ]
                        if n_shots == 1:
                            # The model-gradient quadrature stride: a
                            # compiled-backend-only option that changes how
                            # the model gradient is accumulated.
                            cases.append(
                                (
                                    "model_gradient_sampling_interval",
                                    {"model_gradient_sampling_interval": 2},
                                )
                            )
                            cases.append(("checkpoint_segments", None))
                        for label, extra in cases:
                            name = (
                                f"{prop} {ndim}-D {device.type} "
                                f"storage={storage_mode} shots={n_shots} {label}"
                            )
                            try:
                                torch.manual_seed(0)
                                reentrant = _gradients(
                                    prop,
                                    ndim,
                                    arguments.n,
                                    arguments.nt,
                                    n_shots,
                                    arguments.segments,
                                    device,
                                    storage_mode,
                                    None if label == "checkpoint_segments" else True,
                                    storage_path,
                                    extra,
                                )
                                torch.manual_seed(0)
                                non_reentrant = _gradients(
                                    prop,
                                    ndim,
                                    arguments.n,
                                    arguments.nt,
                                    n_shots,
                                    arguments.segments,
                                    device,
                                    storage_mode,
                                    False,
                                    storage_path,
                                    extra,
                                )
                            except (RuntimeError, NotImplementedError) as error:
                                # Explicit option exclusions are reported so
                                # unsupported cases cannot look like passes.
                                if "is not supported by" not in str(error):
                                    raise
                                print(f"  {'--':>9}  n/a   {name}", flush=True)
                                continue
                            difference = _relative_difference(reentrant, non_reentrant)
                            worst_overall = max(worst_overall, difference)
                            status = (
                                "ok" if difference <= arguments.tolerance else "OVER"
                            )
                            minimum_norm = min(g.norm().item() for g in reentrant)
                            print(
                                f"  {difference:9.2e}  {status:4}  {name} "
                                f"min_gradient_norm={minimum_norm:.3e}",
                                flush=True,
                            )
                            if difference > arguments.tolerance:
                                failures.append((name, difference))
    print(
        f"\nlargest relative gradient difference: {worst_overall:.3e} "
        f"(tolerance {arguments.tolerance:.1e})"
    )
    for name, difference in failures:
        print(f"  OVER TOLERANCE {difference:.3e}  {name}")
    return 0 if not failures else 1


def _peak_cpu_bytes() -> int:
    """This process's peak resident set size, in bytes."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM not found in /proc/self/status")


def _current_cpu_bytes() -> int:
    """This process's current resident set size, in bytes."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS not found in /proc/self/status")


def _timed(call: Callable[[], None], device: torch.device) -> float:
    """Wall-clock seconds of one call."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        return time.perf_counter() - start
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def _peak(call: Callable[[], None], device: torch.device) -> int:
    """Peak memory of one call, in bytes.

    On CUDA this is ``torch.cuda.max_memory_allocated`` around the call. On
    the CPU there is no such counter, so it is the growth in this process's
    peak resident set size -- which is why a CPU memory measurement gets a
    process of its own, with no earlier run in it: ``VmHWM`` is a high-water
    mark that never falls, so a second run in the same process would be
    measured against the first one's peak rather than its own.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    before = _current_cpu_bytes()
    call()
    return max(0, _peak_cpu_bytes() - before)


def run_cell(arguments: argparse.Namespace) -> int:
    """Measure one (propagator, device, segments, mode) cell in this process.

    Prints one JSON object. ``cost`` runs this in a subprocess per cell, and
    one further subprocess per memory repetition (see :func:`_peak`).
    ``--measure time`` runs one untimed warmup and then ``--reps`` timed
    repetitions; ``--measure memory`` runs exactly once.
    """
    device = torch.device(arguments.device)
    use_reentrant = not arguments.non_reentrant
    times: list[float] = []
    peaks: list[int] = []

    def run(n: int, nt: int) -> None:
        models, sources, kwargs = _problem(
            arguments.propagator,
            arguments.ndim,
            n,
            nt,
            arguments.shots,
            device,
        )
        _leaves(models, sources, kwargs, ())
        out = run_segments(
            getattr(deepwave, arguments.propagator),
            models,
            _DX,
            _DT,
            sources,
            n_segments=arguments.segments,
            checkpoint=True,
            use_reentrant=use_reentrant,
            storage_mode="device",
            **kwargs,
        )
        _loss(out).backward()  # type: ignore[no-untyped-call]

    def once() -> None:
        run(arguments.n, arguments.nt)

    if arguments.measure == "memory":
        # A tiny run first, so that the library's one-time allocations (BLAS
        # scratch, kernel loading) are already resident and counted in the
        # baseline rather than in this cell's growth. It is small enough that
        # its own peak cannot be the high-water mark the measurement reads.
        run(16, 20)
        peaks.append(_peak(once, device))
    else:
        for repetition in range(arguments.reps + 1):
            elapsed = _timed(once, device)
            if repetition == 0:
                # Discard: the first run pays one-time allocator growth and,
                # on CUDA, context and kernel-load setup.
                continue
            times.append(elapsed)
    print(
        json.dumps(
            {
                "propagator": arguments.propagator,
                "device": device.type,
                "segments": arguments.segments,
                "reentrant": use_reentrant,
                "times": times,
                "peaks": peaks,
            }
        )
    )
    return 0


def _spread(values: list[float]) -> tuple[float, float, float]:
    """Median, minimum and maximum."""
    return statistics.median(values), min(values), max(values)


def _run_cell_subprocess(
    propagator: str,
    ndim: int,
    device: torch.device,
    segments: int,
    non_reentrant: bool,
    measure: str,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Run one cell in a subprocess and return its JSON result."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cell",
        "--propagator",
        propagator,
        "--ndim",
        str(ndim),
        "--device",
        device.type,
        "--segments",
        str(segments),
        "--reps",
        str(arguments.reps),
        "--n",
        str(arguments.n),
        "--nt",
        str(arguments.nt),
        "--shots",
        str(arguments.shots),
        "--measure",
        measure,
    ]
    if non_reentrant:
        command.append("--non-reentrant")
    completed = subprocess.run(  # noqa: S603 - fixed argv: this interpreter, our own module
        command,
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    result: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    return result


def run_cost(arguments: argparse.Namespace) -> int:
    """Time and peak memory of both modes, per propagator/device/segments."""
    header = (
        f"{'case':38} {'time reentrant':>22} {'time non-reentrant':>22} "
        f"{'dt %':>7} {'peak reentrant':>24} {'peak non-reentrant':>24} "
        f"{'dmem %':>8}"
    )
    print(header)
    print("-" * len(header))
    for propagator in arguments.propagators:
        ndim = 2
        for device in _devices(arguments.device):
            for segments in arguments.segment_counts:
                times = {}
                peaks = {}
                for mode in (False, True):
                    times[mode] = _run_cell_subprocess(
                        propagator, ndim, device, segments, mode, "time", arguments
                    )["times"]
                    # One process per memory repetition: see _peak.
                    peaks[mode] = [
                        _run_cell_subprocess(
                            propagator,
                            ndim,
                            device,
                            segments,
                            mode,
                            "memory",
                            arguments,
                        )["peaks"][0]
                        for _ in range(arguments.reps)
                    ]
                time_r, time_r_lo, time_r_hi = _spread(times[False])
                time_n, time_n_lo, time_n_hi = _spread(times[True])
                peak_r, peak_r_lo, peak_r_hi = _spread(peaks[False])
                peak_n, peak_n_lo, peak_n_hi = _spread(peaks[True])
                case = f"{propagator} {ndim}-D {device.type} segments={segments}"
                print(
                    f"{case:38} "
                    f"{time_r:7.3f}s[{time_r_lo:.3f},{time_r_hi:.3f}] "
                    f"{time_n:7.3f}s[{time_n_lo:.3f},{time_n_hi:.3f}] "
                    f"{100 * (time_n / time_r - 1):+6.1f}% "
                    f"{peak_r / 2**20:8.1f}[{peak_r_lo / 2**20:.1f},"
                    f"{peak_r_hi / 2**20:.1f}]MiB "
                    f"{peak_n / 2**20:8.1f}[{peak_n_lo / 2**20:.1f},"
                    f"{peak_n_hi / 2**20:.1f}]MiB "
                    f"{100 * (peak_n / peak_r - 1):+7.1f}%",
                    flush=True,
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    gradients = subparsers.add_parser(
        "gradients", help="compare the two modes' gradients"
    )
    gradients.add_argument("--propagators", nargs="+", default=list(PROPAGATORS))
    gradients.add_argument("--device", default="all")
    gradients.add_argument(
        "--storage-modes",
        nargs="+",
        dest="storage_modes",
        default=["device", "cpu", "disk"],
    )
    gradients.add_argument("--segments", type=int, default=3)
    gradients.add_argument("--n", type=int, default=16)
    gradients.add_argument("--nt", type=int, default=24)
    gradients.add_argument("--tolerance", type=float, default=1e-12)
    gradients.set_defaults(function=run_gradients)

    cost = subparsers.add_parser("cost", help="time and peak memory of both modes")
    cost.add_argument("--propagators", nargs="+", default=["scalar", "elastic"])
    cost.add_argument("--device", default="all")
    cost.add_argument(
        "--segment-counts", nargs="+", type=int, dest="segment_counts", default=[2, 8]
    )
    cost.add_argument("--reps", type=int, default=5)
    cost.add_argument("--n", type=int, default=120)
    cost.add_argument("--nt", type=int, default=800)
    cost.add_argument("--shots", type=int, default=2)
    cost.set_defaults(function=run_cost)

    cell = subparsers.add_parser("cell", help="one cost cell, in this process")
    cell.add_argument("--propagator", required=True)
    cell.add_argument("--ndim", type=int, default=2)
    cell.add_argument("--device", default="cpu")
    cell.add_argument("--segments", type=int, default=2)
    cell.add_argument("--reps", type=int, default=5)
    cell.add_argument("--n", type=int, default=120)
    cell.add_argument("--nt", type=int, default=800)
    cell.add_argument("--shots", type=int, default=2)
    cell.add_argument("--non-reentrant", action="store_true", dest="non_reentrant")
    cell.add_argument("--measure", choices=["time", "memory"], default="time")
    cell.set_defaults(function=run_cell)

    arguments = parser.parse_args(argv)
    result: int = arguments.function(arguments)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

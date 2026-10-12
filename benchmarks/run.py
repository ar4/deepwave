#!/usr/bin/env python
"""CLI entry point for the deepwave benchmark suite.

Examples:
    python benchmarks/run.py --tier micro --reps 3 --label mylabel
    python benchmarks/run.py --tier meso --propagator scalar --propagator elastic
    python benchmarks/run.py --all --filter acc8
    python benchmarks/run.py --tier macro          # opt-in, heavy: storage_mode sweep
    python benchmarks/run.py --thread-scaling
    python benchmarks/run.py --storage-modes
    python benchmarks/run.py --backend-compare
    python benchmarks/run.py --compare results/a.json results/b.json

Every case is run in its own subprocess (see `_case_runner.py`) so that
`OMP_NUM_THREADS` can be pinned per case *before* that process's `torch`
import, and so `/proc/self/status` VmHWM reflects one case, not a
running maximum across the whole suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cases import (
    DEVICES,
    PROPAGATORS,
    STORAGE_MODES,
    TIERS,
    Case,
    get_cases,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


@functools.lru_cache(maxsize=1)
def _systemd_run_usable() -> str | None:
    """Absolute path to ``systemd-run`` if ``--user --scope`` can cap here, else None.

    Probed once (cached) by launching a trivial capped unit. Returns None when
    there is no usable user systemd session (some CI images / containers); the
    caller then runs cases uncapped -- the pre-existing behaviour. The resolved
    absolute path is reused for the real per-case wrapper.
    """
    path = shutil.which("systemd-run")
    if path is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, resolved executable
            [
                path,
                "--user",
                "--scope",
                "-q",
                "-p",
                "MemoryMax=64M",
                "-p",
                "MemorySwapMax=0",
                "true",
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return path if proc.returncode == 0 else None


def _mem_total_bytes() -> int | None:
    """Host RAM in bytes from /proc/meminfo, or None if it can't be read."""
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def resolve_mem_cap(spec: str | None) -> str | None:
    """Resolve a ``--mem-cap`` spec to a systemd ``MemoryMax`` value, or None.

    - ``"off"``/``"none"``/``"0"`` -> None (no cap; original behaviour).
    - ``None``/``"auto"``          -> 85% of host RAM (leaves headroom for the
      OS and this parent process), or None if RAM size is unknown.
    - anything else                -> passed through verbatim (a systemd size
      string such as ``"18G"``/``"16000M"`` or a byte count).

    Returns None -- and warns once on stderr -- if a cap was requested but
    ``systemd-run --user`` is not usable on this host, so runs degrade to the
    previous uncapped behaviour rather than erroring on every case.
    """
    if spec in ("off", "none", "0"):
        return None
    if spec is None or spec == "auto":
        total = _mem_total_bytes()
        cap: str | None = str(int(total * 0.85)) if total else None
    else:
        cap = spec
    if cap is not None and _systemd_run_usable() is None:
        print(
            "WARNING: --mem-cap requested but `systemd-run --user` is not "
            "usable here; running per-case subprocesses UNCAPPED. A case that "
            "exceeds host RAM may be OOM-killed.",
            file=sys.stderr,
        )
        return None
    return cap


def run_one_case(
    case: Case,
    reps: int,
    warmup: int,
    threads: int,
    timeout: float = 120.0,
    mem_cap: str | None = None,
) -> dict[str, Any]:
    """Run a single case in a fresh subprocess and return its result dict.

    ``mem_cap`` (a resolved systemd ``MemoryMax`` value from
    ``resolve_mem_cap``) wraps the subprocess in ``systemd-run --user --scope``
    so a case that would exceed host RAM is killed inside its own cgroup and
    recorded as a per-case error -- it never triggers a global OOM that would
    take down this process (or, under systemd's OOMPolicy=stop, the whole
    tmux/session scope). ``None`` runs uncapped.
    """
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    storage_scratch_dir: str | None = None
    if case.storage_mode == "disk":
        storage_scratch_dir = tempfile.mkdtemp(prefix="deepwave_bench_disk_")

    payload = {
        "case": dataclasses.asdict(case),
        "reps": reps,
        "warmup": warmup,
        "threads": threads,
        "storage_scratch_dir": storage_scratch_dir,
    }

    argv = [sys.executable, "-m", "benchmarks._case_runner"]
    systemd_run = _systemd_run_usable() if mem_cap else None
    if mem_cap and systemd_run is not None:
        argv = [
            systemd_run,
            "--user",
            "--scope",
            "-q",
            "-p",
            f"MemoryMax={mem_cap}",
            "-p",
            "MemorySwapMax=0",
            *argv,
        ]

    base_result = {"case": {"name": case.name, **dataclasses.asdict(case)}}
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv: this interpreter, our own module
            argv,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {**base_result, "error": f"timeout after {timeout}s"}
    finally:
        if storage_scratch_dir:
            shutil.rmtree(storage_scratch_dir, ignore_errors=True)

    if proc.returncode != 0 or not proc.stdout.strip():
        stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
        # A memory-capped subprocess killed by the cgroup OOM killer exits 137
        # (128 + SIGKILL); surface that as an OOM rather than an opaque code.
        hint = ""
        if mem_cap and proc.returncode in (137, -9):
            hint = f" (out of memory: exceeded --mem-cap {mem_cap})"
        return {
            **base_result,
            "error": f"subprocess exited {proc.returncode}{hint}: {stderr_tail}",
        }
    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        return dict(json.loads(last_line))
    except (json.JSONDecodeError, IndexError) as exc:
        return {
            **base_result,
            "error": (
                f"could not parse subprocess output ({exc}): {proc.stdout[:500]!r}"
            ),
        }


def _fmt(value: Any, spec: str) -> str:
    """Format a numeric/str/None result value for the console table."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return str(value)


def format_table(results: list[dict[str, Any]]) -> str:
    """Render an aligned plain-text table of per-case results."""
    headers = [
        "case",
        "median_s",
        "mad_s",
        "mcells/s",
        "bwd/fwd",
        "vmhwm_mb",
        "cuda_mb",
        "disk_mb",
        "status",
    ]
    rows = []
    for r in results:
        name = r.get("case", {}).get("name", "?")
        if r.get("error"):
            rows.append(
                [name, "-", "-", "-", "-", "-", "-", "-", f"ERROR: {r['error'][:60]}"]
            )
            continue
        timing = r.get("timing", {})
        vmhwm = r.get("vmhwm_bytes")
        vmhwm_mb = vmhwm / 1e6 if isinstance(vmhwm, (int, float)) else None
        cuda_mem = r.get("cuda_max_memory_allocated_bytes")
        cuda_mb = cuda_mem / 1e6 if isinstance(cuda_mem, (int, float)) else None
        disk = r.get("disk_bytes")
        disk_mb = disk / 1e6 if isinstance(disk, (int, float)) else None
        rows.append(
            [
                name,
                _fmt(timing.get("total_median_s"), ".4f"),
                _fmt(timing.get("total_mad_s"), ".4f"),
                _fmt(r.get("mcells_per_s"), ".2f"),
                _fmt(r.get("backward_forward_ratio"), ".2f"),
                _fmt(vmhwm_mb, ".1f"),
                _fmt(cuda_mb, ".1f"),
                _fmt(disk_mb, ".1f") if disk != "N/A" else "N/A",
                "ok",
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(headers))
    ]
    lines = []
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    lines.extend(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    )
    return "\n".join(lines)


def get_environment(threads: int) -> dict[str, Any]:
    """Collect CPU/platform/torch/commit info to embed in a result file.

    Imported lazily (inside the function) so that `--compare` (which needs
    neither torch nor deepwave) stays fast and works even in environments
    without them.
    """
    from benchmarks.harness import get_environment as _get_environment

    return _get_environment(threads)


def write_results(
    path: Path,
    label: str,
    tier: str,
    reps: int,
    warmup: int,
    threads: int,
    results: list[dict[str, Any]],
) -> None:
    """Write one suite/report run (environment + all case results) as JSON."""
    payload = {
        "label": label,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tier": tier,
        "reps": reps,
        "warmup": warmup,
        "threads": threads,
        "environment": get_environment(threads),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _default_label() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cmd_run(args: argparse.Namespace) -> int:
    """Run the normal case suite (tier/all/propagator/filter-selected)."""
    cases = get_cases(
        tier=args.tier,
        all_cases=args.all,
        propagators=args.propagator,
        name_filter=args.filter,
        device=args.device,
    )
    if not cases:
        print("No cases matched the given filters.", file=sys.stderr)
        return 1

    print(
        f"Running {len(cases)} case(s), tier={args.tier}, reps={args.reps}, "
        f"warmup={args.warmup}, threads={args.threads}"
        + (" [SMOKE-ONLY numbers, not for regression comparison]" if args.smoke else "")
    )
    mem_cap = resolve_mem_cap(args.mem_cap)
    results = []
    for case in cases:
        result = run_one_case(
            case,
            reps=args.reps,
            warmup=args.warmup,
            threads=args.threads,
            timeout=args.timeout,
            mem_cap=mem_cap,
        )
        results.append(result)
        status = "ERROR" if result.get("error") else "ok"
        print(f"  [{status}] {case.name}")

    print()
    print(format_table(results))

    label = args.label or _default_label()
    out_path = Path(args.out_dir) / f"{label}.json"
    write_results(
        out_path, label, args.tier, args.reps, args.warmup, args.threads, results
    )
    print(f"\nResults written to {out_path}")
    if args.smoke:
        print(
            "NOTE: this was a smoke-only run (few reps / tiny sizes); do not "
            "use these numbers as a regression baseline."
        )
    return 0


def cmd_thread_scaling(args: argparse.Namespace) -> int:
    """Report wall time/speedup of one n_shots=1 case across thread counts."""
    case = Case(
        propagator="scalar",
        ndim=2,
        accuracy=4,
        dtype="float32",
        mode="forward",
        tier="meso",
        n_shots=1,
    )
    thread_counts = args.threads_list or [1, 2, 4]
    print(
        f"Thread scaling report: {case.name}, n_shots=1, threads={thread_counts}\n"
        "(measures per-step spatial-OpenMP scaling for a single shot)"
    )
    mem_cap = resolve_mem_cap(args.mem_cap)
    results = []
    for threads in thread_counts:
        result = run_one_case(
            case,
            reps=args.reps,
            warmup=args.warmup,
            threads=threads,
            timeout=args.timeout,
            mem_cap=mem_cap,
        )
        result.setdefault("case", {})["name"] = f"{case.name}_threads{threads}"
        results.append(result)
        status = "ERROR" if result.get("error") else "ok"
        print(f"  [{status}] threads={threads}")

    print()
    print(format_table(results))
    base_time = None
    for threads, r in zip(thread_counts, results):
        if r.get("error"):
            continue
        t = r["timing"]["total_median_s"]
        if base_time is None:
            base_time = t
        speedup = base_time / t if t > 0 else None
        speedup_str = _fmt(speedup, ".2f")
        print(
            f"  threads={threads}: speedup vs threads={thread_counts[0]} "
            f"= {speedup_str}x"
        )

    label = args.label or _default_label()
    out_path = Path(args.out_dir) / f"{label}_thread_scaling.json"
    write_results(
        out_path, label, "meso", args.reps, args.warmup, thread_counts[-1], results
    )
    print(f"\nResults written to {out_path}")
    return 0


def cmd_shots_sweep(args: argparse.Namespace) -> int:
    """Report the many-shots/small-model sweep (scalar 2-D 200x500 grid).

    The spatial-OpenMP kernels give each thread little work per time step
    on a small grid; many concurrent shots are then the natural parallelism
    source. This sweep measures that regime (shots 1/20/40, forward and
    forward_backward) -- the square tiers never exercise it. Run with
    `--threads <all host cores>` to answer "does the current
    parallelisation strategy hold up when shots could occupy every
    thread?".
    """
    from benchmarks.cases import shots_sweep_cases

    cases = shots_sweep_cases()
    if args.device != "cpu":
        cases = [dataclasses.replace(c, device=args.device) for c in cases]
    print(
        f"Many-shots/small-model sweep: {len(cases)} case(s), "
        f"reps={args.reps}, warmup={args.warmup}, threads={args.threads}"
    )
    mem_cap = resolve_mem_cap(args.mem_cap)
    results = []
    for case in cases:
        result = run_one_case(
            case,
            reps=args.reps,
            warmup=args.warmup,
            threads=args.threads,
            timeout=args.timeout,
            mem_cap=mem_cap,
        )
        results.append(result)
        status = "ERROR" if result.get("error") else "ok"
        print(f"  [{status}] {case.name}")

    print()
    print(format_table(results))

    label = args.label or _default_label()
    out_path = Path(args.out_dir) / f"{label}.json"
    write_results(
        out_path, label, "shots-sweep", args.reps, args.warmup, args.threads, results
    )
    print(f"\nResults written to {out_path}")
    return 0


def cmd_storage_modes(args: argparse.Namespace) -> int:
    """Report time/peak-RSS(/disk-bytes) of one case across storage_modes."""
    base_case = Case(
        propagator="scalar",
        ndim=3,
        accuracy=4,
        dtype="float32",
        mode="forward_backward",
        tier="micro",
        n_shots=1,
    )
    print(
        f"Storage-mode report: {base_case.propagator} {base_case.ndim}-D, "
        f"tier={base_case.tier} (3-D small), modes={list(STORAGE_MODES)}\n"
        "(checkpointing memory/time tradeoff curve)"
    )
    mem_cap = resolve_mem_cap(args.mem_cap)
    results = []
    for storage_mode in STORAGE_MODES:
        case = dataclasses.replace(base_case, storage_mode=storage_mode)
        result = run_one_case(
            case,
            reps=args.reps,
            warmup=args.warmup,
            threads=args.threads,
            timeout=args.timeout,
            mem_cap=mem_cap,
        )
        results.append(result)
        status = "ERROR" if result.get("error") else "ok"
        print(f"  [{status}] storage_mode={storage_mode}")

    print()
    print(format_table(results))

    label = args.label or _default_label()
    out_path = Path(args.out_dir) / f"{label}_storage_modes.json"
    write_results(
        out_path, label, base_case.tier, args.reps, args.warmup, args.threads, results
    )
    print(f"\nResults written to {out_path}")
    return 0


def _backend_compare_cases() -> list[tuple[str, Case]]:
    """The four base cases for the compiled-vs-``backend="torch_compile"`` comparison.

    Grid sizes match the three configurations an earlier, undocumented
    compiled-vs-``torch.compile`` measurement used (no harness or results
    file for it survives): CPU 512^2 forward, CUDA 512^2
    forward_backward, CUDA 1024^2 forward -- plus one new 3-D case (not in
    the original three). That record specifies grid size only,
    not accuracy/dtype/nt, so two assumptions are documented here rather
    than left implicit: ``accuracy=4``/``dtype="float32"`` are deepwave's own
    function defaults; ``nt=1000`` for the 2-D cases reuses this suite's
    already-established meso-tier 2-D value (``TIER_SIZES["meso"][2]``)
    rather than inventing a new number -- the 512^2 case needs no override at
    all since 512/1000 already *is* the meso-tier 2-D size. The 3-D case
    uses this suite's micro-tier 3-D size (32^3, nt=100) and forward-only
    mode: deliberately modest and low-risk, since ``backend="torch_compile"``
    (torch.compile/Inductor) has triggered CUDA-allocator segfaults and
    elevated RAM use under concurrent xdist workers in this suite's own
    test runs -- less relevant here since ``run_one_case`` isolates each
    case in its own subprocess, but a reason to keep problem sizes modest.
    """
    return [
        (
            "cpu_512_forward",
            Case(
                propagator="scalar",
                ndim=2,
                accuracy=4,
                dtype="float32",
                mode="forward",
                tier="meso",
                device="cpu",
            ),
        ),
        (
            "cuda_512_forward_backward",
            Case(
                propagator="scalar",
                ndim=2,
                accuracy=4,
                dtype="float32",
                mode="forward_backward",
                tier="meso",
                device="cuda",
            ),
        ),
        (
            "cuda_1024_forward",
            Case(
                propagator="scalar",
                ndim=2,
                accuracy=4,
                dtype="float32",
                mode="forward",
                tier="meso",
                device="cuda",
                shape=(1024, 1024),
                nt_override=1000,
            ),
        ),
        (
            "cuda_3d_32_forward",
            Case(
                propagator="scalar",
                ndim=3,
                accuracy=4,
                dtype="float32",
                mode="forward",
                tier="micro",
                device="cuda",
            ),
        ),
    ]


def cmd_backend_compare(args: argparse.Namespace) -> int:
    """Re-measure compiled-backend vs ``backend="torch_compile"`` speed.

    Re-runs the three historically recorded configs plus one new 3-D case
    (see ``_backend_compare_cases``) at ``backend="compiled"`` (the default
    hand-written C/CUDA backend) and ``backend="torch_compile"`` (PyTorch's
    ``torch.compile`` path), and reports the speed ratio between them.

    Each (case, backend) pair runs in its own ``run_one_case`` subprocess --
    the harness's existing per-case isolation, important here for a second
    reason beyond the usual OMP-thread-pinning/clean-VmHWM rationale: it
    contains any Inductor crash to that one subprocess and prevents
    ``backend="torch_compile"``'s process-lifetime compile cache from
    accumulating across cases.
    """
    base_cases = _backend_compare_cases()
    backends = ["compiled", "torch_compile"]
    print(
        f"Backend comparison report: {len(base_cases)} case(s) x "
        f"{len(backends)} backend(s), reps={args.reps}, warmup={args.warmup}"
    )
    mem_cap = resolve_mem_cap(args.mem_cap)
    results = []
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for case_label, base_case in base_cases:
        by_case[case_label] = {}
        for backend in backends:
            case = dataclasses.replace(base_case, backend=backend)
            result = run_one_case(
                case,
                reps=args.reps,
                warmup=args.warmup,
                threads=args.threads,
                timeout=args.timeout,
                mem_cap=mem_cap,
            )
            result.setdefault("case", {})["name"] = f"{case_label}_{backend}"
            results.append(result)
            by_case[case_label][backend] = result
            status = "ERROR" if result.get("error") else "ok"
            print(f"  [{status}] {case_label} backend={backend}")

    print()
    print(format_table(results))
    print()
    print(f"{'case':30}  {'compiled_s':>12}  {'torch_compile_s':>16}  {'ratio':>8}")
    for case_label, per_backend in by_case.items():
        compiled_r = per_backend["compiled"]
        compile_r = per_backend["torch_compile"]
        compiled_t = (
            compiled_r.get("timing", {}).get("total_median_s")
            if not compiled_r.get("error")
            else None
        )
        compile_t = (
            compile_r.get("timing", {}).get("total_median_s")
            if not compile_r.get("error")
            else None
        )
        ratio = (
            compile_t / compiled_t
            if isinstance(compiled_t, (int, float))
            and isinstance(compile_t, (int, float))
            and compiled_t > 0
            else None
        )
        print(
            f"{case_label:30}  {_fmt(compiled_t, '.4f'):>12}  "
            f"{_fmt(compile_t, '.4f'):>16}  {_fmt(ratio, '.2f'):>8}"
        )

    label = args.label or _default_label()
    out_path = Path(args.out_dir) / f"{label}_backend_compare.json"
    write_results(
        out_path,
        label,
        "backend-compare",
        args.reps,
        args.warmup,
        args.threads,
        results,
    )
    print(f"\nResults written to {out_path}")
    return 0


def _case_result_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a result-file's `"results"` list by case name."""
    return {
        r.get("case", {}).get("name", f"?{i}"): r
        for i, r in enumerate(payload.get("results", []))
    }


def _pct_delta(a: float, b: float) -> float:
    """Fractional change from `a` (baseline) to `b` (candidate)."""
    return (b - a) / a if a else float("nan")


def cmd_compare(args: argparse.Namespace) -> int:
    """Print per-case wall-time AND peak-memory deltas between two result files.

    Exits with status 2 if any shared case regressed (median wall time or
    peak memory increased) by more than its respective threshold, for use
    as a CI gate. Peak memory is per-device: CPU cases use `vmhwm_bytes`
    (host RSS, from `/proc/self/status` VmHWM); GPU cases use
    `cuda_max_memory_allocated_bytes` (`torch.cuda.max_memory_allocated()`),
    since host RSS does not track the CUDA caching allocator at all.
    """
    path_a, path_b = args.compare
    payload_a = json.loads(Path(path_a).read_text())
    payload_b = json.loads(Path(path_b).read_text())
    index_a = _case_result_index(payload_a)
    index_b = _case_result_index(payload_b)

    common_names = sorted(set(index_a) & set(index_b))
    only_a = sorted(set(index_a) - set(index_b))
    only_b = sorted(set(index_b) - set(index_a))

    print(f"Comparing {path_a} (baseline) -> {path_b} (candidate)")
    if only_a:
        print(f"  (only in baseline, skipped: {only_a})")
    if only_b:
        print(f"  (only in candidate, skipped: {only_b})")
    print()

    headers = [
        "case",
        "baseline_s",
        "candidate_s",
        "time_delta_%",
        "baseline_mb",
        "candidate_mb",
        "mem_delta_%",
        "flag",
    ]
    rows = []
    any_regression = False
    for name in common_names:
        ra, rb = index_a[name], index_b[name]
        if ra.get("error") or rb.get("error"):
            rows.append([name, "-", "-", "-", "-", "-", "-", "ERROR in one run"])
            continue
        ta = ra["timing"]["total_median_s"]
        tb = rb["timing"]["total_median_s"]
        time_delta = _pct_delta(ta, tb)

        is_cuda = ra.get("case", {}).get("device") == "cuda"
        mem_key = "cuda_max_memory_allocated_bytes" if is_cuda else "vmhwm_bytes"
        mem_a_raw = ra.get(mem_key)
        mem_b_raw = rb.get(mem_key)
        has_mem = isinstance(mem_a_raw, (int, float)) and isinstance(
            mem_b_raw, (int, float)
        )
        mem_a = float(mem_a_raw) if has_mem else 0.0
        mem_b = float(mem_b_raw) if has_mem else 0.0
        mem_delta = _pct_delta(mem_a, mem_b) if has_mem else None

        flags = []
        if time_delta > args.regression_threshold:
            flags.append(f"TIME REGRESSION (>{args.regression_threshold:.0%})")
            any_regression = True
        elif time_delta < -args.regression_threshold:
            flags.append("time improvement")
        if mem_delta is not None:
            if mem_delta > args.memory_regression_threshold:
                flags.append(
                    f"MEMORY REGRESSION (>{args.memory_regression_threshold:.0%})"
                )
                any_regression = True
            elif mem_delta < -args.memory_regression_threshold:
                flags.append("memory improvement")

        rows.append(
            [
                name,
                f"{ta:.4f}",
                f"{tb:.4f}",
                f"{time_delta * 100:+.1f}",
                f"{mem_a / 1e6:.1f}" if has_mem else "-",
                f"{mem_b / 1e6:.1f}" if has_mem else "-",
                f"{mem_delta * 100:+.1f}" if mem_delta is not None else "-",
                ", ".join(flags),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(headers))
    ]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

    if any_regression:
        print(
            f"\nRegression(s) detected above the "
            f"{args.regression_threshold:.0%} time / "
            f"{args.memory_regression_threshold:.0%} memory threshold."
        )
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the `run.py` CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Deepwave performance benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier", choices=TIERS, default="micro", help="Problem size tier."
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="Timed repetitions per case."
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed warmup runs per case."
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Result file label (default: timestamp).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use the full axis cross product instead of the default subset.",
    )
    parser.add_argument(
        "--propagator",
        action="append",
        choices=PROPAGATORS,
        help="Restrict to this propagator (repeatable).",
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Substring filter on case name."
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help="Device to place model/source/receiver tensors on (default: cpu). "
        "'cuda' requires a CUDA-capable torch build and a visible GPU.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="OMP_NUM_THREADS / torch thread count pinned per case.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory for result JSON files.",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Per-case subprocess timeout (s)."
    )
    parser.add_argument(
        "--mem-cap",
        default="auto",
        metavar="SIZE",
        help=(
            "Cap each per-case subprocess's memory via `systemd-run --user` "
            "MemoryMax so an oversized case fails as a recorded per-case error "
            "instead of OOM-killing the box (and, under OOMPolicy=stop, the "
            "whole session). 'auto' (default) = 85%% of host RAM; a systemd "
            "size like '18G'; or 'off' to disable. Ignored, with a warning, "
            "where systemd-run --user is unavailable."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Mark this run's console output as smoke-only (no behaviour "
            "change beyond a warning banner)."
        ),
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE.json", "CANDIDATE.json"),
        default=None,
        help="Compare two result files; print per-case deltas (exit 2 on regression).",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.10,
        help=(
            "Fractional median-time increase considered a regression in "
            "--compare (default 0.10 = 10%%)."
        ),
    )
    parser.add_argument(
        "--memory-regression-threshold",
        type=float,
        default=0.10,
        help=(
            "Fractional peak-memory increase considered a regression in "
            "--compare (default 0.10 = 10%%) -- host RSS (vmhwm_bytes) for "
            "CPU cases, CUDA allocator peak (cuda_max_memory_allocated_"
            "bytes) for GPU cases."
        ),
    )

    parser.add_argument(
        "--thread-scaling",
        action="store_true",
        help=(
            "Run the single-shot thread-scaling report (scalar 2-D meso, "
            "1/2/4 threads) instead of the normal suite."
        ),
    )
    parser.add_argument(
        "--shots-sweep",
        action="store_true",
        help=(
            "Run the many-shots/small-model report (scalar 2-D 200x500, "
            "shots 1/20/40, forward + forward_backward) instead of the "
            "normal suite. Pair with --threads <all host cores>."
        ),
    )
    parser.add_argument(
        "--storage-modes",
        action="store_true",
        help=(
            "Run the storage_mode memory/time report (3-D small) instead "
            "of the normal suite."
        ),
    )
    parser.add_argument(
        "--backend-compare",
        action="store_true",
        help=(
            "Run the compiled-vs-backend='torch_compile' comparison report "
            "(three historically recorded 2-D configs plus one 3-D case, "
            "each at backend='compiled' and backend='torch_compile') instead of "
            "the normal suite."
        ),
    )
    parser.add_argument(
        "--threads-list",
        type=int,
        nargs="+",
        default=None,
        help="Thread counts to use for --thread-scaling (default: 1 2 4).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args and dispatch to the requested command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.compare:
        return cmd_compare(args)
    if args.thread_scaling:
        return cmd_thread_scaling(args)
    if args.shots_sweep:
        return cmd_shots_sweep(args)
    if args.storage_modes:
        return cmd_storage_modes(args)
    if args.backend_compare:
        return cmd_backend_compare(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())

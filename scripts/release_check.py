#!/usr/bin/env python3
"""Run accountable local release checks; write small JSON and per-stage logs.

From the checkout: ``python scripts/release_check.py --stage fast
--wheel /path/to/wheel``. An absolute path to this script also works from
another directory.
The selected interpreter must import the candidate from a clean installation
with its native library beside the Python package; editable/source-tree
imports are rejected. Supply the exact installed wheel with ``--wheel``;
the runner checks its Python and native bytes. Test dependencies must also be
installed.
Use ``--stage all`` for coverage accounting and every local stage. A CPU-only
run reports the CUDA stage as failed; it never calls a CUDA skip a pass.

To include comparisons with the released 0.0.27 package, install it in a
separate environment and set ``DEEPWAVE_V0027_PYTHON`` to that environment's
Python. Unset, those comparisons skip; when set, the runner verifies the
interpreter imports 0.0.27 before collecting or running tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "scripts._release_pytest_plugin"
SOURCE_PATHS = (
    "src",
    "tests",
    "scripts",
    "docs",
    "benchmarks",
    ".github",
    "CMakeLists.txt",
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "CITATION.cff",
    "SECURITY.md",
    "ruff.toml",
    ".clang-format",
    ".clang-tidy",
)
PHYSICS_FILES = (
    "tests/test_convergence.py",
    "tests/test_adjoint_dot_product.py",
    "tests/test_gradient_exactness.py",
    "tests/test_reciprocity.py",
    "tests/test_boundary.py",
    "tests/test_determinism.py",
    "tests/test_adjoint_symbolic.py",
    "tests/test_equivalence_matrix.py",
    "tests/test_elastic_lamb.py",
    "tests/test_vti_physics.py",
    "tests/test_vti_oblique_interface.py",
    "tests/test_vti_visco_pml_reference.py",
    "tests/test_gradient_sampling_math.py",
    "tests/test_gradient_sampling_characterisation.py",
    "tests/test_heterogeneous_cfl.py",
)
STAGES: dict[str, tuple[str, ...]] = {
    "fast": ("tests", "-m", "not slow and not gpu", "-k", "not compile"),
    "physics": (*PHYSICS_FILES, "-m", "not slow and not gpu"),
    "heavy": ("tests", "-m", "slow and not manual_gradcheck_anchor"),
    "compile": (
        "tests",
        "-k",
        "compile",
        "-m",
        "not gpu and not manual_gradcheck_anchor",
    ),
    "anchors": ("tests", "-m", "manual_gradcheck_anchor"),
    "gpu": ("tests", "-m", "gpu"),
}


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - explicit argv, never a shell
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _source_identity() -> dict[str, str]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to identify the candidate")
    commit = _run([git, "rev-parse", "HEAD"]).stdout.strip()
    proc = subprocess.run(  # noqa: S603 - explicit git argv
        [
            git,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_PATHS,
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    digest = hashlib.sha256()
    for relative in sorted(set(proc.stdout.split(b"\0")) - {b""}):
        path = ROOT / os.fsdecode(relative)
        if not path.is_file():
            continue
        digest.update(relative + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"commit": commit, "public_content_sha256": digest.hexdigest()}


def _interpreter_info(python: str) -> dict[str, Any]:
    code = """import hashlib, importlib.util, json, pathlib, platform, sys
import torch, deepwave
path = str(pathlib.Path(deepwave._backend_utils.dll._name).resolve())
modules = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in pathlib.Path(deepwave.__file__).parent.glob('*.py')}
print(json.dumps(dict(python=sys.executable, python_version=sys.version,
 torch=torch.__version__, deepwave=deepwave.__version__, python_modules=modules,
 package=deepwave.__file__, library=path,
 library_sha256=(hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                 if path else None),
 platform=platform.platform(), cuda_available=torch.cuda.is_available(),
 cuda_devices=[dict(index=i, name=torch.cuda.get_device_name(i),
 memory=torch.cuda.get_device_properties(i).total_memory)
 for i in range(torch.cuda.device_count())])))"""
    proc = _run([python, "-c", code])
    if proc.returncode:
        raise RuntimeError(
            f"interpreter import failed ({proc.returncode}): {proc.stderr[-1000:]}"
        )
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    if not info.get("library_sha256"):
        raise RuntimeError("selected interpreter did not resolve a native library")
    _validate_installed_paths(info)
    expected_modules = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "src" / "deepwave").glob("*.py")
    }
    if info.get("python_modules") != expected_modules:
        raise RuntimeError(
            "selected interpreter's Deepwave Python files differ from this "
            "candidate; build and install this candidate before qualification"
        )
    for compiler in ("cc", "c++", "nvcc"):
        executable = shutil.which(compiler)
        if executable is None:
            continue
        result = _run([executable, "--version"])
        if result.returncode == 0:
            info[compiler] = result.stdout.splitlines()[0]
    return info


def _validate_installed_paths(info: dict[str, Any]) -> None:
    """Reject editable Python and mismatched native libraries before test collection."""
    package_file = Path(info["package"]).resolve()
    library = Path(info["library"]).resolve()
    source_package = (ROOT / "src" / "deepwave").resolve()
    if package_file.parent == source_package:
        raise RuntimeError(
            "selected interpreter imports editable/source-tree Deepwave; "
            "install the candidate wheel in a clean target first"
        )
    if package_file.name != "__init__.py" or library.parent != package_file.parent:
        raise RuntimeError(
            "selected interpreter loads Python and native Deepwave from "
            "different package directories"
        )


def _wheel_identity(wheel: Path, info: dict[str, Any]) -> dict[str, str]:
    """Verify that the selected interpreter loaded this exact wheel payload."""
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise RuntimeError(f"candidate wheel is missing or invalid: {wheel}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = {
                name
                for name in archive.namelist()
                if Path(name).parent == Path("deepwave")
            }
            expected = {
                f"deepwave/{name}": digest
                for name, digest in info["python_modules"].items()
            }
            expected[f"deepwave/{Path(info['library']).name}"] = info["library_sha256"]
            packaged_python = {name for name in members if name.endswith(".py")}
            if packaged_python != {name for name in expected if name.endswith(".py")}:
                raise RuntimeError(
                    "candidate wheel Python module list differs from the "
                    "installed package"
                )
            for name, digest in expected.items():
                if name not in members:
                    raise RuntimeError(f"candidate wheel is missing {name}")
                actual = hashlib.sha256()
                with archive.open(name) as payload:
                    for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                        actual.update(chunk)
                if actual.hexdigest() != digest:
                    raise RuntimeError(
                        f"selected interpreter's {name} differs from the "
                        "candidate wheel"
                    )
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"cannot read candidate wheel {wheel}: {error}") from error
    wheel_digest = hashlib.sha256()
    with wheel.open("rb") as payload:
        for chunk in iter(lambda: payload.read(1024 * 1024), b""):
            wheel_digest.update(chunk)
    return {"path": str(wheel.resolve()), "sha256": wheel_digest.hexdigest()}


def _reference_interpreter_info() -> dict[str, Any] | None:
    """Validate an explicitly requested legacy comparison before expensive work."""
    variable = "DEEPWAVE_V0027_PYTHON"
    python = os.environ.get(variable)
    if not python:
        return None
    code = """import hashlib, importlib.metadata, importlib.util, json, pathlib, sys
import torch, deepwave
spec = importlib.util.find_spec('deepwave.libdeepwave_C')
path = spec.origin if spec is not None else None
print(json.dumps(dict(python=sys.executable, python_version=sys.version,
 deepwave=importlib.metadata.version('deepwave'), torch=torch.__version__,
 package=deepwave.__file__, library=path,
 library_sha256=(hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                 if path else None))))"""
    try:
        proc = _run([python, "-c", code], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{variable}: cannot run {python!r}: {error}") from error
    if proc.returncode:
        raise RuntimeError(
            f"{variable}: {python!r} could not import the reference package "
            f"({proc.returncode}): {proc.stderr[-1000:]}"
        )
    try:
        info = json.loads(proc.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"{variable}: invalid interpreter probe result") from error
    if not isinstance(info, dict) or info.get("deepwave") != "0.0.27":
        raise RuntimeError(
            f"{variable}: {python!r} must import deepwave==0.0.27; got {info!r}"
        )
    if not info.get("library_sha256"):
        raise RuntimeError(f"{variable}: reference native library was not resolved")
    return info


def _pytest(
    python: str,
    args: tuple[str, ...],
    out: Path,
    label: str,
    *,
    collect_only: bool = False,
    extra_env: dict[str, str] | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    # Children always run from ROOT, even when the caller supplied --out
    # relative to another working directory. Both processes must use the
    # same report and cache paths.
    out = out.resolve()
    result_path = out / f"{label}.json"
    log_path = out / f"{label}.log"
    # A reused output directory must not turn an exit-zero child that never
    # starts a pytest session (for example --help in PYTEST_ADDOPTS) into a
    # pass by reading the previous invocation's report.
    result_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(extra_env or {})
    env["DEEPWAVE_RELEASE_RESULT"] = str(result_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TMPDIR", str(out))
    env.setdefault("XDG_CACHE_HOME", str(out / "cache"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out / "inductor"))
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    if workers > 1:
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env.setdefault("PIP_CACHE_DIR", str(out / "pip"))
    env.setdefault("CCACHE_DIR", str(out / "ccache"))
    env.setdefault("MYPY_CACHE_DIR", str(out / "mypy"))
    env.setdefault("RUFF_CACHE_DIR", str(out / "ruff"))
    env.setdefault("COVERAGE_FILE", str(out / ".coverage"))
    command = [python, "-m", "pytest", "-q", "-rs", "-p", PLUGIN, *args]
    if workers > 1 and not collect_only:
        command.extend(("-n", str(workers)))
    if collect_only:
        command.append("--collect-only")
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(  # noqa: S603 - explicit argv, never a shell
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.monotonic() - start
    data: dict[str, Any] = {}
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text())
        except (ValueError, OSError):
            data = {}
    selected = data.get("selected_ids")
    deselected = data.get("deselected_ids")
    outcomes = data.get("outcomes")
    valid = (
        proc.returncode == 0
        and isinstance(selected, list)
        and bool(selected)
        and len(set(selected)) == len(selected)
        and isinstance(deselected, list)
        and isinstance(outcomes, dict)
        and not data.get("collection_errors")
        and data.get("pytest_exit_code") == 0
        and (collect_only or set(outcomes) == set(selected))
        and (collect_only or set(outcomes.values()) <= {"passed", "failed", "skipped"})
        and (collect_only or "failed" not in outcomes.values())
        and (collect_only or "passed" in outcomes.values())
    )
    counts = (
        {
            status: list(outcomes.values()).count(status)
            for status in ("passed", "failed", "skipped")
        }
        if isinstance(outcomes, dict)
        else {}
    )
    return {
        "command": command,
        "exit_code": proc.returncode,
        "duration_seconds": round(duration, 3),
        "selected": len(selected) if isinstance(selected, list) else None,
        "passed": counts.get("passed"),
        "failed": counts.get("failed"),
        "skipped": counts.get("skipped"),
        "executed": (counts["passed"] + counts["failed"] if counts else None),
        "deselected": len(deselected) if isinstance(deselected, list) else None,
        "collection_errors": data.get("collection_errors"),
        "valid": bool(valid),
        "log": str(log_path),
        "environment_overrides": extra_env or {},
        "selected_ids": selected if isinstance(selected, list) else [],
    }


def _self_test(out: Path) -> bool:
    """A failed pytest child stays failed when a later child succeeds."""
    fake = out / "test_fake_release_child.py"
    fake.write_text(
        "def test_bad():\n    assert False\n\ndef test_good():\n    assert True\n"
    )
    bad = _pytest(sys.executable, (str(fake), "-k", "test_bad"), out, "fake_bad")
    good = _pytest(sys.executable, (str(fake), "-k", "test_good"), out, "fake_good")
    aggregate = bad["valid"] and good["valid"]
    identity_controls = {}
    for label, package, library in (
        (
            "editable",
            ROOT / "src" / "deepwave" / "__init__.py",
            out / "installed" / "deepwave" / "libdeepwave_C.so",
        ),
        (
            "mixed_package",
            out / "installed" / "deepwave" / "__init__.py",
            out / "other" / "deepwave" / "libdeepwave_C.so",
        ),
    ):
        try:
            _validate_installed_paths(
                {"package": str(package), "library": str(library)}
            )
        except RuntimeError:  # noqa: PERF203 - two self-test probes
            identity_controls[label] = True
        else:
            identity_controls[label] = False
    installed_package = out / "installed" / "deepwave"
    try:
        _validate_installed_paths(
            {
                "package": str(installed_package / "__init__.py"),
                "library": str(installed_package / "libdeepwave_C.so"),
            }
        )
    except RuntimeError:
        identity_controls["same_package"] = False
    else:
        identity_controls["same_package"] = True
    wheel_controls = {}
    wheel = out / "fake.whl"
    fake_info = {
        "python_modules": {"__init__.py": hashlib.sha256(b"new").hexdigest()},
        "library": str(installed_package / "libdeepwave_C.so"),
        "library_sha256": hashlib.sha256(b"native").hexdigest(),
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("deepwave/__init__.py", b"old")
        archive.writestr("deepwave/libdeepwave_C.so", b"native")
    try:
        _wheel_identity(wheel, fake_info)
    except RuntimeError:
        wheel_controls["stale_python"] = True
    else:
        wheel_controls["stale_python"] = False
    fake_info["python_modules"]["__init__.py"] = hashlib.sha256(b"old").hexdigest()
    fake_info["library_sha256"] = hashlib.sha256(b"stale native").hexdigest()
    try:
        _wheel_identity(wheel, fake_info)
    except RuntimeError:
        wheel_controls["stale_native"] = True
    else:
        wheel_controls["stale_native"] = False
    fake_info["library_sha256"] = hashlib.sha256(b"native").hexdigest()
    wheel_controls["matching_payload"] = bool(
        _wheel_identity(wheel, fake_info)["sha256"]
    )
    result = (
        bad["exit_code"] != 0
        and bad["failed"] == 1
        and not bad["valid"]
        and good["valid"]
        and not aggregate
        and all(identity_controls.values())
        and all(wheel_controls.values())
    )
    reference_controls = {}
    for label, reference in (
        ("missing", str(out / "missing_reference_python")),
        ("wrong_version", sys.executable),
    ):
        env = {**os.environ, "DEEPWAVE_V0027_PYTHON": reference}
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            "plan",
            "--out",
            str(out / f"reference_{label}"),
        ]
        started = time.monotonic()
        probe = _run(command, env=env)
        expected = (
            "cannot run" if label == "missing" else "must import deepwave==0.0.27"
        )
        rejected = (
            probe.returncode == 2
            and "DEEPWAVE_V0027_PYTHON" in probe.stderr
            and expected in probe.stderr
        )
        reference_controls[label] = {
            "command": command,
            "environment_overrides": {"DEEPWAVE_V0027_PYTHON": reference},
            "duration_seconds": round(time.monotonic() - started, 3),
            "exit_code": probe.returncode,
            "stderr": probe.stderr,
            "valid": rejected,
        }
        result = result and rejected
    (out / "self_test.json").write_text(
        json.dumps(
            {
                "bad": bad,
                "good": good,
                "aggregate": aggregate,
                "identity_controls": identity_controls,
                "wheel_controls": wheel_controls,
                "reference_controls": reference_controls,
                "valid": result,
            }
        )
    )
    return result


def main() -> int:
    """Collect a complete plan, then run the selected release stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=[*STAGES, "all", "plan", "self-test"], default="all"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter with installed Deepwave"
    )
    parser.add_argument(
        "--wheel", type=Path, help="exact wheel installed into the selected interpreter"
    )
    parser.add_argument(
        "--out", type=Path, help="result directory (default: unique /tmp directory)"
    )
    parser.add_argument(
        "--cuda-device",
        action="append",
        type=int,
        help="GPU index; repeat for each architecture (default: all visible)",
    )
    parser.add_argument(
        "--fast-workers",
        type=int,
        default=3,
        help="bounded pytest workers for the fast stage (default: 3)",
    )
    args = parser.parse_args()
    if args.fast_workers < 1:
        parser.error("--fast-workers must be positive")
    out = args.out or Path(tempfile.mkdtemp(prefix="deepwave-release-check-"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "self-test":
        ok = _self_test(out)
        print(f"self-test: {'passed' if ok else 'FAILED'}; {out}")
        return 0 if ok else 1
    # Resolving a venv's python symlink selects the base interpreter and loses
    # that environment's installed package and dependencies.
    python = str(Path(args.python).absolute())
    try:
        reference = _reference_interpreter_info()
        identity = _source_identity()
        interpreter = _interpreter_info(python)
        if args.wheel is None:
            raise RuntimeError("--wheel is required for release qualification")
        wheel_identity = _wheel_identity(args.wheel, interpreter)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"release check setup failed: {error}", file=sys.stderr)
        return 2
    print(
        f"interpreter: {interpreter['python']} "
        f"(Python {interpreter['python_version'].split()[0]}, "
        f"torch {interpreter['torch']})",
        flush=True,
    )
    names = list(STAGES) if args.stage in ("all", "plan") else [args.stage]
    summary: dict[str, Any] = {
        "source": identity,
        "interpreter": interpreter,
        "wheel": wheel_identity,
        "reference_interpreter": reference,
        "stages": {},
        "coverage": None,
    }
    if args.stage in ("all", "plan"):
        universe = _pytest(
            python,
            ("tests", "-m", "manual_gradcheck_anchor or not manual_gradcheck_anchor"),
            out,
            "collection_all",
            collect_only=True,
        )
        plans = {
            name: _pytest(python, STAGES[name], out, f"plan_{name}", collect_only=True)
            for name in names
        }
        planned = set().union(*(set(plan["selected_ids"]) for plan in plans.values()))
        missing = sorted(set(universe["selected_ids"]) - planned)
        summary["coverage"] = {
            "collected": universe["selected"],
            "planned": len(planned),
            "missing": missing,
            "valid": universe["valid"]
            and all(plan["valid"] for plan in plans.values())
            and not missing,
        }
    if args.stage == "plan":
        summary["valid"] = bool(summary["coverage"]["valid"])
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"plan: {'passed' if summary['valid'] else 'FAILED'}; {out}")
        return 0 if summary["valid"] else 1
    for name in names:
        if name == "gpu" and not interpreter["cuda_available"]:
            summary["stages"][name] = {
                "valid": False,
                "reason": (
                    "CUDA unavailable in selected interpreter; GPU stage not verified"
                ),
            }
            continue
        if name == "gpu":
            indices = args.cuda_device or [
                device["index"] for device in interpreter["cuda_devices"]
            ]
            for index in indices:
                label = f"gpu_{index}"
                gpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(index)}
                identified = _run(
                    [
                        python,
                        "-c",
                        "import torch; print(torch.cuda.get_device_name(0))",
                    ],
                    env=gpu_env,
                )
                if identified.returncode:
                    summary["stages"][label] = {
                        "valid": False,
                        "reason": "CUDA device could not be identified",
                        "stderr": identified.stderr[-500:],
                    }
                    continue
                result = _pytest(
                    python,
                    STAGES["gpu"],
                    out,
                    label,
                    extra_env={"CUDA_VISIBLE_DEVICES": str(index)},
                )
                result["valid"] = result["valid"] and result["passed"] > 0
                result["device_name"] = identified.stdout.strip()
                summary["stages"][label] = {
                    key: value for key, value in result.items() if key != "selected_ids"
                }
                status = "passed" if result["valid"] else "FAILED"
                print(
                    f"{label}: {status}; selected={result['selected']} "
                    f"passed={result['passed']} failed={result['failed']} "
                    f"skipped={result['skipped']}",
                    flush=True,
                )
            distinct_indices = list(dict.fromkeys(indices))
            if len(distinct_indices) >= 2:
                multi_result = _pytest(
                    python,
                    ("tests/test_native_cuda_device_scope.py", "-m", "gpu"),
                    out,
                    "gpu_multi",
                    extra_env={
                        "CUDA_VISIBLE_DEVICES": (
                            f"{distinct_indices[0]},{distinct_indices[1]}"
                        )
                    },
                )
                multi_result["valid"] = (
                    multi_result["valid"]
                    and multi_result["selected"] == 4
                    and multi_result["passed"] == 4
                    and multi_result["skipped"] == 0
                )
                summary["stages"]["gpu_multi"] = {
                    key: value
                    for key, value in multi_result.items()
                    if key != "selected_ids"
                }
            else:
                summary["stages"]["gpu_multi"] = {
                    "valid": False,
                    "reason": (
                        "two visible CUDA devices required for device-scope tests"
                    ),
                    "selected": 4,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 4,
                    "executed": 0,
                }
            continue
        result = _pytest(
            python,
            STAGES[name],
            out,
            name,
            workers=args.fast_workers if name == "fast" else 0,
        )
        summary["stages"][name] = {
            key: value for key, value in result.items() if key != "selected_ids"
        }
        status = "passed" if result["valid"] else "FAILED"
        print(
            f"{name}: {status}; selected={result['selected']} "
            f"passed={result['passed']} failed={result['failed']} "
            f"skipped={result['skipped']}",
            flush=True,
        )
    valid = all(stage["valid"] for stage in summary["stages"].values()) and (
        summary["coverage"] is None or summary["coverage"]["valid"]
    )
    summary["valid"] = bool(valid)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary: {out / 'summary.json'}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

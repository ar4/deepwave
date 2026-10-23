"""Pytest configuration for the Deepwave test suite.

The suite can be run in parallel with ``pytest-xdist``::

    pytest -n auto

Each test is small (typically a couple of shots) and the propagators already
parallelise over shots with a multi-threaded backend. If every xdist worker
*also* lets its numerical libraries spin up a full thread pool, the CPU is
massively oversubscribed and the parallel run is actually slower than a serial
one. The libraries involved (OpenMP/libgomp in the compiled backend, OpenBLAS
behind NumPy/SciPy, MKL, NumExpr, and TorchInductor's compile pool) each read
their own environment variable, so we pin them all to a single thread *per
worker* below. With this, ``N`` workers use ``N`` cores. Serial runs (no xdist)
are left untouched so the backend can still use multiple threads per shot.

The thread limits are applied before NumPy/torch are imported in the worker so
that the libraries pick them up.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_torch_compile_cache() -> None:
    """Give each test its own Dynamo variants without changing compiler limits.

    Shape/dtype variants from earlier tests can exhaust the shared code object's
    recompile limit and silently turn a compiled reference into slow eager work.
    Variants within one test still exercise PyTorch's configured cache policy.
    """
    import torch._dynamo

    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def _restore_torch_globals() -> Iterator[None]:
    """Keep one test's dtype, thread count and anomaly mode from leaking."""
    import torch

    dtype = torch.get_default_dtype()
    threads = torch.get_num_threads()
    anomaly = torch.is_anomaly_enabled()
    try:
        yield
    finally:
        if torch.get_default_dtype() != dtype:
            torch.set_default_dtype(dtype)
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.is_anomaly_enabled() != anomaly:
            torch.autograd.set_detect_anomaly(anomaly)


@pytest.fixture(autouse=True)
def _chdir_to_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every test with its cwd anchored under pytest's own tmp tree.

    Disk-mode storage (``storage_mode="disk"``) creates a
    ``deepwave_tmp_<pid>_<uuid>`` directory under ``storage_path`` (see
    ``_storage.py``'s ``TemporaryStorage``). That default is now the system
    temporary directory rather than the working directory, so this fixture no
    longer governs where the scratch directory lands -- but a test that passes
    ``storage_path="."`` explicitly, or any test that writes a relative path,
    still would. Without the fixture such a directory lands wherever the test
    process happened to be invoked from (leaking into the repo root).
    Chdir-ing into pytest's per-test ``tmp_path`` means it lands under
    pytest's own (auto-cleaned) tmp tree instead. Tests that need to read a
    repo file must use an absolute path anchored on ``__file__``, not a path
    relative to the invocation directory.
    """
    monkeypatch.chdir(tmp_path)


MANUAL_GRADCHECK_ANCHOR_MARKER = "manual_gradcheck_anchor"


def _explicitly_named_files(config: pytest.Config) -> "set[str]":
    """Absolute paths of the test files named directly on the command line.

    A bare directory argument (``pytest tests/``) contributes nothing; only an
    argument that resolves to a file, with or without a ``::nodeid`` suffix,
    counts as naming that file explicitly.
    """
    base = Path(str(config.invocation_params.dir))
    named = set()
    for arg in config.args:
        path = Path(str(arg).split("::")[0])
        if not path.is_absolute():
            path = base / path
        if path.is_file():
            named.add(str(path.resolve()))
    return named


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: "list[pytest.Item]",
) -> None:
    """Auto-mark the gradcheck family ``slow``; deselect the manual anchors.

    The remaining finite-difference gradcheck tests are per-element Jacobian
    checks and are far more expensive per test than anything else in the
    suite. Marking them ``slow`` here, rather than decorating each function
    individually, provides a fast-subset lever::

        pytest -m "not slow"

    Tests marked ``manual_gradcheck_anchor`` are deselected outright rather
    than merely marked. They independently check small-call derivatives,
    including shared errors that backend agreement can miss, and are kept
    out of every automatic selection
    (a marker alone would not achieve that: pytest's ``-m`` is a single
    overwriting option, so any invocation passing its own ``-m`` -- e.g.
    ``-m "not slow"`` -- would replace a global filter and collect them
    again). They still run when the marker name is passed to ``-m``, or when
    their file or a nodeid inside it is named directly on the command line.

    See the ``markers`` registered in pyproject.toml.
    """
    anchors_requested = MANUAL_GRADCHECK_ANCHOR_MARKER in (config.option.markexpr or "")
    named_files = _explicitly_named_files(config)
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if "gradcheck" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if (
            item.get_closest_marker(MANUAL_GRADCHECK_ANCHOR_MARKER) is not None
            and not anchors_requested
            and str(Path(str(item.path)).resolve()) not in named_files
        ):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        os.environ.setdefault(_var, "1")

    import torch

    torch.set_num_threads(1)

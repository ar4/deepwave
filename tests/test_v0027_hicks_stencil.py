"""The changed-stencil migration warning, measured against a real ``v0.0.27``.

Deepwave 0.0.27 chose a Hicks interpolation window's integer cells by
truncating a floating-point sum, ``(location + x).long()``. That is wrong in
two ways: truncation lands one cell too low wherever the sum falls a fraction
of a ULP below an integer, and it rounds towards zero rather than downwards, so
a coordinate between ``-1`` and ``0`` repeats a cell instead of reaching the one
below it. The cells are now ``floor(location) + arange(...)``, computed in
integer arithmetic, which is exact. The weights are unchanged.

Because that moves numbers for an off-node coordinate,
``deepwave.MigrationWarning`` says so where the window is built. The condition
it warns on is not a guess about when the old arithmetic went wrong: Deepwave
reproduces that arithmetic (``_v0027_stencil_indices``) and compares. This file
is the evidence that the reproduction is faithful.

Two groups:

* **Always on.** The reproduction is compared against a transcription of the
  released expression, over a wide sweep of coordinates, halfwidths, dtypes and
  signs; the warning is shown to fire exactly when the two index sets differ
  and to stay silent otherwise; the comparison is shown to follow the dtype
  ``v0.0.27`` would have used (float32 for an omitted ``Hicks(dtype=)``,
  whatever :func:`torch.get_default_dtype` says now) rather than the current
  default dtype; and the example ``docs/upgrading.rst`` quotes is reproduced.
* **Against a real install.** With ``DEEPWAVE_V0027_PYTHON`` pointing at the
  interpreter of an environment with ``deepwave==0.0.27``, the *released*
  ``Hicks`` builds the same sweep in a subprocess and its cells are compared
  with both this release's and the reproduction's. Skipped, with a reason, when
  that variable is unset, because no such interpreter exists in CI.
"""

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings
from typing import Any, Optional

import pytest
import torch

import deepwave
import deepwave._common
from deepwave._location_interpolation import (
    _STENCIL_CAUSE,
    _get_hicks_for_one_location_dim,
    _v0027_stencil_indices,
)

#: The environment variable naming a ``v0.0.27`` interpreter.
_V0027_ENV = "DEEPWAVE_V0027_PYTHON"

#: The Kaiser ``beta`` per halfwidth, as every Deepwave release has had them.
_BETAS = (0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64)

_EPS = 1e-5


def _sweep() -> list[dict[str, Any]]:
    """The parameter sweep, inside what ``v0.0.27``'s ``Hicks`` accepted.

    Coordinates on a four-hundredth-of-a-cell grid at several magnitudes (the
    float32 rounding that causes the difference gets coarser as the exponent
    grows), plus a negative run, which is where truncation towards zero differs
    from flooring. Halfwidths ``1``, ``4`` and ``10`` are the ends and the
    default of the accepted range, and both dtypes ``v0.0.27`` could build
    weights in are covered, as are monopoles and dipoles.
    """
    cases: list[dict[str, Any]] = []
    for dtype in ("float32", "float64"):
        for halfwidth in (1, 4, 10):
            for monopole in (True, False):
                for base in (0, 1, 50, 500, 5000):
                    cases.extend(
                        {
                            "location": base + step / 400.0,
                            "halfwidth": halfwidth,
                            "monopole": monopole,
                            "dtype": dtype,
                        }
                        for step in range(400)
                    )
                cases.extend(
                    {
                        "location": -(step / 400.0) - 0.001,
                        "halfwidth": halfwidth,
                        "monopole": monopole,
                        "dtype": dtype,
                    }
                    for step in range(200)
                )
    return cases


def _released_expression(case: dict[str, Any]) -> list[int]:
    """``v0.0.27``'s window, transcribed from the released source.

    Deliberately written out again here rather than imported from
    ``deepwave._location_interpolation``: this is the independent side of the
    comparison, so it must not share code with the thing it is checking.
    """
    location = float(case["location"])
    halfwidth = int(case["halfwidth"])
    dtype = getattr(torch, case["dtype"])
    if case["monopole"] and abs(location - round(location)) < _EPS:
        return [int(torch.tensor([location]).round().long().item())]
    x = (
        torch.arange(-halfwidth + 1, halfwidth + 1, dtype=dtype)
        - location
        + int(location)
    )
    return [int(v) for v in (location + x).long()]


def _this_release(case: dict[str, Any]) -> list[int]:
    """This release's window: exact integer arithmetic about ``floor``."""
    location = float(case["location"])
    halfwidth = int(case["halfwidth"])
    if case["monopole"] and abs(location - round(location)) < _EPS:
        return [int(torch.tensor([location]).round().long().item())]
    floor_location = math.floor(location)
    return [floor_location + step for step in range(-halfwidth + 1, halfwidth + 1)]


def test_the_reproduction_matches_the_released_expression() -> None:
    """``_v0027_stencil_indices`` is the released arithmetic, case for case."""
    mismatches = []
    for case in _sweep():
        produced = _v0027_stencil_indices(
            float(case["location"]),
            int(case["halfwidth"]),
            getattr(torch, case["dtype"]),
            bool(case["monopole"]),
            _EPS,
        )
        expected = _released_expression(case)
        if produced != expected:
            mismatches.append((case, produced, expected))
    assert not mismatches, mismatches[:5]


def test_the_sweep_contains_both_answers() -> None:
    """The sweep is not vacuous: it holds agreeing and differing cases.

    Without this, a reproduction that returned this release's own answer would
    pass every comparison above by never disagreeing with anything.
    """
    differing = sum(
        1 for case in _sweep() if _released_expression(case) != _this_release(case)
    )
    assert 0 < differing < len(_sweep()), differing


def _stencil_warnings(location: float, halfwidth: int = 4) -> list[str]:
    """Builds one window through the real code path, returning its warnings."""
    saved = set(deepwave._common._warned_once_causes)
    deepwave._common._warned_once_causes.discard(_STENCIL_CAUSE)
    try:
        beta = torch.tensor(_BETAS[halfwidth - 1])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _get_hicks_for_one_location_dim(
                {},
                location,
                halfwidth,
                beta,
                [False, False],
                [-0.5, 99.5],
                100,
                True,
                eps=_EPS,
                compare_v0027=True,
            )
        return [
            str(w.message)
            for w in record
            if issubclass(w.category, deepwave.MigrationWarning)
        ]
    finally:
        deepwave._common._warned_once_causes.clear()
        deepwave._common._warned_once_causes.update(saved)


def test_the_warning_fires_exactly_when_the_cells_differ() -> None:
    """Every differing coordinate warns, and no agreeing coordinate does.

    Checked on a sample of the sweep rather than all of it: each case here goes
    through the real construction path, which is far more work than comparing
    two index lists.
    """
    sample = [case for case in _sweep() if case["dtype"] == "float32"][::97]
    assert sample
    for case in sample:
        if case["halfwidth"] != 4 or not case["monopole"]:
            continue
        location = float(case["location"])
        differs = _released_expression(case) != _this_release(case)
        emitted = _stencil_warnings(location)
        assert bool(emitted) is differs, (location, differs, emitted)


def test_the_warning_is_emitted_once_per_process() -> None:
    """A second differing window says nothing: the cause is spent."""
    saved = set(deepwave._common._warned_once_causes)
    deepwave._common._warned_once_causes.discard(_STENCIL_CAUSE)
    try:
        beta = torch.tensor(_BETAS[3])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(3):
                _get_hicks_for_one_location_dim(
                    {},
                    5.6,
                    4,
                    beta,
                    [False, False],
                    [-0.5, 99.5],
                    100,
                    True,
                    eps=_EPS,
                    compare_v0027=True,
                )
        migrations = [
            w for w in record if issubclass(w.category, deepwave.MigrationWarning)
        ]
        assert len(migrations) == 1, [str(w.message) for w in migrations]
    finally:
        deepwave._common._warned_once_causes.clear()
        deepwave._common._warned_once_causes.update(saved)


def test_the_reproduction_ignores_the_current_default_dtype() -> None:
    """``_v0027_stencil_indices`` follows its ``dtype`` argument, nothing else.

    Every tensor it builds names its dtype, so the arithmetic it reproduces is
    the arithmetic ``v0.0.27`` did, not the arithmetic the caller's
    :func:`torch.set_default_dtype` would give.
    """
    locations = [5.6, -0.4, 50.995, 1.0025]
    under_stock = [
        _v0027_stencil_indices(location, 4, torch.float32, True, _EPS)
        for location in locations
    ]
    torch.set_default_dtype(torch.float64)
    try:
        under_float64 = [
            _v0027_stencil_indices(location, 4, torch.float32, True, _EPS)
            for location in locations
        ]
    finally:
        torch.set_default_dtype(torch.float32)
    assert under_stock == under_float64


def _hicks_stencil_messages(
    locations: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> list[str]:
    """The stencil warnings one ``Hicks`` construction emits."""
    saved = set(deepwave._common._warned_once_causes)
    deepwave._common._warned_once_causes.discard(_STENCIL_CAUSE)
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            deepwave.Hicks(locations, dtype=dtype)
        return [
            str(w.message)
            for w in record
            if issubclass(w.category, deepwave.MigrationWarning)
            and "interpolation window" in str(w.message)
        ]
    finally:
        deepwave._common._warned_once_causes.clear()
        deepwave._common._warned_once_causes.update(saved)


def test_the_warning_does_not_depend_on_the_default_dtype() -> None:
    """``Hicks(dtype=None)`` is compared against ``v0.0.27``'s own float32.

    That release's ``dtype`` argument defaulted to ``torch.float`` and never
    consulted :func:`torch.get_default_dtype`, so the window it would have
    built for an omitted ``dtype=`` is the float32 one whatever the default is
    now. Comparing in the current default instead made a genuine window change
    invisible to anyone who had called ``torch.set_default_dtype(torch.float64)``
    -- exactly the callers most likely to be migrating a float64 script.
    """
    assert torch.get_default_dtype() is torch.float32
    under_stock = _hicks_stencil_messages(torch.tensor([[[5.6, 7.0]]]))
    assert len(under_stock) == 1, under_stock
    torch.set_default_dtype(torch.float64)
    try:
        under_float64 = _hicks_stencil_messages(
            torch.tensor([[[5.6, 7.0]]], dtype=torch.float64)
        )
    finally:
        torch.set_default_dtype(torch.float32)
    assert len(under_float64) == 1, under_float64


def test_an_explicit_dtype_is_compared_in_that_dtype() -> None:
    """``Hicks(dtype=torch.float64)`` was also a call ``v0.0.27`` accepted.

    Asked for float64 weights, that release did its index arithmetic in float64
    too, and for this coordinate that arithmetic gave the same window as exact
    integer arithmetic does. There is nothing to migrate, so nothing is said --
    and the reproduction must follow the dtype that was asked for rather than
    always assuming float32.
    """
    messages = _hicks_stencil_messages(
        torch.tensor([[[5.6, 7.0]]], dtype=torch.float64), dtype=torch.float64
    )
    assert messages == [], messages
    assert _v0027_stencil_indices(
        float(torch.tensor(5.6, dtype=torch.float64).item()),
        4,
        torch.float64,
        True,
        _EPS,
    ) == [2, 3, 4, 5, 6, 7, 8, 9]


def test_the_documented_example() -> None:
    """``docs/upgrading.rst``'s 5.6 example is the cells the code produces."""
    case = {
        "location": float(torch.tensor(5.6).item()),
        "halfwidth": 4,
        "monopole": True,
        "dtype": "float32",
    }
    assert _released_expression(case) == [1, 3, 4, 5, 6, 7, 8, 9]
    assert _this_release(case) == [2, 3, 4, 5, 6, 7, 8, 9]


def test_the_documented_hit_rate() -> None:
    """The 6.2% figure ``docs/upgrading.rst`` quotes, recomputed.

    Off-node coordinates in ``[1, 50]`` on a hundredth-of-a-cell grid, the
    default ``halfwidth=4``, monopole, ``float32`` weights -- the sweep the page
    describes.
    """
    total = 0
    differing = 0
    for step in range(100, 5001):
        case = {
            "location": step / 100.0,
            "halfwidth": 4,
            "monopole": True,
            "dtype": "float32",
        }
        location = float(case["location"])
        if abs(location - round(location)) < _EPS:
            continue
        total += 1
        differing += _released_expression(case) != _this_release(case)
    assert total > 4000
    assert 6.0 <= 100 * differing / total <= 6.4, (differing, total)


# ---------------------------------------------------------------------------
# Against a real v0.0.27 install.
# ---------------------------------------------------------------------------

#: Run in the ``v0.0.27`` interpreter: build each case's window with the
#: *released* per-dimension helper and report its cells.
_V0027_SCRIPT = """
import json, sys, torch
from deepwave.location_interpolation import _get_hicks_for_one_location_dim

betas = (0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64)
cases = json.load(open(sys.argv[1]))
out = []
for case in cases:
    dtype = getattr(torch, case["dtype"])
    beta = torch.tensor(betas[case["halfwidth"] - 1]).to(dtype)
    locations, _ = _get_hicks_for_one_location_dim(
        {},
        case["location"],
        case["halfwidth"],
        beta,
        [False, False],
        [-0.5, 99.5],
        100,
        case["monopole"],
    )
    out.append([int(v) for v in locations])
json.dump(out, open(sys.argv[2], "w"))
"""


def _v0027_python() -> str:
    path = os.environ.get(_V0027_ENV)
    if not path:
        pytest.skip(
            f"set {_V0027_ENV} to the python of an environment with "
            "deepwave==0.0.27 installed to compare Hicks stencils against the "
            "released implementation"
        )
    return path


def test_against_the_released_hicks() -> None:
    """The released ``Hicks`` agrees with the reproduction on every case.

    The claim this settles: the condition Deepwave warns on is "these cells
    differ from the ones ``v0.0.27`` produced", not "differ from what we think
    it produced".
    """
    interpreter = _v0027_python()
    cases = _sweep()
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        script = directory / "v0027.py"
        script.write_text(_V0027_SCRIPT)
        cases_path = directory / "cases.json"
        cases_path.write_text(json.dumps(cases))
        out_path = directory / "out.json"
        result = subprocess.run(  # noqa: S603 - fixed argv, a user-named interpreter
            [interpreter, str(script), str(cases_path), str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        released = json.loads(out_path.read_text())

    assert len(released) == len(cases)
    reproduction_mismatches = []
    differing = 0
    for case, actual in zip(cases, released):
        produced = _v0027_stencil_indices(
            float(case["location"]),
            int(case["halfwidth"]),
            getattr(torch, case["dtype"]),
            bool(case["monopole"]),
            _EPS,
        )
        if produced != actual:
            reproduction_mismatches.append((case, produced, actual))
        if actual != _this_release(case):
            differing += 1
    assert not reproduction_mismatches, reproduction_mismatches[:5]
    # Not vacuous: the released code really does disagree with this one.
    assert 0 < differing < len(cases), differing
    print(
        f"v0.0.27 Hicks stencil comparison: {len(cases)} cases, "
        f"{differing} with different cells, 0 reproduction mismatches",
        file=sys.stderr,
    )


def test_pr122_float32_halfwidth_two_stencil() -> None:
    """PR #122's 31.4 coordinate must address the four integer stencil cells."""
    locations, _ = _get_hicks_for_one_location_dim(
        {},
        31.4,
        2,
        torch.tensor(_BETAS[1], dtype=torch.float32),
        [False, False],
        [-0.5, 99.5],
        100,
        True,
    )
    assert locations.tolist() == [30, 31, 32, 33]
    case = {"location": 31.4, "halfwidth": 2, "monopole": True, "dtype": "float32"}
    assert _released_expression(case) != [30, 31, 32, 33]

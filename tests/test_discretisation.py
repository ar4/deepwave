"""Tests for ``deepwave.discretisation.suggest_discretisation``.

The CFL and resampling helpers that share this module are exercised by
``tests/test_common.py``; what is tested here is the grid-and-time-step
suggestion built on top of them, and the points-per-wavelength table it
designs from.
"""

import math
import pathlib
import warnings
from fractions import Fraction
from typing import Any, Callable

import pytest
import torch

import deepwave
from deepwave._common import check_points_per_wavelength
from deepwave.discretisation import (
    _DISPERSION_ERRORS,
    _MIN_POINTS_PER_WAVELENGTH,
    _ORDERS,
    _RESOLUTION_CHECK_POINTS_PER_WAVELENGTH,
    _points_per_wavelength_rst,
    _relative_cost_per_cell,
    cfl_condition_n,
    critical_courant_number,
    suggest_discretisation,
)
from deepwave.types import Accuracy

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

# The standard central finite-difference coefficients, written out here
# independently of the library so that the table the suggestion designs from
# is checked against the stencils rather than against a copy of itself.
# Regular grid: the second-derivative coefficients c_0, c_1, ... Staggered
# grid: the first-derivative coefficients w_1, w_2, ...
SECOND_DERIVATIVE_COEFFICIENTS = {
    2: (Fraction(-2), Fraction(1)),
    4: (Fraction(-5, 2), Fraction(4, 3), Fraction(-1, 12)),
    6: (Fraction(-49, 18), Fraction(3, 2), Fraction(-3, 20), Fraction(1, 90)),
    8: (
        Fraction(-205, 72),
        Fraction(8, 5),
        Fraction(-1, 5),
        Fraction(8, 315),
        Fraction(-1, 560),
    ),
}
STAGGERED_FIRST_DERIVATIVE_COEFFICIENTS = {
    2: (Fraction(1),),
    4: (Fraction(9, 8), Fraction(-1, 24)),
    6: (Fraction(75, 64), Fraction(-25, 384), Fraction(3, 640)),
    8: (
        Fraction(1225, 1024),
        Fraction(-245, 3072),
        Fraction(49, 5120),
        Fraction(-5, 7168),
    ),
}


def _regular_phase_velocity_ratio(accuracy: int, theta: float) -> float:
    """``v_num / v`` for axial propagation on the regular grid."""
    c = SECOND_DERIVATIVE_COEFFICIENTS[accuracy]
    symbol = -(
        float(c[0])
        + 2 * sum(float(c[m]) * math.cos(m * theta) for m in range(1, len(c)))
    )
    return math.sqrt(max(symbol, 0.0)) / theta


def _staggered_phase_velocity_ratio(accuracy: int, theta: float) -> float:
    """``v_num / v`` for axial propagation on the staggered grid."""
    w = STAGGERED_FIRST_DERIVATIVE_COEFFICIENTS[accuracy]
    modified_k = 2 * sum(
        float(w[i]) * math.sin((2 * (i + 1) - 1) * theta / 2) for i in range(len(w))
    )
    return modified_k / theta


def _smallest_adequate_sampling(
    ratio: Callable[[int, float], float], accuracy: int, tolerance: float
) -> float:
    """The smallest points-per-wavelength meeting ``tolerance`` from there on."""
    step = 0.001
    smallest = None
    for i in range(int((200.0 - 2.0) / step) + 1):
        points = 2.0 + i * step
        if abs(ratio(accuracy, 2 * math.pi / points) - 1) <= tolerance:
            if smallest is None:
                smallest = points
        else:
            smallest = None
    assert smallest is not None
    return smallest


# ---------------------------------------------------------------------------
# The points-per-wavelength table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("dispersion_error", _DISPERSION_ERRORS)
@pytest.mark.parametrize("accuracy", _ORDERS)
def test_tabulated_sampling_matches_the_stencils(
    staggered: bool, dispersion_error: float, accuracy: Accuracy
) -> None:
    """Each tabulated minimum is the stencil's own dispersion threshold.

    The table is quoted to the nearest 0.1, so an entry may sit up to 0.05
    either side of the exact threshold; the check is that it is correct to
    the precision it is written at, not that it rounds up.
    """
    ratio = (
        _staggered_phase_velocity_ratio if staggered else _regular_phase_velocity_ratio
    )
    exact = _smallest_adequate_sampling(ratio, accuracy, dispersion_error)
    tabulated = _MIN_POINTS_PER_WAVELENGTH[staggered][dispersion_error][accuracy]
    assert abs(tabulated - exact) <= 0.05, (
        staggered,
        dispersion_error,
        accuracy,
        exact,
    )


def test_generated_table_carries_every_tabulated_value() -> None:
    """The rendered docs table is the constant, in list-table form."""
    rendered = _points_per_wavelength_rst()
    assert rendered.startswith(".. list-table:: Minimum points-per-wavelength")
    for staggered in (False, True):
        for dispersion_error in _DISPERSION_ERRORS:
            for accuracy in _ORDERS:
                value = _MIN_POINTS_PER_WAVELENGTH[staggered][dispersion_error][
                    accuracy
                ]
                assert f"     - {value}" in rendered
    # One header row plus one row per order, each of four values.
    assert rendered.count("   * - ") == 1 + len(_ORDERS)


def test_docs_include_the_generated_table_rather_than_a_copy() -> None:
    """``discretisation.rst`` includes the table instead of duplicating it."""
    page = (DOCS / "discretisation.rst").read_text()
    assert ".. include:: _generated/points_per_wavelength_table.rst" in page
    assert ".. list-table:: Minimum points-per-wavelength" not in page


def test_resolution_check_threshold_is_the_one_the_library_warns_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_RESOLUTION_CHECK_POINTS_PER_WAVELENGTH`` is what ``_common`` uses.

    Pins the floor against the real behaviour of
    ``check_points_per_wavelength`` rather than against a second copy of the
    number, so that moving the library's threshold fails here.
    """
    # Other tests may already have emitted this once-per-process warning.
    # Isolate just this cause and restore the caller's original state afterwards.
    monkeypatch.setattr(
        deepwave._common,
        "_warned_once_causes",
        deepwave._common._warned_once_causes - {"points_per_wavelength"},
    )
    floor = _RESOLUTION_CHECK_POINTS_PER_WAVELENGTH
    wavelength = 1500.0 / 25.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_points_per_wavelength(1500.0, 25.0, [wavelength / (floor + 0.01)] * 2)
    with pytest.warns(UserWarning, match="six grid cells per wavelength"):
        check_points_per_wavelength(1500.0, 25.0, [wavelength / (floor - 0.01)] * 2)


def test_resolution_warning_says_what_its_velocity_is_and_what_to_do() -> None:
    """The warning has to be honest about its velocity and offer every remedy.

    The velocity it quotes is the smallest **non-zero** cell of the model,
    which a vacuum or fluid cell is excluded from; calling it "the minimum
    velocity" would invite the reader to look for a cell that is not there.
    The remedies it offers have to be the full set: a finer grid, the right
    ``pml_freq``, a higher ``accuracy``, and the helper that computes a grid
    satisfying this check and the CFL condition together.
    """
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    with pytest.warns(UserWarning, match="six grid cells per wavelength") as record:
        check_points_per_wavelength(1500.0, 25.0, [30.0, 30.0])
    message = str(record[0].message)
    assert "minimum non-zero velocity" in message
    assert "smallest non-zero value in the model" in message
    assert "grid_spacing" in message
    assert "pml_freq=" in message
    assert "accuracy" in message
    assert "deepwave.suggest_discretisation" in message


def test_resolution_warning_velocity_excludes_zero_cells() -> None:
    """A vacuum cell does not become the quoted minimum velocity.

    An acoustic model with a valid vacuum region has a true minimum of zero,
    which would make the quoted cells-per-wavelength zero and the warning
    unfalsifiable. The propagators pass the minimum over non-zero cells, so
    the number in the message is one the reader can find in their model.
    """
    ny = nx = 24
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    rho = torch.full((ny, nx), 1000.0, dtype=torch.float64)
    v[:4] = 0.0
    rho[:4] = 0.0
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    with pytest.warns(UserWarning, match="six grid cells per wavelength") as record:
        deepwave.acoustic(
            v,
            rho,
            30.0,
            0.001,
            source_amplitudes_volume_rate=torch.zeros(1, 1, 4, dtype=torch.float64),
            source_locations_volume_rate=torch.tensor([[[ny // 2, nx // 2]]]),
            pml_freq=25.0,
            pml_width=3,
            accuracy=2,
        )
    messages = [
        str(w.message)
        for w in record
        if "six grid cells per wavelength" in str(w.message)
    ]
    assert len(messages) == 1
    assert "minimum non-zero velocity ~1500" in messages[0]


# ---------------------------------------------------------------------------
# suggest_discretisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_vel", [500.0, 1500.0, 3000.0])
@pytest.mark.parametrize("max_freq", [5.0, 15.0, 25.0, 60.0])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_suggestion_never_trips_the_libraries_own_resolution_warning(
    min_vel: float, max_freq: float, ndim: int
) -> None:
    """The suggested grid does not provoke ``check_points_per_wavelength``.

    Checked at the default ``accuracy=4`` and with the frequency the
    suggestion was designed for used as ``pml_freq``, which is the
    conservative case: a real ``pml_freq`` is the dominant frequency and so
    at or below ``max_freq``.
    """
    suggestion = suggest_discretisation(min_vel, max_freq, ndim, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_points_per_wavelength(min_vel, max_freq, suggestion.grid_spacing)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_suggestion_never_trips_the_warning_at_any_chosen_accuracy(ndim: int) -> None:
    """Also true when the accuracy is chosen, and at the 0.1% target."""
    for dispersion_error in _DISPERSION_ERRORS:
        suggestion = suggest_discretisation(
            1500.0, 25.0, ndim, dispersion_error=dispersion_error
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_points_per_wavelength(1500.0, 25.0, suggestion.grid_spacing)


def test_grid_spacing_gives_the_reported_points_per_wavelength() -> None:
    """``grid_spacing`` is the shortest wavelength over the reported sampling."""
    suggestion = suggest_discretisation(1500.0, 25.0, 2)
    wavelength = 1500.0 / 25.0
    assert len(suggestion.grid_spacing) == 2
    assert suggestion.grid_spacing[0] == suggestion.grid_spacing[1]
    assert suggestion.grid_spacing[0] == pytest.approx(
        wavelength / suggestion.points_per_wavelength
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("courant_number", [None, 0.3, 0.8])
def test_dt_is_the_cfl_limit_for_the_suggested_grid(
    ndim: int, courant_number: float
) -> None:
    """``dt`` is the largest step ``cfl_condition_n`` accepts unsubdivided.

    Compared against ``cfl_condition_n`` itself rather than against a second
    copy of the CFL expression: the two are algebraically equal but not
    bit-equal (the library divides by ``max_vel**2 + eps`` and multiplies by
    ``max_vel``), and the library's own arithmetic is what a propagator will
    apply to this ``dt``.
    """
    max_vel = 4500.0
    suggestion = suggest_discretisation(
        1500.0, 25.0, ndim, 4, courant_number=courant_number, max_vel=max_vel
    )
    accepted = cfl_condition_n(
        suggestion.grid_spacing,
        suggestion.dt,
        max_vel,
        courant_number=suggestion.courant_number,
    )
    assert accepted == (suggestion.dt, 1)
    bigger = cfl_condition_n(
        suggestion.grid_spacing,
        math.nextafter(suggestion.dt, math.inf),
        max_vel,
        courant_number=suggestion.courant_number,
    )
    assert bigger[1] == 2
    limit = suggestion.courant_number / (
        max_vel * math.sqrt(sum(1 / spacing**2 for spacing in suggestion.grid_spacing))
    )
    assert suggestion.dt == pytest.approx(limit, rel=1e-12)


def test_dt_defaults_to_assuming_a_homogeneous_model() -> None:
    """Without ``max_vel`` the CFL step is the one ``min_vel`` alone allows."""
    homogeneous = suggest_discretisation(1500.0, 25.0, 2, 4)
    explicit = suggest_discretisation(1500.0, 25.0, 2, 4, max_vel=1500.0)
    assert homogeneous.dt == pytest.approx(explicit.dt)
    faster = suggest_discretisation(1500.0, 25.0, 2, 4, max_vel=4500.0)
    assert faster.dt == pytest.approx(homogeneous.dt / 3, rel=1e-9)
    assert "assumed equal to min_vel" in homogeneous.note
    assert "assumed equal to min_vel" not in faster.note


def test_default_courant_number_is_the_libraries_default() -> None:
    """A defaulted suggestion reports ``DEFAULT_COURANT_NUMBER``."""
    assert (
        suggest_discretisation(1500.0, 25.0, 2).courant_number
        == deepwave.DEFAULT_COURANT_NUMBER
    )


def test_the_resolution_floor_binds_at_the_one_percent_target() -> None:
    """Orders 4, 6 and 8 all land on the floor, and say so."""
    for accuracy in (4, 6, 8):
        suggestion = suggest_discretisation(1500.0, 25.0, 2, accuracy)
        assert (
            suggestion.points_per_wavelength == _RESOLUTION_CHECK_POINTS_PER_WAVELENGTH
        )
        assert "floor" in suggestion.note
    order2 = suggest_discretisation(1500.0, 25.0, 2, 2)
    assert order2.points_per_wavelength == 12.8
    assert "floor" not in order2.note


def test_disabling_the_floor_follows_the_dispersion_table() -> None:
    """``min_points_per_wavelength=0`` gives the table's own numbers."""
    for accuracy in _ORDERS:
        suggestion = suggest_discretisation(
            1500.0, 25.0, 2, accuracy, min_points_per_wavelength=0
        )
        assert (
            suggestion.points_per_wavelength
            == _MIN_POINTS_PER_WAVELENGTH[False][0.01][accuracy]
        )


def test_a_finer_dispersion_target_gives_a_finer_grid() -> None:
    """0.1% asks for more cells than 1% at every order."""
    for accuracy in _ORDERS:
        coarse = suggest_discretisation(
            1500.0, 25.0, 2, accuracy, min_points_per_wavelength=0
        )
        fine = suggest_discretisation(
            1500.0,
            25.0,
            2,
            accuracy,
            dispersion_error=0.001,
            min_points_per_wavelength=0,
        )
        assert fine.grid_spacing[0] < coarse.grid_spacing[0]


def test_propagator_selects_the_staggered_column_and_scheme() -> None:
    """A staggered propagator gets the staggered table entry."""
    staggered = suggest_discretisation(
        1500.0, 25.0, 2, 4, propagator=deepwave.elastic, min_points_per_wavelength=0
    )
    regular = suggest_discretisation(
        1500.0, 25.0, 2, 4, propagator=deepwave.scalar, min_points_per_wavelength=0
    )
    assert staggered.points_per_wavelength == _MIN_POINTS_PER_WAVELENGTH[True][0.01][4]
    assert regular.points_per_wavelength == _MIN_POINTS_PER_WAVELENGTH[False][0.01][4]
    assert "elastic" in staggered.note
    assert "staggered-grid" in staggered.note
    assert "regular-grid" in regular.note


def test_an_unknown_propagator_takes_the_conservative_side_of_each_table() -> None:
    """No propagator: the regular sampling, the staggered Courant limit."""
    unknown = suggest_discretisation(1500.0, 25.0, 2, 4, min_points_per_wavelength=0)
    regular = suggest_discretisation(
        1500.0, 25.0, 2, 4, propagator=deepwave.scalar, min_points_per_wavelength=0
    )
    assert unknown.points_per_wavelength == regular.points_per_wavelength
    # The staggered critical Courant number is the lower of the two, so a
    # value between them is refused when the propagator is not named.
    staggered_limit = critical_courant_number(4, propagator=deepwave.elastic)
    regular_limit = critical_courant_number(4, propagator=deepwave.scalar)
    assert staggered_limit < regular_limit
    between = (staggered_limit + regular_limit) / 2
    with pytest.raises(deepwave.DeepwaveValueError, match="unstable"):
        suggest_discretisation(1500.0, 25.0, 2, 4, courant_number=between)
    suggest_discretisation(
        1500.0, 25.0, 2, 4, courant_number=between, propagator=deepwave.scalar
    )


def test_a_courant_number_above_critical_is_refused() -> None:
    """The requested Courant number is validated against the order's limit."""
    with pytest.raises(deepwave.DeepwaveValueError, match="unstable"):
        suggest_discretisation(1500.0, 25.0, 2, 8, courant_number=0.95)
    with pytest.raises(deepwave.DeepwaveValueError, match="every accuracy order"):
        suggest_discretisation(1500.0, 25.0, 2, courant_number=1.5)


def test_choosing_the_accuracy_excludes_unstable_orders() -> None:
    """With ``accuracy=None`` an order the Courant number breaks is skipped."""
    suggestion = suggest_discretisation(1500.0, 25.0, 2, courant_number=0.95)
    assert suggestion.accuracy == 2
    assert "only stable candidate" in suggestion.note


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_chosen_accuracy_minimises_the_documented_cost_model(ndim: int) -> None:
    """The chosen order is the one the stated cost model prefers."""
    for min_points_per_wavelength in (0.0, _RESOLUTION_CHECK_POINTS_PER_WAVELENGTH):
        suggestion = suggest_discretisation(
            1500.0,
            25.0,
            ndim,
            min_points_per_wavelength=min_points_per_wavelength,
        )
        costs = {}
        for accuracy in _ORDERS:
            points = max(
                _MIN_POINTS_PER_WAVELENGTH[False][0.01][accuracy],
                min_points_per_wavelength,
            )
            costs[accuracy] = points ** (ndim + 1) * _relative_cost_per_cell(
                accuracy, ndim
            )
        assert suggestion.accuracy == min(costs, key=lambda a: (costs[a], a))
        assert f"accuracy={suggestion.accuracy}" in suggestion.note
        assert "runner-up" in suggestion.note


def test_relative_cost_per_cell_is_one_at_the_default_order() -> None:
    """The cost weight is normalised to the default ``accuracy=4``."""
    for ndim in (1, 2, 3):
        assert _relative_cost_per_cell(4, ndim) == 1.0
        assert _relative_cost_per_cell(2, ndim) < 1.0 < _relative_cost_per_cell(8, ndim)


def test_returned_fields_and_note() -> None:
    """The NamedTuple carries the documented fields, and a readable note."""
    suggestion = suggest_discretisation(1500.0, 15.0, 2)
    assert suggestion._fields == (
        "grid_spacing",
        "accuracy",
        "dt",
        "points_per_wavelength",
        "courant_number",
        "note",
    )
    assert suggestion.note.endswith(".")
    assert "points per wavelength" in suggestion.note
    assert "Courant number" in suggestion.note


def test_the_suggestion_runs_a_propagator_without_warnings() -> None:
    """A propagator called with the suggestion is quiet about resolution."""
    torch = pytest.importorskip("torch")
    suggestion = suggest_discretisation(1500.0, 25.0, 2, propagator=deepwave.scalar)
    nt = 20
    v = torch.full((30, 30), 1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deepwave.scalar(
            v,
            suggestion.grid_spacing,
            suggestion.dt,
            source_amplitudes=torch.zeros(1, 1, nt),
            source_locations=torch.tensor([[[15, 15]]]),
            pml_freq=25.0,
            accuracy=suggestion.accuracy,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_vel": 0.0}, "min_vel must be a finite positive"),
        ({"min_vel": -1.0}, "min_vel must be a finite positive"),
        ({"min_vel": float("inf")}, "min_vel must be a finite positive"),
        ({"max_freq": 0.0}, "max_freq must be a finite positive"),
        ({"max_freq": float("nan")}, "max_freq must be a finite positive"),
        ({"ndim": 0}, "ndim must be 1, 2 or 3"),
        ({"ndim": 4}, "ndim must be 1, 2 or 3"),
        ({"accuracy": 3}, "accuracy must be 2, 4, 6, 8 or None"),
        ({"accuracy": 0}, "accuracy must be 2, 4, 6, 8 or None"),
        ({"dispersion_error": 0.05}, "dispersion_error must be 0.01"),
        ({"max_vel": 1000.0}, "cannot be slower than the slowest"),
        ({"max_vel": -1.0}, "max_vel must be a finite positive"),
        ({"courant_number": 0.0}, "courant_number must be a finite positive"),
        (
            {"min_points_per_wavelength": -1.0},
            "min_points_per_wavelength must be a finite non-negative",
        ),
    ],
)
def test_invalid_arguments_raise_with_the_value_named(
    kwargs: dict[str, Any], match: str
) -> None:
    """Every rejection is a ``DeepwaveValueError`` naming what it saw."""
    call = {"min_vel": 1500.0, "max_freq": 25.0, "ndim": 2, **kwargs}
    with pytest.raises(deepwave.DeepwaveValueError, match=match):
        suggest_discretisation(**call)


def test_non_numeric_arguments_raise_a_type_error() -> None:
    """A non-number gets ``DeepwaveTypeError``, not a bare ``TypeError``."""
    with pytest.raises(deepwave.DeepwaveTypeError, match="min_vel must be a number"):
        suggest_discretisation("fast", 25.0, 2)  # type: ignore[arg-type]
    with pytest.raises(deepwave.DeepwaveTypeError, match="max_freq must be a number"):
        suggest_discretisation(1500.0, None, 2)  # type: ignore[arg-type]


def test_ndim_is_checked_against_the_propagator() -> None:
    """A propagator that cannot run in ``ndim`` dimensions is rejected."""
    with pytest.raises(deepwave.DeepwaveValueError, match="supports 2, 3 dimensions"):
        suggest_discretisation(1500.0, 25.0, 1, propagator=deepwave.elastic_vti)


def test_bool_is_not_accepted_as_ndim() -> None:
    """``True`` is not a dimensionality, even though ``True == 1``."""
    with pytest.raises(deepwave.DeepwaveValueError, match="ndim must be 1, 2 or 3"):
        suggest_discretisation(1500.0, 25.0, True)  # type: ignore[arg-type]

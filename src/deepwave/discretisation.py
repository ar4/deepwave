"""Grid, time-step and resampling arithmetic.

The helpers for choosing and checking a discretisation: the CFL time-step
calculators, the critical Courant numbers they are checked against, the
grid-and-time-step suggestion that combines them, and the band-limited
resampling and tapering routines the propagators use internally to move a
signal between the caller's time step and the internal one.

:doc:`/discretisation` is the discussion of what these compute and why.

Most of them previously had no public home of their own and were reachable
only as ``deepwave.common.*``; that spelling still works but is deprecated
(see :mod:`deepwave.common`).
"""

import math
from typing import Any, Callable, NamedTuple, Optional, Union

from deepwave._common import _DEFAULT_COURANT_NUMBER
from deepwave._common import _critical_courant_number as _critical_courant_number
from deepwave._common import (
    cfl_condition as cfl_condition,
)
from deepwave._common import (
    cfl_condition_n as cfl_condition_n,
)
from deepwave._common import (
    cosine_taper_end as cosine_taper_end,
)
from deepwave._common import (
    downsample as downsample,
)
from deepwave._common import (
    upsample as upsample,
)
from deepwave._exceptions import (
    DeepwaveTypeError as _DeepwaveTypeError,
)
from deepwave._exceptions import (
    DeepwaveValueError as _DeepwaveValueError,
)
from deepwave.types import Accuracy as _Accuracy

__all__ = [
    "DEFAULT_COURANT_NUMBER",
    "Discretisation",
    "cfl_condition",
    "cfl_condition_n",
    "cosine_taper_end",
    "critical_courant_number",
    "downsample",
    "suggest_discretisation",
    "upsample",
]

#: The Courant number Deepwave targets when ``courant_number`` is not given.
#:
#: Every propagator, and :func:`deepwave.discretisation.cfl_condition_n`,
#: uses this value to choose the internal time step when the caller leaves
#: ``courant_number=None``. It is deliberately conservative -- the same
#: number for every ``accuracy`` order and every dimensionality, well below
#: the critical Courant number :func:`critical_courant_number` reports for
#: any of them -- because it is the value Deepwave has always used, and a run
#: that does not set ``courant_number`` is reproduced bit-for-bit by passing
#: this constant explicitly.
#:
#: It is exported so that the default can be named rather than hard-coded:
#: ``courant_number=1.2 * deepwave.DEFAULT_COURANT_NUMBER`` says what it
#: means, and ``RunInfo.courant_number`` can be compared against it to tell
#: a defaulted run from an explicit one. Raising ``courant_number`` above it
#: permits a larger internal time step (fewer steps, faster) at the cost of
#: stability margin; the value is validated against
#: :func:`critical_courant_number` and rejected if it would be unstable.
#:
#: Stability: stable (see :doc:`/stability`).
DEFAULT_COURANT_NUMBER = _DEFAULT_COURANT_NUMBER

# Documented and supported here, so they report this module rather than the
# private file the definitions sit in.
for _fn in (cfl_condition, cfl_condition_n, cosine_taper_end, downsample, upsample):
    _fn.__module__ = "deepwave.discretisation"
del _fn


def critical_courant_number(
    accuracy: _Accuracy,
    *,
    propagator: Optional[Union[str, Callable[..., Any], type]] = None,
) -> float:
    """Returns the maximum stable Courant number for a leapfrog scheme.

    These are the homogeneous critical Courant numbers C_crit re-derived from the actual
    finite-difference stencil coefficients (see the appendix of
    :doc:`/discretisation`). A propagation is stable when
    ``dt * v_max * sqrt(sum_i 1/dx_i^2) <= C_crit``.
    Staggered propagators also check the density-weighted material operator;
    ``v_max`` must cover that conservative bound for heterogeneous models.

    The critical number depends on whether the propagator uses a staggered
    grid, so name the propagator and let Deepwave answer that::

        >>> import deepwave
        >>> deepwave.discretisation.critical_courant_number(
        ...     4, propagator=deepwave.elastic
        ... )
        0.857142857142857...

    Args:
        accuracy: The spatial finite-difference accuracy order (2, 4, 6,
            or 8).
        propagator: The propagator the number is for, as the functional
            (``deepwave.elastic``), the Module class (``deepwave.Elastic``)
            or the name (``"elastic"``).

    Returns:
        The critical (maximum stable) Courant number.

    Raises:
        DeepwaveValueError: If ``propagator`` is not given, or if ``accuracy``
            is not one of 2, 4, 6, 8.
        DeepwaveTypeError: If ``propagator`` does not name a propagator.

    """
    resolved: Optional[bool] = None
    if propagator is not None:
        # Imported here rather than at module scope: ``introspection`` pulls
        # in the engine, and this module is imported from ``deepwave``'s own
        # ``__init__``.
        from deepwave.introspection import _resolve_spec

        resolved = _resolve_spec(propagator).staggered
    if resolved is None:
        from deepwave import _engine

        raise _DeepwaveValueError(
            "critical_courant_number needs to know which propagator the "
            "number is for, because the staggered velocity-stress schemes "
            "and the regular-grid schemes have different critical Courant "
            "numbers. Pass propagator=<propagator>, where <propagator> is "
            "the functional (deepwave.elastic), the Module class "
            "(deepwave.Elastic) or the name. The propagators are: "
            f"{', '.join(_engine.PROPAGATOR_NAMES)}."
        )
    return _critical_courant_number(accuracy, resolved)


# ---------------------------------------------------------------------------
# Grid and time-step suggestion
# ---------------------------------------------------------------------------

_ORDERS: tuple[_Accuracy, ...] = (2, 4, 6, 8)

# The dispersion-error targets the points-per-wavelength table is tabulated
# at. Anything else would need the phase-velocity curves re-solved, so these
# two are what ``dispersion_error`` accepts.
_DISPERSION_ERRORS: tuple[float, ...] = (0.01, 0.001)

# Minimum points per wavelength for axial propagation, keyed by
# ``staggered`` -> dispersion-error target -> accuracy order.
#
# These are the numbers in :doc:`/discretisation`'s "Minimum
# points-per-wavelength (axial)" table, and that table is generated from this
# constant (see ``_points_per_wavelength_rst`` below and ``docs/conf.py``), so
# there is one copy of them. They are the smallest points-per-wavelength G at
# which the axial phase-velocity error |v_num/v - 1| of the scheme's own
# stencil first falls to the target and stays there, tabulated to the nearest
# 0.1; ``tests/test_discretisation.py`` re-derives them from the stencil
# coefficients.
_MIN_POINTS_PER_WAVELENGTH: dict[bool, dict[float, dict[_Accuracy, float]]] = {
    False: {
        0.01: {2: 12.8, 4: 5.3, 6: 3.9, 8: 3.4},
        0.001: {2: 40.6, 4: 9.6, 6: 6.0, 8: 4.8},
    },
    True: {
        0.01: {2: 12.8, 4: 5.1, 6: 3.8, 8: 3.3},
        0.001: {2: 40.6, 4: 9.2, 6: 5.8, 8: 4.6},
    },
}

# The points-per-wavelength floor that Deepwave's own runtime resolution check
# imposes: ``check_points_per_wavelength`` warns whenever a propagator is
# given fewer than six cells per wavelength. It is a single accuracy-agnostic
# number, calibrated for the default ``accuracy=4``, so for orders 4, 6 and 8
# it is stricter than the 1% dispersion criterion above.
# ``tests/test_discretisation.py`` pins it against the check's real behaviour.
_RESOLUTION_CHECK_POINTS_PER_WAVELENGTH = 6.0

# The largest trial time step that is certainly at or below the CFL limit for
# any grid Deepwave is used on, and the factor the search grows it by. See
# ``_cfl_limited_dt``.
_DT_SEARCH_START = 1e-9
_DT_SEARCH_BISECTIONS = 60


class Discretisation(NamedTuple):
    """What :func:`suggest_discretisation` returns.

    Stability: provisional (see :doc:`/stability`).

    Attributes:
        grid_spacing: The suggested cell spacing, one entry per dimension
            (equal in every dimension), in the model's length unit. Pass it
            straight to a propagator's ``grid_spacing``.
        accuracy: The spatial finite-difference order the suggestion is for
            (the one that was requested, or the one that was chosen).
        dt: The largest time step, in seconds, that satisfies the CFL
            condition on this grid at ``courant_number``.
        points_per_wavelength: Cells per shortest wavelength that
            ``grid_spacing`` actually achieves.
        courant_number: The Courant number ``dt`` was computed at.
        note: A human-readable sentence saying which constraint was binding,
            what the runner-up accuracy was, and what was assumed.

    """

    grid_spacing: tuple[float, ...]
    accuracy: _Accuracy
    dt: float
    points_per_wavelength: float
    courant_number: float
    note: str


def _positive_float(name: str, value: Any) -> float:
    """Validates one of the positive scalar arguments of the suggestion."""
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise _DeepwaveTypeError(
            f"suggest_discretisation's {name} must be a number, but got "
            f"{value!r} of type {type(value).__name__}."
        ) from e
    if not math.isfinite(out) or out <= 0:
        raise _DeepwaveValueError(
            f"suggest_discretisation's {name} must be a finite positive "
            f"number, but got {value!r}."
        )
    return out


def _relative_cost_per_cell(accuracy: _Accuracy, ndim: int) -> float:
    """A rough relative cost of one cell-update at this accuracy order.

    A regular-grid second-derivative stencil of order ``a`` reads ``a``
    neighbours (``a/2`` on each side) along each of ``ndim`` axes, plus the
    centre cell and the two time levels the leapfrog combines, so an update
    costs roughly ``ndim * a + 3`` multiply-adds. Normalised so that the
    default ``accuracy=4`` costs 1. This counts arithmetic only: it knows
    nothing about cache behaviour, vector width or the halo the wider stencil
    needs, all of which make the high orders relatively dearer in practice.
    """
    return (ndim * accuracy + 3) / (ndim * 4 + 3)


def _cfl_limited_dt(
    grid_spacing: tuple[float, ...], max_vel: float, courant_number: float
) -> float:
    """The largest time step obeying the CFL condition, from ``cfl_condition_n``.

    :func:`deepwave.discretisation.cfl_condition_n` answers "given this ``dt``,
    what internal step obeys the CFL condition"; a suggestion needs the
    reverse. Rather than
    keeping a second copy of the CFL arithmetic here, the limit is found by
    using ``cfl_condition_n`` itself as the oracle: a trial step that comes
    back with ``step_ratio == 1`` was at or below the limit, one that comes
    back with a larger ratio was above it. The search grows a certainly-small
    trial step until it is first rejected and then bisects, so no trial ever
    reaches the ``step_ratio`` at which ``cfl_condition_n`` offers its
    reduce-your-``dt`` advice -- that advice is for a real propagator call,
    and firing it from a probe would spend the once-per-process budget that
    the user's own call needs.
    """

    def obeys(trial: float) -> bool:
        return (
            cfl_condition_n(
                grid_spacing, trial, max_vel, courant_number=courant_number
            )[1]
            == 1
        )

    low = _DT_SEARCH_START
    while not obeys(low):
        # Only reachable for a grid so fine, or a velocity so large, that the
        # stable step is under a nanosecond. Kept as a loop rather than an
        # error so that such a case still gets an answer.
        low /= 1024.0
    high = low * 2.0
    while obeys(high):
        low = high
        high *= 2.0
    for _ in range(_DT_SEARCH_BISECTIONS):
        middle = (low + high) / 2
        if middle <= low or middle >= high:
            break
        if obeys(middle):
            low = middle
        else:
            high = middle
    return low


def _points_per_wavelength_rst() -> str:
    """Renders ``_MIN_POINTS_PER_WAVELENGTH`` as the docs' list-table.

    Called by ``docs/conf.py`` on ``builder-inited`` to write
    ``docs/_generated/points_per_wavelength_table.rst``, which
    :doc:`/discretisation` includes, so the page and the constant this module
    computes from cannot drift apart.
    """
    lines = [
        ".. list-table:: Minimum points-per-wavelength (axial), computed from "
        "the stencils",
        "   :header-rows: 1",
        "   :widths: 10 22 22 22 22",
        "",
        "   * - order",
        "     - regular, 1%",
        "     - regular, 0.1%",
        "     - staggered, 1%",
        "     - staggered, 0.1%",
    ]
    for accuracy in _ORDERS:
        lines.append(f"   * - {accuracy}")
        lines.extend(
            f"     - {_MIN_POINTS_PER_WAVELENGTH[staggered][error][accuracy]}"
            for staggered in (False, True)
            for error in _DISPERSION_ERRORS
        )
    return "\n".join(lines) + "\n"


def suggest_discretisation(
    min_vel: float,
    max_freq: float,
    ndim: int,
    accuracy: Optional[_Accuracy] = None,
    *,
    propagator: Optional[Union[str, Callable[..., Any], type]] = None,
    courant_number: Optional[float] = None,
    max_vel: Optional[float] = None,
    dispersion_error: float = 0.01,
    min_points_per_wavelength: float = _RESOLUTION_CHECK_POINTS_PER_WAVELENGTH,
) -> Discretisation:
    r"""Suggests a grid spacing and time step for a wave you want to model.

    Given the slowest wavespeed in your model and the highest frequency you
    need to propagate, this returns a cell spacing fine enough to carry that
    wave without excessive numerical dispersion, and the largest time step
    that is then stable on it. It is a starting point to sanity-check your
    own numbers against, not an oracle: it knows nothing about your
    geometry, your PML, or how far your waves have to travel.

    Where the numbers come from, none of which is re-derived here:

    * The shortest wavelength is
      :math:`\lambda_\mathrm{min} = v_\mathrm{min}/f_\mathrm{max}` and the
      cells per wavelength are :math:`\lambda_\mathrm{min}/\Delta x` (see
      :doc:`/accuracy`).
    * The minimum points per wavelength for each order is the tabulated
      "Minimum points-per-wavelength (axial)" table in
      :doc:`/discretisation`, which is the smallest sampling at which that
      stencil's **axial** phase-velocity error falls to ``dispersion_error``.
      Axial is the conservative direction: dispersion is not isotropic, and
      at 5 points per wavelength the regular 2-D order-2 axial phase velocity
      is :math:`0.936\,v` while the diagonal is :math:`0.967\,v`, so a
      diagonally travelling wave is slightly *better* resolved than this
      suggestion promises.
    * ``dt`` is the CFL limit from :func:`deepwave.discretisation.cfl_condition_n`, at
      :data:`DEFAULT_COURANT_NUMBER` unless you pass ``courant_number``. The
      number of dimensions enters only through that condition's
      :math:`\sqrt{\sum_i 1/\Delta x_i^2}` term.

    **Two criteria, and which one binds.** Deepwave's own runtime check warns
    when a propagator is given fewer than six cells per wavelength, and that
    single number is calibrated for the default ``accuracy=4``. It is
    stricter than the 1% axial-dispersion criterion for orders 4, 6 and 8
    (which ask for only 5.3, 3.9 and 3.4), because it is also standing in for
    everything the plane-wave dispersion analysis leaves out: how a
    point-source wavefront, a sharp model contrast and the PML are sampled.
    The suggestion therefore respects **both**, taking whichever asks for
    more cells, so that what it hands you never immediately provokes
    Deepwave's own warning. ``note`` says which one bound. The consequence at
    the default 1% target is that orders 4, 6 and 8 all land on the same
    six-cell grid, and the coarser grids the higher orders could otherwise
    run on are not offered; lower ``min_points_per_wavelength`` to see them,
    but expect the warning when you then call a propagator.

    **Choosing the accuracy.** With ``accuracy=None`` the order that
    minimises a rough estimate of the total work is chosen. Cells scale as
    :math:`(1/\Delta x)^\mathrm{ndim}` and time steps as
    :math:`1/\Delta t \propto 1/\Delta x`, so the work goes as
    ``points_per_wavelength ** (ndim + 1)`` times a relative cost per cell of
    ``(ndim * accuracy + 3) / (ndim * 4 + 3)`` -- the multiply-adds a stencil
    of that order needs, normalised to 1 at ``accuracy=4``. That weight is a
    rough relative figure and nothing more: it counts arithmetic and ignores
    cache behaviour, vector width and the wider halo, all of which make high
    orders dearer in practice than it says. Treat the choice as a suggestion
    to time, not a measurement. The chosen order and the runner-up, with the
    ratio between their estimates, are reported in ``note``.

    Args:
        min_vel: The slowest (non-zero) wavespeed in the model, in the same
            length unit as the grid spacing you want back, per second.
        max_freq: The highest frequency, in Hz, that must be propagated
            without excessive dispersion. This is the top of the wavelet's
            significant band, not its dominant frequency.
        ndim: The number of spatial dimensions (1, 2 or 3).
        accuracy: The spatial finite-difference order (2, 4, 6 or 8), or
            ``None`` (the default) to choose the cheapest by the cost model
            above.
        propagator: The propagator the suggestion is for, as the functional
            (``deepwave.elastic``), the Module class (``deepwave.Elastic``)
            or the name (``"elastic"``). It selects the staggered or
            regular-grid column of the dispersion table and the matching
            critical Courant number. With ``None`` (the default) the
            conservative value of each is used: the regular-grid dispersion
            numbers, which ask for slightly more cells, and the staggered
            critical Courant number, which is slightly lower.
        courant_number: The Courant number to compute ``dt`` at. ``None``
            (the default) uses :data:`DEFAULT_COURANT_NUMBER`. A value above
            the critical Courant number for the order is rejected; when
            ``accuracy`` is ``None``, orders whose critical number it exceeds
            are simply not considered.
        max_vel: The fastest wavespeed in the model, in the same length unit
            per second as ``min_vel``, which is what the CFL condition is set
            by. Defaults to ``min_vel``, i.e. to assuming a
            homogeneous model -- **pass it** for any model with faster
            material in it, or the suggested ``dt`` will be too large for the
            grid and a propagator will have to sub-step it.
        dispersion_error: The axial phase-velocity error to design the grid
            for: ``0.01`` (1%, the default) or ``0.001`` (0.1%). The error is
            per wavelength travelled and accumulates, so a 1% error slips a
            full cycle over ~100 wavelengths of propagation; use 0.001 for
            long models.
        min_points_per_wavelength: The floor, in cells per wavelength, applied
            to the dispersion criterion, defaulting to the six cells per
            wavelength that Deepwave's own runtime check wants (see above).
            Pass ``0`` to follow the dispersion table alone.

    Returns:
        A :class:`Discretisation` naming the suggested ``grid_spacing``,
        ``accuracy``, ``dt``, the ``points_per_wavelength`` and
        ``courant_number`` they correspond to, and a ``note``.

    Raises:
        DeepwaveValueError: If ``min_vel``, ``max_freq``, ``max_vel`` or
            ``courant_number`` is not a finite positive number, if
            ``min_points_per_wavelength`` is not a finite non-negative
            number, if ``max_vel`` is below ``min_vel``, if ``ndim`` is not
            1, 2 or 3, if ``accuracy`` is not 2, 4, 6, 8 or ``None``, if
            ``dispersion_error`` is neither 0.01 nor 0.001, or if
            ``courant_number`` exceeds the critical Courant number of the
            requested order (or of every order, when one is being chosen).
        DeepwaveTypeError: If one of the numeric arguments is not a number,
            or ``propagator`` does not name a propagator.

    Examples:
        >>> import deepwave
        >>> suggestion = deepwave.suggest_discretisation(1500, 25, 2)
        >>> suggestion.grid_spacing
        (10.0, 10.0)
        >>> suggestion.accuracy
        4

    """
    min_vel = _positive_float("min_vel", min_vel)
    max_freq = _positive_float("max_freq", max_freq)
    if not isinstance(ndim, int) or isinstance(ndim, bool) or ndim not in (1, 2, 3):
        raise _DeepwaveValueError(
            f"suggest_discretisation's ndim must be 1, 2 or 3, but got {ndim!r}."
        )
    if accuracy is not None and accuracy not in _ORDERS:
        raise _DeepwaveValueError(
            "suggest_discretisation's accuracy must be 2, 4, 6, 8 or None "
            f"(to choose one), but got {accuracy!r}."
        )
    if dispersion_error not in _DISPERSION_ERRORS:
        raise _DeepwaveValueError(
            "suggest_discretisation's dispersion_error must be 0.01 (1%) or "
            f"0.001 (0.1%), the two targets the points-per-wavelength table "
            f"in the documentation is computed at, but got "
            f"{dispersion_error!r}."
        )
    try:
        floor = float(min_points_per_wavelength)
    except (TypeError, ValueError) as e:
        raise _DeepwaveTypeError(
            "suggest_discretisation's min_points_per_wavelength must be a "
            f"number, but got {min_points_per_wavelength!r} of type "
            f"{type(min_points_per_wavelength).__name__}."
        ) from e
    if not math.isfinite(floor) or floor < 0:
        raise _DeepwaveValueError(
            "suggest_discretisation's min_points_per_wavelength must be a "
            f"finite non-negative number, but got {min_points_per_wavelength!r}."
        )
    if max_vel is None:
        fastest = min_vel
        assumed_max_vel = True
    else:
        fastest = _positive_float("max_vel", max_vel)
        assumed_max_vel = False
        if fastest < min_vel:
            raise _DeepwaveValueError(
                f"suggest_discretisation was given max_vel={fastest} below "
                f"min_vel={min_vel}; max_vel is the fastest wavespeed in the "
                "model and cannot be slower than the slowest."
            )
    if courant_number is None:
        courant = DEFAULT_COURANT_NUMBER
    else:
        courant = _positive_float("courant_number", courant_number)

    staggered: Optional[bool] = None
    propagator_name: Optional[str] = None
    if propagator is not None:
        # Imported here rather than at module scope for the same reason as in
        # ``critical_courant_number``: ``introspection`` pulls in the engine,
        # and this module is imported from ``deepwave``'s own ``__init__``.
        from deepwave.introspection import _resolve_spec

        spec = _resolve_spec(propagator)
        staggered = spec.staggered
        propagator_name = spec.name
        if ndim not in spec.ndims:
            raise _DeepwaveValueError(
                f"suggest_discretisation was given ndim={ndim} for "
                f"propagator {spec.name!r}, which supports "
                f"{', '.join(str(n) for n in spec.ndims)} dimensions."
            )
    # With the propagator unknown, take the conservative side of each table:
    # the regular-grid scheme needs marginally more points per wavelength,
    # and the staggered scheme has the marginally lower critical Courant
    # number, so neither answer can be too generous for the other scheme.
    dispersion_table = _MIN_POINTS_PER_WAVELENGTH[bool(staggered)][dispersion_error]
    courant_staggered = True if staggered is None else staggered

    def target(order: _Accuracy) -> float:
        return max(dispersion_table[order], floor)

    candidates = [
        order
        for order in _ORDERS
        if courant <= _critical_courant_number(order, courant_staggered)
    ]
    if not candidates:
        raise _DeepwaveValueError(
            f"suggest_discretisation was given courant_number={courant}, "
            "which is above the critical Courant number of every accuracy "
            "order (the largest is "
            f"{max(_critical_courant_number(o, courant_staggered) for o in _ORDERS)}"
            "); the propagation would be unstable at any order. Lower "
            "courant_number."
        )
    if accuracy is None:
        ranked = sorted(
            candidates,
            key=lambda order: (
                target(order) ** (ndim + 1) * _relative_cost_per_cell(order, ndim),
                order,
            ),
        )
        chosen = ranked[0]
        runner_up: Optional[_Accuracy] = ranked[1] if len(ranked) > 1 else None
    else:
        if accuracy not in candidates:
            raise _DeepwaveValueError(
                f"suggest_discretisation was given courant_number={courant} "
                f"with accuracy={accuracy}, but the critical Courant number "
                "for that order is "
                f"{_critical_courant_number(accuracy, courant_staggered)}; the "
                "propagation would be unstable. Lower courant_number, or "
                "lower accuracy."
            )
        chosen = accuracy
        runner_up = None

    points_per_wavelength = target(chosen)
    grid_spacing = ((min_vel / max_freq) / points_per_wavelength,) * ndim
    dt = _cfl_limited_dt(grid_spacing, fastest, courant)

    parts = []
    if accuracy is None:
        cost = {
            order: target(order) ** (ndim + 1) * _relative_cost_per_cell(order, ndim)
            for order in candidates
        }
        if runner_up is None:
            parts.append(f"accuracy={chosen} is the only stable candidate order")
        else:
            parts.append(
                f"accuracy={chosen} estimated cheapest (runner-up "
                f"{runner_up}, ~{cost[runner_up] / cost[chosen]:.2f}x the "
                "estimated work; the estimate is a heuristic, so time both "
                "if it matters)"
            )
    else:
        parts.append(f"accuracy={chosen} as requested")
    tabulated = dispersion_table[chosen]
    if points_per_wavelength > tabulated:
        parts.append(
            f"{points_per_wavelength:g} points per wavelength, set by the "
            f"{floor:g}-point floor rather than by the "
            f"{dispersion_error * 100:g}% axial-dispersion criterion, which asks "
            f"for {tabulated:g}"
        )
    else:
        parts.append(
            f"{points_per_wavelength:g} points per wavelength, the "
            f"{dispersion_error * 100:g}% axial-dispersion criterion for this order"
        )
    parts.append(
        f"dt from the CFL condition at Courant number {courant:g} with "
        f"max_vel={fastest:g}"
        + (
            " (assumed equal to min_vel: pass max_vel= if the model has "
            "faster material in it)"
            if assumed_max_vel
            else ""
        )
    )
    if propagator_name is not None:
        parts.append(
            f"for {propagator_name}, a "
            f"{'staggered' if staggered else 'regular'}-grid propagator"
        )
    note = "; ".join(parts) + "."

    return Discretisation(
        grid_spacing=grid_spacing,
        accuracy=chosen,
        dt=dt,
        points_per_wavelength=points_per_wavelength,
        courant_number=courant,
        note=note,
    )

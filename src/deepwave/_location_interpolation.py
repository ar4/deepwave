"""Interpolation of sources and receivers at arbitrary locations onto a grid."""

import math
from collections.abc import Sequence
from typing import Optional, Union, cast

import torch

from deepwave import _common
from deepwave._exceptions import (
    DEPRECATION_REMOVAL_VERSION as _DEPRECATION_REMOVAL_VERSION,
)
from deepwave._exceptions import (
    DeepwaveNotImplementedError,
    DeepwaveRuntimeError,
    DeepwaveTypeError,
    DeepwaveValueError,
    MigrationWarning,
)
from deepwave.types import ON_NODE_TOLERANCE

__all__ = ["Hicks"]

# The node-proximity tolerance, in cells. Single-sourced from the public
# ``deepwave.ON_NODE_TOLERANCE`` so that the value the propagators route on
# and the value users are told about cannot drift apart. Its numeric value
# (1e-5) is unchanged, and is also ``Hicks``' default ``eps`` -- the same
# default v0.0.27 shipped.
_DEFAULT_EPS = ON_NODE_TOLERANCE

_LOCATION_INTERPOLATION_MODES = ("auto", "hicks", "none")

#: The largest fractional part, in cells, that Deepwave 0.0.27 accepted in a
#: float source or receiver coordinate. It truncated anything within this of
#: the cell below and rejected the rest, so a coordinate whose fractional part
#: lies in ``(eps, this]`` is exactly the one that a released Deepwave took and
#: that this release now interpolates instead -- the same call, different
#: numbers. Reproduced here as the migration warning's condition, not as a
#: tolerance anything routes on.
_V0027_TRUNCATION_LIMIT = 0.1

#: The once-per-process key of the truncated-float-coordinate migration
#: warning. One key for all nine propagators and both sides of the survey:
#: the cause is the change of meaning, which is the same wherever it is met.
_TRUNCATED_COORDINATE_CAUSE = "migration:truncated-float-coordinate"


def _offset_tensor(
    offset: Optional[list[float]],
    locations: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build a broadcastable offset tensor in the dtype/device of ``locations``."""
    if offset is None or all(o == 0 for o in offset):
        return None
    return torch.tensor(offset, dtype=locations.dtype, device=locations.device)


def _has_off_node_coord(
    locations: torch.Tensor,
    offset: Optional[list[float]],
    eps: float,
    *,
    warn_truncated: bool = False,
) -> bool:
    """Whether any non-ignored coordinate lies more than ``eps`` off its node.

    The test is made in the component's own index space: a coordinate is "on a
    node" when ``location - offset`` is within ``eps`` of an integer (see
    :func:`_should_interpolate`). Ignored coordinates (the ``IGNORE_LOCATION``
    sentinel) are excluded so they never count as off-node.

    With ``warn_truncated``, the same reduction also answers whether any
    coordinate is one Deepwave 0.0.27 accepted and truncated (see
    :func:`_warn_truncated_coordinate_migration`), and the warning is emitted
    from here. The two questions are answered by one fused reduction and a
    single transfer to the host, so asking the second costs no additional
    device synchronisation: on CUDA this function performs exactly one either
    way.
    """
    shifted = locations
    offset_t = _offset_tensor(offset, locations)
    if offset_t is not None:
        shifted = locations - offset_t
    frac = (shifted - shifted.round()).abs()
    valid = locations != float(_common.IGNORE_LOCATION)
    off_node = (frac > eps) & valid
    if not warn_truncated:
        return bool(off_node.any())
    # The fractional part Deepwave 0.0.27 would have discarded, measured on
    # the coordinate the user passed rather than on the component-shifted one:
    # that release had no component offset, and truncated what it was given.
    truncated = locations - locations.floor()
    in_band = (truncated > eps) & (truncated <= _V0027_TRUNCATION_LIMIT)
    answers = torch.stack((off_node.any(), (off_node & in_band).any()))
    any_off_node, any_truncated = answers.tolist()
    if any_truncated:
        _warn_truncated_coordinate_migration(eps)
    return bool(any_off_node)


def _warn_truncated_coordinate_migration(eps: float) -> None:
    """Warns that a float coordinate is interpolated where 0.0.27 truncated it.

    Deepwave 0.0.27 accepted a float source or receiver coordinate whose
    fractional part was at most :data:`_V0027_TRUNCATION_LIMIT` cells and
    silently truncated it to the cell below, and rejected anything further
    off-node. Such a coordinate is now placed where it was asked for, by
    interpolating it (see :class:`Hicks`), so the same call gives different
    numbers from that release. The warning says so once, and names the one
    expression that reproduces the old placement exactly.

    Args:
        eps: The on-node tolerance this call routed on, below which the
            coordinate is snapped to its node and nothing changes.

    """
    _common._warn_once_per_cause(  # noqa: SLF001
        _TRUNCATED_COORDINATE_CAUSE,
        "a float source or receiver coordinate lies more than "
        f"{eps} cells off its grid node but no more than "
        f"{_V0027_TRUNCATION_LIMIT} cells past the node below it. Deepwave "
        "0.0.27 and earlier accepted such a coordinate and truncated it to "
        "that node; it is now interpolated onto the grid (deepwave.Hicks), so "
        "it is modelled where you put it and this call's output differs from "
        "what that release produced. Pass locations.floor().long() to "
        "reproduce the old placement exactly, or "
        "location_interpolation='none' to have an off-node coordinate "
        "rejected instead (either silences this message, a migration warning "
        "that is removed in Deepwave "
        f"{_DEPRECATION_REMOVAL_VERSION}). Silence it with: "
        'warnings.filterwarnings("ignore", '
        "category=deepwave.MigrationWarning).",
        MigrationWarning,
        stacklevel=4,
    )


def _warn_hicks_dtype_migration(default_dtype: torch.dtype) -> None:
    """Warns that omitted ``Hicks(dtype=)`` follows the default dtype now.

    Deepwave 0.0.27 built the interpolation weights in ``float32`` whatever
    ``torch.get_default_dtype`` said; the default is now that dtype. Under the
    stock ``float32`` default the two are the same call and nothing is said.
    Under any other default the weights are built in that dtype instead, which
    is a different rounding path for the same window, so it is said out loud
    once, with the one argument that restores the old build exactly.

    This warning is about the rounding of the weights only. Whether the
    *window* -- the integer cells the weights are applied to -- differs from
    0.0.27's is a separate question, answered per window by
    :func:`_warn_if_stencil_changed`, whose comparison is made in the dtype
    that release would have used rather than in the current default.

    Args:
        default_dtype: The dtype the omitted argument resolved to.

    """
    if default_dtype is torch.float32:
        return
    _common._warn_once_per_cause(  # noqa: SLF001
        f"migration:hicks-default-dtype:{default_dtype}",
        f"Hicks was built with no dtype= and the default dtype is "
        f"{default_dtype}, so its interpolation weights are built in "
        f"{default_dtype}. Deepwave 0.0.27 and earlier always built them in "
        "torch.float32 and promoted them afterwards, so this call's weights, "
        "and any receiver amplitude or gradient computed through them, are "
        "rounded differently from what that release produced: the same window "
        "of cells, with the weights on them computed in a different dtype. A "
        "window whose cells differ from that release's is a separate change, "
        "with a migration warning of its own. Pass "
        "dtype=torch.float32 to reproduce the old weights exactly, or "
        f"dtype={default_dtype} to say that the new ones are what you want "
        "(either silences this message, a migration warning that is removed "
        f"in Deepwave {_DEPRECATION_REMOVAL_VERSION}). Silence it with: "
        'warnings.filterwarnings("ignore", '
        "category=deepwave.MigrationWarning).",
        MigrationWarning,
        stacklevel=3,
    )


def _should_interpolate(
    locations: Optional[torch.Tensor],
    mode: str = "auto",
    eps: float = _DEFAULT_EPS,
    offset: Optional[list[float]] = None,
) -> bool:
    """Decide whether source/receiver locations need Hicks interpolation.

    This implements the auto-detection policy used by the propagators when float
    coordinates are provided:

    - ``"none"``: never interpolate. The caller must provide coordinates that
      lie exactly on grid nodes (integer cells, or a float Tensor whose
      coordinates are all on-node); :func:`_resolve_interpolator` raises a
      ``ValueError`` for an off-node float coordinate rather than silently
      rounding it (see :func:`_resolve_interpolator`).
    - ``"hicks"``: always interpolate.
    - ``"auto"`` (default): interpolate only if ``locations`` is a floating-point
      tensor with at least one (non-ignored) coordinate that does not lie on a
      node of the field component being placed. On-node coordinates can be
      snapped to the exact integer path, giving behaviour identical to integer
      inputs. A floating-point tensor with ``requires_grad=True`` always
      interpolates (even fully on-node): only the interpolated, differentiable
      path can carry gradients back to the positions.

    The node test is made in the component's own index space: a coordinate is
    "on a node" when ``location - offset`` is within ``eps`` of an integer. For
    the regular grid (``offset`` is zero/None) this is just proximity to an
    integer; for a staggered velocity component (e.g. ``offset`` 0.5) the nodes
    sit at half-integer physical positions, so a coordinate that happens to be a
    whole number is correctly treated as off-node and interpolated.

    Args:
        locations: The source or receiver locations, or ``None``.
        mode: One of ``"auto"``, ``"hicks"``, or ``"none"``.
        eps: Tolerance (in cells) for treating a coordinate as on a node.
        offset: Optional per-dimension half-cell component offset (see
            :class:`Hicks`). The on-node test is performed relative to this.

    Returns:
        ``True`` if Hicks interpolation should be applied.

    """
    if mode not in _LOCATION_INTERPOLATION_MODES:
        raise DeepwaveValueError(
            "location_interpolation must be one of "
            f"{_LOCATION_INTERPOLATION_MODES}, got {mode!r}.",
        )
    if locations is None or mode == "none":
        return False
    if mode == "hicks":
        return True
    if not locations.is_floating_point() or locations.numel() == 0:
        return False
    # A float location Tensor that requires grad always interpolates in auto
    # mode: only the interpolated (differentiable Hicks) path can carry
    # gradients back to the positions, and snapping an on-node point to the
    # exact integer path would silently sever them (or, during gradient-based
    # position optimisation, make the gradient vanish whenever a point lands
    # exactly on a node).
    if locations.requires_grad:
        return True
    # Ignored coordinates (the IGNORE_LOCATION sentinel) must not force
    # interpolation; the on-node test excludes them (relevant when an offset
    # shift gives the sentinel a non-zero fractional part).
    #
    # ``warn_truncated`` only on this path, which is the only one a v0.0.27
    # call can reach: ``location_interpolation`` did not exist in that release,
    # so a call that names ``"hicks"`` or ``"none"`` has already been rewritten
    # for this one, and a location Tensor that requires grad could not carry a
    # gradient there at all. The migration question is therefore asked exactly
    # where it can be answered without a second read of the coordinates.
    return _has_off_node_coord(locations, offset, eps, warn_truncated=True)


def _hicks_window_weights(
    x: torch.Tensor,
    beta: torch.Tensor,
    halfwidth: int,
    monopole: bool,
    eps: float,
    grad_safe: bool = False,
) -> torch.Tensor:
    """Kaiser-windowed-sinc (monopole) or its derivative (dipole) weights.

    This is the analytic core shared by the cached default path and the
    differentiable path. The weights are a smooth function of the sub-cell
    offset encoded in ``x`` (the integer window-node positions minus the
    fractional part of the location), so ``d(weights)/d(location)`` is a
    well-defined torch expression; only the choice of integer support cells
    (which set the constant offset of ``x``) is non-differentiable.

    With ``grad_safe=False`` the expression is written identically to the
    historical inline form so the default (non-differentiable) path stays
    bit-for-bit unchanged.

    ``grad_safe=True`` (the differentiable path) adds a tiny floor inside the
    Kaiser square root: at an exactly on-node location the window edge point has
    ``x = +-halfwidth`` so ``u = 1 - (x/halfwidth)**2 = 0``, and although the
    composition ``i0(beta*sqrt(u))`` is smooth in ``u`` (``i0`` is even, so the
    true weight derivative is finite), autograd differentiates through
    ``sqrt(0)`` whose infinite slope turns the backward pass into ``0 * inf =
    NaN``. Evaluating ``sqrt(u + 1e-30)`` instead leaves the forward values
    unchanged (``1e-30`` is far below one ulp of any ``u`` that is not exactly
    zero, and ``i0(beta*1e-15) == 1.0 == i0(0)`` in both float32 and float64)
    while the backward pass converges to the correct analytic limit
    ``d i0(beta*sqrt(u))/du -> beta**2/4`` as ``u -> 0``.

    Args:
        x: The window-node offsets (``grid node - location``) as a tensor.
        beta: Kaiser window parameter.
        halfwidth: Halfwidth of the interpolation window.
        monopole: If True, monopole weights; otherwise dipole (derivative)
            weights along this dimension.
        eps: Small value to prevent division by zero in the dipole expression.
        grad_safe: If True, guard the Kaiser square root so the backward pass
            is finite at on-node locations (see above). Only used by the
            differentiable path; the default path keeps the historical
            expression bit-for-bit.

    Returns:
        A 1-D tensor of weights, the same length as ``x``.

    """
    u = 1 - (x / halfwidth) ** 2
    if grad_safe:
        u = u + 1e-30
    if monopole:
        return torch.sinc(x) * torch.i0(beta * u.sqrt()) / torch.i0(beta)
    return (
        (torch.cos(math.pi * x) - torch.sinc(x))
        / (x**2 + eps)
        * x
        * torch.i0(beta * u.sqrt())
        / torch.i0(beta)
    )


def _get_hicks_for_one_location_dim_differentiable(
    location: torch.Tensor,
    halfwidth: int,
    beta: torch.Tensor,
    monopole: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Hicks weights for one dimension (no reflecting surfaces).

    ``location`` is a 0-D floating-point tensor (it may require grad). The
    integer support cells are selected by ``floor(location)`` and are returned
    as a detached ``long`` tensor: they are piecewise constant in position, so
    they carry no gradient. The weights are a differentiable torch expression of
    ``location`` via the offset ``x``, so ``d(weights)/d(location)`` flows to the
    position.

    No on-node shortcut is taken (unlike the default path): the full window is
    always emitted so the gradient is defined everywhere. At an exactly on-node
    location the window reduces numerically to a unit spike (``sinc`` of a
    non-zero integer is the round-off of ``sin(n*pi)``, ~1e-16), so the
    spread/gather values match the exact path to round-off; the extra cells
    carry only negligible weight.

    As ``location`` crosses an integer the leftmost support cell leaves and a new
    rightmost cell enters. The support is piecewise differentiable: a monopole
    value is continuous at the crossing but its position derivative can jump.
    A dipole value can jump because its endpoint weight need not vanish. Avoid
    crossing an integer position when a continuous position derivative is
    required.

    Args:
        location: A 0-D floating-point tensor giving the location (in this
            dimension's own index space, i.e. with any component offset already
            removed).
        halfwidth: Halfwidth of the interpolation window.
        beta: Kaiser window parameter.
        monopole: If True, monopole weights; otherwise dipole weights along this
            dimension.
        eps: Small value to prevent division by zero in the dipole expression.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Detached ``long`` grid indices and
            the corresponding differentiable weights.

    """
    floor_location = math.floor(float(location.detach()))
    steps = torch.arange(-halfwidth + 1, halfwidth + 1, device=beta.device)
    locations = (steps + floor_location).long()
    x = steps.to(beta.dtype) - location + floor_location
    weights = _hicks_window_weights(
        x,
        beta,
        halfwidth,
        monopole,
        eps,
        grad_safe=True,
    )
    return locations, weights


#: The once-per-process key of the changed-stencil migration warning.
_STENCIL_CAUSE = "migration:hicks-stencil-indices"


def _v0027_stencil_indices(
    location: float,
    halfwidth: int,
    dtype: torch.dtype,
    monopole: bool,
    eps: float,
) -> list[int]:
    """The integer cells Deepwave 0.0.27 would have used for one dimension.

    A faithful transcription of that release's own arithmetic, kept separate
    from the working code so that the two cannot be confused. The point of it
    is the line ``(location + x).long()``: 0.0.27 built the cell indices by
    truncating a floating-point sum, which is one cell too low wherever the sum
    landed a fraction of a ULP below an integer, and which truncates towards
    zero rather than downwards for a negative coordinate. This release builds
    them as ``floor(location) + arange(...)``, in integer arithmetic, which is
    exact. Only the indices are reproduced; the weights are unchanged.

    The window *before* any reflecting-surface folding is what is returned, on
    both sides of the comparison: folding is driven by the window's own first
    and last cell, so it can only move indices that this arithmetic has
    already decided.

    Every operation is performed on the CPU, on values that are already on the
    host, so calling this never touches an accelerator and never synchronises
    one.

    Args:
        location: The coordinate, in cells, in this dimension.
        halfwidth: Halfwidth of the interpolation window.
        dtype: The dtype 0.0.27 would have done this arithmetic in for this
            call: whatever ``dtype=`` it was given, which defaulted to
            ``torch.float`` (float32) and never consulted
            :func:`torch.get_default_dtype`. Every tensor below names it
            explicitly, so the reproduction does not follow the current
            default dtype either.
        monopole: Whether this dimension's weights are monopole weights.
        eps: The on-node tolerance.

    Returns:
        The cell indices, in order.

    """
    if monopole and abs(location - round(location)) < eps:
        # 0.0.27's single-cell shortcut for a monopole on a node, which it took
        # whether or not the node lay on a reflecting surface.
        return [round(location)]
    steps = torch.arange(-halfwidth + 1, halfwidth + 1, dtype=dtype)
    x = steps - location + int(location)
    return cast("list[int]", (location + x).long().tolist())


def _warn_if_stencil_changed(
    location: float,
    floor_location: int,
    halfwidth: int,
    dtype: torch.dtype,
    monopole: bool,
    eps: float,
) -> None:
    """Warns, at construction, when a window's cells differ from 0.0.27's.

    Both index lists are built on the host from ``location`` alone, so nothing
    is read back from the device and no synchronisation is added. Skipped
    entirely once the warning has been emitted, so the cost is paid at most
    once per process.

    The comparison is made *before* any reflecting-surface folding, on both
    sides, because folding is driven by the window's own first and last cell
    and so can only move cells this arithmetic has already chosen. That is
    deliberately conservative in one direction: where a surface folds the
    differing cell away, the cells actually delivered can be identical and the
    warning still fires. Saying so costs a reader one message; staying silent
    would cost them the explanation of a changed number.

    Args:
        location: The coordinate, in cells, in this dimension, with any
            component offset already removed.
        floor_location: ``floor(location)``, the cell this release's window is
            built about (its cells are ``floor_location + arange(-halfwidth +
            1, halfwidth + 1)``).
        halfwidth: Halfwidth of the interpolation window.
        dtype: The dtype 0.0.27 would have built the weights in for this call
            (see :func:`_v0027_stencil_indices`).
        monopole: Whether this dimension's weights are monopole weights.
        eps: The on-node tolerance.

    """
    old = _v0027_stencil_indices(location, halfwidth, dtype, monopole, eps)
    new = [floor_location + step for step in range(-halfwidth + 1, halfwidth + 1)]
    if old == new:
        return
    _common._warn_once_per_cause(  # noqa: SLF001
        _STENCIL_CAUSE,
        f"the Hicks interpolation window for a coordinate at {location} cells "
        f"covers cells {new}, where Deepwave 0.0.27 and earlier would have "
        f"used {old}. That release built the cell indices by truncating a "
        "floating-point sum, which put one cell of the window a whole cell "
        "away from where it belonged; they are now built in exact integer "
        "arithmetic. The interpolation weights are unchanged, so this is a bug "
        "fix with no way to ask for the old cells back, but it does mean that "
        "output computed for an off-node coordinate differs from what that "
        "release produced. This is a migration warning, and is removed in "
        f"Deepwave {_DEPRECATION_REMOVAL_VERSION}. Silence it with: "
        'warnings.filterwarnings("ignore", '
        "category=deepwave.MigrationWarning).",
        MigrationWarning,
        stacklevel=5,
    )


def _get_hicks_for_one_location_dim(
    hicks_weight_cache: dict[tuple[int, int, int], torch.Tensor],
    location: float,
    halfwidth: int,
    beta: torch.Tensor,
    free_surface: list[bool],
    free_surface_loc: list[float],
    size: int,
    monopole: bool = True,
    eps: float = _DEFAULT_EPS,
    rigid_surface: Optional[list[bool]] = None,
    compare_v0027: bool = False,
    v0027_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate Hicks interpolation locations and weights for one dimension.

    Near a reflecting surface the interpolation window extends past the edge of
    the model into the (non-existent) exterior. The method of images is used to
    fold that part of the window back into the model:

    - A **free surface** (the field is zero on the surface; an odd/antisymmetric
      mirror) folds the exterior weights back with a **sign flip** (subtraction).
      A surface lying exactly on a grid node forces that node's value to zero, so
      the node is dropped from the output (which also avoids emitting an
      out-of-model index when the surface sits at or beyond the model edge).
    - A **rigid surface** (the field has zero normal gradient on the surface; an
      even/symmetric mirror) folds the exterior weights back with the **same
      sign** (addition). A surface lying exactly on a node doubles that node's
      weight (the real point and its image coincide there), so the node is kept.

    See :class:`Hicks` for the per-propagator surface conventions (which
    reflection type and surface location match each Deepwave boundary).

    Args:
        hicks_weight_cache: Cache for Hicks weights to avoid recomputing.
        location: The floating-point location to interpolate.
        halfwidth: Halfwidth of the interpolation window.
        beta: Kaiser window parameter.
        free_surface: List of two booleans indicating free surface presence at
            the beginning and end of this dimension.
        free_surface_loc: List of two floats indicating the surface locations
            (used for both free and rigid surfaces on this dimension).
        size: Size of the grid in this dimension.
        monopole: If True, calculate for a monopole; otherwise, for a dipole.
        eps: Small value to prevent division by zero.
        rigid_surface: Optional list of two booleans indicating rigid surface
            presence at the beginning and end of this dimension. A face must not
            be both free and rigid.
        compare_v0027: Whether this dimension is one Deepwave 0.0.27 could
            have interpolated, so that a window whose cells differ from the
            ones its arithmetic gave is worth saying out loud (see
            :func:`_warn_if_stencil_changed`). False for a staggered component
            offset and for a rigid surface, neither of which that release had.
        v0027_dtype: The dtype Deepwave 0.0.27 would have built the weights in
            for this call, used only by that comparison. ``None`` means
            ``beta.dtype``, this release's own weight dtype. The two differ
            when ``Hicks(dtype=None)`` follows a default dtype other than
            float32: 0.0.27's ``dtype`` argument defaulted to ``torch.float``
            and never consulted the default dtype, so comparing in the current
            default would miss a genuine window change (see
            :class:`Hicks`).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Locations and corresponding weights.

    """
    if rigid_surface is None:
        rigid_surface = [False, False]
    if not isinstance(halfwidth, int):
        raise DeepwaveTypeError("halfwidth must be an int.")
    if halfwidth < 1 or halfwidth > 10:
        raise DeepwaveRuntimeError("halfwidth must be in [1, 10].")
    if beta < 0:
        raise DeepwaveRuntimeError("beta must be non-negative.")
    if (
        not isinstance(free_surface_loc, list)
        or len(free_surface_loc) != 2
        or not all(isinstance(f, (int, float)) for f in free_surface_loc)
    ):
        raise DeepwaveRuntimeError("extent must be a list of two floats.")
    for s in range(2):
        if free_surface[s] and rigid_surface[s]:
            raise DeepwaveRuntimeError(
                "a face cannot be both a free and a rigid surface."
            )
    surface = [free_surface[s] or rigid_surface[s] for s in range(2)]
    if (surface[0] or surface[1]) and size <= 0:
        raise DeepwaveRuntimeError("n_grid_points must be positive.")
    # A monopole sitting exactly on a grid node normally reduces to a single
    # unit-weight cell. But if that node lies on (or beyond) a surface plane the
    # image must still be applied (a free surface drops it, a rigid surface
    # doubles it), so fall through to the folding branch in that case. The
    # folding branch handles an on-node location correctly because its Kaiser
    # weights are a unit spike at the node and zero elsewhere.
    snapped = round(location)
    on_surface = (surface[0] and snapped <= free_surface_loc[0]) or (
        surface[1] and snapped >= free_surface_loc[1]
    )
    if monopole and abs(location - snapped) < eps and not on_surface:
        locations = torch.tensor([location]).round().long().to(beta.device)
        weights = torch.ones(1, dtype=beta.dtype, device=beta.device)
    else:
        # Centre the window with floor() (not int(), which truncates toward
        # zero) so the fractional part stays in [0, 1) for negative locations
        # too; otherwise the Kaiser argument can exceed the halfwidth and
        # produce NaN weights. Negative locations arise when a staggered
        # component offset shifts a near-origin point below zero.
        floor_location = math.floor(location)
        if compare_v0027 and not _common._already_warned_once(_STENCIL_CAUSE):  # noqa: SLF001
            _warn_if_stencil_changed(
                location,
                floor_location,
                halfwidth,
                beta.dtype if v0027_dtype is None else v0027_dtype,
                monopole,
                eps,
            )
        key = (int((location - floor_location) / eps), halfwidth, int(monopole))
        x = (
            torch.arange(
                -halfwidth + 1,
                halfwidth + 1,
                dtype=beta.dtype,
                device=beta.device,
            )
            - location
            + floor_location
        )
        locations = (
            torch.arange(
                -halfwidth + 1,
                halfwidth + 1,
                device=beta.device,
            )
            + floor_location
        ).long()

        if key in hicks_weight_cache:
            weights = hicks_weight_cache[key].clone()
        else:
            weights = _hicks_window_weights(x, beta, halfwidth, monopole, eps)
            hicks_weight_cache[key] = weights.clone()
        if surface[0]:
            # <= (not <): a window cell can land exactly ON an integer
            # surface plane (e.g. a custom interior free/rigid plane) with no
            # cell extending past it; that on-plane cell still needs folding
            # (zeroed for free, doubled for rigid) or it silently keeps its
            # full, un-imaged weight.
            if locations[0].item() <= free_surface_loc[0]:
                # idx0: first index at, or on model side of, the surface
                # idx1: first index strictly on the model side of the surface
                # sign: -1 reflects with a flip (free), +1 without (rigid)
                sign = -1.0 if free_surface[0] else 1.0
                idx0 = math.ceil(free_surface_loc[0] - locations[0].item())
                idx1 = idx0 + int(
                    math.isclose(round(free_surface_loc[0]), free_surface_loc[0]),
                )
                flipped_weights = weights[:idx1].flip(0)
                # Clip to the window: if the surface plane sits several cells
                # inside the model (e.g. a custom plane, or an offset-shifted
                # staggered plane), some images mirror past the far end of the
                # window and cannot be represented here; drop them rather than
                # raising. This only occurs for surfaces away from the model
                # edge; edge-default planes always fit (exact imaging).
                n = min(len(flipped_weights), len(weights) - idx0)
                weights[idx0 : idx0 + n] += sign * flipped_weights[:n]
                # Free surfaces zero (and so drop) an on-node surface point;
                # rigid surfaces double and keep it.
                start = idx1 if free_surface[0] else idx0
                locations = locations[start:]
                weights = weights[start:]
            # Never emit a cell left of the model. With a regular grid the fold
            # above already trims to the model; with a staggered offset the
            # shifted surface plane can leave an out-of-model cell on its
            # interior side, which is dropped here (see the Hicks staggered note).
            in_model = locations >= 0
            locations, weights = locations[in_model], weights[in_model]
        if surface[1]:
            # >= (not >): see the symmetric begin-face note above.
            if locations[-1].item() >= free_surface_loc[1]:
                sign = -1.0 if free_surface[1] else 1.0
                idx0 = (
                    len(weights)
                    - 1
                    - math.ceil(locations[-1].item() - free_surface_loc[1])
                )
                idx1 = idx0 - int(
                    math.isclose(round(free_surface_loc[1]), free_surface_loc[1]),
                )
                flipped_weights = weights[idx1 + 1 :].flip(0)
                # Clip to the window (see the begin-face note): images that
                # mirror past the near end of the window are dropped instead of
                # wrapping via a negative start index.
                start_w = idx0 + 1 - len(flipped_weights)
                if start_w < 0:
                    flipped_weights = flipped_weights[-start_w:]
                    start_w = 0
                weights[start_w : idx0 + 1] += sign * flipped_weights
                end = idx1 + 1 if free_surface[1] else idx0 + 1
                locations = locations[:end]
                weights = weights[:end]
            in_model = locations < size
            locations, weights = locations[in_model], weights[in_model]
    return locations, weights


def _get_hicks_locations_and_weights(
    locations: torch.Tensor,
    halfwidth: int,
    beta: torch.Tensor,
    free_surfaces: list[bool],
    free_surface_locs: list[float],
    model_shape: list[int],
    monopole: Union[torch.Tensor, bool] = True,
    dipole_dim: Union[torch.Tensor, int] = 0,
    eps: float = _DEFAULT_EPS,
    offset: Optional[list[float]] = None,
    rigid_surfaces: Optional[list[bool]] = None,
    differentiable: bool = False,
    v0027_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, list[list[list[int]]], list[list[list[torch.Tensor]]]]:
    """Calculate Hicks interpolation locations and weights for all shots/sources.

    This supports an arbitrary number of spatial dimensions (1, 2, or 3),
    matching the number of dimensions of the Deepwave propagator being used.

    Args:
        locations: A torch.Tensor [shot, per_shot, ndim] of original
            source/receiver locations.
        halfwidth: Halfwidth of the interpolation window.
        beta: Kaiser window parameter.
        free_surfaces: List of ``2 * ndim`` booleans indicating free surface
            presence (beginning then end of each dimension).
        free_surface_locs: List of ``2 * ndim`` floats indicating free surface
            locations.
        model_shape: Shape of the model grid (``ndim`` integers).
        monopole: Boolean or torch.Tensor indicating if sources/receivers are
            monopoles.
        dipole_dim: Integer or torch.Tensor in ``[0, ndim)`` indicating the
            dipole orientation dimension.
        eps: Small value to prevent division by zero.
        offset: Optional list of ``ndim`` floats giving the half-cell offset of
            the field component being interpolated relative to the reference
            (pressure / regular-grid) node. Interpolation is performed in the
            component's own index space, so a staggered velocity component
            (offset 0.5 in its propagation direction) is interpolated about its
            true node location. Defaults to zero offset in every dimension.
        rigid_surfaces: Optional list of ``2 * ndim`` booleans indicating rigid
            surface presence (beginning then end of each dimension). Defaults to
            no rigid surfaces. A face must not be both free and rigid.
        differentiable: If True, build the weights as differentiable torch
            expressions of ``locations`` (no float-keyed cache) so gradients flow
            to the positions. Free/rigid surfaces are not supported in this mode.
        v0027_dtype: The dtype Deepwave 0.0.27 would have built the weights in
            for this call, used only by the changed-stencil comparison (see
            :func:`_get_hicks_for_one_location_dim`). ``None`` means
            ``beta.dtype``.

    Returns:
        Tuple[torch.Tensor, List[List[List[int]]], List[List[List[torch.Tensor]]]]:
            Interpolated locations, indices, and weights.

    """
    ndim = locations.shape[2]
    if offset is None:
        offset = [0.0] * ndim
    if rigid_surfaces is None:
        rigid_surfaces = [False] * (2 * ndim)
    if differentiable and (any(free_surfaces) or any(rigid_surfaces)):
        raise DeepwaveNotImplementedError(
            "differentiable=True does not support free or rigid surfaces.",
        )
    hicks_weight_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    n_shots, n_per_shot, _ = locations.shape
    hicks_locations_list: list[list[tuple[int, ...]]] = []
    hicks_idxs: list[list[list[int]]] = []
    weights: list[list[list[torch.Tensor]]] = []
    n_per_shot_hicks = 0
    for shotidx in range(n_shots):
        shot_location_idxs: list[list[int]] = []
        shot_weights: list[list[torch.Tensor]] = []
        n_hicks_locations = 0
        locations_dict: dict[tuple[int, ...], int] = {}
        for i in range(n_per_shot):
            if isinstance(monopole, torch.Tensor):
                monopole_i = bool(monopole[shotidx, i].item())
            else:
                monopole_i = monopole
            if isinstance(dipole_dim, torch.Tensor):
                dipole_dim_i = int(dipole_dim[shotidx, i].item())
            else:
                dipole_dim_i = dipole_dim
            dim_locations: list[torch.Tensor] = []
            dim_weights: list[torch.Tensor] = []
            for d in range(ndim):
                # A dipole only takes its derivative weights along dipole_dim;
                # every other dimension uses monopole weights.
                dim_monopole = monopole_i or (dipole_dim_i != d)
                if differentiable:
                    locations_d, weights_d = (
                        _get_hicks_for_one_location_dim_differentiable(
                            locations[shotidx, i, d] - offset[d],
                            halfwidth,
                            beta,
                            dim_monopole,
                            eps,
                        )
                    )
                else:
                    locations_d, weights_d = _get_hicks_for_one_location_dim(
                        hicks_weight_cache,
                        float(locations[shotidx, i, d].item()) - offset[d],
                        halfwidth,
                        beta,
                        free_surfaces[2 * d : 2 * d + 2],
                        [fl - offset[d] for fl in free_surface_locs[2 * d : 2 * d + 2]],
                        model_shape[d],
                        dim_monopole,
                        eps=eps,
                        rigid_surface=rigid_surfaces[2 * d : 2 * d + 2],
                        compare_v0027=(
                            offset[d] == 0.0
                            and not any(rigid_surfaces[2 * d : 2 * d + 2])
                        ),
                        v0027_dtype=v0027_dtype,
                    )
                dim_locations.append(locations_d)
                dim_weights.append(weights_d)
            shot_weights.append(dim_weights)
            i_idxs: list[int] = []
            cartesian = torch.cartesian_prod(*dim_locations)
            if ndim == 1:
                cartesian = cartesian.reshape(-1, 1)
            for loc in cartesian:
                loc_tuple = tuple(int(v) for v in loc)
                if loc_tuple in locations_dict:
                    i_idxs.append(locations_dict[loc_tuple])
                else:
                    locations_dict[loc_tuple] = n_hicks_locations
                    i_idxs.append(n_hicks_locations)
                    n_hicks_locations += 1
            shot_location_idxs.append(i_idxs)
        hicks_idxs.append(shot_location_idxs)
        weights.append(shot_weights)
        n_per_shot_hicks = max(n_per_shot_hicks, n_hicks_locations)
        hicks_locations_list.append(list(locations_dict.keys()))
    hicks_locations = (
        torch.ones(
            n_shots,
            n_per_shot_hicks,
            ndim,
            dtype=torch.long,
            device=locations.device,
        )
        * _common.IGNORE_LOCATION
    )
    for shotidx in range(n_shots):
        for i, loc in enumerate(hicks_locations_list[shotidx]):
            hicks_locations[shotidx, i] = torch.tensor(loc).to(locations.device)

    return hicks_locations, hicks_idxs, weights


def _describe_offset(offset: Sequence[float]) -> str:
    """A human description of the field component sitting at ``offset``."""
    staggered = [d for d, value in enumerate(offset) if value != 0]
    if not staggered:
        return "the regular grid (pressure/wavefield nodes)"
    dims = ", ".join(str(d) for d in staggered)
    plural = "s" if len(staggered) > 1 else ""
    return f"a component staggered half a cell in dimension{plural} {dims}"


def _warn_on_offset_mismatch(
    interpolator: "Hicks",
    offset: Optional[list[float]],
    kind: str,
) -> None:
    """Warns when a user-supplied ``Hicks`` was built for a different node.

    A ``Hicks`` passed as a location argument is used *as-is*: the
    component's own half-cell ``offset`` is not applied to it, because the
    object already carries whatever offset it was built with. That is the
    right behaviour -- it is how a user requests a custom offset -- but it
    is silent, so a ``Hicks`` built with the default (regular-grid) offset
    and then passed for a staggered velocity component interpolates about
    the wrong nodes, half a cell out, with no indication at all.

    The component is identified by its offset rather than by an argument
    name: the offset is what actually differs, it says physically which
    component is meant, and it is the value the fix has to use.

    Emitted at most once per process per ``(kind, offsets)`` pair.

    Args:
        interpolator: The user-supplied ``Hicks``.
        offset: The half-cell offset of the component it is being used for.
            ``None`` means the regular-grid node (all zeros).
        kind: ``"source"`` or ``"receiver"``, for the message.

    """
    expected = [0.0] * len(interpolator.offset) if offset is None else list(offset)
    actual = list(interpolator.offset)
    if actual == expected:
        return
    from deepwave import _common

    displacement = [a - e for a, e in zip(actual, expected)]
    _common._warn_once_per_cause(  # noqa: SLF001
        f"hicks-offset-mismatch:{kind}:{actual}:{expected}",
        f"deepwave: a Hicks object passed as a {kind} location argument was "
        f"built with offset={actual!r} ({_describe_offset(actual)}), but it "
        f"is being used for a {kind} component whose grid nodes are at "
        f"offset={expected!r} ({_describe_offset(expected)}). A Hicks is "
        "used exactly as given -- it keeps its own offset, and the "
        "component's is not applied to it -- so this interpolates about "
        f"nodes displaced by {displacement!r} cells. If that is deliberate, "
        "ignore this. Otherwise rebuild it as "
        f"Hicks(..., offset={expected!r}), or pass the coordinates as a "
        "plain floating-point Tensor and let Deepwave build an interpolator "
        "with the right offset for each component.",
        UserWarning,
        3,
    )


_LOCATION_UNITS_HINT = (
    "Locations are grid-cell indices, not physical distances -- if you have "
    "coordinates in metres (or another physical unit), divide by "
    "grid_spacing first. Negative indices are not supported, except the "
    "sentinel deepwave.IGNORE_LOCATION (currently "
    f"{_common.IGNORE_LOCATION}), which marks an unused source/receiver slot."
)


def _window_reach(
    locations: torch.Tensor,
    halfwidth: int,
    offset: Optional[Sequence[float]],
    dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest and highest grid index each point's window would occupy.

    In ``dim``, in the component's own index space, and exactly as
    :func:`_get_hicks_for_one_location_dim` builds it: a monopole within
    ``eps`` of a node occupies that node alone, and anything else occupies
    ``floor(x) - halfwidth + 1 .. floor(x) + halfwidth``. Points flagged with
    :data:`deepwave.IGNORE_LOCATION` are reported at index 0, the safe
    coordinate ``Hicks`` itself substitutes for them.
    """
    ignored = (locations == _common.IGNORE_LOCATION).all(dim=-1)
    shifted = locations[..., dim] - (0.0 if offset is None else offset[dim])
    snapped = shifted.round()
    on_node = (shifted - snapped).abs() < eps
    low = torch.where(on_node, snapped, shifted.floor() - (halfwidth - 1))
    high = torch.where(on_node, snapped, shifted.floor() + halfwidth)
    zero = torch.zeros_like(low)
    return torch.where(ignored, zero, low), torch.where(ignored, zero, high)


def _point_description(flat_index: int, per_shot: int) -> str:
    """``"shot 2, point 5"`` for a flat index into a ``[shot, point]`` grid."""
    return f"shot {flat_index // per_shot}, point {flat_index % per_shot}"


def _check_interpolated_bounds(
    locations: torch.Tensor,
    halfwidth: int,
    offset: Optional[Sequence[float]],
    eps: float,
    grid: "_common.HicksGrid",
    name: str,
    *,
    differentiable: bool,
) -> None:
    """Refuse an interpolation window that leaves the model.

    The window of a point that is not on a node spans ``2 * halfwidth``
    cells, so a point fewer than ``halfwidth - 1`` cells from a face has part
    of it outside the model. What happens then depends on the face:

    * a ``free`` or ``rigid`` face whose image :class:`Hicks` reproduces is
      folded back, and is not an error (it is not checked here);
    * any other named face is refused with a
      :exc:`deepwave.DeepwaveNotImplementedError` that says why and names the
      two ways round it;
    * an absorbing (``pml``) face is out of bounds, as it has always been,
      but the message now reports the coordinate the caller wrote rather than
      the expanded window index that first left the model.

    Raised here, before the window is built, so that every message can still
    name the caller's own coordinate.
    """
    if locations.ndim != 3 or locations.shape[2] != len(grid.model_shape):
        return
    if locations.numel() == 0:
        return
    per_shot = max(int(locations.shape[1]), 1)
    ndim = locations.shape[2]
    blocked = {face: (why, reason) for face, why, reason in grid.blocked}
    for dim in range(ndim):
        size = grid.model_shape[dim]
        low, high = _window_reach(locations, halfwidth, offset, dim, eps)
        for face, outside in ((2 * dim, low < 0), (2 * dim + 1, high > size - 1)):
            folded = grid.free[face] or grid.rigid[face]
            if folded and not differentiable:
                continue
            if not bool(outside.any()):
                continue
            flat = int(outside.reshape(-1).nonzero()[0].item())
            where = _point_description(flat, per_shot)
            coordinate = locations.reshape(-1, ndim)[flat, dim].item()
            own_index = math.floor(coordinate)
            if not 0 <= own_index <= size - 1:
                raise DeepwaveRuntimeError(
                    f"{grid.propagator}: {name} is out of bounds: at {where}, "
                    f"dimension {dim} (of {ndim}), the coordinate is "
                    f"{coordinate} (grid cell {own_index}), but the model has "
                    f"only {size} cells (valid indices 0 to {size - 1}). "
                    f"{_LOCATION_UNITS_HINT}"
                )
            reach = int(
                low.reshape(-1)[flat].item()
                if face % 2 == 0
                else high.reshape(-1)[flat].item()
            )
            limit = (
                f"a coordinate of {halfwidth - 1} or more"
                if face % 2 == 0
                else f"a coordinate below {size - halfwidth}"
            )
            window = (
                f"the coordinate is {coordinate}, which is inside the model, "
                "but it is not on a grid node, so it is interpolated over a "
                f"window of halfwidth {halfwidth} that reaches grid cell "
                f"{reach}, past {_common.face_description(face)}"
            )
            workaround = (
                "Give that point an integer (on-node) coordinate in this "
                f"dimension, move it to {limit}, or pass a smaller "
                "location_interpolation_halfwidth"
            )
            if folded:
                raise DeepwaveNotImplementedError(
                    f"{grid.propagator}: {name} at {where}, dimension {dim} "
                    f"(of {ndim}): {window}, which this call makes a "
                    "reflecting surface. Folding the window back off it is "
                    "not implemented for a location Tensor with "
                    "requires_grad=True. "
                    f"{workaround}, or detach the locations."
                )
            if face in blocked:
                why, reason = blocked[face]
                raise DeepwaveNotImplementedError(
                    f"{grid.propagator}: {name} at {where}, dimension {dim} "
                    f"(of {ndim}): {window}, which this call gives "
                    f"boundary={why!r}. Deepwave cannot fold the window back "
                    f"off that face here, because {reason}. {workaround}."
                )
            raise DeepwaveRuntimeError(
                f"{grid.propagator}: {name} is out of bounds: at {where}, "
                f"dimension {dim} (of {ndim}), {window}, which is outside the "
                f"model (valid indices 0 to {size - 1}). {workaround}."
            )


def _resolve_interpolator(
    locations: Union[torch.Tensor, "Hicks", None],
    mode: str = "auto",
    halfwidth: int = 4,
    offset: Optional[list[float]] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = _DEFAULT_EPS,
    kind: Optional[str] = None,
    grid: Optional["_common.HicksGrid"] = None,
    name: str = "locations",
) -> Optional["Hicks"]:
    """Resolve a propagator location argument to a Hicks interpolator, or None.

    This is the single entry point the propagators use to support off-grid and
    directional (dipole) sources/receivers. The location argument may be:

    - a :class:`Hicks` instance, which is returned as-is (this is how a user
      requests dipoles, a custom component ``offset``, or free surfaces — the
      propagator then performs the spread/gather automatically);
    - a floating-point Tensor, which is routed through a freshly built Hicks
      (with the given per-component ``offset``) when interpolation is needed;
    - an integer/``torch.long`` Tensor (or ``None``), for which ``None`` is
      returned so the exact, non-interpolated path is used unchanged.

    When the float location Tensor has ``requires_grad=True`` the auto-built
    Hicks is created in differentiable mode, so gradients flow back to the
    source/receiver positions (survey design, source localisation). A plain
    float Tensor keeps the fast cached, bit-identical path.

    Args:
        locations: The source/receiver location argument (Tensor, Hicks, or
            None).
        mode: The ``location_interpolation`` mode (``"auto"``/``"hicks"``/
            ``"none"``); ignored when ``locations`` is a Hicks instance. In
            ``"none"`` mode an off-node float coordinate is an error (see
            Raises) -- the exact integer path only.
        halfwidth: Halfwidth for an auto-built Hicks.
        offset: Per-dimension component offset for an auto-built Hicks (the
            staggered half-cell offset of the field component).
        dtype: Datatype for an auto-built Hicks. ``None`` means the default
            dtype.
        eps: On-grid tolerance for :func:`_should_interpolate`.
        kind: ``"source"`` or ``"receiver"``, used only in the
            offset-mismatch warning. ``None`` disables that check.
        grid: The model shape and the boundary condition at each of its faces
            (see :func:`deepwave._common.hicks_grid`). Given one, an
            auto-built interpolator folds its window back off a reflecting
            face it can image, and a window that leaves the model any other
            way is refused here -- naming the caller's own coordinate --
            rather than reaching the shared bounds check as an expanded
            window index. ``None`` (the helpers' own tests) builds the
            interpolator knowing nothing about the faces, as before.
        name: The location argument's name, for those messages.

    Returns:
        A :class:`Hicks` interpolator, or ``None`` if no interpolation is
        needed.

    Raises:
        ValueError: if ``mode`` is ``"none"`` and ``locations`` is a float
            Tensor with any coordinate more than ``eps`` off its grid node
            (``"none"`` is the exact integer path only; it no longer silently
            truncates an off-node coordinate).

    """
    if isinstance(locations, Hicks):
        if kind is not None:
            _warn_on_offset_mismatch(locations, offset, kind)
        return locations
    if (
        mode == "none"
        and isinstance(locations, torch.Tensor)
        and locations.is_floating_point()
        and locations.numel() > 0
        and _has_off_node_coord(locations, offset, eps)
    ):
        # ``"none"`` means the exact integer path only. Deepwave 0.0.27
        # accepted an off-node float coordinate within 0.1 cells of the node
        # below and silently truncated it to that node, a precision footgun;
        # it is now an error. Callers who want the old placement do it
        # themselves (``locations.floor().long()``); callers who want
        # interpolation pass ``location_interpolation="auto"`` or ``"hicks"``.
        raise DeepwaveValueError(
            "location_interpolation='none' requires source/receiver coordinates "
            "that lie exactly on grid nodes (within "
            f"{eps} cells), but a float location Tensor has one or more off-node "
            "coordinates. 'none' selects the exact integer path only and no "
            "longer silently truncates an off-node coordinate to the node below "
            "it, as Deepwave 0.0.27 did. Pass "
            "location_interpolation='auto' (or 'hicks') to interpolate them, or "
            "truncate the coordinates to integer cells yourself, e.g. "
            "locations.floor().long()."
        )
    if _should_interpolate(locations, mode, eps, offset):
        # Make the interpolation differentiable w.r.t. the positions whenever the
        # caller has marked the (float) location Tensor as requiring grad. This
        # is the auto-path opt-in: a plain float location Tensor keeps the fast
        # cached, bit-identical path, while ``locations.requires_grad_(True)``
        # additionally lets gradients flow back to the source/receiver positions.
        differentiable = isinstance(locations, torch.Tensor) and locations.requires_grad
        # _should_interpolate's first check is `if locations is None ...:
        # return False`, so reaching here guarantees `locations` is a Tensor;
        # mypy cannot see through that call to narrow the type itself, so this
        # cast just states the invariant (no runtime effect).
        tensor = cast("torch.Tensor", locations)
        if grid is None:
            return Hicks(
                tensor,
                halfwidth=halfwidth,
                offset=offset,
                dtype=dtype,
                differentiable=differentiable,
            )
        # A reflecting face of the model folds the part of the window that
        # would leave it back inside (the method of images), so that a point
        # next to that face interpolates exactly as a hand-built ``Hicks``
        # carrying the same surfaces does. Faces Deepwave cannot image here,
        # and windows that leave an absorbing face, are refused by the check
        # rather than silently truncated.
        _check_interpolated_bounds(
            tensor,
            halfwidth,
            offset,
            eps,
            grid,
            name,
            differentiable=differentiable,
        )
        # ``Hicks`` cannot build a surface image differentiably, so a
        # ``requires_grad`` location Tensor is built without the surfaces --
        # which is exact for every point whose window stays inside the model,
        # and the check above has already refused the ones whose does not.
        surfaces = (any(grid.free) or any(grid.rigid)) and not differentiable
        return Hicks(
            tensor,
            halfwidth=halfwidth,
            offset=offset,
            dtype=dtype,
            differentiable=differentiable,
            free_surfaces=list(grid.free) if surfaces else None,
            rigid_surfaces=list(grid.rigid) if surfaces else None,
            model_shape=list(grid.model_shape) if surfaces else None,
        )
    return None


def _grid_locations(
    locations: Union[torch.Tensor, "Hicks", None],
    interpolator: Optional["Hicks"],
    offset: Optional[list[float]] = None,
) -> Optional[torch.Tensor]:
    """Return the integer grid locations to hand to the propagator backend.

    When an ``interpolator`` is present, its interpolated grid locations are
    used. Otherwise a floating-point location Tensor that did not need
    interpolation (it lies on a component node) is snapped to that component's
    nearest integer array index: ``round(location - offset)``. For the regular
    grid (``offset`` zero/None) this is just ``round(location)``. Because an
    on-node float rounds to the same index an integer input would give, the
    result is bit-identical to passing the integer cells directly. Integer
    Tensors and ``None`` are returned unchanged.

    This rounds *on-node* floats only: an off-node float never reaches here in
    ``location_interpolation="none"`` mode (:func:`_resolve_interpolator` raises
    for it) nor in ``"auto"``/``"hicks"`` mode (an interpolator is built), so
    this snap never silently moves an off-node coordinate.

    Args:
        locations: The original location argument.
        interpolator: The resolved interpolator from
            :func:`_resolve_interpolator`, or ``None``.
        offset: Optional per-dimension half-cell component offset (see
            :class:`Hicks`), so the snap maps to the component's array index.

    Returns:
        An integer location Tensor (or ``None``).

    """
    if interpolator is not None:
        return interpolator.get_locations()
    if isinstance(locations, torch.Tensor) and locations.is_floating_point():
        offset_t = _offset_tensor(offset, locations)
        if offset_t is not None:
            return (locations - offset_t).round().long()
        return locations.round().long()
    # `interpolator` is None here, and by the calling convention established by
    # `_resolve_interpolator` (whose `isinstance(locations, Hicks)` branch
    # returns the Hicks unchanged), a `Hicks` `locations` always produces a
    # non-None interpolator. So `locations` can never be a `Hicks` at this
    # point; this cast just states that invariant for mypy (no runtime effect).
    return cast("Optional[torch.Tensor]", locations)


def _combine_weights(dim_weights: list[torch.Tensor]) -> torch.Tensor:
    """Combine per-dimension Hicks weights into a flat outer product.

    The flattening order (first dimension slowest, last dimension fastest)
    matches the ordering of ``torch.cartesian_prod`` used to build the
    corresponding interpolated grid locations.

    Args:
        dim_weights: A list of ``ndim`` 1-D weight tensors, one per dimension.

    Returns:
        A 1-D tensor of length ``prod(len(w) for w in dim_weights)``.

    """
    combined = dim_weights[0]
    for weights_d in dim_weights[1:]:
        combined = combined.unsqueeze(-1) * weights_d
    return combined.reshape(-1)


def _check_shot_idxs(
    amplitudes: torch.Tensor,
    shot_idxs: Optional[torch.Tensor] = None,
) -> None:
    if shot_idxs is not None and shot_idxs.shape != (len(amplitudes),):
        raise DeepwaveRuntimeError("shot_idxs must have the same length as amplitudes")


class Hicks:
    """Location interpolation onto grid using method of Hicks.

    Hicks (2002, https://doi.org/10.1190/1.1451454) proposed
    using a Kaiser windowed sinc function to interpolate
    source and receiver locations onto a grid when they are
    not centred on a grid point. This class implements this
    approach. It can also be used to create dipole sources
    and receivers.

    The interpolator supports 1, 2, and 3 spatial dimensions (matching the
    Deepwave propagator being used) and is staggered-grid aware via the
    ``offset`` argument, so a field component that does not live on the
    regular-grid (pressure) node can still be interpolated about its true
    location.

    Args:
        locations:
            A three dimensional torch.Tensor [shot, per_shot, ndim] specifying
            the locations to interpolate, provided as float values in
            units of cells relative to the origin of the grid, so
            [[[0.5, 0.5]]] corresponds to the point half a cell from
            the origin in both dimensions. ``ndim`` may be 1, 2, or 3.
        halfwidth:
            An integer specifying the halfwidth of the window that
            will be used to interpolate each point onto the grid. A
            halfwidth of 4 (the default) means that an 8x8 window will
            be used. Possible values are in [1, 10].
        free_surfaces:
            A sequence of ``2 * ndim`` booleans specifying whether the
            corresponding edge of the grid is a free surface, in the order:
            [beginning of first dimension, end of first dimension,
            beginning of second dimension, end of second dimension, ...].
            For example, in 2D `[True, False, False, True]` means that the
            beginning of the first dimension and end of the second dimension are
            free surfaces, while the other two edges are not.
            Defaults to no edges being free surfaces.

            A free surface is a point at which the field is zero (an
            odd/antisymmetric mirror, also called a Dirichlet or
            pressure-release boundary). Part of the interpolation window that
            extends past the surface is folded back into the model with a sign
            flip (the method of images).

            .. note::

                **Matching a Deepwave boundary.** Which condition a face with
                ``pml_width = 0`` produces, and where its mirror plane sits,
                depends on the propagator and the field component; the
                measured table is at :ref:`bare-zero-width-face`, and is the
                single source for it. What ``Hicks`` needs from that table is
                only the surface *location*, which is what
                ``free_surface_locs`` defaults to: one cell outside the model
                edge for a free surface, half a cell outside for a rigid one.
                Choose the surface type and location to match the propagator
                and field component you are placing.

                ``scalar`` and ``scalar_born`` need no ``Hicks`` of their own
                for this: their source and receiver points are on the regular
                grid and their ``boundary="free"``/``"rigid"`` faces mirror
                about exactly these default planes, so a plain float location
                next to such a face is folded automatically, giving bit for
                bit what building this class with those surfaces gives. The
                other seven propagators carry their components on staggered
                nodes, where the image is component dependent, and refuse a
                float location whose window would cross such a face.
        free_surface_locs:
            An optional sequence of ``2 * ndim`` floats giving the location (in
            cells) of each face named in ``free_surfaces``/``rigid_surfaces``,
            in the same order. Defaults to the location Deepwave's own
            boundaries use: one cell outside the model edge (``-1``/
            ``model_shape``) for a free surface, half a cell outside
            (``-0.5``/``model_shape - 0.5``) for a rigid surface.
        model_shape:
            A sequence of ``ndim`` integers specifying the size of the grid.
            This is only used when the model contains free or rigid surfaces.
        monopole:
            A boolean or torch.Tensor of booleans (of shape [shot, per_shot])
            specifying whether the source/receiver is a monopole. If
            False, the point will be a dipole. Default True.
        dipole_dim:
            An integer or torch.Tensor of integers (of shape [shot, per_shot])
            in ``[0, ndim)`` specifying the dimension in which the
            dipole is oriented. This is only used for points that are
            dipoles (not monopoles). Default 0.
        dtype:
            The datatype to use for the interpolation weights. ``None`` (the
            default) means ``torch.get_default_dtype()``, matching the other
            ``dtype`` parameters in the package. Under the usual float32
            default this is the dtype the weights were always built in, so a
            propagator's results are bit-identical. Under a float64 default
            the weights are built in float64 and then promoted (or demoted)
            to the propagator's own dtype when they are used, which is a
            different rounding path from building them in float32: measured
            differences of order 1e-6 relative in the weights themselves and
            in receiver amplitudes from a float32 model, and of order 5e-8
            relative from a float64 model. Pass ``dtype=torch.float`` to
            build the weights in float32 regardless of the default.
        eps:
            A small value to prevent division by zero. Points closer to
            a grid cell centre than this will be rounded to the grid cell.
            Default 1e-5.
        offset:
            An optional sequence of ``ndim`` floats giving the half-cell offset
            of the field component being interpolated relative to the
            regular-grid (pressure) node, used to make interpolation
            staggered-grid aware. Defaults to zero offset in every dimension
            (regular grid).

            .. note::

                **Staggered surfaces.** The default surface locations and the
                ``offset`` shift applied to them are validated for the regular
                grid (zero ``offset``), which covers the ``scalar`` free surface
                and the ``acoustic`` pressure (rigid) surface. For a *staggered*
                component (non-zero ``offset``) the image symmetry is component
                dependent (Robertsson, 1996): different components reflect with
                different signs and about different planes. Out-of-model cells
                are always dropped so the result is safe to use, but to be
                physically exact for a staggered component you should pass the
                correct ``free_surfaces``/``rigid_surfaces`` and explicit
                ``free_surface_locs`` for that component.
        rigid_surfaces:
            A sequence of ``2 * ndim`` booleans specifying whether the
            corresponding edge of the grid is a rigid surface, in the same
            order as ``free_surfaces``. Defaults to no edges being rigid
            surfaces. A face must not be both a free and a rigid surface.

            A rigid surface is a point at which the field has zero normal
            gradient (an even/symmetric mirror, also called a Neumann or
            zero-normal-velocity boundary). The part of the interpolation window
            past the surface is folded back into the model with the **same** sign
            (no flip); a point exactly on the surface has its weight doubled.
        differentiable:
            If True, the interpolation weights are built as differentiable torch
            expressions of ``locations`` (with no float-keyed weight cache), so
            gradients flow from :meth:`source`/:meth:`receiver` back to the
            source/receiver positions. This enables gradient-based survey design
            and source localisation. Default False (the fast, cached,
            bit-for-bit-stable path).

            Stability: provisional (see :doc:`/stability`).

            Weights are differentiable within a cell. At integer crossings the
            support changes: monopole values remain continuous, but their
            position gradients can jump; dipole values can jump too. The
            support indices are detached from autograd.

            ``float64`` is recommended for gradient work. ``IGNORE_LOCATION``
            sentinel points are handled as usual (their gradient is zero). Free
            and rigid surfaces are **not** supported in this mode (a
            ``NotImplementedError`` is raised); dipole sources/receivers are
            supported. See the "Differentiable positions" section of the usage
            documentation for the recommended recipe.

            .. note::

                A ``differentiable=True`` instance holds its interpolation
                weights **in the autograd graph** (they are torch expressions of
                ``locations``). Calling ``backward()`` frees that graph, so a
                second backward through the *same* instance raises the usual
                "backward through the graph a second time" error. In an
                optimisation loop either construct a **fresh** ``Hicks`` instance
                each iteration (the normal pattern: build it from the current
                positions inside the loop), or pass ``retain_graph=True`` to
                ``backward()`` if you deliberately reuse one instance across
                steps.

    """

    def __init__(
        self,
        locations: torch.Tensor,
        halfwidth: int = 4,
        *,
        free_surfaces: Optional[Sequence[bool]] = None,
        free_surface_locs: Optional[Sequence[float]] = None,
        model_shape: Optional[Sequence[int]] = None,
        monopole: Union[torch.Tensor, bool] = True,
        dipole_dim: Union[torch.Tensor, int] = 0,
        dtype: Optional[torch.dtype] = None,
        eps: float = _DEFAULT_EPS,
        offset: Optional[Sequence[float]] = None,
        rigid_surfaces: Optional[Sequence[bool]] = None,
        differentiable: bool = False,
    ) -> None:
        """Initialises the Hicks interpolation class.

        See the class docstring for a description of the arguments.
        """
        if not isinstance(differentiable, bool):
            raise DeepwaveTypeError("differentiable must be a bool.")
        if not isinstance(locations, torch.Tensor):
            raise DeepwaveTypeError("locations must be a torch.Tensor.")
        if locations.ndim != 3:
            raise DeepwaveRuntimeError("locations must have three dimensions.")
        ndim = locations.shape[2]
        if ndim not in (1, 2, 3):
            raise DeepwaveRuntimeError(
                "locations must have a final dimension of size 1, 2, or 3.",
            )
        # The dtype Deepwave 0.0.27 would have built this call's weights in:
        # whatever ``dtype=`` it was given, which defaulted to ``torch.float``
        # (float32) and never consulted the default dtype. Captured before the
        # omitted argument is resolved, and used only for the changed-stencil
        # comparison, which must not follow the current default dtype or it
        # would miss a genuine window change under a float64 default.
        v0027_dtype = torch.float32 if dtype is None else dtype
        if dtype is None:
            dtype = torch.get_default_dtype()
            _warn_hicks_dtype_migration(dtype)
        if not isinstance(dtype, torch.dtype):
            raise DeepwaveTypeError("dtype must be a torch.dtype or None.")
        if not isinstance(halfwidth, int):
            raise DeepwaveTypeError("halfwidth must be an integer")
        if halfwidth < 1 or halfwidth > 10:
            raise DeepwaveRuntimeError("halfwidth must be in [1, 10]")
        if free_surfaces is None:
            free_surfaces = [False] * (2 * ndim)
        if (
            not isinstance(free_surfaces, Sequence)
            or len(free_surfaces) != 2 * ndim
            or any(not isinstance(f, bool) for f in free_surfaces)
        ):
            raise DeepwaveRuntimeError(
                f"free_surfaces must be a sequence of {2 * ndim} bools",
            )
        free_surfaces = list(free_surfaces)
        if rigid_surfaces is None:
            rigid_surfaces = [False] * (2 * ndim)
        if (
            not isinstance(rigid_surfaces, Sequence)
            or len(rigid_surfaces) != 2 * ndim
            or any(not isinstance(f, bool) for f in rigid_surfaces)
        ):
            raise DeepwaveRuntimeError(
                f"rigid_surfaces must be a sequence of {2 * ndim} bools",
            )
        rigid_surfaces = list(rigid_surfaces)
        if any(f and r for f, r in zip(free_surfaces, rigid_surfaces)):
            raise DeepwaveRuntimeError(
                "a face cannot be both a free and a rigid surface",
            )
        if differentiable:
            if not locations.is_floating_point():
                raise DeepwaveRuntimeError(
                    "differentiable=True requires floating-point locations.",
                )
            if any(free_surfaces) or any(rigid_surfaces):
                raise DeepwaveNotImplementedError(
                    "differentiable=True does not support free or rigid "
                    "surfaces; build the Hicks without surfaces (gradients still "
                    "flow to positions), or use differentiable=False.",
                )
        if (any(free_surfaces) or any(rigid_surfaces)) and model_shape is None:
            raise DeepwaveRuntimeError(
                "If there are free or rigid surfaces then model_shape must be "
                "specified",
            )
        if model_shape is not None:
            if not isinstance(model_shape, Sequence) or len(model_shape) != ndim:
                raise DeepwaveRuntimeError(
                    f"model_shape must be a sequence with {ndim} int entries",
                )
            try:
                model_shape = [int(v) for v in model_shape]
            except (ValueError, TypeError) as exc:
                raise DeepwaveTypeError(
                    "model_shape entries must be convertible to int",
                ) from exc
        if free_surface_locs is not None:
            if (
                not isinstance(free_surface_locs, Sequence)
                or len(free_surface_locs) != 2 * ndim
            ):
                raise DeepwaveRuntimeError(
                    f"free_surface_locs must be a sequence of {2 * ndim} floats",
                )
            try:
                free_surface_locs = [float(v) for v in free_surface_locs]
            except (ValueError, TypeError) as exc:
                raise DeepwaveTypeError(
                    "free_surface_locs entries must be convertible to float",
                ) from exc
        if model_shape is None:
            model_shape = [-1] * ndim
        if free_surface_locs is None:
            # Per-face default surface plane, matching the location at which the
            # Deepwave propagators impose the corresponding boundary on a regular
            # (non-staggered) grid:
            #   - free surface (scalar zero-PML face): one cell outside the
            #     model edge, i.e. -1 at the start and ``model_shape`` at the end
            #     (the field is held to zero in the exterior ghost cell);
            #   - rigid surface (acoustic-pressure zero-PML face): half a cell
            #     outside the model edge, i.e. -0.5 and ``model_shape - 0.5``
            #     (the exterior velocity node is held to zero).
            # Staggered components shift these by the component ``offset``; see
            # the class docstring note on staggered surfaces.
            free_surface_locs = []
            for d in range(ndim):
                if free_surfaces[2 * d]:
                    free_surface_locs.append(-1.0)
                else:
                    free_surface_locs.append(-0.5)
                if free_surfaces[2 * d + 1]:
                    free_surface_locs.append(float(model_shape[d]))
                else:
                    free_surface_locs.append(model_shape[d] - 0.5)
        if offset is None:
            offset = [0.0] * ndim
        if not isinstance(offset, Sequence) or len(offset) != ndim:
            raise DeepwaveRuntimeError(f"offset must be a sequence of {ndim} floats")
        try:
            offset = [float(v) for v in offset]
        except (ValueError, TypeError) as exc:
            raise DeepwaveTypeError(
                "offset entries must be convertible to float"
            ) from exc
        if isinstance(monopole, torch.Tensor) and (
            monopole.shape[0] != locations.shape[0]
            or monopole.shape[1] != locations.shape[1]
        ):
            raise DeepwaveRuntimeError("monopole must have dimensions [shot, per_shot]")
        if isinstance(dipole_dim, torch.Tensor) and (
            dipole_dim.shape[0] != locations.shape[0]
            or dipole_dim.shape[1] != locations.shape[1]
        ):
            raise DeepwaveRuntimeError(
                "dipole_dim must have dimensions [shot, per_shot]"
            )
        if isinstance(dipole_dim, torch.Tensor):
            if dipole_dim.numel() and (
                int(dipole_dim.min()) < 0 or int(dipole_dim.max()) >= ndim
            ):
                raise DeepwaveRuntimeError("dipole_dim entries must be in [0, ndim)")
        elif not 0 <= dipole_dim < ndim:
            raise DeepwaveRuntimeError("dipole_dim must be in [0, ndim)")
        betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
        beta = torch.tensor(betas[halfwidth - 1]).to(dtype).to(locations.device)
        self.differentiable = differentiable
        self.locations = locations
        # Retained so that a propagator can tell whether a user-supplied
        # Hicks was built for the component it is being used for; see
        # ``_warn_on_offset_mismatch``.
        self.offset: list[float] = list(offset)
        # Points whose every coordinate equals IGNORE_LOCATION are disabled:
        # they are replaced by a safe in-bounds coordinate for the weight
        # computation, and their source amplitudes / received data are zeroed
        # (see `source` and `receiver`). This preserves the IGNORE_LOCATION
        # convention when interpolation is used. The safe coordinate is the
        # component's own node (``offset``), so that in the component's index
        # space it lands exactly on grid cell 0 (``offset - offset = 0``); using
        # a bare 0 would be off-node for a staggered component (offset 0.5),
        # producing a windowed stencil that reaches out-of-model negative cells.
        self.ignore_mask = (
            (locations == _common.IGNORE_LOCATION).all(dim=-1)
            if locations.numel()
            else torch.zeros(
                locations.shape[:2],
                dtype=torch.bool,
                device=locations.device,
            )
        )
        safe_locations = locations
        if bool(self.ignore_mask.any()):
            safe_locations = locations.clone()
            safe_locations[self.ignore_mask] = torch.tensor(
                offset,
                dtype=locations.dtype,
                device=locations.device,
            )
        self.hicks_locations, self.idxs, self.weights = (
            _get_hicks_locations_and_weights(
                safe_locations,
                halfwidth,
                beta,
                free_surfaces,
                free_surface_locs,
                model_shape,
                monopole,
                dipole_dim,
                eps,
                offset,
                rigid_surfaces,
                differentiable,
                v0027_dtype,
            )
        )

    def get_locations(self, shot_idxs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the interpolated locations.

        The interpolated locations can be provided to a Deepwave
        propagator as the source or receiver locations.

        Args:
            shot_idxs:
                A 1D torch.Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            A torch.Tensor containing the interpolated locations.

        """
        if shot_idxs is not None:
            return self.hicks_locations[shot_idxs]
        return self.hicks_locations

    def source(
        self,
        amplitudes: torch.Tensor,
        shot_idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the amplitudes of the interpolated sources.

        Args:
            amplitudes:
                A torch.Tensor containing the amplitudes of the sources at the
                original locations.
            shot_idxs:
                A 1D torch.Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            The amplitudes of the interpolated sources, ready to be provided
            to a Deepwave propagator.

        """
        if not isinstance(amplitudes, torch.Tensor):
            raise DeepwaveTypeError("amplitudes must be a torch.Tensor.")
        if amplitudes.ndim != 3:
            raise DeepwaveRuntimeError("amplitudes must have three dimensions.")

        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, n_per_shot, nt = amplitudes.shape
        n_per_shot_hicks = self.hicks_locations.shape[1]
        # In the below I put `out` on the CPU because there seems to
        # be a bug (in PyTorch?) that causes the incorrect answer
        # otherwise
        out = torch.zeros(
            n_shots,
            n_per_shot_hicks,
            nt,
            dtype=amplitudes.dtype,
            device=torch.device("cpu"),
        )
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                if bool(self.ignore_mask[hicks_shotidx, i]):
                    continue  # ignored source contributes nothing
                combined = _combine_weights(self.weights[hicks_shotidx][i]).to(
                    device=amplitudes.device
                )
                out[shotidx, self.idxs[hicks_shotidx][i], :] += (
                    amplitudes[shotidx, i][None] * combined[..., None]
                ).cpu()
        return out.to(amplitudes.device)

    def receiver(
        self,
        amplitudes: torch.Tensor,
        shot_idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert receiver amplitudes from interpolated to original locations.

        Args:
            amplitudes:
                A torch.Tensor containing the amplitudes recorded at interpolated
                receiver locations.
            shot_idxs:
                A 1D torch.Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            The amplitudes of receivers at the original locations.

        """
        if not isinstance(amplitudes, torch.Tensor):
            raise DeepwaveTypeError("amplitudes must be a torch.Tensor.")
        if amplitudes.ndim != 3:
            raise DeepwaveRuntimeError("amplitudes must have three dimensions.")

        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, _, nt = amplitudes.shape
        n_per_shot = self.locations.shape[1]
        out = torch.zeros(
            n_shots,
            n_per_shot,
            nt,
            dtype=amplitudes.dtype,
            device=amplitudes.device,
        )
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                if bool(self.ignore_mask[hicks_shotidx, i]):
                    continue  # ignored receiver records nothing (stays zero)
                combined = _combine_weights(
                    [w.to(amplitudes.device) for w in self.weights[hicks_shotidx][i]],
                )
                out[shotidx, i, :] = (
                    amplitudes[shotidx, self.idxs[hicks_shotidx][i]]
                    * combined[..., None]
                ).sum(dim=0)
        return out

"""Common helper functions for Deepwave propagators.

This module provides a collection of utility functions used across various
Deepwave propagators. These functions handle tasks such as input validation,
PML setup, and data preparation for wave propagation simulations.

The wavefield storage-abstraction classes (``StorageMode``, ``TemporaryStorage``,
``IntermediateStorage``, ``StorageManager``) live in :mod:`deepwave._storage`,
and the shared propagation engine (``run_backend_loop``, ``PropagatorSpec``,
``PropagatorOutput``, ``BackendState``, ``ForwardSavedState``,
``run_forward_engine``, ``run_backward_core``, ``run_backward``, the
canonical-call and functorch/second-order helpers, etc.) lives in
:mod:`deepwave._engine`. Both modules were split out of this one (task Q of
the propagator-engine refactor). Call sites (the propagator modules) import
directly from :mod:`deepwave._storage`/:mod:`deepwave._engine`.
"""

import contextlib
import importlib
import inspect
import itertools
import math
from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
)

import torch
import torch._C._functorch as _functorch
import torch.fft

from deepwave._exceptions import (
    DeepwaveFloatingPointError,
    DeepwaveNotImplementedError,
    DeepwaveRuntimeError,
    DeepwaveTypeError,
    DeepwaveValueError,
    MigrationWarning,
    PmlFreqInferenceWarning,
    ZeroWidthFaceWarning,
    warn_at_user_call,
)
from deepwave._storage import (
    STORAGE_DEFAULT,
    ResolvedStorage,
    resolve_storage_arguments,
    warn_if_default_disk_path,
    warn_if_storage_path_too_small,
)

# ``_StorageModeEnum`` is the backend's private ``IntEnum`` of storage
# codes; the public ``StorageMode`` imported below is the *string* alias
# that appears in signatures. They are different things with almost the
# same name, so the enum is aliased here to keep the two apart. The enum
# is never re-exported.
from deepwave._storage import StorageMode as _StorageModeEnum
from deepwave._storage import storage_mode_enum as _storage_mode_enum
from deepwave.types import (
    ON_NODE_TOLERANCE,
    Accuracy,
    LocationsLike,
    MemoryEstimate,
    StorageLike,
    StorageMode,
    StoragePath,
)

if TYPE_CHECKING:
    import deepwave._engine
    import deepwave._location_interpolation


def apply_autograd_function(function: Any, *args: Any) -> Any:
    """Call a ``torch.autograd.Function.apply`` method, which is untyped."""
    return function(*args)


# Some warnings interpolate a value derived from the current model (e.g. its
# velocity range), which can change between calls within the same process --
# an FWI loop's evolving model being the typical case. Python's default
# warning filter only de-duplicates byte-identical messages, so such warnings
# would otherwise re-fire on every call. This set instead de-duplicates by a
# fixed per-cause key, so a guarded warning fires at most once per process
# regardless of how the interpolated value changes. Tests that need to
# re-trigger a guarded warning should clear the relevant entry (or the whole
# set) first.
_warned_once_causes: set[str] = set()


def _torch_is_compiling() -> bool:
    """Use the compiler predicate available in this PyTorch version."""
    compiler = getattr(torch, "compiler", None)
    predicate = getattr(compiler, "is_compiling", None)
    if predicate is not None:
        return bool(predicate())
    # torch.compiler.is_compiling was introduced after the declared floor.
    import torch._dynamo as dynamo

    return bool(dynamo.is_compiling())


def _already_warned_once(cause: str) -> bool:
    """Whether :func:`_warn_once_per_cause` has already spent ``cause``.

    Lets a caller skip work that exists only to decide whether to emit a
    once-per-process warning: once the message has been said, computing
    whether it applies again is wasted. It is deliberately *not* used to
    suppress anything else, so a stale ``True`` can only cost a message that
    was already given.
    """
    return cause in _warned_once_causes


def _warn_once_per_cause(
    cause: str, message: str, category: type[Warning], stacklevel: int
) -> None:
    """Emits `message` at most once per process for a given `cause`.

    A :func:`deepwave.plan` in progress on this context is deliberately not
    allowed to spend the budget: the plan reports the warning, and the real
    call the user makes afterwards must still emit it. The registry is
    therefore left untouched while planning, rather than written and then
    unwound afterwards. Unwinding was the older approach and it was not safe:
    the restore intersected the registry with a snapshot taken before the
    plan, so a cause another thread's *real* run registered while the plan was
    running was discarded with the plan's own, silently re-arming a warning
    that thread had already emitted. Not writing in the first place cannot
    race with anyone.

    Args:
        cause: A fixed key identifying the reason for the warning (not the
            formatted message, which may vary between calls).
        message: The formatted warning text.
        category: The warning category (e.g. `UserWarning`).
        stacklevel: The stack level to fall back on if every frame above this
            one belongs to Deepwave or PyTorch, counted from this function's
            caller. The level actually used is
            :func:`deepwave._exceptions.stacklevel_outside_deepwave`'s, so
            that the warning is reported at the user's own line whichever
            form they called through.

    """
    if cause in _warned_once_causes:
        return
    from deepwave import _plan

    if _plan.active_probe() is None:
        _warned_once_causes.add(cause)
    # ``stacklevel`` is counted from this function's caller and
    # ``warn_at_user_call``'s fallback from its own, one frame lower.
    warn_at_user_call(message, category, stacklevel + 1)


def _round_3sf(x: float) -> float:
    """Rounds `x` to 3 significant figures (`0.0` unchanged)."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, 2 - math.floor(math.log10(abs(x))))


def _fmt_3sf(x: float) -> str:
    """Formats `x` to 3 significant figures without forcing scientific notation.

    Plain `f"{x:.3g}"` switches to scientific notation once `x` reaches 1000
    (e.g. `1500.0` -> `"1.5e+03"`), which is unhelpful for the everyday
    velocity/frequency magnitudes these warnings report. Rounding to 3
    significant figures first and then formatting with `g` keeps ordinary
    magnitudes in fixed-point form.
    """
    if x == 0:
        return "0"
    return f"{_round_3sf(x):g}"


# Sentinel marking an unused source/receiver location; documented at its
# public re-export in ``deepwave/__init__.py`` (rendered via autodata as
# ``deepwave.IGNORE_LOCATION`` in docs/usage.rst).
IGNORE_LOCATION = -1 << 31


def mask_locations(
    locations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits source/receiver location indices into a safe copy and a mask.

    Entries equal to :data:`IGNORE_LOCATION` mark sources/receivers that should
    be skipped. The Python reference backends gather from and scatter to every
    location unconditionally and zero the ignored contributions afterwards, so
    they need a copy of the indices with the ignored entries replaced by a safe
    index (0) together with the boolean mask of the valid entries.

    Args:
        locations: The 1D flattened location indices (per shot).

    Returns:
        A tuple ``(masked_locations, mask)`` where ``mask`` is ``True`` at the
        valid (non-ignored) locations and ``masked_locations`` is ``locations``
        with the ignored entries set to ``0``.

    Note:
        This used to be a boolean-mask assignment (``masked[mask] =
        locations[mask]``, guarded by a ``numel() > 0`` check to avoid
        indexing an empty tensor). Under ``torch.compile``/``torch.export``,
        boolean-mask assignment lowers to a data-dependent ``aten.nonzero``
        (the number of ``True`` entries is not known until the mask's values
        are read), which is opaque to Dynamo and forced two graph breaks
        (source and receiver locations each call this once). The
        ``torch.where`` form below is an elementwise select: it has the same
        (static) output shape as its inputs regardless of how many entries
        are masked, so it is fully traceable and produces byte-identical
        results (including on an empty ``locations``, where the ``numel()``
        guard is now unnecessary).
    """
    mask = locations != IGNORE_LOCATION
    masked = torch.where(mask, locations, torch.zeros_like(locations))
    return masked, mask


def mask_source_amplitudes(
    source_amplitudes: torch.Tensor, source_mask: torch.Tensor
) -> torch.Tensor:
    """Zeros the amplitudes of ignored sources (see :func:`mask_locations`)."""
    masked = torch.zeros_like(source_amplitudes)
    if source_amplitudes.numel() > 0:
        masked[:, source_mask] = source_amplitudes[:, source_mask]
    return masked


def mask_receiver_amplitudes(
    receiver_amplitudes: torch.Tensor, receiver_mask: torch.Tensor
) -> torch.Tensor:
    """Zeros the amplitudes of ignored receivers (see :func:`mask_locations`).

    Returns a new tensor (never mutates ``receiver_amplitudes`` in place):
    the eager/compiled Python backends gather from every receiver
    location unconditionally and zero the ignored contributions afterwards
    with this shared finalise step.
    """
    masked = torch.zeros_like(receiver_amplitudes)
    if receiver_amplitudes.numel() > 0:
        masked[:, receiver_mask] = receiver_amplitudes[:, receiver_mask]
    return masked


# The internal spelling of a location argument: the public
# ``deepwave.LocationsLike``, plus ``None`` for "this component has no
# locations". One alias, so the internal and public statements of what a
# location argument may be cannot drift apart.
_LocationArg = Optional[LocationsLike]


def velocity_component_offsets(ndim: int) -> list[list[float]]:
    """Per-dimension half-cell offsets for the ``ndim`` velocity components.

    On the staggered acoustic/elastic grids the pressure / normal-stress nodes
    sit on the integer grid ``(z, y, x)``, while a velocity component that
    propagates (is staggered) in spatial dimension ``d`` lives half a cell away
    in ``d`` only: ``vz`` at ``(z+½, y, x)``, ``vy`` at ``(z, y+½, x)``, ``vx``
    at ``(z, y, x+½)``. This returns those offsets ordered by spatial dimension
    (slowest to fastest), i.e. ``[vz, vy, vx]`` restricted to the trailing
    ``ndim`` dimensions, so the offset for the component staggered in dimension
    ``d`` is 0.5 in ``d`` and zero elsewhere. Passing such an offset to the
    :class:`deepwave.Hicks` interpolator makes the
    off-grid spread/gather staggered-grid aware, preserving the invariant that
    integer index ``N`` and float ``N + offset`` denote the same physical point.
    """
    offsets: list[list[float]] = []
    for d in range(ndim):
        off = [0.0] * ndim
        off[d] = 0.5
        offsets.append(off)
    return offsets


def resolve_source_interpolators(
    source_locations: abc.Sequence[_LocationArg],
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    offsets: abc.Sequence[Optional[list[float]]],
    location_interpolation: str,
    location_interpolation_halfwidth: int,
    dtype: torch.dtype,
    grid: Optional["HicksGrid"] = None,
) -> tuple[list[Optional[torch.Tensor]], list[Optional[torch.Tensor]]]:
    """Resolve per-component off-grid source specifications.

    For each source component this builds the appropriate
    :class:`deepwave.Hicks` interpolator (using that
    component's half-cell ``offset``), differentiably spreads its amplitudes
    onto the grid, and converts its locations to the integer grid indices the
    backend consumes. Integer/``long`` locations take the exact path and are
    bit-for-bit identical to providing integer cell indices; a component whose
    location argument is a :class:`Hicks` instance uses it directly.

    Args:
        source_locations: Per-component location arguments (Tensor, Hicks, or
            None), in the propagator's own component order.
        source_amplitudes: Per-component source amplitudes, parallel to
            ``source_locations``.
        offsets: Per-component half-cell offsets, parallel to
            ``source_locations`` (``None`` for a regular-grid component).
        location_interpolation: The ``location_interpolation`` mode.
        location_interpolation_halfwidth: Hicks window halfwidth.
        dtype: Datatype for an auto-built Hicks.
        grid: The resolved model faces (see :func:`hicks_grid`), or ``None``
            to build every interpolator without knowledge of them.

    Returns:
        A tuple ``(source_amplitudes, source_locations)`` of the spread
        amplitudes and the integer grid locations, parallel to the inputs.
    """
    import deepwave._location_interpolation as _li

    new_amplitudes: list[Optional[torch.Tensor]] = []
    grid_locs: list[Optional[torch.Tensor]] = []
    names = _location_group_names(
        "source_locations",
        len(source_locations),
        None if grid is None else grid.propagator,
        None if grid is None else len(grid.model_shape),
        "source",
    )
    for loc, amp, off, name in zip(source_locations, source_amplitudes, offsets, names):
        interp = _li._resolve_interpolator(  # noqa: SLF001
            loc,
            location_interpolation,
            location_interpolation_halfwidth,
            offset=off,
            dtype=dtype,
            kind="source",
            grid=grid,
            name=name,
        )
        if interp is not None and amp is not None:
            amp = interp.source(amp)
        new_amplitudes.append(amp)
        grid_locs.append(_li._grid_locations(loc, interp, offset=off))  # noqa: SLF001
    return new_amplitudes, grid_locs


def resolve_receiver_interpolators(
    receiver_locations: abc.Sequence[_LocationArg],
    offsets: abc.Sequence[Optional[list[float]]],
    location_interpolation: str,
    location_interpolation_halfwidth: int,
    dtype: torch.dtype,
    grid: Optional["HicksGrid"] = None,
) -> tuple[
    list[Optional["deepwave._location_interpolation.Hicks"]],
    list[Optional[torch.Tensor]],
]:
    """Resolve per-component off-grid receiver specifications.

    The counterpart of :func:`resolve_source_interpolators` for receivers: it
    returns each component's interpolator (or ``None`` for the exact path) so
    the propagator can gather the recorded data back to the original locations
    with :func:`apply_receiver_interpolators` after propagation, together with
    the integer grid locations the backend records at.

    Args:
        receiver_locations: Per-component location arguments (Tensor, Hicks, or
            None), in the propagator's own component order.
        offsets: Per-component half-cell offsets, parallel to
            ``receiver_locations``.
        location_interpolation: The ``location_interpolation`` mode.
        location_interpolation_halfwidth: Hicks window halfwidth.
        dtype: Datatype for an auto-built Hicks.
        grid: The resolved model faces (see :func:`hicks_grid`), or ``None``
            to build every interpolator without knowledge of them.

    Returns:
        A tuple ``(interpolators, receiver_locations)`` parallel to the input.
    """
    import deepwave._location_interpolation as _li

    interps: list[Optional[deepwave._location_interpolation.Hicks]] = []
    grid_locs: list[Optional[torch.Tensor]] = []
    names = _location_group_names(
        "receiver_locations",
        len(receiver_locations),
        None if grid is None else grid.propagator,
        None if grid is None else len(grid.model_shape),
        "receiver",
    )
    for loc, off, name in zip(receiver_locations, offsets, names):
        interp = _li._resolve_interpolator(  # noqa: SLF001
            loc,
            location_interpolation,
            location_interpolation_halfwidth,
            offset=off,
            dtype=dtype,
            kind="receiver",
            grid=grid,
            name=name,
        )
        interps.append(interp)
        grid_locs.append(_li._grid_locations(loc, interp, offset=off))  # noqa: SLF001
    return interps, grid_locs


def apply_receiver_interpolators(
    outputs: list[torch.Tensor],
    interpolators: list[Optional["deepwave._location_interpolation.Hicks"]],
    start: int,
) -> None:
    """Gather off-grid receiver data back to the original locations, in place.

    ``outputs[start + i]`` holds the data recorded at the (interpolated) grid
    locations of receiver component ``i``; where an interpolator is present it
    is replaced by the data gathered back to that component's original
    locations. Components with no interpolator (the exact path) are untouched.
    """
    for i, interp in enumerate(interpolators):
        if interp is not None:
            outputs[start + i] = interp.receiver(outputs[start + i])


def gather_model_at_sources(
    model: torch.Tensor,
    locations: torch.Tensor,
    model_shape: abc.Sequence[int],
    n_shots: int,
) -> torch.Tensor:
    """Gathers model values at the (flattened) source grid locations.

    The finite-difference source injection scales each source amplitude by the
    model value at the source's grid cell. Locations flagged with
    :data:`IGNORE_LOCATION` are gathered from a safe index (0); the caller
    discards those contributions.

    Args:
        model: The model tensor, either batched ``(n_shots, *spatial)`` or
            shared across shots ``(*spatial)``.
        locations: The flattened per-shot source indices.
        model_shape: The spatial shape of the model.
        n_shots: The number of shots.

    Returns:
        A ``(n_shots, n_sources_per_shot)`` tensor of gathered model values.
    """
    flat_model_shape = 1
    for size in model_shape:
        flat_model_shape *= int(size)
    masked, _ = mask_locations(locations)
    return model.view(-1, flat_model_shape).expand(n_shots, -1).gather(1, masked)


def check_callback_frequency(callback_frequency: int) -> None:
    """Validates the ``callback_frequency`` propagator argument."""
    if not isinstance(callback_frequency, int):
        raise DeepwaveTypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise DeepwaveValueError("callback_frequency must be positive.")


def make_model_parameter(
    model: torch.Tensor, requires_grad: bool, name: str
) -> torch.nn.Parameter:
    """Validates a model tensor and its grad flag, returning an ``nn.Parameter``.

    Used by the propagator ``Module`` constructors, which each register one or
    more model tensors (e.g. ``v``, ``rho``) as parameters after the same pair
    of type checks.

    Args:
        model: The model tensor.
        requires_grad: Whether the parameter requires a gradient.
        name: The model name, used in error messages.

    Returns:
        ``model`` wrapped as a ``torch.nn.Parameter``.

    Raises:
        TypeError: If ``requires_grad`` is not a bool or ``model`` is not a
            ``torch.Tensor``.
    """
    if not isinstance(requires_grad, bool):
        raise DeepwaveTypeError(
            f"{name}_requires_grad must be bool, got {type(requires_grad).__name__}",
        )
    if not isinstance(model, torch.Tensor):
        raise DeepwaveTypeError(f"{name} must be a torch.Tensor.")
    return torch.nn.Parameter(model, requires_grad=requires_grad)


#: Bound by :func:`inherit_args` so that decorating a ``forward`` method
#: leaves its signature and type exactly as written.
F = TypeVar("F", bound=Callable[..., Any])


def check_requires_grad_flag(requires_grad: bool, name: str) -> None:
    """Validates one ``<model>_requires_grad`` flag.

    :func:`make_model_parameter` performs the same check, but only for a model
    that was actually supplied. An *optional* model left as ``None`` never
    reaches it, so its flag would otherwise go unvalidated and a typo such as
    ``dv_requires_grad="yes"`` would be silently accepted. The constructors
    call this before their ``None`` branch so that the flag is checked either
    way.

    Args:
        requires_grad: The flag as the caller supplied it.
        name: The model name, used in the error message.

    Raises:
        TypeError: If ``requires_grad`` is not a bool.
    """
    if not isinstance(requires_grad, bool):
        raise DeepwaveTypeError(
            f"{name}_requires_grad must be bool, got {type(requires_grad).__name__}",
        )


#: The one description of what a propagator ``Module`` does with the model
#: tensors it is given. Single-sourced and appended to all nine class
#: docstrings by :func:`document_propagator_module`, so that "the same
#: paragraph everywhere" is true by construction rather than by discipline.
#: Written dedented, the form ``inspect.cleandoc`` leaves a docstring in, and
#: deliberately naming no model so that it needs no per-class substitution.
MODEL_OWNERSHIP_DOC = """Model ownership and gradients
    Every model tensor given to the constructor is wrapped in a new
    ``torch.nn.Parameter`` that this module owns. The Parameter is a
    distinct leaf tensor that *shares storage* with the tensor you
    passed in: it is not a copy, so an in-place edit of one is visible
    in the other, but it is not the same object. Gradients therefore
    accumulate on the module's own attribute (``module.<model>.grad``)
    and never on the tensor you passed in. A model requires a gradient
    only if you ask for one, through the constructor's ``requires_grad``
    argument or by calling ``module.<model>.requires_grad_(True)``
    afterwards; hand ``module.parameters()`` to a ``torch.optim``
    optimiser as usual. An optional model left as ``None`` is not a
    Parameter at all, so it appears in neither ``parameters()`` nor
    ``state_dict()``; assigning a tensor to that attribute later stores
    a plain tensor rather than registering a Parameter, so it stays out
    of ``parameters()`` and ``state_dict()`` (and out of any checkpoint)
    -- give the model to the constructor, or wrap it yourself in
    ``torch.nn.Parameter``. Because the module owns a *leaf* Parameter, it
    cannot carry an upstream autograd graph: if the model is itself
    produced by another operation -- a network mapping a latent vector
    to a wavespeed, a reparameterisation, a smoothing filter -- wrapping
    it in a module severs that graph and no gradient reaches the
    producer. Use the functional interface for anything
    graph-producing; it takes the model as an ordinary tensor argument
    and backpropagates through it."""


def resolve_requires_grad(
    requires_grad: Optional[Union[bool, abc.Sequence[str]]],
    owned: abc.Sequence[str],
    supplied: abc.Sequence[str],
    individual: abc.Mapping[str, Optional[bool]],
    class_name: str,
) -> dict[str, bool]:
    """Resolves a constructor's ``requires_grad`` argument to a flag per model.

    Args:
        requires_grad: The one-argument form: ``None`` (defer entirely to the
            individual flags), ``True``/``False`` (every model this instance
            owns, or none of them), or a sequence of model names.
        owned: Every model name this class can own, in constructor order.
        supplied: The subset of ``owned`` that was actually given a tensor;
            an optional model left as ``None`` is not owned by the instance
            and so cannot be named.
        individual: The ``<model>_requires_grad`` flags this class accepts,
            by model name, with ``None`` for a flag the caller did not pass.
            Every propagator ``Module`` offers one per model it can own, so
            this covers all of ``owned``; it is keyed by the *public* model
            names, which is what the constructor takes.
        class_name: The Module class name, for error messages.

    Returns:
        One flag per name in ``owned``.

    Raises:
        ValueError: If ``requires_grad`` is combined with an explicitly
            given individual flag, names a model this class does not have,
            or names an optional model that was not supplied.
        TypeError: If ``requires_grad`` is neither a bool, a sequence of
            strings, nor ``None``.
    """
    # Validated here rather than in make_model_parameter, which an optional
    # model left as None never reaches: the flag must be rejected either way.
    # ``None`` means "not passed", which is why the flags are typed optional:
    # without that sentinel a default False is indistinguishable from an
    # explicit one, and the conflict check below can then only catch the half
    # of a contradiction that turns a gradient off.
    for name, flag in individual.items():
        if flag is not None:
            check_requires_grad_flag(flag, name)
            # The same rule the sequence form applies: a flag that names an
            # optional model this instance was not given cannot be honoured,
            # and silently dropping it would quietly withhold a gradient the
            # caller asked for.
            if name not in supplied:
                raise DeepwaveValueError(
                    f"{name}_requires_grad was given, but no {name} was given "
                    f"to {class_name}, so this module does not own one."
                )

    if requires_grad is None:
        return {name: bool(individual.get(name)) for name in owned}

    # Every explicitly given individual flag conflicts, whichever way it
    # points and even when it agrees: one rule ("pass one or the other"),
    # rather than a silent winner that in the requires_grad=True /
    # <model>_requires_grad=False direction would enable a gradient the
    # caller had just asked not to have.
    conflicting = sorted(name for name, flag in individual.items() if flag is not None)
    if conflicting:
        raise DeepwaveValueError(
            f"Pass only requires_grad, not both requires_grad and "
            f"{conflicting[0]}_requires_grad."
        )
    if isinstance(requires_grad, bool):
        chosen = set(supplied) if requires_grad else set()
    elif isinstance(requires_grad, str) or not isinstance(requires_grad, abc.Sequence):
        raise DeepwaveTypeError(
            "requires_grad must be a bool, a sequence of model names, or "
            f"None, got {type(requires_grad).__name__}."
        )
    else:
        chosen = set()
        for name in requires_grad:
            if name not in owned:
                valid = ", ".join(repr(model) for model in owned)
                raise DeepwaveValueError(
                    f"{class_name} has no model {name!r}; requires_grad may "
                    f"name any of: {valid}."
                )
            if name not in supplied:
                raise DeepwaveValueError(
                    f"requires_grad names {name!r}, but no {name} was given "
                    f"to {class_name}, so this module does not own one."
                )
            chosen.add(name)
    return {name: name in chosen for name in owned}


def _parse_args_entries(doc: str) -> list[str]:
    """Extracts the entries of a Google-style ``Args:`` block, verbatim.

    Args:
        doc: A docstring, already normalised by ``inspect.cleandoc`` so that
            its body starts at column zero and its ``Args:`` entries at an
            indent of four. (Python 3.13 dedents docstrings at compile time
            and earlier versions do not, so normalising is what makes this
            parser behave the same on both.)

    Returns:
        One string per documented argument, each keeping its own internal
        indentation and any nested list or blank lines, and each starting at
        a four-space indent. Empty if there is no ``Args:`` block.
    """
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == "Args:")
    except StopIteration:
        return []
    entries: list[str] = []
    current: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped and not line.startswith(" "):
            break  # the next section heading ("Returns:", "Raises:", ...)
        if stripped and line.startswith(" " * 4) and not line.startswith(" " * 5):
            if current:
                entries.append("\n".join(current).rstrip())
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append("\n".join(current).rstrip())
    return entries


def _entry_name(entry: str) -> str:
    """Returns the argument name an ``Args:`` entry documents, or ``""``."""
    head = entry.strip().split(":", 1)[0]
    return head if head.isidentifier() else ""


def inherit_args(functional: Callable[..., Any]) -> Callable[[F], F]:
    """Splices ``functional``'s ``Args:`` entries into a ``forward`` docstring.

    All nine ``Module.forward`` methods are two-line pointers at their
    functional, so ``help(Scalar.forward)`` and an IDE's signature popup show
    nothing about the forty-odd keyword arguments of the very interface the
    documentation advertises as the convenient one. This decorator copies the
    functional's own argument documentation onto ``forward`` at import time,
    so the two can never drift.

    Only entries naming a parameter that ``forward`` actually has are copied.
    That drops the constructor-owned models and storage settings without a
    hand-maintained list to fall out of date, and it correctly drops entries
    for arguments an individual Module does not accept.

    A no-op when either docstring is absent, so that ``python -OO`` (which
    strips docstrings) does not break the package.

    Args:
        functional: The propagator function whose ``Args:`` block is the
            source of truth.

    Returns:
        A decorator that returns its argument unchanged apart from
        ``__doc__``, so the signature and type are untouched.
    """

    def decorate(method: F) -> F:
        source = getattr(functional, "__doc__", None)
        doc = getattr(method, "__doc__", None)
        if not source or not doc:
            return method
        accepted = {
            name
            for name, parameter in inspect.signature(method).parameters.items()
            if name != "self"
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        cleaned = inspect.cleandoc(doc)
        if "\nArgs:\n" in cleaned:
            return method  # already spliced
        entries = [
            entry
            for entry in _parse_args_entries(inspect.cleandoc(source))
            if _entry_name(entry) in accepted
        ]
        if not entries:
            return method
        method.__doc__ = cleaned + "\n\nArgs:\n" + "\n".join(entries) + "\n"
        return method

    return decorate


def document_propagator_module(
    module_class: type[torch.nn.Module], functional: Callable[..., Any]
) -> None:
    """Completes a propagator ``Module``'s documentation at import time.

    Does two things that cannot be written in the class body itself: appends
    the shared :data:`MODEL_OWNERSHIP_DOC` paragraph to the class docstring,
    and splices the functional's ``Args:`` entries onto ``forward``
    (:func:`inherit_args`). Both need the functional, which every propagator
    module defines *after* its Module class, so this is called once at the
    foot of each of the nine module files rather than as a decorator.

    A no-op for whichever docstring is absent, so ``python -OO`` still
    imports.

    Args:
        module_class: The propagator ``Module`` subclass.
        functional: The propagator function it wraps.
    """
    inherit_args(functional)(module_class.forward)
    doc = module_class.__doc__
    if not doc or MODEL_OWNERSHIP_DOC in doc:
        return
    module_class.__doc__ = inspect.cleandoc(doc) + "\n\n" + MODEL_OWNERSHIP_DOC + "\n"


def check_models(
    models: abc.Sequence[Any],
    names: abc.Sequence[str],
    propagator: Optional[str] = None,
) -> None:
    """Rejects non-Tensor or empty model arguments, naming the offender.

    Deliberately cheap (an ``isinstance`` check and a ``numel()`` shape
    query, neither of which reads tensor values): safe to call as the very
    first thing a propagator does with its model arguments, before a
    vmapped-input check has run, and before any other code derives a
    dtype/ndim from them. Without this guard, a Python list or a 0-element
    Tensor passed as a model instead surfaces as an opaque ``AttributeError``
    or ``RuntimeError`` from whatever internal computation first touches it
    (e.g. ``'list' object has no attribute 'dtype'``), far from the actual
    mistake.

    It is also where the **shared model-shape check** lives. The
    cross-model shape comparison used to happen only in
    :func:`extract_models`, deep inside :func:`setup_propagator`, and
    several propagators multiply two models together before they get there
    (to find the maximum wave speed for the CFL condition and the PML:
    ``mu * buoyancy`` on the elastic families, ``c33 * buoyancy`` on the
    stiffness families). A mismatched pair therefore reached the user as
    ``torch`` 's own broadcast error -- "The size of tensor a (29) must
    match the size of tensor b (30)" -- which names neither the models nor
    the propagator. Checking here, in the one function every propagator
    already calls before any arithmetic, means all nine give the same
    message for the same mistake; :func:`extract_models` keeps its own
    check for the models it is handed directly.

    Args:
        models: The model tensors (e.g. velocity, density) passed to the
            propagator.
        names: A name for each entry in `models`, used in error messages
            (e.g. ``"v"``, ``"rho"``).
        propagator: The calling propagator's name, which every message it
            raises begins with. ``None`` omits the prefix, for the helpers
            and tests that call this without one.

    Raises:
        DeepwaveTypeError: If a model is not a ``torch.Tensor``.
        DeepwaveValueError: If a model Tensor has no elements, or if two
            models have different spatial shapes.

    """
    prefix = f"{propagator}: " if propagator else ""
    for name, model in zip(names, models):
        if not isinstance(model, torch.Tensor):
            raise DeepwaveTypeError(
                f"{prefix}{name} must be a torch.Tensor, but got "
                f"{type(model).__name__}. "
                "Convert a Python list with torch.tensor(...) or a NumPy "
                "array with torch.from_numpy(...) first."
            )
        if model.numel() == 0:
            raise DeepwaveValueError(
                f"{prefix}{name} must not be empty, but has shape {tuple(model.shape)}."
            )
    check_model_shapes_match(models, names, propagator)
    check_model_dtypes_and_devices(models, names, propagator)
    if propagator is not None and propagator != "elastic_vti":
        check_material_domain(dict(zip(names, models)), propagator)


def check_material_domain(
    models: abc.Mapping[str, torch.Tensor], propagator: str
) -> None:
    """Check readable physical backgrounds without projecting model values.

    Born perturbations are unconstrained directions. Values hidden by a tensor
    transform remain a caller precondition, while shape/device checks in
    :func:`check_models` still run before this function.
    """
    if (
        _torch_is_compiling()
        or any(m.dtype not in (torch.float32, torch.float64) for m in models.values())
        or not all(values_readable(m) for m in models.values())
    ):
        return

    def require(condition: torch.Tensor, description: str) -> None:
        if not bool(condition.all()):
            raise DeepwaveValueError(f"{propagator}: material requires {description}.")

    for name, model in models.items():
        if name == "q":
            # 1/Q is the attenuation strength; +inf is the exact lossless limit.
            # The comparison also rejects NaN, -inf and zero.
            require(model > 0, "q > 0 (positive infinity denotes the lossless limit)")
            continue
        finite = torch.isfinite(model)
        if not bool(finite.all()):
            count = int((~finite).sum())
            raise DeepwaveValueError(
                f"{name} contains {count} non-finite "
                f"value{'s' if count != 1 else ''} (NaN or Inf) out of "
                f"{model.numel()} elements; {propagator}: material requires "
                f"finite {name}."
            )

    if "v" in models and "rho" not in models:
        # These equations use v**2. Signed and zero speeds are well-defined;
        # stability and PML setup use the maximum absolute speed.
        return
    if "v" in models and "rho" in models:
        require(models["rho"] >= 0, "rho >= 0")
        return
    if {"lamb", "mu", "buoyancy"} <= models.keys():
        lamb, mu, buoyancy = (models[name] for name in ("lamb", "mu", "buoyancy"))
        active = buoyancy > 0
        require(buoyancy >= 0, "buoyancy >= 0")
        require(active | ((lamb == 0) & (mu == 0)), "zero stiffness in vacuum")
        ndim = lamb.ndim
        scale = torch.maximum(lamb.abs(), mu.abs()).clamp_min(1.0)
        if ndim == 1:
            modulus = lamb / scale + 2.0 * (mu / scale)
        elif ndim == 2:
            require(mu >= 0, "mu >= 0 in 2-D")
            modulus = lamb / scale + mu / scale
        else:
            require(mu >= 0, "mu >= 0 in 3-D")
            modulus = lamb / scale + (2.0 / 3.0) * (mu / scale)
        require(~active | (modulus > 0), "positive represented elastic energy")
        return
    if {"c11", "c13", "c33", "buoyancy"} <= models.keys():
        c11, c13, c33, buoyancy = (
            models[name] for name in ("c11", "c13", "c33", "buoyancy")
        )
        active = buoyancy > 0
        require(buoyancy >= 0, "buoyancy >= 0")
        stiffnesses = [c11, c13, c33]
        if "c55" in models:
            stiffnesses.append(models["c55"])
        if "c66" in models:
            stiffnesses.append(models["c66"])
        require(
            active | torch.stack([item == 0 for item in stiffnesses]).all(0),
            "zero stiffness in vacuum",
        )
        require(~active | ((c11 > 0) & (c33 > 0)), "c11, c33 > 0 in material")
        normal = c11 - models["c66"] if "c66" in models else c11
        safe_normal = torch.where(normal > 0, normal, torch.ones_like(normal))
        safe_c33 = torch.where(c33 > 0, c33, torch.ones_like(c33))
        coupling = (c13.abs() / safe_normal.sqrt()) / safe_c33.sqrt()
        roundoff = 8.0 * torch.finfo(c13.dtype).eps
        if "c55" in models:
            shear = models["c55"]
            require(shear >= 0, "c55 >= 0")
            if "c66" in models:
                require(models["c66"] >= 0, "c66 >= 0")
            require(~active | (normal > 0), "positive VTI normal stiffness")
            strict = (shear > 0) & (
                (models["c66"] > 0) if "c66" in models else torch.ones_like(active)
            )
            require(
                ~active | torch.where(strict, coupling < 1, coupling <= 1 + roundoff),
                "positive or permitted semidefinite VTI normal block",
            )
        else:
            require(
                ~active | (coupling <= 1 + roundoff),
                "c11 * c33 >= c13**2 (epsilon >= delta) in the "
                "positive semidefinite pseudo-acoustic block",
            )


def check_model_dtypes_and_devices(
    models: abc.Sequence[torch.Tensor],
    names: abc.Sequence[str],
    propagator: Optional[str] = None,
) -> None:
    """Rejects models that disagree in dtype or device, naming both of them.

    Every model of one call is read by one kernel on one device in one
    precision, so a model unlike the others is always a mistake. It used to
    be caught deep inside :func:`extract_models`, which knows neither the
    propagator nor the caller's argument names, and only after several
    propagators had already multiplied two models together -- so on the
    elastic and stiffness families a device mismatch surfaced as ``torch``'s
    own "Expected all tensors to be on the same device", which names nothing
    the caller wrote. Checking here, in the one function every propagator
    calls before any arithmetic, gives all nine the same message for the
    same mistake.

    Args:
        models: The model tensors, already known to be non-empty Tensors.
        names: A name for each entry in ``models``.
        propagator: The calling propagator's name, which the message begins
            with when it is known.

    Raises:
        DeepwaveRuntimeError: If two models differ in dtype or in device. It
            is a ``RuntimeError``, which is what this condition has always
            raised.

    """
    if len(models) < 2:
        return
    prefix = f"{propagator}: " if propagator else ""
    first_name, first = names[0], models[0]
    for name, model in zip(names[1:], models[1:]):
        if model.dtype != first.dtype:
            raise DeepwaveRuntimeError(
                f"{prefix}all models must have the same dtype, but "
                f"{first_name} has dtype {first.dtype} and {name} has dtype "
                f"{model.dtype}."
            )
        if model.device != first.device:
            raise DeepwaveRuntimeError(
                f"{prefix}all models must be on the same device, but "
                f"{first_name} is on {first.device} and {name} is on "
                f"{model.device}."
            )


def check_model_shapes_match(
    models: abc.Sequence[torch.Tensor],
    names: abc.Sequence[str],
    propagator: Optional[str] = None,
) -> None:
    """Rejects models whose spatial shapes differ, naming both of them.

    The spatial shape is the trailing ``ndim`` dimensions: a model may carry
    a leading batch dimension or not (a batched and an unbatched model are a
    valid pair), so the comparison is of the trailing dimensions only, up to
    the length of the shorter shape. That is the same rule
    :func:`extract_models` applies once the dimensionality is known; the
    point of doing it here is that it does not need to know it.

    Args:
        models: The model tensors, already known to be non-empty Tensors.
        names: A name for each entry in ``models``.
        propagator: The calling propagator's name, which the message begins
            with when it is known.

    Raises:
        DeepwaveValueError: If two models' spatial shapes differ.

    """
    if len(models) < 2:
        return
    first_name, first = names[0], models[0]
    for name, model in zip(names[1:], models[1:]):
        n = min(first.dim(), model.dim())
        if n == 0:
            continue
        if tuple(first.shape[-n:]) != tuple(model.shape[-n:]):
            prefix = f"{propagator}: " if propagator else ""
            raise DeepwaveValueError(
                f"{prefix}all models must have the same spatial shape, but "
                f"{first_name} has shape {tuple(first.shape)} and {name} has "
                f"shape {tuple(model.shape)}."
            )


_SUPPORTED_MODEL_DTYPES = (torch.float32, torch.float64)


def check_model_dtype(models: abc.Sequence[torch.Tensor]) -> None:
    """Rejects half-precision model tensors with an actionable error.

    Called first, before anything else in :func:`setup_propagator` derives
    the "expected" dtype from ``models[0]``. Without this guard, a
    ``float16``/``bfloat16`` model silently becomes the expected dtype for
    every other input, so the first mismatch reported is a misleading one
    against, e.g., ``source_amplitudes`` ("Expected source amplitudes to have
    datatype torch.bfloat16, but found one with dtype torch.float32") -- which
    blames the wrong tensor and gives no hint that half precision itself is
    the problem.

    Args:
        models: The model tensors (e.g. velocity, density) passed to the
            propagator.

    Raises:
        DeepwaveTypeError: If any model tensor has a dtype other than
            ``torch.float32``/``torch.float64``.

    """
    for i, model in enumerate(models):
        if model.dtype not in _SUPPORTED_MODEL_DTYPES:
            raise DeepwaveTypeError(
                f"Model tensor {i} (of {len(models)}) has dtype {model.dtype}, "
                "but Deepwave's propagators require torch.float32 or "
                "torch.float64 -- the compiled finite-difference backend "
                "does not implement half precision. Note that "
                "torch.autocast / torch.cuda.amp are not supported by "
                "Deepwave and are a no-op for it: if a model tensor ended up "
                "in float16/bfloat16 because of an enclosing autocast "
                "context, cast it back explicitly (e.g. `v = v.float()`) "
                "before calling the propagator; do not rely on autocast to "
                "restore full precision here."
            )


def _storage_preflight(
    storage: Optional[ResolvedStorage],
    propagator_name: Optional[str],
    models: abc.Sequence[torch.Tensor],
    model_names: abc.Sequence[str],
    padded_models: abc.Sequence[torch.Tensor],
    ndims: int,
    n_shots: int,
    nt: int,
    model_gradient_sampling_interval: int,
    dtype: torch.dtype,
    device: torch.device,
    q_n_mechanisms: int,
    accuracy: int,
    pml_width: abc.Sequence[int],
) -> None:
    """The run's estimate of its own footprint, made before it pays for it.

    Placed at the one moment where the answer is both accurate and still
    useful: the timing, the survey extraction and the padded shape are
    resolved, so the estimate is the run's own rather than a parametric guess,
    but the wavefield history -- the term that dominates a large run -- has not
    been allocated yet.

    Two things read it, and it is skipped entirely when neither is asking:

    * the default-disk-directory warning, whose size clause needs it, and the
      free-space warning that goes with any disk directory;
    * :func:`deepwave.plan`, which reports it.

    The estimate itself is made by the same shared core
    :func:`estimate_memory` uses, differing only in that the padded shape and
    the sampling interval are read off the resolved run instead of derived
    from arguments.

    Args:
        storage: The resolved storage policy, or None if the caller has none.
        propagator_name: The propagator's name, or None to skip.
        models: The user's models, in the propagator's own order.
        model_names: Their user-facing names, matching ``models``.
        padded_models: The models after FD and PML padding, for the exact
            padded shape.
        ndims: The number of spatial dimensions.
        n_shots: The batch size.
        nt: The number of user-level time samples.
        model_gradient_sampling_interval: The resolved sampling interval.
        dtype: The dtype of the run.
        device: The device of the run.
        q_n_mechanisms: The number of GSLS relaxation mechanisms.
        accuracy: The resolved finite-difference accuracy order.
        pml_width: The resolved per-face PML width, ``2 * ndims`` entries.

    """
    from deepwave import _plan

    probe = _plan.active_probe()
    if storage is None or propagator_name is None:
        return
    wants_disk_warnings = storage.mode == "disk"
    if probe is None and not wants_disk_warnings:
        return

    from deepwave.introspection import _resolve_spec

    spec = _resolve_spec(propagator_name)
    requiring = (
        [name for name, model in zip(model_names, models) if model.requires_grad]
        if torch.is_grad_enabled()
        else []
    )
    known = set(_propagator_user_model_names(propagator_name, ndims))
    if set(requiring) <= known:
        n_storage_slots, n_model_grads = _grad_counts_for_names(
            propagator_name, requiring, ndims
        )
    else:
        # A propagator whose model_names here are not the user-facing set the
        # tables are keyed on. Fall back to the same conservative count rather
        # than mis-indexing a table.
        n_storage_slots = n_model_grads = len(requiring)
    memory = _estimate_memory_core(
        padded_shape=tuple(int(s) for s in padded_models[0].shape[-ndims:]),
        n_shots=n_shots,
        num_steps_stored=nt // model_gradient_sampling_interval,
        dtype=dtype,
        device=device,
        n_forward_wavefields=_spec_n_wavefields(spec, ndims, q_n_mechanisms),
        n_storage_slots=n_storage_slots,
        n_model_grads=n_model_grads,
        storage_mode=_storage_mode_enum(storage.mode),
        storage_compression=storage.lossy_compression,
        n_segments=None if probe is None else probe.n_segments,
        auxiliary_state_elements=_auxiliary_state_elements(
            spec,
            ndims,
            q_n_mechanisms,
            tuple(int(s) for s in padded_models[0].shape[-ndims:]),
            accuracy,
            pml_width,
        ),
    )
    _plan.record_memory(memory)
    history_bytes = memory["backward_storage_bytes"]
    if wants_disk_warnings:
        # Frame chain: user call -> propagator -> setup_propagator -> here ->
        # the warning helper -> _warn_once_per_cause (which adds one of its
        # own), so 5 attributes the warning to the user's own call.
        warn_if_default_disk_path(storage, history_bytes, stacklevel=5)
        warn_if_storage_path_too_small(storage.path, history_bytes, stacklevel=5)


def setup_propagator(
    models: abc.Sequence[torch.Tensor],
    model_pad_modes: abc.Sequence[str],
    model_names: abc.Sequence[str],
    grid_spacing: Union[float, abc.Sequence[float]],
    dt: float,
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    source_locations: abc.Sequence[Optional[torch.Tensor]],
    receiver_locations: abc.Sequence[Optional[torch.Tensor]],
    accuracy: int,
    fd_pad: abc.Sequence[int],
    pml_width: Union[int, abc.Sequence[int]],
    pml_freq: Optional[float],
    max_vel: Optional[float],
    min_nonzero_model_vel: float,
    max_model_vel: float,
    survey_pad: Optional[Union[int, abc.Sequence[Optional[int]]]],
    wavefields: abc.Sequence[Optional[torch.Tensor]],
    origin: Optional[abc.Sequence[int]],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    ndims: int,
    courant_number: Optional[float] = None,
    staggered: bool = False,
    storage: Optional[ResolvedStorage] = None,
    propagator_name: Optional[str] = None,
    q_n_mechanisms: int = 1,
    pml_inference_amplitudes: Optional[abc.Sequence[Optional[torch.Tensor]]] = None,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[float],
    float,
    int,
    int,
    int,
    int,
    Accuracy,
    list[int],
    float,
    float,
    float,
    float,
    bool,
    torch.device,
    torch.dtype,
    float,
    list[int],
    list[int],
]:
    """Performs common setup for all propagators.

    This includes input validation, calculation of the internal time step,
    PML setup, and preparation of source and receiver tensors.

    Args:
        models:
            A sequence of PyTorch Tensors representing the model
            parameters (e.g., velocity, density).
        model_pad_modes:
            A sequence of strings specifying the padding mode
            for each model
        model_names:
            A sequence of names, one per entry in `models`, used to identify
            the offending model in error messages (e.g. ``"v"``, ``"rho"``).
        grid_spacing:
            The spacing between grid points in each dimension.
            Can be a single float (for isotropic spacing) or a sequence of
            floats (for anisotropic spacing).
        dt:
            The time step interval.
        source_amplitudes:
            A sequence of optional Tensors, where each torch.Tensor
            contains the amplitudes of the sources for a source type.
        source_locations:
            A sequence of optional Tensors, where each torch.Tensor
            contains the locations of the sources for a source type.
        receiver_locations:
            A sequence of optional Tensors, where each torch.Tensor
            contains the locations of the receivers for receiver type.
        accuracy:
            The finite-difference accuracy order (e.g., 2, 4, 6, 8).
        fd_pad:
            A sequence of integers specifying the padding for finite
            difference stencils in each dimension.
        pml_width:
            The width of the PML (Perfectly Matched Layer) in grid
            cells. Can be a single integer for all edges or an
            integer for each edge.
        pml_freq:
            The dominant frequency for PML absorption. If None, it is
            inferred from the peak of `source_amplitudes`' spectrum, or, if
            there are no source amplitudes, from `dt` (see
            :func:`set_pml_freq`).
        max_vel:
            The maximum velocity to use for CFL condition
            calculation. If None, inferred from `max_model_vel`.
        min_nonzero_model_vel:
            The minimum non-zero velocity in the model.
            Used for warning about points per wavelength.
        max_model_vel:
            The maximum velocity in the model. Used for CFL
            condition calculation.
        survey_pad:
            Optional padding around the survey area. Can be an optional
            int for all edges or an optional int for each edge.
        wavefields:
            A sequence of optional Tensors representing initial
            wavefields.
        origin:
            Optional sequence of integers specifying the origin of the
            survey within the full model.
        nt:
            Optional integer specifying the number of time steps. If None,
            inferred from `source_amplitudes`.
        model_gradient_sampling_interval:
            The interval at which to sample
            the wavefield for model gradient calculation.
        freq_taper_frac:
            The fraction of the frequency spectrum to taper.
        time_pad_frac:
            The fraction of the time axis to pad with zeros.
        time_taper:
            Whether to apply a Hann window in time.
        ndims:
            The number of spatial dimensions of the model.
        courant_number:
            Optional Courant number for the CFL time-step calculation. If
            None (default), the conservative default (0.6) is used. If given,
            it is validated against the critical Courant number for this
            accuracy order and scheme.
        staggered:
            Whether the propagator uses a staggered grid (acoustic, elastic).
            Selects the correct critical Courant number table. Default False.
        storage:
            The call's resolved storage policy, if it has one. Given it, the
            setup makes the run's own estimate of its footprint once the
            timing and the padded shape are known but before the wavefield
            history is allocated, which fills in the size clause of the
            default-disk-directory warning and answers
            :func:`deepwave.plan`. The estimate is skipped when neither
            needs it.
        propagator_name:
            The propagator's name, needed to make that estimate. Without it
            the estimate is skipped.
        q_n_mechanisms:
            The number of GSLS relaxation mechanisms, which sets the wavefield
            count for viscoacoustic. Ignored by the other families.
        pml_inference_amplitudes:
            Active caller-rate source records before spatial interpolation.
            Used only for automatic PML-frequency inference.

    Returns:
        Tuple:

            - models_out: Processed model Tensors.
            - source_amplitudes_out: Processed source amplitude Tensors.
            - wavefields_out: Processed wavefield Tensors.
            - source_locations_out: Processed source location Tensors.
            - receiver_locations_out: Processed receiver location Tensors.
            - grid_spacing: Processed grid spacing.
            - dt: Internal time step.
            - nt: Number of time steps.
            - n_batch: Batch size.
            - step_ratio: Ratio between user dt and internal dt.
            - model_gradient_sampling_interval: Model gradient sampling interval.
            - accuracy: Finite difference accuracy.
            - pml_width: PML width.
            - pml_freq: PML frequency.
            - max_vel: Maximum velocity.
            - freq_taper_frac: Frequency taper fraction.
            - time_pad_frac: Time padding fraction.
            - time_taper: Time taper flag.
            - device: PyTorch device.
            - dtype: PyTorch data type.
            - resolved_courant_number: The Courant number actually targeted
              (``courant_number`` if given, else the conservative default).
            - survey_origin: The resolved survey origin (one entry per
              dimension) within the full model.
            - survey_extent: The extracted survey region's shape (one entry
              per dimension), before FD/PML padding is added.

    Raises:
        TypeError: If any input is of an incorrect type.
        ValueError: If any input has an invalid value.
        RuntimeError: If there are inconsistencies between inputs.

    """
    check_models(models, model_names, propagator_name)
    check_model_dtype(models)
    # Skipped for a model whose values cannot be read -- a vmap batched model,
    # or one holding no data at all (the same restriction velocity_extremes and
    # set_max_vel document, asked the same way, through values_readable) -- and
    # while torch.compiler.is_compiling(), where the .item() this needs is a
    # data-dependent read that graph-breaks/guards -- consistent with every
    # other merely-diagnostic (non-memory-safety-critical) input validation
    # read in this module (see _validate_location_tensor). A non-finite model
    # not caught here still surfaces as non-finite output.
    if not _torch_is_compiling():
        for name, model in zip(model_names, models):
            if name == "q" and propagator_name == "viscoacoustic":
                # check_models has validated q > 0, including its +inf
                # lossless limit; the generic finite-only rule does not apply.
                continue
            if not values_readable(model):
                continue
            n_nonfinite = int((~torch.isfinite(model)).sum())
            if n_nonfinite:
                raise DeepwaveValueError(
                    f"{name} contains {n_nonfinite} non-finite "
                    f"value{'s' if n_nonfinite != 1 else ''} (NaN or Inf) out of "
                    f"{model.numel()} elements."
                )
    device = models[0].device
    dtype = models[0].dtype
    # Checked before get_n_batch (which would otherwise blame a vaguer
    # "at least one of source_locations or wavefield" on a call that, e.g.,
    # supplied source_amplitudes but forgot the matching source_locations).
    check_source_amplitudes_locations_match(
        source_amplitudes, source_locations, propagator_name, ndims
    )
    n_batch = get_n_batch(source_locations, wavefields)
    grid_spacing = set_grid_spacing(grid_spacing, ndims, propagator_name)
    accuracy = set_accuracy(accuracy)
    for axis, size in enumerate(models[0].shape[-ndims:]):
        if size < accuracy // 2:
            raise DeepwaveValueError(
                f"{propagator_name or 'propagator'}: model axis {axis} has "
                f"{size} cells; accuracy={accuracy} requires at least "
                f"{accuracy // 2} cells per spatial axis."
            )
    if courant_number is not None:
        max_courant_number = _critical_courant_number(accuracy, staggered)
        if courant_number > max_courant_number:
            raise DeepwaveValueError(
                f"courant_number {courant_number} exceeds the critical Courant "
                f"number {max_courant_number} for this accuracy order and "
                "scheme; the propagation would be unstable. Choose a "
                "courant_number at or below the critical value (see "
                "docs/discretisation.rst)."
            )
    if propagator_name in ("acoustic", "acoustic_born") and all(
        values_readable(model) for model in models[:2]
    ):
        from deepwave._staggered_grid import pressure_stability_speed

        max_model_vel = pressure_stability_speed(
            models[0], models[1], accuracy, grid_spacing, courant_number
        )
    elif propagator_name in ("elastic", "elastic_born") and all(
        values_readable(model) for model in models[:3]
    ):
        from deepwave._staggered_grid import elastic_stability_speed

        lamb, mu, buoyancy = [model.detach() for model in models[:3]]
        stiffness_bound = torch.maximum(
            lamb + 2 * mu + (ndims - 1) * lamb.abs(), 2 * mu
        )
        max_model_vel = elastic_stability_speed(
            stiffness_bound,
            buoyancy,
            accuracy,
            max_model_vel,
            courant_number,
            models[:3],
        )
    elif propagator_name in ("elastic_vti", "acoustic_vti") and all(
        values_readable(model) for model in models
    ):
        from deepwave._staggered_grid import elastic_stability_speed

        # The model order here is the public physical order; VTI constitutive
        # and staggered coefficient preparation happens after shared setup.
        c11, c13, c33 = [model.detach() for model in models[:3]]
        if propagator_name == "elastic_vti":
            c55, buoyancy = models[3].detach(), models[-1].detach()
            c66 = models[4].detach() if ndims == 3 else c55
            horizontal = c11 + (ndims - 2) * (c11 - 2 * c66).abs() + c13.abs()
            stiffness_bound = torch.maximum(horizontal, c33 + (ndims - 1) * c13.abs())
            stiffness_bound = torch.maximum(
                stiffness_bound, 2 * torch.maximum(c55, c66)
            )
        else:
            buoyancy = models[3].detach()
            stiffness_bound = torch.maximum(
                (ndims - 1) * c11 + c13.abs(), c33 + (ndims - 1) * c13.abs()
            )
        max_model_vel = elastic_stability_speed(
            stiffness_bound,
            buoyancy,
            accuracy,
            max_model_vel,
            courant_number,
            models,
        )
    max_vel = set_max_vel(max_vel, max_model_vel)
    # ``dt`` is about to become the internal (sub-stepped) time step, but the
    # source amplitudes are still at the user's sample interval, which is what
    # pml_freq inference needs.
    user_dt = dt
    dt, step_ratio = cfl_condition_n(
        grid_spacing,
        dt,
        max_vel,
        courant_number=courant_number,
    )
    resolved_courant_number = (
        _DEFAULT_COURANT_NUMBER if courant_number is None else courant_number
    )
    pml_width = set_pml_width(pml_width, ndims)
    internal_nt = set_nt(nt, source_amplitudes, step_ratio)
    if nt is not None:
        # ``nt`` is the run's length, so the amplitudes are fitted to it here,
        # in the caller's own time samples and before ``pml_freq`` inference
        # or any internal resampling reads them.
        source_amplitudes = fit_source_amplitudes_to_nt(source_amplitudes, nt)
        if pml_inference_amplitudes is not None:
            pml_inference_amplitudes = fit_source_amplitudes_to_nt(
                pml_inference_amplitudes, nt
            )
    nt = internal_nt
    model_gradient_sampling_interval = set_model_gradient_sampling_interval(
        model_gradient_sampling_interval,
        # ``nt`` is the *internal* step count by this point (``set_nt``
        # multiplies the user's by the CFL ``step_ratio``); the sampling
        # interval divides the user-facing count.
        user_nt=nt // step_ratio,
        # ``torch.is_grad_enabled()`` as well as ``requires_grad``: inside
        # ``torch.no_grad()`` (an inference run whose model happens to be an
        # ``nn.Parameter``) no model gradient will be computed, so there is
        # nothing for the interval to sample and no reason to refuse the run.
        # This matches what the ``storage_mode="none"`` guard in
        # ``_engine.run_forward_engine`` sees: it inspects the *internal*
        # models, which autograd has already detached under ``no_grad``.
        model_requires_grad=torch.is_grad_enabled()
        and any(model.requires_grad for model in models),
    )
    freq_taper_frac = set_freq_taper_frac(freq_taper_frac)
    time_pad_frac = set_time_pad_frac(time_pad_frac)
    source_amplitudes_out = set_source_amplitudes(
        source_amplitudes,
        n_batch,
        nt,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    )
    # After ``set_source_amplitudes``, which is where the source amplitudes are
    # validated (dtype, device, shape). Inferring first would mean a float16 or
    # complex or wrong-device amplitude tensor surfaced as a raw torch error
    # from inside the inference FFT instead of Deepwave's own message. The
    # inference still reads the *user's* amplitudes, at the user's sample
    # interval -- not ``source_amplitudes_out``, which has been resampled to
    # the internal time step.
    pml_freq_inferred = pml_freq is None or isinstance(pml_freq, _InferredPmlFreq)
    pml_freq = set_pml_freq(
        pml_freq,
        dt,
        (
            source_amplitudes
            if pml_inference_amplitudes is None
            else pml_inference_amplitudes
        ),
        user_dt,
        propagator_name,
    )
    check_points_per_wavelength(
        min_nonzero_model_vel, pml_freq, grid_spacing, inferred=pml_freq_inferred
    )
    (
        models_out,
        source_locations_out,
        receiver_locations_out,
        wavefields_out,
        survey_extents,
    ) = extract_survey(
        models,
        source_locations,
        receiver_locations,
        wavefields,
        survey_pad,
        origin,
        fd_pad,
        pml_width,
        model_pad_modes,
        n_batch,
        ndims,
        device,
        dtype,
        propagator_name,
    )
    survey_origin = [extent[0] for extent in survey_extents]
    survey_extent = [extent[1] - extent[0] for extent in survey_extents]
    _storage_preflight(
        storage,
        propagator_name,
        models,
        model_names,
        models_out,
        ndims,
        n_batch,
        nt // step_ratio,
        model_gradient_sampling_interval,
        dtype,
        device,
        q_n_mechanisms,
        accuracy,
        pml_width,
    )
    return (
        models_out,
        source_amplitudes_out,
        wavefields_out,
        source_locations_out,
        receiver_locations_out,
        grid_spacing,
        dt,
        nt,
        n_batch,
        step_ratio,
        model_gradient_sampling_interval,
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
        resolved_courant_number,
        survey_origin,
        survey_extent,
    )


def get_n_batch(
    source_locations: abc.Sequence[Optional[torch.Tensor]],
    wavefields: abc.Sequence[Optional[torch.Tensor]],
) -> int:
    """Get the batch size from source_locations or wavefields.

    Args:
        source_locations: Sequence of source location tensors (or None).
        wavefields: Sequence of wavefield tensors (or None).

    Returns:
        Batch size (first dimension of any non-None tensor).

    Raises:
        RuntimeError: If all tensors are None.

    """
    tensors = list(source_locations) + list(wavefields)
    for tensor in tensors:
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise DeepwaveTypeError(
                    f"Expected a torch.Tensor, but got {type(tensor).__name__}",
                )
            return tensor.shape[0]
    raise DeepwaveRuntimeError(
        "At least one input source_locations or wavefield must be non-None.",
    )


def downsample_and_movedim(
    receiver_amplitudes: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> torch.Tensor:
    """Downsample receiver data and move the time dimension to the last axis.

    This is a convenience function that combines the downsampling of
    receiver data (if `step_ratio` > 1) with moving the time dimension
    to be the last dimension, which is the format expected by the user.

    Args:
        receiver_amplitudes: A torch.Tensor containing the receiver amplitudes.
        step_ratio: The integer factor by which to downsample.
        freq_taper_frac: The fraction of the end of the frequency spectrum
            to taper.
        time_pad_frac: The fraction of the time axis to pad with zeros.
        time_taper: Whether to apply a Hann window in time.
        shift: The amount to shift the data in time before downsampling.

    Returns:
        The processed receiver amplitudes.

    """
    receiver_amplitudes = torch.movedim(receiver_amplitudes, 0, -1)
    if receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            shift=shift,
        )
    return receiver_amplitudes


def reject_model_as_grid_spacing(
    grid_spacing: Union[float, abc.Sequence[float]],
    ndims: int,
    propagator_name: Optional[str],
) -> None:
    """Raises if a model Tensor was passed where the grid spacing goes.

    A ``grid_spacing`` Tensor is legitimate -- a 0-dim Tensor, or one holding
    one value per dimension -- so the test is on the element count, not on
    the type: a Tensor with more entries than there are spatial dimensions
    cannot be a spacing and is, in practice, always a model.

    This exists for one specific v0.0.27 call. ``scalar_born``'s second
    positional argument used to be the scattering model and is now
    ``grid_spacing``, so ``scalar_born(v, scatter, dx, dt=dt)`` binds the
    model to ``grid_spacing`` and would otherwise fail several layers down
    with a message about floats. It is applied to every propagator because a
    model in the spacing slot is never anything but a mistake, and the check
    is one ``isinstance`` on a path that runs once per call.
    """
    if not isinstance(grid_spacing, torch.Tensor):
        return
    if grid_spacing.numel() <= ndims:
        return
    if propagator_name == "scalar_born":
        fix = (
            "In Deepwave 0.0.27 scalar_born's second positional argument was "
            "the scattering model; it is now grid_spacing, and the scattering "
            "model is the keyword argument dv (formerly scatter). Fix: "
            "scalar_born(v, grid_spacing, dt, dv=dv, ...)."
        )
    else:
        name = propagator_name or "this propagator"
        fix = (
            f"Check the order of {name}'s positional arguments: every "
            "argument after dt is keyword-only in Deepwave 0.1.0, so a model "
            "must be named."
        )
    raise DeepwaveTypeError(
        "grid_spacing must be a float or a sequence of floats, but a "
        f"torch.Tensor of shape {tuple(grid_spacing.shape)} was passed. A "
        f"grid spacing for a {ndims}-dimensional model is one value, or one "
        f"per dimension ({ndims}); a Tensor with more entries than that is a "
        f"model, not a spacing. {fix} See the Upgrading page of the "
        "documentation."
    )


def _float_value(value: Any) -> float:
    """Let float validate its complete runtime numeric and buffer protocols.

    Any is confined to this conversion boundary: restricting the accepted
    protocols with isinstance would reject values float itself accepts.
    The result is always a float, or the built-in conversion raises.
    """
    return float(value)


def _int_value(value: Any) -> int:
    """Delegate validation of integer-compatible values to the built-in."""
    return int(value)


def _iterable_values(value: Any) -> abc.Iterable[object]:
    """Validate iteration, including the legacy __getitem__ sequence protocol."""
    return cast("abc.Iterable[object]", iter(value))


def set_grid_spacing(
    grid_spacing: Union[float, abc.Sequence[float]],
    ndims: int,
    propagator_name: Optional[str] = None,
) -> list[float]:
    """Ensures grid_spacing is a sequence of length ndims.

    Args:
        grid_spacing: The spacing between grid points in each dimension.
            Can be a single float (for isotropic spacing) or a float for
            each dimension (for anisotropic spacing).
        ndims: The number of spatial dimensions.
        propagator_name: The name of the calling propagator, if there is
            one. Used only to make the "a model was passed as the grid
            spacing" message name the call the caller actually wrote.

    Returns:
        A list of floats representing the grid spacing for each dimension.

    Raises:
        DeepwaveTypeError: If `grid_spacing` is a model-sized Tensor.
        TypeError: If `grid_spacing` is not a float or sequence of floats.
        ValueError: If any element of `grid_spacing` is not positive.
        RuntimeError: If the length of `grid_spacing` is not 1 or `ndims`.

    """
    reject_model_as_grid_spacing(grid_spacing, ndims, propagator_name)
    # Validate input and convert to list even for edge-cases such as 0-dim Tensors
    try:
        # Check if convertible to a float
        processed_grid_spacing = [_float_value(grid_spacing)] * ndims
    except (TypeError, ValueError):
        # Check if a sequence of values convertible to floats
        try:
            processed_grid_spacing = [
                _float_value(spacing) for spacing in _iterable_values(grid_spacing)
            ]
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError(
                "grid_spacing must be a float or sequence of floats."
            ) from e

    if any(spacing <= 0 for spacing in processed_grid_spacing):
        raise DeepwaveValueError("grid_spacing elements must be positive.")

    if len(processed_grid_spacing) != ndims:
        raise DeepwaveRuntimeError(
            f"grid_spacing must have {ndims} elements, "
            f"got {len(processed_grid_spacing)}.",
        )
    return processed_grid_spacing


def set_accuracy(accuracy: int) -> Accuracy:
    """Validates finite difference accuracy.

    Args:
        accuracy: The finite-difference accuracy order (e.g., 2, 4, 6, 8).

    Returns:
        The validated accuracy.

    Raises:
        TypeError: If `accuracy` is not an integer.
        ValueError: If `accuracy` is not one of 2, 4, 6, or 8.

    """
    if not isinstance(accuracy, int):
        raise DeepwaveTypeError("accuracy must be an int.")
    if accuracy not in (2, 4, 6, 8):
        raise DeepwaveValueError(f"accuracy must be 2, 4, 6, or 8, got {accuracy}")
    # Narrowed by the check above; ``Accuracy`` is the public spelling of
    # exactly this closed set.
    return cast("Accuracy", accuracy)


def set_pml_width(pml_width: Union[int, abc.Sequence[int]], ndims: int) -> list[int]:
    """Ensures pml_width is a sequence of length 2 * ndims.

    Args:
        pml_width: The width of the PML (Perfectly Matched Layer) in grid
            cells. Can be a single integer or a sequence of integers for
            each side of each dimension.
        ndims: The number of spatial dimensions.

    Returns:
        A list of integers representing the PML width for each side of each
        dimension.

    Raises:
        TypeError: If `pml_width` is not an int or sequence of ints.
        ValueError: If any element of `pml_width` is negative.
        RuntimeError: If the length of `pml_width` is not 1 or `2 * ndims`.

    """
    # Validate input and convert to list even for edge-cases such as 0-dim Tensors
    try:
        # Check if convertible to an int
        processed_pml_width = [_int_value(pml_width)] * 2 * ndims
        if _float_value(pml_width) != float(_int_value(pml_width)):
            raise DeepwaveTypeError("pml_width must be an int or sequence of ints.")
    except (TypeError, ValueError):
        # Check if a sequence of values convertible to ints
        try:
            pml_width_list = list(_iterable_values(pml_width))
            if any(
                _float_value(width) != float(_int_value(width))
                for width in pml_width_list
            ):
                raise DeepwaveTypeError("pml_width must be an int or sequence of ints.")
            processed_pml_width = [_int_value(width) for width in pml_width_list]
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError(
                "pml_width must be an int or sequence of ints."
            ) from e

    if any(width < 0 for width in processed_pml_width):
        raise DeepwaveValueError("pml_width must be non-negative.")

    if len(processed_pml_width) != 2 * ndims:
        raise DeepwaveRuntimeError(
            f"Expected pml_width to be of length 1 or {2 * ndims}, "
            f"got {len(processed_pml_width)}.",
        )
    return processed_pml_width


# Integer codes for the per-face boundary type, shared across the Python and
# C/CUDA backends (kept in sync with the ``bc`` argument the kernels receive).
BC_PML = 0
BC_FREE = 1
BC_RIGID = 2
BC_PERIODIC = 3
_BOUNDARY_NAME_TO_CODE = {
    "pml": BC_PML,
    "free": BC_FREE,
    "rigid": BC_RIGID,
    "periodic": BC_PERIODIC,
}


def normalize_boundary(
    boundary: Union[str, abc.Sequence[str]],
    ndims: int,
) -> list[str]:
    """Broadcasts a single ``boundary`` name to every face.

    A bare string is one condition for the whole domain, not a sequence of
    one-character faces. ``list("pml")`` is ``['p', 'm', 'l']``, so without
    this a single string is iterated character by character and the call
    fails with a message about a face named ``'p'`` (or, for ``"pml"`` in
    2-D, about a length of 3) -- neither of which mentions the thing the
    user actually wrote. ``Sequence[str]`` admits ``str``, so no type
    checker catches it either. The sibling ``pml_width`` argument has always
    broadcast a scalar to every face; this makes ``boundary`` agree.

    Applied in both places that read the user's ``boundary`` argument
    (:func:`set_boundary` and :func:`warn_bare_zero_width_face`), so that
    the spelled-out ``boundary="pml"`` and an omitted ``boundary`` are the
    same call on all nine propagators, warnings included.

    Args:
        boundary: A single condition name, or a per-face sequence.
        ndims: The number of spatial dimensions.

    Returns:
        A single name repeated ``2 * ndims`` times, or any other sequence as
        a list. The names themselves are not validated here;
        :func:`set_boundary` does that.

    """
    if isinstance(boundary, str):
        return [boundary] * 2 * ndims
    return list(boundary)


def set_boundary(
    boundary: Optional[Union[str, abc.Sequence[str]]],
    pml_width: abc.Sequence[int],
    ndims: int,
) -> list[int]:
    """Validate the ``boundary`` argument and return per-face integer codes.

    Args:
        boundary: ``None`` (all faces absorbing/PML, the default and
            bit-identical to previous behaviour), a single string applying
            to every face, or a sequence of ``2 * ndims``
            strings from ``{"pml", "free", "rigid", "periodic"}``, in the same
            face order as ``pml_width`` (begin dim 0, end dim 0, begin dim 1,
            ...).
        pml_width: The validated per-face PML widths (length ``2 * ndims``).
        ndims: The number of spatial dimensions.

    Returns:
        A list of ``2 * ndims`` integer face codes (see ``BC_*``). ``None``
        maps to all-``BC_PML``.

    Raises:
        TypeError: If ``boundary`` is not a string or a sequence of strings.
        ValueError: If a name is unrecognised, a ``free``/``rigid``/``periodic``
            face does not have ``pml_width == 0``, or ``periodic`` is not set on
            both faces of a dimension.
        RuntimeError: If the length is not ``2 * ndims``.
    """
    if boundary is None:
        return [BC_PML] * 2 * ndims
    try:
        names = normalize_boundary(boundary, ndims)
    except TypeError as e:
        raise DeepwaveTypeError(
            "boundary must be None, a string, or a sequence of strings.",
        ) from e
    if len(names) != 2 * ndims:
        raise DeepwaveRuntimeError(
            f"Expected boundary to be of length {2 * ndims}, got {len(names)}.",
        )
    codes: list[int] = []
    for name in names:
        if not isinstance(name, str) or name not in _BOUNDARY_NAME_TO_CODE:
            raise DeepwaveValueError(
                "Each boundary entry must be one of "
                f"{sorted(_BOUNDARY_NAME_TO_CODE)}, got {name!r}.",
            )
        codes.append(_BOUNDARY_NAME_TO_CODE[name])
    for face, code in enumerate(codes):
        if code != BC_PML and pml_width[face] != 0:
            raise DeepwaveValueError(
                f"Face {face} has boundary {names[face]!r}, which requires "
                f"pml_width == 0 on that face, but got pml_width "
                f"{pml_width[face]}.",
            )
    for dim in range(ndims):
        lo, hi = codes[2 * dim], codes[2 * dim + 1]
        if (lo == BC_PERIODIC) != (hi == BC_PERIODIC):
            raise DeepwaveValueError(
                f"'periodic' must be set on both faces of dimension {dim}; "
                f"got {names[2 * dim]!r} and {names[2 * dim + 1]!r}.",
            )
    return codes


#: The propagators whose source/receiver component sits on the regular
#: (nodal) grid *and* whose zero-width ``free``/``rigid`` faces put their
#: mirror planes exactly where :class:`deepwave.Hicks` puts its own by
#: default: odd about the node one cell outside the edge for ``free``, even
#: about the half-cell plane for ``rigid`` (compare
#: :func:`deepwave._boundaries._fill_boundary_face` with the
#: ``free_surface_locs`` default in :class:`deepwave.Hicks`). For these, and
#: only these, an interpolation window that crosses such a face can be folded
#: back automatically and give exactly what a hand-built ``Hicks`` with those
#: surfaces gives. Every other propagator carries its source/receiver
#: components on staggered nodes, where the image parity is component
#: dependent (Robertsson, 1996) and is not the one these defaults apply.
_HICKS_IMAGE_PROPAGATORS = frozenset({"scalar", "scalar_born"})


class HicksGrid(NamedTuple):
    """What an automatically built ``Hicks`` needs to know about the grid.

    Built once per propagator call by :func:`hicks_grid` and handed to the
    interpolator resolution, which uses it for three things: to fold an
    interpolation window back off a face whose image it can reproduce, to
    refuse (rather than silently truncate) a window that crosses a face whose
    image it cannot, and to report an out-of-bounds location in the
    coordinates the caller wrote rather than in the expanded window indices
    the backend ends up with.

    Attributes:
        model_shape: The spatial shape of the model, in cells.
        free: Per face, whether to fold the window as a free surface.
        rigid: Per face, whether to fold the window as a rigid surface.
        blocked: ``(face, boundary name, reason)`` for each face that is
            neither absorbing nor foldable, so a window crossing it has to be
            refused; the reason is a clause naming why, for the message.
        propagator: The calling propagator's name, for messages.

    """

    model_shape: tuple[int, ...]
    free: tuple[bool, ...]
    rigid: tuple[bool, ...]
    blocked: tuple[tuple[int, str, str], ...]
    propagator: str


def face_description(face: int) -> str:
    """``"the beginning of dimension 1"`` for face index ``face``."""
    end = "end" if face % 2 else "beginning"
    return f"the {end} of dimension {face // 2}"


def _locations_ndim(locations: abc.Sequence[_LocationArg]) -> Optional[int]:
    """The number of spatial dimensions the location arguments carry."""
    for location in locations:
        if isinstance(location, torch.Tensor):
            if location.ndim == 3:
                return int(location.shape[2])
        elif location is not None:
            # A Hicks instance, whose own locations carry the dimension.
            own = getattr(location, "locations", None)
            if isinstance(own, torch.Tensor) and own.ndim == 3:
                return int(own.shape[2])
    return None


def hicks_grid(
    boundary: Optional[Union[str, abc.Sequence[str]]],
    locations: abc.Sequence[_LocationArg],
    model: torch.Tensor,
    propagator: str,
    *,
    whole_model: bool = False,
) -> Optional[HicksGrid]:
    """Resolve what interpolation must do at each face of the model.

    ``boundary`` is read leniently here, because this runs before
    :func:`set_boundary` validates it: anything malformed (a wrong length, an
    unrecognised name) yields ``None`` for the *faces*, leaving that function
    to raise the one authoritative error for it a few lines later. The model
    shape is still reported, because an out-of-bounds coordinate is worth
    naming whatever the boundary says.

    Args:
        boundary: The caller's ``boundary`` argument, unvalidated.
        locations: Every source and receiver location argument of the call,
            used only to read the number of spatial dimensions.
        model: Any of the call's model Tensors, for the spatial shape.
        propagator: The calling propagator's name.
        whole_model: Whether the region that will be propagated is the whole
            model, so that the faces named by ``boundary`` really are the
            faces of ``model``. A call that extracts a sub-region
            (``survey_pad``, ``origin``, or an initial wavefield that sets
            the extents) puts them somewhere this function cannot see, so it
            folds nothing and refuses instead.

    Returns:
        The resolved grid, or ``None`` if the number of spatial dimensions
        cannot be read from ``locations`` (no locations were given, so
        nothing will be interpolated).

    """
    ndim = _locations_ndim(locations)
    if ndim is None or ndim > model.ndim:
        return None
    model_shape = tuple(int(size) for size in model.shape[-ndim:])
    free = [False] * (2 * ndim)
    rigid = [False] * (2 * ndim)
    blocked: list[tuple[int, str, str]] = []
    if boundary is not None:
        names = normalize_boundary(boundary, ndim)
        if len(names) == 2 * ndim and all(
            isinstance(name, str) and name in _BOUNDARY_NAME_TO_CODE for name in names
        ):
            for face, name in enumerate(names):
                code = _BOUNDARY_NAME_TO_CODE[name]
                if code == BC_PML:
                    continue
                if code == BC_PERIODIC:
                    blocked.append(
                        (face, name, "Hicks interpolation has no periodic wrap")
                    )
                elif propagator not in _HICKS_IMAGE_PROPAGATORS:
                    blocked.append(
                        (
                            face,
                            name,
                            f"the image of a {name!r} face for {propagator}'s "
                            "source and receiver components is not implemented",
                        )
                    )
                elif not whole_model:
                    blocked.append(
                        (
                            face,
                            name,
                            "this call propagates a sub-region of the model "
                            "(survey_pad, origin, or an initial wavefield sets "
                            "its extents), so that face is not at the edge of "
                            "the model these coordinates are measured against",
                        )
                    )
                elif code == BC_FREE:
                    free[face] = True
                else:
                    rigid[face] = True
    return HicksGrid(
        model_shape=model_shape,
        free=tuple(free),
        rigid=tuple(rigid),
        blocked=tuple(blocked),
        propagator=propagator,
    )


#: Which ``boundary`` face codes each propagator does not accept in 0.1.0, as
#: ``{propagator: {face name: extent}}``. ``"all"`` marks a face the
#: propagator does not accept anywhere at any accuracy; ``"high"`` marks one
#: it accepts at ``accuracy=2`` everywhere, and at ``accuracy`` 4 or more only
#: on the *low* face of an axis (the beginning of a dimension).
#:
#: A ``"free"``, ``"rigid"`` or ``"periodic"`` face is implemented by refilling
#: the guard ring outside the domain with an image of the interior, which
#: replaces the interior stencil rows next to that face. Where the replacement
#: rows are not the transpose of their partners the scheme has no conserved
#: discrete energy, and a closed run grows exponentially -- measured at
#: between 1.000002 and 1.006 per time step, depending on the face, the
#: accuracy and the medium, which is invisible over a few hundred steps and
#: overwhelming over several thousand. Every entry below is one such face.
#: ``benchmarks/boundary_energy.py`` measures the property directly on the
#: kernels and reproduces the table; it is also how a face is to be cleared
#: for re-enabling, which is an additive change.
_UNAVAILABLE_BOUNDARY_FACES: dict[str, dict[str, str]] = {
    "scalar": {},
    "scalar_born": {},
    "acoustic": {"free": "high", "periodic": "all"},
    "acoustic_born": {"free": "high", "periodic": "all"},
    "viscoacoustic": {"free": "high", "periodic": "all"},
    "acoustic_vti": {"free": "all", "periodic": "all"},
    "elastic": {"free": "all", "rigid": "high", "periodic": "all"},
    "elastic_vti": {"free": "all", "rigid": "high", "periodic": "all"},
    "elastic_born": {"free": "all", "rigid": "high", "periodic": "all"},
}

#: The ``boundary`` name of each ``BC_*`` code, for messages.
_BOUNDARY_CODE_TO_NAME = {code: name for name, code in _BOUNDARY_NAME_TO_CODE.items()}

#: Set while :func:`unchecked_boundary_faces` is active, which switches
#: :func:`check_boundary_supported` off. Private and undocumented: it exists
#: so that the instrument that measures the faces
#: (``benchmarks/boundary_energy.py``) can still reach the fills that are not
#: offered, and for nothing else.
_boundary_faces_unchecked = False


@contextlib.contextmanager
def unchecked_boundary_faces() -> abc.Iterator[None]:
    """Runs :func:`check_boundary_supported` as a no-op inside the block.

    Not part of the public interface and not documented for users: the faces
    it re-opens are the ones measured not to conserve energy, so anything run
    inside it can grow without bound. It exists for
    ``benchmarks/boundary_energy.py``, which has to build the one-step
    operator of a face in order to decide whether that face could be offered.
    """
    global _boundary_faces_unchecked
    previous = _boundary_faces_unchecked
    _boundary_faces_unchecked = True
    try:
        yield
    finally:
        _boundary_faces_unchecked = previous


def check_boundary_supported(
    boundary_codes: abc.Sequence[int],
    accuracy: int,
    ndims: int,
    propagator: str,
) -> None:
    """Rejects a ``boundary`` face this propagator does not offer.

    Called by every propagator immediately after :func:`set_boundary`, so the
    refusal happens while the arguments are being validated, before anything
    is computed, on every backend and through the Module form alike.

    The faces refused are listed in :data:`_UNAVAILABLE_BOUNDARY_FACES`, whose
    comment says what they have in common: the guard-ring fill that implements
    them does not conserve a discrete energy, so the solution can grow without
    bound. Offering a face that has no bound is worse than not offering it,
    and re-offering one later is an additive change, so they are refused
    rather than documented as unreliable.

    Args:
        boundary_codes: The per-face ``BC_*`` codes from :func:`set_boundary`,
            in its interleaved face order (``[d0_lo, d0_hi, d1_lo, ...]``).
        accuracy: The spatial accuracy order of the call.
        ndims: The number of spatial dimensions.
        propagator: The propagator's name, for the message.

    Raises:
        DeepwaveNotImplementedError: If a face is not offered. The message
            names the face, says why it is not offered, and gives the
            alternative.

    """
    if _boundary_faces_unchecked:
        return
    unavailable = _UNAVAILABLE_BOUNDARY_FACES[propagator]
    for face, code in enumerate(boundary_codes):
        name = _BOUNDARY_CODE_TO_NAME[code]
        extent = unavailable.get(name)
        if extent is None:
            continue
        dim, high = face // 2, face % 2 == 1
        if extent == "high" and not (high and accuracy >= 4):
            continue
        where = "end" if high else "beginning"
        place = f"face {face}, the {where} of dimension {dim}"
        reason = (
            f"{propagator}: boundary {name!r} is not available on this "
            f"propagator ({place}): its guard-ring fill does not conserve a "
            "discrete energy, so the solution can grow without bound."
        )
        if name == "periodic":
            remedy = (
                "There is no periodic face on this propagator; 'scalar' and "
                "'scalar_born' accept 'periodic'."
            )
        elif extent == "high":
            remedy = (
                f"Put the face at the beginning of dimension {dim} instead "
                "(flip the model along that axis), where it does conserve, "
                "or use accuracy=2."
            )
        else:
            remedy = (
                "Model a free surface with a vacuum layer instead -- zero "
                "velocities and density, and so zero stiffnesses and zero "
                "buoyancy, in the cells above the surface, with pml_width 0 "
                "on that face -- as described under 'Free surface in the "
                "elastic propagator' in the boundary conditions "
                "documentation."
            )
        raise DeepwaveNotImplementedError(f"{reason} {remedy}")


# What a bare zero-width "pml" face actually produces, per propagator, as
# (condition phrase, measured coefficient phrase, the ``boundary`` name that
# comes closest, a caveat about that name -- empty when it is exact).
#
# Measured, not asserted: every entry is pinned by
# tests/test_boundary.py::test_bare_zero_width_face_reflection_coefficient.
# The table in docs/boundaries.rst reports the same measurements, at more
# decimal places and split by accuracy order, so it cannot be generated from
# this one: the two must be kept in sync by hand. If you change a row here,
# re-measure and change both. They are not left to trust --
# tests/test_boundary.py::test_docs_reflection_table_matches_the_warning_table
# checks every documented value against the range stated here.
#
# The mechanism behind the split: a bare zero-width face is internally
# ``BC_PML`` (``set_boundary`` returns ``BC_PML`` for it) and a ``BC_PML``
# face is never visited by any guard-ring fill, so its guard cells keep their
# initial zero for the whole run. What that *means* depends on which field
# the guard ring holds. Zeroing the wavefield (the regular-grid, scalar
# family) is a Dirichlet/free surface. On the staggered grids the cells just
# outside the last normal-stress row are the normal velocity, so zeroing them
# is a Neumann/rigid surface for pressure.
#
# The elastic families are the subtle case, and the reason this table carries
# a caveat column at all. Outside the last normal-stress row sit *both* the
# normal velocity and the shear stress, and both stay zero forever. That is
# simultaneously a clamped normal velocity and a traction-free tangential
# direction -- a **frictionless** wall. Measured: R = -1 on the normal
# velocity (clamped, as for ``rigid``) but R = +1 on the tangential velocity
# (free, the *opposite* sign to ``rigid``, which is no-slip). So for the
# elastic family ``boundary="rigid"`` is the nearest named condition but is
# not the same one, and the warning must not imply that it is.
# Appended to the elastic families' message: their nearest named condition is
# not an exact substitute (see the block comment above).
_ELASTIC_RIGID_CAVEAT = (
    " Note that boundary='rigid' is not quite this condition: it is no-slip, "
    "clamping the tangential velocity too (measured -0.991 to -1.000 on it), "
    "whereas a "
    "bare zero-width face leaves it free. Use 'rigid' if you want a genuinely "
    "clamped wall; there is no named condition for the frictionless one."
)

_BARE_ZERO_WIDTH_CONDITION: dict[str, tuple[str, str, str, str]] = {
    "scalar": (
        "a free surface",
        "the wavefield reflects with inverted polarity, measured reflection "
        "coefficient -1.000 to -1.004",
        "free",
        "",
    ),
    "scalar_born": (
        "a free surface",
        "the wavefield reflects with inverted polarity, measured reflection "
        "coefficient -1.000 to -1.004 on both the background and the "
        "scattered wavefield",
        "free",
        "",
    ),
    "acoustic": (
        "a rigid surface for pressure",
        "pressure reflects with the same polarity, measured reflection "
        "coefficient +1.000",
        "rigid",
        "",
    ),
    "acoustic_born": (
        "a rigid surface for pressure",
        "pressure reflects with the same polarity, measured reflection "
        "coefficient +1.000 to +1.006",
        "rigid",
        "",
    ),
    "acoustic_vti": (
        "a rigid surface for pressure",
        "the stresses reflect with the same polarity, measured reflection "
        "coefficient +1.000 on both axes",
        "rigid",
        "",
    ),
    "viscoacoustic": (
        "a rigid surface for pressure",
        "pressure reflects with the same polarity, measured reflection "
        "coefficient +0.998 to +0.999 (the deficit is Q attenuation over the "
        "reflected path, not boundary error)",
        "rigid",
        "",
    ),
    "elastic": (
        "a frictionless rigid wall",
        "the normal stress reflects with the same polarity, measured "
        "reflection coefficient +1.000; the normal velocity is clamped "
        "(measured -1.000) but the tangential velocity is left free "
        "(measured +1.000)",
        "rigid",
        _ELASTIC_RIGID_CAVEAT,
    ),
    "elastic_born": (
        "a frictionless rigid wall",
        "the normal stress reflects with the same polarity, measured "
        "reflection coefficient +1.000 to +1.003; the normal velocity is "
        "clamped but the tangential velocity is left free",
        "rigid",
        _ELASTIC_RIGID_CAVEAT,
    ),
    "elastic_vti": (
        "a frictionless rigid wall",
        "the normal stress reflects with the same polarity, measured "
        "reflection coefficient +1.000 on both axes; the normal velocity is "
        "clamped but the tangential velocity is left free",
        "rigid",
        _ELASTIC_RIGID_CAVEAT,
    ),
}


def _face_name(face: int) -> str:
    """``"the beginning of dimension 0"`` etc., for the zero-width-face warning."""
    end = "the end" if face % 2 else "the beginning"
    return f"{end} of dimension {face // 2}"


def warn_bare_zero_width_face(
    boundary: Optional[Union[str, abc.Sequence[str]]],
    pml_width: abc.Sequence[int],
    ndims: int,
    propagator: str,
    stacklevel: int = 3,
) -> None:
    """Warns that a zero-width PML face is silently a reflecting boundary.

    A face whose ``pml_width`` is zero does not absorb. It becomes a
    reflecting boundary whose type depends on the propagator and the field
    component, and it is an *approximate* one: because the face is
    internally still ``BC_PML``, no guard-ring fill ever runs on it, so it
    gets the accidental all-zero exterior rather than the exact odd/even
    image that ``boundary="free"``/``"rigid"`` install. The output is
    unaffected by this warning -- it is the same call, with a message.

    **Which faces warn.** For each face, this warns when::

        pml_width[face] == 0 and (boundary is None or boundary[face] == "pml")

    against the *normalised* ``boundary`` (see :func:`normalize_boundary`),
    so all three spellings of the same call warn identically: ``boundary``
    omitted, the single string ``"pml"``, and the per-face
    ``["pml"] * 2 * ndims``. ``"pml"`` is the spelled-out default, so
    spelling it out cannot change what a call does -- including what it
    says. A face explicitly named ``"free"``, ``"rigid"`` or ``"periodic"``
    never warns -- the user has said what they meant, and such a face cannot
    have a non-zero width anyway (:func:`set_boundary` enforces that). A
    face named ``"pml"`` with width 0 **does** warn: naming it ``"pml"``
    says it absorbs, and it does not.

    Emitted at most once per process per ``(propagator, face)``, guarded by
    :func:`_warn_once_per_cause` rather than by Python's warning filters,
    because an FWI loop would otherwise emit one per iteration. That also
    means ``warnings.simplefilter("always")`` will not make it repeat; the
    set is the suppression mechanism. To silence it, either name the
    condition with ``boundary=`` (the point of the warning) or filter on its
    category::

        warnings.filterwarnings(
            "ignore", category=deepwave.ZeroWidthFaceWarning
        )

    Not folded into :func:`set_boundary`, which does not know which
    propagator it is serving and so could not name the condition that
    propagator actually produces.

    Args:
        boundary: The user's ``boundary`` argument, before resolution --
            ``None``, or the sequence of names. Needed *unresolved*, because
            the resolved codes cannot distinguish "the user wrote 'pml'"
            from "the user wrote nothing".
        pml_width: The per-face PML width, as resolved by
            :func:`set_pml_width`.
        ndims: The number of spatial dimensions.
        propagator: The calling propagator's name; selects the row of
            ``_BARE_ZERO_WIDTH_CONDITION`` and appears in the message.
        stacklevel: The stack level to fall back on if every frame above
            this one belongs to Deepwave or PyTorch. The default of 3 matches
            the intended chain (user code -> the propagator's public function
            -> this function).

    """
    condition, coefficient, suggested_name, caveat = _BARE_ZERO_WIDTH_CONDITION[
        propagator
    ]
    if boundary is not None:
        boundary = normalize_boundary(boundary, ndims)
    for face in range(2 * ndims):
        if pml_width[face] != 0:
            continue
        if boundary is not None and boundary[face] != "pml":
            continue
        suggested = [
            suggested_name if other == face else "pml" for other in range(2 * ndims)
        ]
        # The "exact image" claim is only true where the suggested name is
        # this same condition. Where it is not (the elastic families, whose
        # caveat is non-empty), the caveat is put immediately after the
        # first mention of the name rather than at the end of the message,
        # so a reader cannot act on the suggestion before reaching it.
        exactness = (
            "which also uses the exact image rather than this approximate "
            "one, and silences this message"
            if not caveat
            else "which silences this message"
        )
        _warn_once_per_cause(
            f"bare-zero-width-face:{propagator}:{face}",
            f"deepwave: zero-width PML face on {_face_name(face)} of "
            f"{propagator} is not absorbing -- it is {condition} "
            f"({coefficient}). This is implied by pml_width="
            f"{list(pml_width)!r} alone; nothing in the call says so. Pass "
            f"boundary={suggested!r} to state a condition explicitly "
            f"({exactness})." + caveat + " Or pass "
            f"deepwave.boundaries({ndims}, "
            f"dim{face // 2}_{'end' if face % 2 else 'begin'}="
            f"{suggested_name!r}) to build a matching (boundary, pml_width) "
            "pair. If you meant the face to absorb, give it a non-zero "
            "pml_width. Silence this warning with: "
            'warnings.filterwarnings("ignore", '
            "category=deepwave.ZeroWidthFaceWarning).",
            ZeroWidthFaceWarning,
            stacklevel,
        )


def boundaries(
    ndim: int,
    *,
    default: str = "pml",
    width: Union[int, abc.Sequence[int]] = 20,
    top: Optional[str] = None,
    dim0_begin: Optional[str] = None,
    dim0_end: Optional[str] = None,
    dim1_begin: Optional[str] = None,
    dim1_end: Optional[str] = None,
    dim2_begin: Optional[str] = None,
    dim2_end: Optional[str] = None,
) -> tuple[list[str], list[int]]:
    """Builds a matching ``(boundary, pml_width)`` pair.

    The two arguments have to agree -- a named boundary face must have
    ``pml_width`` 0, and a zero-width face that is *not* named is an
    implicit reflector (see :ref:`bare-zero-width-face`). Building them
    together makes it impossible to get that wrong::

        boundary, pml_width = deepwave.boundaries(2, top="free")
        out = deepwave.scalar(
            v, dx, dt, ..., boundary=boundary, pml_width=pml_width
        )

    In 3-D, and with a per-face width::

        boundary, pml_width = deepwave.boundaries(
            3, top="free", width=[0, 20, 20, 20, 20, 20]
        )

    Faces are named by axis, in the same order and with the same words the
    ``pml_width`` argument uses:

    - beginning of first (slowest) dimension -- ``dim0_begin``
    - end of first dimension -- ``dim0_end``
    - beginning of second dimension -- ``dim1_begin``
    - end of second dimension -- ``dim1_end``
    - ...

    ``top`` is an alias for ``dim0_begin``, provided because the beginning
    of dimension 0 is conventionally the free surface in seismic models;
    Deepwave itself gives no axis a physical meaning, so use
    ``dim0_begin``...``dim2_end`` if your model is oriented differently.
    There are deliberately no ``bottom``/``left``/``right``/``front``/
    ``back`` spellings: they cannot be defined without inventing a
    handedness convention that the rest of the library refuses to fix (see
    :doc:`/units`).

    Args:
        ndim: The number of spatial dimensions: 1, 2 or 3.
        default: The condition for every face not named by a keyword. One of
            ``"pml"`` (the default), ``"free"``, ``"rigid"``, ``"periodic"``.
        width: The PML width in cells, applied **only to** ``"pml"`` faces:
            a single non-negative int, or a sequence of ``2 * ndim`` of them.
            Every non-``"pml"`` face gets width 0 automatically, which is
            what keeps the two returned lists consistent.
        top: An alias for ``dim0_begin``. Giving both raises.
        dim0_begin: The condition for the beginning of dimension 0.
        dim0_end: The condition for the end of dimension 0.
        dim1_begin: The condition for the beginning of dimension 1.
        dim1_end: The condition for the end of dimension 1.
        dim2_begin: The condition for the beginning of dimension 2.
        dim2_end: The condition for the end of dimension 2.

    Returns:
        ``(boundary, pml_width)``: a list of ``2 * ndim`` condition names and
        a list of ``2 * ndim`` widths, ready to pass straight to any
        propagator's ``boundary=`` and ``pml_width=`` arguments. This helper
        does not know which propagator the pair is for, so it builds any
        combination of names; a propagator that does not accept one of them
        (see :ref:`boundary-faces-accepted`) raises when it is given the pair.

    Raises:
        DeepwaveValueError: If ``ndim`` is not 1, 2 or 3; if a ``dim{k}_*``
            keyword is given for a ``k`` that ``ndim`` does not have; if both
            ``top`` and ``dim0_begin`` are given; if a condition name is not
            recognised; if ``width`` is negative or the wrong length; or if a
            ``"pml"`` face would end up with width 0, which is the implicit
            reflector this helper exists to replace.

    Stability: provisional (see :doc:`/stability`).

    """
    if ndim not in (1, 2, 3):
        raise DeepwaveValueError(
            f"deepwave.boundaries' ndim must be 1, 2 or 3, but got {ndim!r}."
        )
    per_face = [
        (dim0_begin, "dim0_begin"),
        (dim0_end, "dim0_end"),
        (dim1_begin, "dim1_begin"),
        (dim1_end, "dim1_end"),
        (dim2_begin, "dim2_begin"),
        (dim2_end, "dim2_end"),
    ]
    if top is not None:
        if dim0_begin is not None:
            raise DeepwaveValueError(
                "deepwave.boundaries was given both top and dim0_begin, which "
                "name the same face (top is an alias for dim0_begin). Pass "
                "only one."
            )
        per_face[0] = (top, "top")
    for value, keyword in per_face[2 * ndim :]:
        if value is not None:
            raise DeepwaveValueError(
                f"deepwave.boundaries was given {keyword} but ndim={ndim}; "
                f"dimensions are numbered 0..{ndim - 1}."
            )
    accepted = sorted(_BOUNDARY_NAME_TO_CODE)
    for value, keyword in [*per_face[: 2 * ndim], (default, "default")]:
        if value is not None and value not in _BOUNDARY_NAME_TO_CODE:
            raise DeepwaveValueError(
                f"deepwave.boundaries' {keyword} must be one of {accepted}, "
                f"but got {value!r}."
            )
    boundary = [
        default if value is None else value for value, _ in per_face[: 2 * ndim]
    ]

    try:
        if isinstance(width, bool):
            raise DeepwaveValueError("Boundary widths must be whole cell counts.")
        if isinstance(width, int):
            widths = [width] * 2 * ndim
        else:
            raw_widths = list(width)
            widths = [int(w) for w in raw_widths]
            if any(
                isinstance(w, bool) or w != converted
                for w, converted in zip(raw_widths, widths)
            ):
                raise DeepwaveValueError("Boundary widths must be whole cell counts.")
    except (TypeError, ValueError, OverflowError) as e:
        raise DeepwaveValueError(
            "deepwave.boundaries' width must be an int or a sequence of "
            f"{2 * ndim} ints, but got {width!r}."
        ) from e
    if len(widths) != 2 * ndim:
        raise DeepwaveValueError(
            f"deepwave.boundaries' width must have 1 or {2 * ndim} "
            f"entries for ndim={ndim}, but got {len(widths)}."
        )
    if any(w < 0 for w in widths):
        raise DeepwaveValueError(
            f"deepwave.boundaries' width must be non-negative, got {widths!r}."
        )
    for face, (name, w) in enumerate(zip(boundary, widths)):
        if name == "pml" and w == 0:
            raise DeepwaveValueError(
                f"deepwave.boundaries would give {_face_name(face)} the "
                "condition 'pml' with width 0, which is the implicit "
                "reflector this helper exists to replace. Pass "
                "default='free' or default='rigid' (or a per-face name such "
                f"as dim{face // 2}_{'end' if face % 2 else 'begin'}='free') "
                "to say which condition you mean, or give the face a "
                "non-zero width so that it really absorbs."
            )
    pml_width = [w if name == "pml" else 0 for name, w in zip(boundary, widths)]
    # The helper must be incapable of returning a pair the propagators
    # reject, so it runs their own validation on the result. This is what
    # catches the "periodic must be on both faces of a dimension" rule
    # without restating it. The re-raise keeps this function's contract
    # uniform: everything it rejects is a DeepwaveValueError, including the
    # rules it does not implement itself.
    try:
        set_boundary(boundary, pml_width, ndim)
    except (ValueError, RuntimeError, TypeError) as e:
        raise DeepwaveValueError(
            f"deepwave.boundaries would return boundary={boundary!r}, "
            f"pml_width={pml_width!r}, which the propagators reject: {e}"
        ) from e
    return boundary, pml_width


# The per-propagator boundary-face fill machinery (``fill_scalar_boundary``,
# ``fill_acoustic_pressure_boundary``, ``fill_acoustic_velocity_boundary``,
# ``mirror_pad_acoustic_models``, ``mirror_pad_scalar_born_models`` and their
# helpers) lives in :mod:`deepwave._boundaries`; it consumes the ``BC_*``
# codes defined above.


# ``pml_freq`` inference (see ``docs/boundaries.rst``). The peak of the source
# energy spectrum is refined by a parabolic fit to the log-power of the three
# bins around it, which is accurate only when the peak sits well away from DC;
# a record holding just a few cycles is therefore zero-padded first, until its
# peak bin index reaches ``_PML_FREQ_MIN_PEAK_BIN``. Interpolation alone
# leaves errors of order a percent for such short records, and padding alone
# (which is what costs time) is wasted on long ones; together they recover a
# Ricker's own centre frequency to better than 0.1% for every (frequency, nt,
# dt) combination measured (see
# tests/test_common.py::test_infer_pml_freq_recovers_a_ricker_centre_frequency),
# which is what keeps the classic seismic call (25 Hz Ricker) resolving to
# 25.0 and therefore reproducing Deepwave <= 0.0.27 exactly.
_PML_FREQ_MIN_PEAK_BIN = 64
_PML_FREQ_MAX_FFT_LENGTH = 1 << 18
# Upper bound on the number of complex elements in one FFT batch, so that
# inference costs a bounded amount of memory even for a source record with
# very many traces.
_PML_FREQ_MAX_FFT_ELEMENTS = 1 << 22
# The fixed default that Deepwave <= 0.0.27 used. Any inferred value other
# than exactly this one gives a different absorbing boundary from that
# release, and so gets the migration warning below.
_PML_FREQ_LEGACY_DEFAULT = 25.0

# How a resolved ``pml_freq`` was arrived at, in the words the migration
# warning uses. One of these is what :func:`warn_pml_freq_migration`'s
# ``inferred_from`` carries. The two fallback wordings are kept apart because
# they are different situations to act on: a call with no sources at all has
# nothing to infer from and never will, while a call whose sources yielded no
# usable peak has amplitudes worth looking at.
_PML_FREQ_FROM_SPECTRUM = "the peak of the source amplitudes' energy spectrum"
_PML_FREQ_FROM_NO_SOURCES = (
    "a fifth of the Nyquist frequency (0.5 / dt / 5), because this call has no "
    "source amplitudes to infer a dominant frequency from"
)
_PML_FREQ_FROM_DEFAULT = (
    "a fifth of the Nyquist frequency (0.5 / dt / 5), because no dominant "
    "frequency could be inferred from the source amplitudes"
)
_PML_FREQ_FROM_ZERO_DT = "the zero-time-step limit"


class _InferredPmlFreq(float):
    """A ``pml_freq`` that a caller inferred on the propagator's behalf.

    A plain ``float`` in every respect -- ``setup_propagator`` converts it to
    one immediately -- carrying one extra bit: that the user did not choose
    this value. :func:`deepwave.checkpointing.run_segments` resolves
    ``pml_freq`` once for the whole source record and then passes it
    explicitly to each segment; wrapping it lets those segments still report
    "inferred from the source amplitudes" in the grid-cells-per-wavelength
    warning, rather than blaming the user for a value they never gave.

    A ``float`` subclass rather than an extra argument so that it travels
    unchanged through the ``**kwargs`` of any wrapper between the caller and
    the propagator.
    """

    __slots__ = ()


def default_pml_freq(dt: float) -> float:
    """Return ``0.5 / abs(dt) / 5``, or zero for the zero-time-step limit.

    A fifth of the Nyquist frequency: the frequency band that a grid and time
    step of this scale are chosen to propagate well. At the classic seismic
    ``dt = 0.004`` this is exactly the 25.0 Hz that Deepwave <= 0.0.27 used
    unconditionally, and it stays in the right order of magnitude for
    ultrasound (``dt = 1e-8`` gives 10 MHz), where the fixed 25.0 was wrong by
    five orders of magnitude.

    Args:
        dt: The user's time step interval.

    Returns:
        The default PML frequency in the same units as ``1 / dt``.

    """
    return 0.5 / abs(dt) / 5 if dt != 0 else 0.0


def infer_pml_freq(
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    dt: float,
) -> float:
    """Infers a PML frequency from the source amplitudes.

    Returns the peak of the source amplitudes' *energy* spectrum -- the
    frequency at which the sources put the most energy into the model, which
    is the frequency the PML should be tuned to absorb. For a Ricker wavelet
    this returns the wavelet's own centre frequency, so a call that used to
    rely on the fixed 25.0 default with a 25 Hz Ricker keeps the same PML (and
    therefore the same numbers) it had in Deepwave <= 0.0.27.

    The spectra of all shots and all sources within a source-amplitude tensor
    are combined as a **sum of squared magnitudes**, not as the spectrum of
    their sum. That is invariant to each source's delay and polarity, so a
    survey whose sources fire at different times, or with opposite signs, gets
    the same answer as a single source with the same wavelet, rather than one
    dominated by interference between the sources. It is also invariant to
    zero-padding the wavelet in time (the zeros contribute nothing), so
    padding a record out to a longer ``nt`` does not move the inferred value.

    Different source-amplitude *channels* (e.g.
    ``source_amplitudes_pressure_rate``, a pressure rate, and
    ``source_amplitudes_y``, a body force) are in different
    units, so their raw magnitudes are not comparable. Each channel's spectrum
    is therefore normalised to unit total energy before the channels are
    summed: every channel that is present contributes equally, whatever its
    units.

    A record whose spectrum peaks at DC -- a step or a constant, with no
    dominant frequency to speak of -- falls back rather than reporting a
    near-zero frequency: ``pml_freq`` near zero means ``alpha`` near zero,
    i.e. a classic (unshifted) PML, which measures worse than any sensible
    non-zero choice.

    Falls back to :func:`default_pml_freq` when there is nothing usable to
    infer from: no source amplitudes, an all-zero or single-sample record, a
    record with no energy away from DC, or amplitudes whose values cannot be
    read (inside ``vmap`` or ``torch.compile``, where a data-dependent read is
    not permitted; that case also warns, since the run would otherwise get a
    silently different PML from the same call outside the transform). A
    forward-mode dual from ``torch.func.jvp`` is *not* such a case: its primal
    is readable, and it is inferred from exactly as the untransformed call
    is.

    Args:
        source_amplitudes: The user's source-amplitude tensors, one per source
            channel, ``None`` for absent channels. Shape ``(shot, source,
            time)``, at the user's ``dt`` (not the internal, sub-stepped one).
        dt: The user's time step interval.

    Returns:
        The inferred PML frequency, in the same units as ``1 / dt``.

    """
    return _infer_pml_freq(source_amplitudes, dt)[0]


def active_source_amplitudes(
    amplitudes: abc.Sequence[Optional[torch.Tensor]],
    locations: abc.Sequence[Optional[Union[torch.Tensor, "deepwave.Hicks"]]],
) -> list[Optional[torch.Tensor]]:
    """Return caller-rate source records with ignored points zeroed.

    Inference happens before spatial Hicks spreading, so the answer does not
    depend on a point's stencil weights.
    """
    from deepwave._location_interpolation import Hicks

    result: list[Optional[torch.Tensor]] = []
    for amplitude, location in zip(amplitudes, locations):
        if amplitude is None or location is None:
            result.append(amplitude)
            continue
        if not isinstance(amplitude, torch.Tensor):
            raise DeepwaveTypeError("source_amplitudes must be a torch.Tensor.")
        if not isinstance(location, (torch.Tensor, Hicks)):
            raise DeepwaveTypeError("source_locations must be a torch.Tensor or Hicks.")
        positions = (
            location if isinstance(location, torch.Tensor) else location.locations
        )
        if (
            amplitude.ndim != 3
            or positions.ndim != 3
            or positions.shape[:2] != amplitude.shape[:2]
        ):
            # The regular source validator reports the mismatched shape.
            result.append(amplitude)
            continue
        inactive = (positions == IGNORE_LOCATION).all(dim=-1).to(amplitude.device)
        result.append(amplitude.masked_fill(inactive.unsqueeze(-1), 0))
    return result


def _infer_pml_freq(
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    dt: float,
) -> tuple[float, str]:
    """:func:`infer_pml_freq`, also saying how the value was arrived at.

    Returns ``(frequency, inferred_from)``, where ``inferred_from`` is
    :data:`_PML_FREQ_FROM_SPECTRUM`, :data:`_PML_FREQ_FROM_NO_SOURCES` or
    :data:`_PML_FREQ_FROM_DEFAULT` -- the
    words :func:`warn_pml_freq_migration` puts in its message, so that it
    states how this call's value was reached rather than listing the ways it
    might have been. Callers that warn use this; :func:`infer_pml_freq` is the
    plain-value form.

    Called directly (not through :func:`infer_pml_freq`) by everything that
    passes a ``stacklevel`` on, so that the frame chain the ``stacklevel``
    counts is unchanged by this split.
    """
    fallback = default_pml_freq(dt)
    if dt == 0:
        return fallback, _PML_FREQ_FROM_ZERO_DT
    present = [
        amplitude
        for amplitude in source_amplitudes
        if amplitude is not None and amplitude.numel() > 0
    ]
    # A record of one time sample has no spectrum to find a peak in, but it is
    # still a source: only an empty ``present`` means the call has no sources.
    amplitudes = [amplitude for amplitude in present if amplitude.shape[-1] > 1]
    if not present:
        return fallback, _PML_FREQ_FROM_NO_SOURCES
    if not amplitudes:
        return fallback, _PML_FREQ_FROM_DEFAULT
    # ``values_readable``, the same probe the ``max_vel`` inference uses: the
    # spectrum below needs the amplitudes' actual numbers, so the question is
    # whether they can be read, not which transform wrapper is present. A
    # genuine vmap ``BatchedTensor`` (and a tensor holding no data at all)
    # answers no; a forward-mode dual (the ``GradTrackingTensor`` that
    # ``torch.func.jvp`` wraps an amplitude in) has a perfectly readable
    # primal and answers yes, so the jvp primal keeps the PML the
    # untransformed call would have inferred rather than silently disagreeing
    # with it.
    if _torch_is_compiling() or not all(
        values_readable(amplitude) for amplitude in amplitudes
    ):
        # Fallback frame chain: user call -> propagator() ->
        # setup_propagator() -> set_pml_freq() -> _infer_pml_freq().
        _warn_once_per_cause(
            "pml_freq_not_inferable",
            "pml_freq was not specified and could not be inferred from the "
            "source amplitudes, because their values cannot be read while "
            "vmapped or inside torch.compile; using "
            f"{_fmt_3sf(fallback)} (a fifth of the Nyquist frequency). Pass "
            "pml_freq=<your dominant frequency> to make this deterministic "
            "and to match what the same call infers outside the transform.",
            UserWarning,
            stacklevel=5,
        )
        return fallback, _PML_FREQ_FROM_DEFAULT
    peak = _pml_freq_from_spectrum(amplitudes, dt)
    if peak is None:
        return fallback, _PML_FREQ_FROM_DEFAULT
    return peak, _PML_FREQ_FROM_SPECTRUM


def _base_level_dispatch() -> Any:
    """Context manager running its body at the base (non-functorch) level.

    A ``torch.func`` transform wraps tensors allocated under its interpreter,
    leaving no concrete storage for the native backend to address.
    """
    try:
        from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack
    except ImportError:  # older torch without this helper
        return contextlib.nullcontext()
    return temporarily_clear_interpreter_stack()


def unwrap_functorch(t: torch.Tensor) -> torch.Tensor:
    """Return the concrete tensor underneath all ``torch.func`` wrappers.

    Unexpected PyTorch errors propagate: treating a failed unwrap as an
    ordinary tensor could bypass batched-input validation or send a tensor
    without concrete storage to the native backend.
    """
    while _functorch.is_functorch_wrapped_tensor(t):
        t = _functorch.get_unwrapped(t)
    return t


def _pml_freq_from_spectrum(
    amplitudes: list[torch.Tensor], dt: float
) -> Optional[float]:
    """The spectral-peak part of :func:`infer_pml_freq`, at the base level.

    Split out, and run inside :func:`_base_level_dispatch` on unwrapped
    amplitudes, so that every tensor it creates and every value it reads back
    is concrete. Under ``torch.func.jvp`` the amplitudes themselves may be
    ordinary tensors and the *result* of the FFT still be a wrapped one, whose
    values some torch versions refuse to read (``RuntimeError: Cannot access
    data pointer of Tensor that doesn't have storage``, seen on torch 2.8 --
    the newest release that supports this project's Python 3.9 floor -- in
    ``torch.func.jvp``/``hessian``/``vmap(grad(...))`` of a call that omits
    ``pml_freq``). Falling back to the default there would be worse than
    fixing it: the transformed call's primal would get a different absorbing
    boundary from the same call outside the transform, which is exactly the
    disagreement this function's own docstring promises not to introduce.

    Args:
        amplitudes: The non-empty source-amplitude tensors.
        dt: The user's time step interval.

    Returns:
        The inferred PML frequency, or ``None`` if the spectrum holds no
        usable peak (the caller then applies :func:`default_pml_freq`).

    """
    with _base_level_dispatch():
        amplitudes = [unwrap_functorch(amplitude) for amplitude in amplitudes]
        return _pml_freq_peak(amplitudes, dt)


def _pml_freq_peak(amplitudes: list[torch.Tensor], dt: float) -> Optional[float]:
    """Locates the spectral peak and turns it into a frequency, or ``None``."""
    n_time = max(amplitude.shape[-1] for amplitude in amplitudes)
    n_fft = n_time
    power = _source_power_spectrum(amplitudes, n_fft)
    peak = _spectrum_peak_bin(power)
    if peak is None:
        return None
    if peak < _PML_FREQ_MIN_PEAK_BIN:
        # Too few bins across the spectral peak for the parabolic refinement
        # below to be accurate (a record only a few cycles long). Zero-pad in
        # time -- which interpolates the spectrum without changing it -- until
        # the peak is far enough from DC, and look again.
        pad = -(-_PML_FREQ_MIN_PEAK_BIN // peak)
        n_fft = min(pad * n_time, max(n_time, _PML_FREQ_MAX_FFT_LENGTH))
        if n_fft > n_time:
            power = _source_power_spectrum(amplitudes, n_fft)
            peak = _spectrum_peak_bin(power)
            if peak is None:
                return None
    offset = 0.0
    if peak + 1 < power.numel():
        # Parabolic interpolation of the peak, in log-power: the three bins
        # around the peak fit a parabola whose vertex is a sub-bin estimate of
        # where the true peak lies. ``tiny`` keeps the logarithm finite when a
        # neighbouring bin is exactly zero.
        tiny = torch.finfo(torch.float64).tiny
        # PyTorch 2.0 forward-mode tensors can expose scalar .item()
        # values but reject .tolist() because their wrapper has no storage.
        log_power = torch.log(power[peak - 1 : peak + 2] + tiny)
        left, middle, right = (log_power[i].item() for i in range(3))
        curvature = left - 2 * middle + right
        if curvature < 0:
            offset = min(max(0.5 * (left - right) / curvature, -0.5), 0.5)
    pml_freq = (peak + offset) / (n_fft * abs(dt))
    if not math.isfinite(pml_freq) or pml_freq <= 0:
        return None
    # Rounded to three significant figures, for two reasons. It makes the
    # inferred value *stable*: the estimate is good to about 0.005%, so
    # trimming a record, padding it, or adding a shot moves the raw estimate
    # in the fourth digit and would otherwise give every such run a slightly
    # different PML. And it makes the classic seismic call exact -- a 25 Hz
    # Ricker infers 24.9994, which rounds to 25.0 and so reproduces Deepwave
    # <= 0.0.27 bit-for-bit rather than merely closely. Three significant
    # figures is far finer than the PML cares about: a factor of two in
    # pml_freq changes the measured boundary error by tens of per cent.
    #
    # Then capped at the Nyquist frequency: the peak bin itself cannot exceed
    # it, but the parabolic offset and the rounding can each push a
    # near-Nyquist estimate just past it (a 166.6 Hz peak at dt = 0.003 rounds
    # to 167.0, above the 166.67 Nyquist), which would then trip
    # ``set_pml_freq``'s "pass a smaller pml_freq" warning about a value the
    # user never passed.
    return min(_round_3sf(pml_freq), 0.5 / abs(dt))


def _source_power_spectrum(
    amplitudes: abc.Sequence[torch.Tensor],
    n_fft: int,
) -> torch.Tensor:
    """Returns the combined source energy spectrum on the CPU, in float64.

    One entry per ``rfft`` bin of length `n_fft`. Each tensor in `amplitudes`
    contributes the sum over its traces of the squared magnitude of their
    spectra, normalised to unit total energy so that channels in different
    units count equally (see :func:`infer_pml_freq`).

    Args:
        amplitudes: Non-empty source-amplitude tensors, shape ``(..., time)``.
        n_fft: The FFT length (``>=`` the longest record; longer zero-pads).

    Returns:
        A 1-D float64 CPU tensor of length ``n_fft // 2 + 1``.

    """
    n_freq = n_fft // 2 + 1
    # Bound the working set: a record with very many traces is transformed in
    # batches rather than all at once.
    chunk = max(1, _PML_FREQ_MAX_FFT_ELEMENTS // n_freq)
    total = torch.zeros(n_freq, dtype=torch.float64, device=amplitudes[0].device)
    for amplitude in amplitudes:
        flat = amplitude.detach().reshape(-1, amplitude.shape[-1])
        channel = torch.zeros(n_freq, dtype=torch.float64, device=flat.device)
        for start in range(0, flat.shape[0], chunk):
            # The FFT runs in the amplitudes' own precision (a float32 record
            # locates its own spectral peak far more accurately than this
            # needs), which keeps a many-trace record's inference cheap, but
            # the *squaring* is done in float64: squaring a float32 magnitude
            # overflows above ~1.8e19 and flushes to zero below ~1e-19, so a
            # float32 record in, say, pascals rather than gigapascals would
            # otherwise lose its whole spectrum and fall back. Taking ``abs``
            # first and widening before squaring makes the inference invariant
            # to the amplitudes' overall scale.
            spectrum = torch.fft.rfft(flat[start : start + chunk], n=n_fft)
            channel += (spectrum.abs().double() ** 2).sum(0)
        # Normalised on the device, without reading the value back: a
        # ``float()`` here would force a host synchronisation on every
        # propagator call. A silent channel divides by ``tiny`` and so
        # contributes nothing; a channel holding a non-finite value makes the
        # total non-finite, which ``_spectrum_peak_bin`` rejects.
        total += channel / channel.sum().clamp(min=torch.finfo(torch.float64).tiny)
    return total.cpu()


def _spectrum_peak_bin(power: torch.Tensor) -> Optional[int]:
    """Returns the index of the largest bin of `power`, or None.

    None means there is nothing to infer from: the spectrum is empty, holds a
    non-finite value, has no energy away from DC, or peaks *at* DC. A
    DC-dominated record (a step, a constant) has no dominant frequency to
    find, and answering with the largest of its sidelobes -- the lowest
    resolvable frequency, an essentially arbitrary function of the record
    length -- would be worse than saying so.

    Args:
        power: A 1-D energy spectrum, bin 0 being DC.

    Returns:
        The peak bin index (``>= 1``), or None.

    """
    if power.numel() < 2 or not bool(torch.isfinite(power).all()):
        return None
    peak = int(power.argmax())
    if peak == 0 or float(power[1:].sum()) <= 0:
        return None
    return peak


def set_pml_freq(
    pml_freq: Optional[float],
    dt: float,
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]] = (),
    user_dt: Optional[float] = None,
    propagator: Optional[str] = None,
) -> float:
    """Sets or validates the PML frequency.

    An explicitly-provided `pml_freq` is used exactly as given (it is never
    overridden or second-guessed). If it is None, it is inferred from the
    source amplitudes by :func:`infer_pml_freq`, or, when there are no source
    amplitudes to infer from, taken from :func:`default_pml_freq`.

    Args:
        pml_freq: The dominant frequency for PML absorption. If None, it is
            inferred (see above).
        dt: The internal time step interval, used to calculate the Nyquist
            frequency that `pml_freq` is checked against.
        source_amplitudes: The user's source-amplitude tensors, used only when
            `pml_freq` is None. Default: no sources.
        user_dt: The user's time step interval, which is the sample interval
            of `source_amplitudes`. Defaults to `dt` (correct when the
            propagator is not sub-stepping).
        propagator: The calling propagator's name, which every message
            raised here begins with. `None` omits the prefix.

    Returns:
        The resolved PML frequency.

    Raises:
        DeepwaveTypeError: If `pml_freq` is not None or convertible to a
            float.
        DeepwaveValueError: If `pml_freq` is negative. Signed and zero time
            steps are permitted.

    """
    prefix = f"{propagator}: " if propagator else ""
    if pml_freq is not None:
        try:
            pml_freq = float(pml_freq)
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError(
                f"{prefix}pml_freq must be None or convertible to a float, "
                f"but got {pml_freq!r}."
            ) from e
    if not isinstance(dt, float):
        raise DeepwaveTypeError(f"{prefix}dt must be a float, but got {dt!r}.")
    nyquist = 0.5 / abs(dt) if dt != 0 else math.inf
    if pml_freq is None:
        if user_dt is None:
            user_dt = dt
        pml_freq, inferred_from = _infer_pml_freq(source_amplitudes, user_dt)
        # The fallback frame chain: user call -> propagator() ->
        # setup_propagator() -> set_pml_freq() -> warn_pml_freq_migration().
        warn_pml_freq_migration(pml_freq, stacklevel=5, inferred_from=inferred_from)
    if pml_freq < 0:
        raise DeepwaveValueError(
            f"{prefix}pml_freq must be non-negative, but got {pml_freq}."
        )
    if pml_freq > nyquist:
        # Same fallback frame chain as above, one frame shorter.
        warn_at_user_call(
            f"pml_freq {pml_freq} is greater than the Nyquist frequency {nyquist}; "
            "pass a smaller pml_freq, or reduce dt so the Nyquist frequency "
            "rises above it.",
            UserWarning,
            4,
        )
    return pml_freq


def warn_pml_freq_migration(
    pml_freq: float, stacklevel: int, inferred_from: str
) -> None:
    """Warns that an inferred `pml_freq` differs from the old fixed default.

    Deepwave <= 0.0.27 used a fixed ``pml_freq`` of 25.0 whenever the argument
    was omitted. It is now inferred from the source amplitudes, which is a
    deliberate behaviour change for calls that omitted it. The classic seismic
    call (a 25 Hz wavelet) infers exactly 25.0, so the frequency migration
    warning stays silent; **any** other inferred value changes the
    absorbing boundary, and therefore the numbers, so it is said out loud,
    together with the one line that restores the old frequency setting. There
    is no quiet band: a small change in the boundary is still a change, and a
    caller comparing against a v0.0.27 result needs to know it happened.

    This warning is a migration aid, and is removed in Deepwave 0.4.0.

    Args:
        pml_freq: The inferred PML frequency.
        stacklevel: The stack level to fall back on if every frame above this
            one belongs to Deepwave or PyTorch, counted from this function's
            own frame. The level actually used is
            :func:`deepwave._exceptions.stacklevel_outside_deepwave`'s,
            because the paths that reach this warning -- a propagator
            function, a ``Module``, a ``checkpoint_segments=`` delegation,
            ``run_segments`` -- are all different depths below the user's own
            line.
        inferred_from: How this value was arrived at, in the words the
            message uses: :data:`_PML_FREQ_FROM_SPECTRUM`,
            :data:`_PML_FREQ_FROM_NO_SOURCES` or
            :data:`_PML_FREQ_FROM_DEFAULT`.

    """
    if pml_freq == _PML_FREQ_LEGACY_DEFAULT or inferred_from == _PML_FREQ_FROM_ZERO_DT:
        # At dt=0 the frequency cannot change the update; no migration warning
        # applies, and the Nyquist fallback has no meaning.
        return
    _warn_once_per_cause(
        # Keyed by the rounded value so an FWI loop, whose repeated calls
        # infer the same frequency, warns once rather than once per iteration.
        f"pml_freq_migration:{_fmt_3sf(pml_freq)}",
        "pml_freq was not specified, so it was resolved to "
        f"{_fmt_3sf(pml_freq)} -- {inferred_from} (see docs/boundaries.rst). "
        f"Deepwave 0.0.27 and earlier instead always used "
        f"{_PML_FREQ_LEGACY_DEFAULT} here, so this call's absorbing boundary, "
        "and therefore its output, differs from what that release produced. "
        "Pass pml_freq=25.0 to restore the old PML frequency setting, or pass "
        "pml_freq=<your dominant frequency> to say what you meant (either "
        "silences this message, a migration warning that is removed in "
        "Deepwave 0.4.0). Silence it with: "
        'warnings.filterwarnings("ignore", '
        "category=deepwave.PmlFreqInferenceWarning).",
        PmlFreqInferenceWarning,
        stacklevel=stacklevel,
    )


def warn_born_edge_scatter_migration(
    scatter: torch.Tensor,
    pad: abc.Sequence[int],
    ndim: int,
    stacklevel: int,
) -> None:
    """Warns that edge cells of ``scalar_born``'s ``dv`` now extend outwards.

    ``scalar_born`` used to pad its scattering model with zeros, so a
    perturbation of the outermost cells of the model region stopped at the
    edge of that region while the background velocity it perturbs was
    extended outwards. The scattered wavefield was then not the linearisation
    of :func:`deepwave.scalar` for such a perturbation. The scattering model
    is now extended outwards like every other model, on this propagator as on
    :func:`deepwave.acoustic_born` and :func:`deepwave.elastic_born`, which
    makes the linearisation exact everywhere.

    That changes the numbers only for a call whose scattering model is
    non-zero in the outermost cells of the region the run extracts, so the
    warning is emitted only when it is -- which needs one reduction over
    those cells. The reduction is skipped when the values cannot be read (a
    ``vmap`` batched model, or a ``torch.compile`` trace, where a
    data-dependent read would break the graph), as the other migration checks
    are, and when the message has already been said.

    Args:
        scatter: The padded scattering model, as
            :func:`setup_propagator` returns it.
        pad: The per-face padding, in cells, that was added to the extracted
            region, length ``2 * ndim``, in the order used throughout
            Deepwave (dim 0 begin, dim 0 end, dim 1 begin, ...).
        ndim: The number of spatial dimensions.
        stacklevel: The stack level to fall back on if every frame above this
            one belongs to Deepwave or PyTorch, counted from this function's
            own frame.

    """
    cause = "born_edge_scatter"
    if (
        _already_warned_once(cause)
        or _torch_is_compiling()
        or not values_readable(scatter)
        or scatter.numel() == 0
    ):
        return
    if not _region_edge_is_nonzero(scatter, pad, ndim):
        return
    _warn_once_per_cause(
        cause,
        "dv is non-zero in the outermost cells of the model region, and "
        "those values are now extended outwards into the padding around it "
        "(the absorbing layer and the finite-difference halo) instead of "
        "being replaced by zeros there. Deepwave 0.0.27 and earlier put "
        "zeros in that padding, so this call's scattered wavefield differs "
        "from what that release computed -- only through those edge cells' "
        "extension into the padding, and only for a dv like this one; a dv "
        "that is zero in the outermost cells of the region computes what it "
        "always did. The scattered wavefield is now the linearisation of "
        "scalar with respect to the velocity for any dv, which is what "
        "acoustic_born and elastic_born already gave. No argument restores "
        "the old numbers: to reproduce a 0.0.27 result, enlarge the model "
        "(or the region survey_pad extracts) so that dv is zero in its "
        "outermost cells. This is a migration warning, removed in Deepwave "
        "0.4.0. Silence it with: "
        'warnings.filterwarnings("ignore", '
        "category=deepwave.MigrationWarning).",
        MigrationWarning,
        stacklevel=stacklevel,
    )


def _region_edge_is_nonzero(
    model: torch.Tensor,
    pad: abc.Sequence[int],
    ndim: int,
) -> bool:
    """Whether a padded model is non-zero on any padded face of its region.

    Reads the outermost cell layer of the unpadded region on each face that
    was actually padded: those are the values replicate padding extends and
    zero padding did not, so they are exactly the cells whose contents decide
    whether the two paddings differ. A face with no padding is skipped, since
    nothing is extended there.
    """
    for dim in range(ndim):
        axis = model.dim() - ndim + dim
        for side in range(2):
            face = 2 * dim + side
            if pad[face] == 0:
                continue
            index = pad[face] if side == 0 else model.shape[axis] - pad[face] - 1
            if not 0 <= index < model.shape[axis]:
                continue
            if bool(model.select(axis, index).any()):
                return True
    return False


def check_q_freq(q_freq: Any, propagator: str, *, parameter: str = "q_freq") -> float:
    """Validates a given (non-None) ``q_freq`` and returns it as a float.

    ``q_freq`` is the reference frequency at which the ``Q`` model was
    specified, so it has to be a single positive, finite number: a
    non-finite one makes ``omega0 = 2*pi*q_freq`` (or the relaxation times
    derived from it) non-finite and fills the whole wavefield with NaNs
    several thousand time steps later, and a value of the wrong type used to
    fail with whatever error the first arithmetic on it happened to raise.
    Both are caught here instead, with a message naming the propagator and
    the value, as :mod:`deepwave._exceptions` requires.

    A one-element tensor or a NumPy scalar is accepted, as any number-like
    value is: the check is on what the value *is*, not on which library
    produced it. A ``bool`` is not a frequency, and a ``str`` is not either,
    even though ``float("25")`` would succeed.

    Args:
        q_freq: The user's ``q_freq``, already known not to be ``None``.
        propagator: The calling propagator's name, for the message.
        parameter: The argument name, also used for attenuation-band endpoints.

    Returns:
        ``q_freq`` as a float.

    Raises:
        deepwave.DeepwaveTypeError: If ``q_freq`` is not a single number.
        deepwave.DeepwaveValueError: If it is not positive and finite.

    """
    if isinstance(q_freq, (bool, str, bytes)):
        raise DeepwaveTypeError(
            f"{propagator}: {parameter} must be a positive number (the frequency at "
            f"which Q is specified), but got {q_freq!r}."
        )
    try:
        value = float(q_freq)
    except (TypeError, ValueError, RuntimeError) as e:
        raise DeepwaveTypeError(
            f"{propagator}: {parameter} must be a positive number (the frequency at "
            f"which Q is specified), but got {q_freq!r}."
        ) from e
    if not math.isfinite(value):
        raise DeepwaveValueError(
            f"{propagator}: {parameter} must be finite, but got {q_freq}."
        )
    if value <= 0:
        raise DeepwaveValueError(
            f"{propagator}: {parameter} must be positive, but got {q_freq}."
        )
    return value


def set_max_vel(max_vel: Optional[float], max_abs_model_vel: float) -> float:
    """Sets or validates the maximum velocity for the CFL condition.

    Args:
        max_vel: The user-specified maximum velocity. If None, `max_abs_model_vel`
            is used.
        max_abs_model_vel: The maximum absolute velocity present in the model.

    Returns:
        The validated maximum absolute velocity.

    Raises:
        TypeError: If `max_vel` or `max_model_vel` are not convertible to a float.
        ValueError: If an explicit `max_vel` is nonfinite or below the
            conservative model speed.

    """
    if max_vel is not None:
        try:
            max_vel = float(max_vel)
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError(
                "max_vel must be None or convertible to a float."
            ) from e
    try:
        max_abs_model_vel = float(max_abs_model_vel)
    except (TypeError, ValueError) as e:
        raise DeepwaveTypeError(
            "max_abs_model_vel must be convertible to a float."
        ) from e
    if max_vel is None:
        return max_abs_model_vel
    max_vel = abs(max_vel)
    if not math.isfinite(max_vel):
        raise DeepwaveValueError("maximum absolute velocity must be finite.")
    if max_vel < max_abs_model_vel and not math.isclose(
        max_vel, max_abs_model_vel, rel_tol=1e-12
    ):
        raise DeepwaveValueError(
            f"max_vel={max_vel} is less than the model's conservative maximum "
            f"wave speed (~{max_abs_model_vel:.6g}, including the "
            "density-weighted stencil bound where applicable). Increase max_vel or "
            "omit it so Deepwave derives the bound from the material."
        )
    return max_vel


def set_nt(
    nt: Optional[int],
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    step_ratio: int,
) -> int:
    """Sets or validates the number of time steps.

    Args:
        nt: The desired number of time steps. If None, it is inferred from
            `source_amplitudes`. When both are given, `nt` wins: the
            amplitudes are zero-padded or truncated in time to that length
            (see :func:`set_source_amplitudes`).
        source_amplitudes: A sequence of optional Tensors, where each torch.Tensor
            contains the amplitudes of the sources for a source type.
        step_ratio: The ratio between the user-specified time step and the
            internal time step.

    Returns:
        The total number of time steps, adjusted by `step_ratio`.

    Raises:
        TypeError: If `nt` is not an int or None, or if `source_amplitudes`
            contains non-torch.Tensor elements.
        ValueError: If `step_ratio` is less than 1, or if `nt` is negative.
        RuntimeError: If `nt` and `source_amplitudes` are both None.

    """
    if nt is not None and not isinstance(nt, int):
        raise DeepwaveTypeError("nt must be an int or None.")
    if step_ratio < 1:
        raise DeepwaveValueError("step_ratio must be >= 1")
    source_amplitudes_not_none = next(
        (a for a in source_amplitudes if a is not None),
        None,
    )
    source_amplitudes_nt = None
    if source_amplitudes_not_none is not None:
        if not isinstance(source_amplitudes_not_none, torch.Tensor):
            raise DeepwaveTypeError("source_amplitudes must be a torch.Tensor or None.")
        source_amplitudes_nt = source_amplitudes_not_none.shape[-1]
    if nt is None:
        if source_amplitudes_nt is None:
            raise DeepwaveRuntimeError("nt or source amplitudes must be specified")
        nt = source_amplitudes_nt
    if nt < 0:
        raise DeepwaveValueError("nt must be >= 0")
    return nt * step_ratio


def set_model_gradient_sampling_interval(
    model_gradient_sampling_interval: int,
    *,
    user_nt: Optional[int] = None,
    model_requires_grad: bool = True,
) -> int:
    """Validates the model gradient sampling interval.

    ``model_gradient_sampling_interval`` (``mgsi``) multiplies the CFL
    ``step_ratio`` to give the interval at which the forward wavefield is
    stored for the gradient quadrature. The current history allocation uses
    complete sampling blocks and requires ``user_nt % mgsi == 0``. Integration
    chunks and callback observations are scheduled independently of that
    history stride; this restriction is checked before any stepping.

    When no model requires a gradient there is nothing for ``mgsi`` to
    sample, so it is reset to 1: the interval then has no effect on any
    returned value and no reason to shorten the propagation.

    Args:
        model_gradient_sampling_interval: The interval at which to sample
            the wavefield for model gradient calculation. It multiplies
            `step_ratio` to give the interval at which the forward wavefield
            is stored, which is used as a divisor and modulus, so it must be
            at least 1 (1 means every step is sampled).
        user_nt: The number of time steps the user asked for (*before* the
            CFL `step_ratio` upsampling). ``None`` skips the divisibility
            check, for callers that only want the type/range validation.
        model_requires_grad: Whether any model requires a gradient *and* a
            gradient will actually be recorded (callers should combine
            ``requires_grad`` with :func:`torch.is_grad_enabled`, so that an
            inference run under ``torch.no_grad()`` is not refused for a
            gradient it will never compute). When ``False`` the returned
            interval is 1 regardless of what was requested.

    Returns:
        The validated sampling interval, or 1 if no model requires a
        gradient.

    Raises:
        deepwave.DeepwaveTypeError: If `model_gradient_sampling_interval` is
            not an integer. This is also a `TypeError`, so an existing
            ``except TypeError`` still catches it.
        deepwave.DeepwaveValueError: If `model_gradient_sampling_interval` is
            less than 1, or if it does not divide `user_nt`. This is also a
            `ValueError`, so an existing ``except ValueError`` still catches
            it.

    """
    if isinstance(model_gradient_sampling_interval, bool) or not isinstance(
        model_gradient_sampling_interval, int
    ):
        raise DeepwaveTypeError("model_gradient_sampling_interval must be an int.")
    if model_gradient_sampling_interval < 1:
        raise DeepwaveValueError("model_gradient_sampling_interval must be >= 1.")
    if not model_requires_grad:
        return 1
    if (
        user_nt is not None
        and model_gradient_sampling_interval > 1
        and user_nt % model_gradient_sampling_interval != 0
    ):
        remainder = user_nt % model_gradient_sampling_interval
        raise DeepwaveValueError(
            f"model_gradient_sampling_interval "
            f"({model_gradient_sampling_interval}) must divide the number of "
            f"time steps ({user_nt}), but leaves a remainder of "
            f"{remainder}. The model-gradient history layout requires "
            "complete sampling blocks. "
            "Use a number of time steps that is a multiple of "
            "model_gradient_sampling_interval (e.g. "
            f"{user_nt - remainder} or "
            f"{user_nt - remainder + model_gradient_sampling_interval}), or "
            "a sampling interval that divides it."
        )
    return model_gradient_sampling_interval


def _validate_unit_fraction(value: float, name: str) -> float:
    """Validates that ``value`` is a float in the closed interval [0, 1].

    Args:
        value: The value to validate.
        name: The argument name to use in error messages.

    Returns:
        The validated value as a float.

    Raises:
        TypeError: If ``value`` is not convertible to a float.
        ValueError: If ``value`` is not within the range [0, 1].
    """
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise DeepwaveTypeError(f"{name} must be convertible to a float.") from e
    if not 0.0 <= value <= 1.0:
        raise DeepwaveValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def set_freq_taper_frac(freq_taper_frac: float) -> float:
    """Validates the frequency taper fraction (the fraction to taper)."""
    return _validate_unit_fraction(freq_taper_frac, "freq_taper_frac")


def set_time_pad_frac(time_pad_frac: float) -> float:
    """Validates the time padding fraction (the fraction to pad with zeros)."""
    return _validate_unit_fraction(time_pad_frac, "time_pad_frac")


def check_source_amplitudes_locations_match(
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    source_locations: abc.Sequence[Optional[torch.Tensor]],
    propagator: Optional[str] = None,
    ndims: Optional[int] = None,
) -> None:
    """Ensures source_amplitudes and source_locations match.

    This function verifies that the number of source amplitude tensors matches
    the number of source location tensors, and that for each pair, they are
    either both None or both non-None. If non-None, it checks that they are
    `torch.Tensor` objects and have matching sizes in their
    `n_sources_per_shot` dimension.

    Args:
        source_amplitudes: A sequence of optional Tensors, where each
            torch.Tensor contains the amplitudes of the sources for a source type.
        source_locations: A sequence of optional Tensors, where each
            torch.Tensor contains the locations of the sources for a source type.
        propagator: The calling propagator's name, which every message raised
            here begins with, and which names the caller's own keywords in
            place of an internal source-type index. `None` omits both.
        ndims: The number of spatial dimensions, which selects the channel
            list the keyword names come from.

    Raises:
        RuntimeError: If the lengths of `source_amplitudes` and
            `source_locations` do not match, or if a pair of
            amplitude/location is inconsistent (one is None and the other is
            not), or if their `n_sources_per_shot` dimensions do not match.
        TypeError: If `source_amplitudes` or `source_locations` are not
            `torch.Tensor` when expected to be.

    """
    prefix = f"{propagator}: " if propagator else ""
    if len(source_amplitudes) != len(source_locations):
        raise DeepwaveRuntimeError(
            f"{prefix}the same number of source_amplitudes "
            f"({len(source_amplitudes)}) "
            f"and source_locations ({len(source_locations)}) must be provided.",
        )
    # The caller's own spellings, so that a missing partner names the
    # keyword that is missing rather than an internal source-type index.
    names = _location_group_names(
        "source_locations", len(source_locations), propagator, ndims, "source"
    )
    suffixes = [
        name.removeprefix("source_locations")
        if name.startswith("source_locations")
        else ""
        for name in names
    ]
    for i, (amplitudes, locations) in enumerate(
        zip(source_amplitudes, source_locations)
    ):
        if (amplitudes is None) != (locations is None):
            raise unpaired_source_channel_error(
                propagator, suffixes[i], amplitudes is None
            )
        if amplitudes is not None and locations is not None:
            if not isinstance(amplitudes, torch.Tensor):
                raise DeepwaveTypeError("source_amplitudes must be a torch.Tensor.")
            if not isinstance(locations, torch.Tensor):
                raise DeepwaveTypeError("source_locations must be a torch.Tensor.")
            if amplitudes.ndim != 3:
                raise DeepwaveRuntimeError(
                    "source amplitudes Tensors should have 3 dimensions, but "
                    f"found one with {amplitudes.ndim}.",
                )
            if locations.ndim != 3:
                raise DeepwaveRuntimeError(
                    "source locations Tensors should have 3 dimensions, but "
                    f"found one with {locations.ndim}.",
                )
            if amplitudes.shape[1] != locations.shape[1]:
                raise DeepwaveRuntimeError(
                    "Expected source amplitudes and locations to be the same "
                    "size in the n_sources_per_shot dimension, got "
                    f"{amplitudes.shape[1]} and {locations.shape[1]}.",
                )


def fit_source_amplitudes_to_nt(
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    user_nt: int,
) -> list[Optional[torch.Tensor]]:
    """Zero-pads or truncates source amplitudes in time to ``user_nt``.

    Applied when the caller gave ``nt`` explicitly, which then sets the
    length of the run: a record shorter than ``nt`` is zero-padded at the end
    (the sources fall silent and the propagation continues), and a longer one
    is truncated (the samples beyond ``nt`` are never injected, so no
    gradient reaches them). It works in the caller's own time samples, before
    any internal resampling, so the result is exactly what passing a
    hand-padded or hand-truncated record would have produced.

    Anything that is not a three-dimensional Tensor is passed through
    untouched, for :func:`set_source_amplitudes` to reject with its own
    message.

    Args:
        source_amplitudes: One optional Tensor per source channel.
        user_nt: The requested number of user-level time samples.

    Returns:
        The amplitudes, each of length ``user_nt``.

    """
    fitted: list[Optional[torch.Tensor]] = []
    for amplitudes in source_amplitudes:
        if (
            not isinstance(amplitudes, torch.Tensor)
            or amplitudes.ndim != 3
            or amplitudes.shape[-1] == user_nt
        ):
            fitted.append(amplitudes)
        elif amplitudes.shape[-1] > user_nt:
            fitted.append(amplitudes[..., :user_nt])
        else:
            fitted.append(
                torch.nn.functional.pad(amplitudes, (0, user_nt - amplitudes.shape[-1]))
            )
    return fitted


def set_source_amplitudes(
    source_amplitudes: abc.Sequence[Optional[torch.Tensor]],
    n_batch: int,
    nt: int,
    step_ratio: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Prepares source amplitudes for propagation.

    This function validates the input source amplitudes, ensures consistency
    in device, dtype, and shape, upsamples them if necessary, and moves the
    time dimension to the slowest dimension.

    Args:
        source_amplitudes: A sequence of optional Tensors, where each
            torch.Tensor contains the amplitudes of the sources for a source type.
        n_batch: The batch size.
        nt: The total number of time steps.
        step_ratio: The ratio between the user-specified time step and the
            internal time step.
        freq_taper_frac: The fraction of the frequency spectrum to taper.
        time_pad_frac: The fraction of the time axis to pad with zeros.
        time_taper: Whether to apply a Hann window in time.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.

    Returns:
        List[torch.Tensor]: A list of processed source amplitude Tensors.

    Raises:
        TypeError: If `source_amplitudes` contains elements that are not
            `torch.Tensor` or `None`.
        RuntimeError: If there are inconsistencies in the dimensions, device,
            dtype, or batch size of the source amplitudes.

    """
    result: list[torch.Tensor] = []
    for amplitudes in source_amplitudes:
        if amplitudes is None:
            result.append(torch.empty(nt, n_batch, 0, device=device, dtype=dtype))
            continue
        if not isinstance(amplitudes, torch.Tensor):
            raise DeepwaveTypeError("source_amplitudes must be a torch.Tensor.")
        if amplitudes.ndim != 3:
            raise DeepwaveRuntimeError(
                "source amplitudes Tensors should have 3 dimensions, but "
                f"found one with {amplitudes.ndim}.",
            )
        if amplitudes.device != device:
            raise DeepwaveRuntimeError(
                "Inconsistent device: Expected all Tensors be on device "
                f"{device}, but found a source amplitudes torch.Tensor on "
                f"device {amplitudes.device}.",
            )
        if amplitudes.dtype != dtype:
            raise DeepwaveRuntimeError(
                "Inconsistent dtype: Expected source amplitudes to have "
                f"datatype {dtype}, but found one with dtype "
                f"{amplitudes.dtype}.",
            )
        if amplitudes.shape[0] != n_batch or amplitudes.shape[2] * step_ratio != nt:
            raise DeepwaveRuntimeError(
                "source_amplitudes has shape "
                f"{tuple(amplitudes.shape)}, but Deepwave expects "
                f"[shot, source, time] = [{n_batch}, *, {nt // step_ratio}]. "
                "If source_amplitudes is [time, shot, ...] instead of "
                "[shot, ..., time], transpose the first and last axes.",
            )
        result.append(
            torch.movedim(
                upsample(
                    amplitudes,
                    step_ratio=step_ratio,
                    freq_taper_frac=freq_taper_frac,
                    time_pad_frac=time_pad_frac,
                    time_taper=time_taper,
                ),
                -1,
                0,
            ).contiguous(),
        )
    return result


def check_points_per_wavelength(
    min_nonzero_vel: float,
    pml_freq: float,
    grid_spacing: abc.Sequence[float],
    inferred: bool = False,
) -> None:
    """Checks if there are enough grid points per wavelength.

    Warns the user if the number of grid cells per wavelength falls below
    a recommended threshold (6 cells per wavelength).

    Args:
        min_nonzero_vel: The minimum non-zero velocity in the model.
        pml_freq: The dominant frequency for PML absorption.
        grid_spacing: The spacing between grid points in each spatial dimension.
        inferred: Whether `pml_freq` was inferred from the source amplitudes
            rather than provided by the user. Only affects the wording of the
            warning, which then says where the frequency came from.

    Raises:
        ValueError: If `min_nonzero_vel`, `pml_freq`, or any element of
            `grid_spacing` is negative or zero.

    """
    # Fallback frame chain (both warnings below): user call -> propagator() ->
    # setup_propagator() -> check_points_per_wavelength().
    if min_nonzero_vel == 0:
        warn_at_user_call(
            "The test for the number of grid cells per wavelength is being "
            "skipped because the model's minimum non-zero velocity could not "
            "be determined (e.g. under vmap or torch.compile, or every "
            "velocity is zero); check resolution manually if in doubt.",
            UserWarning,
            4,
        )
        return
    if min_nonzero_vel < 0:
        raise DeepwaveValueError("min_nonzero_vel must be non-negative.")
    if pml_freq < 0:
        raise DeepwaveValueError("pml_freq must be non-negative.")
    for g in grid_spacing:
        if g <= 0:
            raise DeepwaveValueError("grid_spacing elements must be positive.")

    min_wavelength = float("inf") if pml_freq == 0 else abs(min_nonzero_vel / pml_freq)
    max_spacing = max(abs(dim_spacing) for dim_spacing in grid_spacing)
    cells_per_wavelength = min_wavelength / max_spacing
    if cells_per_wavelength < 6:
        _warn_once_per_cause(
            "points_per_wavelength",
            "At least six grid cells per wavelength is recommended, but there "
            f"are only {cells_per_wavelength:.2f} (frequency ~{_fmt_3sf(pml_freq)}"
            f"{', inferred from the source amplitudes' if inferred else ''}, "
            f"minimum non-zero velocity ~{_fmt_3sf(min_nonzero_vel)}, grid "
            f"spacing ~{_fmt_3sf(max_spacing)}). The velocity is the smallest "
            "non-zero value in the model, so a single slow cell, such as a thin "
            "fluid or near-vacuum layer, can set it. Use a finer grid_spacing; "
            "pass pml_freq=<your dominant frequency> if this is not the "
            "frequency you need to resolve; or raise accuracy, which lowers "
            "dispersion at the same sampling, though this six-cell figure is "
            "calibrated for the default accuracy=4 and does not move with it. "
            "deepwave.suggest_discretisation returns a grid spacing and time "
            "step that satisfy both this check and the CFL condition.",
            UserWarning,
            stacklevel=4,
        )


def cosine_taper_end(signal: torch.Tensor, n_taper: int) -> torch.Tensor:
    """Tapers the end of the final dimension of a torch.Tensor using a cosine.

    A half period, shifted and scaled to taper from 1 to 0, is used. The
    taper is dimensionless, so the result is in the same units as `signal`,
    whatever they are, and the sample interval of the final dimension is
    unchanged.

    Args:
        signal:
            The torch.Tensor that will have its final dimension tapered. Its
            units are the caller's own and are carried through.
        n_taper:
            The length of the cosine taper, in number of samples of the
            final dimension.

    Returns:
        The tapered signal, in the units of `signal`.

    """
    if not isinstance(signal, torch.Tensor):
        raise DeepwaveTypeError("signal must be a torch.Tensor.")
    if not isinstance(n_taper, int):
        raise DeepwaveTypeError("n_taper must be an int.")
    if n_taper < 0:
        raise DeepwaveValueError("n_taper must be non-negative.")
    n_taper = min(n_taper, signal.shape[-1])
    if n_taper > 0:
        taper_values = (
            torch.cos(
                torch.arange(1, n_taper + 1, device=signal.device) / n_taper * math.pi
            )
            + 1
        ).to(signal.dtype) / 2
        signal[..., -n_taper:] *= taper_values
    return signal


def zero_last_element_of_final_dimension(signal: torch.Tensor) -> torch.Tensor:
    """Sets the last element of the final dimension of a torch.Tensor to zero.

    Args:
        signal: The torch.Tensor to modify.

    Returns:
        The modified torch.Tensor with the last element of its final dimension
        set to zero.

    Raises:
        TypeError: If `signal` is not a `torch.Tensor`.

    """
    if not isinstance(signal, torch.Tensor):
        raise DeepwaveTypeError("signal must be a torch.Tensor.")
    if signal.numel() > 0:
        signal[..., -1] = 0
    return signal


def _validate_sample_args(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    shift: Optional[float] = None,
) -> tuple[float, float, Optional[float]]:
    """Validate arguments for upsample and downsample.

    Returns:
        ``(freq_taper_frac, time_pad_frac, shift)`` converted to ``float``
        (``shift`` stays ``None`` if it was ``None``) -- callers must use
        these returned values, not their own original arguments, so that a
        convertible-but-not-already-`float` input (e.g. a numeric string)
        is actually converted everywhere it is used, not just where this
        function itself compares it.
    """
    if not isinstance(signal, torch.Tensor):
        raise DeepwaveTypeError("signal must be a torch.Tensor.")
    if not isinstance(step_ratio, int):
        raise DeepwaveTypeError("step_ratio must be an int.")
    if step_ratio <= 0:
        raise DeepwaveValueError("step_ratio must be positive.")
    try:
        freq_taper_frac = float(freq_taper_frac)
    except (TypeError, ValueError) as e:
        raise DeepwaveTypeError("freq_taper_frac must be a float.") from e
    if not 0.0 <= freq_taper_frac <= 1.0:
        raise DeepwaveValueError(
            f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}."
        )
    try:
        time_pad_frac = float(time_pad_frac)
    except (TypeError, ValueError) as e:
        raise DeepwaveTypeError("time_pad_frac must be a float.") from e
    if not 0.0 <= time_pad_frac <= 1.0:
        raise DeepwaveValueError(
            f"time_pad_frac must be in [0, 1], got {time_pad_frac}."
        )
    if not isinstance(time_taper, bool):
        raise DeepwaveTypeError("time_taper must be a bool.")
    if shift is not None:
        try:
            shift = float(shift)
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError("shift must be a float.") from e
    return freq_taper_frac, time_pad_frac, shift


def upsample(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
) -> torch.Tensor:
    """Upsamples the final dimension of a torch.Tensor by a factor.

    Low-pass upsampling is used to produce an upsampled signal without
    introducing higher frequencies than were present in the input. The
    Nyquist frequency of the input will be zeroed.

    Units are the caller's own and are carried through: the result is in the
    same units as `signal`, and its sample interval is `signal`'s divided by
    `step_ratio`. `step_ratio` is a count and the two fractions are
    dimensionless.

    Args:
        signal:
            The torch.Tensor that will have its final dimension upsampled.
            Its final dimension is time, uniformly sampled; its units are
            the caller's own.
        step_ratio:
            The integer factor by which the signal will be upsampled.
            The input signal is returned if this is 1 (freq_taper_frac,
            time_pad_frac, and time_taper will be ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered before
            upsampling. Defaults to 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before upsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the input signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the signal will be used. Defaults to 0.0.
        time_taper:
            A bool specifying whether to force the upsampled signal to zero
            at both edges (with a Hann window) after its inverse FFT. The
            FFT-based resampling treats the time axis as periodic. A signal
            that does not already decay to (near) zero at its edges -- e.g.
            real/field data cropped from a longer recording -- otherwise
            produces wraparound ringing in the resampled result. Leave as
            False if the signal already tapers to (near) zero at its edges,
            since the window also attenuates genuine signal there.

    Returns:
        The signal after upsampling, in the units of `signal`, at a sample
        interval of `signal`'s divided by `step_ratio`.

    """
    freq_taper_frac, time_pad_frac, _ = _validate_sample_args(
        signal, step_ratio, freq_taper_frac, time_pad_frac, time_taper
    )
    if signal.numel() == 0 or step_ratio == 1:
        return signal

    n_time_pad = int(time_pad_frac * signal.shape[-1]) if time_pad_frac > 0.0 else 0
    if n_time_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_time_pad))
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    signal_f = torch.fft.rfft(signal, norm="ortho") * math.sqrt(step_ratio)
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    if nt % 2 == 0 and signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)
    pad_len = up_nt // 2 + 1 - signal_f.shape[-1]
    if pad_len > 0:
        signal_f = torch.nn.functional.pad(signal_f, (0, pad_len))
    signal = torch.fft.irfft(signal_f, n=up_nt, norm="ortho")
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad * step_ratio]
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1],
            periodic=False,
            device=signal.device,
        )
    return signal


def downsample(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> torch.Tensor:
    """Downsamples the final dimension of a torch.Tensor by a factor.

    Frequencies higher than or equal to the Nyquist frequency of the
    downsampled signal will be zeroed before downsampling.

    The input time length must be divisible by ``step_ratio``. Pad or trim
    the signal explicitly when a different output length is needed.

    Units are the caller's own and are carried through: the result is in the
    same units as `signal`, and its sample interval is `signal`'s multiplied
    by `step_ratio`. `step_ratio` is a count, the two fractions are
    dimensionless, and `shift` is in samples of `signal`.

    Args:
        signal:
            The torch.Tensor that will have its final dimension downsampled.
            Its final dimension is time, uniformly sampled; its units are
            the caller's own. Its length must be divisible by
            ``step_ratio``.
        step_ratio:
            The integer factor by which the signal will be downsampled.
            The input signal is returned if this is 1 and shift is 0
            (freq_taper_frac, time_pad_frac, and time_taper will be
            ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered after
            downsampling. Defaults to 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before downsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the output signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the output signal will be used.
            Defaults to 0.0.
        time_taper:
            A bool specifying whether to force the signal to zero at both
            edges (with a Hann window) before this function's FFT-based
            resampling, which treats the time axis as periodic. A signal
            that does not already decay to (near) zero at its edges -- e.g.
            real/field data cropped from a longer recording -- otherwise
            produces wraparound ringing in the resampled result. Leave as
            False if the signal already tapers to (near) zero at its edges,
            since the window also attenuates genuine signal there.
        shift:
            Amount, in samples of the input `signal`, to shift the data in
            time before downsampling.
            Defaults to 0.0.

    Returns:
        The signal after downsampling, in the units of `signal`, at a sample
        interval of `signal`'s multiplied by `step_ratio`.

    """
    freq_taper_frac, time_pad_frac, validated_shift = _validate_sample_args(
        signal, step_ratio, freq_taper_frac, time_pad_frac, time_taper, shift=shift
    )
    shift = validated_shift if validated_shift is not None else shift
    if signal.numel() == 0 or (step_ratio == 1 and shift == 0.0):
        return signal

    if signal.shape[-1] % step_ratio != 0:
        raise DeepwaveValueError(
            "signal time length must be divisible by step_ratio for downsampling; "
            "pad or trim the signal explicitly."
        )

    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1],
            periodic=False,
            device=signal.device,
        )
    n_time_pad = (
        int(time_pad_frac * (signal.shape[-1] // step_ratio))
        if time_pad_frac > 0.0
        else 0
    )
    if n_time_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_time_pad * step_ratio))
    nt = signal.shape[-1]
    down_nt = nt // step_ratio
    signal_f = torch.fft.rfft(signal, norm="ortho")[..., : down_nt // 2 + 1]
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    if down_nt % 2 == 0 and signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)
    if shift != 0.0:
        freqs = torch.fft.rfftfreq(signal.shape[-1], device=signal.device)[
            : down_nt // 2 + 1
        ]
        signal_f *= torch.exp(-1j * 2 * math.pi * freqs * shift)
    signal = torch.fft.irfft(signal_f, n=down_nt, norm="ortho") / math.sqrt(step_ratio)
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad]
    return signal


def extract_survey(
    models: abc.Sequence[torch.Tensor],
    source_locations: abc.Sequence[Optional[torch.Tensor]],
    receiver_locations: abc.Sequence[Optional[torch.Tensor]],
    wavefields: abc.Sequence[Optional[torch.Tensor]],
    survey_pad: Optional[Union[int, abc.Sequence[Optional[int]]]],
    origin: Optional[abc.Sequence[int]],
    fd_pad: abc.Sequence[int],
    pml_width: abc.Sequence[int],
    model_pad_modes: abc.Sequence[str],
    n_batch: int,
    ndims: int,
    device: torch.device,
    dtype: torch.dtype,
    propagator_name: Optional[str] = None,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[tuple[int, int]],
]:
    """Extracts a region of the model, pads it and prepares related tensors.

    The region can either be chosen based on `survey_pad` and the
    source and receiver locations, or using `origin` and initial
    wavefields.

    This function extracts the appropriate region of the model and
    adjusts source and receiver locations to be relative to the
    new origin.

    This function performs necessary processing to prepare the models and
    locations. Padding is added to the models, and the locations are adjusted
    appropriately. A batch dimension is added to the models if they don't
    already have one. The device and dtype of all models and locations are
    checked and they are made contiguous. Locations that are None are changed
    to be empty Tensors.

    Args:
        models: A sequence of PyTorch Tensors representing the model
            parameters (e.g., velocity, density).
        source_locations: A sequence of optional Tensors, where each
            torch.Tensor contains the locations of the sources for source type.
        receiver_locations: A sequence of optional Tensors, where each
            torch.Tensor contains the locations of the receivers for a receiver type.
        wavefields: A sequence of optional Tensors representing initial
            wavefields.
        survey_pad: Optional padding around the survey area. Can be an int
            or a sequence of optional ints.
        origin: Optional sequence of integers specifying the origin of the
            survey within the full model.
        fd_pad: A sequence of integers specifying the padding for finite
            difference stencils in each dimension.
        pml_width: The width of the PML (Perfectly Matched Layer) in grid
            cells.
        model_pad_modes: A sequence of strings specifying the padding mode
            for each model.
        n_batch: The batch size.
        ndims: The number of spatial dimensions of the model.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.
        propagator_name: The calling propagator's name, used to name the
            caller's own location arguments in any error raised here. ``None``
            falls back to a positional description.

    Returns:
        Tuple:

            - models_out: Processed model Tensors.
            - source_locations_out: Processed source location Tensors.
            - receiver_locations_out: Processed receiver location Tensors.
            - wavefields_out: Processed wavefield Tensors.
            - survey_extents: The resolved ``[beginning, end)`` extents (one
              tuple per dimension) of the survey region within the full model.

    """
    if any(not isinstance(m, torch.Tensor) for m in models):
        raise DeepwaveTypeError("Models must be a torch.Tensor.")
    if survey_pad is not None and origin is not None:
        raise DeepwaveRuntimeError("survey_pad and origin cannot both be specified.")
    locations: list[Optional[torch.Tensor]] = list(source_locations) + list(
        receiver_locations,
    )
    model_spatial_shape = list(models[0].shape[-ndims:])
    location_names = _location_group_names(
        "source_locations", len(source_locations), propagator_name, ndims, "source"
    ) + _location_group_names(
        "receiver_locations",
        len(receiver_locations),
        propagator_name,
        ndims,
        "receiver",
    )
    check_locations_are_within_model(
        model_spatial_shape, locations, location_names, propagator_name
    )
    pad = [fd + pml for fd, pml in zip(fd_pad, pml_width)]
    if survey_pad is None and any(w is not None for w in wavefields):
        survey_extents = get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
            model_spatial_shape=model_spatial_shape,
            locations=locations,
        )
        check_extents_within_model(survey_extents, model_spatial_shape)
        check_locations_within_extents(survey_extents, locations)
    else:
        survey_extents = get_survey_extents_from_locations(
            model_spatial_shape,
            locations,
            survey_pad,
        )
        check_extents_match_wavefields_shape(survey_extents, wavefields, pml_width)
    return (
        list(
            extract_models(
                models,
                survey_extents,
                pad,
                model_pad_modes,
                n_batch,
                device,
                dtype,
            ),
        ),
        list(
            extract_locations(
                "Source",
                source_locations,
                survey_extents,
                pad,
                n_batch,
                device,
            ),
        ),
        list(
            extract_locations(
                "Receiver",
                receiver_locations,
                survey_extents,
                pad,
                n_batch,
                device,
            ),
        ),
        list(
            prepare_wavefields(
                wavefields,
                survey_extents,
                pml_width,
                n_batch,
                device,
                dtype,
            ),
        ),
        survey_extents,
    )


def unpaired_source_channel_error(
    propagator: Optional[str], suffix: str, amplitudes_missing: bool
) -> DeepwaveRuntimeError:
    """The one error for a source channel given only half of its pair.

    Amplitudes and locations describe the same sources, so one without the
    other is always a mistake, and it is the same mistake on every
    propagator and every channel. The message names both of the caller's own
    keywords, in the caller's own spelling, so the fix is the traceback.

    Args:
        propagator: The calling propagator's name, which the message begins
            with when it is known.
        suffix: The text that follows ``source_amplitudes``/
            ``source_locations`` in the caller's own keywords, such as
            ``"_y"``; the empty string on a propagator whose single channel
            takes unsuffixed keywords.
        amplitudes_missing: Whether it is the amplitudes that were not given.

    Returns:
        The exception to raise.

    """
    prefix = f"{propagator}: " if propagator else ""
    given, missing = (
        ("source_locations", "source_amplitudes")
        if amplitudes_missing
        else ("source_amplitudes", "source_locations")
    )
    return DeepwaveRuntimeError(
        f"{prefix}{given}{suffix} was provided but {missing}{suffix} was not "
        "-- the two must be given together, or neither."
    )


def _location_group_names(
    prefix: str,
    n: int,
    propagator: Optional[str] = None,
    ndims: Optional[int] = None,
    kind: str = "source",
) -> list[str]:
    """Names for a group of `n` location Tensors sharing one argument `prefix`.

    An error about a location must name the argument the *caller* wrote. The
    shared setup code gathers a propagator's location arguments
    positionally, so left to itself it can only say ``source_locations[2]``
    -- an internal slot index, which is not a spelling the user can search
    their own call for, and which differs between propagators for the same
    physical channel. Given the propagator's name, the channels are read
    from its spec instead and the names are the real ones
    (``source_locations_y``, or ``sources={"y": ...}``'s flat equivalent),
    so the message points at something the caller actually typed.

    A group of one Tensor (the scalar family, whose spec sets
    ``channel_suffixes=False``) is named ``prefix`` unchanged, which is
    already the caller's spelling. The positional ``f"{prefix}[i]"`` form
    survives only as the fallback for a caller that does not pass a
    propagator name (``deepwave.plan``'s probes and the shared helpers'
    own tests).

    Args:
        prefix: The unsuffixed argument stem, e.g. ``"source_locations"``.
        n: How many location Tensors are in the group.
        propagator: The calling propagator's name, if known.
        ndims: The number of spatial dimensions, which selects the channel
            list.
        kind: ``"source"`` or ``"receiver"``, selecting which channel list.

    Returns:
        One name per location Tensor, in group order.

    """
    if n <= 1:
        return [prefix]
    if propagator is None or ndims is None:
        return [f"{prefix}[{i}]" for i in range(n)]
    names = _location_argument_names(propagator, ndims, kind)
    if names is not None and len(names) == n:
        return names
    return [f"{prefix}[{i}]" for i in range(n)]


#: Where each propagator puts its non-spatial (pressure-like) channels in the
#: list of location arguments it hands to the shared setup code, relative to
#: the staggered ``z``/``y``/``x`` components. The acoustic family inserts
#: the pressure channel at the front; the elastic family appends it. They are
#: both free to, because the order is internal -- which is exactly why an
#: error must not report a position in this list as if it were something the
#: caller wrote. ``acoustic_vti`` has no entry: its pressure channel is
#: synthetic (it expands into ``sigma_h``/``sigma_v``), so its non-spatial
#: channels are named individually below.
_NON_SPATIAL_CHANNELS_FIRST = frozenset({"acoustic", "acoustic_born", "viscoacoustic"})

#: Propagators whose non-spatial channels are gathered in an order other than
#: the one their spec lists them in. ``acoustic_vti`` gathers the two
#: stiffness-component channels ``sigma_h`` before ``sigma_v``, while its
#: signature takes them the other way round.
_NON_SPATIAL_CHANNEL_ORDER: dict[str, tuple[str, ...]] = {
    "acoustic_vti": ("sigma_h", "sigma_v"),
}


def _location_argument_names(
    propagator: str, ndims: int, kind: str
) -> Optional[list[str]]:
    """The caller's own location-argument names, in the order they are gathered.

    An error about a location has to name the argument the caller wrote, not
    the slot the shared setup code happens to hold it in: ``sources={"y":
    ...}``'s location is ``source_locations_y`` on every propagator, while
    its *slot* is 0 on ``elastic`` and 1 on ``acoustic``. The slot index is
    also not a string the caller can find in their own call.

    Returns ``None`` -- so the caller falls back to the indexed description
    -- whenever the spec cannot be consulted. ``tests/test_common.py``
    checks the result against every propagator's real arguments, one channel
    at a time, rather than trusting the layout rule encoded here.

    Args:
        propagator: The calling propagator's name.
        ndims: The number of spatial dimensions.
        kind: ``"source"`` or ``"receiver"``.

    Returns:
        One argument name per gathered location, or ``None``.

    """
    try:
        from deepwave.introspection import _resolve_spec

        spec = _resolve_spec(propagator)
        born = spec.has_bg_receivers()
        if kind == "source":
            names = _one_group(spec, propagator, ndims, "source")
            # A Born propagator runs two propagations from one set of
            # sources, so it hands the same list twice. Both halves are the
            # caller's same argument, so both halves get its name.
            return names * 2 if born else names
        if born:
            # Background receivers first, then scattered: two different
            # arguments, so two different names.
            return _one_group(spec, propagator, ndims, "bg_receiver") + _one_group(
                spec, propagator, ndims, "sc_receiver"
            )
        return _one_group(spec, propagator, ndims, "receiver")
    except Exception:  # noqa: BLE001 - a naming aid must never be the failure
        return None


def _one_group(
    spec: "deepwave._engine.PropagatorSpec", propagator: str, ndims: int, kind: str
) -> list[str]:
    """The argument names of one channel group, in the gathered order."""
    if kind == "source":
        channels = list(spec.source_channels(ndims))
        synthetic = spec.synthetic_source_channels
        index = 1
    else:
        channels = list(spec.receiver_channels(ndims))
        synthetic = spec.synthetic_receiver_channels
        index = 0
    channels = [c for c in channels if c not in synthetic]
    spatial: list[str] = [axis for axis in ("z", "y", "x") if axis in channels]
    other = [c for c in channels if c not in spatial]
    override = _NON_SPATIAL_CHANNEL_ORDER.get(propagator)
    if override is not None:
        other = [c for c in override if c in other] + [
            c for c in other if c not in override
        ]
    ordered = (
        other + spatial
        if propagator in _NON_SPATIAL_CHANNELS_FIRST
        else spatial + other
    )
    return [spec.channel_kwargs(kind, channel)[index] for channel in ordered]


def _apply_one_location_mask(
    propagator: str,
    locations_name: str,
    locations: Any,
    mask_name: str,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Converts one boolean activity mask into the location sentinel.

    Masks and :data:`IGNORE_LOCATION` are one mechanism, not two: a masked-out
    point becomes a point whose coordinates are all the sentinel, which is
    exactly what the rest of the library already knows how to skip. Doing the
    conversion here, once, is what makes the two spellings produce
    bit-identical results.
    """
    if locations is None:
        raise DeepwaveValueError(
            f"{propagator}: {mask_name} was given without {locations_name}. A "
            "mask selects which of that component's points are active, so "
            "there must be points for it to select from."
        )
    if not isinstance(locations, torch.Tensor):
        raise DeepwaveValueError(
            f"{propagator}: {mask_name} cannot be combined with a "
            f"deepwave.Hicks object passed as {locations_name}. A Hicks "
            "object carries its own already-resolved points; disable points "
            "inside it by setting all of their coordinates to "
            "deepwave.IGNORE_LOCATION in the locations it was built from."
        )
    if mask.dtype is not torch.bool:
        raise DeepwaveValueError(
            f"{propagator}: {mask_name} must have dtype torch.bool (True = "
            f"active), but has dtype {mask.dtype}."
        )
    if tuple(mask.shape) != tuple(locations.shape[:-1]):
        raise DeepwaveValueError(
            f"{propagator}: {mask_name} has shape {tuple(mask.shape)}, which "
            f"does not match the leading dimensions of {locations_name}, "
            f"shape {tuple(locations.shape)}. A mask must be "
            f"[shot, point] = {tuple(locations.shape[:-1])}."
        )
    sentinel_rows = (locations == IGNORE_LOCATION).any(dim=-1)
    if bool(sentinel_rows.any()):
        count = int(sentinel_rows.sum())
        raise DeepwaveValueError(
            f"{propagator}: both a mask ({mask_name}) and the "
            f"deepwave.IGNORE_LOCATION sentinel (in {locations_name}) were "
            "given for the same component. Use one or the other: the mask is "
            "the modern spelling ([shot, point] bool, True = active) and the "
            f"sentinel is the original. {count} rows contain the sentinel."
        )
    return locations.masked_fill(~mask.unsqueeze(-1), IGNORE_LOCATION)


def apply_location_masks(
    spec: "deepwave._engine.PropagatorSpec", flat: dict[str, Any]
) -> None:
    """Folds every ``*_mask`` entry of ``flat`` into its locations entry.

    The single place where the boolean-mask spelling of "this point is
    inactive" becomes the :data:`IGNORE_LOCATION` spelling, so the two cannot
    diverge and a masked call is bit-identical to the equivalent
    sentinel call. Runs before
    :func:`check_locations_are_within_model`, so a masked-out point's
    coordinates are never bounds-checked -- they are ignored entirely and need
    not be valid, in range, or unique.

    The mask entries are removed from ``flat`` once consumed, so the caller
    reads back only amplitudes and locations.

    Args:
        spec: The calling propagator's spec.
        flat: The propagator's flat keywords, by name. Modified in place.

    Raises:
        deepwave.DeepwaveValueError: If a mask is given without locations, is
            not boolean, does not match its locations' shape, is combined with
            a :class:`deepwave.Hicks`, or is combined with locations that
            already contain the sentinel.

    """
    kinds = ["source", "sc_receiver" if spec.has_bg_receivers() else "receiver"]
    if spec.has_bg_receivers():
        kinds.append("bg_receiver")
    for kind in kinds:
        for channel in spec.all_channels(kind):
            names = spec.channel_kwargs(kind, channel)
            locations_name, mask_name = names[-2], names[-1]
            mask = flat.pop(mask_name, None)
            if mask is None:
                continue
            flat[locations_name] = _apply_one_location_mask(
                spec.name,
                locations_name,
                flat.get(locations_name),
                mask_name,
                mask,
            )


def check_locations_are_within_model(
    model_shape: abc.Sequence[int],
    locations: abc.Sequence[Optional[torch.Tensor]],
    names: Optional[abc.Sequence[str]] = None,
    propagator: Optional[str] = None,
) -> None:
    """Checks if all locations are within the bounds of the model.

    The happy path stays a single vectorised reduction (one ``.all()`` per
    location Tensor per dimension, unchanged in kind from the check this
    replaced); the offending index is only looked up -- with further, more
    expensive, data-dependent indexing (``nonzero`` plus a scalar read) --
    once a check has already failed, so this does not add any new
    data-dependent read to the success path. Note that the ``.all()``
    reduction itself is still a data-dependent read: under
    ``torch.compile``/``torch.export`` it is the known, separate,
    pre-existing branch that graph-breaks/raises
    ``GuardOnDataDependentSymNode`` (see ``test_scalar.py``'s
    ``test_torch_export_no_longer_fails_on_velocity_extremes``); fixing that
    is out of scope here.

    The all-or-none ``IGNORE_LOCATION`` check below adds exactly **one** more
    such read per location Tensor (``bool(partial.any())``), on the success
    path, before the per-dimension ones. That is deliberate and accepted: this
    function already performs ``ndim`` of them per Tensor, so it neither
    introduces the graph break/``torch.export`` failure (which the existing
    reads already cause) nor changes its character -- it adds one device
    synchronisation per location Tensor per call alongside the ``ndim`` that
    were already there. There is no cheaper formulation, for the same reason
    the bounds check itself cannot be made static: whether a row mixes the
    sentinel with real coordinates is a property of the values, not the shape.

    This branch is deliberately *not* gated behind
    ``torch.compiler.is_compiling()`` the way some other input-validation
    reads in this module are (see ``mask_locations``,
    ``_validate_location_tensor``). Those are skippable under compilation
    because every consumer downstream re-derives the same safe values
    regardless (e.g. ``mask_locations`` always zeroes ignored entries;
    ``_validate_location_tensor``'s integer check is superseded by an
    unconditional ``.long()`` truncation later). This check is different:
    it is the *only* place that confirms a location coordinate actually
    indexes inside ``model_shape`` before that coordinate is turned into a
    1D offset (``extract_locations``/``_convert_to_1d_and_check_uniqueness``)
    and handed to the C/CUDA backend for direct memory gather/scatter. There
    is no independent bounds check on that path (by design, the backend
    trusts the offsets it is given), so skipping this under
    ``torch.compiler.is_compiling()`` would let an out-of-bounds location
    reach the backend as silent undefined behaviour (out-of-bounds memory
    access) instead of a clear Python exception -- a correctness/memory
    -safety regression, not merely a lost diagnostic. There is also no
    bounded rewrite that keeps the check itself static: the whole point of
    the check is to branch on tensor *values* supplied at run time, and a
    static (shape-only) approximation cannot tell whether those values are
    in bounds. So this data-dependent branch, and the resulting
    ``torch.export`` failure, are retained on purpose.

    Args:
        model_shape: A sequence of integers representing the spatial shape
            of the model.
        locations: A sequence of optional Tensors, where each torch.Tensor
            contains the locations to check. Each Tensor is expected to have
            shape ``[shot, point, len(model_shape)]``.
        names: Optional argument name for each entry of `locations`, used to
            identify which input a failure came from (e.g.
            ``"source_locations"``). Defaults to ``"locations[i]"`` when not
            given, or for entries beyond the end of `names`.
        propagator: The calling propagator's name, which the out-of-bounds
            message begins with, matching the propagator-level staggered
            -location bounds errors raised for the same mistake in the same
            code path. `None` (the shared helpers' own tests,
            ``deepwave.plan``'s probes) omits the prefix.

    Raises:
        deepwave.DeepwaveValueError: If `model_shape` is empty or contains
            non-positive elements. This is also a `ValueError`, so an
            existing ``except ValueError`` still catches it.
        deepwave.DeepwaveRuntimeError: If any location is outside the model
            bounds, or mixes `IGNORE_LOCATION` with real coordinates within
            one point. This is also a `RuntimeError`, so an existing
            ``except RuntimeError`` still catches it.

    """
    if not model_shape:
        raise DeepwaveValueError("model_shape must not be empty.")
    for dim_shape in model_shape:
        if dim_shape <= 0:
            raise DeepwaveValueError("model_shape elements must be positive.")
    propagator_prefix = f"{propagator}: " if propagator else ""

    for i, location in enumerate(locations):
        if location is None:
            continue
        name = names[i] if names is not None and i < len(names) else f"locations[{i}]"
        if location.shape[-1] != len(model_shape):
            # Not reachable via the public API: get_ndim (called by every
            # propagator before setup_propagator, which is this function's
            # only caller) already enforces that every location Tensor's
            # last dimension equals the number of spatial dimensions, and
            # `model_shape` here is sliced to exactly that many dimensions
            # by construction. Kept as a defensive invariant check in case a
            # future caller reaches this function without going through
            # get_ndim first.
            raise AssertionError(
                f"{name} has {location.shape[-1]} coordinate columns, but "
                f"the model has {len(model_shape)} spatial dimensions."
            )
        # IGNORE_LOCATION marks an unused source/receiver *slot*, so it is
        # all-or-none across a point's coordinates (see deepwave/__init__.py:
        # "a source or receiver whose coordinates are set to
        # IGNORE_LOCATION"). The per-dimension check below validates each
        # coordinate independently and so used to let a partially-sentinel row
        # such as ``[15, IGNORE_LOCATION]`` through, with two silent failure
        # modes downstream in _convert_to_1d_and_check_uniqueness: with
        # dimension 0 valid, the sentinel coordinate contributes 0 and the
        # point lands on the *edge* of the padded grid (inside the FD pad/PML)
        # and really is injected/recorded there instead of being skipped; with
        # dimension 0 the sentinel, the result is
        # ``IGNORE_LOCATION + stride * coord``, which is no longer the
        # sentinel at all -- surviving in the compiled kernels only by virtue
        # of their ``if (0 <= locations[i])`` guard, and raising a confusing
        # gather error on the eager path.
        is_ignored = location == IGNORE_LOCATION
        partial = is_ignored.any(-1) & ~is_ignored.all(-1)
        if bool(partial.any()):
            bad_idx = partial.nonzero()[0].tolist()
            if len(bad_idx) >= 2:
                location_desc = f"shot {bad_idx[0]}, point {bad_idx[1]}"
            elif len(bad_idx) == 1:
                location_desc = f"index {bad_idx[0]}"
            else:
                location_desc = "index ()"
            coords = location[tuple(bad_idx)].tolist()
            raise DeepwaveRuntimeError(
                f"{name} mixes deepwave.IGNORE_LOCATION with real coordinates "
                f"within a single point: at {location_desc} the coordinates "
                f"are {coords}. IGNORE_LOCATION (currently {IGNORE_LOCATION}) "
                "marks an entire unused source/receiver slot, so it must be "
                "set in every dimension of a point or in none of them. Set "
                "the remaining coordinate(s) of this point to "
                "IGNORE_LOCATION to skip it, or give it a real index."
            )
        for dim, model_dim_shape in enumerate(model_shape):
            dim_location = location[..., dim]
            valid = (dim_location == IGNORE_LOCATION) | (
                (dim_location >= 0) & (dim_location < model_dim_shape)
            )
            if bool(valid.all()):
                continue
            # Failure path only from here on: pinpoint the first offending
            # entry for the error message. This is only reached once the
            # check above has already determined (via the one data-dependent
            # read it performs either way) that an error will be raised, so
            # it adds no new data-dependent read to the passing case.
            bad_idx = (~valid).nonzero()[0].tolist()
            value = dim_location[tuple(bad_idx)].item()
            if len(bad_idx) >= 2:
                location_desc = f"shot {bad_idx[0]}, point {bad_idx[1]}"
            elif len(bad_idx) == 1:
                location_desc = f"index {bad_idx[0]}"
            else:
                location_desc = "index ()"
            if value < 0:
                problem = f"is negative ({value})"
            else:
                problem = (
                    f"is {value}, but the model has only {model_dim_shape} "
                    f"cells (valid indices 0 to {model_dim_shape - 1})"
                )
            raise DeepwaveRuntimeError(
                f"{propagator_prefix}{name} is out of bounds: at "
                f"{location_desc}, dimension "
                f"{dim} (of {len(model_shape)}), the coordinate {problem}. "
                "Locations are grid-cell indices, not physical distances -- "
                "if you have coordinates in metres (or another physical "
                "unit), divide by grid_spacing first. Negative indices are "
                "not supported, except the sentinel deepwave.IGNORE_LOCATION "
                f"(currently {IGNORE_LOCATION}), which marks an unused "
                "source/receiver slot."
            )


def set_survey_pad(
    survey_pad: Optional[Union[int, abc.Sequence[Optional[int]]]],
    ndim: int,
) -> list[int]:
    """Checks `survey_pad` and converts it to a list if it is a scalar.

    Args:
        survey_pad: Optional padding around the survey area. Can be an int
            or a sequence of optional ints.
        ndim: The number of spatial dimensions.

    Returns:
        A list of integers representing the padding for each side of each
        dimension. Negative values indicate padding to the model edge.

    Raises:
        RuntimeError: If `ndim` is not positive, or if `survey_pad` contains
            invalid values or has an incorrect length.

    """
    if ndim <= 0:
        raise DeepwaveRuntimeError("ndim must be positive.")

    # Expand to list
    survey_pad_list: list[int] = []
    if survey_pad is None:
        survey_pad_list = [-1] * 2 * ndim
    elif isinstance(survey_pad, int):
        if survey_pad < 0:
            raise DeepwaveRuntimeError("survey_pad must be non-negative.")
        survey_pad_list = [survey_pad] * 2 * ndim
    else:
        for pad in survey_pad:
            if pad is None:
                survey_pad_list.append(-1)
            elif isinstance(pad, int):
                if pad >= 0:
                    survey_pad_list.append(pad)
                else:
                    raise DeepwaveRuntimeError(
                        "survey_pad entries must be None or non-negative ints.",
                    )
            else:
                raise DeepwaveRuntimeError(
                    "survey_pad entries must be None or non-negative ints.",
                )
    if len(survey_pad_list) != 2 * ndim:
        raise DeepwaveRuntimeError(
            f"survey_pad must have length 2 * dims in model, but got "
            f"{len(survey_pad_list)}.",
        )
    return survey_pad_list


def get_survey_extents_from_locations(
    model_shape: abc.Sequence[int],
    locations: abc.Sequence[Optional[torch.Tensor]],
    survey_pad: Optional[Union[int, abc.Sequence[Optional[int]]]],
) -> list[tuple[int, int]]:
    """Calculate the extents of the model to use for the survey.

    Args:
        model_shape:
            A list containing the shape of the full model.
        locations:
            A list of Tensors containing source/receiver locations.
        survey_pad:
            A list with two entries for each dimension, specifying the padding
            to add around the sources and receivers included in all of the
            shots being propagated. If None, the padding continues to the edge
            of the model.

    Returns:
        A list of length equal to the number of dimensions in the model,
        specifying the extents of the model, as a tuple [beginning, end)
        for each dimension, that will be used for wave propagation.

    Raises:
        ValueError: If `model_shape` is empty or contains non-positive
            elements.

    """
    if not model_shape:
        raise DeepwaveValueError("model_shape must not be empty.")
    for dim_shape in model_shape:
        if dim_shape <= 0:
            raise DeepwaveValueError("model_shape elements must be positive.")

    ndims = len(model_shape)
    extents: list[tuple[int, int]] = []
    survey_pad_list = set_survey_pad(survey_pad, ndims)
    for dim in range(ndims):
        left_pad = survey_pad_list[dim * 2]
        left_extent = get_survey_extents_one_side(
            left_pad,
            "left",
            dim,
            locations,
            model_shape[dim],
        )
        right_pad = survey_pad_list[dim * 2 + 1]
        right_extent = get_survey_extents_one_side(
            right_pad,
            "right",
            dim,
            locations,
            model_shape[dim],
        )
        extents.append((left_extent, right_extent))
    return extents


def get_survey_extents_one_side(
    pad: int,
    side: str,
    dim: int,
    locations: abc.Sequence[Optional[torch.Tensor]],
    shape: int,
) -> int:
    """Get the survey extent for the left or right side of one dimension.

    Args:
        pad:
            Positive int specifying padding for the side, or negative if
            the extracted model should continue to the edge of the model.
        side:
            'left' or 'right'.
        dim:
            Integer specifying the dimension index.
        locations:
            List of Tensors with coordinates for the current dimension.
        shape:
            Int specifying length of full model in current dimension.

    Returns:
        Min/max index as int.

    """
    # if pad < 0, return the edge of the model in that direction (0 if left,
    # the model shape if right). Otherwise, initialise extreme_location to
    # the other side of the model so that we can find the min (left) or
    # max (right) location below.
    if side == "left":
        if pad < 0 or not any(location is not None for location in locations):
            return 0
        extreme_location = shape
    else:
        if pad < 0 or not any(location is not None for location in locations):
            return shape
        extreme_location = 0

    # Find the min (left) or max (right) extent in order for all locations to
    # be at least `pad` distant from the edge.
    for location in locations:
        if location is not None:
            dim_location = location[..., dim]
            dim_location = dim_location[dim_location != IGNORE_LOCATION]
            if dim_location.numel() == 0:
                continue
            # Check if the location is within the model bounds. This check is
            # duplicated from check_locations_are_within_model but is
            # necessary here to ensure that the extents are calculated
            # correctly.
            if dim_location.min() < 0:
                raise DeepwaveRuntimeError("Locations must be >= 0.")
            if dim_location.max() >= shape:
                raise DeepwaveRuntimeError(
                    f"Locations must be within model bounds (0 to {shape - 1}).",
                )

            if side == "left":
                extreme_location = min(
                    extreme_location,
                    int(dim_location.min().item()) - pad,
                )
            else:
                extreme_location = max(
                    extreme_location,
                    int(dim_location.max().item()) + pad,
                )
    # The right side of the extent is not included - it is one past the end. It
    # currently holds the largest value that should be included, though, so we
    # add 1.
    if side == "right":
        extreme_location += 1
    # The extents should not be outside the model, so we clip them to the
    # extents of the model.
    return min(max(extreme_location, 0), shape)


def padded_spatial_shape(
    extents: abc.Sequence[tuple[int, int]],
    pml_width: abc.Sequence[int],
) -> list[int]:
    """The spatial shape of the simulation domain: the extents plus the PML."""
    return [
        extents[dim][1] - extents[dim][0] + pml_width[2 * dim] + pml_width[2 * dim + 1]
        for dim in range(len(extents))
    ]


def model_spatial_shape_of_extents(
    extents: abc.Sequence[tuple[int, int]],
) -> list[int]:
    """The spatial shape of the extracted model region itself, without PML."""
    return [end - begin for begin, end in extents]


def _initial_wavefield_name(
    index: int,
    wavefield_names: Optional[abc.Sequence[str]],
) -> str:
    """The name to use for an initial wavefield in a message.

    The shared setup code is not told the propagator's spelling of each
    initial wavefield (``wavefield_0``, ``pressure_0``, ``vy_0``, ...), so
    callers that know it pass ``wavefield_names``; those that do not get a
    positional description instead. This mirrors ``_location_group_names``,
    which faces the same problem for source and receiver location groups.
    """
    if wavefield_names is not None and index < len(wavefield_names):
        return wavefield_names[index]
    return f"initial wavefield {index}"


def _check_wavefield_spatial_shape(
    wavefield: torch.Tensor,
    extents: abc.Sequence[tuple[int, int]],
    pml_width: abc.Sequence[int],
    index: int,
    wavefield_names: Optional[abc.Sequence[str]],
) -> bool:
    """Classify a provided initial wavefield's spatial shape.

    Two spatial shapes are accepted for an initial wavefield: the simulation
    domain (the extracted model region plus the PML), which is what a
    previous call returned, and the extracted model region itself, which is
    zero-padded internally. The padded shape is checked first, so a shape
    that is a valid padded shape keeps the meaning it has always had.

    Args:
        wavefield: The initial wavefield Tensor.
        extents: The resolved ``[beginning, end)`` extents, one per dimension.
        pml_width: The PML width on each face.
        index: The wavefield's position in the propagator's wavefield list.
        wavefield_names: Optional names for the wavefields, used in messages.

    Returns:
        True if the wavefield is model-shaped and must be zero-padded by
        ``pml_width`` per face; False if it already spans the simulation
        domain.

    Raises:
        DeepwaveValueError: If the shape is neither of the accepted shapes.

    """
    actual = list(wavefield.shape[1:])
    padded = padded_spatial_shape(extents, pml_width)
    if actual == padded:
        return False
    model = model_spatial_shape_of_extents(extents)
    if actual == model:
        return True
    name = _initial_wavefield_name(index, wavefield_names)
    # Name the *region* this call selected, not only the shapes it accepts:
    # the shapes alone say the wavefield is the wrong size, while the region
    # says which part of the model this call is about to propagate, which is
    # what tells the caller whether they meant to continue somewhere else.
    region = (
        f"This call propagates the region origin="
        f"{[extent[0] for extent in extents]}, extent="
        f"{model_spatial_shape_of_extents(extents)}. "
    )
    # ``initial_state=out`` is named first because it is the remedy that
    # needs no arithmetic: it reuses the region the previous call recorded,
    # so there is no shape for the caller to work out at all.
    remedy = (
        "or pass initial_state=out from the previous call (with no "
        "survey_pad or origin), which continues in the region that call "
        "used rather than the one selected here."
    )
    if padded == model:
        raise DeepwaveValueError(
            f"{name} has spatial shape {actual}. {region}For this call the "
            f"accepted shape is {model} (the extracted model region; there "
            f"is no PML on this call for Deepwave to add). Pass that, "
            f"{remedy}",
        )
    raise DeepwaveValueError(
        f"{name} has spatial shape {actual}. {region}For this call the "
        f"accepted shapes are {padded} (the simulation domain: the extracted "
        "region plus the PML, which is what a previous call returned) or "
        f"{model} (the extracted model region, which Deepwave will pad for "
        f"you). Pass one of those, {remedy}",
    )


def _extents_are_valid(
    extents: abc.Sequence[tuple[int, int]],
    model_spatial_shape: abc.Sequence[int],
    locations: Optional[abc.Sequence[Optional[torch.Tensor]]],
) -> bool:
    """Whether a candidate set of survey extents could be used for this call.

    The two conditions are the ones ``extract_survey`` applies to the extents
    it ends up with: the region must lie inside the model, and it must
    contain every source and receiver. Reusing those functions (rather than
    reimplementing their conditions) is what guarantees that "the padded
    interpretation is valid" means exactly "the call would have worked
    today".
    """
    try:
        check_extents_within_model(extents, model_spatial_shape)
        if locations is not None:
            check_locations_within_extents(extents, locations)
    except RuntimeError:
        return False
    return True


def get_survey_extents_from_wavefields(
    wavefields: abc.Sequence[Optional[torch.Tensor]],
    origin: Optional[abc.Sequence[int]],
    pml_width: abc.Sequence[int],
    *,
    model_spatial_shape: Optional[abc.Sequence[int]] = None,
    locations: Optional[abc.Sequence[Optional[torch.Tensor]]] = None,
) -> list[tuple[int, int]]:
    """Determine the extent of the model to extract from the wavefields.

    The spatial shape of a provided initial wavefield may be either the
    simulation domain (the extracted region plus the PML) or the extracted
    model region itself, which Deepwave zero-pads by ``pml_width`` per face.
    The first interpretation is preferred whenever it is valid, so a call
    that works today keeps the meaning it has today; the second is only
    reached by calls that would otherwise raise. When both interpretations
    are valid and select different regions the shape is genuinely ambiguous:
    the first is still used, and a ``UserWarning`` names both readings and
    the extents each would select. Providing ``origin`` does not remove that
    ambiguity, so the warning does not depend on it.

    Args:
        wavefields:
            A list of initial wavefields (potentially None).
        origin:
            A list containing the coordinates of the origin of the wavefields.
            Optional, default all zero.
        pml_width:
            A list containing the padding to add on each side for the PML,
            using the specified pad mode.
        model_spatial_shape:
            The spatial shape of the full model. Optional; when it is not
            provided neither interpretation can be validated, so the padded
            one is used unconditionally.
        locations:
            The source and receiver locations, used (with
            ``model_spatial_shape``) to decide whether an interpretation is
            valid. Optional.

    Returns:
        A list of length equal to the number of dimensions in the model,
        specifying the extents of the model, as a tuple [beginning, end)
        for each dimension, that will be used for wave propagation.

    """
    if not isinstance(pml_width, abc.Sequence):
        # This function has no declared exception-type contract and raises
        # RuntimeError uniformly for every validation failure below
        # (including the equally type-check-shaped "origin must be a list
        # of integers" a few lines down); singling out just this check for
        # TypeError would be inconsistent with the rest of the function and
        # would also break tests/test_extract.py's existing
        # `pytest.raises(RuntimeError)` expectation.
        raise DeepwaveRuntimeError("pml_width must be an abc.Sequence")
    ndims = len(pml_width) // 2
    if origin is not None:
        if len(origin) != ndims:
            raise DeepwaveRuntimeError(f"origin must be a list of length {ndims}.")
        if any(not isinstance(dim_origin, int) for dim_origin in origin):
            raise DeepwaveRuntimeError("origin must be a list of integers.")
        if any(dim_origin < 0 for dim_origin in origin):
            raise DeepwaveRuntimeError("origin coordinates must be non-negative.")
    if any(dim_pml_width < 0 for dim_pml_width in pml_width):
        raise DeepwaveRuntimeError("pml_width must be non-negative.")
    for wavefield in wavefields:
        if wavefield is not None:
            if wavefield.ndim != ndims + 1:
                raise DeepwaveRuntimeError(
                    f"wavefields must have {ndims + 1} dimensions (batch + spatial).",
                )
            spatial_shape = list(wavefield.shape[1:])
            dim_origins = [0 if origin is None else origin[dim] for dim in range(ndims)]
            # The two candidate readings of the provided shape.
            padded_extents: list[tuple[int, int]] = []
            negative_dim: Optional[int] = None
            for dim in range(ndims):
                extent_size = (
                    spatial_shape[dim] - pml_width[dim * 2] - pml_width[dim * 2 + 1]
                )
                if extent_size < 0:
                    negative_dim = dim
                    break
                padded_extents.append(
                    (dim_origins[dim], dim_origins[dim] + extent_size),
                )
            model_extents = [
                (dim_origins[dim], dim_origins[dim] + spatial_shape[dim])
                for dim in range(ndims)
            ]
            padded_valid = negative_dim is None and (
                model_spatial_shape is None
                or _extents_are_valid(padded_extents, model_spatial_shape, locations)
            )
            model_valid = model_spatial_shape is not None and _extents_are_valid(
                model_extents,
                model_spatial_shape,
                locations,
            )
            if padded_valid:
                extents = padded_extents
            elif model_valid:
                extents = model_extents
            elif negative_dim is not None:
                raise DeepwaveRuntimeError(
                    "A calculated survey extent was negative, which probably "
                    "means that a provided wavefield has the wrong shape. "
                    "Did you forget to add PML padding? The wavefield has "
                    f"length {spatial_shape[negative_dim]}, and the "
                    f"PML widths are {pml_width[negative_dim * 2]}, "
                    f"{pml_width[negative_dim * 2 + 1]} in the dimension."
                )
            else:
                # Neither reading works. Return the padded one so that the
                # caller's own checks report why, in the terms they always
                # have (extents outside the model, or locations outside the
                # extents), rather than replacing a precise diagnosis with a
                # vaguer one.
                extents = padded_extents
            # Both readings are usable and they select different regions, so
            # the shape alone cannot say which one was meant. Providing origin
            # does not settle it: origin says where the region begins, not
            # whether the shape includes the PML. The padded reading is still
            # the one used -- that is the meaning such a call has always had --
            # so this is a diagnostic, not a change of behaviour.
            ambiguous = padded_valid and model_valid and padded_extents != model_extents
            if origin is None or ambiguous:
                # Fallback frame chain: user call -> propagator() ->
                # setup_propagator() -> extract_survey() ->
                # get_survey_extents_from_wavefields().
                parts: list[str] = []
                if origin is None:
                    parts.append(
                        "Survey extents were inferred from the wavefield shape "
                        f"to be {extents} because origin was not provided; pass "
                        f"origin explicitly (origin={dim_origins}) if this is "
                        "not what you intended. Initial wavefields given as "
                        "bare tensors carry no record of the region they came "
                        "from, unlike a previous call's output, whose RunInfo "
                        "and auxiliary state both record it: if these came from "
                        "a call that used survey_pad or origin, pass that "
                        "call's output (initial_state=out) where you can, and "
                        "the region is reused with nothing to infer. Where you "
                        "cannot -- a checkpointed loop threads plain tensors, "
                        "so the output object does not survive it -- the "
                        f"explicit origin={dim_origins} above is the remedy.",
                    )
                if ambiguous:
                    whole_model = [
                        spatial_shape[dim] + pml_width[dim * 2] + pml_width[dim * 2 + 1]
                        for dim in range(ndims)
                    ]
                    parts.append(
                        f"The wavefield's spatial shape {spatial_shape} is "
                        "ambiguous: read as a simulation domain (the model "
                        "region plus the PML) it selects survey extents "
                        f"{padded_extents}, and read as the model region for "
                        "Deepwave to pad it selects survey extents "
                        f"{model_extents}. Both fit inside the model and "
                        "contain every source and receiver of this call, so "
                        "the shape alone cannot say which was meant.",
                    )
                    if model_spatial_shape is not None and spatial_shape == list(
                        model_spatial_shape,
                    ):
                        parts.append(
                            "The shape also matches the whole model unpadded.",
                        )
                    parts.append(
                        "The first (simulation domain) reading is used, because "
                        "it is the one this call has always had. Passing origin "
                        "does not remove this ambiguity, and neither reading is "
                        "silenced by it. If you meant the model region for "
                        "Deepwave to pad, pass a wavefield that includes the PML "
                        f"(spatial shape {whole_model}); to continue a previous "
                        "call without any shape ambiguity, pass "
                        "initial_state=out from it.",
                    )
                warn_at_user_call(" ".join(parts), UserWarning, 5)
            return extents
    raise DeepwaveRuntimeError("At least one wavefield must be non-None.")


def check_extents_within_model(
    extents: abc.Sequence[tuple[int, int]],
    model_shape: abc.Sequence[int],
) -> None:
    """Checks if the survey extents are within the bounds of the model.

    Args:
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        model_shape: A sequence of integers representing the spatial shape
            of the model.

    Raises:
        RuntimeError: If any survey extent is outside the model bounds.

    """
    for extent, model_dim_shape in zip(extents, model_shape):
        if extent[0] < 0 or model_dim_shape < extent[1]:
            raise DeepwaveRuntimeError(
                "Survey extents are larger than the model. "
                "This probably occurred because you provided "
                "an initial wavefield of the wrong shape.",
            )


def check_locations_within_extents(
    extents: abc.Sequence[tuple[int, int]],
    locations: abc.Sequence[Optional[torch.Tensor]],
) -> None:
    """Checks if all locations are within the specified survey extents.

    Args:
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        locations: A sequence of optional Tensors, where each torch.Tensor
            contains the locations to check.

    Raises:
        RuntimeError: If any location is outside the specified survey extents.

    """
    for location in locations:
        if location is not None:
            for dim in range(location.shape[-1]):
                dim_location = location[..., dim]
                dim_location = dim_location[dim_location != IGNORE_LOCATION]
                if dim_location.numel() == 0:
                    # Nothing left to validate: the tensor was empty, or every
                    # entry was the IGNORE_LOCATION sentinel.
                    continue
                if (
                    dim_location.min() < extents[dim][0]
                    or extents[dim][1] <= dim_location.max()
                ):
                    raise DeepwaveRuntimeError(
                        "Locations are not within "
                        "survey extents. This probably occurred "
                        "because you provided an "
                        "initial wavefield that does not cover all "
                        "source and receiver locations, given the "
                        "specified origin.",
                    )


def check_extents_match_wavefields_shape(
    extents: abc.Sequence[tuple[int, int]],
    wavefields: abc.Sequence[Optional[torch.Tensor]],
    pad: abc.Sequence[int],
    wavefield_names: Optional[abc.Sequence[str]] = None,
) -> None:
    """Checks if wavefield shapes match the calculated extents and padding.

    Either accepted spatial shape may be used: the simulation domain (the
    extents plus the padding) or the extracted model region itself, which
    will be zero-padded internally by ``prepare_wavefields``.

    Args:
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        wavefields: A sequence of optional Tensors representing initial
            wavefields.
        pad: A sequence of integers specifying the padding for finite
            difference stencils and PML in each dimension.
        wavefield_names: Optional names for the wavefields, used in messages.

    Raises:
        RuntimeError: If a provided wavefield has the wrong number of
            dimensions.
        DeepwaveValueError: If the spatial shape of any provided wavefield is
            neither of the accepted shapes.

    """
    ndims = len(extents)
    if not len(pad) == 2 * ndims:
        raise AssertionError("Expected len(pad) == 2 * ndims")
    for index, wavefield in enumerate(wavefields):
        if wavefield is not None:
            if not len(wavefield.shape) - 1 == ndims:
                raise DeepwaveRuntimeError(
                    f"Expected a wavefield with {ndims + 1} dimensions (batch "
                    f"+ spatial), but found one with {len(wavefield.shape)}.",
                )
            _check_wavefield_spatial_shape(
                wavefield,
                extents,
                pad,
                index,
                wavefield_names,
            )


def reverse_pad(pad: abc.Sequence[int]) -> list[int]:
    """Reverses the order of padding for each dimension.

    Given a sequence of padding values in the order
    [dim0_start, dim0_end, dim1_start, dim1_end, ...], this function returns
    a new list with the order of the dimensions reversed.

    Args:
        pad: A sequence of integers representing padding values for each
            dimension. Expected format:
            [dim0_start, dim0_end, dim1_start, dim1_end, ...]

    Returns:
        List[int]: A new list with the dimension order reversed. Format:
            [dim_N_start, dim_N_end, ..., dim1_start, dim1_end,
             dim0_start, dim0_end].

    """
    ndims = len(pad) // 2
    reversed_pad: list[int] = []
    for dim in range(ndims - 1, -1, -1):
        reversed_pad.extend([pad[2 * dim], pad[2 * dim + 1]])
    return reversed_pad


def replicate_pad(x: torch.Tensor, pad: abc.Sequence[int]) -> torch.Tensor:
    """Pads the trailing dimensions of ``x`` by repeating its edge values.

    This is ``torch.nn.functional.pad(x, pad, mode="replicate")``, computed
    instead as a concatenation of expanded edge slices.

    The two agree bitwise going forward, but not backward: the replicate
    pad's backward scatters the gradient of every padded element onto the
    edge element it was copied from, and on CUDA that accumulation is an
    ``atomicAdd``, so the order in which the padded elements are added
    varies between runs and the summed edge gradient is not reproducible.
    Concatenation's backward instead hands each part of the gradient to the
    tensor it came from, and the expansion's backward sums a whole axis with
    a fixed reduction order, so this spelling is bitwise reproducible.

    ``pad`` follows ``torch.nn.functional.pad``'s ordering: the last
    dimension's two widths first.
    """
    if any(width < 0 for width in pad):
        # Negative widths crop rather than replicate. The callers never ask
        # for one; defer to torch rather than reimplement cropping.
        return torch.nn.functional.pad(x, pad=list(pad), mode="replicate")
    padded = False
    for k in range(len(pad) // 2):
        dim = x.ndim - 1 - k
        low, high = pad[2 * k], pad[2 * k + 1]
        if low == 0 and high == 0:
            continue
        parts = []
        if low > 0:
            shape = list(x.shape)
            shape[dim] = low
            parts.append(x.narrow(dim, 0, 1).expand(shape))
        parts.append(x)
        if high > 0:
            shape = list(x.shape)
            shape[dim] = high
            parts.append(x.narrow(dim, x.shape[dim] - 1, 1).expand(shape))
        x = torch.cat(parts, dim)
        padded = True
    # A pad of zero width on every side still owes the caller its own
    # tensor: the extracted models are written to in place downstream, so
    # returning a view of the user's model would corrupt it.
    return x if padded else x.clone()


def extract_models(
    models: abc.Sequence[torch.Tensor],
    extents: abc.Sequence[tuple[int, int]],
    pad: abc.Sequence[int],
    pad_modes: abc.Sequence[str],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Extract the specified portion of the models and prepares them.

    Args:
        models:
            A list of Tensors to extract from.
        extents:
            A list of tuples specifying the portion of the model to extract.
        pad:
            An list of integers specifying the padding to add on each side
        pad_modes:
            A list of strings specifying the pad mode to use for the PML
            region of each model.
        n_batch:
            An integer specifying the size of the batch dimension.
        device:
            The device that the models should be located on.
        dtype:
            The datatype that the models should have.

    Returns:
        A list containing the desired portion of the models, with padding
        applied and a batch dimension of size either 1 or n_batch.

    """
    ndims = len(pad) // 2
    if not len(models) == len(pad_modes):
        raise AssertionError("Expected len(models) == len(pad_modes)")
    if not len(extents) == ndims:
        raise AssertionError("Expected len(extents) == ndims")

    # Check all models have correct dimensions and consistent attributes
    spatial_shape = models[0].shape[-ndims:]
    for model in models:
        if not isinstance(model, torch.Tensor):
            raise DeepwaveTypeError("Models must be a torch.Tensor.")
        if model.ndim not in (ndims, ndims + 1):
            raise DeepwaveRuntimeError(
                f"Models must have {ndims} or {ndims + 1} dimensions."
            )
        if model.device != device:
            raise DeepwaveRuntimeError(f"All models must be on device {device}.")
        if model.dtype != dtype:
            raise DeepwaveRuntimeError(f"All models must have dtype {dtype}.")
        if model.shape[-ndims:] != spatial_shape:
            raise DeepwaveRuntimeError("All models must have the same spatial shape.")

    models = [m.unsqueeze(0) if m.ndim == ndims else m for m in models]
    for m in models:
        if m.shape[0] not in (1, n_batch):
            raise DeepwaveRuntimeError(
                f"The batch dimension of models must be of size 1 or {n_batch}.",
            )

    region = (slice(None), *(slice(begin, end) for begin, end in extents))
    reversed_pad = reverse_pad(pad)

    if n_batch == 0:
        return [
            torch.empty(0, *model[region].shape[1:], device=device, dtype=dtype)
            for model in models
        ]
    return [
        (
            replicate_pad(model[region], reversed_pad)
            if pad_mode == "replicate"
            else torch.nn.functional.pad(
                model[region],
                pad=reversed_pad,
                mode=pad_mode,
            )
        ).contiguous()
        for model, pad_mode in zip(models, pad_modes)
    ]


def _validate_location_tensor(
    location: torch.Tensor,
    name: str,
    ndims: int,
    n_batch: int,
    eps: float,
) -> None:
    """Validate a single location tensor."""
    if not isinstance(location, torch.Tensor):
        raise DeepwaveTypeError("locations must be a torch.Tensor.")
    # This is diagnostics-only: whatever `location` contains, every
    # downstream consumer already does `location.long()` (see
    # `_convert_to_1d_and_check_uniqueness`), so skipping the check changes
    # nothing about which memory is touched -- it only means a fractional
    # location is silently truncated instead of raising this ValueError.
    # `eps < tensor.abs().max()` is a data-dependent read (forces a
    # Tensor.__bool__/item() to decide the branch), which graph-breaks
    # torch.compile and guards on data in torch.export, so it is skipped
    # while torch.compiler.is_compiling() to keep the traced graph static.
    if (
        not _torch_is_compiling()
        and location.numel() > 0
        and eps < (location - location.long()).abs().max()
    ):
        raise DeepwaveValueError(
            "Locations should be specified as integer numbers of cells "
            f"(within {eps:g} of an integer). If you wish to have a source "
            "or receiver that is not centred on a cell, please consider "
            "using the Hick's method, which is implemented in "
            "deepwave.Hicks."
        )
    if location.ndim != 3:
        raise DeepwaveRuntimeError(
            name + " location Tensors must have three dimensions",
        )
    if location.shape[0] != n_batch:
        raise DeepwaveRuntimeError(
            "Inconsistent batch size: Expected all Tensors to have a "
            f"batch size of {n_batch}, but found a {name.lower()} "
            "locations torch.Tensor with a batch size of "
            f"{location.shape[0]}.",
        )
    if location.shape[-1] != ndims:
        raise DeepwaveRuntimeError(
            f"{name} locations must have {ndims} dimensional "
            f"coordinates, but found one with {location.shape[-1]}.",
        )


def _calculate_origin_and_stride(
    extents: abc.Sequence[tuple[int, int]],
    pad: abc.Sequence[int],
    ndims: int,
) -> tuple[list[int], list[int]]:
    """Calculate origin and stride for extracted survey."""
    origin: list[int] = []
    shape: list[int] = []
    stride: list[int] = [1] * ndims
    for dim in range(ndims):
        origin.append(extents[dim][0] - pad[2 * dim])
        shape.append(
            extents[dim][1] - extents[dim][0] + pad[2 * dim] + pad[2 * dim + 1],
        )
    for dim in range(ndims - 2, -1, -1):
        stride[dim] = stride[dim + 1] * shape[dim + 1]
    return origin, stride


@torch.compiler.disable  # type: ignore[untyped-decorator]
def _check_unique_locations(location_1d: torch.Tensor, name: str, n_batch: int) -> None:
    """Reject active duplicate cells on every eager or compiled invocation."""
    for batch_idx in range(n_batch):
        shot_locations = location_1d[batch_idx]
        shot_locations = shot_locations[shot_locations != IGNORE_LOCATION]
        n_unique = len(torch.unique(shot_locations))
        if len(shot_locations) != n_unique:
            raise DeepwaveRuntimeError(
                f"{name} locations must be unique within each shot. "
                "You cannot have two in the same cell, but in shot "
                f"{batch_idx} there is/are {len(shot_locations)} "
                f"active {name.lower()} locations while only "
                f"{n_unique} is/are unique.",
            )


def _convert_to_1d_and_check_uniqueness(
    location: torch.Tensor,
    name: str,
    ndims: int,
    n_batch: int,
    origin: list[int],
    stride: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Convert locations to 1D and check for uniqueness."""
    shifted_location = location.clone().long().to(device)
    for dim in range(ndims):
        shifted_location[..., dim] = torch.where(
            shifted_location[..., dim] != IGNORE_LOCATION,
            shifted_location[..., dim] - origin[dim],
            IGNORE_LOCATION,
        )

    location_1d = torch.where(
        shifted_location[..., 0] != IGNORE_LOCATION,
        0,
        IGNORE_LOCATION,
    )
    for dim in range(ndims):
        location_1d += torch.where(
            shifted_location[..., dim] != IGNORE_LOCATION,
            shifted_location[..., dim] * stride[dim],
            0,
        )

    _check_unique_locations(location_1d, name, n_batch)
    return location_1d.contiguous()


def extract_locations(
    name: str,
    locations: abc.Sequence[Optional[torch.Tensor]],
    extents: abc.Sequence[tuple[int, int]],
    pad: abc.Sequence[int],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    eps: float = 1e-6,
) -> list[torch.Tensor]:
    """Sets locations relative to the extracted model and prepares them.

    Locations are returned as 1D indices into each batch.

    Args:
        name: A string indicating the type of locations
            (e.g., "Source", "Receiver").
        locations: A sequence of optional Tensors, where each torch.Tensor
            contains the locations to process.
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        pad: A sequence of integers specifying the padding for finite
            difference stencils and PML in each dimension.
        n_batch: The batch size.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.
            Defaults to `torch.long`.
        eps: A small float value used for checking if locations are
            integer-like. Defaults to 1e-6.

    Returns:
        List[torch.Tensor]: A list of processed location Tensors, where each
            location is converted to a 1D index relative to the extracted
            and padded model.

    Raises:
        TypeError: If `locations` contains elements that are not
            `torch.Tensor` or `None`.
        RuntimeError: If there are inconsistencies in the dimensions, batch
            size, or if locations are not unique within a shot.

    """
    ndims = len(extents)
    origin, stride = _calculate_origin_and_stride(extents, pad, ndims)

    extracted_locations: list[torch.Tensor] = []
    for location in locations:
        if location is not None:
            _validate_location_tensor(location, name, ndims, n_batch, eps)
            location_1d = _convert_to_1d_and_check_uniqueness(
                location, name, ndims, n_batch, origin, stride, device
            )
            extracted_locations.append(location_1d)
        else:
            extracted_locations.append(
                torch.empty(n_batch, 0, device=device, dtype=dtype),
            )
    return extracted_locations


def prepare_wavefields(
    wavefields: abc.Sequence[Optional[torch.Tensor]],
    extents: abc.Sequence[tuple[int, int]],
    pad: abc.Sequence[int],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype,
    wavefield_names: Optional[abc.Sequence[str]] = None,
) -> list[torch.Tensor]:
    """Checks and prepares initial wavefields.

    This function validates the input wavefields, ensures consistency
    in device, dtype, and shape, and replaces `None` wavefields with
    zero-filled tensors of the appropriate shape.

    A wavefield may be provided either at the shape of the simulation domain
    (the extents plus the padding), which is what a previous call returned,
    or at the shape of the extracted model region, in which case it is
    zero-padded here by `pad` on each face.

    Args:
        wavefields: A sequence of optional Tensors representing initial
            wavefields.
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        pad: A sequence of integers specifying the padding for finite
            difference stencils and PML in each dimension.
        n_batch: The batch size.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.
        wavefield_names: Optional names for the wavefields, used in messages.

    Returns:
        A list of processed wavefield Tensors, with `None` values
        replaced by zero tensors.

    Raises:
        TypeError: If `wavefields` contains elements that are not
            `torch.Tensor` or `None`.
        RuntimeError: If there are inconsistencies in the dimensions, device,
            dtype, or batch size of the wavefields.
        DeepwaveValueError: If the spatial shape of a wavefield is neither of
            the accepted shapes.

    """
    ndims = len(extents)
    prepared_wavefields: list[torch.Tensor] = []
    for index, wavefield in enumerate(wavefields):
        if wavefield is not None:
            if not isinstance(wavefield, torch.Tensor):
                raise DeepwaveTypeError("Wavefields must be a torch.Tensor.")
            if wavefield.device != device:
                raise DeepwaveRuntimeError(
                    "Inconsistent device: Expected all Tensors be on device "
                    f"{device}, but found a wavefield torch.Tensor on device "
                    f"{wavefield.device}.",
                )
            if wavefield.dtype != dtype:
                raise DeepwaveRuntimeError(
                    "Inconsistent dtype: Expected wavefields to have "
                    f"datatype {dtype}, but found one with dtype "
                    f"{wavefield.dtype}.",
                )
            if wavefield.ndim != ndims + 1:
                raise DeepwaveRuntimeError(
                    f"Wavefields must have {ndims + 1} dimensions, but found "
                    f"one with {wavefield.ndim}.",
                )
            if wavefield.shape[0] != n_batch:
                raise DeepwaveRuntimeError(
                    "Inconsistent batch size: Expected all Tensors to have a "
                    f"batch size of {n_batch}, but found a wavefield with a "
                    f"batch size of {wavefield.shape[0]}.",
                )
            if _check_wavefield_spatial_shape(
                wavefield,
                extents,
                pad,
                index,
                wavefield_names,
            ):
                # Model-shaped: zero-pad it to the simulation domain. This is
                # differentiable, so a wavefield that requires a gradient
                # still receives one, at the shape the user provided.
                wavefield = torch.nn.functional.pad(
                    wavefield,
                    reverse_pad(pad),
                )
            prepared_wavefields.append(wavefield)
        else:
            prepared_wavefields.append(torch.empty(0, device=device, dtype=dtype))
    return prepared_wavefields


# The conservative default Courant number used by cfl_condition_n when the
# caller does not supply one explicitly. Shared with setup_propagator (via
# RunInfo.courant_number) so the "resolved" value it reports for an
# unspecified courant_number matches cfl_condition_n's actual default without
# re-deriving it.
_DEFAULT_COURANT_NUMBER = 0.6

# Critical (maximum stable) Courant numbers C_crit for the leapfrog schemes,
# derived exactly from the finite-difference stencil coefficients in
# regular_grid.h / staggered_grid.h. Regular grid: C_crit = 2 / sqrt(S_M) with
# S_M = -(c0 + 2 sum_m c_m (-1)^m) the peak magnitude of the second-derivative
# symbol; staggered grid: C_crit = 2 / K_M with K_M = 2 sum_m |w_m| the peak
# modified-wavenumber magnitude of the first-derivative symbol. The full
# re-derivation is in the appendix of docs/discretisation.rst.
_CRITICAL_COURANT_REGULAR = {
    2: 1.0,  # 2/sqrt(4)
    4: 0.8660254037844386,  # 2/sqrt(16/3) = sqrt(3)/2
    6: 0.8134892168024669,  # 2/sqrt(272/45)
    8: 0.7843687748745656,  # 2/sqrt(2048/315)
}
_CRITICAL_COURANT_STAGGERED = {
    2: 1.0,  # 2/2
    4: 0.8571428571428571,  # 2/(7/3) = 6/7
    6: 0.8053691275167785,  # 2/(149/60) = 120/149
    8: 0.7774178620857286,  # 2/(2161/840)
}


def _critical_courant_number(accuracy: Accuracy, staggered: bool) -> float:
    """The maximum stable Courant number for a leapfrog scheme.

    The table lookup behind the public
    :func:`deepwave.discretisation.critical_courant_number`, kept separate
    from it so that Deepwave's own stability check can call it without
    tripping that function's ``staggered=`` deprecation warning on every
    propagator call.

    These are the critical Courant numbers C_crit re-derived from the actual
    finite-difference stencil coefficients (see the appendix of
    docs/discretisation.rst). A propagation is stable when
    ``dt * v_max * sqrt(sum_i 1/dx_i^2) <= C_crit``.

    Args:
        accuracy: The spatial finite-difference accuracy order (2, 4, 6, or 8).
        staggered: True for the staggered velocity-stress schemes (acoustic,
            elastic); False for the regular-grid schemes (scalar, scalar_born).

    Returns:
        The critical (maximum stable) Courant number.

    """
    table = _CRITICAL_COURANT_STAGGERED if staggered else _CRITICAL_COURANT_REGULAR
    if accuracy not in table:
        raise DeepwaveValueError(
            f"accuracy must be one of {sorted(table)}, got {accuracy}."
        )
    return table[accuracy]


def cfl_condition_n(
    grid_spacing: Union[float, abc.Sequence[float]],
    dt: float,
    max_vel: float,
    eps: float = 1e-15,
    courant_number: Optional[float] = None,
) -> tuple[float, int]:
    """Calculates the time step interval to obey the CFL condition.

    The output time step will be a factor of the input time step.

    We use the maximum dt to calculate the integer factor
    (step_ratio) we need to divide the provided dt by to ensure it is
    less than or equal to this value.

    Units are the caller's own: any length unit and any time unit will do,
    as long as `grid_spacing`, `dt` and `max_vel` are expressed in the same
    pair. `step_ratio` and the Courant number are dimensionless, and
    `inner_dt` comes back in the unit `dt` was given in.

    Args:
        grid_spacing: The grid spacing in each spatial dimension, in the
            length unit `max_vel` is expressed in. Can be a
            single float (applied as the only entry) or a sequence of floats.
        dt: The time step interval, in the time unit `max_vel` is expressed
            in.
        max_vel: The maximum absolute wavespeed in the model, in
            `grid_spacing` units per `dt` unit.
        eps: Retained for call compatibility. Zero speed is handled explicitly,
            so this parameter no longer regularises the speed or changes the
            selected time step.
        courant_number: An optional Courant number to use instead of the
            conservative default (0.6). If None (the default), 0.6 is used.
            If given, it is
            the actual dimensionless Courant number the time step will target;
            a larger value gives a larger ``inner_dt`` (fewer internal steps).
            This function does not itself enforce an upper bound: a caller
            that needs to guarantee stability should compare it against
            :func:`deepwave.discretisation.critical_courant_number` first
            and raise if it is exceeded, which is what the propagators do
            while setting up a call.

    Returns:
        Tuple:

            - inner_dt: A time step interval that obeys the CFL
              condition, in the same time unit as `dt`.
            - step_ratio: The integer dt / inner_dt, dimensionless.

    """
    try:
        processed_grid_spacing = [_float_value(grid_spacing)]
    except (TypeError, ValueError):
        try:
            processed_grid_spacing = [
                _float_value(spacing) for spacing in _iterable_values(grid_spacing)
            ]
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError(
                "grid_spacing must be a float or sequence of floats."
            ) from e
    if not processed_grid_spacing:
        raise DeepwaveValueError("grid_spacing must not be empty.")
    for g in processed_grid_spacing:
        if not math.isfinite(g):
            raise DeepwaveValueError("grid_spacing elements must be finite.")
        if g <= 0:
            raise DeepwaveValueError("grid_spacing elements must be positive.")
    try:
        dt = float(dt)
    except (TypeError, ValueError) as e:
        raise DeepwaveTypeError("dt must be a float.") from e
    if not math.isfinite(dt):
        raise DeepwaveValueError("dt must be finite.")
    try:
        max_vel = float(max_vel)
    except (TypeError, ValueError) as e:
        raise DeepwaveTypeError("max_vel must be a float.") from e
    if not math.isfinite(max_vel) or max_vel < 0:
        raise DeepwaveRuntimeError("max_vel must be finite and non-negative.")

    if courant_number is not None:
        try:
            courant_number = float(courant_number)
        except (TypeError, ValueError) as e:
            raise DeepwaveTypeError("courant_number must be a float.") from e
        if not math.isfinite(courant_number):
            raise DeepwaveValueError("courant_number must be finite.")
        if courant_number <= 0:
            raise DeepwaveValueError("courant_number must be greater than zero.")
    c_max = _DEFAULT_COURANT_NUMBER if courant_number is None else courant_number
    if dt == 0 or max_vel == 0:
        # No substeps are needed when there is no temporal or spatial advance.
        return dt, 1

    # Compute |dt| * max_vel * ||1 / dx|| / c_max without squaring a
    # dimensioned quantity. Scaling the spatial norm and separating binary
    # exponents avoids intermediate overflow/underflow when units change.
    smallest_spacing = min(processed_grid_spacing)
    spatial_norm = math.hypot(*(smallest_spacing / g for g in processed_grid_spacing))
    dt_mantissa, dt_exponent = math.frexp(abs(dt))
    speed_mantissa, speed_exponent = math.frexp(max_vel)
    spacing_mantissa, spacing_exponent = math.frexp(smallest_spacing)
    courant_mantissa, courant_exponent = math.frexp(c_max)
    mantissa = (
        dt_mantissa
        * speed_mantissa
        * spatial_norm
        / spacing_mantissa
        / courant_mantissa
    )
    exponent = dt_exponent + speed_exponent - spacing_exponent - courant_exponent
    try:
        ratio = math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise DeepwaveValueError("The required CFL step ratio is too large.") from error
    step_ratio = max(1, math.ceil(ratio))
    inner_dt = dt / step_ratio
    if inner_dt == 0:
        raise DeepwaveValueError("The CFL-stable time step is too small to represent.")
    # Fallback frame chain (both warnings below): user call -> propagator()
    # -> setup_propagator() -> cfl_condition_n().
    if step_ratio >= 20:
        _warn_once_per_cause(
            "cfl_step_ratio_extreme",
            f"dt~{_fmt_3sf(dt)} greatly exceeds the CFL-stable internal time step, so "
            f"Deepwave is taking step_ratio={step_ratio} internal steps per "
            "output sample. Substantially reduce dt, increase courant_number, "
            "or check max_vel/grid_spacing for a units mistake.",
            UserWarning,
            stacklevel=4,
        )
    elif step_ratio >= 8:
        _warn_once_per_cause(
            "cfl_step_ratio_mild",
            f"dt~{_fmt_3sf(dt)} exceeds the stable internal time step; Deepwave is "
            f"taking step_ratio={step_ratio} internal steps per output "
            f"sample (~{step_ratio}x compute). Reduce dt, or see "
            "courant_number/max_vel.",
            UserWarning,
            stacklevel=4,
        )
    return inner_dt, step_ratio


def cfl_condition(dy: float, dx: float, *args: Any, **kwargs: Any) -> tuple[float, int]:
    """Calculates the time step interval for 2D models.

    This is a convenience wrapper around `cfl_condition_n` for 2D models,
    and takes its units: any length unit and any time unit, as long as the
    grid spacings, `dt` and `max_vel` share the same pair.

    Args:
        dy: The grid spacing in the y-dimension, in the length unit
            `max_vel` is expressed in.
        dx: The grid spacing in the x-dimension, in the same unit as `dy`.
        *args: Additional positional arguments to pass to `cfl_condition_n`.
        **kwargs: Additional keyword arguments to pass to `cfl_condition_n`.

    Returns:
        Tuple:

            - inner_dt: A time step interval that obeys the CFL
              condition, in the same time unit as `dt`.
            - step_ratio: The integer dt / inner_dt, dimensionless.

    """
    return cfl_condition_n([dy, dx], *args, **kwargs)


def reciprocal_vacuum_safe(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Reciprocal of ``x`` that maps an exactly zero entry to exactly zero.

    ``1 / (x**2 + eps) * x`` rather than ``1 / x``, so that the vacuum
    sentinel (a density, or equivalently a buoyancy, of exactly ``0``) is
    carried through as exactly ``0`` instead of becoming ``inf``/``nan``, and
    so that the expression stays differentiable there. This is the single
    definition of the density/buoyancy inversion used throughout Deepwave.

    Args:
        x: The tensor to invert (a density or a buoyancy).
        eps: An optional float to avoid division by zero. Default 1e-15.

    Returns:
        A torch.Tensor containing the guarded reciprocal of ``x``.

    """
    return 1 / (x**2 + eps) * x


def vpvsrho_to_lambmubuoyancy(
    vp: torch.Tensor,
    vs: torch.Tensor,
    rho: torch.Tensor,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts vp, vs, rho to lambda, mu, buoyancy.

    All input Tensors must have the same shape.

    Args:
        vp: A torch.Tensor containing the p wavespeed.
        vs: A torch.Tensor containing the s wavespeed.
        rho: A torch.Tensor containing the density.
        eps: An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple:

            - lambda: A torch.Tensor containing the first Lamé parameter.
            - mu: A torch.Tensor containing the second Lamé parameter.
            - buoyancy: The buoyancy (1/density) model. A cell with
              ``buoyancy = 0`` is a vacuum, not an infinitely dense cell;
              see :ref:`the vacuum sentinel <vacuum-sentinel>` in
              :doc:`/boundaries`, which also explains how the sentinel is
              averaged onto the staggered nodes. ``rho = 0`` is mapped to
              ``buoyancy = 0`` rather than to infinity, so a vacuum built
              this way carries the sentinel.

    """
    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = reciprocal_vacuum_safe(rho, eps)
    return lamb, mu, buoyancy


def lambmubuoyancy_to_vpvsrho(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts lambda, mu, buoyancy to vp, vs, rho.

    All input Tensors must have the same shape.

    Args:
        lamb: A torch.Tensor containing the first Lamé parameter.
        mu: A torch.Tensor containing the second Lamé parameter.
        buoyancy: The buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. Such a cell is mapped back to
            ``rho = 0`` rather than to an infinite density.
        eps: An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple:

            - vp: A torch.Tensor containing the p wavespeed.
            - vs: A torch.Tensor containing the s wavespeed.
            - rho: A torch.Tensor containing the density.

    """
    vs = (mu * buoyancy).sqrt()
    vp = (lamb * buoyancy + 2 * vs**2).sqrt()
    rho = reciprocal_vacuum_safe(buoyancy, eps)
    return vp, vs, rho


def to_cells(
    coordinates: torch.Tensor,
    grid_spacing: Union[float, abc.Sequence[float]],
    grid_origin: Optional[Union[float, abc.Sequence[float]]] = None,
    *,
    snap: Union[bool, float] = False,
) -> torch.Tensor:
    """Converts physical-unit coordinates to float grid-cell coordinates.

    Computes ``(coordinates - grid_origin) / grid_spacing`` elementwise, with
    ``grid_spacing``/``grid_origin`` broadcasting over the last (``ndim``)
    dimension of ``coordinates``. The result is suitable as float
    ``source_locations``/``receiver_locations`` for the propagators, which
    auto-detect off-grid-node coordinates and Hicks-interpolate them (see
    :class:`deepwave.Hicks`); on-node results are used
    directly by the exact (integer) path.

    With the default ``snap=False`` this function does **not** round: an
    off-node result is returned as-is, and even a physically on-node
    coordinate may not come back as an exact integer due to floating-point
    error. Snapping to the nearest integer grid cell is the propagators'
    documented behaviour, not this function's; call ``.round().long()``
    yourself first if you specifically want integer cell indices, or pass
    ``snap=True`` to clean up the floating-point error only (below).

    Args:
        coordinates: A ``[..., ndim]`` float Tensor of physical-unit
            coordinates.
        grid_spacing: The grid cell spacing: a single float (applied to
            every dimension) or a sequence of ``ndim`` floats.
        grid_origin: The physical-unit coordinate of grid cell 0 in each
            dimension: a single float, a sequence of ``ndim`` floats, or
            ``None`` (default), meaning zero in every dimension. Named for
            the grid, not for the propagators' ``origin``, which is the
            cell index at which a model region begins.
        snap: Whether to remove floating-point error from results that
            should have landed exactly on a node. ``False`` (the default)
            returns the division's result untouched, bit-for-bit.
            ``True`` replaces any result within
            :data:`deepwave.ON_NODE_TOLERANCE` cells of an integer by that
            integer. A float instead of ``True`` uses that tolerance, in
            cells, in place of ``ON_NODE_TOLERANCE``.

            This is a **cleanup of floating-point error, not a rounding of
            genuinely off-node coordinates**: a coordinate that is really
            0.4 cells off a node is returned unchanged, whatever ``snap``
            is, and will still be Hicks-interpolated. Its purpose is the
            case where ``(coordinates - grid_origin) / grid_spacing`` is
            arithmetically an integer but comes back as ``6.999999999999999``
            -- which, without snapping, routes an exactly-on-node survey
            through the interpolated path and changes the numbers. Keyword-only,
            because the other three parameters are positional in released code.

    Returns:
        A ``[..., ndim]`` float Tensor of cell coordinates, the same shape,
        dtype and device as the unsnapped result. The dtype is
        ``coordinates``' own if that is a floating type, and the default
        float dtype otherwise (an integer ``coordinates`` tensor is
        accepted: the conversion is done in floating point, so a fractional
        ``grid_spacing``/``origin`` is not truncated).

    Raises:
        DeepwaveValueError: If ``snap`` is a negative number.

    """
    ndim = coordinates.shape[-1]
    # ``grid_spacing`` and ``grid_origin`` must NOT be cast to
    # ``coordinates``' dtype: an integer ``coordinates`` tensor is a natural
    # way to write survey geometry in whole metres
    # (``torch.tensor([[100, 250], ...])``), and casting to it truncated the
    # spacing (4.5 -> 4) and the origin (2.5 -> 2). The subsequent division
    # promotes to float, so the result looked perfectly reasonable while being
    # up to 12% wrong. Work in a floating dtype throughout instead, which
    # leaves every float-input case bit-identical.
    out_dtype = (
        coordinates.dtype
        if coordinates.is_floating_point()
        else torch.get_default_dtype()
    )
    coordinates = coordinates.to(out_dtype)
    spacing = torch.as_tensor(
        set_grid_spacing(grid_spacing, ndim),
        dtype=out_dtype,
        device=coordinates.device,
    )
    if grid_origin is None:
        cells = coordinates / spacing
    else:
        origin_seq = (
            [float(grid_origin)] * ndim
            if isinstance(grid_origin, (int, float))
            else [float(o) for o in grid_origin]
        )
        if len(origin_seq) != ndim:
            raise DeepwaveRuntimeError(
                f"grid_origin must have {ndim} elements, got {len(origin_seq)}."
            )
        origin_t = torch.as_tensor(
            origin_seq, dtype=out_dtype, device=coordinates.device
        )
        cells = (coordinates - origin_t) / spacing
    return _snap_to_nodes(cells, snap)


def _snap_to_nodes(cells: torch.Tensor, snap: Union[bool, float]) -> torch.Tensor:
    """Applies :func:`to_cells`' ``snap`` argument. ``False`` is a no-op.

    Separated from :func:`to_cells` so that the ``snap=False`` path is
    visibly a straight ``return`` of the division's result: the default must
    stay bit-identical to the behaviour before ``snap`` existed.
    """
    if snap is False:
        return cells
    tolerance = ON_NODE_TOLERANCE if snap is True else float(snap)
    if tolerance < 0:
        raise DeepwaveValueError(
            f"to_cells' snap must be True, False, or a non-negative tolerance "
            f"in cells, but got {snap!r}."
        )
    nearest = cells.round()
    return torch.where((cells - nearest).abs() <= tolerance, nearest, cells)


def product_ratio_parts(a: float, b: float, divisor: float) -> tuple[float, int]:
    """Represent a*b/divisor without overflowing or losing tiny intermediates."""
    a_mantissa, a_exponent = math.frexp(a)
    b_mantissa, b_exponent = math.frexp(b)
    d_mantissa, d_exponent = math.frexp(divisor)
    mantissa, exponent = math.frexp(a_mantissa * b_mantissa / d_mantissa)
    return mantissa, a_exponent + b_exponent - d_exponent + exponent


def setup_pml(
    pml_width: abc.Sequence[int],
    pml_start: abc.Sequence[float],
    grid_spacing: float,
    dt: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float = 0.0,
    r_val: float = 0.001,
    n_power: int = 2,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates a and b profiles for C-PML.

    Only the first fd_pad[0]+pml_width[0] and last fd_pad[1]+pml_width[1]
    elements of the profiles will be non-zero.

    Args:
        pml_width: List of two integers specifying the width of the PML
            region.
        pml_start: List of two floats specifying the coordinates (in grid
            cells) of the start of the PML regions.
        grid_spacing: Cell size along this axis, in distance units. Each face
            is normalised by its own physical PML width.
        dt: Time step interval.
        n: Integer specifying desired profile length, including fd_pad and
           pml_width.
        max_vel: Maximum wave speed.
        dtype: PyTorch datatype to use.
        device: PyTorch device to use.
        pml_freq: The frequency value to use for the profile, usually the
            dominant frequency in the wavefield.
        start: Float specifying the coordinate (in grid cells) of the first
            element. Optional, default 0.
        r_val: The target reflection coefficient, in (0, 1]. Default 0.001.
        n_power: The non-negative polynomial degree. Default 2.
        eps: Retained for call compatibility. An exactly zero denominator is
            handled explicitly, without imposing a unit-dependent damping floor.

    Returns:
        A tuple containing the a and b profiles as Tensors.

    """
    if not math.isfinite(grid_spacing) or grid_spacing <= 0:
        raise DeepwaveValueError("PML grid_spacing must be finite and positive.")
    if not math.isfinite(dt):
        raise DeepwaveValueError("PML dt must be finite.")
    if not all(math.isfinite(value) and value >= 0 for value in (max_vel, pml_freq)):
        raise DeepwaveValueError(
            "PML max_vel and pml_freq must be finite and non-negative."
        )
    if not 0 < r_val <= 1 or n_power < 0:
        raise DeepwaveValueError("PML requires 0 < r_val <= 1 and n_power >= 0.")

    # Scale rates by time before putting them in tensors. Separately storing a
    # tiny rate and huge dt can produce 0*inf even when their product is ordinary.
    velocity_scale = product_ratio_parts(max_vel, abs(dt), grid_spacing)
    frequency_scale = product_ratio_parts(pml_freq, abs(dt), 1.0)
    scales = (velocity_scale, frequency_scale)
    # Geometry contributes at most this factor; leave room for both faces and
    # the frequency shift. Normal profiles retain the requested dtype.
    bound = max(1.0, math.pi, abs((1 + n_power) * math.log(r_val)))
    finfo = torch.finfo(dtype)
    # Bound the smallest nonzero polynomial sample on this unit-spaced grid.
    # Account for fractional interface positions before taking powers, so a
    # large rate cannot hide underflow in the geometry template itself.
    profile_span = 0.0
    for width, interface in zip(pml_width, pml_start):
        if width:
            gap = abs(math.remainder(interface - start, 1.0)) or 1.0
            width_log2 = math.log2(width)
            profile_span = max(
                profile_span,
                width_log2 + n_power * max(0.0, width_log2 - math.log2(gap)),
            )
    if r_val < 1:
        profile_span += max(0.0, -math.log2(-math.log(r_val)))
    low = math.frexp(finfo.tiny)[1] + 4 + math.ceil(profile_span)
    high = math.frexp(finfo.max)[1] - 4 - math.ceil(math.log2(bound))
    logarithmic = any(m != 0 and not low <= e <= high for m, e in scales)
    # Wider arithmetic reduces log/exp round-trip error; it does not hold the
    # extreme rates themselves. Their representation remains logarithmic.
    work_dtype = torch.float64 if logarithmic else dtype
    sigma_shape, alpha_shape, active = pml_sigma_alpha(
        pml_width,
        pml_start,
        1.0,
        n,
        1.0,
        work_dtype,
        device,
        1.0,
        start,
        r_val,
        n_power,
        logarithmic=logarithmic,
    )
    if not logarithmic:
        sigma = sigma_shape * math.ldexp(*velocity_scale)
        alpha = alpha_shape * math.ldexp(*frequency_scale)
        total = sigma + alpha
        denominator = torch.where(total == 0, torch.ones_like(total), total)
        ratio = (sigma / denominator).clamp(0, 1)
    else:
        # Only exceptional scales need logs. Form the ratio before materialising
        # a possibly infinite total damping, and handle zero rates before the
        # log difference so that an inactive face cannot produce -inf - -inf.
        velocity_log, frequency_log = (
            math.log(m) + e * math.log(2) if m else -math.inf for m, e in scales
        )
        sigma_log = sigma_shape + velocity_log
        alpha_log = alpha_shape + frequency_log
        total_log = torch.logaddexp(sigma_log, alpha_log)
        safe_total_log = torch.where(
            torch.isneginf(total_log), torch.zeros_like(total_log), total_log
        )
        ratio = (sigma_log - safe_total_log).exp().clamp(0, 1)
        total = total_log.exp()
    a = torch.exp(-total)
    # expm1 preserves damping when exp(-total) rounds to one. Both clamps state
    # mathematical bounds for non-negative rates; they also let Torch 2.4's
    # compiler avoid spurious 0*inf in its interval analysis of this product.
    b = ratio * torch.expm1(-total).clamp(-1, 0)
    a[~active] = 0
    return a.to(dtype), b.to(dtype)


def setup_pml_sigma_alpha(
    pml_width: abc.Sequence[int],
    pml_start: abc.Sequence[float],
    grid_spacing: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float = 0.0,
    r_val: float = 0.001,
    n_power: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw damping and frequency-shift profiles for multiaxial PML.

    Uses the same per-face law as :func:`setup_pml`; see its argument
    descriptions. Multiaxial PML combines these profiles across axes before
    applying the a/b transform.
    """
    sigma, alpha, _ = pml_sigma_alpha(
        pml_width,
        pml_start,
        grid_spacing,
        n,
        max_vel,
        dtype,
        device,
        pml_freq,
        start,
        r_val,
        n_power,
    )
    return sigma, alpha


def pml_sigma_alpha(
    pml_width: abc.Sequence[int],
    pml_start: abc.Sequence[float],
    grid_spacing: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float,
    r_val: float,
    n_power: int,
    *,
    logarithmic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Polynomial C-PML profiles with a local physical width for each face."""
    x = torch.arange(start, start + n, device=device, dtype=dtype)
    zero = torch.zeros_like(x)
    absent = torch.full_like(x, -math.inf) if logarithmic else zero
    if not any(pml_width):
        return absent, absent.clone(), torch.zeros_like(x, dtype=torch.bool)
    fractions = []
    damping = []
    for side, width in enumerate(pml_width):
        if width == 0:
            fractions.append(zero)
            damping.append(absent)
            continue
        distance = pml_start[side] - x if side == 0 else x - pml_start[side]
        fraction = (distance / width).clamp(min=0, max=1)
        fractions.append(fraction)
        if logarithmic:
            # Take logs before the power or division can underflow. Degree zero
            # is constant only inside its own face, including in log form.
            log_fraction = distance.clamp(min=0, max=width).log() - math.log(width)
            log_profile = (
                torch.where(distance > 0, zero, absent)
                if n_power == 0
                else n_power * log_fraction
            )
            log_sigma0 = (
                math.log1p(n_power)
                + math.log(max_vel)
                + math.log(-math.log(r_val))
                - math.log(2)
                - math.log(width)
                - math.log(grid_spacing)
                if max_vel > 0 and r_val < 1
                else -math.inf
            )
            damping.append(log_sigma0 + log_profile)
        else:
            physical_width = width * grid_spacing
            sigma0 = -(1 + n_power) * max_vel * math.log(r_val) / (2 * physical_width)
            # Avoid 0**0 adding degree-zero damping outside this face.
            profile = (fraction != 0).to(dtype) if n_power == 0 else fraction**n_power
            damping.append(sigma0 * profile)
    fraction = torch.maximum(fractions[0], fractions[1])
    if logarithmic:
        log_alpha0 = math.log(math.pi) + math.log(pml_freq) if pml_freq else -math.inf
        return (
            torch.logaddexp(damping[0], damping[1]),
            log_alpha0 + torch.log1p(-fraction),
            fraction != 0,
        )
    return (
        damping[0] + damping[1],
        math.pi * pml_freq * (1 - fraction),
        fraction != 0,
    )


def zero_interior(
    tensor: torch.Tensor,
    fd_pad: Union[int, abc.Sequence[int]],
    pml_width: abc.Sequence[int],
    dim: int,
) -> torch.Tensor:
    """Zeros out the interior region of a tensor.

    Args:
        tensor: The input torch.Tensor.
        fd_pad: The finite-difference padding. Can be an integer or a sequence
            of integers (e.g., [top, bottom, left, right] for 2D).
        pml_width: The width of the PML regions for each side
            (e.g., [top, bottom, left, right] for 2D).
        dim: The spatial dimension to zero.

    Returns:
        torch.Tensor: A new torch.Tensor with the interior region zeroed out.

    """
    shape = tensor.shape[1:]
    ndim = len(shape)
    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * 2 * ndim
    tensor[
        (slice(None),)
        + (slice(None),) * dim
        + (
            slice(
                fd_pad[dim * 2] + pml_width[dim * 2],
                shape[dim] - pml_width[dim * 2 + 1] - fd_pad[dim * 2 + 1],
            ),
        )
    ].fill_(0)
    return tensor


def zero_interior_box(
    tensor: torch.Tensor,
    fd_pad: Union[int, abc.Sequence[int]],
    pml_width: abc.Sequence[int],
) -> torch.Tensor:
    """Zeros the pure-interior box (interior along *every* spatial dim).

    Used by the multiaxial-PML (M-PML) elastic path: unlike the standard
    per-axis :func:`zero_interior` slabs, an M-PML memory variable is
    legitimately nonzero anywhere any axis is inside its PML (the cross-axis
    damping), so only the intersection of all axes' interiors may be zeroed.

    Args:
        tensor: The input torch.Tensor (batch dim first).
        fd_pad: The finite-difference padding (int or per-side sequence).
        pml_width: The width of the PML regions for each side.

    Returns:
        torch.Tensor: The tensor with the pure-interior box zeroed (in-place).

    """
    shape = tensor.shape[1:]
    ndim = len(shape)
    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * 2 * ndim
    slices: list[slice] = [slice(None)]
    for dim in range(ndim):
        start = fd_pad[dim * 2] + pml_width[dim * 2]
        end = shape[dim] - pml_width[dim * 2 + 1] - fd_pad[dim * 2 + 1]
        if end <= start:
            return tensor
        slices.append(slice(start, end))
    tensor[tuple(slices)].fill_(0)
    return tensor


# NOTE: ``_backend_function`` intentionally stays here (rather than moving to
# deepwave._engine with the rest of the engine block) because
# tests/test_abi_drift.py monkeypatches ``common._backend_function`` and relies
# on deepwave._engine.run_forward_engine/run_backward_core (which import it
# locally, per-call) resolving the lookup through this module's live
# namespace. See deepwave/_engine.py's module docstring.
def _backend_function(
    name: str,
    ndim: int,
    direction: str,
    accuracy: int,
    dtype: torch.dtype,
    device: Any,
    extra: str = "",
) -> Any:
    """Looks up a compiled backend function (thin wrapper for the engine).

    ``extra`` selects a variant symbol (e.g. ``"_mpml"`` for the elastic
    multiaxial-PML kernels, or ``"_sc"`` for scalar_born); it is appended after
    the direction in the C symbol name.
    """
    from deepwave import _backend_utils

    return _backend_utils.get_backend_function(
        name, ndim, direction, accuracy, dtype, device, extra=extra
    )


def match_model_batch(models: abc.Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Expands prepared models to their common leading (shot) dimension.

    Each model tensor decides independently whether it carries a shot
    dimension, so a batched background model may meet an unbatched
    perturbation, or the reverse. Operations that pair models
    element-by-element and demand *exactly* equal shapes -- notably
    ``torch.func.jvp``, whose tangent must match its primal -- cannot use
    the ordinary broadcasting that the kernels rely on, so the leading
    dimension is equalised here first.

    This changes no value: repeating a size-1 shot dimension is precisely
    what an unbatched model means, and autograd sums the gradient back over
    that dimension. When every model already agrees the tensors are returned
    untouched, so nothing is copied on the ordinary path. The expansion is
    materialised (rather than left as a stride-0 view) because the callers
    hand the result to code that writes into it in place.

    Args:
        models: Prepared model tensors, each of shape ``[batch,
            *model_shape]`` with ``batch`` either ``1`` or the shot count.

    Returns:
        The same tensors, each expanded to the largest leading dimension.

    """
    batch = max(model.shape[0] for model in models)
    return [
        model
        if model.shape[0] == batch
        else model.expand(batch, *model.shape[1:]).contiguous()
        for model in models
    ]


def get_ndim(
    models: abc.Sequence[torch.Tensor],
    wavefields: abc.Sequence[Optional[torch.Tensor]],
    locations: abc.Sequence[Optional[LocationsLike]],
    z_fields: abc.Sequence[Optional[torch.Tensor]],
    y_fields: abc.Sequence[Optional[torch.Tensor]],
    x_fields: abc.Sequence[Optional[torch.Tensor]],
) -> int:
    """Get the number of spatial dimensions from models, wavefields, or locations.

    Args:
        models: A sequence of model tensors.
        wavefields: A sequence of wavefield tensors.
        locations: A sequence of location tensors.
        z_fields: A sequence of z-dimension field tensors.
        y_fields: A sequence of y-dimension field tensors.
        x_fields: A sequence of x-dimension field tensors.

    Returns:
        The number of spatial dimensions.

    Raises:
        TypeError: If any field is not a torch.Tensor.
        ValueError: If there is an inconsistency in the number of dimensions.
        RuntimeError: If the number of dimensions cannot be determined.
    """
    ndim = -1
    for field in wavefields:
        if field is not None:
            if not isinstance(field, torch.Tensor):
                raise DeepwaveTypeError(
                    "Wavefields must be of type torch.Tensor, "
                    f"but found {type(field).__name__}"
                )
            if ndim < 0:
                ndim = field.ndim - 1
            elif ndim != field.ndim - 1:
                raise DeepwaveValueError(
                    "There is an inconsistency in the number of "
                    f"wavefield dimensions: Expected {ndim + 1}, "
                    f"but found {field.ndim}."
                )
    for location in locations:
        if location is not None:
            if not isinstance(location, torch.Tensor):
                # A Hicks interpolator (passed as a location argument for
                # dipoles / custom offsets / free surfaces) encodes ndim in its
                # interpolated locations; use those for the ndim determination.
                get_locations = getattr(location, "get_locations", None)
                if callable(get_locations):
                    location = get_locations()
                else:
                    raise DeepwaveTypeError(
                        "Locations must be of type torch.Tensor, "
                        f"but found {type(location).__name__}"
                    )
            if ndim < 0:
                ndim = location.shape[-1]
            elif ndim != location.shape[-1]:
                raise DeepwaveValueError(
                    "The number of spatial coordinates in a location "
                    "Tensor does not match expectations: "
                    f"Expected {ndim}, but found {location.shape[-1]}."
                )
    dim_fields = [x_fields, y_fields, z_fields]
    dim_names = ["x", "y", "z"]
    for dim in range(3):
        for field in dim_fields[dim]:
            if field is not None:
                if not isinstance(field, torch.Tensor):
                    raise DeepwaveTypeError(
                        "Wavefields must be of type torch.Tensor, "
                        f"but found {type(field).__name__}"
                    )
                if ndim < 0:
                    ndim = field.ndim - 1
                elif ndim != field.ndim - 1:
                    raise DeepwaveValueError(
                        "There is an inconsistency in the number of "
                        f"wavefield dimensions: Expected {ndim + 1}, "
                        f"but found {field.ndim}."
                    )
                elif dim + 1 > ndim:
                    raise DeepwaveValueError(
                        "The propagation was determined to be "
                        f"{ndim}d, so fields related to the "
                        f"{dim_names[dim]} dimension should not be "
                        "provided."
                    )
    # Before the model check below, not after it: with nothing to infer the
    # dimensionality from, ``ndim`` is still -1, and the model check would
    # otherwise report the propagation as "-1d" and blame the models, which
    # are not the problem.
    if ndim < 0:
        raise DeepwaveRuntimeError(
            "The number of spatial dimensions could not be "
            "determined. Please provide at least one source or "
            "initial wavefield. These are also what set the number of "
            "shots, so a call with neither has no shot count either."
        )
    for model in models:
        if not isinstance(model, torch.Tensor):
            raise DeepwaveTypeError(
                f"Models must be of type torch.Tensor, but found {type(model).__name__}"
            )
        if model.ndim != ndim and model.ndim != ndim + 1:
            raise DeepwaveValueError(
                "The propagation was determined to be "
                f"{ndim}d, so the provided models must have "
                f"either {ndim} or {ndim + 1} dimensions, but "
                f"found {model.ndim} dimensions."
            )
    return ndim


def is_inside_vmap(tensor: torch.Tensor) -> bool:
    """Check if a tensor is vmapped."""
    return _functorch.is_functorch_wrapped_tensor(
        tensor
    ) or _functorch.is_batchedtensor(tensor)


def values_readable(tensor: torch.Tensor) -> bool:
    """Whether ``tensor``'s values can be read as Python numbers.

    The ordinary tensors all read normally: a plain tensor, one that requires
    grad, the forward-mode dual :func:`torch.func.jvp` builds, the tensor
    :func:`torch.func.grad`/:func:`torch.func.jacrev` differentiates with
    respect to, and a tensor inside :func:`torch.compile` (at the cost of a
    graph break, as any ``.item()`` there is).

    Two kinds of tensor do not. ``torch.func.vmap`` maps its function over a
    batch dimension that the function cannot see, so a single number read out
    of a batched tensor is not defined and PyTorch raises rather than
    answering. A tensor that holds no data at all -- one on the ``meta``
    device, a ``FakeTensor``, a nested tensor -- also raises, for its own
    reason. Both answer this function's question the same way: no number can
    be had here.

    The question is asked by attempting the read, on one element and detached,
    rather than by inspecting the tensor's transform wrapper: PyTorch exposes
    no public predicate for "is this a vmap batched tensor, at any level", and
    the attempted read answers the question that actually matters -- can this
    value be used as a number? -- instead of a question about wrapper types
    that would have to be kept in step with them. What it does *not* do is
    identify *why* the read failed, so a caller reporting the failure has to
    word it for every one of these cases (see
    :func:`max_vel_unreadable_error`).

    Args:
        tensor: The tensor to try to read.

    Returns:
        Whether one of its values could be read as a ``float``.

    """
    if _torch_is_compiling():
        # Dynamo traces a read rather than performing it, so probing here
        # would only add a graph break to a question whose answer is yes: a
        # value inside ``torch.compile`` is readable, at the cost of the one
        # graph break the caller's own read already pays for.
        return True
    probe = tensor.detach()
    try:
        # One element, reached by integer indexing so that the cost is one
        # synchronisation rather than a pass over (or a copy of) the model.
        # An empty tensor has no values to read, but also nothing that could
        # hide them, so it reports readable and the caller's own read decides
        # what an empty model means.
        while probe.ndim > 0:
            if probe.shape[0] == 0:
                return True
            probe = probe[0]
        float(probe)
    except RuntimeError:
        return False
    return True


def max_vel_unreadable_error(propagator_name: str) -> DeepwaveRuntimeError:
    """The error for a model whose maximum wavespeed cannot be read.

    One wording for all nine propagators, and it has to be true of every way
    :func:`values_readable` can answer no, which that function cannot tell
    apart: a model batched by ``torch.func.vmap`` (the case a user meets), and
    a model that holds no values at all (``meta``, fake, nested). So the
    message states the fact -- the values could not be read here -- and names
    both causes rather than asserting one of them.

    ``max_vel`` sets the absorbing boundary's profiles and, through the CFL
    condition, the internal time step; both are one set of numbers for the
    whole call, which is why supplying it is a complete answer even when the
    call stands for a whole batch of models.

    Args:
        propagator_name: The propagator's public name, which begins the
            message.

    Returns:
        The exception to raise.

    """
    return DeepwaveRuntimeError(
        f"{propagator_name}: max_vel was not specified and the model's values "
        "could not be read at this point, so the maximum wavespeed could not "
        "be inferred from them. torch.func.vmap does this to a model it maps "
        "over: a batched tensor stands for one model per batch entry, so it "
        "has no single maximum. A tensor that holds no values -- one on the "
        "meta device, a fake tensor, a nested tensor -- cannot be read "
        "either. Pass max_vel=<the maximum absolute wavespeed over every "
        "model the call stands for>: it scales the absorbing boundary's "
        "profiles and sets the internal time step, which are one set of "
        "numbers for the whole call."
    )


def velocity_extremes(
    v: torch.Tensor, max_vel: Optional[float], propagator_name: str
) -> tuple[float, float]:
    """Returns ``(min_nonzero_abs_vel, max_abs_vel)`` for a velocity model.

    Used by the single-velocity propagators (scalar, scalar_born, acoustic,
    acoustic_born, viscoacoustic) to derive the values needed for the CFL
    condition and PML construction. When the model's values cannot be read --
    which is what ``vmap`` does to a model it batches -- ``max_vel`` must be
    supplied and the minimum is reported as ``0``; under any other
    ``torch.func`` transform the values are read as usual, so nothing has to be
    supplied.

    Both extremes are normally obtained with data-dependent reads (``.item()``
    on a possibly ``nonzero``-filtered tensor), which are opaque to
    ``torch.compile``/``torch.export`` and force a graph break, so both are
    skipped while ``torch.compiler.is_compiling()`` is true: the maximum falls
    back to the user-supplied ``max_vel`` if given (a data-dependent read of
    the actual model maximum is unavoidable, and not skipped, when ``max_vel``
    is not given), and the minimum is reported as ``0.0``, which causes
    :func:`check_points_per_wavelength` to emit its existing "test ... is
    being skipped" warning rather than perform a data-dependent read.

    Outside of compilation the real model maximum is always computed, in a
    single ``torch.aminmax`` pass over ``v.abs()``, even when ``max_vel`` is
    supplied: :func:`set_max_vel` still uses the user-supplied ``max_vel`` as
    the CFL/PML value in that case (this function's return value is not
    substituted for it), but needs the real maximum to reject an
    underestimated ``max_vel``.

    Both extremes leave as Python floats, so neither is part of the autograd
    graph: an inferred maximum reshapes the absorbing boundary and
    chooses the time step, which are properties of the discretisation rather
    than differentiable functions of the model. ``.item()`` is what severs the
    graph; the ``detach()`` before it states that intent at the read rather
    than leaving it implicit.

    Args:
        v: The velocity model.
        max_vel: The user-supplied maximum velocity, or ``None``.
        propagator_name: The propagator's public name, for the error message.

    Returns:
        ``(min_nonzero_abs_vel, max_abs_vel)``.

    Raises:
        DeepwaveRuntimeError: If ``v``'s values cannot be read (a ``vmap``
            batched model) and ``max_vel`` was not given.

    """
    if not values_readable(v):
        if max_vel is None:
            raise max_vel_unreadable_error(propagator_name)
        return 0.0, max_vel
    if _torch_is_compiling():
        # If max_vel was not supplied, the actual model maximum is needed for
        # the CFL condition; that data-dependent read cannot be skipped, even
        # while compiling.
        max_abs = max_vel if max_vel is not None else v.detach().abs().max().item()
        return 0.0, max_abs
    v_abs = v.detach().abs()
    max_abs = torch.aminmax(v_abs).max.item()
    v_nonzero = v_abs[v_abs != 0]
    min_nonzero = v_nonzero.min().item() if v_nonzero.numel() > 0 else 0.0
    return min_nonzero, max_abs


def is_vmap_batched(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is (or wraps) a vmap ``BatchedTensor``.

    Unlike :func:`is_inside_vmap`, this ignores pure forward-mode/grad functorch
    wrappers (e.g. the ``GradTrackingTensor`` that ``torch.func.jvp`` produces):
    it peels every functorch level and reports ``True`` only if some level is a
    batched (vmap) one. This lets forward-mode AD (``jvp``) flow through inputs
    such as ``source_amplitudes`` while still rejecting genuine ``vmap`` over
    those non-model inputs.
    """
    t = tensor
    while _functorch.is_functorch_wrapped_tensor(t):
        if _functorch.is_batchedtensor(t):
            return True
        t = _functorch.get_unwrapped(t)
    return False


def check_inputs_not_vmapped(*args: Optional[LocationsLike]) -> None:
    """Check that none of the parameters is vmapped.

    Only ``vmap`` (batched) inputs are rejected; forward-mode (``jvp``) duals on
    these inputs are permitted (scalar routes them through the Born backend).
    """
    for tensor in args:
        if (
            tensor is not None
            and isinstance(tensor, torch.Tensor)
            and is_vmap_batched(tensor)
        ):
            raise DeepwaveNotImplementedError(
                "Only property models (e.g. velocity) support vmap currently."
            )


def check_finite(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Raises if `x` contains a NaN or Inf, otherwise returns `x` unchanged.

    A cheap (one reduction over `x`) sanity check for a diverged simulation,
    for the tensors Deepwave does not check itself. Every propagator now
    checks its own outputs -- the final wavefields and the receiver
    amplitudes -- before returning them, and raises this same error type if
    any is non-finite; that is what ``check_output_finite=True`` (the
    default) does. What Deepwave still does not check is every *step*, which
    would cost a synchronisation per step and is prohibitively slow, nor any
    tensor that is not a propagator output: a gradient, a loss, or a model an
    optimiser has just updated. Call this on those (or, for early detection
    during a long run, from inside a ``forward_callback``/
    ``backward_callback`` on the wavefield -- see the "Detecting divergence
    (NaN) early" recipe in :doc:`/troubleshooting`) to fail fast with a clear
    message instead of discovering the ``NaN`` much later.

    Args:
        x: The tensor to check.
        name: A label for `x`, used in the error message (e.g.
            ``"receiver data"``, ``"v.grad"``).

    Returns:
        `x`, unchanged, so the call can be chained inline.

    Raises:
        deepwave.DeepwaveFloatingPointError: If `x` contains any non-finite
            (``NaN`` or ``Inf``) value. This is also a `FloatingPointError`
            (the type this function raised historically), so an existing
            ``except FloatingPointError`` still catches it, and it is a
            `deepwave.DeepwaveError`, so ``except deepwave.DeepwaveError``
            catches it alongside Deepwave's other errors.

    Example:
        >>> import deepwave
        >>> out = deepwave.scalar(v, grid_spacing, dt, ...)
        >>> receiver_data = deepwave.check_finite(
        ...     out.receiver_amplitudes, "receiver data"
        ... )

    Stability: provisional (see :doc:`/stability`).

    """
    if not torch.isfinite(x).all():
        n_nan = int(torch.isnan(x).sum())
        n_inf = int(torch.isinf(x).sum())
        raise DeepwaveFloatingPointError(
            f"{name} contains non-finite values ({n_nan} NaN, {n_inf} Inf, "
            f"out of {x.numel()} elements). This usually means the "
            "simulation diverged. Common causes: dt/grid_spacing violating "
            "the CFL condition (see the max_vel argument and the accuracy "
            "documentation page), an unstable optimiser step during FWI "
            "producing an extreme velocity/model update, or a NaN already "
            "present in the input model/source amplitudes. See the "
            "'Detecting divergence (NaN) early' section of the "
            "troubleshooting documentation page for a forward_callback "
            "recipe that catches this at the first bad time step instead of "
            "at the end of the run."
        )
    return x


def _spec_n_wavefields(
    spec: "deepwave._engine.PropagatorSpec", ndim: int, q_n_mechanisms: int
) -> int:
    """The total wavefield count a propagator allocates at ``ndim``.

    Both the physical fields and the absorbing boundary's memory variables,
    which is what :func:`estimate_memory`'s "forward wavefields" number has
    always meant. Taken from the spec, so a propagator that gains or loses a
    field changes this automatically.
    """
    names = cast("Callable[..., list[str]]", spec.wavefield_names)
    if "n_mechanisms" in inspect.signature(names).parameters:
        return len(names(ndim, n_mechanisms=q_n_mechanisms))
    return len(names(ndim))


def _auxiliary_state_elements(
    spec: "deepwave._engine.PropagatorSpec",
    ndim: int,
    q_n_mechanisms: int,
    padded_shape: abc.Sequence[int],
    accuracy: int,
    pml_width: abc.Sequence[int],
) -> int:
    """Elements in **one** state threaded across a segment boundary.

    The state a checkpointed run holds from one segment to the next is the
    propagator's whole wavefield set, but the fields are not all the same
    size, and quoting them all at the full padded grid is what stopped
    ``total_bytes_lower_bound`` being a lower bound once the segment count grew
    (the term is multiplied by that count, so an over-estimate of one state is
    an over-estimate multiplied by ``k``). Two reductions apply, and both are
    computed here so that the number is a *lower* bound on the state, in the
    same direction as every other term:

    * **The FD halo is not carried.** Every wavefield returned by a
      propagator, and therefore every field of a threaded state, has the
      finite-difference halo trimmed off. ``accuracy`` cells per axis is
      removed here; a staggered grid trims one cell more than that, so this
      keeps the figure on the low side rather than guessing which grid a
      propagator uses.
    * **The PML memory variables are stored as boundary slabs.** The always-on
      slab optimisation keeps only ``pml_width + accuracy`` cells at each end
      of the axis a memory variable varies along, which is much less than the
      full grid at realistic PML widths. Where the slabs would meet (a narrow
      model, or a wide PML) the field is left whole, exactly as the
      compression does. Where the optimisation is unavailable -- a propagator
      with no compiled backend, an explicitly requested Python backend, a
      non-default ``boundary``, M-PML -- the real state is bigger than this,
      which again leaves the figure a lower bound.

    Args:
        spec: The propagator's spec.
        ndim: The number of spatial dimensions.
        q_n_mechanisms: The number of GSLS relaxation mechanisms, for the two
            attenuating propagators whose wavefield set depends on it.
        padded_shape: The model shape after FD and PML padding.
        accuracy: The resolved finite-difference accuracy order.
        pml_width: The resolved per-face PML width, ``2 * ndim`` entries.

    Returns:
        The element count of one auxiliary state.

    """
    names_fn = cast("Callable[..., list[str]]", spec.wavefield_names)
    if "n_mechanisms" in inspect.signature(names_fn).parameters:
        names = names_fn(ndim, n_mechanisms=q_n_mechanisms)
    else:
        names = names_fn(ndim)
    # What a propagator returns, and therefore what a state carries, is the
    # padded grid with the FD halo removed.
    view = [max(1, int(padded_shape[d]) - accuracy) for d in range(ndim)]
    view_elements = 1
    for extent in view:
        view_elements *= extent
    slab_axes = spec.pml_slab_axes(ndim) if spec.pml_slab_axes is not None else {}
    total = 0
    for name in names:
        axis = slab_axes.get(name)
        if axis is None:
            total += view_elements
            continue
        # The two retained end slabs, exactly as ``_compress_state`` cuts
        # them; if together they cover the axis, nothing is saved and the
        # whole field is kept.
        slab = pml_width[2 * axis] + pml_width[2 * axis + 1] + 2 * accuracy
        total += view_elements // view[axis] * min(slab, view[axis])
    return total


def _propagator_module(propagator: str) -> Any:
    """The module implementing ``propagator``, imported lazily.

    Lazy because :mod:`deepwave._common` is imported *by* the propagator
    modules; by the time any of the helpers below runs, the import is a
    :data:`sys.modules` lookup.
    """
    return importlib.import_module(f"deepwave.{propagator}")


def _propagator_user_model_names(propagator: str, ndim: int) -> tuple[str, ...]:
    """The user-facing model argument names of ``propagator``, in order.

    The models the *caller* passes -- ``acoustic``'s ``v``/``rho`` rather than
    the internal ``K``/``B*`` the engine holds -- in the order the public
    function takes them, including a Born propagator's perturbation models and
    any optional model tensor the propagator accepts at this ``ndim``.

    Every propagator module carries the ``USER_MODEL_NAMES`` constant. A module
    whose list depends on the number of dimensions (``elastic_vti``, whose
    ``c66`` is a model only in 3-D) additionally provides a
    ``user_model_names`` function, which is then the answer.
    """
    module = _propagator_module(propagator)
    per_ndim = getattr(module, "user_model_names", None)
    if per_ndim is not None:
        return tuple(cast("Callable[[int], abc.Sequence[str]]", per_ndim)(ndim))
    return cast("tuple[str, ...]", module.USER_MODEL_NAMES)


def _has_grad_count_tables(propagator: str) -> bool:
    """Whether ``propagator`` can report exact backward-storage counts.

    Every propagator module names its user-facing models
    (``USER_MODEL_NAMES``), but only the four released in ``v0.0.27`` also
    carry the per-model
    ``storage_slot_flags``/``internal_model_grad_flags`` tables that turn
    "which models require a gradient" into "how many storage slots the
    engine allocates". The seven newer ones do not yet, so
    :func:`estimate_memory` falls back to a conservative count for them
    rather than refusing to answer -- the wavefield count, which dominates
    the estimate, comes from the spec and is exact for all nine. Giving
    the remaining seven exact tables belongs with the wider
    ``estimate_memory`` reshape.
    """
    module = _propagator_module(propagator)
    return all(
        hasattr(module, attribute)
        for attribute in (
            "USER_MODEL_NAMES",
            "storage_slot_flags",
            "internal_model_grad_flags",
        )
    )


def _propagator_models_requiring_grad(
    propagator: str, models: abc.Sequence[torch.Tensor], ndim: int
) -> list[str]:
    """The user-facing names of the models requiring a gradient in a live call.

    Recovers, via the propagator module's own ``user_model_grad_flags``, the
    ``models_requiring_grad`` argument of :func:`estimate_memory` that
    corresponds to a list of the *internal* model tensors the engine is
    holding -- so the out-of-memory message in :mod:`deepwave._engine` can
    quote the exact estimate for the run that failed.
    """
    module = _propagator_module(propagator)
    flags = cast(
        "Callable[[abc.Sequence[torch.Tensor], int], list[bool]]",
        module.user_model_grad_flags,
    )
    return [
        name
        for name, flag in zip(
            _propagator_user_model_names(propagator, ndim), flags(models, ndim)
        )
        if flag
    ]


def _propagator_grad_counts(
    propagator: str, model_requires_grad: abc.Sequence[bool], ndim: int
) -> tuple[int, int]:
    """How many storage slots and gradient-carrying internal models a run has.

    Both come from the propagator module's own ``storage_slot_flags`` and
    ``internal_model_grad_flags`` -- the same functions its forward pass uses --
    so :func:`estimate_memory` cannot drift from what the engine allocates.

    Args:
        propagator: The propagator's name.
        model_requires_grad: One flag per user-facing model, in
            :func:`_propagator_user_model_names` order.
        ndim: The number of spatial dimensions.

    Returns:
        ``(n_storage_slots, n_internal_models_requiring_grad)``.
    """
    module = _propagator_module(propagator)
    slot_flags = cast(
        "Callable[[abc.Sequence[bool], int], list[bool]]", module.storage_slot_flags
    )
    model_flags = cast(
        "Callable[[abc.Sequence[bool], int], list[bool]]",
        module.internal_model_grad_flags,
    )
    return (
        sum(slot_flags(model_requires_grad, ndim)),
        sum(model_flags(model_requires_grad, ndim)),
    )


def _propagator_worst_case_grad_counts(
    propagator: str, n_models_requiring_grad: int, ndim: int
) -> tuple[int, int]:
    """The largest counts any choice of ``n_models_requiring_grad`` models gives.

    Used when the caller says only *how many* models require a gradient, not
    which; each count is maximised independently, so the pair is an upper
    bound rather than the cost of one particular choice.
    """
    if not _has_grad_count_tables(propagator):
        # Conservative fallback: assume each model requiring a gradient costs
        # one storage slot and one internal model gradient.
        return n_models_requiring_grad, n_models_requiring_grad
    names = _propagator_user_model_names(propagator, ndim)
    if not 0 <= n_models_requiring_grad <= len(names):
        raise DeepwaveValueError(
            f"n_models_requiring_grad must be between 0 and {len(names)} for "
            f"propagator {propagator!r} (whose models are {list(names)}), but "
            f"got {n_models_requiring_grad}."
        )
    n_slots = 0
    n_model_grads = 0
    for combination in itertools.combinations(
        range(len(names)), n_models_requiring_grad
    ):
        flags = [i in combination for i in range(len(names))]
        slots, model_grads = _propagator_grad_counts(propagator, flags, ndim)
        n_slots = max(n_slots, slots)
        n_model_grads = max(n_model_grads, model_grads)
    return n_slots, n_model_grads


def estimate_memory(
    propagator: Union[Callable[..., Any], str],
    model_shape: abc.Sequence[int],
    nt: int,
    n_shots: int,
    dtype: torch.dtype = torch.float32,
    *,
    accuracy: int = 4,
    pml_width: Union[int, abc.Sequence[int]] = 20,
    model_gradient_sampling_interval: int = 1,
    storage: StorageLike = STORAGE_DEFAULT,
    storage_mode: Optional[StorageMode] = None,
    storage_path: Optional[StoragePath] = None,
    storage_compression: Optional[bool] = None,
    n_models_requiring_grad: Optional[int] = None,
    models_requiring_grad: Optional[abc.Sequence[str]] = None,
    q_n_mechanisms: int = 1,
    device: Union[str, torch.device] = "cpu",
    n_segments: Optional[int] = None,
) -> MemoryEstimate:
    """Rough byte-size estimate for a propagator call, before running it.

    This is meant to answer "will this run fit in memory" for a proposed
    ``model_shape``/``nt``/``n_shots``/``storage_mode`` *before* spending the
    time to build the tensors and start propagating. It mirrors, as closely as
    is practical without actually allocating anything, the two dominant
    sources of memory use identified for the backward pass:
    ``StorageManager`` (the sampled wavefield history kept for the model
    gradient) and ``allocate_model_gradients`` (the per-thread/per-shot
    scratch buffers used to accumulate that gradient without a race). It also
    gives a (much rougher) figure for the wavefields alive during the forward
    time loop itself.

    **Approximations** (read before trusting the numbers):

    * The number of same-shaped buffers alive at once during the forward
      time loop is equation- and dimensionality-specific (each block below
      is the *total* wavefield count returned by that propagator, both PML
      memory variables included); this is the single biggest approximation.
      Pass ``propagator`` to fill it in from the table below (using
      ``len(model_shape)`` as the number of dimensions); without it, a
      placeholder of 3 is used:

        - ``scalar`` / ``scalar_born`` background: ``2 + 2 * ndim`` (2-D: 6,
          3-D: 8); ``scalar_born`` (background + scattered): double that
          (2-D: 12, 3-D: 16).
        - ``acoustic``: ``1 + 3 * ndim`` (2-D: 7, 3-D: 10).
        - ``elastic``: 13 (2-D), 27 (3-D).

    * The number of *stored wavefield histories* is not the number of models
      you ask for a gradient for: the engine allocates one
      ``StorageManager`` slot per stored spatial derivative, which for
      ``acoustic`` and ``elastic`` is several slots per model. Name the models
      with ``models_requiring_grad`` and this is filled in exactly, from the
      propagator's own slot function (the one its forward pass uses). Give
      only ``n_models_requiring_grad`` and a worst case over that many models
      is used instead; give neither a propagator nor names and the old
      one-slot-per-model assumption remains, which under-predicts
      ``acoustic``'s density gradient by ``1 + ndim`` times and ``elastic``'s
      three-parameter gradient by 9/3 in 3-D.
    * ``scalar`` with a ``backward_callback`` keeps a *second* full wavefield
      history (the raw forward field the callback is given); that extra copy
      is not modelled here.
    * Like ``StorageManager`` itself, any dtype other than
      ``torch.float32`` is costed as 8 bytes/element (i.e. as if it were
      ``torch.float64``); this matches current Deepwave behaviour but is a
      simplification for exotic dtypes.
    * ``model_shape`` should be your *unpadded* model's spatial shape; this
      function adds the finite-difference (``accuracy // 2`` per side) and
      PML padding itself, the same way ``extract_survey`` does. The
      staggered-grid propagators (``acoustic``, ``elastic``) really pad one
      cell less on the far side of each axis, so for them every per-model
      term here is slightly (about 0.7% per axis at 140 padded cells)
      generous.
    * ``storage_compression=True`` is modelled from ``StorageManager``'s own
      block-quantisation formula (8-per-side blocks, 1 byte/element plus one
      ``dtype_size``-wide scale per block): it shrinks the sampled-history
      term, but ``StorageManager`` always keeps two extra full-size
      single-shot buffers alive alongside the compressed history (for
      building the next block before it is written), so compression's net
      saving is smaller than the raw quantisation ratio, especially at small
      ``n_shots``. Disk-mode host/device splits are not modelled;
      ``storage_mode="disk"`` is costed the same as ``"device"``, an upper
      bound on total (device + host/disk) footprint.
    * This is a static estimate: it does not know about ``survey_pad``
      shrinking the model per shot, per-shot-batched models, or checkpointing
      schemes external to Deepwave.
    * The unmodelled fixed per-call overhead is the propagator's *other*
      full-size buffers: the backward pass's own set of padded wavefields
      (a second copy of the forward set counted above), the padding-removed
      wavefields returned to the caller (alive across ``.backward()``), the
      padded/staggered internal model copies, and one gradient buffer per
      internal model requiring a gradient. It scales with the propagator's
      wavefield and model counts, not with ``nt``, so it matters most for
      3-D elastic at small ``nt``: measured at 96^3, 1 shot, ``nt=40``, it is
      about 0.9 GiB (of which 0.27 GiB is the backward wavefield set, 0.24
      GiB the returned wavefields, 0.19 GiB the internal model copies, 0.09
      GiB the ping-pong buffers and 0.08 GiB the model gradients).
    * Measured against real propagator runs (see :doc:`/troubleshooting`'s
      "Measured accuracy of estimate_memory" table), this estimate
      **under-predicts real peak memory at every configuration tested,
      never over-predicts**: by roughly 4-20% for the default
      ``storage_mode="device"`` uncompressed with the storage slots named
      exactly (``models_requiring_grad``), growing markedly worse (up to
      ~70%) as the modelled terms (sampled history, gradient scratch)
      shrink relative to that fixed per-call overhead -- i.e.
      ``storage_compression=True``, a large
      ``model_gradient_sampling_interval``, and ``storage_mode="cpu"``/
      ``"disk"`` are all directions where the gap widens, as is a short
      ``nt`` on a many-wavefield propagator. Treat the
      returned figure as a lower bound and leave headroom, not a ceiling.
    * ``n_segments=`` is part of that guarantee rather than an exception to
      it. Every term it adds is a floor on what it stands for: the boundary
      states are sized field by field, without the finite-difference halo and
      with the PML memory variables costed as the boundary slabs they are
      stored as, and they are counted once per segment only where a model
      gradient is being computed -- a forward-only segmented run holds one
      state, whatever the segment count. Measured against real peaks on a
      64-cubed ``scalar`` run at segment counts from 2 to 120, with and
      without a model gradient, the figure stayed below the measured peak in
      every case.

    Args:
        propagator: The propagator this call would use: its name, its
            function, or its ``Module`` class. Any of the nine is accepted.
            It is **required**, because every number below depends on it: the
            forward wavefield count comes from its spec, and the
            backward-storage slot count from its own storage-slot rule.
            (It used to be optional, substituting a hard-coded guess of three
            forward wavefields -- a figure no propagator's spec agreed with.)
            Raises ``ValueError`` if it is not one of the recognised
            propagators.
        model_shape: The spatial shape of one (unpadded) model, e.g.
            ``v.shape``.
        nt: The number of user-facing time steps (as passed to the
            propagator, i.e. matching ``source_amplitudes.shape[-1]``).
        n_shots: The number of shots propagated together.
        dtype: The dtype of the model/computation.
        accuracy: The finite-difference accuracy order.
        pml_width: The PML width (int, or one value per face as accepted by
            the propagators).
        model_gradient_sampling_interval: As passed to the propagator.
        storage: The storage policy, as one object: a mode string or a
            :class:`deepwave.Storage`. An alternative spelling for the three
            individual ``storage_*`` arguments, which are permanent and not
            deprecated; passing both spellings raises.
        storage_mode: One of ``"device"``, ``"cpu"``, ``"disk"``, or
            ``"none"`` (``"none"`` estimates 0 for both backward-storage and
            gradient-scratch, matching what ``StorageManager`` actually
            allocates in that mode).
        storage_path: Accepted so that one dict of storage arguments can be
            passed to both this function and the propagator. It does not
            affect any byte count: disk mode is costed identically to device
            mode (see the approximations above).
        storage_compression: As passed to the propagator
            (``device``/``cpu``/``disk`` modes only; ignored, matching
            ``StorageManager``, when ``storage_mode="none"``). **Lossy --
            changes the model gradient.** See :func:`deepwave.elastic`.
        n_models_requiring_grad: How many model tensors require a gradient,
            when you do not want to say which (defaults to 1). With
            ``propagator``, the storage-slot and gradient-buffer counts are
            the worst case over any choice of that many of that propagator's
            models, i.e. an upper bound. Prefer
            ``models_requiring_grad``, which is exact. Raises ``ValueError``
            if negative, if it exceeds the propagator's model count, or if it
            contradicts ``models_requiring_grad``.
        models_requiring_grad: Which models require a gradient, by their
            user-facing argument names: ``"v"`` for ``scalar``; ``"v"`` and
            ``"dv"`` for ``scalar_born``; ``"v"`` and ``"rho"`` for
            ``acoustic``; ``"lamb"``, ``"mu"`` and ``"buoyancy"`` for
            ``elastic``. Any propagator's names are
            :meth:`deepwave.PropagatorSpec.model_names`. This is the
            accurate option on the four
            propagators that carry storage-slot tables (``scalar``,
            ``scalar_born``, ``acoustic``, ``elastic``): the number of stored
            wavefield histories is then taken from the propagator's own
            storage-slot function, so it matches what the engine really
            allocates (e.g. ``["v", "rho"]`` in 3-D ``acoustic`` stores four
            histories, not two). On the other seven the same conservative
            one-slot-per-model fallback as ``n_models_requiring_grad`` is
            used. Raises ``ValueError`` for an unrecognised or repeated name,
            or a repeated name.
        storage_compression: If True, the stored wavefield history is
            compressed, which changes the arithmetic this function models.
            A :class:`deepwave.Storage` passed here contributes its ``mode``
            and its ``lossy_compression``, which change the arithmetic.
            ``path`` has no bearing on a byte count.
        q_n_mechanisms: The number of GSLS relaxation mechanisms, for the
            viscoacoustic propagator, whose wavefield count depends on it.
            Ignored by the other families. Defaults to 1.
        device: The device the propagation would run on (only its type, cpu
            vs cuda, matters here).
        n_segments: If given, the number of time segments the run would be
            split into by :func:`deepwave.checkpointing.run_segments` (or by
            the propagator's own ``checkpoint_segments=``). The stored history
            is then divided by ``n_segments`` and an auxiliary-state term added
            for the states held alive between segments. Without it, the
            estimate has no model of checkpointing at all and over-predicts a
            checkpointed run's peak substantially. Held to the same standard
            as the propagator's own spelling of it: a bool or a non-int raises
            ``DeepwaveTypeError``, and a value below 1 or above ``nt`` raises
            ``DeepwaveValueError`` (a run cannot have more segments than time
            steps, and the propagator refuses that too).

            How many auxiliary states this costs depends on whether a model
            gradient is being computed. With one, each segment's state is held
            from the forward pass until that segment is recomputed, so the
            term scales with ``n_segments``. Without one -- no model named in
            ``models_requiring_grad``, or ``n_models_requiring_grad=0`` --
            there is nothing to checkpoint: the segmented driver threads a
            single state forward and drops the previous one, so the term is
            the size of **one** state however many segments there are.

    Returns:
        MemoryEstimate: a dict (see that TypedDict) with the following keys:

            - ``padded_model_shape``: `model_shape` after adding
              finite-difference and PML padding.
            - ``padded_model_elements``: number of elements in one padded
              model.
            - ``n_threads``: the thread count used for the gradient-scratch
              estimate (``torch.get_num_threads()`` on CPU, 1 on CUDA).
            - ``forward_wavefields_bytes``: estimated bytes for the forward
              time-loop wavefields.
            - ``backward_storage_bytes``: estimated bytes for the sampled
              wavefield history kept for the model gradient
              (``StorageManager``).
            - ``gradient_scratch_bytes``: estimated bytes for the
              per-thread/per-shot gradient accumulation buffers
              (``allocate_model_gradients``), one per *internal* model
              requiring a gradient (again more than one per user-facing
              model for ``acoustic``/``elastic``; the gradient buffers
              themselves are part of the unmodelled overhead above).
            - ``auxiliary_state_bytes``: estimated bytes for the segment
              auxiliary states held alive between checkpointed segments. Zero
              unless ``n_segments`` was given and is greater than 1. Each
              state is sized field by field, with the finite-difference halo
              removed and the PML memory variables costed as the boundary
              slabs they are actually stored as, so it is a lower bound on the
              state rather than a full-grid over-estimate of it.
            - ``total_bytes_lower_bound``: the sum of the four components
              above.

    Example:
        >>> import deepwave
        >>> est = deepwave.estimate_memory(
        ...     model_shape=(2301, 751), nt=750, n_shots=115, accuracy=4,
        ...     pml_width=20, propagator="scalar",
        ...     models_requiring_grad=["v"],
        ... )
        >>> est["total_bytes_lower_bound"] / 1e9  # doctest: +SKIP
        648.41...

    (That figure is dominated by ``backward_storage_bytes``, since it stores
    750 full-model snapshots per shot for the gradient -- exactly the cost
    that ``model_gradient_sampling_interval`` or a smaller ``storage_mode``
    lets you trade against recompute or disk I/O; see :doc:`/troubleshooting`.)

    """
    ndims = len(model_shape)
    if ndims == 0:
        raise DeepwaveValueError("model_shape must not be empty.")
    if any(s <= 0 for s in model_shape):
        raise DeepwaveValueError("model_shape elements must be positive.")
    if nt < 0:
        raise DeepwaveValueError("nt must be non-negative.")
    if n_shots <= 0:
        raise DeepwaveValueError("n_shots must be positive.")
    if model_gradient_sampling_interval < 1:
        raise DeepwaveValueError("model_gradient_sampling_interval must be >= 1.")

    if n_models_requiring_grad is not None and n_models_requiring_grad < 0:
        raise DeepwaveValueError("n_models_requiring_grad must be non-negative.")

    # The same standard the propagator-level spelling of this concept
    # (``checkpoint_segments=`` / ``run_segments``' ``n_segments=``) already
    # holds itself to, applied here so that the estimator cannot be handed
    # something the run would refuse -- and, in the float case, cannot return
    # float "bytes" from a TypedDict whose fields are declared ``int``.
    if n_segments is not None:
        if isinstance(n_segments, bool):
            raise DeepwaveTypeError(
                "n_segments must be an int (a number of time segments), not a "
                f"bool, but got {n_segments!r}."
            )
        if not isinstance(n_segments, int):
            raise DeepwaveTypeError(
                "n_segments must be an int (a number of time segments), but "
                f"got {n_segments!r}."
            )
        if n_segments < 1:
            raise DeepwaveValueError(
                "n_segments must be a positive int (a number of time "
                f"segments), but got {n_segments!r}."
            )
        if n_segments > nt:
            raise DeepwaveValueError(
                f"n_segments ({n_segments}) cannot exceed the number of time "
                f"steps ({nt}); each segment must contain at least one step."
            )

    # The wavefield count comes from the propagator's own spec rather than a
    # hand-maintained table, so all nine are covered and the number cannot
    # drift from what the engine allocates. ``propagator`` is required, so the
    # documented-but-wrong placeholder count this used to fall back to when it
    # was omitted no longer exists.
    from deepwave.introspection import _resolve_spec

    spec = _resolve_spec(propagator)
    propagator = spec.name
    n_forward_wavefields = _spec_n_wavefields(spec, ndims, q_n_mechanisms)

    resolved_storage = resolve_storage_arguments(
        storage,
        storage_mode,
        storage_path,
        storage_compression,
    )
    storage_mode = resolved_storage.mode
    storage_compression = resolved_storage.lossy_compression

    # How many backward-storage slots the engine would allocate, and how many
    # internal models would carry a gradient: exact when the caller names the
    # models, a worst case over that many models when it only counts them, and
    # (with no propagator to ask) the count itself.
    n_models = 1 if n_models_requiring_grad is None else n_models_requiring_grad
    if models_requiring_grad is not None:
        if n_models_requiring_grad is not None and n_models_requiring_grad != len(
            models_requiring_grad
        ):
            raise DeepwaveValueError(
                "n_models_requiring_grad "
                f"({n_models_requiring_grad}) is inconsistent with "
                f"models_requiring_grad ({list(models_requiring_grad)}, "
                f"{len(models_requiring_grad)} models); pass only one of them."
            )
        names = _propagator_user_model_names(propagator, ndims)
        unrecognised = sorted(set(models_requiring_grad) - set(names))
        if unrecognised:
            raise DeepwaveValueError(
                f"Unrecognised model name(s) {unrecognised} in "
                f"models_requiring_grad for propagator {propagator!r}; "
                f"expected some of {list(names)}."
            )
        if len(set(models_requiring_grad)) != len(models_requiring_grad):
            raise DeepwaveValueError(
                "models_requiring_grad must not contain duplicates, but got "
                f"{list(models_requiring_grad)}."
            )
        n_storage_slots, n_model_grads = _grad_counts_for_names(
            propagator, models_requiring_grad, ndims
        )
    elif propagator is not None:
        n_storage_slots, n_model_grads = _propagator_worst_case_grad_counts(
            propagator, n_models, ndims
        )
    else:
        n_storage_slots = n_model_grads = n_models

    pml_width_list = set_pml_width(pml_width, ndims)
    accuracy = set_accuracy(accuracy)
    fd_pad = accuracy // 2
    padded_shape = tuple(
        int(model_shape[d])
        + 2 * fd_pad
        + pml_width_list[2 * d]
        + pml_width_list[2 * d + 1]
        for d in range(ndims)
    )
    return _estimate_memory_core(
        padded_shape=padded_shape,
        n_shots=n_shots,
        num_steps_stored=nt // model_gradient_sampling_interval,
        dtype=dtype,
        device=device,
        n_forward_wavefields=n_forward_wavefields,
        n_storage_slots=n_storage_slots,
        n_model_grads=n_model_grads,
        storage_mode=_storage_mode_enum(storage_mode),
        storage_compression=bool(storage_compression),
        n_segments=n_segments,
        # The state threaded across a segment boundary is the propagator's
        # whole wavefield set -- the same set the forward loop keeps alive --
        # so it is read from the same spec rather than a second, driftable
        # table, and sized field by field.
        auxiliary_state_elements=_auxiliary_state_elements(
            spec, ndims, q_n_mechanisms, padded_shape, accuracy, pml_width_list
        ),
    )


def _grad_counts_for_names(
    propagator: str,
    models_requiring_grad: abc.Sequence[str],
    ndims: int,
) -> tuple[int, int]:
    """Storage slots and internal model gradients, for named user models.

    Exact for the propagators that carry grad-count tables, and the documented
    conservative one-slot-per-model fallback for the rest. Shared by the
    parametric :func:`estimate_memory` and by the run-shaped estimate a real
    call makes of itself, so the two cannot answer differently.

    Args:
        propagator: The propagator's name.
        models_requiring_grad: The user-facing model names requiring a
            gradient.
        ndims: The number of spatial dimensions.

    Returns:
        ``(n_storage_slots, n_model_grads)``.

    """
    if _has_grad_count_tables(propagator):
        names = _propagator_user_model_names(propagator, ndims)
        wanted = set(models_requiring_grad)
        return _propagator_grad_counts(
            propagator, [name in wanted for name in names], ndims
        )
    # The same conservative fallback :func:`_propagator_worst_case_grad_counts`
    # uses for the propagators without grad-count tables: one storage slot and
    # one internal model gradient per model named.
    return len(models_requiring_grad), len(models_requiring_grad)


def _estimate_memory_core(
    *,
    padded_shape: tuple[int, ...],
    n_shots: int,
    num_steps_stored: int,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    n_forward_wavefields: int,
    n_storage_slots: int,
    n_model_grads: int,
    storage_mode: "_StorageModeEnum",
    storage_compression: bool,
    n_segments: Optional[int],
    auxiliary_state_elements: int,
) -> MemoryEstimate:
    """The byte arithmetic shared by :func:`estimate_memory` and ``plan()``.

    Both callers answer the same question and must not drift, so the parametric
    form (which derives the padded shape and the slot counts from arguments)
    and the run-shaped form (which reads them off a resolved run) differ only
    in how they *obtain* these inputs, never in what is done with them.

    Args:
        padded_shape: The model shape after FD and PML padding.
        n_shots: The number of shots in the batch.
        num_steps_stored: How many time samples of history are kept, i.e. the
            user-level ``nt`` divided by the model-gradient sampling interval.
        dtype: The floating-point dtype of the run.
        device: The device, whose type selects the threading model.
        n_forward_wavefields: How many same-shaped wavefields are alive at once
            during the forward time loop.
        n_storage_slots: How many ``StorageManager`` history slots the engine
            would allocate.
        n_model_grads: How many internal models would carry a gradient.
        storage_mode: The resolved storage mode.
        storage_compression: Whether the history is lossily compressed.
        n_segments: The number of checkpointed time segments, or None for an
            unsegmented run.
        auxiliary_state_elements: The element count of one state threaded
            across a segment boundary, from :func:`_auxiliary_state_elements`.

    Returns:
        The estimate.

    """
    padded_elements = 1
    for s in padded_shape:
        padded_elements *= s

    # Mirrors StorageManager.dtype_size exactly.
    dtype_size = 4 if dtype == torch.float32 else 8

    is_cuda = torch.device(device).type == "cuda"
    n_threads = 1 if is_cuda else max(1, torch.get_num_threads())

    forward_wavefields_bytes = (
        n_forward_wavefields * n_shots * padded_elements * dtype_size
    )

    resolved_storage_mode = storage_mode

    if resolved_storage_mode == _StorageModeEnum.NONE:
        backward_storage_bytes = 0
        gradient_scratch_bytes = 0
    else:
        # Mirrors StorageManager.__init__'s shot_bytes_comp exactly: one
        # quantised byte per element plus one dtype_size-wide scale per
        # 8-per-side block, rounded up to a whole number of dtype_size-wide
        # elements (StorageManager stores the compressed bytes packed into a
        # tensor of the original dtype).
        if storage_compression:
            n_blocks = 1
            for s in padded_shape:
                n_blocks *= (s + 7) // 8
            shot_bytes_comp = (
                (padded_elements + n_blocks * dtype_size + dtype_size - 1) // dtype_size
            ) * dtype_size
            per_shot_step_bytes = shot_bytes_comp
        else:
            per_shot_step_bytes = padded_elements * dtype_size

        per_slot_bytes = num_steps_stored * n_shots * per_shot_step_bytes
        backward_storage_bytes = per_slot_bytes * n_storage_slots
        if resolved_storage_mode == _StorageModeEnum.CPU:
            # The small on-device double-buffer (store_1a/store_1b), on top
            # of the (host/pinned) main storage above -- always full-size,
            # whether or not compression is active.
            backward_storage_bytes += (
                2 * n_shots * padded_elements * dtype_size * n_storage_slots
            )
            if storage_compression:
                # CPU mode additionally keeps one on-device compressed
                # single-shot buffer (store_2) alongside the main storage.
                backward_storage_bytes += n_shots * shot_bytes_comp * n_storage_slots
        elif storage_compression:
            # DEVICE (and, per the disk-costed-as-device simplification
            # above, DISK) compression trades the single full-size history
            # buffer for two full-size single-shot buffers (store_1a/
            # store_1b, used to build each block before it is written) plus
            # the compressed history (store_2, already in per_model_bytes
            # above).
            backward_storage_bytes += (
                2 * n_shots * padded_elements * dtype_size * n_storage_slots
            )

        tmp_batch = n_shots if is_cuda else min(n_shots, n_threads)
        need_tmp = tmp_batch > 1
        gradient_scratch_bytes = (
            tmp_batch * padded_elements * dtype_size * n_model_grads if need_tmp else 0
        )

    # Checkpointing trades history for auxiliary states: only one segment's
    # history is alive at a time during the backward pass, but every segment's
    # auxiliary state is held from the forward pass until its segment is
    # recomputed. How many are held at once is the whole question, and it has
    # two answers:
    #
    # * A run that computes a model gradient really does hold one state per
    #   segment: each is an input to a checkpoint node and autograd keeps it
    #   until that segment is recomputed.
    # * A run that computes no model gradient holds exactly **one**. There is
    #   no checkpointing to do, so the segmented driver threads a single state
    #   forward and drops the previous one. Multiplying by the segment count
    #   there was the single largest error in this estimate -- it grew without
    #   limit against a measured peak that stays flat.
    #
    # ``auxiliary_state_elements`` is a lower bound on one state (see
    # :func:`_auxiliary_state_elements`), so the product is a lower bound too.
    auxiliary_state_bytes = 0
    if n_segments is not None and n_segments > 1:
        backward_storage_bytes //= n_segments
        computes_gradient = n_storage_slots > 0 or n_model_grads > 0
        n_states_held = n_segments if computes_gradient else 1
        auxiliary_state_bytes = (
            n_states_held * n_shots * auxiliary_state_elements * dtype_size
        )

    total_bytes_lower_bound = (
        forward_wavefields_bytes
        + backward_storage_bytes
        + gradient_scratch_bytes
        + auxiliary_state_bytes
    )

    return {
        "padded_model_shape": padded_shape,
        "padded_model_elements": padded_elements,
        "n_threads": n_threads,
        "forward_wavefields_bytes": forward_wavefields_bytes,
        "backward_storage_bytes": backward_storage_bytes,
        "gradient_scratch_bytes": gradient_scratch_bytes,
        "auxiliary_state_bytes": auxiliary_state_bytes,
        "total_bytes_lower_bound": total_bytes_lower_bound,
    }

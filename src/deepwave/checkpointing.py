"""Segmented propagation helpers shared by Deepwave's checkpointing features.

This module adds an *additive* storage strategy for the backward pass of the
Deepwave propagators: instead of storing a forward-wavefield snapshot at every
time step (the ``storage_mode="device"`` / ``"cpu"`` / ``"disk"`` behaviour),
the propagation is split into ``n_segments`` contiguous segments in time and
each segment is threaded to the next via ``initial_state``. Wrapping each
segment in ``torch.utils.checkpoint`` (reentrant mode) -- which
:func:`run_segments` does for you with ``checkpoint=True`` --
recomputes a segment's forward pass on demand during the backward pass instead
of keeping every segment's activations alive at once, trading extra forward
compute for a large reduction in backward-pass memory: the classic time/memory
trade-off that makes single-device 3-D FWI feasible when ``nt`` is large. See
``docs/example_checkpointing.rst`` for the worked recipe and
:func:`run_segments`'s docstring for the underlying driver.

Exactness
---------
A run split with :func:`run_segments` uses the same whole-trace source
upsampling and receiver downsampling as the equivalent combined run, and
:func:`snapshots` inherits that. Forward values and gradients agree within
floating-point rounding for the same configuration. Bitwise equality is not
unconditional: segment boundaries and checkpoint recomputation can change
the order in which floating-point contributions accumulate. An omitted
``max_vel`` is resolved by the propagator's own preparation
before the whole record is resampled, including heterogeneous and attenuation
CFL bounds. :func:`deepwave.wavefields.measure` still requires
``step_ratio == 1`` for a segmented measurement, because its streaming DFT is
defined on user-rate samples.

A caller driving segments in its *own* outer loop (calling ``run_segments`` or
a propagator once per segment) must still pre-upsample the source itself, with
:func:`deepwave.discretisation.cfl_condition_n` and
:func:`deepwave.discretisation.upsample`, and downsample the concatenated
receiver output at the end (see ``docs/usage.rst``); that bracketing cannot be
done from inside a single segment's call. See :doc:`/design/checkpointing` for
why, and for the alternatives that were considered and rejected.

Public API
----------
* :func:`run_segments` -- the segment driver, with a per-boundary
  ``segment_callback``, optional per-segment ``torch.utils.checkpoint`` wrapping
  (``checkpoint=True``, the standard way to bound backward-pass
  memory), and automatic whole-trace resampling at ``step_ratio > 1``. Reused by
  :func:`snapshots` and :func:`deepwave.wavefields.measure`. Under
  ``checkpoint=True`` the PML memory-variable fields of the state
  threaded between segments are stored as their two per-axis boundary slabs
  rather than full-size, which shrinks the dominant
  ``n_segments x auxiliary-state`` term of checkpointed peak memory without
  changing any value or gradient.
* :func:`snapshots` -- differentiable intermediate wavefield snapshots at chosen
  outer steps, threaded through the autograd graph. Returns a
  :class:`SnapshotResult`.
"""

import contextlib
import dataclasses
import importlib
import inspect
import operator
from collections.abc import Container, Iterator, Mapping, Sequence
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import torch
import torch.utils.checkpoint

from deepwave._common import (
    _infer_pml_freq,
    _InferredPmlFreq,
    _propagator_user_model_names,
)
from deepwave._common import (
    active_source_amplitudes as _active_source_amplitudes,
)
from deepwave._common import (
    cfl_condition_n as _cfl_condition_n,
)
from deepwave._common import (
    downsample as _downsample,
)
from deepwave._common import (
    fit_source_amplitudes_to_nt as _fit_source_amplitudes_to_nt,
)
from deepwave._common import (
    set_accuracy as _set_accuracy,
)
from deepwave._common import (
    set_grid_spacing as _set_grid_spacing,
)
from deepwave._common import (
    set_nt as _set_nt,
)
from deepwave._common import (
    set_pml_width as _set_pml_width,
)
from deepwave._common import (
    upsample as _upsample,
)
from deepwave._common import (
    warn_pml_freq_migration as _warn_pml_freq_migration,
)
from deepwave._engine import (
    AuxiliaryState as _AuxiliaryState,
)
from deepwave._engine import (
    ContinuationOriginError as _ContinuationOriginError,
)
from deepwave._engine import (
    PropagatorSpec as _PropagatorSpec,
)
from deepwave._engine import (
    _reject_legacy_state_keys,
    _stored_region_text,
)
from deepwave._engine import (
    auxiliary_state_of as _auxiliary_state_of,
)
from deepwave._engine import (
    collapse_auxiliary_state_values as _collapse_auxiliary_state_values,
)
from deepwave._engine import (
    expand_channel_mappings as _expand_channel_mappings,
)
from deepwave._engine import (
    flat_state as _flat_state,
)
from deepwave._engine import (
    inherited_max_vel as _inherited_max_vel,
)
from deepwave._engine import (
    inherited_pml_freq as _inherited_pml_freq,
)
from deepwave._engine import (
    receiver_output_names as _receiver_output_names,
)
from deepwave._exceptions import DeepwaveRuntimeError as _DeepwaveRuntimeError
from deepwave._exceptions import (
    DeepwaveTypeError as _DeepwaveTypeError,
)
from deepwave._exceptions import (
    DeepwaveValueError as _DeepwaveValueError,
)
from deepwave._storage import (
    STORAGE_DEFAULT as _STORAGE_DEFAULT,
)
from deepwave._storage import (
    resolve_storage_arguments as _resolve_storage_arguments,
)
from deepwave.types import (
    InitialState as _InitialState,
)
from deepwave.types import (
    PropagatorOutput as _PropagatorOutput,
)
from deepwave.types import (
    StorageLike as _StorageLike,
)
from deepwave.types import (
    StorageMode as _StorageMode,
)
from deepwave.types import (
    StoragePath as _StoragePath,
)

__all__ = ["SnapshotResult", "run_segments", "snapshots"]


class SnapshotResult(NamedTuple):
    """The result of :func:`deepwave.checkpointing.snapshots`.

    A named two-tuple, so ``combined, snaps = snapshots(...)`` and
    ``result.snapshots`` both read naturally.

    Stability: provisional (see :doc:`/stability`).

    Attributes:
        output: The whole propagation's
            :class:`~deepwave.PropagatorOutput`, as a direct call would
            return it.
        snapshots: A mapping from each requested outer step to that step's
            ``{wavefield_name: tensor}`` wavefield state.

    """

    output: _PropagatorOutput
    snapshots: dict[int, dict[str, torch.Tensor]]


# The names run_segments() exposes as explicit parameters of its own, which
# therefore must not be smuggled through the forwarded **kwargs of a caller
# that wraps it (snapshots(), deepwave.wavefields.measure()).
_SEGMENT_DRIVER_RESERVED: tuple[str, ...] = (
    "checkpoint",
    "checkpoint_device",
    "use_reentrant",
)

#: The channel-keyed survey mappings a segment driver absorbs rather than
#: forwards. A driver slices the source amplitudes per segment itself, from
#: its own ``source_amplitudes`` mapping; a ``sources=`` mapping (or a flat
#: ``source_amplitudes*`` keyword) forwarded to every segment whole would make
#: every segment inject the entire record -- silently wrong results with no
#: exception anywhere. So they are not forwarded: they are resolved into the
#: flat keywords here, and the amplitudes are lifted into the mapping the
#: driver slices, which is what makes every call form the propagator accepts
#: work here too.
_SEGMENT_DRIVER_SURVEY_RESERVED: tuple[str, ...] = (
    "sources",
    "receivers",
    "bg_receivers",
)


def _absorb_survey_kwargs(
    propagator: Callable[..., Any],
    kwargs: dict[str, Any],
    source_amplitudes: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Resolves a survey given in the propagator's own call forms.

    A segment driver's ``source_amplitudes=`` mapping is the *driver's*
    spelling of the survey, because the driver is what slices it. That is no
    reason for the propagator's own spellings to be refused: a caller who
    writes ``sources={"p": (amp, loc)}`` for a direct call should be able to
    write it here, and a caller who writes the flat
    ``source_amplitudes_p=``/``source_locations_p=`` keywords should too.

    Both are accepted by resolving them the way the propagator itself would
    -- through :func:`deepwave._engine.expand_channel_mappings`, so the
    duplicate-spelling and unknown-channel errors are the ones a direct call
    gives -- and then lifting the resulting amplitudes out of the forwarded
    keywords into the mapping the driver slices. What each segment receives
    is therefore always the flat locations plus its own slice of the
    amplitudes, exactly as before; only the spellings the caller may use
    have widened.

    Args:
        propagator: The propagator being segmented, whose spec names its
            channels.
        kwargs: The keywords destined for each segment. Modified in place:
            the mappings are removed, their flat equivalents added, and the
            amplitude entries removed.
        source_amplitudes: The driver's own amplitude mapping.

    Returns:
        The amplitude mapping the driver should slice: ``source_amplitudes``
        plus any amplitudes recovered from ``kwargs``, fitted to the whole-run
        ``nt`` if supplied. That keyword is consumed here so it cannot pad or
        truncate each segment separately.

    Raises:
        DeepwaveValueError: If a channel is given in both forms.
        DeepwaveTypeError: If a mapping value has the wrong shape.

    """
    mappings = {
        name: kwargs.pop(name)
        for name in _SEGMENT_DRIVER_SURVEY_RESERVED
        if name in kwargs
    }
    if any(value is not None for value in mappings.values()):
        _expand_channel_mappings(_get_spec(propagator), kwargs, **mappings)
    amplitudes = dict(source_amplitudes)
    for name in list(kwargs):
        if not name.startswith("source_amplitudes"):
            continue
        value = kwargs.pop(name)
        if value is None:
            continue
        if name in amplitudes:
            raise _DeepwaveValueError(
                f"{name} was given both as {name}= and as an entry of "
                "source_amplitudes=. Give it once."
            )
        amplitudes[name] = value
    nt = kwargs.pop("nt", None)
    if nt is not None:
        nt = _set_nt(nt, list(amplitudes.values()), 1)
        fitted = _fit_source_amplitudes_to_nt(list(amplitudes.values()), nt)
        amplitudes = {
            name: cast("torch.Tensor", value)
            for name, value in zip(amplitudes, fitted)
        }
    return amplitudes


#: Keywords the segment driver used to have, and what each is spelled now.
#: Without an explicit check these fall through ``**kwargs`` to the
#: propagator, which never had any of them, so the caller is told that the
#: *propagator* rejected an argument they typed on ``run_segments`` -- and is
#: told nothing about the replacement.
_SEGMENT_DRIVER_RENAMED: dict[str, str] = {
    "callback": "segment_callback",
    "checkpoint_use_reentrant": "use_reentrant",
    "slab_pml": "_slab_pml",
    "checkpoint_slab_pml": "_slab_pml",
}


def _reject_renamed_kwargs(kwargs: Mapping[str, Any], func_name: str) -> None:
    """Raises for a keyword this function used to have, naming its replacement.

    Also refuses ``checkpoint_segments=``, which is not a rename but a
    collision: it is the propagator's own spelling of the segmentation this
    function is performing, and forwarding it would nest one segmentation
    inside the other.

    Args:
        kwargs: The keywords destined for the propagator.
        func_name: The name to blame, which is this function rather than the
            propagator.

    Raises:
        DeepwaveTypeError: If any retired or colliding keyword is present.

    """
    for old, new in _SEGMENT_DRIVER_RENAMED.items():
        if old in kwargs:
            raise _DeepwaveTypeError(
                f"{func_name}() has no {old}= argument: it is now {new}=. "
                f"Pass {new}= instead. (The propagator has no {old}= either, "
                "so leaving this to be forwarded would blame the propagator "
                "for a keyword you typed here.)"
            )
    if "checkpoint_segments" in kwargs:
        raise _DeepwaveTypeError(
            f"{func_name}() has no checkpoint_segments= argument: it is "
            "already the segment driver, and the number of segments is "
            "n_segments= (or seg_lengths=). The propagator does have "
            "checkpoint_segments=, so forwarding this would make every "
            f"segment of the {func_name}() run segment itself again -- "
            "n_segments=a with checkpoint_segments=b would quietly run a*b "
            "segments. Pass the total segment count as n_segments=."
        )


# ---------------------------------------------------------------------------
# Propagator introspection
# ---------------------------------------------------------------------------


def _get_spec(propagator: Callable[..., Any]) -> _PropagatorSpec:
    """Returns the :class:`PropagatorSpec` for a public propagator function.

    Deepwave-internal API consumed across modules (checkpointing.py,
    wavefields.py); not a public interface.
    """
    module = importlib.import_module(propagator.__module__)
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, _PropagatorSpec):
        raise _DeepwaveTypeError(
            f"{propagator!r} does not look like a Deepwave propagator function "
            "(its module has no PropagatorSpec named SPEC). checkpoint(), "
            "snapshots(), run_segments(), and deepwave.wavefields.measure() "
            "operate on any Deepwave propagator whose module exposes one -- "
            "currently every propagator deepwave provides."
        )
    return spec


def _ndim_of(
    model_args: Sequence[torch.Tensor],
    kwargs: Optional[Mapping[str, Any]] = None,
) -> int:
    """Number of spatial dimensions of the propagation.

    Determined the way the propagators themselves determine it (see
    :func:`deepwave._common.get_ndim`): from the trailing dimension of a
    location argument (``source_locations``, ``receiver_locations_y``, ...) in
    ``kwargs``. A model tensor cannot be used, because the propagators accept
    models that are either unbatched (``ndim`` dimensions, shared by every shot)
    or batched per shot (``ndim + 1`` dimensions), so a three-dimensional model
    may equally be a batched 2-D model or an unbatched 3-D one. A
    :class:`deepwave.Hicks` interpolator passed in place of a location tensor
    carries the same information in its interpolated locations, as it does for
    the propagators. ``grid_spacing`` cannot be used either, as it may be a
    single value (meaning equal spacing in every dimension), which would be
    indistinguishable from a 1-D grid.

    The model tensors are only used when no location argument is provided, in
    which case they are taken to be unbatched: a batched model without any
    location argument could not be propagated at all, since the propagators
    would themselves be unable to determine the number of dimensions.

    Deepwave-internal API consumed across modules (checkpointing.py,
    wavefields.py); not a public interface.
    """
    if not model_args:
        raise _DeepwaveValueError("at least one model tensor is required.")
    for name, value in (kwargs or {}).items():
        if "locations" not in name or value is None:
            continue
        locations = value
        if not isinstance(locations, torch.Tensor):
            get_locations = getattr(locations, "get_locations", None)
            if not callable(get_locations):
                continue
            locations = get_locations()
        if locations.dim() > 0:
            return int(locations.shape[-1])
    return model_args[0].dim()


def _segment_lengths(nt: int, n_segments: int) -> list[int]:
    """Splits ``nt`` time steps into ``n_segments`` near-equal contiguous parts.

    The first ``nt % n_segments`` segments are one step longer. Because
    continuation is bit-exact at any boundary when ``step_ratio == 1`` -- which
    :func:`run_segments` arranges for every segment, by driving them at the
    internal time step when the user ``dt`` needs sub-stepping -- the split need
    not be even.
    """
    if n_segments < 1:
        raise _DeepwaveValueError(f"n_segments must be >= 1, got {n_segments}.")
    if n_segments > nt:
        raise _DeepwaveValueError(
            f"n_segments ({n_segments}) cannot exceed the number of time steps "
            f"({nt}); each segment must contain at least one step."
        )
    base, extra = divmod(nt, n_segments)
    return [base + (1 if i < extra else 0) for i in range(n_segments)]


def _nt_of(source_amplitudes: Mapping[str, torch.Tensor]) -> int:
    """Number of time steps shared by every source-amplitude tensor.

    All source-amplitude tensors define the propagation length through their last
    (time) dimension and must agree; raises a clear error if they do not. Callers
    guarantee ``source_amplitudes`` is non-empty (they validate that first).

    Deepwave-internal API consumed across modules (checkpointing.py,
    wavefields.py); not a public interface.
    """
    nts = {name: t.shape[-1] for name, t in source_amplitudes.items()}
    nt = next(iter(nts.values()))
    if any(v != nt for v in nts.values()):
        raise _DeepwaveValueError(
            f"all source-amplitude tensors must share the same time length; got {nts}."
        )
    return nt


# Per-caller remedy clause for _validate_step_ratio's ValueError, appended
# after "Reduce dt (...)". Each names the actual alternative available to
# that caller's own users -- wavefields.measure() hard-codes
# storage_mode="device" internally, so recommending a storage_mode change
# would be meaningless for it.
_STEP_RATIO_REMEDIES: dict[str, str] = {
    "measure": (
        "to use a segmented measurement, or pass n_segments=1 for a "
        "receiver-only measurement (a single-segment run has no step_ratio "
        "restriction)"
    ),
}


def _validate_step_ratio(
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    max_vel: Optional[float],
    courant_number: Optional[float],
    func_name: str,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Raises unless ``step_ratio == 1`` for this call (see module exactness note).

    ``max_vel`` must already be resolved by the propagator's own preparation.
    Raw material magnitudes are not a wavespeed bound: their units differ,
    and a weighted heterogeneous stencil can exceed pointwise speeds.

    ``func_name`` is the caller's own name (currently only ``"measure"``), used
    both to name it in the message and to select an appropriate remedy from
    :data:`_STEP_RATIO_REMEDIES` (see its comment).

    ``kwargs`` are the propagator keyword arguments, used only to determine the
    number of spatial dimensions (see :func:`_ndim_of`), which sets how a single
    ``grid_spacing`` value is expanded and hence the CFL bound.

    Deepwave-internal API consumed across modules (checkpointing.py,
    wavefields.py); not a public interface.
    """
    ndim = _ndim_of(model_args, kwargs)
    gs = _set_grid_spacing(grid_spacing, ndim)
    if max_vel is None:
        raise _DeepwaveValueError("Resolve max_vel before validating segmentation.")
    mv = float(max_vel)
    _, step_ratio = _cfl_condition_n(gs, dt, mv, courant_number=courant_number)
    if step_ratio != 1:
        remedy = _STEP_RATIO_REMEDIES.get(
            func_name,
            "to avoid internal sub-stepping",
        )
        raise _DeepwaveValueError(
            f"{func_name}() requires "
            f"step_ratio == 1, but this call has step_ratio == {step_ratio} "
            "(the user dt is being internally sub-stepped to satisfy the CFL "
            "condition). Segmented propagation only reproduces an unsegmented "
            "run -- forward values and gradients alike -- when step_ratio == "
            "1, because the source amplitudes are resampled to the internal time "
            "step with a global FFT that does not decompose across segment "
            "boundaries. Reduce dt (so that dt <= courant_number * min(dx) / "
            f"max_vel and no sub-stepping is needed) {remedy}."
        )


# ---------------------------------------------------------------------------
# Segment driver (forward-only), reusable by snapshot features
# ---------------------------------------------------------------------------


def _canonical_source_order(
    propagator: Callable[..., Any],
    source_amplitudes: Mapping[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Orders the source-amplitude tensors by the propagator's signature.

    ``pml_freq`` inference sums one normalised spectrum per source channel in
    float64, so the order the channels are visited in is (at the last bit) part
    of the answer. A ``dict`` preserves the order the *caller* happened to
    build it in, which is not a property of the propagator; ordering by the
    propagator's own parameter declaration makes the result depend only on
    which channels are present. Anything the signature does not name (there
    should be nothing) keeps its original position at the end.

    Args:
        propagator: The propagator being segmented.
        source_amplitudes: Mapping from source-amplitude argument name to
            tensor.

    Returns:
        The tensors, in canonical order.

    """
    try:
        declared = list(inspect.signature(propagator).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        declared = []
    position = {name: i for i, name in enumerate(declared)}
    names = sorted(source_amplitudes, key=lambda n: position.get(n, len(position)))
    return [source_amplitudes[name] for name in names]


def _active_source_order(
    propagator: Callable[..., Any],
    source_amplitudes: Mapping[str, torch.Tensor],
    kwargs: Mapping[str, Any],
) -> list[Optional[torch.Tensor]]:
    """Use the direct call's active, unspread sources for PML inference."""
    try:
        declared = list(inspect.signature(propagator).parameters)
    except (TypeError, ValueError):
        declared = []
    position = {name: i for i, name in enumerate(declared)}
    names = sorted(source_amplitudes, key=lambda n: position.get(n, len(position)))
    amplitudes = [source_amplitudes[name] for name in names]
    locations = [
        kwargs.get(name.replace("source_amplitudes", "source_locations", 1))
        for name in names
    ]
    active = _active_source_amplitudes(amplitudes, locations)
    for i, name in enumerate(names):
        mask = kwargs.get(name.replace("source_amplitudes", "source_mask", 1))
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise _DeepwaveValueError("source masks must have dtype torch.bool.")
            amplitude = active[i]
            if amplitude is not None:
                active[i] = amplitude.masked_fill(
                    ~mask.to(amplitude.device).unsqueeze(-1), 0
                )
    return active


def _split_sources(
    source_amplitudes: Mapping[str, torch.Tensor], seg_lengths: Sequence[int]
) -> list[dict[str, torch.Tensor]]:
    """Slices each source tensor along its last (time) dimension per segment."""
    per_segment: list[dict[str, torch.Tensor]] = [{} for _ in range(len(seg_lengths))]
    for name, tensor in source_amplitudes.items():
        chunks = torch.split(tensor, list(seg_lengths), dim=-1)
        for i, chunk in enumerate(chunks):
            per_segment[i][name] = chunk
    return per_segment


@dataclasses.dataclass(frozen=True)
class _SegmentResampling:
    """Resolved whole-trace resampling for a multi-segment run at step_ratio > 1.

    ``dt`` is the internal time step each segment is driven with
    (user dt / step_ratio); the taper fields mirror the propagator kwargs of
    the same names and are applied to both the one-time source upsample and
    the one-time receiver downsample, exactly as a direct call applies them
    to its own internal resampling.
    """

    dt: float
    step_ratio: int
    freq_taper_frac: float
    time_pad_frac: float
    time_taper: bool

    def upsample(self, signal: torch.Tensor) -> torch.Tensor:
        """Upsamples a source tensor from the user rate to the internal rate."""
        if signal.numel() == 0:
            # upsample() returns empty tensors unchanged, but the time length
            # must still scale so per-segment splitting stays consistent.
            return signal.new_zeros(
                *signal.shape[:-1], signal.shape[-1] * self.step_ratio
            )
        return _upsample(
            signal,
            self.step_ratio,
            freq_taper_frac=self.freq_taper_frac,
            time_pad_frac=self.time_pad_frac,
            time_taper=self.time_taper,
        )

    def downsample(self, signal: torch.Tensor) -> torch.Tensor:
        """Downsamples a receiver tensor from the internal rate to the user rate."""
        if signal.numel() == 0:
            return signal
        return _downsample(
            signal,
            self.step_ratio,
            freq_taper_frac=self.freq_taper_frac,
            time_pad_frac=self.time_pad_frac,
            time_taper=self.time_taper,
        )


def _resolve_segment_resampling(
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    ndim: int,
    n_segments: int,
    kwargs: Mapping[str, Any],
) -> Optional[_SegmentResampling]:
    """Resolves whole-trace resampling for a segmented run, or None if not needed.

    Returns ``None`` when the run needs no internal sub-stepping (``step_ratio
    == 1``, so each segment's own resampling is the identity and segmentation is
    already exact) or when there is only one segment (a single segment is a
    plain propagator call, whose own internal resampling is exact -- there is no
    segment boundary to protect).

    ``max_vel`` is resolved from the whole call before this helper runs, so
    the resampling uses exactly the speed chosen by the propagator.

    Deepwave-internal helper; not a public interface.
    """
    if n_segments < 2:
        return None
    gs = _set_grid_spacing(grid_spacing, ndim)
    max_vel = kwargs.get("max_vel")
    courant_number = kwargs.get("courant_number")
    if max_vel is None:
        raise _DeepwaveValueError("Resolve max_vel before segment resampling.")
    speed = abs(float(max_vel))
    inner_dt, step_ratio = _cfl_condition_n(
        gs, dt, speed, courant_number=courant_number
    )
    if step_ratio == 1:
        return None
    _, check = _cfl_condition_n(gs, inner_dt, speed, courant_number=courant_number)
    if check != 1:  # pragma: no cover - floating-point edge case
        raise _DeepwaveRuntimeError(
            "the internally computed dt / step_ratio "
            f"({dt} / {step_ratio}) still fails the CFL check (it would itself "
            f"need step_ratio == {check}), which can only happen through "
            "floating-point rounding. Reduce dt slightly."
        )
    return _SegmentResampling(
        dt=inner_dt,
        step_ratio=step_ratio,
        freq_taper_frac=float(kwargs.get("freq_taper_frac", 0.0)),
        time_pad_frac=float(kwargs.get("time_pad_frac", 0.0)),
        time_taper=bool(kwargs.get("time_taper", False)),
    )


def _output_from_flat(
    spec: _PropagatorSpec,
    ndim: int,
    values: Mapping[str, torch.Tensor],
    fingerprint: Optional[_AuxiliaryState],
    info: Optional[Any],
    n_mechanisms: int = 1,
) -> _PropagatorOutput:
    """Rebuilds a public :class:`PropagatorOutput` from flat state tensors.

    The inverse of :func:`deepwave._engine.flat_state`: ``values`` is keyed by
    the internal wavefield names plus the receiver names, and the result is the
    propagator's public tuple -- in public order, under the public spellings --
    with the auxiliary-state block folded back into a single
    :class:`AuxiliaryState` carrying ``fingerprint``'s recorded run parameters.
    """
    state_names = spec.public_state_names(ndim, n_mechanisms)
    internal = spec.internal_state_name_map(ndim, n_mechanisms)
    folded = _collapse_auxiliary_state_values(
        state_names,
        [values[internal[n]] for n in state_names],
        spec._names_for_mechanisms(spec.auxiliary_state_names, ndim, n_mechanisms),  # noqa: SLF001
        lambda tensors: _AuxiliaryState._from_flat(tensors, fingerprint),  # noqa: SLF001
    )
    names = spec.output_names(ndim)
    folded.extend(values[n] for n in names[len(folded) :])
    return _PropagatorOutput(names, folded, info=info)


def _check_caller_state_keys(
    supplied: Mapping[str, Any],
    caller_state: Any,
    spec: _PropagatorSpec,
    state_names: Sequence[str],
    boundary_internal: Sequence[str],
    public_state_names: Mapping[str, str],
    receiver_names: Container[str],
) -> None:
    """Applies a propagator's ``initial_state`` key rules to a caller's bundle.

    The checkpointed branch of :func:`run_segments` unpacks the caller's
    ``initial_state`` into positional tensors itself, so the propagator never
    sees the mapping and cannot apply its own checks to it. This applies them
    here, with the same messages, so that a direct call,
    ``checkpoint=False`` and ``checkpoint=True`` all give the
    same answer about what is a valid key: an unrecognised one is a typo and
    raises, a pre-fold auxiliary-field spelling is deprecated on the released
    propagators and rejected on the rest, and naming a field both inside
    ``auxiliary_state`` and beside it is ambiguous.

    ``supplied`` is the caller's own mapping with the ``"auxiliary_state"``
    entry removed; ``caller_state`` is that entry (or ``None``).
    """
    valid = [public_state_names[n] for n in state_names if n not in boundary_internal]
    if boundary_internal:
        valid.append("auxiliary_state")
    boundary_public = {public_state_names[n] for n in boundary_internal}
    known = set(public_state_names.values())
    if isinstance(caller_state, _AuxiliaryState):
        for name in supplied:
            if name in caller_state:
                raise _DeepwaveValueError(
                    f"{name!r} was given both inside initial_state's "
                    "auxiliary_state and as a separate initial_state key; "
                    "provide it in only one place."
                )
    for name in supplied:
        if name not in known and name not in receiver_names:
            raise _DeepwaveValueError(
                f"initial_state contains unexpected key {name!r}; the valid "
                f"initial-state names are: {', '.join(valid)}."
            )
    legacy = [
        name
        for name, value in supplied.items()
        if name in boundary_public and value is not None
    ]
    if legacy:
        _reject_legacy_state_keys(legacy, spec.name)


def _combine_outputs(
    spec: _PropagatorSpec,
    ndim: int,
    final_output: _PropagatorOutput,
    receiver_chunks: list[dict[str, torch.Tensor]],
    resampling: Optional[_SegmentResampling] = None,
    nt: Optional[int] = None,
    requested_model_gradient_sampling_interval: int = 1,
) -> _PropagatorOutput:
    """Builds a combined :class:`PropagatorOutput` for the whole propagation.

    The wavefield (state) fields come from the last segment's output; the
    receiver-amplitude fields are the time-concatenation of every segment's
    receiver output. If ``resampling`` is given (the segments ran at the
    internal time step), the concatenated receiver trace is downsampled once,
    as a whole, back to the user rate -- mirroring what a direct call does to
    its own internal-rate receiver trace. State fields are never resampled.

    ``nt`` is the whole propagation's user-level step count. The metadata is
    otherwise the last segment's, which is right for every field of it except
    ``nt``: that one describes a segment, not the record, and would make
    ``out.info.nt`` disagree with the length of ``out``'s own receiver
    amplitudes. ``pml_freq`` needs no such correction -- :func:`run_segments`
    resolves it once from the whole source record and pins every segment to
    it, so the last segment's is the record's.
    """
    names = spec.output_names(ndim)
    receiver_names = _receiver_output_names(spec)
    final_map = final_output.as_dict()
    values: list[Any] = []
    for name in names:
        if name not in receiver_names:
            # Every non-receiver field -- including the single, opaque
            # ``auxiliary_state`` -- is the last segment's, unresampled.
            values.append(final_map[name])
        else:
            chunks = [rc[name] for rc in receiver_chunks]
            cat = torch.cat(chunks, dim=-1)
            if cat.numel() == 0:
                # A channel with no receivers records nothing, but it still
                # reports the record's shape: resampling would leave the
                # concatenated (internal-rate) length in place, which would
                # disagree with the requested channels beside it.
                values.append(
                    cat.reshape(cat.shape[0], 0, cat.shape[-1] if nt is None else nt)
                )
            else:
                values.append(cat if resampling is None else resampling.downsample(cat))
    info = final_output.info
    if info is not None and nt is not None:
        if resampling is None:
            info = dataclasses.replace(info, nt=nt)
        else:
            info = dataclasses.replace(
                info,
                nt=nt,
                dt=info.internal_dt * resampling.step_ratio,
                step_ratio=resampling.step_ratio,
                model_gradient_sampling_interval=requested_model_gradient_sampling_interval,
            )
    return _PropagatorOutput(names, values, info=info)


def _reject_storage_mode_kwarg(
    kwargs: dict[str, Any],
    func_name: str,
    names: Sequence[str] = ("storage_mode",),
) -> None:
    """Raises if any of ``names`` is present in ``kwargs``.

    ``func_name`` already controls these names itself (as an explicit
    parameter, or set internally); accepting one again from ``**kwargs``
    would be ambiguous, so every entry point that forwards arbitrary
    ``**kwargs`` to a propagator rejects them up front with this one shared
    message. ``names`` defaults to just ``"storage_mode"``; ``run_segments()``
    additionally reserves ``storage_path`` and ``storage_compression`` (it
    exposes all three as explicit parameters), so it passes all three here,
    while ``snapshots()``/``measure()`` add
    :data:`_SEGMENT_DRIVER_RESERVED` (the checkpointing parameters
    ``run_segments()`` names, which they set themselves) to theirs.
    """
    for name in names:
        if name in kwargs:
            raise _DeepwaveValueError(
                f"{func_name}() controls {name!r}; do not pass it in kwargs."
            )


# ---------------------------------------------------------------------------
# Slab storage of PML memory variables in the threaded auxiliary state
# ---------------------------------------------------------------------------
#
# A PML memory variable (scalar's ``psi``/``zeta``, acoustic's ``phi``/``psi``,
# elastic's ``m_*``, ...) is exactly zero outside two thin slabs at the two
# ends of one spatial axis, so only those slabs need to cross a checkpoint
# boundary. Threading the slabs instead of the whole field cuts the
# ``n_segments x auxiliary-state`` term of checkpointed peak memory by a factor
# that grows with grid size, and it is bitwise-exact. The argument, with K the
# kept region (per-axis slabs of thickness ``pml_width_side + accuracy`` per
# side, in "pml view" coordinates -- model plus PML, finite-difference padding
# removed) and D its complement:
#
# Forward, bitwise. A segment's output field is built from buffers seeded by
# the re-padded threaded input and written by the compiled kernel. The
# propagator's ``zero_wavefields`` hook is called on every forward entry
# (``_engine.run_forward``) and zeroes each declared field's interior along its
# own axis, i.e. everything beyond ``pml_width`` from that axis's view edge, so
# no interior value from the input survives; any ping-pong partner buffer is
# allocated with ``zeros_like``. The kernel writes the field only inside that
# axis's PML index range (``_engine._pml_region``), whose view-coordinate
# support is at most ``pml_width_side + accuracy`` per side over both grid
# types and both directions -- exactly K. Values on D are therefore exactly
# ``0.0``, so reconstructing them as exact zeros reproduces the uncompressed
# tensor bitwise, and every downstream forward value with it.
#
# Backward, bitwise. Forwards being bitwise-identical, the incoming gradient
# ``g`` w.r.t. a segment's output field is the same in both paths. The
# reconstruction node's backward forwards ``g|K`` and drops ``g|D``, which
# changes nothing, for two independent reasons: ``g`` is the adjoint field
# returned by the next segment's backward, whose support is likewise within K
# (the backward PML ranges; and the PML ``a``/``b`` profiles are exactly zero
# outside the PML, so even the ungated adjoint writes produce zeros there); and
# the segment's own backward seeds its adjoint fields with ``g`` and
# immediately applies ``zero_wavefields`` (``_engine.run_backward_core``),
# which zeroes a superset of D, so ``g|D`` could not reach any leaf gradient
# anyway. Reconstruction uses ``new_zeros`` + ``cat`` with no detaching, so the
# recompute chain stays an ordinary differentiable graph.
#
# The invariant is a property of the compiled backend under standard per-axis
# PML, so :func:`_build_slab_plan` refuses to compress unless the propagator
# declares ``PropagatorSpec.pml_slab_axes``, the compiled backend is the one
# being used, ``mpml_fraction`` is zero (an M-PML memory variable is legitimately
# nonzero wherever *any* axis is in its PML), and ``boundary`` is the default.
#
# Residual risk, recorded here because the optimisation is unconditional. The
# invariant is a property of the compiled kernels, not something checked at
# run time: a propagator that declares ``pml_slab_axes`` for a field whose
# kernel writes outside the two slabs, or a kernel change that widens the
# written PML region beyond ``pml_width_side + accuracy``, would silently
# corrupt the threaded state -- wrong numbers, no exception anywhere.
# ``tests/test_checkpointing.py``'s
# ``test_slab_margin_envelope_pins_pml_region`` exists exactly to catch that
# and is the only guard on it; it must keep running.


@dataclasses.dataclass(frozen=True)
class _SlabSpec:
    """Kept-slab geometry for one PML memory-variable state field (pml view)."""

    axis: int  # spatial axis (0 = slowest); tensor dim is axis + 1
    lo: int  # kept thickness at the low end
    hi: int  # kept thickness at the high end
    full: int  # full extent along axis, for validation and reconstruction


def _signature_default(propagator: Callable[..., Any], name: str) -> Any:
    """The default value of ``propagator``'s ``name`` parameter, or ``None``."""
    parameter = inspect.signature(propagator).parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def _build_slab_plan(
    spec: _PropagatorSpec,
    propagator: Callable[..., Any],
    ndim: int,
    kwargs: Mapping[str, Any],
    state_names: Sequence[str],
    state: Mapping[str, torch.Tensor],
) -> Optional[dict[str, _SlabSpec]]:
    """Plans which threaded state fields may be stored as boundary slabs.

    Returns ``None`` -- meaning "thread every field full-size, exactly as
    without this optimisation" -- whenever the zero-outside-the-slabs invariant
    documented above is not established for this run: an undeclared propagator,
    a request for anything but the compiled backend (``"auto"`` counts, since
    on a propagator that has a compiled kernel it resolves to exactly that),
    M-PML, a non-default ``boundary``, or a propagator whose
    ``accuracy``/``pml_width`` cannot be resolved.
    Individual fields whose two slabs would already span the whole axis (tiny
    or heavily clamped geometries, where the PML regions overlap and the kernel
    may write anywhere) are left out of the plan while the rest still compress.
    """
    if spec.pml_slab_axes is None or not spec.has_compiled_backend:
        return None
    if kwargs.get("backend", "auto") not in ("auto", "compiled"):
        return None
    if kwargs.get("python_backend") not in (None, False):
        return None
    if kwargs.get("boundary") is not None:
        return None
    if float(kwargs.get("mpml_fraction") or 0.0) != 0.0:
        return None
    accuracy_val = kwargs.get("accuracy", _signature_default(propagator, "accuracy"))
    pml_val = kwargs.get("pml_width", _signature_default(propagator, "pml_width"))
    if accuracy_val is None or pml_val is None:
        return None
    accuracy = _set_accuracy(accuracy_val)
    pml = _set_pml_width(pml_val, ndim)
    axes = spec.pml_slab_axes(ndim)
    plan: dict[str, _SlabSpec] = {}
    for name in state_names:
        axis = axes.get(name)
        if axis is None:
            continue
        full = state[name].shape[1 + axis]
        lo = pml[2 * axis] + accuracy
        hi = pml[2 * axis + 1] + accuracy
        if lo + hi >= full:
            continue
        plan[name] = _SlabSpec(axis=axis, lo=lo, hi=hi, full=full)
    return plan or None


def _compress_state(
    state_tensors: tuple[Optional[torch.Tensor], ...],
    state_names: Sequence[str],
    plan: Mapping[str, _SlabSpec],
) -> tuple[Optional[torch.Tensor], ...]:
    """Replaces each planned state field by its two concatenated end slabs.

    ``torch.cat`` allocates fresh contiguous storage, so the full-size tensor
    is genuinely released rather than kept alive by a view -- which is the
    whole point, since reentrant checkpointing saves its tensor inputs.
    """
    compressed: list[Optional[torch.Tensor]] = []
    for name, tensor in zip(state_names, state_tensors):
        slab = plan.get(name)
        if tensor is None or slab is None:
            compressed.append(tensor)
            continue
        dim = slab.axis + 1
        if tensor.shape[dim] != slab.full:
            raise AssertionError(
                f"state field {name!r} has extent {tensor.shape[dim]} along "
                f"dim {dim}, but the slab plan was built for {slab.full}."
            )
        compressed.append(
            torch.cat(
                [
                    tensor.narrow(dim, 0, slab.lo),
                    tensor.narrow(dim, slab.full - slab.hi, slab.hi),
                ],
                dim=dim,
            )
        )
    return tuple(compressed)


def _decompress_state(
    state: dict[str, torch.Tensor],
    plan: Mapping[str, _SlabSpec],
) -> dict[str, torch.Tensor]:
    """Rebuilds full-size state fields from their end slabs, zero-filling the gap."""
    for name, slab in plan.items():
        compressed = state.get(name)
        if compressed is None:
            continue
        dim = slab.axis + 1
        middle = list(compressed.shape)
        middle[dim] = slab.full - slab.lo - slab.hi
        state[name] = torch.cat(
            [
                compressed.narrow(dim, 0, slab.lo),
                compressed.new_zeros(middle),
                compressed.narrow(dim, slab.lo, slab.hi),
            ],
            dim=dim,
        )
    return state


def _suggest_aligned_n_segments(nt: int, mgsi: int, requested: int) -> Optional[int]:
    """A segment count whose near-equal split is all multiples of ``mgsi``.

    :func:`_segment_lengths` gives ``k`` lengths that differ by at most one
    step, so with ``mgsi > 1`` they can only all be multiples of ``mgsi`` if
    they are all equal, i.e. ``k`` divides ``nt`` and ``mgsi`` divides
    ``nt // k``. Both hold exactly when ``k`` divides ``nt // mgsi``, so the
    candidates are the divisors of that quotient; the one nearest
    ``requested`` is returned (ties go to the smaller, i.e. the longer
    segments). ``None`` means there is no such count -- which happens when
    ``mgsi`` does not divide ``nt`` at all, in which case an unsegmented call
    would be refused too and suggesting a segment count would be wrong.
    """
    if mgsi < 1 or nt % mgsi != 0:
        return None
    quotient = nt // mgsi
    divisors = [k for k in range(1, quotient + 1) if quotient % k == 0]
    if not divisors:  # pragma: no cover - quotient >= 1 always has divisor 1
        return None
    return min(divisors, key=lambda k: (abs(k - requested), k))


def _check_segment_mgsi_alignment(
    lengths: Sequence[int],
    mgsi: int,
    step_ratio: int,
) -> None:
    """Raises unless every segment length is a multiple of ``mgsi``.

    Each segment restarts the model-gradient sampling phase at 0, so a segment
    boundary that falls part-way through a sampling period makes the segments
    sample the gradient integral at different steps from the single call they
    replace. ``lengths`` and ``mgsi`` are both in **user** time steps; when the
    run is internally sub-stepped both are multiplied by ``step_ratio``, which
    leaves the divisibility unchanged, so checking the user-level pair covers
    the internal one too. ``step_ratio`` is named only so the message can say
    what the segments will actually run at.
    """
    if mgsi < 2 or len(lengths) < 2:
        return
    misaligned = next(
        (i for i, length in enumerate(lengths) if length % mgsi != 0), None
    )
    if misaligned is None:
        return
    nt = sum(lengths)
    scaled = (
        ""
        if step_ratio == 1
        else (
            f" (the internal sampling interval is {mgsi * step_ratio} after "
            f"scaling by step_ratio={step_ratio})"
        )
    )
    suggestion = _suggest_aligned_n_segments(nt, mgsi, len(lengths))
    if suggestion is None:
        advice = ""
    else:
        advice = (
            f"Use n_segments={suggestion} (lengths "
            f"{', '.join(str(x) for x in _segment_lengths(nt, suggestion))}) or "
        )
    raise _DeepwaveValueError(
        f"Segment {misaligned + 1} of {len(lengths)} is {lengths[misaligned]} "
        "time steps long, which is not a multiple of "
        f"model_gradient_sampling_interval={mgsi}{scaled}. Each segment "
        "restarts the gradient-sampling phase at 0, so a segment boundary that "
        "splits a sampling period would sample the model gradient at the wrong "
        f"steps. Segment lengths are {', '.join(str(x) for x in lengths)}. "
        f"{advice}pass seg_lengths= with every length a multiple of {mgsi}."
    )


@contextlib.contextmanager
def _segment_region_errors() -> Iterator[None]:
    """Restates a continuation-region error in :func:`run_segments`' own terms.

    Each segment continues the previous one, so :func:`run_segments` builds
    the ``initial_state`` every segment after the first is given. The
    propagator's own message for a region disagreement therefore names an
    ``initial_state`` this caller never passed, and offers a remedy ("pass
    initial wavefields that belong to the region") that a
    :func:`run_segments` caller has no way to act on. The disagreement here
    is between the region requested for every segment and the region the
    segments are actually running in, and the remedy is to stop requesting
    one.
    """
    try:
        yield
    except _ContinuationOriginError as err:
        raise _DeepwaveValueError(
            f"{err.selector}= selects the region "
            f"at origin {list(err.selected_origin)}, which disagrees with "
            "the region the wavefields this segment continues were produced "
            f"in ({_stored_region_text(err.stored_origin, err.stored_extent)})"
            f". run_segments passes {err.selector}= to every segment and "
            "builds each segment's initial_state from the previous segment's "
            f"output, so omit {err.selector}= to let run_segments continue "
            "in the region it is already using."
        ) from None


def _any_requires_grad(obj: Any) -> bool:
    """Reports whether ``obj`` holds any tensor with ``requires_grad=True``.

    Recurses into mappings, sequences and the two Deepwave containers that
    hold tensors opaquely (:class:`~deepwave.PropagatorOutput` and
    :class:`~deepwave.AuxiliaryState`), so that a differentiable tensor nested
    inside a keyword argument or a caller's state bundle is still seen. Any
    other object is reported as not requiring grad: nothing else in the input
    set :func:`run_segments` scans can hold a tensor.
    """
    if isinstance(obj, torch.Tensor):
        return bool(obj.requires_grad)
    if isinstance(obj, (_PropagatorOutput, _AuxiliaryState)):
        return any(_any_requires_grad(value) for value in obj.as_dict().values())
    if isinstance(obj, Mapping):
        return any(_any_requires_grad(value) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_any_requires_grad(value) for value in obj)
    return False


def run_segments(
    propagator: Callable[..., Any],
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    source_amplitudes: Mapping[str, torch.Tensor],
    n_segments: Optional[int] = None,
    *,
    seg_lengths: Optional[Sequence[int]] = None,
    initial_state: Optional[_InitialState] = None,
    segment_callback: Optional[Callable[[int, _PropagatorOutput], None]] = None,
    storage: _StorageLike = _STORAGE_DEFAULT,
    storage_mode: Optional[_StorageMode] = None,
    storage_path: Optional[_StoragePath] = None,
    storage_compression: Optional[bool] = None,
    combine: bool = True,
    optional_model_args: Optional[Mapping[str, torch.Tensor]] = None,
    checkpoint: bool = False,
    checkpoint_device: Optional[Union[str, torch.device]] = None,
    use_reentrant: bool = True,
    _slab_pml: bool = True,
    **kwargs: Any,
) -> _PropagatorOutput:
    """Runs a propagator segment-by-segment, threading state via ``initial_state``.

    Stability: provisional (see :doc:`/stability`).

    This is the general driver underlying :func:`snapshots` and
    :func:`deepwave.wavefields.measure`; it is also reusable on its own (e.g. to
    emit a wavefield snapshot at each segment boundary via ``segment_callback``,
    or to bound backward-pass memory with ``checkpoint=True``). Apart from
    the optional per-segment ``torch.utils.checkpoint`` wrapping it performs no
    gradient bookkeeping of its own -- whether the result is differentiable is
    governed entirely by whether the inputs require grad and by ``storage_mode``
    (as for a normal propagator call).

    This takes ``model_args`` as a positional sequence and ``source_amplitudes``
    as a name->tensor mapping. That generic-wrapper convention is what lets one
    implementation drive any current or future Deepwave propagator, but it is
    provisional and it is not the ergonomic way to segment a run: most users
    should reach for ``checkpoint_segments=`` on the propagator itself, which
    builds this call for them.

    **Internal sub-stepping (``step_ratio > 1``) is handled automatically.**
    When the user ``dt`` does not satisfy the CFL condition, Deepwave runs
    internally at a smaller time step, resampling the source amplitudes up and
    the receiver amplitudes down with a global FFT. Because that resampling does
    not decompose across a segment boundary, a run of two or more segments at
    ``step_ratio > 1`` instead resamples the *whole trace* once: every source
    tensor is upsampled a single time, every segment is driven at the internal
    time step (so each segment's own ``step_ratio`` is 1),
    ``model_gradient_sampling_interval`` is scaled so the model
    gradient is sampled at the same steps a direct call samples it, and the
    time-concatenated receiver output is downsampled once at the end. Forward
    values and gradients then match an unsegmented direct call within rounding,
    given the same numerical setup. An omitted ``max_vel`` is resolved once
    through the propagator's own preparation, including its heterogeneous
    and attenuation CFL bounds, before the record is split. This is behaviour, not an
    option: a caller wanting raw segments at the internal rate simply passes the
    internal ``dt`` itself, making ``step_ratio`` 1.

    **Combining with ``model_gradient_sampling_interval > 1``:** the
    model-gradient match above holds only if every segment boundary falls on a
    multiple of the sampling interval, because each segment's sampling phase
    restarts at 0 and a boundary that splits a sampling period would sample the
    gradient at the wrong steps. Rather than leave that to the caller, a
    misaligned split raises ``DeepwaveValueError`` up front, in the forward
    pass, naming the offending segment and a segment count that would work.
    The check is skipped when no model gradient will be computed, since there
    is then nothing for the interval to sample. This is unrelated to
    ``step_ratio`` and applies whether or not internal sub-stepping is
    happening.

    Args:
        propagator: A Deepwave propagator function -- any propagator that
            :func:`deepwave.spec` accepts, currently every propagator
            deepwave provides
            (e.g. ``deepwave.scalar``, ``acoustic``, ``elastic``,
            ``elastic_vti``, ``acoustic_vti``, ``scalar_born``,
            ``viscoacoustic``, ``acoustic_born``, ``elastic_born``). Every
            propagator (including the Born family) takes its required models
            positional (before ``grid_spacing``/``dt``) and its optional
            scattering models
            (``dv``/``drho``/``dlamb``/``dmu``/``dbuoyancy``/``c66``)
            keyword-only (after ``dt``); since ``run_segments`` calls the
            propagator directly (not through a ``torch.autograd.Function``
            boundary), those optional models may simply be passed in
            ``**kwargs`` like any other keyword argument and remain fully
            differentiable -- unless ``checkpoint=True``, which does
            cross such a boundary, and so needs them in
            ``optional_model_args`` instead (see below).
        model_args: The model tensors, in the order the propagator takes them
            positionally (e.g. ``(v,)`` for scalar; ``(lamb, mu, buoyancy)`` for
            elastic).
        grid_spacing: The grid spacing (as passed to the propagator).
        dt: The user time step.
        source_amplitudes: Mapping from source-amplitude argument name (e.g.
            ``"source_amplitudes"``, or ``"source_amplitudes_y"``) to its tensor
            of shape ``[shot, source, time]``. All must share the same time
            length, which defines ``nt``.
        n_segments: The number of near-equal contiguous time segments to split
            into. Exactly one of ``n_segments`` or ``seg_lengths`` must be given.
        seg_lengths: Explicit per-segment step counts (must sum to ``nt``), for
            splits that are not near-equal (e.g. snapshot boundaries). Exactly one
            of ``n_segments`` or ``seg_lengths`` must be given.
        initial_state: An optional starting state (a
            :class:`~deepwave.PropagatorOutput` or a name->tensor mapping),
            passed to the first segment.
        segment_callback: An optional
            ``segment_callback(cumulative_step, output)`` invoked after
            each segment completes, with ``cumulative_step`` the number of
            **user** time steps run so far (running total of the segment
            lengths, ending at ``nt``) and ``output`` that segment's
            :class:`~deepwave.PropagatorOutput`. Segment boundaries always fall
            on user steps, so this count is unaffected by automatic resampling;
            the per-segment ``output``'s receiver fields, however, are at the
            **internal** sample rate while resampling is active (they are
            downsampled only once, after the last segment) -- detect that case
            by comparing the per-segment ``output.info.internal_dt`` with the
            ``dt`` you passed.

            This is distinct from a per-step
            ``forward_callback``/``backward_callback`` forwarded in
            ``kwargs``, which ``kwargs`` passes through to each segment's own
            propagator call. Two things about those are easy to get wrong.
            ``callback_frequency`` counts the steps of each segment's call.
            When whole-trace resampling is active, these are **internal**
            steps, with interval ``output.info.internal_dt``; otherwise they
            are user steps. Multiply the desired user-step frequency by the
            whole-run ``step_ratio`` when resampling is active.
            And **each segment is a fresh propagator call, so**
            ``CallbackState.step`` **restarts at 0 in every segment**: a
            callback that indexes a whole-run array by ``state.step`` will
            overwrite the first segment's entries with every later segment's.
            During the original forward pass, a whole-run time is the
            preceding segment's ``cumulative_step * dt`` plus
            ``state.step * state.dt``. Do not use a mutable segment offset
            for backward callbacks or checkpoint recomputation: they execute
            after that offset has advanced. Forward callbacks can run again
            during checkpoint recomputation, so their side effects must be
            safe to repeat. Use ``segment_callback`` for observations made
            once per segment in the original forward pass.
        storage: The storage policy passed to each segment, as one object:
            either a mode string or a :class:`deepwave.Storage`. An
            alternative spelling for the three individual ``storage_*``
            arguments below, which are permanent and not deprecated; passing
            both spellings raises ``DeepwaveValueError``. Resolved once, here,
            so that every segment runs the same policy.
        storage_mode: The ``storage_mode`` passed to each segment. Defaults to
            ``"device"``, matching the propagators' own default (so a segmented
            run is differentiable by default, the same as a direct call).
        storage_path: The ``storage_path`` passed to each segment (used only by
            ``storage_mode="disk"``). Defaults to the system temporary directory
            (``tempfile.gettempdir()``).
        storage_compression: The ``storage_compression`` passed to
            each segment. **Lossy -- changes the model gradient by about
            0.2-4% in 2-D and 0.2-8% in 3-D.** See :func:`deepwave.elastic`.
            Defaults to False.
        combine: Which sample rate the returned receiver amplitudes are at, and
            how much is retained to produce them. If True (default) return a
            combined :class:`~deepwave.PropagatorOutput` for the whole run --
            final wavefield state plus time-concatenated receiver amplitudes,
            downsampled once as a whole to the **user** sample rate if the run
            was internally sub-stepped. If False return the *last segment's*
            output directly, at the **internal** sample rate, and do not retain
            the per-segment receiver traces, which keeps a purely streaming
            driver (used by :func:`deepwave.wavefields.measure`, whose
            ``segment_callback`` reduces each segment on the fly). Because the
            downsampling FFT is global in time and cannot be applied per
            segment, ``combine=False`` together with active resampling
            (two or more segments at ``step_ratio > 1``) raises
            ``DeepwaveValueError`` rather than quietly returning
            internal-rate data.
        optional_model_args: The keyword-only optional model tensors (``dv``,
            ``drho``, ``dlamb``, ``dmu``, ``dbuoyancy``, ``c66``) as a
            name->tensor mapping. Without ``checkpoint`` these are
            simply merged into ``**kwargs`` (passing them in ``kwargs`` directly
            remains supported and equivalent). With ``checkpoint=True``
            they must be given here rather than in ``kwargs``, because they are
            then threaded positionally through each checkpoint boundary so they
            receive gradients (closure-captured tensors get none under reentrant
            checkpointing). A name given both here and in ``kwargs`` is a
            ``ValueError``.
        checkpoint: If True, wrap each segment except the last in
            ``torch.utils.checkpoint.checkpoint``, so that segment's forward
            activations (the stored wavefield history) are freed after it runs
            and recomputed on demand during the backward pass. This is the
            standard way to bound backward-pass memory: peak backward memory
            becomes roughly one segment's history plus the auxiliary states,
            instead of the whole run's history, at the cost of one extra forward
            pass per checkpointed segment. The last segment is deliberately not
            checkpointed (recomputing it would cost time and save no peak
            memory, since its history is needed immediately). A pass with
            nothing to differentiate -- no model, source, optional-model,
            initial-state or keyword tensor requires grad, or grad mode is off
            -- skips the wrapper entirely and calls each segment directly,
            since recomputation buys nothing without a backward pass. That is
            bit-exact (the wrapper runs the same code, only under
            ``torch.no_grad()``), marginally cheaper, and free of the
            per-segment "None of the inputs have requires_grad=True" warning
            torch would otherwise emit, so ``checkpoint=True`` is safe to
            leave set on a forward-only run. The PML memory-variable fields of
            the state threaded between checkpointed segments are stored as
            their two per-axis boundary
            slabs (thickness ``pml_width + accuracy`` per side) rather than
            full-size, and rebuilt inside the segment: those fields are exactly
            zero outside the slabs, so this is bitwise-exact in both the
            forward values and every gradient, and it shrinks the
            ``n_segments x auxiliary-state`` term that dominates checkpointed
            peak memory by roughly 2-3x, growing with grid size. It applies to
            the threaded state only -- ``segment_callback`` and every returned
            output still see full-size tensors -- and falls back silently and
            exactly to full-size states for propagators that do not declare
            ``PropagatorSpec.pml_slab_axes``, for backends other than the
            compiled one, for ``mpml_fraction > 0``, for a non-default ``boundary``,
            and for grids too small for a slab to be smaller than the field.
            Take the gradient of a checkpointed run with ``loss.backward()``:
            the checkpointing is reentrant by default, and
            ``torch.utils.checkpoint`` refuses ``torch.autograd.grad()`` (and
            an ``inputs=`` argument to ``.backward()``) in that mode, raising
            rather than returning a wrong number. Set ``use_reentrant=False``
            if ``torch.autograd.grad`` is needed.
            Defaults to False.
        checkpoint_device: If given (with ``checkpoint=True``), the
            state tensors threaded between segments are moved to this device
            after each checkpointed segment and moved back to the compute device
            inside the next segment, freeing compute-device memory between
            segments (e.g. ``"cpu"``). Device transfers are differentiable, so
            this does not change gradients. It is not applied after the final
            segment, so the returned state is on the compute device exactly as
            without it. Giving it without ``checkpoint`` is a
            ``ValueError``.
        use_reentrant: The ``use_reentrant`` value passed to every
            ``torch.utils.checkpoint.checkpoint`` call. Defaults to True,
            which saves native wavefield history between segments. Set False
            to use ``torch.autograd.grad`` or ``backward(inputs=...)`` with
            checkpointing; this can retain a history buffer per segment and
            substantially increase memory use.
        _slab_pml: Set False to thread the PML memory-variable fields of the
            checkpointed state full-size instead of as their two per-axis
            boundary slabs. Underscored and outside the stability policy, for
            Deepwave's own tests.
        **kwargs: Remaining keyword arguments forwarded unchanged to every
            segment (``source_locations``, ``receiver_locations``, ``pml_freq``,
            ``max_vel``, ``accuracy``, ``origin``, ...). The survey may be
            given in any call form the propagator itself accepts: a
            ``sources=``/``receivers=``/``bg_receivers=`` mapping is resolved
            into the flat keywords before the first segment runs, and a flat
            ``source_amplitudes*`` keyword is lifted into the
            ``source_amplitudes`` mapping this function slices per segment.
            Neither is forwarded whole, which would give every segment the
            entire record. ``nt``, if supplied, pads or truncates the whole
            source record before splitting it, as on a direct propagator call.
            Giving one channel's amplitudes both ways raises.
            ``storage_mode``,
            ``storage_path``, and ``storage_compression`` must not appear here
            (use the parameters above). Under ``checkpoint=True`` a
            tensor here that requires grad is rejected, since ``kwargs`` are
            captured by the checkpointed closure and would silently receive no
            gradient (only top-level tensor values are checked). If
            ``pml_freq`` is absent, it is resolved **once, from the whole
            source record** (not per segment, which would give the segments
            different PML profiles while one set of boundary memory variables
            is threaded through them all) and then passed to every segment
            explicitly.

    Returns:
        A combined :class:`~deepwave.PropagatorOutput` (``combine=True``)
        or the last segment's output (``combine=False``).
    """
    if initial_state is not None and not isinstance(
        initial_state, (_PropagatorOutput, Mapping)
    ):
        raise _DeepwaveTypeError(
            "initial_state must be a PropagatorOutput or a mapping from "
            "initial-wavefield name to Tensor, got "
            f"{type(initial_state).__name__}."
        )
    _reject_storage_mode_kwarg(
        kwargs,
        "run_segments",
        (
            "storage",
            "storage_mode",
            "storage_path",
            "storage_compression",
        ),
    )
    # Resolved once, here, and handed to every segment as the individual
    # arguments, so that each segment runs the policy this call resolved
    # rather than re-resolving (and re-warning) per segment.
    _storage_policy = _resolve_storage_arguments(
        storage,
        storage_mode,
        storage_path,
        storage_compression,
    )
    storage_mode = _storage_policy.mode
    # Only forward a path when a mode will read it. ``_storage_policy.path``
    # is always a real directory (that is what materialising it means), and
    # handing that to each segment alongside a non-disk mode would make every
    # segment warn that the path is being ignored -- a warning the caller
    # never earned, once per segment.
    storage_path = _storage_policy.path if _storage_policy.mode == "disk" else None
    storage_compression = _storage_policy.lossy_compression
    source_amplitudes = _absorb_survey_kwargs(propagator, kwargs, source_amplitudes)
    _reject_renamed_kwargs(kwargs, "run_segments")
    if not source_amplitudes:
        raise _DeepwaveValueError(
            "run_segments requires at least one source-amplitude tensor to "
            "define the number of time steps."
        )
    if (n_segments is None) == (seg_lengths is None):
        raise _DeepwaveValueError(
            "run_segments requires exactly one of n_segments or seg_lengths."
        )
    spec = _get_spec(propagator)
    ndim = _ndim_of(model_args, kwargs)
    nt = _nt_of(source_amplitudes)
    if seg_lengths is not None:
        lengths = list(seg_lengths)
        if any(length < 1 for length in lengths):
            raise _DeepwaveValueError(f"seg_lengths must all be >= 1, got {lengths}.")
        if sum(lengths) != nt:
            raise _DeepwaveValueError(
                f"seg_lengths must sum to nt (= {nt}), got {lengths} "
                f"(sum {sum(lengths)})."
            )
    elif n_segments is not None:
        lengths = _segment_lengths(nt, n_segments)
    else:  # pragma: no cover - the exactly-one check above guarantees this
        raise _DeepwaveValueError(
            "run_segments requires exactly one of n_segments or seg_lengths."
        )
    # The model tensors whose ``requires_grad`` decides whether the sampling
    # interval is live, gathered before ``optional_model_args`` is folded into
    # ``kwargs`` below. Optional models given directly in ``kwargs`` (supported
    # without checkpointing) are found by name, so the condition matches
    # ``set_model_gradient_sampling_interval``'s own on every path.
    optional_names = set(_propagator_user_model_names(spec.name, ndim))
    grad_models = [
        *model_args,
        *(optional_model_args or {}).values(),
        *(
            value
            for name, value in kwargs.items()
            if name in optional_names and isinstance(value, torch.Tensor)
        ),
    ]
    if optional_model_args:
        clashes = sorted(set(optional_model_args) & set(kwargs))
        if clashes:
            raise _DeepwaveValueError(
                f"{clashes} appear in both optional_model_args and kwargs; pass "
                "each optional model tensor exactly once."
            )
        if not checkpoint:
            kwargs = {**kwargs, **optional_model_args}
            optional_model_args = None
    if checkpoint_device is not None and not checkpoint:
        raise _DeepwaveValueError(
            "checkpoint_device only applies to the state threaded between "
            "checkpointed segments, so it requires checkpoint=True."
        )
    if checkpoint:
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise _DeepwaveValueError(
                    f"kwargs entry {name!r} is a tensor with "
                    "requires_grad=True; under checkpoint=True keyword "
                    "arguments are closure-captured and would receive no "
                    "gradient -- pass differentiable optional model tensors via "
                    "optional_model_args instead."
                )

    if kwargs.get("pml_freq") is None:
        # A run that continues an earlier one takes that run's pml_freq
        # rather than inferring a new one, for the reason
        # `inherited_pml_freq` gives: the state's memory variables belong to
        # the profiles it was built with. Done before the inference below so
        # that the inference does not run, and does not warn.
        inherited = _inherited_pml_freq(None, _auxiliary_state_of(initial_state))
        if inherited is not None:
            kwargs = {**kwargs, "pml_freq": inherited}

    if kwargs.get("max_vel") is None:
        inherited_speed = _inherited_max_vel(None, _auxiliary_state_of(initial_state))
        if inherited_speed is not None:
            kwargs = {**kwargs, "max_vel": inherited_speed}

    if spec.name == "viscoacoustic":
        # Validate before fitting attenuation coefficients or resampling.
        from deepwave import _gsls

        frequency, mechanisms, band = _gsls.validate_configuration(
            kwargs.get("q_freq"), kwargs.get("q_n_mechanisms", 1), kwargs.get("q_band")
        )
        kwargs = {
            **kwargs,
            "q_freq": frequency,
            "q_n_mechanisms": mechanisms,
            "q_band": band,
        }

    if kwargs.get("pml_freq") is None:
        # Resolve pml_freq once, from the whole source record, and pass it to
        # every segment explicitly. Each segment only sees its own slice of
        # the sources, so leaving the propagator to infer per segment would
        # give the segments different PML profiles while threading one set of
        # boundary memory variables through them all -- silently breaking the
        # guarantee that a segmented run equals the single call it replaces.
        # (``origin`` is pinned for the same reason, further down.)
        kwargs = dict(kwargs)
        # Wrapped in ``_InferredPmlFreq`` (a plain float otherwise) so the
        # segments' grid-cells-per-wavelength warning still says the frequency
        # was inferred rather than blaming the user for a value they never
        # gave.
        inferred, inferred_from = _infer_pml_freq(
            _active_source_order(propagator, source_amplitudes, kwargs), dt
        )
        kwargs["pml_freq"] = _InferredPmlFreq(inferred)
        # The fallback frame chain is user call -> run_segments() ->
        # warn_pml_freq_migration. The warning is emitted here, once for the
        # whole record, rather than once per segment: the
        # segments are given the resolved value explicitly, so none of them
        # infers anything.
        _warn_pml_freq_migration(inferred, stacklevel=3, inferred_from=inferred_from)

    if len(lengths) > 1 and kwargs.get("max_vel") is None:
        from deepwave._plan import plan

        # Use the actual front end, stopped before stepping/history allocation.
        # A material-magnitude proxy can miss heterogeneous CFL substeps and
        # would then resample each source slice independently.
        prepared = plan(
            propagator,
            *model_args,
            grid_spacing,
            dt,
            **source_amplitudes,
            **(optional_model_args or {}),
            initial_state=initial_state,
            storage_mode=storage_mode,
            storage_path=storage_path,
            storage_compression=storage_compression,
            **kwargs,
        )
        kwargs = {**kwargs, "max_vel": prepared.info.max_vel}

    resampling = _resolve_segment_resampling(
        model_args, grid_spacing, dt, ndim, len(lengths), kwargs
    )
    if not combine and resampling is not None:
        raise _DeepwaveValueError(
            "combine=False returns the last segment's receiver amplitudes at "
            f"the internal sample rate (step_ratio={resampling.step_ratio}), "
            "because the downsampling FFT is global in time and cannot be "
            "applied per segment. Use combine=True to get user-rate receiver "
            "amplitudes for the whole run, or pass the internal time step "
            f"dt={resampling.dt:.6g} yourself so no resampling is needed."
        )
    # Up front, in the forward, and before any segment runs. Without this the
    # non-checkpointed path raises from inside the first segment (naming a
    # time-step count the caller never typed) and the checkpointed path only
    # raises during .backward(), from the recompute. The check is on the
    # user-level lengths and interval; internal sub-stepping multiplies both by
    # step_ratio, which cannot change the divisibility.
    mgsi = kwargs.get("model_gradient_sampling_interval", 1)
    if (
        isinstance(mgsi, int)
        and not isinstance(mgsi, bool)
        and torch.is_grad_enabled()
        and any(m.requires_grad for m in grad_models)
    ):
        # A non-int (or a bool) is left to the propagator, whose
        # set_model_gradient_sampling_interval owns that type error and words
        # it better than a segment-alignment message could.
        _check_segment_mgsi_alignment(
            lengths,
            mgsi,
            1 if resampling is None else resampling.step_ratio,
        )
    from deepwave._plan import active_probe

    probe = active_probe()
    if probe is not None and probe.n_segments is not None:
        return cast(
            "_PropagatorOutput",
            propagator(
                *model_args,
                grid_spacing,
                dt,
                **source_amplitudes,
                **(optional_model_args or {}),
                initial_state=initial_state,
                storage_mode=storage_mode,
                storage_path=storage_path,
                storage_compression=storage_compression,
                **kwargs,
            ),
        )
    if resampling is not None:
        source_amplitudes = {
            name: resampling.upsample(t) for name, t in source_amplitudes.items()
        }
        split_lengths = [length * resampling.step_ratio for length in lengths]
        seg_dt = resampling.dt
        seg_kwargs = dict(kwargs)
        # A direct call samples the model-gradient integral every
        # step_ratio * model_gradient_sampling_interval internal steps; the
        # segments run at the internal step, so the same stride must be
        # requested of them explicitly or they would store step_ratio times as
        # many wavefield snapshots as the direct call does.
        seg_kwargs["model_gradient_sampling_interval"] = (
            int(kwargs.get("model_gradient_sampling_interval", 1))
            * resampling.step_ratio
        )
    else:
        split_lengths = list(lengths)
        seg_dt = dt
        seg_kwargs = kwargs
    per_segment_sources = _split_sources(source_amplitudes, split_lengths)

    receiver_chunks: list[dict[str, torch.Tensor]] = []
    # The internal flat state names: what the checkpointed branch threads as
    # positional tensors, which is what keeps a segment checkpointable.
    n_mechanisms = kwargs["q_n_mechanisms"] if spec.name == "viscoacoustic" else 1
    state_names = list(
        spec._names_for_mechanisms(spec.wavefield_names, ndim, n_mechanisms)  # noqa: SLF001
    )
    # The public spelling of each of those names, for the two places that meet
    # the propagator's own public interface: the caller's ``initial_state``
    # bundle, and the bundle each segment is started from.
    public_state_names = spec.public_state_name_map(ndim, n_mechanisms)
    # The folded block, in both spellings. ``auxiliary_state_names`` is public,
    # because that is the spelling the collapse and the bundle's own keys use;
    # the flat tensors this function threads are keyed internally, and for the
    # attenuating propagators the two differ (internal ``r_0``, public
    # ``r_mech0``), so the internal list is what selects them out of a flat
    # state.
    boundary_names = list(
        spec._names_for_mechanisms(  # noqa: SLF001
            spec.auxiliary_state_names, ndim, n_mechanisms
        )
    )
    internal_state_names = spec.internal_state_name_map(ndim, n_mechanisms)
    boundary_internal = [internal_state_names[n] for n in boundary_names]
    output_names = spec.output_names(ndim)
    recv_set = _receiver_output_names(spec)
    recv_names = [n for n in output_names if n in recv_set]
    last_output: Optional[_PropagatorOutput] = None
    cumulative_step = 0

    if not checkpoint:
        state: Any = initial_state
        for i, length in enumerate(lengths):
            with _segment_region_errors():
                out = propagator(
                    *model_args,
                    grid_spacing,
                    seg_dt,
                    **per_segment_sources[i],
                    initial_state=state,
                    storage_mode=storage_mode,
                    storage_path=storage_path,
                    storage_compression=storage_compression,
                    **seg_kwargs,
                )
            if combine:
                receiver_chunks.append({n: getattr(out, n) for n in recv_names})
            state = out
            last_output = out
            cumulative_step += length
            if segment_callback is not None:
                segment_callback(cumulative_step, out)
    else:
        source_names = list(source_amplitudes.keys())
        opt_names = list(optional_model_args) if optional_model_args else []
        opt_tensors = (
            [optional_model_args[n] for n in opt_names] if optional_model_args else []
        )
        n_model = len(model_args)
        n_opt = len(opt_names)
        n_src = len(source_names)
        compute_device = model_args[0].device
        # Threading the state as positional tensors (which is what keeps it
        # checkpointable) loses the RunInfo a whole PropagatorOutput would carry,
        # so recover the survey origin from it here, under the same rule
        # resolve_initial_state uses (only when the caller gave neither origin
        # nor survey_pad themselves).
        info = getattr(initial_state, "info", None)
        pin_origin = (
            seg_kwargs.get("origin") is None and seg_kwargs.get("survey_pad") is None
        )
        if info is not None and pin_origin:
            seg_kwargs = dict(seg_kwargs)
            seg_kwargs["origin"] = list(info.origin)
            pin_origin = False

        # The fingerprint of the auxiliary state the segments carry, captured
        # from the first segment that produces one and thereafter reused, so
        # that the state rebuilt from the threaded tensors describes itself the
        # way the propagator's own output did.
        seg_fingerprint: list[Optional[_AuxiliaryState]] = [None]
        # torch.utils.checkpoint may return a plain tuple, which has no
        # ``info``, so the RunInfo is captured here as a side effect instead.
        # Re-running ``wrap`` during the backward pass rewrites the same value,
        # so this is idempotent (contrast with ``state_plan`` below, a captured
        # *input* that must never be mutated).
        seg_info: list[Any] = [None]

        def wrap(
            state_plan: Optional[dict[str, _SlabSpec]],
            segment_kwargs: Mapping[str, Any],
            *args: Any,
        ) -> Any:
            """Runs one segment from plain positional tensors (checkpointable).

            ``state_plan`` describes how the threaded state tensors were
            compressed (``None`` if they were not), and ``segment_kwargs`` is
            the propagator keyword set this particular segment runs with. Both
            are non-tensor arguments, which ``torch.utils.checkpoint`` stores
            as constants and replays unchanged when it re-invokes this function
            during the backward pass -- so neither may be mutated after a call
            captures it. Passing the keywords in rather than reading the
            enclosing ``seg_kwargs`` is what lets the origin pin below apply
            from segment 1 onwards while segment 0's recompute still replays
            byte-identical arguments to its forward.
            """
            margs = list(args[:n_model])
            oargs = list(args[n_model : n_model + n_opt])
            sargs = list(args[n_model + n_opt : n_model + n_opt + n_src])
            state_args = args[n_model + n_opt + n_src :]
            source = dict(zip(source_names, sargs))
            opt_kwargs = dict(zip(opt_names, oargs))
            segment_state = {
                n: t.to(compute_device)
                for n, t in zip(state_names, state_args)
                if t is not None
            }
            if state_plan is not None:
                # After the move, so a checkpoint_device transfer carries only
                # the slabs and the zero fill is allocated on the compute
                # device.
                segment_state = _decompress_state(segment_state, state_plan)
            if boundary_names:
                # Rebuild the opaque auxiliary state from the flat tensors. This
                # is the internal, non-deprecated route: the individual names
                # are the engine's own and no user asked for them. Selected by
                # internal name (``segment_state`` is keyed that way here) and
                # keyed by public name, which is what the bundle holds.
                folded = {
                    public_state_names[n]: segment_state.pop(n)
                    for n in boundary_internal
                    if n in segment_state
                }
                if folded:
                    segment_state["auxiliary_state"] = _AuxiliaryState._from_flat(  # noqa: SLF001
                        folded, seg_fingerprint[0]
                    )
            # The propagator names its ``initial_state`` slots publicly, which
            # is the internal spelling for every propagator but the Born ones.
            segment_state = {
                public_state_names.get(n, n): t for n, t in segment_state.items()
            }
            with _segment_region_errors():
                out = propagator(
                    *margs,
                    grid_spacing,
                    seg_dt,
                    **source,
                    initial_state=segment_state or None,
                    storage_mode=storage_mode,
                    storage_path=storage_path,
                    storage_compression=storage_compression,
                    **opt_kwargs,
                    **segment_kwargs,
                )
            seg_info[0] = out.info
            if boundary_names and seg_fingerprint[0] is None:
                seg_fingerprint[0] = out.auxiliary_state
            # A flat tuple of tensors, not the PropagatorOutput itself.
            # ``torch.utils.checkpoint(use_reentrant=True)`` runs this function
            # under ``torch.no_grad()`` and re-attaches only the ``Tensor``
            # elements of what it returns, so a tensor held inside a
            # non-``Tensor`` element -- every tensor of ``auxiliary_state`` --
            # would come back detached and its gradient path would be lost
            # silently rather than raising.
            flat = _flat_state(out, spec, ndim, n_mechanisms)
            return tuple(flat[n] for n in state_names) + tuple(
                getattr(out, n) for n in recv_names
            )

        if initial_state is None:
            state_tensors: tuple[Optional[torch.Tensor], ...] = ()
        else:
            state_map = (
                initial_state.as_dict()
                if isinstance(initial_state, _PropagatorOutput)
                else dict(initial_state)
            )
            # The caller's bundle may carry its auxiliary state opaquely; the
            # segments thread the individual tensors, so unpack it (and keep
            # its fingerprint, so the state handed to each segment describes
            # itself the way the propagator's own output did).
            if "auxiliary_state" in state_map and not isinstance(
                state_map["auxiliary_state"], _AuxiliaryState
            ):
                state = state_map["auxiliary_state"]
                raise _DeepwaveTypeError(
                    "initial_state's 'auxiliary_state' entry must be a "
                    "deepwave.AuxiliaryState (take it from a previous call's "
                    f"out.auxiliary_state), got {type(state).__name__}."
                )
            caller_state = state_map.pop("auxiliary_state", None)
            # What the caller itself wrote, captured before the auxiliary state
            # is unpacked over the top of it. Threading the state as positional
            # tensors means the propagator never sees the caller's mapping, so
            # the checks it would have applied to it are applied here instead:
            # without them this branch silently ignores a misspelt key (the
            # field it names starts from rest and the numbers are quietly
            # wrong) where a direct call and the non-checkpointed branch both
            # raise.
            supplied = dict(state_map)
            if isinstance(caller_state, _AuxiliaryState):
                # Only the recorded run parameters are wanted here. The tensors
                # are dropped because keeping them would pin segment 0's
                # auxiliary state -- and, with
                # ``use_reentrant=False``, its whole graph edge --
                # for the entire call, for the sake of a dozen scalar fields.
                seg_fingerprint[0] = dataclasses.replace(caller_state, _tensors={})
                state_map.update(caller_state.as_dict())
            _check_caller_state_keys(
                supplied,
                caller_state,
                spec,
                state_names,
                boundary_internal,
                public_state_names,
                recv_set,
            )
            # .get, not [...], so a partial initial-state mapping threads Nones
            # (which torch.utils.checkpoint passes through as constants).
            state_tensors = tuple(
                cast("Optional[torch.Tensor]", state_map.get(public_state_names[n]))
                for n in state_names
            )

        # Whether wrapping the segments can achieve anything. Every tensor a
        # segment could differentiate with respect to is covered: the models
        # and optional models, the source amplitudes (per-segment slices of
        # them, so a differentiable source is seen), the caller's initial
        # state (already unpacked into ``state_tensors`` above, boundary
        # tensors included, and scanned again in its original form as a
        # backstop), and the keyword arguments every segment runs with --
        # which ``wrap`` closes over rather than threading, so they are the
        # one place a differentiable tensor could hide. A later segment's
        # state is a function of these, so a set that requires no grad cannot
        # produce one that does, and the scan is done once rather than per
        # segment. Grad mode is checked too: under ``torch.no_grad()`` nothing
        # downstream can require grad however the inputs are marked.
        #
        # When the answer is no, each segment is called directly instead. That
        # is bit-exact -- ``torch.utils.checkpoint`` runs the very same
        # function, merely under ``torch.no_grad()``, which changes bookkeeping
        # and not arithmetic -- and it is both faster (no recomputation is ever
        # needed without a backward pass) and quieter (torch warns "None of the
        # inputs have requires_grad=True" once per wrapped segment otherwise).
        grad_active = torch.is_grad_enabled() and _any_requires_grad(
            (
                model_args,
                opt_tensors,
                per_segment_sources,
                state_tensors,
                initial_state,
                seg_kwargs,
            )
        )

        # Segment 0's state is the caller's own initial_state and is threaded
        # untouched; the plan is built from segment 0's output shapes and then
        # never rebuilt or mutated, since checkpoint calls capture it and
        # replay it during the backward pass.
        slab_enabled = _slab_pml and len(lengths) > 1
        slab_plan: Optional[dict[str, _SlabSpec]] = None
        state_plan: Optional[dict[str, _SlabSpec]] = None

        for i, length in enumerate(lengths):
            args = (
                tuple(model_args)
                + tuple(opt_tensors)
                + tuple(per_segment_sources[i][n] for n in source_names)
                + state_tensors
            )
            if i == len(lengths) - 1 or not grad_active:
                out_values = wrap(state_plan, seg_kwargs, *args)
            else:
                out_values = torch.utils.checkpoint.checkpoint(
                    wrap, state_plan, seg_kwargs, *args, use_reentrant=use_reentrant
                )
            if i == 0 and pin_origin and seg_info[0] is not None:
                # Every segment after the first starts from a threaded
                # wavefield, and a wavefield alone does not say which region of
                # the model it covers: without being told, each one infers the
                # origin from the shape and warns that it had to, in the
                # forward and again in every backward recompute. The threaded
                # ``AuxiliaryState`` carries segment 0's ``origin`` and is the
                # primary route by which the later segments learn it, so this
                # pin is the backstop for a state that does not carry one.
                # Pinning the origin segment 0 itself resolved is exactly what
                # the ``initial_state.info`` recovery above does. It is a
                # rebind, never a mutation: the dict segment 0's checkpoint
                # captured must keep describing segment 0's own forward, or its
                # recompute would not reproduce it.
                pinned = getattr(seg_info[0], "origin", None)
                if pinned is not None:
                    seg_kwargs = {**seg_kwargs, "origin": list(pinned)}
                pin_origin = False
            # ``wrap`` returns the flat state tensors followed by the
            # receiver tensors, in those two fixed orders.
            named = dict(zip(state_names, out_values[: len(state_names)]))
            named.update(zip(recv_names, out_values[len(state_names) :]))
            if combine:
                receiver_chunks.append({n: named[n] for n in recv_names})
            state_tensors = tuple(named[n] for n in state_names)
            # torch.utils.checkpoint tracks tensor arguments and replays every
            # other argument as a constant, so a non-tensor here would silently
            # drop that field's gradient rather than raise.
            if not all(t is None or isinstance(t, torch.Tensor) for t in state_tensors):
                raise AssertionError(
                    "run_segments threads state as plain tensors, but "
                    f"{spec.name} produced a non-tensor state field."
                )
            if slab_enabled and i < len(lengths) - 1:
                if slab_plan is None:
                    slab_plan = _build_slab_plan(
                        spec,
                        propagator,
                        ndim,
                        seg_kwargs,
                        state_names,
                        {n: named[n] for n in state_names},
                    )
                    if slab_plan is None:
                        slab_enabled = False
                if slab_plan is not None:
                    state_tensors = _compress_state(
                        state_tensors, state_names, slab_plan
                    )
                    state_plan = slab_plan
            if checkpoint_device is not None and i < len(lengths) - 1:
                state_tensors = tuple(
                    t.to(checkpoint_device) if t is not None else None
                    for t in state_tensors
                )
            last_output = _output_from_flat(
                spec, ndim, named, seg_fingerprint[0], seg_info[0], n_mechanisms
            )
            cumulative_step += length
            if segment_callback is not None:
                segment_callback(cumulative_step, last_output)

    if last_output is None:
        raise AssertionError("no segments were run (n_segments must be >= 1).")
    if not combine:
        return last_output
    return _combine_outputs(
        spec,
        ndim,
        last_output,
        receiver_chunks,
        resampling,
        nt=sum(lengths),
        requested_model_gradient_sampling_interval=kwargs.get(
            "model_gradient_sampling_interval", 1
        ),
    )


# ---------------------------------------------------------------------------
# Differentiable intermediate snapshots
# ---------------------------------------------------------------------------


def snapshots(
    propagator: Callable[..., Any],
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    source_amplitudes: Mapping[str, torch.Tensor],
    snapshot_steps: Sequence[int],
    *,
    initial_state: Optional[_InitialState] = None,
    max_vel: Optional[float] = None,
    courant_number: Optional[float] = None,
    **kwargs: Any,
) -> SnapshotResult:
    """Differentiable intermediate wavefield snapshots of a propagation.

    Stability: provisional (see :doc:`/stability`).

    Runs ``propagator`` once, from step 0 to ``nt``, but breaks the run into
    contiguous segments at the requested ``snapshot_steps`` (outer-step indices)
    and returns the full wavefield state at each of those steps **inside the
    autograd graph**. A loss built on any returned snapshot is differentiable
    w.r.t. the model and source tensors, exactly as a loss on the final output
    is (the segments are chained through ``initial_state``, so autograd threads
    the adjoint state across every boundary).

    This is the read-write counterpart to the read-only ``forward_callback``
    wavefield surface: instead of *observing* a detached wavefield mid-run, the
    returned snapshots are live graph nodes you can put a loss on (e.g. to match
    a measured wavefield at an intermediate time, to regularise the field, or to
    build a wavefield-space objective for WRI-style inversion).

    Implemented as a thin, propagator-agnostic layer over :func:`run_segments`
    (each segment is a normal, differentiable propagator call with
    ``storage_mode="device"``). It therefore inherits that function's exactness
    behaviour, and the same ``model_args`` (positional)/``source_amplitudes``
    (name->tensor mapping) generic-wrapper convention: what lets one
    implementation drive any propagator, and provisional along with the rest of
    this module. Users who only want to bound backward-pass memory, rather than
    to differentiate a mid-run wavefield, should reach for
    ``checkpoint_segments=`` on the propagator itself instead.

    Snapshots are supported at any ``step_ratio``. When the user ``dt`` requires
    internal sub-stepping (``step_ratio > 1``), the driver resolves ``max_vel``
    from the whole call if omitted and resamples the whole propagation once,
    so a snapshot still equals the
    wavefield a single unsegmented run would have reached; see
    :func:`run_segments`.

    The snapshot at outer step ``s`` is the wavefield state *after* ``s`` outer
    (user) steps have been applied -- i.e. after ``s * step_ratio`` internal
    steps, the same wavefield a ``forward_callback`` observes when it is invoked
    at ``step == s`` (the forward callback fires *before* the chunk starting at
    ``step``). Snapshots are returned in the propagator's "pml" view (model
    region plus PML, finite-difference padding removed), matching
    ``CallbackState.get_wavefield(name, view="pml")``.

    Args:
        propagator: A Deepwave propagator function (``deepwave.scalar`` etc.).
        model_args: The model tensors in positional order (``(v,)`` for scalar;
            ``(lamb, mu, buoyancy)`` for elastic). Those with ``requires_grad``
            receive a gradient from any loss on the snapshots or final output.
        grid_spacing: Grid spacing, as passed to the propagator.
        dt: The user time step.
        source_amplitudes: Mapping from source-amplitude argument name to tensor
            (``{"source_amplitudes": src}`` for scalar), defining ``nt``.
        snapshot_steps: The outer-step indices at which to return a snapshot.
            Each must satisfy ``1 <= step <= nt``. Duplicates are ignored and the
            order does not matter (the returned mapping is keyed by step).
        initial_state: Optional starting state (PropagatorOutput or mapping),
            threaded into the first segment.
        max_vel: Maximum wave speed, used to resolve the internal time step for
            the whole-trace resampling and passed to the propagator. If omitted,
            the propagator's preparation resolves its conservative bound.
        courant_number: Optional Courant number, affecting the resolved
            ``step_ratio`` and passed to the propagator.
        **kwargs: Remaining propagator keyword arguments forwarded to every
            segment (``source_locations``, ``receiver_locations``, ``pml_freq``,
            ``accuracy``, ``origin``, ...). ``storage_mode`` and
            :func:`run_segments`'s checkpointing parameters must not appear.

    Returns:
        A :class:`SnapshotResult`, a named ``(output, snapshots)`` tuple.
        ``output`` is the whole propagation's
        :class:`~deepwave.PropagatorOutput` (receiver amplitudes over all
        ``nt`` steps and the final wavefield state), as for a direct
        propagator call. ``snapshots`` is a ``dict`` mapping each requested
        step to a ``{wavefield_name: tensor}`` dict of that step's wavefield
        state (differentiable).
    """
    _reject_storage_mode_kwarg(
        kwargs, "snapshots", ("storage_mode", *_SEGMENT_DRIVER_RESERVED)
    )
    source_amplitudes = _absorb_survey_kwargs(propagator, kwargs, source_amplitudes)
    _reject_renamed_kwargs(kwargs, "snapshots")
    if not source_amplitudes:
        raise _DeepwaveValueError(
            "snapshots() requires at least one source-amplitude tensor to "
            "define the number of time steps."
        )
    model_args = list(model_args)
    if max_vel is not None:
        kwargs.setdefault("max_vel", max_vel)
    if courant_number is not None:
        kwargs.setdefault("courant_number", courant_number)

    spec = _get_spec(propagator)
    ndim = _ndim_of(model_args, kwargs)

    nt = _nt_of(source_amplitudes)

    steps_set: set[int] = set()
    for step in snapshot_steps:
        if isinstance(step, bool):
            raise _DeepwaveTypeError(
                "snapshot_steps must contain integer step indices, not bool."
            )
        try:
            steps_set.add(operator.index(step))
        except TypeError as exc:
            raise _DeepwaveTypeError(
                "snapshot_steps must contain integer step indices, got "
                f"{type(step).__name__}."
            ) from exc
    steps = sorted(steps_set)
    if not steps:
        raise _DeepwaveValueError("snapshot_steps must contain at least one step.")
    if steps[0] < 1 or steps[-1] > nt:
        raise _DeepwaveValueError(
            f"every snapshot step must satisfy 1 <= step <= nt (= {nt}); got "
            f"{steps}. Step s returns the wavefield after s "
            "outer steps (step nt is the final state; the pre-run initial state "
            "is simply initial_state)."
        )

    # Segment boundaries are the requested steps, plus nt so the whole
    # propagation runs even if the last snapshot is before the end.
    boundaries = list(steps)
    if boundaries[-1] != nt:
        boundaries.append(nt)
    seg_lengths = [boundaries[0]] + [
        boundaries[i] - boundaries[i - 1] for i in range(1, len(boundaries))
    ]

    steps_set = set(steps)
    snaps: dict[int, dict[str, torch.Tensor]] = {}

    def grab_snapshot(cumulative_step: int, out: _PropagatorOutput) -> None:
        if cumulative_step in steps_set:
            # The internal flat state, so that a snapshot's keys do not depend
            # on which fields the propagator carries opaquely.
            snaps[cumulative_step] = _flat_state(out, spec, ndim)

    combined = run_segments(
        propagator,
        model_args,
        grid_spacing,
        dt,
        source_amplitudes,
        seg_lengths=seg_lengths,
        initial_state=initial_state,
        segment_callback=grab_snapshot,
        storage_mode="device",
        **kwargs,
    )
    return SnapshotResult(combined, snaps)

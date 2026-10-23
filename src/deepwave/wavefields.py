"""Wavefield-space tools: volumetric sources, on-the-fly reductions, trace replay.

This module extends Deepwave from a differentiable *shot-data* operator toward
a differentiable *wavefield* operator: the propagators record and inject at
points, and these helpers let you drive and observe the field itself.
Everything here is additive, pure Python, and built on
the public propagator interface (and, for the segment drivers, on
:mod:`deepwave.checkpointing`'s ``run_segments`` machinery); no compiled-backend
or ABI change is required.

Three capabilities are provided:

* **Volumetric / extended source fields** (:func:`source_field_to_sources`, and
  the ``source_field`` keyword of :func:`deepwave.scalar`): add a dense or
  region-masked forcing ``[shot, nt, *spatial]`` to the wavefield every outer
  step, differentiable w.r.t. the field. Used for WRI (wavefield reconstruction
  inversion) and source encoding. The field is mapped to one exact single-cell
  source per (masked) cell, which sidesteps the unique-cell constraint that
  applies to hand-placed point sources.

* **Differentiable on-the-fly reductions** (:class:`RunningDFT`, :func:`measure`):
  accumulate a declarative reduction -- a running discrete Fourier transform at
  chosen frequencies -- across segments, returned inside the autograd graph. The
  DFT accumulators cost ``O(n_freqs)`` per receiver. Peak memory also includes
  each segment's receiver traces and propagation history. The reduction matches
  an offline FFT of the full traces to round-off (values and gradients).

* **Boundary record / reinject** (:func:`boundary_surface_locations`,
  :func:`record_boundary`, :func:`reinject_boundary`): record traces on a
  rectangular surface with dense receivers, then replay those traces as
  source amplitudes at the surface cells.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import torch

import deepwave
from deepwave._engine import receiver_output_names as _receiver_output_names
from deepwave._exceptions import DeepwaveValueError as _DeepwaveValueError
from deepwave._exceptions import warn_at_user_call as _warn_at_user_call
from deepwave.checkpointing import (
    _SEGMENT_DRIVER_RESERVED,
    _absorb_survey_kwargs,
    _get_spec,
    _ndim_of,
    _nt_of,
    _reject_storage_mode_kwarg,
    _validate_step_ratio,
)
from deepwave.checkpointing import (
    run_segments as _run_segments,
)
from deepwave.types import InitialState as _InitialState
from deepwave.types import PropagatorOutput as _PropagatorOutput

__all__ = [
    "BoundaryRecording",
    "MeasureResult",
    "RunningDFT",
    "boundary_surface_locations",
    "measure",
    "merge_point_sources",
    "record_boundary",
    "reinject_boundary",
    "source_field_to_sources",
]


class MeasureResult(NamedTuple):
    """The result of :func:`deepwave.wavefields.measure`.

    Returned for both values of ``return_output``; unpack it as
    ``measurements, output = measure(...)``, or read
    ``measure(...).measurements``.

    Stability: provisional (see :doc:`/stability`).

    Attributes:
        measurements: One accumulated complex tensor per
            :class:`RunningDFT`, in the order they were given.
        output: The final :class:`~deepwave.PropagatorOutput`, or ``None``
            unless ``return_output=True``.

    """

    measurements: tuple[torch.Tensor, ...]
    output: Optional[_PropagatorOutput]


class BoundaryRecording(NamedTuple):
    """The result of :func:`deepwave.wavefields.record_boundary`.

    A named two-tuple, so ``recording, surface = record_boundary(...)``
    still unpacks and still composes with
    :func:`reinject_boundary`.

    Stability: provisional (see :doc:`/stability`).

    Attributes:
        recording: The recorded surface traces,
            ``[shot, surface_cell, time]``.
        surface_locations: The surface cell coordinates,
            ``[surface_cell, ndim]``.

    """

    recording: torch.Tensor
    surface_locations: torch.Tensor


# ---------------------------------------------------------------------------
# (a) Volumetric / extended source fields
# ---------------------------------------------------------------------------


def source_field_to_sources(
    source_field: torch.Tensor,
    *,
    origin: Optional[Sequence[int]] = None,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a volumetric source field into per-cell point sources.

    Stability: provisional (see :doc:`/stability`).

    A volumetric (extended) source is a forcing ``f(shot, t, *cell)`` added to
    the wavefield at every cell of a region on every outer time step. Deepwave
    injects a point source at an integer grid cell as an *exact* single-cell
    forcing (the Hicks windowed-sinc interpolation collapses to a Kronecker delta
    at an on-node location), so a volumetric field is represented exactly by one
    integer-located point source per cell, with that cell's time series as its
    amplitude. Because each cell appears once, the resulting source locations are
    automatically unique -- the unique-cell constraint that governs hand-placed
    point sources never bites for field forcing.

    The mapping is a pure reshape/gather, so the returned amplitudes are
    differentiable w.r.t. ``source_field``: a gradient on the receiver data or on
    a later wavefield flows straight back to the field.

    Args:
        source_field: The forcing, shape ``[shot, nt, *region_shape]`` where
            ``region_shape`` has ``ndim`` entries (e.g. ``[shot, nt, ny, nx]`` in
            2-D). ``nt`` is the number of time steps and defines the propagation
            length.
        origin: The model cell index (length ``ndim``) of the region's first
            cell, i.e. ``region`` cell ``(0, ..., 0)`` sits at model cell
            ``origin``. Defaults to all zeros (the region starts at the model
            origin). This is added to the in-region cell indices to give
            model-relative source locations.
        mask: An optional boolean tensor of shape ``region_shape`` selecting which
            cells are active. If given, only the ``True`` cells become sources
            (the field values at ``False`` cells are ignored); this is the
            region-masked form and keeps the source count (and cost) down for
            sparse regions.

    Returns:
        A tuple ``(source_amplitudes, source_locations)`` with
        ``source_amplitudes`` of shape ``[shot, n_cell, nt]`` and
        ``source_locations`` of shape ``[shot, n_cell, ndim]`` (integer,
        model-relative), ready to pass to any Deepwave propagator.
    """
    if source_field.dim() < 3:
        raise _DeepwaveValueError(
            "source_field must have shape [shot, nt, *region_shape] (at least 3 "
            f"dimensions: shot, time, and >= 1 spatial), got shape "
            f"{tuple(source_field.shape)}."
        )
    n_shots = source_field.shape[0]
    nt = source_field.shape[1]
    region_shape = source_field.shape[2:]
    ndim = len(region_shape)
    device = source_field.device

    if origin is None:
        origin_list = [0] * ndim
    else:
        origin_list = [int(o) for o in origin]
        if len(origin_list) != ndim:
            raise _DeepwaveValueError(
                f"origin must have {ndim} entries (one per spatial dimension), "
                f"got {len(origin_list)}."
            )

    # Build the [n_cell, ndim] integer coordinates of the (masked) region cells,
    # in the row-major order of a flattened region so amplitudes line up.
    coord_axes = [
        torch.arange(n, device=device) + origin_list[d]
        for d, n in enumerate(region_shape)
    ]
    grids = torch.meshgrid(*coord_axes, indexing="ij")
    coords = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # [prod, ndim]

    # Amplitudes: move time to the end and flatten the spatial dims -> per cell.
    # [shot, nt, *region] -> [shot, *region, nt] -> [shot, prod, nt]
    amp = source_field.movedim(1, -1).reshape(n_shots, -1, nt)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise _DeepwaveValueError("mask must have dtype torch.bool.")
        if tuple(mask.shape) != tuple(region_shape):
            raise _DeepwaveValueError(
                f"mask shape {tuple(mask.shape)} must equal the source_field "
                f"spatial shape {tuple(region_shape)}."
            )
        flat_mask = mask.reshape(-1).to(device)
        coords = coords[flat_mask]
        amp = amp[:, flat_mask]

    if coords.shape[0] == 0:
        raise _DeepwaveValueError(
            "source_field / mask selected zero cells; nothing to inject."
        )

    locations = coords.unsqueeze(0).expand(n_shots, -1, -1).to(torch.long)
    return amp.contiguous(), locations.contiguous()


def merge_point_sources(
    field_amplitudes: torch.Tensor,
    field_locations: torch.Tensor,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenates field-derived sources with explicit point sources.

    The second half of the volumetric-source recipe.
    :func:`source_field_to_sources` turns an extended forcing into one exact
    single-cell source per cell; this merges that set with any point sources
    of your own, so that both can be passed through the propagator's single
    source-amplitude/location pair::

        amp, loc = deepwave.wavefields.source_field_to_sources(
            field, origin=origin, mask=mask
        )
        amp, loc = deepwave.wavefields.merge_point_sources(
            amp, loc, source_amplitudes, source_locations
        )
        out = deepwave.scalar(v, dx, dt, sources={"u": (amp, loc)},
                              receivers={"u": receiver_locations})

    The combined location set still has to be unique within each shot, so
    the explicit source cells must be disjoint from the field's cells (fold
    any coincident point source into the field instead).

    Stability: provisional (see :doc:`/stability`).

    Args:
        field_amplitudes: The amplitudes returned by
            :func:`source_field_to_sources`.
        field_locations: The locations returned by
            :func:`source_field_to_sources`.
        source_amplitudes: The explicit point sources' amplitudes, or
            ``None`` for "no point sources", in which case the field's own
            amplitudes and locations are returned unchanged.
        source_locations: The explicit point sources' locations, as an
            integer torch.Tensor. Off-grid (floating-point or
            :class:`deepwave.Hicks`) point sources cannot share the field's
            exact-node source slot and are rejected.

    Returns:
        Tuple:

            - The concatenated source amplitudes.
            - The concatenated source locations.

    Raises:
        ValueError: If ``source_amplitudes`` is given without
            ``source_locations``, if the two sets have different numbers of
            time samples, or if ``source_locations`` is not an integer
            tensor.

    """
    if source_amplitudes is None:
        return field_amplitudes, field_locations
    if source_locations is None:
        raise _DeepwaveValueError(
            "source_locations must be provided when source_amplitudes is given "
            "together with source_field."
        )
    if source_amplitudes.shape[-1] != field_amplitudes.shape[-1]:
        raise _DeepwaveValueError(
            "source_field and source_amplitudes must have the same number of "
            f"time samples, got {field_amplitudes.shape[-1]} (field) and "
            f"{source_amplitudes.shape[-1]} (points)."
        )
    if not torch.is_floating_point(source_locations):
        loc = source_locations.to(field_locations.dtype)
    else:
        raise _DeepwaveValueError(
            "source_locations must be an integer tensor when combined with "
            "source_field (off-grid / Hicks point sources cannot share the "
            "field's exact-node source slot)."
        )
    amp = torch.cat([field_amplitudes, source_amplitudes], dim=1)
    locations = torch.cat([field_locations, loc], dim=1)
    return amp, locations


#: The pre-0.1.0 private name of :func:`merge_point_sources`, kept as a thin
#: alias so that internal callers and any code that reached for it keep
#: working.
_merge_point_sources = merge_point_sources


# ---------------------------------------------------------------------------
# (b) Differentiable on-the-fly reductions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunningDFT:
    """Declarative running-DFT measurement, accumulated segment-wise.

    Stability: provisional (see :doc:`/stability`).

    Accumulates ``X[f] = sum_t x[t] * exp(-2j*pi*f*t*dt)`` over the propagation,
    where ``x`` is either a receiver-amplitude output or a wavefield of the
    propagator. The sum is built incrementally as segments complete. The DFT
    accumulators scale with the number of frequencies; each segment's trace and
    propagation history also contribute to peak memory. The result matches an
    offline DFT of the assembled trace to round-off and is differentiable.

    Args:
        frequencies: The frequencies at which to evaluate the transform, in the
            reciprocal of ``dt``'s time unit (Hz when ``dt`` is in seconds).
        on: The name of the propagator output to transform. For receiver
            transforms this is a receiver-amplitude output name (e.g.
            ``"receiver_amplitudes"`` for scalar, ``"receiver_amplitudes_p"`` for
            acoustic); ``None`` (default) selects the propagator's first
            receiver-amplitude output. It may instead be a *wavefield* name (e.g.
            ``"wavefield_0"``), which requires per-step segmentation (see
            :func:`measure`), or one of the keys of the propagator's
            ``auxiliary_state`` (e.g. ``"psiy_m1"``), which is treated the same
            way. Those keys are implementation state, not part of the public
            interface: their names, number and shapes may change in any
            release (see :doc:`/stability`).
    """

    frequencies: Sequence[float]
    on: Optional[str] = None


def _resolve_measurement_target(
    on: Optional[str],
    spec: Any,
    ndim: int,
    output_names: Sequence[str],
    boundary_targets: object,
) -> str:
    """Resolves a :attr:`RunningDFT.on` to a concrete output/state field name.

    A *channel* name (``"p"``, ``"u"``, ``"sigma_v"``) is resolved first,
    through :func:`deepwave.spec`, so that callers can name the physical
    quantity rather than a field spelling; anything else is taken as a field
    name and looked up as before.
    """
    if on is None:
        receiver_names = _receiver_output_names(spec)
        return next(name for name in output_names if name in receiver_names)
    public = deepwave.spec(spec.name)
    canonical = public.channel_aliases().get(on, on)
    if canonical in public.receiver_channels(ndim):
        return public.receiver_output_name(canonical)
    del output_names, boundary_targets
    return on


def _dft_kernel(
    frequencies: Sequence[float],
    sample_offset: int,
    length: int,
    dt: float,
    real_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Complex DFT kernel ``exp(-2j*pi*f*t*dt)`` of shape ``[n_freq, length]``.

    ``t`` runs over the global sample indices
    ``sample_offset .. sample_offset + length - 1`` so that the per-segment
    contributions of a partitioned trace sum to the whole-trace transform.

    Frequencies, sample indices and phase are evaluated in float64 for both
    input dtypes. This avoids losing phase precision on long float32 traces;
    the phase is reduced modulo ``2*pi`` before casting to the input's complex
    precision. Constructing frequencies directly at float64 precision also
    avoids losing precision for float64 runs before the transform begins.
    """
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    phase_dtype = torch.float64 if real_dtype == torch.float32 else real_dtype
    t = torch.arange(
        sample_offset, sample_offset + length, device=device, dtype=phase_dtype
    )
    freqs = torch.as_tensor(frequencies, dtype=phase_dtype, device=device)
    phase = (-2.0 * torch.pi * dt) * freqs.reshape(-1, 1) * t.reshape(1, -1)
    # Reducing in float64 before casting is essential for long float32 traces:
    # the unreduced phase grows with the sample index, and casting a large
    # phase to float32 loses enough precision to move the complex exponential
    # by far more than float32 round-off.
    phase = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
    return torch.exp(phase.to(complex_dtype) * 1j)


def measure(
    propagator: Callable[..., Any],
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    source_amplitudes: Mapping[str, torch.Tensor],
    measurements: Sequence[RunningDFT],
    *,
    n_segments: int = 1,
    initial_state: Optional[_InitialState] = None,
    return_output: bool = False,
    **kwargs: Any,
) -> MeasureResult:
    """Run a propagator while accumulating declarative reductions on the fly.

    Stability: provisional (see :doc:`/stability`).

    Runs ``propagator`` from step 0 to ``nt`` in ``n_segments`` contiguous
    segments (threading state through ``initial_state``), accumulating each
    :class:`RunningDFT` in ``measurements`` as the segments complete. Nothing but
    the running accumulators (and, if ``return_output``, the final state) is kept
    across segments. The accumulators cost ``O(n_freqs)`` per receiver; peak
    memory includes a segment's trace and propagation history. The result is
    returned inside the autograd graph.

    Receiver-amplitude measurements work at any ``n_segments``. A measurement on
    a *wavefield* (``on`` names a wavefield, or one of the propagator's
    ``auxiliary_state`` keys) needs the field at every outer
    step, so it forces per-step segmentation (``n_segments = nt``); this makes the
    call ``nt`` propagator invocations. If a different ``n_segments`` was
    explicitly requested, this override is reported via a warning (the
    per-step segmentation itself is exact but has Python-per-step overhead --
    prefer receiver measurements when you can).

    **Requires ``step_ratio == 1`` when the run is segmented** (whenever the
    resolved ``n_segments > 1``, including the per-step wavefield path). Otherwise
    a ``ValueError`` is raised, because splitting a propagation into segments is
    only bit-exact -- and hence the accumulated reduction only equals an offline
    reduction of the whole trace -- when the user ``dt`` already satisfies the CFL
    condition, so Deepwave does no internal sub-stepping (the source amplitudes
    are otherwise resampled to the internal time step with a global FFT that does
    not decompose across a segment boundary; see
    :mod:`deepwave.checkpointing`'s module docstring). A single-segment
    (``n_segments == 1``) receiver measurement runs at any ``step_ratio``.

    Args:
        propagator: A Deepwave propagator function (``deepwave.scalar`` etc.).
        model_args: Model tensors in positional order (``(v,)`` for scalar).
        grid_spacing: Grid spacing, as passed to the propagator.
        dt: The user time step.
        source_amplitudes: Mapping from source-amplitude argument name to tensor,
            defining ``nt`` (e.g. ``{"source_amplitudes": src}``).
        measurements: The reductions to accumulate.
        n_segments: Number of contiguous segments (default 1). Ignored (forced to
            ``nt``) if any measurement is on a wavefield.
        initial_state: Optional starting state threaded into the first segment.
        return_output: If True, populate the result's ``output`` field with the
            final :class:`~deepwave.PropagatorOutput`; it is ``None``
            otherwise. The shape of the return value does not depend on this:
            :class:`MeasureResult` is a ``tuple`` subclass, so a result that
            was sometimes a bare list and sometimes a pair would make
            ``for m in measure(...)`` iterate the wrong thing.
        **kwargs: Remaining propagator keyword arguments forwarded to every
            segment. ``storage_mode`` and
            :func:`deepwave.checkpointing.run_segments`'s checkpointing
            parameters must not appear.

    Returns:
        A :class:`MeasureResult`; unpack it as
        ``measurements, output = measure(...)``. ``measurements`` holds one
        accumulated complex tensor per measurement, in the order of
        ``measurements`` (``result.measurements[i]`` is the reduction of
        ``measurements[i]``). Receiver transforms have shape
        ``[shot, receiver, n_freq]``; wavefield transforms have shape
        ``[shot, *wavefield_spatial, n_freq]``. ``output`` is the final
        :class:`~deepwave.PropagatorOutput` if ``return_output`` is True, and
        ``None`` otherwise.
    """
    _reject_storage_mode_kwarg(
        kwargs, "measure", ("storage_mode", *_SEGMENT_DRIVER_RESERVED)
    )
    source_amplitudes = _absorb_survey_kwargs(propagator, kwargs, source_amplitudes)
    if not source_amplitudes:
        raise _DeepwaveValueError(
            "measure() requires at least one source-amplitude tensor to define "
            "the number of time steps."
        )
    if not measurements:
        raise _DeepwaveValueError("measure() requires at least one measurement.")

    spec = _get_spec(propagator)
    ndim = _ndim_of(model_args, kwargs)
    output_names = spec.output_names(ndim)
    # Not "everything that is not a wavefield name": the opaque
    # ``auxiliary_state`` field is state, and is not one of the internal
    # wavefield names it replaces, so it would otherwise be taken for a
    # receiver output -- and, being first, would become the default target.
    receiver_names = _receiver_output_names(spec)
    recv_names = [n for n in output_names if n in receiver_names]
    # The public state fields, which are what a wavefield measurement may name.
    # Taken from the public output names rather than the internal
    # ``wavefield_names``, since the two spell the Born propagators' scattered
    # fields differently and a state field mistaken for a receiver one would be
    # measured against the wrong index.
    state_field_names = {
        n for n in output_names if n not in receiver_names and n != "auxiliary_state"
    }
    # The individual tensors carried inside ``auxiliary_state``. They may be
    # named too: each is a field of the state at one time step, so it is a
    # wavefield target for segmentation purposes, but it is read out of the
    # mapping rather than off the output object.
    boundary_names = list(spec.auxiliary_state_names(ndim))
    boundary_targets = set(boundary_names)
    if not recv_names:
        raise _DeepwaveValueError("propagator has no receiver-amplitude outputs.")

    nt = _nt_of(source_amplitudes)

    # Resolve each measurement's target output name and whether it is a wavefield.
    targets: list[str] = []
    on_wavefield = False
    for m in measurements:
        name = _resolve_measurement_target(
            m.on, spec, ndim, output_names, boundary_targets
        )
        if name not in output_names and name not in boundary_targets:
            raise _DeepwaveValueError(
                f"measurement target {m.on!r} is not something that can be "
                "measured on this propagator; available channels: "
                f"{list(spec.receiver_channels(ndim))}; available output "
                f"names: {[n for n in output_names if n != 'auxiliary_state']}; "
                "auxiliary_state keys, which are the internal auxiliary-field "
                "names listed on the callback-names page (the bundle itself is "
                f"opaque as an output element): {boundary_names}."
            )
        if name == "auxiliary_state":
            raise _DeepwaveValueError(
                f"measurement target {name!r} is not an output that can be "
                f"measured on this propagator; available outputs: "
                f"{[n for n in output_names if n != 'auxiliary_state']}. The "
                "absorbing boundary's memory variables are carried opaquely "
                f"in auxiliary_state, which is a collection of tensors rather "
                f"than a tensor; name one of its keys ({boundary_names}) to "
                "measure one of them."
            )
        if name in state_field_names or name in boundary_targets:
            on_wavefield = True
        targets.append(name)

    if on_wavefield:
        if n_segments != 1 and n_segments != nt:
            _warn_at_user_call(
                "a wavefield measurement needs the wavefield at every outer "
                "step, so n_segments is forced to nt (per-step segmentation) "
                f"and the requested n_segments={n_segments} is ignored; pass "
                "n_segments=nt (or omit it) to silence this.",
                UserWarning,
                2,
            )
        n_segments = nt

    if n_segments > 1:
        # A segmented run is only bit-exact (and hence the accumulated reduction
        # only matches an offline reduction of the whole trace) when there is no
        # internal sub-stepping. run_segments lifts that restriction for itself
        # by resampling the whole trace and driving its segments at the internal
        # time step, but this streaming DFT is defined on user-rate samples: its
        # kernel dt, its sample indices, and the cumulative_step scaling would
        # all have to move to the internal rate to follow, so the restriction is
        # deliberately kept here.
        from deepwave._plan import plan

        prepared = plan(
            propagator,
            *model_args,
            grid_spacing,
            dt,
            **source_amplitudes,
            initial_state=initial_state,
            **kwargs,
        )
        kwargs = {**kwargs, "max_vel": prepared.info.max_vel}
        _validate_step_ratio(
            model_args,
            grid_spacing,
            dt,
            kwargs.get("max_vel"),
            kwargs.get("courant_number"),
            "measure",
            kwargs,
        )

    # Streaming accumulation over segments. The DFT of each segment is added on
    # the fly (via run_segments' per-boundary callback), so only the running
    # accumulators -- O(n_freqs) -- are kept, never the full trace. The two
    # indexing views share the callback's running outer-step count: for receiver
    # measurements the trace sample index is the global outer-step count; for
    # wavefield measurements the field after k outer steps is indexed at k-1
    # (0-based), matching a per-step DFT.
    # Kept as plain float lists, not tensors, until each use site inside
    # `accumulate` knows the run's actual `real_dtype` -- see `_dft_kernel`'s
    # docstring for why constructing the tensor any earlier loses precision.
    freqs_by_measurement = [list(m.frequencies) for m in measurements]
    accumulators: list[Optional[torch.Tensor]] = [None] * len(measurements)

    def accumulate(cumulative_step: int, out: _PropagatorOutput) -> None:
        for mi, target in enumerate(targets):
            # ``out.auxiliary_state[name]``, never ``getattr(out, name)``: the
            # attribute path is the deprecated pre-fold spelling, so it would
            # warn at a caller who did nothing deprecated, and it does not
            # exist at all on the propagators that were never released with
            # these fields at the top level.
            x = (
                out.auxiliary_state[target]
                if target in boundary_targets
                else getattr(out, target)
            )
            if target in state_field_names or target in boundary_targets:
                # One time sample (this segment's final field) at sample index
                # cumulative_step - 1.
                real_dtype = (
                    x.dtype if torch.is_floating_point(x) else torch.get_default_dtype()
                )
                kernel = _dft_kernel(
                    freqs_by_measurement[mi],
                    cumulative_step - 1,
                    1,
                    dt,
                    real_dtype,
                    x.device,
                )
                # Broadcast the length-one-freq kernel across the wavefield's
                # shot and spatial axes to form a per-cell frequency contribution.
                contrib = x.to(kernel.dtype).unsqueeze(-1) * kernel.reshape(
                    *([1] * x.dim()), -1
                )
            else:
                # Receiver path: x is [shot, receiver, seg_len].
                if x.numel() == 0 or x.shape[-1] == 0:
                    continue
                seg_len = x.shape[-1]
                real_dtype = (
                    x.dtype if torch.is_floating_point(x) else torch.get_default_dtype()
                )
                kernel = _dft_kernel(
                    freqs_by_measurement[mi],
                    cumulative_step - seg_len,
                    seg_len,
                    dt,
                    real_dtype,
                    x.device,
                )
                contrib = torch.einsum("srt,ft->srf", x.to(kernel.dtype), kernel)
            acc = accumulators[mi]
            accumulators[mi] = contrib if acc is None else acc + contrib

    final_output: Optional[_PropagatorOutput] = _run_segments(
        propagator,
        model_args,
        grid_spacing,
        dt,
        source_amplitudes,
        n_segments,
        initial_state=initial_state,
        segment_callback=accumulate,
        storage_mode="device",
        combine=False,
        **kwargs,
    )

    # Derive the empty-receiver fallback dtype/device from the source amplitudes
    # (complex128 for float64 sources, else complex64) so an all-empty result
    # matches what a populated result would have produced.
    sample_src = next(iter(source_amplitudes.values()))
    fallback_complex_dtype = (
        torch.complex128 if sample_src.dtype == torch.float64 else torch.complex64
    )
    results: list[torch.Tensor] = []
    for mi in range(len(measurements)):
        acc = accumulators[mi]
        if acc is None:
            # Only reachable if a receiver target had no receivers at all.
            n_freq = len(list(measurements[mi].frequencies))
            results.append(
                torch.zeros(
                    0,
                    0,
                    n_freq,
                    dtype=fallback_complex_dtype,
                    device=sample_src.device,
                )
            )
        else:
            results.append(acc)
    return MeasureResult(tuple(results), final_output if return_output else None)


# ---------------------------------------------------------------------------
# (c) Boundary record / reinject
# ---------------------------------------------------------------------------


def boundary_surface_locations(
    box: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Integer cell coordinates of the surface (faces) of a rectangular box.

    Stability: provisional (see :doc:`/stability`).

    Enumerates every cell lying on the boundary faces of the axis-aligned box
    ``box`` (each dimension given as a ``(start, stop)`` half-open range, in model
    cell indices). Corner and edge cells appear exactly once. The result is the
    dense receiver/source set for :func:`record_boundary` / :func:`reinject_boundary`.

    Args:
        box: A sequence of ``ndim`` ``(start, stop)`` tuples giving the box extent
            in model cell indices (half-open: cells ``start .. stop - 1``).

    Returns:
        An integer tensor of shape ``[n_surface, ndim]`` of the surface cell
        coordinates.
    """
    axes = [torch.arange(int(s0), int(s1)) for (s0, s1) in box]
    grids = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # [prod, ndim]
    # A cell is on the surface if any coordinate is at a face (start or stop-1).
    on_face = torch.zeros(coords.shape[0], dtype=torch.bool)
    for d, (s0, s1) in enumerate(box):
        cd = coords[:, d]
        on_face |= (cd == int(s0)) | (cd == int(s1) - 1)
    return coords[on_face].contiguous()


def _resolve_receiver_target(
    caller: str,
    propagator: Callable[..., Any],
    ndim: int,
    channel: Optional[str],
    receiver_kwarg: Optional[str],
    output_name: Optional[str],
) -> tuple[str, str]:
    """Resolves a channel (or explicit overrides) to a kwarg/output pair.

    Naming a *channel* is stable in a way that naming a keyword spelling is
    not, and it is unambiguous on the Born propagators, where "the first
    receiver output" is the *background* data.
    """
    if channel is not None and (receiver_kwarg is not None or output_name is not None):
        raise _DeepwaveValueError(
            f"{caller}: pass either channel= or the explicit "
            "receiver_kwarg=/output_name= overrides, not both. channel= "
            "resolves them both through deepwave.spec()."
        )
    spec = deepwave.spec(propagator)
    resolved = channel if channel is not None else spec.receiver_channels(ndim)[0]
    if receiver_kwarg is None:
        receiver_kwarg = spec.receiver_locations_kwarg(resolved)
    if output_name is None:
        output_name = spec.receiver_output_name(resolved)
    return receiver_kwarg, output_name


def record_boundary(
    propagator: Callable[..., Any],
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    source_amplitudes: torch.Tensor,
    source_locations: torch.Tensor,
    box: Sequence[tuple[int, int]],
    *,
    channel: Optional[str] = None,
    source_channel: Optional[str] = None,
    receiver_kwarg: Optional[str] = None,
    output_name: Optional[str] = None,
    **kwargs: Any,
) -> BoundaryRecording:
    """Record the wavefield on a rectangular surface with dense receivers.

    Stability: provisional (see :doc:`/stability`).

    Places a receiver at every cell on the faces of ``box`` (via
    :func:`boundary_surface_locations`) and runs the propagator, returning the
    recorded surface traces and the surface cell coordinates. The source traces
    supplied by the caller excite the selected source channel.

    Use the same explicit ``pml_freq`` when comparing this run with a replay
    run, so their absorbing boundaries have the same configuration.

    Args:
        propagator: A Deepwave propagator function -- any that
            :func:`deepwave.spec` accepts.
        model_args: Model tensors in positional order.
        grid_spacing: Grid spacing.
        dt: Time step.
        source_amplitudes: Source amplitudes ``[shot, source, nt]``.
        source_locations: Source locations ``[shot, source, ndim]``.
        box: The recording surface, as ``(start, stop)`` per dimension.
        channel: The receiver channel to record, e.g. ``"u"`` on
            :func:`deepwave.scalar`, ``"p"`` on :func:`deepwave.elastic`.
            Both the receiver keyword and the output field are resolved from
            it through :func:`deepwave.spec`, so callers name a channel --
            which is stable -- rather than a keyword spelling, which is not.
            Defaults to the propagator's first receiver channel, which is
            ``"u"`` on ``scalar`` and reproduces this function's original
            behaviour exactly.
        source_channel: The source channel to excite, resolved independently
            of ``channel``. Defaults to the propagator's first source channel
            for the dimension of ``box``. For example, use ``"volume_rate"``
            for the acoustic Born background source.
        receiver_kwarg: An explicit override of the receiver-locations
            keyword, for anything the spec cannot express. Passing it
            together with ``channel`` raises.
        output_name: An explicit override of the output field holding the
            recorded traces. Passing it together with ``channel`` raises.
        **kwargs: Remaining propagator keyword arguments.

    Returns:
        A :class:`BoundaryRecording`, a named
        ``(recording, surface_locations)`` tuple where ``recording`` has shape
        ``[shot, n_surface, nt]`` and ``surface_locations`` is
        ``[n_surface, ndim]`` -- the two arguments
        :func:`reinject_boundary` takes, in that order.
    """
    ndim = len(box)
    receiver_kwarg, output_name = _resolve_receiver_target(
        "record_boundary", propagator, ndim, channel, receiver_kwarg, output_name
    )
    spec = deepwave.spec(propagator)
    resolved_source = (
        source_channel if source_channel is not None else spec.source_channels(ndim)[0]
    )
    surface = boundary_surface_locations(box)
    n_shots = source_amplitudes.shape[0]
    recv_locations = surface.unsqueeze(0).expand(n_shots, -1, -1).contiguous()
    out = propagator(
        *model_args,
        grid_spacing,
        dt,
        **{
            spec.source_amplitudes_kwarg(resolved_source): source_amplitudes,
            spec.source_locations_kwarg(resolved_source): source_locations,
            receiver_kwarg: recv_locations,
        },
        **kwargs,
    )
    recording = getattr(out, output_name)
    return BoundaryRecording(recording, surface)


def reinject_boundary(
    propagator: Callable[..., Any],
    model_args: Sequence[torch.Tensor],
    grid_spacing: Union[float, int, Sequence[float]],
    dt: float,
    recording: torch.Tensor,
    surface_locations: torch.Tensor,
    *,
    channel: Optional[str] = None,
    amplitude_scale: float = 1.0,
    **kwargs: Any,
) -> _PropagatorOutput:
    """Replay boundary traces as sources at the supplied surface cells.

    Stability: provisional (see :doc:`/stability`).

    This helper multiplies the supplied traces by ``amplitude_scale`` and uses
    them as source amplitudes at the supplied surface cells. It applies no
    boundary-integral weights, normal derivatives, dipole terms, or automatic
    conversion from recorded-field units to source units. It does not in
    general reconstruct the original field or preserve its phase or amplitude.

    Use the same explicit ``pml_freq`` when comparing the recording and replay
    runs, so their absorbing boundaries have the same configuration.

    Args:
        propagator: A Deepwave propagator function.
        model_args: Model tensors for the replay run.
        grid_spacing: Grid spacing.
        dt: Time step.
        recording: Boundary recording ``[shot, n_surface, nt]`` from
            :func:`record_boundary`.
        surface_locations: Surface cell coordinates ``[n_surface, ndim]``.
        channel: The physical channel to inject into, e.g. ``"u"`` on
            :func:`deepwave.scalar`, ``"p"`` on :func:`deepwave.elastic`. The
            source amplitude and location keywords are resolved from it
            through :func:`deepwave.spec`, so this works on every propagator
            rather than only the ones that spell their source keywords the
            way ``scalar`` does. Defaults to the propagator's first source
            channel, reproducing this function's original behaviour on
            ``scalar``.
        amplitude_scale: A scalar multiplying the supplied traces before
            injection. Default ``1.0``.
        **kwargs: Remaining propagator keyword arguments.

    Returns:
        The propagator output of the replay run.
    """
    spec = deepwave.spec(propagator)
    resolved = (
        channel
        if channel is not None
        else spec.source_channels(surface_locations.shape[-1])[0]
    )
    amplitudes_kwarg = spec.source_amplitudes_kwarg(resolved)
    locations_kwarg = spec.source_locations_kwarg(resolved)
    n_shots = recording.shape[0]
    src_locations = surface_locations.unsqueeze(0).expand(n_shots, -1, -1).contiguous()
    amplitudes = recording * amplitude_scale
    return cast(
        "_PropagatorOutput",
        propagator(
            *model_args,
            grid_spacing,
            dt,
            **{amplitudes_kwarg: amplitudes, locations_kwarg: src_locations},
            **kwargs,
        ),
    )

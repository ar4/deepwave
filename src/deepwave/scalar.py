"""Scalar wave propagation module for Deepwave.

Implements scalar wave equation propagation using finite differences in time
(2nd order) and space (user-selectable order: 2, 4, 6, or 8). Supports PML
boundaries and adjoint mode for gradient computation. For PML details, see

    Pasalic, Damir, and Ray McGarry. "Convolutional perfectly matched
        layer for isotropic and anisotropic acoustic wave equations."
        SEG Technical Program Expanded Abstracts 2010.
        Society of Exploration Geophysicists, 2010. 2925-2929.

Required inputs: wavespeed model (`v`), grid cell size (`grid_spacing`), time
step (`dt`), and either source term (`source_amplitudes` and
`source_locations`) or number of time steps (`nt`).
Outputs: final wavefields (including PML) and receiver amplitudes (empty if no
receivers).
All outputs are differentiable with respect to float torch.Tensor inputs.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave._location_interpolation
from deepwave import (
    _backend_utils,
    _boundaries,
    _common,
    _engine,
    _regular_grid,
    _storage,
)
from deepwave._exceptions import (
    DeepwaveNotImplementedError,
    DeepwaveRuntimeError,
    DeepwaveValueError,
    warn_at_user_call,
)
from deepwave._storage import (
    STORAGE_DEFAULT,
    Storage,
    resolve_storage,
    resolve_storage_arguments,
)
from deepwave.types import (
    Accuracy,
    Callback,
    CallbackState,
    InitialState,
    LocationsLike,
    ReceiverSpec,
    RunInfo,
    SourceSpec,
    StorageLike,
    StorageMode,
    StoragePath,
)


def _dim_names(ndim: int) -> list[str]:
    """Returns the spatial dimension names (slowest to fastest) for ``ndim``."""
    return ["z", "y", "x"][-ndim:]


def scalar_model_names(ndim: int) -> list[str]:
    """Name of the scalar model (the wavespeed)."""
    return ["v"]


# The user-facing model arguments, in the order the three functions below
# expect their ``model_requires_grad`` sequence.
USER_MODEL_NAMES = ("v",)


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(v_requires_grad,)``; the result is in
    :func:`scalar_model_names` order. Scalar's single model is the user's
    ``v`` itself, so this is the identity.
    """
    (v_requires_grad,) = model_requires_grad
    return [v_requires_grad]


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    The single source of truth for scalar's storage-slot count, shared by the
    forward pass (which reaches it through :func:`_engine._per_model_storage_flags`,
    the default this propagator uses) and by
    :func:`deepwave.estimate_memory`, so the two cannot drift.
    """
    return _engine.per_model_slot_flags(
        internal_model_grad_flags(model_requires_grad, ndim)
    )


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models.

    The inverse of :func:`internal_model_grad_flags`, for callers (the
    out-of-memory message in :mod:`deepwave._engine`) that hold the internal
    models rather than the user's.
    """
    return [models[0].requires_grad]


def scalar_wavefield_names(ndim: int) -> list[str]:
    """Names of the scalar wavefields, in backend order.

    The current and previous wavefields are followed by the PML auxiliary
    fields ``psi`` and ``zeta``, one of each per spatial dimension.
    """
    dims = _dim_names(ndim)
    return (
        ["wavefield_0", "wavefield_m1"]
        + [f"psi{d}_m1" for d in dims]
        + [f"zeta{d}_m1" for d in dims]
    )


def scalar_pml_slab_axes(ndim: int) -> dict[str, int]:
    """PML memory-variable wavefields and the axis each is localised along.

    ``psi{d}``/``zeta{d}`` are nonzero only within that dimension's two PML
    slabs: the compiled kernels write them only where that axis's PML flag is
    set (``scalar.c``'s "Assumptions" block; the ``if (pml_z)`` gating in
    ``scalar_kernels.h``), and :func:`_zero_pml_interiors` zeroes each field's
    interior along its own dimension on every forward entry and backward seed.
    """
    dims = _dim_names(ndim)
    axes: dict[str, int] = {}
    for i, d in enumerate(dims):
        axes[f"psi{d}_m1"] = i
        axes[f"zeta{d}_m1"] = i
    return axes


def scalar_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The CFS-CPML memory variables ``psi{d}_m1`` and ``zeta{d}_m1``, one of each
    per spatial dimension. They are a subset of
    :func:`scalar_wavefield_names`, spelled the same way, and are what
    :class:`deepwave.AuxiliaryState` holds.
    """
    dims = _dim_names(ndim)
    return [f"psi{d}_m1" for d in dims] + [f"zeta{d}_m1" for d in dims]


def scalar_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`scalar`, in return order.

    The public final wavefields -- the physical wavefields, followed by the
    single ``auxiliary_state`` that replaces the block of PML memory variables
    -- followed by the receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            scalar_wavefield_names(ndim), scalar_auxiliary_state_names(ndim)
        ),
        "receiver_amplitudes",
    ]


# Declarative description of the scalar propagator: a regular-grid solver with
# a single model and a single source/receiver type.
SPEC = _engine.PropagatorSpec(
    name="scalar",
    status="stable",
    source_channels=_engine.scalar_channels,
    receiver_channels=_engine.scalar_channels,
    channel_suffixes=False,
    staggered=False,
    model_names=scalar_model_names,
    wavefield_names=scalar_wavefield_names,
    n_source_types=lambda ndim: 1,
    n_receiver_types=lambda ndim: 1,
    output_names=scalar_output_names,
    auxiliary_state_names=scalar_auxiliary_state_names,
    pml_slab_axes=scalar_pml_slab_axes,
    spatial_parallel=True,
    # Unlike every other propagator, scalar's pure-Python backends support a
    # backward_callback (via an autograd hook on the primary wavefield) --
    # just not inside vmap (see python_backend_backward_callback_in_vmap's
    # default of False).
    python_backend_backward_callback=True,
    # The only propagator that can retain the forward wavefield through the
    # backward pass and serve it as get_wavefield(role="forward").
    has_forward_wavefield=True,
)


def _zero_pml_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
) -> None:
    """Zeros the interiors of the psi and zeta PML auxiliaries.

    The wavefields are ordered ``[current, previous, psi..., zeta...]``. psi and
    zeta are non-zero only inside the PML region, so their interiors are zeroed
    and the backend skips PML work there. Shared by the forward and backward
    passes, whose gradient wavefields share this layout.
    """
    for i in range(ndim):
        wavefields[2 + i] = _common.zero_interior(
            wavefields[2 + i], fd_pad, pml_width, i
        )
        wavefields[2 + ndim + i] = _common.zero_interior(
            wavefields[2 + ndim + i], fd_pad, pml_width, i
        )


class Scalar(torch.nn.Module):
    """Convenience nn.Module wrapper for scalar wave propagation.

    Stability: stable (see :doc:`/stability`).

    Stores `v` and `grid_spacing` once, so that each :meth:`forward` call
    needs only the arguments that vary between calls.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        requires_grad: Which of this module's models require a gradient.
            ``None`` (the default) defers entirely to the individual
            ``<model>_requires_grad`` flags below, each of which is unset
            (``None``) by default and then means ``False``. ``True`` selects
            every model this module owns, ``False`` selects none, and a
            sequence of model names selects exactly those. Passing it
            together with an individual flag that was given explicitly
            raises ``ValueError`` rather than letting one silently win --
            whichever way the two point, and even when they agree, so that
            there is one rule rather than a direction-dependent one.
        v_requires_grad: A bool specifying whether gradients will be
            computed for `v`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        storage: The storage policy for the wavefield history that the model
            gradient is computed from, as one object: either a mode string
            ("device", "cpu", "disk" or "none") or a
            :class:`deepwave.Storage`, which also carries the path and the
            lossy-compression flag. Defaults to "device". This is an
            alternative spelling for the four
            individual `storage_*` arguments below, which remain permanently
            supported and are **not** deprecated; passing `storage` together
            with any of them raises `DeepwaveValueError`, because there is
            no honest way to merge two policies.
        storage_mode: A string specifying the storage mode for intermediate
            data. One of "device", "cpu", "disk", or "none". Defaults to
            "device". A permanent alias for `Storage.mode`; not deprecated.
        storage_path: The directory that `storage_mode="disk"` writes to. A
            string or any `os.PathLike` (such as a `pathlib.Path`). Defaults
            to the system temporary directory, resolved **when you call**
            rather than when Deepwave is imported, so `TMPDIR` set at any
            point before the call is honoured. Using the default with
            `storage_mode="disk"` emits a one-time warning naming the
            directory, because on many systems it is a RAM-backed `tmpfs`
            far smaller than the disk. A permanent alias for `Storage.path`;
            not deprecated.
        storage_compression: **Lossy: it changes the model gradient, by
            about 0.2-4% in 2-D and 0.2-8% in 3-D in relative L2 error.** The
            name does not say so, which is why this entry does. If True, the
            sampled wavefield history kept for the model gradient is stored
            as 8-bit block max-abs quantisation (8 cells per spatial
            dimension per block, one full-precision scale per block).
            Measured relative L2 error against an
            uncompressed run: about **0.2-4% for 2-D problems and 0.2-8% for
            3-D**. For a matched geometry 3-D is consistently two to five
            times the 2-D error -- a 3-D block shares one scale across 512
            cells against 64 in 2-D -- but the spread across geometries and
            loss functions is wide enough that the two ranges overlap. The
            error does **not** grow with the size of the problem: with the
            survey and the model anomaly held at fixed cell indices it is
            constant to five significant figures from 32x32 to 400x400 and
            from 32^3 to 120^3, and constant in `nt` once the record is long
            enough for the wavefield to reach the receivers. It does depend
            on the survey geometry and on the loss function -- one fixed 3-D
            problem gives 3.4% under a sum of squared receiver amplitudes and
            1.9% under a data residual -- so treat these as ranges rather
            than a number to budget against. A barely illuminated gradient is
            the exception: a record too short for the wavefield to reach the
            receivers at all can give tens of percent, because there the
            gradient itself is nearly empty. The relative maximum error,
            defined as `max|g_comp - g_exact| / max|g_exact|`, is comparable
            to the relative L2 error rather than a fixed multiple of it --
            typically between 0.3 and 3 times it, and in 3-D more often below
            the L2 error than above. Do not use a per-cell relative error as
            a figure of merit: it is unbounded near the gradient's zero
            crossings. The error is essentially independent of dtype
            (float32 and float64 agree to three or four significant figures)
            and exactly independent of `storage_mode`. The worst-affected
            values are the outermost cells of the model: a cell's gradient
            there sums large, largely cancelling contributions from the
            absorbing layer, so the quantisation error does not cancel with
            them. Measured on the gradient of `scalar_born`'s `dv`,
            normalised by the gradient's own maximum, the largest edge-cell
            error is 0.03-0.07 in 1-D, about 1e-4 in 2-D and about 1e-5 in
            3-D, against below 1e-3, 2e-5 and 1e-5 respectively in the
            interior. Disabling compression removes quantisation error; it
            does not remove model-gradient sampling error. Exact discrete
            model derivatives also require an effective history stride of one
            and fixed setup parameters (see :ref:`gradient-guarantees`).
            **Unaffected, bit for bit:** the returned wavefields and
            auxiliary state, the receiver amplitudes, and the gradients with
            respect to the source amplitudes and the initial wavefields --
            compression touches only the stored history that forms the model
            gradient. **Do not use it for `gradcheck`, adjoint or
            dot-product tests, or publication-quality gradients.** In
            exchange it shrinks the offloaded host buffer by close to the
            size of the dtype in bytes -- each cell becomes one byte, plus one
            full-precision scale per block -- which is about **3.8x in 2-D
            and 4.0x in 3-D for float32, and about 7.1x and 7.9x for
            float64**. That ratio is set by the dtype and the number of
            spatial dimensions, not by the size of the problem. Composed with
            `storage_mode="cpu"` it runs roughly 2 to 3 times faster than
            uncompressed offload, because it reduces the PCIe traffic that
            offload is bottlenecked on. Defaults to False.
    """

    def __init__(
        self,
        v: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        v_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        grad = _common.resolve_requires_grad(
            requires_grad,
            ["v"],
            ["v"],
            {"v": v_requires_grad},
            "Scalar",
        )
        self.v = _common.make_model_parameter(v, grad["v"], "v")
        self.grid_spacing = grid_spacing
        # ``None`` here means "the constructor was given no storage policy",
        # which is what lets ``forward`` tell an override from a default.
        self.storage = resolve_storage(
            storage,
            storage_mode,
            storage_path,
            storage_compression,
        )
        # The three individual attributes are back-filled from the resolved
        # policy so that reading ``module.storage_mode`` keeps working. Note
        # that ``storage_path`` is now ``None`` rather than a temporary
        # directory captured at import time: the directory is chosen when the
        # module is called.
        _effective = Storage() if self.storage is None else self.storage
        self.storage_mode = _effective.mode
        self.storage_path = _effective.path
        self.storage_compression = _effective.lossy_compression

    def forward(
        self,
        dt: float,
        *,
        sources: Optional[Mapping[str, SourceSpec]] = None,
        receivers: Optional[Mapping[str, ReceiverSpec]] = None,
        source_amplitudes: Optional[torch.Tensor] = None,
        source_locations: Optional[LocationsLike] = None,
        source_mask: Optional[torch.Tensor] = None,
        receiver_locations: Optional[LocationsLike] = None,
        receiver_mask: Optional[torch.Tensor] = None,
        wavefield_0: Optional[torch.Tensor] = None,
        wavefield_m1: Optional[torch.Tensor] = None,
        initial_state: Optional[InitialState] = None,
        accuracy: Accuracy = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        boundary: Optional[Union[str, Sequence[str]]] = None,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        courant_number: Optional[float] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        checkpoint_segments: Optional[int] = None,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        check_output_finite: bool = True,
        forward_callback: Optional[Callback] = None,
        backward_callback: Optional[Callback] = None,
        callback_frequency: int = 1,
        backward_callback_needs_forward_wavefield: bool = False,
        backend: str = "auto",
        python_backend: Optional[
            Union[Literal["eager", "jit", "compile"], bool]
        ] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
        location_interpolation: Literal["auto", "hicks", "none"] = "auto",
        location_interpolation_halfwidth: int = 4,
    ) -> "deepwave.PropagatorOutput":
        """Perform forward propagation/modelling.

        See :func:`scalar` for details (the constructor-owned `v`/
        `grid_spacing` and the storage settings are supplied from the
        stored module state).
        """
        _override = resolve_storage(
            storage,
            storage_mode,
            storage_path,
            storage_compression,
        )
        if _override is None:
            _override = self.storage
        _storage_for_call: StorageLike = (
            STORAGE_DEFAULT if _override is None else _override
        )
        backend = _engine._coerce_backend(  # noqa: SLF001
            backend, python_backend, stacklevel=5
        ).canonical
        return scalar(
            self.v,
            self.grid_spacing,
            dt,
            source_mask=source_mask,
            receiver_mask=receiver_mask,
            sources=sources,
            receivers=receivers,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            initial_state=initial_state,
            accuracy=accuracy,
            pml_width=pml_width,
            boundary=boundary,
            pml_freq=pml_freq,
            max_vel=max_vel,
            courant_number=courant_number,
            survey_pad=survey_pad,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            checkpoint_segments=checkpoint_segments,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            check_output_finite=check_output_finite,
            forward_callback=forward_callback,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            backward_callback_needs_forward_wavefield=(
                backward_callback_needs_forward_wavefield
            ),
            backend=backend,
            storage=_storage_for_call,
            location_interpolation=location_interpolation,
            location_interpolation_halfwidth=location_interpolation_halfwidth,
        )


def scalar(
    v: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[LocationsLike] = None,
    source_mask: Optional[torch.Tensor] = None,
    receiver_locations: Optional[LocationsLike] = None,
    receiver_mask: Optional[torch.Tensor] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    initial_state: Optional[InitialState] = None,
    accuracy: Accuracy = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    boundary: Optional[Union[str, Sequence[str]]] = None,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    courant_number: Optional[float] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    checkpoint_segments: Optional[int] = None,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    check_output_finite: bool = True,
    forward_callback: Optional[Callback] = None,
    backward_callback: Optional[Callback] = None,
    callback_frequency: int = 1,
    backward_callback_needs_forward_wavefield: bool = False,
    backend: str = "auto",
    python_backend: Optional[Union[Literal["eager", "jit", "compile"], bool]] = None,
    storage: StorageLike = STORAGE_DEFAULT,
    storage_mode: Optional[StorageMode] = None,
    storage_path: Optional[StoragePath] = None,
    storage_compression: Optional[bool] = None,
    location_interpolation: Literal["auto", "hicks", "none"] = "auto",
    location_interpolation_halfwidth: int = 4,
) -> "deepwave.PropagatorOutput":
    """Scalar wave propagation (functional interface).

    Stability: stable (see :doc:`/stability`).

    This function performs forward modelling with the scalar wave equation.
    The outputs are differentiable with respect to the wavespeed, the
    source amplitudes, and the initial wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    The function returns the final wavefields and the data recorded at
    the specified receiver locations. All returned torch.Tensors are
    differentiable with respect to the float torch.Tensors in the input.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dt: A float specifying the time step interval of the input and
            output (internally a smaller interval may be used in
            propagation to obey the CFL (Courant-Friedrichs-Lewy) condition
            for stability). Deepwave is unit-agnostic: choose any single
            consistent system of units for ``v``, ``grid_spacing``, and
            ``dt``; see :doc:`/units`.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is
            ``(amplitudes, locations)``, or
            ``(amplitudes, locations, mask)`` to include an activity mask.
            This propagator has one channel, ``u`` (the scalar wavefield),
            in every dimensionality, spelled that way however this
            propagator's own flat keywords are spelled (they carry no channel
            suffix at all). Exactly equivalent to the flat
            ``source_amplitudes``/``source_locations``/``source_mask``
            keywords, which remain permanently supported; giving the channel
            in both forms raises. Optional.
        receivers: This simulation's receivers, keyed by channel, as the
            canonical alternative to the flat
            ``receiver_locations``/``receiver_mask`` keywords. Each value is
            a locations object, or ``(locations, mask)`` to include an
            activity mask. The one channel is ``u``. Giving the channel in
            both forms raises. Optional.
        source_amplitudes: A torch.Tensor with dimensions [shot, source, time],
            holding the right-hand side of the scalar wave equation; its units
            set the units of the receiver data, and :doc:`/units` gives the
            calibration. For example, if two shots are propagated
            simultaneously, each containing three sources of one hundred time
            samples, the shape would be [2, 3, 100]. The time dimension length
            determines the number of time steps in the simulation. Optional. If
            provided, `source_locations` must also be specified. If not
            provided, `nt` must be specified.
        source_locations: A torch.Tensor with dimensions [shot, source, ndim],
            containing the index in the ndim spatial dimensions of the cell
            that each source is located in, relative to the origin of the
            model. Optional. Must be provided if `source_amplitudes` is. It
            usually has torch.long (int64) datatype, giving integer grid cell
            indices; a floating-point tensor may instead be provided to place
            sources at off-grid positions, handled according to the
            `location_interpolation` argument. For full control (including
            dipole sources and free surfaces) a
            :class:`deepwave.Hicks` object may be passed
            here instead of a tensor, in which case the propagator performs the
            interpolation spread and gather for you (provide the source
            amplitudes at the original, un-interpolated locations). The location
            of each
            source must be unique within the same shot (you cannot have two
            sources in the same shot that both have location [1, 2], for
            example).
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_mask: A boolean torch.Tensor with dimensions [shot, source]
            selecting which sources are active. ``True`` means active;
            ``False`` means that point injects nothing, and its coordinates in
            ``source_locations`` are then ignored entirely (they need not be
            valid, in range, or unique). The shape must match the leading
            dimensions of ``source_locations``, which must be given. This is
            the modern spelling of the :data:`deepwave.IGNORE_LOCATION`
            sentinel and produces bit-identical results; supplying both a mask
            and a sentinel-containing locations tensor for the same component
            raises, as does combining a mask with a :class:`deepwave.Hicks`
            object: a ``Hicks`` carries its own already-resolved points, so
            disable points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_locations: A torch.Tensor with dimensions
            [shot, receiver, ndim], containing the coordinates of the cell
            containing each receiver. Optional. It usually has torch.long
            (int64) datatype; a floating-point tensor may instead be provided
            (see `source_locations` and `location_interpolation`). If not
            provided, the output
            `receiver_amplitudes` torch.Tensor will be empty. If
            backpropagation will be performed, the location of each
            receiver must be unique within the same shot.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_mask: A boolean torch.Tensor with dimensions [shot, receiver]
            selecting which receivers are active. ``True`` means active;
            ``False`` means that point records zeros, and its coordinates in
            ``receiver_locations`` are then ignored entirely (they need not be
            valid, in range, or unique). The shape must match the leading
            dimensions of ``receiver_locations``, which must be given. This is
            the modern spelling of the :data:`deepwave.IGNORE_LOCATION`
            sentinel and produces bit-identical results; supplying both a mask
            and a sentinel-containing locations tensor for the same component
            raises, as does combining a mask with a :class:`deepwave.Hicks`
            object: a ``Hicks`` carries its own already-resolved points, so
            disable points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        wavefield_0: A torch.Tensor specifying the initial wavefield at time
            step 0. It should have (ndim + 1) dimensions, with the first
            dimension being shot and the subsequent ones corresponding to the
            spatial dimensions. Two spatial shapes are accepted. The first is
            the simulation domain, which is the extracted model region plus
            the PML: if two shots are being propagated simultaneously in a
            region of size [20, 30] extracted from the model for the
            simulation, and `pml_width=[1, 2, 3, 4]`, then `wavefield_0`
            would be of shape [2, 23, 37]. This is the shape a previous call
            returns, so it is the shape to use when continuing a simulation.
            The second is the extracted model region itself ([2, 20, 30] in
            that example), which Deepwave zero-pads for you by `pml_width` on
            each face; this is usually the convenient shape when you are
            specifying an initial condition rather than continuing a run,
            such as the initial pressure of a photoacoustic simulation.
            Because a wavefield determines the region that is extracted when
            `survey_pad` is not given, a shape that could be read either way
            is read as the simulation domain, which is what it has always
            meant, and a warning names both readings and the region each
            would select. Providing `origin` does not remove that ambiguity
            -- `origin` says where the region begins, not whether the shape
            includes the PML -- so the warning is issued whether or not you
            provide one.
            Passing `initial_state` from a previous call is never ambiguous,
            since the auxiliary state it carries records the origin and PML
            width it was produced with. Optional, default all zeros.
        wavefield_m1: A torch.Tensor specifying the initial wavefield at time
            step -1 (using Deepwave's internal time step interval, which may
            be smaller than the user provided one to obey the CFL condition).
            See the entry for `wavefield_0` for more details.
        initial_state: An optional bundle of initial wavefields, supplied as
            either a :class:`deepwave.PropagatorOutput` (the return of a
            previous call), that output's `final_state` mapping, or any
            mapping whose keys are the public state field names
            (`wavefield_0`, `wavefield_m1`) and `auxiliary_state` (a
            :class:`deepwave.AuxiliaryState` from a previous call). Each entry
            fills the correspondingly-named initial wavefield that was not
            also passed explicitly. Because the output wavefields are named
            identically to these arguments, passing a previous output back in
            as `initial_state` continues the simulation seamlessly (its
            `receiver_amplitudes` field is ignored), and it is the
            unambiguous route: the auxiliary state records the propagator, accuracy, PML
            width, boundary conditions, number of relaxation mechanisms,
            simulation-domain shape and internal time step it was produced
            with, so a continuation that does not match them raises rather
            than producing quiet nonsense, and one whose `pml_freq` or
            `max_vel` differs -- which reshapes the absorbing boundary
            without changing what the tensors are -- warns. The survey origin
            is recorded and reused: see the continuation rule below.

            **Continuation rule.** When `initial_state` carries a region --
            a `PropagatorOutput` from a previous call, whose `RunInfo.origin`
            and `RunInfo.extent` record the region that run used, or a
            `AuxiliaryState` carrying the same -- and you pass neither
            `survey_pad` nor `origin`, that stored region is reused: the
            continuation propagates exactly the same part of the model, and
            the new call's sources and receivers must lie inside it (they are
            checked; a location outside it raises). If you pass `survey_pad`
            or `origin` explicitly, your value wins and the stored wavefields
            must match the region it implies; a disagreement raises rather
            than silently re-windowing the simulation. Passing the previous
            call's output (`initial_state=out`) is the unambiguous route. A
            mapping of bare tensors records no region at all, so it warns and
            assumes origin zero unless you pass `origin`.

            The individual absorbing-boundary field names (`psiz_m1`, ...,
            `zetax_m1`) are still accepted as keys, with a
            `FutureWarning`; they will stop being accepted in Deepwave
            0.4.0. Providing a wavefield both here and as an explicit
            argument raises an error. For the continuation to exactly
            reproduce a single run over the combined time range, the same
            internal time step and the same absorbing boundary must be used
            throughout. The first holds when
            the user `dt` already satisfies the CFL condition (no internal
            sub-stepping); otherwise the segment boundary introduces a small
            discretisation difference. The second is automatic: when
            `pml_freq` or `max_vel` is omitted, the value recorded in
            `initial_state` is used, rather than one inferred from this
            call's own portion of the source record or from its model, so the
            continuation builds the same absorbing boundary as the run it
            continues. Passing either explicitly overrides that, and a
            mismatch with the recorded value then warns; a mismatched
            internal time step raises. If `initial_state`
            is a `PropagatorOutput` and neither `origin` nor `survey_pad` is
            also passed, the survey origin used to produce it is recovered
            automatically, so the continuation extracts the same region of
            the model rather than assuming an origin of zero.

            The number of shots (the leading dimension of every per-shot
            tensor) is taken from the source locations if any are given,
            otherwise from the initial wavefields. So a call with `nt` and
            no sources takes its shot count from the wavefields given here;
            with neither sources nor initial wavefields the call raises.
            Optional, default None.
        accuracy: An int specifying the finite difference order of accuracy.
            Possible values are 2, 4, 6, and 8, with larger numbers resulting
            in more accurate results but greater computational cost. Optional,
            with a default of 4.
        pml_width: A single number, or two numbers for each dimension,
            specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model.
            If a single value is provided, it will be used for all edges.
            If a sequence is provided, it should contain values for the
            edges in the following order:

            - beginning of first (slowest) dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension
            - ...

            Larger values result in smaller reflections, with values of 10
            to 20 being typical. To make an edge reflective instead of
            absorbing, set the value for that edge to be zero; for the scalar
            propagator such an edge is a free surface (it reflects with
            inverted polarity, a reflection coefficient of approximately -1).
            For example, if your 2D model is
            oriented so that the surface that you want to be reflective is the
            beginning of the second dimension, then you could specify
            `pml_width=[20, 20, 0, 20]`. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
            model. Optional, default 20.
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: An optional single string, applying to every face, or a
            list of `2 * ndim` strings selecting a first-class
            boundary condition per face, in the same order as `pml_width`. Each
            entry is one of `"pml"` (absorbing; `boundary="pml"` is the
            spelled-out default, and behaves in every respect -- including the
            warnings it emits -- as omitting `boundary` does), `"free"`
            (a pressure-release
            surface: the field is held to zero on the boundary and reflects with
            inverted polarity, reflection coefficient -1), `"rigid"` (a rigid
            wall: the field reflects with preserved polarity, +1), or
            `"periodic"` (the face wraps around to the opposite face). A
            `"free"`, `"rigid"`, or `"periodic"` face requires `pml_width` to be
            zero on that face, and `"periodic"` must be set on both faces of a
            dimension. Unlike a bare zero-`pml_width` face (which is only an
            approximate reflector at accuracy above 2), a `"free"`/`"rigid"`
            face fills the exterior with the exact odd/even image at every
            accuracy order, so the analytic reflection coefficient is met to
            better than 1%. See the boundary-conditions documentation for the
            precise ghost-plane conventions. Optional, default None.
            Availability: this propagator accepts all four names, on every
            face, at every accuracy.
        pml_freq: The frequency to use when constructing the PML, usually the
            dominant frequency of the source wavelet: the PML absorbs best
            near this frequency, and choosing it poorly makes the edges more
            reflective. Optional. If not specified, it is **inferred** from
            the peak of the source amplitudes' spectrum (which returns the
            wavelet's own centre frequency for a Ricker), or, if the
            simulation has no sources at all, from ``0.5 / dt / 5``, a fifth
            of the Nyquist frequency. Nothing is inferred when the call
            continues an earlier one: an omitted ``pml_freq`` then takes the
            value recorded in ``initial_state``. The value actually used is
            reported as ``out.info.pml_freq``. Pin it explicitly for any
            published result:
            leaving it inferred makes the absorbing boundary depend on the
            wavelet, so changing the wavelet also changes the boundary. The
            source-amplitude gradient does not follow that path, so pin
            ``pml_freq`` when you differentiate with respect to the source.
            See :ref:`pml-freq`.

            .. note::
               Deepwave 0.0.27 and earlier used a fixed 25 Hz here. A seismic
               call with a 25 Hz wavelet therefore infers exactly 25.0 and
               gets the same numbers as before; a call whose inferred value is
               anything else warns once, naming ``pml_freq=25.0`` as the line
               that restores the old behaviour. That warning is a migration
               aid and is removed in 0.4.0.
        max_vel: A float specifying the maximum velocity, which is used when
            applying the CFL condition and when constructing the PML. If
            not specified, the actual maximum absolute wavespeed in the
            model (or portion of it that is used) will be used. The option
            to specify this is provided to allow you to ensure consistency
            between runs even if there are changes in the wavespeed.
            Optional, default None.
            Leaving this `None` makes the discretisation a property of the
            *current model*: the internal time step (and therefore
            `RunInfo.step_ratio` and the quadrature of the model gradient)
            changes as the model updates, so an inversion's objective is not
            exactly comparable between iterations. Pass an explicit `max_vel`
            -- the maximum wavespeed you expect over the whole inversion --
            for a reproducible objective. Nothing is read from the model
            when the call continues an earlier one: an omitted `max_vel` then
            takes the value recorded in `initial_state`. The value actually
            used is reported as `RunInfo.max_vel`. A value **below** the model's actual
            maximum warns and risks an unstable (diverging or NaN)
            simulation, since the CFL condition is computed from `max_vel`.
            It has to be given when `v` is batched by :func:`torch.func.vmap`,
            which hides the values the maximum is otherwise read from. Every
            other `torch.func` transform (:func:`torch.func.grad`,
            :func:`torch.func.jacrev`, :func:`torch.func.jvp`) leaves them
            readable, so nothing has to be supplied under those. The maximum
            arrives as a Python number, so the absorbing boundary and the
            internal time step it chooses are constants of the call rather
            than differentiable functions of the model. When it is inferred,
            the model gradient at the cell the maximum is read from omits
            the term that acts through those two;
            every other cell's gradient is unaffected. Passing `max_vel`
            removes that path. An exact model gradient of the fixed discrete
            call also requires internal-step sampling and lossless storage;
            see :doc:`/what_deepwave_calculates`.
            :doc:`/capabilities`
            lists, for each propagator, the models the maximum is read from.
        courant_number: A float specifying the Courant number to use for the
            CFL time-step calculation. If None (the default), the
            conservative :data:`deepwave.DEFAULT_COURANT_NUMBER` is used for
            all accuracy orders (the historical behaviour, reproduced
            bit-for-bit). If given, it is the
            actual dimensionless Courant number that the internal time step
            will target; a larger value permits a larger internal time step
            (fewer internal steps, faster) but reduces the stability margin. It
            is validated against the critical Courant number for the chosen
            ``accuracy`` order and errors if the value would be unstable. See
            the discretisation documentation for the critical Courant numbers.
            Optional, default None.
        survey_pad: A single value or list of two values for each dimension,
            all of which are either an int or None, specifying whether the
            simulation domain should be restricted to a region surrounding
            the sources and receivers. If you have a large model, but the
            sources and receivers of individual shots only cover a small
            portion of it (such as in a towed streamer survey), then it may
            be wasteful to simulate wave propagation over the whole model.
            This parameter allows you to specify what distance around sources
            and receivers you would like to extract from the model to use
            for the simulation. A value of None means that no restriction
            of the model should be performed in that direction, while an
            integer value specifies the minimum number of cells that
            the edge of the simulation domain should be from any source
            or receiver in that direction (if possible). If a single value
            is provided, it applies to all directions, so specifying `None`
            will use the whole model for every simulation, while specifying
            `10` will cause the simulation domain to contain at least 10
            cells of padding in each direction around sources and receivers
            if possible. The padding will end if the edge of the model is
            encountered. Specifying a list, in the following order, allows
            the padding in each direction to be controlled:

            - beginning of first (slowest) dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension
            - ...

            Ints and `None` may be mixed, so a `survey_pad` of
            [5, None, None, 10] means that there should be at least 5 cells
            of padding towards the beginning of the first dimension, no
            restriction of the simulation domain towards the end of the
            first dimension or beginning of the second, and 10 cells of
            padding towards the end of the second dimension. If the
            simulation contains one source at [20, 15], one receiver at
            [10, 15], and the model is of shape [40, 50], then the
            extracted simulation domain with this value of `survey_pad`
            would cover the region [5:40, 0:25]. The same simulation
            domain will be used for all shots propagated simultaneously, so
            if this option is used then it is advisable to propagate shots
            that cover similar regions of the model simultaneously so that
            it provides a benefit (if the shots cover opposite ends of a
            dimension, then that whole dimension will be used regardless of
            what `survey_pad` value is used). Optional, default None.
            Cannot be specified if origin is also specified. Passing it
            alongside an `initial_state` that records a different region
            raises, rather than silently re-windowing the simulation: see the
            continuation rule under `initial_state`.
        origin: A list of ints specifying the origin of the provided initial
            wavefields relative to the origin of the model. Only relevant
            if initial wavefields are provided. The origin of a wavefield
            is the cell where the extracted model used for the simulation
            domain began. It does not include the PML. So if a simulation
            is performed using the model region [10:20, 15:30], the origin
            is [10, 15]. Optional, default all zero (or, if `initial_state`
            records a region and `survey_pad` is not specified, the origin
            recorded by that previous call -- see the continuation rule under
            `initial_state`). Passing an `origin` that disagrees with the one
            `initial_state` records raises. Cannot be specified if
            survey_pad is also specified.
        nt: The number of time steps to run the simulation for. Optional: if
            it is not given, the length of the source amplitudes is used, so
            a call with sources need not repeat it.

            When both are given, `nt` is the length of the run and the source
            amplitudes are fitted to it **in your own time samples, before
            any internal resampling**: a record shorter than `nt` is
            zero-padded at the end, so the sources fall silent and the
            propagation continues; a longer one is truncated, so the samples
            beyond `nt` are never injected and no gradient reaches them.
            The result is bit-for-bit what passing a hand-padded or
            hand-truncated record would have produced. `out.info.nt` reports
            the length the run actually used.

            The number of shots (the leading dimension of every per-shot
            tensor) is taken from the source locations if any are given,
            otherwise from the initial wavefields. So with `nt` and no
            sources, the initial wavefields set it; with neither sources nor
            initial wavefields the call raises.
        model_gradient_sampling_interval: An int specifying the number of time
            steps between contributions to the model gradient. The gradient
            with respect to the model is an integral over the backpropagation
            time steps. Under band-limited forward and adjoint fields, a
            sampling rate greater than twice their highest frequency suffices
            for the full-time correlation integral. If the signal bandwidth
            is well below the data Nyquist frequency, a larger interval can
            reduce computational (especially memory) costs. Finite-record
            endpoints and arbitrary initial states can violate the ideal
            assumptions; see `docs/what_deepwave_calculates.rst`.
            Optional, default 1 (integral is sampled every time step
            interval `dt`).

            Two things to know. First, the CFL `step_ratio` **multiplies**
            this: the quadrature stride the kernel actually uses is
            `step_ratio * model_gradient_sampling_interval`, so a `dt` above
            the CFL limit already coarsens the gradient integral before this
            argument is applied (see `docs/what_deepwave_calculates.rst`).
            Second, it applies only to the compiled backend: the pure-Python
            backends (`backend="eager"`/`"torch_compile"`) step every
            internal time step and rely on native autograd, so they return
            the exact discrete adjoint and ignore this argument entirely.
            The two can therefore differ for the same arguments when the
            effective stride exceeds 1. Band-limited wavefields can justify
            this sampling for the time integral, but finite-record transients
            and energy near the sampling limit can cause material differences;
            source bandwidth alone does not provide a universal error bound.

            It must divide `nt` whenever a model requires a gradient, since
            the stride is also the engine's outer time step; a remainder
            would leave the final `nt % model_gradient_sampling_interval`
            steps unpropagated, so it raises instead. When no model requires
            a gradient (including any run under `torch.no_grad()`) it is
            reset to 1 and ignored. It does not alter callback observation
            times; ``callback_frequency`` always counts user time steps.
        checkpoint_segments: If given, split the time axis into this many
            segments and run each of them under
            ``torch.utils.checkpoint``, which is the memory lever for a run
            whose wavefield history does not fit: peak backward memory falls
            to roughly one segment's history plus the states held at the
            segment boundaries, at the cost of running the forward pass of
            each segment a second time during the backward pass. `None` (the
            default) runs the whole time axis in one piece, exactly as
            before. It is the same thing as calling
            :func:`deepwave.checkpointing.run_segments` with
            ``n_segments=`` and ``checkpoint=True``, and returns the same
            combined output -- the same fields, the same `info` and the same
            `auxiliary_state` a direct call gives. `initial_state`, the
            storage arguments and every physics option are supported;
            `nt` may pad or truncate supplied source amplitudes for the whole
            run. Source-free `nt` calls, individual initial-wavefield arguments
            and per-step callbacks are not supported, and say so.

            Take the gradient with ``loss.backward()``. The checkpointing is
            reentrant, and ``torch.utils.checkpoint`` does not support
            ``torch.autograd.grad()`` (or an ``inputs=`` argument to
            ``.backward()``) in that mode: it raises. If you need
            ``torch.autograd.grad``, call
            :func:`deepwave.checkpointing.run_segments` directly with
            ``use_reentrant=False``.
        freq_taper_frac: A float specifying the fraction of the end of the
            source and receiver amplitudes (if present) in the frequency
            domain to cosine taper if they are resampled due to the CFL
            condition. This might be useful to reduce ringing. A value of 0.1
            means that the top 10% of frequencies will be tapered.
            Default 0.0 (no tapering).
        time_pad_frac: A float specifying the amount of zero padding that will
            be added to the source and receiver amplitudes (if present) before
            resampling and removed afterwards, if they are resampled due to
            the CFL condition, as a fraction of their length. This might be
            useful to reduce wraparound artifacts. A value of 0.1 means that
            zero padding of 10% of the number of time samples will be used.
            Default 0.0.
        time_taper: A bool specifying whether to apply a Hann window in time
            to resampled source and receiver amplitudes. Upsampling tapers its
            output after the inverse FFT; downsampling tapers its input before
            the FFT. The FFT-based resampling treats the time axis as periodic,
            so amplitudes that do not decay to (near) zero at the edges --
            e.g. field data cropped from a longer recording -- can produce
            wraparound ringing. Tapering can reduce this ringing at the cost
            of attenuating genuine signal near the edges. Leave as False if
            amplitudes already taper to (near) zero. Default False.
        check_output_finite: Whether to check, before returning, that the
            final wavefields and receiver amplitudes contain no ``NaN`` or
            ``Inf``, raising :class:`deepwave.DeepwaveFloatingPointError` if
            any do. Default True: a diverged run otherwise completes
            silently and only surfaces later, far from its cause. Each
            output is reduced on the device and the answer read back with a
            single synchronisation, so the cost is one pipeline drain per
            call -- a fixed cost, whose relative size is therefore largest on
            the smallest calls. Measured over loops of 100 calls on two
            workstation GPUs: about +1.2% to +1.3% for a 100x100 grid with
            nt=300, +4% to +7% for a 40x40 grid with nt=60, and +0.5% to
            +1.9% for 300x300 with nt=1000 (see :doc:`/troubleshooting`).
            Pass False for a tight loop of very small calls, or where the
            outputs feed something that detects a ``NaN`` itself. The check
            is skipped under ``vmap`` and the other ``torch.func``
            transforms, which do not permit reading a value back from a
            batched tensor.
        forward_callback: A function that will be called during the forward
            pass. See :class:`deepwave.CallbackState` for the
            state that will be provided to the function. A callback can alter
            the running simulation, not only inspect it:
            :meth:`deepwave.CallbackState.set_wavefield` writes into the
            simulation state, and :meth:`deepwave.CallbackState.get_gradient`
            returns a live view into the gradient accumulator whose in-place
            mutation is supported (see
            :doc:`/example_custom_imaging_condition`). Mutating what
            :meth:`deepwave.CallbackState.get_wavefield` or
            :meth:`deepwave.CallbackState.get_model` returns is not: see that
            class's docstring for the full contract.
        backward_callback: A function that will be called during the backward
            pass. See :class:`deepwave.CallbackState` for the
            state that will be provided to the function (the same mutation
            contract applies). Uniquely among the
            propagators, the callback state's
            ``get_wavefield(..., role="forward")`` returns the reconstructed
            forward wavefield time-coincident with the adjoint field (on both
            this propagator's eager backend and its default compiled backend),
            for building custom imaging conditions; see
            :meth:`deepwave.CallbackState.get_wavefield`. If the
            output being differentiated turns out to be unused, so that
            autograd has only an undefined (identically zero) gradient to
            propagate, the callback still fires on its usual schedule and is
            given an adjoint wavefield of zeros, never ``None``. Retaining
            that forward field costs a second full sampled wavefield history
            alongside the usual model-gradient one, so it is **opt-in**: pass
            ``backward_callback_needs_forward_wavefield=True``.
        backward_callback_needs_forward_wavefield: Whether
            ``backward_callback`` needs
            ``get_wavefield(..., role="forward")``. Default `False`.

            Reconstructing the forward wavefield during the backward pass
            requires the forward pass to keep a second full sampled wavefield
            history alongside the model-gradient one. On the default compiled
            backend that is roughly a doubling of
            :func:`deepwave.estimate_memory`'s ``backward_storage_bytes``
            (which does not itself account for this argument). On
            ``backend="eager"`` it costs *more* than that, not less: the
            retained snapshot covers every wavefield, including the
            absorbing boundary's memory variables, which the autograd graph
            would otherwise be free to release. Most backward callbacks -- a
            NaN check, a norm monitor, a progress bar -- never ask for the
            forward field, and should not pay for it; a 3-D run is exactly
            where the callback is most useful and the extra history least
            affordable.

            With `False` (the default) the history is not allocated and
            ``get_wavefield(..., role="forward")`` raises
            :exc:`NotImplementedError` naming this argument.
            :attr:`deepwave.CallbackState.has_forward_wavefield` reports
            which case you are in without a ``try``/``except``.

            The second history uses the same
            ``storage_mode``/``storage_compression`` settings as the primary
            one, so ``storage_mode="cpu"``/``"disk"`` and
            ``storage_compression=True`` reduce it exactly as they do the
            model-gradient storage (see :doc:`/troubleshooting`). The
            argument itself changes no number, but
            ``storage_compression=True`` does change what the callback sees:
            the field ``role="forward"`` returns is then dequantised from the
            same lossy 8-bit history, so it is an approximation of the
            forward field (of order 0.1% of the local block maximum), not the
            field the forward pass actually propagated. An imaging condition
            built on it inherits that error; leave compression off where you
            need the exact field. This argument is provisional (see
            :doc:`/stability`): a future backend that can reconstruct the
            forward field for free would make it unnecessary.
        callback_frequency: The number of **user** time steps between
            callback observations. Forward callbacks see steps
            ``0, f, 2*f, ... < nt``; backward callbacks see the same steps
            in reverse. ``CallbackState.step`` and ``n_steps`` use user steps,
            ``dt`` is the user time step, ``internal_dt`` names the CFL step,
            and ``time`` is ``step * dt`` from this call's start. The final
            state at ``nt`` is returned normally, without an extra callback.
            Native execution chunks meet these boundaries independently of
            the model-gradient history stride, so requesting model gradients
            does not change the observation schedule.

            Each chunk carries a fixed cost that does not shrink with the
            chunk's length: a separate entry into the backend for that chunk
            (on the compiled backend, its own set of kernel launches), the
            construction of a `CallbackState`, and the call to your callback.
            A backward pass split into more than one chunk also re-zeroes the
            per-shot model-gradient scratch buffers before every chunk after
            the first. Splitting a run more finely therefore pays those costs
            more times, so prefer the largest `callback_frequency` that still
            samples often enough for what you are doing, and keep small
            values for debugging and for animations that need every step.
        backend: Which execution backend to use: ``"auto"`` (the default),
            ``"compiled"``, ``"eager"`` or ``"torch_compile"``. See
            :doc:`/backends` for what each value selects, what ``"auto"``
            resolves to, and how the backends agree numerically.
            All four values are implemented for this propagator, and
            ``"auto"`` resolves to the compiled C/CUDA kernel.
        python_backend: Deprecated alias for `backend`, retained for backward
            compatibility. **Removed in Deepwave 0.4.0.** Providing both
            `backend` and `python_backend` raises `ValueError`. Accepts a
            bool or one of ``"eager"``, ``"jit"``, ``"compile"``; ``None``
            (the default) means the alias was not given, leaving `backend` in
            force. :doc:`/backends` gives the full mapping to `backend`
            values.
        storage: The storage policy for the wavefield history that the model
            gradient is computed from, as one object: either a mode string
            ("device", "cpu", "disk" or "none") or a
            :class:`deepwave.Storage`, which also carries the path and the
            lossy-compression flag. Defaults to "device". This is an
            alternative spelling for the four
            individual `storage_*` arguments below, which remain permanently
            supported and are **not** deprecated; passing `storage` together
            with any of them raises `DeepwaveValueError`, because there is
            no honest way to merge two policies.
        storage_mode: A string specifying the storage mode for intermediate
            data. One of "device", "cpu", "disk", or "none". Defaults to
            "device". A permanent alias for `Storage.mode`; not deprecated.
        storage_path: The directory that `storage_mode="disk"` writes to. A
            string or any `os.PathLike` (such as a `pathlib.Path`). Defaults
            to the system temporary directory, resolved **when you call**
            rather than when Deepwave is imported, so `TMPDIR` set at any
            point before the call is honoured. Using the default with
            `storage_mode="disk"` emits a one-time warning naming the
            directory, because on many systems it is a RAM-backed `tmpfs`
            far smaller than the disk. A permanent alias for `Storage.path`;
            not deprecated.
        storage_compression: **Lossy: it changes the model gradient, by
            about 0.2-4% in 2-D and 0.2-8% in 3-D in relative L2 error.** The
            name does not say so, which is why this entry does. If True, the
            sampled wavefield history kept for the model gradient is stored
            as 8-bit block max-abs quantisation (8 cells per spatial
            dimension per block, one full-precision scale per block).
            Measured relative L2 error against an
            uncompressed run: about **0.2-4% for 2-D problems and 0.2-8% for
            3-D**. For a matched geometry 3-D is consistently two to five
            times the 2-D error -- a 3-D block shares one scale across 512
            cells against 64 in 2-D -- but the spread across geometries and
            loss functions is wide enough that the two ranges overlap. The
            error does **not** grow with the size of the problem: with the
            survey and the model anomaly held at fixed cell indices it is
            constant to five significant figures from 32x32 to 400x400 and
            from 32^3 to 120^3, and constant in `nt` once the record is long
            enough for the wavefield to reach the receivers. It does depend
            on the survey geometry and on the loss function -- one fixed 3-D
            problem gives 3.4% under a sum of squared receiver amplitudes and
            1.9% under a data residual -- so treat these as ranges rather
            than a number to budget against. A barely illuminated gradient is
            the exception: a record too short for the wavefield to reach the
            receivers at all can give tens of percent, because there the
            gradient itself is nearly empty. The relative maximum error,
            defined as `max|g_comp - g_exact| / max|g_exact|`, is comparable
            to the relative L2 error rather than a fixed multiple of it --
            typically between 0.3 and 3 times it, and in 3-D more often below
            the L2 error than above. Do not use a per-cell relative error as
            a figure of merit: it is unbounded near the gradient's zero
            crossings. The error is essentially independent of dtype
            (float32 and float64 agree to three or four significant figures)
            and exactly independent of `storage_mode`. The worst-affected
            values are the outermost cells of the model: a cell's gradient
            there sums large, largely cancelling contributions from the
            absorbing layer, so the quantisation error does not cancel with
            them. Measured on the gradient of `scalar_born`'s `dv`,
            normalised by the gradient's own maximum, the largest edge-cell
            error is 0.03-0.07 in 1-D, about 1e-4 in 2-D and about 1e-5 in
            3-D, against below 1e-3, 2e-5 and 1e-5 respectively in the
            interior. Disabling compression removes quantisation error; it
            does not remove model-gradient sampling error. Exact discrete
            model derivatives also require an effective history stride of one
            and fixed setup parameters (see :ref:`gradient-guarantees`).
            **Unaffected, bit for bit:** the returned wavefields and
            auxiliary state, the receiver amplitudes, and the gradients with
            respect to the source amplitudes and the initial wavefields --
            compression touches only the stored history that forms the model
            gradient. **Do not use it for `gradcheck`, adjoint or
            dot-product tests, or publication-quality gradients.** In
            exchange it shrinks the offloaded host buffer by close to the
            size of the dtype in bytes -- each cell becomes one byte, plus one
            full-precision scale per block -- which is about **3.8x in 2-D
            and 4.0x in 3-D for float32, and about 7.1x and 7.9x for
            float64**. That ratio is set by the dtype and the number of
            spatial dimensions, not by the size of the problem. Composed with
            `storage_mode="cpu"` it runs roughly 2 to 3 times faster than
            uncompressed offload, because it reduces the PCIe traffic that
            offload is bottlenecked on. Defaults to False.
        location_interpolation: A string controlling how non-integer (float)
            source/receiver coordinates are handled. One of:

            - "auto" (default): float coordinates that do not lie on grid
              cells are interpolated onto the grid using the method of Hicks
              (Kaiser-windowed sinc); float coordinates that do lie on grid
              cells (within a small tolerance), and integer (torch.long)
              coordinates, take the exact path and are bit-for-bit identical
              to providing integer cell indices.
            - "hicks": always use Hicks interpolation.
            - "none": never interpolate; only the exact path is accepted
              (integer (torch.long) coordinates, or float coordinates that lie
              on grid cells within a small tolerance). An off-cell float
              coordinate raises a ``ValueError`` rather than being silently
              truncated to the cell below it, as ``v0.0.27`` did.

            With genuinely off-grid coordinates the propagator performs
            band-limited injection/recording, an approximation. The manual
            :class:`deepwave.Hicks` interface remains
            available for full control. Defaults to "auto".
            ``"auto"`` interpolates a floating-point locations Tensor only when at least
            one non-ignored coordinate lies more than :data:`deepwave.ON_NODE_TOLERANCE`
            cells off its component's grid node -- **except** that a floating-point
            Tensor with ``requires_grad=True`` is *always* interpolated, even when every
            coordinate is on-node, because only the interpolated path carries gradients
            back to the positions. That changes the forward values for an on-node
            ``requires_grad`` locations Tensor relative to the same Tensor without
            ``requires_grad``. Pass ``location_interpolation="none"`` for the exact path
            (positions are then not differentiable), or accept the band-limited values.
            ``RunInfo.location_interpolation_used`` records which path was
            taken: ``"none"`` or ``"hicks"``, or ``"mixed"`` if this
            propagator has several channels of one kind (several source
            components, say) and they did not all take the same path.
        location_interpolation_halfwidth: The halfwidth of the Hicks
            interpolation window (see
            :class:`deepwave.Hicks`) used when float
            coordinates are interpolated. Defaults to 4.

    Returns:
        deepwave.PropagatorOutput: a tuple subclass whose fields, in
        order, are:

            - wavefield_0: The wavefield at the final time step.
            - wavefield_m1: The wavefield at the second-to-last time step.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables. Opaque: pass it back
              (as part of `initial_state`) to continue the simulation; its
              contents are implementation state rather than physics.
            - receiver_amplitudes: The receiver amplitudes. Empty if no
              receivers were specified.

    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every
        # segment sees the arguments exactly as they were passed.
        return _engine.delegate_checkpoint_segments(scalar, SPEC, dict(locals()))
    mpml_fraction = 0.0
    q_band = None
    q_freq = None
    q_n_mechanisms = 1
    surface_method = "ivm"
    _storage_policy = resolve_storage_arguments(
        storage,
        storage_mode,
        storage_path,
        storage_compression,
    )
    # Reassigning the parameters themselves, rather than introducing new
    # names, is deliberate: every existing downstream use of these three then
    # sees the resolved values, with no site left to forget.
    storage_mode = _storage_policy.mode
    storage_path = _storage_policy.path
    storage_compression = _storage_policy.lossy_compression
    # Resolve the survey specification into the flat channel keywords the
    # rest of this function uses, in one place and before anything
    # numeric happens: deprecated spellings first, then the
    # channel-keyed mappings, then the boolean activity masks (which
    # become the IGNORE_LOCATION sentinel). Nothing is copied or cloned,
    # so gradients reach the caller's own tensors, and a call that uses
    # only the flat keywords passes through unchanged.
    _channels = _engine.merge_deprecated_keywords(
        SPEC,
        {
            "source_amplitudes": source_amplitudes,
            "source_locations": source_locations,
            "source_mask": source_mask,
            "receiver_locations": receiver_locations,
            "receiver_mask": receiver_mask,
        },
    )
    _engine.expand_channel_mappings(
        SPEC,
        _channels,
        sources=sources,
        receivers=receivers,
    )
    _common.apply_location_masks(SPEC, _channels)
    source_amplitudes = cast("Optional[torch.Tensor]", _channels["source_amplitudes"])
    source_locations = cast("Optional[LocationsLike]", _channels["source_locations"])
    receiver_locations = cast(
        "Optional[LocationsLike]", _channels["receiver_locations"]
    )
    _engine.reject_meaningless_options(
        SPEC,
        mpml_fraction=mpml_fraction,
        surface_method=surface_method,
        q_freq=q_freq,
        q_n_mechanisms=q_n_mechanisms,
        q_band=q_band,
    )
    _engine.reject_unsupported_forward_wavefield(
        SPEC, backward_callback_needs_forward_wavefield
    )
    _common.check_models([v], ["v"], "scalar")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, python_backend, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend, python_backend
    # Resolve off-grid / directional source and receiver specifications. A
    # location argument may be a Hicks instance (giving dipoles, a custom
    # component offset, or free surfaces) or a floating-point Tensor (routed
    # through Hicks automatically). Integer/long locations take the exact path
    # and are bit-for-bit identical to before. The scalar propagator is on a
    # regular (non-staggered) grid, so the interpolated component lives on the
    # reference node (zero offset). This is done first, replacing the location
    # arguments with plain integer Tensors, so the rest of the function (vmap
    # check, ndim detection, backend) is unchanged.
    # The scalar propagator is on a regular (non-staggered) grid, so its single
    # source/receiver component lives on the reference node (offset None/zero).
    # Recorded before the location arguments are replaced by the integer grid
    # indices the backend consumes.
    location_interpolation_used = _engine.resolved_location_interpolation(
        location_interpolation, [[source_locations], [receiver_locations]]
    )
    # What interpolation has to do at each face of the model: fold a window
    # back off a reflecting face, or refuse one that would leave the model.
    # ``whole_model`` is what makes the faces named by ``boundary`` the faces
    # of ``v``; a call that propagates an extracted sub-region puts them
    # somewhere these coordinates cannot name.
    pml_inference_amplitudes = _common.active_source_amplitudes(
        [source_amplitudes], [source_locations]
    )
    _hicks_grid = _common.hicks_grid(
        boundary,
        [source_locations, receiver_locations],
        v,
        "scalar",
        whole_model=(
            survey_pad is None
            and origin is None
            and initial_state is None
            and wavefield_0 is None
            and wavefield_m1 is None
        ),
    )
    (source_amplitudes,), (source_locations,) = _common.resolve_source_interpolators(
        [source_locations],
        [source_amplitudes],
        [None],
        location_interpolation,
        location_interpolation_halfwidth,
        v.dtype,
        _hicks_grid,
    )
    receiver_interpolators, (receiver_locations,) = (
        _common.resolve_receiver_interpolators(
            [receiver_locations],
            [None],
            location_interpolation,
            location_interpolation_halfwidth,
            v.dtype,
            _hicks_grid,
        )
    )

    # Fill any initial wavefields supplied via the optional initial_state
    # bundle (e.g. a PropagatorOutput from a previous call), leaving explicit
    # wavefield arguments untouched. Order is preserved for unpacking.
    # The auxiliary state is read here but checked after the call's own
    # parameters have been resolved (there is nothing to compare it against
    # until then).
    supplied_auxiliary_state = _engine.auxiliary_state_of(initial_state)
    # A continuation has nothing of its own to resolve these from -- it
    # typically has no sources, and the first call's `max_vel` need not be the
    # model's own maximum -- so an omitted one takes the value the state's
    # absorbing-boundary memory variables were built with.
    pml_freq = _engine.inherited_pml_freq(pml_freq, supplied_auxiliary_state)
    max_vel = _engine.inherited_max_vel(max_vel, supplied_auxiliary_state)
    _resolved_wavefields, origin = _engine.resolve_initial_state(
        initial_state,
        {
            "wavefield_0": wavefield_0,
            "wavefield_m1": wavefield_m1,
            # psi*/zeta* are the absorbing boundary's memory variables, not
            # physical fields, so they are not exposed as arguments; they
            # travel in `auxiliary_state`.
            "psiz_m1": None,
            "psiy_m1": None,
            "psix_m1": None,
            "zetaz_m1": None,
            "zetay_m1": None,
            "zetax_m1": None,
        },
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=scalar_auxiliary_state_names(3),
        propagator=SPEC.name,
    )
    (
        wavefield_0,
        wavefield_m1,
        psiz_m1,
        psiy_m1,
        psix_m1,
        zetaz_m1,
        zetay_m1,
        zetax_m1,
    ) = _resolved_wavefields.values()

    _common.check_inputs_not_vmapped(
        wavefield_0,
        wavefield_m1,
        source_amplitudes,
        source_locations,
        receiver_locations,
        psiz_m1,
        zetaz_m1,
        psiy_m1,
        zetay_m1,
        psix_m1,
        zetax_m1,
    )
    ndim = _common.get_ndim(
        [v],
        [wavefield_0, wavefield_m1],
        [source_locations, receiver_locations],
        [psiz_m1, zetaz_m1],
        [psiy_m1, zetay_m1],
        [psix_m1, zetax_m1],
    )
    psi: list[Optional[torch.Tensor]] = []
    zeta: list[Optional[torch.Tensor]] = []
    if ndim == 3:
        psi.append(psiz_m1)
        zeta.append(zetaz_m1)
    if ndim >= 2:
        psi.append(psiy_m1)
        zeta.append(zetay_m1)
    if ndim >= 1:
        psi.append(psix_m1)
        zeta.append(zetax_m1)
    initial_wavefields: list[Optional[torch.Tensor]] = [
        wavefield_0,
        wavefield_m1,
        *psi,
        *zeta,
    ]

    # Whether the absorbing profiles are a constant of this call: they are
    # built from the maximum wave speed, so they depend on the model only
    # when it was not supplied. Recorded before ``setup_propagator`` resolves
    # ``max_vel`` to a number.
    max_vel_pinned = max_vel is not None
    min_nonzero_model_vel, max_model_vel = _common.velocity_extremes(
        v, max_vel, "scalar"
    )
    fd_pad = [accuracy // 2] * 2 * ndim
    user_dt = dt
    requested_mgsi = model_gradient_sampling_interval
    (
        models,
        source_amplitudes_out,
        wavefields,
        sources_i,
        receivers_i,
        grid_spacing,
        dt,
        nt,
        n_shots,
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
    ) = _common.setup_propagator(
        [v],
        ["replicate"],
        ["v"],
        grid_spacing,
        dt,
        [source_amplitudes],
        [source_locations],
        [receiver_locations],
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
        survey_pad,
        initial_wavefields,
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        ndim,
        courant_number=courant_number,
        staggered=False,
        storage=_storage_policy,
        propagator_name=SPEC.name,
        pml_inference_amplitudes=pml_inference_amplitudes,
    )
    del v, source_amplitudes, source_locations, receiver_locations

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "scalar")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "scalar")
    boundary_names = _engine.boundary_condition_names(boundary_codes)
    # The survey_pad half of the continuation rule: the region survey_pad
    # selects is only known now, so a continuation whose survey has moved
    # is caught here rather than in resolve_initial_state.
    _engine.check_continuation_survey_pad_origin(
        initial_state, survey_pad, survey_origin
    )
    _engine.check_auxiliary_state(
        supplied_auxiliary_state,
        SPEC,
        ndim,
        accuracy,
        pml_width,
        boundary_names,
        pml_freq,
        inputs_require_grad=_engine.any_requires_grad(
            models, source_amplitudes_out, wavefields
        ),
        wavefields=wavefields,
        max_vel=max_vel,
        internal_dt=dt,
        grid_spacing=grid_spacing,
        mpml_fraction=mpml_fraction,
        surface_method=surface_method,
        q_freq=q_freq,
        q_band=q_band,
    )

    # In the finite difference implementation, the source amplitudes we
    # add to the wavefield each time step are multiplied by
    # -v^2 * dt^2 (where v is the velocity model, stored in models[0]).
    # For simplicity, we multiply the amplitudes by that here.
    # We use 0 as the location of the sources that are to be ignored to
    # avoid out-of-bounds accesses to the model. Since these sources will not be
    # used, their amplitudes are not important.
    model_shape = models[0].shape[-ndim:]
    source_amplitudes_out[0] = (
        -source_amplitudes_out[0]
        * _common.gather_model_at_sources(models[0], sources_i[0], model_shape, n_shots)
        ** 2
        * dt**2
    )

    pml_profiles = _regular_grid.set_pml_profiles(
        pml_width,
        accuracy,
        fd_pad,
        dt,
        grid_spacing,
        max_vel,
        dtype,
        device,
        pml_freq,
        model_shape,
    )

    _common.check_callback_frequency(callback_frequency)
    callback_frequency = _engine.CallbackSchedule(callback_frequency, step_ratio)
    if (
        backend_mode is None
        and backward_callback is not None
        and backward_callback_needs_forward_wavefield
        and int(callback_frequency) % model_gradient_sampling_interval != 0
    ):
        raise DeepwaveValueError(
            "backward_callback_needs_forward_wavefield requires callback_frequency "
            "to be divisible by model_gradient_sampling_interval on the compiled "
            "backend, so each requested forward snapshot was stored. Choose "
            "aligned intervals."
        )

    run_info = RunInfo(
        step_ratio=step_ratio,
        model_gradient_sampling_interval=requested_mgsi,
        model_gradient_stride=_engine.resolved_model_gradient_stride(
            models, backend_mode, step_ratio, model_gradient_sampling_interval
        ),
        dt=user_dt,
        grid_spacing=tuple(float(spacing) for spacing in grid_spacing),
        accuracy=accuracy,
        boundary=tuple(boundary_names),
        mpml_fraction=mpml_fraction,
        surface_method=surface_method,
        q_freq=None if q_freq is None else float(q_freq),
        q_n_mechanisms=q_n_mechanisms,
        q_band=None if q_band is None else (float(q_band[0]), float(q_band[1])),
        internal_dt=dt,
        courant_number=resolved_courant_number,
        nt=nt // step_ratio,
        pml_width=tuple(pml_width),
        pml_freq=pml_freq,
        origin=tuple(survey_origin),
        extent=tuple(survey_extent),
        backend=_engine.resolved_backend_name(backend_mode),
        backward_backend=_engine.resolved_backend_name(backend_mode),
        max_vel=max_vel,
        ndim=ndim,
        dtype=dtype,
        device=device,
        model_batched=tuple(_engine.model_batched(models, ndim)),
        location_interpolation_used=location_interpolation_used,
        storage=_engine.resolved_storage_for_device(_storage_policy, device),
        lossy_compression_used=(
            _storage_policy.lossy_compression and _storage_policy.mode != "none"
        ),
    )
    # Everything this call resolved is now known and nothing
    # proportional to the time axis has been allocated: the point a
    # pre-flight plan stops at. A no-op for a real run.
    _engine.plan_reached(
        SPEC,
        ndim,
        run_info,
        models,
        storage_mode,
        python_backend=backend_mode,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
    )

    outputs = list(
        scalar_func(
            backend_mode,
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            _engine.execution_stride(
                backend_mode, step_ratio, model_gradient_sampling_interval
            ),
            accuracy,
            pml_width,
            boundary_codes,
            n_shots,
            forward_callback,
            backward_callback,
            callback_frequency,
            storage_mode,
            storage_path,
            storage_compression,
            backward_callback_needs_forward_wavefield,
            *models,
            *source_amplitudes_out,
            *sources_i,
            *receivers_i,
            *wavefields,
            max_vel_pinned=max_vel_pinned,
        )
    )

    outputs[-1] = _common.downsample_and_movedim(
        outputs[-1],
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 1
    )

    # Returned as a PropagatorOutput (a tuple subclass), not a list:
    # ``torch.utils.checkpoint(use_reentrant=True)`` wraps the propagator in
    # an autograd Function, and that Function's output only has
    # ``requires_grad`` propagated to it when forward returns a Tensor or a
    # tuple of Tensors. Returning a list (or a non-tuple Sequence) silently
    # detaches every output, which broke gradient checkpointing (see issue
    # #120).
    return _engine.named_output(
        SPEC,
        ndim,
        outputs,
        info=run_info,
        accuracy=accuracy,
        boundary=boundary_names,
        check_output_finite=check_output_finite,
    )


class ScalarForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of scalar wave propagation.

    This class defines the forward and backward passes for the scalar wave
    equation, allowing PyTorch to compute gradients through the wave propagation
    operation. It interfaces directly with the C/CUDA backend.
    """

    @staticmethod
    def forward(
        pml_profiles: list[torch.Tensor],
        grid_spacing: list[float],
        dt: float,
        nt: int,
        step_ratio: int,
        accuracy: int,
        pml_width: list[int],
        n_shots: int,
        forward_callback: Optional[Callback],
        backward_callback: Optional[Callback],
        callback_frequency: int,
        storage_mode_str: StorageMode,
        storage_path: str,
        storage_compression: bool,
        backward_callback_needs_forward_wavefield: bool,
        max_vel_pinned: bool,
        boundary: list[int],
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the scalar wave equation.

        The shared :func:`deepwave._engine.run_forward_engine` engine performs all of
        the boilerplate; this supplies only what is unique to the scalar
        propagator. ``args`` is the flat tensor tuple ``(v, source_amplitudes,
        sources_i, receivers_i, *wavefields)``, where the wavefields are the
        current and previous fields followed by the ``psi`` and ``zeta`` PML
        auxiliaries (one of each per dimension). The raw input wavefields are
        saved for the (double) backward pass.

        Returns the padding-removed output wavefields followed by the receiver
        amplitudes and, as a trailing non-tensor output, the
        :class:`deepwave._engine.ForwardSavedState` that ``setup_context`` needs
        to populate ``ctx``.
        """
        # Opt-in second checkpoint buffer for CallbackState role="forward"
        # (see deepwave.CallbackState): populated only when a
        # backward_callback was supplied AND the caller asked for the forward
        # wavefield, so neither the non-callback case nor the far commoner
        # inspect-only callback pays for it. Uses the SAME storage configuration
        # (storage_mode/storage_path/storage_compression) as the primary
        # checkpoint, so a user who picked "cpu" or "disk" to bound device
        # memory does not silently get a second full-device history for this
        # buffer too. See FUNC(forward)/FUNC(backward) in scalar.c/scalar.cu
        # for how the compiled backend persists/reloads it per mode.
        store_raw_forward = (
            backward_callback is not None and backward_callback_needs_forward_wavefield
        )
        ndim = len(grid_spacing)
        v_arg = args[0]
        raw_storage_manager = _storage.StorageManager(
            tuple(v_arg.shape[-ndim:]),
            v_arg.dtype,
            n_shots,
            nt,
            step_ratio,
            _engine.resolve_storage_mode(storage_mode_str, v_arg.device),
            storage_compression,
            storage_path,
            v_arg.device,
            v_arg.is_cuda,
        )
        raw_storage_manager.allocate(store_raw_forward)

        # The default storage_flags (one slot per model, gated by that model's
        # requires_grad) is exactly what scalar needs (its sole model, v).
        def make_aux_wavefields(
            wavefields: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            # Temporary ping-pong buffers for the psi PML variables. A temporary
            # is needed because updates at a grid point read psi at neighbouring
            # points, so psi cannot be updated in place.
            return [torch.zeros_like(wavefields[2 + i]) for i in range(ndim)]

        def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
            v = state.models[0]
            ndim = state.ndim
            psi = state.wavefields[2 : 2 + ndim]
            zeta = state.wavefields[2 + ndim : 2 + 2 * ndim]
            psin = state.aux_wavefields
            v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
            call_args: tuple[Any, ...] = (
                v.data_ptr(),
                state.source_amplitudes[0].data_ptr(),
                state.wavefields[0].data_ptr(),
                state.wavefields[1].data_ptr(),
                *[field.data_ptr() for field in psi],
                *[field.data_ptr() for field in psin],
                *[field.data_ptr() for field in zeta],
                *state.storage_manager.storage_ptrs,
                state.receiver_amplitudes[0].data_ptr(),
                *[profile.data_ptr() for profile in state.pml_profiles],
                state.sources_i[0].data_ptr(),
                state.receivers_i[0].data_ptr(),
                *state.rdx,
                *state.rdx2,
                state.dt**2,
                step_nt,
                state.n_shots,
                *state.model_shape,
                state.n_sources_per_shot[0],
                state.n_receivers_per_shot[0],
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                state.storage_grad_flag(v.requires_grad),
                v_batched,
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                *raw_storage_manager.storage_ptrs,
                store_raw_forward,
                raw_storage_manager.storage_mode,
                raw_storage_manager.shot_bytes_uncomp,
                raw_storage_manager.shot_bytes_comp,
                raw_storage_manager.storage_compression,
                state.aux,
                state.stream,
            )
            return cast("int", state.backend(*call_args))

        def swap(state: _engine.BackendState) -> None:
            # Ping-pong the current/previous wavefields and the psi/psin
            # temporaries after a chunk with an odd number of internal steps.
            ndim = state.ndim
            w = state.wavefields
            w[0], w[1] = w[1], w[0]
            w[2 : 2 + ndim], state.aux_wavefields[:] = (
                state.aux_wavefields[:],
                w[2 : 2 + ndim],
            )

        outputs, saved_state = _engine.run_forward_engine(
            SPEC,
            pml_profiles=pml_profiles,
            grid_spacing=grid_spacing,
            dt=dt,
            nt=nt,
            step_ratio=step_ratio,
            accuracy=accuracy,
            pml_width=pml_width,
            n_shots=n_shots,
            forward_callback=forward_callback,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            storage_mode_str=storage_mode_str,
            storage_path=storage_path,
            storage_compression=storage_compression,
            args=args,
            zero_wavefields=_zero_pml_interiors,
            make_call=make_call,
            make_aux_wavefields=make_aux_wavefields,
            swap=swap,
            save_wavefields=True,
            boundary=boundary,
            raw_forward_storage_manager=raw_storage_manager,
            store_raw_forward=store_raw_forward,
        )
        return (*outputs, saved_state)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Populates ``ctx`` from the trailing state produced by ``forward``.

        Does only what the modern autograd API permits here:
        ``ctx.save_for_backward`` and cheap attribute assignment, delegated to
        :func:`deepwave._engine.apply_forward_saved_state`, plus the extra
        bookkeeping the Born-based forward-mode (``jvp``) rule needs.
        """
        state = output[-1]
        _engine.apply_forward_saved_state(ctx, state)
        # Forward-mode AD (``torch.func.jvp`` / ``torch.autograd.forward_ad``)
        # reaches ``jvp`` with dual/GradTracking inputs that need not
        # ``requires_grad``, so ``save_for_backward`` (gated on
        # ``needs_backward``) is not guaranteed to have run. Save the primal
        # tensors for ``jvp`` and stash the metadata it needs, unconditionally.
        ctx.save_for_forward(*state.saved_tensors)
        ctx.jvp_grid_spacing = state.grid_spacing
        ctx.jvp_dt = state.dt
        ctx.jvp_nt = state.nt
        ctx.jvp_step_ratio = state.step_ratio
        ctx.jvp_accuracy = state.accuracy
        ctx.jvp_pml_width = state.pml_width
        ctx.jvp_n_shots = state.n_shots
        # ``boundary`` is the last leading (non-tensor) input; recover it for the
        # (double-)backward's and the forward-mode (jvp) rule's ``bc`` threading
        # into the Born backend, which now supports non-PML faces.
        boundary = list(inputs[_N_LEAD - 1])
        ctx.boundary = boundary
        ctx.jvp_boundary = boundary
        # Whether the caller pinned ``max_vel``: the input immediately before
        # ``boundary``. The absorbing profiles are built from the maximum wave
        # speed, so they depend on the model only when it was inferred from it,
        # which is what the forward-mode rule's caveat is about.
        ctx.jvp_max_vel_pinned = bool(inputs[_N_LEAD - 2])

    # torch.func transforms now reach this Function (setup_context migration).
    # vmap still cannot run on the ctypes backend (batched tensors), so it raises
    # an actionable error pointing to backend="eager". Forward-mode AD
    # (jvp), however, is supported on the fast backend via the Born relationship
    # (Born = JVP of scalar); see :func:`_scalar_forward_jvp`.
    vmap = staticmethod(_engine.unsupported_vmap)

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> tuple[Optional[torch.Tensor], ...]:
        """Forward-mode derivative of scalar via the Born relationship.

        See :func:`_scalar_forward_jvp` for the full explanation of how the
        input tangents map onto the scalar Born backend inputs.
        """
        return _scalar_forward_jvp(ctx, tangents)

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object that contains information saved during the
                forward pass.
            args: Gradients of the loss with respect to the outputs of forward.
                The final entry is the (``None``) gradient for the trailing
                ``ForwardSavedState`` output and is dropped.

        Returns:
            Gradients with respect to the inputs of the forward pass.

        """
        ndim = _engine.saved_state_ndim(ctx, "scalar")
        # Drop the grad slot for the trailing ``ForwardSavedState`` output.
        args = args[:-1]
        grad_r = args[-1]
        grad_wavefields = list(args[:-1])
        del args
        # Read ``ctx.saved_tensors`` exactly once: under ``torch.utils.checkpoint``
        # each access re-triggers the saved-tensor unpack hook, which errors
        # (non-reentrant) or silently returns wrong gradients (reentrant).
        saved_tensors = ctx.saved_tensors
        v, sources_i, receivers_i, source_amplitudes = saved_tensors[:4]
        wavefields = saved_tensors[4 : 6 + 2 * ndim]
        pml_profiles = saved_tensors[6 + 2 * ndim : 6 + 5 * ndim]

        grid_spacing = ctx.grid_spacing
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        # run_forward stores this as a one-element list (one source type).
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad[0]

        result = ScalarBackwardFunc.apply(  # type: ignore[no-untyped-call]
            grad_r,
            v,
            pml_profiles,
            sources_i,
            receivers_i,
            ctx.storage_manager,
            ctx.raw_forward_storage_manager,
            ctx.store_raw_forward,
            source_amplitudes,
            grid_spacing,
            dt,
            nt,
            n_shots,
            step_ratio,
            accuracy,
            pml_width,
            source_amplitudes_requires_grad,
            ctx.backward_callback,
            ctx.callback_frequency,
            ctx.boundary,
            *grad_wavefields,
            *wavefields,
        )
        # The forward signature is (pml_profiles, <13 more non-tensor args>,
        # boundary, v, source_amplitudes, sources_i, receivers_i, *wavefields),
        # so the gradient tuple is ``_N_LEAD`` Nones (the leading args, including
        # ``boundary``) followed by the gradients for v, source_amplitudes,
        # sources_i (None), receivers_i (None), wavefields.
        return (
            *([None] * _N_LEAD),
            result[0],  # grad_v
            result[1],  # grad_source_amplitudes
            None,  # sources_i
            None,  # receivers_i
            *result[2:],  # grad_wavefields
        )


# Number of leading non-tensor arguments of ``ScalarForwardFunc.forward``
# (the shared config block plus scalar's own trailing ``boundary`` flag), all of
# which receive ``None`` gradients in ``backward``.
_N_LEAD = _engine.leading_arg_count(ScalarForwardFunc.forward)


def _base_level_dispatch() -> Any:
    """Use the shared PyTorch-version-aware base-dispatch context."""
    return _common._base_level_dispatch()  # noqa: SLF001 - shared internal helper


def _v_tangent_in_pml(
    dv: torch.Tensor, pml_width: Sequence[int], accuracy: int, ndim: int
) -> bool:
    """Whether the velocity tangent ``dv`` is non-zero inside the PML band.

    ``dv`` is the tangent of the padded velocity model, so its spatial extent is
    ``fd_pad`` (= ``accuracy // 2``) of finite-difference halo followed by the
    ``pml_width`` absorbing band on each side and the physical interior in the
    middle. Everything outside the physical interior is zeroed in a copy; if
    anything remains, a v-tangent overlaps the PML (or FD halo) region.
    """
    fd_pad = accuracy // 2
    shape = dv.shape[-ndim:]
    interior: list[Any] = [slice(None)] * (dv.ndim - ndim)
    for i in range(ndim):
        lo = fd_pad + int(pml_width[2 * i])
        hi = shape[i] - fd_pad - int(pml_width[2 * i + 1])
        interior.append(slice(lo, max(lo, hi)))
    edge = dv.clone()
    edge[tuple(interior)] = 0
    return bool(edge.ne(0).any())


def _scalar_forward_jvp(
    ctx: Any, tangents: tuple[Any, ...]
) -> tuple[Optional[torch.Tensor], ...]:
    """Forward-mode derivative (JVP) of the scalar propagator via Born modelling.

    The scattered field of :func:`deepwave.scalar_born` **is** the
    directional derivative of :func:`scalar` with respect to the velocity
    model: differentiating the discrete scalar recurrence in the direction
    ``dv`` reproduces exactly the Born scattered update with ``scatter = dv``
    (including the ``-2 v dv dt^2 f`` source-cell term, which arises because
    ``scalar`` pre-scales each source by ``-v^2 dt^2`` and which the Born
    scattered source supplies automatically). This rule routes
    the forward-mode tangents through the fast compiled Born backend so that
    ``torch.func.jvp`` (and ``torch.autograd.forward_ad``) run on the C/CUDA
    engine rather than requiring ``backend="eager"``.

    How the input tangents map onto the Born backend inputs (all pre-scalings are
    already baked into this Function's inputs by the differentiable preprocessing
    in :func:`scalar`, whose forward-mode derivative torch computes for us):

    * ``v`` tangent -> Born ``scatter``: drives the scattered field's
      ``2 v scatter dt^2 L(u)`` coupling term.
    * ``source_amplitudes`` tangent -> Born's scattered *source*. This
      Function's ``source_amplitudes`` input is already ``-v^2 dt^2 f``, so its
      tangent is ``-(2 v dv dt^2 f + v^2 dt^2 df)`` -- exactly the scattered
      point source Born needs, combining both the velocity and the source
      contributions in one term (linearity). We inject it directly as Born's
      pre-scaled scattered source, so no double counting with ``scatter``.
    * initial-wavefield tangents -> Born's scattered initial wavefields.

    The Born backend forward is called directly (not via ``.apply``) on concrete,
    ``no_grad`` tensors, so it builds no graph and is not re-intercepted by the
    active ``torch.func`` transform. Its scattered outputs are the output
    tangents; the background outputs (equal to the primal) are discarded.

    **PML caveat.** Born does not linearise the PML profile's dependence on ``v``
    (perturbing the artificial absorbing layer is not a physical scattering
    event), so when ``dv`` is non-zero inside the PML region this JVP differs
    from the true derivative of the discrete forward. A warning is emitted in
    that case, mirroring the physical behaviour of ``scalar_born`` itself.
    """
    # Fetch the submodule from sys.modules: the ``deepwave.scalar_born``
    # attribute is shadowed by the re-exported function of the same name in
    # ``deepwave.__init__``.
    import importlib

    scalar_born_module = importlib.import_module("deepwave.scalar_born")

    # Nested functorch transforms (e.g. jvp(jvp(...)), jvp inside grad/vmap)
    # would need the returned tangents to remain tracked at the *outer* level,
    # but this rule computes them as concrete tensors on the compiled backend,
    # which would silently detach the outer transform. Refuse rather than be
    # silently wrong. The depth is 0 under the legacy
    # ``torch.autograd.forward_ad`` API and 1 under a single ``torch.func.jvp``.
    try:
        import torch._C._functorch as _functorch_c

        depth = _functorch_c.get_dynamic_layer_stack_depth()
    except (ImportError, AttributeError):  # pragma: no cover - older torch
        depth = 0
    if depth > 1:
        raise DeepwaveNotImplementedError(
            "scalar: the fast-backend forward-mode (jvp) rule "
            "supports a single torch.func.jvp (or torch.autograd.forward_ad) "
            "level, not nested/composed torch.func transforms (e.g. "
            "jvp(grad(...)), vmap(jvp(...))). Re-run the propagator with "
            'backend="eager" to compose transforms through native '
            "autograd (slower)."
        )

    # The Born backend this rule routes through now threads the first-class
    # boundary ``bc`` codes (task O), so non-PML faces (free/rigid/periodic)
    # are handled: ``ctx.jvp_boundary`` is passed through to
    # ``ScalarBornForwardFunc.forward`` below and its forward guard fill /
    # model reads apply the reflecting image. The forward reads the models
    # (background ``v`` and scatter ``dv``) only at the current grid point, so
    # no guard-ring model mirror is needed for this forward-only rule (the
    # adjoint model mirror, which the compiled backward does need, is applied
    # inside ``scalar_born``'s own backward and by the HVP path).
    unwrap = _engine.unwrap_functorch
    ndim = len(ctx.jvp_grid_spacing)

    # Primal inputs saved via ``save_for_forward`` (same layout as backward);
    # they are always real tensors (never ``None``), so narrow the type.
    saved = [cast("torch.Tensor", unwrap(t)) for t in ctx.saved_tensors]
    v, sources_i, receivers_i, source_amplitudes = saved[:4]
    wavefields = saved[4 : 6 + 2 * ndim]
    pml_profiles = saved[6 + 2 * ndim : 6 + 5 * ndim]

    # Tangents correspond to the Function's positional inputs: the leading
    # non-tensor args, then v, source_amplitudes, sources_i, receivers_i,
    # *wavefields. sources_i / receivers_i are integer indices (no tangent).
    # ``_N_LEAD`` includes scalar's extra trailing ``boundary`` leading arg.
    n = _N_LEAD
    dv = unwrap(tangents[n])
    dsource = unwrap(tangents[n + 1])
    dwavefields = [unwrap(t) for t in tangents[n + 4 :]]

    # Everything from here runs at the base dispatch level. Under
    # ``torch.func.jvp`` the active functorch interpreter would otherwise re-wrap
    # any tensor an op touches or that the backend allocates (its storage
    # buffers), leaving them without the concrete storage the ctypes backend
    # addresses via ``data_ptr()``. The inputs were unwrapped to concrete tensors
    # above; clearing the interpreter stack (a no-op under the legacy
    # ``forward_ad`` API or with no transform active) keeps every subsequent op
    # and allocation on real memory.
    with _base_level_dispatch(), torch.no_grad():
        if dv is None:
            dv = torch.zeros_like(v)
        if dsource is None:
            dsource = torch.zeros_like(source_amplitudes)
        dwavefields = [
            dw if dw is not None else torch.zeros_like(w)
            for dw, w in zip(dwavefields, wavefields)
        ]

        # Only when the profiles really do depend on the model: their amplitude
        # is built from the maximum wave speed, so a pinned ``max_vel`` makes
        # them constants of the call and leaves nothing to linearise.
        if not ctx.jvp_max_vel_pinned and _v_tangent_in_pml(
            dv, ctx.jvp_pml_width, ctx.jvp_accuracy, ndim
        ):
            warn_at_user_call(
                "torch.func.jvp of scalar: the velocity tangent is non-zero "
                "inside the PML, where the PML profile depends on velocity "
                "through the maximum wave speed inferred from the model and "
                "the Born-based forward-mode derivative does not linearise "
                "that dependence, so the returned tangent is inexact there; "
                "pass max_vel= to pin the profiles, or set pml_width=0, if "
                "you need an exact tangent everywhere (see the scalar_born PML "
                "caveat).",
                UserWarning,
                2,
            )

        # Assemble the scalar Born backend inputs. Models: background v and
        # scatter=dv. Sources: background pre-scaled amplitudes and, as the
        # scattered source, their tangent (both share the source locations).
        # Receivers: the scattered receivers share the primal receiver locations
        # so their output is the tangent at the same points. Wavefields: primal
        # (background) then the wavefield tangents (scattered).
        born_args = (
            v,
            dv,
            source_amplitudes,
            dsource,
            sources_i,
            sources_i,
            receivers_i,
            receivers_i,
            *wavefields,
            *dwavefields,
        )
        born_out = scalar_born_module.ScalarBornForwardFunc.forward(
            list(pml_profiles),
            ctx.jvp_grid_spacing,
            ctx.jvp_dt,
            ctx.jvp_nt,
            ctx.jvp_step_ratio,
            ctx.jvp_accuracy,
            ctx.jvp_pml_width,
            ctx.jvp_n_shots,
            None,  # forward_callback
            None,  # backward_callback
            1,  # callback_frequency
            "none",  # storage_mode_str (no backward pass needed)
            ".",  # storage_path
            False,  # storage_compression
            # scalar_born's compiled backend threads the first-class boundary
            # ``bc`` codes (task O STEP C): the real ``ctx.jvp_boundary`` (not
            # forced all-PML) is passed through so non-PML faces get the
            # reflecting image in the forward guard fill.
            list(ctx.jvp_boundary),
            *born_args,
        )

    # Born outputs: (*background_wavefields, *scattered_wavefields,
    # receivers_bg, receivers_sc, saved_state). The scalar forward outputs are
    # (*wavefields, receiver_amplitudes, saved_state); their tangents are the
    # Born scattered wavefields, the scattered receivers, and None (saved_state).
    n_wf = 2 + 2 * ndim
    sc_wavefields = born_out[n_wf : 2 * n_wf]
    receivers_sc = born_out[2 * n_wf + 1]
    return (*sc_wavefields, receivers_sc, None)


class ScalarBackwardFunc(torch.autograd.Function):
    """Autograd function for the backward pass of scalar wave propagation.

    This class defines the forward and backward passes for computing gradients
    of the scalar wave equation, interfacing directly with the C/CUDA backend.
    It is typically called by `ScalarForwardFunc.backward`.
    """

    @staticmethod
    def forward(
        grad_r: torch.Tensor,
        v: torch.Tensor,
        pml_profiles: list[torch.Tensor],
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        storage_manager: _storage.StorageManager,
        raw_storage_manager: _storage.StorageManager,
        store_raw_forward: bool,
        source_amplitudes: torch.Tensor,
        grid_spacing: Sequence[float],
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: list[int],
        source_amplitudes_requires_grad: bool,
        backward_callback: Optional[Callback],
        callback_frequency: int,
        boundary: list[int],
        *wavefields_tuple: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Performs the backward propagation of the scalar wave equation.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            grad_r: Gradient of the loss with respect to `receiver_amplitudes`.
            v: The wavespeed model tensor.
            pml_profiles: List of PML profiles.
            sources_i: 1D indices of source locations.
            receivers_i: 1D indices of receiver locations.
            storage_manager: The storage manager.
            raw_storage_manager: The raw-forward-wavefield storage manager
                allocated by ``ScalarForwardFunc.forward`` (see its docstring)
                using the same storage_mode/path/compression as the primary
                checkpoint; read back (by step index for DEVICE-uncompressed,
                or from the compiled backend's just-loaded staging buffer
                otherwise) to populate
                ``CallbackState.get_wavefield(role="forward")`` for
                ``backward_callback``.
            store_raw_forward: Whether ``raw_storage_manager`` actually holds
                data (a ``backward_callback`` *and*
                ``backward_callback_needs_forward_wavefield=True`` at the
                original forward call); threaded straight to the compiled
                backend.
            source_amplitudes: The source amplitudes tensor.
            grid_spacing: Grid spacing for each spatial dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            n_shots: Number of shots in the batch.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            source_amplitudes_requires_grad: If source amplitudes need grads.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.
            boundary: Per-face boundary-type integer codes (see
                :func:`deepwave._common.set_boundary`) threaded to the compiled
                adjoint as the ``bc`` array so the guard-ring image fill matches
                the forward pass.
            wavefields_tuple: Gradients of the loss with respect to the
                output wavefields from the forward pass, followed by
                initial wavefields (current, previous, PML) from forward pass.

        Returns:
            A tuple containing the gradients of the inputs.

        """
        ndim = len(grid_spacing)
        # Only the output-gradient wavefields are consumed by this (first)
        # backward pass; the trailing initial forward wavefields in
        # ``wavefields_tuple`` are needed only by the double-backward and are
        # re-derived from ``inputs`` in ``setup_context``.
        grad_wavefields = list(wavefields_tuple[: 2 + 2 * ndim])
        del wavefields_tuple

        v = v.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        grad_wavefields = [p.contiguous() for p in grad_wavefields]
        pml_profiles = [p.contiguous() for p in pml_profiles]
        grad_r = grad_r.contiguous()

        # The backward propagation only requires the `v**2 * dt**2` term, so
        # pre-calculating it once saves computation inside the time loop. This
        # is the model pointer passed to the backend (in place of `v`).
        v2dt2 = (v**2 * dt**2).contiguous()

        def make_aux_wavefields(
            grad_wavefields: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            # Temporaries for the psi and zeta gradient PML auxiliaries; their
            # neighbours are read while updating other fields, so they cannot be
            # updated in place. Ordered [psin..., zetan...] to match psi/zeta.
            return [torch.zeros_like(grad_wavefields[2 + i]) for i in range(2 * ndim)]

        def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
            v = state.models[0]
            ndim = state.ndim
            grad_psi = state.wavefields[2 : 2 + ndim]
            grad_zeta = state.wavefields[2 + ndim : 2 + 2 * ndim]
            grad_psin = state.aux_wavefields[:ndim]
            grad_zetan = state.aux_wavefields[ndim:]
            v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
            n_sources_for_grad = (
                state.n_sources_per_shot[0] if state.grad_sources[0].numel() > 0 else 0
            )
            call_args: tuple[Any, ...] = (
                v2dt2.data_ptr(),
                state.grad_receivers[0].data_ptr(),
                state.wavefields[0].data_ptr(),
                state.wavefields[1].data_ptr(),
                *[field.data_ptr() for field in grad_psi],
                *[field.data_ptr() for field in grad_psin],
                *[field.data_ptr() for field in grad_zeta],
                *[field.data_ptr() for field in grad_zetan],
                *state.storage_manager.storage_ptrs,
                state.grad_sources[0].data_ptr(),
                state.grad_models[0].data_ptr(),
                state.grad_models_tmp_ptr[0],
                *[profile.data_ptr() for profile in state.pml_profiles],
                state.sources_i[0].data_ptr(),
                state.receivers_i[0].data_ptr(),
                *state.rdx,
                *state.rdx2,
                step_nt,
                state.n_shots,
                *state.model_shape,
                n_sources_for_grad,
                state.n_receivers_per_shot[0],
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                state.storage_grad_flag(v.requires_grad),
                v_batched,
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                *raw_storage_manager.storage_ptrs,
                store_raw_forward,
                raw_storage_manager.storage_mode,
                raw_storage_manager.shot_bytes_uncomp,
                raw_storage_manager.shot_bytes_comp,
                raw_storage_manager.storage_compression,
                state.aux,
                state.stream,
            )
            return cast("int", state.backend(*call_args))

        wf_names = scalar_wavefield_names(ndim)
        # DEVICE-resident, uncompressed raw-forward storage keeps the full
        # per-step history directly indexable from Python (unchanged from
        # before this buffer honoured storage_mode): store_1a has one slot
        # per step. Every other mode/compression combination -- CPU, disk, or
        # DEVICE with compression -- allocates store_1a as a single-step
        # staging buffer (see StorageManager.allocate); the compiled backward
        # kernel loads (and, for compression, decompresses) the step this
        # callback is about to fire for into that buffer immediately before
        # returning from each chunk's call (see FUNC(backward) in
        # scalar.c/scalar.cu), so reading it directly (no step index) is
        # correct at the point this closure runs. storage_mode="none"
        # allocates nothing at all (StorageManager.allocate is a no-op in
        # that mode, matching the primary checkpoint): there is no per-step
        # data to return, so this returns None (a step with no forward
        # wavefields) rather than reading past the end of an empty tensor;
        # get_wavefield(role="forward") raises a clear error only if and
        # when a callback actually tries to use it, so a callback that never
        # asks for role="forward" (the common case) is unaffected.
        raw_is_full_history = (
            raw_storage_manager.storage_mode == _storage.StorageMode.DEVICE
            and not raw_storage_manager.storage_compression
        )
        raw_is_stored = raw_storage_manager.storage_mode != _storage.StorageMode.NONE

        def forward_wavefields_for_step(
            step: int,
        ) -> Optional[dict[str, torch.Tensor]]:
            if not raw_is_stored:
                return None
            store_1a = raw_storage_manager.storages[0].store_1a
            history_per_user = step_ratio // getattr(
                callback_frequency, "user_step_ratio", step_ratio
            )
            raw = (
                store_1a[step // history_per_user] if raw_is_full_history else store_1a
            )
            return {wf_names[0]: raw}

        def swap(state: _engine.BackendState) -> None:
            # Ping-pong the current/previous gradient wavefields and the
            # psi/zeta temporaries after a chunk with an odd number of steps.
            ndim = state.ndim
            w = state.wavefields
            aux = state.aux_wavefields
            w[0], w[1] = w[1], w[0]
            w[2 : 2 + 2 * ndim], aux[:] = aux[:], w[2 : 2 + 2 * ndim]

        def negate_prev(state: _engine.BackendState) -> None:
            # A mathematical trick avoids negating a wavefield every time step:
            # negating the previous-wavefield gradient before and after the loop
            # lets the loop use a simple pointer swap with a modified update.
            state.wavefields[1] = -state.wavefields[1]

        state = _engine.run_backward_core(
            SPEC,
            models=[v],
            sources_i=[sources_i],
            receivers_i=[receivers_i],
            source_amplitudes=[source_amplitudes],
            pml_profiles=pml_profiles,
            grad_wavefields=grad_wavefields,
            grad_receivers=[grad_r],
            grid_spacing=list(grid_spacing),
            dt=dt,
            nt=nt,
            n_shots=n_shots,
            step_ratio=step_ratio,
            accuracy=accuracy,
            pml_width=pml_width,
            source_amplitudes_requires_grad=[source_amplitudes_requires_grad],
            storage_manager=storage_manager,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            make_aux_wavefields=make_aux_wavefields,
            zero_wavefields=_zero_pml_interiors,
            make_call=make_call,
            swap=swap,
            pre_loop=negate_prev,
            post_loop=negate_prev,
            boundary=boundary,
            forward_wavefields=(
                forward_wavefields_for_step if store_raw_forward else None
            ),
        )

        out_s = state.output_slice
        return (
            state.grad_models[0],
            state.grad_sources[0],
            *[w[out_s] for w in state.wavefields],
        )

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Saves the tensors and metadata the double-backward pass needs.

        Every value comes from ``inputs`` (the arguments to ``forward``); the
        double-backward re-runs a fresh Born forward+backward from these, so no
        forward-computed intermediate has to be threaded through ``output``.
        """
        (
            grad_r,
            v,
            pml_profiles,
            sources_i,
            receivers_i,
            storage_manager,
            _raw_storage_manager,
            _store_raw_forward,
            source_amplitudes,
            grid_spacing,
            dt,
            nt,
            n_shots,
            step_ratio,
            accuracy,
            pml_width,
            source_amplitudes_requires_grad,
            _backward_callback,
            _callback_frequency,
            boundary,
            *wavefields_tuple,
        ) = inputs
        ndim = len(grid_spacing)
        grad_wavefields = list(wavefields_tuple[: 2 + 2 * ndim])
        wavefields = list(wavefields_tuple[2 + 2 * ndim :])
        ctx.save_for_backward(
            v,
            sources_i,
            receivers_i,
            source_amplitudes,
            grad_r,
            *wavefields,
            *grad_wavefields,
            *pml_profiles,
        )
        ctx.grid_spacing = grid_spacing
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.pml_width = pml_width
        ctx.source_amplitudes_requires_grad = source_amplitudes_requires_grad
        ctx.storage_mode = storage_manager.storage_mode
        ctx.storage_path = storage_manager.storage_path
        ctx.storage_compression = storage_manager.storage_compression
        ctx.boundary = boundary

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Performs double backpropagations.

        Args:
            *args: The gradients with respect to the outputs.
        This method implements the backward pass for `ScalarBackwardFunc`, which
        is itself the backward pass of the forward propagation. This
        "double backward" operation is used to compute second-order
        derivatives of the original objective function with respect to the
        model parameters.

        The primary goal of this function is to compute the Hessian-vector
        product (HVP), `H*h`, where `H` is the Hessian of the original
        scalar loss function `J`, and `h` is a perturbation vector.

        This is achieved using the scalar Born propagator, which enables
        calculation of the effect a perturbation of the inputs will have
        on the gradients.

        The background wavefield of the forward pass will be regular forward
        propagation (repeating what was already done during the original
        forward propagation), while the scattered field will calculate the
        effect of the perturbation on the forward wavefield.

        The scattered field of the backward pass will be regular backpropagation
        (repeating what was already done during the original backpropagation),
        while the background wavefield will represent the effect of the
        velocity perturbation on the backward wavefield.

        Args:
            ctx: A context object that contains information saved during the
                forward pass of the backward function.
            ggv: Gradient of the loss with respect to the gradient of `v`.
            ggf: Gradient of the loss with respect to the gradient of
                `source_amplitudes`.
            ggwfc: Gradient of the loss with respect to the gradient of `wfc`.
            ggwfp: Gradient of the loss with respect to the gradient of `wfp`.
            ggpsiy: Gradient of the loss with respect to the gradient of `psiy`.
            ggpsix: Gradient of the loss with respect to the gradient of `psix`.
            ggzetay: Gradient of the loss with respect to the gradient of
                `zetay`.
            ggzetax: Gradient of the loss with respect to the gradient of
                `zetax`.

        Returns:
            Second-order gradients with respect to the inputs of the
            `ScalarBackwardFunc.forward` method.

        """
        # The Born adjoint also samples model derivatives when the seed is
        # a source or initial-state gradient. Reject the whole second-order
        # path; checking only model-gradient seeds misses mixed derivatives.
        if ctx.step_ratio > 1:
            raise DeepwaveRuntimeError(
                "scalar: compiled double backward with sampled model gradients "
                "is unsupported because the sampled gradient's Jacobian is not "
                "symmetric. Use backend='eager' or choose a time step and "
                "model_gradient_sampling_interval giving an effective stride of 1."
            )
        # Retrieve all the tensors saved from the forward and first backward passes.
        # This includes the original forward wavefields (`wfc`, `wfp`), the
        # first-order adjoint sources (`grad_r`), the final state of the first-
        # order adjoint wavefields (`gwfc`, `gwfp`), and model parameters.
        ndim = len(ctx.grid_spacing)
        # The double-backward (HVP) routes through the scalar Born backend,
        # which now threads the first-class boundary ``bc`` codes (task O): the
        # two direct compiled-backend calls below pass ``ctx.boundary`` through
        # so the guard fill applies the reflecting image, and the adjoint call
        # mirror-pads the model coefficients' guard rings (see the
        # ``mirror_pad_scalar_born_models`` call before the backward pass).
        ggv = args[0]
        ggf = args[1]
        gradgrad_wavefields = list(args[2:])
        del args
        # Read ``ctx.saved_tensors`` exactly once (see the note in
        # ``ScalarForwardFunc.backward``): repeated access breaks
        # ``torch.utils.checkpoint``.
        saved_tensors = ctx.saved_tensors
        (
            v,
            sources_i,
            receivers_i,
            source_amplitudes,
            grad_r,
        ) = saved_tensors[:5]
        wavefields = saved_tensors[5 : 7 + 2 * ndim]
        grad_wavefields = saved_tensors[7 + 2 * ndim : 9 + 4 * ndim]
        pml_profiles = saved_tensors[9 + 4 * ndim : 9 + 7 * ndim]
        # Retrieve scalar variables from the context.
        grid_spacing = ctx.grid_spacing
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        storage_mode = ctx.storage_mode
        storage_path = ctx.storage_path
        storage_compression = ctx.storage_compression

        # If no gradient is provided for a given input (e.g. `ggv`), it means
        # we are not interested in the second derivative with respect to that
        # term. We create a zero tensor as a placeholder for the calculation.
        if ggv.numel() == 0:
            ggv = torch.zeros_like(v.detach())
        if ggf.numel() == 0:
            ggf = torch.zeros_like(source_amplitudes.detach())

        v = v.contiguous()
        ggv = ggv.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        ggsource_amplitudes = ggf.contiguous()
        grad_r = grad_r.contiguous()
        pml_profiles = [p.contiguous() for p in pml_profiles]
        fwd_sources_i = sources_i.contiguous()

        # Step 1: Forward pass

        # The forward Born propagation will re-calculate the original forward
        # wavefield `u` and the perturbed forward wavefield `delta_u`.
        # We need snapshots of these fields over the whole grid, but we do not
        # need to record `u` at any specific receiver locations for the double
        # backward pass itself, so `fwd_receivers_i` is empty.
        fwd_receivers_i = torch.empty(0, dtype=torch.long, device=v.device)
        # We do, however, need to record `delta_u` at the original receiver
        # locations if we want to compute the gradient with respect to the
        # adjoint source `grad_r`. So we provide the original receiver
        # locations for the scattered field receivers.
        fwd_ggreceivers_i = receivers_i.contiguous()

        fd_pad = accuracy // 2
        model_shape = v.shape[-ndim:]
        size_with_batch = (n_shots, *model_shape)
        wavefields = [
            _engine.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in wavefields
        ]
        gradgrad_wavefields = [
            _engine.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in gradgrad_wavefields
        ]
        wfc, wfp, *pml_wavefields = wavefields
        del wavefields
        gradgrad_wfc, gradgrad_wfp, *gradgrad_pml_wavefields = gradgrad_wavefields
        del gradgrad_wavefields
        psi: list[torch.Tensor] = pml_wavefields[:ndim]
        zeta: list[torch.Tensor] = pml_wavefields[ndim:]
        del pml_wavefields
        gradgrad_psi: list[torch.Tensor] = gradgrad_pml_wavefields[:ndim]
        gradgrad_zeta: list[torch.Tensor] = gradgrad_pml_wavefields[ndim:]
        del gradgrad_pml_wavefields
        for i in range(ndim):
            psi[i] = _common.zero_interior(psi[i], fd_pad, pml_width, i)
            zeta[i] = _common.zero_interior(zeta[i], fd_pad, pml_width, i)
            gradgrad_psi[i] = _common.zero_interior(
                gradgrad_psi[i], fd_pad, pml_width, i
            )
            gradgrad_zeta[i] = _common.zero_interior(
                gradgrad_zeta[i], fd_pad, pml_width, i
            )

        device = v.device
        dtype = v.dtype
        is_cuda = v.is_cuda
        n_sources_per_shot = fwd_sources_i.numel() // n_shots
        n_receivers_per_shot = fwd_receivers_i.numel() // n_shots
        n_ggreceivers_per_shot = fwd_ggreceivers_i.numel() // n_shots
        psin = [torch.zeros_like(wf) for wf in psi]
        gradgrad_psin = [torch.zeros_like(wf) for wf in gradgrad_psi]
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        ggreceiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        pml_b = [
            min(pml_width[2 * i] + 2 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 2 * fd_pad)
            for i in range(ndim)
        ]

        v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
        ggv_batched = ggv.ndim == ndim + 1 and ggv.shape[0] > 1

        # Storage allocation
        storage_manager = _storage.StorageManager(
            model_shape,
            dtype,
            n_shots,
            nt,
            step_ratio,
            storage_mode,
            storage_compression,
            storage_path,
            device,
            is_cuda,
        )
        for requires_grad in [v.requires_grad, v.requires_grad]:
            storage_manager.allocate(requires_grad)

        if fwd_receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)
        if fwd_ggreceivers_i.numel() > 0:
            ggreceiver_amplitudes.resize_(nt, n_shots, n_ggreceivers_per_shot)
            ggreceiver_amplitudes.fill_(0)

        aux, stream = _engine.get_aux_and_stream(
            v, n_shots, is_cuda, spatial_parallel=True
        )
        forward = _backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "forward",
            accuracy,
            dtype,
            v.device,
        )

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        if wfc.numel() > 0 and nt > 0:
            status = forward(
                v.data_ptr(),
                ggv.data_ptr(),
                source_amplitudes.data_ptr(),
                ggsource_amplitudes.data_ptr(),
                wfc.data_ptr(),
                wfp.data_ptr(),
                *[field.data_ptr() for field in psi],
                *[field.data_ptr() for field in psin],
                *[field.data_ptr() for field in zeta],
                gradgrad_wfc.data_ptr(),
                gradgrad_wfp.data_ptr(),
                *[field.data_ptr() for field in gradgrad_psi],
                *[field.data_ptr() for field in gradgrad_psin],
                *[field.data_ptr() for field in gradgrad_zeta],
                *storage_manager.storage_ptrs,
                receiver_amplitudes.data_ptr(),
                ggreceiver_amplitudes.data_ptr(),
                *[profile.data_ptr() for profile in pml_profiles],
                fwd_sources_i.data_ptr(),
                fwd_receivers_i.data_ptr(),
                fwd_ggreceivers_i.data_ptr(),
                *rdx,
                *rdx2,
                dt**2,
                nt,
                n_shots,
                *model_shape,
                n_sources_per_shot,
                n_receivers_per_shot,
                n_ggreceivers_per_shot,
                step_ratio,
                storage_manager.storage_mode,
                storage_manager.shot_bytes_uncomp,
                storage_manager.shot_bytes_comp,
                v.requires_grad
                and storage_manager.storage_mode != _storage.StorageMode.NONE,
                False,
                v_batched,
                ggv_batched,
                storage_manager.storage_compression,
                0,
                *pml_b,
                *pml_e,
                # scalar_born's compiled backend threads the first-class
                # boundary ``bc`` codes (task O STEP C): the real
                # ``ctx.boundary`` (not forced all-PML) is passed through so
                # non-PML faces get the reflecting image in the forward guard
                # fill (the forward reads the models only at the current grid
                # point, so no model guard-ring mirror is needed here).
                *_engine.bc_begins_ends(ctx.boundary, ndim),
                aux,
                stream,
            )
            if status != 0:
                raise DeepwaveRuntimeError(
                    _engine.backend_failure_message("scalar", "forward", status)
                )

        # Step 2: Backward pass

        bwd_sources_i = sources_i.contiguous()
        bwd_receivers_i = torch.empty(0, dtype=torch.long, device=v.device)
        bwd_greceivers_i = receivers_i.contiguous()

        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        n_sources_per_shot = bwd_sources_i.numel() // n_shots
        n_receivers_per_shot = bwd_receivers_i.numel() // n_shots
        n_greceivers_per_shot = bwd_greceivers_i.numel() // n_shots

        # Set the initial wavefields for the background propagation to zero.
        wfc = _engine.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfp = _engine.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psi = [
            _engine.create_or_pad(
                torch.empty(0),
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for _ in range(ndim)
        ]
        zeta = [
            _engine.create_or_pad(
                torch.empty(0),
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for _ in range(ndim)
        ]
        psin = [torch.zeros_like(wf) for wf in psi]
        zetan = [torch.zeros_like(wf) for wf in zeta]
        grad_wavefields = [
            _engine.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in grad_wavefields
        ]
        grad_wfc, grad_wfp, *grad_pml_wavefields = grad_wavefields
        del grad_wavefields
        grad_psi = grad_pml_wavefields[:ndim]
        grad_zeta = grad_pml_wavefields[ndim:]
        del grad_pml_wavefields
        for i in range(ndim):
            grad_psi[i] = _common.zero_interior(grad_psi[i], fd_pad, pml_width, i)
            grad_zeta[i] = _common.zero_interior(grad_zeta[i], fd_pad, pml_width, i)
        grad_psin = [torch.zeros_like(wf) for wf in grad_psi]
        grad_zetan = [torch.zeros_like(wf) for wf in grad_zeta]
        grad_f = torch.empty(0, device=device, dtype=dtype)
        grad_gf = torch.empty(0, device=device, dtype=dtype)
        pml_b = [
            min(pml_width[2 * i] + 3 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 3 * fd_pad)
            for i in range(ndim)
        ]

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)

        aux, stream = _engine.get_aux_and_stream(
            v, n_shots, is_cuda, spatial_parallel=True
        )
        grad_v_list, _, grad_v_tmp_ptr_list = _engine.allocate_model_gradients(
            [v],
            [v_batched],
            list(model_shape),
            n_shots,
            aux,
            is_cuda,
            storage_manager.storage_mode,
            device,
            dtype,
            spatial_parallel=True,
        )
        grad_v = grad_v_list[0]
        grad_v_tmp_ptr = grad_v_tmp_ptr_list[0]

        backward = _backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
        )

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        grad_wfp = -grad_wfp

        # The compiled scalar_born adjoint reads both model coefficients
        # (background ``v`` and the scattered-model ``ggv``) at neighbour
        # offsets next to a reflecting face, so their guard rings must hold the
        # even/periodic-wrap image there (exactly as ``scalar_born``'s own
        # backward does). An all-``BC_PML`` boundary is a no-op; otherwise fresh
        # mirror-padded clones are used for the compiled call so the saved
        # tensors are never written in place. ``grad_v`` was already allocated
        # from the original ``v`` (same shape, so unaffected).
        v, ggv = _boundaries.mirror_pad_scalar_born_models(
            [v, ggv], ctx.boundary, fd_pad
        )

        if wfc.numel() > 0 and nt > 0 and v.requires_grad:
            status = backward(
                v.data_ptr(),
                ggv.data_ptr(),
                torch.empty(0).data_ptr(),
                grad_r.data_ptr(),
                wfc.data_ptr(),
                wfp.data_ptr(),
                *[field.data_ptr() for field in psi],
                *[field.data_ptr() for field in psin],
                *[field.data_ptr() for field in zeta],
                *[field.data_ptr() for field in zetan],
                grad_wfc.data_ptr(),
                grad_wfp.data_ptr(),
                *[field.data_ptr() for field in grad_psi],
                *[field.data_ptr() for field in grad_psin],
                *[field.data_ptr() for field in grad_zeta],
                *[field.data_ptr() for field in grad_zetan],
                *storage_manager.storage_ptrs,
                grad_f.data_ptr(),
                grad_gf.data_ptr(),
                grad_v.data_ptr(),
                torch.empty(0).data_ptr(),
                grad_v_tmp_ptr,
                torch.empty(0).data_ptr(),
                *[profile.data_ptr() for profile in pml_profiles],
                bwd_sources_i.data_ptr(),
                bwd_receivers_i.data_ptr(),
                bwd_greceivers_i.data_ptr(),
                *rdx,
                *rdx2,
                dt**2,
                nt,
                n_shots,
                *model_shape,
                n_sources_per_shot * source_amplitudes_requires_grad,
                0,
                n_receivers_per_shot,
                n_greceivers_per_shot,
                step_ratio,
                storage_manager.storage_mode,
                storage_manager.shot_bytes_uncomp,
                storage_manager.shot_bytes_comp,
                v.requires_grad
                and storage_manager.storage_mode != _storage.StorageMode.NONE,
                False,
                v_batched,
                ggv_batched,
                storage_manager.storage_compression,
                nt,
                *pml_b,
                *pml_e,
                # scalar_born's compiled backend threads the first-class
                # boundary ``bc`` codes (task O STEP C): the real
                # ``ctx.boundary`` (not forced all-PML) is passed through so
                # non-PML faces get the reflecting image in the adjoint guard
                # fill; the model coefficients (``v``, ``ggv``) were already
                # mirror-padded above.
                *_engine.bc_begins_ends(ctx.boundary, ndim),
                aux,
                stream,
            )
            if status != 0:
                raise DeepwaveRuntimeError(
                    _engine.backend_failure_message("scalar", "backward", status)
                )

        s = (
            slice(None),
            *(slice(fd_pad, dim - fd_pad) for dim in model_shape),
        )
        if nt % 2 == 0:
            return (
                ggreceiver_amplitudes,
                grad_v,
                None,
                None,
                None,
                None,
                None,  # raw_storage_manager
                None,  # store_raw_forward
                grad_f,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,  # boundary
                gradgrad_wfc[s],
                gradgrad_wfp[s],
                *[wf[s] for wf in gradgrad_psi],
                *[wf[s] for wf in gradgrad_zeta],
                wfc[s],
                -wfp[s],
                *[wf[s] for wf in psi],
                *[wf[s] for wf in zeta],
            )
        return (
            ggreceiver_amplitudes,
            grad_v,
            None,
            None,
            None,
            None,
            None,  # raw_storage_manager
            None,  # store_raw_forward
            grad_f,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # boundary
            gradgrad_wfp[s],
            gradgrad_wfc[s],
            *[wf[s] for wf in gradgrad_psin],
            *[wf[s] for wf in gradgrad_zeta],
            wfp[s],
            -wfc[s],
            *[wf[s] for wf in psin],
            *[wf[s] for wf in zetan],
        )


def _forward_step(
    v: torch.Tensor,
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
    pml_profiles: list[torch.Tensor],
    wavefields: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Performs a single time step of scalar wave propagation."""
    ndim = len(grid_spacing)

    wfc, wfp = wavefields[:2]
    psi: list[torch.Tensor] = wavefields[2 : 2 + ndim]
    zeta: list[torch.Tensor] = wavefields[2 + ndim :]
    del wavefields
    a = pml_profiles[::3]
    b = pml_profiles[1::3]
    dbdx = pml_profiles[2::3]
    del pml_profiles
    dudx = [
        _regular_grid.diff1(wfc, dim, accuracy, 1 / grid_spacing[dim], ndim)
        for dim in range(ndim)
    ]
    tmp = [
        (1 + b[dim])
        * _regular_grid.diff2(wfc, dim, accuracy, 1 / grid_spacing[dim] ** 2, ndim)
        + dbdx[dim] * dudx[dim]
        + _regular_grid.diff1(
            a[dim] * psi[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    w_sum = torch.zeros_like(tmp[0])
    for dim in range(ndim):
        w_sum = w_sum + (1 + b[dim]) * tmp[dim] + a[dim] * zeta[dim]

    return (
        [
            v**2 * dt * dt * w_sum + 2 * wfc - wfp,
            wfc,
        ]
        + [b[dim] * dudx[dim] + a[dim] * psi[dim] for dim in range(ndim)]
        + [b[dim] * tmp[dim] + a[dim] * zeta[dim] for dim in range(ndim)]
    )


_forward_step_cache = _engine.StepCache(_forward_step)
_forward_step_opt = _forward_step

# Named offsets into the leading (pre-tensor-block) args scalar_func receives
# from its call site (see _backend_utils.PROPAGATOR_REGISTRY["scalar"]
# .leading_args): the index of "boundary", and the index one past the last
# leading arg -- i.e. where the model/source/wavefield tensor block starts.
_BOUNDARY_IDX = _backend_utils.leading_arg_offset("scalar", "boundary")
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["scalar"].leading_args)


def scalar_python(
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    boundary: list[int],
    n_shots: int,
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    storage_mode_str: StorageMode,
    storage_path: str,
    storage_compression: bool,
    backward_callback_needs_forward_wavefield: bool,
    *args: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Performs the forward propagation of the scalar wave equation.

    This method is written purely in PyTorch, rather than using the
    compiled C/CUDA code. ``args`` is the flat tensor tuple
    ``(v, source_amplitudes, sources_i, receivers_i, *wavefields)``.

    Args:
        pml_profiles: List of PML profiles.
        grid_spacing: List of grid spacing in each spatial dimension.
        dt: Time step interval.
        nt: Total number of time steps.
        step_ratio: Ratio between user dt and internal dt.
        accuracy: Finite difference accuracy order.
        pml_width: List of PML widths for each side.
        boundary: Per-face boundary-type integer codes (see
            :func:`deepwave._common.set_boundary`); ``BC_PML`` faces are absorbing
            and left untouched.
        n_shots: Number of shots in the batch.
        forward_callback: The forward callback.
        backward_callback: The backward callback.
        callback_frequency: The callback frequency, in user time steps.
        backward_callback_needs_forward_wavefield: Whether to retain the
            forward wavefields so ``backward_callback`` can ask for
            ``role="forward"``. See :func:`deepwave.scalar`.
        storage_mode_str: The storage mode (Unused).
        storage_path: The path to disk storage (Unused).
        storage_compression: Whether to apply compression to storage (Unused).
        args: The flat tensor tuple (v, source_amplitudes, sources_i,
            receivers_i, *wavefields).
        backend: The resolved Python backend name (``"eager"`` or
            ``"compile"``), already selected by the caller; used only to name
            it if an unsupported option is rejected.

    Returns:
        A tuple containing the output wavefields and receiver amplitudes.

    """
    ndim = len(grid_spacing)
    num_wf = 2 + 2 * ndim
    args_list = list(args)
    wavefields = args_list[-num_wf:]
    receivers_i = args_list[-num_wf - 1]
    sources_i = args_list[-num_wf - 2]
    source_amplitudes = args_list[-num_wf - 3]
    v = args_list[-num_wf - 4]

    is_batched = any(
        _common.is_inside_vmap(x)
        for x in [v, source_amplitudes, sources_i, receivers_i, *wavefields]
    )
    _engine.reject_unsupported_python_backend_options(
        SPEC,
        backend,
        storage_mode=storage_mode_str,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
        is_batched=is_batched,
    )
    fd_pad = accuracy // 2
    has_boundary = any(code != _common.BC_PML for code in boundary)
    size_with_batch = (n_shots, *v.shape[1:])
    wavefields = [
        _engine.create_or_pad(
            wavefield,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        for wavefield in wavefields
    ]

    device = v.device
    dtype = v.dtype
    shape = v.shape[1:]
    flat_shape = int(torch.prod(torch.tensor(shape)).item())
    ndim = len(shape)
    n_receivers_per_shot = receivers_i.numel() // n_shots
    receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudes_list: list[torch.Tensor] = []

    if receivers_i.numel() > 0 and not is_batched:
        receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
        receiver_amplitudes.fill_(0)

    sources_i_masked, source_mask = _common.mask_locations(sources_i)
    source_amplitudes_masked = _common.mask_source_amplitudes(
        source_amplitudes, source_mask
    )
    receivers_i_masked, receiver_mask = _common.mask_locations(receivers_i)

    wf_names = scalar_wavefield_names(ndim)

    def _register_backward_callback(cb_step: int, wfs: list[torch.Tensor]) -> None:
        """Fire ``backward_callback`` during backward, with forward+adjoint fields.

        The compiled backend cannot surface the forward wavefield in the backward
        callback (it stores the gradient-correlation kernel, not the raw field),
        so in the pure-PyTorch eager backend we obtain the time-coincident
        adjoint field for free: it is the gradient w.r.t. the step's primary
        wavefield tensor. We register an autograd hook on that tensor; when the
        backward pass reaches this step it fires the callback with the adjoint
        field (the hook's ``grad``) and the forward wavefields captured now.

        Autograd may deliver an *undefined* gradient to the hook (it arrives in
        Python as ``None``), which in autograd's semantics means an identically
        zero gradient; it happens whenever the propagator's output turns out to
        be unused on the path being differentiated, and is exactly what
        ``torch.autograd.gradcheck``'s ``check_undefined_grad`` deliberately
        induces. We materialise it as an explicit tensor of zeros so the
        callback always receives a real, mathematically correct adjoint field
        and always fires on its documented schedule. This mirrors both the
        compiled backend (where autograd's ``materialize_grads`` already hands
        the callback a zero-filled adjoint field in the same situation) and
        ``torch.autograd.grad``'s own default treatment of unused inputs. The
        materialisation costs an allocation only in the undefined case; the
        ordinary case pays a single ``is None`` test.
        """
        w_fwd = wfs[0]
        if not w_fwd.requires_grad:
            # The step's wavefield is outside the autograd graph. This is
            # typically the zero-filled initial state at step 0, which the
            # compiled backend nevertheless reaches on its backward sweep (so
            # it fires the callback there). If the propagation is being
            # differentiated at all, splice the wavefield into the graph as a
            # leaf so autograd delivers its adjoint too and both backends fire
            # on the same schedule. If nothing requires grad, no backward pass
            # can reach this step on any backend; return rather than mark the
            # tensor, which would make an otherwise gradient-free
            # propagation's outputs require grad.
            differentiated = torch.is_grad_enabled() and (
                v.requires_grad
                or source_amplitudes_masked.requires_grad
                or any(wf.requires_grad for wf in wfs)
            )
            if not differentiated:
                return
            w_fwd = w_fwd.detach().requires_grad_()
            wfs[0] = w_fwd
        # Retaining the whole forward field per callback step is what makes
        # role="forward" possible, and it is not free: it is opted into. The
        # undefined-gradient path below needs only the primary wavefield's
        # shape/dtype/device to build its zeros, so that much is always kept
        # (three cheap scalars), while the tensors themselves are held only on
        # request.
        forward_snapshot = (
            {n: wf.detach() for n, wf in zip(wf_names, wfs)}
            if backward_callback_needs_forward_wavefield
            else None
        )
        zeros_shape = wfs[0].shape
        zeros_dtype = wfs[0].dtype
        zeros_device = wfs[0].device

        def _hook(grad: Optional[torch.Tensor]) -> None:
            adjoint = (
                torch.zeros(zeros_shape, dtype=zeros_dtype, device=zeros_device)
                if grad is None
                else grad
            )
            backward_callback(  # type: ignore[misc]
                CallbackState(
                    dt * step_ratio,
                    cb_step,
                    {wf_names[0]: adjoint},
                    dict(zip(scalar_model_names(ndim), [v])),
                    {},
                    [fd_pad] * ndim * 2,
                    pml_width,
                    forward_wavefields=forward_snapshot,
                    direction="backward",
                    n_steps=nt // step_ratio,
                    internal_dt=dt,
                )
            )

        w_fwd.register_hook(_hook)  # type: ignore[no-untyped-call]

    for step in range(nt // step_ratio):
        if step % callback_frequency == 0:
            if forward_callback is not None:
                _engine.invoke_forward_callback(
                    forward_callback,
                    CallbackState(
                        dt * step_ratio,
                        step,
                        dict(zip(wf_names, wavefields)),
                        dict(zip(scalar_model_names(ndim), [v])),
                        {},
                        [fd_pad] * ndim * 2,
                        pml_width,
                        direction="forward",
                        n_steps=nt // step_ratio,
                        internal_dt=dt,
                    ),
                    differentiable=torch.is_grad_enabled()
                    and any(
                        t.requires_grad for t in args if isinstance(t, torch.Tensor)
                    ),
                )
            if backward_callback is not None:
                _register_backward_callback(step, wavefields)
        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step
            if receivers_i_masked.numel() > 0:
                val = wavefields[0].view(-1, flat_shape).gather(1, receivers_i_masked)
                if is_batched:
                    receiver_amplitudes_list.append(val)
                else:
                    receiver_amplitudes[t] = val
            if has_boundary:
                wavefields[0] = _boundaries.fill_scalar_boundary(
                    wavefields[0], boundary, fd_pad
                )
            wavefields = _forward_step_opt(
                v, grid_spacing, dt, accuracy, pml_profiles, wavefields
            )
            if source_amplitudes_masked.numel() > 0:
                wavefields[0].view(-1, flat_shape).scatter_add_(
                    1, sources_i_masked, source_amplitudes_masked[t]
                )

    if is_batched and len(receiver_amplitudes_list) > 0:
        receiver_amplitudes = torch.stack(receiver_amplitudes_list)

    receiver_amplitudes_masked = _common.mask_receiver_amplitudes(
        receiver_amplitudes, receiver_mask
    )
    del receiver_amplitudes
    s = (
        slice(None),
        *(slice(fd_pad, -fd_pad) for _ in range(ndim)),
    )
    return tuple(
        [wavefield[s] for wavefield in wavefields] + [receiver_amplitudes_masked]
    )


def scalar_func(
    backend: Optional[Literal["eager", "compile"]],
    *args: Any,
    max_vel_pinned: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Helper function to apply the ScalarForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ScalarForwardFunc`, which is the entry point for the autograd graph
    for scalar wave propagation.

    Args:
        backend: None, or a string specifying whether to use the
            Python backend. Already coerced (no bool) by the caller.
        *args: Variable length argument list to be passed directly to
            `ScalarForwardFunc.apply`.
        max_vel_pinned: Whether the caller supplied `max_vel`, so that the
            PML profiles do not depend on the velocity model. Threaded to
            `ScalarForwardFunc` for the forward-mode (jvp) rule's caveat.

    Returns:
        The results of the forward pass from `ScalarForwardFunc.apply`.

    """
    global _forward_step_opt

    if backend:
        v = cast("torch.Tensor", args[_N_LEADING_ARGS])
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        config_key = (len(args[1]), args[5], v.dtype)
        _forward_step_opt = _forward_step_cache.resolve(backend, config_key)
        return scalar_python(*args, backend=backend)

    # The compiled backend now threads the first-class boundary codes through
    # the ``bc`` ABI, so move ``boundary`` to the *last* leading (non-tensor)
    # autograd-Function argument (scalar's local +1 leading-arg convention,
    # ``_N_LEAD``) and keep it, rather than stripping it.
    boundary = args[_BOUNDARY_IDX]
    args = (
        args[:_BOUNDARY_IDX]
        + args[_BOUNDARY_IDX + 1 : _N_LEADING_ARGS]
        + (max_vel_pinned, boundary)
        + args[_N_LEADING_ARGS:]
    )

    func = ScalarForwardFunc.apply

    # ``forward`` appends a ``ForwardSavedState`` as its trailing output (so
    # ``setup_context`` can populate ``ctx`` without the old-style ``ctx`` arg);
    # strip it so callers see only the wavefield/receiver tensors.
    return cast(
        "tuple[torch.Tensor, ...]",
        func(*args)[:-1],  # type: ignore[no-untyped-call]
    )


_common.document_propagator_module(Scalar, scalar)

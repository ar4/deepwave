"""Variable-density acoustic wave propagation module for Deepwave.

Implements variable-density acoustic wave equation propagation using a
velocity-pressure formulation on a staggered grid with finite differences
in time (2nd order) and space (user-selectable order: 2, 4, 6, or 8).
Supports CPML boundaries and adjoint mode for gradient computation.

The equations solved are:
    rho * d(v)/dt = -grad(p) + f
    1/K * d(p)/dt = -div(v) + s

where p is pressure, v is particle velocity, rho is density, K is bulk modulus,
f is the body-force-density source term (units of force/volume) and s is the
volume-injection source term (injected volume per unit volume per unit time,
so units of 1/time).

The grid uses a staggered layout:
    - Pressure p and bulk modulus K are at integer grid points (y, x).
    - Velocity v and density rho are at half-integer grid points.
      e.g. in 2D: vy at (y+1/2, x), vx at (y, x+1/2).

    Required inputs: wavespeed model (`v`), density model (`rho`),
grid cell size (`grid_spacing`), time step (`dt`), and either a source term
or number of time steps (`nt`).
Outputs: final wavefields (pressure, velocities, and the absorbing boundary's
memory variables, carried opaquely in an auxiliary state) and receiver
amplitudes (empty if no receivers).
All outputs are differentiable with respect to float torch.Tensor inputs.
"""

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave
from deepwave import (
    _backend_utils,
    _boundaries,
    _common,
    _engine,
    _staggered_grid,
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


def acoustic_wavefield_names(ndim: int) -> list[str]:
    """Names of the acoustic wavefields, in the order the backend expects them.

    The pressure (at integer grid points) is followed by the velocity
    components, then the two PML memory variables for each spatial dimension:
    ``phi`` (used in the pressure update) and ``psi`` (used in the velocity
    update).
    """
    dims = _dim_names(ndim)
    return (
        ["pressure_0"]
        + [f"v{d}_0" for d in dims]
        + [f"phi_{d}_0" for d in dims]
        + [f"psi_{d}_0" for d in dims]
    )


def acoustic_pml_slab_axes(ndim: int) -> dict[str, int]:
    """PML memory-variable wavefields and the axis each is localised along.

    ``phi_{d}`` (pressure update) and ``psi_{d}`` (velocity update) are nonzero
    only within that dimension's two PML slabs: the compiled kernels write them
    only where that axis's PML flag is set (the ``if (pml_z)`` gating in
    ``acoustic_kernels.h``; in the backward velocity update the write is
    unconditional but multiplied by that axis's ``a`` profile, which is exactly
    zero outside the PML), and :func:`zero_edges_and_interiors` zeroes each
    field's interior along its own dimension -- its ``pml_edge_idx`` table --
    on every forward entry and backward seed.
    """
    dims = _dim_names(ndim)
    axes: dict[str, int] = {}
    for i, d in enumerate(dims):
        axes[f"phi_{d}_0"] = i
        axes[f"psi_{d}_0"] = i
    return axes


def acoustic_model_names(ndim: int) -> list[str]:
    """Names of the acoustic models (bulk modulus ``K`` and buoyancy ``B*``)."""
    return ["K"] + [f"B{d}" for d in _dim_names(ndim)]


# The user-facing model arguments, in the order the three functions below
# expect their ``model_requires_grad`` sequence.
USER_MODEL_NAMES = ("v", "rho")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(v_requires_grad, rho_requires_grad)``; the
    result is in :func:`acoustic_model_names` order. :func:`prepare_models`
    builds the bulk modulus ``K = rho * v**2`` (so it requires a gradient if
    either does) and one buoyancy per axis from ``rho`` alone -- i.e. asking
    for a density gradient in ``ndim`` dimensions creates ``1 + ndim``
    gradient-carrying internal models, not one.
    """
    v_requires_grad, rho_requires_grad = model_requires_grad
    return [v_requires_grad or rho_requires_grad] + [rho_requires_grad] * ndim


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    One slot per internal model (``K`` plus one buoyancy per axis), so a run
    with ``rho.requires_grad`` stores ``1 + ndim`` wavefield histories.

    The single source of truth for acoustic's storage-slot count, shared by the
    forward pass (which uses the default
    :func:`deepwave._engine._per_model_storage_flags`) and by
    :func:`deepwave.estimate_memory`, so the two cannot drift.
    """
    return _engine.per_model_slot_flags(
        internal_model_grad_flags(model_requires_grad, ndim)
    )


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models.

    A left inverse of :func:`internal_model_grad_flags` for counting purposes:
    ``K`` requiring a gradient is reported as ``v`` requiring one (they cannot
    be told apart from ``K`` alone), which gives back exactly the same internal
    flags, storage slots and gradient buffers. For callers (the out-of-memory
    message in :mod:`deepwave._engine`) that hold the internal models rather
    than the user's.
    """
    return [models[0].requires_grad, models[1].requires_grad]


def acoustic_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The CFS-CPML memory variables ``phi_{d}_0`` (pressure update) and
    ``psi_{d}_0`` (velocity update), one of each per spatial dimension. They
    are a subset of :func:`acoustic_wavefield_names`, spelled the same way,
    and are what :class:`deepwave.AuxiliaryState` holds.
    """
    dims = _dim_names(ndim)
    return [f"phi_{d}_0" for d in dims] + [f"psi_{d}_0" for d in dims]


def acoustic_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`acoustic`.

    Pressure first, then one per velocity component (z, y, x, for the
    dimensions present).
    """
    return ["receiver_amplitudes_p"] + [
        f"receiver_amplitudes_{d}" for d in _dim_names(ndim)
    ]


def acoustic_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`acoustic`, in return order.

    The public final wavefields -- the pressure and velocities, followed by
    the single ``auxiliary_state`` that replaces the block of PML memory
    variables -- followed by the receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            acoustic_wavefield_names(ndim), acoustic_auxiliary_state_names(ndim)
        ),
        *acoustic_receiver_names(ndim),
    ]


# Declarative description of the acoustic propagator: a staggered-grid solver
# with one pressure source/receiver type plus one per velocity component.
SPEC = _engine.PropagatorSpec(
    name="acoustic",
    status="stable",
    source_channels=_engine.volume_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    deprecated_source_channels=_engine.VOLUME_RATE_SOURCE_DEPRECATIONS,
    staggered=True,
    model_names=acoustic_model_names,
    wavefield_names=acoustic_wavefield_names,
    n_source_types=lambda ndim: ndim + 1,
    n_receiver_types=lambda ndim: ndim + 1,
    output_names=acoustic_output_names,
    auxiliary_state_names=acoustic_auxiliary_state_names,
    pml_slab_axes=acoustic_pml_slab_axes,
    # K and buoyancy own the gradients (all buoyancies share rho's status), so
    # their grad/batched flags are the ones passed to the C backend.
    c_model_flag_indices=[0, 1],
    spatial_parallel=True,
)


def prepare_models(v: torch.Tensor, rho: torch.Tensor) -> list[torch.Tensor]:
    """Prepares models for the staggered grid acoustic propagator.

    Computes the bulk modulus K at integer nodes and buoyancy (1/rho) at
    half-integer nodes.

    Args:
        v: Wavespeed model.
        rho: Density model.

    Returns:
        List containing [K, buoyancy_z, buoyancy_y, buoyancy_x] (depending on dim).
    """
    ndim = v.ndim - 1
    models = []

    # Bulk modulus K = rho * v^2
    # Avoid division by zero
    k = rho * v**2
    models.append(k)

    models.extend(_staggered_grid.stagger_density_arithmetic_mean(rho, ndim))

    return models


class Acoustic(torch.nn.Module):
    """Convenience nn.Module wrapper for acoustic wave propagation.

    Stability: stable (see :doc:`/stability`).

    Stores `v`, `rho` and `grid_spacing` once, so that each :meth:`forward`
    call needs only the arguments that vary between calls.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        rho: The density model, of shape ``[*model_shape]`` or, for a
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
        rho_requires_grad: A bool specifying whether gradients will be
            computed for `rho`. Unset by default, which means False. Cannot
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
        storage_compression: **Lossy -- changes the model gradient by
            about 0.2-4% in 2-D and 0.2-8% in 3-D.** See
            :func:`deepwave.elastic`.
    """

    def __init__(
        self,
        v: torch.Tensor,
        rho: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        v_requires_grad: Optional[bool] = None,
        rho_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        grad = _common.resolve_requires_grad(
            requires_grad,
            ["v", "rho"],
            ["v", "rho"],
            {"v": v_requires_grad, "rho": rho_requires_grad},
            "Acoustic",
        )
        self.v = _common.make_model_parameter(v, grad["v"], "v")
        self.rho = _common.make_model_parameter(rho, grad["rho"], "rho")
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
        source_amplitudes_volume_rate: Optional[torch.Tensor] = None,
        source_amplitudes_p: Optional[torch.Tensor] = None,
        source_amplitudes_z: Optional[torch.Tensor] = None,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_locations_volume_rate: Optional[LocationsLike] = None,
        source_locations_p: Optional[LocationsLike] = None,
        source_locations_z: Optional[LocationsLike] = None,
        source_locations_y: Optional[LocationsLike] = None,
        source_locations_x: Optional[LocationsLike] = None,
        source_mask_volume_rate: Optional[torch.Tensor] = None,
        source_mask_z: Optional[torch.Tensor] = None,
        source_mask_y: Optional[torch.Tensor] = None,
        source_mask_x: Optional[torch.Tensor] = None,
        receiver_locations_p: Optional[LocationsLike] = None,
        receiver_locations_z: Optional[LocationsLike] = None,
        receiver_locations_y: Optional[LocationsLike] = None,
        receiver_locations_x: Optional[LocationsLike] = None,
        receiver_mask_p: Optional[torch.Tensor] = None,
        receiver_mask_z: Optional[torch.Tensor] = None,
        receiver_mask_y: Optional[torch.Tensor] = None,
        receiver_mask_x: Optional[torch.Tensor] = None,
        pressure_0: Optional[torch.Tensor] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
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

        See :func:`acoustic` for details (the constructor-owned `v`/`rho`/
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
        return acoustic(
            self.v,
            self.rho,
            self.grid_spacing,
            dt,
            source_amplitudes_p=source_amplitudes_p,
            source_locations_p=source_locations_p,
            source_mask_volume_rate=source_mask_volume_rate,
            source_mask_z=source_mask_z,
            source_mask_y=source_mask_y,
            source_mask_x=source_mask_x,
            receiver_mask_p=receiver_mask_p,
            receiver_mask_z=receiver_mask_z,
            receiver_mask_y=receiver_mask_y,
            receiver_mask_x=receiver_mask_x,
            sources=sources,
            receivers=receivers,
            source_amplitudes_volume_rate=source_amplitudes_volume_rate,
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_volume_rate=source_locations_volume_rate,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            receiver_locations_p=receiver_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            pressure_0=pressure_0,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
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
            backend=backend,
            storage=_storage_for_call,
            location_interpolation=location_interpolation,
            location_interpolation_halfwidth=location_interpolation_halfwidth,
        )


def acoustic(
    v: torch.Tensor,
    rho: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes_volume_rate: Optional[torch.Tensor] = None,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_volume_rate: Optional[LocationsLike] = None,
    source_locations_p: Optional[LocationsLike] = None,
    source_locations_z: Optional[LocationsLike] = None,
    source_locations_y: Optional[LocationsLike] = None,
    source_locations_x: Optional[LocationsLike] = None,
    source_mask_volume_rate: Optional[torch.Tensor] = None,
    source_mask_z: Optional[torch.Tensor] = None,
    source_mask_y: Optional[torch.Tensor] = None,
    source_mask_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[LocationsLike] = None,
    receiver_locations_z: Optional[LocationsLike] = None,
    receiver_locations_y: Optional[LocationsLike] = None,
    receiver_locations_x: Optional[LocationsLike] = None,
    receiver_mask_p: Optional[torch.Tensor] = None,
    receiver_mask_z: Optional[torch.Tensor] = None,
    receiver_mask_y: Optional[torch.Tensor] = None,
    receiver_mask_x: Optional[torch.Tensor] = None,
    pressure_0: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
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
    backend: str = "auto",
    python_backend: Optional[Union[Literal["eager", "jit", "compile"], bool]] = None,
    storage: StorageLike = STORAGE_DEFAULT,
    storage_mode: Optional[StorageMode] = None,
    storage_path: Optional[StoragePath] = None,
    storage_compression: Optional[bool] = None,
    location_interpolation: Literal["auto", "hicks", "none"] = "auto",
    location_interpolation_halfwidth: int = 4,
) -> "deepwave.PropagatorOutput":
    """Acoustic wave propagation (functional interface).

    Stability: stable (see :doc:`/stability`).

    This function performs forward modelling with the variable-density acoustic
    wave equation on a staggered grid.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        rho: The density model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: Spatial grid cell size.
        dt: Time step interval. Deepwave is unit-agnostic: choose any single
            consistent system of units for ``v``, ``rho``, ``grid_spacing``,
            and ``dt``; see :doc:`/units`.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``volume_rate``, ``z``, ``y``,
            ``x`` (``z`` only in 3-D, ``y`` only in 2-D and 3-D); ``p`` is a
            deprecated spelling of ``volume_rate`` here. Exactly
            equivalent to the flat
            ``source_amplitudes_*``/``source_locations_*`` keywords, which
            remain permanently supported; giving one channel in both forms
            raises. Optional.
        receivers: This simulation's receivers, keyed by channel, as the
            canonical alternative to the flat ``receiver_locations*`` keywords.
            Each value is a locations object, or ``(locations, mask)`` to
            include an activity mask. The channels are ``p``, ``z``, ``y``,
            ``x`` (``z`` only in 3-D, ``y`` only in 2-D and 3-D). Giving one
            channel in both forms raises. Optional.
        source_amplitudes_volume_rate: Source amplitudes ``s``: an injected volume per
            unit volume per unit time, so units of 1/time. This differs from
            the elastic family, whose ``source_amplitudes_pressure_rate``
            is a pressure rate; see :doc:`/units` for the conversion. Each
            sample acts at ``(t+0.5)*dt``, half a step after the pressure
            receiver of the same index is recorded. To
            drive the propagation with a volumetric (extended) source instead
            of hand-placed point sources, build the amplitudes and integer
            locations with :func:`deepwave.wavefields.source_field_to_sources`.
        source_amplitudes_p: Deprecated spelling of
            ``source_amplitudes_volume_rate``, which means exactly the same
            thing. It warns :class:`deepwave.DeepwaveFutureWarning` and will
            be removed in Deepwave 0.4.0: the letter ``p`` names a volume
            rate here and a pressure rate on the elastic family, two
            quantities that differ by roughly the bulk modulus, so a source
            channel says which one it injects; see :doc:`/units`. ``p``
            remains the receiver channel's name. Passing both raises.
            Optional.
        source_locations_volume_rate: Source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_p: Deprecated spelling of
            ``source_locations_volume_rate``, as
            ``source_amplitudes_p`` is of ``source_amplitudes_volume_rate``.
            Passing both raises. Optional.
        source_amplitudes_z: Source amplitudes ``f_z``: a body-force density in
            the z direction, with units of force/volume in both 2-D and 3-D.
            Each sample acts at ``t*dt``. This channel exists only in 3-D.
        source_locations_z: Source locations (z-force). This channel exists
            only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_y: Source amplitudes (y-force density), as
            ``source_amplitudes_z``.
        source_locations_y: Source locations (y-force).
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_x: Source amplitudes (x-force density), as
            ``source_amplitudes_z``.
        source_locations_x: Source locations (x-force).
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_mask_volume_rate: A boolean torch.Tensor with dimensions
            [shot, source] selecting which sources of the ``volume_rate``
            channel are active. ``True`` means active; ``False`` means that
            point injects nothing, and its coordinates in
            ``source_locations_volume_rate`` are then ignored entirely (they need
            not be valid, in range, or unique). The shape must match the
            leading dimensions of ``source_locations_volume_rate``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        source_mask_z: A boolean torch.Tensor with dimensions [shot, source]
            selecting which sources of the ``z`` channel (3-D only) are active.
            ``True`` means active; ``False`` means that point injects nothing,
            and its coordinates in ``source_locations_z`` are then ignored
            entirely (they need not be valid, in range, or unique). The shape
            must match the leading dimensions of ``source_locations_z``, which
            must be given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        source_mask_y: A boolean torch.Tensor with dimensions [shot, source]
            selecting which sources of the ``y`` channel are active. ``True``
            means active; ``False`` means that point injects nothing, and its
            coordinates in ``source_locations_y`` are then ignored entirely
            (they need not be valid, in range, or unique). The shape must match
            the leading dimensions of ``source_locations_y``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        source_mask_x: A boolean torch.Tensor with dimensions [shot, source]
            selecting which sources of the ``x`` channel are active. ``True``
            means active; ``False`` means that point injects nothing, and its
            coordinates in ``source_locations_x`` are then ignored entirely
            (they need not be valid, in range, or unique). The shape must match
            the leading dimensions of ``source_locations_x``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_locations_p: Pressure receiver locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_z: Z-component velocity receiver locations. This
            channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_y: Y-component velocity receiver locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_x: X-component velocity receiver locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_mask_p: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``p`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``receiver_locations_p`` are then
            ignored entirely (they need not be valid, in range, or unique). The
            shape must match the leading dimensions of
            ``receiver_locations_p``, which must be given. This is the modern
            spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel and
            produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_mask_z: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``z`` channel (3-D only)
            are active. ``True`` means active; ``False`` means that point
            records zeros, and its coordinates in ``receiver_locations_z`` are
            then ignored entirely (they need not be valid, in range, or
            unique). The shape must match the leading dimensions of
            ``receiver_locations_z``, which must be given. This is the modern
            spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel and
            produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_mask_y: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``y`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``receiver_locations_y`` are then
            ignored entirely (they need not be valid, in range, or unique). The
            shape must match the leading dimensions of
            ``receiver_locations_y``, which must be given. This is the modern
            spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel and
            produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_mask_x: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``x`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``receiver_locations_x`` are then
            ignored entirely (they need not be valid, in range, or unique). The
            shape must match the leading dimensions of
            ``receiver_locations_x``, which must be given. This is the modern
            spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel and
            produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        accuracy: See :func:`deepwave.elastic`.
        pml_width: See :func:`deepwave.elastic`. Unlike the scalar propagator, whose
            zero-width edge is a free surface, here (as for any staggered-grid
            propagator) it is a rigid surface for pressure (reflection coefficient
            approximately +1).
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: An optional single string, applying to every face, or a
            list of `2 * ndim` strings selecting a first-class
            boundary condition per face, in the same order as `pml_width`. Each
            entry is one of `"pml"` (absorbing, using `pml_width`;
            `boundary="pml"` is the spelled-out default, and behaves in every
            respect -- including the warnings it emits -- as omitting
            `boundary` does), `"free"` (a pressure-release
            surface: pressure is held to zero on the surface and reflects with
            inverted polarity, reflection coefficient -1), `"rigid"` (a rigid
            wall: the boundary-normal velocity is held to zero and pressure
            reflects with preserved polarity, +1), or `"periodic"` (the face
            wraps around to the opposite face). A `"free"`, `"rigid"`, or
            `"periodic"` face requires `pml_width` to be zero on that face, and
            `"periodic"` must be set on both faces of a dimension. Unlike a bare
            zero-`pml_width` face (an approximate reflector at accuracy above 2),
            a `"free"`/`"rigid"` face refills the guard ring with the exact
            staggered image every step, so the reflection coefficient is met to
            better than 1% at all accuracy orders. Supported in both the
            compiled and the pure-PyTorch (`backend="eager"`) backends, which
            agree to machine precision in the forward fields and the
            gradients. See the boundary-conditions documentation for the
            ghost-plane conventions.
            Availability: `"pml"` and `"rigid"` are accepted on every face at
            every accuracy, and `"free"` on the beginning of a dimension at
            every accuracy and on the end of a dimension at `accuracy=2`.
            `"free"` on the end of a dimension with `accuracy >= 4`, and
            `"periodic"` on any face, raise
            :class:`deepwave.DeepwaveNotImplementedError`: those guard-ring
            fills do not conserve a discrete energy, so the solution can grow
            without bound. Put such a face at the beginning of the dimension
            instead (flip the model along that axis), or use `accuracy=2`.
            Faces may be accepted in a later release; none is removed.
        pml_freq: See :func:`deepwave.elastic`.
        max_vel: See :func:`deepwave.elastic`.
        courant_number: See :func:`deepwave.elastic`, which documents the staggered-grid
            critical Courant number used here.
        survey_pad: See :func:`deepwave.scalar`.
        pressure_0: Initial pressure (time t=0). Like every initial
            wavefield, it may be provided either at the shape of the
            simulation domain (the extracted model region plus the PML),
            which is the shape a previous call returns, or at the shape of
            the extracted model region itself, which Deepwave zero-pads for
            you; see the `wavefield_0` entry of :func:`deepwave.scalar`.
        vz_0: Initial z-velocity (time t=-0.5).
        vy_0: Initial y-velocity (time t=-0.5).
        vx_0: Initial x-velocity (time t=-0.5).
        initial_state: See :func:`deepwave.elastic`. The mapping keys are this
            propagator's own public state field names (`pressure_0`, `vz_0`,
            ...) and `auxiliary_state`. The individual absorbing-boundary field
            names (`phi_z_0`, ..., `psi_x_0`) are still accepted as keys, with
            a `FutureWarning`; they will stop being accepted in Deepwave
            0.4.0.
        origin: See :func:`deepwave.scalar`.
        nt: See :func:`deepwave.elastic`.
        model_gradient_sampling_interval: See :func:`deepwave.elastic`.
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
        freq_taper_frac: See :func:`deepwave.elastic`.
        time_pad_frac: See :func:`deepwave.elastic`.
        time_taper: See :func:`deepwave.elastic`.
        check_output_finite: See :func:`deepwave.elastic`.
        forward_callback: See :func:`deepwave.elastic`.
        backward_callback: See :func:`deepwave.elastic`.
        callback_frequency: See :func:`deepwave.elastic`.
        backend: Which execution backend to use: ``"auto"`` (the default),
            ``"compiled"``, ``"eager"`` or ``"torch_compile"``. See
            :doc:`/backends` for what each value selects, what ``"auto"``
            resolves to, and how the backends agree numerically.
            ``"auto"`` resolves to the compiled C/CUDA kernel.
        python_backend: See :func:`deepwave.scalar`.
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
        storage_compression: **Lossy -- changes the model gradient by
            about 0.2-4% in 2-D and 0.2-8% in 3-D.** See
            :func:`deepwave.elastic`.
        location_interpolation: See :func:`deepwave.elastic`. This propagator has a
            single pressure node and one velocity component per axis (no stress tensor),
            but the same staggered-grid `N + 0.5` convention applies.
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
        location_interpolation_halfwidth: See :func:`deepwave.elastic`.

    Returns:
        deepwave.PropagatorOutput: a tuple subclass whose fields, in
        order, are:

            - pressure_0: Final pressure wavefield.
            - vz_0: Final z-velocity wavefield (if ndim == 3).
            - vy_0: Final y-velocity wavefield (if ndim >= 2).
            - vx_0: Final x-velocity wavefield.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables. Opaque: pass it back
              (as part of `initial_state`) to continue the simulation; its
              contents are implementation state rather than physics.
            - receiver_amplitudes_p: Recorded pressure data.
            - receiver_amplitudes_z: Recorded z-velocity data (if ndim == 3).
            - receiver_amplitudes_y: Recorded y-velocity data (if ndim >= 2).
            - receiver_amplitudes_x: Recorded x-velocity data.

        This positional order is easy to index incorrectly (``out[-1]`` is the
        x-component velocity receiver amplitudes, not the pressure data);
        prefer accessing the fields by name instead.
    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            acoustic,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
                source_amplitudes_p="source_amplitudes_volume_rate",
                source_locations_p="source_locations_volume_rate",
            ),
        )
    backward_callback_needs_forward_wavefield = False
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
            "source_amplitudes_volume_rate": source_amplitudes_volume_rate,
            "source_locations_volume_rate": source_locations_volume_rate,
            "source_mask_volume_rate": source_mask_volume_rate,
            "source_amplitudes_z": source_amplitudes_z,
            "source_locations_z": source_locations_z,
            "source_mask_z": source_mask_z,
            "source_amplitudes_y": source_amplitudes_y,
            "source_locations_y": source_locations_y,
            "source_mask_y": source_mask_y,
            "source_amplitudes_x": source_amplitudes_x,
            "source_locations_x": source_locations_x,
            "source_mask_x": source_mask_x,
            "receiver_locations_p": receiver_locations_p,
            "receiver_mask_p": receiver_mask_p,
            "receiver_locations_z": receiver_locations_z,
            "receiver_mask_z": receiver_mask_z,
            "receiver_locations_y": receiver_locations_y,
            "receiver_mask_y": receiver_mask_y,
            "receiver_locations_x": receiver_locations_x,
            "receiver_mask_x": receiver_mask_x,
        },
        source_amplitudes_p=(
            "source_amplitudes_volume_rate",
            source_amplitudes_p,
        ),
        source_locations_p=(
            "source_locations_volume_rate",
            source_locations_p,
        ),
    )
    _engine.expand_channel_mappings(
        SPEC,
        _channels,
        sources=sources,
        receivers=receivers,
    )
    _common.apply_location_masks(SPEC, _channels)
    source_amplitudes_volume_rate = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_volume_rate"]
    )
    source_locations_volume_rate = cast(
        "Optional[LocationsLike]", _channels["source_locations_volume_rate"]
    )
    source_amplitudes_z = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_z"]
    )
    source_locations_z = cast(
        "Optional[LocationsLike]", _channels["source_locations_z"]
    )
    source_amplitudes_y = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_y"]
    )
    source_locations_y = cast(
        "Optional[LocationsLike]", _channels["source_locations_y"]
    )
    source_amplitudes_x = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_x"]
    )
    source_locations_x = cast(
        "Optional[LocationsLike]", _channels["source_locations_x"]
    )
    receiver_locations_p = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_p"]
    )
    receiver_locations_z = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_z"]
    )
    receiver_locations_y = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_y"]
    )
    receiver_locations_x = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_x"]
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
    _common.check_models([v, rho], ["v", "rho"], "acoustic")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, python_backend, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend, python_backend
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
            "pressure_0": pressure_0,
            "vz_0": vz_0,
            "vy_0": vy_0,
            "vx_0": vx_0,
            # phi_*/psi_* are the absorbing boundary's memory variables, not
            # physical fields, so they are not exposed as arguments; they
            # travel in `auxiliary_state`.
            "phi_z_0": None,
            "phi_y_0": None,
            "phi_x_0": None,
            "psi_z_0": None,
            "psi_y_0": None,
            "psi_x_0": None,
        },
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=acoustic_auxiliary_state_names(3),
        propagator=SPEC.name,
    )
    (
        pressure_0,
        vz_0,
        vy_0,
        vx_0,
        phi_z_0,
        phi_y_0,
        phi_x_0,
        psi_z_0,
        psi_y_0,
        psi_x_0,
    ) = _resolved_wavefields.values()

    _common.check_inputs_not_vmapped(
        pressure_0,
        source_amplitudes_volume_rate,
        source_amplitudes_z,
        source_amplitudes_y,
        source_amplitudes_x,
        source_locations_volume_rate,
        source_locations_z,
        source_locations_y,
        source_locations_x,
        receiver_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
        vz_0,
        phi_z_0,
        psi_z_0,
        vy_0,
        phi_y_0,
        psi_y_0,
        vx_0,
        phi_x_0,
        psi_x_0,
    )
    ndim = _common.get_ndim(
        [v, rho],
        [pressure_0],
        [
            source_locations_volume_rate,
            source_locations_z,
            source_locations_y,
            source_locations_x,
            receiver_locations_p,
            receiver_locations_z,
            receiver_locations_y,
            receiver_locations_x,
        ],
        [vz_0, phi_z_0, psi_z_0],
        [vy_0, phi_y_0, psi_y_0],
        [vx_0, phi_x_0, psi_x_0],
    )
    if ndim < 3 and (
        source_locations_z is not None or receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            "acoustic: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "z dimension should not be provided."
        )
    if ndim < 2 and (
        source_locations_y is not None or receiver_locations_y is not None
    ):
        raise DeepwaveValueError(
            "acoustic: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "y dimension should not be provided."
        )

    # Prepare initial wavefields list
    initial_wavefields: list[Optional[torch.Tensor]] = []
    initial_wavefields.append(pressure_0)
    if ndim == 3:
        initial_wavefields.append(vz_0)
    if ndim >= 2:
        initial_wavefields.append(vy_0)
    initial_wavefields.append(vx_0)

    # Phi (PML for pressure)
    if ndim == 3:
        initial_wavefields.append(phi_z_0)
    if ndim >= 2:
        initial_wavefields.append(phi_y_0)
    initial_wavefields.append(phi_x_0)

    # Psi (PML for velocity)
    if ndim == 3:
        initial_wavefields.append(psi_z_0)
    if ndim >= 2:
        initial_wavefields.append(psi_y_0)
    initial_wavefields.append(psi_x_0)

    # Whether the absorbing profiles are a constant of this call: they are
    # built from the maximum wave speed, so they depend on the model only
    # when it was not supplied. Recorded before ``setup_propagator`` resolves
    # ``max_vel`` to a number.
    max_vel_pinned = max_vel is not None
    min_nonzero_model_vel, max_model_vel = _common.velocity_extremes(
        v, max_vel, "acoustic"
    )
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

    source_amplitudes_list = []
    source_locations_list_in = []
    if ndim == 3:
        source_amplitudes_list.extend(
            [source_amplitudes_z, source_amplitudes_y, source_amplitudes_x]
        )
        source_locations_list_in.extend(
            [source_locations_z, source_locations_y, source_locations_x]
        )
    elif ndim == 2:
        source_amplitudes_list.extend([source_amplitudes_y, source_amplitudes_x])
        source_locations_list_in.extend([source_locations_y, source_locations_x])
    elif ndim == 1:
        source_amplitudes_list.append(source_amplitudes_x)
        source_locations_list_in.append(source_locations_x)

    receiver_locations_list_in = []
    if ndim == 3:
        receiver_locations_list_in.extend(
            [receiver_locations_z, receiver_locations_y, receiver_locations_x]
        )
    elif ndim == 2:
        receiver_locations_list_in.extend([receiver_locations_y, receiver_locations_x])
    elif ndim == 1:
        receiver_locations_list_in.append(receiver_locations_x)
    source_amplitudes_list.insert(0, source_amplitudes_volume_rate)
    source_locations_list_in.insert(0, source_locations_volume_rate)
    receiver_locations_list_in.insert(0, receiver_locations_p)

    # Resolve off-grid / directional (Hicks) source and receiver specifications
    # per component. Pressure sits on the regular grid (offset None); each
    # velocity component is staggered half a cell in its own propagation
    # dimension and so carries that component's half-cell offset, keeping the
    # invariant that integer index N and float N+offset are the same physical
    # point. Integer/long locations take the exact path and are bit-for-bit
    # identical to before. The lists are ordered [pressure, velocities...].
    component_offsets: list[Optional[list[float]]] = [
        None,
        *_common.velocity_component_offsets(ndim),
    ]
    # Recorded before the location arguments are replaced by the integer grid
    # indices the backend consumes.
    location_interpolation_used = _engine.resolved_location_interpolation(
        location_interpolation,
        [source_locations_list_in, receiver_locations_list_in],
        [component_offsets, component_offsets],
    )
    # Which faces of the model an interpolation window may not cross. None of
    # this propagator's source/receiver components sits on the regular grid,
    # so the image of a reflecting face is component dependent and is not
    # implemented here: such a face is refused rather than approximated (see
    # ``_common.hicks_grid``).
    pml_inference_amplitudes = _common.active_source_amplitudes(
        source_amplitudes_list, source_locations_list_in
    )
    _hicks_grid = _common.hicks_grid(
        boundary,
        [*source_locations_list_in, *receiver_locations_list_in],
        v,
        "acoustic",
    )
    source_amplitudes_list, source_locations_list = (
        _common.resolve_source_interpolators(
            source_locations_list_in,
            source_amplitudes_list,
            component_offsets,
            location_interpolation,
            location_interpolation_halfwidth,
            v.dtype,
            _hicks_grid,
        )
    )
    receiver_interpolators, receiver_locations_list = (
        _common.resolve_receiver_interpolators(
            receiver_locations_list_in,
            component_offsets,
            location_interpolation,
            location_interpolation_halfwidth,
            v.dtype,
            _hicks_grid,
        )
    )

    # A velocity source/receiver must not sit on the final sample of its
    # staggered dimension (that cell is not part of the computational domain).
    # The check runs on the resolved integer grid locations; pressure is at
    # index 0, so the velocity component for dimension ``dim`` is at index
    # ``dim + 1``.
    dim_names = ["z", "y", "x"]
    for dim in range(ndim):
        for locations, name in [
            (source_locations_list, "source"),
            (receiver_locations_list, "receiver"),
        ]:
            location = locations[dim + 1]
            if location is None:
                continue
            dim_location = location[..., -ndim + dim]
            dim_location = dim_location[dim_location != deepwave.IGNORE_LOCATION]
            if (
                dim_location.numel() > 0
                and dim_location.max() >= v.shape[-ndim + dim] - 1
            ):
                raise DeepwaveRuntimeError(
                    f"acoustic: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{v.shape[-ndim + dim] - 1}.",
                )

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
        [v, rho],
        ["replicate", "replicate"],
        ["v", "rho"],
        grid_spacing,
        dt,
        source_amplitudes_list,
        source_locations_list,
        receiver_locations_list,
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
        staggered=True,
        storage=_storage_policy,
        propagator_name=SPEC.name,
        pml_inference_amplitudes=pml_inference_amplitudes,
    )
    del (
        v,
        rho,
        source_amplitudes_volume_rate,
        source_locations_volume_rate,
        source_amplitudes_z,
        source_locations_z,
        source_amplitudes_y,
        source_locations_y,
        source_amplitudes_x,
        source_locations_x,
        receiver_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
    )

    models = prepare_models(models[0], models[1])

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "acoustic")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "acoustic")
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

    # Scale source amplitudes
    # Pressure update: p(t+1) = p(t) - dt * K * (div(v) - s) -> Add dt * K * s
    # Velocity update: v(t+1/2) = v(t-1/2) - dt * B * (grad(p) - f) -> Add dt * B * f
    # models order: K, B_z (3D), B_y (2D/3D), B_x (1D/2D/3D) matches target indices
    model_shape = models[0].shape[-ndim:]

    for i, (src_amp, src_loc, model) in enumerate(
        zip(source_amplitudes_out, sources_i, models)
    ):
        if src_amp.numel() > 0:
            source_amplitudes_out[i] = (
                src_amp
                * _common.gather_model_at_sources(model, src_loc, model_shape, n_shots)
                * dt
            )

    pml_profiles = _staggered_grid.set_pml_profiles(
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
        acoustic_func(
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
            *models,
            *source_amplitudes_out,
            *sources_i,
            *receivers_i,
            *wavefields,
            max_vel_pinned=max_vel_pinned,
        )
    )

    for i in range(len(outputs) - 1 - ndim, len(outputs)):
        outputs[i] = _common.downsample_and_movedim(
            outputs[i],
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
        )

    # Gather off-grid receiver data back to the original locations. The receiver
    # outputs are the trailing (1 + ndim) entries, ordered [pressure,
    # velocities...] to match receiver_interpolators.
    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 1 - ndim
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


def zero_edges_and_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    interior: bool = True,
) -> None:
    """Zeros the edges and/or interiors of the acoustic wavefields.

    Wavefields order: ``[pressure, vz, vy, vx, phi..., psi...]``. The pressure
    (scalar) and its PML memory variables ``phi`` sit on the integer grid; the
    velocities and their PML memory variables ``psi`` sit on the half grid
    (``psi_z`` matches ``vz`` etc.). This builds the acoustic-specific
    ``half_grid`` / ``pml_edge_idx`` tables and delegates the shared zeroing
    pass to :func:`deepwave._staggered_grid.zero_edges_and_interiors`.
    """
    num_vars = len(wavefields)
    # Expected num_vars:
    # 3D: p, vz, vy, vx, phiz, phiy, phix, psiz, psiy, psix (10 vars)
    # 2D: p, vy, vx, phiy, phix, psiy, psix (7 vars)
    # 1D: p, vx, phix, psix (4 vars)
    half_grid = [[False] * ndim for _ in range(num_vars)]
    pml_edge_idx: list[list[int]] = [[] * ndim]

    if ndim == 3:
        # vz (1), psiz (7) -> shifted z (dim 0)
        half_grid[1][0] = True
        half_grid[7][0] = True
        # vy (2), psiy (8) -> shifted y (dim 1)
        half_grid[2][1] = True
        half_grid[8][1] = True
        # vx (3), psix (9) -> shifted x (dim 2)
        half_grid[3][2] = True
        half_grid[9][2] = True
        pml_edge_idx = [[4, 7], [5, 8], [6, 9]]
    elif ndim == 2:
        # vy (1), psiy (5) -> shifted y (dim 0)
        half_grid[1][0] = True
        half_grid[5][0] = True
        # vx (2), psix (6) -> shifted x (dim 1)
        half_grid[2][1] = True
        half_grid[6][1] = True
        pml_edge_idx = [[3, 5], [4, 6]]
    elif ndim == 1:
        # vx (1), psix (3) -> shifted x (dim 0)
        half_grid[1][0] = True
        half_grid[3][0] = True
        pml_edge_idx = [[2, 3]]

    _staggered_grid.zero_edges_and_interiors(
        wavefields,
        ndim,
        fd_pad,
        fd_pad_list,
        pml_width,
        half_grid,
        pml_edge_idx,
        interior=interior,
    )


def _base_level_dispatch() -> Any:
    """Use the shared PyTorch-version-aware base-dispatch context."""
    return _common._base_level_dispatch()  # noqa: SLF001 - shared internal helper


def _model_tangent_in_pml(
    dmodel: torch.Tensor, pml_width: Sequence[int], accuracy: int, ndim: int
) -> bool:
    """Whether a model tangent is non-zero inside the PML band (or FD halo).

    ``dmodel`` (e.g. the ``K`` tangent, the image of a ``v``/``rho`` tangent
    through ``prepare_models``) is on the padded grid: ``fd_pad = accuracy // 2``
    of finite-difference halo, then ``pml_width`` of absorbing band, then the
    physical interior. Anything remaining after zeroing the interior overlaps
    the PML/FD halo (mirrors ``scalar._v_tangent_in_pml``).
    """
    fd_pad = accuracy // 2
    shape = dmodel.shape[-ndim:]
    interior: list[Any] = [slice(None)] * (dmodel.ndim - ndim)
    for i in range(ndim):
        lo = fd_pad + int(pml_width[2 * i])
        hi = shape[i] - fd_pad - int(pml_width[2 * i + 1])
        interior.append(slice(lo, max(lo, hi)))
    edge = dmodel.clone()
    edge[tuple(interior)] = 0
    return bool(edge.ne(0).any())


def _acoustic_forward_jvp(
    ctx: Any, tangents: tuple[Any, ...]
) -> tuple[Optional[torch.Tensor], ...]:
    """Forward-mode derivative (JVP) of the acoustic propagator via Born.

    The scattered field of :func:`deepwave.acoustic_born` **is**
    the directional derivative of :func:`acoustic` (see
    ``docs/acoustic_born_derivation.md`` section 6): differentiating the discrete
    acoustic recurrence in the direction of a model perturbation reproduces the
    Born scattered update, and differentiating the source pre-scaling
    (``dt*K*s`` / ``dt*B_d*f``) supplies the scattered point source
    automatically. This routes the tangents through the fast compiled Born
    backend so ``torch.func.jvp`` (and ``torch.autograd.forward_ad``) run on the
    C/CUDA engine rather than needing ``backend="eager"``.

    Input-tangent mapping (all pre-scalings are already baked into this
    Function's inputs by :func:`acoustic`'s differentiable preprocessing, whose
    forward-mode derivative torch computes for us):

    * model (``K``, ``B_d``) tangents -> Born ``scatter`` (``dK``, ``dB_d``).
    * source-amplitude tangents -> Born's scattered sources. The inputs are
      already ``dt*K*s`` / ``dt*B_d*f``, so their tangents are
      ``dt*(dK*s + K*ds)`` / ``dt*(dB_d*f + B_d*df)`` -- exactly the pre-scaled
      scattered point sources Born needs (combining model and source
      contributions by linearity), injected directly with no double counting.
    * initial-wavefield tangents -> Born's scattered initial wavefields.

    **PML caveat (identical to scalar).** Born does not linearise the PML
    profile's dependence on ``v``, so the jvp is exact only where the model
    tangent is zero in the PML region (or ``pml_width=0``); a warning is emitted
    otherwise.
    """
    acoustic_born_module = importlib.import_module("deepwave.acoustic_born")

    try:
        import torch._C._functorch as _functorch_c

        depth = _functorch_c.get_dynamic_layer_stack_depth()
    except (ImportError, AttributeError):  # pragma: no cover - older torch
        depth = 0
    if depth > 1:
        raise DeepwaveNotImplementedError(
            "acoustic: the fast-backend forward-mode (jvp) rule "
            "supports a single torch.func.jvp (or torch.autograd.forward_ad) "
            "level, not nested/composed torch.func transforms. Re-run with "
            'backend="eager" to compose transforms through native '
            "autograd (slower)."
        )

    # The Born backend this rule routes through threads the first-class
    # boundary ``bc`` codes, so non-PML faces are supported
    # here too -- the tangent is exact wherever the primal reflecting-face
    # rule is (the guard-ring image is a linear operator, applied
    # identically to the primal and its tangent).
    jvp_boundary = list(ctx.jvp_boundary)

    unwrap = _engine.unwrap_functorch
    ndim = len(ctx.jvp_grid_spacing)
    n = ndim + 1
    n_bg_wf = 1 + 3 * ndim

    saved = [cast("torch.Tensor", unwrap(t)) for t in ctx.saved_tensors]
    models = saved[:n]
    source_amplitudes = saved[n : 2 * n]
    sources_i = saved[2 * n : 3 * n]
    receivers_i = saved[3 * n : 4 * n]
    wavefields = saved[4 * n : 4 * n + n_bg_wf]
    pml_profiles = saved[4 * n + n_bg_wf :]

    # Tangents correspond to the Function's positional inputs: leading non-tensor
    # args (including the trailing ``boundary``, ``_N_LEAD``), then models,
    # source amplitudes, sources_i, receivers_i, wavefields.
    lead = _N_LEAD
    model_tangents = [unwrap(t) for t in tangents[lead : lead + n]]
    source_tangents = [unwrap(t) for t in tangents[lead + n : lead + 2 * n]]
    wavefield_tangents = [
        unwrap(t) for t in tangents[lead + 4 * n : lead + 4 * n + n_bg_wf]
    ]

    with _base_level_dispatch(), torch.no_grad():
        model_tangents_f: list[torch.Tensor] = [
            dm if dm is not None else torch.zeros_like(m)
            for dm, m in zip(model_tangents, models)
        ]
        source_tangents_f: list[torch.Tensor] = [
            ds if ds is not None else torch.zeros_like(s)
            for ds, s in zip(source_tangents, source_amplitudes)
        ]
        wavefield_tangents_f: list[torch.Tensor] = [
            dw if dw is not None else torch.zeros_like(w)
            for dw, w in zip(wavefield_tangents, wavefields)
        ]
        # Only when the profiles really do depend on the model: their
        # amplitude is built from the maximum wave speed, so a pinned
        # ``max_vel`` makes them constants of the call.
        if not ctx.jvp_max_vel_pinned and _model_tangent_in_pml(
            model_tangents_f[0], ctx.jvp_pml_width, ctx.jvp_accuracy, ndim
        ):
            warn_at_user_call(
                "torch.func.jvp of acoustic: a model tangent is non-zero "
                "inside the PML, where the PML profile depends on the model "
                "through the maximum wave speed inferred from it and the "
                "Born-based forward-mode derivative does not linearise that "
                "dependence, so the returned tangent is inexact there; pass "
                "max_vel= to pin the profiles, or set pml_width=0, if you need "
                "an exact tangent everywhere (see the acoustic_born PML "
                "caveat).",
                UserWarning,
                2,
            )

        # Background receivers are not needed (the primal comes from forward);
        # the scattered receivers share the original receiver locations so their
        # output is the tangent at the same points.
        empty_rcv = [
            torch.empty(0, dtype=r.dtype, device=r.device) for r in receivers_i
        ]
        born_args = (
            *models,
            *model_tangents_f,
            *source_amplitudes,
            *source_tangents_f,
            *sources_i,
            *sources_i,
            *empty_rcv,
            *receivers_i,
            *wavefields,
            *wavefield_tangents_f,
        )
        born_out = acoustic_born_module.AcousticBornForwardFunc.forward(
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
            jvp_boundary,  # boundary -- same per-face codes as the primal
            *born_args,
        )

    # Born outputs: (*bg_wavefields, *sc_wavefields, *bg_receivers,
    # *sc_receivers, saved_state). The acoustic forward outputs are
    # (*wavefields, *receiver_amplitudes, saved_state); their tangents are the
    # Born scattered wavefields, the scattered receivers, and None.
    sc_wavefields = born_out[n_bg_wf : 2 * n_bg_wf]
    sc_receivers = born_out[2 * n_bg_wf + n : 2 * n_bg_wf + 2 * n]
    return (*sc_wavefields, *sc_receivers, None)


class AcousticForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of acoustic wave propagation."""

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
        max_vel_pinned: bool,
        boundary: list[int],
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the acoustic wave equation.

        Returns the padding-removed output wavefields and receiver amplitudes,
        with a trailing :class:`deepwave._engine.ForwardSavedState` that
        ``setup_context`` uses to populate ``ctx``.
        """
        # The default storage_flags (one slot per model, gated by that model's
        # requires_grad) is exactly what acoustic needs (K and each buoyancy).
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
            zero_wavefields=zero_edges_and_interiors,
            save_wavefields=True,
            boundary=boundary,
        )
        return (*outputs, saved_state)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Populates ``ctx`` from the trailing state produced by ``forward``."""
        state = output[-1]
        _engine.apply_forward_saved_state(ctx, state)
        # Forward-mode AD (``torch.func.jvp`` / ``torch.autograd.forward_ad``)
        # reaches ``jvp`` with dual inputs that need not ``requires_grad``, so
        # ``save_for_backward`` (gated on ``needs_backward``) may not have run.
        # Save the primal tensors the Born-based ``jvp`` rule needs directly
        # from ``inputs`` (models, source amplitudes, source/receiver indices,
        # initial wavefields, PML profiles), plus its metadata, unconditionally
        # (mirrors ScalarForwardFunc.setup_context; the layout here is explicit
        # so it does not depend on ``save_wavefields``).
        ndim = len(inputs[1])
        n = ndim + 1
        n_bg_wf = 1 + 3 * ndim
        # ``boundary`` is the last leading (non-tensor) input, at index
        # ``_N_LEAD - 1``; the tensor inputs start at ``_N_LEAD``.
        lead = _N_LEAD
        models = list(inputs[lead : lead + n])
        source_amplitudes = list(inputs[lead + n : lead + 2 * n])
        sources_i = list(inputs[lead + 2 * n : lead + 3 * n])
        receivers_i = list(inputs[lead + 3 * n : lead + 4 * n])
        wavefields = list(inputs[lead + 4 * n : lead + 4 * n + n_bg_wf])
        pml_profiles = list(inputs[0])
        ctx.save_for_forward(
            *models,
            *source_amplitudes,
            *sources_i,
            *receivers_i,
            *wavefields,
            *pml_profiles,
        )
        boundary = list(inputs[_N_LEAD - 1])
        ctx.boundary = boundary
        ctx.jvp_boundary = boundary
        # Whether the caller pinned ``max_vel``: the input immediately before
        # ``boundary``. The absorbing profiles are built from the maximum wave
        # speed, so they depend on the model only when it was inferred from it,
        # which is what the forward-mode rule's caveat is about.
        ctx.jvp_max_vel_pinned = bool(inputs[_N_LEAD - 2])
        ctx.jvp_grid_spacing = state.grid_spacing
        ctx.jvp_dt = state.dt
        ctx.jvp_nt = state.nt
        ctx.jvp_step_ratio = state.step_ratio
        ctx.jvp_accuracy = state.accuracy
        ctx.jvp_pml_width = state.pml_width
        ctx.jvp_n_shots = state.n_shots

    # torch.func transforms now reach this Function; the ctypes backend cannot
    # process batched/dual tensors, so vmap raises an actionable error pointing
    # to backend="eager". Forward-mode AD (jvp) is supported on the fast
    # backend via the Born relationship (Born scattered output = JVP of
    # acoustic); see :func:`_acoustic_forward_jvp`.
    vmap = staticmethod(_engine.unsupported_vmap)

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> tuple[Optional[torch.Tensor], ...]:
        """Forward-mode derivative of acoustic via the Born relationship.

        See :func:`_acoustic_forward_jvp` for how the input tangents map onto
        the acoustic Born backend inputs.
        """
        return _acoustic_forward_jvp(ctx, tangents)

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Computes gradients during the backward pass.

        Builds a nested differentiable :class:`AcousticBackwardFunc` (mirroring
        ``scalar.ScalarForwardFunc``'s use of
        :class:`ScalarBackwardFunc`), so a ``create_graph=True`` backward yields
        the true acoustic Hessian-vector product (via the acoustic Born backend)
        rather than raising. The nested Function's ``forward`` runs the ordinary
        first-order adjoint; its ``backward`` is the double-backward (one Born
        forward + one Born backward per ``H*h``, see
        ``docs/acoustic_born_derivation.md`` section 5).
        """
        ndim = _engine.saved_state_ndim(ctx, "acoustic")
        n = ndim + 1
        n_bg_wf = 1 + 3 * ndim
        # Drop the grad slot for the trailing ``ForwardSavedState`` output.
        args = args[:-1]
        grad_receivers = list(args[-n:])
        grad_wavefields = list(args[:-n])
        del args

        # saved_tensors layout (save_wavefields=True): models (n), sources_i
        # (n), receivers_i (n), source_amplitudes (n), initial wavefields
        # (n_bg_wf), pml_profiles (rest).
        saved = list(ctx.saved_tensors)
        models = saved[:n]
        sources_i = saved[n : 2 * n]
        receivers_i = saved[2 * n : 3 * n]
        source_amplitudes = saved[3 * n : 4 * n]
        wavefields = saved[4 * n : 4 * n + n_bg_wf]
        pml_profiles = saved[4 * n + n_bg_wf :]

        result = AcousticBackwardFunc.apply(  # type: ignore[no-untyped-call]
            pml_profiles,
            sources_i,
            receivers_i,
            ctx.storage_manager,
            list(ctx.grid_spacing),
            ctx.dt,
            ctx.nt,
            ctx.n_shots,
            ctx.step_ratio,
            ctx.accuracy,
            ctx.pml_width,
            list(ctx.source_amplitudes_requires_grad),
            ctx.backward_callback,
            ctx.callback_frequency,
            ctx.boundary,
            *grad_receivers,
            *models,
            *source_amplitudes,
            *grad_wavefields,
            *wavefields,
        )
        grad_models = result[:n]
        grad_sources = result[n : 2 * n]
        grad_wf = result[2 * n : 2 * n + n_bg_wf]
        return (
            *([None] * _N_LEAD),
            *grad_models,  # models (K, B*)
            *grad_sources,  # source_amplitudes
            *([None] * n),  # sources_i (integer indices)
            *([None] * n),  # receivers_i (integer indices)
            *grad_wf,  # initial wavefields
        )


# Number of leading non-tensor arguments of ``AcousticForwardFunc.forward``
# (the shared config block plus acoustic's own trailing ``boundary`` flag),
# all of which receive ``None`` gradients in ``backward``.
_N_LEAD = _engine.leading_arg_count(AcousticForwardFunc.forward)


def _acoustic_grad_model_batched(models: list[torch.Tensor], ndim: int) -> list[bool]:
    """Per-model batched flags: K, then the buoyancies (share density's status)."""
    batched = [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models[:2]]
    return [batched[0]] + [batched[1]] * (len(models) - 1)


def _acoustic_make_aux_wavefields(
    grad_wavefields: list[torch.Tensor], ndim: int
) -> list[torch.Tensor]:
    """The psi ping-pong temporaries for the acoustic adjoint."""
    return [torch.zeros_like(grad_wavefields[0]) for _ in range(ndim)]


def _acoustic_swap(state: "deepwave._engine.BackendState") -> None:
    """Ping-pong the psi/psin buffers after a chunk with odd internal steps."""
    ndim = state.ndim
    state.wavefields[-ndim:], state.aux_wavefields[:] = (
        state.aux_wavefields[:],
        state.wavefields[-ndim:],
    )


class AcousticBackwardFunc(torch.autograd.Function):
    """First-order acoustic adjoint with a Born-powered double-backward (HVP).

    ``forward`` runs the ordinary first-order adjoint (the acoustic backend);
    ``backward`` is the double-backward that computes the Hessian-vector product
    ``H*h`` via one acoustic Born forward + one acoustic Born backward, mirroring
    ``scalar.ScalarBackwardFunc`` (see
    ``docs/acoustic_born_derivation.md`` section 5). Unlike scalar/scalar_born,
    no ``grad_wfp`` negation is used (acoustic's first-order scheme seeds the
    adjoint fields directly from the output-wavefield gradients).
    """

    @staticmethod
    def forward(
        pml_profiles: list[torch.Tensor],
        sources_i: list[torch.Tensor],
        receivers_i: list[torch.Tensor],
        storage_manager: "deepwave._storage.StorageManager",
        grid_spacing: list[float],
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: list[int],
        source_amplitudes_requires_grad: list[bool],
        backward_callback: Optional["deepwave.Callback"],
        callback_frequency: int,
        boundary: list[int],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Runs the first-order acoustic adjoint via the shared engine."""
        ndim = len(grid_spacing)
        n = ndim + 1
        n_bg_wf = 1 + 3 * ndim
        grad_receivers = [t.contiguous() for t in tensors[:n]]
        models = [t.contiguous() for t in tensors[n : 2 * n]]
        source_amplitudes = [t.contiguous() for t in tensors[2 * n : 3 * n]]
        grad_wavefields = [t.contiguous() for t in tensors[3 * n : 3 * n + n_bg_wf]]
        # tensors[3*n+n_bg_wf:] are the forward's initial wavefields, needed only
        # by the double-backward (re-read from ``inputs`` in setup_context).
        pml_profiles = [p.contiguous() for p in pml_profiles]

        # The compiled adjoint reads the models (K at pressure nodes, each
        # buoyancy at its velocity nodes) at neighbour offsets near a reflecting
        # face, so the guard ring must hold the model's even/wrap image. Unlike
        # scalar's fresh ``v2dt2`` temporary, these models may alias the saved
        # user tensors, so mirror-pad *fresh clones* rather than writing in
        # place. All-PML boundaries are a no-op (returns the inputs unchanged).
        models = _boundaries.mirror_pad_acoustic_models(models, boundary, accuracy // 2)

        state = _engine.run_backward_core(
            SPEC,
            models=models,
            sources_i=sources_i,
            receivers_i=receivers_i,
            source_amplitudes=source_amplitudes,
            pml_profiles=pml_profiles,
            grad_wavefields=grad_wavefields,
            grad_receivers=grad_receivers,
            grid_spacing=grid_spacing,
            dt=dt,
            nt=nt,
            n_shots=n_shots,
            step_ratio=step_ratio,
            accuracy=accuracy,
            pml_width=pml_width,
            source_amplitudes_requires_grad=source_amplitudes_requires_grad,
            storage_manager=storage_manager,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            grad_model_batched=_acoustic_grad_model_batched,
            make_aux_wavefields=_acoustic_make_aux_wavefields,
            zero_wavefields=zero_edges_and_interiors,
            swap=_acoustic_swap,
            boundary=boundary,
        )
        out_s = state.output_slice
        return (
            *state.grad_models,
            *state.grad_sources,
            *[w[out_s] for w in state.wavefields],
        )

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Saves the tensors and metadata the double-backward (HVP) needs."""
        (
            pml_profiles,
            sources_i,
            receivers_i,
            storage_manager,
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
            *tensors,
        ) = inputs
        ndim = len(grid_spacing)
        n = ndim + 1
        n_bg_wf = 1 + 3 * ndim
        ctx.boundary = boundary
        grad_receivers = tensors[:n]
        models = tensors[n : 2 * n]
        source_amplitudes = tensors[2 * n : 3 * n]
        grad_wavefields = tensors[3 * n : 3 * n + n_bg_wf]
        wavefields = tensors[3 * n + n_bg_wf : 3 * n + 2 * n_bg_wf]
        ctx.save_for_backward(
            *grad_receivers,
            *models,
            *source_amplitudes,
            *grad_wavefields,
            *wavefields,
            *pml_profiles,
        )
        ctx.n = n
        ctx.n_bg_wf = n_bg_wf
        ctx.grid_spacing = list(grid_spacing)
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.pml_width = pml_width
        ctx.source_amplitudes_requires_grad = list(source_amplitudes_requires_grad)
        ctx.sources_i = sources_i
        ctx.receivers_i = receivers_i
        ctx.storage_compression = storage_manager.storage_compression

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Double-backward: the acoustic Hessian-vector product ``H*h``.

        One acoustic Born forward (recomputing the forward field and the effect
        of ``h`` on it) plus one acoustic Born backward (ordinary backprop for
        the scattered adjoint field, ``h``'s effect for the background adjoint
        field). Their correlation yields ``H*h`` (section 5.2).
        """
        # The Born adjoint also samples model derivatives when the seed is
        # a source or initial-state gradient. Reject the whole second-order
        # path; checking only model-gradient seeds misses mixed derivatives.
        if ctx.step_ratio > 1:
            raise DeepwaveRuntimeError(
                "acoustic: compiled double backward with sampled model gradients "
                "is unsupported. Use backend='eager' or an effective sampling "
                "stride of 1."
            )
        # The double-backward (HVP) routes through the acoustic Born backend,
        # which threads the first-class boundary ``bc`` codes,
        # so non-PML faces are supported here too: the same per-face
        # codes as the first-order adjoint are re-threaded into both the Born
        # forward (step 1) and the Born backward (step 2) below, and the
        # models correlated by step 2's compiled adjoint are mirror-padded
        # exactly as :class:`AcousticBackwardFunc`'s own first-order adjoint
        # does.
        boundary = list(ctx.boundary)
        acoustic_born_module = importlib.import_module("deepwave.acoustic_born")
        n = ctx.n
        n_bg_wf = ctx.n_bg_wf
        ndim = len(ctx.grid_spacing)

        # Incoming grads: perturbation direction ``h`` w.r.t. the forward
        # outputs (grad_models, grad_sources, final adjoint wavefields).
        gg_models = [a.contiguous() for a in args[:n]]
        gg_sources = [a.contiguous() for a in args[n : 2 * n]]
        gg_wavefields = [a.contiguous() for a in args[2 * n : 2 * n + n_bg_wf]]

        saved = list(ctx.saved_tensors)
        grad_receivers = saved[:n]
        models = [m.contiguous() for m in saved[n : 2 * n]]
        source_amplitudes = saved[2 * n : 3 * n]
        grad_wavefields = saved[3 * n : 3 * n + n_bg_wf]
        wavefields = saved[3 * n + n_bg_wf : 3 * n + 2 * n_bg_wf]
        pml_profiles = [p.contiguous() for p in saved[3 * n + 2 * n_bg_wf :]]
        sources_i = [s.contiguous() for s in ctx.sources_i]
        receivers_i = [r.contiguous() for r in ctx.receivers_i]

        device = models[0].device
        dtype = models[0].dtype
        grid_spacing = ctx.grid_spacing
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width

        # Fill missing perturbation directions with zeros.
        gg_models = [
            g if g.numel() > 0 else torch.zeros_like(m)
            for g, m in zip(gg_models, models)
        ]
        gg_sources = [
            g if g.numel() > 0 else torch.zeros_like(s)
            for g, s in zip(gg_sources, source_amplitudes)
        ]
        gg_wavefields = [
            g if g.numel() > 0 else torch.zeros_like(w)
            for g, w in zip(gg_wavefields, grad_wavefields)
        ]

        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_rcv_i = [empty_long for _ in range(n)]

        # --- Step 1: acoustic Born forward. Recompute the forward field
        # (background) and the effect of ``h`` on it (scattered). Force model
        # storage (so step 2 can read div_bg/div_sc/term stores) by marking the
        # background models as requiring grad; ``.forward`` (called directly, not
        # ``.apply``) builds no graph, so this only toggles the storage flag.
        models_store = [m.detach().requires_grad_(True) for m in models]
        born_out = acoustic_born_module.AcousticBornForwardFunc.forward(
            list(pml_profiles),
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_shots,
            None,  # forward_callback
            None,  # backward_callback
            1,  # callback_frequency
            "device",
            ".",
            ctx.storage_compression,
            boundary,  # boundary -- same per-face codes as the primal/adjoint
            *models_store,
            *gg_models,
            *source_amplitudes,
            *gg_sources,
            *sources_i,
            *sources_i,
            *empty_rcv_i,
            *receivers_i,
            *wavefields,
            *gg_wavefields,
        )
        born_saved_state = born_out[-1]
        storage_manager = born_saved_state.storage_manager
        # Scattered forward wavefields (effect of h on the forward field) and
        # scattered receivers (effect of h at the receivers).
        gg_receiver_out = list(born_out[2 * n_bg_wf + n : 2 * n_bg_wf + 2 * n])
        sc_forward_wavefields = list(born_out[n_bg_wf : 2 * n_bg_wf])

        # --- Step 2: acoustic Born full backward. The scattered adjoint field is
        # ordinary backprop (sourced by the original residual grad_receivers,
        # seeded by grad_wavefields); the background adjoint field is seeded at
        # zero and carries h's effect (via the coupling). Their correlation with
        # the step-1 stores accumulates H*h into grad_K/grad_B_d and grad_f.
        # The compiled adjoint reads both model groups (K/B_* AND the gg_models
        # perturbation direction, which plays acoustic_born's dK/dB_* role) at
        # neighbour offsets near a reflecting face, so both need the mirror-pad
        # guard ring, exactly as acoustic_born's own full adjoint applies it to
        # its [K, B_*] and [dK, dB_*] groups independently.
        fd_pad = accuracy // 2
        models_padded = _boundaries.mirror_pad_acoustic_models(
            [m.detach().requires_grad_(True) for m in models], boundary, fd_pad
        )
        gg_models_padded = _boundaries.mirror_pad_acoustic_models(
            [g.detach() for g in gg_models], boundary, fd_pad
        )
        models_full = models_padded + gg_models_padded
        # Adjoint residuals: background empty (no injection), scattered = the
        # original data residual.
        grad_receivers_full = [
            torch.empty(0, device=device, dtype=dtype) for _ in range(n)
        ] + [r.contiguous() for r in grad_receivers]
        receivers_i_full = empty_rcv_i + receivers_i
        sources_i_full = sources_i + sources_i
        # Initial adjoint wavefields: background zero, scattered = grad_wavefields.
        grad_wavefields_full = [torch.zeros_like(w) for w in grad_wavefields] + [
            w.contiguous() for w in grad_wavefields
        ]
        # Record only the background source gradient (the source HVP).
        sarg = list(ctx.source_amplitudes_requires_grad) + [False] * n

        state2 = _engine.run_backward_core(
            acoustic_born_module.SPEC,
            models=models_full,
            sources_i=sources_i_full,
            receivers_i=receivers_i_full,
            source_amplitudes=[],
            pml_profiles=pml_profiles,
            grad_wavefields=grad_wavefields_full,
            grad_receivers=grad_receivers_full,
            grid_spacing=grid_spacing,
            dt=dt,
            nt=nt,
            n_shots=n_shots,
            step_ratio=step_ratio,
            accuracy=accuracy,
            pml_width=pml_width,
            source_amplitudes_requires_grad=sarg,
            storage_manager=storage_manager,
            backward_callback=None,
            callback_frequency=1,
            grad_model_batched=_engine.model_batched,
            make_aux_wavefields=_acoustic_born_make_aux_wavefields,
            zero_wavefields=acoustic_born_module.zero_pml_interiors_born,
            make_call=acoustic_born_module._make_backward_call(ndim),  # noqa: SLF001
            swap=_acoustic_born_swap,
            boundary=boundary,
        )
        out_s = state2.output_slice
        # HVP for models (K, B_d) and the source; the background adjoint field's
        # final state is the HVP for the initial wavefields.
        hvp_models = list(state2.grad_models[:n])
        hvp_sources = list(state2.grad_sources[:n])
        hvp_wavefields = [w[out_s] for w in state2.wavefields[:n_bg_wf]]

        # Assemble the gradients w.r.t. AcousticBackwardFunc.forward's inputs.
        return (
            None,  # pml_profiles
            None,  # sources_i
            None,  # receivers_i
            None,  # storage_manager
            None,  # grid_spacing
            None,  # dt
            None,  # nt
            None,  # n_shots
            None,  # step_ratio
            None,  # accuracy
            None,  # pml_width
            None,  # source_amplitudes_requires_grad
            None,  # backward_callback
            None,  # callback_frequency
            None,  # boundary
            *gg_receiver_out,  # grad w.r.t. grad_receivers
            *hvp_models,  # grad w.r.t. models (K, B*) = H*h
            *hvp_sources,  # grad w.r.t. source_amplitudes
            *sc_forward_wavefields,  # grad w.r.t. grad_wavefields
            *hvp_wavefields,  # grad w.r.t. initial wavefields
        )


def _acoustic_born_make_aux_wavefields(
    grad_wavefields: list[torch.Tensor], ndim: int
) -> list[torch.Tensor]:
    """Psi ping-pong temporaries for the two-field Born adjoint (bg then sc)."""
    n_bg_wf = 1 + 3 * ndim
    return [torch.zeros_like(grad_wavefields[0]) for _ in range(ndim)] + [
        torch.zeros_like(grad_wavefields[n_bg_wf]) for _ in range(ndim)
    ]


def _acoustic_born_swap(state: "deepwave._engine.BackendState") -> None:
    """Ping-pong the psi blocks of both Born adjoint fields with their aux.

    ``acoustic_born_module._make_backward_call`` feeds
    ``w[1 + ndim : 1 + 2 * ndim]`` (bg) and
    ``w[n_bg_wf + 1 + ndim : n_bg_wf + 1 + 2 * ndim]`` (sc) to the kernel's
    ping-ponged ``psi_*_sc`` slots (paired with the aux buffers); the
    trailing ``... 2 * ndim : ... 3 * ndim`` blocks go to the single-buffer
    ``phi_*_sc`` slots and must NOT be swapped (see the fix for the same bug
    in ``acoustic_born.py``, commit 5969cb0).
    """
    ndim = state.ndim
    n_bg_wf = 1 + 3 * ndim
    w = state.wavefields
    a = state.aux_wavefields
    w[1 + ndim : 1 + 2 * ndim], a[:ndim] = (
        a[:ndim],
        w[1 + ndim : 1 + 2 * ndim],
    )
    w[n_bg_wf + 1 + ndim : n_bg_wf + 1 + 2 * ndim], a[ndim : 2 * ndim] = (
        a[ndim : 2 * ndim],
        w[n_bg_wf + 1 + ndim : n_bg_wf + 1 + 2 * ndim],
    )


def update_velocities(
    models: list[torch.Tensor],
    p: torch.Tensor,
    velocities: list[torch.Tensor],
    psi: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Updates the velocity wavefields and PML memory variables."""
    ndim = len(grid_spacing)

    # Extract PML profiles
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]

    buoyancy = models[1:]

    for dim in range(ndim):
        # Calculate grad p in dim
        gp = _staggered_grid.diff1h(p, dim, accuracy, 1 / grid_spacing[dim], ndim)

        # Update psi
        psi[dim] = ah[dim] * psi[dim] + bh[dim] * gp

        # Update v
        # Note: gp + psi is total effective gradient
        velocities[dim] = velocities[dim] - dt * buoyancy[dim] * (gp + psi[dim])

    return velocities, psi


def update_pressure(
    models: list[torch.Tensor],
    p: torch.Tensor,
    velocities: list[torch.Tensor],
    phi: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Updates the pressure wavefield and PML memory variables."""
    ndim = len(grid_spacing)

    # Extract PML profiles
    a = pml_profiles[::4]
    b = pml_profiles[1::4]

    k = models[0]

    div_v_sum = torch.zeros_like(p)
    for dim in range(ndim):
        # Calculate derivative of v in dim
        dv = _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )

        # Update phi
        phi[dim] = a[dim] * phi[dim] + b[dim] * dv

        # Accumulate div
        div_v_sum = div_v_sum + dv + phi[dim]

    p = p - dt * k * div_v_sum

    return p, phi


_update_velocities_cache = _engine.StepCache(update_velocities)
_update_velocities_opt = update_velocities
_update_pressure_cache = _engine.StepCache(update_pressure)
_update_pressure_opt = update_pressure

# Named offsets into the leading (pre-tensor-block) args acoustic_func
# receives from its call site (see
# _backend_utils.PROPAGATOR_REGISTRY["acoustic"].leading_args): the index of
# "boundary", and the index one past the last leading arg -- i.e. where the
# model/source/wavefield tensor block starts.
_BOUNDARY_IDX = _backend_utils.leading_arg_offset("acoustic", "boundary")
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["acoustic"].leading_args)


def acoustic_python(
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
    *args: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Python backend for acoustic wave propagation.

    ``boundary`` is the per-face integer code list (see
    :func:`deepwave._common.set_boundary`); ``BC_PML`` faces are absorbing and
    the guard ring is left untouched. Non-PML faces have their pressure and
    boundary-normal velocity guard cells refilled with the exact image each step
    (free/rigid/periodic), matching the eager scalar backend. ``backend`` is
    the resolved Python backend name (``"eager"`` or
    ``"compile"``), already selected by the caller; used only to name it if
    an unsupported option is rejected.
    """
    is_batched = any(_common.is_inside_vmap(x) for x in args)
    _engine.reject_unsupported_python_backend_options(
        SPEC,
        backend,
        storage_mode=storage_mode_str,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
        is_batched=is_batched,
    )
    ndim = len(grid_spacing)
    args_list = list(args)

    num_models = ndim + 1
    num_source_types = ndim + 1
    num_receiver_types = ndim + 1
    num_wf = 1 + 3 * ndim

    wavefields = args_list[-num_wf:]
    del args_list[-num_wf:]
    receivers_i = args_list[-num_receiver_types:]
    del args_list[-num_receiver_types:]
    sources_i = args_list[-num_source_types:]
    del args_list[-num_source_types:]
    source_amplitudes = args_list[-num_source_types:]
    del args_list[-num_source_types:]
    models = args_list[-num_models:]
    del args_list[-num_models:]

    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    has_boundary = any(code != _common.BC_PML for code in boundary)
    size_with_batch = (n_shots, *models[0].shape[-ndim:])

    wavefields = [
        _engine.create_or_pad(
            w, fd_pad_list, models[0].device, models[0].dtype, size_with_batch
        )
        for w in wavefields
    ]

    zero_edges_and_interiors(wavefields, ndim, fd_pad, fd_pad_list, pml_width)

    p = wavefields[0]
    velocities = wavefields[1 : 1 + ndim]
    phi = wavefields[1 + ndim : 1 + 2 * ndim]
    psi = wavefields[1 + 2 * ndim :]
    del wavefields

    device = models[0].device
    dtype = models[0].dtype
    shape = models[0].shape[-ndim:]
    flat_shape = int(torch.prod(torch.tensor(shape)).item())
    n_receivers_per_shot_list = [locs.numel() // n_shots for locs in receivers_i]
    receiver_amplitudes = [
        torch.empty(0, device=device, dtype=dtype) for _ in receivers_i
    ]
    receiver_amplitudes_lists: list[list[torch.Tensor]] = [[] for _ in receivers_i]

    for i, (locs, n_per_shot) in enumerate(zip(receivers_i, n_receivers_per_shot_list)):
        if locs.numel() > 0 and not is_batched:
            receiver_amplitudes[i].resize_(nt, n_shots, n_per_shot)
            receiver_amplitudes[i].fill_(0)

    # Prepare sources for injection
    # source_amplitudes here are already scaled by setup in acoustic()
    masked_sources = []
    for i in range(num_source_types):
        sources_i_masked, src_mask = _common.mask_locations(sources_i[i])
        src_amp_masked = _common.mask_source_amplitudes(source_amplitudes[i], src_mask)
        masked_sources.append(
            (sources_i_masked, src_amp_masked, i)  # i is the target_idx
        )

    masked_receivers = []
    for i, locs in enumerate(receivers_i):
        receivers_i_masked, recv_mask = _common.mask_locations(locs)
        masked_receivers.append((receivers_i_masked, recv_mask, i))

    # Wavefield names for callback (same order as the wavefields list).
    wf_names = acoustic_wavefield_names(ndim)
    callback_wavefields = {}
    model_names = acoustic_model_names(ndim)

    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            current_wavefields = [p, *velocities, *phi, *psi]
            callback_wavefields = dict(zip(wf_names, current_wavefields))
            _engine.invoke_forward_callback(
                forward_callback,
                CallbackState(
                    dt * step_ratio,
                    step,
                    callback_wavefields,
                    callback_models,
                    {},
                    fd_pad_list,
                    pml_width,
                    direction="forward",
                    n_steps=nt // step_ratio,
                    internal_dt=dt,
                ),
                differentiable=torch.is_grad_enabled()
                and any(t.requires_grad for t in args if isinstance(t, torch.Tensor)),
            )

        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step

            # Refill the pressure guard ring (read by the velocity update); this
            # also holds free-surface pressure nodes to zero before recording.
            if has_boundary:
                p = _boundaries.fill_acoustic_pressure_boundary(p, boundary, fd_pad)

            # Record receivers
            for i, (receivers_i_masked, _, target_idx) in enumerate(masked_receivers):
                if receivers_i_masked.numel() > 0:
                    val = torch.empty(0)
                    if target_idx == 0:  # Pressure
                        val = p.view(-1, flat_shape).gather(1, receivers_i_masked)
                    elif target_idx > 0 and target_idx <= ndim:  # Velocity
                        v_idx = target_idx - 1
                        val = (
                            velocities[v_idx]
                            .view(-1, flat_shape)
                            .gather(1, receivers_i_masked)
                        )
                    if val.numel() > 0:
                        if is_batched:
                            receiver_amplitudes_lists[i].append(val)
                        else:
                            receiver_amplitudes[i][t] = val

            # Update velocities
            velocities, psi = _update_velocities_opt(
                models,
                p,
                velocities,
                psi,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            # Inject velocity sources
            new_velocities = list(velocities)
            for sources_i_masked, src_amp_masked, target_idx in masked_sources:
                if target_idx > 0 and target_idx <= ndim and src_amp_masked.numel() > 0:
                    v_idx = target_idx - 1
                    new_velocities[v_idx] = (
                        new_velocities[v_idx]
                        .reshape(-1, flat_shape)
                        .scatter_add(1, sources_i_masked, src_amp_masked[t])
                        .view(size_with_batch)
                    )
            velocities = new_velocities

            # Refill the boundary-normal velocity guard ring (read by the
            # pressure update); holds rigid-wall velocity nodes to zero.
            if has_boundary:
                velocities = _boundaries.fill_acoustic_velocity_boundary(
                    velocities, boundary, fd_pad
                )

            # Update pressure
            p, phi = _update_pressure_opt(
                models,
                p,
                velocities,
                phi,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            # Inject pressure sources
            for sources_i_masked, src_amp_masked, target_idx in masked_sources:
                if target_idx == 0 and src_amp_masked.numel() > 0:
                    p = (
                        p.reshape(-1, flat_shape)
                        .scatter_add(1, sources_i_masked, src_amp_masked[t])
                        .view(size_with_batch)
                    )

    if is_batched:
        for i in range(len(receiver_amplitudes)):
            if len(receiver_amplitudes_lists[i]) > 0:
                receiver_amplitudes[i] = torch.stack(receiver_amplitudes_lists[i])

    for i, (_, recv_mask, _) in enumerate(masked_receivers):
        receiver_amplitudes[i] = _common.mask_receiver_amplitudes(
            receiver_amplitudes[i], recv_mask
        )

    # Leave the returned wavefields boundary-consistent (e.g. free-surface
    # pressure nodes held to zero, guard rings imaged) so a continuation run
    # reproduces the in-loop state.
    if has_boundary:
        p = _boundaries.fill_acoustic_pressure_boundary(p, boundary, fd_pad)
        velocities = _boundaries.fill_acoustic_velocity_boundary(
            list(velocities), boundary, fd_pad
        )

    wavefields = [p, *velocities, *phi, *psi]
    s = (slice(None), *(slice(fd_pad, shape - (fd_pad - 1)) for shape in shape))
    return (*[w[s] for w in wavefields], *receiver_amplitudes)


def acoustic_func(
    backend: Optional[Literal["eager", "compile"]] = None,
    *args: Any,
    max_vel_pinned: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Helper function to apply the AcousticForwardFunc.

    Args:
        backend: None, or a string specifying whether to use the
            Python backend. Already coerced (no bool) by the caller.
        *args: Variable length argument list to be passed directly to
            `AcousticForwardFunc.apply`.
        max_vel_pinned: Whether the caller supplied `max_vel`, so that the
            PML profiles do not depend on the model. Threaded to
            `AcousticForwardFunc` for the forward-mode (jvp) rule's caveat.

    Returns:
        The results of the forward pass from `AcousticForwardFunc.apply`.
    """
    global _update_velocities_opt, _update_pressure_opt

    if backend:
        k = cast("torch.Tensor", args[_N_LEADING_ARGS])
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        config_key = (len(args[1]), args[5], k.dtype)
        _update_velocities_opt = _update_velocities_cache.resolve(backend, config_key)
        _update_pressure_opt = _update_pressure_cache.resolve(backend, config_key)
        return acoustic_python(*args, backend=backend)

    # The compiled backend now threads the first-class boundary codes through
    # the ``bc`` ABI, so move ``boundary`` to the *last* leading (non-tensor)
    # autograd-Function argument (acoustic's local +1 leading-arg convention,
    # ``_N_LEAD``) and keep it, rather than stripping it.
    boundary = args[_BOUNDARY_IDX]
    args = (
        args[:_BOUNDARY_IDX]
        + args[_BOUNDARY_IDX + 1 : _N_LEADING_ARGS]
        + (max_vel_pinned, boundary)
        + args[_N_LEADING_ARGS:]
    )

    func = AcousticForwardFunc.apply

    # ``forward`` appends a ``ForwardSavedState`` trailing output for
    # ``setup_context``; strip it so callers see only the output tensors.
    return cast(
        "tuple[torch.Tensor, ...]",
        func(*args)[:-1],  # type: ignore[no-untyped-call]
    )


_common.document_propagator_module(Acoustic, acoustic)

"""Pseudo-acoustic VTI (Duveneck 2008) wave propagation module.

Implements the first-order velocity-stress pseudo-acoustic VTI wave equation
of Duveneck et al. (2008) -- Alkhalifah's ``V_S = 0`` limit of VTI elasticity
-- in variable-density media, on the same staggered grid, C-PML/M-PML
absorbing boundaries, and improved-vacuum free surface as
:mod:`deepwave.elastic` and :mod:`deepwave.elastic_vti`. The medium has a
vertical axis of rotational symmetry (the slowest spatial dimension: ``z`` in
3-D, ``y`` in 2-D).

Only two stress-like scalars survive when the shear moduli vanish:

- ``sigma_v`` -- the normal stress **along** the symmetry axis
  (``sigma_v = sigma_zz`` in 3-D, ``sigma_v = sigma_yy`` in 2-D);
- ``sigma_h`` -- the normal stress **across** the symmetry axis, equal in
  every direction perpendicular to it (``sigma_h = sigma_xx = sigma_yy`` in
  3-D, ``sigma_h = sigma_xx`` in 2-D).

The suffixes are ``v`` for the axial direction and ``h`` for the transverse
one because the symmetry axis is conventionally drawn vertical; Deepwave
attaches no orientation to any dimension, so read them as along and across
the symmetry axis, which is always the slowest spatial dimension.

With ``b = buoyancy = 1/rho`` and the symmetry axis along the slowest
dimension (``z``; drop ``y`` in 2-D)::

    d(sigma_h)/dt = c11 (dvx/dx + dvy/dy) + c13 dvz/dz + s_h(t)
    d(sigma_v)/dt = c13 (dvx/dx + dvy/dy) + c33 dvz/dz + s_v(t)
    d(vx)/dt      = b d(sigma_h)/dx
    d(vy)/dt      = b d(sigma_h)/dy
    d(vz)/dt      = b d(sigma_v)/dz

The stiffnesses are the true (density-normalised-free) VTI stiffnesses,
identical to those of :func:`deepwave.elastic_vti`::

    c33 = rho vp**2              (vp = P velocity along the axis)
    c11 = c33 (1 + 2 eps)
    c13 = c33 sqrt(1 + 2 delta)

so :func:`deepwave.anisotropy.thomsen_to_cij` with ``vs = 0`` produces exactly the
``(c11, c13, c33)`` needed (its ``c55 = c66 = 0``), differentiably.

This propagator is the exact ``c55 = c66 = 0`` reduction of
:func:`deepwave.elastic_vti`, with the two (3-D: three) normal stresses
collapsed onto ``(sigma_h, sigma_v)`` and all shear stresses (identically
zero) and their PML memories dropped. Compared with running ``elastic_vti``
with zero shear moduli it stores roughly half the fields and evaluates none
of the shear stencils.

**Stability (Grechka/Duveneck condition).** The system is well-posed only
where ``eps >= delta`` everywhere, equivalently ``c11 c33 >= c13**2``
elementwise. :func:`acoustic_vti` validates this and raises ``ValueError``
otherwise (``eps = delta``, elliptical anisotropy, is allowed).

**Pseudo-shear source artifact.** A point source in a region where
``eps != delta`` emits a diamond/rhombic pseudo-shear wave (the ``V_S = 0``
kinematic remnant). This is intrinsic to the formulation, not a bug. The
optional helper :func:`deepwave.anisotropy.source_taper` mitigates it by blending
``c13 -> sqrt(c11 c33)`` (``delta -> eps``, locally elliptical) within a
small radius of each source, as suggested by Duveneck et al. (2008).

References:
    Duveneck, E., Milcik, P., Bakker, P. M., and Perkins, C. (2008),
    "Acoustic VTI wave equations and their application for anisotropic
    reverse-time migration", SEG Technical Program Expanded Abstracts,
    2186-2190, doi:10.1190/1.3059320.

    Alkhalifah, T. (1998), "Acoustic approximations for processing in TI
    media", Geophysics 63, 623-631.

    Grechka, V., Zhang, L., and Rector, J. W. (2004), "Shear waves in
    acoustic anisotropic media", Geophysics 69, 576-582.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave
from deepwave import _backend_utils, _boundaries, _common, _engine, _staggered_grid
from deepwave._exceptions import DeepwaveRuntimeError, DeepwaveValueError
from deepwave._storage import (
    STORAGE_DEFAULT,
    Storage,
    resolve_storage,
    resolve_storage_arguments,
)
from deepwave.elastic import prepare_parameters as _prepare_elastic_parameters
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


def acoustic_vti_model_names(ndim: int) -> list[str]:
    """Names of the acoustic VTI models for the given number of dimensions.

    The models are the three VTI stiffnesses at the normal-stress (integer)
    nodes and the buoyancy averaged onto each velocity's staggered node. There
    are no shear stiffnesses (the ``V_S = 0`` reduction removes them entirely).
    """
    if ndim == 3:
        return ["c11", "c13", "c33", "buoyancy_z", "buoyancy_y", "buoyancy_x"]
    if ndim == 2:
        return ["c11", "c13", "c33", "buoyancy_y", "buoyancy_x"]
    raise DeepwaveValueError(
        "acoustic_vti: only 2 or 3 spatial dimensions are supported (in 1-D "
        "only c33 would remain, which is the 1-D scalar/acoustic equation; use "
        "deepwave.acoustic instead)."
    )


# The user-facing model arguments, in the order the public function takes
# them. The buoyancy is one argument here; the models above hold the
# per-component staggered averages of it that the kernel consumes.
USER_MODEL_NAMES = ("c11", "c13", "c33", "buoyancy")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(c11, c13, c33, buoyancy)``; the result is in
    :func:`acoustic_vti_model_names` order -- the three stiffnesses, then one
    buoyancy per axis.
    """
    c11, c13, c33, buoyancy = model_requires_grad
    return [c11, c13, c33] + [buoyancy] * ndim


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    One slot per stored derivative per axis block, not one per model: each
    axis contributes a buoyancy slot and a stiffness slot, and the three
    normal stiffnesses share the latter (they enter the same stored
    quantity, so any of them requiring a gradient stores it once).

    Measured against the engine over every combination of the four models at
    ndim 2 and 3 (``acoustic_vti`` refuses ndim 1).
    """
    c11, c13, c33, buoyancy = model_requires_grad
    normal_stiffness = c11 or c13 or c33
    return [buoyancy, normal_stiffness] * ndim


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models."""
    return [
        models[0].requires_grad,
        models[1].requires_grad,
        models[2].requires_grad,
        models[3].requires_grad,
    ]


def acoustic_vti_wavefield_names(ndim: int) -> list[str]:
    """Names of the acoustic VTI wavefields for the given number of dimensions.

    The order matches the flat storage/return order used by the eager core and
    by the initial-wavefield arguments (so a returned state round-trips as an
    ``initial_state``): the velocities (slowest to fastest dimension), then the
    two normal stresses ``sigma_v`` (axis) and ``sigma_h`` (transverse), then
    the ``ndim`` velocity-derivative C-PML memories ``m_v??`` (for the stress
    updates), then the ``ndim`` stress-derivative C-PML memories (for the
    velocity updates: ``d sigma_v / d(axis)`` and ``d sigma_h / d(transverse)``).
    """
    if ndim == 3:
        return [
            "vz_0",
            "vy_0",
            "vx_0",
            "sigma_v_0",
            "sigma_h_0",
            "m_vzz_0",
            "m_vyy_0",
            "m_vxx_0",
            "m_sigma_v_z_0",
            "m_sigma_h_y_0",
            "m_sigma_h_x_0",
        ]
    if ndim == 2:
        return [
            "vy_0",
            "vx_0",
            "sigma_v_0",
            "sigma_h_0",
            "m_vyy_0",
            "m_vxx_0",
            "m_sigma_v_y_0",
            "m_sigma_h_x_0",
        ]
    raise DeepwaveValueError(
        "acoustic_vti: only 2 or 3 spatial dimensions are supported."
    )


def acoustic_vti_source_receiver_names(ndim: int) -> list[str]:
    """Ordered source/receiver channel names: velocities then the two stresses.

    The two stresses are ordered ``sigma_v`` (axial) before ``sigma_h``
    (transverse), matching the sigma_v-first convention used throughout the
    propagator's keyword-argument channel groups (see A4.2).
    """
    if ndim == 3:
        return ["z", "y", "x", "sigma_v", "sigma_h"]
    return ["y", "x", "sigma_v", "sigma_h"]


def acoustic_vti_source_channels(ndim: int) -> list[str]:
    """Ordered source channel names, including the synthetic scalar channel.

    The scalar source channel is a *pressure rate*, not a volume rate: this
    propagator has no pressure variable and no volume-injection term, and its
    only scalar sources are the two normal-stress rates. It is synthetic --
    there is no backend slot for it -- because it is expanded by the wrapper
    into equal and opposite contributions on both stress channels (see
    :func:`deepwave.acoustic_vti`). It leads the list, matching the
    scalar-channel-first convention of every other multi-channel propagator;
    the remaining channels keep the output order of
    :func:`acoustic_vti_source_receiver_names`.
    """
    return ["pressure_rate", *acoustic_vti_source_receiver_names(ndim)]


def acoustic_vti_receiver_channels(ndim: int) -> list[str]:
    """Ordered receiver channel names, including the synthetic ``p`` channel.

    ``p`` is the pressure, ``-(sigma_v + (ndim - 1) sigma_h) / ndim``: minus
    the mean normal stress, which is what a hydrophone in this medium
    measures and what every other propagator's ``p`` receiver records. It is
    synthetic -- there is no backend slot for it -- because the wrapper
    records it by asking both stress channels for the same points and
    combining what they return. It leads the list, as the scalar channel does
    on every other propagator.
    """
    return ["p", *acoustic_vti_source_receiver_names(ndim)]


def _expand_pressure_rate_sources(
    amplitudes_p: Optional[torch.Tensor],
    locations_p: Optional[LocationsLike],
    amplitudes_v: Optional[torch.Tensor],
    locations_v: Optional[LocationsLike],
    amplitudes_h: Optional[torch.Tensor],
    locations_h: Optional[LocationsLike],
) -> tuple[
    Optional[torch.Tensor],
    Optional[LocationsLike],
    Optional[torch.Tensor],
    Optional[LocationsLike],
]:
    """Drives both normal-stress channels from a pressure-rate source.

    ``acoustic_vti`` has no pressure variable, so its ``p`` source channel is
    synthesised here rather than in the backend: it becomes an equal and
    opposite contribution on each of the two normal-stress rate channels.

    The sign is a flip and nothing else. This propagator scales a stress-rate
    source by ``dt`` and adds it to that stress (``sigma += dt * s``), and
    stress here is tension-positive; :func:`deepwave.elastic` scales its
    compression-positive ``source_amplitudes_pressure_rate`` by ``-dt`` and adds the
    result to *every* normal stress. So a pressure rate ``a`` must contribute
    ``-dt * a`` to each normal stress here too, which means
    ``s_v = s_h = -a``. There is no ``dt``, no modulus and no ``1/ndim``
    factor: the ``dt`` scaling is applied downstream to both channels
    identically, and the two stress channels together are the isotropic part.

    Args:
        amplitudes_p: The pressure-rate amplitudes, or ``None``.
        locations_p: The pressure-rate locations, or ``None``.
        amplitudes_v: Any explicit ``sigma_v`` amplitudes.
        locations_v: Any explicit ``sigma_v`` locations.
        amplitudes_h: Any explicit ``sigma_h`` amplitudes.
        locations_h: Any explicit ``sigma_h`` locations.

    Returns:
        Tuple: the ``sigma_v`` amplitudes and locations, then the ``sigma_h``
        amplitudes and locations, with the pressure-rate sources folded in.

    Raises:
        deepwave.DeepwaveValueError: If only one of the pressure-rate
            amplitudes/locations is given, if a :class:`deepwave.Hicks` is
            given for a channel that also has its own sources, or if the two
            source sets disagree in shot count or dtype.

    """
    if amplitudes_p is None and locations_p is None:
        # The no-op path every existing call takes: not one number moves.
        return amplitudes_v, locations_v, amplitudes_h, locations_h
    if amplitudes_p is None or locations_p is None:
        raise _common.unpaired_source_channel_error(
            "acoustic_vti", "_pressure_rate", amplitudes_p is None
        )
    negated = -amplitudes_p
    out: list[Any] = []
    for channel, amplitudes, locations in (
        ("sigma_v", amplitudes_v, locations_v),
        ("sigma_h", amplitudes_h, locations_h),
    ):
        if amplitudes is None and locations is None:
            out.extend([negated, locations_p])
            continue
        if amplitudes is None or locations is None:
            raise _common.unpaired_source_channel_error(
                "acoustic_vti", f"_{channel}", amplitudes is None
            )
        if not isinstance(locations_p, torch.Tensor) or not isinstance(
            locations, torch.Tensor
        ):
            raise DeepwaveValueError(
                "acoustic_vti: source_locations_pressure_rate may be a deepwave.Hicks "
                f"object only when the {channel} channel has no sources of "
                "its own, because two source sets cannot be concatenated "
                "inside a Hicks. Build the two stress channels yourself "
                "instead, with source_amplitudes_sigma_v = "
                "source_amplitudes_sigma_h = -source_amplitudes_pressure_rate."
            )
        if amplitudes.shape[0] != negated.shape[0]:
            raise DeepwaveValueError(
                f"acoustic_vti: source_amplitudes_pressure_rate has shape "
                f"{tuple(negated.shape)} and source_amplitudes_{channel} has "
                f"shape {tuple(amplitudes.shape)}; they are concatenated "
                "along the source axis, so their shot counts must match."
            )
        if amplitudes.dtype != negated.dtype:
            raise DeepwaveValueError(
                f"acoustic_vti: source_amplitudes_pressure_rate has dtype "
                f"{negated.dtype} and source_amplitudes_{channel} has dtype "
                f"{amplitudes.dtype}; they are concatenated, so their dtypes "
                "must match."
            )
        out.extend(
            [
                torch.cat([amplitudes, negated], dim=1),
                torch.cat([locations, locations_p], dim=1),
            ]
        )
    return (
        cast("Optional[torch.Tensor]", out[0]),
        cast("Optional[LocationsLike]", out[1]),
        cast("Optional[torch.Tensor]", out[2]),
        cast("Optional[LocationsLike]", out[3]),
    )


def _own_receiver_count(locations: Optional[LocationsLike]) -> int:
    """How many receivers of its own a stress channel was asked for.

    A :class:`deepwave.Hicks` counts its own points: the recording comes
    back with one column per point it was built from, whatever the stencil
    did in between.
    """
    if locations is None:
        return 0
    if isinstance(locations, torch.Tensor):
        return int(locations.shape[1])
    return int(locations.locations.shape[1])


def _pressure_receiver_columns(
    own: torch.Tensor, pressure: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Appends pressure points to a stress channel's own, without duplicates.

    A point recorded for the pressure that the channel already records is
    the same gather from the same cell, so asking for it twice would only
    trip the "receiver locations must be unique within each shot" check --
    a confusing failure, since the caller passed each set once, in different
    channels. Such a point is therefore silenced in the concatenated set
    (all of its coordinates become :data:`deepwave.IGNORE_LOCATION`, which
    records zeros and is exempt from the uniqueness check) and the pressure
    reads the column the channel was already recording.

    Duplicates *within* either set are left alone: those are one channel
    asking for the same cell twice, which is what the uniqueness check is
    for, and it still reports them.

    Args:
        own: The channel's own receiver locations, ``[shot, n_own, ndim]``.
        pressure: The pressure receiver locations, ``[shot, n_p, ndim]``.

    Returns:
        Tuple: the concatenated locations, and the column each shot's
        pressure receivers should be read from, ``[shot, n_p]``.

    """
    combined = torch.cat([own, pressure], dim=1)
    device = combined.device
    n_shots, n_total = combined.shape[0], combined.shape[1]
    n_own = own.shape[1]
    columns = torch.arange(n_own, n_total, device=device).expand(n_shots, -1).clone()
    if n_total == 0:
        return combined, columns
    positions = torch.arange(n_total, device=device)
    for shot in range(n_shots):
        # The first row equal to each row, by value. unique(dim=0) sorts, so
        # the inverse map groups equal rows and an amin over the group gives
        # the earliest of them.
        _, inverse = torch.unique(combined[shot], dim=0, return_inverse=True)
        first = torch.full(
            (int(inverse.max()) + 1,), n_total, dtype=torch.long, device=device
        )
        first = first.scatter_reduce(0, inverse, positions, reduce="amin")
        source = first[inverse][n_own:]
        seen = source < n_own
        columns[shot] = torch.where(seen, source, columns[shot])
        combined[shot, n_own:][seen] = deepwave.IGNORE_LOCATION
    return combined, columns


def _expand_pressure_receivers(
    locations_p: Optional[LocationsLike],
    locations_v: Optional[LocationsLike],
    locations_h: Optional[LocationsLike],
) -> tuple[
    Optional[LocationsLike],
    Optional[LocationsLike],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Records a pressure by asking both normal-stress channels for it.

    ``acoustic_vti`` has no pressure variable, so its ``p`` receiver channel
    is synthesised here rather than in the backend: the pressure points are
    added to both normal-stress channels' points, and the two recordings are
    combined afterwards into ``-(sigma_v + (ndim - 1) sigma_h) / ndim``,
    minus the mean normal stress. Both stresses live at the integer nodes,
    so one set of coordinates addresses both without any staggering
    correction.

    Args:
        locations_p: The pressure receiver locations, or ``None``.
        locations_v: Any explicit ``sigma_v`` receiver locations.
        locations_h: Any explicit ``sigma_h`` receiver locations.

    Returns:
        Tuple: the ``sigma_v`` and ``sigma_h`` locations with the pressure
        points added, then the columns of each channel's recording that hold
        the pressure, ``[shot, n_p]`` each. The columns are ``None`` when
        there are no pressure receivers, in which case nothing else changed
        either.

    Raises:
        deepwave.DeepwaveValueError: If a :class:`deepwave.Hicks` is given
            for a channel that also has receivers of its own, or if the two
            receiver sets disagree in shot count.

    """
    pressure_points = (
        locations_p
        if locations_p is None or isinstance(locations_p, torch.Tensor)
        else locations_p.locations
    )
    if pressure_points is None or pressure_points.shape[1] == 0:
        # A channel given no points records the same empty tensor as one
        # given no locations at all, so the two cases are one here.
        return locations_v, locations_h, None, None
    out: list[Optional[LocationsLike]] = []
    columns: list[Optional[torch.Tensor]] = []
    for channel, locations in (("sigma_v", locations_v), ("sigma_h", locations_h)):
        if locations is None:
            # The channel records the pressure points and nothing else, so
            # every column of its recording is a pressure column.
            out.append(locations_p)
            columns.append(
                torch.arange(
                    pressure_points.shape[1], device=pressure_points.device
                ).expand(1, -1)
            )
            continue
        if not isinstance(locations_p, torch.Tensor) or not isinstance(
            locations, torch.Tensor
        ):
            # Name whichever argument is the Hicks object, rather than
            # always blaming the pressure channel: either side may be the
            # one that cannot absorb the other's points.
            offender, other = (
                ("receiver_locations_p", f"the {channel}")
                if not isinstance(locations_p, torch.Tensor)
                else (f"receiver_locations_{channel}", "the p")
            )
            raise DeepwaveValueError(
                f"acoustic_vti: {offender} may be a deepwave.Hicks "
                f"object only when {other} channel has no receivers of "
                "its own, because two receiver sets cannot be combined "
                "inside a Hicks. Record the two stresses yourself instead, "
                "and combine them as -(sigma_v + (ndim - 1) sigma_h) / ndim."
            )
        if locations.shape[0] != locations_p.shape[0]:
            raise DeepwaveValueError(
                f"acoustic_vti: receiver_locations_p has shape "
                f"{tuple(locations_p.shape)} and receiver_locations_{channel} "
                f"has shape {tuple(locations.shape)}; they are recorded by "
                "one channel together, so their shot counts must match."
            )
        merged, merged_columns = _pressure_receiver_columns(locations, locations_p)
        out.append(merged)
        columns.append(merged_columns)
    return out[0], out[1], columns[0], columns[1]


def acoustic_vti_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The C-PML memory variables ``m_*``: one velocity-derivative memory per
    spatial dimension (used by the stress updates) and one stress-derivative
    memory per spatial dimension (used by the velocity updates). They are a
    subset of :func:`acoustic_vti_wavefield_names`, spelled the same way, and
    are what :class:`deepwave.AuxiliaryState` holds.
    """
    return [
        name for name in acoustic_vti_wavefield_names(ndim) if name.startswith("m_")
    ]


def _all_auxiliary_state_names() -> list[str]:
    """Every auxiliary-state name over both supported dimensionalities.

    The 2-D wavefield names are not a subset of the 3-D ones: the axial stress
    memory is ``m_sigma_v_y_0`` in 2-D but ``m_sigma_v_z_0`` in 3-D, and
    ``m_sigma_h_y_0`` exists only in 3-D. The initial-wavefield mapping handed
    to :func:`deepwave._engine.resolve_initial_state` is the union of both, so
    the set of names that mapping carries opaquely must be the union too --
    the ndim-3 list alone would leave ``m_sigma_v_y_0`` looking like a public
    state field.
    """
    return list(
        dict.fromkeys(
            acoustic_vti_auxiliary_state_names(3)
            + acoustic_vti_auxiliary_state_names(2)
        )
    )


def acoustic_vti_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`acoustic_vti`.

    One per receiver channel: the pressure, then the velocity components
    (slowest to fastest dimension), then the axial and transverse normal
    stresses.
    """
    return [
        f"receiver_amplitudes_{name}" for name in acoustic_vti_receiver_channels(ndim)
    ]


def acoustic_vti_output_names(ndim: int) -> list[str]:
    """Ordered names of the public outputs (final wavefields then receivers).

    The public final wavefields -- the velocities and the two normal stresses,
    followed by the single ``auxiliary_state`` that replaces the block of C-PML
    memory variables -- followed by the receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            acoustic_vti_wavefield_names(ndim),
            acoustic_vti_auxiliary_state_names(ndim),
        ),
        *acoustic_vti_receiver_names(ndim),
    ]


def _c_model_flag_indices(ndim: int) -> list[int]:
    """Indices of the independent models whose flags are passed to C.

    One entry per user-facing model: c11, c13, c33, and one representative of
    the (3-D: three) buoyancy copies (they all derive from the same user
    tensor, so their requires-grad/batched state is identical). Unlike
    elastic_vti there is no ndim-dependent shear stiffness, so this set is
    the same (4 entries) in 2-D and 3-D.
    """
    del ndim
    return [0, 1, 2, -1]


# Declarative description of the acoustic VTI propagator. There are ``ndim + 2``
# source/receiver channels (one per velocity component plus the two normal
# stresses ``sigma_h`` and ``sigma_v``).
SPEC = _engine.PropagatorSpec(
    name="acoustic_vti",
    status="experimental",
    ndims=(2, 3),
    source_channels=acoustic_vti_source_channels,
    receiver_channels=acoustic_vti_receiver_channels,
    synthetic_source_channels=frozenset({"pressure_rate"}),
    synthetic_receiver_channels=frozenset({"p"}),
    supported_options=frozenset({"mpml_fraction"}),
    staggered=True,
    model_names=acoustic_vti_model_names,
    wavefield_names=acoustic_vti_wavefield_names,
    n_source_types=lambda ndim: ndim + 2,
    n_receiver_types=lambda ndim: ndim + 2,
    output_names=acoustic_vti_output_names,
    auxiliary_state_names=acoustic_vti_auxiliary_state_names,
    public_wavefield_names=acoustic_vti_wavefield_names,
    c_model_flag_indices=_c_model_flag_indices,
    spatial_parallel=True,
)


def _prepare_buoyancy(buoyancy: torch.Tensor) -> list[torch.Tensor]:
    """Averages buoyancy onto the staggered velocity nodes (IVM/vacuum aware).

    Reuses the isotropic elastic buoyancy preparation (Zeng et al. 2012): the
    buoyancy at a velocity node staggered in dimension ``d`` is the inverse of
    the two-point arithmetic mean of the density across ``d``, with vacuum
    (``buoyancy = 0``) cells masked so a free surface is representable. No
    shear-modulus averaging is needed (there is no shear stiffness), so only
    the buoyancy outputs are returned.

    Args:
        buoyancy: Buoyancy (1/density) at the integer grid nodes.

    Returns:
        ``[buoyancy_z, buoyancy_y, buoyancy_x]`` in 3-D or
        ``[buoyancy_y, buoyancy_x]`` in 2-D, each averaged onto the
        corresponding velocity's staggered node.
    """
    ndim = buoyancy.ndim - 1
    # elastic.prepare_parameters returns [<shear averages...>, buoyancy_z,
    # buoyancy_y, buoyancy_x]; only the trailing ``ndim`` buoyancy entries are
    # needed here. Passing ``buoyancy`` as the (unused) shear argument keeps the
    # buoyancy averaging bit-identical to the elastic propagators without
    # duplicating the masking logic.
    return _prepare_elastic_parameters(buoyancy, buoyancy)[-ndim:]


def _acoustic_vti_axis_boundary(boundary: list[int]) -> list[int]:
    """Per-face boundary codes with every transverse face forced to PML.

    ``sigma_v`` (the axial normal stress) is guard-filled only on the axis
    face (dimension 0 -- z in 3-D, y in 2-D): per the module docstring's
    governing equations, ``sigma_v`` is fed by the axis-derivative term only
    (``d(sigma_v)/dt = c13 (dvx/dx + dvy/dy) + c33 dvz/dz``, and only
    ``d(vz)/dt = b d(sigma_v)/dz`` reads ``sigma_v`` back), so its transverse
    guard cells are never read by the forward kernel (confirmed against
    acoustic_vti.c's ``forward_vy_cell``/``forward_vx_cell``, which read only
    ``sigma_h``). Forcing the transverse codes to ``BC_PML`` makes
    :func:`deepwave._boundaries.fill_acoustic_pressure_boundary`'s existing
    per-axis ``if lo == 0 and hi == 0: continue`` skip every transverse face,
    reusing the acoustic image rule unchanged rather than inventing a new one
    (see ``docs/boundaries.rst``'s acoustic image-rule section).
    """
    out = [_common.BC_PML] * len(boundary)
    out[0], out[1] = boundary[0], boundary[1]
    return out


def _acoustic_vti_transverse_boundary(boundary: list[int]) -> list[int]:
    """Per-face boundary codes with the axis face forced to PML.

    ``sigma_h`` (the transverse normal stress) is guard-filled only on the
    transverse faces (every dimension but 0): per the module docstring,
    ``sigma_h`` is fed by the transverse-derivative terms
    (``d(sigma_h)/dt = c11 (dvx/dx + dvy/dy) + c13 dvz/dz``) and only
    ``d(vx)/dt``/``d(vy)/dt`` (transverse) read it back -- its axis guard
    cells are never read by the forward kernel (``forward_vz_cell`` reads
    only ``sigma_v``). See :func:`_acoustic_vti_axis_boundary` for the
    symmetric argument.
    """
    out = list(boundary)
    out[0], out[1] = _common.BC_PML, _common.BC_PML
    return out


def _velocity_extremes_acoustic_vti(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    buoyancy: torch.Tensor,
) -> tuple[float, float]:
    """Returns ``(max, min nonzero)`` qP phase-speed estimates for CFL/PML.

    The maximum qP phase velocity of the ``V_S = 0`` medium is attained on a
    symmetry axis, so ``vmax = sqrt(max(c11, c33) * b)`` pointwise is a rigorous
    (and tight) upper bound on the phase speed; its global maximum is used for
    the CFL time step and PML profiles. The minimum is the smallest nonzero of
    the axial and horizontal P speeds (``sqrt(c33 b)``, ``sqrt(c11 b)``), used
    only for the grid-dispersion (cells-per-wavelength) warning.

    Both leave as Python numbers, so neither is part of the autograd graph:
    they reshape the absorbing boundary and choose the internal time step,
    which are properties of the discretisation rather than differentiable
    functions of the model. ``.item()`` is what severs the graph; the
    ``detach()`` on the way in states that intent at the read.
    """
    del c13  # c13 does not raise the qP phase-speed maximum above the axes.
    c11 = c11.detach()
    c33 = c33.detach()
    buoyancy = buoyancy.detach()
    bound_sq = torch.maximum(c11, c33) * buoyancy
    max_model_vel = float(bound_sq.clamp(min=0).max().sqrt().item())
    vp_axis = (c33 * buoyancy).clamp(min=0).sqrt()
    vp_horz = (c11 * buoyancy).clamp(min=0).sqrt()
    nonzero = torch.cat(
        [vp_axis[vp_axis != 0].flatten(), vp_horz[vp_horz != 0].flatten()]
    )
    min_nonzero_model_vel = float(nonzero.min().item()) if nonzero.numel() > 0 else 0.0
    return max_model_vel, min_nonzero_model_vel


def acoustic_vti_source_taper(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    source_locations: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Locally elliptical ``c13`` around sources to suppress pseudo-shear.

    **This returns a modified model, not a mask or a window.** The ``c13``
    it gives back has been blended towards ``sqrt(c11 c33)`` within
    ``radius`` cells of every source, so a propagator run on it is running on
    a medium that differs from the one you passed in, near the sources: the
    anisotropy there has been made elliptical. That is the point -- it is
    what removes the pseudo-shear artefact -- but it means the simulated
    wavefield is not the wavefield of your own model in that neighbourhood,
    and a gradient computed with respect to the tapered ``c13`` is not the
    gradient with respect to yours there either. Keep ``radius`` small (a few
    cells), and if the near-source medium matters to you, do not use this.

    A point source in a region where ``eps != delta`` (i.e.
    ``c13 != sqrt(c11 c33)``) emits a diamond/rhombic pseudo-shear wave, the
    ``V_S = 0`` kinematic remnant. Following Duveneck et al. (2008), this helper
    blends ``c13`` towards the locally elliptical value ``sqrt(c11 c33)``
    (``delta -> eps``, for which no pseudo-shear artifact is emitted) inside a
    ``radius`` (in grid cells) of each source location, with a raised-cosine
    taper::

        w(r) = 0.5 (1 + cos(pi r / radius))   for r < radius, else 0
        c13_tapered = w sqrt(c11 c33) + (1 - w) c13

    so ``c13`` equals ``sqrt(c11 c33)`` exactly at a source (``r = 0``, ``w = 1``)
    and is returned unchanged at and beyond ``radius`` (``w = 0``), varying
    smoothly in between. The blend perturbs the model near the source; keep
    ``radius`` small (a few cells).

    The taper is the union over every location in ``source_locations`` (the
    nearest source governs each cell), and is applied on the model's trailing
    ``ndim`` spatial dimensions (broadcasting over any leading shot dimension of
    the models). Distances are measured in grid cells (Euclidean).

    Args:
        c11: The c11 stiffness model.
        c13: The c13 stiffness model (the one that is tapered).
        c33: The c33 stiffness model.
        source_locations: Integer source locations, shape
            ``[..., ndim]`` (e.g. ``[n_shots, n_sources, ndim]``); the leading
            dimensions are flattened and every listed location contributes.
            This helper does not interpret
            :data:`deepwave.IGNORE_LOCATION`: pass only the locations you
            want tapered.
        radius: The taper radius in grid cells (``> 0``).

    Returns:
        A new ``c13`` tensor (same shape/dtype/device as the input ``c13``) with
        the near-source blend applied. Differentiable with respect to
        ``c11``, ``c13``, and ``c33``.
    """
    if radius <= 0:
        raise DeepwaveValueError(
            f"source_taper: radius must be positive, but got {radius}."
        )
    ndim = source_locations.shape[-1]
    spatial_shape = c13.shape[-ndim:]
    if c11.shape != c13.shape or c33.shape != c13.shape:
        raise DeepwaveValueError(
            "source_taper: c11, c13, and c33 must have the same shape."
        )
    device = c13.device
    locs = source_locations.reshape(-1, ndim).to(device=device, dtype=c13.dtype)
    # Per-axis coordinate grids on the model's spatial dimensions.
    axes = [torch.arange(int(n), device=device, dtype=c13.dtype) for n in spatial_shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    # Squared distance to the nearest source location.
    dist_sq = torch.full(
        tuple(spatial_shape), float("inf"), device=device, dtype=c13.dtype
    )
    for loc in locs:
        d = sum((grids[d_i] - loc[d_i]) ** 2 for d_i in range(ndim))
        dist_sq = torch.minimum(dist_sq, d)
    dist = dist_sq.sqrt()
    inside = dist < radius
    weight = torch.where(
        inside,
        0.5 * (1 + torch.cos(torch.pi * dist / radius)),
        torch.zeros_like(dist),
    )
    # Broadcast the spatial weight over any leading (shot) dimensions.
    weight = weight.reshape((1,) * (c13.ndim - ndim) + tuple(spatial_shape))
    c13_elliptical = torch.sqrt((c11 * c33).clamp(min=0))
    return weight * c13_elliptical + (1 - weight) * c13


class AcousticVti(torch.nn.Module):
    """A PyTorch Module for pseudo-acoustic VTI wave propagation.

    Stability: experimental (see :doc:`/stability`).

    This Module stores the VTI stiffnesses (``c11``, ``c13``, ``c33``) and the
    buoyancy as ``torch.nn.Parameter`` objects, allowing gradients to be
    computed with respect to them if desired. See :func:`acoustic_vti` for a
    description of the physics, arguments, and outputs.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.
    The one exception to that freedom is this family: its symmetry axis
    is dimension 0 by construction, so here the axis order is a physics
    constraint rather than a documentation convention.

    Args:
        c11: The c11 stiffness model (horizontal P modulus), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        c13: The c13 stiffness model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c33: The c33 stiffness model (vertical P modulus), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        buoyancy: The buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. It is of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size (a single number or one per
            dimension).
        requires_grad: Which of this module's models require a gradient.
            ``None`` (the default) and ``False`` select none, ``True``
            selects every model this module owns, and a sequence of model
            names selects exactly those. Naming a model this class does not
            have, or an optional model that was not supplied, raises
            ``ValueError``. It is an alternative spelling for the
            ``<model>_requires_grad`` flags below, each of which is unset
            (``None``) by default and then means ``False``. Passing it
            together with an individual flag that was given explicitly
            raises ``ValueError`` rather than letting one silently win --
            whichever way the two point, and even when they agree, so that
            there is one rule rather than a direction-dependent one.
        c11_requires_grad: A bool specifying whether gradients will be
            computed for `c11`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        c13_requires_grad: A bool specifying whether gradients will be
            computed for `c13`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        c33_requires_grad: A bool specifying whether gradients will be
            computed for `c33`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        buoyancy_requires_grad: A bool specifying whether gradients will be
            computed for `buoyancy`. Unset by default, which means False. Cannot
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
            Compiled backend only (the eager backend requires
            `"device"`).
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
        c11: torch.Tensor,
        c13: torch.Tensor,
        c33: torch.Tensor,
        buoyancy: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        c11_requires_grad: Optional[bool] = None,
        c13_requires_grad: Optional[bool] = None,
        c33_requires_grad: Optional[bool] = None,
        buoyancy_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        names = ["c11", "c13", "c33", "buoyancy"]
        grad = _common.resolve_requires_grad(
            requires_grad,
            names,
            names,
            {
                "c11": c11_requires_grad,
                "c13": c13_requires_grad,
                "c33": c33_requires_grad,
                "buoyancy": buoyancy_requires_grad,
            },
            "AcousticVti",
        )
        self.c11 = _common.make_model_parameter(c11, grad["c11"], "c11")
        self.c13 = _common.make_model_parameter(c13, grad["c13"], "c13")
        self.c33 = _common.make_model_parameter(c33, grad["c33"], "c33")
        self.buoyancy = _common.make_model_parameter(
            buoyancy, grad["buoyancy"], "buoyancy"
        )
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
        source_amplitudes_pressure_rate: Optional[torch.Tensor] = None,
        source_amplitudes_z: Optional[torch.Tensor] = None,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_amplitudes_sigma_v: Optional[torch.Tensor] = None,
        source_amplitudes_sigma_h: Optional[torch.Tensor] = None,
        source_locations_pressure_rate: Optional[LocationsLike] = None,
        source_locations_z: Optional[LocationsLike] = None,
        source_locations_y: Optional[LocationsLike] = None,
        source_locations_x: Optional[LocationsLike] = None,
        source_locations_sigma_v: Optional[LocationsLike] = None,
        source_locations_sigma_h: Optional[LocationsLike] = None,
        source_mask_pressure_rate: Optional[torch.Tensor] = None,
        source_mask_z: Optional[torch.Tensor] = None,
        source_mask_y: Optional[torch.Tensor] = None,
        source_mask_x: Optional[torch.Tensor] = None,
        source_mask_sigma_v: Optional[torch.Tensor] = None,
        source_mask_sigma_h: Optional[torch.Tensor] = None,
        receiver_locations_p: Optional[LocationsLike] = None,
        receiver_locations_z: Optional[LocationsLike] = None,
        receiver_locations_y: Optional[LocationsLike] = None,
        receiver_locations_x: Optional[LocationsLike] = None,
        receiver_locations_sigma_v: Optional[LocationsLike] = None,
        receiver_locations_sigma_h: Optional[LocationsLike] = None,
        receiver_mask_p: Optional[torch.Tensor] = None,
        receiver_mask_z: Optional[torch.Tensor] = None,
        receiver_mask_y: Optional[torch.Tensor] = None,
        receiver_mask_x: Optional[torch.Tensor] = None,
        receiver_mask_sigma_v: Optional[torch.Tensor] = None,
        receiver_mask_sigma_h: Optional[torch.Tensor] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        sigma_v_0: Optional[torch.Tensor] = None,
        sigma_h_0: Optional[torch.Tensor] = None,
        initial_state: Optional[InitialState] = None,
        accuracy: Accuracy = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        boundary: Optional[Union[str, Sequence[str]]] = None,
        pml_freq: Optional[float] = None,
        mpml_fraction: float = 0.0,
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
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
        location_interpolation: Literal["auto", "hicks", "none"] = "auto",
        location_interpolation_halfwidth: int = 4,
    ) -> "deepwave.PropagatorOutput":
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`acoustic_vti` except that the
        stiffnesses, ``buoyancy``, ``grid_spacing``, and the storage settings
        (``storage_mode``/``storage_path``/``storage_compression``) do not
        need to be provided again -- they are supplied from the stored
        module state. See :func:`acoustic_vti` for a description of the
        inputs and outputs.
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
            backend, None, stacklevel=5
        ).canonical
        return acoustic_vti(
            self.c11,
            self.c13,
            self.c33,
            self.buoyancy,
            self.grid_spacing,
            dt,
            source_mask_pressure_rate=source_mask_pressure_rate,
            source_mask_z=source_mask_z,
            source_mask_y=source_mask_y,
            source_mask_x=source_mask_x,
            source_mask_sigma_v=source_mask_sigma_v,
            source_mask_sigma_h=source_mask_sigma_h,
            receiver_mask_p=receiver_mask_p,
            receiver_mask_z=receiver_mask_z,
            receiver_mask_y=receiver_mask_y,
            receiver_mask_x=receiver_mask_x,
            receiver_mask_sigma_v=receiver_mask_sigma_v,
            receiver_mask_sigma_h=receiver_mask_sigma_h,
            sources=sources,
            receivers=receivers,
            source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
            source_locations_pressure_rate=source_locations_pressure_rate,
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_amplitudes_sigma_v=source_amplitudes_sigma_v,
            source_amplitudes_sigma_h=source_amplitudes_sigma_h,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            source_locations_sigma_v=source_locations_sigma_v,
            source_locations_sigma_h=source_locations_sigma_h,
            receiver_locations_p=receiver_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_sigma_v=receiver_locations_sigma_v,
            receiver_locations_sigma_h=receiver_locations_sigma_h,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            sigma_v_0=sigma_v_0,
            sigma_h_0=sigma_h_0,
            initial_state=initial_state,
            accuracy=accuracy,
            pml_width=pml_width,
            boundary=boundary,
            pml_freq=pml_freq,
            mpml_fraction=mpml_fraction,
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


def acoustic_vti(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes_pressure_rate: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_amplitudes_sigma_v: Optional[torch.Tensor] = None,
    source_amplitudes_sigma_h: Optional[torch.Tensor] = None,
    source_locations_pressure_rate: Optional[LocationsLike] = None,
    source_locations_z: Optional[LocationsLike] = None,
    source_locations_y: Optional[LocationsLike] = None,
    source_locations_x: Optional[LocationsLike] = None,
    source_locations_sigma_v: Optional[LocationsLike] = None,
    source_locations_sigma_h: Optional[LocationsLike] = None,
    source_mask_pressure_rate: Optional[torch.Tensor] = None,
    source_mask_z: Optional[torch.Tensor] = None,
    source_mask_y: Optional[torch.Tensor] = None,
    source_mask_x: Optional[torch.Tensor] = None,
    source_mask_sigma_v: Optional[torch.Tensor] = None,
    source_mask_sigma_h: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[LocationsLike] = None,
    receiver_locations_z: Optional[LocationsLike] = None,
    receiver_locations_y: Optional[LocationsLike] = None,
    receiver_locations_x: Optional[LocationsLike] = None,
    receiver_locations_sigma_v: Optional[LocationsLike] = None,
    receiver_locations_sigma_h: Optional[LocationsLike] = None,
    receiver_mask_p: Optional[torch.Tensor] = None,
    receiver_mask_z: Optional[torch.Tensor] = None,
    receiver_mask_y: Optional[torch.Tensor] = None,
    receiver_mask_x: Optional[torch.Tensor] = None,
    receiver_mask_sigma_v: Optional[torch.Tensor] = None,
    receiver_mask_sigma_h: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigma_v_0: Optional[torch.Tensor] = None,
    sigma_h_0: Optional[torch.Tensor] = None,
    initial_state: Optional[InitialState] = None,
    accuracy: Accuracy = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    boundary: Optional[Union[str, Sequence[str]]] = None,
    pml_freq: Optional[float] = None,
    mpml_fraction: float = 0.0,
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
    storage: StorageLike = STORAGE_DEFAULT,
    storage_mode: Optional[StorageMode] = None,
    storage_path: Optional[StoragePath] = None,
    storage_compression: Optional[bool] = None,
    location_interpolation: Literal["auto", "hicks", "none"] = "auto",
    location_interpolation_halfwidth: int = 4,
) -> "deepwave.PropagatorOutput":
    """Pseudo-acoustic VTI wave propagation (functional interface).

    Stability: experimental (see :doc:`/stability`).

    Forward modelling of the Duveneck et al. (2008) first-order velocity-stress
    pseudo-acoustic VTI wave equation (the ``V_S = 0`` limit of VTI elasticity)
    in variable-density media. The outputs are differentiable with respect to
    the input float torch.Tensors (each stiffness and the buoyancy) and the
    source amplitudes.

    The propagator uses the same staggered grid, C-PML/M-PML absorbing
    boundaries, and off-grid (Hicks) source/receiver interpolation as
    :func:`deepwave.elastic_vti`, and is its exact ``c55 = c66 = 0`` reduction:
    the two normal stresses ``sigma_v`` (along the symmetry axis) and
    ``sigma_h`` (in the transverse plane) evolve, all shear stresses being
    identically zero and dropped along with their PML memories. See the module
    docstring for the governing equations.

    The symmetry axis is the slowest spatial dimension: ``z`` in 3-D, ``y`` in
    2-D. There is no 1-D form (only ``c33`` would survive, i.e. the 1-D scalar
    equation), so 1-D inputs raise an error.

    All stiffnesses are specified at the normal-stress (integer grid) nodes,
    with units of pressure (e.g. Pa in SI); ``buoyancy`` is 1/density and is
    internally averaged onto the staggered velocity nodes exactly as in
    :func:`deepwave.elastic` (a ``buoyancy = 0`` cell, with zero stiffnesses,
    is a vacuum, giving a free surface). The stiffnesses relate to the vertical
    P velocity ``vp`` and Thomsen ``eps``/``delta`` by ``c33 = rho vp**2``,
    ``c11 = c33 (1 + 2 eps)``, ``c13 = c33 sqrt(1 + 2 delta)``;
    :func:`deepwave.anisotropy.thomsen_to_cij` with ``vs = 0`` produces exactly these
    (its ``c55`` output is discarded).

    **Stability.** The medium must satisfy the Grechka/Duveneck condition
    ``c11 c33 >= c13**2`` (equivalently ``eps >= delta``) everywhere, else the
    pseudo-acoustic system is ill-posed; this is validated and a ``ValueError``
    is raised otherwise (``eps = delta``, elliptical anisotropy, is allowed and
    emits no pseudo-shear artifact).

    **Pseudo-shear source artifact.** Where ``eps != delta`` a point source also
    emits a spurious diamond/rhombic pseudo-shear wave (intrinsic to the
    ``V_S = 0`` formulation). :func:`deepwave.anisotropy.source_taper` optionally
    mitigates it by making ``c13`` locally elliptical near the sources.

    Sources and receivers may be placed on any velocity component (``z``, ``y``,
    ``x``; force-density sources, scaled internally by buoyancy and ``dt`` as in
    :func:`deepwave.elastic`) and on either normal stress (``sigma_h``,
    ``sigma_v``; stress-rate sources ``s_h``/``s_v``, scaled internally by
    ``dt``). An isotropic explosive/pressure source is the same wavelet applied
    to both the ``sigma_h`` and ``sigma_v`` channels.

    The compiled C/CUDA backend (``backend="compiled"``, and what the
    default ``backend="auto"`` selects here)
    implements both forward propagation and a hand-derived adjoint, matching
    the rest of the compiled propagator family (:func:`deepwave.elastic_vti`,
    etc.); see the ``backend`` Args entry below for the model-gradient
    ``step_ratio`` caveat. The eager Python backend (``backend="eager"``)
    instead provides gradients via native autograd through the time loop; the
    ``"torch_compile"`` Python mode runs
    the same eager time loop with the two step functions compiled.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.
    The one exception to that freedom is this family: its symmetry axis
    is dimension 0 by construction, so here the axis order is a physics
    constraint rather than a documentation convention.

    Args:
        c11: The c11 stiffness model (horizontal P modulus), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        c13: The c13 stiffness model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c33: The c33 stiffness model (vertical P modulus), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        buoyancy: The buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. It is of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size (a single number for all
            dimensions, or one per dimension).
        dt: The time step interval of the input and output. Deepwave is
            unit-agnostic: choose any single consistent system of units for
            the stiffnesses, ``buoyancy``, ``grid_spacing``, and ``dt``; see
            :doc:`/units`.
        source_amplitudes_z: Source amplitudes for the ``z`` velocity component
            (3-D only). See :func:`deepwave.elastic`. This channel exists only
            in 3-D.
        source_amplitudes_y: Source amplitudes for the ``y`` velocity
            component: a body-force density, as ``source_amplitudes_z``.
        source_amplitudes_x: Source amplitudes for the ``x`` velocity
            component: a body-force density, as ``source_amplitudes_z``.
        source_amplitudes_sigma_h: Source amplitudes for ``sigma_h``, the
            normal stress **across** the symmetry axis (a stress-rate source
            term ``s_h``, with units of pressure/time).
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``pressure_rate``, ``z``,
            ``y``, ``x``, ``sigma_v``, ``sigma_h`` (``z`` only in 3-D);
            these are equivalent to the flat
            ``source_amplitudes_*``/``source_locations_*`` keywords, which
            remain permanently supported; giving one channel in both forms
            raises. Optional.
        receivers: This simulation's receivers, keyed by channel, as the
            canonical alternative to the flat ``receiver_locations*`` keywords.
            Each value is a locations object, or ``(locations, mask)`` to
            include an activity mask. The channels are ``p``, ``z``, ``y``,
            ``x``, ``sigma_v``, ``sigma_h`` (``z`` only in 3-D). Giving one
            channel in both forms raises. Optional.
        source_amplitudes_pressure_rate: The amplitudes of a compression-positive
            pressure-rate source, with dimensions [shot, source, time]. This
            propagator has no pressure variable and no volume-injection
            term: its only scalar sources are the two normal-stress rates,
            so ``source_amplitudes_pressure_rate`` is a convenience that drives both of
            them at once. It is exactly equivalent to passing
            ``source_amplitudes_sigma_v = -source_amplitudes_pressure_rate`` and
            ``source_amplitudes_sigma_h = -source_amplitudes_pressure_rate`` at
            ``source_locations_pressure_rate`` on both stress channels --
            the sign flip
            is because stress here is tension-positive while a pressure is
            compression-positive -- and is bit-identical to writing that out
            by hand. The quantity is therefore a pressure rate (units of
            pressure/time), the same quantity as the elastic family's
            ``source_amplitudes_pressure_rate``, *not* the acoustic
            family's volume rate; see :doc:`/units`. If the
            ``sigma_v``/``sigma_h`` channels also have their own sources,
            these are concatenated with them along the source axis. The
            matching ``p`` *receiver* channel records the pressure this
            drives; see ``receiver_locations_p``. Optional; if given,
            ``source_locations_pressure_rate`` must also be given.
        source_amplitudes_sigma_v: Source amplitudes for ``sigma_v``, the
            normal stress **along** the symmetry axis (a stress-rate source
            term ``s_v``, with units of pressure/time). The same
            wavelet on both stress channels is an explosive/pressure source.
            Stress is tension-positive, so a positive amplitude is
            dilatational, the opposite sign to the elastic family's
            compression-positive ``source_amplitudes_pressure_rate``; see :doc:`/units`.
        source_locations_z: See :func:`deepwave.elastic` (and similarly for the
            other source/receiver location arguments). This channel exists only
            in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_y: See ``source_locations_z``.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_x: See ``source_locations_z``.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_pressure_rate: Integer locations of the
            ``source_amplitudes_pressure_rate`` sources, with dimensions
            [shot, source, ndim]. Applied to both
            stress channels. A :class:`deepwave.Hicks` object is accepted
            only when neither stress channel has sources of its own, since
            two source sets cannot be concatenated inside a Hicks; build the
            two channels yourself in that case. Optional; required if
            ``source_amplitudes_pressure_rate`` is given.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_sigma_h: Integer locations of the ``sigma_h``
            sources, the normal stress across the symmetry axis.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_sigma_v: Integer locations of the ``sigma_v`` sources.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_mask_pressure_rate: A boolean torch.Tensor with dimensions
            [shot, source] selecting which sources of the ``pressure_rate``
            channel are active. ``True`` means active; ``False`` means that
            point injects nothing, and its coordinates in
            ``source_locations_pressure_rate`` are then ignored entirely (they need
            not be valid, in range, or unique). The shape must match the
            leading dimensions of ``source_locations_pressure_rate``, which must be
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
        source_mask_sigma_v: A boolean torch.Tensor with dimensions [shot,
            source] selecting which sources of the ``sigma_v`` channel are
            active. ``True`` means active; ``False`` means that point injects
            nothing, and its coordinates in ``source_locations_sigma_v`` are
            then ignored entirely (they need not be valid, in range, or
            unique). The shape must match the leading dimensions of
            ``source_locations_sigma_v``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        source_mask_sigma_h: A boolean torch.Tensor with dimensions [shot,
            source] selecting which sources of the ``sigma_h`` channel are
            active. ``True`` means active; ``False`` means that point injects
            nothing, and its coordinates in ``source_locations_sigma_h`` are
            then ignored entirely (they need not be valid, in range, or
            unique). The shape must match the leading dimensions of
            ``source_locations_sigma_h``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_locations_z: See ``source_locations_z``. This channel exists
            only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_y: See ``source_locations_z``.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_x: See ``source_locations_z``.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_sigma_h: Integer locations of ``sigma_h``
            receivers, the normal stress **across** the symmetry axis.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_sigma_v: Integer locations of ``sigma_v``
            receivers, the normal stress **along** the symmetry axis.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_p: Integer locations of the pressure receivers.
            This propagator has no pressure variable, so the pressure is
            recorded as minus the mean normal stress,
            ``-(sigma_v + (ndim - 1) sigma_h) / ndim``: the same quantity
            every other propagator's ``p`` receiver records, and what a
            hydrophone in this medium measures. It is computed by asking
            both stress channels for these points, so a
            :class:`deepwave.Hicks` object is accepted only when neither
            stress channel has receivers of its own (two receiver sets
            cannot be concatenated inside a Hicks); record the two stresses
            yourself and combine them in that case. A point whose
            coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates
            set to the sentinel raises. Optional.
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
        receiver_mask_p: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``p`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``receiver_locations_p`` are then
            ignored entirely (they need not be valid, in range, or unique).
            The shape must match the leading dimensions of
            ``receiver_locations_p``, which must be given. This is the modern
            spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel and
            produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component
            raises, as does combining a mask with a :class:`deepwave.Hicks`
            object. Optional.
        receiver_mask_sigma_v: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``sigma_v`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``receiver_locations_sigma_v`` are
            then ignored entirely (they need not be valid, in range, or
            unique). The shape must match the leading dimensions of
            ``receiver_locations_sigma_v``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_mask_sigma_h: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``sigma_h`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``receiver_locations_sigma_h`` are
            then ignored entirely (they need not be valid, in range, or
            unique). The shape must match the leading dimensions of
            ``receiver_locations_sigma_h``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        accuracy: See :func:`deepwave.elastic`. Defaults to 4.
        pml_width: See :func:`deepwave.elastic`. Defaults to 20.
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: An optional single string, applying to every face, or a
            list of `2 * ndim` strings selecting a
            first-class boundary condition per face, in the same order as
            `pml_width` and with the same semantics as
            :func:`deepwave.acoustic`'s `boundary` (`"pml"`, `"free"`,
            `"rigid"`, or `"periodic"`; a non-`"pml"` face requires
            `pml_width == 0` on that face). acoustic_vti has two normal
            stresses instead of acoustic's one pressure: `sigma_v` (the axial
            stress) plays the "pressure" role -- and is guard-filled -- only
            on the axis face (dimension 0: `z` in 3-D, `y` in 2-D), while
            `sigma_h` (the transverse stress) plays that role, and is
            guard-filled, only on the transverse faces; each velocity is
            imaged on its own axis exactly as in :func:`deepwave.acoustic`.
            This follows directly from acoustic_vti's governing equations
            (see the module docstring): `sigma_v` is fed only by the axis
            derivative and feeds only the axis velocity back, `sigma_h`
            only by the transverse derivatives. See `docs/boundaries.rst`
            for the full image-rule citation. Validated on the eager
            (`backend="eager"`) and the default compiled C/CUDA
            backends; the `"torch_compile"` Python mode shares the eager
            guard-fill code path but is not separately validated for
            `boundary=`.
            Availability: `"pml"` and `"rigid"` are accepted on every face at
            every accuracy. `"free"` and `"periodic"` raise
            :class:`deepwave.DeepwaveNotImplementedError`: those guard-ring
            fills do not conserve a discrete energy, so the solution can grow
            without bound. Model a free surface with a vacuum layer instead
            (zero stiffnesses and zero `buoyancy` in the cells above the
            surface). Faces may be accepted in a later release; none is
            removed.
        pml_freq: See :func:`deepwave.elastic`.
        mpml_fraction: The multiaxial PML (M-PML) cross-damping fraction. A small
            positive value (0.15-0.2) can cure the C-PML corner instability
            that strong anisotropy sometimes triggers, at the cost of slightly
            larger boundary reflections. Defaults to 0.0 (standard C-PML).
        max_vel: See :func:`deepwave.elastic`. If not provided it is bounded from the
            stiffnesses (see below).
        courant_number: See :func:`deepwave.elastic`.
        survey_pad: See :func:`deepwave.scalar`.
        vz_0: The initial ``z`` velocity wavefield (3-D only). See
            :func:`deepwave.elastic` (and similarly for the other initial
            wavefields and PML memory variables).
        vy_0: The initial ``y`` velocity wavefield.
        vx_0: The initial ``x`` velocity wavefield.
        sigma_v_0: The initial normal stress along the symmetry axis.
        sigma_h_0: The initial normal stress across the symmetry axis.
        initial_state: See :func:`deepwave.elastic`. The mapping keys are this
            propagator's own public state field names (`vz_0`, `sigma_v_0`,
            ...) and `auxiliary_state`. The individual absorbing-boundary field
            names (`m_vzz_0`, ..., `m_sigma_h_x_0`) are no longer accepted as
            keys.
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
        backward_callback: See :func:`deepwave.elastic`. Not supported by the eager
            backend (raises if given).
        callback_frequency: See :func:`deepwave.elastic`.
        backend: Which execution backend to use: ``"auto"`` (the default),
            ``"compiled"``, ``"eager"`` or ``"torch_compile"``. See
            :doc:`/backends` for what each value selects, what ``"auto"``
            resolves to, and how the backends agree numerically.
            Both forward and backward are implemented in the compiled
            C/CUDA backend, so ``"auto"`` uses it. At
            ``model_gradient_sampling_interval == 1`` with a courant-safe
            ``dt`` (``step_ratio == 1``), model gradients
            (``c11``/``c13``/``c33``/``buoyancy``) match the eager backend to
            near machine precision; at ``step_ratio > 1`` (from either a
            coarse ``dt`` or ``model_gradient_sampling_interval > 1``) they
            are a bounded, non-machine-precision approximation (the model
            gradient is sampled and rescaled once per ``step_ratio`` block
            rather than exactly integrated), matching the shared behaviour of
            :func:`deepwave.elastic_vti`. The forward pass and the
            source/receiver-amplitude gradients are exact regardless of
            ``step_ratio``.
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
            Compiled backend only (the eager backend requires
            `"device"`).
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
        location_interpolation: See :func:`deepwave.elastic`.
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

            - vz_0: Final z-velocity wavefield (if ndim == 3).
            - vy_0: Final y-velocity wavefield.
            - vx_0: Final x-velocity wavefield.
            - sigma_v_0: Final normal stress along the symmetry axis.
            - sigma_h_0: Final normal stress across the symmetry axis.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables. Opaque: pass it back
              (as part of `initial_state`) to continue the simulation; its
              contents are implementation state rather than physics.
            - receiver_amplitudes_p: Recorded pressure data, computed as
              minus the mean normal stress,
              ``-(sigma_v + (ndim - 1) sigma_h) / ndim``.
            - receiver_amplitudes_z: Recorded z-velocity data (if ndim == 3).
            - receiver_amplitudes_y: Recorded y-velocity data.
            - receiver_amplitudes_x: Recorded x-velocity data.
            - receiver_amplitudes_sigma_v: Recorded normal stress along the
              symmetry axis.
            - receiver_amplitudes_sigma_h: Recorded normal stress across the
              symmetry axis.

        This positional order is easy to index incorrectly; prefer accessing
        the fields by name instead.
    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            acoustic_vti,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
            ),
        )
    backward_callback_needs_forward_wavefield = False
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
            "source_amplitudes_pressure_rate": source_amplitudes_pressure_rate,
            "source_locations_pressure_rate": source_locations_pressure_rate,
            "source_mask_pressure_rate": source_mask_pressure_rate,
            "source_amplitudes_z": source_amplitudes_z,
            "source_locations_z": source_locations_z,
            "source_mask_z": source_mask_z,
            "source_amplitudes_y": source_amplitudes_y,
            "source_locations_y": source_locations_y,
            "source_mask_y": source_mask_y,
            "source_amplitudes_x": source_amplitudes_x,
            "source_locations_x": source_locations_x,
            "source_mask_x": source_mask_x,
            "source_amplitudes_sigma_v": source_amplitudes_sigma_v,
            "source_locations_sigma_v": source_locations_sigma_v,
            "source_mask_sigma_v": source_mask_sigma_v,
            "source_amplitudes_sigma_h": source_amplitudes_sigma_h,
            "source_locations_sigma_h": source_locations_sigma_h,
            "source_mask_sigma_h": source_mask_sigma_h,
            "receiver_locations_z": receiver_locations_z,
            "receiver_mask_z": receiver_mask_z,
            "receiver_locations_y": receiver_locations_y,
            "receiver_mask_y": receiver_mask_y,
            "receiver_locations_x": receiver_locations_x,
            "receiver_mask_x": receiver_mask_x,
            "receiver_locations_p": receiver_locations_p,
            "receiver_mask_p": receiver_mask_p,
            "receiver_locations_sigma_v": receiver_locations_sigma_v,
            "receiver_mask_sigma_v": receiver_mask_sigma_v,
            "receiver_locations_sigma_h": receiver_locations_sigma_h,
            "receiver_mask_sigma_h": receiver_mask_sigma_h,
        },
    )
    _engine.expand_channel_mappings(
        SPEC,
        _channels,
        sources=sources,
        receivers=receivers,
    )
    _common.apply_location_masks(SPEC, _channels)
    source_amplitudes_pressure_rate = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_pressure_rate"]
    )
    source_locations_pressure_rate = cast(
        "Optional[LocationsLike]", _channels["source_locations_pressure_rate"]
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
    source_amplitudes_sigma_v = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_sigma_v"]
    )
    source_locations_sigma_v = cast(
        "Optional[LocationsLike]", _channels["source_locations_sigma_v"]
    )
    source_amplitudes_sigma_h = cast(
        "Optional[torch.Tensor]", _channels["source_amplitudes_sigma_h"]
    )
    source_locations_sigma_h = cast(
        "Optional[LocationsLike]", _channels["source_locations_sigma_h"]
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
    receiver_locations_p = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_p"]
    )
    receiver_locations_sigma_v = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_sigma_v"]
    )
    receiver_locations_sigma_h = cast(
        "Optional[LocationsLike]", _channels["receiver_locations_sigma_h"]
    )
    (
        source_amplitudes_sigma_v,
        source_locations_sigma_v,
        source_amplitudes_sigma_h,
        source_locations_sigma_h,
    ) = _expand_pressure_rate_sources(
        source_amplitudes_pressure_rate,
        source_locations_pressure_rate,
        source_amplitudes_sigma_v,
        source_locations_sigma_v,
        source_amplitudes_sigma_h,
        source_locations_sigma_h,
    )
    # How many receivers each stress channel was asked for in its own right,
    # before the pressure points join them: the columns of its recording
    # that belong to it rather than to the pressure channel.
    _n_own_receivers_sigma_v = _own_receiver_count(receiver_locations_sigma_v)
    _n_own_receivers_sigma_h = _own_receiver_count(receiver_locations_sigma_h)
    (
        receiver_locations_sigma_v,
        receiver_locations_sigma_h,
        _pressure_columns_sigma_v,
        _pressure_columns_sigma_h,
    ) = _expand_pressure_receivers(
        receiver_locations_p,
        receiver_locations_sigma_v,
        receiver_locations_sigma_h,
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
    _common.check_models(
        [c11, c13, c33, buoyancy],
        ["c11", "c13", "c33", "buoyancy"],
        "acoustic_vti",
    )
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend
    # Fill any initial wavefields supplied via ``initial_state``.
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
            "vz_0": vz_0,
            "vy_0": vy_0,
            "vx_0": vx_0,
            "sigma_v_0": sigma_v_0,
            "sigma_h_0": sigma_h_0,
            # m_* are the absorbing boundary's memory variables, not physical
            # fields, so they are not exposed as arguments; they travel in
            # `auxiliary_state`.
            "m_vzz_0": None,
            "m_vyy_0": None,
            "m_vxx_0": None,
            "m_sigma_v_z_0": None,
            "m_sigma_v_y_0": None,
            "m_sigma_h_y_0": None,
            "m_sigma_h_x_0": None,
        },
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=_all_auxiliary_state_names(),
        propagator=SPEC.name,
    )
    (
        vz_0,
        vy_0,
        vx_0,
        sigma_v_0,
        sigma_h_0,
        m_vzz_0,
        m_vyy_0,
        m_vxx_0,
        m_sigma_v_z_0,
        m_sigma_v_y_0,
        m_sigma_h_y_0,
        m_sigma_h_x_0,
    ) = _resolved_wavefields.values()

    _common.check_inputs_not_vmapped(
        source_amplitudes_z,
        source_amplitudes_y,
        source_amplitudes_x,
        source_amplitudes_sigma_v,
        source_amplitudes_sigma_h,
        source_locations_z,
        source_locations_y,
        source_locations_x,
        source_locations_sigma_v,
        source_locations_sigma_h,
        receiver_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
        receiver_locations_sigma_v,
        receiver_locations_sigma_h,
        vz_0,
        vy_0,
        vx_0,
        sigma_v_0,
        sigma_h_0,
        m_vzz_0,
        m_vyy_0,
        m_vxx_0,
        m_sigma_v_z_0,
        m_sigma_v_y_0,
        m_sigma_h_y_0,
        m_sigma_h_x_0,
    )

    raw_models: list[torch.Tensor] = [c11, c13, c33, buoyancy]
    all_wavefields: list[Optional[torch.Tensor]] = [
        vz_0,
        vy_0,
        vx_0,
        sigma_v_0,
        sigma_h_0,
        m_vzz_0,
        m_vyy_0,
        m_vxx_0,
        m_sigma_v_z_0,
        m_sigma_v_y_0,
        m_sigma_h_y_0,
        m_sigma_h_x_0,
    ]
    all_locations: list[Optional[LocationsLike]] = [
        source_locations_z,
        source_locations_y,
        source_locations_x,
        source_locations_sigma_v,
        source_locations_sigma_h,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
        receiver_locations_sigma_v,
        receiver_locations_sigma_h,
    ]
    ndim = _common.get_ndim(raw_models, all_wavefields, all_locations, [], [], [])
    if ndim < 2:
        raise DeepwaveValueError(
            "acoustic_vti: at least 2 spatial dimensions are required. In 1-D "
            "the VTI constitutive relation degenerates to the c33-only scalar "
            "equation (no anisotropy is expressible); use deepwave.acoustic "
            "or deepwave.scalar instead."
        )
    # The pressure-rate source and the ``p`` receiver are expanded into the
    # two stress channels above, so by the time the shared bounds check runs
    # the caller's Tensor sits in a ``sigma_h`` slot and would be reported
    # under that name. Check them here, while the caller's own spelling is
    # still attached to them; the shared check then finds nothing left to
    # report about them.
    _synthetic_locations = [
        (name, value)
        for name, value in (
            ("source_locations_pressure_rate", source_locations_pressure_rate),
            ("receiver_locations_p", receiver_locations_p),
        )
        if isinstance(value, torch.Tensor)
    ]
    if _synthetic_locations:
        _common.check_locations_are_within_model(
            list(c11.shape[-ndim:]),
            [value for _, value in _synthetic_locations],
            [name for name, _ in _synthetic_locations],
            "acoustic_vti",
        )
    if ndim < 3 and (
        source_locations_z is not None or receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            f"acoustic_vti: the propagation was determined to be {ndim}d, so "
            "locations related to the z dimension should not be provided."
        )

    if not 0 <= mpml_fraction < 1:
        raise DeepwaveValueError(
            "acoustic_vti: mpml_fraction (the multiaxial-PML cross-damping fraction) "
            f"must be in [0, 1), but got {mpml_fraction}."
        )

    source_amplitudes: list[Optional[torch.Tensor]] = []
    source_locations_in: list[Optional[LocationsLike]] = []
    receiver_locations_in: list[Optional[LocationsLike]] = []
    initial_wavefields: list[Optional[torch.Tensor]] = []
    if ndim >= 3:
        source_amplitudes.append(source_amplitudes_z)
        source_locations_in.append(source_locations_z)
        receiver_locations_in.append(receiver_locations_z)
        initial_wavefields.extend([vz_0, m_vzz_0, m_sigma_v_z_0, m_sigma_h_y_0])
    source_amplitudes.append(source_amplitudes_y)
    source_locations_in.append(source_locations_y)
    receiver_locations_in.append(receiver_locations_y)
    source_amplitudes.append(source_amplitudes_x)
    source_locations_in.append(source_locations_x)
    receiver_locations_in.append(receiver_locations_x)
    # NOTE: the compiled C/CUDA kernel's source/receiver argument ABI is
    # h-then-v for this pair (see acoustic_vti.c: f_sigma_h/f_sigma_v,
    # sources_sigma_h_i/sources_sigma_v_i, receivers_sigma_h_i/
    # receivers_sigma_v_i -- unlike the wavefields, which are v-then-h). This
    # internal channel order is a backend calling-convention detail and must
    # NOT be changed; the public v-then-h channel order (A4.2) is achieved by
    # relabelling via acoustic_vti_source_receiver_names and swapping the
    # trailing two receiver-amplitude outputs below, not by reordering the
    # backend argument lists here.
    source_amplitudes.append(source_amplitudes_sigma_h)
    source_locations_in.append(source_locations_sigma_h)
    receiver_locations_in.append(receiver_locations_sigma_h)
    source_amplitudes.append(source_amplitudes_sigma_v)
    source_locations_in.append(source_locations_sigma_v)
    receiver_locations_in.append(receiver_locations_sigma_v)

    # Assemble the initial wavefields in the flat storage order (see
    # acoustic_vti_wavefield_names): velocities, the two stresses, then the
    # velocity-derivative and stress-derivative PML memories.
    if ndim == 3:
        initial_wavefields = [
            vz_0,
            vy_0,
            vx_0,
            sigma_v_0,
            sigma_h_0,
            m_vzz_0,
            m_vyy_0,
            m_vxx_0,
            m_sigma_v_z_0,
            m_sigma_h_y_0,
            m_sigma_h_x_0,
        ]
    else:
        initial_wavefields = [
            vy_0,
            vx_0,
            sigma_v_0,
            sigma_h_0,
            m_vyy_0,
            m_vxx_0,
            m_sigma_v_y_0,
            m_sigma_h_x_0,
        ]

    # Off-grid (Hicks) source/receiver interpolation: velocity components carry
    # their half-cell staggering offsets; the two stresses live at integer
    # nodes (offset None), like a pressure channel.
    component_offsets: list[Optional[list[float]]] = [
        *_common.velocity_component_offsets(ndim),
        None,
        None,
    ]
    # Recorded before the location arguments are replaced by the integer grid
    # indices the backend consumes.
    location_interpolation_used = _engine.resolved_location_interpolation(
        location_interpolation,
        [source_locations_in, receiver_locations_in],
        [component_offsets, component_offsets],
    )
    # Which faces of the model an interpolation window may not cross. None of
    # this propagator's source/receiver components sits on the regular grid,
    # so the image of a reflecting face is component dependent and is not
    # implemented here: such a face is refused rather than approximated (see
    # ``_common.hicks_grid``).
    pml_inference_amplitudes = _common.active_source_amplitudes(
        source_amplitudes, source_locations_in
    )
    _hicks_grid = _common.hicks_grid(
        boundary,
        [*source_locations_in, *receiver_locations_in],
        c11,
        "acoustic_vti",
    )
    source_amplitudes, source_locations = _common.resolve_source_interpolators(
        source_locations_in,
        source_amplitudes,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        c11.dtype,
        _hicks_grid,
    )
    receiver_interpolators, receiver_locations = _common.resolve_receiver_interpolators(
        receiver_locations_in,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        c11.dtype,
        _hicks_grid,
    )

    # A velocity source/receiver must not sit on the final sample of its own
    # staggered dimension (there is no valid half-node there).
    dim_names = ["z", "y", "x"]
    for dim in range(ndim):
        for locations, name in [
            (source_locations, "source"),
            (receiver_locations, "receiver"),
        ]:
            location = locations[dim]
            if location is None:
                continue
            dim_location = location[..., -ndim + dim]
            dim_location = dim_location[dim_location != deepwave.IGNORE_LOCATION]
            if (
                dim_location.numel() > 0
                and dim_location.max() >= c11.shape[-ndim + dim] - 1
            ):
                raise DeepwaveRuntimeError(
                    f"acoustic_vti: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{c11.shape[-ndim + dim] - 1}.",
                )

    if not all(_common.values_readable(prop) for prop in raw_models):
        if max_vel is None:
            raise _common.max_vel_unreadable_error("acoustic_vti")
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        max_model_vel, min_nonzero_model_vel = _velocity_extremes_acoustic_vti(
            c11, c13, c33, buoyancy
        )
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

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
        raw_models,
        ["replicate"] * len(raw_models),
        ["c11", "c13", "c33", "buoyancy"],
        grid_spacing,
        dt,
        source_amplitudes,
        source_locations,
        receiver_locations,
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
    del c11, c13, c33, buoyancy, raw_models
    del source_amplitudes, source_locations, receiver_locations
    del initial_wavefields

    # Average buoyancy onto the velocity nodes; the stiffnesses are used
    # unaveraged at the integer nodes.
    c11_p, c13_p, c33_p, buoyancy_p = models
    models = [c11_p, c13_p, c33_p]
    models.extend(_prepare_buoyancy(buoyancy_p))
    del c11_p, c13_p, c33_p, buoyancy_p

    # Pre-scale the source amplitudes. Velocity (force-density) sources scale by
    # the buoyancy at the source location times dt (as in deepwave.elastic);
    # the two stress-rate sources s_h, s_v scale by dt (additive to the stress
    # updates, i.e. sigma += dt * s per step).
    model_shape = models[0].shape[-ndim:]
    n_src = ndim + 2
    for dim in range(ndim):
        if source_amplitudes_out[dim].numel() > 0:
            source_amplitudes_out[dim] = (
                source_amplitudes_out[dim]
                * _common.gather_model_at_sources(
                    models[3 + dim], sources_i[dim], model_shape, n_shots
                )
                * dt
            )
    for i in (n_src - 2, n_src - 1):  # sigma_h, sigma_v
        if source_amplitudes_out[i].numel() > 0:
            source_amplitudes_out[i] = source_amplitudes_out[i] * dt

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
        mpml_fraction=mpml_fraction,
    )

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "acoustic_vti")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "acoustic_vti")
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
        mpml=mpml_fraction > 0,
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
        acoustic_vti_func(
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
            n_shots,
            forward_callback,
            backward_callback,
            callback_frequency,
            storage_mode,
            storage_path,
            storage_compression,
            mpml_fraction > 0,
            boundary_codes,
            *models,
            *source_amplitudes_out,
            *sources_i,
            *receivers_i,
            *wavefields,
        )
    )

    # The trailing ``ndim + 2`` outputs are the receiver amplitudes.
    for i in range(len(outputs) - n_src, len(outputs)):
        outputs[i] = _common.downsample_and_movedim(
            outputs[i],
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
        )

    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - n_src
    )

    # The backend's receiver-amplitude channel order (built above, matching
    # the compiled kernel's source/receiver ABI) is h-then-v for this trailing
    # pair; the public channel order (A4.2, matching
    # acoustic_vti_source_receiver_names) is v-then-h. Swap the two receiver-
    # amplitude tensors here (a pure Python relabelling of already-computed
    # outputs) rather than reordering the backend argument lists.
    outputs[-2], outputs[-1] = outputs[-1], outputs[-2]

    # The synthetic pressure receiver channel. Both normal stresses recorded
    # the pressure points among their own (see _expand_pressure_receivers);
    # read each channel's pressure columns back out, combine the two into
    # minus the mean normal stress, and give the result the first receiver
    # slot -- the scalar channel leads the receivers on every propagator.
    # Each stress channel is then cut back to the receivers it was asked for
    # in its own right, which is the empty tensor every unrequested receiver
    # output is when it was asked for none.
    receivers_start = len(outputs) - n_src
    if _pressure_columns_sigma_v is not None and _pressure_columns_sigma_h is not None:
        stress_pressures = []
        for index, columns, n_own in (
            (-2, _pressure_columns_sigma_v, _n_own_receivers_sigma_v),
            (-1, _pressure_columns_sigma_h, _n_own_receivers_sigma_h),
        ):
            recorded = outputs[index]
            at = columns.to(recorded.device).expand(recorded.shape[0], -1)
            stress_pressures.append(
                recorded.gather(1, at[..., None].expand(-1, -1, recorded.shape[-1]))
            )
            outputs[index] = recorded[:, :n_own] if n_own > 0 else recorded.new_empty(0)
        pressure = -(stress_pressures[0] + (ndim - 1) * stress_pressures[1]) / ndim
    else:
        pressure = outputs[-1].new_empty(0)
    outputs.insert(receivers_start, pressure)

    final = tuple(outputs)
    return _engine.named_output(
        SPEC,
        ndim,
        final,
        info=run_info,
        accuracy=accuracy,
        boundary=boundary_names,
        mpml=mpml_fraction > 0,
        check_output_finite=check_output_finite,
    )


def update_velocities_acoustic_vti(
    models: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    velocities: list[torch.Tensor],
    s_mem_vars: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Updates the velocity wavefields and their C-PML memory variables (VTI).

    Each velocity is advanced from the half-node derivative of the normal stress
    that acts along its direction: the axial velocity (dimension 0) from
    ``d sigma_v / d(axis)``, and each transverse velocity from
    ``d sigma_h / d(transverse)`` (buoyancy is isotropic, so this is the same
    staggered update as ``elastic.update_velocities`` restricted to
    the surviving normal-stress terms). ``normal_stresses = [sigma_v, sigma_h]``
    and ``s_mem_vars`` holds the stress-derivative memory per velocity dimension.
    """
    ndim = len(grid_spacing)
    # Half-node PML profiles (the velocity update differentiates the
    # integer-node stresses onto the half nodes where the velocities live).
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    buoyancy = models[-ndim:]

    new_velocities: list[torch.Tensor] = []
    new_mem: list[torch.Tensor] = []
    for dim in range(ndim):
        # Dimension 0 is the symmetry axis (fed by sigma_v); transverse
        # dimensions are fed by sigma_h.
        feeding_stress = normal_stresses[0] if dim == 0 else normal_stresses[1]
        d = _staggered_grid.diff1h(
            feeding_stress, dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        mem = ah[dim] * s_mem_vars[dim] + bh[dim] * d
        d = d + mem
        new_mem.append(mem)
        new_velocities.append(velocities[dim] + buoyancy[dim] * dt * d)
    return new_velocities, new_mem


def update_stresses_acoustic_vti(
    models: list[torch.Tensor],
    velocities: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    v_mem_vars: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Updates the two normal stresses and their C-PML memory variables (VTI).

    With ``d[i]`` the PML-corrected integer-node ``d v_i / d x_i`` and the
    symmetry axis as dimension 0::

        sigma_v += dt (c33 d[0] + c13 (d[1] + ... transverse ...))
        sigma_h += dt (c13 d[0] + c11 (d[1] + ... transverse ...))

    ``normal_stresses = [sigma_v, sigma_h]``; ``v_mem_vars`` holds the
    velocity-derivative memory per dimension.
    """
    ndim = len(grid_spacing)
    a = pml_profiles[::4]
    b = pml_profiles[1::4]

    dvxdx = [
        _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]
    new_mem: list[torch.Tensor] = []
    for dim in range(ndim):
        mem = a[dim] * v_mem_vars[dim] + b[dim] * dvxdx[dim]
        dvxdx[dim] = dvxdx[dim] + mem
        new_mem.append(mem)

    c11, c13, c33 = models[0], models[1], models[2]
    d_axis = dvxdx[0]
    div_h = dvxdx[1]
    for dim in range(2, ndim):
        div_h = div_h + dvxdx[dim]
    sigma_v = normal_stresses[0] + dt * (c33 * d_axis + c13 * div_h)
    sigma_h = normal_stresses[1] + dt * (c13 * d_axis + c11 * div_h)
    return [sigma_v, sigma_h], new_mem


_update_velocities_acoustic_vti_cache = _engine.StepCache(
    update_velocities_acoustic_vti
)
_update_velocities_acoustic_vti_opt = update_velocities_acoustic_vti
_update_stresses_acoustic_vti_cache = _engine.StepCache(update_stresses_acoustic_vti)
_update_stresses_acoustic_vti_opt = update_stresses_acoustic_vti

# Index one past the last leading (pre-tensor-block) arg acoustic_vti_func
# receives from its call site -- i.e. where the model/source/wavefield
# tensor block starts (see
# _backend_utils.PROPAGATOR_REGISTRY["acoustic_vti"].leading_args). Unlike
# most other propagators, acoustic_vti_func has no "boundary" offset to
# derive: its call site already places ``boundary`` in the compiled
# backend's expected (trailing) slot, so no reshuffle is needed.
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["acoustic_vti"].leading_args)


def _zero_edges_and_interiors_acoustic_vti(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    mpml: bool,
) -> None:
    """Zeros the staggered-grid edges and PML memory interiors (acoustic VTI).

    Builds the acoustic-VTI ``half_grid`` / ``pml_edge_idx`` tables (see
    :func:`acoustic_vti_wavefield_names` for the field order) and delegates to
    the shared staggered-grid zeroing pass. A velocity ``v_i`` and its
    stress-derivative memory ``m_sigma_*_i`` live half a cell along dimension
    ``i`` (their outer edge in ``i`` is zeroed); the normal stresses and the
    velocity-derivative memories sit at integer nodes. Each dimension damps
    exactly two memory variables (one velocity-derivative, one
    stress-derivative).
    """
    num_vars = len(wavefields)
    half_grid: list[list[bool]] = [[False] * ndim for _ in range(num_vars)]
    if ndim == 3:
        # Fields: 0 vz, 1 vy, 2 vx, 3 sigma_v, 4 sigma_h, 5 m_vzz, 6 m_vyy,
        # 7 m_vxx, 8 m_sigma_v_z, 9 m_sigma_h_y, 10 m_sigma_h_x.
        half_grid[0][0] = True  # vz half in z
        half_grid[8][0] = True  # m_sigma_v_z half in z
        half_grid[1][1] = True  # vy half in y
        half_grid[9][1] = True  # m_sigma_h_y half in y
        half_grid[2][2] = True  # vx half in x
        half_grid[10][2] = True  # m_sigma_h_x half in x
        pml_edge_idx = [[5, 8], [6, 9], [7, 10]]
    else:
        # Fields: 0 vy, 1 vx, 2 sigma_v, 3 sigma_h, 4 m_vyy, 5 m_vxx,
        # 6 m_sigma_v_y, 7 m_sigma_h_x.
        half_grid[0][0] = True  # vy half in y (axis)
        half_grid[6][0] = True  # m_sigma_v_y half in y
        half_grid[1][1] = True  # vx half in x
        half_grid[7][1] = True  # m_sigma_h_x half in x
        pml_edge_idx = [[4, 6], [5, 7]]

    _staggered_grid.zero_edges_and_interiors(
        wavefields,
        ndim,
        fd_pad,
        fd_pad_list,
        pml_width,
        half_grid,
        pml_edge_idx,
        interior=True,
        mpml=mpml,
    )


def _acoustic_vti_eager_core(
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
    mpml: bool,
    boundary: list[int],
    *args: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Dedicated eager velocity-stress driver for the pseudo-acoustic VTI system.

    Runs the reduced field set (``ndim`` velocities, two normal stresses, and
    ``2 * ndim`` C-PML memories) through the leapfrog time loop in plain
    PyTorch, so gradients are provided by native autograd. Structurally
    parallel to ``elastic._elastic_eager_core`` (so the compiled
    kernels of a later task can be templated from it) but with the shear
    stresses and their stencils/memories removed. ``mpml`` is accepted for
    signature parity but unused (the eager path consumes the already-multiaxial
    ``pml_profiles`` directly by broadcasting). ``storage_mode_str``,
    ``storage_path``, and ``storage_compression`` are accepted for signature
    parity with the compiled-backend Function (see
    ``elastic._elastic_eager_core``); the eager backend has no
    manual backward storage, so only the ``"device"``/no-compression defaults
    are supported.

    ``boundary`` is the per-face first-class boundary code list;
    ``BC_PML`` (0) faces are untouched (a no-op fill), so
    ``boundary=None`` (mapped upstream to all-``BC_PML``) is bit-identical to
    the historical behaviour. ``sigma_v`` is guard-filled only on the axis
    face and ``sigma_h`` only on the transverse faces (see
    ``_acoustic_vti_axis_boundary``/``_acoustic_vti_transverse_boundary``),
    reusing ``_boundaries.fill_acoustic_pressure_boundary`` unchanged for each;
    the velocities reuse ``_boundaries.fill_acoustic_velocity_boundary``
    unchanged (each imaged on its own axis, exactly as in acoustic).

    ``args`` is the flat tensor block ``models, source_amplitudes, sources_i,
    receivers_i, wavefields`` (counts from :data:`SPEC`). ``backend`` is the
    resolved Python backend name (``"eager"`` or ``"compile"``),
    already selected by the caller; used only to name it if an unsupported
    option is rejected.
    """
    del mpml, storage_path
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
    n_models = 3 + ndim
    n_src = ndim + 2
    n_wavefields = 3 * ndim + 2

    args_list = list(args)
    del args
    wavefields_flat = args_list[-n_wavefields:]
    del args_list[-n_wavefields:]
    receivers_i = args_list[-n_src:]
    del args_list[-n_src:]
    sources_i = args_list[-n_src:]
    del args_list[-n_src:]
    source_amplitudes = args_list[-n_src:]
    del args_list[-n_src:]
    models = args_list[-n_models:]
    del args_list

    device = models[0].device
    dtype = models[0].dtype
    model_shape = models[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *model_shape)
    has_boundary = any(code != _common.BC_PML for code in boundary)
    axis_boundary = _acoustic_vti_axis_boundary(boundary)
    transverse_boundary = _acoustic_vti_transverse_boundary(boundary)

    wavefields_flat = [
        _engine.create_or_pad(wavefield, fd_pad_list, device, dtype, size_with_batch)
        for wavefield in wavefields_flat
    ]
    _zero_edges_and_interiors_acoustic_vti(
        wavefields_flat, ndim, fd_pad, fd_pad_list, pml_width, mpml=False
    )

    # Unpack the reduced field set (see acoustic_vti_wavefield_names).
    velocities = list(wavefields_flat[:ndim])
    # The two normal stresses are ordered axis-first: sigma_v then sigma_h.
    normal_stresses = [wavefields_flat[ndim], wavefields_flat[ndim + 1]]
    v_mem_vars = list(wavefields_flat[ndim + 2 : 2 * ndim + 2])
    s_mem_vars = list(wavefields_flat[2 * ndim + 2 : 3 * ndim + 2])
    del wavefields_flat

    wf_names = acoustic_vti_wavefield_names(ndim)
    model_names = acoustic_vti_model_names(ndim)
    callback_models = dict(zip(model_names, models))

    n_receivers_per_shot = [locs.numel() // n_shots for locs in receivers_i]
    receiver_amplitudes: list[torch.Tensor] = [
        torch.empty(0, device=device, dtype=dtype) for _ in range(n_src)
    ]
    receiver_amplitudes_lists: list[list[torch.Tensor]] = [[] for _ in range(n_src)]
    for i, loc in enumerate(receivers_i):
        if loc.numel() > 0 and not is_batched:
            receiver_amplitudes[i] = torch.zeros(
                nt, n_shots, n_receivers_per_shot[i], device=device, dtype=dtype
            )

    sources_i_masked = []
    source_amplitudes_masked = []
    for i, loc in enumerate(sources_i):
        masked, source_mask = _common.mask_locations(loc)
        sources_i_masked.append(masked)
        source_amplitudes_masked.append(
            _common.mask_source_amplitudes(source_amplitudes[i], source_mask)
        )
    del sources_i, source_amplitudes

    receivers_mask = []
    receivers_i_masked = []
    for loc in receivers_i:
        masked, recv_mask = _common.mask_locations(loc)
        receivers_mask.append(recv_mask)
        receivers_i_masked.append(masked)
    del receivers_i

    def record_receiver(channel: int, field: torch.Tensor, t: int) -> None:
        if receivers_i_masked[channel].numel() == 0:
            return
        val = field.reshape(-1, flat_model_shape).gather(1, receivers_i_masked[channel])
        if is_batched:
            receiver_amplitudes_lists[channel].append(val)
        else:
            receiver_amplitudes[channel][t] = val

    def inject_source(channel: int, field: torch.Tensor, t: int) -> torch.Tensor:
        if source_amplitudes_masked[channel].numel() == 0:
            return field
        return (
            field.clone()
            .reshape(-1, flat_model_shape)
            .scatter_add_(
                1, sources_i_masked[channel], source_amplitudes_masked[channel][t]
            )
            .reshape(size_with_batch)
        )

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = dict(
                zip(
                    wf_names,
                    [
                        *velocities,
                        normal_stresses[0],
                        normal_stresses[1],
                        *v_mem_vars,
                        *s_mem_vars,
                    ],
                )
            )
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
                and any(
                    t.requires_grad
                    for t in [
                        *models,
                        *source_amplitudes_masked,
                        *callback_wavefields.values(),
                    ]
                ),
            )
        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step

            # Refill the sigma_v/sigma_h guard rings (read by the velocity
            # update below); this also holds free-surface stress nodes to
            # zero before recording. sigma_v is imaged only on the axis face
            # (dimension 0), sigma_h only on the transverse faces -- see
            # _acoustic_vti_axis_boundary/_acoustic_vti_transverse_boundary.
            if has_boundary:
                normal_stresses[0] = _boundaries.fill_acoustic_pressure_boundary(
                    normal_stresses[0], axis_boundary, fd_pad
                )
                normal_stresses[1] = _boundaries.fill_acoustic_pressure_boundary(
                    normal_stresses[1], transverse_boundary, fd_pad
                )

            # Record receivers at the start of the step (as elastic does).
            for ridx in range(ndim):  # velocity receivers
                record_receiver(ridx, velocities[ridx], t)
            record_receiver(n_src - 2, normal_stresses[1], t)  # sigma_h
            record_receiver(n_src - 1, normal_stresses[0], t)  # sigma_v

            velocities, s_mem_vars = _update_velocities_acoustic_vti_opt(
                models,
                normal_stresses,
                velocities,
                s_mem_vars,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            # Velocity (force-density) sources, injected after the update.
            new_velocities = list(velocities)
            for sidx in range(ndim):
                new_velocities[sidx] = inject_source(sidx, new_velocities[sidx], t)
            velocities = new_velocities

            # Refill the boundary-normal velocity guard ring (read by the
            # stress update below); holds rigid-wall velocity nodes to zero.
            if has_boundary:
                velocities = _boundaries.fill_acoustic_velocity_boundary(
                    list(velocities), boundary, fd_pad
                )

            normal_stresses, v_mem_vars = _update_stresses_acoustic_vti_opt(
                models,
                velocities,
                normal_stresses,
                v_mem_vars,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            # Stress-rate sources s_h, s_v injected after the stress update.
            # normal_stresses = [sigma_v, sigma_h].
            normal_stresses[1] = inject_source(n_src - 2, normal_stresses[1], t)
            normal_stresses[0] = inject_source(n_src - 1, normal_stresses[0], t)

    if is_batched:
        for i in range(len(receiver_amplitudes)):
            if len(receiver_amplitudes_lists[i]) > 0:
                receiver_amplitudes[i] = torch.stack(receiver_amplitudes_lists[i])

    receiver_amplitudes_out = [
        _common.mask_receiver_amplitudes(amp, receivers_mask[i])
        for i, amp in enumerate(receiver_amplitudes)
    ]

    # Leave the returned wavefields boundary-consistent (e.g. free-surface
    # stress nodes held to zero, guard rings imaged) so a continuation run
    # reproduces the in-loop state (mirrors acoustic/viscoacoustic).
    if has_boundary:
        normal_stresses[0] = _boundaries.fill_acoustic_pressure_boundary(
            normal_stresses[0], axis_boundary, fd_pad
        )
        normal_stresses[1] = _boundaries.fill_acoustic_pressure_boundary(
            normal_stresses[1], transverse_boundary, fd_pad
        )
        velocities = _boundaries.fill_acoustic_velocity_boundary(
            list(velocities), boundary, fd_pad
        )

    # Crop the finite-difference halo off the final wavefields.
    s = (
        slice(None),
        *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
    )
    final_wavefields = [
        *velocities,
        normal_stresses[0],
        normal_stresses[1],
        *v_mem_vars,
        *s_mem_vars,
    ]
    return (
        *[field[s] for field in final_wavefields],
        *receiver_amplitudes_out,
    )


class AcousticVtiForwardFunc(torch.autograd.Function):
    """Forward propagation of the pseudo-acoustic VTI wave equation (compiled).

    Structurally identical to ``elastic_vti.ElasticVtiForwardFunc``
    (same generic engine, same storage/ping-pong machinery), reduced to the
    pseudo-acoustic field set. Both forward propagation (task 7.H milestone
    H1) and the hand-derived adjoint (:meth:`backward`, milestone H2) are
    implemented; see :meth:`backward`'s docstring for the model-gradient
    ``step_ratio`` approximation it shares with the rest of the compiled
    propagator family.
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
        mpml: bool,
        boundary: list[int],
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the pseudo-acoustic VTI wave equation.

        See ``elastic_vti.ElasticVtiForwardFunc.forward``; this is
        identical except for the models (reduced to ``c11, c13, c33,
        buoyancy``) and the storage flags that derive from them (2 slots per
        axis block -- ``buoyancy``, ``cnormal`` -- instead of elastic_vti's 4/3
        since there is no shear).

        Additional args:
            boundary: Per-face first-class boundary codes (see
                ``deepwave._common.set_boundary``); ``[BC_PML] * 2 * ndim``
                when ``boundary=None`` was requested, which keeps the
                compiled call byte-identical (the C/CUDA ``has_bc`` guard
                skips the fill for all-PML codes -- see acoustic_vti.c).
        """

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # One backward-storage slot per stored derivative, grouped by axis
            # block (axis[3-D only], transverse-1, transverse-0/base) exactly
            # as the .c/.cu SETUP_STORE ordering: buoyancy, then cnormal
            # (any of c11/c13/c33) for each block. models = [c11, c13, c33,
            # (buoyancy_z,) buoyancy_y, buoyancy_x] (see _acoustic_vti_model).
            buoyancy_rg = models[-1].requires_grad
            cnormal_rg = any(m.requires_grad for m in models[0:3])
            blocks = 3 if ndim == 3 else 2
            return [buoyancy_rg, cnormal_rg] * blocks

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            _zero_edges_and_interiors_acoustic_vti(
                wavefields, ndim, fd_pad, fd_pad_list, pml_width, mpml
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
            storage_flags=storage_flags,
            zero_wavefields=zero_wavefields,
            backend_extra="_mpml" if mpml else "",
            boundary=boundary,
        )
        return (*outputs, saved_state)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Populates ``ctx`` from the trailing state produced by ``forward``."""
        _engine.apply_forward_saved_state(ctx, output[-1])
        # ``boundary`` is the last leading input; the compiled backward call
        # needs it too (to thread the same ``bc`` codes) and the backward
        # adjoint needs it to mirror-pad the model guard rings (see
        # ``_boundaries.mirror_pad_acoustic_vti_models``).
        ctx.boundary = inputs[_N_LEAD - 1]

    # The ctypes backend cannot process batched/dual tensors; raise an
    # actionable error pointing to backend="eager".
    vmap = staticmethod(_engine.unsupported_vmap)
    jvp = staticmethod(_engine.unsupported_jvp)

    @staticmethod
    @_engine.no_second_order("acoustic_vti")
    def backward(
        ctx: Any,
        *args: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass (task 7.H milestone H2).

        The hand-derived adjoint (see the acoustic_vti.c header and
        docs/acoustic_vti_derivation.md Sec. 4-5). Structurally identical to
        ``elastic_vti.ElasticVtiForwardFunc.backward``: the
        wavefield/memory-variable structure (and hence the auxiliary
        allocation and ping-pong swap) is analogous, reduced to the
        pseudo-acoustic field set -- only the stress-derivative C-PML
        memories (m_sigma_v_z(3-D)/m_sigma_h_y(3-D)/m_sigma_v_y(2-D)/
        m_sigma_h_x) ping-pong; the velocity-derivative ones (m_vzz/m_vyy/
        m_vxx) do not (see the acoustic_vti.c backward() comments for why).
        """
        # Drop the grad slot for the trailing ``ForwardSavedState`` output.
        args = args[:-1]

        def make_aux_wavefields(
            grad_wavefields: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            # One ping-pong aux buffer per stress-derivative C-PML memory:
            # 3-D: m_sigma_v_z, m_sigma_h_y, m_sigma_h_x (3); 2-D:
            # m_sigma_v_y, m_sigma_h_x (2). See acoustic_vti_wavefield_names.
            n_aux = 3 if ndim == 3 else 2
            return [torch.zeros_like(grad_wavefields[0]) for _ in range(n_aux)]

        def swap(state: _engine.BackendState) -> None:
            # The stress-derivative C-PML memories are the trailing entries
            # of the wavefield list (see acoustic_vti_wavefield_names):
            # indices 8 through 10 inclusive in 3-D (m_sigma_v_z,
            # m_sigma_h_y, m_sigma_h_x), indices 6 through 7 inclusive in
            # 2-D (m_sigma_v_y, m_sigma_h_x).
            ndim = state.ndim
            gw = state.wavefields
            aw = state.aux_wavefields
            if ndim == 3:
                gw[8:11], aw[-3:] = aw[-3:], gw[8:11]
            else:
                gw[6:8], aw[-2:] = aw[-2:], gw[6:8]

        mpml = getattr(ctx, "backend_extra", "") == "_mpml"

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            _zero_edges_and_interiors_acoustic_vti(
                wavefields, ndim, fd_pad, fd_pad_list, pml_width, mpml
            )

        boundary = cast("list[int]", getattr(ctx, "boundary", []))

        def model_transform(models: list[torch.Tensor]) -> list[torch.Tensor]:
            # The compiled adjoint velocity update reads c33 (paired with
            # sigma_v, axis-only) and c11 (paired with sigma_h,
            # transverse-only) at neighbour offsets; c13 needs no mirror-pad
            # (see _boundaries.mirror_pad_acoustic_vti_models for the full
            # citation). Each buoyancy is read at neighbour offsets along its
            # own axis only, as in acoustic.
            return _boundaries.mirror_pad_acoustic_vti_models(
                models, boundary, ctx.accuracy // 2
            )

        return _engine.run_backward(
            ctx,
            SPEC,
            args,
            make_aux_wavefields=make_aux_wavefields,
            zero_wavefields=zero_wavefields,
            swap=swap,
            cpu_tmp_batch=ctx.n_shots,
            n_leading=_N_LEAD,
            boundary=boundary,
            model_transform=model_transform,
        )


# Number of leading non-tensor arguments of ``AcousticVtiForwardFunc.forward``
# (the shared config block plus the trailing ``mpml``/``boundary`` pair), all
# of which receive ``None`` gradients in ``backward`` (this propagator has no
# custom gradient-count logic to keep in sync -- unlike elastic's separate
# ``ElasticBackwardFunc``, acoustic_vti's own ``backward`` above delegates
# entirely to ``_engine.run_backward(..., n_leading=_N_LEAD)``, which builds
# its returned tuple's leading ``None``s from this same value, so adding
# ``boundary`` here needed no separate manual fix).
_N_LEAD = _engine.leading_arg_count(AcousticVtiForwardFunc.forward)


def acoustic_vti_func(
    backend: Optional[Literal["eager", "compile"]] = None,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """Dispatch the acoustic VTI forward pass to the eager or compiled backend.

    The eager Python backend (``backend="eager"``) provides gradients via
    native autograd; ``"torch_compile"`` (torch.compile)
    runs the same eager time loop with the two step functions compiled.
    The compiled C/CUDA backend (``backend=None``, the default)
    implements both forward propagation and the hand-derived adjoint; see
    :func:`acoustic_vti`'s ``backend`` Args entry for the model-gradient
    ``step_ratio`` approximation it shares with the rest of the compiled
    propagator family.

    Args:
        backend: ``"eager"``, ``"compile"``, or ``None`` (compiled
            C/CUDA, the default). Already coerced (no bool) by the caller.
        *args: The flat argument block forwarded to the selected backend.

    Returns:
        The forward-pass outputs (final wavefields then receiver amplitudes).
    """
    global _update_velocities_acoustic_vti_opt, _update_stresses_acoustic_vti_opt

    if backend:
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        _update_velocities_acoustic_vti_opt = (
            _update_velocities_acoustic_vti_cache.resolve(backend)
        )
        _update_stresses_acoustic_vti_opt = _update_stresses_acoustic_vti_cache.resolve(
            backend
        )
        return _acoustic_vti_eager_core(*args, backend=backend)

    # Compiled C/CUDA backend. ``forward`` appends a ``ForwardSavedState``
    # trailing output for ``setup_context``; strip it so callers see only the
    # output tensors (mirrors elastic_vti_func).
    return cast(
        "tuple[torch.Tensor, ...]",
        AcousticVtiForwardFunc.apply(*args)[:-1],
    )


_common.document_propagator_module(AcousticVti, acoustic_vti)

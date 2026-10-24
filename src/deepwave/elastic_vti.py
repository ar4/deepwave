"""Elastic VTI (vertically transverse isotropic) wave propagation module.

Implements elastic wave propagation in VTI media, in which the medium has a
vertical axis of rotational symmetry (the slowest spatial dimension: ``z`` in
3-D, ``y`` in 2-D). This is Deepwave's first anisotropic propagator. It
solves the velocity-stress formulation of the anisotropic elastic wave
equation on the same staggered grid as :mod:`deepwave.elastic`, using the
C-PML (or optionally multiaxial M-PML) absorbing boundaries and the improved
vacuum free-surface treatment of the isotropic elastic propagator. Only the
constitutive (stress-strain) relation differs.

The medium is described by the stiffnesses ``c11``, ``c13``, ``c33``,
``c55`` (and ``c66`` in 3-D), together with the buoyancy (1/density). With
the symmetry axis as the "3" direction of Voigt notation, the stress-rate
updates are (3-D, axis = z)::

    d(sigma_xx)/dt = c11 dvx/dx + c12 dvy/dy + c13 dvz/dz
    d(sigma_yy)/dt = c12 dvx/dx + c11 dvy/dy + c13 dvz/dz
    d(sigma_zz)/dt = c13 (dvx/dx + dvy/dy) + c33 dvz/dz
    d(sigma_xy)/dt = c66 (dvx/dy + dvy/dx)
    d(sigma_xz)/dt = c55 (dvx/dz + dvz/dx)
    d(sigma_yz)/dt = c55 (dvy/dz + dvz/dy)

with ``c12 = c11 - 2 c66``, and (2-D P-SV in the (y, x) plane, axis = y)::

    d(sigma_xx)/dt = c11 dvx/dx + c13 dvy/dy
    d(sigma_yy)/dt = c13 dvx/dx + c33 dvy/dy
    d(sigma_xy)/dt = c55 (dvx/dy + dvy/dx)

The isotropic elastic propagator is the special case ``c11 = c33 = lambda +
2 mu``, ``c13 = lambda``, ``c55 = c66 = mu``. There is no 1-D form: with a
single spatial dimension only ``c33`` would survive, which is the isotropic
1-D elastic equation, so 1-D inputs raise an error (use :func:`deepwave.elastic`).

The helper :func:`deepwave.anisotropy.thomsen_to_cij` converts Thomsen (1986) parameters
(vp, vs, rho, epsilon, delta, gamma) to these stiffnesses and the buoyancy,
using the exact (not weak-anisotropy linearised) relations, and is
differentiable, so inversion may be performed in Thomsen parameter space by
chaining autograd through it; :func:`cij_to_thomsen` inverts it.
:func:`alkhalifah_to_thomsen` converts the P-wave-derivable
Alkhalifah-Tsvankin (1995) NMO-velocity/anellipticity parameterisation
(vp, vnmo, eta) to Thomsen's (epsilon, delta), so a model built from
velocity-analysis-derived parameters can be chained straight into
:func:`deepwave.anisotropy.thomsen_to_cij`, and
:func:`thomsen_to_alkhalifah` converts back.
All of them are documented in, and should be imported from,
:mod:`deepwave.anisotropy`.
"""

from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Literal,
    Optional,
    Union,
    cast,
)

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
from deepwave.elastic import (
    _elastic_eager_core,
    _elastic_public_wavefield_order,
    elastic_auxiliary_state_names,
    elastic_public_wavefield_names,
    elastic_receiver_names,
    elastic_wavefield_names,
    update_velocities,
    zero_edges_and_interiors,
)
from deepwave.elastic import (
    prepare_parameters as _prepare_isotropic_parameters,
)
from deepwave.types import (
    Accuracy,
    Callback,
    InitialState,
    LocationsLike,
    ReceiverSpec,
    RunInfo,
    SourceSpec,
    StorageLike,
    StorageMode,
    StoragePath,
)


def elastic_vti_model_names(ndim: int) -> list[str]:
    """Names of the elastic VTI models for the given number of dimensions.

    The models are the VTI stiffnesses at the normal-stress nodes (including,
    in 3-D, the derived ``c12 = c11 - 2 c66``), the shear stiffnesses
    evaluated at the staggered shear-stress locations (one per off-diagonal
    stress component: ``c55`` for the stresses involving the symmetry axis,
    ``c66`` for the horizontal-plane stress), and the buoyancy for each
    velocity component.
    """
    if ndim == 3:
        return [
            "c11",
            "c12",
            "c13",
            "c33",
            "c55_zy",
            "c55_zx",
            "c66_yx",
            "buoyancy_z",
            "buoyancy_y",
            "buoyancy_x",
        ]
    if ndim == 2:
        return ["c11", "c13", "c33", "c55_yx", "buoyancy_y", "buoyancy_x"]
    raise DeepwaveValueError(
        "elastic_vti: only 2 or 3 spatial dimensions are supported (in 1-D "
        "only c33 would remain, which is the isotropic elastic equation; use "
        "deepwave.elastic instead)."
    )


# The user-facing model arguments common to every supported ndim, in the
# order the public function takes them. ``c66`` is a model argument too, but
# only in 3-D, so the ndim-dependent answer comes from
# :func:`user_model_names` below.
USER_MODEL_NAMES = ("c11", "c13", "c33", "c55", "buoyancy")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is in :func:`user_model_names` order for this
    ``ndim``: ``(c11, c13, c33, c55, buoyancy)`` in 2-D, with ``c66``
    appended in 3-D. The result is in :func:`elastic_vti_model_names` order.

    In 3-D the internal ``c12`` is ``c11 - 2 * c66``, so it carries a
    gradient if either does.
    """
    if ndim == 3:
        c11, c13, c33, c55, buoyancy, c66 = model_requires_grad
        return [
            c11,
            c11 or c66,
            c13,
            c33,
            c55,
            c55,
            c66,
            buoyancy,
            buoyancy,
            buoyancy,
        ]
    c11, c13, c33, c55, buoyancy = model_requires_grad
    return [c11, c13, c33, c55, buoyancy, buoyancy]


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    One slot per stored derivative, grouped by axis, in the same shape as
    :func:`deepwave.elastic`'s -- with elastic's single staggered-shear group
    split into ``c55`` and (in 3-D) ``c66``. ``c66`` also enters the normal
    stiffness group, because the internal ``c12`` is built from it.

    Measured against the engine over every combination of the models at
    ndim 2 and 3 (``elastic_vti`` refuses ndim 1).
    """
    if ndim == 3:
        c11, c13, c33, c55, buoyancy, c66 = model_requires_grad
        normal_stiffness = c11 or c13 or c33 or c66
        return [
            buoyancy,
            normal_stiffness,
            c55,
            c55,
            buoyancy,
            normal_stiffness,
            c66,
            buoyancy,
            normal_stiffness,
        ]
    c11, c13, c33, c55, buoyancy = model_requires_grad
    normal_stiffness = c11 or c13 or c33
    return [buoyancy, normal_stiffness, c55, buoyancy, normal_stiffness]


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models."""
    if ndim == 3:
        return [
            models[0].requires_grad,
            models[2].requires_grad,
            models[3].requires_grad,
            models[4].requires_grad,
            models[7].requires_grad,
            models[6].requires_grad,
        ]
    return [
        models[0].requires_grad,
        models[1].requires_grad,
        models[2].requires_grad,
        models[3].requires_grad,
        models[4].requires_grad,
    ]


def user_model_names(ndim: int) -> tuple[str, ...]:
    """The user-facing model arguments at ``ndim``, in the order taken.

    ``c66`` is an independent stiffness only in 3-D, where it is required; in
    2-D it does not enter the P-SV equations and must be ``None``. Every
    other model argument is the same in both.
    """
    if ndim == 3:
        return (*USER_MODEL_NAMES, "c66")
    return USER_MODEL_NAMES


def _c_model_flag_indices(ndim: int) -> list[int]:
    """Indices of the independent models whose flags are passed to C.

    One entry per user-facing model: the stiffnesses at the normal-stress
    nodes, one representative of the ``c55``-derived shear models (they all
    derive from the same user tensor, so their requires-grad/batched state is
    identical), ``c66`` (3-D), and one representative buoyancy.
    """
    if ndim == 3:
        # c11, c12, c13, c33, c55 (via c55_zy), c66 (via c66_yx), buoyancy.
        return [0, 1, 2, 3, 4, 6, -1]
    # c11, c13, c33, c55 (via c55_yx), buoyancy.
    return [0, 1, 2, 3, -1]


def elastic_vti_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`deepwave.elastic_vti`, in return order.

    The public final wavefields -- the velocities and stresses in the
    isotropic elastic propagator's public (grouped-by-kind) order, this
    propagator sharing that layout and differing only in its models, followed
    by the single ``auxiliary_state`` that replaces the block of PML memory
    variables -- followed by this propagator's own receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            elastic_public_wavefield_names(ndim), elastic_auxiliary_state_names(ndim)
        ),
        *elastic_receiver_names(ndim),
    ]


# Declarative description of the elastic VTI propagator. The staggered-grid
# wavefield layout (and hence the wavefield and output names) is identical to
# the isotropic elastic propagator's; only the models differ.
SPEC = _engine.PropagatorSpec(
    name="elastic_vti",
    status="experimental",
    ndims=(2, 3),
    source_channels=_engine.pressure_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    supported_options=frozenset({"mpml_fraction"}),
    staggered=True,
    model_names=elastic_vti_model_names,
    wavefield_names=elastic_wavefield_names,
    n_source_types=lambda ndim: ndim + 1,
    n_receiver_types=lambda ndim: ndim + 1,
    output_names=elastic_vti_output_names,
    auxiliary_state_names=elastic_auxiliary_state_names,
    public_wavefield_names=elastic_public_wavefield_names,
    c_model_flag_indices=_c_model_flag_indices,
    spatial_parallel=True,
)


def thomsen_to_cij(
    vp: torch.Tensor,
    vs: torch.Tensor,
    rho: torch.Tensor,
    epsilon: torch.Tensor,
    delta: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """Converts Thomsen (1986) parameters to VTI stiffnesses.

    Uses the exact (not weak-anisotropy linearised) relations. With the
    symmetry axis as the Voigt "3" direction::

        c33 = rho * vp**2
        c55 = rho * vs**2
        c11 = c33 * (1 + 2 * epsilon)
        c66 = c55 * (1 + 2 * gamma)
        c13 = sqrt(2 * delta * c33 * (c33 - c55) + (c33 - c55)**2) - c55
        buoyancy = 1 / rho

    where ``vp`` and ``vs`` are the P and S wave speeds *along the symmetry
    axis*. The ``c13`` expression inverts Thomsen's exact definition
    ``delta = ((c13 + c55)**2 - (c33 - c55)**2) / (2 c33 (c33 - c55))``,
    taking the positive square-root branch, which corresponds to
    ``c13 + c55 > 0``. This branch holds for virtually all real materials;
    "anomalous" media with ``c13 + c55 < 0`` cannot be represented in Thomsen
    parameters unambiguously and are not supported. The argument of the
    square root is non-negative iff
    ``delta >= -(c33 - c55) / (2 c33)``; more negative ``delta`` values are
    rejected with an error. Cells with ``rho = 0`` (or ``vp = vs = 0``), the
    vacuum convention of the elastic propagators, map to all-zero
    stiffnesses (the square root is masked there so no NaN gradients are
    produced) and to ``buoyancy = 0`` exactly, which is the vacuum sentinel
    the propagators themselves expect.

    The buoyancy (the reciprocal of the density) is returned alongside the
    stiffnesses because it, not ``rho``, is what
    :func:`deepwave.elastic_vti` and
    :func:`deepwave.acoustic_vti` take, so the returned tuple can be passed
    straight to them. It is computed with the same zero-guarded reciprocal as
    :func:`deepwave.vpvsrho_to_lambmubuoyancy`, so a vacuum cell yields
    exactly ``0`` rather than ``inf``.

    The function is differentiable, so gradients with respect to the Thomsen
    parameters may be obtained by chaining autograd through it (inverting in
    Thomsen space).

    Args:
        vp: P-wave speed along the symmetry axis.
        vs: S-wave speed along the symmetry axis.
        rho: Density.
        epsilon: Thomsen's epsilon, ``(c11 - c33) / (2 c33)``.
        delta: Thomsen's delta (exact definition, see above).
        gamma: Thomsen's gamma, ``(c66 - c55) / (2 c55)``. Required to build
            ``c66``, which is only used in 3-D. Optional, default None.

    Returns:
        Tuple[torch.Tensor, ...]: ``(c11, c13, c33, c55, c66, buoyancy)``,
        with ``c66`` set to ``None`` when ``gamma`` is ``None``. In 2-D,
        where ``c66`` does not enter the equations, unpack it into ``_``;
        in 3-D pass it to the propagator's ``c66`` keyword argument.

    Raises:
        DeepwaveValueError: If ``vp``, ``vs`` or ``rho`` contains negative
            values, if ``c33 < c55`` (``vp < vs``) anywhere, or if ``delta``
            is more negative than ``-(c33 - c55) / (2 c33)`` anywhere.

    """
    if bool((rho < 0).any()) or bool((vp < 0).any()) or bool((vs < 0).any()):
        raise DeepwaveValueError(
            "vp, vs, and rho must be non-negative, but their minimum values "
            f"are {float(vp.min())}, {float(vs.min())} and "
            f"{float(rho.min())}. Vacuum cells are represented by "
            "vp = vs = rho = 0, not by negative values."
        )
    c33 = rho * vp**2
    c55 = rho * vs**2
    if bool((c33 < c55).any()):
        raise DeepwaveValueError(
            "thomsen_to_cij requires c33 >= c55 (vp >= vs along the symmetry "
            f"axis), but the largest c55 - c33 is {float((c55 - c33).max())}."
        )
    c11 = c33 * (1 + 2 * epsilon)
    dc = c33 - c55
    arg = 2 * delta * c33 * dc + dc**2
    if bool((arg < 0).any()):
        raise DeepwaveValueError(
            "The argument of the square root in the exact c13 relation, "
            "2*delta*c33*(c33-c55) + (c33-c55)**2, is negative for at least "
            "one cell. This requires delta >= -(c33-c55)/(2*c33); more "
            "negative delta values do not correspond to a real c13 with "
            "c13 + c55 > 0 and are not supported. The most negative value "
            f"of that argument is {float(arg.min())}."
        )
    # Mask the sqrt where its argument is zero (e.g. vacuum cells with
    # vp = vs = rho = 0) so that backpropagation does not produce NaN there
    # (the derivative of sqrt at 0 is infinite); those cells get c13 = -c55
    # (= 0 in vacuum) with zero gradient.
    positive = arg > 0
    arg_safe = torch.where(positive, arg, torch.ones_like(arg))
    c13 = torch.where(positive, torch.sqrt(arg_safe), torch.zeros_like(arg)) - c55
    c66 = c55 * (1 + 2 * gamma) if gamma is not None else None
    # The same zero-guarded reciprocal as vpvsrho_to_lambmubuoyancy, so that a
    # vacuum cell (rho == 0) maps to buoyancy == 0 exactly.
    buoyancy = _common.reciprocal_vacuum_safe(rho)
    return c11, c13, c33, c55, c66, buoyancy


def cij_to_thomsen(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55: torch.Tensor,
    c66: Optional[torch.Tensor],
    buoyancy: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Converts VTI stiffnesses to Thomsen (1986) parameters.

    The exact inverse of :func:`deepwave.anisotropy.thomsen_to_cij` (not the
    weak-anisotropy linearised approximation). With the symmetry axis as the Voigt "3"
    direction, this inverts the relations documented in
    :func:`deepwave.anisotropy.thomsen_to_cij`::

        rho = 1 / buoyancy
        vp = sqrt(c33 * buoyancy)
        vs = sqrt(c55 * buoyancy)
        epsilon = (c11 - c33) / (2 * c33)
        delta = ((c13 + c55)**2 - (c33 - c55)**2) / (2 * c33 * (c33 - c55))
        gamma = (c66 - c55) / (2 * c55)

    The arguments are in the order :func:`deepwave.anisotropy.thomsen_to_cij` returns
    them, and the return values are in the order
    :func:`deepwave.anisotropy.thomsen_to_cij` takes them, so the two compose directly
    in both directions:
    ``cij_to_thomsen(*thomsen_to_cij(vp, vs, rho, epsilon, delta, gamma))``
    round-trips to the inputs (to floating-point precision) whenever
    ``vp > vs`` (so ``c33 != c55`` and ``delta`` is not degenerate) and
    ``rho > 0``, and ``thomsen_to_cij(*cij_to_thomsen(...))`` likewise
    returns the stiffnesses it was given. Note that this function takes
    ``buoyancy``, not ``rho``: that is what the propagators use, and what
    :func:`deepwave.anisotropy.thomsen_to_cij` now returns.

    A vacuum cell (``buoyancy == 0``, equivalently ``rho == 0``, the vacuum
    convention of the elastic propagators) has no recoverable Thomsen
    parameters at all: :func:`deepwave.anisotropy.thomsen_to_cij` maps every such cell
    to all-zero stiffnesses whatever its ``vp``/``vs``/``epsilon``/``delta``/
    ``gamma`` were. It is mapped here to ``vp = vs = rho = epsilon = delta =
    gamma = 0``, which is the canonical vacuum representative: feeding it
    back to :func:`deepwave.anisotropy.thomsen_to_cij` reproduces exactly the all-zero
    stiffnesses and ``buoyancy = 0`` it came from. The masking also keeps the
    gradients there finite.

    The ``vp == vs`` (isotropic, ``c33 == c55``) case is also singular --
    :func:`deepwave.anisotropy.thomsen_to_cij` maps every ``delta`` to the
    same ``c13`` there --
    but, unlike vacuum, it has no canonical representative to collapse to, so
    it is deliberately *not* special-cased: the formula above is evaluated
    directly and produces a division by zero (``nan``/``inf``) for such
    cells, loudly, rather than inventing a value. The same applies to
    ``gamma`` when ``c55 == 0`` (a pseudo-acoustic, ``vs = 0`` model).

    Args:
        c11: The c11 stiffness.
        c13: The c13 stiffness.
        c33: The c33 stiffness.
        c55: The c55 stiffness (shear modulus along the symmetry axis).
        c66: The c66 stiffness (horizontal-plane shear modulus), or ``None``
            if it is not available (in 2-D it does not enter the equations).
            ``gamma`` is then ``None`` too.
        buoyancy: The reciprocal of density, as the propagators take it.

    Returns:
        Tuple[torch.Tensor, ...]: ``(vp, vs, rho, epsilon, delta, gamma)``,
        with ``gamma`` set to ``None`` when ``c66`` is ``None``.

    Raises:
        DeepwaveValueError: If ``buoyancy`` contains negative values.

    """
    if bool((buoyancy < 0).any()):
        raise DeepwaveValueError(
            "buoyancy must be non-negative, but its minimum value is "
            f"{float(buoyancy.min())}. Vacuum cells are represented by "
            "buoyancy = 0, not by a negative value."
        )
    # The same zero-guarded reciprocal as lambmubuoyancy_to_vpvsrho, so that
    # the vacuum sentinel buoyancy == 0 maps to rho == 0 exactly.
    rho = _common.reciprocal_vacuum_safe(buoyancy)
    # Vacuum cells are masked out of every expression below (each of which is
    # 0/0 there) and set to zero, the canonical vacuum Thomsen parameters.
    solid = buoyancy > 0

    def _mask_vacuum(value: torch.Tensor) -> torch.Tensor:
        """``value`` where the medium is solid, 0 in vacuum."""
        return torch.where(solid, value, torch.zeros_like(value))

    def _safe(denominator: torch.Tensor) -> torch.Tensor:
        """``denominator``, with vacuum cells replaced by 1 to avoid 0/0."""
        return torch.where(solid, denominator, torch.ones_like(denominator))

    def _masked_sqrt(argument: torch.Tensor) -> torch.Tensor:
        """``sqrt(argument)``, masked where the argument is not positive.

        The derivative of ``sqrt`` at 0 is infinite, so the argument is
        replaced by 1 wherever it vanishes (in vacuum, and also for a
        pseudo-acoustic ``c55 = 0``) before the square root is taken; those
        cells get 0 with a zero gradient rather than a NaN one.
        """
        positive = argument > 0
        safe = torch.where(positive, argument, torch.ones_like(argument))
        return torch.where(positive, torch.sqrt(safe), torch.zeros_like(argument))

    vp = _masked_sqrt(c33 * buoyancy)
    vs = _masked_sqrt(c55 * buoyancy)
    c33_safe = _safe(c33)
    epsilon = _mask_vacuum((c11 - c33) / (2 * c33_safe))
    dc = c33 - c55
    delta = _mask_vacuum(((c13 + c55) ** 2 - dc**2) / (2 * c33_safe * _safe(dc)))
    gamma = _mask_vacuum((c66 - c55) / (2 * _safe(c55))) if c66 is not None else None
    return vp, vs, rho, epsilon, delta, gamma


def alkhalifah_to_thomsen(
    vp: torch.Tensor,
    vnmo: torch.Tensor,
    eta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts Alkhalifah-Tsvankin (1995) parameters to Thomsen (1986) epsilon/delta.

    P-wave surface seismic data alone cannot resolve the full Thomsen
    parameter set: time processing (NMO, DMO, time migration) in a VTI
    medium is controlled only by the *P-wave NMO velocity* and the
    *anellipticity* parameter ``eta``, not by ``epsilon``/``delta``
    individually (Alkhalifah and Tsvankin, "Velocity analysis for
    transversely isotropic media", *Geophysics* **60**, 1995, 1550-1566).
    This helper converts that P-wave-derivable parameterisation back to
    Thomsen's ``epsilon``/``delta`` using the exact relations::

        delta = (vnmo**2 / vp**2 - 1) / 2
        epsilon = eta * (1 + 2 * delta) + delta

    which invert Alkhalifah and Tsvankin's defining pair ``vnmo = vp *
    sqrt(1 + 2*delta)`` and ``eta = (epsilon - delta) / (1 + 2*delta)``.

    The function is plain differentiable PyTorch arithmetic (no in-place
    ops), so gradients with respect to ``vp``/``vnmo``/``eta`` may be
    obtained by chaining autograd through it -- e.g. inverting directly in
    Alkhalifah-Tsvankin parameter space, then continuing the chain through
    :func:`deepwave.anisotropy.thomsen_to_cij` to build a VTI stiffness model (see that
    function, and :func:`deepwave.elastic_vti`/:func:`deepwave.acoustic_vti`
    for the propagators that consume the resulting Cij).

    Args:
        vp: P-wave velocity along the symmetry axis (the vertical P-wave
            velocity).
        vnmo: The short-spread P-wave normal-moveout (NMO) velocity, the
            quantity conventional velocity analysis of surface seismic data
            actually estimates.
        eta: Alkhalifah and Tsvankin's anellipticity parameter, which
            controls P-wave time processing beyond the NMO velocity (dip
            moveout, time migration); zero for elliptical anisotropy
            (``epsilon == delta``).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(epsilon, delta)``, Thomsen's
        parameters, suitable as input to :func:`deepwave.anisotropy.thomsen_to_cij`
        (with a chosen ``vs``, since shear information is not recoverable from
        P-wave-only ``vnmo``/``eta``).

    """
    delta = (vnmo**2 / vp**2 - 1) / 2
    epsilon = eta * (1 + 2 * delta) + delta
    return epsilon, delta


def thomsen_to_alkhalifah(
    vp: torch.Tensor,
    epsilon: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts Thomsen (1986) epsilon/delta to Alkhalifah-Tsvankin (1995) parameters.

    The exact inverse of :func:`alkhalifah_to_thomsen`, using Alkhalifah and
    Tsvankin's defining pair directly::

        vnmo = vp * sqrt(1 + 2 * delta)
        eta = (epsilon - delta) / (1 + 2 * delta)

    so ``alkhalifah_to_thomsen(vp, *thomsen_to_alkhalifah(vp, epsilon,
    delta))`` returns ``(epsilon, delta)`` and ``thomsen_to_alkhalifah(vp,
    *alkhalifah_to_thomsen(vp, vnmo, eta))`` returns ``(vnmo, eta)``, both to
    floating-point precision. It is the conversion to use when a model is
    held in Thomsen parameters but the *P-wave-derivable* quantities are
    wanted -- to compare with, or initialise from, a conventional
    NMO-velocity/anellipticity model, or to invert in the parameterisation
    that surface P-wave data actually constrains (see
    :func:`alkhalifah_to_thomsen` for why that is the NMO velocity and
    ``eta``, not ``epsilon`` and ``delta`` individually).

    The function is plain differentiable PyTorch arithmetic (no in-place
    ops), so gradients with respect to ``vp``/``epsilon``/``delta`` may be
    obtained by chaining autograd through it.

    Args:
        vp: P-wave velocity along the symmetry axis (the vertical P-wave
            velocity).
        epsilon: Thomsen's epsilon, ``(c11 - c33) / (2 c33)``.
        delta: Thomsen's delta (exact definition, see
            :func:`deepwave.anisotropy.thomsen_to_cij`). Must be greater than ``-0.5``
            everywhere, since ``1 + 2 delta`` is the squared ratio of the NMO
            to the vertical velocity.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(vnmo, eta)``, the short-spread
        P-wave normal-moveout velocity and the anellipticity parameter,
        suitable as input to :func:`alkhalifah_to_thomsen`.

    Raises:
        DeepwaveValueError: If ``delta <= -0.5`` anywhere, which would make
            ``1 + 2 delta`` non-positive and ``vnmo`` imaginary.

    References:
        T. Alkhalifah, I. Tsvankin, "Velocity analysis for transversely
        isotropic media," Geophysics, vol. 60, no. 5, 1995.

    """
    moveout = 1 + 2 * delta
    if bool((moveout <= 0).any()):
        raise DeepwaveValueError(
            "thomsen_to_alkhalifah requires delta > -0.5, so that "
            "1 + 2*delta (the squared ratio of the NMO to the vertical "
            "velocity) is positive, but the minimum value of delta is "
            f"{float(delta.min())}."
        )
    vnmo = vp * torch.sqrt(moveout)
    eta = (epsilon - delta) / moveout
    return vnmo, eta


def prepare_parameters(
    c55: torch.Tensor,
    c66: Optional[torch.Tensor],
    buoyancy: torch.Tensor,
) -> list[torch.Tensor]:
    """Prepares elastic VTI properties for the free surface method.

    Applies the Zeng et al. (2012) improved-vacuum averaging to the VTI
    moduli by mapping each stiffness to the isotropic elastic parameter it
    generalises and reusing the isotropic scheme
    (``elastic.prepare_parameters``):

    - ``c55`` generalises the shear modulus ``mu`` for the shear stresses
      involving the symmetry axis (``sigma_xz``/``sigma_yz`` in 3-D, the
      single shear stress in 2-D), so it is harmonically averaged (with
      vacuum masking) onto those staggered shear-stress nodes exactly as
      ``mu`` is.
    - ``c66`` generalises ``mu`` for the horizontal-plane shear stress
      ``sigma_xy`` (3-D only), so it is harmonically averaged onto that node
      in the same way.
    - The buoyancy is (as in the isotropic propagator) the inverse of the
      arithmetic mean of the density, evaluated at each staggered velocity
      node, with vacuum masking.

    The normal-stress-node stiffnesses (``c11``, ``c12``, ``c13``, ``c33``)
    are used unaveraged at the nodes where the user supplies them, exactly as
    ``lambda`` (and the ``lambda + 2 mu`` combination) are in the isotropic
    propagator.

    Args:
        c55: The c55 stiffness (shear modulus along the symmetry axis).
        c66: **Required in 3-D; must be None in 2-D** (it does not enter the
            2-D P-SV equations). The c66 stiffness (horizontal-plane shear
            modulus).
        buoyancy: Buoyancy (1/rho).

    Returns:
        List[torch.Tensor]: ``[c55_zy, c55_zx, c66_yx, buoyancy_z,
        buoyancy_y, buoyancy_x]`` in 3-D or ``[c55_yx, buoyancy_y,
        buoyancy_x]`` in 2-D.

    """
    ndim = c55.ndim - 1
    if ndim == 3:
        if c66 is None:
            raise DeepwaveValueError("elastic_vti: c66 is required in 3-D.")
        # [c55_zy, c55_zx, (c55_yx unused), buoyancy_z, buoyancy_y, buoyancy_x]
        iso_c55 = _prepare_isotropic_parameters(c55, buoyancy)
        # Only the yx (sigma_xy node) average of c66 is needed.
        c66_yx = _prepare_isotropic_parameters(c66, buoyancy)[2]
        return [iso_c55[0], iso_c55[1], c66_yx, iso_c55[3], iso_c55[4], iso_c55[5]]
    # 2-D: [c55_yx, buoyancy_y, buoyancy_x]
    return _prepare_isotropic_parameters(c55, buoyancy)


class ElasticVti(torch.nn.Module):
    """A PyTorch Module for elastic VTI wave propagation.

    Stability: experimental (see :doc:`/stability`).

    This Module stores the VTI stiffnesses (``c11``, ``c13``, ``c33``,
    ``c55``, and, for 3-D, ``c66``) and the buoyancy as
    ``torch.nn.Parameter`` objects, allowing gradients to be computed with
    respect to them if desired.

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
        c55: The c55 stiffness model (shear modulus for planes containing
            the symmetry axis), of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        buoyancy: The buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. It is of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            (for isotropic grids) or a sequence of numbers.
        c66: **Required in 3-D; must be None in 2-D** (it does not enter
            the 2-D P-SV equations). The c66 stiffness model
            (horizontal-plane shear modulus), of shape ``[*model_shape]``
            or, for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
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
        c55_requires_grad: A bool specifying whether gradients will be
            computed for `c55`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        buoyancy_requires_grad: A bool specifying whether gradients will be
            computed for `buoyancy`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        c66_requires_grad: A bool specifying whether gradients will be
            computed for `c66`. Unset by default, which means False. Cannot
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
        c11: torch.Tensor,
        c13: torch.Tensor,
        c33: torch.Tensor,
        c55: torch.Tensor,
        buoyancy: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        c66: Optional[torch.Tensor] = None,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        c11_requires_grad: Optional[bool] = None,
        c13_requires_grad: Optional[bool] = None,
        c33_requires_grad: Optional[bool] = None,
        c55_requires_grad: Optional[bool] = None,
        buoyancy_requires_grad: Optional[bool] = None,
        c66_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        owned = ["c11", "c13", "c33", "c55", "buoyancy", "c66"]
        supplied = [name for name in owned if name != "c66" or c66 is not None]
        grad = _common.resolve_requires_grad(
            requires_grad,
            owned,
            supplied,
            {
                "c11": c11_requires_grad,
                "c13": c13_requires_grad,
                "c33": c33_requires_grad,
                "c55": c55_requires_grad,
                "buoyancy": buoyancy_requires_grad,
                "c66": c66_requires_grad,
            },
            "ElasticVti",
        )
        self.c11 = _common.make_model_parameter(c11, grad["c11"], "c11")
        self.c13 = _common.make_model_parameter(c13, grad["c13"], "c13")
        self.c33 = _common.make_model_parameter(c33, grad["c33"], "c33")
        self.c55 = _common.make_model_parameter(c55, grad["c55"], "c55")
        # ``buoyancy`` is registered before ``c66`` so that the parameters,
        # and therefore ``state_dict()``, are in the order
        # ``deepwave.spec("elastic_vti").model_names(ndim)`` reports.
        self.buoyancy = _common.make_model_parameter(
            buoyancy, grad["buoyancy"], "buoyancy"
        )
        self.c66: Optional[torch.Tensor] = (
            None
            if c66 is None
            else _common.make_model_parameter(c66, grad["c66"], "c66")
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
        source_locations_pressure_rate: Optional[LocationsLike] = None,
        source_locations_z: Optional[LocationsLike] = None,
        source_locations_y: Optional[LocationsLike] = None,
        source_locations_x: Optional[LocationsLike] = None,
        source_mask_pressure_rate: Optional[torch.Tensor] = None,
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
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        sigmazz_0: Optional[torch.Tensor] = None,
        sigmayz_0: Optional[torch.Tensor] = None,
        sigmaxz_0: Optional[torch.Tensor] = None,
        sigmayy_0: Optional[torch.Tensor] = None,
        sigmaxy_0: Optional[torch.Tensor] = None,
        sigmaxx_0: Optional[torch.Tensor] = None,
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

        The inputs are the same as for :func:`deepwave.elastic_vti` except that the
        stiffnesses, `buoyancy`, `grid_spacing`, and the storage settings
        (`storage_mode`/`storage_path`/`storage_compression`) do not need to
        be provided again -- they are supplied from the stored module state.
        See :func:`deepwave.elastic_vti` for a description of the inputs and outputs.
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
        return elastic_vti(
            self.c11,
            self.c13,
            self.c33,
            self.c55,
            self.buoyancy,
            self.grid_spacing,
            dt,
            c66=self.c66,
            source_mask_pressure_rate=source_mask_pressure_rate,
            source_mask_z=source_mask_z,
            source_mask_y=source_mask_y,
            source_mask_x=source_mask_x,
            receiver_mask_p=receiver_mask_p,
            receiver_mask_z=receiver_mask_z,
            receiver_mask_y=receiver_mask_y,
            receiver_mask_x=receiver_mask_x,
            sources=sources,
            receivers=receivers,
            source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_pressure_rate=source_locations_pressure_rate,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            receiver_locations_p=receiver_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            sigmazz_0=sigmazz_0,
            sigmayz_0=sigmayz_0,
            sigmaxz_0=sigmaxz_0,
            sigmayy_0=sigmayy_0,
            sigmaxy_0=sigmaxy_0,
            sigmaxx_0=sigmaxx_0,
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


def elastic_vti(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    # Required in 3-D, must be None in 2-D -- see the c66 entry below.
    c66: Optional[torch.Tensor] = None,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes_pressure_rate: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_pressure_rate: Optional[LocationsLike] = None,
    source_locations_z: Optional[LocationsLike] = None,
    source_locations_y: Optional[LocationsLike] = None,
    source_locations_x: Optional[LocationsLike] = None,
    source_mask_pressure_rate: Optional[torch.Tensor] = None,
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
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigmazz_0: Optional[torch.Tensor] = None,
    sigmayz_0: Optional[torch.Tensor] = None,
    sigmaxz_0: Optional[torch.Tensor] = None,
    sigmayy_0: Optional[torch.Tensor] = None,
    sigmaxy_0: Optional[torch.Tensor] = None,
    sigmaxx_0: Optional[torch.Tensor] = None,
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
    """Elastic VTI wave propagation (functional interface).

    Stability: experimental (see :doc:`/stability`).

    This function performs forward modelling with the elastic wave equation
    in a vertically transverse isotropic (VTI) medium. The outputs are
    differentiable with respect to the input float torch.Tensors (including
    every stiffness and the buoyancy).

    The propagator uses the same staggered grid, C-PML/M-PML absorbing
    boundaries, improved-vacuum free surface, source/receiver types, and
    input/output conventions as :func:`deepwave.elastic`; only the medium
    description differs. See :func:`deepwave.elastic` for a full description
    of the shared arguments and of the returned wavefields and receiver
    amplitudes. The isotropic propagator is recovered exactly by setting
    ``c11 = c33 = lamb + 2 mu``, ``c13 = lamb``, ``c55 = mu`` (and
    ``c66 = mu`` in 3-D). ``c66`` is the one model argument whose necessity
    depends on the dimensionality: it is **required in 3-D and must be None
    in 2-D**; every other model is required in both.

    The symmetry axis of the medium is the slowest spatial dimension: ``z``
    in 3-D, and ``y`` in a 2-D (P-SV in the (y, x) plane) simulation. There
    is no 1-D VTI propagator: with a single spatial dimension only ``c33``
    survives in the constitutive relation, which is the isotropic 1-D
    elastic equation, so 1-D inputs raise an error (use
    :func:`deepwave.elastic`).

    All stiffnesses are specified at the normal-stress (integer grid) nodes,
    with units of pressure (e.g. Pa if working in SI units); ``buoyancy`` is
    1/density. The shear stiffnesses ``c55``/``c66`` are internally averaged
    onto the staggered shear-stress nodes, and the buoyancy onto the
    staggered velocity nodes, exactly as ``mu`` and ``buoyancy`` are in the
    isotropic propagator (see ``prepare_parameters``), which also
    implements the free surface for vacuum cells. In 3-D,
    ``c12 = c11 - 2 c66`` is derived internally. Readable backgrounds are
    checked for finite values and positive elastic energy: in active cells
    ``c11 > 0``, ``c33 > 0`` and nonnegative shear stiffnesses are required.
    The normal block must satisfy ``c13**2 < c33 * (c11 - c66)`` in 3-D
    or ``c13**2 < c11 * c33`` in 2-D for positive shear stiffnesses.
    Supported zero-shear fluid limits permit a semidefinite normal block.
    Vacuum has zero buoyancy and zero stiffness. See :doc:`materials` for
    the complete domain and tensor-transform preconditions.

    When ``max_vel`` is omitted, setup chooses a conservative speed bound
    from the constitutive tensor and staggered density averaging. It accounts
    for heterogeneous coefficient contrast as well as directional wave speeds;
    a pointwise phase-speed maximum alone need not bound the discrete operator.
    An explicit ``max_vel`` must cover this bound. See :doc:`discretisation`.

    Use :func:`deepwave.anisotropy.thomsen_to_cij` to build the stiffnesses from Thomsen
    parameters; because it is differentiable, inversion may be performed in
    Thomsen parameter space.

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
        c55: The c55 stiffness model (shear modulus for planes containing
            the symmetry axis), of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c66: **Required in 3-D; must be None in 2-D** (it does not enter
            the 2-D P-SV equations). The c66 stiffness model
            (horizontal-plane shear modulus), of shape ``[*model_shape]``
            or, for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See
            :ref:`batched-models`. This is the one model whose
            necessity depends on the number of dimensions.
        buoyancy: The buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. Together with zero
            stiffnesses it creates a free surface, as in
            :func:`deepwave.elastic`. It is of shape ``[*model_shape]``
            or, for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dt: The time step interval of the input and output. See
            :func:`deepwave.elastic` for the unit-consistency requirement
            (here applying to the stiffnesses, ``buoyancy``,
            ``grid_spacing``, and ``dt``).
        source_amplitudes_z: See :func:`deepwave.elastic`. This channel exists
            only in 3-D.
        source_amplitudes_y: See :func:`deepwave.elastic`.
        source_amplitudes_x: See :func:`deepwave.elastic`.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``pressure_rate``, ``z``,
            ``y``, ``x`` (``z`` only in 3-D); these are equivalent to the flat
            ``source_amplitudes_*``/``source_locations_*`` keywords, which
            remain permanently supported; giving one channel in both forms
            raises. Optional.
        receivers: This simulation's receivers, keyed by channel, as the
            canonical alternative to the flat ``receiver_locations*`` keywords.
            Each value is a locations object, or ``(locations, mask)`` to
            include an activity mask. The channels are ``p``, ``z``, ``y``,
            ``x`` (``z`` only in 3-D). Giving one channel in both forms raises.
            Optional.
        source_amplitudes_pressure_rate: See :func:`deepwave.elastic`.
        source_locations_z: See :func:`deepwave.elastic`. This channel exists
            only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_y: See :func:`deepwave.elastic`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_x: See :func:`deepwave.elastic`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_pressure_rate: See :func:`deepwave.elastic`.
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
        receiver_locations_z: See :func:`deepwave.elastic`. This channel exists
            only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_y: See :func:`deepwave.elastic`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_x: See :func:`deepwave.elastic`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_p: See :func:`deepwave.elastic`. In 2-D the
            recorded pressure is the in-plane mean normal stress
            ``-(sigma_xx + sigma_yy)/2``, which under plane strain is not
            the thermodynamic pressure; see :func:`deepwave.elastic`'s
            ``receiver_amplitudes_p`` entry for the full statement.
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
        accuracy: See :func:`deepwave.elastic`. Default 4.
        pml_width: See :func:`deepwave.elastic`. Default 20.
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: See :func:`deepwave.elastic`. The same per-field
            free/rigid/periodic guard-ring image method;
            the physics/citation reasoning is unchanged by VTI's stiffness
            (see ``docs/boundaries.rst``'s Policy section and
            ``_boundaries.py``'s ``fill_elastic_stress_boundary``/
            ``fill_elastic_velocity_boundary`` module comment).
            Availability: `"pml"` is accepted on every face at every accuracy,
            and `"rigid"` on the beginning of a dimension at every accuracy
            and on the end of a dimension at `accuracy=2`. `"rigid"` on the
            end of a dimension with `accuracy >= 4`, and `"free"` and
            `"periodic"` on any face, raise
            :class:`deepwave.DeepwaveNotImplementedError`: those guard-ring
            fills do not conserve a discrete energy, so the solution can grow
            without bound. Model a free surface with a vacuum layer (the
            improved vacuum method) instead; put a `"rigid"` face at the
            beginning of the dimension (flip the model along that axis), or
            use `accuracy=2`. Faces may be accepted in a later release; none
            is removed.
        pml_freq: See :func:`deepwave.elastic`.
        mpml_fraction: The multiaxial PML (M-PML) cross-damping fraction. See
            :func:`deepwave.elastic`. Anisotropic media are the primary use
            case for M-PML: strong anisotropy is a known trigger of C-PML
            instability, and a small positive ``mpml_fraction`` (0.15-0.2) cures
            it at the cost of slightly larger boundary reflections. Default
            0.0 (standard C-PML, byte-identical to not having the feature).
        max_vel: See :func:`deepwave.elastic`, and the note above about the
            default bound computed from the stiffnesses.
        courant_number: See :func:`deepwave.elastic`.
        survey_pad: See :func:`deepwave.scalar`.
        vz_0: See :func:`deepwave.elastic` (and similarly for all the other
            initial wavefields and PML memory variables).
        vy_0: See :func:`deepwave.elastic`.
        vx_0: See :func:`deepwave.elastic`.
        sigmazz_0: See :func:`deepwave.elastic`.
        sigmayz_0: See :func:`deepwave.elastic`.
        sigmaxz_0: See :func:`deepwave.elastic`.
        sigmayy_0: See :func:`deepwave.elastic`.
        sigmaxy_0: See :func:`deepwave.elastic`.
        sigmaxx_0: See :func:`deepwave.elastic`.
        initial_state: See :func:`deepwave.elastic`. The individual
            absorbing-boundary field names (`m_vzz_0`, ..., `m_sigmaxxx_0`)
            are no longer accepted as keys; pass `auxiliary_state` instead.
        origin: See :func:`deepwave.elastic`.
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
        order, are the same names as :func:`deepwave.elastic`'s Returns
        (velocities, then stresses, each grouped by kind and ordered
        slowest-to-fastest dimension, then the auxiliary state, followed by the
        receiver amplitudes: pressure, then the velocity components):

            - vz_0: Final velocity wavefield in the z-dimension (if ndim == 3).
            - vy_0: Final velocity wavefield in the y-dimension.
            - vx_0: Final velocity wavefield in the x-dimension.
            - sigmazz_0: Final stress wavefield (zz component) (if ndim == 3).
            - sigmayz_0: Final stress wavefield (yz component) (if ndim == 3).
            - sigmaxz_0: Final stress wavefield (xz component) (if ndim == 3).
            - sigmayy_0: Final stress wavefield (yy component).
            - sigmaxy_0: Final stress wavefield (xy component).
            - sigmaxx_0: Final stress wavefield (xx component).
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables. Opaque: pass it back
              (as part of `initial_state`) to continue the simulation; its
              contents are implementation state rather than physics.
            - receiver_amplitudes_p: Recorded pressure data. In 2-D the
              recorded pressure is the in-plane mean normal stress
              ``-(sigma_xx + sigma_yy)/2``, which under plane strain is not
              the thermodynamic pressure; see :func:`deepwave.elastic`'s
              ``receiver_amplitudes_p`` entry for the full statement.
            - receiver_amplitudes_z: Recorded z-component velocity data
              (if ndim == 3).
            - receiver_amplitudes_y: Recorded y-component velocity data.
            - receiver_amplitudes_x: Recorded x-component velocity data.

        This positional order is easy to index incorrectly (``out[-1]`` is
        the x-component velocity receiver amplitudes, not the pressure
        data); prefer accessing the fields by name instead.

    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            elastic_vti,
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
            "receiver_locations_p": receiver_locations_p,
            "receiver_mask_p": receiver_mask_p,
            "receiver_locations_z": receiver_locations_z,
            "receiver_mask_z": receiver_mask_z,
            "receiver_locations_y": receiver_locations_y,
            "receiver_mask_y": receiver_mask_y,
            "receiver_locations_x": receiver_locations_x,
            "receiver_mask_x": receiver_mask_x,
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
    _common.check_models(
        [c11, c13, c33, c55, buoyancy],
        ["c11", "c13", "c33", "c55", "buoyancy"],
        "elastic_vti",
    )
    if c66 is not None:
        _common.check_models(
            [c11, c13, c33, c55, buoyancy, c66],
            ["c11", "c13", "c33", "c55", "buoyancy", "c66"],
            "elastic_vti",
        )
    if c66 is not None or c11.ndim == 2:
        material = {
            "c11": c11,
            "c13": c13,
            "c33": c33,
            "c55": c55,
            "buoyancy": buoyancy,
        }
        if c66 is not None:
            material["c66"] = c66
        _common.check_material_domain(material, "elastic_vti")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend
    # Fill any initial wavefields supplied via the optional initial_state
    # bundle, leaving explicit wavefield arguments untouched.
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
            "sigmazz_0": sigmazz_0,
            "sigmayz_0": sigmayz_0,
            "sigmaxz_0": sigmaxz_0,
            "sigmayy_0": sigmayy_0,
            "sigmaxy_0": sigmaxy_0,
            "sigmaxx_0": sigmaxx_0,
            # m_* are the absorbing boundary's memory variables, not physical
            # fields, so they are not exposed as arguments; they travel in
            # `auxiliary_state`.
            "m_vzz_0": None,
            "m_vzy_0": None,
            "m_vzx_0": None,
            "m_vyz_0": None,
            "m_vxz_0": None,
            "m_vyy_0": None,
            "m_vyx_0": None,
            "m_vxy_0": None,
            "m_vxx_0": None,
            "m_sigmazzz_0": None,
            "m_sigmayzy_0": None,
            "m_sigmaxzx_0": None,
            "m_sigmayzz_0": None,
            "m_sigmaxzz_0": None,
            "m_sigmayyy_0": None,
            "m_sigmaxyy_0": None,
            "m_sigmaxyx_0": None,
            "m_sigmaxxx_0": None,
        },
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=elastic_auxiliary_state_names(3),
        propagator=SPEC.name,
    )
    (
        vz_0,
        vy_0,
        vx_0,
        sigmazz_0,
        sigmayz_0,
        sigmaxz_0,
        sigmayy_0,
        sigmaxy_0,
        sigmaxx_0,
        m_vzz_0,
        m_vzy_0,
        m_vzx_0,
        m_vyz_0,
        m_vxz_0,
        m_vyy_0,
        m_vyx_0,
        m_vxy_0,
        m_vxx_0,
        m_sigmazzz_0,
        m_sigmayzy_0,
        m_sigmaxzx_0,
        m_sigmayzz_0,
        m_sigmaxzz_0,
        m_sigmayyy_0,
        m_sigmaxyy_0,
        m_sigmaxyx_0,
        m_sigmaxxx_0,
    ) = _resolved_wavefields.values()

    _common.check_inputs_not_vmapped(
        source_amplitudes_pressure_rate,
        source_amplitudes_z,
        source_amplitudes_y,
        source_amplitudes_x,
        source_locations_z,
        source_locations_y,
        source_locations_x,
        source_locations_pressure_rate,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
        receiver_locations_p,
        vz_0,
        sigmazz_0,
        sigmayz_0,
        sigmaxz_0,
        m_vzz_0,
        m_vzy_0,
        m_vzx_0,
        m_vyz_0,
        m_vxz_0,
        m_sigmazzz_0,
        m_sigmayzy_0,
        m_sigmaxzx_0,
        m_sigmayzz_0,
        m_sigmaxzz_0,
        vy_0,
        sigmayy_0,
        sigmaxy_0,
        m_vyy_0,
        m_vyx_0,
        m_vxy_0,
        m_sigmayyy_0,
        m_sigmaxyy_0,
        m_sigmaxyx_0,
        vx_0,
        m_vxx_0,
        m_sigmaxxx_0,
    )
    raw_models: list[torch.Tensor] = [c11, c13, c33, c55]
    raw_model_names: list[str] = ["c11", "c13", "c33", "c55"]
    if c66 is not None:
        raw_models.append(c66)
        raw_model_names.append("c66")
    raw_models.append(buoyancy)
    raw_model_names.append("buoyancy")
    ndim = _common.get_ndim(
        raw_models,
        [],
        [
            source_locations_z,
            source_locations_y,
            source_locations_x,
            source_locations_pressure_rate,
            receiver_locations_z,
            receiver_locations_y,
            receiver_locations_x,
            receiver_locations_p,
        ],
        [
            vz_0,
            sigmazz_0,
            sigmayz_0,
            sigmaxz_0,
            m_vzz_0,
            m_vzy_0,
            m_vzx_0,
            m_vyz_0,
            m_vxz_0,
            m_sigmazzz_0,
            m_sigmayzy_0,
            m_sigmaxzx_0,
            m_sigmayzz_0,
            m_sigmaxzz_0,
        ],
        [
            vy_0,
            sigmayy_0,
            sigmaxy_0,
            m_vyy_0,
            m_vyx_0,
            m_vxy_0,
            m_sigmayyy_0,
            m_sigmaxyy_0,
            m_sigmaxyx_0,
        ],
        [vx_0, m_vxx_0, m_sigmaxxx_0],
    )
    if ndim < 2:
        raise DeepwaveValueError(
            "elastic_vti: at least 2 spatial dimensions are required. In 1-D "
            "the VTI constitutive relation degenerates to the c33-only "
            "isotropic elastic equation (no anisotropy is expressible), so "
            "a 1-D VTI propagator would add nothing over deepwave.elastic; "
            "use that instead."
        )
    if ndim == 2 and c66 is not None:
        raise DeepwaveValueError(
            "elastic_vti: c66 must be None in 2-D: it only enters the "
            "constitutive relation through the horizontal-plane shear stress "
            "sigma_xy (and through c12 = c11 - 2*c66), neither of which "
            "exists in the 2-D P-SV equations. The models have 2 spatial "
            "dimensions, so drop the c66 argument."
        )
    if ndim == 3 and c66 is None:
        raise DeepwaveValueError(
            "elastic_vti: c66 is required in 3-D: it carries the "
            "horizontal-plane shear stress sigma_xy, and c12 = c11 - 2*c66. "
            "The models have 3 spatial dimensions, so pass c66=<the "
            "horizontal-plane shear modulus> (c66 = mu recovers isotropy, and "
            "deepwave.anisotropy.thomsen_to_cij returns it when given "
            "gamma)."
        )
    if ndim < 3 and (
        source_locations_z is not None or receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            "elastic_vti: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "z dimension should not be provided."
        )

    if not 0 <= mpml_fraction < 1:
        raise DeepwaveValueError(
            "elastic_vti: mpml_fraction (the multiaxial-PML cross-damping fraction) "
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
        initial_wavefields.extend(
            [
                vz_0,
                sigmazz_0,
                sigmayz_0,
                sigmaxz_0,
                m_vzz_0,
                m_vzy_0,
                m_vzx_0,
                m_vyz_0,
                m_vxz_0,
                m_sigmazzz_0,
                m_sigmayzy_0,
                m_sigmaxzx_0,
                m_sigmayzz_0,
                m_sigmaxzz_0,
            ]
        )
    source_amplitudes.append(source_amplitudes_y)
    source_locations_in.append(source_locations_y)
    receiver_locations_in.append(receiver_locations_y)
    initial_wavefields.extend(
        [
            vy_0,
            sigmayy_0,
            sigmaxy_0,
            m_vyy_0,
            m_vyx_0,
            m_vxy_0,
            m_sigmayyy_0,
            m_sigmaxyy_0,
            m_sigmaxyx_0,
        ]
    )
    source_amplitudes.append(source_amplitudes_x)
    source_locations_in.append(source_locations_x)
    receiver_locations_in.append(receiver_locations_x)
    initial_wavefields.extend(
        [
            vx_0,
            sigmaxx_0,
            m_vxx_0,
            m_sigmaxxx_0,
        ]
    )
    source_amplitudes.append(source_amplitudes_pressure_rate)
    source_locations_in.append(source_locations_pressure_rate)
    receiver_locations_in.append(receiver_locations_p)

    # Resolve off-grid / directional (Hicks) source and receiver
    # specifications per component, exactly as in the isotropic elastic
    # propagator (the staggered-grid component offsets are identical).
    component_offsets: list[Optional[list[float]]] = [
        *_common.velocity_component_offsets(ndim),
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
        "elastic_vti",
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

    # A velocity source/receiver must not sit on the final sample of its
    # staggered dimension.
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
                    f"elastic_vti: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{c11.shape[-ndim + dim] - 1}.",
                )

    if not all(_common.values_readable(prop) for prop in raw_models):
        if max_vel is None:
            raise _common.max_vel_unreadable_error("elastic_vti")
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        max_model_vel, min_nonzero_model_vel = _velocity_extremes_vti(
            c11, c13, c33, c55, c66, buoyancy
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
        raw_model_names,
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
    del c11, c13, c33, c55, c66, buoyancy, raw_models
    del source_amplitudes, source_locations, receiver_locations
    del initial_wavefields

    # Build the backend model list from the padded user models. In 2-D it is
    # c11, c13, c33 followed by c55_yx and the buoyancies; in 3-D it is c11,
    # c12, c13, c33 followed by c55_zy, c55_zx, c66_yx and the buoyancies.
    if ndim == 3:
        c11_p, c13_p, c33_p, c55_p, c66_p, buoyancy_p = models
        c12_p = c11_p - 2 * c66_p
        models = [c11_p, c12_p, c13_p, c33_p]
        models.extend(prepare_parameters(c55_p, c66_p, buoyancy_p))
        del c11_p, c12_p, c13_p, c33_p, c55_p, c66_p, buoyancy_p
    else:
        c11_p, c13_p, c33_p, c55_p, buoyancy_p = models
        models = [c11_p, c13_p, c33_p]
        models.extend(prepare_parameters(c55_p, None, buoyancy_p))
        del c11_p, c13_p, c33_p, c55_p, buoyancy_p

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "elastic_vti")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "elastic_vti")
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

    # Pre-scale the source amplitudes exactly as the isotropic elastic
    # propagator does: velocity (force-density) sources by the buoyancy at
    # the source location times dt, pressure sources by -dt.
    model_shape = models[0].shape[-ndim:]
    # The z force sources scale by buoyancy_z (the third-from-last model).
    if ndim == 3 and source_amplitudes_out[-4].numel() > 0:
        source_amplitudes_out[-4] = (
            source_amplitudes_out[-4]
            * _common.gather_model_at_sources(
                models[-3], sources_i[-4], model_shape, n_shots
            )
            * dt
        )
    # The y force sources scale by buoyancy_y (the second-from-last model).
    if source_amplitudes_out[-3].numel() > 0:
        source_amplitudes_out[-3] = (
            source_amplitudes_out[-3]
            * _common.gather_model_at_sources(
                models[-2], sources_i[-3], model_shape, n_shots
            )
            * dt
        )
    # The x force sources scale by buoyancy_x (the last model).
    if source_amplitudes_out[-2].numel() > 0:
        source_amplitudes_out[-2] = (
            source_amplitudes_out[-2]
            * _common.gather_model_at_sources(
                models[-1], sources_i[-2], model_shape, n_shots
            )
            * dt
        )
    # The pressure sources are scaled by -dt only.
    if source_amplitudes_out[-1].numel() > 0:
        source_amplitudes_out[-1] = -source_amplitudes_out[-1] * dt

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

    _common.check_callback_frequency(callback_frequency)
    callback_frequency = _engine.CallbackSchedule(callback_frequency, step_ratio)

    # Run the forward propagator
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
        elastic_vti_func(
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
            mpml_fraction > 0,
            *models,
            *source_amplitudes_out,
            *sources_i,
            *receivers_i,
            *wavefields,
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

    # Gather off-grid receiver data back to the original locations (see the
    # isotropic elastic propagator).
    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 1 - ndim
    )

    # Pressure is calculated as the negative arithmetic mean of normal stresses
    if outputs[-1] is not None:
        outputs[-1] = -outputs[-1] / ndim

    # Reorder the wavefields to that expected by users (grouped by kind).
    outputs[: -1 - ndim] = [outputs[i] for i in _elastic_public_wavefield_order(ndim)]

    # Receivers: pressure first, then the velocity components.
    final = (*outputs[: -1 - ndim], outputs[-1], *outputs[-1 - ndim : -1])

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


def _velocity_extremes_vti(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55: torch.Tensor,
    c66: Optional[torch.Tensor],
    buoyancy: torch.Tensor,
) -> tuple[float, float]:
    """Returns (max, min nonzero) velocity estimates for CFL/PML setup.

    The maximum is a rigorous upper bound on the VTI phase velocity obtained
    from the trace of the (positive semi-definite, for stable media)
    Christoffel matrix: ``rho v^2 <= tr(Gamma)``, where for a unit
    propagation direction the trace is at most ``max(c11, c33) + c55`` in
    2-D and ``max(c11 + c66 + c55, c33 + 2 c55)`` in 3-D. The bound is
    evaluated pointwise (times the buoyancy). Shared propagation setup then
    augments it with the heterogeneous staggered-operator bound; this
    pointwise estimate alone is not the final CFL bound.

    The minimum is the smallest nonzero of the vertical P and S speeds
    (``sqrt(c33 * buoyancy)`` and ``sqrt(c55 * buoyancy)``), used for the
    grid-dispersion (cells-per-wavelength) warning.

    Both leave as Python numbers, so neither is part of the autograd graph:
    they reshape the absorbing boundary and choose the internal time step,
    which are properties of the discretisation rather than differentiable
    functions of the model. ``.item()`` is what severs the graph; the
    ``detach()`` on the way in states that intent at the read.
    """
    del c13  # c13 does not enter the trace bound.
    c11 = c11.detach()
    c33 = c33.detach()
    c55 = c55.detach()
    c66 = None if c66 is None else c66.detach()
    buoyancy = buoyancy.detach()
    if c66 is not None:
        bound_sq = torch.maximum(c11 + c66 + c55, c33 + 2 * c55) * buoyancy
    else:
        bound_sq = (torch.maximum(c11, c33) + c55) * buoyancy
    max_model_vel = float(bound_sq.clamp(min=0).max().sqrt().item())
    vp0 = (c33 * buoyancy).clamp(min=0).sqrt()
    vs0 = (c55 * buoyancy).clamp(min=0).sqrt()
    nonzero = torch.cat([vp0[vp0 != 0].flatten(), vs0[vs0 != 0].flatten()])
    min_nonzero_model_vel = float(nonzero.min().item()) if nonzero.numel() > 0 else 0.0
    return max_model_vel, min_nonzero_model_vel


class ElasticVtiForwardFunc(torch.autograd.Function):
    """Forward propagation of the elastic VTI wave equation."""

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
        """Performs the forward propagation of the elastic VTI wave equation.

        See ``elastic.ElasticForwardFunc``; this is identical
        except for the models (VTI stiffnesses) and the storage flags that
        derive from them.

        Args:
            pml_profiles: List of PML profiles.
            grid_spacing: Grid spacing for each spatial dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            n_shots: Number of shots in the batch.
            forward_callback: The forward callback.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.
            storage_mode_str: Storage mode ("device", "cpu", "disk", "none").
            storage_path: Path for disk storage.
            storage_compression: Whether to use compression.
            mpml: Whether to dispatch to the multiaxial-PML (``_mpml``) backend
                symbols. True exactly when the caller requested ``mpml_fraction > 0``;
                the profiles are then full-grid multiaxial profiles.
            boundary: Per-face first-class boundary codes (see
                ``deepwave._common.set_boundary``); ``[BC_PML] * 2 * ndim``
                when ``boundary=None`` was requested, which keeps the
                compiled call byte-identical (the C/CUDA ``has_bc`` guard
                skips the fill for all-PML codes -- see elastic_vti.c),
                mirroring ``acoustic_vti.AcousticVtiForwardFunc.forward``'s
                own ``boundary`` parameter.
            args: Property models, source amplitudes, source locations (1D),
                  receiver locations (1D), and initial wavefields.
        """

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # One backward-storage slot per stored derivative. The normal
            # velocity derivatives (dv?d?) feed the gradients of every
            # normal-stress-node stiffness; each shear derivative sum feeds
            # only its own shear stiffness's gradient; the dv?dbuoyancy
            # terms feed the buoyancy gradient.
            buoyancy_rg = models[-1].requires_grad
            if ndim == 3:
                cnormal_rg = any(m.requires_grad for m in models[0:4])
                c55_rg = models[4].requires_grad
                c66_rg = models[6].requires_grad
                return [
                    # z block: dvzdbuoyancy, dvzdz, then the two shear sums
                    # (xz and yz), both gated by c55.
                    buoyancy_rg,
                    cnormal_rg,
                    c55_rg,
                    c55_rg,
                    # y block: dvydbuoyancy, dvydy, dvydx_plus_dvxdy (c66)
                    buoyancy_rg,
                    cnormal_rg,
                    c66_rg,
                    # x block: dvxdbuoyancy, dvxdx
                    buoyancy_rg,
                    cnormal_rg,
                ]
            cnormal_rg = any(m.requires_grad for m in models[0:3])
            c55_rg = models[3].requires_grad
            return [
                buoyancy_rg,
                cnormal_rg,
                c55_rg,
                buoyancy_rg,
                cnormal_rg,
            ]

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            zero_edges_and_interiors(
                wavefields, ndim, fd_pad, fd_pad_list, pml_width, mpml=mpml
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
            make_call=_engine.canonical_forward_call,
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
        # ``_boundaries.mirror_pad_elastic_vti_models``). There is no
        # jvp/HVP path to guard here (unlike elastic): elastic_vti's ``jvp``
        # is unconditionally ``_engine.unsupported_jvp`` and its ``backward``
        # is unconditionally decorated with
        # ``@_engine.no_second_order("elastic_vti")`` (it has no
        # Born-powered double-backward at all).
        ctx.boundary = inputs[_N_LEAD - 1]

    # The ctypes backend cannot process batched/dual tensors; raise an
    # actionable error pointing to backend="eager".
    vmap = staticmethod(_engine.unsupported_vmap)
    jvp = staticmethod(_engine.unsupported_jvp)

    @staticmethod
    @_engine.no_second_order("elastic_vti")
    def backward(
        ctx: Any,
        *args: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        See ``elastic.ElasticForwardFunc.backward``; the
        wavefield/memory-variable structure (and hence the auxiliary
        allocation and ping-pong swap) is identical, so only the spec (and
        with it the models whose gradients are assembled) differs.
        """
        # Drop the grad slot for the trailing ``ForwardSavedState`` output.
        args = args[:-1]

        def make_aux_wavefields(
            grad_wavefields: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            aux: list[torch.Tensor] = []
            if ndim >= 3:
                aux += [torch.zeros_like(grad_wavefields[0]) for _ in range(5)]
            if ndim >= 2:
                aux += [torch.zeros_like(grad_wavefields[0]) for _ in range(3)]
            aux.append(torch.zeros_like(grad_wavefields[0]))
            return aux

        def swap(state: _engine.BackendState) -> None:
            # Identical to the isotropic elastic swap: the stress PML memory
            # variables ping-pong with the auxiliary buffers.
            ndim = state.ndim
            gw = state.wavefields
            aw = state.aux_wavefields
            if ndim == 3:
                gw[9:14], aw[-9:-4] = aw[-9:-4], gw[9:14]
                gw[20:23], aw[-4:-1] = aw[-4:-1], gw[20:23]
                gw[26], aw[-1] = aw[-1], gw[26]
            else:
                gw[6:9], aw[-4:-1] = aw[-4:-1], gw[6:9]
                gw[12], aw[-1] = aw[-1], gw[12]

        mpml = getattr(ctx, "backend_extra", "") == "_mpml"

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            zero_edges_and_interiors(
                wavefields, ndim, fd_pad, fd_pad_list, pml_width, mpml=mpml
            )

        boundary = cast("list[int]", getattr(ctx, "boundary", []))

        def model_transform(models: list[torch.Tensor]) -> list[torch.Tensor]:
            # The compiled adjoint velocity update reads the stress-role
            # stiffnesses (c11/c12/c13/c33/c55_*/c66_yx) at neighbour offsets
            # (see _boundaries.mirror_pad_elastic_vti_models for the full
            # citation); each buoyancy is read at neighbour offsets along its
            # own axis only, as in isotropic elastic.
            return _boundaries.mirror_pad_elastic_vti_models(
                models, boundary, ctx.accuracy // 2
            )

        return _engine.run_backward(
            ctx,
            SPEC,
            args,
            make_aux_wavefields=make_aux_wavefields,
            zero_wavefields=zero_wavefields,
            make_call=_engine.canonical_backward_call,
            swap=swap,
            cpu_tmp_batch=ctx.n_shots,
            n_leading=_N_LEAD,
            boundary=boundary,
            model_transform=model_transform,
        )


# Number of leading non-tensor arguments of ``ElasticVtiForwardFunc.forward``
# (the shared config block plus the trailing ``mpml``/``boundary`` pair),
# all of which receive ``None`` gradients in ``backward`` --
# ``_engine.run_backward``'s ``[None] * n_leading`` prefix (see its
# docstring) recomputes this dynamically, so no manual gradient-tuple update
# was needed here, unlike elastic's hand-assembled ``ElasticForwardFunc.
# backward``/``ElasticBackwardFunc.backward`` return tuples.
_N_LEAD = _engine.leading_arg_count(ElasticVtiForwardFunc.forward)


def update_stresses_vti(
    models: list[torch.Tensor],
    velocities: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    v_mem_vars_normal: list[torch.Tensor],
    v_mem_vars_shear: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
    fd_coeffs: Optional[list[float]] = None,
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[torch.Tensor],
    list[list[torch.Tensor]],
]:
    """Updates the stress wavefields and PML memory variables (VTI).

    Identical to ``elastic.update_stresses`` except for the
    constitutive relation: the normal stresses are updated with the VTI
    stiffness matrix and the shear stresses with ``c55``/``c66`` (each
    already averaged onto its staggered node) instead of the isotropic Lame
    combination. With the spatial dimensions indexed slowest (symmetry axis)
    to fastest, and ``d[i]`` the PML-corrected ``dv_i/dx_i``:

    - 3-D (models ``[c11, c12, c13, c33, c55_zy, c55_zx, c66_yx,
      buoyancy_*]``)::

          sigma_zz += dt * (c33 d[0] + c13 d[1] + c13 d[2])
          sigma_yy += dt * (c13 d[0] + c11 d[1] + c12 d[2])
          sigma_xx += dt * (c13 d[0] + c12 d[1] + c11 d[2])

    - 2-D (models ``[c11, c13, c33, c55_yx, buoyancy_*]``)::

          sigma_yy += dt * (c33 d[0] + c13 d[1])
          sigma_xx += dt * (c13 d[0] + c11 d[1])

    ``fd_coeffs`` is unused (elastic_vti's own custom-FD-coefficients option
    has been removed); the parameter is kept, always None, only because
    ``elastic._elastic_eager_core`` -- shared with the
    isotropic propagator, out of scope for this removal -- still calls
    ``update_stresses_fn`` with a trailing ``fd_coeffs`` positional argument.
    """
    ndim = len(grid_spacing)

    # Calculate derivatives of velocity in dimension i, in dimension i (vi_i)
    dvxdx = [
        _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    # Calculate derivatives of velocity in dimension i, in dimension j (vi_j)
    dvxdy: list[list[torch.Tensor]] = [
        [torch.empty(0) for _ in range(ndim)] for _ in range(ndim)
    ]
    for dim1 in range(ndim):
        for dim2 in range(ndim):
            if dim1 == dim2:
                continue
            dvxdy[dim1][dim2] = _staggered_grid.diff1h(
                velocities[dim1],
                dim2,
                accuracy,
                1 / grid_spacing[dim2],
                ndim,
            )

    # Extract PML profiles
    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    del pml_profiles

    # PML correct all derivatives
    for dim in range(ndim):
        v_mem_vars_normal[dim] = a[dim] * v_mem_vars_normal[dim] + b[dim] * dvxdx[dim]
        dvxdx[dim] = dvxdx[dim] + v_mem_vars_normal[dim]

    for dim1 in range(ndim):
        for dim2 in range(ndim):
            if dim1 == dim2:
                continue
            mem = v_mem_vars_shear[dim1][dim2]
            new_mem = ah[dim2] * mem + bh[dim2] * dvxdy[dim1][dim2]
            v_mem_vars_shear[dim1][dim2] = new_mem
            dvxdy[dim1][dim2] = dvxdy[dim1][dim2] + new_mem

    # Update normal stresses with the VTI constitutive relation.
    if ndim == 3:
        c11, c12, c13, c33 = models[0], models[1], models[2], models[3]
        normal_stresses = [
            normal_stresses[0]
            + dt * (c33 * dvxdx[0] + c13 * dvxdx[1] + c13 * dvxdx[2]),
            normal_stresses[1]
            + dt * (c13 * dvxdx[0] + c11 * dvxdx[1] + c12 * dvxdx[2]),
            normal_stresses[2]
            + dt * (c13 * dvxdx[0] + c12 * dvxdx[1] + c11 * dvxdx[2]),
        ]
        c55_zy, c55_zx, c66_yx = models[4], models[5], models[6]
        shear_models = [
            [torch.empty(0), c55_zy, c55_zx],
            [c55_zy, torch.empty(0), c66_yx],
            [c55_zx, c66_yx, torch.empty(0)],
        ]
    else:
        c11, c13, c33 = models[0], models[1], models[2]
        normal_stresses = [
            normal_stresses[0] + dt * (c33 * dvxdx[0] + c13 * dvxdx[1]),
            normal_stresses[1] + dt * (c13 * dvxdx[0] + c11 * dvxdx[1]),
        ]
        c55_yx = models[3]
        shear_models = [
            [torch.empty(0), c55_yx],
            [c55_yx, torch.empty(0)],
        ]

    # Update shear stresses
    for dim1 in range(ndim):
        for dim2 in range(dim1 + 1, ndim):
            shear_stresses[dim1][dim2] = shear_stresses[dim1][dim2] + dt * shear_models[
                dim1
            ][dim2] * (dvxdy[dim1][dim2] + dvxdy[dim2][dim1])
            shear_stresses[dim2][dim1] = shear_stresses[dim1][dim2]

    return (
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
    )


def elastic_vti_python(
    *args: Any,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Eager elastic VTI forward pass.

    Thin wrapper over ``elastic._elastic_eager_core`` supplying
    the VTI stress update (the velocity update is physics-independent and is
    shared with the isotropic propagator) and the VTI model names. ``*args``
    is the full leading block (ending in the ``mpml`` flag) followed by the
    tensor inputs, matching ``_elastic_eager_core``'s trailing parameters --
    including its ``boundary`` parameter (right after ``pml_width``, at
    ``args[7]``), which :func:`deepwave.elastic_vti` now populates for real
    (previously a hardcoded all-``BC_PML`` no-op splice lived here since
    ``elastic_vti`` had no public ``boundary`` keyword yet). The guard-fill
    arithmetic in ``_elastic_eager_core``/
    ``_boundaries.fill_elastic_stress_boundary``/
    ``fill_elastic_velocity_boundary`` is physics-agnostic (driven purely by
    the field lists passed in), so no change was needed there for VTI.
    ``backend`` is the resolved Python backend name (``"eager"``
    or ``"compile"``), already selected by the caller; used only to name it
    if an unsupported option is rejected.
    """
    return _elastic_eager_core(
        _update_velocities_opt,
        _update_stresses_opt,
        elastic_vti_model_names,
        *args,
        spec=SPEC,
        backend=backend,
    )


_update_velocities_cache = _engine.StepCache(update_velocities)
_update_velocities_opt = update_velocities
_update_stresses_cache = _engine.StepCache(update_stresses_vti)
_update_stresses_opt = update_stresses_vti

# Named offsets into the leading (pre-tensor-block) args elastic_vti_func
# receives from its call site (see
# _backend_utils.PROPAGATOR_REGISTRY["elastic_vti"].leading_args): the index
# of "boundary", and the index one past the last leading arg -- i.e. where
# the model/source/wavefield tensor block starts.
_BOUNDARY_IDX = _backend_utils.leading_arg_offset("elastic_vti", "boundary")
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["elastic_vti"].leading_args)


def elastic_vti_func(
    backend: Optional[Literal["eager", "compile"]] = None,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """A helper function to apply the ElasticVtiForwardFunc.

    Args:
        backend: None, or a string specifying whether to use the
            Python backend. Already coerced (no bool) by the caller.
        *args: Variable length argument list to be passed directly to
            `ElasticVtiForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ElasticVtiForwardFunc.apply`.

    """
    global _update_velocities_opt, _update_stresses_opt

    if backend:
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        _update_velocities_opt = _update_velocities_cache.resolve(backend)
        _update_stresses_opt = _update_stresses_cache.resolve(backend)
        return elastic_vti_python(*args, backend=backend)

    # ``args`` (the leading block, as received from ``elastic_vti()``) has
    # ``boundary_codes`` at ``_BOUNDARY_IDX``, right after ``pml_width`` --
    # the same slot ``elastic_vti_python``/``_elastic_eager_core`` read it
    # from. The compiled path instead needs it as a *trailing* leading arg
    # (mirroring ``ElasticVtiForwardFunc.forward``'s ``mpml, boundary`` order,
    # exactly as ``elastic_func``/``acoustic_vti``'s own
    # ``AcousticVtiForwardFunc`` do for their own ``boundary``), so move it
    # there.
    boundary = args[_BOUNDARY_IDX]
    args = (
        args[:_BOUNDARY_IDX]
        + args[_BOUNDARY_IDX + 1 : _N_LEADING_ARGS]
        + (boundary,)
        + args[_N_LEADING_ARGS:]
    )

    # ``forward`` appends a ``ForwardSavedState`` trailing output for
    # ``setup_context``; strip it so callers see only the output tensors.
    return cast(
        "tuple[torch.Tensor, ...]",
        ElasticVtiForwardFunc.apply(*args)[:-1],
    )


_common.document_propagator_module(ElasticVti, elastic_vti)

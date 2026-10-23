"""Elastic TTI (tilted transverse isotropy) wave propagation module.

Stability: experimental (see :doc:`/stability`).

Implements elastic wave propagation in a tilted transversely-isotropic (TTI)
medium: a transversely-isotropic medium whose symmetry axis has been rotated
away from the vertical by a dip angle ``tilt``. The medium is described by the
same VTI stiffnesses as :func:`deepwave.elastic_vti` (``c11``, ``c13``,
``c33``, ``c55``) together with a per-cell ``tilt`` field; the rotated
(general) stiffness tensor is obtained by applying the Bond transformation
(Bond, 1943, Bell Syst. Tech. J. 22, 1) to the VTI tensor, which is the Voigt
form of the 4th-rank tensor rotation ``C'_ijkl = R_ip R_jq R_kr R_ls
C_pqrs``.

Only the 2-D P-SV problem (propagation in the (y, x) plane, tilt about the
out-of-plane axis) is implemented. In 2-D the Bond rotation of a VTI tensor
produces a **monoclinic** stiffness with six independent in-plane constants
``C11'``, ``C13'``, ``C33'``, ``C55'`` and the two normal/shear coupling
constants ``C15'``, ``C35'`` (which vanish at ``tilt = 0``, recovering VTI
exactly). The ``c66`` stiffness and the horizontal shear plane, the 3-D
azimuth degree of freedom, and the full triclinic 3-D stencil are not part
of the 2-D P-SV equations, so this propagator has no arguments for them (see
the module docstring's Scope note).

The 2-D stress-rate relation, with the symmetry-axis dimension (``y``) indexed
slowest and ``x`` fastest, and ``e_yy = dvy/dy``, ``e_xx = dvx/dx``,
``gamma = dvx/dy + dvy/dx``::

    d(sigma_yy)/dt = C33' e_yy + C13' e_xx + C35' gamma
    d(sigma_xx)/dt = C13' e_yy + C11' e_xx + C15' gamma
    d(sigma_xy)/dt = C35' e_yy + C15' e_xx + C55' gamma

At ``tilt = 0`` (``C15' = C35' = 0``) this reduces exactly to the
:func:`deepwave.elastic_vti` 2-D constitutive relation.

Scope and backends. There is no compiled C/CUDA backend yet; both
Python modes are implemented -- ``"eager"`` and ``"torch_compile"``
(``torch.compile``) -- since the compiled mode only needs this module's own
``update_stresses_tti`` and ``elastic.update_velocities`` to be traceable,
independent of any compiled kernel. The default ``backend="auto"``
resolves to ``"eager"``, while ``backend="compiled"`` raises
``NotImplementedError`` rather than quietly falling back to it; pass
``"torch_compile"`` explicitly to opt into the (usually
faster) alternative. Only 2-D is implemented; 3-D inputs raise
``NotImplementedError``. The
default ``mpml_fraction`` is 0.0, matching every other propagator; tilted
anisotropy is a well-known trigger of C-PML instability, so users who see
PML instability with a tilted medium may want to pass ``mpml_fraction`` around
0.15 to enable a small multiaxial-PML cross-damping fraction
(Meza-Fajardo & Papageorgiou, 2008, BSSA 98, 1811; the M-PML machinery is
shared with :func:`deepwave.elastic`).

Discretisation. The staggered grid, velocity update, C-PML/M-PML memory
variables, source/receiver handling and free surface are inherited unchanged
from :func:`deepwave.elastic`/:func:`deepwave.elastic_vti`; only the
constitutive stress update differs. The normal strains live at the
normal-stress (integer) nodes and the shear strain at the staggered
shear-stress node, so the ``C15'``/``C35'`` cross-terms require interpolation
between the two node locations. This is done with a four-point average and its
exact transpose, arranged so the discrete stiffness operator is symmetric
(energy-conserving, hence reciprocal). The cross-term interpolation is
second-order accurate in space regardless of ``accuracy``; see
``docs/elastic_tti_derivation.md``.

See :doc:`/capabilities` for how this compares to the other 10 propagators
(backends, boundary faces, callbacks, differentiation, ...) at a glance.
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
from deepwave import _common, _engine, _staggered_grid
from deepwave._exceptions import (
    DeepwaveNotImplementedError,
    DeepwaveRuntimeError,
    DeepwaveValueError,
)
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
)
from deepwave.elastic import (
    prepare_parameters as _prepare_isotropic_parameters,
)
from deepwave.elastic_vti import _velocity_extremes_vti
from deepwave.types import (
    Accuracy,
    Callback,
    LocationsLike,
    ReceiverSpec,
    RunInfo,
    SourceSpec,
    StorageLike,
    StorageMode,
    StoragePath,
)


def elastic_tti_model_names(ndim: int) -> list[str]:
    """Names of the elastic TTI models (2-D only).

    The models are the rotated (Bond-transformed) stiffnesses at the
    normal-stress nodes (``c11_rot``, ``c13_rot``, ``c33_rot``), the rotated
    shear and normal/shear-coupling stiffnesses averaged onto the staggered
    shear-stress node (``c55_rot_yx``, ``c15_rot_yx``, ``c35_rot_yx``), and the
    buoyancy for each velocity component.

    Every rotated stiffness carries ``_rot`` in its name. The unsuffixed
    ``c11``/``c13``/``c33``/``c55`` are the *public* arguments of
    :func:`deepwave.elastic_tti`: the stiffnesses of the untilted VTI medium,
    before the Bond rotation by ``tilt``. Those are a different quantity from
    these, and equal to them only where ``tilt`` is zero, so they do not share
    a name: a callback asking for ``c11`` would otherwise get the rotated
    constant while every other propagator's ``c11`` is the unrotated one.
    """
    if ndim == 2:
        return [
            "c11_rot",
            "c13_rot",
            "c33_rot",
            "c55_rot_yx",
            "c15_rot_yx",
            "c35_rot_yx",
            "buoyancy_y",
            "buoyancy_x",
        ]
    raise DeepwaveValueError(
        "elastic_tti: currently only 2 spatial dimensions are supported "
        "(3-D TTI elastic propagation is not yet implemented)."
    )


# The user-facing model arguments, in the order the public function takes
# them. ``tilt`` is one of them: it is a model-shaped, differentiable tensor
# (zero everywhere when omitted, which reproduces
# :func:`deepwave.elastic_vti`). There is no ``c66`` or ``azimuth``: both are
# 3-D-only degrees of freedom and this propagator is 2-D only, so rather than
# carrying arguments that could only ever be rejected, it does not take them.
# Adding them back as keyword-only arguments when 3-D TTI exists will not be
# a breaking change.
USER_MODEL_NAMES = ("c11", "c13", "c33", "c55", "buoyancy", "tilt")


def elastic_tti_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`deepwave.elastic_tti`, in return order.

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


# Declarative description of the elastic TTI propagator. The staggered-grid
# wavefield layout (and hence the wavefield and output names) is identical to
# the isotropic elastic propagator's; only the models differ.
SPEC = _engine.PropagatorSpec(
    name="elastic_tti",
    status="experimental",
    ndims=(2,),
    source_channels=_engine.pressure_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    supported_options=frozenset({"mpml_fraction"}),
    staggered=True,
    model_names=elastic_tti_model_names,
    wavefield_names=elastic_wavefield_names,
    n_source_types=lambda ndim: ndim + 1,
    n_receiver_types=lambda ndim: ndim + 1,
    output_names=elastic_tti_output_names,
    auxiliary_state_names=elastic_auxiliary_state_names,
    public_wavefield_names=elastic_public_wavefield_names,
    c_model_flag_indices=lambda ndim: [0, 1, 2, 3, 4, 5, -1],
    spatial_parallel=True,
    has_compiled_backend=False,
)


def _bond_rotate_2d(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55: torch.Tensor,
    tilt: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Bond-rotate a 2-D VTI stiffness by ``tilt`` (radians) about the axis.

    Returns the six independent in-plane monoclinic constants
    ``(C11', C13', C33', C55', C15', C35')`` in the reduced Voigt basis
    ``(xx, yy=axis, xy)``. Implemented as ``C' = M Cred M^T`` with the 2-D
    Bond matrix ``M`` for a rotation that takes the symmetry axis toward the
    fast (``x``) dimension; every operation is differentiable, so autograd
    propagates gradients back to ``c11``/``c13``/``c33``/``c55`` and ``tilt``.

    At ``tilt = 0`` this returns ``(c11, c13, c33, c55, 0, 0)`` exactly.
    """
    cos = torch.cos(tilt)
    sin = torch.sin(tilt)
    c2 = cos * cos
    s2 = sin * sin
    cs = cos * sin
    # 2-D Bond matrix rows for the reduced Voigt basis (xx, yy, xy).
    m = [
        [c2, s2, 2 * cs],
        [s2, c2, -2 * cs],
        [-cs, cs, c2 - s2],
    ]

    def entry(a: int, b: int) -> torch.Tensor:
        # (M Cred M^T)_{ab} for Cred = [[c11,c13,0],[c13,c33,0],[0,0,c55]].
        return (
            m[a][0] * m[b][0] * c11
            + (m[a][0] * m[b][1] + m[a][1] * m[b][0]) * c13
            + m[a][1] * m[b][1] * c33
            + m[a][2] * m[b][2] * c55
        )

    c11r = entry(0, 0)
    c33r = entry(1, 1)
    c13r = entry(0, 1)
    c55r = entry(2, 2)
    c15r = entry(0, 2)
    c35r = entry(1, 2)
    return c11r, c13r, c33r, c55r, c15r, c35r


def _avg_to_shear(f: torch.Tensor) -> torch.Tensor:
    """Four-point average from the integer nodes onto the shear-stress node.

    Maps a field defined at the normal-stress (integer) nodes to the staggered
    shear-stress node, using the same four cells (and trailing zero pad) as the
    shear-modulus averaging of ``elastic.prepare_parameters``, so
    the interpolated quantities are co-located with ``c55_rot_yx``.
    """
    a = (f[..., 1:, 1:] + f[..., :-1, 1:] + f[..., 1:, :-1] + f[..., :-1, :-1]) * 0.25
    return torch.nn.functional.pad(a, (0, 1, 0, 1))


def _avg_from_shear(h: torch.Tensor) -> torch.Tensor:
    """Exact transpose of :func:`_avg_to_shear` (shear node -> integer nodes).

    For inputs that vanish on the trailing row/column (which is the case for
    every quantity it is applied to here -- the shear-node stiffness fields are
    zero there by construction), this is the exact adjoint of
    :func:`_avg_to_shear`, making the combined cross-term stiffness operator
    symmetric and hence the scheme reciprocal.
    """
    hp = torch.nn.functional.pad(h, (1, 0, 1, 0))
    return (
        hp[..., 1:, 1:] + hp[..., :-1, 1:] + hp[..., 1:, :-1] + hp[..., :-1, :-1]
    ) * 0.25


def _positive_shear_coupling(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55_yx: torch.Tensor,
    c15_yx: torch.Tensor,
    c35_yx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound mixed stiffness by the normal energy available at each node.

    Each integer node contributes one quarter of its normal-strain energy to
    each adjacent shear node. The Schur complement of that local block is
    positive when the shear stiffness exceeds the average of
    ``c_mixed.T @ C_normal^-1 @ c_mixed`` over the four corners.
    """
    x = c15_yx[..., :-1, :-1]
    y = c35_yx[..., :-1, :-1]
    terms = []
    for i in (0, 1):
        for j in (0, 1):
            a = c11[..., i : c11.shape[-2] - 1 + i, j : c11.shape[-1] - 1 + j]
            b = c13[..., i : c13.shape[-2] - 1 + i, j : c13.shape[-1] - 1 + j]
            d = c33[..., i : c33.shape[-2] - 1 + i, j : c33.shape[-1] - 1 + j]
            determinant = a * d - b.square()
            numerator = d * x.square() - 2 * b * x * y + a * y.square()
            terms.append(
                torch.where(
                    determinant > 0,
                    numerator / determinant.clamp_min(torch.finfo(a.dtype).tiny),
                    torch.zeros_like(numerator),
                )
            )
    required = torch.nn.functional.pad(sum(terms) * 0.25, (0, 1, 0, 1))
    available = c55_yx.clamp_min(0) * (1.0 - 1e-6)
    scale = torch.where(
        required > available,
        torch.sqrt(available / required.clamp_min(torch.finfo(required.dtype).tiny)),
        torch.ones_like(required),
    )
    return c15_yx * scale, c35_yx * scale


def prepare_parameters(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55: torch.Tensor,
    tilt: torch.Tensor,
    buoyancy: torch.Tensor,
) -> list[torch.Tensor]:
    """Build the 2-D TTI backend model list from padded user models.

    Applies the Bond rotation (differentiably) to obtain the monoclinic
    stiffnesses, keeps ``C11'``/``C13'``/``C33'`` at the integer normal-stress
    nodes, and averages ``C55'``/``C15'``/``C35'`` onto the staggered
    shear-stress node. ``C55'`` uses the same vacuum-masked harmonic average as
    the isotropic shear modulus (so a zero-tilt medium reproduces
    ``elastic_vti`` exactly, including at free surfaces); the sign-changing
    coupling stiffnesses ``C15'``/``C35'`` use an arithmetic average (harmonic
    averaging is undefined for quantities that change sign), followed by a
    local Schur-complement limiter where heterogeneous averaging would make
    the assembled staggered energy indefinite. Buoyancy is
    averaged onto the velocity nodes exactly as in the isotropic propagator.
    """
    c11r, c13r, c33r, c55r, c15r, c35r = _bond_rotate_2d(c11, c13, c33, c55, tilt)
    # c55_rot_yx (harmonic, vacuum-masked) plus the two velocity-node
    # buoyancies.
    c55_rot_yx, buoyancy_y, buoyancy_x = _prepare_isotropic_parameters(c55r, buoyancy)
    c15_rot_yx = _avg_to_shear(c15r)
    c35_rot_yx = _avg_to_shear(c35r)
    c15_rot_yx, c35_rot_yx = _positive_shear_coupling(
        c11r, c13r, c33r, c55_rot_yx, c15_rot_yx, c35_rot_yx
    )
    return [
        c11r,
        c13r,
        c33r,
        c55_rot_yx,
        c15_rot_yx,
        c35_rot_yx,
        buoyancy_y,
        buoyancy_x,
    ]


def update_stresses_tti(
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
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[torch.Tensor],
    list[list[torch.Tensor]],
]:
    """Updates the stress wavefields and PML memory variables (2-D TTI).

    Identical to ``elastic_vti.update_stresses_vti`` except that
    the constitutive relation is the general (monoclinic) 2-D stiffness with
    the ``C15'``/``C35'`` normal/shear coupling. The strains are computed and
    PML-corrected exactly as in the VTI kernel; the coupling terms interpolate
    between the integer normal-stress node (where ``dvxdx`` lives) and the
    staggered shear-stress node (where the shear strain lives) with
    :func:`_avg_to_shear` and its exact transpose :func:`_avg_from_shear`, so
    the discrete stiffness operator is symmetric. With ``models = [c11_rot,
    c13_rot, c33_rot, c55_rot_yx, c15_rot_yx, c35_rot_yx, buoyancy_y,
    buoyancy_x]``::

        e_yy = dvy/dy   (integer node)   e_xx = dvx/dx   (integer node)
        gamma = dvx/dy + dvy/dx          (shear node)
        sigma_yy += dt (c33_rot e_yy + c13_rot e_xx + A^T(c35_rot_yx gamma))
        sigma_xx += dt (c13_rot e_yy + c11_rot e_xx + A^T(c15_rot_yx gamma))
        sigma_xy += dt (c35_rot_yx A(e_yy) + c15_rot_yx A(e_xx)
                        + c55_rot_yx gamma)
    """
    ndim = len(grid_spacing)
    if ndim != 2:
        raise DeepwaveNotImplementedError(
            "elastic_tti: only 2 spatial dimensions are supported."
        )

    # Diagonal (normal) strains dv_i/dx_i at the integer normal-stress nodes.
    dvxdx = [
        _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    # Off-diagonal (shear) strains dv_i/dx_j at the staggered shear-stress node.
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

    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    del pml_profiles

    # PML-correct the integer-node diagonal strains (integer profiles a, b).
    for dim in range(ndim):
        v_mem_vars_normal[dim] = a[dim] * v_mem_vars_normal[dim] + b[dim] * dvxdx[dim]
        dvxdx[dim] = dvxdx[dim] + v_mem_vars_normal[dim]

    # PML-correct the half-node shear strains (half profiles ah, bh).
    for dim1 in range(ndim):
        for dim2 in range(ndim):
            if dim1 == dim2:
                continue
            mem = v_mem_vars_shear[dim1][dim2]
            new_mem = ah[dim2] * mem + bh[dim2] * dvxdy[dim1][dim2]
            v_mem_vars_shear[dim1][dim2] = new_mem
            dvxdy[dim1][dim2] = dvxdy[dim1][dim2] + new_mem

    c11_rot, c13_rot, c33_rot = models[0], models[1], models[2]
    c55_rot_yx, c15_rot_yx, c35_rot_yx = models[3], models[4], models[5]

    # e_yy = dvxdx[0] (axis), e_xx = dvxdx[1]; gamma at the shear node.
    e_yy = dvxdx[0]
    e_xx = dvxdx[1]
    gamma = dvxdy[0][1] + dvxdy[1][0]

    # Normal stresses (integer node). The coupling term interpolates the
    # shear-node quantity c*_rot_yx * gamma back to the integer node via A^T.
    normal_stresses = [
        normal_stresses[0]
        + dt * (c33_rot * e_yy + c13_rot * e_xx + _avg_from_shear(c35_rot_yx * gamma)),
        normal_stresses[1]
        + dt * (c13_rot * e_yy + c11_rot * e_xx + _avg_from_shear(c15_rot_yx * gamma)),
    ]

    # Shear stress (shear node). The coupling term interpolates the integer-node
    # normal strains onto the shear node via A (the transpose of A^T above, so
    # the combined stiffness operator is symmetric).
    shear_stresses[0][1] = shear_stresses[0][1] + dt * (
        c35_rot_yx * _avg_to_shear(e_yy)
        + c15_rot_yx * _avg_to_shear(e_xx)
        + c55_rot_yx * gamma
    )
    shear_stresses[1][0] = shear_stresses[0][1]

    return (
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
    )


def elastic_tti_python(
    *args: Any,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Eager elastic TTI forward pass.

    Thin wrapper over ``elastic._elastic_eager_core`` supplying
    the TTI (monoclinic) stress update (honouring the ``torch.compile``
    module globals selected by :func:`elastic_tti_func`) and the velocity
    update, which is physics-independent and shared with the isotropic
    propagator. ``backend`` is used only to name a backend if an unsupported
    option is rejected -- it is the caller's own pre-resolution choice (see
    :func:`elastic_tti_func`), not necessarily the resolved mode actually
    running.
    """
    return _elastic_eager_core(
        _update_velocities_opt,
        _update_stresses_opt,
        elastic_tti_model_names,
        *args,
        spec=SPEC,
        backend=backend,
    )


_update_velocities_compile: Optional[Any] = None
_update_velocities_opt: Any = update_velocities
_update_stresses_compile: Optional[Any] = None
_update_stresses_opt: Any = update_stresses_tti


def elastic_tti_func(
    backend: Optional[Literal["eager", "compile"]],
    requested_backend: str,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """Dispatches elastic TTI propagation to one of the two Python modes.

    There is no compiled C/CUDA backend, so ``backend`` is always
    truthy by the time it reaches this function (:func:`elastic_tti`
    already resolves ``"auto"`` via ``_engine.resolve_eager_only_backend``
    before dispatching); the falsy check below is a defensive invariant,
    not reachable via the public API.

    ``requested_backend`` is the caller's own pre-resolution ``backend``
    choice (``"auto"`` if not given explicitly, else ``"eager"``/
    ``"torch_compile"``) -- passed through to
    :func:`elastic_tti_python` for its unsupported-option message instead of
    the resolved ``mode``, so a default ``backend="auto"`` call is never
    reported as having asked for
    ``backend="eager"`` (see :func:`_engine.reject_unsupported_python_backend_options`).
    """
    global _update_velocities_compile, _update_velocities_opt
    global _update_stresses_compile, _update_stresses_opt

    if not backend:  # pragma: no cover - guarded in elastic_tti()
        raise DeepwaveNotImplementedError(
            "elastic_tti: only the eager Python backend is supported."
        )
    mode = _engine.resolve_backend_mode(backend)
    if mode == "compile":
        if _update_velocities_compile is None:
            _update_velocities_compile = torch.compile(
                update_velocities, fullgraph=True
            )
        _update_velocities_opt = _update_velocities_compile
        if _update_stresses_compile is None:
            _update_stresses_compile = torch.compile(
                update_stresses_tti, fullgraph=True
            )
        _update_stresses_opt = _update_stresses_compile
    else:  # eager
        _update_velocities_opt = update_velocities
        _update_stresses_opt = update_stresses_tti
    return elastic_tti_python(*args, backend=requested_backend)


def elastic_tti(
    c11: torch.Tensor,
    c13: torch.Tensor,
    c33: torch.Tensor,
    c55: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    tilt: Optional[torch.Tensor] = None,
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
    initial_state: Optional[
        Union["deepwave.PropagatorOutput", Mapping[str, torch.Tensor]]
    ] = None,
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
    """Elastic TTI wave propagation (functional interface; 2-D, eager backend).

    Stability: experimental (see :doc:`/stability`).

    Forward modelling with the 2-D elastic wave equation in a tilted
    transversely-isotropic medium. The medium is a VTI medium (stiffnesses
    ``c11``, ``c13``, ``c33``, ``c55``) whose symmetry axis is rotated away
    from the vertical (the slow ``y`` dimension) by the per-cell dip angle
    ``tilt`` (radians), via the Bond transformation. The outputs are
    differentiable with respect to every input float torch.Tensor, including
    the stiffnesses, the buoyancy, and ``tilt``.

    The propagator uses the same staggered grid, C-PML/M-PML absorbing
    boundaries, improved-vacuum free surface, source/receiver types, and
    input/output conventions as :func:`deepwave.elastic_vti` (and hence
    :func:`deepwave.elastic`); only the constitutive relation differs (see the
    module docstring and ``docs/elastic_tti_derivation.md``). With ``tilt``
    everywhere zero (or ``None``) the result matches :func:`deepwave.elastic_vti`
    to machine precision.

    Scope. Only 2-D is implemented (3-D raises ``NotImplementedError``), and
    the model set is therefore the 2-D P-SV one: ``c11``, ``c13``, ``c33``,
    ``c55``, ``buoyancy`` and ``tilt``. There is no compiled C/CUDA
    backend yet: the default ``backend="auto"`` resolves to the eager Python
    backend, and ``backend="compiled"`` raises ``NotImplementedError``
    rather than quietly falling back to it. Both Python modes --
    ``"eager"`` and ``"torch_compile"`` -- are available. A
    compiled C/CUDA backend and 3-D TTI support are follow-up work.

    Stability. Tilted anisotropy is a known trigger of C-PML instability.
    ``mpml_fraction`` defaults to 0.0, matching every other propagator, but users
    who see PML instability with a tilted medium may want to pass
    ``mpml_fraction`` around 0.15 to enable a small multiaxial-PML cross-damping
    that stabilises the tilted-medium boundaries at the cost of slightly
    larger boundary reflections; a validated operating envelope (maximum
    tilt vs. anisotropy strength) is documented in
    ``docs/elastic_tti_derivation.md`` and exercised by the test suite.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.
    The one exception to that freedom is this family: its symmetry axis
    is dimension 0 by construction, so here the axis order is a physics
    constraint rather than a documentation convention.

    Args:
        c11: The c11 stiffness model (of the untilted VTI medium), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        c13: The c13 stiffness model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c33: The c33 stiffness model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c55: The c55 stiffness model, of shape ``[*model_shape]`` or, for a
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
        grid_spacing: The spatial grid cell size (a single number or one per
            dimension).
        dt: The time step interval of the input and output. See
            :func:`deepwave.elastic` for the unit-consistency requirement
            (here applying to the stiffnesses, ``buoyancy``,
            ``grid_spacing``, and ``dt``).
        tilt: The dip angle (radians) model of the symmetry axis away from
            the ``y`` (slow) dimension, of shape ``[*model_shape]`` or, for
            a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See
            :ref:`batched-models`. A scalar tensor is also accepted and is
            broadcast to the model shape. ``None`` (the default) means zero
            tilt everywhere, which reproduces
            :func:`deepwave.elastic_vti`. Differentiable.
        source_amplitudes_z: Not applicable here: this propagator is 2-D only,
            so it has no ``z`` channel. Passing it raises.
        source_amplitudes_y: See :func:`deepwave.elastic_vti`.
        source_amplitudes_x: See :func:`deepwave.elastic_vti`.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``pressure_rate``, ``y``,
            ``x`` (2-D only: there is no ``z`` channel); these are equivalent
            to the flat ``source_amplitudes_*``/``source_locations_*`` keywords,
            which remain permanently supported; giving one channel in both
            forms raises. Optional.
        receivers: This simulation's receivers, keyed by channel, as the
            canonical alternative to the flat ``receiver_locations*`` keywords.
            Each value is a locations object, or ``(locations, mask)`` to
            include an activity mask. The channels are ``p``, ``y``, ``x`` (2-D
            only: there is no ``z`` channel). Giving one channel in both forms
            raises. Optional.
        source_amplitudes_pressure_rate: See :func:`deepwave.elastic_vti`.
        source_locations_z: Not applicable here: this propagator is 2-D only,
            so it has no ``z`` channel. Passing it raises, so nothing else
            in this entry applies to it.
        source_locations_y: See :func:`deepwave.elastic_vti`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_x: See :func:`deepwave.elastic_vti`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_pressure_rate: See :func:`deepwave.elastic_vti`.
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
        source_mask_z: Not applicable here: this propagator is 2-D only, so it
            has no ``z`` channel. Passing it raises.
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
        receiver_locations_z: Not applicable here: this propagator is 2-D only,
            so it has no ``z`` channel. Passing it raises, so nothing else
            in this entry applies to it.
        receiver_locations_y: See :func:`deepwave.elastic_vti`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_x: See :func:`deepwave.elastic_vti`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_p: See :func:`deepwave.elastic_vti`. In 2-D the
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
        receiver_mask_z: Not applicable here: this propagator is 2-D only, so
            it has no ``z`` channel. Passing it raises.
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
        vz_0: See :func:`deepwave.elastic` (and similarly for the other initial
            wavefields).
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
        accuracy: See :func:`deepwave.elastic`. Default 4. Note the
            ``C15``/``C35`` cross-terms are second-order accurate regardless of
            ``accuracy`` (see the module docstring).
        pml_width: See :func:`deepwave.elastic`. Default 20.
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: See :func:`deepwave.elastic_vti`.
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
        mpml_fraction: The multiaxial-PML cross-damping fraction. Default 0.0; see
            the Stability note above for why TTI users may want ~0.15.
        max_vel: See :func:`deepwave.elastic_vti`. The default bound is
            computed from the untilted stiffnesses; because the maximum phase
            velocity is rotation-invariant, the VTI bound is also a valid bound
            for any tilt.
        courant_number: See :func:`deepwave.elastic`.
        survey_pad: See :func:`deepwave.scalar`.
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
            `nt`, the individual initial-wavefield arguments, `source_field`
            and the per-step callbacks are not, and say so.

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
            There is no compiled C/CUDA backend for this propagator, so
            ``"auto"`` resolves silently to ``"eager"`` (check
            ``out.info.backend``) and ``"compiled"`` raises
            :exc:`deepwave.DeepwaveNotImplementedError`; pass
            ``"torch_compile"`` to opt into the other Python mode.
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
            Only `"device"` is available for `elastic_tti`: it has no
            compiled backend, so every call runs on the eager Python
            backend, and every other mode raises there.
        storage_path: The directory that `storage_mode="disk"` writes to. A
            string or any `os.PathLike` (such as a `pathlib.Path`). Defaults
            to the system temporary directory, resolved **when you call**
            rather than when Deepwave is imported, so `TMPDIR` set at any
            point before the call is honoured. Using the default with
            `storage_mode="disk"` emits a one-time warning naming the
            directory, because on many systems it is a RAM-backed `tmpfs`
            far smaller than the disk. A permanent alias for `Storage.path`;
            not deprecated.
        storage_compression: **Lossy where it is available -- it changes
            the model gradient by about 0.2-4% in 2-D and 0.2-8% in 3-D.**
            Not available for `elastic_tti`: compression is a
            compiled-backend facility and `elastic_tti` has no compiled
            backend, so requesting it raises rather than changing any number.
            See :func:`deepwave.elastic`.
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
        deepwave.PropagatorOutput: the same outputs, in the same order, as
        :func:`deepwave.elastic_vti`. In 2-D the recorded pressure is the
        in-plane mean normal stress ``-(sigma_xx + sigma_yy)/2``, which
        under plane strain is not the thermodynamic pressure; see
        :func:`deepwave.elastic`'s ``receiver_amplitudes_p`` entry for the
        full statement.
    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            elastic_tti,
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
        "elastic_tti",
    )
    if tilt is not None:
        _common.check_models([tilt], ["tilt"], "elastic_tti")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    del backend
    # The caller's own pre-resolution choice, kept for the unsupported-option
    # message -- see reject_unsupported_python_backend_options's docstring.
    requested_backend = backend_request.canonical

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
        _m_vzz_0,
        _m_vzy_0,
        _m_vzx_0,
        _m_vyz_0,
        _m_vxz_0,
        m_vyy_0,
        m_vyx_0,
        m_vxy_0,
        m_vxx_0,
        _m_sigmazzz_0,
        _m_sigmayzy_0,
        _m_sigmaxzx_0,
        _m_sigmayzz_0,
        _m_sigmaxzz_0,
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
        vy_0,
        vx_0,
        sigmazz_0,
        sigmayz_0,
        sigmaxz_0,
        sigmayy_0,
        sigmaxy_0,
        sigmaxx_0,
    )

    raw_models: list[torch.Tensor] = [c11, c13, c33, c55, buoyancy]
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
        [vz_0, sigmazz_0, sigmayz_0, sigmaxz_0],
        [vy_0, sigmayy_0, sigmaxy_0],
        [vx_0, sigmaxx_0],
    )
    if ndim < 2:
        raise DeepwaveValueError(
            "elastic_tti: at least 2 spatial dimensions are required; use "
            "deepwave.elastic for 1-D."
        )
    if ndim == 3:
        raise DeepwaveNotImplementedError(
            "elastic_tti: only 2-D is currently supported. 3-D TTI -- which "
            "needs the azimuth degree of freedom and the c66 stiffness as "
            "well as tilt -- is not yet implemented."
        )
    # Resolved once ndim is known, so that backend="compiled" -- which this
    # propagator can never satisfy -- can name the whole configuration.
    backend_mode = _engine.resolve_eager_only_backend(
        backend_request,
        "elastic_tti",
        dtype=c11.dtype,
        ndim=ndim,
        device=c11.device,
    )
    if source_locations_z is not None or receiver_locations_z is not None:
        raise DeepwaveValueError(
            "elastic_tti: the propagation was determined to be 2d, so "
            "locations related to the z dimension should not be provided."
        )
    if not 0 <= mpml_fraction < 1:
        raise DeepwaveValueError(
            "elastic_tti: mpml_fraction (the multiaxial-PML cross-damping fraction) "
            f"must be in [0, 1), but got {mpml_fraction}."
        )

    # Materialise tilt as a differentiable model-shaped tensor (None -> 0).
    if tilt is None:
        tilt_full = torch.zeros_like(c11)
    else:
        # Broadcast a scalar or model-shaped tilt to c11's shape while keeping
        # the gradient path (addition of a zero tensor broadcasts and is an
        # identity on the values).
        tilt_full = torch.zeros_like(c11) + tilt.to(dtype=c11.dtype, device=c11.device)
    raw_models = [c11, c13, c33, c55, tilt_full, buoyancy]
    raw_model_names = ["c11", "c13", "c33", "c55", "tilt", "buoyancy"]

    source_amplitudes: list[Optional[torch.Tensor]] = []
    source_locations_in: list[Optional[LocationsLike]] = []
    receiver_locations_in: list[Optional[LocationsLike]] = []
    initial_wavefields: list[Optional[torch.Tensor]] = []
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
        "elastic_tti",
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

    dim_names = ["y", "x"]
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
                    f"elastic_tti: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{c11.shape[-ndim + dim] - 1}.",
                )

    if not all(_common.values_readable(prop) for prop in raw_models):
        if max_vel is None:
            raise _common.max_vel_unreadable_error("elastic_tti")
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        # The maximum phase velocity is rotation-invariant, so the VTI bound
        # (from the untilted stiffnesses) also bounds the tilted medium.
        max_model_vel, min_nonzero_model_vel = _velocity_extremes_vti(
            c11, c13, c33, c55, None, buoyancy
        )
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

    user_dt = dt
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
    del c11, c13, c33, c55, buoyancy, tilt, tilt_full, raw_models
    del source_amplitudes, source_locations, receiver_locations
    del initial_wavefields

    c11_p, c13_p, c33_p, c55_p, tilt_p, buoyancy_p = models
    models = prepare_parameters(c11_p, c13_p, c33_p, c55_p, tilt_p, buoyancy_p)
    del c11_p, c13_p, c33_p, c55_p, tilt_p, buoyancy_p

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "elastic_tti")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "elastic_tti")
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
    )

    model_shape = models[0].shape[-ndim:]
    # y force sources scale by buoyancy_y (second-from-last model).
    if source_amplitudes_out[-3].numel() > 0:
        source_amplitudes_out[-3] = (
            source_amplitudes_out[-3]
            * _common.gather_model_at_sources(
                models[-2], sources_i[-3], model_shape, n_shots
            )
            * dt
        )
    # x force sources scale by buoyancy_x (last model).
    if source_amplitudes_out[-2].numel() > 0:
        source_amplitudes_out[-2] = (
            source_amplitudes_out[-2]
            * _common.gather_model_at_sources(
                models[-1], sources_i[-2], model_shape, n_shots
            )
            * dt
        )
    # Pressure sources scale by -dt.
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

    run_info = RunInfo(
        step_ratio=step_ratio,
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
        python_backend=requested_backend,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
    )

    outputs = list(
        elastic_tti_func(
            backend_mode,
            requested_backend,
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio * model_gradient_sampling_interval,
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

    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 1 - ndim
    )

    if outputs[-1] is not None:
        outputs[-1] = -outputs[-1] / ndim

    outputs[: -1 - ndim] = [outputs[i] for i in _elastic_public_wavefield_order(ndim)]

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


class ElasticTti(torch.nn.Module):
    """A PyTorch Module for 2-D elastic TTI wave propagation.

    Stability: experimental (see :doc:`/stability`).

    Stores the VTI stiffnesses (``c11``, ``c13``, ``c33``, ``c55``), the
    ``tilt`` field and the buoyancy as ``torch.nn.Parameter`` objects, so
    gradients may be computed with respect to any of them (including ``tilt``).
    See :func:`elastic_tti` for the scope, backend, and stability notes.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.
    The one exception to that freedom is this family: its symmetry axis
    is dimension 0 by construction, so here the axis order is a physics
    constraint rather than a documentation convention.

    Args:
        c11: The c11 stiffness model (of the untilted VTI medium), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        c13: The c13 stiffness model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c33: The c33 stiffness model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        c55: The c55 stiffness model, of shape ``[*model_shape]`` or, for a
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
        grid_spacing: The spatial grid cell size.
        tilt: The dip angle (radians) model, or None for zero tilt, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. A scalar tensor is also
            accepted and is broadcast to the model shape.
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
        tilt_requires_grad: A bool specifying whether gradients will be
            computed for `tilt`. Unset by default, which means False. Cannot
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
            Only `"device"` is available for `elastic_tti`: it has no
            compiled backend, so every call runs on the eager Python
            backend, and every other mode raises there.
        storage_path: The directory that `storage_mode="disk"` writes to. A
            string or any `os.PathLike` (such as a `pathlib.Path`). Defaults
            to the system temporary directory, resolved **when you call**
            rather than when Deepwave is imported, so `TMPDIR` set at any
            point before the call is honoured. Using the default with
            `storage_mode="disk"` emits a one-time warning naming the
            directory, because on many systems it is a RAM-backed `tmpfs`
            far smaller than the disk. A permanent alias for `Storage.path`;
            not deprecated.
        storage_compression: **Lossy where it is available -- it changes
            the model gradient by about 0.2-4% in 2-D and 0.2-8% in 3-D.**
            Not available for `elastic_tti`: compression is a
            compiled-backend facility and `elastic_tti` has no compiled
            backend, so requesting it raises rather than changing any number.
            See :func:`deepwave.elastic`.
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
        tilt: Optional[torch.Tensor] = None,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        c11_requires_grad: Optional[bool] = None,
        c13_requires_grad: Optional[bool] = None,
        c33_requires_grad: Optional[bool] = None,
        c55_requires_grad: Optional[bool] = None,
        buoyancy_requires_grad: Optional[bool] = None,
        tilt_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        owned = ["c11", "c13", "c33", "c55", "buoyancy", "tilt"]
        optional = {"tilt": tilt}
        supplied = [
            name for name in owned if name not in optional or optional[name] is not None
        ]
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
                "tilt": tilt_requires_grad,
            },
            "ElasticTti",
        )
        self.c11 = _common.make_model_parameter(c11, grad["c11"], "c11")
        self.c13 = _common.make_model_parameter(c13, grad["c13"], "c13")
        self.c33 = _common.make_model_parameter(c33, grad["c33"], "c33")
        self.c55 = _common.make_model_parameter(c55, grad["c55"], "c55")
        # ``buoyancy`` is registered before ``tilt`` so that the parameters,
        # and therefore ``state_dict()``, are in the order
        # ``deepwave.spec("elastic_tti").model_names(ndim)`` reports.
        self.buoyancy = _common.make_model_parameter(
            buoyancy, grad["buoyancy"], "buoyancy"
        )
        self.tilt: Optional[torch.Tensor] = (
            None
            if tilt is None
            else _common.make_model_parameter(tilt, grad["tilt"], "tilt")
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
        initial_state: Optional[
            Union["deepwave.PropagatorOutput", Mapping[str, torch.Tensor]]
        ] = None,
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

        The inputs are the same as for :func:`elastic_tti` except that the
        stiffnesses, ``tilt``, ``buoyancy``, ``grid_spacing``, and the storage
        settings (``storage_mode``/``storage_path``/``storage_compression``)
        are taken from the Module (supplied from the stored module state).
        See :func:`elastic_tti`.
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
        return elastic_tti(
            self.c11,
            self.c13,
            self.c33,
            self.c55,
            self.buoyancy,
            self.grid_spacing,
            dt,
            tilt=self.tilt,
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


_common.document_propagator_module(ElasticTti, elastic_tti)

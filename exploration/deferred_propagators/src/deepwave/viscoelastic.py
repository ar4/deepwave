"""Viscoelastic (elastic + Qp/Qs GSLS attenuation) wave propagation module.

Stability: experimental (see :doc:`/stability`).

Implements the isotropic elastic velocity-stress wave equation
(:mod:`deepwave.elastic`) augmented with generalised-standard-linear-solid
(GSLS) relaxation mechanisms on the stress constitutive law, so that the
medium has independent, finite, frequency-dependent quality factors for
the mean/bulk modulus (``Qp``) and the deviatoric/shear modulus (``Qs``).
Longitudinal attenuation combines both, and the split is defined in the
run's own ``ndim`` dimensions. It is built on the same staggered
velocity-stress grid as :func:`deepwave.elastic`; only the stress
constitutive update changes.

**Physics.** The isotropic stress tensor is split into its mean (isotropic)
part and its deviatoric (shear) part::

    sigma_ii = K_iso * theta + 2 * mu * (eps_ii - theta / ndim)   (no sum)
    sigma_ij = 2 * mu_ij * eps_ij                                  (i != j)

where ``theta = div(v)`` (integrated to a strain), ``eps`` is the strain
tensor, ``K_iso = lamb + (2/ndim) * mu`` and ``ndim`` is 2 or 3. The mean
part relaxes with a GSLS mechanism set built from ``K_iso`` and ``Qp``; the
deviatoric part relaxes with an INDEPENDENT GSLS mechanism set built from
the shear modulus and ``Qs`` (see :func:`deepwave._gsls.sls_coefficients`,
the same constant-Q machinery :mod:`deepwave.viscoacoustic` uses for its
single scalar modulus, applied here to two elastic moduli separately). Each
of Qp and Qs gets its own set of ``q_n_mechanisms`` memory variables per
independent stress-tensor driving term: one shared mean-part memory
variable set (the mean/isotropic term is identical for every normal
stress), one deviatoric memory-variable set per normal-stress direction
(``ndim`` of them, since each direction's deviatoric strain-rate differs),
and one deviatoric memory-variable set per unique shear-stress pair
(``ndim * (ndim - 1) / 2`` of them).

For a memory variable ``r`` (any of the above) with Crank-Nicolson
coefficients ``(m1, c2, ar)`` (see :func:`deepwave._gsls.sls_coefficients`)
and driving strain-rate term ``S``::

    r_new = ar * r - c2 * S
    field += dt * m1 * S + (dt / 2) * (r + r_new)

(the memory-variable recursion itself is identical to
``viscoacoustic.update_pressure``'s; only the ``+`` sign
coupling it into the stress differs from pressure's ``-`` coupling, a
consequence of the elastic stress and acoustic pressure sign conventions
being opposite for the same physical divergence term -- see
:func:`update_stresses`). As ``Qp, Qs -> infinity``, every memory variable
decays to (and, started from zero, stays at) zero and ``m1`` reduces to the
relaxed modulus, so the stress update reduces EXACTLY to
``elastic.update_stresses``'s (verified to machine precision
by ``tests/test_viscoelastic.py``).

The GSLS constant-Q fit itself (log-spaced relaxation frequencies across
``q_band``, per-mechanism weights fit once by differentiable linear least
squares) is the SAME machinery :mod:`deepwave.viscoacoustic` uses (Emmerich
& Korn 1987; Blanch, Robertsson & Symes 1995; Carcione's textbook
treatment); the elastic velocity-stress formulation and the mean/deviatoric
attenuation split follow the standard viscoelastic literature (Robertsson,
Blanch & Symes 1994; Bohlen 2002, "Parallel 3-D viscoelastic finite
difference seismic modelling").

There is no compiled C/CUDA backend yet; both Python
modes (``"eager"`` and ``"torch_compile"`` -- ``torch.compile``) are
implemented, since the compiled mode only needs
``update_stresses``/``elastic.update_velocities``
themselves to be traceable, independent of any compiled kernel.
The default ``backend="auto"`` resolves to ``"eager"``, while
``backend="compiled"`` raises ``NotImplementedError`` rather than quietly
falling back to it; pass ``"torch_compile"`` explicitly to
opt into the (usually faster) alternative. Only 2-D and 3-D are
supported (shear waves require at least two spatial dimensions).

**Free surface.** A free surface is supported two ways, exactly as for
:func:`deepwave.elastic`: (1) the improved vacuum method (IVM) -- zero
``lamb``/``mu``/``buoyancy`` in an "air" region, which represents arbitrary/
topographic surfaces, and needs no special-casing here since every GSLS
coefficient is forced to exactly zero when its driving modulus is zero (see
:func:`deepwave._gsls.sls_coefficients`); or (2) the ``boundary=`` argument,
a fast flat-face image-method shortcut for a boundary lying exactly on a
domain edge.

**Free/rigid/periodic faces (``boundary=``).** Reuses
:func:`deepwave.elastic`'s own per-field guard-fill functions
(``_boundaries.fill_elastic_stress_boundary``/``fill_elastic_velocity_
boundary``) unchanged, since velocities are computed by the identical
``elastic.update_velocities`` and the stress tensor occupies
the same staggered-grid layout. The GSLS memory variables (mean and
deviatoric) are never guard-filled: no spatial derivative of any memory
variable appears in :func:`update_stresses` (each is a purely pointwise
recursion driven by the local strain-rate computed from the already
guard-filled velocities/stresses), so their ghost regions are never read --
see ``docs/boundaries.rst``. Note the mean-part memory variable ``r_mean``
is *shared* by every normal stress at a cell (see the module docstring
above), so it could not be zeroed on a free-surface plane the way
``sigma_AA`` is even if a spatial-derivative argument required it -- doing
so would corrupt the other, non-surface-normal, stresses that also read it.

See :doc:`/capabilities` for how this compares to the other 10 propagators
(backends, boundary faces, callbacks, differentiation, ...) at a glance.

See ``docs/viscoelastic.rst`` for the full derivation, including the
explicit algebraic reduction to ``elastic.update_stresses`` in
the ``Qp, Qs -> infinity`` limit.

Outputs: final wavefields (velocities, stresses, PML memory variables, GSLS
Qp/Qs memory variables) and receiver amplitudes (empty if no receivers).
All outputs are differentiable with respect to float ``torch.Tensor``
inputs.
"""

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave
from deepwave import _boundaries, _common, _engine, _gsls, _staggered_grid
from deepwave._exceptions import (
    DeepwaveNotImplementedError,
    DeepwaveValueError,
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
    LocationsLike,
    ReceiverSpec,
    RunInfo,
    SourceSpec,
    StorageLike,
    StorageMode,
    StoragePath,
)

# ``deepwave/__init__.py`` does ``from deepwave.elastic import Elastic,
# elastic``, which overwrites the ``elastic`` ATTRIBUTE of the ``deepwave``
# package namespace with the FUNCTION -- so both ``from deepwave import
# elastic`` and ``import deepwave.elastic as elastic`` (which also resolves
# via that same attribute) would silently bind the function, not the
# module. ``importlib.import_module`` looks up ``sys.modules`` directly,
# bypassing that shadowing, to get the actual submodule.
elastic: Any = importlib.import_module("deepwave.elastic")


def _dim_names(ndim: int) -> list[str]:
    """Returns the spatial dimension names (slowest to fastest) for ``ndim``."""
    return ["z", "y", "x"][-ndim:]


def _shear_pairs(ndim: int) -> list[tuple[int, int]]:
    """Unique shear-stress dimension pairs ``(i, j)``, ``i < j``.

    Matches ``elastic.elastic_model_names``'s
    ``mu_zy, mu_zx, mu_yx`` ordering (``z=0, y=1, x=2``): ``[(0, 1), (0, 2),
    (1, 2)]`` for 3-D, ``[(0, 1)]`` for 2-D.
    """
    return [(i, j) for i in range(ndim) for j in range(i + 1, ndim)]


def _ordered_pairs(ndim: int) -> list[tuple[int, int]]:
    """All ordered dimension pairs ``(i, j)``, ``i != j``.

    Used for the PML memory variables, which (unlike the physical shear
    stress itself) are NOT symmetric: the memory variable for ``d(v_i)/dj``
    and the one for ``d(v_j)/di`` are distinct fields (see
    ``elastic.update_stresses``/``update_velocities``).
    """
    return [(i, j) for i in range(ndim) for j in range(ndim) if i != j]


def viscoelastic_wavefield_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """Names of the viscoelastic wavefields, in backend order.

    The elastic block (velocities, stresses, and their PML memory
    variables) uses the SAME names, in the SAME backend order, as
    ``elastic.elastic_wavefield_names`` (this propagator
    reuses that exact field layout so it can reuse ``elastic.py``'s
    validated per-field half-grid/PML tables and velocity update
    unchanged). The GSLS Qp/Qs memory variables follow: ``rp_0`` ..
    ``rp_{n_mechanisms-1}`` (the shared mean/Qp memory variables), then
    ``rs_<dim>_0`` .. for each normal direction (deviatoric/Qs, one set per
    direction), then ``rs_<dim1><dim2>_0`` .. for each unique shear pair
    (deviatoric/Qs, one set per pair).
    """
    dims = _dim_names(ndim)
    pairs = _shear_pairs(ndim)
    names = list(elastic.elastic_wavefield_names(ndim))
    names += [f"rp_{mech}" for mech in range(n_mechanisms)]
    for d in dims:
        names += [f"rs_{d}_{mech}" for mech in range(n_mechanisms)]
    for i, j in pairs:
        tag = f"{dims[i]}{dims[j]}"
        names += [f"rs_{tag}_{mech}" for mech in range(n_mechanisms)]
    return names


def viscoelastic_model_names(ndim: int) -> list[str]:
    """Names of the viscoelastic kernel model fields.

    ``lamb``, ``mu`` and the staggered shear moduli (``mu_zy``/``mu_zx``/
    ``mu_yx``) are identical to ``elastic.elastic_model_names``.
    ``m1p``/``c2p``/``arp`` are the shared mean/Qp GSLS coefficients (built
    from ``K_iso = lamb + (2/ndim) * mu``); ``m1s``/``c2s``/``ars`` are the
    UNSTAGGERED deviatoric/Qs GSLS coefficients (built from ``2 * mu``, used
    by the normal-stress deviatoric term); ``m1s_<dim1><dim2>``/
    ``c2s_<dim1><dim2>``/``ars_<dim1><dim2>`` are the STAGGERED deviatoric/Qs
    coefficients per shear pair (built from the staggered shear modulus
    ``mu_<dim1><dim2>`` and a matching harmonic-mean-staggered ``qs``). The
    buoyancies (one per velocity component) are last, as in
    ``elastic.elastic_model_names``.
    """
    dims = _dim_names(ndim)
    pairs = _shear_pairs(ndim)
    names = ["lamb", "mu"]
    names += [f"mu_{dims[i]}{dims[j]}" for i, j in pairs]
    names += ["m1p", "c2p", "arp", "m1s", "c2s", "ars"]
    for i, j in pairs:
        tag = f"{dims[i]}{dims[j]}"
        names += [f"m1s_{tag}", f"c2s_{tag}", f"ars_{tag}"]
    names += [f"buoyancy_{d}" for d in dims]
    return names


# The user-facing model arguments, in the order the public function takes
# them; the GSLS coefficients above are all derived from these five.
USER_MODEL_NAMES = ("lamb", "mu", "buoyancy", "qp", "qs")


def _elastic_block_order(ndim: int) -> tuple[list[int], list[int]]:
    """The elastic block's backend->public permutation, split in two.

    Returns the physical fields' indices (velocities and stresses, in the
    elastic family's grouped-by-kind public order) and the
    absorbing-boundary memory variables' indices, separately: the GSLS
    memory block is published between them, so the two halves do not stay
    adjacent here as they do in :func:`deepwave.elastic`.

    ``elastic.elastic_public_wavefield_names`` is the single
    definition of that permutation; this recovers the index list from it (the
    names are unique, so the reconstruction is exact) rather than restating
    the table here, where it could drift. The elastic block is the leading
    ``len(elastic_wavefield_names(ndim))`` entries of
    :func:`viscoelastic_wavefield_names`, so these indices are equally valid
    in the viscoelastic list.
    """
    backend = elastic.elastic_wavefield_names(ndim)
    fold = set(elastic.elastic_auxiliary_state_names(ndim))
    physical: list[int] = []
    memory: list[int] = []
    for name in elastic.elastic_public_wavefield_names(ndim):
        (memory if name in fold else physical).append(backend.index(name))
    return physical, memory


def _gsls_block_order(ndim: int, n_mechanisms: int) -> list[tuple[int, str]]:
    """The GSLS memory block in public order, as ``(backend index, name)``.

    The backend groups this block by **kind**, each kind's mechanisms
    contiguous (``rp_0, rp_1, rs_y_0, rs_y_1, ...``); the public order groups
    it by **mechanism** (``rp_mech0, rs_y_mech0, ..., rp_mech1, ...``), which
    is the transpose of that ``(kind, mechanism)`` layout. At one mechanism
    the two coincide, so the transpose is only observable at
    ``n_mechanisms > 1``.

    The layout is read off :func:`viscoelastic_wavefield_names` itself rather
    than computed from assumed positions, so it cannot drift from the real
    backend order. The public spelling marks the mechanism index explicitly
    (the backend's bare ``_0`` reads like the *time* index every elastic name
    carries) and writes each shear pair's dimensions in the same order as the
    corresponding stress (internal ``rs_yx`` pairs with ``sigmaxy``, so it is
    published as ``rs_xy``).
    """
    names = viscoelastic_wavefield_names(ndim, n_mechanisms)
    start = len(elastic.elastic_wavefield_names(ndim))
    dims = _dim_names(ndim)
    pair_renames = {
        f"rs_{dims[i]}{dims[j]}": f"rs_{dims[j]}{dims[i]}"
        for i, j in _shear_pairs(ndim)
    }
    kinds: list[str] = []
    index: dict[tuple[str, str], int] = {}
    for offset, name in enumerate(names[start:]):
        kind, mech = name.rsplit("_", 1)
        if kind not in kinds:
            kinds.append(kind)
        index[(kind, mech)] = start + offset
    ordered = [
        (index[(kind, str(mech))], f"{pair_renames.get(kind, kind)}_mech{mech}")
        for mech in range(n_mechanisms)
        for kind in kinds
    ]
    if sorted(i for i, _ in ordered) != list(range(start, len(names))):
        raise AssertionError(
            "the viscoelastic GSLS block regrouping is not a permutation of "
            f"{names[start:]}; the backend layout is not the assumed "
            "kind-major one."
        )
    return ordered


def viscoelastic_public_wavefield_order(ndim: int, n_mechanisms: int = 1) -> list[int]:
    """Indices reordering the backend wavefields into the public return order.

    Two transformations compose: the elastic family's regrouping of the
    physical fields by kind (:func:`_elastic_block_order`), and the GSLS
    memory block's regrouping by mechanism (:func:`_gsls_block_order`). The
    GSLS block is published between the physical fields and the
    absorbing-boundary memory, so that the collapsed ``auxiliary_state``
    lands after it. The same list orders
    :func:`viscoelastic_public_wavefield_names` and the output tensors
    :func:`viscoelastic` returns, so the two cannot disagree.
    """
    physical, memory = _elastic_block_order(ndim)
    return [
        *physical,
        *(i for i, _ in _gsls_block_order(ndim, n_mechanisms)),
        *memory,
    ]


def viscoelastic_state_name_renames(ndim: int, n_mechanisms: int = 1) -> dict[str, str]:
    """Public spellings of the GSLS memory-variable state fields.

    Every elastic-block field maps to itself; each GSLS memory variable gains
    an explicit ``_mech{i}`` marker and, for the shear pairs, the dimension
    order of the stress it belongs to. See :func:`_gsls_block_order`.
    """
    backend = viscoelastic_wavefield_names(ndim, n_mechanisms)
    return {backend[i]: public for i, public in _gsls_block_order(ndim, n_mechanisms)}


def viscoelastic_public_wavefield_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """The viscoelastic state fields in public return order and spelling.

    The physical fields come first, in the elastic family's grouped-by-kind
    order, then the GSLS memory block grouped by mechanism. The
    auxiliary-state collapse is computed in *this* order, since that is where
    the index of the first auxiliary field is meaningful.
    """
    backend = viscoelastic_wavefield_names(ndim, n_mechanisms)
    renames = viscoelastic_state_name_renames(ndim, n_mechanisms)
    return [
        renames.get(backend[i], backend[i])
        for i in viscoelastic_public_wavefield_order(ndim, n_mechanisms)
    ]


def viscoelastic_auxiliary_state_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The GSLS attenuation memory variables ``r{p,s}_*_mech{i}``, then the
    CFS-CPML memory variables ``m_*``: this propagator reuses
    ``elastic.elastic_wavefield_names``' field layout verbatim for its
    physical block, so its absorbing-boundary set is exactly
    ``elastic.elastic_auxiliary_state_names``, in the same public order.

    Both blocks are implementation state whose identity, count and shapes are
    properties of the attenuation and absorbing-boundary implementations
    rather than of the physics the caller asked for, so both are carried in
    the single opaque :class:`deepwave.AuxiliaryState` rather than as public
    output fields.

    Listed in the public order
    (:func:`viscoelastic_public_wavefield_names`), which is the order the
    auxiliary state itself holds them in. The GSLS block is the reason this
    list depends on ``n_mechanisms``.
    """
    return [public for _, public in _gsls_block_order(ndim, n_mechanisms)] + list(
        elastic.elastic_auxiliary_state_names(ndim)
    )


def viscoelastic_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`viscoelastic`.

    Pressure first, then one per velocity component (z, y, x, for the
    dimensions present).
    """
    return ["receiver_amplitudes_p"] + [
        f"receiver_amplitudes_{d}" for d in _dim_names(ndim)
    ]


def viscoelastic_output_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """Names of the public outputs of :func:`viscoelastic`, in return order.

    The public final wavefields -- the velocities and stresses in the elastic
    family's grouped-by-kind order, followed by the single
    ``auxiliary_state`` that replaces the block of GSLS and PML memory
    variables -- followed by the receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            viscoelastic_public_wavefield_names(ndim, n_mechanisms),
            viscoelastic_auxiliary_state_names(ndim, n_mechanisms),
        ),
        *viscoelastic_receiver_names(ndim),
    ]


# Declarative description of the viscoelastic propagator. Same staggered-grid
# layout as elastic, plus the GSLS Qp/Qs memory variables and their model
# fields. No compiled backend exists yet (see the module docstring), so
# c_model_flag_indices (only consumed by C-backend dispatch) is left unset.
SPEC = _engine.PropagatorSpec(
    name="viscoelastic",
    status="experimental",
    ndims=(2, 3),
    source_channels=_engine.pressure_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    supported_options=frozenset(
        {"mpml_fraction", "q_freq", "q_n_mechanisms", "q_band"}
    ),
    staggered=True,
    model_names=viscoelastic_model_names,
    wavefield_names=viscoelastic_wavefield_names,
    n_source_types=lambda ndim: ndim + 1,
    n_receiver_types=lambda ndim: ndim + 1,
    output_names=viscoelastic_output_names,
    auxiliary_state_names=viscoelastic_auxiliary_state_names,
    public_wavefield_names=viscoelastic_public_wavefield_names,
    state_name_renames=viscoelastic_state_name_renames,
    spatial_parallel=True,
    has_compiled_backend=False,
)


def _stagger_shear_harmonic(
    field: torch.Tensor, mu: torch.Tensor, ndim: int
) -> list[torch.Tensor]:
    """Harmonic-mean-stagger ``field`` onto the shear-stress nodes.

    Uses the SAME 4-point harmonic mean and vacuum mask (``mu`` non-zero) as
    ``elastic.prepare_parameters`` uses to stagger ``mu`` itself
    onto ``mu_zy``/``mu_zx``/``mu_yx``; used here to bring ``qs`` onto those
    same nodes so its GSLS relaxation defect pairs with the correctly
    staggered shear modulus at every shear-stress grid point.

    Unlike ``mu``'s own staggering, cells outside the mask (a vacuum
    neighbour) are filled with a SAFE placeholder (``1.0``, not zero):
    ``qs`` is a quality factor, not a modulus, so it has no physical reason
    to be zero in vacuum, and a literal zero would make the GSLS relaxation
    defect ``weights / qs`` infinite; the corresponding shear modulus
    (already correctly zero-masked by ``elastic.prepare_parameters``) makes
    the final GSLS coefficient exactly zero there regardless of this
    placeholder's value, so the placeholder is harmless.
    """
    rfmax = 1 / torch.finfo(field.dtype).max ** (1 / 2)
    field_safe = torch.where(field.abs() > rfmax, field, torch.ones_like(field))
    out: list[torch.Tensor] = []
    if ndim >= 3:
        mask = (
            (rfmax < mu[..., 1:, 1:, :].abs())
            .logical_and(rfmax < mu[..., :-1, :-1, :].abs())
            .logical_and(rfmax < mu[..., 1:, :-1, :].abs())
            .logical_and(rfmax < mu[..., :-1, 1:, :].abs())
        )
        val = 4 / (
            1 / field_safe[..., 1:, 1:, :]
            + 1 / field_safe[..., :-1, :-1, :]
            + 1 / field_safe[..., 1:, :-1, :]
            + 1 / field_safe[..., :-1, 1:, :]
        )
        zy = torch.where(mask, val, torch.ones_like(val))
        zy = torch.nn.functional.pad(zy, (0, 0, 0, 1, 0, 1), value=1.0)
        out.append(zy)
        del mask, val, zy
        mask = (
            (rfmax < mu[..., 1:, :, 1:].abs())
            .logical_and(rfmax < mu[..., :-1, :, :-1].abs())
            .logical_and(rfmax < mu[..., 1:, :, :-1].abs())
            .logical_and(rfmax < mu[..., :-1, :, 1:].abs())
        )
        val = 4 / (
            1 / field_safe[..., 1:, :, 1:]
            + 1 / field_safe[..., :-1, :, :-1]
            + 1 / field_safe[..., 1:, :, :-1]
            + 1 / field_safe[..., :-1, :, 1:]
        )
        zx = torch.where(mask, val, torch.ones_like(val))
        zx = torch.nn.functional.pad(zx, (0, 1, 0, 0, 0, 1), value=1.0)
        out.append(zx)
        del mask, val, zx
    if ndim >= 2:
        mask = (
            (rfmax < mu[..., 1:, 1:].abs())
            .logical_and(rfmax < mu[..., :-1, :-1].abs())
            .logical_and(rfmax < mu[..., 1:, :-1].abs())
            .logical_and(rfmax < mu[..., :-1, 1:].abs())
        )
        val = 4 / (
            1 / field_safe[..., 1:, 1:]
            + 1 / field_safe[..., :-1, :-1]
            + 1 / field_safe[..., 1:, :-1]
            + 1 / field_safe[..., :-1, 1:]
        )
        yx = torch.where(mask, val, torch.ones_like(val))
        yx = torch.nn.functional.pad(yx, (0, 1, 0, 1), value=1.0)
        out.append(yx)
    return out


def prepare_models(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    qp: torch.Tensor,
    qs: torch.Tensor,
    dt: float,
    omega0: float,
    q_n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
) -> list[torch.Tensor]:
    """Prepares the kernel model fields for the viscoelastic propagator.

    Maps the user models ``(lamb, mu, buoyancy, Qp, Qs)`` to the fields the
    kernel consumes: ``lamb``, ``mu`` and the staggered shear moduli
    unchanged (``elastic.prepare_parameters``); the shared
    mean/Qp GSLS coefficients ``(m1p, c2p, arp)`` built from ``K_iso =
    lamb + (2/ndim) * mu`` and ``qp``; the unstaggered deviatoric/Qs GSLS
    coefficients ``(m1s, c2s, ars)`` built from ``2 * mu`` and ``qs``; one
    staggered deviatoric/Qs GSLS coefficient triple per shear pair, built
    from the staggered shear modulus and a matching harmonic-mean-staggered
    ``qs`` (:func:`_stagger_shear_harmonic`); and the buoyancies (last, as
    in ``elastic.elastic_model_names``).

    See :func:`deepwave._gsls.sls_coefficients` for the ``(m1, c2, ar)``
    formulas and :func:`update_stresses` for how they combine into the
    stress update.

    Args:
        lamb: First Lame parameter (relaxed).
        mu: Second Lame parameter / shear modulus (relaxed).
        buoyancy: Buoyancy (1 / density).
        qp: Quality factor for the mean/bulk (P-wave-controlling) part of
            the stress tensor, at ``omega0`` (``q_n_mechanisms == 1``) or
            the target constant-Q value across ``q_band`` (``q_n_mechanisms
            > 1``).
        qs: Quality factor for the deviatoric/shear (S-wave-controlling)
            part of the stress tensor, same convention as ``qp``.
        dt: Internal time step (after any CFL sub-stepping).
        omega0: Reference angular frequency (``q_n_mechanisms == 1`` only).
        q_n_mechanisms: Number of GSLS relaxation mechanisms (shared by Qp
            and Qs).
        q_band: ``(f1, f2)`` frequency band for the GSLS constant-Q fit;
            required when ``q_n_mechanisms > 1``.

    Returns:
        The flat model list matching :func:`viscoelastic_model_names`.
    """
    ndim = lamb.ndim - 1
    rfmax = 1 / torch.finfo(lamb.dtype).max ** (1 / 2)

    staggered = elastic.prepare_parameters(mu, buoyancy)
    n_shear = ndim * (ndim - 1) // 2
    mu_shear = staggered[:n_shear]
    buoyancies = staggered[n_shear:]
    del staggered

    def _q_safe(q: torch.Tensor) -> torch.Tensor:
        mask = q.abs() > rfmax
        return torch.where(mask, q, torch.ones_like(q))

    qp_safe = _q_safe(qp)
    qs_safe = _q_safe(qs)

    k_iso = lamb + (2.0 / ndim) * mu
    m1p, c2p, arp = _gsls.sls_coefficients(
        k_iso, qp_safe, dt, omega0, q_n_mechanisms, q_band
    )
    m1s, c2s, ars = _gsls.sls_coefficients(
        2.0 * mu, qs_safe, dt, omega0, q_n_mechanisms, q_band
    )

    qs_stag = _stagger_shear_harmonic(qs_safe, mu, ndim)
    shear_gsls: list[torch.Tensor] = []
    for mu_ij, qs_ij in zip(mu_shear, qs_stag):
        m1_ij, c2_ij, ar_ij = _gsls.sls_coefficients(
            mu_ij, qs_ij, dt, omega0, q_n_mechanisms, q_band
        )
        shear_gsls.extend([m1_ij, c2_ij, ar_ij])

    return [
        lamb,
        mu,
        *mu_shear,
        m1p,
        c2p,
        arp,
        m1s,
        c2s,
        ars,
        *shear_gsls,
        *buoyancies,
    ]


def _resolve_q_freq(q_freq: Optional[float]) -> float:
    """The reference frequency for the Qp/Qs models, as a float.

    ``q_freq`` is required when ``q_n_mechanisms == 1``, and refused above it:
    a multi-mechanism run fits a constant Qp/Qs across ``q_band``, so there is
    no single reference frequency, and its coefficients never read one. Both
    cases have already been checked by the time this is called, so ``None``
    here can only be the multi-mechanism case; it becomes 0.0, which no
    coefficient reads, rather than a frequency that could be mistaken for a
    default.

    It is never taken from ``pml_freq``: the PML's tuning frequency and the
    frequency at which Qp/Qs were measured are different physical quantities
    that happen to be similar numbers in a seismic survey, and letting one
    default to the other made the attenuation model depend on the wavelet.
    """
    return float(q_freq) if q_freq is not None else 0.0


def _gsls_accumulate(
    s: torch.Tensor,
    m1: torch.Tensor,
    c2: torch.Tensor,
    ar: torch.Tensor,
    r_list: list[torch.Tensor],
    dt: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One GSLS Crank-Nicolson step for a stress-sign-convention field.

    Returns ``(dt * m1 * s + (dt / 2) * sum_l(r_old_l + r_new_l),
    new_r_list)``, generalising ``viscoacoustic.update_pressure``'s
    per-mechanism recursion (``r_new_l = ar_l * r_old_l -
    c2_l * s``, UNCHANGED here) to an arbitrary number of independent memory
    variable GROUPS (mean, one per normal direction, one per shear pair);
    the caller adds the returned contribution to the stress it attenuates
    with a ``+`` sign (see the module docstring for why this differs from
    viscoacoustic's pressure, which uses ``-``).
    """
    n_mech = len(r_list)
    if n_mech == 1:
        r_new = ar * r_list[0] - c2 * s
        contribution = dt * m1 * s + (dt / 2) * (r_list[0] + r_new)
        return contribution, [r_new]
    new_r_list: list[torch.Tensor] = []
    sum_r = torch.zeros_like(s)
    for i in range(n_mech):
        r_old = r_list[i]
        r_new_i = ar[i] * r_old - c2[i] * s
        new_r_list.append(r_new_i)
        sum_r = sum_r + r_old + r_new_i
    contribution = dt * m1 * s + (dt / 2) * sum_r
    return contribution, new_r_list


# The velocity update has no Qp/Qs dependence at all (attenuation only
# enters the stress constitutive law); reuse elastic's implementation
# unchanged rather than duplicating it.
update_velocities = elastic.update_velocities


def update_stresses(
    models: list[torch.Tensor],
    velocities: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    v_mem_vars_normal: list[torch.Tensor],
    v_mem_vars_shear: list[list[torch.Tensor]],
    r_mean: list[torch.Tensor],
    r_dev_normal: list[list[torch.Tensor]],
    r_dev_shear: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
]:
    """Updates the stress wavefields, their PML memory variables, and Qp/Qs.

    Identical to ``elastic.update_stresses`` through the
    velocity-derivative/PML-correction stage (both derivative computation
    and the M-PML memory-variable bookkeeping are unchanged); only the
    constitutive combination into ``normal_stresses``/``shear_stresses``
    differs, splitting the elastic ``lamb * v_strain_sum + 2 * mu *
    dvxdx[dim]`` term into a mean part (``K_iso * v_strain_sum``,
    attenuated via ``r_mean``/Qp) and a deviatoric part (``2 * mu *
    (dvxdx[dim] - v_strain_sum / ndim)``, attenuated via
    ``r_dev_normal[dim]``/Qs), and the shear term ``mu_ij * (dvxdy[i][j] +
    dvxdy[j][i])`` into a purely deviatoric term attenuated via
    ``r_dev_shear[k]``/Qs (``k`` indexing :func:`_shear_pairs`). See
    ``docs/viscoelastic.rst`` for the derivation, including the algebraic
    verification that this reduces EXACTLY to elastic's update as
    ``Qp, Qs -> infinity``.
    """
    ndim = len(grid_spacing)

    dvxdx = [
        _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

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

    n_shear = ndim * (ndim - 1) // 2
    m1p, c2p, arp = models[2 + n_shear], models[3 + n_shear], models[4 + n_shear]
    m1s, c2s, ars = models[5 + n_shear], models[6 + n_shear], models[7 + n_shear]
    shear_gsls = models[8 + n_shear : 8 + n_shear + 3 * n_shear]

    v_strain_sum = torch.sum(torch.stack(dvxdx), dim=0)

    mean_contribution, r_mean_new = _gsls_accumulate(
        v_strain_sum, m1p, c2p, arp, r_mean, dt
    )

    new_normal_stresses: list[torch.Tensor] = []
    new_r_dev_normal: list[list[torch.Tensor]] = []
    for dim in range(ndim):
        dev = dvxdx[dim] - v_strain_sum / ndim
        contribution, r_new_list = _gsls_accumulate(
            dev, m1s, c2s, ars, r_dev_normal[dim], dt
        )
        new_normal_stresses.append(
            normal_stresses[dim] + mean_contribution + contribution
        )
        new_r_dev_normal.append(r_new_list)
    normal_stresses = new_normal_stresses

    new_r_dev_shear: list[list[torch.Tensor]] = []
    if ndim > 1:
        pairs = _shear_pairs(ndim)
        for k, (i, j) in enumerate(pairs):
            m1s_ij = shear_gsls[3 * k]
            c2s_ij = shear_gsls[3 * k + 1]
            ars_ij = shear_gsls[3 * k + 2]
            s_shear = dvxdy[i][j] + dvxdy[j][i]
            contribution, r_new_list = _gsls_accumulate(
                s_shear, m1s_ij, c2s_ij, ars_ij, r_dev_shear[k], dt
            )
            new_val = shear_stresses[i][j] + contribution
            shear_stresses[i][j] = new_val
            shear_stresses[j][i] = new_val
            new_r_dev_shear.append(r_new_list)

    return (
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
        r_mean_new,
        new_r_dev_normal,
        new_r_dev_shear,
    )


# Module-level indirection selected by viscoelastic_func: eager (default)
# or "compile" (torch.compile), mirroring
# deepwave.elastic's identical _update_velocities_opt/_update_stresses_opt
# pattern. update_velocities never changes (it has no Qp/Qs dependence, see
# above); only update_stresses (viscoelastic's own GSLS-augmented kernel)
# has a compiled variant.
_update_velocities_compile: Optional[Any] = None
_update_velocities_opt: Any = update_velocities
_update_stresses_compile: Optional[Any] = None
_update_stresses_opt: Any = update_stresses


def zero_edges_and_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    n_mechanisms: int = 1,
    mpml: bool = False,
) -> None:
    """Zeros the edges and/or interiors of wavefields.

    The elastic-block prefix of ``wavefields`` (see
    :func:`viscoelastic_wavefield_names`) uses the EXACT
    ``half_grid``/``pml_edge_idx`` tables ``elastic.zero_edges_and_interiors``
    uses (reproduced here verbatim -- they are a
    private table internal to that function, not otherwise importable --
    since this propagator deliberately mirrors elastic's field layout/order
    for that block). The appended GSLS Qp/Qs memory variables are none of
    them M-PML cross-damping variables (so none are added to
    ``pml_edge_idx``) and are co-located with the stress component they
    attenuate: the mean (``rp``) and per-direction deviatoric-normal
    (``rs_<dim>``) memory variables sit on the SAME (unstaggered) grid as
    the normal stresses; the per-pair deviatoric-shear (``rs_<dim1><dim2>``)
    memory variables sit on the SAME (doubly-staggered) grid as their shear
    stress.
    """
    num_vars = len(wavefields)
    half_grid: list[list[bool]] = [[False] * ndim for _ in range(num_vars)]
    if ndim == 3:
        for i in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
            half_grid[i][0] = True
        for i in [2, 5, 7, 12, 14, 16, 18, 19, 20, 22]:
            half_grid[i][1] = True
        for i in [3, 6, 8, 13, 16, 18, 19, 21, 23, 26]:
            half_grid[i][2] = True
        pml_edge_idx = [
            [4, 7, 8, 9, 12, 13],
            [5, 10, 17, 19, 20, 21],
            [6, 11, 18, 22, 25, 26],
        ]
        n_elastic = 27
    elif ndim == 2:
        for i in [0, 2, 4, 5, 6, 8]:
            half_grid[i][0] = True
        for i in [2, 4, 5, 7, 9, 12]:
            half_grid[i][1] = True
        pml_edge_idx = [[3, 5, 6, 7], [4, 8, 11, 12]]
        n_elastic = 13
    else:
        raise DeepwaveValueError("viscoelastic: only ndim in (2, 3) is supported.")

    pairs = _shear_pairs(ndim)
    idx = n_elastic + n_mechanisms  # skip rp (never half-grid)
    for _ in range(ndim):
        idx += n_mechanisms  # rs_<dim> groups (never half-grid)
    for i, j in pairs:
        for _ in range(n_mechanisms):
            half_grid[idx][i] = True
            half_grid[idx][j] = True
            idx += 1

    _staggered_grid.zero_edges_and_interiors(
        wavefields,
        ndim,
        fd_pad,
        fd_pad_list,
        pml_width,
        half_grid,
        pml_edge_idx,
        interior=False,
        mpml=mpml,
    )


class Viscoelastic(torch.nn.Module):
    """Convenience nn.Module wrapper for viscoelastic wave propagation.

    Stability: experimental (see :doc:`/stability`).

    Stores ``lamb``, ``mu``, ``buoyancy``, ``qp``, ``qs`` and
    ``grid_spacing`` once, so that each :meth:`forward` call needs only the
    arguments that vary between calls.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        lamb: The first Lame parameter model (relaxed), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        mu: The second Lame parameter / shear modulus model (relaxed), of
            shape ``[*model_shape]`` or, for a batched model,
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
        qp: The quality factor model for the mean/bulk part of the stress
            tensor, of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        qs: The quality factor model for the deviatoric/shear part of the
            stress tensor, of shape ``[*model_shape]`` or, for a batched
            model, ``[n_shots, *model_shape]``, where ``model_shape`` has
            ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size (a single number, or
            one per dimension).
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
        lamb_requires_grad: A bool specifying whether gradients will be
            computed for `lamb`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        mu_requires_grad: A bool specifying whether gradients will be
            computed for `mu`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        buoyancy_requires_grad: A bool specifying whether gradients will be
            computed for `buoyancy`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        qp_requires_grad: A bool specifying whether gradients will be
            computed for `qp`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        qs_requires_grad: A bool specifying whether gradients will be
            computed for `qs`. Unset by default, which means False. Cannot
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
            Only `"device"` is supported on the eager Python
            backend.
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
            Not supported on the eager Python backend.
    """

    def __init__(
        self,
        lamb: torch.Tensor,
        mu: torch.Tensor,
        buoyancy: torch.Tensor,
        qp: torch.Tensor,
        qs: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        lamb_requires_grad: Optional[bool] = None,
        mu_requires_grad: Optional[bool] = None,
        buoyancy_requires_grad: Optional[bool] = None,
        qp_requires_grad: Optional[bool] = None,
        qs_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        names = ["lamb", "mu", "buoyancy", "qp", "qs"]
        grad = _common.resolve_requires_grad(
            requires_grad,
            names,
            names,
            {
                "lamb": lamb_requires_grad,
                "mu": mu_requires_grad,
                "buoyancy": buoyancy_requires_grad,
                "qp": qp_requires_grad,
                "qs": qs_requires_grad,
            },
            "Viscoelastic",
        )
        self.lamb = _common.make_model_parameter(lamb, grad["lamb"], "lamb")
        self.mu = _common.make_model_parameter(mu, grad["mu"], "mu")
        self.buoyancy = _common.make_model_parameter(
            buoyancy, grad["buoyancy"], "buoyancy"
        )
        self.qp = _common.make_model_parameter(qp, grad["qp"], "qp")
        self.qs = _common.make_model_parameter(qs, grad["qs"], "qs")
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
        q_freq: Optional[float] = None,
        q_n_mechanisms: int = 1,
        q_band: Optional[tuple[float, float]] = None,
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

        See :func:`viscoelastic` for details (the constructor-owned `lamb`/
        `mu`/`buoyancy`/`qp`/`qs`/`grid_spacing` and the storage settings are
        supplied from the stored module state).
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
        return viscoelastic(
            self.lamb,
            self.mu,
            self.buoyancy,
            self.qp,
            self.qs,
            self.grid_spacing,
            dt,
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
            q_freq=q_freq,
            q_n_mechanisms=q_n_mechanisms,
            q_band=q_band,
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


def viscoelastic(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    qp: torch.Tensor,
    qs: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
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
    q_freq: Optional[float] = None,
    q_n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
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
    """Viscoelastic (elastic + Qp/Qs GSLS attenuation) wave propagation.

    Stability: experimental (see :doc:`/stability`).

    Forward modelling with the isotropic elastic velocity-stress wave
    equation plus independent GSLS attenuation of the mean (Qp) and
    deviatoric (Qs) parts of the stress tensor, on a staggered grid. See the
    module docstring for the equations. Only 2-D and 3-D are supported
    (shear waves require at least two spatial dimensions). There is no
    compiled C/CUDA backend: the default ``backend="auto"`` resolves to
    ``"eager"``, ``backend="compiled"`` raises ``NotImplementedError``
    rather than quietly falling back to it, and
    ``"torch_compile"`` is also available.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        lamb: The first Lame parameter model (the relaxed / low-frequency
            value), of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        mu: The second Lame parameter / shear modulus model (relaxed), of
            shape ``[*model_shape]`` or, for a batched model,
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
        qp: The quality factor model for the mean/bulk part of the stress
            tensor, of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. It is given at
            the reference angular frequency ``omega0 = 2*pi*q_freq``
            (``q_n_mechanisms == 1``) or the target constant-Q value across
            ``q_band`` (``q_n_mechanisms > 1``). A differentiable model
            input. Larger ``Qp`` means weaker P-wave attenuation; ``Qp,
            Qs -> inf`` recovers the lossless elastic equation.
        qs: The quality factor model for the deviatoric/shear part of the
            stress tensor, of shape ``[*model_shape]`` or, for a batched
            model, ``[n_shots, *model_shape]``, where ``model_shape`` has
            ``ndim`` entries. See :ref:`batched-models`. It follows the same
            convention as ``qp``. Larger ``Qs`` means weaker
            S-wave attenuation.
        grid_spacing: Spatial grid cell size.
        dt: Time step interval. See :func:`deepwave.elastic` for the
            unit-consistency requirement across ``lamb``/``mu``/``buoyancy``/
            ``grid_spacing``/``dt`` (``qp``/``qs``/``q_freq`` have their own
            units documented above); see :doc:`/units`.
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
        source_amplitudes_pressure_rate: Pressure-rate source amplitudes (units of
            pressure/time), added identically to every normal stress (see
            :func:`deepwave.elastic` and :doc:`/units`).
        source_locations_pressure_rate: Pressure source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_z: Z-force-density source amplitudes (units of
            force/volume, see :func:`deepwave.elastic`). This channel exists
            only in 3-D.
        source_locations_z: Z-force source locations. This channel exists only
            in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_y: Y-force-density source amplitudes.
        source_locations_y: Y-force source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_x: X-force-density source amplitudes.
        source_locations_x: X-force source locations.
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
        receiver_locations_p: Pressure receiver locations (records the
            negative mean of the normal stresses, see
            :func:`deepwave.elastic`). In 2-D the recorded pressure is the
            in-plane mean normal stress ``-(sigma_xx + sigma_yy)/2``, which
            under plane strain is not the thermodynamic pressure; see
            :func:`deepwave.elastic`'s ``receiver_amplitudes_p`` entry for
            the full statement.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_z: Z-velocity receiver locations. This channel
            exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_y: Y-velocity receiver locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_x: X-velocity receiver locations.
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
        vz_0: Initial z-velocity (time t=-0.5). Like every initial
            wavefield, it may be provided either at the shape of the
            simulation domain (the extracted model region plus the PML),
            which is the shape a previous call returns, or at the shape of
            the extracted model region itself, which Deepwave zero-pads for
            you; see the `wavefield_0` entry of :func:`deepwave.scalar`.
        vy_0: Initial y-velocity (time t=-0.5).
        vx_0: Initial x-velocity (time t=-0.5).
        sigmazz_0: Initial sigma_zz (time t=0).
        sigmayz_0: Initial sigma_yz (time t=0).
        sigmaxz_0: Initial sigma_xz (time t=0).
        sigmayy_0: Initial sigma_yy (time t=0).
        sigmaxy_0: Initial sigma_xy (time t=0).
        sigmaxx_0: Initial sigma_xx (time t=0).
        initial_state: See :func:`deepwave.elastic`. The mapping keys are
            this propagator's own public state field names (`vy_0`, ...,
            `sigmaxx_0`) and `auxiliary_state`. The GSLS memory variables
            are carried inside `auxiliary_state`, because how many there are
            depends on `q_n_mechanisms`, so an initial relaxation state is
            supplied by passing a previous call's state back, most simply as
            `initial_state=out`. Neither they (`rp_mech0`, `rs_y_mech0`,
            ...) nor the individual absorbing-boundary field names
            (`m_vyy_0`, ..., `m_sigmaxxx_0`) are accepted as keys.
        accuracy: See :func:`deepwave.elastic`.
        pml_width: See :func:`deepwave.elastic`.
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: See :func:`deepwave.elastic` -- the same first-class
            flat-face ``"pml"``/``"free"``/``"rigid"``/``"periodic"``
            mechanism, reusing the identical eager guard-fill functions
            (``_boundaries.fill_elastic_stress_boundary``/
            ``fill_elastic_velocity_boundary``). The GSLS Qp/Qs memory
            variables need no guard-fill of their own (see the module
            docstring's "Free/rigid/periodic faces" section). Only ``"pml"``
            everywhere (``boundary=None``) and the vacuum-cell improved
            vacuum method (IVM, via zero ``lamb``/``mu``/``buoyancy``) are
            supported for topographic/irregular surfaces; this argument only
            covers a flat face lying exactly on a domain edge, same as
            :func:`deepwave.elastic`. There is no ``surface_method="pm"``
            analogue yet.
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
        mpml_fraction: The multiaxial PML (M-PML) cross-damping fraction (see
            :func:`deepwave.elastic`). Defaults to 0.0 (standard C-PML).
            Must satisfy ``0 <= mpml_fraction < 1``.
        q_freq: Reference frequency at which ``Qp``/``Qs`` are specified, in
            the same units as ``1 / dt``. **Required** when
            ``q_n_mechanisms == 1`` (the default), and must be ``None`` when
            ``q_n_mechanisms > 1``, where the constant-``Q`` fit spans
            ``q_band`` instead and no single reference frequency applies.
            There is no default: ``Qp``/``Qs`` have no meaning without the
            frequency they were measured at, and it is never taken from
            ``pml_freq``, which tunes the absorbing boundary and is a
            different physical quantity (it is also inferred from the
            wavelet, which would have made the attenuation model depend on
            the source). ``omega0 = 2*pi*q_freq``. It must be a positive,
            finite number (a one-element tensor or a NumPy scalar is
            accepted); anything else raises.
        q_n_mechanisms: Number of generalised-standard-linear-solid (GSLS)
            relaxation mechanisms, shared by Qp and Qs. Defaults to 1 (a
            single SLS mechanism per Qp/Qs; ``q_band`` must be ``None``).
            For ``q_n_mechanisms > 1``, ``q_band=(f1, f2)`` is required:
            ``q_n_mechanisms`` relaxation frequencies are chosen log-spaced
            across the band and their per-mechanism weights are fit (in
            float64, differentiably w.r.t. ``qp``/``qs``) by least squares
            so the effective ``Q(f)`` approximates the target ``qp``/``qs``
            across the band (a constant-Q least-squares fit).
            Supported on the eager Python backend only.
        q_band: ``(f1, f2)`` frequency band for the GSLS constant-Q fit;
            required when ``q_n_mechanisms > 1``, must be ``None`` when
            ``q_n_mechanisms == 1``.
        max_vel: See :func:`deepwave.elastic`.
        courant_number: See :func:`deepwave.elastic`.
        survey_pad: See :func:`deepwave.scalar`.
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
        backward_callback: See :func:`deepwave.elastic`. Not supported on the eager
            Python backend.
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
            Only `"device"` is supported on the eager Python
            backend.
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
            Not supported on the eager Python backend.
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
        order, are the physical fields in :func:`deepwave.elastic`'s public
        kind-grouped order, then the GSLS memory variables grouped by
        relaxation mechanism, then ``auxiliary_state``, then the receiver
        amplitudes:

            - vz_0: Final z-velocity wavefield (if ndim == 3).
            - vy_0: Final y-velocity wavefield (if ndim >= 2).
            - vx_0: Final x-velocity wavefield.
            - sigmazz_0: Final sigma_zz wavefield (if ndim == 3).
            - sigmayz_0: Final sigma_yz wavefield (if ndim == 3).
            - sigmaxz_0: Final sigma_xz wavefield (if ndim == 3).
            - sigmayy_0: Final sigma_yy wavefield (if ndim >= 2).
            - sigmaxy_0: Final sigma_xy wavefield (if ndim >= 2).
            - sigmaxx_0: Final sigma_xx wavefield.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables and the GSLS relaxation
              memory variables (the mean/Qp and deviatoric/Qs variables, one
              set per mechanism). Opaque: pass it back (as part of
              ``initial_state``) to continue the simulation; its contents are
              implementation state rather than physics. The field list
              therefore does not change with ``q_n_mechanisms``.
            - receiver_amplitudes_p: Recorded pressure data. In 2-D the
              recorded pressure is the in-plane mean normal stress
              ``-(sigma_xx + sigma_yy)/2``, which under plane strain is not
              the thermodynamic pressure; see :func:`deepwave.elastic`'s
              ``receiver_amplitudes_p`` entry for the full statement.
            - receiver_amplitudes_z: Recorded z-velocity data (if ndim == 3).
            - receiver_amplitudes_y: Recorded y-velocity data (if ndim >= 2).
            - receiver_amplitudes_x: Recorded x-velocity data.

        See :func:`deepwave.wavefield_names` for the exact field list
        for a given ``ndim``/``q_n_mechanisms``. This positional order is
        easy to index incorrectly; prefer accessing the fields by name
        instead.
    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            viscoelastic,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
            ),
        )
    backward_callback_needs_forward_wavefield = False
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
        [lamb, mu, buoyancy, qp, qs],
        ["lamb", "mu", "buoyancy", "qp", "qs"],
        "viscoelastic",
    )
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    del backend
    # The caller's own pre-resolution choice, kept for the unsupported-option
    # message -- see reject_unsupported_python_backend_options's docstring.
    requested_backend = backend_request.canonical

    if q_n_mechanisms < 1:
        raise DeepwaveValueError(
            f"viscoelastic: q_n_mechanisms must be >= 1, but got {q_n_mechanisms}."
        )
    if q_n_mechanisms == 1:
        if q_band is not None:
            raise DeepwaveValueError(
                "viscoelastic: q_band must be None when q_n_mechanisms == 1 "
                "(the default single-mechanism SLS is controlled by q_freq "
                "instead)."
            )
        if q_freq is None:
            raise DeepwaveValueError(
                "viscoelastic: q_freq is required. It is the reference frequency at "
                "which Qp/Qs is specified, in the same units as 1/dt, and it has "
                "no default: a reference frequency that Deepwave chose would "
                "silently redefine the attenuation model. Pass the frequency "
                "your Qp/Qs was measured or chosen at, e.g. q_freq=25.0."
            )
        _common.check_q_freq(q_freq, "viscoelastic")
    else:
        if q_band is None:
            raise DeepwaveValueError(
                "viscoelastic: q_band=(f1, f2) is required when q_n_mechanisms > 1."
            )
        q_band_f1, q_band_f2 = q_band
        if not (0 < q_band_f1 < q_band_f2):
            raise DeepwaveValueError(
                "viscoelastic: q_band=(f1, f2) must satisfy 0 < f1 < f2, but "
                f"got {(q_band_f1, q_band_f2)}."
            )
        if q_freq is not None:
            raise DeepwaveValueError(
                "viscoelastic: q_freq must be None when q_n_mechanisms > 1, which "
                "fits a constant Qp/Qs across a frequency band rather than at a "
                "single reference frequency; pass the band as q_band=(f1, f2)."
            )

    # Initial wavefields reach this call either as explicit arguments or
    # inside the optional ``initial_state`` bundle, and either way they are
    # part of the survey, so both must be visible to the ndim determination
    # below. The bundle is therefore read here, before it is merged into the
    # explicit arguments (which happens once ndim is known, because ndim
    # decides which state slots exist). Only its wavefield entries can say
    # what ndim is: receiver amplitudes have a different rank, and the
    # auxiliary state is not a Tensor.
    _recv_names = _engine.receiver_output_names(SPEC)
    _state_map: Optional[Mapping[str, torch.Tensor]]
    if isinstance(initial_state, deepwave.PropagatorOutput):
        _state_map = initial_state._asdict()
    else:
        _state_map = initial_state
    _state_wavefields: list[Optional[torch.Tensor]] = (
        [
            t
            for k, t in _state_map.items()
            if k not in _recv_names and isinstance(t, torch.Tensor)
        ]
        if isinstance(_state_map, Mapping)
        else []
    )
    ndim = _common.get_ndim(
        [lamb, mu, buoyancy, qp, qs],
        [
            vz_0,
            sigmazz_0,
            sigmayz_0,
            sigmaxz_0,
            vy_0,
            sigmayy_0,
            sigmaxy_0,
            vx_0,
            sigmaxx_0,
            *_state_wavefields,
        ],
        [
            source_locations_pressure_rate,
            source_locations_z,
            source_locations_y,
            source_locations_x,
            receiver_locations_p,
            receiver_locations_z,
            receiver_locations_y,
            receiver_locations_x,
        ],
        [vz_0, sigmazz_0, sigmayz_0, sigmaxz_0],
        [vy_0, sigmayy_0, sigmaxy_0],
        [vx_0, sigmaxx_0],
    )
    if ndim not in (2, 3):
        raise DeepwaveValueError(
            "viscoelastic: only 2-D and 3-D are supported (shear waves require "
            f"at least two spatial dimensions), but got ndim={ndim}."
        )

    # Resolved once ndim is known, so that backend="compiled" -- which this
    # propagator can never satisfy -- can name the whole configuration.
    backend_mode = _engine.resolve_eager_only_backend(
        backend_request,
        "viscoelastic",
        dtype=lamb.dtype,
        ndim=ndim,
        device=lamb.device,
    )

    if not 0 <= mpml_fraction < 1:
        raise DeepwaveValueError(
            "viscoelastic: mpml_fraction (the multiaxial-PML cross-damping fraction) "
            f"must be in [0, 1), but got {mpml_fraction}."
        )

    dims = _dim_names(ndim)
    pairs = _shear_pairs(ndim)

    # Wavefield-argument mapping. The GSLS memory variables are neither
    # output fields nor keyword arguments: how many there are depends on
    # q_n_mechanisms, so they travel in auxiliary_state alongside the m_*
    # absorbing-boundary memory and are supplied by passing a previous
    # call's state back through initial_state. They still need a slot here,
    # under the public spelling the fold set is keyed by, because this
    # mapping is what the fold reads the tensors out of; the order is the
    # backend's, which is what resolve_initial_state documents its mapping
    # as.
    explicit_elastic: dict[str, Optional[torch.Tensor]] = {
        "vz_0": vz_0,
        "vy_0": vy_0,
        "vx_0": vx_0,
        "sigmazz_0": sigmazz_0,
        "sigmayz_0": sigmayz_0,
        "sigmaxz_0": sigmaxz_0,
        "sigmayy_0": sigmayy_0,
        "sigmaxy_0": sigmaxy_0,
        "sigmaxx_0": sigmaxx_0,
    }
    gsls_group_names = (
        ["rp"]
        + [f"rs_{d}" for d in dims]
        + [f"rs_{dims[i]}{dims[j]}" for i, j in pairs]
    )
    gsls_public = viscoelastic_state_name_renames(ndim, q_n_mechanisms)

    elastic_wf_names = elastic.elastic_wavefield_names(ndim)
    wavefield_args: dict[str, Optional[torch.Tensor]] = {}
    for name in elastic_wf_names:
        wavefield_args[name] = explicit_elastic.get(name)
    for g in gsls_group_names:
        for mech in range(q_n_mechanisms):
            wavefield_args[gsls_public[f"{g}_{mech}"]] = None

    # Read before the call's own parameters are resolved; checked once the
    # parameters it must agree with have been.
    supplied_auxiliary_state = _engine.auxiliary_state_of(initial_state)
    # A continuation has nothing of its own to resolve these from -- it
    # typically has no sources, and the first call's `max_vel` need not be the
    # model's own maximum -- so an omitted one takes the value the state's
    # absorbing-boundary memory variables were built with.
    pml_freq = _engine.inherited_pml_freq(pml_freq, supplied_auxiliary_state)
    max_vel = _engine.inherited_max_vel(max_vel, supplied_auxiliary_state)
    resolved_wavefields, origin = _engine.resolve_initial_state(
        initial_state,
        wavefield_args,
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=viscoelastic_auxiliary_state_names(3, q_n_mechanisms),
        propagator=SPEC.name,
    )

    vz_0 = resolved_wavefields.get("vz_0")
    vy_0 = resolved_wavefields.get("vy_0")
    vx_0 = resolved_wavefields.get("vx_0")
    sigmazz_0 = resolved_wavefields.get("sigmazz_0")
    sigmayz_0 = resolved_wavefields.get("sigmayz_0")
    sigmaxz_0 = resolved_wavefields.get("sigmaxz_0")
    sigmayy_0 = resolved_wavefields.get("sigmayy_0")
    sigmaxy_0 = resolved_wavefields.get("sigmaxy_0")
    sigmaxx_0 = resolved_wavefields.get("sigmaxx_0")
    pml_mem_0 = [
        resolved_wavefields[name]
        for name in elastic_wf_names
        if name not in explicit_elastic
    ]

    r_group_0: dict[str, list[Optional[torch.Tensor]]] = {
        g: [
            resolved_wavefields[gsls_public[f"{g}_{mech}"]]
            for mech in range(q_n_mechanisms)
        ]
        for g in gsls_group_names
    }

    _common.check_inputs_not_vmapped(
        vz_0,
        vy_0,
        vx_0,
        sigmazz_0,
        sigmayz_0,
        sigmaxz_0,
        sigmayy_0,
        sigmaxy_0,
        sigmaxx_0,
        *pml_mem_0,
        *[t for lst in r_group_0.values() for t in lst],
        source_amplitudes_pressure_rate,
        source_amplitudes_z,
        source_amplitudes_y,
        source_amplitudes_x,
        source_locations_pressure_rate,
        source_locations_z,
        source_locations_y,
        source_locations_x,
        receiver_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
    )

    # Elastic-block initial wavefields, in elastic's OWN backend order (see
    # elastic_wavefield_names/_elastic_eager_core), so this list can be fed
    # straight through the same field-count-based unpacking pattern.
    elastic_named: dict[str, Optional[torch.Tensor]] = {
        "vz_0": vz_0,
        "vy_0": vy_0,
        "vx_0": vx_0,
        "sigmazz_0": sigmazz_0,
        "sigmayz_0": sigmayz_0,
        "sigmaxz_0": sigmaxz_0,
        "sigmayy_0": sigmayy_0,
        "sigmaxy_0": sigmaxy_0,
        "sigmaxx_0": sigmaxx_0,
    }
    pml_mem_iter = iter(pml_mem_0)
    initial_wavefields: list[Optional[torch.Tensor]] = []
    for name in elastic_wf_names:
        if name in elastic_named:
            initial_wavefields.append(elastic_named[name])
        else:
            initial_wavefields.append(next(pml_mem_iter))
    for g in gsls_group_names:
        initial_wavefields.extend(r_group_0[g])

    min_nonzero_model_vel, max_model_vel = _elastic_velocity_extremes(
        lamb, mu, buoyancy, max_vel
    )
    if not all(_common.values_readable(t) for t in (qp, qs)) and max_vel is None:
        raise _common.max_vel_unreadable_error("viscoelastic")
    if all(_common.values_readable(t) for t in (lamb, mu, buoyancy, qp, qs)):
        # Match prepare_models' vacuum-Q substitution before estimating the
        # coefficients that its staggered update will actually use.
        q_floor = 1 / torch.finfo(lamb.dtype).max ** (1 / 2)
        qp_cfl = torch.where(qp.detach().abs() > q_floor, qp.detach(), 1.0)
        qs_cfl = torch.where(qs.detach().abs() > q_floor, qs.detach(), 1.0)
        k_relaxed = lamb.detach() + (2.0 / ndim) * mu.detach()
        k_unrelaxed = k_relaxed * _gsls.unrelaxed_factor(qp_cfl, q_n_mechanisms, q_band)
        mu_unrelaxed = mu.detach() * _gsls.unrelaxed_factor(
            qs_cfl, q_n_mechanisms, q_band
        )
        vp2 = (k_unrelaxed + 2.0 * (ndim - 1) / ndim * mu_unrelaxed) * buoyancy.detach()
        vs2 = mu_unrelaxed * buoyancy.detach()
        # Staggered shear/Q and buoyancy averages mix adjacent cells. Bounds
        # from independent coefficient maxima cover those mixtures.
        k_bound = (
            k_relaxed.abs().max()
            * _gsls.unrelaxed_factor(qp_cfl, q_n_mechanisms, q_band).max()
        )
        shear_bound = (
            mu.detach().abs().max()
            * _gsls.unrelaxed_factor(qs_cfl, q_n_mechanisms, q_band).max()
        )
        mixed_bound = (
            k_bound + 2.0 * (ndim - 1) / ndim * shear_bound
        ) * buoyancy.detach().abs().max()
        max_model_vel = float(
            torch.maximum(
                torch.maximum(vp2, vs2).clamp_min(0).max(), mixed_bound
            ).sqrt()
        )
        if max_vel is not None and abs(float(max_vel)) < max_model_vel:
            raise DeepwaveValueError(
                "viscoelastic: max_vel must bound the unrelaxed P and S speeds "
                f"(at least {max_model_vel:.6g} for this model)."
            )
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

    source_amplitudes_list: list[Optional[torch.Tensor]] = []
    source_locations_list_in: list[Optional[LocationsLike]] = []
    if ndim == 3:
        source_amplitudes_list.extend(
            [source_amplitudes_z, source_amplitudes_y, source_amplitudes_x]
        )
        source_locations_list_in.extend(
            [source_locations_z, source_locations_y, source_locations_x]
        )
    else:
        source_amplitudes_list.extend([source_amplitudes_y, source_amplitudes_x])
        source_locations_list_in.extend([source_locations_y, source_locations_x])

    receiver_locations_list_in: list[Optional[LocationsLike]] = []
    if ndim == 3:
        receiver_locations_list_in.extend(
            [receiver_locations_z, receiver_locations_y, receiver_locations_x]
        )
    else:
        receiver_locations_list_in.extend([receiver_locations_y, receiver_locations_x])
    # Matches deepwave.elastic's own channel order (velocities first, then
    # p LAST -- see elastic.py's source-amplitude scaling block, which
    # indexes p via source_amplitudes_out[-1]; this differs from
    # deepwave.viscoacoustic's p-FIRST convention, since this propagator's
    # eager driver reuses elastic's source/receiver injection/extraction
    # indexing verbatim).
    source_amplitudes_list.append(source_amplitudes_pressure_rate)
    source_locations_list_in.append(source_locations_pressure_rate)
    receiver_locations_list_in.append(receiver_locations_p)

    component_offsets: list[Optional[list[float]]] = [
        *_common.velocity_component_offsets(ndim),
        None,
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
        lamb,
        "viscoelastic",
    )
    source_amplitudes_list, source_locations_list = (
        _common.resolve_source_interpolators(
            source_locations_list_in,
            source_amplitudes_list,
            component_offsets,
            location_interpolation,
            location_interpolation_halfwidth,
            lamb.dtype,
            _hicks_grid,
        )
    )
    receiver_interpolators, receiver_locations_list = (
        _common.resolve_receiver_interpolators(
            receiver_locations_list_in,
            component_offsets,
            location_interpolation,
            location_interpolation_halfwidth,
            lamb.dtype,
            _hicks_grid,
        )
    )

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
        [lamb, mu, buoyancy, qp, qs],
        ["replicate"] * 5,
        ["lamb", "mu", "buoyancy", "qp", "qs"],
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
        q_n_mechanisms=q_n_mechanisms,
    )
    del lamb, mu, buoyancy, qp, qs

    q_freq_resolved = _resolve_q_freq(q_freq)
    omega0 = 2.0 * math.pi * q_freq_resolved

    models = prepare_models(
        models[0],
        models[1],
        models[2],
        models[3],
        models[4],
        dt,
        omega0,
        q_n_mechanisms,
        q_band,
    )

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "viscoelastic")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "viscoelastic")
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
        n_mechanisms=q_n_mechanisms,
        mpml=mpml_fraction > 0,
        inputs_require_grad=_engine.any_requires_grad(
            models, source_amplitudes_out, wavefields
        ),
        wavefields=wavefields,
        max_vel=max_vel,
        internal_dt=dt,
        fold_names=viscoelastic_auxiliary_state_names(ndim, q_n_mechanisms),
    )

    model_shape = models[0].shape[-ndim:]
    # Scale the velocity-component source amplitudes by buoyancy (last ndim
    # models); the pressure source is scaled only by -dt (see
    # deepwave.elastic's identical p-channel convention: velocity channels
    # first, p LAST).
    for k in range(ndim):
        idx = -(ndim + 1) + k
        if source_amplitudes_out[idx].numel() > 0:
            source_amplitudes_out[idx] = (
                source_amplitudes_out[idx]
                * _common.gather_model_at_sources(
                    models[-ndim + k], sources_i[idx], model_shape, n_shots
                )
                * dt
            )
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
        names=viscoelastic_output_names(ndim, q_n_mechanisms),
        state_names=viscoelastic_public_wavefield_names(ndim, q_n_mechanisms),
        fold_names=viscoelastic_auxiliary_state_names(ndim, q_n_mechanisms),
        python_backend=requested_backend,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
    )

    outputs = list(
        viscoelastic_func(
            backend_mode,
            requested_backend,
            q_n_mechanisms,
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

    # Gather off-grid receiver data back to the original locations while the
    # receiver outputs are still in [velocities..., pressure] order (matching
    # receiver_interpolators, built in that same order); the gather is
    # linear, so it commutes with the normal-stress mean applied below to
    # form the pressure trace (mirrors deepwave.elastic's identical
    # ordering/comment).
    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 1 - ndim
    )

    # Pressure is the negative arithmetic mean of the normal stresses (the
    # driver sums them, unscaled, since the mean/sign convention is a
    # property of the receiver, not the physics); it is the LAST output
    # (matching elastic's p-last internal channel order -- see the module
    # docstring), so move it to the FRONT of the receiver block to match the
    # public p-first naming (viscoelastic_output_names).
    outputs[-1] = -outputs[-1] / ndim

    # Reorder the state tensors into the public order (physical fields
    # grouped by kind, then the GSLS memory grouped by mechanism, then the
    # absorbing-boundary memory). The SAME index list names them in
    # viscoelastic_public_wavefield_names, so the two cannot disagree.
    n_state = len(viscoelastic_wavefield_names(ndim, q_n_mechanisms))
    outputs[:n_state] = [
        outputs[i] for i in viscoelastic_public_wavefield_order(ndim, q_n_mechanisms)
    ]

    outputs = [*outputs[: -1 - ndim], outputs[-1], *outputs[-1 - ndim : -1]]

    # SPEC.wavefield_names/output_names default to q_n_mechanisms == 1 (see
    # PropagatorSpec's fixed Callable[[int], list[str]] contract, which cannot
    # see this call's q_n_mechanisms), so the two name lists are passed
    # explicitly from the n_mechanisms-aware functions.
    return _engine.named_output(
        SPEC,
        ndim,
        outputs,
        info=run_info,
        accuracy=accuracy,
        boundary=boundary_names,
        n_mechanisms=q_n_mechanisms,
        mpml=mpml_fraction > 0,
        names=viscoelastic_output_names(ndim, q_n_mechanisms),
        state_names=viscoelastic_public_wavefield_names(ndim, q_n_mechanisms),
        fold_names=viscoelastic_auxiliary_state_names(ndim, q_n_mechanisms),
        check_output_finite=check_output_finite,
    )


def _elastic_velocity_extremes(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    max_vel: Optional[float],
) -> tuple[float, float]:
    """Returns ``(min_nonzero_vel, max_vel)`` from the relaxed P/S speeds.

    The caller replaces the maximum with the unrelaxed constitutive bound
    before choosing the CFL time step.
    """
    if not all(_common.values_readable(prop) for prop in [lamb, mu, buoyancy]):
        if max_vel is None:
            raise _common.max_vel_unreadable_error("viscoelastic")
        return 0.0, max_vel
    vp, vs, _ = _common.lambmubuoyancy_to_vpvsrho(
        lamb.detach(), mu.detach(), buoyancy.detach()
    )
    max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
    vp_nonzero = vp[vp != 0]
    min_nonzero_vp = vp_nonzero.abs().min().item() if vp_nonzero.numel() > 0 else 0.0
    vs_nonzero = vs[vs != 0]
    min_nonzero_vs = vs_nonzero.abs().min().item() if vs_nonzero.numel() > 0 else 0.0
    if min_nonzero_vp == 0 and min_nonzero_vs == 0:
        min_nonzero_model_vel = 0.0
    elif min_nonzero_vp == 0:
        min_nonzero_model_vel = float(min_nonzero_vs)
    elif min_nonzero_vs == 0:
        min_nonzero_model_vel = float(min_nonzero_vp)
    else:
        min_nonzero_model_vel = float(min(min_nonzero_vp, min_nonzero_vs))
    return min_nonzero_model_vel, max_model_vel


def viscoelastic_func(
    backend: Optional[Literal["eager", "compile"]],
    requested_backend: str,
    q_n_mechanisms: int,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """Dispatches viscoelastic propagation to one of the two Python modes.

    There is no compiled C/CUDA backend, so ``backend`` is always
    truthy by the time it reaches this function (``viscoelastic()`` already
    resolves ``"auto"`` via ``_engine.resolve_eager_only_backend`` before
    dispatching); the falsy check below is a defensive invariant, not
    reachable via the public API.

    ``requested_backend`` is the caller's own pre-resolution ``backend``
    choice (``"auto"`` if not given explicitly, else ``"eager"``/
    ``"torch_compile"``) -- passed through to
    :func:`viscoelastic_python` for its unsupported-option message instead of
    the resolved ``mode``, so a default ``backend="auto"`` call is never
    reported as having asked for
    ``backend="eager"`` (see :func:`_engine.reject_unsupported_python_backend_options`).
    """
    global _update_velocities_compile, _update_velocities_opt
    global _update_stresses_compile, _update_stresses_opt

    if not backend:  # pragma: no cover - guarded in viscoelastic()
        raise DeepwaveNotImplementedError(
            "viscoelastic: only the eager Python backend is supported."
        )
    mode = _engine.resolve_backend_mode(backend)
    if mode == "compile":
        if _update_velocities_compile is None:
            _update_velocities_compile = torch.compile(
                update_velocities, fullgraph=True
            )
        _update_velocities_opt = _update_velocities_compile
        if _update_stresses_compile is None:
            _update_stresses_compile = torch.compile(update_stresses, fullgraph=True)
        _update_stresses_opt = _update_stresses_compile
    else:  # eager
        _update_velocities_opt = update_velocities
        _update_stresses_opt = update_stresses
    return viscoelastic_python(q_n_mechanisms, *args, backend=requested_backend)


def viscoelastic_python(
    q_n_mechanisms: int,
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
    mpml: bool,
    *args: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Eager viscoelastic forward pass (Qp/Qs GSLS attenuation on elastic).

    The elastic-block wavefield bookkeeping (unpacking, PML, source/receiver
    injection/extraction) is a direct port of ``elastic._elastic_eager_core``,
    including its per-face ``boundary`` guard-fill
    calls (``boundary`` is the per-face ``BC_*`` integer code list,
    ``BC_PML`` everywhere for the default/no-boundary case -- see
    ``_boundaries.fill_elastic_stress_boundary`` and
    ``_boundaries.fill_elastic_velocity_boundary``); the GSLS Qp/Qs memory
    variables are threaded through the
    SAME loop alongside it, updated by :func:`update_stresses`. The memory
    variables themselves are never guard-filled: no spatial derivative of
    any GSLS memory variable appears anywhere in :func:`update_stresses`
    (each is a purely pointwise recursion driven by the local strain-rate),
    so their ghost regions are never read -- see the module docstring's
    "Free/rigid/periodic faces" section.

    ``backend`` is the resolved Python backend name (``"eager"``
    or ``"compile"``), already selected by the caller; used only to name it
    if an unsupported option is rejected.
    """
    ndim = len(grid_spacing)
    is_batched = any(_common.is_inside_vmap(x) for x in args)
    _engine.reject_unsupported_python_backend_options(
        SPEC,
        backend,
        storage_mode=storage_mode_str,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
        is_batched=is_batched,
    )
    args_list = list(args)
    del args

    device = args_list[0].device
    dtype = args_list[0].dtype
    model_shape = args_list[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *model_shape)
    has_boundary = any(code != _common.BC_PML for code in boundary)

    n_source_types = ndim + 1
    n_models = len(viscoelastic_model_names(ndim))
    n_elastic_wf = len(elastic.elastic_wavefield_names(ndim))
    n_shear = ndim * (ndim - 1) // 2

    # The call site (viscoelastic()) supplies *models, *source_amplitudes,
    # *sources_i, *receivers_i, *wavefields in that order; wavefields is the
    # only variable-length (q_n_mechanisms-dependent) group, so pop the
    # fixed-size groups from the FRONT (mirroring viscoacoustic_python's
    # pattern) rather than assuming wavefields is first (which is
    # deepwave.elastic._elastic_eager_core's fixed-size-wavefields layout,
    # not applicable here).
    models = args_list[:n_models]
    del args_list[:n_models]
    source_amplitudes = args_list[:n_source_types]
    del args_list[:n_source_types]
    sources_i = args_list[:n_source_types]
    del args_list[:n_source_types]
    receivers_i = args_list[:n_source_types]
    del args_list[:n_source_types]
    wavefields_flat = args_list
    del args_list

    wavefields_flat = [
        _engine.create_or_pad(w, fd_pad_list, device, dtype, size_with_batch)
        for w in wavefields_flat
    ]

    zero_edges_and_interiors(
        wavefields_flat, ndim, fd_pad, fd_pad_list, pml_width, q_n_mechanisms, mpml
    )

    # Unpack the elastic block: identical layout/order to
    # deepwave.elastic._elastic_eager_core (reproduced verbatim for this
    # block, since it is elastic's own private indexing scheme).
    if ndim == 3:
        v_z, v_y, v_x = wavefields_flat[0], wavefields_flat[14], wavefields_flat[23]
        s_zz, s_yy, s_xx = wavefields_flat[1], wavefields_flat[15], wavefields_flat[24]
        s_yz, s_xz, s_xy = wavefields_flat[2], wavefields_flat[3], wavefields_flat[16]
        m_v_zz, m_v_yy, m_v_xx = (
            wavefields_flat[4],
            wavefields_flat[17],
            wavefields_flat[25],
        )
        m_v_zy, m_v_zx = wavefields_flat[5], wavefields_flat[6]
        m_v_yz, m_v_xz = wavefields_flat[7], wavefields_flat[8]
        m_v_yx, m_v_xy = wavefields_flat[18], wavefields_flat[19]
        m_s_zzz, m_s_yyy, m_s_xxx = (
            wavefields_flat[9],
            wavefields_flat[20],
            wavefields_flat[26],
        )
        m_s_yzy, m_s_xzx = wavefields_flat[10], wavefields_flat[11]
        m_s_yzz, m_s_xzz = wavefields_flat[12], wavefields_flat[13]
        m_s_xyy, m_s_xyx = wavefields_flat[21], wavefields_flat[22]

        velocities = [v_z, v_y, v_x]
        normal_stresses = [s_zz, s_yy, s_xx]
        shear_stresses = [
            [torch.empty(0), s_yz, s_xz],
            [s_yz, torch.empty(0), s_xy],
            [s_xz, s_xy, torch.empty(0)],
        ]
        v_mem_vars_normal = [m_v_zz, m_v_yy, m_v_xx]
        v_mem_vars_shear = [
            [torch.empty(0), m_v_zy, m_v_zx],
            [m_v_yz, torch.empty(0), m_v_yx],
            [m_v_xz, m_v_xy, torch.empty(0)],
        ]
        s_mem_vars_normal = [m_s_zzz, m_s_yyy, m_s_xxx]
        s_mem_vars_shear = [
            [torch.empty(0), m_s_yzy, m_s_xzx],
            [m_s_yzz, torch.empty(0), m_s_xyx],
            [m_s_xzz, m_s_xyy, torch.empty(0)],
        ]
        del (
            v_z,
            v_y,
            v_x,
            s_zz,
            s_yy,
            s_xx,
            s_yz,
            s_xz,
            s_xy,
            m_v_zz,
            m_v_yy,
            m_v_xx,
            m_v_zy,
            m_v_zx,
            m_v_yz,
            m_v_xz,
            m_v_yx,
            m_v_xy,
            m_s_zzz,
            m_s_yyy,
            m_s_xxx,
            m_s_yzy,
            m_s_xzx,
            m_s_yzz,
            m_s_xzz,
            m_s_xyy,
            m_s_xyx,
        )
        v_names = ["vz_0", "vy_0", "vx_0"]
        normal_s_names = ["sigmazz_0", "sigmayy_0", "sigmaxx_0"]
        shear_s_names = [["sigmayz_0", "sigmaxz_0"], ["sigmaxy_0"]]
    else:  # ndim == 2
        v_y, v_x = wavefields_flat[0], wavefields_flat[9]
        s_yy, s_xx = wavefields_flat[1], wavefields_flat[10]
        s_xy = wavefields_flat[2]
        m_v_yy, m_v_xx = wavefields_flat[3], wavefields_flat[11]
        m_v_yx, m_v_xy = wavefields_flat[4], wavefields_flat[5]
        m_s_yyy, m_s_xxx = wavefields_flat[6], wavefields_flat[12]
        m_s_xyy, m_s_xyx = wavefields_flat[7], wavefields_flat[8]

        velocities = [v_y, v_x]
        normal_stresses = [s_yy, s_xx]
        shear_stresses = [[torch.empty(0), s_xy], [s_xy, torch.empty(0)]]
        v_mem_vars_normal = [m_v_yy, m_v_xx]
        v_mem_vars_shear = [[torch.empty(0), m_v_yx], [m_v_xy, torch.empty(0)]]
        s_mem_vars_normal = [m_s_yyy, m_s_xxx]
        s_mem_vars_shear = [[torch.empty(0), m_s_xyx], [m_s_xyy, torch.empty(0)]]
        del (
            v_y,
            v_x,
            s_yy,
            s_xx,
            s_xy,
            m_v_yy,
            m_v_xx,
            m_v_yx,
            m_v_xy,
            m_s_yyy,
            m_s_xxx,
            m_s_xyy,
            m_s_xyx,
        )
        v_names = ["vy_0", "vx_0"]
        normal_s_names = ["sigmayy_0", "sigmaxx_0"]
        shear_s_names = [["sigmaxy_0"]]

    gsls_flat = wavefields_flat[n_elastic_wf:]
    del wavefields_flat
    r_mean = gsls_flat[:q_n_mechanisms]
    gi = q_n_mechanisms
    r_dev_normal: list[list[torch.Tensor]] = []
    for _ in range(ndim):
        r_dev_normal.append(gsls_flat[gi : gi + q_n_mechanisms])
        gi += q_n_mechanisms
    r_dev_shear: list[list[torch.Tensor]] = []
    for _ in range(n_shear):
        r_dev_shear.append(gsls_flat[gi : gi + q_n_mechanisms])
        gi += q_n_mechanisms
    del gsls_flat

    wf_names = viscoelastic_wavefield_names(ndim, q_n_mechanisms)

    n_receivers_per_shot = [locs.numel() // n_shots for locs in receivers_i]
    receiver_amplitudes: list[torch.Tensor] = [
        torch.empty(0, device=device, dtype=dtype) for _ in range(ndim + 1)
    ]
    receiver_amplitudes_lists: list[list[torch.Tensor]] = [[] for _ in range(ndim + 1)]
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

    model_names = viscoelastic_model_names(ndim)
    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            current = [
                *velocities,
                *normal_stresses,
                *[shear_stresses[i][j] for i, j in _shear_pairs(ndim)],
                *v_mem_vars_normal,
                *[v_mem_vars_shear[i][j] for i, j in _ordered_pairs(ndim)],
                *s_mem_vars_normal,
                *[s_mem_vars_shear[i][j] for i, j in _ordered_pairs(ndim)],
                *r_mean,
                *[t for lst in r_dev_normal for t in lst],
                *[t for lst in r_dev_shear for t in lst],
            ]
            callback_wavefields = dict(zip(wf_names, current))
            _engine.invoke_forward_callback(
                forward_callback,
                CallbackState(
                    dt,
                    step,
                    callback_wavefields,
                    callback_models,
                    {},
                    fd_pad_list,
                    pml_width,
                    direction="forward",
                    n_steps=nt // step_ratio,
                ),
                differentiable=torch.is_grad_enabled()
                and any(
                    t.requires_grad
                    for t in [*models, *source_amplitudes_masked, *current]
                ),
            )

        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step

            # Refill the stress guard ring (read by the velocity update via
            # sigma_AA/the shared-axis shears' derivatives, and by the
            # pressure receiver record just below); on a free face this also
            # holds the on-surface normal-stress node to zero. Must precede
            # the receiver record (mirrors elastic.py's _elastic_eager_core)
            # so a surface-row pressure/velocity receiver sees the correctly
            # zeroed/imaged field, not the pre-fill value.
            if has_boundary:
                normal_stresses, shear_stresses = (
                    _boundaries.fill_elastic_stress_boundary(
                        normal_stresses, shear_stresses, boundary, fd_pad
                    )
                )

            for ridx in range(ndim):
                if receivers_i_masked[ridx].numel() > 0:
                    val_vel = (
                        velocities[ridx]
                        .reshape(-1, flat_model_shape)
                        .gather(1, receivers_i_masked[ridx])
                    )
                    if is_batched:
                        receiver_amplitudes_lists[ridx].append(val_vel)
                    else:
                        receiver_amplitudes[ridx][t] = val_vel
            if receivers_i_masked[-1].numel() > 0:
                val_stress = torch.stack(
                    [
                        normal_stresses[idx]
                        .reshape(-1, flat_model_shape)
                        .gather(1, receivers_i_masked[-1])
                        for idx in range(ndim)
                    ]
                ).sum(dim=0)
                if is_batched:
                    receiver_amplitudes_lists[-1].append(val_stress)
                else:
                    receiver_amplitudes[-1][t] = val_stress

            velocities, s_mem_vars_normal, s_mem_vars_shear = _update_velocities_opt(
                models,
                normal_stresses,
                shear_stresses,
                velocities,
                s_mem_vars_normal,
                s_mem_vars_shear,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            new_velocities_list = list(velocities)
            for sidx in range(ndim):
                if source_amplitudes_masked[sidx].numel() > 0:
                    new_velocities_list[sidx] = (
                        new_velocities_list[sidx]
                        .clone()
                        .reshape(-1, flat_model_shape)
                        .scatter_add_(
                            1, sources_i_masked[sidx], source_amplitudes_masked[sidx][t]
                        )
                        .reshape(size_with_batch)
                    )
            velocities = list(new_velocities_list)
            del new_velocities_list

            # Refill the velocity guard ring (read by every normal stress's
            # own divergence term and by the shared-axis shears' updates);
            # on a free face every component's ghost region is zeroed (see
            # _boundaries.fill_elastic_velocity_boundary's docstring).
            if has_boundary:
                velocities = _boundaries.fill_elastic_velocity_boundary(
                    velocities, boundary, fd_pad
                )

            (
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
                r_mean,
                r_dev_normal,
                r_dev_shear,
            ) = _update_stresses_opt(
                models,
                velocities,
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
                r_mean,
                r_dev_normal,
                r_dev_shear,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            if source_amplitudes_masked[-1].numel() > 0:
                new_normal_stresses = list(normal_stresses)
                for idx in range(ndim):
                    new_normal_stresses[idx] = (
                        new_normal_stresses[idx]
                        .clone()
                        .reshape(-1, flat_model_shape)
                        .scatter_add_(
                            1, sources_i_masked[-1], source_amplitudes_masked[-1][t]
                        )
                        .reshape(size_with_batch)
                    )
                normal_stresses = list(new_normal_stresses)
                del new_normal_stresses

    # Leave the returned wavefields boundary-consistent (e.g. free-surface
    # stress nodes held to zero, guard rings imaged) so a continuation run
    # reproduces the in-loop state (mirrors elastic.py's identical post-loop
    # refill).
    if has_boundary:
        normal_stresses, shear_stresses = _boundaries.fill_elastic_stress_boundary(
            normal_stresses, shear_stresses, boundary, fd_pad
        )
        velocities = _boundaries.fill_elastic_velocity_boundary(
            list(velocities), boundary, fd_pad
        )

    if is_batched:
        for i in range(len(receiver_amplitudes)):
            if len(receiver_amplitudes_lists[i]) > 0:
                receiver_amplitudes[i] = torch.stack(receiver_amplitudes_lists[i])

    receiver_amplitudes_masked = [
        _common.mask_receiver_amplitudes(amp, receivers_mask[i])
        for i, amp in enumerate(receiver_amplitudes)
    ]

    s = (slice(None), *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape))
    final_wavefields_list: list[torch.Tensor] = []
    if ndim == 3:
        final_wavefields_list.extend(
            [
                velocities[0],
                normal_stresses[0],
                shear_stresses[0][1],
                shear_stresses[0][2],
                v_mem_vars_normal[0],
                v_mem_vars_shear[0][1],
                v_mem_vars_shear[0][2],
                v_mem_vars_shear[1][0],
                v_mem_vars_shear[2][0],
                s_mem_vars_normal[0],
                s_mem_vars_shear[0][1],
                s_mem_vars_shear[0][2],
                s_mem_vars_shear[1][0],
                s_mem_vars_shear[2][0],
                velocities[1],
                normal_stresses[1],
                shear_stresses[1][2],
                v_mem_vars_normal[1],
                v_mem_vars_shear[1][2],
                v_mem_vars_shear[2][1],
                s_mem_vars_normal[1],
                s_mem_vars_shear[2][1],
                s_mem_vars_shear[1][2],
                velocities[2],
                normal_stresses[2],
                v_mem_vars_normal[2],
                s_mem_vars_normal[2],
            ]
        )
    else:
        final_wavefields_list.extend(
            [
                velocities[0],
                normal_stresses[0],
                shear_stresses[0][1],
                v_mem_vars_normal[0],
                v_mem_vars_shear[0][1],
                v_mem_vars_shear[1][0],
                s_mem_vars_normal[0],
                s_mem_vars_shear[1][0],
                s_mem_vars_shear[0][1],
                velocities[1],
                normal_stresses[1],
                v_mem_vars_normal[1],
                s_mem_vars_normal[1],
            ]
        )
    final_wavefields_list.extend(r_mean)
    for lst in r_dev_normal:
        final_wavefields_list.extend(lst)
    for lst in r_dev_shear:
        final_wavefields_list.extend(lst)

    del v_names, normal_s_names, shear_s_names

    return (
        *[field[s] for field in final_wavefields_list],
        *receiver_amplitudes_masked,
    )


_common.document_propagator_module(Viscoelastic, viscoelastic)

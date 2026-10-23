"""Elastic Born (linearised) wave propagation -- physics core.

Stability: experimental (see :doc:`/stability`).

Implements the isotropic elastic Born update rules derived in
``docs/elastic_born_derivation.md``: given a background elastic model
``(lamb, mu, buoyancy)`` and a scattering perturbation
``(dlamb, dmu, dbuoyancy)``, the scattered field ``(dv, dsigma)`` obeys the
same isotropic elastic operator as the background field, driven by virtual
sources built from the *background* field's own strain-rate / stress-
divergence terms times the scattering models (see derivation doc, S2).

This module provides the per-substep Born update functions
(:func:`update_velocities_born`, :func:`update_stresses_born`), the
scattering-model chain rule (:func:`prepare_scatter_parameters`), the eager
driver loop (:func:`elastic_born_eager_core`), the compiled-backend autograd
:class:`ElasticBornForwardFunc`, and the public ``elastic_born(...)`` API +
:class:`ElasticBorn` module.

The default, ``backend="auto"``, runs the compiled C/CUDA
forward on the shared step-based engine (:func:`deepwave._engine.run_forward_engine`,
storage/callbacks/step_ratio all honoured); without M-PML the backward uses
the compiled scattered-only and full two-field adjoint kernels
(``elastic_born.c``/``.cu``'s ``backward_sc``/``backward``, derivation
S4.1-4.3) on CPU and, when a CUDA runtime is present, on GPU too. M-PML
never reaches the compiled backend at all -- ``mpml_fraction > 0`` raises there
(pass ``backend="eager"``) -- so the eager-autograd recompute bridge in the
compiled backward, which would service it, is currently unreachable.
``backend="eager"``
selects the pure-PyTorch backend for both directions. Second-order
differentiation through the compiled backend raises
(:func:`deepwave._engine.no_second_order`); use the eager backend for
Hessian-vector products. The eager path is validated against finite
differences of the tested nonlinear ``elastic`` propagator (Born == FD, Born
linearity, jvp == Born, gradcheck/gradgradcheck, free-surface and 2D/3D
coverage), and the compiled path against the eager driver's autograd.
"""

import collections.abc
import importlib
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

import torch

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

import deepwave
from deepwave import (
    _backend_utils,
    _boundaries,
    _common,
    _engine,
    _staggered_grid,
    _storage,
)
from deepwave._engine import scattered_state_name as _scattered_name
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

# NOTE: ``deepwave/__init__.py`` does ``from deepwave.elastic import ...
# elastic``, which rebinds the package attribute ``deepwave.elastic`` to the
# *function* ``elastic`` (shadowing the submodule). ``import deepwave.elastic
# as x`` therefore resolves to that function, not the module. Import the name
# directly from the submodule (which looks it up in ``sys.modules`` via the
# submodule's own namespace, bypassing the shadowed package attribute), and
# fetch the submodule object itself via importlib for the many helper names
# the Born driver reuses (wavefield/model naming, structuring, masking).
from deepwave.elastic import prepare_parameters as _prepare_parameters
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

_elastic_module = importlib.import_module("deepwave.elastic")


def prepare_scatter_parameters(
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    dmu: torch.Tensor,
    dbuoyancy: torch.Tensor,
) -> list[torch.Tensor]:
    """Forward-mode derivative of :func:`elastic.prepare_parameters`.

    Maps the public scattering potentials ``(dmu, dbuoyancy)`` to the
    scattering images of the *prepared* (staggered) model set that the
    kernel actually consumes: ``(dmu_zy, dmu_zx, dmu_yx, dbuoyancy_z,
    dbuoyancy_y, dbuoyancy_x)`` (3D; fewer in lower ``ndim``), i.e. exactly
    the tangent of ``prepare_parameters`` at ``(mu, buoyancy)`` in the
    direction ``(dmu, dbuoyancy)``.

    This mirrors ``acoustic_born.prepare_scatter``: it does not hand-derive
    the harmonic/arithmetic-averaging derivatives, it pushes the tangent
    through the exact same masked ``torch.where`` composition
    ``prepare_parameters`` uses, so the IVM vacuum masking (SS7.1 of the
    derivation) and node alignment are inherited automatically, and its VJP
    (used to chain kernel gradients ``grad_dmu_*``/``grad_dbuoyancy_*`` back
    to ``grad_dmu``/``grad_dbuoyancy``) is handled by ordinary autograd with
    no hand-written backward.

    ``jvp`` requires each tangent to have exactly its primal's shape, which
    ordinary broadcasting would not give when (say) a batched ``mu`` meets an
    unbatched ``dmu``, so the four are first equalised on their leading (shot)
    dimension by :func:`deepwave._common.match_model_batch`. That is a no-op
    when they already agree.

    Args:
        mu: Background shear modulus (unstaggered).
        buoyancy: Background buoyancy (unstaggered).
        dmu: Shear-modulus scattering potential (same shape as ``mu``).
        dbuoyancy: Buoyancy scattering potential (same shape as ``buoyancy``).

    Returns:
        List of scattering-model tangents in the same order as
        ``elastic.prepare_parameters`` returns the background staggered
        models (``[dmu_zy, dmu_zx, dmu_yx, dbuoyancy_z, dbuoyancy_y,
        dbuoyancy_x]`` in 3D).
    """
    mu, buoyancy, dmu, dbuoyancy = _common.match_model_batch(
        [mu, buoyancy, dmu, dbuoyancy]
    )
    return _engine.model_preparation_jvp(
        _prepare_parameters, (mu, buoyancy), (dmu, dbuoyancy)
    )


def _update_velocities_terms(
    models: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    s_mem_vars_normal: list[torch.Tensor],
    s_mem_vars_shear: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]]]:
    """Like ``elastic.update_velocities``, but returns the pre-buoyancy term.

    Returns ``sigma_sum`` (per dimension), the exact quantity
    ``elastic.c`` checkpoints as ``dv{i}dbuoyancy_store_1 / dt`` (i.e.
    ``W_i`` in the derivation document, SS1.2/SS2.4) -- the divergence-of-
    stress term that the velocity update multiplies by buoyancy. This lets
    the Born coupling read the *same* term used for the background update
    without a second, PML-memory-variable-duplicating pass.

    This function's arithmetic is an exact line-for-line copy of
    ``elastic.update_velocities`` up to the point where ``buoyancy`` would
    be multiplied in; only the final buoyancy multiply/velocity-add is
    omitted (done instead by the caller, once for the background field and
    once -- with the coupling term -- for the scattered field).
    """
    ndim = len(grid_spacing)

    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]

    dsigmaxxdx = [
        _staggered_grid.diff1h(
            normal_stresses[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    for dim in range(ndim):
        s_mem_vars_normal[dim] = (
            ah[dim] * s_mem_vars_normal[dim] + bh[dim] * dsigmaxxdx[dim]
        )
        dsigmaxxdx[dim] = dsigmaxxdx[dim] + s_mem_vars_normal[dim]

    sigma_sum = list(dsigmaxxdx)

    for dim1 in range(ndim - 1):
        for dim2 in range(dim1 + 1, ndim):
            d_shear_d_dim2 = _staggered_grid.diff1(
                shear_stresses[dim1][dim2],
                dim2,
                accuracy,
                1 / grid_spacing[dim2],
                ndim,
            )
            mem = s_mem_vars_shear[dim1][dim2]
            new_mem = a[dim2] * mem + b[dim2] * d_shear_d_dim2
            s_mem_vars_shear[dim1][dim2] = new_mem
            d_shear_d_dim2 = d_shear_d_dim2 + new_mem
            sigma_sum[dim1] = sigma_sum[dim1] + d_shear_d_dim2

            d_shear_d_dim1 = _staggered_grid.diff1(
                shear_stresses[dim1][dim2],
                dim1,
                accuracy,
                1 / grid_spacing[dim1],
                ndim,
            )
            mem = s_mem_vars_shear[dim2][dim1]
            new_mem = a[dim1] * mem + b[dim1] * d_shear_d_dim1
            s_mem_vars_shear[dim2][dim1] = new_mem
            d_shear_d_dim1 = d_shear_d_dim1 + new_mem
            sigma_sum[dim2] = sigma_sum[dim2] + d_shear_d_dim1

    return sigma_sum, s_mem_vars_normal, s_mem_vars_shear


def update_velocities_born(
    bg_models: list[torch.Tensor],
    sc_models: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    normal_stresses_sc: list[torch.Tensor],
    shear_stresses_sc: list[list[torch.Tensor]],
    velocities: list[torch.Tensor],
    velocities_sc: list[torch.Tensor],
    s_mem_vars_normal: list[torch.Tensor],
    s_mem_vars_shear: list[list[torch.Tensor]],
    s_mem_vars_normal_sc: list[torch.Tensor],
    s_mem_vars_shear_sc: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[torch.Tensor],
    list[list[torch.Tensor]],
]:
    """Coupled background+scattered velocity update (derivation SS2.4/SS2.5).

    ``velocities`` updates exactly as the nonlinear ``elastic`` propagator
    (background field, background models). ``velocities_sc`` obeys the
    *same* operator (background models, scattered stresses) *plus* the Born
    virtual force ``dbuoyancy_i * W_i`` where ``W_i`` is the background
    field's own divergence-of-stress term (the ``sigma_sum`` returned by
    :func:`_update_velocities_terms`).

    ``bg_models``/``sc_models`` are the prepared model lists
    ``[lamb, mu, mu_zy, mu_zx, mu_yx, buoyancy_z, buoyancy_y, buoyancy_x]``
    (3D; fewer components in 1D/2D) and their scattering-model images
    (``sc_models`` = ``[dlamb, dmu, dmu_zy, dmu_zx, dmu_yx, dbuoyancy_z,
    dbuoyancy_y, dbuoyancy_x]``), i.e. exactly what
    :func:`prepare_scatter_parameters` (plus passthrough ``dlamb``/``dmu``)
    produces.
    """
    ndim = len(grid_spacing)

    sigma_sum, s_mem_vars_normal, s_mem_vars_shear = _update_velocities_terms(
        bg_models,
        normal_stresses,
        shear_stresses,
        s_mem_vars_normal,
        s_mem_vars_shear,
        pml_profiles,
        grid_spacing,
        dt,
        accuracy,
    )
    sigma_sum_sc, s_mem_vars_normal_sc, s_mem_vars_shear_sc = _update_velocities_terms(
        bg_models,
        normal_stresses_sc,
        shear_stresses_sc,
        s_mem_vars_normal_sc,
        s_mem_vars_shear_sc,
        pml_profiles,
        grid_spacing,
        dt,
        accuracy,
    )

    buoyancy = bg_models[-ndim:]
    dbuoyancy = sc_models[-ndim:]

    velocities = [
        velocities[dim] + buoyancy[dim] * dt * sigma_sum[dim] for dim in range(ndim)
    ]
    velocities_sc = [
        velocities_sc[dim]
        + buoyancy[dim] * dt * sigma_sum_sc[dim]
        + dbuoyancy[dim] * dt * sigma_sum[dim]
        for dim in range(ndim)
    ]

    return (
        velocities,
        velocities_sc,
        s_mem_vars_normal,
        s_mem_vars_shear,
        s_mem_vars_normal_sc,
        s_mem_vars_shear_sc,
    )


def _update_stresses_terms(
    models: list[torch.Tensor],
    velocities: list[torch.Tensor],
    v_mem_vars_normal: list[torch.Tensor],
    v_mem_vars_shear: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[
    list[torch.Tensor],
    list[list[torch.Tensor]],
    torch.Tensor,
    list[torch.Tensor],
    list[list[torch.Tensor]],
]:
    """Like ``elastic.update_stresses``, but returns the pre-model terms.

    Returns ``dvxdx`` (per-dimension normal strain rate, ``Dv_k`` in the
    derivation), ``dvxdy`` (per-(dim1,dim2) shear strain-rate component),
    and ``v_strain_sum`` (``sum_k Dv_k``) -- the exact quantities
    ``elastic.c`` checkpoints as ``dv{k}d{k}_store_1/dt`` and (summed) the
    dilatation used by ``grad_lamb``. As with
    :func:`_update_velocities_terms`, this is a line-for-line copy of
    ``elastic.update_stresses`` up to the point where ``lamb``/``mu``/
    ``mu_ij`` would be multiplied in.
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

    v_strain_sum = torch.sum(torch.stack(dvxdx), dim=0)

    return dvxdx, dvxdy, v_strain_sum, v_mem_vars_normal, v_mem_vars_shear


def update_stresses_born(
    bg_models: list[torch.Tensor],
    sc_models: list[torch.Tensor],
    velocities: list[torch.Tensor],
    velocities_sc: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    normal_stresses_sc: list[torch.Tensor],
    shear_stresses_sc: list[list[torch.Tensor]],
    v_mem_vars_normal: list[torch.Tensor],
    v_mem_vars_shear: list[list[torch.Tensor]],
    v_mem_vars_normal_sc: list[torch.Tensor],
    v_mem_vars_shear_sc: list[list[torch.Tensor]],
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
    list[torch.Tensor],
    list[list[torch.Tensor]],
]:
    """Coupled background+scattered stress update (derivation SS2.4/SS2.5).

    ``normal_stresses``/``shear_stresses`` update exactly as the nonlinear
    ``elastic`` propagator. The scattered stresses obey the same operator
    (background models, scattered velocities) plus the Born virtual
    injections ``dlamb * sum_k(Dv_k) + 2*dmu*Dv_i`` (normal) and
    ``dmu_ij * (Dv_j^i + Dv_i^j)`` (shear), where ``Dv_k``/``Dv_j^i`` are the
    *background* field's own strain-rate terms.
    """
    ndim = len(grid_spacing)

    dvxdx, dvxdy, v_strain_sum, v_mem_vars_normal, v_mem_vars_shear = (
        _update_stresses_terms(
            bg_models,
            velocities,
            v_mem_vars_normal,
            v_mem_vars_shear,
            pml_profiles,
            grid_spacing,
            dt,
            accuracy,
        )
    )
    dvxdx_sc, dvxdy_sc, v_strain_sum_sc, v_mem_vars_normal_sc, v_mem_vars_shear_sc = (
        _update_stresses_terms(
            bg_models,
            velocities_sc,
            v_mem_vars_normal_sc,
            v_mem_vars_shear_sc,
            pml_profiles,
            grid_spacing,
            dt,
            accuracy,
        )
    )

    lamb, mu = bg_models[0], bg_models[1]
    dlamb, dmu = sc_models[0], sc_models[1]

    normal_stresses = [
        normal_stresses[dim] + dt * (lamb * v_strain_sum + 2 * mu * dvxdx[dim])
        for dim in range(ndim)
    ]
    normal_stresses_sc = [
        normal_stresses_sc[dim]
        + dt * (lamb * v_strain_sum_sc + 2 * mu * dvxdx_sc[dim])
        + dt * (dlamb * v_strain_sum + 2 * dmu * dvxdx[dim])
        for dim in range(ndim)
    ]

    if ndim > 1:
        empty = torch.empty(0)
        if ndim == 3:
            mu_zy, mu_zx, mu_yx = bg_models[2], bg_models[3], bg_models[4]
            dmu_zy, dmu_zx, dmu_yx = sc_models[2], sc_models[3], sc_models[4]
            shear_models = [
                [empty, mu_zy, mu_zx],
                [mu_zy, empty, mu_yx],
                [mu_zx, mu_yx, empty],
            ]
            dshear_models = [
                [empty, dmu_zy, dmu_zx],
                [dmu_zy, empty, dmu_yx],
                [dmu_zx, dmu_yx, empty],
            ]
        else:  # ndim == 2
            mu_yx = bg_models[2]
            dmu_yx = sc_models[2]
            shear_models = [[empty, mu_yx], [mu_yx, empty]]
            dshear_models = [[empty, dmu_yx], [dmu_yx, empty]]

        for dim1 in range(ndim):
            for dim2 in range(dim1 + 1, ndim):
                strain_sum_bg = dvxdy[dim1][dim2] + dvxdy[dim2][dim1]
                strain_sum_sc = dvxdy_sc[dim1][dim2] + dvxdy_sc[dim2][dim1]

                shear_stresses[dim1][dim2] = (
                    shear_stresses[dim1][dim2]
                    + dt * shear_models[dim1][dim2] * strain_sum_bg
                )
                shear_stresses[dim2][dim1] = shear_stresses[dim1][dim2]

                shear_stresses_sc[dim1][dim2] = (
                    shear_stresses_sc[dim1][dim2]
                    + dt * shear_models[dim1][dim2] * strain_sum_sc
                    + dt * dshear_models[dim1][dim2] * strain_sum_bg
                )
                shear_stresses_sc[dim2][dim1] = shear_stresses_sc[dim1][dim2]

    return (
        normal_stresses,
        shear_stresses,
        normal_stresses_sc,
        shear_stresses_sc,
        v_mem_vars_normal,
        v_mem_vars_shear,
        v_mem_vars_normal_sc,
        v_mem_vars_shear_sc,
    )


# ``torch.compile`` Python-backend slots: the coupled velocity/stress
# step functions above are already standalone (no closures over Python
# wavefield state -- every tensor is an explicit argument), so they can be
# ``torch.compile``d directly, mirroring
# :mod:`deepwave.elastic_vti`'s ``_update_velocities_opt``/
# ``_update_stresses_opt`` slots.
_update_velocities_born_cache = _engine.StepCache(update_velocities_born)
_update_velocities_born_opt = update_velocities_born
_update_stresses_born_cache = _engine.StepCache(update_stresses_born)
_update_stresses_born_opt = update_stresses_born


# ---------------------------------------------------------------------------
# Naming / declarative spec (mirrors acoustic_born + elastic).
# ---------------------------------------------------------------------------


def _dim_names(ndim: int) -> list[str]:
    """Returns the spatial dimension names (slowest to fastest)."""
    return ["z", "y", "x"][-ndim:]


def elastic_born_model_names(ndim: int) -> list[str]:
    """Names of the elastic Born models: background then scattering.

    The background block is exactly :func:`elastic.elastic_model_names`; the
    scattering block is the same names prefixed ``d`` (the perturbations of
    the prepared model set the kernel consumes, per derivation S3). One
    spelling for a Born perturbation everywhere: ``dlamb``, ``dmu_yx``,
    ``dbuoyancy_y``, matching the propagator's own ``dlamb``/``dmu``/
    ``dbuoyancy`` arguments and :func:`deepwave.acoustic_born`'s internal
    ``dK``/``dB*``, rather than a second ``_sc`` convention.
    """
    base = cast("list[str]", _elastic_module.elastic_model_names(ndim))
    return base + [f"d{name}" for name in base]


# The user-facing model arguments, in the order the public function takes
# them: the three background models, then the three perturbations.
USER_MODEL_NAMES = ("lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is
    ``(lamb, mu, buoyancy, dlamb, dmu, dbuoyancy)``; the result is in
    :func:`elastic_born_model_names` order, which is the background elastic
    models followed by their scattered counterparts. The off-diagonal shear
    moduli (one in 2-D, three in 3-D) are staggered harmonic averages of
    ``mu``, and the scattered ones depend on ``mu`` as well as ``dmu``.
    """
    lamb, mu, buoyancy, dlamb, dmu, dbuoyancy = model_requires_grad
    n_offdiagonal = {1: 0, 2: 1, 3: 3}[ndim]
    background = [lamb, mu] + [mu] * n_offdiagonal + [buoyancy] * ndim
    scattered = (
        [dlamb, dmu] + [mu or dmu] * n_offdiagonal + [buoyancy or dbuoyancy] * ndim
    )
    return background + scattered


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    Cross-wired like :func:`deepwave.acoustic_born`'s: the background
    history feeds every model's gradient, the scattered history only the
    background models'. The count per half is ``elastic``'s own slot count
    at this dimensionality (2, 5 and 9 for ndim 1, 2 and 3), so in 3-D this
    is 18 slots against only 16 internal models -- the slot count is not
    bounded by the model count.

    Measured against the engine over every combination of the six models at
    ndim 1, 2 and 3.
    """
    lamb, mu, buoyancy, dlamb, dmu, dbuoyancy = model_requires_grad
    any_background = lamb or mu or buoyancy
    any_model = any_background or dlamb or dmu or dbuoyancy
    per_half = {1: 2, 2: 5, 3: 9}[ndim]
    return [any_model] * per_half + [any_background] * per_half


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models."""
    n_background = 2 + {1: 0, 2: 1, 3: 3}[ndim] + ndim
    return [
        models[0].requires_grad,
        models[1].requires_grad,
        models[2 + {1: 0, 2: 1, 3: 3}[ndim]].requires_grad,
        models[n_background].requires_grad,
        models[n_background + 1].requires_grad,
        models[n_background + 2 + {1: 0, 2: 1, 3: 3}[ndim]].requires_grad,
    ]


def elastic_born_wavefield_names(ndim: int) -> list[str]:
    """Names of the elastic Born wavefields, in backend order.

    The background wavefields (in :func:`elastic.elastic_wavefield_names`
    order) are followed by the corresponding scattered wavefields, each
    spelled by inserting ``_sc`` before the time index
    (:func:`deepwave._engine.scattered_state_name`); the Born set is thus
    doubled (26 in 2D, 54 in 3D).
    """
    base = cast("list[str]", _elastic_module.elastic_wavefield_names(ndim))
    return base + [_scattered_name(name) for name in base]


def _public_wavefield_order(ndim: int) -> list[int]:
    """Backend->public wavefield permutation (see elastic; reproduced here).

    Applied per field block (background, scattered). Kept local to avoid
    reaching into ``elastic``'s private ``_elastic_public_wavefield_order``.
    """
    if ndim == 3:
        return [
            0,
            14,
            23,
            1,
            2,
            3,
            15,
            16,
            24,
            4,
            5,
            6,
            7,
            8,
            17,
            18,
            19,
            25,
            9,
            10,
            11,
            12,
            13,
            20,
            21,
            22,
            26,
        ]
    if ndim == 2:
        return [0, 9, 1, 2, 10, 3, 4, 5, 11, 6, 7, 8, 12]
    return list(range(len(_elastic_module.elastic_wavefield_names(ndim))))


def elastic_born_public_wavefield_names(ndim: int) -> list[str]:
    """The elastic Born state fields in public return order and spelling.

    The permutation :func:`deepwave.elastic` applies to its own wavefields
    (grouping them by kind rather than by spatial dimension) is applied to the
    background block and again to the scattered block. Every name is already
    its public spelling -- there is one Born spelling, ``_sc`` before the time
    index (see :func:`elastic_born_wavefield_names`) -- so only the order
    differs from the backend's.
    """
    order = _public_wavefield_order(ndim)
    base = cast("list[str]", _elastic_module.elastic_wavefield_names(ndim))
    bg_wf = [base[i] for i in order]
    sc_wf = [_scattered_name(base[i]) for i in order]
    return bg_wf + sc_wf


def elastic_born_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The CFS-CPML memory variables ``m_*`` of the background field and their
    ``_sc`` counterparts of the scattered field, one per velocity or stress
    spatial derivative that the absorbing boundary integrates. They are a
    subset of :func:`elastic_born_wavefield_names`, spelled the same way, and
    are what :class:`deepwave.AuxiliaryState` holds. Listed in the public
    (grouped-by-kind, background then scattered) order, which is the order the
    auxiliary state itself holds them in.
    """
    return [
        name
        for name in elastic_born_public_wavefield_names(ndim)
        if name.startswith("m_")
    ]


def elastic_born_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`elastic_born`.

    The background block first, then the scattered block; within each,
    pressure first then one per velocity component (z, y, x, for the
    dimensions present).
    """
    receivers = ["receiver_amplitudes_p"] + [
        f"receiver_amplitudes_{d}" for d in _dim_names(ndim)
    ]
    return ["bg_" + name for name in receivers] + ["sc_" + name for name in receivers]


def elastic_born_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`elastic_born`, in return order.

    The public final wavefields -- the background velocities and stresses, the
    single ``auxiliary_state`` that replaces both blocks of PML memory
    variables at the position of the first of them, and then the scattered
    velocities and stresses -- followed by the background and then the
    scattered receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            elastic_born_public_wavefield_names(ndim),
            elastic_born_auxiliary_state_names(ndim),
        ),
        *elastic_born_receiver_names(ndim),
    ]


SPEC = _engine.PropagatorSpec(
    name="elastic_born",
    status="experimental",
    source_channels=_engine.pressure_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    bg_receiver_channels=_engine.pressure_velocity_channels,
    supported_options=frozenset({"mpml_fraction"}),
    staggered=True,
    model_names=elastic_born_model_names,
    wavefield_names=elastic_born_wavefield_names,
    n_source_types=lambda ndim: 2 * (ndim + 1),
    n_receiver_types=lambda ndim: 2 * (ndim + 1),
    output_names=elastic_born_output_names,
    auxiliary_state_names=elastic_born_auxiliary_state_names,
    public_wavefield_names=elastic_born_public_wavefield_names,
    spatial_parallel=True,
)


# ---------------------------------------------------------------------------
# Wavefield (de)structuring -- identical layout to elastic._elastic_eager_core,
# factored so the doubled Born field set reuses it for both blocks.
# ---------------------------------------------------------------------------

_StructuredWavefields = tuple[
    list[torch.Tensor],  # velocities
    list[torch.Tensor],  # normal_stresses
    list[list[torch.Tensor]],  # shear_stresses (symmetric: [i][j] == [j][i])
    list[torch.Tensor],  # v_mem_vars_normal
    list[list[torch.Tensor]],  # v_mem_vars_shear (directional, NOT symmetric)
    list[torch.Tensor],  # s_mem_vars_normal
    list[list[torch.Tensor]],  # s_mem_vars_shear (directional, NOT symmetric)
]


def _structure_wavefields(wf: list[torch.Tensor], ndim: int) -> _StructuredWavefields:
    """Groups a flat backend-order wavefield list into the elastic containers.

    The flat order and container layout match
    :func:`elastic._elastic_eager_core` exactly (see that function's inline
    comments for the shear/memory-variable index conventions).
    """
    e = torch.empty(0)
    if ndim == 3:
        velocities = [wf[0], wf[14], wf[23]]
        normal_stresses = [wf[1], wf[15], wf[24]]
        s_yz, s_xz, s_xy = wf[2], wf[3], wf[16]
        shear_stresses = [
            [e, s_yz, s_xz],
            [s_yz, e, s_xy],
            [s_xz, s_xy, e],
        ]
        v_mem_vars_normal = [wf[4], wf[17], wf[25]]
        v_mem_vars_shear = [
            [e, wf[5], wf[6]],
            [wf[7], e, wf[18]],
            [wf[8], wf[19], e],
        ]
        s_mem_vars_normal = [wf[9], wf[20], wf[26]]
        s_mem_vars_shear = [
            [e, wf[10], wf[11]],
            [wf[12], e, wf[22]],
            [wf[13], wf[21], e],
        ]
    elif ndim == 2:
        velocities = [wf[0], wf[9]]
        normal_stresses = [wf[1], wf[10]]
        s_xy = wf[2]
        shear_stresses = [[e, s_xy], [s_xy, e]]
        v_mem_vars_normal = [wf[3], wf[11]]
        v_mem_vars_shear = [[e, wf[4]], [wf[5], e]]
        s_mem_vars_normal = [wf[6], wf[12]]
        s_mem_vars_shear = [[e, wf[8]], [wf[7], e]]
    else:
        velocities = [wf[0]]
        normal_stresses = [wf[1]]
        shear_stresses = [[]]
        v_mem_vars_normal = [wf[2]]
        v_mem_vars_shear = [[]]
        s_mem_vars_normal = [wf[3]]
        s_mem_vars_shear = [[]]
    return (
        velocities,
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
        s_mem_vars_normal,
        s_mem_vars_shear,
    )


def _flatten_wavefields(
    struct: _StructuredWavefields, ndim: int, s: tuple[Any, ...]
) -> list[torch.Tensor]:
    """Inverse of :func:`_structure_wavefields`; applies the crop slice ``s``.

    Produces the flat backend-order list (matching
    :func:`elastic.elastic_wavefield_names`) so the Born background block is
    byte-identical in layout to plain ``elastic``'s outputs.
    """
    (
        velocities,
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
        s_mem_vars_normal,
        s_mem_vars_shear,
    ) = struct
    if ndim == 3:
        flat = [
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
    elif ndim == 2:
        flat = [
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
    else:
        flat = [
            velocities[0],
            normal_stresses[0],
            v_mem_vars_normal[0],
            s_mem_vars_normal[0],
        ]
    return [field[s] for field in flat]


# ---------------------------------------------------------------------------
# Eager Born driver (doubled background + scattered fields in one pass).
# ---------------------------------------------------------------------------


def _record_receivers(
    receivers_i_masked: list[torch.Tensor],
    velocities: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    flat_model_shape: int,
    ndim: int,
    t: int,
    receiver_amplitudes: list[torch.Tensor],
    receiver_amplitudes_lists: list[list[torch.Tensor]],
    is_batched: bool,
) -> None:
    """Records velocity + (summed normal-stress) receiver samples at step ``t``.

    The pressure receiver stores the raw ``sum_i sigma_ii`` (the front-end
    forms ``-sum/ndim``), matching :func:`elastic._elastic_eager_core`.
    """
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


def _inject_velocity_sources(
    velocities: list[torch.Tensor],
    source_amplitudes_masked: list[torch.Tensor],
    sources_i_masked: list[torch.Tensor],
    ndim: int,
    t: int,
    flat_model_shape: int,
    size_with_batch: tuple[int, ...],
) -> list[torch.Tensor]:
    """Adds the (pre-scaled) velocity/force source samples for step ``t``."""
    out = list(velocities)
    for sidx in range(ndim):
        if source_amplitudes_masked[sidx].numel() > 0:
            out[sidx] = (
                out[sidx]
                .clone()
                .reshape(-1, flat_model_shape)
                .scatter_add_(
                    1, sources_i_masked[sidx], source_amplitudes_masked[sidx][t]
                )
                .reshape(size_with_batch)
            )
    return out


def _inject_pressure_source(
    normal_stresses: list[torch.Tensor],
    source_amplitudes_masked_p: torch.Tensor,
    sources_i_masked_p: torch.Tensor,
    ndim: int,
    t: int,
    flat_model_shape: int,
    size_with_batch: tuple[int, ...],
) -> list[torch.Tensor]:
    """Adds the (pre-scaled) pressure source sample equally to all sigma_ii."""
    if source_amplitudes_masked_p.numel() == 0:
        return normal_stresses
    out = list(normal_stresses)
    for idx in range(ndim):
        out[idx] = (
            out[idx]
            .clone()
            .reshape(-1, flat_model_shape)
            .scatter_add_(1, sources_i_masked_p, source_amplitudes_masked_p[t])
            .reshape(size_with_batch)
        )
    return out


def elastic_born_eager_core(
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    boundary: list[int],
    n_shots: int,
    mpml: bool = False,
    *args: torch.Tensor,
    forward_callback: Optional[Callback] = None,
    callback_frequency: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Eager coupled background+scattered elastic Born driver.

    Positional ``args`` layout (all produced by :func:`elastic_born`):
    ``bg_models``, ``sc_models`` (each ``n_models`` prepared models),
    ``bg_source_amplitudes``, ``sc_source_amplitudes`` (each ``ndim+1``,
    pre-scaled), ``sources_i`` (``ndim+1`` shared locations),
    ``bg_receivers_i``, ``sc_receivers_i`` (each ``ndim+1``), then the
    background and scattered initial wavefields (each ``n_wf``).

    Drives :func:`update_velocities_born`/:func:`update_stresses_born` for the
    full time loop, recording background then scattered receivers.

    ``forward_callback``, if not None, is called every ``callback_frequency``
    coarse steps (of ``step_ratio`` fine time steps each) with a
    :class:`deepwave.CallbackState` whose wavefields dict is keyed by
    :func:`elastic_born_wavefield_names` and whose models dict is keyed by
    :func:`elastic_born_model_names`, mirroring
    ``elastic._elastic_eager_core``'s own callback firing.
    Defaulting to None keeps this internal-bridge callers (the jvp M-PML
    fallback in :mod:`deepwave.elastic`, and the eager-autograd backward
    bridge in :class:`ElasticBornForwardFunc`) unchanged, since neither passes
    a callback. There is no ``backward_callback`` parameter: this backend
    runs under plain autograd, so it has no adjoint pass of its own to hook
    into (the caller rejects a ``backward_callback`` before reaching here).

    ``boundary`` is the per-face ``BC_*`` integer code list (``2 * ndim``
    entries, ``BC_PML`` everywhere for the default/no-boundary case); non-PML
    faces are guard-filled each half-step via
    ``_boundaries.fill_elastic_stress_boundary``/``fill_elastic_velocity_boundary``
    (the same functions ``elastic._elastic_eager_core`` uses),
    applied independently to the background then scattered field -- both
    obey the identical isotropic elastic system on the same grid (see the
    module docstring), so the same per-field image parity applies to each
    without a new physics derivation.

    ``mpml`` sits in the same position as :func:`elastic_born_compiled_core`'s
    (right after ``n_shots``) so a single argument list drives either core.
    It is accepted for signature parity but is
    unused here (mirrors ``elastic._elastic_eager_core``): the
    eager backend consumes the ``pml_profiles`` -- already multiaxial when
    ``mpml`` is true -- directly via broadcasting, so no separate M-PML code
    path is needed. The interior-zeroing that ``mpml`` would otherwise switch
    is disabled on the eager path anyway (``interior=False`` below).

    Deepwave-internal API consumed by the base propagator's forward-mode (jvp)
    and second-order (HVP) rules; not a public interface.
    """
    del mpml
    has_boundary = any(code != _common.BC_PML for code in boundary)
    ndim = len(grid_spacing)
    is_batched = any(_common.is_inside_vmap(x) for x in args)
    n_wf = len(_elastic_module.elastic_wavefield_names(ndim))
    n_models = len(_elastic_module.elastic_model_names(ndim))
    n_src = ndim + 1
    args_list = list(args)
    del args

    idx = 0

    def take(n: int) -> list[torch.Tensor]:
        nonlocal idx
        out = args_list[idx : idx + n]
        idx += n
        return out

    bg_models = take(n_models)
    sc_models = take(n_models)
    bg_source_amplitudes = take(n_src)
    sc_source_amplitudes = take(n_src)
    sources_i = take(n_src)
    bg_receivers_i = take(n_src)
    sc_receivers_i = take(n_src)
    bg_wavefields = take(n_wf)
    sc_wavefields = take(n_wf)

    device = bg_models[0].device
    dtype = bg_models[0].dtype
    model_shape = bg_models[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *model_shape)

    def pad_and_zero(wf: list[torch.Tensor]) -> list[torch.Tensor]:
        wf = [
            _engine.create_or_pad(w, fd_pad_list, device, dtype, size_with_batch)
            for w in wf
        ]
        _elastic_module.zero_edges_and_interiors(
            wf, ndim, fd_pad, fd_pad_list, pml_width, False
        )
        return wf

    bg_wavefields = pad_and_zero(bg_wavefields)
    sc_wavefields = pad_and_zero(sc_wavefields)

    (
        velocities,
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
        s_mem_vars_normal,
        s_mem_vars_shear,
    ) = _structure_wavefields(bg_wavefields, ndim)
    (
        velocities_sc,
        normal_stresses_sc,
        shear_stresses_sc,
        v_mem_vars_normal_sc,
        v_mem_vars_shear_sc,
        s_mem_vars_normal_sc,
        s_mem_vars_shear_sc,
    ) = _structure_wavefields(sc_wavefields, ndim)

    # Receiver output buffers (background block then scattered block).
    def make_recv(
        receivers_i: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]], list[int]]:
        npers = [loc.numel() // n_shots for loc in receivers_i]
        amps: list[torch.Tensor] = [
            torch.empty(0, device=device, dtype=dtype) for _ in range(n_src)
        ]
        for i, loc in enumerate(receivers_i):
            if loc.numel() > 0 and not is_batched:
                amps[i] = torch.zeros(nt, n_shots, npers[i], device=device, dtype=dtype)
        lists: list[list[torch.Tensor]] = [[] for _ in range(n_src)]
        return amps, lists, npers

    bg_amps, bg_amp_lists, _ = make_recv(bg_receivers_i)
    sc_amps, sc_amp_lists, _ = make_recv(sc_receivers_i)

    # Mask sources (shared locations) and receivers (per block).
    sources_i_masked = []
    bg_src_masked = []
    sc_src_masked = []
    for i, loc in enumerate(sources_i):
        masked, src_mask = _common.mask_locations(loc)
        sources_i_masked.append(masked)
        bg_src_masked.append(
            _common.mask_source_amplitudes(bg_source_amplitudes[i], src_mask)
        )
        sc_src_masked.append(
            _common.mask_source_amplitudes(sc_source_amplitudes[i], src_mask)
        )

    def mask_recv(
        receivers_i: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        masked_list = []
        mask_list = []
        for loc in receivers_i:
            masked, rmask = _common.mask_locations(loc)
            masked_list.append(masked)
            mask_list.append(rmask)
        return masked_list, mask_list

    bg_recv_masked, bg_recv_mask = mask_recv(bg_receivers_i)
    sc_recv_masked, sc_recv_mask = mask_recv(sc_receivers_i)

    wf_names = elastic_born_wavefield_names(ndim)
    model_names = elastic_born_model_names(ndim)
    callback_models = dict(zip(model_names, [*bg_models, *sc_models]))
    # Identity crop (no fd_pad trim): the callback sees the live, still-padded
    # wavefields, mirroring deepwave.elastic._elastic_eager_core's callback.
    callback_slice: tuple[Any, ...] = (slice(None),) * (ndim + 1)

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_bg_struct: _StructuredWavefields = (
                velocities,
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
                s_mem_vars_normal,
                s_mem_vars_shear,
            )
            callback_sc_struct: _StructuredWavefields = (
                velocities_sc,
                normal_stresses_sc,
                shear_stresses_sc,
                v_mem_vars_normal_sc,
                v_mem_vars_shear_sc,
                s_mem_vars_normal_sc,
                s_mem_vars_shear_sc,
            )
            current_wavefields = [
                *_flatten_wavefields(callback_bg_struct, ndim, callback_slice),
                *_flatten_wavefields(callback_sc_struct, ndim, callback_slice),
            ]
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
                and any(
                    t.requires_grad
                    for t in [
                        *bg_models,
                        *sc_models,
                        *bg_src_masked,
                        *sc_src_masked,
                        *current_wavefields,
                    ]
                ),
            )
        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step

            # Refill the stress guard ring of each field independently (read
            # by the velocity update, and by the pressure receiver record
            # below); on a free face this also holds the on-surface
            # normal-stress node to zero. Mirrors
            # deepwave.elastic._elastic_eager_core's placement.
            if has_boundary:
                normal_stresses, shear_stresses = (
                    _boundaries.fill_elastic_stress_boundary(
                        normal_stresses, shear_stresses, boundary, fd_pad
                    )
                )
                normal_stresses_sc, shear_stresses_sc = (
                    _boundaries.fill_elastic_stress_boundary(
                        normal_stresses_sc, shear_stresses_sc, boundary, fd_pad
                    )
                )

            _record_receivers(
                bg_recv_masked,
                velocities,
                normal_stresses,
                flat_model_shape,
                ndim,
                t,
                bg_amps,
                bg_amp_lists,
                is_batched,
            )
            _record_receivers(
                sc_recv_masked,
                velocities_sc,
                normal_stresses_sc,
                flat_model_shape,
                ndim,
                t,
                sc_amps,
                sc_amp_lists,
                is_batched,
            )

            (
                velocities,
                velocities_sc,
                s_mem_vars_normal,
                s_mem_vars_shear,
                s_mem_vars_normal_sc,
                s_mem_vars_shear_sc,
            ) = _update_velocities_born_opt(
                bg_models,
                sc_models,
                normal_stresses,
                shear_stresses,
                normal_stresses_sc,
                shear_stresses_sc,
                velocities,
                velocities_sc,
                s_mem_vars_normal,
                s_mem_vars_shear,
                s_mem_vars_normal_sc,
                s_mem_vars_shear_sc,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            velocities = _inject_velocity_sources(
                velocities,
                bg_src_masked,
                sources_i_masked,
                ndim,
                t,
                flat_model_shape,
                size_with_batch,
            )
            velocities_sc = _inject_velocity_sources(
                velocities_sc,
                sc_src_masked,
                sources_i_masked,
                ndim,
                t,
                flat_model_shape,
                size_with_batch,
            )

            # Refill the velocity guard ring of each field independently
            # (read by every normal stress's own divergence term and by the
            # shared-axis shears' updates); on a free face every
            # component's ghost region is zeroed. Mirrors
            # deepwave.elastic._elastic_eager_core's placement.
            if has_boundary:
                velocities = _boundaries.fill_elastic_velocity_boundary(
                    velocities, boundary, fd_pad
                )
                velocities_sc = _boundaries.fill_elastic_velocity_boundary(
                    velocities_sc, boundary, fd_pad
                )

            (
                normal_stresses,
                shear_stresses,
                normal_stresses_sc,
                shear_stresses_sc,
                v_mem_vars_normal,
                v_mem_vars_shear,
                v_mem_vars_normal_sc,
                v_mem_vars_shear_sc,
            ) = _update_stresses_born_opt(
                bg_models,
                sc_models,
                velocities,
                velocities_sc,
                normal_stresses,
                shear_stresses,
                normal_stresses_sc,
                shear_stresses_sc,
                v_mem_vars_normal,
                v_mem_vars_shear,
                v_mem_vars_normal_sc,
                v_mem_vars_shear_sc,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            normal_stresses = _inject_pressure_source(
                normal_stresses,
                bg_src_masked[-1],
                sources_i_masked[-1],
                ndim,
                t,
                flat_model_shape,
                size_with_batch,
            )
            normal_stresses_sc = _inject_pressure_source(
                normal_stresses_sc,
                sc_src_masked[-1],
                sources_i_masked[-1],
                ndim,
                t,
                flat_model_shape,
                size_with_batch,
            )

    # Leave the returned wavefields boundary-consistent (e.g. free-surface
    # stress nodes held to zero, guard rings imaged) so a continuation run
    # reproduces the in-loop state, mirroring
    # deepwave.elastic._elastic_eager_core's own post-loop refill.
    if has_boundary:
        normal_stresses, shear_stresses = _boundaries.fill_elastic_stress_boundary(
            normal_stresses, shear_stresses, boundary, fd_pad
        )
        normal_stresses_sc, shear_stresses_sc = (
            _boundaries.fill_elastic_stress_boundary(
                normal_stresses_sc, shear_stresses_sc, boundary, fd_pad
            )
        )
        velocities = _boundaries.fill_elastic_velocity_boundary(
            list(velocities), boundary, fd_pad
        )
        velocities_sc = _boundaries.fill_elastic_velocity_boundary(
            list(velocities_sc), boundary, fd_pad
        )

    if is_batched:
        for amps, lists in ((bg_amps, bg_amp_lists), (sc_amps, sc_amp_lists)):
            for i in range(n_src):
                if len(lists[i]) > 0:
                    amps[i] = torch.stack(lists[i])

    bg_amps = [
        _common.mask_receiver_amplitudes(amp, bg_recv_mask[i])
        for i, amp in enumerate(bg_amps)
    ]
    sc_amps = [
        _common.mask_receiver_amplitudes(amp, sc_recv_mask[i])
        for i, amp in enumerate(sc_amps)
    ]

    s = (
        slice(None),
        *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
    )
    bg_struct: _StructuredWavefields = (
        velocities,
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
        s_mem_vars_normal,
        s_mem_vars_shear,
    )
    sc_struct: _StructuredWavefields = (
        velocities_sc,
        normal_stresses_sc,
        shear_stresses_sc,
        v_mem_vars_normal_sc,
        v_mem_vars_shear_sc,
        s_mem_vars_normal_sc,
        s_mem_vars_shear_sc,
    )
    bg_final = _flatten_wavefields(bg_struct, ndim, s)
    sc_final = _flatten_wavefields(sc_struct, ndim, s)

    return (*bg_final, *sc_final, *bg_amps, *sc_amps)


# ---------------------------------------------------------------------------
# Compiled (C/CUDA) dispatch.
#
# :class:`ElasticBornForwardFunc` runs the compiled forward on the shared
# step-based engine; its ``backward`` uses the compiled scattered-only and full
# two-field adjoint kernels (elastic_born.c/.cu's ``backward_sc``/``backward``,
# derivation S4.1-4.3) without M-PML, on CPU and CUDA. The public API selects
# the eager backend before calling this Function when M-PML is requested. The
# forward takes exactly the eager core's
# positional layout, plus a leading ``mpml`` flag (and the callback/storage
# leading params), so both cores are driven from one argument list.
#
# ``elastic_born_compiled_core`` below is a thin bridge kept ONLY because
# :mod:`deepwave.elastic`'s jvp/HVP rules call it (and
# ``elastic_born_backward_compiled``) BY NAME with a fixed legacy signature; it
# builds a throwaway StorageManager over the per-slot engine kernel and returns
# the legacy packed-store tensor those callers expect.
# ---------------------------------------------------------------------------


def elastic_born_compiled_core(
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    boundary: list[int],
    n_shots: int,
    mpml: bool,
    *args: torch.Tensor,
    store_bg: bool = False,
    store_sc: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Compiled coupled background+scattered elastic Born forward driver.

    ``boundary`` is the per-face ``BC_*`` integer code list (see
    :func:`deepwave._common.set_boundary`); threaded through to the
    compiled backend's ``bc`` ABI parameter exactly as
    ``elastic._elastic_make_forward_call_pm`` does.

    Positional ``args`` layout is identical to :func:`elastic_born_eager_core`
    (``bg_models``, ``sc_models``, ``bg_source_amplitudes``,
    ``sc_source_amplitudes``, shared ``sources_i``, ``bg_receivers_i``,
    ``sc_receivers_i``, background then scattered initial wavefields). Calls
    the ``elastic_born_iso_*_forward_*`` C/CUDA kernel once (a single chunk,
    ``start_t=0``) over the full internal time loop.

    Post-M1 this is a thin bridge over the shared engine's
    :class:`deepwave._storage.StorageManager`, kept ONLY because
    :mod:`deepwave.elastic` (``ElasticForwardFunc.jvp``/the HVP path) and
    :class:`ElasticBornForwardFunc`'s own step-based ``backward``/``backward_sc``
    regenerate-stores re-run call this function BY NAME with this EXACT signature
    and return contract -- it must keep working unchanged for them. When
    ``store_bg``/``store_sc`` is requested, it builds a throwaway
    ``storage_mode="device"`` :class:`~deepwave._storage.StorageManager`
    (``N_BG_STORE`` engine storage slots per side), drives the per-slot-storage
    compiled forward kernel once, then ``torch.stack``s each side's per-slot
    device tensors back into the single packed ``[N_BG_STORE, nt // step_ratio,
    n_shots, *grid]`` tensor this function's callers still expect (the exact
    packed shape the pre-M1 forward produced directly; for ``step_ratio == 1``,
    ``nt // step_ratio == nt``, so this is bit-identical to the pre-M1
    behaviour). ``step_ratio`` is genuinely forwarded to the kernel: for
    ``step_ratio > 1`` the resulting packed store is correspondingly smaller
    than ``nt`` (one snapshot per ``step_ratio``-sized group), and the
    step-based ``backward``/``backward_sc`` kernels consume it correctly -- they
    gate every gradient-store write/read and gradient-accumulation flag on
    ``t % step_ratio == 0`` and scale each accumulated gradient by
    ``step_ratio`` (mirroring :mod:`deepwave.elastic` and the ``.cu`` twin).
    (A latent bug -- the C ``backward``/``backward_sc`` kernels were missing both
    the ``t % step_ratio == 0`` gate on their gradient flags and the
    ``* step_ratio`` accumulation factor, making ``step_ratio > 1`` gradients
    wrong for every non-``device`` storage mode and a different quadrature for
    ``device`` -- was fixed in elastic_born.c.)
    Returns the same output tuple as the eager core: ``(*bg_final, *sc_final,
    *bg_amps, *sc_amps[, bg_store][, sc_store])``.

    Deepwave-internal API consumed by the base propagator's forward-mode (jvp)
    and second-order (HVP) rules, and by this module's own compiled-backward
    regenerate-stores bridge; not a public interface.
    """
    ndim = len(grid_spacing)
    n_wf = len(_elastic_module.elastic_wavefield_names(ndim))
    n_models = len(_elastic_module.elastic_model_names(ndim))
    n_src = ndim + 1
    args_list = [a.contiguous() for a in args]

    idx = 0

    def take(n: int) -> list[torch.Tensor]:
        nonlocal idx
        out = args_list[idx : idx + n]
        idx += n
        return out

    bg_models = take(n_models)
    sc_models = take(n_models)
    bg_source_amplitudes = take(n_src)
    sc_source_amplitudes = take(n_src)
    sources_i = take(n_src)
    bg_receivers_i = take(n_src)
    sc_receivers_i = take(n_src)
    bg_wavefields = take(n_wf)
    sc_wavefields = take(n_wf)

    device = bg_models[0].device
    dtype = bg_models[0].dtype
    is_cuda = bg_models[0].is_cuda
    model_shape = tuple(bg_models[0].shape[-ndim:])
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *model_shape)

    def pad_and_zero(wf: list[torch.Tensor]) -> list[torch.Tensor]:
        wf = [
            _engine.create_or_pad(w, fd_pad_list, device, dtype, size_with_batch)
            for w in wf
        ]
        _elastic_module.zero_edges_and_interiors(
            wf, ndim, fd_pad, fd_pad_list, pml_width, mpml
        )
        return [w.contiguous() for w in wf]

    bg_wavefields = pad_and_zero(bg_wavefields)
    sc_wavefields = pad_and_zero(sc_wavefields)

    sources_i = [s.contiguous() for s in sources_i]
    bg_receivers_i = [r.contiguous() for r in bg_receivers_i]
    sc_receivers_i = [r.contiguous() for r in sc_receivers_i]

    n_sources_bg = [s.numel() // n_shots for s in sources_i]
    # Shared source *locations*: the scattered velocity sources share the
    # background counts (present iff the background amplitude is present); the
    # scattered pressure source is absent (empty amplitude -> count 0).
    n_sources_sc = [
        n_sources_bg[i] if sc_source_amplitudes[i].numel() > 0 else 0
        for i in range(n_src)
    ]
    n_receivers_bg = [r.numel() // n_shots for r in bg_receivers_i]
    n_receivers_sc = [r.numel() // n_shots for r in sc_receivers_i]

    def make_recv(counts: list[int]) -> list[torch.Tensor]:
        out = []
        for c in counts:
            if c > 0:
                out.append(torch.zeros(nt, n_shots, c, device=device, dtype=dtype))
            else:
                out.append(torch.empty(0, device=device, dtype=dtype))
        return out

    bg_amps = make_recv(n_receivers_bg)
    sc_amps = make_recv(n_receivers_sc)

    # Background/scattered gradient stores: N_BG_STORE (9/5/2 for 3/2/1-D)
    # engine storage slots per side, built via a throwaway device-mode
    # StorageManager and stacked back into the legacy packed
    # [N_BG_STORE, nt // step_ratio, n_shots, *grid] shape below (see the
    # updated docstring above -- this bridge exists only to keep this
    # function's legacy-signature callers (elastic's jvp/HVP and the
    # step-based backward regenerate-stores re-run) working unchanged).
    n_bg_store = {1: 2, 2: 5, 3: 9}[ndim]

    # PML region bounds (staggered forward; matches common._pml_region).
    pml_b = [
        min(pml_width[2 * i] + fd_pad, model_shape[i] - (fd_pad - 1))
        for i in range(ndim)
    ]
    pml_e = [
        max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - (fd_pad - 1))
        for i in range(ndim)
    ]

    rdx = [1.0 / dx for dx in grid_spacing]
    aux, stream = _engine.get_aux_and_stream(bg_models[0], n_shots, is_cuda)

    # M-PML never reaches the compiled core: the public forward raises
    # NotImplementedError for mpml_fraction > 0 on the compiled backend and routes M-PML
    # through the eager path instead. There is therefore no ``elastic_born_*_mpml``
    # compiled symbol; adding one is a future M-PML pass (wire the kernel template
    # in CMakeLists.txt and register the ``_mpml`` extra in backend_utils, as the
    # elastic propagator does).
    if mpml:
        raise AssertionError(
            "compiled elastic_born has no M-PML backend; use the eager path."
        )
    backend = _backend_utils.get_backend_function(
        "elastic_born",
        ndim,
        "forward",
        accuracy,
        dtype,
        device,
    )

    def batched(m: torch.Tensor) -> bool:
        return m.ndim == ndim + 1 and m.shape[0] > 1

    # Throwaway device-mode StorageManager: N_BG_STORE slots for the
    # background terms, then N_BG_STORE more for the scattered terms (matching
    # the C kernel's bg-group-then-sc-group parameter order). ``.allocate``
    # with the per-side flag False leaves that side's slots as size-0 stubs
    # (safe: the kernel gates every write on ``store_bg``/``store_sc``, so an
    # unallocated slot is never actually written).
    storage_manager = _storage.StorageManager(
        model_shape,
        dtype,
        n_shots,
        nt,
        step_ratio,
        _storage.StorageMode.DEVICE,
        False,
        "",
        device,
        is_cuda,
    )
    for _ in range(n_bg_store):
        storage_manager.allocate(store_bg)
    for _ in range(n_bg_store):
        storage_manager.allocate(store_sc)

    backend(
        *[m.data_ptr() for m in bg_models],
        *[m.data_ptr() for m in sc_models],
        *[a.data_ptr() for a in bg_source_amplitudes],
        *[a.data_ptr() for a in sc_source_amplitudes],
        *[w.data_ptr() for w in bg_wavefields],
        *[w.data_ptr() for w in sc_wavefields],
        *[a.data_ptr() for a in bg_amps],
        *[a.data_ptr() for a in sc_amps],
        *storage_manager.storage_ptrs,
        *[p.data_ptr() for p in pml_profiles],
        *[loc.data_ptr() for loc in sources_i],
        *[loc.data_ptr() for loc in bg_receivers_i],
        *[loc.data_ptr() for loc in sc_receivers_i],
        *rdx,
        dt,
        nt,
        n_shots,
        *model_shape,
        *n_sources_bg,
        *n_sources_sc,
        *n_receivers_bg,
        *n_receivers_sc,
        step_ratio,
        storage_manager.storage_mode,
        storage_manager.shot_bytes_uncomp,
        storage_manager.shot_bytes_comp,
        batched(bg_models[0]),
        batched(bg_models[1]),
        batched(bg_models[-1]),
        batched(sc_models[0]),
        batched(sc_models[1]),
        batched(sc_models[-1]),
        store_bg,
        store_sc,
        storage_manager.storage_compression,
        0,  # start_t: this bridge always runs a single, unchunked full call.
        *pml_b,
        *pml_e,
        # The C signature takes the bc codes begins-then-ends (mirroring
        # pml_b/pml_e above); ``boundary`` is in interleaved set_boundary
        # face order, so it must be reordered here, exactly as the engine
        # path and scalar.py's raw call sites do.
        *_engine.bc_begins_ends(boundary, len(model_shape)),
        aux,
        stream,
    )

    s = (
        slice(None),
        *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
    )
    # The wavefields are already in backend order; just crop the FD padding.
    bg_final = [w[s] for w in bg_wavefields]
    sc_final = [w[s] for w in sc_wavefields]

    # Test/adjoint hook: expose the requested packed stores (spatial dims cropped
    # to the interior grid, matching the wavefield crop) as trailing outputs, in
    # the order (bg_store, sc_store). Store layout is [n_store, nt // step_ratio,
    # n_shots, *grid] (stacked from the per-slot StorageManager device tensors
    # above -- for step_ratio == 1 this is bit-identical to the pre-M1
    # packed-buffer shape). When only store_bg is
    # set, ``fwd[-1]`` is bg_store (the backward_sc consumer's contract); when
    # both are set, bg is [-2], sc [-1].
    store_crop = (slice(None), slice(None), slice(None), *s[1:])
    extra: list[torch.Tensor] = []
    if store_bg:
        bg_store = torch.stack(
            [storage_manager.storages[i].store_1a for i in range(n_bg_store)],
            dim=0,
        )
        extra.append(bg_store[store_crop])
    if store_sc:
        sc_store = torch.stack(
            [
                storage_manager.storages[n_bg_store + i].store_1a
                for i in range(n_bg_store)
            ],
            dim=0,
        )
        extra.append(sc_store[store_crop])
    if extra:
        return (*bg_final, *sc_final, *bg_amps, *sc_amps, *extra)

    return (*bg_final, *sc_final, *bg_amps, *sc_amps)


def _elastic_born_backward_sc_compiled(
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    boundary: list[int],
    n_shots: int,
    mpml: bool,
    args: tuple[torch.Tensor, ...],
    grad_r_sc: list[torch.Tensor],
    sc_seeds: Optional[list[torch.Tensor]] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Compiled scattered-only elastic Born adjoint (derivation SS4.1).

    ``boundary`` is the per-face ``BC_*`` integer code list; threaded to the
    forward re-run below and to the compiled adjoint's ``bc`` ABI parameter.
    The background ([bg_models]) and scattered ([sc_models]) model groups
    are each mirror-padded independently (:func:`deepwave._boundaries.\
mirror_pad_elastic_models`) before the adjoint kernel reads them at
    neighbour offsets near a reflecting face -- both play the identical
    "elastic model set" role on the same grid (see the module docstring).

    Runs the ``elastic_born_iso_*_backward_sc_*`` kernel via
    :func:`deepwave._engine.run_backward_core`: propagates a single scattered
    adjoint field with the plain elastic adjoint operator (background models
    fixed), seeded by the scattered receiver residuals ``grad_r_sc`` (and
    optional scattered-final-wavefield seeds ``sc_seeds``), and correlates it
    with the forward's background engine gradient stores to accumulate the
    scattering-model gradients ``grad_dlamb``/``grad_dmu``/``grad_dmu_ij``/
    ``grad_dbuoyancy_i`` (prepared-model space).

    ``args`` is the SAME positional layout the forward core
    (:func:`elastic_born_compiled_core`) consumes. This is a STANDALONE
    (no ``ctx``) test/diagnostic entry point -- kept for
    ``tests/test_elastic_born.py``'s direct kernel exercising -- and, like
    :func:`elastic_born_compiled_core`, is a throwaway-StorageManager bridge:
    it re-runs the forward once (``store_bg=True``) to regenerate the
    background stores, then builds a fresh ``StorageManager`` and copies them
    in, since (unlike :meth:`ElasticBornForwardFunc.backward`'s REAL
    production dispatch, which reads ``ctx.storage_manager`` directly with no
    re-run) this standalone function has no forward-produced storage manager
    to reuse. Returns ``(grad_sc_models, grad_f_sc, sc_adjoint_final)`` (all
    interior-cropped for the wavefields).
    """
    if mpml:
        raise DeepwaveNotImplementedError(
            "elastic_born: the compiled backward_sc does not support M-PML; "
            'use backend="eager".'
        )
    ndim = len(grid_spacing)
    n_wf = len(_elastic_module.elastic_wavefield_names(ndim))
    n_models = len(_elastic_module.elastic_model_names(ndim))
    n_src = ndim + 1
    n_bg_store = {1: 2, 2: 5, 3: 9}[ndim]
    args_list = [a.contiguous() for a in args]

    idx = 0

    def take(n: int) -> list[torch.Tensor]:
        nonlocal idx
        out = args_list[idx : idx + n]
        idx += n
        return out

    bg_models = take(n_models)
    sc_models = take(n_models)
    take(n_src)  # bg_source_amplitudes (unused by the adjoint operator)
    sc_source_amplitudes = take(n_src)
    sources_i = [s.contiguous() for s in take(n_src)]
    take(n_src)  # bg_receivers_i (unused: scattered-only path)
    sc_receivers_i = [r.contiguous() for r in take(n_src)]
    take(n_wf)  # bg_wavefields (unused)
    take(n_wf)  # sc_wavefields (initial states; adjoint seeds come from sc_seeds)

    device = bg_models[0].device
    dtype = bg_models[0].dtype
    is_cuda = bg_models[0].is_cuda
    model_shape = tuple(bg_models[0].shape[-ndim:])
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim

    # Regenerate the packed background stores by re-running the forward with
    # store_bg=True (this bridge's only re-run; see docstring), then build a
    # throwaway device-mode StorageManager and copy the per-slot (re-padded)
    # data in.
    fwd = elastic_born_compiled_core(
        pml_profiles,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        boundary,
        n_shots,
        mpml,
        *args,
        store_bg=True,
    )
    bg_store_interior = fwd[-1]

    storage_manager = _storage.StorageManager(
        model_shape,
        dtype,
        n_shots,
        nt,
        step_ratio,
        _storage.StorageMode.DEVICE,
        False,
        "",
        device,
        is_cuda,
    )
    for _ in range(n_bg_store):
        storage_manager.allocate(True)
    for i in range(n_bg_store):
        padded = torch.nn.functional.pad(bg_store_interior[i], fd_pad_list)
        storage_manager.storages[i].store_1a.copy_(padded)

    if sc_seeds is None:
        sc_seeds = [torch.empty(0, device=device, dtype=dtype) for _ in range(n_wf)]

    def zero_wavefields(
        wavefields: list[torch.Tensor],
        ndim: int,
        fd_pad: int,
        fd_pad_list: list[int],
        pml_width: list[int],
    ) -> None:
        _elastic_module.zero_edges_and_interiors(
            wavefields, ndim, fd_pad, fd_pad_list, pml_width, mpml=mpml
        )

    # This standalone bridge always wants a full gradient wherever a source
    # LOCATION exists, regardless of any tensor's own requires_grad (mirrors
    # the pre-M2 code's unconditional grad_f_sc allocation, gated only on the
    # location count).
    source_amplitudes_requires_grad = [True] * n_src
    # allocate_model_gradients only allocates a real (non-empty) gradient
    # buffer for a model with requires_grad=True; this standalone bridge
    # tests the compiled kernel directly (bypassing autograd), so the
    # scattering models it was handed need not actually be marked
    # requires_grad. Force it, exactly as the pre-M2 code did
    # (``[m.detach().requires_grad_(True) for m in sc_models]``) -- omitting
    # this leaves grad_dsc's data pointers NULL and the kernel writes through
    # them (segfault).
    sc_models_rg = [m.detach().requires_grad_(True) for m in sc_models]

    # The compiled adjoint reads both model groups at neighbour offsets near
    # a reflecting face, so each needs the mirror-pad guard ring, applied
    # independently (see the docstring above). A no-op when boundary is
    # all-PML.
    models = _boundaries.mirror_pad_elastic_models(
        bg_models, boundary, fd_pad
    ) + _boundaries.mirror_pad_elastic_models(sc_models_rg, boundary, fd_pad)

    state = _engine.run_backward_core(
        SPEC,
        models=models,
        sources_i=sources_i,
        receivers_i=sc_receivers_i,
        source_amplitudes=sc_source_amplitudes,
        pml_profiles=pml_profiles,
        grad_wavefields=sc_seeds,
        grad_receivers=grad_r_sc,
        grid_spacing=grid_spacing,
        dt=dt,
        nt=nt,
        n_shots=n_shots,
        step_ratio=step_ratio,
        accuracy=accuracy,
        pml_width=pml_width,
        source_amplitudes_requires_grad=source_amplitudes_requires_grad,
        storage_manager=storage_manager,
        backward_callback=None,
        callback_frequency=1,
        make_aux_wavefields=_elastic_module._elastic_make_aux_wavefields,  # noqa: SLF001
        zero_wavefields=zero_wavefields,
        make_call=_make_elastic_born_backward_sc_call(ndim),
        swap=_elastic_module._elastic_swap,  # noqa: SLF001
        wavefield_names=_elastic_module.elastic_wavefield_names(ndim),
        backend_extra="_sc",
        boundary=boundary,
    )

    grad_sc = list(state.grad_models[n_models:])
    grad_f_sc = list(state.grad_sources)
    out_s = state.output_slice
    sc_adjoint_final = [w[out_s] for w in state.wavefields]
    return grad_sc, grad_f_sc, sc_adjoint_final


def elastic_born_backward_compiled(
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    boundary: list[int],
    n_shots: int,
    mpml: bool,
    args: tuple[torch.Tensor, ...],
    grad_r_bg: list[torch.Tensor],
    grad_r_sc: list[torch.Tensor],
    bg_seeds: list[torch.Tensor],
    sc_seeds: list[torch.Tensor],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Compiled full two-field elastic Born adjoint (derivation SS4.2/SS4.3).

    ``boundary`` is the per-face ``BC_*`` integer code list; threaded to the
    forward re-run below and to the compiled adjoint's ``bc`` ABI parameter.
    The background and scattered model groups are each mirror-padded
    independently (:func:`deepwave._boundaries.mirror_pad_elastic_models`)
    before the adjoint kernel reads them at neighbour offsets near a
    reflecting face.

    Runs the ``elastic_born_iso_*_backward_*`` kernel via
    :func:`deepwave._engine.run_backward_core`: propagates a scattered adjoint
    field (plain elastic adjoint, as ``backward_sc``) AND a background adjoint
    field (the SS4.2 coupled adjoint), seeded respectively by the
    scattered/background receiver residuals (``grad_r_sc``/``grad_r_bg``) and
    final-wavefield gradients (``sc_seeds``/``bg_seeds``). Correlating the two
    fields with the forward's ``bg_store`` and ``sc_store`` yields the scattering
    -model gradients (scattered adjoint x ``bg_store``) and the background-model
    gradients (background adjoint x ``bg_store`` PLUS scattered adjoint x
    ``sc_store``; two terms each, SS4.3), plus background/scattered source and
    initial-wavefield gradients.

    ``args`` is the SAME positional layout the forward core
    (:func:`elastic_born_compiled_core`) consumes. This is a STANDALONE (no
    ``ctx``) test/diagnostic entry point -- kept for
    ``tests/test_elastic_born.py``'s direct kernel exercising, and because
    :mod:`deepwave.elastic`'s jvp/HVP rules call it BY NAME with this EXACT
    signature and return contract -- and, like
    :func:`_elastic_born_backward_sc_compiled`, is a throwaway-StorageManager
    bridge: it re-runs the forward once (``store_bg=store_sc=True``) to
    regenerate BOTH packed stores, then builds a fresh ``StorageManager`` (two
    engine-storage-slot groups, background then scattered) and copies them in,
    since (unlike :meth:`ElasticBornForwardFunc._backward_full_dispatch`'s REAL
    production dispatch, which reads ``ctx.storage_manager`` directly with no
    re-run) this standalone function has no forward-produced storage manager
    to reuse. Returns ``(grad_bg_models, grad_sc_models, grad_f_bg, grad_f_sc,
    bg_adjoint_final, sc_adjoint_final)`` (all interior-cropped for the
    wavefields).
    """
    if mpml:
        raise DeepwaveNotImplementedError(
            "elastic_born: the compiled backward does not support M-PML; "
            'use backend="eager".'
        )
    ndim = len(grid_spacing)
    n_wf = len(_elastic_module.elastic_wavefield_names(ndim))
    n_models = len(_elastic_module.elastic_model_names(ndim))
    n_src = ndim + 1
    n_bg_store = {1: 2, 2: 5, 3: 9}[ndim]
    args_list = [a.contiguous() for a in args]

    idx = 0

    def take(n: int) -> list[torch.Tensor]:
        nonlocal idx
        out = args_list[idx : idx + n]
        idx += n
        return out

    bg_models = take(n_models)
    sc_models = take(n_models)
    bg_source_amplitudes = take(n_src)
    sc_source_amplitudes = take(n_src)
    sources_i = [s.contiguous() for s in take(n_src)]
    bg_receivers_i = [r.contiguous() for r in take(n_src)]
    sc_receivers_i = [r.contiguous() for r in take(n_src)]
    take(n_wf)  # bg_wavefields (initial states; adjoint seeds come from bg_seeds)
    take(n_wf)  # sc_wavefields (initial states; adjoint seeds come from sc_seeds)

    device = bg_models[0].device
    dtype = bg_models[0].dtype
    is_cuda = bg_models[0].is_cuda
    model_shape = tuple(bg_models[0].shape[-ndim:])

    # Regenerate BOTH packed stores by re-running the forward with
    # store_bg=store_sc=True (this bridge's only re-run; see docstring), then
    # build a fresh device-mode StorageManager (background group then
    # scattered group, matching the kernel's parameter order) and copy the
    # per-slot (re-padded) data in.
    fwd = elastic_born_compiled_core(
        pml_profiles,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        boundary,
        n_shots,
        mpml,
        *args,
        store_bg=True,
        store_sc=True,
    )
    bg_store_interior = fwd[-2]
    sc_store_interior = fwd[-1]

    storage_manager = _storage.StorageManager(
        model_shape,
        dtype,
        n_shots,
        nt,
        step_ratio,
        _storage.StorageMode.DEVICE,
        False,
        "",
        device,
        is_cuda,
    )
    for _ in range(2 * n_bg_store):
        storage_manager.allocate(True)
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    for i in range(n_bg_store):
        padded = torch.nn.functional.pad(bg_store_interior[i], fd_pad_list)
        storage_manager.storages[i].store_1a.copy_(padded)
    for i in range(n_bg_store):
        padded = torch.nn.functional.pad(sc_store_interior[i], fd_pad_list)
        storage_manager.storages[n_bg_store + i].store_1a.copy_(padded)

    # Owner-based requires-grad flags (mu governs all mu_ij; buoyancy_x governs
    # all b_i -- consistent with the single kernel flag per family). The
    # buoyancy family is the trailing ``ndim`` models; the mu family is
    # between. allocate_model_gradients (inside run_backward_core) only
    # allocates a real (non-empty) gradient buffer for a model with
    # requires_grad=True, and the kernel's per-family bool params are read
    # straight off these forced tensors, so every model in a family must be
    # forced together -- omitting this leaves some family members' grad
    # pointers NULL while the kernel's shared flag says to write (segfault;
    # M2 lesson).
    mu_family = bg_models[1 : n_models - ndim]
    buoyancy_family = bg_models[n_models - ndim :]
    lamb_rg = bg_models[0].requires_grad
    mu_rg = any(m.requires_grad for m in mu_family)
    buoyancy_rg = any(m.requires_grad for m in buoyancy_family)
    dmu_family = sc_models[1 : n_models - ndim]
    dbuoyancy_family = sc_models[n_models - ndim :]
    dlamb_rg = sc_models[0].requires_grad
    dmu_rg = any(m.requires_grad for m in dmu_family)
    dbuoyancy_rg = any(m.requires_grad for m in dbuoyancy_family)
    bg_rg_flags = (
        [lamb_rg] + [mu_rg] * len(mu_family) + [buoyancy_rg] * len(buoyancy_family)
    )
    sc_rg_flags = (
        [dlamb_rg] + [dmu_rg] * len(dmu_family) + [dbuoyancy_rg] * len(dbuoyancy_family)
    )
    bg_models_rg = [
        m.detach().requires_grad_(rg) for m, rg in zip(bg_models, bg_rg_flags)
    ]
    sc_models_rg = [
        m.detach().requires_grad_(rg) for m, rg in zip(sc_models, sc_rg_flags)
    ]
    # The compiled adjoint reads both model groups at neighbour offsets near
    # a reflecting face, so each needs the mirror-pad guard ring, applied
    # independently (see the docstring above). A no-op when boundary is
    # all-PML.
    models = _boundaries.mirror_pad_elastic_models(
        bg_models_rg, boundary, fd_pad
    ) + _boundaries.mirror_pad_elastic_models(sc_models_rg, boundary, fd_pad)

    # This standalone bridge always wants a full gradient wherever a source
    # LOCATION exists, regardless of any tensor's own requires_grad (mirrors
    # _elastic_born_backward_sc_compiled's identical comment/pattern: the
    # kernel's record_from_wavefield/record_pressure calls are unconditional
    # on location count, not on any Python-side requires_grad reasoning).
    # sources_i is DUPLICATED to 2*n_src (see _make_elastic_born_backward_call's
    # docstring): run_backward_core derives n_sources_per_shot from
    # len(sources_i), which must match len(source_amplitudes_requires_grad).
    # source_amplitudes (the REAL bg/sc tensors) still gate make_call's
    # per-field n_sources_bg/n_sources_sc (structurally-absent source types,
    # e.g. the scattered pressure source, must record no gradient even though
    # the shared location block has an entry for it).
    source_amplitudes = [*bg_source_amplitudes, *sc_source_amplitudes]
    source_amplitudes_requires_grad = [True] * (2 * n_src)
    sources_i = [*sources_i, *sources_i]

    receivers_i = [*bg_receivers_i, *sc_receivers_i]
    grad_receivers = [g.contiguous() for g in (*grad_r_bg, *grad_r_sc)]
    grad_wavefields = [*bg_seeds, *sc_seeds]

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
        backward_callback=None,
        callback_frequency=1,
        make_aux_wavefields=_elastic_born_make_aux_wavefields,
        zero_wavefields=_elastic_born_zero_wavefields(mpml),
        make_call=_make_elastic_born_backward_call(ndim),
        swap=_elastic_born_swap,
        boundary=boundary,
    )

    grad_bg = list(state.grad_models[:n_models])
    grad_sc = list(state.grad_models[n_models:])
    grad_f_bg = list(state.grad_sources[:n_src])
    grad_f_sc = list(state.grad_sources[n_src:])
    out_s = state.output_slice
    n_wf_out = len(state.wavefields) // 2
    bg_adjoint_final = [w[out_s] for w in state.wavefields[:n_wf_out]]
    sc_adjoint_final = [w[out_s] for w in state.wavefields[n_wf_out:]]
    return grad_bg, grad_sc, grad_f_bg, grad_f_sc, bg_adjoint_final, sc_adjoint_final


def _elastic_born_store_flags(
    bg_models: list[torch.Tensor], sc_models: list[torch.Tensor]
) -> tuple[bool, bool]:
    """Whether the background/scattered engine storage slots are needed.

    ``need_bg``: any model (background OR scattering) requires grad -- the
    background stores feed every model gradient via ``backward_sc``'s imaging
    condition (derivation SS4.1). ``need_sc``: any BACKGROUND model requires
    grad -- the scattered stores feed only the two-term background-model
    gradients of the full two-field backward (SS2.5/SS4.3). Shared by
    ``ElasticBornForwardFunc.forward``'s ``storage_flags`` and
    ``make_call`` so the two never drift apart.
    """
    need_bg = any(m.requires_grad for m in bg_models) or any(
        m.requires_grad for m in sc_models
    )
    need_sc = any(m.requires_grad for m in bg_models)
    return need_bg, need_sc


def _elastic_born_zero_wavefields(mpml: bool) -> Any:
    """Builds the ``zero_wavefields`` closure for the engine forward pass.

    ``run_forward_engine`` hands this the FULL doubled (background then
    scattered) wavefield list; ``elastic.zero_edges_and_interiors`` hardcodes
    index tables sized to a single field's wavefield count, so it must be
    called once per half (mirrors :func:`elastic_born_compiled_core`'s
    ``pad_and_zero``, which does the same split).
    """

    def zero_wavefields(
        wavefields: list[torch.Tensor],
        ndim: int,
        fd_pad: int,
        fd_pad_list: list[int],
        pml_width: list[int],
    ) -> None:
        n_wf = len(wavefields) // 2
        _elastic_module.zero_edges_and_interiors(
            wavefields[:n_wf], ndim, fd_pad, fd_pad_list, pml_width, mpml=mpml
        )
        _elastic_module.zero_edges_and_interiors(
            wavefields[n_wf:], ndim, fd_pad, fd_pad_list, pml_width, mpml=mpml
        )

    return zero_wavefields


def _elastic_born_make_aux_wavefields(
    grad_wavefields: list[torch.Tensor], ndim: int
) -> list[torch.Tensor]:
    """Doubled ``m_sigma*`` aux ping-pong buffers for the full two-field backward.

    ``run_backward_core`` hands this the FULL doubled (background then
    scattered) ``grad_wavefields`` list; ``elastic._elastic_make_aux_wavefields``
    hardcodes index tables sized to a SINGLE field's wavefield count, so it must
    be called once per half (mirrors :func:`_elastic_born_zero_wavefields`,
    which does the same split). The background half's aux buffers come first,
    then the scattered half's -- matching ``elastic_born.c``'s
    ``FUNC(backward)`` parameter order exactly (verified independently for
    each half against the M2 aux-ordering lesson: every slot has the same
    pointer type, so a name-order mismatch is invisible to a type-only ABI
    check and only caught by a multi-chunk correctness test).
    """
    n_wf = len(grad_wavefields) // 2
    bg_aux = cast(
        "list[torch.Tensor]",
        _elastic_module._elastic_make_aux_wavefields(  # noqa: SLF001
            grad_wavefields[:n_wf], ndim
        ),
    )
    sc_aux = cast(
        "list[torch.Tensor]",
        _elastic_module._elastic_make_aux_wavefields(  # noqa: SLF001
            grad_wavefields[n_wf:], ndim
        ),
    )
    return bg_aux + sc_aux


def _elastic_born_swap(state: "deepwave._engine.BackendState") -> None:
    """Ping-pongs the ``m_sigma*`` memory variables for BOTH adjoint fields.

    Mirrors ``deepwave.elastic._elastic_swap``, applied independently to the
    background half (first ``n_wf`` wavefields / first ``n_aux`` aux buffers)
    and the scattered half (second ``n_wf`` / second ``n_aux``), since
    ``state.wavefields``/``state.aux_wavefields`` are the FULL doubled
    (background-then-scattered) lists ``run_backward_core`` builds for
    elastic_born's full two-field backward (see
    :func:`_elastic_born_make_aux_wavefields`). The per-``ndim`` index tables
    are ``_elastic_swap``'s own (unchanged), just applied at both the
    zero offset (background) and the ``n_wf``/``n_aux`` offset (scattered).
    """
    ndim = state.ndim
    gw = state.wavefields
    aw = state.aux_wavefields
    n_wf = len(gw) // 2
    n_aux = len(aw) // 2

    def swap_half(wf_off: int, aux_off: int) -> None:
        if ndim == 3:
            gw[wf_off + 9 : wf_off + 14], aw[aux_off + 0 : aux_off + 5] = (
                aw[aux_off + 0 : aux_off + 5],
                gw[wf_off + 9 : wf_off + 14],
            )
            gw[wf_off + 20 : wf_off + 23], aw[aux_off + 5 : aux_off + 8] = (
                aw[aux_off + 5 : aux_off + 8],
                gw[wf_off + 20 : wf_off + 23],
            )
            gw[wf_off + 26], aw[aux_off + 8] = aw[aux_off + 8], gw[wf_off + 26]
        elif ndim == 2:
            gw[wf_off + 6 : wf_off + 9], aw[aux_off + 0 : aux_off + 3] = (
                aw[aux_off + 0 : aux_off + 3],
                gw[wf_off + 6 : wf_off + 9],
            )
            gw[wf_off + 12], aw[aux_off + 3] = aw[aux_off + 3], gw[wf_off + 12]
        else:
            gw[wf_off + 3], aw[aux_off + 0] = aw[aux_off + 0], gw[wf_off + 3]

    swap_half(0, 0)
    swap_half(n_wf, n_aux)


def _make_elastic_born_forward_call(ndim: int) -> Any:
    """Builds the ``make_call`` closure for the coupled elastic Born forward.

    Mirrors ``elastic_born.c``'s ``FUNC(forward)`` parameter order exactly:
    background models, scattering models, background sources, scattered
    sources, background wavefields, scattered wavefields, background
    receivers, scattered receivers, background gradient stores (engine
    storage slots), scattered gradient stores, PML profiles, shared source
    locations, background receiver locations, scattered receiver locations,
    then the scalar/size trailers (rdx, dt, chunk nt, n_shots, model shape,
    source/receiver counts, step_ratio, storage_mode, shot_bytes_*, batched
    flags, store_bg/store_sc, storage_compression, start_t, pml_b/pml_e,
    aux/stream).

    ``state.sources_i`` carries a DUPLICATED shared source-location block (see
    :meth:`ElasticBornForwardFunc.forward`'s repacking of the flat args to
    conform to :meth:`PropagatorSpec.split_backend_args`'s generic
    ``n_source_types``-sized slicing); only the first half is forwarded here
    to the kernel's single shared-location parameter.
    """
    n_src = ndim + 1
    n_model = len(_elastic_module.elastic_model_names(ndim))
    n_wf = len(_elastic_module.elastic_wavefield_names(ndim))

    def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
        bg_models = state.models[:n_model]
        sc_models = state.models[n_model:]
        f_bg = state.source_amplitudes[:n_src]
        f_sc = state.source_amplitudes[n_src:]
        wf_bg = state.wavefields[:n_wf]
        wf_sc = state.wavefields[n_wf:]
        r_bg = state.receiver_amplitudes[:n_src]
        r_sc = state.receiver_amplitudes[n_src:]
        shared_src_i = state.sources_i[:n_src]
        rcv_i_bg = state.receivers_i[:n_src]
        rcv_i_sc = state.receivers_i[n_src:]
        lamb, mu = bg_models[0], bg_models[1]
        buoyancy_x = bg_models[-1]
        dlamb, dmu = sc_models[0], sc_models[1]
        dbuoyancy_x = sc_models[-1]

        def batched(m: torch.Tensor) -> bool:
            return m.ndim == ndim + 1 and m.shape[0] > 1

        need_bg, need_sc = _elastic_born_store_flags(bg_models, sc_models)
        # storage_mode="none" allocates NO backward-storage buffers (see
        # StorageManager: the per-slot tensors stay empty when the mode is
        # NONE), so the engine store pointers are not writable. The kernel
        # gates its snapshot writes on these store bools, so they MUST be forced
        # False in NONE mode or the -Ofast SAVE_SNAPSHOT block writes through an
        # empty (0-element) buffer and corrupts the heap. This matches the base
        # elastic/acoustic behaviour for storage_mode="none": the forward stores
        # nothing, so the backward has no forward history to correlate and the
        # model gradient is (correctly) not reconstructed -- source and
        # wavefield gradients are unaffected. (The doubled-field Born forward's
        # store bools were previously requires_grad-only, unguarded by mode; the
        # NONE path was never exercised before M5. The identical unguarded
        # pattern in acoustic_born.py's forward still segfaults on
        # storage_mode="none" with a differentiable model -- escalated.)
        if state.storage_manager.storage_mode == _storage.StorageMode.NONE:
            need_bg = need_sc = False
        n_sources_bg = state.n_sources_per_shot[:n_src]
        # ``state.sources_i`` is the DUPLICATED shared-location block (see
        # ``ElasticBornForwardFunc.forward``'s repacking), so
        # ``state.n_sources_per_shot[n_src:2*n_src]`` (derived from those
        # locations) is ALWAYS equal to ``n_sources_bg`` -- even for a source
        # type the scattered field has no amplitude for (e.g. no scattered
        # pressure source). The kernel must be told 0 scattered sources in
        # that case (an amplitude-gated count, not a location-gated one), or
        # it injects from an empty amplitude buffer -- exactly reproducing
        # elastic_born_compiled_core's pre-M1 ``n_sources_sc`` gate.
        n_sources_sc = [
            n_sources_bg[i] if f_sc[i].numel() > 0 else 0 for i in range(n_src)
        ]
        n_receivers_bg = state.n_receivers_per_shot[:n_src]
        n_receivers_sc = state.n_receivers_per_shot[n_src:]
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in bg_models],
                *[m.data_ptr() for m in sc_models],
                *[a.data_ptr() for a in f_bg],
                *[a.data_ptr() for a in f_sc],
                *[w.data_ptr() for w in wf_bg],
                *[w.data_ptr() for w in wf_sc],
                *[a.data_ptr() for a in r_bg],
                *[a.data_ptr() for a in r_sc],
                *state.storage_manager.storage_ptrs,
                *[p.data_ptr() for p in state.pml_profiles],
                *[loc.data_ptr() for loc in shared_src_i],
                *[loc.data_ptr() for loc in rcv_i_bg],
                *[loc.data_ptr() for loc in rcv_i_sc],
                *state.rdx,
                state.dt,
                step_nt,
                state.n_shots,
                *state.model_shape,
                *n_sources_bg,
                *n_sources_sc,
                *n_receivers_bg,
                *n_receivers_sc,
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                batched(lamb),
                batched(mu),
                batched(buoyancy_x),
                batched(dlamb),
                batched(dmu),
                batched(dbuoyancy_x),
                need_bg,
                need_sc,
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                state.aux,
                state.stream,
            ),
        )

    return make_call


def _make_elastic_born_backward_sc_call(ndim: int) -> Any:
    """Builds the ``make_call`` closure for the compiled scattered-only adjoint.

    Mirrors ``elastic_born.c``'s ``FUNC(backward_sc)`` parameter order exactly:
    background models (read-only), scattered data residuals, scattered
    adjoint wavefields (in/out), ``m_sigma*`` aux ping-pong buffers, the
    background engine gradient stores (read via SETUP_STORE_LOAD), scattering-
    model gradients (accumulated) + their thread buffers, scattered source-
    amplitude gradients, PML profiles, shared source locations, scattered
    receiver locations, then the scalar/size trailers.

    ``state.models`` is ``[bg_models, sc_models]`` (matching the ``models=``
    list ``ElasticBornForwardFunc._backward_sc_dispatch`` passes to
    :func:`deepwave._engine.run_backward_core`): only the background half is
    read by the kernel (the adjoint operator uses fixed background models);
    only the scattering half of ``state.grad_models`` is written (the
    background half is allocated but never touched -- the caller returns
    ``None`` for those positions, mirroring ``acoustic_born.py``'s scattered-
    only dispatch).
    """
    n_src = ndim + 1
    n_model = len(_elastic_module.elastic_model_names(ndim))
    n_bg_store = {1: 2, 2: 5, 3: 9}[ndim]

    def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
        bg_models = state.models[:n_model]
        sc_models = state.models[n_model:]
        grad_dsc = state.grad_models[n_model:]
        grad_dsc_tmp_ptr = state.grad_models_tmp_ptr[n_model:]

        def batched(m: torch.Tensor) -> bool:
            return m.ndim == ndim + 1 and m.shape[0] > 1

        # sources_i is the shared (bg==sc) location block; state.source_amplitudes
        # is the SCATTERED amplitudes only. n_sources_per_shot (location-derived)
        # must be gated by whether the scattered amplitude for that direction is
        # actually present -- mirrors the forward's n_sources_sc fix (M1): the
        # scattered pressure source is structurally absent (empty amplitude),
        # so its count must be 0 even though the shared location exists.
        n_sources = [
            state.n_sources_per_shot[i] if state.source_amplitudes[i].numel() > 0 else 0
            for i in range(n_src)
        ]
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in bg_models],
                *[g.data_ptr() for g in state.grad_receivers],
                *[w.data_ptr() for w in state.wavefields],
                *[a.data_ptr() for a in state.aux_wavefields],
                # ctx.storage_manager (in the production dispatch) is the
                # FORWARD's storage manager, which allocated BOTH the
                # background AND scattered store groups (2*N_BG_STORE slots);
                # backward_sc's kernel only reads the background group, so
                # only its first N_BG_STORE*5 pointers are passed (mirrors
                # acoustic_born.py's _make_backward_sc_call slicing).
                *state.storage_manager.storage_ptrs[: 5 * n_bg_store],
                *[g.data_ptr() for g in grad_dsc],
                *grad_dsc_tmp_ptr,
                *[g.data_ptr() for g in state.grad_sources],
                *[p.data_ptr() for p in state.pml_profiles],
                *[loc.data_ptr() for loc in state.sources_i],
                *[loc.data_ptr() for loc in state.receivers_i],
                *state.rdx,
                state.dt,
                step_nt,
                state.n_shots,
                *state.model_shape,
                *n_sources,
                *state.n_receivers_per_shot,
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                batched(bg_models[0]),
                batched(bg_models[1]),
                batched(bg_models[-1]),
                True,  # dlamb_requires_grad: backward_sc always wants every
                True,  # dmu_requires_grad: scattering-model gradient (all
                True,  # dbuoyancy_requires_grad: computed in one adjoint pass)
                batched(sc_models[0]),
                batched(sc_models[1]),
                batched(sc_models[-1]),
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                state.aux,
                state.stream,
            ),
        )

    return make_call


def _make_elastic_born_backward_call(ndim: int) -> Any:
    """Builds the ``make_call`` closure for the compiled full two-field adjoint.

    Mirrors ``elastic_born.c``'s ``FUNC(backward)`` parameter order exactly:
    background models, scattering models, background then scattered data
    residuals, background then scattered adjoint wavefields, background-field
    then scattered-field ``m_sigma*`` aux ping-pong buffers, the ``bg_store``
    engine-storage-slot group then the ``sc_store`` group (BOTH read via
    SETUP_STORE_LOAD -- unlike ``backward_sc``, which only reads the bg group,
    this kernel needs every pointer the forward's ``StorageManager``
    allocated, so ``storage_ptrs`` is passed WHOLE, unsliced), background then
    scattering model gradients (+ thread buffers), background then scattered
    source-amplitude gradients, PML profiles, the single SHARED source-
    location block (the C kernel has only one -- unlike ``receivers_i``, this
    is never doubled), background then scattered receiver locations, then the
    scalar/size trailers.

    ``state.models`` is ``[*bg_models, *sc_models]`` (each forced, by the
    caller, to the "owner family" requires_grad flag -- lamb/mu/buoyancy for
    the background half, dlamb/dmu/dbuoyancy for the scattering half --
    mirroring the pre-M3 ``elastic_born_backward_compiled``'s
    ``bg_rg_flags``/``sc_rg_flags``; NULL gradient pointers otherwise segfault,
    see the M2 addendum). ``state.grad_models`` is therefore
    ``[*grad_bg, *grad_sc]``; the boolean requires_grad C params are read
    straight off the (forced) ``state.models`` tensors rather than
    recomputed. ``state.sources_i`` is the shared block DUPLICATED to
    ``2 * n_src`` by the caller (mirrors the forward's M1 repacking trick):
    ``run_backward_core`` derives ``n_sources_per_shot`` from ``len(sources_i)``,
    which must match ``len(source_amplitudes_requires_grad)`` (``2 * n_src``,
    doubled bg/sc) or its per-index zip breaks, even though the C kernel has
    only ONE shared source-location parameter -- only the first half is ever
    forwarded to the backend call. ``state.receivers_i``/
    ``state.grad_receivers``/``state.n_receivers_per_shot``/
    ``state.source_amplitudes``/``state.grad_sources`` are all doubled
    (background half then scattered half).
    """
    n_src = ndim + 1
    n_model = len(_elastic_module.elastic_model_names(ndim))

    def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
        bg_models = state.models[:n_model]
        sc_models = state.models[n_model:]
        grad_bg = state.grad_models[:n_model]
        grad_sc = state.grad_models[n_model:]
        grad_bg_tmp_ptr = state.grad_models_tmp_ptr[:n_model]
        grad_sc_tmp_ptr = state.grad_models_tmp_ptr[n_model:]
        n_wf = len(state.wavefields) // 2

        def batched(m: torch.Tensor) -> bool:
            return m.ndim == ndim + 1 and m.shape[0] > 1

        # state.sources_i is the shared (bg==sc) location block DUPLICATED to
        # 2*n_src (see docstring); state.source_amplitudes/n_sources_per_shot
        # are doubled [bg, sc] in lockstep. n_sources_per_shot (location-
        # derived, structurally identical in both halves since duplicated)
        # must still be gated per-field by whether that field's amplitude is
        # actually present -- mirrors the forward's n_sources_sc fix (M1): the
        # scattered pressure source is structurally absent (empty amplitude),
        # so its count must be 0 even though the shared location exists (and
        # likewise for any absent background-side amplitude).
        n_sources_bg = [
            state.n_sources_per_shot[i] if state.source_amplitudes[i].numel() > 0 else 0
            for i in range(n_src)
        ]
        n_sources_sc = [
            state.n_sources_per_shot[n_src + i]
            if state.source_amplitudes[n_src + i].numel() > 0
            else 0
            for i in range(n_src)
        ]
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in bg_models],
                *[m.data_ptr() for m in sc_models],
                *[g.data_ptr() for g in state.grad_receivers[:n_src]],
                *[g.data_ptr() for g in state.grad_receivers[n_src:]],
                *[w.data_ptr() for w in state.wavefields[:n_wf]],
                *[w.data_ptr() for w in state.wavefields[n_wf:]],
                *[a.data_ptr() for a in state.aux_wavefields],
                # ctx.storage_manager (in the production dispatch) is the
                # FORWARD's storage manager, which allocated BOTH the
                # background AND scattered store groups (2*N_BG_STORE slots,
                # bg group then sc group); this kernel reads both, so every
                # pointer is forwarded (contrast backward_sc's bg-only slice).
                *state.storage_manager.storage_ptrs,
                *[g.data_ptr() for g in grad_bg],
                *[g.data_ptr() for g in grad_sc],
                *grad_bg_tmp_ptr,
                *grad_sc_tmp_ptr,
                *[g.data_ptr() for g in state.grad_sources[:n_src]],
                *[g.data_ptr() for g in state.grad_sources[n_src:]],
                *[p.data_ptr() for p in state.pml_profiles],
                *[loc.data_ptr() for loc in state.sources_i[:n_src]],
                *[loc.data_ptr() for loc in state.receivers_i[:n_src]],
                *[loc.data_ptr() for loc in state.receivers_i[n_src:]],
                *state.rdx,
                state.dt,
                step_nt,
                state.n_shots,
                *state.model_shape,
                *n_sources_bg,
                *n_sources_sc,
                *state.n_receivers_per_shot[:n_src],
                *state.n_receivers_per_shot[n_src:],
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                batched(bg_models[0]),
                batched(bg_models[1]),
                batched(bg_models[-1]),
                bg_models[0].requires_grad,
                bg_models[1].requires_grad,
                bg_models[-1].requires_grad,
                sc_models[0].requires_grad,
                sc_models[1].requires_grad,
                sc_models[-1].requires_grad,
                batched(sc_models[0]),
                batched(sc_models[1]),
                batched(sc_models[-1]),
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                state.aux,
                state.stream,
            ),
        )

    return make_call


def _use_compiled_adjoint(mpml: bool, is_cuda: bool) -> bool:
    """Whether the compiled Born adjoint kernels can service this backward.

    The compiled scattered-only and full adjoint kernels
    (:func:`_elastic_born_backward_sc_compiled`,
    :func:`elastic_born_backward_compiled`) exist on CPU and -- as of task N --
    on CUDA, both only without M-PML. Under M-PML the backward must instead
    route through the eager-recompute bridge (the validated
    :func:`elastic_born_eager_core` re-run under autograd). The CUDA adjoint
    kernels are compiled and ABI-locked to the CPU twins but are GPU-validated
    by the maintainer before release, so the compiled GPU path is only selected
    when a CUDA runtime is actually present (``torch.cuda.is_available()``);
    there is no other way to reach it on this CPU-only host. Factored out as a
    tiny pure function so the dispatch decision is unit-testable.
    """
    return not mpml and (not is_cuda or torch.cuda.is_available())


class ElasticBornForwardFunc(torch.autograd.Function):
    """Compiled forward and compiled CPU/CUDA adjoint without M-PML.

    ``forward`` runs the fast compiled :func:`elastic_born_compiled_core`.
    ``backward`` dispatches (see :func:`_use_compiled_adjoint`): without M-PML it
    uses the compiled scattered-only or full adjoint kernels on CPU and CUDA.
    The fallback eager-recompute branch is unreachable through the public
    interface: M-PML selects eager before this Function is called, while an
    explicit compiled request raises.
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
        forward_callback: Optional["deepwave.Callback"],
        backward_callback: Optional["deepwave.Callback"],
        callback_frequency: int,
        storage_mode_str: StorageMode,
        storage_path: str,
        storage_compression: bool,
        mpml: bool,
        boundary: list[int],
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Runs the compiled Born forward via the shared step-based engine.

        ``boundary`` is the per-face ``BC_*`` integer code list (see
        :func:`deepwave._common.set_boundary`); threaded through to
        :func:`deepwave._engine.run_forward_engine`, which appends it to
        ``state.bc`` and hence to the compiled ``FUNC(forward)`` call's
        trailing ``bc[2*ndim]`` argument (elastic_born.c/elastic_born.cu) --
        the same guard-ring fill the eager backend implements via
        :func:`elastic_born_eager_core`.

        Converted from a self-contained full-``nt``
        single call (:func:`elastic_born_compiled_core`, now kept only as the
        legacy bridge :meth:`backward`'s compiled dispatch and
        :mod:`deepwave.elastic`'s jvp/HVP rely on) to
        :func:`deepwave._engine.run_forward_engine`, exactly like
        ``acoustic_born.AcousticBornForwardFunc``. Gradient
        stores now live in the engine's ``StorageManager`` (allocated per
        :func:`_elastic_born_store_flags`) rather than a bespoke packed buffer
        filled by a backward re-run.

        The incoming ``args`` layout (identical to :func:`elastic_born_eager_core`)
        has a single SHARED ``sources_i`` block (``ndim+1`` locations, common to
        both fields), but ``PropagatorSpec.split_backend_args`` needs
        ``sources_i`` sized to match ``n_source_types`` (``2*(ndim+1)``, since
        source amplitudes -- and hence expected source-location count -- are
        doubled bg/sc). This is repacked here by duplicating the shared block
        into two identical halves; :func:`_make_elastic_born_forward_call`
        forwards only the first half to the kernel's single shared-location
        parameter.
        """
        ndim = len(grid_spacing)
        n_model = len(_elastic_module.elastic_model_names(ndim))
        n_src = ndim + 1
        src_i_end = 2 * n_model + 2 * n_src + n_src
        src_i_start = src_i_end - n_src
        repacked_args = (
            *args[:src_i_end],
            *args[src_i_start:src_i_end],
            *args[src_i_end:],
        )

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            bg_models = models[:n_model]
            sc_models = models[n_model:]
            need_bg, need_sc = _elastic_born_store_flags(bg_models, sc_models)
            n_bg_store = {1: 2, 2: 5, 3: 9}[ndim]
            return [need_bg] * n_bg_store + [need_sc] * n_bg_store

        # No explicit ``torch.no_grad()`` here: PyTorch's modern (ctx-free)
        # ``Function.apply()`` machinery already disables grad-tracking for
        # ``forward()``'s own body (mirrors ``AcousticBornForwardFunc.forward``,
        # which relies on the same automatic behaviour rather than wrapping
        # explicitly).
        outputs, saved_state = _engine.run_forward_engine(
            SPEC,
            pml_profiles=pml_profiles,
            grid_spacing=list(grid_spacing),
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
            args=repacked_args,
            storage_flags=storage_flags,
            zero_wavefields=_elastic_born_zero_wavefields(mpml),
            make_call=_make_elastic_born_forward_call(ndim),
            save_wavefields=True,
            boundary=boundary,
        )
        return (*outputs, saved_state)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Populates ``ctx`` from the trailing state produced by ``forward``."""
        _engine.apply_forward_saved_state(ctx, output[-1])
        # ``boundary`` is the last leading input, inserted right after
        # ``mpml``; recovered here so ``backward`` can re-thread the same
        # per-face codes into the compiled adjoint's ``bc`` ABI parameter.
        ctx.boundary = inputs[_N_LEAD - 1]
        # ``mpml`` is now the second-to-last leading input (see _N_LEAD); the
        # compiled Born adjoint kernels have no M-PML variants, so
        # ``backward`` routes M-PML through the eager Born driver (mirrors
        # ``elastic.py``'s ``jvp_mpml`` handling).
        ctx.mpml = bool(inputs[_N_LEAD - 2])

    # ``torch.func`` transforms now reach this Function (setup_context
    # migration), but the ctypes C/CUDA backend cannot process the batched
    # (``vmap``) or dual (``jvp``) tensors a transform would hand it, so raise
    # an actionable error pointing at ``backend="eager"`` (which elastic_born
    # provides) rather than let PyTorch emit its generic message. ``generate_vmap_rule``
    # is deliberately not enabled. Mirrors ``scalar_born``/``elastic``.
    vmap = staticmethod(_engine.unsupported_vmap)
    jvp = staticmethod(_engine.unsupported_jvp)

    @staticmethod
    @_engine.no_second_order("elastic_born")
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        """VJP for the compiled elastic Born forward.

        Dispatches (see :func:`_use_compiled_adjoint`). When the compiled adjoint
        kernels are available -- CPU or CUDA inputs and no M-PML -- it chooses
        between the compiled scattered-only adjoint
        (:func:`_elastic_born_backward_sc_compiled`, derivation SS4.1) and the
        compiled full adjoint (:func:`elastic_born_backward_compiled`,
        SS4.2/SS4.3). The scattered-only kernel is used whenever *no*
        background-side quantity needs a gradient -- no background model
        (``lamb``/``mu``/``buoyancy``), no background source amplitude, and no
        background initial wavefield -- because in that case the scattered
        output's only differentiable dependence is on the scattering models
        (``dlamb``/``dmu``/``dbuoyancy``), the scattered source amplitudes, and
        the scattered initial wavefields, all of which the fixed-background
        adjoint covers (mirrors ``acoustic_born.py``'s dispatch). Otherwise the
        compiled full two-field kernel is used.

        Under M-PML the compiled adjoints are not used at all, so the backward
        re-runs the fully validated eager driver under autograd (the eager
        bridge) instead. (:func:`_use_compiled_adjoint` also gates the CUDA case
        on ``torch.cuda.is_available()``, which is always true for genuine CUDA
        inputs; the extra check only matters for exercising the pure dispatch
        function in isolation, e.g. in unit tests on a CPU-only host.)
        """
        # ``ctx.saved_tensors`` is now populated by ``apply_forward_saved_state``
        # (via ``run_forward_engine``'s ``ForwardSavedState``), grouped by
        # tensor KIND (all models, then sources_i -- DUPLICATED, see
        # ``forward``'s repacking -- then all receivers_i, then all source
        # amplitudes, then all wavefields, then pml_profiles). The rest of this
        # method (and its two dispatch helpers, and the eager bridge below) all
        # consume :func:`elastic_born_eager_core`'s OLD flat layout instead,
        # grouped by FIELD (background block fully before scattered block, with
        # a single shared source-location block in between). Reconstruct that
        # old layout here once, up front, and leave everything downstream
        # unchanged.
        ndim = _engine.saved_state_ndim(ctx, "elastic_born")
        n_wf = len(_elastic_module.elastic_wavefield_names(ndim))
        n_models = len(_elastic_module.elastic_model_names(ndim))
        n_src = ndim + 1

        saved = list(ctx.saved_tensors)
        idx = 0
        models = saved[idx : idx + 2 * n_models]
        idx += 2 * n_models
        sources_i_dup = saved[idx : idx + 2 * n_src]
        idx += 2 * n_src
        receivers_i = saved[idx : idx + 2 * n_src]
        idx += 2 * n_src
        source_amplitudes = saved[idx : idx + 2 * n_src]
        idx += 2 * n_src
        wavefields = saved[idx : idx + 2 * n_wf]
        idx += 2 * n_wf
        pml_profiles = saved[idx:]

        args = [
            *models[:n_models],
            *models[n_models:],
            *source_amplitudes[:n_src],
            *source_amplitudes[n_src:],
            *sources_i_dup[:n_src],
            *receivers_i[:n_src],
            *receivers_i[n_src:],
            *wavefields[:n_wf],
            *wavefields[n_wf:],
        ]

        # Positional layout of ``args`` (see elastic_born_eager_core):
        #   bg_models, sc_models, bg_src_amps, sc_src_amps, sources_i,
        #   bg_receivers_i, sc_receivers_i, bg_wavefields, sc_wavefields.
        bg_model_sl = slice(0, n_models)
        sc_model_sl = slice(n_models, 2 * n_models)
        bg_src_sl = slice(2 * n_models, 2 * n_models + n_src)
        sc_src_sl = slice(2 * n_models + n_src, 2 * n_models + 2 * n_src)
        wf_start = 2 * n_models + 5 * n_src
        bg_wf_sl = slice(wf_start, wf_start + n_wf)
        sc_wf_sl = slice(wf_start + n_wf, wf_start + 2 * n_wf)

        need_bg_model = any(a.requires_grad for a in args[bg_model_sl])
        need_bg_src = any(a.requires_grad for a in args[bg_src_sl])
        need_bg_wf = any(a.requires_grad for a in args[bg_wf_sl])
        # Both CPU and CUDA have compiled adjoints without M-PML. Public calls
        # with M-PML select eager before reaching this Function; the legacy
        # recompute bridge below is retained for the internal dispatch contract.
        is_cuda = args[0].is_cuda
        if _use_compiled_adjoint(ctx.mpml, is_cuda):
            if not (need_bg_model or need_bg_src or need_bg_wf):
                return ElasticBornForwardFunc._backward_sc_dispatch(
                    ctx,
                    pml_profiles,
                    args,
                    grad_outputs,
                    ndim,
                    n_wf,
                    n_models,
                    n_src,
                    sc_model_sl,
                    sc_src_sl,
                    sc_wf_sl,
                )

            # Full two-field VJP via the compiled ``backward`` kernel (derivation
            # SS4.2/SS4.3).
            return ElasticBornForwardFunc._backward_full_dispatch(
                ctx,
                pml_profiles,
                args,
                grad_outputs,
                ndim,
                n_wf,
                n_models,
                n_src,
                bg_model_sl,
                sc_model_sl,
                bg_src_sl,
                sc_src_sl,
                bg_wf_sl,
                sc_wf_sl,
            )

        enable_grad = cast(
            "Callable[[], AbstractContextManager[None]]", torch.enable_grad
        )
        with enable_grad():
            leaves = [a.detach().requires_grad_(a.requires_grad) for a in args]
            outs = elastic_born_eager_core(
                pml_profiles,
                ctx.grid_spacing,
                ctx.dt,
                ctx.nt,
                ctx.step_ratio,
                ctx.accuracy,
                ctx.pml_width,
                ctx.boundary,
                ctx.n_shots,
                ctx.mpml,
                *leaves,
            )
            needed = [leaf for leaf in leaves if leaf.requires_grad]
            diff_outs = []
            diff_gos = []
            for out, go in zip(outs, grad_outputs):
                if out.requires_grad:
                    diff_outs.append(out)
                    diff_gos.append(go)
            grads: list[Optional[torch.Tensor]]
            if needed and diff_outs:
                grads = list(
                    torch.autograd.grad(
                        diff_outs,
                        needed,
                        grad_outputs=diff_gos,
                        allow_unused=True,
                    )
                )
            else:
                grads = [None] * len(needed)
        grad_iter = iter(grads)
        arg_grads: list[Optional[torch.Tensor]] = [
            next(grad_iter) if leaf.requires_grad else None for leaf in leaves
        ]
        lead: tuple[Optional[torch.Tensor], ...] = (None,) * _N_LEAD
        return (*lead, *arg_grads)

    @staticmethod
    def _backward_sc_dispatch(
        ctx: Any,
        pml_profiles: list[torch.Tensor],
        args: list[torch.Tensor],
        grad_outputs: tuple[torch.Tensor, ...],
        ndim: int,
        n_wf: int,
        n_models: int,
        n_src: int,
        sc_model_sl: slice,
        sc_src_sl: slice,
        sc_wf_sl: slice,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Scattered-only compiled VJP (fixed background, derivation SS4.1).

        Converted from a self-contained full-``nt``
        single call (:func:`_elastic_born_backward_sc_compiled`, which re-ran
        the forward to regenerate ``bg_store``) to
        :func:`deepwave._engine.run_backward_core`, reading the REAL
        ``ctx.storage_manager`` the forward's ``run_forward_engine`` call
        already populated -- no re-run. The ``m_sigma*`` memory-variable aux
        ping-pong (``elastic._elastic_make_aux_wavefields``/
        ``elastic._elastic_swap``, reused verbatim: the scattered
        field's wavefield layout is identical to plain elastic's) replaces the
        kernel's old per-call calloc/free.

        The forward output order is ``(*bg_final, *sc_final, *bg_amps,
        *sc_amps)``, so ``grad_outputs`` for the scattered final wavefields
        (``sc_seeds``) and scattered receivers (``grad_r_sc``) are sliced out.
        ``models=[bg_models, sc_models]`` (only the background half is read by
        the kernel; only the scattering half of the returned
        ``state.grad_models`` is ever written -- the background half is
        allocated but untouched, mirroring ``acoustic_born.py``'s scattered-
        only dispatch). Every background-side slot in the returned grad tuple
        is ``None`` (zero gradient on this path). ``ctx.set_materialize_grads``
        defaults to true, so each ``grad_output`` is a real (zero-filled)
        tensor rather than ``None``.
        """
        sc_seeds = list(grad_outputs[n_wf : 2 * n_wf])
        # The compiled backend reads these through a raw data_ptr() assuming a
        # C-contiguous [nt, n_shots, n_receivers] layout, but autograd may
        # deliver non-contiguous strided views here (e.g. checkpoint()'s
        # time-sliced/movedim'd receiver gradients); force contiguous first,
        # mirroring acoustic_born.py's AcousticBornForwardFunc.backward fix
        # and this module's own _make_elastic_born_backward_call bridge.
        grad_r_sc = [
            g.contiguous()
            for g in grad_outputs[2 * n_wf + n_src : 2 * n_wf + 2 * n_src]
        ]

        # The kernel's dlamb/dmu/dbuoyancy_requires_grad flags are hardcoded
        # True below (backward_sc always computes the full scattering-model
        # correlation in one pass), so allocate_model_gradients must always
        # provide real (non-empty) buffers for all three scattering-model
        # families, regardless of which specific ones this particular
        # backward call's args actually have requires_grad=True (a user
        # differentiating only dmu, say, would otherwise leave dlamb/
        # dbuoyancy's grad buffers NULL and the kernel writes through them --
        # segfault). Force detached copies for run_backward_core's models=
        # only; args[pos].requires_grad below (which gates whether a
        # particular caller-visible grad gets returned at all) still reads
        # the ORIGINAL (unforced) tensors.
        sc_models_rg = [m.detach().requires_grad_(True) for m in args[sc_model_sl]]
        # The compiled adjoint reads both model groups at neighbour offsets
        # near a reflecting face, so each needs the mirror-pad guard ring
        # (:func:`deepwave._boundaries.mirror_pad_elastic_models`), applied
        # independently to the background ([bg_models]) and scattering
        # ([sc_models]) groups -- both play the identical "elastic model
        # set" role on the same grid (see the module docstring). A no-op
        # when ``ctx.boundary`` is all-PML.
        fd_pad = ctx.accuracy // 2
        models = _boundaries.mirror_pad_elastic_models(
            list(args[:n_models]), ctx.boundary, fd_pad
        ) + _boundaries.mirror_pad_elastic_models(sc_models_rg, ctx.boundary, fd_pad)
        sc_source_amplitudes = list(args[sc_src_sl])
        # ``args`` groups by FIELD in elastic_born_eager_core's flat layout:
        # both model blocks, then both source-amplitude blocks, then a single
        # SHARED source-location block, then both receiver-location blocks,
        # then both wavefield blocks. Locate the shared source-location block.
        src_i_start = 2 * n_models + 2 * n_src
        sources_i = list(args[src_i_start : src_i_start + n_src])
        sc_receivers_i = list(args[src_i_start + 2 * n_src : src_i_start + 3 * n_src])
        # allocate_model_gradients's ``source_amplitudes_requires_grad``-gated
        # grad_sources allocation must NOT be keyed off each scattered
        # amplitude tensor's own .requires_grad: the kernel's
        # record_from_wavefield calls are unconditional on location COUNT
        # alone (gated only by whether the scattered amplitude for that
        # direction is structurally present -- see the make_call's ``n_sources``
        # and the M1 forward fix), with no requires_grad check in the C body.
        # A real scattered source can have a location but a caller who never
        # set requires_grad_(True) on its amplitude tensor (they just don't
        # want THAT particular gradient back); gating the BUFFER size on that
        # unrelated flag under-allocates it while the kernel still writes the
        # full count, corrupting the heap (segfault). This is the identical bug
        # M3 fixed in the full ``_backward_full_dispatch``; the returned-gradient
        # loop below still reads each amplitude's ORIGINAL .requires_grad, so a
        # caller only ever gets back the gradients it asked for. Gating on
        # location presence (as M3 does) harmlessly over-allocates the
        # structurally-empty scattered pressure slot, which the kernel then
        # never writes.
        n_sources_sc = [s.numel() // ctx.n_shots for s in sources_i]
        source_amplitudes_requires_grad_sc = [n > 0 for n in n_sources_sc]

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            _elastic_module.zero_edges_and_interiors(
                wavefields, ndim, fd_pad, fd_pad_list, pml_width, mpml=ctx.mpml
            )

        state = _engine.run_backward_core(
            SPEC,
            models=models,
            sources_i=sources_i,
            receivers_i=sc_receivers_i,
            source_amplitudes=sc_source_amplitudes,
            pml_profiles=pml_profiles,
            grad_wavefields=sc_seeds,
            grad_receivers=grad_r_sc,
            grid_spacing=list(ctx.grid_spacing),
            dt=ctx.dt,
            nt=ctx.nt,
            n_shots=ctx.n_shots,
            step_ratio=ctx.step_ratio,
            accuracy=ctx.accuracy,
            pml_width=ctx.pml_width,
            source_amplitudes_requires_grad=source_amplitudes_requires_grad_sc,
            storage_manager=ctx.storage_manager,
            backward_callback=ctx.backward_callback,
            callback_frequency=ctx.callback_frequency,
            make_aux_wavefields=_elastic_module._elastic_make_aux_wavefields,  # noqa: SLF001
            zero_wavefields=zero_wavefields,
            make_call=_make_elastic_born_backward_sc_call(ndim),
            swap=_elastic_module._elastic_swap,  # noqa: SLF001
            wavefield_names=_elastic_module.elastic_wavefield_names(ndim),
            backend_extra="_sc",
            boundary=ctx.boundary,
        )

        arg_grads: list[Optional[torch.Tensor]] = [None] * len(args)
        grad_sc = state.grad_models[n_models:]
        # Scattering models (dlamb/dmu/dbuoyancy in prepared space).
        for j, pos in enumerate(range(sc_model_sl.start, sc_model_sl.stop)):
            if args[pos].requires_grad:
                arg_grads[pos] = grad_sc[j]
        # Scattered source amplitudes.
        for j, pos in enumerate(range(sc_src_sl.start, sc_src_sl.stop)):
            if args[pos].requires_grad:
                arg_grads[pos] = state.grad_sources[j]
        # Scattered initial wavefields.
        out_s = state.output_slice
        for j, pos in enumerate(range(sc_wf_sl.start, sc_wf_sl.stop)):
            if args[pos].requires_grad:
                arg_grads[pos] = state.wavefields[j][out_s]

        lead: tuple[Optional[torch.Tensor], ...] = (None,) * _N_LEAD
        return (*lead, *arg_grads)

    @staticmethod
    def _backward_full_dispatch(
        ctx: Any,
        pml_profiles: list[torch.Tensor],
        args: list[torch.Tensor],
        grad_outputs: tuple[torch.Tensor, ...],
        ndim: int,
        n_wf: int,
        n_models: int,
        n_src: int,
        bg_model_sl: slice,
        sc_model_sl: slice,
        bg_src_sl: slice,
        sc_src_sl: slice,
        bg_wf_sl: slice,
        sc_wf_sl: slice,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Full two-field compiled VJP (background + scattered, SS4.2/SS4.3).

        Converted from a self-contained full-``nt``
        single call (:func:`elastic_born_backward_compiled`, which re-ran the
        forward TWICE to regenerate BOTH ``bg_store``/``sc_store``) to
        :func:`deepwave._engine.run_backward_core`, reading the REAL
        ``ctx.storage_manager`` the forward's ``run_forward_engine`` call
        already populated for BOTH groups -- no re-run. The doubled
        ``m_sigma*`` memory-variable aux ping-pong
        (:func:`_elastic_born_make_aux_wavefields`/:func:`_elastic_born_swap`,
        background half then scattered half, each independently ordered per
        ``deepwave.elastic._elastic_make_aux_wavefields``'s convention --
        3D-only block, then 2D+ block, then the always-present element LAST --
        checked separately for both halves per the M2 aux-ordering lesson)
        replaces the kernel's old per-call calloc/free.

        The forward output order is ``(*bg_final, *sc_final, *bg_amps,
        *sc_amps)``, so ``grad_outputs`` splits into the background/scattered
        final-wavefield seeds and the background/scattered receiver residuals.
        ``models=[*bg_models_rg, *sc_models_rg]``, each forced to the "owner
        family" requires_grad flag (lamb/mu/buoyancy for the background half,
        dlamb/dmu/dbuoyancy for the scattering half) mirroring the pre-M3
        ``elastic_born_backward_compiled``'s ``bg_rg_flags``/``sc_rg_flags`` --
        the kernel's per-family bool params are read straight off these forced
        tensors, and NULL gradient pointers otherwise segfault (M2 lesson).
        ``ctx.set_materialize_grads`` defaults true, so each ``grad_output``
        is a real (zero-filled) tensor.
        """
        bg_seeds = list(grad_outputs[0:n_wf])
        sc_seeds = list(grad_outputs[n_wf : 2 * n_wf])
        # The compiled backend reads these through a raw data_ptr() assuming a
        # C-contiguous [nt, n_shots, n_receivers] layout, but autograd may
        # deliver non-contiguous strided views here (e.g. checkpoint()'s
        # time-sliced/movedim'd receiver gradients); force contiguous first,
        # mirroring acoustic_born.py's AcousticBornForwardFunc.backward fix
        # and this module's own _make_elastic_born_backward_call bridge.
        grad_r_bg = [g.contiguous() for g in grad_outputs[2 * n_wf : 2 * n_wf + n_src]]
        grad_r_sc = [
            g.contiguous()
            for g in grad_outputs[2 * n_wf + n_src : 2 * n_wf + 2 * n_src]
        ]

        bg_models = list(args[bg_model_sl])
        sc_models = list(args[sc_model_sl])

        # Owner-based requires-grad flags (mu governs all mu_ij; buoyancy_x
        # governs all b_i -- consistent with the single kernel flag per
        # family). allocate_model_gradients (inside run_backward_core) only
        # allocates a real (non-empty) gradient buffer for a model with
        # requires_grad=True, and the kernel's per-family bool params are
        # read straight off these forced tensors, so every model in a family
        # must be forced together -- omitting this leaves some family
        # members' grad pointers NULL while the kernel's shared flag says to
        # write (segfault; M2 lesson).
        mu_family = bg_models[1 : n_models - ndim]
        buoyancy_family = bg_models[n_models - ndim :]
        lamb_rg = bg_models[0].requires_grad
        mu_rg = any(m.requires_grad for m in mu_family)
        buoyancy_rg = any(m.requires_grad for m in buoyancy_family)
        dmu_family = sc_models[1 : n_models - ndim]
        dbuoyancy_family = sc_models[n_models - ndim :]
        dlamb_rg = sc_models[0].requires_grad
        dmu_rg = any(m.requires_grad for m in dmu_family)
        dbuoyancy_rg = any(m.requires_grad for m in dbuoyancy_family)
        bg_rg_flags = (
            [lamb_rg] + [mu_rg] * len(mu_family) + [buoyancy_rg] * len(buoyancy_family)
        )
        sc_rg_flags = (
            [dlamb_rg]
            + [dmu_rg] * len(dmu_family)
            + [dbuoyancy_rg] * len(dbuoyancy_family)
        )
        bg_models_rg = [
            m.detach().requires_grad_(rg) for m, rg in zip(bg_models, bg_rg_flags)
        ]
        sc_models_rg = [
            m.detach().requires_grad_(rg) for m, rg in zip(sc_models, sc_rg_flags)
        ]
        # The compiled adjoint reads both model groups at neighbour offsets
        # near a reflecting face, so each needs the mirror-pad guard ring
        # (:func:`deepwave._boundaries.mirror_pad_elastic_models`), applied
        # independently. A no-op when ``ctx.boundary`` is all-PML.
        fd_pad = ctx.accuracy // 2
        models = _boundaries.mirror_pad_elastic_models(
            bg_models_rg, ctx.boundary, fd_pad
        ) + _boundaries.mirror_pad_elastic_models(sc_models_rg, ctx.boundary, fd_pad)

        bg_source_amplitudes = list(args[bg_src_sl])
        sc_source_amplitudes = list(args[sc_src_sl])
        source_amplitudes = [*bg_source_amplitudes, *sc_source_amplitudes]

        # ``args`` groups by FIELD in elastic_born_eager_core's flat layout:
        # both model blocks, then both source-amplitude blocks, then a single
        # SHARED source-location block, then both receiver-location blocks,
        # then both wavefield blocks. sources_i is DUPLICATED to 2*n_src (see
        # _make_elastic_born_backward_call's docstring): run_backward_core
        # derives n_sources_per_shot from len(sources_i), which must match
        # len(source_amplitudes_requires_grad); only the first half is ever
        # forwarded to the kernel's single shared source-location parameter.
        src_i_start = 2 * n_models + 2 * n_src
        shared_sources_i = list(args[src_i_start : src_i_start + n_src])
        sources_i = [*shared_sources_i, *shared_sources_i]
        bg_receivers_i = list(args[src_i_start + n_src : src_i_start + 2 * n_src])
        sc_receivers_i = list(args[src_i_start + 2 * n_src : src_i_start + 3 * n_src])
        receivers_i = [*bg_receivers_i, *sc_receivers_i]

        # allocate_model_gradients's ``source_amplitudes_requires_grad``-gated
        # grad_sources allocation must NOT be keyed off each amplitude
        # tensor's own .requires_grad: the kernel's record_from_wavefield/
        # record_pressure calls are unconditional on location COUNT alone
        # (``if (n_sources_z_per_shot > 0) { record_from_wavefield(...) }``,
        # no requires_grad check in the C body at all -- see backward_sc's
        # identical pattern). A real source can have a location but a caller
        # who never set requires_grad_(True) on its amplitude tensor (they
        # just don't want THAT particular gradient back); gating the BUFFER
        # size on that unrelated flag under-allocates it while the kernel
        # still writes the full count, corrupting the heap. sources_i is the
        # shared (bg==sc) location block, so the per-direction location
        # count is identical for both halves.
        n_sources_shared = [s.numel() // ctx.n_shots for s in shared_sources_i]
        loc_present = [n > 0 for n in n_sources_shared]
        source_amplitudes_requires_grad = loc_present + loc_present

        grad_wavefields = [*bg_seeds, *sc_seeds]
        grad_receivers = [*grad_r_bg, *grad_r_sc]

        state = _engine.run_backward_core(
            SPEC,
            models=models,
            sources_i=sources_i,
            receivers_i=receivers_i,
            source_amplitudes=source_amplitudes,
            pml_profiles=pml_profiles,
            grad_wavefields=grad_wavefields,
            grad_receivers=grad_receivers,
            grid_spacing=list(ctx.grid_spacing),
            dt=ctx.dt,
            nt=ctx.nt,
            n_shots=ctx.n_shots,
            step_ratio=ctx.step_ratio,
            accuracy=ctx.accuracy,
            pml_width=ctx.pml_width,
            source_amplitudes_requires_grad=source_amplitudes_requires_grad,
            storage_manager=ctx.storage_manager,
            backward_callback=ctx.backward_callback,
            callback_frequency=ctx.callback_frequency,
            make_aux_wavefields=_elastic_born_make_aux_wavefields,
            zero_wavefields=_elastic_born_zero_wavefields(ctx.mpml),
            make_call=_make_elastic_born_backward_call(ndim),
            swap=_elastic_born_swap,
            boundary=ctx.boundary,
        )

        arg_grads: list[Optional[torch.Tensor]] = [None] * len(args)

        def scatter(sl: slice, grads: list[torch.Tensor]) -> None:
            for j, pos in enumerate(range(sl.start, sl.stop)):
                if args[pos].requires_grad:
                    arg_grads[pos] = grads[j]

        scatter(bg_model_sl, list(state.grad_models[:n_models]))
        scatter(sc_model_sl, list(state.grad_models[n_models:]))
        scatter(bg_src_sl, list(state.grad_sources[:n_src]))
        scatter(sc_src_sl, list(state.grad_sources[n_src:]))
        out_s = state.output_slice
        n_wf_out = len(state.wavefields) // 2
        scatter(bg_wf_sl, [w[out_s] for w in state.wavefields[:n_wf_out]])
        scatter(sc_wf_sl, [w[out_s] for w in state.wavefields[n_wf_out:]])

        lead: tuple[Optional[torch.Tensor], ...] = (None,) * _N_LEAD
        return (*lead, *arg_grads)


# Count of leading non-tensor arguments of ``ElasticBornForwardFunc.forward``
# (the implicit ``ctx`` is not counted), all of which receive ``None`` gradients
# in ``backward``.
_N_LEAD = _engine.leading_arg_count(ElasticBornForwardFunc.forward)


def elastic_born_func(
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
    storage_mode: StorageMode,
    storage_path: str,
    storage_compression: bool,
    mpml: bool,
    boundary: list[int],
    *args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Applies :class:`ElasticBornForwardFunc` (compiled forward dispatch).

    ``boundary`` is the per-face ``BC_*`` integer code list; see
    :meth:`ElasticBornForwardFunc.forward`.
    """
    return cast(
        "tuple[torch.Tensor, ...]",
        ElasticBornForwardFunc.apply(  # type: ignore[no-untyped-call]
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_shots,
            forward_callback,
            backward_callback,
            callback_frequency,
            storage_mode,
            storage_path,
            storage_compression,
            mpml,
            boundary,
            *args,
        )[:-1],  # drop the trailing ForwardSavedState (mirrors acoustic_born_func)
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def _model_velocity_extremes(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    max_vel: Optional[float],
) -> tuple[float, float]:
    """Min-nonzero / max wavespeed from (lamb, mu, buoyancy) (see elastic())."""
    if not all(_common.values_readable(p) for p in (lamb, mu, buoyancy)):
        if max_vel is None:
            raise _common.max_vel_unreadable_error("elastic_born")
        return 0.0, max_vel
    vp, vs, _ = _common.lambmubuoyancy_to_vpvsrho(
        lamb.detach(), mu.detach(), buoyancy.detach()
    )
    max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
    vp_nz = vp[vp != 0]
    vs_nz = vs[vs != 0]
    min_vp = vp_nz.abs().min().item() if vp_nz.numel() > 0 else 0.0
    min_vs = vs_nz.abs().min().item() if vs_nz.numel() > 0 else 0.0
    candidates = [c for c in (min_vp, min_vs) if c > 0]
    min_nonzero = float(min(candidates)) if candidates else 0.0
    return min_nonzero, float(max_model_vel)


def elastic_born(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    dlamb: Optional[torch.Tensor] = None,
    dmu: Optional[torch.Tensor] = None,
    dbuoyancy: Optional[torch.Tensor] = None,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    bg_receivers: Optional[Mapping[str, ReceiverSpec]] = None,
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
    bg_receiver_locations_p: Optional[LocationsLike] = None,
    bg_receiver_locations_z: Optional[LocationsLike] = None,
    bg_receiver_locations_y: Optional[LocationsLike] = None,
    bg_receiver_locations_x: Optional[LocationsLike] = None,
    bg_receiver_mask_p: Optional[torch.Tensor] = None,
    bg_receiver_mask_z: Optional[torch.Tensor] = None,
    bg_receiver_mask_y: Optional[torch.Tensor] = None,
    bg_receiver_mask_x: Optional[torch.Tensor] = None,
    sc_receiver_locations_p: Optional[LocationsLike] = None,
    sc_receiver_locations_z: Optional[LocationsLike] = None,
    sc_receiver_locations_y: Optional[LocationsLike] = None,
    sc_receiver_locations_x: Optional[LocationsLike] = None,
    sc_receiver_mask_p: Optional[torch.Tensor] = None,
    sc_receiver_mask_z: Optional[torch.Tensor] = None,
    sc_receiver_mask_y: Optional[torch.Tensor] = None,
    sc_receiver_mask_x: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigmazz_0: Optional[torch.Tensor] = None,
    sigmayz_0: Optional[torch.Tensor] = None,
    sigmaxz_0: Optional[torch.Tensor] = None,
    sigmayy_0: Optional[torch.Tensor] = None,
    sigmaxy_0: Optional[torch.Tensor] = None,
    sigmaxx_0: Optional[torch.Tensor] = None,
    vz_sc_0: Optional[torch.Tensor] = None,
    vy_sc_0: Optional[torch.Tensor] = None,
    vx_sc_0: Optional[torch.Tensor] = None,
    sigmazz_sc_0: Optional[torch.Tensor] = None,
    sigmayz_sc_0: Optional[torch.Tensor] = None,
    sigmaxz_sc_0: Optional[torch.Tensor] = None,
    sigmayy_sc_0: Optional[torch.Tensor] = None,
    sigmaxy_sc_0: Optional[torch.Tensor] = None,
    sigmaxx_sc_0: Optional[torch.Tensor] = None,
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
    """Isotropic elastic Born (linearised) wave propagation.

    Stability: experimental (see :doc:`/stability`).

    The background field solves the ordinary (nonlinear) isotropic elastic
    system for ``(lamb, mu, buoyancy)``; the scattered field is the exact
    first-order linearisation for perturbations ``(dlamb, dmu, dbuoyancy)``,
    driven by virtual sources built from the background field's own strain-rate
    / stress-divergence terms times the scattering models (see
    ``docs/elastic_born_derivation.md``). A single set of source amplitudes per
    component drives both fields; the scattered field is *automatically* driven
    by the velocity/force "source-cell" term ``dt * dbuoyancy_i`` (the pressure
    source, pre-scaled by ``-dt`` with no model factor, carries no model
    perturbation and so contributes no scattered source; the derivation is in
    ``docs/elastic_born_derivation.md``, under the scattered-source section).

    **Current scope.** The default, ``backend="auto"``, runs the
    compiled C/CUDA forward, and uses the compiled adjoint kernels for
    backpropagation on both CPU and CUDA; under M-PML (``mpml_fraction > 0``) the
    compiled forward is not yet available and the call raises rather than
    falling back (see ``docs/elastic_born_derivation.md`` SS8-10). Passing
    ``backend="eager"`` selects the pure-PyTorch backend for both forward
    and backward; ``"torch_compile"`` runs the same eager time loop with
    the two coupled step functions compiled. The true Hessian-vector
    product and ``torch.func.jvp`` routing
    remain follow-up work. Both backends support float/Hicks source-receiver
    interpolation (``location_interpolation``), as in :func:`deepwave.elastic`.
    Gradients w.r.t. ``lamb, mu, buoyancy, dlamb,
    dmu, dbuoyancy``, source amplitudes, and initial wavefields are all
    supported through ordinary autograd.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        lamb: The background first Lame parameter model (see
            :func:`deepwave.elastic`), of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        mu: The background shear modulus model, of shape ``[*model_shape]``
            or, for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        buoyancy: The background buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. It is of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: Spatial grid cell size.
        dt: Time step interval. See :func:`deepwave.elastic` for the
            unit-consistency requirement across ``lamb``/``mu``/``buoyancy``/
            ``grid_spacing``/``dt``.
        dlamb: The ``lamb`` scattering perturbation model (``None`` ->
            zero), of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        dmu: The ``mu`` scattering perturbation model (``None`` -> zero), of
            shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        dbuoyancy: The ``buoyancy`` scattering perturbation model (``None``
            -> zero), of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        source_amplitudes_z: Shared z-force-density source amplitudes (both
            fields), with units of force/volume as in :func:`deepwave.elastic`.
            To drive the propagation with a volumetric (extended) source
            instead of hand-placed point sources, build the amplitudes and
            integer locations with
            :func:`deepwave.wavefields.source_field_to_sources`. This channel
            exists only in 3-D.
        source_amplitudes_y: Shared y-force-density source amplitudes.
        source_amplitudes_x: Shared x-force-density source amplitudes.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``pressure_rate``, ``z``, ``y``,
            ``x`` (``z`` only in 3-D, ``y`` only in 2-D and 3-D); these
            are equivalent to the flat
            ``source_amplitudes_*``/``source_locations_*`` keywords, which
            remain permanently supported; giving one channel in both forms
            raises. Optional.
        receivers: This simulation's receivers, keyed by channel, as the
            canonical alternative to the flat ``sc_receiver_locations*``
            keywords. Each value is a locations object, or ``(locations,
            mask)`` to include an activity mask. The channels are ``p``, ``z``,
            ``y``, ``x`` (``z`` only in 3-D, ``y`` only in 2-D and 3-D). Giving
            one channel in both forms raises. Optional.
        bg_receivers: The receivers that record the **background**
            wavefield, keyed by channel, as the canonical alternative to the
            flat ``bg_receiver_locations*`` keywords. Values take the same
            forms as ``receivers``'. Giving one channel in both forms
            raises. Optional.
        source_amplitudes_pressure_rate: Shared pressure-rate source amplitudes, with
            units of pressure/time as in :func:`deepwave.elastic` (not the
            volume-injection rate of the acoustic family; see :doc:`/units`).
        source_locations_z: z-force source locations (integer grid indices).
            This channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_y: y-force source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_x: x-force source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_pressure_rate: Pressure source locations.
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
        sc_receiver_locations_z: Scattered-field z-velocity receiver locations.
            This channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_locations_y: Scattered-field y-velocity receiver locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_locations_x: Scattered-field x-velocity receiver locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_locations_p: Scattered-field pressure receiver
            locations. In 2-D the recorded pressure is the in-plane mean
            normal stress ``-(sigma_xx + sigma_yy)/2``, which under plane
            strain is not the thermodynamic pressure; see
            :func:`deepwave.elastic`'s ``receiver_amplitudes_p`` entry for
            the full statement.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_mask_p: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``p`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``sc_receiver_locations_p`` are then
            ignored entirely (they need not be valid, in range, or unique). The
            shape must match the leading dimensions of
            ``sc_receiver_locations_p``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        sc_receiver_mask_z: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``z`` channel (3-D only)
            are active. ``True`` means active; ``False`` means that point
            records zeros, and its coordinates in ``sc_receiver_locations_z``
            are then ignored entirely (they need not be valid, in range, or
            unique). The shape must match the leading dimensions of
            ``sc_receiver_locations_z``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        sc_receiver_mask_y: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``y`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``sc_receiver_locations_y`` are then
            ignored entirely (they need not be valid, in range, or unique). The
            shape must match the leading dimensions of
            ``sc_receiver_locations_y``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        sc_receiver_mask_x: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``x`` channel are
            active. ``True`` means active; ``False`` means that point records
            zeros, and its coordinates in ``sc_receiver_locations_x`` are then
            ignored entirely (they need not be valid, in range, or unique). The
            shape must match the leading dimensions of
            ``sc_receiver_locations_x``, which must be given. This is the
            modern spelling of the :data:`deepwave.IGNORE_LOCATION` sentinel
            and produces bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        bg_receiver_locations_z: Optional background z-velocity receivers. This
            channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations_y: Optional background y-velocity receivers.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations_x: Optional background x-velocity receivers.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations_p: Optional background pressure receivers. In
            2-D the recorded pressure is the in-plane mean normal stress
            ``-(sigma_xx + sigma_yy)/2``, which under plane strain is not
            the thermodynamic pressure; see :func:`deepwave.elastic`'s
            ``receiver_amplitudes_p`` entry for the full statement.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_mask_p: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``p`` channel recording
            the background wavefield are active. ``True`` means active;
            ``False`` means that point records zeros, and its coordinates in
            ``bg_receiver_locations_p`` are then ignored entirely (they need
            not be valid, in range, or unique). The shape must match the
            leading dimensions of ``bg_receiver_locations_p``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        bg_receiver_mask_z: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``z`` channel (3-D only)
            recording the background wavefield are active. ``True`` means
            active; ``False`` means that point records zeros, and its
            coordinates in ``bg_receiver_locations_z`` are then ignored
            entirely (they need not be valid, in range, or unique). The shape
            must match the leading dimensions of ``bg_receiver_locations_z``,
            which must be given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        bg_receiver_mask_y: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``y`` channel recording
            the background wavefield are active. ``True`` means active;
            ``False`` means that point records zeros, and its coordinates in
            ``bg_receiver_locations_y`` are then ignored entirely (they need
            not be valid, in range, or unique). The shape must match the
            leading dimensions of ``bg_receiver_locations_y``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        bg_receiver_mask_x: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers of the ``x`` channel recording
            the background wavefield are active. ``True`` means active;
            ``False`` means that point records zeros, and its coordinates in
            ``bg_receiver_locations_x`` are then ignored entirely (they need
            not be valid, in range, or unique). The shape must match the
            leading dimensions of ``bg_receiver_locations_x``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        vz_0: Initial background vz wavefield, as in :func:`deepwave.elastic`.
        vy_0: Initial background vy wavefield.
        vx_0: Initial background vx wavefield.
        sigmazz_0: Initial background sigmazz wavefield.
        sigmayz_0: Initial background sigmayz wavefield.
        sigmaxz_0: Initial background sigmaxz wavefield.
        sigmayy_0: Initial background sigmayy wavefield.
        sigmaxy_0: Initial background sigmaxy wavefield.
        sigmaxx_0: Initial background sigmaxx wavefield.
        vz_sc_0: Initial scattered vz wavefield.
        vy_sc_0: Initial scattered vy wavefield.
        vx_sc_0: Initial scattered vx wavefield.
        sigmazz_sc_0: Initial scattered sigmazz wavefield.
        sigmayz_sc_0: Initial scattered sigmayz wavefield.
        sigmaxz_sc_0: Initial scattered sigmaxz wavefield.
        sigmayy_sc_0: Initial scattered sigmayy wavefield.
        sigmaxy_sc_0: Initial scattered sigmaxy wavefield.
        sigmaxx_sc_0: Initial scattered sigmaxx wavefield.
        accuracy: See :func:`deepwave.elastic`.
        pml_width: See :func:`deepwave.elastic`.
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
            :func:`deepwave.elastic`'s `boundary` (`"pml"`, `"free"`,
            `"rigid"`, or `"periodic"`; a non-`"pml"` face requires
            `pml_width == 0` on that face). Applied independently to the
            background field `(v, sigma)` and the scattered field `(dv,
            dsigma)`: both obey the identical isotropic elastic staggered-
            grid system on the same grid (see the module docstring), so the
            same per-field image parity (the flat-face parity table
            documented on `deepwave.elastic`) applies to each
            independently, with no new physics beyond what
            :func:`deepwave.elastic` already validates. Validated on the
            eager (`backend="eager"`) and the default compiled
            C/CUDA backends; the `"torch_compile"` Python mode shares the
            eager guard-fill code path but is not separately validated for
            `boundary=`.
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
        mpml_fraction: Multiaxial-PML fraction, as in :func:`deepwave.elastic`.
        max_vel: See :func:`deepwave.elastic`.
        courant_number: See :func:`deepwave.elastic`.
        survey_pad: See :func:`deepwave.scalar`.
        initial_state: See :func:`deepwave.elastic`. The mapping keys are this
            propagator's own public state field names -- the background names of
            :func:`deepwave.elastic` (e.g. `vz_0`) and the scattered names,
            which carry `sc` before the time index (e.g. `vz_sc_0`) -- plus
            `auxiliary_state`. The individual absorbing-boundary field names
            (`m_vzz_0`, ... and their scattered counterparts) are not accepted
            as keys: pass `initial_state=out` (or
            `initial_state=out.final_state`) to carry them across.
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
        forward_callback: See :func:`deepwave.elastic`. Supported on both the
            default compiled backend and the eager backend (``backend=
            "eager"``).
        backward_callback: See :func:`deepwave.elastic`. Compiled backend only.
        callback_frequency: See :func:`deepwave.elastic`.
        backend: Which execution backend to use: ``"auto"`` (the default),
            ``"compiled"``, ``"eager"`` or ``"torch_compile"``. See
            :doc:`/backends` for what each value selects, what ``"auto"``
            resolves to, and how the backends agree numerically.
            ``"auto"`` uses the compiled C/CUDA forward, paired with the
            compiled adjoint kernels, on CPU and on CUDA alike;
            ``out.info.backward_backend`` reports which backend the backward
            pass would use, and currently always matches
            ``out.info.backend``. The compiled backend has no M-PML forward
            kernel, so a call with ``mpml_fraction > 0`` is one whose options only
            the eager backend implements: ``"auto"`` resolves to ``"eager"``
            for it (and ``out.info.backend`` says so), which is much slower,
            while ``"compiled"`` raises
            :exc:`deepwave.DeepwaveNotImplementedError` because that spelling
            asserts the compiled kernel ran.
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
            Compiled backend only. The forward allocates these
            store slots whenever a model requires grad, so a
            large allocation happens in the forward; `"none"`
            behaves as for :func:`deepwave.acoustic_born`.
        storage_path: The directory that `storage_mode="disk"` writes to. A
            string or any `os.PathLike` (such as a `pathlib.Path`). Defaults
            to the system temporary directory, resolved **when you call**
            rather than when Deepwave is imported, so `TMPDIR` set at any
            point before the call is honoured. Using the default with
            `storage_mode="disk"` emits a one-time warning naming the
            directory, because on many systems it is a RAM-backed `tmpfs`
            far smaller than the disk. A permanent alias for `Storage.path`;
            not deprecated.
            Compiled backend only.
        storage_compression: **Lossy -- changes the model gradient by
            about 0.2-4% in 2-D and 0.2-8% in 3-D.** See
            :func:`deepwave.elastic`.
            Compiled backend only.
        location_interpolation: See :func:`deepwave.elastic`. A single source
            interpolator is built from the shared source locations and its spread
            applied once, then shared by the background and the (internally
            derived) scattered source -- both drive the same physical locations
            (see the module docstring's source-cell paragraph). The background
            and scattered receivers have independent interpolators.
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
        deepwave.PropagatorOutput: a tuple subclass whose fields, in order,
        are:

            - vz_0: Final background z-velocity wavefield (if ndim == 3).
            - vy_0: Final background y-velocity wavefield (if ndim >= 2).
            - vx_0: Final background x-velocity wavefield.
            - sigmazz_0: Final background zz-stress wavefield (if ndim == 3).
            - sigmayz_0: Final background yz-stress wavefield (if ndim == 3).
            - sigmaxz_0: Final background xz-stress wavefield (if ndim == 3).
            - sigmayy_0: Final background yy-stress wavefield (if ndim >= 2).
            - sigmaxy_0: Final background xy-stress wavefield (if ndim >= 2).
            - sigmaxx_0: Final background xx-stress wavefield.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables, of both the background
              and the scattered field. Opaque: pass it back (as part of
              `initial_state`) to continue the simulation; its contents are
              implementation state rather than physics.
            - vz_sc_0: Final scattered z-velocity wavefield (if ndim == 3).
            - vy_sc_0: Final scattered y-velocity wavefield (if ndim >= 2).
            - vx_sc_0: Final scattered x-velocity wavefield.
            - sigmazz_sc_0: Final scattered zz-stress wavefield (if ndim == 3).
            - sigmayz_sc_0: Final scattered yz-stress wavefield (if ndim == 3).
            - sigmaxz_sc_0: Final scattered xz-stress wavefield (if ndim == 3).
            - sigmayy_sc_0: Final scattered yy-stress wavefield (if ndim >= 2).
            - sigmaxy_sc_0: Final scattered xy-stress wavefield (if ndim >= 2).
            - sigmaxx_sc_0: Final scattered xx-stress wavefield.
            - bg_receiver_amplitudes_p: Recorded background pressure data.
              In 2-D the recorded pressure is the in-plane mean normal
              stress ``-(sigma_xx + sigma_yy)/2``, which under plane strain
              is not the thermodynamic pressure; see
              :func:`deepwave.elastic`'s ``receiver_amplitudes_p`` entry for
              the full statement.
            - bg_receiver_amplitudes_z: Recorded background z-velocity data
              (if ndim == 3).
            - bg_receiver_amplitudes_y: Recorded background y-velocity data
              (if ndim >= 2).
            - bg_receiver_amplitudes_x: Recorded background x-velocity data.
            - sc_receiver_amplitudes_p: Recorded scattered pressure data.
              In 2-D the recorded pressure is the in-plane mean normal
              stress ``-(sigma_xx + sigma_yy)/2``, which under plane strain
              is not the thermodynamic pressure; see
              :func:`deepwave.elastic`'s ``receiver_amplitudes_p`` entry for
              the full statement.
            - sc_receiver_amplitudes_z: Recorded scattered z-velocity data
              (if ndim == 3).
            - sc_receiver_amplitudes_y: Recorded scattered y-velocity data
              (if ndim >= 2).
            - sc_receiver_amplitudes_x: Recorded scattered x-velocity data.

        This positional order is easy to index incorrectly (``out[-1]`` is the
        scattered x-component velocity receiver amplitudes, not the pressure
        data); prefer accessing the receiver fields by name instead.
    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            elastic_born,
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
            "sc_receiver_locations_p": sc_receiver_locations_p,
            "sc_receiver_mask_p": sc_receiver_mask_p,
            "sc_receiver_locations_z": sc_receiver_locations_z,
            "sc_receiver_mask_z": sc_receiver_mask_z,
            "sc_receiver_locations_y": sc_receiver_locations_y,
            "sc_receiver_mask_y": sc_receiver_mask_y,
            "sc_receiver_locations_x": sc_receiver_locations_x,
            "sc_receiver_mask_x": sc_receiver_mask_x,
            "bg_receiver_locations_p": bg_receiver_locations_p,
            "bg_receiver_mask_p": bg_receiver_mask_p,
            "bg_receiver_locations_z": bg_receiver_locations_z,
            "bg_receiver_mask_z": bg_receiver_mask_z,
            "bg_receiver_locations_y": bg_receiver_locations_y,
            "bg_receiver_mask_y": bg_receiver_mask_y,
            "bg_receiver_locations_x": bg_receiver_locations_x,
            "bg_receiver_mask_x": bg_receiver_mask_x,
        },
    )
    _engine.expand_channel_mappings(
        SPEC,
        _channels,
        sources=sources,
        receivers=receivers,
        bg_receivers=bg_receivers,
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
    sc_receiver_locations_p = cast(
        "Optional[LocationsLike]", _channels["sc_receiver_locations_p"]
    )
    sc_receiver_locations_z = cast(
        "Optional[LocationsLike]", _channels["sc_receiver_locations_z"]
    )
    sc_receiver_locations_y = cast(
        "Optional[LocationsLike]", _channels["sc_receiver_locations_y"]
    )
    sc_receiver_locations_x = cast(
        "Optional[LocationsLike]", _channels["sc_receiver_locations_x"]
    )
    bg_receiver_locations_p = cast(
        "Optional[LocationsLike]", _channels["bg_receiver_locations_p"]
    )
    bg_receiver_locations_z = cast(
        "Optional[LocationsLike]", _channels["bg_receiver_locations_z"]
    )
    bg_receiver_locations_y = cast(
        "Optional[LocationsLike]", _channels["bg_receiver_locations_y"]
    )
    bg_receiver_locations_x = cast(
        "Optional[LocationsLike]", _channels["bg_receiver_locations_x"]
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
        [lamb, mu, buoyancy], ["lamb", "mu", "buoyancy"], "elastic_born"
    )
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend
    if dlamb is None and dmu is None and dbuoyancy is None:
        raise DeepwaveValueError(
            "elastic_born: at least one scattering model (dlamb, dmu, "
            "dbuoyancy) must be provided."
        )
    if dlamb is None:
        dlamb = torch.zeros_like(lamb)
    if dmu is None:
        dmu = torch.zeros_like(mu)
    if dbuoyancy is None:
        dbuoyancy = torch.zeros_like(buoyancy)
    _common.check_models(
        [dlamb, dmu, dbuoyancy], ["dlamb", "dmu", "dbuoyancy"], "elastic_born"
    )

    _loc_tensors = [
        source_locations_z,
        source_locations_y,
        source_locations_x,
        source_locations_pressure_rate,
        sc_receiver_locations_z,
        sc_receiver_locations_y,
        sc_receiver_locations_x,
        sc_receiver_locations_p,
        bg_receiver_locations_z,
        bg_receiver_locations_y,
        bg_receiver_locations_x,
        bg_receiver_locations_p,
    ]
    # initial_state may be a whole PropagatorOutput (a NamedTuple, so it lacks
    # a mapping's ``.get``/``.values``/``.items``) or a plain mapping. Normalise
    # a PropagatorOutput to a mapping up front so it can be indexed and
    # round-tripped; the full type/key validation happens in
    # resolve_initial_state below.
    _recv_names = _engine.receiver_output_names(SPEC)
    if isinstance(initial_state, deepwave.PropagatorOutput):
        _state_map: Optional[
            Mapping[str, Union[torch.Tensor, _engine.AuxiliaryState]]
        ] = initial_state._asdict()
    else:
        _state_map = initial_state
    # Only the wavefield entries help infer ndim; a full PropagatorOutput's
    # receiver-amplitude entries have a different rank and must be excluded, as
    # must its ``auxiliary_state`` entry (an AuxiliaryState, not a Tensor).
    _wf_tensors = (
        [
            t
            for k, t in _state_map.items()
            if k not in _recv_names and isinstance(t, torch.Tensor)
        ]
        if isinstance(_state_map, collections.abc.Mapping)
        else []
    )
    # The initial wavefields passed as explicit arguments are as much a part
    # of the survey as those inside the bundle, so they fix ndim too. They are
    # grouped by the dimension each belongs to, as on the other propagators,
    # so that a field of a dimension this call does not have is reported
    # rather than quietly ignored.
    ndim = _common.get_ndim(
        [lamb, mu, buoyancy],
        _wf_tensors,
        _loc_tensors,
        [
            vz_0,
            sigmazz_0,
            sigmayz_0,
            sigmaxz_0,
            vz_sc_0,
            sigmazz_sc_0,
            sigmayz_sc_0,
            sigmaxz_sc_0,
        ],
        [vy_0, sigmayy_0, sigmaxy_0, vy_sc_0, sigmayy_sc_0, sigmaxy_sc_0],
        [vx_0, sigmaxx_0, vx_sc_0, sigmaxx_sc_0],
    )
    if ndim < 3 and (
        source_locations_z is not None
        or sc_receiver_locations_z is not None
        or bg_receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            f"elastic_born: the propagation was determined to be {ndim}d, so "
            "z-dimension locations should not be provided."
        )
    if ndim < 2 and (
        source_locations_y is not None
        or sc_receiver_locations_y is not None
        or bg_receiver_locations_y is not None
    ):
        raise DeepwaveValueError(
            f"elastic_born: the propagation was determined to be {ndim}d, so "
            "y-dimension locations should not be provided."
        )

    if mpml_fraction > 0 and ndim < 2:
        raise DeepwaveValueError(
            "elastic_born: mpml_fraction > 0 (multiaxial PML) requires at least two "
            "spatial dimensions."
        )

    # Assemble per-component source/receiver lists in [velocities..., pressure]
    # order (matching elastic and the eager core's pressure-last convention).
    source_amplitudes: list[Optional[torch.Tensor]] = []
    source_locations_in: list[Optional[LocationsLike]] = []
    sc_receiver_locations_in: list[Optional[LocationsLike]] = []
    bg_receiver_locations_in: list[Optional[LocationsLike]] = []

    wf_names = cast("list[str]", _elastic_module.elastic_wavefield_names(ndim))

    if ndim >= 3:
        source_amplitudes.append(source_amplitudes_z)
        source_locations_in.append(source_locations_z)
        sc_receiver_locations_in.append(sc_receiver_locations_z)
        bg_receiver_locations_in.append(bg_receiver_locations_z)
    if ndim >= 2:
        source_amplitudes.append(source_amplitudes_y)
        source_locations_in.append(source_locations_y)
        sc_receiver_locations_in.append(sc_receiver_locations_y)
        bg_receiver_locations_in.append(bg_receiver_locations_y)
    source_amplitudes.append(source_amplitudes_x)
    source_locations_in.append(source_locations_x)
    sc_receiver_locations_in.append(sc_receiver_locations_x)
    bg_receiver_locations_in.append(bg_receiver_locations_x)
    source_amplitudes.append(source_amplitudes_pressure_rate)
    source_locations_in.append(source_locations_pressure_rate)
    sc_receiver_locations_in.append(sc_receiver_locations_p)
    bg_receiver_locations_in.append(bg_receiver_locations_p)

    # Background then scattered initial wavefields. resolve_initial_state fills
    # each named slot from initial_state (the original PropagatorOutput or
    # mapping, not the normalised _state_map above, so a PropagatorOutput's
    # RunInfo is still visible for origin recovery), ignoring any receiver-
    # amplitude fields of a whole PropagatorOutput and validating the keys.
    # Only the physical fields (v*_0, sigma*_0, and their _sc counterparts;
    # see A4.3) are exposed as explicit keyword arguments -- the internal m_*
    # CPML memory variables are always seeded as None here (still
    # round-trippable via initial_state=out).
    _physical_wf: dict[str, Optional[torch.Tensor]] = {
        "vz_0": vz_0,
        "vy_0": vy_0,
        "vx_0": vx_0,
        "sigmazz_0": sigmazz_0,
        "sigmayz_0": sigmayz_0,
        "sigmaxz_0": sigmaxz_0,
        "sigmayy_0": sigmayy_0,
        "sigmaxy_0": sigmaxy_0,
        "sigmaxx_0": sigmaxx_0,
        "vz_sc_0": vz_sc_0,
        "vy_sc_0": vy_sc_0,
        "vx_sc_0": vx_sc_0,
        "sigmazz_sc_0": sigmazz_sc_0,
        "sigmayz_sc_0": sigmayz_sc_0,
        "sigmaxz_sc_0": sigmaxz_sc_0,
        "sigmayy_sc_0": sigmayy_sc_0,
        "sigmaxy_sc_0": sigmaxy_sc_0,
        "sigmaxx_sc_0": sigmaxx_sc_0,
    }
    # Keyed by the public state field names (the scattered fields carry `sc`
    # before the time index) but ordered by the backend's own flat wavefield
    # order, which is what the unpacking below and the cores require. The
    # public return order additionally permutes the fields (grouping them by
    # kind); that permutation is applied to the outputs, not here.
    _wf_args: dict[str, Optional[torch.Tensor]] = {}
    for name in elastic_born_wavefield_names(ndim):
        _wf_args[name] = _physical_wf.get(name)
    supplied_auxiliary_state = _engine.auxiliary_state_of(initial_state)
    # A continuation has nothing of its own to resolve these from -- it
    # typically has no sources, and the first call's `max_vel` need not be the
    # model's own maximum -- so an omitted one takes the value the state's
    # absorbing-boundary memory variables were built with.
    pml_freq = _engine.inherited_pml_freq(pml_freq, supplied_auxiliary_state)
    max_vel = _engine.inherited_max_vel(max_vel, supplied_auxiliary_state)
    _resolved_wavefields, origin = _engine.resolve_initial_state(
        initial_state,
        _wf_args,
        origin=origin,
        survey_pad=survey_pad,
        ignore=_recv_names,
        auxiliary_state_names=elastic_born_auxiliary_state_names(3),
        propagator=SPEC.name,
    )
    initial_wavefields = list(_resolved_wavefields.values())

    _common.check_inputs_not_vmapped(
        lamb,
        mu,
        buoyancy,
        dlamb,
        dmu,
        dbuoyancy,
        *source_amplitudes,
        *source_locations_in,
        *sc_receiver_locations_in,
        *bg_receiver_locations_in,
    )

    # Resolve off-grid / directional (Hicks) source and receiver
    # specifications. Each velocity component is staggered half a cell in its
    # own dimension; pressure (built from the normal stresses) sits on the
    # regular grid (offset None); the lists are ordered [velocities...,
    # pressure] (see deepwave.elastic). A single source interpolator is built
    # from source_locations_in and its spread applied once, then shared by the
    # background and the (internally derived) scattered source -- they drive
    # identical physical locations (see the module docstring's source-cell
    # paragraph). The background and scattered receivers have independent
    # interpolators.
    component_offsets: list[Optional[list[float]]] = [
        *_common.velocity_component_offsets(ndim),
        None,
    ]
    # Recorded before the location arguments are replaced by the integer grid
    # indices the backend consumes.
    location_interpolation_used = _engine.resolved_location_interpolation(
        location_interpolation,
        [
            source_locations_in,
            bg_receiver_locations_in,
            sc_receiver_locations_in,
        ],
        [component_offsets, component_offsets, component_offsets],
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
        [*source_locations_in, *bg_receiver_locations_in, *sc_receiver_locations_in],
        lamb,
        "elastic_born",
    )
    source_amplitudes, source_locations = _common.resolve_source_interpolators(
        source_locations_in,
        source_amplitudes,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        lamb.dtype,
        _hicks_grid,
    )
    (
        bg_receiver_interpolators,
        bg_receiver_locations,
    ) = _common.resolve_receiver_interpolators(
        bg_receiver_locations_in,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        lamb.dtype,
        _hicks_grid,
    )
    (
        sc_receiver_interpolators,
        sc_receiver_locations,
    ) = _common.resolve_receiver_interpolators(
        sc_receiver_locations_in,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        lamb.dtype,
        _hicks_grid,
    )

    # A velocity source/receiver must not sit on the final sample of its
    # staggered dimension (that cell is not part of the computational
    # domain). The check runs on the resolved integer grid locations; the
    # lists are ordered [velocities..., pressure] (mirrors deepwave.elastic).
    dim_names = ["z", "y", "x"]
    for dim in range(ndim):
        for locations, name in [
            (source_locations, "source"),
            (bg_receiver_locations, "bg_receiver"),
            (sc_receiver_locations, "sc_receiver"),
        ]:
            location = locations[dim]
            if location is None:
                continue
            dim_location = location[..., -ndim + dim]
            dim_location = dim_location[dim_location != deepwave.IGNORE_LOCATION]
            if (
                dim_location.numel() > 0
                and dim_location.max() >= lamb.shape[-ndim + dim] - 1
            ):
                raise DeepwaveRuntimeError(
                    f"elastic_born: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{lamb.shape[-ndim + dim] - 1}.",
                )

    min_nonzero_model_vel, max_model_vel = _model_velocity_extremes(
        lamb, mu, buoyancy, max_vel
    )
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim
    n_src = ndim + 1

    all_source_amplitudes = source_amplitudes + source_amplitudes
    all_source_locations = source_locations + source_locations
    all_receiver_locations = bg_receiver_locations + sc_receiver_locations

    user_dt = dt
    requested_mgsi = model_gradient_sampling_interval
    (
        models,
        source_amplitudes_out,
        wavefields,
        sources_i,
        receivers_i,
        grid_spacing_out,
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
        [lamb, mu, buoyancy, dlamb, dmu, dbuoyancy],
        ["replicate"] * 6,
        ["lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy"],
        grid_spacing,
        dt,
        all_source_amplitudes,
        all_source_locations,
        all_receiver_locations,
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
    lamb_p, mu_p, buoyancy_p, dlamb_p, dmu_p, dbuoyancy_p = models
    del lamb, mu, buoyancy, dlamb, dmu, dbuoyancy, models

    bg_models = [lamb_p, mu_p, *_prepare_parameters(mu_p, buoyancy_p)]
    sc_models = [
        dlamb_p,
        dmu_p,
        *prepare_scatter_parameters(mu_p, buoyancy_p, dmu_p, dbuoyancy_p),
    ]
    model_shape = bg_models[0].shape[-ndim:]

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "elastic_born")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "elastic_born")
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
            bg_models, sc_models, source_amplitudes_out, wavefields
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

    # Scale sources: background velocity by dt*buoyancy_i, the scattered
    # "source-cell" virtual source by dt*dbuoyancy_i; pressure background by
    # -dt (no model factor) and therefore NO scattered pressure source.
    empty = torch.empty(0, device=device, dtype=dtype)
    for i in range(ndim):
        base = source_amplitudes_out[i]
        if base.numel() > 0:
            b_bg = bg_models[-ndim + i]
            b_sc = sc_models[-ndim + i]
            gathered = _common.gather_model_at_sources(
                b_bg, sources_i[i], model_shape, n_shots
            )
            dgathered = _common.gather_model_at_sources(
                b_sc, sources_i[i], model_shape, n_shots
            )
            source_amplitudes_out[i + n_src] = base * dgathered * dt
            source_amplitudes_out[i] = base * gathered * dt
        else:
            source_amplitudes_out[i + n_src] = base
    p = ndim  # pressure index within each (ndim+1) block
    if source_amplitudes_out[p].numel() > 0:
        source_amplitudes_out[p + n_src] = empty
        source_amplitudes_out[p] = -source_amplitudes_out[p] * dt
    else:
        source_amplitudes_out[p + n_src] = source_amplitudes_out[p]

    pml_profiles = _staggered_grid.set_pml_profiles(
        pml_width,
        accuracy,
        fd_pad,
        dt,
        grid_spacing_out,
        max_vel,
        dtype,
        device,
        pml_freq,
        model_shape,
        mpml_fraction=mpml_fraction,
    )

    n_wf = len(wf_names)
    mpml = mpml_fraction > 0
    if mpml:
        # The one option gap "auto" has to resolve around: elastic_born is
        # the only M-PML propagator with no compiled M-PML forward, so a
        # call that asks for M-PML asks for something only the eager backend
        # implements, and "auto" -- "whichever backend implements what I
        # asked for" -- selects it. Resolved before RunInfo is built, so
        # out.info.backend reports the backend that actually ran.
        # backend="compiled" raises here instead, because that spelling is
        # an assertion that the compiled kernel ran.
        backend_mode = _engine.resolve_compiled_option_gap(
            backend_request,
            "elastic_born",
            gap="the compiled M-PML (mpml_fraction > 0) forward is not yet available",
            remedy=(
                'Pass backend="eager" for M-PML (which is what '
                'backend="auto" selects here), or mpml_fraction=0.0 for the '
                "compiled backend."
            ),
            dtype=dtype,
            ndim=ndim,
            device=device,
            extra=(f"mpml_fraction={mpml_fraction}",),
        )
    _common.check_callback_frequency(callback_frequency)
    callback_frequency = _engine.CallbackSchedule(callback_frequency, step_ratio)

    # The one argument list drives either core (compiled or eager); it matches
    # elastic_born_eager_core's positional layout exactly.
    core_args = (
        *bg_models,
        *sc_models,
        *source_amplitudes_out[:n_src],
        *source_amplitudes_out[n_src:],
        *sources_i[:n_src],
        *receivers_i[:n_src],
        *receivers_i[n_src:],
        *wavefields[:n_wf],
        *wavefields[n_wf:],
    )

    run_info = RunInfo(
        step_ratio=step_ratio,
        model_gradient_sampling_interval=requested_mgsi,
        model_gradient_stride=_engine.resolved_model_gradient_stride(
            bg_models + sc_models,
            backend_mode,
            step_ratio,
            model_gradient_sampling_interval,
        ),
        dt=user_dt,
        grid_spacing=tuple(float(spacing) for spacing in grid_spacing_out),
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
        # The public M-PML path selects eager before the compiled Function is
        # called; report the backend used for its backward pass as well.
        backward_backend=(
            _engine.resolved_backend_name(backend_mode)
            if backend_mode is not None
            else (
                "compiled"
                if _use_compiled_adjoint(mpml_fraction > 0, device.type == "cuda")
                else "eager"
            )
        ),
        max_vel=max_vel,
        ndim=ndim,
        dtype=dtype,
        device=device,
        model_batched=tuple(_engine.model_batched(bg_models + sc_models, ndim)),
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
        bg_models + sc_models,
        storage_mode,
        python_backend=backend_mode,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
    )

    if not backend_mode:
        # Default: the compiled (fast) C/CUDA forward with compiled adjoint
        # kernels; the public M-PML path selects eager before this branch.
        # storage_mode="none" keeps no forward wavefield history, so the model
        # gradient (an imaging correlation of the forward and adjoint fields)
        # cannot be formed. Forward-only "none" is fully supported (a valid
        # low-memory mode); combining it with a differentiable input is
        # rejected with a clear error rather than silently returning a wrong
        # (zero) model gradient or -- as the analogous unguarded acoustic_born
        # path currently does -- segfaulting when the backward reads the
        # unallocated store (escalated). Use "device"/"cpu"/"disk" for
        # gradients w.r.t. the models/sources/wavefields.
        if storage_mode == "none" and any(
            isinstance(t, torch.Tensor) and t.requires_grad for t in core_args
        ):
            raise DeepwaveRuntimeError(
                'elastic_born: storage_mode="none" retains no forward '
                "wavefield history, so it cannot compute gradients; it is a "
                "forward-only (no-grad) low-memory mode. A model, source "
                "amplitude, or initial wavefield requires grad -- use "
                'storage_mode of "device", "cpu", or "disk" for the backward '
                "pass."
            )
        outputs = list(
            elastic_born_func(
                pml_profiles,
                grid_spacing_out,
                dt,
                nt,
                step_ratio * model_gradient_sampling_interval,
                accuracy,
                pml_width,
                n_shots,
                forward_callback,
                backward_callback,
                callback_frequency,
                storage_mode,
                storage_path,
                storage_compression,
                mpml,
                boundary_codes,
                *core_args,
            )
        )
    else:
        global _update_velocities_born_opt, _update_stresses_born_opt

        # See _engine.float_grid_spacing: the traced step functions must
        # receive Python floats, not ints. setup_propagator already
        # guarantees this, so this is defence in depth.
        grid_spacing_out = _engine.float_grid_spacing(grid_spacing_out)
        config_key = (len(grid_spacing_out), accuracy, dtype)
        _update_velocities_born_opt = _update_velocities_born_cache.resolve(
            backend_mode, config_key
        )
        _update_stresses_born_opt = _update_stresses_born_cache.resolve(
            backend_mode, config_key
        )
        # The pure-PyTorch eager/compiled backends run native autograd, not
        # the shared step-based engine, so they have no StorageManager and no
        # chunked backward-callback machinery (a backward_callback needs the
        # adjoint pass the engine drives, which these backends do not have).
        # forward_callback has no such dependency -- elastic_born_eager_core
        # runs its own per-step loop -- so it is threaded through instead of
        # rejected (mirrors deepwave.elastic's own eager backend). Unlike the
        # other propagators, elastic_born_eager_core is reached directly from
        # here rather than through an elastic_born_func wrapper, so the
        # shared rejection runs at this call site instead of inside the
        # pure-Python function itself.
        _engine.reject_unsupported_python_backend_options(
            SPEC,
            backend_mode,
            storage_mode=storage_mode,
            storage_compression=storage_compression,
            has_backward_callback=backward_callback is not None,
            is_batched=False,  # SPEC never allows a backward_callback here.
        )
        outputs = list(
            elastic_born_eager_core(
                pml_profiles,
                grid_spacing_out,
                dt,
                nt,
                step_ratio,
                accuracy,
                pml_width,
                boundary_codes,
                n_shots,
                mpml,
                *core_args,
                forward_callback=forward_callback,
                callback_frequency=callback_frequency,
            )
        )

    # Downsample the 2*(ndim+1) receiver outputs (last block of outputs).
    n_recv = 2 * n_src
    for i in range(len(outputs) - n_recv, len(outputs)):
        outputs[i] = _common.downsample_and_movedim(
            outputs[i],
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
        )

    # Gather off-grid receiver data back to the original locations, while the
    # receiver outputs are still in [velocities..., pressure] order (matching
    # bg_receiver_interpolators/sc_receiver_interpolators), i.e. before
    # finalize_receivers's pressure-mean/pressure-first reordering below. The
    # gather is linear, so it commutes with the normal-stress mean applied to
    # form the pressure trace.
    _common.apply_receiver_interpolators(outputs, bg_receiver_interpolators, 2 * n_wf)
    _common.apply_receiver_interpolators(
        outputs, sc_receiver_interpolators, 2 * n_wf + n_src
    )

    # Split outputs.
    bg_wf = outputs[:n_wf]
    sc_wf = outputs[n_wf : 2 * n_wf]
    bg_recv = outputs[2 * n_wf : 2 * n_wf + n_src]  # [v_z, v_y, v_x, p_sum]
    sc_recv = outputs[2 * n_wf + n_src : 2 * n_wf + 2 * n_src]

    order = _public_wavefield_order(ndim)
    bg_wf = [bg_wf[i] for i in order]
    sc_wf = [sc_wf[i] for i in order]

    def finalize_receivers(recv: list[torch.Tensor]) -> list[torch.Tensor]:
        # Pressure (last) -> -sum/ndim, then move to front (pressure-first).
        pressure = recv[-1]
        if pressure.numel() > 0:
            pressure = -pressure / ndim
        return [pressure, *recv[:-1]]

    bg_recv = finalize_receivers(bg_recv)
    sc_recv = finalize_receivers(sc_recv)

    final = tuple(bg_wf) + tuple(sc_wf) + tuple(bg_recv) + tuple(sc_recv)

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


class ElasticBorn(torch.nn.Module):
    """A Module wrapper around :func:`elastic_born`.

    Stability: experimental (see :doc:`/stability`).

    Stores ``lamb, mu, buoyancy, grid_spacing`` (and, if desired,
    ``dlamb/dmu/dbuoyancy``), matching :class:`elastic.Elastic` and
    :class:`acoustic_born.AcousticBorn`. As in the functional
    :func:`elastic_born`, at least one of ``dlamb``/``dmu``/``dbuoyancy`` must
    be provided (the others default to ``None``, treated as zero
    perturbations); the module defers this validation to :func:`elastic_born`
    itself, which is called on every :meth:`forward`.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        lamb: The background first Lamé parameter model, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        mu: The background second Lamé parameter model, mu, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        buoyancy: The background buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell; see
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. It is of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            (for isotropic grids) or a sequence of numbers (for anisotropic
            grids).
        dlamb: The perturbation to the first Lamé parameter, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. At least one of
            `dlamb`/`dmu`/`dbuoyancy` must be provided.
        dmu: The perturbation to the second Lamé parameter, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. At least one of
            `dlamb`/`dmu`/`dbuoyancy` must be provided.
        dbuoyancy: The perturbation to the buoyancy, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. At least one of
            `dlamb`/`dmu`/`dbuoyancy` must be provided.
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
        dlamb_requires_grad: A bool specifying whether gradients will be
            computed for `dlamb`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        dmu_requires_grad: A bool specifying whether gradients will be
            computed for `dmu`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        dbuoyancy_requires_grad: A bool specifying whether gradients will be
            computed for `dbuoyancy`. Unset by default, which means False. Cannot
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
            Compiled backend only. The forward allocates these
            store slots whenever a model requires grad, so a
            large allocation happens in the forward; `"none"`
            behaves as for :func:`deepwave.acoustic_born`.
        storage_path: The directory that `storage_mode="disk"` writes to. A
            string or any `os.PathLike` (such as a `pathlib.Path`). Defaults
            to the system temporary directory, resolved **when you call**
            rather than when Deepwave is imported, so `TMPDIR` set at any
            point before the call is honoured. Using the default with
            `storage_mode="disk"` emits a one-time warning naming the
            directory, because on many systems it is a RAM-backed `tmpfs`
            far smaller than the disk. A permanent alias for `Storage.path`;
            not deprecated.
            Compiled backend only.
        storage_compression: **Lossy -- changes the model gradient by
            about 0.2-4% in 2-D and 0.2-8% in 3-D.** See
            :func:`deepwave.elastic`.
            Compiled backend only.
    """

    def __init__(
        self,
        lamb: torch.Tensor,
        mu: torch.Tensor,
        buoyancy: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        dlamb: Optional[torch.Tensor] = None,
        dmu: Optional[torch.Tensor] = None,
        dbuoyancy: Optional[torch.Tensor] = None,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        lamb_requires_grad: Optional[bool] = None,
        mu_requires_grad: Optional[bool] = None,
        buoyancy_requires_grad: Optional[bool] = None,
        dlamb_requires_grad: Optional[bool] = None,
        dmu_requires_grad: Optional[bool] = None,
        dbuoyancy_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        owned = ["lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy"]
        optional = {"dlamb": dlamb, "dmu": dmu, "dbuoyancy": dbuoyancy}
        supplied = [
            name for name in owned if name not in optional or optional[name] is not None
        ]
        grad = _common.resolve_requires_grad(
            requires_grad,
            owned,
            supplied,
            {
                "lamb": lamb_requires_grad,
                "mu": mu_requires_grad,
                "buoyancy": buoyancy_requires_grad,
                "dlamb": dlamb_requires_grad,
                "dmu": dmu_requires_grad,
                "dbuoyancy": dbuoyancy_requires_grad,
            },
            "ElasticBorn",
        )
        self.lamb = _common.make_model_parameter(lamb, grad["lamb"], "lamb")
        self.mu = _common.make_model_parameter(mu, grad["mu"], "mu")
        self.buoyancy = _common.make_model_parameter(
            buoyancy, grad["buoyancy"], "buoyancy"
        )
        self.dlamb: Optional[torch.Tensor] = (
            None
            if dlamb is None
            else _common.make_model_parameter(dlamb, grad["dlamb"], "dlamb")
        )
        self.dmu: Optional[torch.Tensor] = (
            None
            if dmu is None
            else _common.make_model_parameter(dmu, grad["dmu"], "dmu")
        )
        self.dbuoyancy: Optional[torch.Tensor] = (
            None
            if dbuoyancy is None
            else _common.make_model_parameter(dbuoyancy, grad["dbuoyancy"], "dbuoyancy")
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
        bg_receivers: Optional[Mapping[str, ReceiverSpec]] = None,
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
        bg_receiver_locations_p: Optional[LocationsLike] = None,
        bg_receiver_locations_z: Optional[LocationsLike] = None,
        bg_receiver_locations_y: Optional[LocationsLike] = None,
        bg_receiver_locations_x: Optional[LocationsLike] = None,
        bg_receiver_mask_p: Optional[torch.Tensor] = None,
        bg_receiver_mask_z: Optional[torch.Tensor] = None,
        bg_receiver_mask_y: Optional[torch.Tensor] = None,
        bg_receiver_mask_x: Optional[torch.Tensor] = None,
        sc_receiver_locations_p: Optional[LocationsLike] = None,
        sc_receiver_locations_z: Optional[LocationsLike] = None,
        sc_receiver_locations_y: Optional[LocationsLike] = None,
        sc_receiver_locations_x: Optional[LocationsLike] = None,
        sc_receiver_mask_p: Optional[torch.Tensor] = None,
        sc_receiver_mask_z: Optional[torch.Tensor] = None,
        sc_receiver_mask_y: Optional[torch.Tensor] = None,
        sc_receiver_mask_x: Optional[torch.Tensor] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        sigmazz_0: Optional[torch.Tensor] = None,
        sigmayz_0: Optional[torch.Tensor] = None,
        sigmaxz_0: Optional[torch.Tensor] = None,
        sigmayy_0: Optional[torch.Tensor] = None,
        sigmaxy_0: Optional[torch.Tensor] = None,
        sigmaxx_0: Optional[torch.Tensor] = None,
        vz_sc_0: Optional[torch.Tensor] = None,
        vy_sc_0: Optional[torch.Tensor] = None,
        vx_sc_0: Optional[torch.Tensor] = None,
        sigmazz_sc_0: Optional[torch.Tensor] = None,
        sigmayz_sc_0: Optional[torch.Tensor] = None,
        sigmaxz_sc_0: Optional[torch.Tensor] = None,
        sigmayy_sc_0: Optional[torch.Tensor] = None,
        sigmaxy_sc_0: Optional[torch.Tensor] = None,
        sigmaxx_sc_0: Optional[torch.Tensor] = None,
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

        See :func:`elastic_born` for the meaning of every argument (the
        constructor-owned ``lamb``/``mu``/``buoyancy``/``dlamb``/``dmu``/
        ``dbuoyancy``/``grid_spacing`` and the storage settings are supplied
        from the stored module state).
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
        return elastic_born(
            self.lamb,
            self.mu,
            self.buoyancy,
            self.grid_spacing,
            dt,
            dlamb=self.dlamb,
            dmu=self.dmu,
            dbuoyancy=self.dbuoyancy,
            sources=sources,
            receivers=receivers,
            bg_receivers=bg_receivers,
            source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_mask_pressure_rate=source_mask_pressure_rate,
            source_mask_z=source_mask_z,
            source_mask_y=source_mask_y,
            source_mask_x=source_mask_x,
            bg_receiver_mask_p=bg_receiver_mask_p,
            bg_receiver_mask_z=bg_receiver_mask_z,
            bg_receiver_mask_y=bg_receiver_mask_y,
            bg_receiver_mask_x=bg_receiver_mask_x,
            sc_receiver_mask_p=sc_receiver_mask_p,
            sc_receiver_mask_z=sc_receiver_mask_z,
            sc_receiver_mask_y=sc_receiver_mask_y,
            sc_receiver_mask_x=sc_receiver_mask_x,
            source_locations_pressure_rate=source_locations_pressure_rate,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            bg_receiver_locations_p=bg_receiver_locations_p,
            bg_receiver_locations_z=bg_receiver_locations_z,
            bg_receiver_locations_y=bg_receiver_locations_y,
            bg_receiver_locations_x=bg_receiver_locations_x,
            sc_receiver_locations_p=sc_receiver_locations_p,
            sc_receiver_locations_z=sc_receiver_locations_z,
            sc_receiver_locations_y=sc_receiver_locations_y,
            sc_receiver_locations_x=sc_receiver_locations_x,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            sigmazz_0=sigmazz_0,
            sigmayz_0=sigmayz_0,
            sigmaxz_0=sigmaxz_0,
            sigmayy_0=sigmayy_0,
            sigmaxy_0=sigmaxy_0,
            sigmaxx_0=sigmaxx_0,
            vz_sc_0=vz_sc_0,
            vy_sc_0=vy_sc_0,
            vx_sc_0=vx_sc_0,
            sigmazz_sc_0=sigmazz_sc_0,
            sigmayz_sc_0=sigmayz_sc_0,
            sigmaxz_sc_0=sigmaxz_sc_0,
            sigmayy_sc_0=sigmayy_sc_0,
            sigmaxy_sc_0=sigmaxy_sc_0,
            sigmaxx_sc_0=sigmaxx_sc_0,
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


_common.document_propagator_module(ElasticBorn, elastic_born)

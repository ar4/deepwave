"""Elastic wave propagation module for Deepwave.

Implements elastic wave equation propagation using a velocity-stress
formulation with finite differences in time (2nd order) and space
(user-selectable order: 2, 4, 6, or 8). Supports Convolutional Perfectly Matched
Layer (C-PML) boundaries for absorbing reflections.

This module provides both a `torch.nn.Module` interface (`Elastic` class)
and a functional interface (`elastic` function) for performing elastic
wave simulations. The outputs are differentiable with respect to the
material parameters (Lam'e parameters and buoyancy) and source amplitudes.
"""

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave
from deepwave import _backend_utils, _boundaries, _common, _engine, _staggered_grid
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


def elastic_wavefield_names(ndim: int) -> list[str]:
    """Names of the elastic wavefields, in the order the backend stores them.

    Each spatial dimension contributes a block of a velocity component, its
    associated stress components, and the PML memory variables for the
    velocity and stress updates. The blocks are ordered from the slowest
    (``z``) to the fastest (``x``) dimension.
    """
    names: list[str] = []
    if ndim >= 3:
        names += [
            "vz_0",
            "sigmazz_0",
            "sigmayz_0",
            "sigmaxz_0",
            "m_vzz_0",
            "m_vzy_0",
            "m_vzx_0",
            "m_vyz_0",
            "m_vxz_0",
            "m_sigmazzz_0",
            "m_sigmayzy_0",
            "m_sigmaxzx_0",
            "m_sigmayzz_0",
            "m_sigmaxzz_0",
        ]
    if ndim >= 2:
        names += [
            "vy_0",
            "sigmayy_0",
            "sigmaxy_0",
            "m_vyy_0",
            "m_vyx_0",
            "m_vxy_0",
            "m_sigmayyy_0",
            "m_sigmaxyy_0",
            "m_sigmaxyx_0",
        ]
    names += ["vx_0", "sigmaxx_0", "m_vxx_0", "m_sigmaxxx_0"]
    return names


def elastic_pml_slab_axes(ndim: int) -> dict[str, int]:
    """PML memory-variable wavefields and the axis each is localised along.

    Every ``m_*`` memory variable is named after the derivative it stores, so
    the letter immediately before the ``_0`` suffix is its axis (``m_vzy_0``
    holds the y-derivative of ``vz``, hence axis ``y``). Each is nonzero only
    within that one axis's two PML slabs: the compiled kernels write it only
    where that axis's PML flag is set (the ``if (pml_z)`` gating in
    ``elastic_kernels.h``), and :func:`zero_edges_and_interiors` zeroes its
    interior along that same axis -- its ``pml_edge_idx`` table -- on every
    forward entry and backward seed. The velocities and stresses are physical,
    full-domain fields and are absent from the mapping.

    This holds for the standard (per-axis) PML only. Under M-PML
    (``mpml_fraction > 0``) each direction's damping is nonzero wherever *any* axis
    is in its PML, so the fields are not per-axis localised and only the
    pure-interior box is zeroed; consumers of this mapping must therefore
    reject ``mpml_fraction > 0`` themselves (as
    ``deepwave.checkpointing._build_slab_plan`` does).
    """
    dims = _dim_names(ndim)
    return {
        name: dims.index(name[-3])
        for name in elastic_wavefield_names(ndim)
        if name.startswith("m_")
    }


def _elastic_public_wavefield_order(ndim: int) -> list[int]:
    """Indices reordering the backend wavefields into the public return order.

    The backend stores the wavefields grouped by spatial dimension (a velocity
    component, its stress components, then the PML memory variables). Users,
    however, expect them grouped by kind: all velocities, then all stresses,
    then all velocity PML variables, then all stress PML variables. This returns
    the permutation applied to :func:`elastic_wavefield_names` (and to the
    output tensors) to achieve that public ordering. It is the identity for
    ``ndim == 1`` (a single block, nothing to reorder).
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
    return list(range(len(elastic_wavefield_names(ndim))))


def elastic_public_wavefield_names(ndim: int) -> list[str]:
    """The elastic wavefields in public return order (grouped by kind).

    :func:`elastic_wavefield_names` is the backend order, grouped by spatial
    dimension; this applies :func:`_elastic_public_wavefield_order` to it, the
    same permutation :func:`elastic` applies to the output tensors. The
    auxiliary-state collapse is computed in *this* order, since that is where
    the index of the first auxiliary field is meaningful.
    """
    backend = elastic_wavefield_names(ndim)
    return [backend[i] for i in _elastic_public_wavefield_order(ndim)]


def elastic_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The CFS-CPML memory variables ``m_*``, one per velocity or stress spatial
    derivative that the absorbing boundary integrates. They are a subset of
    :func:`elastic_wavefield_names`, spelled the same way, and are what
    :class:`deepwave.AuxiliaryState` holds. Listed in the public
    (grouped-by-kind) order, which is the order the auxiliary state itself
    holds them in.
    """
    return [
        name for name in elastic_public_wavefield_names(ndim) if name.startswith("m_")
    ]


def elastic_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`elastic`.

    Pressure first, then one per velocity component (z, y, x, for the
    dimensions present).
    """
    return ["receiver_amplitudes_p"] + [
        f"receiver_amplitudes_{d}" for d in _dim_names(ndim)
    ]


def elastic_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`elastic`, in return order.

    The public final wavefields -- the velocities and stresses in the public,
    grouped-by-kind order, followed by the single ``auxiliary_state`` that
    replaces the block of PML memory variables -- followed by the receiver
    amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            elastic_public_wavefield_names(ndim), elastic_auxiliary_state_names(ndim)
        ),
        *elastic_receiver_names(ndim),
    ]


def elastic_model_names(ndim: int) -> list[str]:
    """Names of the elastic models for the given number of dimensions.

    The models are the two Lame parameters, the shear modulus evaluated at the
    staggered shear-stress locations (one per off-diagonal stress component),
    and the buoyancy for each velocity component.
    """
    if ndim >= 3:
        return [
            "lamb",
            "mu",
            "mu_zy",
            "mu_zx",
            "mu_yx",
            "buoyancy_z",
            "buoyancy_y",
            "buoyancy_x",
        ]
    if ndim == 2:
        return ["lamb", "mu", "mu_yx", "buoyancy_y", "buoyancy_x"]
    return ["lamb", "mu", "buoyancy_x"]


# The user-facing model arguments, in the order the three functions below
# expect their ``model_requires_grad`` sequence.
USER_MODEL_NAMES = ("lamb", "mu", "buoyancy")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(lamb, mu, buoyancy)``; the result is in
    :func:`elastic_model_names` order, i.e. ``lamb``, ``mu``, one staggered
    ``mu`` per off-diagonal stress component, then one buoyancy per velocity
    component.
    """
    lamb_requires_grad, mu_requires_grad, buoyancy_requires_grad = model_requires_grad
    n_off_diagonal = {1: 0, 2: 1, 3: 3}[ndim]
    return (
        [lamb_requires_grad, mu_requires_grad]
        + [mu_requires_grad] * n_off_diagonal
        + [buoyancy_requires_grad] * ndim
    )


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    One slot per stored spatial derivative rather than one per model: with all
    of ``lamb``/``mu``/``buoyancy`` requiring gradients that is 2 slots in 1-D,
    5 in 2-D and 9 in 3-D.

    The single source of truth for elastic's storage-slot count, shared by the
    forward pass (``ElasticForwardFunc.forward``'s ``storage_flags``) and by
    :func:`deepwave.estimate_memory`, so the two cannot drift.
    """
    lamb_requires_grad, mu_requires_grad, buoyancy_requires_grad = model_requires_grad
    stress_requires_grad = lamb_requires_grad or mu_requires_grad
    flags: list[bool] = []
    if ndim >= 3:
        flags += [
            buoyancy_requires_grad,
            stress_requires_grad,
            mu_requires_grad,
            mu_requires_grad,
        ]
    if ndim >= 2:
        flags += [buoyancy_requires_grad, stress_requires_grad, mu_requires_grad]
    flags += [buoyancy_requires_grad, stress_requires_grad]
    return flags


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models.

    The inverse of :func:`internal_model_grad_flags`, for callers (the
    out-of-memory message in :mod:`deepwave._engine`) that hold the internal
    models rather than the user's.
    """
    return [
        models[0].requires_grad,
        models[1].requires_grad,
        models[-1].requires_grad,
    ]


# Declarative description of the elastic propagator: a staggered-grid solver
# with one velocity source/receiver type per spatial dimension plus a pressure
# (normal-stress) type.
SPEC = _engine.PropagatorSpec(
    name="elastic",
    status="stable",
    source_channels=_engine.pressure_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    deprecated_source_channels=_engine.PRESSURE_RATE_SOURCE_DEPRECATIONS,
    supported_options=frozenset({"mpml_fraction", "surface_method"}),
    staggered=True,
    model_names=elastic_model_names,
    wavefield_names=elastic_wavefield_names,
    n_source_types=lambda ndim: ndim + 1,
    n_receiver_types=lambda ndim: ndim + 1,
    output_names=elastic_output_names,
    auxiliary_state_names=elastic_auxiliary_state_names,
    public_wavefield_names=elastic_public_wavefield_names,
    pml_slab_axes=elastic_pml_slab_axes,
    # lambda, mu, and buoyancy own the gradients, so their grad/batched flags
    # are the ones passed to the C backend.
    c_model_flag_indices=[0, 1, -1],
    spatial_parallel=True,
)


def prepare_parameters(mu: torch.Tensor, buoyancy: torch.Tensor) -> list[torch.Tensor]:
    """Prepares elastic properties for the free surface method.

    This function applies the Zeng et al. (2012) logic for all cell types.

    Args:
        mu (torch.Tensor): Lame parameter mu.
        buoyancy (torch.Tensor): Buoyancy (1/rho).

    Returns:
        tuple: (mu_yx, buoyancy_y, buoyancy_x)
    """
    ndim = mu.ndim - 1
    rfmax = 1 / torch.finfo(mu.dtype).max ** (1 / 2)
    parameters = []

    # Mu (harmonic mean)
    mu_safe = torch.where(mu.abs() > rfmax, mu, torch.ones_like(mu))
    if ndim >= 3:
        mask = (
            (rfmax < mu[..., 1:, 1:, :].abs())
            .logical_and(rfmax < mu[..., :-1, :-1, :].abs())
            .logical_and(rfmax < mu[..., 1:, :-1, :].abs())
            .logical_and(rfmax < mu[..., :-1, 1:, :].abs())
        )
        mu_zy_val = 4 / (
            1 / mu_safe[..., 1:, 1:, :]
            + 1 / mu_safe[..., :-1, :-1, :]
            + 1 / mu_safe[..., 1:, :-1, :]
            + 1 / mu_safe[..., :-1, 1:, :]
        )
        mu_zy = torch.where(mask, mu_zy_val, torch.zeros_like(mu_zy_val))
        mu_zy = torch.nn.functional.pad(mu_zy, (0, 0, 0, 1, 0, 1))
        parameters.append(mu_zy)
        del mask, mu_zy_val, mu_zy
        mask = (
            (rfmax < mu[..., 1:, :, 1:].abs())
            .logical_and(rfmax < mu[..., :-1, :, :-1].abs())
            .logical_and(rfmax < mu[..., 1:, :, :-1].abs())
            .logical_and(rfmax < mu[..., :-1, :, 1:].abs())
        )
        mu_zx_val = 4 / (
            1 / mu_safe[..., 1:, :, 1:]
            + 1 / mu_safe[..., :-1, :, :-1]
            + 1 / mu_safe[..., 1:, :, :-1]
            + 1 / mu_safe[..., :-1, :, 1:]
        )
        mu_zx = torch.where(mask, mu_zx_val, torch.zeros_like(mu_zx_val))
        mu_zx = torch.nn.functional.pad(mu_zx, (0, 1, 0, 0, 0, 1))
        parameters.append(mu_zx)
        del mask, mu_zx_val, mu_zx
    if ndim >= 2:
        mask = (
            (rfmax < mu[..., 1:, 1:].abs())
            .logical_and(rfmax < mu[..., :-1, :-1].abs())
            .logical_and(rfmax < mu[..., 1:, :-1].abs())
            .logical_and(rfmax < mu[..., :-1, 1:].abs())
        )
        mu_yx_val = 4 / (
            1 / mu_safe[..., 1:, 1:]
            + 1 / mu_safe[..., :-1, :-1]
            + 1 / mu_safe[..., 1:, :-1]
            + 1 / mu_safe[..., :-1, 1:]
        )
        mu_yx = torch.where(mask, mu_yx_val, torch.zeros_like(mu_yx_val))
        mu_yx = torch.nn.functional.pad(mu_yx, (0, 1, 0, 1))
        parameters.append(mu_yx)
        del mask, mu_yx_val, mu_yx

    # Buoyancy (inverse of arithmetic mean of density)
    parameters.extend(
        _staggered_grid.stagger_density_arithmetic_mean(
            buoyancy, ndim, invert_first=True
        )
    )

    return parameters


class Elastic(torch.nn.Module):
    """A PyTorch Module for elastic wave propagation.

    Stability: stable (see :doc:`/stability`).

    This Module provides a convenient way to perform elastic wave propagation,
    storing the Lamé parameters (`lamb` and `mu`), the buoyancy (`buoyancy`)
    and `grid_spacing` once, so that each :meth:`forward` call needs only the
    arguments that vary between calls.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        lamb: The first Lamé parameter model, lambda, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        mu: The second Lamé parameter model, mu, of shape
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
        grid_spacing: The spatial grid cell size. It can be a single number
            (for isotropic grids) or a sequence of numbers (for anisotropic
            grids).
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
        lamb_requires_grad: A bool specifying whether gradients should be
            computed for `lamb`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        mu_requires_grad: A bool specifying whether gradients should be
            computed for `mu`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        buoyancy_requires_grad: A bool specifying whether gradients should be
            computed for `buoyancy`. Unset by default, which means False.
            Cannot be combined with `requires_grad`.
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
        lamb: torch.Tensor,
        mu: torch.Tensor,
        buoyancy: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        lamb_requires_grad: Optional[bool] = None,
        mu_requires_grad: Optional[bool] = None,
        buoyancy_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        grad = _common.resolve_requires_grad(
            requires_grad,
            ["lamb", "mu", "buoyancy"],
            ["lamb", "mu", "buoyancy"],
            {
                "lamb": lamb_requires_grad,
                "mu": mu_requires_grad,
                "buoyancy": buoyancy_requires_grad,
            },
            "Elastic",
        )
        self.lamb = _common.make_model_parameter(lamb, grad["lamb"], "lamb")
        self.mu = _common.make_model_parameter(mu, grad["mu"], "mu")
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
        source_amplitudes_p: Optional[torch.Tensor] = None,
        source_amplitudes_z: Optional[torch.Tensor] = None,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_locations_pressure_rate: Optional[LocationsLike] = None,
        source_locations_p: Optional[LocationsLike] = None,
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
        surface_method: Literal["ivm", "pm"] = "ivm",
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

        See :func:`elastic` for details (the constructor-owned `lamb`/`mu`/
        `buoyancy`/`grid_spacing` and the storage settings are supplied from
        the stored module state).
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
        return elastic(
            self.lamb,
            self.mu,
            self.buoyancy,
            self.grid_spacing,
            dt,
            source_amplitudes_p=source_amplitudes_p,
            source_locations_p=source_locations_p,
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
            surface_method=surface_method,
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


def elastic(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes_pressure_rate: Optional[torch.Tensor] = None,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_pressure_rate: Optional[LocationsLike] = None,
    source_locations_p: Optional[LocationsLike] = None,
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
    surface_method: Literal["ivm", "pm"] = "ivm",
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
    """Elastic wave propagation (functional interface).

    Stability: stable (see :doc:`/stability`).

    This function performs forward modelling with the elastic wave equation.
    The outputs are differentiable with respect to the input float torch.Tensors.

    For computational performance, multiple shots may be propagated
    simultaneously.

    This propagator uses a staggered grid. See the
    `documentation <https://ausargeo.com/deepwave/elastic>`_
    for a description of the grid layout.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        lamb: The first Lamé parameter model, lambda, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        mu: The second Lamé parameter model, mu, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        buoyancy: The buoyancy (1/density) model. A cell with
            ``buoyancy = 0`` is a vacuum, not an infinitely dense cell: it is
            a *sentinel*, not a literal reciprocal density, so do not compute
            ``1/density`` for vacuum cells (build the parameters from
            ``vp = vs = rho = 0`` via
            :func:`deepwave.vpvsrho_to_lambmubuoyancy`, which maps
            ``rho = 0`` to ``buoyancy = 0``). A vacuum cell has zero density
            and, with ``lamb = mu = 0``, zero stiffness, which is how a free
            surface is created (the "improved vacuum" method: set the vacuum
            cells' ``lamb``, ``mu`` and ``buoyancy`` to zero and use
            ``pml_width = 0`` on that face). See
            :ref:`the vacuum sentinel <vacuum-sentinel>` in
            :doc:`/boundaries`, which also explains how the sentinel is
            averaged onto the staggered nodes. It is of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dt: The time step interval of the input and output (internally a
            smaller interval may be used in propagation to obey the CFL
            (Courant-Friedrichs-Lewy) condition for stability). Due to the
            staggered time grid, stress-related fields (including pressure)
            are at integer time steps `t*dt`, while velocity-related fields
            are at half time steps `(t-0.5)*dt`. Deepwave is unit-agnostic:
            choose any single consistent system of units for ``lamb``,
            ``mu``, ``buoyancy``, ``grid_spacing``, and ``dt``; see
            :doc:`/units`.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``pressure_rate``, ``z``, ``y``,
            ``x`` (``z`` only in 3-D, ``y`` only in 2-D and 3-D); ``p`` is a
            deprecated spelling of ``pressure_rate`` here. Exactly
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
        source_amplitudes_pressure_rate: A Tensor with dimensions [shot, source, time]
            of pressure rates, with units of pressure/time (a positive
            amplitude is compressional). This differs from the acoustic
            family, whose ``source_amplitudes_volume_rate`` is a
            volume-injection rate; see :doc:`/units` for the conversion. Each sample is
            added to the update from `p(t*dt)` to `p((t+1)*dt)`, so it acts
            at `(t+0.5)*dt`, half a step after the pressure receiver of the
            same index is recorded. To drive the propagation with a
            volumetric (extended) source instead of hand-placed point
            sources, build the amplitudes and integer locations with
            :func:`deepwave.wavefields.source_field_to_sources`.
        source_amplitudes_p: Deprecated spelling of
            ``source_amplitudes_pressure_rate``, which means exactly the same
            thing. It warns :class:`deepwave.DeepwaveFutureWarning` and will
            be removed in Deepwave 0.4.0: the letter ``p`` names a pressure
            rate here and a volume rate on the acoustic family, two
            quantities that differ by roughly the bulk modulus, so a source
            channel says which one it injects; see :doc:`/units`. ``p``
            remains the receiver channel's name. Passing both raises.
            Optional.
        source_amplitudes_z: A Tensor with dimensions [shot, source, time] of
            force densities in the z direction, with units of force/volume in
            both 2-D and 3-D. Each sample is added to the update from
            `vz((t-0.5)*dt)` to `vz((t+0.5)*dt)`, so it acts at `t*dt`, half a
            step after the velocity receiver of the same index is recorded.
            This channel exists only in 3-D.
        source_amplitudes_y: A Tensor with dimensions [shot, source, time]
            of force densities in the y direction, with units of
            force/volume. Timed as ``source_amplitudes_z``.
        source_amplitudes_x: A Tensor with dimensions [shot, source, time]
            of force densities in the x direction, with units of
            force/volume. Timed as ``source_amplitudes_z``.
        source_locations_pressure_rate: A Tensor containing the locations of pressure
            sources.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_p: Deprecated spelling of
            ``source_locations_pressure_rate``, as
            ``source_amplitudes_p`` is of ``source_amplitudes_pressure_rate``.
            Passing both raises. Optional.
        source_locations_z: A Tensor with dimensions [shot, source, ndim],
            containing the index in the ndim spatial dimensions of the cell
            that each source oriented in the z spatial dimension is located in,
            relative to the origin of the model.
            This channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_y: A Tensor containing the locations of sources
            oriented in the y spatial dimension.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_x: A Tensor containing the locations of sources
            oriented in the x spatial dimension.
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
        receiver_locations_p: A Tensor containing the coordinates of the
            pressure receivers. In 2-D the recorded pressure is the in-plane
            mean normal stress ``-(sigma_xx + sigma_yy)/2``, which under
            plane strain is not the thermodynamic pressure; see
            :func:`deepwave.elastic`'s ``receiver_amplitudes_p`` entry for
            the full statement.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_z: A Tensor with dimensions [shot, receiver, ndim],
            containing the coordinates of the cell containing each receiver
            oriented in the z spatial dimension.
            This channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_y: A Tensor containing the coordinates of the
            receivers oriented in the y spatial dimension.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        receiver_locations_x: A Tensor containing the coordinates of the
            receivers oriented in the x spatial dimension.
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
        vz_0: Initial vz (velocity in the z dimension) wavefield at time
            step -1/2. Like every initial wavefield, it may be provided
            either at the shape of the simulation domain (the extracted model
            region plus the PML), which is the shape a previous call returns,
            or at the shape of the extracted model region itself, which
            Deepwave zero-pads for you; see the `wavefield_0` entry of
            :func:`deepwave.scalar` for the full description.
        vy_0: Initial vy (velocity in the y dimension) wavefield at time
            step -1/2.
        vx_0: Initial vx (velocity in the x dimension) wavefield at time
            step -1/2.
        sigmazz_0: Initial value for the zz component of the stress field at
            time step 0.
        sigmayz_0: Initial value for the yz component of the stress field at
            time step 0.
        sigmaxz_0: Initial value for the xz component of the stress field at
            time step 0.
        sigmayy_0: Initial value for the yy component of the stress field at
            time step 0.
        sigmaxy_0: Initial value for the xy component of the stress field at
            time step 0.
        sigmaxx_0: Initial value for the xx component of the stress field at
            time step 0.
        initial_state: An optional bundle of initial wavefields, supplied as
            either a :class:`deepwave.PropagatorOutput` (the return of a
            previous call), that output's `final_state` mapping, or any mapping
            whose keys are the public state field names (`vz_0`, `sigmazz_0`,
            ...) and `auxiliary_state` (a :class:`deepwave.AuxiliaryState` from a
            previous call). Each entry fills the correspondingly-named initial
            wavefield that was not also passed explicitly (matching is by name,
            so the public-vs-backend wavefield ordering is irrelevant), so a
            previous output can be passed back in to continue the simulation
            (its `receiver_amplitudes_*` fields are ignored), and it is the
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

            The individual absorbing-boundary field names (`m_vzz_0`, ...,
            `m_sigmaxxx_0`) are still accepted as keys, with a
            `FutureWarning`; they will stop being accepted in Deepwave
            0.4.0. Providing a
            wavefield both here and as an explicit argument raises an error.
            For the continuation to exactly reproduce a single run over
            the combined time range, the same internal time step and the same
            absorbing boundary must be used throughout. The first holds when
            the user `dt` already satisfies the CFL condition (no internal
            sub-stepping); otherwise the segment boundary introduces a small
            discretisation difference. The second is automatic: when
            `pml_freq` or `max_vel` is omitted, the value recorded in
            `initial_state` is used, rather than one inferred from this
            call's own portion of the source record or from its model, so the
            continuation builds the same absorbing boundary as the run it
            continues. Passing either explicitly overrides that, and a
            mismatch with the recorded value then warns; a mismatched
            internal time step raises. When a gradient will be taken,
            each call retains the initial wavefields it is given until its
            backward pass runs -- the second-order (Hessian-vector-product)
            path replays the propagation from them -- so a differentiable
            continuation loop holds one full wavefield set per iteration, and
            dropping your own reference to the previous output does not release
            it; run segments that do not need a gradient under
            ``torch.no_grad()``, or wrap segments in
            ``torch.utils.checkpoint.checkpoint`` over
            :func:`deepwave.checkpointing.run_segments`, to bound the peak
            memory. If `initial_state` is a `PropagatorOutput` and neither
            `origin` nor `survey_pad` is also passed, the survey origin used
            to produce it is recovered automatically, so the continuation
            extracts the same region of the model rather than assuming an
            origin of zero.

            The number of shots (the leading dimension of every per-shot
            tensor) is taken from the source locations if any are given,
            otherwise from the initial wavefields. So a call with `nt` and
            no sources takes its shot count from the wavefields given here;
            with neither sources nor initial wavefields the call raises.
            Optional, default None.
        accuracy: The finite difference order of accuracy. Possible values are
            2, 4, 6, and 8, with larger numbers resulting in more accurate
            results but greater computational cost. Default 4.
        pml_width: A single number, or a sequence of numbers for each side,
            specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model.
            If a single value is provided, it will be used for all edges.
            If a sequence is provided, it should contain values for the
            edges in the following order:

            - beginning of first dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension
            - ...

            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective rigid surface, set the
            value for that edge to be zero. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
            model. Optional, default 20.
            A face whose ``pml_width`` is zero does not absorb: it becomes an implicit
            reflecting boundary whose type depends on the propagator and the field
            component. See :ref:`bare-zero-width-face` for the per-propagator reflection
            coefficient and surface location, and prefer naming the condition with
            ``boundary`` (or :func:`deepwave.boundaries`), which also uses the exact
            image rather than this approximate one.
        boundary: An optional single string, applying to every face, or a
            list of `2 * ndim` strings selecting a
            first-class flat-face boundary condition per face, in the same
            order as `pml_width` and with the same semantics as
            :func:`deepwave.acoustic`'s `boundary` (`"pml"`, `"free"`,
            `"rigid"`, or `"periodic"`; a non-`"pml"` face requires
            `pml_width == 0` on that face). This is a fast, flat-face
            image-method shortcut for the common case of a boundary lying
            exactly on a domain edge -- it does **not** replace the improved
            vacuum method (IVM, see the `buoyancy` argument above), which
            remains the only way to represent an irregular or topographic
            free surface. The per-field parities (which of the stress-tensor
            and velocity components are imaged odd/even, and which need no
            imaging at all) are derived from Robertsson (1996)'s stress-image
            method and documented, with citations, in docs/boundaries.rst's
            Policy section. Validated on both the eager
            (`backend="eager"`) and the default compiled C/CUDA
            backends, against analytic reflection coefficients, against the
            improved vacuum method, and by gradient checking. Forward-mode AD
            (`torch.func.jvp`) and the
            compiled second-order (HVP) path -- which route through
            `elastic_born`'s compiled backend -- also support a non-`"pml"`
            face (`elastic_born` has its own `bc`-threaded compiled kernel).
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
        mpml_fraction: The multiaxial PML (M-PML) cross-damping fraction, an
            opt-in stability option for elastic media. Inside each absorbing
            layer the standard PML damps only the direction normal to that
            layer's face; with ``mpml_fraction = p`` the damping profile of
            that layer is also applied, scaled by ``p``, to each of the other
            axes, so every axis in the layer is damped by ``p`` times that
            layer's own profile in addition to whatever its own layer
            contributes. This is the multiaxial PML of Meza-Fajardo and
            Papageorgiou (2008, https://doi.org/10.1785/0120070223), in which
            the same fraction is called ``p``. The default of 0 gives the
            standard CFS-C-PML, with behaviour and performance byte-identical
            to not having the feature. Elastic simulations that combine a thin
            region between two facing free surfaces with a high Poisson's ratio
            and a perpendicular PML can suffer a slow exponential (late-time)
            instability, which a small positive value cures. The trade-off is
            that the M-PML is no longer perfectly matched, so the layer
            reflects more: on a homogeneous elastic half-space with
            ``pml_width=20``, the reflection coefficient measured by
            ``benchmarks/mpml_reflection.py`` rises from 2.3e-5 (normal
            incidence) to 4.2e-4 (49 degrees) at ``p = 0``, to 8.7e-5 to 2.6e-3
            at ``p = 0.15``, and to 1.1e-4 to 3.3e-3 at ``p = 0.20``. Use the
            smallest value that stabilises your model; 0.1-0.2 is the usual
            range, and there is no universal value (it is medium-dependent).
            Must satisfy ``0 <= mpml_fraction < 1`` and requires at least two
            spatial dimensions. Optional, default 0.0. See the
            boundary-conditions documentation.
        surface_method: How the free surface / topography implied by the vacuum
            cells (``buoyancy = mu = 0``, see the ``buoyancy`` argument) is
            discretised. One of:

            - ``"ivm"`` (default): the improved vacuum method. The stress-free
              condition emerges from setting the vacuum cells' parameters to
              zero; the interior update kernel is used everywhere unchanged.
              Simple and the historical behaviour, but its accuracy degrades as
              Poisson's ratio increases.
            - ``"pm"``: the parameter-modified method of Cao and Chen (2018,
              *Geophysics* 83(6), T313-T332,
              https://doi.org/10.1190/geo2018-0098.1). This option is
              provisional (see :doc:`/stability`): the accuracy
              characteristics of the method are still being mapped, while
              ``surface_method="ivm"`` (the default) is stable. The value
              ``"pm"`` keeps working and keeps its meaning. In addition
              to the vacuum
              cells, the medium parameters at the solid cells lying on the
              discrete surface are modified once, before time stepping, so the
              stress-free condition is satisfied adaptively for any Poisson's
              ratio: the per-component normal-stress stiffnesses are replaced by
              the surface-reduced values (a horizontal-surface, vertical-surface
              or corner/edge cell each keeping only the constitutive terms its
              traction-free condition allows), the surface-plane densities are
              halved, and (in 3D) the surface-plane shear moduli are halved.
              This gives markedly smaller surface-wave error than ``"ivm"`` at
              large Poisson's ratio, at no per-step cost (the interior kernel is
              still used unchanged). Arbitrary staircase topography is supported
              through the vacuum region alone -- there is no separate geometry
              argument. Recommended sampling is about 15 grid points per minimum
              (shear) wavelength near the surface. Requires at least one vacuum
              cell (raises ``ValueError`` otherwise, as there is no surface to
              modify), and rejects models containing fluid regions (``mu == 0``
              with ``lamb != 0``, e.g. a water layer) with ``ValueError``,
              because the vacuum-based classifier cannot distinguish fluid
              from air. Supported in every dimensionality (1D, 2D and 3D) on
              the eager backend (``backend="eager"``, no
              restrictions); in 1D the modification reduces to freezing the
              normal stress at the discrete surface cell (the exact
              stress-free condition), and no density or shear modification
              arises. The default compiled C/CUDA backend
              (``backend="compiled"``) currently implements PM in 2D only
              (other dimensionalities raise ``NotImplementedError``), and
              additionally requires PML boundaries (``boundary=...``
              first-class boundary faces are not supported), and ``mpml_fraction=0``
              -- combinations violating these raise
              ``NotImplementedError``. The ``torch.compile`` Python mode
              (``backend="torch_compile"``) always raises
              ``NotImplementedError`` for ``"pm"``. See the boundary-conditions
              documentation. Optional, default ``"ivm"``.
        max_vel: The conservative speed bound used for the CFL condition and
            PML. If not specified, a bound is derived from the physical speeds
            and the density-weighted constitutive operator. Sharp density
            contrasts can require a larger bound than the pointwise P-wave
            and S-wave speeds. Optional, default None.
            Leaving this `None` makes the discretisation a property of the
            *current model*: the internal time step (and therefore
            `RunInfo.step_ratio` and the quadrature of the model gradient)
            changes as the model updates, so an inversion's objective is not
            exactly comparable between iterations. Pass an explicit `max_vel`
            -- a conservative bound over the whole inversion --
            for a reproducible objective. Nothing is read from the model
            when the call continues an earlier one: an omitted `max_vel` then
            takes the value recorded in `initial_state`. The value actually
            used is reported as `RunInfo.max_vel`. A value **below** the
            material/operator bound raises before propagation. Use
            `deepwave.plan` with automatic inference to inspect the bound.
            It has to be given when `lamb`, `mu` or `buoyancy` is batched by
            :func:`torch.func.vmap`, which hides the values the maximum is
            otherwise read from. Every other `torch.func` transform
            (:func:`torch.func.grad`, :func:`torch.func.jacrev`,
            :func:`torch.func.jvp`) leaves them readable, so nothing has to be
            supplied under those. The maximum arrives as a Python number, so
            the absorbing boundary and the internal time step it chooses are
            constants of the call rather than differentiable functions of the
            model. When it is inferred, the model gradient at the cell the
            maximum is read from omits the term that acts through those two;
            every other cell's gradient is unaffected. Passing `max_vel`
            removes that path. An exact model gradient of the fixed discrete
            call also requires internal-step sampling and lossless storage;
            see :doc:`/what_deepwave_calculates`.
            :doc:`/capabilities` lists, for each
            propagator, the models the maximum is read from.
        courant_number: Optional Courant number for the CFL time step. If None
            (default), the conservative
            :data:`deepwave.DEFAULT_COURANT_NUMBER` is used (historical
            behaviour, bit-identical). If given, it is the actual dimensionless Courant
            number targeted; it is validated against the critical Courant
            number for the chosen accuracy order (staggered-grid values) and
            errors if unstable. See the discretisation documentation. Optional,
            default None.
        survey_pad: Padding to apply around sources and receivers to restrict
            the simulation domain. See the documentation for the `scalar`
            propagator for a full description.
        origin: The origin of the provided initial wavefields relative to the
            origin of the model. See :func:`deepwave.scalar` for how this
            interacts with `initial_state` (the continuation rule): passing
            an `origin` that disagrees with the region `initial_state`
            records raises.
        nt: The number of time steps to run the simulation for. Optional: if
            it is not given, the length of the source amplitudes is used.
            When both are given, ``nt`` is the length of the run and the
            source amplitudes are zero-padded or truncated in time to it.

            The number of shots (the leading dimension of every per-shot
            tensor) is taken from the source locations if any are given,
            otherwise from the initial wavefields; with neither the call
            raises. See :func:`deepwave.scalar`.
        model_gradient_sampling_interval: The number of time steps between
            contributions to the model gradient. See :func:`deepwave.scalar`
            for the full description; in particular, the CFL ``step_ratio``
            multiplies it (the quadrature stride is
            ``step_ratio * model_gradient_sampling_interval``), it must
            divide ``nt`` whenever a model requires a gradient, and it
            applies only to the compiled backend -- the pure-Python backends
            step every internal time step and return the exact discrete
            adjoint regardless of it.
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
        freq_taper_frac: The fraction of the end of the source and
            receiver amplitudes in the frequency domain to cosine taper if
            they are resampled due to the CFL condition.
        time_pad_frac: The amount of zero padding that will be added to the
            source and receiver amplitudes before resampling and removed
            afterwards, if they are resampled due to the CFL condition, as a
            fraction of their length.
        time_taper: Whether to apply a Hann window to resampled source and
            receiver amplitudes. Upsampling tapers its output after the
            inverse FFT; downsampling tapers its input before the FFT. The
            FFT-based resampling treats the time axis as periodic, so
            amplitudes that do not decay to (near) zero at the edges -- e.g.
            field data cropped from a longer recording -- can produce
            wraparound ringing. Tapering can reduce this ringing at the cost
            of attenuating genuine signal near the edges. Leave as False if
            amplitudes already taper to (near) zero at the edges.
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
            mutation is supported. Mutating what
            :meth:`deepwave.CallbackState.get_wavefield` or
            :meth:`deepwave.CallbackState.get_model` returns is not: see that
            class's docstring for the full contract.
        backward_callback: A function that will be called during the backward
            pass. See :class:`deepwave.CallbackState` for the
            state that will be provided to the function (the same mutation
            contract applies). Retrieving the reconstructed forward
            wavefield during a backward callback
            (``get_wavefield(..., role="forward")``) is a capability of
            :func:`deepwave.scalar` only; see its ``backward_callback``
            documentation. If the output being differentiated turns out to be
            unused, so that autograd has only an undefined (identically zero)
            gradient to propagate, the callback still fires on its usual
            schedule and is given an adjoint wavefield of zeros, never
            ``None``.
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
            ``"auto"`` resolves to the compiled C/CUDA kernel. Note that
            ``surface_method="pm"`` is implemented only on the compiled
            backend in 2-D and on the eager backend in any dimension; the
            other combinations raise rather than silently changing backend.
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

            - "auto" (default): float coordinates that do not lie on the grid
              node of the field component being placed are interpolated onto the
              grid using the method of Hicks (Kaiser-windowed sinc); float
              coordinates that do lie on a node (within a small tolerance), and
              integer (torch.long) coordinates, take the exact path and are
              bit-for-bit identical to providing integer cell indices.
            - "hicks": always use Hicks interpolation.
            - "none": never interpolate; only the exact path is accepted
              (integer (torch.long) coordinates, or float coordinates that lie
              on a node within a small tolerance). An off-node float
              coordinate raises a ``ValueError`` rather than being silently
              truncated to the node below it, as ``v0.0.27`` did.

            Coordinates are in grid cells relative to the model origin. Because
            this is a staggered-grid propagator, each velocity component lives
            half a cell from the normal-stress (pressure) node in its own
            dimension, so integer index ``N`` and float ``N + 0.5`` denote the
            same physical point for that component (a ``vx`` source at integer
            ``3`` equals float ``3.5``); the pressure component is on the regular
            grid. With genuinely off-grid coordinates the propagator performs
            band-limited injection/recording, an approximation. A
            :class:`deepwave.Hicks` object may be passed
            as any location argument for full control (dipoles, a custom
            component offset, or free surfaces); the propagator uses its offset
            as given and does not check it against the component. Defaults to
            "auto".
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

            - vz_0: Final velocity wavefield in the z-dimension (if ndim == 3).
            - vy_0: Final velocity wavefield in the y-dimension (if ndim >= 2).
            - vx_0: Final velocity wavefield in the x-dimension.
            - sigmazz_0: Final stress wavefield (zz component) (if ndim == 3).
            - sigmayz_0: Final stress wavefield (yz component) (if ndim == 3).
            - sigmaxz_0: Final stress wavefield (xz component) (if ndim == 3).
            - sigmayy_0: Final stress wavefield (yy component) (if ndim >= 2).
            - sigmaxy_0: Final stress wavefield (xy component) (if ndim >= 2).
            - sigmaxx_0: Final stress wavefield (xx component).
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables. Opaque: pass it back
              (as part of `initial_state`) to continue the simulation; its
              contents are implementation state rather than physics.
            - receiver_amplitudes_p: Recorded pressure data, with units of
              pressure. The samples correspond to times `t*dt`. This "pressure"
              is minus the mean of the recorded normal stresses,
              ``-(sigma_xx + ... )/ndim``. In 3D that is the usual pressure
              ``-(sigma_xx+sigma_yy+sigma_zz)/3``, but in 2D it is the in-plane
              mean normal stress ``-(sigma_xx+sigma_yy)/2``, which under plane
              strain differs from the true thermodynamic pressure (the
              out-of-plane stress ``sigma_zz = lamb*(eps_xx+eps_yy)`` is not
              included). Bear this in mind when comparing 2D elastic "pressure"
              to the acoustic propagator's pressure or to a 3D run.
            - receiver_amplitudes_z: Recorded z-component velocity data, with
              units of velocity. The samples correspond to times `(t-0.5)*dt`
              (if ndim == 3).
            - receiver_amplitudes_y: Recorded y-component velocity data, with
              units of velocity. The samples correspond to times `(t-0.5)*dt`
              (if ndim >= 2).
            - receiver_amplitudes_x: Recorded x-component velocity data, with
              units of velocity. The samples correspond to times `(t-0.5)*dt`.

            The receiver outputs are returned in the order pressure (p) first,
            followed by the velocity components (z, y, x, for the dimensions
            present), matching the order the propagator returns them in the
            output tuple.

            This positional order is easy to index incorrectly (``out[-1]`` is
            the x-component velocity receiver amplitudes, not the pressure
            data); prefer accessing the fields by name instead: the wavefields
            use their initial-state names (``vz_0``, ``sigmazz_0``, ...,
            ``m_sigmaxxx_0``) and the receiver amplitudes are exposed as
            ``receiver_amplitudes_p``, ``receiver_amplitudes_z``,
            ``receiver_amplitudes_y``, and ``receiver_amplitudes_x``.

    """
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            elastic,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
                source_amplitudes_p="source_amplitudes_pressure_rate",
                source_locations_p="source_locations_pressure_rate",
            ),
        )
    backward_callback_needs_forward_wavefield = False
    q_band = None
    q_freq = None
    q_n_mechanisms = 1
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
        source_amplitudes_p=(
            "source_amplitudes_pressure_rate",
            source_amplitudes_p,
        ),
        source_locations_p=(
            "source_locations_pressure_rate",
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
    _common.check_models([lamb, mu, buoyancy], ["lamb", "mu", "buoyancy"], "elastic")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, python_backend, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend, python_backend
    # Fill any initial wavefields supplied via the optional initial_state
    # bundle (e.g. a PropagatorOutput from a previous call), leaving explicit
    # wavefield arguments untouched. Matching is by name, so the public-vs-
    # backend wavefield reordering is irrelevant. Order is preserved for
    # unpacking.
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
    ndim = _common.get_ndim(
        [lamb, mu, buoyancy],
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
    if ndim < 3 and (
        source_locations_z is not None or receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            "elastic: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "z dimension should not be provided."
        )
    if ndim < 2 and (
        source_locations_y is not None or receiver_locations_y is not None
    ):
        raise DeepwaveValueError(
            "elastic: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "y dimension should not be provided."
        )

    if not 0 <= mpml_fraction < 1:
        raise DeepwaveValueError(
            "elastic: mpml_fraction (the multiaxial-PML cross-damping fraction) must "
            f"be in [0, 1), but got {mpml_fraction}."
        )
    if mpml_fraction > 0 and ndim < 2:
        raise DeepwaveValueError(
            "elastic: mpml_fraction > 0 (multiaxial PML) requires at least two "
            "spatial dimensions, but the propagation was determined to be "
            f"{ndim}d. Multiaxial PML adds cross-axis damping, which has no "
            "meaning in 1D; leave mpml_fraction at its default of 0."
        )

    if surface_method not in ("ivm", "pm"):
        raise DeepwaveValueError(
            'elastic: surface_method must be "ivm" or "pm", but got '
            f"{surface_method!r}."
        )
    if surface_method == "pm":
        if backend_mode == "compile":
            raise DeepwaveNotImplementedError(
                'elastic: surface_method="pm" (the Cao and Chen 2018 '
                "parameter-modified free surface) is not implemented for the "
                'backend="torch_compile" Python mode; use the default backend '
                "(which selects the compiled C/CUDA backend here) or "
                'backend="eager". The default surface_method="ivm" is '
                "unaffected on every backend."
            )
        if ndim != 2 and backend_mode != "eager":
            raise DeepwaveNotImplementedError(
                f'elastic: surface_method="pm" in {ndim}D is currently only '
                "implemented on the eager Python backend; pass "
                'backend="eager". The compiled C/CUDA backend '
                "implements PM in 2D only."
            )
        if not bool((mu == 0).any()):
            raise DeepwaveValueError(
                'elastic: surface_method="pm" requires vacuum cells (a '
                "zero-mu region representing the air above the free surface "
                "or topography), "
                "but none were found: mu has no zero entries. The "
                "parameter-modified method realises the free surface by "
                "modifying the medium parameters at the solid cells adjacent "
                "to that vacuum, so a purely solid model has no surface to "
                "apply it to. Build the vacuum region with vp = vs = rho = 0 "
                "(see the boundary-conditions documentation), or use the "
                'default surface_method="ivm".'
            )
        if bool(((mu == 0) & (lamb != 0)).any()):
            raise DeepwaveValueError(
                'elastic: surface_method="pm" defines the free surface by '
                "vacuum cells (vp = vs = rho = 0), but cells with mu == 0 and "
                "lamb != 0 were found (an acoustic/fluid region, such as a "
                "water layer). The parameter-modified classifier would "
                "treat the fluid as vacuum and modify the solid cells "
                "beneath it as if they bordered air, which is not the "
                "physics of a fluid-solid interface. Use the default "
                'surface_method="ivm" for models containing fluid regions.'
            )

    # Check that sources and receivers are not on the last row or column,
    # as these are not used
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
    if ndim >= 2:
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

    # Resolve off-grid / directional (Hicks) source and receiver specifications
    # per component. Each velocity component is staggered half a cell in its own
    # propagation dimension and carries that component's half-cell offset, while
    # the pressure component (built from the normal stresses) sits on the
    # regular grid (offset None). This keeps the invariant that integer index N
    # and float N+offset denote the same physical point. Integer/long locations
    # take the exact path and are bit-for-bit identical to before. The lists are
    # ordered [velocities..., pressure].
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
        lamb,
        "elastic",
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
    receiver_interpolators, receiver_locations = _common.resolve_receiver_interpolators(
        receiver_locations_in,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        lamb.dtype,
        _hicks_grid,
    )

    # A velocity source/receiver must not sit on the final sample of its
    # staggered dimension (that cell is not part of the computational domain).
    # The check runs on the resolved integer grid locations; the velocity
    # components occupy the leading ``ndim`` entries (pressure is last).
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
                and dim_location.max() >= lamb.shape[-ndim + dim] - 1
            ):
                raise DeepwaveRuntimeError(
                    f"elastic: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{lamb.shape[-ndim + dim] - 1}.",
                )
    # Whether the absorbing profiles are a constant of this call: they are
    # built from the maximum wave speed, so they depend on the model only
    # when it was not supplied.
    max_vel_pinned = max_vel is not None
    if not all(_common.values_readable(prop) for prop in [lamb, mu, buoyancy]):
        if max_vel is None:
            raise _common.max_vel_unreadable_error("elastic")
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        vp, vs, _ = _common.lambmubuoyancy_to_vpvsrho(
            lamb.detach(),
            mu.detach(),
            buoyancy.detach(),
        )
        max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
        vp_nonzero = vp[vp != 0]
        min_nonzero_vp = (
            vp_nonzero.abs().min().item() if vp_nonzero.numel() > 0 else 0.0
        )
        vs_nonzero = vs[vs != 0]
        min_nonzero_vs = (
            vs_nonzero.abs().min().item() if vs_nonzero.numel() > 0 else 0.0
        )
        if min_nonzero_vp == 0 and min_nonzero_vs == 0:
            min_nonzero_model_vel = 0.0
        elif min_nonzero_vp == 0:
            min_nonzero_model_vel = float(min_nonzero_vs)
        elif min_nonzero_vs == 0:
            min_nonzero_model_vel = float(min_nonzero_vp)
        else:
            min_nonzero_model_vel = float(min(min_nonzero_vp, min_nonzero_vs))
        del vp, vs, vp_nonzero, vs_nonzero
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
        [lamb, mu, buoyancy],
        ["replicate"] * 3,
        ["lamb", "mu", "buoyancy"],
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
    del lamb, mu, buoyancy
    if ndim >= 3:
        del (
            source_amplitudes_z,
            source_locations_z,
            receiver_locations_z,
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
        )
    if ndim >= 2:
        del (
            source_amplitudes_y,
            source_locations_y,
            receiver_locations_y,
            vy_0,
            sigmayy_0,
            sigmaxy_0,
            m_vyy_0,
            m_vyx_0,
            m_vxy_0,
            m_sigmayyy_0,
            m_sigmaxyy_0,
            m_sigmaxyx_0,
        )
    del (
        source_amplitudes_x,
        source_locations_x,
        receiver_locations_x,
        vx_0,
        sigmaxx_0,
        m_vxx_0,
        m_sigmaxxx_0,
    )
    del source_amplitudes, source_locations, receiver_locations

    models.extend(prepare_parameters(models[1], models[2]))
    del models[2]  # Remove buoyancy as it is no longer needed

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "elastic")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "elastic")
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

    model_shape = models[0].shape[-ndim:]
    # Scale the z source amplitudes by buoyancy_z (models[-3]).
    if ndim == 3 and source_amplitudes_out[-4].numel() > 0:
        source_amplitudes_out[-4] = (
            source_amplitudes_out[-4]
            * _common.gather_model_at_sources(
                models[-3], sources_i[-4], model_shape, n_shots
            )
            * dt
        )
    # Scale the y source amplitudes by buoyancy_y (models[-2]).
    if ndim >= 2 and source_amplitudes_out[-3].numel() > 0:
        source_amplitudes_out[-3] = (
            source_amplitudes_out[-3]
            * _common.gather_model_at_sources(
                models[-2], sources_i[-3], model_shape, n_shots
            )
            * dt
        )
    # Scale the x source amplitudes by buoyancy_x (models[-1]).
    if source_amplitudes_out[-2].numel() > 0:
        source_amplitudes_out[-2] = (
            source_amplitudes_out[-2]
            * _common.gather_model_at_sources(
                models[-1], sources_i[-2], model_shape, n_shots
            )
            * dt
        )
    # source_amplitudes_pressure_rate
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

    # Parameter-modified free surface on the compiled C/CUDA backend (2-D):
    # compute the per-component surface-stiffness arrays once (differentiable
    # in lamb/mu) and pre-fold the surface density doubling into the staggered
    # buoyancies, then hand the four stiffness arrays to the DW_PM kernel
    # variant. The eager backend does this inside its swapped update kernels;
    # here only the four output arrays cross the ABI. The gradient of the four
    # arrays w.r.t. lamb/mu (and of the folded buoyancies w.r.t. buoyancy) is
    # completed by autograd, so the compiled adjoint only needs the constant-
    # coefficient transpose.
    pm_c_arrays: Optional[list[torch.Tensor]] = None
    if surface_method == "pm" and not backend_mode:
        if any(code != _common.BC_PML for code in boundary_codes):
            raise DeepwaveNotImplementedError(
                'elastic: surface_method="pm" on the compiled C/CUDA backend '
                "does not support first-class boundary faces (boundary=...); "
                'use backend="eager".'
            )
        if mpml_fraction > 0:
            raise DeepwaveNotImplementedError(
                'elastic: surface_method="pm" cannot be combined with '
                "mpml_fraction > 0 on the compiled C/CUDA backend (DW_PM and DW_MPML "
                'are separate compiled build variants). Use backend="eager".'
            )
        masks = _pm_surface_masks(models[1], ndim)
        stiffness = _pm_normal_stiffness(models[0], models[1], masks)
        pm_c_arrays = [
            stiffness[0][0].contiguous(),
            stiffness[0][1].contiguous(),
            stiffness[1][0].contiguous(),
            stiffness[1][1].contiguous(),
        ]
        buoyancy_y_pm, buoyancy_x_pm = _pm_modify_buoyancy(
            [models[-2], models[-1]], masks
        )
        models[-2] = buoyancy_y_pm
        models[-1] = buoyancy_x_pm

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
        elastic_func(
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
            surface_method=surface_method,
            pm_c_arrays=pm_c_arrays,
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

    # Gather off-grid receiver data back to the original locations, while the
    # receiver outputs are still in [velocities..., pressure] order (matching
    # receiver_interpolators), i.e. before the pressure-first reordering below.
    # The gather is linear, so it commutes with the normal-stress mean applied
    # to form the pressure trace.
    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 1 - ndim
    )

    # Pressure is calculated as the negative arithmetic mean of normal stresses
    if outputs[-1] is not None:
        outputs[-1] = -outputs[-1] / ndim

    # Reorder the wavefields to that expected by users (grouped by kind). The
    # same permutation names the public outputs in elastic_output_names.
    outputs[: -1 - ndim] = [outputs[i] for i in _elastic_public_wavefield_order(ndim)]

    # Receivers: pressure (the last output) is moved to the front, ahead of the
    # velocity-component receivers, matching the documented pressure-first order.
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


def zero_edges_and_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    interior: bool = True,
    mpml: bool = False,
) -> None:
    """Zeros the edges and/or interiors of wavefields.

    This function modifies the input `wavefields` in-place.

    Args:
        wavefields: A list of wavefield tensors to modify.
        ndim: The number of spatial dimensions.
        fd_pad: The finite difference padding.
        fd_pad_list: A list of finite difference padding for each dimension.
        pml_width: A list of PML widths for each dimension.
        interior: If True, zeros the interior of the wavefields.
        mpml: If True (multiaxial PML), a PML memory variable is legitimately
            nonzero anywhere *any* axis is inside its PML (the cross-axis
            damping), so only the pure-interior box (interior along every
            axis) is zeroed, rather than the standard per-axis interior slabs.

    Builds the elastic-specific ``half_grid`` / ``pml_edge_idx`` tables and
    delegates the shared zeroing pass to
    :func:`deepwave._staggered_grid.zero_edges_and_interiors`.
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
    elif ndim == 2:
        for i in [0, 2, 4, 5, 6, 8]:
            half_grid[i][0] = True
        for i in [2, 4, 5, 7, 9, 12]:
            half_grid[i][1] = True
        pml_edge_idx = [[3, 5, 6, 7], [4, 8, 11, 12]]
    elif ndim == 1:
        for i in [0, 3]:
            half_grid[i][0] = True
        pml_edge_idx = [[2, 3]]
    else:
        return

    _staggered_grid.zero_edges_and_interiors(
        wavefields,
        ndim,
        fd_pad,
        fd_pad_list,
        pml_width,
        half_grid,
        pml_edge_idx,
        interior=interior,
        mpml=mpml,
    )


def _base_level_dispatch() -> Any:
    """Use the shared PyTorch-version-aware base-dispatch context."""
    return _common._base_level_dispatch()  # noqa: SLF001 - shared internal helper


def _model_tangent_in_pml(
    dmodel: torch.Tensor, pml_width: Sequence[int], accuracy: int, ndim: int
) -> bool:
    """Whether a model tangent is non-zero inside the PML band (or FD halo).

    Mirrors ``acoustic._model_tangent_in_pml``. ``dmodel`` is on the
    padded grid (``fd_pad = accuracy // 2`` of finite-difference halo, then
    ``pml_width`` of absorbing band, then the physical interior); anything left
    after zeroing the interior overlaps the PML/FD halo, where the Born-based
    forward-mode derivative does not linearise the PML profile.
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


def _elastic_forward_jvp(
    ctx: Any, tangents: tuple[Any, ...]
) -> tuple[Optional[torch.Tensor], ...]:
    """Forward-mode derivative (JVP) of the elastic propagator via Born.

    The scattered field of :func:`deepwave.elastic_born` **is** the
    directional derivative of :func:`elastic` (see
    ``docs/elastic_born_derivation.md`` section 6): differentiating the discrete
    velocity-stress recurrence in the direction of a model perturbation
    reproduces the Born scattered update, and differentiating the source
    pre-scaling supplies the scattered source automatically.

    Because :func:`elastic`'s differentiable front-end composes the prepared
    (staggered) model set via :func:`prepare_parameters` and pre-scales the
    sources *before* :class:`ElasticForwardFunc`, ``torch.func.jvp`` hands this
    rule the tangents of those prepared quantities directly:

    * prepared-model tangents ``(dlamb, dmu, dmu_zy, dmu_zx, dmu_yx, db_z, db_y,
      db_x)`` -- exactly the Born ``scatter`` models (the chain rule through
      ``prepare_parameters`` is done for us by forward-mode AD, so this rule does
      not call ``prepare_scatter_parameters``).
    * source-amplitude tangents -- the pre-scaled scattered sources: for a
      velocity/force source the input is ``dt * buoyancy_i * f`` so its tangent
      is ``dt * (db_i * f + buoyancy_i * df)``; for the pressure source the input
      is ``-dt * f_p`` so its tangent is ``-dt * df_p`` (no model factor -- the
      elastic pressure asymmetry of section 5-src). Both inject directly with no
      double counting.
    * initial-wavefield tangents -> Born's scattered initial wavefields.

    The tangent is produced by one call to the compiled Born forward core
    (``elastic_born.elastic_born_compiled_core``, the same fast
    C/CUDA kernel the public ``elastic_born`` default path uses) on the unwrapped
    primal tensors; the primal output comes from
    :meth:`ElasticForwardFunc.forward` (the fast compiled backend), so
    ``torch.func.jvp`` runs entirely on the compiled engine. M-PML has no
    compiled Born variant, so it falls back to the (pure-torch, fully validated)
    eager Born driver, exactly as the forward/backward dispatch does.

    **PML caveat (identical to scalar/acoustic).** Born does not linearise the
    PML profile's dependence on the model, so the jvp is exact only where the
    model tangent is zero in the PML region (or ``pml_width=0``); a warning is
    emitted otherwise.
    """
    # elastic_born threads bc, so this Born-routed jvp
    # composes a non-PML boundary= face on the compiled path too -- the
    # tangent is exact wherever the primal reflecting-face rule is (the
    # guard-ring image is a linear operator, applied identically to the
    # primal and its tangent).
    jvp_boundary = list(getattr(ctx, "boundary", []))
    elastic_born_module = importlib.import_module("deepwave.elastic_born")

    try:
        import torch._C._functorch as _functorch_c

        depth = _functorch_c.get_dynamic_layer_stack_depth()
    except (ImportError, AttributeError):  # older torch
        depth = 0
    if depth > 1:
        raise DeepwaveNotImplementedError(
            "elastic: the fast-backend forward-mode (jvp) rule "
            "supports a single torch.func.jvp (or torch.autograd.forward_ad) "
            "level, not nested/composed torch.func transforms. Re-run with "
            'backend="eager" to compose transforms through native '
            "autograd (slower)."
        )

    unwrap = _engine.unwrap_functorch
    ndim = len(ctx.jvp_grid_spacing)
    n_models = len(elastic_model_names(ndim))
    n_src = ndim + 1
    n_wf = len(elastic_wavefield_names(ndim))

    saved = [cast("torch.Tensor", unwrap(t)) for t in ctx.saved_tensors]
    models = saved[:n_models]
    source_amplitudes = saved[n_models : n_models + n_src]
    sources_i = saved[n_models + n_src : n_models + 2 * n_src]
    receivers_i = saved[n_models + 2 * n_src : n_models + 3 * n_src]
    wavefields = saved[n_models + 3 * n_src : n_models + 3 * n_src + n_wf]
    pml_profiles = saved[n_models + 3 * n_src + n_wf :]

    # Tangents correspond to the Function's positional inputs: the leading
    # non-tensor args (index 0 .. _N_LEAD-1), then models, source
    # amplitudes, sources_i, receivers_i, wavefields.
    lead = _N_LEAD
    model_tangents = [unwrap(t) for t in tangents[lead : lead + n_models]]
    source_tangents = [
        unwrap(t) for t in tangents[lead + n_models : lead + n_models + n_src]
    ]
    wf_start = lead + n_models + 3 * n_src
    wavefield_tangents = [unwrap(t) for t in tangents[wf_start : wf_start + n_wf]]

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
        # ``models[0]`` is ``lamb`` on the padded grid; any prepared-model
        # tangent shares its grid, so the PML test on ``dlamb`` is representative.
        # Only when the profiles really do depend on the model: their
        # amplitude is built from the maximum wave speed, so a pinned
        # ``max_vel`` makes them constants of the call.
        if not ctx.jvp_max_vel_pinned and _model_tangent_in_pml(
            model_tangents_f[0], ctx.jvp_pml_width, ctx.jvp_accuracy, ndim
        ):
            warn_at_user_call(
                "torch.func.jvp of elastic: a model tangent is non-zero "
                "inside the PML, where the PML profile depends on the model "
                "through the maximum wave speed inferred from it and the "
                "Born-based forward-mode derivative does not linearise that "
                "dependence, so the returned tangent is inexact there; pass "
                "max_vel= to pin the profiles, or set pml_width=0, if you need "
                "an exact tangent everywhere (see the elastic_born PML "
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
            *empty_rcv,
            *receivers_i,
            *wavefields,
            *wavefield_tangents_f,
        )
        mpml = getattr(ctx, "jvp_mpml", False)
        if mpml:
            # No compiled Born M-PML variant: fall back to the eager driver.
            born_out = elastic_born_module.elastic_born_eager_core(
                list(pml_profiles),
                ctx.jvp_grid_spacing,
                ctx.jvp_dt,
                ctx.jvp_nt,
                ctx.jvp_step_ratio,
                ctx.jvp_accuracy,
                ctx.jvp_pml_width,
                jvp_boundary,
                ctx.jvp_n_shots,
                mpml,
                *born_args,
            )
        else:
            born_out = elastic_born_module.elastic_born_compiled_core(
                list(pml_profiles),
                ctx.jvp_grid_spacing,
                ctx.jvp_dt,
                ctx.jvp_nt,
                ctx.jvp_step_ratio,
                ctx.jvp_accuracy,
                ctx.jvp_pml_width,
                jvp_boundary,
                ctx.jvp_n_shots,
                mpml,
                *born_args,
            )

    # Born outputs: (*bg_wavefields, *sc_wavefields, *bg_receivers,
    # *sc_receivers). The elastic forward outputs are (*wavefields,
    # *receiver_amplitudes, saved_state); their tangents are the Born scattered
    # wavefields, the scattered receivers, and None (for saved_state).
    sc_wavefields = born_out[n_wf : 2 * n_wf]
    sc_receivers = born_out[2 * n_wf + n_src : 2 * n_wf + 2 * n_src]
    return (*sc_wavefields, *sc_receivers, None)


def _elastic_make_forward_call_pm(pm_c: list[torch.Tensor], c_batched: bool) -> Any:
    """Builds the DW_PM forward ``make_call`` (parameter-modified free surface).

    Mirrors :func:`deepwave._engine.canonical_forward_call` exactly (elastic's
    ``c_model_flag_indices`` is ``[0, 1, -1]``, so the grad/batched flags agree),
    then appends the trailing ``c_batched`` flag and the four surface-stiffness
    array pointers (``c00``, ``c01``, ``c10``, ``c11``) the ``DW_PM`` compiled
    symbol expects (see ``elastic.c``/``elastic.cu``'s ``FUNC(forward)`` under
    ``DW_PM``). The four arrays never change during time stepping, so they are
    captured once here.
    """

    def extra_args(state: "_engine.BackendState") -> tuple[Any, ...]:
        return (c_batched, *[c.data_ptr() for c in pm_c])

    return _engine.make_forward_call_with_extra(extra_args)


class ElasticForwardFunc(torch.autograd.Function):
    """Forward propagation of the elastic wave equation."""

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
        max_vel_pinned: bool,
        boundary: list[int],
        pm_c_batched: Optional[bool],
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the elastic wave equation.

        This method is called by PyTorch during the forward pass. It prepares
        the input tensors and calls the appropriate C/CUDA function for wave
        propagation. The tensors and metadata that the backward pass needs are
        bundled into a trailing :class:`deepwave._engine.ForwardSavedState`
        output that ``setup_context`` uses to populate ``ctx``.

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
            max_vel_pinned: Whether the caller supplied ``max_vel``, so that
                the PML profiles are a constant of the call rather than a
                function of the model. Read by the forward-mode (jvp) rule,
                whose caveat only applies when they are a function of it.
            boundary: Per-face ``BC_*`` integer codes (``_common.set_boundary``'s
                output; all-``BC_PML`` for the default/no-boundary case).
                Threaded into ``run_forward_engine`` below, which appends it
                to ``state.bc`` and hence to the compiled ``FUNC(forward)``
                call's trailing ``bc[2*ndim]`` argument (elastic.c/
                elastic.cu) -- the same guard-ring fill the
                eager backend implements via ``_elastic_eager_core``.
            pm_c_batched: None unless the parameter-modified free surface
                (``DW_PM``) is active, in which case it is the four surface-
                stiffness arrays' per-shot-batched flag and those four arrays
                are the trailing four ``args`` tensors (dispatched via
                ``backend_extra="_pm"``). Never combined with ``mpml``
                (validated upstream in :func:`elastic`).
            args: Property models, source amplitudes, source locations (1D),
                  receiver locations (1D), initial wavefields, and (for
                  ``DW_PM``) the four trailing surface-stiffness arrays.

        Returns:
            A tuple containing the final wavefields, memory variables, and
            receiver amplitudes.

        """

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # One backward-storage slot per stored derivative; whether each is
            # stored depends on which Lame parameters / buoyancy require grad.
            # Single-sourced with deepwave.estimate_memory via the
            # module-level rule.
            return storage_slot_flags(user_model_grad_flags(models, ndim), ndim)

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

        # Parameter-modified free surface (DW_PM): the four surface-stiffness
        # arrays are the trailing four inputs; pop them off before the engine
        # slices ``args`` into models/sources/wavefields, and thread them into
        # a DW_PM make_call.
        engine_args = args
        if pm_c_batched is not None:
            engine_args = args[:-4]
            pm_c = list(args[-4:])
            make_call = _elastic_make_forward_call_pm(pm_c, pm_c_batched)
            backend_extra = "_pm"
        else:
            make_call = _engine.canonical_forward_call
            backend_extra = "_mpml" if mpml else ""

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
            args=engine_args,
            storage_flags=storage_flags,
            zero_wavefields=zero_wavefields,
            make_call=make_call,
            backend_extra=backend_extra,
            # Save the raw input wavefields so the Born-powered double-backward
            # (Hessian-vector product, :class:`ElasticBackwardFunc`) can recompute
            # the forward field and the effect of ``h`` on it.
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
        # Save the primal tensors the Born-based ``jvp`` rule needs directly from
        # ``inputs`` (prepared models, pre-scaled source amplitudes, source /
        # receiver indices, initial wavefields, PML profiles), plus its metadata,
        # unconditionally (mirrors Scalar/AcousticForwardFunc.setup_context). The
        # tensor inputs start at ``_N_LEAD`` (after the shared leading block and
        # elastic's trailing ``mpml``/``boundary``/``pm_c_batched`` triple);
        # ``pml_profiles`` is the first leading input and ``grid_spacing`` the
        # second.
        ndim = len(inputs[1])
        n_models = len(elastic_model_names(ndim))
        n_src = ndim + 1
        n_wf = len(elastic_wavefield_names(ndim))
        lead = _N_LEAD
        models = list(inputs[lead : lead + n_models])
        source_amplitudes = list(inputs[lead + n_models : lead + n_models + n_src])
        sources_i = list(inputs[lead + n_models + n_src : lead + n_models + 2 * n_src])
        receivers_i = list(
            inputs[lead + n_models + 2 * n_src : lead + n_models + 3 * n_src]
        )
        wf_start = lead + n_models + 3 * n_src
        wavefields = list(inputs[wf_start : wf_start + n_wf])
        pml_profiles = list(inputs[0])
        ctx.save_for_forward(
            *models,
            *source_amplitudes,
            *sources_i,
            *receivers_i,
            *wavefields,
            *pml_profiles,
        )
        ctx.jvp_grid_spacing = state.grid_spacing
        ctx.jvp_dt = state.dt
        ctx.jvp_nt = state.nt
        ctx.jvp_step_ratio = state.step_ratio
        ctx.jvp_accuracy = state.accuracy
        ctx.jvp_pml_width = state.pml_width
        ctx.jvp_n_shots = state.n_shots
        # ``mpml`` is the second-to-last leading input and ``boundary`` the
        # last; the compiled Born backend has no M-PML variant, so the jvp
        # routes M-PML through the eager Born driver (see
        # _elastic_forward_jvp). ``boundary`` can be non-PML (elastic's own
        # compiled forward/backward and elastic_born's own compiled backend
        # both have bc-threaded kernels), so this jvp rule threads the same
        # per-face ``ctx.boundary`` codes straight through to elastic_born
        # (see _elastic_forward_jvp).
        # Leading tail order (see forward's signature): ..., mpml,
        # max_vel_pinned, boundary, pm_c_batched.
        ctx.jvp_mpml = bool(inputs[_N_LEAD - 4])
        # The absorbing profiles are built from the maximum wave speed, so
        # they depend on the model only when it was inferred from it, which
        # is what the forward-mode rule's caveat is about.
        ctx.jvp_max_vel_pinned = bool(inputs[_N_LEAD - 3])
        ctx.boundary = inputs[_N_LEAD - 2]
        # Parameter-modified free surface: None when not PM, else the per-shot-
        # batched flag; the four stiffness arrays are then the trailing four
        # tensor inputs (saved for the backward's compiled adjoint).
        ctx.pm_c_batched = inputs[_N_LEAD - 1]
        ctx.pm_c = list(inputs[-4:]) if inputs[_N_LEAD - 1] is not None else None

    # torch.func transforms now reach this Function; the ctypes backend cannot
    # process batched/dual tensors, so vmap raises an actionable error pointing
    # to backend="eager". Forward-mode AD (jvp) is supported via the Born
    # relationship (Born scattered output = JVP of elastic); see
    # :func:`_elastic_forward_jvp`.
    vmap = staticmethod(_engine.unsupported_vmap)

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> tuple[Optional[torch.Tensor], ...]:
        """Forward-mode derivative of elastic via the Born relationship.

        See :func:`_elastic_forward_jvp` for how the input tangents map onto the
        elastic Born backend inputs.
        """
        return _elastic_forward_jvp(ctx, tangents)

    @staticmethod
    def backward(
        ctx: Any,
        *args: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        Builds a nested differentiable :class:`ElasticBackwardFunc` (mirroring
        ``acoustic.AcousticForwardFunc``'s use of
        :class:`AcousticBackwardFunc`), so a ``create_graph=True`` backward
        yields the true elastic Hessian-vector product (via the elastic Born
        backend) rather than raising. The nested Function's ``forward`` runs the
        ordinary first-order adjoint; its ``backward`` is the double-backward
        (one Born forward + one Born backward per ``H*h``, see
        ``docs/elastic_born_derivation.md`` section 5).

        Args:
            ctx: A context object with saved information from the forward pass.
            args: Gradients of the outputs of the forward pass.

        Returns:
            A tuple containing the gradients with respect to the inputs of the
            forward pass.

        """
        ndim = _engine.saved_state_ndim(ctx, "elastic")
        n_models = len(elastic_model_names(ndim))
        n_src = ndim + 1
        n_wf = len(elastic_wavefield_names(ndim))
        # Drop the grad slot for the trailing ``ForwardSavedState`` output, then
        # split into wavefield gradients (first) and receiver gradients (last).
        args = args[:-1]
        grad_receivers = list(args[-n_src:])
        grad_wavefields = list(args[:-n_src])
        del args

        # With save_wavefields on, the saved tensors run in the order models
        # then sources_i then receivers_i then source_amplitudes then the initial
        # wavefields and finally the PML profiles.
        saved = list(ctx.saved_tensors)
        models = saved[:n_models]
        sources_i = saved[n_models : n_models + n_src]
        receivers_i = saved[n_models + n_src : n_models + 2 * n_src]
        source_amplitudes = saved[n_models + 2 * n_src : n_models + 3 * n_src]
        wavefields = saved[n_models + 3 * n_src : n_models + 3 * n_src + n_wf]
        pml_profiles = saved[n_models + 3 * n_src + n_wf :]

        mpml = getattr(ctx, "backend_extra", "") == "_mpml"
        boundary = getattr(ctx, "boundary", [_common.BC_PML] * 2 * ndim)
        pm_c_batched = getattr(ctx, "pm_c_batched", None)
        pm_c = getattr(ctx, "pm_c", None)
        pm_tensors = tuple(pm_c) if pm_c is not None else ()

        result = ElasticBackwardFunc.apply(  # type: ignore[no-untyped-call]
            mpml,
            boundary,
            pm_c_batched,
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
            *grad_receivers,
            *models,
            *source_amplitudes,
            *grad_wavefields,
            *wavefields,
            *pm_tensors,
        )
        grad_models = result[:n_models]
        grad_sources = result[n_models : n_models + n_src]
        grad_wf = result[n_models + n_src : n_models + n_src + n_wf]
        # For DW_PM the nested Function returns the four stiffness-array
        # gradients after the wavefield gradients; they map back to the four
        # trailing stiffness-array inputs of ``forward``.
        grad_pm = (
            result[n_models + n_src + n_wf : n_models + n_src + n_wf + 4]
            if pm_c is not None
            else ()
        )
        return (
            # ``_N_LEAD`` None gradients for the leading non-tensor config block
            # (computed dynamically so this always matches ``forward``'s actual
            # leading-argument count).
            *([None] * _N_LEAD),
            *grad_models,  # models (lamb, mu, mu_ij, buoyancy_i)
            *grad_sources,  # source_amplitudes
            *([None] * n_src),  # sources_i (integer indices)
            *([None] * n_src),  # receivers_i (integer indices)
            *grad_wf,  # initial wavefields
            *grad_pm,  # DW_PM surface-stiffness arrays (c00, c01, c10, c11)
        )


# Number of leading non-tensor arguments of ``ElasticForwardFunc.forward`` (the
# shared config block plus elastic's own trailing ``mpml``/``boundary``
# pair), all of which receive ``None`` gradients in ``backward``.
_N_LEAD = _engine.leading_arg_count(ElasticForwardFunc.forward)


def _elastic_make_aux_wavefields(
    grad_wavefields: list[torch.Tensor], ndim: int
) -> list[torch.Tensor]:
    """The ``m_sigma*`` ping-pong temporaries for the elastic adjoint.

    Only the ``m_sigma*`` memory-variable family is double-buffered (derivation
    section 5.3): 4 buffers in 2D (``m_sigmayyy``, ``m_sigmaxyy``, ``m_sigmaxyx``,
    ``m_sigmaxxx``), 9 in 3D, 1 in 1D. The ``m_v*`` family is updated in place.
    """
    aux: list[torch.Tensor] = []
    if ndim >= 3:
        aux += [torch.zeros_like(grad_wavefields[0]) for _ in range(5)]
    if ndim >= 2:
        aux += [torch.zeros_like(grad_wavefields[0]) for _ in range(3)]
    aux.append(torch.zeros_like(grad_wavefields[0]))
    return aux


def _elastic_swap(state: "deepwave._engine.BackendState") -> None:
    """Ping-pong the ``m_sigma*`` memory variables with their aux buffers.

    Swaps only the ``m_sigma*`` family (derivation section 5.3: the velocity-
    adjoint block's memory variables), never the in-place ``m_v*`` family. The
    indices depend on ``ndim`` because the number of memory variables per
    dimension differs.
    """
    ndim = state.ndim
    gw = state.wavefields
    aw = state.aux_wavefields
    if ndim == 3:
        gw[9:14], aw[-9:-4] = aw[-9:-4], gw[9:14]
        gw[20:23], aw[-4:-1] = aw[-4:-1], gw[20:23]
        gw[26], aw[-1] = aw[-1], gw[26]
    elif ndim == 2:
        gw[6:9], aw[-4:-1] = aw[-4:-1], gw[6:9]
        gw[12], aw[-1] = aw[-1], gw[12]
    else:
        gw[3], aw[-1] = aw[-1], gw[3]


def _elastic_make_backward_call_pm(
    pm_c: list[torch.Tensor],
    c_batched: bool,
    grad_pm_c: list[torch.Tensor],
    grad_pm_tmp_ptr: list[int],
) -> Any:
    """Builds the DW_PM backward ``make_call`` (parameter-modified free surface).

    Mirrors :func:`deepwave._engine.canonical_backward_call` exactly (see
    :func:`_elastic_make_forward_call_pm`), then appends the trailing
    ``c_batched`` flag, the four surface-stiffness input pointers, the four
    gradient-output pointers (``grad_c00``..``grad_c11``), and the four
    per-shot gradient-temporary pointers the ``DW_PM`` compiled backward
    expects (see ``elastic.c``/``elastic.cu``'s ``FUNC(backward)`` under
    ``DW_PM``; the CPU symbol accumulates into the finals directly and ignores
    the temporaries, the CUDA symbol reduces the per-shot temporaries into the
    finals across parallel shots).
    """

    def extra_args(state: "_engine.BackendState") -> tuple[Any, ...]:
        return (
            c_batched,
            *[c.data_ptr() for c in pm_c],
            *[g.data_ptr() for g in grad_pm_c],
            *grad_pm_tmp_ptr,
        )

    return _engine.make_backward_call_with_extra(extra_args)


class ElasticBackwardFunc(torch.autograd.Function):
    """First-order elastic adjoint with a Born-powered double-backward (HVP).

    ``forward`` runs the ordinary first-order adjoint (the elastic backend);
    ``backward`` is the double-backward that computes the Hessian-vector product
    ``H*h`` via one elastic Born forward + one elastic Born backward, mirroring
    ``acoustic.AcousticBackwardFunc`` (see
    ``docs/elastic_born_derivation.md`` section 5). Elastic's first-order scheme
    seeds the adjoint velocity/stress fields directly from the output-wavefield
    gradients (no ``grad_wfp`` negation), and the only ping-pong is the
    ``m_sigma*`` memory-variable aux swap (section 5.3).
    """

    @staticmethod
    def forward(
        mpml: bool,
        boundary: list[int],
        pm_c_batched: Optional[bool],
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
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Runs the first-order elastic adjoint via the shared engine.

        ``boundary`` is threaded into
        ``run_backward_core`` below (elastic.c/elastic.cu's ``bc[2*ndim]``
        ABI parameter) and the model mirror-pad just below
        (see :func:`_boundaries.mirror_pad_elastic_models`).

        ``pm_c_batched`` is None unless the parameter-modified free surface
        (``DW_PM``) is active, in which case the four surface-stiffness arrays
        are the trailing four ``tensors`` and their gradients are returned
        after the wavefield gradients.
        """
        ndim = len(grid_spacing)
        n_models = len(elastic_model_names(ndim))
        n_src = ndim + 1
        n_wf = len(elastic_wavefield_names(ndim))
        pm_c = (
            [t.contiguous() for t in tensors[-4:]] if pm_c_batched is not None else None
        )
        if pm_c is not None:
            tensors = tensors[:-4]
        grad_receivers = [t.contiguous() for t in tensors[:n_src]]
        models = [t.contiguous() for t in tensors[n_src : n_src + n_models]]
        # See mirror_pad_elastic_models's docstring: FUNC(backward_v*_cell)
        # differentiates lamb/mu/every mu_* shear stiffness with a neighbour
        # offset (like acoustic's K); FUNC(backward_sigma*_cell)
        # differentiates each buoyancy_d with a neighbour offset on its own
        # axis (like acoustic's own buoyancy_d). A no-op when boundary is
        # all-BC_PML.
        models = _boundaries.mirror_pad_elastic_models(models, boundary, accuracy // 2)
        source_amplitudes = [
            t.contiguous() for t in tensors[n_src + n_models : 2 * n_src + n_models]
        ]
        grad_wavefields = [
            t.contiguous()
            for t in tensors[2 * n_src + n_models : 2 * n_src + n_models + n_wf]
        ]
        # tensors[2*n_src+n_models+n_wf:] are the forward's initial wavefields,
        # needed only by the double-backward (re-read from ``inputs`` in
        # setup_context).
        pml_profiles = [p.contiguous() for p in pml_profiles]

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

        # Parameter-modified free surface (DW_PM): allocate the four
        # stiffness-array gradient buffers (finals + per-shot temporaries, the
        # same final/temp/reduce scheme :func:`allocate_model_gradients` uses
        # for the models) and thread the arrays + gradients into a DW_PM
        # make_call. Their gradients complete the chain rule to lamb/mu via
        # autograd once returned.
        grad_pm_c: list[torch.Tensor] = []
        grad_pm_tmp: list[torch.Tensor] = []
        extra_grad_scratch: list[torch.Tensor] = []
        if pm_c is not None:
            is_cuda = models[0].is_cuda
            aux, _stream = _engine.get_aux_and_stream(
                models[0], n_shots, is_cuda, spatial_parallel=SPEC.spatial_parallel
            )
            grad_pm_c, grad_pm_tmp, grad_pm_tmp_ptr = _engine.allocate_model_gradients(
                pm_c,
                [bool(pm_c_batched)] * 4,
                n_shots,
                aux,
                is_cuda,
                storage_manager.storage_mode,
                models[0].device,
                models[0].dtype,
                cpu_tmp_batch=n_shots,
                spatial_parallel=SPEC.spatial_parallel,
            )
            make_call = _elastic_make_backward_call_pm(
                pm_c, bool(pm_c_batched), grad_pm_c, grad_pm_tmp_ptr
            )
            # The DW_PM backend reduces these per-shot scratch buffers into
            # ``grad_pm_c`` at the end of every backend call, exactly as it
            # does for the model gradients, so they must be re-zeroed between
            # chunks when a ``backward_callback`` splits the adjoint loop.
            extra_grad_scratch = grad_pm_tmp
            backend_extra = "_pm"
        else:
            make_call = _engine.canonical_backward_call
            backend_extra = "_mpml" if mpml else ""

        state = _engine.run_backward_core(
            SPEC,
            extra_grad_scratch=extra_grad_scratch,
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
            make_aux_wavefields=_elastic_make_aux_wavefields,
            zero_wavefields=zero_wavefields,
            make_call=make_call,
            swap=_elastic_swap,
            cpu_tmp_batch=n_shots,
            backend_extra=backend_extra,
            boundary=boundary,
        )
        # Keep the per-shot temporaries alive until the backend has reduced
        # them (they are referenced only through raw pointers in make_call).
        del grad_pm_tmp
        out_s = state.output_slice
        return (
            *state.grad_models,
            *state.grad_sources,
            *[w[out_s] for w in state.wavefields],
            *grad_pm_c,
        )

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Saves the tensors and metadata the double-backward (HVP) needs."""
        (
            mpml,
            boundary,
            pm_c_batched,
            pml_profiles,
            sources_i,
            receivers_i,
            _storage_manager,
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
            *tensors,
        ) = inputs
        # DW_PM appends the four stiffness arrays as the trailing tensor
        # inputs; drop them from the double-backward's view (the HVP path does
        # not support the parameter-modified free surface).
        if pm_c_batched is not None:
            tensors = tensors[:-4]
        ndim = len(grid_spacing)
        n_models = len(elastic_model_names(ndim))
        n_src = ndim + 1
        n_wf = len(elastic_wavefield_names(ndim))
        ctx.mpml = mpml
        ctx.boundary = boundary
        ctx.pm_c_batched = pm_c_batched
        grad_receivers = tensors[:n_src]
        models = tensors[n_src : n_src + n_models]
        source_amplitudes = tensors[n_src + n_models : 2 * n_src + n_models]
        grad_wavefields = tensors[2 * n_src + n_models : 2 * n_src + n_models + n_wf]
        wavefields = tensors[
            2 * n_src + n_models + n_wf : 2 * n_src + n_models + 2 * n_wf
        ]
        ctx.save_for_backward(
            *grad_receivers,
            *models,
            *source_amplitudes,
            *grad_wavefields,
            *wavefields,
            *pml_profiles,
        )
        ctx.n_models = n_models
        ctx.n_src = n_src
        ctx.n_wf = n_wf
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

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Double-backward: the elastic Hessian-vector product ``H*h``.

        One elastic Born forward (recomputing the forward field and the effect
        of ``h`` on it) plus one elastic Born full backward (ordinary backprop
        for the scattered adjoint field, ``h``'s effect for the background
        adjoint field). Their correlation yields ``H*h`` (derivation section
        5.2). ``h`` lives in the prepared-model space (``dlamb``, ``dmu``,
        ``dmu_ij``, ``db_i``) -- the chain rule through ``prepare_parameters`` is
        handled by ``elastic``'s differentiable front-end, exactly as acoustic's
        HVP works in ``(K, B_d)`` space.
        """
        # The Born adjoint also samples model derivatives when the seed is
        # a source or initial-state gradient. Reject the whole second-order
        # path; checking only model-gradient seeds misses mixed derivatives.
        if ctx.step_ratio > 1:
            raise DeepwaveRuntimeError(
                "elastic: compiled double backward with sampled model gradients "
                "is unsupported. Use backend='eager' or an effective sampling "
                "stride of 1."
            )
        elastic_born_module = importlib.import_module("deepwave.elastic_born")
        # The compiled elastic Born backend has no M-PML variants, so the HVP
        # cannot route through it under M-PML. First-order M-PML gradients are
        # fully supported (by ``forward`` above); only the second-order path is
        # unavailable on the compiled backend. Compose HVPs under M-PML on the
        # eager backend, where second-order works through native autograd.
        if getattr(ctx, "mpml", False):
            raise DeepwaveNotImplementedError(
                "elastic: second-order differentiation (double-backward / "
                "Hessian-vector products) is not supported with M-PML on the "
                'compiled backend. Re-run with backend="eager".'
            )
        # elastic_born threads bc, so this HVP composes a
        # non-PML boundary on the compiled path too: the same per-face codes
        # as the first-order adjoint are re-threaded into both the Born
        # forward (step 1) and the Born backward (step 2) below, and the
        # models correlated by step 2's compiled adjoint are mirror-padded
        # exactly as :class:`ElasticBackwardFunc`'s own first-order adjoint
        # does.
        boundary = list(getattr(ctx, "boundary", []))
        if getattr(ctx, "pm_c_batched", None) is not None:
            raise DeepwaveNotImplementedError(
                "elastic: second-order differentiation (double-backward / "
                "Hessian-vector products) is not implemented with "
                'surface_method="pm" on the compiled backend. Use '
                'backend="eager" for double-backward (create_graph=True) with '
                "the parameter-modified free surface."
            )
        n_models = ctx.n_models
        n_src = ctx.n_src
        n_wf = ctx.n_wf

        # Incoming grads: perturbation direction ``h`` w.r.t. the first-order
        # backward's outputs (grad_models, grad_sources, final adjoint
        # wavefields).
        gg_models = [a.contiguous() for a in args[:n_models]]
        gg_sources = [a.contiguous() for a in args[n_models : n_models + n_src]]
        gg_wavefields = [
            a.contiguous() for a in args[n_models + n_src : n_models + n_src + n_wf]
        ]

        saved = list(ctx.saved_tensors)
        grad_receivers = saved[:n_src]
        models = [m.contiguous() for m in saved[n_src : n_src + n_models]]
        source_amplitudes = saved[n_src + n_models : 2 * n_src + n_models]
        grad_wavefields = saved[2 * n_src + n_models : 2 * n_src + n_models + n_wf]
        wavefields = saved[
            2 * n_src + n_models + n_wf : 2 * n_src + n_models + 2 * n_wf
        ]
        pml_profiles = [
            p.contiguous() for p in saved[2 * n_src + n_models + 2 * n_wf :]
        ]
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

        empty_rcv_i = [
            torch.empty(0, dtype=r.dtype, device=r.device) for r in receivers_i
        ]

        # --- Step 1: elastic Born forward. Recompute the forward field
        # (background) and the effect of ``h`` on it (scattered). No gradient
        # stores are needed here (Step 2's compiled backward regenerates them);
        # we only read the scattered forward wavefields (effect of ``h`` on the
        # forward field) and the scattered receivers (effect of ``h`` at the
        # receivers). Background receivers are empty; scattered receivers share
        # the original receiver locations. ``args`` layout matches the Born
        # forward core: bg_models, sc_models, bg_src, sc_src, sources_i,
        # bg_receivers_i, sc_receivers_i, bg_wavefields, sc_wavefields.
        born_forward_args = (
            *models,  # bg_models (background prepared model set)
            *gg_models,  # sc_models = h (perturbation direction)
            *source_amplitudes,  # bg source amplitudes
            *gg_sources,  # sc source amplitudes (source perturbation)
            *sources_i,  # shared source locations
            *empty_rcv_i,  # bg receivers (unused)
            *receivers_i,  # sc receivers (original locations)
            *wavefields,  # bg initial wavefields (forward initial)
            *gg_wavefields,  # sc initial wavefields
        )
        born_out = elastic_born_module.elastic_born_compiled_core(
            list(pml_profiles),
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            boundary,
            n_shots,
            False,  # mpml (never true here: guarded above)
            *born_forward_args,
        )
        # Effect of ``h`` on the forward field / at the receivers.
        sc_forward_wavefields = list(born_out[n_wf : 2 * n_wf])
        gg_receiver_out = list(born_out[2 * n_wf + n_src : 2 * n_wf + 2 * n_src])

        # --- Step 2: elastic Born full backward. The scattered adjoint field is
        # ordinary backprop (sourced by the original residual grad_receivers,
        # seeded by grad_wavefields); the background adjoint field is seeded at
        # zero and carries ``h``'s effect (via the SS4.2 coupling). Their
        # correlation with the (regenerated) stores accumulates ``H*h`` into the
        # background-model gradient, plus the source and initial-wavefield HVP
        # terms. Background models require grad so the kernel writes grad_bg
        # (= H*h); the scattering models (h) are fixed (detached).
        born_backward_args = (
            *[m.detach().requires_grad_(True) for m in models],  # bg_models
            *[g.detach() for g in gg_models],  # sc_models = h
            *source_amplitudes,
            *gg_sources,
            *sources_i,
            *empty_rcv_i,  # bg receivers (no injection)
            *receivers_i,  # sc receivers (original residual injection)
            *wavefields,
            *gg_wavefields,
        )
        grad_r_bg = [torch.empty(0, device=device, dtype=dtype) for _ in range(n_src)]
        grad_r_sc = [r.contiguous() for r in grad_receivers]
        bg_seeds = [torch.zeros_like(w) for w in grad_wavefields]
        sc_seeds = [w.contiguous() for w in grad_wavefields]

        (
            hvp_models,
            _grad_sc,
            hvp_sources,
            _grad_f_sc,
            hvp_wavefields,
            _sc_adjoint_final,
        ) = elastic_born_module.elastic_born_backward_compiled(
            list(pml_profiles),
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            boundary,
            n_shots,
            False,  # mpml
            born_backward_args,
            grad_r_bg,
            grad_r_sc,
            bg_seeds,
            sc_seeds,
        )

        # Assemble the gradients w.r.t. ElasticBackwardFunc.forward's inputs
        # (after the 17 leading non-tensor args -- mpml, boundary, and
        # pm_c_batched among them -- but the returned tuple
        # must still have a slot for each or autograd raises "returned an
        # incorrect number of gradients"; see ScalarBackwardFunc.backward's
        # identical fix): first the receiver-gradient slot, then the models,
        # then the source amplitudes, then the wavefield gradients, and
        # finally the initial wavefields.
        return (
            None,  # mpml
            None,  # boundary
            None,  # pm_c_batched
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
            *gg_receiver_out,  # grad w.r.t. grad_receivers
            *hvp_models,  # grad w.r.t. models (lamb, mu, mu_ij, buoyancy) = H*h
            *hvp_sources,  # grad w.r.t. source_amplitudes
            *sc_forward_wavefields,  # grad w.r.t. grad_wavefields
            *hvp_wavefields,  # grad w.r.t. initial wavefields
        )


def update_velocities(
    models: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    velocities: list[torch.Tensor],
    s_mem_vars_normal: list[torch.Tensor],
    s_mem_vars_shear: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
    fd_coeffs: Optional[list[float]] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]]]:
    """Updates the velocity wavefields and PML memory variables.

    ``fd_coeffs`` is unused and always None (elastic's own custom-FD-
    coefficients option has been removed); the parameter is kept only for
    signature parity with the VTI/TTI propagators' own (separate,
    out-of-scope for this removal) ``update_velocities``-shaped callables.
    """
    ndim = len(grid_spacing)

    # Extract PML profiles
    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    del pml_profiles

    # Calculate derivatives of normal stresses
    dsigmaxxdx = [
        _staggered_grid.diff1h(
            normal_stresses[dim],
            dim,
            accuracy,
            1 / grid_spacing[dim],
            ndim,
        )
        for dim in range(ndim)
    ]

    # Update memory variables for siii
    for dim in range(ndim):
        s_mem_vars_normal[dim] = (
            ah[dim] * s_mem_vars_normal[dim] + bh[dim] * dsigmaxxdx[dim]
        )
        dsigmaxxdx[dim] = dsigmaxxdx[dim] + s_mem_vars_normal[dim]

    sigma_sum = list(dsigmaxxdx)

    # Calculate derivatives of shear stresses
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
            # Out-of-place: the shear term may carry a batch dimension that
            # the normal-stress term in ``sigma_sum`` does not, for example
            # when a shear-only stiffness (``elastic_vti``'s ``c55``) is the
            # tensor a ``torch.func.vmap`` batches over. In-place addition
            # would refuse to broadcast the accumulator.
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

    buoyancy = models[-ndim:]

    velocities = [
        velocities[dim] + buoyancy[dim] * dt * sigma_sum[dim] for dim in range(ndim)
    ]

    return (
        velocities,
        s_mem_vars_normal,
        s_mem_vars_shear,
    )


def update_stresses(
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
    """Updates the stress wavefields and PML memory variables.

    ``fd_coeffs`` is unused and always None (elastic's own custom-FD-
    coefficients option has been removed); the parameter is kept only for
    signature parity with the VTI/TTI propagators' own (separate,
    out-of-scope for this removal) ``update_stresses``-shaped callables.
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

    lamb = models[0]
    mu = models[1]

    # Update normal stresses
    v_strain_sum = torch.sum(torch.stack(dvxdx), dim=0)
    normal_stresses = [
        normal_stresses[dim] + dt * (lamb * v_strain_sum + 2 * mu * dvxdx[dim])
        for dim in range(ndim)
    ]

    # Update shear stresses
    if ndim > 1:
        if ndim == 3:
            mu_zy, mu_zx, mu_yx = models[2], models[3], models[4]
            shear_models = [
                [torch.empty(0), mu_zy, mu_zx],
                [mu_zy, torch.empty(0), mu_yx],
                [mu_zx, mu_yx, torch.empty(0)],
            ]
        else:  # ndim == 2
            mu_yx = models[2]
            shear_models = [[torch.empty(0), mu_yx], [mu_yx, torch.empty(0)]]

        for dim1 in range(ndim):
            for dim2 in range(dim1 + 1, ndim):
                shear_stresses[dim1][dim2] = shear_stresses[dim1][
                    dim2
                ] + dt * shear_models[dim1][dim2] * (
                    dvxdy[dim1][dim2] + dvxdy[dim2][dim1]
                )
                shear_stresses[dim2][dim1] = shear_stresses[dim1][dim2]

    return (
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
    )


def _pm_surface_masks(
    mu: torch.Tensor,
    ndim: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Classifies every cell by its parameter-modified free-surface type.

    The free surface (arbitrary staircase topography) is defined implicitly by
    the vacuum cells (``mu == 0``, the air above the surface). Following Cao and
    Chen (2018) -- see the derivation record in
    ``docs/dev/boundary_conditions/CAO_CHEN_PM_SUMMARY.md`` (their Tables 1-2
    and equations 23-25) -- each solid cell touching vacuum is either a *face*
    cell (its adjacent vacuum lies along exactly one grid axis: the ``H``
    horizontal-surface type when that axis is the depth axis, the
    ``VL``/``VR``/``VF``/``VB`` vertical-surface types for a lateral axis) or a
    *transition* cell (vacuum along two or more axes: the outer corners and,
    in 3-D, edges of eq 25).

    Returns ``(face, transition)`` where ``face[k]`` marks the solid cells
    whose only vacuum-facing axis is spatial axis ``k`` and ``transition``
    marks the multi-axis cells. Interior cells (no adjacent vacuum) appear in
    no mask. The two directions along a lateral axis are not distinguished --
    no coefficient in Tables 1-2 depends on the side, because the vacuum-side
    staggered nodes are already zeroed by the base harmonic averaging.

    ``mu`` has the leading batch dimension followed by the ``ndim`` spatial
    dimensions, the first of which is depth (increasing into the solid), so
    "above" is the smaller-index neighbour along spatial axis 0. Only vacuum
    *above* counts for the depth axis, matching the paper's height-function
    topography; the roof of a cavity (vacuum below) keeps the improved-vacuum
    base treatment. The classification depends only on the (fixed) vacuum
    geometry, not on the parameter values, so it carries no gradient; the
    gradient of the propagator with respect to ``lamb``/``mu``/``buoyancy``
    flows through the modified coefficient *values* placed at these cells (see
    :func:`_pm_normal_stiffness`, :func:`_pm_modify_buoyancy` and
    :func:`_pm_modify_shear`).
    """
    solid = mu != 0
    vac = ~solid

    def neighbour_vacuum(axis: int, side: int) -> torch.Tensor:
        """True where the ``side`` (-1/+1) neighbour along ``axis`` is vacuum."""
        out = torch.zeros_like(vac)
        dim = vac.ndim - ndim + axis
        dst = [slice(None)] * vac.ndim
        src = [slice(None)] * vac.ndim
        if side < 0:
            dst[dim] = slice(1, None)
            src[dim] = slice(None, -1)
        else:
            dst[dim] = slice(None, -1)
            src[dim] = slice(1, None)
        out[tuple(dst)] = vac[tuple(src)]
        return out

    axis_vacuum = [neighbour_vacuum(0, -1)]
    axis_vacuum.extend(
        neighbour_vacuum(axis, -1) | neighbour_vacuum(axis, 1)
        for axis in range(1, ndim)
    )

    face: list[torch.Tensor] = []
    for k in range(ndim):
        other_any = torch.zeros_like(vac)
        for j in range(ndim):
            if j != k:
                other_any = other_any | axis_vacuum[j]
        face.append(solid & axis_vacuum[k] & ~other_any)
    multi = torch.zeros_like(vac)
    for k in range(ndim):
        for j in range(k + 1, ndim):
            multi = multi | (axis_vacuum[k] & axis_vacuum[j])
    transition = solid & multi
    return face, transition


def _pm_normal_stiffness(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    masks: tuple[list[torch.Tensor], torch.Tensor],
) -> list[list[torch.Tensor]]:
    """Per-component normal-stress stiffness coefficients for the PM method.

    Returns the ``ndim x ndim`` matrix ``C`` such that the normal-stress
    update is ``d sigma_i = dt * sum_j C[i][j] * d v_j / d x_j`` (component
    ``i`` is the normal stress along spatial axis ``i``). In the interior
    ``C[i][j] = (lamb + 2 mu)`` on the diagonal and ``lamb`` off it,
    reproducing the stock isotropic constitutive relation exactly. At the
    free-surface cells the stress-free condition reduces these per component
    (Cao and Chen 2018, Tables 1-2 / eqs 23-25): with the modified P-modulus
    ``alpha = 2 mu (lamb + mu) / (lamb + 2 mu)`` and (3-D) the modified
    cross-coupling ``beta = mu lamb / (lamb + 2 mu)``, a face cell with
    surface normal along axis ``k`` freezes the ``sigma_kk`` row entirely and,
    in each surviving row ``i != k``, replaces the diagonal with ``alpha``,
    the coupling to the other in-plane axis with ``beta`` (3-D only -- there
    is no third axis in 2-D) and the coupling to the eliminated axis ``k``
    with zero; a transition (corner/edge) cell freezes every row. ``masks`` is
    the tuple from :func:`_pm_surface_masks`.
    """
    face, transition = masks
    ndim = len(face)
    lp2m = lamb + 2.0 * mu
    # alpha/beta are only consumed at solid surface cells (where lp2m > 0);
    # guard the division so vacuum cells (lp2m == 0) do not create NaNs that
    # would poison the gradient through the torch.where below.
    safe = lp2m > 0
    denom = torch.where(safe, lp2m, torch.ones_like(lp2m))
    alpha = 2.0 * mu * (lamb + mu) / denom
    beta = mu * lamb / denom
    zero = torch.zeros_like(lamb)
    stiffness: list[list[torch.Tensor]] = []
    for i in range(ndim):
        other_face = torch.zeros_like(transition)
        for k in range(ndim):
            if k != i:
                other_face = other_face | face[k]
        row: list[torch.Tensor] = []
        for j in range(ndim):
            if i == j:
                row.append(
                    torch.where(
                        transition | face[i],
                        zero,
                        torch.where(other_face, alpha, lp2m),
                    )
                )
            else:
                frozen = transition | face[i] | face[j]
                base = lamb
                for k in range(ndim):
                    if k != i and k != j:
                        base = torch.where(face[k], beta, base)
                row.append(torch.where(frozen, zero, base))
        stiffness.append(row)
    return stiffness


def _pm_modify_buoyancy(
    buoyancies: list[torch.Tensor],
    masks: tuple[list[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """Applies the PM surface density halving to the staggered buoyancies.

    A velocity node lying in the discrete free-surface plane sees half the solid
    density (the arithmetic average of the solid density with the vacuum's
    zero), i.e. its buoyancy (``1/density``) is doubled (Cao and Chen 2018,
    Tables 1-2 density columns). At a face cell with surface normal along axis
    ``k`` the in-plane velocity nodes (every axis ``i != k``) are doubled; the
    along-normal node keeps its base value (full on the solid side, already
    zeroed on the vacuum side by the improved-vacuum harmonic averaging in
    :func:`prepare_parameters`). A transition (corner/edge) cell doubles every
    component (eq 25). ``buoyancies`` is one staggered buoyancy per spatial
    axis; ``masks`` is from :func:`_pm_surface_masks`.
    """
    face, transition = masks
    ndim = len(face)
    out: list[torch.Tensor] = []
    for i in range(ndim):
        double = transition
        for k in range(ndim):
            if k != i:
                double = double | face[k]
        out.append(torch.where(double, 2.0 * buoyancies[i], buoyancies[i]))
    return out


def _pm_modify_shear(
    shear_models: list[list[torch.Tensor]],
    masks: tuple[list[torch.Tensor], torch.Tensor],
) -> list[list[torch.Tensor]]:
    """Applies the PM in-surface-plane shear halving (3-D only).

    In 3-D, the shear-stress node of the pair ``(i, j)`` at a face cell whose
    surface normal is the third axis ``k`` lies *in* the discrete surface
    plane, so its modulus takes the vacuum average: half the interior value
    (Cao and Chen 2018, Table 2 -- the ``mu eps``-type entries); a transition
    (corner/edge) cell halves every pair (eq 25). In 2-D the single shear node
    is staggered in both spatial axes and therefore never lies in a surface
    plane, and this function is not applied: the validated 2-D behaviour (and
    the compiled ``DW_PM`` variant it is parity-tested against) leaves the
    shear moduli entirely to the base harmonic averaging, which also skips
    eq 25's halving at the isolated 2-D corner cells. ``shear_models`` is the
    symmetric matrix of staggered shear moduli (diagonal unused); ``masks`` is
    from :func:`_pm_surface_masks`.
    """
    face, transition = masks
    ndim = len(face)
    out = [list(row) for row in shear_models]
    for i in range(ndim):
        for j in range(i + 1, ndim):
            halve = transition
            for k in range(ndim):
                if k != i and k != j:
                    halve = halve | face[k]
            halved = torch.where(halve, 0.5 * shear_models[i][j], shear_models[i][j])
            out[i][j] = halved
            out[j][i] = halved
    return out


def update_velocities_pm(
    models: list[torch.Tensor],
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    velocities: list[torch.Tensor],
    s_mem_vars_normal: list[torch.Tensor],
    s_mem_vars_shear: list[list[torch.Tensor]],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
    fd_coeffs: Optional[list[float]] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]]]:
    """Velocity update for the parameter-modified (PM) free surface.

    Identical to :func:`update_velocities` except that the staggered buoyancies
    are halved-density-corrected at the free-surface cells
    (:func:`_pm_modify_buoyancy`) before the velocity update. The stress
    derivatives, PML memory-variable bookkeeping and velocity integration are
    unchanged, so with no vacuum cells this reduces exactly to the stock update.

    ``fd_coeffs`` is unused and always None; see :func:`update_velocities`.
    """
    ndim = len(grid_spacing)

    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    del pml_profiles

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

    buoyancy = _pm_modify_buoyancy(
        list(models[-ndim:]), _pm_surface_masks(models[1], ndim)
    )

    velocities = [
        velocities[dim] + buoyancy[dim] * dt * sigma_sum[dim] for dim in range(ndim)
    ]

    return (
        velocities,
        s_mem_vars_normal,
        s_mem_vars_shear,
    )


def update_stresses_pm(
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
    """Stress update for the parameter-modified (PM) free surface.

    Identical to :func:`update_stresses` except that the normal-stress update
    uses the per-component surface-modified stiffness matrix from
    :func:`_pm_normal_stiffness` (``d sigma_i = dt * sum_j C[i][j] dv_j/dx_j``)
    instead of the single shared ``(lamb, mu)`` constitutive relation, and (in
    3-D) the shear moduli at surface-plane nodes are halved by
    :func:`_pm_modify_shear`. The derivatives and PML memory-variable
    bookkeeping are unchanged, so with no vacuum cells this reduces (to
    rounding) to the stock update.

    ``fd_coeffs`` is unused and always None; see :func:`update_stresses`.
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

    lamb = models[0]
    mu = models[1]

    # Per-component surface-modified normal-stress update (the crux of PM).
    masks = _pm_surface_masks(mu, ndim)
    stiffness = _pm_normal_stiffness(lamb, mu, masks)
    new_normal_stresses = []
    for i in range(ndim):
        contribution = stiffness[i][0] * dvxdx[0]
        for j in range(1, ndim):
            contribution = contribution + stiffness[i][j] * dvxdx[j]
        new_normal_stresses.append(normal_stresses[i] + dt * contribution)
    normal_stresses = new_normal_stresses

    if ndim > 1:
        if ndim == 3:
            mu_zy, mu_zx, mu_yx = models[2], models[3], models[4]
            shear_models = [
                [torch.empty(0), mu_zy, mu_zx],
                [mu_zy, torch.empty(0), mu_yx],
                [mu_zx, mu_yx, torch.empty(0)],
            ]
            # In 3-D some shear nodes lie in the discrete surface plane and
            # take the halved vacuum-average modulus (Table 2); in 2-D no
            # shear node does and the base averaging is left untouched (see
            # :func:`_pm_modify_shear`).
            shear_models = _pm_modify_shear(shear_models, masks)
        else:  # ndim == 2
            mu_yx = models[2]
            shear_models = [[torch.empty(0), mu_yx], [mu_yx, torch.empty(0)]]

        for dim1 in range(ndim):
            for dim2 in range(dim1 + 1, ndim):
                shear_stresses[dim1][dim2] = shear_stresses[dim1][
                    dim2
                ] + dt * shear_models[dim1][dim2] * (
                    dvxdy[dim1][dim2] + dvxdy[dim2][dim1]
                )
                shear_stresses[dim2][dim1] = shear_stresses[dim1][dim2]

    return (
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
    )


def _off_diagonal_shear(matrix: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Flattens a square shear-memory matrix, skipping its diagonal.

    ``v_mem_vars_shear``/``s_mem_vars_shear`` hold a ``torch.empty(0)``
    placeholder at ``matrix[i][i]``; the real per-axis tensors are the
    off-diagonal entries, in row-major order, matching the flattened order
    of ``v_mem_shear_names``/``s_mem_shear_names`` (see PR #121 for the zip
    misalignment this replaces).
    """
    return [
        matrix[i][j]
        for i in range(len(matrix))
        for j in range(len(matrix[i]))
        if j != i
    ]


def _elastic_eager_core(
    update_velocities_fn: Any,
    update_stresses_fn: Any,
    model_names_fn: Any,
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
    spec: _engine.PropagatorSpec,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Shared eager velocity-stress driver for elastic-family propagators.

    The wavefield/PML bookkeeping is identical across the isotropic and
    anisotropic (VTI) elastic propagators (they share the staggered-grid layout
    of :func:`elastic_wavefield_names`); only the constitutive stress update and
    the model names differ. Those vary parts are injected as
    ``update_velocities_fn``, ``update_stresses_fn`` and ``model_names_fn`` so
    both propagators reuse this body.

    ``mpml`` is accepted for signature parity with the C-backend Function but is
    unused here: the eager backend consumes the (already multiaxial when
    ``mpml`` is true) ``pml_profiles`` directly via broadcasting, so no separate
    code path is needed.

    ``boundary`` is the per-face ``BC_*`` integer code list (``2 * ndim``
    entries, ``BC_PML`` everywhere for the default/no-boundary case); non-PML
    faces are guard-filled each half-step via
    ``_boundaries.fill_elastic_stress_boundary``/``fill_elastic_velocity_boundary``
    (see the module-level derivation comment above those functions in
    ``_boundaries.py`` for the per-field parity citations).

    ``spec`` is the calling propagator's spec (``elastic`` or ``elastic_vti``)
    and ``backend`` is the resolved Python backend name
    (``"eager"`` or ``"compile"``), already selected by the
    caller; both are used only to name the propagator/backend if an
    unsupported option is rejected.
    """
    del mpml
    has_boundary = any(code != _common.BC_PML for code in boundary)
    is_batched = any(_common.is_inside_vmap(x) for x in args)
    _engine.reject_unsupported_python_backend_options(
        spec,
        backend,
        storage_mode=storage_mode_str,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
        is_batched=is_batched,
    )
    ndim = len(grid_spacing)
    args_list = list(args)
    del args
    if ndim == 3:
        wavefields_flat = args_list[-27:]
        del args_list[-27:]
    elif ndim == 2:
        wavefields_flat = args_list[-13:]
        del args_list[-13:]
    else:
        wavefields_flat = args_list[-4:]
        del args_list[-4:]

    device = args_list[0].device
    dtype = args_list[0].dtype
    model_shape = args_list[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *model_shape)
    wavefields_flat = [
        _engine.create_or_pad(
            wavefield,
            fd_pad_list,
            device,
            dtype,
            size_with_batch,
        )
        for wavefield in wavefields_flat
    ]

    zero_edges_and_interiors(
        wavefields_flat, ndim, fd_pad, fd_pad_list, pml_width, False
    )

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
        # shear_stresses[i][j] = sigma_ij, where i and j are spatial dimension
        # indices. As sigma_ij = sigma_ji, this is symmetric.
        # For 3D (z=0, y=1, x=2):
        # > shear_stresses[0][1] = shear_stresses[1][0] = s_yz
        # > shear_stresses[0][2] = shear_stresses[2][0] = s_xz
        # > shear_stresses[1][2] = shear_stresses[2][1] = s_xy
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
        # s_mem_vars_shear[i][j] is the memory variable for the shear stress
        # derivative that contributes to the v_i update, where the derivative
        # is with respect to dimension j.
        # e.g. s_mem_vars_shear[0][1] is for d(s_yz)/dy, which contributes to
        # v_z. The corresponding variable is m_s_yzy.
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
        shear_s_names = [
            ["sigmayz_0", "sigmaxz_0"],
            [
                "sigmaxy_0",
            ],
        ]
        v_mem_normal_names = ["m_vzz_0", "m_vyy_0", "m_vxx_0"]
        v_mem_shear_names = [
            ["m_vzy_0", "m_vzx_0"],
            ["m_vyz_0", "m_vyx_0"],
            ["m_vxz_0", "m_vxy_0"],
        ]
        s_mem_normal_names = ["m_sigmazzz_0", "m_sigmayyy_0", "m_sigmaxxx_0"]
        s_mem_shear_names = [
            ["m_sigmayzy_0", "m_sigmaxzx_0"],
            ["m_sigmayzz_0", "m_sigmaxyx_0"],
            ["m_sigmaxzz_0", "m_sigmaxyy_0"],
        ]

    elif ndim == 2:
        v_y, v_x = wavefields_flat[0], wavefields_flat[9]
        s_yy, s_xx = wavefields_flat[1], wavefields_flat[10]
        s_xy = wavefields_flat[2]
        m_v_yy, m_v_xx = wavefields_flat[3], wavefields_flat[11]
        m_v_yx, m_v_xy = wavefields_flat[4], wavefields_flat[5]
        m_s_yyy, m_s_xxx = wavefields_flat[6], wavefields_flat[12]
        m_s_xyy, m_s_xyx = wavefields_flat[7], wavefields_flat[8]

        velocities = [v_y, v_x]
        normal_stresses = [s_yy, s_xx]
        # shear_stresses[i][j] = sigma_ij, where i and j are spatial dimension
        # indices (y=0, x=1). As sigma_ij = sigma_ji, this is symmetric.
        shear_stresses = [[torch.empty(0), s_xy], [s_xy, torch.empty(0)]]
        v_mem_vars_normal = [m_v_yy, m_v_xx]
        # v_mem_vars_shear[i][j] is the memory variable for d(v_i)/dj.
        # e.g. v_mem_vars_shear[0][1] = m_v_yx for d(v_y)/dx.
        v_mem_vars_shear = [[torch.empty(0), m_v_yx], [m_v_xy, torch.empty(0)]]
        s_mem_vars_normal = [m_s_yyy, m_s_xxx]
        # s_mem_vars_shear[i][j] is the memory variable for the shear stress
        # derivative that contributes to the v_i update, where the derivative
        # is with respect to dimension j.
        # e.g. s_mem_vars_shear[0][1] is for d(s_xy)/dx, which contributes to
        # v_y. The corresponding variable is m_s_xyx.
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
        shear_s_names = [
            [
                "sigmaxy_0",
            ],
        ]
        v_mem_normal_names = ["m_vyy_0", "m_vxx_0"]
        v_mem_shear_names = [
            ["m_vyx_0", "m_vxy_0"],
        ]
        s_mem_normal_names = ["m_sigmayyy_0", "m_sigmaxxx_0"]
        s_mem_shear_names = [
            ["m_sigmaxyx_0", "m_sigmaxyy_0"],
        ]

    else:
        v_x = wavefields_flat[0]
        s_xx = wavefields_flat[1]
        m_v_xx = wavefields_flat[2]
        m_s_xxx = wavefields_flat[3]

        velocities = [
            v_x,
        ]
        normal_stresses = [
            s_xx,
        ]
        shear_stresses = [
            [],
        ]
        v_mem_vars_normal = [
            m_v_xx,
        ]
        v_mem_vars_shear = [
            [],
        ]
        s_mem_vars_normal = [
            m_s_xxx,
        ]
        s_mem_vars_shear = [
            [],
        ]
        del v_x, s_xx, m_v_xx, m_s_xxx

        v_names = [
            "vx_0",
        ]
        normal_s_names = [
            "sigmaxx_0",
        ]
        shear_s_names = [
            [],
        ]
        v_mem_normal_names = [
            "m_vxx_0",
        ]
        s_mem_normal_names = [
            "m_sigmaxxx_0",
        ]
        v_mem_shear_names = [
            [],
        ]
        s_mem_shear_names = [
            [],
        ]
    del wavefields_flat

    # One source/receiver type per velocity component plus pressure; the model
    # count is propagator-specific (isotropic elastic vs VTI) and is derived
    # from the injected model-name function.
    n_source_types = ndim + 1
    n_models = len(model_names_fn(ndim))
    receivers_i = args_list[-n_source_types:]
    del args_list[-n_source_types:]
    sources_i = args_list[-n_source_types:]
    del args_list[-n_source_types:]
    source_amplitudes = args_list[-n_source_types:]
    del args_list[-n_source_types:]
    models = args_list[-n_models:]
    del args_list[-n_models:]
    del args_list

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

    model_names = model_names_fn(ndim)
    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = dict(zip(v_names, velocities))
            callback_wavefields.update(dict(zip(normal_s_names, normal_stresses)))
            # shear_stresses is symmetric, so shear_s_names only lists the
            # upper triangle (one row per i, columns j > i); slicing off the
            # diagonal placeholder and the already-emitted lower triangle
            # keeps the zip aligned (see PR #121 for the misalignment this
            # replaces).
            for i in range(len(shear_s_names)):
                callback_wavefields.update(
                    dict(zip(shear_s_names[i], shear_stresses[i][i + 1 :]))
                )
            callback_wavefields.update(dict(zip(v_mem_normal_names, v_mem_vars_normal)))
            callback_wavefields.update(
                dict(
                    zip(
                        [n for row in v_mem_shear_names for n in row],
                        _off_diagonal_shear(v_mem_vars_shear),
                    )
                )
            )
            callback_wavefields.update(dict(zip(s_mem_normal_names, s_mem_vars_normal)))
            callback_wavefields.update(
                dict(
                    zip(
                        [n for row in s_mem_shear_names for n in row],
                        _off_diagonal_shear(s_mem_vars_shear),
                    )
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

            # Refill the stress guard ring (read by the velocity update via
            # sigma_AA/the shared-axis shears' derivatives, and by the
            # pressure receiver record below); on a free face this also
            # holds the on-surface normal-stress node to zero.
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
            if receivers_i_masked[-1].numel() > 0:  # pressure receiver
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

            velocities, s_mem_vars_normal, s_mem_vars_shear = update_velocities_fn(
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
                            1,
                            sources_i_masked[sidx],
                            source_amplitudes_masked[sidx][t],
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
            ) = update_stresses_fn(
                models,
                velocities,
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
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
                            1,
                            sources_i_masked[-1],
                            source_amplitudes_masked[-1][t],
                        )
                        .reshape(size_with_batch)
                    )
                normal_stresses = list(new_normal_stresses)
                del new_normal_stresses

    # Leave the returned wavefields boundary-consistent (e.g. free-surface
    # stress nodes held to zero, guard rings imaged) so a continuation run
    # reproduces the in-loop state (mirrors acoustic_python's own post-loop
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

    s = (
        slice(None),
        *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
    )
    final_wavefields_list = []
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
    elif ndim == 2:
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
    else:
        final_wavefields_list.extend(
            [
                velocities[0],
                normal_stresses[0],
                v_mem_vars_normal[0],
                s_mem_vars_normal[0],
            ]
        )

    return (
        *[field[s] for field in final_wavefields_list],
        *receiver_amplitudes_masked,
    )


def elastic_python(
    *args: Any,
    backend: str,
) -> tuple[torch.Tensor, ...]:
    """Eager isotropic-elastic forward pass.

    Thin wrapper over :func:`_elastic_eager_core` supplying the isotropic
    velocity/stress updates (honouring the ``torch.compile`` module globals
    selected by :func:`elastic_func`) and the isotropic model names.
    ``*args`` is the full leading block (ending in the ``mpml`` flag) followed
    by the tensor inputs, matching ``_elastic_eager_core``'s trailing
    parameters. ``backend`` is the resolved Python backend name
    (``"eager"`` or ``"compile"``), already selected by the caller; used only
    to name it if an unsupported option is rejected.
    """
    return _elastic_eager_core(
        _update_velocities_opt,
        _update_stresses_opt,
        elastic_model_names,
        *args,
        spec=SPEC,
        backend=backend,
    )


_update_velocities_cache = _engine.StepCache(update_velocities)
_update_velocities_opt = update_velocities
_update_stresses_cache = _engine.StepCache(update_stresses)
_update_stresses_opt = update_stresses

# Named offsets into the leading (pre-tensor-block) args elastic_func
# receives from its call site (see
# _backend_utils.PROPAGATOR_REGISTRY["elastic"].leading_args): the index of
# "boundary", and the index one past the last leading arg -- i.e. where the
# model/source/wavefield tensor block starts (before the ``pm_c_batched``
# insertion below shifts it further).
_BOUNDARY_IDX = _backend_utils.leading_arg_offset("elastic", "boundary")
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["elastic"].leading_args)


def elastic_func(
    backend: Optional[Literal["eager", "compile"]] = None,
    *args: Any,
    surface_method: Literal["ivm", "pm"] = "ivm",
    pm_c_arrays: Optional[list[torch.Tensor]] = None,
    max_vel_pinned: bool = False,
) -> tuple[torch.Tensor, ...]:
    """A helper function to apply the ElasticForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ElasticForwardFunc`, which is the entry point for the autograd graph
    for elastic wave propagation.

    Args:
        backend: None, or a string specifying whether to use the
            Python backend. Already coerced (no bool) by the caller.
        surface_method: ``"ivm"`` (the default, improved vacuum method) or
            ``"pm"`` (Cao and Chen 2018 parameter-modified free surface).
            On the eager backend ``"pm"`` swaps in the surface-modified update
            kernels (:func:`update_stresses_pm`/:func:`update_velocities_pm`);
            on the compiled backend it dispatches to the ``DW_PM`` kernel
            variant (see ``pm_c_arrays``). The caller (:func:`elastic`)
            validates the supported combinations before dispatch.
        pm_c_arrays: None on the eager/``ivm`` paths; on the compiled
            ``surface_method="pm"`` path, the four per-component surface-
            stiffness arrays (``c00``, ``c01``, ``c10``, ``c11``) computed by
            :func:`_pm_normal_stiffness`, threaded to the ``DW_PM`` kernel as
            trailing differentiable inputs so their gradients flow back to
            lamb/mu via autograd.
        *args: Variable length argument list to be passed directly to
            `ElasticForwardFunc.apply`.
        max_vel_pinned: Whether the caller supplied `max_vel`, so that the
            PML profiles do not depend on the model. Threaded to
            `ElasticForwardFunc` for the forward-mode (jvp) rule's caveat.

    Returns:
        The results of the forward pass from `ElasticForwardFunc.apply`.

    """
    global _update_velocities_opt, _update_stresses_opt

    if backend:
        mode = _engine.resolve_backend_mode(backend)
        if surface_method == "pm":
            # Among the Python modes the parameter-modified free surface is
            # eager-only; the compiled mode does not yet implement it. (The
            # compiled C/CUDA backend implements it in 2-D and never reaches
            # this branch; its combinations are validated in elastic().)
            if mode != "eager":
                raise DeepwaveNotImplementedError(
                    'elastic: surface_method="pm" (the Cao and Chen 2018 '
                    "parameter-modified free surface) is currently only "
                    "supported on the eager Python backend "
                    '(backend="eager"); backend="torch_compile" '
                    "does not yet implement it."
                )
            _update_velocities_opt = update_velocities_pm
            _update_stresses_opt = update_stresses_pm
            return elastic_python(*args, backend=backend)
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        _update_velocities_opt = _update_velocities_cache.resolve(backend)
        _update_stresses_opt = _update_stresses_cache.resolve(backend)
        return elastic_python(*args, backend=backend)

    # ``args`` (the leading block, as received from ``elastic()``) has
    # ``boundary_codes`` at ``_BOUNDARY_IDX``, right after ``pml_width`` --
    # the same slot ``acoustic_python``/``_elastic_eager_core`` read it
    # from. The compiled path instead needs it as a *trailing* leading arg
    # (mirroring ``ElasticForwardFunc.forward``'s ``mpml, boundary`` order,
    # exactly as ``viscoacoustic_func`` does for its own ``boundary``), so
    # move it there.
    boundary = args[_BOUNDARY_IDX]
    args = (
        args[:_BOUNDARY_IDX]
        + args[_BOUNDARY_IDX + 1 : _N_LEADING_ARGS]
        + (max_vel_pinned, boundary)
        + args[_N_LEADING_ARGS:]
    )

    # Parameter-modified free surface (DW_PM): a leading ``pm_c_batched`` flag
    # (None when not PM, else the four stiffness arrays' per-shot-batched
    # state) is inserted right after ``boundary`` (the last leading arg but
    # one, at ``_N_LEAD - 1``) and before the model/source/wavefield tensors,
    # and the four stiffness arrays (``c00``, ``c01``, ``c10``, ``c11``) are
    # appended as trailing differentiable inputs so their gradients (computed
    # by the compiled adjoint) flow back through autograd to lamb/mu. The
    # base call is unchanged (``pm_c_batched=None``, no trailing arrays).
    ndim = len(cast("list[float]", args[1]))
    _pm_idx = _N_LEAD - 1
    if pm_c_arrays is not None:
        c_batched = _engine.model_batched([pm_c_arrays[0]], ndim)[0]
        args = (*args[:_pm_idx], c_batched, *args[_pm_idx:], *pm_c_arrays)
    else:
        args = (*args[:_pm_idx], None, *args[_pm_idx:])

    # ``forward`` appends a ``ForwardSavedState`` trailing output for
    # ``setup_context``; strip it so callers see only the output tensors.
    return cast(
        "tuple[torch.Tensor, ...]",
        ElasticForwardFunc.apply(*args)[:-1],  # type: ignore[no-untyped-call]
    )


_common.document_propagator_module(Elastic, elastic)

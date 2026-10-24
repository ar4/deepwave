"""Viscoacoustic (SLS/GSLS) wave propagation module for Deepwave.

Implements the variable-density acoustic wave equation augmented with one or
more standard-linear-solid (SLS) relaxation mechanisms on the pressure
update, so that the medium has a finite, frequency-dependent quality factor
``Q`` (attenuation). It is built on the same velocity-pressure staggered grid
as :mod:`deepwave.acoustic`; only the pressure/constitutive update changes
(one extra memory variable ``r`` per mechanism is added at the pressure
nodes). ``q_n_mechanisms == 1`` (the default) is a single SLS mechanism; ``q_
n_mechanisms > 1`` is a generalised-SLS (GSLS) constant-Q fit across a
frequency band (``q_band``) -- see :func:`gsls_fit_weights` -- supported on
the eager backend (``backend="eager"``) and the default compiled C/CUDA
backend (the ``DW_GSLS`` kernel variant); the ``"torch_compile"`` mode
raises ``NotImplementedError``.

The equations solved are::

    rho * d(v)/dt = -grad(p)
    d(p)/dt = -K (1 + tau) div(v) - r
    d(r)/dt = -(1/tausig) ( r + K tau div(v) )

where ``p`` is pressure, ``v`` particle velocity, ``rho`` density,
``K = rho v**2`` the *relaxed* (zero-frequency) bulk modulus, ``r`` the SLS
memory variable, and ``tau = taueps/tausig - 1`` the relaxation defect. The
user supplies a quality factor ``Q`` at the reference angular frequency
``omega0 = 2*pi*q_freq`` (``q_freq`` is required); it is mapped
internally to the single-SLS relaxation times via the constant-Q-at-omega0
relations::

    tausig = (sqrt(1 + 1/Q**2) - 1/Q) / omega0
    taueps = 1 / (omega0**2 * tausig)

so that the SLS quality factor equals ``Q`` exactly at ``omega0``. Away from
``omega0`` the exact SLS curve is ``Q(omega) = (Q/2)(omega/omega0 +
omega0/omega)`` (attenuation is greatest at ``omega0``). ``v``, ``rho`` and
``Q`` are all differentiable model inputs.

See ``docs/viscoacoustic.rst`` and the header of ``viscoacoustic.c`` for the
full derivation of the discretisation (Crank-Nicolson memory-variable update)
and the hand-derived adjoint.

Outputs: final wavefields (pressure, memory variable(s), velocities, PML
memory variables) and receiver amplitudes (empty if no receivers). All
outputs are differentiable with respect to float torch.Tensor inputs.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave
from deepwave import (
    _backend_utils,
    _boundaries,
    _common,
    _engine,
    _gsls,
    _staggered_grid,
)
from deepwave._exceptions import (
    DeepwaveNotImplementedError,
    DeepwaveRuntimeError,
    DeepwaveValueError,
)
from deepwave._gsls import gsls_fit_weights as gsls_fit_weights
from deepwave._gsls import gsls_relaxation_frequencies as gsls_relaxation_frequencies
from deepwave._gsls import q_to_relaxation_times as q_to_relaxation_times
from deepwave._storage import (
    STORAGE_DEFAULT,
    Storage,
    StorageManager,
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


def viscoacoustic_wavefield_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """Names of the viscoacoustic wavefields, in backend order.

    The pressure and the ``n_mechanisms`` GSLS memory variables ``r_0`` ..
    ``r_{n_mechanisms-1}`` (all at the integer / pressure nodes) are followed
    by the velocity components, then the two PML memory variables for each
    spatial dimension: ``phi`` (pressure update) and ``psi`` (velocity
    update). ``n_mechanisms`` defaults to 1 (the single-SLS case, where the
    sole memory variable is named ``r_0``); for ``n_mechanisms > 1`` each
    relaxation mechanism gets its own memory variable, ``r_0`` .. ``r_{L-1}``
    (mechanism index, not a "time step 0" suffix -- unlike every other name
    in this list).
    """
    dims = _dim_names(ndim)
    return (
        ["pressure_0", *[f"r_{i}" for i in range(n_mechanisms)]]
        + [f"v{d}_0" for d in dims]
        + [f"phi_{d}_0" for d in dims]
        + [f"psi_{d}_0" for d in dims]
    )


def viscoacoustic_pml_slab_axes(ndim: int) -> dict[str, int]:
    """PML memory-variable wavefields and the axis each is localised along.

    Same layout as ``acoustic.acoustic_pml_slab_axes``:
    ``phi_{d}`` and ``psi_{d}`` are nonzero only within that dimension's two
    PML slabs (per-axis write gating in ``viscoacoustic_kernels.h``, whose
    backward velocity update writes unconditionally but multiplied by that
    axis's ``a`` profile, which is exactly zero outside the PML; plus the
    per-axis interior zeroing that :func:`zero_edges_and_interiors` performs
    via its ``pml_edge_idx`` table on every entry).

    The GSLS memory variables ``r_0``..``r_{L-1}`` are deliberately absent:
    they are physical relaxation variables, nonzero throughout the interior,
    and are not in ``pml_edge_idx``.
    """
    dims = _dim_names(ndim)
    axes: dict[str, int] = {}
    for i, d in enumerate(dims):
        axes[f"phi_{d}_0"] = i
        axes[f"psi_{d}_0"] = i
    return axes


def viscoacoustic_model_names(ndim: int) -> list[str]:
    """Names of the viscoacoustic kernel model fields.

    ``m1 = K(1+tau)`` (the pressure-update modulus, i.e. the unrelaxed
    modulus), ``c2 = K tau b_r`` (the memory-variable driving coefficient) and
    ``ar`` (the memory-variable decay factor) live at the pressure nodes;
    ``B*`` are the buoyancies at the staggered velocity nodes. All are derived
    differentiably from the user inputs ``v``, ``rho`` and ``Q`` (see
    :func:`prepare_models`).
    """
    return ["m1", "c2", "ar"] + [f"B{d}" for d in _dim_names(ndim)]


# The user-facing model arguments, in the order the public function takes
# them; the kernel models above are all derived from these three.
USER_MODEL_NAMES = ("v", "rho", "q")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(v, rho, q)``; the result is in
    :func:`viscoacoustic_model_names` order -- the three GSLS coefficients
    ``m1``, ``c2`` and ``ar``, then one buoyancy per axis.

    ``m1`` and ``c2`` are functions of the bulk modulus ``K = rho * v**2``
    and of ``q``; ``ar`` is a function of ``q`` alone; the buoyancies come
    from ``rho``.

    Stated for the default ``q_n_mechanisms=1``. With more mechanisms the
    engine routes ``ar`` through a second, larger storage manager and the
    slot count differs; see the note in :func:`storage_slot_flags`.
    """
    v, rho, q = model_requires_grad
    bulk_modulus = v or rho
    return [bulk_modulus or q, bulk_modulus or q, q] + [rho] * ndim


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    The shared rule -- one slot per internal model requiring a gradient --
    is correct here, unlike for the other anisotropic and Born propagators.
    Measured against the engine over every combination of the three models
    at ndim 1, 2 and 3, with ``q_n_mechanisms=1``.

    At ``q_n_mechanisms > 1`` the measured count is one lower, because the
    ``ar`` slot moves into a separate GSLS storage manager that is allocated
    with ``requires_grad=False``. That is not modelled here (and may itself
    be a defect in the multi-mechanism gradient path); the estimate is stated
    for the default single mechanism.
    """
    return _engine.per_model_slot_flags(
        internal_model_grad_flags(model_requires_grad, ndim)
    )


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models.

    ``m1`` requiring a gradient is reported as ``v`` requiring one and
    ``ar`` as ``q``: over-reporting rather than guessing, per the same rule
    the other propagators follow.
    """
    return [
        models[0].requires_grad,
        models[3].requires_grad,
        models[2].requires_grad,
    ]


def viscoacoustic_auxiliary_state_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The GSLS attenuation memory variables ``r_mech{i}``, then the CFS-CPML
    memory variables ``phi_{d}_0`` (pressure update) and ``psi_{d}_0``
    (velocity update), one of each per spatial dimension. Both blocks are
    implementation state whose identity, count and shapes are properties of
    the attenuation and absorbing-boundary implementations rather than of the
    physics the caller asked for, so both are carried in the single opaque
    :class:`deepwave.AuxiliaryState` rather than as public output fields.

    Listed in the public order (:func:`viscoacoustic_public_wavefield_names`),
    which is the order the auxiliary state itself holds them in. The GSLS
    block is the reason this list depends on ``n_mechanisms``.
    """
    dims = _dim_names(ndim)
    return (
        [f"r_mech{i}" for i in range(n_mechanisms)]
        + [f"phi_{d}_0" for d in dims]
        + [f"psi_{d}_0" for d in dims]
    )


def viscoacoustic_public_wavefield_order(ndim: int, n_mechanisms: int = 1) -> list[int]:
    """Indices reordering the backend wavefields into the public return order.

    The backend stores the GSLS attenuation memory variables immediately after
    the pressure they belong to, before the velocities. Publicly they follow
    every physical field, so that the block folded into ``auxiliary_state``
    (the GSLS memory and the absorbing boundary's, together) is contiguous and
    lands after the physical fields rather than between them.

    The same list orders :func:`viscoacoustic_public_wavefield_names` and the
    output tensors :func:`viscoacoustic` returns, so the two cannot disagree.
    """
    names = viscoacoustic_wavefield_names(ndim, n_mechanisms)
    gsls = [i for i, name in enumerate(names) if name in _gsls_names(n_mechanisms)]
    rest = [i for i in range(len(names)) if i not in set(gsls)]
    pressure, physical = rest[:1], rest[1:]
    memory = [i for i in physical if names[i].startswith(("phi_", "psi_"))]
    velocities = [i for i in physical if i not in set(memory)]
    return [*pressure, *velocities, *gsls, *memory]


def _gsls_names(n_mechanisms: int) -> frozenset[str]:
    """The backend names of the GSLS attenuation memory variables."""
    return frozenset(f"r_{i}" for i in range(n_mechanisms))


def viscoacoustic_state_name_renames(
    ndim: int, n_mechanisms: int = 1
) -> dict[str, str]:
    """Public spellings of the GSLS memory-variable state fields.

    Internally the mechanism index is a bare suffix (``r_0``), which reads
    like the ``_0`` *time* index that every other name in
    :func:`viscoacoustic_wavefield_names` carries. Publicly it is marked as
    what it is: ``r_mech0``. Every other state field maps to itself.
    """
    del ndim
    return {f"r_{i}": f"r_mech{i}" for i in range(n_mechanisms)}


def viscoacoustic_public_wavefield_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """The viscoacoustic state fields in public return order and spelling.

    The pressure and the velocities, then the GSLS memory variables, then the
    absorbing boundary's memory: :func:`viscoacoustic_public_wavefield_order`
    applied to :func:`viscoacoustic_wavefield_names`, with the GSLS variables
    respelled per :func:`viscoacoustic_state_name_renames`. The
    auxiliary-state collapse is computed in *this* order, since that is where
    the index of the first auxiliary field is meaningful.
    """
    renames = viscoacoustic_state_name_renames(ndim, n_mechanisms)
    backend = viscoacoustic_wavefield_names(ndim, n_mechanisms)
    return [
        renames.get(backend[i], backend[i])
        for i in viscoacoustic_public_wavefield_order(ndim, n_mechanisms)
    ]


def viscoacoustic_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`viscoacoustic`.

    Pressure first, then one per velocity component (z, y, x, for the
    dimensions present).
    """
    return ["receiver_amplitudes_p"] + [
        f"receiver_amplitudes_{d}" for d in _dim_names(ndim)
    ]


def viscoacoustic_output_names(ndim: int, n_mechanisms: int = 1) -> list[str]:
    """Names of the public outputs of :func:`viscoacoustic`, in return order.

    The public final wavefields -- the pressure and the velocities, followed
    by the single ``auxiliary_state`` that replaces the block of GSLS and PML
    memory variables -- followed by the receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            viscoacoustic_public_wavefield_names(ndim, n_mechanisms),
            viscoacoustic_auxiliary_state_names(ndim, n_mechanisms),
        ),
        *viscoacoustic_receiver_names(ndim),
    ]


def _c_model_flag_indices(ndim: int) -> list[int]:
    """Indices of the models whose grad/batched flags are passed to C.

    ``m1`` and ``c2`` share their requires-grad/batched state (both are
    products of ``K`` and Q-derived factors, so they depend on exactly the same
    user inputs), so one representative (``m1``, index 0) covers both; ``ar``
    (index 2) depends only on ``Q``; and the buoyancies share one representative
    (the last model).
    """
    return [0, 2, -1]


# Declarative description of the viscoacoustic propagator. Same staggered-grid
# layout as acoustic, plus the SLS memory variable ``r`` and its two extra
# pressure-node model fields.
SPEC = _engine.PropagatorSpec(
    name="viscoacoustic",
    status="experimental",
    source_channels=_engine.pressure_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    supported_options=frozenset({"q_freq", "q_n_mechanisms", "q_band"}),
    staggered=True,
    model_names=viscoacoustic_model_names,
    wavefield_names=viscoacoustic_wavefield_names,
    n_source_types=lambda ndim: ndim + 1,
    n_receiver_types=lambda ndim: ndim + 1,
    output_names=viscoacoustic_output_names,
    auxiliary_state_names=viscoacoustic_auxiliary_state_names,
    public_wavefield_names=viscoacoustic_public_wavefield_names,
    state_name_renames=viscoacoustic_state_name_renames,
    pml_slab_axes=viscoacoustic_pml_slab_axes,
    c_model_flag_indices=_c_model_flag_indices,
    spatial_parallel=True,
)


def prepare_models(
    v: torch.Tensor,
    rho: torch.Tensor,
    q: torch.Tensor,
    dt: float,
    omega0: float,
    q_n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
) -> list[torch.Tensor]:
    """Prepares the kernel model fields for the viscoacoustic propagator.

    Maps the user models ``(v, rho, Q)`` to the fields the kernel consumes:
    the pressure-update modulus ``m1 = K(1+tau)``, the memory-variable
    coefficient ``c2 = K tau b_r``, the memory decay factor ``ar``, and the
    buoyancies (identical to ``acoustic.prepare_models``). Here
    ``K = rho v**2`` is the relaxed modulus, ``tau = taueps/tausig - 1``, and
    the Crank-Nicolson r-update coefficients are::

        ar  = (1 - dt/(2 tausig)) / (1 + dt/(2 tausig))
        b_r = (dt/tausig) / (1 + dt/(2 tausig))

    For ``q_n_mechanisms > 1`` (requires ``q_band``), ``c2``/``ar`` instead
    have a new LEADING axis of size ``q_n_mechanisms`` (one Crank-Nicolson
    coefficient pair per relaxation mechanism, see
    :func:`gsls_fit_weights`/:func:`gsls_relaxation_frequencies`), and ``m1
    = K(1 + sum_l tau_l)`` sums the per-mechanism relaxation defects.

    Args:
        v: Wavespeed (low-frequency phase velocity) model.
        rho: Density model.
        q: Quality factor at ``omega0`` (``q_n_mechanisms == 1``) or the
            target constant-Q value across ``q_band`` (``q_n_mechanisms >
            1``).
        dt: Internal time step (after any CFL sub-stepping).
        omega0: Reference angular frequency (``q_n_mechanisms == 1`` only).
        q_n_mechanisms: Number of GSLS relaxation mechanisms.
        q_band: ``(f1, f2)`` frequency band for the GSLS constant-Q fit;
            required when ``q_n_mechanisms > 1``.

    Returns:
        ``[m1, c2, ar, buoyancy_z?, buoyancy_y?, buoyancy_x]``.
    """
    ndim = v.ndim - 1
    models: list[torch.Tensor] = []

    k = rho * v**2  # relaxed bulk modulus, at pressure nodes
    m1, c2, ar = _gsls.sls_coefficients(k, q, dt, omega0, q_n_mechanisms, q_band)
    models.append(m1)
    models.append(c2)
    models.append(ar)

    # Buoyancy (inverse of arithmetic mean of density) -- identical to acoustic.
    models.extend(_staggered_grid.stagger_density_arithmetic_mean(rho, ndim))

    return models


def _resolve_q_freq(q_freq: Optional[float]) -> float:
    """The reference frequency for the Q model, as a float.

    ``q_freq`` is required when ``q_n_mechanisms == 1``, and refused above it:
    a multi-mechanism run fits a constant Q across ``q_band``, so there is no
    single reference frequency, and its coefficients never read one. Both
    cases have already been checked by the time this is called, so ``None``
    here can only be the multi-mechanism case; it becomes 0.0, which no
    coefficient reads, rather than a frequency that could be mistaken for a
    default.

    It is never taken from ``pml_freq``: the PML's tuning frequency and the
    frequency at which Q was measured are different physical quantities that
    happen to be similar numbers in a seismic survey, and letting one default
    to the other made the attenuation model depend on the wavelet.
    """
    return float(q_freq) if q_freq is not None else 0.0


class Viscoacoustic(torch.nn.Module):
    """Convenience nn.Module wrapper for viscoacoustic wave propagation.

    Stability: experimental (see :doc:`/stability`).

    Stores ``v``, ``rho``, ``q`` and ``grid_spacing`` once, so that each
    :meth:`forward` call needs only the arguments that vary between calls.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed (low-frequency phase velocity) model, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        rho: The density model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        q: The quality factor model at the reference frequency, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size (a single number, or one
            per dimension).
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
        v_requires_grad: A bool specifying whether gradients will be
            computed for `v`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        rho_requires_grad: A bool specifying whether gradients will be
            computed for `rho`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        q_requires_grad: A bool specifying whether gradients will be
            computed for `q`. Unset by default, which means False. Cannot
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
        q: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        v_requires_grad: Optional[bool] = None,
        rho_requires_grad: Optional[bool] = None,
        q_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        names = ["v", "rho", "q"]
        grad = _common.resolve_requires_grad(
            requires_grad,
            names,
            names,
            {
                "v": v_requires_grad,
                "rho": rho_requires_grad,
                "q": q_requires_grad,
            },
            "Viscoacoustic",
        )
        self.v = _common.make_model_parameter(v, grad["v"], "v")
        self.rho = _common.make_model_parameter(rho, grad["rho"], "rho")
        self.q = _common.make_model_parameter(q, grad["q"], "q")
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
        pressure_0: Optional[torch.Tensor] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        initial_state: Optional[InitialState] = None,
        accuracy: Accuracy = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        boundary: Optional[Union[str, Sequence[str]]] = None,
        pml_freq: Optional[float] = None,
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

        See :func:`viscoacoustic` for details (the constructor-owned `v`/
        `rho`/`q`/`grid_spacing` and the storage settings are supplied from
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
            backend, None, stacklevel=5
        ).canonical
        return viscoacoustic(
            self.v,
            self.rho,
            self.q,
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
            pressure_0=pressure_0,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            initial_state=initial_state,
            accuracy=accuracy,
            pml_width=pml_width,
            boundary=boundary,
            pml_freq=pml_freq,
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


def viscoacoustic(
    v: torch.Tensor,
    rho: torch.Tensor,
    q: torch.Tensor,
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
    pressure_0: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    initial_state: Optional[InitialState] = None,
    accuracy: Accuracy = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    boundary: Optional[Union[str, Sequence[str]]] = None,
    pml_freq: Optional[float] = None,
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
    """Viscoacoustic (single-SLS) wave propagation (functional interface).

    Stability: experimental (see :doc:`/stability`).

    Forward modelling with the variable-density acoustic wave equation plus one
    standard-linear-solid relaxation mechanism (constant-Q attenuation) on a
    staggered grid. See the module docstring for the equations and the
    ``Q -> (taueps, tausig)`` mapping.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed model (the low-frequency / relaxed phase velocity;
            ``K = rho v**2`` is the relaxed modulus), of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        rho: The density model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        q: The quality factor ``Q`` model at the reference angular frequency
            ``omega0 = 2*pi*q_freq``, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See
            :ref:`batched-models`. A differentiable model input. Larger
            ``Q`` means weaker attenuation; ``Q = +inf`` gives the exact lossless
            acoustic equation, including in selected cells of an otherwise
            attenuating model.
        grid_spacing: Spatial grid cell size.
        dt: Time step interval. Deepwave is unit-agnostic: choose any single
            consistent system of units for ``v``, ``rho``, ``grid_spacing``,
            and ``dt`` (``q``/``q_freq`` have their own units documented
            above); see :doc:`/units`.
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
            canonical alternative to the flat ``receiver_locations*`` keywords.
            Each value is a locations object, or ``(locations, mask)`` to
            include an activity mask. The channels are ``p``, ``z``, ``y``,
            ``x`` (``z`` only in 3-D, ``y`` only in 2-D and 3-D). Giving one
            channel in both forms raises. Optional.
        source_amplitudes_pressure_rate: External pressure-equation source
            rate, in pressure per unit time (Pa/s in SI). Each internal step
            adds ``dt_internal * amplitude`` directly to pressure; no source
            is added to the relaxation memories. This is not a constitutive
            volume-injection rate. In the lossless limit, compare with the
            acoustic source ``source_amplitudes_volume_rate`` by multiplying
            it by ``K = rho * v**2``.
        source_amplitudes_z: Z-force-density source amplitudes (units of
            force/volume), as in :func:`deepwave.acoustic`. This channel exists
            only in 3-D.
        source_amplitudes_y: Y-force-density source amplitudes.
        source_amplitudes_x: X-force-density source amplitudes.
        source_locations_pressure_rate: Pressure-rate source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_z: Z-force source locations. This channel exists only
            in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_locations_y: Y-force source locations.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
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
        receiver_locations_p: Pressure receiver locations.
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
        pressure_0: Initial pressure (time t=0). Like every initial
            wavefield, it may be provided either at the shape of the
            simulation domain (the extracted model region plus the PML),
            which is the shape a previous call returns, or at the shape of
            the extracted model region itself, which Deepwave zero-pads for
            you; see the `wavefield_0` entry of :func:`deepwave.scalar`.
        vz_0: Initial z-velocity (time t=-0.5).
        vy_0: Initial y-velocity (time t=-0.5).
        vx_0: Initial x-velocity (time t=-0.5).
        initial_state: See :func:`deepwave.elastic`. The mapping keys are
            this propagator's own public state field names (`pressure_0`,
            `vz_0`, ...) and `auxiliary_state`. The GSLS memory variables
            are carried inside `auxiliary_state`, because how many there are
            depends on `q_n_mechanisms`, so an initial relaxation state is
            supplied by passing a previous call's state back, most simply as
            `initial_state=out`. Neither they (`r_mech0`, ...) nor the
            individual absorbing-boundary field names (`phi_y_0`, ...,
            `psi_x_0`) are accepted as keys.
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
            :func:`deepwave.acoustic`'s `boundary` (`"pml"`, `"free"`,
            `"rigid"`, or `"periodic"`; a non-`"pml"` face requires
            `pml_width == 0` on that face). The pressure image rule is
            identical to acoustic's (pressure odd about the surface for
            `"free"`, even about the half-cell wall for `"rigid"`); the SLS
            memory variable `r` is filled with the exact same image (no
            separate physics derivation is needed -- see docs/boundaries.rst,
            and Robertsson (1996) for the free-surface treatment these image
            rules follow). Validated on the eager (`backend="eager"`) and
            the default compiled C/CUDA backends, against analytic
            reflection coefficients and by gradient checking; the
            `"torch_compile"` mode
            shares the eager guard-fill code path but is not separately
            validated for `boundary=`.
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
        q_freq: Reference frequency at which ``Q`` is specified, in the same
            units as ``1 / dt``. **Required** when ``q_n_mechanisms == 1``
            (the default), and must be ``None`` when ``q_n_mechanisms > 1``,
            where the constant-``Q`` fit spans ``q_band`` instead and no
            single reference frequency applies. There is no default: ``Q``
            has no meaning without the frequency it was measured at, and it
            is never taken from ``pml_freq``, which tunes the absorbing
            boundary and is a different physical quantity (it is also
            inferred from the wavelet, which would have made the attenuation
            model depend on the source). ``omega0 = 2*pi*q_freq``. It must be
            a positive, finite number (a one-element tensor or a NumPy scalar
            is accepted); anything else raises.
        q_n_mechanisms: Number of generalised-standard-linear-solid (GSLS)
            relaxation mechanisms. Defaults to 1 (a single SLS mechanism;
            ``q_band`` must be ``None``). For ``q_n_mechanisms > 1``,
            ``q_band=(f1, f2)`` is required: ``q_n_mechanisms`` relaxation
            frequencies are chosen log-spaced across the band and their
            per-mechanism nonnegative weights are fit once in float64,
            independently of ``q``, to the weak-loss band target. Their
            scaling by ``1/q`` remains differentiable. The resulting passive
            modulus can have an actual effective Q different from ``q``;
            inspect it with :func:`deepwave.attenuation.response`. More
            mechanisms do not guarantee a better fit. Each mechanism gets its own memory
            variable, carried in ``auxiliary_state``, each with its own
            Crank-Nicolson update, summed into the pressure update.
            ``q_n_mechanisms > 1`` is supported on the eager backend
            (``backend="eager"``) and the default compiled C/CUDA
            backend (a ``DW_GSLS`` kernel variant; not combinable there
            with ``storage_compression=True`` or per-shot batched models --
            those combinations raise ``NotImplementedError`` and work on
            the eager backend); the ``"torch_compile"`` mode
            raises ``NotImplementedError``.
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
        order, are:

            - pressure_0: Final pressure wavefield.
            - vz_0: Final z-velocity wavefield (if ndim == 3).
            - vy_0: Final y-velocity wavefield (if ndim >= 2).
            - vx_0: Final x-velocity wavefield.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables and the GSLS relaxation
              memory variables (one per mechanism). Opaque: pass it back (as
              part of ``initial_state``) to continue the simulation; its
              contents are implementation state rather than physics. The field
              list therefore does not change with ``q_n_mechanisms``.
            - receiver_amplitudes_p: Recorded pressure data.
            - receiver_amplitudes_z: Recorded z-velocity data (if ndim == 3).
            - receiver_amplitudes_y: Recorded y-velocity data (if ndim >= 2).
            - receiver_amplitudes_x: Recorded x-velocity data.

        This positional order is easy to index incorrectly (``out[-1]`` is
        the x-component velocity receiver amplitudes, not the pressure
        data); prefer accessing the fields by name instead.
    """
    q_freq, q_n_mechanisms, q_band = _gsls.validate_configuration(
        q_freq, q_n_mechanisms, q_band
    )
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            viscoacoustic,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
            ),
        )
    backward_callback_needs_forward_wavefield = False
    mpml_fraction = 0.0
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
    _common.check_models([v, rho, q], ["v", "rho", "q"], "viscoacoustic")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend

    if q_n_mechanisms > 1:
        if backend_mode and backend_mode != "eager":
            raise DeepwaveNotImplementedError(
                "viscoacoustic: q_n_mechanisms > 1 (multi-mechanism GSLS Q) is "
                "supported on the eager Python backend and the default "
                'compiled C/CUDA backend; backend="torch_compile" '
                "does not implement it."
            )
        # Compiled (DW_GSLS) backend restriction: its L-times-bigger r_old
        # checkpoint blobs bypass the spatial-dims-based compression codec.
        if not backend_mode and storage_compression:
            raise DeepwaveNotImplementedError(
                "viscoacoustic: q_n_mechanisms > 1 does not support "
                "storage_compression=True on the default compiled "
                "C/CUDA backend (the per-mechanism checkpoint blobs "
                "bypass the spatial compression codec). Use "
                "storage_compression=False."
            )

    # Wavefield-argument mapping. The GSLS memory variables are neither
    # output fields nor keyword arguments: their count depends on
    # q_n_mechanisms, so they travel in auxiliary_state alongside the
    # phi_*/psi_* absorbing-boundary memory and are supplied by passing a
    # previous call's state back through initial_state. They still need a
    # slot here, under the public spelling the fold set is keyed by, because
    # this mapping is what the fold reads the tensors out of.
    r_arg_names = [f"r_mech{i}" for i in range(q_n_mechanisms)]
    wavefield_args: dict[str, Optional[torch.Tensor]] = {"pressure_0": pressure_0}
    for name in r_arg_names:
        wavefield_args[name] = None
    wavefield_args.update(
        {
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
        }
    )
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
        auxiliary_state_names=viscoacoustic_auxiliary_state_names(3, q_n_mechanisms),
        propagator=SPEC.name,
    )
    pressure_0 = resolved_wavefields["pressure_0"]
    r_list_0 = [resolved_wavefields[name] for name in r_arg_names]
    vz_0 = resolved_wavefields["vz_0"]
    vy_0 = resolved_wavefields["vy_0"]
    vx_0 = resolved_wavefields["vx_0"]
    phi_z_0 = resolved_wavefields["phi_z_0"]
    phi_y_0 = resolved_wavefields["phi_y_0"]
    phi_x_0 = resolved_wavefields["phi_x_0"]
    psi_z_0 = resolved_wavefields["psi_z_0"]
    psi_y_0 = resolved_wavefields["psi_y_0"]
    psi_x_0 = resolved_wavefields["psi_x_0"]

    _common.check_inputs_not_vmapped(
        pressure_0,
        *r_list_0,
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
        [v, rho, q],
        [pressure_0, *r_list_0],
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
        [vz_0, phi_z_0, psi_z_0],
        [vy_0, phi_y_0, psi_y_0],
        [vx_0, phi_x_0, psi_x_0],
    )
    if ndim < 3 and (
        source_locations_z is not None or receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            "viscoacoustic: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "z dimension should not be provided."
        )
    if ndim < 2 and (
        source_locations_y is not None or receiver_locations_y is not None
    ):
        raise DeepwaveValueError(
            "viscoacoustic: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "y dimension should not be provided."
        )

    # Prepare initial wavefields list: [pressure, r_0..r_{L-1}, velocities,
    # phi, psi].
    initial_wavefields: list[Optional[torch.Tensor]] = []
    initial_wavefields.append(pressure_0)
    initial_wavefields.extend(r_list_0)
    if ndim == 3:
        initial_wavefields.append(vz_0)
    if ndim >= 2:
        initial_wavefields.append(vy_0)
    initial_wavefields.append(vx_0)

    if ndim == 3:
        initial_wavefields.append(phi_z_0)
    if ndim >= 2:
        initial_wavefields.append(phi_y_0)
    initial_wavefields.append(phi_x_0)

    if ndim == 3:
        initial_wavefields.append(psi_z_0)
    if ndim >= 2:
        initial_wavefields.append(psi_y_0)
    initial_wavefields.append(psi_x_0)

    min_nonzero_model_vel, max_model_vel = _common.velocity_extremes(
        v, max_vel, "viscoacoustic"
    )
    if not _common.values_readable(q) and max_vel is None:
        raise _common.max_vel_unreadable_error("viscoacoustic")
    if _common.values_readable(v) and _common.values_readable(q):
        unrelaxed_speed = _gsls.unrelaxed_speed(
            v.detach(), q.detach(), q_n_mechanisms, q_band
        )
        max_model_vel = float(unrelaxed_speed.max())
        if _common.values_readable(rho):
            max_model_vel = _staggered_grid.pressure_stability_speed(
                unrelaxed_speed,
                rho,
                _common.set_accuracy(accuracy),
                _common.set_grid_spacing(grid_spacing, ndim, "viscoacoustic"),
                courant_number,
            )
        if max_vel is not None and abs(float(max_vel)) < max_model_vel:
            raise DeepwaveValueError(
                "viscoacoustic: max_vel must bound the unrelaxed wave speed "
                "and the density-weighted spatial operator "
                f"(at least {max_model_vel:.6g} for this model)."
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
    source_amplitudes_list.insert(0, source_amplitudes_pressure_rate)
    source_locations_list_in.insert(0, source_locations_pressure_rate)
    receiver_locations_list_in.insert(0, receiver_locations_p)

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
        "viscoacoustic",
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
                    f"viscoacoustic: {name}_locations_"
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
        [v, rho, q],
        ["replicate", "replicate", "replicate"],
        ["v", "rho", "q"],
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
    del (
        v,
        rho,
        q,
        source_amplitudes_pressure_rate,
        source_locations_pressure_rate,
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

    q_freq_resolved = _resolve_q_freq(q_freq)
    omega0 = 2.0 * math.pi * q_freq_resolved

    models = prepare_models(
        models[0], models[1], models[2], dt, omega0, q_n_mechanisms, q_band
    )
    if (
        q_n_mechanisms > 1
        and not backend_mode
        and models[0].ndim == ndim + 1
        and models[0].shape[0] > 1
    ):
        # m1 is batched exactly when any of v/rho/q is per-shot batched (it
        # is derived from all three), so it is the single reliable probe.
        raise DeepwaveNotImplementedError(
            "viscoacoustic: q_n_mechanisms > 1 does not support per-shot "
            "batched models (v/rho/q with a leading shot dimension > 1) on "
            'the default compiled C/CUDA backend. Use backend="eager" for '
            "that combination."
        )

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "viscoacoustic")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "viscoacoustic")
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
        fold_names=viscoacoustic_auxiliary_state_names(ndim, q_n_mechanisms),
    )

    # Pressure-rate amplitudes already have pressure/time units, so scale them
    # by dt only. Velocity sources are force/volume and retain buoyancy
    # scaling. The first source is pressure; the remaining sources correspond
    # to the buoyancy models at indices 3.. .
    scaling_models = [None, *models[3:]]
    model_shape = models[0].shape[-ndim:]
    for i, (src_amp, src_loc, model) in enumerate(
        zip(source_amplitudes_out, sources_i, scaling_models)
    ):
        if src_amp.numel() > 0:
            source_amplitudes_out[i] = (
                src_amp
                * dt
                * (
                    1.0
                    if model is None
                    else _common.gather_model_at_sources(
                        model, src_loc, model_shape, n_shots
                    )
                )
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
        names=viscoacoustic_output_names(ndim, q_n_mechanisms),
        state_names=viscoacoustic_public_wavefield_names(ndim, q_n_mechanisms),
        fold_names=viscoacoustic_auxiliary_state_names(ndim, q_n_mechanisms),
        python_backend=backend_mode,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
    )

    outputs = list(
        viscoacoustic_func(
            backend_mode,
            q_n_mechanisms,
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

    # Reorder the state tensors into the public order (the pressure and the
    # velocities, then the GSLS memory, then the absorbing-boundary memory).
    # The SAME index list names them in viscoacoustic_public_wavefield_names,
    # so the two cannot disagree.
    n_state = len(viscoacoustic_wavefield_names(ndim, q_n_mechanisms))
    outputs[:n_state] = [
        outputs[i] for i in viscoacoustic_public_wavefield_order(ndim, q_n_mechanisms)
    ]

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
        names=viscoacoustic_output_names(ndim, q_n_mechanisms),
        state_names=viscoacoustic_public_wavefield_names(ndim, q_n_mechanisms),
        fold_names=viscoacoustic_auxiliary_state_names(ndim, q_n_mechanisms),
        check_output_finite=check_output_finite,
    )


def zero_edges_and_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    interior: bool = True,
    n_mechanisms: int = 1,
) -> None:
    """Zeros the edges and/or interiors of wavefields.

    Wavefields order: ``[pressure, r_0..r_{n_mechanisms-1}, vz, vy, vx,
    phi..., psi...]``. The pressure and the ``n_mechanisms`` memory
    variables ``r_0``.. (integer / pressure nodes) need no edge zeroing; the
    velocities are on the half grid, and the PML memory variables
    ``phi``/``psi`` are co-located with the pressure / velocity nodes
    respectively (mirroring ``acoustic.zero_edges_and_interiors``,
    shifted by ``n_mechanisms`` for the memory-variable fields). Builds the
    viscoacoustic-specific tables and delegates the shared zeroing pass to
    :func:`deepwave._staggered_grid.zero_edges_and_interiors`.
    """
    num_vars = len(wavefields)
    half_grid = [[False] * ndim for _ in range(num_vars)]
    pml_edge_idx: list[list[int]] = [[] * ndim]

    # Index layout (relative to acoustic, shifted by n_mechanisms for the
    # memory variables): p(0), r_0..r_{L-1}(1..L), velocities(v_start..
    # v_start+ndim-1), phi(phi_start..), psi(psi_start..).
    v_start = 1 + n_mechanisms
    phi_start = v_start + ndim
    psi_start = phi_start + ndim
    if ndim == 3:
        half_grid[v_start + 0][0] = True
        half_grid[psi_start + 0][0] = True
        half_grid[v_start + 1][1] = True
        half_grid[psi_start + 1][1] = True
        half_grid[v_start + 2][2] = True
        half_grid[psi_start + 2][2] = True
        pml_edge_idx = [
            [phi_start + 0, psi_start + 0],
            [phi_start + 1, psi_start + 1],
            [phi_start + 2, psi_start + 2],
        ]
    elif ndim == 2:
        half_grid[v_start + 0][0] = True
        half_grid[psi_start + 0][0] = True
        half_grid[v_start + 1][1] = True
        half_grid[psi_start + 1][1] = True
        pml_edge_idx = [[phi_start + 0, psi_start + 0], [phi_start + 1, psi_start + 1]]
    elif ndim == 1:
        half_grid[v_start][0] = True
        half_grid[psi_start][0] = True
        pml_edge_idx = [[phi_start, psi_start]]

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


def _stack_gsls_memory_variables(
    wavefields: list[torch.Tensor], n_mechanisms: int
) -> None:
    """Replaces the ``n_mechanisms`` padded memory variables with stack views.

    The compiled ``DW_GSLS`` kernel reads the memory variables through the
    single historical ``rmem`` pointer as one contiguous
    ``(n_mechanisms, n_shots, *spatial)`` buffer (a bigger buffer, not new
    ABI pointers, since ``n_mechanisms`` is an unbounded runtime value). The
    engine's generic per-wavefield machinery (padding, zeroing, output
    slicing, callbacks) keeps operating on per-mechanism tensors, so this
    stacks the already-padded-and-zeroed tensors into one contiguous buffer
    and replaces the list entries (indices ``1 .. n_mechanisms``) with views
    into it in place -- element 1's ``data_ptr()`` is then the stacked
    buffer's base pointer, and every later read/write through the views (the
    backend call, callbacks, output slicing) sees the same storage.
    """
    if n_mechanisms <= 1:
        return
    r_stack = torch.stack(wavefields[1 : 1 + n_mechanisms])
    for mech in range(n_mechanisms):
        wavefields[1 + mech] = r_stack[mech]


def _gsls_flags(
    state: "_engine.BackendState",
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """The per-model-group (requires_grad, batched) flags for the GSLS call.

    Mirrors the canonical call's ``c_model_flag_indices``-driven flags, with
    the (L, batch, *spatial)-shaped ``ar`` handled explicitly: its batched
    flag is always False (per-shot batched models are rejected upstream for
    the compiled GSLS path), and its requires-grad flag is read directly from
    the tensor rather than a representative slice (slicing inside the
    grad-disabled autograd ``Function`` body would drop the flag).
    """
    m1 = state.models[0]
    ar = state.models[2]
    b = state.models[-1]
    ndim = state.ndim
    return (
        state.storage_grad_flag(m1.requires_grad),
        state.storage_grad_flag(ar.requires_grad),
        state.storage_grad_flag(b.requires_grad),
        m1.ndim == ndim + 1 and m1.shape[0] > 1,
        False,
        b.ndim == ndim + 1 and b.shape[0] > 1,
    )


def _gsls_wavefield_ptrs(state: "_engine.BackendState", n_mechanisms: int) -> list[int]:
    """The wavefield pointers for the GSLS call: one ``rmem`` base pointer.

    ``state.wavefields`` holds ``[p, r_0 .. r_{L-1}, v..., phi..., psi...]``
    with the ``r_l`` entries being views into one contiguous stacked buffer
    (see :func:`_stack_gsls_memory_variables`); the kernel receives the
    historical pointer list ``[p, rmem, v..., phi..., psi...]`` where
    ``rmem`` is the stack's base pointer (``r_0``'s ``data_ptr``).
    """
    wf = state.wavefields
    return [
        wf[0].data_ptr(),
        wf[1].data_ptr(),
        *[w.data_ptr() for w in wf[1 + n_mechanisms :]],
    ]


def _gsls_storage_ptrs(
    state: "_engine.BackendState", gsls_storage_manager: "StorageManager"
) -> list[Any]:
    """The checkpoint-store pointers for the GSLS call.

    The main storage manager's ``ar`` slot (pointer positions 10..14: the
    third model's five-pointer group) is allocated empty for the GSLS path
    (see the ``storage_flags`` override in ``ViscoacousticForwardFunc.
    forward``) and replaced by the second, ``(L, *model_shape)``-shot-shaped
    manager's group, so ``ar``'s per-time-step ``r_old`` checkpoint buffer is
    ``L``-times bigger behind the same ABI pointer slot.
    """
    main = list(state.storage_manager.storage_ptrs)
    return main[:10] + list(gsls_storage_manager.storage_ptrs) + main[15:]


def _viscoacoustic_gsls_make_forward_call(
    n_mechanisms: int, gsls_storage_manager: "StorageManager"
) -> Any:
    """Builds the ``make_call`` closure for the compiled ``DW_GSLS`` forward.

    Mirrors :func:`deepwave._engine.canonical_forward_call` with the GSLS
    substitutions: the stacked ``rmem`` pointer, the substituted ``ar``
    checkpoint-store pointer group, explicit per-model-group flags, and the
    one trailing ``n_mechanisms`` argument the ``_gsls`` compiled symbols
    take (see viscoacoustic.c/viscoacoustic.cu's ``FUNC(forward)``).
    """

    def make_call(state: "_engine.BackendState", step: int, step_nt: int) -> int:
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in state.models],
                *[a.data_ptr() for a in state.source_amplitudes],
                *_gsls_wavefield_ptrs(state, n_mechanisms),
                *_gsls_storage_ptrs(state, gsls_storage_manager),
                *[a.data_ptr() for a in state.receiver_amplitudes],
                *[p.data_ptr() for p in state.pml_profiles],
                *[loc.data_ptr() for loc in state.sources_i],
                *[loc.data_ptr() for loc in state.receivers_i],
                *_engine.call_grid_args(state, step_nt),
                *state.n_sources_per_shot,
                *state.n_receivers_per_shot,
                *_engine.call_storage_meta_args(state),
                *_gsls_flags(state),
                *_engine.call_pml_tail_args(state, step),
                n_mechanisms,
            ),
        )

    return make_call


def _viscoacoustic_gsls_make_backward_call(
    n_mechanisms: int, gsls_storage_manager: "StorageManager"
) -> Any:
    """Builds the ``make_call`` closure for the compiled ``DW_GSLS`` backward.

    Mirrors :func:`deepwave._engine.canonical_backward_call` with the same
    GSLS substitutions as :func:`_viscoacoustic_gsls_make_forward_call`.
    """

    def make_call(state: "_engine.BackendState", step: int, step_nt: int) -> int:
        n_sources_for_grad = [
            count if grad.numel() > 0 else 0
            for count, grad in zip(state.n_sources_per_shot, state.grad_sources)
        ]
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in state.models],
                *[g.data_ptr() for g in state.grad_receivers],
                *_gsls_wavefield_ptrs(state, n_mechanisms),
                *[w.data_ptr() for w in state.aux_wavefields],
                *_gsls_storage_ptrs(state, gsls_storage_manager),
                *[g.data_ptr() for g in state.grad_sources],
                *[g.data_ptr() for g in state.grad_models],
                *state.grad_models_tmp_ptr,
                *[p.data_ptr() for p in state.pml_profiles],
                *[loc.data_ptr() for loc in state.sources_i],
                *[loc.data_ptr() for loc in state.receivers_i],
                *_engine.call_grid_args(state, step_nt),
                *n_sources_for_grad,
                *state.n_receivers_per_shot,
                *_engine.call_storage_meta_args(state),
                *_gsls_flags(state),
                *_engine.call_pml_tail_args(state, step),
                n_mechanisms,
            ),
        )

    return make_call


class ViscoacousticForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of viscoacoustic propagation."""

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
        boundary: list[int],
        n_mechanisms: int,
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the viscoacoustic wave equation.

        Additional args:
            boundary: Per-face first-class boundary codes (see
                ``deepwave._common.set_boundary``); ``[BC_PML] * 2 * ndim``
                when ``boundary=None`` was requested, which keeps the
                compiled call byte-identical (the C/CUDA ``has_bc`` guard
                skips the fill for all-PML codes -- see viscoacoustic.c).

            n_mechanisms: The GSLS relaxation-mechanism count. 1 (the
                default single-SLS mechanism) keeps the historical call
                completely unchanged; > 1 dispatches to the ``DW_GSLS``
                compiled kernel variant (``backend_extra="_gsls"``), with
                the ``n_mechanisms`` memory variables stacked behind the
                single historical ``rmem`` pointer, ``c2``/``ar`` as
                per-mechanism-valued single-named models, and a second,
                ``(n_mechanisms, *model_shape)``-shot-shaped
                :class:`deepwave._storage.StorageManager` (threaded through
                the existing ``raw_forward_storage_manager`` hook) for the
                ``n_mechanisms``-times-bigger per-time-step ``r_old``
                checkpoint storage.
        """
        gsls = n_mechanisms > 1

        # Second, independent checkpoint manager for the GSLS ar (r_old)
        # storage: shot-shaped (n_mechanisms, *model_shape) so every byte
        # size/offset it reports is n_mechanisms-times the main manager's.
        # Allocated here (not inside the engine) and threaded to the
        # backward pass via ForwardSavedState.raw_forward_storage_manager.
        gsls_storage_manager: Optional[StorageManager] = None
        if gsls:
            ndim = len(grid_spacing)
            model_rep = args[0]
            ar_model = args[2]
            gsls_storage_manager = StorageManager(
                (n_mechanisms, *model_rep.shape[-ndim:]),
                model_rep.dtype,
                n_shots,
                nt,
                step_ratio,
                _engine.resolve_storage_mode(storage_mode_str, model_rep.device),
                storage_compression,
                storage_path,
                model_rep.device,
                model_rep.is_cuda,
            )
            gsls_storage_manager.allocate(ar_model.requires_grad)

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # One slot per model as usual, except the GSLS path allocates the
            # ar slot empty in the MAIN manager (its r_old checkpoints live
            # in the n_mechanisms-times-bigger second manager instead -- see
            # _gsls_storage_ptrs).
            flags = [m.requires_grad for m in models]
            if gsls:
                flags[2] = False
            return flags

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            zero_edges_and_interiors(
                wavefields,
                ndim,
                fd_pad,
                fd_pad_list,
                pml_width,
                n_mechanisms=n_mechanisms,
            )
            _stack_gsls_memory_variables(wavefields, n_mechanisms)

        if gsls:
            make_call = _viscoacoustic_gsls_make_forward_call(
                n_mechanisms,
                cast("StorageManager", gsls_storage_manager),
            )
        else:
            make_call = _engine.canonical_forward_call

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
            make_call=make_call,
            backend_extra="_gsls" if gsls else "",
            boundary=boundary,
            raw_forward_storage_manager=gsls_storage_manager,
            wavefield_names_override=viscoacoustic_wavefield_names(
                len(grid_spacing), n_mechanisms
            )
            if gsls
            else None,
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
        # ``_boundaries.mirror_pad_viscoacoustic_models``).
        # ``n_mechanisms`` follows ``boundary`` as the last leading input;
        # the backward pass needs it to dispatch/pass the same GSLS geometry.
        ctx.n_mechanisms = inputs[_N_LEAD - 1]
        ctx.boundary = inputs[_N_LEAD - 2]

    vmap = staticmethod(_engine.unsupported_vmap)
    jvp = staticmethod(_engine.unsupported_jvp)

    @staticmethod
    @_engine.no_second_order("viscoacoustic")
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Computes gradients during the backward pass.

        Like ``acoustic.AcousticForwardFunc``, this calls the
        non-differentiable C/CUDA backend directly; there is no analytic
        Hessian-vector product, so second-order derivatives raise (see
        :func:`deepwave._engine.no_second_order`).
        """
        args = args[:-1]
        n_mechanisms = int(getattr(ctx, "n_mechanisms", 1))
        gsls = n_mechanisms > 1

        def grad_model_batched(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # Flags follow c_model_flag_indices = [0 (m1, covers m1+c2), 2 (ar),
            # -1 (buoyancy)], one boolean per kernel model. m1 and c2 share
            # their batched state; buoyancies share theirs. For the GSLS path
            # (n_mechanisms > 1), c2/ar carry a leading mechanism axis
            # ((L, batch, *spatial)), so the ndim + 1 shape test would
            # misread the mechanism axis as a shot batch; per-shot batched
            # models are rejected upstream on that path, so both are
            # unbatched by construction.
            m1_batched = models[0].ndim == ndim + 1 and models[0].shape[0] > 1
            if gsls:
                ar_batched = False
            else:
                ar_batched = models[2].ndim == ndim + 1 and models[2].shape[0] > 1
            b_batched = models[3].ndim == ndim + 1 and models[3].shape[0] > 1
            return [m1_batched, m1_batched, ar_batched] + [b_batched] * ndim

        def make_aux_wavefields(
            grad_wavefields: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            return [torch.zeros_like(grad_wavefields[0]) for _ in range(ndim)]

        def swap(state: _engine.BackendState) -> None:
            ndim = state.ndim
            state.wavefields[-ndim:], state.aux_wavefields[:] = (
                state.aux_wavefields[:],
                state.wavefields[-ndim:],
            )

        def zero_wavefields(
            wavefields: list[torch.Tensor],
            ndim: int,
            fd_pad: int,
            fd_pad_list: list[int],
            pml_width: list[int],
        ) -> None:
            zero_edges_and_interiors(
                wavefields,
                ndim,
                fd_pad,
                fd_pad_list,
                pml_width,
                n_mechanisms=n_mechanisms,
            )
            _stack_gsls_memory_variables(wavefields, n_mechanisms)

        boundary = cast("list[int]", getattr(ctx, "boundary", []))

        def model_transform(models: list[torch.Tensor]) -> list[torch.Tensor]:
            # The compiled adjoint's B_EFF combination reads m1/c2 at
            # neighbour offsets along every axis (like acoustic's K) and each
            # buoyancy at neighbour offsets along its own axis (like
            # acoustic's buoyancy_d); ar is read only at the centre point.
            # See _boundaries.mirror_pad_viscoacoustic_models.
            return _boundaries.mirror_pad_viscoacoustic_models(
                models, boundary, ctx.accuracy // 2
            )

        if gsls:
            make_call = _viscoacoustic_gsls_make_backward_call(
                n_mechanisms,
                cast(
                    "StorageManager",
                    ctx.raw_forward_storage_manager,
                ),
            )
        else:
            make_call = _engine.canonical_backward_call

        return _engine.run_backward(
            ctx,
            SPEC,
            args,
            grad_model_batched=grad_model_batched,
            make_aux_wavefields=make_aux_wavefields,
            zero_wavefields=zero_wavefields,
            make_call=make_call,
            swap=swap,
            n_leading=_N_LEAD,
            boundary=boundary,
            model_transform=model_transform,
            wavefield_names=viscoacoustic_wavefield_names(
                len(ctx.grid_spacing), n_mechanisms
            )
            if gsls
            else None,
        )


# Number of leading non-tensor arguments of
# ``ViscoacousticForwardFunc.forward`` (the shared config block plus the
# trailing ``boundary``/``n_mechanisms`` pair), all of which receive
# ``None`` gradients in ``backward`` (this propagator has no custom
# gradient-count logic to keep in sync -- its own ``backward`` above
# delegates entirely to ``_engine.run_backward(..., n_leading=_N_LEAD)``,
# which builds its returned tuple's leading ``None``s from this same value,
# so adding ``boundary`` here needed no separate manual fix; mirrors
# acoustic_vti's identical finding).
_N_LEAD = _engine.leading_arg_count(ViscoacousticForwardFunc.forward)


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
    """Updates the velocity wavefields and PML memory variables.

    Identical to the acoustic velocity update; the buoyancy models are the
    trailing ``ndim`` entries of ``models`` (``[m1, c2, ar, buoyancy...]``).
    """
    ndim = len(grid_spacing)
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    buoyancy = models[3:]

    for dim in range(ndim):
        gp = _staggered_grid.diff1h(p, dim, accuracy, 1 / grid_spacing[dim], ndim)
        psi[dim] = ah[dim] * psi[dim] + bh[dim] * gp
        velocities[dim] = velocities[dim] - dt * buoyancy[dim] * (gp + psi[dim])

    return velocities, psi


def update_pressure(
    models: list[torch.Tensor],
    p: torch.Tensor,
    r_list: list[torch.Tensor],
    velocities: list[torch.Tensor],
    phi: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Updates pressure ``p``, the memory variables ``r_list`` and PML ``phi``.

    Implements the Crank-Nicolson GSLS update (see the module docstring),
    generalised to ``len(r_list)`` independent relaxation mechanisms::

        theta       = div(v)  (PML-augmented)
        r_new_l     = ar_l * r_l - c2_l * theta          (each mechanism l)
        p_new       = p - dt * m1 * theta
                        - (dt/2) * sum_l (r_l + r_new_l)

    For a single mechanism (``len(r_list) == 1``) ``c2``/``ar`` (``models[1]``/
    ``models[2]``) are plain per-cell fields, exactly as before (this branch
    is bit-identical to the pre-multi-mechanism code); for more than one
    mechanism they carry a leading mechanism axis (see
    :func:`prepare_models`), indexed here per mechanism.
    """
    ndim = len(grid_spacing)
    a = pml_profiles[::4]
    b = pml_profiles[1::4]

    m1 = models[0]
    c2 = models[1]
    ar = models[2]

    div_v_sum = torch.zeros_like(p)
    for dim in range(ndim):
        dv = _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        phi[dim] = a[dim] * phi[dim] + b[dim] * dv
        div_v_sum = div_v_sum + dv + phi[dim]

    n_mech = len(r_list)
    if n_mech == 1:
        r_new = ar * r_list[0] - c2 * div_v_sum
        p = p - dt * m1 * div_v_sum - (dt / 2) * (r_list[0] + r_new)
        return p, [r_new], phi

    new_r_list: list[torch.Tensor] = []
    sum_r = torch.zeros_like(p)
    for i in range(n_mech):
        r_old = r_list[i]
        r_new_i = ar[i] * r_old - c2[i] * div_v_sum
        new_r_list.append(r_new_i)
        sum_r = sum_r + r_old + r_new_i
    p = p - dt * m1 * div_v_sum - (dt / 2) * sum_r

    return p, new_r_list, phi


_update_velocities_cache = _engine.StepCache(update_velocities)
_update_velocities_opt = update_velocities
_update_pressure_cache = _engine.StepCache(update_pressure)
_update_pressure_opt = update_pressure

# Named offsets into the leading (pre-tensor-block) args viscoacoustic_func
# receives from its call site (see
# _backend_utils.PROPAGATOR_REGISTRY["viscoacoustic"].leading_args): the
# index of "boundary", and the index one past the last leading arg -- i.e.
# where the model/source/wavefield tensor block starts (before the
# ``q_n_mechanisms`` insertion below shifts it further).
_BOUNDARY_IDX = _backend_utils.leading_arg_offset("viscoacoustic", "boundary")
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["viscoacoustic"].leading_args)


def viscoacoustic_python(
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
    """Python (eager/compiled) backend for viscoacoustic propagation.

    ``boundary`` is the per-face first-class boundary code list;
    ``BC_PML`` (0) faces are untouched (a no-op fill), so
    ``boundary=None`` (mapped upstream to all-``BC_PML``) is bit-identical to
    the historical behaviour. The guard-ring fill reuses
    ``_boundaries.fill_acoustic_pressure_boundary``/
    ``fill_acoustic_velocity_boundary`` unchanged for ``p``/the velocities
    (viscoacoustic's staggered pressure-velocity grid and velocity equation
    are identical to acoustic's); the SLS memory variable ``r`` is filled
    with the SAME pressure fill (not a separately-derived rule -- see
    viscoacoustic.c's ``FUNC(fill_p_bc_line)`` comment and
    ``ROBERTSSON_FS_SUMMARY.md`` Sec 1.3 for the citation).

    ``backend`` is the resolved Python backend name (``"eager"``
    or ``"compile"``), already selected by the caller; used only to name it
    if an unsupported option is rejected.
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

    num_models = 3 + ndim
    num_source_types = ndim + 1
    num_receiver_types = ndim + 1

    # Pop from the FRONT (models, source_amplitudes, sources_i, receivers_i
    # -- all fixed-size given ndim); whatever remains is the wavefields block
    # (p, r_0..r_{n_mechanisms-1}, v, phi, psi), whose length depends on
    # n_mechanisms and so is inferred from what is left rather than assumed.
    models = args_list[:num_models]
    del args_list[:num_models]
    source_amplitudes = args_list[:num_source_types]
    del args_list[:num_source_types]
    sources_i = args_list[:num_source_types]
    del args_list[:num_source_types]
    receivers_i = args_list[:num_receiver_types]
    del args_list[:num_receiver_types]
    wavefields = args_list
    n_mechanisms = len(wavefields) - 3 * ndim - 1

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

    zero_edges_and_interiors(
        wavefields,
        ndim,
        fd_pad,
        fd_pad_list,
        pml_width,
        n_mechanisms=n_mechanisms,
    )

    p = wavefields[0]
    r_list = wavefields[1 : 1 + n_mechanisms]
    velocities = wavefields[1 + n_mechanisms : 1 + n_mechanisms + ndim]
    phi = wavefields[1 + n_mechanisms + ndim : 1 + n_mechanisms + 2 * ndim]
    psi = wavefields[1 + n_mechanisms + 2 * ndim :]
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

    masked_sources = []
    for i in range(num_source_types):
        sources_i_masked, src_mask = _common.mask_locations(sources_i[i])
        src_amp_masked = _common.mask_source_amplitudes(source_amplitudes[i], src_mask)
        masked_sources.append((sources_i_masked, src_amp_masked, i))

    masked_receivers = []
    for i, locs in enumerate(receivers_i):
        receivers_i_masked, recv_mask = _common.mask_locations(locs)
        masked_receivers.append((receivers_i_masked, recv_mask, i))

    wf_names = viscoacoustic_wavefield_names(ndim, n_mechanisms)
    callback_wavefields = {}
    model_names = viscoacoustic_model_names(ndim)
    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            current_wavefields = [p, *r_list, *velocities, *phi, *psi]
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

            # Refill the pressure and SLS memory-variable guard rings (read
            # by the velocity update); this also holds free-surface
            # pressure/memory-variable nodes to zero before recording. The
            # memory variable ``r`` uses the same fill as ``p`` (see the
            # module-level note above this function).
            if has_boundary:
                p = _boundaries.fill_acoustic_pressure_boundary(p, boundary, fd_pad)
                r_list = [
                    _boundaries.fill_acoustic_pressure_boundary(r, boundary, fd_pad)
                    for r in r_list
                ]

            for i, (receivers_i_masked, _, target_idx) in enumerate(masked_receivers):
                if receivers_i_masked.numel() > 0:
                    val = torch.empty(0)
                    if target_idx == 0:
                        val = p.view(-1, flat_shape).gather(1, receivers_i_masked)
                    elif target_idx > 0 and target_idx <= ndim:
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

            p, r_list, phi = _update_pressure_opt(
                models,
                p,
                r_list,
                velocities,
                phi,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

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
    # pressure/memory-variable nodes held to zero, guard rings imaged) so a
    # continuation run reproduces the in-loop state (mirrors acoustic).
    if has_boundary:
        p = _boundaries.fill_acoustic_pressure_boundary(p, boundary, fd_pad)
        r_list = [
            _boundaries.fill_acoustic_pressure_boundary(r, boundary, fd_pad)
            for r in r_list
        ]
        velocities = _boundaries.fill_acoustic_velocity_boundary(
            list(velocities), boundary, fd_pad
        )

    wavefields = [p, *r_list, *velocities, *phi, *psi]
    s = (slice(None), *(slice(fd_pad, shape - (fd_pad - 1)) for shape in shape))
    return (*[w[s] for w in wavefields], *receiver_amplitudes)


def viscoacoustic_func(
    backend: Optional[Literal["eager", "compile"]] = None,
    q_n_mechanisms: int = 1,
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """Dispatch the viscoacoustic forward pass to the eager or compiled backend.

    Args:
        backend: ``"eager"``, ``"compile"``, or ``None``
            (compiled C/CUDA, the default).
        q_n_mechanisms: The GSLS relaxation-mechanism count. The Python
            backends infer it from the wavefield count and ignore this
            parameter; the compiled path threads it as a new leading argument
            into :class:`ViscoacousticForwardFunc` (1, the default, keeps the
            compiled call dispatching to the unchanged single-mechanism
            symbols; > 1 dispatches to the ``DW_GSLS`` kernel variant via
            ``backend_extra="_gsls"``).
        *args: The flat argument block forwarded to the selected backend.

    Returns:
        The forward-pass outputs (final wavefields then receiver amplitudes).
    """
    global _update_velocities_opt, _update_pressure_opt

    if backend:
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        _update_velocities_opt = _update_velocities_cache.resolve(backend)
        _update_pressure_opt = _update_pressure_cache.resolve(backend)
        return viscoacoustic_python(*args, backend=backend)

    # ``boundary`` is at ``_BOUNDARY_IDX``, right after ``pml_width``
    # (matching :func:`viscoacoustic_python`'s layout above). The compiled
    # backend threads the first-class boundary codes through the ``bc`` ABI
    # (mirrors acoustic_func's identical reshuffle), so move ``boundary`` to
    # the *last* leading (non-tensor) argument before the
    # model/source/wavefield tensors.
    boundary = args[_BOUNDARY_IDX]
    args = (
        args[:_BOUNDARY_IDX]
        + args[_BOUNDARY_IDX + 1 : _N_LEADING_ARGS]
        + (boundary,)
        + args[_N_LEADING_ARGS:]
    )

    # ``q_n_mechanisms`` is inserted as the last leading (non-tensor)
    # argument, immediately after ``boundary`` (now the last leading arg)
    # and before the model/source/wavefield tensors, matching the new
    # ``n_mechanisms`` parameter :meth:`ViscoacousticForwardFunc.forward`
    # gained right after its own ``boundary`` parameter.
    args = (
        *args[:_N_LEADING_ARGS],
        q_n_mechanisms,
        *args[_N_LEADING_ARGS:],
    )

    func = ViscoacousticForwardFunc.apply
    return cast(
        "tuple[torch.Tensor, ...]",
        func(*args)[:-1],
    )


_common.document_propagator_module(Viscoacoustic, viscoacoustic)

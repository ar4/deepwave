"""Variable-density acoustic Born (linearised) wave propagation module.

Implements first-order Born (linearised) forward modelling of the
variable-density acoustic wave equation described in :mod:`deepwave.acoustic`.
A background field ``(p, v_d)`` and a scattered field ``(dp, dv_d)`` are
propagated together on the same staggered grid; the scattered field is driven
by virtual sources built from the scattering perturbations ``dK`` (bulk
modulus) and ``dB_d`` (buoyancy, one per spatial dimension), which the
front-end derives from user-facing velocity/density perturbations ``dv``,
``drho`` via the same differentiable ``prepare_models`` composition acoustic
itself uses for ``v, rho -> K, B``.

See ``docs/acoustic_born_derivation.md`` for the full derivation (the
linearised system, the adjoint, and the second-order/HVP structure).

**Backends.** The default, ``backend="auto"``, runs the compiled
C/CUDA coupled forward (``acoustic_born.c``) with the compiled CPU adjoint
kernels: a scattered-only adjoint (``backward_sc``) for pure LSRTM
(gradients w.r.t. ``dv``/``drho`` and the scattered-side initial
wavefields), and the full two-field adjoint (``backward``) whenever a
background-side quantity needs a gradient -- because a single shared source
amplitude drives both fields and the scattered output depends on the
background ``v``/``rho`` both directly and through the field's propagation
into the Born coupling. ``backend="eager"`` selects a pure-PyTorch
coupled Born driver that runs the same time-stepping under native autograd
(:func:`acoustic_born_python`); it is slower but composes with ``torch.func``
transforms (``vmap`` over the velocity model, forward-mode ``jvp``), which
the compiled ctypes backend cannot.

**Differentiable inputs** (both backends): ``dv``, ``drho``, the background
``v``/``rho``, the source amplitudes (of any component), and both the
scattered- and background-side initial wavefields (all validated against
directional finite differences and ``gradcheck``). The base
:mod:`deepwave.acoustic` forward-mode (``jvp``) and Hessian-vector-product
rules are built on this module's cores.

Both backends support float/Hicks source-receiver interpolation
(``location_interpolation``), as in :func:`deepwave.acoustic`. Forward
callbacks are available on both backends; backward callbacks and the
cpu/disk/none storage modes are available on the compiled backend only.
"""

import importlib
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
from deepwave import _boundaries, _common, _engine, _staggered_grid, _storage
from deepwave._engine import scattered_state_name as _scattered_name
from deepwave._exceptions import DeepwaveRuntimeError, DeepwaveValueError
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

# The ``deepwave.acoustic`` attribute on the package is shadowed by the
# re-exported function of the same name in ``deepwave.__init__`` (see
# ``deepwave.scalar._scalar_forward_jvp`` for the same workaround); fetch the
# submodule from ``sys.modules`` via importlib instead of ``import ... as``
# (which resolves via attribute access and would grab the function).
_acoustic_module = importlib.import_module("deepwave.acoustic")


def _dim_names(ndim: int) -> list[str]:
    """Returns the spatial dimension names (slowest to fastest) for ``ndim``."""
    return ["z", "y", "x"][-ndim:]


def acoustic_born_model_names(ndim: int) -> list[str]:
    """Names of the acoustic Born models: background then scattering."""
    dims = _dim_names(ndim)
    return ["K"] + [f"B{d}" for d in dims] + ["dK"] + [f"dB{d}" for d in dims]


# The user-facing model arguments, in the order the public function takes
# them: the two background models, then the two perturbations.
USER_MODEL_NAMES = ("v", "rho", "dv", "drho")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(v, rho, dv, drho)``; the result is in
    :func:`acoustic_born_model_names` order, which is the background models
    (``K`` and one buoyancy per axis) followed by the scattered ones
    (``dK`` and one perturbed buoyancy per axis).

    The background bulk modulus ``K = rho * v**2`` carries a gradient if
    either ``v`` or ``rho`` does; the background buoyancies come from ``rho``
    alone. The scattered bulk modulus is
    ``dK = 2 * rho * v * dv + v**2 * drho``, so it depends on all four, and
    the scattered buoyancies on ``rho`` and ``drho``.
    """
    v, rho, dv, drho = model_requires_grad
    return [v or rho] + [rho] * ndim + [v or rho or dv or drho] + [rho or drho] * ndim


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    Not one slot per internal model requiring a gradient: the two halves are
    cross-wired, because the *background* wavefield history is what every
    model's gradient is formed from, while the *scattered* history is needed
    only by the background models' gradients. So asking for a perturbation
    gradient alone still stores the background history, and asking for a
    background gradient stores both.

    Measured against the engine over every combination of the four models at
    ndim 1, 2 and 3; the anti-drift test in ``tests/test_ergonomics.py``
    re-checks it.
    """
    v, rho, dv, drho = model_requires_grad
    background = v or rho
    return [background or dv or drho] * (ndim + 1) + [background] * (ndim + 1)


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models.

    A left inverse of :func:`internal_model_grad_flags` for counting
    purposes. ``K`` requiring a gradient is reported as ``v`` requiring one
    and ``dK`` as ``dv``, since neither can be told apart from its partner
    given only the internal model; that over-reports rather than guesses, and
    gives back the same slot count.
    """
    return [
        models[0].requires_grad,
        models[1].requires_grad,
        models[ndim + 1].requires_grad,
        models[ndim + 2].requires_grad,
    ]


def acoustic_born_wavefield_names(ndim: int) -> list[str]:
    """Names of the acoustic Born wavefields, in backend order.

    The background wavefields (in the same order as :mod:`deepwave.acoustic`)
    are followed by the corresponding scattered wavefields, each spelled by
    inserting ``_sc`` *before* the time index (:func:`_scattered_name`).
    """
    base = cast("list[str]", _acoustic_module.acoustic_wavefield_names(ndim))
    return base + [_scattered_name(name) for name in base]


def acoustic_born_public_wavefield_names(ndim: int) -> list[str]:
    """The acoustic Born state fields in public return order and spelling.

    The backend order (see :func:`acoustic_born_wavefield_names`) is already
    the public order, and every name is already its public spelling, so this
    is that list. It exists so that the spec has one function to call for
    each of the two, whether or not they ever differ.
    """
    return acoustic_born_wavefield_names(ndim)


def acoustic_born_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The CFS-CPML memory variables ``phi_{d}_0`` (pressure update) and
    ``psi_{d}_0`` (velocity update) of the background field, and their
    ``phi_{d}_sc_0``/``psi_{d}_sc_0`` counterparts of the scattered field,
    one of each per spatial dimension. They are a subset of
    :func:`acoustic_born_wavefield_names`, spelled the same way, and are what
    :class:`deepwave.AuxiliaryState` holds.
    """
    dims = _dim_names(ndim)
    background = [f"phi_{d}_0" for d in dims] + [f"psi_{d}_0" for d in dims]
    return background + [_scattered_name(name) for name in background]


def acoustic_born_receiver_names(ndim: int) -> list[str]:
    """Names of the receiver-amplitude outputs of :func:`acoustic_born`.

    The background block first, then the scattered block; within each,
    pressure first then one per velocity component (z, y, x, for the
    dimensions present).
    """
    receivers = ["receiver_amplitudes_p"] + [
        f"receiver_amplitudes_{d}" for d in _dim_names(ndim)
    ]
    return ["bg_" + name for name in receivers] + ["sc_" + name for name in receivers]


def acoustic_born_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`acoustic_born`, in return order.

    The public final wavefields -- the background pressure and velocities, the
    single ``auxiliary_state`` that replaces both blocks of PML memory
    variables at the position of the first of them, and then the scattered
    pressure and velocities -- followed by the background and then the
    scattered receiver amplitudes.
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            acoustic_born_public_wavefield_names(ndim),
            acoustic_born_auxiliary_state_names(ndim),
        ),
        *acoustic_born_receiver_names(ndim),
    ]


# Declarative description of the acoustic Born propagator: a staggered-grid
# solver with background and scattered fields, background+scattering models,
# and (background, scattered) source/receiver types per field component.
SPEC = _engine.PropagatorSpec(
    name="acoustic_born",
    status="experimental",
    source_channels=_engine.volume_rate_velocity_channels,
    receiver_channels=_engine.pressure_velocity_channels,
    bg_receiver_channels=_engine.pressure_velocity_channels,
    staggered=True,
    model_names=acoustic_born_model_names,
    wavefield_names=acoustic_born_wavefield_names,
    n_source_types=lambda ndim: 2 * (ndim + 1),
    n_receiver_types=lambda ndim: 2 * (ndim + 1),
    output_names=acoustic_born_output_names,
    auxiliary_state_names=acoustic_born_auxiliary_state_names,
    public_wavefield_names=acoustic_born_public_wavefield_names,
    spatial_parallel=True,
)


def prepare_scatter(
    v: torch.Tensor, rho: torch.Tensor, dv: torch.Tensor, drho: torch.Tensor
) -> list[torch.Tensor]:
    """Converts velocity/density perturbations to constitutive perturbations.

    Returns ``[dK, dB_z, dB_y, dB_x]`` (depending on ndim), the exact
    directional derivative of ``acoustic.prepare_models`` at
    ``(v, rho)`` in the direction ``(dv, drho)`` -- i.e. ``dK = 2 rho v dv +
    v**2 drho`` and ``dB_d`` is the same staggered arithmetic averaging
    ``prepare_models`` applies to ``rho``, applied to ``drho`` and then
    divided by ``-rho_avg_d**2`` (the derivative of ``1/rho_avg_d``). Computed
    via forward AD so the masked (vacuum-safe) averaging in
    ``prepare_models`` does not have to be re-derived by hand.

    ``jvp`` requires each tangent to have exactly its primal's shape, which
    ordinary broadcasting would not give when (say) a batched ``v`` meets an
    unbatched ``dv``, so the four are first equalised on their leading (shot)
    dimension by :func:`deepwave._common.match_model_batch`. That is a no-op
    when they already agree.
    """
    v, rho, dv, drho = _common.match_model_batch([v, rho, dv, drho])
    return _engine.model_preparation_jvp(
        _acoustic_module.prepare_models, (v, rho), (dv, drho)
    )


class AcousticBorn(torch.nn.Module):
    """A Module wrapper around :func:`acoustic_born`.

    Stability: experimental (see :doc:`/stability`).

    Stores ``v``, ``rho``, ``grid_spacing`` (and, if desired, ``dv``/``drho``)
    once, so that each :meth:`forward` call needs only the arguments that vary
    between calls. As in the functional :func:`acoustic_born`, at least one of
    ``dv``/``drho`` must be provided (the other defaults to ``None``, treated
    as a zero perturbation); the module defers this validation to
    :func:`acoustic_born` itself, which is called on every :meth:`forward`.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The background wavespeed model, of shape ``[*model_shape]`` or,
            for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        rho: The background density model, of shape ``[*model_shape]`` or,
            for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dv: The wavespeed perturbation model, of shape ``[*model_shape]``
            or, for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
            At least one of `dv`/`drho` must be provided.
        drho: The density perturbation model, of shape ``[*model_shape]``
            or, for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
            At least one of `dv`/`drho` must be provided.
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
        dv_requires_grad: A bool specifying whether gradients will be
            computed for `dv`. Unset by default, which means False. Cannot
            be combined with `requires_grad`.
        drho_requires_grad: A bool specifying whether gradients will be
            computed for `drho`. Unset by default, which means False. Cannot
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
            Only `"device"` is permitted with `backend="eager"`
            (the eager backend does not use the engine's storage
            manager).
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
            Not supported with `backend="eager"`.
    """

    def __init__(
        self,
        v: torch.Tensor,
        rho: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        *,
        dv: Optional[torch.Tensor] = None,
        drho: Optional[torch.Tensor] = None,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        v_requires_grad: Optional[bool] = None,
        rho_requires_grad: Optional[bool] = None,
        dv_requires_grad: Optional[bool] = None,
        drho_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        supplied = ["v", "rho"]
        if dv is not None:
            supplied.append("dv")
        if drho is not None:
            supplied.append("drho")
        grad = _common.resolve_requires_grad(
            requires_grad,
            ["v", "rho", "dv", "drho"],
            supplied,
            {
                "v": v_requires_grad,
                "rho": rho_requires_grad,
                "dv": dv_requires_grad,
                "drho": drho_requires_grad,
            },
            "AcousticBorn",
        )
        self.v = _common.make_model_parameter(v, grad["v"], "v")
        self.rho = _common.make_model_parameter(rho, grad["rho"], "rho")
        self.dv: Optional[torch.Tensor] = (
            None if dv is None else _common.make_model_parameter(dv, grad["dv"], "dv")
        )
        self.drho: Optional[torch.Tensor] = (
            None
            if drho is None
            else _common.make_model_parameter(drho, grad["drho"], "drho")
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
        source_amplitudes_volume_rate: Optional[torch.Tensor] = None,
        source_amplitudes_z: Optional[torch.Tensor] = None,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_locations_volume_rate: Optional[LocationsLike] = None,
        source_locations_z: Optional[LocationsLike] = None,
        source_locations_y: Optional[LocationsLike] = None,
        source_locations_x: Optional[LocationsLike] = None,
        source_mask_volume_rate: Optional[torch.Tensor] = None,
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
        pressure_0: Optional[torch.Tensor] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        pressure_sc_0: Optional[torch.Tensor] = None,
        vz_sc_0: Optional[torch.Tensor] = None,
        vy_sc_0: Optional[torch.Tensor] = None,
        vx_sc_0: Optional[torch.Tensor] = None,
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
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
        location_interpolation: Literal["auto", "hicks", "none"] = "auto",
        location_interpolation_halfwidth: int = 4,
    ) -> "deepwave.PropagatorOutput":
        """Perform forward propagation/modelling.

        See :func:`acoustic_born` for the meaning of every argument (the
        constructor-owned ``v``/``rho``/``dv``/``drho``/``grid_spacing`` and the
        storage settings are supplied from the stored module state).
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
        return acoustic_born(
            self.v,
            self.rho,
            self.grid_spacing,
            dt,
            dv=self.dv,
            drho=self.drho,
            source_mask_volume_rate=source_mask_volume_rate,
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
            sources=sources,
            receivers=receivers,
            bg_receivers=bg_receivers,
            source_amplitudes_volume_rate=source_amplitudes_volume_rate,
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_volume_rate=source_locations_volume_rate,
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
            pressure_0=pressure_0,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            pressure_sc_0=pressure_sc_0,
            vz_sc_0=vz_sc_0,
            vy_sc_0=vy_sc_0,
            vx_sc_0=vx_sc_0,
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
            storage=_storage_for_call,
            backend=backend,
            location_interpolation=location_interpolation,
            location_interpolation_halfwidth=location_interpolation_halfwidth,
        )


def zero_pml_interiors_born(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
) -> None:
    """Zeros the edges/interiors of the background and scattered wavefields.

    Deepwave-internal API consumed by the base propagator's forward-mode (jvp)
    and second-order (HVP) rules; not a public interface.
    """
    n = len(_acoustic_module.acoustic_wavefield_names(ndim))
    bg = wavefields[:n]
    sc = wavefields[n:]
    _acoustic_module.zero_edges_and_interiors(bg, ndim, fd_pad, fd_pad_list, pml_width)
    _acoustic_module.zero_edges_and_interiors(sc, ndim, fd_pad, fd_pad_list, pml_width)
    wavefields[:n] = bg
    wavefields[n:] = sc


def acoustic_born(
    v: torch.Tensor,
    rho: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    dv: Optional[torch.Tensor] = None,
    drho: Optional[torch.Tensor] = None,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    bg_receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes_volume_rate: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_volume_rate: Optional[LocationsLike] = None,
    source_locations_z: Optional[LocationsLike] = None,
    source_locations_y: Optional[LocationsLike] = None,
    source_locations_x: Optional[LocationsLike] = None,
    source_mask_volume_rate: Optional[torch.Tensor] = None,
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
    pressure_0: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    pressure_sc_0: Optional[torch.Tensor] = None,
    vz_sc_0: Optional[torch.Tensor] = None,
    vy_sc_0: Optional[torch.Tensor] = None,
    vx_sc_0: Optional[torch.Tensor] = None,
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
    storage: StorageLike = STORAGE_DEFAULT,
    storage_mode: Optional[StorageMode] = None,
    storage_path: Optional[StoragePath] = None,
    storage_compression: Optional[bool] = None,
    location_interpolation: Literal["auto", "hicks", "none"] = "auto",
    location_interpolation_halfwidth: int = 4,
) -> "deepwave.PropagatorOutput":
    """Acoustic Born (linearised) wave propagation (functional interface).

    Stability: experimental (see :doc:`/stability`).

    Performs first-order Born forward modelling with the variable-density
    acoustic wave equation: the background field solves the ordinary
    (nonlinear) acoustic system for ``v, rho``; the scattered field is the
    exact linearisation for perturbations ``dv``, ``drho``, driven by virtual
    sources built from the constitutive perturbations ``dK = 2 rho v dv + v**2
    drho`` and ``dB_d = -d(rho_avg_d)/rho_avg_d**2`` (see
    ``docs/acoustic_born_derivation.md``).

    A single set of source amplitudes per component drives both fields: the
    background field is driven as in :func:`deepwave.acoustic`
    (scaled by ``dt*K``/``dt*B_d``); the scattered field is *automatically*
    driven by the same amplitudes scaled by ``dt*dK``/``dt*dB_d`` -- the
    "source-cell" term that arises because that scaling is itself
    model-dependent (mirrors :func:`deepwave.scalar_born`'s single
    shared ``source_amplitudes``). No separate scattered-source argument is
    needed or accepted.

    **Backend note.** ``torch.func`` transforms (``vmap``, forward-mode
    ``jvp``) and higher-order derivatives are not available on the default
    compiled backend, but the pure-PyTorch ``backend="eager"`` driver
    runs under native autograd and composes with them (``vmap`` over the
    velocity model, ``jvp``).

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The background wavespeed model, of shape ``[*model_shape]`` or,
            for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        rho: The background density model, of shape ``[*model_shape]`` or,
            for a batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: Spatial grid cell size.
        dt: Time step interval. See :func:`deepwave.acoustic` for the
            unit-consistency requirement across ``v``/``rho``/``grid_spacing``/
            ``dt``.
        dv: The wavespeed scattering perturbation model (``None`` -> zero),
            of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. At least one
            of ``dv``/``drho`` must be provided.
        drho: The density scattering perturbation model (``None`` -> zero),
            of shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        sources: This simulation's sources, keyed by channel. This is the
            canonical way to specify a survey. Each value is ``(amplitudes,
            locations)``, or ``(amplitudes, locations, mask)`` to include an
            activity mask. The channels are ``volume_rate``, ``z``, ``y``,
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
        source_amplitudes_volume_rate: Source amplitudes (a volume-injection rate per
            unit volume, with units of 1/time), as in
            :func:`deepwave.acoustic`; see :doc:`/units`.
            To drive the propagation with a
            volumetric (extended) source instead of hand-placed point sources,
            build the amplitudes and integer locations with
            :func:`deepwave.wavefields.source_field_to_sources`.
        source_locations_volume_rate: Source locations, as in
            :func:`deepwave.acoustic`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_z: Source amplitudes (a z-direction body-force
            density, with units of force/volume), as in
            :func:`deepwave.acoustic`. This channel exists only in 3-D.
        source_locations_z: Source locations (z-force), as in
            :func:`deepwave.acoustic`. This channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_y: Source amplitudes (y-force), as in
            :func:`deepwave.acoustic`.
        source_locations_y: Source locations (y-force), as in
            :func:`deepwave.acoustic`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it injects nothing. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        source_amplitudes_x: Source amplitudes (x-force), as in
            :func:`deepwave.acoustic`.
        source_locations_x: Source locations (x-force), as in
            :func:`deepwave.acoustic`.
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
        sc_receiver_locations_p: Pressure receiver locations for the *scattered*
            receiver amplitudes (the Born data).
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_locations_z: Z-component velocity receiver locations for
            the scattered receiver amplitudes. This channel exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_locations_y: Y-component velocity receiver locations for the
            scattered receiver amplitudes.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_locations_x: X-component velocity receiver locations for the
            scattered receiver amplitudes.
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
        bg_receiver_locations_p: Optional pressure receiver locations for the
            background receiver amplitudes.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations_z: Optional z-component velocity receiver
            locations for the background receiver amplitudes. This channel
            exists only in 3-D.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations_y: Optional y-component velocity receiver
            locations for the background receiver amplitudes.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations_x: Optional x-component velocity receiver
            locations for the background receiver amplitudes.
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
        pressure_0: Initial background pressure, as in :func:`deepwave.acoustic`.
        vz_0: Initial background z-velocity.
        vy_0: Initial background y-velocity.
        vx_0: Initial background x-velocity.
        pressure_sc_0: Initial scattered pressure.
        vz_sc_0: Initial scattered z-velocity.
        vy_sc_0: Initial scattered y-velocity.
        vx_sc_0: Initial scattered x-velocity.
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
            `pml_width == 0` on that face). Applied independently to the
            background field `(p, v_d)` and the scattered field `(dp,
            dv_d)`: both are an ordinary acoustic staggered-grid field on the
            same grid (see the module docstring), so the same acoustic
            per-axis image rule (pressure odd/even, normal velocity
            even/odd, tangential velocity untouched -- see
            `docs/boundaries.rst`) applies to each independently, with no
            new physics beyond what :func:`deepwave.acoustic` already
            validates. Validated on the eager (`backend="eager"`)
            and the default compiled C/CUDA backends; the
            `"torch_compile"` Python mode shares the eager guard-fill code
            path but is not separately validated for `boundary=`.
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
        courant_number: See :func:`deepwave.elastic`.
        survey_pad: See :func:`deepwave.scalar`.
        initial_state: See :func:`deepwave.elastic`. The mapping keys are this
            propagator's own public state field names -- the background names of
            :func:`deepwave.acoustic` (e.g. `pressure_0`) and the scattered
            names, which carry `sc` before the time index (e.g.
            `pressure_sc_0`) -- plus `auxiliary_state`. The individual
            absorbing-boundary field names (`phi_z_0`, ..., `psi_x_0` and their
            scattered counterparts) are not accepted as keys: pass
            `initial_state=out` (or `initial_state=out.final_state`) to carry
            them across.
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
        forward_callback: See :func:`deepwave.elastic`. Supported on both the default
            compiled backend and the eager backend (`backend="eager"`).
        backward_callback: See :func:`deepwave.elastic`. Supported only on the default
            compiled backend; the eager backend rejects it.
        callback_frequency: See :func:`deepwave.elastic`.
        backend: Which execution backend to use: ``"auto"`` (the default),
            ``"compiled"``, ``"eager"`` or ``"torch_compile"``. See
            :doc:`/backends` for what each value selects, what ``"auto"``
            resolves to, and how the backends agree numerically.
            ``"auto"`` resolves to the compiled C/CUDA forward paired with
            the compiled adjoint kernels, and supports callbacks and the
            ``"cpu"``/``"disk"``/``"none"`` storage modes. ``"eager"`` and
            ``"torch_compile"`` run the pure-PyTorch coupled Born driver,
            whose two step functions are compiled in the latter.
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
            Only `"device"` is permitted with `backend="eager"`
            (the eager backend does not use the engine's storage
            manager).
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
            Not supported with `backend="eager"`.
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

            - pressure_0: Final background pressure wavefield.
            - vz_0: Final background z-velocity wavefield (if ndim == 3).
            - vy_0: Final background y-velocity wavefield (if ndim >= 2).
            - vx_0: Final background x-velocity wavefield.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables, of both the background
              and the scattered field. Opaque: pass it back (as part of
              `initial_state`) to continue the simulation; its contents are
              implementation state rather than physics.
            - pressure_sc_0: Final scattered pressure wavefield.
            - vz_sc_0: Final scattered z-velocity wavefield (if ndim == 3).
            - vy_sc_0: Final scattered y-velocity wavefield (if ndim >= 2).
            - vx_sc_0: Final scattered x-velocity wavefield.
            - bg_receiver_amplitudes_p: Recorded background pressure data.
            - bg_receiver_amplitudes_z: Recorded background z-velocity data
              (if ndim == 3).
            - bg_receiver_amplitudes_y: Recorded background y-velocity data
              (if ndim >= 2).
            - bg_receiver_amplitudes_x: Recorded background x-velocity data.
            - sc_receiver_amplitudes_p: Recorded scattered pressure data.
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
            acoustic_born,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
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
    _common.check_models([v, rho], ["v", "rho"], "acoustic_born")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, None, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend
    if dv is None and drho is None:
        raise DeepwaveValueError(
            "acoustic_born: at least one scattering model (dv, drho) must be provided."
        )
    if dv is None:
        dv = torch.zeros_like(v)
    if drho is None:
        drho = torch.zeros_like(rho)
    _common.check_models([dv, drho], ["dv", "drho"], "acoustic_born")
    # Initial wavefields reach this call either as explicit arguments or
    # inside the optional ``initial_state`` bundle, and either way they are
    # part of the survey, so both must be visible to the ndim determination
    # below. The bundle is therefore read here, before it is merged into the
    # explicit arguments (which happens once ndim is known, because ndim
    # decides which state slots exist). Only its wavefield entries can say
    # what ndim is: receiver amplitudes have a different rank, and the
    # auxiliary state is not a Tensor.
    _recv_names = _engine.receiver_output_names(SPEC)
    _state_map: Optional[Mapping[str, Union[torch.Tensor, _engine.AuxiliaryState]]]
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
    # The survey fixes ndim, never the models: a batched model carries a
    # leading shot dimension, so ``v.ndim`` would read a 2-D batch as a 3-D
    # model. This is the same determination every other propagator makes.
    ndim = _common.get_ndim(
        [v, rho, dv, drho],
        [pressure_0, pressure_sc_0, *_state_wavefields],
        [
            source_locations_volume_rate,
            source_locations_z,
            source_locations_y,
            source_locations_x,
            sc_receiver_locations_p,
            sc_receiver_locations_z,
            sc_receiver_locations_y,
            sc_receiver_locations_x,
            bg_receiver_locations_p,
            bg_receiver_locations_z,
            bg_receiver_locations_y,
            bg_receiver_locations_x,
        ],
        [vz_0, vz_sc_0],
        [vy_0, vy_sc_0],
        [vx_0, vx_sc_0],
    )
    # Without these, a component that does not exist at this ndim is dropped
    # silently by the per-ndim assembly below (the user's source vanishes),
    # rather than raising as it does on every other propagator.
    if ndim < 3 and (
        source_locations_z is not None
        or sc_receiver_locations_z is not None
        or bg_receiver_locations_z is not None
    ):
        raise DeepwaveValueError(
            "acoustic_born: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "z dimension should not be provided."
        )
    if ndim < 2 and (
        source_locations_y is not None
        or sc_receiver_locations_y is not None
        or bg_receiver_locations_y is not None
    ):
        raise DeepwaveValueError(
            "acoustic_born: the propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "y dimension should not be provided."
        )

    # The initial_state slots are named the way the outputs are: the public
    # state names, in the propagator's own (backend) order, so the mapping
    # doubles as the unpacking order. Only the physical fields (pressure_0,
    # v*_0, and their scattered counterparts; see A4.3) are exposed as
    # explicit keyword arguments -- the phi_*/psi_* CPML memory variables are
    # always seeded as None here and travel in `auxiliary_state`.
    # resolve_initial_state accepts either a mapping or a whole
    # PropagatorOutput (whose receiver fields are ignored) and validates the
    # keys. The auxiliary state is read here but checked after the call's own
    # parameters have been resolved (there is nothing to compare it against
    # until then).
    wf_names = acoustic_born_public_wavefield_names(ndim)
    _physical_wf: dict[str, Optional[torch.Tensor]] = {
        "pressure_0": pressure_0,
        "vz_0": vz_0,
        "vy_0": vy_0,
        "vx_0": vx_0,
        "pressure_sc_0": pressure_sc_0,
        "vz_sc_0": vz_sc_0,
        "vy_sc_0": vy_sc_0,
        "vx_sc_0": vx_sc_0,
    }
    wf_args: dict[str, Optional[torch.Tensor]] = {}
    for name in wf_names:
        wf_args[name] = _physical_wf.get(name)
    supplied_auxiliary_state = _engine.auxiliary_state_of(initial_state)
    # A continuation has nothing of its own to resolve these from -- it
    # typically has no sources, and the first call's `max_vel` need not be the
    # model's own maximum -- so an omitted one takes the value the state's
    # absorbing-boundary memory variables were built with.
    pml_freq = _engine.inherited_pml_freq(pml_freq, supplied_auxiliary_state)
    max_vel = _engine.inherited_max_vel(max_vel, supplied_auxiliary_state)
    _resolved_wavefields, origin = _engine.resolve_initial_state(
        initial_state,
        wf_args,
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=acoustic_born_auxiliary_state_names(3),
        propagator=SPEC.name,
    )
    initial_wavefields: list[Optional[torch.Tensor]] = list(
        _resolved_wavefields.values()
    )

    source_amplitudes_list = [source_amplitudes_volume_rate]
    source_locations_list_in = [source_locations_volume_rate]
    if ndim == 3:
        source_amplitudes_list.append(source_amplitudes_z)
        source_locations_list_in.append(source_locations_z)
    if ndim >= 2:
        source_amplitudes_list.append(source_amplitudes_y)
        source_locations_list_in.append(source_locations_y)
    source_amplitudes_list.append(source_amplitudes_x)
    source_locations_list_in.append(source_locations_x)

    receiver_locations_list_in = [sc_receiver_locations_p]
    bg_receiver_locations_list_in = [bg_receiver_locations_p]
    if ndim == 3:
        receiver_locations_list_in.append(sc_receiver_locations_z)
        bg_receiver_locations_list_in.append(bg_receiver_locations_z)
    if ndim >= 2:
        receiver_locations_list_in.append(sc_receiver_locations_y)
        bg_receiver_locations_list_in.append(bg_receiver_locations_y)
    receiver_locations_list_in.append(sc_receiver_locations_x)
    bg_receiver_locations_list_in.append(bg_receiver_locations_x)

    _common.check_inputs_not_vmapped(
        v,
        rho,
        dv,
        drho,
        *source_amplitudes_list,
        *source_locations_list_in,
        *receiver_locations_list_in,
        *bg_receiver_locations_list_in,
    )

    # Resolve off-grid / directional (Hicks) source and receiver
    # specifications. Pressure sits on the regular grid (offset None); each
    # velocity component is staggered half a cell in its own dimension (see
    # deepwave.acoustic). A single source interpolator is built from
    # source_locations_list_in and its spread applied once, then shared by the
    # background and the (internally derived) scattered source -- they drive
    # identical physical locations (see the module docstring's source-cell
    # paragraph). The background and scattered receivers have independent
    # interpolators.
    component_offsets: list[Optional[list[float]]] = [
        None,
        *_common.velocity_component_offsets(ndim),
    ]
    # Recorded before the location arguments are replaced by the integer grid
    # indices the backend consumes.
    location_interpolation_used = _engine.resolved_location_interpolation(
        location_interpolation,
        [
            source_locations_list_in,
            bg_receiver_locations_list_in,
            receiver_locations_list_in,
        ],
        [component_offsets, component_offsets, component_offsets],
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
        [
            *source_locations_list_in,
            *bg_receiver_locations_list_in,
            *receiver_locations_list_in,
        ],
        v,
        "acoustic_born",
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
    (
        bg_receiver_interpolators,
        bg_receiver_locations_list,
    ) = _common.resolve_receiver_interpolators(
        bg_receiver_locations_list_in,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        v.dtype,
        _hicks_grid,
    )
    (
        receiver_interpolators,
        receiver_locations_list,
    ) = _common.resolve_receiver_interpolators(
        receiver_locations_list_in,
        component_offsets,
        location_interpolation,
        location_interpolation_halfwidth,
        v.dtype,
        _hicks_grid,
    )

    # A velocity source/receiver must not sit on the final sample of its
    # staggered dimension (that cell is not part of the computational
    # domain). The check runs on the resolved integer grid locations;
    # pressure is at index 0, so the velocity component for dimension
    # ``dim`` is at index ``dim + 1`` (mirrors deepwave.acoustic).
    dim_names = ["z", "y", "x"]
    for dim in range(ndim):
        for locations, name in [
            (source_locations_list, "source"),
            (bg_receiver_locations_list, "bg_receiver"),
            (receiver_locations_list, "sc_receiver"),
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
                    f"acoustic_born: {name}_locations_"
                    f"{dim_names[-ndim + dim]} is out of bounds: a "
                    f"{dim_names[-ndim + dim]} location is staggered half a "
                    "cell past the node it names, so its "
                    f"{dim_names[-ndim + dim]} coordinate must be less than "
                    f"{v.shape[-ndim + dim] - 1}.",
                )

    min_nonzero_model_vel, max_model_vel = _common.velocity_extremes(
        v, max_vel, "acoustic_born"
    )
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

    # models: [v, rho] fed to setup_propagator for padding/survey_pad only;
    # the actual K/B/dK/dB models are derived afterward (mirrors acoustic()).
    all_source_amplitudes = source_amplitudes_list + source_amplitudes_list
    all_source_locations = source_locations_list + source_locations_list
    all_receiver_locations = bg_receiver_locations_list + receiver_locations_list

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
        [v, rho, dv, drho],
        ["replicate", "replicate", "replicate", "replicate"],
        ["v", "rho", "dv", "drho"],
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
    v_p, rho_p, dv_p, drho_p = models
    del v, rho, dv, drho, models

    bg_models = _acoustic_module.prepare_models(v_p, rho_p)
    sc_models = prepare_scatter(v_p, rho_p, dv_p, drho_p)
    models_out = bg_models + sc_models  # [K, B_*, dK, dB_*]
    model_shape = models_out[0].shape[-ndim:]

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "acoustic_born")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "acoustic_born")
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
            models_out, source_amplitudes_out, wavefields
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

    n_src = ndim + 1
    # Scale source amplitudes: background by dt*model, scattered (automatic
    # "source-cell" virtual source) by dt*dmodel, sharing the same input
    # amplitudes and locations (see the docstring's source-cell paragraph).
    for i in range(n_src):
        src_amp = source_amplitudes_out[i]
        src_loc = sources_i[i]
        model = bg_models[i]
        dmodel = sc_models[i]
        if src_amp.numel() > 0:
            gathered = _common.gather_model_at_sources(
                model, src_loc, model_shape, n_shots
            )
            dgathered = _common.gather_model_at_sources(
                dmodel, src_loc, model_shape, n_shots
            )
            source_amplitudes_out[i + n_src] = src_amp * dgathered * dt
            source_amplitudes_out[i] = src_amp * gathered * dt
        else:
            source_amplitudes_out[i + n_src] = source_amplitudes_out[i]

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
    )

    _common.check_callback_frequency(callback_frequency)
    callback_frequency = _engine.CallbackSchedule(callback_frequency, step_ratio)

    # The single argument list drives either core (compiled or eager); it
    # matches acoustic_born_python's positional layout exactly.
    core_args = (
        *models_out,
        *source_amplitudes_out,
        *sources_i,
        *receivers_i,
        *wavefields,
    )

    run_info = RunInfo(
        step_ratio=step_ratio,
        model_gradient_sampling_interval=requested_mgsi,
        model_gradient_stride=_engine.resolved_model_gradient_stride(
            models_out, backend_mode, step_ratio, model_gradient_sampling_interval
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
        backward_backend=_engine.resolved_backend_name(backend_mode),
        max_vel=max_vel,
        ndim=ndim,
        dtype=dtype,
        device=device,
        model_batched=tuple(_engine.model_batched(models_out, ndim)),
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
        models_out,
        storage_mode,
        python_backend=backend_mode,
        storage_compression=storage_compression,
        has_backward_callback=backward_callback is not None,
    )

    if not backend_mode:
        # Default: the compiled (fast) C/CUDA forward with the compiled CPU
        # adjoint kernels; see AcousticBornForwardFunc.
        #
        # storage_mode="none" keeps no forward wavefield history, so the model
        # gradient (an imaging correlation of the forward and adjoint fields)
        # cannot be formed. Forward-only "none" is fully supported (a valid
        # low-memory mode); combining it with a differentiable input is rejected
        # with a clear error rather than segfaulting when the backward reads the
        # unallocated store (see the storage-write gate in _make_forward_call
        # above). Use
        # "device"/"cpu"/"disk" for gradients w.r.t. the models/sources/
        # wavefields.
        if storage_mode == "none" and any(
            isinstance(t, torch.Tensor) and t.requires_grad for t in core_args
        ):
            raise DeepwaveRuntimeError(
                'acoustic_born: storage_mode="none" retains no forward '
                "wavefield history, so it cannot compute gradients; it is a "
                "forward-only (no-grad) low-memory mode. A model, source "
                "amplitude, or initial wavefield requires grad -- use "
                'storage_mode of "device", "cpu", or "disk" for the backward '
                "pass."
            )
        outputs = list(
            acoustic_born_func(
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
                boundary_codes,
                *core_args,
            )
        )
    else:
        global _update_velocities_born_opt, _update_pressure_born_opt

        # See _engine.float_grid_spacing: the traced step functions must
        # receive Python floats, not ints. setup_propagator already
        # guarantees this, so this is defence in depth.
        grid_spacing_out = _engine.float_grid_spacing(grid_spacing_out)
        _update_velocities_born_opt = _update_velocities_born_cache.resolve(
            backend_mode
        )
        _update_pressure_born_opt = _update_pressure_born_cache.resolve(backend_mode)
        # The pure-PyTorch eager/compiled backends run native autograd, not
        # the shared step-based engine, so they have no storage manager and no
        # chunked backward-callback machinery (a backward_callback needs the
        # adjoint pass the engine drives, which these backends do not have).
        # forward_callback has no such dependency -- acoustic_born_python runs
        # its own per-step loop -- so it is threaded through instead of
        # rejected (mirrors deepwave.acoustic's own eager backend). Unlike the
        # other propagators, acoustic_born_python is reached directly from
        # here rather than through an acoustic_born_func wrapper, so the
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
            acoustic_born_python(
                pml_profiles,
                grid_spacing_out,
                dt,
                nt,
                step_ratio,
                accuracy,
                pml_width,
                boundary_codes,
                n_shots,
                forward_callback,
                callback_frequency,
                *core_args,
            )
        )

    n_wf = len(wf_names)
    for i in range(n_wf, n_wf + 2 * (ndim + 1)):
        outputs[i] = _common.downsample_and_movedim(
            outputs[i],
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
        )

    # Gather off-grid receiver data back to the original locations. The
    # receiver outputs are the trailing 2 * (ndim + 1) entries: the background
    # block (matching bg_receiver_interpolators), then the scattered block
    # (matching receiver_interpolators).
    _common.apply_receiver_interpolators(outputs, bg_receiver_interpolators, n_wf)
    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, n_wf + ndim + 1
    )

    return _engine.named_output(
        SPEC,
        ndim,
        outputs,
        info=run_info,
        accuracy=accuracy,
        boundary=boundary_names,
        check_output_finite=check_output_finite,
    )


def _make_forward_call(ndim: int) -> Any:
    """Builds the ``make_call`` closure for the coupled forward pass.

    Mirrors the exact parameter order of ``acoustic_born.c``'s ``forward``:
    background models, scatter models, background sources, scattered
    sources, background wavefields, scattered wavefields, background-term
    stores, background receivers, scattered receivers, pml profiles,
    background then scattered source/receiver indices, then the scalar/size
    trailers.
    """
    n_bg = ndim + 1
    n_bg_wf = 1 + 3 * ndim

    def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
        bg_models = state.models[:n_bg]
        sc_models = state.models[n_bg:]
        f_bg = state.source_amplitudes[:n_bg]
        f_sc = state.source_amplitudes[n_bg:]
        wf_bg = state.wavefields[:n_bg_wf]
        wf_sc = state.wavefields[n_bg_wf:]
        r_bg = state.receiver_amplitudes[:n_bg]
        r_sc = state.receiver_amplitudes[n_bg:]
        src_i_bg = state.sources_i[:n_bg]
        src_i_sc = state.sources_i[n_bg:]
        rcv_i_bg = state.receivers_i[:n_bg]
        rcv_i_sc = state.receivers_i[n_bg:]
        dk = sc_models[0]
        db = sc_models[1:]
        k = bg_models[0]
        b = bg_models[1:]
        # storage_mode="none" allocates NO backward-storage buffers (see
        # StorageManager: the per-slot tensors stay empty when the mode is
        # NONE), so the engine store pointers are not writable. The kernel
        # gates its snapshot writes on these store bools, so they MUST be
        # forced False in NONE mode or the -Ofast SAVE_SNAPSHOT block writes
        # through an empty (0-element) buffer and corrupts the heap. This
        # matches the base
        # scalar/acoustic/elastic behaviour for storage_mode="none": the
        # forward stores nothing, so the backward has no forward history to
        # correlate and the model gradient is (correctly) not reconstructed.
        store_none = state.storage_manager.storage_mode == _storage.StorageMode.NONE
        need_bg = (
            False if store_none else k.requires_grad or any(m.requires_grad for m in b)
        )
        need_sc = (
            False
            if store_none
            else dk.requires_grad or any(m.requires_grad for m in db)
        )
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in bg_models],
                *[m.data_ptr() for m in sc_models],
                *[a.data_ptr() for a in f_bg],
                *[a.data_ptr() for a in f_sc],
                *[w.data_ptr() for w in wf_bg],
                *[w.data_ptr() for w in wf_sc],
                *state.storage_manager.storage_ptrs,
                *[a.data_ptr() for a in r_bg],
                *[a.data_ptr() for a in r_sc],
                *[p.data_ptr() for p in state.pml_profiles],
                *[loc.data_ptr() for loc in src_i_bg],
                *[loc.data_ptr() for loc in src_i_sc],
                *[loc.data_ptr() for loc in rcv_i_bg],
                *[loc.data_ptr() for loc in rcv_i_sc],
                *_engine.call_grid_args(state, step_nt),
                *state.n_sources_per_shot[:n_bg],
                *state.n_sources_per_shot[n_bg:],
                *state.n_receivers_per_shot[:n_bg],
                *state.n_receivers_per_shot[n_bg:],
                *_engine.call_storage_meta_args(state),
                need_bg,
                need_bg,
                need_sc,
                need_sc,
                k.ndim == ndim + 1 and k.shape[0] > 1,
                b[0].ndim == ndim + 1 and b[0].shape[0] > 1,
                *_engine.call_pml_tail_args(state, step),
            ),
        )

    return make_call


def _make_backward_sc_call(ndim: int) -> Any:
    """Builds the ``make_call`` closure for the scattered-only backward pass.

    ``state.models`` here is ``[K, B_*, dK, dB_*]`` (background then scatter,
    matching the ``models=`` list passed to :func:`run_backward_core`); only
    the scatter half owns a gradient (``state.grad_models`` accordingly has
    empty background entries).
    """
    n_bg = ndim + 1

    def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
        bg_models = state.models[:n_bg]
        sc_models = state.models[n_bg:]
        k = bg_models[0]
        buoyancy = bg_models[1:]
        dk = sc_models[0]
        db = sc_models[1:]
        w = state.wavefields  # scattered adjoint only: [psc, v*sc, phi_*_sc, psi_*_sc]
        a = state.aux_wavefields
        psi_sc = w[1 + 2 * ndim : 1 + 3 * ndim]
        psin_sc = a[:ndim]
        grad_dk = state.grad_models[n_bg]
        grad_db = state.grad_models[n_bg + 1 :]
        n_f = [
            state.n_sources_per_shot[i] if state.grad_sources[i].numel() > 0 else 0
            for i in range(n_bg)
        ]
        return cast(
            "int",
            state.backend(
                k.data_ptr(),
                *[b.data_ptr() for b in buoyancy],
                *[r.data_ptr() for r in state.grad_receivers],
                w[0].data_ptr(),
                *[f.data_ptr() for f in w[1 : 1 + ndim]],
                *[f.data_ptr() for f in w[1 + ndim : 1 + 2 * ndim]],
                *[f.data_ptr() for f in psi_sc],
                *[f.data_ptr() for f in psin_sc],
                # The forward now allocates 2*(ndim+1) store slots (background
                # then scattered); the scattered-only backward consumes only the
                # (ndim+1) background-term slots (div_bg / term_bg_d).
                *state.storage_manager.storage_ptrs[: 5 * (ndim + 1)],
                *[f.data_ptr() for f in state.grad_sources],
                grad_dk.data_ptr(),
                *[m.data_ptr() for m in grad_db],
                *state.grad_models_tmp_ptr[n_bg:],
                *[p.data_ptr() for p in state.pml_profiles],
                *[s.data_ptr() for s in state.sources_i],
                *[r.data_ptr() for r in state.receivers_i],
                *_engine.call_grid_args(state, step_nt),
                *n_f,
                *state.n_receivers_per_shot,
                *_engine.call_storage_meta_args(state),
                dk.requires_grad,
                any(m.requires_grad for m in db),
                k.ndim == ndim + 1 and k.shape[0] > 1,
                buoyancy[0].ndim == ndim + 1 and buoyancy[0].shape[0] > 1,
                dk.ndim == ndim + 1 and dk.shape[0] > 1,
                db[0].ndim == ndim + 1 and db[0].shape[0] > 1,
                *_engine.call_pml_tail_args(state, step),
            ),
        )

    return make_call


def _make_backward_call(ndim: int) -> Any:
    """Builds the ``make_call`` closure for the full two-field backward pass.

    Propagates both the background and scattered adjoint fields (mirroring
    ``acoustic_born.c``'s ``backward``), accumulating background *and* scattered
    model gradients plus background *and* scattered source gradients. Consumes
    both the background-term and scattered-term stores. Used when a
    background-side quantity (``v``/``rho``, a source amplitude, or a background
    initial wavefield) requires a gradient, and by the acoustic HVP.

    ``state.models`` is ``[K, B_*, dK, dB_*]``; ``state.wavefields`` is the
    background adjoint wavefields followed by the scattered ones (each block in
    ``acoustic_wavefield_names`` order); ``state.aux_wavefields`` is the psi
    ping-pong temporaries, background block then scattered block. The
    receivers/sources/grad_receivers/grad_sources lists are each background
    (ndim+1) then scattered (ndim+1). The phi/psi wavefield slots map to the C
    argument order exactly as in the scattered-only pass (see
    :func:`_make_backward_sc_call`).

    Deepwave-internal API consumed by the base propagator's forward-mode (jvp)
    and second-order (HVP) rules; not a public interface.
    """
    n_bg = ndim + 1
    n_bg_wf = 1 + 3 * ndim

    def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
        bg_models = state.models[:n_bg]
        sc_models = state.models[n_bg:]
        k = bg_models[0]
        buoyancy = bg_models[1:]
        dk = sc_models[0]
        db = sc_models[1:]
        w = state.wavefields
        a = state.aux_wavefields
        # Background adjoint field.
        bg_p = w[0]
        bg_v = w[1 : 1 + ndim]
        bg_phi = w[1 + ndim : 1 + 2 * ndim]
        bg_psi = w[1 + 2 * ndim : 1 + 3 * ndim]
        bg_psin = a[:ndim]
        # Scattered adjoint field.
        sc_p = w[n_bg_wf]
        sc_v = w[n_bg_wf + 1 : n_bg_wf + 1 + ndim]
        sc_phi = w[n_bg_wf + 1 + ndim : n_bg_wf + 1 + 2 * ndim]
        sc_psi = w[n_bg_wf + 1 + 2 * ndim : n_bg_wf + 1 + 3 * ndim]
        sc_psin = a[ndim : 2 * ndim]
        n_f = [
            state.n_sources_per_shot[i] if state.grad_sources[i].numel() > 0 else 0
            for i in range(2 * n_bg)
        ]
        return cast(
            "int",
            state.backend(
                *[m.data_ptr() for m in bg_models],
                *[m.data_ptr() for m in sc_models],
                *[r.data_ptr() for r in state.grad_receivers],
                bg_p.data_ptr(),
                *[f.data_ptr() for f in bg_v],
                *[f.data_ptr() for f in bg_phi],
                *[f.data_ptr() for f in bg_psi],
                *[f.data_ptr() for f in bg_psin],
                sc_p.data_ptr(),
                *[f.data_ptr() for f in sc_v],
                *[f.data_ptr() for f in sc_phi],
                *[f.data_ptr() for f in sc_psi],
                *[f.data_ptr() for f in sc_psin],
                *state.storage_manager.storage_ptrs,
                *[f.data_ptr() for f in state.grad_sources],
                *[m.data_ptr() for m in state.grad_models],
                *state.grad_models_tmp_ptr,
                *[p.data_ptr() for p in state.pml_profiles],
                *[s.data_ptr() for s in state.sources_i],
                *[r.data_ptr() for r in state.receivers_i],
                *_engine.call_grid_args(state, step_nt),
                *n_f,
                *state.n_receivers_per_shot,
                *_engine.call_storage_meta_args(state),
                k.requires_grad,
                any(m.requires_grad for m in buoyancy),
                dk.requires_grad,
                any(m.requires_grad for m in db),
                k.ndim == ndim + 1 and k.shape[0] > 1,
                buoyancy[0].ndim == ndim + 1 and buoyancy[0].shape[0] > 1,
                dk.ndim == ndim + 1 and dk.shape[0] > 1,
                db[0].ndim == ndim + 1 and db[0].shape[0] > 1,
                *_engine.call_pml_tail_args(state, step),
            ),
        )

    return make_call


class AcousticBornForwardFunc(torch.autograd.Function):
    """Forward propagation function for acoustic Born modelling."""

    @staticmethod
    def forward(
        pml_profiles: list[torch.Tensor],
        grid_spacing: Sequence[float],
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
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Forward propagation of the acoustic Born wave equation.

        ``boundary`` is the per-face integer code list (see
        :func:`deepwave._common.set_boundary`); threaded through to the
        compiled backend's ``bc`` ABI exactly as :func:`deepwave.acoustic`'s
        own forward does, applied independently to the background and
        scattered fields (see :func:`_make_forward_call`).
        """
        ndim = len(grid_spacing)

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # Two blocks of (ndim + 1) store slots: background-term stores
            # (div_bg / term_bg_d) then scattered-term stores (div_sc /
            # term_sc_d). Background-term stores feed every model gradient
            # (grad_dK/grad_dB_d and grad_K/grad_B_d), so they are needed if any
            # model requires grad. Scattered-term stores feed only the
            # background model gradient (the cross-term of section 4.3), so they
            # are needed only when a background model (K/B_d) requires grad.
            k, *b_bg = models[: ndim + 1]
            dk, *db = models[ndim + 1 :]
            need_bg_model = k.requires_grad or any(m.requires_grad for m in b_bg)
            need_sc_model = dk.requires_grad or any(m.requires_grad for m in db)
            need_bg_store = need_bg_model or need_sc_model
            return [need_bg_store] * (ndim + 1) + [need_bg_model] * (ndim + 1)

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
            args=args,
            storage_flags=storage_flags,
            zero_wavefields=zero_pml_interiors_born,
            make_call=_make_forward_call(ndim),
            boundary=boundary,
        )
        return (*outputs, saved_state)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Populates ``ctx`` from the trailing state produced by ``forward``."""
        _engine.apply_forward_saved_state(ctx, output[-1])
        # ``boundary`` is the last leading (non-tensor) input (see
        # ``AcousticBornForwardFunc.forward``: inserted right before
        # ``*args``); recovered here so ``backward`` can re-thread it into
        # the compiled adjoint's ``bc`` ABI.
        ctx.boundary = inputs[_N_LEAD - 1]

    # The compiled ctypes backend cannot process batched/dual tensors, so the
    # functorch transforms raise actionably and point the user at the
    # pure-PyTorch backend="eager" driver, which composes with them
    # through native autograd.
    vmap = staticmethod(_engine.unsupported_vmap)
    jvp = staticmethod(_engine.unsupported_jvp)

    @staticmethod
    @_engine.no_second_order("acoustic_born")
    def backward(ctx: Any, *args: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        """Backward propagation of the acoustic Born wave equation.

        Dispatches between the full two-field adjoint (background AND scattered)
        and the scattered-only adjoint. The full adjoint is used whenever a
        background-side quantity needs a gradient -- the background ``v``/``rho``
        (i.e. ``K``/``B_d``), a source amplitude, or a background initial
        wavefield -- because the scattered output depends on those both directly
        and through the background field's propagation into the Born coupling.
        Otherwise (pure LSRTM in ``dv``/``drho`` and/or the scattered initial
        wavefields) the cheaper scattered-only adjoint (``backward_sc``)
        suffices. See ``docs/acoustic_born_derivation.md`` section 4.3.
        """
        args = args[:-1]
        ndim = _engine.saved_state_ndim(ctx, "acoustic_born")
        n_bg = ndim + 1
        n_bg_wf = 1 + 3 * ndim
        n_wf_total = 2 * n_bg_wf
        n_rec = 2 * n_bg
        n_src = 2 * n_bg
        grad_wavefields_all = list(args[:n_wf_total])
        # acoustic_born's output order is (bg wavefields, sc wavefields, bg
        # receivers, sc receivers); the backward args mirror that order. The
        # receiver-gradient tensors reach here as non-contiguous strided views
        # (the public receiver outputs were produced by ``downsample_and_movedim``
        # *outside* this Function, so autograd routes their gradients back
        # through movedim's adjoint); the compiled backend reads them through a
        # raw ``data_ptr()`` assuming a C-contiguous ``[nt, n_shots,
        # n_receivers]`` layout, so they MUST be made contiguous first or a
        # multi-receiver shot injects the wrong adjoint-source samples (mirrors
        # deepwave.acoustic.AcousticBackwardFunc's own grad_receivers
        # contiguity fix).
        grad_receivers_all = [
            g.contiguous() for g in args[n_wf_total : n_wf_total + n_rec]
        ]
        del args

        saved = list(ctx.saved_tensors)
        n_models = 2 * n_bg
        models = saved[:n_models]
        sources_i = saved[n_models : n_models + n_src]
        receivers_i = saved[n_models + n_src : n_models + n_src + n_rec]
        source_amplitudes = saved[
            n_models + n_src + n_rec : n_models + 2 * n_src + n_rec
        ]
        pml_profiles = saved[n_models + 2 * n_src + n_rec :]

        # The compiled adjoint reads the models (K/dK at pressure nodes, each
        # buoyancy/dbuoyancy at its velocity nodes) at neighbour offsets near a
        # reflecting face, so the guard ring must hold the model's even/wrap
        # image, exactly as ``acoustic.AcousticBackwardFunc``
        # does for plain acoustic -- applied independently to the background
        # ([K, B_*]) and scattered ([dK, dB_*]) model groups, since each plays
        # the same "pressure"/"velocity" role on the same staggered grid (see
        # the module docstring). All-PML boundaries are a no-op (returns the
        # inputs unchanged); fresh clones are returned so the (possibly
        # aliased) saved user model tensors are never written in place.
        boundary = ctx.boundary
        fd_pad = ctx.accuracy // 2
        models = _boundaries.mirror_pad_acoustic_models(
            models[:n_bg], boundary, fd_pad
        ) + _boundaries.mirror_pad_acoustic_models(models[n_bg:], boundary, fd_pad)

        # Which forward inputs need a gradient. The forward's positional inputs
        # are: the leading non-tensor args (``_N_LEAD``), then models
        # [K, B*, dK, dB*] (2*n_bg), source amplitudes (2*n_bg), sources_i
        # (2*n_bg), receivers_i (2*n_bg), then the wavefields (background block
        # then scattered block).
        nig = ctx.needs_input_grad
        lead = _N_LEAD
        need_bg_model = any(nig[lead : lead + n_bg])
        need_bg_src = any(nig[lead + 2 * n_bg : lead + 3 * n_bg])
        wf_start = lead + 8 * n_bg
        need_bg_wf = any(nig[wf_start : wf_start + n_bg_wf])
        need_bg_adjoint = need_bg_model or need_bg_src or need_bg_wf

        lead_t: tuple[Optional[torch.Tensor], ...] = (None,) * lead
        # sources_i then receivers_i: integer indices, never differentiable.
        loc_grads = (None,) * n_src + (None,) * n_rec

        def grad_model_batched(models: list[torch.Tensor], ndim: int) -> list[bool]:
            return [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models]

        if need_bg_adjoint:
            # Full two-field adjoint: propagate both the background and the
            # scattered adjoint fields, accumulating all model, source, and
            # initial-wavefield gradients.
            def make_aux_wavefields(
                grad_wf: list[torch.Tensor], ndim: int
            ) -> list[torch.Tensor]:
                # psi ping-pong temporaries: background block then scattered.
                return [torch.zeros_like(grad_wf[0]) for _ in range(ndim)] + [
                    torch.zeros_like(grad_wf[n_bg_wf]) for _ in range(ndim)
                ]

            def swap(state: _engine.BackendState) -> None:
                # Each state block is [pressure, velocities, phi, psi]. Only
                # psi is ping-ponged by the native adjoint.
                w = state.wavefields
                a = state.aux_wavefields
                w[1 + 2 * ndim : 1 + 3 * ndim], a[:ndim] = (
                    a[:ndim],
                    w[1 + 2 * ndim : 1 + 3 * ndim],
                )
                (
                    w[n_bg_wf + 1 + 2 * ndim : n_bg_wf + 1 + 3 * ndim],
                    a[ndim : 2 * ndim],
                ) = (
                    a[ndim : 2 * ndim],
                    w[n_bg_wf + 1 + 2 * ndim : n_bg_wf + 1 + 3 * ndim],
                )

            state = _engine.run_backward_core(
                SPEC,
                models=models,
                sources_i=sources_i,
                receivers_i=receivers_i,
                source_amplitudes=[],
                pml_profiles=pml_profiles,
                grad_wavefields=grad_wavefields_all,
                grad_receivers=grad_receivers_all,
                grid_spacing=list(ctx.grid_spacing),
                dt=ctx.dt,
                nt=ctx.nt,
                n_shots=ctx.n_shots,
                step_ratio=ctx.step_ratio,
                accuracy=ctx.accuracy,
                pml_width=ctx.pml_width,
                source_amplitudes_requires_grad=list(
                    ctx.source_amplitudes_requires_grad
                ),
                storage_manager=ctx.storage_manager,
                backward_callback=ctx.backward_callback,
                callback_frequency=ctx.callback_frequency,
                grad_model_batched=grad_model_batched,
                make_aux_wavefields=make_aux_wavefields,
                zero_wavefields=zero_pml_interiors_born,
                make_call=_make_backward_call(ndim),
                swap=swap,
                boundary=boundary,
            )
            out_s = state.output_slice
            return (
                lead_t
                + tuple(state.grad_models)
                + tuple(state.grad_sources)
                + loc_grads
                + tuple(w[out_s] for w in state.wavefields)
            )

        # Scattered-only adjoint: only dv/drho and/or scattered initial
        # wavefields need gradients.
        grad_wavefields_sc = grad_wavefields_all[n_bg_wf:]
        grad_receivers_sc = grad_receivers_all[n_bg:]
        sources_i_sc = sources_i[n_bg:]
        receivers_i_sc = receivers_i[n_bg:]
        source_amplitudes_sc = source_amplitudes[n_bg:]
        source_amplitudes_requires_grad_sc = list(
            ctx.source_amplitudes_requires_grad[n_bg:]
        )

        def make_aux_wavefields_sc(
            grad_wf: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            return [torch.zeros_like(grad_wf[0]) for _ in range(ndim)]

        def swap_sc(state: _engine.BackendState) -> None:
            # The trailing psi block owns the native ping-pong buffer.
            state.wavefields[-ndim:], state.aux_wavefields[:] = (
                state.aux_wavefields[:],
                state.wavefields[-ndim:],
            )

        state = _engine.run_backward_core(
            SPEC,
            models=models,
            sources_i=sources_i_sc,
            receivers_i=receivers_i_sc,
            source_amplitudes=source_amplitudes_sc,
            pml_profiles=pml_profiles,
            grad_wavefields=grad_wavefields_sc,
            grad_receivers=grad_receivers_sc,
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
            grad_model_batched=grad_model_batched,
            make_aux_wavefields=make_aux_wavefields_sc,
            zero_wavefields=_acoustic_module.zero_edges_and_interiors,
            make_call=_make_backward_sc_call(ndim),
            swap=swap_sc,
            wavefield_names=_acoustic_module.acoustic_wavefield_names(ndim),
            backend_extra="_sc",
            boundary=boundary,
        )

        # grad w.r.t. background (K, B*) is zero on this path.
        model_grads = (None,) * n_bg + tuple(state.grad_models[n_bg:])
        src_grads: tuple[Optional[torch.Tensor], ...] = (None,) * n_bg + tuple(
            state.grad_sources
        )
        wf_grads_bg = (None,) * n_bg_wf
        out_s = state.output_slice
        wf_grads_sc = tuple(w[out_s] for w in state.wavefields)
        return lead_t + model_grads + src_grads + loc_grads + wf_grads_bg + wf_grads_sc


# Number of leading non-tensor arguments of ``AcousticBornForwardFunc.forward``,
# all of which receive ``None`` gradients in ``backward``.
_N_LEAD = _engine.leading_arg_count(AcousticBornForwardFunc.forward)


def acoustic_born_func(*args: Any) -> tuple[torch.Tensor, ...]:
    """Helper function to apply :class:`AcousticBornForwardFunc`."""
    func = AcousticBornForwardFunc.apply
    return cast("tuple[torch.Tensor, ...]", func(*args)[:-1])  # type: ignore[no-untyped-call]


def _update_velocities_born(
    bg_models: list[torch.Tensor],
    sc_models: list[torch.Tensor],
    p: torch.Tensor,
    psc: torch.Tensor,
    velocities: list[torch.Tensor],
    velocitiessc: list[torch.Tensor],
    psi: list[torch.Tensor],
    psisc: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    """One coupled Born velocity update (background then scattered).

    Background: ``v_d -= dt * B_d * (grad_d(p) + psi_d)`` -- identical to
    ``acoustic.update_velocities``. Scattered: the same acoustic
    update on ``(dp, dv_d)`` with the background ``B_d``, plus the Born virtual
    source ``dt * dB_d * term_bg_d`` where ``term_bg_d = grad_d(p) + psi_d`` is
    the background field's own effective pressure gradient (see
    ``docs/acoustic_born_derivation.md`` and ``acoustic_born.c``'s forward).
    """
    ndim = len(grid_spacing)
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    buoyancy = bg_models[1:]
    dbuoyancy = sc_models[1:]
    new_velocities = list(velocities)
    new_velocitiessc = list(velocitiessc)
    for dim in range(ndim):
        gp = _staggered_grid.diff1h(p, dim, accuracy, 1 / grid_spacing[dim], ndim)
        psi[dim] = ah[dim] * psi[dim] + bh[dim] * gp
        term = gp + psi[dim]
        new_velocities[dim] = velocities[dim] - dt * buoyancy[dim] * term

        gpsc = _staggered_grid.diff1h(psc, dim, accuracy, 1 / grid_spacing[dim], ndim)
        psisc[dim] = ah[dim] * psisc[dim] + bh[dim] * gpsc
        termsc = gpsc + psisc[dim]
        new_velocitiessc[dim] = (
            velocitiessc[dim] - dt * buoyancy[dim] * termsc - dt * dbuoyancy[dim] * term
        )
    return new_velocities, new_velocitiessc, psi, psisc


def _update_pressure_born(
    bg_models: list[torch.Tensor],
    sc_models: list[torch.Tensor],
    p: torch.Tensor,
    psc: torch.Tensor,
    velocities: list[torch.Tensor],
    velocitiessc: list[torch.Tensor],
    phi: list[torch.Tensor],
    phisc: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """One coupled Born pressure update (background then scattered).

    Background: ``p -= dt * K * (div(v) + phi)`` -- identical to
    ``acoustic.update_pressure``. Scattered: the same acoustic
    update on ``dp`` with the background ``K``, plus the Born virtual source
    ``dt * dK * div_bg`` where ``div_bg = div(v) + phi`` is the background
    field's own effective velocity divergence.
    """
    ndim = len(grid_spacing)
    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    k = bg_models[0]
    dk = sc_models[0]

    div_v = torch.zeros_like(p)
    div_vsc = torch.zeros_like(psc)
    for dim in range(ndim):
        dv = _staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        phi[dim] = a[dim] * phi[dim] + b[dim] * dv
        div_v = div_v + dv + phi[dim]

        dvsc = _staggered_grid.diff1(
            velocitiessc[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        phisc[dim] = a[dim] * phisc[dim] + b[dim] * dvsc
        div_vsc = div_vsc + dvsc + phisc[dim]

    p = p - dt * k * div_v
    psc = psc - dt * k * div_vsc - dt * dk * div_v
    return p, psc, phi, phisc


# ``torch.compile`` Python-backend slots: the coupled velocity/pressure
# step functions above are already standalone (no closures over Python
# wavefield state -- every tensor is an explicit argument), so they can be
# ``torch.compile``d directly, mirroring
# :mod:`deepwave.elastic_vti`'s ``_update_velocities_opt``/
# ``_update_stresses_opt`` slots.
_update_velocities_born_cache = _engine.StepCache(_update_velocities_born)
_update_velocities_born_opt: Any = _update_velocities_born
_update_pressure_born_cache = _engine.StepCache(_update_pressure_born)
_update_pressure_born_opt: Any = _update_pressure_born


def acoustic_born_python(
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
    callback_frequency: int,
    *args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Pure-PyTorch coupled acoustic Born forward driver (``backend``).

    Runs the same coupled background+scattered K/buoyancy time-stepping as the
    compiled ``acoustic_born.c`` forward, but in native PyTorch so gradients (and
    ``torch.func`` transforms such as ``vmap``/``jvp``) flow through autograd
    rather than the hand-written adjoint kernels. Mirrors the structure of
    ``acoustic.acoustic_python`` (and
    ``elastic_born.elastic_born_eager_core``) doubled for the two
    coupled fields.

    ``forward_callback``, if not None, is called every ``callback_frequency``
    coarse steps (of ``step_ratio`` fine time steps each) with a
    :class:`deepwave.CallbackState` whose wavefields dict is keyed by
    :func:`acoustic_born_wavefield_names` and whose models dict is keyed by
    :func:`acoustic_born_model_names`, mirroring
    ``acoustic.acoustic_python``'s own callback firing. There is
    no ``backward_callback`` parameter: this backend runs under plain
    autograd, so it has no adjoint pass of its own to hook into (the caller
    rejects a ``backward_callback`` before reaching here).

    ``boundary`` is the per-face integer code list (see
    :func:`deepwave._common.set_boundary`); ``BC_PML`` faces are absorbing and
    the guard ring is left untouched. Non-PML faces have the background and
    scattered fields' pressure and boundary-normal velocity guard cells
    refilled with the exact image each step (free/rigid/periodic), applied
    independently to each field via :func:`deepwave._boundaries.\
fill_acoustic_pressure_boundary`/:func:`deepwave._boundaries.\
fill_acoustic_velocity_boundary` (the same functions
    ``acoustic.acoustic_python`` uses -- both fields are an ordinary acoustic
    staggered-grid field on the same grid).

    ``args`` is the flat tensor tuple ``(*models, *source_amplitudes,
    *sources_i, *receivers_i, *wavefields)`` where each of ``models``,
    ``source_amplitudes``, ``sources_i``, ``receivers_i`` is the background block
    (``ndim + 1`` entries) followed by the scattered block, and ``wavefields`` is
    the background wavefields (``1 + 3 * ndim``) followed by the scattered ones.
    The ``source_amplitudes`` are already scaled (background by ``dt * K`` /
    ``dt * B_d``, scattered by ``dt * dK`` / ``dt * dB_d``) by
    :func:`acoustic_born`.

    Returns the final background wavefields, then the final scattered
    wavefields, then the background receiver amplitudes, then the scattered
    receiver amplitudes -- in :func:`acoustic_born_output_names` order.
    """
    ndim = len(grid_spacing)
    n_bg = ndim + 1
    n_bg_wf = 1 + 3 * ndim
    num_models = 2 * n_bg
    num_source_types = 2 * n_bg
    num_receiver_types = 2 * n_bg
    num_wf = 2 * n_bg_wf

    args_list = list(args)
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

    bg_models = models[:n_bg]
    sc_models = models[n_bg:]
    k = bg_models[0]

    is_batched = any(
        _common.is_inside_vmap(x)
        for x in [*models, *source_amplitudes, *sources_i, *receivers_i, *wavefields]
    )

    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    has_boundary = any(code != _common.BC_PML for code in boundary)
    device = k.device
    dtype = k.dtype
    shape = k.shape[-ndim:]
    flat_shape = int(torch.prod(torch.tensor(shape)).item())
    size_with_batch = (n_shots, *shape)

    wavefields = [
        _engine.create_or_pad(w, fd_pad_list, device, dtype, size_with_batch)
        for w in wavefields
    ]
    zero_pml_interiors_born(wavefields, ndim, fd_pad, fd_pad_list, pml_width)

    # Background then scattered field views (acoustic_wavefield_names order:
    # pressure, velocities, phi, psi).
    p = wavefields[0]
    velocities = wavefields[1 : 1 + ndim]
    phi = wavefields[1 + ndim : 1 + 2 * ndim]
    psi = wavefields[1 + 2 * ndim : 1 + 3 * ndim]
    psc = wavefields[n_bg_wf]
    velocitiessc = wavefields[n_bg_wf + 1 : n_bg_wf + 1 + ndim]
    phisc = wavefields[n_bg_wf + 1 + ndim : n_bg_wf + 1 + 2 * ndim]
    psisc = wavefields[n_bg_wf + 1 + 2 * ndim : n_bg_wf + 1 + 3 * ndim]

    # Receiver buffers: background block (indices 0..n_bg-1) then scattered.
    n_recv_per_shot = [locs.numel() // n_shots for locs in receivers_i]
    receiver_amplitudes = [
        torch.empty(0, device=device, dtype=dtype) for _ in receivers_i
    ]
    receiver_amplitudes_lists: list[list[torch.Tensor]] = [[] for _ in receivers_i]
    for i, (locs, n_per_shot) in enumerate(zip(receivers_i, n_recv_per_shot)):
        if locs.numel() > 0 and not is_batched:
            receiver_amplitudes[i].resize_(nt, n_shots, n_per_shot)
            receiver_amplitudes[i].fill_(0)
    masked_receivers = []
    for i, locs in enumerate(receivers_i):
        receivers_i_masked, recv_mask = _common.mask_locations(locs)
        masked_receivers.append((receivers_i_masked, recv_mask, i))

    # Source injection: same block layout; amplitudes already scaled.
    masked_sources = []
    for i in range(num_source_types):
        sources_i_masked, src_mask = _common.mask_locations(sources_i[i])
        src_amp_masked = _common.mask_source_amplitudes(source_amplitudes[i], src_mask)
        masked_sources.append((sources_i_masked, src_amp_masked, i))

    wf_names = acoustic_born_wavefield_names(ndim)
    model_names = acoustic_born_model_names(ndim)
    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            current_wavefields = [
                p,
                *velocities,
                *phi,
                *psi,
                psc,
                *velocitiessc,
                *phisc,
                *psisc,
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
                and any(t.requires_grad for t in args if isinstance(t, torch.Tensor)),
            )
        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step
            # Refill the pressure guard ring of each field independently (read by
            # the velocity update); this also holds free-surface pressure nodes
            # to zero before recording (no-op when every face is PML).
            if has_boundary:
                p = _boundaries.fill_acoustic_pressure_boundary(p, boundary, fd_pad)
                psc = _boundaries.fill_acoustic_pressure_boundary(psc, boundary, fd_pad)

            # Record receivers (background field into the bg block, scattered field
            # into the sc block) BEFORE the updates, matching acoustic_born.c.
            for receivers_i_masked, _, i in masked_receivers:
                if receivers_i_masked.numel() == 0:
                    continue
                block = i // n_bg
                type_idx = i % n_bg
                if block == 0:
                    field = p if type_idx == 0 else velocities[type_idx - 1]
                else:
                    field = psc if type_idx == 0 else velocitiessc[type_idx - 1]
                val = field.view(-1, flat_shape).gather(1, receivers_i_masked)
                if is_batched:
                    receiver_amplitudes_lists[i].append(val)
                else:
                    receiver_amplitudes[i][t] = val

            # Coupled velocity update.
            velocities, velocitiessc, psi, psisc = _update_velocities_born_opt(
                bg_models,
                sc_models,
                p,
                psc,
                velocities,
                velocitiessc,
                psi,
                psisc,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            # Inject velocity sources (background then scattered).
            new_velocities = list(velocities)
            new_velocitiessc = list(velocitiessc)
            for sources_i_masked, src_amp_masked, i in masked_sources:
                type_idx = i % n_bg
                if not (0 < type_idx <= ndim and src_amp_masked.numel() > 0):
                    continue
                v_idx = type_idx - 1
                target = new_velocities if i // n_bg == 0 else new_velocitiessc
                target[v_idx] = (
                    target[v_idx]
                    .reshape(-1, flat_shape)
                    .scatter_add(1, sources_i_masked, src_amp_masked[t])
                    .view(size_with_batch)
                )
            velocities = new_velocities
            velocitiessc = new_velocitiessc

            # Refill the boundary-normal velocity guard ring of each field
            # independently (read by the pressure update); holds rigid-wall
            # velocity nodes to zero (no-op for all-PML).
            if has_boundary:
                velocities = _boundaries.fill_acoustic_velocity_boundary(
                    list(velocities), boundary, fd_pad
                )
                velocitiessc = _boundaries.fill_acoustic_velocity_boundary(
                    list(velocitiessc), boundary, fd_pad
                )

            # Coupled pressure update.
            p, psc, phi, phisc = _update_pressure_born_opt(
                bg_models,
                sc_models,
                p,
                psc,
                velocities,
                velocitiessc,
                phi,
                phisc,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            # Inject pressure sources (background then scattered).
            for sources_i_masked, src_amp_masked, i in masked_sources:
                if i % n_bg != 0 or src_amp_masked.numel() == 0:
                    continue
                injected = (
                    (p if i // n_bg == 0 else psc)
                    .reshape(-1, flat_shape)
                    .scatter_add(1, sources_i_masked, src_amp_masked[t])
                    .view(size_with_batch)
                )
                if i // n_bg == 0:
                    p = injected
                else:
                    psc = injected

    if is_batched:
        for i in range(len(receiver_amplitudes)):
            if len(receiver_amplitudes_lists[i]) > 0:
                receiver_amplitudes[i] = torch.stack(receiver_amplitudes_lists[i])
    for _, recv_mask, i in masked_receivers:
        receiver_amplitudes[i] = _common.mask_receiver_amplitudes(
            receiver_amplitudes[i], recv_mask
        )

    # Leave the returned wavefields boundary-consistent (e.g. free-surface
    # pressure nodes held to zero, guard rings imaged) so a continuation run
    # reproduces the in-loop state, matching acoustic_python.
    if has_boundary:
        p = _boundaries.fill_acoustic_pressure_boundary(p, boundary, fd_pad)
        psc = _boundaries.fill_acoustic_pressure_boundary(psc, boundary, fd_pad)
        velocities = _boundaries.fill_acoustic_velocity_boundary(
            list(velocities), boundary, fd_pad
        )
        velocitiessc = _boundaries.fill_acoustic_velocity_boundary(
            list(velocitiessc), boundary, fd_pad
        )

    out_wavefields = [
        p,
        *velocities,
        *phi,
        *psi,
        psc,
        *velocitiessc,
        *phisc,
        *psisc,
    ]
    s = (slice(None), *(slice(fd_pad, dim_size - (fd_pad - 1)) for dim_size in shape))
    return (*[w[s] for w in out_wavefields], *receiver_amplitudes)


_common.document_propagator_module(AcousticBorn, acoustic_born)

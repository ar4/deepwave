"""Scalar Born wave propagation module.

Implements Born forward modelling (and its adjoint for backpropagation)
of the scalar wave equation. The implementation is similar to that
described in the scalar module, with the addition of a scattered
wavefield that uses 2 / v * scatter * dt^2 * wavefield as the source term.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union, cast

import torch

import deepwave
from deepwave import (
    _backend_utils,
    _boundaries,
    _common,
    _engine,
    _exceptions,
    _regular_grid,
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


def scalar_born_model_names(ndim: int) -> list[str]:
    """Names of the scalar Born models (background velocity and scatterer)."""
    return ["v", "dv"]


# The user-facing model arguments, in the order the three functions below
# expect their ``model_requires_grad`` sequence.
USER_MODEL_NAMES = ("v", "dv")


def internal_model_grad_flags(
    model_requires_grad: Sequence[bool], ndim: int
) -> list[bool]:
    """``requires_grad`` of each internal model, from the user-facing flags.

    ``model_requires_grad`` is ``(v_requires_grad, dv_requires_grad)``; the
    result is in :func:`scalar_born_model_names` order. The internal models
    are the user's ``v`` and ``dv`` (the wavespeed perturbation) themselves.
    """
    v_requires_grad, dv_requires_grad = model_requires_grad
    return [v_requires_grad, dv_requires_grad]


def storage_slot_flags(model_requires_grad: Sequence[bool], ndim: int) -> list[bool]:
    """Which backward-storage slots the forward pass allocates.

    The background wavefield history (``w_store``) is needed for either
    model's gradient; the scattered history (``wsc_store``) only for ``v``'s.
    Note that this is *not* one slot per model: with only ``dv`` requiring a
    gradient just one slot is stored.

    The single source of truth for scalar_born's storage-slot count, shared by
    the forward pass (``ScalarBornForwardFunc.forward``'s ``storage_flags``)
    and by :func:`deepwave.estimate_memory`, so the two cannot drift.
    """
    v_requires_grad, dv_requires_grad = model_requires_grad
    return [v_requires_grad or dv_requires_grad, v_requires_grad]


def user_model_grad_flags(models: Sequence[torch.Tensor], ndim: int) -> list[bool]:
    """The user-facing ``requires_grad`` flags, recovered from the internal models.

    The inverse of :func:`internal_model_grad_flags`, for callers (the
    out-of-memory message in :mod:`deepwave._engine`) that hold the internal
    models rather than the user's.
    """
    return [models[0].requires_grad, models[1].requires_grad]


def scalar_born_wavefield_names(ndim: int) -> list[str]:
    """Names of the scalar Born wavefields, in backend order.

    The background fields (current, previous, then ``psi``/``zeta`` per
    dimension) are followed by the corresponding scattered fields.
    """
    dims = _dim_names(ndim)
    background = (
        ["wavefield_0", "wavefield_m1"]
        + [f"psi{d}_m1" for d in dims]
        + [f"zeta{d}_m1" for d in dims]
    )
    scattered = (
        ["wavefield_sc_0", "wavefield_sc_m1"]
        + [f"psi{d}_sc_m1" for d in dims]
        + [f"zeta{d}_sc_m1" for d in dims]
    )
    return background + scattered


def scalar_born_pml_slab_axes(ndim: int) -> dict[str, int]:
    """PML memory-variable wavefields and the axis each is localised along.

    Both the background and the scattered ``psi{d}``/``zeta{d}`` sets are
    nonzero only within that dimension's two PML slabs: the compiled kernels
    write them only where that axis's PML flag is set (the ``if (pml_z)``
    gating in ``scalar_born_kernels.h``, mirroring ``scalar_kernels.h``), and
    :func:`_zero_born_pml_interiors` (forward and background backward) and
    :func:`_zero_pml_interiors_sc` (scattered-only backward) zero each field's
    interior along its own dimension on every entry.
    """
    dims = _dim_names(ndim)
    axes: dict[str, int] = {}
    for i, d in enumerate(dims):
        axes[f"psi{d}_m1"] = i
        axes[f"zeta{d}_m1"] = i
        axes[f"psi{d}_sc_m1"] = i
        axes[f"zeta{d}_sc_m1"] = i
    return axes


def scalar_born_auxiliary_state_names(ndim: int) -> list[str]:
    """Names of the wavefields carried opaquely in the auxiliary state.

    The CFS-CPML memory variables ``psi{d}_m1`` and ``zeta{d}_m1`` of the
    background field and their ``_sc`` counterparts of the scattered field,
    one of each per spatial dimension. They are a subset of
    :func:`scalar_born_wavefield_names`, spelled the same way, and are what
    :class:`deepwave.AuxiliaryState` holds.
    """
    dims = _dim_names(ndim)
    return (
        [f"psi{d}_m1" for d in dims]
        + [f"zeta{d}_m1" for d in dims]
        + [f"psi{d}_sc_m1" for d in dims]
        + [f"zeta{d}_sc_m1" for d in dims]
    )


def scalar_born_output_names(ndim: int) -> list[str]:
    """Names of the public outputs of :func:`scalar_born`, in return order.

    The public final wavefields -- the background and scattered physical
    wavefields, with the single ``auxiliary_state`` that replaces both blocks
    of PML memory variables at the position of the first of them -- are
    followed by the background receiver amplitudes and then the scattered
    receiver amplitudes (the background data is returned first, the reverse of
    the input ordering).
    """
    return [
        *_engine.collapse_auxiliary_state_names(
            scalar_born_wavefield_names(ndim),
            scalar_born_auxiliary_state_names(ndim),
        ),
        "bg_receiver_amplitudes",
        "sc_receiver_amplitudes",
    ]


# Declarative description of the scalar Born propagator: a regular-grid solver
# with background and scattered fields, two models, and (background, scattered)
# source/receiver types.
SPEC = _engine.PropagatorSpec(
    name="scalar_born",
    status="stable",
    source_channels=_engine.scalar_channels,
    receiver_channels=_engine.scalar_channels,
    bg_receiver_channels=_engine.scalar_channels,
    channel_suffixes=False,
    staggered=False,
    model_names=scalar_born_model_names,
    wavefield_names=scalar_born_wavefield_names,
    n_source_types=lambda ndim: 2,
    n_receiver_types=lambda ndim: 2,
    output_names=scalar_born_output_names,
    auxiliary_state_names=scalar_born_auxiliary_state_names,
    pml_slab_axes=scalar_born_pml_slab_axes,
    spatial_parallel=True,
)


def _zero_born_pml_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
) -> None:
    """Zeros the interiors of the background and scattered psi/zeta auxiliaries.

    The wavefields are ordered ``[background(current, previous, psi, zeta),
    scattered(current, previous, psi, zeta)]``. psi and zeta are non-zero only
    inside the PML region, so their interiors are zeroed and the backend skips
    PML work there.
    """
    sc = 2 + 2 * ndim
    for i in range(ndim):
        for base in (2, sc + 2):  # psi and psisc
            wavefields[base + i] = _common.zero_interior(
                wavefields[base + i], fd_pad, pml_width, i
            )
        for base in (2 + ndim, sc + 2 + ndim):  # zeta and zetasc
            wavefields[base + i] = _common.zero_interior(
                wavefields[base + i], fd_pad, pml_width, i
            )


def _zero_pml_interiors_sc(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
) -> None:
    """Zeros the psi/zeta interiors of a single regular-grid field set.

    Used for the scattered-only backward pass, whose gradient wavefields are
    the lone ``[current, previous, psi, zeta]`` scattered subset.
    """
    for i in range(ndim):
        wavefields[2 + i] = _common.zero_interior(
            wavefields[2 + i], fd_pad, pml_width, i
        )
        wavefields[2 + ndim + i] = _common.zero_interior(
            wavefields[2 + ndim + i], fd_pad, pml_width, i
        )


class ScalarBorn(torch.nn.Module):
    """A Module wrapper around :func:`scalar_born`.

    Stability: stable (see :doc:`/stability`).

    This is a convenience module that allows you to only specify
    `v`, `dv`, and `grid_spacing` once. They will then be added to the
    list of arguments passed to :func:`scalar_born` when you call the
    forward method.

    The `scatter` attribute is a deprecated read-only alias for `dv`.
    **Removed in Deepwave 0.4.0** (see the stability policy documentation
    page).

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: An initial guess of the wavespeed model, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dv: An initial guess of the wavespeed scattering perturbation, of
            shape ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. At least one of
            `dv`/`scatter` must be provided.
        scatter: Deprecated alias for `dv`, retained for backward
            compatibility. **Removed in Deepwave 0.4.0** (see the stability
            policy documentation page). It takes exactly the shapes `dv`
            takes, batched or not. Providing both `dv` and `scatter` raises
            `ValueError`.
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
        v_requires_grad: A bool specifying whether the `requires_grad`
            attribute of the wavespeed should be set. Unset by default,
            which means False. Cannot be combined with `requires_grad`.
        dv_requires_grad: A bool specifying whether the `requires_grad`
            attribute of the wavespeed perturbation should be set. Unset by
            default, which means False. Cannot be combined with
            `requires_grad`.
        scatter_requires_grad: Deprecated alias for `dv_requires_grad`.
            **Removed in Deepwave 0.4.0** (see the stability policy
            documentation page). Providing it without `scatter`,
            or together with `dv_requires_grad`, raises `ValueError`.
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
        grid_spacing: Union[float, Sequence[float]],
        *,
        dv: Optional[torch.Tensor] = None,
        scatter: Optional[torch.Tensor] = None,
        requires_grad: Optional[Union[bool, Sequence[str]]] = None,
        v_requires_grad: Optional[bool] = None,
        dv_requires_grad: Optional[bool] = None,
        scatter_requires_grad: Optional[bool] = None,
        storage: StorageLike = STORAGE_DEFAULT,
        storage_mode: Optional[StorageMode] = None,
        storage_path: Optional[StoragePath] = None,
        storage_compression: Optional[bool] = None,
    ) -> None:
        """Initialises the Module."""
        super().__init__()
        # ScalarBorn(v, scatter, dx) is the v0.0.27 spelling; with `scatter`
        # now keyword-only the model lands in `grid_spacing`, so name that
        # before any other validation runs.
        _common.reject_model_as_grid_spacing(grid_spacing, v.ndim, "scalar_born")
        # `scatter` is the v0.0.27 name for the wavespeed perturbation,
        # retained as a deprecated alias for `dv`, matching the
        # scalar_born() functional interface.
        if scatter is not None:
            _exceptions.warn_deprecated_name("scatter", "dv", stacklevel=2)
            if dv is not None:
                raise _exceptions.DeepwaveValueError(
                    "scalar_born: pass only dv, not both dv and the deprecated scatter."
                )
            dv = scatter
            if scatter_requires_grad is not None:
                # Its own warning: one deprecated name, one message, so that a
                # call passing both hears about both rather than only the one
                # that happens to be checked first.
                _exceptions.warn_deprecated_name(
                    "scatter_requires_grad", "dv_requires_grad", stacklevel=2
                )
                if dv_requires_grad is not None:
                    raise _exceptions.DeepwaveValueError(
                        "scalar_born: pass only dv_requires_grad, not both "
                        "dv_requires_grad and the deprecated "
                        "scatter_requires_grad."
                    )
                dv_requires_grad = scatter_requires_grad
        elif scatter_requires_grad is not None:
            # Previously ignored in silence, which quietly dropped the
            # caller's request for a gradient.
            raise _exceptions.DeepwaveValueError(
                "scalar_born: scatter_requires_grad was given without scatter. "
                "It is a "
                "deprecated alias for dv_requires_grad; pass that instead."
            )
        del scatter
        if dv is None:
            raise _exceptions.DeepwaveValueError(
                "scalar_born: at least one scattering model (dv) must be provided."
            )
        grad = _common.resolve_requires_grad(
            requires_grad,
            ["v", "dv"],
            ["v", "dv"],
            {"v": v_requires_grad, "dv": dv_requires_grad},
            "ScalarBorn",
        )
        self.v = _common.make_model_parameter(v, grad["v"], "v")
        self.dv = _common.make_model_parameter(dv, grad["dv"], "dv")
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

    @property
    def scatter(self) -> torch.Tensor:
        """Deprecated alias for the `dv` attribute, retained for compatibility.

        **Removed in Deepwave 0.4.0** (see the stability policy documentation
        page). Read-only: assign to `dv` instead.
        """
        _exceptions.warn_deprecated_name("scatter", "dv", stacklevel=2)
        return self.dv

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Loads a state dict, remapping the ``v0.0.27`` ``scatter`` key to ``dv``.

        The scattering model was called ``scatter`` in ``v0.0.27`` and is
        called ``dv`` now, so a checkpoint saved by ``v0.0.27`` would
        otherwise fail to load with ``Missing key(s) in state_dict: "dv".
        Unexpected key(s) in state_dict: "scatter".`` -- a break that an
        attribute alias cannot cover, because ``load_state_dict`` matches on
        the key. The key is remapped here instead, with a
        :class:`deepwave.DeepwaveFutureWarning`; the remap is removed in
        Deepwave 0.4.0. That category subclasses ``FutureWarning``, which
        matters here: this hook is called by PyTorch rather than by the user,
        so the warning is attributed to ``torch``, and Python's default
        filters would hide a ``DeprecationWarning`` raised from there.

        The remap is deliberately skipped when ``dv`` is already present: a
        checkpoint carrying both keys is a user error, and
        ``load_state_dict``'s own ``unexpected_keys`` reporting is the right
        place for it to surface. ``prefix`` is respected, so a ``ScalarBorn``
        nested inside a larger module is handled, and ``strict=False`` is
        unaffected.

        Args:
            state_dict: The state dict being loaded, mutated in place (which
                is what PyTorch's own renamed-parameter shims do).
            prefix: This module's prefix within ``state_dict``.
            *args: The remaining positional arguments PyTorch passes through.
            **kwargs: The remaining keyword arguments PyTorch passes through.
        """
        old_key = prefix + "scatter"
        new_key = prefix + "dv"
        if old_key in state_dict and new_key not in state_dict:
            # DeepwaveFutureWarning subclasses FutureWarning, not
            # DeprecationWarning: this hook is called by
            # torch.nn.Module.load_state_dict, so no fixed stacklevel reaches
            # the user's own frame (the recursion depth varies with module
            # nesting) and the warning is attributed to torch. CPython's
            # default filters un-ignore DeprecationWarning only for
            # __main__, so such a warning would never be seen.
            _exceptions.warn_deprecated_name(
                "The state_dict key 'scatter'",
                "'dv'",
                stacklevel=2,
                detail=(
                    "It has been loaded as 'dv'; re-save the checkpoint to "
                    "silence this."
                ),
            )
            state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        dt: float,
        *,
        sources: Optional[Mapping[str, SourceSpec]] = None,
        receivers: Optional[Mapping[str, ReceiverSpec]] = None,
        bg_receivers: Optional[Mapping[str, ReceiverSpec]] = None,
        source_amplitudes: Optional[torch.Tensor] = None,
        source_locations: Optional[LocationsLike] = None,
        source_mask: Optional[torch.Tensor] = None,
        bg_receiver_locations: Optional[LocationsLike] = None,
        bg_receiver_mask: Optional[torch.Tensor] = None,
        sc_receiver_locations: Optional[LocationsLike] = None,
        receiver_locations: Optional[LocationsLike] = None,
        sc_receiver_mask: Optional[torch.Tensor] = None,
        wavefield_0: Optional[torch.Tensor] = None,
        wavefield_m1: Optional[torch.Tensor] = None,
        wavefield_sc_0: Optional[torch.Tensor] = None,
        wavefield_sc_m1: Optional[torch.Tensor] = None,
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

        See :func:`scalar_born` for details (the constructor-owned `v`/`dv`/
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
        return scalar_born(
            self.v,
            self.grid_spacing,
            dt,
            dv=self.dv,
            source_mask=source_mask,
            bg_receiver_mask=bg_receiver_mask,
            sc_receiver_mask=sc_receiver_mask,
            sources=sources,
            receivers=receivers,
            bg_receivers=bg_receivers,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            bg_receiver_locations=bg_receiver_locations,
            sc_receiver_locations=sc_receiver_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            wavefield_sc_0=wavefield_sc_0,
            wavefield_sc_m1=wavefield_sc_m1,
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


def scalar_born(
    v: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    *,
    dv: Optional[torch.Tensor] = None,
    scatter: Optional[torch.Tensor] = None,
    sources: Optional[Mapping[str, SourceSpec]] = None,
    receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    bg_receivers: Optional[Mapping[str, ReceiverSpec]] = None,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[LocationsLike] = None,
    source_mask: Optional[torch.Tensor] = None,
    bg_receiver_locations: Optional[LocationsLike] = None,
    bg_receiver_mask: Optional[torch.Tensor] = None,
    sc_receiver_locations: Optional[LocationsLike] = None,
    receiver_locations: Optional[LocationsLike] = None,
    sc_receiver_mask: Optional[torch.Tensor] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    wavefield_sc_0: Optional[torch.Tensor] = None,
    wavefield_sc_m1: Optional[torch.Tensor] = None,
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
    """Scalar Born wave propagation (functional interface).

    Stability: stable (see :doc:`/stability`).

    This function performs Born forward modelling with the scalar wave
    equation. The outputs are differentiable with respect to the wavespeed,
    the wavespeed perturbation, the source amplitudes, and the initial
    wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    Most arguments and returns are the same as those of :func:`scalar`, so
    only those that are different will be described here.

    Deepwave orders spatial dimensions from slowest-varying to
    fastest-varying and attaches no physical meaning to any of them; the
    component suffixes count from the fastest axis, so a 2-D run uses
    ``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See
    :ref:`axis-convention`.

    Args:
        v: The wavespeed model, of shape ``[*model_shape]`` or, for a
            batched model, ``[n_shots, *model_shape]``, where
            ``model_shape`` has ``ndim`` entries. See :ref:`batched-models`.
        grid_spacing: The spatial grid cell size.
        dt: The temporal grid cell size. See :func:`scalar` for the
            unit-consistency requirement across ``v``/``grid_spacing``/``dt``.
        dv: The wavespeed scattering perturbation model, of shape
            ``[*model_shape]`` or, for a batched model,
            ``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim``
            entries. See :ref:`batched-models`. It is
            the same quantity as :func:`deepwave.acoustic_born`'s ``dv``,
            and the direction in which the Born forward linearises ``v``. The
            scattered field is exactly the directional derivative of
            :func:`scalar` with respect to ``v`` in this direction (the
            kernel's coupling term is ``2 v dt^2 dv w``, the first-order term
            of ``d(v^2) = 2 v dv``), and Deepwave's test suite pins the
            scattered field against a finite difference of :func:`scalar` in
            that direction.
            At least one of ``dv``/``scatter`` must be provided.
        scatter: Deprecated alias for ``dv``, retained for backward
            compatibility. **Removed in Deepwave 0.4.0** (see the stability
            policy documentation page). It takes exactly the shapes ``dv``
            takes, batched or not. Providing both ``dv`` and ``scatter``
            raises ``ValueError``.
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
        receivers: The receivers that record the **scattered** wavefield,
            keyed by channel, as the canonical alternative to the flat
            ``sc_receiver_locations``/``sc_receiver_mask`` keywords (of which
            ``receiver_locations`` is a deprecated spelling).
            Each value is a locations object, or ``(locations, mask)`` to
            include an activity mask. The one channel is ``u``. Giving the
            channel in both forms raises. Optional.
        bg_receivers: The receivers that record the **background**
            wavefield, keyed by channel, as the canonical alternative to the
            flat ``bg_receiver_locations``/``bg_receiver_mask`` keywords.
            Values take the same forms, and use the same one channel, as
            ``receivers``'. Giving the channel in both forms raises.
            Optional.
        source_amplitudes: A torch.Tensor with dimensions [shot, source, time]
            holding the background source amplitudes, which are the same
            quantity, with the same units, as :func:`deepwave.scalar`'s; see
            :doc:`/units`.
        source_locations: A torch.Tensor containing the source locations.
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
        sc_receiver_locations: A torch.Tensor containing the locations of the
            receivers that record the **scattered** wavefield. The ``sc_``
            spelling makes the name match the order of the returned tuple, in
            which the background data comes first and the scattered data
            second; the same spelling is used by
            :func:`deepwave.acoustic_born` and :func:`deepwave.elastic_born`.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        sc_receiver_mask: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers are active. ``True`` means
            active; ``False`` means that point records zeros, and its
            coordinates in ``sc_receiver_locations`` are then ignored entirely
            (they need not be valid, in range, or unique). The shape must match
            the leading dimensions of ``sc_receiver_locations``, which must be
            given. This is the modern spelling of the
            :data:`deepwave.IGNORE_LOCATION` sentinel and produces
            bit-identical results; supplying both a mask and a
            sentinel-containing locations tensor for the same component raises,
            as does combining a mask with a :class:`deepwave.Hicks` object: a
            ``Hicks`` carries its own already-resolved points, so disable
            points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        receiver_locations: Deprecated spelling of
            ``sc_receiver_locations``, which means exactly the same thing. It
            is the spelling ``v0.0.27`` shipped; it warns
            :class:`deepwave.DeepwaveFutureWarning` and will be removed in
            Deepwave 0.4.0. Passing both raises.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_locations: A torch.Tensor with dimensions
            [shot, receiver, ndim], containing the coordinates of the cell
            containing each receiver of the background wavefield. Optional.
            It should have torch.long (int64) datatype. If not provided,
            the output `bg_receiver_amplitudes` torch.Tensor will be empty.
            If backpropagation will be performed, the location of each
            background receiver must be unique within the same shot.
            A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
            ignored: it records zeros. A point with only *some* coordinates set to the
            sentinel raises. Compare against :data:`deepwave.IGNORE_LOCATION`; do not
            hard-code its value.
        bg_receiver_mask: A boolean torch.Tensor with dimensions [shot,
            receiver] selecting which receivers recording the background
            wavefield are active. ``True`` means active; ``False`` means that
            point records zeros, and its coordinates in
            ``bg_receiver_locations`` are then ignored entirely (they need not
            be valid, in range, or unique). The shape must match the leading
            dimensions of ``bg_receiver_locations``, which must be given. This
            is the modern spelling of the :data:`deepwave.IGNORE_LOCATION`
            sentinel and produces bit-identical results; supplying both a mask
            and a sentinel-containing locations tensor for the same component
            raises, as does combining a mask with a :class:`deepwave.Hicks`
            object: a ``Hicks`` carries its own already-resolved points, so
            disable points inside it by setting all of their coordinates to
            :data:`deepwave.IGNORE_LOCATION` in the locations it was built
            from. Optional.
        accuracy: See :func:`deepwave.scalar`.
        pml_width: See :func:`deepwave.scalar`.
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
            `"periodic"` (the face wraps around to the opposite face). Both the
            background and the scattered Born wavefields obey the chosen faces
            (mirroring is linear, so the scattered field uses the same image
            rules). A `"free"`, `"rigid"`, or `"periodic"` face requires
            `pml_width` to be zero on that face, and `"periodic"` must be set on
            both faces of a dimension. Unlike a bare zero-`pml_width` face (only
            an approximate reflector above accuracy 2), a `"free"`/`"rigid"`
            face fills the exterior with the exact odd/even image at every
            accuracy order, meeting the analytic reflection coefficient to
            better than 1%. See the boundary-conditions documentation for the
            precise ghost-plane conventions. Optional, default None.
            Availability: this propagator accepts all four names, on every
            face, at every accuracy.
        pml_freq: See :func:`deepwave.scalar`.
        max_vel: See :func:`deepwave.scalar`.
        courant_number: See :func:`deepwave.scalar`.
        survey_pad: See :func:`deepwave.scalar`.
        wavefield_0: The wavefield at time step 0. Like every initial
            wavefield, it may be provided either at the shape of the
            simulation domain (the extracted model region plus the PML),
            which is the shape a previous call returns, or at the shape of
            the extracted model region itself, which Deepwave zero-pads for
            you; see the `wavefield_0` entry of :func:`deepwave.scalar`.
        wavefield_m1: The wavefield at time step -1.
        wavefield_sc_0: The scattered wavefield at time step 0.
        wavefield_sc_m1: The scattered wavefield at time step -1.
        initial_state: See :func:`deepwave.scalar`. The mapping keys are this
            propagator's own public state field names (`wavefield_0`,
            `wavefield_m1`, `wavefield_sc_0`, `wavefield_sc_m1`) and
            `auxiliary_state`; its `bg_receiver_amplitudes` and
            `sc_receiver_amplitudes` fields are ignored. The individual
            absorbing-boundary field names (`psiz_m1`, ..., `zetax_sc_m1`) are
            still accepted as keys, with a `FutureWarning`; they will stop
            being accepted in Deepwave 0.4.0.
        origin: See :func:`deepwave.scalar`.
        nt: See :func:`deepwave.scalar`.
        model_gradient_sampling_interval: See :func:`deepwave.scalar`.
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
        freq_taper_frac: See :func:`deepwave.scalar`.
        time_pad_frac: See :func:`deepwave.scalar`.
        time_taper: See :func:`deepwave.scalar`.
        check_output_finite: See :func:`deepwave.scalar`.
        forward_callback: See :func:`deepwave.scalar`.
        backward_callback: See :func:`deepwave.scalar`.
        callback_frequency: See :func:`deepwave.scalar`.
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
        location_interpolation: See :func:`deepwave.scalar`. The single source
            interpolator is shared by the background and scattered source inputs, while
            the background and scattered receivers are gathered independently.
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
        location_interpolation_halfwidth: See :func:`deepwave.scalar`.

    Returns:
        deepwave.PropagatorOutput: a tuple subclass whose fields, in
        order, are:

            - wavefield_0: The non-scattered wavefield at the final time step.
            - wavefield_m1: The non-scattered wavefield at the
              second-to-last time step.
            - auxiliary_state: A :class:`deepwave.AuxiliaryState` holding the
              absorbing boundary's memory variables, for both the background
              and the scattered wavefield. Opaque: pass it back (as part of
              `initial_state`) to continue the simulation; its contents are
              implementation state rather than physics.
            - wavefield_sc_0: The scattered wavefield.
            - wavefield_sc_m1: The scattered wavefield.
            - bg_receiver_amplitudes: The receiver amplitudes recorded at the
              provided receiver locations, extracted from the background
              (non-scattered) wavefield.
            - sc_receiver_amplitudes: The receiver amplitudes recorded at
              the provided receiver locations, extracted from the scattered
              wavefield. ``out.receiver_amplitudes`` is a **permanent** alias
              of this field -- the spelling ``v0.0.27`` shipped, kept for good
              and never warning -- but ``out.names`` and ``out.as_dict()``
              report the ``sc_`` spelling.

            The names match the order: the **background** receiver amplitudes
            (``bg_receiver_amplitudes``) come first and the **scattered** ones
            (``sc_receiver_amplitudes``) second, so ``out[-2]`` is the
            background data and ``out[-1]`` is the scattered data. The same
            ``bg_``/``sc_`` pairing is used by
            :func:`deepwave.acoustic_born` and :func:`deepwave.elastic_born`,
            and by the ``bg_receivers=``/``receivers=`` arguments: note that
            ``receivers=`` and :attr:`deepwave.PropagatorOutput.receivers`
            both mean the *scattered* data.

            The fields are also accessible by name, which removes the need to
            remember the positional order: the wavefields use their
            initial-state names (``wavefield_0``, ``wavefield_m1``, ...,
            ``wavefield_sc_0``, ...) and the receiver data is exposed as
            ``bg_receiver_amplitudes`` (background) and
            ``sc_receiver_amplitudes`` (scattered).

    """
    # Before anything else, including the "no scattering model" check: a
    # v0.0.27 call spelled scalar_born(v, scatter, dx, dt=dt) binds the
    # scattering model to grid_spacing, and every message downstream of here
    # would blame something other than the argument the caller got wrong.
    _common.reject_model_as_grid_spacing(grid_spacing, v.ndim, "scalar_born")
    if checkpoint_segments is not None:
        # Delegated before anything is resolved, so that every segment sees
        # the arguments exactly as they were passed -- except for the
        # deprecated spellings, which are resolved here, once, before the
        # delegation: the segments' re-entrant calls then only ever see the
        # canonical keywords, so the deprecation warns once, at this call,
        # rather than once per segment from inside Deepwave.
        return _engine.delegate_checkpoint_segments(
            scalar_born,
            SPEC,
            _engine.resolve_deprecated_arguments(
                SPEC,
                dict(locals()),
                receiver_locations="sc_receiver_locations",
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
            "source_amplitudes": source_amplitudes,
            "source_locations": source_locations,
            "source_mask": source_mask,
            "sc_receiver_locations": sc_receiver_locations,
            "sc_receiver_mask": sc_receiver_mask,
            "bg_receiver_locations": bg_receiver_locations,
            "bg_receiver_mask": bg_receiver_mask,
        },
        receiver_locations=("sc_receiver_locations", receiver_locations),
    )
    _engine.expand_channel_mappings(
        SPEC,
        _channels,
        sources=sources,
        receivers=receivers,
        bg_receivers=bg_receivers,
    )
    _common.apply_location_masks(SPEC, _channels)
    source_amplitudes = cast("Optional[torch.Tensor]", _channels["source_amplitudes"])
    source_locations = cast("Optional[LocationsLike]", _channels["source_locations"])
    sc_receiver_locations = cast(
        "Optional[LocationsLike]", _channels["sc_receiver_locations"]
    )
    bg_receiver_locations = cast(
        "Optional[LocationsLike]", _channels["bg_receiver_locations"]
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
    _common.check_models([v], ["v"], "scalar_born")
    backend_request = _engine._coerce_backend(  # noqa: SLF001
        backend, python_backend, stacklevel=3
    )
    backend_mode = backend_request.mode
    del backend, python_backend
    # `scatter` is the v0.0.27 name for the scattering potential, retained as
    # a deprecated alias for `dv`.
    if scatter is not None:
        _exceptions.warn_deprecated_name("scatter", "dv", stacklevel=2)
        if dv is not None:
            raise _exceptions.DeepwaveValueError(
                "scalar_born: pass only dv, not both dv and the deprecated scatter."
            )
        dv = scatter
    del scatter
    if dv is None:
        raise _exceptions.DeepwaveValueError(
            "scalar_born: at least one scattering model (dv) must be provided."
        )
    # The rest of this function (and the autograd Function it dispatches to)
    # uses the local name `scatter` for the scattering potential throughout.
    scatter = dv
    del dv
    _common.check_models([scatter], ["dv"], "scalar_born")
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
            "wavefield_sc_0": wavefield_sc_0,
            "wavefield_sc_m1": wavefield_sc_m1,
            "psiz_sc_m1": None,
            "psiy_sc_m1": None,
            "psix_sc_m1": None,
            "zetaz_sc_m1": None,
            "zetay_sc_m1": None,
            "zetax_sc_m1": None,
        },
        origin=origin,
        survey_pad=survey_pad,
        ignore=_engine.receiver_output_names(SPEC),
        auxiliary_state_names=scalar_born_auxiliary_state_names(3),
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
        wavefield_sc_0,
        wavefield_sc_m1,
        psiz_sc_m1,
        psiy_sc_m1,
        psix_sc_m1,
        zetaz_sc_m1,
        zetay_sc_m1,
        zetax_sc_m1,
    ) = _resolved_wavefields.values()

    # Resolve off-grid / directional (Hicks) source and receiver specifications.
    # scalar_born is on a regular (non-staggered) grid, so every component lives
    # on the reference node (offset None). A single source interpolator is built
    # from source_locations and its spread applied once, then shared by the
    # background and scattered source inputs (they use identical locations); the
    # background and scattered receivers have independent interpolators. Integer
    # /long locations take the exact path and are bit-for-bit identical to
    # before.
    # Recorded before the location arguments are replaced by the integer grid
    # indices the backend consumes.
    location_interpolation_used = _engine.resolved_location_interpolation(
        location_interpolation,
        [[source_locations], [bg_receiver_locations], [sc_receiver_locations]],
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
        [source_locations, bg_receiver_locations, sc_receiver_locations],
        v,
        "scalar_born",
        whole_model=(
            survey_pad is None
            and origin is None
            and initial_state is None
            and wavefield_0 is None
            and wavefield_m1 is None
            and wavefield_sc_0 is None
            and wavefield_sc_m1 is None
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
    (
        receiver_interpolators,
        (bg_receiver_locations, sc_receiver_locations),
    ) = _common.resolve_receiver_interpolators(
        [bg_receiver_locations, sc_receiver_locations],
        [None, None],
        location_interpolation,
        location_interpolation_halfwidth,
        v.dtype,
        _hicks_grid,
    )

    _common.check_inputs_not_vmapped(
        wavefield_0,
        wavefield_m1,
        wavefield_sc_0,
        wavefield_sc_m1,
        source_amplitudes,
        source_locations,
        sc_receiver_locations,
        bg_receiver_locations,
        psiz_m1,
        zetaz_m1,
        psiz_sc_m1,
        zetaz_sc_m1,
        psiy_m1,
        zetay_m1,
        psiy_sc_m1,
        zetay_sc_m1,
        psix_m1,
        zetax_m1,
        psix_sc_m1,
        zetax_sc_m1,
    )
    ndim = _common.get_ndim(
        [v, scatter],
        [wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1],
        [source_locations, sc_receiver_locations, bg_receiver_locations],
        [
            psiz_m1,
            zetaz_m1,
            psiz_sc_m1,
            zetaz_sc_m1,
        ],
        [psiy_m1, zetay_m1, psiy_sc_m1, zetay_sc_m1],
        [psix_m1, zetax_m1, psix_sc_m1, zetax_sc_m1],
    )
    psi: list[Optional[torch.Tensor]] = []
    zeta: list[Optional[torch.Tensor]] = []
    psi_sc: list[Optional[torch.Tensor]] = []
    zeta_sc: list[Optional[torch.Tensor]] = []
    if ndim == 3:
        psi.append(psiz_m1)
        zeta.append(zetaz_m1)
        psi_sc.append(psiz_sc_m1)
        zeta_sc.append(zetaz_sc_m1)
    if ndim >= 2:
        psi.append(psiy_m1)
        zeta.append(zetay_m1)
        psi_sc.append(psiy_sc_m1)
        zeta_sc.append(zetay_sc_m1)
    if ndim >= 1:
        psi.append(psix_m1)
        zeta.append(zetax_m1)
        psi_sc.append(psix_sc_m1)
        zeta_sc.append(zetax_sc_m1)
    initial_wavefields: list[Optional[torch.Tensor]] = [
        wavefield_0,
        wavefield_m1,
        *psi,
        *zeta,
        wavefield_sc_0,
        wavefield_sc_m1,
        *psi_sc,
        *zeta_sc,
    ]

    min_nonzero_model_vel, max_model_vel = _common.velocity_extremes(
        v, max_vel, "scalar_born"
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
        [v, scatter],
        ["replicate", "replicate"],
        ["v", "dv"],
        grid_spacing,
        dt,
        [source_amplitudes, source_amplitudes],
        [source_locations, source_locations],
        [bg_receiver_locations, sc_receiver_locations],
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
    del (
        v,
        scatter,
        source_amplitudes,
        source_locations,
        bg_receiver_locations,
        sc_receiver_locations,
    )

    boundary_codes = _common.set_boundary(boundary, pml_width, ndim)
    _common.check_boundary_supported(boundary_codes, accuracy, ndim, "scalar_born")
    _common.warn_bare_zero_width_face(boundary, pml_width, ndim, "scalar_born")
    # The scattering model is extended outwards like the background velocity,
    # which is what makes the scattered wavefield the linearisation of
    # `scalar` even for a perturbation of the region's outermost cells. That
    # is a change of numbers for such a perturbation, and only for one.
    _common.warn_born_edge_scatter_migration(
        models[1],
        [fd_pad[face] + pml_width[face] for face in range(2 * ndim)],
        ndim,
        stacklevel=3,
    )
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

    model_shape = models[0].shape[-ndim:]
    # Background (multiply source amplitudes by -v^2*dt^2)
    source_amplitudes_out[0] = (
        -source_amplitudes_out[0]
        * _common.gather_model_at_sources(models[0], sources_i[0], model_shape, n_shots)
        ** 2
        * dt**2
    )
    # Scattered (multiply source amplitudes by -2*v*scatter*dt^2)
    source_amplitudes_out[1] = (
        -2
        * source_amplitudes_out[1]
        * _common.gather_model_at_sources(models[0], sources_i[1], model_shape, n_shots)
        * _common.gather_model_at_sources(models[1], sources_i[1], model_shape, n_shots)
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
        scalar_born_func(
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
        )
    )

    outputs[-2] = _common.downsample_and_movedim(
        outputs[-2],
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )
    outputs[-1] = _common.downsample_and_movedim(
        outputs[-1],
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    # Gather off-grid receiver data back to the original locations. The final
    # two outputs are the background then the scattered receiver amplitudes,
    # matching receiver_interpolators = [bg, scattered].
    _common.apply_receiver_interpolators(
        outputs, receiver_interpolators, len(outputs) - 2
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


class ScalarBornForwardFunc(torch.autograd.Function):
    """Forward propagation function for scalar Born modelling."""

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
        *args: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Forward propagation of the scalar Born wave equation.

        Runs on the shared :func:`deepwave._engine.run_forward_engine` engine. ``args``
        is the flat tensor tuple ``(v, scatter, source_amplitudes,
        source_amplitudessc, sources_i, unused, receivers_i, receiverssc_i,
        *wavefields)``, where the wavefields are the background fields (current,
        previous, psi, zeta) followed by the corresponding scattered fields. The
        second source-location slot is unused (the scattered source shares the
        background source locations).

        Returns the padding-removed background and scattered output wavefields
        followed by the background and scattered receiver amplitudes.
        """
        ndim = len(grid_spacing)
        # Indices into the wavefield list (background then scattered):
        # [wfc, wfp, psi..., zeta..., wfcsc, wfpsc, psisc..., zetasc...].
        sc = 2 + 2 * ndim  # start of the scattered fields

        def storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
            # w_store (background; needed for either model grad) and wsc_store,
            # from the single-sourced module-level rule that
            # deepwave.estimate_memory also uses.
            return storage_slot_flags(user_model_grad_flags(models, ndim), ndim)

        def make_aux_wavefields(
            wavefields: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            # Temporary ping-pong buffers for the background and scattered psi.
            return [torch.zeros_like(wavefields[2 + i]) for i in range(ndim)] + [
                torch.zeros_like(wavefields[sc + 2 + i]) for i in range(ndim)
            ]

        def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
            v, scatter = state.models
            ndim = state.ndim
            w = state.wavefields
            psi = w[2 : 2 + ndim]
            zeta = w[2 + ndim : 2 + 2 * ndim]
            psisc = w[sc + 2 : sc + 2 + ndim]
            zetasc = w[sc + 2 + ndim : sc + 2 + 2 * ndim]
            psin = state.aux_wavefields[:ndim]
            psinsc = state.aux_wavefields[ndim:]
            v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
            scatter_batched = scatter.ndim == ndim + 1 and scatter.shape[0] > 1
            call_args: tuple[Any, ...] = (
                v.data_ptr(),
                scatter.data_ptr(),
                state.source_amplitudes[0].data_ptr(),
                state.source_amplitudes[1].data_ptr(),
                w[0].data_ptr(),
                w[1].data_ptr(),
                *[field.data_ptr() for field in psi],
                *[field.data_ptr() for field in psin],
                *[field.data_ptr() for field in zeta],
                w[sc].data_ptr(),
                w[sc + 1].data_ptr(),
                *[field.data_ptr() for field in psisc],
                *[field.data_ptr() for field in psinsc],
                *[field.data_ptr() for field in zetasc],
                *state.storage_manager.storage_ptrs,
                state.receiver_amplitudes[0].data_ptr(),
                state.receiver_amplitudes[1].data_ptr(),
                *[profile.data_ptr() for profile in state.pml_profiles],
                state.sources_i[0].data_ptr(),
                state.receivers_i[0].data_ptr(),
                state.receivers_i[1].data_ptr(),
                *state.rdx,
                *state.rdx2,
                state.dt**2,
                step_nt,
                state.n_shots,
                *state.model_shape,
                state.n_sources_per_shot[0],
                state.n_receivers_per_shot[0],
                state.n_receivers_per_shot[1],
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                state.storage_grad_flag(v.requires_grad),
                state.storage_grad_flag(scatter.requires_grad),
                v_batched,
                scatter_batched,
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                state.aux,
                state.stream,
            )
            return cast("int", state.backend(*call_args))

        def swap(state: _engine.BackendState) -> None:
            # Ping-pong the current/previous and psi/psin buffers for both the
            # background and scattered fields (zeta is not swapped on forward).
            ndim = state.ndim
            w = state.wavefields
            aux = state.aux_wavefields
            w[0], w[1] = w[1], w[0]
            w[2 : 2 + ndim], aux[:ndim] = aux[:ndim], w[2 : 2 + ndim]
            w[sc], w[sc + 1] = w[sc + 1], w[sc]
            w[sc + 2 : sc + 2 + ndim], aux[ndim:] = (
                aux[ndim:],
                w[sc + 2 : sc + 2 + ndim],
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
            zero_wavefields=_zero_born_pml_interiors,
            make_call=make_call,
            make_aux_wavefields=make_aux_wavefields,
            swap=swap,
            save_wavefields=False,
            boundary=boundary,
        )
        return (*outputs, saved_state)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        """Populates ``ctx`` from the forward's trailing state, plus ``non_sc``.

        ``non_sc`` (whether the background/non-scattered adjoint pass is needed)
        is recomputed here from the forward inputs -- it depends only on which
        inputs require grad, so no forward-computed state has to be threaded
        through ``output``.
        """
        _engine.apply_forward_saved_state(ctx, output[-1])
        # ``boundary`` is the last leading (non-tensor) input; recover it for
        # ``backward`` (see :func:`deepwave._common.set_boundary`).
        ctx.boundary = list(inputs[_N_LEAD - 1])
        args = inputs[_N_LEAD:]
        ndim = len(inputs[1])  # grid_spacing
        wavefields = args[8:]
        ctx.non_sc = (
            args[0].requires_grad
            or args[2].requires_grad
            or any(w.requires_grad for w in wavefields[: 2 + 2 * ndim])
        )

    # torch.func transforms now reach this Function; the ctypes backend cannot
    # process batched/dual tensors, so raise an actionable error pointing to
    # backend="eager" (generate_vmap_rule deliberately not enabled).
    vmap = staticmethod(_engine.unsupported_vmap)
    jvp = staticmethod(_engine.unsupported_jvp)

    @staticmethod
    @_engine.no_second_order("scalar_born")
    def backward(
        ctx: Any,
        *args: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward propagation of the scalar Born wave equation.

        Decorated with :func:`deepwave._engine.no_second_order` (a stricter
        replacement for ``@once_differentiable``): this backward calls the
        non-differentiable C/CUDA backend directly rather than building a
        nested differentiable ``Function`` (as :class:`ScalarForwardFunc` does
        via :class:`ScalarBackwardFunc` for its Hessian-vector product), so
        ``scalar_born`` does not support second-order derivatives. A
        reverse-over-reverse Hessian-vector product through this propagator
        would yield an incorrect (typically zero) result for the model
        gradient rather than the true second-order term, so the decorator
        raises a clear error when a second-order graph is built through it
        (``create_graph=True``); use the ``scalar`` propagator if you need a
        Hessian-vector product.

        Args:
            ctx: Context object from the forward pass.
            args: Gradients of the outputs of the forward pass.

        Returns:
            A tuple containing the gradients with respect to the inputs of the
            forward pass.

        """
        # Drop the grad slot for the trailing ``ForwardSavedState`` output.
        args = args[:-1]
        grad_r = args[-2]
        grad_rsc = args[-1]
        grad_wavefields = list(args[:-2])
        del args
        # run_forward saved (v, scatter, sources_i, unused, receivers_i,
        # receiverssc_i, source_amplitudes, source_amplitudessc, *pml_profiles);
        # the source amplitudes and the unused source slot are not needed here.
        saved = list(ctx.saved_tensors)
        v, scatter, sources_i = saved[0], saved[1], saved[2]
        receivers_i, receiverssc_i = saved[4], saved[5]
        pml_profiles = saved[8:]

        v = v.contiguous()
        scatter = scatter.contiguous()
        grad_r = grad_r.contiguous()
        grad_rsc = grad_rsc.contiguous()
        grad_wavefields = [p.contiguous() for p in grad_wavefields]
        pml_profiles = [p.contiguous() for p in pml_profiles]
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        receiverssc_i = receiverssc_i.contiguous()

        grid_spacing = ctx.grid_spacing
        ndim = len(grid_spacing)
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        # run_forward stores these as a (background, scattered) list.
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad[0]
        source_amplitudessc_requires_grad = ctx.source_amplitudes_requires_grad[1]
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        non_sc = ctx.non_sc
        dtype = v.dtype
        storage_manager = ctx.storage_manager
        # Index of the first scattered field in the wavefield list.
        sc = 2 + 2 * ndim
        boundary = ctx.boundary

        # Unlike the forward (which reads v/scatter only at the current grid
        # point), the compiled adjoint reads both at neighbour offsets near a
        # reflecting face, so their guard rings must hold the even/wrap image.
        # They may alias the saved user tensors, so mirror-pad fresh clones
        # rather than writing in place. All-PML boundaries are a no-op.
        v, scatter = _boundaries.mirror_pad_scalar_born_models(
            [v, scatter], boundary, accuracy // 2
        )

        backward = _backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
        )
        backward_sc = _backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
            extra="_sc",
        )

        lead = (None,) * _N_LEAD
        # Four Nones for the sources_i, unused, receivers_i, receiverssc_i slots.
        locs = (None,) * 4

        if non_sc:
            # Background and scattered adjoint in a single pass over all fields.
            # The wavefields are [background(wfc, wfp, psi, zeta), scattered(...)];
            # the auxiliaries are the psi/zeta ping-pong temporaries, ordered
            # [psin, zetan, psinsc, zetansc].
            def make_aux_wavefields(
                w: list[torch.Tensor], ndim: int
            ) -> list[torch.Tensor]:
                return (
                    [torch.zeros_like(w[2 + i]) for i in range(ndim)]
                    + [torch.zeros_like(w[2 + ndim + i]) for i in range(ndim)]
                    + [torch.zeros_like(w[sc + 2 + i]) for i in range(ndim)]
                    + [torch.zeros_like(w[sc + 2 + ndim + i]) for i in range(ndim)]
                )

            def swap(state: _engine.BackendState) -> None:
                ndim = state.ndim
                w = state.wavefields
                a = state.aux_wavefields
                w[0], w[1] = w[1], w[0]
                w[2 : 2 + ndim], a[:ndim] = a[:ndim], w[2 : 2 + ndim]
                w[2 + ndim : 2 + 2 * ndim], a[ndim : 2 * ndim] = (
                    a[ndim : 2 * ndim],
                    w[2 + ndim : 2 + 2 * ndim],
                )
                w[sc], w[sc + 1] = w[sc + 1], w[sc]
                w[sc + 2 : sc + 2 + ndim], a[2 * ndim : 3 * ndim] = (
                    a[2 * ndim : 3 * ndim],
                    w[sc + 2 : sc + 2 + ndim],
                )
                w[sc + 2 + ndim : sc + 2 + 2 * ndim], a[3 * ndim : 4 * ndim] = (
                    a[3 * ndim : 4 * ndim],
                    w[sc + 2 + ndim : sc + 2 + 2 * ndim],
                )

            def negate(state: _engine.BackendState) -> None:
                # The adjoint pointer-swap trick negates the previous-wavefield
                # gradients (background and scattered) before and after the loop.
                state.wavefields[1] = -state.wavefields[1]
                state.wavefields[sc + 1] = -state.wavefields[sc + 1]

            def make_call(state: _engine.BackendState, step: int, step_nt: int) -> int:
                model_v, model_scatter = state.models
                ndim = state.ndim
                w = state.wavefields
                a = state.aux_wavefields
                psi = w[2 : 2 + ndim]
                zeta = w[2 + ndim : 2 + 2 * ndim]
                psisc = w[sc + 2 : sc + 2 + ndim]
                zetasc = w[sc + 2 + ndim : sc + 2 + 2 * ndim]
                psin = a[:ndim]
                zetan = a[ndim : 2 * ndim]
                psinsc = a[2 * ndim : 3 * ndim]
                zetansc = a[3 * ndim : 4 * ndim]
                n_f = (
                    state.n_sources_per_shot[0]
                    if state.grad_sources[0].numel() > 0
                    else 0
                )
                n_fsc = (
                    state.n_sources_per_shot[1]
                    if state.grad_sources[1].numel() > 0
                    else 0
                )
                call_args: tuple[Any, ...] = (
                    model_v.data_ptr(),
                    model_scatter.data_ptr(),
                    state.grad_receivers[0].data_ptr(),
                    state.grad_receivers[1].data_ptr(),
                    w[0].data_ptr(),
                    w[1].data_ptr(),
                    *[f.data_ptr() for f in psi],
                    *[f.data_ptr() for f in psin],
                    *[f.data_ptr() for f in zeta],
                    *[f.data_ptr() for f in zetan],
                    w[sc].data_ptr(),
                    w[sc + 1].data_ptr(),
                    *[f.data_ptr() for f in psisc],
                    *[f.data_ptr() for f in psinsc],
                    *[f.data_ptr() for f in zetasc],
                    *[f.data_ptr() for f in zetansc],
                    *state.storage_manager.storage_ptrs,
                    state.grad_sources[0].data_ptr(),
                    state.grad_sources[1].data_ptr(),
                    state.grad_models[0].data_ptr(),
                    state.grad_models[1].data_ptr(),
                    state.grad_models_tmp_ptr[0],
                    state.grad_models_tmp_ptr[1],
                    *[p.data_ptr() for p in state.pml_profiles],
                    state.sources_i[0].data_ptr(),
                    state.receivers_i[0].data_ptr(),
                    state.receivers_i[1].data_ptr(),
                    *state.rdx,
                    *state.rdx2,
                    state.dt**2,
                    step_nt,
                    state.n_shots,
                    *state.model_shape,
                    n_f,
                    n_fsc,
                    state.n_receivers_per_shot[0],
                    state.n_receivers_per_shot[1],
                    state.step_ratio,
                    state.storage_manager.storage_mode,
                    state.storage_manager.shot_bytes_uncomp,
                    state.storage_manager.shot_bytes_comp,
                    state.storage_grad_flag(model_v.requires_grad),
                    state.storage_grad_flag(model_scatter.requires_grad),
                    model_v.ndim == ndim + 1 and model_v.shape[0] > 1,
                    model_scatter.ndim == ndim + 1 and model_scatter.shape[0] > 1,
                    state.storage_manager.storage_compression,
                    step,
                    *state.pml_b,
                    *state.pml_e,
                    *state.bc,
                    state.aux,
                    state.stream,
                )
                return cast("int", backward(*call_args))

            state = _engine.run_backward_core(
                SPEC,
                models=[v, scatter],
                sources_i=[sources_i, sources_i],
                receivers_i=[receivers_i, receiverssc_i],
                source_amplitudes=[],
                pml_profiles=pml_profiles,
                grad_wavefields=grad_wavefields,
                grad_receivers=[grad_r, grad_rsc],
                grid_spacing=list(grid_spacing),
                dt=dt,
                nt=nt,
                n_shots=n_shots,
                step_ratio=step_ratio,
                accuracy=accuracy,
                pml_width=pml_width,
                source_amplitudes_requires_grad=[
                    source_amplitudes_requires_grad,
                    source_amplitudessc_requires_grad,
                ],
                storage_manager=storage_manager,
                backward_callback=backward_callback,
                callback_frequency=callback_frequency,
                make_aux_wavefields=make_aux_wavefields,
                zero_wavefields=_zero_born_pml_interiors,
                make_call=make_call,
                swap=swap,
                pre_loop=negate,
                post_loop=negate,
                boundary=boundary,
            )
            out_s = state.output_slice
            return (
                *lead,
                state.grad_models[0],
                state.grad_models[1],
                state.grad_sources[0],
                state.grad_sources[1],
                *locs,
                *(w[out_s] for w in state.wavefields),
            )

        # Scattered-only adjoint: v does not require a gradient, so only the
        # scattered field is propagated (via the ``_sc`` backend) and only the
        # scatterer gradient is produced. The scattered subset has the same
        # layout as a plain regular-grid field: [wfcsc, wfpsc, psisc, zetasc].
        grad_wavefields_sc = grad_wavefields[sc:]

        def make_aux_wavefields_sc(
            w: list[torch.Tensor], ndim: int
        ) -> list[torch.Tensor]:
            return [torch.zeros_like(w[2 + i]) for i in range(ndim)] + [
                torch.zeros_like(w[2 + ndim + i]) for i in range(ndim)
            ]

        def swap_sc(state: _engine.BackendState) -> None:
            ndim = state.ndim
            w = state.wavefields
            a = state.aux_wavefields
            w[0], w[1] = w[1], w[0]
            w[2 : 2 + ndim], a[:ndim] = a[:ndim], w[2 : 2 + ndim]
            w[2 + ndim : 2 + 2 * ndim], a[ndim : 2 * ndim] = (
                a[ndim : 2 * ndim],
                w[2 + ndim : 2 + 2 * ndim],
            )

        def negate_sc(state: _engine.BackendState) -> None:
            state.wavefields[1] = -state.wavefields[1]

        def make_call_sc(state: _engine.BackendState, step: int, step_nt: int) -> int:
            model_v, model_scatter = state.models
            ndim = state.ndim
            w = state.wavefields
            a = state.aux_wavefields
            psisc = w[2 : 2 + ndim]
            zetasc = w[2 + ndim : 2 + 2 * ndim]
            psinsc = a[:ndim]
            zetansc = a[ndim : 2 * ndim]
            n_fsc = (
                state.n_sources_per_shot[0] if state.grad_sources[0].numel() > 0 else 0
            )
            call_args: tuple[Any, ...] = (
                model_v.data_ptr(),
                state.grad_receivers[0].data_ptr(),
                w[0].data_ptr(),
                w[1].data_ptr(),
                *[f.data_ptr() for f in psisc],
                *[f.data_ptr() for f in psinsc],
                *[f.data_ptr() for f in zetasc],
                *[f.data_ptr() for f in zetansc],
                *state.storage_manager.storage_ptrs[:5],  # Not scattered
                state.grad_sources[0].data_ptr(),
                state.grad_models[1].data_ptr(),
                state.grad_models_tmp_ptr[1],
                *[p.data_ptr() for p in state.pml_profiles],
                state.sources_i[0].data_ptr(),
                state.receivers_i[0].data_ptr(),
                *state.rdx,
                *state.rdx2,
                state.dt**2,
                step_nt,
                state.n_shots,
                *state.model_shape,
                n_fsc,
                state.n_receivers_per_shot[0],
                state.step_ratio,
                state.storage_manager.storage_mode,
                state.storage_manager.shot_bytes_uncomp,
                state.storage_manager.shot_bytes_comp,
                state.storage_grad_flag(model_scatter.requires_grad),
                model_v.ndim == ndim + 1 and model_v.shape[0] > 1,
                model_scatter.ndim == ndim + 1 and model_scatter.shape[0] > 1,
                state.storage_manager.storage_compression,
                step,
                *state.pml_b,
                *state.pml_e,
                *state.bc,
                state.aux,
                state.stream,
            )
            return cast("int", backward_sc(*call_args))

        state = _engine.run_backward_core(
            SPEC,
            models=[v, scatter],
            sources_i=[sources_i],
            receivers_i=[receiverssc_i],
            source_amplitudes=[],
            pml_profiles=pml_profiles,
            grad_wavefields=grad_wavefields_sc,
            grad_receivers=[grad_rsc],
            grid_spacing=list(grid_spacing),
            dt=dt,
            nt=nt,
            n_shots=n_shots,
            step_ratio=step_ratio,
            accuracy=accuracy,
            pml_width=pml_width,
            source_amplitudes_requires_grad=[source_amplitudessc_requires_grad],
            storage_manager=storage_manager,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            make_aux_wavefields=make_aux_wavefields_sc,
            zero_wavefields=_zero_pml_interiors_sc,
            make_call=make_call_sc,
            swap=swap_sc,
            pre_loop=negate_sc,
            post_loop=negate_sc,
            boundary=boundary,
            wavefield_names=scalar_born_wavefield_names(ndim)[sc:],
        )
        out_s = state.output_slice
        # v has no gradient in this path, and the background wavefields are not
        # propagated, so their gradients are None.
        return (
            lead
            + (None, state.grad_models[1], None, state.grad_sources[0])
            + locs
            + (None,) * (2 + 2 * ndim)
            + tuple(w[out_s] for w in state.wavefields)
        )


# Number of leading non-tensor arguments of ``ScalarBornForwardFunc.forward``,
# all of which receive ``None`` gradients in ``backward``.
_N_LEAD = _engine.leading_arg_count(ScalarBornForwardFunc.forward)


def _forward_step(
    v: torch.Tensor,
    scatter: torch.Tensor,
    grid_spacing: list[float],
    dt: float,
    accuracy: int,
    pml_profiles: list[torch.Tensor],
    wavefields: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Performs a single time step of scalar Born wave propagation."""
    ndim = len(grid_spacing)

    wfc, wfp = wavefields[:2]
    psi = list(wavefields[2 : 2 + ndim])
    zeta = list(wavefields[2 + ndim : 2 + 2 * ndim])
    wfcsc, wfpsc = wavefields[2 + 2 * ndim : 4 + 2 * ndim]
    psisc = list(wavefields[4 + 2 * ndim : 4 + 3 * ndim])
    zetasc = list(wavefields[4 + 3 * ndim : 4 + 4 * ndim])
    del wavefields
    a = pml_profiles[::3]
    b = pml_profiles[1::3]
    dbdx = pml_profiles[2::3]
    del pml_profiles
    dudx = [
        _regular_grid.diff1(wfc, dim, accuracy, 1 / grid_spacing[dim], ndim)
        for dim in range(ndim)
    ]
    duscdx = [
        _regular_grid.diff1(wfcsc, dim, accuracy, 1 / grid_spacing[dim], ndim)
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
    tmpsc = [
        (1 + b[dim])
        * _regular_grid.diff2(wfcsc, dim, accuracy, 1 / grid_spacing[dim] ** 2, ndim)
        + dbdx[dim] * duscdx[dim]
        + _regular_grid.diff1(
            a[dim] * psisc[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    w_sum = torch.zeros_like(tmp[0])
    wsc_sum = torch.zeros_like(tmpsc[0])
    for dim in range(ndim):
        w_sum = w_sum + (1 + b[dim]) * tmp[dim] + a[dim] * zeta[dim]
        wsc_sum = wsc_sum + (1 + b[dim]) * tmpsc[dim] + a[dim] * zetasc[dim]

    return list(
        [
            v**2 * dt * dt * w_sum + 2 * wfc - wfp,
            wfc,
        ]
        + [b[dim] * dudx[dim] + a[dim] * psi[dim] for dim in range(ndim)]
        + [b[dim] * tmp[dim] + a[dim] * zeta[dim] for dim in range(ndim)]
        + [
            v**2 * dt * dt * wsc_sum
            + 2 * wfcsc
            - wfpsc
            + 2 * v * scatter * dt * dt * w_sum,
            wfcsc,
        ]
        + [b[dim] * duscdx[dim] + a[dim] * psisc[dim] for dim in range(ndim)]
        + [b[dim] * tmpsc[dim] + a[dim] * zetasc[dim] for dim in range(ndim)]
    )


_forward_step_cache = _engine.StepCache(_forward_step)
_forward_step_opt = _forward_step

# Named offsets into the leading (pre-tensor-block) args scalar_born_func
# receives from its call site (see
# _backend_utils.PROPAGATOR_REGISTRY["scalar_born"].leading_args): the index
# of "boundary", and the index one past the last leading arg -- i.e. where
# the model/source/wavefield tensor block starts.
_BOUNDARY_IDX = _backend_utils.leading_arg_offset("scalar_born", "boundary")
_N_LEADING_ARGS = len(_backend_utils.PROPAGATOR_REGISTRY["scalar_born"].leading_args)


def scalar_born_python(
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
    """Forward propagation of the scalar Born wave equation.

    ``args`` is the flat tensor tuple ``(v, scatter, source_amplitudes,
    source_amplitudessc, sources_i, unused, receivers_i, receiverssc_i,
    *wavefields)``.

    Args:
        pml_profiles: List of PML profiles.
        grid_spacing: List of grid spacing in each spatial dimension.
        dt: Time step size.
        nt: Number of time steps.
        step_ratio: Step ratio for storing wavefields.
        accuracy: Accuracy of the finite-difference scheme.
        pml_width: Width of the PML.
        boundary: Per-face boundary-type integer codes (see
            :func:`deepwave._common.set_boundary`); ``BC_PML`` faces are absorbing
            and left untouched. The same image fill is applied to both the
            background and the scattered wavefields.
        n_shots: Number of shots.
        forward_callback: The forward callback.
        backward_callback: The backward callback.
        callback_frequency: The callback frequency.
        storage_mode_str: The storage mode (Unused).
        storage_path: The path to disk storage (Unused).
        storage_compression: Whether to apply compression to storage (Unused).
        args: The flat tensor tuple (v, scatter, source_amplitudes,
            source_amplitudessc, sources_i, unused, receivers_i, receiverssc_i,
            *wavefields).
        backend: The resolved Python backend name (``"eager"`` or
            ``"compile"``), already selected by the caller; used only to name
            it if an unsupported option is rejected.

    Returns:
        A tuple containing the final wavefields and receiver data.

    """
    ndim = len(grid_spacing)
    num_wf = 4 + 4 * ndim
    args_list = list(args)
    wavefields = args_list[-num_wf:]
    receiverssc_i = args_list[-num_wf - 1]
    receivers_i = args_list[-num_wf - 2]
    # args_list[-num_wf - 3] is the unused second source-location slot.
    sources_i = args_list[-num_wf - 4]
    source_amplitudessc = args_list[-num_wf - 5]
    source_amplitudes = args_list[-num_wf - 6]
    scatter = args_list[-num_wf - 7]
    v = args_list[-num_wf - 8]

    is_batched = any(
        _common.is_inside_vmap(x)
        for x in [
            v,
            scatter,
            source_amplitudes,
            source_amplitudessc,
            sources_i,
            receivers_i,
            receiverssc_i,
            *wavefields,
        ]
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
    n_receiverssc_per_shot = receiverssc_i.numel() // n_shots
    receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudessc = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudes_list: list[torch.Tensor] = []
    receiver_amplitudessc_list: list[torch.Tensor] = []

    if receivers_i.numel() > 0 and not is_batched:
        receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
        receiver_amplitudes.fill_(0)
    if receiverssc_i.numel() > 0 and not is_batched:
        receiver_amplitudessc.resize_(nt, n_shots, n_receiverssc_per_shot)
        receiver_amplitudessc.fill_(0)

    # The background and scattered sources share the same locations.
    sources_i_masked, source_mask = _common.mask_locations(sources_i)
    source_amplitudes_masked = _common.mask_source_amplitudes(
        source_amplitudes, source_mask
    )
    source_amplitudessc_masked = _common.mask_source_amplitudes(
        source_amplitudessc, source_mask
    )

    receivers_i_masked, receiver_mask = _common.mask_locations(receivers_i)
    receiverssc_i_masked, receiversc_mask = _common.mask_locations(receiverssc_i)

    dim_names = ["z", "y", "x"]

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = {
                "wavefield_0": wavefields[0],
                "wavefield_m1": wavefields[1],
                "wavefield_sc_0": wavefields[2 + 2 * ndim],
                "wavefield_sc_m1": wavefields[3 + 2 * ndim],
            }
            for i in range(ndim):
                callback_wavefields[f"psi{dim_names[-ndim + i]}_m1"] = wavefields[2 + i]
                callback_wavefields[f"zeta{dim_names[-ndim + i]}_m1"] = wavefields[
                    2 + ndim + i
                ]
                callback_wavefields[f"psi{dim_names[-ndim + i]}_sc_m1"] = wavefields[
                    4 + 2 * ndim + i
                ]
                callback_wavefields[f"zeta{dim_names[-ndim + i]}_sc_m1"] = wavefields[
                    4 + 3 * ndim + i
                ]

            _engine.invoke_forward_callback(
                forward_callback,
                CallbackState(
                    dt * step_ratio,
                    step,
                    callback_wavefields,
                    {"v": v, "dv": scatter},
                    {},
                    [fd_pad] * 2 * ndim,
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
            if receivers_i_masked.numel() > 0:
                val = wavefields[0].view(-1, flat_shape).gather(1, receivers_i_masked)
                if is_batched:
                    receiver_amplitudes_list.append(val)
                else:
                    receiver_amplitudes[t] = val
            if receiverssc_i_masked.numel() > 0:
                val = (
                    wavefields[2 + 2 * ndim]
                    .view(-1, flat_shape)
                    .gather(1, receiverssc_i_masked)
                )
                if is_batched:
                    receiver_amplitudessc_list.append(val)
                else:
                    receiver_amplitudessc[t] = val
            if has_boundary:
                # Both the background and the scattered nodal fields obey the
                # same first-class faces (mirroring is linear, so the scattered
                # field uses the same odd/even/wrap images as the background).
                wavefields[0] = _boundaries.fill_scalar_boundary(
                    wavefields[0], boundary, fd_pad
                )
                wavefields[2 + 2 * ndim] = _boundaries.fill_scalar_boundary(
                    wavefields[2 + 2 * ndim], boundary, fd_pad
                )
            wavefields = _forward_step_opt(
                v,
                scatter,
                grid_spacing,
                dt,
                accuracy,
                pml_profiles,
                wavefields,
            )
            if source_amplitudes_masked.numel() > 0:
                wavefields[0].view(-1, flat_shape).scatter_add_(
                    1, sources_i_masked, source_amplitudes_masked[t]
                )
                wavefields[2 + 2 * ndim].view(-1, flat_shape).scatter_add_(
                    1, sources_i_masked, source_amplitudessc_masked[t]
                )

    if is_batched and len(receiver_amplitudes_list) > 0:
        receiver_amplitudes = torch.stack(receiver_amplitudes_list)
    if is_batched and len(receiver_amplitudessc_list) > 0:
        receiver_amplitudessc = torch.stack(receiver_amplitudessc_list)

    receiver_amplitudes_masked = _common.mask_receiver_amplitudes(
        receiver_amplitudes, receiver_mask
    )
    del receiver_amplitudes
    receiver_amplitudessc_masked = _common.mask_receiver_amplitudes(
        receiver_amplitudessc, receiversc_mask
    )
    del receiver_amplitudessc
    s = (
        slice(None),
        *(slice(fd_pad, -fd_pad) for _ in range(ndim)),
    )
    return tuple(
        [wavefield[s] for wavefield in wavefields]
        + [receiver_amplitudes_masked, receiver_amplitudessc_masked]
    )


def scalar_born_func(
    backend: Optional[Literal["eager", "compile"]],
    *args: Any,
) -> tuple[torch.Tensor, ...]:
    """Helper function to apply the ScalarBornForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ScalarBornForwardFunc`, which is the entry point for the autograd graph
    for scalar Born wave propagation.

    Args:
        backend: None, or a string specifying whether to use the
            Python backend. Already coerced (no bool) by the caller.
        *args: Variable length argument list to be passed directly to
            `ScalarBornForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ScalarBornForwardFunc.apply`.

    """
    global _forward_step_opt

    if backend:
        # grid_spacing must reach the traced step functions as Python
        # floats: see _engine.float_grid_spacing for why an int silently
        # costs 8 significant digits under torch.compile on CUDA.
        args = (*args[:1], _engine.float_grid_spacing(args[1]), *args[2:])
        _forward_step_opt = _forward_step_cache.resolve(backend)
        return scalar_born_python(*args, backend=backend)

    # The compiled backend now threads the first-class boundary codes through
    # the ``bc`` ABI, so move ``boundary`` to the *last* leading (non-tensor)
    # autograd-Function argument (mirroring scalar's ``_N_LEAD`` convention)
    # and keep it, rather than stripping it.
    boundary = args[_BOUNDARY_IDX]
    args = (
        args[:_BOUNDARY_IDX]
        + args[_BOUNDARY_IDX + 1 : _N_LEADING_ARGS]
        + (boundary,)
        + args[_N_LEADING_ARGS:]
    )

    func = ScalarBornForwardFunc.apply

    # ``forward`` appends a ``ForwardSavedState`` trailing output for
    # ``setup_context``; strip it so callers see only the output tensors.
    return cast(
        "tuple[torch.Tensor, ...]",
        func(*args)[:-1],  # type: ignore[no-untyped-call]
    )


_common.document_propagator_module(ScalarBorn, scalar_born)

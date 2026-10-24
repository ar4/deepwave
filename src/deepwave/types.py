"""Public types for Deepwave's interface.

Home of the types that appear in Deepwave's public signatures and returns:
:class:`CallbackState`, :class:`PropagatorOutput`, :class:`RunInfo`,
:class:`MemoryEstimate`, the :class:`Callback` protocol, and the type
aliases that name the closed sets of values the propagators accept.

Every one of them is documented and supported under its ``deepwave.<Name>``
spelling, and each sets ``__module__ = "deepwave"`` so that ``repr``, a
``pickle`` round trip, ``mypy``'s output and the generated API extract all
say ``deepwave.RunInfo`` rather than naming this file.

A **leaf module**: at run time it imports only :mod:`torch`, the standard
library, :mod:`deepwave._exceptions` and the leaf pytree adapter, so
every other module in the package can import *from* it with no risk of an
import cycle. That is the whole reason it exists as a separate file rather
than living in ``_common`` or ``_engine``, which refer to each other. Do not
add a run-time import of any other ``deepwave`` module here: it would create
a cycle that only shows up under some import orders.

The aliases here name the shapes that Deepwave's public signatures actually
accept. Before them, the documented "you may pass a
:class:`deepwave.Hicks` object as any location argument" route was a
``mypy --strict`` error at every call site, because the parameters were
annotated ``Optional[torch.Tensor]``.
"""

import contextlib

# ``types`` and ``typing`` here are the standard library's, not this module:
# Python 3's absolute imports mean a bare ``types`` never resolves to
# ``deepwave.types``.
import os
from collections import abc
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
    cast,
    get_args,
    runtime_checkable,
)

import torch

from deepwave import _pytree
from deepwave._exceptions import (
    DeepwaveNotImplementedError as _DeepwaveNotImplementedError,
)
from deepwave._exceptions import (
    DeepwaveRuntimeError as _DeepwaveRuntimeError,
)
from deepwave._exceptions import (
    DeepwaveTypeError as _DeepwaveTypeError,
)
from deepwave._exceptions import (
    DeepwaveValueError as _DeepwaveValueError,
)
from deepwave._exceptions import (
    warn_deprecated_name as _warn_deprecated_name,
)

if TYPE_CHECKING:  # pragma: no cover -- import-cycle-free typing only
    from types import EllipsisType

    from deepwave._engine import AuxiliaryState
    from deepwave._location_interpolation import Hicks
    from deepwave._storage import Storage

__all__ = [
    "ON_NODE_TOLERANCE",
    "Accuracy",
    "Callback",
    "CallbackDirection",
    "CallbackKind",
    "CallbackMode",
    "CallbackRole",
    "CallbackState",
    "InitialState",
    "LocationInterp",
    "LocationsLike",
    "MemoryEstimate",
    "PropagatorOutput",
    "ReceiverSpec",
    "Reduction",
    "RunInfo",
    "SourceSpec",
    "StorageLike",
    "StorageMode",
    "StoragePath",
    "WavefieldView",
]


def _resolve_forward_refs_in(alias: Any, module: str) -> None:
    """Names the module each of ``alias``'s forward references belongs to.

    The names in a string annotation are looked up, by default, in the
    globals of the module whose signature is being inspected -- not in the
    module where the alias was written. An alias defined here in terms of a
    name this module only imports under ``TYPE_CHECKING`` therefore has
    nothing to resolve against at run time, and
    :func:`typing.get_type_hints` on any function annotated with it raises
    ``NameError``. Recording the defining module on the reference itself
    makes it resolve there instead, which needs no import and so keeps this
    module a leaf.

    ``__forward_module__`` is a typing implementation detail, so this is
    written to be inert rather than fatal on an interpreter that does not
    have it: the annotations then behave exactly as they did before, which
    is a missing convenience and not a new failure.
    """
    for arg in get_args(alias):
        if (
            isinstance(arg, ForwardRef)
            and getattr(arg, "__forward_module__", 1) is None
        ):
            with contextlib.suppress(AttributeError):  # older typing
                arg.__forward_module__ = module


#: Anything that can be given as a source or receiver location argument: a
#: ``torch.Tensor`` of cell indices with shape ``[shot, point, ndim]``, or a
#: :class:`deepwave.Hicks` object wrapping off-grid coordinates.
#:
#: Stability: stable (see :doc:`/stability`).
LocationsLike = Union[torch.Tensor, "Hicks"]
_resolve_forward_refs_in(LocationsLike, "deepwave._location_interpolation")

#: One entry of a ``sources=`` mapping: the amplitudes and locations of that
#: channel's sources, optionally followed by a boolean ``[shot, point]``
#: activity mask (``True`` = active).
#:
#: Stability: stable (see :doc:`/stability`).
SourceSpec = Union[
    "tuple[torch.Tensor, LocationsLike]",
    "tuple[torch.Tensor, LocationsLike, Optional[torch.Tensor]]",
]

#: One entry of a ``receivers=`` or ``bg_receivers=`` mapping: that channel's
#: receiver locations, either bare or paired with a boolean ``[shot, point]``
#: activity mask (``True`` = active).
#:
#: Stability: stable (see :doc:`/stability`).
ReceiverSpec = Union[
    "LocationsLike",
    "tuple[LocationsLike, Optional[torch.Tensor]]",
]

# The aliases below name closed sets of values that a runtime check already
# enforces. Each one records, in a comment, the check it mirrors, and
# ``tests/test_types.py::test_literal_aliases_match_their_runtime_validators``
# asserts the two agree -- so widening a validator without widening the alias
# (or the reverse) fails a test rather than silently mis-typing the API.
#
# ``boundary`` entries are deliberately **not** in this list: they stay
# ``str`` with runtime validation, because a ``Literal`` there would make
# adding a boundary condition a typing break for every user who annotated a
# variable. See :doc:`/stability`.

#: The values ``storage_mode`` accepts. Mirrors
#: ``deepwave._engine._ALL_STORAGE_MODES``.
#:
#: Not to be confused with the private ``deepwave._storage.StorageMode``, an
#: ``IntEnum`` of the backend's internal storage codes. This alias is the
#: public *string* spelling; the enum is never re-exported.
#:
#: Stability: stable (see :doc:`/stability`).
StorageMode = Literal["device", "cpu", "disk", "none"]

#: What ``storage_path`` accepts: a string, or any ``os.PathLike`` such as a
#: :class:`pathlib.Path`. ``None`` (the default everywhere) means the system
#: temporary directory, resolved when the call is made.
#:
#: Stability: stable (see :doc:`/stability`).
StoragePath = Union[str, os.PathLike[str]]

#: What ``storage`` accepts: a :data:`StorageMode` string, or a
#: :class:`deepwave.Storage` carrying the whole policy at once.
#:
#: Stability: provisional (see :doc:`/stability`), because
#: :class:`deepwave.Storage` is.
StorageLike = Optional[Union[str, "Storage"]]
_resolve_forward_refs_in(StorageLike, "deepwave._storage")

#: Which part of a wavefield, model or gradient a :class:`CallbackState`
#: accessor returns. A propagator works on a domain that is the region of the
#: model the run uses, surrounded by the absorbing layer (``pml_width`` cells
#: per face), surrounded in turn by the finite-difference halo the stencil
#: needs. The three values name the three nestings of that domain:
#:
#: - ``"full"``: everything, halo and absorbing layer included. Indices are
#:   the propagator's own, so cell ``(0, 0)`` is the outermost halo cell.
#: - ``"pml"``: the halo removed, the absorbing layer kept. Cell ``(0, 0)``
#:   is the outermost cell of the absorbing layer.
#: - ``"inner"`` (the default): the halo and the absorbing layer both
#:   removed, leaving exactly the region of the model the run used. Cell
#:   ``(0, 0)`` is the model cell at :attr:`RunInfo.origin`, so these indices
#:   are the ones a caller's own model arrays use.
#:
#: Stability: stable (see :doc:`/stability`).
WavefieldView = Literal["full", "pml", "inner"]

#: Which wavefield a ``backward_callback`` is asking for. Mirrors the
#: ``role`` check in :meth:`CallbackState.get_wavefield`.
#:
#: Stability: stable (see :doc:`/stability`).
CallbackRole = Literal["adjoint", "forward"]

#: Whether a :class:`CallbackState` accessor returns a tensor you intend to
#: write through (``"write"``) or only to read (``"read"``, a detached view).
#:
#: Stability: stable (see :doc:`/stability`).
CallbackMode = Literal["read", "write"]

#: Which of :class:`CallbackState`'s three name spaces
#: :meth:`CallbackState.writable` is being asked about. The same string can
#: name both a model and its gradient (``"v"`` is both), so the question is
#: only well posed once the space is named.
#:
#: Stability: stable (see :doc:`/stability`).
CallbackKind = Literal["wavefield", "model", "gradient"]

#: Which pass a callback is being called from. The value of
#: :attr:`CallbackState.direction`.
#:
#: Stability: stable (see :doc:`/stability`).
CallbackDirection = Literal["forward", "backward"]

#: The values ``location_interpolation`` accepts. Mirrors
#: ``deepwave._location_interpolation._LOCATION_INTERPOLATION_MODES``.
#:
#: Stability: stable (see :doc:`/stability`).
LocationInterp = Literal["auto", "hicks", "none"]

#: The values a misfit function's ``reduction`` accepts. The runtime tuple
#: ``deepwave.misfits._REDUCTIONS`` is derived from this alias.
#:
#: Stability: stable (see :doc:`/stability`).
Reduction = Literal["mean", "sum", "none"]

#: The spatial finite-difference orders the propagators implement. Mirrors
#: the ``accuracy`` validation in ``deepwave._common.check_inputs`` and the
#: critical-Courant-number tables.
#:
#: Annotation-only: passing a computed ``int`` still works at run time, but
#: ``mypy`` will reject it. Silence that with ``typing.cast``, e.g.
#: ``accuracy=cast(deepwave.Accuracy, order)``.
#:
#: Stability: stable (see :doc:`/stability`).
Accuracy = Literal[2, 4, 6, 8]

#: What ``initial_state`` accepts: a previous call's
#: :class:`PropagatorOutput`, or a mapping of physical field names to tensors
#: and ``"auxiliary_state"`` to an :class:`AuxiliaryState` bundle.
#:
#: Stability: stable (see :doc:`/stability`).
_StateValue = Union[torch.Tensor, "AuxiliaryState"]
_resolve_forward_refs_in(_StateValue, "deepwave._engine")
InitialState = Union["PropagatorOutput", abc.Mapping[str, _StateValue]]
_resolve_forward_refs_in(InitialState, "deepwave.types")

#: How close to a grid node a coordinate must be, **in cells**, for Deepwave
#: to treat it as being *on* that node.
#:
#: This one number decides three things:
#:
#: * ``location_interpolation="auto"`` builds a :class:`deepwave.Hicks`
#:   interpolator only if some coordinate is further than this from its
#:   component's node (but see that argument's documentation for the
#:   ``requires_grad`` exception, which always interpolates);
#: * ``location_interpolation="none"`` rejects a locations Tensor that has
#:   any coordinate further than this off-node;
#: * :func:`deepwave.to_cells` with ``snap=True`` cleans up floating-point
#:   error out to this distance.
#:
#: It is the single source for all three: ``Hicks``' default ``eps`` is this
#: value, and so is the propagators' internal node test. Changing it changes
#: which calls are interpolated, and therefore changes numbers.
#:
#: Stability: stable (see :doc:`/stability`).
ON_NODE_TOLERANCE = 1e-5


#: Callback wavefield/model/gradient names that shipped in ``v0.0.27`` and
#: have since been renamed, mapped to their current names. Looking one up
#: through :class:`CallbackState` still works, with a
#: :class:`deepwave.DeepwaveFutureWarning`;
#: these aliases are removed in 0.4.0.
#: ``scalar_born``'s scattering potential was ``scatter``; it is now ``dv``,
#: matching the propagator's own parameter and every other Born propagator's
#: perturbation naming.
_DEPRECATED_CALLBACK_NAMES = {"scatter": "dv"}


class CallbackState:
    """State provided to user callbacks.

    What you may write to
    ^^^^^^^^^^^^^^^^^^^^^

    Three separate promises, deliberately not one:

    1. :meth:`get_gradient` returns a **live view into the gradient
       accumulator, and mutating it in place is supported**. This accumulator
       is intermediate, even at the last callback. To scale the complete model
       gradient, use a PyTorch tensor gradient hook or transform ``model.grad``
       after ``.backward()``; see :doc:`/example_custom_imaging_condition`.
       The internal accumulator is available only on the compiled backend,
       which is the only backend that populates the gradient dictionary at
       all: the eager backward path and every forward callback are given an
       empty one, so there is nothing to mutate there. Ask
       :meth:`writable` rather than assuming.
    2. :meth:`get_wavefield` and :meth:`get_model` return, in the default
       ``mode="write"``, **the tensor Deepwave itself holds, and writing
       into it in place changes the running simulation in inference calls**.
       During differentiable runs a forward callback may observe these tensors
       but may not mutate them: an arbitrary edit has no defined adjoint and
       raises :class:`deepwave.DeepwaveRuntimeError`. :meth:`set_wavefield` is
       the checked route into the
       wavefields -- it validates shape, dtype and device instead of letting
       a mismatch become a silent broadcast or a host-to-device copy in the
       middle of the time loop -- and is what to prefer where it applies.
       There is no ``set_model``; a model is written in place. A name that
       is *derived* from the internal state rather than being part of it is
       read-only, and :meth:`writable` is how to ask which a name is.
    3. Nothing you keep past the end of the callback is stable. ``"read"``
       mode gives a *detached view*, not a snapshot: the buffer it points at
       is overwritten at the next time step. **If you keep a tensor past the
       end of the callback, ``.clone()`` it** --
       :doc:`/example_callback_animation` is the modelled idiom.

    When PyTorch invokes a backward node with an undefined adjoint, the
    callback receives a zero tensor for that field rather than ``None``.
    PyTorch can also prune an unused backward node entirely, in which case
    its callback does not fire. A callback is not a finalisation hook.

    Attributes:
        step: User time-step index. Forward callbacks observe
            ``0, f, 2*f, ... < n_steps`` for ``callback_frequency=f``;
            backward callbacks traverse that same set in reverse at the
            matching state boundary. There is no extra callback at
            ``n_steps``; the return value carries the final state.
        direction: ``"forward"`` or ``"backward"``.
        n_steps: The number of user time steps in this call. PyTorch may prune
            backward callback nodes that are not on the requested gradient
            path. In particular, eager ``torch.autograd.grad`` can prune the
            step-zero hook when initial wavefields are not requested; a
            backward callback is not a guaranteed finalisation hook.
        dt: The caller's time step. ``time == step * dt`` is measured from
            the start of this call.
        internal_dt: The CFL-resolved time step used inside propagation.
        time: Time of this state boundary in the caller's units. Pressure and
            stress fields are at integer boundaries while staggered velocity
            components have their usual half-step offsets; ``time`` does not
            erase those component-specific offsets.

    In a ``forward_callback`` the wavefields are the forward wavefields; in a
    ``backward_callback`` they are the adjoint wavefields, and the
    time-coincident *forward* wavefields may additionally be available
    through ``role="forward"`` -- see :meth:`get_wavefield`,
    :attr:`has_forward_wavefield` and :attr:`available_roles`.
    """

    def __init__(
        self,
        dt: float,
        step: int,
        wavefields: dict[str, torch.Tensor],
        models: dict[str, torch.Tensor],
        gradients: dict[str, torch.Tensor],
        fd_pad: list[int],
        pml_width: list[int],
        forward_wavefields: Optional[dict[str, torch.Tensor]] = None,
        *,
        direction: CallbackDirection,
        n_steps: int,
        internal_dt: Optional[float] = None,
    ) -> None:
        """Initialise the state.

        Deepwave constructs this; a caller normally only receives one. A test
        double that constructs its own must supply every argument below.

        Args:
            dt: The user time step size.
            step: The current user time-step index.
            wavefields: A dictionary of wavefields.
            models: A dictionary of models.
            gradients: A dictionary of gradients.
            fd_pad: The padding used for the finite difference stencil.
            pml_width: The width of the PML.
            forward_wavefields: Optionally, in a ``backward_callback``, a
                dictionary of the *forward* wavefields reconstructed at the
                same time step as the adjoint wavefields in ``wavefields``.
            direction: Which pass this state belongs to.
            n_steps: The number of user time steps in this pass.
            internal_dt: The CFL-resolved time step.
        """
        self.dt = dt
        self.internal_dt = dt if internal_dt is None else internal_dt
        self.step = step
        self.time = step * dt
        self.direction = direction
        self.n_steps = n_steps
        self._wavefields = wavefields
        self._models = models
        self._gradients = gradients
        self._fd_pad = fd_pad
        self._pml_width = pml_width
        self._forward_wavefields = forward_wavefields
        self._forward_mutation_allowed = True
        self._ndim = next(iter(models.values())).ndim - 1

    @property
    def has_forward_wavefield(self) -> bool:
        """Whether ``get_wavefield(..., role="forward")`` will work here.

        The branch-on-able form of the :exc:`NotImplementedError`
        :meth:`get_wavefield` raises for ``role="forward"``: ``True`` exactly
        when the backend supplied the reconstructed forward wavefield for
        this callback. Lets one callback be written that computes an imaging
        condition where the forward field is available and falls back to
        inspecting the adjoint field where it is not, without a
        ``try``/``except``.
        """
        return self._forward_wavefields is not None

    @property
    def available_roles(self) -> tuple[str, ...]:
        """The ``role`` values :meth:`get_wavefield` will accept here.

        ``("adjoint",)``, or ``("adjoint", "forward")`` when the backend
        supplied the forward wavefield. ``None``, which means the same as
        ``"adjoint"``, is not listed: it is not a distinct capability to
        branch on.
        """
        if self._forward_wavefields is not None:
            return ("adjoint", "forward")
        return ("adjoint",)

    def get_wavefield(
        self,
        name: str,
        view: WavefieldView = "inner",
        role: Optional[CallbackRole] = None,
        mode: CallbackMode = "write",
    ) -> torch.Tensor:
        """Get a wavefield.

        Args:
            name: The name of the wavefield.
            view: The part of the wavefield to return. One of
                'inner' (the model region, default),
                'pml' (the model region plus the PML), or
                'full' (the entire wavefield, including finite
                difference padding).
            role: Which wavefield to return, when both are available inside a
                ``backward_callback``. ``None`` (default) or ``"adjoint"``
                returns the primary wavefield of the callback (the forward
                field in a forward callback, the adjoint field in a backward
                callback). ``"forward"`` returns the reconstructed
                *forward* wavefield time-coincident with the adjoint field,
                for building custom imaging conditions; it is available only
                when the backend supplied forward wavefields, which
                :attr:`has_forward_wavefield` and :attr:`available_roles`
                report. Under ``storage_compression=True`` it is
                reconstructed from the lossy 8-bit history, so it is an
                approximation of the forward field (of order 0.1% of the
                local block maximum) rather than the field the forward pass
                propagated.
            mode: ``"write"`` (the default) returns exactly the tensor
                Deepwave holds. ``"read"`` returns ``.detach()`` of it,
                removing the autograd edge and stating that you do not intend
                to write through it.

                ``"read"`` is a *detached view*, not a copy: it does not make
                the tensor immutable and it does not stop the buffer being
                overwritten at the next time step. **If you keep it past the
                end of the callback, ``.clone()`` it.**

                ``"write"`` is the **permanent** default: it is not
                scheduled to flip in any release (see :doc:`/stability`).
                Writing into the returned tensor in place is supported for
                an internal wavefield name, exactly as in ``v0.0.27``;
                :meth:`writable` reports which names those are, and
                :meth:`set_wavefield` is the checked route that validates
                shape, dtype and device for you.

        Returns:
            The specified part of the wavefield.

        Raises:
            NotImplementedError: If ``role="forward"`` is asked for where the
                backend did not supply the forward wavefield.
        """
        if role == "forward":
            if self._forward_wavefields is None:
                raise _DeepwaveNotImplementedError(
                    'get_wavefield(role="forward") requires the forward '
                    "wavefield to be retained. Pass "
                    "backward_callback_needs_forward_wavefield=True to the "
                    "propagator (deepwave.scalar(...), say) to retain it. "
                    "All nine accept that argument, but only a propagator "
                    "whose spec reports has_forward_wavefield implements "
                    'it: deepwave.spec("<propagator>").has_forward_wavefield '
                    "says which, and passing True to one of the others "
                    "raises there rather than here. It is provided by "
                    "scalar's backend=\"eager\" backend and by scalar's "
                    "default compiled C/CUDA backend, in both cases only "
                    "with that argument. Record the forward field with a "
                    "forward_callback instead."
                )
            return self._mode(
                self._get_view(
                    self._lookup(self._forward_wavefields, name, "forward wavefield"),
                    view,
                ),
                mode,
            )
        if role not in (None, "adjoint"):
            # ``role="backward"`` existed only on the development branch, as a
            # synonym for ``"adjoint"``; no release ever accepted it, so it is
            # named here as the wrong spelling rather than deprecated. Compared
            # through an ``object`` binding because the annotation no longer
            # admits it, while a run-time caller can still pass it.
            given: object = role
            extra = (
                ' Use role="adjoint", which is what "backward" meant.'
                if given == "backward"
                else ""
            )
            raise _DeepwaveValueError(
                f'role must be None, "adjoint", or "forward", got {role!r}.{extra}'
            )
        return self._mode(
            self._get_view(self._lookup(self._wavefields, name, "wavefield"), view),
            mode,
        )

    def get_model(
        self,
        name: str,
        view: WavefieldView = "inner",
        mode: CallbackMode = "write",
    ) -> torch.Tensor:
        """Get a model.

        Args:
            name: The name of the model.
            view: The part of the model to return. One of
                'inner' (the model region, default),
                'pml' (the model region plus the PML), or
                'full' (the entire model, including finite
                difference padding).
            mode: ``"write"`` (the default) returns exactly the tensor
                Deepwave holds; ``"read"`` returns ``.detach()`` of it. As for
                :meth:`get_wavefield`, ``"read"`` is a detached *view*, not a
                copy -- **``.clone()`` it if you keep it past the end of the
                callback** -- and ``"write"`` is the **permanent** default,
                not scheduled to flip in any release (see
                :doc:`/stability`).

        Returns:
            The specified part of the model. For an internal model name,
            writing into it in place is supported and changes the running
            simulation, as it did in ``v0.0.27``; there is no ``set_model``
            because writing in place *is* the route. ``writable(name,
            "model")`` reports whether that holds for this name.
        """
        return self._mode(
            self._get_view(self._lookup(self._models, name, "model"), view), mode
        )

    def get_gradient(
        self,
        name: str,
        view: WavefieldView = "inner",
        mode: CallbackMode = "write",
    ) -> torch.Tensor:
        """Get a gradient.

        Args:
            name: The name of the gradient.
            view: The part of the gradient to return. One of
                'inner' (the model region, default),
                'pml' (the model region plus the PML), or
                'full' (the entire gradient, including finite
                difference padding).
            mode: ``"write"`` (the default) returns the live accumulator;
                ``"read"`` returns ``.detach()`` of it. Unlike
                :meth:`get_wavefield` and :meth:`get_model`, this one is
                never a detached snapshot of a buffer that is about to be
                overwritten: mutating the gradient is the point. As on those
                two, ``"write"`` is the **permanent** default.

        Returns:
            The specified part of the gradient, as a live view into the
            intermediate gradient accumulator: mutating it in place changes
            the gradient Deepwave ultimately returns, and the changed result
            need not be a derivative of the unchanged forward loss. This is
            not the completed gradient with respect to the caller's model.
            Only the compiled
            backend populates gradients;
            :meth:`writable` reports whether this one does.
        """
        return self._mode(
            self._get_view(self._lookup(self._gradients, name, "gradient"), view), mode
        )

    def set_wavefield(
        self,
        name: str,
        tensor: torch.Tensor,
        view: WavefieldView = "inner",
    ) -> None:
        """Write into a wavefield of the running simulation.

        The supported way to alter the simulation state from a callback.
        ``view`` mirrors :meth:`get_wavefield`'s default, so
        ``state.set_wavefield(n, state.get_wavefield(n) + 1)`` round-trips.

        The value is copied into Deepwave's own tensor; ``tensor`` itself is
        not retained, and no broadcasting, casting or device transfer is
        performed. A mismatch raises rather than silently doing the
        conversion, because a silent host-to-device copy in the middle of a
        time loop is a performance trap and a silent cast loses precision
        where it is least visible.

        Args:
            name: The name of the wavefield to write into.
            tensor: The values to write. Must match the target's shape, dtype
                and device exactly.
            view: The part of the wavefield ``tensor`` corresponds to, as for
                :meth:`get_wavefield`.

        Raises:
            DeepwaveValueError: If ``tensor``'s shape or device differs from
                the target's.
            DeepwaveTypeError: If ``tensor``'s dtype differs from the
                target's.
            KeyError: If ``name`` is not a wavefield of this callback, listing
                the names that are.
        """
        target = self._get_view(self._lookup(self._wavefields, name, "wavefield"), view)
        if not self._forward_mutation_allowed:
            raise _DeepwaveRuntimeError(
                "A forward_callback may not change a wavefield during a "
                "differentiable run. The callback has no adjoint for that edit; "
                "use an observational callback or run under torch.no_grad()."
            )
        if tensor.shape != target.shape:
            raise _DeepwaveValueError(
                f"set_wavefield({name!r}, ..., view={view!r}) expects a tensor "
                f"of shape {list(target.shape)} (the shape "
                f"get_wavefield({name!r}, view={view!r}) returns), but the "
                f"given tensor has shape {list(tensor.shape)}."
            )
        if tensor.dtype != target.dtype:
            raise _DeepwaveTypeError(
                f"set_wavefield({name!r}, ...) expects a tensor of dtype "
                f"{target.dtype}, matching the simulation, but the given "
                f"tensor has dtype {tensor.dtype}. Cast it explicitly if that "
                "is what you intend."
            )
        if tensor.device != target.device:
            raise _DeepwaveValueError(
                f"set_wavefield({name!r}, ...) expects a tensor on device "
                f"{target.device}, where the simulation is running, but the "
                f"given tensor is on {tensor.device}. Move it explicitly if "
                "that is what you intend."
            )
        target.copy_(tensor)

    def writable(self, name: str, kind: CallbackKind = "wavefield") -> bool:
        """Whether writing to ``name`` is supported in this callback.

        The branch-on-able form of the class docstring's three promises, so
        one callback can be written against all nine propagators and both
        passes without a ``try``/``except`` around every write. It reports
        whether a write is *supported*, not merely whether one would happen
        to land somewhere today.

        Args:
            name: The name to ask about.
            kind: Which name space ``name`` belongs to. The same string can be
                both a model and a gradient name (``"v"`` is both), so the
                space has to be named for the question to be well posed.

        Returns:
            For ``kind="gradient"``, whether that gradient is present and so
            can be mutated in place through :meth:`get_gradient` (true only on
            the compiled backend's backward pass). For ``kind="wavefield"``,
            whether :meth:`set_wavefield` will accept that name, which is also
            whether ``get_wavefield(name)`` may be written into in place. For
            ``kind="model"``, whether ``get_model(name)`` may be written into
            in place: true for an internal model name this callback holds,
            false for a name that is merely *derived* from the internal state
            (there is nothing behind such a name to write into). An unknown
            name returns ``False`` rather than raising -- this is a query.

            A deprecated name that the accessors still resolve (the
            ``DEPRECATED_CALLBACK_NAMES`` aliases) answers for the name it
            resolves to, so this agrees with the accessor it predicts. Being
            a query, it does not warn about the alias; the accessor does.

            During a differentiable forward pass, model and wavefield writes
            are unsupported and return ``False`` even when the name is present.
        """
        if kind == "gradient":
            return self._present(self._gradients, name)
        if kind == "wavefield":
            return self._forward_mutation_allowed and self._present(
                self._wavefields, name
            )
        if kind == "model":
            return self._forward_mutation_allowed and self._present(self._models, name)
        raise _DeepwaveValueError(
            f'kind must be "wavefield", "model", or "gradient", got {kind!r}.'
        )

    @staticmethod
    def _present(mapping: dict[str, torch.Tensor], name: str) -> bool:
        """Whether :meth:`_lookup` would find ``name``, without warning.

        The silent half of :meth:`_lookup`'s name resolution, for
        :meth:`writable`: a query must answer for the same name set the
        accessor accepts, deprecated aliases included, but must not emit the
        deprecation warning for merely being asked.
        """
        if name in mapping:
            return True
        current = _DEPRECATED_CALLBACK_NAMES.get(name)
        return current is not None and current in mapping

    @staticmethod
    def _mode(x: torch.Tensor, mode: CallbackMode) -> torch.Tensor:
        """Applies ``mode`` to an already-sliced accessor result."""
        if mode == "write":
            return x
        if mode == "read":
            return x.detach()
        raise _DeepwaveValueError(f'mode must be "read" or "write", got {mode!r}.')

    @staticmethod
    def _lookup(mapping: dict[str, torch.Tensor], name: str, kind: str) -> torch.Tensor:
        """Looks ``name`` up in ``mapping``, raising a KeyError listing what exists.

        Shared by :meth:`get_wavefield` (both its ``role="forward"`` and
        primary-wavefield lookups), :meth:`get_model`, and :meth:`get_gradient`:
        callback names (``K``, ``Bz``, ``m1``, ``mu_zy``, ...) are internal and
        unguessable, so a bare ``KeyError`` on a typo leaves a user with no way
        to find the right name; listing what is actually present does.

        A name in ``DEPRECATED_CALLBACK_NAMES`` resolves to its current
        name (with a :class:`deepwave.DeepwaveFutureWarning`) when that
        current name is present, so callbacks written against a released name
        keep working. Those aliases are deprecated as of 0.1.0 and are
        removed in 0.4.0 (see the stability policy documentation page).
        """
        if name not in mapping:
            current = _DEPRECATED_CALLBACK_NAMES.get(name)
            if current is not None and current in mapping:
                _warn_deprecated_name(
                    f"The {kind} name {name!r}", repr(current), stacklevel=3
                )
                return mapping[current]
            raise KeyError(
                f"{kind} {name!r} is not available; {kind}s present: {sorted(mapping)}."
            )
        return mapping[name]

    def _get_view(self, x: torch.Tensor, view: WavefieldView) -> torch.Tensor:
        if view == "full":
            return x
        if view == "pml":
            idx: tuple[Union[EllipsisType, slice], ...] = (
                ...,
                *(
                    slice(
                        self._fd_pad[dim * 2],
                        x.shape[-self._ndim + dim] - self._fd_pad[dim * 2 + 1],
                    )
                    for dim in range(self._ndim)
                ),
            )
            return x[idx]
        if view == "inner":
            idx_inner: tuple[Union[EllipsisType, slice], ...] = (
                ...,
                *(
                    slice(
                        self._fd_pad[dim * 2] + self._pml_width[dim * 2],
                        x.shape[-self._ndim + dim]
                        - self._fd_pad[dim * 2 + 1]
                        - self._pml_width[dim * 2 + 1],
                    )
                    for dim in range(self._ndim)
                ),
            )
            return x[idx_inner]
        raise _DeepwaveValueError(
            f"view must be 'full', 'pml', or 'inner', but got {view}"
        )


CallbackState.__module__ = "deepwave"


@runtime_checkable
class Callback(Protocol):
    """A propagator callback: called with the live :class:`CallbackState`.

    Anything callable with one positional argument satisfies it -- a
    ``def``, a ``lambda``, a bound method, a ``functools.partial``, or an
    object with a ``__call__``. The parameter is **positional-only** (the
    ``/`` in the signature below), so your callback's parameter may be named
    anything; a protocol without it would silently reject every callback
    whose parameter is not called ``state``.

    ``runtime_checkable`` is set, so ``isinstance(cb, deepwave.Callback)``
    works -- but be aware that, as for every runtime-checkable protocol, it
    checks only for the *presence* of ``__call__``, not the signature. It
    will therefore accept a callable of the wrong arity; use it as a
    sanity check, not as validation.

    Example::

        def report(state: deepwave.CallbackState) -> None:
            print(state.step, state.get_wavefield("wavefield_0").norm())

        out = deepwave.scalar(v, dx, dt, ..., forward_callback=report)

    Stability: stable (see :doc:`/stability`).
    """

    def __call__(self, state: CallbackState, /) -> None:
        """Receives the live state for one time step."""
        ...


Callback.__module__ = "deepwave"


class MemoryEstimate(TypedDict):
    """The dict returned by :func:`estimate_memory`."""

    padded_model_shape: tuple[int, ...]
    padded_model_elements: int
    n_threads: int
    forward_wavefields_bytes: int
    backward_storage_bytes: int
    gradient_scratch_bytes: int
    auxiliary_state_bytes: int
    total_bytes_lower_bound: int


MemoryEstimate.__module__ = "deepwave"


@dataclass(frozen=True)
class RunInfo:
    """Non-differentiable metadata about how a propagator call ran.

    Every propagator attaches one to its return, as ``out.info``. It records
    what the call *resolved* -- the values the engine actually used, which are
    not always the ones the caller asked for (``dt`` may be sub-stepped to
    satisfy the CFL condition, ``pml_freq`` may be inferred from the source
    spectrum, the survey region may be cut down around the sources and
    receivers) -- so that a later call can be pinned to the same choices.

    Existing fields follow the stable compatibility policy (see
    :doc:`/stability`). New resolved metadata may be added as attributes;
    an existing field is changed only through that policy.

    Attributes:
        dt: The user-level time step, in the caller's own time unit, exactly
            as passed. ``internal_dt`` is this divided by ``step_ratio``.
        grid_spacing: The cell size along each dimension, in the caller's own
            length unit, one entry per dimension (a scalar argument is
            expanded). Resolved.
        accuracy: The finite-difference order of accuracy in space.
        boundary: The per-face boundary condition, length ``2 * ndim``, in
            the face order ``(begin dim 0, end dim 0, begin dim 1, ...)``,
            spelled as the ``boundary=`` argument spells them (``"pml"``,
            ``"free"``, ``"rigid"``, ``"periodic"``). Resolved, including the
            faces a zero ``pml_width`` turned into something else.
        mpml_fraction: The multiaxial-PML cross-damping fraction used (see
            :func:`deepwave.elastic`). 0.0 on a standard C-PML run.
        surface_method: How the free surface implied by elastic vacuum cells
            was discretised (see :func:`deepwave.elastic`).
        q_freq: The reference frequency of the attenuation model, or ``None``
            on a propagator without one.
        q_n_mechanisms: The number of relaxation mechanisms in the
            attenuation model (1 where there is none).
        q_band: The frequency band the constant-``Q`` fit spans, or ``None``.
        step_ratio: The number of internal time steps taken per user-level
            time sample. 1 unless the user ``dt`` exceeded the CFL limit.
            Resolved.
        model_gradient_sampling_interval: The positive integer requested by
            the caller, including when no model gradient is active.
        model_gradient_stride: The effective native history/correlation stride
            in internal steps (``step_ratio * model_gradient_sampling_interval``),
            1 for active eager model differentiation, or ``None`` when no
            model gradient is active. It describes sampling, not the accuracy
            of the complete parameter derivative.
        internal_dt: The internal time step, in the time units of ``dt``: the
            user ``dt`` divided by ``step_ratio``. Resolved.
        courant_number: The Courant number the internal time step was chosen
            to satisfy. The user's ``courant_number`` argument when one was
            given, otherwise the propagator's conservative default. Resolved.
        nt: The number of user-level time samples propagated. The caller's
            ``nt`` when one was given (the source amplitudes are padded or
            truncated to it), otherwise the length of the source amplitudes.
            Resolved.
        pml_width: The per-face PML widths, in cells, length ``2 * ndim``, in
            the face order ``(begin dim 0, end dim 0, begin dim 1, ...)``.
            Resolved (a scalar argument is expanded to every face).
        pml_freq: The PML frequency, in the reciprocal of ``dt``'s time unit
            (Hz when ``dt`` is in seconds): the ``pml_freq`` argument when one
            was given, otherwise the value inferred from the source
            amplitudes' spectrum (or ``0.5 / dt / 5`` when there were no
            sources). Pass it back in as ``pml_freq=`` to pin a later run to
            the same absorbing boundary. Resolved.
        origin: The index, in cells, in the full model, of cell 0 of the
            region the run extracted, one entry per dimension. Resolved (from
            the caller's ``origin``, from ``survey_pad``, or from the source
            and receiver locations).
        extent: The spatial shape of the region extracted from the model, in
            cells, before FD and PML padding, one entry per dimension.
            Resolved.
        max_vel: The conservative speed bound, including density/stiffness
            weighting of staggered operators, in ``grid_spacing`` units per ``dt``
            unit, used to choose the internal time step and to build the PML
            profiles: the ``max_vel`` argument when one was given, otherwise
            the maximum over the model.
            Resolved. ``None`` if the propagator did not record it.
        ndim: The number of spatial dimensions. Resolved (from the model and
            location shapes when it was not given explicitly). 0 if the
            propagator did not record it.
        dtype: The floating-point dtype the simulation ran in. Resolved from
            the models. ``None`` if the propagator did not record it.
        device: The device the simulation ran on. Resolved from the models.
            ``None`` if the propagator did not record it.
        model_batched: One flag per model, in the propagator's model order,
            saying whether that model was given per shot (an extra leading
            batch dimension) rather than shared by every shot. Resolved.
            Empty if the propagator did not record it.
        wavefield_names: The public state field names of this call, in return
            order -- exactly the keys of
            :attr:`PropagatorOutput.final_state`, and therefore the names
            ``initial_state`` accepts. Depends on ``ndim`` and, for the
            attenuating propagators, on the number of relaxation mechanisms.
            Empty if the propagator did not record it.
        receiver_names: The receiver-amplitude field names of this call, in
            return order. Empty if the propagator did not record it.
        location_interpolation_used: The location interpolation actually used
            for this call's source and receiver coordinates: ``"none"`` for
            the exact (on-cell) path, ``"hicks"`` for Kaiser-windowed sinc
            interpolation, or ``"mixed"`` when one kind of location was split
            across the two paths -- some of a propagator's source channels
            interpolated and others not, say. Sources are compared with
            sources and receivers with receivers, never with each other, so
            an off-node source with on-node receivers is ``"hicks"``, not
            ``"mixed"``; only a propagator with several channels of one kind
            can report ``"mixed"``. A channel with no locations (``None``, an
            empty Tensor, or nothing but :data:`deepwave.IGNORE_LOCATION`) has
            taken neither path and is not counted. ``"auto"`` resolves to one
            of those. The empty string if the propagator did not record it.

            This is how to tell whether ``location_interpolation="auto"``
            routed your coordinates through Hicks -- in particular for
            on-node positions with ``requires_grad=True``, which are
            *always* interpolated (only that path carries gradients back to
            the positions), and so give different forward values from the
            same positions without ``requires_grad``.
        backend: The backend that ran the forward pass, as one of
            ``"compiled"``, ``"eager"`` or
            ``"torch_compile"``. Resolved, so it is never ``"auto"``: that is
            a request, and this is the answer to it. This is how to find out
            what ``backend="auto"`` chose -- including inside a distributed
            worker, where a warning would not have been seen. The empty
            string if the propagator did not record it.
        backward_backend: The backend that would run the backward pass, in
            the same vocabulary. It is reported separately because a
            propagator is permitted to pair a compiled forward with an
            adjoint that runs on a different backend, but **no configuration
            reachable today does**: on every propagator and every backend it
            is currently equal to :attr:`backend`. (The one near-miss is
            :func:`deepwave.elastic_born` under M-PML, whose adjoint would
            use an eager-recompute bridge -- but its compiled forward raises
            for ``mpml_fraction > 0`` before any :class:`RunInfo` is built.) Compare
            the two rather than assuming they agree; do not rely on their
            being equal. The empty string if the propagator did not record
            it.
        storage: The fully resolved storage policy, as a
            :class:`deepwave.Storage`. ``path`` is the directory that was
            actually chosen (``mode="disk"`` only), so a run that fell back to
            the system temporary directory can be read back off here rather
            than guessed at. ``mode`` is the *resolved* mode, which records the
            ``"cpu"``-to-``"device"`` downgrade the engine performs when the
            compute device is already the CPU -- otherwise invisible. ``None``
            if the propagator did not record it.
        lossy_compression_used: Whether the wavefield history was actually
            stored with lossy compression. Resolved, not requested: it is
            ``False`` whenever no history was kept at all (``mode="none"``),
            even if compression was asked for, matching what the storage layer
            does.

    """

    step_ratio: int
    internal_dt: float
    courant_number: float
    nt: int
    pml_width: tuple[int, ...]
    pml_freq: float
    origin: tuple[int, ...]
    extent: tuple[int, ...]
    dt: float = 0.0
    grid_spacing: tuple[float, ...] = ()
    accuracy: int = 0
    boundary: tuple[str, ...] = ()
    mpml_fraction: float = 0.0
    surface_method: str = ""
    q_freq: Optional[float] = None
    q_n_mechanisms: int = 1
    q_band: Optional[tuple[float, float]] = None
    max_vel: Optional[float] = None
    ndim: int = 0
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    model_batched: tuple[bool, ...] = ()
    wavefield_names: tuple[str, ...] = ()
    receiver_names: tuple[str, ...] = ()
    location_interpolation_used: str = ""
    backend: str = ""
    backward_backend: str = ""
    storage: Optional["Storage"] = None
    lossy_compression_used: bool = False
    model_gradient_sampling_interval: int = 1
    model_gradient_stride: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        """Raises ``AttributeError``, naming ``internal_dt`` for ``dt_internal``.

        ``dt_internal`` was the development branch's spelling of
        :attr:`internal_dt` and no release ever had it, so it is not a field.
        It is named here because the two spellings are easy to confuse and a
        bare ``AttributeError`` would not say which way round the words go.
        Only reached for a name normal lookup did not find, so it costs
        nothing on any real attribute.
        """
        if name == "dt_internal":
            raise AttributeError(
                "RunInfo has no attribute 'dt_internal'; the internal time "
                "step is RunInfo.internal_dt."
            )
        raise AttributeError(f"RunInfo has no attribute {name!r}")


RunInfo.__module__ = "deepwave"


#: The three receiver-output field prefixes, longest first. Used to split a
#: propagator's output names into the channel-keyed views
#: :attr:`PropagatorOutput.receivers` and
#: :attr:`PropagatorOutput.bg_receivers`.
_RECEIVER_PREFIXES: tuple[str, ...] = (
    "bg_receiver_amplitudes",
    "sc_receiver_amplitudes",
    "receiver_amplitudes",
)


class PropagatorOutput(tuple[Any, ...]):
    """Named, attribute-accessible view of a propagator's outputs.

    Returned by every propagator. Each final wavefield and receiver-amplitude
    tensor is accessible by name (e.g. ``out.receiver_amplitudes``), which
    removes the most common indexing mistake: the position of
    ``receiver_amplitudes`` in the positional return shifts with the number of
    spatial dimensions, so code written for 2-D silently grabs the wrong tensor
    in 3-D.

    The object *is* a ``tuple`` (it subclasses ``tuple`` rather than merely
    implementing ``collections.abc.Sequence``): iterating, unpacking, and
    integer or slice indexing yield the values in the current positional
    return order. Positional unpacking therefore remains available, though
    it should be reviewed against the compatibility policy when upgrading::

        out = scalar(...)
        out.receiver_amplitudes        # by name (ndim-independent)
        *state, receivers = out        # by position, as before

    For a fixed propagator and dimensionality, the current field names and
    tuple order follow the stable compatibility policy (see :doc:`/stability`).
    Future metadata is added as attributes, not tuple elements. Named access
    is recommended: a positive index can silently mean a different field on
    another propagator or in another number of dimensions. For example,
    ``out[-2]`` is the opaque auxiliary state on ``scalar``, but a receiver
    field on ``scalar_born``.

    Exactly one element is not a ``torch.Tensor``: ``auxiliary_state``, a
    :class:`AuxiliaryState` holding the absorbing boundary's memory variables
    and any other auxiliary state, which sits at the index the first of those
    fields used to occupy. It is a real tuple element (so ``len(out)``,
    iteration and unpacking see it) and also a typed property, and it is what
    ``initial_state=`` consumes to continue a simulation. Code that iterates
    the whole tuple expecting tensors -- ``sum(o.sum() for o in out)`` and the
    like -- must skip it or iterate its tensors.

    Subclassing ``tuple`` (not ``abc.Sequence``) is deliberate and load-bearing:
    :class:`torch.autograd.Function` and :mod:`torch.utils.checkpoint` propagate
    ``requires_grad`` to an output only when it is a ``Tensor`` or a genuine
    ``tuple``. An ``abc.Sequence`` that is not a ``tuple`` is
    silently detached (the root cause of issue #120), so returning a
    ``PropagatorOutput`` straight through ``torch.utils.checkpoint`` would break
    gradients if it were not a real ``tuple``. Being a real ``tuple`` makes the
    named return compose with checkpointing exactly like the plain tuple return
    would; the non-``Tensor`` ``auxiliary_state`` element does not change that
    for the tensor elements, and outside a checkpoint the tensors it holds keep
    their ``grad_fn``.

    ``torch.utils.checkpoint(..., use_reentrant=True)`` is the exception, and
    the reason to prefer ``use_reentrant=False`` (which PyTorch also
    recommends) when a checkpointed call's state is fed onward. The reentrant
    implementation runs the wrapped function under ``torch.no_grad()`` and
    re-attaches only the ``Tensor`` elements of what it returns, so:

    * the return is degraded to a plain ``tuple``, losing the field names and
      ``info`` -- pre-existing, and independent of ``auxiliary_state``; and
    * every tensor held *inside* ``auxiliary_state`` comes back detached, so
      continuing a simulation from it gives no gradient through the absorbing
      boundary. ``use_reentrant=False`` preserves both the subclass and those
      gradients. A function wrapped in a reentrant checkpoint that needs the
      auxiliary state downstream must return its tensors as top-level elements
      of its own return value, which is what
      :func:`deepwave.checkpointing.run_segments` does internally.

    The public state fields use the same names as the propagator's
    initial-state arguments (``wavefield_0``, ``wavefield_m1``, ...), so an
    output can be fed straight back in to continue a simulation. Fields that do
    not exist for the given number of dimensions (e.g. the ``z`` components in
    a 2-D simulation) are simply absent. :attr:`final_state` is the mapping of
    exactly the fields ``initial_state=`` consumes.

    ``info`` is a :class:`RunInfo` carrying non-differentiable metadata about
    how the call actually ran (resolved ``step_ratio``, internal ``dt``,
    Courant number, ``nt``, PML width, PML frequency, survey origin/extent and
    more). It is plain attribute metadata, not a tuple element: it does not
    appear in iteration, unpacking, ``len()``, or indexing. A propagator always
    supplies one. It is unreachable in exactly two situations: under
    ``torch.utils.checkpoint(..., use_reentrant=True)``, where the return is
    degraded to a plain ``tuple`` that has no attributes at all, and under
    ``torch.nn.DataParallel.gather``, which does not reach the point of
    building an output.

    Registered as a **pytree node**, so ``torch.func`` and ``torch.compile``
    see the tensors inside it rather than one opaque object: a function that
    returns a whole output can be wrapped in :func:`torch.func.vmap`, carried
    as the auxiliary value of :func:`torch.func.grad` or
    :func:`torch.func.jacrev` (``has_aux=True``), compiled, or walked with
    ``torch.utils._pytree``'s ``tree_map``. The children are the tuple elements
    in return order (with ``auxiliary_state``, itself a pytree node, expanding
    into the tensors it holds, so the leaves are exactly the output's tensors);
    the field names and ``info`` travel as the node's context, because the
    names vary per instance and the object cannot be rebuilt without them. A
    rebuilt output therefore still carries the names and the ``info`` of the
    run it came from, and its ``auxiliary_state`` still carries the recorded
    run parameters -- ``format_version`` included -- that make it checkable as
    an ``initial_state=``.

    **Not usable as a full ``torch.nn.DataParallel`` return value on more than
    one GPU** (``DistributedDataParallel`` is unaffected). A ``forward()``
    wrapping a propagator should return a single tensor (e.g.
    ``out.receiver_amplitudes``) or a small tuple of plain tensors instead;
    see :doc:`/example_distributed` for the pattern and
    :doc:`/design/propagator_output` for why this is unsupported by design.
    """

    _names: tuple[str, ...]
    info: "RunInfo"

    #: Field names that are allowed to shadow a class attribute, because that
    #: attribute exists precisely to expose the field with a useful static
    #: type. Every other collision is rejected (see ``__new__``).
    _SHADOWABLE: frozenset[str] = frozenset({"auxiliary_state"})

    #: Reserved instance attributes. They live in the instance ``__dict__``
    #: rather than on the class, so ``hasattr(cls, name)`` does not see them,
    #: but a field of the same name would still shadow the instance attribute.
    #: ``_final_state`` is the cache :attr:`final_state` writes.
    _RESERVED: frozenset[str] = frozenset(
        {"_names", "info", "_final_state", "_receivers", "_bg_receivers"}
    )

    def __new__(
        cls,
        names: abc.Sequence[str],
        values: Optional[abc.Sequence[Any]] = None,
        *,
        info: Optional["RunInfo"] = None,
    ) -> "PropagatorOutput":
        """Builds the tuple of ``values`` and records the field ``names``."""
        if values is None:
            raise _DeepwaveTypeError(
                "PropagatorOutput cannot be reconstructed from a single "
                "iterable of values, which is what torch.nn.DataParallel."
                "gather does when a module's forward() returns one. "
                "PropagatorOutput's field names vary per instance (they "
                "depend on ndim and, for viscoacoustic, on "
                "q_n_mechanisms), so they cannot be recovered from the class "
                "alone. Return plain tensors from forward() instead, e.g. "
                "`return out.receiver_amplitudes`. See "
                "docs/example_distributed.rst."
            )
        if info is None:
            raise _DeepwaveTypeError(
                "PropagatorOutput requires info=<RunInfo>: every output "
                "carries the metadata of the run that produced it."
            )
        names = tuple(names)
        values = tuple(values)
        if len(names) != len(values):
            raise AssertionError(
                f"PropagatorOutput got {len(names)} names but {len(values)} values."
            )
        # A field whose name collides with an existing attribute or method
        # (e.g. ``index``/``count`` from ``tuple``, or ``_asdict``) would be
        # shadowed: ``out.<name>`` would silently return the method instead of
        # the tensor, because __getattr__ only runs on a failed lookup. Reject
        # such names loudly rather than hand back the wrong object. This fires
        # deterministically (output names are static per propagator), so any
        # future colliding field is caught in tests, not in user code. A name
        # in ``_SHADOWABLE`` is exempt, because the class attribute it shadows
        # is a property written to return that very field.
        for name in names:
            if name not in cls._SHADOWABLE and (
                hasattr(cls, name) or name in cls._RESERVED
            ):
                raise AssertionError(
                    f"PropagatorOutput field name {name!r} collides with an "
                    f"existing attribute and would be inaccessible by name."
                )
        obj = super().__new__(cls, values)
        obj._names = names
        obj.info = info
        return obj

    def __getnewargs_ex__(
        self,
    ) -> tuple[tuple[tuple[str, ...], tuple[Any, ...]], dict[str, Any]]:
        """Returns the ``__new__`` args/kwargs so pickling/copying round-trips."""
        # ``tuple`` subclasses are reconstructed via ``__new__``; supply both
        # positional constructor arguments (so pickling/copying round-trips the
        # field names) plus ``info`` as a keyword argument, since ``info`` is
        # keyword-only on ``__new__`` and so cannot be carried by the
        # positional-only ``__getnewargs__``.
        return (self._names, tuple(self)), {"info": self.info}

    @property
    def names(self) -> tuple[str, ...]:
        """The field names, in return order."""
        return self._names

    @property
    def auxiliary_state(self) -> "AuxiliaryState":
        """The opaque absorbing-boundary and auxiliary state of this run.

        Raises ``AttributeError`` on a propagator that has no auxiliary-state
        field. Declared as a property, rather than left to ``__getattr__``,
        because it is the one element of the tuple that is not a
        :class:`torch.Tensor` and so needs a static type of its own.
        """
        try:
            index = self._names.index("auxiliary_state")
        except ValueError:
            raise AttributeError(
                f"{type(self).__name__!r} has no field 'auxiliary_state'; "
                f"available fields: {', '.join(self._names)}"
            ) from None
        return cast("AuxiliaryState", self[index])

    @property
    def final_state(self) -> abc.Mapping[str, _StateValue]:
        """The fields that ``initial_state=`` consumes, as a read-only mapping.

        The public state fields plus ``auxiliary_state``, excluding the receiver
        amplitudes. ``prop(..., initial_state=out)`` and
        ``prop(..., initial_state=out.final_state)`` are equivalent; the
        mapping form is the one to build by hand when only part of the state
        is being replaced. Mutating it raises.
        """
        cached = self.__dict__.get("_final_state")
        if cached is None:
            state_names = self.info.wavefield_names
            if not state_names:
                raise _DeepwaveRuntimeError(
                    "final_state is unavailable on this output: the "
                    "propagator did not record its public state field names "
                    "in RunInfo.wavefield_names."
                )
            values = self.as_dict()
            cached = {n: values[n] for n in state_names}
            # A plain dict, not the proxy: the cache lives in ``__dict__`` and
            # is therefore part of what pickle and copy carry, and a
            # ``mappingproxy`` cannot be pickled.
            self.__dict__["_final_state"] = cached
        return MappingProxyType(cached)

    def _receiver_view(self, prefix: str) -> dict[str, torch.Tensor]:
        """The channel-keyed receiver fields carrying ``prefix``.

        Built by stripping the prefix off :attr:`names`, so the views follow
        the field names automatically and hold no state of their own. The
        longer prefixes are tested first, since ``receiver_amplitudes`` is a
        suffix of both ``bg_receiver_amplitudes`` and
        ``sc_receiver_amplitudes``.
        """
        longer = [p for p in _RECEIVER_PREFIXES if p != prefix and prefix in p]
        view: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self._names):
            if not name.startswith(prefix) or any(name.startswith(p) for p in longer):
                continue
            channel = name[len(prefix) :].lstrip("_")
            view[channel or "u"] = cast("torch.Tensor", self[index])
        return view

    def _cached_receiver_view(
        self, cache_key: str, prefix: str
    ) -> abc.Mapping[str, torch.Tensor]:
        """A cached, read-only :meth:`_receiver_view`, or ``AttributeError``."""
        cached = self.__dict__.get(cache_key)
        if cached is None:
            cached = self._receiver_view(prefix)
            if not cached:
                raise AttributeError(
                    f"{type(self).__name__!r} has no field "
                    f"{cache_key.lstrip('_')!r}; available fields: "
                    f"{', '.join(self._names)}"
                )
            # A plain dict, not the proxy: the cache lives in ``__dict__``,
            # which pickle and copy carry, and a ``mappingproxy`` cannot be
            # pickled.
            self.__dict__[cache_key] = cached
        return MappingProxyType(cached)

    @property
    def receivers(self) -> abc.Mapping[str, torch.Tensor]:
        """The recorded receiver data, keyed by channel, as a read-only mapping.

        The same vocabulary the ``receivers=`` argument uses, so a wrapper can
        round-trip ``receivers={channel: locations}`` into
        ``out.receivers[channel]`` without knowing how this propagator spells
        its output fields. ``"u"`` on the two scalar propagators; ``"p"``,
        ``"y"``, ``"x"`` (and ``"z"`` in 3-D) on the acoustic and elastic
        families; ``"sigma_v"``/``"sigma_h"`` as well on ``acoustic_vti``.

        **Every channel of the propagator is present**, whether or not it was
        requested: a channel with no receiver locations holds the same empty
        tensor the returned tuple holds for it. An unrequested channel raising
        ``KeyError`` would be a worse interface than a documented empty
        tensor, because a caller could not tell it apart from a typo. That
        empty tensor is shaped like the data it would have held --
        ``[n_shots, 0, nt]``, the same shot count and the same number of time
        samples as a requested channel -- so
        ``torch.cat(list(out.receivers.values()), dim=1)`` is a valid
        multichannel gather however few channels were asked for.

        On a Born propagator this is the **scattered** data, matching the
        ``sc_receiver_*`` field names; the background data is
        :attr:`bg_receivers`.

        Mutating the mapping raises.
        """
        # A Born propagator spells its scattered receiver fields ``sc_``;
        # every other propagator has only the bare spelling. ``out.receivers``
        # is the scattered data in both cases, which is the whole point of
        # the ``sc_`` spelling: the name now matches the tuple order.
        prefix = (
            "sc_receiver_amplitudes"
            if any(n.startswith("sc_receiver_amplitudes") for n in self._names)
            else "receiver_amplitudes"
        )
        return self._cached_receiver_view("_receivers", prefix)

    @property
    def bg_receivers(self) -> abc.Mapping[str, torch.Tensor]:
        """The recorded background data of a Born propagator, keyed by channel.

        As :attr:`receivers`, for the background (unperturbed) wavefield's
        receivers. Raises ``AttributeError`` on the eight propagators that
        have no background receivers, rather than returning an empty mapping:
        an empty mapping would read as "the background data was not
        requested", which is a different thing from "this propagator has
        none".
        """
        return self._cached_receiver_view("_bg_receivers", "bg_receiver_amplitudes")

    def detach(self) -> "PropagatorOutput":
        """A copy of this output with every field detached from the graph.

        Truncated backpropagation through a long record -- running many
        segments but backpropagating through only the most recent ones --
        means cutting the autograd graph at a segment join deliberately.
        Every state field has to be cut for the graph to actually be cut:
        detaching only the ``auxiliary_state`` leaves the physical wavefields
        attached, so the join still carries the whole history and the
        gradient is the untruncated one::

            out = scalar(v, dx, dt, source_amplitudes=chunk)
            loss(out.receiver_amplitudes).backward()
            state = out.detach()          # the whole join, cut
            out = scalar(v, dx, dt, source_amplitudes=next_chunk,
                         initial_state=state)

        The ``auxiliary_state`` of the copy is detached through
        :meth:`AuxiliaryState.detach`, so it records that the detachment was
        deliberate and the next call does not warn about a lost gradient.
        ``info`` and the field names are carried over unchanged.
        """
        values = [
            value.detach() if hasattr(value, "detach") else value for value in self
        ]
        return PropagatorOutput(self._names, values, info=self.info)

    def __getattr__(self, name: str) -> torch.Tensor:
        """Returns the tensor field ``name``, or raises ``AttributeError``."""
        # Only invoked when normal attribute lookup fails. Guard the internal
        # attributes, which are not set yet during unpickling / copy, and
        # anything the class itself defines: a property that raises
        # ``AttributeError`` (``auxiliary_state`` on a propagator that has
        # none) counts as a failed lookup, so without this it would land back
        # here and recurse.
        if name in self._RESERVED:
            raise AttributeError(name)
        if hasattr(type(self), name):
            # A property that raised ``AttributeError`` (``auxiliary_state`` on
            # a propagator that has none) lands here; re-raise rather than
            # looking the name up again, which would call the property again.
            raise AttributeError(
                f"{type(self).__name__!r} has no field {name!r}; available "
                f"fields: {', '.join(self._names)}"
            )
        try:
            index = self._names.index(name)
        except ValueError:
            return self._no_such_field(name)
        return cast("torch.Tensor", self[index])

    def _no_such_field(self, name: str) -> torch.Tensor:
        """Raises ``AttributeError`` for ``name``, listing the fields there are.

        An absorbing-boundary or attenuation-memory auxiliary field lands here
        too. ``v0.0.27`` returned a plain tuple, so no release ever answered to
        one of those names as an *attribute*; they are carried opaquely in
        :attr:`auxiliary_state`, and the way to continue a simulation is
        ``initial_state=out``. The message says so, because the positional
        return of the four propagators ``v0.0.27`` released does still hold
        them, so code being migrated by name arrives here.
        """
        state: Optional[AuxiliaryState]
        try:
            state = self.auxiliary_state
        except AttributeError:
            state = None
        available = ", ".join(self._names)
        if state is not None and name in state:
            raise AttributeError(
                f"{type(self).__name__!r} has no field {name!r}: it is an "
                "absorbing-boundary or attenuation-memory auxiliary field, "
                "which is carried opaquely in out.auxiliary_state rather than "
                "as an output attribute. Pass initial_state=out (or "
                "initial_state=out.final_state) to continue a simulation. "
                f"Available fields: {available}"
            )
        extra = (
            ""
            if state is None
            else f" (out.auxiliary_state additionally holds {', '.join(state.names)})"
        )
        raise AttributeError(
            f"{type(self).__name__!r} has no field {name!r}; available "
            f"fields: {available}{extra}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Returns an ordered mapping of field name to value."""
        return dict(zip(self._names, self))

    def _asdict(self) -> dict[str, Any]:
        """Namedtuple-shaped alias of :meth:`as_dict`."""
        return self.as_dict()

    def __repr__(self) -> str:
        """Returns ``ClassName(field=shape, ...)`` for readable printing."""
        fields = ", ".join(
            f"{name}={tuple(value.shape)}"
            if isinstance(value, torch.Tensor)
            else f"{name}={value!r}"
            for name, value in zip(self._names, self)
        )
        return f"{type(self).__name__}({fields})"


PropagatorOutput.__module__ = "deepwave"


#: The pytree context of a :class:`PropagatorOutput`: its field names and the
#: :class:`RunInfo` of the run that produced it.
_PropagatorOutputContext = tuple[tuple[str, ...], RunInfo]


def _flatten_propagator_output(
    output: PropagatorOutput,
) -> tuple[list[Any], _PropagatorOutputContext]:
    """The pytree children of an output: its tuple elements, in return order."""
    return list(output), (output.names, output.info)


def _unflatten_propagator_output(
    children: abc.Iterable[Any], context: _PropagatorOutputContext
) -> PropagatorOutput:
    """Rebuilds an output from its tuple elements and its names/``info``."""
    names, info = context
    return PropagatorOutput(names, tuple(children), info=info)


def _flatten_propagator_output_with_keys(
    output: PropagatorOutput,
) -> tuple[list[tuple[Any, Any]], _PropagatorOutputContext]:
    """:func:`_flatten_propagator_output`, keying each child by its field name."""
    children, context = _flatten_propagator_output(output)
    keys = [_pytree.attribute_key(name) for name in output.names]
    return list(zip(keys, children)), context


# Registering the class as a pytree node is what lets ``torch.func`` and
# ``torch.compile`` see *through* a returned ``PropagatorOutput`` to the
# tensors inside it. Without it the object is a leaf, so ``vmap`` cannot add
# its batch dimension to the fields, ``jacrev(has_aux=True)`` cannot carry it
# as the auxiliary output, and ``torch.utils._pytree.tree_map`` cannot walk it.
#
# The children are the tuple elements in return order, so the flattened order
# is the documented positional return order. The context is the pair
# (field names, ``info``): the names vary per instance (they depend on ndim
# and, for the viscous propagators, on the number of relaxation mechanisms),
# so unlike a ``namedtuple`` they cannot be recovered from the class, and the
# object cannot be rebuilt without them. ``info`` travels with them because it
# is attribute metadata rather than a tuple element, and a transform that
# rebuilds the output must give it back the ``info`` of the run that produced
# it -- a transform does not change how the run was discretised. The two
# per-instance caches (``final_state`` and the receiver views) are not carried:
# they are derived from the children and the names, so the rebuilt object
# regenerates them on demand.
#
# One of the children -- ``auxiliary_state`` -- is not a tensor, so it is
# registered as a pytree node too (in ``deepwave._engine``) and flattens to the
# tensors it holds. The leaves of an output are therefore exactly its tensors.
#
# The registration goes through ``deepwave._pytree`` and uses the keyed
# registration API supported by the declared PyTorch 2.4 minimum.
_pytree.register(
    PropagatorOutput,
    _flatten_propagator_output,
    _unflatten_propagator_output,
    _flatten_propagator_output_with_keys,
)

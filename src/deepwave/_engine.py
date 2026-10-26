"""Shared propagation engine for Deepwave propagators.

This module holds the step-based backend loop, the ``PropagatorSpec``/
``PropagatorOutput``/``BackendState``/``ForwardSavedState`` data structures,
the forward/backward orchestration functions (``run_forward_engine``,
``run_backward_core``, ``run_backward``), the "canonical call" helpers, and
the functorch/second-order guards (``no_second_order``, ``unsupported_vmap``,
``unsupported_jvp``, etc.) shared by every propagator's autograd Function.

It was split out of ``deepwave._common`` (task Q of the propagator-engine
refactor).
``_backend_function`` intentionally stays defined in ``deepwave._common``
(imported here lazily, inside the two functions that call it) rather than
moving, so that ``tests/test_abi_drift.py``'s ``common._backend_function``
monkeypatch -- which relies on the lookup happening through the ``common``
module's live namespace -- keeps working unchanged.
"""

import functools
import importlib
import inspect
import math
import threading
from collections import abc
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import torch
import torch._C._functorch as _functorch

from deepwave import _pytree
from deepwave._exceptions import (
    DEPRECATION_REMOVAL_VERSION as _DEPRECATION_REMOVAL_VERSION,
)
from deepwave._exceptions import (
    DeepwaveFloatingPointError,
    DeepwaveNotImplementedError,
    DeepwaveRuntimeError,
    DeepwaveTypeError,
    DeepwaveValueError,
    warn_at_user_call,
    warn_deprecated_name,
)
from deepwave._plan import plan_reached as plan_reached

# ``_StorageModeEnum`` is the backend's private ``IntEnum`` of storage
# codes; the public ``StorageMode`` imported below is the *string* alias
# that appears in signatures. They are different things with almost the
# same name, so the enum is aliased here to keep the two apart. The enum
# is never re-exported.
from deepwave._storage import (
    StorageManager,
    downgraded_storage_mode,
    storage_mode_enum,
)
from deepwave._storage import StorageMode as _StorageModeEnum

# Re-exported (``as`` form, so it is an explicit re-export) for the nine
# propagators, which all import ``_engine`` already and would otherwise each
# grow an import of their own.
from deepwave._storage import (
    resolved_storage_for_device as resolved_storage_for_device,
)
from deepwave.types import (
    _RECEIVER_PREFIXES,
    PropagatorOutput,
    RunInfo,
    StorageMode,
)

# PyTorch's forward-AD level stack can be invalidated when two host threads
# enter forward AD simultaneously. The two Born model-preparation paths
# share this lock; it is never held while propagating wavefields.
_internal_jvp_lock = threading.Lock()


def model_preparation_jvp(
    function: Callable[..., list[torch.Tensor]],
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    """Differentiate model preparation while preserving saved-tensor hooks.

    Torch 2.4's ``torch.func.jvp`` disables saved-tensor hooks, which prevents
    Born preparation inside non-reentrant checkpointing. Ordinary forward AD
    preserves those hooks and the reverse-mode graph of the resulting tangent.
    An existing forward-AD level needs functorch's nested-transform handling.
    Both routes share the lock because the forward-AD level stack is global.
    """
    with _internal_jvp_lock:
        # PyTorch exposes no public query for an already active dual level.
        if torch.autograd.forward_ad._current_level >= 0:  # noqa: SLF001
            # PyTorch's annotations do not consistently preserve the callable's
            # output tree type across supported releases.
            jvp = cast(
                "Callable[..., tuple[list[torch.Tensor], list[torch.Tensor]]]",
                torch.func.jvp,
            )
            _, result = jvp(function, primals, tangents)
            return list(result)
        # Reentrant checkpoint recomputation runs inside backward, where forward
        # AD is disabled. Enable it locally, as torch.func.jvp does, and restore
        # the caller's setting on exit.
        with (
            torch.autograd.forward_ad._set_fwd_grad_enabled(True),  # noqa: SLF001
            torch.autograd.forward_ad.dual_level(),
        ):
            duals = [
                torch.autograd.forward_ad.make_dual(primal, tangent)  # type: ignore[no-untyped-call]
                for primal, tangent in zip(primals, tangents)
            ]
            result = []
            for output in function(*duals):
                _, tangent = torch.autograd.forward_ad.unpack_dual(output)  # type: ignore[no-untyped-call]
                result.append(torch.zeros_like(output) if tangent is None else tangent)
            return result


if TYPE_CHECKING:
    from deepwave.types import Callback, CallbackState


def create_or_pad(
    tensor: torch.Tensor,
    fd_pad: Union[int, abc.Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
    size: abc.Sequence[int],
) -> torch.Tensor:
    """Creates a zero tensor of a specified size or pads an existing tensor.

    If the input `tensor` is empty (numel == 0), a new zero tensor with the
    given `size` is created. Otherwise, the `tensor` is padded according to
    `fd_pad`.

    Args:
        tensor: The input torch.Tensor to be created or padded.
        fd_pad: The padding to apply. Can be an integer (for uniform padding)
            or a sequence of integers (for per-dimension padding).
        device: The PyTorch device for the tensor.
        dtype: The PyTorch data type for the tensor.
        size: The desired size of the tensor if it needs to be created.

    Returns:
        torch.Tensor: The created or padded torch.Tensor.

    """
    from deepwave._common import reverse_pad

    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * (len(size) - 1) * 2
    if tensor.numel() == 0:
        return torch.zeros(size, device=device, dtype=dtype)
    if max(fd_pad) == 0:
        return tensor.clone()
    reversed_fd_pad = reverse_pad(fd_pad)
    return (torch.nn.functional.pad(tensor, reversed_fd_pad)).requires_grad_(
        tensor.requires_grad
    )


# ---------------------------------------------------------------------------
# Shared backend engine helpers
#
# The autograd Functions of every propagator (scalar, scalar_born, acoustic,
# elastic) share the same boilerplate for interfacing with the compiled
# C/CUDA backend: resolving the storage mode, choosing the threading/stream
# ``aux`` value, and allocating model-gradient (and per-thread temporary)
# buffers. The functions below centralise that boilerplate so the propagator
# modules only need to describe *what* their models and wavefields are, not
# repeat *how* to set these up.
# ---------------------------------------------------------------------------


def resolve_storage_mode(
    storage_mode_str: StorageMode, device: torch.device
) -> "_StorageModeEnum":
    """Convert a storage-mode string into a private ``StorageMode`` enum value.

    On CPU devices the ``"cpu"`` storage mode is equivalent to ``"device"``
    (there is no separate host memory to spill to), so it is transparently
    downgraded.

    Args:
        storage_mode_str: One of ``"device"``, ``"cpu"``, ``"disk"``, or
            ``"none"``.
        device: The device the propagation will run on.

    Returns:
        The corresponding :class:`deepwave._storage.StorageMode` value.

    Raises:
        ValueError: If ``storage_mode_str`` is not a recognised mode.

    """
    return storage_mode_enum(downgraded_storage_mode(storage_mode_str, device))


def resolve_backend_mode(backend: str) -> str:
    """Resolves the coerced ``backend`` value to a step-function mode string.

    Shared by every propagator's ``*_func`` dispatcher. Called only after
    :func:`_coerce_backend` has already resolved ``"compiled"`` to ``None``
    (and folded in any deprecated ``python_backend`` alias), so ``backend``
    here is always one of the two Python-mode strings; this helper
    lowercases and validates it defensively.

    Returns:
        Either ``"compile"`` or ``"eager"``.

    Raises:
        TypeError: If ``backend`` is not a str.
        ValueError: If the string is not a recognised mode.
    """
    if not isinstance(backend, str):
        raise DeepwaveTypeError(f"backend must be str, but got {type(backend)}")
    mode = backend.lower()
    if mode not in ("compile", "eager"):
        raise DeepwaveValueError(f"Unknown backend value {mode!r}.")
    return mode


@dataclass
class StepCache:
    """Lazily compile a step function once using PyTorch's normal cache.

    PyTorch owns specialisation, recompilation and cache limits for the
    returned callable. Deepwave does not change process-wide compiler
    settings or keep a second cache of wrappers by configuration.

    Attributes:
        base: The eager step function.
    """

    base: Callable[..., Any]
    _compile: Optional[Callable[..., Any]] = field(default=None, init=False, repr=False)

    def resolve(
        self,
        backend: Optional[str],
    ) -> Callable[..., Any]:
        """Returns the step function for the requested backend mode.

        The ``torch.compile`` wrapper is created on first use and reused.

        Args:
            backend: The propagator's already-coerced ``backend`` value;
                falsy (``None``/``""``) returns :attr:`base` unchanged.

        Returns:
            :attr:`base`, or the cached compiled variant.
        """
        if not backend:
            return self.base
        mode = resolve_backend_mode(backend)
        if mode == "compile":
            if self._compile is None:
                self._compile = torch.compile(self.base)
            return self._compile
        return self.base


def float_grid_spacing(grid_spacing: abc.Sequence[float]) -> list[float]:
    """Coerces a grid spacing to Python ``float``s for the traced backends.

    Every step function computes its finite-difference scale factor as
    ``1 / grid_spacing[dim]`` (and, in the regular-grid propagators,
    ``1 / grid_spacing[dim] ** 2``) *inside* the region ``torch.compile``
    traces, so the type of ``grid_spacing``'s entries is not merely cosmetic:
    it decides how that reciprocal is evaluated by the generated kernel.

    ``grid_spacing`` is typed ``list[float]`` everywhere, and every public
    propagator entry point already guarantees that via
    :func:`deepwave._common.set_grid_spacing`. But Python's numeric tower
    makes ``int`` a silently-acceptable ``float`` to both the interpreter and
    ``mypy``, so an ``int`` can still reach the traced step functions through
    the private ``*_func`` dispatchers -- and on CUDA that is a **silent
    correctness bug**, not a style problem:

    * dynamo specialises an ``int`` argument on the first compilation, but
      promotes it to a dynamic ``SymInt`` as soon as a second call passes a
      different value (``automatic_dynamic_shapes``, on by default);
    * inductor then emits the reciprocal as a division of the *integer*
      kernel parameter, e.g. ``(tl.broadcast_to(1 / ks2, [XBLOCK]))`` with
      ``ks2`` declared ``i64``;
    * Triton's integer true-division evaluates in **fp32** and the fp32
      result is only then widened to fp64, so an otherwise pure-float64
      propagation silently inherits single precision -- a relative error of
      ``2**-26`` (1.49e-08) rather than ~1e-16.

    This is an upstream ``torch.compile`` defect -- reproduced on torch 2.13.0
    without deepwave by ``torch.compile(lambda x, n: x * (1 / n),
    dynamic=True)`` on a float64 ``x`` and an ``int`` ``n``, which is wrong by
    exactly ``2**-26``; CUDA/Triton only, the CPU C++ backend divides in
    double correctly. It is invisible in float32 runs, which is why it only
    ever surfaced in deepwave's float64 tests.

    A dynamic *float* symbol is passed to the kernel in double precision and
    is unaffected, so coercing here at the compile boundary -- outside the
    traced region, where the coercion is a plain Python operation rather than
    a symbolic one -- removes the exposure without giving up dynamic shapes.

    Args:
        grid_spacing: The grid spacing in each spatial dimension.

    Returns:
        The same spacings as a new list of Python ``float``s.
    """
    return [float(spacing) for spacing in grid_spacing]


#: The internal step-function modes: the vocabulary every helper downstream
#: of :func:`_coerce_backend` has always spoken.
BackendMode = Literal["eager", "compile"]

#: The canonical ``backend`` values, in documentation order. ``"auto"`` and
#: ``"compiled"`` both run the compiled C/CUDA kernel where one exists; they
#: differ only in what happens when one does not (see :class:`BackendRequest`).
BACKEND_VALUES = ("auto", "compiled", "eager", "torch_compile")

#: The string values the deprecated ``python_backend`` argument accepts, and
#: the canonical ``backend`` value each of them names. ``"compile"`` is
#: ``v0.0.27``'s spelling of ``torch_compile`` and is accepted only here,
#: under the deprecated argument that shipped it; ``backend`` itself does not
#: take it. ``"jit"`` -- ``v0.0.27``'s ``torch.jit.script`` mode -- maps to
#: ``"eager"``, the mode that now runs those step functions: Deepwave no
#: longer calls ``torch.jit``, and the eager functions are the same
#: arithmetic, differing only in floating-point rounding.
_PYTHON_BACKEND_VALUES = MappingProxyType(
    {"eager": "eager", "jit": "eager", "compile": "torch_compile"}
)

#: Canonical ``backend`` value -> the internal step-function mode that the
#: rest of the engine has always spoken. ``None`` means "no Python mode: call
#: the compiled C/CUDA symbol".
_BACKEND_MODES: "MappingProxyType[str, Optional[BackendMode]]" = MappingProxyType(
    {
        "auto": None,
        "compiled": None,
        "eager": "eager",
        "torch_compile": "compile",
    }
)

#: The inverse of :data:`_BACKEND_MODES` for the *resolved* modes only, used
#: to report what actually ran (``RunInfo.backend``). ``"auto"`` never appears
#: on the right-hand side: it is a request, not a result.
_RESOLVED_BACKEND_NAMES = MappingProxyType(
    {None: "compiled", "eager": "eager", "compile": "torch_compile"}
)


class BackendRequest(NamedTuple):
    """A caller's ``backend`` argument, canonicalised once.

    Attributes:
        canonical: The request in the canonical vocabulary
            (:data:`BACKEND_VALUES`), with any deprecated
            ``python_backend`` already folded in. This is what a
            ``Module.forward`` hands to its functional: the *request*, not the
            resolution, so that ``"auto"`` stays ``"auto"``.
        mode: The internal step-function mode -- ``None`` (call the compiled
            C/CUDA symbol) or one of ``"eager"``/``"compile"``.
            Exactly the vocabulary every downstream helper has always
            received.
    """

    canonical: str
    mode: Optional[BackendMode]

    @property
    def strict(self) -> bool:
        """Whether the caller demanded the compiled kernel and nothing else.

        ``True`` only for an explicit ``backend="compiled"``. ``"auto"``
        shares :attr:`mode` with it but tolerates a propagator that has no
        compiled kernel at all, falling back to the eager backend.
        """
        return self.canonical == "compiled"


def resolved_backend_name(mode: Optional[str]) -> str:
    """Names a resolved step-function mode in the canonical vocabulary.

    Args:
        mode: A resolved internal mode: ``None``, ``"eager"`` or
            ``"compile"``.

    Returns:
        The matching :data:`BACKEND_VALUES` entry, never ``"auto"``.
    """
    return _RESOLVED_BACKEND_NAMES[mode]


def resolved_model_gradient_stride(
    models: abc.Sequence[torch.Tensor],
    backend_mode: Optional[str],
    step_ratio: int,
    sampling_interval: int,
) -> Optional[int]:
    """Native history stride, eager differentiation stride, or no model gradient."""
    if not torch.is_grad_enabled() or not any(model.requires_grad for model in models):
        return None
    if backend_mode is None:
        return step_ratio * sampling_interval
    return 1


class CallbackSchedule(int):
    """User-step callback frequency with the call's CFL subdivision attached.

    It remains an ``int`` for the eager loops and autograd argument layout;
    the compiled loop additionally reads ``user_step_ratio`` to split native
    calls independently of its retained-history stride.
    """

    user_step_ratio: int

    def __new__(cls, frequency: int, user_step_ratio: int) -> "CallbackSchedule":
        result = super().__new__(cls, frequency)
        result.user_step_ratio = user_step_ratio
        return result


def execution_stride(
    backend_mode: Optional[str], step_ratio: int, sampling_interval: int
) -> int:
    """Native history stride, or one user step for eager differentiation."""
    return step_ratio * sampling_interval if backend_mode is None else step_ratio


def _run_configuration(
    dtype: Optional[torch.dtype],
    ndim: int,
    device: Optional[torch.device],
    extra: abc.Sequence[str] = (),
) -> str:
    """Formats the call configuration named by a compiled-backend gap message."""
    dtype_name = str(dtype).removeprefix("torch.") if dtype is not None else "?"
    device_name = str(device) if device is not None else "?"
    return ", ".join([dtype_name, f"ndim={ndim}", device_name, *extra])


def no_compiled_backend_error(
    propagator: str,
    *,
    dtype: Optional[torch.dtype],
    ndim: int,
    device: Optional[torch.device],
) -> DeepwaveNotImplementedError:
    """Builds the error for ``backend="compiled"`` on an eager-only propagator.

    Args:
        propagator: The propagator's public name.
        dtype: The dtype the call would have run in.
        ndim: The number of spatial dimensions.
        device: The device the call would have run on.

    Returns:
        The exception to raise (never raised here, so the caller's traceback
        points at the propagator).
    """
    return DeepwaveNotImplementedError(
        f"{propagator}: no compiled C/CUDA backend exists for this propagator "
        f'({_run_configuration(dtype, ndim, device)}). Pass backend="auto" '
        'to run the eager PyTorch backend, or backend="eager" to select it '
        "explicitly."
    )


def compiled_backend_gap_error(
    propagator: str,
    gap: str,
    remedy: str,
    *,
    dtype: Optional[torch.dtype],
    ndim: int,
    device: Optional[torch.device],
    extra: abc.Sequence[str] = (),
) -> DeepwaveNotImplementedError:
    """Builds the error for a compiled backend that lacks one requested option.

    Distinct from :func:`no_compiled_backend_error`: here the propagator does
    have a compiled backend, but not for this particular option. This is what
    ``backend="compiled"`` gets, because ``"compiled"`` is an assertion that
    the compiled kernel ran. A caller using :func:`resolve_compiled_option_gap`
    provides an explicit eager fallback for ``"auto"`` instead.
    ``out.info.backend`` is where the caller reads which backend ran.

    Args:
        propagator: The propagator's public name.
        gap: What the compiled backend does not implement, as a clause
            following the propagator name.
        remedy: The one-line fix, as one or more complete sentences.
        dtype: The dtype the call runs in.
        ndim: The number of spatial dimensions.
        device: The device the call runs on.
        extra: Further configuration terms to name, such as ``"mpml_fraction=0.1"``.

    Returns:
        The exception to raise.
    """
    return DeepwaveNotImplementedError(
        f"{propagator}: {gap} "
        f"({_run_configuration(dtype, ndim, device, extra)}). {remedy}"
    )


def resolve_compiled_option_gap(
    request: "BackendRequest",
    propagator: str,
    *,
    gap: str,
    remedy: str,
    dtype: Optional[torch.dtype],
    ndim: int,
    device: Optional[torch.device],
    extra: abc.Sequence[str] = (),
) -> Optional[BackendMode]:
    """Resolves ``backend`` where the compiled kernel lacks a requested option.

    Implements an explicit option fallback chosen by a propagator's front
    end (currently elastic-Born M-PML). This helper is not a general backend
    search: transform support and other native restrictions may still require
    an explicit eager request from the caller.

    It can be much slower, so the choice is reported rather than hidden:
    ``out.info.backend`` names the backend that actually ran, and is
    machine-checkable inside a training loop or a distributed worker, where
    a warning would be invisible. A caller who needs the compiled kernel
    turns the assumption into an assertion by passing
    ``backend="compiled"``, which raises here instead.

    Args:
        request: The canonicalised ``backend`` request.
        propagator: The propagator's public name, for the error text.
        gap: What the compiled backend does not implement, as a clause
            following the propagator name.
        remedy: The one-line fix for the ``"compiled"`` caller.
        dtype: The dtype the call runs in.
        ndim: The number of spatial dimensions.
        device: The device the call runs on.
        extra: Further configuration terms to name, such as ``"mpml_fraction=0.1"``.

    Returns:
        The resolved backend mode: the one explicitly requested if there was
        one, otherwise ``"eager"``.

    Raises:
        DeepwaveNotImplementedError: If ``backend="compiled"`` was requested,
            which this call cannot satisfy.

    """
    if request.mode is not None:
        return request.mode
    if request.strict:
        raise compiled_backend_gap_error(
            propagator,
            gap,
            remedy,
            dtype=dtype,
            ndim=ndim,
            device=device,
            extra=extra,
        )
    return "eager"


def _map_deprecated_backend(python_backend: object) -> str:
    """Maps a deprecated ``python_backend`` value to its ``backend`` equivalent.

    The mapping is faithful to what ``v0.0.27`` did, which is not what the
    name suggests: there, ``python_backend=True`` selected
    ``torch.compile`` when the installed library had been built with OpenMP
    and ``torch.jit.script`` otherwise. It never meant "eager".

    Deepwave no longer calls ``torch.jit``, so the two values that chose it --
    ``"jit"``, and ``True`` without OpenMP -- map to ``"eager"``: the same
    step functions, run by the interpreter instead of TorchScript, so the
    results differ only in floating-point rounding.

    Never receives ``None``: :func:`_coerce_backend` reads ``None`` as "the
    alias was not passed" and never reaches here with it.

    Args:
        python_backend: The raw deprecated-alias value: a ``bool``, or one of
            ``"eager"``, ``"jit"``, ``"compile"``.

    Returns:
        A canonical :data:`BACKEND_VALUES` entry.

    Raises:
        ValueError: If ``python_backend`` is not one of the above.
    """
    if isinstance(python_backend, bool):
        if not python_backend:
            # v0.0.27 ran the compiled C backend; "auto" is that, and is also
            # the only sensible reading on the seven propagators that never
            # shipped there.
            return "auto"
        from deepwave import _backend_utils

        return "torch_compile" if _backend_utils.USE_OPENMP else "eager"
    if isinstance(python_backend, str) and python_backend in _PYTHON_BACKEND_VALUES:
        return _PYTHON_BACKEND_VALUES[python_backend]
    raise DeepwaveValueError(
        "python_backend must be None, a bool, 'eager', 'jit', or 'compile', "
        f"got {python_backend!r}."
    )


def _coerce_backend(
    backend: object,
    python_backend: object = None,
    *,
    stacklevel: int = 3,
) -> BackendRequest:
    """Coerces ``backend``, folding in the deprecated ``python_backend`` alias.

    Called first thing in each of the nine propagator functionals and their
    Module ``forward`` methods -- the single place the public vocabulary is
    canonicalised, so that :func:`resolve_backend_mode` (and everything else
    downstream) only ever sees :attr:`BackendRequest.mode`: ``None`` or one of
    the two internal mode strings.

    ``python_backend`` is still accepted as a deprecated alias for
    ``backend`` on the four propagators shipped in v0.0.27. Passing any
    non-``None`` value
    triggers a :class:`deepwave.DeepwaveFutureWarning` naming ``backend`` as
    the replacement and the release that removes it, and the value is mapped
    as :func:`_map_deprecated_backend` describes. Passing both a non-default
    ``backend`` and ``python_backend`` raises ``ValueError``.

    ``python_backend=None`` means "not given". In ``v0.0.27`` it was outside
    the parameter's declared type and behaviourally identical to omitting it
    (both selected the compiled backend), so reading it as "not given" costs
    only a deprecation warning on a usage that was never declared, and
    lets the alias be typed honestly rather than hidden behind a sentinel.

    Args:
        backend: The raw ``backend`` argument as received by a propagator
            (``object`` rather than a narrower type so mypy does not
            consider the branches below unreachable).
        python_backend: The raw deprecated-alias argument as received by a
            propagator, or ``None`` if it was not given.
        stacklevel: The stack level to fall back on if every frame above this
            one belongs to Deepwave or PyTorch. The level actually used is
            :func:`deepwave._exceptions.stacklevel_outside_deepwave`'s, which
            reaches the user's own line whichever entry point they called
            through -- a propagator function's body is one frame below them,
            while a ``Module.forward`` adds ``torch.nn.Module``'s ``__call__``
            machinery.

    Returns:
        The canonicalised request.

    Raises:
        ValueError: If ``python_backend`` is passed together with a
            non-default ``backend``, if ``python_backend`` is not one of its
            valid values, or if ``backend`` is not one of the documented
            values.
    """
    if python_backend is not None:
        # DeepwaveFutureWarning subclasses FutureWarning, not
        # DeprecationWarning: CPython's default filters un-ignore
        # DeprecationWarning only for __main__, so a deprecation attributed to
        # a library caller's own module would not be shown at all, while a
        # FutureWarning is shown wherever it is attributed.
        warn_deprecated_name("python_backend", "backend", stacklevel=stacklevel)
        if backend != "auto":
            raise DeepwaveValueError(
                "Pass only backend, not both backend and the deprecated python_backend."
            )
        backend = _map_deprecated_backend(python_backend)
    if not isinstance(backend, str) or backend not in BACKEND_VALUES:
        accepted = ", ".join(repr(value) for value in BACKEND_VALUES)
        raise DeepwaveValueError(f"backend must be one of {accepted}, got {backend!r}.")
    return BackendRequest(canonical=backend, mode=_BACKEND_MODES[backend])


def get_aux_and_stream(
    model: torch.Tensor,
    n_shots: int,
    is_cuda: bool,
    spatial_parallel: bool = False,
) -> tuple[int, Union[int, "torch.Stream"]]:
    """Determine the backend ``aux`` value and CUDA stream for a propagation.

    On CUDA, ``aux`` is the device id and the current CUDA stream is returned.
    On CPU, ``aux`` is the number of threads to use (1 when OpenMP is not
    available) and the stream is ``0``.

    Args:
        model: A representative model tensor (used to find the device/stream).
        n_shots: The number of shots being propagated simultaneously.
        is_cuda: Whether the propagation runs on a CUDA device.
        spatial_parallel: Whether the CPU backend parallelises a single shot's
            interior stencil across threads. When ``True`` the thread count is
            not capped at ``n_shots`` (so a single shot can saturate all cores);
            when ``False`` it is capped at ``n_shots`` (the legacy per-shot
            OpenMP scheme, where extra threads would sit idle anyway).

    Returns:
        A tuple ``(aux, stream)``.

    Raises:
        RuntimeError: If ``is_cuda`` and the current CUDA stream on the
            model's device is not that device's default stream. The compiled
            CUDA backend configures its kernels via ``__constant__`` device
            globals written by a synchronous ``cudaMemcpyToSymbol``, which is
            not stream-safe; running on a non-default stream can silently
            produce wrong results.

    """
    if is_cuda:
        aux = model.get_device()
        stream = torch.cuda.current_stream(aux)
        if stream != torch.cuda.default_stream(aux):
            raise DeepwaveRuntimeError(
                "Deepwave's compiled CUDA backend requires the default CUDA "
                "stream (kernel configuration uses device globals that are "
                "not stream-safe). Run outside torch.cuda.stream(), or use "
                'backend="eager".'
            )
        return aux, stream
    from deepwave import _backend_utils

    if _backend_utils.USE_OPENMP:
        n_threads = torch.get_num_threads()
        if not spatial_parallel:
            n_threads = min(n_shots, n_threads)
        return max(1, n_threads), 0
    return 1, 0


def allocate_model_gradients(
    models: abc.Sequence[torch.Tensor],
    model_batched: abc.Sequence[bool],
    n_shots: int,
    aux: int,
    is_cuda: bool,
    storage_mode: "_StorageModeEnum",
    device: torch.device,
    dtype: torch.dtype,
    cpu_tmp_batch: Optional[int] = None,
    spatial_parallel: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Allocate gradient and per-thread temporary buffers for the models.

    For each model that requires a gradient a zero-filled buffer of the model's
    shape is created. When the model gradient is accumulated by more than one
    thread/shot (and the model is not itself batched per-shot), a separate
    temporary buffer is allocated so each worker can accumulate into its own
    slice without a race; the slices are summed by the backend afterwards.

    Args:
        models: The model tensors.
        model_batched: For each model, whether a different model is provided per
            shot (in which case no shared temporary buffer is needed).
        n_shots: The number of shots.
        aux: The threading/device ``aux`` value from :func:`get_aux_and_stream`.
        is_cuda: Whether the propagation runs on a CUDA device.
        storage_mode: The active private ``StorageMode`` enum value (no
            temporary buffer is needed when storage is disabled).
        device: The device to allocate on.
        dtype: The dtype to allocate.
        cpu_tmp_batch: The number of rows in the CPU temporary buffer. Defaults
            to ``aux`` (one slice per thread). Some propagators size this to
            ``n_shots`` instead.
        spatial_parallel: Whether the CPU backend uses intra-shot spatial
            parallelism. When ``True``, no per-thread CPU scratch is allocated:
            the kernel accumulates directly into the shared model gradient
            (each cell is written by exactly one thread, and time steps and
            shots are barrier-separated, so there is no race).

    Returns:
        A tuple ``(grad_models, grad_models_tmp, grad_models_tmp_ptr)`` where
        ``grad_models_tmp_ptr`` is the list of data pointers the backend should
        accumulate into (pointing at the temporary buffer when one was
        allocated, otherwise directly at the gradient buffer).

    """
    grad_models = [torch.empty(0, device=device, dtype=dtype) for _ in models]
    grad_models_tmp = [torch.empty(0, device=device, dtype=dtype) for _ in models]
    grad_models_tmp_ptr = [g.data_ptr() for g in grad_models]
    for i, model in enumerate(models):
        if model.requires_grad:
            grad_models[i].resize_(*model.shape)
            grad_models[i].fill_(0)
            grad_models_tmp_ptr[i] = grad_models[i].data_ptr()

    if storage_mode == _StorageModeEnum.NONE:
        return grad_models, grad_models_tmp, grad_models_tmp_ptr

    from deepwave import _backend_utils

    if is_cuda:
        need_tmp = n_shots > 1
        tmp_batch = n_shots
    elif spatial_parallel:
        # Intra-shot spatial parallelism: threads partition the grid disjointly
        # per time step, so the kernel accumulates directly into the shared
        # gradient without any per-thread scratch (or subsequent reduction).
        need_tmp = False
        tmp_batch = 0
    else:
        need_tmp = aux > 1 and _backend_utils.USE_OPENMP
        tmp_batch = aux if cpu_tmp_batch is None else cpu_tmp_batch

    if need_tmp:
        for i, model in enumerate(models):
            if model.requires_grad and not model_batched[i]:
                # ``model.shape`` rather than ``model_shape``: for every
                # propagator whose unbatched models are plain
                # ``model_shape``-shaped tensors the two are identical, but a
                # model carrying extra leading axes (viscoacoustic's
                # per-mechanism ``c2``/``ar``, shaped ``(L, 1, *model_shape)``)
                # needs a correspondingly bigger per-shot scratch slice.
                grad_models_tmp[i].resize_(tmp_batch, *model.shape)
                grad_models_tmp[i].fill_(0)
                grad_models_tmp_ptr[i] = grad_models_tmp[i].data_ptr()

    return grad_models, grad_models_tmp, grad_models_tmp_ptr


def check_storage_mode_model_gradient(
    spec: "PropagatorSpec",
    models: abc.Sequence[torch.Tensor],
    ndim: int,
    storage_mode: "_StorageModeEnum",
) -> None:
    """Refuses ``storage_mode="none"`` when a model gradient was asked for.

    ``storage_mode="none"`` keeps no forward wavefield history, so the imaging
    correlation that forms the model gradient cannot be computed. The engine
    correctly leaves that term out -- but the propagators' Python front ends
    reparameterise the inputs in terms of the models (scalar, for instance,
    premultiplies the source amplitudes by -v^2 dt^2 gathered at the source
    cells), and that path is differentiable, so a model gradient still came
    back: not zero, but a plausible-looking spike at the source locations that
    is entirely wrong (~1300% off on a small scalar case). An optimisation loop
    would happily descend on it. Refusing the combination in one shared place
    keeps it from drifting per propagator -- acoustic_born and elastic_born
    already raise their own (broader) version of this earlier.

    Deliberately model-specific: source-amplitude and initial-wavefield
    gradients are genuinely correct in ``"none"`` mode (they do not need the
    forward history), so they are still allowed.

    Called both by the engine, on the compiled path, and by the pre-flight
    resolution, which must raise the same error whichever backend the call
    resolved to.

    Args:
        spec: The propagator's spec.
        models: The internal models.
        ndim: The number of spatial dimensions.
        storage_mode: The resolved storage mode.

    Raises:
        RuntimeError: If the mode is ``"none"`` and a model requires a
            gradient.

    """
    if storage_mode != _StorageModeEnum.NONE:
        return
    requiring, name_qualifier = models_requiring_grad_names(spec, models, ndim)
    if not requiring:
        return
    raise DeepwaveRuntimeError(
        f'{spec.name} storage_mode="none" retains no forward '
        "wavefield history, so it cannot compute a model "
        "gradient; it is a forward-only (no-grad) low-memory "
        f"mode. {', '.join(requiring)}{name_qualifier} "
        f"require{'s' if len(requiring) == 1 else ''} grad -- use "
        'storage_mode of "device", "cpu", or "disk" for the '
        "backward pass. (Gradients with respect to the source "
        "amplitudes and the initial wavefields do not need the "
        'forward history and remain available in "none" mode for '
        "this propagator; acoustic_born and elastic_born reject "
        'every gradient in "none" mode, including those.)'
    )


def models_requiring_grad_names(
    spec: "PropagatorSpec", models: abc.Sequence[torch.Tensor], ndim: int
) -> tuple[list[str], str]:
    """Names of the models requiring a gradient, user-facing where possible.

    The engine holds the *internal* models (acoustic's ``K``/``By``/``Bx``
    rather than the user's ``v``/``rho``; viscoacoustic's ``m1``/``c2``/``ar``
    rather than ``v``/``rho``/``q``), so naming them directly in an error
    message quotes identifiers the caller has never seen. Where the propagator
    module provides the pair ``USER_MODEL_NAMES``/``user_model_grad_flags``
    -- introduced for :func:`deepwave.estimate_memory` and reused here
    rather than adding a second mapping -- the internal flags are projected
    back onto the user's argument names.

    That projection is the propagator's own documented left inverse, so where
    two user models collapse into one internal model it over-reports rather
    than guesses: ``acoustic`` builds ``K = rho * v**2``, which requires a
    gradient if either ``v`` or ``rho`` does and cannot be attributed to one
    of them afterwards, so a run with only ``rho.requires_grad`` is reported
    as ``v, rho``. For an error message that is the safe direction (both
    really are implicated in the refused gradient), and reusing the existing
    mapping is preferable to a second, message-only one that could drift.

    Args:
        spec: The propagator's :class:`PropagatorSpec`.
        models: The internal model tensors, in ``spec.model_names(ndim)`` order.
        ndim: The number of spatial dimensions.

    Returns:
        ``(names, qualifier)``. ``qualifier`` is empty when ``names`` are the
        user's own argument names, and a parenthetical warning that they are
        not when the propagator has no such mapping and the internal names had
        to be used instead.
    """
    from deepwave import _common

    try:
        return (
            _common._propagator_models_requiring_grad(  # noqa: SLF001
                spec.name, models, ndim
            ),
            "",
        )
    except AttributeError:
        # No user_model_grad_flags on this propagator module (every module
        # carries USER_MODEL_NAMES, but only the four propagators with
        # grad-count tables -- scalar, scalar_born, acoustic, elastic -- can
        # project the internal flags back onto the user's arguments). Fall
        # back to the internal names, labelled as such so the reader knows
        # not to look for them in the signature.
        return (
            [
                name
                for name, model in zip(spec.model_names(ndim), models)
                if model.requires_grad
            ],
            " (internal model names, derived from the model arguments)",
        )


def backend_failure_message(
    propagator: str,
    direction: str,
    status: int,
    storage_mode_str: str = "device",
    storage_path: Optional[str] = None,
) -> str:
    """Builds the error text for a non-zero compiled-backend status code.

    The compiled backends report every failure as the same status code, so the
    message cannot say what went wrong from the code alone. It can say what is
    *most likely*: with ``storage_mode="disk"`` the overwhelmingly common cause
    is that the history could not be written, and the two facts that settle it
    -- which directory, and how much room it has -- are known here and nowhere
    the reader can easily look.

    The first sentence is deliberately unchanged whatever the storage mode: it
    is the part callers and tests match on.

    Args:
        propagator: The propagator's name (e.g. ``"scalar"``).
        direction: ``"forward"`` or ``"backward"``.
        status: The non-zero status code the compiled backend returned.
        storage_mode_str: The storage mode the call was using.
        storage_path: The directory ``"disk"`` mode was writing to, if known.

    Returns:
        The formatted error message, naming the propagator, direction, and
        status code, and pointing at the troubleshooting page.
    """
    disk_clause = ""
    if storage_mode_str == "disk" and storage_path is not None:
        from deepwave._storage import _free_bytes, format_bytes

        free = _free_bytes(storage_path)
        room = "" if free is None else f" has {format_bytes(free)} free and"
        disk_clause = (
            f' With storage_mode="disk" this is most often a disk write '
            f"failure: storage_path={storage_path!r}{room} this run's "
            "wavefield history is written there. Check that the directory is "
            'writable and has room, or use storage_mode="cpu".'
        )
    return (
        f"Compiled backend failed: the {propagator!r} propagator's compiled "
        f"{direction} pass returned non-zero status code {status}."
        f"{disk_clause} See "
        "docs/troubleshooting.rst (the 'Compiled backend failed' section) "
        "for common causes and how to get more diagnostic information."
    )


def invoke_forward_callback(
    callback: "Callback", state: "CallbackState", *, differentiable: bool
) -> None:
    """Call an observer and reject edits that lack an adjoint rule."""
    state._forward_mutation_allowed = not differentiable  # noqa: SLF001
    if not differentiable:
        callback(state)
        return
    tracked = [
        *state._wavefields.values(),  # noqa: SLF001
        *state._models.values(),  # noqa: SLF001
    ]
    versions = [tensor._version for tensor in tracked]  # noqa: SLF001
    callback(state)
    if any(
        tensor._version != version  # noqa: SLF001 - detect in-place edits
        for tensor, version in zip(tracked, versions)
    ):
        raise DeepwaveRuntimeError(
            "A forward_callback changed a wavefield or model during a "
            "differentiable run. The callback has no adjoint for that edit; "
            "use an observational callback or run under torch.no_grad()."
        )


def run_backend_loop(
    *,
    propagator: str,
    direction: str,
    active: bool,
    n_outer: int,
    step_ratio: int,
    callback: Optional["Callback"],
    callback_frequency: int,
    call: Callable[[int, int], int],
    build_state: Callable[[int], "CallbackState"],
    swap: Optional[Callable[[], None]] = None,
    grad_scratch: Optional[abc.Sequence[torch.Tensor]] = None,
    storage_mode_str: str = "device",
    storage_path: Optional[str] = None,
    forbid_forward_mutation: bool = False,
) -> None:
    """Runs the chunked time-stepping loop shared by every propagator.

    Each pass advances in native chunks bounded by the requested user-step
    observations. The native history stride remains ``step_ratio`` internal
    steps and is independent of these chunk lengths. Forward callbacks run
    before their chunk; backward callbacks run after reaching the same state
    boundary in reverse. An odd internal chunk length swaps ping-pong buffers.

    The propagator supplies the chunk-varying behaviour as callables that close
    over its own tensors:

    Args:
        propagator: The propagator's name (e.g. ``"scalar"``), used only to
            name it in the error message if the backend reports failure.
        direction: ``"forward"`` or ``"backward"``.
        active: Whether there is anything to propagate (skip if not).
        n_outer: The number of native history-stride intervals.
        step_ratio: The native history stride in internal time steps.
        callback: The user callback, or ``None``.
        callback_frequency: User steps between callbacks, carrying the CFL
            ratio in a :class:`CallbackSchedule`. Ignored when there is no callback.
        call: ``call(internal_boundary, internal_length)`` invokes the backend
            for one chunk and returns status 0 on success. For a backward call,
            ``internal_boundary`` is the chunk's high-time end.
        swap: Called (with no arguments) to ping-pong the propagator's buffers
            after a chunk with an odd number of internal steps. ``None`` for
            propagators whose forward pass does not swap.
        grad_scratch: The per-shot/per-thread model-gradient scratch buffers
            used by this (backward) pass, if any. The compiled backends reduce
            the scratch into the final gradient at the end of *every* backend
            call, so when the pass is split into more than one chunk the
            scratch must be re-zeroed between chunks; otherwise chunk ``k``'s
            partial sum is added in again by every later chunk and the model
            gradient comes back inflated by a chunk-count-dependent factor.
        storage_mode_str: The call's storage mode, used only to enrich the
            failure message: a compiled backend that fails under
            ``"disk"`` has most often failed to write.
        storage_path: The directory ``"disk"`` mode is writing to, named in
            that message along with its free space.
            Ignored on the forward pass and when there is only one chunk.
        build_state: ``build_state(step)`` builds the :class:`CallbackState`
            passed to ``callback``. Required if ``callback`` is not ``None``.
        forbid_forward_mutation: Reject edits to callback wavefields/models
            when their forward pass participates in autograd.

    Raises:
        RuntimeError: If the backend reports a non-zero status code.
        ValueError: If ``direction`` is not ``"forward"`` or ``"backward"``.

    """
    if not active:
        return
    if direction not in ("forward", "backward"):
        raise DeepwaveValueError(
            f"direction must be 'forward' or 'backward', got {direction}"
        )
    internal_nt = n_outer * step_ratio
    user_step_ratio = getattr(callback_frequency, "user_step_ratio", step_ratio)
    if internal_nt % user_step_ratio:
        raise AssertionError("callback user steps must align with internal time")
    user_nt = internal_nt // user_step_ratio
    observed = list(range(0, user_nt, int(callback_frequency))) if callback else [0]
    boundaries = [step * user_step_ratio for step in observed] + [internal_nt]
    chunks = len(observed)

    # The scratch buffers arrive zeroed, so only chunks after the first need
    # clearing; skipping the first keeps the single-chunk (no-callback) path,
    # which is by far the common one, entirely free of extra work.
    zero_scratch = (
        [t for t in (grad_scratch or ()) if t.numel() > 0]
        if direction == "backward" and chunks > 1
        else []
    )

    sequence = range(chunks) if direction == "forward" else range(chunks - 1, -1, -1)
    for chunk, index in enumerate(sequence):
        user_step = observed[index]
        start = boundaries[index]
        end = boundaries[index + 1]
        if zero_scratch and chunk > 0:
            for scratch in zero_scratch:
                scratch.zero_()
        if direction == "forward" and callback is not None:
            invoke_forward_callback(
                callback, build_state(user_step), differentiable=forbid_forward_mutation
            )
        length = end - start
        status = call(start if direction == "forward" else end, length)
        if status != 0:
            raise DeepwaveRuntimeError(
                backend_failure_message(
                    propagator, direction, status, storage_mode_str, storage_path
                )
            )
        if swap is not None and length % 2 != 0:
            swap()
        if direction == "backward" and callback is not None:
            callback(build_state(user_step))


#: The stability tiers of :doc:`/stability`, as a type. ``PropagatorSpec``
#: only ever uses ``"stable"`` or ``"experimental"`` today (no propagator is
#: provisional or deprecated), but the alias names the whole documented
#: vocabulary so that a later graduation or deprecation is a data change here
#: rather than a change of type.
Stability = Literal["stable", "provisional", "experimental", "deprecated"]


def _no_auxiliary_state_names(ndim: int) -> list[str]:
    """The default ``PropagatorSpec.auxiliary_state_names``: fold nothing."""
    del ndim
    return []


def _no_channels(ndim: int) -> list[str]:
    """The default ``PropagatorSpec.bg_receiver_channels``: no such channels."""
    del ndim
    return []


def scalar_channels(ndim: int) -> list[str]:
    """The single generic-wavefield channel of the two scalar propagators.

    Their field is a generic scalar wavefield rather than a pressure, so the
    channel is named ``"u"``, even though their flat keywords carry no suffix
    at all.
    """
    del ndim
    return ["u"]


def velocity_channels(ndim: int) -> list[str]:
    """One velocity channel per spatial dimension, slowest axis first.

    The slowest-to-fastest axis order the whole library uses. In 1-D that is
    ``x`` alone: the fastest axis is the only axis, and a ``y`` channel there
    would name a component the propagator has no wavefield for.
    """
    if ndim == 3:
        return ["z", "y", "x"]
    if ndim == 1:
        return ["x"]
    return ["y", "x"]


def pressure_velocity_channels(ndim: int) -> list[str]:
    """The pressure/velocity *receiver* channel set shared by seven of the nine.

    ``p`` (the pressure the propagator's family records) followed by the
    velocity channels. ``p`` names one quantity on the receiver side on every
    propagator that has it -- a pressure -- which is why it stays the receiver
    channel's name while the source side spells out which rate it injects (see
    :func:`volume_rate_velocity_channels`).
    """
    return ["p", *velocity_channels(ndim)]


def volume_rate_velocity_channels(ndim: int) -> list[str]:
    """The *source* channel set of the acoustic family.

    ``volume_rate`` (units of 1/time) followed by the velocity channels. The
    scalar source channel is named for the quantity it injects rather than
    ``p``, because ``p`` as a source name means a volume rate here and a
    pressure rate on the elastic family (see
    :func:`pressure_rate_velocity_channels`) -- two quantities that differ by
    roughly the bulk modulus, about 10^9 in SI.
    """
    return ["volume_rate", *velocity_channels(ndim)]


def pressure_rate_velocity_channels(ndim: int) -> list[str]:
    """The *source* channel set of pressure-rate propagators.

    As :func:`volume_rate_velocity_channels`, with ``pressure_rate`` (units of
    pressure/time) as the scalar source channel. Used by the elastic family,
    ``acoustic_vti`` and viscoacoustic.
    """
    return ["pressure_rate", *velocity_channels(ndim)]


#: ``p`` as a *source* channel name on the acoustic family, and the name that
#: replaces it. It is accepted until :data:`DEPRECATION_REMOVAL_VERSION` and
#: warns :class:`deepwave.DeepwaveFutureWarning`. Kept as a shared constant so
#: that the two families' replacements cannot be swapped by accident.
VOLUME_RATE_SOURCE_DEPRECATIONS: abc.Mapping[str, str] = MappingProxyType(
    {"p": "volume_rate"}
)

#: As :data:`VOLUME_RATE_SOURCE_DEPRECATIONS`, for the elastic family and
#: ``acoustic_vti``, whose scalar source is a pressure rate (units of
#: pressure/time) rather than a volume rate (units of 1/time).
PRESSURE_RATE_SOURCE_DEPRECATIONS: abc.Mapping[str, str] = MappingProxyType(
    {"p": "pressure_rate"}
)

#: The empty channel-alias mapping, shared by every propagator that declares
#: no aliases. A ``MappingProxyType`` so a spec's aliases cannot be mutated
#: through the (frozen) dataclass.
_EMPTY_ALIASES: abc.Mapping[str, str] = MappingProxyType({})

#: Every option that :func:`reject_meaningless_options` polices, with the
#: default value that is accepted silently on every propagator. A propagator
#: lists the ones it actually implements in
#: ``PropagatorSpec.supported_options``; a non-default value of any other
#: entry raises. The values here are the single source of truth for "the
#: default" and are pinned against the real signatures by
#: ``tests/test_signature_consistency.py``'s shared-default check.
SHARED_OPTION_DEFAULTS: abc.Mapping[str, Any] = MappingProxyType(
    {
        "mpml_fraction": 0.0,
        "surface_method": "ivm",
        "q_freq": None,
        "q_n_mechanisms": 1,
        "q_band": None,
    }
)

#: The kinds of channel-keyed argument group a propagator can have, and the
#: keyword-name stem each one uses. ``source`` has two stems (amplitudes and
#: locations); the receiver kinds have one each. See :func:`channel_kwargs`.
_CHANNEL_KIND_STEMS: abc.Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "source": ("source_amplitudes", "source_locations", "source_mask"),
        "receiver": ("receiver_locations", "receiver_mask"),
        "bg_receiver": ("bg_receiver_locations", "bg_receiver_mask"),
        "sc_receiver": ("sc_receiver_locations", "sc_receiver_mask"),
    }
)


@dataclass(frozen=True)
class PropagatorSpec:
    """Declarative description of what is physically unique to a propagator.

    Everything the shared engine needs to know that differs between the
    scalar, scalar_born, acoustic, and elastic propagators is captured here as
    data, rather than being hard-coded in each propagator's autograd Function.
    The ndim-dependent quantities (model and wavefield names) are provided as
    small functions of the number of spatial dimensions, and the counts used to
    unpack the flat argument list passed to the autograd Function are derived
    from them.

    Attributes:
        name: The backend function family name (e.g. ``"scalar"``), used to
            look up the compiled C/CUDA function.
        ndims: The numbers of spatial dimensions this propagator supports.
            ``(1, 2, 3)`` for most; ``(2, 3)`` for the anisotropic
            propagators. Declarative, so that
            :func:`deepwave.spec` can report it without making a call that
            raises.
        status: This propagator's documented maturity label (see
            :doc:`/stability`). The single source of truth for it, and a
            label only: nothing emits a runtime warning about it, and the
            interface promise is identical on all nine propagators.
            ``deepwave/_capabilities.py`` reports this field in the
            "Status" column of the generated capability matrix, and
            :func:`deepwave.spec` reports it without making a call.
        staggered: Whether the propagator uses a staggered grid (acoustic,
            elastic) rather than a regular grid (scalar, scalar_born).
        model_names: Returns the ordered names of the models for a given ndim.
            These name the model tensors in callbacks and gradient dicts.
        wavefield_names: Returns the ordered names of the wavefields for a given
            ndim, matching the order the backend stores them in.
        n_source_types: Returns the number of distinct source types (and source
            location tensors) for a given ndim.
        n_receiver_types: Returns the number of distinct receiver types (and
            receiver location tensors) for a given ndim.
        output_names: Returns the ordered names of the propagator's public
            outputs (the public final wavefields, with the auxiliary-state block
            collapsed to the single name ``"auxiliary_state"``, followed by the
            receiver amplitudes) for a given ndim, in the same order as the
            positional return tuple. Used to build the named return (see
            :class:`PropagatorOutput`). Derived from ``wavefield_names``,
            ``public_wavefield_names`` and ``auxiliary_state_names`` by
            :func:`collapse_auxiliary_state_names`, never hand-written, so that
            the names and the values cannot disagree.
        auxiliary_state_names: Returns, for a given ndim, the subset of
            ``public_wavefield_names`` that is carried opaquely in an
            :class:`AuxiliaryState` rather than as a public output field: the
            absorbing boundary's memory variables and the attenuation memory
            variables, whose identity, count and shapes are properties of
            those implementations rather than of the physics asked for. The
            names are the *public* spellings, since the collapse is computed
            in the public order and spelling. The default returns an empty
            list, i.e. no fold at all, which leaves ``output_names`` equal to
            the public wavefield names.
        public_wavefield_names: Returns the propagator's state fields in
            *public* return order and public spelling for a given ndim, before
            the auxiliary-state collapse. ``None`` (the default) means the
            public order is the backend order and the public spellings are the
            internal ones, i.e. ``wavefield_names``. The order differs for the
            elastic family, whose public order groups the fields by kind while
            the backend interleaves them by dimension; the collapse must be
            computed in the public order, since that is where "the index of
            the first auxiliary field" is meaningful. The spelling differs for
            the Born propagators, whose scattered fields carry ``_sc``
            before the time index in both spellings (``vy_sc_0``).
        state_name_renames: Returns, for a given ndim, the ``{internal name:
            public name}`` entries of the state fields this propagator spells
            differently in its public interface. ``None`` (the default) means
            every public spelling is the internal one. A folded name can be
            renamed too: it has no public field of its own, but the
            :class:`AuxiliaryState` bundle is keyed by the public spelling, so
            the two can differ (the attenuating propagators' GSLS memory is
            ``r_0`` internally and ``r_mech0`` in the bundle). Anything that
            translates a whole flat state must therefore go through
            :meth:`internal_state_name_map` (or its inverse) rather than
            assuming the folded names map to themselves. The map is what lets
            the internal consumers of the flat state (:func:`flat_state` and
            :mod:`deepwave.checkpointing`) translate between the two
            spellings; it must agree with ``public_wavefield_names``, which
            :mod:`tests.test_auxiliary_state` pins.
        c_model_flag_indices: The model indices whose requires-grad and batched
            flags are passed to the C backend (the "independent" models that own
            a gradient). Used by the canonical backend call builders so that the
            staggered propagators need no explicit ``make_call``. ``None`` for
            the regular-grid propagators, which build their calls explicitly
            because their PML auxiliaries are interleaved into the argument list.
            May be a callable of ndim for propagators whose set of independent
            models depends on the number of dimensions (e.g. elastic_vti, where
            ``c12`` exists only in 3-D).
        spatial_parallel: Whether the compiled CPU backend parallelises a single
            shot's interior stencil across threads (intra-shot spatial OpenMP)
            rather than distributing whole shots one-per-thread. When ``True``,
            the CPU thread count is not capped at ``n_shots`` and the per-thread
            model-gradient scratch is not allocated (the kernel accumulates
            directly into the shared gradient, since each cell is written by
            exactly one thread and time steps/shots are barrier-separated).
        python_backend_storage_modes: The ``storage_mode`` values the pure-Python
            (``eager``/``torch_compile``) backends accept; any other value is
            rejected by :func:`reject_unsupported_python_backend_options`. Every
            propagator's pure-Python backend runs its own native-autograd time
            loop with no ``StorageManager``, so only the ``"device"`` default is
            supported everywhere.
        python_backend_storage_compression: Whether the pure-Python backends
            accept ``storage_compression=True``. ``False`` (the default, and
            currently every propagator's actual behaviour) means compression is
            rejected.
        python_backend_backward_callback: Whether a ``backward_callback`` works
            at all on the pure-Python backends. ``False`` (the default, and
            currently every propagator except ``scalar``) means one is rejected
            unconditionally; ``scalar`` is the only propagator whose pure-Python
            backends support it (via an autograd hook on the primary wavefield).
        python_backend_backward_callback_in_vmap: Whether a ``backward_callback``
            that the pure-Python backends otherwise support (see
            ``python_backend_backward_callback``) also works inside
            ``torch.func.vmap``. ``False`` (the default) means it is rejected
            when batched; currently no propagator supports this combination
            (``scalar`` accepts a ``backward_callback`` on the Python backends
            but rejects one inside ``vmap``).
        has_compiled_backend: Whether this propagator has a compiled C/CUDA
            backend. True for every released propagator.
        source_channels: Returns, for a given ndim, the propagator's source
            channel names in public (output) order -- ``["u"]`` for the two
            scalar propagators, ``["p", "z", "y", "x"]`` for the 3-D acoustic
            and elastic families, and so on. A *channel* is the physical
            quantity a source injects or a receiver records; it is the
            vocabulary of the ``sources=``/``receivers=`` mappings and of
            :func:`deepwave.spec`, and it is deliberately independent of the
            flat keyword spelling, which :func:`channel_kwargs` derives from
            it. Includes any synthetic channel (see
            ``synthetic_source_channels``).
        receiver_channels: As ``source_channels``, for the receiver side. It
            must equal the channel order of the propagator's receiver
            *outputs*, so that ``deepwave.receiver_names()`` and the returned
            tuple agree.
        bg_receiver_channels: As ``receiver_channels``, for a Born
            propagator's background receivers. The default returns an empty
            list, which is how a non-Born propagator declares that it has no
            ``bg_receivers=`` parameter at all.
        channel_suffixes: Whether this propagator's flat channel keywords
            carry a ``_<channel>`` suffix. ``False`` on ``scalar`` and
            ``scalar_born`` only, whose single channel is spelled with no
            suffix at all (``source_amplitudes=``, not
            ``source_amplitudes_u=``).
        channel_aliases: Maps an accepted alternative channel spelling to the
            canonical channel name, canonicalised before any lookup, on both
            the source and the receiver side. Empty on every propagator: each
            channel has exactly one spelling. Any entry here would be
            *permanent*: it is additive vocabulary, not a deprecation. A
            spelling that is going away belongs in
            ``deprecated_source_channels`` instead.
        deprecated_source_channels: Maps a deprecated *source* channel
            spelling to the name that replaces it:
            ``{"p": "volume_rate"}`` on the acoustic family and
            ``{"p": "pressure_rate"}`` on the elastic family and
            ``acoustic_vti``. The old spelling keeps working, in both the
            flat and the ``sources=`` form, until
            :data:`DEPRECATION_REMOVAL_VERSION`, warning
            :class:`deepwave.DeepwaveFutureWarning` once per call per name.
            It applies to the source side only: ``p`` names a pressure, and
            stays the canonical *receiver* channel name, on every propagator
            that records one.
        synthetic_source_channels: Source channels that the propagator's
            public interface offers but that the backend does not have a
            slot for, because the wrapper expands them into other channels.
            ``{"pressure_rate"}`` on ``acoustic_vti``, whose
            ``source_amplitudes_pressure_rate`` drives both stress channels.
            A synthetic channel is counted in ``source_channels`` but not in
            ``n_source_types``.
        synthetic_receiver_channels: As ``synthetic_source_channels``, for
            the receiver side: a receiver channel the propagator computes
            from other channels' recordings rather than from a backend slot
            of its own. ``{"p"}`` on ``acoustic_vti``, whose pressure is
            ``-(sigma_v + (ndim - 1) sigma_h) / ndim``. A synthetic receiver
            channel still occupies its own output tuple slot, so it is
            counted in ``receiver_channels`` but not in
            ``n_receiver_types``.
        supported_options: The shared options (see
            :data:`SHARED_OPTION_DEFAULTS`) this propagator actually
            implements. Every propagator *accepts* every shared option at its
            default value, so that a generic wrapper can forward one uniform
            keyword set; a non-default value of an option that is not in this
            set raises (see :func:`reject_meaningless_options`). This one
            field drives the validator, the error messages and
            ``deepwave/_capabilities.py``' table, so they cannot drift apart.
        has_forward_wavefield: Whether this propagator can retain the forward
            wavefield through the backward pass and hand it to a
            ``backward_callback`` as
            :meth:`deepwave.CallbackState.get_wavefield`'s ``role="forward"``.
            ``True`` on ``scalar`` alone today. All nine *accept*
            ``backward_callback_needs_forward_wavefield`` so that a generic
            wrapper can forward one uniform keyword set, but asking for it
            where this is ``False`` raises
            :exc:`deepwave.DeepwaveNotImplementedError` (see
            :func:`reject_unsupported_forward_wavefield`): the capability is
            intended to work everywhere and simply is not implemented yet,
            which is what distinguishes it from a *meaningless* shared
            option. This one field drives the validator, the error message
            and ``deepwave/_capabilities.py``' table.
        pml_slab_axes: Maps each PML memory-variable wavefield name to the
            spatial axis (0 = slowest dimension) along which that field is
            architecturally zero outside the two per-axis PML slabs, as a
            function of ndim. The invariant has two sources, and a field may
            only be declared here when it has both: (a) the compiled kernels
            write the field only inside that one axis's PML index range
            (``pml_b``/``pml_e``, see :func:`_pml_region`), and (b) this
            propagator's ``zero_wavefields`` unconditionally zeroes that
            axis's interior of the field on every forward entry and on every
            backward gradient-wavefield seed, so an interior-nonzero value
            cannot survive a call. Only declare fields with both properties in
            every configuration reachable through the compiled backend, except
            for runtime variants the consumer gates on
            (``deepwave.checkpointing._build_slab_plan`` rejects ``mpml_fraction >
            0``, a non-default ``boundary``, and the non-compiled backends).
            Physical fields (wavefields, pressure, velocities, stresses) are
            simply absent from the mapping; so are memory variables that are
            not PML-localised -- the viscoacoustic GSLS memory variables
            ``r_0``..``r_{L-1}`` are the canonical example, being nonzero
            throughout the interior. ``None`` (the default) disables slab
            storage of this propagator's auxiliary states entirely.

    """

    name: str
    staggered: bool
    model_names: Callable[[int], list[str]]
    wavefield_names: Callable[[int], list[str]]
    n_source_types: Callable[[int], int]
    n_receiver_types: Callable[[int], int]
    output_names: Callable[[int], list[str]]
    # Fields with defaults start here (a dataclass cannot put a defaulted
    # field before an undefaulted one), which is why ``status`` is not
    # declared next to ``name`` the way it is documented above.
    status: Stability = "stable"
    ndims: tuple[int, ...] = (1, 2, 3)
    source_channels: Callable[[int], list[str]] = _no_channels
    receiver_channels: Callable[[int], list[str]] = _no_channels
    bg_receiver_channels: Callable[[int], list[str]] = _no_channels
    channel_suffixes: bool = True
    # Python 3.11's dataclasses rejects mappingproxy instances as defaults.
    # The factory retains the shared immutable value without a mutable default.
    channel_aliases: abc.Mapping[str, str] = field(
        default_factory=lambda: _EMPTY_ALIASES
    )
    deprecated_source_channels: abc.Mapping[str, str] = field(
        default_factory=lambda: _EMPTY_ALIASES
    )
    synthetic_source_channels: frozenset[str] = frozenset()
    synthetic_receiver_channels: frozenset[str] = frozenset()
    supported_options: frozenset[str] = frozenset()
    auxiliary_state_names: Callable[[int], list[str]] = _no_auxiliary_state_names
    public_wavefield_names: Optional[Callable[[int], list[str]]] = None
    state_name_renames: Optional[Callable[[int], dict[str, str]]] = None
    c_model_flag_indices: Optional[
        Union[abc.Sequence[int], Callable[[int], abc.Sequence[int]]]
    ] = None
    spatial_parallel: bool = False
    python_backend_storage_modes: tuple[str, ...] = ("device",)
    python_backend_storage_compression: bool = False
    python_backend_backward_callback: bool = False
    python_backend_backward_callback_in_vmap: bool = False
    has_compiled_backend: bool = True
    has_forward_wavefield: bool = False
    pml_slab_axes: Optional[Callable[[int], dict[str, int]]] = None

    def n_models(self, ndim: int) -> int:
        """Number of model tensors for the given ndim."""
        return len(self.model_names(ndim))

    def n_wavefields(self, ndim: int) -> int:
        """Number of wavefield tensors for the given ndim."""
        return len(self.wavefield_names(ndim))

    def has_bg_receivers(self) -> bool:
        """Whether this propagator has a background-receiver channel group."""
        return bool(self.bg_receiver_channels(2))

    def all_channels(self, kind: str) -> list[str]:
        """Every channel of ``kind``, over both numbers of dimensions.

        The union of the 2-D and 3-D channel lists, in 3-D order, so that a
        key can be validated before ``ndim`` is known (it is resolved from
        the model, which is checked later). A key that is valid only in 3-D
        -- ``"z"`` -- therefore passes this check in a 2-D call and is
        rejected further downstream, by the propagator's existing "locations
        related to the z dimension should not be provided" check, which
        already words that case well.
        """
        channels = self._channels(kind, 3)
        extra = [c for c in self._channels(kind, 2) if c not in channels]
        return list(channels) + extra

    def _channels(self, kind: str, ndim: int) -> list[str]:
        """The channel list of ``kind`` for ``ndim``."""
        if kind == "source":
            return list(self.source_channels(ndim))
        if kind in ("receiver", "sc_receiver"):
            return list(self.receiver_channels(ndim))
        if kind == "bg_receiver":
            return list(self.bg_receiver_channels(ndim))
        raise AssertionError(f"unknown channel kind {kind!r}")

    def channel_kwargs(self, kind: str, channel: str) -> tuple[str, ...]:
        """The flat keyword names of one channel of one kind, in group order.

        See the module-level :func:`channel_kwargs`, which this implements;
        the method form lets :mod:`deepwave._common` use it without importing
        this module (they cannot import each other at module level).
        """
        stems = _CHANNEL_KIND_STEMS.get(kind)
        if stems is None:
            raise AssertionError(f"unknown channel kind {kind!r}")
        if not self.channel_suffixes:
            return stems
        return tuple(f"{stem}_{channel}" for stem in stems)

    def canonical_channel(self, channel: str, kind: str = "receiver") -> str:
        """Resolves a spelling to the canonical channel name it names.

        Permanent aliases resolve on both sides; a deprecated spelling
        resolves on the source side only, because that is the only side it
        was ever ambiguous on. Resolution is silent: the caller that saw the
        user's own spelling is the one that warns (see
        :meth:`deprecated_channel_replacement`), since only it knows whether
        the spelling came from the user or from Deepwave's own internals.
        """
        channel = self.channel_aliases.get(channel, channel)
        if kind == "source":
            channel = self.deprecated_source_channels.get(channel, channel)
        return channel

    def deprecated_channel_replacement(self, kind: str, channel: str) -> Optional[str]:
        """The replacement for a deprecated spelling of ``kind``, or ``None``.

        ``None`` means ``channel`` is not a deprecated spelling on this side,
        which includes every spelling on the receiver side.
        """
        if kind != "source":
            return None
        return self.deprecated_source_channels.get(
            self.channel_aliases.get(channel, channel)
        )

    def channel_alias_groups(self) -> dict[str, list[str]]:
        """Maps each canonical channel to the aliases that also name it."""
        groups: dict[str, list[str]] = {}
        for alias, canonical in self.channel_aliases.items():
            groups.setdefault(canonical, []).append(alias)
        return groups

    def describe_channels(self, kind: str) -> str:
        """A human-readable list of the valid keys of ``kind``, for messages.

        Names every channel with its aliases, marks the 3-D-only ones, and
        names any deprecated spelling with what replaced it, so that an
        "unknown channel" error can be acted on without opening the
        documentation.
        """
        two_d = set(self._channels(kind, 2))
        groups = self.channel_alias_groups()
        deprecated: dict[str, list[str]] = {}
        if kind == "source":
            for old, new in self.deprecated_source_channels.items():
                deprecated.setdefault(new, []).append(old)
        parts = []
        for channel in self.all_channels(kind):
            part = repr(channel)
            aliases = groups.get(channel, [])
            if aliases:
                spelled = ", ".join(repr(a) for a in sorted(aliases))
                part += f" (alias {spelled})" if len(aliases) == 1 else f" ({spelled})"
            old_names = deprecated.get(channel, [])
            if old_names:
                spelled = ", ".join(repr(a) for a in sorted(old_names))
                part += f" (replaces the deprecated {spelled})"
            if channel not in two_d:
                part += " (3-D only)"
            parts.append(part)
        return ", ".join(parts)

    @staticmethod
    def _names_for_mechanisms(
        names: Callable[..., Any], ndim: int, n_mechanisms: int
    ) -> Any:
        """Resolve a runtime-sized state list without changing fixed families."""
        if "n_mechanisms" in inspect.signature(names).parameters:
            return names(ndim, n_mechanisms=n_mechanisms)
        return names(ndim)

    def public_state_names(self, ndim: int, n_mechanisms: int = 1) -> list[str]:
        """The state field names in public return order, before the collapse."""
        if self.public_wavefield_names is None:
            return list(
                self._names_for_mechanisms(self.wavefield_names, ndim, n_mechanisms)
            )
        return list(
            self._names_for_mechanisms(self.public_wavefield_names, ndim, n_mechanisms)
        )

    def public_state_name_map(self, ndim: int, n_mechanisms: int = 1) -> dict[str, str]:
        """Maps each internal flat wavefield name to its public field name.

        The two spellings differ where the public interface renames a state
        field: the attenuating propagators' GSLS memory, spelled ``r_mech0``
        publicly and ``r_0`` internally. Every other name maps to itself --
        the Born propagators' scattered fields are spelled ``vy_sc_0`` on
        both sides, there being one Born spelling.

        A folded name is not exempt. It has no public *field* of its own, but
        the :class:`AuxiliaryState` bundle is keyed by the public spelling, so
        a folded name can be renamed like any other -- ``r_0``/``r_mech0``
        is. Code holding a flat (internally keyed) state must therefore apply
        this map, or :meth:`internal_state_name_map` the other way, rather
        than assuming the auxiliary names need no translation.
        """
        renames = (
            {}
            if self.state_name_renames is None
            else self._names_for_mechanisms(self.state_name_renames, ndim, n_mechanisms)
        )
        return {
            name: renames.get(name, name)
            for name in self._names_for_mechanisms(
                self.wavefield_names, ndim, n_mechanisms
            )
        }

    def internal_state_name_map(
        self, ndim: int, n_mechanisms: int = 1
    ) -> dict[str, str]:
        """Maps each public state field name back to its internal flat name.

        The inverse of :meth:`public_state_name_map`, for the internal
        consumers that hold public field names (a
        :class:`PropagatorOutput`'s) and need the backend's own.
        """
        return {
            public: internal
            for internal, public in self.public_state_name_map(
                ndim, n_mechanisms
            ).items()
        }

    def split_backend_args(
        self,
        ndim: int,
        args: abc.Sequence[torch.Tensor],
        n_wavefields_override: Optional[int] = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Splits the flat argument tuple into its logical groups.

        The propagators pass their tensors to the autograd Function as a single
        flat tuple, ordered ``models, source_amplitudes, source_locations,
        receiver_locations, wavefields``. This uses the spec's counts to undo
        that flattening.

        Args:
            ndim: The number of spatial dimensions.
            args: The flat tuple of tensors.
            n_wavefields_override: Overrides ``self.n_wavefields(ndim)``, for a
                propagator whose wavefield count depends on a runtime value the
                fixed ``wavefield_names`` callable cannot see (viscoacoustic's
                ``q_n_mechanisms``). ``None`` (the default) uses the spec's
                count unchanged.

        Returns:
            A tuple ``(models, source_amplitudes, sources_i, receivers_i,
            wavefields)``.

        """
        n_models = self.n_models(ndim)
        n_sources = self.n_source_types(ndim)
        n_receivers = self.n_receiver_types(ndim)
        n_wavefields = (
            n_wavefields_override
            if n_wavefields_override is not None
            else self.n_wavefields(ndim)
        )
        expected = n_models + 2 * n_sources + n_receivers + n_wavefields
        if len(args) != expected:
            raise AssertionError(
                f"Expected {expected} backend tensors for {self.name} "
                f"(ndim={ndim}), got {len(args)}."
            )
        items = list(args)
        models = items[:n_models]
        del items[:n_models]
        source_amplitudes = items[:n_sources]
        del items[:n_sources]
        sources_i = items[:n_sources]
        del items[:n_sources]
        receivers_i = items[:n_receivers]
        del items[:n_receivers]
        wavefields = items
        return models, source_amplitudes, sources_i, receivers_i, wavefields


def channel_kwargs(spec: PropagatorSpec, kind: str, channel: str) -> tuple[str, ...]:
    """The flat keyword names of one channel of one kind, in group order.

    The single rule that derives every flat channel keyword from a channel
    name, so that no propagator hard-codes the pairing. ``kind`` is one of
    ``"source"``, ``"receiver"``, ``"bg_receiver"``, ``"sc_receiver"``; the
    returned names are the amplitudes, locations and mask stems of that kind
    (the source kind has all three, the receiver kinds have locations and
    mask), each suffixed with ``_<channel>`` unless the propagator declares
    ``channel_suffixes=False``, in which case the stem is used bare.

    Args:
        spec: The propagator's spec.
        kind: The channel-argument group.
        channel: A canonical channel name (resolve aliases first, with
            :meth:`PropagatorSpec.canonical_channel`).

    Returns:
        The keyword names, in canonical signature order.

    """
    return spec.channel_kwargs(kind, channel)


def _spec_value_error(spec: PropagatorSpec, message: str) -> DeepwaveValueError:
    """A ``DeepwaveValueError`` prefixed with the propagator's name."""
    return DeepwaveValueError(f"{spec.name}: {message}")


def _normalise_source_spec(
    spec: PropagatorSpec, channel: str, value: Any
) -> tuple[Any, Any, Any]:
    """Unpacks a ``sources=`` value into ``(amplitudes, locations, mask)``.

    ``SourceSpec`` is a union of a 2-tuple and a 3-tuple; normalising to a
    fixed-length 3-tuple here is what lets every caller (and ``mypy``) treat
    the two forms identically.
    """
    if isinstance(value, torch.Tensor) or not isinstance(value, tuple):
        raise DeepwaveTypeError(
            f"{spec.name}: sources[{channel!r}] must be a tuple "
            f"(amplitudes, locations) or (amplitudes, locations, mask), not "
            f"{type(value).__name__}."
        )
    if len(value) == 2:
        return (value[0], value[1], None)
    if len(value) == 3:
        return (value[0], value[1], value[2])
    raise DeepwaveTypeError(
        f"{spec.name}: sources[{channel!r}] must be a 2-tuple "
        f"(amplitudes, locations) or a 3-tuple (amplitudes, locations, mask), "
        f"but has {len(value)} elements."
    )


def _normalise_receiver_spec(
    spec: PropagatorSpec, kind: str, channel: str, value: Any
) -> tuple[Any, Any]:
    """Unpacks a ``receivers=`` value into ``(locations, mask)``."""
    parameter = "bg_receivers" if kind == "bg_receiver" else "receivers"
    if isinstance(value, tuple):
        if len(value) == 2:
            return (value[0], value[1])
        raise DeepwaveTypeError(
            f"{spec.name}: {parameter}[{channel!r}] must be a locations "
            f"object or a 2-tuple (locations, mask), but has "
            f"{len(value)} elements."
        )
    return (value, None)


def _reject_duplicate_channel(
    spec: PropagatorSpec,
    kind: str,
    parameter: str,
    channel: str,
    flat: abc.Mapping[str, Any],
    names: abc.Sequence[str],
) -> None:
    """Raises if a channel is given both in a mapping and as flat keywords."""
    given = [name for name in names if flat.get(name) is not None]
    if not given:
        return
    spelled = "=/".join(given) + "="
    key = spec.canonical_channel(channel, kind)
    example = (
        f"{parameter}={{{key!r}: (amplitudes, locations)}}"
        if parameter == "sources"
        else f"{parameter}={{{key!r}: locations}}"
    )
    raise _spec_value_error(
        spec,
        f"channel {channel!r} was given both in {parameter}= and as the flat "
        f"keywords {spelled}. Pass the channel once: either {example} (the "
        f"documented form) or the flat {spelled} keywords.",
    )


def _reject_duplicate_mapping_key(
    spec: PropagatorSpec,
    parameter: str,
    key: str,
    channel: str,
    names: abc.Sequence[str],
    written: abc.Mapping[str, str],
) -> None:
    """Raises if this mapping already named the same channel under another key.

    Two keys of one mapping can name one channel, because a channel can have
    more than one accepted spelling: the deprecated ``"p"`` for a source
    channel's quantity name, or any permanent alias a propagator declares.
    The keys differ, so the mapping itself cannot notice; without this check
    the collision is reported by :func:`_reject_duplicate_channel` as a clash
    with flat keywords, which in this case the caller never passed.
    """
    earlier = next((written[name] for name in names if name in written), None)
    if earlier is None:
        return
    spellings = f"{earlier!r} and {key!r}"
    raise _spec_value_error(
        spec,
        f"channel {channel!r} was given twice inside {parameter}=, under two "
        f"of its spellings ({spellings}). Pass it once, under either "
        "spelling.",
    )


def _resolve_mapping_channel(
    spec: PropagatorSpec, kind: str, parameter: str, key: str
) -> str:
    """Canonicalises and validates one mapping key, or raises."""
    if not isinstance(key, str):
        raise DeepwaveTypeError(
            f"{spec.name}: {parameter} keys must be channel-name strings, "
            f"but got {type(key).__name__}."
        )
    replacement = spec.deprecated_channel_replacement(kind, key)
    if replacement is not None:
        warn_deprecated_channel_spelling(
            spec,
            f"the {parameter}= channel key {key!r}",
            f"{replacement!r}",
            # warn_deprecated_channel_spelling <- _resolve_mapping_channel
            # <- _merge_one_mapping <- expand_channel_mappings <- the
            # propagator <- the user.
            stacklevel=6,
        )
    channel = spec.canonical_channel(key, kind)
    if channel not in spec.all_channels(kind):
        # The scattered receivers are spelled ``sc_receiver_*`` as arguments
        # but are what ``PropagatorSpec.receiver_channels`` returns, so the
        # advice names the method that exists rather than the argument prefix.
        introspection = "receiver" if kind == "sc_receiver" else kind
        raise _spec_value_error(
            spec,
            f"{parameter}: unknown channel {key!r}. Valid "
            f"{kind.replace('_', ' ')} channels are "
            f"{spec.describe_channels(kind)}. Use "
            f"deepwave.spec(deepwave.{spec.name}).{introspection}_channels(ndim) "
            "to list them.",
        )
    return channel


def _merge_one_mapping(
    spec: PropagatorSpec,
    kind: str,
    parameter: str,
    mapping: Optional[abc.Mapping[str, Any]],
    flat: dict[str, Any],
) -> None:
    """Merges one channel mapping into ``flat`` in place."""
    if mapping is None:
        return
    if not isinstance(mapping, abc.Mapping):
        shape = (
            "(amplitudes, locations[, mask])"
            if kind == "source"
            else "locations, or (locations, mask)"
        )
        raise DeepwaveTypeError(
            f"{spec.name}: {parameter}= must be a mapping of channel name to "
            f"{shape}, not {type(mapping).__name__}."
        )
    # Which key of *this* mapping wrote each flat entry, so that a channel
    # named twice inside one mapping (once canonically and once by its
    # permanent alias) is reported as what it is, rather than as a collision
    # with flat keywords the caller never passed.
    written: dict[str, str] = {}
    for key, value in mapping.items():
        channel = _resolve_mapping_channel(spec, kind, parameter, key)
        names = channel_kwargs(spec, kind, channel)
        _reject_duplicate_mapping_key(spec, parameter, key, channel, names, written)
        _reject_duplicate_channel(spec, kind, parameter, key, flat, names)
        written.update(dict.fromkeys(names, key))
        if kind == "source":
            amplitudes, locations, mask = _normalise_source_spec(spec, key, value)
            flat[names[0]] = amplitudes
            flat[names[1]] = locations
            flat[names[2]] = mask
        else:
            locations, mask = _normalise_receiver_spec(spec, kind, key, value)
            flat[names[0]] = locations
            flat[names[1]] = mask


def expand_channel_mappings(
    spec: PropagatorSpec,
    flat: dict[str, Any],
    *,
    sources: Optional[abc.Mapping[str, Any]] = None,
    receivers: Optional[abc.Mapping[str, Any]] = None,
    bg_receivers: Optional[abc.Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Merges the channel-keyed mappings into the flat keyword dictionary.

    The single choke point of the mapping form, called as the first
    executable statement of every propagator functional. It rewrites
    ``sources=``/``receivers=``/``bg_receivers=`` into exactly the flat
    keywords the propagator already understands and then gets out of the way,
    so "the mapping form equals the flat form" is true by construction rather
    than by testing: everything numeric downstream sees one code path.

    The objects handed in are the objects the propagator receives -- nothing
    is copied, cloned or moved -- so ``grad_fn`` and ``data_ptr`` are
    preserved and gradients reach the caller's own tensors.

    Args:
        spec: The calling propagator's spec.
        flat: The propagator's own flat channel keywords, by name. Modified
            in place and returned.
        sources: The ``sources=`` mapping, or ``None``.
        receivers: The ``receivers=`` mapping, or ``None``.
        bg_receivers: The ``bg_receivers=`` mapping, or ``None`` (and
            unavailable by signature on a non-Born propagator).

    Returns:
        ``flat``, with the mapping entries merged in.

    Raises:
        deepwave.DeepwaveValueError: If a channel is given in both forms, or
            a key names no channel of this propagator.
        deepwave.DeepwaveTypeError: If a mapping value has the wrong shape.

    """
    _merge_one_mapping(spec, "source", "sources", sources, flat)
    # ``receivers=`` means the scattered receivers on a Born propagator,
    # whose canonical flat spelling is ``sc_receiver_*``. ``scalar_born``
    # additionally still accepts the deprecated bare spelling, but the
    # caller folds that into the canonical entry before calling this (see
    # :func:`merge_deprecated_keywords`), so the collision check below sees
    # it.
    receiver_kind = "sc_receiver" if spec.has_bg_receivers() else "receiver"
    _merge_one_mapping(spec, receiver_kind, "receivers", receivers, flat)
    _merge_one_mapping(spec, "bg_receiver", "bg_receivers", bg_receivers, flat)
    return flat


def warn_deprecated_channel_spelling(
    spec: PropagatorSpec, subject: str, replacement: str, *, stacklevel: int
) -> None:
    """Announces one deprecated channel spelling, once per call per name.

    The one place the wording of a channel deprecation lives, so that the
    flat keywords and the ``sources=`` mapping say the same thing about the
    same spelling. The text names the replacement, the release that removes
    the old spelling, and the warning class to filter on.
    """
    warn_deprecated_name(f"{spec.name}: {subject}", replacement, stacklevel=stacklevel)


def merge_deprecated_keywords(
    spec: PropagatorSpec,
    flat: dict[str, Any],
    *,
    stacklevel: int = 3,
    **deprecated: Any,
) -> dict[str, Any]:
    """Folds deprecated keywords into the canonical entries of ``flat``.

    Deepwave has two families of deprecated channel keyword, and this one
    helper resolves both, so there is a single message and no per-propagator
    hand-written check:

    * ``source_amplitudes_p`` and ``source_locations_p`` on the released
      ``acoustic`` and ``elastic`` propagators, replaced by the respective
      ``volume_rate`` and ``pressure_rate`` source spellings (see
      ``PropagatorSpec.deprecated_source_channels``);
      and
    * ``scalar_born``'s bare ``receiver_locations``, which v0.0.27 shipped
      and which is replaced by ``sc_receiver_locations``.

    Each warns :class:`deepwave.DeepwaveFutureWarning` once per call per
    name, keeps working until :data:`DEPRECATION_REMOVAL_VERSION`, and
    computes exactly what the replacement computes.

    Args:
        spec: The calling propagator's spec.
        flat: The propagator's canonical flat keywords, by name. Modified in
            place and returned.
        stacklevel: The warning's ``stacklevel``, counted from this function.
            The default points at the caller of the propagator that called
            this.
        **deprecated: ``<deprecated name>=(<replacement name>, <value>)`` for
            each deprecated keyword the propagator still accepts.

    Returns:
        ``flat``, with any given deprecated values written into their
        replacement entries.

    Raises:
        deepwave.DeepwaveValueError: If both spellings of one keyword are
            given.

    """
    for old, (new, value) in deprecated.items():
        if value is None:
            continue
        if flat.get(new) is not None:
            raise _spec_value_error(
                spec,
                f"pass {new} or the deprecated {old}, not both: they are two "
                "spellings of the same argument.",
            )
        warn_deprecated_channel_spelling(spec, old, new, stacklevel=stacklevel + 1)
        flat[new] = value
    return flat


def resolve_deprecated_arguments(
    spec: PropagatorSpec,
    arguments: dict[str, Any],
    *,
    stacklevel: int = 3,
    **deprecated: str,
) -> dict[str, Any]:
    """Rewrites deprecated spellings in a propagator's bound arguments.

    :func:`merge_deprecated_keywords` resolves the deprecated spellings of a
    call that the propagator runs itself. A ``checkpoint_segments=`` call
    never gets that far: it is delegated to the segment driver, which calls
    the propagator once per segment, so a deprecated keyword left in the
    bound arguments would be forwarded verbatim into every one of those
    re-entrant calls and warn again in each -- with the warning attributed to
    Deepwave's own re-entry point rather than to the user's call site. This
    resolves the deprecated spellings *before* the delegation instead, so
    that the segments only ever see canonical keywords, and the warning is
    emitted exactly once per call per deprecated name, at the call site.

    Args:
        spec: The calling propagator's spec.
        arguments: The propagator's bound arguments, as captured by
            ``locals()`` on entry. Modified in place and returned.
        stacklevel: The warning's ``stacklevel``, counted from this function.
            The default points at the caller of the propagator that called
            this.
        **deprecated: ``<deprecated name>=<replacement name>`` for each
            deprecated keyword the propagator still accepts.

    Returns:
        ``arguments``, with any deprecated value moved to its replacement
        entry and the deprecated entry set back to ``None``.

    Raises:
        deepwave.DeepwaveValueError: If both spellings of one keyword are
            given.

    """
    merge_deprecated_keywords(
        spec,
        arguments,
        stacklevel=stacklevel + 1,
        **{old: (new, arguments.get(old)) for old, new in deprecated.items()},
    )
    for old in deprecated:
        arguments[old] = None
    return arguments


#: The channel-keyed survey mappings, in the order
#: :func:`expand_channel_mappings` takes them. One list, so that the
#: propagators' own expansion and the segment driver's cannot come to accept
#: different sets.
SURVEY_MAPPING_PARAMETERS = ("sources", "receivers", "bg_receivers")


def resolve_survey_mappings(
    spec: PropagatorSpec,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Rewrites ``sources=``/``receivers=`` into flat keywords, before delegating.

    The segmented counterpart of the expansion every propagator functional
    performs on itself, and the exact analogue of
    :func:`resolve_deprecated_arguments`: a ``checkpoint_segments=`` call
    returns before it reaches its own
    :func:`expand_channel_mappings`, so without this the mapping form is
    forwarded verbatim to a driver that reads the survey off the flat
    ``source_amplitudes*`` parameters and therefore cannot see it. Resolving
    here means the segments only ever see the flat keywords, which is the
    one form the driver, the checkpoint bridge and the per-segment
    re-entrant calls already agree on.

    The expansion is the same function the unsegmented call uses, so "the
    mapping form equals the flat form" stays true by construction under
    segmentation too, including the duplicate-spelling and unknown-channel
    errors, which are raised here with the wording a direct call gives.

    Args:
        spec: The calling propagator's spec.
        arguments: The propagator's bound arguments, as captured by
            ``locals()`` on entry. Modified in place and returned.

    Returns:
        ``arguments``, with each mapping merged into the flat channel
        keywords and the mapping entry itself set back to ``None`` (so that
        the delegation's "only forward what the caller set" filter drops
        it).

    Raises:
        deepwave.DeepwaveValueError: If a channel is given in both forms, or
            a key names no channel of this propagator.
        deepwave.DeepwaveTypeError: If a mapping value has the wrong shape.

    """
    mappings = {
        name: arguments.get(name)
        for name in SURVEY_MAPPING_PARAMETERS
        if name in arguments
    }
    if not any(value is not None for value in mappings.values()):
        return arguments
    expand_channel_mappings(spec, arguments, **mappings)
    for name in mappings:
        arguments[name] = None
    return arguments


def reject_meaningless_options(spec: PropagatorSpec, **options: Any) -> None:
    """Raises if a shared option that this propagator cannot use is non-default.

    Every propagator accepts every shared option at its default value, so
    that a generic wrapper can forward one uniform keyword set without
    knowing which propagator it is driving. A *meaningful* non-default value
    of an option the propagator does not implement is an instruction it
    cannot carry out, and is therefore an error rather than something to
    ignore silently.

    Args:
        spec: The calling propagator's spec.
        **options: The shared options as the caller received them. Every key
            must be one of :data:`SHARED_OPTION_DEFAULTS`.

    Raises:
        deepwave.DeepwaveValueError: If any named option that is not in
            ``spec.supported_options`` differs from its documented default.

    """
    offenders = []
    for name, value in options.items():
        if name not in SHARED_OPTION_DEFAULTS:
            raise AssertionError(f"unknown shared option {name!r}")
        if name in spec.supported_options:
            continue
        if value != SHARED_OPTION_DEFAULTS[name]:
            offenders.append((name, value))
    if not offenders:
        return
    raise _spec_value_error(spec, _meaningless_option_message(spec, offenders))


def reject_unsupported_forward_wavefield(spec: PropagatorSpec, wanted: bool) -> None:
    """Raises if this propagator cannot retain the forward wavefield.

    Only ``scalar`` currently exposes the public option. This internal guard
    protects callers that request it through a propagator without support;
    such a request raises
    :exc:`deepwave.DeepwaveNotImplementedError` rather than
    :exc:`deepwave.DeepwaveValueError`.

    Args:
        spec: The calling propagator's spec.
        wanted: The value the caller passed.

    Raises:
        deepwave.DeepwaveNotImplementedError: If ``wanted`` is true and
            ``spec.has_forward_wavefield`` is false.
    """
    if not wanted or spec.has_forward_wavefield:
        return
    supporting = ", ".join(
        name for name in PROPAGATOR_NAMES if propagator_spec(name).has_forward_wavefield
    )
    raise DeepwaveNotImplementedError(
        f"{spec.name}: backward_callback_needs_forward_wavefield=True is not "
        f"implemented for this propagator, so a backward_callback here cannot "
        f'be served get_wavefield(role="forward"). None of its backends '
        f"retains the forward wavefield through the backward pass (most "
        f"store a pre-multiplied gradient-correlation kernel instead, from "
        f"which the field cannot be recovered). Record the forward field with "
        f"a forward_callback instead, or omit the argument. The propagators "
        f"that implement it are: {supporting}."
    )


#: The nine shipped propagator functionals, in the order used throughout
#: the library and the test suite. Defined here, next to
#: :class:`PropagatorSpec`, because the spec-driven helpers below need to walk
#: every propagator's spec; ``deepwave/_capabilities.py`` re-exports it.
PROPAGATOR_NAMES: tuple[str, ...] = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)


def propagator_spec(name: str) -> "PropagatorSpec":
    """The spec of the named propagator, imported lazily.

    Lazy because :mod:`deepwave._engine` is imported by every propagator
    module: a top-level import of them here would be circular.
    """
    module = importlib.import_module(f"deepwave.{name}")
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, PropagatorSpec):  # static
        raise AssertionError(  # noqa: TRY004
            f"deepwave.{name} has no PropagatorSpec named SPEC"
        )
    return spec


def _supporting_propagators(option: str) -> list[str]:
    """The propagator names whose spec lists ``option`` as supported.

    Generated from the specs themselves so that the remedy sentence of the
    error message cannot go stale when a propagator gains or loses an option.
    """
    return [
        name
        for name in PROPAGATOR_NAMES
        if option in propagator_spec(name).supported_options
    ]


#: The explanation of each shared option, used to word the "this has no
#: effect here" error. One entry per :data:`SHARED_OPTION_DEFAULTS` key, keyed
#: by the option name, as a ``(reason, remedy)`` pair. ``{name}`` is the
#: calling propagator.
_OPTION_REASONS: abc.Mapping[str, str] = MappingProxyType(
    {
        "mpml_fraction": (
            "The multiaxial PML (Meza-Fajardo and Papageorgiou, 2008) exists "
            "to cure a late-time corner instability that needs shear waves "
            "and a free-surface waveguide mode to arise, and {name} has "
            "neither. Pass mpml_fraction=0.0 or omit it."
        ),
        "surface_method": (
            "It selects how the free surface implied by elastic vacuum cells "
            "(buoyancy = mu = 0) is discretised, and {name} has no such "
            "construction. For a free surface here, use "
            'boundary=[..., "free", ...] with pml_width 0 on that face. Pass '
            'surface_method="ivm" or omit it.'
        ),
        "q_freq": (
            "q_freq, q_n_mechanisms and q_band configure the constant-Q "
            "generalised-standard-linear-solid attenuation model, and {name} "
            "is lossless. Use deepwave.viscoacoustic for acoustic attenuation, "
            "or omit them."
        ),
    }
)


def _meaningless_option_message(
    spec: PropagatorSpec, offenders: abc.Sequence[tuple[str, Any]]
) -> str:
    """Words the :func:`reject_meaningless_options` error.

    Names every offending option with the value actually given, explains in
    one sentence why the option cannot mean anything here, and closes with
    the propagators that *do* implement it -- generated from their specs, so
    the list cannot go stale.
    """
    given = ", ".join(f"{name}={value!r}" for name, value in offenders)
    verb = "have" if len(offenders) > 1 else "has"
    first = offenders[0][0]
    reason_key = "q_freq" if first in ("q_n_mechanisms", "q_band") else first
    supporting = _supporting_propagators(first)
    support = (
        " The propagators that use it are: " + ", ".join(supporting) + "."
        if supporting
        else ""
    )
    # ``surface_method`` is the one option whose non-default value is
    # meaningless for two different reasons: an acoustic-family propagator has
    # no vacuum-cell free surface at all, while the non-isotropic elastic
    # propagators have one but no "pm" analogue of it. The second case is not
    # "no effect" but "not implemented", so it is worded as its own sentence
    # rather than squeezed into the shared template.
    if reason_key == "surface_method" and _has_vacuum_cells(spec):
        return (
            f'surface_method="pm" is not implemented for {spec.name}. The '
            "parameter-modified free surface of Cao and Chen (2018) is "
            "currently derived only for the isotropic elastic system; there "
            "is no VTI/Born analogue in Deepwave yet. "
            f"{spec.name} uses the improved vacuum method (surface_method="
            '"ivm", the default) for its vacuum cells. Pass '
            f'surface_method="ivm" or omit it.{support}'
        )
    reason = _OPTION_REASONS[reason_key].format(name=spec.name)
    return f"{given} {verb} no effect for {spec.name}. {reason}{support}"


def _has_vacuum_cells(spec: PropagatorSpec) -> bool:
    """Whether this propagator has the elastic vacuum-cell free surface.

    The elastic family does (its free surface is implied by cells with
    ``buoyancy = mu = 0``); the scalar, acoustic and pseudo-acoustic VTI
    propagators have no shear modulus and therefore no such construction.
    """
    return spec.name.startswith("elastic")


# The full historical ``storage_mode`` value set, shared by every propagator's
# public signature (``Literal["device", "cpu", "disk", "none"]``); used to name
# the unsupported alternatives in the message below without hard-coding them at
# each call site.
_ALL_STORAGE_MODES: tuple[str, ...] = ("device", "cpu", "disk", "none")


def reject_unsupported_python_backend_options(
    spec: PropagatorSpec,
    backend: str,
    *,
    storage_mode: StorageMode,
    storage_compression: bool,
    has_backward_callback: bool,
    is_batched: bool,
) -> None:
    """Rejects a pure-Python backend call using a capability it doesn't support.

    Single source of truth for what used to be nine separately maintained (and
    wording-drifted) guard blocks, one near the top of each propagator's
    ``eager``/``torch_compile`` backend function. Reads the calling
    propagator's supported capabilities from ``spec``'s ``python_backend_*``
    fields and raises a ``RuntimeError`` naming the rejected value, what is
    supported instead, and ``backend`` -- the Python backend the caller
    actually selected -- so the message can never claim a different backend
    than the one in use (the historical defect this replaces: some sites
    hard-coded ``backend="eager"`` in their message even when reached via
    ``backend="torch_compile"``).

    Args:
        spec: The calling propagator's capability declaration.
        backend: The Python backend name to report in the message --
            resolved (``"eager"`` or ``"compile"``) for a
            propagator with a compiled backend, since reaching this function
            at all means the caller explicitly chose one of those two; the
            caller's own pre-resolution choice otherwise (see above).
        storage_mode: The resolved ``storage_mode`` string.
        storage_compression: Whether storage compression was requested.
        has_backward_callback: Whether a ``backward_callback`` was supplied.
        is_batched: Whether any input tensor is inside ``torch.func.vmap``
            (only consulted when ``has_backward_callback`` is true).

    Raises:
        ValueError: If ``storage_mode`` is not one of the recognised
            ``_ALL_STORAGE_MODES`` values (the same error every other
            propagator entry point raises for a typo, checked here first so a
            typo is never misreported as a backend-capability limit).
        RuntimeError: If ``storage_mode``, ``storage_compression``, or
            ``has_backward_callback`` names a capability ``spec`` says
            ``backend`` does not support.

    """
    # Validated through the shared helper purely for its message; the enum
    # it returns is not needed here.
    storage_mode_enum(storage_mode)
    if storage_mode not in spec.python_backend_storage_modes:
        supported = " or ".join(
            f'"{mode}"' for mode in spec.python_backend_storage_modes
        )
        unsupported = "/".join(
            mode
            for mode in _ALL_STORAGE_MODES
            if mode not in spec.python_backend_storage_modes
        )
        if spec.has_compiled_backend:
            remedy = (
                'use the default backend="auto" (which selects the compiled '
                f"backend here) for {unsupported} storage."
            )
        else:
            remedy = (
                f"{unsupported} storage is not available for {spec.name} at all "
                "(it has no compiled C/CUDA backend to fall back to)."
            )
        raise DeepwaveRuntimeError(
            f'storage_mode="{storage_mode}" is not supported by {spec.name}\'s '
            f'backend="{backend}" (supported: {supported}); {remedy}'
        )
    if storage_compression and not spec.python_backend_storage_compression:
        if spec.has_compiled_backend:
            remedy = (
                'use the default backend="auto" (which selects the compiled '
                "backend here) for compressed storage."
            )
        else:
            remedy = (
                f"compressed storage is not available for {spec.name} at all "
                "(it has no compiled C/CUDA backend to fall back to)."
            )
        raise DeepwaveRuntimeError(
            f"Storage compression is not supported by {spec.name}'s "
            f'backend="{backend}"; {remedy}'
        )
    if has_backward_callback:
        if not spec.python_backend_backward_callback:
            if spec.has_compiled_backend:
                remedy = (
                    'use the default backend="auto" (which selects the '
                    "compiled backend here)."
                )
            else:
                remedy = (
                    f"backward_callback is not available for {spec.name} at all "
                    "(it has no compiled C/CUDA backend to fall back to)."
                )
            raise DeepwaveRuntimeError(
                f"backward_callback is not supported by {spec.name}'s "
                f'backend="{backend}"; {remedy}'
            )
        if is_batched and not spec.python_backend_backward_callback_in_vmap:
            raise DeepwaveRuntimeError(
                "backward_callback is not supported inside vmap by "
                f'{spec.name}\'s backend="{backend}".'
            )


# --- AuxiliaryState and the public/internal name collapse -------------------

#: Removal version stated in every deprecation this module emits. Re-exported
#: from :mod:`deepwave._exceptions`, which every module can import without a
#: cycle, so that all of Deepwave's deprecation messages name one version.
DEPRECATION_REMOVAL_VERSION = _DEPRECATION_REMOVAL_VERSION


#: Per-face boundary condition names, indexed by the integer face codes
#: :func:`deepwave._common.set_boundary` produces. Kept here so that
#: :class:`AuxiliaryState` can record the resolved conditions as readable
#: strings without importing ``_common``.
BOUNDARY_CONDITION_NAMES: tuple[str, ...] = ("pml", "free", "rigid", "periodic")

_warned_once_causes: set[str] = set()


def _warn_once(
    cause: str, message: str, category: type[Warning], stacklevel: int
) -> None:
    """Emits ``message`` at most once per process for a given ``cause``.

    As in :func:`deepwave._common._warn_once_per_cause`, a plan in progress
    reports the warning without spending the budget the real call needs, and
    does so by not writing to the registry rather than by unwinding a write
    afterwards -- an unwind cannot tell its own entries from those another
    thread added while it was running. ``stacklevel`` is likewise only the
    fallback: the level actually used is
    :func:`deepwave._exceptions.stacklevel_outside_deepwave`'s, so the
    warning is reported at the user's own line.
    """
    if cause in _warned_once_causes:
        return
    from deepwave import _plan

    if _plan.active_probe() is None:
        _warned_once_causes.add(cause)
    # ``stacklevel`` is counted from this function's caller and
    # ``warn_at_user_call``'s fallback from its own, one frame lower.
    warn_at_user_call(message, category, stacklevel + 1)


def boundary_condition_names(codes: abc.Sequence[int]) -> tuple[str, ...]:
    """Turns per-face boundary condition codes into their names."""
    return tuple(BOUNDARY_CONDITION_NAMES[code] for code in codes)


def scattered_state_name(name: str) -> str:
    """The Born spelling of the scattered counterpart of state field ``name``.

    One spelling, everywhere: ``_sc`` goes *before* the time index, so
    ``pressure_0`` becomes ``pressure_sc_0`` and ``m_vyy_0`` becomes
    ``m_vyy_sc_0``. This is what :func:`deepwave.scalar_born` has always
    spelled its scattered fields (``wavefield_sc_0``, ``psiy_sc_m1``), and
    what the acoustic and elastic Born propagators' public outputs already
    used for their scattered physical fields; appending ``_sc`` instead would
    read as though ``sc`` were a second time index, and left the two Born
    families disagreeing with each other and with themselves.

    Args:
        name: A state field name ending in a time index (``_0``, ``_m1``).

    Returns:
        The same name with ``_sc`` inserted before that time index.

    Raises:
        AssertionError: If ``name`` has no recognised time index, which would
            mean the caller is renaming something that is not a state field.
    """
    stem, sep, index = name.rpartition("_")
    if not sep or not (
        index.isdigit() or (index.startswith("m") and index[1:].isdigit())
    ):
        raise AssertionError(f"{name!r} does not end in a time index")
    return f"{stem}_sc_{index}"


def collapse_auxiliary_state_names(
    public_wavefield_names: abc.Sequence[str],
    auxiliary_state_names: abc.Collection[str],
) -> list[str]:
    """Replaces the auxiliary-state block of a name list with one name.

    ``public_wavefield_names`` is a propagator's state fields in public return
    order; ``auxiliary_state_names`` is the subset of them that is carried
    opaquely. Every listed name is removed and the single name
    ``"auxiliary_state"`` is inserted at the index the first removed name
    occupied, so the surviving fields keep their relative order and the
    leading physical fields keep their indices. An empty
    ``auxiliary_state_names`` returns the list unchanged, which is what a
    propagator that folds nothing gets.

    This is the name-side half of a pair: :func:`collapse_auxiliary_state_values`
    performs the identical collapse on the values, driven by the same two
    lists, so names and values cannot drift apart.
    """
    if not auxiliary_state_names:
        return list(public_wavefield_names)
    collapsed: list[str] = []
    inserted = False
    for name in public_wavefield_names:
        if name in auxiliary_state_names:
            if not inserted:
                collapsed.append("auxiliary_state")
                inserted = True
            continue
        collapsed.append(name)
    if not inserted:
        raise AssertionError(
            "None of the auxiliary-state names "
            f"{sorted(auxiliary_state_names)} appear in the public wavefield "
            f"names {list(public_wavefield_names)}."
        )
    return collapsed


#: The layout version of :class:`AuxiliaryState`'s tensor bundle, recorded on
#: every state a run returns. Bump it only when the layout changes: when the
#: set of names, their meanings or their shapes stop being readable as they
#: were. Adding a recorded run parameter, renaming the class, changing a
#: message or changing the numbers a propagator produces are not layout
#: changes, so a state saved by one release stays loadable by the next unless
#: the bundle itself was rearranged.
AUXILIARY_STATE_FORMAT_VERSION = 2


@dataclass(frozen=True)
class _StateDescription:
    """Versioned numerical interpretation of an auxiliary tensor bundle."""

    format_version: int
    propagator: Optional[str]
    ndim: Optional[int]
    accuracy: Optional[int]
    grid_spacing: Optional[tuple[float, ...]]
    pml_width: Optional[tuple[int, ...]]
    boundary: Optional[tuple[str, ...]]
    pml_freq: Optional[float]
    max_vel: Optional[float]
    internal_dt: Optional[float]
    mpml_fraction: Optional[float]
    surface_method: Optional[str]
    q_freq: Optional[float]
    q_band: Optional[tuple[float, float]]
    n_mechanisms: Optional[int]
    fit_convention: Optional[str]
    scheme: Optional[str]
    origin: Optional[tuple[int, ...]]
    shape: Optional[tuple[int, ...]]
    field_names: Optional[tuple[str, ...]]
    field_shapes: Optional[tuple[tuple[int, ...], ...]]


@dataclass(frozen=True, eq=False, repr=False)
class AuxiliaryState:
    """Opaque absorbing-boundary and auxiliary state of one propagation.

    A propagator's absorbing boundary (a CFS-CPML) carries memory variables
    from one time step to the next, and continuing a simulation exactly
    requires carrying them across the join. They are implementation state, not
    physics: this object holds them so that a continuation is a matter of
    passing one value back (``prop(..., initial_state=out)``) rather than of
    naming fields whose number and shape are properties of the
    absorbing-boundary implementation.

    **The bundle is opaque as an output element**: do not index it
    positionally, and do not build on how many tensors it holds, on their
    order or on their shapes (see :doc:`/stability`). Its keys are named --
    they are the provisional internal diagnostic names listed in
    :doc:`/callback_names`. The mapping surface exists for inspection and
    debugging; the supported continuation use is to hand an ``AuxiliaryState`` back to
    a propagator unchanged.

    It is a read-only mapping from name to :class:`torch.Tensor`
    (:meth:`names`, ``bs[name]``, ``name in bs``, ``len(bs)``, iteration,
    :meth:`keys`, :meth:`items`, :meth:`values`, :meth:`as_dict`), plus
    :meth:`to` for moving it between devices and dtypes and :meth:`detach`
    for cutting the autograd graph at a join on purpose. It is not a
    :class:`torch.Tensor` and never becomes one; the tensors it holds keep
    their ``grad_fn``, so a continuation is differentiable end to end.

    ``==`` is identity comparison, deliberately: a generated ``__eq__`` would
    compare the tensors and return a ``Tensor``, making ``bs1 == bs2`` raise
    inside any ``if`` or ``in``. Use :meth:`compatible_with` to ask whether
    one auxiliary state may be reused by another call.

    Registered as a **pytree node**, so ``torch.func`` and ``torch.compile``
    reach the tensors it holds (it is one element of every propagator's
    output, so without that the output could not be flattened to tensors at
    all). The children are the tensors; the tensor names and every recorded
    run parameter, ``format_version`` included, travel as the node's context
    and come back unchanged, so a state that has been through a transform is
    still one a continuation checks itself against rather than an unrecorded
    bundle.

    Attributes:
        propagator: The name of the propagator that produced it (e.g.
            ``"scalar"``), or ``None`` for a state that was assembled by hand
            rather than returned by a run.
        ndim: The number of spatial dimensions of the run that produced it.
        accuracy: The finite-difference order of accuracy that run used.
        pml_width: The resolved per-face PML widths, length ``2 * ndim``.
        boundary: The resolved per-face boundary conditions, length
            ``2 * ndim``, from ``{"pml", "free", "rigid", "periodic"}``.
        n_mechanisms: The number of relaxation mechanisms (1 except for the
            viscoacoustic propagator).
        mpml: Whether multiaxial PML cross-damping was in use.
        shape: The shape of the tensors, ``[n_shot, *spatial]``, where the
            spatial part is the extracted survey region plus the PML. Checked:
            a continuation whose resolved simulation domain is a different
            shape is not continuing the same simulation.
        dtype: The dtype of the tensors.
        device: The device the tensors are on.
        origin: The index, in the full model, of cell 0 of the region the run
            extracted, length ``ndim``. Used (not checked) to restore the same
            survey region on a continuation. The region's *size* is checked,
            through ``shape``.
        pml_freq: The PML frequency, in Hz, that run resolved.
        max_vel: The maximum wavespeed, in m/s, that run used to build the
            PML profiles and choose its internal time step. A changed resolved
            value is incompatible with checked continuation.
        internal_dt: The internal (sub-stepped) time step, in seconds, that
            run took. Checked: it is the condition every propagator's
            ``initial_state`` documentation names for a continuation that
            exactly reproduces a single combined run.
        scheme: Names the absorbing-boundary implementation the tensors belong
            to, so that a future change of scheme cannot silently reuse state
            from the old one.
        detached_deliberately: Whether the tensors were detached from the
            autograd graph on purpose, by :meth:`detach`. A state that arrived
            detached by accident (see :meth:`detach`) is warned about; one
            marked this way is accepted silently. It is not part of
            :meth:`compatible_with`: it says nothing about what the tensors
            mean.
        format_version: The version of the tensor bundle's layout and numerical
            interpretation. A correction can require a new version, including
            in a patch release. Saved states require a compatible implementation;
            they are not a long-term interchange format. Handing
            back a state whose ``format_version`` differs from the one this
            release writes raises :class:`deepwave.DeepwaveValueError`, naming
            both versions, rather than reading the tensors as something they
            are not.

    A field recorded as ``None`` was not supplied by a run and is neither
    reported nor checked.

    """

    _tensors: abc.Mapping[str, torch.Tensor]
    propagator: Optional[str] = None
    ndim: Optional[int] = None
    accuracy: Optional[int] = None
    pml_width: Optional[tuple[int, ...]] = None
    boundary: Optional[tuple[str, ...]] = None
    n_mechanisms: Optional[int] = None
    mpml: Optional[bool] = None
    mpml_fraction: Optional[float] = None
    grid_spacing: Optional[tuple[float, ...]] = None
    surface_method: Optional[str] = None
    q_freq: Optional[float] = None
    q_band: Optional[tuple[float, float]] = None
    fit_convention: Optional[str] = None
    field_names: Optional[tuple[str, ...]] = None
    field_shapes: Optional[tuple[tuple[int, ...], ...]] = None
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    origin: Optional[tuple[int, ...]] = None
    pml_freq: Optional[float] = None
    scheme: Optional[str] = "cfs-cpml"
    detached_deliberately: bool = False
    max_vel: Optional[float] = None
    internal_dt: Optional[float] = None
    format_version: int = AUXILIARY_STATE_FORMAT_VERSION
    _description: _StateDescription = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Copies the tensor mapping, so that nothing can mutate it later.

        The class documents itself as frozen and read-only. Without this it
        would only be frozen in its own fields: ``AuxiliaryState(my_dict)``
        would keep ``my_dict``, and a later ``my_dict[name] = other`` would
        change what the "read-only" state holds -- and, if it had already
        been handed to a propagator, what a continuation ran from.

        It also rejects a non-integer ``format_version``. The layout check is
        an equality test and ``1.0 == 1`` in Python, so a float would be
        accepted here and then silently pass for a version it does not name.
        ``bool`` is excluded with it, being an ``int`` subclass.
        """
        object.__setattr__(self, "_tensors", dict(self._tensors))
        if not isinstance(self.format_version, int) or isinstance(
            self.format_version, bool
        ):
            raise DeepwaveTypeError(
                "auxiliary_state's format_version must be an int, but got "
                f"{self.format_version!r} of type "
                f"{type(self.format_version).__name__}."
            )
        names = self.field_names
        shapes = self.field_shapes
        if names is None and self._tensors:
            names = tuple(self._tensors)
            object.__setattr__(self, "field_names", names)
        if shapes is None and self._tensors:
            shapes = tuple(tuple(t.shape) for t in self._tensors.values())
            object.__setattr__(self, "field_shapes", shapes)
        object.__setattr__(
            self,
            "_description",
            _StateDescription(
                self.format_version,
                self.propagator,
                self.ndim,
                self.accuracy,
                self.grid_spacing,
                self.pml_width,
                self.boundary,
                self.pml_freq,
                self.max_vel,
                self.internal_dt,
                self.mpml_fraction,
                self.surface_method,
                self.q_freq,
                self.q_band,
                self.n_mechanisms,
                self.fit_convention,
                self.scheme,
                self.origin,
                self.shape,
                names,
                shapes,
            ),
        )

    @classmethod
    def _from_flat(
        cls,
        tensors: abc.Mapping[str, torch.Tensor],
        fingerprint: Optional["AuxiliaryState"] = None,
    ) -> "AuxiliaryState":
        """Builds an auxiliary state from a flat ``{name: tensor}`` mapping.

        Used by code that already holds the individual tensors --
        :mod:`deepwave.checkpointing` threads them as positional tensors
        because that is what makes a segment checkpointable, and rebuilds a
        auxiliary state from them for each segment call. ``fingerprint``, if
        given, is an existing auxiliary state whose recorded run parameters
        describe these tensors too (for checkpointing, the first segment's).
        Without one, the result records nothing and is therefore checked
        against nothing.
        """
        values = dict(tensors)
        if fingerprint is None:
            return cls(values)
        return cls(
            values,
            propagator=fingerprint.propagator,
            ndim=fingerprint.ndim,
            accuracy=fingerprint.accuracy,
            pml_width=fingerprint.pml_width,
            boundary=fingerprint.boundary,
            n_mechanisms=fingerprint.n_mechanisms,
            mpml=fingerprint.mpml,
            mpml_fraction=fingerprint.mpml_fraction,
            grid_spacing=fingerprint.grid_spacing,
            surface_method=fingerprint.surface_method,
            q_freq=fingerprint.q_freq,
            q_band=fingerprint.q_band,
            fit_convention=fingerprint.fit_convention,
            field_names=fingerprint.field_names,
            field_shapes=fingerprint.field_shapes,
            shape=(
                tuple(next(iter(values.values())).shape)
                if values
                else fingerprint.shape
            ),
            dtype=fingerprint.dtype,
            device=fingerprint.device,
            origin=fingerprint.origin,
            pml_freq=fingerprint.pml_freq,
            scheme=fingerprint.scheme,
            max_vel=fingerprint.max_vel,
            internal_dt=fingerprint.internal_dt,
            # Carried, not defaulted: the caller's own state is the fingerprint
            # on a checkpointed continuation, so letting this fall back to the
            # class default would hide a format_version the caller actually
            # passed -- the compatibility check downstream would then compare
            # this release's version against itself and accept a state written
            # by a layout this release cannot read.
            format_version=fingerprint.format_version,
            # Carried explicitly: this constructor copies the fingerprint
            # field by field, so a deliberately detached state threaded
            # through a segmented run would otherwise lose its intent at the
            # first segment boundary and start warning. It is carried only
            # while it is still true of these tensors: the flag exempts a
            # state from the lost-gradient warning, so inheriting it onto
            # tensors that do carry gradient history would silently exempt
            # every later call too.
            detached_deliberately=(
                fingerprint.detached_deliberately
                and all(
                    tensor.grad_fn is None and not tensor.requires_grad
                    for tensor in values.values()
                )
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the tensors, in order."""
        return tuple(self._tensors)

    def keys(self) -> abc.KeysView[str]:
        """The names of the tensors, as a mapping view."""
        return self.as_dict().keys()

    def items(self) -> abc.ItemsView[str, torch.Tensor]:
        """The ``(name, tensor)`` pairs, as a mapping view."""
        return self.as_dict().items()

    def values(self) -> abc.ValuesView[torch.Tensor]:
        """The tensors, as a mapping view."""
        return self.as_dict().values()

    def as_dict(self) -> dict[str, torch.Tensor]:
        """A fresh ``dict`` of the tensors; mutating it does not affect this."""
        return dict(self._tensors)

    def __getitem__(self, name: str) -> torch.Tensor:
        """The tensor called ``name``."""
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not part of this auxiliary state; it holds "
                f"{', '.join(self._tensors) or '(nothing)'}."
            ) from None

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is one of the tensor names."""
        return name in self._tensors

    def __iter__(self) -> abc.Iterator[str]:
        """Iterates over the tensor names, like a mapping."""
        return iter(self._tensors)

    def __len__(self) -> int:
        """The number of tensors."""
        return len(self._tensors)

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "AuxiliaryState":
        """A copy with the tensors moved to ``device`` and/or cast to ``dtype``.

        The recorded ``device``/``dtype`` are updated to match. Neither is part
        of the compatibility check (:meth:`compatible_with`), so a round trip
        through another device -- which is what
        :func:`deepwave.checkpointing.run_segments`'s ``checkpoint_device``
        does -- is legitimate.
        """
        if device is None and dtype is None:
            return self
        moved = {n: t.to(device=device, dtype=dtype) for n, t in self._tensors.items()}
        first = next(iter(moved.values()), None)
        return replace(
            self,
            _tensors=moved,
            dtype=self.dtype if first is None else first.dtype,
            device=self.device if first is None else first.device,
        )

    def detach(self) -> "AuxiliaryState":
        """A copy whose tensors are detached from the autograd graph.

        Cutting the graph at a segment join deliberately -- truncated
        backpropagation through a long simulation -- is indistinguishable,
        from inside a propagator, from having lost the gradient by accident,
        so this records the intent: a state detached this way is accepted
        silently, while one that arrived detached is warned about.

        This detaches the auxiliary state alone, which is **not** by itself
        enough to truncate anything: the physical wavefields of the same join
        are separate fields and stay attached, so the graph is not cut, no
        memory is freed, and the gradient is still the full one. Use
        :meth:`PropagatorOutput.detach`, which cuts every field of the join,
        unless you are assembling the continuation state field by field and
        detaching the rest yourself.
        """
        return replace(
            self,
            _tensors={n: t.detach() for n, t in self._tensors.items()},
            detached_deliberately=True,
        )

    def _mismatches(self, other: "AuxiliaryState") -> list[str]:
        """The recorded run parameters of ``other`` that this one contradicts."""
        differing: list[str] = []
        for item in fields(_StateDescription):
            name = item.name
            if name == "format_version":
                continue
            mine = getattr(self._description, name)
            theirs = getattr(other._description, name)  # noqa: SLF001
            if mine is None or theirs is None:
                continue
            if isinstance(mine, float) and isinstance(theirs, float):
                same = math.isclose(mine, theirs, rel_tol=1e-12, abs_tol=0.0)
            elif (
                isinstance(mine, tuple)
                and isinstance(theirs, tuple)
                and len(mine) == len(theirs)
                and all(isinstance(x, float) for x in mine + theirs)
            ):
                same = all(
                    math.isclose(x, y, rel_tol=1e-12, abs_tol=0.0)
                    for x, y in zip(mine, theirs)
                )
            else:
                same = mine == theirs
            if not same:
                differing.append(name)
        if self.mpml is not None and other.mpml is not None and self.mpml != other.mpml:
            differing.append("mpml")
        return differing

    def compatible_with(self, other: "AuxiliaryState") -> bool:
        """Whether a call described by ``other`` may reuse this auxiliary state.

        Compares ``format_version`` -- a bundle written to a different layout
        is not readable as this one whatever else matches -- and then the
        recorded run parameters that determine what the tensors
        mean: propagator, ndim, accuracy, pml_width, boundary, n_mechanisms,
        mpml, scheme, shape and internal_dt, as well as the resolved grid,
        attenuation and absorbing-boundary profile. ``dtype`` and ``device``
        are deliberately excluded (see :meth:`to`). A changed ``pml_freq`` or
        ``max_vel`` is incompatible with checked continuation.

        ``shape`` is included -- although :meth:`to` cannot change it -- because
        it is what pins the *size* of the survey region the tensors describe.
        ``origin`` pins where that region starts; without ``shape`` a
        continuation whose sources or receivers have moved outside the
        previous region can have the state re-read at a larger extent and
        silently zero-padded, displacing every field.
        """
        return self.format_version == other.format_version and not self._mismatches(
            other
        )

    def _check_compatible(self, other: "AuxiliaryState", stacklevel: int = 3) -> None:
        """Raises unless ``other``'s call may reuse this auxiliary state."""
        # Checked before anything else: a bundle written to a different layout
        # cannot be read as this one, so the recorded run parameters that the
        # other checks compare are not meaningful until the layouts agree.
        if self.format_version != other.format_version:
            raise DeepwaveValueError(
                f"This auxiliary_state has format_version="
                f"{self.format_version!r}, but this version of Deepwave "
                f"writes and reads format_version={other.format_version!r}. "
                "The layout of the auxiliary state changed between the two, "
                "so its tensors cannot be read as this call's state. Produce "
                "the state again with this version of Deepwave, or drop "
                "initial_state to start from rest."
            )
        differing = self._mismatches(other)
        if differing:
            recorded = ", ".join(f"{n}={getattr(self, n)!r}" for n in differing)
            requested = ", ".join(f"{n}={getattr(other, n)!r}" for n in differing)
            source = (
                "a previous run"
                if self.propagator is None
                else f"a {self.propagator} run"
            )
            raise DeepwaveValueError(
                f"This auxiliary_state came from {source} with {recorded}; this "
                f"call has {requested}. An auxiliary state can only be reused by "
                "a call with the same propagator, ndim, accuracy, pml_width, "
                "boundary, number of relaxation mechanisms, PML scheme, "
                "simulation-domain shape and internal time step. "
                "Drop initial_state to start from rest, or re-run the first "
                "call with the settings you want."
            )
        if self.propagator is None and len(self._tensors):
            _warn_once(
                f"auxiliary_state_unrecorded:{other.propagator}",
                "This auxiliary_state records no propagator, so none of the "
                "consistency checks could be performed: it was assembled by "
                "hand, or its recorded run parameters were blanked, rather "
                "than being returned by a propagation. Deepwave cannot tell "
                "whether these tensors belong to this call's propagator, "
                "accuracy, PML or grid, and a mismatch will produce finite "
                "but wrong numbers rather than an error. Pass a boundary "
                "state that a run returned (initial_state=out, or "
                "out.final_state) if you want it checked.",
                UserWarning,
                stacklevel,
            )

    def __repr__(self) -> str:
        """A one-line summary that prints no tensor values."""
        parts = [self.propagator if self.propagator is not None else "unrecorded"]
        if self.ndim is not None:
            parts.append(f"ndim={self.ndim}")
        if self.accuracy is not None:
            parts.append(f"accuracy={self.accuracy}")
        if self.pml_width is not None:
            parts.append(f"pml_width={self.pml_width}")
        parts.append(f"{len(self._tensors)} fields")
        if self.shape is not None:
            parts.append(f"shape={self.shape}")
        if self.dtype is not None:
            parts.append(str(self.dtype).replace("torch.", ""))
        if self.device is not None:
            parts.append(str(self.device))
        if self.detached_deliberately:
            parts.append("detached_deliberately")
        return f"{type(self).__name__}({', '.join(parts)})"

    __str__ = __repr__


# ``AuxiliaryState`` is public (``deepwave.AuxiliaryState``) but, unlike the
# other public types, it stays defined here rather than in ``deepwave.types``:
# it is produced and consumed entirely by this engine, and
# ``PropagatorOutput`` refers to it only in annotations. It reports the
# public module for the same reasons the types in ``deepwave.types`` do --
# ``repr``, ``pickle``, ``mypy`` output and the API extract.
AuxiliaryState.__module__ = "deepwave"

#: The recorded run parameters of an :class:`AuxiliaryState`: every field
#: except the tensor bundle, in declaration order. Read from the dataclass
#: rather than listed, so a field added later is carried by the pytree
#: registration below without anything having to be kept in step by hand --
#: including ``format_version``, which a round trip must preserve exactly or a
#: state saved by one release would come back claiming to be this release's
#: layout.
AUXILIARY_STATE_PARAMETER_NAMES: tuple[str, ...] = tuple(
    item.name
    for item in fields(AuxiliaryState)
    if item.name not in ("_tensors", "_description")
)

#: The pytree context of an :class:`AuxiliaryState`: the tensor names, in
#: order, and the recorded run parameters as a tuple in
#: :data:`AUXILIARY_STATE_PARAMETER_NAMES` order.
_AuxiliaryStateContext = tuple[tuple[str, ...], tuple[Any, ...]]


def _flatten_auxiliary_state(
    state: AuxiliaryState,
) -> tuple[list[Any], _AuxiliaryStateContext]:
    """The pytree children of an auxiliary state: its tensors, in order."""
    return list(state.values()), (
        state.names,
        tuple(getattr(state, name) for name in AUXILIARY_STATE_PARAMETER_NAMES),
    )


def _unflatten_auxiliary_state(
    children: abc.Iterable[Any], context: _AuxiliaryStateContext
) -> AuxiliaryState:
    """Rebuilds an auxiliary state from its tensors and its run parameters."""
    names, parameters = context
    values = tuple(children)
    # ``zip`` truncates silently, so a mismatch would produce a state holding
    # fewer tensors than it was flattened from -- which a continuation would
    # then read as a missing field rather than as the error it is.
    if len(values) != len(names):
        raise DeepwaveValueError(
            f"AuxiliaryState: cannot be rebuilt from {len(values)} tensors, "
            f"because the {len(names)} names it was flattened from "
            f"({', '.join(names)}) each need one. A pytree unflatten must be "
            "given back exactly the children the matching flatten produced."
        )
    return AuxiliaryState(
        dict(zip(names, values)),
        **dict(zip(AUXILIARY_STATE_PARAMETER_NAMES, parameters)),
    )


def _flatten_auxiliary_state_with_keys(
    state: AuxiliaryState,
) -> tuple[list[tuple[Any, Any]], _AuxiliaryStateContext]:
    """:func:`_flatten_auxiliary_state`, keying each tensor by its name."""
    children, context = _flatten_auxiliary_state(state)
    return [
        (_pytree.mapping_key(name), child) for name, child in zip(state.names, children)
    ], context


# Registered as a pytree node so that ``torch.func`` and ``torch.compile`` can
# see the tensors an auxiliary state holds: it is one element of every
# propagator's returned ``PropagatorOutput``, so without this the output could
# not be flattened to tensors at all, and ``vmap`` could not add its batch
# dimension to the boundary's memory variables.
#
# The children are the tensors and everything else is context, which is what
# keeps the object's meaning intact through a round trip: the tensor *names*
# and every recorded run parameter -- ``format_version`` included -- come back
# unchanged, so a rebuilt state is still checkable against the call that
# receives it (:meth:`AuxiliaryState.compatible_with`) rather than an
# unrecorded bundle that no consistency check applies to. The tensor names are
# the internal spelling that stays outside the public interface; they are
# context, not keys a caller is invited to rely on.
#
# As with :class:`deepwave.PropagatorOutput`, the registration goes through
# ``deepwave._pytree`` and is optional: a PyTorch that cannot offer it leaves
# the type a pytree leaf rather than failing the import.
_pytree.register(
    AuxiliaryState,
    _flatten_auxiliary_state,
    _unflatten_auxiliary_state,
    _flatten_auxiliary_state_with_keys,
)


def collapse_auxiliary_state_values(
    public_wavefield_names: abc.Sequence[str],
    values: abc.Sequence[torch.Tensor],
    auxiliary_state_names: abc.Collection[str],
    make_state: Callable[[dict[str, torch.Tensor]], AuxiliaryState],
) -> list[Any]:
    """Replaces the auxiliary-state block of a value list with one object.

    The value-side twin of :func:`collapse_auxiliary_state_names`, driven by
    the same two name lists so that the two cannot disagree: the tensors named
    in ``auxiliary_state_names`` are collected (in ``public_wavefield_names``
    order) and handed to ``make_state``, whose result takes the position of
    the first of them.
    """
    if not auxiliary_state_names:
        return list(values)
    if len(public_wavefield_names) != len(values):
        raise AssertionError(
            f"Got {len(values)} state values for "
            f"{len(public_wavefield_names)} state names."
        )
    collected: dict[str, torch.Tensor] = {}
    collapsed: list[Any] = []
    slot: Optional[int] = None
    for name, value in zip(public_wavefield_names, values):
        if name in auxiliary_state_names:
            collected[name] = value
            if slot is None:
                slot = len(collapsed)
                collapsed.append(None)
            continue
        collapsed.append(value)
    if slot is None:
        raise AssertionError(
            "None of the auxiliary-state names "
            f"{sorted(auxiliary_state_names)} appear in the public wavefield "
            f"names {list(public_wavefield_names)}."
        )
    collapsed[slot] = make_state(collected)
    return collapsed


#: Suffixes that mark a propagator parameter as a physical initial field. The
#: segment driver threads the whole state between segments, so re-applying one
#: of these at every boundary would overwrite that state; ``initial_state=`` is
#: the supported way to start a segmented run from a field.
_INITIAL_FIELD_SUFFIXES = ("_0", "_m1")

#: Storage parameters that the segment driver takes as its own, rather than
#: forwarding, so that its per-segment calls all share one policy.
_STORAGE_PARAMETERS = (
    "storage",
    "storage_mode",
    "storage_path",
    "storage_compression",
)


def check_checkpoint_segments(
    propagator: Callable[..., Any],
    spec: "PropagatorSpec",
    arguments: "abc.Mapping[str, Any]",
    n_segments: Any,
) -> tuple[list[str], list[str], list[str]]:
    """Validates a ``checkpoint_segments=`` call, and reads its argument roles.

    Split out from the delegation so that :func:`deepwave.plan`, which resolves
    the *unsegmented* call in order to report the whole run, still raises
    everything a segmented call would raise about its arguments. Otherwise a
    plan would quietly accept a combination the run refuses, which is the one
    thing a pre-flight must not do.

    The three roles are read off the propagator's own signature and spec
    rather than tabulated nine times: the required models are the leading
    positional parameters, the optional models are the parameters the spec
    names as user models over any dimensionality it supports, and the
    source-amplitude mapping is the parameters spelled ``source_amplitudes*``.

    Args:
        propagator: The propagator function.
        spec: Its spec.
        arguments: Its bound arguments, without ``checkpoint_segments``.
        n_segments: The requested number of segments.

    Returns:
        ``(required_models, optional_models, source_names)``, each a list of
        parameter names.

    Raises:
        DeepwaveTypeError: If ``n_segments`` is a bool.
        DeepwaveValueError: If it is not a positive int, or if the call uses an
            argument the segment driver cannot express.

    """
    from deepwave._common import _propagator_user_model_names

    if isinstance(n_segments, bool):
        raise DeepwaveTypeError(
            "checkpoint_segments must be an int (a number of time segments), "
            f"not a bool, but got {n_segments!r}."
        )
    if not isinstance(n_segments, int) or n_segments < 1:
        raise DeepwaveValueError(
            "checkpoint_segments must be a positive int (a number of time "
            f"segments), but got {n_segments!r}."
        )

    parameters = list(inspect.signature(propagator).parameters)
    required_models = parameters[: parameters.index("grid_spacing")]
    model_names = {
        name
        for ndim in spec.ndims
        for name in _propagator_user_model_names(spec.name, ndim)
    }
    optional_models = [
        name
        for name in parameters
        if name in model_names and name not in required_models
    ]
    source_names = [name for name in parameters if name.startswith("source_amplitudes")]

    def reject(name: str, why: str, instead: str) -> None:
        raise DeepwaveValueError(
            f"{name}= is not supported with checkpoint_segments=: {why} {instead}"
        )

    if arguments.get("nt") is not None and not any(
        arguments.get(name) is not None for name in source_names
    ):
        reject(
            "nt",
            "with no source amplitudes there is no record for the driver to "
            "divide into segments, and nt would be re-applied to every one "
            "of them.",
            "Pass source amplitudes of the length you want.",
        )
    for name in parameters:
        if name.endswith(_INITIAL_FIELD_SUFFIXES) and arguments.get(name) is not None:
            reject(
                name,
                "an initial field would be re-applied at every segment "
                "boundary, overwriting the state threaded between segments.",
                "Pass initial_state= instead.",
            )
    for name in ("forward_callback", "backward_callback"):
        if arguments.get(name) is not None:
            reject(
                name,
                "each segment restarts the step counter, so the step index "
                "the callback sees would be wrong and the per-segment "
                "histories would not compose.",
                "Drive the segments yourself with "
                "deepwave.checkpointing.run_segments if you need per-step "
                "callbacks.",
            )
    return required_models, optional_models, source_names


def delegate_checkpoint_segments(
    propagator: Callable[..., Any],
    spec: "PropagatorSpec",
    arguments: dict[str, Any],
) -> "PropagatorOutput":
    """Runs a propagator call as ``checkpoint_segments`` checkpointed segments.

    This is what ``checkpoint_segments=k`` on a propagator means: exactly the
    :func:`deepwave.checkpointing.run_segments` call the user would otherwise
    write by hand, built here so that the nine propagators do not each carry
    their own copy of it, and so that the model, source and optional-model
    lists are read off the propagator's own signature and spec rather than
    tabulated nine times.

    Args:
        propagator: The propagator function being called.
        spec: Its spec.
        arguments: Its bound arguments, as captured by ``locals()`` on entry,
            including ``checkpoint_segments`` itself.

    Returns:
        The combined :class:`deepwave.PropagatorOutput` for the whole run --
        the same fields, the same ``info`` and the same ``auxiliary_state`` a
        direct call gives.

    Raises:
        DeepwaveTypeError: If ``checkpoint_segments`` is a bool.
        DeepwaveValueError: If it is not a positive int, or if the call uses an
            argument the segment driver cannot express.

    """
    from deepwave.checkpointing import run_segments

    n_segments = arguments.pop("checkpoint_segments")
    # Before anything is read off the arguments: the survey mappings are a
    # call form the propagator accepts, and the driver below reads the survey
    # from the flat keywords, so they are resolved into those first.
    resolve_survey_mappings(spec, arguments)
    required_models, optional_models, source_names = check_checkpoint_segments(
        propagator, spec, arguments, n_segments
    )
    signature = inspect.signature(propagator)

    model_args = [arguments.pop(name) for name in required_models]
    grid_spacing = arguments.pop("grid_spacing")
    dt = arguments.pop("dt")
    source_amplitudes = {}
    for name in source_names:
        value = arguments.pop(name)
        if value is not None:
            source_amplitudes[name] = value
    # ``nt`` sets the length of the whole record, so it is applied to the
    # amplitudes here, once, rather than forwarded to every segment.
    nt = arguments.pop("nt", None)
    if nt is not None:
        from deepwave._common import fit_source_amplitudes_to_nt

        fitted = fit_source_amplitudes_to_nt(list(source_amplitudes.values()), nt)
        source_amplitudes = dict(zip(source_amplitudes, fitted))
    optional_model_args = {}
    for name in optional_models:
        value = arguments.pop(name)
        if value is not None:
            optional_model_args[name] = value
    storage_arguments = {
        name: arguments.pop(name) for name in _STORAGE_PARAMETERS if name in arguments
    }
    initial_state = arguments.pop("initial_state", None)

    # Everything the caller left alone is dropped rather than forwarded, so
    # that a parameter the segment driver reserves is only ever seen when the
    # caller actually set it. The survey mappings are already resolved to
    # flat keywords above, so what is forwarded here is always the flat
    # form -- the one spelling the driver reads.
    forwarded = {
        name: value
        for name, value in arguments.items()
        if signature.parameters[name].default is not value
    }
    return run_segments(
        propagator,
        model_args,
        grid_spacing,
        dt,
        source_amplitudes,
        n_segments,
        initial_state=initial_state,
        checkpoint=True,
        optional_model_args=optional_model_args or None,
        **storage_arguments,
        **forwarded,
    )


def non_finite_output_message(
    propagator: str,
    name: str,
    n_non_finite: int,
    info: Optional[RunInfo] = None,
) -> str:
    """The wording of the output-finiteness failure, in one place.

    Kept as a function so that the message the propagators raise and the
    message the documentation quotes cannot drift apart.

    Args:
        propagator: The propagator's public name (e.g. ``"scalar"``).
        name: The name of the first output field found to be non-finite.
        n_non_finite: How many of that field's elements are non-finite.
        info: The run metadata, used for the time-step count and step, or
            None to leave those out. Under ``checkpoint_segments`` this is
            the failing *segment's* metadata, which the wording says.

    Returns:
        The message text.

    """
    ran = ""
    if info is not None:
        ran = (
            f" (nt={info.nt}, dt={info.internal_dt * info.step_ratio:g} for the "
            "propagation that produced it, which under checkpoint_segments is "
            "one segment rather than the whole run)"
        )
    return (
        f"deepwave.{propagator} produced non-finite output: {name!r} contains "
        f"{n_non_finite} non-finite values{ran}. This usually means the "
        "simulation diverged. Check the CFL condition "
        "(deepwave.discretisation.cfl_condition), that the model contains no "
        "zero, negative or non-finite values, and that any initial wavefield "
        "and the source amplitudes are finite. Pass check_output_finite=False "
        "to disable this check."
    )


def check_outputs_finite(
    propagator: str,
    names: abc.Sequence[str],
    values: abc.Sequence[Any],
    info: Optional[RunInfo] = None,
) -> None:
    """Raises if any tensor a propagator is about to return is non-finite.

    The output-only divergence check behind every propagator's
    ``check_output_finite`` argument. It costs **one** device
    synchronisation, not one per output: each output tensor is reduced to a
    0-dimensional flag on the device, the flags are stacked, and a single
    ``.item()`` reads the answer back. The per-output second pass that names
    the offender and counts its bad elements runs only when the run has
    already failed, so its cost never lands on a healthy call.

    ``torch.isfinite`` materialises a boolean mask before reducing it, so the
    check transiently allocates one byte per element of the largest output it
    is looking at. That is small beside the output itself (one byte against
    four or eight) and is freed as soon as the reduction is done, but it is
    not nothing on a run that is already at the edge of its memory.

    Only the top-level tensors are checked -- the final wavefields and the
    receiver amplitudes. The ``auxiliary_state`` element is skipped: its keys,
    count and meaning are absorbing-boundary implementation state rather than
    public output (see :class:`deepwave.AuxiliaryState`), so naming one of them
    would report something the caller cannot act on, and a diverging absorbing
    boundary feeds the physical wavefield that *is* checked, so a continuation
    from such a state fails on the next call.

    The check is also skipped for outputs carrying a functorch wrapper
    (``vmap``, ``jvp``, ``torch.func.grad``): reading a value back from a
    batched tensor is not permitted inside those transforms.

    Args:
        propagator: The propagator's public name (e.g. ``"scalar"``).
        names: The public output field names.
        values: The matching output values, in the same order. Entries that
            are not tensors are ignored.
        info: The run metadata, quoted in the message.

    Raises:
        deepwave.DeepwaveFloatingPointError: If any checked output contains a
            ``NaN`` or an ``Inf``.

    """
    fields = [
        (name, value)
        for name, value in zip(names, values)
        if isinstance(value, torch.Tensor)
    ]
    if not fields or any(
        _functorch.is_functorch_wrapped_tensor(value) for _, value in fields
    ):
        return
    flags = torch.stack([torch.isfinite(value).all() for _, value in fields])
    if bool(flags.all().item()):
        return
    for name, value in fields:
        finite = torch.isfinite(value)
        if bool(finite.all().item()):
            continue
        raise DeepwaveFloatingPointError(
            non_finite_output_message(
                propagator, name, int((~finite).sum().item()), info
            )
        )
    raise AssertionError(  # the stacked flags said otherwise
        f"deepwave.{propagator}'s output finiteness check disagreed with "
        "itself between its two passes."
    )


def link_empty_outputs(
    outputs: abc.Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Gives an empty output the autograd connection the others carry.

    An unrequested receiver channel is returned as an empty Tensor rather
    than omitted, so that the positional return and the named view have the
    same fields on every call. How that Tensor is *built*, though,
    used to differ between the backends: the compiled backend allocates it
    inside its ``torch.autograd.Function``, which marks every returned
    Tensor differentiable, while an eager core allocates it with
    ``torch.empty`` in plain Python and then never writes to it, leaving a
    detached leaf. ``acoustic_vti`` managed to disagree with *itself*, its
    synthetic pressure channel and its cut-back stress channels going
    through ``Tensor.new_empty``, which does not inherit the flag, on both
    backends.

    So ``out.receivers["y"].sum().backward()`` on an unrequested channel
    raised ``RuntimeError: element 0 of tensors does not require grad``
    under ``backend="eager"`` and quietly succeeded under
    ``backend="compiled"`` -- output autograd behaviour that depended on the
    backend, which the interface forbids. The working behaviour is the one
    to keep: an empty Tensor contributes nothing to any gradient, so
    backward through it is a no-op, and a spurious error on an output the
    caller did not ask for is not something to preserve.

    Attaching the empties to the graph here, on the one funnel every
    propagator returns through, makes that behaviour the behaviour
    everywhere: an empty output requires grad exactly when the call's
    non-empty outputs do, and backward through it reaches the same inputs
    with a zero gradient, on every backend and every propagator. When
    nothing in the call requires grad (including under ``torch.no_grad``),
    there is nothing to attach to and the outputs are returned unchanged.

    Args:
        outputs: A propagator's flat positional outputs.

    Returns:
        The same outputs, with any detached empty ones reattached.

    """
    outputs = tuple(outputs)
    empty = [
        i
        for i, t in enumerate(outputs)
        if t.numel() == 0 and not t.requires_grad and t.is_floating_point()
    ]
    if not empty:
        return outputs
    anchors = [t for t in outputs if t.requires_grad and t.ndim > 0]
    if not anchors:
        return outputs
    # A zero-size slice of each anchor: a view, so this costs no copy and no
    # kernel over the anchor's data, and summing them makes backward from an
    # empty output reach everything any output depends on -- which is what
    # the compiled backend's Function does, since its backward returns a
    # gradient for every input.
    link = anchors[0].narrow(0, 0, 0).reshape(-1)
    for anchor in anchors[1:]:
        link = link + anchor.narrow(0, 0, 0).reshape(-1).to(link.dtype)
    linked = list(outputs)
    for i in empty:
        linked[i] = outputs[i] + link.to(outputs[i].dtype)
    return tuple(linked)


def shape_empty_receivers(
    outputs: abc.Sequence[torch.Tensor],
    names: abc.Sequence[str],
    info: Optional[RunInfo],
) -> tuple[torch.Tensor, ...]:
    """Gives every receiver output the same rank, requested or not.

    A receiver channel the call did not ask for records nothing, but it is
    still returned -- an omitted channel would be indistinguishable from a
    misspelled one. What it returns is an empty Tensor, and the useful empty
    Tensor is one shaped like the data: ``[n_shots, 0, nt]``, with the same
    number of shots and the same number of time samples as a channel that
    *was* requested. That way
    ``torch.cat(list(out.receivers.values()), dim=1)`` builds a multichannel
    gather whether or not every channel was asked for, and code that indexes
    ``amplitudes.shape[-1]`` does not have to special-case the empty case.

    ``n_shots`` and ``nt`` are read from a requested sibling channel when
    there is one, and otherwise from the returned state and ``info``.

    Args:
        outputs: A propagator's flat positional outputs, receivers last.
        names: The public output names, in return order.
        info: The run metadata, used for ``nt`` when no channel was
            requested.

    Returns:
        The same outputs, with every empty receiver output reshaped.

    """
    outputs = tuple(outputs)
    n_receivers = sum(1 for name in names if name.startswith(_RECEIVER_PREFIXES))
    if n_receivers == 0:
        return outputs
    start = len(outputs) - n_receivers
    receivers = outputs[start:]
    if all(amplitudes.dim() == 3 for amplitudes in receivers):
        return outputs
    n_shots = 0
    nt = 0
    for amplitudes in receivers:
        if amplitudes.dim() == 3:
            n_shots = amplitudes.shape[0]
            nt = amplitudes.shape[2]
            break
    else:
        for tensor in outputs[:start]:
            if tensor.dim() > 1:
                n_shots = tensor.shape[0]
                break
        nt = 0 if info is None else info.nt
    shaped = list(outputs)
    for index in range(start, len(outputs)):
        if outputs[index].dim() != 3:
            shaped[index] = outputs[index].reshape(n_shots, 0, nt)
    return tuple(shaped)


def named_output(
    spec: "PropagatorSpec",
    ndim: int,
    outputs: abc.Sequence[torch.Tensor],
    info: Optional[RunInfo] = None,
    *,
    accuracy: Optional[int] = None,
    boundary: Optional[abc.Sequence[str]] = None,
    n_mechanisms: int = 1,
    mpml: bool = False,
    scheme: str = "cfs-cpml",
    names: Optional[abc.Sequence[str]] = None,
    state_names: Optional[abc.Sequence[str]] = None,
    fold_names: Optional[abc.Sequence[str]] = None,
    check_output_finite: bool = True,
) -> PropagatorOutput:
    """Wraps a propagator's positional outputs in a :class:`PropagatorOutput`.

    ``outputs`` is the flat list of state tensors in *public* order (see
    ``PropagatorSpec.public_wavefield_names``) followed by the receiver
    amplitudes. The names come from ``spec.output_names(ndim)``, so the named
    view tracks the positional return automatically. Any state fields the spec
    declares as auxiliary state are collected into a single
    :class:`AuxiliaryState` occupying the position of the first of them.

    ``info`` is attached as run metadata. The three fields the engine can fill
    itself -- ``ndim``, ``wavefield_names`` and ``receiver_names`` -- are
    filled here when the propagator left them at their defaults, so that every
    propagator gets them without repeating the derivation.

    Args:
        spec: The calling propagator's specification.
        ndim: The number of spatial dimensions.
        outputs: The flat positional outputs, before the collapse.
        info: The run metadata (see :class:`RunInfo`).
        accuracy: The finite-difference order of accuracy this call resolved,
            recorded in the auxiliary state's compatibility fingerprint.
        boundary: The resolved per-face boundary conditions (see
            :func:`boundary_condition_names`), likewise recorded.
        n_mechanisms: The number of relaxation mechanisms, likewise recorded.
        mpml: Whether multiaxial PML cross-damping was used, likewise recorded.
        scheme: Names the absorbing-boundary implementation, likewise recorded.
        names: The public output names, overriding ``spec.output_names(ndim)``.
            Only the attenuating propagators pass this: their field count
            depends on ``q_n_mechanisms``, which the spec's
            ``Callable[[int], list[str]]`` name functions cannot see.
        state_names: The public state names, overriding
            ``spec.public_state_names(ndim)``, for the same reason. Pass both
            or neither.
        fold_names: The state names carried opaquely in the auxiliary state,
            overriding ``spec.auxiliary_state_names(ndim)``, for the same
            reason: the attenuating propagators fold their GSLS memory
            variables, and how many of those there are depends on
            ``q_n_mechanisms``.
        check_output_finite: Whether to check that the returned wavefields and
            receiver amplitudes contain no ``NaN`` or ``Inf`` (see
            :func:`check_outputs_finite`). This is the single funnel every
            propagator returns through, so the check lives here rather than in
            each of them.

    Raises:
        deepwave.DeepwaveFloatingPointError: If ``check_output_finite`` and any
            returned wavefield or receiver-amplitude tensor is non-finite.

    """
    if (names is None) != (state_names is None):
        raise AssertionError(
            "named_output's names and state_names overrides must be given "
            "together (one describes the prefix of the other)."
        )
    names, n_public_state = resolved_output_names(
        spec, ndim, names, state_names, fold_names
    )
    # Before anything reads them: an unrequested (empty) output must carry
    # the same autograd connection on every backend (see
    # :func:`link_empty_outputs`).
    outputs = link_empty_outputs(outputs)
    # ... and then every receiver output carries the data's rank, whether or
    # not it was requested (see :func:`shape_empty_receivers`).
    outputs = shape_empty_receivers(outputs, names, info)
    state_names = list(
        spec.public_state_names(ndim) if state_names is None else state_names
    )
    fold = spec.auxiliary_state_names(ndim) if fold_names is None else list(fold_names)
    if not fold:
        if len(names) != len(outputs):
            raise AssertionError(
                f"{spec.name} returned {len(outputs)} outputs but output_names "
                f"gave {len(names)} names for ndim={ndim}."
            )
        if check_output_finite:
            check_outputs_finite(spec.name, names, outputs, info)
        return PropagatorOutput(
            names,
            outputs,
            info=complete_run_info(info, ndim, names, n_public_state),
        )
    if len(names) != len(outputs) - len(fold) + 1:
        raise AssertionError(
            f"{spec.name} returned {len(outputs)} outputs but output_names "
            f"gave {len(names)} names for ndim={ndim} with {len(fold)} of them "
            "folded into the auxiliary state."
        )
    n_state = len(state_names)

    def make_state(tensors: dict[str, torch.Tensor]) -> AuxiliaryState:
        first = next(iter(tensors.values()), None)
        return AuxiliaryState(
            tensors,
            propagator=spec.name,
            ndim=ndim,
            accuracy=accuracy,
            pml_width=None if info is None else tuple(info.pml_width),
            boundary=None if boundary is None else tuple(boundary),
            n_mechanisms=n_mechanisms,
            mpml=mpml,
            mpml_fraction=None if info is None else info.mpml_fraction,
            grid_spacing=None if info is None else info.grid_spacing,
            surface_method=None if info is None else info.surface_method,
            q_freq=None if info is None else info.q_freq,
            q_band=None if info is None else info.q_band,
            fit_convention=(
                "gsls-nnls-debye-v1" if n_mechanisms > 1 else "single-sls-exact-v1"
            )
            if spec.name == "viscoacoustic"
            else None,
            shape=None if first is None else tuple(first.shape),
            dtype=None if first is None else first.dtype,
            device=None if first is None else first.device,
            origin=None if info is None else tuple(info.origin),
            pml_freq=None if info is None else info.pml_freq,
            scheme=scheme,
            max_vel=None if info is None else info.max_vel,
            internal_dt=None if info is None else info.internal_dt,
        )

    values = collapse_auxiliary_state_values(
        state_names, outputs[:n_state], fold, make_state
    )
    values.extend(outputs[n_state:])
    if check_output_finite:
        check_outputs_finite(spec.name, names, values, info)
    return PropagatorOutput(
        names, values, info=complete_run_info(info, ndim, names, n_public_state)
    )


def resolved_output_names(
    spec: "PropagatorSpec",
    ndim: int,
    names: Optional[abc.Sequence[str]] = None,
    state_names: Optional[abc.Sequence[str]] = None,
    fold_names: Optional[abc.Sequence[str]] = None,
) -> tuple[list[str], int]:
    """The public field names of a call, and how many of them are state.

    Together these are what :func:`complete_run_info` needs to fill
    :attr:`deepwave.RunInfo.wavefield_names` and
    :attr:`deepwave.RunInfo.receiver_names`. Shared by :func:`named_output`,
    which fills them when a run returns, and by the pre-flight resolution,
    which must produce the same ``RunInfo`` without a run to name the outputs
    of -- so the two cannot answer differently.

    Args:
        spec: The propagator's spec.
        ndim: The number of spatial dimensions.
        names: The propagator's override of the public field names, when its
            output shape depends on something the spec cannot see (the number
            of relaxation mechanisms), or None for the spec's own list.
        state_names: The matching override of the state field names, or None.
        fold_names: The matching override of the names carried opaquely in
            the auxiliary state, or None for the spec's own list.

    Returns:
        ``(names, n_public_state)``: the public field names in return order,
        and how many of them, from the front, are state rather than receiver
        amplitudes.

    """
    if (names is None) != (state_names is None):
        raise AssertionError(
            "the output-name overrides must be given together (one describes "
            "the prefix of the other)."
        )
    resolved = list(spec.output_names(ndim) if names is None else names)
    resolved_state = list(
        spec.public_state_names(ndim) if state_names is None else state_names
    )
    fold = spec.auxiliary_state_names(ndim) if fold_names is None else list(fold_names)
    if not fold:
        return resolved, len(resolved_state)
    return resolved, len(collapse_auxiliary_state_names(resolved_state, fold))


def complete_run_info(
    info: Optional[RunInfo],
    ndim: int,
    output_names: abc.Sequence[str],
    n_state_fields: int,
) -> Optional[RunInfo]:
    """Fills the ``RunInfo`` fields the engine itself can determine.

    ``output_names`` is the propagator's public field list *in return order*
    and ``n_state_fields`` is how many of its leading entries are state, the
    rest being receiver amplitudes. Slicing that list, rather than reusing the
    backend-order state names, is what makes the recorded names agree with the
    return order on the propagators whose public order permutes the backend's
    (the elastic family) -- and therefore what makes
    :attr:`RunInfo.wavefield_names` and :attr:`PropagatorOutput.final_state`
    the promised "in return order".

    A propagator that builds its :class:`PropagatorOutput` directly, rather
    than through :func:`named_output`, must call this itself; the attenuating
    propagators do, because their field count depends on the number of
    relaxation mechanisms, which a :class:`PropagatorSpec` cannot see.
    """
    if info is None:
        return None
    updates: dict[str, Any] = {}
    if not info.ndim:
        updates["ndim"] = ndim
    if not info.wavefield_names:
        updates["wavefield_names"] = tuple(output_names[:n_state_fields])
    if not info.receiver_names:
        updates["receiver_names"] = tuple(output_names[n_state_fields:])
    return replace(info, **updates) if updates else info


def receiver_output_names(spec: "PropagatorSpec") -> abc.Set[str]:
    """The propagator's output names that are receiver amplitudes.

    These are the public outputs that are *not* re-feedable state (so they are
    ignored when an entire :class:`PropagatorOutput` is supplied as an
    ``initial_state``). Computed from the spec at the maximum ndim so the full
    set of receiver names is covered.

    ``"auxiliary_state"`` is state, not a receiver output, but it is not one of
    ``wavefield_names`` either (those are the internal flat names the fold
    replaces), so it has to be subtracted explicitly. Leaving it in would make
    ``initial_state=out`` silently *ignore* the entire auxiliary state -- wrong
    numbers with no exception -- rather than restore it.
    """
    return (
        set(spec.output_names(3))
        - set(spec.wavefield_names(3))
        - set(spec.public_state_names(3))
        - {"auxiliary_state"}
    )


def flat_state(
    out: PropagatorOutput, spec: "PropagatorSpec", ndim: int, n_mechanisms: int = 1
) -> dict[str, torch.Tensor]:
    """A propagator output's state as a flat ``{internal name: tensor}`` dict.

    The inverse of the auxiliary-state fold, for internal consumers that thread
    state as individual tensors (:mod:`deepwave.checkpointing` must, because
    ``torch.utils.checkpoint`` only tracks tensor arguments). The keys are
    ``spec.wavefield_names(ndim)``, whatever the public field names are: a
    public field whose spelling differs from the internal one (see
    :meth:`PropagatorSpec.public_state_name_map`) is looked up under its
    public name and returned under its internal one.
    """
    values = out.as_dict()
    state = values.pop("auxiliary_state", None)
    if state is not None:
        values.update(state.as_dict())
    public = spec.public_state_name_map(ndim, n_mechanisms)
    return {name: values[public[name]] for name in public}


def auxiliary_state_of(initial_state: Optional[Any]) -> Optional[AuxiliaryState]:
    """The :class:`AuxiliaryState` an ``initial_state`` argument carries, if any.

    Read before the call's own parameters are resolved (which is where
    ``initial_state`` is merged) and checked afterwards with
    :func:`check_auxiliary_state`, once there is something to check it against.
    """
    if isinstance(initial_state, PropagatorOutput):
        try:
            return initial_state.auxiliary_state
        except AttributeError:
            return None
    if isinstance(initial_state, abc.Mapping):
        candidate = initial_state.get("auxiliary_state")
        if isinstance(candidate, AuxiliaryState):
            return candidate
    return None


def inherited_pml_freq(
    pml_freq: Optional[float],
    auxiliary_state: Optional[AuxiliaryState],
) -> Optional[float]:
    """The ``pml_freq`` a continuation uses when the caller omitted it.

    A continuation typically has no sources of its own -- the wavelet was
    injected by the first call -- so inferring ``pml_freq`` from this call's
    source amplitudes would fall back to a value unrelated to the one the
    state's memory variables were built with, and the call would both report
    an inference the user did not ask for and then complain that it does not
    match the state. The state records the frequency the profiles were built
    with, so an omitted ``pml_freq`` takes it: the continuation resolves what
    the first call resolved, which is the only value that continues the same
    simulation.

    An explicitly given ``pml_freq`` is returned unchanged, mismatch warning
    and all: the caller has said what they want.

    Args:
        pml_freq: The caller's ``pml_freq`` argument.
        auxiliary_state: The auxiliary state the call's ``initial_state``
            carries, if any (see :func:`auxiliary_state_of`).

    Returns:
        The ``pml_freq`` to resolve the call with, still ``None`` when there
        is nothing to inherit.

    """
    if pml_freq is not None or auxiliary_state is None:
        return pml_freq
    return auxiliary_state.pml_freq


def inherited_max_vel(
    max_vel: Optional[float],
    auxiliary_state: Optional[AuxiliaryState],
) -> Optional[float]:
    """The ``max_vel`` a continuation uses when the caller omitted it.

    ``max_vel`` sets both the absorbing boundary's profiles and, through the
    CFL condition, the internal time step. The first call's value is recorded
    in the state; a continuation that omits the argument would instead take
    the model's own maximum, which is a different number whenever the first
    call gave one -- giving different profiles from the ones the state's
    memory variables were built with, and, through a different internal time
    step, a state the call then refuses outright. So an omitted ``max_vel``
    takes the recorded one.

    Inheriting a recorded value cannot quietly make a run unstable: if the
    model handed to the continuation is faster than the value inherited,
    :func:`deepwave._common.set_max_vel` says so, in the same words it uses
    for a ``max_vel`` passed by hand.

    An explicitly given ``max_vel`` is returned unchanged.

    Args:
        max_vel: The caller's ``max_vel`` argument.
        auxiliary_state: The auxiliary state the call's ``initial_state``
            carries, if any (see :func:`auxiliary_state_of`).

    Returns:
        The ``max_vel`` to resolve the call with, still ``None`` when there
        is nothing to inherit.

    """
    if max_vel is not None or auxiliary_state is None:
        return max_vel
    return auxiliary_state.max_vel


def stored_region_of(
    initial_state: Optional[Any],
) -> tuple[Optional[tuple[int, ...]], Optional[tuple[int, ...]]]:
    """The ``(origin, extent)`` of the model region an ``initial_state`` records.

    Either element is ``None`` when the state does not record it. A
    :class:`PropagatorOutput` carrying a :class:`RunInfo` records both; a bare
    :class:`AuxiliaryState` (or a mapping holding one) records only the origin.
    ``RunInfo`` is preferred over the auxiliary state's own record so that this
    reads the *same* field the origin recovery in
    :func:`resolve_initial_state` uses -- a check keyed on a different field
    than the recovery can reject the very region the recovery would have
    chosen.

    Purely descriptive: it validates nothing and raises nothing, because it is
    read before ``initial_state`` has been validated.
    """
    origin: Optional[tuple[int, ...]] = None
    extent: Optional[tuple[int, ...]] = None
    if isinstance(initial_state, PropagatorOutput):
        info = initial_state.info
        if info is not None:
            origin = tuple(info.origin)
            extent = tuple(info.extent)
    if origin is None:
        state = auxiliary_state_of(initial_state)
        if state is not None and state.origin is not None:
            origin = tuple(state.origin)
    return origin, extent


def _channel_interpolates(
    mode: str,
    location: Any,
    offset: Optional[list[float]],
) -> Optional[bool]:
    """Whether one channel is interpolated, or ``None`` if it has no locations.

    A channel holds no locations when it is ``None``, an empty Tensor, or a
    Tensor whose every point is the :data:`deepwave.IGNORE_LOCATION` sentinel:
    all three are ways of saying "nothing here", so all three mean the channel
    has taken neither path. A :class:`deepwave.Hicks` is an explicit request
    for interpolation, so it always counts as interpolated.
    """
    from deepwave import _common
    from deepwave import _location_interpolation as _li

    if isinstance(location, _li.Hicks):
        return True
    if not isinstance(location, torch.Tensor) or location.numel() == 0:
        return None
    if bool(torch.all(location == _common.IGNORE_LOCATION)):
        return None
    return _li._should_interpolate(location, mode, offset=offset)  # noqa: SLF001


def resolved_location_interpolation(
    mode: str,
    location_groups: abc.Sequence[abc.Sequence[Any]],
    offset_groups: Optional[
        abc.Sequence[Optional[abc.Sequence[Optional[list[float]]]]]
    ] = None,
) -> str:
    """Which location interpolation a call's coordinates actually resolve to.

    ``"auto"`` -- the default -- means "interpolate only where it is needed",
    so what it resolved to is a property of the coordinates, not of the
    argument. This reports the answer for the call as a whole:

    * ``"none"`` -- nothing was interpolated; every channel that has locations
      took the exact on-node path;
    * ``"hicks"`` -- something was interpolated, and within each kind of
      location (the sources, the receivers, ...) every channel that has
      locations took the same path;
    * ``"mixed"`` -- one kind of location was split across the two paths: some
      of its channels were interpolated and others were not.

    ``"none"`` and ``"hicks"`` are deliberately the same words the
    ``location_interpolation`` argument uses, so that the answer is spelled
    the way the request would be. ``"mixed"`` has no argument counterpart
    because it is not a mode you can ask for: it is what per-component
    routing produces on a multi-channel propagator whose channels differ,
    and reporting it as plain ``"hicks"`` would tell a reader that
    coordinates were band-limited when some of them were not.

    Sources are compared with sources and receivers with receivers, never with
    each other, so the ordinary single-channel arrangement of an off-node
    source and on-node receivers is ``"hicks"`` and not ``"mixed"``: there is
    no per-component routing there to report. Only a propagator with several
    channels of one kind can be ``"mixed"``.

    Channels with no locations are ignored: a propagator with nothing on a
    channel has taken neither path there. ``None``, an empty Tensor and a
    Tensor of nothing but :data:`deepwave.IGNORE_LOCATION` all count as
    nothing.

    Recorded in :attr:`RunInfo.location_interpolation_used`.

    Args:
        mode: The ``location_interpolation`` argument.
        location_groups: The propagator's location arguments, grouped by kind
            (one group for the sources, one for each set of receivers), before
            they are resolved to integer grid indices. Each entry of a group is
            one channel: a Tensor, a :class:`deepwave.Hicks`, or ``None``.
        offset_groups: The half-cell component offsets, parallel to
            ``location_groups``. ``None``, either for the whole argument or for
            one group, means those channels are on the reference node, which is
            the case for the regular-grid propagators.

    """
    if offset_groups is None:
        offset_groups = [None] * len(location_groups)
    interpolated = 0
    mixed = False
    for locations, offsets in zip(location_groups, offset_groups):
        group_offsets: abc.Sequence[Optional[list[float]]] = (
            [None] * len(locations) if offsets is None else offsets
        )
        states = [
            state
            for state in (
                _channel_interpolates(mode, location, offset)
                for location, offset in zip(locations, group_offsets)
            )
            if state is not None
        ]
        if states and not all(state == states[0] for state in states):
            mixed = True
        interpolated += sum(states)
    if mixed:
        return "mixed"
    if interpolated == 0:
        return "none"
    return "hicks"


def resolved_state_shape(
    wavefields: Optional[abc.Sequence[Optional[torch.Tensor]]],
) -> Optional[tuple[int, ...]]:
    """The shape a call's prepared state tensors resolved to, if it is known.

    ``wavefields`` is a propagator's list of state tensors as
    :func:`deepwave._common.prepare_wavefields` returns it: every supplied
    wavefield padded to the simulation domain, and a zero-element placeholder
    (``torch.empty(0)``, one-dimensional) wherever nothing was supplied. Every
    real entry has the same shape, so the first one describes them all.
    ``None`` when there is no real entry to read, which leaves the recorded
    shape unchecked rather than guessed at.
    """
    if wavefields is None:
        return None
    for wavefield in wavefields:
        if wavefield is not None and wavefield.dim() > 1:
            return tuple(wavefield.shape)
    return None


def check_auxiliary_state(
    auxiliary_state: Optional[AuxiliaryState],
    spec: "PropagatorSpec",
    ndim: int,
    accuracy: int,
    pml_width: abc.Sequence[int],
    boundary: abc.Sequence[str],
    pml_freq: float,
    n_mechanisms: int = 1,
    mpml: bool = False,
    scheme: str = "cfs-cpml",
    *,
    inputs_require_grad: bool,
    wavefields: Optional[abc.Sequence[Optional[torch.Tensor]]] = None,
    max_vel: Optional[float] = None,
    internal_dt: Optional[float] = None,
    grid_spacing: Optional[Union[float, abc.Sequence[float]]] = None,
    mpml_fraction: Optional[float] = None,
    surface_method: Optional[str] = None,
    q_freq: Optional[float] = None,
    q_band: Optional[tuple[float, float]] = None,
    fold_names: Optional[abc.Sequence[str]] = None,
) -> None:
    """Rejects an auxiliary state that this call cannot reuse.

    Raises :class:`deepwave.DeepwaveValueError` when any of the recorded
    propagator, ndim, accuracy, pml_width, boundary, number of relaxation
    mechanisms, mpml flag, scheme, simulation-domain shape or internal time
    step or other recorded numerical configuration differs from this call's
    resolved value. An unidentifiable manually constructed state can only be
    checked for the fields it records.

    ``dtype`` and ``device`` are deliberately not checked here:
    :func:`deepwave._common.prepare_wavefields` already raises specific,
    better-worded errors for both, and checking them here would break the
    legitimate device round trip of
    :func:`deepwave.checkpointing.run_segments`'s ``checkpoint_device``.

    The *shape* is checked, from ``wavefields`` -- this call's prepared
    (padded) state tensors, whose spatial shape is the simulation domain it
    resolved. A state returned by a run knows the domain it belongs to, and
    :func:`deepwave._common.get_survey_extents_from_wavefields` will otherwise
    fall back to reading it as an unpadded model-region wavefield when the
    padded reading no longer contains every source and receiver -- silently
    continuing at a different extent, with every field displaced by the PML
    width. ``prepare_wavefields`` cannot catch that, because by then the
    extents have already been chosen to make the shape fit.

    ``inputs_require_grad`` says whether any model, source-amplitude or
    initial-wavefield tensor of *this* call requires grad; the caller computes
    it from the tensors it already holds (:func:`any_requires_grad`). It is one
    of the four conditions of :func:`warn_if_gradient_history_lost`.
    """
    if auxiliary_state is None:
        return
    # All or nothing. An auxiliary state is documented as one opaque bundle, not
    # a set of individually optional fields, so a bundle holding only some of
    # them is a mistake -- and one that would otherwise be silent, because the
    # absent memory variables would simply start from zero. An empty bundle
    # stays legal: it says "no auxiliary state", which is what
    # warn_if_gradient_history_lost already special-cases. Checked here rather
    # than in resolve_initial_state because the expected field list depends on
    # ndim, which is not resolved until later. ``fold_names`` overrides the
    # spec's list for the attenuating propagators, whose fold set depends on
    # ``q_n_mechanisms`` (see :func:`named_output`).
    expected_names = (
        spec.auxiliary_state_names(ndim) if fold_names is None else list(fold_names)
    )
    if len(auxiliary_state) and expected_names:
        missing = [name for name in expected_names if name not in auxiliary_state]
        if missing:
            raise DeepwaveValueError(
                "initial_state's auxiliary_state is missing "
                f"{', '.join(repr(name) for name in missing)}. A boundary "
                "state is one opaque bundle and must be passed back whole "
                f"(this call's has {len(expected_names)} fields; the one "
                f"supplied has {len(auxiliary_state)}). Pass initial_state=out, "
                "or out.auxiliary_state unchanged; leave auxiliary_state out "
                "of initial_state altogether to start its fields from rest."
            )
    state_shape = resolved_state_shape(wavefields)
    auxiliary_state._check_compatible(  # noqa: SLF001 - same-module private API
        AuxiliaryState(
            {},
            propagator=spec.name,
            ndim=ndim,
            accuracy=accuracy,
            pml_width=tuple(pml_width),
            boundary=tuple(boundary),
            n_mechanisms=n_mechanisms,
            mpml=mpml,
            mpml_fraction=mpml_fraction,
            grid_spacing=(
                None
                if grid_spacing is None
                else tuple(float(x) for x in grid_spacing)
                if isinstance(grid_spacing, abc.Sequence)
                else (float(grid_spacing),) * ndim
            ),
            surface_method=surface_method,
            q_freq=q_freq,
            q_band=q_band,
            fit_convention=(
                "gsls-nnls-debye-v1" if n_mechanisms > 1 else "single-sls-exact-v1"
            )
            if spec.name == "viscoacoustic"
            else None,
            shape=state_shape,
            field_names=tuple(expected_names),
            field_shapes=(
                tuple(state_shape for _ in expected_names)
                if state_shape is not None
                else None
            ),
            origin=None,
            pml_freq=pml_freq,
            scheme=scheme,
            max_vel=max_vel,
            internal_dt=internal_dt,
        ),
        stacklevel=4,
    )
    warn_if_gradient_history_lost(auxiliary_state, inputs_require_grad, stacklevel=3)


def any_requires_grad(*groups: Any) -> bool:
    """Whether any tensor in ``groups`` requires grad.

    Each argument may be a :class:`torch.Tensor`, ``None``, or an iterable of
    either (a propagator's model list, its per-component source-amplitude
    list, or its prepared initial wavefields). Written to accept what the
    propagators already hold at the point where they check their boundary
    state, so that no propagator has to assemble a tensor list for the sake
    of the check.
    """
    for group in groups:
        if group is None:
            continue
        if isinstance(group, torch.Tensor):
            if group.requires_grad:
                return True
            continue
        for tensor in group:
            if tensor is not None and tensor.requires_grad:
                return True
    return False


def warn_if_gradient_history_lost(
    auxiliary_state: Optional[AuxiliaryState],
    inputs_require_grad: bool,
    stacklevel: int = 3,
) -> None:
    """Warns when an auxiliary state arrived detached but ought not to have.

    ``torch.utils.checkpoint(..., use_reentrant=True)`` runs the wrapped
    function under ``torch.no_grad()`` and re-attaches only the top-level
    :class:`torch.Tensor` elements of its return, so the tensors held inside
    a returned ``auxiliary_state`` come back detached. Threading such a state
    into the next call severs the gradient across the join, and nothing
    raises: the gradient is simply computed as though the earlier run had not
    happened.

    The warning fires only when all four of these hold, so that legitimate
    code is not nagged:

    1. the state has a recorded fingerprint, so it came from a real run
       rather than being assembled by hand;
    2. *every* one of its tensors is detached;
    3. gradients are enabled at this call;
    4. some model, source-amplitude or initial-wavefield input to this call
       requires grad. The initial wavefields are included because inverting
       for one of them -- a photoacoustic initial-pressure reconstruction,
       say -- with no model or source parameter under inversion is precisely
       a case where the severed gradient matters and nothing else would
       report it.

    A state marked by :meth:`AuxiliaryState.detach` is exempt: the detachment
    was asked for.
    """
    if auxiliary_state is None or auxiliary_state.detached_deliberately:
        return
    if auxiliary_state.propagator is None or not len(auxiliary_state):
        return
    if not inputs_require_grad or not torch.is_grad_enabled():
        return
    if any(
        tensor.grad_fn is not None or tensor.requires_grad
        for tensor in auxiliary_state.values()
    ):
        return
    _warn_once(
        f"auxiliary_state_detached:{auxiliary_state.propagator}",
        "This auxiliary_state's tensors carry no gradient history, so the "
        "gradient of this call will not reach the run that produced it. A "
        "propagator wrapped in `torch.utils.checkpoint(..., "
        "use_reentrant=True)` returns its auxiliary state detached, because "
        "the reentrant implementation re-attaches only the tensors at the "
        "top level of the return. Pass `use_reentrant=False`, or, if the "
        "detachment is intended (truncated backpropagation), pass "
        "`initial_state=out.detach()`.",
        UserWarning,
        stacklevel,
    )


class ContinuationOriginError(DeepwaveValueError):
    """A continuation was asked for a different region than its state covers.

    A :class:`deepwave.DeepwaveValueError`, so it is caught and reported like
    any other bad argument; the distinct type exists only so that
    :func:`deepwave.checkpointing.run_segments`, which manufactures the
    ``initial_state`` its later segments are given, can recognise this one
    error and restate it in terms the caller can act on (it never passed that
    state, so advice about it is unactionable). Not exported.

    The three arguments the restatement needs are carried on the exception
    rather than parsed back out of its message.
    """

    def __init__(
        self,
        message: str,
        *,
        selector: str,
        selected_origin: abc.Sequence[int],
        stored_origin: abc.Sequence[int],
        stored_extent: Optional[abc.Sequence[int]],
    ) -> None:
        super().__init__(message)
        #: ``"origin"`` or ``"survey_pad"``: which argument chose the region.
        self.selector = selector
        #: The origin that argument selected for this call.
        self.selected_origin = tuple(selected_origin)
        #: The origin the state being continued was produced at.
        self.stored_origin = tuple(stored_origin)
        #: That state's extent, when it records one.
        self.stored_extent = None if stored_extent is None else tuple(stored_extent)


def _stored_region_text(
    stored_origin: abc.Sequence[int],
    stored_extent: Optional[abc.Sequence[int]],
) -> str:
    """``"origin [5, 5], extent [11, 21]"``, or just the origin if unrecorded."""
    extent = f", extent {list(stored_extent)}" if stored_extent is not None else ""
    return f"origin {list(stored_origin)}{extent}"


def _initial_state_supplies_a_field(initial_state: Optional[Any]) -> bool:
    """Whether ``initial_state`` actually carries a field, so describes a region.

    An empty or all-``None`` state describes no region, so there is nothing
    for a region check to disagree with. Descriptive only: it validates
    nothing, because it runs before ``initial_state`` has been validated.
    """
    items: dict[str, Any]
    if isinstance(initial_state, PropagatorOutput):
        try:
            items = dict(initial_state.final_state)
        except (AttributeError, DeepwaveRuntimeError):
            return False
    elif isinstance(initial_state, abc.Mapping):
        items = dict(initial_state)
    else:
        return False
    state = items.pop("auxiliary_state", None)
    if isinstance(state, AuxiliaryState) and len(state):
        return True
    return any(value is not None for value in items.values())


def check_continuation_survey_pad_origin(
    initial_state: Optional[Any],
    survey_pad: Optional[Union[int, abc.Sequence[Optional[int]]]],
    survey_origin: abc.Sequence[int],
) -> None:
    """Polices the ``survey_pad`` half of the continuation rule.

    The companion of :func:`_check_continuation_origin`, which covers an
    explicit ``origin=``. ``survey_pad`` cannot be checked there: the region
    it selects is not known until the survey extents have been resolved
    against *this* call's own source and receiver positions, which happens
    after ``initial_state`` has been merged. So it is checked here, from the
    origin the call actually resolved.

    Without it, a continuation that moves the survey -- the same
    ``survey_pad`` around sources that have moved -- selects a
    same-shaped region somewhere else in the model and continues the stored
    wavefields there. The shape check passes (the shapes do agree), nothing
    else looks at the region in model coordinates, and every field is
    silently propagated through the wrong part of the model.

    Args:
        initial_state: The caller's ``initial_state`` argument, if any.
        survey_pad: The caller's ``survey_pad`` argument. ``None`` means this
            call did not select its region that way, so there is nothing for
            this function to check.
        survey_origin: The origin this call resolved, which with
            ``survey_pad`` given is the one ``survey_pad`` selected.

    Raises:
        ContinuationOriginError: If the region ``survey_pad`` selected is not
            the region ``initial_state`` was produced in.
    """
    if survey_pad is None or initial_state is None:
        return
    stored_origin, stored_extent = stored_region_of(initial_state)
    if stored_origin is None:
        return
    if not _initial_state_supplies_a_field(initial_state):
        return
    if tuple(survey_origin) == stored_origin:
        return
    raise ContinuationOriginError(
        f"survey_pad selects the region at origin {list(survey_origin)}, "
        "which disagrees with the region the initial_state was produced in "
        f"({_stored_region_text(stored_origin, stored_extent)}). Continuing "
        "there would propagate the stored wavefields through a different "
        "part of the model. Omit survey_pad to continue in the stored "
        "region, or start a new simulation (leave initial_state out) if the "
        "survey really has moved.",
        selector="survey_pad",
        selected_origin=tuple(survey_origin),
        stored_origin=stored_origin,
        stored_extent=stored_extent,
    )


def _check_continuation_origin(
    *,
    origin: Optional[abc.Sequence[int]],
    stored_origin: Optional[tuple[int, ...]],
    stored_extent: Optional[tuple[int, ...]],
    state_has_fields: bool,
) -> None:
    """Polices the ``origin`` half of the continuation rule.

    Refuses an explicit ``origin`` that disagrees with the region the state
    records. That used to pass silently and continue the simulation over a
    different part of the model than the wavefields describe, which displaces
    every field rather than producing any error at all.

    The ``survey_pad`` half of the same rule is
    :func:`check_continuation_survey_pad_origin`, which cannot run here: the
    origin ``survey_pad`` selects is not known until the survey extents have
    been resolved against this call's own sources and receivers.

    It is keyed strictly on *disagreement*, never on the mere presence of
    both: :func:`deepwave.checkpointing.run_segments` deliberately re-passes
    the stored origin as an explicit ``origin=`` alongside the same state, so
    a check keyed on presence would fire on every segment of every
    checkpointed run.

    The remaining case -- a state that records no region at all, so that
    origin zero has to be assumed -- is already reported by
    :func:`get_survey_extents_from_wavefields`, which says so in terms of the
    extents it actually inferred. This function deliberately does not warn
    about it a second time.

    Args:
        origin: The caller's ``origin`` argument.
        stored_origin: The origin the state records, if it records one.
        stored_extent: The extent the state records, if it records one.
        state_has_fields: Whether the state actually supplied a field (a
            wavefield or an auxiliary-state field; receiver amplitudes are
            outputs, not state, and say nothing about the region). An empty
            or all-``None`` state describes no region, so there is nothing
            to disagree with.

    Raises:
        ContinuationOriginError: If ``origin`` disagrees with
            ``stored_origin``.
    """
    if not state_has_fields:
        return
    if origin is None or stored_origin is None:
        return
    if tuple(origin) != stored_origin:
        raise ContinuationOriginError(
            f"origin={list(origin)} disagrees with the region the "
            "initial_state was produced in "
            f"({_stored_region_text(stored_origin, stored_extent)}). Omit "
            "origin to continue in the stored region, or pass initial "
            f"wavefields that belong to the region origin={list(origin)} "
            "selects.",
            selector="origin",
            selected_origin=tuple(origin),
            stored_origin=stored_origin,
            stored_extent=stored_extent,
        )


def resolve_initial_state(
    initial_state: Optional[Any],
    wavefield_args: dict[str, Optional[torch.Tensor]],
    origin: Optional[abc.Sequence[int]] = None,
    survey_pad: Optional[Union[int, abc.Sequence[Optional[int]]]] = None,
    ignore: abc.Container[str] = (),
    auxiliary_state_names: abc.Collection[str] = (),
    propagator: Optional[str] = None,
) -> tuple[dict[str, Optional[torch.Tensor]], Optional[list[int]]]:
    """Merges an optional ``initial_state`` bundle into explicit wavefield args.

    ``initial_state`` lets a previous propagation's state be fed straight back
    in to continue a simulation, without naming every initial-wavefield
    argument by hand. It may be either:

    - a :class:`PropagatorOutput` (e.g. the return of a previous propagation
      call); its receiver-amplitude fields are ignored (they are outputs, not
      state) so a whole output can be passed directly; or
    - any mapping from public state name to tensor, plus the key
      ``"auxiliary_state"`` holding a :class:`AuxiliaryState` -- which is what
      :attr:`PropagatorOutput.final_state` is.

    Each entry fills the correspondingly-named initial-wavefield argument that
    was not given explicitly. Because the propagators name their output
    wavefields with the *same* names as their initial-wavefield arguments, the
    round trip ``out = scalar(...); scalar(..., initial_state=out)`` continues
    the simulation (see the propagators' ``initial_state`` documentation for
    the condition under which it exactly reproduces a single combined run).

    A ``"auxiliary_state"`` entry is unpacked into the individual internal
    wavefield slots it holds. Supplying a field both inside it and as a
    separate key, or both in ``initial_state`` and as an explicit argument,
    raises rather than silently preferring one, and so does supplying a
    auxiliary state that holds only some of its fields (it is one bundle, and
    the absent fields would otherwise silently start from rest). An empty
    auxiliary state is accepted, and means exactly that.

    The individual absorbing-boundary and attenuation-memory auxiliary field
    names are not accepted as keys on any propagator: they are carried
    opaquely, under the one ``"auxiliary_state"`` key, and a call that names
    one raises pointing at ``initial_state=out``.

    When ``initial_state`` is a :class:`PropagatorOutput` carrying a
    :class:`RunInfo` (i.e. it was produced by an earlier propagation, not
    hand-built), and the caller supplied neither ``origin`` nor
    ``survey_pad`` themselves, the survey origin of that earlier run is
    recovered from its ``RunInfo`` and returned so the continuation call
    extracts the same region rather than guessing origin zero from the
    wavefield shape. A bare :class:`AuxiliaryState` records its own ``origin``
    and is used the same way.

    Args:
        initial_state: The bundle, or ``None`` (in which case ``wavefield_args``
            is returned unchanged, paired with ``origin``).
        wavefield_args: An ordered mapping from each initial-wavefield argument
            name to the value passed explicitly (or ``None``). Defines the set
            of valid names, in the propagator's internal flat order.
        origin: The caller's own ``origin`` argument, if any. Passed through
            unchanged unless it is ``None`` and can be recovered from
            ``initial_state``.
        survey_pad: The caller's own ``survey_pad`` argument, if any. Only
            used to decide whether recovering ``origin`` is appropriate
            (``survey_pad`` and ``origin`` are mutually exclusive), so it is
            not itself returned.
        ignore: Names to skip silently if present in ``initial_state`` (the
            receiver output names of a full :class:`PropagatorOutput`).
        auxiliary_state_names: The names this propagator carries in its
            :class:`AuxiliaryState`, spelled as the bundle keys them and as
            ``wavefield_args`` names their slots. That is the internal
            spelling for the absorbing-boundary auxiliaries, whose two
            spellings coincide, but not for the attenuating propagators'
            GSLS memory, which is ``r_0`` internally and ``r_mech0`` here.
            Empty for a propagator that folds nothing, which leaves every
            name a public one.
        propagator: The calling propagator's name, used to decide whether the
            pre-fold auxiliary key spellings are deprecated or rejected.

    Returns:
        A tuple of:

        - ``wavefield_args`` with the ``None`` entries filled from
          ``initial_state``. The order (hence unpacking order) is preserved.
        - The resolved ``origin``: ``origin`` unchanged if given, otherwise
          recovered from ``initial_state`` when possible, otherwise ``None``.

    Raises:
        DeepwaveTypeError: if ``initial_state`` is neither a
            :class:`PropagatorOutput` nor a mapping, or if its
            ``"auxiliary_state"`` entry is not a :class:`AuxiliaryState`.
        DeepwaveValueError: if a wavefield is supplied both in
            ``initial_state`` and as an explicit argument (ambiguous), if
            ``initial_state`` contains an unrecognised key (catches typos), or
            if its ``"auxiliary_state"`` holds some but not all of this
            propagator's auxiliary-state fields.

    """
    resolved_origin = list(origin) if origin is not None else None
    if initial_state is None:
        return wavefield_args, resolved_origin
    recovering_origin = resolved_origin is None and survey_pad is None
    # The region the state was produced in, when it records one. Read even
    # when the caller passed an explicit origin, so the two can be compared:
    # continuing in a different region than the wavefields describe displaces
    # every field, and used to do so silently.
    stored_origin: Optional[tuple[int, ...]] = None
    stored_extent: Optional[tuple[int, ...]] = None
    if isinstance(initial_state, PropagatorOutput):
        items: dict[str, Any] = dict(initial_state.as_dict())
        if initial_state.info is not None:
            stored_origin = tuple(initial_state.info.origin)
            stored_extent = tuple(initial_state.info.extent)
        if recovering_origin and stored_origin is not None:
            resolved_origin = list(stored_origin)
    elif isinstance(initial_state, abc.Mapping):
        items = dict(initial_state)
    else:
        raise DeepwaveTypeError(
            "initial_state must be a PropagatorOutput or a mapping from "
            "initial-wavefield name to Tensor, got "
            f"{type(initial_state).__name__}."
        )

    from_auxiliary_state: set[str] = set()
    if "auxiliary_state" in items:
        state = items.pop("auxiliary_state")
        if not isinstance(state, AuxiliaryState):
            raise DeepwaveTypeError(
                "initial_state's 'auxiliary_state' entry must be a "
                "deepwave.AuxiliaryState (take it from a previous call's "
                f"out.auxiliary_state), got {type(state).__name__}."
            )
        if stored_origin is None and state.origin is not None:
            # Only as the fallback the recovery below uses, never over the
            # top of a RunInfo's own record: the check and the recovery must
            # read the same field, or the check can reject the very region
            # the recovery would have chosen. The fingerprint travels with
            # the tensors themselves, which is what makes it the right
            # fallback for a hand-assembled mapping carrying no RunInfo.
            stored_origin = tuple(state.origin)
        if resolved_origin is None and recovering_origin and state.origin is not None:
            resolved_origin = list(state.origin)
        for name, value in state.items():
            if name in items:
                raise DeepwaveValueError(
                    f"{name!r} was given both inside initial_state's "
                    "auxiliary_state and as a separate initial_state key; "
                    "provide it in only one place."
                )
            items[name] = value
            from_auxiliary_state.add(name)

    _check_continuation_origin(
        origin=origin,
        stored_origin=stored_origin,
        stored_extent=stored_extent,
        # ``ignore`` holds the receiver outputs of a whole PropagatorOutput,
        # which are not state and so say nothing about which region the
        # state covers.
        state_has_fields=any(
            value is not None for name, value in items.items() if name not in ignore
        ),
    )

    resolved = dict(wavefield_args)
    legacy: list[str] = []
    for name, value in items.items():
        if name not in resolved:
            if name in ignore:
                continue
            valid = _public_state_key_names(resolved, auxiliary_state_names)
            raise DeepwaveValueError(
                f"initial_state contains unexpected key {name!r}; the valid "
                f"initial-state names are: {', '.join(valid)}."
            )
        if (
            value is not None
            and name in auxiliary_state_names
            and name not in from_auxiliary_state
        ):
            legacy.append(name)
        if value is None:
            # A None entry contributes nothing (e.g. a hand-built mapping with
            # an explicit None), so it neither fills nor conflicts with the
            # explicit argument.
            continue
        if resolved[name] is not None:
            container = (
                "initial_state's auxiliary_state"
                if name in from_auxiliary_state
                else "initial_state"
            )
            raise DeepwaveValueError(
                f"{name!r} was given both in {container} and as an explicit "
                "argument; provide it in only one place."
            )
        resolved[name] = value
    if legacy:
        _reject_legacy_state_keys(legacy, propagator)
    return resolved, resolved_origin


def _public_state_key_names(
    wavefield_args: abc.Mapping[str, Any],
    auxiliary_state_names: abc.Collection[str],
) -> list[str]:
    """The ``initial_state`` keys to name in an unrecognised-key message."""
    names = [n for n in wavefield_args if n not in auxiliary_state_names]
    if auxiliary_state_names:
        names.append("auxiliary_state")
    return names


def _reject_legacy_state_keys(
    names: abc.Sequence[str], propagator: Optional[str]
) -> None:
    """Rejects the auxiliary field names as ``initial_state`` keys.

    ``initial_state`` did not exist in ``v0.0.27``, which took each auxiliary
    field as its own keyword argument and returned it as a positional output,
    so no release ever accepted one of these names as an ``initial_state``
    key. They are named here rather than reported as merely unrecognised,
    because the positional return of the four propagators ``v0.0.27``
    released does still hold them, so code being migrated by name arrives
    here on all nine.

    Args:
        names: The offending keys, in the order the state supplied them.
        propagator: The calling propagator's name, or ``None`` where the
            caller does not know it.

    Raises:
        deepwave.DeepwaveValueError: Always. The function exists to phrase
            this one message in one place, for the propagator front end and
            the segment driver alike.

    """
    listed = ", ".join(repr(n) for n in names)
    plural = "keys" if len(names) > 1 else "key"
    verb = "name" if len(names) > 1 else "names"
    prefix = "" if propagator is None else f"{propagator}: "
    raise DeepwaveValueError(
        f"{prefix}initial_state {plural} {listed} {verb} absorbing-boundary "
        "or attenuation-memory auxiliary fields, which are not accepted as "
        "initial_state keys. To continue a run, pass the previous output as "
        "initial_state=out (or initial_state=out.final_state, or its "
        "auxiliary_state)."
    )


@dataclass
class BackendState:
    """Prepared tensors and parameters for one backend pass.

    The shared forward/backward engine assembles this and hands it to the
    propagator's ``call`` hook, which uses its fields to build the single
    explicit ctypes call that mirrors the propagator's C signature.
    """

    spec: "PropagatorSpec"
    ndim: int
    backend: Any
    models: list[torch.Tensor]
    source_amplitudes: list[torch.Tensor]
    sources_i: list[torch.Tensor]
    receivers_i: list[torch.Tensor]
    wavefields: list[torch.Tensor]
    receiver_amplitudes: list[torch.Tensor]
    storage_manager: "StorageManager"
    pml_profiles: list[torch.Tensor]
    grid_spacing: list[float]
    dt: float
    nt: int
    step_ratio: int
    pml_width: list[int]
    n_shots: int
    model_shape: tuple[int, ...]
    fd_pad: int
    fd_pad_list: list[int]
    pml_b: list[int]
    pml_e: list[int]
    # Per-face boundary-type codes for the first-class boundary faces, in the
    # ``pml_b``/``pml_e`` begins-then-ends order the C/CUDA backends expect
    # (``[d0_lo, d1_lo, ...]`` then ``[d0_hi, d1_hi, ...]``). All nine
    # propagators now thread real ``bc`` codes (each resolves its public
    # ``boundary`` argument via ``_common.set_boundary`` before reaching this
    # state -- that call always returns a full ``2*ndim``-length code list,
    # all-``BC_PML`` for the default/no-boundary case, never ``None``).
    bc: list[int]
    n_sources_per_shot: list[int]
    n_receivers_per_shot: list[int]
    aux: int
    stream: Any
    # Populated only for the backward pass:
    grad_models: list[torch.Tensor]
    # The per-shot/per-thread gradient scratch buffers that
    # ``grad_models_tmp_ptr`` points into (empty tensors where no scratch was
    # needed). Held here rather than in a caller local so that (a) the buffers
    # cannot be freed while the backend still holds their raw pointers, and
    # (b) ``run_backend_loop`` can re-zero them between chunks -- the compiled
    # backends reduce the scratch into ``grad_models`` at the end of *every*
    # backend call, so a scratch left dirty would be added in again by each
    # subsequent chunk.
    grad_models_tmp: list[torch.Tensor]
    grad_models_tmp_ptr: list[int]
    grad_sources: list[torch.Tensor]
    grad_receivers: list[torch.Tensor]
    aux_wavefields: list[torch.Tensor]

    @property
    def rdx(self) -> list[float]:
        """Reciprocal grid spacing in each dimension."""
        return [1.0 / dx for dx in self.grid_spacing]

    @property
    def rdx2(self) -> list[float]:
        """Reciprocal squared grid spacing (regular-grid propagators)."""
        return [1.0 / dx**2 for dx in self.grid_spacing]

    @property
    def output_slice(self) -> tuple[Any, ...]:
        """Slice that removes the finite-difference padding from a wavefield.

        ``model_shape`` is the padded simulation-domain shape, so the interior
        is obtained by trimming ``fd_pad_list`` cells from each side.
        """
        fd = self.fd_pad_list
        ms = self.model_shape
        return (
            slice(None),
            *(slice(fd[2 * i], ms[i] - fd[2 * i + 1]) for i in range(self.ndim)),
        )

    def storage_grad_flag(self, requires_grad: bool) -> bool:
        """Whether the backend should store data for ``requires_grad``."""
        return (
            requires_grad and self.storage_manager.storage_mode != _StorageModeEnum.NONE
        )


def _grid_padding(staggered: bool, fd_pad: int, ndim: int) -> list[int]:
    """Per-side finite-difference padding for the given grid type."""
    if staggered:
        return [fd_pad, fd_pad - 1] * ndim
    return [fd_pad] * (2 * ndim)


def _pml_region(
    staggered: bool,
    direction: str,
    model_shape: abc.Sequence[int],
    pml_width: abc.Sequence[int],
    fd_pad: int,
    ndim: int,
) -> tuple[list[int], list[int]]:
    """Computes the [pml_b, pml_e) interior bounds skipped by the PML kernels.

    The PML's spatial-derivative stencil widens its influence into the interior;
    the widening is one ``fd_pad`` per derivative order, which differs between
    the forward and backward passes and between the regular and staggered grids.
    """
    if staggered:
        lo = fd_pad if direction == "forward" else 2 * fd_pad
        hi = (fd_pad - 1) if direction == "forward" else (2 * fd_pad - 1)
        clamp = fd_pad - 1
    else:
        lo = 2 * fd_pad if direction == "forward" else 3 * fd_pad
        hi = lo
        clamp = fd_pad
    pml_b = [min(pml_width[2 * i] + lo, model_shape[i] - clamp) for i in range(ndim)]
    pml_e = [
        max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - hi) for i in range(ndim)
    ]
    return pml_b, pml_e


def bc_begins_ends(boundary: Optional[list[int]], ndim: int) -> list[int]:
    """Reorder per-face boundary codes into the backend's begins-then-ends order.

    ``boundary`` is in ``set_boundary`` face order (``[d0_lo, d0_hi, d1_lo,
    d1_hi, ...]``); the C/CUDA backends take the begin codes first then the end
    codes (mirroring ``pml_b``/``pml_e``): ``[d0_lo, d1_lo, ...] + [d0_hi,
    d1_hi, ...]``.

    Every propagator resolves its public ``boundary`` argument via
    ``_common.set_boundary`` (all-``BC_PML`` for the default/no-boundary
    case) before it reaches here, so in practice ``boundary`` is always a
    real ``2*ndim``-length code list, never ``None`` -- the ``None`` case
    below is this function's own defensive default, matching
    ``run_forward_core``/``run_backward_core``'s ``boundary: Optional[...] =
    None`` parameter default, and is not currently exercised by any
    propagator. The byte-identity of the ``boundary=None``/all-PML user case
    comes from the C-side ``has_bc`` guard (all-zero codes skip the fill),
    not from omitting the argument.
    """
    if boundary is None:
        return []
    begins = [int(boundary[2 * i]) for i in range(ndim)]
    ends = [int(boundary[2 * i + 1]) for i in range(ndim)]
    return begins + ends


def _model_flag_models(state: "BackendState") -> list[torch.Tensor]:
    """The models whose grad/batched flags are passed to the C backend."""
    indices = state.spec.c_model_flag_indices
    if indices is None:
        raise AssertionError(
            f"{state.spec.name} has no c_model_flag_indices; it must supply an "
            "explicit make_call."
        )
    if callable(indices):
        indices = indices(state.ndim)
    return [state.models[i] for i in indices]


def model_batched(models: list[torch.Tensor], ndim: int) -> list[bool]:
    """Whether each model carries a per-shot batch dimension."""
    return [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models]


def per_model_slot_flags(model_requires_grad: abc.Sequence[bool]) -> list[bool]:
    """The default slot rule: one backward-storage slot per (internal) model.

    Pure function of the models' ``requires_grad`` flags, so that
    :func:`deepwave.estimate_memory` can apply exactly the same rule the
    forward pass does (via :func:`_per_model_storage_flags`) without any
    tensors. See each propagator module's ``storage_slot_flags``.
    """
    return list(model_requires_grad)


def _per_model_storage_flags(models: list[torch.Tensor], ndim: int) -> list[bool]:
    """One backward-storage slot per model, gated by that model's requires_grad.

    The default for propagators whose stored data maps one-to-one to their
    models (scalar, acoustic). scalar_born and elastic supply their own because
    their storage slots do not correspond directly to individual models.
    """
    return per_model_slot_flags([m.requires_grad for m in models])


def call_grid_args(state: "BackendState", step_nt: int) -> tuple[Any, ...]:
    """The reciprocal-grid-spacing/dt/nt/n_shots/model-shape argument block.

    Identical, in this exact order, across every compiled propagator call
    (forward and backward, canonical and propagator-specific alike).
    """
    return (
        *state.rdx,
        state.dt,
        step_nt,
        state.n_shots,
        *state.model_shape,
    )


def call_storage_meta_args(state: "BackendState") -> tuple[Any, ...]:
    """The step-ratio/storage-mode/shot-bytes argument block.

    Identical, in this exact order, across every compiled propagator call
    (forward and backward, canonical and propagator-specific alike).
    """
    return (
        state.step_ratio,
        state.storage_manager.storage_mode,
        state.storage_manager.shot_bytes_uncomp,
        state.storage_manager.shot_bytes_comp,
    )


def call_pml_tail_args(state: "BackendState", step: int) -> tuple[Any, ...]:
    """The storage-compression/step/pml/aux/stream trailing argument block.

    Identical, in this exact order, across every compiled propagator call
    (forward and backward, canonical and propagator-specific alike).
    """
    return (
        state.storage_manager.storage_compression,
        step,
        *state.pml_b,
        *state.pml_e,
        *state.bc,
        state.aux,
        state.stream,
    )


def _canonical_forward_args(
    state: "BackendState", step: int, step_nt: int
) -> tuple[Any, ...]:
    """The canonical staggered-grid forward call's full argument tuple."""
    flag_models = _model_flag_models(state)
    return (
        *[m.data_ptr() for m in state.models],
        *[a.data_ptr() for a in state.source_amplitudes],
        *[w.data_ptr() for w in state.wavefields],
        *state.storage_manager.storage_ptrs,
        *[a.data_ptr() for a in state.receiver_amplitudes],
        *[p.data_ptr() for p in state.pml_profiles],
        *[loc.data_ptr() for loc in state.sources_i],
        *[loc.data_ptr() for loc in state.receivers_i],
        *call_grid_args(state, step_nt),
        *state.n_sources_per_shot,
        *state.n_receivers_per_shot,
        *call_storage_meta_args(state),
        *[state.storage_grad_flag(m.requires_grad) for m in flag_models],
        *model_batched(flag_models, state.ndim),
        *call_pml_tail_args(state, step),
    )


def canonical_forward_call(state: "BackendState", step: int, step_nt: int) -> int:
    """The canonical staggered-grid forward backend call.

    The staggered propagators (acoustic, elastic) share one C argument order, so
    the call is built here from :class:`BackendState` rather than being written
    out in each propagator. Only which models own a gradient differs, and that is
    captured by ``spec.c_model_flag_indices``.
    """
    return cast("int", state.backend(*_canonical_forward_args(state, step, step_nt)))


def make_forward_call_with_extra(
    extra_args: Callable[["BackendState"], tuple[Any, ...]],
) -> Callable[["BackendState", int, int], int]:
    """Wraps :func:`canonical_forward_call` with propagator-specific trailing args.

    For a propagator whose compiled forward symbol takes exactly the canonical
    staggered-grid argument order, then some extra trailing pointers/flags --
    e.g. elastic's parameter-modified free surface, which appends the per-shot-
    batched flag and the four surface-stiffness array pointers.
    """

    def make_call(state: "BackendState", step: int, step_nt: int) -> int:
        return cast(
            "int",
            state.backend(
                *_canonical_forward_args(state, step, step_nt), *extra_args(state)
            ),
        )

    return make_call


def _canonical_backward_args(
    state: "BackendState", step: int, step_nt: int
) -> tuple[Any, ...]:
    """The canonical staggered-grid backward call's full argument tuple."""
    flag_models = _model_flag_models(state)
    n_sources_for_grad = [
        count if grad.numel() > 0 else 0
        for count, grad in zip(state.n_sources_per_shot, state.grad_sources)
    ]
    return (
        *[m.data_ptr() for m in state.models],
        *[g.data_ptr() for g in state.grad_receivers],
        *[w.data_ptr() for w in state.wavefields],
        *[w.data_ptr() for w in state.aux_wavefields],
        *state.storage_manager.storage_ptrs,
        *[g.data_ptr() for g in state.grad_sources],
        *[g.data_ptr() for g in state.grad_models],
        *state.grad_models_tmp_ptr,
        *[p.data_ptr() for p in state.pml_profiles],
        *[loc.data_ptr() for loc in state.sources_i],
        *[loc.data_ptr() for loc in state.receivers_i],
        *call_grid_args(state, step_nt),
        *n_sources_for_grad,
        *state.n_receivers_per_shot,
        *call_storage_meta_args(state),
        *[state.storage_grad_flag(m.requires_grad) for m in flag_models],
        *model_batched(flag_models, state.ndim),
        *call_pml_tail_args(state, step),
    )


def canonical_backward_call(state: "BackendState", step: int, step_nt: int) -> int:
    """The canonical staggered-grid backward backend call (see forward)."""
    return cast("int", state.backend(*_canonical_backward_args(state, step, step_nt)))


def make_backward_call_with_extra(
    extra_args: Callable[["BackendState"], tuple[Any, ...]],
) -> Callable[["BackendState", int, int], int]:
    """Wraps :func:`canonical_backward_call` with propagator-specific trailing args.

    See :func:`make_forward_call_with_extra`.
    """

    def make_call(state: "BackendState", step: int, step_nt: int) -> int:
        return cast(
            "int",
            state.backend(
                *_canonical_backward_args(state, step, step_nt), *extra_args(state)
            ),
        )

    return make_call


@dataclass
class ForwardSavedState:
    """Everything a propagator's ``backward`` needs beyond the forward inputs.

    :func:`run_forward_engine` is intentionally free of any autograd ``ctx``:
    the modern ``Function.forward`` / ``setup_context`` split forbids populating
    ``ctx`` inside ``forward``. The engine therefore returns this bundle as the
    Function's final (non-tensor) output; :meth:`Function.setup_context` reads it
    off ``output`` and hands it to :func:`apply_forward_saved_state`, which does
    the actual ``ctx.save_for_backward`` plus cheap attribute assignment.

    Almost every field here is recomputable from the forward *inputs* (the model,
    location, amplitude, and wavefield tensors plus the scalar/metadata
    parameters). The one genuine exception is ``storage_manager``: it is created
    inside the engine and *filled by the backend during the time loop* (the
    backward-pass wavefield snapshots), so it cannot be recovered from the inputs
    in ``setup_context`` and must flow through the output — which is the whole
    reason this bundle is threaded through the Function's return value.

    ``needs_backward`` mirrors the old ``run_forward`` guard: the tensors and
    metadata are attached to ``ctx`` only when some input requires grad (so no
    graph is built otherwise), exactly as before.
    """

    needs_backward: bool
    saved_tensors: tuple[torch.Tensor, ...]
    grid_spacing: list[float]
    dt: float
    nt: int
    n_shots: int
    step_ratio: int
    accuracy: int
    pml_width: list[int]
    source_amplitudes_requires_grad: list[bool]
    backward_callback: Optional["Callback"]
    callback_frequency: int
    storage_manager: "StorageManager"
    backend_extra: str = ""
    # Opt-in second checkpoint buffer for the compiled scalar backend's
    # CallbackState role="forward" support (see scalar.py's ScalarForwardFunc/
    # ScalarBackwardFunc). ``None`` for every propagator that does not supply
    # one -- i.e. every propagator but scalar today.
    raw_forward_storage_manager: Optional["StorageManager"] = None
    # Whether that buffer was actually allocated and filled. Distinct from
    # ``raw_forward_storage_manager is not None``: scalar always builds the
    # manager (its pointers are part of the compiled ABI) but allocates its
    # storage only when the caller opted in, so the backward must be told
    # which of the two happened rather than inferring it from the manager.
    store_raw_forward: bool = False


def apply_forward_saved_state(ctx: Any, saved_state: "ForwardSavedState") -> None:
    """Populate an autograd ``ctx`` from a :class:`ForwardSavedState`.

    Called from a propagator's ``setup_context``. Does *only* the two things the
    modern API permits there: ``ctx.save_for_backward`` and cheap attribute
    assignment. The gating on ``needs_backward`` reproduces the old
    ``run_forward`` behaviour, where the context was populated only when a
    backward pass could actually be required.
    """
    if not saved_state.needs_backward:
        return
    ctx.save_for_backward(*saved_state.saved_tensors)
    ctx.grid_spacing = saved_state.grid_spacing
    ctx.dt = saved_state.dt
    ctx.nt = saved_state.nt
    ctx.n_shots = saved_state.n_shots
    ctx.step_ratio = saved_state.step_ratio
    ctx.accuracy = saved_state.accuracy
    ctx.pml_width = saved_state.pml_width
    ctx.source_amplitudes_requires_grad = saved_state.source_amplitudes_requires_grad
    ctx.backward_callback = saved_state.backward_callback
    ctx.callback_frequency = saved_state.callback_frequency
    ctx.storage_manager = saved_state.storage_manager
    ctx.backend_extra = saved_state.backend_extra
    ctx.raw_forward_storage_manager = saved_state.raw_forward_storage_manager
    ctx.store_raw_forward = saved_state.store_raw_forward


def _forward_oom_message(
    spec: "PropagatorSpec",
    model_shape: tuple[int, ...],
    nt: int,
    n_shots: int,
    dtype: torch.dtype,
    device: torch.device,
    accuracy: int,
    pml_width: list[int],
    step_ratio: int,
    storage_mode_str: StorageMode,
    storage_compression: bool,
    models: Optional[abc.Sequence[torch.Tensor]] = None,
) -> str:
    """Builds an actionable message for a CUDA out-of-memory error.

    Reports the configuration that failed and, best-effort, the
    :func:`deepwave.estimate_memory` forward+backward estimate for it,
    plus the levers available to reduce memory use. ``models`` (the internal
    model tensors, when the caller has them) makes the estimate's
    stored-history term exact rather than a worst case, by naming which
    models require a gradient. The
    :func:`estimate_memory` call is guarded so that any failure inside it
    (e.g. an unrecognised ``spec.name``) can never mask the original
    out-of-memory error; it is simply omitted from the message in that case.

    Called only from :func:`run_forward_engine`'s
    ``except torch.cuda.OutOfMemoryError`` handler, which chains the result
    onto the original error with ``raise ... from e``, so this function never
    needs to (and does not) touch the original exception itself.
    """
    message = (
        f"{spec.name} propagation ran out of GPU memory for this "
        f"configuration: padded model shape {model_shape}, nt={nt}, "
        f"n_shots={n_shots}, dtype={dtype}."
    )
    estimate_line = _forward_oom_estimate_line(
        spec,
        model_shape,
        nt,
        n_shots,
        dtype,
        device,
        accuracy,
        pml_width,
        step_ratio,
        storage_mode_str,
        storage_compression,
        models,
    )
    if estimate_line is not None:
        message += estimate_line
    message += (
        " To reduce memory use, consider: shot batching (chunk n_shots "
        'across calls), storage_mode="cpu" or "disk"'
        + ("" if storage_compression else ", storage_compression=True")
        + ", model_gradient_sampling_interval, or "
        "checkpointing via torch.utils.checkpoint over "
        "deepwave.checkpointing.run_segments."
    )
    return message


def _forward_oom_estimate_line(
    spec: "PropagatorSpec",
    model_shape: tuple[int, ...],
    nt: int,
    n_shots: int,
    dtype: torch.dtype,
    device: torch.device,
    accuracy: int,
    pml_width: list[int],
    step_ratio: int,
    storage_mode_str: StorageMode,
    storage_compression: bool,
    models: Optional[abc.Sequence[torch.Tensor]] = None,
) -> Optional[str]:
    """The :func:`estimate_memory` sentence for :func:`_forward_oom_message`.

    Returns ``None`` (rather than raising) on any failure -- an unrecognised
    ``spec.name``, a ``model_shape`` too small for its own padding, or
    anything else -- so a problem here can never mask the original
    out-of-memory error it is reporting alongside.
    """
    try:
        from deepwave._common import (
            _propagator_models_requiring_grad,
            estimate_memory,
        )

        ndim = len(model_shape)
        fd_pad = accuracy // 2
        unpadded_shape = tuple(
            model_shape[d] - 2 * fd_pad - pml_width[2 * d] - pml_width[2 * d + 1]
            for d in range(ndim)
        )
        if not all(s > 0 for s in unpadded_shape):
            return None
        # Every propagator now answers, so the estimate is no longer gated on
        # membership of a four-entry table.
        propagator = spec.name
        # Naming the models that require a gradient makes the stored-history
        # term exact rather than a worst case over that many models.
        models_requiring_grad = (
            None
            if models is None
            else _propagator_models_requiring_grad(propagator, models, ndim)
        )
        est = estimate_memory(
            propagator,
            unpadded_shape,
            nt,
            n_shots,
            dtype,
            accuracy=accuracy,
            pml_width=pml_width,
            model_gradient_sampling_interval=step_ratio,
            storage_mode=storage_mode_str,
            models_requiring_grad=models_requiring_grad,
            storage_compression=storage_compression,
            device=device,
        )
        return (
            " deepwave.estimate_memory predicts at least "
            f"{est['total_bytes_lower_bound'] / 1e9:.2f} GB "
            "(forward + backward) for this configuration."
        )
    except Exception:  # noqa: BLE001 - best-effort estimate; never mask the OOM
        return None


def run_forward_engine(
    spec: "PropagatorSpec",
    *,
    pml_profiles: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    n_shots: int,
    forward_callback: Optional["Callback"],
    backward_callback: Optional["Callback"],
    callback_frequency: int,
    storage_mode_str: StorageMode,
    storage_path: str,
    storage_compression: bool,
    args: abc.Sequence[torch.Tensor],
    storage_flags: Callable[
        [list[torch.Tensor], int], list[bool]
    ] = _per_model_storage_flags,
    zero_wavefields: Callable[..., None],
    make_call: Callable[["BackendState", int, int], int] = canonical_forward_call,
    make_aux_wavefields: Optional[
        Callable[[list[torch.Tensor], int], list[torch.Tensor]]
    ] = None,
    swap: Optional[Callable[["BackendState"], None]] = None,
    save_wavefields: bool = False,
    backend_extra: str = "",
    boundary: Optional[list[int]] = None,
    raw_forward_storage_manager: Optional["StorageManager"] = None,
    store_raw_forward: bool = False,
    wavefield_names_override: Optional[list[str]] = None,
) -> tuple[tuple[torch.Tensor, ...], "ForwardSavedState"]:
    """Runs a forward pass for a spec-described propagator (ctx-free).

    This performs all of the boilerplate shared by the propagators (scalar,
    scalar_born, acoustic, elastic): unpacking and making the inputs contiguous,
    allocating backward storage, padding and zeroing the wavefields, computing
    PML bounds, allocating receiver buffers, choosing the threading/stream,
    running the chunked time loop, and removing the padding from the returned
    wavefields. The propagator supplies only what is unique to it:
    ``storage_flags`` (the requires-grad condition for each backward-storage
    slot), ``zero_wavefields`` (its grid-specific edge/interior zeroing), and
    ``make_call`` (the single explicit backend call).

    The regular-grid propagators (scalar, scalar_born) additionally ping-pong
    their wavefields during the forward pass, so they supply ``make_aux_wavefields``
    (the temporary ping-pong buffers, e.g. ``psin``) and ``swap`` (exchanges the
    current/previous and temporary buffers after a chunk with an odd number of
    internal steps); the staggered propagators leave both ``None``. They also set
    ``save_wavefields`` so the raw input wavefields are saved for the backward
    pass (needed by scalar's double-backward).

    This function is deliberately free of any autograd ``ctx`` so that it can be
    called from the modern ``Function.forward`` (which never receives ``ctx``).
    It returns the padding-removed final wavefields followed by the receiver
    amplitudes (the order the autograd Function must return), *and* a
    :class:`ForwardSavedState` bundling everything the backward pass needs beyond
    the inputs. The Function returns that bundle as its trailing (non-tensor)
    output and, in ``setup_context``, hands it to
    :func:`apply_forward_saved_state` to populate ``ctx``.

    ``raw_forward_storage_manager`` (default ``None``, unused by every existing
    caller but scalar) is threaded straight into the returned
    :class:`ForwardSavedState` unchanged, for a propagator (scalar) whose
    ``make_call`` allocates and writes its own second checkpoint buffer (see
    ``ScalarForwardFunc.forward``) and needs it to survive into the backward
    pass alongside ``storage_manager``.

    ``store_raw_forward`` (default ``False``) says whether that buffer was
    actually allocated and filled, which scalar's backward needs in order to
    know whether it can serve ``role="forward"``: the manager is built
    unconditionally, because its pointers are part of the compiled ABI, but
    its storage is allocated only when the caller opted in.

    ``wavefield_names_override`` (default ``None``, unused by every existing
    caller but viscoacoustic) overrides the spec's fixed ``wavefield_names(ndim)``
    for the public callback-dict naming, mirroring :func:`run_backward_core`'s
    existing ``wavefield_names`` override -- for a propagator whose wavefield
    count depends on a runtime value (the ``q_n_mechanisms`` memory
    variables). Its length also overrides the flat-argument unpacking count
    passed to :meth:`PropagatorSpec.split_backend_args`.
    """
    from deepwave._common import _backend_function
    from deepwave.types import CallbackState

    ndim = len(grid_spacing)
    n_wavefields_override = (
        len(wavefield_names_override) if wavefield_names_override is not None else None
    )
    models, source_amplitudes, sources_i, receivers_i, wavefields = (
        spec.split_backend_args(ndim, args, n_wavefields_override)
    )
    models = [m.contiguous() for m in models]
    source_amplitudes = [a.contiguous() for a in source_amplitudes]
    pml_profiles = [p.contiguous() for p in pml_profiles]
    sources_i = [s.contiguous() for s in sources_i]
    receivers_i = [r.contiguous() for r in receivers_i]

    device = models[0].device
    dtype = models[0].dtype
    is_cuda = models[0].is_cuda
    model_shape = tuple(models[0].shape[-ndim:])
    try:
        storage_mode = resolve_storage_mode(storage_mode_str, device)
        check_storage_mode_model_gradient(spec, models, ndim, storage_mode)
        n_sources_per_shot = [s.numel() // n_shots for s in sources_i]
        n_receivers_per_shot = [r.numel() // n_shots for r in receivers_i]

        # Resolved above the history allocation on purpose. None of these five
        # reads anything that has been allocated -- they depend only on the
        # accuracy, the dimensionality, the model shape, the PML widths, the
        # dtype, the device, whether the grid is staggered, and the boundary
        # conditions -- and one of them, the compiled-symbol lookup, is how a
        # combination that was never built is discovered. Doing it here means
        # that discovery happens before the wavefield history is allocated
        # rather than after, so an unbuilt combination fails immediately
        # instead of first reserving the largest buffer in the run.
        fd_pad = accuracy // 2
        fd_pad_list = _grid_padding(spec.staggered, fd_pad, ndim)
        pml_b, pml_e = _pml_region(
            spec.staggered, "forward", model_shape, pml_width, fd_pad, ndim
        )
        bc = bc_begins_ends(boundary, ndim)
        backend = _backend_function(
            spec.name, ndim, "forward", accuracy, dtype, device, extra=backend_extra
        )

        storage_manager = StorageManager(
            model_shape,
            dtype,
            n_shots,
            nt,
            step_ratio,
            storage_mode,
            storage_compression,
            storage_path,
            device,
            is_cuda,
        )
        for flag in storage_flags(models, ndim):
            storage_manager.allocate(flag)

        # The raw (unpadded) input wavefields are saved for the backward pass; the
        # local ``wavefields`` name is rebound to the padded copies just below, so
        # capture the save-for-backward set now, before padding.
        needs_backward = (
            any(m.requires_grad for m in models)
            or any(a.requires_grad for a in source_amplitudes)
            or any(w.requires_grad for w in wavefields)
        )
        saved_state = ForwardSavedState(
            needs_backward=needs_backward,
            saved_tensors=(
                *models,
                *sources_i,
                *receivers_i,
                *source_amplitudes,
                *(wavefields if save_wavefields else []),
                *pml_profiles,
            ),
            grid_spacing=grid_spacing,
            dt=dt,
            nt=nt,
            n_shots=n_shots,
            step_ratio=step_ratio,
            accuracy=accuracy,
            pml_width=pml_width,
            source_amplitudes_requires_grad=[
                a.requires_grad for a in source_amplitudes
            ],
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            storage_manager=storage_manager,
            backend_extra=backend_extra,
            raw_forward_storage_manager=raw_forward_storage_manager,
            store_raw_forward=store_raw_forward,
        )

        size_with_batch = (n_shots, *model_shape)
        wavefields = [
            create_or_pad(w, fd_pad_list, device, dtype, size_with_batch)
            for w in wavefields
        ]
        aux_wavefields = (
            make_aux_wavefields(wavefields, ndim)
            if make_aux_wavefields is not None
            else []
        )
        zero_wavefields(wavefields, ndim, fd_pad, fd_pad_list, pml_width)

        receiver_amplitudes = [
            torch.empty(0, device=device, dtype=dtype) for _ in receivers_i
        ]
        for amplitudes, locs, n_per_shot in zip(
            receiver_amplitudes, receivers_i, n_receivers_per_shot
        ):
            if locs.numel() > 0:
                amplitudes.resize_(nt, n_shots, n_per_shot)
                amplitudes.fill_(0)

        aux, stream = get_aux_and_stream(
            models[0], n_shots, is_cuda, spatial_parallel=spec.spatial_parallel
        )

        state = BackendState(
            spec=spec,
            ndim=ndim,
            backend=backend,
            models=models,
            source_amplitudes=source_amplitudes,
            sources_i=sources_i,
            receivers_i=receivers_i,
            wavefields=wavefields,
            receiver_amplitudes=receiver_amplitudes,
            storage_manager=storage_manager,
            pml_profiles=pml_profiles,
            grid_spacing=grid_spacing,
            dt=dt,
            nt=nt,
            step_ratio=step_ratio,
            pml_width=pml_width,
            n_shots=n_shots,
            model_shape=model_shape,
            fd_pad=fd_pad,
            fd_pad_list=fd_pad_list,
            pml_b=pml_b,
            pml_e=pml_e,
            bc=bc,
            n_sources_per_shot=n_sources_per_shot,
            n_receivers_per_shot=n_receivers_per_shot,
            aux=aux,
            stream=stream,
            grad_models=[],
            grad_models_tmp=[],
            grad_models_tmp_ptr=[],
            grad_sources=[],
            grad_receivers=[],
            aux_wavefields=aux_wavefields,
        )

        model_dict = dict(zip(spec.model_names(ndim), models))
        callback_wavefield_names = (
            wavefield_names_override
            if wavefield_names_override is not None
            else spec.wavefield_names(ndim)
        )

        def build_state(step: int) -> "CallbackState":
            user_ratio = getattr(callback_frequency, "user_step_ratio", step_ratio)
            return CallbackState(
                dt * user_ratio,
                step,
                dict(zip(callback_wavefield_names, wavefields)),
                model_dict,
                {},
                fd_pad_list,
                pml_width,
                direction="forward",
                n_steps=nt // user_ratio,
                internal_dt=dt,
            )

        run_backend_loop(
            propagator=spec.name,
            direction="forward",
            active=wavefields[0].numel() > 0 and nt > 0,
            n_outer=nt // step_ratio,
            step_ratio=step_ratio,
            callback=forward_callback,
            callback_frequency=callback_frequency,
            call=lambda step, step_nt: make_call(state, step, step_nt),
            swap=(lambda: swap(state)) if swap is not None else None,
            build_state=build_state,
            storage_mode_str=storage_mode_str,
            storage_path=storage_path,
            forbid_forward_mutation=any_requires_grad(
                models, source_amplitudes, wavefields
            ),
        )

        s = state.output_slice
        return (*[w[s] for w in wavefields], *receiver_amplitudes), saved_state
    except torch.cuda.OutOfMemoryError as e:
        raise type(e)(
            _forward_oom_message(
                spec,
                model_shape,
                nt,
                n_shots,
                dtype,
                device,
                accuracy,
                pml_width,
                step_ratio,
                storage_mode_str,
                storage_compression,
                models,
            )
        ) from e


def leading_arg_count(forward: Callable[..., Any]) -> int:
    """Count the leading non-tensor parameters of a propagator ``forward``.

    Every propagator's autograd ``Function.forward`` takes a block of leading
    non-tensor configuration parameters (``pml_profiles``, ``grid_spacing``,
    ``dt``, ..., plus any propagator-specific flag such as scalar's ``boundary``
    or the elastic family's ``mpml``) followed by a variadic ``*args`` of the
    property models, source amplitudes, indices, and wavefield tensors. Those
    leading parameters all receive ``None`` gradients in ``backward``. This
    helper returns how many there are by counting the parameters that precede
    the ``*args`` (``VAR_POSITIONAL``) parameter, so each module can derive its
    ``_N_LEAD`` directly from the signature rather than hard-coding a count.

    A leading ``self``/``ctx`` parameter (present on old-style
    ``autograd.Function.forward`` methods that take the context explicitly) is
    not counted, since it is not one of the propagator's inputs.
    """
    count = 0
    for parameter in inspect.signature(forward).parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            break
        if parameter.name in ("self", "ctx"):
            continue
        count += 1
    return count


def _no_second_order_message(propagator: str) -> str:
    """Actionable error text raised by :func:`no_second_order` for ``propagator``."""
    return (
        f"Second-order differentiation (double-backward / Hessian-vector "
        f"products) is not implemented for the {propagator} propagator. Its "
        f"hand-derived adjoint has no analytic second-order path, so attempting "
        f"to differentiate through its backward pass (for example, calling "
        f"torch.autograd.grad(..., create_graph=True) and then differentiating "
        f"the resulting gradient) would silently return incorrect (typically "
        f"zero) results rather than the true second-order derivative. Deepwave "
        f"therefore raises this error instead of returning a wrong answer. "
        f'Re-run the call with backend="eager": that backend steps the same '
        f"physics under native autograd, which differentiates its own backward "
        f"pass, so create_graph=True and the torch.func transforms built on it "
        f"(torch.func.jacrev, torch.func.hessian, torch.func.jvp) give correct "
        f"second-order results there. It is slower than the compiled backend. "
        f"The scalar, acoustic, and elastic propagators support second-order "
        f"differentiation on the compiled backend as well. Failing that, "
        f"compute Hessian-vector products via finite differences of first-order "
        f"gradients (perturb the model, recompute the gradient, and take the "
        f"difference)."
    )


def no_second_order(
    propagator: str,
) -> Callable[
    [Callable[..., tuple[Optional[torch.Tensor], ...]]],
    Callable[..., tuple[Optional[torch.Tensor], ...]],
]:
    """Decorator forbidding second-order autograd through a ``Function.backward``.

    Used as ``@no_second_order("<propagator name>")``; the propagator name is
    interpolated into the raised error message. This is a stricter replacement
    for :func:`torch.autograd.function.once_differentiable` for propagators
    whose hand-derived adjoint (``backward``) has no analytic second-order
    (double-backward) counterpart: scalar_born, viscoacoustic, elastic_vti,
    acoustic_born, acoustic_vti, and elastic_born. Their ``backward`` calls the
    non-differentiable C/CUDA
    backend directly rather than building a nested differentiable ``Function``
    (as scalar, acoustic, and elastic do -- via their ``*BackwardFunc`` and the
    Born backend -- for their Hessian-vector products).

    ``once_differentiable`` alone does *not* protect these propagators. It
    only attaches an error node (``DelayedError``) to the backward outputs
    when a grad-output requires grad, and even then it *detaches* those
    outputs. Because these propagators reparameterise the user's model inside
    the differentiable Python front-end (for example acoustic_born maps
    ``v, rho -> K = rho * v**2`` before the ``Function``), that detach severs
    the only edge that reaches the user's leaf. The error node is left off any
    path to the differentiation target, so autograd prunes it and silently
    returns the reparameterisation term alone (missing the physics
    second-derivative) instead of raising. For the same reason a "poison" node
    placed on the backward *outputs* cannot work: the missing physics
    second-derivative edge simply does not exist in the graph, so there is no
    node lying on the target's path to poison.

    The reliable signal is therefore taken at entry to ``backward``:
    ``torch.is_grad_enabled()`` is ``True`` only when the caller built the
    backward under ``create_graph=True`` (or an equivalent higher-order
    context), i.e. exactly when a second-order graph is being constructed
    through this propagator. Plain first-order backward (``.backward()`` or
    ``torch.autograd.grad(...)`` with ``create_graph=False``), including every
    ``gradcheck`` path and any ``backward`` executed under ``torch.no_grad``,
    runs with grad disabled and is unaffected -- first-order results are
    byte-identical to running the undecorated ``backward`` under
    ``no_grad``. When grad is enabled we raise a clear, actionable error
    rather than let autograd build a silently-wrong second-order graph.
    """

    def decorator(
        fn: Callable[..., tuple[Optional[torch.Tensor], ...]],
    ) -> Callable[..., tuple[Optional[torch.Tensor], ...]]:
        @functools.wraps(fn)
        def wrapper(ctx: Any, *args: Any) -> tuple[Optional[torch.Tensor], ...]:
            if torch.is_grad_enabled():
                raise DeepwaveRuntimeError(_no_second_order_message(propagator))
            with torch.no_grad():
                return fn(ctx, *args)

        return wrapper

    return decorator


def _functorch_unsupported_message(transform: str) -> str:
    """Actionable error text for an unsupported ``torch.func`` transform.

    The setup_context migration lets ``torch.func`` transforms *reach* the
    compiled-backend autograd ``Function`` -- but the ``ctypes`` C/CUDA backend
    cannot process the batched (``vmap``) or dual (``jvp``) tensors that a
    transform would hand it, so the raw failure is an opaque PyTorch message
    that never mentions Deepwave's escape hatch. This helper replaces it with a
    message that names the transform and points at the pure-PyTorch eager
    backend (every propagator now provides ``backend="eager"``).
    """
    return (
        f"torch.func.{transform} is not supported on Deepwave's compiled "
        f"(C/CUDA) propagator backend: the ctypes backend cannot operate on "
        f"the batched/dual tensors that {transform} produces. Re-run the "
        f"propagator with the pure-PyTorch eager backend, backend="
        f'"eager" (slower but fully differentiable and functorch-compatible), '
        f"which supports {transform} through native autograd. Note the eager "
        f"backend currently supports vmap only over property models (e.g. "
        f"velocity), and requires max_vel to be given."
    )


def saved_state_ndim(ctx: Any, propagator: str) -> int:
    """``len(ctx.grid_spacing)``, with an actionable error if it is missing.

    Every compiled-backend ``backward`` starts by reading the spatial
    dimensionality out of the context its ``setup_context`` populated. That
    context is populated only when the forward pass saw that a backward could
    be needed (:func:`apply_forward_saved_state` gates on the inputs'
    ``requires_grad``), which is true for every ordinary use: with nothing
    requiring a gradient, autograd never calls ``backward`` at all.

    The reverse-mode ``torch.func`` transforms -- ``grad``, ``vjp``,
    ``jacrev``, ``jacfwd``'s reverse half, ``hessian`` -- break that
    correspondence. They differentiate a tensor that does *not* carry
    ``requires_grad`` through the forward pass, so nothing is saved, and then
    run a backward against a generated node anyway. Without this check the
    result is an ``AttributeError`` naming a private autograd class, which
    tells the caller nothing; with it, they get the same escape hatch
    ``vmap`` already offers (see :func:`unsupported_vmap`).

    Args:
        ctx: The autograd context handed to ``backward``.
        propagator: The public propagator name, for the message.

    Returns:
        The number of spatial dimensions.

    Raises:
        RuntimeError: If the forward pass saved no state.
    """
    grid_spacing = getattr(ctx, "grid_spacing", None)
    if grid_spacing is None:
        raise DeepwaveRuntimeError(
            f"deepwave.{propagator}'s backward pass was reached without the "
            "state that its forward pass saves. On Deepwave's compiled "
            "(C/CUDA) backend that happens under the reverse-mode torch.func "
            "transforms -- torch.func.grad, torch.func.vjp, torch.func.jacrev "
            "and torch.func.hessian -- which hide the inputs' requires_grad "
            "from the forward pass and then differentiate anyway. Re-run the "
            'propagator with the pure-PyTorch eager backend, backend="eager" '
            "(slower, but every torch.func transform works through native "
            "autograd there). An ordinary .backward() or torch.autograd.grad "
            "on the compiled backend is unaffected, and forward-mode "
            "torch.func.jvp is supported on the compiled backend of scalar, "
            "acoustic and elastic."
        )
    return len(grid_spacing)


def unsupported_vmap(info: Any, in_dims: Any, *args: Any) -> tuple[torch.Tensor, ...]:
    """``vmap`` staticmethod for compiled-backend Functions: raise, actionably.

    Registered instead of ``generate_vmap_rule=True``: an auto-generated rule
    would silently hand the ctypes backend batched tensors it cannot handle, so
    it is deliberately *not* enabled. This points the user to the eager backend.
    """
    raise DeepwaveRuntimeError(_functorch_unsupported_message("vmap"))


def unsupported_jvp(ctx: Any, *tangents: Any) -> tuple[torch.Tensor, ...]:
    """``jvp`` staticmethod for compiled-backend Functions: raise, actionably.

    Forward-mode AD over the ctypes backend is available for scalar, acoustic,
    and elastic (Born-based ``jvp`` rules); the propagators that register this
    fallback (scalar_born, viscoacoustic, elastic_vti, acoustic_born,
    acoustic_vti, elastic_born) have no
    compiled ``jvp`` rule, so point the user to the eager backend rather than
    let PyTorch emit its generic "implement the jvp function" error.
    """
    raise DeepwaveRuntimeError(_functorch_unsupported_message("jvp"))


def unwrap_functorch(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return the concrete tensor underlying a ``torch.func``-wrapped one.

    Inside a ``Function.jvp`` reached through ``torch.func.jvp`` the saved and
    tangent tensors are functorch wrappers (e.g. ``GradTrackingTensor``) whose
    storage the ctypes C/CUDA backend cannot address via ``data_ptr()``. Peel
    every functorch level to recover the underlying concrete tensor so it can be
    handed to the compiled backend. Under the legacy
    ``torch.autograd.forward_ad`` API (or with no transform active) the tensors
    are already concrete and this is a no-op. ``None`` (an absent tangent) passes
    through unchanged.
    """
    from deepwave import _common

    return None if t is None else _common.unwrap_functorch(t)


def run_backward_core(
    spec: "PropagatorSpec",
    *,
    models: list[torch.Tensor],
    sources_i: list[torch.Tensor],
    receivers_i: list[torch.Tensor],
    source_amplitudes: list[torch.Tensor],
    pml_profiles: list[torch.Tensor],
    grad_wavefields: list[torch.Tensor],
    grad_receivers: list[torch.Tensor],
    grid_spacing: list[float],
    dt: float,
    nt: int,
    n_shots: int,
    step_ratio: int,
    accuracy: int,
    pml_width: list[int],
    source_amplitudes_requires_grad: list[bool],
    storage_manager: "StorageManager",
    backward_callback: Optional["Callback"],
    callback_frequency: int,
    grad_model_batched: Callable[[list[torch.Tensor], int], list[bool]] = model_batched,
    make_aux_wavefields: Callable[[list[torch.Tensor], int], list[torch.Tensor]],
    zero_wavefields: Callable[..., None],
    make_call: Callable[["BackendState", int, int], int] = canonical_backward_call,
    swap: Callable[["BackendState"], None],
    cpu_tmp_batch: Optional[int] = None,
    pre_loop: Optional[Callable[["BackendState"], None]] = None,
    post_loop: Optional[Callable[["BackendState"], None]] = None,
    wavefield_names: Optional[list[str]] = None,
    backend_extra: str = "",
    boundary: Optional[list[int]] = None,
    forward_wavefields: Optional[
        Callable[[int], Optional[dict[str, torch.Tensor]]]
    ] = None,
    extra_grad_scratch: Optional[abc.Sequence[torch.Tensor]] = None,
) -> "BackendState":
    """Runs the shared adjoint pass given already-unpacked inputs.

    This is the body common to every propagator's backward pass: it allocates
    the source/model gradient buffers (including per-thread temporaries),
    pads/zeroes the gradient wavefields and the auxiliary ping-pong buffers,
    computes the (wider) backward PML bounds, and runs the chunked adjoint loop.
    Callers differ only in where the inputs come from (the autograd context or
    explicit arguments) and how the resulting gradients are assembled, so this
    returns the populated :class:`BackendState` for the caller to read.

    The number of source-gradient buffers is taken from
    ``source_amplitudes_requires_grad`` (so a caller that adjoints only a subset
    of source types, like scalar_born's scattered-only pass, can pass a shorter
    list). ``wavefield_names`` overrides the callback wavefield names (defaulting
    to ``spec.wavefield_names``), for the same reason.

    ``pre_loop`` and ``post_loop`` run immediately before and after the time
    loop (used, e.g., by scalar's negation trick on the previous wavefield).

    ``forward_wavefields`` (default ``None``, unused by every existing caller
    but scalar) is an optional ``step -> {name: tensor} | None`` callable;
    when given, its return value is passed as ``CallbackState``'s
    ``forward_wavefields`` argument so a ``backward_callback`` can read the
    reconstructed forward wavefield via ``get_wavefield(role="forward")`` on
    the compiled backend (see ``ScalarBackwardFunc.forward``, the only
    current caller). It is called once per callback firing regardless of
    whether the callback actually reads ``role="forward"``, so it may return
    ``None`` for a step with no data available (e.g. scalar's raw-forward
    checkpoint in ``storage_mode="none"``, which allocates nothing) rather
    than raise; ``get_wavefield(role="forward")`` itself raises a clear error
    only if and when the callback tries to use it.

    ``extra_grad_scratch`` (default ``None``) lets a propagator that allocates
    model-gradient scratch of its own -- outside the ``models`` list handled by
    :func:`allocate_model_gradients`, as elastic's parameter-modified
    free-surface stiffness arrays do -- have those buffers re-zeroed between
    chunks along with the engine's own. See ``grad_scratch`` in
    :func:`run_backend_loop` for why that is required.
    """
    from deepwave._common import _backend_function
    from deepwave.types import CallbackState

    ndim = len(grid_spacing)
    n_sources = len(source_amplitudes_requires_grad)

    device = models[0].device
    dtype = models[0].dtype
    is_cuda = models[0].is_cuda
    model_shape = tuple(models[0].shape[-ndim:])
    n_sources_per_shot = [s.numel() // n_shots for s in sources_i]
    n_receivers_per_shot = [r.numel() // n_shots for r in receivers_i]

    grad_sources = []
    for i in range(n_sources):
        grad_f = torch.empty(0, device=device, dtype=dtype)
        if source_amplitudes_requires_grad[i]:
            grad_f.resize_(nt, n_shots, n_sources_per_shot[i])
            grad_f.fill_(0)
        grad_sources.append(grad_f)

    fd_pad = accuracy // 2
    fd_pad_list = _grid_padding(spec.staggered, fd_pad, ndim)
    size_with_batch = (n_shots, *model_shape)
    grad_wavefields = [
        create_or_pad(w, fd_pad_list, device, dtype, size_with_batch)
        for w in grad_wavefields
    ]
    aux_wavefields = make_aux_wavefields(grad_wavefields, ndim)
    zero_wavefields(grad_wavefields, ndim, fd_pad, fd_pad_list, pml_width)

    pml_b, pml_e = _pml_region(
        spec.staggered, "backward", model_shape, pml_width, fd_pad, ndim
    )

    aux, stream = get_aux_and_stream(
        models[0], n_shots, is_cuda, spatial_parallel=spec.spatial_parallel
    )
    grad_models, grad_models_tmp, grad_models_tmp_ptr = allocate_model_gradients(
        models,
        grad_model_batched(models, ndim),
        n_shots,
        aux,
        is_cuda,
        storage_manager.storage_mode,
        device,
        dtype,
        cpu_tmp_batch=cpu_tmp_batch,
        spatial_parallel=spec.spatial_parallel,
    )

    backend = _backend_function(
        spec.name, ndim, "backward", accuracy, dtype, device, extra=backend_extra
    )

    state = BackendState(
        spec=spec,
        ndim=ndim,
        backend=backend,
        models=models,
        source_amplitudes=source_amplitudes,
        sources_i=sources_i,
        receivers_i=receivers_i,
        wavefields=grad_wavefields,
        receiver_amplitudes=[],
        storage_manager=storage_manager,
        pml_profiles=pml_profiles,
        grid_spacing=grid_spacing,
        dt=dt,
        nt=nt,
        step_ratio=step_ratio,
        pml_width=pml_width,
        n_shots=n_shots,
        model_shape=model_shape,
        fd_pad=fd_pad,
        fd_pad_list=fd_pad_list,
        pml_b=pml_b,
        pml_e=pml_e,
        bc=bc_begins_ends(boundary, ndim),
        n_sources_per_shot=n_sources_per_shot,
        n_receivers_per_shot=n_receivers_per_shot,
        aux=aux,
        stream=stream,
        grad_models=grad_models,
        grad_models_tmp=grad_models_tmp,
        grad_models_tmp_ptr=grad_models_tmp_ptr,
        grad_sources=grad_sources,
        grad_receivers=grad_receivers,
        aux_wavefields=aux_wavefields,
    )

    model_dict = dict(zip(spec.model_names(ndim), models))
    grad_model_dict = dict(zip(spec.model_names(ndim), grad_models))
    names = (
        wavefield_names if wavefield_names is not None else spec.wavefield_names(ndim)
    )

    def build_state(step: int) -> "CallbackState":
        user_ratio = getattr(callback_frequency, "user_step_ratio", step_ratio)
        return CallbackState(
            dt * user_ratio,
            step,
            dict(zip(names, grad_wavefields)),
            model_dict,
            grad_model_dict,
            fd_pad_list,
            pml_width,
            forward_wavefields=(
                forward_wavefields(step) if forward_wavefields is not None else None
            ),
            direction="backward",
            n_steps=nt // user_ratio,
            internal_dt=dt,
        )

    with storage_manager.backward_lock:
        if pre_loop is not None:
            pre_loop(state)

        run_backend_loop(
            propagator=spec.name,
            direction="backward",
            active=grad_wavefields[0].numel() > 0 and nt > 0,
            n_outer=nt // step_ratio,
            step_ratio=step_ratio,
            callback=backward_callback,
            callback_frequency=callback_frequency,
            call=lambda step, step_nt: make_call(state, step, step_nt),
            swap=lambda: swap(state),
            build_state=build_state,
            grad_scratch=[*grad_models_tmp, *(extra_grad_scratch or ())],
        )

        if post_loop is not None:
            post_loop(state)
        if models[0].is_cuda:
            torch.cuda.synchronize(models[0].device)

    return state


def run_backward(
    ctx: Any,
    spec: "PropagatorSpec",
    args: abc.Sequence[torch.Tensor],
    *,
    grad_model_batched: Callable[[list[torch.Tensor], int], list[bool]] = model_batched,
    make_aux_wavefields: Callable[[list[torch.Tensor], int], list[torch.Tensor]],
    zero_wavefields: Callable[..., None],
    make_call: Callable[["BackendState", int, int], int] = canonical_backward_call,
    swap: Callable[["BackendState"], None],
    cpu_tmp_batch: Optional[int] = None,
    n_leading: int,
    boundary: Optional[list[int]] = None,
    model_transform: Optional[
        Callable[[list[torch.Tensor]], list[torch.Tensor]]
    ] = None,
    wavefield_names: Optional[list[str]] = None,
) -> tuple[Optional[torch.Tensor], ...]:
    """Runs a backward pass for a spec-described propagator's autograd Function.

    Unpacks the output gradients (``args``) and the tensors saved in the forward
    pass (``ctx``), delegates the state-building and adjoint loop to
    :func:`run_backward_core`, and assembles the gradient tuple aligned with the
    forward inputs.

    The propagator supplies only what is unique to it: ``grad_model_batched``
    (per-model batched flags), ``make_aux_wavefields`` (its ping-pong auxiliary
    buffers), its grid-specific ``zero_wavefields``, the explicit backend
    ``make_call``, and the ``swap`` that ping-pongs its buffers. ``cpu_tmp_batch``
    overrides the per-thread gradient buffer size, and ``n_leading`` is the
    propagator's full count of leading non-tensor arguments (from
    :func:`leading_arg_count`), which receive ``None`` gradients.

    ``boundary`` (default ``None``, unused by every existing caller) threads
    the first-class boundary ``bc`` codes through to :func:`run_backward_core`,
    for a propagator (e.g. viscoacoustic, B5) whose compiled adjoint reads its
    models at neighbour offsets near a reflecting face; ``model_transform``
    (default ``None``, a no-op) is applied to the unpacked ``models`` list
    right before they are handed to :func:`run_backward_core`, for that same
    propagator to mirror-pad the model guard rings (see
    ``_boundaries.mirror_pad_acoustic_models`` for the pattern this mirrors --
    acoustic does this inline because it does not use this helper).

    ``wavefield_names`` (default ``None``, unused by every existing caller but
    viscoacoustic) is threaded straight through to
    :func:`run_backward_core`'s existing override of the same name, for a
    propagator whose wavefield count depends on a runtime value the spec's
    fixed ``wavefield_names(ndim)`` cannot see.
    """
    ndim = saved_state_ndim(ctx, spec.name)
    n_models = spec.n_models(ndim)
    n_sources = spec.n_source_types(ndim)
    n_receivers = spec.n_receiver_types(ndim)

    grad_receivers = [g.contiguous() for g in args[-n_receivers:]]
    grad_wavefields = list(args[:-n_receivers])

    saved = list(ctx.saved_tensors)
    models = [t.contiguous() for t in saved[:n_models]]
    del saved[:n_models]
    if model_transform is not None:
        models = model_transform(models)
    sources_i = [t.contiguous() for t in saved[:n_sources]]
    del saved[:n_sources]
    receivers_i = [t.contiguous() for t in saved[:n_receivers]]
    del saved[:n_receivers]
    source_amplitudes = [t.contiguous() for t in saved[:n_sources]]
    del saved[:n_sources]
    pml_profiles = [t.contiguous() for t in saved]

    state = run_backward_core(
        spec,
        models=models,
        sources_i=sources_i,
        receivers_i=receivers_i,
        source_amplitudes=source_amplitudes,
        pml_profiles=pml_profiles,
        grad_wavefields=grad_wavefields,
        grad_receivers=grad_receivers,
        grid_spacing=ctx.grid_spacing,
        dt=ctx.dt,
        nt=ctx.nt,
        n_shots=ctx.n_shots,
        step_ratio=ctx.step_ratio,
        accuracy=ctx.accuracy,
        pml_width=ctx.pml_width,
        source_amplitudes_requires_grad=ctx.source_amplitudes_requires_grad,
        storage_manager=ctx.storage_manager,
        backward_callback=ctx.backward_callback,
        callback_frequency=ctx.callback_frequency,
        grad_model_batched=grad_model_batched,
        make_aux_wavefields=make_aux_wavefields,
        zero_wavefields=zero_wavefields,
        make_call=make_call,
        swap=swap,
        cpu_tmp_batch=cpu_tmp_batch,
        backend_extra=getattr(ctx, "backend_extra", ""),
        boundary=boundary,
        wavefield_names=wavefield_names,
    )

    out_s = state.output_slice
    return tuple(
        [None] * n_leading
        + state.grad_models
        + state.grad_sources
        + [None] * n_sources
        + [None] * n_receivers
        + [w[out_s] for w in state.wavefields]
    )

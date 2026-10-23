"""Pre-flight: what a propagator call would resolve to, and what it would cost.

The point of this module is that it contains **no model of a propagator call**.
:func:`plan` answers by making the call -- running the propagator's own front
end, its own shared setup, its own validation -- and stopping it at the moment
before propagation and gradient-history allocation. Everything it reports is
therefore produced by the code a real run uses, and cannot drift from it: a
resolution step added to a propagator tomorrow is in the answer tomorrow,
whether or not anyone remembers this file exists.

The stop is a private exception raised from :func:`plan_reached`, which each
propagator calls immediately after it has built its :class:`deepwave.RunInfo`
and immediately before it enters the time loop. When nothing is planning,
:func:`plan_reached` is a cheap ``ContextVar`` read and returns.
"""

import contextvars
import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from deepwave._exceptions import (
    DeepwaveRuntimeError,
    DeepwaveTypeError,
    capture_deepwave_warnings,
)
from deepwave._storage import format_bytes

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from collections.abc import Sequence

    import torch

    from deepwave._engine import PropagatorSpec
    from deepwave.types import MemoryEstimate, RunInfo

__all__ = ["Plan", "plan"]


class _PlanReached(Exception):  # noqa: N818 -- a control-flow signal, not an error
    """Internal signal: the resolution reached the point a plan stops at.

    Never seen by a caller -- :func:`plan` catches it. It is an ``Exception``
    rather than a return value because the stop has to unwind out of the middle
    of a propagator's body, several frames below :func:`plan`, without every
    intermediate frame having to know that planning is a thing.
    """


@dataclasses.dataclass
class _PlanProbe:
    """The collector a planning call threads through the propagator."""

    n_segments: Optional[int] = None
    info: Optional["RunInfo"] = None
    memory: Optional["MemoryEstimate"] = None


_PROBE: contextvars.ContextVar[Optional[_PlanProbe]] = contextvars.ContextVar(
    "deepwave_plan_probe", default=None
)


def active_probe() -> Optional[_PlanProbe]:
    """The probe of the :func:`plan` call in progress, or None for a real run."""
    return _PROBE.get()


def record_memory(memory: "MemoryEstimate") -> None:
    """Hands the run-shaped memory estimate to a planning call, if there is one.

    Args:
        memory: The estimate the resolved call made of itself.

    """
    probe = _PROBE.get()
    if probe is not None:
        probe.memory = memory


def plan_reached(
    spec: "PropagatorSpec",
    ndim: int,
    info: "RunInfo",
    models: "list[torch.Tensor]",
    storage_mode_str: Any,
    names: Optional["Sequence[str]"] = None,
    state_names: Optional["Sequence[str]"] = None,
    fold_names: Optional["Sequence[str]"] = None,
    *,
    python_backend: Optional[str] = None,
    storage_compression: bool = False,
    has_backward_callback: bool = False,
) -> None:
    """The point a plan stops at: everything resolved, nothing yet propagated.

    Called by every propagator between building its :class:`deepwave.RunInfo`
    and entering the time loop. Returns immediately unless a :func:`plan` call
    is in progress on this context.

    The engine-side storage checks are performed here rather than left to
    the engine, because the engine is entered only on the compiled path and
    :func:`plan` must raise the same errors whichever backend the call resolved
    to. They are *called*, not reimplemented: the engine runs the same
    functions.

    The Python-backend capability rejection is in the same position for the
    same reason, and it is run **first**, because that is the order the real
    call hits them: a propagator's Python backend rejects an unsupported
    ``storage_mode`` before anything asks whether that mode could carry a
    model gradient, so a plan that checked the gradient first would blame the
    gradient for a failure the run blames on the backend. The rejection lives
    *below* this point in every propagator's source (inside the ``*_python``
    path, and two propagators reach it only through a shared eager core), so
    it cannot be reached by moving the stop; it is called here with the same
    arguments the propagator will pass it.

    Args:
        spec: The propagator's spec.
        ndim: The number of spatial dimensions.
        info: The :class:`deepwave.RunInfo` this call would return.
        models: The internal models, exactly as the engine would receive them.
        storage_mode_str: The resolved storage mode, before the device-specific
            downgrade the engine applies.
        names: The propagator's override of its public field names, passed
            exactly as it will pass it to ``named_output``, or None where it
            uses the spec's own list. Viscoacoustic needs it because its output
            shape depends on the number of relaxation mechanisms.
        state_names: The matching override of the state field names.
        fold_names: The matching override of the state field names carried
            opaquely in the auxiliary state.
        python_backend: The Python backend name this call would report in an
            unsupported-option message, or None/empty where the call resolved
            to the compiled backend and no such rejection can happen. For a
            propagator that has a compiled backend this is the resolved mode
            (``"eager"``/``"compile"``), as
            :func:`deepwave._engine.reject_unsupported_python_backend_options`
            documents.
        storage_compression: Whether lossy storage compression was requested.
        has_backward_callback: Whether a ``backward_callback`` was supplied.

    Raises:
        _PlanReached: Always, when a plan is in progress.

    """
    probe = _PROBE.get()
    if probe is None:
        return
    from deepwave._engine import (
        check_storage_mode_model_gradient,
        complete_run_info,
        reject_unsupported_python_backend_options,
        resolve_storage_mode,
        resolved_output_names,
    )

    if python_backend:
        reject_unsupported_python_backend_options(
            spec,
            python_backend,
            storage_mode=storage_mode_str,
            storage_compression=storage_compression,
            has_backward_callback=has_backward_callback,
            # A plan is not run inside ``torch.func.vmap``: the refusals a
            # transform triggers are already documented as outside what
            # ``plan()`` can raise.
            is_batched=False,
        )
    storage_mode = resolve_storage_mode(storage_mode_str, models[0].device)
    check_storage_mode_model_gradient(spec, models, ndim, storage_mode)
    # The two name fields are filled when a run names its outputs, which is
    # after the point this stops at, so they are filled here by the same
    # functions -- otherwise a plan's RunInfo would differ from the run's in
    # exactly the two fields nobody would think to check.
    output_names, n_public_state = resolved_output_names(
        spec, ndim, names, state_names, fold_names
    )
    probe.info = complete_run_info(info, ndim, output_names, n_public_state)
    raise _PlanReached


@dataclasses.dataclass(frozen=True)
class Plan:
    """What a propagator call would resolve to, and what it would cost.

    Returned by :func:`deepwave.plan`. Every field is produced by the
    propagator's own resolution code, so it is what the call would really do
    rather than a second opinion about it.

    Stability: provisional (see :doc:`/stability`) -- fields may be added, and
    :meth:`summary` may be reworded, as more of a run's resolved configuration
    becomes worth reporting. The fields it has are not renamed or withdrawn
    without a release's notice under
    :class:`deepwave.DeepwaveFutureWarning`.

    Attributes:
        info: Exactly the :class:`deepwave.RunInfo` the call would attach to
            its output, built by the same code. Read the resolved internal time
            step, step ratio, PML width and frequency, survey origin and
            extent, backend and storage policy off it.
        memory: The call's estimate of its own footprint, in the same shape
            :func:`deepwave.estimate_memory` returns. It is more accurate than
            the parametric form: the padded shape, the sampling interval, the
            storage-slot count and the wavefield count are read off the
            resolved run rather than derived from arguments. It remains a
            **lower bound**, including under ``checkpoint_segments=`` -- see
            :func:`deepwave.estimate_memory` for what it does not model.
        resampling: Whether the call would sub-step, i.e. whether
            ``info.step_ratio`` is greater than 1. The receiver amplitudes are
            resampled back to the requested rate, and ``info.internal_dt``
            gives the rate they were computed at.
        warnings: Deepwave warnings the call would emit, as text. Collected
            on this execution context without changing process warning filters
            or spending the once-per-process warning budget.
        checkpoint_segments: The number of checkpointed time segments this plan
            costed, or ``None`` for an unsegmented call. The resolution in
            ``info`` always describes the whole run; only ``memory`` reflects
            the segmentation.

    """

    info: "RunInfo"
    memory: "MemoryEstimate"
    resampling: bool
    warnings: tuple[str, ...] = ()
    checkpoint_segments: Optional[int] = None

    def __repr__(self) -> str:
        """One short line: the shape, the timing and the bill."""
        shape = "x".join(str(s) for s in self.memory["padded_model_shape"])
        return (
            f"Plan(padded_shape={shape}, nt={self.info.nt}, "
            f"step_ratio={self.info.step_ratio}, "
            f"backend={self.info.backend!r}, "
            f"memory>={format_bytes(self.memory['total_bytes_lower_bound'])}, "
            f"warnings={len(self.warnings)})"
        )

    __str__ = __repr__

    def summary(self) -> str:
        """A short human-readable block describing the planned call.

        Returns:
            A multi-line block covering the resolved grid and padded shape,
            the timing
            (the user's ``nt`` and ``dt``, and the internal step count and
            ``internal_dt`` they resolve to), the backend, the storage
            policy, the memory breakdown, and any warnings the call would
            emit.

        """
        info = self.info
        memory = self.memory
        shape = " x ".join(str(s) for s in memory["padded_model_shape"])
        extent = " x ".join(str(s) for s in info.extent)
        storage = "unrecorded" if info.storage is None else repr(info.storage)
        lines = [
            f"extent {extent} (origin {tuple(info.origin)}), padded to {shape}",
            # Both time steps, named: ``nt`` counts the user's samples and
            # ``internal_dt`` is the sub-stepped one, so printing the pair
            # "nt N at dt <internal_dt>" invited the reader to multiply them
            # and get the wrong record length by a factor of step_ratio.
            f"nt {info.nt} at dt {info.internal_dt * info.step_ratio:.6g} s "
            f"-> {info.nt * info.step_ratio} internal steps at "
            f"{info.internal_dt:.6g} s "
            f"(step_ratio {info.step_ratio}, courant {info.courant_number:g})",
            f"pml_width {tuple(info.pml_width)} at {info.pml_freq:g} Hz, "
            f"max_vel {info.max_vel}",
            f"backend {info.backend!r} "
            f"(backward {info.backward_backend!r}) on {info.device}, "
            f"{info.dtype}",
            f"storage {storage}, lossy_compression_used {info.lossy_compression_used}",
        ]
        if self.checkpoint_segments is not None:
            lines.append(f"checkpoint_segments {self.checkpoint_segments}")
        lines.append(
            "memory at least "
            f"{format_bytes(memory['total_bytes_lower_bound'])} = "
            f"{format_bytes(memory['forward_wavefields_bytes'])} wavefields + "
            f"{format_bytes(memory['backward_storage_bytes'])} history + "
            f"{format_bytes(memory['gradient_scratch_bytes'])} "
            "gradient scratch + "
            f"{format_bytes(memory['auxiliary_state_bytes'])} auxiliary states"
        )
        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s):")
            lines.extend(f"  {text}" for text in self.warnings)
        else:
            lines.append("no warnings")
        return "\n".join(lines)


Plan.__module__ = "deepwave"


def _resolve_callable(propagator: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
    """The propagator to call, from a name or the function itself.

    A :class:`~torch.nn.Module` **class** is refused rather than quietly
    treated as its functional: the two do not answer the same question. The
    functional honours the ``requires_grad`` of the tensor it is handed, while
    a ``Module`` constructor stores its models with ``requires_grad=False``
    unless told otherwise, so planning the class reports the gradient
    machinery of a run the ``Module`` would not perform. The class alone also
    carries none of the tensors that would settle the question, so there is
    nothing to construct a faithful instance from.

    Args:
        propagator: The propagator's name, or the functional itself, or a
            constructed ``Module`` instance (which is callable and not a
            type, so it is planned as it stands).

    Returns:
        The callable to run.

    Raises:
        DeepwaveTypeError: If a class is given rather than an instance.

    """
    if isinstance(propagator, type):
        from deepwave.introspection import _resolve_spec

        # Raises its own error, listing the nine, for a class that is not a
        # propagator at all.
        name = _resolve_spec(propagator).name
        raise DeepwaveTypeError(
            f"plan() cannot plan the {propagator.__name__} class itself, only "
            "a propagator you could actually call. The class carries none of "
            "the tensors its constructor would store, and the constructor "
            "keeps requires_grad only where its own requires_grad= arguments "
            f"say so, so planning {propagator.__name__} would report the run "
            f"deepwave.{name} does rather than the one "
            f"{propagator.__name__}(...) does. Pass "
            f"the functional deepwave.{name}, or the name {name!r}, or a "
            f"constructed instance: plan({propagator.__name__}(...), dt, ...)."
        )
    if callable(propagator):
        return propagator
    import deepwave
    from deepwave.introspection import _resolve_spec

    return getattr(deepwave, _resolve_spec(propagator).name)  # type: ignore[no-any-return]


def plan(
    propagator: Union[str, Callable[..., Any]],
    *model_args: Any,
    **call_kwargs: Any,
) -> Plan:
    """Resolves a propagator call without running it, and reports what it costs.

    Call it exactly as you would call the propagator::

        p = deepwave.plan(
            deepwave.scalar, v, dx, dt,
            source_amplitudes=s, source_locations=sl, receiver_locations=rl,
        )
        print(p.summary())

    The propagator's name works too (``deepwave.plan("scalar", ...)``), so a
    plan can be driven from a configuration file.

    This is not a model of a propagator call: it *is* the call, stopped at the
    moment before propagation and gradient-history allocation. The resolution --
    the internal time step, the step ratio, the inferred PML frequency, the
    survey extraction, the backend choice, the storage policy -- is done by the
    propagator's own code, so :attr:`Plan.info` is the very
    :class:`deepwave.RunInfo` the run would return, and cannot drift from it.

    **What it raises.** Every error your call would raise about its arguments:
    a bad model, mismatched source and location shapes, receivers outside the
    model, a ``courant_number`` above critical, a
    ``model_gradient_sampling_interval`` that does not divide the time steps,
    an unknown ``storage_mode``, ``storage_mode="none"`` with a model requiring
    a gradient, and an unsupported option for the requested backend.

    **What it cannot raise**, because these depend on the run rather than on
    its arguments: an out-of-memory condition; a compiled-backend failure
    (including a disk write failure, and the missing-symbol error for an
    ``(ndim, accuracy, dtype, device)`` combination that was not built, since
    which kernel variant is needed is settled after the point this stops at);
    and the refusals that a ``torch.func`` transform or a double backward
    triggers on the result.

    **What it costs.** It does the propagator's setup, so it pads the models
    and resamples the source amplitudes just as the run would. Front-end
    preprocessing can allocate tensors proportional to the source time length;
    planning never allocates the propagation or gradient history.

    **Warnings.** :attr:`Plan.warnings` collects Deepwave's own warnings on
    this execution context. It does not alter Python's process warning filters,
    so unrelated threads emit warnings normally. Third-party warnings raised
    during setup follow the caller's ordinary warning policy and are not added
    to the plan. A Deepwave warning promoted to an error by the caller's filter
    is still collected as text during planning; the real call raises as usual.

    Args:
        propagator: The propagator to plan: the function itself
            (``deepwave.scalar``), its name as a string (``"scalar"``), or a
            constructed :class:`~torch.nn.Module` **instance**
            (``Scalar(v, dx)``), which is planned exactly as it would run. The
            ``Module`` *class* is refused, because it is not something that
            can be called and the plan for it would differ from the plan for
            any instance built from it -- see ``_resolve_callable``.
        *model_args: The positional arguments of that propagator, unchanged.
        **call_kwargs: Its keyword arguments, unchanged.
            ``checkpoint_segments=`` is understood here as well: the arguments
            a segmented call refuses are refused here too, and then the
            resolution reported describes the whole run while the memory
            estimate accounts for the segmentation.

    Returns:
        Plan: the resolved call and its estimated cost.

    Raises:
        DeepwaveRuntimeError: If the propagator returned without reaching the
            point a plan stops at, which means it is not instrumented.

    """
    func = _resolve_callable(propagator)
    segments = call_kwargs.get("checkpoint_segments")
    probe = _PlanProbe(n_segments=segments)
    token = _PROBE.set(probe)
    try:
        with capture_deepwave_warnings() as caught:
            try:
                func(*model_args, **call_kwargs)
            except _PlanReached:
                pass
            else:
                raise DeepwaveRuntimeError(
                    "plan() ran the whole of "
                    f"{getattr(func, '__name__', propagator)!r} without "
                    "reaching the point it stops at, which should be "
                    "unreachable: every propagator signals it between building "
                    "its RunInfo and entering the time loop. Please report "
                    "this."
                )
    finally:
        # The once-per-process warning registry needs no unwinding: a warning
        # helper that finds a plan active on its context reports the warning
        # without recording the cause, so a plan never spends the budget the
        # real call needs and never has to guess which entries were its own.
        _PROBE.reset(token)
    if probe.info is None or probe.memory is None:  # pragma: no cover - defensive
        raise DeepwaveRuntimeError(
            "plan() stopped at the right point but collected an incomplete "
            "answer. Please report this."
        )
    return Plan(
        info=probe.info,
        memory=probe.memory,
        resampling=probe.info.step_ratio > 1,
        warnings=tuple(caught),
        checkpoint_segments=segments,
    )


plan.__module__ = "deepwave"

"""Declared capability matrix and runtime conformance probes.

The published table comes from :class:`deepwave.PropagatorSpec` and explicit
declarations below. Runtime probes check those promises; a broken or missing
kernel cannot silently remove a feature from documentation. Probes use small
CPU problems and nonzero derivative directions, so they test availability,
not scientific accuracy.

This module also renders the second generated docs table,
``docs/callback_names.rst``'s per-propagator list of the **internal callback
names** (:func:`internal_names` and :func:`callback_names_rst`). That one is
declared rather than probed, and for the same reason the two declared
capability columns are: every name comes from the ``PropagatorSpec``
callables ``model_names``/``wavefield_names``/``auxiliary_state_names``,
which are the exact callables :mod:`deepwave._engine` (and each propagator's
own eager backend) zips the live tensors against when it builds a
:class:`deepwave.CallbackState`, so the table cannot disagree with the names
a callback is actually handed. ``tests/test_callback_names.py`` confirms
that by running real callbacks and comparing.
"""

import dataclasses
import inspect
import tempfile
import warnings
from typing import Any, Callable, Optional, cast

import torch
from torch.func import jvp, vmap

import deepwave
from deepwave import _backend_utils
from deepwave._engine import PropagatorSpec, propagator_spec
from deepwave._exceptions import DeepwaveValueError
from deepwave.acoustic import acoustic_output_names
from deepwave.acoustic_born import acoustic_born_output_names
from deepwave.acoustic_vti import acoustic_vti_output_names
from deepwave.checkpointing import _get_spec
from deepwave.elastic import elastic_output_names
from deepwave.elastic_born import elastic_born_output_names
from deepwave.elastic_vti import elastic_vti_output_names
from deepwave.scalar import scalar_output_names
from deepwave.scalar_born import scalar_born_output_names
from deepwave.viscoacoustic import viscoacoustic_output_names
from deepwave.wavelets import ricker

#: The nine shipped propagator functionals, in the same order used throughout
#: the test suite's own registries (e.g.
#: ``tests/test_signature_consistency.py``'s ``_FUNCTIONALS``).
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

_DTYPE = torch.float64
_N = 10
_NT = 6
_DX = 5.0
_DT = 0.001
_FREQ = 25.0
_MAX_VEL = 4000.0

# The receiver output whose name identifies the propagator's primary
# pressure/p-wave-like channel (or, for acoustic_vti, its x velocity channel,
# which the backend records directly -- that propagator's pressure receiver
# is synthesised from the two normal stresses, so probing it would probe the
# wrapper rather than the kernel), and the function that lists a
# propagator's full output tuple names in order.
# Both are needed to find the right element of the returned tuple to probe
# (mirrors ``tests/test_adjoint_dot_product.py``'s ``_REC_NAME``/
# ``_OUTPUT_NAMES``, which established this pattern first).
_REC_NAME: dict[str, str] = {
    "scalar": "receiver_amplitudes",
    "scalar_born": "sc_receiver_amplitudes",
    "acoustic": "receiver_amplitudes_p",
    "acoustic_born": "sc_receiver_amplitudes_p",
    "acoustic_vti": "receiver_amplitudes_x",
    "viscoacoustic": "receiver_amplitudes_p",
    "elastic": "receiver_amplitudes_p",
    "elastic_born": "sc_receiver_amplitudes_p",
    "elastic_vti": "receiver_amplitudes_p",
}
_OUTPUT_NAMES: dict[str, Callable[[int], list[str]]] = {
    "scalar": scalar_output_names,
    "scalar_born": scalar_born_output_names,
    "acoustic": acoustic_output_names,
    "acoustic_born": acoustic_born_output_names,
    "acoustic_vti": acoustic_vti_output_names,
    "viscoacoustic": viscoacoustic_output_names,
    "elastic": elastic_output_names,
    "elastic_born": elastic_born_output_names,
    "elastic_vti": elastic_vti_output_names,
}


def _survey(ndim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A single source and a single receiver, offset from each other."""
    source_amplitudes = (
        ricker(_FREQ, _NT, _DT, 1.5 / _FREQ).to(_DTYPE).reshape(1, 1, -1)
    )
    source_locations = torch.tensor([[[_N // 2] * ndim]])
    receiver_locations = torch.tensor([[[min(_N // 2 + 2, _N - 1)] * ndim]])
    return source_amplitudes, source_locations, receiver_locations


def _base_kwargs(prop: str, ndim: int, backend: Optional[str]) -> dict[str, Any]:
    """Keyword arguments common to every propagator: grid, survey, backend."""
    source_amplitudes, source_locations, receiver_locations = _survey(ndim)
    kwargs: dict[str, Any] = {
        "grid_spacing": _DX,
        "dt": _DT,
        "pml_width": 2,
        "accuracy": 2,
        "max_vel": _MAX_VEL,
        "backend": backend if backend else "auto",
    }
    # The receiver keyword is resolved from the spec rather than spelled out
    # here, so that the Born propagators' ``sc_`` spelling (and any future
    # change to it) cannot drift from what the propagators actually accept.
    spec = deepwave.spec(prop)
    # The source and receiver sides do not share a channel name: a source is
    # named for the quantity it injects and a receiver for the field it
    # records. ``acoustic_vti``'s scalar channels are both synthesised, so
    # its x velocity is used instead, to probe the kernel rather than the
    # wrapper.
    if prop == "acoustic_vti":
        source_channel = receiver_channel = "x"
    else:
        source_channel = spec.source_channels(ndim)[0]
        receiver_channel = spec.receiver_channels(ndim)[0]
    kwargs[spec.source_amplitudes_kwarg(source_channel)] = source_amplitudes
    kwargs[spec.source_locations_kwarg(source_channel)] = source_locations
    kwargs[spec.receiver_locations_kwarg(receiver_channel)] = receiver_locations
    if prop == "viscoacoustic":
        kwargs["q_freq"] = _FREQ
    return kwargs


def _models(prop: str, ndim: int) -> tuple[torch.Tensor, ...]:
    """The positional model tensors for ``prop`` at the given dimensionality.

    The first tensor returned is always the "primary" model (the one whose
    gradient/tangent/batch dimension the probes below vary) -- velocity for
    the acoustic-family propagators, the first Lame parameter for the
    elastic family, ``c11`` for the VTI stiffnesses.
    """
    shape = (_N,) * ndim
    v = torch.full(shape, 1500.0, dtype=_DTYPE)
    rho = torch.full(shape, 1000.0, dtype=_DTYPE)
    q = torch.full(shape, 30.0, dtype=_DTYPE)
    lamb = torch.full(shape, 4.0e9, dtype=_DTYPE)
    mu = torch.full(shape, 2.0e9, dtype=_DTYPE)
    buoyancy = torch.full(shape, 1.0 / 1000.0, dtype=_DTYPE)
    c11 = torch.full(shape, 9.0e9, dtype=_DTYPE)
    c13 = torch.full(shape, 3.0e9, dtype=_DTYPE)
    c33 = torch.full(shape, 9.0e9, dtype=_DTYPE)
    c55 = torch.full(shape, 2.0e9, dtype=_DTYPE)
    if prop in ("scalar", "scalar_born"):
        return (v,)
    if prop in ("acoustic", "acoustic_born"):
        return (v, rho)
    if prop == "acoustic_vti":
        return (c11, c13, c33, buoyancy)
    if prop == "viscoacoustic":
        return (v, rho, q)
    if prop in ("elastic", "elastic_born"):
        return (lamb, mu, buoyancy)
    if prop == "elastic_vti":
        return (c11, c13, c33, c55, buoyancy)
    raise DeepwaveValueError(f"unknown propagator {prop!r}")


def _extra_kwargs(prop: str, ndim: int) -> dict[str, Any]:
    """Keyword-only arguments a propagator needs beyond ``_base_kwargs``.

    Born perturbations (nonzero so derivative probes have a signal), the
    background-receiver locations the Born
    pairs require, and ``elastic_vti``'s
    ``c66`` (required in 3-D only; must stay ``None`` in 2-D).
    """
    shape = (_N,) * ndim
    zeros = torch.zeros(shape, dtype=_DTYPE)
    _, _, receiver_locations = _survey(ndim)
    if prop == "scalar_born":
        return {
            "dv": torch.full(shape, 20.0, dtype=_DTYPE),
            "bg_receiver_locations": receiver_locations,
        }
    if prop == "acoustic_born":
        return {
            "dv": torch.full(shape, 20.0, dtype=_DTYPE),
            "drho": zeros,
            "bg_receiver_locations_p": receiver_locations,
        }
    if prop == "elastic_born":
        return {
            "dlamb": torch.full(shape, 1e8, dtype=_DTYPE),
            "dmu": torch.full(shape, 5e7, dtype=_DTYPE),
            "dbuoyancy": zeros,
            "bg_receiver_locations_p": receiver_locations,
        }
    if prop == "elastic_vti" and ndim == 3:
        return {"c66": torch.full(shape, 2.0e9, dtype=_DTYPE)}
    return {}


def _receiver_index(prop: str, ndim: int) -> int:
    """The index of ``prop``'s primary receiver output in its output tuple."""
    return _OUTPUT_NAMES[prop](ndim).index(_REC_NAME[prop])


def _call(
    prop: str,
    ndim: int,
    backend: Optional[str],
    primary_model: Optional[torch.Tensor] = None,
    **overrides: Any,
) -> tuple[torch.Tensor, ...]:
    """Run ``prop`` on a tiny problem, returning its full output tuple.

    ``primary_model`` substitutes for the first positional model tensor
    (letting callers vary/batch/differentiate it); any other keyword in
    ``overrides`` is layered on top of the base and extra keyword arguments
    (e.g. to set ``mpml_fraction``, ``boundary``, ``storage_mode``, a callback).
    """
    fn = getattr(deepwave, prop)
    models_override = overrides.pop("_models_override", None)
    models = tuple(models_override) if models_override else _models(prop, ndim)
    if primary_model is not None:
        models = (primary_model, *models[1:])
    kwargs = _base_kwargs(prop, ndim, backend)
    kwargs.update(_extra_kwargs(prop, ndim))
    kwargs.update(overrides)
    result: tuple[torch.Tensor, ...] = fn(*models, **kwargs)
    return result


def _receiver(
    prop: str,
    ndim: int,
    backend: Optional[str],
    primary_model: Optional[torch.Tensor] = None,
    **overrides: Any,
) -> torch.Tensor:
    """Like :func:`_call`, but returns only the primary receiver output."""
    out = _call(prop, ndim, backend, primary_model, **overrides)
    return out[_receiver_index(prop, ndim)]


def _succeeds(fn: Callable[[], Any]) -> bool:
    """Observe a conformance call, never to determine the published contract.

    The caller compares the result with a declaration. An unexpected failure
    or nonfinite returned tensor of a promised case therefore fails testing
    instead of shrinking support.
    """

    def finite(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return not (value.is_floating_point() or value.is_complex()) or bool(
                torch.isfinite(value).all()
            )
        if isinstance(value, (tuple, list)):
            return all(finite(item) for item in value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        return True

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fn()
    except Exception:  # noqa: BLE001 - see docstring
        return False
    return finite(result)


def _probe_dims(name: str) -> tuple[int, ...]:
    """Which of 1-D/2-D/3-D ``name`` accepts on its default backend."""
    supported = []
    for ndim in (1, 2, 3):

        def call(ndim: int = ndim) -> torch.Tensor:
            return _receiver(name, ndim, None)

        if _succeeds(call):
            supported.append(ndim)
    return tuple(supported)


def _has_compiled_backend(name: str) -> bool:
    """Whether ``name`` has a compiled C/CUDA backend at all.

    Cheap and exact: looks the compiled symbol up in the loaded shared
    library (the same check ``tests/helpers.py``'s ``c_backend_available``
    performs), rather than calling the propagator.
    """
    try:
        _backend_utils.get_backend_function(
            name, 2, "forward", 2, torch.float64, torch.device("cpu")
        )
    except AttributeError:
        return False
    return True


def _probe_backends(name: str) -> str:
    """Probe supported backend values, including torch.compile execution."""
    has_compiled = _has_compiled_backend(name)

    def call() -> torch.Tensor:
        return _receiver(name, 2, "torch_compile")

    compile_ok = _succeeds(call)
    if has_compiled and compile_ok:
        return "``auto`` (default), ``compiled``, ``eager``, ``torch_compile``"
    if compile_ok:
        return (
            "``auto`` (default, resolves to ``eager``), ``eager``, "
            "``torch_compile`` (no compiled C/CUDA backend; "
            "``compiled`` raises NotImplementedError)"
        )
    if has_compiled:
        return (
            "``auto`` (default), ``compiled``, ``eager`` "
            "(``torch_compile`` raises NotImplementedError)"
        )
    return (
        "``auto`` (default, resolves to ``eager``), ``eager`` only "
        "(``compiled``/``torch_compile`` raise NotImplementedError)"
    )


#: The ``boundary`` configurations probed for the "Boundary faces accepted"
#: column, as ``(label, boundary, zero_pml_width)``: each face type on its own
#: and then one mixed per-face combination, since accepting each type
#: individually is not by itself evidence that they can be combined (the
#: previous version of this probe tried only the mixed combination, and
#: reported it as support for every type). An explicit non-PML face is probed
#: with ``pml_width=0`` everywhere -- an explicit face and an absorbing layer
#: on the same face would be contradictory -- while ``pml`` keeps the probes'
#: usual width, since a zero-width PML absorbs nothing and would not exercise
#: the absorbing path this entry is about.
#:
#: ``free`` and ``rigid`` are each probed twice, on every face and on the low
#: faces alone (the beginning of each axis, with the high faces left as
#: zero-width ``pml``), because which of the two a propagator accepts is
#: exactly the distinction :func:`deepwave._common.check_boundary_supported`
#: makes: a face whose guard-ring fill conserves a discrete energy on the low
#: side of an axis need not conserve on the high side. The probe uses the
#: default ``accuracy``, and the distinction only exists from ``accuracy=4``
#: upwards, so :doc:`/boundaries` rather than this column is the full rule.
_BOUNDARY_PROBES: tuple[tuple[str, list[str], bool], ...] = (
    ("free on every face", ["free"] * 4, True),
    ("free on the low faces", ["free", "pml", "free", "pml"], True),
    ("rigid on every face", ["rigid"] * 4, True),
    ("rigid on the low faces", ["rigid", "pml", "rigid", "pml"], True),
    ("periodic", ["periodic"] * 4, True),
    ("pml", ["pml"] * 4, False),
    ("mixed per-face combination", ["free", "pml", "rigid", "pml"], True),
)


def _probe_boundary_faces(name: str) -> str:
    """Which boundary faces ``name`` accepts and runs, per backend.

    Each face type in :data:`_BOUNDARY_PROBES` is called separately, on both
    the default and the ``eager`` backend, and the backends that ran it are
    joined with ``" + "`` exactly as :func:`_probe_second_order` and
    :func:`_probe_jvp` do -- a face type accepted by the compiled kernel but
    not by the pure-Python one (or the reverse) is a real difference this
    column previously could not express, since it probed one backend only.

    "Accepted" is all this cell can honestly claim: the probe runs a handful
    of time steps on a 10-cell grid and records whether the call raised, so
    it establishes that the configuration is implemented and runs, not that
    it is long-time stable or accurate (see the "Boundary faces accepted"
    section of ``docs/capabilities.rst``, and :doc:`/boundaries` for what
    each face means and how well it reflects).
    """
    backends: dict[str, list[str]] = {}
    for backend, label in ((None, "default"), ("eager", "eager")):
        for face, boundary, zero_pml_width in _BOUNDARY_PROBES:
            # Explicitly at the propagators' own default accuracy, rather
            # than at the probes' usual 2: which side of an axis a face is
            # accepted on depends on the accuracy, and the default is the
            # answer this column should give.
            overrides: dict[str, Any] = {"boundary": boundary, "accuracy": 4}
            if zero_pml_width:
                overrides["pml_width"] = [0, 0, 0, 0]

            def call(
                backend: Optional[str] = backend,
                overrides: dict[str, Any] = overrides,
            ) -> torch.Tensor:
                return _receiver(name, 2, backend, **overrides)

            if _succeeds(call):
                backends.setdefault(face, []).append(label)
    # Faces supported on the same backends are reported together, so the
    # common case (every face on every backend) stays a single short cell
    # rather than five repetitive ones.
    groups: dict[tuple[str, ...], list[str]] = {}
    for face, _, _ in _BOUNDARY_PROBES:
        groups.setdefault(tuple(backends.get(face, ())), []).append(face)
    parts = []
    for labels, faces in groups.items():
        joined = ", ".join(faces)
        parts.append(
            f"{joined} ({' + '.join(labels)})" if labels else f"{joined} (not accepted)"
        )
    return "; ".join(parts)


def _probe_mpml_fraction(name: str) -> str:
    """Whether ``name`` accepts ``mpml_fraction``, and on which backend(s).

    ``mpml_fraction`` being in the signature is necessary but not sufficient: on
    ``elastic_born`` (found while probing, not assumed) the compiled M-PML
    *forward* kernel is not implemented yet, so ``mpml_fraction >
    0`` raises ``NotImplementedError`` on the default backend and only
    ``backend="eager"`` actually applies it -- a real
    backend-availability gap the plain "is it a keyword" signature check
    would have missed entirely.
    """
    if "mpml_fraction" not in _get_spec(getattr(deepwave, name)).supported_options:
        return "no"
    backends_ok = []
    for backend, label in ((None, "default"), ("eager", "eager")):

        def call(backend: Optional[str] = backend) -> torch.Tensor:
            return _receiver(name, 2, backend, mpml_fraction=0.1)

        if _succeeds(call):
            backends_ok.append(label)
    if set(backends_ok) == {"default", "eager"}:
        return "yes (all backends)"
    if backends_ok:
        return "yes (" + " + ".join(backends_ok) + " only)"
    return "yes (signature only; rejected at runtime on every backend)"


def _vacuum_models(name: str, ndim: int) -> list[torch.Tensor]:
    """Model overrides that put a vacuum (free-surface) layer in the model.

    The elastic family's free surface is implied by cells with
    ``buoyancy = mu = 0``; ``surface_method`` selects how those cells are
    discretised, so a probe of it needs some.
    """
    models = list(_models(name, ndim))
    # elastic's positional models are (lamb, mu, buoyancy). A vacuum cell is
    # one where *all three* vanish: zeroing only mu and buoyancy would leave
    # an acoustic (fluid) region, which "pm" deliberately refuses.
    for index in (0, 1, 2):
        model = models[index].clone()
        model[0] = 0.0
        models[index] = model
    return models


def _probe_surface_method(name: str) -> str:
    """Whether ``name`` implements a non-default ``surface_method``.

    Probes *behaviour*, not the signature. Every propagator now declares
    every shared option, so that a generic wrapper can forward one uniform
    keyword set; "is it a keyword" would therefore report universal support,
    which would be a lie this module publishes. What actually distinguishes
    the propagators is whether the non-default value ``"pm"`` is accepted,
    and on which backends -- exactly as for ``mpml_fraction``.
    """
    if "surface_method" not in _get_spec(getattr(deepwave, name)).supported_options:
        return "no"
    # ``surface_method="pm"`` is only meaningful where there are vacuum cells
    # (a zero-mu region standing for the air above a free surface), and
    # raises with a clear message when there are none. The base probe model
    # is uniform, so the probe supplies its own vacuum layer -- otherwise
    # this would report "rejected at runtime" for the one propagator that
    # implements the option, and publish that.
    backends_ok = []
    for backend, label in ((None, "default"), ("eager", "eager")):

        def call(backend: Optional[str] = backend) -> torch.Tensor:
            return _receiver(
                name,
                2,
                backend,
                surface_method="pm",
                _models_override=_vacuum_models(name, 2),
            )

        if _succeeds(call):
            backends_ok.append(label)
    if set(backends_ok) == {"default", "eager"}:
        return 'yes ("pm", all backends)'
    if backends_ok:
        return 'yes ("pm", ' + " + ".join(backends_ok) + " only)"
    return 'yes (signature only; "pm" rejected at runtime on every backend)'


def _probe_q_options(name: str) -> str:
    """Whether ``name`` implements the constant-Q GSLS options.

    Behavioural, for the same reason as :func:`_probe_surface_method`:
    only viscoacoustic declares and implements ``q_freq``.
    """
    if "q_freq" not in _get_spec(getattr(deepwave, name)).supported_options:
        return "no"
    backends_ok = []
    for backend, label in ((None, "default"), ("eager", "eager")):

        def call(backend: Optional[str] = backend) -> torch.Tensor:
            return _receiver(name, 2, backend, q_freq=25.0)

        if _succeeds(call):
            backends_ok.append(label)
    if set(backends_ok) == {"default", "eager"}:
        return "yes (all backends)"
    if backends_ok:
        return "yes (" + " + ".join(backends_ok) + " only)"
    return "yes (signature only; rejected at runtime on every backend)"


def _probe_storage_modes(name: str) -> tuple[str, ...]:
    """Which ``storage_mode`` values ``name`` accepts on its default backend."""
    with tempfile.TemporaryDirectory() as storage_path:

        def try_mode(mode: str, storage_path: str = storage_path) -> bool:
            overrides: dict[str, Any] = {"storage_mode": mode}
            if mode == "disk":
                overrides["storage_path"] = storage_path
            return _succeeds(lambda: _receiver(name, 2, None, **overrides))

        return tuple(
            mode for mode in ("device", "cpu", "disk", "none") if try_mode(mode)
        )


def _python_backend_storage_modes(name: str) -> tuple[str, ...]:
    """Which ``storage_mode`` values ``name``'s pure-Python backends accept.

    Read directly from :attr:`deepwave._engine.PropagatorSpec.
    python_backend_storage_modes` -- the exact field
    :func:`deepwave._engine.reject_unsupported_python_backend_options` uses
    to accept or reject a ``storage_mode`` on ``backend="eager"``/
    ``"torch_compile"`` -- rather than probed, so this can never disagree with the
    validator that actually enforces it.
    """
    spec = _get_spec(getattr(deepwave, name))
    return spec.python_backend_storage_modes


def _probe_batching(name: str) -> str:
    """Whether ``vmap`` over the primary model works, and on which backend."""
    primary = _models(name, 2)[0]
    batch = torch.stack([primary, primary * 1.001, primary * 1.002])
    for backend, label in (("eager", "eager"), (None, "default")):

        def f(v: torch.Tensor, backend: Optional[str] = backend) -> torch.Tensor:
            return _receiver(name, 2, backend, v)

        def call(f: Callable[[torch.Tensor], torch.Tensor] = f) -> torch.Tensor:
            return cast("torch.Tensor", vmap(f)(batch))

        if _succeeds(call):
            return f"yes, over property models ({label} backend)"
    return "not supported"


def _probe_second_order(name: str) -> str:
    """Whether double-backward works, including the compiled stride limit.

    A single time step is not enough to describe this capability: scalar,
    acoustic and elastic support compiled double backward only when the
    effective model-gradient sampling stride is one. Probe a CFL-stable step
    and the usual larger step separately so the table does not mistake a
    conditional capability for either unconditional support or no support.
    """
    supported = []
    for backend, label in ((None, "default"), ("eager", "eager")):
        primary = _models(name, 2)[0].clone().requires_grad_(True)

        def run(
            dt: float,
            backend: Optional[str] = backend,
            primary: torch.Tensor = primary,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = _receiver(name, 2, backend, primary, dt=dt, pml_freq=_FREQ)
            (grad,) = torch.autograd.grad((out**2).sum(), primary, create_graph=True)
            (second,) = torch.autograd.grad(
                (grad * torch.ones_like(grad)).sum(), primary
            )
            return out, grad, second

        if _succeeds(lambda: run(_DT / 5)):
            if not _succeeds(lambda: run(_DT)):
                supported.append(f"{label} (effective stride 1)")
            else:
                supported.append(label)
    if not supported:
        return "not supported"
    return " + ".join(supported)


def _probe_jvp(name: str) -> str:
    """Whether a nonzero ``torch.func.jvp`` works, per backend."""
    supported = []
    for backend, label in ((None, "default"), ("eager", "eager")):
        primary = _models(name, 2)[0]
        tangent = torch.ones_like(primary)

        def f(v: torch.Tensor, backend: Optional[str] = backend) -> torch.Tensor:
            return _receiver(name, 2, backend, v)

        def call(
            f: Callable[[torch.Tensor], torch.Tensor] = f,
            primary: torch.Tensor = primary,
            tangent: torch.Tensor = tangent,
        ) -> Any:
            result = jvp(f, (primary,), (tangent,))
            primal, derivative = result[0], result[1]
            if (
                not bool(torch.isfinite(primal).all())
                or not bool(torch.isfinite(derivative).all())
                or not bool((derivative != 0).any())
            ):
                raise AssertionError("JVP probe must have a finite nonzero derivative")
            return primal, derivative

        if _succeeds(call):
            supported.append(label)
    if not supported:
        return "not supported"
    return " + ".join(supported)


def _probe_forward_callback(name: str) -> bool:
    """Whether a ``forward_callback`` is accepted and actually invoked."""
    fired: list[int] = []

    def cb(state: Any) -> None:
        fired.append(state.step)

    if not _succeeds(lambda: _call(name, 2, None, forward_callback=cb)):
        return False
    return len(fired) > 0


def _probe_backward_callback(name: str) -> tuple[str, bool]:
    """Which backend(s) accept ``backward_callback``, and role="forward".

    Returns ``(backends, role_forward)`` where ``backends`` names which of
    "default"/"eager" fire the callback during ``.backward()`` (mirrors
    :func:`_probe_second_order`'s pattern), and ``role_forward`` is whether
    ``role="forward"`` is usable from inside that callback -- a narrower
    capability documented at :class:`deepwave.CallbackState`, currently only
    provided by ``scalar``.

    Retaining the forward wavefield is opt-in, so the probe must ask for it.
    Only propagators whose spec reports ``has_forward_wavefield`` expose the
    opt-in argument. The probe asks the spec first so the ordinary backward
    callback capability is tested independently of forward-field retention.
    """
    supported = []
    role_forward = False
    opt_in = _get_spec(getattr(deepwave, name)).has_forward_wavefield
    extra: dict[str, Any] = (
        {"backward_callback_needs_forward_wavefield": True} if opt_in else {}
    )
    for backend, label in ((None, "default"), ("eager", "eager")):
        fired: list[int] = []
        role_results: list[bool] = []

        def cb(
            state: Any,
            fired: list[int] = fired,
            role_results: list[bool] = role_results,
        ) -> None:
            fired.append(state.step)
            try:
                # The exact wavefield name does not matter: a supported
                # role="forward" either finds it (no exception) or raises
                # KeyError (found the capability, not this placeholder
                # name); only the *absence* of the capability raises
                # NotImplementedError (see CallbackState.get_wavefield).
                state.get_wavefield("__deepwave_capabilities_probe__", role="forward")
                role_results.append(True)
            except KeyError:
                role_results.append(True)
            except NotImplementedError:
                role_results.append(False)

        primary = _models(name, 2)[0].clone().requires_grad_(True)

        def run(
            backend: Optional[str] = backend,
            cb: Callable[[Any], None] = cb,
            primary: torch.Tensor = primary,
            extra: dict[str, Any] = extra,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = _receiver(name, 2, backend, primary, backward_callback=cb, **extra)
            loss = out.sum()
            loss.backward()  # type: ignore[no-untyped-call]
            if primary.grad is None:
                raise AssertionError("backward probe did not produce a model gradient")
            return out, primary.grad

        if _succeeds(run) and fired:
            supported.append(label)
            if any(role_results):
                role_forward = True
    if not supported:
        return "not supported", False
    return " + ".join(supported), role_forward


def _propagator_status(name: str) -> str:
    """``name``'s declared stability tier (see :doc:`/stability`).

    Read directly from :attr:`deepwave._engine.PropagatorSpec.status`, the
    one place the label is written down, rather than inferred from whether a
    call happened to warn. No call warns about its maturity: the tier is a
    documentation label, and the interface promise is the same on all
    nine propagators. An earlier version of this function made a plain
    call and looked for the substring "experimental" in whatever warnings
    came back, which was indirect even while such a warning existed -- a
    warning raised for an unrelated reason that merely used the word would
    have been read as a tier.
    """
    spec = _get_spec(getattr(deepwave, name))
    return spec.status


@dataclasses.dataclass(frozen=True)
class PropagatorCapabilities:
    """One declared or independently observed capability row."""

    name: str
    status: str
    dims: tuple[int, ...]
    backends: str
    boundary_faces: str
    storage_modes: tuple[str, ...]
    python_backend_storage_modes: tuple[str, ...]
    mpml_fraction: str
    surface_method: str
    q_options: str
    batching: str
    second_order: str
    jvp: str
    forward_callback: bool
    backward_callback: str
    role_forward: bool


_BOUNDARY_FACES = {
    "scalar": (
        "free on every face, free on the low faces, rigid on every face, "
        "rigid on the low faces, periodic, pml, mixed per-face combination "
        "(default + eager)"
    ),
    "acoustic": (
        "free on every face, periodic (not accepted); free on the low faces, "
        "rigid on every face, rigid on the low faces, pml, mixed per-face "
        "combination (default + eager)"
    ),
    "acoustic_vti": (
        "free on every face, free on the low faces, periodic, mixed per-face "
        "combination (not accepted); rigid on every face, rigid on the low "
        "faces, pml (default + eager)"
    ),
    "elastic": (
        "free on every face, free on the low faces, rigid on every face, "
        "periodic, mixed per-face combination (not accepted); rigid on the "
        "low faces, pml (default + eager)"
    ),
}


def declared_capabilities(
    names: tuple[str, ...] = PROPAGATOR_NAMES,
) -> list[PropagatorCapabilities]:
    """Return supported contracts without running a propagator or loading a DLL."""
    backends = "``auto`` (default), ``compiled``, ``eager``, ``torch_compile``"
    storage = ("device", "cpu", "disk", "none")
    rows = []
    for name in names:
        spec = propagator_spec(name)
        boundary_family = (
            "scalar"
            if name in ("scalar", "scalar_born")
            else "acoustic"
            if name in ("acoustic", "acoustic_born", "viscoacoustic")
            else "acoustic_vti"
            if name == "acoustic_vti"
            else "elastic"
        )
        compiled_higher = name in ("scalar", "acoustic", "elastic")
        supports_batching = name not in ("acoustic_born", "elastic_born")
        rows.append(
            PropagatorCapabilities(
                name=name,
                status=spec.status,
                dims=spec.ndims,
                backends=backends,
                boundary_faces=_BOUNDARY_FACES[boundary_family],
                storage_modes=storage,
                python_backend_storage_modes=spec.python_backend_storage_modes,
                mpml_fraction=(
                    "yes (all backends)"
                    if "mpml_fraction" in spec.supported_options
                    else "no"
                ),
                surface_method=(
                    'yes ("pm", all backends)'
                    if "surface_method" in spec.supported_options
                    else "no"
                ),
                q_options=(
                    "yes (all backends)" if "q_freq" in spec.supported_options else "no"
                ),
                batching=(
                    "yes, over property models (eager backend)"
                    if supports_batching
                    else "not supported"
                ),
                second_order=(
                    "default (effective stride 1) + eager"
                    if compiled_higher
                    else "eager"
                ),
                jvp="default + eager" if compiled_higher else "eager",
                forward_callback=True,
                backward_callback=(
                    "default + eager"
                    if spec.python_backend_backward_callback
                    else "default"
                ),
                role_forward=spec.has_forward_wavefield,
            )
        )
    return rows


def probe_capabilities(
    names: tuple[str, ...] = PROPAGATOR_NAMES,
) -> list[PropagatorCapabilities]:
    """Observe every propagator in ``names`` for declaration conformance."""
    rows = []
    for name in names:
        backward_callback, role_forward = _probe_backward_callback(name)
        rows.append(
            PropagatorCapabilities(
                name=name,
                status=_propagator_status(name),
                dims=_probe_dims(name),
                backends=_probe_backends(name),
                boundary_faces=_probe_boundary_faces(name),
                storage_modes=_probe_storage_modes(name),
                python_backend_storage_modes=_python_backend_storage_modes(name),
                mpml_fraction=_probe_mpml_fraction(name),
                surface_method=_probe_surface_method(name),
                q_options=_probe_q_options(name),
                batching=_probe_batching(name),
                second_order=_probe_second_order(name),
                jvp=_probe_jvp(name),
                forward_callback=_probe_forward_callback(name),
                backward_callback=backward_callback,
                role_forward=role_forward,
            )
        )
    return rows


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def _list_table(title: str, header: list[str], body: list[list[str]]) -> list[str]:
    """Renders one Sphinx ``list-table`` directive as a list of RST lines."""
    lines = [title, "-" * len(title), "", ".. list-table::", "   :header-rows: 1", ""]
    lines.append("   * - " + "\n     - ".join(header))
    lines.extend("   * - " + "\n     - ".join(row) for row in body)
    lines.append("")
    return lines


def _render_storage_modes(r: PropagatorCapabilities) -> str:
    """Renders the per-backend storage-mode cell for the scope/backends table.

    ``storage_mode`` support differs by backend for nearly every propagator
    (the default ``"auto"`` backend typically resolves to the compiled
    C/CUDA kernel, which supports all four modes; ``"eager"``/
    ``"torch_compile"`` share one narrower set, declared by
    ``PropagatorSpec.python_backend_storage_modes``), so a single unqualified
    list of modes would be wrong for whichever backend it did not describe.
    Naming the backend(s) in the cell keeps every cell true regardless. A
    propagator with no compiled C/CUDA backend at all (see
    :func:`_has_compiled_backend`) has no ``compiled`` entry to show: its
    ``backend="auto"`` resolves *silently* to the same pure-Python backends
    already named (``backend="compiled"`` instead raises
    ``NotImplementedError`` -- it does not resolve to anything), so a
    separate ``compiled`` entry would claim a distinct backend that does not
    exist -- the same class of wrong cell this function exists to avoid, and
    the "Backends" column already says so for these propagators.
    """
    python = ", ".join(r.python_backend_storage_modes)
    if not propagator_spec(r.name).has_compiled_backend:
        return f"``eager``/``torch_compile``: {python}"
    compiled = ", ".join(r.storage_modes)
    return f"``compiled``: {compiled}; ``eager``/``torch_compile``: {python}"


def capabilities_rst(rows: Optional[list[PropagatorCapabilities]] = None) -> str:
    """Renders the full generated capability matrix as RST source.

    Three tables (scope/backends, differentiation, callbacks) rather than one
    wide one, purely for readability -- the underlying data is the single
    ``PropagatorCapabilities`` list either way.
    """
    if rows is None:
        rows = declared_capabilities()
    lines: list[str] = []
    lines += _list_table(
        "Scope and backends",
        [
            "Propagator",
            "Status",
            "Dims",
            "Backends",
            "Boundary faces accepted",
            "Storage modes",
        ],
        [
            [
                f"``{r.name}``",
                r.status,
                ", ".join(f"{d}-D" for d in r.dims),
                r.backends,
                r.boundary_faces,
                _render_storage_modes(r),
            ]
            for r in rows
        ],
    )
    lines += _list_table(
        "Physics options and differentiation",
        [
            "Propagator",
            "mpml_fraction",
            "surface_method",
            "Q options",
            "Batching (vmap)",
            "Second-order (create_graph)",
            "jvp",
        ],
        [
            [
                f"``{r.name}``",
                r.mpml_fraction,
                r.surface_method,
                r.q_options,
                r.batching,
                r.second_order,
                r.jvp,
            ]
            for r in rows
        ],
    )
    lines += _list_table(
        "Callbacks",
        [
            "Propagator",
            "forward_callback",
            "backward_callback",
            'get_wavefield(role="forward")',
        ],
        [
            [
                f"``{r.name}``",
                _yn(r.forward_callback),
                r.backward_callback,
                _yn(r.role_forward),
            ]
            for r in rows
        ],
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The internal callback names (docs/callback_names.rst).
#
# Declared, not probed: read from each propagator's PropagatorSpec name
# callables, which are the same callables the engine and the eager backends
# zip the live tensors against when they build a CallbackState (see
# deepwave._engine's build_state). Nothing here is a hand-copied name, so the
# Born propagators' ``<field>_sc_0`` spelling, elastic_born's ``d<model>``
# perturbations appear -- or change -- here automatically.
# ---------------------------------------------------------------------------


def _spec_n_mechanisms(spec: PropagatorSpec) -> Optional[int]:
    """The ``n_mechanisms`` the spec's name callables default to, if any.

    Viscoacoustic carries one memory field per relaxation mechanism. Read
    its callable's default so generated tables match the propagation default.

    Returns:
        The default ``n_mechanisms``, or ``None`` for a propagator whose
        names do not depend on it.
    """
    parameter = inspect.signature(spec.wavefield_names).parameters.get("n_mechanisms")
    if parameter is None:
        return None
    default = parameter.default
    if not isinstance(default, int):  # pragma: no cover - static
        raise AssertionError(  # noqa: TRY004 - a spec bug, not a caller's type
            f"{spec.name}'s wavefield_names has no default n_mechanisms"
        )
    return default


@dataclasses.dataclass(frozen=True)
class PropagatorInternalNames:
    """One propagator's internal callback names at one dimensionality.

    ``models`` are the names :meth:`deepwave.CallbackState.get_model` accepts,
    and are also the gradient-accumulator names
    :meth:`deepwave.CallbackState.get_gradient` accepts; ``wavefields`` are
    the names :meth:`deepwave.CallbackState.get_wavefield` accepts;
    ``auxiliary`` is the subset of ``wavefields`` that the public interface
    folds into one opaque ``auxiliary_state`` field -- the absorbing
    boundary's memory variables and the attenuating propagators' constant-Q
    memory.
    """

    name: str
    ndim: int
    models: tuple[str, ...]
    wavefields: tuple[str, ...]
    auxiliary: tuple[str, ...]
    n_mechanisms: Optional[int]


def internal_names(
    names: tuple[str, ...] = PROPAGATOR_NAMES,
) -> list[PropagatorInternalNames]:
    """The internal callback names of every propagator, per dimensionality.

    One row per ``(propagator, ndim)`` pair that the propagator declares in
    ``PropagatorSpec.ndims``, because several of the names are
    dimensionality-dependent (``elastic`` steps ``m_vxx_0`` in 1-D and adds
    ``m_vyy_0``/``m_vyx_0``/``m_vxy_0`` in 2-D, and ``mu_yx`` appears among
    its models only from 2-D on).

    ``auxiliary_state_names`` is declared in the *public* spelling, since it
    names a subset of the public state fields, and the two spellings differ
    for the attenuating propagators' constant-Q memory (public ``r_mech0``,
    internal ``r_0``). It is therefore translated through
    ``PropagatorSpec.internal_state_name_map`` before being reported here --
    a callback is handed the internal spelling -- and is returned in
    ``wavefield_names`` order rather than its own declaration order, so that
    the table's two wavefield columns read down in step.
    """
    rows = []
    for name in names:
        spec = propagator_spec(name)
        for ndim in spec.ndims:
            wavefields = tuple(spec.wavefield_names(ndim))
            to_internal = spec.internal_state_name_map(ndim)
            auxiliary = {
                to_internal.get(public, public)
                for public in spec.auxiliary_state_names(ndim)
            }
            missing = auxiliary - set(wavefields)
            if missing:  # pragma: no cover - guards a spec inconsistency
                raise AssertionError(
                    f"{name} ({ndim}-D) declares auxiliary state names "
                    f"{sorted(missing)} that are not among its internal "
                    f"wavefield names {list(wavefields)}."
                )
            rows.append(
                PropagatorInternalNames(
                    name=name,
                    ndim=ndim,
                    models=tuple(spec.model_names(ndim)),
                    wavefields=wavefields,
                    auxiliary=tuple(n for n in wavefields if n in auxiliary),
                    n_mechanisms=_spec_n_mechanisms(spec),
                )
            )
    return rows


def _names_cell(names: tuple[str, ...]) -> str:
    """Renders one cell listing ``names`` as inline literals."""
    if not names:
        return "(none)"
    return ", ".join(f"``{name}``" for name in names)


def _dims_cell(row: PropagatorInternalNames) -> str:
    """Renders the dimensionality cell, naming ``q_n_mechanisms`` where it bites.

    The viscoacoustic rows are true only for a particular number of relaxation
    mechanisms, so their cell says which; the other families
    have nothing to qualify. Both halves are derived (see
    :func:`_spec_n_mechanisms`), so a propagator that gained or lost the
    dependence would change this cell rather than leaving a stale note.
    """
    if row.n_mechanisms is None:
        return f"{row.ndim}-D"
    return f"{row.ndim}-D, ``q_n_mechanisms={row.n_mechanisms}``"


def callback_names_rst(
    rows: Optional[list[PropagatorInternalNames]] = None,
) -> str:
    """Renders the internal-callback-name table as RST source.

    One table rather than one per propagator: the whole point of the page is
    that the nine can be compared and that a name can be looked up without
    first knowing which propagator family it belongs to.
    """
    if rows is None:
        rows = internal_names()
    lines = _list_table(
        "Internal names, per propagator",
        [
            "Propagator",
            "Dimensionality",
            "Models (``get_model``, ``get_gradient``)",
            "Wavefields (``get_wavefield``)",
            "Of which absorbing-boundary/attenuation auxiliary",
        ],
        [
            [
                f"``{row.name}``",
                _dims_cell(row),
                _names_cell(row.models),
                _names_cell(row.wavefields),
                _names_cell(row.auxiliary),
            ]
            for row in rows
        ],
    )
    return "\n".join(lines) + "\n"

"""Tests for the ``backend`` parameter's value set and resolution.

``backend`` is ``str``, not a closed ``Literal``: a closed set of
implementation technologies would be a promise to keep those technologies.
The documented values are ``"auto"`` (the default), ``"compiled"``,
``"eager"`` and ``"torch_compile"``.

The distinction this file exists to pin is between ``"auto"`` and
``"compiled"``: all nine retained propagators have a compiled kernel and
resolve both values to that backend for supported configurations.
``"auto"`` may choose eager for a supported option without a compiled
implementation (for example ``elastic_born`` with ``mpml_fraction > 0``);
``"compiled"`` refuses that option instead.

``deepwave.wavelets`` and the deprecated ``python_backend`` alias are covered
in ``tests/test_python_backend.py``.
"""

import ast
import inspect
import pathlib
import warnings

import pytest
import torch

import deepwave
import deepwave._capabilities as capabilities
import deepwave._engine
from deepwave._capabilities import PROPAGATOR_NAMES

_MODULES = {
    "scalar": "Scalar",
    "scalar_born": "ScalarBorn",
    "acoustic": "Acoustic",
    "acoustic_born": "AcousticBorn",
    "acoustic_vti": "AcousticVti",
    "elastic": "Elastic",
    "elastic_born": "ElasticBorn",
    "elastic_vti": "ElasticVti",
    "viscoacoustic": "Viscoacoustic",
}


def _assert_outputs_equal(left: tuple, right: tuple, label: str) -> None:
    """Compares two propagator outputs field by field, bit for bit.

    The output tuple carries a :class:`AuxiliaryState` among its tensors,
    which ``torch.equal`` cannot take directly, so the
    auxiliary state is compared entry by entry.
    """
    assert len(left) == len(right), label
    for i, (a, b) in enumerate(zip(left, right)):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f"{label}: output {i} differs"
        else:
            a_items, b_items = dict(a.items()), dict(b.items())
            assert a_items.keys() == b_items.keys(), f"{label}: state keys differ"
            for key, value in a_items.items():
                assert torch.equal(value, b_items[key]), (
                    f"{label}: auxiliary_state[{key!r}] differs"
                )


def _ndim(name: str) -> int:
    """The smallest number of dimensions this propagator supports."""
    return capabilities._probe_dims(name)[0]


def _quiet_call(name: str, backend: str = "auto", **overrides: object) -> tuple:
    """A tiny run of ``name``, with the experimental-tier warning filtered."""
    with warnings.catch_warnings():
        return capabilities._call(name, _ndim(name), backend, **overrides)


# ---------------------------------------------------------------------------
# The parameter itself
# ---------------------------------------------------------------------------


def test_backend_auto_is_default_on_all_entry_points() -> None:
    """All 18 retained entry points declare ``backend: str = "auto"``."""
    entry_points = [(name, getattr(deepwave, name)) for name in PROPAGATOR_NAMES] + [
        (f"{cls}.forward", getattr(deepwave, cls).forward) for cls in _MODULES.values()
    ]
    assert len(entry_points) == 18
    for label, fn in entry_points:
        parameter = inspect.signature(fn).parameters["backend"]
        assert parameter.default == "auto", label
        assert parameter.annotation is str, label


def test_backend_unknown_value_raises() -> None:
    """An unknown value raises, listing every documented value."""
    with pytest.raises(ValueError, match="backend must be one of") as excinfo:
        _quiet_call("scalar", backend="turbo")
    message = str(excinfo.value)
    for value in ("auto", "compiled", "eager", "torch_compile"):
        assert repr(value) in message, value
    # The builtin mixin means an existing ``except ValueError`` still
    # catches it.
    assert isinstance(excinfo.value, deepwave.DeepwaveValueError)


def test_compile_is_not_a_backend_spelling() -> None:
    """``backend="compile"`` is rejected, naming the value that replaces it.

    ``"compile"`` is the internal execution mode's name and ``v0.0.27``'s
    ``python_backend`` value; it was never a ``backend`` value in a release,
    so ``backend`` takes only ``"torch_compile"``.
    """
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _quiet_call("scalar", backend="compile")
    assert "'torch_compile'" in str(excinfo.value)


@pytest.mark.parametrize("spelling", ["jit", "torch_jit"])
def test_jit_is_not_a_backend_spelling(spelling: str) -> None:
    """Neither ``torch.jit`` spelling is accepted, and neither is offered.

    Deepwave does not call ``torch.jit`` at all, so there is no execution
    mode for these to name. The rejection lists the values that do exist and
    mentions no ``jit`` spelling, since a message that offers one would send
    the reader back to a value that raises.
    """
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _quiet_call("scalar", backend=spelling)
    message = str(excinfo.value)
    assert "jit" not in message.replace(f"{spelling!r}", "")
    assert "'torch_compile'" in message


def test_deepwave_never_calls_torch_jit() -> None:
    """No ``torch.jit`` use remains anywhere in the package's code.

    The ``torch.jit.script`` execution mode is gone, and so are the two
    ``@torch.jit.script`` decorators the boundary-fill helpers carried: a
    reader should not have to work out whether some path still scripts. The
    check is on the parsed syntax tree, so the prose that records what
    ``v0.0.27``'s ``python_backend`` did (which named ``torch.jit.script``)
    is not mistaken for a call.
    """

    def names_torch_jit(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and node.attr == "jit"
        )

    source_dir = pathlib.Path(deepwave.__file__).parent
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in sorted(source_dir.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if names_torch_jit(node)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# "auto" versus the older "compiled" default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_backend_auto_matches_compiled_bit_for_bit(name: str) -> None:
    """Where a compiled kernel exists, ``"auto"`` *is* the old default.

    This is the guard that the new default changed no numbers on the nine
    propagators that have a compiled backend.
    """
    auto = _quiet_call(name, backend="auto")
    compiled = _quiet_call(name, backend="compiled")
    _assert_outputs_equal(auto, compiled, name)
    assert auto.info.backend == "compiled"


# ---------------------------------------------------------------------------
# Output autograd behaviour does not depend on the backend
# ---------------------------------------------------------------------------


#: The propagators whose tiny probe leaves at least one receiver channel
#: unrequested, so that its output is the empty ``(0,)`` Tensor this section
#: is about. ``scalar``/``scalar_born`` have a single receiver channel and
#: always fill it, so they have no empty output to disagree about.
_EMPTY_CHANNEL = tuple(
    name for name in PROPAGATOR_NAMES if name not in ("scalar", "scalar_born")
)


@pytest.mark.parametrize("name", _EMPTY_CHANNEL)
def test_empty_output_autograd_does_not_depend_on_the_backend(name: str) -> None:
    """An unrequested channel's empty output is differentiable on every backend.

    The compiled backend allocates its empties inside its
    ``torch.autograd.Function``, which marks every returned Tensor
    differentiable; the eager cores allocate theirs with ``torch.empty`` and
    never write to them, which used to leave a detached leaf. So
    ``out.receivers["x"].sum().backward()`` on a channel the caller did not
    ask for raised ``RuntimeError: element 0 of tensors does not require
    grad`` under ``backend="eager"`` while quietly contributing a zero
    gradient under ``backend="compiled"`` -- and ``acoustic_vti``, whose
    synthetic pressure channel and cut-back stress channels go through
    ``Tensor.new_empty`` (which does not inherit the flag), disagreed with
    *itself* within a single call on both backends.

    Backward through an empty Tensor is a no-op, so the working behaviour is
    the one every backend now has: the empty output requires grad exactly
    when the call's other outputs do, and backward through it contributes a
    zero gradient rather than raising.
    """
    ndim = _ndim(name)
    backends = ("eager", "compiled")
    flags = {}
    for backend in backends:
        model = capabilities._models(name, ndim)[0].clone().requires_grad_(True)
        out = _quiet_call(name, backend=backend, primary_model=model)
        empty = [t for t in out if isinstance(t, torch.Tensor) and t.numel() == 0]
        assert empty, f"{name}: the probe requested every channel"
        assert len({t.requires_grad for t in empty}) == 1, (
            f"{name} ({backend}): its own empty outputs disagree with each other"
        )
        # Never raises, and adds nothing: there is nothing to sum.
        empty[0].sum().backward()
        assert model.grad is not None, f"{name} ({backend})"
        assert not model.grad.any(), f"{name} ({backend}): not a zero gradient"
        flags[backend] = [t.requires_grad for t in empty]
    assert all(flags[backends[0]]), name
    assert len({tuple(f) for f in flags.values()}) == 1, (
        f"{name}: requires_grad on an empty output depends on the backend"
    )


# ---------------------------------------------------------------------------
# "compiled" is strict
# ---------------------------------------------------------------------------


def test_backend_compiled_strict_names_the_option() -> None:
    """A compiled backend that lacks one *option* raises under ``"compiled"``.

    ``elastic_born`` has no M-PML forward kernel, so a call with
    ``mpml_fraction > 0`` asks for something only the eager backend implements.
    ``"auto"`` has an explicit fallback for this option; ``"compiled"``
    asserts that the compiled kernel ran, so it raises, naming the option.
    """

    def run(backend: str) -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return capabilities._call("elastic_born", 2, backend, mpml_fraction=0.1)

    with pytest.raises(NotImplementedError) as excinfo:
        run("compiled")
    message = str(excinfo.value)
    assert message.startswith("elastic_born:"), message
    assert "M-PML" in message or "mpml_fraction" in message, message
    assert 'backend="eager"' in message
    # "auto" runs it, on the backend that has it, and says which that was.
    run("auto")


# ---------------------------------------------------------------------------
# RunInfo.backend / RunInfo.backward_backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_run_info_backend_recorded(name: str) -> None:
    """Every propagator reports a *resolved* backend, never the request."""
    out = _quiet_call(name)
    resolved = ("compiled", "eager", "torch_compile")
    assert out.info.backend in resolved, name
    assert out.info.backward_backend in resolved, name
    assert out.info.backend != "auto"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("auto", "compiled"),
        ("compiled", "compiled"),
        ("eager", "eager"),
        ("torch_compile", "torch_compile"),
    ],
)
def test_run_info_backend_reports_what_ran(backend: str, expected: str) -> None:
    """``RunInfo.backend`` names the backend in the canonical vocabulary."""
    out = _quiet_call("scalar", backend=backend)
    assert out.info.backend == expected


@pytest.mark.parametrize("name", [n for n in PROPAGATOR_NAMES if n != "elastic_born"])
def test_backward_backend_matches_backend_everywhere_else(name: str) -> None:
    """Only ``elastic_born`` can split its forward and backward backends."""
    out = _quiet_call(name)
    assert out.info.backward_backend == out.info.backend, name


def test_run_info_backward_backend_splits_for_elastic_born() -> None:
    """``elastic_born``'s compiled forward pairs with the adjoint it can use.

    Without M-PML on CPU both directions are compiled; the M-PML case cannot
    run compiled at all (it raises, see above), and an explicitly requested
    Python backend is used for both directions.
    """
    out = _quiet_call("elastic_born", backend="auto")
    assert (out.info.backend, out.info.backward_backend) == ("compiled", "compiled")
    eager = _quiet_call("elastic_born", backend="eager")
    assert (eager.info.backend, eager.info.backward_backend) == ("eager", "eager")


def test_no_module_forward_collapses_auto_to_compiled() -> None:
    """No ``forward`` may turn its ``backend`` request into a resolution.

    Each ``forward`` canonicalises ``backend`` and hands the result to its
    functional. The idiom this guards against coerced the coerced-to-``None``
    compiled mode back into the literal ``"compiled"``; harmless while
    ``"compiled"`` was the default and lenient, but under the new strict
    ``"compiled"`` it would make a default call to either eager-only
    propagator raise. Checked on the source of all nine so the trap cannot
    return to a Module this suite does not exercise numerically.
    """
    for name, cls_name in _MODULES.items():
        source = inspect.getsource(getattr(deepwave, cls_name).forward)
        assert 'or "compiled"' not in source, (
            f"{name}.forward collapses its backend request to 'compiled'"
        )
        assert ".canonical" in source, (
            f"{name}.forward must forward the canonical request, not the mode"
        )


def test_checkpointing_slab_plan_survives_the_auto_default() -> None:
    """The slab-plan optimisation is keyed on the backend and must accept "auto".

    ``_build_slab_plan`` bails out for any backend other than the compiled
    one. It used to compare against the literal ``"compiled"`` default, so the
    rename to ``"auto"`` would have silently disabled the optimisation on
    every checkpointed run -- a performance regression with no error.
    """
    import deepwave.checkpointing

    source = inspect.getsource(deepwave.checkpointing._build_slab_plan)
    assert '"auto"' in source
    assert 'kwargs.get("backend", "compiled") != "compiled"' not in source


# ---------------------------------------------------------------------------
# "auto" resolves to a backend that implements every requested option
# ---------------------------------------------------------------------------


def _elastic_born_mpml_kwargs(mpml_fraction: float) -> dict:
    """A tiny ``elastic_born`` call, with or without M-PML."""
    n = 24
    lamb = torch.full((n, n), 2.25e9, dtype=torch.float64)
    mu = torch.full((n, n), 1.0e9, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / 2200.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15, 30, 0.0005, 0.06).double().reshape(1, 1, -1) * 1e6
    )
    return {
        "args": (lamb, mu, buoyancy, 4.0, 0.0005),
        "kwargs": {
            "dlamb": torch.full_like(lamb, 1e7),
            "source_amplitudes_y": amplitudes,
            "source_locations_y": torch.tensor([[[8, 8]]]),
            "sc_receiver_locations_y": torch.tensor([[[16, 16]]]),
            "pml_width": 5,
            "pml_freq": 15.0,
            "max_vel": 2000.0,
            "mpml_fraction": mpml_fraction,
        },
    }


def test_auto_resolves_around_a_compiled_option_gap() -> None:
    """``elastic_born`` with ``mpml_fraction > 0`` runs, on the backend that has it.

    This explicit option fallback can be much slower, which is why
    ``out.info.backend`` has to say which one ran.
    """
    call = _elastic_born_mpml_kwargs(0.1)
    out = deepwave.elastic_born(*call["args"], backend="auto", **call["kwargs"])
    assert out.info.backend == "eager"


def test_auto_still_picks_the_compiled_backend_without_the_gap() -> None:
    """The resolution is per call, not per propagator."""
    call = _elastic_born_mpml_kwargs(0.0)
    out = deepwave.elastic_born(*call["args"], backend="auto", **call["kwargs"])
    assert out.info.backend == "compiled"


def test_compiled_asserts_the_compiled_kernel_ran() -> None:
    """``backend="compiled"`` is an assertion, so the gap raises there."""
    call = _elastic_born_mpml_kwargs(0.1)
    with pytest.raises(deepwave.DeepwaveNotImplementedError) as excinfo:
        deepwave.elastic_born(*call["args"], backend="compiled", **call["kwargs"])
    message = str(excinfo.value)
    assert message.startswith("elastic_born:")
    assert "mpml_fraction=0.1" in message
    assert 'backend="eager"' in message


def test_compiled_option_gap_error_is_a_not_implemented_error() -> None:
    """The builtin base is preserved, so ``except NotImplementedError`` works."""
    call = _elastic_born_mpml_kwargs(0.1)
    with pytest.raises(NotImplementedError):
        deepwave.elastic_born(*call["args"], backend="compiled", **call["kwargs"])


def test_explicit_python_backends_run_the_gap_option() -> None:
    """A named Python backend implements it too, so it is not overridden."""
    call = _elastic_born_mpml_kwargs(0.1)
    out = deepwave.elastic_born(*call["args"], backend="eager", **call["kwargs"])
    assert out.info.backend == "eager"


def _scalar_born_second_order_call(
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds a second-order graph through ``scalar_born`` on ``backend``.

    Returns ``(v, grad)``: the model the gradient was taken with respect to,
    and that gradient, built with ``create_graph=True`` so that a second
    differentiation is possible. The compiled backend's guard fires while
    that graph is being built.
    """
    ny, nx, nt = 12, 12, 20
    v = (torch.ones(ny, nx, dtype=torch.float64) * 1500.0).requires_grad_(True)
    scatter = torch.zeros(ny, nx, dtype=torch.float64)
    scatter[ny // 2, nx // 2 + 1] = 100.0
    out = deepwave.scalar_born(
        v,
        5.0,
        0.004,
        dv=scatter,
        source_amplitudes=torch.randn(1, 1, nt, dtype=torch.float64),
        source_locations=torch.tensor([[[ny // 2, nx // 2]]]),
        sc_receiver_locations=torch.tensor([[[ny // 2, nx // 2 + 3]]]),
        pml_width=3,
        accuracy=2,
        pml_freq=25.0,
        backend=backend,
    )[-1]
    (grad,) = torch.autograd.grad(out.pow(2).sum(), v, create_graph=True)
    return v, grad


def test_second_order_error_names_eager_and_torch_func() -> None:
    """The second-order refusal has to say what to do instead.

    The propagators whose hand-derived adjoint has no analytic second-order
    path refuse ``create_graph=True`` on the compiled backend rather than
    returning a wrong answer. The fix is ``backend="eager"``, which steps the
    same physics under native autograd, so the message must name it -- and
    name ``torch.func``, since the transforms built on double-backward
    (``jacrev``, ``hessian``) hit the same guard.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _scalar_born_second_order_call("compiled")
    message = str(excinfo.value)
    assert "Second-order differentiation" in message
    assert 'backend="eager"' in message
    assert "torch.func" in message


def test_second_order_works_on_the_eager_backend() -> None:
    """The fix the message names actually works, so the advice is not empty."""
    v, grad = _scalar_born_second_order_call("eager")
    assert grad.requires_grad
    (second,) = torch.autograd.grad(grad.sum(), v)
    assert second.shape == v.shape

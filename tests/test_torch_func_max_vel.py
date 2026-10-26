"""Inferring ``max_vel`` under ``torch.func``, on all nine propagators.

``max_vel`` scales the absorbing boundary's profiles and, through the CFL
condition, sets the internal time step. It is inferred from the model values
when it is not given -- including under every ``torch.func`` transform whose
tensors can still be read (``grad``, ``jacrev``, ``jvp``, and a ``vmap`` over
something other than a model). ``vmap`` is what hides a model's values, so
``max_vel`` has to be given when a model that the maximum is read from is
batched; the error names the argument and is worded for every way a read can
fail, a model holding no values at all (``meta``, fake, nested) included.

Before this, any functorch-wrapped model was treated as unreadable, so a
``grad``, ``jacrev`` or ``jvp`` with respect to a model raised "If using
BatchedTensor inputs, you must specify max_vel" even though the wrapped
model's values were perfectly readable.

Which models each propagator reads the maximum from is documented in
``docs/capabilities.rst`` and parsed from there below, so the page and the
behaviour cannot drift apart. Only the cases that *must* raise are pinned:
whether batching some other model also raises is deliberately left open, so
that a future bound reading more of the models is an addition rather than a
test to rewrite.
"""

import inspect
import pathlib
import re
import warnings
from typing import Any

import pytest
import torch
from determinism_cases import DT, DX, PROPAGATORS, build_case
from helpers import output_tensors

import deepwave
from deepwave._common import max_vel_unreadable_error, values_readable

#: A handful of steps: these tests are about which value is resolved and
#: whether the call runs at all, not about the propagated wavefield.
NT = 6

#: The two propagators that refuse ``vmap`` over a model outright, for their
#: own pre-existing reason (they list their models among the arguments
#: ``check_inputs_not_vmapped`` rejects), so a batched model never reaches the
#: question this file is about. Checked in
#: ``test_the_born_propagators_refuse_vmap_over_a_model_outright``.
VMAP_OVER_A_MODEL_UNSUPPORTED = ("acoustic_born", "elastic_born")

#: The propagators a model can be mapped over.
VMAP_PROPAGATORS = tuple(
    name for name in PROPAGATORS if name not in VMAP_OVER_A_MODEL_UNSUPPORTED
)

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
_CAPABILITIES = _DOCS / "capabilities.rst"

#: The title of the table in ``docs/capabilities.rst`` that this file reads.
_TABLE_TITLE = ".. list-table:: Models the maximum wave speed is read from"

#: An inline literal, e.g. ``c11`` -- how the table spells a model's name.
_LITERAL = re.compile(r"``([A-Za-z0-9_]+)``")

#: Table entries not exercised here, each with its reason. ``c66`` is a 3-D
#: model (``elastic_vti`` requires it to be ``None`` in 2-D) and every case in
#: this file is built on the shared 2-D grid, so the pair is named rather than
#: quietly dropped -- ``test_every_unexercised_entry_is_in_the_table`` fails if
#: it stops being a table entry.
NOT_EXERCISED: frozenset[tuple[str, str]] = frozenset({("elastic_vti", "c66")})


def _documented_maximum_models() -> dict[str, tuple[str, ...]]:
    """Reads the capability page's table of which models set the maximum.

    Parsed rather than repeated, because a copy of the table here could agree
    with the code while the page said something else -- which is the failure
    the table exists to prevent.

    Returns:
        Each propagator's models, in the order the page lists them.

    """
    lines = _CAPABILITIES.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.strip() == _TABLE_TITLE
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    table: dict[str, tuple[str, ...]] = {}
    propagator: str = ""
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        row = line.strip()
        if row.startswith("* - "):
            names = _LITERAL.findall(row)
            # The header row ("* - Propagator") has no literals.
            propagator = names[0] if names else ""
        elif row.startswith("- ") and propagator:
            table[propagator] = tuple(_LITERAL.findall(row))
            propagator = ""
    assert table, f"no rows found under {_TABLE_TITLE!r}"
    return table


#: ``(propagator, model)`` for every table entry this file exercises.
MAXIMUM_MODEL_CASES: tuple[tuple[str, str], ...] = tuple(
    (name, model)
    for name, models in sorted(_documented_maximum_models().items())
    if name in VMAP_PROPAGATORS
    for model in models
    if (name, model) not in NOT_EXERCISED
)


def _positional_model_names(name: str) -> list[str]:
    """``name``'s models that are passed positionally, in call order.

    Derived the way :func:`determinism_cases.build_case` derives it, so the
    index of a model here is its index in that builder's model list.

    Args:
        name: The propagator's name.

    Returns:
        The positional models' names.

    """
    parameters = list(inspect.signature(getattr(deepwave, name)).parameters)
    positional = parameters[: parameters.index("grid_spacing")]
    return [
        model for model in deepwave.spec(name).model_names(2) if model in positional
    ]


def _case(
    name: str, *, max_vel: Any = None, nt: int = NT
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """The models and keyword arguments of one eager, detached call.

    Built from the same spec-driven builder the determinism tests use, so the
    case list is the propagator list and a new propagator cannot be missed.
    ``max_vel`` is removed unless a value is given.

    The builder's initial wavefields are dropped: padding one calls
    ``Tensor.requires_grad_()``, which no ``torch.func`` transform permits, so
    a call that starts from rest is what isolates the question this file is
    about from that separate limitation.
    """
    kwargs, _leaves, _names = build_case(
        name, torch.device("cpu"), torch.float64, backend="eager", nt=nt
    )
    for field in deepwave.spec(name).wavefield_names(2):
        kwargs.pop(field, None)
    models = [model.detach().clone() for model in kwargs.pop("_models")]
    kwargs = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
    if max_vel is None:
        del kwargs["max_vel"]
    else:
        kwargs["max_vel"] = max_vel
    return models, kwargs


def _run(name: str, models: list[torch.Tensor], kwargs: dict[str, Any]) -> Any:
    """Call ``name`` with these models, ignoring its resolution warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(deepwave, name)(*models, DX, DT, **kwargs)


# ---------------------------------------------------------------------------
# The readability question itself
# ---------------------------------------------------------------------------


def test_a_plain_tensors_values_are_readable() -> None:
    """The ordinary case, including a tensor that requires grad."""
    assert values_readable(torch.full((3, 4), 1500.0))
    assert values_readable(torch.full((3, 4), 1500.0, requires_grad=True))


def test_an_empty_tensor_reports_readable() -> None:
    """Nothing hides its values, so the caller's own read decides what to do."""
    assert values_readable(torch.zeros(0))
    assert values_readable(torch.zeros(3, 0, 4))


def test_a_grad_or_jvp_wrapped_tensor_is_readable() -> None:
    """A ``grad``/``jacrev``/``jvp`` wrapper has a readable primal."""
    seen: list[bool] = []
    v = torch.full((3, 4), 1500.0)
    torch.func.grad(lambda t: (seen.append(values_readable(t)), t.sum())[1])(v)
    torch.func.jvp(
        lambda t: (seen.append(values_readable(t)), t.sum())[1],
        (v,),
        (torch.ones_like(v),),
    )
    assert seen == [True, True]


def test_a_vmapped_tensor_is_not_readable_at_any_level() -> None:
    """A batched tensor hides its values, however the transforms are nested."""
    seen: list[bool] = []
    v = torch.full((3, 4), 1500.0)
    torch.func.vmap(lambda t: (seen.append(values_readable(t)), t.sum())[1])(v)
    torch.func.vmap(
        torch.func.grad(lambda t: (seen.append(values_readable(t)), t.sum())[1])
    )(v)
    torch.func.grad(
        lambda t: torch.func.vmap(
            lambda row: (seen.append(values_readable(row)), row.sum())[1]
        )(t).sum()
    )(v)
    assert seen == [False, False, False]


def test_values_are_reported_readable_while_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamo traces a read rather than performing it, so probing is skipped.

    Probing under compilation would add a graph break to a question whose
    answer is yes; the caller's own read pays for the one break there is.
    """
    monkeypatch.setattr("deepwave._common._torch_is_compiling", lambda: True)
    seen: list[bool] = []
    torch.func.vmap(lambda t: (seen.append(values_readable(t)), t.sum())[1])(
        torch.full((3, 4), 1500.0)
    )
    assert seen == [True]


def test_a_tensor_that_holds_no_values_is_not_readable() -> None:
    """A ``meta`` tensor has a shape and a dtype but no values to read.

    Nothing about it is batched, so the error this leads to must not claim
    that it is: see
    ``test_the_error_is_true_of_a_model_that_holds_no_values``.
    """
    assert not values_readable(torch.zeros(4, 4, device="meta"))


def test_the_error_names_the_argument_and_the_propagator() -> None:
    """One wording for all nine, naming ``max_vel=`` and why it is needed."""
    error = max_vel_unreadable_error("scalar")
    assert isinstance(error, deepwave.DeepwaveRuntimeError)
    assert str(error).startswith("scalar: ")
    assert "max_vel=" in str(error)
    assert "vmap" in str(error)


def test_the_error_is_true_of_a_model_that_holds_no_values() -> None:
    """A ``meta`` model reaches the same error, which has to fit it too.

    The readability probe cannot tell *why* a read failed, so one message
    covers both causes. It therefore states the fact (the values could not be
    read) and names ``vmap`` and a valueless tensor as the two ways that
    happens, rather than asserting the model is batched -- which, here, it is
    not.
    """
    models, kwargs = _case("scalar")
    with pytest.raises(deepwave.DeepwaveRuntimeError) as excinfo:
        _run("scalar", [models[0].to("meta")], kwargs)
    message = str(excinfo.value)
    assert message.startswith("scalar: "), message
    assert "max_vel=" in message, message
    assert "could not be read" in message, message
    assert "meta device" in message, message
    # The one thing it must not do is state a cause that does not hold.
    assert "is batched" not in message, message


# ---------------------------------------------------------------------------
# All nine propagators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
def test_grad_with_max_vel_omitted_resolves_the_untransformed_value(
    name: str,
) -> None:
    """``torch.func.grad`` over a model no longer requires ``max_vel``.

    The value it resolves, and therefore the internal time step, is the one
    the same call resolves outside the transform: a transform does not change
    how the run is discretised.
    """
    models, kwargs = _case(name)
    plain = _run(name, models, kwargs)
    resolved: list[tuple[Any, ...]] = []

    def loss(model: torch.Tensor) -> torch.Tensor:
        out = _run(name, [model, *models[1:]], kwargs)
        resolved.append((out.info.max_vel, out.info.internal_dt, out.info.step_ratio))
        return torch.stack([t.sum() for t in output_tensors(out)]).sum()

    gradient = torch.func.grad(loss)(models[0])
    assert resolved == [
        (plain.info.max_vel, plain.info.internal_dt, plain.info.step_ratio)
    ]
    assert plain.info.max_vel is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0, "vacuous: the gradient is identically zero"


def test_the_table_covers_every_propagator() -> None:
    """The capability page lists the maximum's models for all nine."""
    assert sorted(_documented_maximum_models()) == sorted(PROPAGATORS)


def test_every_unexercised_entry_is_in_the_table() -> None:
    """The named exceptions are real table entries, not stale ones."""
    table = _documented_maximum_models()
    for name, model in sorted(NOT_EXERCISED):
        assert model in table[name], (name, model)


def _mapped_over_one_model(
    name: str, model: str, models: list[torch.Tensor], kwargs: dict[str, Any]
) -> torch.Tensor:
    """Maps ``name``'s last output over a batch of one, batching ``model``."""
    index = _positional_model_names(name).index(model)

    def first_output(batched: torch.Tensor) -> torch.Tensor:
        call_models = list(models)
        call_models[index] = batched
        return output_tensors(_run(name, call_models, kwargs))[-1]

    return torch.func.vmap(first_output)(models[index].unsqueeze(0))


@pytest.mark.parametrize(("name", "model"), MAXIMUM_MODEL_CASES)
def test_batching_a_model_the_maximum_is_read_from_requires_max_vel(
    name: str, model: str
) -> None:
    """Each documented model, batched with ``max_vel`` omitted, raises.

    These are the cases that certainly require it: ``vmap`` hides the values
    the maximum would be read from, so there is nothing to infer, and the
    message names the argument, the propagator and the reason.

    Batching a model the maximum is *not* read from is deliberately not tested
    either way: today several of those also raise (they are checked alongside
    the ones the bound uses), and a bound that read more of the models
    tomorrow would only add cases, which is the additive direction.
    """
    models, kwargs = _case(name)
    with pytest.raises(deepwave.DeepwaveRuntimeError) as excinfo:
        _mapped_over_one_model(name, model, models, kwargs)
    message = str(excinfo.value)
    assert message.startswith(f"{name}: "), message
    assert "max_vel=" in message, message


@pytest.mark.parametrize(("name", "model"), MAXIMUM_MODEL_CASES)
def test_giving_max_vel_lets_the_same_batched_call_run(name: str, model: str) -> None:
    """Supplying the argument is the whole fix: the same call then runs."""
    models, kwargs = _case(name, max_vel=3000.0)
    mapped = _mapped_over_one_model(name, model, models, kwargs)
    assert mapped.shape[0] == 1
    assert torch.isfinite(mapped).all()


@pytest.mark.parametrize("name", PROPAGATORS)
def test_omitting_max_vel_gives_the_same_gradient_as_pinning_it(name: str) -> None:
    """A run that infers the maximum is bitwise the run that is given it.

    Under ``torch.func.grad``, omitting ``max_vel`` and passing the value the
    inference resolves produce the same gradient bit for bit. That is the
    property a caller can rely on: switching between the two does not change
    the numbers, forward or backward, so the absorbing-boundary profiles and
    the internal time step the maximum chooses behave as constants of the
    call.

    This does not test the ``detach()`` calls in the inference: the reads end
    in ``.item()``, which yields a Python number and so severs the graph on
    its own, and this test passes with or without them. It tests the property
    those calls make explicit.
    """
    models, kwargs = _case(name)
    inferred = _run(name, models, kwargs).info.max_vel
    assert inferred is not None
    pinned = dict(kwargs, max_vel=inferred)

    def loss(model: torch.Tensor, call_kwargs: dict[str, Any]) -> torch.Tensor:
        out = _run(name, [model, *models[1:]], call_kwargs)
        return torch.stack([t.sum() for t in output_tensors(out)]).sum()

    from_inference = torch.func.grad(lambda m: loss(m, kwargs))(models[0])
    from_pinned = torch.func.grad(lambda m: loss(m, pinned))(models[0])
    assert from_inference.abs().max() > 0, "vacuous: the gradient is zero"
    assert torch.equal(from_inference, from_pinned)


@pytest.mark.parametrize("name", VMAP_OVER_A_MODEL_UNSUPPORTED)
def test_the_born_propagators_refuse_vmap_over_a_model_outright(name: str) -> None:
    """A separate, pre-existing limitation, named here so the gap is explicit.

    These two list their models among the arguments they reject as vmapped, so
    a batched model is refused before ``max_vel`` is looked at -- with or
    without ``max_vel``. They are therefore excluded from the ``vmap`` cases
    above rather than silently unparametrised.
    """
    models, kwargs = _case(name, max_vel=3000.0)

    def first_output(model: torch.Tensor) -> torch.Tensor:
        out = _run(name, [model, *models[1:]], kwargs)
        return output_tensors(out)[-1]

    with pytest.raises(NotImplementedError, match="Only property models"):
        torch.func.vmap(first_output)(models[0].unsqueeze(0))


def test_compiling_a_vmapped_call_works_when_max_vel_is_given() -> None:
    """``torch.compile`` of a ``vmap`` needs ``max_vel=`` passed explicitly.

    While Dynamo is tracing, the readability probe is skipped (probing would
    only add a graph break to a question whose answer, inside a compiled
    region, is yes), so a batched model is not recognised as one and Deepwave's
    own message is not what a caller sees: omitting ``max_vel`` there surfaces
    PyTorch's own refusal to run ``.item()`` under ``vmap``. Supplying it is
    the documented route, and that is what is pinned here -- the combination
    that works.
    """
    models, kwargs = _case("scalar", max_vel=3000.0)
    compiled = torch.compile(
        torch.func.vmap(
            lambda model: output_tensors(_run("scalar", [model], kwargs))[-1]
        ),
        backend="eager",
    )
    mapped = compiled(models[0].unsqueeze(0))
    assert mapped.shape[0] == 1
    assert torch.isfinite(mapped).all()


def test_jvp_over_the_model_needs_no_max_vel_on_the_default_backend() -> None:
    """The fast-backend ``jvp`` rule infers it too, from the readable primal."""
    models, kwargs = _case("scalar")
    del kwargs["backend"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        primal, tangent = torch.func.jvp(
            lambda v: deepwave.scalar(v, DX, DT, **kwargs).receiver_amplitudes,
            (models[0],),
            (0.01 * models[0],),
        )
    assert torch.isfinite(primal).all()
    assert torch.isfinite(tangent).all()
    assert tangent.abs().max() > 0, "vacuous: the tangent is identically zero"

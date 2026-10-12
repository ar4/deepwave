"""Tests for what a propagator ``Module`` does with the models it is given.

All nine register every supplied model through one helper,
``_common.make_model_parameter``, i.e. ``torch.nn.Parameter(model,
requires_grad=flag)``. Five class docstrings used to describe that in five
different ways and one of them (``ScalarBorn``'s "a copy will be made") was
simply false. There is now one paragraph, single-sourced in
``_common.MODEL_OWNERSHIP_DOC`` and appended to all nine at import time, and
this file tests that what the paragraph *claims* is true -- particularly the
claim that wrapping a network-produced model in a Module severs the graph,
which is the one users are most likely to be caught by.

It also covers keyword-only constructor arguments after ``grid_spacing``
and the individual ``<model>_requires_grad`` flags on every Module.
"""

import inspect
import warnings
from typing import Any

import pytest
import torch

import deepwave
from deepwave import _common

_RELEASED = {
    "Scalar": ["v"],
    "ScalarBorn": ["v", "dv"],
    "Acoustic": ["v", "rho"],
    "Elastic": ["lamb", "mu", "buoyancy"],
}

_ALL_MODULES = [
    "Scalar",
    "ScalarBorn",
    "Acoustic",
    "AcousticBorn",
    "AcousticVti",
    "Elastic",
    "ElasticBorn",
    "ElasticVti",
    "Viscoacoustic",
]


def _model(value: float = 1500.0) -> torch.Tensor:
    return torch.full((8, 8), value)


#: Every model each Module can own, and the subset ``_build`` supplies. The
#: difference is the optional models, which is what makes an individual flag
#: naming one of them an error.
_MODELS = {
    "Scalar": ["v"],
    "ScalarBorn": ["v", "dv"],
    "Acoustic": ["v", "rho"],
    "AcousticBorn": ["v", "rho", "dv", "drho"],
    "AcousticVti": ["c11", "c13", "c33", "buoyancy"],
    "Elastic": ["lamb", "mu", "buoyancy"],
    "ElasticBorn": ["lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy"],
    "ElasticVti": ["c11", "c13", "c33", "c55", "c66", "buoyancy"],
    "Viscoacoustic": ["v", "rho", "q"],
}

_SUPPLIED = {
    "Scalar": ["v"],
    "ScalarBorn": ["v", "dv"],
    "Acoustic": ["v", "rho"],
    "AcousticBorn": ["v", "rho", "dv"],
    "AcousticVti": ["c11", "c13", "c33", "buoyancy"],
    "Elastic": ["lamb", "mu", "buoyancy"],
    "ElasticBorn": ["lamb", "mu", "buoyancy", "dlamb"],
    "ElasticVti": ["c11", "c13", "c33", "c55", "buoyancy"],
    "Viscoacoustic": ["v", "rho", "q"],
}


def _build(name: str, **kwargs: Any) -> torch.nn.Module:
    """Constructs ``name`` with the minimum set of models it requires."""
    cls = getattr(deepwave, name)
    v = _model()
    stiff = _model(2.25e9)
    shear = _model(1.0e9)
    buoyancy = _model(1 / 2200.0)
    q = _model(40.0)
    with warnings.catch_warnings():
        if name == "Scalar":
            return cls(v, 4.0, **kwargs)
        if name == "ScalarBorn":
            return cls(v, 4.0, dv=torch.zeros(8, 8), **kwargs)
        if name == "Acoustic":
            return cls(v, _model(2200.0), 4.0, **kwargs)
        if name == "AcousticBorn":
            return cls(v, _model(2200.0), 4.0, dv=torch.zeros(8, 8), **kwargs)
        if name == "AcousticVti":
            return cls(stiff, stiff, stiff, buoyancy, 4.0, **kwargs)
        if name == "Elastic":
            return cls(stiff, shear, buoyancy, 4.0, **kwargs)
        if name == "ElasticBorn":
            return cls(stiff, shear, buoyancy, 4.0, dlamb=torch.zeros(8, 8), **kwargs)
        if name == "ElasticVti":
            return cls(stiff, stiff, stiff, shear, buoyancy, 4.0, **kwargs)
        if name == "Viscoacoustic":
            return cls(v, _model(2200.0), q, 4.0, **kwargs)
        return cls(stiff, shear, buoyancy, q, q, 4.0, **kwargs)


# ---------------------------------------------------------------------------
# The shared paragraph, and the claims it makes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_all_nine_share_the_ownership_paragraph(name: str) -> None:
    """The exact ``_common`` constant appears in every class docstring."""
    doc = getattr(deepwave, name).__doc__
    assert doc is not None
    assert _common.MODEL_OWNERSHIP_DOC in doc, name


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_module_parameter_shares_storage_but_is_not_the_input(name: str) -> None:
    """The Parameter is a distinct leaf that shares the caller's storage."""
    module = _build(name)
    parameters = list(module.parameters())
    assert parameters, name
    for parameter in parameters:
        assert isinstance(parameter, torch.nn.Parameter)
        assert parameter.is_leaf


def test_module_parameter_is_not_a_copy() -> None:
    """Sharing storage, not copying: an in-place edit of one shows in the other."""
    v = _model()
    module = deepwave.Scalar(v, 4.0)
    assert module.v is not v
    assert module.v.data_ptr() == v.data_ptr()
    with torch.no_grad():
        v[0, 0] = 1234.0
    assert module.v[0, 0].item() == 1234.0


def test_module_severs_upstream_graph() -> None:
    """A network-produced model loses its graph inside a Module, keeps it outside.

    This is the concrete claim of the ownership paragraph's last two
    sentences, and the reason it tells users to reach for the functional
    interface for anything graph-producing.
    """
    source_amplitudes = torch.randn(1, 1, 6)
    source_locations = torch.tensor([[[4, 4]]])
    receiver_locations = torch.tensor([[[6, 6]]])

    latent = torch.ones(1, requires_grad=True)
    v_module = 1500.0 * torch.ones(8, 8) * latent
    module = deepwave.Scalar(v_module, 4.0)
    module.v.requires_grad_(True)
    out = module(
        0.002,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=25.0,
    )
    out.receiver_amplitudes.sum().backward()
    assert module.v.grad is not None, "the module's own Parameter does get a gradient"
    assert latent.grad is None, "but the graph to the producer is severed"

    latent2 = torch.ones(1, requires_grad=True)
    v_functional = 1500.0 * torch.ones(8, 8) * latent2
    out2 = deepwave.scalar(
        v_functional,
        4.0,
        0.002,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=25.0,
    )
    out2.receiver_amplitudes.sum().backward()
    assert latent2.grad is not None, "the functional interface backpropagates through"


def test_optional_model_left_none_is_not_a_parameter() -> None:
    """An optional model left ``None`` is in neither ``parameters()`` nor state."""
    module = deepwave.AcousticBorn(_model(), _model(2200.0), 4.0, dv=torch.zeros(8, 8))
    assert module.drho is None
    assert "drho" not in module.state_dict()


# ---------------------------------------------------------------------------
# Constructor shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_constructors_reject_positional_requires_grad(name: str) -> None:
    """Nothing after ``grid_spacing`` may be given positionally."""
    parameters = list(inspect.signature(getattr(deepwave, name).__init__).parameters)
    index = parameters.index("grid_spacing")
    kinds = inspect.signature(getattr(deepwave, name).__init__).parameters
    for parameter in parameters[index + 1 :]:
        assert kinds[parameter].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name}.{parameter}"
        )


def test_released_modules_reject_positional_flag_with_the_documented_error() -> None:
    """Reject positional flags across supported Python error-message formats."""
    v = _model()
    rho = _model(2200.0)
    lamb = _model(2.25e9)
    cases = [
        (lambda: deepwave.Scalar(v, 4.0, True), "Scalar.__init__"),
        (lambda: deepwave.Acoustic(v, rho, 4.0, True), "Acoustic.__init__"),
        (lambda: deepwave.Elastic(lamb, lamb, lamb, 4.0, True), "Elastic.__init__"),
    ]
    for call, prefix in cases:
        with pytest.raises(TypeError, match="positional arguments but") as excinfo:
            call()
        message = str(excinfo.value)
        # Python 3.9 omits the class qualifier on this interpreter-generated
        # error. Deepwave's keyword-only restriction is the same on both.
        assert prefix in message or message.startswith("__init__() takes ")


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_every_module_has_individual_flags(name: str) -> None:
    """All nine Modules carry one ``<model>_requires_grad`` per model."""
    flags = [
        parameter
        for parameter in inspect.signature(getattr(deepwave, name).__init__).parameters
        if parameter.endswith("_requires_grad")
    ]
    assert flags, name
    for model in _MODELS[name]:
        assert f"{model}_requires_grad" in flags, f"{name}: no flag for {model}"


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_individual_flag_for_an_unsupplied_optional_model_raises(name: str) -> None:
    """A flag naming an optional model this instance was not given raises.

    The same rule the ``requires_grad=[...]`` sequence form applies: the
    module does not own that model, so honouring the flag is impossible and
    dropping it silently would withhold a gradient that was asked for.
    """
    optional = sorted(set(_MODELS[name]) - set(_SUPPLIED[name]))
    if not optional:
        pytest.skip(f"{name} has no optional model")
    for model in optional:
        with pytest.raises(deepwave.DeepwaveValueError, match=f"no {model} was given"):
            _build(name, **{f"{model}_requires_grad": True})


# ---------------------------------------------------------------------------
# requires_grad=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_requires_grad_none_is_todays_behaviour(name: str) -> None:
    """The default requires no gradient anywhere: bit-exact with before."""
    module = _build(name)
    assert not any(p.requires_grad for p in module.parameters()), name


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_requires_grad_true_and_false(name: str) -> None:
    """``True`` selects every owned model; ``False`` selects none."""
    cls = getattr(deepwave, name)
    signature = inspect.signature(cls.__init__)
    assert "requires_grad" in signature.parameters, name

    base = _build(name)
    owned = list(base.state_dict())

    for value, expected in ((True, True), (False, False)):
        module = _rebuild_with(name, requires_grad=value)
        flags = {
            key: parameter.requires_grad for key, parameter in module.named_parameters()
        }
        assert set(flags) == set(owned), name
        assert all(flag is expected for flag in flags.values()), (name, value)


def _rebuild_with(name: str, **kwargs: object) -> torch.nn.Module:
    """Rebuilds ``name`` with extra constructor keywords."""
    cls = getattr(deepwave, name)
    v = _model()
    stiff = _model(2.25e9)
    shear = _model(1.0e9)
    buoyancy = _model(1 / 2200.0)
    q = _model(40.0)
    with warnings.catch_warnings():
        if name == "Scalar":
            return cls(v, 4.0, **kwargs)
        if name == "ScalarBorn":
            return cls(v, 4.0, dv=torch.zeros(8, 8), **kwargs)
        if name == "Acoustic":
            return cls(v, _model(2200.0), 4.0, **kwargs)
        if name == "AcousticBorn":
            return cls(v, _model(2200.0), 4.0, dv=torch.zeros(8, 8), **kwargs)
        if name == "AcousticVti":
            return cls(stiff, stiff, stiff, buoyancy, 4.0, **kwargs)
        if name == "Elastic":
            return cls(stiff, shear, buoyancy, 4.0, **kwargs)
        if name == "ElasticBorn":
            return cls(stiff, shear, buoyancy, 4.0, dlamb=torch.zeros(8, 8), **kwargs)
        if name == "ElasticVti":
            return cls(stiff, stiff, stiff, shear, buoyancy, 4.0, **kwargs)
        if name == "Viscoacoustic":
            return cls(v, _model(2200.0), q, 4.0, **kwargs)
        return cls(stiff, shear, buoyancy, q, q, 4.0, **kwargs)


def test_requires_grad_sequence_selects_exactly_those_models() -> None:
    """A sequence of names selects exactly the named models."""
    module = _rebuild_with("Elastic", requires_grad=["lamb", "buoyancy"])
    flags = {name: p.requires_grad for name, p in module.named_parameters()}
    assert flags == {"lamb": True, "mu": False, "buoyancy": True}


def test_requires_grad_forms_give_identical_gradients() -> None:
    """``requires_grad=["v"]``, ``v_requires_grad=True`` and ``requires_grad_``.

    All three routes to "``v`` requires a gradient" must produce a
    bit-identical ``v.grad``.
    """
    kwargs = {
        "source_amplitudes": torch.randn(1, 1, 8, dtype=torch.float64),
        "source_locations": torch.tensor([[[4, 4]]]),
        "receiver_locations": torch.tensor([[[6, 6]]]),
        "pml_freq": 25.0,
    }

    def grad_of(module: torch.nn.Module) -> torch.Tensor:
        out = module(0.002, **kwargs)
        out.receiver_amplitudes.sum().backward()
        assert module.v.grad is not None
        return module.v.grad.clone()

    v = torch.full((8, 8), 1500.0, dtype=torch.float64)
    by_list = grad_of(deepwave.Scalar(v, 4.0, requires_grad=["v"]))
    by_flag = grad_of(deepwave.Scalar(v, 4.0, v_requires_grad=True))
    module = deepwave.Scalar(v, 4.0)
    module.v.requires_grad_(True)
    by_method = grad_of(module)
    assert torch.equal(by_list, by_flag)
    assert torch.equal(by_list, by_method)


def test_requires_grad_unknown_name_raises() -> None:
    """An unknown model name raises, listing the valid ones."""
    with pytest.raises(ValueError, match="has no model 'velocity'") as excinfo:
        _rebuild_with("Elastic", requires_grad=["velocity"])
    message = str(excinfo.value)
    for name in ("'lamb'", "'mu'", "'buoyancy'"):
        assert name in message


def test_requires_grad_unsupplied_optional_model_raises() -> None:
    """Naming an optional model that was not given raises, saying so."""
    with pytest.raises(ValueError, match="no drho was given"):
        _rebuild_with("AcousticBorn", requires_grad=["drho"])


def test_requires_grad_true_skips_unsupplied_optional_models() -> None:
    """``True`` means "every model this instance owns", not "every name"."""
    module = _rebuild_with("AcousticBorn", requires_grad=True)
    flags = {name: p.requires_grad for name, p in module.named_parameters()}
    assert flags == {"v": True, "rho": True, "dv": True}


def test_requires_grad_and_individual_flag_conflict_raises() -> None:
    """Combining the two forms raises rather than letting one silently win."""
    with pytest.raises(ValueError, match="Pass only requires_grad"):
        _rebuild_with("Scalar", requires_grad=True, v_requires_grad=True)


def test_requires_grad_conflict_is_caught_in_both_directions() -> None:
    """An explicit ``False`` conflicts just as an explicit ``True`` does.

    The individual flags default to ``None`` ("not given") precisely so that
    this direction can be distinguished from the default. Before that, an
    explicit ``<model>_requires_grad=False`` alongside ``requires_grad=True``
    was silently discarded and the model was given a gradient the caller had
    just asked it not to have -- the silent direction being the expensive
    one (extra memory, extra time, an unexpected ``.grad``).
    """
    for requires_grad in (True, ["v"]):
        with pytest.raises(ValueError, match="Pass only requires_grad"):
            _rebuild_with("Scalar", requires_grad=requires_grad, v_requires_grad=False)


def test_requires_grad_conflicts_even_when_the_two_agree() -> None:
    """One rule: pass one form or the other, never both.

    ``requires_grad=False`` with ``v_requires_grad=False`` says the same
    thing twice, but accepting agreement and rejecting disagreement would
    make the rule depend on the values rather than on which arguments were
    passed.
    """
    with pytest.raises(ValueError, match="Pass only requires_grad"):
        _rebuild_with("Scalar", requires_grad=False, v_requires_grad=False)


def test_omitted_individual_flag_does_not_conflict() -> None:
    """``requires_grad=`` alone is the ordinary case and still works."""
    assert _rebuild_with("Scalar", requires_grad=True).v.requires_grad
    assert not _rebuild_with("Scalar", requires_grad=False).v.requires_grad
    assert not _rebuild_with("Scalar", v_requires_grad=False).v.requires_grad
    assert _rebuild_with("Scalar", v_requires_grad=True).v.requires_grad


def test_requires_grad_wrong_type_raises() -> None:
    """A non-bool, non-sequence value is a ``TypeError``."""
    with pytest.raises(TypeError, match="requires_grad must be"):
        _rebuild_with("Scalar", requires_grad=1.5)


# ---------------------------------------------------------------------------
# The unvalidated-flag hole on optional models
# ---------------------------------------------------------------------------


def test_optional_model_requires_grad_flag_validated() -> None:
    """A bad flag is rejected even when its model is ``None``.

    ``make_model_parameter`` validates the flag, but an optional model left
    ``None`` never reaches it, so ``ScalarBorn(..., scatter_requires_grad=...)``
    and its kin used to accept anything.
    """
    with pytest.raises(TypeError, match="requires_grad must be bool"):
        deepwave.ScalarBorn(_model(), 4.0, dv=torch.zeros(8, 8), dv_requires_grad="yes")


def test_scatter_requires_grad_without_scatter_raises() -> None:
    """The deprecated flag is no longer silently ignored.

    Passing it without ``scatter`` used to drop the caller's request for a
    gradient on the floor.
    """
    with pytest.raises(ValueError, match="scatter_requires_grad was given"):
        deepwave.ScalarBorn(
            _model(), 4.0, dv=torch.zeros(8, 8), scatter_requires_grad=True
        )


def test_scatter_requires_grad_with_scatter_still_works() -> None:
    """The deprecated pair still does what it always did, and both names warn."""
    with pytest.warns(deepwave.DeepwaveFutureWarning) as record:
        module = deepwave.ScalarBorn(
            _model(), 4.0, scatter=torch.zeros(8, 8), scatter_requires_grad=True
        )
    assert module.dv.requires_grad
    messages = [str(w.message) for w in record]
    assert any("scatter is deprecated" in m for m in messages), messages
    assert any("scatter_requires_grad is deprecated" in m for m in messages), messages


# ---------------------------------------------------------------------------
# Spliced forward docstrings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_MODULES)
def test_forward_docstring_contains_every_keyword(name: str) -> None:
    """Every ``forward`` parameter appears in the spliced ``Args:`` block."""
    import re

    forward = getattr(deepwave, name).forward
    doc = forward.__doc__
    assert doc is not None
    assert doc.startswith("Perform forward propagation/modelling."), name
    documented = set(re.findall(r"^    (\w+):", doc, re.M))
    parameters = {
        parameter
        for parameter in inspect.signature(forward).parameters
        if parameter != "self"
    }
    assert not parameters - documented, f"{name}: {sorted(parameters - documented)}"


def test_splicer_is_a_no_op_without_docstrings() -> None:
    """``python -OO`` strips docstrings; the splicer must not crash."""

    def functional() -> None:
        pass

    def method(self: object, *, dt: float = 0.0) -> None:
        pass

    functional.__doc__ = None
    method.__doc__ = None
    assert _common.inherit_args(functional)(method) is method
    assert method.__doc__ is None

    class Bare:
        def forward(self) -> None:
            pass

    Bare.__doc__ = None
    _common.document_propagator_module(Bare, functional)
    assert Bare.__doc__ is None

"""One interface, nine propagators: the conformance suite.

Deepwave's promise is that the nine propagators are one interface with
nine physics bodies, not nine libraries that happen to live in one
package. A wrapper written against ``scalar`` must drive ``elastic_vti``
after changing only the model tensors it passes, and the only differences it
should have to cope with are ones physics forces: which models exist, and
what a source injects or a receiver records.

This file is the executable statement of that promise. Every check here is
driven by the published registry -- :func:`deepwave.spec` and the
:class:`deepwave.PropagatorSpec` it returns -- rather than by a hand-written
table of per-propagator facts, so a propagator cannot pass by being listed as
an exception. Where a genuine per-propagator difference exists it is read out
of the spec (``model_names``, ``source_channels``, ``receiver_channels``,
``output_names``), which is the same thing a user's own generic wrapper would
read.

Everything is checked in **both** call forms, the functional and the
``torch.nn.Module``, because a caller may hold either.

The families of checks, in file order:

* **Calling convention.** Positional models, then ``grid_spacing``, then
  ``dt``, then keyword-only everything; one default and one annotation for
  every keyword name that two propagators share; one identical configuration
  tail on all nine; channel and state keyword groups in the spec's order.
* **Survey arguments.** The ``sources=``/``receivers=`` mapping form and the
  flat keyword form accept exactly the spec's channel names and compute
  bit-identical results; an unknown name raises the same class with the same
  message shape everywhere.
* **The output object.** One type, the same access patterns (by name, by
  declared slot, as the documented sequence), one ``info`` field set with one
  set of field types, and a pytree registration that round-trips.
* **Errors and warnings.** The conditions the documentation promises raise or
  warn the same class, with the same message shape, on all nine.
* **The Module form.** The constructor/``forward`` split, the
  ``<model>_requires_grad`` flags, the ``state_dict`` keys and the
  ``requires_grad`` semantics follow one rule generated from the spec.
* **Introspection and documentation.** Every spec is well formed; every
  public name is documented and every documented name is public; every
  propagator docstring documents exactly its own parameters and states the
  dimensions of the physical ones.

The runs are deliberately tiny (a dozen cells a side, eight time steps, CPU),
because nothing here is a numerical check: every assertion is about names,
types, classes, message shapes and bitwise equality between two spellings of
the same call.
"""

import dataclasses
import inspect
import pathlib
import re
import types
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import pytest
import torch
from torch.utils import _pytree as pytree

import deepwave

PROPAGATORS: tuple[str, ...] = deepwave._engine.PROPAGATOR_NAMES

#: Each propagator's ``torch.nn.Module`` class. Written out rather than
#: derived from the name so that the mapping itself is checked: a Module that
#: was renamed, or one that stopped being exported, fails at import.
MODULES: dict[str, type] = {
    "scalar": deepwave.Scalar,
    "scalar_born": deepwave.ScalarBorn,
    "acoustic": deepwave.Acoustic,
    "acoustic_born": deepwave.AcousticBorn,
    "acoustic_vti": deepwave.AcousticVti,
    "viscoacoustic": deepwave.Viscoacoustic,
    "elastic": deepwave.Elastic,
    "elastic_born": deepwave.ElasticBorn,
    "elastic_vti": deepwave.ElasticVti,
}

FORMS = ("function", "module")

_N = 10
_DX = 5.0
_DT = 0.0005
_NT = 6
_FREQ = 15.0
_DTYPE = torch.float64

#: One value per *model name*, shared by every propagator that takes a model
#: of that name. Building the fixtures this way is itself a conformance
#: check: a name that means one quantity on one propagator and a different
#: quantity on another would produce an unphysical model somewhere.
_MODEL_VALUES: dict[str, float] = {
    "v": 1500.0,
    "dv": 0.0,
    "rho": 2200.0,
    "drho": 0.0,
    "q": 40.0,
    "qp": 40.0,
    "qs": 30.0,
    "lamb": 2.0e9,
    "dlamb": 0.0,
    "mu": 1.0e9,
    "dmu": 0.0,
    "buoyancy": 1.0 / 2200.0,
    "dbuoyancy": 0.0,
    "c11": 9.0e9,
    "c13": 2.0e9,
    "c33": 8.0e9,
    "c55": 1.0e9,
    "c66": 1.2e9,
    "tilt": 0.1,
}

# ---------------------------------------------------------------------------
# Spec-driven fixtures: one tiny call, assembled for any propagator.
# ---------------------------------------------------------------------------


def _signature(name: str) -> inspect.Signature:
    return inspect.signature(getattr(deepwave, name))


def _positional_models(name: str) -> tuple[str, ...]:
    """The model names this propagator takes positionally."""
    return tuple(
        p.name
        for p in _signature(name).parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.name not in ("grid_spacing", "dt")
    )


def _keyword_models(name: str, ndim: int) -> tuple[str, ...]:
    """The model names this propagator takes as keywords at ``ndim``."""
    positional = set(_positional_models(name))
    return tuple(
        m for m in deepwave.spec(name).model_names(ndim) if m not in positional
    )


def _model(model_name: str, ndim: int, dtype: torch.dtype = _DTYPE) -> torch.Tensor:
    return torch.full((_N,) * ndim, _MODEL_VALUES[model_name], dtype=dtype)


def _amplitudes(dtype: torch.dtype = _DTYPE) -> torch.Tensor:
    return deepwave.wavelets.ricker(_FREQ, _NT, _DT, 0.1).to(dtype).reshape(1, 1, -1)


def _locations(ndim: int) -> torch.Tensor:
    return torch.tensor([[[_N // 2] * ndim]])


def _survey_kwargs(name: str, ndim: int, *, mapping: bool) -> dict[str, Any]:
    """One source and one receiver, spelled either flat or as a mapping."""
    spec = deepwave.spec(name)
    source_channel = spec.source_channels(ndim)[0]
    receiver_channel = spec.receiver_channels(ndim)[0]
    amplitudes, locations = _amplitudes(), _locations(ndim)
    if mapping:
        kwargs: dict[str, Any] = {
            "sources": {source_channel: (amplitudes, locations)},
            "receivers": {receiver_channel: locations},
        }
        if spec.has_bg_receivers:
            kwargs["bg_receivers"] = {spec.bg_receiver_channels(ndim)[0]: locations}
        return kwargs
    kwargs = {
        spec.source_amplitudes_kwarg(source_channel): amplitudes,
        spec.source_locations_kwarg(source_channel): locations,
        spec.receiver_locations_kwarg(receiver_channel): locations,
    }
    if spec.has_bg_receivers:
        channel = spec.bg_receiver_channels(ndim)[0]
        kwargs[spec.bg_receiver_locations_kwarg(channel)] = locations
    return kwargs


def _config_kwargs(name: str) -> dict[str, Any]:
    """The non-survey arguments every call here pins, so runs are comparable."""
    kwargs: dict[str, Any] = {
        "accuracy": 2,
        "pml_width": 3,
        "pml_freq": _FREQ,
        "max_vel": 3000.0,
    }
    if "q_freq" in deepwave.spec(name).supported_options:
        kwargs["q_freq"] = _FREQ
    return kwargs


def _call(
    name: str,
    form: str = "function",
    ndim: int = 2,
    *,
    survey: Optional[str] = "flat",
    models: Optional[Sequence[torch.Tensor]] = None,
    **overrides: Any,
) -> Any:
    """Runs ``name`` in ``form`` on a tiny problem, with ``overrides`` applied.

    ``survey`` chooses the spelling of the default one-source, one-receiver
    survey: ``"flat"``, ``"mapping"``, or ``None`` to supply none at all so
    that the caller can pass its own.
    """
    kwargs: dict[str, Any] = {}
    if survey is not None:
        kwargs.update(_survey_kwargs(name, ndim, mapping=survey == "mapping"))
    kwargs.update(_config_kwargs(name))
    for model_name in _keyword_models(name, ndim):
        kwargs[model_name] = _model(model_name, ndim)
    kwargs.update(overrides)
    if models is None:
        models = [_model(m, ndim) for m in _positional_models(name)]
    if form == "function":
        return getattr(deepwave, name)(*models, _DX, _DT, **kwargs)
    constructor = {m: kwargs.pop(m) for m in _keyword_models(name, ndim) if m in kwargs}
    return MODULES[name](*models, _DX, **constructor)(_DT, **kwargs)


def _forget_once_per_process_warnings() -> None:
    """Re-arms the warnings that are emitted at most once per cause per process.

    Several categories are deliberately raised only the first time their cause
    is met, so that a loop over many calls does not print the same sentence
    thousands of times. A test that checks the same cause on every propagator
    and in both call forms is exactly the case that suppression hides, so it
    clears the record first.
    """
    deepwave._common._warned_once_causes.clear()
    deepwave._engine._warned_once_causes.clear()


def _quiet_call(*args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _call(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. One calling convention.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
def test_positional_arguments_are_models_then_grid_spacing_then_dt(name: str) -> None:
    """The required models, ``grid_spacing`` and ``dt``, and nothing else.

    The one shape a caller can rely on without reading a signature:
    ``prop(*models, grid_spacing, dt, **everything_else)``. The models taken
    positionally are exactly the propagator's required, non-perturbation
    ones, which is what makes the leading arguments readable at a call site
    without keywords.
    """
    parameters = list(_signature(name).parameters.values())
    positional = [
        p for p in parameters if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert [p.name for p in positional[-2:]] == ["grid_spacing", "dt"]
    model_names = set(deepwave.spec(name).model_names(max(deepwave.spec(name).ndims)))
    for parameter in positional[:-2]:
        assert parameter.name in model_names, parameter.name
        assert parameter.default is inspect.Parameter.empty, parameter.name
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters[len(positional) :]
    )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_optional_and_perturbation_models_are_keyword_only(name: str) -> None:
    """A model that is optional, or a Born perturbation, is passed by name.

    Only the models that every call of this propagator must supply are
    positional. A perturbation (``dv``, ``dlamb``, ...) and a model that
    exists at some dimensionalities but not others (``c66``, ``tilt``) are
    keyword-only, so that a signature does not change arity with ``ndim``.
    """
    spec = deepwave.spec(name)
    parameters = _signature(name).parameters
    for ndim in spec.ndims:
        for model_name in _keyword_models(name, ndim):
            assert parameters[model_name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[model_name].default is None


def test_shared_keywords_have_one_default_and_one_annotation() -> None:
    """A keyword two propagators share means the same thing on both.

    Name, default *and* annotation: a wrapper that forwards a keyword set to
    whichever propagator it was handed can only be written if the keyword it
    forwards has one meaning. A shared name with two defaults is a silent
    behaviour change on switching propagator; a shared name with two
    annotations is one a type checker cannot see through.
    """
    defaults: dict[str, tuple[str, Any]] = {}
    annotations: dict[str, tuple[str, Any]] = {}
    for name in PROPAGATORS:
        for parameter in _signature(name).parameters.values():
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                continue
            seen = defaults.setdefault(parameter.name, (name, parameter.default))
            same = seen[1] == parameter.default or seen[1] is parameter.default
            assert same, (
                f"{parameter.name}: {seen[0]}={seen[1]!r} "
                f"but {name}={parameter.default!r}"
            )
            seen_annotation = annotations.setdefault(
                parameter.name, (name, parameter.annotation)
            )
            assert seen_annotation[1] == parameter.annotation, (
                f"{parameter.name}: {seen_annotation[0]} and {name} annotate it "
                "differently"
            )


def test_shared_configuration_tail_keeps_one_relative_order() -> None:
    """Shared configuration keywords keep one order as solvers omit options."""
    tails = {}
    for name in PROPAGATORS:
        names = [
            p.name
            for p in _signature(name).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        tails[name] = tuple(names[names.index("initial_state") :])
    for left in PROPAGATORS:
        for right in PROPAGATORS:
            shared = set(tails[left]) & set(tails[right])
            assert tuple(p for p in tails[left] if p in shared) == tuple(
                p for p in tails[right] if p in shared
            ), (left, right)


@pytest.mark.parametrize("form", FORMS)
def test_module_forward_shares_the_functional_configuration_tail(form: str) -> None:
    """The Module's ``forward`` carries the same tail as its functional."""
    del form
    for name in PROPAGATORS:
        functional = [
            p.name
            for p in _signature(name).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        forward = [
            p.name
            for p in inspect.signature(MODULES[name].forward).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        tail = functional[functional.index("initial_state") :]
        assert forward[forward.index("initial_state") :] == tail, name


@pytest.mark.parametrize("name", PROPAGATORS)
def test_channel_keyword_groups_follow_the_spec_channel_order(name: str) -> None:
    """The signature lists a channel group in the order the spec declares it.

    The spec's channel order is the order of the returned receiver tuple, so
    a signature that lists the same channels in a different order gives a
    reader two orders to learn for one set of channels.
    """
    spec = deepwave.spec(name)
    ndim = max(spec.ndims)
    parameters = [
        p.name
        for p in _signature(name).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    groups: list[tuple[str, Callable[[str], str]]] = [
        ("source_amplitudes", spec.source_amplitudes_kwarg),
        ("source_locations", spec.source_locations_kwarg),
        ("source_mask", spec.source_mask_kwarg),
    ]
    for label, kwarg in groups:
        expected = [kwarg(c) for c in spec.source_channels(ndim)]
        found = [p for p in parameters if p in set(expected)]
        assert found == expected, (name, label, found, expected)
    for label, kwarg in (
        ("receiver_locations", spec.receiver_locations_kwarg),
        ("receiver_mask", spec.receiver_mask_kwarg),
    ):
        expected = [kwarg(c) for c in spec.receiver_channels(ndim)]
        found = [p for p in parameters if p in set(expected)]
        assert found == expected, (name, label, found, expected)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_state_keywords_follow_the_spec_return_order(name: str) -> None:
    """``<field>_0`` keywords are listed in the order the fields are returned.

    An output is fed straight back in as ``initial_state``, so the two lists
    describe the same thing and are ordered the same way.
    """
    spec = deepwave.spec(name)
    ndim = max(spec.ndims)
    expected = [n for n in spec.wavefield_names(ndim) if n != "auxiliary_state"]
    parameters = [
        p.name
        for p in _signature(name).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    found = [p for p in parameters if p in set(expected)]
    assert found == expected, (name, found, expected)


# ---------------------------------------------------------------------------
# 2. The survey arguments: two spellings, one meaning.
# ---------------------------------------------------------------------------


def _tensors(out: Any) -> list[torch.Tensor]:
    return [leaf for leaf in pytree.tree_leaves(out) if isinstance(leaf, torch.Tensor)]


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_mapping_and_flat_survey_forms_are_the_same_call(name: str, form: str) -> None:
    """``sources={ch: ...}`` and ``source_amplitudes_<ch>=`` compute the same.

    Not merely "both run": the two spellings must be bit-identical, because
    a user choosing between them is choosing a notation, not a numerical
    method.
    """
    flat = _quiet_call(name, form, survey="flat")
    mapped = _quiet_call(name, form, survey="mapping")
    assert flat.names == mapped.names
    left, right = _tensors(flat), _tensors(mapped)
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_every_receiver_output_has_the_data_rank(
    name: str, form: str, ndim: int
) -> None:
    """A channel that recorded nothing still reports ``[n_shots, 0, nt]``.

    An unrequested receiver channel is returned rather than omitted, and what
    it returns is shaped like the data it would have held: the same shot
    count and the same number of time samples as a requested channel, with
    zero receivers. That is what makes
    ``torch.cat(list(out.receivers.values()), dim=1)`` a valid multichannel
    gather whether or not every channel was asked for.
    """
    if ndim not in deepwave.spec(name).ndims:
        pytest.skip(f"{name} has no {ndim}-D implementation")
    out = _quiet_call(name, form, ndim=ndim)
    n_shots = _amplitudes().shape[0]
    views = [out.receivers]
    if deepwave.spec(name).has_bg_receivers:
        views.append(out.bg_receivers)
    requested = 0
    for view in views:
        for channel, amplitudes in view.items():
            assert amplitudes.dim() == 3, (name, channel, amplitudes.shape)
            assert amplitudes.shape[0] == n_shots, (name, channel)
            assert amplitudes.shape[2] == out.info.nt, (name, channel)
            requested += amplitudes.shape[1]
        assert torch.cat(list(view.values()), dim=1).shape == (
            n_shots,
            sum(a.shape[1] for a in view.values()),
            out.info.nt,
        )
    assert requested > 0, name


def test_empty_receiver_preserves_dtype_device_and_autograd_link() -> None:
    """Reshaping an unrequested channel keeps its differentiable empty value."""
    model = torch.full((32, 32), 1500.0, dtype=torch.float64).requires_grad_()
    source = torch.ones(1, 1, 8, dtype=torch.float64).requires_grad_()
    out = deepwave.scalar(
        model,
        4.0,
        0.001,
        source_amplitudes=source,
        source_locations=torch.tensor([[[16, 16]]]),
        pml_width=4,
        pml_freq=25.0,
        max_vel=1500.0,
    )
    empty = out.receiver_amplitudes
    assert empty.shape == (1, 0, 8)
    assert empty.dtype == source.dtype
    assert empty.device == source.device
    assert empty.requires_grad
    model_grad, source_grad = torch.autograd.grad(empty.sum(), (model, source))
    assert torch.count_nonzero(model_grad) == 0
    assert torch.count_nonzero(source_grad) == 0


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("nt", [_NT - 2, _NT + 3])
def test_nt_with_source_amplitudes_pads_or_truncates(
    name: str, form: str, nt: int
) -> None:
    """``nt`` alongside source amplitudes sets the run's length.

    A shorter record is zero-padded and a longer one truncated, in the
    caller's own time samples, so the answer is bit-for-bit what passing a
    hand-fitted record would have given.
    """
    amplitudes = _amplitudes()
    fitted = (
        amplitudes[..., :nt]
        if nt < amplitudes.shape[-1]
        else torch.nn.functional.pad(amplitudes, (0, nt - amplitudes.shape[-1]))
    )
    survey = _survey_kwargs(name, 2, mapping=False)
    amplitudes_kwarg = next(k for k in survey if k.startswith("source_amplitudes"))
    with_nt = _quiet_call(name, form, survey=None, **survey, nt=nt)
    hand = _quiet_call(name, form, survey=None, **{**survey, amplitudes_kwarg: fitted})
    assert with_nt.info.nt == nt
    assert hand.info.nt == nt
    left, right = _tensors(with_nt), _tensors(hand)
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert torch.equal(a, b), name


def test_invalid_nt_with_source_amplitudes_uses_deepwave_error() -> None:
    """Validate nt before fitting an otherwise valid source record to it."""
    from deepwave import DeepwaveTypeError

    with pytest.raises(DeepwaveTypeError, match="nt must be an int or None"):
        deepwave.scalar(
            torch.full((32, 32), 1500.0),
            4.0,
            0.001,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[16, 16]]]),
            nt="bad",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("nt", [8, 12])
@pytest.mark.parametrize("segments", [None, 2])
def test_explicit_nt_preserves_source_gradient_link(
    nt: int, segments: Optional[int]
) -> None:
    """Padding and truncation agree with an explicit record, including grads."""
    base = torch.full((32, 32), 1500.0, dtype=torch.float64)
    wavelet = torch.arange(1, 11, dtype=torch.float64).reshape(1, 1, -1)
    fitted = (
        wavelet[..., :nt]
        if nt < wavelet.shape[-1]
        else torch.nn.functional.pad(wavelet, (0, nt - wavelet.shape[-1]))
    )
    common = {
        "source_locations": torch.tensor([[[16, 16]]]),
        "receiver_locations": torch.tensor([[[17, 17]]]),
        "pml_width": 4,
        "pml_freq": 25.0,
        "max_vel": 1500.0,
        "checkpoint_segments": segments,
    }

    def run(
        source: torch.Tensor, explicit_nt: Optional[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = base.clone().requires_grad_()
        source = source.clone().requires_grad_()
        out = deepwave.scalar(
            model, 4.0, 0.004, source_amplitudes=source, nt=explicit_nt, **common
        )
        loss = out.receiver_amplitudes.square().sum()
        # The segment wrapper defaults to reentrant checkpointing, which
        # supports backward() but not torch.autograd.grad().
        loss.backward()
        model_grad, source_grad = model.grad, source.grad
        assert model_grad is not None
        assert source_grad is not None
        assert out.info.nt == nt
        assert out.receiver_amplitudes.abs().sum() > 0
        return out.receiver_amplitudes, model_grad, source_grad

    actual, model_grad, source_grad = run(wavelet, nt)
    expected, expected_model_grad, expected_source_grad = run(fitted, None)
    assert torch.equal(actual, expected)
    assert torch.equal(model_grad, expected_model_grad)
    assert torch.equal(
        source_grad[..., : min(nt, wavelet.shape[-1])],
        expected_source_grad[..., : min(nt, wavelet.shape[-1])],
    )
    if nt < wavelet.shape[-1]:
        assert torch.count_nonzero(source_grad[..., nt:]) == 0


@pytest.mark.parametrize("name", PROPAGATORS)
def test_mapping_form_accepts_every_spec_channel_name(name: str) -> None:
    """Every channel the spec declares is a key the mapping argument takes.

    Both directions matter. A declared channel that the argument rejects is a
    published name that cannot be used; an accepted key that the spec does
    not declare is a name no wrapper can discover.
    """
    spec = deepwave.spec(name)
    ndim = 2 if 2 in spec.ndims else spec.ndims[0]
    amplitudes, locations = _amplitudes(), _locations(ndim)
    source = {spec.source_channels(ndim)[0]: (amplitudes, locations)}
    receiver = {spec.receiver_channels(ndim)[0]: locations}
    for channel in spec.source_channels(ndim):
        out = _quiet_call(
            name,
            survey=None,
            sources={channel: (amplitudes, locations)},
            receivers=receiver,
        )
        assert isinstance(out, deepwave.PropagatorOutput)
        assert out.names == spec.output_names(ndim)
    for channel in spec.receiver_channels(ndim):
        out = _quiet_call(
            name, survey=None, sources=source, receivers={channel: locations}
        )
        assert channel in out.receivers
    for channel in spec.bg_receiver_channels(ndim):
        out = _quiet_call(
            name,
            survey=None,
            sources=source,
            receivers=receiver,
            bg_receivers={channel: locations},
        )
        assert channel in out.bg_receivers


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("parameter", ["sources", "receivers"])
def test_unknown_channel_name_raises_one_error_shape(name: str, parameter: str) -> None:
    """An unknown mapping key is rejected the same way everywhere.

    The message begins with the propagator's name, quotes the key that was
    not understood, lists the ones that are, and names the introspection call
    that produces that list -- so the fix is visible from the traceback
    without opening the documentation.
    """
    ndim = 2 if 2 in deepwave.spec(name).ndims else deepwave.spec(name).ndims[0]
    value: Any = (
        (_amplitudes(), _locations(ndim))
        if parameter == "sources"
        else _locations(ndim)
    )
    other = "receivers" if parameter == "sources" else "sources"
    valid: Any = (
        {
            deepwave.spec(name).source_channels(ndim)[0]: (
                _amplitudes(),
                _locations(ndim),
            )
        }
        if other == "sources"
        else {deepwave.spec(name).receiver_channels(ndim)[0]: _locations(ndim)}
    )
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _quiet_call(
            name,
            survey=None,
            **{other: valid, parameter: {"definitely_not_a_channel": value}},
        )
    message = str(excinfo.value)
    assert message.startswith(f"{name}: {parameter}: ")
    assert "unknown channel 'definitely_not_a_channel'" in message
    assert f"deepwave.spec(deepwave.{name})." in message
    advice = re.search(r"deepwave\.spec\(deepwave\.\w+\)\.(\w+)\(ndim\)", message)
    assert advice is not None, message
    assert hasattr(deepwave.spec(name), advice.group(1)), (
        f"{name}: the {parameter}= error advises "
        f"spec.{advice.group(1)}(ndim), which PropagatorSpec does not have"
    )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_unknown_mapping_key_type_raises_one_error_shape(name: str) -> None:
    """A non-string mapping key is a ``DeepwaveTypeError`` everywhere."""
    with pytest.raises(deepwave.DeepwaveTypeError) as excinfo:
        _quiet_call(
            name,
            survey=None,
            sources={0: (_amplitudes(), _locations(2))},
            receivers={deepwave.spec(name).receiver_channels(2)[0]: _locations(2)},
        )
    assert str(excinfo.value).startswith(f"{name}: sources ")


@pytest.mark.parametrize("name", PROPAGATORS)
def test_deprecated_source_channel_spellings_warn_one_way(name: str) -> None:
    """A deprecated channel spelling keeps working and says so.

    The spec publishes the deprecated spellings, so this check needs no table
    of its own: whatever ``deprecated_source_channels`` lists must be
    accepted, must warn under the one deprecation category, and must name its
    replacement and the release that removes it.
    """
    spec = deepwave.spec(name)
    deprecated = dict(spec.deprecated_source_channels())
    if not deprecated:
        pytest.skip(f"{name} has no deprecated source channel spelling")
    for old, new in deprecated.items():
        with pytest.warns(deepwave.DeepwaveFutureWarning) as record:
            _call(
                name,
                survey=None,
                sources={old: (_amplitudes(), _locations(2))},
                receivers={spec.receiver_channels(2)[0]: _locations(2)},
            )
        messages = [str(w.message) for w in record]
        assert any(new in m for m in messages), messages
        assert any("0.4.0" in m for m in messages), messages
        assert any("DeepwaveFutureWarning" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 3. Swapping one propagator for another.
# ---------------------------------------------------------------------------


class _RecordingCallback:
    """A callback that records the step indices it was invoked at."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def __call__(self, state: deepwave.CallbackState) -> None:
        self.steps.append(state.step)


def _common_keywords(callback: Any) -> dict[str, Any]:
    """Every shared keyword this file can set, at a *non-default* value.

    Non-default deliberately: a keyword that all nine accept only because
    they all ignore it at its default would prove nothing. Each value here is
    one the propagator has to act on.

    Three groups are absent, and each is absent for a reason the spec states
    rather than a reason the caller has to memorise. The physics options
    (``mpml_fraction``, ``surface_method``, ``q_freq``, ``q_n_mechanisms``,
    ``q_band``) are listed per propagator in
    :attr:`deepwave.PropagatorSpec.supported_options`, and a non-default value
    of one a propagator does not implement raises rather than being ignored.
    ``backward_callback_needs_forward_wavefield`` is
    :attr:`deepwave.PropagatorSpec.has_forward_wavefield`. The remainder --
    ``nt``, ``origin``, ``checkpoint_segments`` -- are exercised separately
    below, because they are mutually exclusive with members of this set on
    all nine alike.
    """
    return {
        "accuracy": 2,
        "pml_width": 3,
        "boundary": "pml",
        "pml_freq": 12.0,
        "max_vel": 3000.0,
        "courant_number": 0.4,
        "survey_pad": 2,
        "model_gradient_sampling_interval": 2,
        "freq_taper_frac": 0.1,
        "time_pad_frac": 0.1,
        "time_taper": True,
        "check_output_finite": False,
        "forward_callback": callback,
        "callback_frequency": 2,
        "backend": "eager",
        "storage_compression": False,
        "location_interpolation": "hicks",
        "location_interpolation_halfwidth": 2,
    }


@pytest.mark.parametrize("form", FORMS)
def test_one_keyword_dict_drives_every_propagator(form: str) -> None:
    """The swap: one keyword dictionary, nine propagators, one code path.

    This is the property the whole interface exists to provide. The call is
    assembled from the spec alone -- models from ``model_names``, one source
    quantity and one receiver quantity from the channel lists -- and *every*
    other keyword is the same object for all nine. The results are then
    read back through one code path: receivers by channel name, the state by
    field name, and the metadata off ``out.info``.

    A propagator that lacked one of these keywords, spelled it differently,
    or acted on it differently would fail here rather than in a user's script
    a year after the interface was frozen.
    """
    callback = _RecordingCallback()
    common = _common_keywords(callback)
    results = {}
    for name in PROPAGATORS:
        spec = deepwave.spec(name)
        # Identical dict object every time: nothing is adjusted per
        # propagator, and nothing a propagator does may mutate it.
        before = dict(common)
        options = {"q_freq": _FREQ} if "q_freq" in spec.supported_options else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _call(name, form, **common, **options)
        assert common == before, f"{name} mutated the shared keyword dictionary"

        # One read-back path for all nine.
        receivers = out.receivers
        assert tuple(receivers) == spec.receiver_channels(2)
        recorded = receivers[spec.receiver_channels(2)[0]]
        assert recorded.shape[:2] == (1, 1)
        assert recorded.shape[2] == out.info.nt
        state = out.final_state
        assert tuple(state) == spec.wavefield_names(2)
        assert set(state) == set(out.info.wavefield_names)
        results[name] = (
            out.info.nt,
            out.info.pml_freq,
            out.info.pml_width,
            out.info.max_vel,
            out.info.origin,
            out.info.extent,
            out.info.backend,
            out.info.location_interpolation_used,
            out.info.dtype,
        )

    # Every propagator resolved the shared keywords to the same answers.
    assert len(set(results.values())) == 1, results
    assert callback.steps, "forward_callback was never invoked"
    assert set(callback.steps) == set(range(0, _NT, 2))


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"origin": [1, 1]}, id="origin"),
        pytest.param({"checkpoint_segments": 2}, id="checkpoint_segments"),
    ],
)
def test_the_mutually_exclusive_keywords_also_swap(
    form: str, extra: dict[str, Any]
) -> None:
    """``origin`` and ``checkpoint_segments`` behave the same on all nine.

    They are kept out of the one shared dictionary only because they exclude
    members of it, not because a propagator treats them differently.
    """
    results = {}
    for name in PROPAGATORS:
        spec = deepwave.spec(name)
        options = {"q_freq": _FREQ} if "q_freq" in spec.supported_options else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _call(
                name,
                form,
                accuracy=2,
                pml_width=3,
                pml_freq=12.0,
                max_vel=3000.0,
                backend="eager",
                **extra,
                **options,
            )
        results[name] = (out.info.nt, out.info.origin, out.info.extent)
        assert tuple(out.final_state) == spec.wavefield_names(2)
    assert len(set(results.values())) == 1, results


@pytest.mark.parametrize("form", FORMS)
def test_a_sourceless_continuation_runs_the_same_way_on_all_nine(form: str) -> None:
    """``initial_state=`` plus ``nt=``, with no sources at all.

    A continuation with no new sources is how a long record is run in
    pieces, and it is the call that has the least information to work from:
    there are no source locations to take the number of dimensions or the
    survey region from, only the state being continued. Every propagator has
    to reach the same answer from the same inputs.
    """
    results = {}
    for name in PROPAGATORS:
        spec = deepwave.spec(name)
        options = {"q_freq": _FREQ} if "q_freq" in spec.supported_options else {}
        first = _quiet_call(name, form, backend="eager", **options)
        receiver = {spec.receiver_channels(2)[0]: _locations(2)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            second = _call(
                name,
                form,
                survey=None,
                receivers=receiver,
                initial_state=first,
                nt=_NT,
                backend="eager",
                **options,
            )
        assert second.names == first.names
        assert second.info.nt == _NT
        assert second.info.ndim == first.info.ndim
        assert second.info.origin == first.info.origin
        assert second.info.extent == first.info.extent
        results[name] = (second.info.nt, second.info.ndim, second.info.extent)
    assert len(set(results.values())) == 1, results


@pytest.mark.parametrize("name", PROPAGATORS)
def test_swapping_only_needs_the_models_and_the_quantity_names(name: str) -> None:
    """Everything a generic caller must know is answerable from the spec.

    The three questions a wrapper asks when it is handed a propagator it has
    never seen: which model tensors does it take, what can I inject, and what
    can I record. Each is answered by the spec, and each answer is directly
    usable as a call argument and as an output key.
    """
    spec = deepwave.spec(name)
    ndim = 2
    models = {m: _model(m, ndim) for m in spec.model_names(ndim)}
    positional = [models.pop(m) for m in _positional_models(name)]
    source_channel = spec.source_channels(ndim)[0]
    receiver_channel = spec.receiver_channels(ndim)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = getattr(deepwave, name)(
            *positional,
            _DX,
            _DT,
            **models,
            sources={source_channel: (_amplitudes(), _locations(ndim))},
            receivers={receiver_channel: _locations(ndim)},
            **_config_kwargs(name),
        )
    assert out.receivers[receiver_channel].shape == (1, 1, _NT)
    assert out.final_state["auxiliary_state"] is out.auxiliary_state
    continued = getattr(deepwave, name)(
        *positional,
        _DX,
        _DT,
        **models,
        sources={source_channel: (_amplitudes(), _locations(ndim))},
        receivers={receiver_channel: _locations(ndim)},
        initial_state=out,
        **_config_kwargs(name),
    )
    assert continued.names == out.names


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_off_node_locations_are_interpolated_on_every_channel(
    name: str, form: str
) -> None:
    """All nine interpolate a float location, on every channel they take.

    A coordinate given as a float that does not sit on a node is placed by
    the method of Hicks rather than truncated to a cell, and that is true of
    every source and every receiver channel of every propagator, not of a
    subset of them. The run reports that it interpolated, and the receivers
    record a signal, so the check cannot pass on a call that quietly did
    nothing.
    """
    ndim = 2
    spec = deepwave.spec(name)
    off_node = torch.tensor([[[_N / 2 + 0.3, _N / 2 + 0.4]]], dtype=_DTYPE)
    survey: dict[str, Any] = {}
    for channel in spec.source_channels(ndim):
        survey[spec.source_amplitudes_kwarg(channel)] = _amplitudes()
        survey[spec.source_locations_kwarg(channel)] = off_node
    for channel in spec.receiver_channels(ndim):
        survey[spec.receiver_locations_kwarg(channel)] = off_node
    if spec.has_bg_receivers:
        for channel in spec.bg_receiver_channels(ndim):
            survey[spec.bg_receiver_locations_kwarg(channel)] = off_node

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _segment_call(name, form, ndim, survey, _config_kwargs(name))

    assert out.info.location_interpolation_used == "hicks"
    # The scattered receivers of a Born propagator are zero here, because its
    # perturbation models are zero; the background ones carry the signal.
    recorded = [
        getattr(out, field)
        for field in deepwave.receiver_names(name, ndim)
        if not field.startswith("sc_")
    ]
    assert recorded
    for amplitudes in recorded:
        assert amplitudes.abs().max() > 0


def _segment_survey(name: str, ndim: int, start: int, stop: int) -> dict[str, Any]:
    """The default survey, with only time samples ``start:stop`` of the source."""
    spec = deepwave.spec(name)
    kwargs = _survey_kwargs(name, ndim, mapping=False)
    channel = spec.source_channels(ndim)[0]
    amplitudes_kwarg = spec.source_amplitudes_kwarg(channel)
    kwargs[amplitudes_kwarg] = kwargs[amplitudes_kwarg][..., start:stop]
    return kwargs


def _segment_call(
    name: str,
    form: str,
    ndim: int,
    survey: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """One run of ``name``, with exactly the arguments given and no others."""
    kwargs: dict[str, Any] = {**survey, **config}
    for model_name in _keyword_models(name, ndim):
        kwargs[model_name] = _model(model_name, ndim)
    models = [_model(m, ndim) for m in _positional_models(name)]
    if form == "function":
        return getattr(deepwave, name)(*models, _DX, _DT, **kwargs)
    constructor = {m: kwargs.pop(m) for m in _keyword_models(name, ndim) if m in kwargs}
    return MODULES[name](*models, _DX, **constructor)(_DT, **kwargs)


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_continuation_needs_no_numerical_keywords_repeated(
    name: str, form: str
) -> None:
    """A continuation that omits every numerical keyword equals the single run.

    The parameters a run resolves for itself -- ``pml_freq`` from the source
    amplitudes, ``max_vel`` from the model -- have nothing to be resolved
    from in the call that continues it: its share of the source record is a
    different signal, and the first call's ``max_vel`` need not have come
    from the model at all. They are recorded in the state, and a continuation
    that leaves them out takes them from there, so splitting a run in two and
    passing nothing but the state reproduces the single run exactly and says
    nothing while doing it.

    The split is bitwise because both halves take the same internal time step
    (no sub-stepping at this ``dt``), which is the condition the propagators'
    ``initial_state`` documentation names.
    """
    ndim = 2
    split = _NT // 2
    # ``max_vel`` is deliberately above the model's own maximum and
    # ``pml_freq`` away from the truncated segments' own spectra, so that a
    # continuation which re-derived either would get a different number.
    config = {"pml_freq": _FREQ, "max_vel": 3000.0}
    if "q_freq" in deepwave.spec(name).supported_options:
        config["q_freq"] = _FREQ
    required = {k: v for k, v in config.items() if k == "q_freq"}

    _forget_once_per_process_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = _segment_call(
            name, form, ndim, _segment_survey(name, ndim, 0, _NT), config
        )
        first = _segment_call(
            name, form, ndim, _segment_survey(name, ndim, 0, split), config
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = _segment_call(
            name,
            form,
            ndim,
            {
                **_segment_survey(name, ndim, split, _NT),
                "initial_state": first,
            },
            required,
        )
    assert [str(w.message) for w in caught] == []
    assert second.info.pml_freq == whole.info.pml_freq
    assert second.info.max_vel == whole.info.max_vel
    assert second.info.internal_dt == whole.info.internal_dt

    nonzero = False
    for field in deepwave.receiver_names(name, ndim):
        joined = torch.cat([getattr(first, field), getattr(second, field)], dim=-1)
        assert torch.equal(joined, getattr(whole, field)), field
        nonzero = nonzero or bool(joined.any())
    assert nonzero, "the comparison would be vacuous with all-zero receivers"
    for field in deepwave.wavefield_names(name, ndim):
        if field == "auxiliary_state":
            continue
        assert torch.equal(getattr(second, field), getattr(whole, field)), field
    for field in whole.auxiliary_state.names:
        assert torch.equal(
            second.auxiliary_state[field], whole.auxiliary_state[field]
        ), field


# ---------------------------------------------------------------------------
# 4. The output object.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_output_access_patterns_agree_with_the_spec(name: str, form: str) -> None:
    """By name, by declared slot and as a sequence: three views, one object.

    ``spec.output_names(ndim)`` is the contract. The returned object's
    ``names`` must be exactly that tuple, indexing it must agree with
    attribute access at every position, and the derived views
    (``final_state``, ``receivers``, ``bg_receivers``) must be keyed by
    exactly what the spec says they are.
    """
    spec = deepwave.spec(name)
    out = _quiet_call(name, form)
    assert type(out) is deepwave.PropagatorOutput
    assert out.names == spec.output_names(2)
    assert len(out) == len(out.names)
    for index, field in enumerate(out.names):
        assert out[index] is getattr(out, field), (name, field)
    assert tuple(out) == tuple(out[i] for i in range(len(out)))
    assert tuple(out.final_state) == spec.wavefield_names(2)
    assert tuple(out.receivers) == spec.receiver_channels(2)
    if spec.has_bg_receivers:
        assert tuple(out.bg_receivers) == spec.bg_receiver_channels(2)
    else:
        with pytest.raises(AttributeError):
            _ = out.bg_receivers
    assert isinstance(out.auxiliary_state, deepwave.AuxiliaryState)
    with pytest.raises(AttributeError):
        _ = out.definitely_not_a_field


@pytest.mark.parametrize("form", FORMS)
def test_run_info_fields_are_identical_across_propagators(form: str) -> None:
    """One ``info`` record: the same fields, holding the same kinds of value.

    A field that exists on some propagators and not others, or that holds a
    ``str`` on one and ``None`` on another, cannot be read by code that does
    not already know which propagator it is holding.

    The two attenuation-model fields ``q_freq`` and ``q_band`` are the one
    exception, and a stated one: they describe a physics a propagator either
    has or has not, so they are ``None`` exactly on the propagators whose
    spec does not list ``q_freq`` among its supported options. That is
    checked here instead of the type equality.
    """
    declared = tuple(f.name for f in dataclasses.fields(deepwave.RunInfo))
    physics = ("q_freq", "q_band")
    compared = tuple(f for f in declared if f not in physics)
    reference: Optional[dict[str, str]] = None
    for name in PROPAGATORS:
        info = _quiet_call(name, form).info
        assert isinstance(info, deepwave.RunInfo)
        attenuating = "q_freq" in deepwave.spec(name).supported_options
        assert (info.q_freq is not None) == attenuating, name
        if not attenuating:
            assert info.q_band is None, name
        observed = {f: type(getattr(info, f)).__name__ for f in compared}
        if reference is None:
            reference = observed
        else:
            differences = {
                f: (reference[f], observed[f])
                for f in compared
                if reference[f] != observed[f]
            }
            assert not differences, (name, differences)
        assert info.wavefield_names == deepwave.spec(name).wavefield_names(2)
        assert info.receiver_names == deepwave.spec(name).receiver_names(2)
        assert info.backend
        assert info.backend != "auto"


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_output_pytree_registration_round_trips(name: str, form: str) -> None:
    """Flattening and unflattening an output gives the same object back.

    The leaves are exactly the output's tensors, with ``auxiliary_state``
    expanding into the tensors it holds, and the names and ``info`` travel as
    context, so a rebuilt output is still usable as an ``initial_state=``.
    """
    out = _quiet_call(name, form)
    leaves, treespec = pytree.tree_flatten(out)
    assert leaves
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    rebuilt = pytree.tree_unflatten(leaves, treespec)
    assert type(rebuilt) is deepwave.PropagatorOutput
    assert rebuilt.names == out.names
    assert rebuilt.info == out.info
    assert isinstance(rebuilt.auxiliary_state, deepwave.AuxiliaryState)
    assert rebuilt.auxiliary_state.names == out.auxiliary_state.names
    for a, b in zip(_tensors(rebuilt), _tensors(out)):
        assert a is b


# ---------------------------------------------------------------------------
# 5. Errors and warnings: one condition, one class, one message shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_unknown_keyword_is_a_type_error(name: str, form: str) -> None:
    """An argument that does not exist is rejected, never absorbed."""
    with pytest.raises(TypeError) as excinfo:
        _quiet_call(name, form, definitely_not_an_argument=1)
    assert "definitely_not_an_argument" in str(excinfo.value)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_mismatched_model_shapes_raise_one_error(name: str) -> None:
    """Two models of different spatial shapes: one check, one message shape."""
    positional = _positional_models(name)
    if len(positional) < 2:
        pytest.skip(f"{name} takes one model, so two cannot disagree")
    models = [_model(m, 2) for m in positional]
    models[-1] = models[-1][:-1]
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _quiet_call(name, models=models)
    message = str(excinfo.value)
    assert message.startswith(f"{name}: all models must have the same spatial shape")
    assert positional[0] in message
    assert positional[-1] in message


@pytest.mark.parametrize("attribute", ["dtype", "device"])
def test_mismatched_model_dtype_or_device_raises_one_error(attribute: str) -> None:
    """One model unlike the others is Deepwave's error, named after the call.

    Deepwave's own validation errors are ``Deepwave*Error`` subclasses (so
    ``except deepwave.DeepwaveError`` catches them and ``except
    RuntimeError`` still does) and begin with the propagator's name (so the
    traceback says which call in a multi-propagator script failed).
    """
    if attribute == "device" and not torch.cuda.is_available():
        pytest.skip("a second device is needed to mix devices")
    failures = []
    for name in PROPAGATORS:
        positional = _positional_models(name)
        if len(positional) < 2:
            continue
        models = [_model(m, 2) for m in positional]
        models[-1] = models[-1].float() if attribute == "dtype" else models[-1].cuda()
        try:
            _quiet_call(name, models=models)
        except Exception as error:  # noqa: BLE001
            if not isinstance(error, deepwave.DeepwaveError) or not str(
                error
            ).startswith(f"{name}: "):
                failures.append((name, type(error).__name__, str(error)[:90]))
        else:
            failures.append((name, "no error", ""))
    assert not failures, failures


@pytest.mark.parametrize("name", PROPAGATORS)
def test_out_of_bounds_locations_name_the_argument_that_was_passed(name: str) -> None:
    """A location error names the keyword the caller wrote.

    Checked on every channel, because the argument a propagator validates is
    not always the argument the caller passed: a channel that the propagator
    expands into other channels internally must still report the name that
    was written at the call site.
    """
    spec = deepwave.spec(name)
    ndim = 2 if 2 in spec.ndims else spec.ndims[0]
    far_away = torch.tensor([[[10**6] + [0] * (ndim - 1)]])
    failures = []
    for kind, channels, kwarg in (
        ("source", spec.source_channels(ndim), spec.source_locations_kwarg),
        ("receiver", spec.receiver_channels(ndim), spec.receiver_locations_kwarg),
    ):
        del kind
        for channel in channels:
            argument = kwarg(channel)
            with pytest.raises(deepwave.DeepwaveError) as excinfo:
                _quiet_call(name, **{argument: far_away})
            message = str(excinfo.value)
            assert message.startswith(f"{name}: "), message
            if argument not in message:
                failures.append((argument, message[:110]))
    assert not failures, failures


@pytest.mark.parametrize("name", PROPAGATORS)
def test_locations_without_amplitudes_raise_one_error(name: str) -> None:
    """Locations with no amplitudes is the same mistake on all nine.

    It is the commonest survey-assembly slip, and the propagator that reports
    it differently is the one whose users will not recognise it.
    """
    spec = deepwave.spec(name)
    channel = spec.source_channels(2)[0]
    kwargs = _survey_kwargs(name, 2, mapping=False)
    kwargs.pop(spec.source_amplitudes_kwarg(channel))
    kwargs["nt"] = _NT
    kwargs.update(_config_kwargs(name))
    kwargs.update({m: _model(m, 2) for m in _keyword_models(name, 2)})
    models = [_model(m, 2) for m in _positional_models(name)]
    functional = getattr(deepwave, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(deepwave.DeepwaveRuntimeError) as excinfo:
            functional(*models, _DX, _DT, **kwargs)
    message = str(excinfo.value)
    assert message.startswith(f"{name}: {spec.source_locations_kwarg(channel)} ")
    assert spec.source_amplitudes_kwarg(channel) in message


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize(
    ("value", "exception"),
    [("nope", deepwave.DeepwaveTypeError), (-1.0, deepwave.DeepwaveValueError)],
)
def test_a_rejected_option_value_is_a_deepwave_error(
    name: str, value: Any, exception: type
) -> None:
    """A rejected argument *value* raises Deepwave's own error, named.

    ``pml_freq`` stands for the shared options here: the wrong type and an
    impossible value are two different mistakes, so they raise two different
    classes, and both messages begin with the propagator's name so that a
    traceback out of a script driving several propagators says which call
    failed.
    """
    with pytest.raises(exception) as excinfo:
        _quiet_call(name, pml_freq=value)
    assert str(excinfo.value).startswith(f"{name}: pml_freq ")


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_zero_width_pml_face_warns_one_category(name: str, form: str) -> None:
    """A zero-width absorbing face reflects, and says so under one category.

    It is the idiom for "make this face reflect", and it is silent about a
    large physical choice unless it warns, so every propagator warns and
    every message names the category that silences it.
    """
    _forget_once_per_process_warnings()
    with pytest.warns(deepwave.ZeroWidthFaceWarning) as record:
        _call(name, form, pml_width=0)
    messages = [str(w.message) for w in record]
    assert messages
    for message in messages:
        assert "ZeroWidthFaceWarning" in message, message


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_pml_freq_above_nyquist_warns_the_same_way(name: str, form: str) -> None:
    """A ``pml_freq`` the time step cannot represent warns, never silently runs."""
    with pytest.warns(UserWarning, match="Nyquist") as record:
        _call(name, form, pml_freq=1.0e6)
    assert any("Nyquist" in str(w.message) for w in record), [
        str(w.message) for w in record
    ]


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("form", FORMS)
def test_deprecated_backend_spelling_only_on_released_propagators(
    name: str, form: str
) -> None:
    """The four released APIs retain their alias; new APIs use backend."""
    if name in {"scalar", "scalar_born", "acoustic", "elastic"}:
        with pytest.warns(deepwave.DeepwaveFutureWarning, match="python_backend"):
            _call(name, form, python_backend="eager")
    else:
        with pytest.raises(TypeError, match="python_backend"):
            _call(name, form, python_backend="eager")


# ---------------------------------------------------------------------------
# 6. The Module form.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
def test_module_constructor_takes_the_models_and_forward_takes_dt(name: str) -> None:
    """The split is the same on all nine: models in, ``dt`` and survey out.

    The constructor holds what a Module is for -- the model tensors, which
    become its parameters, and the storage policy those parameters' gradients
    need. Everything that varies between shots is passed to the call.
    """
    init = inspect.signature(MODULES[name].__init__).parameters
    forward = inspect.signature(MODULES[name].forward).parameters
    positional_init = [
        p.name
        for p in init.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != "self"
    ]
    assert positional_init == [*_positional_models(name), "grid_spacing"]
    positional_forward = [
        p.name
        for p in forward.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != "self"
    ]
    assert positional_forward == ["dt"]
    all_models = set()
    for ndim in deepwave.spec(name).ndims:
        all_models.update(deepwave.spec(name).model_names(ndim))
    assert all_models.isdisjoint(set(forward)), name
    for keyword in ("storage", "storage_mode", "storage_path"):
        assert keyword in init, (name, keyword)
        assert keyword in forward, (name, keyword)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_module_requires_grad_flags_are_generated_from_the_spec(name: str) -> None:
    """One ``<model>_requires_grad`` per model, in the spec's model order.

    Generated from the spec rather than written out, so the flags cannot
    drift from the models: exactly one per model the propagator can take at
    any of its dimensionalities, in the order the spec lists them, plus the
    one ``requires_grad`` that sets them all.
    """
    spec = deepwave.spec(name)
    ordered: list[str] = []
    for ndim in spec.ndims:
        for model_name in spec.model_names(ndim):
            if model_name not in ordered:
                ordered.append(model_name)
    init = [
        p.name
        for p in inspect.signature(MODULES[name].__init__).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    flags = [p for p in init if p.endswith("_requires_grad")]
    parameters = _signature(name).parameters
    # A deprecated spelling of a model carries a flag too, so that the old
    # and the new spelling of the same call are the same call.
    aliases = [flag for flag in flags if flag[: -len("_requires_grad")] not in ordered]
    for flag in aliases:
        stem = flag[: -len("_requires_grad")]
        assert stem in parameters, (name, stem)
        assert stem not in ordered, (name, stem)
    assert [f for f in flags if f not in aliases] == [
        f"{m}_requires_grad" for m in ordered
    ], name
    assert init[init.index(flags[0]) - 1] == "requires_grad"


@pytest.mark.parametrize("name", PROPAGATORS)
def test_module_state_dict_keys_are_the_model_names(name: str) -> None:
    """``state_dict`` is keyed by the spec's model names, in its order.

    The state a Module owns is exactly its models, under exactly the names
    the functional takes them by, so a checkpoint can be read without knowing
    which propagator wrote it.
    """
    spec = deepwave.spec(name)
    ndim = max(spec.ndims)
    expected = list(spec.model_names(ndim))
    models = [_model(m, ndim) for m in _positional_models(name)]
    constructor = {m: _model(m, ndim) for m in _keyword_models(name, ndim)}
    module = MODULES[name](*models, _DX, **constructor)
    assert list(module.state_dict()) == expected
    assert [k for k, _ in module.named_parameters()] == expected
    assert not list(module.named_buffers())


@pytest.mark.parametrize("name", PROPAGATORS)
def test_module_requires_grad_handling_is_one_rule(name: str) -> None:
    """``requires_grad`` sets every model; a per-model flag sets exactly one.

    The blanket flag and the per-model flags are the same switch at two
    granularities, and a model's flag is the model's own: setting one must
    not set another.
    """
    spec = deepwave.spec(name)
    model_names = list(spec.model_names(2))
    models = [_model(m, 2) for m in _positional_models(name)]
    constructor = {m: _model(m, 2) for m in _keyword_models(name, 2)}

    default = MODULES[name](*models, _DX, **constructor)
    assert not any(p.requires_grad for p in default.parameters()), name

    everything = MODULES[name](*models, _DX, requires_grad=True, **constructor)
    assert all(p.requires_grad for p in everything.parameters()), name

    for model_name in model_names:
        one = MODULES[name](
            *models, _DX, **constructor, **{f"{model_name}_requires_grad": True}
        )
        flags = {k: v.requires_grad for k, v in one.named_parameters()}
        assert flags[model_name], (name, model_name)
        assert not any(v for k, v in flags.items() if k != model_name), (
            name,
            model_name,
            flags,
        )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_module_and_functional_compute_the_same_thing(name: str) -> None:
    """The two forms are two spellings of one call, not two implementations."""
    functional = _quiet_call(name, "function")
    module = _quiet_call(name, "module")
    assert functional.names == module.names
    for a, b in zip(_tensors(functional), _tensors(module)):
        assert torch.equal(a, b), name


# ---------------------------------------------------------------------------
# 7. Introspection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
def test_spec_is_well_formed(name: str) -> None:
    """Every published spec answers every published question.

    Reached through all three spellings that name a propagator (the
    functional, the Module class and the plain name), because a generic
    wrapper may hold any of them.
    """
    spec = deepwave.spec(name)
    assert spec is deepwave.spec(getattr(deepwave, name))
    assert spec is deepwave.spec(MODULES[name])
    assert spec.name == name
    assert spec.status in ("stable", "provisional", "experimental", "deprecated")
    assert spec.ndims
    assert all(isinstance(n, int) for n in spec.ndims)
    assert all(n > 0 for n in spec.ndims)
    assert isinstance(spec.supported_options, frozenset)
    assert isinstance(spec.has_bg_receivers, bool)
    assert isinstance(spec.has_forward_wavefield, bool)
    for ndim in spec.ndims:
        models = spec.model_names(ndim)
        assert models
        assert len(set(models)) == len(models)
        assert set(models) <= set(_MODEL_VALUES), (name, ndim, models)
        sources = spec.source_channels(ndim)
        receivers = spec.receiver_channels(ndim)
        assert sources
        assert receivers
        outputs = spec.output_names(ndim)
        assert outputs == spec.wavefield_names(ndim) + spec.receiver_names(ndim)
        assert "auxiliary_state" in spec.wavefield_names(ndim)
        background = tuple(
            spec.bg_receiver_output_name(c) for c in spec.bg_receiver_channels(ndim)
        )
        scattered = tuple(spec.receiver_output_name(c) for c in receivers)
        assert spec.receiver_names(ndim) == background + scattered
        if spec.has_bg_receivers:
            assert spec.bg_receiver_channels(ndim)
        else:
            assert spec.bg_receiver_channels(ndim) == ()
        for channel in sources:
            for kwarg in (
                spec.source_amplitudes_kwarg(channel),
                spec.source_locations_kwarg(channel),
                spec.source_mask_kwarg(channel),
            ):
                assert kwarg in _signature(name).parameters, (name, kwarg)
        for channel in receivers:
            for kwarg in (
                spec.receiver_locations_kwarg(channel),
                spec.receiver_mask_kwarg(channel),
            ):
                assert kwarg in _signature(name).parameters, (name, kwarg)
    assert deepwave.wavefield_names(name, spec.ndims[0]) == spec.wavefield_names(
        spec.ndims[0]
    )
    assert deepwave.receiver_names(name, spec.ndims[0]) == spec.receiver_names(
        spec.ndims[0]
    )
    with pytest.raises(deepwave.DeepwaveValueError):
        spec.model_names(max(spec.ndims) + 1)
    with pytest.raises(deepwave.DeepwaveValueError):
        spec.receiver_output_name("definitely_not_a_channel")


def test_spec_rejects_anything_that_is_not_a_propagator() -> None:
    """One resolver, one error, for all three spellings."""
    with pytest.raises(deepwave.DeepwaveValueError):
        deepwave.spec("definitely_not_a_propagator")
    with pytest.raises(deepwave.DeepwaveTypeError):
        deepwave.spec(len)


_DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
_AUTODOC = re.compile(r"^\s*\.\.\s+auto\w+::\s+([\w\.]+)", re.MULTILINE)


def _documented_targets() -> tuple[set[str], str]:
    """Every autodoc target in ``docs/``, and the concatenated source."""
    blob = "\n".join(path.read_text() for path in sorted(_DOCS.rglob("*.rst")))
    return set(_AUTODOC.findall(blob)), blob


def test_every_public_name_is_documented() -> None:
    """Nothing is exported without a reference entry, in either direction.

    ``deepwave.__all__`` is the promise; the reference pages are where it is
    explained. A name in one and not the other means a user either cannot
    find a published name, or is reading about one they cannot import.
    """
    targets, blob = _documented_targets()
    undocumented = []
    for name in deepwave.__all__:
        if name == "__version__":
            continue
        if f"deepwave.{name}" in targets:
            continue
        if isinstance(getattr(deepwave, name), types.ModuleType) and any(
            target.startswith(f"deepwave.{name}.") for target in targets
        ):
            continue
        if re.search(rf"deepwave\.{re.escape(name)}\b", blob):
            continue
        undocumented.append(name)
    assert not undocumented, undocumented
    unexported = [
        target
        for target in targets
        if target.startswith("deepwave.")
        and "." not in target[len("deepwave.") :]
        and target[len("deepwave.") :] not in deepwave.__all__
    ]
    assert not unexported, unexported


@pytest.mark.parametrize("name", PROPAGATORS)
def test_each_propagator_has_its_own_reference_page_entry(name: str) -> None:
    """Both spellings of every propagator are autodocumented."""
    targets, _ = _documented_targets()
    assert f"deepwave.{name}" in targets
    assert f"deepwave.{MODULES[name].__name__}" in targets


# ---------------------------------------------------------------------------
# 8. Docstrings.
# ---------------------------------------------------------------------------

#: A docstring entry name, at exactly four spaces of indentation. Continuation
#: lines and nested bullets are indented further, so this matches entries only.
_ENTRY = re.compile(
    r"^ {4}([A-Za-z_][A-Za-z0-9_]*):(.*?)(?=^ {4}[A-Za-z_][A-Za-z0-9_]*:|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _args_entries(function: Any) -> dict[str, str]:
    """The ``Args:`` entries of a Google-style docstring, name to body."""
    match = re.search(
        r"\nArgs:\n(.*?)\nReturns:\n", inspect.getdoc(function) or "", re.DOTALL
    )
    assert match, f"{function.__name__} has no Args:/Returns: sections"
    return {
        name: " ".join(body.split()) for name, body in _ENTRY.findall(match.group(1))
    }


@pytest.mark.parametrize("name", PROPAGATORS)
def test_docstring_documents_exactly_its_own_parameters(name: str) -> None:
    """Every parameter is documented, and nothing that is not a parameter is.

    Both directions: an undocumented parameter is a published argument nobody
    is told about, and a documented name that is not a parameter is an
    argument that does not exist.
    """
    entries = _args_entries(getattr(deepwave, name))
    parameters = set(_signature(name).parameters)
    assert set(entries) == parameters, {
        "undocumented": sorted(parameters - set(entries)),
        "not a parameter": sorted(set(entries) - parameters),
    }


#: A statement, inside one entry, of what the quantity is measured in: an
#: explicit unit, a dimension expression, a named physical quantity whose
#: dimensions are fixed, or a cross-reference to an entry that gives one.
_DIMENSION = re.compile(
    r"""
    \bunits?\ of\b | \bunit-agnostic\b | \bunit[\ -]consistency\b
    | \bdimensionless\b | \bradians\b | /units\b | \bm/s\b | \bHz\b
    | \b(?:force|pressure|volume|stress|velocity)\s*/
    | \b(?:force|body-force|moment|volume-injection|pressure|stress)[\ -]
      (?:densit|rate)
    | \b1/(?:time|density|volume)\b
    | :func:`[^`]+` | \bas\ ``\w+``
    """,
    re.VERBOSE,
)

#: A sentence that fixes the units of several arguments at once. Deepwave
#: imposes no unit system, so the statement each propagator makes is that one
#: consistent system must be chosen across a named group of arguments.
_UNIT_SENTENCE = re.compile(r"[^.;]*\bunit[^.;]*[.;]")


def _states_dimensions(entries: Mapping[str, str], doc: str, argument: str) -> bool:
    """Whether the docstring says what ``argument`` is measured in.

    Three ways, all of them used by the nine as they stand: the argument's
    own entry gives a unit or a dimensioned quantity; a unit-consistency
    sentence names the argument (individually, or collectively as "the
    stiffnesses"); or the argument is a perturbation of a model whose own
    entry answers the question.
    """
    if _DIMENSION.search(entries.get(argument, "")):
        return True
    flat = " ".join(doc.split())
    for sentence in _UNIT_SENTENCE.findall(flat):
        if f"``{argument}``" in sentence:
            return True
        if re.fullmatch(r"c\d+", argument) and "stiffness" in sentence:
            return True
    if argument.startswith("d") and argument[1:] in entries:
        return _states_dimensions(entries, doc, argument[1:])
    return False


@pytest.mark.parametrize("name", PROPAGATORS)
def test_physical_arguments_state_their_dimensions(name: str) -> None:
    """Every physical argument says what it is measured in.

    Deepwave does not impose a unit system, so "what is this in?" is answered
    either by an explicit dimension (``units of force/volume``), by the
    statement that the caller chooses one consistent system for a named group
    of arguments, or by a cross-reference to the entry that gives one. Every
    model tensor, ``grid_spacing``, ``dt`` and every source-amplitude
    argument has to answer it: they are the arguments whose wrong scaling
    produces a plausible wrong answer rather than an error.
    """
    spec = deepwave.spec(name)
    doc = inspect.getdoc(getattr(deepwave, name)) or ""
    entries = _args_entries(getattr(deepwave, name))
    required = [*spec.model_names(max(spec.ndims)), "grid_spacing", "dt"]
    required += [
        spec.source_amplitudes_kwarg(c) for c in spec.source_channels(max(spec.ndims))
    ]
    silent = [
        argument
        for argument in required
        if not _states_dimensions(entries, doc, argument)
    ]
    assert not silent, {argument: entries.get(argument, "")[:90] for argument in silent}


@pytest.mark.parametrize("name", PROPAGATORS)
def test_receiver_outputs_are_documented_with_their_quantity(name: str) -> None:
    """The ``Returns:`` section names every receiver field the run produces."""
    doc = inspect.getdoc(getattr(deepwave, name)) or ""
    match = re.search(r"\nReturns:\n(.*)", doc, re.DOTALL)
    assert match, name
    returns = match.group(1)
    if re.search(r":func:`deepwave\.\w+`", returns):
        # A propagator whose outputs are another's may say so rather than
        # repeat the list; the entry it points at is checked in its own right.
        return
    for field in deepwave.spec(name).receiver_names(2):
        assert field in returns, (name, field)

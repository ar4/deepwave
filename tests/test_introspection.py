"""Introspection: ``deepwave.spec`` and the reflection APIs that use it.

The point of these is that a generic wrapper -- and Deepwave's
own ``record_boundary``/``RunningDFT``/``estimate_memory``/``run_segments``
-- should reflect over a published vocabulary instead of matching strings or
guessing from a positional slot. The tests therefore check both that the
vocabulary is right and that the four APIs actually use it.
"""

import copy
import dataclasses
import pickle
import warnings
from typing import Any

import pytest
import torch

import deepwave

PROPAGATORS = deepwave._engine.PROPAGATOR_NAMES


def test_private_spec_mapping_defaults_use_factories() -> None:
    """The public specs must import on Python 3.11's dataclasses too."""
    fields = {f.name: f for f in dataclasses.fields(deepwave._engine.PropagatorSpec)}
    for name in ("channel_aliases", "deprecated_source_channels"):
        assert fields[name].default is dataclasses.MISSING
        assert fields[name].default_factory() is deepwave._engine._EMPTY_ALIASES


def _ndims(name: str) -> list[int]:
    """Every dimensionality this propagator declares, 1-D included."""
    return list(deepwave.spec(name).ndims)


def _name_and_ndim_params() -> list[Any]:
    """Every ``(propagator, ndim)`` pair, 1-D included, as test parameters."""
    return [
        pytest.param(name, ndim, id=f"{name}-{ndim}d")
        for name in PROPAGATORS
        for ndim in _ndims(name)
    ]


NAME_AND_NDIM = _name_and_ndim_params()


# ---------------------------------------------------------------------------
# deepwave.spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
def test_spec_accepts_callable_module_and_name(name: str) -> None:
    """The three spellings of a propagator all resolve to the same object."""
    functional = getattr(deepwave, name)
    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    assert deepwave.spec(functional) is deepwave.spec(name)
    assert deepwave.spec(module_class) is deepwave.spec(name)
    assert deepwave.spec(name).name == name


def test_spec_rejects_a_non_propagator() -> None:
    """Anything else raises, naming the three accepted forms."""
    with pytest.raises(deepwave.DeepwaveTypeError) as excinfo:
        deepwave.spec(len)
    message = str(excinfo.value)
    assert isinstance(excinfo.value, TypeError)
    assert "deepwave.elastic" in message
    assert "deepwave.Elastic" in message
    with pytest.raises(deepwave.DeepwaveValueError, match="not the name"):
        deepwave.spec("nope")


@pytest.mark.parametrize("name", PROPAGATORS)
def test_spec_fields_are_immutable_and_picklable(name: str) -> None:
    """The projection is plain, frozen data: comparable, copyable, picklable."""
    spec = deepwave.spec(name)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]
    assert spec == deepwave.spec(name)
    assert pickle.loads(pickle.dumps(spec)) == spec  # noqa: S301
    assert copy.deepcopy(spec) == spec
    # The repr shows the interface, not a set of function objects.
    assert "function" not in repr(spec)
    assert "lambda" not in repr(spec)
    assert name in repr(spec)


@pytest.mark.parametrize(("name", "ndim"), NAME_AND_NDIM)
def test_spec_channel_lists_match_signatures(name: str, ndim: int) -> None:
    """Every declared channel has real keywords, and vice versa.

    The invariant that stops the published vocabulary drifting from the
    signatures it describes. Parametrised per ``(propagator, ndim)`` rather
    than looping inside, so that one broken dimensionality is reported as
    itself instead of taking the propagator's other dimensionalities with it.
    """
    import inspect

    spec = deepwave.spec(name)
    parameters = set(inspect.signature(getattr(deepwave, name)).parameters)
    for channel in spec.source_channels(ndim):
        assert spec.source_amplitudes_kwarg(channel) in parameters
        assert spec.source_locations_kwarg(channel) in parameters
        assert spec.source_mask_kwarg(channel) in parameters
    for channel in spec.receiver_channels(ndim):
        assert spec.receiver_locations_kwarg(channel) in parameters
        assert spec.receiver_mask_kwarg(channel) in parameters
    for channel in spec.bg_receiver_channels(ndim):
        assert spec.bg_receiver_locations_kwarg(channel) in parameters
    # The synthetic channels are the ones with no backend slot, so the
    # counts reconcile only after removing them.
    private = deepwave._engine.propagator_spec(name)
    real_sources = [
        c for c in spec.source_channels(ndim) if c not in spec.synthetic_source_channels
    ]
    real_receivers = [
        c
        for c in spec.receiver_channels(ndim)
        if c not in spec.synthetic_receiver_channels
    ]
    factor = 2 if spec.has_bg_receivers else 1
    assert private.n_source_types(ndim) == factor * len(real_sources), (
        f"{name} {ndim}-D: the public vocabulary names {len(real_sources)} real "
        f"source channel(s) but the propagator has "
        f"{private.n_source_types(ndim)} source type(s)"
    )
    assert private.n_receiver_types(ndim) == factor * len(real_receivers), (
        f"{name} {ndim}-D: the public vocabulary names "
        f"{len(real_receivers)} real receiver channel(s) but the "
        f"propagator has {private.n_receiver_types(ndim)} receiver type(s)"
    )


@pytest.mark.parametrize(("name", "ndim"), NAME_AND_NDIM)
def test_model_names_are_the_arguments_the_propagator_takes(
    name: str, ndim: int
) -> None:
    """``model_names`` names the caller's models, not the internal ones.

    The docstring promises "the propagator's model tensors, in the order it
    takes them", which is the list a generic wrapper assembles a call from.
    Reporting the *prepared* models instead (``acoustic``'s ``K``/``By``/
    ``Bx`` for the ``v``/``rho`` it really takes) is not a wording defect but
    a false contract: the arity and the quantities are both wrong. Checked
    against the real signature, in signature order, so it cannot drift.
    """
    import inspect

    parameters = list(inspect.signature(getattr(deepwave, name)).parameters)
    models = deepwave.spec(name).model_names(ndim)
    assert models, f"{name}: no model names"
    phantom = [model for model in models if model not in parameters]
    assert not phantom, (
        f"{name} {ndim}-D: model_names reports {phantom}, which are not "
        f"parameters of deepwave.{name}"
    )
    positions = [parameters.index(model) for model in models]
    assert positions == sorted(positions), (
        f"{name} {ndim}-D: model_names order {list(models)} does not follow "
        "the signature order"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scalar", ("v",)),
        ("scalar_born", ("v", "dv")),
        ("acoustic", ("v", "rho")),
        ("acoustic_born", ("v", "rho", "dv", "drho")),
        ("viscoacoustic", ("v", "rho", "q")),
        ("elastic", ("lamb", "mu", "buoyancy")),
        ("elastic_born", ("lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy")),
        ("acoustic_vti", ("c11", "c13", "c33", "buoyancy")),
        ("elastic_vti", ("c11", "c13", "c33", "c55", "buoyancy")),
    ],
)
def test_model_names_are_pinned_per_propagator(
    name: str, expected: tuple[str, ...]
) -> None:
    """The 2-D answer for every propagator, spelled out.

    The test above proves the list is *consistent* with the signature; this
    one pins *which* arguments are the models, which no other check can
    infer.
    """
    assert deepwave.spec(name).model_names(2) == expected


def test_model_names_track_ndim_where_the_models_do() -> None:
    """``elastic_vti``'s ``c66`` is a model only in 3-D, and is listed only there."""
    spec = deepwave.spec("elastic_vti")
    assert "c66" not in spec.model_names(2)
    assert spec.model_names(3) == (*spec.model_names(2), "c66")


@pytest.mark.parametrize("name", PROPAGATORS)
def test_estimate_memory_takes_the_model_names_the_spec_publishes(name: str) -> None:
    """``models_requiring_grad`` accepts exactly what ``model_names`` reports."""
    from deepwave import estimate_memory

    models = deepwave.spec(name).model_names(2)
    estimate = estimate_memory(
        name, (32, 32), 16, 1, models_requiring_grad=list(models)
    )
    assert estimate["total_bytes_lower_bound"] > 0
    with pytest.raises(ValueError, match="Unrecognised model name"):
        estimate_memory(name, (32, 32), 16, 1, models_requiring_grad=["not_a_model"])


@pytest.mark.parametrize("name", PROPAGATORS)
def test_spec_rejects_an_unsupported_ndim(name: str) -> None:
    """Asking about a dimensionality the propagator lacks raises."""
    spec = deepwave.spec(name)
    missing = next((n for n in (1, 2, 3) if n not in spec.ndims), None)
    if missing is None:
        pytest.skip(f"{name} supports every dimensionality")
    with pytest.raises(deepwave.DeepwaveValueError, match="does not support"):
        spec.source_channels(missing)


# ---------------------------------------------------------------------------
# wavefield_names / receiver_names
# ---------------------------------------------------------------------------


def _models(name: str, ndim: int) -> list[torch.Tensor]:
    shape = (12,) * ndim
    v = torch.full(shape, 1500.0, dtype=torch.float64)
    rho = torch.full(shape, 2000.0, dtype=torch.float64)
    q = torch.full(shape, 40.0, dtype=torch.float64)
    lamb = torch.full(shape, 2.0e9, dtype=torch.float64)
    mu = torch.full(shape, 1.0e9, dtype=torch.float64)
    buoyancy = torch.full(shape, 1.0 / 2000.0, dtype=torch.float64)
    stiff = torch.full(shape, 9.0e9, dtype=torch.float64)
    weak = torch.full(shape, 2.0e9, dtype=torch.float64)
    return {
        "scalar": [v],
        "scalar_born": [v],
        "acoustic": [v, rho],
        "acoustic_born": [v, rho],
        "acoustic_vti": [stiff, weak, stiff, buoyancy],
        "viscoacoustic": [v, rho, q],
        "elastic": [lamb, mu, buoyancy],
        "elastic_born": [lamb, mu, buoyancy],
        "elastic_vti": [stiff, weak, stiff, mu, buoyancy],
    }[name]


def _call(name: str, ndim: int, **extra: Any) -> deepwave.PropagatorOutput:
    spec = deepwave.spec(name)
    channel = spec.source_channels(ndim)[1 if name == "acoustic_vti" else 0]
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 12, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[6] * ndim]])
    kwargs: dict[str, Any] = {
        spec.source_amplitudes_kwarg(channel): amplitudes,
        spec.source_locations_kwarg(channel): locations,
        spec.receiver_locations_kwarg(spec.receiver_channels(ndim)[0]): locations,
        "pml_freq": 15.0,
        "pml_width": 2,
        "accuracy": 2,
        "max_vel": 2500.0,
    }
    if name in ("viscoacoustic",):
        kwargs["q_freq"] = 15.0
    if name == "scalar_born":
        kwargs["dv"] = torch.zeros((12,) * ndim, dtype=torch.float64)
    if name == "acoustic_born":
        kwargs["dv"] = torch.zeros((12,) * ndim, dtype=torch.float64)
    if name == "elastic_born":
        kwargs["dlamb"] = torch.zeros((12,) * ndim, dtype=torch.float64)
    if name == "elastic_vti" and ndim == 3:
        kwargs["c66"] = torch.full((12,) * ndim, 1.0e9, dtype=torch.float64)
    kwargs.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out: deepwave.PropagatorOutput = getattr(deepwave, name)(
            *_models(name, ndim), 5.0, 0.0005, **kwargs
        )
    return out


@pytest.mark.parametrize("name", PROPAGATORS)
def test_wavefield_names_matches_final_state_keys(name: str) -> None:
    """``wavefield_names`` is exactly what ``initial_state=`` consumes."""
    for ndim in _ndims(name):
        out = _call(name, ndim)
        assert deepwave.wavefield_names(name, ndim) == tuple(out.final_state)


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
@pytest.mark.parametrize("q_n_mechanisms", [1, 3])
def test_wavefield_names_honours_q_n_mechanisms(name: str, q_n_mechanisms: int) -> None:
    """The two viscous propagators' state depends on the mechanism count."""
    extra: dict[str, Any] = {"q_n_mechanisms": q_n_mechanisms}
    if q_n_mechanisms > 1:
        # A multi-mechanism fit needs an explicit band (and refuses the single
        # reference frequency ``_call`` pins), and only the eager backend
        # implements it.
        extra["q_band"] = (5.0, 50.0)
        extra["q_freq"] = None
        extra["backend"] = "eager"
    out = _call(name, 2, **extra)
    assert deepwave.wavefield_names(name, 2, q_n_mechanisms=q_n_mechanisms) == tuple(
        out.final_state
    )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_wavefield_names_rejects_an_unknown_option(name: str) -> None:
    """An option that cannot change the names raises rather than being ignored."""
    with pytest.raises(deepwave.DeepwaveValueError, match="unknown option"):
        deepwave.wavefield_names(name, 2, nonsense=1)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_receiver_names_matches_output_tuple(name: str) -> None:
    """``receiver_names`` is the receiver slice of the real return, in order."""
    for ndim in _ndims(name):
        out = _call(name, ndim)
        names = deepwave.receiver_names(name, ndim)
        assert names == tuple(n for n in out.names if n in set(names))
        assert set(names).isdisjoint(deepwave.wavefield_names(name, ndim))
        assert tuple(out.names) == deepwave.spec(name).output_names(ndim)


@pytest.mark.parametrize("name", ["scalar_born", "acoustic_born", "elastic_born"])
def test_receiver_names_puts_background_first_on_born(name: str) -> None:
    """The Born order is background block, then scattered block."""
    names = deepwave.receiver_names(name, 2)
    half = len(names) // 2
    assert all(n.startswith("bg_receiver_amplitudes") for n in names[:half])
    assert all(n.startswith("sc_receiver_amplitudes") for n in names[half:])


# ---------------------------------------------------------------------------
# The four reflection APIs
# ---------------------------------------------------------------------------


def test_record_boundary_channel_resolves() -> None:
    """``channel=`` equals the explicit kwarg/output pair it replaces."""
    from deepwave.wavefields import record_boundary

    n = 24
    lamb = torch.full((n, n), 2.0e9, dtype=torch.float64)
    mu = torch.full((n, n), 1.0e9, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / 2000.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 20, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[12, 12]]])
    box = [(8, 16), (8, 16)]
    shared = {"pml_freq": 15.0, "pml_width": 3, "accuracy": 2, "max_vel": 2500.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_channel, surface_a = record_boundary(
            deepwave.elastic,
            (lamb, mu, buoyancy),
            5.0,
            0.0005,
            amplitudes,
            locations,
            box,
            channel="p",
            source_channel="pressure_rate",
            **shared,
        )
        explicit, surface_b = record_boundary(
            deepwave.elastic,
            (lamb, mu, buoyancy),
            5.0,
            0.0005,
            amplitudes,
            locations,
            box,
            receiver_kwarg="receiver_locations_p",
            output_name="receiver_amplitudes_p",
            source_channel="pressure_rate",
            **shared,
        )
    assert torch.equal(by_channel, explicit)
    assert torch.equal(surface_a, surface_b)


def test_record_boundary_channel_with_an_override_raises() -> None:
    """``channel=`` and the explicit overrides are alternatives, not both."""
    from deepwave.wavefields import record_boundary

    v = torch.full((16, 16), 1500.0, dtype=torch.float64)
    amplitudes = torch.zeros(1, 1, 8, dtype=torch.float64)
    locations = torch.tensor([[[8, 8]]])
    with pytest.raises(deepwave.DeepwaveValueError, match="not both"):
        record_boundary(
            deepwave.scalar,
            (v,),
            5.0,
            0.0005,
            amplitudes,
            locations,
            [(4, 10), (4, 10)],
            channel="u",
            output_name="receiver_amplitudes",
        )


def test_record_boundary_scalar_default_is_unchanged() -> None:
    """The default still records ``scalar``'s single channel, as before."""
    from deepwave.wavefields import record_boundary

    v = torch.full((16, 16), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 12, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[8, 8]]])
    default, _ = record_boundary(
        deepwave.scalar,
        (v,),
        5.0,
        0.0005,
        amplitudes,
        locations,
        [(4, 10), (4, 10)],
        pml_freq=15.0,
    )
    named, _ = record_boundary(
        deepwave.scalar,
        (v,),
        5.0,
        0.0005,
        amplitudes,
        locations,
        [(4, 10), (4, 10)],
        channel="u",
        pml_freq=15.0,
    )
    assert torch.equal(default, named)


def test_reinject_boundary_channel_on_elastic() -> None:
    """Recorded elastic pressure traces can be replayed as pressure-rate sources.

    ``reinject_boundary`` hard-coded ``source_amplitudes=``/
    ``source_locations=``, which made it ``scalar``-only in practice while
    its docstring claimed any propagator. Its ``channel=`` selects the source.
    """
    from deepwave.wavefields import record_boundary, reinject_boundary

    n = 24
    lamb = torch.full((n, n), 2.0e9, dtype=torch.float64)
    mu = torch.full((n, n), 1.0e9, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / 2000.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 20, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[12, 12]]])
    shared = {"pml_freq": 15.0, "pml_width": 3, "accuracy": 2, "max_vel": 2500.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recording, surface = record_boundary(
            deepwave.elastic,
            (lamb, mu, buoyancy),
            5.0,
            0.0005,
            amplitudes,
            locations,
            [(8, 16), (8, 16)],
            channel="p",
            source_channel="pressure_rate",
            **shared,
        )
        out = reinject_boundary(
            deepwave.elastic,
            (lamb, mu, buoyancy),
            5.0,
            0.0005,
            recording,
            surface,
            channel="pressure_rate",
            receiver_locations_p=locations,
            **shared,
        )
    assert torch.isfinite(out.receiver_amplitudes_p).all()
    assert float(out.receiver_amplitudes_p.abs().max()) > 0.0


def test_running_dft_defaults_to_first_receiver() -> None:
    """An omitted target selects the first receiver output."""
    from deepwave.wavefields import RunningDFT

    assert RunningDFT([5.0]).on is None
    assert RunningDFT([5.0], on="u").on == "u"


def test_running_dft_accepts_a_channel_or_a_field_name() -> None:
    """``on="u"`` and ``on="receiver_amplitudes"`` name the same target."""
    from deepwave.wavefields import RunningDFT, measure

    v = torch.full((16, 16), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 16, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[8, 8]]])
    shared: dict[str, Any] = {
        "source_locations": locations,
        "receiver_locations": locations,
        "pml_freq": 15.0,
        "pml_width": 3,
        "max_vel": 2500.0,
    }
    by_channel, _ = measure(
        deepwave.scalar,
        [v],
        5.0,
        0.0005,
        {"source_amplitudes": amplitudes},
        [RunningDFT([15.0], on="u")],
        **shared,
    )
    by_field, _ = measure(
        deepwave.scalar,
        [v],
        5.0,
        0.0005,
        {"source_amplitudes": amplitudes},
        [RunningDFT([15.0], on="receiver_amplitudes")],
        **shared,
    )
    assert torch.equal(by_channel[0], by_field[0])


def test_running_dft_unknown_target_lists_both_vocabularies() -> None:
    """The error names the channels *and* the output field names."""
    from deepwave.wavefields import RunningDFT, measure

    v = torch.full((12, 12), 1500.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="not something") as excinfo:
        measure(
            deepwave.scalar,
            [v],
            5.0,
            0.0005,
            {"source_amplitudes": torch.zeros(1, 1, 6, dtype=torch.float64)},
            [RunningDFT([15.0], on="nope")],
            source_locations=torch.tensor([[[6, 6]]]),
            receiver_locations=torch.tensor([[[6, 6]]]),
            pml_freq=15.0,
            pml_width=3,
            max_vel=2500.0,
        )
    message = str(excinfo.value)
    assert "available channels" in message
    assert "available output names" in message


def test_segment_drivers_accept_the_propagators_own_call_forms() -> None:
    """A survey in the propagator's spelling is resolved, not refused.

    ``run_segments`` slices the source amplitudes per segment from its own
    mapping, so a ``sources=`` forwarded whole to every segment would give
    each segment the whole record. It is therefore resolved into the flat
    keywords and the amplitudes lifted into the sliced mapping, up front,
    rather than refused: the driver accepts every call form the propagator
    it drives accepts.
    """
    from deepwave.checkpointing import run_segments

    v = torch.full((16, 16), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 16, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[8, 8]]])
    shared: dict[str, Any] = {
        "pml_freq": 15.0,
        "pml_width": 3,
        "max_vel": 1500.0,
    }
    expected = deepwave.scalar(
        v,
        5.0,
        0.0005,
        source_amplitudes=amplitudes,
        source_locations=locations,
        receiver_locations=locations,
        checkpoint_segments=2,
        **shared,
    )
    got = run_segments(
        deepwave.scalar,
        [v],
        5.0,
        0.0005,
        {},
        2,
        checkpoint=True,
        sources={"u": (amplitudes, locations)},
        receivers={"u": locations},
        **shared,
    )
    assert torch.equal(got.receiver_amplitudes, expected.receiver_amplitudes)


def test_segment_drivers_reject_a_channel_given_twice() -> None:
    """One spelling per channel: the driver's mapping and a flat keyword clash."""
    from deepwave.checkpointing import run_segments

    amplitudes = (
        deepwave.wavelets.ricker(15.0, 16, 0.0005, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[8, 8]]])
    with pytest.raises(deepwave.DeepwaveValueError, match="Give it once"):
        run_segments(
            deepwave.elastic,
            [
                torch.full((16, 16), 2.25e9, dtype=torch.float64),
                torch.full((16, 16), 1.0e9, dtype=torch.float64),
                torch.full((16, 16), 1.0 / 2200.0, dtype=torch.float64),
            ],
            5.0,
            0.0005,
            {"source_amplitudes_y": amplitudes},
            2,
            checkpoint=True,
            source_amplitudes_y=amplitudes,
            source_locations_y=locations,
            pml_freq=15.0,
            pml_width=3,
            max_vel=2500.0,
        )


def _total_bytes(estimate: Any) -> int:
    """The total from an estimate, whether it is a mapping or an object.

    ``estimate_memory``'s return shape is provisional and may yet be
    reshaped; this test is about which propagators it accepts, not about
    that.
    """
    if isinstance(estimate, dict):
        return int(estimate["total_bytes_lower_bound"])
    return int(estimate.total_bytes)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_estimate_memory_accepts_callable_and_covers_nine(name: str) -> None:
    """Every propagator can be estimated, by callable, class or name."""
    base = {"model_shape": (64, 64), "nt": 50, "n_shots": 1}
    functional = getattr(deepwave, name)
    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    by_name = _total_bytes(deepwave.estimate_memory(name, **base))
    assert by_name > 0
    assert _total_bytes(deepwave.estimate_memory(functional, **base)) == by_name
    assert _total_bytes(deepwave.estimate_memory(module_class, **base)) == by_name


@pytest.mark.parametrize("name", PROPAGATORS)
def test_estimate_memory_wavefield_counts_match_spec(name: str) -> None:
    """The wavefield count comes from the propagator, and the four are pinned.

    The four numbers the hand-maintained table used to hold must not move;
    the other seven are new coverage.
    """
    from deepwave._common import _spec_n_wavefields

    historical = {
        ("scalar", 2): 6,
        ("scalar", 3): 8,
        ("scalar_born", 2): 12,
        ("scalar_born", 3): 16,
        ("acoustic", 2): 7,
        ("acoustic", 3): 10,
        ("elastic", 2): 13,
        ("elastic", 3): 27,
    }
    private = deepwave._engine.propagator_spec(name)
    for ndim in _ndims(name):
        count = _spec_n_wavefields(private, ndim, 1)
        assert count > 0
        expected = historical.get((name, ndim))
        if expected is not None:
            assert count == expected, (
                f"{name} {ndim}-D wavefield count moved from {expected} to {count}"
            )


def test_estimate_memory_honours_q_n_mechanisms() -> None:
    """A viscous propagator's estimate grows with the mechanism count."""
    base = {"model_shape": (64, 64), "nt": 50, "n_shots": 1}
    one = _total_bytes(deepwave.estimate_memory("viscoacoustic", **base))
    three = _total_bytes(
        deepwave.estimate_memory("viscoacoustic", **base, q_n_mechanisms=3)
    )
    assert three > one


def test_run_segments_extension_contract_names_the_public_spec() -> None:
    """The documented extension point is a public function, not a private one."""
    from deepwave.checkpointing import run_segments

    doc = run_segments.__doc__ or ""
    assert "_get_spec" not in doc
    assert "deepwave.spec" in doc

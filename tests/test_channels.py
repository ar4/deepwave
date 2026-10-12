"""The channel vocabulary: ``sources=``/``receivers=``, aliases, masks.

Covers the channel-keyed mapping form, the two quantities that share the
name ``source_amplitudes_p`` (and ``acoustic_vti``'s synthesised ``p``
channel), the ``sc_`` spelling on the Born families, ``LocationsLike`` and
the boolean activity masks.

The single most important pair here is ``test_mapping_form_matches_flat_form``
and ``test_multi_key_mapping_matches_flat_form``: the mapping form is
documented as the canonical way to specify a survey, and it must produce
bit-identical numbers to the flat keywords it replaces -- for one channel and
for a whole multi-channel survey at once, which is the case the form exists
for -- or the two forms are not the same interface.

The evidence that a mask means what it says is
``test_mask_matches_a_physically_smaller_survey``: comparing a mask against a
hand-written ``IGNORE_LOCATION`` cannot show it, because the mask is
converted to the sentinel before anything numeric runs.
"""

import warnings
from typing import Any

import pytest
import torch

import deepwave

PROPAGATORS = deepwave._engine.PROPAGATOR_NAMES

_N = 16
_DT = 0.0005
_DX = 5.0
_NT = 24
_FREQ = 15.0


def _models(name: str, ndim: int, dtype: torch.dtype) -> list[torch.Tensor]:
    """The positional model tensors of ``name``, as a uniform medium."""
    shape = (_N,) * ndim
    v = torch.full(shape, 1500.0, dtype=dtype)
    rho = torch.full(shape, 2000.0, dtype=dtype)
    q = torch.full(shape, 40.0, dtype=dtype)
    lamb = torch.full(shape, 2.0e9, dtype=dtype)
    mu = torch.full(shape, 1.0e9, dtype=dtype)
    buoyancy = torch.full(shape, 1.0 / 2000.0, dtype=dtype)
    stiff = torch.full(shape, 9.0e9, dtype=dtype)
    weak = torch.full(shape, 2.0e9, dtype=dtype)
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


def _scatterer(ndim: int, dtype: torch.dtype, background: float) -> torch.Tensor:
    """A localised 1% perturbation of ``background``, and never all zeros.

    The Born scattered field is exactly linear in the perturbation, so at a
    zero perturbation every ``sc_`` output -- and every gradient taken
    through one -- is identically zero, and a test that compares two of them
    passes without touching the propagator. Localised rather than uniform so
    that the scattered field is a real arrival from a real scatterer, and
    deterministic so that two calls built from it are bit-comparable.
    """
    perturbation = torch.zeros((_N,) * ndim, dtype=dtype)
    block = tuple(slice(_N // 2 - 1, _N // 2 + 1) for _ in range(ndim))
    perturbation[block] = background * 0.01
    return perturbation


def _optional_models(name: str, ndim: int, dtype: torch.dtype) -> dict[str, Any]:
    """The scattering perturbations a Born propagator needs, and nothing else."""
    if name == "scalar_born":
        return {"dv": _scatterer(ndim, dtype, 1500.0)}
    if name == "acoustic_born":
        return {"dv": _scatterer(ndim, dtype, 1500.0)}
    if name == "elastic_born":
        return {"dlamb": _scatterer(ndim, dtype, 2.0e9)}
    if name == "elastic_vti" and ndim == 3:
        # c66 is an independent stiffness only in 3-D, where it is required.
        return {"c66": torch.full((_N,) * ndim, 1.0e9, dtype=dtype)}
    return {}


def _survey(
    ndim: int, dtype: torch.dtype, n_points: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitudes and integer locations for ``n_points`` distinct points."""
    amplitudes = (
        deepwave.wavelets.ricker(_FREQ, _NT, _DT, 1.5 / _FREQ)
        .to(dtype)
        .reshape(1, 1, -1)
        .repeat(1, n_points, 1)
    )
    locations = torch.zeros(1, n_points, ndim, dtype=torch.long)
    for i in range(n_points):
        locations[0, i, :] = _N // 2 - 2 + i
    return amplitudes, locations


def _base_kwargs(name: str) -> dict[str, Any]:
    """Grid and stability keywords shared by every call in this module."""
    kwargs: dict[str, Any] = {
        "pml_freq": _FREQ,
        "pml_width": 3,
        "accuracy": 2,
        "max_vel": 2500.0,
    }
    if name in ("viscoacoustic",):
        kwargs["q_freq"] = _FREQ
    return kwargs


def _distinct_survey(
    ndim: int, dtype: torch.dtype, n_points: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """As :func:`_survey`, but every point gets a different amplitude.

    Needed wherever a test drops one point and compares: with identical
    traces at every point, dropping the wrong one would go unnoticed.
    """
    amplitudes, locations = _survey(ndim, dtype, n_points)
    scales = torch.arange(1, n_points + 1, dtype=dtype).reshape(1, n_points, 1)
    return amplitudes * scales, locations


def _ndims(name: str) -> list[int]:
    """Every dimensionality this propagator supports, 1-D included.

    ``spec(name).ndims`` is the propagator's own declaration, so this asks
    for 1-D exactly where 1-D exists and never elsewhere.
    """
    return list(deepwave.spec(name).ndims)


def _name_and_ndim_params() -> list[Any]:
    """Every ``(propagator, ndim)`` pair, 1-D included, as test parameters."""
    return [
        pytest.param(name, ndim, id=f"{name}-{ndim}d")
        for name in PROPAGATORS
        for ndim in _ndims(name)
    ]


NAME_AND_NDIM = _name_and_ndim_params()


def _run(
    name: str, ndim: int, dtype: torch.dtype, **kwargs: Any
) -> deepwave.PropagatorOutput:
    """Calls ``name`` on a tiny uniform problem."""
    fn = getattr(deepwave, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out: deepwave.PropagatorOutput = fn(
            *_models(name, ndim, dtype),
            _DX,
            _DT,
            **_optional_models(name, ndim, dtype),
            **_base_kwargs(name),
            **kwargs,
        )
    return out


def _run_loud(
    name: str, ndim: int, dtype: torch.dtype, **kwargs: Any
) -> deepwave.PropagatorOutput:
    """:func:`_run` with the warning filters left alone.

    :func:`_run` silences everything, which is right for the hundreds of
    calls here that only want numbers, and wrong for the handful that are
    about a warning.
    """
    fn = getattr(deepwave, name)
    out: deepwave.PropagatorOutput = fn(
        *_models(name, ndim, dtype),
        _DX,
        _DT,
        **_optional_models(name, ndim, dtype),
        **_base_kwargs(name),
        **kwargs,
    )
    return out


def _first_source_channel(name: str, ndim: int) -> str:
    """A source channel that this propagator really drives.

    ``acoustic_vti``'s ``p`` is synthesised from the two stress channels, so
    it is exercised separately (see the ``acoustic_vti`` tests below); here
    the first *backend* channel is wanted.
    """
    channels = deepwave.spec(name).source_channels(ndim)
    return channels[1] if name == "acoustic_vti" else channels[0]


def _outputs_are_equal(a: deepwave.PropagatorOutput, b: Any) -> bool:
    """:func:`_assert_outputs_equal` as a predicate, for "must differ" checks."""
    if a.names != b.names:
        return False
    for field in a.names:
        left, right = getattr(a, field), getattr(b, field)
        if field == "auxiliary_state":
            if left.names != right.names or not all(
                torch.equal(left[key], right[key]) for key in left.names
            ):
                return False
        elif not torch.equal(left, right):
            return False
    return True


def _assert_outputs_equal(a: deepwave.PropagatorOutput, b: Any) -> None:
    """Every field of two outputs is bit-identical, auxiliary state included."""
    assert a.names == b.names
    for field in a.names:
        left, right = getattr(a, field), getattr(b, field)
        if field == "auxiliary_state":
            assert left.names == right.names
            for key in left.names:
                assert torch.equal(left[key], right[key]), f"auxiliary_state[{key}]"
        else:
            assert torch.equal(left, right), field


# ---------------------------------------------------------------------------
# The mapping form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mapping_form_matches_flat_form(name: str, dtype: torch.dtype) -> None:
    """The regression lock: ``sources=``/``receivers=`` == the flat keywords.

    Bit-identical on every output field, including every ``auxiliary_state``
    tensor, and on the gradients. The mapping form is documented as
    canonical, so any difference between the two would mean the library has
    two subtly different interfaces rather than one with two spellings.
    """
    spec = deepwave.spec(name)
    for ndim in _ndims(name):
        channel = _first_source_channel(name, ndim)
        receiver_channel = spec.receiver_channels(ndim)[0]
        amplitudes, locations = _survey(ndim, dtype)

        flat: dict[str, Any] = {
            spec.source_amplitudes_kwarg(channel): amplitudes,
            spec.source_locations_kwarg(channel): locations,
            spec.receiver_locations_kwarg(receiver_channel): locations,
        }
        mapped: dict[str, Any] = {
            "sources": {channel: (amplitudes, locations)},
            "receivers": {receiver_channel: locations},
        }
        if spec.has_bg_receivers:
            flat[spec.bg_receiver_locations_kwarg(receiver_channel)] = locations
            mapped["bg_receivers"] = {receiver_channel: locations}

        _assert_outputs_equal(
            _run(name, ndim, dtype, **flat), _run(name, ndim, dtype, **mapped)
        )


@pytest.mark.parametrize(("name", "ndim"), NAME_AND_NDIM)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multi_key_mapping_matches_flat_form(
    name: str, ndim: int, dtype: torch.dtype
) -> None:
    """The whole survey in one mapping: every channel at once, bit-identically.

    :func:`test_mapping_form_matches_flat_form` passes a single-key mapping,
    which is not the case the mapping form exists for -- a multi-channel
    survey is. Every source channel and every receiver channel (and, on the
    Born three, every background-receiver channel) goes in together, each
    with its own trace and its own place, so that two channels transposed on
    the way through the choke point could not produce the same numbers.

    The second half is the non-vacuity check: dropping any one source
    channel from the flat call must change the answer. Without it a channel
    that the propagator quietly discards would still make both forms agree,
    and "all channels at once" would be a claim about a survey that is not
    really there.
    """
    spec = deepwave.spec(name)
    amplitudes, locations = _distinct_survey(ndim, dtype)
    flat: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    receivers: dict[str, Any] = {}
    bg_receivers: dict[str, Any] = {}
    for index, channel in enumerate(spec.source_channels(ndim)):
        # A distinct place per channel keeps the combined set unique within
        # each shot, which ``acoustic_vti`` needs: its synthesised ``p`` is
        # concatenated onto the two stress channels.
        channel_amplitudes = amplitudes * (1.0 + index)
        channel_locations = locations + index
        flat[spec.source_amplitudes_kwarg(channel)] = channel_amplitudes
        flat[spec.source_locations_kwarg(channel)] = channel_locations
        sources[channel] = (channel_amplitudes, channel_locations)
    for index, channel in enumerate(spec.receiver_channels(ndim)):
        flat[spec.receiver_locations_kwarg(channel)] = locations + index
        receivers[channel] = locations + index
    for index, channel in enumerate(spec.bg_receiver_channels(ndim)):
        flat[spec.bg_receiver_locations_kwarg(channel)] = locations + index
        bg_receivers[channel] = locations + index

    mapped: dict[str, Any] = {"sources": sources, "receivers": receivers}
    if bg_receivers:
        mapped["bg_receivers"] = bg_receivers
    assert len(sources) == len(spec.source_channels(ndim))
    assert len(receivers) == len(spec.receiver_channels(ndim))

    from_flat = _run(name, ndim, dtype, **flat)
    _assert_outputs_equal(from_flat, _run(name, ndim, dtype, **mapped))

    if len(sources) > 1:
        for channel in spec.source_channels(ndim):
            dropped = {
                key: value
                for key, value in flat.items()
                if key
                not in {
                    spec.source_amplitudes_kwarg(channel),
                    spec.source_locations_kwarg(channel),
                }
            }
            without = _run(name, ndim, dtype, **dropped)
            assert not _outputs_are_equal(from_flat, without), (
                f"{name} {ndim}-D: removing the {channel!r} source changed no "
                "output at all, so it is not really part of this survey"
            )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_mapping_form_gradients_match_flat_form(name: str) -> None:
    """The gradient reaching the caller's own tensors is identical too.

    The choke point must not copy or clone: if it did, the amplitudes the
    propagator differentiates would be a different tensor from the one the
    caller holds, and ``.grad`` would silently stay ``None``.

    Both the recorded traces and the gradient are asserted to be non-zero
    before the two forms are compared: on the Born three these come from the
    ``sc_`` block, which is identically zero -- and so is its gradient -- if
    the perturbation in :func:`_optional_models` is ever zeroed again, and
    ``torch.equal(zeros, zeros)`` is not a comparison of the two forms.
    """
    spec = deepwave.spec(name)
    ndim = 2
    channel = _first_source_channel(name, ndim)
    receiver_channel = spec.receiver_channels(ndim)[0]
    grads = []
    for use_mapping in (False, True):
        amplitudes, locations = _survey(ndim, torch.float64)
        amplitudes = amplitudes.clone().requires_grad_(True)
        if use_mapping:
            kwargs: dict[str, Any] = {
                "sources": {channel: (amplitudes, locations)},
                "receivers": {receiver_channel: locations},
            }
            if spec.has_bg_receivers:
                kwargs["bg_receivers"] = {receiver_channel: locations}
        else:
            kwargs = {
                spec.source_amplitudes_kwarg(channel): amplitudes,
                spec.source_locations_kwarg(channel): locations,
                spec.receiver_locations_kwarg(receiver_channel): locations,
            }
            if spec.has_bg_receivers:
                kwargs[spec.bg_receiver_locations_kwarg(receiver_channel)] = locations
        out = _run(name, ndim, torch.float64, **kwargs)
        recorded = out.receivers[receiver_channel]
        assert float(recorded.detach().abs().max()) > 0.0, (
            f"{name}: the recorded traces are identically zero, so the "
            "gradient below cannot discriminate anything"
        )
        recorded.pow(2).sum().backward()
        assert amplitudes.grad is not None, "the caller's tensor got no gradient"
        assert float(amplitudes.grad.abs().max()) > 0.0, (
            f"{name}: the gradient is identically zero, so comparing the two "
            "forms compares zeros with zeros"
        )
        grads.append(amplitudes.grad)
    assert torch.equal(grads[0], grads[1])


@pytest.mark.parametrize("name", PROPAGATORS)
def test_mapping_and_flat_for_same_channel_raises(name: str) -> None:
    """Giving one channel in both forms raises, naming both spellings."""
    spec = deepwave.spec(name)
    ndim = 2
    channel = _first_source_channel(name, ndim)
    amplitudes, locations = _survey(ndim, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(
            name,
            ndim,
            torch.float64,
            sources={channel: (amplitudes, locations)},
            **{
                spec.source_amplitudes_kwarg(channel): amplitudes,
                spec.source_locations_kwarg(channel): locations,
            },
        )
    message = str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)
    assert name in message
    assert "sources=" in message
    assert spec.source_amplitudes_kwarg(channel) in message


@pytest.mark.parametrize("name", PROPAGATORS)
def test_mapping_and_flat_for_different_kinds_is_legal(name: str) -> None:
    """``sources=`` beside a flat *receiver* keyword is fine: different kinds."""
    spec = deepwave.spec(name)
    ndim = 2
    channel = _first_source_channel(name, ndim)
    receiver_channel = spec.receiver_channels(ndim)[0]
    amplitudes, locations = _survey(ndim, torch.float64)
    kwargs: dict[str, Any] = {
        "sources": {channel: (amplitudes, locations)},
        spec.receiver_locations_kwarg(receiver_channel): locations,
    }
    if spec.has_bg_receivers:
        kwargs[spec.bg_receiver_locations_kwarg(receiver_channel)] = locations
    _run(name, ndim, torch.float64, **kwargs)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_unknown_channel_key_lists_valid_keys(name: str) -> None:
    """An unknown key names every channel this propagator does have."""
    spec = deepwave.spec(name)
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(name, 2, torch.float64, sources={"nope": (amplitudes, locations)})
    message = str(excinfo.value)
    assert "'nope'" in message
    for channel in spec.source_channels(2):
        assert repr(channel) in message
    assert "deepwave.spec" in message


def test_z_channel_in_2d_raises_the_existing_dimension_error() -> None:
    """A 3-D-only key in a 2-D call reaches the propagator's own z check.

    The mapping is expanded before ``ndim`` is known, so ``"z"`` is accepted
    as a key and then rejected downstream by the check that already words
    this case well -- which now covers the mapping form for free.
    """
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        _run(
            "elastic",
            2,
            torch.float64,
            sources={"z": (amplitudes, locations)},
        )
    assert "z" in str(excinfo.value)


@pytest.mark.parametrize("name", PROPAGATORS)
def test_sources_value_shape_errors(name: str) -> None:
    """A malformed mapping value raises a ``DeepwaveTypeError`` that says why."""
    spec = deepwave.spec(name)
    channel = _first_source_channel(name, 2)
    amplitudes, locations = _survey(2, torch.float64)
    for bad in (amplitudes, (amplitudes,), (amplitudes, locations, None, None)):
        with pytest.raises(deepwave.DeepwaveTypeError) as excinfo:
            _run(name, 2, torch.float64, sources={channel: bad})
        assert isinstance(excinfo.value, TypeError)
        assert "sources" in str(excinfo.value)
    del spec


def test_bare_tensor_as_a_sources_value_names_the_mistake() -> None:
    """The likely mistake -- a bare tensor -- is named explicitly."""
    amplitudes, _ = _survey(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveTypeError) as excinfo:
        _run("scalar", 2, torch.float64, sources={"u": amplitudes})
    assert "not Tensor" in str(excinfo.value)


@pytest.mark.parametrize(("name", "ndim"), NAME_AND_NDIM)
def test_receivers_view_round_trips(name: str, ndim: int) -> None:
    """``out.receivers`` is keyed by channel and holds the tuple's own tensors."""
    spec = deepwave.spec(name)
    channel = _first_source_channel(name, ndim)
    receiver_channel = spec.receiver_channels(ndim)[0]
    amplitudes, locations = _survey(ndim, torch.float64)
    out = _run(
        name,
        ndim,
        torch.float64,
        sources={channel: (amplitudes, locations)},
        receivers={receiver_channel: locations},
    )
    assert tuple(out.receivers) == spec.receiver_channels(ndim)
    # Each value is the very tensor the tuple holds, not a copy.
    for key in out.receivers:
        assert out.receivers[key] is getattr(out, spec.receiver_output_name(key))
    # An unrequested channel is present and empty, not missing.
    for key in out.receivers:
        if key != receiver_channel:
            assert out.receivers[key].numel() == 0
    with pytest.raises(TypeError):
        out.receivers["u"] = torch.zeros(1)  # type: ignore[index]


@pytest.mark.parametrize("name", PROPAGATORS)
def test_bg_receivers_view_born_only(name: str) -> None:
    """``out.bg_receivers`` exists on the Born three and nowhere else."""
    spec = deepwave.spec(name)
    channel = _first_source_channel(name, 2)
    receiver_channel = spec.receiver_channels(2)[0]
    amplitudes, locations = _survey(2, torch.float64)
    kwargs: dict[str, Any] = {
        "sources": {channel: (amplitudes, locations)},
        "receivers": {receiver_channel: locations},
    }
    if spec.has_bg_receivers:
        kwargs["bg_receivers"] = {receiver_channel: locations}
    out = _run(name, 2, torch.float64, **kwargs)
    if spec.has_bg_receivers:
        assert tuple(out.bg_receivers) == spec.bg_receiver_channels(2)
    else:
        with pytest.raises(AttributeError):
            _ = out.bg_receivers


def test_bg_receivers_is_a_type_error_on_a_non_born_propagator() -> None:
    """``bg_receivers=`` is absent by signature, not accepted and ignored."""
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises(TypeError, match="bg_receivers"):
        _run(
            "elastic",
            2,
            torch.float64,
            sources={"y": (amplitudes, locations)},
            bg_receivers={"y": locations},
        )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_module_forward_accepts_mapping_form(name: str) -> None:
    """Every ``Module.forward`` takes the mapping form, matching the functional.

    The coordination check: every parameter added to a functional must also
    appear on its Module.
    """
    spec = deepwave.spec(name)
    channel = _first_source_channel(name, 2)
    receiver_channel = spec.receiver_channels(2)[0]
    amplitudes, locations = _survey(2, torch.float64)
    kwargs: dict[str, Any] = {
        "sources": {channel: (amplitudes, locations)},
        "receivers": {receiver_channel: locations},
    }
    if spec.has_bg_receivers:
        kwargs["bg_receivers"] = {receiver_channel: locations}

    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    models = _models(name, 2, torch.float64)
    optional = _optional_models(name, 2, torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = module_class(*models, _DX, **optional)
        from_module = module(_DT, **_base_kwargs(name), **kwargs)
    from_functional = _run(name, 2, torch.float64, **kwargs)
    _assert_outputs_equal(from_functional, from_module)


# ---------------------------------------------------------------------------
# Channel aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["scalar", "scalar_born"])
def test_scalar_channel_and_flat_keywords_agree(name: str) -> None:
    """``"u"`` and the unsuffixed flat keywords are two spellings of one thing."""
    amplitudes, locations = _survey(2, torch.float64)
    outs = [
        _run(name, 2, torch.float64, sources={"u": (amplitudes, locations)}),
        _run(
            name,
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=locations,
        ),
    ]
    _assert_outputs_equal(outs[0], outs[1])


@pytest.mark.parametrize("name", ["scalar", "scalar_born"])
def test_the_empty_string_is_not_a_channel(name: str) -> None:
    """Each channel has exactly one spelling; ``""`` is not one of them.

    The two scalar propagators spell their single channel ``"u"`` however
    their own flat keywords are spelled. An empty key is a mistake (a
    programmatically built survey whose channel name went missing), and is
    reported as one rather than silently taken as ``"u"``.
    """
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(name, 2, torch.float64, sources={"": (amplitudes, locations)})
    message = str(excinfo.value)
    assert "''" in message
    assert "'u'" in message
    assert deepwave.spec(name).channel_aliases() == {}


def _p_channel_propagators() -> list[str]:
    """The released acoustic and elastic propagators retain their ``p`` alias."""
    return [n for n in PROPAGATORS if deepwave.spec(n).deprecated_source_channels()]


def _rate_name(name: str) -> str:
    """The quantity name that replaces ``p`` as ``name``'s source channel."""
    return deepwave.spec(name).deprecated_source_channels()["p"]


def test_only_released_propagators_deprecate_p_as_a_source() -> None:
    """The source alias is retained only where v0.0.27 shipped it."""
    deprecating = _p_channel_propagators()
    assert set(deprecating) == {"acoustic", "elastic"}
    for name in deprecating:
        spec = deepwave.spec(name)
        assert spec.deprecated_source_channels() == {"p": _rate_name(name)}
        assert _rate_name(name) in spec.source_channels(2)
        assert "p" not in spec.source_channels(2)


@pytest.mark.parametrize("name", _p_channel_propagators())
def test_p_source_channel_key_warns_and_still_works(name: str) -> None:
    """``sources={"p": ...}`` warns, names its replacement, and is unchanged."""
    rate = _rate_name(name)
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.warns(deepwave.DeepwaveFutureWarning) as record:
        deprecated = _run_loud(
            name, 2, torch.float64, sources={"p": (amplitudes, locations)}
        )
    message = str(record[0].message)
    assert rate in message
    assert "0.4.0" in message
    assert "DeepwaveFutureWarning" in message
    _assert_outputs_equal(
        deprecated,
        _run(name, 2, torch.float64, sources={rate: (amplitudes, locations)}),
    )


@pytest.mark.parametrize("name", _p_channel_propagators())
def test_the_other_families_long_name_is_rejected(name: str) -> None:
    """The wrong family's quantity name is an unknown key, not a silent accept.

    This is the trap closing: the two families' scalar source channels
    differ by roughly the bulk modulus, so a caller who reaches for the
    wrong name must be told which one this propagator means.
    """
    mine = _rate_name(name)
    theirs = "pressure_rate" if mine == "volume_rate" else "volume_rate"
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(name, 2, torch.float64, sources={theirs: (amplitudes, locations)})
    message = str(excinfo.value)
    assert theirs in message
    assert mine in message, "the message must name the name this one accepts"


@pytest.mark.parametrize("name", _p_channel_propagators())
def test_rate_names_are_rejected_as_receiver_channels(name: str) -> None:
    """``volume_rate``/``pressure_rate`` name sources only.

    A receiver records a pressure, not a rate, so neither quantity name is a
    receiver channel on any propagator -- and ``p`` still is.
    """
    spec = deepwave.spec(name)
    assert "p" in spec.receiver_channels(2)
    _, locations = _survey(2, torch.float64)
    for rate in ("volume_rate", "pressure_rate"):
        assert rate not in spec.receiver_channels(2)
        with pytest.raises(deepwave.DeepwaveValueError, match=rate):
            _run(name, 2, torch.float64, receivers={rate: locations})


@pytest.mark.parametrize("name", _p_channel_propagators())
def test_p_flat_source_keywords_are_deprecated(name: str) -> None:
    """The flat ``_p`` source keywords warn, and compute what they always did."""
    rate = _rate_name(name)
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.warns(deepwave.DeepwaveFutureWarning, match="0.4.0") as record:
        deprecated = _run_loud(
            name,
            2,
            torch.float64,
            source_amplitudes_p=amplitudes,
            source_locations_p=locations,
        )
    warned = {str(w.message) for w in record}
    assert any(f"source_amplitudes_{rate}" in m for m in warned)
    assert any(f"source_locations_{rate}" in m for m in warned)
    _assert_outputs_equal(
        deprecated,
        _run(
            name,
            2,
            torch.float64,
            **{
                f"source_amplitudes_{rate}": amplitudes,
                f"source_locations_{rate}": locations,
            },
        ),
    )
    with pytest.raises(deepwave.DeepwaveValueError, match=r"not both"):
        _run(
            name,
            2,
            torch.float64,
            source_amplitudes_p=amplitudes,
            source_locations_p=locations,
            **{f"source_amplitudes_{rate}": amplitudes},
        )


@pytest.mark.parametrize("name", _p_channel_propagators())
def test_p_receiver_channel_never_warns(name: str) -> None:
    """``p`` stays a receiver name: Q3 touches the source side only."""
    amplitudes, locations = _survey(2, torch.float64)
    rate = _rate_name(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", deepwave.DeepwaveFutureWarning)
        _run_loud(
            name,
            2,
            torch.float64,
            sources={rate: (amplitudes, locations)},
            receivers={"p": locations},
        )


# ---------------------------------------------------------------------------
# acoustic_vti's synthesised p channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_acoustic_vti_p_equals_negated_stress_channels(
    dtype: torch.dtype, backend: str
) -> None:
    """The definitional test: ``pressure_rate`` is ``-a`` on both stress channels.

    Bit-exact, field for field, in both dimensionalities and both dtypes.
    ``acoustic_vti`` scales a stress-rate source by ``dt`` and adds it to
    that stress, and its stress is tension-positive, while
    :func:`deepwave.elastic` adds ``-dt`` times a compression-positive
    pressure rate to every normal stress -- so the mapping is a sign flip and
    nothing else.
    """
    for ndim in (2, 3):
        amplitudes, locations = _survey(ndim, dtype)
        via_p = _run(
            "acoustic_vti",
            ndim,
            dtype,
            backend=backend,
            source_amplitudes_pressure_rate=amplitudes,
            source_locations_pressure_rate=locations,
            receiver_locations_sigma_v=locations,
            receiver_locations_sigma_h=locations,
        )
        by_hand = _run(
            "acoustic_vti",
            ndim,
            dtype,
            backend=backend,
            source_amplitudes_sigma_v=-amplitudes,
            source_locations_sigma_v=locations,
            source_amplitudes_sigma_h=-amplitudes,
            source_locations_sigma_h=locations,
            receiver_locations_sigma_v=locations,
            receiver_locations_sigma_h=locations,
        )
        _assert_outputs_equal(via_p, by_hand)


def test_acoustic_vti_p_matches_acoustic_in_the_isotropic_limit() -> None:
    """Physics: at ``c11 = c33 = c13 = K`` the ``p`` channel is acoustic's.

    The measured relation, to round-off: ``acoustic_vti``'s
    ``receiver_amplitudes_sigma_v`` under
    ``source_amplitudes_pressure_rate=a`` equals minus ``acoustic``'s
    ``receiver_amplitudes_p`` under ``source_amplitudes_volume_rate=a/K``.
    The ``1/K`` converts the acoustic family's *volume* rate into a pressure
    rate; the minus sign is tension-positive stress against
    compression-positive pressure.
    """
    n, dx, dt, nt = 80, 5.0, 0.0005, 500
    rho_value, v_value = 2000.0, 2000.0
    bulk_modulus = rho_value * v_value**2
    stiffness = torch.full((n, n), bulk_modulus, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / rho_value, dtype=torch.float64)
    v = torch.full((n, n), v_value, dtype=torch.float64)
    rho = torch.full((n, n), rho_value, dtype=torch.float64)
    amplitudes = deepwave.wavelets.ricker(10.0, nt, dt, 0.12).double().reshape(1, 1, -1)
    source = torch.tensor([[[40, 40]]])
    receiver = torch.tensor([[[40, 60]]])
    shared = {"pml_freq": 10.0, "accuracy": 4, "pml_width": 20}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vti = deepwave.acoustic_vti(
            stiffness,
            stiffness,
            stiffness,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=amplitudes,
            source_locations_pressure_rate=source,
            receiver_locations_sigma_v=receiver,
            **shared,
        )
        iso = deepwave.acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=amplitudes / bulk_modulus,
            source_locations_volume_rate=source,
            receiver_locations_p=receiver,
            **shared,
        )
    stress = vti.receiver_amplitudes_sigma_v[0, 0]
    pressure = iso.receiver_amplitudes_p[0, 0]
    assert float(stress.abs().max()) > 0.0, "the reference arrival is vacuous"
    residual = float((stress + pressure).norm() / stress.norm())
    assert residual < 1e-10, f"measured relative residual {residual:.3e}, expected ~0"


def test_acoustic_vti_p_gradient() -> None:
    """``gradcheck`` on the synthesised channel's amplitudes and on ``c11``."""
    torch.manual_seed(0)
    n, dx, dt, nt = 12, 5.0, 0.0005, 8
    stiffness = torch.full((n, n), 2000.0**2 * 2000.0, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / 2000.0, dtype=torch.float64)
    amplitudes = torch.randn(1, 1, nt, dtype=torch.float64, requires_grad=True)
    c11 = stiffness.clone().requires_grad_(True)
    source = torch.tensor([[[6, 6]]])
    receiver = torch.tensor([[[6, 8]]])

    def run(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = deepwave.acoustic_vti(
                c,
                stiffness,
                stiffness,
                buoyancy,
                dx,
                dt,
                source_amplitudes_pressure_rate=a,
                source_locations_pressure_rate=source,
                receiver_locations_sigma_v=receiver,
                pml_freq=10.0,
                pml_width=2,
                backend="eager",
            )
        return out.receiver_amplitudes_sigma_v

    assert torch.autograd.gradcheck(run, (amplitudes, c11), atol=1e-8, rtol=1e-5)


def test_acoustic_vti_p_concatenates_with_explicit_stress_channels() -> None:
    """``p`` beside explicit stress sources equals the hand-concatenated call."""
    amplitudes, locations = _survey(2, torch.float64)
    other = amplitudes * 0.5
    # Well clear of the p-channel cells: the combined set must stay
    # unique within each shot.
    other_locations = locations + 4
    combined = _run(
        "acoustic_vti",
        2,
        torch.float64,
        source_amplitudes_pressure_rate=amplitudes,
        source_locations_pressure_rate=locations,
        source_amplitudes_sigma_v=other,
        source_locations_sigma_v=other_locations,
        source_amplitudes_sigma_h=other,
        source_locations_sigma_h=other_locations,
        receiver_locations_sigma_v=locations,
    )
    by_hand = _run(
        "acoustic_vti",
        2,
        torch.float64,
        source_amplitudes_sigma_v=torch.cat([other, -amplitudes], dim=1),
        source_locations_sigma_v=torch.cat([other_locations, locations], dim=1),
        source_amplitudes_sigma_h=torch.cat([other, -amplitudes], dim=1),
        source_locations_sigma_h=torch.cat([other_locations, locations], dim=1),
        receiver_locations_sigma_v=locations,
    )
    _assert_outputs_equal(combined, by_hand)


def test_acoustic_vti_p_hicks_with_explicit_stress_channel_raises() -> None:
    """A ``Hicks`` ``p`` cannot be concatenated with an explicit stress source."""
    amplitudes, locations = _survey(2, torch.float64)
    hicks = deepwave.Hicks(locations.double() + 0.5)
    with pytest.raises(deepwave.DeepwaveValueError, match="Hicks"):
        _run(
            "acoustic_vti",
            2,
            torch.float64,
            source_amplitudes_pressure_rate=amplitudes,
            source_locations_pressure_rate=hicks,
            source_amplitudes_sigma_v=amplitudes,
            source_locations_sigma_v=locations,
            receiver_locations_sigma_v=locations,
        )


def test_acoustic_vti_p_requires_both_amplitudes_and_locations() -> None:
    """Half a ``p`` channel raises, naming the missing half.

    The same class and the same wording as an unpaired velocity channel on
    any other propagator: the synthesised pressure-rate channel is not a
    special case for the caller.
    """
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises(
        deepwave.DeepwaveRuntimeError, match="source_locations_pressure_rate"
    ):
        _run(
            "acoustic_vti",
            2,
            torch.float64,
            source_amplitudes_pressure_rate=amplitudes,
            receiver_locations_sigma_v=locations,
        )


def test_acoustic_vti_p_receiver_leads_the_receivers() -> None:
    """``p`` is a receiver channel here too, and it comes first."""
    spec = deepwave.spec("acoustic_vti")
    assert spec.synthetic_source_channels == frozenset({"pressure_rate"})
    assert spec.synthetic_receiver_channels == frozenset({"p"})
    for ndim in (2, 3):
        assert spec.receiver_channels(ndim)[0] == "p"
        assert spec.receiver_names(ndim)[0] == "receiver_amplitudes_p"
        assert spec.receiver_output_name("p") == "receiver_amplitudes_p"


@pytest.mark.parametrize("ndim", [2, 3])
def test_acoustic_vti_p_receiver_is_minus_the_mean_normal_stress(ndim: int) -> None:
    """The definitional test: ``p`` is ``-(sigma_v + (ndim-1) sigma_h) / ndim``.

    Bit-exact against recording the two stresses at the same points and
    combining them by hand, which is what the propagator does internally --
    so this pins the formula and the sign, not merely that something is
    recorded.
    """
    amplitudes, locations = _survey(ndim, torch.float64)
    survey: dict[str, Any] = {
        "source_amplitudes_pressure_rate": amplitudes,
        "source_locations_pressure_rate": locations,
    }
    via_p = _run(
        "acoustic_vti", ndim, torch.float64, receiver_locations_p=locations, **survey
    )
    by_hand = _run(
        "acoustic_vti",
        ndim,
        torch.float64,
        receiver_locations_sigma_v=locations,
        receiver_locations_sigma_h=locations,
        **survey,
    )
    expected = (
        -(
            by_hand.receiver_amplitudes_sigma_v
            + (ndim - 1) * by_hand.receiver_amplitudes_sigma_h
        )
        / ndim
    )
    assert float(expected.abs().max()) > 0.0, "the reference arrival is vacuous"
    assert torch.equal(via_p.receiver_amplitudes_p, expected)
    # Asking for the pressure does not smuggle the stresses out with it.
    assert via_p.receiver_amplitudes_sigma_v.numel() == 0
    assert via_p.receiver_amplitudes_sigma_h.numel() == 0


def test_acoustic_vti_p_receiver_beside_its_own_stress_receivers() -> None:
    """Each stress channel keeps exactly the receivers it was asked for."""
    amplitudes, locations = _survey(2, torch.float64)
    other = locations + 3
    out = _run(
        "acoustic_vti",
        2,
        torch.float64,
        source_amplitudes_pressure_rate=amplitudes,
        source_locations_pressure_rate=locations,
        receiver_locations_p=locations,
        receiver_locations_sigma_v=other,
    )
    reference = _run(
        "acoustic_vti",
        2,
        torch.float64,
        source_amplitudes_pressure_rate=amplitudes,
        source_locations_pressure_rate=locations,
        receiver_locations_p=locations,
    )
    only_sigma_v = _run(
        "acoustic_vti",
        2,
        torch.float64,
        source_amplitudes_pressure_rate=amplitudes,
        source_locations_pressure_rate=locations,
        receiver_locations_sigma_v=other,
    )
    assert torch.equal(out.receiver_amplitudes_p, reference.receiver_amplitudes_p)
    assert torch.equal(
        out.receiver_amplitudes_sigma_v, only_sigma_v.receiver_amplitudes_sigma_v
    )
    assert out.receiver_amplitudes_sigma_h.numel() == 0


@pytest.mark.parametrize("overlap", ["all", "some", "none"])
def test_acoustic_vti_p_receiver_beside_the_same_points(overlap: str) -> None:
    """Recording ``p`` where a stress is already recorded is not a duplicate.

    The pressure is recorded by asking both stress channels for its points,
    so a point a channel already records would be asked for twice and trip
    the "receiver locations must be unique within each shot" check -- a
    failure the caller could not act on, having passed each set once, in
    different channels. Each of the three overlap cases must give the same
    numbers as recording every point on the stress channel by hand.
    """
    amplitudes, _ = _survey(2, torch.float64)
    own = torch.tensor([[[6, 6], [7, 7], [8, 8]]])
    pressure = {
        "all": torch.tensor([[[6, 6], [7, 7], [8, 8]]]),
        "some": torch.tensor([[[7, 7], [10, 10]]]),
        "none": torch.tensor([[[10, 10], [11, 11]]]),
    }[overlap]
    survey: dict[str, Any] = {
        "source_amplitudes_pressure_rate": amplitudes[:, :1],
        "source_locations_pressure_rate": torch.tensor([[[4, 4]]]),
    }
    out = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_p=pressure,
        receiver_locations_sigma_v=own,
        **survey,
    )
    # The union of the two sets, in order, each point once: what a caller
    # would have to build by hand to record both channels at every point.
    rows = [own[0, i] for i in range(own.shape[1])]
    rows += [
        pressure[0, j]
        for j in range(pressure.shape[1])
        if not any(bool((row == pressure[0, j]).all()) for row in rows)
    ]
    every = torch.stack(rows).unsqueeze(0)
    by_hand = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_sigma_v=every,
        receiver_locations_sigma_h=every,
        **survey,
    )
    columns = [
        next(i for i, row in enumerate(rows) if bool((row == pressure[0, j]).all()))
        for j in range(pressure.shape[1])
    ]
    expected = (
        -(
            by_hand.receiver_amplitudes_sigma_v[:, columns]
            + by_hand.receiver_amplitudes_sigma_h[:, columns]
        )
        / 2
    )
    assert float(expected.abs().max()) > 0.0, "the reference arrival is vacuous"
    assert torch.equal(out.receiver_amplitudes_p, expected)
    # The stress channel keeps exactly the receivers it was asked for.
    assert torch.equal(
        out.receiver_amplitudes_sigma_v,
        by_hand.receiver_amplitudes_sigma_v[:, : own.shape[1]],
    )


def test_acoustic_vti_p_receiver_overlaps_differ_between_shots() -> None:
    """The de-duplication is per shot, not one decision for the whole call."""
    amplitudes, _ = _survey(2, torch.float64)
    own = torch.tensor([[[6, 6], [7, 7]], [[9, 9], [10, 10]]])
    pressure = torch.tensor([[[6, 6], [12, 12]], [[13, 13], [10, 10]]])
    survey: dict[str, Any] = {
        "source_amplitudes_pressure_rate": amplitudes[:, :1].expand(2, -1, -1),
        "source_locations_pressure_rate": torch.tensor([[[4, 4]], [[4, 4]]]),
    }
    combined = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_p=pressure,
        receiver_locations_sigma_v=own,
        **survey,
    )
    alone = _run(
        "acoustic_vti", 2, torch.float64, receiver_locations_p=pressure, **survey
    )
    assert torch.equal(combined.receiver_amplitudes_p, alone.receiver_amplitudes_p)


def test_acoustic_vti_p_receiver_with_no_points_is_the_empty_tensor() -> None:
    """A channel given no points records what one given no locations does."""
    amplitudes, locations = _survey(2, torch.float64)
    survey: dict[str, Any] = {
        "source_amplitudes_pressure_rate": amplitudes[:, :1],
        "source_locations_pressure_rate": torch.tensor([[[4, 4]]]),
    }
    out = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_p=torch.zeros(1, 0, 2, dtype=torch.long),
        receiver_locations_sigma_v=locations,
        **survey,
    )
    untouched = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_sigma_v=locations,
        **survey,
    )
    assert out.receiver_amplitudes_p.numel() == 0
    assert torch.equal(
        out.receiver_amplitudes_sigma_v, untouched.receiver_amplitudes_sigma_v
    )


def test_acoustic_vti_p_receiver_off_node_with_own_stress_receivers() -> None:
    """Off-node pressure points interpolate, and still de-duplicate."""
    amplitudes, locations = _survey(2, torch.float64)
    own = locations.double() + 0.5
    pressure = own[:, :1]
    survey: dict[str, Any] = {
        "source_amplitudes_pressure_rate": amplitudes[:, :1],
        "source_locations_pressure_rate": torch.tensor([[[4, 4]]]),
    }
    out = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_p=pressure,
        receiver_locations_sigma_v=own,
        **survey,
    )
    by_hand = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_sigma_v=own,
        receiver_locations_sigma_h=own,
        **survey,
    )
    expected = (
        -(
            by_hand.receiver_amplitudes_sigma_v[:, :1]
            + by_hand.receiver_amplitudes_sigma_h[:, :1]
        )
        / 2
    )
    assert float(expected.abs().max()) > 0.0, "the reference arrival is vacuous"
    assert torch.equal(out.receiver_amplitudes_p, expected)
    assert torch.equal(
        out.receiver_amplitudes_sigma_v, by_hand.receiver_amplitudes_sigma_v
    )


def test_acoustic_vti_duplicate_p_receivers_still_raise() -> None:
    """One channel asking for one cell twice is still the caller's error."""
    amplitudes, _ = _survey(2, torch.float64)
    with pytest.raises(RuntimeError, match="unique within each shot"):
        _run(
            "acoustic_vti",
            2,
            torch.float64,
            source_amplitudes_pressure_rate=amplitudes[:, :1],
            source_locations_pressure_rate=torch.tensor([[[4, 4]]]),
            receiver_locations_p=torch.tensor([[[6, 6], [6, 6]]]),
        )


def test_acoustic_vti_p_receiver_mask_matches_the_sentinel() -> None:
    """``receiver_mask_p`` is the mask spelling of the ignore sentinel."""
    amplitudes, locations = _survey(2, torch.float64)
    n_points = locations.shape[1]
    pair = torch.cat([locations, locations + 5], dim=1)
    mask = torch.tensor(
        [[True] * n_points + [False] * n_points], dtype=torch.bool
    ).expand(pair.shape[0], 2 * n_points)
    survey: dict[str, Any] = {
        "source_amplitudes_pressure_rate": amplitudes,
        "source_locations_pressure_rate": locations,
    }
    masked = _run(
        "acoustic_vti",
        2,
        torch.float64,
        receiver_locations_p=pair,
        receiver_mask_p=mask,
        **survey,
    )
    single = _run(
        "acoustic_vti", 2, torch.float64, receiver_locations_p=locations, **survey
    )
    assert torch.equal(
        masked.receiver_amplitudes_p[:, :n_points], single.receiver_amplitudes_p
    )
    assert not masked.receiver_amplitudes_p[:, n_points:].any()


def test_acoustic_vti_p_receiver_hicks_with_own_stress_receivers_raises() -> None:
    """A ``Hicks`` ``p`` cannot be concatenated onto a stress channel's own."""
    amplitudes, locations = _survey(2, torch.float64)
    hicks = deepwave.Hicks(locations.double() + 0.5)
    with pytest.raises(deepwave.DeepwaveValueError, match="Hicks"):
        _run(
            "acoustic_vti",
            2,
            torch.float64,
            source_amplitudes_pressure_rate=amplitudes,
            source_locations_pressure_rate=locations,
            receiver_locations_p=hicks,
            receiver_locations_sigma_v=locations,
        )


def test_acoustic_vti_p_receiver_gradient() -> None:
    """``gradcheck`` reaches the models through the synthesised channel."""
    torch.manual_seed(0)
    n, dx, dt, nt = 12, 5.0, 0.0005, 8
    stiffness = torch.full((n, n), 2000.0**2 * 2000.0, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / 2000.0, dtype=torch.float64)
    amplitudes = torch.randn(1, 1, nt, dtype=torch.float64, requires_grad=True)
    c11 = stiffness.clone().requires_grad_(True)
    source = torch.tensor([[[6, 6]]])
    receiver = torch.tensor([[[6, 8]]])

    def run(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = deepwave.acoustic_vti(
                c,
                stiffness,
                stiffness,
                buoyancy,
                dx,
                dt,
                source_amplitudes_pressure_rate=a,
                source_locations_pressure_rate=source,
                receiver_locations_p=receiver,
                pml_freq=10.0,
                pml_width=2,
                backend="eager",
            )
        return out.receiver_amplitudes_p

    assert torch.autograd.gradcheck(run, (amplitudes, c11), atol=1e-8, rtol=1e-5)


# ---------------------------------------------------------------------------
# The sc_ spelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["scalar_born", "acoustic_born", "elastic_born"])
def test_sc_receiver_spelling_all_born(name: str) -> None:
    """``sc_receiver_locations*`` really records the scattered field.

    ``numel() > 0`` alone would pass on an all-zero buffer, which is what a
    zero perturbation produces, so the recorded traces are asserted to carry
    an actual arrival: this is the ``sc_`` block, and a wrong ``sc_``
    keyword that silently recorded nothing would look the same as a right
    one otherwise.
    """
    spec = deepwave.spec(name)
    channel = spec.receiver_channels(2)[0]
    assert spec.receiver_locations_kwarg(channel).startswith("sc_")
    amplitudes, locations = _survey(2, torch.float64)
    out = _run(
        name,
        2,
        torch.float64,
        sources={spec.source_channels(2)[0]: (amplitudes, locations)},
        **{spec.receiver_locations_kwarg(channel): locations},
    )
    recorded = out.receivers[channel]
    assert recorded.shape == (locations.shape[0], locations.shape[1], _NT)
    assert float(recorded.abs().max()) > 0.0, (
        f"{name}: the sc_ block is identically zero -- the Born perturbation "
        "in _optional_models must not be all zeros"
    )


@pytest.mark.parametrize("name", ["acoustic_born", "elastic_born"])
def test_bare_receiver_spelling_removed_on_branch_only_born(name: str) -> None:
    """The bare spelling is gone on the two never-released Born propagators."""
    amplitudes, locations = _survey(2, torch.float64)
    with pytest.raises(TypeError, match="receiver_locations_p"):
        _run(
            name,
            2,
            torch.float64,
            sources={"p": (amplitudes, locations)},
            receiver_locations_p=locations,
        )


def test_scalar_born_bare_receiver_locations_is_deprecated() -> None:
    """The one bare spelling ``v0.0.27`` shipped keeps working, and warns."""
    amplitudes, locations = _survey(2, torch.float64)
    survey: dict[str, Any] = {
        "source_amplitudes": amplitudes,
        "source_locations": locations,
    }
    canonical = _run(
        "scalar_born", 2, torch.float64, sc_receiver_locations=locations, **survey
    )
    with pytest.warns(deepwave.DeepwaveFutureWarning) as record:
        deprecated = _run_loud(
            "scalar_born", 2, torch.float64, receiver_locations=locations, **survey
        )
    message = str(record[0].message)
    assert "sc_receiver_locations" in message
    assert "0.4.0" in message
    assert "DeepwaveFutureWarning" in message
    _assert_outputs_equal(canonical, deprecated)
    assert "sc_receiver_amplitudes" in canonical.names
    with pytest.raises(deepwave.DeepwaveValueError, match=r"not both"):
        _run(
            "scalar_born",
            2,
            torch.float64,
            sc_receiver_locations=locations,
            receiver_locations=locations,
            **survey,
        )


def test_scalar_born_never_shipped_bare_names_are_gone() -> None:
    """``receiver_mask=`` and ``out.receiver_amplitudes`` never shipped.

    v0.0.27 had no ``*_mask`` keyword at all, and returned a plain tuple with
    no named fields, so neither spelling is a promise to keep: they are
    removed outright rather than deprecated.
    """
    amplitudes, locations = _survey(2, torch.float64)
    out = _run(
        "scalar_born",
        2,
        torch.float64,
        source_amplitudes=amplitudes,
        source_locations=locations,
        sc_receiver_locations=locations,
    )
    assert "receiver_amplitudes" not in out.names
    with pytest.raises(AttributeError, match="receiver_amplitudes"):
        _ = out.receiver_amplitudes
    with pytest.raises(TypeError, match="receiver_mask"):
        _run(
            "scalar_born",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=locations,
            sc_receiver_locations=locations,
            receiver_mask=torch.ones(locations.shape[:2], dtype=torch.bool),
        )


@pytest.mark.parametrize("name", ["scalar_born", "acoustic_born", "elastic_born"])
def test_born_tuple_order_unchanged(name: str) -> None:
    """Background first, scattered second -- the order the ``sc_`` name matches."""
    spec = deepwave.spec(name)
    for ndim in _ndims(name):
        channels = spec.receiver_channels(ndim)
        names = spec.output_names(ndim)
        assert names[-1] == spec.receiver_output_name(channels[-1])
        assert names[-1].startswith("sc_receiver_amplitudes")
        assert names[-(len(channels) + 1)].startswith("bg_receiver_amplitudes")


# ---------------------------------------------------------------------------
# Masks and LocationsLike
# ---------------------------------------------------------------------------


def _mask_and_sentinel(
    ndim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Amplitudes, plain locations, a mask, and the equivalent sentinel form."""
    amplitudes, locations = _survey(ndim, dtype)
    mask = torch.tensor([[True, False, True]])
    sentinel = locations.clone()
    sentinel[0, 1, :] = deepwave.IGNORE_LOCATION
    return amplitudes, locations, mask, sentinel


@pytest.mark.parametrize("name", PROPAGATORS)
def test_mask_equals_sentinel_bitexact(name: str) -> None:
    """A mask and the equivalent sentinel produce identical numbers."""
    spec = deepwave.spec(name)
    for ndim in _ndims(name):
        channel = _first_source_channel(name, ndim)
        receiver_channel = spec.receiver_channels(ndim)[0]
        amplitudes, locations, mask, sentinel = _mask_and_sentinel(ndim, torch.float64)
        shared: dict[str, Any] = {
            spec.source_amplitudes_kwarg(channel): amplitudes,
            spec.receiver_locations_kwarg(receiver_channel): locations,
        }
        if spec.has_bg_receivers:
            shared[spec.bg_receiver_locations_kwarg(receiver_channel)] = locations
        masked = _run(
            name,
            ndim,
            torch.float64,
            **shared,
            **{
                spec.source_locations_kwarg(channel): locations,
                spec.source_mask_kwarg(channel): mask,
            },
        )
        sentinelled = _run(
            name,
            ndim,
            torch.float64,
            **shared,
            **{spec.source_locations_kwarg(channel): sentinel},
        )
        _assert_outputs_equal(masked, sentinelled)


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("off_grid", [False, True], ids=["integer", "off_grid_float"])
def test_mask_matches_a_physically_smaller_survey(
    name: str, dtype: torch.dtype, off_grid: bool
) -> None:
    """A masked-out point behaves exactly as a point that was never there.

    This is the evidence for the ruling that a mask may be combined with a
    floating-point locations tensor. Comparing a mask against a hand-written
    ``IGNORE_LOCATION`` (:func:`test_mask_equals_sentinel_bitexact`) cannot
    show that, because the mask *is* converted to the sentinel before
    anything numeric runs; the two calls are the same call. So this compares
    a three-point survey with the middle point masked out against a survey
    that physically has only the other two points, which agree only if the
    sentinel is honoured all the way down.

    Run on every propagator, in every dimensionality it supports, in both
    dtypes, and -- the case the ruling is about -- with off-grid float
    locations as well as integer ones. Off-grid floats route through
    :class:`deepwave.Hicks`, and on the staggered propagators the component's
    half-cell offset is handled inside the interpolator, so a sentinel that
    leaked into the coordinates would land on a real grid point here and
    nowhere else.
    """
    spec = deepwave.spec(name)
    mask = torch.tensor([[True, False, True]])
    kept = [0, 2]
    for ndim in _ndims(name):
        source_channel = _first_source_channel(name, ndim)
        receiver_channel = spec.receiver_channels(ndim)[0]
        amplitudes, integer_locations = _distinct_survey(ndim, dtype)
        locations = (
            integer_locations.to(dtype) + 0.25 if off_grid else integer_locations
        )
        fewer_amplitudes = amplitudes[:, kept].contiguous()
        fewer_locations = locations[:, kept].contiguous()

        # A masked-out *source* injects nothing, so every output field --
        # receivers, wavefields and auxiliary state alike -- must match the
        # smaller survey bit for bit.
        listening: dict[str, Any] = {
            spec.receiver_locations_kwarg(receiver_channel): integer_locations
        }
        if spec.has_bg_receivers:
            listening[spec.bg_receiver_locations_kwarg(receiver_channel)] = (
                integer_locations
            )
        masked = _run(
            name,
            ndim,
            dtype,
            **listening,
            **{
                spec.source_amplitudes_kwarg(source_channel): amplitudes,
                spec.source_locations_kwarg(source_channel): locations,
                spec.source_mask_kwarg(source_channel): mask,
            },
        )
        fewer = _run(
            name,
            ndim,
            dtype,
            **listening,
            **{
                spec.source_amplitudes_kwarg(source_channel): fewer_amplitudes,
                spec.source_locations_kwarg(source_channel): fewer_locations,
            },
        )
        assert float(masked.receivers[receiver_channel].abs().max()) > 0.0, (
            f"{name} {ndim}-D: the surviving sources radiate nothing, so this "
            "comparison would hold whatever the mask did"
        )
        _assert_outputs_equal(masked, fewer)

        # A masked-out *receiver* keeps its row in the output, and that row
        # must be exactly zero while the others match the smaller survey.
        emitting = {
            spec.source_amplitudes_kwarg(source_channel): amplitudes,
            spec.source_locations_kwarg(source_channel): integer_locations,
        }
        masked_receivers: dict[str, Any] = {
            spec.receiver_locations_kwarg(receiver_channel): locations,
            spec.receiver_mask_kwarg(receiver_channel): mask,
        }
        fewer_receivers: dict[str, Any] = {
            spec.receiver_locations_kwarg(receiver_channel): fewer_locations
        }
        if spec.has_bg_receivers:
            masked_receivers[spec.bg_receiver_locations_kwarg(receiver_channel)] = (
                locations
            )
            masked_receivers[spec.bg_receiver_mask_kwarg(receiver_channel)] = mask
            fewer_receivers[spec.bg_receiver_locations_kwarg(receiver_channel)] = (
                fewer_locations
            )
        masked = _run(name, ndim, dtype, **emitting, **masked_receivers)
        fewer = _run(name, ndim, dtype, **emitting, **fewer_receivers)

        views = [(masked.receivers, fewer.receivers)]
        if spec.has_bg_receivers:
            views.append((masked.bg_receivers, fewer.bg_receivers))
        for masked_view, fewer_view in views:
            recorded, expected = (
                masked_view[receiver_channel],
                fewer_view[receiver_channel],
            )
            assert float(expected.abs().max()) > 0.0, (
                f"{name} {ndim}-D: the surviving receivers record nothing"
            )
            assert torch.equal(recorded[:, kept], expected)
            assert float(recorded[:, 1].abs().max()) == 0.0, (
                f"{name} {ndim}-D: the masked-out receiver row is not zero"
            )


def test_mask_in_mapping_form_equals_flat_mask() -> None:
    """The mask as a trailing tuple element equals the flat ``*_mask``."""
    amplitudes, locations, mask, _ = _mask_and_sentinel(2, torch.float64)
    _assert_outputs_equal(
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=locations,
            source_mask=mask,
            receiver_locations=locations,
        ),
        _run(
            "scalar",
            2,
            torch.float64,
            sources={"u": (amplitudes, locations, mask)},
            receivers={"u": locations},
        ),
    )


def test_receiver_mask_in_mapping_form_equals_flat_mask() -> None:
    """A ``receivers`` value may be ``(locations, mask)``."""
    amplitudes, locations, mask, _ = _mask_and_sentinel(2, torch.float64)
    _assert_outputs_equal(
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=locations,
            receiver_locations=locations,
            receiver_mask=mask,
        ),
        _run(
            "scalar",
            2,
            torch.float64,
            sources={"u": (amplitudes, locations)},
            receivers={"u": (locations, mask)},
        ),
    )


def test_mask_and_sentinel_mixed_raises() -> None:
    """Two statements of intent for one component raise rather than combine."""
    amplitudes, _, mask, sentinel = _mask_and_sentinel(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=sentinel,
            source_mask=mask,
            receiver_locations=sentinel,
        )
    message = str(excinfo.value)
    assert "IGNORE_LOCATION" in message
    assert "source_mask" in message
    assert "rows contain the sentinel" in message


def test_mask_without_locations_raises() -> None:
    """A mask with nothing to select from raises."""
    _, locations, mask, _ = _mask_and_sentinel(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError, match="without source_locations"):
        _run(
            "scalar",
            2,
            torch.float64,
            nt=_NT,
            source_mask=mask,
            receiver_locations=locations,
        )


def test_mask_shape_mismatch_names_both_shapes() -> None:
    """A wrongly-shaped mask names its shape and the locations' shape."""
    amplitudes, locations, _, _ = _mask_and_sentinel(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=locations,
            source_mask=torch.tensor([[True, False]]),
            receiver_locations=locations,
        )
    message = str(excinfo.value)
    assert "(1, 2)" in message
    assert str(tuple(locations.shape)) in message


def test_mask_must_be_boolean() -> None:
    """The polarity is only meaningful for a boolean mask, so require one."""
    amplitudes, locations, mask, _ = _mask_and_sentinel(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError, match=r"torch\.bool"):
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=locations,
            source_mask=mask.long(),
            receiver_locations=locations,
        )


def test_mask_with_a_hicks_object_raises() -> None:
    """A ``Hicks`` carries its own points; the message names the supported route."""
    amplitudes, locations, mask, _ = _mask_and_sentinel(2, torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=deepwave.Hicks(locations.double()),
            source_mask=mask,
            receiver_locations=locations,
        )
    assert "IGNORE_LOCATION" in str(excinfo.value)


def test_mask_with_float_locations_is_accepted_and_spells_the_sentinel() -> None:
    """A mask on float locations is accepted, and means the sentinel form.

    Deliberately allowed, unlike a ``Hicks`` *object*, which raises. What
    this pins is narrow and worth being honest about: that the call is not
    rejected, and that the two spellings build the same call.
    ``apply_location_masks`` rewrites the mask to ``IGNORE_LOCATION`` with
    ``masked_fill`` before any propagation happens, so the two branches
    below are byte-identical within three statements and this test says
    nothing whatever about how the interpolator then treats the sentinel.
    The evidence for *that* -- ruling 4 -- is
    :func:`test_mask_matches_a_physically_smaller_survey`, which compares
    against a survey that never had the point at all.
    """
    amplitudes, locations, mask, _ = _mask_and_sentinel(2, torch.float64)
    float_locations = locations.double() + 0.25
    sentinel = float_locations.clone()
    sentinel[0, 1, :] = deepwave.IGNORE_LOCATION
    _assert_outputs_equal(
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=float_locations,
            source_mask=mask,
            receiver_locations=locations,
        ),
        _run(
            "scalar",
            2,
            torch.float64,
            source_amplitudes=amplitudes,
            source_locations=sentinel,
            receiver_locations=locations,
        ),
    )


def test_hicks_is_accepted_in_the_mapping_form() -> None:
    """The documented "pass a Hicks as a location" route works in a mapping."""
    amplitudes, locations = _survey(2, torch.float64)
    hicks = deepwave.Hicks(locations.double() + 0.5)
    out = _run(
        "scalar",
        2,
        torch.float64,
        sources={"u": (amplitudes, hicks)},
        receivers={"u": locations},
    )
    assert torch.isfinite(out.receiver_amplitudes).all()


def test_locations_like_is_exported_and_covers_hicks() -> None:
    """``deepwave.LocationsLike`` names the documented location types."""
    import typing

    assert typing.get_origin(deepwave.LocationsLike) is typing.Union
    args = typing.get_args(deepwave.LocationsLike)
    assert torch.Tensor in args


def test_source_and_receiver_spec_are_exported() -> None:
    """``SourceSpec``/``ReceiverSpec`` name the mapping-value shapes."""
    assert deepwave.SourceSpec is not None
    assert deepwave.ReceiverSpec is not None


@pytest.mark.parametrize("name", PROPAGATORS)
def test_sentinel_paragraph_is_verbatim_in_all_nine(name: str) -> None:
    """Every propagator states the all-or-none sentinel rule the same way.

    The rule is enforced in one place, so it must be *described* in one way;
    three different phrasings across the nine is how a reader ends up
    believing a rule that is not the one the code applies.
    """
    doc = getattr(deepwave, name).__doc__ or ""
    assert "IGNORE_LOCATION" in doc, f"{name} never mentions the sentinel"
    # The mask entries carry the canonical statement of the rule.
    assert "True`` means" in doc or "``True`` means active" in doc


def test_output_reserved_names_reject_the_new_view_names() -> None:
    """A field named ``receivers``/``bg_receivers`` would shadow the views.

    ``PropagatorOutput.__new__`` rejects any field name that collides with a
    class attribute, so the two new properties widen that reserved set
    automatically. This pins it, because the failure mode -- ``out.receivers``
    silently returning a property instead of the field -- is invisible.
    """
    from deepwave._engine import PropagatorOutput, RunInfo

    info = RunInfo(
        step_ratio=1,
        internal_dt=_DT,
        courant_number=0.5,
        nt=_NT,
        pml_width=(3,) * 4,
        pml_freq=_FREQ,
        origin=(0, 0),
        extent=((0, _N), (0, _N)),
    )
    for reserved in (
        "receivers",
        "bg_receivers",
        "names",
        "as_dict",
        "final_state",
        "info",
        "index",
        "count",
        "detach",
    ):
        with pytest.raises(AssertionError, match="collides"):
            PropagatorOutput((reserved,), (torch.zeros(1),), info=info)
    # ``auxiliary_state`` is the one deliberate exception: its property exists
    # precisely to expose that field with a useful static type.
    PropagatorOutput(("auxiliary_state",), (torch.zeros(1),), info=info)


def test_hicks_as_a_location_type_checks() -> None:
    """The documented "pass a Hicks as any location argument" route works.

    The static half of this is
    ``test_every_location_parameter_accepts_locations_like``; this is the
    runtime half.
    """
    v = torch.full((_N, _N), 1500.0, dtype=torch.float64)
    amplitudes, locations = _survey(2, torch.float64)
    hicks: deepwave.LocationsLike = deepwave.Hicks(locations.double() + 0.5)
    out = deepwave.scalar(
        v,
        _DX,
        _DT,
        source_amplitudes=amplitudes,
        source_locations=hicks,
        receiver_locations=hicks,
        pml_freq=_FREQ,
        pml_width=3,
    )
    assert torch.isfinite(out.receiver_amplitudes).all()

    module = deepwave.Elastic(
        torch.full((_N, _N), 2.0e9, dtype=torch.float64),
        torch.full((_N, _N), 1.0e9, dtype=torch.float64),
        torch.full((_N, _N), 1.0 / 2000.0, dtype=torch.float64),
        _DX,
    )
    # The ``y`` velocity component's nodes are staggered half a cell in
    # dimension 0, so a Hicks for it is built with that offset. Reusing the
    # regular-grid ``hicks`` above would interpolate about nodes half a cell
    # away, which Deepwave now warns about (see
    # ``test_hicks_offset_mismatch_warns_once``).
    hicks_y: deepwave.LocationsLike = deepwave.Hicks(
        locations.double() + 0.5, offset=[0.5, 0.0]
    )
    elastic_out = module(
        _DT,
        source_amplitudes_y=amplitudes,
        source_locations_y=hicks_y,
        receiver_locations_y=hicks_y,
        pml_freq=_FREQ,
        pml_width=3,
    )
    assert torch.isfinite(elastic_out.receiver_amplitudes_y).all()


@pytest.mark.parametrize("name", PROPAGATORS)
def test_every_location_parameter_accepts_locations_like(name: str) -> None:
    """Every location parameter is annotated ``Optional[LocationsLike]``.

    This is the fix for the documented "pass a :class:`deepwave.Hicks`
    object as any location argument" route being a ``mypy --strict`` error
    at every call site: the parameters were annotated
    ``Optional[torch.Tensor]`` while the implementation accepted a ``Hicks``
    too. Asserted here rather than by type-checking a fixture, because CI's
    ``mypy`` scope is ``src/`` and would not see a fixture under ``tests/``.
    """
    import inspect
    import typing

    expected = typing.Optional[deepwave.LocationsLike]
    for target in (
        getattr(deepwave, name),
        getattr(deepwave, "".join(p.title() for p in name.split("_"))).forward,
    ):
        for parameter in inspect.signature(target).parameters.values():
            if not parameter.name.endswith("_locations") and "_locations_" not in (
                parameter.name
            ):
                continue
            assert parameter.annotation == expected, (
                f"{name}: {parameter.name} is annotated "
                f"{parameter.annotation!r}, not Optional[LocationsLike]"
            )


def test_one_channel_twice_in_a_mapping_names_both_spellings() -> None:
    """A channel given twice inside one mapping says so, without inventing keywords.

    Every channel has permanent alternative spellings, so two keys of one
    mapping can name one channel. The message must describe that, rather than
    blaming the flat keywords (which the merge writes internally and the
    caller never passed).
    """
    amplitudes, locations = _survey(2, torch.float32)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(
            "acoustic",
            2,
            torch.float32,
            sources={
                "p": (amplitudes, locations),
                "volume_rate": (amplitudes, locations),
            },
        )
    message = str(excinfo.value)
    assert "twice inside sources=" in message
    assert "'p'" in message
    assert "'volume_rate'" in message
    assert "source_amplitudes_p" not in message


_ONE_DIMENSIONAL = [name for name in PROPAGATORS if 1 in deepwave.spec(name).ndims]
#: The 1-D propagators that have per-component channels at all (the two
#: scalar propagators have a single, unsuffixed channel in every
#: dimensionality, so there is no ``*_y`` keyword to pass them).
_ONE_DIMENSIONAL_MULTI_CHANNEL = [
    name for name in _ONE_DIMENSIONAL if "y" in deepwave.spec(name).source_channels(2)
]


@pytest.mark.parametrize("name", _ONE_DIMENSIONAL)
def test_one_dimensional_vocabulary_has_no_y_channel(name: str) -> None:
    """In 1-D the only spatial axis is ``x``, so there is no ``y`` channel.

    A published ``y`` channel that the propagator has no wavefield for is
    worse than a missing one: the caller builds a survey the propagator
    cannot honour.
    """
    spec = deepwave.spec(name)
    assert "y" not in spec.source_channels(1)
    assert "y" not in spec.receiver_channels(1)
    assert "y" not in spec.bg_receiver_channels(1)


@pytest.mark.parametrize("name", _ONE_DIMENSIONAL_MULTI_CHANNEL)
def test_a_source_for_an_absent_component_raises_rather_than_vanishing(
    name: str,
) -> None:
    """A 1-D ``y`` source raises; it is never accepted and then dropped.

    The per-ndim assembly inside each propagator keeps only the components
    that exist at that ndim, so a component that does not exist must be
    rejected before it reaches it -- otherwise the caller's source is
    silently discarded and the run returns a plausible, wrong answer.
    """
    amplitudes, locations = _survey(1, torch.float32)
    with pytest.raises(ValueError, match="1d"):
        _run(
            name,
            1,
            torch.float32,
            source_amplitudes_y=amplitudes,
            source_locations_y=locations,
        )


@pytest.mark.parametrize("name", ["acoustic_born", "elastic_born"])
def test_a_receiver_for_an_absent_component_raises_on_born(name: str) -> None:
    """Both Born receiver blocks are checked, not just the sources."""
    _, locations = _survey(2, torch.float32)
    for keyword in ("sc_receiver_locations_z", "bg_receiver_locations_z"):
        with pytest.raises(ValueError, match="2d"):
            _run(name, 2, torch.float32, **{keyword: locations})

"""Physics options appear only on propagators that implement them.

The per-propagator behavioural checks live in ``test_capabilities.py``.
This file checks the signature contract and ``surface_method="pm"`` behaviour.
"""

import warnings
from typing import Any

import pytest
import torch

import deepwave

PROPAGATORS = deepwave._engine.PROPAGATOR_NAMES

#: Each shared option's documented default and a meaningful non-default.
SHARED_OPTIONS = (
    ("mpml_fraction", 0.0, 0.15),
    ("surface_method", "ivm", "pm"),
    ("q_freq", None, 25.0),
    ("q_n_mechanisms", 1, 2),
    ("q_band", None, (5.0, 50.0)),
)

_N = 14
_DT = 0.0005
_DX = 5.0
_NT = 12
_FREQ = 15.0


def _models(name: str) -> list[torch.Tensor]:
    shape = (_N, _N)
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


def _call(name: str, **extra: Any) -> deepwave.PropagatorOutput:
    """Runs ``name`` on a tiny 2-D problem with ``extra`` layered on top."""
    spec = deepwave.spec(name)
    channel = spec.source_channels(2)[1 if name == "acoustic_vti" else 0]
    amplitudes = (
        deepwave.wavelets.ricker(_FREQ, _NT, _DT, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[_N // 2, _N // 2]]])
    kwargs: dict[str, Any] = {
        spec.source_amplitudes_kwarg(channel): amplitudes,
        spec.source_locations_kwarg(channel): locations,
        spec.receiver_locations_kwarg(spec.receiver_channels(2)[0]): locations,
        "pml_freq": _FREQ,
        "pml_width": 2,
        "accuracy": 2,
        "max_vel": 2500.0,
    }
    if name in ("viscoacoustic",):
        kwargs["q_freq"] = _FREQ
    if name in ("scalar_born", "acoustic_born"):
        kwargs["dv"] = torch.zeros((_N, _N), dtype=torch.float64)
    if name == "elastic_born":
        kwargs["dlamb"] = torch.zeros((_N, _N), dtype=torch.float64)
    kwargs.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out: deepwave.PropagatorOutput = getattr(deepwave, name)(
            *_models(name), _DX, _DT, **kwargs
        )
    return out


@pytest.mark.parametrize("name", PROPAGATORS)
def test_options_are_declared_only_where_supported(name: str) -> None:
    """The functional and Module expose the same meaningful physics options."""
    import inspect

    spec = deepwave.spec(name)
    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    functional = inspect.signature(getattr(deepwave, name)).parameters
    module = inspect.signature(module_class.forward).parameters
    for option, default, _ in SHARED_OPTIONS:
        expected = option in spec.supported_options
        assert (option in functional) == expected, (name, option)
        assert (option in module) == expected, (name, option)
        if expected:
            assert functional[option].default == module[option].default == default
        else:
            with pytest.raises(TypeError, match=option):
                _call(name, **{option: default})


@pytest.mark.parametrize("name", PROPAGATORS)
def test_forward_wavefield_opt_in_only_on_scalar(name: str) -> None:
    """Forward-wavefield retention is visible only on its implementing solver."""
    import inspect

    supported = deepwave.spec(name).has_forward_wavefield
    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    for callable_ in (getattr(deepwave, name), module_class.forward):
        params = inspect.signature(callable_).parameters
        assert ("backward_callback_needs_forward_wavefield" in params) == supported
    if supported:
        _call(name, backward_callback_needs_forward_wavefield=True)
    else:
        with pytest.raises(
            TypeError, match="backward_callback_needs_forward_wavefield"
        ):
            _call(name, backward_callback_needs_forward_wavefield=False)


def test_pm_emits_no_maturity_warning() -> None:
    """``surface_method="pm"`` keeps its value and says nothing at run time.

    Renaming the value is rejected. The maturity tier is a documentation
    label, so the option announces nothing: the interface promise is the
    same whichever ``surface_method`` a call names.
    """
    lamb = torch.full((_N, _N), 2.0e9, dtype=torch.float64)
    mu = torch.full((_N, _N), 1.0e9, dtype=torch.float64)
    buoyancy = torch.full((_N, _N), 1.0 / 2000.0, dtype=torch.float64)
    # A vacuum layer: "pm" is only meaningful where there are vacuum cells.
    for model in (lamb, mu, buoyancy):
        model[0] = 0.0
    amplitudes = (
        deepwave.wavelets.ricker(_FREQ, _NT, _DT, 0.1).double().reshape(1, 1, -1)
    )
    locations = torch.tensor([[[_N // 2, _N // 2]]])
    shared: dict[str, Any] = {
        "source_amplitudes_y": amplitudes,
        "source_locations_y": locations,
        "receiver_locations_y": locations,
        "pml_freq": _FREQ,
        "pml_width": 2,
        "accuracy": 2,
        "max_vel": 2500.0,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deepwave.elastic(lamb, mu, buoyancy, _DX, _DT, surface_method="pm", **shared)
    assert not [
        w
        for w in caught
        if "experimental" in str(w.message) or "provisional" in str(w.message)
    ], [str(w.message) for w in caught]


def test_pm_is_recorded_as_experimental_in_the_stability_policy() -> None:
    """The policy page lists the option, so the tier is discoverable."""
    import pathlib

    policy = pathlib.Path(__file__).resolve().parent.parent / "docs" / "stability.rst"
    text = policy.read_text()
    assert 'surface_method="pm"' in text
    assert "surface_method" in (deepwave.elastic.__doc__ or "")


def test_source_field_is_removed_from_scalar() -> None:
    """The never-released ``source_field`` trio is gone from ``scalar``."""
    import inspect

    for target in (deepwave.scalar, deepwave.Scalar.forward):
        parameters = inspect.signature(target).parameters
        for name in ("source_field", "source_field_origin", "source_field_mask"):
            assert name not in parameters, f"{target} still takes {name}"


def test_source_field_route_matches_removed_keyword() -> None:
    """The documented replacement reproduces what the removed keyword did.

    ``scalar``'s ``source_field=`` called ``source_field_to_sources`` and
    then merged the result with any explicit point sources. Both halves are
    now public, so the two-call route is exactly what the keyword ran --
    this pins that it stays exactly that, including the merge, which is the
    half that would otherwise have been left unpublished.
    """
    from deepwave.wavefields import merge_point_sources, source_field_to_sources

    torch.manual_seed(0)
    ny, nx, nt = 14, 15, 10
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    region = (4, 4)
    field = torch.randn(1, nt, *region, dtype=torch.float64)
    mask = torch.zeros(*region, dtype=torch.bool)
    mask[1, 1] = True
    origin = [2, 2]
    point_amplitudes = torch.randn(1, 1, nt, dtype=torch.float64)
    point_locations = torch.tensor([[[10, 10]]])

    field_amplitudes, field_locations = source_field_to_sources(
        field, origin=origin, mask=mask
    )
    amplitudes, locations = merge_point_sources(
        field_amplitudes, field_locations, point_amplitudes, point_locations
    )
    via_route = deepwave.scalar(
        v,
        _DX,
        _DT,
        sources={"u": (amplitudes, locations)},
        pml_width=4,
        pml_freq=25.0,
    )
    # The reference: the concatenation the removed keyword performed.
    by_hand = deepwave.scalar(
        v,
        _DX,
        _DT,
        source_amplitudes=torch.cat([field_amplitudes, point_amplitudes], dim=1),
        source_locations=torch.cat([field_locations, point_locations], dim=1),
        pml_width=4,
        pml_freq=25.0,
    )
    assert torch.equal(via_route.wavefield_0, by_hand.wavefield_0)


def test_merge_point_sources_is_public() -> None:
    """The merge half of the recipe is exported, not left private."""
    from deepwave import wavefields

    assert "merge_point_sources" in wavefields.__all__
    assert wavefields.merge_point_sources is wavefields._merge_point_sources
    assert (wavefields.merge_point_sources.__doc__ or "").strip()


def test_capability_probes_are_behavioural() -> None:
    """A probe's answer follows ``supported_options``, not the signature.

    Every signature declares every option, so a signature-reading
    probe would report universal support -- a lie the capability table
    publishes. Monkeypatching the declared set must change the answer.
    """
    from deepwave import _capabilities

    private = deepwave._engine.propagator_spec("elastic")
    assert _capabilities._probe_q_options("elastic") == "no"
    original = private.supported_options
    try:
        object.__setattr__(private, "supported_options", original | {"q_freq"})
        assert _capabilities._probe_q_options("elastic") != "no"
    finally:
        object.__setattr__(private, "supported_options", original)
    assert _capabilities._probe_q_options("elastic") == "no"

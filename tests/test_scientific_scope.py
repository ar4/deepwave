"""Release scope and independent constitutive checks for retained attenuation."""

import importlib.machinery
from decimal import Decimal, localcontext

import pytest
import torch

import deepwave
from deepwave._gsls import q_to_relaxation_times, sls_coefficients, unrelaxed_factor


@pytest.mark.parametrize(
    ("module", "class_name"),
    [("elastic_tti", "ElasticTti"), ("viscoelastic", "Viscoelastic")],
)
def test_deferred_propagators_are_not_public(module: str, class_name: str) -> None:
    """Unqualified research implementations are absent from the package API."""
    assert not hasattr(deepwave, module)
    assert not hasattr(deepwave, class_name)
    # Inspect the actual package path, bypassing stale editable-install maps.
    assert (
        importlib.machinery.PathFinder.find_spec(
            f"deepwave.{module}", deepwave.__path__
        )
        is None
    )
    assert module not in deepwave._engine.PROPAGATOR_NAMES


def _run(q: torch.Tensor) -> deepwave.PropagatorOutput:
    model = torch.ones_like(q)
    source = torch.zeros(1, 1, 4, dtype=q.dtype)
    source[..., 0] = 1.0
    point = torch.tensor([[[5]]])
    return deepwave.viscoacoustic(
        model,
        model,
        q,
        1.0,
        1e-6,
        q_freq=0.1,
        pml_freq=0.1,
        pml_width=0,
        boundary="rigid",
        source_amplitudes_pressure_rate=source,
        source_locations_pressure_rate=point,
        receiver_locations_p=point,
        backend="eager",
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), -float("inf")])
def test_attenuation_rejects_invalid_quality(value: float) -> None:
    """Quality factors must be positive; NaN and negative infinity are invalid."""
    with pytest.raises(
        deepwave.DeepwaveValueError, match=r"q must be positive|q > 0|finite q"
    ):
        _run(torch.full((16,), value))


def test_small_positive_quality_has_finite_correct_relaxation() -> None:
    """Low-Q relaxation agrees with high-precision constitutive theory."""
    q = torch.tensor(1e-4, dtype=torch.float32, requires_grad=True)
    epsilon, sigma = q_to_relaxation_times(q, 1.0)
    assert sigma > 0
    # Evaluate the original (unrationalized) formula with enough precision to
    # resolve its cancellation; this oracle does not reuse the implementation.
    with localcontext() as context:
        context.prec = 60
        inv_q = 1 / Decimal.from_float(float(q.detach()))
        sigma_reference = (1 + inv_q * inv_q).sqrt() - inv_q
        epsilon_reference = 1 / sigma_reference
    assert float(sigma.detach()) == pytest.approx(float(sigma_reference), rel=2e-7)
    assert float(epsilon.detach()) == pytest.approx(float(epsilon_reference), rel=2e-7)
    # Independent frequency-domain SLS constitutive law, evaluated at omega=1.
    modulus = (1 + 1j * epsilon.double()) / (1 + 1j * sigma.double())
    actual_q = modulus.real / modulus.imag
    torch.testing.assert_close(actual_q, q.double(), rtol=2e-6, atol=0)
    m1, c2, ar = sls_coefficients(torch.ones_like(q), q, 1e-6, 1.0)
    torch.testing.assert_close(m1, unrelaxed_factor(q), rtol=2e-6, atol=0)
    assert all(torch.isfinite(t).all() for t in (m1, c2, ar))
    assert torch.isfinite(torch.autograd.grad(m1 + c2 + ar, q)[0])
    out = _run(q.expand(16))
    assert torch.isfinite(out.receiver_amplitudes_p).all()
    assert torch.isfinite(torch.autograd.grad(out.receiver_amplitudes_p.sum(), q)[0])


@pytest.mark.parametrize("quality", [1e12, float("inf")])
def test_large_quality_is_supported_lossless_limit(quality: float) -> None:
    """Both large finite Q and positive infinity support the lossless limit.

    test_viscoacoustic.test_acoustic_limit independently compares this limit
    with acoustic propagation and its gradients on eager and native backends.
    """
    out = _run(torch.full((16,), quality))
    assert torch.isfinite(out.receiver_amplitudes_p).all()


def test_unrepresentable_unrelaxed_stiffness_is_rejected() -> None:
    """Positive Q cannot promise a finite float32 unrelaxed modulus."""
    q = torch.full((16,), 1e-30)
    epsilon, sigma = q_to_relaxation_times(q, 1.0)
    assert torch.isfinite(epsilon).all()
    assert (sigma > 0).all()
    with pytest.raises(deepwave.DeepwaveValueError, match=r"unrelaxed.*non-finite"):
        _run(q)


@pytest.mark.parametrize("band", [(1.0, float("inf")), (float("nan"), 2.0)])
def test_attenuation_band_requires_finite_frequencies(
    band: tuple[float, float],
) -> None:
    """Invalid bands fail before constructing the relaxation fit."""
    model = torch.ones(16)
    with pytest.raises(deepwave.DeepwaveValueError, match=r"q_band.*finite"):
        deepwave.viscoacoustic(
            model, model, model, 1.0, 0.1, q_n_mechanisms=3, q_band=band
        )


@pytest.mark.parametrize("count", [True, False, 1.5, "3", float("inf")])
def test_attenuation_mechanism_count_requires_integer(count: object) -> None:
    """Mechanism counts cannot silently truncate or use boolean switches."""
    model = torch.ones(16)
    with pytest.raises(deepwave.DeepwaveTypeError, match=r"q_n_mechanisms.*integer"):
        deepwave.viscoacoustic(model, model, model, 1.0, 0.1, q_n_mechanisms=count)


@pytest.mark.parametrize("band", [(True, 2.0), (1.0, "2"), (1.0,), (1.0, 2.0, 3.0)])
def test_attenuation_band_requires_two_numeric_endpoints(band: object) -> None:
    """Band endpoints have the same numeric contract as the reference frequency."""
    model = torch.ones(16)
    with pytest.raises(
        (deepwave.DeepwaveTypeError, deepwave.DeepwaveValueError), match="q_band"
    ):
        deepwave.viscoacoustic(
            model, model, model, 1.0, 0.1, q_n_mechanisms=3, q_band=band
        )


@pytest.mark.parametrize("entry", ["direct", "segments", "checkpoint", "plan"])
@pytest.mark.parametrize(
    "config",
    [
        {"q_n_mechanisms": 1.5, "q_band": (1.0, 2.0)},
        {"q_n_mechanisms": True, "q_freq": 1.0},
        {"q_n_mechanisms": 3, "q_band": (1.0, float("inf"))},
        {"q_n_mechanisms": 3, "q_band": (True, 2.0)},
        {"q_freq": float("nan")},
        {"q_freq": 0.1, "_quality": 0.0},
        {"q_freq": 0.1, "_quality": 1e-30},
    ],
)
def test_attenuation_configuration_validation_composes(
    entry: str, config: dict
) -> None:
    """Direct, planned and segmented entry points reject the same bad physics."""
    model = torch.ones(16)
    config = dict(config)
    quality = torch.full_like(model, config.pop("_quality", 1.0))
    source = torch.zeros(1, 1, 4)
    common = {
        "source_locations_pressure_rate": torch.tensor([[[5]]]),
        "pml_freq": 0.1,
        "pml_width": 2,
        "backend": "eager",
        **config,
    }

    def run() -> None:
        if entry == "segments":
            deepwave.checkpointing.run_segments(
                deepwave.viscoacoustic,
                [model, model, quality],
                1.0,
                0.1,
                {"source_amplitudes_pressure_rate": source},
                n_segments=2,
                **common,
            )
        else:
            if entry == "checkpoint":
                common["checkpoint_segments"] = 2
            if entry == "plan":
                deepwave.plan(
                    "viscoacoustic",
                    model,
                    model,
                    quality,
                    1.0,
                    0.1,
                    source_amplitudes_pressure_rate=source,
                    **common,
                )
            else:
                deepwave.viscoacoustic(
                    model,
                    model,
                    quality,
                    1.0,
                    0.1,
                    source_amplitudes_pressure_rate=source,
                    **common,
                )

    with pytest.raises(
        (deepwave.DeepwaveTypeError, deepwave.DeepwaveValueError),
        match=r"q_|q must|q > 0|finite q|unrelaxed",
    ):
        run()

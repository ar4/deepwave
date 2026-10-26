"""Physical material-domain checks on the real propagator preparation path."""

import pytest
import torch

import deepwave


def _plan(name: str, models: tuple[torch.Tensor, ...], **extra: object) -> None:
    """Resolve a small real call through functional preparation and plan."""
    ndim = models[0].ndim
    location = torch.tensor([[[4] * ndim]])
    spec = deepwave.spec(name)
    source = spec.source_channels(ndim)[0]
    receiver = spec.receiver_channels(ndim)[0]
    options = {
        "nt": 2,
        "pml_width": 2,
        "pml_freq": 25.0,
        "max_vel": 10.0,
        "sources": {source: (torch.zeros(1, 1, 2, dtype=torch.float64), location)},
        "receivers": {receiver: location},
        **extra,
    }
    deepwave.plan(name, *models, 5.0, 0.001, **options)


@pytest.mark.parametrize(
    ("ndim", "lamb", "mu"),
    [(1, -1.5, 1.0), (2, -0.5, 1.0), (3, -0.5, 1.0), (3, 1.0, 0.0)],
)
def test_negative_lambda_and_fluid_are_valid_in_represented_dimension(
    ndim: int, lamb: float, mu: float
) -> None:
    """The retained strain energy controls validity, not lambda's sign."""
    one = torch.ones((8,) * ndim, dtype=torch.float64)
    _plan("elastic", (one * lamb, one * mu, one))


@pytest.mark.parametrize(
    ("ndim", "lamb", "mu"),
    [(2, 1.0, -0.1), (2, -1.1, 1.0), (3, -0.9, 1.0)],
)
def test_invalid_elastic_energy_rejected_early(
    ndim: int, lamb: float, mu: float
) -> None:
    """A positive longitudinal modulus alone does not suffice in 2/3-D."""
    one = torch.ones((8,) * ndim, dtype=torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError, match="material requires"):
        _plan("elastic", (one * lamb, one * mu, one))


def test_vacuum_requires_zero_stiffness_and_valid_fluid_remains_allowed() -> None:
    """Vacuum is an all-zero sentinel; an active zero-shear fluid is valid."""
    one = torch.ones((8, 8), dtype=torch.float64)
    lamb, mu, buoyancy = one.clone(), torch.zeros_like(one), one.clone()
    for model in (lamb, mu, buoyancy):
        model[0] = 0
    _plan("elastic", (lamb, mu, buoyancy))
    lamb[0] = 1
    with pytest.raises(deepwave.DeepwaveValueError, match="zero stiffness in vacuum"):
        _plan("elastic", (lamb, mu, buoyancy))


def test_positive_looking_vti_blocks_can_be_invalid() -> None:
    """The active normal block, including c66 in 3-D, must have positive energy."""
    one2 = torch.ones(8, 8, dtype=torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError, match="normal block"):
        _plan("elastic_vti", (one2, one2 * 1.1, one2, one2 * 0.2, one2))
    one3 = torch.ones(8, 8, 8, dtype=torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError, match="normal block"):
        _plan(
            "elastic_vti",
            (one3, one3 * 0.5, one3, one3 * 0.2, one3),
            c66=one3 * 0.9,
        )


def test_pseudo_acoustic_elliptical_limit_allowed_but_negative_diagonal_rejected() -> (
    None
):
    """The zero-determinant pseudo-acoustic block is an intentional limit."""
    one = torch.ones(8, 8, dtype=torch.float64)
    _plan("acoustic_vti", (one, one, one, one))
    with pytest.raises(deepwave.DeepwaveValueError, match="c11, c33"):
        _plan("acoustic_vti", (-one, one, one, one))


@pytest.mark.parametrize("name", ["scalar", "acoustic", "viscoacoustic"])
def test_signed_and_zero_speed_are_defined(name: str) -> None:
    """Squared speed permits either sign and the zero-speed limit."""
    one = torch.ones(8, dtype=torch.float64)
    models = {
        "scalar": (-one,),
        "acoustic": (-one, one),
        "viscoacoustic": (-one, one, one * 20),
    }[name]
    extra = {"q_freq": 25.0} if name == "viscoacoustic" else {}
    _plan(name, models, **extra)
    _plan(name, (torch.zeros_like(models[0]), *models[1:]), **extra)


def test_born_perturbation_is_not_a_material() -> None:
    """A negative Born direction is valid about a positive background."""
    one = torch.ones(8, dtype=torch.float64)
    _plan("scalar_born", (one,), dv=-one * 0.2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_undefined_material_values_rejected(bad: float) -> None:
    """Reject NaN, infinite speed and negative infinite Q before propagation."""
    one = torch.ones(8, dtype=torch.float64)
    speed = one.clone()
    speed[0] = bad
    with pytest.raises(deepwave.DeepwaveValueError, match="finite v"):
        _plan("acoustic", (speed, one))
    quality = one.clone() * 20
    quality[0] = -bad if bad == float("inf") else bad
    with pytest.raises(deepwave.DeepwaveValueError, match="q > 0"):
        _plan("viscoacoustic", (one, one, quality), q_freq=25.0)


def test_domain_check_shared_by_functional_module_plan_and_segments() -> None:
    """All public preparation routes reject the same invalid background."""
    speed = torch.ones(8, dtype=torch.float64)
    speed[0] = float("nan")
    amplitude = torch.zeros(1, 1, 4, dtype=torch.float64)
    location = torch.tensor([[[4]]])
    options = {
        "source_amplitudes": amplitude,
        "source_locations": location,
        "pml_width": 2,
        "pml_freq": 25.0,
        "max_vel": 10.0,
    }
    calls = (
        lambda: deepwave.scalar(speed, 5.0, 0.001, **options),
        lambda: deepwave.Scalar(speed, 5.0)(0.001, **options),
        lambda: _plan("scalar", (speed,)),
        lambda: deepwave.scalar(speed, 5.0, 0.001, checkpoint_segments=2, **options),
    )
    for call in calls:
        with pytest.raises(deepwave.DeepwaveValueError, match="finite v"):
            call()

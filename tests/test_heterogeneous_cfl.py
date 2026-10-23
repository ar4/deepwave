"""Independent weighted-operator CFL checks at sharp density interfaces."""

import math

import pytest
import torch

import deepwave
from deepwave import _common, _gsls, _staggered_grid


def _periodic_operator(speed, density, spacing, accuracy):
    """Build divergence directly from the published staggered coefficients."""
    coefficients = {
        2: (1.0,),
        4: (9 / 8, -1 / 24),
        6: (75 / 64, -25 / 384, 3 / 640),
        8: (1225 / 1024, -245 / 3072, 49 / 5120, -5 / 7168),
    }[accuracy]
    cells = density.numel()
    indices = torch.arange(cells).reshape(density.shape)
    rows = torch.arange(cells)
    blocks = []
    for axis, dx in enumerate(spacing):
        derivative = torch.zeros(cells, cells, dtype=torch.float64)
        for offset, coefficient in enumerate(coefficients):
            plus = indices.roll(-offset, axis).flatten()
            minus = indices.roll(offset + 1, axis).flatten()
            derivative.index_put_(
                (rows, plus),
                torch.full_like(rows, coefficient / dx, dtype=torch.float64),
                accumulate=True,
            )
            derivative.index_put_(
                (rows, minus),
                torch.full_like(rows, -coefficient / dx, dtype=torch.float64),
                accumulate=True,
            )
        edge_density = 0.5 * (density + density.roll(-1, axis))
        blocks.append(derivative / edge_density.flatten().sqrt()[None, :])
    return (density * speed.square()).flatten().sqrt()[:, None] * torch.cat(blocks, 1)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
@pytest.mark.parametrize("mechanisms", [0, 1, 3])
def test_weighted_heterogeneous_operator_is_below_resolved_cfl(
    ndim, accuracy, mechanisms
):
    """The full operator, including far stencil edges, satisfies h² Lambda < 4."""
    shape = (16,) if ndim == 1 else (6,) * ndim
    density = torch.ones(shape, dtype=torch.float64)
    density[(3,) * ndim] = 1e6
    speed = torch.linspace(1.0, 1.2, density.numel()).reshape(shape)
    if mechanisms:
        q = torch.logspace(-2, 2, density.numel()).reshape(shape)
        speed = _gsls.unrelaxed_speed(
            speed, q, mechanisms, (0.1, 1.0) if mechanisms > 1 else None
        )
    spacing = [1.0 + 0.3 * axis for axis in range(ndim)]
    operator = _periodic_operator(speed, density, spacing, accuracy)
    eigenvalue = float(torch.linalg.svdvals(operator)[0].square())
    bound = _staggered_grid.pressure_stability_speed(speed, density, accuracy, spacing)
    user_dt = 0.6 / (float(speed.max()) * math.sqrt(sum(dx**-2 for dx in spacing)))
    internal_dt, _ = _common.cfl_condition_n(spacing, user_dt, bound)
    assert internal_dt**2 * eigenvalue < 4
    # The previous pointwise bound really is unsafe for higher-order stencils.
    if accuracy > 2 and mechanisms == 0:
        assert user_dt**2 * eigenvalue > 4


@pytest.mark.parametrize("family", ["acoustic", "viscoacoustic"])
@pytest.mark.parametrize("accuracy", [4, 6, 8])
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_sharp_density_interface_stays_bounded_and_rejects_underbound_max_vel(
    family, accuracy, backend
):
    """A formerly explosive pressure seed survives a fixed-duration record."""
    density = torch.ones(24, dtype=torch.float64)
    density[12] = 1e6
    speed = torch.ones_like(density)
    initial = torch.zeros(1, 24, dtype=speed.dtype)
    initial[0, 12] = 1
    models = [speed, density]
    options = {}
    if family == "viscoacoustic":
        models.append(torch.full_like(speed, 1e6))
        options["q_freq"] = 0.1
    propagator = getattr(deepwave, family)
    kwargs = dict(
        accuracy=accuracy,
        nt=40,
        pml_width=0,
        boundary="rigid",
        pml_freq=0.1,
        backend=backend,
        pressure_0=initial,
        origin=[0],
        **options,
    )
    with pytest.raises(deepwave.DeepwaveValueError, match="max_vel"):
        propagator(*models, 1.0, 0.3, max_vel=2.0, **kwargs)
    output = propagator(*models, 1.0, 0.3, **kwargs)
    assert output.info.step_ratio > 2
    assert float(output.pressure_0.abs().max()) < 5


@pytest.mark.parametrize("family", ["elastic", "elastic_vti", "acoustic_vti"])
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_elastic_fluid_limit_at_sharp_density_interface(family, backend):
    """The three velocity/stress families retain the stable fluid reduction."""
    density = torch.ones(12, 16, dtype=torch.float64)
    density[:, 8] = 1e6
    initial = torch.zeros(1, 12, 16, dtype=density.dtype)
    initial[0, 6, 8] = 1
    zero = torch.zeros_like(density)
    if family == "elastic":
        models = [density, zero, density.reciprocal()]
        seeds = {"sigmayy_0": initial, "sigmaxx_0": initial}
    elif family == "elastic_vti":
        models = [density, density, density, zero, density.reciprocal()]
        seeds = {"sigmayy_0": initial, "sigmaxx_0": initial}
    else:
        models = [density, density, density, density.reciprocal()]
        seeds = {"sigma_v_0": initial, "sigma_h_0": initial}
    kwargs = dict(
        accuracy=4,
        nt=16,
        pml_width=0,
        pml_freq=0.1,
        backend=backend,
        origin=[0, 0],
        **seeds,
    )
    propagator = getattr(deepwave, family)
    with pytest.raises(deepwave.DeepwaveValueError, match="max_vel"):
        propagator(*models, 1.0, 0.06, max_vel=2.0, **kwargs)
    output = propagator(*models, 1.0, 0.06, **kwargs)
    assert output.info.step_ratio > 1
    field = output.sigma_v_0 if family == "acoustic_vti" else output.sigmaxx_0
    assert float(field.abs().max()) < 5


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_uniform_elastic_uses_directional_speed_bound(ndim, accuracy):
    """Constant fluid and solid symbols retain their physical CFL bound.

    The longitudinal/transverse eigenvalues of the isotropic Christoffel
    matrix are vp²/vs² times the squared staggered derivative norm, even
    though the full constitutive norm is larger in three dimensions.
    """
    shape = (8,) * ndim
    for vs in (0.0, 0.5):
        density = torch.full(shape, 2.0, dtype=torch.float64)
        mu = density * vs**2
        lamb = density - 2 * mu
        buoyancy = density.reciprocal()
        bound = torch.maximum(lamb + 2 * mu + (ndim - 1) * lamb.abs(), 2 * mu)
        assert (
            _staggered_grid.elastic_stability_speed(
                bound, buoyancy, accuracy, 1.0, material_models=[lamb, mu, buoyancy]
            )
            == 1.0
        )
        out = deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            1.0,
            0.1,
            nt=4,
            accuracy=accuracy,
            max_vel=1.0,
            pml_width=0,
            pml_freq=0.1,
            vy_0=torch.zeros((1, *shape), dtype=density.dtype),
            origin=[0] * ndim,
        )
        assert out.info.max_vel == 1.0
        assert out.info.step_ratio == 1


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_density_staggering_avoids_overflow_of_representable_mean(ndim, dtype):
    """Finite high densities retain a finite nonzero inverse at every edge."""
    from deepwave._staggered_grid import stagger_density_arithmetic_mean

    density = torch.full((3,) * ndim, torch.finfo(dtype).max * 0.75, dtype=dtype)
    expected = density.reciprocal()
    assert bool(torch.count_nonzero(expected))
    for axis, buoyancy in enumerate(stagger_density_arithmetic_mean(density, ndim)):
        interior = [slice(None)] * ndim
        interior[axis] = slice(None, -1)
        assert torch.equal(buoyancy[tuple(interior)], expected[tuple(interior)])
        edge = [slice(None)] * ndim
        edge[axis] = -1
        assert torch.count_nonzero(buoyancy[tuple(edge)]) == 0


def test_pressure_bound_accepts_vacuum_and_uniform_density():
    """Zero-density nodes use the propagator's zero-buoyancy convention."""
    speed = torch.ones(12, dtype=torch.float64)
    for density in (torch.ones_like(speed), torch.tensor([0.0, 1.0] * 6)):
        bound = _staggered_grid.pressure_stability_speed(speed, density, 4, [1.0])
        assert math.isfinite(bound)
        assert bound >= 1
    assert (
        _staggered_grid.pressure_stability_speed(
            speed, torch.ones_like(speed), 4, [1.0]
        )
        == 1.0
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 8])
@pytest.mark.parametrize("low_face", ["rigid", "free"])
def test_actual_pressure_boundary_operator_respects_cfl(ndim, accuracy, low_face):
    """Recover the actual derivative matrices from independent basis seeds.

    The velocity-only and pressure-only experiments separate both staggered
    operators without using their implementation's derivative helpers. Their
    composition gives the heterogeneous wave amplification eigenvalues,
    including ghost closures and asymmetric physical grid spacing.
    """
    shape = (8,) if ndim == 1 else (6,) * ndim
    cells = math.prod(shape)
    density = torch.ones(shape, dtype=torch.float64)
    density[(3,) * ndim] = 1e6
    speed = torch.ones_like(density)
    basis = torch.eye(cells, dtype=speed.dtype).reshape(cells, *shape)
    spacing = [1.0 + 0.3 * axis for axis in range(ndim)]
    boundary = [face for _ in range(ndim) for face in (low_face, "rigid")]
    probe_dt = 1e-8
    kwargs = {
        "accuracy": accuracy,
        "nt": 1,
        "pml_width": 0,
        "pml_freq": 0.1,
        "origin": [0] * ndim,
        "boundary": boundary,
        "backend": "eager",
    }
    pressure_seed = deepwave.acoustic(
        speed, density, spacing, probe_dt, pressure_0=basis, **kwargs
    )
    gradient, divergence = [], []
    for axis in "zyx"[-ndim:]:
        gradient.append(
            -getattr(pressure_seed, f"v{axis}_0").reshape(cells, cells).T / probe_dt
        )
        velocity_seed = deepwave.acoustic(
            speed, density, spacing, probe_dt, **{f"v{axis}_0": basis}, **kwargs
        )
        divergence.append(-velocity_seed.pressure_0.reshape(cells, cells).T / probe_dt)
    amplification = -torch.cat(divergence, 1) @ torch.cat(gradient, 0)
    eigenvalues = torch.linalg.eigvals(amplification)
    maximum = float(eigenvalues.real.max())
    assert maximum > 0
    assert float(eigenvalues.imag.abs().max()) < 1e-8 * maximum
    requested_dt = 0.6 / math.sqrt(sum(dx**-2 for dx in spacing))
    internal_dt, _ = _common.cfl_condition_n(
        spacing, requested_dt, pressure_seed.info.max_vel
    )
    assert internal_dt**2 * maximum < 4


@pytest.mark.parametrize("backend", ["eager", "compiled"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_automatic_segmented_cfl_preserves_whole_record(backend, checkpoint):
    """Small material magnitudes must not hide a weighted CFL subdivision.

    A density ratio of 1e6 gives R=3 despite every primitive value being <=1.
    Segment-local FFT resampling changes this receiver by about 9.7%.
    Whole-record preparation must preserve values and all nonzero gradients.
    """
    speed = torch.ones(12, 16, dtype=torch.float64)
    density = torch.full_like(speed, 1e-6)
    density[:, 8] = 1.0
    amp = torch.zeros(1, 1, 16, dtype=speed.dtype)
    amp[..., 3:7] = torch.tensor([0.2, 1.0, 0.3, -0.1])
    kwargs = {
        "source_locations_volume_rate": torch.tensor([[[6, 8]]]),
        "receiver_locations_p": torch.tensor([[[6, 9]]]),
        "pml_width": 0,
        "pml_freq": 0.1,
        "accuracy": 4,
        "backend": backend,
    }
    outputs, gradients = [], []
    for segmented in (False, True):
        leaves = [value.clone().requires_grad_() for value in (speed, density, amp)]
        if segmented:
            out = deepwave.checkpointing.run_segments(
                deepwave.acoustic,
                leaves[:2],
                1.0,
                0.06,
                {"source_amplitudes_volume_rate": leaves[2]},
                2,
                checkpoint=checkpoint,
                **kwargs,
            )
        else:
            out = deepwave.acoustic(
                *leaves[:2],
                1.0,
                0.06,
                source_amplitudes_volume_rate=leaves[2],
                **kwargs,
            )
        assert out.info.step_ratio == 3
        loss = out.receiver_amplitudes_p.square().sum() * 1e12
        assert float(loss.detach()) > 1
        loss.backward()
        outputs.append(out)
        gradients.append([leaf.grad for leaf in leaves])
    for name in ("pressure_0", "receiver_amplitudes_p"):
        assert torch.allclose(
            getattr(outputs[0], name), getattr(outputs[1], name), rtol=1e-11, atol=1e-14
        )
    for direct, segmented in zip(*gradients):
        assert direct is not None
        assert segmented is not None
        assert float(direct.norm()) > 0
        relative = (direct - segmented).norm() / direct.norm()
        assert float(relative) < 1e-10


@pytest.mark.parametrize("family", ["acoustic", "viscoacoustic"])
def test_segmented_measure_checks_actual_weighted_cfl(family):
    """The user-rate DFT must reject a hidden subdivision before streaming."""
    speed = torch.ones(12, 16, dtype=torch.float64)
    density = torch.full_like(speed, 1e-6)
    density[:, 8] = 1.0
    models = [speed, density]
    kwargs = {
        "source_locations_pressure_rate": torch.tensor([[[6, 8]]]),
        "receiver_locations_p": torch.tensor([[[6, 9]]]),
        "pml_width": 0,
        "pml_freq": 0.1,
        "accuracy": 4,
    }
    amplitude_name = "source_amplitudes_pressure_rate"
    if family == "acoustic":
        kwargs["source_locations_volume_rate"] = kwargs.pop(
            "source_locations_pressure_rate"
        )
        amplitude_name = "source_amplitudes_volume_rate"
    else:
        models.append(torch.full_like(speed, 20.0))
        kwargs["q_freq"] = 0.1
    with pytest.raises(deepwave.DeepwaveValueError, match="step_ratio == 1"):
        deepwave.wavefields.measure(
            getattr(deepwave, family),
            models,
            1.0,
            0.06,
            {amplitude_name: torch.ones(1, 1, 16, dtype=speed.dtype)},
            [deepwave.wavefields.RunningDFT([0.1])],
            n_segments=2,
            **kwargs,
        )

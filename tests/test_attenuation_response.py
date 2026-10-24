"""Independent algebra and constrained-fit checks for passive attenuation."""

import itertools
import math

import pytest
import torch

import deepwave
from deepwave import _common, _gsls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("q_value", [0.01, 1.0, 20.0, 1e6])
def test_single_sls_response_matches_independent_rational_modulus(dtype, q_value):
    """The complex response agrees with the independent two-time SLS ratio."""
    q = torch.tensor([q_value], dtype=dtype, requires_grad=True)
    f = torch.tensor([0.0, 5.0, 25.0, 50.0], dtype=torch.float64)
    actual = deepwave.attenuation.response(q, f, q_freq=25.0)
    inv_q = 1.0 / q_value
    factor = math.sqrt(1.0 + inv_q * inv_q) + inv_q
    tau_sigma = 1.0 / (2.0 * math.pi * 25.0 * factor)
    tau_epsilon = factor / (2.0 * math.pi * 25.0)
    z = 1j * 2.0 * math.pi * f
    expected = (1.0 + z * tau_epsilon) / (1.0 + z * tau_sigma)
    tolerance = 4e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(
        actual.detach().cpu(),
        torch.as_tensor(expected).to(actual.dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    assert actual[0, 0] == 1
    assert bool((actual.imag >= 0).all())
    actual.real.sum().backward()
    assert q.grad is not None
    assert bool(torch.isfinite(q.grad).all())
    assert bool((q.grad != 0).all())


def _independent_design(band, n):
    f1, f2 = band
    log_f1, log_f2 = math.log(f1), math.log(f2)
    peaks = torch.tensor(
        [math.exp(log_f1 + (log_f2 - log_f1) * i / (n - 1)) for i in range(n)],
        dtype=torch.float64,
    )
    probes = torch.exp(
        torch.linspace(log_f1, log_f2, max(200, 20 * n), dtype=torch.float64)
    )
    x = probes[:, None] / peaks[None, :]
    return x / (1 + x.square())


@pytest.mark.parametrize("band", [(10.0, 10.001), (10.0, 20.0), (1.0, 1e4)])
@pytest.mark.parametrize("n", [2, 3])
def test_small_nnls_fit_matches_independent_subset_enumeration(band, n):
    """A complete small active-set enumeration gives the same objective."""
    design = _independent_design(band, n)
    target = torch.ones(design.shape[0], dtype=torch.float64)
    best = float("inf")
    for count in range(n + 1):
        for subset in itertools.combinations(range(n), count):
            if subset:
                solution = torch.linalg.lstsq(
                    design[:, subset], target[:, None], driver="gelsd"
                ).solution[:, 0]
                if bool((solution < -1e-10).any()):
                    continue
                predicted = design[:, subset] @ solution
            else:
                predicted = torch.zeros_like(target)
            best = min(best, float((predicted - target).square().sum()))
    _, weights = _gsls.gsls_fit_weights(band, n, torch.float64, torch.device("cpu"))
    objective = float((design @ weights - target).square().sum())
    assert bool((weights >= 0).all())
    assert objective <= best + 1e-8


@pytest.mark.parametrize("n", [2, 3, 8, 32])
@pytest.mark.parametrize("ratio", [1.000001, 2.0, 100.0, 1e4])
def test_gsls_fit_kkt_and_global_passivity(n, ratio):
    """The fit is optimal to tolerance and its modulus stays passive."""
    band = (10.0, 10.0 * ratio)
    _, w = _gsls.gsls_fit_weights(band, n, torch.float64, torch.device("cpu"))
    design = _independent_design(band, n)
    grad = design.T @ (design @ w - 1)
    scale = max(
        1.0,
        float(
            (design.T @ torch.ones(design.shape[0], dtype=torch.float64)).abs().max()
        ),
    )
    assert bool((w >= 0).all())
    assert bool(torch.isfinite(w).all())
    assert float(grad[w > 0].abs().max()) < 3e-10 * scale
    if bool((w == 0).any()):
        assert float(grad[w == 0].min()) > -3e-10 * scale
    q = torch.tensor([0.01, 1.0, 20.0, 1e6], dtype=torch.float64)
    frequencies = torch.tensor([0.0, 1e-5, 1.0, 10.0, 1e4, 1e8])
    modulus = deepwave.attenuation.response(
        q, frequencies, q_n_mechanisms=n, q_band=band
    )
    assert bool(torch.isfinite(modulus).all())
    assert bool((modulus.real >= 1).all())
    assert bool((modulus.imag >= 0).all())
    assert bool((modulus[:, 0] == 1).all())


def test_multi_response_gradient_matches_finite_difference():
    """The public response retains the derivative through spatial Q."""
    q = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
    frequency = torch.tensor([8.0, 20.0, 50.0])

    def evaluate(value):
        return deepwave.attenuation.response(
            value, frequency, q_n_mechanisms=3, q_band=(5.0, 60.0)
        ).imag.sum()

    derivative = torch.autograd.grad(evaluate(q), q)[0]
    h = 1e-4
    difference = (evaluate(q.detach() + h) - evaluate(q.detach() - h)) / (2 * h)
    assert abs(float(derivative)) > 1e-6
    assert torch.allclose(derivative, difference, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mechanisms", [1, 3])
def test_high_frequency_response_and_q_gradient_match_unrelaxed_limit(
    dtype, mechanisms
):
    """Finite frequencies must not poison Q gradients through unused branches."""
    q = torch.tensor([20.0], dtype=dtype, requires_grad=True)
    frequency = torch.tensor([0.0, torch.finfo(dtype).max], dtype=dtype)
    options = (
        {"q_freq": 25.0}
        if mechanisms == 1
        else {"q_n_mechanisms": mechanisms, "q_band": (10.0, 100.0)}
    )
    result = deepwave.attenuation.response(q, frequency, **options)
    if mechanisms == 1:
        inv_q = q.reciprocal()
        expected = 1.0 + 2.0 * inv_q * (torch.sqrt(1.0 + inv_q.square()) + inv_q)
    else:
        _, weights = _gsls.gsls_fit_weights((10.0, 100.0), mechanisms, dtype, q.device)
        expected = 1.0 + weights.sum() / q
    gradient = torch.autograd.grad(result.real.sum() + result.imag.sum(), q)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), q)[0]
    assert bool(torch.isfinite(result).all())
    assert bool(torch.isfinite(gradient).all())
    assert result[0, 0] == 1.0
    tolerance = 4e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(result[:, 1].real, expected, rtol=tolerance, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=tolerance, atol=0)


@pytest.mark.parametrize("mechanisms", [1, 3])
def test_response_frequency_derivatives_at_dc_and_relaxation_peak(mechanisms):
    """The stable rational evaluation retains smooth first and second derivatives."""
    q = torch.tensor([20.0], dtype=torch.float64, requires_grad=True)
    root = math.sqrt(1.0 + 1.0 / 400.0) + 1.0 / 20.0
    frequency = torch.tensor(
        [0.0, 25.0 * root], dtype=torch.float64, requires_grad=True
    )
    options = (
        {"q_freq": 25.0}
        if mechanisms == 1
        else {"q_n_mechanisms": mechanisms, "q_band": (10.0, 100.0)}
    )
    if mechanisms == 1:
        inverse = q.reciprocal()
        factor = torch.sqrt(1.0 + inverse.square()) + inverse
        x = frequency / (25.0 * factor.unsqueeze(-1))
        strength = (2.0 * inverse * factor).unsqueeze(-1)
    else:
        times, weights = _gsls.gsls_fit_weights(
            (10.0, 100.0), mechanisms, q.dtype, q.device
        )
        x = 2.0 * math.pi * times[:, None, None] * frequency
        strength = weights[:, None, None] / q[None, :, None]
    storage = strength * x.square() / (1.0 + x.square())
    loss = strength * x / (1.0 + x.square())
    expected = (storage + loss).sum()
    actual = deepwave.attenuation.response(q, frequency, **options)
    actual_gradient = torch.autograd.grad(
        (actual.real + actual.imag).sum(), frequency, create_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected, frequency, create_graph=True)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1e-12, atol=1e-14
    )
    # Include the DC limiting curvature as well as positive relaxation peaks.
    actual_curvature = torch.autograd.grad(actual_gradient.sum(), frequency)[0]
    expected_curvature = torch.autograd.grad(expected_gradient.sum(), frequency)[0]
    torch.testing.assert_close(
        actual_curvature, expected_curvature, rtol=1e-11, atol=1e-14
    )


@pytest.mark.parametrize("band", [(1e-300, 1e300), (1e-100, 1e100)])
def test_extreme_representable_fit_band_avoids_basis_overflow(band):
    """The Debye design is finite even when a raw frequency ratio overflows."""
    times, strengths = _gsls.gsls_fit_weights(
        band, 3, torch.float64, torch.device("cpu")
    )
    assert bool(torch.isfinite(times).all())
    assert bool((times > 0).all())
    assert bool(torch.isfinite(strengths).all())
    assert bool((strengths >= 0).all())


@pytest.mark.parametrize("mechanisms", [1, 3])
def test_unrepresentable_float32_q_rejected_before_response(mechanisms):
    """Extreme Q cannot silently return NaNs or a changed material law."""
    q = torch.tensor(1e-40, dtype=torch.float32)
    options = (
        {"q_freq": 25.0}
        if mechanisms == 1
        else {"q_n_mechanisms": mechanisms, "q_band": (10.0, 100.0)}
    )
    with pytest.raises(deepwave.DeepwaveValueError, match="representable"):
        deepwave.attenuation.response(q, torch.tensor([0.0, 25.0]), **options)


@pytest.mark.parametrize("mechanisms", [1, 3])
@pytest.mark.parametrize("q_value", [0.05, 20.0])
def test_coupled_discrete_energy_dissipates_with_production_coefficients(
    mechanisms, q_value
):
    """The coupled leapfrog/CN update obeys the passive energy identity.

    A small arbitrary divergence matrix has the required negative-transpose
    velocity pairing. The pressure and memory recurrences use the production
    coefficients; the energy and its predicted dissipation are evaluated by
    the independent Maxwell-branch representation.
    """
    generator = torch.Generator().manual_seed(2308)
    divergence = torch.randn(4, 5, dtype=torch.float64, generator=generator)
    density = torch.tensor([1.0, 1.2, 0.9, 1.5, 1.1], dtype=torch.float64)
    bulk = torch.tensor([2.0, 3.0, 2.5, 3.5], dtype=torch.float64)
    q = torch.full((4,), q_value, dtype=torch.float64)
    band = (100.0, 1000.0) if mechanisms > 1 else None
    unrelaxed = bulk * _gsls.unrelaxed_factor(q, mechanisms, band)
    weighted = unrelaxed.sqrt()[:, None] * divergence / density.sqrt()[None, :]
    eigenvalue_bound = torch.linalg.svdvals(weighted)[0].square()
    dt = float(1.5 / eigenvalue_bound.sqrt())
    m1, c2, ar = _gsls.sls_coefficients(
        bulk, q, dt, 2.0 * math.pi * 1000.0, mechanisms, band
    )
    if mechanisms == 1:
        tau_epsilon, tau_sigma = _gsls.q_to_relaxation_times(q, 2.0 * math.pi * 1000.0)
        strength = (tau_epsilon / tau_sigma - 1.0).unsqueeze(0)
        tau_sigma = tau_sigma.unsqueeze(0)
        c2, ar = c2.unsqueeze(0), ar.unsqueeze(0)
    else:
        times, weights = _gsls.gsls_fit_weights(
            band, mechanisms, torch.float64, torch.device("cpu")
        )
        tau_sigma = times[:, None].expand(-1, 4)
        strength = weights[:, None] / q[None, :]
    branches = bulk[None, :] * strength
    branch_safe = torch.where(branches > 0, branches, torch.ones_like(branches))
    assert torch.allclose(m1, bulk + branches.sum(0), rtol=1e-13)
    strain = torch.randn(4, dtype=torch.float64, generator=generator) * 0.05
    z = (
        torch.randn(mechanisms, 4, dtype=torch.float64, generator=generator)
        * branches.sqrt()
        * 0.05
    )
    velocity_low = torch.randn(5, dtype=torch.float64, generator=generator) * 0.05

    def elastic_energy(e, memory):
        branch_energy = torch.where(branches > 0, memory.square() / branch_safe, 0.0)
        return 0.5 * (bulk * e.square()).sum() + 0.5 * branch_energy.sum()

    energies = []
    for _ in range(100):
        sigma = bulk * strain + z.sum(0)
        velocity_high = velocity_low - dt * (divergence.T @ sigma) / density
        theta = divergence @ velocity_high
        old_r = -z / tau_sigma
        new_r = ar * old_r - c2 * theta
        new_z = -tau_sigma * new_r
        new_strain = strain + dt * theta
        new_pressure = -sigma - dt * m1 * theta - 0.5 * dt * (old_r + new_r).sum(0)
        new_sigma = -new_pressure
        assert torch.allclose(
            new_sigma, bulk * new_strain + new_z.sum(0), rtol=1e-11, atol=1e-11
        )
        next_velocity = velocity_high - dt * (divergence.T @ new_sigma) / density
        old_energy = (
            elastic_energy(strain, z)
            + 0.5 * (velocity_low * density * velocity_high).sum()
        )
        new_energy = (
            elastic_energy(new_strain, new_z)
            + 0.5 * (velocity_high * density * next_velocity).sum()
        )
        z_mid = 0.5 * (z + new_z)
        dissipation = (
            dt
            * torch.where(
                branches > 0,
                z_mid.square() / (branch_safe * tau_sigma),
                0.0,
            ).sum()
        )
        assert torch.allclose(
            new_energy - old_energy, -dissipation, rtol=1e-9, atol=1e-10
        )
        energies.append(float(new_energy))
        strain, z, velocity_low = new_strain, new_z, velocity_high
    assert min(energies) >= -1e-12
    assert energies[-1] <= energies[0] + 1e-12


@pytest.mark.parametrize("mechanisms", [1, 3])
def test_heterogeneous_one_dimensional_operator_respects_default_cfl(mechanisms):
    """The actual staggered density mean bounds the weighted operator.

    For accuracy two, each edge density is the arithmetic mean of its two
    cells. Even a six-order density contrast then gives
    ``K_inf(cell) / rho_edge <= 2 * v_inf(cell)**2``. The test computes the
    full heterogeneous operator's largest eigenvalue rather than substituting
    a homogeneous dispersion relation.
    """
    count = 12
    dx = 10.0
    density = torch.tensor([1.0, 1e6] * (count // 2), dtype=torch.float64)
    speed = torch.linspace(1000.0, 1800.0, count, dtype=torch.float64)
    q = torch.logspace(-2, 2, count, dtype=torch.float64)
    band = (10.0, 1000.0) if mechanisms > 1 else None
    factor = _gsls.unrelaxed_factor(q, mechanisms, band)
    stiffness = density * speed.square() * factor
    maximum_speed = float((speed * factor.sqrt()).max())
    edge_density = 0.5 * (density + density.roll(-1))
    divergence = torch.zeros(count, count, dtype=torch.float64)
    for cell in range(count):
        divergence[cell, cell] = 1.0 / dx
        divergence[cell, (cell - 1) % count] = -1.0 / dx
    operator = stiffness.sqrt()[:, None] * divergence / edge_density.sqrt()[None, :]
    eigenvalue = torch.linalg.svdvals(operator)[0].square()
    assert float(eigenvalue) <= 8.0 * maximum_speed**2 / dx**2 * (1 + 1e-12)
    user_dt = 0.59 * dx / maximum_speed
    internal_dt, ratio = _common.cfl_condition_n(dx, user_dt, maximum_speed)
    assert ratio == 1
    assert float(internal_dt**2 * eigenvalue) < 4.0


def test_strong_loss_long_record_and_gradients_with_cfl_substeps():
    """A stiff three-branch run stays finite and has four live sensitivities."""
    count, nt = 64, 160
    v = torch.full((count,), 1500.0, dtype=torch.float64, requires_grad=True)
    rho = torch.full((count,), 2000.0, dtype=torch.float64, requires_grad=True)
    q = torch.full((count,), 0.01, dtype=torch.float64, requires_grad=True)
    source = torch.zeros(1, 1, nt, dtype=torch.float64)
    source[0, 0, 1] = 1000.0
    source.requires_grad_()
    kwargs = {
        "nt": nt,
        "source_amplitudes_pressure_rate": source,
        "source_locations_pressure_rate": torch.tensor([[[count // 2]]]),
        "receiver_locations_p": torch.tensor([[[count // 2]]]),
        "q_n_mechanisms": 3,
        "q_band": (10.0, 1000.0),
        "pml_width": 10,
        "pml_freq": 25.0,
    }
    native = deepwave.viscoacoustic(v, rho, q, 10.0, 0.001, **kwargs)
    eager = deepwave.viscoacoustic(
        v.detach(), rho.detach(), q.detach(), 10.0, 0.001, backend="eager", **kwargs
    )
    trace = native.receiver_amplitudes_p[0, 0]
    assert native.info.step_ratio > 1
    assert torch.allclose(
        native.receiver_amplitudes_p, eager.receiver_amplitudes_p, rtol=1e-10
    )
    assert bool(torch.isfinite(trace).all())
    assert float(trace[-20:].abs().max()) < 0.2 * float(trace.abs().max())
    native.receiver_amplitudes_p.square().sum().backward()
    for model in (v, rho, q, source):
        assert model.grad is not None
        assert bool(torch.isfinite(model.grad).all())
        assert float(model.grad.abs().max()) > 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mechanisms", [1, 3])
def test_exact_lossless_cells_and_q_derivative(dtype, mechanisms):
    """An infinite-Q cell is exactly lossless beside a finite-Q attenuating cell."""
    q = torch.tensor([20.0, float("inf")], dtype=dtype, requires_grad=True)
    frequencies = torch.tensor([0.0, 5.0, 25.0, 50.0], dtype=dtype)
    configuration = (
        {"q_freq": 25.0}
        if mechanisms == 1
        else {"q_n_mechanisms": mechanisms, "q_band": (5.0, 60.0)}
    )
    response = deepwave.attenuation.response(q, frequencies, **configuration)
    torch.testing.assert_close(
        response[1], torch.ones_like(response[1]), rtol=0, atol=0
    )
    assert bool((response[0, 1:].imag > 0).all())
    (gradient,) = torch.autograd.grad(response.imag.sum(), q)
    assert torch.isfinite(gradient).all()
    assert gradient[0] != 0
    assert gradient[1] == 0

"""Tests for deepwave.time_dispersion.

The correction removes the O(dt^2) temporal dispersion of the explicit
second-order leapfrog time stepping. The ground-truth checks are:

1. The exact scalar arbiter: the leapfrog recursion of a single oscillator,
   ``u_{n+1} - 2 u_n + u_{n-1} = -dt^2 Omega0^2 u_n``, oscillates at exactly
   ``omega_num = (2/dt) arcsin(Omega0 dt/2)`` -- pinned to machine precision
   with no wave equation.
2. A direct propagation check: the coarse-dt receiver phase is shifted relative
   to the fine-dt reference by exactly the amount ``Omega(omega)`` predicts, and
   the shift is removed by the correction.
3. Coarse-vs-fine: the corrected coarse-dt traces match the fine-dt traces with
   a large (>5x) reduction in relative L2 error versus the uncorrected traces.
4. Round-trip (``correct_receivers(correct_source(x)) ~= x``) to ``<1e-6`` for
   band-limited ``x``, and differentiability (gradcheck) through both transforms.
"""

import math
import re

import numpy as np
import pytest
import torch

import deepwave
from deepwave import scalar
from deepwave.discretisation import cfl_condition_n, downsample, upsample
from deepwave.time_dispersion import (
    cfl_internal_dt,
    correct_receivers,
    correct_source,
)


def _omega_map(omega: torch.Tensor, dt: float) -> torch.Tensor:
    """Physical frequency Omega(omega) = (2/dt) sin(omega dt/2)."""
    return (2.0 / dt) * torch.sin(omega * dt / 2.0)


def _omega_inverse_map(physical: torch.Tensor, dt: float) -> torch.Tensor:
    """Simulation frequency omega(Omega) = (2/dt) arcsin(Omega dt/2)."""
    return (2.0 / dt) * torch.arcsin(physical * dt / 2.0)


# ---------------------------------------------------------------------------
# 1. Exact leapfrog dispersion relation (the load-bearing identity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.001, 0.002, 0.004])
@pytest.mark.parametrize("freq0", [5.0, 12.0, 30.0])
def test_leapfrog_dispersion_relation(dt: float, freq0: float) -> None:
    """u_n = sin(omega_num n dt) solves the leapfrog oscillator exactly.

    Verifies ``omega_num = (2/dt) arcsin(Omega0 dt/2)`` to machine precision by
    substituting the closed-form mode into the recursion
    ``u_{n+1} - 2 u_n + u_{n-1} = -dt^2 Omega0^2 u_n``.
    """
    omega0 = 2.0 * math.pi * freq0
    omega_num = (2.0 / dt) * math.asin(omega0 * dt / 2.0)
    n = torch.arange(0, 60, dtype=torch.float64)
    for mode in (torch.sin(omega_num * n * dt), torch.cos(omega_num * n * dt)):
        lhs = mode[2:] - 2.0 * mode[1:-1] + mode[:-2]
        rhs = -(dt**2) * omega0**2 * mode[1:-1]
        assert (lhs - rhs).abs().max().item() < 1e-12
    # And the numerical frequency is genuinely shifted up: omega_num > Omega0.
    assert omega_num > omega0


def test_frequency_maps_are_mutual_inverses() -> None:
    """Omega(omega(x)) == x and omega(Omega(x)) == x on the valid band."""
    dt = 0.003
    # Physical frequencies strictly below the 2/dt bound.
    physical = torch.linspace(0.0, 0.999 * (2.0 / dt), 200, dtype=torch.float64)
    round_a = _omega_map(_omega_inverse_map(physical, dt), dt)
    assert torch.allclose(round_a, physical, atol=1e-9)
    # Simulation frequencies up to the Nyquist-like bound pi/dt.
    sim = torch.linspace(0.0, math.pi / dt, 200, dtype=torch.float64)
    round_b = _omega_inverse_map(_omega_map(sim, dt), dt)
    assert torch.allclose(round_b, sim, atol=1e-9)


# ---------------------------------------------------------------------------
# 2 + 3. Propagation: direct phase-shift relation and coarse-vs-fine accuracy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def propagation_traces() -> dict[str, object]:
    """Run one homogeneous 2-D experiment shared by the propagation tests.

    Returns the fine-dt reference trace, the uncorrected coarse-dt trace, and
    the fully corrected coarse-dt trace (source pre-corrected, receivers
    post-corrected), all at the coarse sampling, plus the geometry needed to
    form the analytic phase prediction.
    """
    torch.manual_seed(0)
    v0 = 1500.0
    ny, nx = 180, 90
    dx = 8.0
    grid = [dx, dx]
    v = torch.full((ny, nx), v0, dtype=torch.float64)
    freq = 25.0
    dt_coarse = 0.0022
    ratio = 6
    dt_fine = dt_coarse / ratio
    t_total = 0.72
    nt_c = round(t_total / dt_coarse)
    internal_dt = cfl_internal_dt(grid, dt_coarse, v0)
    # Chosen so the coarse run needs no internal subdivision: internal_dt == dt.
    assert cfl_condition_n(grid, dt_coarse, v0)[1] == 1
    assert cfl_condition_n(grid, dt_fine, v0)[1] == 1

    src_row, rec_row, col = 20, 100, 45
    offset = (rec_row - src_row) * dx
    src_loc = torch.tensor([[[src_row, col]]])
    rec_loc = torch.tensor([[[rec_row, col]]])

    def run(dt: float, wavelet: torch.Tensor) -> torch.Tensor:
        out = scalar(
            v,
            grid,
            dt,
            source_amplitudes=wavelet.reshape(1, 1, -1),
            source_locations=src_loc,
            receiver_locations=rec_loc,
            pml_width=20,
            pml_freq=freq,
            accuracy=6,
        )
        return out[-1].reshape(-1)

    wav_c = deepwave.wavelets.ricker(
        freq, nt_c, dt_coarse, peak_time=1.6 / freq, dtype=torch.float64
    )
    # The fine source is the SAME continuous wavelet, band-limited-resampled to
    # the fine grid, so the only difference between the runs is the time step.
    wav_f = upsample(wav_c.reshape(1, -1), ratio).reshape(-1)

    rec_fine = downsample(run(dt_fine, wav_f).reshape(1, -1), ratio).reshape(-1)
    rec_coarse = run(dt_coarse, wav_c)
    rec_corrected = correct_receivers(
        run(dt_coarse, correct_source(wav_c, internal_dt)), internal_dt
    )

    n = min(len(rec_fine), len(rec_coarse), len(rec_corrected))
    return {
        "fine": rec_fine[:n],
        "coarse": rec_coarse[:n],
        "corrected": rec_corrected[:n],
        "internal_dt": internal_dt,
        "dt": dt_coarse,
        "offset": offset,
        "v0": v0,
        "n": n,
    }


def test_direct_relation_phase_shift(propagation_traces: dict[str, object]) -> None:
    """Coarse-dt phase shift matches Omega(omega); correction removes it.

    In a homogeneous medium the receiver phase advances by ``k(omega) * offset``
    with ``k = Omega(omega)/v`` for the leapfrog scheme, so the coarse-minus-fine
    phase difference over the source band is ``(omega - Omega(omega)) * offset /
    v``. We verify the measured difference matches this prediction and collapses
    after correction.
    """
    fine = propagation_traces["fine"]
    coarse = propagation_traces["coarse"]
    corrected = propagation_traces["corrected"]
    dt = propagation_traces["dt"]
    internal_dt = propagation_traces["internal_dt"]
    offset = propagation_traces["offset"]
    v0 = propagation_traces["v0"]
    n = propagation_traces["n"]

    d_fine = torch.fft.rfft(fine)
    d_coarse = torch.fft.rfft(coarse)
    d_corr = torch.fft.rfft(corrected)
    freqs = torch.fft.rfftfreq(n, dt)
    omega = 2.0 * math.pi * freqs
    band = (freqs >= 14.0) & (freqs <= 32.0)

    measured = torch.from_numpy(
        np.unwrap(torch.angle((d_coarse / d_fine)[band]).numpy())
    )
    predicted = ((omega - _omega_map(omega, internal_dt)) * offset / v0)[band]
    # The unwrapped measurement carries an arbitrary constant; compare shapes by
    # removing the mean of each. The measured span is well below pi (safe unwrap).
    measured_c = measured - measured.mean()
    predicted_c = predicted - predicted.mean()
    assert (measured_c - predicted_c).abs().max().item() < 0.03
    # The dispersion is real (not a null test): the phase varies across the band.
    uncorrected_span = (measured_c.max() - measured_c.min()).item()
    assert uncorrected_span > 0.2

    corrected_ph = torch.from_numpy(
        np.unwrap(torch.angle((d_corr / d_fine)[band]).numpy())
    )
    corrected_ph = corrected_ph - corrected_ph.mean()
    corrected_span = (corrected_ph.max() - corrected_ph.min()).item()
    # Correction shrinks the residual phase distortion by a large factor.
    assert corrected_span < uncorrected_span / 5.0


def test_coarse_vs_fine_improvement(propagation_traces: dict[str, object]) -> None:
    """Corrected coarse-dt traces match fine-dt traces far better than raw ones.

    Both runs share the grid (identical spatial dispersion, which cancels in the
    difference), so the residual after correction is dominated by the stated
    spatial-dispersion / interpolation floor. We require a >5x reduction in
    relative L2 error and an absolute corrected error below 8%.
    """
    fine = propagation_traces["fine"]
    coarse = propagation_traces["coarse"]
    corrected = propagation_traces["corrected"]

    def rel_l2(a: torch.Tensor) -> float:
        return ((a - fine).norm() / fine.norm()).item()

    uncorrected = rel_l2(coarse)
    remaining = rel_l2(corrected)
    assert uncorrected > 0.3  # meaningful dispersion is present to correct
    assert remaining < 0.08
    assert uncorrected / remaining > 5.0


# ---------------------------------------------------------------------------
# 4. Round-trip and differentiability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_round_trip_band_limited(dtype: torch.dtype) -> None:
    """inverse(forward(x)) ~= x and forward(inverse(x)) ~= x for band-limited x.

    The pre-corrected signal is broader band, so the intermediate is given a
    longer record (``nt_out``) before mapping back -- exactly the wrap-around
    headroom the API's ``nt_out`` exists to provide.
    """
    dt = 0.004
    nt = 800
    tol = 1e-6 if dtype == torch.float64 else 2e-4
    t = torch.arange(nt, dtype=torch.float64) * dt
    x = torch.zeros(nt, dtype=torch.float64)
    for freq, amp, phase in [(4.0, 1.0, 0.3), (7.0, 0.6, 1.1), (10.0, 0.4, 2.0)]:
        x = x + amp * torch.sin(2.0 * math.pi * freq * t + phase)
    x = (x * torch.hann_window(nt, periodic=False, dtype=torch.float64)).to(dtype)

    # source (forward) then receiver (inverse)
    forward = correct_source(x, dt, nt_out=4 * nt)
    recovered = correct_receivers(forward, dt, nt_out=nt)
    assert (recovered - x).norm().item() / x.norm().item() < tol

    # receiver (inverse) then source (forward)
    inverse = correct_receivers(x, dt, nt_out=4 * nt)
    recovered2 = correct_source(inverse, dt, nt_out=nt)
    assert (recovered2 - x).norm().item() / x.norm().item() < tol


def test_differentiable_gradcheck() -> None:
    """Gradients flow through both transforms (exact linear tensor ops)."""
    dt = 0.003
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: correct_source(z, dt), (x,), atol=1e-6)
    assert torch.autograd.gradcheck(lambda z: correct_receivers(z, dt), (x,), atol=1e-6)
    assert torch.autograd.gradcheck(
        lambda z: correct_source(z, dt, nt_out=9), (x,), atol=1e-6
    )


def test_gradient_flows_to_source() -> None:
    """A scalar loss on the corrected data backpropagates to the raw source."""
    dt = 0.003
    src = torch.randn(3, 1, 40, dtype=torch.float64, requires_grad=True)
    loss = correct_receivers(correct_source(src, dt), dt).pow(2).sum()
    loss.backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()
    assert src.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Batching / dtype / device-neutral behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("func", [correct_source, correct_receivers])
def test_dtype_preserved(dtype: torch.dtype, func: object) -> None:
    """The floating dtype of the input is preserved."""
    x = torch.randn(4, 32, dtype=dtype)
    assert func(x, 0.002).dtype == dtype


@pytest.mark.parametrize("func", [correct_source, correct_receivers])
def test_batched_leading_dims(func: object) -> None:
    """Arbitrary leading batch dimensions are supported; time is the last axis."""
    dt = 0.002
    x = torch.randn(3, 4, 5, 64, dtype=torch.float64)
    out = func(x, dt)
    assert out.shape == (3, 4, 5, 64)
    # Batched result equals the per-trace result.
    single = func(x[1, 2, 3], dt)
    assert torch.allclose(out[1, 2, 3], single, atol=1e-10)


@pytest.mark.parametrize("func", [correct_source, correct_receivers])
def test_nt_out_controls_length(func: object) -> None:
    """nt_out sets the output time length; default matches the input."""
    dt = 0.002
    x = torch.randn(2, 50, dtype=torch.float64)
    assert func(x, dt).shape[-1] == 50
    assert func(x, dt, nt_out=80).shape[-1] == 80
    assert func(x, dt, nt_out=30).shape[-1] == 30


@pytest.mark.parametrize("func", [correct_source, correct_receivers])
@pytest.mark.parametrize("shape", [(2, 0), (0, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nt_out", [None, 8])
def test_empty_transform_preserves_zero_input_gradient(
    func: object, shape: tuple[int, int], dtype: torch.dtype, nt_out: object
) -> None:
    """Accepted empty time/batch axes keep the transform's autograd contract."""
    signal = torch.empty(shape, dtype=dtype, requires_grad=True)
    output = func(signal, 0.002, nt_out=nt_out)
    assert output.shape == (shape[0], shape[1] if nt_out is None else nt_out)
    assert output.dtype == signal.dtype
    assert output.device == signal.device
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert signal.grad is not None
    assert signal.grad.shape == signal.shape
    assert torch.count_nonzero(signal.grad) == 0


@pytest.mark.parametrize("func", [correct_source, correct_receivers])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dt", [1e-320, 1e308])
def test_transform_is_invariant_to_time_units(
    func: object, dtype: torch.dtype, dt: float
) -> None:
    """Scaling both time axes leaves sample values and input gradients unchanged."""
    signal = torch.linspace(-1, 1, 14, dtype=dtype).reshape(2, 7).requires_grad_()
    reference = func(signal, 0.002, nt_out=9)
    actual = func(signal, dt, nt_out=9)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference)
    reference_grad = torch.autograd.grad(reference.square().sum(), signal)[0]
    actual_grad = torch.autograd.grad(actual.square().sum(), signal)[0]
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, reference_grad)


# ---------------------------------------------------------------------------
# Input validation and the CFL helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [correct_source, correct_receivers])
def test_input_validation(func: object) -> None:
    """The transforms reject malformed inputs with clear errors."""
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        func([1.0, 2.0, 3.0], 0.002)
    with pytest.raises(ValueError, match="at least one dimension"):
        func(torch.tensor(1.0), 0.002)
    with pytest.raises(ValueError, match="float32 or float64"):
        func(torch.arange(4), 0.002)
    with pytest.raises(ValueError, match="float32 or float64"):
        func(torch.randn(4, dtype=torch.bfloat16), 0.002)
    with pytest.raises(ValueError, match="float32 or float64"):
        func(torch.randn(4, dtype=torch.float16), 0.002)
    finite_positive = "internal_dt must be a finite positive number"
    with pytest.raises(ValueError, match=finite_positive):
        func(torch.randn(4), 0.0)
    with pytest.raises(ValueError, match=finite_positive):
        func(torch.randn(4), -0.002)
    with pytest.raises(ValueError, match=finite_positive):
        func(torch.randn(4), float("inf"))
    with pytest.raises(ValueError, match=finite_positive):
        func(torch.randn(4), float("nan"))
    with pytest.raises(TypeError, match="nt_out must be an int"):
        func(torch.randn(4), 0.002, nt_out=1.5)
    with pytest.raises(ValueError, match="nt_out must be positive"):
        func(torch.randn(4), 0.002, nt_out=0)


def test_cfl_internal_dt_matches_propagator() -> None:
    """cfl_internal_dt returns exactly the propagator's internal time step."""
    grid = [10.0, 12.0]
    dt = 0.004
    max_vel = 2000.0
    inner_dt, step_ratio = cfl_condition_n(grid, dt, max_vel)
    assert cfl_internal_dt(grid, dt, max_vel) == inner_dt
    assert step_ratio >= 1
    # A courant_number matching the propagator's is honoured.
    inner_dt_c, _ = cfl_condition_n(grid, dt, max_vel, courant_number=0.4)
    assert cfl_internal_dt(grid, dt, max_vel, courant_number=0.4) == inner_dt_c


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_cfl_internal_dt_single_spacing_uses_every_dimension(ndim: int) -> None:
    """A single spacing applies to every dimension, as it does in the propagators.

    The leapfrog CFL bound in ``ndim`` dimensions with equal spacing ``dx`` is
    ``max_dt = C dx / (v sqrt(ndim))``, and the propagator takes
    ``step_ratio = ceil(dt / max_dt)`` internal steps, so the internal step is
    ``dt / step_ratio``. Deriving it here from first principles, rather than
    from the sequence form of the same call, pins the dimension dependence.
    """
    dx = 4.0
    dt = 0.004
    max_vel = 1500.0
    courant_number = 0.6
    max_dt = courant_number * dx / (max_vel * math.sqrt(ndim))
    expected = dt / math.ceil(dt / max_dt)
    assert cfl_internal_dt(dx, dt, max_vel, ndim=ndim) == pytest.approx(expected)
    # Identical to spelling the spacings out per dimension.
    assert cfl_internal_dt(dx, dt, max_vel, ndim=ndim) == cfl_internal_dt(
        [dx] * ndim, dt, max_vel
    )


def test_cfl_internal_dt_single_spacing_matches_propagator() -> None:
    """The guarantee holds for the value a propagator is given as a single float."""
    dx = 4.0
    dt = 0.004
    v0 = 1500.0
    v = torch.full((20, 20), v0)
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=torch.zeros(1, 1, 3),
        source_locations=torch.zeros(1, 1, 2, dtype=torch.long),
        pml_freq=25.0,
    )
    assert cfl_internal_dt(dx, dt, v0, ndim=2) == out.info.internal_dt
    # The 1-D bound, which is what a single value alone would have implied, is
    # a different (too permissive) step.
    assert cfl_internal_dt([dx], dt, v0) != out.info.internal_dt


def test_cfl_internal_dt_single_spacing_requires_ndim() -> None:
    """A single grid_spacing value without ndim is rejected, not silently 1-D."""
    with pytest.raises(ValueError, match="ndim must be given"):
        cfl_internal_dt(4.0, 0.004, 1500.0)
    with pytest.raises(ValueError, match="ndim must be given"):
        cfl_internal_dt(torch.tensor(4.0), 0.004, 1500.0)


def test_cfl_internal_dt_invalid_ndim() -> None:
    """A bad ndim, or one disagreeing with a sequence spacing, is rejected."""
    with pytest.raises(TypeError, match=re.escape("ndim must be an int.")):
        cfl_internal_dt(4.0, 0.004, 1500.0, ndim=2.0)
    with pytest.raises(ValueError, match=re.escape("ndim must be >= 1.")):
        cfl_internal_dt(4.0, 0.004, 1500.0, ndim=0)
    with pytest.raises(RuntimeError, match="grid_spacing must have 3 elements"):
        cfl_internal_dt([4.0, 4.0], 0.004, 1500.0, ndim=3)

"""Tests for deepwave.das (distributed acoustic sensing receivers).

The physics-based tests propagate a P wave through a homogeneous isotropic
2-D elastic medium with an explosion (pressure) source, which radiates a pure
P wave, and measure it with DAS channels built from paired particle-velocity
receivers at the gauge endpoints (float coordinates -> Hicks interpolation):

* directivity: the measured strain-rate amplitude follows the cos^2(theta)
  law of axial-strain sensing (broadside ~ blind);
* gauge-length convergence: as the gauge length shrinks, the gauge-averaged
  strain rate converges (quadratically, like a centred difference) to the
  pointwise t.grad(v).t computed from a very fine finite difference of
  adjacent velocity receivers;
* differentiability: an L2 loss on DAS channels has a nonzero gradient with
  respect to the elastic model parameters that matches a directional finite
  difference (float64);
* units: the same physical experiment on grids with different spacing gives
  the same physical strain rate (gauge_length is physical, not cells).

Additional closed-form / propagator-based tests cover
``deepwave.das.gauge_average`` (exactness on a linear-in-position field,
even/odd window convention, differentiability) and
``deepwave.das.helical_fiber_response`` (alpha=0 reduces to the axial
response, the "magic angle" weights are exactly 1/3 each and always sum to
1, differentiability, and a synthetic plane-wave check of the combined
axial+transverse response against the analytic cos^2/sin^2 prediction).
"""

import math

import pytest
import torch

from deepwave import DeepwaveTypeError, DeepwaveValueError, das, elastic
from deepwave.wavelets import ricker


@pytest.mark.parametrize("spacing", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_channel_spacing_rejected(spacing):
    """Invalid spacing cannot silently turn gauge averaging into a no-op."""
    with pytest.raises(DeepwaveValueError, match="channel_spacing"):
        das.gauge_average(torch.ones(1, 4, 8), 2.0, spacing)


# arccos(1/sqrt(3)) radians (54.7356 degrees): the helical "magic angle" at
# which the three cable-frame weights are each exactly 1/3.
MAGIC_ANGLE = math.acos(1.0 / math.sqrt(3.0))

# Homogeneous medium shared by the physics tests.
VP = 1500.0
VS = 800.0
RHO = 2200.0
FREQ = 25.0
DX = 5.0
DT = 0.001


def _run_elastic_p_source(
    ny,
    nx,
    source_cell,
    receiver_locations,
    nt,
    dx=DX,
    dt=DT,
    freq=FREQ,
    amplitude=1.0,
    requires_grad=False,
    accuracy=4,
    max_vel=VP,
):
    """Propagate an explosion (P) source; record vy and vx at the locations.

    ``receiver_locations`` is a float tensor [n, 2] in grid cells. Both
    velocity components are recorded at every location (the propagator's
    Hicks interpolation handles the per-component staggered offsets, so both
    sample the same physical points). Returns (vy, vx, models), each
    receiver tensor of shape [1, n, nt_out].
    """
    dtype = torch.double
    lamb = torch.full((ny, nx), RHO * (VP**2 - 2 * VS**2), dtype=dtype)
    mu = torch.full((ny, nx), RHO * VS**2, dtype=dtype)
    buoyancy = torch.full((ny, nx), 1.0 / RHO, dtype=dtype)
    if requires_grad:
        lamb.requires_grad_()
        mu.requires_grad_()
        buoyancy.requires_grad_()

    peak_time = 1.5 / freq
    source_amplitudes = amplitude * ricker(
        freq, nt, dt, peak_time, dtype=dtype
    ).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(source_cell)]])
    rec = receiver_locations.to(dtype).unsqueeze(0)  # [1, n, 2] float -> Hicks

    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=source_amplitudes,
        source_locations_pressure_rate=source_locations,
        receiver_locations_y=rec,
        receiver_locations_x=rec,
        pml_width=20,
        pml_freq=freq,
        max_vel=max_vel,
        accuracy=accuracy,
    )
    return (
        out.receiver_amplitudes_y,
        out.receiver_amplitudes_x,
        (lamb, mu, buoyancy),
    )


def _fiber_points(center, direction, n_points, spacing_cells=1.0):
    """Ordered points of a straight fibre through ``center`` (grid cells)."""
    direction = torch.tensor(direction, dtype=torch.double)
    direction = direction / direction.norm()
    center = torch.tensor(center, dtype=torch.double)
    offsets = (torch.arange(n_points, dtype=torch.double) - (n_points - 1) / 2) * (
        spacing_cells
    )
    return center + offsets[:, None] * direction


def test_fiber_gauge_locations_straight_fiber() -> None:
    """Endpoints straddle the channels along the tangent, gauge/2 each side."""
    pts = _fiber_points([10.0, 20.0], [0.0, 1.0], 5)
    gauge_length = 10.0  # m; dx = 5 m -> +/- 1 cell
    plus, minus, tangents = das.fiber_gauge_locations(pts, gauge_length, 5.0)
    assert plus.shape == pts.shape
    assert minus.shape == pts.shape
    assert torch.allclose(tangents, torch.tensor([[0.0, 1.0]]).double().expand(5, 2))
    assert torch.allclose(plus - pts, torch.tensor([0.0, 1.0]).double())
    assert torch.allclose(pts - minus, torch.tensor([0.0, 1.0]).double())
    # Midpoint of each gauge is the channel.
    assert torch.allclose((plus + minus) / 2, pts)


def test_fiber_gauge_locations_curved_fiber_tangents() -> None:
    """On a circular fibre, the tangent is perpendicular to the radius."""
    n = 50
    angles = torch.linspace(0, math.pi, n, dtype=torch.double)
    radius = 30.0
    center = torch.tensor([50.0, 50.0]).double()
    pts = center + radius * torch.stack(
        [torch.sin(angles), torch.cos(angles)],
        dim=1,
    )
    _, _, tangents = das.fiber_gauge_locations(pts, 10.0, 5.0)
    radial = (pts - center) / radius
    # Interior tangents (centred differences) are perpendicular to the radius.
    dot = (tangents[1:-1] * radial[1:-1]).sum(dim=1)
    assert dot.abs().max() < 1e-9
    assert torch.allclose(
        torch.linalg.vector_norm(tangents, dim=1),
        torch.ones(n).double(),
    )


def test_fiber_gauge_locations_anisotropic_spacing() -> None:
    """Anisotropic grid_spacing: tangents are physical, offsets in cells."""
    # Fibre along dim 0 (spacing 10 m/cell), gauge 10 m -> +/- 0.5 cells.
    pts = _fiber_points([10.0, 20.0], [1.0, 0.0], 3)
    plus, _minus, tangents = das.fiber_gauge_locations(pts, 10.0, [10.0, 5.0])
    assert torch.allclose(tangents[1], torch.tensor([1.0, 0.0]).double())
    assert torch.allclose(plus[1] - pts[1], torch.tensor([0.5, 0.0]).double())
    # A diagonal fibre in cells is not at 45 degrees physically.
    pts_diag = _fiber_points([10.0, 20.0], [1.0, 1.0], 3)
    _, _, t_diag = das.fiber_gauge_locations(pts_diag, 10.0, [10.0, 5.0])
    expected = torch.tensor([10.0, 5.0]).double()
    expected = expected / expected.norm()
    assert torch.allclose(t_diag[1], expected)


def test_fiber_gauge_locations_errors() -> None:
    """Invalid fibres/gauges raise clear errors."""
    good = _fiber_points([10.0, 20.0], [0.0, 1.0], 3)
    with pytest.raises(ValueError, match="at least 2 points"):
        das.fiber_gauge_locations(good[:1], 10.0, 5.0)
    with pytest.raises(ValueError, match="2-D"):
        das.fiber_gauge_locations(good.flatten(), 10.0, 5.0)
    with pytest.raises(ValueError, match="at least 2 spatial"):
        das.fiber_gauge_locations(good[:, :1], 10.0, 5.0)
    with pytest.raises(ValueError, match="gauge_length"):
        das.fiber_gauge_locations(good, 0.0, 5.0)
    with pytest.raises(ValueError, match="gauge_length"):
        das.fiber_gauge_locations(good, float("inf"), 5.0)
    with pytest.raises(ValueError, match="gauge_length"):
        das.fiber_gauge_locations(good, float("nan"), 5.0)
    with pytest.raises(TypeError, match="floating-point"):
        das.fiber_gauge_locations(good.long(), 10.0, 5.0)
    with pytest.raises(ValueError, match="coincident"):
        das.fiber_gauge_locations(
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
            10.0,
            5.0,
        )


def test_strain_rate_linear_field_exact() -> None:
    """For a linear velocity field the gauge difference is exact: t.G.t.

    v(x) = G x has a constant gradient, so the finite-gauge difference equals
    the pointwise strain rate exactly, for any gauge length and orientation.
    This validates the projection/difference algebra with no propagator.
    """
    torch.manual_seed(0)
    g_mat = torch.randn(2, 2).double()  # dv_i/dx_j (physical units)
    dx = 5.0
    gauge_length = 7.0
    pts = _fiber_points([50.0, 50.0], [1.0, 2.0], 4)
    plus, minus, tangents = das.fiber_gauge_locations(pts, gauge_length, dx)

    def v_at(locations):
        phys = locations * dx  # [n, 2] physical coordinates
        v = phys @ g_mat.T  # v_i = G_ij x_j
        # -> per-channel receiver tensors [shot=1, channel, time=2]
        return {
            name: v[:, i].reshape(1, -1, 1).repeat(1, 1, 2)
            for i, name in enumerate(("y", "x"))
        }

    sr = das.strain_rate_from_velocities(
        v_at(plus),
        v_at(minus),
        tangents,
        gauge_length,
    )
    expected = torch.einsum("ci,ij,cj->c", tangents, g_mat, tangents)
    assert torch.allclose(sr[0, :, 0], expected, rtol=1e-12, atol=1e-12)


def test_strain_rate_from_velocities_errors() -> None:
    """Mismatched shapes and bad gauge lengths raise clear errors."""
    tangents = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).double()
    v = torch.zeros(1, 2, 10).double()
    good = {"y": v, "x": v}
    with pytest.raises(ValueError, match="gauge_length"):
        das.strain_rate_from_velocities(good, good, tangents, -1.0)
    with pytest.raises(ValueError, match="gauge_length"):
        das.strain_rate_from_velocities(good, good, tangents, float("inf"))
    bad = torch.zeros(1, 3, 10).double()
    with pytest.raises(ValueError, match="channel count"):
        das.strain_rate_from_velocities({"y": v, "x": bad}, good, tangents, 10.0)
    bad_time = torch.zeros(1, 2, 3).double()
    with pytest.raises(ValueError, match="same \\[shot, channel, time\\] shape"):
        das.strain_rate_from_velocities({"y": v, "x": bad_time}, good, tangents, 10.0)


def test_strain_rate_from_velocities_channel_keys() -> None:
    """The mapping keys are the propagators' own channel names, per ndim."""
    tangents = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).double()
    v = torch.zeros(1, 2, 10).double()
    good = {"y": v, "x": v}

    # A missing key names both the keys given and the keys expected.
    with pytest.raises(DeepwaveValueError) as excinfo:
        das.strain_rate_from_velocities({"x": v}, good, tangents, 10.0)
    message = str(excinfo.value)
    assert "v_receivers_plus" in message
    assert "'y', 'x'" in message  # expected, in slowest-to-fastest order
    assert "'x'" in message  # given

    # So does an unexpected one (here the 3-D key, in a 2-D run).
    with pytest.raises(DeepwaveValueError, match="'z'"):
        das.strain_rate_from_velocities(good, {"z": v, "y": v, "x": v}, tangents, 10.0)

    # 3-D uses z, y, x, in that (slowest-to-fastest) order.
    tangents_3d = torch.tensor([[1.0, 0.0, 0.0]]).double()
    v3 = torch.zeros(1, 1, 4).double()
    with pytest.raises(DeepwaveValueError, match=r"'z', 'y', 'x'"):
        das.strain_rate_from_velocities(
            {"y": v3, "x": v3},
            {"y": v3, "x": v3},
            tangents_3d,
            10.0,
        )
    # The z column is the one that is selected by the "z" key: with a tangent
    # along z, only the "z" velocities reach the output.
    out = das.strain_rate_from_velocities(
        {"z": torch.ones(1, 1, 4).double(), "y": v3, "x": v3},
        {"z": v3, "y": torch.ones(1, 1, 4).double(), "x": v3},
        tangents_3d,
        10.0,
    )
    assert torch.allclose(out, torch.full((1, 1, 4), 0.1).double())

    # A tangents column count that is not 2 or 3 is rejected by name.
    with pytest.raises(DeepwaveValueError, match=r"tangents.shape\[-1\]=1"):
        das.strain_rate_from_velocities(
            {"x": v},
            {"x": v},
            torch.zeros(2, 1).double(),
            10.0,
        )


def test_strain_rate_from_velocities_rejects_sequence() -> None:
    """A sequence of tensors is refused with a message naming the new form."""
    tangents = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).double()
    v = torch.zeros(1, 2, 10).double()
    with pytest.raises(DeepwaveTypeError) as excinfo:
        das.strain_rate_from_velocities([v, v], {"y": v, "x": v}, tangents, 10.0)
    message = str(excinfo.value)
    assert "mapping" in message
    assert "'y': receiver_amplitudes_y" in message
    assert "list" in message


def test_directivity_follows_cos_squared() -> None:
    """DAS strain-rate amplitude follows the cos^2(theta) directivity law.

    An explosion source radiates a P wave whose particle velocity at the
    fibre (far field, k*r ~ 42) is along the propagation direction x-hat.
    Three short straight fibres centred on the same point, at 0/45/90 degrees
    to the propagation direction, measure amplitudes in the ratio
    1 : cos^2(45) = 0.5 : ~0 (broadside blindness). A small gauge (6 m,
    ~0.1 wavelengths) keeps the gauge-averaging (sinc) factor's angle
    dependence below 1%; wavefront curvature contributes O(1/(k r)) ~ 2%.

    Also checks the amplitude scale: for a plane P wave along the fibre,
    strain rate = -(1/c) dv_t/dt, so the theta=0 peak must match the time
    derivative of the recorded velocity divided by vp.
    """
    ny = nx = 240
    source_cell = (120, 60)
    center = [120.0, 140.0]  # 80 cells = 400 m from the source, along x
    nt = 440
    gauge_length = 6.0
    n_ch = 3  # use the middle channel (centred-difference tangent)

    fibres = {
        0.0: _fiber_points(center, [0.0, 1.0], n_ch),
        45.0: _fiber_points(center, [1.0, 1.0], n_ch),
        90.0: _fiber_points(center, [1.0, 0.0], n_ch),
    }
    plus_all, minus_all, tangents_all = [], [], []
    for pts in fibres.values():
        plus, minus, tangents = das.fiber_gauge_locations(pts, gauge_length, DX)
        plus_all.append(plus)
        minus_all.append(minus)
        tangents_all.append(tangents)

    # One propagation records every endpoint pair, plus a probe at the
    # common fibre centre for the amplitude-scale check.
    probe = torch.tensor([center]).double()
    locations = torch.cat([*plus_all, *minus_all, probe])
    vy, vx, _ = _run_elastic_p_source(ny, nx, source_cell, locations, nt)

    n = 3 * n_ch
    peaks = {}
    for i, (angle, _) in enumerate(fibres.items()):
        sl_p = slice(i * n_ch, (i + 1) * n_ch)
        sl_m = slice(n + i * n_ch, n + (i + 1) * n_ch)
        sr = das.strain_rate_from_velocities(
            {"y": vy[:, sl_p], "x": vx[:, sl_p]},
            {"y": vy[:, sl_m], "x": vx[:, sl_m]},
            tangents_all[i],
            gauge_length,
        )
        peaks[angle] = sr[0, n_ch // 2].abs().max().item()

    ratio_45 = peaks[45.0] / peaks[0.0]
    ratio_90 = peaks[90.0] / peaks[0.0]
    # cos^2(45 deg) = 0.5. Measured (CPU, float64): 0.5070.
    assert abs(ratio_45 - 0.5) < 0.03, f"45-degree ratio {ratio_45} != 0.5"
    # Broadside is blind up to wavefront curvature, O(1/(k r)) ~ 0.02.
    # Measured: 0.0194.
    assert ratio_90 < 0.06, f"broadside ratio {ratio_90} not ~ 0"

    # Amplitude scale: strain rate = -(1/vp) dv_t/dt for the theta=0 fibre.
    v_probe = vx[0, -1]
    dv_dt = (v_probe[2:] - v_probe[:-2]) / (2 * DT)
    expected_peak = dv_dt.abs().max().item() / VP
    # Measured relative difference: 0.028 (gauge sinc factor + curvature).
    assert abs(peaks[0.0] - expected_peak) / expected_peak < 0.1, (
        f"theta=0 peak {peaks[0.0]} vs (1/c)|dv/dt| peak {expected_peak}"
    )


def test_gauge_length_convergence() -> None:
    """Shrinking the gauge converges quadratically to the pointwise limit.

    The pointwise reference t.grad(v).t is computed independently from a very
    fine (1 m ~ lambda/60) finite difference of adjacent velocity receivers.
    Halving the gauge length must reduce the relative error by ~4x (the
    gauge average of exp(ikx) is sinc(k L/2), whose error is quadratic in L).
    """
    ny = nx = 240
    source_cell = (120, 60)
    center = [120.0, 140.0]
    nt = 440
    gauge_lengths = [24.0, 12.0, 6.0]

    pts = _fiber_points(center, [0.0, 1.0], 3)
    plus_all, minus_all, tangents = [], [], None
    for gauge_length in gauge_lengths:
        plus, minus, tangents = das.fiber_gauge_locations(pts, gauge_length, DX)
        plus_all.append(plus[1:2])
        minus_all.append(minus[1:2])
    tangent = tangents[1:2]

    # Independent fine-FD reference for the pointwise strain rate: velocity
    # receivers 1 m apart (0.1 cells each side of the channel), differenced.
    delta = 1.0  # m
    ref_plus = torch.tensor([[center[0], center[1] + delta / 2 / DX]]).double()
    ref_minus = torch.tensor([[center[0], center[1] - delta / 2 / DX]]).double()

    locations = torch.cat([*plus_all, *minus_all, ref_plus, ref_minus])
    vy, vx, _ = _run_elastic_p_source(ny, nx, source_cell, locations, nt)

    n = len(gauge_lengths)
    # The fibre is along x, so t.grad(v).t = dvx/dx; the reference difference
    # quotient uses the same tangential projection (tangent = x-hat).
    sr_ref = (vx[0, 2 * n] - vx[0, 2 * n + 1]) / delta

    errors = []
    for i, gauge_length in enumerate(gauge_lengths):
        sr = das.strain_rate_from_velocities(
            {"y": vy[:, i : i + 1], "x": vx[:, i : i + 1]},
            {"y": vy[:, n + i : n + i + 1], "x": vx[:, n + i : n + i + 1]},
            tangent,
            gauge_length,
        )[0, 0]
        errors.append(((sr - sr_ref).norm() / sr_ref.norm()).item())

    # Errors decrease monotonically and ~quadratically (ratio ~4 per halving).
    # Measured (CPU, float64): 0.472 / 0.140 / 0.040 (ratios 3.36, 3.48; a
    # touch below 4 because the Ricker is broadband, not monochromatic, and
    # the 24 m gauge is not yet asymptotic).
    assert errors[0] > errors[1] > errors[2], f"not monotone: {errors}"
    for i in range(1, len(errors)):
        ratio = errors[i - 1] / errors[i]
        assert 2.5 < ratio < 6.0, f"not ~quadratic: {errors} (ratio {ratio})"
    # The smallest gauge (6 m ~ 0.1 wavelengths) is within a few percent.
    assert errors[-1] < 0.05, f"6 m gauge error too large: {errors[-1]}"


def test_gradient_matches_directional_fd() -> None:
    """The DAS-loss gradient w.r.t. the elastic model matches a FD check.

    L2 loss on DAS strain-rate channels; gradients flow through the
    projection/difference to the velocity receivers and back through the
    propagator's adjoint to lamb/mu/buoyancy. Verified against a central
    directional finite difference in float64 on a tiny model (max_vel fixed
    so the internal time step does not change with the perturbation, and
    dt below the CFL limit so step_ratio == 1 and the model gradient is the
    exact discrete adjoint).
    """
    torch.manual_seed(1)
    ny = nx = 50
    source_cell = (12, 12)
    nt = 250
    gauge_length = 8.0

    pts = _fiber_points([30.0, 30.0], [1.0, 2.0], 4)
    plus, minus, tangents = das.fiber_gauge_locations(pts, gauge_length, DX)
    locations = torch.cat([plus, minus])
    n_ch = pts.shape[0]
    # Headroom above the perturbed models' vp, and identical in every run so
    # the internal time step is identical (the loss is then a smooth function
    # of the models and the FD check is meaningful).
    max_vel = VP * 1.05

    def loss_from(models_override=None, requires_grad=False):
        if models_override is None:
            vy, vx, models = _run_elastic_p_source(
                ny,
                nx,
                source_cell,
                locations,
                nt,
                requires_grad=requires_grad,
                max_vel=max_vel,
            )
        else:
            lamb, mu, buoyancy = models_override
            dtype = torch.double
            peak_time = 1.5 / FREQ
            source_amplitudes = ricker(FREQ, nt, DT, peak_time, dtype=dtype).reshape(
                1,
                1,
                -1,
            )
            out = elastic(
                lamb,
                mu,
                buoyancy,
                DX,
                DT,
                source_amplitudes_pressure_rate=source_amplitudes,
                source_locations_pressure_rate=torch.tensor([[list(source_cell)]]),
                receiver_locations_y=locations.unsqueeze(0),
                receiver_locations_x=locations.unsqueeze(0),
                pml_width=20,
                pml_freq=FREQ,
                max_vel=max_vel,
                accuracy=4,
            )
            vy, vx = out.receiver_amplitudes_y, out.receiver_amplitudes_x
            models = (lamb, mu, buoyancy)
        sr = das.strain_rate_from_velocities(
            {"y": vy[:, :n_ch], "x": vx[:, :n_ch]},
            {"y": vy[:, n_ch:], "x": vx[:, n_ch:]},
            tangents,
            gauge_length,
        )
        return (sr**2).sum(), models

    loss, models = loss_from(requires_grad=True)
    grads = torch.autograd.grad(loss, models)
    assert all(g.abs().max() > 0 for g in grads), "a model gradient is zero"

    # Directional FD: perturb all three models along a random relative
    # direction (0.1% of each parameter's scale per cell).
    directions = [torch.randn_like(m) * 1e-3 * m.detach().abs().mean() for m in models]
    directional = sum((g * d).sum() for g, d in zip(grads, directions))
    loss_p, _ = loss_from([m.detach() + d for m, d in zip(models, directions)])
    loss_m, _ = loss_from([m.detach() - d for m, d in zip(models, directions)])
    fd = (loss_p - loss_m) / 2
    rel_err = (directional - fd).abs().item() / fd.abs().item()
    # Measured (CPU, float64): 1.7e-5.
    assert rel_err < 1e-3, f"directional {directional} vs FD {fd} ({rel_err})"


def test_units_grid_spacing_invariance() -> None:
    """The same physical experiment gives the same physical strain rate.

    The medium, source position, fibre, gauge length (all physical) and dt
    are held fixed while the grid spacing doubles (5 m -> 10 m). The source
    is a cell delta with no 1/volume normalisation, so its amplitude is
    scaled by 1/cell-volume to represent the same physical source. The
    physical strain-rate trace must then agree within discretisation error.
    """
    freq = 10.0  # 15 points per P wavelength even on the 10 m grid
    dt = 0.002
    nt = 320
    gauge_length = 20.0
    source_phys = (600.0, 250.0)  # m
    center_phys = [600.0, 650.0]  # m; 400 m from the source
    domain = 1200.0  # m

    peaks = {}
    for dx in (5.0, 10.0):
        n = int(domain / dx)
        pts = _fiber_points(
            [c / dx for c in center_phys],
            [1.0, 2.0],
            3,
            spacing_cells=5.0 / dx,
        )
        plus, minus, tangents = das.fiber_gauge_locations(pts, gauge_length, dx)
        locations = torch.cat([plus, minus])
        vy, vx, _ = _run_elastic_p_source(
            n,
            n,
            (int(source_phys[0] / dx), int(source_phys[1] / dx)),
            locations,
            nt,
            dx=dx,
            dt=dt,
            freq=freq,
            amplitude=(5.0 / dx) ** 2,  # same physical (volume-integrated) source
        )
        sr = das.strain_rate_from_velocities(
            {"y": vy[:, :3], "x": vx[:, :3]},
            {"y": vy[:, 3:], "x": vx[:, 3:]},
            tangents,
            gauge_length,
        )
        peaks[dx] = sr[0, 1]

    trace_5, trace_10 = peaks[5.0], peaks[10.0]
    # Measured (CPU, float64): peak ratio 1.0010.
    peak_ratio = trace_10.abs().max().item() / trace_5.abs().max().item()
    assert abs(peak_ratio - 1) < 0.1, f"peak ratio {peak_ratio} != 1"
    # Arrival (peak) times agree to within a few samples (measured: equal).
    t5 = trace_5.abs().argmax().item()
    t10 = trace_10.abs().argmax().item()
    assert abs(t5 - t10) <= 5, f"peak times differ: {t5} vs {t10} samples"
    # The waveforms agree overall (dominated by dispersion on the 10 m grid).
    # Measured relative L2 difference: 0.081.
    rel = ((trace_10 - trace_5).norm() / trace_5.norm()).item()
    assert rel < 0.2, f"waveform mismatch {rel}"


# --------------------------------------------------------------------------
# gauge_average
# --------------------------------------------------------------------------


def test_gauge_average_linear_field_odd_window_exact() -> None:
    """Odd window: the moving average is an exactly centred mean.

    For a field linear in channel index, the mean over a symmetric window
    equals the value at the centre channel exactly (away from the array
    edges, where replicate padding breaks linearity).
    """
    data = (2.0 + 3.0 * torch.arange(30.0)).double().reshape(1, 30, 1)
    gauge_length, channel_spacing = 15.0, 5.0  # window = round(15/5) = 3 (odd)
    out = das.gauge_average(data, gauge_length, channel_spacing, dim=1)
    left = 1  # (3 - 1) // 2
    interior = slice(left, 30 - left)
    assert torch.allclose(out[0, interior, 0], data[0, interior, 0])


def test_gauge_average_linear_field_even_window_half_channel_offset() -> None:
    """Even window: the documented -0.5-channel convention holds exactly.

    ``left = window // 2, right = window - 1 - left`` puts one more channel
    on the lower-index side, so the reported average at channel ``i`` equals
    the linear field evaluated at ``i - 0.5`` (interior channels only).
    """
    data = (2.0 + 3.0 * torch.arange(30.0)).double().reshape(1, 30, 1)
    gauge_length, channel_spacing = 20.0, 5.0  # window = round(20/5) = 4 (even)
    out = das.gauge_average(data, gauge_length, channel_spacing, dim=1)
    left, right = 2, 1  # window // 2, window - 1 - left
    interior = slice(left, 30 - right)
    expected = 2.0 + 3.0 * (torch.arange(30.0).double() - 0.5)
    assert torch.allclose(out[0, interior, 0], expected[interior])


def test_gauge_average_hand_worked_example() -> None:
    """A small hand-computed example pins down both conventions exactly."""
    data = torch.arange(6.0).double()  # [0, 1, 2, 3, 4, 5]
    # window = 3 (odd): centred mean, edges replicate-padded.
    out_odd = das.gauge_average(data, 3.0, 1.0, dim=0)
    expected_odd = torch.tensor([1 / 3, 1.0, 2.0, 3.0, 4.0, 14 / 3]).double()
    assert torch.allclose(out_odd, expected_odd)
    # window = 2 (even): left=1, right=0 -> mean of [i-1, i].
    out_even = das.gauge_average(data, 2.0, 1.0, dim=0)
    expected_even = torch.tensor([0.0, 0.5, 1.5, 2.5, 3.5, 4.5]).double()
    assert torch.allclose(out_even, expected_even)


def test_gauge_average_window_one_is_noop() -> None:
    """A gauge shorter than one channel spacing rounds to a window of 1."""
    data = torch.randn(2, 10, 4).double()
    out = das.gauge_average(data, gauge_length=1.0, channel_spacing=10.0, dim=1)
    assert torch.equal(out, data)


def test_gauge_average_negative_dim_and_shape_preserved() -> None:
    """``dim`` accepts negative indices and the output shape matches input."""
    data = torch.randn(2, 3, 17, 4).double()
    out_pos = das.gauge_average(data, 12.0, 4.0, dim=2)
    out_neg = das.gauge_average(data, 12.0, 4.0, dim=-2)
    assert out_pos.shape == data.shape
    assert torch.equal(out_pos, out_neg)


def test_gauge_average_errors() -> None:
    """Invalid gauge_length/channel_spacing/dim raise clear errors."""
    data = torch.randn(1, 10, 1).double()
    with pytest.raises(ValueError, match="gauge_length"):
        das.gauge_average(data, 0.0, 5.0, dim=1)
    with pytest.raises(ValueError, match="channel_spacing"):
        das.gauge_average(data, 10.0, 0.0, dim=1)
    with pytest.raises(ValueError, match="dim"):
        das.gauge_average(data, 10.0, 5.0, dim=5)


def test_gauge_average_differentiable() -> None:
    """Gradients flow through gauge_average (it is a plain linear op)."""
    data = torch.randn(1, 12, 1).double().requires_grad_()
    out = das.gauge_average(data, 15.0, 5.0, dim=1)
    loss = (out**2).sum()
    (grad,) = torch.autograd.grad(loss, data)
    assert grad.abs().max() > 0


# --------------------------------------------------------------------------
# helical_fiber_response
# --------------------------------------------------------------------------


def _perp_tangent(tangent: torch.Tensor) -> torch.Tensor:
    """90-degree rotation of a physical unit tangent, shape [..., 2]."""
    return torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)


def test_helical_fiber_response_alpha_zero_matches_axial() -> None:
    """wrap_angle=0 reduces exactly to the existing axial response."""
    torch.manual_seed(2)
    axial = torch.randn(1, 5, 7).double()
    transverse = [torch.randn(1, 5, 7).double()]
    out = das.helical_fiber_response(axial, transverse, wrap_angle=0.0)
    assert torch.equal(out, axial)


def test_helical_fiber_response_ninety_degrees_is_transverse_average() -> None:
    """wrap_angle=pi/2 (purely circumferential) ignores the axial term.

    cos^2(pi/2) = 0, sin^2(pi/2)/2 = 1/2, so the response is the mean of the
    two transverse normal strain rates (the axial value is irrelevant).
    """
    axial = torch.randn(1, 4, 3).double()
    t0 = torch.randn(1, 4, 3).double()
    t1 = torch.randn(1, 4, 3).double()
    out = das.helical_fiber_response(axial, [t0, t1], wrap_angle=math.pi / 2)
    assert torch.allclose(out, (t0 + t1) / 2)


def test_helical_fiber_response_magic_angle_equal_thirds() -> None:
    """At the magic angle the three cable-frame weights are each exactly 1/3.

    arccos(1/sqrt(3)) ~= 0.9553 radians. Response = mean of the axial and
    both transverse normal strain rates -- sensitivity to the strain
    tensor's trace only (volumetric strain).
    """
    axial = torch.tensor(3.0).double()
    transverse = [torch.tensor(6.0).double(), torch.tensor(9.0).double()]
    out = das.helical_fiber_response(axial, transverse, wrap_angle=MAGIC_ANGLE)
    assert torch.allclose(out, torch.tensor((3.0 + 6.0 + 9.0) / 3.0).double())


def test_helical_fiber_response_weights_sum_to_one() -> None:
    """cos^2(alpha) + sin^2(alpha)/2 + sin^2(alpha)/2 == 1 for any alpha.

    Feeding identical unit values for the axial and both transverse
    components must return exactly 1, for any wrap angle.
    """
    for wrap_angle in (0.0, 0.3, math.pi / 6, MAGIC_ANGLE, 1.1, math.pi / 2, 2.15):
        axial = torch.tensor(1.0).double()
        transverse = [torch.tensor(1.0).double(), torch.tensor(1.0).double()]
        out = das.helical_fiber_response(axial, transverse, wrap_angle=wrap_angle)
        assert torch.allclose(out, torch.tensor(1.0).double(), atol=1e-12)


def test_helical_fiber_response_errors() -> None:
    """Wrong sequence lengths / mismatched shapes raise clear errors."""
    axial = torch.zeros(1, 3, 5)
    with pytest.raises(ValueError, match=r"1 .* or.*2"):
        das.helical_fiber_response(axial, [], wrap_angle=math.pi / 6)
    with pytest.raises(ValueError, match=r"1 .* or.*2"):
        das.helical_fiber_response(axial, [axial, axial, axial], wrap_angle=0.5)
    with pytest.raises(ValueError, match="same shape"):
        das.helical_fiber_response(axial, [torch.zeros(1, 4, 5)], wrap_angle=0.5)


def test_helical_fiber_response_wrap_angle_is_keyword_only_radians() -> None:
    """The old degrees spelling fails loudly, as does a positional angle.

    ``wrap_angle`` is keyword-only precisely so that the change of unit
    (degrees -> radians) cannot happen silently: both the old keyword and a
    bare positional angle raise ``TypeError``.
    """
    axial = torch.zeros(1, 3, 5)
    transverse = [torch.zeros(1, 3, 5)]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        das.helical_fiber_response(axial, transverse, wrap_angle_deg=40.0)
    with pytest.raises(TypeError, match="positional argument"):
        das.helical_fiber_response(axial, transverse, 40.0)


def test_helical_fiber_response_differentiable() -> None:
    """Gradients flow to both the axial and transverse inputs.

    Since the response is a plain weighted sum, the gradient with respect
    to each input must equal its (constant) weight exactly.
    """
    wrap_angle = math.radians(40.0)  # radians, as the function requires
    axial = torch.randn(1, 3, 2).double().requires_grad_()
    t0 = torch.randn(1, 3, 2).double().requires_grad_()
    t1 = torch.randn(1, 3, 2).double().requires_grad_()
    out = das.helical_fiber_response(axial, [t0, t1], wrap_angle=wrap_angle)
    loss = out.sum()
    grad_axial, grad_t0, grad_t1 = torch.autograd.grad(loss, [axial, t0, t1])

    weight_axial = math.cos(wrap_angle) ** 2
    weight_transverse = math.sin(wrap_angle) ** 2 / 2
    assert torch.allclose(
        grad_axial,
        torch.full_like(grad_axial, weight_axial),
    )
    assert torch.allclose(
        grad_t0,
        torch.full_like(grad_t0, weight_transverse),
    )
    assert torch.allclose(
        grad_t1,
        torch.full_like(grad_t1, weight_transverse),
    )


def test_helical_fiber_response_plane_wave_matches_analytic() -> None:
    """A helically wound fibre's response matches the plane-wave prediction.

    For a plane P wave, the normal strain rate along any direction n-hat is
    proportional to (n-hat . prop-direction-hat)^2 with a shared amplitude
    factor -(A/c) g'(.) (see the module docstring's directivity derivation).
    A straight fibre at 45 degrees to the propagation direction (x-hat) has
    axial weight cos^2(45) = 0.5 and (in-plane) transverse weight
    sin^2(45) = 0.5, so the helical combination at wrap angle alpha must
    match weight_axial(alpha) * 0.5 + weight_transverse(alpha) * 0.5,
    referenced against the theta=0 fibre's peak (which measures the shared
    amplitude factor directly, weight 1).
    """
    ny = nx = 240
    source_cell = (120, 60)
    center = [120.0, 140.0]  # 400 m from the source, along x (prop-direction)
    nt = 440
    gauge_length = 6.0  # small: keeps the gauge-sinc angle dependence < 1%
    n_ch = 3

    pts0 = _fiber_points(center, [0.0, 1.0], n_ch)  # theta=0 (aligned, x-hat)
    pts45 = _fiber_points(center, [1.0, 1.0], n_ch)  # theta=45 degrees

    plus0, minus0, tangents0 = das.fiber_gauge_locations(pts0, gauge_length, DX)
    plus45, minus45, tangents45 = das.fiber_gauge_locations(pts45, gauge_length, DX)
    perp45 = _perp_tangent(tangents45)
    half_offset = perp45 * (gauge_length / 2) / DX
    perp_plus45 = pts45 + half_offset
    perp_minus45 = pts45 - half_offset

    locations = torch.cat(
        [plus0, minus0, plus45, minus45, perp_plus45, perp_minus45],
    )
    vy, vx, _ = _run_elastic_p_source(ny, nx, source_cell, locations, nt)

    n = n_ch

    def sl(i: int) -> slice:
        return slice(i * n, (i + 1) * n)

    axial0 = das.strain_rate_from_velocities(
        {"y": vy[:, sl(0)], "x": vx[:, sl(0)]},
        {"y": vy[:, sl(1)], "x": vx[:, sl(1)]},
        tangents0,
        gauge_length,
    )
    axial45 = das.strain_rate_from_velocities(
        {"y": vy[:, sl(2)], "x": vx[:, sl(2)]},
        {"y": vy[:, sl(3)], "x": vx[:, sl(3)]},
        tangents45,
        gauge_length,
    )
    transverse45 = das.strain_rate_from_velocities(
        {"y": vy[:, sl(4)], "x": vx[:, sl(4)]},
        {"y": vy[:, sl(5)], "x": vx[:, sl(5)]},
        perp45,
        gauge_length,
    )

    mid = n_ch // 2  # centred-difference channel, as in the directivity test
    peak_axial0 = axial0[0, mid].abs().max().item()

    for wrap_angle in (0.0, math.pi / 6, MAGIC_ANGLE, math.pi / 2):
        helical45 = das.helical_fiber_response(
            axial45[:, mid : mid + 1],
            [transverse45[:, mid : mid + 1]],
            wrap_angle=wrap_angle,
        )
        measured_ratio = helical45.abs().max().item() / peak_axial0

        weight_axial = math.cos(wrap_angle) ** 2
        weight_transverse = math.sin(wrap_angle) ** 2 / 2
        predicted_ratio = weight_axial * 0.5 + weight_transverse * 0.5

        assert abs(measured_ratio - predicted_ratio) < 0.06, (
            f"wrap_angle={wrap_angle}: measured {measured_ratio} "
            f"vs predicted {predicted_ratio}"
        )

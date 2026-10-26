"""Tests for deepwave.transducers (ultrasound/NDT array helpers)."""

import math

import pytest
import torch

from deepwave import DeepwaveValueError, scalar
from deepwave import transducers as td
from deepwave.wavelets import ricker, tone_burst


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_units_and_geometry_rejected(value):
    """Invalid units cannot silently erase delays or produce NaN apertures."""
    with pytest.raises(DeepwaveValueError, match="finite"):
        td.apply_delays(torch.ones(8), torch.tensor([0.1]), value)
    with pytest.raises(DeepwaveValueError, match="finite"):
        td.delays_to_focus(torch.zeros(1, 2), [1.0, 1.0], value)
    with pytest.raises(DeepwaveValueError, match="finite"):
        td.line_element([0.0, 0.0], [1.0, 0.0], 1.0, spacing=value)
    with pytest.raises(DeepwaveValueError, match="finite"):
        td.line_element([value, 0.0], [1.0, 0.0], 1.0)


def test_line_element_endpoints_symmetric_2d() -> None:
    """Line-element endpoints are symmetric about the centre; spacing obeyed."""
    centre = [50.0, 10.0]
    loc, wts = td.line_element(
        centre,
        direction=[0.0, 1.0],
        length=20.0,
        spacing=1.0,
        dtype=torch.float64,
    )
    n = loc.shape[0]
    assert loc.shape == (n, 2)
    assert wts.shape == (n,)
    # Endpoints symmetric about centre.
    mid = (loc[0] + loc[-1]) / 2
    assert torch.allclose(mid, torch.tensor(centre, dtype=torch.float64))
    # Endpoints span the requested length.
    assert torch.isclose(
        torch.linalg.vector_norm(loc[-1] - loc[0]),
        torch.tensor(20.0, dtype=torch.float64),
    )
    # Consecutive spacing <= requested spacing.
    step = torch.linalg.vector_norm(loc[1:] - loc[:-1], dim=1)
    assert (step <= 1.0 + 1e-9).all()


def test_line_element_3d_direction_normalised() -> None:
    """A 3-D line element lies along the (unnormalised) direction."""
    loc, _ = td.line_element(
        [10.0, 10.0, 10.0],
        direction=[0.0, 3.0, 4.0],  # length 5, not unit
        length=10.0,
        spacing=2.0,
        dtype=torch.float64,
    )
    # All points collinear with the unit direction [0, 0.6, 0.8].
    unit = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
    disp = loc - torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    # Each displacement is a scalar multiple of unit.
    for d in disp:
        s = d @ unit
        assert torch.allclose(d, s * unit, atol=1e-9)


def test_arc_element_on_circle() -> None:
    """All arc-element points lie at the requested radius from the centre."""
    centre = torch.tensor([50.0, 50.0], dtype=torch.float64)
    loc, _ = td.arc_element(
        centre,
        radius=30.0,
        angles=(-0.6, 0.6),
        spacing=1.0,
        dtype=torch.float64,
    )
    r = torch.linalg.vector_norm(loc - centre, dim=1)
    assert torch.allclose(r, torch.full_like(r, 30.0), atol=1e-9)
    # Arc spacing obeyed.
    step = torch.linalg.vector_norm(loc[1:] - loc[:-1], dim=1)
    assert (step <= 1.0 + 1e-6).all()


def test_arc_element_full_circle_drops_seam() -> None:
    """A full 2*pi arc drops the duplicated seam point (no repeated cell)."""
    centre = torch.tensor([50.0, 50.0], dtype=torch.float64)
    loc, wts = td.arc_element(
        centre,
        radius=10.0,
        angles=(0.0, 2 * math.pi),
        spacing=1.0,
        dtype=torch.float64,
    )
    assert loc.shape[0] == wts.shape[0]
    # First and last points are distinct (the closing seam was removed).
    assert not torch.allclose(loc[0], loc[-1], atol=1e-9)
    # No two points coincide.
    dists = torch.cdist(loc, loc)
    dists.fill_diagonal_(float("inf"))
    assert float(dists.min()) > 1e-6


def test_line_element_negative_length_raises() -> None:
    """line_element rejects a negative length."""
    centre = torch.tensor([10.0, 10.0], dtype=torch.float64)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="length must be non-negative"):
        td.line_element(centre, direction, length=-5.0, dtype=torch.float64)


def test_element_builders_thread_centre_device() -> None:
    """Element builders place their tensors on the centre argument's device.

    On a CPU-only box this at least confirms the device is threaded through
    (no hard-coded CPU construction that would mismatch a CUDA centre); on the
    maintainer's GPU it exercises the CUDA path.
    """
    dev = torch.device("cpu")
    centre2 = torch.tensor([5.0, 5.0], dtype=torch.float64, device=dev)
    loc, wts = td.line_element(
        centre2, torch.tensor([1.0, 0.0]), length=6.0, dtype=torch.float64
    )
    assert loc.device == dev
    assert wts.device == dev
    loc_a, wts_a = td.arc_element(centre2, 8.0, (-0.5, 0.5), dtype=torch.float64)
    assert loc_a.device == dev
    assert wts_a.device == dev


def test_disc_element_in_plane_and_within_radius() -> None:
    """Disc points lie in the normal plane and within the disc radius."""
    centre = torch.tensor([20.0, 20.0, 20.0], dtype=torch.float64)
    normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    loc, wts = td.disc_element(
        centre,
        radius=8.0,
        normal=normal,
        spacing=1.5,
        dtype=torch.float64,
    )
    disp = loc - centre
    # In-plane: displacement perpendicular to normal.
    assert torch.allclose(
        disp @ normal, torch.zeros(loc.shape[0], dtype=torch.float64), atol=1e-9
    )
    # Within radius.
    r = torch.linalg.vector_norm(disp, dim=1)
    assert (r <= 8.0 + 1e-9).all()
    assert wts.shape == (loc.shape[0],)


def test_bowl_element_on_sphere() -> None:
    """Bowl points lie on the sphere of radius R about the centre of curvature."""
    centre = torch.tensor([10.0, 20.0, 20.0], dtype=torch.float64)
    focus = torch.tensor([40.0, 20.0, 20.0], dtype=torch.float64)  # dist 30 = R
    radius = 30.0
    loc, _ = td.bowl_element(
        centre,
        radius=radius,
        diameter=30.0,
        focus=focus,
        spacing=1.5,
        dtype=torch.float64,
    )
    r = torch.linalg.vector_norm(loc - focus, dim=1)
    assert torch.allclose(r, torch.full_like(r, radius), atol=1e-6)
    # The apex (first point) is on the axis at the centre.
    assert torch.allclose(loc[0], centre, atol=1e-6)


def test_bowl_invalid_diameter() -> None:
    """Aperture larger than the sphere is rejected."""
    with pytest.raises(ValueError, match="diameter"):
        td.bowl_element(
            [0.0, 0.0, 0.0],
            radius=10.0,
            diameter=30.0,
            focus=[10.0, 0.0, 0.0],
        )


def test_window_single_point_and_hamming_blackman() -> None:
    """n=1 short-circuits to a single weight; hamming/blackman formulas evaluate."""
    single = td.window_weights(1, "hann", dtype=torch.float64)
    assert torch.equal(single, torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="n must be at least 1"):
        td.window_weights(0, "hann")

    n = 21
    hamming = td.window_weights(n, "hamming", dtype=torch.float64)
    blackman = td.window_weights(n, "blackman", dtype=torch.float64)
    for w in (hamming, blackman):
        assert w.shape == (n,)
        assert torch.allclose(w, w.flip(0), atol=1e-12)  # symmetric
        assert torch.isclose(w.max(), w[n // 2])  # peak at centre
    # Hamming doesn't taper fully to zero at the ends (unlike Hann); Blackman
    # does.
    assert hamming[0] > 1e-3
    assert blackman[0].abs() < 1e-3


def test_line_element_zero_length_gives_single_point() -> None:
    """A zero length collapses the segment to one point at the centre."""
    centre = [3.0, 4.0]
    loc, wts = td.line_element(
        centre, direction=[1.0, 0.0], length=0.0, dtype=torch.float64
    )
    assert loc.shape == (1, 2)
    assert wts.shape == (1,)
    assert torch.allclose(loc[0], torch.tensor(centre, dtype=torch.float64))


def test_line_element_non_positive_spacing_raises() -> None:
    """Non-positive spacing is rejected, not silently divided by."""
    with pytest.raises(ValueError, match="spacing must be positive"):
        td.line_element([0.0, 0.0], [1.0, 0.0], length=5.0, spacing=0.0)


def test_line_element_shape_mismatch_raises() -> None:
    """Centre and direction must be 1-D of the same length."""
    with pytest.raises(ValueError, match="centre and direction must be 1-D"):
        td.line_element([0.0, 0.0], [1.0, 0.0, 0.0], length=5.0)


def test_line_element_zero_direction_raises() -> None:
    """A zero direction vector is undefined and must raise."""
    with pytest.raises(ValueError, match="direction must be non-zero"):
        td.line_element([0.0, 0.0], [0.0, 0.0], length=5.0)


def test_arc_element_validation() -> None:
    """arc_element rejects mis-shaped centres, bad angle pairs, and radius<=0."""
    with pytest.raises(ValueError, match="centre_of_curvature must have shape"):
        td.arc_element([0.0, 0.0, 0.0], radius=5.0, angles=(-0.5, 0.5))
    with pytest.raises(ValueError, match="angles must be"):
        td.arc_element([0.0, 0.0], radius=5.0, angles=(-0.5, 0.5, 0.1))
    with pytest.raises(ValueError, match="radius must be positive"):
        td.arc_element([0.0, 0.0], radius=0.0, angles=(-0.5, 0.5))


def test_disc_element_validation() -> None:
    """disc_element rejects mis-shaped centre/normal and radius<=0."""
    with pytest.raises(ValueError, match="disc_element is 3-D"):
        td.disc_element([0.0, 0.0], radius=5.0, normal=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="radius must be positive"):
        td.disc_element([0.0, 0.0, 0.0], radius=0.0, normal=[1.0, 0.0, 0.0])


def test_bowl_element_validation() -> None:
    """bowl_element rejects mis-shaped apex/focus and radius<=0."""
    with pytest.raises(ValueError, match="bowl_element is 3-D"):
        td.bowl_element([0.0, 0.0], radius=10.0, diameter=5.0, focus=[10.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="radius must be positive"):
        td.bowl_element(
            [0.0, 0.0, 0.0], radius=0.0, diameter=5.0, focus=[10.0, 0.0, 0.0]
        )


def test_radial_window_hann_hamming_blackman() -> None:
    """Disc radial taper (hann/hamming/blackman): 1 at centre, ~0/0.08 at rim."""
    for kind, rim_expected in (("hann", 0.0), ("hamming", 0.08), ("blackman", 0.0)):
        loc, wts = td.disc_element(
            [0.0, 0.0, 0.0],
            radius=10.0,
            normal=[0.0, 0.0, 1.0],
            spacing=1.0,
            window=kind,
            dtype=torch.float64,
        )
        assert wts.shape == (loc.shape[0],)
        centre_idx = torch.argmin(torch.linalg.vector_norm(loc, dim=1))
        assert torch.isclose(
            wts[centre_idx], torch.tensor(1.0, dtype=torch.float64), atol=1e-6
        )
        rim_idx = torch.argmax(torch.linalg.vector_norm(loc, dim=1))
        assert wts[rim_idx] == pytest.approx(rim_expected, abs=1e-2)
    with pytest.raises(ValueError, match="Unknown window"):
        td.disc_element(
            [0.0, 0.0, 0.0],
            radius=5.0,
            normal=[0.0, 0.0, 1.0],
            window="nosuchwindow",
        )


def test_delays_to_focus_non_positive_c_raises() -> None:
    """A non-positive wave speed is rejected, not silently inverted."""
    locations = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="c must be positive"):
        td.delays_to_focus(locations, focus=[1.0, 1.0], c=0.0)
    with pytest.raises(ValueError, match="c must be positive"):
        td.delays_to_focus(locations, focus=[1.0, 1.0], c=-1500.0)


def test_apply_delays_wrong_ndim_raises() -> None:
    """apply_delays rejects a non-1-D wavelet."""
    with pytest.raises(ValueError, match="wavelet must be 1-D"):
        td.apply_delays(torch.zeros(2, 10), torch.zeros(2), dt=1e-8)


def test_apply_delays_rejects_non_vector_delays() -> None:
    """Delays follow the documented one-delay-per-point [n] contract."""
    wavelet = torch.zeros(8)
    with pytest.raises(ValueError, match=r"delays must be 1-D of shape \[n\]"):
        td.apply_delays(wavelet, torch.zeros(2, 1), dt=1.0)


def test_apply_delays_nt_pads_and_truncates() -> None:
    """A different ``nt`` pads (nt > len) or truncates (nt < len) before shifting."""
    dt = 1e-8
    nt0 = 100
    wavelet = ricker(1e6, nt0, dt, peak_time=nt0 * dt / 3, dtype=torch.float64)
    delays = torch.zeros(1, dtype=torch.float64)

    padded = td.apply_delays(wavelet, delays, dt, nt=150)
    assert padded.shape == (1, 150)
    truncated = td.apply_delays(wavelet, delays, dt, nt=50)
    assert truncated.shape == (1, 50)
    # Zero delay: padding/truncating then shifting-by-zero matches padding/
    # truncating the original wavelet directly (no phase-ramp effect at
    # delay=0).
    manual_pad = torch.nn.functional.pad(wavelet, (0, 50))
    assert torch.allclose(padded[0], manual_pad, atol=1e-10)
    assert torch.allclose(truncated[0], wavelet[:50], atol=1e-10)


def test_element_receiver_trace_validation() -> None:
    """element_receiver_trace rejects non-1-D weights and a receiver-count mismatch."""
    ra = torch.randn(2, 5, 100)
    with pytest.raises(ValueError, match="weights must be 1-D"):
        td.element_receiver_trace(ra, torch.randn(5, 1))
    with pytest.raises(ValueError, match="must equal the number of weights"):
        td.element_receiver_trace(ra, torch.randn(3))


def test_disc_element_zero_normal_raises() -> None:
    """A zero disc normal is undefined and must raise, not silently NaN."""
    with pytest.raises(ValueError, match="normal must be non-zero"):
        td.disc_element([0.0, 0.0, 0.0], radius=5.0, normal=[0.0, 0.0, 0.0])


def test_bowl_element_focus_equals_apex_raises() -> None:
    """A bowl axis needs focus != apex; must raise, not silently NaN."""
    with pytest.raises(ValueError, match="focus must differ from apex"):
        td.bowl_element(
            [1.0, 2.0, 3.0],
            radius=10.0,
            diameter=5.0,
            focus=[1.0, 2.0, 3.0],
        )


def test_delays_to_focus_mismatched_focus_shape_raises() -> None:
    """A mis-shaped focus must raise, not silently broadcast."""
    locations = torch.zeros(3, 2)
    with pytest.raises(ValueError, match=r"focus must have shape \[2\]"):
        td.delays_to_focus(locations, focus=[1.0], c=1500.0)
    with pytest.raises(ValueError, match="locations must have shape"):
        td.delays_to_focus(torch.zeros(2, 3, 2), focus=[1.0, 2.0], c=1500.0)


def test_apply_delays_non_positive_dt_raises() -> None:
    """A non-positive dt must raise, not silently reverse the shift direction."""
    wavelet = torch.zeros(10)
    delays = torch.zeros(3)
    with pytest.raises(ValueError, match="dt must be positive"):
        td.apply_delays(wavelet, delays, dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        td.apply_delays(wavelet, delays, dt=-0.001)


def test_window_windows() -> None:
    """Window shapes/values: rect is all ones, Hann tapers to ~0 at ends."""
    n = 21
    rect = td.window_weights(n, "rect", dtype=torch.float64)
    assert rect.shape == (n,)
    assert torch.allclose(rect, torch.ones(n, dtype=torch.float64))
    hann = td.window_weights(n, "hann", dtype=torch.float64)
    assert hann[0].abs() < 1e-12
    assert hann[-1].abs() < 1e-12
    assert torch.isclose(hann.max(), torch.tensor(1.0, dtype=torch.float64))
    # Symmetric window.
    assert torch.allclose(hann, hann.flip(0), atol=1e-12)
    with pytest.raises(ValueError, match="Unknown window"):
        td.window_weights(n, "nosuchwindow")
    with pytest.raises(ValueError, match="Unknown window"):
        td.window_weights(1, "nosuchwindow")


def test_delays_to_focus_geometry() -> None:
    """delays_to_focus matches the hand-computed travel-time differences.

    ``c`` is in grid cells per unit time, so the same physical setup (water at
    1500 m/s on 0.5 mm cells) is expressed as ``c = 1500 / 0.5e-3`` cells per
    second and the distances stay in cells.
    """
    # Two points; focus equidistant vertically offset.
    loc = torch.tensor([[0.0, 0.0], [0.0, 10.0]], dtype=torch.float64)
    focus = torch.tensor([20.0, 5.0], dtype=torch.float64)
    c = 1500.0 / 0.5e-3  # 1500 m/s on 0.5 mm cells -> 3e6 cells/s
    delays = td.delays_to_focus(loc, focus, c)
    # Both points are symmetric about the focus in x, equal distance -> equal
    # (zero) delay.
    assert torch.allclose(delays, torch.zeros(2, dtype=torch.float64), atol=1e-15)

    # Asymmetric: point directly under focus is nearer -> larger delay.
    loc2 = torch.tensor([[0.0, 5.0], [0.0, 25.0]], dtype=torch.float64)
    d2 = td.delays_to_focus(loc2, focus, c)
    dist0 = math.hypot(20.0, 0.0)  # cells
    dist1 = math.hypot(20.0, 20.0)  # cells
    expected = torch.tensor(
        [(max(dist0, dist1) - dist0) / c, (max(dist0, dist1) - dist1) / c],
        dtype=torch.float64,
    )
    assert torch.allclose(d2, expected, atol=1e-15)
    assert (d2 >= 0).all()


def test_delays_to_focus_c_is_cells_per_unit_time() -> None:
    """``c`` converts cell distances directly; ``grid_spacing`` is gone.

    Delays scale as ``1 / c``, and the same physical problem written in two
    unit systems (mm/us and SI) gives the same delays once each is expressed
    in its own time unit.
    """
    loc = torch.tensor([[0.0, 0.0], [0.0, 30.0]], dtype=torch.float64)
    focus = torch.tensor([40.0, 0.0], dtype=torch.float64)

    # Straight cells-per-unit-time arithmetic: 40 and 50 cells from the focus.
    delays = td.delays_to_focus(loc, focus, c=10.0)
    assert torch.allclose(
        delays,
        torch.tensor([(50.0 - 40.0) / 10.0, 0.0], dtype=torch.float64),
    )
    # Halving c doubles every delay.
    assert torch.allclose(td.delays_to_focus(loc, focus, c=5.0), 2 * delays)

    # Water at 1500 m/s on 0.1 mm cells, in mm/us (delays in us) and in SI
    # (delays in s): the same physics, differing only by the 1e6 unit factor.
    d_mm_us = td.delays_to_focus(loc, focus, c=1.5 / 0.1)
    d_si = td.delays_to_focus(loc, focus, c=1500.0 / 1e-4)
    assert torch.allclose(d_mm_us, d_si * 1e6)

    # The removed parameter fails loudly, by keyword and positionally.
    with pytest.raises(TypeError, match="grid_spacing"):
        td.delays_to_focus(loc, focus, c=10.0, grid_spacing=1e-4)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="positional argument"):
        td.delays_to_focus(loc, focus, 10.0, 1e-4)  # type: ignore[call-arg]


def test_apply_delays_fractional_shift_exact() -> None:
    """FFT fractional-delay shifts a ricker by exactly the requested amount."""
    dt = 1e-8
    nt = 600
    f = 1e6
    r = ricker(f, nt, dt, peak_time=nt * dt / 3, dtype=torch.float64)
    shift = 37.4 * dt  # sub-sample (fractional) delay
    shifted = td.apply_delays(r, torch.tensor([shift], dtype=torch.float64), dt, nt)[0]
    # Analytic ground truth: same ricker with a shifted peak time.
    ref = ricker(f, nt, dt, peak_time=nt * dt / 3 + shift, dtype=torch.float64)
    assert (shifted - ref).abs().max() < 1e-10
    # Cross-correlation peak lag between the shifted and original wavelet
    # equals the (rounded) requested shift in samples.
    xc = torch.fft.irfft(
        torch.fft.rfft(shifted) * torch.fft.rfft(r).conj(),
        n=nt,
    )
    lag = int(xc.argmax())
    assert lag == round(shift / dt)


def test_apply_delays_gradients() -> None:
    """Gradients flow through apply_delays to the base wavelet and delays."""
    dt = 1e-8
    nt = 400
    w = ricker(1e6, nt, dt, peak_time=nt * dt / 3, dtype=torch.float64).requires_grad_()
    delays = torch.tensor([1e-7, 2e-7], dtype=torch.float64, requires_grad=True)
    amps = td.apply_delays(w, delays, dt, nt)
    loss = (amps**2).sum()
    loss.backward()
    assert w.grad is not None
    assert w.grad.abs().sum() > 0
    assert delays.grad is not None
    assert delays.grad.abs().sum() > 0


def test_element_receiver_trace_weighted_sum_and_grad() -> None:
    """element_receiver_trace is a weighted sum; gradients flow through weights."""
    ra = torch.randn(2, 5, 100, dtype=torch.float64)
    weights = torch.arange(1.0, 6.0, dtype=torch.float64, requires_grad=True)
    trace = td.element_receiver_trace(ra, weights)
    assert trace.shape == (2, 100)
    expected = torch.einsum("srt,r->st", ra, weights.detach())
    assert torch.allclose(trace.detach(), expected)
    trace.sum().backward()
    assert weights.grad is not None
    assert weights.grad.abs().sum() > 0


def test_focused_line_array_focuses_at_geometric_focus() -> None:
    """A delay-focused line array concentrates energy near its geometric focus.

    Small 2-D scalar run: a linear array along the top of a homogeneous medium
    is delayed to focus at a chosen interior point. Accumulating the wavefield
    energy over time, the peak should sit within a few cells of the geometric
    focus.
    """
    torch.manual_seed(0)
    dtype = torch.float64
    ny, nx = 80, 90
    dx = 0.5e-3  # 0.5 mm cells
    c = 1500.0
    v = torch.full((ny, nx), c, dtype=dtype)

    freq = 0.5e6
    dt = 4e-8
    nt = 900

    # Linear array along the top (dim0 = 6), spanning dim1.
    array_depth = 6.0
    focus = torch.tensor([55.0, 45.0], dtype=dtype)
    loc, weights = td.line_element(
        [array_depth, 45.0],
        direction=[0.0, 1.0],
        length=60.0,
        spacing=1.0,
        dtype=dtype,
    )
    # c in cells per unit time: 1500 m/s on 0.5 mm cells is 3e6 cells/s, so the
    # delays come back in seconds, matching dt.
    delays = td.delays_to_focus(loc, focus, c / dx)
    # Fire near the start of the record so the beam has the whole run to reach
    # the focus, rather than relying on tone_burst's default peak time.
    n_cycles = 3
    base = tone_burst(
        freq,
        n_cycles,
        nt=nt,
        dt=dt,
        peak_time=n_cycles / freq / 2,
        dtype=dtype,
    )
    source_amplitudes, source_locations = td.element_source(
        base,
        loc,
        weights,
        delays=delays,
        dt=dt,
        nt=nt,
    )

    energy = torch.zeros(ny, nx, dtype=dtype)

    def accumulate(state) -> None:
        wf = state.get_wavefield("wavefield_0", view="inner")
        energy.add_(wf[0] ** 2)

    scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        accuracy=4,
        pml_width=15,
        pml_freq=freq,
        forward_callback=accumulate,
    )

    # Ignore the immediate near-field of the array itself (top rows).
    energy[: int(array_depth) + 12, :] = 0.0
    peak = torch.argmax(energy)
    py, px = int(peak // nx), int(peak % nx)
    err = math.hypot(py - float(focus[0]), px - float(focus[1]))
    assert err <= 5.0, f"focus at ({py},{px}), expected {focus.tolist()}, err {err}"


def test_builder_renamed_origins_by_keyword() -> None:
    """Each geometry names its own reference point; ``origin`` is gone."""
    dtype = torch.float64

    # line_element: `centre` is the midpoint of the segment.
    centre = [50.0, 10.0]
    loc, _ = td.line_element(
        centre=centre, direction=[0.0, 1.0], length=20.0, dtype=dtype
    )
    assert torch.allclose(
        (loc[0] + loc[-1]) / 2, torch.tensor(centre, dtype=dtype), atol=1e-12
    )

    # arc_element: `centre_of_curvature` is not a point on the arc; every arc
    # point is `radius` cells away from it.
    cc = torch.tensor([50.0, 50.0], dtype=dtype)
    loc_a, _ = td.arc_element(
        centre_of_curvature=cc, radius=30.0, angles=(-0.6, 0.6), dtype=dtype
    )
    r = torch.linalg.vector_norm(loc_a - cc, dim=1)
    assert torch.allclose(r, torch.full_like(r, 30.0), atol=1e-9)
    assert float(torch.cdist(cc.unsqueeze(0), loc_a).min()) > 1.0

    # disc_element: `centre` is the disc centre and is one of the points.
    disc_centre = torch.tensor([20.0, 20.0, 20.0], dtype=dtype)
    loc_d, _ = td.disc_element(
        centre=disc_centre, radius=8.0, normal=[1.0, 0.0, 0.0], dtype=dtype
    )
    assert float(torch.cdist(disc_centre.unsqueeze(0), loc_d).min()) < 1e-9

    # bowl_element: `apex` is a point on the bowl, and is *not* its centre of
    # curvature -- that is the separate `focus`.
    apex = torch.tensor([10.0, 20.0, 20.0], dtype=dtype)
    bowl_focus = torch.tensor([40.0, 20.0, 20.0], dtype=dtype)
    loc_b, _ = td.bowl_element(
        apex=apex, radius=30.0, diameter=30.0, focus=bowl_focus, dtype=dtype
    )
    assert torch.allclose(loc_b[0], apex, atol=1e-6)
    r_b = torch.linalg.vector_norm(loc_b - bowl_focus, dim=1)
    assert torch.allclose(r_b, torch.full_like(r_b, 30.0), atol=1e-6)

    # The old spelling fails loudly on every one of them.
    for call in (
        lambda: td.line_element(origin=centre, direction=[0.0, 1.0], length=1.0),
        lambda: td.arc_element(origin=cc, radius=1.0, angles=(0.0, 1.0)),
        lambda: td.disc_element(origin=disc_centre, radius=1.0, normal=[1.0, 0.0, 0.0]),
        lambda: td.bowl_element(
            origin=apex, radius=30.0, diameter=1.0, focus=bowl_focus
        ),
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument 'origin'"):
            call()


def test_builder_window_default_is_rect_and_window_weights_default_is_hann() -> None:
    """The two defaults deliberately differ, and the difference is pinned.

    An element should not be apodised unless asked, so every builder's
    ``window`` defaults to ``"rect"``; calling ``window_weights`` at all is
    asking for a window, so its ``kind`` defaults to ``"hann"``.
    """
    dtype = torch.float64
    n = 21

    # window_weights: bare call == explicit hann, and != rect.
    bare = td.window_weights(n, dtype=dtype)
    assert torch.equal(bare, td.window_weights(n, "hann", dtype=dtype))
    assert not torch.allclose(bare, torch.ones(n, dtype=dtype))
    assert bare[0].abs() < 1e-12  # a Hann window tapers to zero at the ends

    # Every builder: bare call == explicit rect (all ones), and != hann.
    builders = (
        lambda w: td.line_element(
            [0.0, 0.0], [0.0, 1.0], length=20.0, dtype=dtype, **w
        ),
        lambda w: td.arc_element(
            [0.0, 0.0], radius=10.0, angles=(-0.5, 0.5), dtype=dtype, **w
        ),
        lambda w: td.disc_element(
            [0.0, 0.0, 0.0], radius=6.0, normal=[0.0, 0.0, 1.0], dtype=dtype, **w
        ),
        lambda w: td.bowl_element(
            [0.0, 0.0, 0.0],
            radius=20.0,
            diameter=12.0,
            focus=[20.0, 0.0, 0.0],
            dtype=dtype,
            **w,
        ),
    )
    for builder in builders:
        _, default_wts = builder({})
        _, rect_wts = builder({"window": "rect"})
        _, hann_wts = builder({"window": "hann"})
        assert torch.equal(default_wts, rect_wts)
        assert torch.allclose(default_wts, torch.ones_like(default_wts))
        assert not torch.allclose(default_wts, hann_wts)

    # The module-level function's old spelling is gone.
    assert not hasattr(td, "window")
    assert "window_weights" in td.__all__
    assert "window" not in td.__all__


def test_element_source_shapes_and_weights() -> None:
    """element_source expands to [n_shots, n, nt] and applies the weights."""
    dtype = torch.float64
    nt = 64
    loc, weights = td.line_element(
        [5.0, 5.0], direction=[0.0, 1.0], length=6.0, window="hann", dtype=dtype
    )
    n = loc.shape[0]
    assert n > 1
    wavelet = ricker(1e6, nt, 1e-8, peak_time=nt * 1e-8 / 3, dtype=dtype)

    amps, locs = td.element_source(wavelet, loc, weights)
    assert amps.shape == (1, n, nt)
    assert locs.shape == (1, n, 2)
    # Every point emits the base wavelet scaled by its own weight.
    assert torch.allclose(amps[0], wavelet.unsqueeze(0) * weights.unsqueeze(1))
    assert torch.allclose(locs[0], loc)
    # Not all weights are 1: the weighting really happened.
    assert not torch.allclose(weights, torch.ones_like(weights))

    # n_shots broadcasts a 1-D wavelet without copying.
    amps4, locs4 = td.element_source(wavelet, loc, weights, n_shots=4)
    assert amps4.shape == (4, n, nt)
    assert locs4.shape == (4, n, 2)
    assert amps4.stride(0) == 0  # expand, not repeat
    assert locs4.stride(0) == 0
    assert torch.allclose(amps4[3], amps[0])

    # nt pads/truncates exactly as apply_delays does.
    assert td.element_source(wavelet, loc, weights, nt=100)[0].shape == (1, n, 100)
    short = td.element_source(wavelet, loc, weights, nt=30)[0]
    assert short.shape == (1, n, 30)
    assert torch.allclose(short[0, 0], wavelet[:30] * weights[0])


def test_element_source_applies_delays() -> None:
    """With delays, element_source is apply_delays times the weights."""
    dtype = torch.float64
    dt = 1e-8
    nt = 256
    loc, weights = td.line_element(
        [5.0, 5.0], direction=[0.0, 1.0], length=8.0, window="hamming", dtype=dtype
    )
    n = loc.shape[0]
    wavelet = ricker(1e6, nt, dt, peak_time=nt * dt / 3, dtype=dtype)
    delays = td.delays_to_focus(loc, [40.0, 5.0], c=1500.0 / 0.5e-3)

    amps, _ = td.element_source(wavelet, loc, weights, delays=delays, dt=dt, nt=nt)
    expected = td.apply_delays(wavelet, delays, dt, nt) * weights.unsqueeze(1)
    assert amps.shape == (1, n, nt)
    assert torch.allclose(amps[0], expected, atol=1e-12)
    # A delayed source really differs from an undelayed one.
    assert not torch.allclose(amps[0], td.element_source(wavelet, loc, weights)[0][0])


def test_element_source_multi_shot_wavelet() -> None:
    """A [n_shots, nt] wavelet gives one delayed source per shot."""
    dtype = torch.float64
    dt = 1e-8
    nt = 128
    loc, weights = td.line_element(
        [5.0, 5.0], direction=[0.0, 1.0], length=4.0, dtype=dtype
    )
    n = loc.shape[0]
    w0 = ricker(1e6, nt, dt, peak_time=nt * dt / 3, dtype=dtype)
    w1 = ricker(2e6, nt, dt, peak_time=nt * dt / 3, dtype=dtype)
    wavelet = torch.stack([w0, w1])
    delays = torch.linspace(0.0, 5 * dt, n, dtype=dtype)

    amps, locs = td.element_source(wavelet, loc, weights, delays=delays, dt=dt)
    assert amps.shape == (2, n, nt)
    assert locs.shape == (2, n, 2)
    for shot, w in enumerate((w0, w1)):
        expected = td.apply_delays(w, delays, dt, nt) * weights.unsqueeze(1)
        assert torch.allclose(amps[shot], expected, atol=1e-12)

    # An explicit n_shots that agrees is accepted; one that disagrees raises
    # rather than broadcasting.
    assert td.element_source(wavelet, loc, weights, n_shots=2)[0].shape == (2, n, nt)
    with pytest.raises(DeepwaveValueError, match=r"n_shots=3 disagrees"):
        td.element_source(wavelet, loc, weights, n_shots=3)


def test_element_source_validation() -> None:
    """element_source's errors name the shapes (or the argument) it saw."""
    dtype = torch.float64
    loc, weights = td.line_element(
        [5.0, 5.0], direction=[0.0, 1.0], length=4.0, dtype=dtype
    )
    n = loc.shape[0]
    wavelet = torch.zeros(32, dtype=dtype)
    delays = torch.zeros(n, dtype=dtype)

    with pytest.raises(DeepwaveValueError, match="dt is required when delays is given"):
        td.element_source(wavelet, loc, weights, delays=delays)
    with pytest.raises(DeepwaveValueError, match=r"weights \[3\] and locations"):
        td.element_source(wavelet, loc, torch.ones(3, dtype=dtype))
    with pytest.raises(DeepwaveValueError, match=r"got \[2, 3, 32\]"):
        td.element_source(torch.zeros(2, 3, 32, dtype=dtype), loc, weights)
    with pytest.raises(DeepwaveValueError, match=r"weights must have shape \[n\]"):
        td.element_source(wavelet, loc, weights.unsqueeze(1))
    with pytest.raises(DeepwaveValueError, match=r"locations must have shape"):
        td.element_source(wavelet, loc.unsqueeze(0), weights)
    with pytest.raises(DeepwaveValueError, match=r"delays must have shape \[n\]"):
        td.element_source(wavelet, loc, weights, delays=torch.zeros(2), dt=1e-8)
    with pytest.raises(DeepwaveValueError, match="n_shots must be at least 1"):
        td.element_source(wavelet, loc, weights, n_shots=0)
    # Every one of them is a ValueError too.
    with pytest.raises(ValueError, match="dt is required"):
        td.element_source(wavelet, loc, weights, delays=delays)


def test_element_source_gradients() -> None:
    """Gradients flow through element_source to wavelet, weights and delays."""
    dtype = torch.float64
    dt = 1e-8
    nt = 200
    loc, weights = td.line_element(
        [5.0, 5.0], direction=[0.0, 1.0], length=6.0, window="hann", dtype=dtype
    )
    n = loc.shape[0]
    w = ricker(1e6, nt, dt, peak_time=nt * dt / 3, dtype=dtype).requires_grad_()
    weights = weights.detach().clone().requires_grad_()
    delays = torch.linspace(0.0, 10 * dt, n, dtype=dtype).requires_grad_()
    locations = loc.detach().clone().requires_grad_()

    amps, locs = td.element_source(
        w, locations, weights, delays=delays, dt=dt, nt=nt, n_shots=3
    )
    (amps.pow(2).sum() + locs.sum()).backward()
    for name, t in (
        ("wavelet", w),
        ("weights", weights),
        ("delays", delays),
        ("locations", locations),
    ):
        assert t.grad is not None, name
        assert t.grad.abs().sum() > 0, name


def test_element_source_gradcheck() -> None:
    """element_source passes torch.autograd.gradcheck end-to-end."""
    dtype = torch.float64
    dt = 0.5
    nt = 12
    loc = torch.tensor([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]], dtype=dtype)

    def f(
        wavelet: torch.Tensor, weights: torch.Tensor, delays: torch.Tensor
    ) -> torch.Tensor:
        amps, _ = td.element_source(
            wavelet, loc, weights, delays=delays, dt=dt, nt=nt, n_shots=2
        )
        return amps

    torch.manual_seed(0)
    wavelet = torch.randn(nt, dtype=dtype, requires_grad=True)
    weights = torch.rand(3, dtype=dtype, requires_grad=True)
    delays = torch.tensor([0.0, 0.3, 0.7], dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(f, (wavelet, weights, delays))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_focus_and_delays_must_be_finite(value):
    """Reject nonfinite geometry and delay laws before FFT arithmetic."""
    locations = torch.tensor([[1.0, 2.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        td.delays_to_focus(locations, [value, 2.0], 1.0)
    with pytest.raises(ValueError, match="finite"):
        td.delays_to_focus(locations * value, [3.0, 2.0], 1.0)
    with pytest.raises(ValueError, match="finite"):
        td.apply_delays(torch.ones(8), torch.tensor([value]), 0.1)

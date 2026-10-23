"""Tests for deepwave.wavefields.

Volumetric sources, on-the-fly reductions, and boundary record/reinject --
the module's three capabilities:

* (a) Volumetric / extended source fields: the two-call
  ``source_field_to_sources`` + ``merge_point_sources`` route equals an
  explicit per-cell integer-source construction; the field gradient passes a
  finite-difference gradcheck; a WRI-flavoured optimisation with a field-source
  penalty reduces its loss.
* (b) Differentiable on-the-fly reductions: the segment-wise running DFT equals
  an offline FFT/DFT of the assembled trace, in both value and gradient.
* (c) Boundary record/reinject: recording on a box selects independent source
  and receiver channels, while replay injects the supplied traces linearly.
"""

import warnings

import pytest
import torch

import deepwave
from deepwave import scalar
from deepwave.checkpointing import snapshots
from deepwave.wavefields import (
    BoundaryRecording,
    MeasureResult,
    RunningDFT,
    boundary_surface_locations,
    measure,
    merge_point_sources,
    record_boundary,
    reinject_boundary,
    source_field_to_sources,
)
from deepwave.wavelets import ricker


def _ricker(freq, nt, dt):
    return ricker(freq, nt, dt, 1.0 / freq).reshape(1, 1, -1)


# ---------------------------------------------------------------------------
# (a) Volumetric / extended source fields
# ---------------------------------------------------------------------------


def test_source_field_matches_explicit_sources():
    """The source-field route equals building per-cell integer sources by hand."""
    torch.manual_seed(0)
    ny, nx, dt, nt = 16, 18, 0.001, 30
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    field = torch.randn(2, nt, ny, nx, dtype=torch.float64)
    amp, loc = source_field_to_sources(field)
    assert amp.shape == (2, ny * nx, nt)
    assert loc.shape == (2, ny * nx, 2)
    kw = {"pml_width": 10, "pml_freq": 25}
    out_manual = scalar(v, dx, dt, source_amplitudes=amp, source_locations=loc, **kw)
    # The documented two-call route, which replaced the removed
    # ``source_field=`` keyword, reproduces it exactly.
    route_amp, route_loc = merge_point_sources(amp, loc, None, None)
    out_route = scalar(v, dx, dt, sources={"u": (route_amp, route_loc)}, **kw)
    assert torch.equal(out_manual.wavefield_0, out_route.wavefield_0)


def test_source_field_single_cell_exact():
    """A one-cell source field injects into exactly that cell (Hicks delta)."""
    ny, nx, dt = 20, 20, 0.001
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    field = torch.zeros(1, 1, ny, nx, dtype=torch.float64)
    field[0, 0, 10, 12] = 1.0
    amp, loc = source_field_to_sources(field)
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=loc,
        pml_width=0,
        pml_freq=25,
    )
    nz = torch.nonzero(out.wavefield_0[0].abs() > 1e-14)
    assert nz.shape[0] == 1
    assert tuple(nz[0].tolist()) == (10, 12)


def test_source_field_region_masked():
    """A region-masked field with origin places sources only on masked cells."""
    torch.manual_seed(1)
    ny, nx, dt, nt = 24, 24, 0.001, 12
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    # A 6x6 sub-region at origin (8, 9), with a mask selecting some cells.
    region = (6, 6)
    field = torch.randn(1, nt, *region, dtype=torch.float64)
    mask = torch.zeros(*region, dtype=torch.bool)
    mask[1:4, 2:5] = True
    origin = [8, 9]
    amp, loc = source_field_to_sources(field, origin=origin, mask=mask)
    assert amp.shape[1] == int(mask.sum())
    # Locations are within the masked sub-region.
    assert int(loc[..., 0].min()) >= origin[0] + 1
    assert int(loc[..., 0].max()) <= origin[0] + 3
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=loc,
        pml_width=8,
        pml_freq=25,
    )
    assert torch.isfinite(out.wavefield_0).all()


def test_source_field_gradcheck():
    """The gradient w.r.t. the source field passes a finite-difference check."""
    torch.manual_seed(2)
    ny, nx, dt, nt = 10, 11, 0.002, 8
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    rl = torch.tensor([[[2, 2], [7, 8]]])
    field = torch.randn(1, nt, ny, nx, dtype=torch.float64, requires_grad=True)

    def f(fld):
        amp, loc = source_field_to_sources(fld)
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=loc,
            receiver_locations=rl,
            pml_width=6,
            pml_freq=25,
            max_vel=1500.0,
        )
        return out.receiver_amplitudes

    assert torch.autograd.gradcheck(f, (field,), atol=1e-6, rtol=1e-4)


def test_source_field_to_sources_validation():
    """source_field_to_sources rejects bad shapes and an all-empty selection."""
    with pytest.raises(ValueError, match="at least 3"):
        source_field_to_sources(torch.zeros(5, 5))
    field = torch.zeros(1, 4, 6, 6)
    with pytest.raises(ValueError, match="origin must have"):
        source_field_to_sources(field, origin=[0])
    with pytest.raises(ValueError, match="mask shape"):
        source_field_to_sources(field, mask=torch.zeros(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="zero cells"):
        source_field_to_sources(field, mask=torch.zeros(6, 6, dtype=torch.bool))


def test_merge_point_sources_with_explicit_points():
    """A source field combined with explicit point sources concatenates."""
    torch.manual_seed(6)
    ny, nx, dt, nt = 20, 20, 0.001, 12
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    region = (4, 4)
    field = torch.randn(1, nt, *region, dtype=torch.float64)
    mask = torch.zeros(*region, dtype=torch.bool)
    mask[1, 1] = True  # exactly one field cell, at model cell (3, 3)
    origin = [2, 2]
    field_amp, field_loc = source_field_to_sources(field, origin=origin, mask=mask)

    pt_amp = torch.randn(1, 1, nt, dtype=torch.float64)
    pt_loc = torch.tensor([[[10, 10]]])  # disjoint from the field's (3, 3) cell

    kw = {"pml_width": 6, "pml_freq": 25}
    merged_amp, merged_loc = merge_point_sources(field_amp, field_loc, pt_amp, pt_loc)
    out_merged = scalar(
        v,
        dx,
        dt,
        source_amplitudes=merged_amp,
        source_locations=merged_loc,
        **kw,
    )
    manual_amp = torch.cat([field_amp, pt_amp], dim=1)
    manual_loc = torch.cat([field_loc, pt_loc], dim=1)
    out_manual = scalar(
        v, dx, dt, source_amplitudes=manual_amp, source_locations=manual_loc, **kw
    )
    assert torch.equal(out_merged.wavefield_0, out_manual.wavefield_0)


def test_merge_point_sources_validation():
    """_merge_point_sources rejects an incomplete or incompatible point-source pair."""
    from deepwave.wavefields import merge_point_sources as _merge_point_sources

    field_amp = torch.zeros(1, 2, 5)
    field_loc = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="source_locations must be provided"):
        _merge_point_sources(field_amp, field_loc, torch.zeros(1, 1, 5), None)
    with pytest.raises(ValueError, match="same number of time samples"):
        _merge_point_sources(
            field_amp,
            field_loc,
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 2, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="must be an integer tensor"):
        _merge_point_sources(
            field_amp, field_loc, torch.zeros(1, 1, 5), torch.zeros(1, 1, 2)
        )


def test_source_field_wri_flavored_optimization_reduces_loss():
    """A WRI-flavoured objective (data misfit + field-source penalty) descends.

    A dense source field standing in for the wave-equation residual is optimised
    to fit receiver data recorded from a point source, with an energy penalty on
    the field itself. The combined loss must decrease under gradient descent.
    """
    torch.manual_seed(3)
    ny, nx, dt, nt = 20, 22, 0.001, 40
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    rl = torch.tensor([[[3, 3], [16, 4], [10, 18], [5, 15]]])
    kw = {"pml_width": 8, "pml_freq": 15, "max_vel": 1500.0}
    # Target data from a point source at (10, 11).
    target = scalar(
        v,
        dx,
        dt,
        source_amplitudes=_ricker(15.0, nt, dt).double(),
        source_locations=torch.tensor([[[10, 11]]]),
        receiver_locations=rl,
        **kw,
    ).receiver_amplitudes.detach()

    # Invert for a localised extended source field (standing in for the
    # wave-equation residual of WRI) plus an energy penalty on the field.
    origin, region = [8, 9], (5, 5)
    field = torch.zeros(1, nt, *region, dtype=torch.float64, requires_grad=True)
    penalty = 1e-3
    opt = torch.optim.LBFGS([field], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        amp, loc = source_field_to_sources(field, origin=origin)
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=loc,
            receiver_locations=rl,
            **kw,
        )
        misfit = (out.receiver_amplitudes - target).pow(2).sum()
        loss = misfit + penalty * field.pow(2).sum()
        loss.backward()
        return loss

    initial = closure().detach().item()
    opt.step(closure)
    final = closure().detach().item()
    assert final < 0.01 * initial


# ---------------------------------------------------------------------------
# (b) Differentiable on-the-fly reductions
# ---------------------------------------------------------------------------


def _scalar_dft_problem(nt=48, dt=0.001):
    torch.manual_seed(4)
    ny, nx = 16, 18
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.randn(1, 1, nt, dtype=torch.float64)
    sl = torch.tensor([[[8, 9]]])
    rl = torch.tensor([[[3, 3], [12, 4], [10, 14]]])
    kw = {
        "source_locations": sl,
        "receiver_locations": rl,
        "pml_width": 8,
        "pml_freq": 25,
        "max_vel": 1500.0,
    }
    return v, dx, dt, src, kw, nt


def test_running_dft_equals_rfft_at_bins():
    """Segment-wise running DFT equals torch.fft.rfft at exact bin frequencies."""
    v, dx, dt, src, kw, nt = _scalar_dft_problem()
    bins = [2, 5, 9]
    freqs = [k / (nt * dt) for k in bins]
    res, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT(freqs, on="u")],
        n_segments=7,
        **kw,
    )
    out = scalar(v, dx, dt, source_amplitudes=src, **kw)
    offline = torch.fft.rfft(out.receiver_amplitudes, dim=-1)[..., bins]
    assert torch.allclose(res[0], offline, atol=1e-10, rtol=0)


def test_running_dft_equals_direct_dft_arbitrary_freqs():
    """Running DFT at off-bin frequencies equals a direct DFT of the full trace."""
    v, dx, dt, src, kw, nt = _scalar_dft_problem()
    freqs = [13.7, 41.2, 88.0]
    res, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT(freqs, on="u")],
        n_segments=5,
        **kw,
    )
    out = scalar(v, dx, dt, source_amplitudes=src, **kw)
    t = torch.arange(nt, dtype=torch.float64)
    ker = torch.exp(
        (-2j * torch.pi * dt)
        * torch.tensor(freqs, dtype=torch.float64).reshape(-1, 1)
        * t.reshape(1, -1)
    )
    direct = torch.einsum("srt,ft->srf", out.receiver_amplitudes.to(ker.dtype), ker)
    assert torch.allclose(res[0], direct, atol=1e-10, rtol=0)


@pytest.mark.parametrize("n_segments", [1, 3, 8])
def test_running_dft_independent_of_segmentation(n_segments):
    """The accumulated DFT does not depend on how the run is segmented."""
    v, dx, dt, src, kw, _nt = _scalar_dft_problem()
    freqs = [20.0, 60.0]
    res1, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT(freqs, on="u")],
        n_segments=1,
        **kw,
    )
    resn, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT(freqs, on="u")],
        n_segments=n_segments,
        **kw,
    )
    assert torch.allclose(res1[0], resn[0], atol=1e-10, rtol=0)


def test_running_dft_gradient_equals_offline():
    """Gradient of a loss on the running DFT equals that of the offline DFT."""
    v, dx, dt, src, kw, nt = _scalar_dft_problem()
    freqs = [13.7, 41.2]
    t = torch.arange(nt, dtype=torch.float64)
    ker = torch.exp(
        (-2j * torch.pi * dt)
        * torch.tensor(freqs, dtype=torch.float64).reshape(-1, 1)
        * t.reshape(1, -1)
    )

    v1 = v.clone().requires_grad_(True)
    res, _ = measure(
        scalar,
        [v1],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT(freqs, on="u")],
        n_segments=5,
        **kw,
    )
    res[0].abs().pow(2).sum().backward()

    v2 = v.clone().requires_grad_(True)
    out = scalar(v2, dx, dt, source_amplitudes=src, **kw)
    direct = torch.einsum("srt,ft->srf", out.receiver_amplitudes.to(ker.dtype), ker)
    direct.abs().pow(2).sum().backward()

    assert torch.allclose(v1.grad, v2.grad, atol=1e-10, rtol=0)


def test_running_dft_on_wavefield_matches_snapshot_dft():
    """A wavefield running DFT equals an offline DFT of per-step snapshots."""
    torch.manual_seed(5)
    ny, nx, dt, nt = 14, 16, 0.001, 20
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.randn(1, 1, nt, dtype=torch.float64)
    sl = torch.tensor([[[7, 8]]])
    kw = {"source_locations": sl, "pml_width": 6, "pml_freq": 25}
    freqs = [30.0, 120.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, _ = measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [RunningDFT(freqs, on="wavefield_0")],
            max_vel=1500.0,
            **kw,
        )
    _, snaps = snapshots(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        list(range(1, nt + 1)),
        max_vel=1500.0,
        **kw,
    )
    t = torch.arange(nt, dtype=torch.float64)
    ker = torch.exp(
        (-2j * torch.pi * dt)
        * torch.tensor(freqs, dtype=torch.float64).reshape(-1, 1)
        * t.reshape(1, -1)
    )
    stack = torch.stack([snaps[k]["wavefield_0"] for k in range(1, nt + 1)], dim=0)
    offline = torch.einsum("tsyx,ft->syxf", stack.to(ker.dtype), ker)
    assert torch.allclose(res[0], offline, atol=1e-10, rtol=0)


def test_dft_kernel_non_default_dtype_cpu():
    """_dft_kernel honours a non-default (float64) dtype and matches a manual DFT."""
    from deepwave.wavefields import _dft_kernel

    freqs = torch.tensor([10.0, 25.0], dtype=torch.float32)
    dt = 0.001
    kernel = _dft_kernel(freqs, 3, 5, dt, torch.float64, torch.device("cpu"))
    assert kernel.dtype == torch.complex128
    assert kernel.shape == (2, 5)
    t = torch.arange(3, 8, dtype=torch.float64)
    expected = torch.exp(
        (-2j * torch.pi * dt)
        * freqs.to(torch.float64).reshape(-1, 1)
        * t.reshape(1, -1)
    )
    assert torch.allclose(kernel, expected)


def test_running_dft_long_float32_trace_matches_float64_fft():
    """Long float32 records retain phase accuracy in the running DFT kernel."""
    from deepwave.wavefields import _dft_kernel

    nt = 100_000
    dt = 0.001
    bin_index = 1234
    frequency = bin_index / (nt * dt)
    generator = torch.Generator().manual_seed(20260929)
    signal = torch.randn(nt, dtype=torch.float32, generator=generator)
    kernel = _dft_kernel([frequency], 0, nt, dt, torch.float32, torch.device("cpu"))

    running = (signal.to(torch.complex64) @ kernel.T)[0]
    independent = torch.fft.fft(signal.to(torch.float64))[bin_index]

    error = (running.to(torch.complex128) - independent).abs()
    assert error <= 2e-3 + 2e-5 * independent.abs()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dft_kernel_frequencies_moved_to_data_device():
    """_dft_kernel moves a CPU frequencies tensor onto the data's CUDA device.

    Regression for the fix that coerces ``frequencies`` to the data's device;
    before it, a CPU ``frequencies`` and a CUDA wavefield raised a device
    mismatch. Runs only on the maintainer's GPU.
    """
    from deepwave.wavefields import _dft_kernel

    freqs = torch.tensor([10.0, 25.0])  # on CPU
    dev = torch.device("cuda")
    kernel = _dft_kernel(freqs, 0, 4, 0.001, torch.float32, dev)
    assert kernel.device.type == "cuda"
    assert kernel.shape == (2, 4)


def test_measure_requires_step_ratio_one_when_segmented():
    """A segmented measure() rejects step_ratio > 1 (source-resampling artefact)."""
    ny, nx = 32, 32
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = _ricker(15.0, 40, 0.004).double()
    kw = {
        "source_locations": torch.tensor([[[16, 16]]]),
        "receiver_locations": torch.tensor([[[4, 4]]]),
        "pml_freq": 15.0,
        "max_vel": 2000.0,
    }
    # The message must name measure() itself and its own remedy (n_segments=1,
    # the one case exempt from the step_ratio == 1 requirement), not
    # checkpoint()'s storage_mode advice (measure() hard-codes
    # storage_mode="device" internally).
    with pytest.raises(ValueError, match=r"measure\(\) requires.*n_segments=1"):
        measure(
            scalar,
            [v],
            dx,
            0.004,
            {"source_amplitudes": src},
            [RunningDFT([10.0], on="u")],
            n_segments=2,
            **kw,
        )
    # A single segment is exempt (no cross-boundary source resampling).
    res, _ = measure(
        scalar,
        [v],
        dx,
        0.004,
        {"source_amplitudes": src},
        [RunningDFT([10.0], on="u")],
        n_segments=1,
        **kw,
    )
    assert isinstance(res, tuple)
    assert res[0].shape[-1] == 1


def test_measure_batched_model_matches_offline_dft():
    """A per-shot batched model measures correctly (ndim comes from locations)."""
    _v, dx, dt, src, kw, nt = _scalar_dft_problem()
    v = torch.stack(
        [
            torch.full((16, 18), 1500.0, dtype=torch.float64),
            torch.full((16, 18), 1700.0, dtype=torch.float64),
        ],
    )
    src = src.repeat(2, 1, 1)
    kw = dict(kw)
    kw["source_locations"] = kw["source_locations"].repeat(2, 1, 1)
    kw["receiver_locations"] = kw["receiver_locations"].repeat(2, 1, 1)
    kw["max_vel"] = 1700.0
    bins = [2, 5]
    freqs = [k / (nt * dt) for k in bins]
    res, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT(freqs, on="u")],
        n_segments=6,
        **kw,
    )
    out = scalar(v, dx, dt, source_amplitudes=src, **kw)
    offline = torch.fft.rfft(out.receiver_amplitudes)[..., bins]
    assert res[0].shape == (2, 3, len(bins))
    assert torch.allclose(res[0], offline, atol=1e-10, rtol=0)


def test_measure_returns_one_tensor_per_measurement():
    """measure() returns one tensor per measurement, in the order given."""
    v, dx, dt, src, kw, _nt = _scalar_dft_problem()
    res, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT([10.0], on="u"), RunningDFT([20.0, 30.0], on="u")],
        n_segments=1,
        **kw,
    )
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res[0].shape[-1] == 1
    assert res[1].shape[-1] == 2


def test_measure_rejects_bad_arguments():
    """measure() rejects empty measurements, reserved kwargs, and bad targets."""
    v, dx, dt, src, kw, _nt = _scalar_dft_problem()
    with pytest.raises(ValueError, match="at least one measurement"):
        measure(scalar, [v], dx, dt, {"source_amplitudes": src}, [], **kw)
    with pytest.raises(ValueError, match="storage_mode"):
        measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [RunningDFT([10.0], on="u")],
            storage_mode="device",
            **kw,
        )
    with pytest.raises(ValueError, match="is not something that can be measured"):
        measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [RunningDFT([10.0], on="nope")],
            **kw,
        )


def test_measure_rejects_empty_source_amplitudes():
    """measure() rejects an empty source_amplitudes dict (nt is undefined)."""
    v, dx, dt, _src, kw, _nt = _scalar_dft_problem()
    with pytest.raises(ValueError, match="at least one source-amplitude"):
        measure(scalar, [v], dx, dt, {}, [RunningDFT([10.0], on="u")], **kw)


def test_running_dft_on_wavefield_warns_and_forces_n_segments():
    """A non-1/non-nt n_segments on a wavefield measurement warns, forcing nt."""
    torch.manual_seed(5)
    ny, nx, dt, nt = 14, 16, 0.001, 20
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.randn(1, 1, nt, dtype=torch.float64)
    sl = torch.tensor([[[7, 8]]])
    kw = {"source_locations": sl, "pml_width": 6, "pml_freq": 25, "max_vel": 1500.0}
    with pytest.warns(UserWarning, match="forced to nt"):
        res, _ = measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [RunningDFT([30.0], on="wavefield_0")],
            n_segments=3,
            **kw,
        )
    assert res[0].shape[0] == 1  # shot
    assert res[0].shape[-1] == 1  # n_freq


def test_measure_with_no_receivers_returns_empty_and_return_output():
    """No-receiver measure() falls back to an empty result; return_output works.

    A measure() with no receivers accumulates nothing and falls back to a
    correctly-shaped/dtyped empty result; return_output=True also returns the
    final PropagatorOutput.
    """
    v, dx, dt, src, _kw, _nt = _scalar_dft_problem()
    kw = {
        "source_locations": torch.tensor([[[8, 9]]]),
        "pml_width": 8,
        "pml_freq": 25,
        "max_vel": 1500.0,
    }
    res, _ = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT([10.0], on="u")],
        **kw,
    )
    assert res[0].shape == (0, 0, 1)
    assert res[0].dtype == torch.complex128

    res2, out = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT([10.0], on="u")],
        return_output=True,
        **kw,
    )
    assert out is not None
    assert res2[0].shape == (0, 0, 1)


# ---------------------------------------------------------------------------
# (c) Boundary record / reinject
# ---------------------------------------------------------------------------


def test_boundary_surface_locations_perimeter():
    """The surface enumeration is the box perimeter, each cell once."""
    box = [(2, 6), (3, 8)]  # 4 x 5 box
    surf = boundary_surface_locations(box)
    # Perimeter of a 4x5 block = 4*5 - 2*3 (interior 2x3) = 20 - 6 = 14.
    assert surf.shape == (14, 2)
    # No duplicates.
    assert len(torch.unique(surf, dim=0)) == surf.shape[0]
    # All within the box, at least one coord on a face.
    for cell in surf:
        y, x = int(cell[0]), int(cell[1])
        assert 2 <= y < 6
        assert 3 <= x < 8
        assert y in (2, 5) or x in (3, 7)


@pytest.mark.parametrize(
    ("name", "ndim", "receiver_channel", "source_channel"),
    [
        ("scalar", 2, "u", "u"),
        ("elastic", 2, "p", "pressure_rate"),
        ("elastic", 2, "x", "pressure_rate"),
        ("elastic", 2, "p", "x"),
        ("acoustic_born", 2, "p", "volume_rate"),
        ("acoustic_born", 2, "x", "volume_rate"),
        ("acoustic_born", 2, "p", "x"),
        ("elastic", 3, "z", "pressure_rate"),
        ("elastic", 3, "p", "z"),
        ("elastic", 3, "z", "z"),
    ],
)
def test_record_boundary_selects_source_and_receiver_independently(
    name, ndim, receiver_channel, source_channel
):
    """Surface traces match a direct call for independent channel choices."""
    spec = deepwave.spec(name)
    fn = getattr(deepwave, name)
    shape = (14,) * ndim
    dtype = torch.float64
    v = torch.full(shape, 1500.0, dtype=dtype)
    if name == "scalar":
        models = (v,)
    elif name == "elastic":
        models = (
            torch.full(shape, 2.0e9, dtype=dtype),
            torch.full(shape, 1.0e9, dtype=dtype),
            torch.full(shape, 1.0 / 2000.0, dtype=dtype),
        )
    else:
        models = (v, torch.full(shape, 2000.0, dtype=dtype))
    nt = 16
    amplitudes = torch.zeros((1, 1, nt), dtype=dtype)
    amplitudes[..., 0] = 1.0
    location = torch.full((1, 1, ndim), 5, dtype=torch.long)
    box = [(5, 9)] * ndim
    surface = boundary_surface_locations(box)
    recv_locations = surface.unsqueeze(0)
    common = {"pml_width": 3, "pml_freq": 20.0, "max_vel": 2500.0, "accuracy": 2}
    if name == "acoustic_born":
        dv = torch.zeros(shape, dtype=dtype)
        dv[(6,) * ndim] = 15.0
        common["dv"] = dv

    # The defaults are checked separately from explicit choices. In particular,
    # a receiver channel must never also select a same-named source channel.
    for receiver_choice, source_choice in (
        (None, None),
        (receiver_channel, None),
        (None, source_channel),
        (receiver_channel, source_channel),
    ):
        resolved_receiver = receiver_choice or spec.receiver_channels(ndim)[0]
        resolved_source = source_choice or spec.source_channels(ndim)[0]
        recorded, got_surface = record_boundary(
            fn,
            models,
            5.0,
            0.0005,
            amplitudes,
            location,
            box,
            channel=receiver_choice,
            source_channel=source_choice,
            **common,
        )
        direct = fn(
            *models,
            5.0,
            0.0005,
            **{
                spec.source_amplitudes_kwarg(resolved_source): amplitudes,
                spec.source_locations_kwarg(resolved_source): location,
                spec.receiver_locations_kwarg(resolved_receiver): recv_locations,
            },
            **common,
        )
        assert torch.equal(got_surface, surface)
        assert torch.equal(
            recorded, getattr(direct, spec.receiver_output_name(resolved_receiver))
        )
        assert recorded.abs().max() > 0


def test_reinject_boundary_matches_direct_injection_and_source_gradient():
    """Replay is exactly the corresponding source injection, including AD."""
    dtype = torch.float64
    v = torch.full((16, 16), 1500.0, dtype=dtype)
    surface = boundary_surface_locations([(6, 10), (6, 10)])
    recording = torch.randn((1, surface.shape[0], 18), dtype=dtype)
    receiver_locations = torch.tensor([[[8, 8]]])
    common = {"pml_width": 3, "pml_freq": 20.0, "max_vel": 1500.0}
    scale = -1.75
    replay_input = recording.clone().requires_grad_(True)
    replay = reinject_boundary(
        scalar,
        [v],
        5.0,
        0.0005,
        replay_input,
        surface,
        amplitude_scale=scale,
        receiver_locations=receiver_locations,
        **common,
    )
    direct_input = recording.clone().requires_grad_(True)
    direct = scalar(
        v,
        5.0,
        0.0005,
        source_amplitudes=direct_input * scale,
        source_locations=surface.unsqueeze(0),
        receiver_locations=receiver_locations,
        **common,
    )
    assert torch.equal(replay.receiver_amplitudes, direct.receiver_amplitudes)
    assert torch.equal(replay.wavefield_0, direct.wavefield_0)
    replay.receiver_amplitudes.square().sum().backward()
    direct.receiver_amplitudes.square().sum().backward()
    assert torch.equal(replay_input.grad, direct_input.grad)
    assert replay_input.grad.abs().max() > 0


def test_reinject_amplitude_scale_is_linear():
    """``amplitude_scale=k`` scales the reinjected output linearly in ``k``.

    The propagator is linear in its source amplitudes and ``amplitude_scale``
    multiplies them, so the replay run's wavefield and receiver data are
    proportional to it. The old spelling ``scale=`` fails loudly.
    """
    ny, nx, nt, dt = 24, 24, 30, 0.001
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    wav = _ricker(25.0, nt, dt).double()
    common = {"pml_freq": 25.0, "max_vel": 1500.0}
    box = [(8, 16), (8, 16)]
    rec, surf = record_boundary(
        scalar, [v], dx, dt, wav, torch.tensor([[[4, 4]]]), box, **common
    )
    interior = torch.tensor([[[11, 12], [12, 11]]])
    kwargs = {"receiver_locations": interior, **common}
    base = reinject_boundary(scalar, [v], dx, dt, rec, surf, **kwargs)
    k = -3.25
    scaled = reinject_boundary(
        scalar, [v], dx, dt, rec, surf, amplitude_scale=k, **kwargs
    )
    for got, want in (
        (scaled.receiver_amplitudes, k * base.receiver_amplitudes),
        (scaled.wavefield_0, k * base.wavefield_0),
    ):
        assert torch.allclose(
            got, want, rtol=1e-10, atol=1e-10 * float(want.abs().max())
        )
    # A zero scale reinjects nothing at all.
    quiet = reinject_boundary(
        scalar, [v], dx, dt, rec, surf, amplitude_scale=0.0, **kwargs
    )
    assert float(quiet.receiver_amplitudes.abs().max()) == 0.0
    # The pre-release spelling is gone, and loudly so.
    with pytest.raises(TypeError, match="unexpected keyword argument 'scale'"):
        reinject_boundary(scalar, [v], dx, dt, rec, surf, scale=k, **kwargs)


def test_record_boundary_shapes():
    """record_boundary returns [shot, n_surface, nt] traces on the box surface."""
    ny, nx, dt, nt = 40, 40, 0.0005, 60
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    wav = _ricker(20.0, nt, dt).double()
    box = [(10, 20), (12, 22)]
    rec, surf = record_boundary(
        scalar,
        [v],
        dx,
        dt,
        wav,
        torch.tensor([[[5, 5]]]),
        box,
        pml_width=10,
        pml_freq=20.0,
    )
    assert surf.shape[0] == boundary_surface_locations(box).shape[0]
    assert rec.shape == (1, surf.shape[0], nt)


def test_running_dft_on_auxiliary_state_key_matches_snapshot_dft():
    """An auxiliary-state running DFT equals an offline DFT of per-step snapshots.

    ``psiy_m1`` is one of the absorbing boundary's memory variables, carried
    inside the opaque ``auxiliary_state``. It is a field of the state at one
    time step, so it is measured exactly like ``wavefield_0``: per-step
    segmentation, one sample per outer step.
    """
    torch.manual_seed(5)
    ny, nx, dt, nt = 14, 16, 0.001, 20
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.randn(1, 1, nt, dtype=torch.float64)
    sl = torch.tensor([[[7, 8]]])
    kw = {"source_locations": sl, "pml_width": 6, "pml_freq": 25}
    freqs = [30.0, 120.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, _ = measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [RunningDFT(freqs, on="psiy_m1")],
            max_vel=1500.0,
            **kw,
        )
    _, snaps = snapshots(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        list(range(1, nt + 1)),
        max_vel=1500.0,
        **kw,
    )
    t = torch.arange(nt, dtype=torch.float64)
    ker = torch.exp(
        (-2j * torch.pi * dt)
        * torch.tensor(freqs, dtype=torch.float64).reshape(-1, 1)
        * t.reshape(1, -1)
    )
    stack = torch.stack([snaps[k]["psiy_m1"] for k in range(1, nt + 1)], dim=0)
    offline = torch.einsum("tsyx,ft->syxf", stack.to(ker.dtype), ker)
    assert torch.allclose(res[0], offline, atol=1e-10, rtol=0)


def test_running_dft_on_auxiliary_state_key_emits_no_deprecation_warning():
    """The value comes from ``out.auxiliary_state[name]``, not an attribute.

    ``getattr(out, "psiy_m1")`` is not a field at all: an auxiliary field's
    individual name was never part of the interface, so asking for it raises.
    Reading a measurement that way would break, and routing one through any
    deprecated spelling would warn at a caller who did nothing deprecated, so
    this asserts that nothing on the path warns.
    """
    ny, nx, dt, nt = 12, 12, 0.001, 4
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.zeros(1, 1, nt, dtype=torch.float64)
    src[..., 0] = 1.0
    kw = {"source_locations": torch.tensor([[[6, 6]]]), "pml_width": 3, "pml_freq": 25}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [RunningDFT([30.0], on="psiy_m1")],
            n_segments=nt,
            max_vel=1500.0,
            **kw,
        )
    assert not [
        w for w in caught if issubclass(w.category, (DeprecationWarning, FutureWarning))
    ]


def test_measure_unknown_target_lists_both_name_groups():
    """A typo's error names the public outputs and the auxiliary-state keys."""
    ny, nx, dt, nt = 12, 12, 0.001, 4
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.zeros(1, 1, nt, dtype=torch.float64)
    kw = {"source_locations": torch.tensor([[[6, 6]]]), "pml_width": 3, "pml_freq": 25}
    with pytest.raises(
        ValueError, match="is not something that can be measured"
    ) as excinfo:
        measure(
            scalar,
            [v],
            5.0,
            dt,
            {"source_amplitudes": src},
            [RunningDFT([30.0], on="wavefield_9")],
            max_vel=1500.0,
            **kw,
        )
    message = str(excinfo.value)
    assert "'wavefield_9' is not something that can be measured" in message
    # Group zero: the channel vocabulary, which ``on=`` also accepts.
    assert "available channels" in message
    assert "'u'" in message
    # Group one: the public outputs.
    assert "available output names" in message
    for name in ("wavefield_0", "wavefield_m1", "receiver_amplitudes"):
        assert name in message
    # Group two: the auxiliary-state keys, as the internal names listed on
    # the callback-names page.
    for name in ("psiy_m1", "psix_m1", "zetay_m1", "zetax_m1"):
        assert name in message
    assert "auxiliary_state keys" in message
    assert "listed on the callback-names page" in message


def test_measure_on_auxiliary_state_itself_still_raises():
    """``auxiliary_state`` is a collection of tensors, not a measurable tensor."""
    ny, nx, dt, nt = 12, 12, 0.001, 4
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = torch.zeros(1, 1, nt, dtype=torch.float64)
    kw = {"source_locations": torch.tensor([[[6, 6]]]), "pml_width": 3, "pml_freq": 25}
    with pytest.raises(ValueError, match="is not an output") as excinfo:
        measure(
            scalar,
            [v],
            5.0,
            dt,
            {"source_amplitudes": src},
            [RunningDFT([30.0], on="auxiliary_state")],
            max_vel=1500.0,
            **kw,
        )
    message = str(excinfo.value)
    assert "'auxiliary_state' is not an output that can be measured" in message
    assert "carried opaquely" in message
    assert "psiy_m1" in message


# ---------------------------------------------------------------------------
# Named results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("return_output", [False, True])
def test_measure_returns_named_tuple(return_output):
    """measure() returns a MeasureResult for both values of return_output.

    The shape of the return value must not depend on ``return_output``:
    ``MeasureResult`` is a ``tuple`` subclass, so a result that was sometimes
    a bare list and sometimes a pair would make ``for m in measure(...)``
    iterate the wrong thing without saying so. ``measurements`` comes first,
    so the common case is ``measure(...)[0]``.
    """
    v, dx, dt, src, kw, _nt = _scalar_dft_problem()
    result = measure(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        [RunningDFT([10.0], on="u"), RunningDFT([20.0, 30.0], on="u")],
        n_segments=1,
        return_output=return_output,
        **kw,
    )
    assert isinstance(result, MeasureResult)
    assert result._fields == ("measurements", "output")
    measurements, output = result
    assert measurements is result.measurements
    assert isinstance(measurements, tuple)
    assert len(measurements) == 2
    assert measurements[0].shape[-1] == 1
    assert measurements[1].shape[-1] == 2
    if return_output:
        assert output is not None
        assert output.receiver_amplitudes.shape[0] == 1
    else:
        assert output is None


def test_record_boundary_returns_named_tuple():
    """record_boundary() returns a BoundaryRecording that still unpacks.

    Field order is preserved, so ``rec, surf = record_boundary(...)`` is
    unchanged and ``reinject_boundary(..., *result)`` keeps composing.
    """
    ny, nx, nt, dt = 24, 24, 30, 0.001
    dx = 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    src = _ricker(25.0, nt, dt).double()
    common = {"pml_freq": 25.0, "max_vel": 1500.0}
    box = [(8, 16), (8, 16)]
    result = record_boundary(
        scalar,
        [v],
        dx,
        dt,
        src,
        torch.tensor([[[4, 4]]]),
        box,
        **common,
    )
    assert isinstance(result, BoundaryRecording)
    assert result._fields == ("recording", "surface_locations")
    rec, surf = result
    assert rec is result.recording
    assert surf is result.surface_locations
    assert rec.shape == (1, surf.shape[0], nt)
    # Still composes positionally with reinject_boundary.
    reout = reinject_boundary(scalar, [v], dx, dt, rec, surf, **common)
    assert reout.wavefield_0.shape[0] == 1

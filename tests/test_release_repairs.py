"""Small independent invariants for the release-review failure classes."""

from __future__ import annotations

import errno
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING

import pytest
import torch

import deepwave

if TYPE_CHECKING:
    from pathlib import Path


def test_release_reporting_preserves_failure_when_log_cannot_be_written(
    monkeypatch, tmp_path
):
    """A full temporary filesystem must not hide the original test failure."""
    from pathlib import Path

    from scripts._release_pytest_plugin import ReleaseResult

    result = ReleaseResult()
    monkeypatch.setenv("DEEPWAVE_RELEASE_RESULT", str(tmp_path / "result.json"))

    def full_filesystem(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "open", full_filesystem)
    report = pytest.TestReport(
        "synthetic_failure",
        ("test.py", 0, "synthetic_failure"),
        {},
        "failed",
        "original assertion failure",
        "call",
    )
    result.pytest_runtest_logreport(report)
    assert result.outcomes == {"synthetic_failure": "failed"}
    assert any("No space left" in error for error in result.collection_errors)


@pytest.mark.parametrize("pml_freq", [None, 0.1])
@pytest.mark.parametrize("argument", ["source_amplitudes", "source_locations"])
def test_pml_inference_preserves_source_type_errors(
    pml_freq: float | None, argument: str
) -> None:
    """Inference must not dereference invalid inputs before validation."""
    kwargs = {
        "source_amplitudes": torch.ones(1, 1, 4),
        "source_locations": torch.tensor([[[4]]]),
    }
    kwargs[argument] = kwargs[argument].tolist()
    with pytest.raises(deepwave.DeepwaveTypeError, match=argument):
        deepwave.scalar(torch.ones(8), 1.0, 0.1, pml_freq=pml_freq, **kwargs)


def test_running_dft_default_targets_first_receiver_output() -> None:
    """An omitted target accumulates the first receiver trace."""
    source = torch.ones(1, 1, 8, dtype=torch.float64)
    point = torch.tensor([[[8]]])
    model = torch.full((16,), 1.5, dtype=source.dtype)
    kwargs = {
        "source_locations": point,
        "receiver_locations": point,
        "max_vel": 2.0,
        "pml_width": 2,
        "pml_freq": 0.1,
    }
    measured, _ = deepwave.wavefields.measure(
        deepwave.scalar,
        [model],
        1.0,
        0.1,
        {"source_amplitudes": source},
        [deepwave.wavefields.RunningDFT([0.0])],
        n_segments=1,
        **kwargs,
    )
    trace = deepwave.scalar(
        model, 1.0, 0.1, source_amplitudes=source, **kwargs
    ).receiver_amplitudes
    torch.testing.assert_close(measured[0][..., 0], trace.sum(-1).to(torch.complex128))


def test_source_field_rejects_integer_mask() -> None:
    """Integer masks cannot silently become source selectors."""
    with pytest.raises(deepwave.DeepwaveValueError, match=r"torch\.bool"):
        deepwave.wavefields.source_field_to_sources(
            torch.ones(1, 2, 4), mask=torch.ones(4, dtype=torch.int64)
        )


def test_odd_highest_fourier_bin_survives_resampling() -> None:
    """The last rFFT bin is a real frequency when the record is odd."""
    n, k, ratio = 5, 2, 3
    t = torch.arange(n, dtype=torch.float64)
    signal = torch.cos(2 * torch.pi * k * t / n)
    up = deepwave.discretisation.upsample(signal, ratio)
    expected = torch.cos(
        2 * torch.pi * k * torch.arange(n * ratio, dtype=t.dtype) / (n * ratio)
    )
    torch.testing.assert_close(up, expected, atol=2e-14, rtol=0)
    torch.testing.assert_close(
        deepwave.discretisation.downsample(expected, ratio), signal
    )


@pytest.mark.parametrize("family", ["scalar", "acoustic", "elastic"])
@pytest.mark.parametrize("first_input", ["model", "source"])
def test_sampled_model_double_backward_is_rejected(
    family: str, first_input: str
) -> None:
    """A sampled first-gradient Jacobian cannot use the directional HVP rule."""
    source = torch.ones(1, 1, 8, dtype=torch.float64, requires_grad=True)
    common = {
        "max_vel": 2.0,
        "pml_width": 2,
        "pml_freq": 0.1,
        "model_gradient_sampling_interval": 2,
    }
    if family == "scalar":
        model = torch.full((10,), 1.5, dtype=source.dtype, requires_grad=True)
        point = torch.tensor([[[5]]])
        out = deepwave.scalar(
            model,
            1.0,
            0.1,
            source_amplitudes=source,
            source_locations=point,
            receiver_locations=point,
            **common,
        )
        trace = out.receiver_amplitudes
    elif family == "acoustic":
        model = torch.full((10,), 1.5, dtype=source.dtype, requires_grad=True)
        point = torch.tensor([[[5]]])
        out = deepwave.acoustic(
            model,
            torch.ones_like(model),
            1.0,
            0.1,
            source_amplitudes_volume_rate=source,
            source_locations_volume_rate=point,
            receiver_locations_p=point,
            **common,
        )
        trace = out.receiver_amplitudes_p
    else:
        model = torch.full((10, 10), 2.0, dtype=source.dtype, requires_grad=True)
        point = torch.tensor([[[5, 5]]])
        common["max_vel"] = 3.0
        out = deepwave.elastic(
            model,
            torch.ones_like(model),
            torch.ones_like(model),
            1.0,
            0.1,
            source_amplitudes_pressure_rate=source,
            source_locations_pressure_rate=point,
            receiver_locations_p=point,
            **common,
        )
        trace = out.receiver_amplitudes_p
    first = model if first_input == "model" else source
    (gradient,) = torch.autograd.grad(trace.square().sum(), first, create_graph=True)
    with pytest.raises(deepwave.DeepwaveRuntimeError, match="sampled model gradients"):
        torch.autograd.grad(gradient, model, grad_outputs=torch.ones_like(gradient))


def test_unrelaxed_speed_controls_viscoacoustic_cfl() -> None:
    """A low-Q run must substep before the coupled wave update diverges."""
    model = torch.full((16,), 1.5, dtype=torch.float64)
    q = torch.ones_like(model)
    source = torch.zeros(1, 1, 48, dtype=model.dtype)
    source[..., 0] = 1
    point = torch.tensor([[[8]]])
    out = deepwave.viscoacoustic(
        model,
        torch.ones_like(model),
        q,
        1.0,
        0.35,
        q_freq=0.01,
        source_amplitudes_pressure_rate=source,
        source_locations_pressure_rate=point,
        receiver_locations_p=point,
        pml_width=0,
        boundary="rigid",
        pml_freq=0.1,
        backend="eager",
    )
    assert out.info.step_ratio >= 3
    assert out.receiver_amplitudes_p.abs().max() < 10


@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_differentiable_forward_callback_rejects_state_edit(backend: str) -> None:
    """A reset without an adjoint rule must not return a false source VJP."""
    source = torch.zeros(1, 1, 12, dtype=torch.float64, requires_grad=True)
    point = torch.tensor([[[12]]])

    def reset(state: deepwave.CallbackState) -> None:
        if state.step == 4:
            field = state.get_wavefield("wavefield_0", view="full")
            state.set_wavefield("wavefield_0", torch.zeros_like(field), view="full")

    with pytest.raises(deepwave.DeepwaveRuntimeError, match="no adjoint"):
        deepwave.scalar(
            torch.full((24,), 1.5, dtype=source.dtype),
            1.0,
            0.2,
            source_amplitudes=source,
            source_locations=point,
            receiver_locations=point,
            pml_width=3,
            pml_freq=0.1,
            backend=backend,
            forward_callback=reset,
        )


def test_segmented_hicks_inference_matches_direct() -> None:
    """Inference uses the active original records in both call forms."""
    dt, nt = 0.001, 64
    t = torch.arange(nt, dtype=torch.float64) * dt
    envelope = torch.exp(-(((t - 0.032) / 0.012) ** 2))
    source = torch.stack(
        (
            1.2 * envelope * torch.sin(2 * torch.pi * 31.25 * t),
            envelope * torch.sin(2 * torch.pi * 125 * t),
        )
    ).unsqueeze(0)
    model = torch.full((48,), 1500.0, dtype=t.dtype)
    kwargs = {
        "source_locations": torch.tensor([[[20.5], [24.0]]]),
        "receiver_locations": torch.tensor([[[26]]]),
        "max_vel": 1500.0,
        "pml_width": 5,
        "backend": "compiled",
    }
    direct = deepwave.scalar(model, 5.0, dt, source_amplitudes=source, **kwargs)
    segmented = deepwave.checkpointing.run_segments(
        deepwave.scalar,
        [model],
        5.0,
        dt,
        {"source_amplitudes": source},
        n_segments=2,
        **kwargs,
    )
    assert direct.info.pml_freq == segmented.info.pml_freq
    torch.testing.assert_close(
        direct.receiver_amplitudes, segmented.receiver_amplitudes, rtol=0, atol=0
    )


def test_run_segments_run_info_uses_callers_time_axis() -> None:
    """The combined result reports caller dt/nt and the whole-call CFL ratio."""
    nt, dx, dt = 12, 1.0, 0.5
    model = torch.full((24,), 1.5, dtype=torch.float64)
    source = torch.zeros(1, 1, nt, dtype=model.dtype)
    source[..., 0] = 1
    locations = torch.tensor([[[8]]])
    kwargs = {
        "source_amplitudes": source,
        "source_locations": locations,
        "receiver_locations": locations,
        "max_vel": 2.0,
        "pml_width": 2,
        "pml_freq": 0.1,
    }

    direct = deepwave.scalar(model, dx, dt, **kwargs)
    segmented = deepwave.checkpointing.run_segments(
        deepwave.scalar,
        [model],
        dx,
        dt,
        {"source_amplitudes": source},
        n_segments=3,
        source_locations=locations,
        receiver_locations=locations,
        max_vel=2.0,
        pml_width=2,
        pml_freq=0.1,
    )

    assert direct.info.step_ratio > 1
    assert segmented.info.dt == dt
    assert segmented.info.nt == nt
    assert segmented.info.step_ratio == direct.info.step_ratio
    assert segmented.receiver_amplitudes.shape == (1, 1, nt)


def test_zero_time_receiver_output_keeps_public_axis_order() -> None:
    """An empty time axis does not move shots or receivers in the result."""
    n_shots, n_receivers = 2, 3
    source = torch.empty((n_shots, 1, 0), dtype=torch.float64)
    source_locations = torch.tensor([[[5]], [[6]]])
    receiver_locations = torch.tensor([[[7], [8], [9]], [[10], [11], [12]]])

    out = deepwave.scalar(
        torch.full((24,), 1.5, dtype=source.dtype),
        1.0,
        0.1,
        source_amplitudes=source,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=2.0,
        pml_width=2,
        pml_freq=0.1,
    )

    assert out.receiver_amplitudes.shape == (n_shots, n_receivers, 0)
    assert out.receiver_amplitudes.dtype == source.dtype
    assert out.receiver_amplitudes.device == source.device


@pytest.mark.parametrize(("mode", "compress"), [("disk", False), ("device", True)])
def test_retained_history_backward_overlap(
    mode: str, compress: bool, tmp_path: Path
) -> None:
    """Two cotangents on one retained graph cannot share writable staging."""
    torch.manual_seed(28)
    model = (1.4 + 0.2 * torch.rand(72, dtype=torch.float64)).requires_grad_()
    source = torch.randn(1, 1, 100, dtype=model.dtype)
    out = deepwave.scalar(
        model,
        1.0,
        0.2,
        source_amplitudes=source,
        source_locations=torch.tensor([[[32]]]),
        receiver_locations=torch.tensor([[[38], [42]]]),
        max_vel=2.0,
        pml_width=5,
        pml_freq=0.1,
        storage_mode=mode,
        storage_path=tmp_path if mode == "disk" else None,
        storage_compression=compress,
    )
    trace = out.receiver_amplitudes
    cotangents = [torch.randn_like(trace), torch.randn_like(trace)]

    def vjp(index: int) -> torch.Tensor:
        return torch.autograd.grad(
            trace, model, grad_outputs=cotangents[index], retain_graph=True
        )[0]

    expected = [vjp(0), vjp(1)]
    for _ in range(2):
        barrier = Barrier(2)

        def concurrent(index: int) -> torch.Tensor:
            barrier.wait(timeout=10)
            return vjp(index)

        with ThreadPoolExecutor(max_workers=2) as pool:
            observed = list(pool.map(concurrent, (0, 1)))
        for actual, reference in zip(observed, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_segmented_inference_accepts_cpu_geometry_and_masks() -> None:
    """Masks follow amplitude placement through direct and segmented inference."""
    dtype = torch.float64
    model = torch.full((16, 16), 1500.0, device="cuda", dtype=dtype)
    source = (
        torch.stack(
            (
                deepwave.wavelets.ricker(25.0, 16, 0.001, 0.006, dtype=dtype),
                1000 * deepwave.wavelets.ricker(125.0, 16, 0.001, 0.006, dtype=dtype),
            )
        )
        .unsqueeze(0)
        .cuda()
    )
    kwargs = {
        "source_locations": torch.tensor([[[7, 7], [8, 8]]]),
        "source_mask": torch.tensor([[True, False]]),
        "receiver_locations": torch.tensor([[[7, 8]]]),
        "max_vel": 1600.0,
        "pml_width": 2,
    }
    direct = deepwave.scalar(model, 5.0, 0.001, source_amplitudes=source, **kwargs)
    segmented = deepwave.checkpointing.run_segments(
        deepwave.scalar,
        [model],
        5.0,
        0.001,
        {"source_amplitudes": source},
        2,
        **kwargs,
    )
    assert segmented.info.pml_freq == direct.info.pml_freq
    torch.testing.assert_close(
        segmented.receiver_amplitudes, direct.receiver_amplitudes, rtol=0, atol=0
    )

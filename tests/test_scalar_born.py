"""Tests for deepwave.scalar_born."""

from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
from test_utils import _set_coords, _set_sources, scattered

from deepwave import IGNORE_LOCATION, ScalarBorn, scalar_born
from deepwave.backend_utils import USE_OPENMP
from deepwave.common import cfl_condition, downsample, upsample
from deepwave.wavelets import ricker

torch._dynamo.config.cache_size_limit = 256  # noqa: SLF001
torch.autograd.set_detect_anomaly(True)


def test_python_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_born_scatter(
            propagator=scalarbornprop, prop_kwargs={"python_backend": "jit"}
        )
        run_born_scatter(
            propagator=scalarbornprop, prop_kwargs={"python_backend": "compile"}
        )


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((50,), (5,)),
        ((50, 50), (5, 5)),
        ((10, 10, 10), (5, 5, 5)),
    ],
)
def test_born_scatter(nx, dx) -> None:
    """Test Born propagation in a 2D model with a point scatterer."""
    if len(nx) == 1:
        expected_diffs = {
            2: 27,
            4: 2.2,
            6: 1.0,
            8: 0.75,
        }
    elif len(nx) == 2:
        expected_diffs = {2: 0.14, 4: 0.0025, 6: 0.003, 8: 0.00062}
    else:
        expected_diffs = {
            2: 0.01,
            4: 0.0012,
            6: 0.0017,
            8: 0.00075,
        }
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [True, False]:
            expected, actual = run_born_scatter(
                propagator=scalarbornprop,
                dt=0.001 if accuracy == 4 else 0.0001,
                nx=nx,
                dx=dx,
                prop_kwargs={
                    "pml_width": 30,
                    "python_backend": python,
                    "accuracy": accuracy,
                },
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        print((actuals[True] - actuals[False]).abs().max().item(), flush=True)
        assert torch.allclose(actuals[True], actuals[False], atol=1e-3)


def test_born_scatter_v_batched_2d() -> None:
    """Test Born propagation in a batched 2D velocity model."""
    expected_diffs = {2: 0.14, 4: 0.0025, 6: 0.003, 8: 0.00062}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [True, False]:
            expected, actual = run_born_scatter(
                c=torch.tensor([[[1500.0]], [[1600.0]]]),
                propagator=scalarbornprop,
                dt=0.001 if accuracy == 4 else 0.0001,
                prop_kwargs={
                    "pml_width": 30,
                    "python_backend": python,
                    "accuracy": accuracy,
                },
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        assert torch.allclose(actuals[True], actuals[False], atol=1e-4)


def test_born_scatter_scatter_batched_2d() -> None:
    """Test Born propagation in a batched 2D scatter model."""
    expected_diffs = {2: 0.14, 4: 0.0025, 6: 0.003, 8: 0.00062}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [True, False]:
            expected, actual = run_born_scatter(
                dscatter=torch.tensor([[[50.0]], [[100.0]]]),
                propagator=scalarbornprop,
                dt=0.001 if accuracy == 4 else 0.0001,
                prop_kwargs={
                    "pml_width": 30,
                    "python_backend": python,
                    "accuracy": accuracy,
                },
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        assert torch.allclose(actuals[True], actuals[False], atol=1e-4)


def test_born_scatter_2d_module() -> None:
    """Test Born propagation using the Module interface."""
    expected, actual = run_born_scatter(
        propagator=scalarbornprop,
        dt=0.001,
        prop_kwargs={"pml_width": 30},
        functional=False,
    )
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm() < 0.0025


def scalarbornprop(
    model: torch.Tensor,
    scatter: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
    bg_receiver_locations: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Optional[Union[int, List[int]]] = None,
    survey_pad: Optional[Union[int, List[int]]] = None,
    origin: Optional[List[int]] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    psiz_m1: Optional[torch.Tensor] = None,
    psiy_m1: Optional[torch.Tensor] = None,
    psix_m1: Optional[torch.Tensor] = None,
    zetaz_m1: Optional[torch.Tensor] = None,
    zetay_m1: Optional[torch.Tensor] = None,
    zetax_m1: Optional[torch.Tensor] = None,
    wavefield_sc_0: Optional[torch.Tensor] = None,
    wavefield_sc_m1: Optional[torch.Tensor] = None,
    psiz_sc_m1: Optional[torch.Tensor] = None,
    psiy_sc_m1: Optional[torch.Tensor] = None,
    psix_sc_m1: Optional[torch.Tensor] = None,
    zetaz_sc_m1: Optional[torch.Tensor] = None,
    zetay_sc_m1: Optional[torch.Tensor] = None,
    zetax_sc_m1: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    functional: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Wraps the scalar born propagator."""
    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs["max_vel"] = 2000

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs["pml_width"] = pml_width
    prop_kwargs["survey_pad"] = survey_pad
    prop_kwargs["origin"] = origin

    device = model.device
    if source_amplitudes is not None:
        source_amplitudes = source_amplitudes.to(device)
        source_locations = source_locations.to(device)
    if receiver_locations is not None:
        receiver_locations = receiver_locations.to(device)
    if bg_receiver_locations is not None:
        bg_receiver_locations = bg_receiver_locations.to(device)

    if functional:
        return scalar_born(
            model,
            scatter,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            bg_receiver_locations=bg_receiver_locations,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            psiz_m1=psiz_m1,
            psiy_m1=psiy_m1,
            psix_m1=psix_m1,
            zetaz_m1=zetaz_m1,
            zetay_m1=zetay_m1,
            zetax_m1=zetax_m1,
            wavefield_sc_0=wavefield_sc_0,
            wavefield_sc_m1=wavefield_sc_m1,
            psiz_sc_m1=psiz_sc_m1,
            psiy_sc_m1=psiy_sc_m1,
            psix_sc_m1=psix_sc_m1,
            zetaz_sc_m1=zetaz_sc_m1,
            zetay_sc_m1=zetay_sc_m1,
            zetax_sc_m1=zetax_sc_m1,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            **prop_kwargs,
        )
    prop = ScalarBorn(model, scatter, dx)
    return prop(
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        bg_receiver_locations=bg_receiver_locations,
        wavefield_0=wavefield_0,
        wavefield_m1=wavefield_m1,
        psiz_m1=psiz_m1,
        psiy_m1=psiy_m1,
        psix_m1=psix_m1,
        zetaz_m1=zetaz_m1,
        zetay_m1=zetay_m1,
        zetax_m1=zetax_m1,
        wavefield_sc_0=wavefield_sc_0,
        wavefield_sc_m1=wavefield_sc_m1,
        psiz_sc_m1=psiz_sc_m1,
        psiy_sc_m1=psiy_sc_m1,
        psix_sc_m1=psix_sc_m1,
        zetaz_sc_m1=zetaz_sc_m1,
        zetay_sc_m1=zetay_sc_m1,
        zetax_sc_m1=zetax_sc_m1,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def scalarbornpropchained(
    model: torch.Tensor,
    scatter: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
    receiver_locations: Optional[torch.Tensor],
    bg_receiver_locations: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Optional[Union[int, List[int]]] = None,
    survey_pad: Optional[Union[int, List[int]]] = None,
    origin: Optional[List[int]] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    psiz_m1: Optional[torch.Tensor] = None,
    psiy_m1: Optional[torch.Tensor] = None,
    psix_m1: Optional[torch.Tensor] = None,
    zetaz_m1: Optional[torch.Tensor] = None,
    zetay_m1: Optional[torch.Tensor] = None,
    zetax_m1: Optional[torch.Tensor] = None,
    wavefield_sc_0: Optional[torch.Tensor] = None,
    wavefield_sc_m1: Optional[torch.Tensor] = None,
    psiz_sc_m1: Optional[torch.Tensor] = None,
    psiy_sc_m1: Optional[torch.Tensor] = None,
    psix_sc_m1: Optional[torch.Tensor] = None,
    zetaz_sc_m1: Optional[torch.Tensor] = None,
    zetay_sc_m1: Optional[torch.Tensor] = None,
    zetax_sc_m1: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    functional: bool = True,
    n_chained: int = 2,
) -> Tuple[torch.Tensor, ...]:
    """Wraps multiple scalar born propagators chained sequentially."""
    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs["max_vel"] = 2000

    prop_kwargs["pml_freq"] = 25

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs["pml_width"] = pml_width
    prop_kwargs["survey_pad"] = survey_pad
    prop_kwargs["origin"] = origin

    device = model.device
    if source_amplitudes is not None:
        source_amplitudes = source_amplitudes.to(device)
        source_locations = source_locations.to(device)
    if receiver_locations is not None:
        receiver_locations = receiver_locations.to(device)
    if bg_receiver_locations is not None:
        bg_receiver_locations = bg_receiver_locations.to(device)

    max_vel = 2000
    dt, step_ratio = cfl_condition(dx[0], dx[1], dt, max_vel)
    if source_amplitudes is not None:
        source_amplitudes = upsample(source_amplitudes, step_ratio)
        nt_per_segment = (
            (
                (source_amplitudes.shape[-1] + n_chained - 1) // n_chained
                + step_ratio
                - 1
            )
            // step_ratio
        ) * step_ratio
    else:
        nt *= step_ratio
        nt_per_segment = (
            ((nt + n_chained - 1) // n_chained + step_ratio - 1) // step_ratio
        ) * step_ratio

    wfc = wavefield_0
    wfp = wavefield_m1
    psiy = psiy_m1
    psix = psix_m1
    zetay = zetay_m1
    zetax = zetax_m1
    wfcsc = wavefield_sc_0
    wfpsc = wavefield_sc_m1
    psiysc = psiy_sc_m1
    psixsc = psix_sc_m1
    zetaysc = zetay_sc_m1
    zetaxsc = zetax_sc_m1

    if receiver_locations is not None:
        if source_amplitudes is not None:
            receiver_amplitudes = torch.zeros(
                receiver_locations.shape[0],
                receiver_locations.shape[1],
                source_amplitudes.shape[-1],
                dtype=model.dtype,
                device=model.device,
            )
        else:
            receiver_amplitudes = torch.zeros(
                receiver_locations.shape[0],
                receiver_locations.shape[1],
                nt,
                dtype=model.dtype,
                device=model.device,
            )

    if bg_receiver_locations is not None:
        if source_amplitudes is not None:
            bg_receiver_amplitudes = torch.zeros(
                bg_receiver_locations.shape[0],
                bg_receiver_locations.shape[1],
                source_amplitudes.shape[-1],
                dtype=model.dtype,
                device=model.device,
            )
        else:
            bg_receiver_amplitudes = torch.zeros(
                bg_receiver_locations.shape[0],
                bg_receiver_locations.shape[1],
                nt,
                dtype=model.dtype,
                device=model.device,
            )

    for segment_idx in range(n_chained):
        if source_amplitudes is not None:
            segment_source_amplitudes = source_amplitudes[
                ...,
                nt_per_segment * segment_idx : min(
                    nt_per_segment * (segment_idx + 1),
                    source_amplitudes.shape[-1],
                ),
            ]
            segment_nt = None
        else:
            segment_source_amplitudes = None
            segment_nt = (
                min(nt_per_segment * (segment_idx + 1), source_amplitudes.shape[-1])
                - nt_per_segment * segment_idx
            )
        (
            wfc,
            wfp,
            psiy,
            psix,
            zetay,
            zetax,
            wfcsc,
            wfpsc,
            psiysc,
            psixsc,
            zetaysc,
            zetaxsc,
            segment_bg_receiver_amplitudes,
            segment_receiver_amplitudes,
        ) = scalar_born(
            model,
            scatter,
            dx,
            dt,
            source_amplitudes=segment_source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            bg_receiver_locations=bg_receiver_locations,
            wavefield_0=wfc,
            wavefield_m1=wfp,
            psiy_m1=psiy,
            psix_m1=psix,
            zetay_m1=zetay,
            zetax_m1=zetax,
            wavefield_sc_0=wfcsc,
            wavefield_sc_m1=wfpsc,
            psiy_sc_m1=psiysc,
            psix_sc_m1=psixsc,
            zetay_sc_m1=zetaysc,
            zetax_sc_m1=zetaxsc,
            nt=segment_nt,
            model_gradient_sampling_interval=step_ratio,
            **prop_kwargs,
        )
        if receiver_locations is not None:
            receiver_amplitudes[
                ...,
                nt_per_segment * segment_idx : min(
                    nt_per_segment * (segment_idx + 1),
                    receiver_amplitudes.shape[-1],
                ),
            ] = segment_receiver_amplitudes
        if bg_receiver_locations is not None:
            bg_receiver_amplitudes[
                ...,
                nt_per_segment * segment_idx : min(
                    nt_per_segment * (segment_idx + 1),
                    bg_receiver_amplitudes.shape[-1],
                ),
            ] = segment_bg_receiver_amplitudes

    if receiver_locations is not None:
        receiver_amplitudes = downsample(receiver_amplitudes, step_ratio)
    if bg_receiver_locations is not None:
        bg_receiver_amplitudes = downsample(bg_receiver_amplitudes, step_ratio)

    return (
        wfc,
        wfp,
        psiy,
        psix,
        zetay,
        zetax,
        wfcsc,
        wfpsc,
        psiysc,
        psixsc,
        zetaysc,
        zetaxsc,
        bg_receiver_amplitudes,
        receiver_amplitudes,
    )


def run_born_scatter(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 150,
    dscatter: Union[float, torch.Tensor] = 150,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.0001,
    nx: Tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the expected and actual point scattered waveform at point."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(c, torch.Tensor):
        c = c.to(device)
        min_c = c.min().item()
        shot_c = c.flatten().tolist()
    else:
        min_c = c
        shot_c = [c] * num_shots
    if isinstance(dscatter, torch.Tensor):
        dscatter = dscatter.to(device)
        shot_dscatter = dscatter.flatten().tolist()
    else:
        shot_dscatter = [dscatter] * num_shots
    model = torch.ones(*nx, device=device, dtype=dtype) * c

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / min_c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
    x_p = _set_coords(1, 1, nx.tolist(), "middle")[0, 0]
    scatter = torch.zeros_like(model) * dscatter
    if isinstance(dscatter, torch.Tensor):
        if ndim == 3:
            scatter[..., x_p[-3], x_p[-2], x_p[-1]] = dscatter.flatten()
        elif ndim == 2:
            scatter[..., x_p[-2], x_p[-1]] = dscatter.flatten()
        else:
            scatter[..., x_p[-1]] = dscatter.flatten()
    else:
        if ndim == 3:
            scatter[..., x_p[-3], x_p[-2], x_p[-1]] = dscatter
        elif ndim == 2:
            scatter[..., x_p[-2], x_p[-1]] = dscatter
        else:
            scatter[..., x_p[-1]] = dscatter
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    expected = torch.zeros(num_shots, num_receivers_per_shot, nt, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver, :] += scattered(
                    x_r[shot, receiver],
                    x_s[shot, source],
                    x_p,
                    dx,
                    dt,
                    shot_c[shot],
                    shot_dscatter[shot],
                    -sources["amplitude"][shot, source, :],
                    ndim,
                ).to(dtype)

    actual = propagator(
        model,
        scatter,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )[-1]

    return expected, actual


def run_born_forward(
    c: Union[float, torch.Tensor],
    freq: float,
    dx: Union[float, List[float]],
    dt: float,
    nx: Tuple[int, ...],
    num_shots: int,
    num_sources_per_shot: int,
    num_receivers_per_shot: int,
    propagator: Any,
    prop_kwargs: Optional[Dict[str, Any]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    dc: float = 100,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    """Create a random model and forward propagate."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        torch.ones(*nx, device=device, dtype=dtype) * c
        + torch.randn(*nx, dtype=dtype).to(device) * dc
    )
    scatter = torch.randn(*nx, dtype=dtype).to(device) * dc

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    return propagator(
        model,
        scatter,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )


def run_born_forward_2d(
    c: float = 1500,
    freq: float = 25,
    dx: Tuple[float, float] = (5, 5),
    dt: float = 0.004,
    nx: Tuple[int, int] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    """Runs run_forward with default parameters for 2D."""
    return run_born_forward(
        c,
        freq,
        dx,
        dt,
        nx,
        num_shots,
        num_sources_per_shot,
        num_receivers_per_shot,
        propagator,
        prop_kwargs,
        device,
        dtype,
        **kwargs,
    )


def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        for python in [True, False]:
            actual_cpu = run_born_forward_2d(
                propagator=scalarbornprop,
                device=torch.device("cpu"),
                prop_kwargs={"python_backend": python},
            )
            actual_gpu = run_born_forward_2d(
                propagator=scalarbornprop,
                device=torch.device("cuda"),
                prop_kwargs={"python_backend": python},
            )
            for cpui, gpui in zip(actual_cpu, actual_gpu):
                assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_unused_source_receiver(
    c=1500,
    dc=100,
    freq=25,
    dx=(5, 5),
    dt=0.005,
    nx=(50, 50),
    num_shots=2,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    num_scatter_receivers_per_shot=3,
    propagator=scalarbornprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Test forward and backward propagation when a source and receiver are unused."""
    assert num_shots > 1
    assert num_sources_per_shot > 1
    assert num_receivers_per_shot > 1
    assert num_scatter_receivers_per_shot > num_shots
    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    for python in [True, False]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["python_backend"] = python

        model = (
            torch.ones(*nx, device=device, dtype=dtype) * c
            + torch.randn(*nx, device=device, dtype=dtype) * dc
        )
        scatter = torch.randn(*nx, dtype=dtype).to(device) * dc

        nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
        x_rsc = _set_coords(
            num_shots,
            num_scatter_receivers_per_shot,
            nx.tolist(),
            "bottom",
        )
        sources = _set_sources(x_s, freq, dt, nt, dtype)

        # Forward with source and receiver ignored
        modeli = model.clone()
        modeli.requires_grad_()
        scatteri = scatter.clone()
        scatteri.requires_grad_()
        for i in range(num_shots):
            x_s[i, i, :] = IGNORE_LOCATION
            x_r[i, i, :] = IGNORE_LOCATION
            x_rsc[i, i + 1, :] = IGNORE_LOCATION
        sources["locations"] = x_s
        out_ignored = propagator(
            modeli,
            scatteri,
            dx.tolist(),
            dt,
            sources["amplitude"],
            sources["locations"],
            x_rsc,
            x_r,
            prop_kwargs=prop_kwargs,
        )

        ((out_ignored[-1] ** 2).sum() + (out_ignored[-2] ** 2).sum()).backward()

        # Forward with amplitudes of sources that will be ignored set to zero
        modelf = model.clone()
        modelf.requires_grad_()
        scatterf = scatter.clone()
        scatterf.requires_grad_()
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
        x_rsc = _set_coords(
            num_shots,
            num_scatter_receivers_per_shot,
            nx.tolist(),
            "bottom",
        )
        for i in range(num_shots):
            sources["amplitude"][i, i].fill_(0)
        sources["locations"] = x_s
        out_filled = propagator(
            modelf,
            scatterf,
            dx.tolist(),
            dt,
            sources["amplitude"],
            sources["locations"],
            x_rsc,
            x_r,
            prop_kwargs=prop_kwargs,
        )
        # Set receiver amplitudes of receiver that will be ignored to zero
        for i in range(num_shots):
            out_filled[-1][i, i + 1].fill_(0)
            out_filled[-2][i, i].fill_(0)

        ((out_filled[-1] ** 2).sum() + (out_filled[-2] ** 2).sum()).backward()

        for ofi, oii in zip(out_filled, out_ignored):
            assert torch.allclose(ofi, oii)

        assert torch.allclose(modelf.grad, modeli.grad)
        assert torch.allclose(scatterf.grad, scatteri.grad)


def run_scalarbornfunc(nt: int = 3) -> None:
    """Runs scalar_born_func for testing purposes."""
    from deepwave.scalar_born import scalar_born_func

    torch.manual_seed(1)
    ny = 25
    nx = 26
    n_batch = 2
    dt = 0.001
    dy = dx = 5
    pml_width = [3, 3, 3, 3]
    n_sources_per_shot = 2
    step_ratio = 1
    accuracy = 4
    fd_pad = accuracy // 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = 1500 * torch.ones(1, ny, nx, dtype=torch.double, device=device)
    c += 100 * torch.rand_like(c)
    scatter = torch.randn_like(c)
    wfc = torch.randn(
        n_batch,
        ny - 2 * fd_pad,
        nx - 2 * fd_pad,
        dtype=torch.double,
        device=device,
    )
    wfp = torch.randn_like(wfc)
    psiy = torch.randn_like(wfc)
    psix = torch.randn_like(wfc)
    zetay = torch.randn_like(wfc)
    zetax = torch.randn_like(wfc)
    wfcsc = torch.randn_like(wfc)
    wfpsc = torch.randn_like(wfc)
    psiysc = torch.randn_like(wfc)
    psixsc = torch.randn_like(wfc)
    zetaysc = torch.randn_like(wfc)
    zetaxsc = torch.randn_like(wfc)
    source_amplitudes = torch.randn(
        nt,
        n_batch,
        n_sources_per_shot,
        dtype=torch.double,
        device=device,
    )
    source_amplitudessc = torch.randn(
        nt,
        n_batch,
        n_sources_per_shot,
        dtype=torch.double,
        device=device,
    )
    sources_i = (
        torch.tensor([[7 * nx + 7, 8 * nx + 8], [9 * nx + 9, 10 * nx + 10]])
        .long()
        .to(device)
    )
    receivers_i = (
        torch.tensor(
            [
                [7 * nx + 7, 8 * nx + 8, 9 * nx + 9],
                [11 * nx + 11, 12 * nx + 12, 13 * nx + 12],
            ],
        )
        .long()
        .to(device)
    )
    receiverssc_i = (
        torch.tensor(
            [
                [7 * nx + 7, 8 * nx + 8, 9 * nx + 9, 8 * nx + 9],
                [11 * nx + 11, 12 * nx + 12, 13 * nx + 12, 13 * nx + 11],
            ],
        )
        .long()
        .to(device)
    )
    c.requires_grad_()
    scatter.requires_grad_()
    source_amplitudes.requires_grad_()
    source_amplitudessc.requires_grad_()
    wfc.requires_grad_()
    wfp.requires_grad_()
    psiy.requires_grad_()
    psix.requires_grad_()
    zetay.requires_grad_()
    zetax.requires_grad_()
    wfcsc.requires_grad_()
    wfpsc.requires_grad_()
    psiysc.requires_grad_()
    psixsc.requires_grad_()
    zetaysc.requires_grad_()
    zetaxsc.requires_grad_()
    ay = torch.randn(ny, dtype=torch.double, device=device)
    ax = torch.randn(nx, dtype=torch.double, device=device)
    by = torch.randn(ny, dtype=torch.double, device=device)
    bx = torch.randn(nx, dtype=torch.double, device=device)
    by[fd_pad + pml_width[0] : ny - fd_pad - pml_width[1]].fill_(0)
    bx[fd_pad + pml_width[2] : nx - fd_pad - pml_width[3]].fill_(0)
    ay[fd_pad + pml_width[0] : ny - fd_pad - pml_width[1]].fill_(0)
    ax[fd_pad + pml_width[2] : nx - fd_pad - pml_width[3]].fill_(0)
    dbydy = torch.randn(ny, dtype=torch.double, device=device)
    dbxdx = torch.randn(nx, dtype=torch.double, device=device)
    dbydy[2 * fd_pad + pml_width[0] : ny - 2 * fd_pad - pml_width[1]].fill_(0)
    dbxdx[2 * fd_pad + pml_width[2] : nx - 2 * fd_pad - pml_width[3]].fill_(0)
    ay = ay[None, :, None]
    ax = ax[None, None, :]
    by = by[None, :, None]
    bx = bx[None, None, :]
    dbydy = dbydy[None, :, None]
    dbxdx = dbxdx[None, None, :]

    for sa in [source_amplitudes, source_amplitudes.detach()]:

        def wrap(python):
            inputs = (
                python,
                c,
                scatter,
                sa,
                source_amplitudessc,
                [ay, by, dbydy, ax, bx, dbxdx],
                sources_i,
                None,
                receivers_i,
                receiverssc_i,
                [dy, dx],
                dt,
                nt,
                step_ratio,
                accuracy,
                pml_width,
                n_batch,
                None,
                None,
                1,
                "device",
                ".",
                False,
                wfc,
                wfp,
                psiy,
                psix,
                zetay,
                zetax,
                wfcsc,
                wfpsc,
                psiysc,
                psixsc,
                zetaysc,
                zetaxsc,
            )
            out = scalar_born_func(*inputs)
            v = [torch.randn_like(o.contiguous()) for o in out]
            return torch.autograd.grad(
                out,
                [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
                grad_outputs=v,
            )

        torch.manual_seed(1)
        out_compiled = wrap(False)
        torch.manual_seed(1)
        out_python = wrap(True)
        for oc, op in zip(out_compiled, out_python):
            assert torch.allclose(oc, op)


def test_scalarbornfunc() -> None:
    """Test scalar_born_func with different time steps."""
    run_scalarbornfunc(nt=4)
    run_scalarbornfunc(nt=5)


def test_born_gradcheck() -> None:
    """Test gradcheck in a 2D model with Born propagator."""
    run_born_gradcheck(propagator=scalarbornprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((4,), (5,)),
        ((4, 3), (5, 5)),
        ((4, 3, 3), (6, 6, 6)),
    ],
)
def test_born_gradcheck_2nd_order(nx, dx) -> None:
    """Test gradcheck with a 2nd order accurate propagator."""
    run_born_gradcheck(
        propagator=scalarbornprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2}
    )


def test_born_gradcheck_6th_order() -> None:
    """Test gradcheck with a 6th order accurate propagator."""
    run_born_gradcheck(propagator=scalarbornprop, prop_kwargs={"accuracy": 6})


def test_born_gradcheck_8th_order() -> None:
    """Test gradcheck with a 8th order accurate propagator."""
    run_born_gradcheck(propagator=scalarbornprop, prop_kwargs={"accuracy": 8})


def test_born_gradcheck_cfl() -> None:
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        dt=0.002,
        atol=2e-7,
        rtol=1e-8,
        nt_add=100,
    )


def test_born_gradcheck_odd_timesteps() -> None:
    """Test gradcheck with one more timestep."""
    run_born_gradcheck(propagator=scalarbornprop, nt_add=1)


def test_born_gradcheck_one_shot() -> None:
    """Test gradcheck with one shot."""
    run_born_gradcheck(propagator=scalarbornprop, num_shots=1)


def test_born_gradcheck_no_sources() -> None:
    """Test gradcheck with no sources."""
    run_born_gradcheck(propagator=scalarbornprop, num_sources_per_shot=0)


def test_born_gradcheck_no_receivers() -> None:
    """Test gradcheck with no receivers."""
    run_born_gradcheck(propagator=scalarbornprop, num_receivers_per_shot=0)


def test_gradcheck_no_scatter_receivers() -> None:
    """Test gradcheck with no scatter receivers."""
    run_born_gradcheck(propagator=scalarbornprop, num_scatter_receivers_per_shot=0)


def test_born_gradcheck_survey_pad() -> None:
    """Test gradcheck with survey_pad."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        survey_pad=0,
        provide_wavefields=False,
    )


def test_born_gradcheck_partial_wavefields() -> None:
    """Test gradcheck with wavefields that do not cover the model."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        origin=(0, 4),
        nx=(12, 9),
        wavefield_size=(4 + 6, 1 + 6),
    )


def test_born_gradcheck_chained() -> None:
    """Test gradcheck when two propagators are chained."""
    run_born_gradcheck(propagator=scalarbornpropchained)


def test_born_gradcheck_negative() -> None:
    """Test gradcheck with negative velocity."""
    run_born_gradcheck(c=-1500, propagator=scalarbornprop)


def test_born_gradcheck_zero() -> None:
    """Test gradcheck with zero velocity."""
    run_born_gradcheck(c=0, dc=0, propagator=scalarbornprop)


def test_born_gradcheck_different_pml() -> None:
    """Test gradcheck with different pml widths."""
    run_born_gradcheck(propagator=scalarbornprop, pml_width=[0, 1, 5, 10], atol=5e-8)


def test_born_gradcheck_no_pml() -> None:
    """Test gradcheck with no pml."""
    run_born_gradcheck(propagator=scalarbornprop, pml_width=0, atol=2e-8)


def test_born_gradcheck_different_dx() -> None:
    """Test gradcheck with different dx values."""
    run_born_gradcheck(propagator=scalarbornprop, dx=(4, 5), atol=5e-8)


def test_born_gradcheck_single_cell() -> None:
    """Test gradcheck with a single model cell."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        nx=(1, 1),
        num_shots=1,
        num_sources_per_shot=1,
        num_receivers_per_shot=1,
        num_scatter_receivers_per_shot=1,
    )


def test_born_gradcheck_big() -> None:
    """Test gradcheck with a big model."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        nx=(5 + 2 * (3 + 3 * 2), 4 + 2 * (3 + 3 * 2)),
    )


def test_negative_vel(
    c=1500,
    dc=100,
    freq=25,
    dx=(5, 5),
    dt=0.005,
    nx=(50, 50),
    num_shots=2,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    propagator=scalarbornprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Test propagation with a zero or negative velocity or dt."""
    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    for python in [True, False]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["python_backend"] = python

        nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
        sources = _set_sources(x_s, freq, dt, nt, dtype)

        # Positive velocity
        model = (
            torch.ones(*nx, device=device, dtype=dtype) * c
            + torch.randn(*nx, device=device, dtype=dtype) * dc
        )
        scatter = torch.randn(*nx, dtype=dtype).to(device)
        out_positive = propagator(
            model,
            scatter,
            dx.tolist(),
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            x_r,
            prop_kwargs=prop_kwargs,
        )

        # Negative velocity
        out = propagator(
            -model,
            scatter,
            dx.tolist(),
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.allclose(out_positive[0], out[0])
        assert torch.allclose(out_positive[6], -out[6])
        assert torch.allclose(out_positive[-1], -out[-1])

        # Negative dt
        out = propagator(
            model,
            scatter,
            dx.tolist(),
            -dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.allclose(out_positive[0], out[0])
        assert torch.allclose(out_positive[6], out[6])
        assert torch.allclose(out_positive[-1], out[-1])

        # Zero velocity
        out = propagator(
            torch.zeros_like(model),
            scatter,
            dx.tolist(),
            -dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.allclose(out[0], torch.zeros_like(out[0]))
        assert torch.allclose(out[6], torch.zeros_like(out[6]))


def test_born_gradcheck_only_v_2d() -> None:
    """Test gradcheck with only v requiring gradient."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        scatter_requires_grad=False,
        source_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        wavefieldsc_0_requires_grad=False,
        wavefieldsc_m1_requires_grad=False,
        psisc_m1_requires_grad=False,
        zetasc_m1_requires_grad=False,
    )


def test_born_gradcheck_only_scatter_2d() -> None:
    """Test gradcheck with only scatter requiring gradient."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        v_requires_grad=False,
        source_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        wavefieldsc_0_requires_grad=False,
        wavefieldsc_m1_requires_grad=False,
        psisc_m1_requires_grad=False,
        zetasc_m1_requires_grad=False,
    )


def test_born_gradcheck_only_source_2d() -> None:
    """Test gradcheck with only source requiring gradient."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        v_requires_grad=False,
        scatter_requires_grad=False,
        source_requires_grad=True,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        wavefieldsc_0_requires_grad=False,
        wavefieldsc_m1_requires_grad=False,
        psisc_m1_requires_grad=False,
        zetasc_m1_requires_grad=False,
    )


def test_born_gradcheck_only_wavefield_0_2d() -> None:
    """Test gradcheck with only wavefield_0 requiring gradient."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        v_requires_grad=False,
        scatter_requires_grad=False,
        source_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        wavefieldsc_0_requires_grad=False,
        wavefieldsc_m1_requires_grad=False,
        psisc_m1_requires_grad=False,
        zetasc_m1_requires_grad=False,
    )


def test_born_gradcheck_only_wavefieldsc_0_2d() -> None:
    """Test gradcheck with only wavefieldsc_0 requiring gradient."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        v_requires_grad=False,
        scatter_requires_grad=False,
        source_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        wavefieldsc_m1_requires_grad=False,
        psisc_m1_requires_grad=False,
        zetasc_m1_requires_grad=False,
    )


def test_born_gradcheck_v_batched() -> None:
    """Test gradcheck using a different velocity for each shot."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        c=torch.tensor([[[1500.0]], [[1600.0]]]),
    )


def test_born_gradcheck_scatter_batched() -> None:
    """Test gradcheck using a different scatter for each shot."""
    run_born_gradcheck(
        propagator=scalarbornprop,
        dscatter=torch.tensor([[[150.0]], [[100.0]]]),
    )


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 3), (5, 5)),
        ((5, 3, 3), (6, 6, 6)),
    ],
)
def test_storage(
    nx,
    dx,
    c=1500,
    dc=100,
    freq=25,
    dt=0.005,
    num_shots=2,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    num_scatter_receivers_per_shot=2,
    propagator=scalarbornprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Test gradients with different storage options."""
    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if prop_kwargs is None:
        prop_kwargs = {}
    if "pml_width" not in prop_kwargs:
        prop_kwargs["pml_width"] = 3

    model = (
        torch.ones(*nx, device=device, dtype=dtype) * c
        + torch.randn(*nx, device=device, dtype=dtype) * dc
    )
    scatter = torch.randn(*nx, dtype=dtype).to(device) * dc

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    x_rsc = _set_coords(
        num_shots,
        num_scatter_receivers_per_shot,
        nx.tolist(),
        "bottom",
    )
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    # Store uncompressed in device memory (default)
    modeld = model.clone()
    modeld.requires_grad_()
    scatterd = scatter.clone()
    scatterd.requires_grad_()
    out = propagator(
        modeld,
        scatterd,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_rsc,
        x_r,
        prop_kwargs=prop_kwargs,
    )

    ((out[-1] ** 2).sum() + (out[-2] ** 2).sum()).backward()

    modes = ["device", "disk", "none"]
    if model.is_cuda:
        modes.append("cpu")

    atol_model = [2, 0.02, 2e-3]
    atol_scatter = [2e-3, 1e-4, 1e-5]

    for mode in modes:
        for compression in [False, True]:
            if mode == "device" and not compression:  # Default
                continue
            modeli = model.clone()
            modeli.requires_grad_()
            scatteri = scatter.clone()
            scatteri.requires_grad_()
            prop_kwargs = dict(prop_kwargs)
            prop_kwargs["storage_mode"] = mode
            prop_kwargs["storage_compression"] = compression
            out = propagator(
                modeli,
                scatteri,
                dx.tolist(),
                dt,
                sources["amplitude"],
                sources["locations"],
                x_rsc,
                x_r,
                prop_kwargs=prop_kwargs,
            )

            ((out[-1] ** 2).sum() + (out[-2] ** 2).sum()).backward()

            if mode != "none":
                if compression:
                    assert torch.allclose(
                        modeld.grad, modeli.grad, atol=atol_model[len(nx) - 1]
                    )
                    assert torch.allclose(
                        scatterd.grad, scatteri.grad, atol=atol_scatter[len(nx) - 1]
                    )
                else:
                    assert torch.allclose(
                        modeld.grad,
                        modeli.grad,
                        atol=modeld.grad.detach().abs().max().item() * 1e-5,
                    )
                    assert torch.allclose(
                        scatterd.grad,
                        scatteri.grad,
                        atol=scatterd.grad.detach().abs().max().item() * 1e-5,
                    )


def run_born_gradcheck(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 100,
    dscatter: Union[float, torch.Tensor] = 100,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.001,
    nx: Tuple[int, ...] = (4, 3),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    num_scatter_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Union[int, List[int]] = 3,
    survey_pad: Optional[Union[int, List[int]]] = None,
    origin: Optional[List[int]] = None,
    wavefield_size: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.double,
    v_requires_grad: bool = True,
    scatter_requires_grad: bool = True,
    source_requires_grad: bool = True,
    provide_wavefields: bool = True,
    wavefield_0_requires_grad: bool = True,
    wavefield_m1_requires_grad: bool = True,
    psi_m1_requires_grad: bool = True,
    zeta_m1_requires_grad: bool = True,
    wavefieldsc_0_requires_grad: bool = True,
    wavefieldsc_m1_requires_grad: bool = True,
    psisc_m1_requires_grad: bool = True,
    zetasc_m1_requires_grad: bool = True,
    atol: float = 2e-8,
    rtol: float = 1e-5,
    nt_add: int = 0,
) -> None:
    """Run PyTorch's gradcheck."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(c, torch.Tensor):
        c = c.to(device)
    if isinstance(dscatter, torch.Tensor):
        dscatter = dscatter.to(device)
    model = torch.ones(*nx, device=device, dtype=dtype) * c
    model += torch.rand(*model.shape, device=device, dtype=dtype) * dc
    scatter = torch.rand(*nx, device=device, dtype=dtype) * dscatter
    min_c = model.abs().min().item()

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    if min_c != 0:
        nt = int((2 * torch.norm(nx.float() * dx) / min_c + 0.1 + 2 / freq) / dt)
    else:
        nt = int((2 * torch.norm(nx.float() * dx) / 1500 + 0.1 + 2 / freq) / dt)
    nt += nt_add

    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        sources = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources["amplitude"].requires_grad_(source_requires_grad)
        nt = None
    else:
        sources = {"amplitude": None, "locations": None}
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
    else:
        x_r = None
    if num_scatter_receivers_per_shot > 0:
        x_r_sc = _set_coords(num_shots, num_scatter_receivers_per_shot, nx.tolist())
    else:
        x_r_sc = None
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(2 * ndim)]
    if provide_wavefields:
        if wavefield_size is None:
            wavefield_size = tuple(
                nx[i] + pml_width[2 * i] + pml_width[2 * i + 1] for i in range(ndim)
            )
        wavefield_0 = torch.zeros(
            num_shots,
            *wavefield_size,
            dtype=dtype,
            device=device,
        )
        wavefield_m1 = torch.zeros_like(wavefield_0)
        psi_m1 = [torch.zeros_like(wavefield_0) for _ in range(ndim)]
        zeta_m1 = [torch.zeros_like(wavefield_0) for _ in range(ndim)]
        wavefield_sc_0 = torch.zeros_like(wavefield_0)
        wavefield_sc_m1 = torch.zeros_like(wavefield_0)
        psi_sc_m1 = [torch.zeros_like(wavefield_0) for _ in range(ndim)]
        zeta_sc_m1 = [torch.zeros_like(wavefield_0) for _ in range(ndim)]
        wavefield_0.requires_grad_(wavefield_0_requires_grad)
        wavefield_m1.requires_grad_(wavefield_m1_requires_grad)
        wavefield_sc_0.requires_grad_(wavefieldsc_0_requires_grad)
        wavefield_sc_m1.requires_grad_(wavefieldsc_m1_requires_grad)
        for i in range(ndim):
            psi_m1[i].requires_grad_(psi_m1_requires_grad)
            zeta_m1[i].requires_grad_(zeta_m1_requires_grad)
            psi_sc_m1[i].requires_grad_(psisc_m1_requires_grad)
            zeta_sc_m1[i].requires_grad_(zetasc_m1_requires_grad)
    else:
        wavefield_0 = None
        wavefield_m1 = None
        psi_m1 = []
        zeta_m1 = []
        wavefield_sc_0 = None
        wavefield_sc_m1 = None
        psi_sc_m1 = []
        zeta_sc_m1 = []

    psi_m1 = [None] * (3 - len(psi_m1)) + psi_m1
    zeta_m1 = [None] * (3 - len(zeta_m1)) + zeta_m1
    psi_sc_m1 = [None] * (3 - len(psi_sc_m1)) + psi_sc_m1
    zeta_sc_m1 = [None] * (3 - len(zeta_sc_m1)) + zeta_sc_m1

    model.requires_grad_(v_requires_grad)
    scatter.requires_grad_(scatter_requires_grad)
    if prop_kwargs is None:
        prop_kwargs = {}

    out_idxs = []

    def wrap(python):
        prop_kwargs["python_backend"] = python
        inputs = (
            model,
            scatter,
            dx.tolist(),
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            x_r_sc,
            prop_kwargs,
            pml_width,
            survey_pad,
            origin,
            wavefield_0,
            wavefield_m1,
            *psi_m1,
            *zeta_m1,
            wavefield_sc_0,
            wavefield_sc_m1,
            *psi_sc_m1,
            *zeta_sc_m1,
            nt,
            1,
            True,
        )
        out = propagator(*inputs)
        if len(out_idxs) == 0:
            for i, o in enumerate(out):
                if o.requires_grad:
                    out_idxs.append(i)
        out = [out[i] for i in out_idxs]
        v = [torch.randn_like(o.contiguous()) for o in out]
        return tuple(out) + torch.autograd.grad(
            out,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=v,
        )

    torch.manual_seed(1)
    out_python = wrap(True)
    torch.manual_seed(1)
    out_compiled = wrap(False)
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 4), (5, 5)),
        ((5, 4, 4), (6, 6, 6)),
    ],
)
def test_reciprocity(nx, dx):
    """Test source-receiver reciprocity."""
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 25
    nt = 250
    dt = 0.001
    peak_time = 0.05
    c = 1500
    dc = 150
    dscatter = 150

    torch.manual_seed(1)
    model = (
        torch.ones(*nx, device=device, dtype=dtype) * c
        + torch.rand(*nx, device=device, dtype=dtype) * dc
    )
    scatter = torch.rand(*nx, device=device, dtype=dtype) * dscatter

    source_amplitudes = ricker(freq, nt, dt, peak_time, dtype=dtype).reshape(1, 1, -1)

    loc_a = torch.tensor([d // 4 for d in nx], device=device).reshape(1, 1, -1)
    loc_b = torch.tensor([d - d // 4 for d in nx], device=device).reshape(1, 1, -1)

    prop_kwargs = {"accuracy": 4}

    """Helper for test_reciprocity."""
    # Run 1: src_comp at A, rec_comp at B
    kwargs1 = {
        "source_amplitudes": source_amplitudes,
        "source_locations": loc_a,
        "bg_receiver_locations": loc_b,
    }

    outputs1 = scalarbornprop(
        model, scatter, list(dx), dt, **kwargs1, prop_kwargs=prop_kwargs
    )

    rec1 = outputs1[-2]

    # Run 2: rec_comp at B, src_comp at A
    kwargs2 = {
        "source_amplitudes": source_amplitudes,
        "source_locations": loc_b,
        "bg_receiver_locations": loc_a,
    }

    outputs2 = scalarbornprop(
        model, scatter, list(dx), dt, **kwargs2, prop_kwargs=prop_kwargs
    )

    rec2 = outputs2[-2]

    assert torch.allclose(rec1, rec2, atol=1e-3)

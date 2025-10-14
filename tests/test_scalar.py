"""Tests for deepwave.scalar."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
from test_utils import (
    _set_coords,
    _set_sources,
    direct_1d,
    direct_2d_approx,
    direct_3d,
    run_reciprocity_check,
    scattered,
)

from deepwave import IGNORE_LOCATION, Scalar, scalar
from deepwave.backend_utils import USE_OPENMP
from deepwave.common import cfl_condition, downsample, upsample
from deepwave.wavelets import ricker

torch._dynamo.config.cache_size_limit = 256  # noqa: SLF001
torch.autograd.set_detect_anomaly(True)


def scalarprop(
    model: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
    receiver_locations: Optional[torch.Tensor],
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
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    functional: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Wraps the scalar propagator."""
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

    if functional:
        return scalar(
            model,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            psiz_m1=psiz_m1,
            psiy_m1=psiy_m1,
            psix_m1=psix_m1,
            zetaz_m1=zetaz_m1,
            zetay_m1=zetay_m1,
            zetax_m1=zetax_m1,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            **prop_kwargs,
        )

    prop = Scalar(model, dx)
    return prop(
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        wavefield_0=wavefield_0,
        wavefield_m1=wavefield_m1,
        psiy_m1=psiy_m1,
        psix_m1=psix_m1,
        zetay_m1=zetay_m1,
        zetax_m1=zetax_m1,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def scalarpropchained(
    model: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
    receiver_locations: Optional[torch.Tensor],
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
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    functional: bool = True,
    n_chained: int = 2,
) -> Tuple[torch.Tensor, ...]:
    """Wraps multiple scalar propagators chained sequentially."""
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
        wfc, wfp, psiy, psix, zetay, zetax, segment_receiver_amplitudes = scalar(
            model,
            dx,
            dt,
            source_amplitudes=segment_source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wfc,
            wavefield_m1=wfp,
            psiy_m1=psiy,
            psix_m1=psix,
            zetay_m1=zetay,
            zetax_m1=zetax,
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

    if receiver_locations is not None:
        receiver_amplitudes = downsample(receiver_amplitudes, step_ratio)

    return wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes


def test_pml_width_list() -> None:
    """Verify that PML width supplied as int and list give same result."""
    _, actual_int = run_direct(propagator=scalarprop, prop_kwargs={"pml_width": 20})
    _, actual_list = run_direct(
        propagator=scalarprop,
        prop_kwargs={"pml_width": [20, 20, 20, 20]},
    )
    assert torch.allclose(actual_int, actual_list)


def test_python_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_direct(propagator=scalarprop, prop_kwargs={"python_backend": "jit"})
        run_direct(propagator=scalarprop, prop_kwargs={"python_backend": "compile"})


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((50,), (5,)),
        ((50, 50), (5, 5)),
        ((10, 10, 10), (5, 5, 5)),
    ],
)
def test_direct(nx, dx):
    """Test propagation in a constant model."""
    if len(nx) == 1:
        expected_diffs = {2: 315, 4: 35, 6: 10, 8: 8}
    elif len(nx) == 2:
        expected_diffs = {2: 16, 4: 1.48, 6: 0.22, 8: 0.048}
    else:
        expected_diffs = {2: 1.4, 4: 0.5, 6: 0.2, 8: 0.04}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [True, False]:
            expected, actual = run_direct(
                nx=nx,
                dx=dx,
                propagator=scalarprop,
                prop_kwargs={"python_backend": python, "accuracy": accuracy},
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        assert torch.allclose(
            actuals[True], actuals[False], atol=5e-4 * actuals[True].abs().max().item()
        )


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((50,), (5,)),
        ((50, 50), (5, 5)),
        ((10, 10, 10), (5, 5, 5)),
    ],
)
def test_wavefield_decays(nx, dx) -> None:
    """Test that the PML causes the wavefield amplitude to decay."""
    for python in [True, False]:
        out = run_forward(
            propagator=scalarprop,
            nx=nx,
            dx=dx,
            nt=25000 if len(nx) == 1 else 2500,
            prop_kwargs={"python_backend": python},
        )
        for outi in out[:-1]:
            assert outi.norm() < 4e-5


def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        for python in [True, False]:
            actual_cpu = run_forward(
                propagator=scalarprop,
                device=torch.device("cpu"),
                prop_kwargs={"python_backend": python},
            )
            actual_gpu = run_forward(
                propagator=scalarprop,
                device=torch.device("cuda"),
                prop_kwargs={"python_backend": python},
            )
            # Check wavefields
            for cpui, gpui in zip(actual_cpu[:-1], actual_gpu[:-1]):
                assert torch.allclose(cpui, gpui.cpu(), atol=2e-6)
            # Check receiver amplitudes
            cpui = actual_cpu[-1]
            gpui = actual_gpu[-1]
            assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_direct_2d_module() -> None:
    """Test propagation with the Module interface."""
    expected, actual = run_direct(propagator=scalarprop, functional=False)
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 1.48


def test_scatter_2d() -> None:
    """Test propagation in a 2D model with a point scatterer."""
    expected, actual = run_scatter_2d(
        propagator=scalarprop,
        dt=0.001,
        prop_kwargs={"pml_width": 30},
    )
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 0.008


def test_v_batched() -> None:
    """Test forward using a different velocity for each shot."""
    expected_diff = 1.3
    actuals = {}
    for python in [True, False]:
        expected, actual = run_direct(
            c=torch.tensor([[[1500.0]], [[1600.0]]]),
            propagator=scalarprop,
            prop_kwargs={"python_backend": python},
        )
        diff = (expected - actual.cpu()).flatten()
        assert diff.norm().item() < expected_diff
        actuals[python] = actual
    assert torch.allclose(
        actuals[True], actuals[False], atol=5e-4 * actuals[True].abs().max().item()
    )


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
    propagator=scalarprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Test forward and backward propagation when a source and receiver are unused."""
    assert num_shots > 1
    assert num_sources_per_shot > 1
    assert num_receivers_per_shot > 1
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

        nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
        sources = _set_sources(x_s, freq, dt, nt, dtype)

        n_warmup = 3 if python else 1

        for _warmup in range(n_warmup):
            # Forward with source and receiver ignored
            modeli = model.clone()
            modeli.requires_grad_()
            for i in range(num_shots):
                x_s[i, i, :] = IGNORE_LOCATION
                x_r[i, i, :] = IGNORE_LOCATION
            sources["locations"] = x_s
            out_ignored = propagator(
                modeli,
                dx.tolist(),
                dt,
                sources["amplitude"],
                sources["locations"],
                x_r,
                prop_kwargs=prop_kwargs,
            )

            (out_ignored[-1] ** 2).sum().backward()

        # Forward with amplitudes of sources that will be ignored set to zero
        modelf = model.clone()
        modelf.requires_grad_()
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
        for i in range(num_shots):
            sources["amplitude"][i, i].fill_(0)
        sources["locations"] = x_s
        out_filled = propagator(
            modelf,
            dx.tolist(),
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs=prop_kwargs,
        )
        # Set receiver amplitudes of receiver that will be ignored to zero
        for i in range(num_shots):
            out_filled[-1][i, i].fill_(0)

        (out_filled[-1] ** 2).sum().backward()

        for ofi, oii in zip(out_filled, out_ignored):
            assert torch.allclose(ofi, oii)

        assert torch.allclose(modelf.grad, modeli.grad)


def run_scalarfunc(nt: int = 3) -> None:
    """Runs scalar_func for testing purposes."""
    from deepwave.scalar import scalar_func

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
    source_amplitudes = torch.randn(
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
    c.requires_grad_()
    source_amplitudes.requires_grad_()
    wfc.requires_grad_()
    wfp.requires_grad_()
    psiy.requires_grad_()
    psix.requires_grad_()
    zetay.requires_grad_()
    zetax.requires_grad_()
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

    def wrap(python):
        inputs = (
            python,
            c,
            source_amplitudes,
            [
                ay,
                by,
                dbydy,
                ax,
                bx,
                dbxdx,
            ],
            sources_i,
            receivers_i,
            [
                dy,
                dx,
            ],
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_batch,
            None,
            None,
            1,
            *[
                wfc,
                wfp,
                psiy,
                psix,
                zetay,
                zetax,
            ],
        )
        out = scalar_func(*inputs)
        v = [torch.randn_like(o.contiguous()) for o in out]
        grads = torch.autograd.grad(
            out,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=v,
            create_graph=True,
        )
        w = [torch.randn_like(o.contiguous()) for o in grads]
        grad_grads = torch.autograd.grad(
            grads,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=w,
        )
        return grads + grad_grads

    torch.manual_seed(1)
    out_compiled = wrap(False)
    torch.manual_seed(1)
    out_python = wrap("jit")
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op)


def test_scalarfunc() -> None:
    """Test scalar_func with different time steps."""
    run_scalarfunc(nt=4)
    run_scalarfunc(nt=5)


def test_gradcheck() -> None:
    """Test gradcheck in a 2D model."""
    run_gradcheck(propagator=scalarprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((4,), (5,)),
        ((4, 3), (5, 5)),
        ((4, 3, 3), (6, 6, 6)),
    ],
)
def test_gradcheck_2nd_order(nx, dx) -> None:
    """Test gradcheck with a 2nd order accurate propagator."""
    run_gradcheck(propagator=scalarprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2})


def test_gradcheck_6th_order() -> None:
    """Test gradcheck with a 6th order accurate propagator."""
    run_gradcheck(propagator=scalarprop, prop_kwargs={"accuracy": 6})


def test_gradcheck_8th_order() -> None:
    """Test gradcheck with a 8th order accurate propagator."""
    run_gradcheck(propagator=scalarprop, prop_kwargs={"accuracy": 8})


def test_gradcheck_cfl() -> None:
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_gradcheck(propagator=scalarprop, dt=0.002, atol=5e-8, gradgrad=False)


def test_gradcheck_cfl_gradgrad() -> None:
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_gradcheck(
        propagator=scalarprop,
        dt=0.002,
        prop_kwargs={"time_taper": True},
        gradgrad=True,
        source_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        only_receivers_out=True,
    )


def test_gradcheck_odd_timesteps() -> None:
    """Test gradcheck with one more timestep."""
    run_gradcheck(propagator=scalarprop, nt_add=1)


def test_gradcheck_one_shot() -> None:
    """Test gradcheck with one shot."""
    run_gradcheck(propagator=scalarprop, num_shots=1)


def test_gradcheck_no_sources() -> None:
    """Test gradcheck with no sources."""
    run_gradcheck(propagator=scalarprop, num_sources_per_shot=0)


def test_gradcheck_no_receivers() -> None:
    """Test gradcheck with no receivers."""
    run_gradcheck(propagator=scalarprop, num_receivers_per_shot=0)


def test_gradcheck_survey_pad() -> None:
    """Test gradcheck with survey_pad."""
    run_gradcheck(
        propagator=scalarprop,
        survey_pad=0,
        nx=(12, 9),
        provide_wavefields=False,
    )


def test_gradcheck_partial_wavefields() -> None:
    """Test gradcheck with wavefields that do not cover the model."""
    run_gradcheck(
        propagator=scalarprop,
        origin=(0, 4),
        nx=(12, 9),
        wavefield_size=(4 + 6, 1 + 6),
    )


def test_gradcheck_chained() -> None:
    """Test gradcheck when two propagators are chained."""
    run_gradcheck(propagator=scalarpropchained)


def test_gradcheck_negative() -> None:
    """Test gradcheck with negative velocity."""
    run_gradcheck(c=-1500, propagator=scalarprop)


def test_gradcheck_zero() -> None:
    """Test gradcheck with zero velocity."""
    run_gradcheck(c=0, dc=0, propagator=scalarprop)


def test_gradcheck_different_pml() -> None:
    """Test gradcheck with different pml widths."""
    run_gradcheck(propagator=scalarprop, pml_width=[0, 1, 5, 10], atol=5e-8)


def test_gradcheck_no_pml() -> None:
    """Test gradcheck with no pml."""
    run_gradcheck(propagator=scalarprop, pml_width=0)


def test_gradcheck_different_dx() -> None:
    """Test gradcheck with different dx values."""
    run_gradcheck(propagator=scalarprop, dx=(4, 5), dt=0.0005, atol=4e-8)


def test_gradcheck_single_cell() -> None:
    """Test gradcheck with a single model cell."""
    run_gradcheck(
        propagator=scalarprop,
        nx=(1, 1),
        num_shots=1,
        num_sources_per_shot=1,
        num_receivers_per_shot=1,
    )


def test_gradcheck_big() -> None:
    """Test gradcheck with a big model."""
    run_gradcheck(
        propagator=scalarprop,
        nx=(5 + 2 * (3 + 3 * 2), 4 + 2 * (3 + 3 * 2)),
        atol=2e-8,
        gradgrad=False,
    )


def test_gradcheck_big_gradgrad() -> None:
    """Test gradcheck with a big model."""
    run_gradcheck(
        propagator=scalarprop,
        nx=(5 + 2 * (3 + 3 * 2), 4 + 2 * (3 + 3 * 2)),
        prop_kwargs={"time_taper": True},
        gradgrad=True,
        source_requires_grad=True,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
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
    propagator=scalarprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Test propagation with a zero or negative velocity or dt."""
    nx = torch.tensor(nx).long()
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
        out_positive = propagator(
            model,
            dx,
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs=prop_kwargs,
        )

        # Negative velocity
        out = propagator(
            -model,
            dx,
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.allclose(out_positive[0], out[0])
        assert torch.allclose(out_positive[-1], out[-1])

        # Negative dt
        out = propagator(
            model,
            dx,
            -dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.allclose(out_positive[0], out[0])
        assert torch.allclose(out_positive[-1], out[-1])

        # Zero velocity
        out = propagator(
            torch.zeros_like(model),
            dx,
            -dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.allclose(out[0], torch.zeros_like(out[0]))


def test_gradcheck_only_v_2d() -> None:
    """Test gradcheck with only v requiring gradient."""
    run_gradcheck(
        propagator=scalarprop,
        source_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
    )


def test_gradcheck_only_source_2d() -> None:
    """Test gradcheck with only source requiring gradient."""
    run_gradcheck(
        propagator=scalarprop,
        v_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
    )


def test_gradcheck_only_wavefield_0_2d() -> None:
    """Test gradcheck with only wavefield_0 requiring gradient."""
    run_gradcheck(
        propagator=scalarprop,
        v_requires_grad=False,
        source_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
    )


def test_gradcheck_v_batched() -> None:
    """Test gradcheck using a different velocity for each shot."""
    run_gradcheck(propagator=scalarprop, c=torch.tensor([[[1500.0]], [[1600.0]]]))


def run_direct(
    c: Union[float, torch.Tensor] = 1500,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.0001,
    nx: Tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = scalarprop,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the expected and actual direct waveform at a point."""
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
    model = torch.ones(*nx, device=device, dtype=dtype) * c

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    nt = int((2 * torch.norm(nx.float() * dx) / min_c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 1:
        direct = direct_1d
    elif len(nx) == 2:
        direct = direct_2d_approx
    elif len(nx) == 3:
        direct = direct_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(num_shots, num_receivers_per_shot, nt, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver, :] += direct(
                    x_r[shot, receiver],
                    x_s[shot, source],
                    dx,
                    dt,
                    shot_c[shot],
                    -sources["amplitude"][shot, source, :],
                ).to(dtype)

    actual = propagator(
        model,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )[-1]

    return expected, actual


def run_forward(
    c: Union[float, torch.Tensor] = 1500,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.004,
    nx: Tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    dc: float = 100,
    nt: Optional[int] = None,
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

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    if nt is None:
        nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    return propagator(
        model,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )


def run_scatter(
    c: float,
    dc: float,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the expected and actual point scattered waveform at point."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.ones(*nx, device=device, dtype=dtype) * c
    model_const = model.clone()

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
    x_p = _set_coords(1, 1, nx.tolist(), "middle")[0, 0]
    model[torch.split((x_p).long(), 1)] += dc
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
                    c,
                    dc,
                    -sources["amplitude"][shot, source, :],
                    ndim,
                ).to(dtype)

    y_const = propagator(
        model_const,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        prop_kwargs=prop_kwargs,
    )[-1]
    y = propagator(
        model,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        prop_kwargs=prop_kwargs,
    )[-1]

    actual = y - y_const

    return expected, actual


def run_scatter_2d(
    c=1500,
    dc=150,
    freq=25,
    dx=(5, 5),
    dt=0.0001,
    nx=(50, 50),
    num_shots=2,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    propagator=None,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Runs run_scatter with default parameters for 2D."""
    return run_scatter(
        c,
        dc,
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
    )


def run_gradcheck(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 100,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.001,
    nx: Tuple[int, ...] = (4, 3),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Union[int, List[int]] = 3,
    survey_pad: Optional[Union[int, List[int]]] = None,
    origin: Optional[List[int]] = None,
    wavefield_size: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.double,
    v_requires_grad: bool = True,
    source_requires_grad: bool = True,
    provide_wavefields: bool = True,
    wavefield_0_requires_grad: bool = True,
    wavefield_m1_requires_grad: bool = True,
    psi_m1_requires_grad: bool = True,
    zeta_m1_requires_grad: bool = True,
    only_receivers_out: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    gradgrad: bool = True,
    nt_add: int = 0,
) -> None:
    """Run PyTorch's gradcheck to test the gradient."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(c, torch.Tensor):
        min_c = c.abs().min().item()
        c = c.to(device)
    else:
        min_c = abs(c)
    model = (
        torch.ones(*nx, device=device, dtype=dtype) * c
        + torch.rand(*nx, device=device, dtype=dtype) * dc
    )

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
        wavefield_0.requires_grad_(wavefield_0_requires_grad)
        wavefield_m1.requires_grad_(wavefield_m1_requires_grad)
        for i in range(ndim):
            psi_m1[i].requires_grad_(psi_m1_requires_grad)
            zeta_m1[i].requires_grad_(zeta_m1_requires_grad)
    else:
        wavefield_0 = None
        wavefield_m1 = None
        psi_m1 = []
        zeta_m1 = []

    psi_m1 = [None] * (3 - len(psi_m1)) + psi_m1
    zeta_m1 = [None] * (3 - len(zeta_m1)) + zeta_m1

    model.requires_grad_(v_requires_grad)
    if prop_kwargs is None:
        prop_kwargs = {}

    out_idxs = []
    grad_idxs = []

    def wrap(python):
        prop_kwargs["python_backend"] = python
        inputs = (
            model,
            dx,
            dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs,
            pml_width,
            survey_pad,
            origin,
            wavefield_0,
            wavefield_m1,
            *psi_m1,
            *zeta_m1,
            nt,
            1,
            True,
        )
        out = propagator(*inputs)
        if only_receivers_out:
            out = out[-1:]
        if len(out_idxs) == 0:
            for i, o in enumerate(out):
                if o.requires_grad:
                    out_idxs.append(i)
        out = [out[i] for i in out_idxs]
        v = [torch.randn_like(o.contiguous()) for o in out]
        grads = torch.autograd.grad(
            out,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=v,
            create_graph=gradgrad,
        )
        if not gradgrad:
            return grads

        if len(grad_idxs) == 0:
            for i, g in enumerate(grads):
                if g.requires_grad:
                    grad_idxs.append(i)
        grads = tuple([grads[i] for i in grad_idxs])
        if len(grads) == 0:
            return grads
        w = [torch.randn_like(o.contiguous()) for o in grads]
        grad_grads = torch.autograd.grad(
            grads,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=w,
        )
        return tuple(out) + grads + grad_grads

    torch.manual_seed(1)
    out_python = wrap("jit")
    torch.manual_seed(1)
    out_compiled = wrap(False)
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op, atol=atol, rtol=rtol)


# Tests for Scalar.__init__
def test_scalar_init_v_non_tensor() -> None:
    """Test that Scalar.__init__ raises TypeError if v is not a tensor."""
    with pytest.raises(TypeError, match=re.escape("v must be a torch.Tensor.")):
        Scalar([1, 2, 3], 1.0)


def test_scalar_init_v_requires_grad_non_bool() -> None:
    """Test that Scalar.__init__ raises TypeError if v_requires_grad is not a bool."""
    v = torch.ones(10, 10)
    with pytest.raises(
        TypeError, match=re.escape("v_requires_grad must be bool, got int")
    ):
        Scalar(v, 1.0, v_requires_grad=1)
    with pytest.raises(
        TypeError, match=re.escape("v_requires_grad must be bool, got float")
    ):
        Scalar(v, 1.0, v_requires_grad=0.5)
    with pytest.raises(
        TypeError, match=re.escape("v_requires_grad must be bool, got str")
    ):
        Scalar(v, 1.0, v_requires_grad="True")


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

    torch.manual_seed(1)
    model = (
        torch.ones(*nx, device=device, dtype=dtype) * c
        + torch.rand(*nx, device=device, dtype=dtype) * dc
    )

    source_amplitudes = ricker(freq, nt, dt, peak_time, dtype=dtype).reshape(1, 1, -1)

    loc_a = torch.tensor([d // 4 for d in nx], device=device).reshape(1, 1, -1)
    loc_b = torch.tensor([d - d // 4 for d in nx], device=device).reshape(1, 1, -1)

    prop_kwargs = {"accuracy": 2}

    run_reciprocity_check(
        [model],
        dx,
        dt,
        source_amplitudes,
        "",
        "",
        -1,
        -1,
        loc_a,
        loc_b,
        prop_kwargs,
        scalarprop,
    )

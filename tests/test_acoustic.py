"""Tests for deepwave.acoustic."""

from pathlib import Path
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
)

from deepwave import IGNORE_LOCATION, Acoustic, acoustic
from deepwave.backend_utils import USE_OPENMP
from deepwave.wavelets import ricker

torch._dynamo.config.cache_size_limit = 256  # noqa: SLF001
torch.autograd.set_detect_anomaly(True)


def acousticprop(
    model: torch.Tensor,
    rho: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_p: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Optional[Union[int, List[int]]] = None,
    survey_pad: Optional[Union[int, List[int]]] = None,
    origin: Optional[List[int]] = None,
    pressure_0: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    phi_z_0: Optional[torch.Tensor] = None,
    phi_y_0: Optional[torch.Tensor] = None,
    phi_x_0: Optional[torch.Tensor] = None,
    psi_z_0: Optional[torch.Tensor] = None,
    psi_y_0: Optional[torch.Tensor] = None,
    psi_x_0: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    gradient_mask: Optional[torch.Tensor] = None,
    functional: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Wraps the acoustic propagator."""
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
    if source_amplitudes_p is not None:
        source_amplitudes_p = source_amplitudes_p.to(device)
        source_locations_p = source_locations_p.to(device)
    if source_amplitudes_z is not None:
        source_amplitudes_z = source_amplitudes_z.to(device)
        source_locations_z = source_locations_z.to(device)
    if source_amplitudes_y is not None:
        source_amplitudes_y = source_amplitudes_y.to(device)
        source_locations_y = source_locations_y.to(device)
    if source_amplitudes_x is not None:
        source_amplitudes_x = source_amplitudes_x.to(device)
        source_locations_x = source_locations_x.to(device)
    if receiver_locations_p is not None:
        receiver_locations_p = receiver_locations_p.to(device)
    if receiver_locations_z is not None:
        receiver_locations_z = receiver_locations_z.to(device)
    if receiver_locations_y is not None:
        receiver_locations_y = receiver_locations_y.to(device)
    if receiver_locations_x is not None:
        receiver_locations_x = receiver_locations_x.to(device)

    if functional:
        return acoustic(
            model,
            rho,
            dx,
            dt,
            source_amplitudes_p=source_amplitudes_p,
            source_locations_p=source_locations_p,
            source_amplitudes_z=source_amplitudes_z,
            source_locations_z=source_locations_z,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_x=source_locations_x,
            receiver_locations_p=receiver_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            pressure_0=pressure_0,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            phi_z_0=phi_z_0,
            phi_y_0=phi_y_0,
            phi_x_0=phi_x_0,
            psi_z_0=psi_z_0,
            psi_y_0=psi_y_0,
            psi_x_0=psi_x_0,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            gradient_mask=gradient_mask,
            **prop_kwargs,
        )

    prop = Acoustic(model, rho, dx)
    return prop(
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        source_amplitudes_z=source_amplitudes_z,
        source_locations_z=source_locations_z,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_x=source_locations_x,
        receiver_locations_p=receiver_locations_p,
        receiver_locations_z=receiver_locations_z,
        receiver_locations_y=receiver_locations_y,
        receiver_locations_x=receiver_locations_x,
        pressure_0=pressure_0,
        vz_0=vz_0,
        vy_0=vy_0,
        vx_0=vx_0,
        phi_z_0=phi_z_0,
        phi_y_0=phi_y_0,
        phi_x_0=phi_x_0,
        psi_z_0=psi_z_0,
        psi_y_0=psi_y_0,
        psi_x_0=psi_x_0,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        gradient_mask=gradient_mask,
        **prop_kwargs,
    )


def test_python_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_forward(propagator=acousticprop, prop_kwargs={"python_backend": "jit"})
        run_forward(propagator=acousticprop, prop_kwargs={"python_backend": "compile"})


def _compute_gradient(mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    nt = 4
    v = torch.ones(5, 5, requires_grad=True)
    rho = torch.ones_like(v, requires_grad=True)
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 0] = 1
    locations = torch.tensor([[[2, 2]]], dtype=torch.long)
    outputs = acoustic(
        v,
        rho,
        1.0,
        0.001,
        source_amplitudes_p=source_amplitudes,
        source_locations_p=locations,
        receiver_locations_p=locations,
        nt=nt,
        pml_width=2,
        gradient_mask=mask,
    )
    receivers = outputs[-3:]
    loss = sum(o.sum() for o in receivers)
    loss.backward()
    assert v.grad is not None
    assert rho.grad is not None
    return v.grad.detach(), rho.grad.detach()


def _run_forward_for_storage(storage_dir: Path, mask: Optional[torch.Tensor]) -> int:
    storage_dir.mkdir()
    nt = 3
    v = torch.ones(5, 5, requires_grad=True)
    rho = torch.ones_like(v)
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 0] = 1
    locations = torch.tensor([[[2, 2]]], dtype=torch.long)
    outputs = acoustic(
        v,
        rho,
        1.0,
        0.001,
        source_amplitudes_p=source_amplitudes,
        source_locations_p=locations,
        receiver_locations_p=locations,
        nt=nt,
        pml_width=2,
        storage_mode="disk",
        storage_path=str(storage_dir),
        storage_compression=False,
        gradient_mask=mask,
    )
    assert outputs
    return sum(p.stat().st_size for p in storage_dir.rglob("*") if p.is_file())


def test_gradient_mask_reduces_storage(tmp_path: Path) -> None:
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[2, 2] = True
    masked_size = _run_forward_for_storage(tmp_path / "masked", mask)
    unmasked_size = _run_forward_for_storage(tmp_path / "unmasked", None)
    assert masked_size < (0.1 * unmasked_size)


def test_gradient_mask_zeroes_outside_mask() -> None:
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[2, 2] = True
    grad_v, grad_rho = _compute_gradient(mask)
    assert torch.count_nonzero(grad_v[mask]) == 1
    assert torch.count_nonzero(grad_v[~mask]) == 0
    assert torch.count_nonzero(grad_rho[mask]) == 1
    assert torch.count_nonzero(grad_rho[~mask]) == 0


def test_gradient_mask_default_computes_everywhere() -> None:
    full_mask = torch.ones(5, 5, dtype=torch.bool)
    grad_no_mask_v, grad_no_mask_rho = _compute_gradient(None)
    grad_full_mask_v, grad_full_mask_rho = _compute_gradient(full_mask)
    assert torch.count_nonzero(grad_no_mask_v) > 0
    assert torch.count_nonzero(grad_no_mask_rho) > 0
    assert torch.allclose(grad_no_mask_v, grad_full_mask_v, rtol=1e-30)
    assert torch.allclose(grad_no_mask_rho, grad_full_mask_rho, rtol=1e-30)


def test_gradient_mask_python_backend_raises() -> None:
    nt = 4
    v = torch.ones(5, 5)
    rho = torch.ones_like(v)
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 0] = 1
    locations = torch.tensor([[[2, 2]]], dtype=torch.long)
    mask = torch.ones(5, 5, dtype=torch.bool)
    with pytest.raises(
        RuntimeError,
        match=r"gradient_mask is not supported in the Python backend\.",
    ):
        acoustic(
            v,
            rho,
            1.0,
            0.001,
            source_amplitudes_p=source_amplitudes,
            source_locations_p=locations,
            receiver_locations_p=locations,
            nt=nt,
            pml_width=2,
            python_backend="eager",
            gradient_mask=mask,
        )


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
        expected_diffs = {2: 1, 4: 0.1, 6: 0.05, 8: 0.05}
    elif len(nx) == 2:
        expected_diffs = {2: 1.5, 4: 0.1, 6: 0.03, 8: 0.02}
    else:
        expected_diffs = {2: 0.2, 4: 0.05, 6: 0.02, 8: 0.02}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [True, False]:
            expected, actual = run_direct(
                nx=nx,
                dx=dx,
                propagator=acousticprop,
                prop_kwargs={"python_backend": python, "accuracy": accuracy},
            )
            diff = (expected - actual.cpu()).flatten()
            assert (
                diff.norm().item() / expected.flatten().norm().item()
                < expected_diffs[accuracy]
            )
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
        out = list(
            run_forward(
                propagator=acousticprop,
                nx=nx,
                dx=dx,
                nt=25000 if len(nx) == 1 else 2500,
                prop_kwargs={"python_backend": python},
            )
        )
        pmax = out[-(len(dx) + 1)].abs().max().item()
        vmax = [o.abs().max().item() for o in out[-len(dx) :]]
        out[0] /= pmax
        for i in range(len(dx)):
            out[1 + i] /= vmax[i]
            out[1 + len(dx) + i] /= vmax[i]
            out[1 + 2 * len(dx) + i] /= pmax
        for outi in out[: -(len(dx) + 1)]:
            assert outi.norm() < 5e-4


def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        for python in [True, False]:
            actual_cpu = run_forward(
                propagator=acousticprop,
                device=torch.device("cpu"),
                prop_kwargs={"python_backend": python},
                dtype=torch.double,
            )
            actual_gpu = run_forward(
                propagator=acousticprop,
                device=torch.device("cuda"),
                prop_kwargs={"python_backend": python},
                dtype=torch.double,
            )
            for cpui, gpui in zip(actual_cpu, actual_gpu):
                assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_acoustic_2d_module() -> None:
    """Test propagation with the Module interface."""
    run_forward(propagator=acousticprop, functional=False)


def test_variable_density() -> None:
    """Test that variable density produces different results from constant."""
    # Run with constant density
    out_const = run_forward(propagator=acousticprop, rho_val=1.0, drho=0.0)
    # Run with variable density
    out_var = run_forward(propagator=acousticprop, rho_val=1.0, drho=0.5)
    # They should be different
    assert not torch.allclose(out_const[-3], out_var[-3])


def test_v_batched() -> None:
    """Test forward using a different velocity for each shot."""
    actuals = {}
    for python in [True, False]:
        # Run with batched velocity
        out = run_forward(
            c=torch.tensor([[[1500.0]], [[1600.0]]]),
            propagator=acousticprop,
            prop_kwargs={"python_backend": python},
        )
        actuals[python] = out[-3]
    assert torch.allclose(
        actuals[True], actuals[False], atol=5e-4 * actuals[True].abs().max().item()
    )


def test_rho_batched() -> None:
    """Test forward using a different rho for each shot."""
    actuals = {}
    for python in [True, False]:
        out = run_forward(
            rho_val=torch.tensor([[[1900.0]], [[2000.0]]]),
            propagator=acousticprop,
            prop_kwargs={"python_backend": python},
        )
        actuals[python] = out[-3]
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
    propagator=acousticprop,
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
        if dtype is None:
            dtype = torch.double

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["python_backend"] = python

        model = (
            torch.ones(*nx, device=device, dtype=dtype) * c
            + torch.randn(*nx, device=device, dtype=dtype) * dc
        )
        rho = torch.ones_like(model)

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
                rho,
                dx.tolist(),
                dt,
                source_amplitudes_p=sources["amplitude"],
                source_locations_p=sources["locations"],
                receiver_locations_p=x_r,
                prop_kwargs=prop_kwargs,
            )

            (out_ignored[-(len(dx) + 1)] ** 2).sum().backward()

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
            rho,
            dx.tolist(),
            dt,
            source_amplitudes_p=sources["amplitude"],
            source_locations_p=sources["locations"],
            receiver_locations_p=x_r,
            prop_kwargs=prop_kwargs,
        )
        # Set receiver amplitudes of receiver that will be ignored to zero
        for i in range(num_shots):
            out_filled[-(len(dx) + 1)][i, i].fill_(0)

        (out_filled[-(len(dx) + 1)] ** 2).sum().backward()

        for ofi, oii in zip(out_filled, out_ignored):
            assert torch.allclose(ofi, oii)

        assert torch.allclose(modelf.grad, modeli.grad)


def test_gradcheck() -> None:
    """Test gradcheck in a 2D model."""
    run_gradcheck(propagator=acousticprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 4), (5, 5)),
        ((5, 4, 4), (6, 6, 6)),
    ],
)
def test_gradcheck_2nd_order(nx, dx) -> None:
    """Test gradcheck with a 2nd order accurate propagator."""
    run_gradcheck(propagator=acousticprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2})


def test_gradcheck_6th_order() -> None:
    """Test gradcheck with a 6th order accurate propagator."""
    run_gradcheck(propagator=acousticprop, prop_kwargs={"accuracy": 6})


def test_gradcheck_8th_order() -> None:
    """Test gradcheck with a 8th order accurate propagator."""
    run_gradcheck(propagator=acousticprop, prop_kwargs={"accuracy": 8})


def test_gradcheck_cfl() -> None:
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_gradcheck(propagator=acousticprop, dt=0.002, nt_add=1000, atol=5e-8)


def test_gradcheck_odd_timesteps() -> None:
    """Test gradcheck with one more timestep."""
    run_gradcheck(propagator=acousticprop, nt_add=1)


def test_gradcheck_one_shot() -> None:
    """Test gradcheck with one shot."""
    run_gradcheck(propagator=acousticprop, num_shots=1)


def test_gradcheck_no_sources() -> None:
    """Test gradcheck with no sources."""
    run_gradcheck(propagator=acousticprop, num_sources_per_shot=0)


def test_gradcheck_no_receivers() -> None:
    """Test gradcheck with no receivers."""
    run_gradcheck(propagator=acousticprop, num_receivers_per_shot=0)


def test_gradcheck_survey_pad() -> None:
    """Test gradcheck with survey_pad."""
    run_gradcheck(
        propagator=acousticprop,
        survey_pad=0,
        nx=(12, 9),
        provide_wavefields=False,
    )


def test_gradcheck_partial_wavefields() -> None:
    """Test gradcheck with wavefields that do not cover the model."""
    run_gradcheck(
        propagator=acousticprop,
        origin=(0, 4),
        nx=(12, 9),
        wavefield_size=(4 + 6, 1 + 6),
    )


def test_gradcheck_negative() -> None:
    """Test gradcheck with negative velocity."""
    run_gradcheck(c=-1500, propagator=acousticprop)


def test_gradcheck_zero() -> None:
    """Test gradcheck with zero velocity."""
    run_gradcheck(c=0, dc=0, propagator=acousticprop)


def test_gradcheck_different_pml() -> None:
    """Test gradcheck with different pml widths."""
    run_gradcheck(propagator=acousticprop, pml_width=[0, 1, 5, 10], atol=5e-8)


def test_gradcheck_no_pml() -> None:
    """Test gradcheck with no pml."""
    run_gradcheck(propagator=acousticprop, pml_width=0, nx=(11, 11))


def test_gradcheck_different_dx() -> None:
    """Test gradcheck with different dx values."""
    run_gradcheck(propagator=acousticprop, dx=(4, 5), dt=0.0005, atol=4e-8)


def test_gradcheck_big() -> None:
    """Test gradcheck with a big model."""
    run_gradcheck(
        propagator=acousticprop,
        nx=(5 + 2 * (3 + 3 * 2), 4 + 2 * (3 + 3 * 2)),
        atol=2e-8,
    )


def test_gradcheck_only_v_2d() -> None:
    """Test gradcheck with only v requiring gradient."""
    run_gradcheck(
        propagator=acousticprop,
        source_requires_grad=False,
        pressure_0_requires_grad=False,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=False,
    )


def test_gradcheck_only_rho_2d() -> None:
    """Test gradcheck with only rho requiring gradient."""
    run_gradcheck(
        propagator=acousticprop,
        v_requires_grad=False,
        source_requires_grad=False,
        pressure_0_requires_grad=False,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=True,
    )


def test_gradcheck_only_source_2d() -> None:
    """Test gradcheck with only source requiring gradient."""
    run_gradcheck(
        propagator=acousticprop,
        v_requires_grad=False,
        pressure_0_requires_grad=False,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=False,
    )


def test_gradcheck_only_pressure_0_2d() -> None:
    """Test gradcheck with only pressure_0 requiring gradient."""
    run_gradcheck(
        propagator=acousticprop,
        v_requires_grad=False,
        source_requires_grad=False,
        pressure_0_requires_grad=True,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=False,
    )


def test_gradcheck_v_batched() -> None:
    """Test gradcheck using a different velocity for each shot."""
    run_gradcheck(propagator=acousticprop, c=torch.tensor([[[1500.0]], [[1600.0]]]))


def test_gradcheck_rho_batched() -> None:
    """Test gradcheck using a different rho for each shot."""
    run_gradcheck(
        propagator=acousticprop, rho_val=torch.tensor([[[1900.0]], [[2000.0]]])
    )


@pytest.mark.parametrize(("v_requires_grad"), [True, False])
@pytest.mark.parametrize(("rho_requires_grad"), [True, False])
@pytest.mark.parametrize(("num_shots"), [1, 2])
@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 3), (5, 5)),
        ((5, 3, 3), (6, 6, 6)),
    ],
)
def test_storage(
    v_requires_grad,
    rho_requires_grad,
    num_shots,
    nx,
    dx,
    c=1500,
    dc=100,
    freq=25,
    dt=0.005,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    propagator=acousticprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Test gradients with different storage options."""
    if not v_requires_grad and not rho_requires_grad:
        return
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
    rho = torch.ones(*nx, device=device, dtype=dtype)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    # Store uncompressed in device memory (default)
    modeld = model.clone()
    modeld.requires_grad_(v_requires_grad)
    rhod = rho.clone()
    rhod.requires_grad_(rho_requires_grad)
    out = propagator(
        modeld,
        rhod,
        dx.tolist(),
        dt,
        source_amplitudes_p=sources["amplitude"],
        source_locations_p=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
    )

    (out[-(len(dx) + 1)] ** 2).sum().backward()

    modes = ["device", "disk", "none"]
    if model.is_cuda:
        modes.append("cpu")

    for mode in modes:
        for compression in [False, True]:
            if mode == "device" and not compression:  # Default
                continue
            modeli = model.clone()
            modeli.requires_grad_(v_requires_grad)
            rhoi = rho.clone()
            rhoi.requires_grad_(rho_requires_grad)
            prop_kwargs = dict(prop_kwargs)
            prop_kwargs["storage_mode"] = mode
            prop_kwargs["storage_compression"] = compression
            out = propagator(
                modeli,
                rhoi,
                dx.tolist(),
                dt,
                source_amplitudes_p=sources["amplitude"],
                source_locations_p=sources["locations"],
                receiver_locations_p=x_r,
                prop_kwargs=prop_kwargs,
            )

            (out[-(len(dx) + 1)] ** 2).sum().backward()

            if mode != "none":
                if compression:
                    if v_requires_grad:
                        assert torch.allclose(
                            modeld.grad,
                            modeli.grad,
                            atol=0.1 * modeld.grad.detach().abs().max().item(),
                        )
                    if rho_requires_grad:
                        assert torch.allclose(
                            rhod.grad,
                            rhoi.grad,
                            atol=0.1 * rhod.grad.detach().abs().max().item(),
                        )
                else:
                    if v_requires_grad:
                        assert torch.allclose(
                            modeld.grad,
                            modeli.grad,
                            atol=modeld.grad.detach().abs().max().item() * 1e-5,
                        )
                    if rho_requires_grad:
                        assert torch.allclose(
                            rhod.grad,
                            rhoi.grad,
                            atol=rhod.grad.detach().abs().max().item() * 1e-5,
                        )


def run_direct(
    c: Union[float, torch.Tensor] = 1500,
    rho_val: Union[float, torch.Tensor] = 2000,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.0001,
    nx: Tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = acousticprop,
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
    if isinstance(rho_val, torch.Tensor):
        rho_val = rho_val.to(device)
    model = torch.ones(*nx, device=device, dtype=dtype) * c
    rho = torch.ones(*nx, device=device, dtype=dtype) * rho_val

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
    source_diff = torch.gradient(sources["amplitude"], dim=-1, spacing=dt)[0]
    rho_scalar = rho_val
    if isinstance(rho_val, torch.Tensor):
        rho_scalar = rho_val.item()
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver, :] += direct(
                    x_r[shot, receiver],
                    x_s[shot, source],
                    dx,
                    dt,
                    shot_c[shot],
                    rho_scalar * source_diff[shot, source, :],
                ).to(dtype)

    actual = propagator(
        model,
        rho,
        dx.tolist(),
        dt,
        source_amplitudes_p=sources["amplitude"],
        source_locations_p=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )[-len(nx) - 1]

    return expected, actual


def run_forward(
    c: Union[float, torch.Tensor] = 1500.0,
    rho_val: Union[float, torch.Tensor] = 2000.0,
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
    dc: float = 100.0,
    drho: float = 100.0,
    nt: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    """Create a random model and forward propagate."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(c, torch.Tensor):
        c = c.to(device)
    if isinstance(rho_val, torch.Tensor):
        rho_val = rho_val.to(device)
    model = (
        torch.ones(*nx, device=device, dtype=dtype) * c
        + torch.randn(*nx, dtype=dtype).to(device) * dc
    )
    rho = (
        torch.ones(*nx, device=device, dtype=dtype) * rho_val
        + torch.randn(*nx, dtype=dtype).to(device) * drho
    )

    # Handle batched input for setup
    c_val = c.abs().min().item() if isinstance(c, torch.Tensor) else abs(c)

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    if nt is None:
        nt = int((2 * torch.norm(nx.float() * dx) / c_val + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    receiver_locations_z = None
    receiver_locations_y = None
    receiver_locations_x = None

    if len(nx) >= 3:
        receiver_locations_z = x_r
    if len(nx) >= 2:
        receiver_locations_y = x_r
    receiver_locations_x = x_r

    return propagator(
        model,
        rho,
        dx.tolist(),
        dt,
        source_amplitudes_p=sources["amplitude"],
        source_locations_p=sources["locations"],
        receiver_locations_p=x_r,
        receiver_locations_z=receiver_locations_z,
        receiver_locations_y=receiver_locations_y,
        receiver_locations_x=receiver_locations_x,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )


def run_gradcheck(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 100.0,
    rho_val: Union[float, torch.Tensor] = 2000.0,
    drho: float = 100.0,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.001,
    nx: Tuple[int, ...] = (10, 10),
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
    rho_requires_grad: bool = True,
    source_requires_grad: bool = True,
    provide_wavefields: bool = True,
    pressure_0_requires_grad: bool = True,
    v_0_requires_grad: bool = True,
    phi_0_requires_grad: bool = True,
    psi_0_requires_grad: bool = True,
    only_receivers_out: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-5,
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

    if isinstance(rho_val, torch.Tensor):
        rho_val = rho_val.to(device)
    rho = (
        torch.ones(*nx, device=device, dtype=dtype) * rho_val
        + torch.rand(*nx, device=device, dtype=dtype) * drho
    )

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    if min_c != 0:
        nt = int((2 * torch.norm(nx.float() * dx) / min_c + 0.1 + 2 / freq) / dt)
    else:
        nt = int((2 * torch.norm(nx.float() * dx) / 1500 + 0.1 + 2 / freq) / dt)
    nt += nt_add

    source_amplitudes = {
        "source_amplitudes_p": None,
        "source_amplitudes_z": None,
        "source_amplitudes_y": None,
        "source_amplitudes_x": None,
    }
    source_locations = {
        "source_locations_p": None,
        "source_locations_z": None,
        "source_locations_y": None,
        "source_locations_x": None,
    }
    receiver_locations = {
        "receiver_locations_p": None,
        "receiver_locations_z": None,
        "receiver_locations_y": None,
        "receiver_locations_x": None,
    }
    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        sources_x = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_x["amplitude"] = sources_x["amplitude"].to(device)
        sources_x["amplitude"].requires_grad_(source_requires_grad)
        source_amplitudes["source_amplitudes_x"] = sources_x["amplitude"]
        source_locations["source_locations_x"] = sources_x["locations"]
        sources_p = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_p["amplitude"] = sources_p["amplitude"].to(device)
        sources_p["amplitude"].requires_grad_(source_requires_grad)
        source_amplitudes["source_amplitudes_p"] = sources_p["amplitude"]
        source_locations["source_locations_p"] = sources_p["locations"]
        if ndim >= 2:
            sources_y = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
            sources_y["amplitude"] = sources_y["amplitude"].to(device)
            sources_y["amplitude"].requires_grad_(source_requires_grad)
            source_amplitudes["source_amplitudes_y"] = sources_y["amplitude"]
            source_locations["source_locations_y"] = sources_y["locations"]
        if ndim >= 3:
            sources_z = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
            sources_z["amplitude"] = sources_z["amplitude"].to(device)
            sources_z["amplitude"].requires_grad_(source_requires_grad)
            source_amplitudes["source_amplitudes_z"] = sources_z["amplitude"]
            source_locations["source_locations_z"] = sources_z["locations"]
        nt = None
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
        receiver_locations["receiver_locations_x"] = x_r
        receiver_locations["receiver_locations_p"] = x_r
        if ndim >= 2:
            receiver_locations["receiver_locations_y"] = x_r
        if ndim >= 3:
            receiver_locations["receiver_locations_z"] = x_r

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
        pressure_0 = torch.zeros(
            num_shots,
            *wavefield_size,
            dtype=dtype,
            device=device,
        )
        pressure_0.requires_grad_(pressure_0_requires_grad)

        vz_0, vy_0, vx_0 = None, None, None
        phi_z_0, phi_y_0, phi_x_0 = None, None, None
        psi_z_0, psi_y_0, psi_x_0 = None, None, None

        if ndim == 3:
            vz_0 = torch.zeros_like(pressure_0)
            phi_z_0 = torch.zeros_like(pressure_0)
            psi_z_0 = torch.zeros_like(pressure_0)
        if ndim >= 2:
            vy_0 = torch.zeros_like(pressure_0)
            phi_y_0 = torch.zeros_like(pressure_0)
            psi_y_0 = torch.zeros_like(pressure_0)

        vx_0 = torch.zeros_like(pressure_0)
        phi_x_0 = torch.zeros_like(pressure_0)
        psi_x_0 = torch.zeros_like(pressure_0)

        # Now set requires_grad for velocities and pml variables
        for v_field in [vz_0, vy_0, vx_0]:
            if v_field is not None:
                v_field.requires_grad_(v_0_requires_grad)

        for phi_field in [phi_z_0, phi_y_0, phi_x_0]:
            if phi_field is not None:
                phi_field.requires_grad_(phi_0_requires_grad)

        for psi_field in [psi_z_0, psi_y_0, psi_x_0]:
            if psi_field is not None:
                psi_field.requires_grad_(psi_0_requires_grad)

    else:
        pressure_0 = None
        vz_0, vy_0, vx_0 = None, None, None
        phi_z_0, phi_y_0, phi_x_0 = None, None, None
        psi_z_0, psi_y_0, psi_x_0 = None, None, None

    model.requires_grad_(v_requires_grad)
    rho.requires_grad_(rho_requires_grad)
    if prop_kwargs is None:
        prop_kwargs = {}

    out_idxs = []

    def wrap(python):
        prop_kwargs["python_backend"] = python
        inputs = (
            model,
            rho,
            dx.tolist(),
            dt,
            *source_amplitudes.values(),
            *source_locations.values(),
            *receiver_locations.values(),
            prop_kwargs,
            pml_width,
            survey_pad,
            origin,
            pressure_0,
            vz_0,
            vy_0,
            vx_0,
            phi_z_0,
            phi_y_0,
            phi_x_0,
            psi_z_0,
            psi_y_0,
            psi_x_0,
            nt,
            1,
            None,
            True,
        )
        out = propagator(*inputs)
        if only_receivers_out:
            out = out[-ndim - 1 :]
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
        )
        return tuple(out) + grads

    torch.manual_seed(1)
    out_python = wrap("jit")
    torch.manual_seed(1)
    out_compiled = wrap(False)
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((10,), (5,)),
        ((10, 10), (5, 5)),
        ((10, 10, 10), (6, 6, 6)),
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
    rho = torch.ones_like(model)

    source_amplitudes = ricker(freq, nt, dt, peak_time, dtype=dtype).reshape(1, 1, -1)

    loc_a = torch.tensor([d // 4 for d in nx], device=device).reshape(1, 1, -1)
    loc_b = torch.tensor([d - d // 4 for d in nx], device=device).reshape(1, 1, -1)

    prop_kwargs = {"accuracy": 2}

    ndim = len(nx)
    if ndim == 3:
        comps = ["_p", "_z", "_y", "_x"]
        idxs = [-4, -3, -2, -1]
    elif ndim == 2:
        comps = ["_p", "_y", "_x"]
        idxs = [-3, -2, -1]
    elif ndim == 1:
        comps = ["_p", "_x"]
        idxs = [-2, -1]

    for comp, idx in zip(comps, idxs):
        run_reciprocity_check(
            [model, rho],
            dx,
            dt,
            source_amplitudes,
            comp,
            comp,
            idx,
            idx,
            loc_a,
            loc_b,
            prop_kwargs,
            acousticprop,
        )

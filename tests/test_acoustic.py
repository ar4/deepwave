"""Tests for deepwave.acoustic."""

from typing import Any, Optional, Union

import pytest
import torch
from helpers import (
    _build_multicomponent_receivers,
    _build_multicomponent_sources,
    _default_wavefield_size,
    _normalize_pml_width,
    _requires_grad_if_present,
    _set_coords,
    _set_sources,
    auxiliary_state_arg,
    compute_backend_parity_grad,
    compute_backend_parity_nt,
    direct_1d,
    direct_2d_approx,
    direct_3d,
    element_tensors,
    output_tensors,
    run_checkpoint_equivalence,
    run_python_vs_compiled_backend_parity,
    run_reciprocity_check,
)
from torch.func import jvp, vmap

from deepwave import IGNORE_LOCATION, Acoustic, acoustic
from deepwave._backend_utils import USE_OPENMP
from deepwave.discretisation import cfl_condition, upsample
from deepwave.wavelets import ricker


@pytest.fixture(scope="module", autouse=True)
def _detect_anomaly():
    """Runs this module's tests with autograd anomaly detection enabled.

    Anomaly mode is a global of the interpreter, so enabling it at import
    time would leave it on for every other test module collected into the
    same process, where it costs an order of magnitude in run time for no
    benefit. Scoping it to this module and restoring the previous setting
    afterwards keeps it where it is wanted.
    """
    previous = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(previous)


# ``python_backend=True`` resolves to "torch_compile" if the OpenMP build is
# available, else "eager" (v0.0.27 chose torch.jit.script there, a mode that
# no longer exists). Tests below that loop over "the non-compiled Python
# backend vs the compiled C backend" use that same resolution rather than
# hardcoding one, so they exercise whichever path this build actually takes.
_PYTHON_BACKEND_TRUE_EQUIVALENT = "torch_compile" if USE_OPENMP else "eager"


def acousticprop(
    model: torch.Tensor,
    rho: torch.Tensor,
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes_volume_rate: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_volume_rate: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Optional[Union[int, list[int]]] = None,
    survey_pad: Optional[Union[int, list[int]]] = None,
    origin: Optional[list[int]] = None,
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
    functional: bool = True,
) -> tuple[torch.Tensor, ...]:
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
    if source_amplitudes_volume_rate is not None:
        source_amplitudes_volume_rate = source_amplitudes_volume_rate.to(device)
        source_locations_volume_rate = source_locations_volume_rate.to(device)
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

    # phi_*/psi_* are the absorbing boundary's memory variables, not physical
    # fields, so they are no longer direct keyword arguments: they travel
    # inside the single ``auxiliary_state`` that ``initial_state`` carries.
    # Any that are None are simply absent, and that part of the state starts
    # from rest.
    phi_psi_state = auxiliary_state_arg(
        phi_z_0=phi_z_0,
        phi_y_0=phi_y_0,
        phi_x_0=phi_x_0,
        psi_z_0=psi_z_0,
        psi_y_0=psi_y_0,
        psi_x_0=psi_x_0,
    )

    if functional:
        return acoustic(
            model,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=source_amplitudes_volume_rate,
            source_locations_volume_rate=source_locations_volume_rate,
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
            initial_state=phi_psi_state,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            **prop_kwargs,
        )

    prop = Acoustic(model, rho, dx)
    return prop(
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
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
        initial_state=phi_psi_state,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def test_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_forward(propagator=acousticprop, prop_kwargs={"backend": "torch_compile"})


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((50,), (5,)),
        ((50, 50), (5, 5)),
        pytest.param((10, 10, 10), (5, 5, 5), marks=pytest.mark.slow),
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
        for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
            expected, actual = run_direct(
                nx=nx,
                dx=dx,
                propagator=acousticprop,
                prop_kwargs={"backend": python, "accuracy": accuracy},
            )
            diff = (expected - actual.cpu()).flatten()
            assert (
                diff.norm().item() / expected.flatten().norm().item()
                < expected_diffs[accuracy]
            )
            actuals[python] = actual
        assert torch.allclose(
            actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT],
            actuals["compiled"],
            atol=5e-4 * actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT].abs().max().item(),
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
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        out = run_forward(
            propagator=acousticprop,
            nx=nx,
            dx=dx,
            nt=25000 if len(nx) == 1 else 2500,
            prop_kwargs={"backend": python},
        )
        dims = ["z", "y", "x"][-len(dx) :]
        state = out.auxiliary_state
        pmax = out.receiver_amplitudes_p.abs().max().item()
        vmax = {
            d: getattr(out, f"receiver_amplitudes_{d}").abs().max().item() for d in dims
        }
        # The final wavefields, each scaled by the magnitude of the receiver
        # data of the quantity it belongs to.
        scaled = [out.pressure_0 / pmax]
        for d in dims:
            scaled.append(getattr(out, f"v{d}_0") / vmax[d])
            scaled.append(state[f"phi_{d}_0"] / vmax[d])
            scaled.append(state[f"psi_{d}_0"] / pmax)
        for outi in scaled:
            assert outi.norm() < 5e-4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        actual_cpu = run_forward(
            propagator=acousticprop,
            device=torch.device("cpu"),
            prop_kwargs={"backend": python},
            dtype=torch.double,
        )
        actual_gpu = run_forward(
            propagator=acousticprop,
            device=torch.device("cuda"),
            prop_kwargs={"backend": python},
            dtype=torch.double,
        )
        for cpui, gpui in zip(output_tensors(actual_cpu), output_tensors(actual_gpu)):
            assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_acoustic_2d_module() -> None:
    """Test propagation with the Module interface."""
    run_forward(propagator=acousticprop, functional=False)


def test_variable_density() -> None:
    """Test that variable density produces different results from constant.

    The densities are the file's ordinary ones (2000 +/- 100), and that is
    load-bearing rather than incidental. This test used to ask for
    ``rho_val=1.0, drho=0.5``, which builds ``rho`` as
    ``1.0 + 0.5 * randn(50, 50)``: with this file's fixed seed that gives 67
    **non-positive** densities out of 2500, down to -0.97. A negative density
    is not a model, the simulation diverges, and the "variable" run's
    wavefields came back as ``NaN`` -- which the assertion below then read as
    "different from the constant run", so the test passed for the wrong
    reason. It only became visible when ``check_output_finite`` started
    defaulting to ``True`` and the propagator began saying so.
    """
    # Run with constant density
    out_const = run_forward(propagator=acousticprop, drho=0.0)
    # Run with variable density
    out_var = run_forward(propagator=acousticprop)
    # They should be different
    assert not torch.allclose(out_const[-3], out_var[-3])
    # ...and both finite, which is what makes "different" mean anything here.
    assert torch.isfinite(out_const[-3]).all()
    assert torch.isfinite(out_var[-3]).all()


def test_v_batched() -> None:
    """Test forward using a different velocity for each shot."""
    actuals = {}
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        # Run with batched velocity
        out = run_forward(
            c=torch.tensor([[[1500.0]], [[1600.0]]]),
            propagator=acousticprop,
            prop_kwargs={"backend": python},
        )
        actuals[python] = out[-3]
    assert torch.allclose(
        actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT],
        actuals["compiled"],
        atol=5e-4 * actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT].abs().max().item(),
    )


def test_rho_batched() -> None:
    """Test forward using a different rho for each shot."""
    actuals = {}
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        out = run_forward(
            rho_val=torch.tensor([[[1900.0]], [[2000.0]]]),
            propagator=acousticprop,
            prop_kwargs={"backend": python},
        )
        actuals[python] = out[-3]
    assert torch.allclose(
        actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT],
        actuals["compiled"],
        atol=5e-4 * actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT].abs().max().item(),
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

    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.double

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["backend"] = python

        model = (
            torch.ones(*nx, device=device, dtype=dtype) * c
            + torch.randn(*nx, device=device, dtype=dtype) * dc
        )
        rho = torch.ones_like(model)

        nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
        sources = _set_sources(x_s, freq, dt, nt, dtype)

        n_warmup = 3 if python != "compiled" else 1

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
                source_amplitudes_volume_rate=sources["amplitude"],
                source_locations_volume_rate=sources["locations"],
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
            source_amplitudes_volume_rate=sources["amplitude"],
            source_locations_volume_rate=sources["locations"],
            receiver_locations_p=x_r,
            prop_kwargs=prop_kwargs,
        )
        # Set receiver amplitudes of receiver that will be ignored to zero
        for i in range(num_shots):
            out_filled[-(len(dx) + 1)][i, i].fill_(0)

        (out_filled[-(len(dx) + 1)] ** 2).sum().backward()

        # Every element is a Tensor except auxiliary_state, which holds
        # several; compare the tensors either way.
        for ofi, oii in zip(out_filled, out_ignored):
            for filled, ignored in zip(element_tensors(ofi), element_tensors(oii)):
                assert torch.allclose(filled, ignored)

        assert torch.allclose(modelf.grad, modeli.grad)


def test_backend_parity() -> None:
    """Test backend parity in a 2D model."""
    run_backend_parity(propagator=acousticprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 4), (5, 5)),
        ((5, 4, 4), (6, 6, 6)),
    ],
)
def test_backend_parity_2nd_order(nx, dx) -> None:
    """Test backend parity with a 2nd order accurate propagator."""
    run_backend_parity(
        propagator=acousticprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2}
    )


def test_backend_parity_6th_order() -> None:
    """Test backend parity with a 6th order accurate propagator."""
    run_backend_parity(propagator=acousticprop, prop_kwargs={"accuracy": 6})


def test_backend_parity_8th_order() -> None:
    """Test backend parity with a 8th order accurate propagator."""
    run_backend_parity(propagator=acousticprop, prop_kwargs={"accuracy": 8})


@pytest.mark.slow
def test_backend_parity_cfl() -> None:
    """Test backend parity with a timestep greater than the CFL limit."""
    run_backend_parity(propagator=acousticprop, dt=0.002, nt_add=1000, atol=5e-8)


def test_backend_parity_odd_timesteps() -> None:
    """Test backend parity with one more timestep."""
    run_backend_parity(propagator=acousticprop, nt_add=1)


def test_backend_parity_one_shot() -> None:
    """Test backend parity with one shot."""
    run_backend_parity(propagator=acousticprop, num_shots=1)


def test_backend_parity_no_sources() -> None:
    """Test backend parity with no sources."""
    run_backend_parity(propagator=acousticprop, num_sources_per_shot=0)


def test_backend_parity_no_receivers() -> None:
    """Test backend parity with no receivers."""
    run_backend_parity(propagator=acousticprop, num_receivers_per_shot=0)


def test_backend_parity_survey_pad() -> None:
    """Test backend parity with survey_pad."""
    run_backend_parity(
        propagator=acousticprop,
        survey_pad=0,
        nx=(12, 9),
        provide_wavefields=False,
    )


def test_backend_parity_partial_wavefields() -> None:
    """Test backend parity with wavefields that do not cover the model."""
    run_backend_parity(
        propagator=acousticprop,
        origin=(0, 4),
        nx=(12, 9),
        wavefield_size=(4 + 6, 1 + 6),
    )


def test_backend_parity_negative() -> None:
    """Backends agree for negative background velocity."""
    run_backend_parity(c=-1500, propagator=acousticprop)


def test_backend_parity_zero() -> None:
    """Backends agree for zero background velocity."""
    run_backend_parity(c=0, dc=0, propagator=acousticprop)


def test_backend_parity_different_pml() -> None:
    """Test backend parity with different pml widths."""
    run_backend_parity(propagator=acousticprop, pml_width=[0, 1, 5, 10], atol=5e-8)


def test_backend_parity_no_pml() -> None:
    """Test backend parity with no pml."""
    run_backend_parity(propagator=acousticprop, pml_width=0, nx=(11, 11))


def test_backend_parity_different_dx() -> None:
    """Test backend parity with different dx values."""
    run_backend_parity(propagator=acousticprop, dx=(4, 5), dt=0.0005, atol=4e-8)


def test_backend_parity_big() -> None:
    """Test backend parity with a big model."""
    run_backend_parity(
        propagator=acousticprop,
        nx=(5 + 2 * (3 + 3 * 2), 4 + 2 * (3 + 3 * 2)),
        atol=2e-8,
    )


def test_backend_parity_only_v_2d() -> None:
    """Test backend parity with only v requiring gradient."""
    run_backend_parity(
        propagator=acousticprop,
        source_requires_grad=False,
        pressure_0_requires_grad=False,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=False,
    )


def test_backend_parity_only_rho_2d() -> None:
    """Test backend parity with only rho requiring gradient."""
    run_backend_parity(
        propagator=acousticprop,
        v_requires_grad=False,
        source_requires_grad=False,
        pressure_0_requires_grad=False,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=True,
    )


def test_backend_parity_only_source_2d() -> None:
    """Test backend parity with only source requiring gradient."""
    run_backend_parity(
        propagator=acousticprop,
        v_requires_grad=False,
        pressure_0_requires_grad=False,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=False,
    )


def test_backend_parity_only_pressure_0_2d() -> None:
    """Test backend parity with only pressure_0 requiring gradient."""
    run_backend_parity(
        propagator=acousticprop,
        v_requires_grad=False,
        source_requires_grad=False,
        pressure_0_requires_grad=True,
        v_0_requires_grad=False,
        phi_0_requires_grad=False,
        psi_0_requires_grad=False,
        rho_requires_grad=False,
    )


def test_backend_parity_v_batched() -> None:
    """Test backend parity using a different velocity for each shot."""
    run_backend_parity(
        propagator=acousticprop, c=torch.tensor([[[1500.0]], [[1600.0]]])
    )


def test_backend_parity_rho_batched() -> None:
    """Test backend parity using a different rho for each shot."""
    run_backend_parity(
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
        source_amplitudes_volume_rate=sources["amplitude"],
        source_locations_volume_rate=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
    )

    (out[-(len(dx) + 1)] ** 2).sum().backward()

    # ``storage_mode="none"`` is deliberately NOT in this list. It retains no
    # forward wavefield history, so no model gradient can be formed from it;
    # the engine now raises rather than returning the plausible-looking but
    # wrong gradient the propagators' own front-end reparameterisation used to
    # produce. This loop runs models that require grad, so it cannot include
    # "none". That rejection, and the fact that source-amplitude gradients and
    # forward-only runs still work in "none" mode, are covered by
    # tests/test_storage_mode_none_model_gradient.py.
    modes = ["device", "disk"]
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
                source_amplitudes_volume_rate=sources["amplitude"],
                source_locations_volume_rate=sources["locations"],
                receiver_locations_p=x_r,
                prop_kwargs=prop_kwargs,
            )

            (out[-(len(dx) + 1)] ** 2).sum().backward()

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
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.0001,
    nx: tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = acousticprop,
    prop_kwargs: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        source_amplitudes_volume_rate=sources["amplitude"],
        source_locations_volume_rate=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )[-len(nx) - 1]

    return expected, actual


def run_forward(
    c: Union[float, torch.Tensor] = 1500.0,
    rho_val: Union[float, torch.Tensor] = 2000.0,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.004,
    nx: tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    dc: float = 100.0,
    drho: float = 100.0,
    nt: Optional[int] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
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
        source_amplitudes_volume_rate=sources["amplitude"],
        source_locations_volume_rate=sources["locations"],
        receiver_locations_p=x_r,
        receiver_locations_z=receiver_locations_z,
        receiver_locations_y=receiver_locations_y,
        receiver_locations_x=receiver_locations_x,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )


def run_backend_parity(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 100.0,
    rho_val: Union[float, torch.Tensor] = 2000.0,
    drho: float = 100.0,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.001,
    nx: tuple[int, ...] = (10, 10),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Union[int, list[int]] = 3,
    survey_pad: Optional[Union[int, list[int]]] = None,
    origin: Optional[list[int]] = None,
    wavefield_size: Optional[tuple[int, ...]] = None,
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
    """Compare the eager and compiled backends' gradients (no finite differencing)."""
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

    nt = compute_backend_parity_nt(nx, dx, min_c, freq, dt, nt_add)

    key_order = ["p", "z", "y", "x"]
    source_amplitudes: dict[str, Optional[torch.Tensor]] = {
        f"source_amplitudes_{k}": None for k in key_order
    }
    source_locations: dict[str, Optional[torch.Tensor]] = {
        f"source_locations_{k}": None for k in key_order
    }
    receiver_locations: dict[str, Optional[torch.Tensor]] = {
        f"receiver_locations_{k}": None for k in key_order
    }
    x_r = None
    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        source_amplitudes, source_locations = _build_multicomponent_sources(
            key_order, x_s, freq, dt, nt, dtype, device, ndim, source_requires_grad
        )
        nt = None
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
        receiver_locations = _build_multicomponent_receivers(key_order, x_r, ndim)

    pml_width = _normalize_pml_width(pml_width, ndim)
    if provide_wavefields:
        if wavefield_size is None:
            wavefield_size = _default_wavefield_size(nx, pml_width, ndim)
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
        _requires_grad_if_present([vz_0, vy_0, vx_0], v_0_requires_grad)
        _requires_grad_if_present([phi_z_0, phi_y_0, phi_x_0], phi_0_requires_grad)
        _requires_grad_if_present([psi_z_0, psi_y_0, psi_x_0], psi_0_requires_grad)

    else:
        pressure_0 = None
        vz_0, vy_0, vx_0 = None, None, None
        phi_z_0, phi_y_0, phi_x_0 = None, None, None
        psi_z_0, psi_y_0, psi_x_0 = None, None, None

    model.requires_grad_(v_requires_grad)
    rho.requires_grad_(rho_requires_grad)
    if prop_kwargs is None:
        prop_kwargs = {}

    def wrap(python):
        prop_kwargs["backend"] = python
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
            True,
        )
        out = propagator(*inputs)
        if only_receivers_out:
            out = out[-ndim - 1 :]
        # The auxiliary state is one output element holding several tensors;
        # the parity comparison is over tensors.
        return compute_backend_parity_grad(output_tensors(out), inputs)

    run_python_vs_compiled_backend_parity(wrap, "eager", atol=atol, rtol=rtol)


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
        # The scalar channel is ``_p`` as a receiver and ``_volume_rate`` as
        # a source here; every velocity channel is spelled the same on both
        # sides.
        source_comp = "_volume_rate" if comp == "_p" else comp
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
            source_comps=(source_comp, source_comp),
        )


def test_vmap_acoustic_props() -> None:
    """Test vmap over acoustic property models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v_base = 1500.0
    rho_base = 2200.0

    batch_size = 3
    v_batch = torch.empty(batch_size, ny, nx, device=device)
    rho_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        v_batch[i] = v_base + i * 100.0
        rho_batch[i] = rho_base + i * 50.0

    n_shots = 2
    n_receivers_per_shot = 3
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2

    receiver_locations = torch.zeros(
        n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
    )
    receiver_locations[..., 0] = ny // 2
    receiver_locations[..., 1] = torch.linspace(
        0, nx - 1, n_receivers_per_shot, device=device
    ).long()

    source_amplitudes = (
        ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1).repeat(n_shots, 1, 1)
    )

    def prop(v, rho):
        return acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]

    res_vmap = vmap(prop)(v_batch, rho_batch)

    res_loop = torch.stack([prop(v_batch[i], rho_batch[i]) for i in range(batch_size)])

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_vmap_acoustic_partial() -> None:
    """Test vmap over rho only (v shared)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v = torch.full((ny, nx), 1500.0, device=device)

    batch_size = 3
    rho_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        rho_batch[i] = 2200.0 + i * 50.0

    n_shots = 1
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2
    receiver_locations = source_locations.clone()

    source_amplitudes = ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1)

    def prop(r):
        return acoustic(
            v,
            r,
            dx,
            dt,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]

    res_vmap = vmap(prop)(rho_batch)

    res_loop = torch.stack([prop(rho_batch[i]) for i in range(batch_size)])

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_jvp_acoustic() -> None:
    """Test jvp on acoustic against a finite difference of the same function.

    Source and receiver are colocated, so no travel distance is required for
    the receiver to see the source's effect; what does need checking is
    whether the perturbation directions (``v_tangent``/``rho_tangent``, which
    vary over the whole 8x8 grid) are actually exercised by ``nt=30``, i.e.
    whether the wave (and its round trip back to the colocated receiver)
    reaches the far corners of the domain in time. Domain half-diagonal from
    the centre source/receiver to a corner is
    ``sqrt((4*5)**2 + (4*5)**2) = 28.28 m``; round trip at ``v=1500 m/s`` is
    ``2 * 28.28 / 1500 = 0.0377 s``, well within ``nt * dt = 30 * 0.004 =
    0.12 s`` (a 3.2x margin), so all 64 grid cells' perturbations reach the
    receiver before the simulation ends.

    Note this only checks that ``torch.func.jvp``'s forward-mode result
    agrees with a finite difference of the *same* eager implementation -- a
    deterministic bug that is consistently present in both the analytic
    derivative and the two perturbed forward evaluations (e.g. a sign or
    scale error in the physics) would not be caught here, since both sides
    of the comparison route through the identical (bugged) code. This test
    is only sensitive to bugs in the jvp/autograd transform machinery
    itself (e.g. a broken custom vjp/jvp rule or an autograd-graph-breaking
    in-place op), not to physics bugs in the eager stencil.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    torch.manual_seed(0)
    v0 = torch.full((ny, nx), 1500.0, device=device, dtype=dtype)
    rho0 = torch.full((ny, nx), 2200.0, device=device, dtype=dtype)

    v_tangent = torch.randn_like(v0) * 10.0
    rho_tangent = torch.randn_like(rho0) * 10.0

    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2
    receiver_locations = source_locations.clone()
    source_amplitudes = (
        ricker(25.0, 30, dt, 0.05, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    def prop(v, r):
        # Output order for ndim=2 is (..., receiver_amplitudes_p,
        # receiver_amplitudes_y, receiver_amplitudes_x); only
        # receiver_locations_p is supplied above, so the y/x receiver
        # amplitude tensors (indices -2, -1) are empty (numel 0) and any
        # comparison against them is vacuously true. Index -3 selects the
        # actual (non-empty) pressure receiver amplitudes -- see the
        # equivalent -3/-2/-1 = p/y/x convention used by test_reciprocity.
        return acoustic(
            v,
            r,
            dx,
            dt,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-3]

    _, jvp_out = jvp(prop, (v0, rho0), (v_tangent, rho_tangent))

    # Finite Difference
    epsilon = 1e-6
    out_plus = prop(v0 + epsilon * v_tangent, rho0 + epsilon * rho_tangent)
    out_minus = prop(v0 - epsilon * v_tangent, rho0 - epsilon * rho_tangent)
    fd_out = (out_plus - out_minus) / (2 * epsilon)

    # A flat per-element atol/rtol is too tight near the receiver waveform's
    # zero crossings: the central-difference estimate has a local truncation
    # error that is negligible relative to the signal's overall scale
    # (measured relative error ~1e-7 of the max amplitude across several
    # seeds) but can be a large *relative* error for individual samples that
    # happen to be near zero. Compare against the signal's own scale instead.
    atol = 1e-4 * fd_out.abs().max().item()
    assert torch.allclose(jvp_out, fd_out, atol=atol, rtol=0)


def test_vmap_error_check() -> None:
    """Ensure error is raised if non-property inputs are vmapped."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    v = torch.full((ny, nx), 1500.0, device=device)
    rho = torch.full((ny, nx), 2200.0, device=device)

    # Try to vmap over source amplitudes
    batch_size = 2
    source_amplitudes_batch = torch.randn(batch_size, 1, 1, 10, device=device)

    def prop(amps):
        return acoustic(
            v,
            rho,
            5.0,
            0.004,
            source_amplitudes_volume_rate=amps,
            source_locations_volume_rate=torch.zeros(
                1, 1, 2, dtype=torch.long, device=device
            ),
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )

    with pytest.raises(NotImplementedError, match="Only property models"):
        vmap(prop)(source_amplitudes_batch)


@pytest.mark.parametrize(
    ("c2", "rho2"),
    [
        (1500.0, 2500.0),  # density contrast only
        (1500.0, 4000.0),  # larger density contrast
        (2500.0, 1000.0),  # velocity contrast only
        (2000.0, 2000.0),  # combined velocity + density contrast
        (1500.0, 500.0),  # impedance drop -> negative R (polarity flip)
    ],
)
def test_normal_incidence_reflection_coefficient(c2, rho2) -> None:
    """1-D normal-incidence reflection amplitude matches the analytic R=ΔZ/ΣZ.

    A plane (1-D) acoustic wave normally incident on a planar contrast in
    acoustic impedance Z = rho * c reflects with pressure reflection coefficient
    R = (Z2 - Z1) / (Z2 + Z1). This exercises the variable-density physics
    (which otherwise has only self-consistency / external-file coverage) against
    a closed-form ground truth, separately for density-only, velocity-only, and
    combined contrasts.

    We compare the energy (L2 norm) of the reflected pulse to the incident
    pulse, which — unlike the raw peak amplitude — is insensitive to the
    (non-dissipative) numerical-dispersion broadening the reflected event picks
    up over its longer travel path. A well-resolved grid (~30 points per
    wavelength, 8th-order) keeps the residual error well under 2%.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    nx = 600
    dx = 5.0
    dt = 0.0004
    freq = 10.0  # ~30 grid points per wavelength
    ix0 = 320  # interface
    isrc = 120
    irec = 250  # between source and interface (records incident then reflected)
    c1 = 1500.0
    rho1 = 1000.0
    nt = int(1.3 / dt)

    v = torch.full((1, nx), c1, device=device, dtype=dtype)
    v[:, ix0:] = c2
    rho = torch.full((1, nx), rho1, device=device, dtype=dtype)
    rho[:, ix0:] = rho2

    z1, z2 = rho1 * c1, rho2 * c2
    r_analytic = (z2 - z1) / (z2 + z1)

    source_locations = torch.zeros(1, 1, 1, dtype=torch.long, device=device)
    source_locations[..., 0] = isrc
    receiver_locations = torch.zeros(1, 1, 1, dtype=torch.long, device=device)
    receiver_locations[..., 0] = irec
    source_amplitudes = (
        ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    out = acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes,
        source_locations_volume_rate=source_locations,
        receiver_locations_p=receiver_locations,
        pml_width=[40, 40],
        pml_freq=freq,
        accuracy=8,
    )
    trace = out.receiver_amplitudes_p[0, 0]

    # The incident pulse passes the receiver at t_inc; the (upper-layer-only)
    # reflected pulse returns at t_ref. Split the record at the midpoint.
    t_inc = (irec - isrc) * dx / c1
    t_ref = ((ix0 - isrc) + (ix0 - irec)) * dx / c1
    mid = int((t_inc + t_ref) / 2 / dt)
    incident = trace[:mid]
    reflected = trace[mid:]

    # Energy-based reflection coefficient; sign from the peak polarity.
    sign = torch.sign(reflected[reflected.abs().argmax()])
    r_measured = sign * (reflected.norm() / incident.norm())

    assert abs(r_measured.item() - r_analytic) < 0.02


def test_rigid_surface_reflection_coefficient() -> None:
    """A zero-``pml_width`` acoustic edge is a rigid surface: R_pressure = +1.

    Unlike the scalar propagator (whose zero-PML edge is a free surface, R = -1),
    the acoustic propagator is staggered: the grid point just outside the edge is
    a normal-*velocity* node, and holding it to zero imposes zero normal velocity
    — an unambiguous rigid wall (this does not depend on interpretation, because
    particle velocity is a named physical quantity). Pressure therefore reflects
    with *preserved* polarity, R = +1.

    The reflecting edge is isolated by subtracting an otherwise identical run with
    that edge absorbing. In 1-D a lossless boundary gives unit reflected/direct
    energy; the positive sign is the rigid-surface signature. (The reflected
    *peak* is slightly below the direct peak because the staggered reflection
    reshapes the waveform, while its energy is conserved.) ``accuracy=2`` is used,
    for which the zero-exterior closure is an exact image.
    """
    dtype = torch.double
    nx, dx, dt, freq = 200, 5.0, 0.0005, 25.0
    src, rec, nt = 45, 22, 900
    v = torch.full((nx,), 1500.0, dtype=dtype)
    rho = torch.full((nx,), 1000.0, dtype=dtype)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)

    def run(pml):
        return acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=torch.tensor([[[src]]]),
            receiver_locations_p=torch.tensor([[[rec]]]),
            pml_width=pml,
            pml_freq=freq,
            accuracy=2,
        ).receiver_amplitudes_p[0, 0]

    direct = run([30, 30])
    reflection = run([0, 30]) - direct

    energy_ratio = (reflection.norm() / direct.norm()).item()
    peak_ratio = (
        reflection[reflection.abs().argmax()] / direct[direct.abs().argmax()]
    ).item()

    assert abs(energy_ratio - 1.0) < 0.02  # lossless, |R| = 1
    assert peak_ratio > 0.5  # preserved: rigid surface, R = +1


def test_spatial_accuracy_reduces_dispersion() -> None:
    """Higher spatial ``accuracy`` order must reduce numerical dispersion.

    The acoustic propagator with constant density obeys the same (non-dispersive)
    wave equation as ``scalar``, so the analytic 1-D Green's function is an
    accurate reference. As in ``run_direct``, the acoustic pressure response to a
    pressure source equals the scalar Green's function driven by ``rho`` times the
    time-derivative of the source. With a tiny Courant number (so the residual is
    dominated by *spatial* dispersion), the misfit against the analytic solution
    decreases monotonically as the spatial accuracy order increases. This
    validates that the ``accuracy`` argument delivers higher-order forward
    fidelity (the existing accuracy tests only check gradient self-consistency).
    """
    dtype = torch.double
    c = 1500.0
    rho = 2000.0
    freq = 25.0
    dx = 6.0  # points-per-wavelength = c / (freq * dx) = 10
    dt = 0.05 * dx / c  # Courant 0.05 => time-stepping error negligible
    nx = 200
    src_i = 40
    rec_i = 160  # 120 cells = 12 wavelengths of propagation
    model = torch.full((nx,), c, dtype=dtype)
    rho_model = torch.full((nx,), rho, dtype=dtype)
    nt = int((1.2 * nx * dx / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[[src_i]]])
    receiver_locations = torch.tensor([[[rec_i]]])
    source_diff = torch.gradient(amp, dim=-1, spacing=dt)[0]
    expected = direct_1d(
        torch.tensor([float(rec_i)]),
        torch.tensor([float(src_i)]),
        torch.tensor([dx], dtype=dtype),
        dt,
        c,
        rho * source_diff[0, 0],
    )

    misfit = {}
    for accuracy in (2, 4, 6, 8):
        out = acoustic(
            model,
            rho_model,
            [dx],
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=accuracy,
            pml_width=20,
            max_vel=c,
            pml_freq=freq,
        )[-2][0, 0]  # [-2] is the pressure receiver in 1-D
        n = min(len(out), len(expected))
        misfit[accuracy] = (
            (out[:n] - expected[:n]).norm() / expected[:n].norm()
        ).item()

    # Measured (CPU, float64): 2 -> ~1.36, 4 -> ~0.33, 6 -> ~0.084, 8 -> ~0.039.
    assert misfit[2] > 2 * misfit[4]  # 2nd -> 4th order: large reduction
    assert misfit[4] > 2 * misfit[6]  # 4th -> 6th order: large reduction
    assert misfit[8] < misfit[4]  # 8th clearly beats 4th (robust, away from floor)
    assert misfit[2] > 0.5  # 2nd-order dispersion over 12 wavelengths is large
    assert misfit[6] < 0.12  # 6th/8th order is accurate at 10 ppw


def test_spatial_accuracy_reduces_dispersion_2d() -> None:
    """Higher spatial ``accuracy`` reduces numerical dispersion in 2-D (acoustic).

    The 2-D counterpart of ``test_spatial_accuracy_reduces_dispersion``,
    filling the gap between the existing 1-D and 3-D acoustic dispersion checks.
    With constant density the acoustic equation is the (non-dispersive) wave
    equation, so the *exact* frequency-domain 2-D Green's function
    (``direct_2d_approx``) is the analytic reference; as in ``run_direct`` the
    pressure response to a pressure source equals that Green's function driven
    by ``rho`` times the time-derivative of the source. A small Courant number
    isolates the *spatial* dispersion, whose misfit falls by roughly 5x per
    accuracy order to a small residual floor.
    """
    dtype = torch.double
    c = 1500.0
    rho = 2000.0
    freq = 25.0
    dx = 6.0  # points-per-wavelength = c / (freq * dx) = 10
    dt = 0.08 * dx / c  # small Courant => time-stepping error negligible
    ny = 70
    nx = 170
    src = (ny // 2, 35)
    rec = (ny // 2, 135)  # 100 cells = 10 wavelengths along x
    model = torch.full((ny, nx), c, dtype=dtype)
    rho_model = torch.full((ny, nx), rho, dtype=dtype)
    r_cells = abs(rec[1] - src[1])
    nt = int((1.3 * r_cells * dx / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])
    source_diff = torch.gradient(amp, dim=-1, spacing=dt)[0]
    expected = direct_2d_approx(
        torch.tensor([float(rec[0]), float(rec[1])]),
        torch.tensor([float(src[0]), float(src[1])]),
        torch.tensor([dx, dx], dtype=dtype),
        dt,
        c,
        rho * source_diff[0, 0],
    )

    misfit = {}
    for accuracy in (2, 4, 6, 8):
        out = acoustic(
            model,
            rho_model,
            dx,
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=accuracy,
            pml_width=20,
            max_vel=c,
            pml_freq=freq,
        ).receiver_amplitudes_p[0, 0]
        n = min(len(out), len(expected))
        misfit[accuracy] = (
            (out[:n] - expected[:n]).norm() / expected[:n].norm()
        ).item()

    # Measured (CPU, float64): 2 -> ~1.39, 4 -> ~0.35, 6 -> ~0.083, 8 -> ~0.029.
    assert misfit[2] > 2 * misfit[4]  # 2nd -> 4th order: large reduction
    assert misfit[4] > 2 * misfit[6]  # 4th -> 6th order: large reduction
    assert misfit[8] < misfit[4]  # 8th clearly beats 4th (robust, away from floor)
    assert misfit[2] > 0.5  # 2nd-order dispersion is large
    assert misfit[6] < 0.12  # 6th/8th order is accurate at 10 ppw


def test_spatial_accuracy_reduces_dispersion_3d(record_property) -> None:
    """Higher spatial ``accuracy`` reduces numerical dispersion in 3-D (acoustic).

    The 3-D counterpart of ``test_spatial_accuracy_reduces_dispersion``,
    exercising the separately-compiled 3-D acoustic kernels. With constant
    density the acoustic equation is the (non-dispersive) wave equation, so the
    exact free-space 3-D Green's function (``direct_3d``) is the analytic
    reference: as in ``run_direct``, the pressure response to a pressure source
    equals that Green's function driven by ``rho`` times the time-derivative of
    the source. A small Courant number isolates the *spatial* dispersion, whose
    misfit falls by roughly 5x per accuracy order to a small residual floor.
    """
    dtype = torch.double
    c = 1500.0
    rho = 2000.0
    freq = 25.0
    dx = 6.0  # points-per-wavelength = c / (freq * dx) = 10
    dt = 0.1 * dx / c  # small Courant => time-stepping error negligible
    nz = ny = 28
    nx = 90
    src = (nz // 2, ny // 2, 12)
    rec = (nz // 2, ny // 2, 70)  # 58 cells ~ 5.8 wavelengths along x
    model = torch.full((nz, ny, nx), c, dtype=dtype)
    rho_model = torch.full((nz, ny, nx), rho, dtype=dtype)
    r_cells = abs(rec[2] - src[2])
    nt = int((1.4 * r_cells * dx / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])
    source_diff = torch.gradient(amp, dim=-1, spacing=dt)[0]
    expected = direct_3d(
        torch.tensor([float(i) for i in rec]),
        torch.tensor([float(i) for i in src]),
        torch.tensor([dx, dx, dx], dtype=dtype),
        dt,
        c,
        rho * source_diff[0, 0],
    )

    misfit = {}
    for accuracy in (2, 4, 6, 8):
        out = acoustic(
            model,
            rho_model,
            dx,
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=accuracy,
            pml_width=15,
            max_vel=c,
            pml_freq=freq,
        ).receiver_amplitudes_p[0, 0]
        n = min(len(out), len(expected))
        misfit[accuracy] = (
            (out[:n] - expected[:n]).norm() / expected[:n].norm()
        ).item()

    record_property("green_function_relative_errors", misfit)
    # Measured (CPU, float64): 2 -> ~1.28, 4 -> ~0.28, 6 -> ~0.075, 8 -> ~0.035.
    assert misfit[2] > 2 * misfit[4]  # 2nd -> 4th order: large reduction
    assert misfit[4] > 2 * misfit[6]  # 4th -> 6th order: large reduction
    assert misfit[8] < misfit[4]  # 8th clearly beats 4th (robust, away from floor)
    assert misfit[2] > 0.5  # 2nd-order dispersion is large
    assert misfit[6] < 0.12  # 6th/8th order is accurate at 10 ppw


def test_pml_reflection_is_small_with_sufficient_width() -> None:
    """A sufficiently wide asymmetric PML suppresses boundary reflection.

    A 1-D homogeneous acoustic experiment records the pressure near the left
    edge while the right boundary absorbs with a PML of varying width.
    ``pml_width`` adds absorbing cells *outside* the model, so the model interior
    is held fixed (constant receiver-to-model-edge distance) and only the PML
    thickness varies. Subtracting a reflection-free reference (right model edge
    pushed far away) isolates the energy reflected by the PML. Converts the PML
    from a "trusted" component into a measured one (Workstream A1.4 / H).
    """
    dtype = torch.double
    c = 1500.0
    rho = 2000.0
    freq = 25.0
    dx = 5.0
    dt = 0.6 * dx / c
    left_pml = 30
    src_i = left_pml + 10
    rec_i = src_i + 15
    model_nx = rec_i + 60  # receiver-to-(right model edge) distance, constant
    nt = 400
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[[src_i]]])
    receiver_locations = torch.tensor([[[rec_i]]])

    def trace(nx: int, right_pml: int) -> torch.Tensor:
        v = torch.full((nx,), c, dtype=dtype)
        r = torch.full((nx,), rho, dtype=dtype)
        return acoustic(
            v,
            r,
            [dx],
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=4,
            pml_width=[left_pml, right_pml],
            max_vel=c,
            pml_freq=freq,
        )[-2][0, 0]

    reference = trace(rec_i + 2000, 30)

    coeffs = []
    for width in (2, 5, 10, 20):
        t = trace(model_nx, width)
        n = min(len(t), len(reference))
        coeffs.append(((t[:n] - reference[:n]).norm() / reference[:n].norm()).item())

    # With damping normalised by each face's own width, five cells already
    # reduce the residual near the numerical floor. At that floor, 10 and 20
    # cells need not be strictly ordered, so test the physical reduction and
    # low residual rather than monotonicity of sub-0.1% differences.
    assert coeffs[0] > 0.05  # a 2-cell PML reflects appreciably
    assert coeffs[1] < coeffs[0] / 10
    assert coeffs[2] < coeffs[1]
    assert max(coeffs[2:]) < 0.002


def test_pml_reflection_decreases_with_width_2d() -> None:
    """A wider PML reflects less in a 2-D acoustic domain.

    The 1-D acoustic PML test isolates one boundary by reference-subtraction
    against a far edge; in 2-D a point source radiates to all four faces, so we
    instead record a single off-axis pressure receiver and subtract a
    reflection-free reference obtained by re-running with a very wide (near-
    perfect) PML. The physical direct field and cylindrical coda cancel, leaving
    only the finite-width PML's reflected energy. Exercises the separately-
    compiled 2-D acoustic PML kernel against a ground truth (A1.4 / H). The
    acoustic PML absorbs strongly, so the transition is sampled at narrow widths
    (2/3/4) to stay above the absorption floor.
    """
    dtype = torch.double
    c = 1500.0
    rho = 2000.0
    freq = 25.0
    dx = 5.0
    dt = 0.5 * dx / c
    n = 60  # square model interior, identical for every run
    src = (n // 2, n // 2)
    rec = (n // 2 + 15, n // 2 + 15)  # off-axis: sees reflections from all faces
    v = torch.full((n, n), c, dtype=dtype)
    rho_model = torch.full((n, n), rho, dtype=dtype)
    diag = (2 * (n * dx) ** 2) ** 0.5
    nt = int((1.5 * diag / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])

    def trace(width: int) -> torch.Tensor:
        return acoustic(
            v,
            rho_model,
            dx,
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=4,
            pml_width=width,
            max_vel=c,
            pml_freq=freq,
        ).receiver_amplitudes_p[0, 0]

    # A 30-cell PML is effectively reflection-free at this scale (the reference).
    reference = trace(30)
    widths = (2, 3, 4)
    coeffs = [((trace(w) - reference).norm() / reference.norm()).item() for w in widths]

    # Measured (CPU, float64): ~0.224, 0.031, 0.0060 for width 2/3/4.
    assert all(coeffs[i] < coeffs[i - 1] for i in range(1, len(coeffs))), (
        f"PML reflection not monotonically decreasing in width: {coeffs}"
    )
    assert coeffs[0] > 0.05  # a 2-cell PML reflects appreciably
    assert coeffs[0] > 5 * coeffs[1]  # widening 2 -> 3 cells sharply cuts it
    assert coeffs[-1] < 0.02  # a 4-cell acoustic PML reflects < 2% to the receiver


def test_pml_residual_energy_decreases_with_width_3d() -> None:
    """A wider PML traps less residual energy in a 3-D acoustic domain.

    The 3-D counterpart of the acoustic PML-width test, exercising the
    separately-compiled 3-D acoustic PML kernels in float64 (A1.4 / H). A
    pressure point source radiates to all six faces, so the 1-D
    reference-subtraction trick cannot isolate a single boundary; instead we
    measure the total pressure energy left inside the fixed-size model interior
    long after the pulse has had time to cross and exit it. The interior is the
    same size for every run and the number of time steps is fixed, so the direct
    field has departed identically in all cases and the only thing that differs
    is how much energy the PML reflected back or failed to absorb.

    ``pml_width`` pads absorbing cells *outside* the model, so the returned
    pressure wavefield has shape ``model + 2*width`` per dimension; we crop back
    to the central model box before measuring its energy. ``nt`` is sized to two
    crossings of the *widest* padded domain so every run has settled into the
    asymptotic (PML-reflection-only) regime — the acoustic PML absorbs so well
    that, sized only to the interior, the residual is dominated by transient
    energy still clearing the larger padded domains and the trend is masked.
    """
    dtype = torch.double
    c = 1500.0
    rho = 2000.0
    freq = 25.0
    dx = 5.0
    dt = 0.5 * dx / c
    n = 32  # cubic model interior, identical for every run
    widths = (2, 5, 10)
    src = (n // 2,) * 3
    model = torch.full((n, n, n), c, dtype=dtype)
    rho_model = torch.full((n, n, n), rho, dtype=dtype)
    big = n + 2 * max(widths)  # widest padded domain
    diag = (3 * (big * dx) ** 2) ** 0.5
    nt = int((2.0 * diag / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])

    def interior_residual(width: int) -> float:
        wf = acoustic(
            model,
            rho_model,
            dx,
            dt,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=source_locations,
            accuracy=4,
            pml_width=width,
            max_vel=c,
            pml_freq=freq,
        ).pressure_0  # final pressure wavefield, padded by `width` cells per face
        interior = wf[:, width : width + n, width : width + n, width : width + n]
        return interior.norm().item()

    energies = [interior_residual(width) for width in widths]
    # Normalise to the width-2 case -> dimensionless retained-energy ratios.
    coeffs = [e / energies[0] for e in energies]

    # Measured (CPU, float64): 1.0, 0.061, 0.0056 for width 2/5/10.
    assert all(energies[i] < energies[i - 1] for i in range(1, len(energies))), (
        f"PML residual energy not monotonically decreasing in width: {coeffs}"
    )
    assert coeffs[1] < 0.3  # widening 2 -> 5 cells sharply cuts trapped energy
    assert coeffs[-1] < 0.05  # a 10-cell PML retains < 5% of the width-2 residual


def test_second_order_hvp_equals_fd_of_gradient() -> None:
    """The acoustic Hessian-vector product equals a finite difference of grad.

    The double-backward (one acoustic Born forward + one acoustic Born backward,
    see docs/acoustic_born_derivation.md section 5) computes H*h without forming
    H. Checked against ``(grad J(v + eps h) - grad J(v - eps h)) / (2 eps)`` for
    a data-misfit loss and an interior direction ``h``.

    ``nt`` must be long enough for ``h`` (which varies over the whole 10x10
    grid) to be exercised everywhere, not just near the source/receiver: the
    Hessian-vector product at a given cell requires the wave to travel
    source->cell->receiver, so cells far from the direct path need a round
    trip through most of the domain to register at the receiver at all. Using
    the same crossing-time formula as ``run_backend_parity``/``run_direct``
    (``2 * norm(nx * dx) / min_c``), with domain size ``nx * dx = (60, 60)``
    m and ``min_c ~= 1500`` m/s: ``2 * sqrt(60**2 + 60**2) / 1500 = 0.113 s``,
    i.e. >= 283 steps at ``dt = 4e-4`` s. The previous ``nt=16``
    (``nt * dt = 6.4`` ms) was shorter than even the *direct*
    source-receiver travel time (``2 * dx / min_c = 8`` ms), so the receiver
    trace was dominated by the finite-difference stencil's local coupling
    (its numerical domain of dependence grows much faster than the physical
    wave speed at this CFL number) rather than genuine wave-equation
    transport -- verified by inspecting the raw receiver trace, which was
    already the same order of magnitude at step 15 as at step 29 (well past
    the true 8 ms arrival). ``nt=300`` (``nt * dt = 0.12`` s) clears the
    283-step requirement with margin; HVP-vs-FD relative error at this
    ``nt`` is ~3e-7 (measured), comfortably under the ``1e-6`` assertion
    below.
    """
    ny, nx = 10, 10
    dx = 6.0
    dt = 4e-4
    nt = 300
    torch.manual_seed(0)
    v0 = (
        torch.full((ny, nx), 1500.0, dtype=torch.float64)
        + torch.rand(ny, nx, dtype=torch.float64) * 30
    )
    rho = torch.full((ny, nx), 2200.0, dtype=torch.float64)
    source_amplitudes = torch.randn(1, 1, nt, dtype=torch.float64)
    source_locations = torch.tensor([[[ny // 2, nx // 2]]])
    receiver_locations = torch.tensor([[[ny // 2, nx // 2 - 2]]])
    torch.manual_seed(1)
    h = torch.rand(ny, nx, dtype=torch.float64)

    kwargs = {
        "source_amplitudes_volume_rate": source_amplitudes,
        "source_locations_volume_rate": source_locations,
        "receiver_locations_p": receiver_locations,
        "pml_width": 3,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
        "accuracy": 4,
    }

    def grad_v(v_in: torch.Tensor) -> torch.Tensor:
        vv = v_in.clone().requires_grad_(True)
        out = acoustic(vv, rho, dx, dt, **kwargs)
        loss = out[-3].pow(2).sum()  # pressure receiver (2-D: 3 receiver types)
        (g,) = torch.autograd.grad(loss, vv, create_graph=True)
        return g

    vv = v0.clone().requires_grad_(True)
    out = acoustic(vv, rho, dx, dt, **kwargs)
    loss = out[-3].pow(2).sum()
    (g,) = torch.autograd.grad(loss, vv, create_graph=True)
    (hvp,) = torch.autograd.grad(g, vv, grad_outputs=h)

    eps = 1e-1
    fd = (grad_v(v0 + eps * h) - grad_v(v0 - eps * h)) / (2 * eps)
    assert (hvp - fd).abs().max() / fd.abs().max() < 1e-6


def test_second_order_hvp_equals_fd_of_gradient_odd_nt() -> None:
    """Same as above with an ODD ``nt``.

    Regression test: ``_acoustic_born_swap`` (the HVP double-backward's
    ping-pong closure) swapped the wrong wavefield block (the single-buffer
    "phi" block instead of the actually ping-ponged "psi" block) -- the same
    bug class fixed in ``acoustic_born.py`` by commit 5969cb0. The HVP call
    always runs as a single chunk (``backward_callback=None``), so the swap
    only fires when ``step_nt = nt`` (at ``step_ratio == 1``) is odd; every
    other HVP/gradgradcheck test in this file uses an even ``nt`` and so
    never exercised this path.

    ``nt`` uses the same domain-crossing conditioning as the even-``nt``
    version above (``nt=300``, itself replacing a too-short ``nt=16`` --
    see that docstring for the arithmetic), offset by one step to stay odd:
    ``nt=301`` gives ``nt * dt = 0.1204`` s, still comfortably past the
    283-step (``0.113`` s) domain-crossing requirement; HVP-vs-FD relative
    error at this ``nt`` is ~3e-7 (measured).
    """
    ny, nx = 10, 10
    dx = 6.0
    dt = 4e-4
    nt = 301
    torch.manual_seed(0)
    v0 = (
        torch.full((ny, nx), 1500.0, dtype=torch.float64)
        + torch.rand(ny, nx, dtype=torch.float64) * 30
    )
    rho = torch.full((ny, nx), 2200.0, dtype=torch.float64)
    source_amplitudes = torch.randn(1, 1, nt, dtype=torch.float64)
    source_locations = torch.tensor([[[ny // 2, nx // 2]]])
    receiver_locations = torch.tensor([[[ny // 2, nx // 2 - 2]]])
    torch.manual_seed(1)
    h = torch.rand(ny, nx, dtype=torch.float64)

    kwargs = {
        "source_amplitudes_volume_rate": source_amplitudes,
        "source_locations_volume_rate": source_locations,
        "receiver_locations_p": receiver_locations,
        "pml_width": 3,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
        "accuracy": 4,
    }

    def grad_v(v_in: torch.Tensor) -> torch.Tensor:
        vv = v_in.clone().requires_grad_(True)
        out = acoustic(vv, rho, dx, dt, **kwargs)
        loss = out[-3].pow(2).sum()  # pressure receiver (2-D: 3 receiver types)
        (g,) = torch.autograd.grad(loss, vv, create_graph=True)
        return g

    vv = v0.clone().requires_grad_(True)
    out = acoustic(vv, rho, dx, dt, **kwargs)
    loss = out[-3].pow(2).sum()
    (g,) = torch.autograd.grad(loss, vv, create_graph=True)
    (hvp,) = torch.autograd.grad(g, vv, grad_outputs=h)

    eps = 1e-1
    fd = (grad_v(v0 + eps * h) - grad_v(v0 - eps * h)) / (2 * eps)
    assert (hvp - fd).abs().max() / fd.abs().max() < 1e-6


def test_second_order_gradgradcheck() -> None:
    """Gradgradcheck of acoustic w.r.t. v, rho, and source amplitudes.

    Source at (2, 2), receiver at (2, 1): direct travel distance ``dx = 6``
    m, direct travel time ``6 / 1500 = 4`` ms. The previous ``nt=10``
    (``nt * dt = 4`` ms) sat exactly at that direct-arrival time, i.e. the
    receiver had not genuinely seen the source's wave yet -- any signal at
    the receiver was purely the finite-difference stencil's local coupling
    (verified: at this grid's spacing/accuracy the receiver trace is
    already non-negligible after 1-2 steps, well before the true 4 ms
    arrival). ``nt=50`` (``nt * dt = 20`` ms) gives 5x margin over the
    direct-arrival time and was verified to produce a genuine multi-cycle
    reverberating waveform, not just the initial near-field ramp.

    This does not reach full round-trip coverage of the whole 5x5 grid (the
    ``2 * norm(nx * dx) / min_c`` formula used elsewhere in this file gives
    ``2 * sqrt(30**2 + 30**2) / 1500 = 0.057`` s, i.e. ~141 steps): unlike
    the HVP tests above, gradgradcheck's cost scales with ``nt`` through
    both the per-step forward/backward cost *and* the number of source
    amplitude elements perturbed (``samp`` has ``nt`` entries, each
    individually finite-differenced), so 141 steps would be impractically
    slow for a routine test. ``nt=50`` is a practical balance, verified to
    still run in ~40s.
    """
    ny, nx = 5, 5
    dx = 6.0
    dt = 4e-4
    nt = 50
    torch.manual_seed(0)
    v = (
        torch.full((ny, nx), 1500.0, dtype=torch.float64)
        + torch.rand(ny, nx, dtype=torch.float64) * 30
    ).requires_grad_(True)
    rho = (
        torch.full((ny, nx), 2200.0, dtype=torch.float64)
        + torch.rand(ny, nx, dtype=torch.float64) * 30
    ).requires_grad_(True)
    samp = torch.randn(1, 1, nt, dtype=torch.float64).requires_grad_(True)
    loc = torch.tensor([[[ny // 2, nx // 2]]])
    rloc = torch.tensor([[[ny // 2, nx // 2 - 1]]])

    def f(v_: torch.Tensor, rho_: torch.Tensor, s_: torch.Tensor) -> torch.Tensor:
        out = acoustic(
            v_,
            rho_,
            dx,
            dt,
            source_amplitudes_volume_rate=s_,
            source_locations_volume_rate=loc,
            receiver_locations_p=rloc,
            pml_width=2,
            max_vel=2000.0,
            nt=nt,
            pml_freq=25.0,
            accuracy=4,
        )
        return out[-3]

    assert torch.autograd.gradgradcheck(
        f, (v, rho, samp), eps=1e-3, atol=2e-3, rtol=2e-3
    )


def test_first_order_gradient_finite_and_deterministic() -> None:
    """Plain first-order gradients are finite, non-zero, and deterministic."""
    ny, nx = 12, 12
    dx = 5.0
    dt = 0.004
    nt = 25
    source_amplitudes = ricker(25.0, nt, dt, 0.04).reshape(1, 1, nt).double()
    source_locations = torch.tensor([[[ny // 2, nx // 2]]])
    receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 3]]])

    def grad_v() -> torch.Tensor:
        v = (torch.ones(ny, nx, dtype=torch.float64) * 1500.0).requires_grad_(True)
        rho = torch.ones(ny, nx, dtype=torch.float64) * 1000.0
        out = acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            accuracy=2,
        ).receiver_amplitudes_p
        return torch.autograd.grad(out.pow(2).sum(), v)[0]

    g0 = grad_v()
    g1 = grad_v()
    assert torch.isfinite(g0).all()
    assert g0.abs().sum() > 0
    assert torch.equal(g0, g1)


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_gradient(use_reentrant: bool) -> None:
    """Gradient checkpointing must not change the gradient (issue #120)."""
    ny, nx, dx = 25, 25, 5.0
    nt, freq, dt0, max_vel = 60, 25.0, 0.004, 1600.0
    n_shots = 2

    v = 1500.0 * torch.ones(ny, nx)
    v[ny // 2 :] = 1600.0
    rho = 1000.0 * torch.ones(ny, nx)

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    source_locations[..., 1] = 5
    source_locations[:, 0, 0] = torch.tensor([ny // 4, 3 * ny // 4])
    receiver_locations = torch.zeros(n_shots, nx, 2, dtype=torch.long)
    receiver_locations[..., 1] = 5
    receiver_locations[:, :, 0] = torch.arange(nx).repeat(n_shots, 1)

    dt, step_ratio = cfl_condition(dx, dx, dt0, max_vel)
    source_amplitudes = upsample(
        ricker(freq, nt, dt0, 1.5 / freq).repeat(n_shots, 1, 1), step_ratio
    )

    def call(model: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return acoustic(
            model,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            model_gradient_sampling_interval=step_ratio,
        )

    run_checkpoint_equivalence(v, call, use_reentrant=use_reentrant)


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    ny, nx = 8, 8
    v = torch.full((ny, nx), 1500.0)
    rho = torch.full((ny, nx), 2200.0)
    source_amplitudes = torch.zeros(1, 1, 8)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        acoustic(
            v,
            rho,
            5.0,
            0.001,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            max_vel=2000.0,
            backend="eager",
            storage_compression=True,
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_acoustic_output_names(ndim) -> None:
    """The returned PropagatorOutput's field names match acoustic_output_names.

    ``out._names`` (set by ``_engine.named_output`` from ``SPEC.output_names``)
    must equal ``acoustic_output_names(ndim)`` exactly, in order: this is the
    contract that lets ``out.receiver_amplitudes_p`` and the other named
    fields resolve to the correct positional tensor for every ``ndim``.
    """
    from deepwave.acoustic import acoustic_output_names

    n = 8
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0)
    rho = torch.full(shape, 2200.0)
    source_amplitudes_volume_rate = torch.zeros(1, 1, 8)
    source_amplitudes_volume_rate[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    out = acoustic(
        v,
        rho,
        5.0,
        0.004,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=loc,
        receiver_locations_p=loc,
        pml_width=3,
    )
    assert out._names == tuple(acoustic_output_names(ndim))
    for name in out._names:
        assert hasattr(out, name)


@pytest.mark.parametrize("nt", [3, 4])
def test_pml_state_hessian_vector_matches_autograd(nt: int) -> None:
    """The Born-backed double backward includes terminal PML cotangents."""
    results = []
    for backend in ("compiled", "eager"):
        generator = torch.Generator().manual_seed(913)
        v = torch.full((12, 12), 1500.0, dtype=torch.float64, requires_grad=True)
        rho = torch.full_like(v, 2200.0, requires_grad=True)
        pressure = (
            1000 * torch.randn(1, 20, 20, dtype=v.dtype, generator=generator)
        ).requires_grad_()
        out = acoustic(
            v,
            rho,
            5.0,
            0.0004,
            pressure_0=pressure,
            origin=[0, 0],
            nt=nt,
            pml_width=4,
            pml_freq=25.0,
            max_vel=2000.0,
            backend=backend,
        )
        loss = sum(
            (
                value * torch.randn(value.shape, dtype=value.dtype, generator=generator)
            ).mean()
            for value in out.auxiliary_state.values()
        )
        leaves = (v, rho, pressure)
        first = torch.autograd.grad(loss, leaves, create_graph=True)
        directions = tuple(
            torch.randn(g.shape, dtype=g.dtype, generator=generator) for g in first
        )
        second = torch.autograd.grad(first, leaves, grad_outputs=directions)
        assert all(torch.isfinite(g).all() for g in second)
        assert any(torch.count_nonzero(g) for g in second)
        results.append(second)
    torch.testing.assert_close(results[0], results[1], rtol=1e-8, atol=1e-10)

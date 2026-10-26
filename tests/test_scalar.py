"""Tests for deepwave.scalar."""

import re
from typing import Any, Optional, Union

import pytest
import torch
from helpers import (
    _default_wavefield_size,
    _normalize_pml_width,
    _set_coords,
    _set_sources,
    _zero_scalar_style_wavefields,
    auxiliary_state_arg,
    compute_backend_parity_grad_and_gradgrad,
    compute_backend_parity_nt,
    direct_1d,
    direct_2d_approx,
    direct_3d,
    elements_equal,
    output_tensors,
    run_checkpoint_equivalence,
    run_python_vs_compiled_backend_parity,
    run_reciprocity_check,
    scattered,
)
from torch.func import jvp, vmap

import deepwave._common
import deepwave._engine
import deepwave._location_interpolation
from deepwave import IGNORE_LOCATION, Scalar, scalar
from deepwave._backend_utils import USE_OPENMP
from deepwave._common import _critical_courant_number, set_boundary
from deepwave.discretisation import cfl_condition, cfl_condition_n, downsample, upsample
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


# ``python_backend=True`` selects torch.compile on an OpenMP build and the
# eager backend otherwise -- what v0.0.27 did, save that its non-OpenMP
# choice, torch.jit.script, no longer exists. Tests below that loop over "a
# Python backend vs the compiled C backend" use that same resolution rather
# than hardcoding one, so they exercise whichever path this build actually
# takes.
_PYTHON_BACKEND_TRUE_EQUIVALENT = "torch_compile" if USE_OPENMP else "eager"


def scalarprop(
    model: torch.Tensor,
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
    receiver_locations: Optional[torch.Tensor],
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Optional[Union[int, list[int]]] = None,
    survey_pad: Optional[Union[int, list[int]]] = None,
    origin: Optional[list[int]] = None,
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
) -> tuple[torch.Tensor, ...]:
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

    # psi*/zeta* are the absorbing boundary's memory variables, not physical
    # fields, so they are not direct keyword arguments; they travel in the
    # opaque auxiliary_state. Building one by hand is what the tests need and
    # users do not: a run's own auxiliary_state is handed straight back.
    psi_zeta_state = auxiliary_state_arg(
        psiz_m1=psiz_m1,
        psiy_m1=psiy_m1,
        psix_m1=psix_m1,
        zetaz_m1=zetaz_m1,
        zetay_m1=zetay_m1,
        zetax_m1=zetax_m1,
    )

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
            initial_state=psi_zeta_state,
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
        initial_state=psi_zeta_state,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def scalarpropchained(
    model: torch.Tensor,
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
    receiver_locations: Optional[torch.Tensor],
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Optional[Union[int, list[int]]] = None,
    survey_pad: Optional[Union[int, list[int]]] = None,
    origin: Optional[list[int]] = None,
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
) -> tuple[torch.Tensor, ...]:
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
    auxiliary_state = auxiliary_state_arg(
        psiz_m1=psiz_m1,
        psiy_m1=psiy_m1,
        psix_m1=psix_m1,
        zetaz_m1=zetaz_m1,
        zetay_m1=zetay_m1,
        zetax_m1=zetax_m1,
    )

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
        out = scalar(
            model,
            dx,
            dt,
            source_amplitudes=segment_source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wfc,
            wavefield_m1=wfp,
            initial_state=auxiliary_state,
            nt=segment_nt,
            model_gradient_sampling_interval=step_ratio,
            **prop_kwargs,
        )
        wfc = out.wavefield_0
        wfp = out.wavefield_m1
        auxiliary_state = {"auxiliary_state": out.auxiliary_state}
        segment_receiver_amplitudes = out.receiver_amplitudes
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

    state = auxiliary_state["auxiliary_state"] if auxiliary_state is not None else None
    return (wfc, wfp, state, receiver_amplitudes)


def test_pml_width_list() -> None:
    """Verify that PML width supplied as int and list give same result."""
    _, actual_int = run_direct(propagator=scalarprop, prop_kwargs={"pml_width": 20})
    _, actual_list = run_direct(
        propagator=scalarprop,
        prop_kwargs={"pml_width": [20, 20, 20, 20]},
    )
    assert torch.allclose(actual_int, actual_list)


def test_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_direct(propagator=scalarprop, prop_kwargs={"backend": "torch_compile"})


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((50,), (5,)),
        pytest.param((50, 50), (5, 5), marks=pytest.mark.slow),
        pytest.param((10, 10, 10), (5, 5, 5), marks=pytest.mark.slow),
    ],
)
def test_direct(nx, dx, numerics_device):
    """Test propagation in a constant model."""
    if len(nx) == 1:
        expected_diffs = {2: 315, 4: 35, 6: 10, 8: 8}
    elif len(nx) == 2:
        expected_diffs = {2: 16, 4: 1.48, 6: 0.22, 8: 0.048}
    else:
        expected_diffs = {2: 1.4, 4: 0.5, 6: 0.2, 8: 0.04}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
            expected, actual = run_direct(
                nx=nx,
                device=torch.device(numerics_device),
                dx=dx,
                propagator=scalarprop,
                prop_kwargs={"backend": python, "accuracy": accuracy},
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
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
            propagator=scalarprop,
            nx=nx,
            dx=dx,
            nt=25000 if len(nx) == 1 else 2500,
            prop_kwargs={"backend": python},
        )
        for outi in output_tensors(out[:-1]):
            assert outi.norm() < 4e-5


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        actual_cpu = run_forward(
            propagator=scalarprop,
            device=torch.device("cpu"),
            prop_kwargs={"backend": python},
        )
        actual_gpu = run_forward(
            propagator=scalarprop,
            device=torch.device("cuda"),
            prop_kwargs={"backend": python},
        )
        # Check wavefields
        for cpui, gpui in zip(
            output_tensors(actual_cpu[:-1]), output_tensors(actual_gpu[:-1])
        ):
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
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        expected, actual = run_direct(
            c=torch.tensor([[[1500.0]], [[1600.0]]]),
            propagator=scalarprop,
            prop_kwargs={"backend": python},
        )
        diff = (expected - actual.cpu()).flatten()
        assert diff.norm().item() < expected_diff
        actuals[python] = actual
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

    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["backend"] = python

        model = (
            torch.ones(*nx, device=device, dtype=dtype) * c
            + torch.randn(*nx, device=device, dtype=dtype) * dc
        )

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

        for ofi, oii in zip(output_tensors(out_filled), output_tensors(out_ignored)):
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
            [
                ay,
                by,
                dbydy,
                ax,
                bx,
                dbxdx,
            ],
            [
                dy,
                dx,
            ],
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            # ``boundary`` config arg (all-PML here): scalar_func expects it
            # right after ``pml_width`` (see the ``scalar`` entry point).
            set_boundary(None, pml_width, 2),
            n_batch,
            None,
            None,
            1,
            "device",
            ".",
            False,
            # ``backward_callback_needs_forward_wavefield``: scalar's 16th
            # and last leading (non-tensor) argument, immediately before the
            # tensor block.
            False,
            c,
            source_amplitudes,
            sources_i,
            receivers_i,
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
    out_python = wrap("eager")
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op)


def test_scalarfunc() -> None:
    """Test scalar_func with different time steps."""
    run_scalarfunc(nt=4)
    run_scalarfunc(nt=5)


def test_backend_parity() -> None:
    """Test backend parity in a 2D model."""
    run_backend_parity(propagator=scalarprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((4,), (5,)),
        ((4, 3), (5, 5)),
        ((4, 3, 3), (6, 6, 6)),
    ],
)
def test_backend_parity_2nd_order(nx, dx) -> None:
    """Test backend parity with a 2nd order accurate propagator."""
    run_backend_parity(propagator=scalarprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2})


def test_backend_parity_6th_order() -> None:
    """Test backend parity with a 6th order accurate propagator."""
    run_backend_parity(propagator=scalarprop, prop_kwargs={"accuracy": 6})


def test_backend_parity_8th_order() -> None:
    """Test backend parity with a 8th order accurate propagator."""
    run_backend_parity(propagator=scalarprop, nx=(4, 4), prop_kwargs={"accuracy": 8})


def test_model_gradient_quadrature_error() -> None:
    """Finite-record sampling error can differ from the full derivative.

    The model (velocity) gradient is a time integral of the forward and adjoint
    wavefields. Deepwave samples that integral only every
    ``step_ratio * model_gradient_sampling_interval`` internal steps and
    multiplies by that stride. At stride 1 this is the full finite-call
    derivative. Coarser sampling can also
    integrate suitable bandlimited correlations exactly; this finite record
    is a counterexample, not a universal claim of quadrature error. This is an
    absolute finite-difference
    gradient check (a directional derivative, in float64) that quantifies the
    quadrature error as a function of the stride.

    Part 1 fixes ``step_ratio == 1`` (so the forward is identical for every
    stride) and sweeps ``model_gradient_sampling_interval`` -- which sets the
    kernel stride directly -- to give a clean, apples-to-apples comparison. The
    error is ~1e-9 at stride 1 (the adjoint is exact) and grows monotonically:
    measured relative errors ~3e-2 (stride 2), ~6e-1 (stride 4), ~2 (stride 8).

    Part 2 exercises the genuine CFL path: at a real ``step_ratio > 1`` (a
    ``dt`` above the CFL limit) the autograd model gradient departs measurably
    from the finite-difference truth, whereas at ``step_ratio == 1`` it agrees
    to machine precision.
    """
    torch.manual_seed(0)
    ny, nx, dx = 10, 10, 5.0
    max_vel, freq, nt = 1501.0, 15.0, 40
    v0 = 1500.0 * torch.ones(ny, nx, dtype=torch.float64)
    sl = torch.tensor([[[5, 5]]])
    rl = torch.tensor([[[7, 3]]])
    direction = torch.randn(ny, nx, dtype=torch.float64).sign()
    eps = 1e-3

    def loss(v: torch.Tensor, dt: float, mgsi: int) -> torch.Tensor:
        amp = ricker(freq, nt, dt, 1.0 / freq).reshape(1, 1, nt).double()
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=sl,
            receiver_locations=rl,
            pml_freq=freq,
            max_vel=max_vel,
            pml_width=3,
            model_gradient_sampling_interval=mgsi,
        )
        return (out[-1] ** 2).sum()

    def directional(dt: float, mgsi: int) -> tuple:
        v = v0.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(loss(v, dt, mgsi), v)
        dd_auto = (grad * direction).sum().item()
        lp = loss(v0 + eps * direction, dt, mgsi).item()
        lm = loss(v0 - eps * direction, dt, mgsi).item()
        dd_fd = (lp - lm) / (2 * eps)
        return dd_auto, dd_fd

    # Part 1: fixed forward (step_ratio == 1), vary the quadrature stride.
    dt1 = 0.0014  # below the CFL limit for this grid/velocity -> step_ratio == 1
    _, sr1 = cfl_condition_n([dx, dx], dt1, max_vel)
    assert sr1 == 1
    # Forward is independent of mgsi, so a single FD reference suffices.
    _, dd_fd_ref = directional(dt1, 1)
    errors = []
    for stride in (1, 2, 4, 8):
        dd_auto, _ = directional(dt1, stride)
        errors.append(abs(dd_auto - dd_fd_ref) / abs(dd_fd_ref))
    # Exact at stride 1, monotonically worse as the quadrature coarsens.
    assert errors[0] < 1e-6
    assert errors[0] < errors[1] < errors[2] < errors[3]

    # Part 2: genuine CFL path. step_ratio == 1 is exact; step_ratio > 1 is not.
    dd_auto_1, dd_fd_1 = directional(dt1, 1)
    assert abs(dd_auto_1 - dd_fd_1) / abs(dd_fd_1) < 1e-6
    dt_big = 0.006  # above the CFL limit -> step_ratio > 1
    _, sr_big = cfl_condition_n([dx, dx], dt_big, max_vel)
    assert sr_big > 1
    dd_auto_r, dd_fd_r = directional(dt_big, 1)
    assert abs(dd_auto_r - dd_fd_r) / abs(dd_fd_r) > 1e-3


def test_courant_number_default_bit_identical() -> None:
    """courant_number=None reproduces the default 0.6 behaviour bit-for-bit."""
    ny, nx, dx = 20, 20, 5.0
    dt, freq, nt, max_vel = 0.004, 15.0, 40, 1500.0
    v = 1500.0 * torch.ones(ny, nx)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[15, 5]]])
    amp = ricker(freq, nt, dt, 1.0 / freq).reshape(1, 1, nt)
    kwargs = {
        "source_amplitudes": amp,
        "source_locations": sl,
        "receiver_locations": rl,
        "pml_freq": freq,
        "max_vel": max_vel,
    }
    out_default = scalar(v, dx, dt, **kwargs)
    out_none = scalar(v, dx, dt, courant_number=None, **kwargs)
    for a, b in zip(out_default, out_none):
        assert elements_equal(a, b)


def test_courant_number_stable_and_fewer_steps() -> None:
    """0.9 * C_crit (order 2) is stable and uses fewer internal steps than 0.6.

    For the 2nd-order scheme the critical Courant number is exactly 1.0, so
    0.9 is a valid, still-stable choice. For this configuration the default
    (0.6) needs step_ratio 5 while 0.9 needs only 3 -- a 40% reduction in
    internal steps (a ~1.7x speed-up at the order-2 limit). The propagation
    must remain bounded (no CFL blow-up) over a long run.
    """
    ny, nx, dx = 40, 40, 5.0
    dt, freq, nt, max_vel = 0.006, 15.0, 400, 1500.0
    ccrit = _critical_courant_number(2, staggered=False)
    assert ccrit == pytest.approx(1.0)
    cn = 0.9 * ccrit

    _, sr_default = cfl_condition_n([dx, dx], dt, max_vel)
    _, sr_cn = cfl_condition_n([dx, dx], dt, max_vel, courant_number=cn)
    assert sr_default == 5
    assert sr_cn == 3  # 40% fewer internal steps than the default

    v = 1500.0 * torch.ones(ny, nx)
    sl = torch.tensor([[[20, 20]]])
    amp = ricker(freq, nt, dt, 1.0 / freq).reshape(1, 1, nt)
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=sl,
        pml_freq=freq,
        max_vel=max_vel,
        accuracy=2,
        courant_number=cn,
    )
    wavefield = out[0]
    assert torch.isfinite(wavefield).all()
    # Energy stays bounded: the short source has long since radiated into the
    # PML, so the final field is many orders below any unstable growth.
    assert wavefield.abs().max() < 1.0


def test_courant_number_above_critical_raises() -> None:
    """Requesting a Courant number above the critical value raises."""
    ny, nx, dx = 20, 20, 5.0
    dt, freq, nt, max_vel = 0.004, 15.0, 20, 1500.0
    v = 1500.0 * torch.ones(ny, nx)
    sl = torch.tensor([[[10, 10]]])
    amp = ricker(freq, nt, dt, 1.0 / freq).reshape(1, 1, nt)
    # Order 4 regular-grid critical Courant number is ~0.866; 0.95 is unstable.
    with pytest.raises(ValueError, match="exceeds the critical Courant number"):
        scalar(
            v,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=sl,
            pml_freq=freq,
            max_vel=max_vel,
            accuracy=4,
            courant_number=0.95,
        )


# Tests for propagator input guards (non-Tensor / empty / non-finite models,
# and shape-mismatched source_amplitudes).
def test_scalar_list_model_raises_type_error() -> None:
    """A Python list passed as the model raises a clear TypeError."""
    v = [[1500.0] * 10 for _ in range(10)]
    with pytest.raises(TypeError, match=re.escape("v must be a torch.Tensor")):
        scalar(v, 4.0, 0.001)


def test_scalar_empty_model_raises_value_error() -> None:
    """A 0-element model raises a clear ValueError."""
    v = torch.empty(0, 10)
    with pytest.raises(ValueError, match=re.escape("v must not be empty")):
        scalar(v, 4.0, 0.001)


def test_scalar_nan_model_raises_value_error() -> None:
    """A model containing NaN/Inf raises a ValueError naming the count."""
    v = 1500.0 * torch.ones(10, 10)
    v[3, 3] = float("nan")
    v[4, 4] = float("inf")
    sl = torch.tensor([[[5, 5]]])
    amp = torch.randn(1, 1, 10)
    with pytest.raises(ValueError, match=re.escape("v contains 2 non-finite values")):
        scalar(v, 4.0, 0.001, source_amplitudes=amp, source_locations=sl)


def test_max_vel_too_low_raises() -> None:
    """A too-low max_vel is rejected before stepping.

    This is the propagator-level regression test for the fix that made
    :func:`deepwave._common.velocity_extremes`'s underlying data-dependent
    read run even when ``max_vel`` is supplied, so :func:`set_max_vel`'s
    comparison is no longer starved of the real model maximum.
    """
    v = 1500.0 * torch.ones(10, 10)
    sl = torch.tensor([[[5, 5]]])
    amp = torch.randn(1, 1, 10)
    with pytest.raises(deepwave.DeepwaveValueError, match="conservative maximum"):
        scalar(v, 4.0, 0.001, source_amplitudes=amp, source_locations=sl, max_vel=1.0)


def test_transposed_source_amplitudes_shape_error() -> None:
    """A [time, shot, source] source_amplitudes reports the expected shape."""
    v = 1500.0 * torch.ones(10, 10)
    sl = torch.tensor([[[5, 5]]])
    # Correct shape is [shot=1, source=1, time=20]; this is transposed.
    amp = torch.randn(20, 1, 1)
    with pytest.raises(
        RuntimeError,
        match=re.escape("Deepwave expects [shot, source, time]"),
    ):
        scalar(v, 4.0, 0.001, source_amplitudes=amp, source_locations=sl)


def test_amplitudes_without_locations_names_missing_argument() -> None:
    """Providing source_amplitudes without source_locations names the gap."""
    v = 1500.0 * torch.ones(10, 10)
    amp = torch.randn(1, 1, 20)
    # receiver_locations is given so get_ndim can establish the number of
    # spatial dimensions independently of source_amplitudes/source_locations
    # -- otherwise a call with no locations or wavefields at all fails
    # earlier, in get_ndim, with an unrelated "could not be determined"
    # error before setup_propagator's source_amplitudes/source_locations
    # check is ever reached.
    rl = torch.tensor([[[5, 5]]])
    with pytest.raises(
        RuntimeError,
        match=re.escape("source_amplitudes was provided but source_locations was not"),
    ):
        scalar(v, 4.0, 0.001, source_amplitudes=amp, receiver_locations=rl)


def test_backend_parity_cfl() -> None:
    """Test backend parity with a timestep greater than the CFL limit."""
    run_backend_parity(propagator=scalarprop, dt=0.002, atol=5e-8, gradgrad=False)


def test_backend_parity_cfl_gradgrad() -> None:
    """Sampled compiled model gradients reject reverse-over-reverse AD."""
    with pytest.raises(RuntimeError, match="sampled model gradients"):
        run_backend_parity(
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


def test_backend_parity_odd_timesteps() -> None:
    """Test backend parity with one more timestep."""
    run_backend_parity(propagator=scalarprop, nt_add=1)


def test_backend_parity_one_shot() -> None:
    """Test backend parity with one shot."""
    run_backend_parity(propagator=scalarprop, num_shots=1)


def test_backend_parity_no_sources() -> None:
    """Test backend parity with no sources."""
    run_backend_parity(propagator=scalarprop, num_sources_per_shot=0)


def test_backend_parity_no_receivers() -> None:
    """Test backend parity with no receivers."""
    run_backend_parity(propagator=scalarprop, num_receivers_per_shot=0)


def test_backend_parity_survey_pad() -> None:
    """Test backend parity with survey_pad."""
    run_backend_parity(
        propagator=scalarprop,
        survey_pad=0,
        nx=(12, 9),
        provide_wavefields=False,
    )


def test_backend_parity_partial_wavefields() -> None:
    """Test backend parity with wavefields that do not cover the model."""
    run_backend_parity(
        propagator=scalarprop,
        origin=(0, 4),
        nx=(12, 9),
        wavefield_size=(4 + 6, 1 + 6),
    )


def test_backend_parity_chained() -> None:
    """Test backend parity when two propagators are chained."""
    run_backend_parity(propagator=scalarpropchained)


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_gradient(use_reentrant: bool) -> None:
    """Gradient checkpointing must not change the gradient (issue #120).

    The propagation is run in time segments, threading the full wavefield
    state (wavefield_0/m1, psi*, zeta*) from one segment to the next, with the
    non-final segment wrapped in ``torch.utils.checkpoint`` -- the pattern used
    in ``docs/example_checkpointing.py``. Checkpointing only trades compute for
    memory, so the gradient w.r.t. the velocity model must be identical to the
    single-segment (non-checkpointed) gradient. This guards against the
    propagator returning a ``list`` (which silently detaches every output of a
    reentrant checkpoint) and against reading ``ctx.saved_tensors`` more than
    once (which breaks the non-reentrant checkpoint unpack hook).
    """
    import torch.utils.checkpoint

    ny, nx, dx = 25, 25, 5.0
    nt, freq, dt0, max_vel = 80, 25.0, 0.004, 1600.0
    n_shots, pml = 2, 20

    v = 1500.0 * torch.ones(ny, nx)
    v[ny // 2 :] = 1600.0

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    source_locations[..., 1] = 5
    source_locations[:, 0, 0] = torch.tensor([ny // 4, 3 * ny // 4])
    receiver_locations = torch.zeros(n_shots, nx, 2, dtype=torch.long)
    receiver_locations[..., 1] = 5
    receiver_locations[:, :, 0] = torch.arange(nx).repeat(n_shots, 1)

    dt, step_ratio = cfl_condition(dx, dx, dt0, max_vel)
    source_amplitudes = upsample(
        ricker(freq, nt, dt0, 1.5 / freq).repeat(n_shots, 1, 1),
        step_ratio,
    )
    nt_up = source_amplitudes.shape[-1]

    aux_names = ("psiy_m1", "psix_m1", "zetay_m1", "zetax_m1")

    def segment(v, chunk, wf0, wfm1, *aux):
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=chunk,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            wavefield_0=wf0,
            wavefield_m1=wfm1,
            initial_state=auxiliary_state_arg(**dict(zip(aux_names, aux))),
            model_gradient_sampling_interval=step_ratio,
        )
        # A flat tuple of tensors, not the PropagatorOutput: a reentrant
        # checkpoint re-attaches only the top-level Tensor elements of what
        # the wrapped function returns, so tensors returned inside
        # auxiliary_state would come back detached and the state's gradient
        # path across the segment boundary would be lost.
        return (
            out.wavefield_0,
            out.wavefield_m1,
            *(out.auxiliary_state[n] for n in aux_names),
            out.receiver_amplitudes,
        )

    def grad_of(n_segments):
        v_in = v.clone().requires_grad_(True)
        state = [torch.zeros(n_shots, ny + 2 * pml, nx + 2 * pml) for _ in range(6)]
        rec = torch.zeros(n_shots, nx, nt_up)
        k = 0
        chunks = torch.chunk(source_amplitudes, n_segments, dim=-1)
        for i, chunk in enumerate(chunks):
            args = (v_in, chunk, *state)
            if i == len(chunks) - 1:
                out = segment(*args)
            else:
                out = torch.utils.checkpoint.checkpoint(
                    segment, *args, use_reentrant=use_reentrant
                )
            *state, r = out
            rec[..., k : k + chunk.shape[-1]] = r
            k += chunk.shape[-1]
        (rec**2).sum().backward()
        return v_in.grad.clone()

    reference = grad_of(1)
    checkpointed = grad_of(2)
    assert reference.norm() > 0
    assert torch.allclose(reference, checkpointed, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_gradient_named_return(use_reentrant: bool) -> None:
    """The named return must compose with ``torch.utils.checkpoint``.

    The named return (:class:`deepwave.PropagatorOutput`, always
    returned since ``return_dataclass`` was removed) subclasses ``tuple``
    precisely so that reentrant checkpoint (whose wrapping autograd
    ``Function`` propagates ``requires_grad`` only to Tensor / tuple-of-Tensor
    outputs) does not silently detach it -- the same failure mode as issue #120.
    The gradient w.r.t. the velocity model must therefore be identical to the
    non-checkpointed gradient.
    """
    import torch.utils.checkpoint

    ny, nx, dx = 25, 25, 5.0
    nt, freq, dt0, max_vel = 60, 25.0, 0.004, 1600.0
    n_shots = 2

    v = 1500.0 * torch.ones(ny, nx)
    v[ny // 2 :] = 1600.0

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    source_locations[..., 1] = 5
    source_locations[:, 0, 0] = torch.tensor([ny // 4, 3 * ny // 4])
    receiver_locations = torch.zeros(n_shots, nx, 2, dtype=torch.long)
    receiver_locations[..., 1] = 5
    receiver_locations[:, :, 0] = torch.arange(nx).repeat(n_shots, 1)

    dt, step_ratio = cfl_condition(dx, dx, dt0, max_vel)
    source_amplitudes = upsample(
        ricker(freq, nt, dt0, 1.5 / freq).repeat(n_shots, 1, 1),
        step_ratio,
    )

    def call(v_in: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # scalar() always returns a PropagatorOutput (a tuple subclass).
        return scalar(
            v_in,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            model_gradient_sampling_interval=step_ratio,
        )

    run_checkpoint_equivalence(v, call, use_reentrant)


@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_checkpoint_double_backprop(
    backend: Union[bool, str],
) -> None:
    """Double backprop must work through non-reentrant checkpoint (issue #120).

    Second-order gradients (here a Hessian-vector product) are differentiated
    through the double-backward node ``ScalarBackwardFunc.backward``, which also
    reads ``ctx.saved_tensors`` and so must read it exactly once. The Hessian-
    vector product is compared against the non-checkpointed value, which must be
    identical because checkpointing only trades compute for memory.

    ``use_reentrant=True`` is deliberately not tested: reentrant checkpoint is
    incompatible with second-order autograd in PyTorch itself (it raises when
    ``torch.autograd.grad`` / ``create_graph`` is used). The ``"compile"``
    backend is likewise excluded because ``torch.compile`` does not support
    double backward.
    """
    import torch.utils.checkpoint

    ny, nx, dx = 20, 20, 5.0
    nt, freq, dt0, max_vel = 40, 25.0, 0.004, 1600.0
    n_shots = 1

    v = 1500.0 * torch.ones(ny, nx)
    v[ny // 2 :] = 1550.0
    torch.manual_seed(0)
    vec = torch.randn(ny, nx)

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = 4
    receiver_locations = torch.zeros(n_shots, nx, 2, dtype=torch.long)
    receiver_locations[..., 1] = 4
    receiver_locations[:, :, 0] = torch.arange(nx).repeat(n_shots, 1)

    dt, step_ratio = cfl_condition(dx, dx, dt0, max_vel)
    source_amplitudes = upsample(
        ricker(freq, nt, dt0, 1.5 / freq).repeat(n_shots, 1, 1),
        step_ratio,
    )

    def run(model: torch.Tensor) -> torch.Tensor:
        return scalar(
            model,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            # The time samples are already at the internal CFL rate.
            # Second derivatives require unsampled model histories.
            model_gradient_sampling_interval=1,
            backend=backend,
        )[-1]

    def hessian_vector_product(checkpointed: bool) -> torch.Tensor:
        model = v.clone().requires_grad_(True)
        if checkpointed:
            rec = torch.utils.checkpoint.checkpoint(run, model, use_reentrant=False)
        else:
            rec = run(model)
        (grad,) = torch.autograd.grad((rec**2).sum(), model, create_graph=True)
        (hvp,) = torch.autograd.grad((grad * vec).sum(), model)
        return hvp

    reference = hessian_vector_product(checkpointed=False)
    checkpointed = hessian_vector_product(checkpointed=True)
    assert reference.norm() > 0
    assert torch.allclose(reference, checkpointed, atol=1e-5, rtol=1e-4)


def test_backend_parity_negative() -> None:
    """Backends agree for negative background velocity."""
    run_backend_parity(c=-1500, propagator=scalarprop)


def test_backend_parity_zero() -> None:
    """Backends agree for zero background velocity."""
    run_backend_parity(c=0, dc=0, propagator=scalarprop)


def test_backend_parity_different_pml() -> None:
    """Test backend parity with different pml widths."""
    run_backend_parity(propagator=scalarprop, pml_width=[0, 1, 5, 10], atol=5e-8)


def test_backend_parity_no_pml() -> None:
    """Test backend parity with no pml."""
    run_backend_parity(propagator=scalarprop, pml_width=0)


def test_backend_parity_different_dx() -> None:
    """Test backend parity with different dx values."""
    run_backend_parity(propagator=scalarprop, dx=(4, 5), dt=0.0005, atol=4e-8)


def test_backend_parity_single_cell() -> None:
    """Test backend parity with a single model cell."""
    run_backend_parity(
        propagator=scalarprop,
        nx=(1, 1),
        prop_kwargs={"accuracy": 2},
        num_shots=1,
        num_sources_per_shot=1,
        num_receivers_per_shot=1,
    )


def test_backend_parity_big() -> None:
    """Test backend parity with a big model."""
    run_backend_parity(
        propagator=scalarprop,
        nx=(5 + 2 * (3 + 3 * 2), 4 + 2 * (3 + 3 * 2)),
        atol=2e-8,
        gradgrad=False,
    )


def test_backend_parity_big_gradgrad() -> None:
    """Test backend parity with a big model."""
    run_backend_parity(
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

    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["backend"] = python

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

        # The scalar equation depends on v**2.
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

        # With zero velocity the source is multiplied by v**2 = 0.
        out = propagator(
            torch.zeros_like(model),
            dx,
            -dt,
            sources["amplitude"],
            sources["locations"],
            x_r,
            prop_kwargs=prop_kwargs,
        )
        assert torch.count_nonzero(out[0]) == 0
        assert torch.count_nonzero(out[-1]) == 0


def test_backend_parity_only_v_2d() -> None:
    """Test backend parity with only v requiring gradient."""
    run_backend_parity(
        propagator=scalarprop,
        source_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
    )


def test_backend_parity_only_source_2d() -> None:
    """Test backend parity with only source requiring gradient.

    With the other inputs fixed this map is linear. Compare its first
    derivative directly; the common helper also represents an exactly zero
    second derivative correctly, but computing it adds no evidence here.
    """
    run_backend_parity(
        propagator=scalarprop,
        v_requires_grad=False,
        wavefield_0_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        gradgrad=False,
    )


def test_backend_parity_only_wavefield_0_2d() -> None:
    """Test backend parity with only wavefield_0 requiring gradient.

    With the other inputs fixed this map is linear, so compare its first
    derivative directly without redundant second-derivative work.
    """
    run_backend_parity(
        propagator=scalarprop,
        v_requires_grad=False,
        source_requires_grad=False,
        wavefield_m1_requires_grad=False,
        psi_m1_requires_grad=False,
        zeta_m1_requires_grad=False,
        gradgrad=False,
    )


def test_backend_parity_v_batched() -> None:
    """Test backend parity using a different velocity for each shot."""
    run_backend_parity(propagator=scalarprop, c=torch.tensor([[[1500.0]], [[1600.0]]]))


@pytest.mark.parametrize(("num_shots"), [1, 2])
@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 3), (5, 5)),
        ((5, 3, 3), (6, 6, 6)),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA unavailable"
                ),
            ],
        ),
    ],
)
def test_storage(
    num_shots,
    nx,
    dx,
    record_property,
    device,
    c=1500,
    dc=100,
    freq=25,
    dt=0.001,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    num_scatter_receivers_per_shot=2,
    propagator=scalarprop,
    prop_kwargs=None,
    dtype=None,
):
    """Compare first and second derivatives at effective history stride one."""
    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if prop_kwargs is None:
        prop_kwargs = {}
    if "pml_width" not in prop_kwargs:
        prop_kwargs["pml_width"] = 3

    # Use the same material and source timing on CPU and CUDA. A separate
    # CPU generator keeps material draws from advancing the source RNG.
    generator = torch.Generator().manual_seed(1)
    model = (c + torch.randn(*nx, dtype=dtype, generator=generator) * dc).to(device)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    # Store uncompressed in device memory (default)
    modeld = model.clone()
    modeld.requires_grad_()
    out = propagator(
        modeld,
        dx.tolist(),
        dt,
        sources["amplitude"],
        sources["locations"],
        x_r,
        prop_kwargs=prop_kwargs,
    )

    loss = (out[-1] ** 2).sum()
    (gd,) = torch.autograd.grad(loss, modeld, create_graph=True)
    (g2d,) = torch.autograd.grad((gd**2).sum(), modeld)

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

    atol_model = [2, 0.02, 2e-3]

    for mode in modes:
        for compression in [False, True]:
            if mode == "device" and not compression:  # Default
                continue
            modeli = model.clone()
            modeli.requires_grad_()
            prop_kwargs = dict(prop_kwargs)
            prop_kwargs["storage_mode"] = mode
            prop_kwargs["storage_compression"] = compression
            out = propagator(
                modeli,
                dx.tolist(),
                dt,
                sources["amplitude"],
                sources["locations"],
                x_r,
                prop_kwargs=prop_kwargs,
            )

            loss = (out[-1] ** 2).sum()
            (gi,) = torch.autograd.grad(loss, modeli, create_graph=True)
            (g2i,) = torch.autograd.grad((gi**2).sum(), modeli)

            if compression:
                assert torch.allclose(gd, gi, atol=atol_model[len(nx) - 1])
                assert float(gd.detach().norm()) > 0
                assert float(g2d.norm()) > 0
                gradient_error = float((gd - gi).norm() / gd.norm())
                second_error = float((g2d - g2i).norm() / g2d.norm())
                record_property(f"{mode}_compressed_gradient_error", gradient_error)
                record_property(f"{mode}_compressed_second_error", second_error)
                # Eight-bit block quantisation gives approximate derivatives;
                # second derivatives can amplify it. Use a scoped relative
                # budget, rather than absolute limits calibrated at another dt.
                assert gradient_error < 0.05
                assert second_error < 0.1
            else:
                assert torch.allclose(
                    gd, gi, atol=gd.detach().abs().max().item() * 1e-5
                )
                assert torch.allclose(
                    g2d, g2i, atol=g2d.detach().abs().max().item() * 1e-5
                )


def run_direct(
    c: Union[float, torch.Tensor] = 1500,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.0001,
    nx: tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = scalarprop,
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
    dc: float = 100,
    nt: Optional[int] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
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
    dx: Union[float, list[float]],
    dt: float,
    nx: tuple[int, ...],
    num_shots: int,
    num_sources_per_shot: int,
    num_receivers_per_shot: int,
    propagator: Any,
    prop_kwargs: Optional[dict[str, Any]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def run_backend_parity(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 100,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.001,
    nx: tuple[int, ...] = (4, 3),
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

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    nt = compute_backend_parity_nt(nx, dx, min_c, freq, dt, nt_add)

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
    pml_width = _normalize_pml_width(pml_width, ndim)
    if provide_wavefields and wavefield_size is None:
        wavefield_size = _default_wavefield_size(nx, pml_width, ndim)
    wavefield_0, wavefield_m1, psi_m1, zeta_m1 = _zero_scalar_style_wavefields(
        provide_wavefields,
        wavefield_size,
        num_shots,
        ndim,
        dtype,
        device,
        wavefield_0_requires_grad,
        wavefield_m1_requires_grad,
        psi_m1_requires_grad,
        zeta_m1_requires_grad,
    )

    model.requires_grad_(v_requires_grad)
    if prop_kwargs is None:
        prop_kwargs = {}

    def wrap(python):
        prop_kwargs["backend"] = python
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
        # The auxiliary state is one output element holding several tensors;
        # the parity comparison is over tensors.
        return compute_backend_parity_grad_and_gradgrad(
            output_tensors(out), inputs, gradgrad
        )

    run_python_vs_compiled_backend_parity(wrap, "eager", atol=atol, rtol=rtol)


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


def test_vmap_v() -> None:
    """Test vmap over the velocity model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v_base = 1500.0
    n_shots = 2
    n_receivers_per_shot = 3

    # Create a batch of velocities
    batch_size = 3
    v_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        v_batch[i] = v_base + i * 100.0

    # Setup source and receivers
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

    def prop(v):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]

    # Run vmap
    res_vmap = vmap(prop)(v_batch)

    # Run loop
    res_loop = torch.stack([prop(v_batch[i]) for i in range(batch_size)])

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_jvp_v() -> None:
    """Test forward-mode AD (jvp) with respect to velocity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v0 = torch.full((ny, nx), 1500.0, device=device, dtype=dtype)

    # Perturbation
    v_tangent = torch.randn(ny, nx, device=device, dtype=dtype) * 10.0

    n_shots = 1
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2

    receiver_locations = torch.zeros(n_shots, 3, 2, dtype=torch.long, device=device)
    receiver_locations[..., 0] = ny // 2
    receiver_locations[..., 1] = torch.linspace(0, nx - 1, 3, device=device).long()

    source_amplitudes = (
        ricker(25.0, 30, dt, 0.05, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    def prop(v):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]

    # Run JVP
    _, jvp_out = jvp(prop, (v0,), (v_tangent,))

    # Finite Difference check
    epsilon = 1e-6
    out_plus = prop(v0 + epsilon * v_tangent)
    out_minus = prop(v0 - epsilon * v_tangent)
    fd_out = (out_plus - out_minus) / (2 * epsilon)

    # Check that JVP matches FD reasonably well
    assert torch.allclose(jvp_out, fd_out)


@pytest.mark.slow
def test_hessian_v() -> None:
    """Test Hessian-vector product (double backward) vs JVP of grad."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v0 = torch.full((ny, nx), 1500.0, device=device)
    v0.requires_grad_(True)

    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2

    receiver_locations = torch.zeros(1, 3, 2, dtype=torch.long, device=device)
    receiver_locations[..., 0] = ny // 2
    receiver_locations[..., 1] = torch.linspace(0, nx - 1, 3, device=device).long()

    source_amplitudes = ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1)

    def loss_func(v):
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]
        return (out**2).sum()

    # 1. Compute HVP using standard double backward
    loss = loss_func(v0)
    grads = torch.autograd.grad(loss, v0, create_graph=True)[0]
    v_tangent = torch.randn_like(v0)
    hvp_standard = torch.autograd.grad(torch.sum(grads * v_tangent), v0)[0]

    # 2. Compute HVP using functional jvp of grad
    def grad_func(v):
        return torch.func.grad(loss_func)(v)

    _, hvp_functional = jvp(grad_func, (v0,), (v_tangent,))

    assert torch.allclose(hvp_standard, hvp_functional, atol=1e-5)


@pytest.mark.slow
def test_vmap_grad_v() -> None:
    """Test vmap over grad with respect to velocity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v_base = 1500.0

    batch_size = 3
    v_batch = torch.full((batch_size, ny, nx), v_base, device=device)
    for i in range(batch_size):
        v_batch[i] += i * 100.0
    v_batch.requires_grad_(True)

    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2

    receiver_locations = torch.zeros(1, 3, 2, dtype=torch.long, device=device)
    receiver_locations[..., 0] = ny // 2
    receiver_locations[..., 1] = torch.linspace(0, nx - 1, 3, device=device).long()

    source_amplitudes = ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1)

    def loss_func(v):
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]
        return (out**2).sum()

    # vmap of grad
    vmapped_grad_func = vmap(torch.func.grad(loss_func))
    grads_vmap = vmapped_grad_func(v_batch)

    # loop of grad
    grads_loop = torch.stack(
        [torch.func.grad(loss_func)(v_batch[i]) for i in range(batch_size)]
    )

    assert torch.allclose(grads_vmap, grads_loop, atol=1e-5)


def test_vmap_error_check() -> None:
    """Ensure error is raised if non-property inputs are vmapped."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    v = torch.full((ny, nx), 1500.0, device=device)

    # Try to vmap over source amplitudes
    batch_size = 2
    source_amplitudes_batch = torch.randn(batch_size, 1, 1, 10, device=device)

    def prop(amps):
        return scalar(
            v,
            5.0,
            0.004,
            source_amplitudes=amps,
            source_locations=torch.zeros(1, 1, 2, dtype=torch.long, device=device),
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )

    with pytest.raises(NotImplementedError, match="Only property models"):
        vmap(prop)(source_amplitudes_batch)


def test_functorch_c_backend_error_is_actionable() -> None:
    """A ``vmap`` transform on the compiled backend must fail actionably.

    After the setup_context migration (C2.1) ``vmap``/``jvp`` reach the
    compiled-backend autograd ``Function`` instead of stopping earlier. The
    ctypes backend cannot process ``vmap``'s batched tensors, so rather than
    PyTorch's opaque default message ("does not have vmap support") Deepwave
    raises a message that names the transform and points at
    ``backend="eager"`` (which genuinely supports vmap over property
    models, covered by ``test_vmap_v``).

    Forward-mode AD (``jvp``) is *no longer* in this list: scalar now supports
    ``torch.func.jvp`` on the fast compiled backend via the Born relationship
    (C2.2, tested in ``tests/test_scalar_jvp.py``). The other propagators keep
    raising for ``jvp`` (they have no Born kernel).
    """
    ny, nx = 8, 8
    v = torch.full((ny, nx), 1500.0)
    source_amplitudes = torch.zeros(1, 1, 6)
    source_amplitudes[..., 1] = 1.0
    # Build the location tensors outside ``prop`` so the transform does not wrap
    # them (a tensor created inside the transform level is treated as batched,
    # which trips the earlier front-end vmap check instead of reaching the
    # Function); this mirrors normal usage where locations are set up once.
    source_locations = torch.full((1, 1, 2), 3, dtype=torch.long)
    receiver_locations = torch.full((1, 1, 2), 4, dtype=torch.long)

    def prop(model: torch.Tensor) -> torch.Tensor:
        # Default backend="compiled" -> the compiled C backend / Function.
        return scalar(
            model,
            5.0,
            0.004,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
        )[-1]

    with pytest.raises(RuntimeError, match=r'backend="eager"') as exc:
        vmap(prop)(torch.stack([v, v * 1.01]))
    assert "torch.func.vmap" in str(exc.value)


def test_spatial_accuracy_reduces_dispersion() -> None:
    """Higher spatial ``accuracy`` order must reduce numerical dispersion.

    In a homogeneous 1-D medium the continuum wave equation is non-dispersive,
    and the 1-D Green's function (``direct_1d``) is an accurate reference
    (a quadrature of the exact step-function response). We make the
    time-stepping error negligible with a tiny Courant number so that, for the
    low/medium accuracy orders, the misfit against the analytic solution is
    dominated by *spatial* numerical dispersion. A small residual floor
    (the cumsum-quadrature reference error, the O(dt^2) temporal term, and any
    PML reflection) limits the very highest order, so we assert the monotone
    *reduction* of dispersion with order rather than a convergence rate. This
    validates that the ``accuracy`` argument actually delivers higher-order
    forward accuracy (the existing accuracy tests only check gradient
    self-consistency, not forward fidelity).
    """
    dtype = torch.double
    c = 1500.0
    freq = 25.0
    dx = 6.0  # points-per-wavelength = c / (freq * dx) = 10
    dt = 0.05 * dx / c  # Courant 0.05 => time-stepping error negligible
    nx = 200
    src_i = 40
    rec_i = 160  # 120 cells = 12 wavelengths of propagation
    model = torch.full((nx,), c, dtype=dtype)
    nt = int((1.2 * nx * dx / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[[src_i]]])
    receiver_locations = torch.tensor([[[rec_i]]])
    # direct_1d expects the source injected with the opposite sign (matches
    # the convention used by run_direct / test_direct).
    expected = direct_1d(
        torch.tensor([float(rec_i)]),
        torch.tensor([float(src_i)]),
        torch.tensor([dx], dtype=dtype),
        dt,
        c,
        -amp[0, 0],
    )

    misfit = {}
    for accuracy in (2, 4, 6, 8):
        out = scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=accuracy,
            pml_width=20,
            max_vel=c,
            pml_freq=freq,
        )[-1][0, 0]
        n = min(len(out), len(expected))
        misfit[accuracy] = (
            (out[:n] - expected[:n]).norm() / expected[:n].norm()
        ).item()

    # Measured (CPU, float64): 2 -> ~1.0, 4 -> ~0.18, 6 -> ~0.040, 8 -> ~0.015.
    # Each higher order reduces the dispersion error by roughly 5x until the
    # residual floor (~0.015) is reached at the highest order.
    assert misfit[2] > 2 * misfit[4]  # 2nd -> 4th order: large reduction
    assert misfit[4] > 2 * misfit[6]  # 4th -> 6th order: large reduction
    assert misfit[8] < misfit[4]  # 8th clearly beats 4th (robust, away from floor)
    assert misfit[2] > 0.5  # 2nd-order dispersion over 12 wavelengths is large
    assert misfit[6] < 0.08  # 6th/8th order is accurate at 10 ppw


def test_spatial_accuracy_reduces_dispersion_2d() -> None:
    """Higher spatial ``accuracy`` reduces numerical dispersion in 2-D.

    The 2-D counterpart of ``test_spatial_accuracy_reduces_dispersion``,
    filling the gap between the existing 1-D and 3-D dispersion checks (2-D is
    the most common case in practice). The reference is the *exact*
    frequency-domain 2-D Green's function (``direct_2d_approx``, the
    ``i/4 H_0^(1)`` Hankel-function point-source response), which the simulation
    must converge to as the spatial order increases. A small Courant number
    keeps the time-stepping error negligible, so the misfit is dominated by
    *spatial* numerical dispersion and falls by roughly 5x per accuracy order
    until a small residual floor (the 2-D coda, the O(dt^2) temporal term, PML
    reflection) is reached.
    """
    dtype = torch.double
    c = 1500.0
    freq = 25.0
    dx = 6.0  # points-per-wavelength = c / (freq * dx) = 10
    dt = 0.08 * dx / c  # small Courant => time-stepping error negligible
    ny = 70
    nx = 170
    src = (ny // 2, 35)
    rec = (ny // 2, 135)  # 100 cells = 10 wavelengths along x
    model = torch.full((ny, nx), c, dtype=dtype)
    r_cells = abs(rec[1] - src[1])
    nt = int((1.3 * r_cells * dx / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])
    expected = direct_2d_approx(
        torch.tensor([float(rec[0]), float(rec[1])]),
        torch.tensor([float(src[0]), float(src[1])]),
        torch.tensor([dx, dx], dtype=dtype),
        dt,
        c,
        -amp[0, 0],
    )

    misfit = {}
    for accuracy in (2, 4, 6, 8):
        out = scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=accuracy,
            pml_width=20,
            max_vel=c,
            pml_freq=freq,
        )[-1][0, 0]
        n = min(len(out), len(expected))
        misfit[accuracy] = (
            (out[:n] - expected[:n]).norm() / expected[:n].norm()
        ).item()

    # Measured (CPU, float64): 2 -> ~1.10, 4 -> ~0.21, 6 -> ~0.039, 8 -> ~0.011.
    assert misfit[2] > 2 * misfit[4]  # 2nd -> 4th order: large reduction
    assert misfit[4] > 2 * misfit[6]  # 4th -> 6th order: large reduction
    assert misfit[8] < misfit[4]  # 8th clearly beats 4th (robust, away from floor)
    assert misfit[2] > 0.5  # 2nd-order dispersion over 10 wavelengths is large
    assert misfit[6] < 0.08  # 6th/8th order is accurate at 10 ppw


def test_spatial_accuracy_reduces_dispersion_3d(record_property) -> None:
    """Higher spatial ``accuracy`` reduces numerical dispersion in 3-D.

    The 3-D counterpart of ``test_spatial_accuracy_reduces_dispersion``. It
    exercises the separately-compiled 3-D kernels and validates them against the
    *exact* free-space 3-D Green's function (``direct_3d``): in three dimensions
    the homogeneous-medium response to a point source is a pure spherically
    spreading, time-delayed copy of the source (no near-field tail or quadrature
    approximation), so it is the cleanest analytic reference of the set. A small
    Courant number keeps the time-stepping error negligible, so the misfit is
    dominated by *spatial* numerical dispersion and falls by roughly 5x per
    accuracy order until a small residual floor (PML reflection, the O(dt^2)
    temporal term) is reached.
    """
    dtype = torch.double
    c = 1500.0
    freq = 25.0
    dx = 6.0  # points-per-wavelength = c / (freq * dx) = 10
    dt = 0.1 * dx / c  # small Courant => time-stepping error negligible
    nz = ny = 28
    nx = 90
    src = (nz // 2, ny // 2, 12)
    rec = (nz // 2, ny // 2, 70)  # 58 cells ~ 5.8 wavelengths along x
    model = torch.full((nz, ny, nx), c, dtype=dtype)
    r_cells = abs(rec[2] - src[2])
    nt = int((1.4 * r_cells * dx / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])
    expected = direct_3d(
        torch.tensor([float(i) for i in rec]),
        torch.tensor([float(i) for i in src]),
        torch.tensor([dx, dx, dx], dtype=dtype),
        dt,
        c,
        -amp[0, 0],
    )

    misfit = {}
    for accuracy in (2, 4, 6, 8):
        out = scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=accuracy,
            pml_width=15,
            max_vel=c,
            pml_freq=freq,
        )[-1][0, 0]
        n = min(len(out), len(expected))
        misfit[accuracy] = (
            (out[:n] - expected[:n]).norm() / expected[:n].norm()
        ).item()

    record_property("green_function_relative_errors", misfit)
    # Measured (CPU, float64): 2 -> ~0.99, 4 -> ~0.17, 6 -> ~0.032, 8 -> ~0.014.
    assert misfit[2] > 2 * misfit[4]  # 2nd -> 4th order: large reduction
    assert misfit[4] > 2 * misfit[6]  # 4th -> 6th order: large reduction
    assert misfit[8] < misfit[4]  # 8th clearly beats 4th (robust, away from floor)
    assert misfit[2] > 0.5  # 2nd-order dispersion is large
    assert misfit[6] < 0.08  # 6th/8th order is accurate at 10 ppw


@pytest.mark.parametrize(
    "c2",
    [
        3000.0,  # speed-up: R = +1/3
        2500.0,  # speed-up: R = +0.25
        1000.0,  # slow-down: R = -0.2 (polarity flip)
        750.0,  # slow-down: R = -1/3 (polarity flip)
    ],
)
def test_normal_incidence_reflection_coefficient(c2) -> None:
    """1-D reflection amplitude at a velocity step matches R=(c2-c1)/(c2+c1).

    The scalar wave equation u_tt = c**2 u_xx has no density, so the natural
    interface conditions (continuity of u and u_x) give a reflection
    coefficient that depends only on the wavespeeds: a wave travelling in a
    medium of speed c1 and normally incident on a step to speed c2 reflects
    with amplitude R = (c2 - c1) / (c2 + c1). This checks that Deepwave's scalar
    finite-difference scheme imposes the correct interface physics (and the
    correct polarity flip when the wave slows down, c2 < c1), against a
    closed-form ground truth — the scalar counterpart of the acoustic
    ``test_normal_incidence_reflection_coefficient`` (which carries the density
    through Z = rho*c).

    As in the acoustic test we compare the energy (L2 norm) of the reflected
    pulse to the incident pulse, which — unlike the raw peak amplitude — is
    insensitive to the (non-dissipative) numerical-dispersion broadening the
    reflected event picks up over its longer path; the sign is taken from the
    reflected-vs-incident peak polarity. A well-resolved grid (~30 points per
    wavelength, 8th-order) keeps the residual error well under 1%.
    """
    dtype = torch.double
    nx = 600
    dx = 5.0
    dt = 0.0004
    freq = 10.0  # ~30 grid points per wavelength
    ix0 = 320  # interface
    isrc = 120
    irec = 250  # between source and interface (records incident then reflected)
    c1 = 1500.0
    nt = int(1.3 / dt)

    v = torch.full((nx,), c1, dtype=dtype)
    v[ix0:] = c2
    r_analytic = (c2 - c1) / (c2 + c1)

    source_locations = torch.tensor([[[isrc]]])
    receiver_locations = torch.tensor([[[irec]]])
    source_amplitudes = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=[40, 40],
        pml_freq=freq,
        accuracy=8,
        max_vel=max(c1, c2),
    )
    trace = out.receiver_amplitudes[0, 0]

    # The incident pulse passes the receiver at t_inc; the reflected pulse
    # returns at t_ref. Split the record at the temporal midpoint.
    t_inc = (irec - isrc) * dx / c1
    t_ref = ((ix0 - isrc) + (ix0 - irec)) * dx / c1
    mid = int((t_inc + t_ref) / 2 / dt)
    incident = trace[:mid]
    reflected = trace[mid:]

    # Energy-based reflection coefficient; sign from the reflected-vs-incident
    # peak polarity (so it is independent of the source-wavelet polarity).
    inc_sign = torch.sign(incident[incident.abs().argmax()])
    refl_sign = torch.sign(reflected[reflected.abs().argmax()])
    r_measured = ((refl_sign / inc_sign) * (reflected.norm() / incident.norm())).item()

    assert abs(r_measured - r_analytic) < 0.02


def test_free_surface_reflection_coefficient() -> None:
    """A zero-``pml_width`` scalar edge is a free surface: R_field = -1.

    The reflecting edge is isolated by subtracting an otherwise identical run
    with that edge absorbing, leaving only the reflected wave. In 1-D (no
    geometric spreading) a lossless boundary reflects with unit amplitude, so the
    reflected/direct energy ratio is 1; the *sign* of the reflection distinguishes
    a free surface (the field is held to zero at the boundary, an odd/Dirichlet
    condition, polarity inverted) from a rigid one (even/Neumann, preserved).

    This pins the intrinsic, interpretation-independent boundary facts for the
    scalar propagator: the evolved field reflects with coefficient -1. Whether
    that is called a "free" (pressure-release) or a "rigid" (clamped) surface
    depends on whether the field is read as a pressure or a displacement; the
    reflection coefficient of the field is -1 either way. ``accuracy=2`` is used,
    for which the zero-exterior closure is an exact image.
    """
    dtype = torch.double
    nx, dx, dt, freq = 200, 5.0, 0.0005, 25.0
    src, rec, nt = 45, 22, 900
    v = torch.full((nx,), 1500.0, dtype=dtype)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)

    def run(pml):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=torch.tensor([[[src]]]),
            receiver_locations=torch.tensor([[[rec]]]),
            pml_width=pml,
            pml_freq=freq,
            accuracy=2,
        )[-1][0, 0]

    direct = run([30, 30])  # both edges absorbing -> direct only
    reflection = run([0, 30]) - direct  # left edge reflecting -> + reflected wave

    energy_ratio = (reflection.norm() / direct.norm()).item()
    peak_ratio = (
        reflection[reflection.abs().argmax()] / direct[direct.abs().argmax()]
    ).item()

    assert abs(energy_ratio - 1.0) < 0.02  # lossless, |R| = 1
    assert peak_ratio < -0.5  # inverted: free surface, R = -1


def test_pml_reflection_decreases_with_width(record_property) -> None:
    """Residual PML reflection must shrink as the PML width grows.

    A 1-D homogeneous experiment records the field near the left edge while the
    right boundary absorbs with a PML of varying width. ``pml_width`` adds
    absorbing cells *outside* the user's model, so we hold the model interior
    fixed (the receiver-to-model-edge distance is constant) and vary only the
    PML thickness, isolating the width effect from geometry. A reflection-free
    reference is obtained by pushing the right model edge far enough away that
    its boundary reflection cannot return within the recording window;
    subtracting it isolates the energy reflected by the PML. This converts the
    PML from a "trusted" component into a measured one (Workstream A1.4 / H).
    """
    dtype = torch.double
    c = 1500.0
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
        model = torch.full((nx,), c, dtype=dtype)
        return scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=4,
            pml_width=[left_pml, right_pml],
            max_vel=c,
            pml_freq=freq,
        )[-1][0, 0]

    # Reflection-free reference: the right model edge pushed far away.
    reference = trace(rec_i + 2000, 30)

    coeffs = []
    for width in (2, 5, 10, 20):
        t = trace(model_nx, width)
        n = min(len(t), len(reference))
        residual = t[:n] - reference[:n]
        coeffs.append((residual.norm() / reference[:n].norm()).item())

    no_pml = float((trace(model_nx, 0) - reference).norm() / reference.norm())
    record_property("pml_relative_errors", coeffs)
    record_property("no_pml_relative_error", no_pml)
    # A reflecting control checks boundary excitation independently of the
    # arbitrary expectation that a two-cell absorber must reflect badly.
    assert no_pml > 0.9
    assert all(coeffs[i] < coeffs[i - 1] for i in range(1, len(coeffs))), (
        f"PML reflection not monotonically decreasing in width: {coeffs}"
    )
    assert coeffs[0] > 10 * coeffs[-1]
    assert coeffs[-1] < 0.02  # a 20-cell PML absorbs to < 2%


def test_pml_reflection_decreases_with_width_2d() -> None:
    """A wider PML reflects less in a 2-D domain (most-used scalar kernel).

    The 1-D PML test isolates one boundary by reference-subtraction against a
    far edge; in 2-D a point source radiates to all four faces, so that trick
    does not isolate a single boundary and the cylindrical coda contaminates an
    interior-energy measurement. Instead we record a single off-axis receiver
    and subtract a *reflection-free reference* obtained by re-running with a very
    wide (near-perfect) PML: the physical direct field and coda are identical in
    both runs and cancel, leaving only the energy the finite-width PML reflected
    back to the receiver. This exercises the separately-compiled 2-D PML kernel
    against a ground truth (A1.4 / H).
    """
    dtype = torch.double
    c = 1500.0
    freq = 25.0
    dx = 5.0
    dt = 0.5 * dx / c
    n = 60  # square model interior, identical for every run
    src = (n // 2, n // 2)
    rec = (n // 2 + 15, n // 2 + 15)  # off-axis: sees reflections from all faces
    model = torch.full((n, n), c, dtype=dtype)
    diag = (2 * (n * dx) ** 2) ** 0.5
    nt = int((1.5 * diag / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])

    def trace(width: int) -> torch.Tensor:
        return scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=4,
            pml_width=width,
            max_vel=c,
            pml_freq=freq,
        ).receiver_amplitudes[0, 0]

    # A 30-cell PML is effectively reflection-free at this scale (the reference).
    reference = trace(30)
    widths = (2, 5, 10)
    coeffs = [((trace(w) - reference).norm() / reference.norm()).item() for w in widths]

    # Measured (CPU, float64): ~0.136, 0.0088, 0.0018 for width 2/5/10.
    assert all(coeffs[i] < coeffs[i - 1] for i in range(1, len(coeffs))), (
        f"PML reflection not monotonically decreasing in width: {coeffs}"
    )
    assert coeffs[0] > 0.05  # a 2-cell PML reflects appreciably
    assert coeffs[0] > 5 * coeffs[1]  # widening 2 -> 5 cells sharply cuts it
    assert coeffs[-1] < 0.01  # a 10-cell PML reflects < 1% to the receiver


def test_pml_residual_energy_decreases_with_width_3d() -> None:
    """A wider PML traps less residual energy in a 3-D domain.

    The 3-D counterpart of the PML-width test, exercising the separately
    compiled 3-D PML kernels (A1.4 / H). A point source radiates to all six
    faces, so the 1-D reference-subtraction trick does not isolate a single
    boundary; instead we measure the total wave energy left inside the model
    interior long after the pulse has had time to cross and exit it. The model
    interior is the same fixed size for every run and the number of time steps
    is fixed, so the direct field has departed identically in all cases and the
    only thing that differs is how much energy the PML reflected back or failed
    to absorb. A better (wider) PML leaves less residual interior energy.

    ``pml_width`` pads absorbing cells *outside* the model, so the returned
    wavefield has shape ``model + 2*width`` per dimension; we crop back to the
    central model box (the physical interior) before measuring its energy.
    """
    dtype = torch.double
    c = 1500.0
    freq = 25.0
    dx = 5.0
    dt = 0.5 * dx / c
    n = 40  # cubic model interior, identical for every run
    src = (n // 2,) * 3
    model = torch.full((n, n, n), c, dtype=dtype)
    diag = (3 * (n * dx) ** 2) ** 0.5
    # Run long enough for the pulse to cross and leave a well-absorbed interior.
    nt = int((2.0 * diag / c + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])

    def interior_residual(width: int) -> float:
        wf = scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=source_locations,
            accuracy=4,
            pml_width=width,
            max_vel=c,
            pml_freq=freq,
        )[0]  # final wavefield, padded by `width` cells per face
        interior = wf[:, width : width + n, width : width + n, width : width + n]
        return interior.norm().item()

    energies = [interior_residual(width) for width in (2, 5, 10, 20)]
    # Normalise to the width-2 case -> dimensionless retained-energy ratios.
    coeffs = [e / energies[0] for e in energies]

    # Measured (CPU, float64): 1.0, 0.147, 0.046, 0.016 for width 2/5/10/20.
    assert all(energies[i] < energies[i - 1] for i in range(1, len(energies))), (
        f"PML residual energy not monotonically decreasing in width: {coeffs}"
    )
    assert coeffs[1] < 0.3  # widening 2 -> 5 cells sharply cuts trapped energy
    assert coeffs[-1] < 0.05  # a 20-cell PML retains < 5% of the width-2 residual


def test_time_reversal_symmetry() -> None:
    """The lossless leapfrog scheme is exactly time-reversible.

    Propagating a source-free initial wavefield forward, then swapping the two
    final wavefields and propagating again, must reconstruct the initial state
    to machine precision in a lossless domain (``pml_width=0``). The leapfrog
    update ``u_{n+1} = 2u_n - u_{n-1} + c^2 dt^2 L u_n`` is reversible by
    construction, so this is *not* a check on the spatial operator ``L`` or the
    velocity scaling (those are covered by the dispersion / Green's-function
    tests); rather it is a guard against any accidental numerical dissipation or
    amplification, and against time-asymmetry, in the interior stepping and the
    reflecting (``pml_width=0``) boundary. ``nt`` is large enough that the pulse
    reaches and reflects off the edges, so the boundary is exercised.
    """
    dtype = torch.double
    c = 1500.0
    dx = 5.0
    dt = 0.6 * dx / c
    ny, nx = 60, 60
    model = torch.full((ny, nx), c, dtype=dtype)
    yy, xx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    bump = torch.exp(-(((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) / 8.0)).to(dtype)
    nt = 120  # front travels ~72 cells > half-domain: edges reached & reflected

    def prop(w0: torch.Tensor, wm1: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return scalar(
            model,
            dx,
            dt,
            wavefield_0=w0.clone(),
            wavefield_m1=wm1.clone(),
            source_amplitudes=None,
            source_locations=None,
            accuracy=4,
            pml_width=0,
            pml_freq=25.0,  # unused at pml_width=0; set only to silence warning
            max_vel=c,
            nt=nt,
        )

    # Zero initial velocity => the two initial wavefields are equal.
    forward = prop(bump, bump)
    final, final_m1 = forward[0], forward[1]
    # Reverse: feed (u_{N-1}, u_N) so the leapfrog steps backwards in time.
    reverse = prop(final_m1, final)
    recon_err = (reverse[0] - bump).norm() / bump.norm()
    assert recon_err < 1e-10, f"time reversal recon error {recon_err.item():.3e}"


def _small_scalar_call() -> tuple[torch.Tensor, ...]:
    """A tiny scalar() call, with max_vel supplied.

    Used by the graph-break and torch.export tests below.
    """
    nz, ny = 8, 8
    v = torch.ones(nz, ny) * 1500.0
    dx = 5.0
    dt = 0.001
    nt = 8
    source_amplitudes = torch.zeros(1, 1, nt)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    return scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=2000.0,
    )


def test_velocity_extremes_reduces_dynamo_graph_breaks() -> None:
    """Check that the known, fixable torch.compile graph breaks are gone.

    History (this repo, torch 2.12.1+cpu, this exact ``_small_scalar_call``):

    - 20 breaks originally.
    - 17 after the first fix: ``deepwave._common.velocity_extremes`` used to
      always read ``v[v != 0].abs().min().item()`` and
      ``v.abs().max().item()``, even when the caller supplied ``max_vel``
      (making the max read redundant) and regardless of whether the model
      was being compiled at all; those two data-dependent reads (an
      ``aten.nonzero`` and two ``Tensor.item()`` calls) were opaque to
      ``torch.compile`` and forced three graph breaks.
    - 11 after this follow-up, which addressed every other *fixable*
      data-dependent read in ``common.py``:
      ``mask_locations`` (an ``aten.nonzero`` from boolean-mask assignment,
      hit twice -- once per source/receiver call) is now a shape-static
      ``torch.where`` select, with the same output for every input including
      empty tensors; ``StorageManager.__init__``'s ``int(torch.prod(...
      ).item())`` (also hit twice, once per ``StorageManager`` instance) was
      a spurious Tensor round-trip over what is really a product of Python
      ints (``shot_shape``) and is now plain ``math.prod``;
      ``_validate_location_tensor``'s integer-valued-locations check is
      skipped while ``torch.compiler.is_compiling()``. The duplicate-cell
      check is required by native scatter semantics and now runs outside the
      compiled graph on every invocation.

    What's left (3 reasons, matching ``out.break_reasons`` today) is either
    unrelated to this data-dependent-guards effort or deliberately not
    fixable:

    - A functorch/vmap compatibility skip in
      ``is_vmap_batched``/``check_inputs_not_vmapped`` (an opaque pybind
      builtin), unrelated to any data-dependent *value* read.
    - ``check_locations_are_within_model``'s bounds check (``if
      bool(valid.all())``). Unlike the others, this is retained on purpose:
      it is the only guard against an out-of-bounds location reaching the
      C/CUDA backend's unconditional gather/scatter, so skipping it under
      compilation would trade a clear Python exception for silent
      out-of-bounds memory access. There is no static (shape-only) rewrite
      that preserves this property, since the whole point is to branch on
      run-time tensor values. See the function's docstring for the full
      rationale.
    - ``step_unsupported`` in ``allocate``, the C-backend call itself, which
      is legitimately opaque to Dynamo (the autograd.Function boundary).

    This pins down which
    specific functions must no longer appear in any break's stack. The
    uniqueness check's deliberate graph break is allowed.
    """
    out = torch._dynamo.explain(_small_scalar_call)()
    # Dynamo partitions these guards differently across Torch versions.
    # Assert the removed causes directly instead of pinning a version-specific
    # total that also counts required material and native-safety checks.
    fixed_functions = {
        "velocity_extremes",
        "mask_locations",
        "mask_source_amplitudes",
        "_validate_location_tensor",
    }
    for reason in out.break_reasons:
        for frame in reason.user_stack:
            assert frame.name not in fixed_functions, (
                f"a graph break traced back into {frame.name}, which this "
                "torch.compile/export data-dependent-guards pass should "
                f"have removed entirely: {reason.reason}"
            )
            # StorageManager.__init__'s int(torch.prod(...).item()) was the
            # only "__init__" on this call path; it is now math.prod, so no
            # break should trace into any __init__ in common.py.
            assert not (
                frame.name == "__init__" and frame.filename.endswith("common.py")
            ), (
                "a graph break traced back into a common.py __init__ "
                "(expected StorageManager's .item() round-trip to be gone): "
                f"{reason.reason}"
            )


def _small_scalar_call_indexed() -> torch.Tensor:
    """Same as ``_small_scalar_call``, but indexes the last positional output.

    Exercises ``PropagatorOutput.__getitem__`` (inherited from ``tuple``)
    under ``torch.compile``, now that every propagator always returns a
    ``PropagatorOutput`` rather than a plain positional tuple.
    """
    return _small_scalar_call()[-1]


def _small_scalar_call_named() -> torch.Tensor:
    """Same as ``_small_scalar_call``, but reads ``receiver_amplitudes`` by name.

    Exercises ``PropagatorOutput.__getattr__`` under ``torch.compile``.
    """
    return _small_scalar_call().receiver_amplitudes


def test_compile_indexed_output_matches_eager() -> None:
    """A ``torch.compile``'d function indexing ``out[-1]`` matches eager.

    ``PropagatorOutput`` is a ``tuple`` subclass, so positional indexing of
    the compiled call's return value must give the same tensor as the
    uncompiled (eager) call.
    """
    eager = _small_scalar_call_indexed()
    compiled = torch.compile(_small_scalar_call_indexed)()
    assert torch.equal(eager, compiled)


def test_compile_named_output_matches_eager() -> None:
    """A ``torch.compile``'d function reading ``out.receiver_amplitudes`` matches eager.

    ``PropagatorOutput.__getattr__`` name lookup must keep working, and give
    the same tensor as ``out[-1]``, when the whole call is compiled.
    """
    eager = _small_scalar_call_named()
    compiled = torch.compile(_small_scalar_call_named)()
    assert torch.equal(eager, compiled)
    assert torch.equal(eager, _small_scalar_call_indexed())


class _VelocityExtremesExportWrapper(torch.nn.Module):
    """Exercise the supplied-bound helper inside an actually exported graph."""

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        minimum, maximum = deepwave._common.velocity_extremes(v, 2000.0, "scalar")
        return v * maximum + minimum


@pytest.mark.parametrize("value", [-1500.0, 0.0, 1500.0])
def test_torch_export_no_longer_fails_on_velocity_extremes(value: float) -> None:
    """The helper exports without reading data-dependent model extrema.

    With an explicit bound, tracing uses that bound and reports an unknown
    minimum as zero. Run the exported graph on changed values too, so a
    value accidentally captured during export cannot make this test pass.
    Full propagator export remains unsupported; its first failure depends
    on the PyTorch version and is not a regression oracle for this helper.
    Location bounds and duplicate-write safety have their own direct tests.
    """
    v = torch.full((8, 8), value)
    exported = torch.export.export(_VelocityExtremesExportWrapper(), (v,)).module()
    for values in (v, -v, torch.zeros_like(v), v + 1):
        torch.testing.assert_close(exported(values), values * 2000.0, rtol=0, atol=0)


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    nz, ny = 8, 8
    v = torch.ones(nz, ny) * 1500.0
    source_amplitudes = torch.zeros(1, 1, 8)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        scalar(
            v,
            5.0,
            0.001,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            max_vel=2000.0,
            backend="eager",
            storage_compression=True,
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_output_names(ndim) -> None:
    """The returned PropagatorOutput's field names match scalar_output_names.

    ``out._names`` (set by ``_engine.named_output`` from ``SPEC.output_names``)
    must equal ``scalar_output_names(ndim)`` exactly, in order: this is the
    contract that lets ``out.receiver_amplitudes`` and the other named fields
    resolve to the correct positional tensor for every ``ndim``.
    """
    from deepwave.scalar import scalar_output_names

    n = 8
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0)
    source_amplitudes = torch.zeros(1, 1, 8)
    source_amplitudes[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    out = scalar(
        v,
        5.0,
        0.004,
        source_amplitudes=source_amplitudes,
        source_locations=loc,
        receiver_locations=loc,
        pml_width=3,
    )
    assert out._names == tuple(scalar_output_names(ndim))
    for name in out._names:
        assert hasattr(out, name)

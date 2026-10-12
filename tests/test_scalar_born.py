"""Tests for deepwave.scalar_born."""

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
    compute_backend_parity_grad,
    compute_backend_parity_nt,
    element_tensors,
    output_tensors,
    run_checkpoint_equivalence,
    run_python_vs_compiled_backend_parity,
    run_reciprocity_check,
    scattered,
)
from torch.func import jvp, vmap

import deepwave
from deepwave import IGNORE_LOCATION, ScalarBorn, scalar, scalar_born
from deepwave._backend_utils import USE_OPENMP
from deepwave._common import set_boundary
from deepwave.discretisation import cfl_condition, downsample, upsample
from deepwave.scalar_born import scalar_born_output_names
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


def test_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_born_scatter(
            propagator=scalarbornprop, prop_kwargs={"backend": "torch_compile"}
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
        for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
            expected, actual = run_born_scatter(
                propagator=scalarbornprop,
                dt=0.001 if accuracy == 4 else 0.0001,
                nx=nx,
                dx=dx,
                prop_kwargs={
                    "pml_width": 30,
                    "backend": python,
                    "accuracy": accuracy,
                },
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        assert torch.allclose(
            actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT], actuals["compiled"], atol=1e-3
        )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_born_equals_directional_fd_of_scalar(ndim: int) -> None:
    """``dv`` is a wavespeed perturbation, not a separately-scaled potential.

    ``scalar_born``'s docstring called ``dv`` "the scattering potential" while
    ``acoustic_born``'s calls its ``dv`` a "wavespeed scattering
    perturbation". The two are in fact the same quantity -- the kernel's
    coupling term is ``2 * v * dt**2 * dv * w``, i.e. the first-order term of
    ``delta(v**2) = 2 v delta v`` -- but nothing in the suite proved it, so
    the ``d<model>`` naming convention shared by the three Born propagators
    rested on a reading of the C source. ``acoustic_born`` has had
    ``test_born_equals_directional_fd_of_acoustic`` for exactly this; this is
    its ``scalar_born`` counterpart.

    The scattered receiver amplitudes must equal the central finite
    difference of ``scalar()`` with respect to ``v`` in the direction ``dv``.
    """
    shape = {1: (14,), 2: (12, 12), 3: (8, 8, 8)}[ndim]
    v = torch.full(shape, 1500.0, dtype=torch.double)
    dv = torch.zeros(shape, dtype=torch.double)
    dv[tuple(s // 2 for s in shape)] = 10.0
    dx, dt, nt = 6.0, 0.0004, 20
    source_locations = torch.zeros(1, 1, ndim, dtype=torch.long)
    for d in range(ndim):
        source_locations[..., d] = shape[d] // 2
    receiver_locations = source_locations.clone()
    receiver_locations[..., -1] = shape[-1] // 2 - 2
    torch.manual_seed(0)
    kwargs: dict = {
        "source_amplitudes": torch.randn(1, 1, nt, dtype=torch.double),
        "source_locations": source_locations,
        "receiver_locations": receiver_locations,
        "pml_width": 4,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
        "accuracy": 4,
    }

    def scalar_run(v_: torch.Tensor) -> torch.Tensor:
        return scalar(v_, dx, dt, **kwargs)[-1]

    names = scalar_born_output_names(ndim)
    # scalar_born names its two receiver sets explicitly; the flat
    # `receiver_locations` in `kwargs` is for `scalar` alone.
    born_kwargs = {
        name: value for name, value in kwargs.items() if name != "receiver_locations"
    }
    out = scalar_born(
        v,
        dx,
        dt,
        dv=dv,
        bg_receiver_locations=receiver_locations,
        sc_receiver_locations=receiver_locations,
        **born_kwargs,
    )
    scattered_receivers = dict(zip(names, out))["sc_receiver_amplitudes"]

    # The background output must also match plain scalar(), or a scattered
    # field compared against the wrong background would prove nothing.
    background = dict(zip(names, out))["bg_receiver_amplitudes"]
    assert torch.allclose(background, scalar_run(v), atol=1e-8, rtol=1e-8)

    eps = 1e-3
    fd = (scalar_run(v + eps * dv) - scalar_run(v - eps * dv)) / (2 * eps)
    assert fd.abs().max() > 0, "the finite difference is zero; test is vacuous"
    relative = (scattered_receivers - fd).abs().max() / fd.abs().max()
    assert relative < 1e-5


def test_born_scatter_v_batched_2d() -> None:
    """Test Born propagation in a batched 2D velocity model."""
    expected_diffs = {2: 0.14, 4: 0.0025, 6: 0.003, 8: 0.00062}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
            expected, actual = run_born_scatter(
                c=torch.tensor([[[1500.0]], [[1600.0]]]),
                propagator=scalarbornprop,
                dt=0.001 if accuracy == 4 else 0.0001,
                prop_kwargs={
                    "pml_width": 30,
                    "backend": python,
                    "accuracy": accuracy,
                },
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        assert torch.allclose(
            actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT], actuals["compiled"], atol=1e-4
        )


def test_born_scatter_scatter_batched_2d() -> None:
    """Test Born propagation in a batched 2D scatter model."""
    expected_diffs = {2: 0.14, 4: 0.0025, 6: 0.003, 8: 0.00062}
    for accuracy in [2, 4, 6, 8]:
        actuals = {}
        for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
            expected, actual = run_born_scatter(
                dscatter=torch.tensor([[[50.0]], [[100.0]]]),
                propagator=scalarbornprop,
                dt=0.001 if accuracy == 4 else 0.0001,
                prop_kwargs={
                    "pml_width": 30,
                    "backend": python,
                    "accuracy": accuracy,
                },
            )
            diff = (expected - actual.cpu()).flatten()
            assert diff.norm().item() < expected_diffs[accuracy]
            actuals[python] = actual
        assert torch.allclose(
            actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT], actuals["compiled"], atol=1e-4
        )


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
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
    bg_receiver_locations: Optional[torch.Tensor] = None,
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
) -> tuple[torch.Tensor, ...]:
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

    # psi*/zeta* are the absorbing boundary's memory variables, not physical
    # fields, so they are no longer direct keyword arguments: they travel
    # inside the single ``auxiliary_state`` that ``initial_state`` carries.
    # Any that are None are simply absent, and that part of the state starts
    # from rest.
    psi_zeta_state = auxiliary_state_arg(
        psiz_m1=psiz_m1,
        psiy_m1=psiy_m1,
        psix_m1=psix_m1,
        zetaz_m1=zetaz_m1,
        zetay_m1=zetay_m1,
        zetax_m1=zetax_m1,
        psiz_sc_m1=psiz_sc_m1,
        psiy_sc_m1=psiy_sc_m1,
        psix_sc_m1=psix_sc_m1,
        zetaz_sc_m1=zetaz_sc_m1,
        zetay_sc_m1=zetay_sc_m1,
        zetax_sc_m1=zetax_sc_m1,
    )

    if functional:
        return scalar_born(
            model,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            bg_receiver_locations=bg_receiver_locations,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            wavefield_sc_0=wavefield_sc_0,
            wavefield_sc_m1=wavefield_sc_m1,
            initial_state=psi_zeta_state,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            **prop_kwargs,
        )
    prop = ScalarBorn(model, dx, dv=scatter)
    return prop(
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        bg_receiver_locations=bg_receiver_locations,
        wavefield_0=wavefield_0,
        wavefield_m1=wavefield_m1,
        wavefield_sc_0=wavefield_sc_0,
        wavefield_sc_m1=wavefield_sc_m1,
        initial_state=psi_zeta_state,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def scalarbornpropchained(
    model: torch.Tensor,
    scatter: torch.Tensor,
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor],
    source_locations: Optional[torch.Tensor],
    receiver_locations: Optional[torch.Tensor],
    bg_receiver_locations: Optional[torch.Tensor] = None,
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
) -> tuple[torch.Tensor, ...]:
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
    wfcsc = wavefield_sc_0
    wfpsc = wavefield_sc_m1
    # The absorbing boundary's memory variables travel as one opaque
    # ``auxiliary_state``, so each segment hands the next its whole state
    # object rather than eight individually named tensors.
    auxiliary_state = auxiliary_state_arg(
        psiy_m1=psiy_m1,
        psix_m1=psix_m1,
        zetay_m1=zetay_m1,
        zetax_m1=zetax_m1,
        psiy_sc_m1=psiy_sc_m1,
        psix_sc_m1=psix_sc_m1,
        zetay_sc_m1=zetay_sc_m1,
        zetax_sc_m1=zetax_sc_m1,
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
        out = scalar_born(
            model,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=segment_source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            bg_receiver_locations=bg_receiver_locations,
            wavefield_0=wfc,
            wavefield_m1=wfp,
            wavefield_sc_0=wfcsc,
            wavefield_sc_m1=wfpsc,
            initial_state=auxiliary_state,
            nt=segment_nt,
            model_gradient_sampling_interval=step_ratio,
            **prop_kwargs,
        )
        wfc = out.wavefield_0
        wfp = out.wavefield_m1
        wfcsc = out.wavefield_sc_0
        wfpsc = out.wavefield_sc_m1
        auxiliary_state = {"auxiliary_state": out.auxiliary_state}
        segment_bg_receiver_amplitudes = out.bg_receiver_amplitudes
        segment_receiver_amplitudes = out.sc_receiver_amplitudes
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

    # A flat tuple of tensors in the pre-fold order, which is what the
    # backend-parity comparison this helper feeds expects; the boundary
    # state is unpacked here rather than threaded on.
    state = auxiliary_state["auxiliary_state"]
    return (
        wfc,
        wfp,
        state["psiy_m1"],
        state["psix_m1"],
        state["zetay_m1"],
        state["zetax_m1"],
        wfcsc,
        wfpsc,
        state["psiy_sc_m1"],
        state["psix_sc_m1"],
        state["zetay_sc_m1"],
        state["zetax_sc_m1"],
        bg_receiver_amplitudes,
        receiver_amplitudes,
    )


def run_born_scatter(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 150,
    dscatter: Union[float, torch.Tensor] = 150,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.0001,
    nx: tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    dc: float = 100,
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
    dx: tuple[float, float] = (5, 5),
    dt: float = 0.004,
    nx: tuple[int, int] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        actual_cpu = run_born_forward_2d(
            propagator=scalarbornprop,
            device=torch.device("cpu"),
            prop_kwargs={"backend": python},
        )
        actual_gpu = run_born_forward_2d(
            propagator=scalarbornprop,
            device=torch.device("cuda"),
            prop_kwargs={"backend": python},
        )
        for cpui, gpui in zip(output_tensors(actual_cpu), output_tensors(actual_gpu)):
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

        # The two constructions below must be compared on an equally
        # "warmed-up" backend. A ``torch.compile``d step function is compiled
        # on its first call and can be recompiled on a later one (automatic
        # dynamic-shape promotion), and the generated kernel reassociates the
        # same arithmetic and so rounds differently: the first construction
        # would otherwise be computed partly by one artefact and the second
        # entirely by another. In this test's default float32
        # that difference is ~1e-6 absolute on traces of amplitude ~1 (it
        # falls to ~1e-15 in float64, i.e. it tracks eps and is pure
        # rounding, not a semantic difference in IGNORE_LOCATION handling),
        # but it lands on final wavefields that have decayed to ~1e-3, where
        # allclose's relative test sees it as ~1e-3. Repeating the first
        # construction until the executor has settled removes the artefact
        # instead of tolerating it. Same warmup as
        # test_scalar/test_acoustic's copies of this test.
        n_warmup = 3 if python != "compiled" else 1

        for _warmup in range(n_warmup):
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

        # Every element is a Tensor except auxiliary_state, which holds
        # several; compare the tensors either way.
        for ofi, oii in zip(out_filled, out_ignored):
            for filled, ignored in zip(element_tensors(ofi), element_tensors(oii)):
                assert torch.allclose(filled, ignored)

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
                [ay, by, dbydy, ax, bx, dbxdx],
                [dy, dx],
                dt,
                nt,
                step_ratio,
                accuracy,
                pml_width,
                # ``boundary`` config arg (all-PML here): scalar_born_func
                # expects it right after ``pml_width`` (see the ``scalar_born``
                # entry point).
                set_boundary(None, pml_width, 2),
                n_batch,
                None,
                None,
                1,
                "device",
                ".",
                False,
                c,
                scatter,
                sa,
                source_amplitudessc,
                sources_i,
                sources_i,
                receivers_i,
                receiverssc_i,
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
        out_python = wrap("eager")
        for oc, op in zip(out_compiled, out_python):
            assert torch.allclose(oc, op)


def test_scalarbornfunc() -> None:
    """Test scalar_born_func with different time steps."""
    run_scalarbornfunc(nt=4)
    run_scalarbornfunc(nt=5)


def test_born_backend_parity() -> None:
    """Test backend parity in a 2D model with Born propagator."""
    run_born_backend_parity(propagator=scalarbornprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((4,), (5,)),
        ((4, 3), (5, 5)),
        ((4, 3, 3), (6, 6, 6)),
    ],
)
def test_born_backend_parity_2nd_order(nx, dx) -> None:
    """Test backend parity with a 2nd order accurate propagator."""
    run_born_backend_parity(
        propagator=scalarbornprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2}
    )


def test_born_backend_parity_6th_order() -> None:
    """Test backend parity with a 6th order accurate propagator."""
    run_born_backend_parity(propagator=scalarbornprop, prop_kwargs={"accuracy": 6})


def test_born_backend_parity_8th_order() -> None:
    """Test backend parity with a 8th order accurate propagator."""
    run_born_backend_parity(
        propagator=scalarbornprop, nx=(4, 4), prop_kwargs={"accuracy": 8}
    )


def test_born_backend_parity_cfl() -> None:
    """Test backend parity with a timestep greater than the CFL limit."""
    run_born_backend_parity(
        propagator=scalarbornprop,
        dt=0.002,
        atol=2e-7,
        rtol=1e-8,
        nt_add=100,
    )


def test_born_backend_parity_odd_timesteps() -> None:
    """Test backend parity with one more timestep."""
    run_born_backend_parity(propagator=scalarbornprop, nt_add=1)


def test_born_backend_parity_one_shot() -> None:
    """Test backend parity with one shot."""
    run_born_backend_parity(propagator=scalarbornprop, num_shots=1)


def test_born_backend_parity_no_sources() -> None:
    """Test backend parity with no sources."""
    run_born_backend_parity(propagator=scalarbornprop, num_sources_per_shot=0)


def test_born_backend_parity_no_receivers() -> None:
    """Test backend parity with no receivers."""
    run_born_backend_parity(propagator=scalarbornprop, num_receivers_per_shot=0)


def test_backend_parity_no_scatter_receivers() -> None:
    """Test backend parity with no scatter receivers."""
    run_born_backend_parity(propagator=scalarbornprop, num_scatter_receivers_per_shot=0)


def test_born_backend_parity_survey_pad() -> None:
    """Test backend parity with survey_pad."""
    run_born_backend_parity(
        propagator=scalarbornprop,
        survey_pad=0,
        provide_wavefields=False,
    )


def test_born_backend_parity_partial_wavefields() -> None:
    """Test backend parity with wavefields that do not cover the model."""
    run_born_backend_parity(
        propagator=scalarbornprop,
        origin=(0, 4),
        nx=(12, 9),
        wavefield_size=(4 + 6, 1 + 6),
    )


def test_born_backend_parity_chained() -> None:
    """Test backend parity when two propagators are chained."""
    run_born_backend_parity(propagator=scalarbornpropchained)


def test_born_backend_parity_negative() -> None:
    """Reject negative background speed before stepping."""
    with pytest.raises(ValueError, match="v > 0"):
        run_born_backend_parity(c=-1500, propagator=scalarbornprop)


def test_born_backend_parity_zero() -> None:
    """Reject zero background speed before stepping."""
    with pytest.raises(ValueError, match="v > 0"):
        run_born_backend_parity(c=0, dc=0, propagator=scalarbornprop)


def test_born_backend_parity_different_pml() -> None:
    """Test backend parity with different pml widths."""
    run_born_backend_parity(
        propagator=scalarbornprop, pml_width=[0, 1, 5, 10], atol=5e-8
    )


def test_born_backend_parity_no_pml() -> None:
    """Test backend parity with no pml."""
    run_born_backend_parity(propagator=scalarbornprop, pml_width=0, atol=2e-8)


def test_born_backend_parity_different_dx() -> None:
    """Test backend parity with different dx values."""
    run_born_backend_parity(propagator=scalarbornprop, dx=(4, 5), atol=5e-8)


def test_born_backend_parity_single_cell() -> None:
    """Test backend parity with a single model cell."""
    run_born_backend_parity(
        propagator=scalarbornprop,
        nx=(1, 1),
        prop_kwargs={"accuracy": 2},
        num_shots=1,
        num_sources_per_shot=1,
        num_receivers_per_shot=1,
        num_scatter_receivers_per_shot=1,
    )


def test_born_backend_parity_big() -> None:
    """Test backend parity with a big model."""
    run_born_backend_parity(
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

        # Negative physical background speed is outside the material domain.
        with pytest.raises(ValueError, match="v > 0"):
            propagator(
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
        assert torch.allclose(out_positive.wavefield_0, out.wavefield_0)
        assert torch.allclose(out_positive.wavefield_sc_0, out.wavefield_sc_0)
        assert torch.allclose(
            out_positive.sc_receiver_amplitudes, out.sc_receiver_amplitudes
        )

        # Scalar zero speed is also rejected; signed Born directions remain valid.
        with pytest.raises(ValueError, match="v > 0"):
            propagator(
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


def test_born_backend_parity_only_v_2d() -> None:
    """Test backend parity with only v requiring gradient."""
    run_born_backend_parity(
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


def test_born_backend_parity_only_scatter_2d() -> None:
    """Test backend parity with only scatter requiring gradient."""
    run_born_backend_parity(
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


def test_born_backend_parity_only_source_2d() -> None:
    """Test backend parity with only source requiring gradient."""
    run_born_backend_parity(
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


def test_born_backend_parity_only_wavefield_0_2d() -> None:
    """Test backend parity with only wavefield_0 requiring gradient."""
    run_born_backend_parity(
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


def test_born_backend_parity_only_wavefieldsc_0_2d() -> None:
    """Test backend parity with only wavefieldsc_0 requiring gradient."""
    run_born_backend_parity(
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


def test_born_backend_parity_v_batched() -> None:
    """Test backend parity using a different velocity for each shot."""
    run_born_backend_parity(
        propagator=scalarbornprop,
        c=torch.tensor([[[1500.0]], [[1600.0]]]),
    )


def test_born_backend_parity_scatter_batched() -> None:
    """Test backend parity using a different scatter for each shot."""
    run_born_backend_parity(
        propagator=scalarbornprop,
        dscatter=torch.tensor([[[150.0]], [[100.0]]]),
    )


@pytest.mark.parametrize(("v_requires_grad"), [True, False])
@pytest.mark.parametrize(("scatter_requires_grad"), [True, False])
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
    v_requires_grad,
    scatter_requires_grad,
    num_shots,
    nx,
    dx,
    device,
    c=1500,
    dc=100,
    freq=25,
    dt=0.005,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    num_scatter_receivers_per_shot=2,
    propagator=scalarbornprop,
    prop_kwargs=None,
    dtype=None,
):
    """Test gradients with different storage options."""
    if not v_requires_grad and not scatter_requires_grad:
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

    # Material draws must not change source timing or depend on the device's
    # RNG algorithm: both backends should run the same physical experiment.
    generator = torch.Generator().manual_seed(1)
    model = (c + torch.randn(*nx, dtype=dtype, generator=generator) * dc).to(device)
    scatter = torch.randn(*nx, dtype=dtype, generator=generator).to(device) * dc

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
    modeld.requires_grad_(v_requires_grad)
    scatterd = scatter.clone()
    scatterd.requires_grad_(scatter_requires_grad)
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

    atol_model = [4, 0.02, 2e-3]
    atol_scatter = [2e-3, 1e-4, 4e-5]
    # The outermost cells of `dv` are extended into the padding, so their
    # gradient is the cell's own term plus one from every padded cell beyond
    # it. Inside the absorbing layer those terms are individually much larger
    # than the value they sum to, so the quantisation error of the compressed
    # history is amplified there while the signal cancels; measured at 0.070
    # in 1-D, 1.4e-4 in 2-D and 1e-5 in 3-D, against interiors of 8.4e-4,
    # 2e-5 and below 1e-5. The interior keeps the tight bound, so this
    # allowance cannot hide a change there.
    atol_scatter_edge = [0.2, 1e-3, 1e-4]

    def assert_scatter_grad_close(expected, actual, atol, atol_edge):
        """`torch.allclose`, with a looser bound on the outermost cells."""
        edge = torch.ones_like(expected, dtype=torch.bool)
        edge[tuple(slice(1, -1) if size > 2 else slice(None) for size in nx)] = False
        difference = (actual - expected).abs()
        assert difference[edge].max() <= atol_edge
        if (~edge).any():
            assert difference[~edge].max() <= atol

    for mode in modes:
        for compression in [False, True]:
            if mode == "device" and not compression:  # Default
                continue
            modeli = model.clone()
            modeli.requires_grad_(v_requires_grad)
            scatteri = scatter.clone()
            scatteri.requires_grad_(scatter_requires_grad)
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

            if compression:
                if v_requires_grad:
                    assert torch.allclose(
                        modeld.grad, modeli.grad, atol=atol_model[len(nx) - 1]
                    )
                if scatter_requires_grad:
                    assert_scatter_grad_close(
                        scatterd.grad,
                        scatteri.grad,
                        atol_scatter[len(nx) - 1],
                        atol_scatter_edge[len(nx) - 1],
                    )
            else:
                if v_requires_grad:
                    assert torch.allclose(
                        modeld.grad,
                        modeli.grad,
                        atol=modeld.grad.detach().abs().max().item() * 1e-5,
                    )
                if scatter_requires_grad:
                    assert torch.allclose(
                        scatterd.grad,
                        scatteri.grad,
                        atol=scatterd.grad.detach().abs().max().item() * 1e-5,
                    )


def run_born_backend_parity(
    c: Union[float, torch.Tensor] = 1500,
    dc: float = 100,
    dscatter: Union[float, torch.Tensor] = 100,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.001,
    nx: tuple[int, ...] = (4, 3),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    num_scatter_receivers_per_shot: int = 2,
    propagator: Any = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Union[int, list[int]] = 3,
    survey_pad: Optional[Union[int, list[int]]] = None,
    origin: Optional[list[int]] = None,
    wavefield_size: Optional[tuple[int, ...]] = None,
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
    """Compare eager and compiled backend gradients (no finite differencing)."""
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
    if num_scatter_receivers_per_shot > 0:
        x_r_sc = _set_coords(num_shots, num_scatter_receivers_per_shot, nx.tolist())
    else:
        x_r_sc = None
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
    wavefield_sc_0, wavefield_sc_m1, psi_sc_m1, zeta_sc_m1 = (
        _zero_scalar_style_wavefields(
            provide_wavefields,
            wavefield_size,
            num_shots,
            ndim,
            dtype,
            device,
            wavefieldsc_0_requires_grad,
            wavefieldsc_m1_requires_grad,
            psisc_m1_requires_grad,
            zetasc_m1_requires_grad,
        )
    )

    model.requires_grad_(v_requires_grad)
    scatter.requires_grad_(scatter_requires_grad)
    if prop_kwargs is None:
        prop_kwargs = {}

    out_idxs = []

    def wrap(python):
        prop_kwargs["backend"] = python
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
        # The auxiliary state is one output element holding several tensors;
        # the parity comparison is over tensors.
        return compute_backend_parity_grad(output_tensors(out), inputs, out_idxs)

    run_python_vs_compiled_backend_parity(wrap, "eager", atol=atol, rtol=rtol)


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

    run_reciprocity_check(
        [model, scatter],
        dx,
        dt,
        source_amplitudes,
        "",
        "",
        -2,
        -2,
        loc_a,
        loc_b,
        prop_kwargs,
        scalarbornprop,
        receiver_kwarg="bg_receiver_locations",
        atol=1e-3,
    )


def test_vmap_born_props() -> None:
    """Test vmap over velocity and scatter models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v_base = 1500.0

    batch_size = 3
    v_batch = torch.empty(batch_size, ny, nx, device=device)
    scatter_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        v_batch[i] = v_base + i * 100.0
        scatter_batch[i] = 0.1 * i

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

    def prop(v, scatter):
        return scalar_born(
            v,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]  # Return scattered receiver amplitudes

    # Run vmap
    res_vmap = vmap(prop)(v_batch, scatter_batch)

    # Run loop
    res_loop = torch.stack(
        [prop(v_batch[i], scatter_batch[i]) for i in range(batch_size)]
    )

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_vmap_born_partial() -> None:
    """Test vmap over scatter model only (v shared)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v = torch.full((ny, nx), 1500.0, device=device)

    batch_size = 3
    scatter_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        scatter_batch[i] = 0.1 * (i + 1)

    n_shots = 1
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2
    receiver_locations = source_locations.clone()  # Just one receiver

    source_amplitudes = ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1)

    def prop(s):
        return scalar_born(
            v,
            dx,
            dt,
            dv=s,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]

    res_vmap = vmap(prop)(scatter_batch)

    res_loop = torch.stack([prop(scatter_batch[i]) for i in range(batch_size)])

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_jvp_born() -> None:
    """Test jvp on scalar_born."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    v0 = torch.full((ny, nx), 1500.0, device=device, dtype=dtype)
    scatter0 = torch.zeros_like(v0)

    v_tangent = torch.randn_like(v0) * 10.0
    scatter_tangent = torch.randn_like(scatter0) * 0.1

    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2
    receiver_locations = source_locations.clone()
    source_amplitudes = (
        ricker(25.0, 30, dt, 0.05, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    def prop(v, s):
        # We check the scattered wavefield output
        return scalar_born(
            v,
            dx,
            dt,
            dv=s,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )[-1]

    _, jvp_out = jvp(prop, (v0, scatter0), (v_tangent, scatter_tangent))

    # Finite Difference
    epsilon = 1e-6
    out_plus = prop(v0 + epsilon * v_tangent, scatter0 + epsilon * scatter_tangent)
    out_minus = prop(v0 - epsilon * v_tangent, scatter0 - epsilon * scatter_tangent)
    fd_out = (out_plus - out_minus) / (2 * epsilon)

    assert torch.allclose(jvp_out, fd_out, atol=1e-5)


def test_vmap_error_check() -> None:
    """Ensure error is raised if non-property inputs are vmapped."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    v = torch.full((ny, nx), 1500.0, device=device)
    scatter = torch.zeros_like(v)

    # Try to vmap over source amplitudes
    batch_size = 2
    source_amplitudes_batch = torch.randn(batch_size, 1, 1, 10, device=device)

    def prop(amps):
        return scalar_born(
            v,
            5.0,
            0.004,
            dv=scatter,
            source_amplitudes=amps,
            source_locations=torch.zeros(1, 1, 2, dtype=torch.long, device=device),
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )

    with pytest.raises(NotImplementedError, match="Only property models"):
        vmap(prop)(source_amplitudes_batch)


def _born_jvp_problem(ndim, device, dtype, pml_width=4, n=24):
    """Build a scalar/scalar_born problem for the Born=JVP equivalence tests.

    The velocity perturbation (the Born ``scatter``) is a small box centred on
    the (single) source cell, i.e. it deliberately *overlaps the source* and
    lies entirely in the physical interior (clear of the PML). Returns the
    keyword arguments common to ``scalar`` and ``scalar_born`` plus ``v`` and
    the perturbation ``dv``.
    """
    dx = 5.0
    dt = 0.004
    nt = 40
    freq = 25.0
    shape = (n,) * ndim
    centre = tuple(s // 2 for s in shape)

    v = torch.full(shape, 1500.0, device=device, dtype=dtype)
    # A mild, non-constant background so the test is not trivially symmetric.
    if ndim >= 1:
        v[..., n // 2 :] += 200.0

    # Perturbation: a small interior box that includes the source cell.
    dv = torch.zeros_like(v)
    sl = tuple(slice(c - 2, c + 3) for c in centre)
    torch.manual_seed(0)
    dv[sl] = torch.randn(dv[sl].shape, device=device, dtype=dtype) * 20.0
    assert dv[centre] != 0  # genuinely overlaps the source

    n_shots = 1
    source_locations = torch.zeros(n_shots, 1, ndim, dtype=torch.long, device=device)
    receiver_locations = torch.zeros(n_shots, 4, ndim, dtype=torch.long, device=device)
    for d in range(ndim):
        source_locations[..., d] = centre[d]
        # Receivers near (but inside) the boundary, spread along the last dim.
        receiver_locations[..., d] = centre[d]
    receiver_locations[..., -1] = torch.linspace(
        pml_width + 1, n - pml_width - 2, 4, device=device
    ).long()

    source_amplitudes = (
        ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    kwargs = {
        "grid_spacing": dx,
        "dt": dt,
        "source_amplitudes": source_amplitudes,
        "source_locations": source_locations,
        "receiver_locations": receiver_locations,
        "pml_width": pml_width,
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    return v, dv, kwargs


@pytest.mark.parametrize("ndim", [1, 2])
def test_born_is_jvp_of_scalar(ndim) -> None:
    """scalar_born is the exact discrete forward-mode derivative (JVP) of scalar.

    Differentiating the nonlinear ``scalar`` propagator with respect to the
    velocity model in the direction ``dv`` (forward-mode AD) must reproduce,
    to machine precision, the scattered field that ``scalar_born`` computes
    with ``scatter = dv``. This includes the subtle source-cell term
    ``-2 v dv dt^2 f`` that arises because ``scalar`` pre-scales each source by
    ``-v^2 dt^2``: the perturbation box here overlaps the source, so that term
    is exercised. It is already supplied by Born's second (scattered) source.

    The background (primal) receivers must likewise equal ``scalar`` run on the
    unperturbed model. The perturbation is confined to the interior here;
    test_born_jvp_model_edge covers one that is not.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(ndim, device, dtype)

    def scalar_receivers(vv):
        return scalar(vv, backend="eager", **kwargs)[-1]

    primal, tangent = jvp(scalar_receivers, (v,), (dv,))

    # scalar_born with the eager and with the (default) compiled C backend must
    # both equal the forward-mode derivative of scalar. Record background
    # receivers at the same locations to also check the primal output. The
    # eager backend reproduces the JVP exactly (same float ops); the compiled C
    # backend is built with -Ofast / --use_fast_math, so it is compared at a
    # looser backend-consistency tolerance rather than to machine precision.
    # scalar_born names its two receiver sets explicitly; the flat
    # `receiver_locations` in `kwargs` is for `scalar` alone.
    born_kwargs = {
        name: value for name, value in kwargs.items() if name != "receiver_locations"
    }
    for backend, tol in (("eager", 1e-12), ("compiled", 1e-6)):
        out = scalar_born(
            v,
            dv=dv,
            bg_receiver_locations=kwargs["receiver_locations"],
            sc_receiver_locations=kwargs["receiver_locations"],
            backend=backend,
            **born_kwargs,
        )
        bg_receivers, sc_receivers = out[-2], out[-1]
        assert (sc_receivers - tangent).abs().max() <= tol * tangent.abs().max()
        assert (bg_receivers - primal).abs().max() <= tol * primal.abs().max()


def test_born_jvp_model_edge() -> None:
    """Born equals the JVP of scalar for a perturbation of the model's edge.

    The outermost cells of the model are extended outwards to fill the
    absorbing layer and the finite-difference halo, and that is true of the
    scattering model as well as of the background velocity. So a perturbation
    that lives on the edge of the model perturbs the layer too, and the Born
    = JVP equivalence is exact for it, as it is for a perturbation of the
    interior and for an arbitrary one when there is no absorbing layer at
    all. A scattering model that was zeroed outside the model region instead
    would disagree with the JVP here by a large fraction of the signal.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    n = 24
    pml_width = 4
    v, _, kwargs = _born_jvp_problem(2, device, dtype, pml_width=pml_width, n=n)

    def scalar_receivers(vv, kw):
        return scalar(vv, backend="eager", **kw)[-1]

    def born_scatter(dv, kw):
        # scalar_born names its receiver sets explicitly; the flat
        # `receiver_locations` in `kw` is for `scalar` alone.
        born_kw = {
            name: value for name, value in kw.items() if name != "receiver_locations"
        }
        return scalar_born(
            v,
            backend="eager",
            dv=dv,
            sc_receiver_locations=kw["receiver_locations"],
            **born_kw,
        )[-1]

    # 1. Perturbation on the model's outermost rows, which the absorbing
    #    layer and the finite-difference halo are filled from.
    dv_edge = torch.zeros_like(v)
    torch.manual_seed(2)
    dv_edge[0] = torch.randn(n, device=device, dtype=dtype) * 20.0
    dv_edge[-1] = torch.randn(n, device=device, dtype=dtype) * 20.0
    _, tangent_edge = jvp(lambda vv: scalar_receivers(vv, kwargs), (v,), (dv_edge,))
    born_edge = born_scatter(dv_edge, kwargs)
    assert tangent_edge.abs().max() > 0
    rel = (tangent_edge - born_edge).abs().max() / tangent_edge.abs().max()
    assert rel <= 1e-12

    # 2. With pml_width=0 the equivalence is exact for an arbitrary perturbation.
    kwargs0 = {**kwargs, "pml_width": 0}
    dv_any = torch.randn_like(v) * 10.0
    _, tangent0 = jvp(lambda vv: scalar_receivers(vv, kwargs0), (v,), (dv_any,))
    born0 = born_scatter(dv_any, kwargs0)
    assert torch.allclose(born0, tangent0, atol=1e-11, rtol=0)


def test_born_linear_in_scatter() -> None:
    """The Born scattered field is linear in the scattering potential.

    Born modelling linearises the wave equation, so the scattered output is a
    linear operator applied to ``scatter``: for scalars a, b and potentials
    m1, m2, born(a*m1 + b*m2) == a*born(m1) + b*born(m2). This is the defining
    property of the Born/linearised forward map and is independent of the JVP
    relationship above.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, m1, kwargs = _born_jvp_problem(2, device, dtype)
    torch.manual_seed(3)
    m2 = torch.zeros_like(v)
    m2[6:18, 6:18] = torch.randn(12, 12, device=device, dtype=dtype) * 15.0
    a, b = 1.7, -0.6

    # scalar_born names its receiver sets explicitly; the flat
    # `receiver_locations` in `kwargs` is for `scalar` alone.
    born_kwargs = {
        name: value for name, value in kwargs.items() if name != "receiver_locations"
    }

    def born_scatter(m):
        return scalar_born(
            v,
            backend="eager",
            dv=m,
            sc_receiver_locations=kwargs["receiver_locations"],
            **born_kwargs,
        )[-1]

    combined = born_scatter(a * m1 + b * m2)
    superposed = a * born_scatter(m1) + b * born_scatter(m2)
    assert torch.allclose(combined, superposed, atol=1e-12, rtol=0)


def test_second_order_raises() -> None:
    """Double-backward through scalar_born raises a clear error.

    The scalar_born backward has no analytic second-order path. A
    Hessian-vector product would silently return an incorrect (often zero)
    result, so :func:`deepwave._engine.no_second_order` makes it raise instead.
    The non-Born ``scalar`` propagator should be used when a Hessian-vector
    product is required.
    """
    ny, nx = 12, 12
    dx = 5.0
    dt = 0.004
    nt = 25
    source_amplitudes = ricker(25.0, nt, dt, 0.04).reshape(1, 1, nt).double()
    source_locations = torch.tensor([[[ny // 2, nx // 2]]])
    receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 3]]])
    v = (torch.ones(ny, nx, dtype=torch.float64) * 1500.0).requires_grad_(True)
    scatter = torch.zeros(ny, nx, dtype=torch.float64)
    scatter[ny // 2, nx // 2 + 1] = 100.0

    out = scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        pml_width=3,
        accuracy=2,
    )[-1]
    loss = out.pow(2).sum()
    # Building a second-order graph (create_graph=True) invokes the backward
    # with grad enabled, which is where the guard fires -- on this very call,
    # before a second .grad() is even needed.
    with pytest.raises(RuntimeError, match="Second-order differentiation"):
        torch.autograd.grad(loss, v, create_graph=True)


def test_first_order_unaffected_by_second_order_guard() -> None:
    """Plain first-order scalar_born gradients are unchanged by the guard."""
    ny, nx = 12, 12
    dx = 5.0
    dt = 0.004
    nt = 25
    source_amplitudes = ricker(25.0, nt, dt, 0.04).reshape(1, 1, nt).double()
    source_locations = torch.tensor([[[ny // 2, nx // 2]]])
    receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 3]]])

    def grad_v() -> torch.Tensor:
        v = (torch.ones(ny, nx, dtype=torch.float64) * 1500.0).requires_grad_(True)
        scatter = torch.zeros(ny, nx, dtype=torch.float64)
        scatter[ny // 2, nx // 2 + 1] = 100.0
        out = scalar_born(
            v,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            pml_width=3,
            accuracy=2,
        )[-1]
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
    scatter = 100.0 * torch.ones(ny, nx)

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
        return scalar_born(
            model,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            model_gradient_sampling_interval=step_ratio,
        )

    run_checkpoint_equivalence(v, call, use_reentrant=use_reentrant)


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    nz, ny = 8, 8
    v = torch.ones(nz, ny) * 1500.0
    dv = torch.zeros(nz, ny)
    source_amplitudes = torch.zeros(1, 1, 8)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        scalar_born(
            v,
            5.0,
            0.001,
            dv=dv,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            max_vel=2000.0,
            backend="eager",
            storage_compression=True,
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_born_output_names(ndim) -> None:
    """The returned PropagatorOutput's field names match scalar_born_output_names.

    ``out._names`` (set by ``_engine.named_output`` from ``SPEC.output_names``)
    must equal ``scalar_born_output_names(ndim)`` exactly, in order: this is
    the contract that lets ``out.sc_receiver_amplitudes`` and the other named
    fields resolve to the correct positional tensor for every ``ndim``.
    """
    from deepwave.scalar_born import scalar_born_output_names

    n = 8
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0)
    scatter = torch.full(shape, 1.0)
    source_amplitudes = torch.zeros(1, 1, 8)
    source_amplitudes[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    out = scalar_born(
        v,
        5.0,
        0.004,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=loc,
        sc_receiver_locations=loc,
        bg_receiver_locations=loc,
        pml_width=3,
    )
    assert out._names == tuple(scalar_born_output_names(ndim))
    for name in out._names:
        assert hasattr(out, name)


# ---------------------------------------------------------------------------
# The scatter -> dv rename: `dv` is the name everywhere (constructor argument,
# stored attribute, and CallbackState model/gradient key), with `scatter`
# retained as a deprecated alias in each of those three places because
# v0.0.27 shipped all three.
# ---------------------------------------------------------------------------


def _tiny_born_kwargs() -> dict:
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    locations = torch.full((1, 1, 2), 5, dtype=torch.long)
    return {
        "source_amplitudes": source_amplitudes,
        "source_locations": locations,
        "sc_receiver_locations": locations,
    }


def test_scalarborn_stores_dv_attribute() -> None:
    """The stored scattering-potential attribute is ``dv``."""
    v = 1500 * torch.ones(10, 10)
    dv = torch.ones(10, 10)
    prop = ScalarBorn(v, 5.0, dv=dv)
    assert torch.equal(prop.dv, dv)
    assert "dv" in dict(prop.named_parameters())


def test_scalarborn_scatter_attribute_is_deprecated_alias() -> None:
    """``.scatter`` still reads the same tensor, with a deprecation warning."""
    v = 1500 * torch.ones(10, 10)
    dv = torch.ones(10, 10)
    prop = ScalarBorn(v, 5.0, dv=dv)
    with pytest.warns(deepwave.DeepwaveFutureWarning, match="scatter is deprecated"):
        scatter = prop.scatter
    assert scatter is prop.dv


def test_scalarborn_scatter_attribute_is_read_only() -> None:
    """``.scatter`` cannot be assigned to: it is a read-only alias for ``dv``."""
    v = 1500 * torch.ones(10, 10)
    prop = ScalarBorn(v, 5.0, dv=torch.ones(10, 10))
    with pytest.raises(AttributeError):
        prop.scatter = torch.zeros(10, 10)


def test_scalarborn_scatter_constructor_alias_sets_dv() -> None:
    """The deprecated ``scatter=`` constructor argument warns and populates ``dv``."""
    v = 1500 * torch.ones(10, 10)
    dv = torch.ones(10, 10)
    with pytest.warns(deepwave.DeepwaveFutureWarning, match="scatter is deprecated"):
        prop = ScalarBorn(v, 5.0, scatter=dv)
    assert torch.equal(prop.dv, dv)


def test_scalar_born_model_names_use_dv() -> None:
    """``scalar_born``'s callback model key is ``dv``, not ``scatter``."""
    from deepwave.scalar_born import scalar_born_model_names

    assert scalar_born_model_names(2) == ["v", "dv"]


def test_callback_model_scatter_key_is_deprecated_alias() -> None:
    """``get_model("scatter")`` still resolves to ``dv``, with a warning."""
    v = 1500 * torch.ones(10, 10)
    dv = torch.ones(10, 10)
    seen: list[torch.Tensor] = []

    def callback(state: Any) -> None:
        if not seen:
            with pytest.warns(
                deepwave.DeepwaveFutureWarning, match="'scatter' is deprecated"
            ):
                seen.append(state.get_model("scatter", "full"))
            seen.append(state.get_model("dv", "full"))

    scalar_born(
        v,
        5.0,
        0.004,
        dv=dv,
        forward_callback=callback,
        callback_frequency=20,
        **_tiny_born_kwargs(),
    )
    assert len(seen) == 2
    assert seen[0] is seen[1]


def test_callback_gradient_scatter_key_is_deprecated_alias() -> None:
    """``get_gradient("scatter")`` still resolves to ``dv``, with a warning."""
    v = 1500 * torch.ones(10, 10)
    dv = torch.ones(10, 10, requires_grad=True)
    seen: list[bool] = []

    def callback(state: Any) -> None:
        if not seen:
            with pytest.warns(
                deepwave.DeepwaveFutureWarning, match="'scatter' is deprecated"
            ):
                alias = state.get_gradient("scatter", "full")
            assert alias is state.get_gradient("dv", "full")
            seen.append(True)

    out = scalar_born(
        v,
        5.0,
        0.004,
        dv=dv,
        backward_callback=callback,
        callback_frequency=20,
        **_tiny_born_kwargs(),
    )
    out[-1].sum().backward()
    assert seen


def test_callback_unknown_model_name_still_raises_key_error() -> None:
    """A name that is neither current nor a deprecated alias still raises."""
    v = 1500 * torch.ones(10, 10)
    dv = torch.ones(10, 10)
    errors: list[str] = []

    def callback(state: Any) -> None:
        if not errors:
            with pytest.raises(KeyError, match="model 'bogus' is not available"):
                state.get_model("bogus")
            errors.append("raised")

    scalar_born(
        v,
        5.0,
        0.004,
        dv=dv,
        forward_callback=callback,
        callback_frequency=20,
        **_tiny_born_kwargs(),
    )
    assert errors

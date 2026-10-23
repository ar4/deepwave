"""Model gradients must not depend on how a callback chunks the adjoint loop.

Supplying a ``backward_callback`` splits the backward pass into
``ceil(n_outer / callback_frequency)` separate backend calls instead of one
(``_engine.run_backend_loop``). That is purely a control-flow decision: the
gradient a user gets back must be identical to the gradient of the same call
with no callback at all, for every ``callback_frequency``.

It was not. On CUDA, when a model is shared across shots (not batched
per-shot) and ``n_shots > 1``, ``_engine.allocate_model_gradients`` hands the
backend a per-shot scratch buffer that the compiled ``combine_grad_*`` kernels
reduce into the final gradient at the end of *every* backend call. The scratch
was zeroed once at allocation, so chunk ``k``'s partial sum was added in again
by every later chunk and the returned model gradient came back inflated by a
chunk-count-dependent factor -- two to three orders of magnitude on the shapes
below, silently, in the middle of the documented custom-imaging-condition
workflow (``docs/example_custom_imaging_condition.rst``). The CPU backends were
unaffected: the ones that use intra-shot spatial parallelism allocate no
scratch at all.

**All nine compiled propagators were affected**, not only the six with a
plain positional-model signature. The Born family (``scalar_born``,
``acoustic_born``, ``elastic_born``) and ``elastic(surface_method="pm")`` are
covered by the second group of tests below; the PM case is the only coverage
of ``run_backward_core(extra_grad_scratch=...)``, the separate path by which
the ``DW_PM`` stiffness-array scratch (``grad_c00``..``grad_c11``, reduced by
the same ``combine_grad`` kernels at the end of every call) is re-zeroed.
Measured relative error against the no-callback reference at
``callback_frequency=1`` on CUDA before the fix: scalar_born 5.8e+02,
acoustic_born 1.5e+02, elastic_born 1.6e+02, elastic PM 1.5e+02.

These tests pin the invariant for every propagator that reaches the shared
engine, on both devices, so neither the scratch-zeroing nor a future change to
the chunking can regress it unnoticed.
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave import vpvsrho_to_lambmubuoyancy

DTYPE = torch.float64
NY, NX, NT, N_SHOTS = 30, 34, 150, 3
DX, DT = 5.0, 0.0005
PML_FREQ = 25.0

# A ``build`` returns everything one propagation needs: the callable, its
# positional arguments, its keyword arguments, and the tensors whose gradients
# are compared. ``callback_frequency`` is added to the kwargs by ``_gradients``.
Build = Callable[[torch.device], tuple[Any, tuple, dict, list[torch.Tensor]]]

# Each simple propagator's constant model values (physically sensible,
# homogeneous, comfortably inside the CFL limit at the DX/DT above) and the
# source/receiver component suffixes its 2-D signature uses.
CASES = {
    "scalar": ([1500.0], "", ""),
    "acoustic": ([1500.0, 2200.0], "_p", "_p"),
    "viscoacoustic": ([1500.0, 2200.0, 40.0], "_pressure_rate", "_p"),
    "elastic": ([1.2e10, 0.8e10, 1 / 2200.0], "_p", "_p"),
    "acoustic_vti": ([1.2e10, 0.4e10, 1.0e10, 1 / 2200.0], "_sigma_v", "_y"),
    "elastic_vti": ([1.2e10, 0.4e10, 1.0e10, 0.5e10, 1 / 2200.0], "_y", "_y"),
}


def _survey(
    device: torch.device, ny: int = NY, nt: int = NT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A per-shot-offset source/receiver pair and a random source wavelet.

    The shots must differ so that each one's contribution to the shared model
    gradient is distinct; identical shots would make an over-counted reduction
    much harder to distinguish from a correct one.
    """
    sources = torch.zeros(N_SHOTS, 1, 2, dtype=torch.long, device=device)
    receivers = torch.zeros(N_SHOTS, 1, 2, dtype=torch.long, device=device)
    for shot in range(N_SHOTS):
        sources[shot, 0] = torch.tensor([ny // 5, 6 + shot])
        receivers[shot, 0] = torch.tensor([ny // 5 + 6, 14 - shot])
    generator = torch.Generator().manual_seed(0)
    amplitudes = torch.randn(N_SHOTS, 1, nt, dtype=DTYPE, generator=generator).to(
        device
    )
    return amplitudes, sources, receivers


def _full(value: float, device: torch.device, shape: tuple[int, int]) -> torch.Tensor:
    return torch.full(shape, value, dtype=DTYPE, device=device, requires_grad=True)


def _simple_build(name: str) -> Build:
    """A ``Build`` for a propagator whose models are all positional."""

    def build(device: torch.device) -> tuple[Any, tuple, dict, list[torch.Tensor]]:
        values, source_suffix, receiver_suffix = CASES[name]
        models = [_full(value, device, (NY, NX)) for value in values]
        amplitudes, sources, receivers = _survey(device)
        kwargs = {
            "pml_freq": PML_FREQ,
            f"source_amplitudes{source_suffix}": amplitudes,
            f"source_locations{source_suffix}": sources,
            f"receiver_locations{receiver_suffix}": receivers,
        }
        if name == "viscoacoustic":
            # The frequency Q is specified at; required on the attenuating
            # propagators, refused on the rest.
            kwargs["q_freq"] = PML_FREQ
        return getattr(deepwave, name), (*models, DX, DT), kwargs, models

    return build


# --------------------------------------------------------------------------
# The Born family: scattering models are keyword-only (the argument-order
# convention), so
# they cannot go through _simple_build's positional-model path.
# --------------------------------------------------------------------------


def _scalar_born_build(device: torch.device) -> tuple[Any, tuple, dict, list]:
    v = _full(1500.0, device, (NY, NX))
    dv = _full(30.0, device, (NY, NX))
    amplitudes, sources, receivers = _survey(device)
    kwargs = {
        "pml_freq": PML_FREQ,
        "dv": dv,
        "source_amplitudes": amplitudes,
        "source_locations": sources,
        "receiver_locations": receivers,
    }
    return deepwave.scalar_born, (v, DX, DT), kwargs, [v, dv]


def _acoustic_born_build(device: torch.device) -> tuple[Any, tuple, dict, list]:
    v = _full(1500.0, device, (NY, NX))
    rho = _full(2200.0, device, (NY, NX))
    dv = _full(30.0, device, (NY, NX))
    drho = _full(20.0, device, (NY, NX))
    amplitudes, sources, receivers = _survey(device)
    kwargs = {
        "pml_freq": PML_FREQ,
        "dv": dv,
        "drho": drho,
        "source_amplitudes_volume_rate": amplitudes,
        "source_locations_volume_rate": sources,
        "sc_receiver_locations_p": receivers,
    }
    return deepwave.acoustic_born, (v, rho, DX, DT), kwargs, [v, rho, dv, drho]


def _elastic_born_build(device: torch.device) -> tuple[Any, tuple, dict, list]:
    lamb = _full(1.2e10, device, (NY, NX))
    mu = _full(0.8e10, device, (NY, NX))
    buoyancy = _full(1 / 2200.0, device, (NY, NX))
    dlamb = _full(1e8, device, (NY, NX))
    dmu = _full(1e8, device, (NY, NX))
    dbuoyancy = _full(1e-6, device, (NY, NX))
    amplitudes, sources, receivers = _survey(device)
    kwargs = {
        "pml_freq": PML_FREQ,
        "dlamb": dlamb,
        "dmu": dmu,
        "dbuoyancy": dbuoyancy,
        "source_amplitudes_pressure_rate": amplitudes,
        "source_locations_pressure_rate": sources,
        "sc_receiver_locations_p": receivers,
    }
    models = [lamb, mu, buoyancy, dlamb, dmu, dbuoyancy]
    return deepwave.elastic_born, (lamb, mu, buoyancy, DX, DT), kwargs, models


def _elastic_pm_build(device: torch.device) -> tuple[Any, tuple, dict, list]:
    """``elastic(surface_method="pm")``: the ``extra_grad_scratch`` path.

    ``pm`` requires a vacuum region (``mu == 0``) to hang the free surface on,
    so the model is a solid half-space beneath five rows of vacuum. Its four
    ``DW_PM`` stiffness arrays get their own per-shot gradient scratch,
    allocated outside the engine's ``models`` list and threaded in through
    ``run_backward_core(extra_grad_scratch=...)``.
    """
    vacuum = 5
    vp = torch.zeros(NY, NX, dtype=DTYPE, device=device)
    vs = torch.zeros(NY, NX, dtype=DTYPE, device=device)
    rho = torch.zeros(NY, NX, dtype=DTYPE, device=device)
    vp[vacuum:] = 2000.0
    vs[vacuum:] = 1150.0
    rho[vacuum:] = 2000.0
    lamb, mu, buoyancy = (
        model.detach().requires_grad_(True)
        for model in vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    )
    amplitudes, sources, receivers = _survey(device)
    kwargs = {
        "pml_freq": PML_FREQ,
        "surface_method": "pm",
        # A zero-width PML on the vacuum face, real PML elsewhere: the free
        # surface is realised by the parameter modification, not by a boundary
        # condition, so this does not trip the closed-free-axis warning.
        "pml_width": [0, 10, 10, 10],
        "source_amplitudes_pressure_rate": amplitudes,
        "source_locations_pressure_rate": sources,
        "receiver_locations_p": receivers,
    }
    models = [lamb, mu, buoyancy]
    return deepwave.elastic, (lamb, mu, buoyancy, DX, DT), kwargs, models


EXTRA_CASES: dict[str, Build] = {
    "acoustic_born": _acoustic_born_build,
    "elastic_born": _elastic_born_build,
    "elastic_pm": _elastic_pm_build,
    "scalar_born": _scalar_born_build,
}


def _gradients(
    build: Build, device: torch.device, callback_frequency: Optional[int]
) -> list[torch.Tensor]:
    """Runs one propagation and returns the gradient of every model.

    ``callback_frequency=None`` means "no callback at all", i.e. the
    single-chunk reference the chunked runs must reproduce.
    """
    function, args, kwargs, models = build(device)
    kwargs = dict(kwargs)
    if callback_frequency is not None:
        kwargs["backward_callback"] = lambda state: None
        kwargs["callback_frequency"] = callback_frequency
    out = function(*args, **kwargs)
    # Sum every non-empty output tensor: the multi-component propagators
    # return a long list in which the channels this survey did not request are
    # empty, and several return wavefields as well as receiver amplitudes.
    # ``output_tensors`` expands the ``auxiliary_state`` element into the
    # tensors it holds, so the absorbing boundary's memory variables are in
    # the loss too. Using all of them keeps it sensitive to each model.
    sum(tensor.sum() for tensor in output_tensors(out) if tensor.numel() > 0).backward()
    gradients = []
    for model in models:
        assert model.grad is not None
        gradients.append(model.grad.detach().clone())
    return gradients


def _assert_chunking_invariant(
    build: Build,
    name: str,
    device: torch.device,
    callback_frequencies: Sequence[int] = (1, 3, 7),
) -> None:
    reference = _gradients(build, device, None)
    # The reference must actually depend on every model, or a wrong gradient
    # would compare equal to a wrong gradient and the test would prove nothing.
    assert all(gradient.norm() > 0 for gradient in reference), (
        f"{name} on {device}: the no-callback reference gradient is zero for at "
        "least one model, so this test would be vacuous"
    )
    for callback_frequency in callback_frequencies:
        chunked = _gradients(build, device, callback_frequency)
        for index, (actual, expected) in enumerate(zip(chunked, reference)):
            relative = ((actual - expected).norm() / expected.norm()).item()
            assert relative < 1e-10, (
                f"{name} on {device}, model {index}: gradient with "
                f"callback_frequency={callback_frequency} differs from the "
                f"no-callback gradient by a relative {relative:.3e}"
            )


@pytest.mark.parametrize("name", sorted(CASES))
def test_model_gradient_is_independent_of_callback_frequency_cpu(name: str) -> None:
    """Chunking the adjoint loop must not change the CPU model gradient."""
    _assert_chunking_invariant(_simple_build(name), name, torch.device("cpu"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_nonmatching_callback_and_history_intervals_cpu(name: str) -> None:
    """Callback boundaries inside a two-user-step history stride are safe."""
    original = _simple_build(name)

    def build(device: torch.device):
        function, args, kwargs, models = original(device)
        return function, args, {**kwargs, "model_gradient_sampling_interval": 2}, models

    _assert_chunking_invariant(
        build, name, torch.device("cpu"), callback_frequencies=(3, 7)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", sorted(CASES))
def test_model_gradient_is_independent_of_callback_frequency_cuda(name: str) -> None:
    """Chunking the adjoint loop must not change the CUDA model gradient.

    This is the direction that was broken: the per-shot gradient scratch the
    CUDA backends reduce from is only allocated when ``n_shots > 1`` and the
    model is shared across shots, which is exactly this survey.
    """
    _assert_chunking_invariant(_simple_build(name), name, torch.device("cuda"))


@pytest.mark.parametrize("name", sorted(EXTRA_CASES))
def test_born_and_pm_model_gradient_is_independent_of_callback_frequency_cpu(
    name: str,
) -> None:
    """The same invariant for the Born family and ``surface_method="pm"``.

    Two ``callback_frequency`` values rather than three, since these are the
    slowest propagators in the file; 7 does not divide ``NT``, so the last
    chunk is short.
    """
    _assert_chunking_invariant(
        EXTRA_CASES[name], name, torch.device("cpu"), callback_frequencies=(1, 7)
    )


@pytest.mark.parametrize("name", sorted(EXTRA_CASES))
def test_born_and_pm_nonmatching_callback_history_cpu(name: str) -> None:
    """Born and PM gradients survive callback cuts inside history intervals."""
    original = EXTRA_CASES[name]

    def build(device: torch.device):
        function, args, kwargs, models = original(device)
        return function, args, {**kwargs, "model_gradient_sampling_interval": 2}, models

    _assert_chunking_invariant(
        build, name, torch.device("cpu"), callback_frequencies=(3, 7)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", sorted(EXTRA_CASES))
def test_born_and_pm_model_gradient_is_independent_of_callback_frequency_cuda(
    name: str,
) -> None:
    """The broken direction for the Born family and ``surface_method="pm"``.

    ``elastic_pm`` is the only test in the suite that exercises
    ``run_backward_core(extra_grad_scratch=...)``: the ``DW_PM`` stiffness
    scratch lives outside the engine's ``models`` list and is re-zeroed
    between chunks by that argument alone.
    """
    _assert_chunking_invariant(
        EXTRA_CASES[name], name, torch.device("cuda"), callback_frequencies=(1, 7)
    )

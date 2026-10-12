"""Measure finite-record model-gradient quadrature with fixed numerical setup.

Full eager differentiation supplies the finite-call reference on tiny grids.
The native default keeps user-rate histories. Errors at effective stride two
are measured rather than confused with the exact stride-one adjoint contract.
"""

import inspect
import math

import pytest
import torch
from test_gradient_exactness import _base_velocity, _models, _quiet

import deepwave


@pytest.mark.parametrize("family", deepwave._engine.PROPAGATOR_NAMES)
@pytest.mark.parametrize(
    ("frequency", "nt"), [(25.0, 64), (25.0, 128), (180.0, 64), (180.0, 128)]
)
def test_finite_record_sampling_error(family, frequency, nt, record_property):
    """Record nonzero gradient norm and directional errors for all families."""
    spec = deepwave.spec(family)
    function = getattr(deepwave, family)
    source = torch.tensor([[[8, 6]]])
    receivers = torch.tensor([[[8, 8], [9, 8]]])
    dt = 0.001
    amplitude = deepwave.wavelets.ricker(
        frequency, nt, dt, 1.0 / frequency, dtype=torch.float64
    ).reshape(1, 1, -1)
    # A tapered trace-energy objective limits endpoint discontinuities while
    # retaining the finite causal start and terminal adjoint of a real call.
    taper = torch.sin(torch.arange(nt, dtype=torch.float64) * math.pi / nt).square()
    direction = torch.cos(torch.arange(256, dtype=torch.float64).reshape(16, 16) / 31)

    def run(backend):
        velocity = _base_velocity().requires_grad_(True)
        models = _models(family, velocity)
        parameters = list(inspect.signature(function).parameters)
        positional_names = parameters[: parameters.index("grid_spacing")]
        positional = [models[name] for name in positional_names if name in models]
        kwargs = {
            name: value
            for name, value in models.items()
            if name not in positional_names
        }
        kwargs.update(
            pml_width=3, pml_freq=25.0, accuracy=4, max_vel=3000.0, backend=backend
        )
        for index, channel in enumerate(spec.source_channels(2)):
            kwargs[spec.source_amplitudes_kwarg(channel)] = amplitude
            kwargs[spec.source_locations_kwarg(channel)] = source + torch.tensor(
                [[[0, index]]]
            )
        channel = spec.receiver_channels(2)[0]
        kwargs[spec.receiver_locations_kwarg(channel)] = receivers
        if family == "viscoacoustic":
            kwargs["q_freq"] = 25.0
        output = function(*positional, 5.0, dt, **kwargs)
        trace = getattr(output, spec.receiver_output_name(channel))
        loss = (trace.square() * taper).sum()
        (gradient,) = torch.autograd.grad(loss, velocity)
        return output.info, gradient

    with _quiet():
        sampled_info, sampled = run("compiled")
        full_info, full = run("eager")
    assert sampled_info.step_ratio == full_info.step_ratio == 2
    assert sampled_info.model_gradient_stride == 2
    assert full_info.model_gradient_stride == 1
    norm = float(full.norm())
    assert norm > 0
    assert bool(torch.isfinite(full).all())
    assert float(sampled.norm()) > 0
    assert bool(torch.isfinite(sampled).all())
    error = float((sampled - full).norm()) / norm
    directional_scale = norm * float(direction.norm())
    directional_error = (
        abs(float(((sampled - full) * direction).sum())) / directional_scale
    )
    assert math.isfinite(error)
    assert math.isfinite(directional_error)
    # These bounds belong to this fixed experiment, with room above the
    # measured limits. They guard its fidelity without claiming an arbitrary
    # finite-record objective must satisfy them.
    assert error < 5e-4
    assert directional_error < 1e-5
    if frequency == 25.0 and nt == 128:
        assert error < 2e-6
    full_directional = float((full * direction).sum())
    assert full_directional != 0
    record_property("frequency_hz", frequency)
    record_property("record_seconds", nt * dt)
    record_property("full_gradient_norm", norm)
    record_property("relative_gradient_error", error)
    record_property("scaled_directional_error", directional_error)
    record_property("full_directional_derivative", full_directional)

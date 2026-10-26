"""Demonstrates optimising source amplitudes to match a target wavefield.

This script optimises source amplitudes to minimise the difference between
the final wavefield and a target wavefield. This shows that optimisation
can involve any of the inputs and outputs, not just the usual task of
adjusting the velocity model to match the receiver data.
"""

import math
import os
from pathlib import Path

import torch
import torchvision

import deepwave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny = 200
nx = 200
dx = 5.0
dt = 0.004
nt = 40 if smoke else 100
n_shots = 1
n_sources_per_shot = 100 if smoke else 400
freq = 25
peak_time = 1.5 / freq
# Resolved relative to this script's own location, not the working
# directory, so the example runs correctly from any current directory.
target_path = Path(__file__).resolve().parent / "target.jpg"
target = (
    torchvision.io.read_image(
        str(target_path), torchvision.io.ImageReadMode.GRAY
    ).float()
    / 255
)
target = (target[0] - target.mean()).to(device)

v = 2000 * torch.ones(ny, nx, device=device)

# source_locations
source_locations = torch.zeros(
    n_shots,
    n_sources_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
torch.manual_seed(1)
grid_cells = torch.cartesian_prod(torch.arange(ny), torch.arange(nx))
source_cell_idxs = torch.randperm(len(grid_cells))[:n_sources_per_shot]
source_locations = (
    grid_cells[source_cell_idxs]
    .reshape(1, n_sources_per_shot, 2)
    .long()
    .to(device)
)

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)
source_amplitudes.requires_grad_()

optimiser = torch.optim.LBFGS([source_amplitudes])
loss_fn = torch.nn.MSELoss()
loss_history: list[float] = []


def closure():
    """Closure function for the LBFGS optimiser."""
    optimiser.zero_grad()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        pml_width=0,
    )
    y = out.wavefield_0[0]
    loss = loss_fn(y, target) + 1e-3 * source_amplitudes.norm()
    loss.backward()
    loss_history.append(loss.item())
    return loss.item()


for _i in range(3 if smoke else 50):
    optimiser.step(closure)

(source_amplitudes.detach().cpu().numpy().tofile("source_amplitudes.bin"))

target_abs_max = target.abs().max().item()


def forward_callback(state):
    """A function called during the forward pass."""
    # Scale the wavefield to be between 0 and 1 and save as an image
    val = (
        state.get_wavefield("wavefield_0")[0].cpu() / target_abs_max / 2
        + 0.5
    )
    torchvision.utils.save_image(val, f"wavefield_{state.step:03d}.jpg")


reference = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes.detach(),
    source_locations=source_locations,
    pml_width=0,
    forward_callback=forward_callback,
    callback_frequency=1,
)

# Alternative method of saving snapshots
# We want to save every time step, so we will create a loop over time steps.
# When we call the wave propagator, we only want it to advance by
# one time step.
# We can achieve this by calling the propagator with each time sample of the
# source amplitudes. As we discussed in the checkpointing example, however,
# that might not give us exactly the result that we want due to
# upscaling within Deepwave to obey the CFL condition. We therefore perform
# the upscaling ourselves and then call the propagator with chunks of the
# upscaled source amplitudes that correspond to one pre-upsampling time
# step. Passing ``initial_state=out`` carries the whole propagator state --
# both wavefields and, when a PML is in use, its auxiliary fields -- from
# one chunk into the next, which is what makes the restarted run equivalent
# to the single call above.
inner_dt, step_ratio = deepwave.discretisation.cfl_condition(
    dx, dx, dt, 2000
)
upsampled_amplitudes = deepwave.discretisation.upsample(
    source_amplitudes.detach(), step_ratio
)

# ``pml_freq`` must be given explicitly here. When it is omitted, Deepwave
# infers it from the peak of the source amplitudes' spectrum -- but each
# chunk is only a few samples long, so every chunk would infer a different
# frequency and, with a non-zero ``pml_width``, every chunk would get a
# different absorbing boundary. This example uses ``pml_width=0``, so the
# value cannot affect its numbers, but a restarted run with a PML must pass
# the same ``pml_freq`` to every call.
out = None
for i in range(nt):
    chunk = upsampled_amplitudes[..., i * step_ratio : (i + 1) * step_ratio]
    out = deepwave.scalar(
        v,
        dx,
        inner_dt,
        source_amplitudes=chunk,
        source_locations=source_locations,
        pml_width=0,
        pml_freq=freq,
        initial_state=out,
    )
    val = out.wavefield_0[0].cpu() / target_abs_max / 2 + 0.5
    torchvision.utils.save_image(val, f"wavefield_{i:03d}.jpg")

assert out is not None
# The two methods above propagate the same source through the same model, so
# they must agree. They are not required to agree bit-exactly: the single
# call resamples the source amplitudes once for the whole record, while the
# loop resamples once and then splits the result, and the two orderings of
# the same arithmetic differ at rounding level.
restart_error = (out.wavefield_0 - reference.wavefield_0).abs().max().item()
reference_scale = reference.wavefield_0.abs().max().item()
assert restart_error <= 1e-5 * reference_scale, (
    f"restarted propagation disagrees with the single call: "
    f"{restart_error} against a field maximum of {reference_scale}"
)

assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

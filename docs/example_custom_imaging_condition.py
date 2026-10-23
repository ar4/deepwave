"""Demonstrates callbacks for custom imaging conditions.

Part 1 (default compiled backend) uses Deepwave's callback feature to record
source-side illumination during forward propagation and to divide the
completed velocity gradient by the illumination after backward
propagation to implement a simple illumination compensation.

Part 2 (default compiled backend) builds a *true* imaging condition that
needs the forward and adjoint wavefields at the *same* time step, which
``state.get_wavefield(name, role="forward")`` provides alongside the adjoint
field. We form a zero-lag forward x adjoint cross-correlation image.

Retaining that forward wavefield costs a second full wavefield history, so
it is opt-in: Part 2 passes
``backward_callback_needs_forward_wavefield=True``. Part 1 does not, and so
does not pay for it -- which is the point of the flag. A callback that only
inspects the adjoint field (a NaN check, a norm monitor) should never double
the memory of a 3-D run.
"""

import os

import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter

import deepwave

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the model
# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny = 60 if smoke else 200
nx = 60 if smoke else 200
v_true = torch.ones(ny, nx, device=device) * 1500
layer_spacing = ny // 5
for interface_idx in range(1, 5):
    v_true[interface_idx * layer_spacing :] += 50
dx = 5.0

n_shots = 3 if smoke else 20
n_sources_per_shot = 1
d_source = 10
first_source = 5
source_depth = 0

n_receivers_per_shot = nx
receiver_depth = 0

dt = 0.008
freq = 25
nt = 140 if smoke else 500
peak_time = 1.5 / freq

# Smooth to use as starting model
v = torch.tensor(1 / gaussian_filter(1 / v_true.cpu().numpy(), 40)).to(
    device
)
v.requires_grad_()

# source_locations
source_locations = torch.zeros(
    n_shots,
    n_sources_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
source_locations[..., 0] = source_depth
source_locations[:, 0, 1] = torch.arange(n_shots) * d_source + first_source

# receiver_locations
receiver_locations = torch.zeros(
    n_shots,
    n_receivers_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
receiver_locations[..., 0] = receiver_depth
receiver_locations[:, :, 1] = (torch.arange(n_receivers_per_shot)).repeat(
    n_shots, 1
)

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# Generate true data
out_true = deepwave.scalar(
    v_true,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)

# Calculate the standard gradient
out = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
)
torch.nn.MSELoss()(
    out.receiver_amplitudes, out_true.receiver_amplitudes
).backward()

standard_gradient = v.grad.detach().cpu().clone()
v.grad.zero_()
del out

# Calculate the gradient with illumination compensation
illumination = None


def forward_callback(state):
    """Callback function for the forward pass to calculate illumination."""
    global illumination
    if illumination is None:
        illumination = (
            state.get_wavefield("wavefield_0", view="inner").detach() ** 2
        ).sum(dim=0)
    else:
        illumination += (
            state.get_wavefield("wavefield_0", view="inner").detach() ** 2
        ).sum(dim=0)


out = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    forward_callback=forward_callback,
    callback_frequency=1,
)
torch.nn.MSELoss()(
    out.receiver_amplitudes, out_true.receiver_amplitudes
).backward()

# This is an optimiser preconditioner, applied after all contributions
# (including source scaling and model padding) reach the leaf gradient.
with torch.no_grad():
    v.grad /= illumination + 1e-3 * illumination.max()

compensated_gradient = v.grad.detach().cpu().clone()

# -----------------------------------------------------------------------
# Part 2: a true forward x adjoint imaging condition (default backend)
# -----------------------------------------------------------------------
# Both scalar backends can surface the forward field in the backward
# callback when asked to retain it; this part uses the default compiled one.
v.grad.zero_()
rtm_image = torch.zeros(ny, nx, device=device)


def imaging_callback(state):
    """Accumulate sum_t forward(t) * adjoint(t) (zero-lag correlation)."""
    forward = state.get_wavefield(
        "wavefield_0", view="inner", role="forward"
    )
    adjoint = state.get_wavefield("wavefield_0", view="inner")
    rtm_image.add_((forward * adjoint).sum(dim=0))


out = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    backward_callback=imaging_callback,
    backward_callback_needs_forward_wavefield=True,
    callback_frequency=1,
)
torch.nn.MSELoss()(
    out.receiver_amplitudes, out_true.receiver_amplitudes
).backward()
rtm_image = rtm_image.detach().cpu().clone()

# Plot
_, ax = plt.subplots(1, 3, figsize=(15.5, 3.5))
vmax = torch.quantile(standard_gradient, 0.92)
ax[0].imshow(standard_gradient, cmap="gray", vmin=-vmax, vmax=vmax)
ax[0].set_title("Standard gradient")
vmax = torch.quantile(compensated_gradient, 0.92)
ax[1].imshow(compensated_gradient, cmap="gray", vmin=-vmax, vmax=vmax)
ax[1].set_title("Compensated gradient")
vmax = torch.quantile(rtm_image.abs(), 0.98)
ax[2].imshow(rtm_image, cmap="gray", vmin=-vmax, vmax=vmax)
ax[2].set_title("Forward x adjoint image (compiled)")
plt.tight_layout()
plt.savefig("example_custom_imaging_condition.jpg")

assert torch.isfinite(compensated_gradient).all(), (
    "compensated gradient is not finite"
)
assert compensated_gradient.abs().max() > 0, (
    "compensated gradient is all zeros"
)
assert torch.isfinite(rtm_image).all(), "rtm image is not finite"
assert rtm_image.abs().max() > 0, "rtm image is all zeros"

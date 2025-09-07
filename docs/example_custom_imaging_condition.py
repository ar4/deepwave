"""This script demonstrates the use of callbacks to implement a
custom imaging condition (illumination compensation).
"""

import torch
from scipy.ndimage import gaussian_filter
import deepwave
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the model
ny = 200
nx = 200
v_true = torch.ones(ny, nx, device=device) * 1500
for interface_idx in range(1, 5):
    v_true[interface_idx * 40 :] += 50
dx = 5.0

n_shots = 20
n_sources_per_shot = 1
d_source = 10
first_source = 5
source_depth = 0

n_receivers_per_shot = nx
receiver_depth = 0

dt = 0.008
freq = 25
nt = 500
peak_time = 1.5 / freq

# Smooth to use as starting model
v = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 40)).to(device)
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
receiver_locations[:, :, 1] = (torch.arange(n_receivers_per_shot)).repeat(n_shots, 1)

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
torch.nn.MSELoss()(out[-1], out_true[-1]).backward()

standard_gradient = v.grad.detach().cpu().clone()
v.grad.zero_()
del out

# Calculate the gradient with illumination compensation
illumination = None


def forward_callback(state):
    global illumination
    if illumination is None:
        illumination = (
            state.get_wavefield("wavefield_0", view="full").detach() ** 2
        ).sum(dim=0)
    else:
        illumination += (
            state.get_wavefield("wavefield_0", view="full").detach() ** 2
        ).sum(dim=0)


def backward_callback(state):
    if state.step == 0:
        gradient = state.get_gradient("v", view="full")
        gradient /= illumination + 1e-3 * illumination.max()


out = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    forward_callback=forward_callback,
    backward_callback=backward_callback,
    callback_frequency=1,
)
torch.nn.MSELoss()(out[-1], out_true[-1]).backward()

compensated_gradient = v.grad.detach().cpu().clone()

# Plot
_, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
vmax = torch.quantile(standard_gradient, 0.92)
ax[0].imshow(standard_gradient, cmap="gray", vmin=-vmax, vmax=vmax)
ax[0].set_title("Standard gradient")
vmax = torch.quantile(compensated_gradient, 0.92)
ax[1].imshow(compensated_gradient, cmap="gray", vmin=-vmax, vmax=vmax)
ax[1].set_title("Compensated gradient")
plt.tight_layout()
plt.savefig("example_custom_imaging_condition.jpg")

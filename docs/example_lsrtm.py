"""Demonstrates Least-Squares Reverse-Time Migration (LSRTM) using Deepwave.

This script shows how to invert for the scattering potential with Born
modelling, including handling direct arrivals.
"""

import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter

import deepwave
from deepwave import scalar, scalar_born

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny_full = 2301
nx_full = 751
dx = 4.0
v_true_full = torch.from_file("marmousi_vp.bin", size=ny_full * nx_full)
v_true_full = v_true_full.reshape(ny_full, nx_full)

# Select portion of model for inversion
ny = 600
nx = 250
v_true = v_true_full[:ny, :nx]

# Smooth to use as starting model
v_mig = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 5)).to(device)

n_shots_full = 115

n_sources_per_shot = 1
d_source = 20  # 20 * 4m = 80m
first_source = 10  # 10 * 4m = 40m
source_depth = 2  # 2 * 4m = 8m

n_receivers_per_shot_full = 384
d_receiver = 6  # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 2  # 2 * 4m = 8m

freq = 25
nt_full = 750
dt = 0.004
peak_time = 1.5 / freq

observed_data_full = torch.from_file(
    "marmousi_data.bin",
    size=n_shots_full * n_receivers_per_shot_full * nt_full,
).reshape(n_shots_full, n_receivers_per_shot_full, nt_full)

# Select portion of data for inversion
n_shots = 20
n_receivers_per_shot = 100
nt = 300
observed_data = observed_data_full[
    :n_shots,
    :n_receivers_per_shot,
    :nt,
].to(device)

# source_locations
source_locations = torch.zeros(
    n_shots,
    n_sources_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

# receiver_locations
receiver_locations = torch.zeros(
    n_shots,
    n_receivers_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = (
    torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
).repeat(n_shots, 1)

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# Estimate direct arrival by forward modelling through
# migration velocity model, and then subtract it from
# the observed data. We set max_vel to be the maximum
# velocity in the true velocity model so that Deepwave's
# internal time step size will be the same as when the
# observed dataset was created, to get a better match.
out = scalar(
    v_mig,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_freq=freq,
    accuracy=8,
    max_vel=v_true.max(),
)
observed_scattered_data = observed_data - out[-1]

_, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
ax[0].imshow(observed_data[0].cpu().T, aspect="auto", cmap="gray")
ax[0].set_title("Observed")
ax[1].imshow(out[-1][0].cpu().T, aspect="auto", cmap="gray")
ax[1].set_title("Predicted")
ax[2].imshow(observed_scattered_data[0].cpu().T, aspect="auto", cmap="gray")
ax[2].set_title("Observed - Predicted")
plt.tight_layout()
plt.savefig("example_lsrtm_scattered.jpg")

# Create scattering amplitude that we will invert for
scatter = torch.zeros_like(v_mig)
scatter.requires_grad_()

# Setup optimiser to perform inversion
optimiser = torch.optim.LBFGS([scatter])
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 3

for _epoch in range(n_epochs):

    def closure():
        """Closure function for the LBFGS optimiser."""
        optimiser.zero_grad()
        out = scalar_born(
            v_mig,
            scatter,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
        )
        loss = 1e6 * loss_fn(out[-1], observed_scattered_data)
        loss.backward()
        return loss.item()

    optimiser.step(closure)

# Plot
vmin, vmax = torch.quantile(
    scatter.detach(), torch.tensor([0.05, 0.95]).to(device)
)
plt.figure(figsize=(10.5, 3.5))
plt.imshow(
    scatter.detach().cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
plt.savefig("example_lsrtm.jpg")

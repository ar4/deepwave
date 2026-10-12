"""Demonstrates Least-Squares Reverse-Time Migration (LSRTM) using Deepwave.

This script shows how to invert for the scattering potential with Born
modelling, including handling direct arrivals.
"""

import math
import os

import matplotlib.pyplot as plt
import torch
from get_marmousi import load_data, load_vp
from scipy.ndimage import gaussian_filter

import deepwave
from deepwave import scalar, scalar_born

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# Dimension 0 is depth and dimension 1 is horizontal position, the
# convention used throughout this documentation.
ny_full = 751
nx_full = 2301
dx = 4.0
# ``load_vp`` reads marmousi_vp.bin from the current directory and checks
# that it is in the depth-first layout Deepwave 0.1.0 writes (row 0 is the
# 1500 m/s water layer; an old-layout file holds the same number of floats
# and so would reshape without error, silently giving a rotated model). In
# smoke mode it returns a generated stand-in model of the same shape and
# velocity range instead, so no download is needed.
v_true_full = load_vp(smoke=smoke)

# Select portion of model for inversion
ny = 100 if smoke else 250
nx = 200 if smoke else 600
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

# Select portion of data for inversion
n_shots = 6 if smoke else 20
n_receivers_per_shot = 20 if smoke else 100
nt = 80 if smoke else 300
# ``load_data`` takes the first n_shots shots, n_receivers_per_shot
# receivers and nt time samples of marmousi_data.bin. In smoke mode there is
# no downloaded model for that file to have come from, so it instead
# generates the data by propagating through the model below.
observed_data = load_data(
    v_true, n_shots, n_receivers_per_shot, nt, smoke=smoke
).to(device)

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
receiver_locations[:, :, 1] = (
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
observed_scattered_data = observed_data - out.receiver_amplitudes

_, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
ax[0].imshow(observed_data[0].cpu().T, aspect="auto", cmap="gray")
ax[0].set_title("Observed")
ax[1].imshow(out.receiver_amplitudes[0].cpu().T, aspect="auto", cmap="gray")
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
n_epochs = 2 if smoke else 3

loss_history: list[float] = []

for _epoch in range(n_epochs):

    def closure():
        """Closure function for the LBFGS optimiser."""
        optimiser.zero_grad()
        out = scalar_born(
            v_mig,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            sc_receiver_locations=receiver_locations,
            pml_freq=freq,
        )
        loss = 1e6 * loss_fn(
            out.sc_receiver_amplitudes, observed_scattered_data
        )
        loss.backward()
        loss_history.append(float(loss.detach()))
        return loss.item()

    optimiser.step(closure)

# Plot
vmin, vmax = torch.quantile(
    scatter.detach(), torch.tensor([0.05, 0.95]).to(device)
)
plt.figure(figsize=(10.5, 3.5))
plt.imshow(
    scatter.detach().cpu(),
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
plt.savefig("example_lsrtm.jpg")

assert all(math.isfinite(loss) for loss in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

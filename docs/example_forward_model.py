"""Demonstrates forward modelling using Deepwave.

This script demonstrates forward modelling using Deepwave with the
Marmousi velocity model. It loads the Marmousi Vp model, sets up source
and receiver locations, propagates seismic waves, and generates synthetic
receiver data. Results are then plotted and saved to a file.
"""

import os

import matplotlib.pyplot as plt
import torch
from get_marmousi import load_vp

import deepwave
from deepwave import scalar

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
v_full = load_vp(smoke=smoke)
ny = 150 if smoke else ny_full
nx = 300 if smoke else nx_full
v = v_full[:ny, :nx].to(device)

n_shots = 4 if smoke else 115

n_sources_per_shot = 1
d_source = 20  # 20 * 4m = 80m
first_source = 10  # 10 * 4m = 40m
source_depth = 2  # 2 * 4m = 8m

n_receivers_per_shot = 40 if smoke else 384
d_receiver = 6  # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 2  # 2 * 4m = 8m

freq = 25
nt = 200 if smoke else 750
dt = 0.004
peak_time = 1.5 / freq

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
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# Propagate
out = scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    accuracy=8,
    pml_freq=freq,
)

# Plot
receiver_amplitudes = out.receiver_amplitudes
vmin, vmax = torch.quantile(
    receiver_amplitudes[0],
    torch.tensor([0.05, 0.95]).to(device),
)
mid_shot = n_shots // 2
mid_receiver = n_receivers_per_shot // 2
_, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
ax[0].imshow(
    receiver_amplitudes[mid_shot].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[1].imshow(
    receiver_amplitudes[:, mid_receiver].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[0].set_xlabel("Channel")
ax[0].set_ylabel("Time Sample")
ax[1].set_xlabel("Shot")
plt.tight_layout()
plt.savefig("example_forward_model.jpg")

receiver_amplitudes.cpu().numpy().tofile("marmousi_data.bin")

assert torch.isfinite(receiver_amplitudes).all(), (
    "receiver data is not finite"
)
assert receiver_amplitudes.abs().max() > 0, "receiver data is all zeros"

"""Demonstrates Reverse-Time Migration (RTM) using Deepwave.

RTM forms an image of subsurface reflectivity by cross-correlating a
forward-propagated source wavefield with a backward-propagated receiver
wavefield. Here that image is obtained as the gradient of a Born
data-misfit objective with respect to a scattering potential that starts
at zero: one forward propagation, one backpropagation, one optimiser step.
The script also shows how to use a tapered mute to attenuate direct
arrivals, and how to accumulate the gradient over batches of shots so that
the whole survey need not be in memory at once.
"""

import math
import os

import matplotlib.pyplot as plt
import torch
from get_marmousi import load_data, load_vp
from scipy.ndimage import gaussian_filter

import deepwave
from deepwave import scalar_born

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# Dimension 0 is depth and dimension 1 is horizontal position, the
# convention used throughout this documentation.
ny_full = 751
nx_full = 2301
dx_full = 4.0
# ``load_vp`` reads marmousi_vp.bin from the current directory and checks
# that it is in the depth-first layout Deepwave 0.1.0 writes (row 0 is the
# 1500 m/s water layer; an old-layout file holds the same number of floats
# and so would reshape without error, silently giving a rotated model). In
# smoke mode it returns a generated stand-in model of the same shape and
# velocity range instead, so no download is needed.
v = load_vp(smoke=smoke)
if smoke:
    v = v[:150, :300]

# Smooth to use as migration model
v_mig = torch.tensor(1 / gaussian_filter(1 / v.numpy(), 40))
v_mig = v_mig[::2, ::2].to(device)
ny = v_mig.shape[0]
nx = v_mig.shape[1]
dx = dx_full * 2

n_shots = 8 if smoke else 115

n_sources_per_shot = 1
d_source = 10  # 10 * 8m = 80m
first_source = 5  # 5 * 8m = 40m
source_depth = 1  # 1 * 8m = 8m

n_receivers_per_shot = 40 if smoke else 384
d_receiver = 3  # 3 * 8m = 24m
first_receiver = 0  # 0 * 8m = 0m
receiver_depth = 1  # 1 * 8m = 8m

freq = 25
nt = 150 if smoke else 750
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

# Load observed data. ``load_data`` takes the first n_shots shots,
# n_receivers_per_shot receivers and nt time samples of marmousi_data.bin.
# In smoke mode there is no downloaded model for that file to have come
# from, so it instead generates the data by propagating through ``v``. Note
# that the data is recorded on the original 4 m grid whatever the migration
# grid is, as real data would be: only the migration model is subsampled.
observed_data = load_data(
    v, n_shots, n_receivers_per_shot, nt, smoke=smoke
).to(device)

# Create mask to attenuate direct arrival
mask = torch.ones_like(observed_data)
flat_len = 100
taper_len = 200
taper = torch.cos(torch.arange(taper_len) / taper_len * math.pi / 2)
mute_len = flat_len + 2 * taper_len
mute = torch.zeros(mute_len, device=device)
mute[:taper_len] = taper
mute[-taper_len:] = taper.flip(0)
v_direct = 1700
for shot_idx in range(n_shots):
    sx = (shot_idx * d_source + first_source) * dx
    for receiver_idx in range(n_receivers_per_shot):
        rx = (receiver_idx * d_receiver + first_receiver) * dx
        dist = abs(sx - rx)
        arrival_time = dist / v_direct / dt
        mute_start = int(arrival_time) - mute_len // 2
        mute_end = mute_start + mute_len
        if mute_start > nt:
            continue
        actual_mute_start = max(mute_start, 0)
        actual_mute_end = min(mute_end, nt)
        mask[shot_idx, receiver_idx, actual_mute_start:actual_mute_end] = (
            mute[
                actual_mute_start - mute_start : actual_mute_end
                - mute_start
            ]
        )
observed_scatter_masked = observed_data * mask

vmin, vmax = torch.quantile(
    observed_data[0], torch.tensor([0.05, 0.95]).to(device)
)
_, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
ax[0].imshow(
    observed_data[0].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[0].set_title("Observed")
ax[1].imshow(mask[0].cpu().T, aspect="auto", cmap="gray")
ax[1].set_title("Mask")
ax[2].imshow(
    observed_scatter_masked[0].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[2].set_title("Masked data")
plt.tight_layout()
plt.savefig("example_rtm_mask.jpg")

# Create scattering amplitude that we will invert for
scatter = torch.zeros_like(v_mig)
scatter.requires_grad_()

# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([scatter], lr=1e9)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 1
n_batch = 8 if smoke else 46  # The number of batches to use
n_shots_per_batch = (n_shots + n_batch - 1) // n_batch
for _epoch in range(n_epochs):
    epoch_loss = 0
    optimiser.zero_grad()
    for batch in range(n_batch):
        batch_start = batch * n_shots_per_batch
        batch_end = min(batch_start + n_shots_per_batch, n_shots)
        if batch_end <= batch_start:
            continue
        s = slice(batch_start, batch_end)
        out = scalar_born(
            v_mig,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=source_amplitudes[s],
            source_locations=source_locations[s],
            sc_receiver_locations=receiver_locations[s],
            pml_freq=freq,
        )
        loss = (
            loss_fn(
                out.sc_receiver_amplitudes * mask[s],
                observed_scatter_masked[s],
            )
            * (batch_end - batch_start)
            / n_shots
        )
        # Weight each batch mean by its share of the survey. In particular,
        # the final, smaller batch must not overweight its shots.
        epoch_loss += loss.item()
        loss.backward()
    print(epoch_loss)
    optimiser.step()

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
plt.savefig("example_rtm.jpg")

scatter.detach().cpu().numpy().tofile("marmousi_scatter.bin")

assert torch.isfinite(scatter).all(), "scattering image is not finite"
assert scatter.abs().max() > 0, "scattering image is all zeros"

"""Demonstrates joint migration inversion using Deepwave.

Both the velocity model and the scattering potential are inverted
simultaneously.

*** NOTE: This example does not currently seem to be working correctly. ***
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.ndimage import gaussian_filter

import deepwave
from deepwave import scalar_born

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny_full = 2301
nx_full = 751
dx = 4.0
v_true_full = torch.from_file(
    "marmousi_vp.bin", size=ny_full * nx_full
).reshape(
    ny_full,
    nx_full,
)

# Select portion of model for inversion
ny = 600
nx = 250
v_true = v_true_full[:ny, :nx]

# Smooth to use as starting model
v_init = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 40)).to(
    device
)
v = v_init.clone()
v.requires_grad_()

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

# Setup optimiser to perform inversion
scatter = torch.zeros(ny, nx, device=device)
scatter.requires_grad_()
optimiser = torch.optim.LBFGS([v, scatter], lr=1)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 14


def closure():
    """Closure function for the LBFGS optimiser."""
    optimiser.zero_grad()
    # Remove high wavenumbers from the velocity model
    v_smooth = torchvision.transforms.functional.gaussian_blur(
        v[None],
        [11, 11],
    ).squeeze()
    # Remove low wavenumbers from the scattering model
    scatter_sharp = scatter - (
        torchvision.transforms.functional.gaussian_blur(
            scatter[None],
            [11, 11],
        ).squeeze()
    )
    out = scalar_born(
        v_smooth,
        scatter_sharp,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        bg_receiver_locations=receiver_locations,
        pml_freq=freq,
    )
    loss = 1e10 * loss_fn(out[-1] + out[-2], observed_data)
    loss.backward()
    print(loss.detach().item())
    return loss.item()


for _epoch in range(n_epochs):
    print(_epoch)
    optimiser.step(closure)

v_smooth = torchvision.transforms.functional.gaussian_blur(
    v.detach().cpu()[None],
    [11, 11],
).squeeze()
scatter_sharp = scatter.detach().cpu() - (
    torchvision.transforms.functional.gaussian_blur(
        scatter.detach().cpu()[None],
        [11, 11],
    ).squeeze()
)

# Plot
vmin = v_true.min()
vmax = v_true.max()
smin, smax = torch.quantile(scatter_sharp, torch.tensor([0.02, 0.98]))
_, ax = plt.subplots(4, figsize=(10.5, 12.5), sharex=True, sharey=True)
ax[0].imshow(
    v_init.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[0].set_title("Initial")
ax[1].imshow(v_smooth.T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
ax[1].set_title("Out velocity")
ax[2].imshow(
    scatter_sharp.T, aspect="auto", cmap="gray", vmin=smin, vmax=smax
)
ax[2].set_title("Out scatter")
ax[3].imshow(
    v_true.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[3].set_title("True")
plt.tight_layout()
plt.savefig("example_joint_migration_inversion.jpg")

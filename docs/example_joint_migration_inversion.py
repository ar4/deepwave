"""Demonstrates joint migration inversion using Deepwave.

Both the velocity model and the scattering potential are inverted
simultaneously.

Fitting both models with a single optimiser is delicate: the
scattering potential's gradient is typically much larger than the
velocity model's, so a naive joint update lets the scattering model
absorb the data residual while the velocity model barely moves. We
avoid that by giving the two models their own optimisers (with their
own learning rates), and by running a short "migration" warmup, in
which only the scattering model is updated (as in the
:doc:`LSRTM example <example_lsrtm>`), before switching to joint
updates. See :doc:`example_joint_migration_inversion` for more
discussion of why this is needed.
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torchvision
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

# Create scattering amplitude that we will invert for
scatter = torch.zeros(ny, nx, device=device)
scatter.requires_grad_()

# Setup optimisers to perform inversion. The scattering potential and
# the velocity model have very different gradient magnitudes and
# units, so each gets its own Adam optimiser and learning rate rather
# than sharing a single LBFGS optimiser. The velocity learning rate is
# in m/s and is decayed during the joint stage so that early, large
# corrections to the (very smooth) starting model don't overshoot once
# the model is closer to the truth.
v_lr = 20.0
scatter_lr = 2e-2
v_lr_decay = 0.85
optimiser_v = torch.optim.Adam([v], lr=v_lr)
optimiser_s = torch.optim.Adam([scatter], lr=scatter_lr)
loss_fn = torch.nn.MSELoss()

# We normalise the loss by the observed data's energy. This keeps the
# loss magnitude, and thus the raw gradient magnitudes, independent of
# the (somewhat arbitrary) source amplitude and receiver spacing used,
# so the learning rates above don't need to be re-tuned if those
# change.
observed_data_energy = observed_data.pow(2).mean()

# Run optimisation/inversion. We begin with a short "migration"
# warmup, updating only the scattering model with the velocity model
# frozen (the same idea as the LSRTM example), so that the scattering
# model has already captured most of the reflectivity before the
# velocity model starts to move. We then switch to joint updates of
# both models, with the velocity learning rate decaying over time.
n_warmup_epochs = 1 if smoke else 3
n_joint_epochs = 2 if smoke else 11
n_epochs = n_warmup_epochs + n_joint_epochs


def closure(*, update_v):
    """Closure function for the optimisers.

    Computes the data misfit loss and its gradients with respect to
    the velocity model (only when `update_v` is True) and the
    scattering model.
    """
    optimiser_v.zero_grad()
    optimiser_s.zero_grad()
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
        dx,
        dt,
        dv=scatter_sharp,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        bg_receiver_locations=receiver_locations,
        pml_freq=freq,
    )
    loss = (
        loss_fn(
            out.sc_receiver_amplitudes + out.bg_receiver_amplitudes,
            observed_data,
        )
        / observed_data_energy
    )
    loss.backward()
    if update_v:
        optimiser_v.step()
    optimiser_s.step()
    return loss


loss_history: list[float] = []
for epoch in range(n_epochs):
    warmup = epoch < n_warmup_epochs
    if not warmup:
        # Decay the velocity learning rate during the joint stage
        decay_step = epoch - n_warmup_epochs
        for group in optimiser_v.param_groups:
            group["lr"] = v_lr * (v_lr_decay**decay_step)
    loss = closure(update_v=not warmup)
    print(epoch, loss.item())
    loss_history.append(float(loss.detach()))

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
ax[0].imshow(v_init.cpu(), aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v_smooth, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
ax[1].set_title("Out velocity")
ax[2].imshow(
    scatter_sharp, aspect="auto", cmap="gray", vmin=smin, vmax=smax
)
ax[2].set_title("Out scatter")
ax[3].imshow(v_true.cpu(), aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
ax[3].set_title("True")
plt.tight_layout()
plt.savefig("example_joint_migration_inversion.jpg")

assert all(math.isfinite(loss) for loss in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

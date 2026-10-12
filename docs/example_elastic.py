# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# These examples use the Deepwave 0.1 interface and require PyTorch 2.4
# or later, as declared by Deepwave. Install plotting support too.
# %pip install "deepwave>=0.1.0" matplotlib

# %% [markdown]
# # Elastic wave propagation and FWI
#
# This example shows how to set up elastic models, generate synthetic
# data, and perform inversion for vp, vs, and rho parameters.
#
# By default this runs the full inversion used to produce the docs page
# (20 LBFGS epochs). Set the environment variable
# `DEEPWAVE_EXAMPLE_FAST=1` (e.g. `%env DEEPWAVE_EXAMPLE_FAST=1` in the
# cell below, before running the rest of the notebook) to shrink the
# optimisation to a quick smoke test -- useful for CI or for confirming
# the notebook runs before committing to the full, slower inversion.

# %%
"""Demonstrates elastic wave propagation and Full-Waveform Inversion."""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import elastic

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# Set DEEPWAVE_EXAMPLE_FAST=1 to shrink the inversion below to a quick
# (non-representative) smoke test instead of the full 20-epoch LBFGS run.
# ``DEEPWAVE_SMOKE=1`` implies the same shrinkage.
_FAST = smoke or os.environ.get("DEEPWAVE_EXAMPLE_FAST", "0") not in (
    "0",
    "",
    "false",
    "False",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny = 30
nx = 100
dx = 4.0

vp_background = torch.ones(ny, nx, device=device) * 1500
vs_background = torch.ones(ny, nx, device=device) * 1000
rho_background = torch.ones(ny, nx, device=device) * 2200

vp_true = vp_background.clone()
vp_true[10:20, 30:40] = 1600
vs_true = vs_background.clone()
vs_true[10:20, 45:55] = 1100
rho_true = rho_background.clone()
rho_true[10:20, 60:70] = 2300

n_shots = 3 if smoke else 8

n_sources_per_shot = 1
d_source = 12
first_source = 8
source_depth = 2

n_receivers_per_shot = nx - 1
d_receiver = 1
first_receiver = 0
receiver_depth = 2

freq = 15
nt = 120 if smoke else 200
dt = 0.004
peak_time = 1.5 / freq

# %% [markdown]
# ## Acquisition geometry and source wavelet

# %%
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

# %% [markdown]
# ## Synthesise observed data from the true model

# %%
# Create observed data using true models
observed_data = elastic(
    *deepwave.vpvsrho_to_lambmubuoyancy(vp_true, vs_true, rho_true),
    dx,
    dt,
    source_amplitudes_y=source_amplitudes,
    source_locations_y=source_locations,
    receiver_locations_y=receiver_locations,
    pml_freq=freq,
).receiver_amplitudes_y

# %% [markdown]
# ## Invert for vp, vs, and rho
#
# `n_epochs` and the LBFGS `max_iter` are both shrunk in fast mode: each
# outer epoch calls the closure up to `max_iter` times, so the two
# multiply -- the default (20 epochs x 20 max_iter) is 400 forward+
# backward elastic propagations, which is what makes the full run slow.

# %%
# Setup optimiser to perform inversion
vp = vp_background.clone().requires_grad_()
vs = vs_background.clone().requires_grad_()
rho = rho_background.clone().requires_grad_()
lbfgs_max_iter = 3 if _FAST else 20
optimiser = torch.optim.LBFGS([vp, vs, rho], max_iter=lbfgs_max_iter)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 2 if _FAST else 20

loss_history = []
for _epoch in range(n_epochs):

    def closure():
        """Closure function for the LBFGS optimiser."""
        optimiser.zero_grad()
        out = elastic(
            *deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            dx,
            dt,
            source_amplitudes_y=source_amplitudes,
            source_locations_y=source_locations,
            receiver_locations_y=receiver_locations,
            pml_freq=freq,
        ).receiver_amplitudes_y
        loss = 1e20 * loss_fn(out, observed_data)
        loss.backward()
        loss_history.append(loss.item())
        return loss.item()

    optimiser.step(closure)

# %% [markdown]
# ## Plot true vs. recovered models

# %%
# Plot
vpmin = vp_true.min()
vpmax = vp_true.max()
vsmin = vs_true.min()
vsmax = vs_true.max()
rhomin = rho_true.min()
rhomax = rho_true.max()
_, ax = plt.subplots(2, 3, figsize=(10.5, 5.25), sharex=True, sharey=True)
ax[0, 0].imshow(
    vp_true.cpu(), aspect="auto", cmap="gray", vmin=vpmin, vmax=vpmax
)
ax[0, 0].set_title("True vp")
ax[0, 1].imshow(
    vs_true.cpu(), aspect="auto", cmap="gray", vmin=vsmin, vmax=vsmax
)
ax[0, 1].set_title("True vs")
ax[0, 2].imshow(
    rho_true.cpu(), aspect="auto", cmap="gray", vmin=rhomin, vmax=rhomax
)
ax[0, 2].set_title("True rho")
ax[1, 0].imshow(
    vp.detach().cpu(), aspect="auto", cmap="gray", vmin=vpmin, vmax=vpmax
)
ax[1, 0].set_title("Out vp")
ax[1, 1].imshow(
    vs.detach().cpu(), aspect="auto", cmap="gray", vmin=vsmin, vmax=vsmax
)
ax[1, 1].set_title("Out vs")
ax[1, 2].imshow(
    rho.detach().cpu(),
    aspect="auto",
    cmap="gray",
    vmin=rhomin,
    vmax=rhomax,
)
ax[1, 2].set_title("Out rho")
plt.tight_layout()
plt.savefig("example_elastic.jpg")
plt.show()

# %%
assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

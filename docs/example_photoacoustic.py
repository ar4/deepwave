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
# # Photoacoustic initial-pressure reconstruction with Deepwave
#
# Photoacoustic tomography images an initial pressure distribution ``p0``
# (created when a short laser pulse is absorbed by, e.g., blood vessels)
# from the acoustic waves it launches, recorded on a surrounding array.
# This is an initial-value problem: at ``t = 0`` the pressure equals ``p0``
# and its time derivative is zero. Because Deepwave is differentiable, the
# reconstruction is just gradient descent on the initial wavefield.
#
# We (1) forward propagate a small blood-vessel-like ``p0`` phantom with
# the scalar propagator to synthesise the data recorded on a ring of
# receivers, then (2) reconstruct ``p0`` from that data by making the
# initial wavefield a ``requires_grad`` parameter and minimising the data
# misfit.
#
# Leapfrog two-level convention: the scalar propagator advances a field
# given its values at steps 0 (``wavefield_0``) and -1 (``wavefield_m1``).
# Setting both levels to ``p0`` gives zero discrete velocity at the
# intervening half-step. This approximates rest at t = 0; the example's
# documentation explains the time-centering error and a second-order start.

# %%
"""Photoacoustic initial-pressure reconstruction with Deepwave."""

import math
import os

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import conv2d, mse_loss, pad

from deepwave import scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# %% [markdown]
# ## Medium and phantom
#
# A homogeneous soft-tissue/water medium, and a blood-vessel-like
# initial-pressure phantom that is blurred so its features are
# band-limited (at least ~6 cells per wavelength, avoiding grid
# dispersion), which also keeps the linear reconstruction well
# conditioned.

# %%
# Medium: homogeneous soft tissue / water, c = 1500 m/s.
ny, nx = (60, 60) if smoke else (100, 100)
dx = 0.5e-3  # 0.5 mm grid cells
c = 1500.0
v = torch.full((ny, nx), c, device=device)

pml_width = 10 if smoke else 20
dt = 5e-8  # 50 ns => 20 MHz sampling; comfortably above CFL here
nt = 320 if smoke else 520

# Blood-vessel-like phantom: a few bright "vessels" (lines and a blob). The
# feature positions and sizes are fractions of ny/nx (equal to the default
# 100-cell-grid pixel coordinates when ny == nx == 100), so they stay
# proportioned to the grid when it shrinks. We blur it so its features span
# several cells: this keeps the initial pressure band-limited (at least ~6
# cells per wavelength, avoiding grid dispersion) and keeps the linear
# reconstruction well-conditioned.
p0_sharp = torch.zeros(ny, nx, device=device)
p0_sharp[
    int(0.40 * ny) : int(0.43 * ny), int(0.25 * nx) : int(0.70 * nx)
] = 1.0  # horizontal vessel
p0_sharp[
    int(0.30 * ny) : int(0.65 * ny), int(0.55 * nx) : int(0.57 * nx)
] = 0.8  # vertical vessel
yy, xx = torch.meshgrid(
    torch.arange(ny, device=device),
    torch.arange(nx, device=device),
    indexing="ij",
)
p0_sharp[
    ((yy - 0.62 * ny) ** 2 + (xx - 0.38 * nx) ** 2) < (0.06 * ny) ** 2
] = 0.6

_blur = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
_blur = _blur / _blur.sum()


def blur(a):
    """Separable binomial blur to band-limit the phantom."""
    a = conv2d(a[None, None], _blur.view(1, 1, 5, 1), padding=(2, 0))
    a = conv2d(a, _blur.view(1, 1, 1, 5), padding=(0, 2))
    return a[0, 0]


p0_true = blur(blur(p0_sharp))

# %% [markdown]
# ## Acquisition and forward model
#
# A ring of receivers surrounds the phantom. Float locations are
# automatically Hicks-interpolated onto the grid by the propagator.

# %%
# Receiver ring surrounding the phantom. Float locations are automatically
# Hicks-interpolated onto the grid by the propagator.
n_receivers = 24 if smoke else 64
ring_radius = 0.4 * ny
angles = torch.arange(n_receivers) * (2 * torch.pi / n_receivers)
receiver_locations = torch.stack(
    [
        ny / 2 + ring_radius * torch.cos(angles),
        nx / 2 + ring_radius * torch.sin(angles),
    ],
    dim=1,
).to(device)
receiver_locations = receiver_locations.unsqueeze(0)  # [1, n_rec, 2]


def pad_to_sim(p0_inner):
    """Embed [ny, nx] into the [1, ny+2*pml, nx+2*pml] sim domain."""
    padded = pad(p0_inner, (pml_width, pml_width, pml_width, pml_width))
    return padded.unsqueeze(0)


def forward(p0_inner):
    """Propagate an initial pressure and return the receiver data."""
    wavefield = pad_to_sim(p0_inner)
    out = scalar(
        v,
        dx,
        dt,
        receiver_locations=receiver_locations,
        wavefield_0=wavefield,
        wavefield_m1=wavefield,  # zero half-step velocity; see notes above
        nt=nt,
        pml_width=pml_width,
        pml_freq=1e6,  # ~1 MHz content; pinned (no sources to infer from)
        accuracy=4,
    )
    return out.receiver_amplitudes  # [1, n_receivers, nt]


# %% [markdown]
# ## Synthesise data, then reconstruct ``p0`` by gradient descent
#
# The map ``p0 -> receiver data`` is linear, so LBFGS converges rapidly
# here.

# %%
# --- 1. Synthesise the observed data ---------------------------------
observed = forward(p0_true).detach()

# --- 2. Reconstruct p0 by gradient descent on the initial wavefield ---
# The map p0 -> receiver data is linear, so LBFGS converges rapidly here.
p0_est = torch.zeros(ny, nx, device=device, requires_grad=True)
optimiser = torch.optim.LBFGS(
    [p0_est],
    lr=1.0,
    max_iter=10 if smoke else 40,
    history_size=10 if smoke else 20,
    line_search_fn="strong_wolfe",
)
loss_history = []


def closure():
    """One LBFGS evaluation: forward model, misfit, and its gradient."""
    optimiser.zero_grad()
    loss = mse_loss(forward(p0_est), observed)
    loss.backward()
    loss_history.append(loss.item())
    return loss


optimiser.step(closure)
print(f"iterations: {len(loss_history)}  final loss {loss_history[-1]:.3e}")

p0_rec = p0_est.detach()
rel_err = (p0_rec - p0_true).norm() / p0_true.norm()
print(f"final relative reconstruction error: {rel_err.item():.3f}")

# %% [markdown]
# ## Plot

# %%
vmax = float(p0_true.max())
_, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(p0_true.cpu(), cmap="hot", vmin=0, vmax=vmax)
ax[0].set_title("True initial pressure $p_0$")
ax[1].imshow(p0_rec.cpu(), cmap="hot", vmin=0, vmax=vmax)
ax[1].set_title(f"Reconstruction (rel. err {rel_err.item():.2f})")
ax[2].semilogy(loss_history)
ax[2].set_title("Data misfit")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("MSE loss")
for a in ax[:2]:
    a.set_xlabel("x (cell)")
    a.set_ylabel("y (cell)")
plt.tight_layout()
plt.savefig("example_photoacoustic.jpg")
plt.show()

# %%
assert all(math.isfinite(loss) for loss in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], (
    "reconstruction loss did not decrease"
)

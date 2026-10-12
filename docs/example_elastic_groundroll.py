"""Demonstrates elastic wave propagation with a focus on groundroll."""

import os

import matplotlib.pyplot as plt
import torch

import deepwave

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny = 60 if smoke else 100
nx = 120 if smoke else 300
vp = 1500 * torch.ones(ny, nx)
vs = 1000 * torch.ones(ny, nx)
rho = 2200 * torch.ones(ny, nx)

x_r_y = torch.zeros(1, nx, 2, dtype=torch.long)
x_r_y[0, :, 1] = torch.arange(nx)
x_r_x = torch.zeros(1, nx - 1, 2, dtype=torch.long)
x_r_x[0, :, 1] = torch.arange(nx - 1)
x_r_x[0, :, 0] = 1

vp[0] = 0
vs[0] = 0
rho[0] = 0
out = deepwave.elastic(
    *deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
    grid_spacing=2,
    dt=0.004,
    source_amplitudes_y=deepwave.wavelets.ricker(
        25, 50, 0.004, 0.06, shape=(1, 1)
    ),
    source_locations_y=torch.tensor([[[0, nx // 2]]]),
    receiver_locations_y=x_r_y,
    receiver_locations_x=x_r_x,
    pml_width=[0, 20, 20, 20],
)

_, ax = plt.subplots(4, figsize=(3.5, 5), sharey=True, sharex=True)
ax[0].imshow(out.vy_0[0, :-20, 20:-20], cmap="gray")
ax[0].set_title("Wavefield y")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(out.vx_0[0, :-20, 20:-20], cmap="gray")
ax[1].set_title("Wavefield x")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].imshow(
    out.receiver_amplitudes_y[0].T,
    cmap="gray",
    aspect="auto",
    vmin=-1e-8,
    vmax=1e-8,
)
ax[2].set_title("Data (y)")
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[3].imshow(
    out.receiver_amplitudes_x[0].T,
    cmap="gray",
    aspect="auto",
    vmin=-1e-8,
    vmax=1e-8,
)
ax[3].set_title("Data (x)")
ax[3].set_xticks([])
ax[3].set_yticks([])
plt.tight_layout()
plt.savefig("example_elastic_groundroll.jpg")

assert torch.isfinite(out.receiver_amplitudes_y).all(), (
    "y receiver data is not finite"
)
assert out.receiver_amplitudes_y.abs().max() > 0, (
    "y receiver data is all zeros"
)
assert torch.isfinite(out.receiver_amplitudes_x).all(), (
    "x receiver data is not finite"
)
assert out.receiver_amplitudes_x.abs().max() > 0, (
    "x receiver data is all zeros"
)

"""Demonstrates elastic free-surface topography with surface_method="pm".

Deepwave represents a free surface (flat or arbitrarily rugged/staircase
topography) purely through vacuum cells (``vp = vs = rho = 0``) -- there is
no separate surface-geometry argument. ``surface_method="pm"`` selects the
parameter-modified method of Cao and Chen (2018), which gives markedly
smaller surface-wave error than the default improved vacuum method (IVM)
at large Poisson's ratio, at no extra per-step cost. See :doc:`elastic`
and :doc:`boundaries` for the derivation.

This example builds a small 2-D model with a hill-and-valley topographic
surface (vacuum above it), places a shallow vertical-force source on the
surface, and shows the resulting surface wave both as a wavefield
snapshot (visibly hugging the terrain, distinct from the body waves
radiating into the half-space) and as a receiver gather along the surface
(a clear moveout trend).
"""

import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import elastic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny, nx = 60, 140
dx = 8.0
freq = 15.0
dt = 0.001
nt = 150 if smoke else 420
peak_time = 1.5 / freq

# A hill-and-valley topographic surface (in grid cells from the top).
xx = torch.arange(nx, device=device).float()
surface = (
    20
    + 12 * torch.exp(-((xx - 95) ** 2) / (2 * 15**2))
    - 6 * torch.exp(-((xx - 40) ** 2) / (2 * 10**2))
)
surface = surface.round().long().clamp(2, ny - 8)

vp = torch.full((ny, nx), 1800.0, device=device)
vs = torch.full((ny, nx), 900.0, device=device)
rho = torch.full((ny, nx), 2000.0, device=device)
yy = torch.arange(ny, device=device).reshape(-1, 1)
vacuum = yy < surface.reshape(1, -1)  # everything above the surface line
vp[vacuum] = 0.0
vs[vacuum] = 0.0
rho[vacuum] = 0.0
lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

source_x = 20
source_y = surface[source_x]
source_locations = torch.tensor([[[source_y, source_x]]], device=device)
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .to(device)
)

# Receivers spread along the surface, following the topography.
receiver_x = torch.arange(25, 135, 5, device=device)
receiver_y = surface[receiver_x]
receiver_locations = torch.stack(
    [receiver_y, receiver_x], dim=-1
).unsqueeze(0)

out = elastic(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    source_amplitudes_y=source_amplitudes,
    source_locations_y=source_locations,
    receiver_locations_y=receiver_locations,
    surface_method="pm",
    backend="eager",
    pml_freq=freq,
    max_vel=float(vp.max()) + 1.0,
)
wavefield = out.vy_0[0].detach().cpu()
gather = out.receiver_amplitudes_y[0].detach().cpu()

_, ax = plt.subplots(1, 2, figsize=(11, 4))
vmax = float(wavefield.abs().max()) * 0.15
ax[0].imshow(wavefield, cmap="seismic", vmin=-vmax, vmax=vmax)
ax[0].plot(xx.cpu(), surface.cpu(), color="k", lw=1)
ax[0].set_ylim(60, 0)
ax[0].set_title("Wavefield (vy): surface wave hugging the terrain")

rmax = float(gather.abs().max())
ax[1].imshow(
    gather.T, cmap="gray", aspect="auto", vmin=-0.3 * rmax, vmax=0.3 * rmax
)
ax[1].set_xlabel("Receiver index (along the surface)")
ax[1].set_ylabel("Time step")
ax[1].set_title("Receiver gather along the surface")

plt.tight_layout()
plt.savefig("example_topography.jpg")

assert torch.isfinite(gather).all(), "receiver gather is not finite"
assert gather.abs().max() > 0, "receiver gather is all zeros"

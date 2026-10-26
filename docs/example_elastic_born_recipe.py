"""Recipe: deepwave.elastic_born background/scattered fields.

A minimal, self-contained companion to the :doc:`elastic_born` reference
page: builds a background elastic model from vp/vs/rho with
:func:`deepwave.vpvsrho_to_lambmubuoyancy`, defines a localised mu
(shear-modulus) scattering perturbation, and runs
:func:`deepwave.elastic_born` to compare the background and scattered
receiver gathers it produces in a single coupled call. It also demonstrates
that the scattered output is differentiable
with respect to the scattering perturbation, the basis of elastic LSRTM/FWI
gradients (see :doc:`example_acoustic_born` for a full LSRTM inversion loop
in the simpler variable-density acoustic case).
"""

import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import elastic_born

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

ny, nx = 50, 60
dx = 5.0
freq = 25.0
dt = 0.0007
nt = 250
peak_time = 1.5 / freq

vp = 1500 * torch.ones(ny, nx, device=device)
vs = 800 * torch.ones(ny, nx, device=device)
rho = 2000 * torch.ones(ny, nx, device=device)
lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

# A localised mu (shear-modulus) scatterer.
yy, xx = torch.meshgrid(
    torch.arange(ny, device=device),
    torch.arange(nx, device=device),
    indexing="ij",
)
dmu = (
    0.3 * mu * torch.exp(-((yy - 25) ** 2 + (xx - 30) ** 2) / 8.0)
).requires_grad_()

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

n_shots = 2
n_receivers_per_shot = 10 if smoke else 30
source_locations = torch.zeros(
    n_shots, 1, 2, dtype=torch.long, device=device
)
source_locations[:, 0, 0] = 3
source_locations[:, 0, 1] = torch.linspace(10, nx - 10, n_shots).long()
receiver_locations = torch.zeros(
    n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
)
receiver_locations[..., 0] = ny - 3
receiver_locations[:, :, 1] = torch.linspace(
    2, nx - 2, n_receivers_per_shot
).long()
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_shots, 1, 1)
    .to(device)
)

# mu + dmu locally raises the P-velocity above vp.max(); derive max_vel
# from the perturbed model so the internal time step is safely CFL-stable.
vp_perturbed, _, _ = deepwave.lambmubuoyancy_to_vpvsrho(
    lamb, mu + dmu.detach(), buoyancy
)
max_vel = float(vp_perturbed.max())

out = elastic_born(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    dmu=dmu,
    # The canonical, channel-keyed way to specify a survey. ``receivers=``
    # is the scattered data and ``bg_receivers=`` the background data, the
    # same pairing the returned fields use. The equivalent flat keywords
    # (``source_amplitudes_y=``, ``sc_receiver_locations_y=``,
    # ``bg_receiver_locations_y=``) remain permanently supported.
    sources={"y": (source_amplitudes, source_locations)},
    receivers={"y": receiver_locations},
    bg_receivers={"y": receiver_locations},
    pml_freq=freq,
    max_vel=max_vel,
)

# The scattered field is differentiable with respect to the scattering
# perturbation -- the basis of elastic LSRTM/FWI gradients.
loss = out.receivers["y"].pow(2).sum()
loss.backward()
if dmu.grad is None or not torch.isfinite(dmu.grad).all():
    raise RuntimeError("dmu.grad should be finite after backward().")
print(
    "scattered-data loss:",
    float(loss.detach()),
    "| max |d(loss)/d(dmu)|:",
    float(dmu.grad.abs().max()),
)

_, ax = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True, sharey=True)
bg = out.bg_receivers["y"][0].detach().cpu()
sc = out.receivers["y"][0].detach().cpu()
# Each field is shown with its own colour scale (independently normalised):
# the scattered field is orders of magnitude weaker than the background
# field it rides on top of (a single small mu perturbation), so a shared
# scale would make it invisible.
bg_vmax = float(bg.abs().max())
sc_vmax = float(sc.abs().max())
ax[0].imshow(bg.T, aspect="auto", cmap="gray", vmin=-bg_vmax, vmax=bg_vmax)
ax[0].set_title("Background field (vy)")
ax[1].imshow(sc.T, aspect="auto", cmap="gray", vmin=-sc_vmax, vmax=sc_vmax)
ax[1].set_title("Scattered field (vy), independent scale")
plt.tight_layout()
plt.savefig("example_elastic_born_recipe.jpg")

assert torch.isfinite(sc).all(), "scattered receiver data is not finite"
assert sc.abs().max() > 0, "scattered receiver data is all zeros"

"""Demonstrates Thomsen-space FWI with deepwave.acoustic_vti.

:func:`deepwave.acoustic_vti` takes VTI stiffnesses (``c11``, ``c13``,
``c33``) directly, but practitioners usually reason and invert in Thomsen
(1986) parameters (``vp``, ``epsilon``, ``delta``) instead. This is exactly
the "converters" case in :ref:`model-parameterisations`:
:func:`deepwave.anisotropy.thomsen_to_cij` is an ordinary
differentiable function, so optimising ``vp``/``epsilon`` directly and
rebuilding the stiffnesses each iteration composes with autograd for free,
with no propagator-side support required.

The pseudo-acoustic VTI system additionally requires ``epsilon >= delta``
everywhere (else it is ill-posed); rather than clamping ``epsilon`` (which
would zero its gradient wherever the clamp is active), this example
reparameterises ``epsilon = delta + softplus(epsilon_free)``, which is
guaranteed ``>= delta`` for *any* real ``epsilon_free`` while staying
smoothly differentiable -- the same chain-rule pattern again, this time
enforcing a physical constraint rather than converting units.

A small crosswell-style survey inverts for both ``vp`` and ``epsilon``
jointly from a uniform starting guess, using the masked-subregion pattern
from :ref:`model-parameterisations` to keep the update within a central
region away from the sources/PML (as in :doc:`example_viscoacoustic`, this
small survey's limited illumination near the edges produces artefacts
there if left unmasked). Even so, ``epsilon`` recovers less completely
than ``vp`` -- a real, well-known trade-off in VTI FWI: anisotropy
parameters need wider propagation-angle diversity than a narrow-aperture
survey like this one provides, while ``vp`` is comparatively well
constrained even at near-vertical incidence.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import acoustic_vti
from deepwave.anisotropy import thomsen_to_cij

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny, nx = 40, 60
dx = 10.0
freq = 20.0
dt = 0.001
nt = 300
peak_time = 1.5 / freq

vp_true = torch.full((ny, nx), 2000.0, device=device)
vp_true[15:25, 25:35] += 200.0
rho = torch.full((ny, nx), 2000.0, device=device)
vs = torch.zeros((ny, nx), device=device)
delta = torch.full((ny, nx), 0.05, device=device)
eps_true = torch.full((ny, nx), 0.10, device=device)
eps_true[15:25, 25:35] = 0.22

n_shots = 3
n_receivers_per_shot = 20
source_locations = torch.zeros(
    n_shots, 1, 2, dtype=torch.long, device=device
)
source_locations[:, 0, 0] = 3
source_locations[:, 0, 1] = torch.linspace(10, nx - 10, n_shots).long()
receiver_locations = torch.zeros(
    n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
)
receiver_locations[:, :, 0] = ny - 3
receiver_locations[:, :, 1] = torch.linspace(
    5, nx - 5, n_receivers_per_shot
).long()
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_shots, 1, 1)
    .to(device)
)
# A generous fixed max_vel: the qP velocity along the axis is
# vp * sqrt(1 + 2 * epsilon), and vp/epsilon both grow during inversion.
max_vel = 1.3 * float((vp_true * torch.sqrt(1 + 2 * eps_true)).max())


def run(vp: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """Build stiffnesses from Thomsen parameters and forward model."""
    c11, c13, c33, _, _, buoyancy = thomsen_to_cij(
        vp, vs, rho, epsilon, delta
    )
    return acoustic_vti(
        c11=c11,
        c13=c13,
        c33=c33,
        buoyancy=buoyancy,
        grid_spacing=dx,
        dt=dt,
        source_amplitudes_sigma_v=source_amplitudes,
        source_locations_sigma_v=source_locations,
        receiver_locations_sigma_v=receiver_locations,
        pml_freq=freq,
        max_vel=max_vel,
    ).receiver_amplitudes_sigma_v


observed = run(vp_true, eps_true).detach()

yy, xx = torch.meshgrid(
    torch.arange(ny, device=device),
    torch.arange(nx, device=device),
    indexing="ij",
)
invert_mask = (yy > 8) & (xx > 8) & (xx < nx - 8)

vp_background = 2000.0
# epsilon_free=0 -> softplus(0)=log(2) -> epsilon = delta + log(2) != 0.10,
# so invert softplus to start exactly at the background epsilon=0.10.
eps_free_background = float(
    torch.log(torch.expm1(torch.tensor(0.10 - 0.05)))
)
vp_update = torch.zeros((ny, nx), device=device, requires_grad=True)
eps_free_update = torch.zeros((ny, nx), device=device, requires_grad=True)
optimiser = torch.optim.Adam(
    [
        {"params": [vp_update], "lr": 5.0},
        {"params": [eps_free_update], "lr": 0.05},
    ]
)
loss_fn = torch.nn.MSELoss()
zeros = torch.zeros((ny, nx), device=device)

n_epochs = 20 if smoke else 120
loss_history: list[float] = []
for _epoch in range(n_epochs):
    optimiser.zero_grad()
    vp = vp_background + torch.where(invert_mask, vp_update, zeros)
    eps_free = eps_free_background + torch.where(
        invert_mask, eps_free_update, zeros
    )
    epsilon = delta + torch.nn.functional.softplus(eps_free)
    loss = 1e10 * loss_fn(run(vp, epsilon), observed)
    loss.backward()
    loss_history.append(float(loss.item()))
    optimiser.step()

with torch.no_grad():
    vp_final = vp_background + torch.where(invert_mask, vp_update, zeros)
    eps_free_final = eps_free_background + torch.where(
        invert_mask, eps_free_update, zeros
    )
    eps_final = delta + torch.nn.functional.softplus(eps_free_final)

_, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
ax[0, 0].imshow(vp_true.cpu(), vmin=2000, vmax=2200)
ax[0, 0].set_title("True vp")
ax[0, 1].imshow(vp_final.cpu(), vmin=2000, vmax=2200)
ax[0, 1].set_title("Inverted vp")
ax[1, 0].imshow(eps_true.cpu(), vmin=0.08, vmax=0.24)
ax[1, 0].set_title("True epsilon")
ax[1, 1].imshow(eps_final.cpu(), vmin=0.08, vmax=0.24)
ax[1, 1].set_title("Inverted epsilon")
plt.tight_layout()
plt.savefig("example_vti.jpg")

assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

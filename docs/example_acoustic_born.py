"""Demonstrates LSRTM with the variable-density acoustic Born propagator.

This is the :func:`deepwave.acoustic_born` analogue of
:doc:`example_lsrtm`, which uses :func:`deepwave.scalar_born`. It shows how
to invert for a velocity scattering perturbation, starting from the
residual between the observed data and the direct-arrival prediction of a
smooth background model.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import acoustic, acoustic_born

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

ny, nx = 60, 80
dx = 5.0
freq = 20.0
dt = 0.001
nt = 220 if smoke else 300
peak_time = 1.5 / freq

# Background (migration) velocity: smooth, no small-scale features.
v_bg = (1500 + 2.0 * torch.arange(ny).reshape(-1, 1)).expand(ny, nx).float()
v_bg = v_bg.to(device)
rho = (2200 * torch.ones(ny, nx)).to(device)

# True model: the background plus two point-like scatterers, the target
# of the inversion.
dv_true = torch.zeros(ny, nx, device=device)
dv_true[20, 30] = 120.0
dv_true[35, 55] = -100.0
v_true = v_bg + dv_true

n_shots = 2 if smoke else 6
n_receivers_per_shot = 10 if smoke else 40
source_locations = torch.zeros(
    n_shots, 1, 2, dtype=torch.long, device=device
)
source_locations[:, 0, 0] = 3
source_locations[:, 0, 1] = torch.linspace(5, nx - 5, n_shots).long()
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

# "Observed" data: full nonlinear forward through the true model.
observed = acoustic(
    v_true,
    rho,
    dx,
    dt,
    source_amplitudes_volume_rate=source_amplitudes,
    source_locations_volume_rate=source_locations,
    receiver_locations_p=receiver_locations,
    pml_freq=freq,
    max_vel=float(v_true.max()),
).receiver_amplitudes_p

# Direct-arrival prediction of the background (no scatterers) model.
background = acoustic(
    v_bg,
    rho,
    dx,
    dt,
    source_amplitudes_volume_rate=source_amplitudes,
    source_locations_volume_rate=source_locations,
    receiver_locations_p=receiver_locations,
    pml_freq=freq,
    max_vel=float(v_true.max()),
).receiver_amplitudes_p

observed_scattered = observed - background

dv = torch.zeros(ny, nx, device=device, requires_grad=True)
optimiser = torch.optim.LBFGS([dv])
loss_fn = torch.nn.MSELoss()

n_epochs = 1 if smoke else 8
loss_history: list[float] = []
for _epoch in range(n_epochs):

    def closure() -> float:
        """Closure function for the LBFGS optimiser."""
        optimiser.zero_grad()
        out = acoustic_born(
            v_bg,
            rho,
            dx,
            dt,
            dv=dv,
            source_amplitudes_volume_rate=source_amplitudes,
            source_locations_volume_rate=source_locations,
            sc_receiver_locations_p=receiver_locations,
            pml_freq=freq,
            max_vel=float(v_true.max()),
        )
        loss = 1e8 * loss_fn(
            out.sc_receiver_amplitudes_p, observed_scattered
        )
        loss.backward()
        loss_history.append(float(loss.item()))
        return float(loss.item())

    optimiser.step(closure)

_, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True)
vmax = float(dv_true.abs().max())
ax[0].imshow(
    dv_true.cpu(), cmap="seismic", vmin=-vmax, vmax=vmax, aspect="auto"
)
ax[0].set_title("True scattering (dv)")
ax[1].imshow(
    dv.detach().cpu(), cmap="seismic", vmin=-vmax, vmax=vmax, aspect="auto"
)
ax[1].set_title("LSRTM recovered (dv)")
plt.tight_layout()
plt.savefig("example_acoustic_born.jpg")

assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

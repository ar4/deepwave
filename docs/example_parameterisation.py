"""FWI in parameters other than the propagator's native inputs.

Deepwave's propagators take physical model tensors directly (wavespeed,
density, stiffnesses, ...), but nothing requires you to *optimise* those
exact tensors. Because propagation is an ordinary differentiable PyTorch
operation, any invertible reparameterisation -- built from plain tensor
arithmetic -- composes with autograd: build the propagator's native inputs
from your own optimised variables, call the propagator, and ``backward()``
chains the gradient through your reparameterisation automatically. No
dedicated "impedance equation" or other special-cased physics is needed.

This example demonstrates two such reparameterisations, both on a small
crosswell-style survey (sources on the left edge, receivers on the right):

1. Acoustic impedance/velocity: optimise P-velocity ``vp`` and impedance
   ``Z = rho * vp`` directly, building ``rho = Z / vp`` each iteration.
2. Scalar slowness: optimise slowness ``s = 1 / v``, building ``v = 1 / s``
   each iteration.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import acoustic, scalar

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

ny, nx = (30, 36) if smoke else (50, 60)
dx = 10.0
freq = 18.0
dt = 0.001
nt = 300 if smoke else 400
n_epochs = 8 if smoke else 220
yy, xx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")

# Crosswell geometry, shared by both sections: sources down the left edge,
# receivers down the right edge, so waves transmit through the anomaly.
n_src = 3 if smoke else 5
src_y = torch.linspace(5, ny - 5, n_src).long()
source_locations = torch.stack(
    [src_y, torch.full((n_src,), 5)], dim=1
).unsqueeze(1)
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq)
    .reshape(1, 1, -1)
    .repeat(n_src, 1, 1)
    .to(device)
)
n_rec = 6 if smoke else 16
rec_y = torch.linspace(5, ny - 5, n_rec).long()
receiver_locations = (
    torch.stack([rec_y, torch.full((n_rec,), nx - 5)], dim=1)
    .unsqueeze(0)
    .repeat(n_src, 1, 1)
    .to(device)
)
source_locations = source_locations.to(device)


## Acoustic: vp/impedance parameterisation


anomaly_cy, anomaly_cx = ny // 2, nx // 2
anomaly_denom_vp = 29 if smoke else 80
anomaly_denom_rho = 43 if smoke else 120
vp_true = 1500 + 300 * torch.exp(
    -((yy - anomaly_cy) ** 2 + (xx - anomaly_cx) ** 2) / anomaly_denom_vp
)
rho_true = 2200 + 400 * torch.exp(
    -((yy - anomaly_cy) ** 2 + (xx - anomaly_cx) ** 2) / anomaly_denom_rho
)
vp_true, rho_true = vp_true.to(device), rho_true.to(device)


def run_acoustic(vp: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Forward model and return the pressure receiver gathers."""
    return acoustic(
        vp,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes,
        source_locations_volume_rate=source_locations,
        receiver_locations_p=receiver_locations,
        pml_freq=freq,
    ).receiver_amplitudes_p


observed_p = run_acoustic(vp_true, rho_true).detach()

# Optimize vp and Z = rho * vp directly -- not the propagator's own
# (vp, rho) inputs. rho is rebuilt from them every iteration, so its
# gradient is whatever autograd's chain rule gives: d(loss)/d(rho)
# distributed to d(loss)/d(vp) and d(loss)/d(Z).
vp = (1500.0 * torch.ones(ny, nx, device=device)).requires_grad_()
Z = 1500.0 * 2200.0 * torch.ones(ny, nx, device=device)
Z = Z.requires_grad_()
optimizer = torch.optim.Adam(
    [{"params": [vp], "lr": 9.0}, {"params": [Z], "lr": 3500.0}]
)

acoustic_loss_history: list[float] = []
for _epoch in range(n_epochs):
    optimizer.zero_grad()
    rho = Z / vp
    loss = torch.nn.functional.mse_loss(run_acoustic(vp, rho), observed_p)
    loss.backward()
    optimizer.step()
    acoustic_loss_history.append(float(loss.detach()))

Z_true = rho_true * vp_true
vp_range = {"vmin": vp_true.min(), "vmax": vp_true.max()}
Z_range = {"vmin": Z_true.min(), "vmax": Z_true.max()}
fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
ax[0, 0].imshow(vp_true.cpu().T, cmap="viridis", **vp_range)
ax[0, 0].set_title("True vp")
ax[0, 1].imshow(vp.detach().cpu().T, cmap="viridis", **vp_range)
ax[0, 1].set_title("Inverted vp")
ax[1, 0].imshow(Z_true.cpu().T, cmap="viridis", **Z_range)
ax[1, 0].set_title("True Z = rho * vp")
ax[1, 1].imshow(Z.detach().cpu().T, cmap="viridis", **Z_range)
ax[1, 1].set_title("Inverted Z")
fig.tight_layout()
fig.savefig("example_parameterisation_acoustic.jpg")


## Scalar: slowness parameterisation


v_true = 1500 + 350 * torch.exp(
    -((yy - anomaly_cy) ** 2 + (xx - anomaly_cx) ** 2) / anomaly_denom_vp
)
v_true = v_true.to(device)


def run_scalar(v: torch.Tensor) -> torch.Tensor:
    """Forward model and return the receiver gathers."""
    return scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
    ).receiver_amplitudes


observed = run_scalar(v_true).detach()

# Optimize slowness s = 1/v; v = 1/s is rebuilt every iteration. Slowness
# is the natural variable in traveltime tomography (linear in traveltime
# along a raypath) and a common FWI reparameterisation choice.
s = 1.0 / (1500.0 * torch.ones(ny, nx, device=device))
s = s.requires_grad_()
optimizer = torch.optim.Adam([s], lr=3e-6)

scalar_loss_history: list[float] = []
for _epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(run_scalar(1.0 / s), observed)
    loss.backward()
    optimizer.step()
    scalar_loss_history.append(float(loss.detach()))

v_final = (1.0 / s).detach()
v_range = {"vmin": v_true.min(), "vmax": v_true.max()}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True)
ax[0].imshow(v_true.cpu().T, cmap="viridis", **v_range)
ax[0].set_title("True v")
ax[1].imshow(v_final.cpu().T, cmap="viridis", **v_range)
ax[1].set_title("Inverted v (via slowness s = 1/v)")
fig.tight_layout()
fig.savefig("example_parameterisation_scalar.jpg")

assert all(math.isfinite(loss) for loss in acoustic_loss_history), (
    "acoustic parameterisation loss history contains non-finite values"
)
assert acoustic_loss_history[-1] < acoustic_loss_history[0], (
    "acoustic parameterisation loss did not decrease"
)
assert all(math.isfinite(loss) for loss in scalar_loss_history), (
    "scalar parameterisation loss history contains non-finite values"
)
assert scalar_loss_history[-1] < scalar_loss_history[0], (
    "scalar parameterisation loss did not decrease"
)

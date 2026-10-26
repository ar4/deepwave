"""Demonstrates Q (attenuation) inversion with deepwave.viscoacoustic.

:func:`deepwave.viscoacoustic` models a quality factor ``Q`` per grid cell
using generalised-standard-linear-solid (GSLS) relaxation mechanisms.
``q_n_mechanisms > 1`` (with a ``q_band``) fits several mechanisms so ``Q``
is close to constant across a whole frequency band rather than exact only
at one reference frequency -- this example uses two mechanisms. There is
then no single reference frequency, so ``q_freq`` is not given (a
single-mechanism call must give it).

A small crosswell-style survey images a localised low-Q (highly
attenuating) anomaly on top of an otherwise nearly-elastic (high-Q)
background, starting from a uniform high-Q guess. ``Q`` is inverted in
log-space (``q = exp(logq)``, always positive) and, following the
masked-subregion pattern in :ref:`model-parameterisations`, only a
central region away from the sources/PML is optimised -- the region near
the top/side edges is poorly illuminated by this small survey and produces
inversion artefacts there if left unmasked.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import viscoacoustic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny, nx = 40, 60
dx = 10.0
freq = 25.0
dt = 0.0008
nt = 300
peak_time = 1.5 / freq

v = torch.full((ny, nx), 2000.0, device=device)
rho = torch.full((ny, nx), 2000.0, device=device)
q_true = torch.full((ny, nx), 1000.0, device=device)
q_true[15:25, 25:35] = 20.0  # a highly-attenuating target anomaly
# The band-fit q is a target, not necessarily the achieved quality factor.
# At the wavelet frequency this q=20 anomaly has Q_eff about 21.05.
modulus_at_freq = deepwave.attenuation.response(
    torch.tensor(20.0, device=device),
    torch.tensor([freq], device=device),
    q_n_mechanisms=2,
    q_band=(10.0, 50.0),
)
actual_q_at_freq = modulus_at_freq.real / modulus_at_freq.imag
print("Anomaly Q_eff at 25 Hz:", float(actual_q_at_freq[0]))

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
    (1e6 * deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .reshape(1, 1, -1)
    .repeat(n_shots, 1, 1)
    .to(device)
)

# Pressure-rate forcing is in Pa/s. Keep the numerical setup fixed as Q
# changes: v is the relaxed speed, while max_vel must bound the faster
# unrelaxed speed. For this two-mechanism band and Q >= 10, 3000 m/s
# conservatively bounds it. The propagator checks the bound on each call.
q_floor = 10.0


def run(q: torch.Tensor) -> torch.Tensor:
    """Forward model and return the pressure receiver gathers."""
    return viscoacoustic(
        v,
        rho,
        q,
        dx,
        dt,
        source_amplitudes_pressure_rate=source_amplitudes,
        source_locations_pressure_rate=source_locations,
        receiver_locations_p=receiver_locations,
        q_n_mechanisms=2,
        q_band=(10.0, 50.0),
        pml_freq=freq,
        max_vel=3000.0,
    ).receiver_amplitudes_p


observed = run(q_true).detach()
data_scale = observed.square().mean().sqrt()
assert data_scale > 0, "observed data is all zeros"

yy, xx = torch.meshgrid(
    torch.arange(ny, device=device),
    torch.arange(nx, device=device),
    indexing="ij",
)
invert_mask = (yy > 8) & (xx > 8) & (xx < nx - 8)
logq_background = math.log(1000.0 - q_floor)
update = torch.zeros((ny, nx), device=device, requires_grad=True)
optimiser = torch.optim.Adam([update], lr=0.05)
loss_fn = torch.nn.MSELoss()

n_epochs = 15 if smoke else 90
loss_history: list[float] = []
for _epoch in range(n_epochs):
    optimiser.zero_grad()
    logq = logq_background + torch.where(
        invert_mask, update, torch.zeros_like(update)
    )
    q = q_floor + torch.exp(logq)
    loss = loss_fn(run(q) / data_scale, observed / data_scale)
    loss.backward()
    loss_history.append(float(loss.item()))
    optimiser.step()

with torch.no_grad():
    logq_final = logq_background + torch.where(
        invert_mask, update, torch.zeros_like(update)
    )
    q_final = q_floor + torch.exp(logq_final)

_, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True)
ax[0].imshow(q_true.cpu(), vmin=0, vmax=200)
ax[0].set_title("True Q")
ax[1].imshow(q_final.cpu(), vmin=0, vmax=200)
ax[1].set_title("Inverted Q")
plt.tight_layout()
plt.savefig("example_viscoacoustic.jpg")

assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

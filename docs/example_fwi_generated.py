"""A complete FWI run on a generated model, with no data to download.

This script builds its own "true" velocity model, generates the observed
data from it, and then inverts for the model starting from a smooth guess.
It is the shortest complete full-waveform inversion in this documentation:
everything it needs is here, so it can be copied into a file and run. At
the default size it takes a few minutes on a CPU and under a minute on a
GPU; ``DEEPWAVE_SMOKE=1`` (see :ref:`running-the-examples`) shrinks it to
seconds.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# --- The model -------------------------------------------------------
#
# Dimension 0 is depth and dimension 1 is horizontal position, the
# convention used throughout this documentation. Lengths are in metres,
# times in seconds, and velocities in metres per second.
ny = 40 if smoke else 80  # cells in depth
nx = 80 if smoke else 160  # cells horizontally
dx = 10.0  # m, the same spacing in both dimensions
dt = 0.002  # s

# Background: velocity rising linearly with depth, from 1500 m/s at the
# surface to 2290 m/s at the bottom of the 800 m model.
depth = torch.arange(ny, dtype=torch.float32).reshape(ny, 1) * dx
horizontal = torch.arange(nx, dtype=torch.float32).reshape(1, nx) * dx
v_background = (1500.0 + 1.0 * depth).repeat(1, nx)

# The feature to be recovered: a fast, roughly circular anomaly 300 m
# across and up to 400 m/s faster than its surroundings, centred 280 m
# below the middle of the surface.
radius = 150.0
anomaly_depth = 0.35 * ny * dx
distance = torch.sqrt(
    (depth - anomaly_depth) ** 2 + (horizontal - nx * dx / 2) ** 2
)
v_true = (v_background + 400.0 * torch.exp(-((distance / radius) ** 2))).to(
    device
)

# The starting model knows the background but not the anomaly. Starting
# from the correct background is deliberate: FWI recovers detail from a
# model that is already kinematically close, and cycle-skipping (see
# :doc:`example_fwi`) is what happens when it is not.
v_init = v_background.to(device)

# --- The survey ------------------------------------------------------
#
# Sources and receivers sit two cells below the surface, spread across the
# model. ``source_amplitudes`` has shape [shot, source, time] and the
# location tensors have shape [shot, source_or_receiver, dimension], with
# the dimension column ordered [depth, horizontal] to match the model.
n_shots = 3 if smoke else 8
n_sources_per_shot = 1
n_receivers_per_shot = nx // 2
freq = 12.0  # Hz, the dominant frequency of the source wavelet
nt = 200 if smoke else 600  # time samples, so 0.4 s or 1.2 s recorded
peak_time = 1.5 / freq

source_locations = torch.zeros(
    n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device
)
source_locations[..., 0] = 2
source_locations[:, 0, 1] = torch.linspace(
    nx * 0.1, nx * 0.9, n_shots
).long()

receiver_locations = torch.zeros(
    n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
)
receiver_locations[..., 0] = 2
receiver_locations[:, :, 1] = (
    torch.arange(n_receivers_per_shot) * 2
).repeat(n_shots, 1)

source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# --- Numerical parameters, fixed for every propagation ---------------
#
# ``max_vel`` and ``pml_freq`` are given explicitly and never change, so
# every iteration discretises the problem in exactly the same way. Left
# out, Deepwave would infer them from the model and from the source
# amplitudes, and an inferred ``max_vel`` would then drift as the model is
# updated, changing the internal time step from one iteration to the next.
# ``max_vel`` must be at least the fastest velocity the model reaches at
# any point during the inversion, not just its starting value, so it is set
# above the true maximum here. See :doc:`accuracy` for ``max_vel`` and
# :ref:`pml-freq` for ``pml_freq``.
max_vel = 3000.0
pml_freq = freq


def propagate(v):
    """Propagates every shot through ``v`` and returns the receiver data."""
    return scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=max_vel,
        pml_freq=pml_freq,
    ).receiver_amplitudes


# --- The observed data -----------------------------------------------
#
# Generating the observed data with the same propagator that the inversion
# uses makes this an "inverse crime": there is no modelling error at all,
# so the inversion has an easier time of it than it would with real data.
with torch.no_grad():
    observed_data = propagate(v_true)

# --- The inversion ---------------------------------------------------
#
# The step size is chosen as a velocity update, in metres per second,
# rather than as an abstract learning rate. The gradient of a data misfit
# with respect to velocity has awkward units and a magnitude that depends
# on the source amplitude, the number of shots and the loss function, so a
# learning rate carried over from another problem means nothing here. Adam
# divides each cell's gradient by a running estimate of its own magnitude,
# so its learning rate *is* roughly the velocity change per cell per
# iteration: ``lr=10.0`` asks for about 10 m/s. Choose it relative to the
# scale of the model: a few metres per second per iteration, against
# velocities spanning hundreds, makes visible progress in tens of
# iterations while keeping each update a small perturbation of the model
# the gradient was computed for. Too large and the inversion oscillates or
# diverges; too small and it stalls. (Plain ``SGD`` instead leaves the
# gradient's magnitude in the step, so there the same reasoning is applied
# by clipping the gradient -- as :doc:`example_fwi` does -- and setting the
# learning rate to the wanted update divided by the clipping threshold.)
max_update = 10.0  # m/s per iteration
n_epochs = 25 if smoke else 200

v = v_init.clone().requires_grad_()
optimiser = torch.optim.Adam([v], lr=max_update)
loss_fn = torch.nn.MSELoss()

# The cells immediately around a source or receiver receive a
# disproportionate share of the gradient, which would put a bullseye
# artefact at every one of them. Zeroing the gradient over the top few rows
# removes that, and costs nothing here because the anomaly is deeper. See
# :doc:`example_large_edge_gradient` for why the gradient behaves that way
# and for other ways of handling it.
gradient_mask = torch.ones(ny, 1, device=device)
gradient_mask[:4] = 0.0

loss_history: list[float] = []
for _epoch in range(n_epochs):
    optimiser.zero_grad()
    loss = loss_fn(propagate(v), observed_data)
    loss.backward()
    loss_history.append(float(loss.item()))
    v.grad *= gradient_mask
    optimiser.step()

with torch.no_grad():
    final_loss = float(loss_fn(propagate(v), observed_data))
    initial_rmse = float((v_init - v_true).pow(2).mean().sqrt())
    final_rmse = float((v.detach() - v_true).pow(2).mean().sqrt())
print(
    f"data misfit {loss_history[0]:.4g} -> {final_loss:.4g} "
    f"({loss_history[0] / final_loss:.1f}x reduction); "
    f"model RMSE {initial_rmse:.1f} -> {final_rmse:.1f} m/s"
)

# --- The result ------------------------------------------------------
#
# The anomaly is recovered where the survey illuminates it, and only there.
# Every source and receiver is at the surface, so the top of the model is
# crossed by many ray paths at many angles and the bottom by few at almost
# the same angle. The recovery is therefore partial: the anomaly comes back
# in the right place with the right sign but only part of its amplitude, it
# is smeared vertically, and the corners beyond the outermost sources are
# barely updated at all. Widening the survey, recording for longer, adding
# lower frequencies, or putting receivers in a borehole all change what can
# be recovered -- and none of them is a change to how the inversion is
# written.
vmin = float(v_true.min())
vmax = float(v_true.max())
_, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), sharex=True, sharey=True)
ax[0].imshow(v_init.cpu(), aspect="auto", vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.detach().cpu(), aspect="auto", vmin=vmin, vmax=vmax)
ax[1].set_title("Inverted")
im = ax[2].imshow(v_true.cpu(), aspect="auto", vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.colorbar(im, ax=ax, label="Velocity (m/s)")
plt.savefig("example_fwi_generated.jpg")

assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert final_loss < loss_history[0] / 2, (
    "data misfit did not at least halve"
)
# A handful of iterations on the reduced grid move the data misfit but
# not yet the model measurably, so this check applies to the full run.
assert smoke or final_rmse < initial_rmse, (
    "model did not move towards the true model"
)

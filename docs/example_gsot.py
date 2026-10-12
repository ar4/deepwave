"""Demonstrates the Graph Space Optimal Transport (GSOT) method."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfilt

import deepwave

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"


def gsot(y_pred: torch.Tensor, y: torch.Tensor, eta: float):
    """Graph Space Optimal Transport cost function.

    Args:
        y_pred: The predicted signal, of dimensions [shot, receiver, time]
        y: The true signal
        eta: Parameter controlling the relative weight of the two
             terms in the cost function (which relate to differences in
             the time and amplitude dimensions)

    Returns:
        The cost function value/loss

    """
    loss = torch.tensor(0.0, device=y_pred.device)
    for s in range(y.shape[0]):
        for r in range(y.shape[1]):
            nt = y.shape[-1]
            c = np.zeros([nt, nt])
            for i in range(nt):
                for j in range(nt):
                    c[i, j] = (
                        eta * (i - j) ** 2
                        + (y_pred.detach()[s, r, i] - y[s, r, j]) ** 2
                    )
            row_ind, col_ind = linear_sum_assignment(c)
            y_sigma = y[s, r, col_ind]
            loss = (
                loss
                + (
                    eta
                    * torch.tensor(row_ind - col_ind, device=y_pred.device)
                    ** 2
                    + (y_pred[s, r] - y_sigma) ** 2
                ).sum()
            )
    return loss


# Loss vs shift

freq = 25
dt = 0.004
nt = 40 if smoke else 100

source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, (nt // 2) * dt)

shifts = list(range(-8, 9)) if smoke else list(range(-30, 31))
gsot_losses = [
    gsot(
        deepwave.wavelets.ricker(freq, nt, dt, (nt // 2 + shift) * dt)[
            None, None
        ],
        source_amplitudes[None, None],
        0.003,
    )
    for shift in shifts
]

l2_losses = []
for shift in shifts:
    l2_losses.append(
        torch.linalg.vector_norm(
            deepwave.wavelets.ricker(freq, nt, dt, (nt // 2 + shift) * dt)
            - source_amplitudes,
        )
        ** 2,
    )

plt.figure(figsize=(10.5, 5))
plt.plot(source_amplitudes.flatten(), c="b", lw=2, label="Target")
plt.plot(
    deepwave.wavelets.ricker(
        freq, nt, dt, (nt // 2 + shift) * dt
    ).flatten(),
    c="r",
    lw=2,
    label="Shifted",
)
plt.xticks([])
plt.yticks([])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("example_gsot_shifted.png")

plt.figure(figsize=(10.5, 5))
plt.plot(shifts, l2_losses, c="y", lw=2, label="$\\ell_2$")
plt.plot(shifts, gsot_losses, c="g", lw=2, label="GSOT")
plt.xlabel("Shift")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("example_gsot_shift_loss.png")


# Inversion

ny = 60 if smoke else 100
nx = 90 if smoke else 216
freq = 25
dt = 0.004
nt = 60 if smoke else 200
v_true = 1600 * torch.ones(ny, nx)

source_amplitudes = deepwave.wavelets.ricker(
    freq, nt, dt, 1.5 / freq
).reshape(1, 1, -1)
# Filter the source amplitudes to remove low frequencies
sos = butter(4, 15, "hp", fs=1 / dt, output="sos")
source_amplitudes = torch.tensor(
    sosfilt(sos, source_amplitudes.numpy())
).float()

source_locations = torch.tensor([[[ny // 2, 20]]])
receiver_locations = torch.tensor([[[ny // 2, nx - 1 - 20]]])

observed_data = deepwave.scalar(
    v_true,
    grid_spacing=4,
    dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
).receiver_amplitudes

vs = []
loss_histories: dict[str, list[float]] = {}
for loss_type in ["gsot", "l2"]:
    # Starting model
    v = 1500 * torch.ones(ny, nx)
    v.requires_grad_()

    # Setup optimiser to perform inversion
    optimiser = torch.optim.Adam([v], lr=10)

    # Run optimisation/inversion
    n_epochs = 8 if smoke else 250

    for _epoch in range(n_epochs):

        def closure():
            """Closure function for the Adam optimiser."""
            optimiser.zero_grad()
            pred_data = deepwave.scalar(
                v,
                grid_spacing=4,
                dt=0.004,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
            ).receiver_amplitudes
            if loss_type == "gsot":
                loss = gsot(pred_data, observed_data, 0.003)
            else:
                loss = 2e2 * torch.nn.MSELoss()(pred_data, observed_data)
            loss.backward()
            loss_histories.setdefault(loss_type, []).append(
                float(loss.detach())
            )
            return loss

        optimiser.step(closure)
    vs.append(v.detach())

# Compute the forward modelled data in the initial model and in the
# final models produced when using the GSOT and MSE cost functions
# so that they can by plotted and compared

initial_data = deepwave.scalar(
    1500 * torch.ones(ny, nx),
    grid_spacing=4,
    dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
).receiver_amplitudes

gsot_data = deepwave.scalar(
    vs[0],
    grid_spacing=4,
    dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
).receiver_amplitudes

l2_data = deepwave.scalar(
    vs[1],
    grid_spacing=4,
    dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
).receiver_amplitudes

plt.figure(figsize=(10.5, 5))
plt.plot(observed_data.flatten()[100:180], c="b", lw=2, label="Target")
plt.plot(initial_data.flatten()[100:180], c="r", lw=2, label="Initial")
plt.plot(l2_data.flatten()[100:180], c="y", lw=2, label="$\\ell_2$")
plt.plot(gsot_data.flatten()[100:180], c="g", lw=2, label="GSOT")
plt.xticks([])
plt.yticks([])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("example_gsot_inversion.png")

assert all(
    math.isfinite(loss) for hist in loss_histories.values() for loss in hist
), "a loss history contains non-finite values"
for _loss_type, _hist in loss_histories.items():
    assert _hist[-1] < _hist[0], f"{_loss_type} loss did not decrease"

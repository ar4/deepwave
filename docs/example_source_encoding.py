"""Random-polarity source encoding for stochastic FWI."""

import os

import matplotlib.pyplot as plt
import torch

import deepwave

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# ---------------------------------------------------------------------
# 1. A small cross-hole problem, shared by every gradient computed below
# ---------------------------------------------------------------------
# The true model is a homogeneous background with a Gaussian slow
# anomaly; the starting model, from which every gradient in this example
# is computed, is the background alone. So the "correct" gradient points
# towards the anomaly, which makes it easy to recognise a good gradient
# in the plot below and gives every method the same, non-trivial target.
ny, nx = (20, 32) if smoke else (44, 72)
dx = 10.0
dt = 0.002
freq = 15.0
nt = 140 if smoke else 350
peak_time = 1.5 / freq
anomaly_cy, anomaly_cx = ny // 2, nx // 2
anomaly_sy = 4 if smoke else 8
anomaly_sx = 5 if smoke else 10

y_grid = torch.arange(ny, device=device).float()
x_grid = torch.arange(nx, device=device).float()
yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
v_true = 1500.0 - 200.0 * torch.exp(
    -(
        ((yy - anomaly_cy) / anomaly_sy) ** 2
        + ((xx - anomaly_cx) / anomaly_sx) ** 2
    )
)
v_start = 1500.0 * torch.ones(ny, nx, device=device)

# Sources down the left edge, one per shot; receivers down the right
# edge, the same array for every shot. Sharing one receiver array across
# shots is what makes summing their recordings below meaningful: sample
# ``t`` of receiver ``r`` means the same physical thing (the wavefield at
# that receiver's location) for every shot.
n_shots = 6 if smoke else 12
source_locations = torch.zeros(
    n_shots, 1, 2, dtype=torch.long, device=device
)
source_locations[:, 0, 0] = torch.linspace(
    4, ny - 4, n_shots, device=device
).long()
source_locations[:, 0, 1] = 2

n_rec = 8 if smoke else 20
receiver_locations = torch.zeros(
    n_shots, n_rec, 2, dtype=torch.long, device=device
)
receiver_locations[..., 0] = (
    torch.linspace(2, ny - 2, n_rec, device=device)
    .long()
    .repeat(n_shots, 1)
)
receiver_locations[..., 1] = nx - 3

wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time).to(device)
source_amplitudes = wavelet.reshape(1, 1, -1).repeat(n_shots, 1, 1)

observed_data = deepwave.scalar(
    v_true,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_freq=freq,
).receiver_amplitudes.detach()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error data misfit."""
    return torch.nn.MSELoss()(pred, target)


# ---------------------------------------------------------------------
# 2. The ordinary gradient: one propagation per shot, summed
# ---------------------------------------------------------------------
# This is what every FWI example elsewhere in these docs does: propagate
# each shot on its own, compare it with that shot's observed data, and
# accumulate the gradient of the total misfit by calling ``backward()``
# once per shot (gradients accumulate into ``.grad`` across repeated
# ``backward()`` calls, which is exactly summing the per-shot
# contributions).
v_sequential = v_start.clone().requires_grad_()
for shot in range(n_shots):
    pred = deepwave.scalar(
        v_sequential,
        dx,
        dt,
        source_amplitudes=source_amplitudes[shot : shot + 1],
        source_locations=source_locations[shot : shot + 1],
        receiver_locations=receiver_locations[shot : shot + 1],
        pml_freq=freq,
    ).receiver_amplitudes
    mse(pred, observed_data[shot : shot + 1]).backward()
assert v_sequential.grad is not None
grad_sequential = v_sequential.grad.clone()

# ---------------------------------------------------------------------
# 3. Source encoding: one propagation per iteration, not one per shot
# ---------------------------------------------------------------------
# Instead of propagating each shot separately, give every source a random
# polarity, +1 or -1, and fire all of them at once as a single "super-shot"
# -- several sources sharing one propagation, which is exactly what
# Deepwave's batched source arguments already support (a
# ``source_locations`` entry has shape ``[shot, source, 2]``; here that
# middle axis holds one source per encoded shot, instead of the usual
# one). Summing the observed data for the individual shots with the
# same random signs produces the matching "observed" super-shot data.
# A single propagation of the super-shot, compared against that summed
# data, gives a **stochastic gradient**: an approximation of the true
# summed gradient that is cheap to compute (one propagation instead of
# ``n_shots``) but noisy.
#
# The noise has a name: **cross-talk**. Superposing several shots means the
# wavefield each source's receiver signal is compared against also contains
# every other source's wavefield. Squaring that sum in the misfit produces
# cross terms between different shots' wavefields, on top of the wanted
# per-shot term, and it is those cross terms that pollute the gradient. This
# is the entire cost of the technique, and the reason it is a "recipe" and
# not a free lunch.
encoded_source_locations = source_locations[:, 0, :].unsqueeze(0)


def encoded_gradient(n_shots_in_encoding: int = n_shots) -> torch.Tensor:
    """One realisation of the encoded gradient, from freshly-drawn signs."""
    signs = (
        torch.randint(0, 2, (n_shots_in_encoding,), device=device) * 2 - 1
    ).float()
    encoded_amplitudes = (signs[:, None] * wavelet[None, :]).unsqueeze(0)
    encoded_observed = (
        signs[:, None, None] * observed_data[:n_shots_in_encoding]
    ).sum(dim=0, keepdim=True)
    v = v_start.clone().requires_grad_()
    pred = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=encoded_amplitudes,
        source_locations=encoded_source_locations[:, :n_shots_in_encoding],
        receiver_locations=receiver_locations[:1],
        pml_freq=freq,
    ).receiver_amplitudes
    mse(pred, encoded_observed).backward()
    assert v.grad is not None
    return v.grad.clone()


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two flattened tensors."""
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()


# Redrawing the random signs matters because cross-talk is *noise*, not
# bias: its expected value over random signs is zero (each unwanted
# cross term is, on average, as likely to be positive as negative), so
# averaging encoded gradients from independently redrawn signs averages
# the cross-talk down while the wanted signal, which is the same in
# every realisation, adds up undiminished. In a real stochastic-FWI run
# this happens for free across optimiser iterations, because each
# iteration redraws its own signs; the same iterations that pay for the
# encoded propagations also do the averaging. A single realisation is
# still useful -- it is an unbiased gradient estimate -- but it is
# noticeably noisier than the summed gradient, which is why the plot
# below compares one realisation against an average of several.
n_realisations = 4 if smoke else 12
single_realisation_grad = encoded_gradient()
realisation_grads = [single_realisation_grad] + [
    encoded_gradient() for _ in range(n_realisations - 1)
]
grad_encoded_averaged = torch.stack(realisation_grads).mean(dim=0)

correlation_single = pearson(grad_sequential, single_realisation_grad)
correlation_averaged = pearson(grad_sequential, grad_encoded_averaged)

# A running average, one more realisation at a time, to show cross-talk
# averaging down as the plot below illustrates.
running_correlation = []
running_sum = torch.zeros_like(v_start)
for i, g in enumerate(realisation_grads, start=1):
    running_sum = running_sum + g
    running_correlation.append(pearson(grad_sequential, running_sum / i))

print(f"correlation, single realisation:   {correlation_single:.4f}")
print(
    f"correlation, {n_realisations:2d} realisations averaged: "
    f"{correlation_averaged:.4f}"
)

# ---------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
grads = [
    grad_sequential,
    single_realisation_grad,
    grad_encoded_averaged,
]
titles = [
    f"Sequential ({n_shots} shots)",
    "Encoded, 1 realisation",
    f"Encoded, {n_realisations} realisations averaged",
]
limit = grad_sequential.abs().max().item()
for a, grad, title in zip(ax[:3], grads, titles):
    a.imshow(
        grad.detach().cpu().T,
        cmap="seismic",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    a.set_title(title, fontsize=9)
ax[3].plot(
    range(1, n_realisations + 1), running_correlation, marker="o", ms=3
)
ax[3].axhline(0.0, color="k", lw=0.8, ls="--")
ax[3].set_xlabel("realisations averaged")
ax[3].set_ylabel("correlation with sequential gradient")
ax[3].set_title("Cross-talk averaging down", fontsize=9)
plt.tight_layout()
plt.savefig("example_source_encoding.jpg")

# ---------------------------------------------------------------------
# 5. Verification
# ---------------------------------------------------------------------
# A single realisation is noisy -- it is not asserted on here, and it is
# normal for it to sometimes correlate only weakly, or even (rarely)
# negatively -- but the *averaged* encoded gradient, over enough
# realisations, must correlate positively and comfortably with the
# sequential-shots gradient it is standing in for: if it did not, the
# technique would not be a usable approximation of ordinary FWI.
assert correlation_averaged > 0.5, (
    "the averaged encoded gradient does not correlate positively (and "
    "robustly) with the sequential-shots gradient: "
    f"{correlation_averaged:.4f}"
)

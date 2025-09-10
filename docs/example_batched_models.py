"""This script demonstrates how to use batched models in Deepwave.

This is useful when you want to propagate multiple shots through different
models simultaneously.
"""

import matplotlib.pyplot as plt
import torch

import deepwave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Part 1: Forward and backward propagation with batched models ----
# In this part, we demonstrate simultaneous propagation of two shots in
# two different velocity models.

# Create two velocity models
ny = 100
nx = 200
dx = 4.0
v_background1 = 1500.0
v_background2 = 3000.0
v = torch.ones(2, ny, nx, device=device)
v[0] *= v_background1
v[1] *= v_background2

# Add a high-velocity anomaly to each model in a different location
v[0, ny // 2 : ny // 2 + 3, nx // 4 : nx // 4 + 3] = 1800.0
v[1, ny // 2 : ny // 2 + 3, 3 * nx // 4 : 3 * nx // 4 + 3] = 3600.0

# Plot the two velocity models
_, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
ax[0].imshow(v[0].cpu().detach(), cmap="viridis", vmin=1500, vmax=3600)
ax[0].set_title("Model 1")
ax[1].imshow(v[1].cpu().detach(), cmap="viridis", vmin=1500, vmax=3600)
ax[1].set_title("Model 2")
plt.tight_layout()
plt.savefig("example_batched_models_part1_models.jpg")

# Set up the wave propagation
n_shots = 2
n_sources_per_shot = 1
n_receivers_per_shot = nx
freq = 25
nt = 250
dt = 0.004
peak_time = 1.5 / freq

# Create source and receiver locations for two shots
source_locations = torch.zeros(
    n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device
)
source_locations[..., 0] = 5  # y
source_locations[..., 1] = nx // 2  # x

receiver_locations = torch.zeros(
    n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
)
receiver_locations[..., 0] = 5  # y
receiver_locations[..., 1] = torch.arange(n_receivers_per_shot)  # x

# Create source amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# Propagate waves
out = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    max_vel=v.max().item(),
)
receiver_data = out[-1]

# Plot the receiver data
vmax = torch.quantile(receiver_data, 0.95)
_, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
ax[0].imshow(
    receiver_data[0].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=-vmax,
    vmax=vmax,
)
ax[0].set_title("Data from Model 1")
ax[1].imshow(
    receiver_data[1].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=-vmax,
    vmax=vmax,
)
ax[1].set_title("Data from Model 2")
plt.tight_layout()
plt.savefig("example_batched_models_part1_data.jpg")

# Backpropagate to calculate gradients
v_init = torch.ones(2, ny, nx, device=device)
v_init[0] *= v_background1
v_init[1] *= v_background2
v_init.requires_grad_()
out = deepwave.scalar(
    v_init,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    max_vel=v.max().item(),
)

torch.nn.MSELoss()(out[-1], receiver_data).backward()

# Plot the gradients
grad = v_init.grad.detach().cpu()
vmax = torch.quantile(grad.abs(), 0.95)
_, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
ax[0].imshow(grad[0], cmap="viridis", vmin=-vmax, vmax=vmax)
ax[0].set_title("Gradient for Model 1")
ax[1].imshow(grad[1], cmap="viridis", vmin=-vmax, vmax=vmax)
ax[1].set_title("Gradient for Model 2")
plt.tight_layout()
plt.savefig("example_batched_models_part1_gradients.jpg")


# ---- Part 2: Inversion with a model-difference penalty ----
# In this part, we demonstrate an inversion where each shot has its own
# velocity model. We add a penalty to the loss function to encourage
# all the models to be similar.

# Use a small number of shots for a quick example
n_shots_inversion = 3

# Create a single "true" model to generate data
v_true = torch.full((ny, nx), 2500.0, device=device)
v_true[ny // 2 - 10 : ny // 2 + 10, nx // 2 - 10 : nx // 2 + 10] = (
    2000.0  # Low-velocity anomaly
)

# Generate true data for all shots from the same true model
source_locations_inv = torch.zeros(
    n_shots_inversion, 1, 2, dtype=torch.long, device=device
)
source_locations_inv[..., 0] = 5  # y source
# Spread sources across the model
source_locations_inv[:, 0, 1] = torch.linspace(
    nx // 4, 3 * nx // 4, n_shots_inversion
).long()  # x source

receiver_locations_inv = torch.zeros(
    n_shots_inversion, nx, 2, dtype=torch.long, device=device
)
receiver_locations_inv[..., 0] = ny - 5  # y receiver
receiver_locations_inv[..., 1] = torch.arange(nx)  # x receiver

source_amplitudes_inv = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots_inversion, 1, 1)
    .to(device)
)

true_data = deepwave.scalar(
    v_true,
    dx,
    dt,
    source_amplitudes=source_amplitudes_inv,
    source_locations=source_locations_inv,
    receiver_locations=receiver_locations_inv,
)[-1]

# Create initial models for inversion: one for each shot
# Start with homogeneous models
v_init = torch.full((n_shots_inversion, ny, nx), 2500.0, device=device)
v_init.requires_grad_()

# Store a copy for plotting
v_start = v_init.clone().detach()

# Set up optimiser
optimiser = torch.optim.SGD([v_init], lr=1.0)
loss_fn = torch.nn.MSELoss()

n_plots = 3
plot_freq = 20

fig, ax = plt.subplots(
    n_plots + 1,
    n_shots_inversion,
    figsize=(12, 8),
    sharex=True,
    sharey=True,
)
for i in range(n_shots_inversion):
    ax[0, i].set_title(f"Shot {i}")

# Inversion loop
n_epochs = n_plots * plot_freq + 1
for epoch in range(n_epochs):
    # The weight of the penalty term increases with epochs
    penalty_weight = 1e4 * (epoch / (n_epochs - 1)) ** 2

    optimiser.zero_grad()

    # Forward propagate all shots, each with its own model
    pred_data = deepwave.scalar(
        v_init,
        dx,
        dt,
        source_amplitudes=source_amplitudes_inv,
        source_locations=source_locations_inv,
        receiver_locations=receiver_locations_inv,
    )[-1]

    # Data misfit loss
    loss_mse = 1e9 * loss_fn(pred_data, true_data)

    # Model difference penalty
    # Penalise the variance between the models
    loss_penalty = torch.pow(v_init - v_init.mean(dim=0), 2).mean()

    # Total loss
    loss = loss_mse + penalty_weight * loss_penalty
    loss.backward()
    optimiser.step()

    if epoch % plot_freq == 0:
        plot_idx = epoch // plot_freq
        vmin, vmax = torch.quantile(
            v_init.cpu().detach(), torch.tensor([0.05, 0.95])
        )
        for i, v in enumerate(v_init):
            ax[plot_idx, i].imshow(
                v.cpu().detach(), cmap="viridis", vmin=vmin, vmax=vmax
            )

plt.tight_layout()
plt.savefig("example_batched_models_part2_inversion.jpg")

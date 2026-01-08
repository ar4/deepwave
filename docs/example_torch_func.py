"""Demonstrates using PyTorch's torch.func features with Deepwave.

Deepwave supports using `torch.func` features such as `vmap` (vectorizing
map) and forward-mode automatic differentiation (AD) to perform efficient
batch processing and sensitivity analysis.

This example demonstrates two key applications:
1.  **Forward-Mode AD for Scattering:** Using `torch.func.jvp`
    (Jacobian-Vector Product) to compute the scattered wavefield (linearized
    response) from a velocity perturbation. We verify this matches the
    output of the explicit `scalar_born` propagator.
2.  **Batched Sensitivity Analysis:** Using `torch.func.vmap` combined with
    `jvp` to efficiently calculate the scattered response for a batch of
    different perturbations (e.g., potential defects or anomalies) in
    parallel, without needing to manually loop or replicate the background
    model.

Note:
    Using `torch.func` with Deepwave currently requires:
    -   Operating on property models (velocity, scattering potential), not
        source/receiver locations.
    -   Using the pure Python backend (`python_backend="eager"`).
    -   Explicitly specifying `max_vel`, as `vmap` hides the concrete values
        needed for automatic time-step calculation.
"""

import matplotlib.pyplot as plt
import torch

import deepwave

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# 1. Setup the Simulation
# -----------------------------------------------------------------------

# Define a simple 2D velocity model (background)
ny = 100
nx = 100
dx = 4.0
v_bg = 1500 * torch.ones(ny, nx, device=device)
v_bg[ny // 2 :] = 2000  # Two layers

# Define a perturbation (scattering potential) - a small square anomaly
scatter = torch.zeros_like(v_bg)
scatter[ny // 2 - 5 : ny // 2 + 5, nx // 2 - 5 : nx // 2 + 5] = 100

# Acquisition parameters
n_shots = 1
n_sources_per_shot = 1
n_receivers_per_shot = nx
nt = 200
dt = 0.004
freq = 25
peak_time = 1.5 / freq

# Source and receiver locations
source_locations = torch.zeros(
    n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device
)
source_locations[0, 0, 0] = 10
source_locations[0, 0, 1] = nx // 2

receiver_locations = torch.zeros(
    n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
)
receiver_locations[0, :, 0] = 10
receiver_locations[0, :, 1] = torch.arange(nx)

# Source wavelet (Ricker)
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .to(device)
)

# --------------------------------------------------------------------------
# 2. Born Modeling (Reference)
# --------------------------------------------------------------------------
# Calculate the scattered wavefield using the explicit Born propagator.
# This linearizes the wave equation around v_bg.

print("Running explicit Born modeling...")
out_born = deepwave.scalar_born(
    v_bg,
    scatter,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_freq=freq,
)[-1]  # Last output is the scattered receiver data


# --------------------------------------------------------------------------
# 3. Forward-Mode AD with torch.func.jvp
# --------------------------------------------------------------------------
# We can achieve the exact same result by differentiating the nonlinear
# propagator `scalar` using forward-mode AD. This computes the directional
# derivative of the output with respect to the velocity model, in the
# direction of the perturbation `scatter`.


def forward_sim(v):
    """Wrap the nonlinear simulation."""
    # Note: We must specify `max_vel` and `python_backend="eager"`
    return deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
        max_vel=2000,
        python_backend="eager",
    )[-1]


print("Running forward-mode AD (JVP)...")
# Calculate the Jacobian-Vector Product
# primals = (v_bg,), tangents = (scatter,)
_, out_jvp = torch.func.jvp(forward_sim, (v_bg,), (scatter,))

# Verify that the results match
diff = (out_born - out_jvp).abs().max().item()
print(f"Max difference between Born and JVP: {diff:.6e}")


# --------------------------------------------------------------------------
# 4. Efficient Batch Processing with vmap
# --------------------------------------------------------------------------
# Suppose we want to evaluate the scattering response for multiple different
# perturbations (e.g., checking sensitivity to anomalies at different
# locations).
# We can use `vmap` to vectorize the JVP calculation over a batch of
# perturbations.

# Create a batch of 3 perturbations
scatter_batch = torch.stack(
    [
        scatter,  # Center
        scatter.roll(-20, dims=1),  # Shifted left
        scatter.roll(20, dims=1),  # Shifted right
    ]
)
print(
    f"Running batched JVP with vmap for {len(scatter_batch)} "
    "perturbations..."
)


def get_jvp(tangent):
    """Return the JVP of a single perturbation."""
    # We fix the primal (v_bg) and vary the tangent
    return torch.func.jvp(forward_sim, (v_bg,), (tangent,))[1]


# Vectorize `get_jvp` over the perturbation input
out_vmap = torch.func.vmap(get_jvp)(scatter_batch)

# Verify against explicit Born modeling with batched input
# Deepwave natively supports batched inputs, so we can pass the batch
# directly.
# We need to expand other inputs to match the batch size N=3.
N = len(scatter_batch)
out_born_batch = deepwave.scalar_born(
    v_bg.expand(N, -1, -1),
    scatter_batch,
    dx,
    dt,
    source_amplitudes=source_amplitudes.expand(N, -1, -1),
    source_locations=source_locations.expand(N, -1, -1),
    receiver_locations=receiver_locations.expand(N, -1, -1),
    pml_freq=freq,
)[-1]

batch_diff = (out_born_batch - out_vmap[:, 0]).abs().max().item()
print(f"Max difference for batched vmap: {batch_diff:.6e}")


# --------------------------------------------------------------------------
# 5. Plotting
# --------------------------------------------------------------------------
# Plot the results for the batched case

fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

vmin, vmax = out_born.min().item(), out_born.max().item()

# Plot Born Result
ax[0].imshow(
    out_born[0].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[0].set_ylabel("Time Sample")
ax[0].set_title("Scalar Born")

# Plot JVP Result
ax[1].imshow(
    out_jvp[0].cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[1].set_title("torch.func jvp")

# Plot the difference
ax[2].imshow(
    out_born[0].cpu().T - out_jvp[0].cpu().T,
    aspect="auto",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
)
ax[2].set_title("Difference")

ax[0].set_xlabel("Receiver Channel")
ax[1].set_xlabel("Receiver Channel")
ax[2].set_xlabel("Receiver Channel")

plt.tight_layout()
plt.savefig("example_torch_func_jvp.jpg")


# Plot the results for the batched case

fig, ax = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)

vmin, vmax = out_born_batch.min().item(), out_born_batch.max().item()

for i in range(3):
    # Plot Born Result
    ax[i, 0].imshow(
        out_born_batch[i].cpu().T,
        aspect="auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    ax[i, 0].set_ylabel("Time Sample")
    if i == 0:
        ax[i, 0].set_title("Scalar Born")

    # Plot vmap(JVP) Result
    ax[i, 1].imshow(
        out_vmap[i, 0].cpu().T,
        aspect="auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    if i == 0:
        ax[i, 1].set_title("torch.func vmap(jvp)")

    # Plot the difference
    ax[i, 2].imshow(
        out_born_batch[i].cpu().T - out_vmap[i, 0].cpu().T,
        aspect="auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    if i == 0:
        ax[i, 2].set_title("Difference")

    # Label the perturbation
    loc = "Center" if i == 0 else ("Left" if i == 1 else "Right")
    ax[i, 0].text(
        5,
        20,
        f"Perturbation: {loc}",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.5},
    )

ax[-1, 0].set_xlabel("Receiver Channel")
ax[-1, 1].set_xlabel("Receiver Channel")

plt.tight_layout()
plt.savefig("example_torch_func_vmap_jvp.jpg")

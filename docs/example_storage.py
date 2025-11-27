"""Demonstrates 3D FWI with storage options to reduce memory usage.

This script shows how to perform 3D FWI using Deepwave, and how to use
the storage features to reduce GPU memory consumption by storing
intermediate data on the CPU (or disk) and optionally compressing it.
"""

import torch

import deepwave
from deepwave import scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the 3D model
# A small 30x30x30 grid for demonstration purposes
nx = 30
ny = 30
nz = 10
dx = 10.0
dy = 10.0
dz = 10.0

# Create a background velocity model
v_background = torch.full((nz, ny, nx), 1500.0, device=device)

# Create the true model by adding a block anomaly
v_true = v_background.clone()
# Anomaly in the center
v_true[4:8, 13:18, 13:18] = 1600

# Starting model for inversion (smooth background)
v = v_background.clone()
v.requires_grad_()

# Acquisition setup
# 4 sources positioned around the center
n_shots = 4
n_sources_per_shot = 1
# Dense receiver grid on the surface
n_receivers = nx * ny

source_locations = torch.zeros(
    n_shots, n_sources_per_shot, 3, dtype=torch.long, device=device
)
receiver_locations = torch.zeros(
    n_shots, n_receivers, 3, dtype=torch.long, device=device
)

# Source locations
source_coords = [
    [0, 0, 0],
    [0, 29, 0],
    [0, 0, 29],
    [0, 29, 29],
]
for i in range(n_shots):
    source_locations[i, 0, :] = torch.tensor(source_coords[i])

# Receivers: Full surface grid for every shot
rec_y, rec_x = torch.meshgrid(
    torch.arange(ny, device=device),
    torch.arange(nx, device=device),
    indexing="ij",
)
# Stack to (n_receivers, 3) -> z=0, y, x
rec_locs = torch.stack(
    [torch.zeros_like(rec_y), rec_y, rec_x], dim=-1
).reshape(-1, 3)
receiver_locations[:] = rec_locs

# Time parameters
freq = 25
nt = 150  # Short simulation for speed
dt = 0.004
peak_time = 1.5 / freq
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# PML width has 6 entries (2 for each dimension)
pml_width = [0, 20, 20, 20, 20, 20]  # Top is reflective

# Generate observed data using the true model
observed_data = scalar(
    v_true,
    [dz, dy, dx],
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=pml_width,
)[-1]

# Inversion loop
optimizer = torch.optim.SGD([v], lr=1e8)
loss_fn = torch.nn.MSELoss()

for _epoch in range(1):
    optimizer.zero_grad()

    # We use storage_mode='cpu' and storage_compression=True to reduce
    # GPU memory usage during the forward pass required for backpropagation.
    # This stores the intermediate data in CPU memory, compressed.
    out = scalar(
        v,
        [dz, dy, dx],
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=pml_width,
        storage_mode="cpu",
        storage_compression=True,
    )

    loss = loss_fn(out[-1], observed_data)
    loss.backward()
    optimizer.step()

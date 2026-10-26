"""Demonstrates callbacks for recording snapshots.

This script records snapshots of the forward and backward propagated
wavefields, and the velocity gradient, and creates an animation of them.
"""

import os

import matplotlib
import torch

import deepwave

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the model
# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
ny = 40 if smoke else 100
nx = 60 if smoke else 200
v_true = torch.ones(ny, nx, device=device) * 1500
v = v_true.clone()
v_true[ny // 2 :, :] = 2000
dx = 5.0
dt = 0.004
freq = 25
nt = 40 if smoke else 150
peak_time = 1.5 / freq

# Set up the source and receiver
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .to(device)
)
source_locations = torch.tensor(
    [[[0, nx // 2]]], dtype=torch.long, device=device
)
receiver_locations = torch.zeros(1, nx, 2, dtype=torch.long, device=device)
receiver_locations[0, :, 0] = 0
receiver_locations[0, :, 1] = torch.arange(nx)

# Generate true data
out_true = deepwave.scalar(
    v_true,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    max_vel=v_true.max().item(),
)

v.requires_grad_()

# Storage for snapshots from callbacks.
#
# Each pass calls back ceil(nt / callback_frequency) times, so the
# buffers are sized with a ceiling division. Floor division
# (`nt // callback_frequency`) is one short whenever `callback_frequency`
# does not divide `nt`, which is exactly the case you reach by raising
# `callback_frequency` on a long run.
callback_frequency = 1
n_snapshots = -(-nt // callback_frequency)
forward_snapshots = torch.zeros(n_snapshots, ny, nx)
backward_snapshots = torch.zeros(n_snapshots, ny, nx)
gradient_snapshots = torch.zeros(n_snapshots, ny, nx)


# Define callback functions


# A callable class, just to show how it is done
class ForwardCallback:
    """A callable class to store snapshots during forward propagation."""

    def __init__(self):
        """Initialises the ForwardCallback."""
        self.step = 0  # Forward propagation starts at time step 0

    def __call__(self, state):
        """A function called during the forward pass."""
        # We only need the wavefield for the first shot in the batch
        forward_snapshots[self.step] = (
            state.get_wavefield("wavefield_0")[0].cpu().clone()
        )
        self.step += 1


# Both directions now observe the same user-time grid. Counting invocations
# remains convenient; `state.step // callback_frequency` is also a valid
# snapshot index in either direction, even when the final interval is short
# or model-gradient histories use a different storage stride.
# def forward_callback(state):
#     forward_snapshots[state.step // callback_frequency] = (
#         state.get_wavefield("wavefield_0")[0].cpu().clone()
#     )


class BackwardCallback:
    """A callable class to store snapshots during backward propagation."""

    def __init__(self):
        """Initialises the BackwardCallback."""
        # The backward pass runs from the last time step to the first,
        # so the first call fills the last snapshot slot.
        self.index = n_snapshots - 1

    def __call__(self, state):
        """A function called during the backward pass."""
        # We use [0] to select the first shot in the batch.
        backward_snapshots[self.index] = (
            state.get_wavefield("wavefield_0")[0].cpu().clone()
        )
        gradient_snapshots[self.index] = (
            state.get_gradient("v")[0].cpu().clone()
        )
        self.index -= 1


# Run the propagation
out = deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    max_vel=v_true.max().item(),
    forward_callback=ForwardCallback(),
    backward_callback=BackwardCallback(),
    callback_frequency=callback_frequency,
)

# Backpropagate
torch.nn.MSELoss()(
    out.receiver_amplitudes, out_true.receiver_amplitudes
).backward()

# Create and save the animation
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

vmax_forward = torch.quantile(forward_snapshots, 0.99)
vmax_backward = torch.quantile(backward_snapshots, 0.99)
vmax_gradient = torch.quantile(gradient_snapshots, 0.99)

im1 = axes[0].imshow(
    forward_snapshots[-1],
    vmin=-vmax_forward,
    vmax=vmax_forward,
    cmap="seismic",
    animated=True,
)
im2 = axes[1].imshow(
    backward_snapshots[-1],
    vmin=-vmax_backward,
    vmax=vmax_backward,
    cmap="seismic",
    animated=True,
)
im3 = axes[2].imshow(
    gradient_snapshots[-1],
    vmin=-vmax_gradient,
    vmax=vmax_gradient,
    cmap="gray",
    animated=True,
)

axes[2].set_title("Gradient")

plt.tight_layout()

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])


def init():
    """Initialises the animation."""
    im1.set_data(forward_snapshots[-1])
    im2.set_data(backward_snapshots[-1])
    im3.set_data(gradient_snapshots[-1])
    return im1, im2, im3


def update(frame):
    """Updates the animation for each frame."""
    # Plot frames backwards in time
    idx = len(backward_snapshots) - 1 - frame
    im1.set_data(forward_snapshots[idx])
    im2.set_data(backward_snapshots[idx])
    im3.set_data(gradient_snapshots[idx])
    axes[0].set_title(f"Forward ({idx * callback_frequency * dt:.3f} s)")
    axes[1].set_title(f"Backward ({idx * callback_frequency * dt:.3f} s)")
    return im1, im2, im3


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(backward_snapshots),
    init_func=init,
    interval=10,
    blit=True,
)

ani.save("example_callback_animation.gif", writer="pillow")

assert torch.isfinite(v.grad).all(), "velocity gradient is not finite"
assert v.grad.abs().max() > 0, "velocity gradient is all zeros"

"""This script demonstrates the use of callbacks to create an animation of
wave propagation and gradient formation during Reverse-Time Migration (RTM).
"""

import torch
import deepwave
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the model
ny = 100
nx = 200
v_true = torch.ones(ny, nx, device=device) * 1500
v = v_true.clone()
v_true[ny // 2 :, :] = 2000
dx = 5.0
dt = 0.004
freq = 25
nt = 150
peak_time = 1.5 / freq

# Set up the source and receiver
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(device)
)
source_locations = torch.tensor([[[0, nx // 2]]], dtype=torch.long, device=device)
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

# Storage for snapshots from callbacks
callback_frequency = 1
forward_snapshots = torch.zeros(nt // callback_frequency, ny, nx)
backward_snapshots = torch.zeros(nt // callback_frequency, ny, nx)
gradient_snapshots = torch.zeros(nt // callback_frequency, ny, nx)


# Define callback functions


# A callable class, just to show how it is done
class ForwardCallback:
    def __init__(self):
        self.step = 0  # Forward propagation starts at time step 0

    def __call__(self, state):
        """A function called during the forward pass."""
        # We only need the wavefield for the first shot in the batch
        forward_snapshots[self.step] = (
            state.get_wavefield("wavefield_0")[0].cpu().clone()
        )
        self.step += 1


# Could instead have used a function
# def forward_callback(state):
#     forward_snapshots[state.step] = (
#         state.get_wavefield("wavefield_0")[0].cpu().clone()
#     )


def backward_callback(state):
    """A function called during the backward pass."""
    # We use [0] to select the first shot in the batch
    backward_snapshots[state.step] = state.get_wavefield("wavefield_0")[0].cpu().clone()
    gradient_snapshots[state.step] = state.get_gradient("v")[0].cpu().clone()


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
    backward_callback=backward_callback,
    callback_frequency=callback_frequency,
)

# Backpropagate
torch.nn.MSELoss()(out[-1], out_true[-1]).backward()

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
    im1.set_data(forward_snapshots[-1])
    im2.set_data(backward_snapshots[-1])
    im3.set_data(gradient_snapshots[-1])
    return im1, im2, im3


def update(frame):
    # Plot frames backwards in time
    idx = len(backward_snapshots) - 1 - frame
    im1.set_data(forward_snapshots[idx])
    im2.set_data(backward_snapshots[idx])
    im3.set_data(gradient_snapshots[idx])
    axes[0].set_title(f"Forward ({idx * callback_frequency * dt:.3f} s)")
    axes[1].set_title(f"Backward ({idx * callback_frequency * dt:.3f} s)")
    return im1, im2, im3


ani = animation.FuncAnimation(
    fig, update, frames=len(backward_snapshots), init_func=init, interval=10, blit=True
)

ani.save("example_callback_animation.gif", writer="pillow")

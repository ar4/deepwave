"""Demonstrates elastic wave propagation with a complex free surface."""

import matplotlib
import torch
import torchvision

import deepwave

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dx = 1.0
dt = 0.004
nt = 100
n_shots = 1
freq = 25
peak_time = 1.5 / freq
mask = (
    torchvision.io.read_image(
        "example_elastic_complex_freesurface.png",
        torchvision.io.ImageReadMode.GRAY,
    ).float()[0]
    / 255
).to(device)

vp = 2000 * mask
vs = 1500 * mask
rho = 2000 * mask

ny = vp.shape[-2]
nx = vp.shape[-1]

callback_frequency = 1
snapshots = []

out = deepwave.elastic(
    *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
    grid_spacing=dx,
    dt=dt,
    source_amplitudes_y=(
        deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1)
    ),
    source_locations_y=torch.tensor([[[ny // 2, nx // 2]]]),
    pml_width=0,
    forward_callback=lambda x: snapshots.append(
        x.get_wavefield("vy_0")[0].cpu().clone()
    ),
    callback_frequency=callback_frequency,
)

# Create and save the animation
fig = plt.figure(figsize=(4, 3))

vmax = torch.quantile(snapshots[-1], 0.99)

im = plt.imshow(
    snapshots[-1],
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
    animated=True,
)

plt.xticks([])
plt.yticks([])
plt.tight_layout()


def init():
    """Initialises the animation."""
    im.set_data(snapshots[-1])
    return [im]


def update(frame):
    """Updates the animation for each frame."""
    im.set_data(snapshots[frame])
    return [im]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(snapshots),
    init_func=init,
    interval=60,
    blit=True,
)

ani.save("example_elastic_complex_freesurface.gif", writer="pillow", dpi=72)

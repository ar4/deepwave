"""Demonstrates PyTorch checkpointing to reduce memory usage.

This script performs a simple Full-Waveform Inversion (FWI), using
PyTorch's checkpointing feature to reduce memory consumption during wave
propagation, at the cost of increased computation. The lever is the
``checkpoint_segments`` argument that every Deepwave propagator accepts.
"""

import math
import os

import matplotlib.pyplot as plt
import torch
from get_marmousi import load_data, load_vp
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from torchaudio.functional import biquad

import deepwave
from deepwave import scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
# Dimension 0 is depth and dimension 1 is horizontal position, the
# convention used throughout this documentation.
ny_full = 751
nx_full = 2301
dx = 4.0
# ``load_vp`` reads marmousi_vp.bin from the current directory and checks
# that it is in the depth-first layout Deepwave 0.1.0 writes (row 0 is the
# 1500 m/s water layer; an old-layout file holds the same number of floats
# and so would reshape without error, silently giving a rotated model). In
# smoke mode it returns a generated stand-in model of the same shape and
# velocity range instead, so no download is needed.
v_true_full = load_vp(smoke=smoke)

# Select portion of model for inversion
ny = 60 if smoke else 250
nx = 150 if smoke else 600
v_true = v_true_full[:ny, :nx]

# Smooth to use as starting model
v_init = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 40))

n_shots_full = 115

n_sources_per_shot = 1
d_source = 20  # 20 * 4m = 80m
first_source = 10  # 10 * 4m = 40m
source_depth = 2  # 2 * 4m = 8m

n_receivers_per_shot_full = 384
d_receiver = 6  # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 2  # 2 * 4m = 8m

freq = 25
nt_full = 750
dt = 0.004
peak_time = 1.5 / freq


# Select portion of data for inversion
n_shots = 4 if smoke else 20
n_receivers_per_shot = 20 if smoke else 100
nt = 80 if smoke else 300
# ``load_data`` takes the first n_shots shots, n_receivers_per_shot
# receivers and nt time samples of marmousi_data.bin. In smoke mode there is
# no downloaded model for that file to have come from, so it instead
# generates the data by propagating through the model above.
observed_data = load_data(
    v_true, n_shots, n_receivers_per_shot, nt, smoke=smoke
).to(device)

# source_locations
source_locations = torch.zeros(
    n_shots,
    n_sources_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
source_locations[..., 0] = source_depth
source_locations[:, 0, 1] = torch.arange(n_shots) * d_source + first_source

# receiver_locations
receiver_locations = torch.zeros(
    n_shots,
    n_receivers_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
receiver_locations[..., 0] = receiver_depth
receiver_locations[:, :, 1] = (
    torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
).repeat(n_shots, 1)

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

max_vel = 2500


# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    """A PyTorch module that represents the velocity model."""

    def __init__(self, initial, min_vel, max_vel):
        """Initialises the Model.

        Args:
            initial (torch.Tensor): The initial velocity model.
            min_vel (float): The minimum allowed velocity.
            max_vel (float): The maximum allowed velocity.
        """
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel)),
        )

    def forward(self):
        """Performs the forward pass of the model."""
        return (
            torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
            + self.min_vel
        )


model = Model(v_init, 1000, 2500).to(device)

# Setup optimiser to perform inversion
optimiser = torch.optim.LBFGS(
    model.parameters(), line_search_fn="strong_wolfe"
)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 1 if smoke else 2
n_segments = 2 if smoke else 5
cutoff_freq = 10

sos = butter(6, cutoff_freq, fs=1 / dt, output="sos")
sos = [
    torch.tensor(sosi).to(observed_data.dtype).to(device) for sosi in sos
]


def taper(x):
    """Applies a cosine taper to the end of the input tensor."""
    return deepwave.discretisation.cosine_taper_end(x, 100)


def filt(x):
    """Applies a Butterworth filter to the input tensor."""
    return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])


observed_data = filt(taper(observed_data))

loss_history: list[float] = []
for _epoch in range(n_epochs):

    def closure():
        """Closure function for the LBFGS optimiser."""
        optimiser.zero_grad()
        v = model()
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            checkpoint_segments=n_segments,
        )
        receiver_amplitudes = filt(taper(out.receiver_amplitudes))
        loss = 1e6 * loss_fn(receiver_amplitudes, observed_data)
        loss.backward()
        loss_history.append(float(loss.item()))
        return loss

    optimiser.step(closure)

# Plot
v = model()
vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
ax[0].imshow(v_init.cpu(), aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(
    v.detach().cpu(), aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu(), aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig("example_checkpointing.jpg")

v.detach().cpu().numpy().tofile("marmousi_v_inv.bin")

assert all(math.isfinite(x) for x in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

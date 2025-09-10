"""Demonstrates Full-Waveform Inversion (FWI) using Deepwave.

This script covers two approaches: a simple inversion and a more advanced
one incorporating constrained velocity and frequency filtering to address
cycle-skipping and stability issues.
"""

import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
from torchaudio.functional import biquad

import deepwave
from deepwave import scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny_full = 2301
nx_full = 751
dx = 4.0
v_true_full = torch.from_file("marmousi_vp.bin", size=ny_full * nx_full)
v_true_full = v_true_full.reshape(ny_full, nx_full)

# Select portion of model for inversion
ny = 600
nx = 250
v_true = v_true_full[:ny, :nx]

# Smooth to use as starting model
v_init = torch.tensor(1 / gaussian_filter(1 / v_true.numpy(), 40)).to(
    device
)
v = v_init.clone()
v.requires_grad_()

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

observed_data_full = torch.from_file(
    "marmousi_data.bin",
    size=n_shots_full * n_receivers_per_shot_full * nt_full,
).reshape(n_shots_full, n_receivers_per_shot_full, nt_full)

# Select portion of data for inversion
n_shots = 20
n_receivers_per_shot = 100
nt = 300
observed_data = observed_data_full[
    :n_shots,
    :n_receivers_per_shot,
    :nt,
].to(device)

# source_locations
source_locations = torch.zeros(
    n_shots,
    n_sources_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

# receiver_locations
receiver_locations = torch.zeros(
    n_shots,
    n_receivers_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = (
    torch.arange(n_receivers_per_shot) * d_receiver + first_receiver
).repeat(n_shots, 1)

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)


## First attempt: simple inversion


# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([v], lr=1e9, momentum=0.9)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 250

for _epoch in range(n_epochs):
    optimiser.zero_grad()
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
    )
    loss = loss_fn(out[-1], observed_data)
    loss.backward()
    torch.nn.utils.clip_grad_value_(
        v, torch.quantile(v.grad.detach().abs(), 0.98)
    )
    optimiser.step()

# Plot
vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
ax[0].imshow(
    v_init.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[0].set_title("Initial")
ax[1].imshow(
    v.detach().cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[1].set_title("Out")
ax[2].imshow(
    v_true.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig("example_simple_fwi.jpg")


## Second attempt: constrained velocity and frequency filtering


# Define a function to taper the ends of traces
def taper(x):
    """Tapers the ends of traces using a cosine taper."""
    return deepwave.common.cosine_taper_end(x, 100)


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


observed_data = taper(observed_data)
model = Model(v_init, 1000, 2500).to(device)

# Run optimisation/inversion
n_epochs = 2

for cutoff_freq in [10, 15, 20, 25, 30]:
    sos = butter(6, cutoff_freq, fs=1 / dt, output="sos")
    sos = [
        torch.tensor(sosi).to(observed_data.dtype).to(device)
        for sosi in sos
    ]

    def filt(x, sos):
        """Applies a Butterworth filter to the input tensor."""
        return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])

    observed_data_filt = filt(observed_data, sos)
    optimiser = torch.optim.LBFGS(
        model.parameters(), line_search_fn="strong_wolfe"
    )
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
                max_vel=2500,
                pml_freq=freq,
                time_pad_frac=0.2,
            )
            out_filt = filt(taper(out[-1]), sos)
            loss = 1e6 * loss_fn(out_filt, observed_data_filt)
            loss.backward()
            return loss

        optimiser.step(closure)

v = model()
vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
ax[0].imshow(
    v_init.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[0].set_title("Initial")
ax[1].imshow(
    v.detach().cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[1].set_title("Out")
ax[2].imshow(
    v_true.cpu().T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax
)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig("example_increasing_freq_fwi.jpg")

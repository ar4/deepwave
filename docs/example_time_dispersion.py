"""Removing temporal dispersion with deepwave.time_dispersion.

Explicit second-order leapfrog time stepping makes each temporal frequency
propagate slightly too slowly (temporal dispersion), an error controlled
only by the time step and not by the spatial ``accuracy`` order. This
example runs the same homogeneous 2-D scalar model at a coarse time step
(near the CFL limit) and at a fine reference time step, and shows that
the Koene-Robertsson
correction -- pre-transforming the source and post-transforming the receiver
data -- brings the coarse-dt trace into close agreement with the fine-dt
reference.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import scalar
from deepwave.discretisation import downsample, upsample
from deepwave.time_dispersion import (
    cfl_internal_dt,
    correct_receivers,
    correct_source,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# Homogeneous model
v0 = 1500.0
ny, nx = 180, 90
dx = 8.0
grid = [dx, dx]
v = torch.full((ny, nx), v0, device=device)

freq = 25.0
dt_coarse = 0.0022  # near the CFL limit -> step_ratio == 1
ratio = 6
dt_fine = dt_coarse / ratio
# The source-receiver distance is 80 cells * 8m = 640m at v0=1500m/s, i.e.
# a travel time of ~0.427s, so t_total must stay comfortably above that in
# both modes for the wave to have reached the receiver.
t_total = 0.6 if smoke else 0.72
nt_coarse = round(t_total / dt_coarse)

# The step that actually governs the dispersion is the propagator's internal
# time step. Here dt_coarse is CFL-stable, so step_ratio == 1 and
# internal_dt == dt_coarse; cfl_internal_dt confirms it.
internal_dt = cfl_internal_dt(grid, dt_coarse, v0)

# A single source and a single distant receiver
source_locations = torch.tensor([[[20, 45]]], device=device)
receiver_locations = torch.tensor([[[100, 45]]], device=device)


def run(dt: float, wavelet: torch.Tensor) -> torch.Tensor:
    """Forward model and return the single receiver trace."""
    out = scalar(
        v,
        grid,
        dt,
        source_amplitudes=wavelet.reshape(1, 1, -1),
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=20,
        pml_freq=freq,
    )
    return out.receiver_amplitudes.reshape(-1)


# Coarse-dt Ricker wavelet, and the SAME continuous wavelet resampled to the
# fine grid (so the only difference between the runs is the time step).
wavelet_coarse = deepwave.wavelets.ricker(
    freq, nt_coarse, dt_coarse, peak_time=1.6 / freq
).to(device)
wavelet_fine = upsample(wavelet_coarse.reshape(1, -1), ratio).reshape(-1)

# Fine-dt reference, downsampled to the coarse grid for comparison
rec_fine = downsample(
    run(dt_fine, wavelet_fine).reshape(1, -1), ratio
).reshape(-1)

# Coarse-dt trace without any correction
rec_coarse = run(dt_coarse, wavelet_coarse)

# Coarse-dt trace WITH the correction: pre-correct the source, model, then
# post-correct the recorded data.
wavelet_corrected = correct_source(wavelet_coarse, internal_dt)
rec_corrected = correct_receivers(
    run(dt_coarse, wavelet_corrected), internal_dt
)

n = min(len(rec_fine), len(rec_coarse), len(rec_corrected))
rec_fine = rec_fine[:n]
rec_coarse = rec_coarse[:n]
rec_corrected = rec_corrected[:n]


def rel_l2(a: torch.Tensor) -> float:
    """Relative L2 error of ``a`` versus the fine-dt reference."""
    return (a - rec_fine).norm().item() / rec_fine.norm().item()


err_uncorrected = rel_l2(rec_coarse)
err_corrected = rel_l2(rec_corrected)
print(f"relative L2 error, uncorrected: {err_uncorrected:.4f}")
print(f"relative L2 error, corrected:   {err_corrected:.4f}")
print(f"improvement factor: {err_uncorrected / err_corrected:.1f}x")

# Plot the three traces
t = torch.arange(n).cpu() * dt_coarse
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, rec_fine.cpu(), "k-", label="fine-dt reference")
ax.plot(t, rec_coarse.cpu(), "C3--", label="coarse-dt, uncorrected")
ax.plot(t, rec_corrected.cpu(), "C0:", label="coarse-dt, corrected")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Koene-Robertsson time-dispersion correction")
ax.legend()
fig.tight_layout()
fig.savefig("example_time_dispersion.jpg")

assert math.isfinite(err_uncorrected), (
    "uncorrected relative L2 error is not finite"
)
assert math.isfinite(err_corrected), (
    "corrected relative L2 error is not finite"
)
assert err_corrected < err_uncorrected, (
    "time-dispersion correction did not reduce the relative L2 error"
)

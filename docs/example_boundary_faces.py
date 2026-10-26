"""Demonstrates the first-class ``boundary`` per-face argument.

In addition to the default all-PML (absorbing) edges, every propagator
accepts an explicit ``boundary`` argument selecting a boundary *type* per
face: "pml" (absorbing), "free" (pressure-release, inverted-polarity
reflection, R = -1), "rigid" (preserved-polarity reflection, R = +1), or
"periodic" (wraps to the opposite face). Which of those a given propagator
accepts differs -- ``scalar``, used here, accepts all four on every face,
while the other propagators accept fewer; see :doc:`boundaries` for the
table, and for the full derivation of each condition's reflection
coefficient and image formula.

This example shows both effects on a small homogeneous 2-D model:

1. A source placed near one edge, comparing the receiver trace's
   reflected arrival for a "free" vs a "rigid" boundary against a "pml"
   (absorbing, no reflection) baseline on the same edge -- confirming the
   opposite reflection polarity.
2. A snapshot of the wavefield with "periodic" boundaries on a pair of
   faces, showing energy that has left one edge re-entering the opposite
   edge travelling in the same direction.
"""

import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

v0 = 1500.0
dx = 10.0
freq = 25.0
accuracy = 4
ny = nx = 50
peak_time = 1.5 / freq
dt = 0.001

v = torch.full((ny, nx), v0, device=device)

# --- 1. Free vs rigid reflection polarity -----------------------------
# A source placed close to the last face of the second dimension, with a
# co-located receiver to record the direct arrival followed by that
# face's reflection.
nt = 140 if smoke else 200
sy, sx = ny // 2, nx - 8
source_locations = torch.tensor([[[sy, sx]]], device=device)
receiver_locations = torch.tensor([[[sy, sx]]], device=device)
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .to(device)
)


def run_trace(boundary, pml_width) -> torch.Tensor:
    """Forward model and return the single receiver trace."""
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=freq,
        boundary=boundary,
        backend="eager",
        max_vel=v0,
    )
    return out.receiver_amplitudes.reshape(-1)


trace_pml = run_trace(None, 15)
trace_free = run_trace(["pml", "pml", "pml", "free"], [15, 15, 15, 0])
trace_rigid = run_trace(["pml", "pml", "pml", "rigid"], [15, 15, 15, 0])

# Subtracting the (reflection-free) pml baseline isolates the reflected
# arrival, which the direct arrival otherwise dominates.
reflected_free = trace_free - trace_pml
reflected_rigid = trace_rigid - trace_pml

# --- 2. Periodic wraparound --------------------------------------------
nt_periodic = 160 if smoke else 220
source_locations_periodic = torch.tensor([[[sy, 8]]], device=device)
source_amplitudes_periodic = (
    deepwave.wavelets.ricker(freq, nt_periodic, dt, peak_time)
    .reshape(1, 1, -1)
    .to(device)
)
out_periodic = scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes_periodic,
    source_locations=source_locations_periodic,
    accuracy=accuracy,
    pml_width=[15, 15, 0, 0],
    pml_freq=freq,
    boundary=["pml", "pml", "periodic", "periodic"],
    backend="eager",
    max_vel=v0,
)
wavefield_periodic = out_periodic.wavefield_0[0].detach().cpu()

_, ax = plt.subplots(1, 2, figsize=(9, 4))
t = torch.arange(nt) * dt
ax[0].plot(t, reflected_free.cpu(), label="free (R=-1)")
ax[0].plot(t, reflected_rigid.cpu(), label="rigid (R=+1)")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("pml-subtracted (reflected-only) amplitude")
ax[0].set_title("Free vs rigid reflection polarity")
ax[0].legend()

vmax = float(wavefield_periodic.abs().max()) * 0.5
ax[1].imshow(wavefield_periodic, cmap="seismic", vmin=-vmax, vmax=vmax)
ax[1].set_title("Periodic (x) wraparound")

plt.tight_layout()
plt.savefig("example_boundary_faces.jpg")

assert torch.isfinite(reflected_free).all(), (
    "reflected_free trace is not finite"
)
assert reflected_free.abs().max() > 0, "reflected_free trace is all zeros"
assert torch.isfinite(wavefield_periodic).all(), (
    "periodic wavefield is not finite"
)
assert wavefield_periodic.abs().max() > 0, "periodic wavefield is all zeros"

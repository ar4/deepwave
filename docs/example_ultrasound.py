"""Focused ultrasound: a delay-focused array and a fluid-solid interface.

Everything here is in one unit system: millimetres, microseconds and
megahertz. Nothing converts between unit systems, because there is nothing
to convert -- Deepwave only ever combines these numbers in dimensionless
groups.

Part 1 builds the two excitation wavelets ultrasound and NDT use. Part 2
drives a linear array element with a focusing delay law and measures where
the beam really focuses. Part 3 puts water over bone and measures the
fluid-solid reflection coefficient.
"""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import elastic, scalar, transducers
from deepwave.wavelets import gaussian_pulse, tone_burst

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly
# as an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# --- section: units ---
# One unit system throughout: mm, microseconds, MHz.
dx = 0.05  # mm cells (50 um): 10 cells per wavelength at 3 MHz in water
dt = 0.01  # microseconds (10 ns)
v_water = 1.5  # mm/us, i.e. 1500 m/s
rho_water = 1.0  # water density, in whatever mass unit you like
freq = 3.0  # MHz: the transducer's centre frequency

wavelength = v_water / freq  # 0.5 mm
cells_per_wavelength = wavelength / dx  # 10
courant = v_water * dt / dx  # 0.3
print(f"{cells_per_wavelength:.1f} cells/wavelength, Courant {courant:.2f}")

# `transducers` works in grid cells, so the speed it wants is a speed in
# cells per unit time. That division is the only conversion in this script.
c_cells = v_water / dx  # 30 cells/us

# --- section: wavelets ---
# A tone burst and a Gaussian pulse: the two standard excitations.
# `peak_time` is given explicitly. Left out, every wavelet builder centres
# itself in the record (`peak_time = nt * dt / 2`), which is what a plot
# wants but wastes half the run when the wavelet is only there to start a
# simulation, so these two are placed at the front of their records.
n_cycles = 3
burst_duration = n_cycles / freq  # 1 us
peak_time = burst_duration / 2
nt_wavelet = 200  # 2 us of record, for the plot
burst = tone_burst(
    freq=freq,
    n_cycles=n_cycles,
    nt=nt_wavelet,
    dt=dt,
    peak_time=peak_time,
)
pulse = gaussian_pulse(
    freq=freq,
    bandwidth_frac=0.6,  # fractional -6 dB bandwidth (scipy convention)
    nt=nt_wavelet,
    dt=dt,
    peak_time=peak_time,
)

# Both builders hit the frequency you asked for. `rfftfreq`'s `d` is the
# sample spacing in microseconds, so the frequency axis is in MHz.
wavelet_freqs = torch.fft.rfftfreq(nt_wavelet, d=dt)
burst_bin = int(torch.fft.rfft(burst).abs().argmax())
pulse_bin = int(torch.fft.rfft(pulse).abs().argmax())
print(
    f"peak frequency: tone burst {wavelet_freqs[burst_bin]:.2f} MHz, "
    f"Gaussian pulse {wavelet_freqs[pulse_bin]:.2f} MHz"
)

# --- section: array ---
# A linear array element, focused by a delay law. Sizes are in mm and are
# converted to cells once, here.
aperture_mm = 4.0 if smoke else 6.0
focal_depth_mm = 4.0 if smoke else 6.0
aperture = aperture_mm / dx  # cells
focal_depth = focal_depth_mm / dx  # cells
f_number = focal_depth / aperture  # 1.0: a strongly focused aperture

array_y = 12.0  # the array face, 12 cells below the top of the model
patch = 30 if smoke else 40  # half-width in cells of the measured patch
margin = patch + 12.0  # water beyond the focus and beside the aperture
ny = int(array_y + focal_depth + margin)
nx = int(aperture + 2 * margin) + 1
centre_x = (nx - 1) / 2
focus_y = array_y + focal_depth
pml_width = 20

# The element is tessellated at half-cell spacing, so most of its points
# sit off the grid nodes; the propagator band-limit-interpolates them onto
# the grid with Hicks interpolation automatically.
locations, weights = transducers.line_element(
    centre=[array_y, centre_x],
    direction=[0.0, 1.0],
    length=aperture,
    spacing=0.5,
    window="hann",  # apodised, to suppress the beam's side lobes
)

# `c` is in cells per unit time, so the delays come back in microseconds,
# the same unit as `dt`.
delays = transducers.delays_to_focus(
    locations,
    focus=[focus_y, centre_x],
    c=c_cells,
)

# Every contribution is timed to reach the focus together, at the moment
# the farthest element point's wavelet gets there.
focus_point = torch.tensor([focus_y, centre_x])
range_cells = torch.linalg.vector_norm(locations - focus_point, dim=1)
t_focus = peak_time + float(range_cells.max()) / c_cells
nt = round((t_focus + burst_duration) / dt)
print(
    f"{locations.shape[0]} element points, f-number {f_number:.1f}, "
    f"focal arrival {t_focus:.3f} us, {nt} time samples"
)

# `element_source` applies the apodisation weights and the delay law and
# returns the pair a propagator takes. Assembling it by hand means
# remembering `* weights[:, None]`, whose omission does not raise: it just
# quietly changes the beam pattern.
drive = tone_burst(
    freq=freq,
    n_cycles=n_cycles,
    nt=nt,
    dt=dt,
    peak_time=peak_time,
)
source_amplitudes, source_locations = transducers.element_source(
    drive,
    locations,
    weights,
    delays=delays,
    dt=dt,
)

# A patch of receivers around the focus, to measure the beam where it
# matters. Integer cells: no interpolation, no extra cost.
patch_y = torch.arange(int(focus_y) - patch, int(focus_y) + patch + 1)
patch_x = torch.arange(int(centre_x) - patch, int(centre_x) + patch + 1)
grid_y, grid_x = torch.meshgrid(patch_y, patch_x, indexing="ij")
receiver_locations = torch.stack(
    [grid_y.reshape(-1), grid_x.reshape(-1)],
    dim=1,
).unsqueeze(0)

v = torch.full((ny, nx), v_water)


def run_array(amplitudes):
    """Propagate one array excitation and return the receiver patch."""
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=pml_width,
        # Always set pml_freq. In this unit system it is in MHz.
        pml_freq=freq,
        accuracy=8,
    )
    n_y, n_x = len(patch_y), len(patch_x)
    return out.receiver_amplitudes[0].reshape(n_y, n_x, nt)


focused = run_array(source_amplitudes)

# The same element with no delay law at all, for comparison: an unfocused,
# apodised aperture radiating a plane wave.
plane_amplitudes, _ = transducers.element_source(drive, locations, weights)
unfocused = run_array(plane_amplitudes)

# --- section: focus-checks ---
# The peak pressure each cell ever sees: how a focused field is normally
# measured, and what a hydrophone scan reports.
peak_focused = focused.abs().amax(dim=-1)
peak_unfocused = unfocused.abs().amax(dim=-1)


def minus_6db_width(profile):
    """Width in cells of the region where a profile is within -6 dB."""
    above = (profile >= profile.max() / 2).nonzero().flatten()
    return float(above[-1] - above[0]) + 1.0


def envelope(trace):
    """The analytic-signal envelope of a real trace."""
    n_time = trace.shape[-1]
    spectrum = torch.fft.fft(trace)
    weight = torch.zeros(n_time, dtype=spectrum.dtype)
    weight[0] = 1
    weight[1 : (n_time + 1) // 2] = 2
    if n_time % 2 == 0:
        weight[n_time // 2] = 1
    return torch.fft.ifft(spectrum * weight).abs()


focal_row = int(focus_y) - int(patch_y[0])
focal_col = int(centre_x) - int(patch_x[0])
lateral_focused = peak_focused[focal_row]
lateral_unfocused = peak_unfocused[focal_row]
width_focused = minus_6db_width(lateral_focused)
width_unfocused = minus_6db_width(lateral_unfocused)

# Laterally, the peak-pressure maximum is at the focus. This is the
# direction in which a focused aperture localises sharply: the -6 dB beam
# width is of order the wavelength times the f-number.
lateral_peak_bin = int(peak_focused.max(dim=0).values.argmax())
lateral_peak_x = float(patch_x[lateral_peak_bin])
print(
    f"lateral peak at x = {lateral_peak_x:.0f} cells (focus at "
    f"{centre_x:.0f}); -6 dB width {width_focused:.0f} cells focused, "
    f"{width_unfocused:.0f} unfocused"
)

# Axially, the delay law is confirmed by *time of flight*, not by an
# amplitude peak. The depth of field of a focused aperture is roughly
# 7 * wavelength * f_number**2 cells -- about 70 cells here -- so the
# on-axis peak pressure is nearly flat through the focal region, and as
# with any focused aperture its maximum sits well in front of the
# geometric focus. The arrival time is sharp instead: the focal trace's
# envelope peaks when the delay law says it should.
focal_trace = focused[focal_row, focal_col]
t_measured = float(envelope(focal_trace).argmax()) * dt
one_cell_of_travel = dx / v_water  # us
axial_error = abs(t_measured - t_focus) / one_cell_of_travel  # cells
print(
    f"focal arrival: predicted {t_focus:.3f} us, measured "
    f"{t_measured:.3f} us ({axial_error:.2f} cells of travel apart)"
)

assert torch.isfinite(peak_focused).all(), "focal field is not finite"
assert torch.isfinite(peak_unfocused).all(), "plane field is not finite"
assert abs(lateral_peak_x - centre_x) <= 1.0, (
    f"lateral peak at {lateral_peak_x}, over one cell from the focus at "
    f"{centre_x}"
)
assert axial_error <= 1.0, (
    f"focal arrival is {axial_error:.2f} cells of travel from the instant "
    f"the delay law predicts"
)
assert width_focused < aperture / 3, (
    f"the beam is {width_focused} cells wide at the focal depth, not "
    f"focused down from the {aperture:.0f}-cell aperture"
)
assert width_unfocused > 1.5 * width_focused, (
    f"dropping the delay law did not broaden the beam "
    f"({width_unfocused} vs {width_focused} cells)"
)

# --- section: fluid-solid ---
# Water over cortical bone. A fluid has no shear modulus, so its vs is 0;
# Deepwave's harmonic averaging of mu then imposes the fluid-solid
# condition (normal stress transmitted, no shear traction) for you.
vp_bone, vs_bone, rho_bone = 3.6, 1.8, 1.9  # mm/us, mm/us, and rho
interface = 100  # cells below the top of the model
ny_fs = 160 if smoke else 200
nx_fs = 121

vp = torch.full((ny_fs, nx_fs), v_water)
vs = torch.zeros(ny_fs, nx_fs)
rho = torch.full((ny_fs, nx_fs), rho_water)
vp[interface:] = vp_bone
vs[interface:] = vs_bone
rho[interface:] = rho_bone
lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

source_y, receiver_y, fs_x = 30, 50, nx_fs // 2
nt_fs = 600
fs_wavelet = gaussian_pulse(
    freq=freq,
    bandwidth_frac=0.6,
    nt=nt_fs,
    dt=dt,
    peak_time=peak_time,
    shape=(1, 1),  # [n_shots, n_sources, nt], with nothing copied
)
fs_out = elastic(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    source_amplitudes_pressure_rate=fs_wavelet,
    source_locations_pressure_rate=torch.tensor([[[source_y, fs_x]]]),
    receiver_locations_p=torch.tensor([[[receiver_y, fs_x]]]),
    pml_width=pml_width,
    pml_freq=freq,
    accuracy=4,
)
fs_trace = fs_out.receiver_amplitudes_p[0, 0]
fs_envelope = envelope(fs_trace)

# Direct arrival and interface reflection, separated in time. Taking the
# *difference* of their arrival times cancels whatever time the source
# waveform itself takes to reach its peak.
direct_cells = receiver_y - source_y
reflect_cells = (interface - source_y) + (interface - receiver_y)
gap_samples = (reflect_cells - direct_cells) / c_cells / dt
split = round((direct_cells + reflect_cells) / 2 / c_cells / dt)
i_direct = int(fs_envelope[:split].argmax())
i_reflect = split + int(fs_envelope[split:].argmax())
gap_error = abs((i_reflect - i_direct) - gap_samples)

# The normal-incidence reflection coefficient, corrected for the
# 1/sqrt(range) spreading of a 2-D wave over the two path lengths.
spreading = math.sqrt(reflect_cells / direct_cells)
ratio = float(fs_envelope[i_reflect] / fs_envelope[i_direct])
r_measured = ratio * spreading
z_water = rho_water * v_water
z_bone = rho_bone * vp_bone
r_predicted = (z_bone - z_water) / (z_bone + z_water)
print(
    f"fluid-solid: reflection {i_reflect - i_direct} samples after the "
    f"direct arrival (predicted {gap_samples:.0f}); R = {r_measured:.3f} "
    f"(predicted {r_predicted:.3f})"
)

assert torch.isfinite(fs_trace).all(), "fluid-solid trace is not finite"
assert float(mu[:interface].abs().max()) == 0.0, (
    "the fluid region should carry no shear modulus"
)
assert gap_error <= 3, (
    f"the reflection arrived {gap_error:.0f} samples from its predicted "
    f"two-way time"
)
assert abs(r_measured - r_predicted) < 0.1 * r_predicted, (
    f"measured reflection coefficient {r_measured:.3f} is not within 10% "
    f"of the impedance-contrast prediction {r_predicted:.3f}"
)

# --- section: plot ---
time_axis = torch.arange(nt_wavelet) * dt
extent = [
    float(patch_x[0]) * dx,
    float(patch_x[-1]) * dx,
    float(patch_y[-1]) * dx,
    float(patch_y[0]) * dx,
]
_, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].plot(time_axis, burst, label="tone burst")
ax[0, 0].plot(time_axis, pulse, "--", label="Gaussian pulse")
ax[0, 0].set_xlabel("Time (us)")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].set_title(f"{freq:g} MHz excitation wavelets")
ax[0, 0].legend()

ax[0, 1].imshow(peak_focused, extent=extent, cmap="magma", aspect="equal")
ax[0, 1].plot(centre_x * dx, focus_y * dx, "w+", markersize=12)
ax[0, 1].set_xlabel("x (mm)")
ax[0, 1].set_ylabel("Depth (mm)")
ax[0, 1].set_title("Peak pressure (+ marks the geometric focus)")

peak_scale = lateral_focused.max()
ax[1, 0].plot(patch_x * dx, lateral_focused / peak_scale, label="focused")
ax[1, 0].plot(
    patch_x * dx,
    lateral_unfocused / peak_scale,
    "--",
    label="no delay law",
)
ax[1, 0].set_xlabel("x (mm)")
ax[1, 0].set_ylabel("Peak pressure (normalised)")
ax[1, 0].set_title(f"Beam profile at the focus ({focus_y * dx:g} mm)")
ax[1, 0].legend()

ax[1, 1].plot(torch.arange(nt_fs) * dt, fs_trace)
ax[1, 1].set_xlabel("Time (us)")
ax[1, 1].set_ylabel("Pressure")
ax[1, 1].set_title(f"Water over bone: R = {r_measured:.2f}")

plt.tight_layout()
plt.savefig("example_ultrasound.jpg")
plt.show()

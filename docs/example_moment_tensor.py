"""Builds an earthquake (moment tensor) source from body force couples."""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import elastic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"

# ---------------------------------------------------------------------
# 1. Fault geometry to moment tensor
# ---------------------------------------------------------------------
# Convention (Aki & Richards, Quantitative Seismology, 1980):
#
#   Frame: right-handed Cartesian (x1, x2, x3) = (North, East, Down).
#          Depth increases downwards, so x3 is positive downwards.
#   strike: azimuth of the fault's horizontal trace, in degrees, measured
#          clockwise from North, in the range [0, 360). The fault dips to
#          the right of the strike direction.
#   dip:   angle of the fault plane below horizontal, in degrees, in the
#          range [0, 90].
#   rake:  direction of slip of the hanging wall relative to the foot
#          wall, measured within the fault plane, in degrees, anticlockwise
#          from the strike direction, in the range (-180, 180]. rake = 0 is
#          left-lateral strike slip, rake = 90 is pure reverse (thrust)
#          slip, rake = -90 is pure normal slip.
#   m0:    scalar moment. Units of force times length (N m in SI). The
#          simulation below is 2-D, which represents a source that is
#          uniform and infinite in the direction out of the model plane,
#          so m0 here is a moment per unit length out of the plane
#          (N m / m, i.e. N). The value 1e12 N m / m is, for instance, a
#          moment of 1e15 N m (about magnitude 4) spread over a 1 km fault
#          length.
strike = 90.0
dip = 60.0
rake = 90.0
m0 = 1e12

strike_rad = math.radians(strike)
dip_rad = math.radians(dip)
rake_rad = math.radians(rake)

# The unit slip vector and the unit fault normal, in (North, East, Down).
slip = torch.tensor(
    [
        math.cos(rake_rad) * math.cos(strike_rad)
        + math.cos(dip_rad) * math.sin(rake_rad) * math.sin(strike_rad),
        math.cos(rake_rad) * math.sin(strike_rad)
        - math.cos(dip_rad) * math.sin(rake_rad) * math.cos(strike_rad),
        -math.sin(rake_rad) * math.sin(dip_rad),
    ],
    dtype=torch.float64,
)
normal = torch.tensor(
    [
        -math.sin(dip_rad) * math.sin(strike_rad),
        math.sin(dip_rad) * math.cos(strike_rad),
        -math.cos(dip_rad),
    ],
    dtype=torch.float64,
)

# The moment tensor of a shear dislocation is the symmetric outer product
# of the slip vector and the fault normal, scaled by the scalar moment.
moment_tensor = m0 * (torch.outer(slip, normal) + torch.outer(normal, slip))

# The same thing written out component by component, which is the form the
# relations are usually quoted in. The two must agree. The named products
# below are the combinations of angle functions that recur in them.
sin_dip_cos_rake = math.sin(dip_rad) * math.cos(rake_rad)
sin_2dip_sin_rake = math.sin(2 * dip_rad) * math.sin(rake_rad)
cos_dip_cos_rake = math.cos(dip_rad) * math.cos(rake_rad)
cos_2dip_sin_rake = math.cos(2 * dip_rad) * math.sin(rake_rad)
moment_tensor_components = torch.zeros(3, 3, dtype=torch.float64)
moment_tensor_components[0, 0] = -m0 * (
    sin_dip_cos_rake * math.sin(2 * strike_rad)
    + sin_2dip_sin_rake * math.sin(strike_rad) ** 2
)
moment_tensor_components[1, 1] = m0 * (
    sin_dip_cos_rake * math.sin(2 * strike_rad)
    - sin_2dip_sin_rake * math.cos(strike_rad) ** 2
)
moment_tensor_components[2, 2] = m0 * sin_2dip_sin_rake
moment_tensor_components[0, 1] = m0 * (
    sin_dip_cos_rake * math.cos(2 * strike_rad)
    + sin_2dip_sin_rake * math.sin(2 * strike_rad) / 2
)
moment_tensor_components[0, 2] = -m0 * (
    cos_dip_cos_rake * math.cos(strike_rad)
    + cos_2dip_sin_rake * math.sin(strike_rad)
)
moment_tensor_components[1, 2] = -m0 * (
    cos_dip_cos_rake * math.sin(strike_rad)
    - cos_2dip_sin_rake * math.cos(strike_rad)
)
moment_tensor_components[1, 0] = moment_tensor_components[0, 1]
moment_tensor_components[2, 0] = moment_tensor_components[0, 2]
moment_tensor_components[2, 1] = moment_tensor_components[1, 2]

# ---------------------------------------------------------------------
# 2. From the (North, East, Down) frame to Deepwave's grid
# ---------------------------------------------------------------------
# Deepwave orders its spatial dimensions slowest to fastest, (z, y, x) in
# 3-D and (y, x) in 2-D, and attaches no physical meaning to them: you
# choose. Here the model is a vertical cross-section in which
#
#   Deepwave y (first dimension)  = Down  = the frame's third axis,
#   Deepwave x (second dimension) = North = the frame's first axis.
#
# The fault above strikes East-West, so this cross-section is
# perpendicular to the strike and contains both the slip vector and the
# fault normal. A 2-D P-SV simulation can only carry the components of
# the moment tensor that lie in the model plane; the components involving
# the out-of-plane axis (East here) drive SH motion, which this
# simulation does not model.
in_plane = [2, 0]  # (Down, North) rows and columns of the moment tensor
moment_tensor_2d = moment_tensor[in_plane][:, in_plane]

# ---------------------------------------------------------------------
# 3. Model and survey
# ---------------------------------------------------------------------
vp = 3000.0
vs = 1500.0
rho = 2200.0
dx = 10.0
dt = 0.002
freq = 15.0
peak_time = 1.5 / freq
pml_width = 20

# Receivers sit on a circle around the source. The radius is measured in
# grid cells: 100 cells is 1000 m, five P wavelengths, which is far enough
# from the source for the far-field radiation pattern to have formed.
radius = 50 if smoke else 100
n_receivers = 36 if smoke else 72
# Enough time steps to reach the S wave travel time at that radius.
nt = 170 if smoke else 340

half_width = radius + pml_width + 5
ny = nx = 2 * half_width + 1
source_y = source_x = half_width

lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(
    torch.full((ny, nx), vp, device=device),
    torch.full((ny, nx), vs, device=device),
    torch.full((ny, nx), rho, device=device),
)

# The wavelet is the moment history M(t): the source amplitudes below are
# proportional to the moment itself, not to its time derivative.
wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time).to(device)

angles = torch.arange(n_receivers, dtype=torch.float64) / n_receivers
angles = angles * 2 * math.pi
# Unit vector from the source towards each receiver, in Deepwave's (y, x)
# order, so the angle is measured from the downward axis towards North.
direction = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
receiver_locations = (
    (
        torch.tensor([source_y, source_x], dtype=torch.float64)
        + radius * direction
    )
    .to(torch.float32)[None]
    .to(device)
)

# ---------------------------------------------------------------------
# 4. Moment tensor to equivalent body force couples
# ---------------------------------------------------------------------
# A moment tensor component M_ij is a force couple: two opposing point
# forces in direction i, separated by a small distance along direction j.
# The equivalent body force density of a point moment tensor source at
# position s is f_i = -d/dx_j (M_ij delta(x - s)), so each couple is a
# finite difference approximation of a derivative of a point source: the
# force on the +j side of s is +M_ij, the force on the -j side is -M_ij,
# and the pair is divided by their separation (the lever arm) because a
# difference approximates a derivative only after dividing by the step.
# Deepwave's velocity sources are force *densities* (force per unit
# volume, see the units documentation), so each force is also divided by
# the volume of a grid cell. In 2-D that volume is dx * dx per unit
# length out of the plane.
#
# The two forces of a couple are placed half a cell either side of the
# source position, so their separation is one cell, dx. Deepwave reads
# float locations as physical positions in grid cells and removes each
# component's own staggered-grid half-cell offset itself: vy lives at
# (y + 1/2, x) and vx at (y, x + 1/2), so a vy force asked for at
# (y + 1/2, x) lands exactly on a vy node and is injected exactly, while
# one asked for at (y, x + 1/2) falls between nodes and is interpolated
# (Hicks, a windowed sinc). Working in physical positions means you do
# not have to do that bookkeeping yourself.
separation = torch.tensor(
    [[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]],
    dtype=torch.float64,
)
couple_locations = (
    (torch.tensor([source_y, source_x], dtype=torch.float64) + separation)
    .to(torch.float32)[None]
    .to(device)
)
lever_arm = dx
cell_volume = dx * dx


def couple_amplitudes(moment: torch.Tensor) -> tuple:
    """Body force density amplitudes for a 2-D moment tensor.

    Returns the amplitudes of the four forces in the y direction and the
    four in the x direction, in the order of ``couple_locations``.
    """
    # The first two positions are separated along y and so carry the
    # couples of M_yy (forces in y) and M_xy (forces in x); the second two
    # are separated along x and carry M_yx and M_xx.
    weights_y = torch.tensor(
        [moment[0, 0], -moment[0, 0], moment[0, 1], -moment[0, 1]],
        dtype=torch.float64,
    )
    weights_x = torch.tensor(
        [moment[1, 0], -moment[1, 0], moment[1, 1], -moment[1, 1]],
        dtype=torch.float64,
    )
    scale = 1.0 / (lever_arm * cell_volume)
    return (
        (weights_y * scale).to(torch.float32).to(device)[None, :, None]
        * wavelet,
        (weights_x * scale).to(torch.float32).to(device)[None, :, None]
        * wavelet,
    )


def propagate_couples(moment: torch.Tensor) -> torch.Tensor:
    """Propagates a 2-D moment tensor source and returns radial velocity."""
    amplitudes_y, amplitudes_x = couple_amplitudes(moment)
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        # The channel-keyed mapping form: with two source channels and two
        # receiver channels it says the same thing as the six flat keywords
        # it replaces, in half the lines and with each channel's amplitudes
        # and locations side by side. The flat form remains permanently
        # supported and is used by the other examples.
        sources={
            "y": (amplitudes_y, couple_locations),
            "x": (amplitudes_x, couple_locations),
        },
        receivers={"y": receiver_locations, "x": receiver_locations},
        pml_width=pml_width,
        pml_freq=freq,
    )
    return radial_velocity(out)


def radial_velocity(out: deepwave.PropagatorOutput) -> torch.Tensor:
    """Velocity along the source-receiver direction, positive outwards."""
    velocity_y = out.receiver_amplitudes_y[0].cpu().to(torch.float64)
    velocity_x = out.receiver_amplitudes_x[0].cpu().to(torch.float64)
    return (
        direction[:, 0, None] * velocity_y
        + direction[:, 1, None] * velocity_x
    )


# The double couple we are interested in.
radial_double_couple = propagate_couples(moment_tensor_2d)

# The same recipe applied to an isotropic moment tensor, M0 times the
# identity, which is an explosion. Its radiation pattern is uniform, so it
# provides the reference waveform against which the double couple's
# amplitudes are calibrated below.
radial_explosion = propagate_couples(torch.eye(2, dtype=torch.float64) * m0)

# An independent check of the recipe's overall sign and scale, using
# Deepwave's own explosive source. The body force above has an equivalent
# description as a stress glut: it adds -M_ij delta(x - s) to the stress.
# For an isotropic moment tensor M0 * w(t), and since pressure is minus
# the mean normal stress, that is a pressure of M0 * w(t) spread over one
# cell. Deepwave's pressure source is specified as a pressure *rate* (see
# the units documentation), so the equivalent amplitude is the time
# derivative of that, M0 * dw/dt / cell_volume.
out_pressure_source = elastic(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    # ``"pressure_rate"`` names the quantity this propagator's scalar
    # source channel injects; ``"p"`` is a deprecated spelling of it.
    sources={
        "pressure_rate": (
            (m0 / cell_volume * torch.gradient(wavelet, spacing=dt)[0])[
                None, None
            ],
            torch.tensor([[[source_y, source_x]]], device=device),
        )
    },
    receivers={"y": receiver_locations, "x": receiver_locations},
    pml_width=pml_width,
    pml_freq=freq,
)
radial_pressure_source = radial_velocity(out_pressure_source)

# ---------------------------------------------------------------------
# 5. Measuring the P-wave radiation pattern
# ---------------------------------------------------------------------
# The window opens at the P travel time and closes at the S travel time,
# so it contains the P arrival and no S energy. Every receiver is the same
# distance from the source, so the P waveform is identical at all of them
# apart from a scale factor: that scale factor is the radiation pattern.
# Projecting each receiver's windowed trace onto the explosion's waveform
# therefore measures the pattern, and because the explosion's own pattern
# is M0 in every direction, the result is directly the P radiation pattern
# divided by M0.
time = torch.arange(nt, dtype=torch.float64) * dt
window = (time >= radius * dx / vp) & (time <= radius * dx / vs)
reference = radial_explosion[:, window].mean(dim=0)
reference_energy = (reference * reference).sum()


def pattern_of(radial: torch.Tensor) -> torch.Tensor:
    """Projects windowed traces onto the reference waveform."""
    return (radial[:, window] * reference).sum(dim=1) / reference_energy


measured = pattern_of(radial_double_couple)
measured_explosion = pattern_of(radial_explosion)
calibration = (
    radial_explosion[:, window] * radial_pressure_source[:, window]
).sum() / (radial_pressure_source[:, window] ** 2).sum()

# The far-field P-wave radiation pattern of a moment tensor is the moment
# tensor contracted twice with the unit source-receiver direction. For a
# double couple this equals 2 * M0 * (direction . slip)(direction . normal),
# so it vanishes where the direction is perpendicular to the slip vector or
# to the fault normal: the two nodal planes are the fault plane itself and
# the auxiliary plane perpendicular to the slip. They are indistinguishable
# from P polarities alone, which is the classic fault plane ambiguity.
predicted = torch.einsum(
    "ai,ij,aj->a", direction, moment_tensor_2d / m0, direction
)
slip_2d = slip[in_plane]
normal_2d = normal[in_plane]
nodal_angles = torch.tensor(
    [
        math.atan2(slip_2d[1], slip_2d[0]),
        math.atan2(normal_2d[1], normal_2d[0]),
    ],
    dtype=torch.float64,
)
nodal_angles = torch.cat([nodal_angles, nodal_angles + math.pi]) % (
    2 * math.pi
)

# ---------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(11, 5))
ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
# Zero at the bottom of the plot, so the display is a cross-section with
# depth downwards and North to the right.
ax_polar.set_theta_zero_location("S")
dense_angles = torch.linspace(0, 2 * math.pi, 721, dtype=torch.float64)
dense_direction = torch.stack(
    [torch.cos(dense_angles), torch.sin(dense_angles)], dim=-1
)
dense_predicted = torch.einsum(
    "ai,ij,aj->a", dense_direction, moment_tensor_2d / m0, dense_direction
)
ax_polar.plot(
    dense_angles, dense_predicted.abs(), "k-", lw=1, label="predicted"
)
positive = measured > 0
ax_polar.plot(
    angles[positive],
    measured[positive].abs(),
    "r.",
    label="measured, outward",
)
ax_polar.plot(
    angles[~positive],
    measured[~positive].abs(),
    "b.",
    label="measured, inward",
)
for nodal_angle in nodal_angles:
    ax_polar.plot([nodal_angle, nodal_angle], [0, 1], "g--", lw=0.8)
ax_polar.set_title("P radiation pattern\n(0 is down, 90 is North)")
ax_polar.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=8)

ax_traces = fig.add_subplot(1, 2, 2)
limit = radial_double_couple.abs().max().item()
ax_traces.imshow(
    radial_double_couple.T,
    aspect="auto",
    cmap="seismic",
    vmin=-limit,
    vmax=limit,
    extent=(0.0, 360.0, nt * dt, 0.0),
)
for edge in (radius * dx / vp, radius * dx / vs):
    ax_traces.axhline(edge, color="k", ls="--", lw=0.8)
for nodal_angle in nodal_angles:
    ax_traces.axvline(math.degrees(nodal_angle), color="g", ls="--", lw=0.8)
ax_traces.set_xlabel("Receiver angle (degrees)")
ax_traces.set_ylabel("Time (s)")
ax_traces.set_title("Radial velocity\n(dashed: P and S travel times)")
plt.tight_layout()
plt.savefig("example_moment_tensor.jpg")

# ---------------------------------------------------------------------
# 7. Verification
# ---------------------------------------------------------------------
assert torch.allclose(
    moment_tensor, moment_tensor_components, rtol=1e-12, atol=1e-12 * m0
), "the two forms of the moment tensor relations disagree"

assert abs(calibration - 1) < 0.05, (
    "the couple recipe disagrees with Deepwave's own explosive source in "
    f"sign or amplitude: ratio {calibration:.4f}"
)

assert (measured_explosion - 1).abs().max() < 0.02, (
    "the isotropic source's measured radiation pattern is not uniform: "
    f"{measured_explosion.min():.4f} to {measured_explosion.max():.4f}"
)

assert (measured - predicted).abs().max() < 0.03, (
    "the measured P radiation pattern does not match the prediction: "
    f"largest difference {(measured - predicted).abs().max():.4f}"
)

lobes = predicted.abs() > 0.2
assert (
    torch.sign(measured[lobes]) == torch.sign(predicted[lobes])
).all(), (
    "the polarity of the measured radiation pattern does not match the "
    "prediction in every quadrant"
)

receiver_spacing = 2 * math.pi / n_receivers
for nodal_angle in nodal_angles:
    offset = (angles - nodal_angle + math.pi) % (2 * math.pi) - math.pi
    nearby = offset.abs() < math.pi / 4
    minimum = torch.argmin(
        torch.where(
            nearby, measured.abs(), torch.full_like(measured, math.inf)
        )
    )
    assert offset[minimum].abs() <= receiver_spacing, (
        "the radiation pattern's minimum is not at the predicted nodal "
        f"direction {math.degrees(nodal_angle):.1f} degrees"
    )
    assert measured[minimum].abs() < 0.02 * measured.abs().max(), (
        "the radiation pattern does not fall to zero at the predicted "
        f"nodal direction {math.degrees(nodal_angle):.1f} degrees"
    )

print(f"calibration against Deepwave's explosive source: {calibration:.4f}")
print(
    "largest difference from the predicted pattern: "
    f"{(measured - predicted).abs().max():.4f}"
)

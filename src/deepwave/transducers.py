"""Transducer-array helpers for ultrasound and NDT.

Stability: provisional (see :doc:`/stability`) -- this module is expected to
gain element types, functions and constructor keywords. The unit convention
has settled -- positions in grid cells, speeds in cells per unit time, delays
and ``dt`` in one time unit of your choosing -- and what it has today is not
respelled or withdrawn without a release's notice under
:class:`deepwave.DeepwaveFutureWarning`.

Deepwave's propagators inject and record at point sources/receivers whose
locations are given in **grid cells** (see
``docs/dev/location_coordinate_semantics.md`` and :doc:`boundaries`). A
physical transducer element is an *extended* aperture:
a line, arc, disc or focused bowl. The standard way to represent such an element
in a grid-based finite-difference code (as in k-Wave's ``kWaveArray``) is to
tessellate the aperture into many closely spaced points and drive/record them
together, optionally with per-point apodisation weights and per-point time
delays for beam focusing/steering.

This module provides exactly that, as pure tensor-in/tensor-out functions that
compose with the propagators' floating-point location support (float locations
are automatically band-limited-interpolated onto the grid via Hicks):

* element builders -- :func:`line_element`, :func:`arc_element` (2-D),
  :func:`disc_element` and :func:`bowl_element` (3-D) -- each return a
  ``(locations, weights)`` tuple. ``locations`` has shape ``[n, ndim]`` in grid
  cells; ``weights`` has shape ``[n]`` and holds the window (apodisation)
  amplitude;
* :func:`window_weights` builds standalone per-point window weights;
* :func:`delays_to_focus` computes per-point transmit delays to focus/steer a
  beam, and :func:`apply_delays` turns a base wavelet into per-point amplitudes
  ``[n, nt]`` by spectrally exact (FFT phase-shift) fractional-delay shifting --
  differentiable with respect to both the base wavelet and the delays;
* :func:`element_source` assembles a builder's ``(locations, weights)``, an
  optional delay law and a base wavelet into the
  ``(source_amplitudes, source_locations)`` pair a propagator takes;
* :func:`element_receiver_trace` collapses the per-point receiver traces of an
  element into a single weighted element trace.

**Units.** All quantities are in **your** units. Positions are in grid cells.
``c`` is a speed in **cells per unit time**, and delays and ``dt`` are in the
same unit of time. If your model is in metres and seconds, ``c_cells =
c_physical / grid_spacing``. Nothing in this module needs to know what a metre
is.

For water at 1500 m/s, in the two unit systems :doc:`/ultrasound` describes::

    c = 1.5 / 0.1        # mm/us/MHz: 1.5 mm/us on 0.1 mm cells -> 15 cells/us
    c = 1500.0 / 1e-4    # SI: 1500 m/s on 1e-4 m cells -> 1.5e7 cells/s

The first gives delays in microseconds, to pair with a ``dt`` in
microseconds; the second gives delays in seconds, to pair with a ``dt`` in
seconds.
"""

import math
from collections.abc import Sequence
from typing import Optional, Union

import torch

from deepwave import _windows
from deepwave._exceptions import DeepwaveValueError as _DeepwaveValueError

__all__ = [
    "apply_delays",
    "arc_element",
    "bowl_element",
    "delays_to_focus",
    "disc_element",
    "element_receiver_trace",
    "element_source",
    "line_element",
    "window_weights",
]


def _point_window(
    n: int,
    kind: str,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-point window weights for an ``n``-point element.

    Shared by :func:`window_weights` and the element builders' ``window``
    parameter.
    """
    if n < 1:
        raise _DeepwaveValueError("n must be at least 1.")
    if n == 1:
        _windows.canonicalize(kind)  # validate even on the n=1 short-circuit
        return torch.ones(1, dtype=dtype, device=device)

    k = torch.arange(float(n), dtype=dtype, device=device)
    x = 2 * k / (n - 1) - 1  # symmetric coordinate: 0 at centre, +-1 at edges
    return _windows.weights(x, kind)


def window_weights(
    n: int,
    kind: str = _windows.DEFAULT,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return per-point window (apodisation) weights for an ``n``-point element.

    A window/taper shapes the amplitude across an aperture, most commonly to
    reduce beam side lobes. The returned window is *not* normalised (its peak
    is 1 for the tapered windows, and every value is 1 for ``"rect"``);
    normalise it yourself if you need the element's total amplitude to be
    independent of ``kind``.

    ``kind`` defaults to ``"hann"`` here, while the element builders'
    ``window`` parameter defaults to ``"rect"``. The difference is deliberate:
    an element should not be apodised unless you ask for it, whereas calling
    this function at all is asking for a window.

    Args:
        n: The number of points in the element.
        kind: The window type: ``"rect"`` (a.k.a. ``"uniform"``, ``"boxcar"``
            or ``"none"``; all ones), ``"hann"`` (a.k.a. ``"hanning"``,
            default), ``"hamming"``, or ``"blackman"``.
        dtype: The PyTorch datatype. Optional, defaults to PyTorch's default.
        device: The PyTorch device. Optional, defaults to the CPU.

    Returns:
        A tensor of shape ``[n]`` containing the weights.

    """
    return _point_window(n, kind, dtype=dtype, device=device)


def _as_tensor(
    x: Union[Sequence[float], torch.Tensor],
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    value = torch.as_tensor(x, dtype=dtype, device=device)
    if not bool(torch.isfinite(value).all()):
        raise _DeepwaveValueError("geometry coordinates and directions must be finite.")
    return value


def _device_of(x: Union[Sequence[float], torch.Tensor]) -> Optional[torch.device]:
    """The device of ``x`` if it is a tensor, else ``None`` (PyTorch default/CPU).

    Element builders thread this device through every internally constructed
    tensor so that, e.g., a ``centre`` given on CUDA yields locations on CUDA.
    """
    return x.device if isinstance(x, torch.Tensor) else None


def _n_points_for_length(length: float, spacing: float) -> int:
    """Number of points so that consecutive spacing is <= ``spacing``."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise _DeepwaveValueError("spacing must be positive and finite.")
    if not math.isfinite(length) or length < 0:
        raise _DeepwaveValueError("length must be finite and non-negative.")
    if length <= 0:
        return 1
    return math.ceil(length / spacing) + 1


def line_element(
    centre: Union[Sequence[float], torch.Tensor],
    direction: Union[Sequence[float], torch.Tensor],
    length: float,
    spacing: float = 1.0,
    window: str = "rect",
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tessellate a straight line-segment aperture (2-D or 3-D).

    The element is a line segment of the given ``length`` (in grid cells),
    centred on ``centre`` and oriented along ``direction``. It is sampled by
    ``n`` equally spaced points whose spacing is ``<= spacing`` cells; the
    endpoints are symmetric about ``centre``.

    Args:
        centre: The centre of the segment, shape ``[ndim]`` (grid cells).
        direction: The orientation vector, shape ``[ndim]`` (need not be
            normalised).
        length: The total length of the segment (grid cells); must be
            non-negative (a length of 0 gives a single point at ``centre``).
        spacing: The maximum spacing between adjacent points (grid cells).
            Optional, default 1 cell.
        window: Window (apodisation) type passed to :func:`window_weights`.
            Optional, default ``"rect"`` (uniform) -- note that this differs
            from :func:`window_weights`'s own ``"hann"`` default, because an
            element should not be apodised unless you ask for it.
        dtype: The PyTorch datatype. Optional, defaults to PyTorch's default.

    Returns:
        ``(locations, weights)`` where ``locations`` has shape ``[n, ndim]``
        (grid cells) and ``weights`` has shape ``[n]``. Prefer passing both to
        :func:`element_source`, which applies the weights (and any delay law)
        and returns the ``(source_amplitudes, source_locations)`` pair a
        propagator takes; the raw pair is the low-level path, and forgetting
        to apply ``weights`` silently changes the beam pattern.

    """
    dtype = dtype or torch.get_default_dtype()
    if length < 0:
        raise _DeepwaveValueError(f"length must be non-negative, got {length}.")
    device = _device_of(centre)
    centre_t = _as_tensor(centre, dtype, device)
    direction_t = _as_tensor(direction, dtype, device)
    if centre_t.shape != direction_t.shape or centre_t.dim() != 1:
        raise _DeepwaveValueError("centre and direction must be 1-D of equal length.")
    norm = torch.linalg.vector_norm(direction_t)
    if norm == 0:
        raise _DeepwaveValueError("direction must be non-zero.")
    unit = direction_t / norm

    n = _n_points_for_length(length, spacing)
    if n == 1:
        offsets = torch.zeros(1, dtype=dtype, device=device)
    else:
        offsets = torch.linspace(-length / 2, length / 2, n, dtype=dtype, device=device)
    locations = centre_t.unsqueeze(0) + offsets.unsqueeze(1) * unit.unsqueeze(0)
    weights_t = _point_window(n, window, dtype=dtype, device=device)
    return locations, weights_t


def arc_element(
    centre_of_curvature: Union[Sequence[float], torch.Tensor],
    radius: float,
    angles: Sequence[float],
    spacing: float = 1.0,
    window: str = "rect",
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Tessellate a circular-arc aperture (2-D only).

    Models a curved (focused) 2-D element: points lie on a circle of the given
    ``radius`` about ``centre_of_curvature``, spanning the angular range
    ``angles = (theta_start, theta_end)`` in radians. The arc's centre of
    curvature is *not* a point on the arc. The point at angle :math:`\theta` is
    at ``centre_of_curvature + radius * [cos(theta), sin(theta)]``, i.e.
    ``theta`` is measured from the first (slow) grid dimension towards the
    second. Adjacent points are spaced ``<= spacing`` cells along the arc.

    A curved array focused at a point is obtained by placing an arc of the
    desired radius about that focal point.

    For a *full* circle (an angular span that is an integer multiple of
    :math:`2\pi`) the final sample would coincide with the first; the duplicated
    seam point is dropped so every returned cell is distinct.

    Args:
        centre_of_curvature: The centre of the circle the arc lies on, shape
            ``[2]`` (grid cells). This is the arc's geometric focus, not a
            point on the arc itself.
        radius: The arc radius (grid cells).
        angles: ``(theta_start, theta_end)`` in radians.
        spacing: Maximum arc spacing between adjacent points (grid cells).
        window: Window (apodisation) type passed to :func:`window_weights`.
            Optional, default ``"rect"`` (uniform) -- note that this differs
            from :func:`window_weights`'s own ``"hann"`` default, because an
            element should not be apodised unless you ask for it.
        dtype: The PyTorch datatype. Optional.

    Returns:
        ``(locations, weights)`` with ``locations`` of shape ``[n, 2]``. Prefer
        passing both to :func:`element_source`, which applies the weights (and
        any delay law) and returns the
        ``(source_amplitudes, source_locations)`` pair a propagator takes; the
        raw pair is the low-level path, and forgetting to apply ``weights``
        silently changes the beam pattern.

    """
    dtype = dtype or torch.get_default_dtype()
    device = _device_of(centre_of_curvature)
    centre_t = _as_tensor(centre_of_curvature, dtype, device)
    if centre_t.shape != (2,):
        raise _DeepwaveValueError(
            "arc_element is 2-D: centre_of_curvature must have shape [2].",
        )
    if len(angles) != 2:
        raise _DeepwaveValueError("angles must be (theta_start, theta_end).")
    if not math.isfinite(radius) or radius <= 0:
        raise _DeepwaveValueError("radius must be positive and finite.")

    theta0, theta1 = float(angles[0]), float(angles[1])
    if not math.isfinite(theta0) or not math.isfinite(theta1):
        raise _DeepwaveValueError("angles must be finite.")
    arc_length = radius * abs(theta1 - theta0)
    n = _n_points_for_length(arc_length, spacing)
    # A full circle (angular span an integer multiple of 2*pi) closes on itself,
    # so the last sample coincides with the first; drop the duplicated seam point
    # to keep every cell distinct (a repeated source/receiver location).
    span_mod = abs(theta1 - theta0) % (2 * math.pi)
    full_circle = n > 1 and min(span_mod, 2 * math.pi - span_mod) < 1e-9
    theta = torch.linspace(theta0, theta1, n, dtype=dtype, device=device)
    if full_circle:
        theta = theta[:-1]
        n -= 1
    offsets = torch.stack(
        [radius * torch.cos(theta), radius * torch.sin(theta)],
        dim=1,
    )
    locations = centre_t.unsqueeze(0) + offsets
    weights_t = _point_window(n, window, dtype=dtype, device=device)
    return locations, weights_t


def _orthonormal_basis(
    normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit vectors spanning the plane perpendicular to ``normal`` (3-D)."""
    # Pick the axis least aligned with normal to avoid degeneracy.
    ref = torch.zeros(3, dtype=normal.dtype, device=normal.device)
    ref[int(normal.abs().argmin())] = 1.0
    u = torch.linalg.cross(normal, ref)
    u = u / torch.linalg.vector_norm(u)
    w = torch.linalg.cross(normal, u)
    w = w / torch.linalg.vector_norm(w)
    return u, w


def _ring_radii(max_radius: float, spacing: float) -> list[float]:
    """Radii of concentric rings (including 0) covering a disc of radius R."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise _DeepwaveValueError("spacing must be positive and finite.")
    n_rings = math.ceil(max_radius / spacing) if max_radius > 0 else 0
    return [max_radius * i / n_rings for i in range(n_rings + 1)] if n_rings else [0.0]


def disc_element(
    centre: Union[Sequence[float], torch.Tensor],
    radius: float,
    normal: Union[Sequence[float], torch.Tensor],
    spacing: float = 1.0,
    window: str = "rect",
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tessellate a flat circular-disc aperture (3-D).

    Points fill a flat disc of the given ``radius`` centred on ``centre`` and
    lying in the plane whose normal is ``normal`` (an unfocused, planar circular
    element). The disc is tessellated by concentric rings spaced ``<= spacing``
    cells, with points spaced ``<= spacing`` along each ring, giving an
    approximately uniform areal density.

    Args:
        centre: The disc centre, shape ``[3]`` (grid cells).
        radius: The disc radius (grid cells).
        normal: The disc-plane normal, shape ``[3]`` (need not be normalised).
        spacing: Target point spacing (grid cells).
        window: Window (apodisation) applied as a function of the *radial*
            fraction (0 at centre, 1 at the rim); see :func:`window_weights`
            for the available types. Optional, default ``"rect"`` (uniform) --
            note that this differs from :func:`window_weights`'s own
            ``"hann"`` default, because an element should not be apodised
            unless you ask for it.
        dtype: The PyTorch datatype. Optional.

    Returns:
        ``(locations, weights)`` with ``locations`` of shape ``[n, 3]``. Prefer
        passing both to :func:`element_source`, which applies the weights (and
        any delay law) and returns the
        ``(source_amplitudes, source_locations)`` pair a propagator takes; the
        raw pair is the low-level path, and forgetting to apply ``weights``
        silently changes the beam pattern.

    """
    dtype = dtype or torch.get_default_dtype()
    device = _device_of(centre)
    centre_t = _as_tensor(centre, dtype, device)
    normal_t = _as_tensor(normal, dtype, device)
    if centre_t.shape != (3,) or normal_t.shape != (3,):
        raise _DeepwaveValueError(
            "disc_element is 3-D: centre and normal need shape [3]."
        )
    if not math.isfinite(radius) or radius <= 0:
        raise _DeepwaveValueError("radius must be positive and finite.")
    normal_norm = torch.linalg.vector_norm(normal_t)
    if normal_norm == 0:
        raise _DeepwaveValueError("normal must be non-zero.")
    nrm = normal_t / normal_norm
    u, w = _orthonormal_basis(nrm)

    points = []
    radial_frac = []
    for r in _ring_radii(radius, spacing):
        if r == 0:
            points.append(centre_t)
            radial_frac.append(0.0)
            continue
        n_on_ring = max(1, math.ceil(2 * math.pi * r / spacing))
        for j in range(n_on_ring):
            psi = 2 * math.pi * j / n_on_ring
            offset = r * (math.cos(psi) * u + math.sin(psi) * w)
            points.append(centre_t + offset)
            radial_frac.append(r / radius)
    locations = torch.stack(points, dim=0)
    weights_t = _radial_window(radial_frac, window, dtype, device)
    return locations, weights_t


def bowl_element(
    apex: Union[Sequence[float], torch.Tensor],
    radius: float,
    diameter: float,
    focus: Union[Sequence[float], torch.Tensor],
    spacing: float = 1.0,
    window: str = "rect",
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tessellate an axisymmetric focused-bowl aperture (3-D).

    Models a spherically focused single-element transducer (a spherical cap),
    the workhorse geometry of therapeutic and photoacoustic ultrasound. The
    bowl is the cap of a sphere of radius-of-curvature ``radius`` whose centre
    of curvature is at ``focus``; ``apex`` is the pole of the cap on the
    sphere, and the axis points from ``apex`` toward ``focus``. The cap
    extends out to an aperture of the given ``diameter``. The surface is
    tessellated by concentric rings of constant polar angle, spaced ``<=
    spacing`` cells along the surface.

    This mirrors k-Wave's ``makeBowl`` parameterisation (apex position, radius
    of curvature, aperture diameter, focus direction).

    Args:
        apex: The bowl apex/pole, shape ``[3]`` (grid cells). This is a point
            *on* the bowl surface; the bowl's centre of curvature is the
            separate ``focus`` argument.
        radius: The radius of curvature (grid cells).
        diameter: The aperture diameter (grid cells); must be ``<= 2*radius``.
        focus: The centre of curvature / geometric focus, shape ``[3]`` (grid
            cells). Its distance from ``apex`` should be ``radius``; only its
            direction from ``apex`` is used to define the axis.
        spacing: Target surface point spacing (grid cells).
        window: Window (apodisation) as a function of polar-angle fraction
            (0 at apex, 1 at rim); see :func:`window_weights` for the
            available types. Optional, default ``"rect"`` (uniform) -- note
            that this differs from :func:`window_weights`'s own ``"hann"``
            default, because an element should not be apodised unless you ask
            for it.
        dtype: The PyTorch datatype. Optional.

    Returns:
        ``(locations, weights)`` with ``locations`` of shape ``[n, 3]``. Prefer
        passing both to :func:`element_source`, which applies the weights (and
        any delay law) and returns the
        ``(source_amplitudes, source_locations)`` pair a propagator takes; the
        raw pair is the low-level path, and forgetting to apply ``weights``
        silently changes the beam pattern.

    """
    dtype = dtype or torch.get_default_dtype()
    device = _device_of(apex)
    apex_t = _as_tensor(apex, dtype, device)
    focus_t = _as_tensor(focus, dtype, device)
    if apex_t.shape != (3,) or focus_t.shape != (3,):
        raise _DeepwaveValueError("bowl_element is 3-D: apex and focus need shape [3].")
    if not math.isfinite(radius) or radius <= 0:
        raise _DeepwaveValueError("radius must be positive and finite.")
    if not math.isfinite(diameter) or diameter <= 0 or diameter > 2 * radius:
        raise _DeepwaveValueError("diameter must be in (0, 2*radius].")

    axis = focus_t - apex_t
    axis_norm = torch.linalg.vector_norm(axis)
    if axis_norm == 0:
        raise _DeepwaveValueError(
            "focus must differ from apex (the axis is undefined)."
        )
    axis = axis / axis_norm
    u, w = _orthonormal_basis(axis)
    # Centre of curvature.
    cc = apex_t + radius * axis
    # Maximum polar angle (measured at the centre of curvature, from the
    # apex direction -axis) set by the aperture radius.
    phi_max = math.asin(min(1.0, (diameter / 2) / radius))
    surf_len = radius * phi_max  # arc length apex->rim along the surface
    n_rings = math.ceil(surf_len / spacing) if surf_len > 0 else 0

    points = []
    polar_frac = []
    ring_phis = (
        [phi_max * i / n_rings for i in range(n_rings + 1)] if n_rings else [0.0]
    )
    for phi in ring_phis:
        # Direction from centre of curvature to the surface point:
        # -cos(phi)*axis (toward apex) + sin(phi)*(in-plane).
        if phi == 0:
            points.append(cc - radius * axis)
            polar_frac.append(0.0)
            continue
        ring_r = radius * math.sin(phi)  # off-axis radius of this ring
        n_on_ring = max(1, math.ceil(2 * math.pi * ring_r / spacing))
        for j in range(n_on_ring):
            psi = 2 * math.pi * j / n_on_ring
            in_plane = math.cos(psi) * u + math.sin(psi) * w
            direction = -math.cos(phi) * axis + math.sin(phi) * in_plane
            points.append(cc + radius * direction)
            polar_frac.append(phi / phi_max if phi_max > 0 else 0.0)
    locations = torch.stack(points, dim=0)
    weights_t = _radial_window(polar_frac, window, dtype, device)
    return locations, weights_t


def _radial_window(
    fracs: Sequence[float],
    kind: str,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Window weights as a function of radial fraction in [0, 1].

    ``fracs[i]`` is 0 at the aperture centre and 1 at the rim; the tapered
    windows are evaluated at these fractions (the ``x >= 0`` half of the
    shared window's symmetric coordinate) so that concentric-ring
    tessellations get a smooth, radially symmetric taper.
    """
    frac = torch.as_tensor(fracs, dtype=dtype, device=device)
    return _windows.weights(frac, kind)


def delays_to_focus(
    locations: torch.Tensor,
    focus: Union[Sequence[float], torch.Tensor],
    c: float,
) -> torch.Tensor:
    """Per-point transmit delays to focus/steer a beam at ``focus``.

    For each element point, returns the time delay to apply on transmit so
    that all points' contributions arrive in phase at the focal point. Points
    farther from the focus fire earlier; the delays are non-negative and the
    point nearest the focus has the largest delay. Placing the focus far away
    in a chosen direction approximates plane-wave steering.

    This helper assumes isotropic physical cell spacing. Convert physical
    speed to cells per unit time with the single common grid spacing; for an
    anisotropic grid, compute delays from physical coordinates separately.

    All quantities are in **your** units. Positions are in grid cells, ``c``
    is a speed in **cells per unit time**, and the returned delays are in that
    same unit of time -- the one to use for ``dt``. If your model is in metres
    and seconds, ``c_cells = c_physical / grid_spacing``, and the delays come
    back in seconds.

    Args:
        locations: Element point locations, shape ``[n, ndim]`` (grid cells).
        focus: The focal point, shape ``[ndim]`` (grid cells).
        c: The wave speed between the element and the focus, in grid cells per
            unit time.

    Returns:
        A tensor of shape ``[n]`` of non-negative delays, in the time unit
        implied by ``c``.

    """
    if not math.isfinite(c) or c <= 0:
        raise _DeepwaveValueError("c must be positive and finite.")
    if locations.dim() != 2:
        raise _DeepwaveValueError(
            f"locations must have shape [n, ndim], got {list(locations.shape)}.",
        )
    dtype = locations.dtype
    ndim = locations.shape[-1]
    focus_t = torch.as_tensor(focus, dtype=dtype, device=locations.device)
    if focus_t.shape != (ndim,):
        raise _DeepwaveValueError(
            f"focus must have shape [{ndim}] (locations.shape[-1]), got "
            f"{list(focus_t.shape)}.",
        )
    if not torch.isfinite(locations).all() or not torch.isfinite(focus_t).all():
        raise _DeepwaveValueError("locations and focus must be finite.")
    dist = torch.linalg.vector_norm(locations - focus_t, dim=1)  # [n], in cells
    # Farthest point fires first (delay 0); nearest fires last.
    delays: torch.Tensor = (dist.max() - dist) / c
    return delays


def _resample_wavelet(wavelet: torch.Tensor, nt: Optional[int]) -> torch.Tensor:
    """Zero-pad or truncate ``wavelet``'s trailing time axis to ``nt`` samples."""
    if nt is None or nt == wavelet.shape[-1]:
        return wavelet
    if nt < wavelet.shape[-1]:
        return wavelet[..., :nt]
    return torch.nn.functional.pad(wavelet, (0, nt - wavelet.shape[-1]))


def _fft_delay_shift(
    wavelet: torch.Tensor,
    delays: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Shift ``wavelet`` ``[..., nt]`` by each of ``delays`` ``[n]``.

    Returns ``[..., n, nt]``. A positive delay shifts later in time.
    """
    n_time = wavelet.shape[-1]
    spectrum = torch.fft.rfft(wavelet)  # [..., nf]
    # Angular frequency of each rfft bin. Build directly in the wavelet's dtype
    # so high-index frequencies keep full precision (a float32 default dtype
    # here would otherwise poison the fractional-delay phase).
    freqs = torch.fft.rfftfreq(
        n_time,
        d=dt,
        dtype=wavelet.dtype,
        device=wavelet.device,
    )
    omega = 2 * math.pi * freqs  # [nf]
    # exp(-i omega delay) shifts a signal later in time by `delay`.
    phase = -omega.unsqueeze(0) * delays.unsqueeze(1)  # [n, nf]
    shifted = spectrum.unsqueeze(-2) * torch.polar(
        torch.ones_like(phase),
        phase,
    )
    result: torch.Tensor = torch.fft.irfft(shifted, n=n_time, dim=-1)
    return result


def apply_delays(
    wavelet: torch.Tensor,
    delays: torch.Tensor,
    dt: float,
    nt: Optional[int] = None,
) -> torch.Tensor:
    """Build per-point amplitudes by fractional-delay-shifting a base wavelet.

    Each element point emits the same base ``wavelet`` shifted in time by its
    delay. Shifting is done by a spectral (FFT) phase ramp, which applies an
    arbitrary *fractional* (sub-sample) delay exactly for a band-limited signal
    -- the same spectral philosophy Deepwave uses for source/receiver time
    resampling. A positive delay shifts the wavelet later in time.

    The operation is differentiable with respect to both ``wavelet`` and
    ``delays`` (the phase ramp is a smooth function of the delays), so delay
    laws can be optimised end-to-end.

    ``delays`` and ``dt`` are in the same unit of time, whichever unit that is
    -- the one :func:`delays_to_focus` returns for your ``c``. Nothing here
    needs to know what a second is.

    .. note::

        The FFT shift is circular: energy shifted past the end of the record
        wraps to the beginning. Make ``nt`` long enough (and pad the base
        wavelet) that the delayed wavelet stays within the record.

    Args:
        wavelet: The base wavelet, shape ``[nt0]``.
        delays: Per-point delays, shape ``[n]``, in the same time unit as
            ``dt``. May be negative.
        dt: The time sample spacing.
        nt: The output number of time samples. Optional; if given and different
            from ``len(wavelet)`` the wavelet is zero-padded or truncated to
            ``nt`` before shifting.

    Returns:
        A tensor of shape ``[n, nt]`` of per-point amplitudes.

    """
    if not math.isfinite(dt) or dt <= 0:
        raise _DeepwaveValueError(f"dt must be positive and finite, got {dt}.")
    if wavelet.dim() != 1:
        raise _DeepwaveValueError("wavelet must be 1-D of shape [nt].")
    if delays.dim() != 1:
        raise _DeepwaveValueError(
            f"delays must be 1-D of shape [n], got shape {list(delays.shape)}."
        )
    if not torch.isfinite(delays).all():
        raise _DeepwaveValueError("delays must be finite.")
    return _fft_delay_shift(_resample_wavelet(wavelet, nt), delays, dt)


def element_source(
    wavelet: torch.Tensor,
    locations: torch.Tensor,
    weights: torch.Tensor,
    *,
    delays: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
    nt: Optional[int] = None,
    n_shots: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble an element into the source pair a propagator takes.

    Takes the ``(locations, weights)`` an element builder returns, a base
    ``wavelet``, and an optional delay law, and returns
    ``(source_amplitudes, source_locations)`` of shapes ``[n_shots, n, nt]``
    and ``[n_shots, n, ndim]``, ready to pass straight to a propagator. This
    is the recommended way to drive an element: it applies the apodisation
    ``weights``, which are easy to forget and whose omission silently changes
    the beam pattern.

    Every step is differentiable, with respect to ``wavelet``, ``weights``,
    ``locations`` and ``delays`` alike, so a delay law or an apodisation can
    itself be optimised.

    ``delays`` and ``dt`` are in the same unit of time, whichever unit that is
    -- the one :func:`delays_to_focus` returns for a ``c`` in cells per unit
    time. Nothing here needs to know what a second is.

    The shot dimension is added with :meth:`torch.Tensor.expand`, so a
    single-shot wavelet costs no extra memory when broadcast to many shots.

    Args:
        wavelet: The base wavelet, shape ``[nt0]`` (the same wavelet in every
            shot) or ``[n_shots, nt0]`` (one per shot).
        locations: Element point locations, shape ``[n, ndim]`` (grid cells),
            from an element builder.
        weights: Per-point apodisation weights, shape ``[n]``, from the same
            builder.
        delays: Per-point transmit delays, shape ``[n]``, e.g. from
            :func:`delays_to_focus`. Optional; if given, ``dt`` is required.
        dt: The time sample spacing, in the same unit as ``delays``. Required
            if and only if ``delays`` is given.
        nt: The output number of time samples. Optional; if given and
            different from ``wavelet.shape[-1]`` the wavelet is zero-padded or
            truncated to ``nt``, exactly as in :func:`apply_delays`.
        n_shots: The number of shots. Optional; defaults to 1 for a 1-D
            wavelet and to ``wavelet.shape[0]`` for a 2-D one. Giving a value
            that disagrees with a 2-D wavelet's leading dimension raises
            rather than broadcasting.

    Returns:
        ``(source_amplitudes, source_locations)`` of shapes
        ``[n_shots, n, nt]`` and ``[n_shots, n, ndim]``.

    Raises:
        DeepwaveValueError: If ``delays`` is given without ``dt``; if
            ``weights`` and ``locations`` describe different numbers of
            points; if ``wavelet`` is neither 1-D nor 2-D; if ``delays`` is
            not 1-D of length ``n``; or if ``n_shots`` disagrees with a 2-D
            wavelet's leading dimension.

    """
    if locations.dim() != 2:
        raise _DeepwaveValueError(
            f"locations must have shape [n, ndim], got {list(locations.shape)}.",
        )
    if weights.dim() != 1:
        raise _DeepwaveValueError(
            f"weights must have shape [n], got {list(weights.shape)}.",
        )
    if weights.shape[0] != locations.shape[0]:
        raise _DeepwaveValueError(
            f"weights and locations must describe the same number of points, "
            f"got weights {list(weights.shape)} and locations "
            f"{list(locations.shape)}; they should come from the same element "
            f"builder call.",
        )
    if wavelet.dim() not in (1, 2):
        raise _DeepwaveValueError(
            f"wavelet must have shape [nt] or [n_shots, nt], got "
            f"{list(wavelet.shape)}.",
        )
    n_points = locations.shape[0]
    if wavelet.dim() == 2:
        if n_shots is not None and n_shots != wavelet.shape[0]:
            raise _DeepwaveValueError(
                f"n_shots={n_shots} disagrees with the wavelet's leading "
                f"dimension, {list(wavelet.shape)}. Omit n_shots to take it "
                f"from the wavelet, or pass a 1-D wavelet to repeat one "
                f"wavelet over {n_shots} shots.",
            )
        n_shots_out = wavelet.shape[0]
    else:
        n_shots_out = 1 if n_shots is None else n_shots
        if n_shots_out < 1:
            raise _DeepwaveValueError(f"n_shots must be at least 1, got {n_shots}.")

    base = _resample_wavelet(wavelet, nt)
    if delays is None:
        amplitudes = base.unsqueeze(-2).expand(
            *base.shape[:-1],
            n_points,
            base.shape[-1],
        )
    else:
        if dt is None:
            raise _DeepwaveValueError(
                f"dt is required when delays is given, but dt is None (delays "
                f"has shape {list(delays.shape)}). delays and dt must be in "
                f"the same unit of time.",
            )
        if delays.dim() != 1 or delays.shape[0] != n_points:
            raise _DeepwaveValueError(
                f"delays must have shape [n] matching the {n_points} element "
                f"points, got {list(delays.shape)} with locations "
                f"{list(locations.shape)}.",
            )
        # [nt] -> [n, nt]; [n_shots, nt] -> [n_shots, n, nt].
        amplitudes = _fft_delay_shift(base, delays, dt)
    amplitudes = amplitudes * weights.to(amplitudes.dtype).unsqueeze(-1)
    if amplitudes.dim() == 2:
        amplitudes = amplitudes.unsqueeze(0).expand(n_shots_out, -1, -1)
    source_locations = locations.unsqueeze(0).expand(n_shots_out, -1, -1)
    return amplitudes, source_locations


def element_receiver_trace(
    receiver_amplitudes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Collapse an element's per-point receiver traces into one trace.

    Returns the discrete (apodisation-)weighted sum of traces at the element's
    tessellation points. Builder weights do not include area quadrature or
    normalisation, so the result and its units depend on point density.

    Args:
        receiver_amplitudes: Recorded traces with the element's points on the
            second-to-last axis, shape ``[..., n, nt]`` (e.g. the propagator's
            ``[shot, receiver, time]`` output with ``receiver == n``).
        weights: Per-point weights, shape ``[n]`` (e.g. from an element
            builder or :func:`window_weights`).

    Returns:
        The weighted-sum element trace, shape ``[..., nt]``.

    """
    if weights.dim() != 1:
        raise _DeepwaveValueError("weights must be 1-D of shape [n].")
    if receiver_amplitudes.shape[-2] != weights.shape[0]:
        raise _DeepwaveValueError(
            "receiver_amplitudes.shape[-2] must equal the number of weights.",
        )
    w = weights.to(receiver_amplitudes.dtype)
    return (receiver_amplitudes * w.unsqueeze(-1)).sum(dim=-2)

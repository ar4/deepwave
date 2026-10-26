r"""Distributed acoustic sensing (DAS) strain-rate receivers.

Stability: provisional (see :doc:`/stability`) -- this module is expected to
gain helpers, so its shape may still be corrected.

DAS turns an ordinary fibre-optic cable into a dense line of seismic sensors:
interrogating light injected into the fibre backscatters at every point along
its length (Rayleigh scattering from intrinsic fibre inhomogeneities), and the
phase of that backscatter is sensitive to the *axial strain* the fibre
experiences. In practice, systems report the strain rate (or the strain)
averaged over a short **gauge length** ``L_g`` -- the fibre is not sampled
truly point-wise, but as the difference between two points ``L_g`` apart,
divided by ``L_g``. This module implements exactly that measurement in terms
of Deepwave's existing particle-velocity receivers, so DAS data can be
modelled (and inverted, since everything here is a plain differentiable
tensor operation) without any change to the propagators or kernels.

Physics
-------

For a fibre with unit tangent :math:`\hat t` at a point :math:`x`, the axial
strain-rate is the tangential-tangential component of the strain-rate tensor,

.. math::

    \dot\varepsilon_{tt}(x, t)
        = \hat t \cdot (\nabla v(x, t)) \cdot \hat t
        = \sum_{ij} t_i t_j \, \partial v_i / \partial x_j,

where :math:`v` is the particle-velocity field. A real interrogator does not
measure this pointwise derivative; it measures the *gauge-length-averaged*
strain rate, which is exactly a finite difference of the tangential particle
velocity between the two ends of the gauge:

.. math::

    \dot\varepsilon(x, t) \approx
        \frac{v_t(x + \tfrac12 L_g \hat t, t) - v_t(x - \tfrac12 L_g \hat
        t, t)}{L_g}, \qquad v_t = \hat t \cdot v.

This is simultaneously the physically correct interferometric-gauge model
*and* the natural implementation on top of Deepwave: place a pair of
particle-velocity receivers at the two gauge endpoints (Deepwave's
floating-point receiver locations and Hicks interpolation place them off-grid
exactly, and the staggered velocity components are handled per-component), project
onto the tangent, and difference. As :math:`L_g \to 0` this recipe converges to
the pointwise :math:`\dot\varepsilon_{tt}` above (see
:func:`strain_rate_from_velocities`); real gauge lengths (metres to tens of
metres) are deliberately kept finite because that is what an interrogator
actually reports.

Only the **elastic** propagator exposes true particle-velocity receivers for
every spatial component, so it is the natural target here (see
:mod:`deepwave.elastic`). The acoustic propagator also has particle-velocity
receivers (``receiver_amplitudes_z`` / ``_y`` / ``_x``) and the recipe below
applies to it unchanged. The scalar propagator has no particle-velocity field
(only a scalar wavefield), so it is not supported by this module.

Directivity
-----------

Projecting onto the tangent twice (once through :math:`v_t` at each gauge
end, once implicitly through the strain-rate tensor's index contraction with
:math:`\hat t \hat t`) gives DAS its characteristic **cosine-squared
directivity**: for a plane P wave travelling in direction :math:`\hat n` with
particle velocity :math:`v = A \hat n\, g(t - \hat n \cdot x / c)`, the
gradient is :math:`\partial v_i/\partial x_j = -(A/c)\, n_i n_j\, g'(\cdot)`,
so

.. math::

    \dot\varepsilon_{tt} = -\frac{A}{c}\, g'(\cdot)\, (\hat t \cdot
    \hat n)^2 = -\frac{A}{c}\, g'(\cdot) \, \cos^2\theta,

where :math:`\theta` is the angle between the fibre and the propagation
direction. A fibre is maximally sensitive to waves travelling along it
(:math:`\theta = 0`) and blind to waves crossing it at a right angle
(:math:`\theta = 90^\circ`, "broadside"). This is a fundamental property of
axial-strain sensing, not a numerical artefact, and it is verified directly
in ``tests/test_das.py``.

Gauge averaging as post-processing
-----------------------------------

:func:`strain_rate_from_velocities` already models the gauge physically, as
a two-point difference across the interferometric measurement itself. A
second, independent form of gauge averaging shows up whenever the fibre is
sampled far more densely than the gauge length: interrogators commonly
report each channel as a **moving average over the gauge window** of an
underlying finer-grained per-channel signal, rather than (or in addition
to) a single two-point difference across the whole gauge.
:func:`gauge_average` implements exactly that: a windowed mean along the
channel dimension of any per-channel tensor (for example the output of
:func:`strain_rate_from_velocities` computed with a small internal
two-point gauge, densely spaced).

Helically wound fibres
-----------------------

Real DAS cables often wind the fibre helically around a central strength
member rather than laying it straight, trading axial sensitivity for
robustness and (at the right angle) directional uniformity. Following
Kuvshinov (2016, *Geophysical Prospecting* 64(3), 671-688, "Interaction of
helically wound fibre-optic cables with plane seismic waves",
doi:10.1111/1365-2478.12303), :func:`helical_fiber_response` combines the
cable-frame axial and transverse normal strain rates into the fibre's
effective response at a given wrap angle (in radians, like every other angle
in Deepwave), including the "magic angle" (:math:`\arccos(1/\sqrt3) \approx
0.9553`, i.e. :math:`54.7356^\circ`) at which the fibre becomes a pure
volumetric-strain sensor. See its docstring for the formula, the derivation
summary, and exactly how to build its transverse-strain-rate inputs from
paired velocity receivers (the same two-point-gauge recipe as
:func:`strain_rate_from_velocities`, oriented normal to the cable instead
of along it).

Units
-----

Positions (``fiber_points`` and the gauge endpoints returned by
:func:`fiber_gauge_locations`) are in **grid cells**, consistent with every
other location argument in Deepwave. ``gauge_length`` is a **physical**
length (the same physical units as ``grid_spacing``, e.g. metres), *not* a
number of cells: it divides a physical velocity difference, so it must be
physical for the result to be a physical strain rate (units of 1/time, e.g.
1/s for m/s velocities and metres). :func:`fiber_gauge_locations` is the only
function here that needs ``grid_spacing``, to convert the physical half-gauge
offset into a grid-cell displacement along the tangent;
:func:`strain_rate_from_velocities` needs no ``grid_spacing`` because both its
inputs (particle velocities and the physical gauge length) are already in
physical units. Angles (:func:`helical_fiber_response`'s ``wrap_angle``) are
in **radians**, as everywhere else in Deepwave.

Differentiability
------------------

Every operation in this module (index selection, elementwise multiply, sum,
divide) is a plain differentiable ``torch`` op, so gradients flow through
:func:`strain_rate_from_velocities` to the velocity receivers Deepwave already
returns, and from there through the elastic propagator's exact discrete
adjoint back to the material model (Lame parameters/buoyancy) and source
amplitudes exactly as for any other receiver quantity. No special handling is
needed to use DAS data as an FWI objective.

Limitations
-----------

* **Straight-segment tangent approximation.** :func:`fiber_gauge_locations`
  estimates the tangent at each channel from a finite difference of the
  supplied ``fiber_points``, i.e. it assumes the fibre is locally straight
  between neighbouring points. For a curved fibre sampled too coarsely
  relative to its radius of curvature, both the tangent estimate and the
  implicit assumption that the two gauge ends lie *on* the fibre (rather than
  on a chord) become approximate. Supply enough points to resolve the
  curvature you care about.
* **Gauge endpoints must stay in-model.** The two gauge-end locations
  returned by :func:`fiber_gauge_locations` are plain float coordinates with
  no bounds checking here; if a channel near the edge of the model places an
  endpoint outside the domain (or its Hicks interpolation window), the
  propagator will raise (or silently ignore it, if you pass
  ``deepwave.IGNORE_LOCATION``). Keep the fibre, plus half a gauge length,
  inside the survey area (and away from the PML, per the usual receiver
  guidance).
* No 1-D case: a fibre needs a tangent, which requires at least 2 points and
  at least 2 spatial dimensions.
* :func:`helical_fiber_response` neglects the cable-frame shear-strain
  terms (:math:`\dot\varepsilon_{xz}`, :math:`\dot\varepsilon_{yz}`,
  :math:`\dot\varepsilon_{xy}`), following Kuvshinov (2016): for a
  symmetric periodic helix these average to zero over one wrap and do not
  appear in the formula. It also does not construct the transverse gauge
  endpoints for you (unlike :func:`fiber_gauge_locations` for the axial
  case) -- see its docstring for the short recipe.

"""

import math
from collections.abc import Mapping, Sequence
from typing import Union

import torch

from deepwave import _common
from deepwave._exceptions import DeepwaveTypeError as _DeepwaveTypeError
from deepwave._exceptions import DeepwaveValueError as _DeepwaveValueError

__all__ = [
    "fiber_gauge_locations",
    "gauge_average",
    "helical_fiber_response",
    "strain_rate_from_velocities",
]


def fiber_gauge_locations(
    fiber_points: torch.Tensor,
    gauge_length: float,
    grid_spacing: Union[float, Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-channel DAS gauge endpoints and tangents from fibre points.

    ``fiber_points`` describes the fibre as an ordered polyline: point ``i``
    and point ``i + 1`` are adjacent along the fibre (the order matters, the
    absolute spacing does not need to be uniform). One DAS **channel** is
    placed at each input point (so ``n`` points give ``n`` channels, matching
    a common real-world convention where the channel spacing is the point
    spacing of ``fiber_points``, independent of the gauge length). The
    channel's tangent is estimated from its neighbours by a centred finite
    difference (``(fiber_points[i+1] - fiber_points[i-1]) / 2``); the two
    endpoints of the fibre use a one-sided (forward/backward) difference
    instead. The tangent's *sign* (which neighbour is "forward") does not
    affect the strain rate computed by :func:`strain_rate_from_velocities`
    (the projection is quadratic in the tangent), only its axis.

    The two gauge endpoints for each channel are then placed at
    ``fiber_points[i] +/- (gauge_length / 2) * tangent[i]``, converting the
    physical half-gauge-length offset into a grid-cell displacement with
    ``grid_spacing``.

    Args:
        fiber_points: Ordered fibre sample points, shape ``[n, ndim]``
            (``n >= 2``), in grid cells (float). Consecutive points must be
            distinct.
        gauge_length: The physical gauge length, in the same physical units
            as ``grid_spacing`` (e.g. metres). Must be positive.
        grid_spacing: The physical size of a grid cell: a single number for
            isotropic spacing, or one number per dimension.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

            - plus_locations: Shape ``[n, ndim]``, grid cells: the ``+
              tangent`` gauge endpoint of each channel.
            - minus_locations: Shape ``[n, ndim]``, grid cells: the ``-
              tangent`` gauge endpoint of each channel.
            - tangents: Shape ``[n, ndim]``: the unit tangent of each channel,
              in physical direction cosines (accounts for anisotropic
              ``grid_spacing``).

    """
    if fiber_points.dim() != 2:
        raise _DeepwaveValueError(
            f"fiber_points must be 2-D with shape [n, ndim], got shape "
            f"{list(fiber_points.shape)}.",
        )
    n, ndim = fiber_points.shape
    if n < 2:
        raise _DeepwaveValueError(
            f"fiber_points must contain at least 2 points, got {n}."
        )
    if ndim < 2:
        raise _DeepwaveValueError(
            f"fiber_points must have at least 2 spatial dimensions (a fibre "
            f"needs a tangent), got ndim={ndim}.",
        )
    if not fiber_points.is_floating_point():
        raise _DeepwaveTypeError("fiber_points must be a floating-point tensor.")
    if not bool(torch.isfinite(fiber_points).all()):
        raise _DeepwaveValueError("fiber_points must be finite.")
    if not (math.isfinite(gauge_length) and gauge_length > 0):
        raise _DeepwaveValueError(f"gauge_length must be positive, got {gauge_length}.")

    spacing_list: list[float] = _common.set_grid_spacing(grid_spacing, ndim)
    spacing = torch.tensor(
        spacing_list,
        dtype=fiber_points.dtype,
        device=fiber_points.device,
    )

    # Centred difference in grid-cell space (one-sided at the two ends).
    diffs_cells = torch.empty_like(fiber_points)
    diffs_cells[0] = fiber_points[1] - fiber_points[0]
    diffs_cells[-1] = fiber_points[-1] - fiber_points[-2]
    if n > 2:
        diffs_cells[1:-1] = (fiber_points[2:] - fiber_points[:-2]) / 2

    # Physical-direction vector (not yet unit length): scale each cell-space
    # component by the physical size of a cell in that dimension so that
    # anisotropic grid_spacing does not distort the tangent's direction.
    diffs_phys = diffs_cells * spacing
    norms = torch.linalg.vector_norm(diffs_phys, dim=-1, keepdim=True)
    if bool((norms == 0).any()):
        raise _DeepwaveValueError(
            "a channel has a zero-length tangent, so its fibre direction is "
            "undefined. This happens when consecutive fiber_points are "
            "(physically) coincident, or when the fibre folds back on itself so "
            "that a channel's two neighbours coincide and their centred "
            "difference cancels (a hairpin at gauge scale). Ensure consecutive "
            "points are distinct and the fibre does not reverse within a single "
            "channel spacing.",
        )
    tangents = diffs_phys / norms

    half_offset_phys = tangents * (gauge_length / 2)
    half_offset_cells = half_offset_phys / spacing

    plus_locations = fiber_points + half_offset_cells
    minus_locations = fiber_points - half_offset_cells
    return plus_locations, minus_locations, tangents


def _velocity_channel_names(ndim: int) -> tuple[str, ...]:
    """Returns the velocity channel names of an ``ndim``-dimensional run.

    These are the channel names the propagators themselves use -- the key
    ``"y"`` of a ``receivers=`` mapping, equivalently the suffix of the
    ``receiver_amplitudes_y`` output -- in the library's slowest-to-fastest
    axis order, which is also the column order of a fibre's ``tangents``.
    """
    if ndim not in (2, 3):
        raise _DeepwaveValueError(
            f"tangents must have 2 or 3 columns: a fibre needs a tangent, so "
            f"at least 2 spatial dimensions, and Deepwave models at most 3. "
            f"Got tangents.shape[-1]={ndim}.",
        )
    return ("z", "y", "x")[-ndim:]


def _ordered_velocity_receivers(
    parameter: str,
    v_receivers: Mapping[str, torch.Tensor],
    channels: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    """Returns one mapping's tensors in ``channels`` (tangent-column) order."""
    if not isinstance(v_receivers, Mapping):
        example = ", ".join(
            f"{channel!r}: receiver_amplitudes_{channel}" for channel in channels
        )
        raise _DeepwaveTypeError(
            f"{parameter} must be a mapping from velocity channel name to "
            f"receiver amplitudes, like {{{example}}}, but got "
            f"{type(v_receivers).__name__}. A sequence is no longer accepted, "
            f"because its order had to match the columns of tangents with "
            f"nothing to check it.",
        )
    given = set(v_receivers)
    if given != set(channels):
        expected_text = ", ".join(repr(channel) for channel in channels)
        given_text = ", ".join(repr(key) for key in sorted(given, key=str)) or "no keys"
        raise _DeepwaveValueError(
            f"{parameter} must contain exactly one entry per velocity "
            f"channel. For tangents with {len(channels)} columns the expected "
            f"keys are {expected_text}, but got {given_text}.",
        )
    return tuple(v_receivers[channel] for channel in channels)


def strain_rate_from_velocities(
    v_receivers_plus: Mapping[str, torch.Tensor],
    v_receivers_minus: Mapping[str, torch.Tensor],
    tangents: torch.Tensor,
    gauge_length: float,
) -> torch.Tensor:
    r"""Compute DAS strain rate from paired particle-velocity receivers.

    Combines particle-velocity receivers recorded at the two gauge endpoints
    (typically the ``plus_locations``/``minus_locations`` returned by
    :func:`fiber_gauge_locations`, fed as receiver locations to
    :func:`deepwave.elastic` or :func:`deepwave.acoustic`, one call per
    spatial component) into the tangential gauge-averaged strain rate

    .. math::

        \dot\varepsilon = \frac{v_t^+ - v_t^-}{L_g}, \qquad
        v_t^\pm = \hat t \cdot v^\pm.

    Each velocity component is passed under the channel name the propagators
    use for it (``"y"`` for ``receiver_amplitudes_y``, the same key a
    ``receivers=`` mapping takes), so which component is which is stated
    rather than inferred from position: the columns of ``tangents`` are the
    spatial dimensions in the library's slowest-to-fastest order, so the
    keys are ``"y"`` and ``"x"`` in 2-D and ``"z"``, ``"y"`` and ``"x"`` in
    3-D, and this function matches them up itself.

    Args:
        v_receivers_plus: The particle-velocity receiver amplitudes recorded
            at the ``+ tangent`` gauge endpoints, as a mapping from velocity
            channel name to a tensor of shape ``[shot, channel, time]``: one
            entry per spatial dimension, ``{"y": ..., "x": ...}`` in 2-D and
            ``{"z": ..., "y": ..., "x": ...}`` in 3-D.
        v_receivers_minus: The matching receivers at the ``- tangent`` gauge
            endpoints, with the same keys and shapes as ``v_receivers_plus``.
        tangents: Unit tangents, shape ``[channel, ndim]`` (as returned by
            :func:`fiber_gauge_locations`).
        gauge_length: The physical gauge length used to place the endpoints
            (same value passed to :func:`fiber_gauge_locations`). Must be
            positive.

    Returns:
        torch.Tensor: Strain rate, shape ``[shot, channel, time]``.

    Raises:
        DeepwaveTypeError: If either velocity argument is not a mapping (for
            example a sequence of tensors).
        DeepwaveValueError: If the keys are not exactly the velocity channels
            of a run with ``tangents.shape[-1]`` spatial dimensions, if
            ``gauge_length`` is not positive, or if the receiver tensors do
            not all have the same ``[shot, channel, time]`` shape with
            ``channel`` matching ``tangents.shape[0]``.

    """
    ndim = int(tangents.shape[-1])
    channels = _velocity_channel_names(ndim)
    ordered_plus = _ordered_velocity_receivers(
        "v_receivers_plus",
        v_receivers_plus,
        channels,
    )
    ordered_minus = _ordered_velocity_receivers(
        "v_receivers_minus",
        v_receivers_minus,
        channels,
    )
    if not (math.isfinite(gauge_length) and gauge_length > 0):
        raise _DeepwaveValueError(
            f"gauge_length must be positive, got {gauge_length}.",
        )
    n_channels = tangents.shape[0]
    ref_shape = None
    for v in (*ordered_plus, *ordered_minus):
        if v.dim() != 3 or v.shape[1] != n_channels:
            raise _DeepwaveValueError(
                f"Each velocity receiver tensor must have shape "
                f"[shot, {n_channels}, time] (channel count matching "
                f"tangents.shape[0]), got {list(v.shape)}.",
            )
        if ref_shape is None:
            ref_shape = v.shape
        elif v.shape != ref_shape:
            raise _DeepwaveValueError(
                "All v_receivers_plus/v_receivers_minus tensors must share "
                f"the same [shot, channel, time] shape, got {list(ref_shape)} "
                f"and {list(v.shape)}.",
            )

    v_t_plus = torch.zeros_like(ordered_plus[0])
    v_t_minus = torch.zeros_like(ordered_minus[0])
    for i in range(ndim):
        t_i = tangents[:, i].reshape(1, -1, 1)
        v_t_plus = v_t_plus + t_i * ordered_plus[i]
        v_t_minus = v_t_minus + t_i * ordered_minus[i]

    strain_rate: torch.Tensor = (v_t_plus - v_t_minus) / gauge_length
    return strain_rate


def gauge_average(
    data: torch.Tensor,
    gauge_length: float,
    channel_spacing: float,
    dim: int = -2,
) -> torch.Tensor:
    r"""Apply an interrogator's finite-gauge moving average to per-channel data.

    :func:`strain_rate_from_velocities` already models the gauge physically,
    as a two-point difference (the interferometric measurement itself). This
    function models a *second*, independent source of gauge averaging that
    shows up whenever a fibre is sampled far more densely than the gauge
    length: many interrogators report each channel as a **moving average
    over the gauge window** of an underlying finer-grained per-channel
    signal (e.g. many closely spaced two-point strain-rate estimates, or any
    other per-channel quantity), rather than (or in addition to) a single
    two-point difference spanning the whole gauge. Applying this to the
    output of :func:`strain_rate_from_velocities`, computed with a small
    internal two-point gauge on densely spaced channels, approximates that
    measurement.

    Given a window of ``n = round(gauge_length / channel_spacing)`` channels
    (clamped to at least 1; ties round to even, Python's ``round()``), the
    output at channel ``i`` is the mean of the ``n`` channels
    ``[i - left, i + right]`` (channels beyond the array ends are replicated
    from the nearest in-bounds channel, so the array length and channel
    positions are unchanged):

    .. math::

        \bar d_i = \frac{1}{n} \sum_{k=-\text{left}}^{\text{right}} d_{i+k}.

    **Even/odd window convention.** For odd ``n``
    (``left == right == (n - 1) / 2``), the window is exactly centred on
    channel ``i``: an unbiased centred moving average. For even ``n``, no
    window of channels is exactly centred on any one channel (the true
    window centre falls half a channel spacing away, between two channels);
    this function uses ``left = n // 2``, ``right = n - 1 - left``, i.e. the
    window includes one *more* channel on the lower-index side than the
    higher-index side. Equivalently, the value reported at channel ``i`` is
    the average actually centred at position ``i - 0.5`` channels. If this
    half-channel offset matters for your application (e.g. exact two-way
    time picks), shift your channel coordinates by ``-0.5 * channel_spacing``
    when ``n`` is even, or choose ``gauge_length``/``channel_spacing`` so
    that ``n`` is odd.

    Args:
        data: Per-channel data of any shape, e.g. ``[shot, channel, time]``
            (the default ``dim=-2`` matches the channel axis of
            :func:`strain_rate_from_velocities`'s output).
        gauge_length: The physical gauge length (same physical units as
            ``channel_spacing``, e.g. metres). Must be positive.
        channel_spacing: The physical spacing between adjacent channels
            (same physical units as ``gauge_length``). Must be positive.
        dim: The channel dimension of ``data`` to average over. Defaults to
            ``-2``.

    Returns:
        torch.Tensor: Gauge-averaged data, the same shape and dtype as
        ``data``.

    """
    if not (math.isfinite(gauge_length) and gauge_length > 0):
        raise _DeepwaveValueError(f"gauge_length must be positive, got {gauge_length}.")
    if not math.isfinite(channel_spacing) or channel_spacing <= 0:
        raise _DeepwaveValueError(
            f"channel_spacing must be positive, got {channel_spacing}."
        )
    ndim = data.dim()
    if not -ndim <= dim < ndim:
        raise _DeepwaveValueError(
            f"dim={dim} is out of range for a tensor with {ndim} dimension(s).",
        )
    dim = dim % ndim

    window = max(1, round(gauge_length / channel_spacing))
    if window == 1:
        return data
    left = window // 2
    right = window - 1 - left

    # Window-split convention (replicate padding at the ends;
    # left = window // 2, right = window - 1 - left) cross-referenced from
    # SWEEP's `gauge_average` (github.com/DeepWave-KAUST/sweep, MIT licence,
    # src/sweep/equations/das.py, commit
    # 9eff76deadabcc7c84de761f35ad42637e5a959e), per the Kuvshinov/Zhao
    # simulation-paper moving-average model it cites. Reimplemented
    # independently here (torch avg_pool1d rather than their manual
    # unfold/F.pad combination); not a verbatim copy.
    moved = data.movedim(dim, -1)
    shape = moved.shape
    flat = moved.reshape(-1, 1, shape[-1])
    # `_common.replicate_pad` rather than `F.pad(mode="replicate")`: the two
    # agree bitwise going forward, but the latter's CUDA backward scatters
    # the padded elements' gradients onto the edge channels with an
    # `atomicAdd`, which makes this function's gradient depend on execution
    # order. See its docstring.
    padded = _common.replicate_pad(flat, (left, right))
    averaged = torch.nn.functional.avg_pool1d(padded, kernel_size=window, stride=1)
    return averaged.reshape(shape).movedim(-1, dim)


def helical_fiber_response(
    axial_strain_rate: torch.Tensor,
    transverse_strain_rates: Sequence[torch.Tensor],
    *,
    wrap_angle: float,
) -> torch.Tensor:
    r"""Combine cable-frame normal strain rates into a helical-fibre response.

    Real DAS cables often wind the fibre helically around a central strength
    member rather than laying it straight; the fibre's *local* strain then
    responds to a weighted combination of the *cable's* normal strain-rate
    components, not just the axial one. Following Kuvshinov (2016,
    *Geophysical Prospecting* 64(3), 671-688, "Interaction of helically
    wound fibre-optic cables with plane seismic waves",
    doi:10.1111/1365-2478.12303), for a fibre wound at **wrap angle**
    :math:`\alpha` to the cable axis, in radians (:math:`\alpha = 0` is a
    straight fibre along the cable; :math:`\alpha = \pi/2` is a fibre wound
    purely circumferentially), and neglecting the shear-strain terms that
    time-average to zero for a symmetric periodic helix, the effective
    axial strain rate the fibre reports is

    .. math::

        \dot\varepsilon_\text{helix} = \cos^2\alpha \; \dot\varepsilon_{zz}
            + \frac{\sin^2\alpha}{2} \left( \dot\varepsilon_{xx}
            + \dot\varepsilon_{yy} \right),

    where :math:`z` is the cable's local axial direction (the deployment
    tangent :math:`\hat t` used throughout this module) and :math:`x, y` are
    *any* orthonormal pair spanning the plane normal to it -- the
    transverse-plane sum :math:`\dot\varepsilon_{xx} + \dot\varepsilon_{yy}`
    is invariant to that choice (it equals
    :math:`\operatorname{tr}(\dot\varepsilon) - \dot\varepsilon_{zz}`, the
    strain-rate tensor's trace restricted to the transverse plane, which
    does not depend on the transverse basis), so any orthonormal transverse
    pair gives the same answer. The three weights sum to 1
    (:math:`\cos^2\alpha + \sin^2\alpha/2 + \sin^2\alpha/2 = 1`): this is
    always a weighted average of the three cable-frame normal strain rates.

    :math:`\alpha = 0` reduces this exactly to
    :func:`strain_rate_from_velocities`'s straight-fibre axial response
    (weight 1 on the axial term, 0 on the transverse terms). At the
    **"magic angle"** :math:`\alpha = \arccos(1/\sqrt3) \approx 0.9553`
    radians (:math:`54.7356^\circ`), :math:`\cos^2\alpha = \sin^2\alpha / 2
    = 1/3`: the three
    weights are exactly equal, so the fibre responds only to the trace of
    the strain-rate tensor, i.e. it becomes a pure **volumetric-strain
    (dilatation)** sensor, insensitive to shear -- the winding angle real
    interrogator cables commonly use because it removes shear-wave/
    orientation ambiguity.

    How to obtain the inputs from Deepwave's velocity receivers:
        ``axial_strain_rate`` is exactly the output of
        :func:`strain_rate_from_velocities`, using the real fibre's tangent
        (:func:`fiber_gauge_locations`'s ``tangents``), as in this module's
        main recipe.

        Each entry of ``transverse_strain_rates`` uses the *same* two-point
        gauge construction, but oriented along a direction :math:`\hat n`
        orthogonal to the tangent instead of along it: given the channel
        points and a unit :math:`\hat n` in physical direction-cosine space
        (orthogonal to ``tangents``),

        .. code-block:: python

            half_offset_cells = gauge_length / 2 * n_hat / grid_spacing
            n_plus = channel_points + half_offset_cells
            n_minus = channel_points - half_offset_cells

        (the same arithmetic :func:`fiber_gauge_locations` uses internally
        for the tangent) -- record velocities at ``n_plus``/``n_minus`` and
        call :func:`strain_rate_from_velocities` with :math:`\hat n` in
        place of ``tangents``. Because :math:`\dot\varepsilon_{nn}` is
        quadratic in :math:`\hat n`, its sign does not matter.

        * **2-D** (``elastic``/``acoustic`` with ``ndim=2``): there is only
          one in-plane transverse direction (rotate the physical tangent
          ``(t0, t1) -> (-t1, t0)``, a 90-degree rotation that preserves
          unit length regardless of grid anisotropy since it is done in
          physical space). The 2-D propagators enforce plane strain (no
          out-of-plane motion), so the missing out-of-plane transverse
          component is exactly 0 -- pass a length-1 sequence containing just
          the in-plane transverse strain rate.
        * **3-D**: pass both orthogonal transverse strain rates as a
          length-2 sequence (any orthonormal pair spanning the plane normal
          to the tangent; see the trace argument above for why the specific
          choice does not matter).

    Note:
        This function implements Kuvshinov's continuous formula for general
        :math:`\alpha` from first principles. SWEEP's ``helical_das_response``
        (github.com/DeepWave-KAUST/sweep, MIT licence,
        ``src/sweep/equations/das.py``, commit
        ``9eff76deadabcc7c84de761f35ad42637e5a959e``) was consulted as a
        cross-reference for the overall API shape (normal strain-rate
        components + wrap angle in, one fibre-response tensor out), but it
        hardcodes only its two named angles with unnormalised integer
        weights (1:1:1 at its "35.3 deg", 4:1:1 at its "54.7 deg", i.e. not
        divided by the sum of weights); no code from it is used here.

    Args:
        axial_strain_rate: The cable-axial normal strain rate
            :math:`\dot\varepsilon_{zz}`, any shape (typically ``[shot,
            channel, time]``).
        transverse_strain_rates: A sequence of 1 (2-D, plane strain) or 2
            (3-D) transverse normal strain-rate tensors, each the same shape
            as ``axial_strain_rate``.
        wrap_angle: The helical wrap angle :math:`\alpha`, in **radians**
            (like every other angle in Deepwave), measured from the cable
            axis: 0 is a straight fibre, ``math.pi / 2`` is purely
            circumferential winding, and the magic angle is
            ``math.acos(1 / math.sqrt(3))``. Keyword-only. An angle in
            degrees is converted with ``math.radians``.

    Returns:
        torch.Tensor: The helical-fibre effective strain rate
        :math:`\dot\varepsilon_\text{helix}`, the same shape as
        ``axial_strain_rate``.

    Raises:
        DeepwaveValueError: If ``transverse_strain_rates`` does not contain 1
            or 2 tensors, or if one of them does not have the same shape as
            ``axial_strain_rate``.

    """
    if len(transverse_strain_rates) not in (1, 2):
        raise _DeepwaveValueError(
            "transverse_strain_rates must contain 1 (2-D, plane-strain) or "
            f"2 (3-D) tensors, got {len(transverse_strain_rates)}.",
        )
    for t in transverse_strain_rates:
        if t.shape != axial_strain_rate.shape:
            raise _DeepwaveValueError(
                "Each transverse_strain_rates entry must have the same "
                f"shape as axial_strain_rate ({list(axial_strain_rate.shape)}), "
                f"got {list(t.shape)}.",
            )

    weight_axial = math.cos(wrap_angle) ** 2
    weight_transverse = math.sin(wrap_angle) ** 2 / 2

    transverse_sum = transverse_strain_rates[0]
    for t in transverse_strain_rates[1:]:
        transverse_sum = transverse_sum + t

    result: torch.Tensor = (
        weight_axial * axial_strain_rate + weight_transverse * transverse_sum
    )
    return result

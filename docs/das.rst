DAS (distributed acoustic sensing)
==================================

.. py:module:: deepwave.das

Distributed acoustic sensing (DAS) turns a fibre-optic cable into a dense
line of seismic sensors, and has become one of the fastest-growing
acquisition modalities (downhole, along roads and railways, on dark telecom
fibre, on the seafloor). An interrogator unit sends laser pulses down the
fibre and measures the phase of the light Rayleigh-backscattered from each
section of it. That phase is sensitive to the **axial strain** of the fibre,
so the instrument effectively records strain (or, more commonly, strain
rate) *along the fibre direction* at thousands of positions ("channels")
simultaneously.

Deepwave models DAS data with the :mod:`deepwave.das` module, entirely as
receiver-side post-processing of the particle-velocity receivers that the
elastic (and acoustic) propagators already provide -- no new propagator or
kernel is involved, and everything is differentiable, so DAS data can be
used directly as an FWI objective.

What DAS measures
-----------------

For a fibre with unit tangent :math:`\hat t` at position :math:`x`, the
axial strain rate is the tangential-tangential component of the strain-rate
tensor of the particle-velocity field :math:`v`:

.. math::

    \dot\varepsilon_{tt}(x, t)
        = \hat t \cdot \bigl(\nabla v(x, t)\bigr) \cdot \hat t
        = \sum_{ij} t_i \, t_j \, \frac{\partial v_i}{\partial x_j}.

A real interrogator does not measure this pointwise quantity. It compares
the phase of backscattered light from two sections of fibre a **gauge
length** :math:`L_g` apart (typically metres to tens of metres), which is
the strain rate *averaged over the gauge* -- exactly a finite difference of
the tangential particle velocity between the two gauge ends:

.. math::

    \dot\varepsilon(x, t) \approx
        \frac{v_t(x + \tfrac12 L_g \hat t,\, t)
            - v_t(x - \tfrac12 L_g \hat t,\, t)}{L_g},
    \qquad v_t = \hat t \cdot v.

This finite-difference-of-two-points form is both the physical truth (the
interferometric gauge) and the natural implementation on top of Deepwave:
place *pairs* of particle-velocity receivers at the two gauge ends, project
onto the tangent, difference, and divide by :math:`L_g`. The gauge ends are
generally off-grid; Deepwave's floating-point receiver coordinates place
them exactly (band-limited Hicks interpolation, with the staggered velocity
components handled per-component -- see :doc:`usage`). As
:math:`L_g \to 0` the recipe converges (quadratically, like any centred
difference) to the pointwise :math:`\dot\varepsilon_{tt}`. Keeping
:math:`L_g` finite instead reproduces what an instrument with that gauge
length actually reports, including the loss of sensitivity when the gauge
spans an appreciable fraction of a wavelength.

Broadside insensitivity (:math:`\cos^2\theta` directivity)
----------------------------------------------------------

The double projection onto :math:`\hat t` gives DAS its characteristic
directivity. For a plane P wave travelling in direction :math:`\hat n` with
particle velocity :math:`v = A\, \hat n\, g(t - \hat n \cdot x / c)`,

.. math::

    \frac{\partial v_i}{\partial x_j} = -\frac{A}{c}\, n_i\, n_j\,
        g'(\cdot)
    \quad\Longrightarrow\quad
    \dot\varepsilon_{tt} = -\frac{A}{c}\, g'(\cdot)\,
        (\hat t \cdot \hat n)^2
      = -\frac{A}{c}\, g'(\cdot) \cos^2\theta,

where :math:`\theta` is the angle between the fibre and the propagation
direction. A fibre is therefore most sensitive to P waves travelling *along*
it and blind to P waves crossing it at right angles ("broadside"), with the
amplitude scale set by :math:`(\omega / c)\, |v|`. This falls out of the
implementation automatically and is verified numerically in
``tests/test_das.py``: P waves crossing a straight fibre at
:math:`0^\circ/45^\circ/90^\circ` give measured amplitude ratios of
:math:`1 : 0.507 : 0.019` against the theoretical :math:`1 : 0.5 : 0`.
(S waves have a different pattern, :math:`\sin\theta\cos\theta`, which the
same machinery also reproduces automatically -- it is all just
:math:`\hat t \cdot \nabla v \cdot \hat t`.)

Recipe: DAS with the elastic propagator
---------------------------------------

The fibre is described by an ordered ``[n, ndim]`` tensor of points in grid
cells (a polyline; one DAS channel per point).
:func:`deepwave.das.fiber_gauge_locations` turns it into per-channel gauge
endpoint pairs and unit tangents; the endpoints become float-coordinate
velocity receiver locations (each spatial component recorded at the same
physical points); :func:`deepwave.das.strain_rate_from_velocities` combines
the recorded velocities into strain rate::

    import torch
    import deepwave
    from deepwave import das

    # Fibre: a straight cable at depth, along x (points in grid cells).
    fiber_points = torch.stack(
        [
            torch.full((100,), 60.0),          # y (cells)
            40.0 + torch.arange(100.0),        # x (cells): one channel/cell
        ],
        dim=1,
    )
    gauge_length = 20.0                        # physical units (e.g. m)
    plus, minus, tangents = das.fiber_gauge_locations(
        fiber_points, gauge_length, grid_spacing=5.0,
    )

    # Record both velocity components at both gauge ends of every channel.
    locations = torch.cat([plus, minus]).unsqueeze(0)  # [1, 2n, 2], float
    out = deepwave.elastic(
        lamb, mu, buoyancy, 5.0, dt,
        source_amplitudes_y=source_amplitudes,
        source_locations_y=source_locations,
        receiver_locations_y=locations,
        receiver_locations_x=locations,
        pml_freq=freq,
    )

    n = fiber_points.shape[0]
    vy, vx = out.receiver_amplitudes_y, out.receiver_amplitudes_x
    strain_rate = das.strain_rate_from_velocities(
        {"y": vy[:, :n], "x": vx[:, :n]},   # + gauge ends
        {"y": vy[:, n:], "x": vx[:, n:]},   # - gauge ends
        tangents,
        gauge_length,
    )                                       # [shot, channel, time]

Each velocity component is passed under the channel name the propagators
use for it (``"y"`` for ``receiver_amplitudes_y``, the same key that a
``receivers=`` mapping takes), so nothing depends on the order you write
them in: the function matches the keys to the columns of ``tangents``
itself, using the library's slowest-to-fastest axis order. In 3-D, also
record ``receiver_locations_z`` and pass
``{"z": vz, "y": vy, "x": vx}``. Because the velocity components live at
half time steps on the staggered grid, the strain-rate samples correspond
to times :math:`(t - 0.5)\,\Delta t`, like every velocity receiver output.

**Which propagators?** The elastic propagator is the natural home for DAS
(fibres respond to elastic ground motion, including S waves). The
**acoustic** propagator also provides particle-velocity receivers
(``receiver_locations_z``/``_y``/``_x``), so the identical recipe applies to
it. The **scalar** propagator records only its scalar wavefield -- there are
no particle-velocity receivers to pair up -- so it is not supported.

Gauge averaging as post-processing
-----------------------------------

:func:`~deepwave.das.strain_rate_from_velocities` already models the gauge
physically, as the two-point difference of the interferometric measurement
itself. A second, independent form of gauge averaging shows up whenever the
fibre is sampled far more densely than the gauge length: interrogators
commonly report each channel as a **moving average over the gauge window**
of an underlying finer-grained per-channel signal, rather than (or in
addition to) a single two-point difference spanning the whole gauge.
:func:`~deepwave.das.gauge_average` implements exactly that -- a windowed
mean along the channel dimension of any per-channel tensor::

    smoothed = das.gauge_average(
        strain_rate, gauge_length=20.0, channel_spacing=1.0, dim=-2,
    )

For a window of ``n = round(gauge_length / channel_spacing)`` channels, an
odd ``n`` gives an exactly centred moving average; an even ``n`` (no window
of channels is exactly centred on any one channel) uses the convention
``left = n // 2, right = n - 1 - left``, i.e. the reported value at channel
``i`` is the average actually centred half a channel toward the
lower-index side (see the function's docstring for the exact statement and
the boundary/replicate-padding behaviour).

Helically wound fibres
-----------------------

Real DAS cables often wind the fibre helically around a central strength
member rather than laying it straight. Following Kuvshinov (2016,
*Geophysical Prospecting* 64(3), 671-688, "Interaction of helically wound
fibre-optic cables with plane seismic waves", doi:10.1111/1365-2478.12303):
at wrap angle :math:`\alpha` to the cable axis, the fibre's effective axial
strain rate is a weighted combination of the cable-frame normal strain
rates (the keyword-only ``wrap_angle``, :math:`\alpha` in radians),

.. math::

    \dot\varepsilon_\text{helix} = \cos^2\alpha \; \dot\varepsilon_{zz}
        + \frac{\sin^2\alpha}{2} \left( \dot\varepsilon_{xx}
        + \dot\varepsilon_{yy} \right),

where :math:`z` is the cable axis (the fibre tangent used everywhere else
in this module) and :math:`x, y` are any orthonormal pair spanning the
transverse plane (their strain-rate *sum* is invariant to that choice).
:math:`\alpha = 0` reduces this exactly to the straight-fibre axial
response above. At the **magic angle**
:math:`\alpha = \arccos(1/\sqrt3) \approx 0.9553` radians
(:math:`54.7356^\circ`), the three weights
are each exactly 1/3, so the fibre becomes a pure volumetric-strain
(dilatation) sensor -- the winding angle real cables commonly use because
it removes shear-wave/orientation ambiguity.

:func:`~deepwave.das.helical_fiber_response` computes
:math:`\dot\varepsilon_\text{helix}` from the axial strain rate (exactly
:func:`~deepwave.das.strain_rate_from_velocities`'s output using the fibre
tangent) and one (2-D, plane strain) or two (3-D) transverse strain rates,
which are obtained with the *same* two-point gauge recipe, oriented normal
to the fibre instead of along it -- see the function's docstring for the
exact endpoint arithmetic and the derivation, and for the attribution note
on SWEEP's ``helical_das_response``, consulted as an API cross-reference
(not a code source; it hardcodes only two angles with unnormalised
integer weights, whereas this implements Kuvshinov's continuous formula).

Units
-----

* ``fiber_points`` (and the returned endpoint locations) are in **grid
  cells**, like every location argument in Deepwave.
* ``gauge_length`` is **physical** (the same units as ``grid_spacing``,
  e.g. metres). :func:`~deepwave.das.fiber_gauge_locations` converts it to a
  cell offset internally; :func:`~deepwave.das.strain_rate_from_velocities`
  divides the physical velocity difference by it, so with velocities in m/s
  and ``gauge_length`` in metres the strain rate comes out in 1/s.
* The returned ``tangents`` are physical direction cosines (anisotropic
  ``grid_spacing`` is accounted for).
* ``channel_spacing`` (:func:`~deepwave.das.gauge_average`) is **physical**,
  the same units as ``gauge_length`` -- it is only ever used as their ratio,
  so any consistent physical unit works.
* ``wrap_angle`` (:func:`~deepwave.das.helical_fiber_response`) is in
  **radians**, measured from the cable axis, like every other angle in
  Deepwave. It is keyword-only. Convert a wrap angle you have in degrees
  with ``math.radians``: the magic angle is
  ``math.acos(1 / math.sqrt(3))`` (0.9553 radians, 54.7356 degrees).

Differentiability
-----------------

:func:`~deepwave.das.strain_rate_from_velocities` is a composition of plain
differentiable torch operations, and the velocity receivers themselves are
differentiable outputs of the propagator, so the gradient of any loss on
DAS strain-rate channels flows back to the material models
(``lamb``/``mu``/``buoyancy``), the source amplitudes, and any other
differentiable input -- FWI with DAS data requires no additional setup,
exactly like FWI on conventional receivers. This is checked against a
finite difference in ``tests/test_das.py``.
:func:`~deepwave.das.gauge_average` (a windowed
mean) and :func:`~deepwave.das.helical_fiber_response` (a weighted sum) are
likewise plain differentiable torch operations, checked directly against
their known constant gradients in ``tests/test_das.py``.

Limitations
-----------

* **Straight-segment tangents.** Tangents are estimated by centred finite
  differences of the ``fiber_points`` polyline (one-sided at the fibre's two
  ends), which assumes the fibre is locally straight between neighbouring
  points. Sample a curved fibre finely relative to its radius of curvature,
  or the tangents (and the placement of the gauge ends, which lie on the
  local tangent line rather than the curved fibre) become approximate.
* **Gauge ends must stay in-model.** No bounds checking is performed here;
  keep the fibre, plus half a gauge length at each end (plus the Hicks
  interpolation halfwidth), inside the model, or the propagator will raise.
* **Strain (as opposed to strain-rate) output** is not provided as a
  separate helper: Deepwave's elastic receivers record particle *velocity*,
  whose gauge difference is strain *rate*. If your instrument delivers
  strain, integrate the strain rate over time (e.g.
  ``torch.cumsum(strain_rate, dim=-1) * dt``), or differentiate your
  observed strain data instead, whichever better suits your processing.
* No 1-D support (a fibre needs a tangent direction, so at least two
  spatial dimensions and two fibre points).
* :func:`~deepwave.das.helical_fiber_response` neglects the cable-frame
  shear-strain terms, following Kuvshinov (2016): they average to zero over
  one wrap for a symmetric periodic helix. It also does not build the
  transverse gauge endpoints for you (unlike
  :func:`~deepwave.das.fiber_gauge_locations` for the axial case) -- see its
  docstring for the short recipe.

API
---

fiber_gauge_locations Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.das.fiber_gauge_locations

strain_rate_from_velocities Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.das.strain_rate_from_velocities

gauge_average Function
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.das.gauge_average

helical_fiber_response Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.das.helical_fiber_response

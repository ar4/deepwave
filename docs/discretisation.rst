Discretisation: the exact scheme
================================

This page derives the shared interior discretisation and gives its exact
coefficients: the time-stepping scheme, the spatial
finite-difference stencils, the stability (CFL) limit and where it comes from,
the grid sampling needed to control dispersion, and the source/receiver and
resampling conventions. Everything below is the scheme Deepwave *actually*
runs; the coefficients are transcribed from ``src/deepwave/regular_grid.h`` and
``src/deepwave/staggered_grid.h``, and the stability numbers are re-derived from
those same coefficients in the :ref:`appendix <cfl-appendix>`.

.. _time-stepping:

Time stepping
-------------

The non-attenuating interior wave equations use an explicit,
**second-order-accurate leapfrog** (centred-in-time) update. Constitutive
attenuation and absorbing-boundary memory updates are described on the
:doc:`viscoacoustic` and :doc:`boundaries` pages. Time is the only dimension whose accuracy
order is fixed; the spatial order is the user-selectable ``accuracy``.

Regular-grid scalar propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scalar`` and ``scalar_born`` propagators solve

.. math::

   \frac{\partial^2 u}{\partial t^2} = v^2\,\nabla^2 u - v^2 f ,

where :math:`u` is the modelled field, :math:`v` the wavespeed, and :math:`f`
the source term. On a single regular grid the leapfrog update is (ignoring the
PML memory variables, which add absorbing terms but not time levels)

.. math::

   u^{n+1}_{\mathbf i} = 2\,u^{n}_{\mathbf i} - u^{n-1}_{\mathbf i}
     + (v_{\mathbf i}\,\Delta t)^2\left(\nabla^2_h u^{n} - f^{n}\right)_{\mathbf i},

with :math:`\nabla^2_h` the discrete Laplacian below and :math:`f` the
public source density defined in :doc:`units`. The homogeneous update is the
following line in ``scalar.c``; the separately injected source has already been
scaled by :math:`-v^2\Delta t^2` in ``scalar.py``:

.. code-block:: c

   wfp_t[i] = v_shot[i] * v_shot[i] * dt2 * w_sum + 2 * wfc_t[i] - wfp_t[i];

where ``w_sum`` is :math:`(\nabla^2_h u^n + \text{PML terms})`, ``wfc`` is
:math:`u^n`, and ``wfp`` holds :math:`u^{n-1}` on entry and :math:`u^{n+1}` on
exit (the two-time-level ring buffer, see below).

Staggered-grid acoustic and elastic propagators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``acoustic`` and ``elastic`` propagators use the **Virieux (1986)
velocity--stress staggered grid** (extended to higher spatial order in the
manner of **Levander, 1988**): the pressure/stress and the particle-velocity
components live on grids offset by half a cell in space *and* are advanced on
time levels offset by half a step. Schematically, for the acoustic system
(pressure :math:`p`, particle velocity :math:`\mathbf w`, buoyancy
:math:`b=1/\rho`, modulus :math:`\kappa=\rho v^2`),

.. math::

   \mathbf w^{\,n+\frac12} &= \mathbf w^{\,n-\frac12}
        - \Delta t\, b\, \nabla_h^{+} p^{\,n}, \\
   p^{\,n+1} &= p^{\,n}
        - \Delta t\, \kappa\, \nabla_h^{-}\!\cdot \mathbf w^{\,n+\frac12} + \dots,

where :math:`\nabla_h^{\pm}` are the *staggered first-derivative* operators
below (forward/backward half-cell). The elastic propagator advances the
velocity--stress system of the same family. Because velocity and stress are each
updated from the other, this is still a single second-order leapfrog in time;
the half-step time stagger is the standard velocity--stress arrangement, not an
extra time level.

Two-time-level state and quiescent initial conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scalar recurrence needs two field levels, :math:`u^n` and
:math:`u^{n-1}` (``wavefield_0`` and ``wavefield_m1``). The staggered
first-order systems instead retain one level of each component: stress at
:math:`n\Delta t` and velocity at :math:`(n-1/2)\Delta t`. Absorbing-boundary
and constitutive memory is carried in ``auxiliary_state``. Together these form
the complete continuation state. Resuming with ``initial_state`` preserves the
recurrence when the grid, internal step and coefficients are the same.
Whole-record FFT resampling must also be preserved; independently resampling
source segments need not reproduce one full call (see :doc:`operator_contract`).

When you do not supply initial wavefields they default to zero. Zero in *both*
stored levels is a **consistent quiescent past**: it represents a medium that
was exactly at rest for all :math:`t<0` (:math:`u\equiv 0`,
:math:`\partial_t u\equiv 0`), not merely a single zeroed snapshot. The
propagator's boundary bookkeeping relies on this — the C sources document that
"the first and last ``accuracy/2`` elements in each spatial dimension are zero
in all wavefields" and that the auxiliary PML fields start zero — so the default
run begins from a physically consistent, source-driven initial-value problem.

.. _spatial-stencils:

Spatial finite-difference stencils
----------------------------------

Deepwave uses **central Taylor (maximally accurate)** finite-difference
coefficients — the classical weights of **Fornberg (1988)**. The order is set by
``accuracy`` :math:`M\in\{2,4,6,8\}`, and the *radius* of the stencil (the number
of guard cells, ``FD_PAD``) is :math:`M/2`. All coefficients below are exact
rationals as they appear in the source (stored as ``double`` literals such as
``4.0/12.0``).

Regular-grid second derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scalar Laplacian uses the central second-derivative stencil

.. math::

   \left.\frac{\partial^2 u}{\partial x^2}\right|_{i}
     \approx \frac{1}{\Delta x^2}\left(c_0\,u_i
        + \sum_{m=1}^{M/2} c_m\,(u_{i+m}+u_{i-m})\right).

.. list-table:: Central second-derivative weights :math:`c_m` (from ``regular_grid.h``)
   :header-rows: 1
   :widths: 8 16 16 16 16 16

   * - order
     - :math:`c_0`
     - :math:`c_1`
     - :math:`c_2`
     - :math:`c_3`
     - :math:`c_4`
   * - 2
     - :math:`-2`
     - :math:`1`
     -
     -
     -
   * - 4
     - :math:`-\tfrac{5}{2}`
     - :math:`\tfrac{4}{3}`
     - :math:`-\tfrac{1}{12}`
     -
     -
   * - 6
     - :math:`-\tfrac{49}{18}`
     - :math:`\tfrac{3}{2}`
     - :math:`-\tfrac{3}{20}`
     - :math:`\tfrac{1}{90}`
     -
   * - 8
     - :math:`-\tfrac{205}{72}`
     - :math:`\tfrac{8}{5}`
     - :math:`-\tfrac{1}{5}`
     - :math:`\tfrac{8}{315}`
     - :math:`-\tfrac{1}{560}`

The first-derivative terms that appear only inside the PML (``DIFFZ1`` etc.) use
the matching central *first*-derivative weights :math:`\{\tfrac12\}`,
:math:`\{\tfrac{2}{3},-\tfrac{1}{12}\}`,
:math:`\{\tfrac34,-\tfrac{3}{20},\tfrac{1}{60}\}`,
:math:`\{\tfrac45,-\tfrac15,\tfrac{4}{105},-\tfrac{1}{280}\}` for orders
2/4/6/8.

Staggered-grid first derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The acoustic/elastic propagators evaluate first derivatives *between* grid
points (half-cell staggered). The derivative at the point half a cell above
:math:`i` is

.. math::

   \left.\frac{\partial u}{\partial x}\right|_{i+\frac12}
     \approx \frac{1}{\Delta x}\sum_{m=1}^{M/2}
        w_m\,\bigl(u_{i+m} - u_{i-(m-1)}\bigr),

i.e. :math:`w_m` weights the difference of the pair of samples
:math:`(2m-1)` half-cells either side of the derivative point.

.. list-table:: Staggered first-derivative weights :math:`w_m` (from ``staggered_grid.h``)
   :header-rows: 1
   :widths: 8 20 20 20 20

   * - order
     - :math:`w_1`
     - :math:`w_2`
     - :math:`w_3`
     - :math:`w_4`
   * - 2
     - :math:`1`
     -
     -
     -
   * - 4
     - :math:`\tfrac{9}{8}`
     - :math:`-\tfrac{1}{24}`
     -
     -
   * - 6
     - :math:`\tfrac{75}{64}`
     - :math:`-\tfrac{25}{384}`
     - :math:`\tfrac{3}{640}`
     -
   * - 8
     - :math:`\tfrac{1225}{1024}`
     - :math:`-\tfrac{245}{3072}`
     - :math:`\tfrac{49}{5120}`
     - :math:`-\tfrac{5}{7168}`

.. _cfl:

Stability: the CFL condition
----------------------------

An explicit leapfrog is only stable if the time step is small enough. A von
Neumann analysis of the leapfrog wave update (substitute a plane wave
:math:`u^n_{\mathbf i}=g^n e^{i\mathbf k\cdot\mathbf x_{\mathbf i}}` and require
:math:`|g|\le1`) gives the classical Courant--Friedrichs--Lewy limit

.. math::

   \Delta t\; v_{\max}\;\sqrt{\sum_i \frac{1}{\Delta x_i^2}}\;\le\;C_{\text{crit}},

where the dimensionless critical Courant number :math:`C_{\text{crit}}` depends
only on the spatial stencil (order and regular/staggered scheme). It is
**re-derived exactly** from the stencil coefficients in the
:ref:`appendix <cfl-appendix>`; the results are

.. list-table:: Critical Courant number :math:`C_{\text{crit}}` (derived from the stencils)
   :header-rows: 1
   :widths: 12 22 22

   * - order
     - regular grid
     - staggered grid
   * - 2
     - :math:`1.000`
     - :math:`1.000`
   * - 4
     - :math:`0.866`
     - :math:`0.857`
   * - 6
     - :math:`0.813`
     - :math:`0.805`
   * - 8
     - :math:`0.784`
     - :math:`0.777`

**What Deepwave actually uses.** By default Deepwave does *not* run at these
limits. :func:`deepwave.discretisation.cfl_condition_n` picks the internal time step
from a single **conservative constant Courant number of 0.6**, the same for
every accuracy order and every number of dimensions:

.. math::

   \Delta t_{\text{internal}} \le
     \frac{0.6}{v_{\max}\sqrt{\sum_i 1/\Delta x_i^2}} .

If the requested ``dt`` is larger, the propagator subdivides it into the
smallest integer number of internal steps (``step_ratio``) that satisfies this
bound. One constant covers **all** dimensionalities because the number of
dimensions enters the CFL limit *only* through the
:math:`\sqrt{\sum_i 1/\Delta x_i^2}` term — the critical Courant number itself
is a property of the 1-D stencil, unchanged by adding dimensions. Using 0.6
(rather than, say, 0.78--1.0) provides a margin below the homogeneous stencil
limit. A pointwise wavespeed alone does not bound arbitrary heterogeneous
velocity/stress operators: a high-order derivative can couple a stiff cell
to a distant velocity node with much lower density.

The staggered propagators therefore include a density-weighted operator bound
in the conservative speed reported as ``RunInfo.max_vel``. Acoustic and
viscoacoustic use the weighted derivative stencil; elastic families bound
the constitutive matrix and staggered buoyancy. Spatially constant elastic
materials retain their directional Christoffel speed bound. This can increase automatic
subdivision at sharp material contrasts, even if the pointwise physical
speed is constant. An explicit ``max_vel`` must cover this bound as well as
the physical speed. Use :func:`deepwave.plan` to inspect it before a large
run. The bound concerns the interior energy pairing; physical boundary
closures and PML require their separate validation.

The margin between 0.6 and the true :math:`C_{\text{crit}}` is real headroom.
Because the number of internal steps scales with :math:`1/C`, a low-order run
that is safe at :math:`C=0.9` instead of :math:`0.6` uses
:math:`0.6/0.9\approx0.67\times` as many steps — a ~1.5x speed-up (up to ~1.7x
at :math:`C_{\text{crit}}=1.0`). You can opt in to a larger, still-stable
Courant number via the ``courant_number`` argument of the propagators (see
:doc:`accuracy`); it is validated against the :math:`C_{\text{crit}}` table
above and errors if you request an unstable value. The default (``None``) is
equivalent to explicitly selecting 0.6. The heterogeneous operator correction
can change subdivision relative to earlier releases.

.. _dispersion:

Grid sampling and numerical dispersion
--------------------------------------

Even when stable, a finite-difference wave travels at a slightly wrong,
frequency-dependent speed (numerical *dispersion*). The spatial part depends on
how many grid points sample a wavelength. For a plane wave sampled at :math:`G`
points per wavelength (so :math:`\theta = k\,\Delta x = 2\pi/G`), the numerical
phase velocity of the regular-grid scheme is

.. math::

   \frac{v_{\text{num}}}{v} = \frac{\tilde k(\theta)\,\Delta x}{\theta},
   \qquad
   \tilde k(\theta)^2\Delta x^2 = -\Bigl(c_0 + 2\sum_{m\ge1} c_m\cos m\theta\Bigr),

(axial propagation), with the analogous expression
:math:`\tilde k\,\Delta x = 2\sum_m w_m\sin\!\bigl((2m-1)\theta/2\bigr)` for the
staggered first-derivative scheme.

**Criterion.** For each order we report the smallest points-per-wavelength
:math:`G` at which the axial phase-velocity error
:math:`|v_{\text{num}}/v - 1|` first falls to :math:`\le 1\%` and to
:math:`\le 0.1\%`. This is a *local* error; over a propagation distance
:math:`D` the accumulated phase error grows linearly,
:math:`\Delta\phi = 2\pi\,(D/\lambda)\,|v_{\text{num}}/v-1|`, so a 1% velocity
error slips a full cycle after ~100 wavelengths of propagation. Use the 1%
column for short offsets and the 0.1% column for long models.

.. include:: _generated/points_per_wavelength_table.rst

Deepwave's built-in "at least 6 points per wavelength" warning (see
:doc:`accuracy`) is calibrated for the default ``accuracy=4``: at 6 ppw the
order-4 axial phase error is ~0.6%, order-6 ~0.1%, order-8 ~0.02%, but order-2
is still ~4.5% — a 2nd-order run needs a much finer grid (or a higher order) for
the same fidelity.

.. _suggest-discretisation:

**Letting Deepwave do the arithmetic.**
:func:`deepwave.suggest_discretisation` puts the table above and the CFL
condition together: give it the slowest wavespeed in your model and the
highest frequency you need to propagate, and it returns a cell spacing, a
time step, and a sentence saying which constraint was binding::

    >>> import deepwave
    >>> suggestion = deepwave.suggest_discretisation(1500, 25, 2, max_vel=4500)
    >>> suggestion.grid_spacing, round(suggestion.dt, 6)
    ((10.0, 10.0), 0.000943)

It applies **both** criteria and takes whichever asks for more cells: the
dispersion table above, and the six-cells-per-wavelength floor of Deepwave's
own runtime check. Those two disagree — the 1% column asks for only 5.3, 3.9
and 3.4 cells at orders 4, 6 and 8 — and the floor is what binds there,
because the plane-wave dispersion criterion is a statement about a single
Fourier component on an infinite uniform grid, while the runtime check is a
blunter guard on everything that analysis omits (how a point source, a sharp
contrast and the PML are sampled). A grid the suggestion returns therefore
never provokes Deepwave's own warning. The price is that at the default 1%
target orders 4, 6 and 8 all land on the same six-cell grid; pass
``min_points_per_wavelength=0`` to follow the dispersion table alone, and
expect the warning when you then call a propagator.

The table is **axial**, and axial is the conservative direction (see grid
anisotropy immediately below), so a diagonally travelling wave is slightly
better resolved than the suggestion promises.

**Grid anisotropy.** Numerical dispersion is *not* isotropic: a wave travelling
along a grid axis and one travelling diagonally are dispersed by different
amounts, because the diagonal direction samples the stencil differently. For the
regular 2-D order-2 scheme at 5 ppw, for instance, the axial phase velocity is
:math:`0.936\,v` while the diagonal is :math:`0.967\,v` — the diagonal is faster
(less dispersed). The anisotropy shrinks with more points per wavelength and
with higher order, but never vanishes; keep it in mind when comparing arrivals
at different azimuths.

**Temporal dispersion is separate.** The :math:`O(\Delta t^2)` time-stepping
error is independent of the spatial ``accuracy`` order. Raising ``accuracy``
reduces spatial dispersion but does **nothing** to the temporal dispersion —
that is controlled only by ``dt`` (hence by the Courant number). A dedicated
time-dispersion-correction utility, :mod:`deepwave.time_dispersion` (a
forward/inverse Koene-Robertsson transform of the wavelet and data), is
available for callers who want to reduce temporal dispersion when using a
larger ``dt`` — see :doc:`time_dispersion` and
:doc:`example_time_dispersion`.

.. _source-convention:

Source and receiver convention
------------------------------

A point source in Deepwave is injected as a **cell delta**: the source amplitude
at time step :math:`n` is added to the single grid cell containing the source,
with **no** :math:`1/(\Delta x_1\cdots\Delta x_n)` cell-volume normalisation.
Concretely, in the scalar propagator the amplitude is pre-scaled by
:math:`-v^2\,\Delta t^2` (the ``-v**2 * dt**2`` factor applied in ``scalar.py``
before stepping) and added to one cell — it is *not* divided by the cell area or
volume.

For a point source of strength :math:`S(t)`, the source density supplied at a
cell is :math:`S(t)/\Delta V`, where :math:`\Delta V` is the product of
spatial grid spacings. Equivalently, a fixed supplied amplitude represents a
strength proportional to :math:`\Delta V`. In reduced dimensions that is
strength per unit omitted length or area. See :doc:`units` for the physical
quantity and sign of each channel.

The transpose relation is between **geometrical scatter and gather on the same
component grid**. Adding an unscaled amplitude to a cell is the transpose of
reading that cell under the ordinary tensor dot product. Physical source
injection additionally includes channel-dependent time-step and material
factors; different physical source and receiver channels are not automatically
transposes of one another. The full derivative reverses every operation in the
forward composition, including those factors and resampling; see
:doc:`operator_contract` and :ref:`gradient-guarantees`.

.. _resampling:

Temporal resampling
-------------------

When the internal step is finer than the user's ``dt`` (``step_ratio > 1``),
Deepwave uses Fourier interpolation of a **finite periodic record**, with
a band-limiting projection. It is a periodic sinc interpolation, not an
assumption that the physical record outside the interval is known. :func:`deepwave.discretisation.upsample` takes the real FFT of
the source wavelet, zero-pads the spectrum up to the finer Nyquist (zeroing the
original Nyquist bin so no energy is created above the input band), and inverse
transforms; :func:`deepwave.discretisation.downsample` truncates the spectrum to the
coarser Nyquist (zeroing that bin for even output lengths) before the inverse
transform. Spectral padding and truncation retain only existing Fourier bins; optional cosine frequency tapering and time
padding are available to suppress ringing and wrap-around. A positive taper
fraction does not change the Nyquist-bin rule. Downsampling requires the input
time length to be divisible by ``step_ratio``; pad or trim explicitly when a
different output length is required. Odd output lengths are supported and have
no Nyquist bin to remove. Time-domain windows and finite-record padding can
broaden spectra; the bandlimit statement concerns the Fourier interpolation
step itself. Its normalisation and transpose are derived in
:doc:`operator_contract`.

.. _cfl-appendix:

Appendix: re-deriving the critical Courant numbers
--------------------------------------------------

The :math:`C_{\text{crit}}` table above is derived here directly from the
stencil coefficients, so nothing is taken on faith.

Regular grid
~~~~~~~~~~~~~

Insert a plane wave into the leapfrog update. The discrete second-derivative
operator has Fourier symbol (per dimension, in units of :math:`1/\Delta x^2`)

.. math::

   \widehat{D^2}(\theta) = c_0 + 2\sum_{m\ge1} c_m\cos m\theta ,

which is most negative at :math:`\theta=k\Delta x=\pi` (the grid Nyquist).
Writing :math:`S_M \equiv -\widehat{D^2}(\pi) = -\bigl(c_0 + 2\sum_m c_m(-1)^m\bigr)`
for its maximum magnitude, the leapfrog amplification factor stays on the unit
circle (:math:`|g|=1`) provided

.. math::

   (v\,\Delta t)^2\,S_M\sum_i\frac{1}{\Delta x_i^2}\le 4
   \;\Longleftrightarrow\;
   \Delta t\,v\sqrt{\textstyle\sum_i 1/\Delta x_i^2}\le \frac{2}{\sqrt{S_M}}
   \equiv C_{\text{crit}} .

Evaluating :math:`S_M` from the second-derivative weights:

.. list-table::
   :header-rows: 1
   :widths: 10 22 26

   * - order
     - :math:`S_M`
     - :math:`C_{\text{crit}}=2/\sqrt{S_M}`
   * - 2
     - :math:`4`
     - :math:`1.0000`
   * - 4
     - :math:`16/3`
     - :math:`\sqrt3/2 = 0.8660`
   * - 6
     - :math:`272/45`
     - :math:`0.8135`
   * - 8
     - :math:`2048/315`
     - :math:`0.7844`

Worked example, order 4:
:math:`S_M = -\bigl(-\tfrac52 + 2(-\tfrac43 - \tfrac1{12})\bigr) = \tfrac{16}{3}`,
so :math:`C_{\text{crit}} = 2/\sqrt{16/3} = \sqrt3/2 \approx 0.866`.

Staggered grid
~~~~~~~~~~~~~~~

The staggered first-derivative operator has the *purely imaginary* symbol
:math:`i\,\tilde k(\theta)\Delta x` with
:math:`\tilde k(\theta)\Delta x = 2\sum_m w_m\sin\!\bigl((2m-1)\theta/2\bigr)`.
Its magnitude is maximised at :math:`\theta=\pi`, where every sine term reaches
:math:`\pm1` and aligns with the weight's sign, giving

.. math::

   K_M \equiv \max_\theta |\tilde k(\theta)\Delta x| = 2\sum_m |w_m| .

The velocity--stress leapfrog composes two such first derivatives, so its
stability condition is the same form as above with :math:`\sqrt{S_M}` replaced
by :math:`K_M`:

.. math::

   \Delta t\,v\sqrt{\textstyle\sum_i 1/\Delta x_i^2}\le \frac{2}{K_M}
   \equiv C_{\text{crit}} .

.. list-table::
   :header-rows: 1
   :widths: 10 26 26

   * - order
     - :math:`K_M = 2\sum_m|w_m|`
     - :math:`C_{\text{crit}}=2/K_M`
   * - 2
     - :math:`2`
     - :math:`1.0000`
   * - 4
     - :math:`7/3`
     - :math:`6/7 = 0.8571`
   * - 6
     - :math:`149/60`
     - :math:`0.8054`
   * - 8
     - :math:`2161/840`
     - :math:`0.7774`

Worked example, order 4:
:math:`K_M = 2(\tfrac98 + \tfrac1{24}) = \tfrac73`, so
:math:`C_{\text{crit}} = 2/(7/3) = 6/7 \approx 0.857`.

Both derived tables match the values baked into
``deepwave.discretisation.critical_courant_number`` and quoted at the top of this
page.

References
----------

* J. Virieux (1986), "P-SV wave propagation in heterogeneous media: Velocity-
  stress finite-difference method", *Geophysics* **51**\ (4), 889--901.
* A. R. Levander (1988), "Fourth-order finite-difference P-SV seismograms",
  *Geophysics* **53**\ (11), 1425--1436.
* B. Fornberg (1988), "Generation of finite difference formulas on arbitrarily
  spaced grids", *Mathematics of Computation* **51**\ (184), 699--706.

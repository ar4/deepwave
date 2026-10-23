Units and source conventions
============================

A frequent source of confusion is the amplitude and units of the source and the
recorded data. The conventions differ between the propagators because they
solve different equations, so they are collected here. In all cases the
``source_amplitudes*`` arguments have shape ``[shot, source, time]`` and are
added as a source term at each (internal) time step. The recorded receiver
data has shape ``[shot, receiver, time]`` and contains the value of the
corresponding field sampled at each receiver location.

For most inversion and imaging work the *relative* scaling between sources and
receivers is what matters, and any overall constant cancels or is absorbed into
a calibration; but if you need physically meaningful amplitudes, the
conventions below apply.

.. warning::

   The scalar *source* channel is named for the quantity it injects, and the
   two families inject different quantities. In the acoustic family
   (:func:`~deepwave.acoustic`, :func:`~deepwave.acoustic_born`) it is
   ``volume_rate``, a
   *volume-injection rate per unit volume* with units of 1/time; in the
   elastic family (:func:`~deepwave.elastic`, :func:`~deepwave.elastic_vti`,
   :func:`~deepwave.elastic_born`) and in :func:`~deepwave.acoustic_vti` it
   is ``pressure_rate``, with units of pressure/time. Viscoacoustic also
   uses ``pressure_rate``, applied directly to its pressure equation. The
   volume-rate and pressure-rate quantities differ by a modulus: see :ref:`converting-between-propagators`. Moving a wavelet from
   one family to the other unchanged rescales your source by that modulus
   (of order :math:`10^9` in SI units for rock).

   The letter ``p`` used to name both, which is why it is deprecated as a
   source name. On the released ``acoustic`` and ``elastic`` APIs,
   ``source_amplitudes_p`` and ``sources={"p": ...}`` still work with a
   :class:`deepwave.DeepwaveFutureWarning` until ``0.4.0``. ``p`` is unchanged
   as a *receiver* channel, where it has always meant a pressure.

.. _calibration-table:

Calibration table
-----------------

The quantity each source argument represents, derived from the injection code.
:math:`\lambda` and :math:`\mu` are the Lamé parameters, :math:`K` the bulk
modulus, :math:`\rho` the density, :math:`n` the number of spatial dimensions,
and :math:`\Delta V` the product of the grid spacings (:math:`\Delta z\Delta x`
in 2-D, :math:`\Delta z\Delta y\Delta x` in 3-D). Here :math:`h` is the
internal time step, ``out.info.internal_dt``, which can be smaller than the
user sample interval ``dt``. SI units are given for
concreteness; Deepwave itself is unit-agnostic (see
:ref:`choosing-units-for-ultrasound`).

.. list-table::
   :header-rows: 1
   :widths: 18 16 16 22 12 16

   * - Propagator
     - Argument
     - Enters
     - Physical quantity
     - SI units
     - 2-D versus 3-D
   * - :func:`~deepwave.scalar`, :func:`~deepwave.scalar_born`
     - ``source_amplitudes``
     - the scalar field :math:`u`, pre-scaled by :math:`-v^2h^2`
     - the right-hand side :math:`f` of
       :math:`\nabla^2u-\ddot u/v^2=f`
     - :math:`[u]/\mathrm{m}^2` (:math:`\mathrm{Pa}/\mathrm{m}^2` when
       :math:`u` is pressure)
     - Units unchanged. :math:`f\,\Delta V` is the point-source strength in
       3-D; in 2-D it is the strength *per metre* of a line source along the
       omitted axis.
   * - :func:`~deepwave.acoustic`, :func:`~deepwave.acoustic_born`
     - ``source_amplitudes_volume_rate``
     - pressure :math:`p`, pre-scaled by :math:`h\,K`
     - volume-injection rate per unit volume :math:`s` in
       :math:`K^{-1}\dot p=-\nabla\cdot\mathbf{v}+s`
     - :math:`\mathrm{s}^{-1}`
     - Units unchanged. :math:`s\,\Delta V` is the injected volume rate:
       :math:`\mathrm{m}^3/\mathrm{s}` in 3-D,
       :math:`\mathrm{m}^3/\mathrm{s}` *per metre* of the omitted axis in 2-D.
   * - :func:`~deepwave.viscoacoustic`
     - ``source_amplitudes_pressure_rate``
     - pressure :math:`p`, pre-scaled by :math:`h`
     - additive external pressure rate :math:`a` in the pressure equation;
       no source is added to the relaxation memories
     - :math:`\mathrm{Pa/s}`
     - Units unchanged. :math:`a\,\Delta V` is an integrated pressure-rate
       source in 3-D and per omitted-axis length in 2-D.
   * - :func:`~deepwave.acoustic`, :func:`~deepwave.acoustic_born`,
       :func:`~deepwave.viscoacoustic`
     - ``source_amplitudes_z``, ``_y``, ``_x``
     - the matching velocity component, pre-scaled by
       :math:`h/\rho`
     - body-force density :math:`\mathbf{f}` in
       :math:`\rho\dot{\mathbf{v}}=-\nabla p+\mathbf{f}`
     - :math:`\mathrm{N}/\mathrm{m}^3`
     - Units unchanged. :math:`f\,\Delta V` is the total force
       (:math:`\mathrm{N}`) in 3-D and the force *per metre* of the line
       source (:math:`\mathrm{N}/\mathrm{m}`) in 2-D.
   * - :func:`~deepwave.acoustic_vti`
     - ``source_amplitudes_pressure_rate``
     - both normal stresses, pre-scaled by :math:`h` after a sign
       flip: :math:`s_H=s_V=-\dot p`
     - pressure rate :math:`\dot p` (positive is compressional), exactly as
       for the elastic family
     - :math:`\mathrm{Pa}/\mathrm{s}`
     - Synthesised by a wrapper, not by the kernel: it is bit-identical to
       passing ``source_amplitudes_sigma_v = source_amplitudes_sigma_h =
       -source_amplitudes_pressure_rate``. In the isotropic limit
       (:math:`c_{11}=c_{33}=c_{13}=K`) it was **measured** that
       ``acoustic_vti``'s ``receiver_amplitudes_sigma_v`` under
       ``source_amplitudes_pressure_rate=a`` equals minus
       ``acoustic``'s ``receiver_amplitudes_p`` under
       ``source_amplitudes_volume_rate=a/K``,
       to a relative residual of 8e-16, which is the :math:`K` between a
       volume rate and a pressure rate, and the sign between a
       tension-positive stress and a compression-positive pressure.
   * - :func:`~deepwave.acoustic_vti`
     - ``source_amplitudes_sigma_h``, ``_sigma_v``
     - :math:`\sigma_H` / :math:`\sigma_V`, pre-scaled by :math:`h`
     - stress rate :math:`s_H`, :math:`s_V` (tension positive, so the
       opposite sign convention to ``source_amplitudes_pressure_rate``)
     - :math:`\mathrm{Pa}/\mathrm{s}`
     - Units unchanged. :math:`s\,\Delta V` is a moment rate,
       :math:`\mathrm{N\,m}/\mathrm{s}` in 3-D and per metre in 2-D.
   * - :func:`~deepwave.acoustic_vti`
     - ``source_amplitudes_z``, ``_y``, ``_x``
     - the matching velocity component, pre-scaled by
       :math:`h/\rho`
     - body-force density, as for :func:`~deepwave.acoustic`
     - :math:`\mathrm{N}/\mathrm{m}^3`
     - As for :func:`~deepwave.acoustic`.
   * - :func:`~deepwave.elastic`, :func:`~deepwave.elastic_vti`,
           :func:`~deepwave.elastic_born`
     - ``source_amplitudes_pressure_rate``
     - every normal stress, pre-scaled by :math:`-h`
     - pressure rate :math:`\dot p` (positive is compressional), i.e. an
       isotropic moment-rate source
     - :math:`\mathrm{Pa}/\mathrm{s}`
     - Units unchanged. :math:`\dot p\,\Delta V` is the isotropic moment
       rate, :math:`\mathrm{N\,m}/\mathrm{s}` in 3-D and per metre in 2-D.
   * - :func:`~deepwave.elastic`, :func:`~deepwave.elastic_vti`,
           :func:`~deepwave.elastic_born`
     - ``source_amplitudes_z``, ``_y``, ``_x``
     - the matching velocity component, pre-scaled by
       :math:`h/\rho`
     - body-force density :math:`\mathbf{f}` in
       :math:`\rho\dot{\mathbf{v}}=\nabla\cdot\mathbf{\sigma}+\mathbf{f}`
     - :math:`\mathrm{N}/\mathrm{m}^3`
     - Units unchanged. :math:`f\,\Delta V` is the total force
       (:math:`\mathrm{N}`) in 3-D and the force *per metre* of the line
       source (:math:`\mathrm{N}/\mathrm{m}`) in 2-D.

**Sources are named by quantity; receivers are named by field.** The two
propagator families would otherwise use the same letter ``p`` for two
quantities that differ by roughly the bulk modulus -- about :math:`10^9` in
SI -- so each source channel is named for the quantity it injects
(``volume_rate`` / ``pressure_rate``). The receiver side keeps the plain
``receiver_locations_p``/``receiver_amplitudes_p``, deliberately: it names a
*field* the receiver records, there is no second quantity for it to be
confused with, and renaming it would be churn with no defect behind it.

:func:`~deepwave.acoustic_vti` records its ``p`` receiver as
:math:`-(\sigma_V+(n-1)\sigma_H)/n`, minus the mean normal stress, which is
the same mean-stress definition used by the elastic family; it is not a
claim that different constitutive equations produce identical pressures. It is
formed from the two recordings after the run rather than inside the time
loop, so it costs nothing that recording the two stresses would not.

Receivers record the corresponding field itself: the scalar field :math:`u`,
pressure, a velocity component, or (for :func:`~deepwave.acoustic_vti`) a
normal stress. The elastic family's ``receiver_amplitudes_p`` is the negative
arithmetic *mean* of the recorded normal stresses, so it is a pressure in the
usual compression-positive sense. **In 2-D that mean is over the two in-plane
normal stresses only**, so it is the in-plane mean normal stress rather than
the thermodynamic pressure; the two differ whenever the out-of-plane stress
does not equal the in-plane average.

Every amplitude is a density
----------------------------

Every entry above is a *density* -- a per-unit-volume quantity -- because a
source amplitude is added to a single grid cell with no cell-volume
normalisation (see :ref:`source-convention`). Writing a continuum point source
of strength :math:`S(t)` as :math:`S(t)\,\delta(\mathbf{x}-\mathbf{x}_s)` and
representing :math:`\delta` on the grid as :math:`1/\Delta V` at the source
cell, the number you supply is :math:`S(t)/\Delta V`.

Two consequences follow. First, the same amplitude is a *different* physical
source on a different grid. Halve the grid spacing in 2-D and the equivalent
point-source strength falls by a factor of four, so an unchanged amplitude
gives a receiver amplitude four times smaller. Second, the 2-D and 3-D meanings
differ in exactly the way the last column of the table records. The units of
the number you supply are the same, but a 2-D model is a 3-D model that is
invariant along the omitted axis, so the source is a *line* source, and
:math:`(\text{amplitude})\times\Delta V` is a strength per unit length of that
line rather than a total.

.. _converting-between-propagators:

Converting between propagators
------------------------------

Source conversion requires specifying which observable is to agree. The
relations below are continuum relations for a **homogeneous isotropic medium**,
compatible zero initial fields, no body-force sources, and an unbounded domain
(or the direct arrival before different boundary responses matter). They are
not identities between finite-difference outputs on different grids: spatial
stencils, time staggering, and numerical dispersion can differ.

Let :math:`s(\mathbf{x},t)` be acoustic ``volume_rate`` (units
:math:`\mathrm{s}^{-1}`), :math:`a(\mathbf{x},t)` elastic ``pressure_rate``
(:math:`\mathrm{Pa/s}`), and :math:`f(\mathbf{x},t)` the scalar PDE source
(:math:`\mathrm{Pa/m^2}` when :math:`u` is pressure). Define

.. math::

    M=\lambda+2\mu=\rho v_P^2,\qquad
    K_n=\lambda+\frac{2\mu}{n},\qquad
    A(\mathbf{x},t)=\int_{-\infty}^{t}a(\mathbf{x},\tau)\,d\tau,
    \qquad L=\partial_t^2-v_P^2\nabla^2.

Here :math:`n` is the number of propagated spatial dimensions;
:math:`M,K_n` and :math:`A` have pressure units. In 2D, :math:`K_n=\lambda+\mu`
relates the in-plane mean normal stress to in-plane dilatation; it is not the
3D thermodynamic bulk modulus :math:`K_3=\lambda+2\mu/3`.

**Acoustic pressure to scalar.** For acoustic wavespeed :math:`v_P` and bulk
modulus :math:`M`, differentiating :math:`p_t=-M\nabla\cdot\mathbf{v}+Ms`
and using :math:`\rho\mathbf{v}_t=-\nabla p` gives

.. math::

    Lp=M\,\partial_t s,\qquad
    f=-\rho\,\partial_t s.

The second relation follows because scalar solves :math:`Lu=-v_P^2 f`.
It matches the continuum pressure, including at the source, under the stated
assumptions.

**Elastic mean stress to scalar.** Write :math:`p=-\operatorname{tr}_n\sigma/n`
and :math:`\theta=\nabla\cdot\mathbf{v}`. The elastic equations give

.. math::

    p_t=-K_n\theta+a,\qquad
    \rho\theta_{tt}=M\nabla^2\theta-\nabla^2 a,

and therefore

.. math::

    Lp=\partial_t a-\frac{M-K_n}{\rho}\nabla^2 A,
    \qquad
    p=\frac{K_n}{M}L^{-1}\partial_t a
      +\left(1-\frac{K_n}{M}\right)A.

:math:`L^{-1}` denotes the causal, zero-initial-data wave solution operator.
The second term is **local to the source support**. At receivers outside that
support, elastic pressure can consequently be reproduced by scalar using

.. math::
    :label: elastic_to_scalar

    f=-\frac{K_n}{\rho v_P^4}\,\partial_t a.

At the source this formula alone omits the local :math:`A` term. A scalar
source reproducing the full continuum pressure is instead
:math:`f=-\partial_t a/v_P^2+(M-K_n)\nabla^2 A/M`; its spatial derivative
makes it a different distributed source construction. These formulas do not
extend unchanged across heterogeneous interfaces or elastic mode-converting
boundaries.

**Same imposed volume rate versus same pressure waveform.** An isotropic
injected strain rate :math:`(s/n)I` contributes :math:`-K_n s I` to the
elastic stress-rate equation, so :math:`a=K_n s` represents that same imposed
volume rate. For :math:`\mu>0` it does **not** make elastic mean pressure equal
to acoustic pressure at speed :math:`v_P`. Matching their off-source pressure
instead requires :math:`a=M^2s/K_n`, as follows by equating the two coefficients
of :math:`L^{-1}\partial_t s` above. For a fluid (:math:`\mu=0`),
:math:`K_n=M`, the local term vanishes, and these distinctions collapse to
:math:`a=Ms` and :math:`f=-\rho\partial_t s`.

All three sources are still grid densities: apply the same cell-volume
normalisation if comparing a fixed integrated point or line source.

.. _source-time-indexing:

When each source sample is applied
----------------------------------

Distinguish the user sample interval :math:`dt` from the internal interval
:math:`h=\texttt{out.info.internal_dt}=dt/R`, where
:math:`R=\texttt{out.info.step_ratio}`. The following describes the leapfrog
clock before any optional time tapering or resampling filter effects.

On staggered grids, internal receiver index :math:`k` records stress/pressure
at :math:`kh` and velocity at :math:`(k-\tfrac12)h`. Source index :math:`k`
drives an update:

* a force-density sample drives :math:`\mathbf v^{k-1/2}` to
  :math:`\mathbf v^{k+1/2}`, and is evaluated at :math:`kh`;
* a pressure/stress-rate sample drives stress from :math:`k` to :math:`k+1`,
  and is evaluated at :math:`(k+\tfrac12)h`.

The staggered families downsample without an additional component-dependent
half-step correction. Thus returned pressure sample :math:`j` has nominal time
:math:`j\,dt`, while returned velocity sample :math:`j` has nominal time
:math:`j\,dt-h/2`, **not** :math:`j\,dt-dt/2` when :math:`R>1`.
To supply a pressure-rate waveform with a specified physical time origin,
evaluate its user-rate samples at :math:`j\,dt+h/2`; the bandlimited internal
interpolation then supplies the corresponding internal midpoint samples.
Force samples use :math:`j\,dt`. The remaining finite-record interpolation
and filtering assumptions are described in :doc:`discretisation`.

On the regular scalar grid, source and receiver index :math:`k` both have
nominal time :math:`kh`; there is no staggered half-step offset. Comparing a
staggered solution to a scalar or analytic reference requires accounting for
this distinction. A half-internal-step timing error has narrowband relative
waveform error of order :math:`\omega h/2`, where :math:`\omega` is angular
frequency, not :math:`\omega dt/2` unless no substepping occurs.

Scalar
------

The scalar propagator solves the constant-density scalar wave equation
:math:`\nabla^2u-\ddot u/v^2=f`, with :math:`f` the supplied source amplitude
(internally Deepwave scales each source sample by :math:`-v^2h^2` at the
source cell; you do not apply this yourself). The receiver data is the scalar
wavefield sampled at the receivers. Source and receiver therefore share a
consistent set of units, set by the units of your source amplitudes.

The Born (``scalar_born``) propagator follows the same convention for its
background source, and its scattered output is the linearised (single
scattering) wavefield in the supplied velocity-perturbation direction
``scatter``, which has the same velocity units as ``v``.

Acoustic (variable density)
---------------------------

The acoustic propagator uses a velocity--pressure (first-order) formulation on a
staggered grid. You may inject a volume-injection source
(``source_amplitudes_volume_rate``, added to the pressure update after scaling by
:math:`h\,K`) and force-density sources on any velocity component
(``source_amplitudes_z``/``_y``/``_x``), and you may record pressure and/or the
particle-velocity components, each with its own locations. See the
:doc:`acoustic` API page for the exact ordering of the returned receiver
tensors.

Elastic (isotropic)
-------------------

The elastic propagator injects and records several distinct quantities, each
with its own ``source_amplitudes_*`` / ``*_locations`` / receiver arguments:

* ``*_z``, ``*_y``, ``*_x``: force-density sources, with units
  :math:`\mathrm{N/m^3}`. Internal sample :math:`k` acts at :math:`kh`, while
  the velocity receiver of that internal index records :math:`(k-1/2)h`.
* ``source_amplitudes_pressure_rate``: pressure rate, with units Pa/s, added
  with a negative sign to every normal stress. Internal sample :math:`k`
  acts at :math:`(k+1/2)h`, while the pressure receiver records :math:`kh`.

See :ref:`source-time-indexing` for the user-rate clock when :math:`h<dt`.

Receivers can record the particle-velocity components and/or pressure. The
returned receiver tensors are ordered pressure-first; see the :doc:`elastic`
API page for the precise return order, which differs from the acoustic
propagator.

Viscoacoustic, acoustic VTI and elastic VTI
-------------------------------------------

:mod:`deepwave.viscoacoustic` uses a direct pressure-rate source
:math:`a` (``source_amplitudes_pressure_rate``, Pa/s). With
:math:`\theta=\nabla\cdot\mathbf v`, the sourced constitutive equations are

.. math::

    p_t=-K\left(1+\sum_l\tau_l\right)\theta-\sum_l r_l+a,
    \qquad
    (r_l)_t=-\frac{r_l+K\tau_l\theta}{\tau_{\sigma,l}}.

Here :math:`r_l` has units Pa/s and :math:`\tau_l` is dimensionless. The
source adds :math:`h a` only to pressure. It is not a prescribed eigen-volume
rate, which would replace :math:`\theta` by :math:`\theta-s` in **both**
equations. In the lossless limit a pressure-rate source :math:`a` matches an
acoustic volume-rate source :math:`s=a/K`. Velocity force-density sources and
pressure/velocity receivers follow the acoustic conventions.

The quality factor ``q`` is dimensionless. Single-SLS ``q_freq`` and GSLS
``q_band`` endpoints are cyclic frequencies (Hz for seconds; inverse
microseconds for a microsecond time unit). ``q_freq`` is required for one
mechanism, whereas multiple mechanisms require ``q_band``. The supplied
wavespeed determines the relaxed modulus :math:`K=\rho v^2`, not the phase
speed at the reference frequency; see :doc:`viscoacoustic` and :doc:`materials`.

:mod:`deepwave.acoustic_vti` takes stress-rate sources
on its two normal stresses rather than a pressure source. Because stress is
tension-positive, the same wavelet that is compressional as an elastic
``source_amplitudes_pressure_rate`` is dilatational as an
``source_amplitudes_sigma_*``.

:mod:`deepwave.elastic_vti` follows the same
source/receiver conventions as the elastic propagator. Its stiffness models
(``c11``, ``c13``, ``c33``, ``c55``, and ``c66`` in 3-D) have units of
pressure, e.g. Pa if you are working in SI units, exactly like ``lamb``/``mu``
in the isotropic elastic propagator.

.. _choosing-units-for-ultrasound:

Choosing units for ultrasound
-----------------------------

The equations allow any consistent system of units, provided material,
source, and receiver quantities are converted together. Equivalent physical
results are subject to floating-point conditioning and representability, and
to numerical near-zero guards in vacuum-safe inverses and material averages;
arbitrary changes of unit scale need not give identical numerical results.
Seismic work uses metres and seconds; ultrasound and NDT work at
millimetre/microsecond scales, for which a millimetre--microsecond--megahertz
system gives convenient speed, time, and frequency values (a
water speed of ``1.5`` mm/µs, ``dt = 0.01`` µs, ``pml_freq = 3.0`` MHz). When ``pml_freq`` is not
given, Deepwave infers it from your source amplitudes, so it comes out in your
own frequency unit automatically -- ``3.0`` in the MHz system above, ``3e6`` in
SI -- rather than the fixed seismic ``25`` that Deepwave 0.0.27 and earlier
used regardless of scale. Pass it explicitly anyway for anything you publish:
inference makes the boundary depend on the wavelet (see :ref:`pml-freq`). The :doc:`ultrasound` page gives a full worked unit example,
the excitation wavelets (:func:`~deepwave.wavelets.tone_burst`,
:func:`~deepwave.wavelets.gaussian_pulse`), the transducer-array helpers, and a
concept mapping for readers coming from k-Wave.

Staggered grids and time staggering
-----------------------------------

For the acoustic and elastic propagators the field components live at different
half-cell positions in space and (for velocities) at half steps in time. This
is handled internally, but it is the reason the velocity receivers are
recorded at :math:`j\,dt-h/2` while pressure receivers are at
:math:`j\,dt`, and the reason the same-channel source samples act half an
internal step later (see :ref:`source-time-indexing`). When placing sources or receivers
off-grid with floating point coordinates (see :doc:`usage`), Deepwave's Hicks
interpolation can be made aware of these half-cell component offsets.

The offsets also set the exact form that source--receiver reciprocity takes
when the two channels differ. Swapping a scalar source with a pressure
receiver, or one velocity channel with another, reproduces the trace exactly;
swapping a scalar source with a velocity receiver reverses the sign and
shifts the internal trace by one internal sample, of duration :math:`h`.
After resampling this is a fractional :math:`1/R` of a returned sample when
:math:`R>1`, not an integer shift of the returned array. The two fields are
recorded half an internal step apart and the source samples sit half an
internal step later still. On the elastic
family, whose scalar source channel injects a pressure rate rather than a
volume rate, that relation also carries a factor of the bulk modulus at the
scalar channel's own point. These are exact properties of the discrete
scheme, for sources and receivers anywhere on the grid (including inside the
absorbing boundary), and ``tests/test_reciprocity.py`` checks every channel
pairing against them. They hold as stated only where a single real modulus
converts between the scalar source rate and pressure, so on an attenuating
medium the mixed pairings depart by O(1/Q), and on an anisotropic one by the
departure of the stiffness from isotropy.

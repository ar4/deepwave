Viscoacoustic (attenuating) wave equation
=========================================

.. note::

   Stability: **experimental** -- the label records that this propagator has
   never appeared in a release, so ``0.1.0`` is the first time real use meets
   its design. It is not a lesser promise: its interface is frozen exactly as
   the other eight propagators' are, calling it emits no warning about its
   maturity, and every other Deepwave promise applies to it unchanged. See
   :doc:`stability`.

The :func:`deepwave.viscoacoustic` propagator solves the variable-density
acoustic wave equation augmented with one or more **standard-linear-solid (SLS)**
relaxation mechanism, giving the medium a finite, frequency-dependent quality
factor :math:`Q` (i.e. anelastic attenuation and the physical velocity
dispersion that necessarily accompanies it). It is built on the same
velocity--pressure staggered grid as the :doc:`acoustic <acoustic>`
propagator; only the pressure/constitutive update changes, through one memory
variable per relaxation mechanism carried at the pressure nodes. Reading the
:doc:`acoustic <acoustic>` page first is recommended -- this page describes
only what attenuation adds.

See :doc:`materials` for the material domain, dimensions, and limits of the
interior energy argument, and :doc:`units` for source and receiver normalisation.

Minimal example
^^^^^^^^^^^^^^^^

:func:`deepwave.viscoacoustic` takes the same wavespeed and density models as
:func:`deepwave.acoustic`, plus a quality-factor model ``q``:

.. code-block:: python

    import torch
    import deepwave

    v = 1500 * torch.ones(40, 40)
    rho = 2200 * torch.ones(40, 40)
    q = 20 * torch.ones(40, 40)  # quality factor: smaller means more attenuation
    out = deepwave.viscoacoustic(
        v, rho, q, grid_spacing=5, dt=0.001,
        source_amplitudes_pressure_rate=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations_pressure_rate=torch.tensor([[[5, 20]]]),
        receiver_locations_p=torch.tensor([[[35, 20]]]),
        q_freq=25,  # the frequency at which q is the quality factor, in Hz
        pml_width=10,
        pml_freq=25,
    )
    out.receiver_amplitudes_p  # shape [1, 1, 200]

Governing equations
^^^^^^^^^^^^^^^^^^^^

With :math:`p` the pressure, :math:`\mathbf{v}` the particle velocity,
:math:`\rho` the density, :math:`b=1/\rho` the buoyancy, :math:`K=\rho v^2`
the **relaxed** (zero-frequency) bulk modulus and :math:`r` the SLS memory
variable, and :math:`a` an external pressure rate (Pa/s),

.. math::
    :label: visco_pde

    \begin{aligned}
    \rho\,\partial_t \mathbf{v} &= -\nabla p,\\
    \partial_t p &= -K(1+\tau)\,\nabla\!\cdot\!\mathbf{v} - r + a,\\
    \partial_t r &= -\frac{1}{\tau_\sigma}\left(r + K\,\tau\,\nabla\!\cdot\!\mathbf{v}\right),
    \end{aligned}

where :math:`\tau = \tau_\varepsilon/\tau_\sigma - 1` is the dimensionless
relaxation defect and :math:`\tau_\varepsilon \ge \tau_\sigma > 0` are the
strain- and stress-relaxation times. The velocity update is identical to the
acoustic propagator. The source :math:`a` enters the pressure equation only;
no source is injected into the relaxation memory. This is a direct pressure
rate, not a constitutive volume-injection rate, which would enter both
constitutive equations through the dilatation. The memory :math:`r` has units
Pa/s; :math:`K` has units Pa and :math:`\tau` is dimensionless. As
:math:`\tau\to 0` (i.e. :math:`\tau_\varepsilon\to\tau_\sigma`) the memory
variable decouples and, for zero initial memory, remains zero. The pressure
and velocity equations then reduce to acoustic with pressure-rate forcing
:math:`a=K s`, where :math:`s` is acoustic ``volume_rate``. A nonzero supplied
initial memory instead leaves a decaying forcing term.

Specifying attenuation: the ``Q`` model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rather than the relaxation times, the user supplies a quality factor ``Q`` at
a reference angular frequency :math:`\omega_0 = 2\pi\,f_0`, where
:math:`f_0` is the ``q_freq`` argument, which is required. ``Q`` is a
differentiable model input of the same shape as ``v`` and ``rho`` (so
attenuation may be spatially varying and inverted for). Internally it is mapped
to the SLS relaxation times by the **constant-Q-at-**:math:`\omega_0`
single-mechanism relations

.. math::
    :label: q_mapping

    \tau_\sigma = \frac{\sqrt{1 + 1/Q^2} - 1/Q}{\omega_0},
    \qquad
    \tau_\varepsilon = \frac{1}{\omega_0^2\,\tau_\sigma},
    \qquad\Longrightarrow\qquad
    \tau_\varepsilon\,\tau_\sigma = \frac{1}{\omega_0^2}.

Larger ``Q`` means weaker attenuation; :math:`Q\to\infty` gives
:math:`\tau\to 0` and recovers the lossless acoustic equation with zero
initial memory and matching source units. The constitutive perturbation is
:math:`O(1/Q)`; the finite-record waveform error also depends on the source,
propagation distance, and objective.

Complex modulus, dispersion and the exact ``Q(omega)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the unforced constitutive response (:math:`a=0`), eliminating :math:`r` in the frequency domain
(:math:`\partial_t\to i\omega`) gives the acoustic relation
:math:`i\omega\,p = -M(\omega)\,\nabla\!\cdot\!\mathbf{v}` with the SLS complex
modulus

.. math::
    :label: complex_modulus

    M(\omega) = K\,\frac{1 + i\omega\tau_\varepsilon}{1 + i\omega\tau_\sigma}
    = K\,\frac{(1+\omega^2\tau_\varepsilon\tau_\sigma) + i\omega(\tau_\varepsilon-\tau_\sigma)}
             {1+\omega^2\tau_\sigma^2}.

Thus :math:`K` is the **relaxed** (:math:`\omega\to 0`) modulus, so the input
``v`` is the low-frequency phase velocity; the unrelaxed
(:math:`\omega\to\infty`) modulus is :math:`M_U = K(1+\tau)`. The quality
factor of the mechanism is the ratio of the real to imaginary parts,

.. math::
    :label: q_of_omega

    Q(\omega) = \frac{\operatorname{Re}M}{\operatorname{Im}M}
    = \frac{1 + \omega^2\tau_\varepsilon\tau_\sigma}{\omega(\tau_\varepsilon-\tau_\sigma)}
    = \frac{Q}{2}\left(\frac{\omega}{\omega_0} + \frac{\omega_0}{\omega}\right),

using :eq:`q_mapping`. This curve attains its minimum :math:`Q(\omega_0)=Q` at
the reference frequency (peak attenuation) and rises symmetrically in
:math:`\log\omega` away from it: a single SLS is only *approximately* constant-Q,
exact at :math:`\omega_0` and increasingly less attenuating (higher
:math:`Q(\omega)`) toward the band edges. Over a decade centred on
:math:`\omega_0` the deviation is a factor
:math:`\tfrac12(\sqrt{10}+1/\sqrt{10})\approx 1.74`; a genuinely
band-flat :math:`Q` is approximated with several mechanisms (see below).

The wavenumber is :math:`k(\omega)=\omega\sqrt{\rho/M(\omega)}`, giving phase
velocity and (amplitude) attenuation coefficient

.. math::
    :label: phase_and_alpha

    c(\omega) = \frac{1}{\operatorname{Re}\sqrt{\rho/M(\omega)}},
    \qquad
    \alpha(\omega) = \omega\,\bigl|\operatorname{Im}\sqrt{\rho/M(\omega)}\bigr|
    \;\approx\; \frac{\omega}{2\,c(\omega)\,Q(\omega)}\quad (Q\gg 1),

so a plane (or 1-D line) wave at :math:`\omega_0` decays with propagation
distance :math:`x` as :math:`\exp\!\left(-\alpha x\right) \approx
\exp\!\left(-\omega_0 x / [2\,c\,Q]\right)`. Both :eq:`phase_and_alpha` are
reproduced by the discrete scheme: measured against the exact SLS formulae in a
homogeneous medium, the phase velocity matches to :math:`<0.1\%` at
:math:`f=15,25,40\,\mathrm{Hz}` (:math:`Q=100`) and the attenuation matches to
:math:`<0.05\%` at :math:`Q=20` and :math:`Q=100` (test suite).

Discretisation
^^^^^^^^^^^^^^

Here :math:`h=\texttt{out.info.internal_dt}` is the internal time step;
:math:`n` indexes that internal clock. The velocity update is the acoustic one.
Writing
:math:`\theta = \nabla\!\cdot\!\mathbf{v}^{\,n+1/2}` for the (PML-augmented)
divergence at the half time level, the memory variable is advanced with an
Crank--Nicolson (trapezoidal) step centred at
:math:`n+1/2`, and the pressure uses the time-centred average
:math:`r^{\,n+1/2}=(r^n+r^{n+1})/2`:

.. math::
    :label: visco_discrete

    \begin{aligned}
    r^{\,n+1} &= a_r\,r^{\,n} - c_2\,\theta,
      \qquad a_r = \frac{1 - h/(2\tau_\sigma)}{1 + h/(2\tau_\sigma)},
      \quad c_2 = K\tau\,b_r,
      \quad b_r = \frac{h/\tau_\sigma}{1 + h/(2\tau_\sigma)},\\
    p^{\,n+1} &= p^{\,n} - h\,m_1\,\theta - \tfrac{h}{2}\bigl(r^{\,n}+r^{\,n+1}\bigr)+h a^{n+1/2},
      \qquad m_1 = K(1+\tau).
    \end{aligned}

For the isolated homogeneous relaxation ODE, Crank--Nicolson has
:math:`a_r=(1-h/(2\tau_\sigma))/(1+h/(2\tau_\sigma))`, hence
:math:`|a_r|<1` for :math:`h,\tau_\sigma>0`. This removes an explicit-ODE
restriction tied only to the relaxation time; it is not a stability proof for
the coupled wave/PML system. Its acoustic CFL estimate must use the
**unrelaxed** speed :math:`\sqrt{K(1+\tau)/\rho}`, not the relaxed input
``v``. The trapezoidal update is second-order in time. As :math:`Q\to\infty`,
:math:`c_2\to0` and :math:`m_1\to K`; :math:`\tau_\sigma\to1/\omega_0`
and :math:`a_r` generally does **not** approach one. Zero initial memory then
stays zero, giving the acoustic limit with matching source units.

Python maps :math:`(v,\rho,Q)` to the
three per-node fields :math:`m_1,c_2,a_r` (plus the buoyancies, exactly as in
the acoustic propagator) and hands them to the compiled kernel; gradients with
respect to :math:`v,\rho,Q` are obtained by chaining autograd through that map,
so every model gradient -- including :math:`\partial\mathcal{L}/\partial Q` --
is available.

Adjoint
^^^^^^^

The discrete adjoint is the exact transpose of :eq:`visco_discrete`. With
:math:`\bar p,\bar r,\bar{\mathbf v}` the adjoint (co-state) fields and the
effective field :math:`B_{\mathrm{eff}} = h\,(m_1-c_2/2)\,\bar p +
c_2\,\bar r` (the acoustic backward uses just :math:`h\,K\,\bar p`), the
velocity-adjoint and PML recursions are identical to the acoustic ones with
:math:`h\,K\,\bar p` replaced by :math:`B_{\mathrm{eff}}`. The genuinely
new pieces are the memory-variable adjoint -- **an SLS-like relaxation run
backward in time** -- and the model-gradient accumulation:

.. math::
    :label: visco_adjoint

    \begin{aligned}
    \bar r^{\,n} &= a_r\,\bar r^{\,n+1} - \tfrac{h}{2}(1+a_r)\,\bar p^{\,n+1},\\
    \partial\mathcal L/\partial m_1 &\mathrel{+}= -h\,\theta\,\bar p,\qquad
    \partial\mathcal L/\partial c_2   \mathrel{+}= \theta\bigl(\tfrac{h}{2}\bar p - \bar r\bigr),\qquad
    \partial\mathcal L/\partial a_r   \mathrel{+}= r^{\,n}\bigl(\bar r - \tfrac{h}{2}\bar p\bigr).
    \end{aligned}

The forward pass stores only :math:`\theta` (for the :math:`m_1` and
:math:`c_2` gradients) and :math:`r^{\,n}` (for :math:`a_r`). The adjoint is
verified two ways in the test suite: the compiled backward matches the
pure-PyTorch eager reference for the forward field and every gradient
(including :math:`\partial\mathcal L/\partial Q`) to :math:`\sim 10^{-6}` in
1-D/2-D/3-D, and an absolute central finite-difference gradient check passes
with respect to ``v``, ``rho``, ``Q`` and the source amplitudes. Exact model
gradient comparisons require an effective model-gradient sampling stride of
one; larger strides use the documented temporal quadrature approximation
(see :doc:`operator_contract`). The reciprocity test interchanges pressure
sources and receivers in a homogeneous medium; it does not establish
unweighted reciprocity for arbitrary heterogeneous direct pressure forcing.
The full derivation is in
the header of ``src/deepwave/viscoacoustic.c``.

Units and locations
^^^^^^^^^^^^^^^^^^^^

``v`` is the low-frequency (relaxed) wave speed and ``rho`` the density, with
the same units and staggered-node conventions as :func:`deepwave.acoustic`
(pressure and :math:`Q` at the integer nodes, velocities/buoyancy half a cell
away). ``Q`` is dimensionless. The reference frequency ``q_freq`` is in the
same frequency unit as ``dt`` implies, and **is required**: a default
single-mechanism call must pass it. ``Q`` means nothing without the frequency
it applies at, so there is nothing sensible to default it to -- in particular
not ``pml_freq``, which tunes the absorbing boundary, is itself inferred from
the wavelet when omitted, and would therefore have made the attenuation model
a function of the source you happened to use. (With ``q_n_mechanisms > 1``
``q_freq`` must be ``None`` instead: the constant-``Q`` fit spans ``q_band``,
and no single reference frequency applies.) It must be a positive, finite
number: a zero, negative, infinite or ``NaN`` reference frequency raises
:class:`deepwave.DeepwaveValueError`, and a value that is not a number at all
raises :class:`deepwave.DeepwaveTypeError`, rather than filling the wavefield
with ``NaN`` thousands of time steps later.
The memory variable is carried in the opaque ``auxiliary_state``, not as an
output wavefield of its own, because how many of them there are follows
``q_n_mechanisms`` rather than the physics you asked for. Passing the whole
output back as ``initial_state`` continues a simulation and carries it
across the join with everything else; there is no ``r_0=`` (or
``r_mech0=``) keyword argument.

Stability and the acoustic limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The relaxation update is unconditionally stable, but the coupled wave update
uses the *unrelaxed* stiffness for its CFL bound. Deepwave computes the
corresponding conservative speed from ``v``, ``q`` and the density-weighted
spatial stencil when ``max_vel`` is omitted. Sharp density contrasts can need
smaller steps than the pointwise unrelaxed speed suggests, especially at
higher spatial orders. An explicit ``max_vel`` must cover both bounds; inspect
``deepwave.plan(...).info.max_vel`` with automatic inference to obtain the
required value for a fixed model. ``Q`` must be positive;
very large finite ``Q`` approaches the acoustic constitutive law. For
matching sources use ``pressure_rate = rho * v**2 * volume_rate_acoustic``;
the same numerical amplitude has different units in these two channels.

Multi-mechanism (band-flat) Q
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single SLS is exactly constant-:math:`Q` only at :math:`\omega_0`
(:eq:`q_of_omega`). Band-flat :math:`Q` requires a *sum* of SLS mechanisms
whose relaxation times are spread across the band: pass
``q_n_mechanisms=L`` (with ``L > 1``) and ``q_band=(f1, f2)``. The
propagator chooses ``L`` relaxation frequencies log-spaced across the band
and fits **nonnegative** weights to the weak-loss target on log-spaced
frequency samples. This deterministic constrained least-squares fit is
performed once independently of the spatially varying, differentiable
target ``q``. Each mechanism
gets its own memory variable, each with its own
Crank--Nicolson update, summed into the pressure update; all of them are
carried in ``auxiliary_state``, so the output fields do not change with
``L``. Supported on the
eager Python backend and the compiled C/CUDA backend (what ``backend="auto"``,
the default, resolves to for this propagator; on the compiled backend it
cannot be combined with ``storage_compression=True`` or per-shot batched
models; the ``"torch_compile"`` Python mode does not
implement it). The default ``q_n_mechanisms=1`` single-mechanism case is unchanged
and remains the common choice in seismic viscoacoustic FWI.

The band fit uses the weak-attenuation (large-``Q``) approximation. Unlike the
single-SLS relation, it does not enforce the target ``Q`` exactly. With
strengths :math:`a_l=w_l/q\geq0`, relaxation times :math:`\tau_l>0`, and
:math:`x_l=\omega\tau_l`, the actual dimensionless modulus is

.. math::

   M(\omega)/K = 1 + \sum_l a_l\frac{i x_l}{1+i x_l}.

Its storage part is :math:`1+\sum_l a_l x_l^2/(1+x_l^2)\geq1`; its loss part
is :math:`\sum_l a_l x_l/(1+x_l^2)\geq0` at every nonnegative frequency.
The actual :math:`Q_{\mathrm{eff}}` is their ratio, equal to
:math:`(q+S)/L` for :math:`S=\sum_l w_l x_l^2/(1+x_l^2)` and
:math:`L=\sum_l w_l x_l/(1+x_l^2)`. At strong loss it may differ materially
from the target ``q`` even when the weak-loss fit residual is small.
Use :func:`deepwave.attenuation.response` to inspect the actual modulus,
phase and :math:`Q_{\mathrm{eff}}` over the frequencies of interest. Its
``frequencies`` argument uses cycles per time unit; it returns a complex
array of shape ``(*q.shape, frequencies.numel())`` and preserves gradients
through ``q``. DC returns one. The positive-loss convention is
:math:`\exp(i\omega t)`. More mechanisms or a narrower band do not guarantee
a better physical fit; measure the response for the intended signal.

At the geometric centre of a 10--1000 Hz band, the current nonnegative fit
gives these actual values in float64. The final column is the largest
relative error in :math:`Q_{\mathrm{eff}}` over 200 log-spaced in-band
frequencies. These measurements characterise the weak-loss fit, not the
accuracy of a propagated waveform.

.. list-table:: Measured multi-mechanism response
   :header-rows: 1

   * - Target ``q``
     - Mechanisms
     - :math:`Q_{\mathrm{eff}}` at 100 Hz
     - Maximum in-band relative error
   * - 0.01
     - 2
     - 5.07
     - 506
   * - 0.01
     - 3
     - 2.32
     - 392
   * - 20
     - 2
     - 42.8
     - 1.14
   * - 20
     - 3
     - 21.4
     - 0.214

The ``q=0.01`` rows show why the target cannot be presented as the achieved
quality factor. Reproduce a centre value with::

    import torch
    from deepwave.attenuation import response

    m = response(torch.tensor(0.01, dtype=torch.float64),
                 torch.tensor([100.0]), q_n_mechanisms=3,
                 q_band=(10.0, 1000.0))
    actual_q = m.real / m.imag

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.viscoacoustic(
        ...,
        sources={"pressure_rate": (amplitudes, source_locations)},
        receivers={"p": receiver_locations},
    )

This propagator's source channels are ``pressure_rate``, ``z``, ``y``, ``x`` and its
receiver channels are ``p``, ``z``, ``y``, ``x`` (the ``z`` channel is absent in 2-D).
``"pressure_rate"`` is a direct external pressure rate in Pa/s, adding
``internal_dt * amplitude`` to pressure without modulus scaling or a memory
source. There is no ``"volume_rate"`` or ``"p"`` source alias on this
unreleased propagator. ``"p"`` remains the pressure receiver channel.

The equivalent flat ``source_amplitudes_*``/``source_locations_*``/
``receiver_locations_p`` keywords are equally permanent and
produce bit-identical results; giving one channel in both forms
raises. See :doc:`usage` for the full vocabulary.

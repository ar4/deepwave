Viscoelastic (attenuating elastic) wave equation
=================================================

.. note::

   Stability: **experimental** -- the label records that this propagator has
   never appeared in a release, so ``0.1.0`` is the first time real use meets
   its design. It is not a lesser promise: its interface is frozen exactly as
   the other ten propagators' are, calling it emits no warning about its
   maturity, and every other Deepwave promise applies to it unchanged. See
   :doc:`stability`.

The :func:`deepwave.viscoelastic` propagator solves the isotropic elastic
velocity-stress wave equation (see :doc:`elastic <elastic>`) augmented with
independent **generalised-standard-linear-solid (GSLS)** relaxation
mechanisms on the mean (bulk) and deviatoric (shear) parts of the stress
tensor, giving the medium two independent, finite, frequency-dependent
quality factors: ``Qp`` for the mean/bulk modulus and ``Qs`` for the
deviatoric/shear modulus. Longitudinal attenuation generally depends on both.
The constitutive split is ``ndim``-dimensional, so a 2-D run is a 2-D medium,
not a plane-strain reduction of the 3-D attenuating law. It is built on the same staggered velocity-stress
grid as :doc:`elastic <elastic>`; only the stress constitutive update
changes. Reading the :doc:`elastic <elastic>` and :doc:`viscoacoustic
<viscoacoustic>` pages first is recommended -- this page describes only
what the Qp/Qs attenuation adds, reusing viscoacoustic's constant-Q GSLS
fit machinery (the constant-Q least-squares fit over ``q_band``) unchanged.

There is no compiled C/CUDA backend at all, so the default ``backend="auto"``
resolves to ``"eager"`` **silently** (check ``out.info.backend`` to
confirm this; ``backend="compiled"`` always raises ``NotImplementedError``
here, since there is no compiled kernel for it to assert). Both Python
modes are implemented -- ``"eager"`` and ``"torch_compile"``
(``torch.compile``) -- since the compiled mode only needs this module's own
``update_stresses`` and ``deepwave.elastic.update_velocities`` to be
traceable, independent of any compiled kernel. Only 2-D and 3-D
are supported -- shear waves require at least two spatial dimensions.

A free surface is fully supported, two ways: the improved vacuum method
(zero ``lamb``/``mu``/``buoyancy`` in an "air" region, for arbitrary/
topographic surfaces -- see :doc:`elastic <elastic>`) needs no special
handling here, since every GSLS relaxation coefficient is forced to exactly
zero wherever its driving modulus is zero; and the ``boundary=`` argument
(a flat-face image-method shortcut for a boundary lying exactly on a domain
edge) reuses :func:`deepwave.elastic`'s own guard-fill functions unchanged,
since velocities and the stress tensor share the same staggered-grid layout.
The GSLS memory variables need no guard-fill of their own -- no spatial
derivative of any memory variable appears in the update equations, so their
ghost regions are never read (see :doc:`boundaries`). There is no
``surface_method="pm"`` analogue yet.

Minimal example
^^^^^^^^^^^^^^^^

:func:`deepwave.viscoelastic` takes the same Lamé-parameter/buoyancy model
as :func:`deepwave.elastic`, plus bulk and shear modulus quality-factor models
``qp``/``qs``:

.. code-block:: python

    import torch
    import deepwave

    vp = 1500 * torch.ones(40, 40)
    vs = 1000 * torch.ones(40, 40)
    rho = 2200 * torch.ones(40, 40)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    qp = 30 * torch.ones(40, 40)
    qs = 20 * torch.ones(40, 40)

    out = deepwave.viscoelastic(
        lamb, mu, buoyancy, qp, qs, grid_spacing=5, dt=0.001,
        source_amplitudes_y=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations_y=torch.tensor([[[5, 20]]]),
        receiver_locations_y=torch.tensor([[[35, 20]]]),
        q_freq=25,  # the frequency at which qp/qs are the quality factors, in Hz
        pml_width=10,
        pml_freq=25,
    )
    out.receiver_amplitudes_y  # shape [1, 1, 200]

Governing equations: the mean/deviatoric split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elastic's isotropic stress-strain relation splits cleanly into a mean
(isotropic) part and a deviatoric (traceless) part. With :math:`\lambda,\mu`
the Lame parameters, :math:`n` the number of spatial dimensions (2 or 3),
:math:`\theta=\nabla\!\cdot\!\mathbf v` the (PML-augmented) velocity
divergence, and

.. math::
    :label: k_iso

    K_{\mathrm{iso}} = \lambda + \frac{2\mu}{n},

the normal and shear stress rates are

.. math::
    :label: elastic_split

    \dot\sigma_{ii} = K_{\mathrm{iso}}\,\theta
      + 2\mu\Bigl(\dot\varepsilon_{ii} - \tfrac{\theta}{n}\Bigr)
      \quad(\text{no sum}),
    \qquad
    \dot\sigma_{ij} = 2\mu_{ij}\,\dot\varepsilon_{ij}\quad(i\neq j),

which is algebraically identical to elastic's
:math:`\dot\sigma_{ii}=\lambda\theta+2\mu\dot\varepsilon_{ii}` (substitute
:math:`K_{\mathrm{iso}}-2\mu/n=\lambda`) and
:math:`\dot\sigma_{ij}=\mu_{ij}(\partial_j v_i+\partial_i v_j)=2\mu_{ij}\dot\varepsilon_{ij}`.
The mean part (the :math:`K_{\mathrm{iso}}\theta` term, identical for every
normal stress) is attenuated with a GSLS mechanism set built from
:math:`K_{\mathrm{iso}}` and ``Qp``; the deviatoric part -- the
:math:`2\mu(\dot\varepsilon_{ii}-\theta/n)` term for each normal stress
AND the :math:`2\mu_{ij}\dot\varepsilon_{ij}` shear term -- is attenuated
with an INDEPENDENT GSLS mechanism set built from the (respectively
unstaggered or staggered) shear modulus and ``Qs``. This decomposition,
and the elastic velocity-stress formulation itself, follow the standard
viscoelastic finite-difference literature (Robertsson, Blanch & Symes,
*Viscoelastic finite-difference modeling*, Geophysics 59(9), 1994; Bohlen,
*Parallel 3-D viscoelastic finite difference seismic modelling*, Computers
& Geosciences 28(8), 2002).

Because a fully general elastic stress update needs :math:`2n+1` independent
memory-variable driving terms (the shared mean term, one deviatoric term
per normal-stress direction, and one per unique shear pair), each with its
own GSLS mechanism set, this propagator uses :math:`(1+n+n(n-1)/2)\cdot L`
memory-variable fields for ``q_n_mechanisms=L`` mechanisms (7 groups in
3-D, 4 in 2-D; see ``deepwave.viscoelastic.viscoelastic_wavefield_names``).

Per-mechanism GSLS update
^^^^^^^^^^^^^^^^^^^^^^^^^^

Every memory-variable group uses the SAME Crank-Nicolson recursion as
viscoacoustic's single mechanism (see ``deepwave._gsls.sls_coefficients``
and :eq:`q_mapping`/:eq:`visco_discrete` on the :doc:`viscoacoustic
<viscoacoustic>` page) -- a relaxed modulus :math:`M` and target quality
factor :math:`Q` map to Crank-Nicolson coefficients :math:`(m_1,c_2,a_r)`,
and for a driving strain-rate term :math:`S` (:math:`\theta` for the mean
part; :math:`\dot\varepsilon_{ii}-\theta/n` for a deviatoric-normal part;
:math:`2\dot\varepsilon_{ij}` for a deviatoric-shear part) each mechanism's
memory variable :math:`r_l` evolves identically to viscoacoustic's own
memory variable:

.. math::
    :label: visco_elastic_r

    r_l^{\,n+1} = a_{r,l}\,r_l^{\,n} - c_{2,l}\,S.

The ONLY difference from viscoacoustic is the sign with which the field
being updated couples to :math:`m_1` and the memory variables. Pressure
(acoustic) obeys :math:`\dot p = -M\,\theta` (compression DECREASES
pressure with this codebase's sign convention), so
:math:`p^{\,n+1}=p^{\,n}-\Delta t\,m_1\,S-\tfrac{\Delta t}2(r^n+r^{n+1})`.
Stress obeys the OPPOSITE sign, :math:`\dot\sigma=+M\,S` (elastic's
:math:`\dot\sigma_{ii}=+\lambda\theta+\dots`), so the coupling is
:math:`+`, not :math:`-`:

.. math::
    :label: visco_elastic_field

    \text{field}^{\,n+1} = \text{field}^{\,n}
      + \Delta t\,m_1\,S + \tfrac{\Delta t}{2}\bigl(r^n+r^{n+1}\bigr).

This sign flip is derived directly from the complex-modulus definition
:math:`M(\omega)`, independent of any assumed sign relation between
pressure and stress (see the commit history / ``docs/viscoelastic.rst``'s
source for the full frequency-domain derivation): writing
:math:`\Sigma(\omega)=M(\omega)\,S(\omega)` (either sign) and eliminating
the memory variable's own frequency-domain representation gives, for BOTH
sign conventions, the identical recursion :eq:`visco_elastic_r` for
:math:`r`, with only the coupling-into-the-field sign differing. The memory
variable's own dynamics therefore never need to be re-derived per field
convention -- only the coupling sign does.

Reduction to elastic
^^^^^^^^^^^^^^^^^^^^^

As ``Qp, Qs -> infinity``, every :math:`\tau_l\to0`, so :math:`m_1\to M`
(the relaxed modulus) and, started from zero, every memory variable stays
at zero (:math:`c_{2,l}\to0` makes :eq:`visco_elastic_r` a pure decay).
Substituting :math:`m_1=K_{\mathrm{iso}}` (mean) and :math:`m_1=2\mu` or
:math:`\mu_{ij}` (deviatoric) into :eq:`visco_elastic_field` and summing the
mean and deviatoric contributions for a normal stress reproduces elastic's
:math:`\dot\sigma_{ii}=\lambda\theta+2\mu\dot\varepsilon_{ii}` EXACTLY
(the :math:`K_{\mathrm{iso}}` and :math:`2\mu/n` terms combine to
:math:`\lambda`, per :eq:`k_iso`); the shear update reduces to elastic's
:math:`\dot\sigma_{ij}=2\mu_{ij}\dot\varepsilon_{ij}` identically. This
reduction is verified in the test suite to machine precision (the residual
is the physical :math:`O(1/Q)` attenuation, exactly mirroring
viscoacoustic's own ``Q\to\infty`` gate): forward and every model gradient
match :func:`deepwave.elastic` to :math:`\sim10^{-9}`--:math:`10^{-12}`
relative at :math:`Q_p=Q_s=10^{10}`--:math:`10^{12}`.

The velocity update has NO Qp/Qs dependence at all (attenuation only
enters the stress constitutive law), so
``deepwave.viscoelastic.update_velocities`` reuses
``deepwave.elastic.update_velocities`` completely unchanged.

P-wave and S-wave attenuation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a pure shear disturbance (zero volumetric strain, e.g. an SH/SV plane
wave), only the deviatoric mechanism contributes, so the S-wave modulus is
exactly :math:`\mu(\omega)` -- the GSLS complex modulus
(:eq:`complex_modulus` on the viscoacoustic page) built from the relaxed
shear modulus and ``Qs`` alone, independent of ``Qp``. A general
directional P-wave, however, has BOTH nonzero volumetric AND deviatoric
strain (only a spherically symmetric dilatation has none), so the P-wave
modulus mixes both mechanisms:

.. math::
    :label: p_wave_modulus

    M_P(\omega) = K_{\mathrm{iso}}(\omega) + \frac{2(n-1)}{n}\,\mu(\omega),

the complex generalisation of the P-wave modulus
:math:`M=\lambda+2\mu=K_{\mathrm{iso}}+\tfrac{2(n-1)}n\mu`. This matches
standard viscoelastic theory, where P-wave attenuation is a genuine
combination of the bulk and shear quality factors, not a function of
``Qp`` alone. The test suite isolates each: the S-wave test measures decay
and dispersion of a transverse-force-generated wave (theory
:eq:`p_wave_modulus`-analogous with :math:`\mu(\omega)` alone) to
:math:`\sim1\%`; the P-wave test sets ``Qs`` very large (so :math:`\mu`
is real/unattenuated) and measures a longitudinal-force-generated wave
against :eq:`p_wave_modulus` to :math:`\sim1\text{-}2\%`.

Adjoint
^^^^^^^

Unlike viscoacoustic (which has a hand-derived compiled adjoint), the
viscoelastic eager backend has NO custom ``torch.autograd.Function``:
``deepwave.viscoelastic.viscoelastic_python`` is a plain composition of
differentiable PyTorch tensor operations (mirroring
:func:`deepwave.elastic`'s and :func:`deepwave.viscoacoustic`'s own eager
paths), so native autograd traces the whole time loop and the backward
pass is exact by construction -- the adjoint-dot-product identity
:math:`\langle F\mathbf s,\mathbf r\rangle=\langle\mathbf s,F^{*}\mathbf
r\rangle` holds to machine precision (:math:`\sim10^{-10}` relative, limited
only by float64 rounding, in the test suite), and there is no separate
hand-adjoint derivation to maintain since this propagator has no compiled
backend at all (``torch_compile`` traces the same autograd-differentiable
composition, not a separate kernel).

Units, locations and initial state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``lamb``, ``mu`` and ``buoyancy`` have the same units and staggered-node
conventions as :func:`deepwave.elastic`. ``qp``/``qs`` are dimensionless
quality factors, defined on the SAME (unstaggered) grid as ``lamb``/``mu``;
``qs`` is additionally harmonic-mean-staggered onto the shear-stress nodes
internally (the same 4-point average and vacuum mask
``deepwave.elastic.prepare_parameters`` uses for ``mu`` itself -- see
``deepwave.viscoelastic._stagger_shear_harmonic``), so a spatially
varying ``qs`` is handled consistently with the medium's own shear-modulus
staggering. ``qp``/``qs`` must be positive; a literal zero is tolerated in
vacuum cells (internally clamped to a safe placeholder, since the
zero-valued modulus there already forces the corresponding GSLS
coefficients to zero) but is not meaningful elsewhere. The reference
frequency ``q_freq`` (required for the default single mechanism, where it must
be a positive, finite number, and rejected above it) and
``q_n_mechanisms``/``q_band``
have the identical semantics to :func:`deepwave.viscoacoustic`, shared by
both ``qp`` and ``qs`` (the same log-spaced relaxation frequencies and
fitted weights are reused for both, since the GSLS fit
(the constant-Q least-squares fit) depends only on ``q_band`` and
``q_n_mechanisms``, not on the target ``Q`` value itself).

None of the memory variables is an output field or an explicit keyword
argument: there is no ``rp_0=``/``rs_z_0=``/... form. They -- and the
absorbing boundary's memory variables -- are carried opaquely in
``auxiliary_state`` and are reachable only via ``initial_state``, exactly
like :func:`deepwave.viscoacoustic`'s. How many of them there are follows
``q_n_mechanisms``, so folding them away is what keeps the output fields
the same whatever that is set to.

Stability
^^^^^^^^^

The GSLS relaxation update is unconditionally stable, but the coupled wave
update uses the unrelaxed bulk and shear moduli for its CFL bound. Deepwave
computes the corresponding maximum P- and S-wave speeds when ``max_vel`` is
omitted; an explicit bound must cover those speeds. The elastic M-PML
(``mpml_fraction > 0``) stabiliser is inherited unchanged: the GSLS memory
variables sit alongside, and are computed independently of, the M-PML
memory variables inside ``deepwave.viscoelastic.update_stresses``, so
the corner-instability fix and cure verified for
:func:`deepwave.elastic` (Meza-Fajardo & Papageorgiou, 2008) apply
identically here (verified in the test suite with finite, moderate Qp/Qs,
not just the ``Q\to\infty`` limit).

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.viscoelastic(
        ...,
        sources={"pressure_rate": (amplitudes, source_locations)},
        receivers={"p": receiver_locations},
    )

This propagator's source channels are ``pressure_rate``, ``z``, ``y``, ``x`` and its
receiver channels are ``p``, ``z``, ``y``, ``x`` (the ``z`` channel is absent in 2-D).
``"pressure_rate"`` names the quantity the scalar source injects; see
:doc:`units`. ``"p"`` is still accepted there, with a
:class:`deepwave.DeepwaveFutureWarning`, and is removed in ``0.4.0``.
It is unchanged as a *receiver* channel name.

The equivalent flat ``source_amplitudes_*``/``source_locations_*``/
``receiver_locations_p`` keywords are equally permanent and
produce bit-identical results; giving one channel in both forms
raises. See :doc:`usage` for the full vocabulary.

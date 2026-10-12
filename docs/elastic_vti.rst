Elastic VTI (anisotropic) wave equation
=======================================

.. note::

   Stability: **experimental** -- the label records that this propagator has
   never appeared in a release, so ``0.1.0`` is the first time real use meets
   its design. It is not a lesser promise: its interface is frozen exactly as
   the other eight propagators' are, calling it emits no warning about its
   maturity, and every other Deepwave promise applies to it unchanged. See
   :doc:`stability`.

The :func:`deepwave.elastic_vti` propagator solves the elastic wave equation
in a *vertically transverse isotropic* (VTI) medium: an anisotropic medium
with a single axis of rotational symmetry, taken to be vertical (the slowest
spatial dimension -- :math:`z` in 3D, :math:`y` in 2D). VTI is the most common
form of seismic anisotropy, arising from fine horizontal layering and from
aligned horizontal fractures or shale fabric. It is Deepwave's first
anisotropic propagator.

It shares the entire numerical machinery of the isotropic
:doc:`elastic <elastic>` propagator -- the velocity--stress formulation on the
same staggered grid, the C-PML (and optional multiaxial M-PML) absorbing
boundaries, the improved-vacuum (Zeng et al., 2012) free surface, the
staggering of sources and receivers, and the storage/adjoint skeleton. **Only
the constitutive (stress--strain) relation differs.** Reading the
:doc:`elastic <elastic>` page first is recommended; this page describes only
what anisotropy changes.

See :doc:`materials` for the material domain, dimensions, and limits of the
interior energy argument, and :doc:`units` for source and receiver normalisation.

Minimal example
^^^^^^^^^^^^^^^^

:func:`deepwave.elastic_vti` takes the VTI stiffnesses and a buoyancy model,
a grid spacing, a time step, and a source. This example builds the
stiffnesses from Thomsen parameters with :func:`deepwave.anisotropy.thomsen_to_cij`
(see the *Thomsen parameters* section below):

.. code-block:: python

    import torch
    import deepwave

    vp0 = 1500 * torch.ones(40, 40)
    vs0 = 800 * torch.ones(40, 40)
    rho = 2200 * torch.ones(40, 40)
    epsilon = 0.2 * torch.ones(40, 40)
    delta = 0.1 * torch.ones(40, 40)
    gamma = 0.15 * torch.ones(40, 40)
    c11, c13, c33, c55, c66, buoyancy = deepwave.anisotropy.thomsen_to_cij(vp0, vs0, rho, epsilon, delta, gamma)

    out = deepwave.elastic_vti(
        c11=c11, c13=c13, c33=c33, c55=c55, buoyancy=buoyancy,
        grid_spacing=5, dt=0.001,
        source_amplitudes_y=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations_y=torch.tensor([[[5, 20]]]),
        receiver_locations_y=torch.tensor([[[35, 20]]]),
        pml_width=10,
        pml_freq=25,
    )
    out.receiver_amplitudes_y  # shape [1, 1, 200]

Alongside the receiver data, ``out`` also carries the final wavefields plus
one opaque ``auxiliary_state`` holding the absorbing boundary's memory
variables. Passing the whole output back in as ``initial_state`` continues
the propagation in a later call (see :doc:`usage`).

Constitutive relation
^^^^^^^^^^^^^^^^^^^^^^

For a general linear-elastic medium the stress is related to the strain by the
fourth-order stiffness tensor :math:`c_{ijkl}`,

.. math::
    :label: hooke_general

    \sigma_{ij} = c_{ijkl}\epsilon_{kl},
    \qquad
    \epsilon_{kl} = \tfrac{1}{2}\left(\partial_k u_l + \partial_l u_k\right),

which in Voigt notation (:math:`11\!\to\!1`, :math:`22\!\to\!2`,
:math:`33\!\to\!3`, :math:`23\!\to\!4`, :math:`13\!\to\!5`, :math:`12\!\to\!6`)
becomes a symmetric :math:`6\times 6` matrix :math:`C_{IJ}`. For a VTI medium
with the symmetry axis along direction :math:`3` this matrix has five
independent stiffnesses :math:`c_{11}, c_{13}, c_{33}, c_{55}, c_{66}`:

.. math::
    :label: cij_vti

    C =
    \begin{pmatrix}
    c_{11} & c_{12} & c_{13} & 0 & 0 & 0\\
    c_{12} & c_{11} & c_{13} & 0 & 0 & 0\\
    c_{13} & c_{13} & c_{33} & 0 & 0 & 0\\
    0 & 0 & 0 & c_{55} & 0 & 0\\
    0 & 0 & 0 & 0 & c_{55} & 0\\
    0 & 0 & 0 & 0 & 0 & c_{66}
    \end{pmatrix},
    \qquad c_{12} = c_{11} - 2c_{66}.

The isotropic elastic medium is the special case
:math:`c_{11}=c_{33}=\lambda + 2\mu`, :math:`c_{13}=c_{12}=\lambda`,
:math:`c_{55}=c_{66}=\mu`, so :func:`deepwave.elastic_vti` reproduces
:func:`deepwave.elastic` exactly in that limit (verified to machine precision in
the test suite, for the forward field and for every gradient, in 2D and 3D).

**3D.** With the symmetry axis as :math:`z` (Voigt ``3``), :math:`x`\ =``1``,
:math:`y`\ =``2``, the stress-rate updates are

.. math::
    :label: vti_stress_3d

    \begin{aligned}
    \dot\sigma_{xx} &= c_{11}\partial_x v_x + c_{12}\partial_y v_y + c_{13}\partial_z v_z,\\
    \dot\sigma_{yy} &= c_{12}\partial_x v_x + c_{11}\partial_y v_y + c_{13}\partial_z v_z,\\
    \dot\sigma_{zz} &= c_{13}\partial_x v_x + c_{13}\partial_y v_y + c_{33}\partial_z v_z,\\
    \dot\sigma_{yz} &= c_{55}\left(\partial_y v_z + \partial_z v_y\right),\\
    \dot\sigma_{xz} &= c_{55}\left(\partial_x v_z + \partial_z v_x\right),\\
    \dot\sigma_{xy} &= c_{66}\left(\partial_x v_y + \partial_y v_x\right).
    \end{aligned}

**2D (P--SV in the** :math:`(y,x)` **plane, axis =** :math:`y`\ **).** Here
:math:`y` is Voigt ``3`` and :math:`x` is ``1``; :math:`c_{66}` and
:math:`\sigma_{xy}`-plane terms drop out and :math:`c_{66}` must be ``None``:

.. math::
    :label: vti_stress_2d

    \begin{aligned}
    \dot\sigma_{xx} &= c_{11}\partial_x v_x + c_{13}\partial_y v_y,\\
    \dot\sigma_{yy} &= c_{13}\partial_x v_x + c_{33}\partial_y v_y,\\
    \dot\sigma_{xy} &= c_{55}\left(\partial_x v_y + \partial_y v_x\right).
    \end{aligned}

There is no 1D VTI propagator: with a single spatial dimension only
:math:`c_{33}` survives, which is the isotropic 1D elastic equation, so 1D
inputs raise an error (use :func:`deepwave.elastic`).

The velocity update is unchanged from the isotropic case,
:math:`\dot v_i = b\,\partial_j\sigma_{ij}` with :math:`b=1/\rho` the buoyancy;
anisotropy enters only through the stiffnesses above.

Units and locations
^^^^^^^^^^^^^^^^^^^^

All stiffnesses :math:`c_{IJ}` have units of pressure (Pa in SI), the same as
the Lamé parameters they generalise, and are supplied at the normal-stress
(integer) grid nodes. The buoyancy is :math:`1/\rho` (units
:math:`\mathrm{m^3\,kg^{-1}}`). As in the isotropic propagator, the shear
stiffnesses are internally averaged onto the staggered shear-stress nodes and
the buoyancy onto the staggered velocity nodes.

Free-surface / improved-vacuum averaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The improved-vacuum free surface treats a cell with ``buoyancy = 0`` (and zero
stiffnesses) as a vacuum, with the model parameters averaged onto the staggered
nodes exactly as in :func:`deepwave.elastic`. Each VTI stiffness is averaged in
the way appropriate to the elastic modulus it generalises:

* :math:`c_{55}` generalises the shear modulus :math:`\mu` for the stresses
  whose planes contain the symmetry axis (:math:`\sigma_{xz}`, :math:`\sigma_{yz}`
  in 3D, and the single shear stress in 2D). It is harmonically averaged (with
  vacuum masking) onto those shear-stress nodes exactly as :math:`\mu` is.
* :math:`c_{66}` generalises :math:`\mu` for the horizontal-plane shear stress
  :math:`\sigma_{xy}` (3D only) and is harmonically averaged onto that node in
  the same way.
* The normal-node stiffnesses :math:`c_{11}, c_{12}, c_{13}, c_{33}` are used
  unaveraged at the nodes where the user supplies them, exactly as
  :math:`\lambda` and :math:`\lambda+2\mu` are in the isotropic propagator.

Thomsen parameters
^^^^^^^^^^^^^^^^^^^

Anisotropy is often described by Thomsen (1986) parameters
:math:`(v_{P0}, v_{S0}, \rho, \epsilon, \delta, \gamma)`, where
:math:`v_{P0}` and :math:`v_{S0}` are the P- and S-wave speeds *along the
symmetry axis*, :math:`\epsilon` and :math:`\gamma` measure the fractional
difference between horizontal and vertical P and SH velocities, and
:math:`\delta` controls the near-vertical curvature of the qP wavefront (and
hence the short-spread NMO velocity). The helper
:func:`deepwave.anisotropy.thomsen_to_cij` converts these to stiffnesses using the
**exact** (not the weak-anisotropy linearised) relations

.. math::
    :label: thomsen

    \begin{aligned}
    c_{33} &= \rho v_{P0}^2, & c_{55} &= \rho v_{S0}^2,\\
    c_{11} &= c_{33}(1 + 2\epsilon), & c_{66} &= c_{55}(1 + 2\gamma),\\
    c_{13} &= \sqrt{2\delta\, c_{33}(c_{33}-c_{55}) + (c_{33}-c_{55})^2} - c_{55}.
    \end{aligned}

The :math:`c_{13}` expression inverts Thomsen's exact definition
:math:`\delta = \left[(c_{13}+c_{55})^2 - (c_{33}-c_{55})^2\right] /
\left[2 c_{33}(c_{33}-c_{55})\right]`, taking the positive root, which
corresponds to :math:`c_{13}+c_{55} > 0`; this holds for essentially all real
materials. The square-root argument is non-negative iff
:math:`\delta \ge -(c_{33}-c_{55})/(2 c_{33})`; more negative :math:`\delta`
cannot be represented and is rejected with a clear error. The conversion is
differentiable, so inversion may be performed directly in Thomsen-parameter
space by chaining autograd through it (verified by finite-difference
gradient checks in the test suite).

Alongside the five stiffnesses, the helper returns the buoyancy
:math:`b = 1/\rho`, because that -- not :math:`\rho` -- is what the
propagators take, so its whole return value can be unpacked straight into
:func:`deepwave.elastic_vti`. A vacuum cell (:math:`\rho = 0`) is mapped to
:math:`b = 0` exactly, which is the vacuum sentinel the propagators expect,
rather than to the infinity a plain ``1 / rho`` would produce.
:func:`deepwave.anisotropy.cij_to_thomsen` inverts the whole conversion,
taking the buoyancy in the same place.

P-wave surface data alone cannot resolve :math:`\epsilon` and :math:`\delta`
separately -- time processing (NMO, DMO, time migration) is controlled only
by the P-wave NMO velocity :math:`v_{nmo}` and the anellipticity parameter
:math:`\eta` (Alkhalifah and Tsvankin, 1995). The helper
:func:`deepwave.anisotropy.alkhalifah_to_thomsen` converts :math:`(v_{P0}, v_{nmo},
\eta)` to :math:`(\epsilon, \delta)` via the exact relations
:math:`\delta = (v_{nmo}^2/v_{P0}^2 - 1)/2` and :math:`\epsilon = \eta(1 +
2\delta) + \delta`, so a model estimated from NMO-velocity analysis can be
chained straight into :func:`deepwave.anisotropy.thomsen_to_cij` (choosing an
:math:`v_{S0}`, since shear information is not recoverable from P-wave-only
data) to build the Cij stiffness model that :func:`deepwave.elastic_vti` (or
:func:`deepwave.acoustic_vti`, with :math:`v_{S0}=0`) consumes -- also a
differentiable chain, so FWI may be performed directly in
:math:`(v_{P0}, v_{nmo}, \eta)` parameter space.
:func:`deepwave.anisotropy.thomsen_to_alkhalifah` is its inverse, converting
:math:`(v_{P0}, \epsilon, \delta)` back to :math:`(v_{nmo}, \eta)` via
:math:`v_{nmo} = v_{P0}\sqrt{1 + 2\delta}` and
:math:`\eta = (\epsilon - \delta)/(1 + 2\delta)`, for comparing a Thomsen
model with, or initialising it from, a conventional NMO-velocity model.

Phase velocities
^^^^^^^^^^^^^^^^^

Independent tests also propagate 3D SH, qP and qSV plane-wave initial states
with unequal ``c55`` and ``c66`` and an oblique horizontal azimuth. Refining
space and time converges to the continuum Christoffel solution. Separate
normal-incidence P and SV tests cross a welded interface between two VTI
materials and converge to the analytic reflection/transmission waveforms.
These checks cover those modes and geometries; they do not establish an
error bound for arbitrary oblique interfaces or staircase topography.

For a plane wave with unit propagation direction :math:`\mathbf{n}` the phase
velocities follow from the Christoffel matrix
:math:`\Gamma_{ik} = c_{ijkl} n_j n_l`; the three eigenvalues
:math:`\rho v^2` give the qP, qSV and (in 3D) SH phase speeds, which depend on
the angle to the symmetry axis. The propagator reproduces the analytic
Christoffel qP and qSV phase velocities across angle (measured from
band-limited plane waves in a homogeneous strongly-anisotropic medium) to
within the grid-dispersion budget; this is the test that detects
axis-convention or constitutive-coupling errors, which the isotropic limit
(:math:`c_{11}=c_{33}`) cannot.

Stability
^^^^^^^^^

Readable backgrounds are checked for finite values and positive represented
elastic energy. Active cells require positive normal stiffnesses and
nonnegative shear stiffnesses. Positive shear stiffnesses require a positive
normal block; supported zero-shear fluid limits permit a semidefinite block.
Vacuum requires zero buoyancy and zero stiffness. See :doc:`materials` for
the exact 2D/3D conditions and tensor-transform preconditions.

When ``max_vel`` is omitted, setup derives a conservative speed from the
constitutive tensor and staggered coefficient averaging, including
heterogeneous contrast. A pointwise phase-speed maximum alone need not bound
the discrete operator. An explicit speed must cover the resolved bound; it
cannot bypass the material or CFL checks. See :doc:`discretisation`.

Strong anisotropy is a known trigger of C-PML corner instability. The
multiaxial-PML option (``mpml_fraction``, typically 0.15--0.2) adds a small amount of
cross-axis damping that cures it at the cost of slightly larger boundary
reflections; see :doc:`boundaries`.

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.elastic_vti(
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

Elastic TTI (tilted anisotropic) wave equation
================================================

.. note::

   Stability: **experimental** -- the label records that this propagator has
   never appeared in a release, so ``0.1.0`` is the first time real use meets
   its design. It is not a lesser promise: its interface is frozen exactly as
   the other ten propagators' are, calling it emits no warning about its
   maturity, and every other Deepwave promise applies to it unchanged. See
   :doc:`stability`.

The :func:`deepwave.elastic_tti` propagator solves the elastic wave equation
in a *tilted transversely isotropic* (TTI) medium: a :doc:`VTI
<elastic_vti>` medium whose symmetry axis has been rotated away from
vertical, in the plane of propagation, by a per-cell dip angle ``tilt``
(radians). TTI is common where VTI layering is itself dipping, for example
on the flank of a salt body or below a tilted unconformity. It shares the
entire numerical machinery of the :doc:`elastic VTI <elastic_vti>`
propagator -- the staggered velocity-stress grid, the C-PML (and optional
multiaxial M-PML) absorbing boundaries, the improved-vacuum free surface,
and the source/receiver and storage/adjoint conventions -- so reading
:doc:`elastic_vti` (and :doc:`elastic <elastic>` beneath it) first is
recommended; this page describes only what the tilt adds. The full discrete
derivation, including the Bond-rotation algebra, the phase-velocity oracle
used to validate it, and the ``mpml_fraction`` stability study, is in
:doc:`elastic_tti_derivation`.

Minimal example
^^^^^^^^^^^^^^^^

:func:`deepwave.elastic_tti` takes the same VTI stiffnesses as
:func:`deepwave.elastic_vti` plus a per-cell ``tilt`` angle in radians:

.. code-block:: python

    import torch
    import deepwave

    vp0 = 1500 * torch.ones(40, 40)
    vs0 = 800 * torch.ones(40, 40)
    rho = 2200 * torch.ones(40, 40)
    epsilon = 0.2 * torch.ones(40, 40)
    delta = 0.1 * torch.ones(40, 40)
    c11, c13, c33, c55, _, buoyancy = deepwave.anisotropy.thomsen_to_cij(vp0, vs0, rho, epsilon, delta)
    tilt = 0.3 * torch.ones(40, 40)  # radians

    out = deepwave.elastic_tti(
        c11=c11, c13=c13, c33=c33, c55=c55, buoyancy=buoyancy,
        grid_spacing=5, dt=0.001,
        tilt=tilt,
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

:func:`deepwave.elastic_tti` takes the same VTI stiffnesses as
:func:`deepwave.elastic_vti` -- :math:`c_{11}, c_{13}, c_{33}, c_{55}` of the
*untilted* medium (there is no ``c66`` parameter: as in the 2-D VTI case it
does not enter the in-plane P-SV equations) -- plus a ``tilt`` model giving the
symmetry axis's dip angle away from :math:`y` (the slow dimension) toward
:math:`x` (the fast dimension), at every cell. Rotating a VTI stiffness
tensor by a general in-plane angle produces a *monoclinic* medium: the
in-plane stress-rate relation, in the reduced Voigt basis
:math:`(\sigma_{xx}, \sigma_{yy}, \sigma_{xy})`, has **six** independent
rotated stiffnesses :math:`C_{11}', C_{13}', C_{33}', C_{55}', C_{15}',
C_{35}'` instead of VTI's four, obtained from :math:`c_{11}, c_{13}, c_{33},
c_{55}` and ``tilt`` by the Bond transformation (Bond, 1943, *Bell Syst.
Tech. J.* **22**, 1). The two new constants :math:`C_{15}'` and
:math:`C_{35}'` couple the normal strains to the shear strain and vanish
exactly at ``tilt = 0``, at which point the other four also reduce exactly
to :math:`c_{11}, c_{13}, c_{33}, c_{55}` -- so :func:`deepwave.elastic_tti`
reproduces :func:`deepwave.elastic_vti` to machine precision whenever
``tilt`` is zero (or ``None``, its default) everywhere. The rotation is
applied once, differentiably, when the model is prepared, so gradients with
respect to the stiffnesses, the buoyancy, and ``tilt`` itself are all
available.

Implementation
^^^^^^^^^^^^^^

The velocity update is unchanged from :func:`deepwave.elastic_vti` --
buoyancy is isotropic, so tilt does not affect it. Only the stress update
changes: the normal strains :math:`\partial_y v_y`, :math:`\partial_x v_x`
live at the integer normal-stress node, exactly as in the isotropic and VTI
propagators, while the shear strain :math:`\partial_x v_y + \partial_y v_x`
lives at the staggered shear-stress node. The :math:`C_{15}'`/:math:`C_{35}'`
cross-terms therefore combine quantities defined at two different node
locations and require an interpolation between them -- the same four-point
average already used to move the shear stiffness :math:`c_{55}` onto the
shear-stress node, applied here (and its exact transpose, used in the
opposite direction) so that the discrete stiffness operator stays symmetric
and the scheme remains energy-conserving and exactly reciprocal. At strong
heterogeneous contrasts, the averaged mixed coefficients are
scaled only where their local Schur complement would be negative; this keeps
the discrete elastic energy positive and preserves the zero-tilt reduction.
One
consequence worth knowing: the interpolation makes the cross-term
contribution second-order accurate in space regardless of the ``accuracy``
setting, which still controls the order of the underlying first-derivative
stencils elsewhere. :doc:`elastic_tti_derivation` shows this is accurate to
well within a couple of percent against an analytic Christoffel
phase-velocity solution, across a sweep of tilt angles and propagation
directions.

Because the maximum phase velocity of a homogeneous anisotropic medium does
not depend on how the medium is rotated, the same trace-of-Christoffel bound
:func:`deepwave.elastic_vti` uses (computed from the untilted stiffnesses)
is a valid upper bound for any ``tilt``, and is reused directly for the CFL
condition and the PML profiles when ``max_vel`` is not supplied.

Scope
^^^^^

Only 2-D is currently implemented; 3-D TTI (which additionally needs an
``azimuth`` angle and the ``c66`` stiffness) is a recorded follow-up. There
is no compiled C/CUDA backend at all, so the default ``backend="auto"``
resolves to the eager Python backend **silently** (no warning; check
``out.info.backend`` to confirm what ran). ``backend="compiled"`` always
raises ``NotImplementedError`` here, since there is no compiled kernel for
it to assert. ``"torch_compile"`` (``torch.compile``)
is available, since it only needs the eager reference kernels to be
traceable.

Stability
^^^^^^^^^

Tilted anisotropy is a known trigger of classical C-PML corner instability:
the mismatch between the phase and group velocity directions in the
boundary layer can excite a growing mode. As with :doc:`elastic_vti`, the
multiaxial-PML option (``mpml_fraction``, typically 0.15--0.2) adds a small amount
of cross-axis damping that cures this at the cost of slightly larger
boundary reflections; ``elastic_tti`` defaults ``mpml_fraction`` to 0.0, matching
every other propagator. :doc:`elastic_tti_derivation` documents a validated
operating envelope at ``mpml_fraction = 0.15`` swept across tilt angle and
anisotropy strength, including 90-degree tilt and strong or negative
Thomsen :math:`\delta`. See :doc:`boundaries` for the general PML
discussion.

A domain that is closed on **every** face (all faces reflecting, for example
``pml_width=0`` everywhere) is not long-time stable when the tilt is non-zero:
the rotated stencils lose their symmetry where they are truncated at a
reflecting face, and the trapped wavefield grows slowly. The growth is a
boundary effect, so it is faster in smaller boxes: measured with a 30 degree
tilt, a 12 x 12 cell box grew by about 0.1% per time step, while a 60 x 200
cell box grew by a factor of about three over 40,000 steps. The same medium
stayed bounded over 40,000 steps with absorbing faces on one dimension and
reflecting faces on the other, and with a reflecting top face over absorbing
sides and base (including over a low-velocity surface layer); an untilted
medium is stable in a closed box. Keep some absorbing faces in long tilted
runs.

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.elastic_tti(
        ...,
        sources={"pressure_rate": (amplitudes, source_locations)},
        receivers={"p": receiver_locations},
    )

This propagator's source channels are ``pressure_rate``, ``y``, ``x`` and its
receiver channels are ``p``, ``y``, ``x`` (this propagator is 2-D only).
``"pressure_rate"`` names the quantity the scalar source injects; see
:doc:`units`. ``"p"`` is still accepted there, with a
:class:`deepwave.DeepwaveFutureWarning`, and is removed in ``0.4.0``.
It is unchanged as a *receiver* channel name.

The equivalent flat ``source_amplitudes_*``/``source_locations_*``/
``receiver_locations_p`` keywords are equally permanent and
produce bit-identical results; giving one channel in both forms
raises. See :doc:`usage` for the full vocabulary.

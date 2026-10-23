Pseudo-acoustic VTI wave equation
=================================

.. note::

   Stability: **experimental** -- the label records that this propagator has
   never appeared in a release, so ``0.1.0`` is the first time real use meets
   its design. It is not a lesser promise: its interface is frozen exactly as
   the other eight propagators' are, calling it emits no warning about its
   maturity, and every other Deepwave promise applies to it unchanged. See
   :doc:`stability`.

The :func:`deepwave.acoustic_vti` propagator solves the *pseudo-acoustic* VTI
wave equation of Duveneck et al. (2008): the ``V_S = 0`` limit (Alkhalifah,
1998) of the elastic VTI equation, in a *vertically transverse isotropic*
medium with variable density. Setting the shear-wave speed to zero collapses
the elastic VTI system onto a first-order velocity--stress system with only
**two** stress-like scalars, giving the industry-standard kinematic ("acoustic")
engine for anisotropic reverse-time migration and full-waveform inversion at a
fraction of the cost of full elastic VTI.

It shares the numerical machinery of the isotropic :doc:`elastic <elastic>` and
:doc:`elastic VTI <elastic_vti>` propagators -- the velocity--stress formulation
on the same staggered grid, the C-PML (and optional multiaxial M-PML) absorbing
boundaries, and the staggering of sources and receivers -- and is the exact
``c55 = c66 = 0`` reduction of :func:`deepwave.elastic_vti`. Reading the
:doc:`elastic VTI <elastic_vti>` page first is recommended; this page describes
only what the pseudo-acoustic reduction changes.

.. note::

   The compiled C/CUDA backend, which ``backend="auto"`` (the default for
   this propagator, matching every other compiled propagator in Deepwave)
   resolves to, implements both forward propagation and its hand-derived
   adjoint: forward matches the eager backend to near machine precision, and
   gradients (every stiffness and the buoyancy) match the eager backend to
   near machine precision when the effective model-gradient stride is one
   and storage is lossless. Larger strides use the same sampled-gradient
   approximation as :func:`deepwave.elastic_vti`; the stride is
   ``out.info.step_ratio * model_gradient_sampling_interval``, so interval 1
   alone does not imply exact accumulation when CFL substepping is needed.
   This is the approximation shared with the elastic-VTI compiled backend
   under the same setting, not specific to this propagator. Both backends
   have been checked against the full physics/gradient test battery (the
   elastic_vti oracle, the isotropic-limit and analytic Christoffel oracles,
   reciprocity, PML absorption, ``torch.autograd.gradcheck``, and the
   discrete adjoint dot-product identity), not merely against each other.
   The eager Python backend (``backend="eager"``) instead provides gradients
   via native autograd through the time loop. ``backend="torch_compile"``
   runs that same eager time loop with the two step functions compiled.

See :doc:`materials` for the material domain, dimensions, and limits of the
interior energy argument, and :doc:`units` for source and receiver normalisation.

Minimal example
^^^^^^^^^^^^^^^^

:func:`deepwave.acoustic_vti` takes the three VTI stiffnesses, a buoyancy
model, a grid spacing, a time step, and a source. This example builds the
stiffnesses from Thomsen parameters with :func:`deepwave.anisotropy.thomsen_to_cij`
(``vs0 = 0``, as the *Parameterisation and units* section below explains):

.. code-block:: python

    import torch
    import deepwave

    vp0 = 1500 * torch.ones(40, 40)
    vs0 = torch.zeros(40, 40)
    rho = 2200 * torch.ones(40, 40)
    epsilon = 0.2 * torch.ones(40, 40)
    delta = 0.1 * torch.ones(40, 40)
    c11, c13, c33, c55, _, buoyancy = deepwave.anisotropy.thomsen_to_cij(vp0, vs0, rho, epsilon, delta)

    out = deepwave.acoustic_vti(
        c11=c11, c13=c13, c33=c33, buoyancy=buoyancy,
        grid_spacing=5, dt=0.001,
        source_amplitudes_sigma_h=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_amplitudes_sigma_v=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations_sigma_h=torch.tensor([[[5, 20]]]),
        source_locations_sigma_v=torch.tensor([[[5, 20]]]),
        receiver_locations_sigma_h=torch.tensor([[[35, 20]]]),
        pml_width=10,
        pml_freq=25,
    )
    out.receiver_amplitudes_sigma_h  # shape [1, 1, 200]

Alongside the receiver data, ``out`` also carries the final wavefields plus
one opaque ``auxiliary_state`` holding the absorbing boundary's memory
variables. Passing the whole output back in as ``initial_state`` continues
the propagation in a later call (see :doc:`usage`).

Fields and governing equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the shear moduli vanish the elastic VTI normal stresses collapse onto two
scalars: the normal stress **across** the symmetry axis :math:`\sigma_H`
(equal in every direction perpendicular to it: :math:`\sigma_H = \sigma_{xx} =
\sigma_{yy}` in 3D, :math:`\sigma_H = \sigma_{xx}` in 2D) and the normal
stress **along** the symmetry axis :math:`\sigma_V`
(:math:`\sigma_V = \sigma_{zz}` in 3D, :math:`\sigma_V = \sigma_{yy}` in 2D).
The ``v`` and ``h`` suffixes read as vertical and horizontal only because
the symmetry axis is conventionally drawn vertical; Deepwave attaches no
orientation to any dimension, so read them as along and across that axis. With :math:`b = 1/\rho` the buoyancy and
the symmetry axis along the slowest spatial dimension :math:`z` (drop :math:`y`
in 2D), the system is

.. math::
    :label: acoustic_vti_3d

    \begin{aligned}
    \dot\sigma_H &= c_{11}\left(\partial_x v_x + \partial_y v_y\right) + c_{13}\,\partial_z v_z + s_H,\\
    \dot\sigma_V &= c_{13}\left(\partial_x v_x + \partial_y v_y\right) + c_{33}\,\partial_z v_z + s_V,\\
    \dot v_x &= b\,\partial_x \sigma_H,\qquad
    \dot v_y = b\,\partial_y \sigma_H,\qquad
    \dot v_z = b\,\partial_z \sigma_V,
    \end{aligned}

where :math:`s_H` and :math:`s_V` are optional stress-rate source terms.
Compared with running :func:`deepwave.elastic_vti` with zero shear moduli, this
stores roughly half the fields (two normal stresses instead of the full stress
tensor, and only :math:`2\,n_\mathrm{dim}` C-PML memory variables) and evaluates
none of the shear stencils.

There is no 1D form: with a single spatial dimension only :math:`c_{33}`
survives, which is the 1D scalar/acoustic equation, so 1D inputs raise an error
(use :func:`deepwave.acoustic` or :func:`deepwave.scalar`).

Parameterisation and units
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The three stiffnesses are the true (density-normalised-free) VTI stiffnesses,
identical to those of :func:`deepwave.elastic_vti`, supplied at the
normal-stress (integer) grid nodes with units of pressure (Pa in SI). In terms
of the axial P velocity :math:`v_{P0}` and Thomsen (1986)
:math:`\epsilon,\delta`,

.. math::
    :label: acoustic_vti_thomsen

    c_{33} = \rho v_{P0}^2, \qquad
    c_{11} = c_{33}(1 + 2\epsilon), \qquad
    c_{13} = c_{33}\sqrt{1 + 2\delta}.

These are exactly what :func:`deepwave.anisotropy.thomsen_to_cij` produces with
:math:`v_{S0} = 0` (its general ``c13`` relation reduces to
:math:`c_{33}\sqrt{1+2\delta}`, and ``c55 = c66 = 0``), so inversion may be
performed differentiably in Thomsen space by chaining autograd through it; the
returned ``c55`` (identically zero) is simply discarded, while the buoyancy it
also returns is passed straight to the propagator. That buoyancy
:math:`b = 1/\rho` is internally averaged onto the staggered velocity nodes
exactly as in :func:`deepwave.elastic` (a ``buoyancy = 0`` cell, with zero
stiffnesses, is a vacuum, giving a free surface); the stiffnesses are used
unaveraged.

Sources and receivers
^^^^^^^^^^^^^^^^^^^^^^

Here :math:`h=\texttt{out.info.internal_dt}` is the internal step.
Sources and receivers may be placed on any velocity component (``z`` -- 3D only
--, ``y``, ``x``; force-density sources, scaled internally by the buoyancy and
:math:`h` as in :func:`deepwave.elastic`) and on either normal stress
(``sigma_h``, ``sigma_v``; stress-rate sources :math:`s_H`, :math:`s_V`, scaled
internally by :math:`h`). An isotropic explosive/pressure source is the
**same wavelet applied to both** the ``sigma_h`` and ``sigma_v`` channels.

Stability
^^^^^^^^^

The pseudo-acoustic VTI system is well-posed only where
:math:`\epsilon \ge \delta` everywhere, equivalently

.. math::
    :label: grechka

    c_{11} c_{33} \ge c_{13}^2 \quad\text{(elementwise)}.

This is the Grechka--Zhang--Rector (2004) / Duveneck (2008) condition;
:func:`deepwave.acoustic_vti` validates it and raises ``ValueError`` otherwise.
The elliptical case :math:`\epsilon = \delta` (:math:`c_{13} = \sqrt{c_{11}c_{33}}`)
is permitted and is a useful, artifact-free test point. For the CFL time step
and PML profiles the maximum qP phase speed is bounded (when ``max_vel`` is not
supplied) by :math:`\sqrt{\max(c_{11}, c_{33})\,b}` pointwise, which is attained
on a symmetry axis. Strong anisotropy is a known trigger of C-PML corner
instability; the multiaxial-PML option (``mpml_fraction``, typically 0.15--0.2) cures
it at the cost of slightly larger boundary reflections (see :doc:`boundaries`).

Pseudo-shear source artifact
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Where :math:`\epsilon \ne \delta`, a point source in the pseudo-acoustic VTI
system emits, in addition to the qP wave, a spurious diamond/rhombic
**pseudo-shear** wave -- the kinematic remnant of the shear wave that the
``V_S = 0`` approximation is supposed to remove. This is intrinsic to the
formulation, not a numerical defect. The optional helper
:func:`deepwave.anisotropy.source_taper` mitigates it, following Duveneck et
al. (2008), by blending :math:`c_{13}` towards the locally elliptical value
:math:`\sqrt{c_{11}c_{33}}` (:math:`\delta \to \epsilon`, for which no
pseudo-shear is emitted) within a small raised-cosine radius of each source.

What it returns is a **modified** :math:`c_{13}` model, not a mask: the
propagator you then hand it to is running on a medium that differs from the
one you built, within ``radius`` cells of every source, where the anisotropy
has been made elliptical. That is what suppresses the artefact, but it also
means the near-source wavefield is not your model's wavefield, and a gradient
taken with respect to the tapered :math:`c_{13}` is not the gradient with
respect to yours there. Keep the radius to a few cells, and leave the helper
alone if the near-source medium is what you are trying to recover.

Correctness
^^^^^^^^^^^

Because :func:`deepwave.acoustic_vti` is the exact ``c55 = c66 = 0`` reduction of
:func:`deepwave.elastic_vti`, its forward field is bit-identical to
``elastic_vti`` with zero shear moduli (:math:`\sigma_V = \sigma_{zz}`,
:math:`\sigma_H = \sigma_{xx} = \sigma_{yy}`, velocities equal) in 2D and 3D --
the primary correctness check in the test suite. Independently, the measured qP
phase velocity across angle (from band-limited plane waves in a homogeneous
strongly-anisotropic medium) matches the exact ``V_S = 0`` Christoffel solution
within the grid-dispersion budget, and every model and source gradient matches
finite differences.

References
^^^^^^^^^^

Duveneck, E., Milcik, P., Bakker, P. M., and Perkins, C. (2008), "Acoustic VTI
wave equations and their application for anisotropic reverse-time migration",
*SEG Technical Program Expanded Abstracts*, 2186--2190, doi:10.1190/1.3059320.

Alkhalifah, T. (1998), "Acoustic approximations for processing in transversely
isotropic media", *Geophysics* 63, 623--631.

Grechka, V., Zhang, L., and Rector, J. W. (2004), "Shear waves in acoustic
anisotropic media", *Geophysics* 69, 576--582.

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.acoustic_vti(
        ...,
        sources={"y": (amplitudes, source_locations)},
        receivers={"y": receiver_locations},
    )

This propagator's source channels are ``pressure_rate``, ``z``, ``y``, ``x``,
``sigma_v``, ``sigma_h`` and its receiver channels are ``p``, ``z``, ``y``,
``x``, ``sigma_v``, ``sigma_h`` (the ``z`` channel is absent in 2-D).
``"pressure_rate"`` names the quantity the scalar source injects; see
:doc:`units`. ``"p"`` is still accepted there, with a
:class:`deepwave.DeepwaveFutureWarning`, and is removed in ``0.4.0``.

Both scalar channels are synthesised, because this propagator has no
pressure variable. The ``pressure_rate`` source drives both stress channels,
each receiving ``-source_amplitudes_pressure_rate``. The ``p`` receiver
records minus the mean normal stress,
``-(sigma_v + (ndim - 1) * sigma_h) / ndim`` -- the pressure a hydrophone in
this medium measures, and the same quantity every other propagator's ``p``
receiver records.

The equivalent flat ``source_amplitudes_*``/``source_locations_*``/
``receiver_locations_*`` keywords are equally permanent and
produce bit-identical results; giving one channel in both forms
raises. See :doc:`usage` for the full vocabulary.


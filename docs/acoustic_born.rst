Born Modelling with the Variable-Density Acoustic Wave Equation
=================================================================

.. note::

   Stability: **experimental** -- the label records that this propagator has
   never appeared in a release, so ``0.1.0`` is the first time real use meets
   its design. It is not a lesser promise: its interface is frozen exactly as
   the other ten propagators' are, calling it emits no warning about its
   maturity, and every other Deepwave promise applies to it unchanged. See
   :doc:`stability`.

This page describes :func:`deepwave.acoustic_born`, Born (linearised)
modelling for the variable-density acoustic wave equation described in
:doc:`acoustic`. It follows the same pattern as :doc:`scalar_born` (read
that page first for the general idea of Born modelling in Deepwave), but
linearises around the bulk-modulus/buoyancy parameterisation that
:func:`deepwave.acoustic` uses instead of scalar's single wavespeed.

Minimal example
----------------

:func:`deepwave.acoustic_born` takes the same background model as
:func:`deepwave.acoustic` plus scattering perturbations ``dv``/``drho``:

.. code-block:: python

    import torch
    import deepwave

    v = 1500 * torch.ones(40, 40)
    rho = 2200 * torch.ones(40, 40)
    dv = torch.zeros(40, 40)
    dv[20, 20] = 50.0  # a small wavespeed perturbation
    out = deepwave.acoustic_born(
        v, rho, grid_spacing=5, dt=0.001,
        dv=dv,
        source_amplitudes_volume_rate=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations_volume_rate=torch.tensor([[[5, 20]]]),
        sc_receiver_locations_p=torch.tensor([[[35, 20]]]),
        pml_width=10,
        pml_freq=25,
    )
    out.sc_receiver_amplitudes_p  # the scattered (Born) pressure, shape [1, 1, 200]

Alongside the receiver data, ``out`` also carries the final background and
scattered wavefields plus one opaque ``auxiliary_state`` holding both fields'
absorbing-boundary memory variables. Passing the whole output back in as
``initial_state`` continues the propagation in a later call (see
:doc:`usage`).

Derivation
----------

:func:`deepwave.acoustic` solves the first-order variable-density system

.. math::

    \frac{\partial \mathbf{v}}{\partial t} = -B \nabla p + B\mathbf{f},
    \qquad
    \frac{\partial p}{\partial t} = -K \nabla\cdot\mathbf{v} + Ks,

with buoyancy :math:`B = 1/\rho` and bulk modulus :math:`K = \rho c^2`.
Perturbing the constitutive fields, :math:`K \to K + \delta K` and
:math:`B \to B + \delta B`, and splitting each field into a background and a
first-order scattered part, :math:`p = \bar p + \delta p` and
:math:`\mathbf{v} = \bar{\mathbf{v}} + \delta\mathbf{v}`, gives, at zeroth
order, the ordinary background system in :math:`(K, B)`, and at first order

.. math::

    \frac{\partial \delta\mathbf{v}}{\partial t}
        = -B \nabla \delta p - \delta B \nabla \bar p,
    \qquad
    \frac{\partial \delta p}{\partial t}
        = -K \nabla\cdot\delta\mathbf{v} - \delta K \nabla\cdot\bar{\mathbf{v}}.

The scattered field :math:`(\delta p, \delta\mathbf{v})` therefore obeys
**the same acoustic operator as the background field**, driven by two
virtual sources built from the background field itself: a virtual pressure
injection :math:`-\delta K \, \nabla\cdot\bar{\mathbf{v}}` and a virtual body
force :math:`-\delta B_d \, \partial_d \bar p` for each velocity component
:math:`d`. Because :math:`\delta K` lives at the same staggered node as
:math:`K` (the pressure node) and :math:`\delta B_d` lives at the same node
as :math:`B_d` (the corresponding velocity node), no interpolation of the
scattering models is needed -- the main practical payoff of linearising in
the constitutive :math:`(K, B)` space rather than directly in
:math:`(v, \rho)`.

The user-facing perturbations are the more physically intuitive velocity and
density perturbations ``dv``/``drho``; :func:`deepwave.acoustic_born` derives
the constitutive perturbations :math:`\delta K`/:math:`\delta B_d` from them
via the same differentiable ``v, rho -> K, B`` composition that
:func:`deepwave.acoustic` itself uses, so gradients with respect to ``dv``
and ``drho`` are exact, not an approximation of the constitutive-space
gradient.

See :doc:`acoustic_born_derivation` for the full discrete derivation,
including the coupled forward update, the adjoint (both the scattered-only
and full two-field cases), and the second-order (Hessian-vector product,
forward-mode ``jvp``) structure.

Implementation
--------------

Deepwave's acoustic Born propagator simultaneously simulates two coupled
fields on the same staggered grid and with the same C-PML boundary as
:func:`deepwave.acoustic`:

1.  **Background field** :math:`(\bar p, \bar{\mathbf{v}})`: propagates in
    the background model exactly as :func:`deepwave.acoustic` would.
2.  **Scattered field** :math:`(\delta p, \delta\mathbf{v})`: propagates in
    the same background model, driven by the virtual sources above, which
    are computed from quantities the background update already forms (its
    pressure gradient and velocity divergence), so the coupling adds no new
    finite-difference stencils.

For a small perturbation (so the linear approximation holds), the sum of the
background and scattered outputs approximates the output of the regular
:func:`deepwave.acoustic` propagator run with the perturbed model
``v + dv``/``rho + drho``.

**Inputs and outputs**

The propagator takes the same background model as :func:`deepwave.acoustic`
(``v``, ``rho``) plus the scattering perturbations:

*   **Background wavespeed/density** (``v``, ``rho``).
*   **Scattering perturbations** (``dv``, ``drho``), applied to the same
    background model.

Receiver data can be recorded from both fields independently: the
``sc_receiver_locations_*`` arguments record the scattered field (the Born
data, used for LSRTM/FWI-style gradients), and the optional
``bg_receiver_locations_*`` arguments record the background field (useful
for comparing against a conventional forward model). See
:func:`deepwave.acoustic_born`'s signature below for the full set of
per-component source/receiver arguments, which mirror
:func:`deepwave.acoustic`'s doubled for the background/scattered pair.

.. seealso::

   :doc:`example_acoustic_born` demonstrates least-squares reverse-time
   migration (LSRTM) with this propagator.

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.acoustic_born(
        ...,
        sources={"volume_rate": (amplitudes, source_locations)},
        receivers={"p": receiver_locations},
        bg_receivers={"p": receiver_locations},
    )

This propagator's source channels are ``volume_rate``, ``z``, ``y``, ``x`` and its
receiver channels are ``p``, ``z``, ``y``, ``x`` (the ``z`` channel is absent in 2-D).
``"volume_rate"`` names the quantity the scalar source injects; see
:doc:`units`. ``"p"`` is still accepted there, with a
:class:`deepwave.DeepwaveFutureWarning`, and is removed in ``0.4.0``.
It is unchanged as a *receiver* channel name.
``receivers=`` means the **scattered** receivers and ``bg_receivers=`` the
background ones, matching the order of the returned tuple (background
first, scattered second) and the ``sc_``/``bg_`` flat spellings. The
recorded data comes back as ``out.receivers[channel]`` and
``out.bg_receivers[channel]``.

The equivalent flat ``source_amplitudes_*``/``source_locations_*``/
``sc_receiver_locations_p`` keywords are equally permanent and
produce bit-identical results; giving one channel in both forms
raises. See :doc:`usage` for the full vocabulary.


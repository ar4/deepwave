Scalar Wave Equation
====================

This page derives the constant-density acoustic scalar wave equation that
:func:`deepwave.scalar` and :func:`deepwave.scalar_born` solve, and shows the
exact update that Deepwave applies at each time step, including the PML
absorbing boundary. It is reference material for readers who want the
underlying physics and numerics; if you only want to run a propagation, the
example below and :doc:`quickstart` are enough.

Minimal example
----------------

:func:`deepwave.scalar` takes a wavespeed model, a grid spacing, a time step,
and a source, and returns the final wavefields and the recorded receiver
data:

.. code-block:: python

    import torch
    import deepwave

    v = 1500 * torch.ones(40, 40)
    out = deepwave.scalar(
        v, grid_spacing=5, dt=0.001,
        source_amplitudes=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations=torch.tensor([[[5, 20]]]),
        receiver_locations=torch.tensor([[[35, 20]]]),
        pml_width=10,
        pml_freq=25,
    )
    out.receiver_amplitudes  # shape [1, 1, 200]

See :doc:`quickstart` for a complete runnable example that also plots the
result, and :doc:`usage` for the full set of arguments.

Derivation
----------

We derive the constant density acoustic wave equation (a scalar wave equation) by applying Newton's second law (:math:`F=ma`) and the principle of conservation of mass. For a constant density medium, these principles yield, to first order:

.. math::
    :label: newtons_2nd_law

    -\nabla P + \mathbf{s} = \rho_0\ddot{\mathbf{u}}

and

.. math::
    :label: conservation_of_mass

    \dot{\rho} = -\rho_0\nabla\cdot\dot{\mathbf{u}},

where:

*   :math:`P` is pressure.
*   :math:`\mathbf{s}` is an external body force per unit volume (in both 2D and 3D: every term in :eq:`newtons_2nd_law` has units of force per unit volume). See :doc:`units` for what changes between 2D and 3D, which is the physical interpretation of a grid cell's amplitude rather than its units.
*   :math:`\mathbf{u}` is particle displacement.
*   :math:`\rho` is the total density.
*   :math:`\rho_0` is the constant background density.

By applying the divergence operator to :eq:`newtons_2nd_law` and the time derivative to :eq:`conservation_of_mass`, we can combine these two equations to obtain:

.. math::
    :label: p_rho_wave_equation

    -\nabla^2 P + \nabla\cdot\mathbf{s} = -\ddot{\rho}.

Assuming pressure and density are proportional, with a constant of proportionality :math:`c^2` (where :math:`c` is the wave speed):

.. math::
    :label: c_definition

    P = c^2\rho

Substituting this relationship into :eq:`p_rho_wave_equation` and rearranging the terms yields the scalar wave equation:

.. math::
    :label: constant_density_acoustic_wave_equation

    \nabla^2 P -\frac{1}{c^2}\ddot{P} = \nabla\cdot\mathbf{s}.

Implementation
--------------

Deepwave implements the scalar wave equation in the form:

.. math::
    :label: scalar_wave_equation

    \nabla^2 u - \frac{1}{c^2}\ddot{u} = f,

where :math:`u` is a scalar field (e.g., pressure) that obeys the wave equation, :math:`c` is the wave speed, and :math:`f` is a scalar source term. Sources are applied at grid cell centres. For instance, if :math:`u` represents pressure (as in :eq:`constant_density_acoustic_wave_equation`), an explosive point source centred on a grid cell corresponds to a single scalar source in :eq:`scalar_wave_equation`. Note that :math:`f` is the divergence :math:`\nabla\cdot\mathbf{s}` of the body force of :eq:`newtons_2nd_law`, so it has units of :math:`[u]` per unit area, and that the value supplied for a cell is a density: it is not divided by the cell volume, so the equivalent continuum point-source strength is the amplitude times the cell volume (see :doc:`units`).

Deepwave employs central finite differences for both time (second-order accurate) and space (user-specified order of accuracy) to numerically solve this equation within a defined time and space domain.

**Perfectly Matched Layer (PML)**

To prevent unwanted reflections from the spatial boundaries of the computational domain, Deepwave uses a `Perfectly Matched Layer (PML) <https://en.wikipedia.org/wiki/Perfectly_matched_layer>`_. For the scalar wave equation, Deepwave adopts the method proposed by `Pasalic and McGarry <https://doi.org/10.1190/1.3513453>`_. In this approach, spatial derivatives are modified within the PML regions:

.. math::
    :label: pml_deriv

    \frac{\partial}{\partial \tilde{x}} = \frac{\partial}{\partial x} + \psi,

where :math:`\psi` is an auxiliary operator defined at time step :math:`t` as:

.. math::
    :label: psi_update

    \psi^t = a\psi^{t-1} + b\left(\frac{\partial}{\partial x}\right)_t,

with :math:`a` and :math:`b` being grid-cell-dependent values. Applying this modification to :eq:`scalar_wave_equation` (simplified to one spatial dimension for illustration):

.. math::
    :label: pml_wave_equation

    \begin{align}
    c^2\frac{\partial^2 u^t}{\partial \tilde{x}^2} - \frac{\partial^2 u^t}{\partial t^2} &= c^2 f^t \\
    c^2\frac{\partial}{\partial \tilde{x}}\left(\frac{\partial u^t}{\partial x} + p^t\right) - \frac{\partial^2 u^t}{\partial t^2} &= c^2 f^t \\
    c^2\left(\frac{\partial^2 u^t}{\partial x^2} + \frac{\partial p^t}{\partial x} + z^t\right) - \frac{\partial^2 u^t}{\partial t^2} &= c^2 f^t,
    \end{align}

where :math:`p^t` and :math:`z^t` are new auxiliary variables defined as:

.. math::
    :label: p_eqn

    p^t = \psi^t u^t

.. math::
    :label: z_eqn

    z^t = \psi^t \left(\frac{\partial u^t}{\partial x} + p^t\right)

Using finite differences in time with a time step interval :math:`h_t`, the update equation for :math:`u` is:

.. math::
    :label: u_timestep

    u^{t+1} = c^2h_t^2\left(\frac{\partial^2 u^t}{\partial x^2} + \frac{\partial p^t}{\partial x} + z^t\right) + 2u^t - u^{t-1} - c^2h_t^2 f^t

To ensure computational efficiency, especially on GPUs where independent calculations are preferred, the auxiliary variable update equations are rearranged. Using :eq:`psi_update` in :eq:`p_eqn` and :eq:`z_eqn`, we can calculate :math:`p` and :math:`z` as:

.. math::
    :label: p_update

    p^t = ap^{t-1} + b\frac{\partial u^t}{\partial_x}

.. math::
    :label: z_update

    z^t = az^{t-1} + b\left(\frac{\partial^2 u^t}{\partial_x^2} + \frac{\partial p^t}{\partial x}\right)

The direct dependence of :math:`z^t` on the spatial derivative of :math:`p^t` is problematic for parallel computation. Substituting :eq:`p_update` into :eq:`z_update` yields an expression for :math:`z^t` that is more amenable to parallelisation:

.. math::
    :label: z_update_independent

    z^t = az^{t-1} + b\left(\frac{\partial^2 u^t}{\partial_x^2} + \frac{\partial \left(ap^{t-1} + b\frac{\partial u^t}{\partial_x}\right)}{\partial x}\right)

Applying the same transformation to :eq:`u_timestep` results in:

.. math::
    :label: u_timestep_independent

    u^{t+1} = c^2h_t^2\left(\frac{\partial^2 u^t}{\partial x^2} + \frac{\partial \left(ap^{t-1} + b\frac{\partial u^t}{\partial_x}\right)}{\partial x} + \left(az^{t-1} + b\left(\frac{\partial^2 u^t}{\partial_x^2} + \frac{\partial \left(ap^{t-1} + b\frac{\partial u^t}{\partial_x}\right)}{\partial x}\right)\right)\right) + 2u^t - u^{t-1} - c^2h_t^2 f^t

**Matrix Form of Time Stepping**

Through further rearrangement, each time step can be expressed in a compact matrix form. This representation facilitates the internal computation of the wavefield and auxiliary variables:

.. math::
    :label: scalar_timestep_matrix

    \begin{pmatrix}
    u^{t+1} \\
    u^t \\
    z^t \\
    p^t \\
    r^t
    \end{pmatrix} = 
    \begin{pmatrix}
    c^2h_t^2(1+b)\left((1+b)\partial_x^2 +\partial_x(b)\partial_x\right) + 2 & -1 & c^2h_t^2a & c^2h_t^2(1+b)\left(\partial_x a\right) & -c^2h_t^2 \\
    1 & 0 & 0 & 0 & 0\\
    b\left((1+b)\partial_{x}^2+\partial_x(b)\partial_x\right) & 0 & a & b\left(\partial_x a\right) & 0\\
    b\partial_x & 0 & 0 & a & 0 \\
    \delta_r & 0 & 0 & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
    u^t \\
    u^{t-1} \\
    z^{t-1} \\
    p^{t-1} \\
    f^t
    \end{pmatrix}

**Output Data**

Time sample :math:`t` of the output receiver data :math:`r` is generated by recording :math:`u^t` at the specified receiver locations :math:`\delta_r`. This ensures that the receiver data covers the same time range and has the same time sampling as the source data. It's important to note that the wavefield :math:`u` is not affected by time sample :math:`t` of the source until time step :math:`t+1` (as shown in :eq:`u_timestep`, where :math:`f^t` is added to :math:`u^{t+1}`). Consequently, the first time sample of the receiver data will not be influenced by the source, and the last time sample of the source will not affect the receiver data.

The propagator returns the final wavefield states :math:`u^{T}` and :math:`u^{T-1}` (``wavefield_0``/``wavefield_m1``), the absorbing boundary's memory variables :math:`p_y^{T-1}`, :math:`p_x^{T-1}`, :math:`z_y^{T-1}`, :math:`z_x^{T-1}` carried opaquely as one ``auxiliary_state`` field, and the recorded receiver data :math:`r`, where :math:`T` is the total number of time steps. Passing the whole output back in as ``initial_state`` continues propagation in a subsequent call; see :doc:`usage`.

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.scalar(
        ...,
        sources={"u": (amplitudes, source_locations)},
        receivers={"u": receiver_locations},
    )

This propagator's source channels are ``u`` and its receiver
channels are ``u`` (the ``z`` channel is absent in 2-D).

The equivalent flat ``source_amplitudes_*``/``source_locations_*``/
``receiver_locations`` keywords remain permanently supported and
produce bit-identical results; giving one channel in both forms
raises. See :doc:`usage` for the full vocabulary.


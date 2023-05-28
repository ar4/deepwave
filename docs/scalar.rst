Scalar wave equation
====================

Derivation
^^^^^^^^^^

To derive the constant density acoustic wave equation (which is a scalar wave equation), we use Newton's second law (:math:`F=ma`) and conservation of mass. Applied to a constant density medium, these yield, to first order,

.. math::
    :label: newtons_2nd_law

    -\nabla P + \mathbf{f} = \rho_0\ddot{\mathbf{u}}

and

.. math::
    :label: conservation_of_mass

    \dot{\rho} = -\rho_0\nabla\cdot\dot{\mathbf{u}},

where :math:`P` is pressure, :math:`\mathbf{f}` is an external force per unit area (in 2D, or per unit volume in 3D), :math:`\mathbf{u}` is particle displacement, :math:`\rho` is total density, and :math:`\rho_0` is the constant background density. Applying the divergence operator to :eq:`newtons_2nd_law` and the time derivative to :eq:`conservation_of_mass`, we can combine these two equations to yield

.. math::
    :label: p_rho_wave_equation

    -\nabla^2 P + \nabla\cdot\mathbf{f} = -\ddot{\rho}.

Pressure and density are proportional, and we will call the constant of proportionality :math:`c^2`,

.. math::
    :label: c_definition

    P = c^2\rho

Substituting this into :eq:`p_rho_wave_equation` and rearranging yields the scalar wave equation

.. math::
    :label: constant_density_acoustic_wave_equation

    \nabla^2 P -\frac{1}{c^2}\ddot{P} = \nabla\cdot\mathbf{f}.

The dimensions of each term in 2D are :math:`[ML^{-2}T^{-2}]`.

Implementation
^^^^^^^^^^^^^^

The scalar wave equation implemented in Deepwave is

.. math::
    :label: scalar_wave_equation

    \nabla^2 u - \frac{1}{c^2}\ddot{u} = f,

where :math:`u` is a scalar field that obeys the wave equation, such as pressure, :math:`c` is the wave speed, and :math:`f` is a scalar source term. Sources are added to grid cell centres. If :math:`u` is pressure (as in :eq:`constant_density_acoustic_wave_equation`) then an explosive point source centred on a grid cell corresponds to a single scalar source in :eq:`scalar_wave_equation`.

Deepwave uses central finite differences in time (with second-order accuracy) and space (with a user-specified order of accuracy) to approximately solve this equation on a given time and space domain.

To prevent unwanted reflections from the spatial boundaries of this domain, Deepwave uses a `Perfectly Matched Layer (PML) <https://en.wikipedia.org/wiki/Perfectly_matched_layer>`_. For the scalar wave equation, Deepwave uses the method of `Pasalic and McGarry <https://doi.org/10.1190/1.3513453>`_. In this approach, spatial derivatives are replaced by

.. math::
    :label: pml_deriv

    \frac{\partial}{\partial \tilde{x}} = \frac{\partial}{\partial x} + \psi,

where :math:`\psi` is an operator defined at time step :math:`t` as

.. math::
    :label: psi_update

    \psi^t = a\psi^{t-1} + b\left(\frac{\partial}{\partial x}\right)_t,

and :math:`a` and :math:`b` are values that are determined for each grid cell depending on its location.

Applying this to :eq:`scalar_wave_equation`, and, for simplicity, just using one spatial dimension,

.. math::
    :label: pml_wave_equation

    \begin{align}
    c^2\frac{\partial^2 u^t}{\partial \tilde{x}^2} - \frac{\partial^2 u^t}{\partial t^2} &= c^2 f^t \\
    c^2\frac{\partial}{\partial \tilde{x}}\left(\frac{\partial u^t}{\partial x} + p^t\right) - \frac{\partial^2 u^t}{\partial t^2} &= c^2 f^t \\
    c^2\left(\frac{\partial^2 u^t}{\partial x^2} + \frac{\partial p^t}{\partial x} + z^t\right) - \frac{\partial^2 u^t}{\partial t^2} &= c^2 f^t,
    \end{align}

where we define new auxiliary variables

.. math::
    :label: p_eqn

    p^t = \psi^t u^t

and

.. math::
    :label: z_eqn

    z^t = \psi^t \left(\frac{\partial u^t}{\partial x} + p^t\right)

Using finite differences in time, with time step interval :math:`h_t`, the update equation for :math:`u` is thus

.. math::
    :label: u_timestep

    u^{t+1} = c^2h_t^2\left(\frac{\partial^2 u^t}{\partial x^2} + \frac{\partial p^t}{\partial x} + z^t\right) + 2u^t - u^{t-1} - c^2h_t^2 f^t

Using :eq:`psi_update` in :eq:`p_eqn` and :eq:`z_eqn`, we can calculate :math:`p` and :math:`z` using

.. math::
    :label: p_update

    p^t = ap^{t-1} + b\frac{\partial u^t}{\partial_x}

and

.. math::
    :label: z_update

    z^t = az^{t-1} + b\left(\frac{\partial^2 u^t}{\partial_x^2} + \frac{\partial p^t}{\partial x}\right)

We see that :math:`z^t` depends on the spatial derivative of :math:`p^t`. This is problematic to calculate, especially on a GPU (where it is preferable for each element's calculation to be independent from those of its neighbours). We therefore substitute :eq:`p_update` into :eq:`z_update` to get

.. math::
    :label: z_update_independent

    z^t = az^{t-1} + b\left(\frac{\partial^2 u^t}{\partial_x^2} + \frac{\partial \left(ap^{t-1} + b\frac{\partial u^t}{\partial_x}\right)}{\partial x}\right)

Applying the same change to :eq:`u_timestep`, we get

.. math::
    :label: u_timestep_independent

    u^{t+1} = c^2h_t^2\left(\frac{\partial^2 u^t}{\partial x^2} + \frac{\partial \left(ap^{t-1} + b\frac{\partial u^t}{\partial_x}\right)}{\partial x} + \left(az^{t-1} + b\left(\frac{\partial^2 u^t}{\partial_x^2} + \frac{\partial \left(ap^{t-1} + b\frac{\partial u^t}{\partial_x}\right)}{\partial x}\right)\right)\right) + 2u^t - u^{t-1} - c^2h_t^2 f^t

With some rearrangement, we can express each timestep in matrix form,

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

Time sample :math:`t` of the output receiver data :math:`r` is produced by recording :math:`u^t` at the specified receiver locations :math:`\delta_r`. This means that the receiver data covers the same time range as the source data. Since the wavefield :math:`u` isn't affected by time sample :math:`t` of the source until time step :math:`t+1` (see :eq:`u_timestep`, where :math:`f^t` is added to :math:`u^{t+1}`), the first time sample of the receiver data will not be affected by the source, and the last time sample of the source does not affect the receiver data.

The propagator returns :math:`u^{T}`, :math:`u^{T-1}`, :math:`p_y^{T-1}`, :math:`p_x^{T-1}`, :math:`z_y^{T-1}`, :math:`z_x^{T-1}`, and :math:`r`, where :math:`T` is the number of time steps. 

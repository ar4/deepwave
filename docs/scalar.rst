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

where

.. math::
    :label: psi_update

    \psi_t = a\psi_{t-1} + b\left(\frac{\partial}{\partial x}\right)_t,

and :math:`a` and :math:`b` are values that are determined for each grid cell depending on its location.

Applying this to :eq:`scalar_wave_equation`, and, for simplicity, just using one spatial dimension,

.. math::
    :label: pml_wave_equation

    \begin{align}
    c^2\frac{\partial^2 u_t}{\partial \tilde{x}^2} - \frac{\partial^2 u_t}{\partial t^2} &= c^2 f_t \\
    c^2\frac{\partial}{\partial \tilde{x}}\left(\frac{\partial u_t}{\partial x} + p_t\right) - \frac{\partial^2 u_t}{\partial t^2} &= c^2 f_t \\
    c^2\left(\frac{\partial^2 u_t}{\partial x^2} + \frac{\partial p_t}{\partial x} + z_t\right) - \frac{\partial^2 u_t}{\partial t^2} &= c^2 f_t,
    \end{align}

where

.. math::
    :label: p_eqn

    p_t = \psi u_t

and

.. math::
    :label: z_eqn

    z_t = \psi \left(\frac{\partial u_t}{\partial x} + p_t\right)

Using finite differences in time, the update equation for :math:`u` is thus

.. math::
    :label: u_timestep

    u_{t+1} = c^2\Delta_t^2\left(\frac{\partial^2 u_t}{\partial x^2} + \frac{\partial p_t}{\partial x} + z_t\right) + 2u_t - u_{t-1} - c^2\Delta_t^2 f_t

Using :eq:`psi_update` in :eq:`p_eqn` and :eq:`z_eqn`, we can calculate :math:`p` and :math:`z` using

.. math::
    :label: p_update

    p_t = ap_{t-1} + b\frac{\partial u_t}{\partial_x}

and

.. math::
    :label: z_update

    z_t = az_{t-1} + b\left(\frac{\partial^2 u_t}{\partial_x^2} + \frac{\partial p_t}{\partial x}\right)

We see that :math:`z_t` depends on the spatial derivative of :math:`p_t`. This is problematic to calculate, especially on a GPU (where it is preferable for each element's calculation to be independent from those of its neighbours). We therefore substitute :eq:`p_update` into :eq:`z_update` to get

.. math::
    :label: z_update_independent

    z_t = az_{t-1} + b\left(\frac{\partial^2 u_t}{\partial_x^2} + \frac{\partial \left(ap_{t-1} + b\frac{\partial u_t}{\partial_x}\right)}{\partial x}\right)

Applying the same change to :eq:`u_timestep`, we get

.. math::
    :label: u_timestep_independent

    u_{t+1} = c^2\Delta_t^2\left(\frac{\partial^2 u_t}{\partial x^2} + \frac{\partial \left(ap_{t-1} + b\frac{\partial u_t}{\partial_x}\right)}{\partial x} + \left(az_{t-1} + b\left(\frac{\partial^2 u_t}{\partial_x^2} + \frac{\partial \left(ap_{t-1} + b\frac{\partial u_t}{\partial_x}\right)}{\partial x}\right)\right)\right) + 2u_t - u_{t-1} - c^2\Delta_t^2 f_t

With some rearrangement, we can express each timestep in matrix form,

.. math::
    :label: scalar_timestep_matrix

    \begin{pmatrix}
    u_{t+1} \\
    u_t \\
    z_t \\
    p_t
    \end{pmatrix} = 
    \begin{pmatrix}
    c^2\Delta_t^2(1+b)\left((1+b)\partial_x^2 +\partial_x(b)\partial_x\right) + 2 & -1 & c^2\Delta_t^2a & c^2\Delta_t^2(1+b)\left(\partial_x(a) + a\partial_x\right) & -c^2\Delta_t^2 \\
    1 & 0 & 0 & 0 & 0\\
    b\left((1+b)\partial_{x^2}+\partial_x(b)\partial_x\right) & 0 & a & b\left(\partial_x(a) + a \partial_x\right) & 0\\
    b\partial_x & 0 & 0 & a & 0
    \end{pmatrix}
    \begin{pmatrix}
    u_t \\
    u_{t-1} \\
    z_{t-1} \\
    p_{t-1} \\
    f_t
    \end{pmatrix}

Time sample :math:`t` of the output receiver data is produced by recording :math:`u_t` at the specified receiver locations. This means that the receiver data covers the same time range as the source data. Since the wavefield :math:`u` isn't affected by time sample :math:`t` of the source until time step :math:`t+1` (see :eq:`u_timestep`, where :math:`f_t` is added to :math:`u_{t+1}`), the first time sample of the receiver data will not be affected by the source, and the last time sample of the source does not affect the receiver data.

The propagator returns :math:`u_{T}`, :math:`u_{T-1}`, :math:`p_{y, T-1}`, :math:`p_{x, T-1}`, :math:`z_{y, T-1}`, and :math:`z_{x, T-1}`, where :math:`T` is the number of time steps. 

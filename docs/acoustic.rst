Acoustic Wave Equation
======================

Derivation
----------

The acoustic wave equation in a heterogeneous medium with variable density :math:`\rho(\mathbf{x})` and bulk modulus :math:`K(\mathbf{x})` is derived from the linearized equations of momentum conservation and mass conservation (continuity equation).

**Momentum Conservation (Equation of Motion):**

.. math::
    :label: momentum_conservation

    \rho \frac{\partial \mathbf{v}}{\partial t} = -\nabla p + \mathbf{f},

where:

*   :math:`\mathbf{v}` is the particle velocity vector.
*   :math:`p` is the acoustic pressure.
*   :math:`\mathbf{f}` is the body force source term (force per unit volume).
*   :math:`\rho` is the mass density.

**Mass Conservation (Continuity Equation):**

.. math::
    :label: mass_conservation

    \frac{1}{K} \frac{\partial p}{\partial t} = -\nabla \cdot \mathbf{v} + s,

where:

*   :math:`K = \rho c^2` is the adiabatic bulk modulus, with :math:`c` being the acoustic wave speed.
*   :math:`s` is a source term (volume injection rate).

Implementation
--------------

The system is discretized using a finite-difference time-domain (FDTD) method on a staggered grid.

**Staggered Grid Layout**

For a 2D domain in the :math:`(y, x)` plane (using :math:`y` for the first dimension and :math:`x` for the second to match the array indexing convention):

*   Pressure :math:`p` and bulk modulus :math:`K` are defined at grid nodes :math:`(y, x)`.
*   Vertical velocity :math:`v_y` and density :math:`\rho_y` are defined at half-integer steps in :math:`y`: :math:`(y+1/2, x)`.
*   Horizontal velocity :math:`v_x` and density :math:`\rho_x` are defined at half-integer steps in :math:`x`: :math:`(y, x+1/2)`.

**Time Integration**

Time stepping is performed using an explicit second-order leapfrog scheme. Pressure and particle velocities are staggered in time by half a time step :math:`h_t/2`.

1.  **Update Particle Velocities:**

    .. math:: 
        :label: velocity_update

        \mathbf{v}^{t+1/2} = \mathbf{v}^{t-1/2} - h_t \frac{1}{\rho} \nabla p^t + h_t \frac{1}{\rho} \mathbf{f}^t.

2.  **Update Pressure:**

    .. math:: 
        :label: pressure_update

        p^{t+1} = p^t - h_t K (\nabla \cdot \mathbf{v}^{t+1/2} - s^{t+1/2}).

**Absorbing Boundaries (CPML)**

To mitigate artificial reflections from the domain boundaries, a Convolutional Perfectly Matched Layer (CPML) is employed. This method introduces memory variables to implement the coordinate stretching.

For the velocity update, we replace the gradient :math:`\nabla p` with :math:`\nabla p + \mathbf{\psi}`. The memory variable :math:`\mathbf{\psi}` is updated as:

.. math::
    \psi_i^t = a_i \psi_i^{t-1} + b_i \partial_i p^t

where :math:`i \in \{y, x\}`.

For the pressure update, we replace the divergence :math:`\nabla \cdot \mathbf{v}` with :math:`\nabla \cdot \mathbf{v} + \sum_i \phi_i`. The memory variable :math:`\mathbf{\phi}` is updated as:

.. math::
    \phi_i^{t+1/2} = a_i \phi_i^{t-1/2} + b_i \partial_i v_i^{t+1/2}

The coefficients :math:`a` and :math:`b` are determined by the PML damping profile.

Matrix Form
-----------

We express the update steps in matrix form, which facilitates the derivation of the adjoint operators. We use :math:`B` for buoyancy (:math:`1/\rho`) and :math:`h_t` for the time step :math:`h_t`. The notation :math:`y'` and :math:`x'` denotes locations shifted by half a grid cell (e.g., :math:`y+1/2`).

**1. Velocity Update**

This step updates the particle velocities and the gradient memory variables.

.. math::

    \begin{pmatrix}
    v_y^{t+\frac{1}{2}} \\
    v_x^{t+\frac{1}{2}} \\
    \psi_y^t \\
    \psi_x^t
    \end{pmatrix} =
    \begin{pmatrix}
    -h_t B^{y'x} (1+b^{y'}) \partial_{y'} & 1 & 0 & -h_t B^{y'x} a^{y'} & 0 & h_t B^{y'x} & 0 \\
    -h_t B^{yx'} (1+b^{x'}) \partial_{x'} & 0 & 1 & 0 & -h_t B^{yx'} a^{x'} & 0 & h_t B^{yx'} \\
    b^{y'} \partial_{y'} & 0 & 0 & a^{y'} & 0 & 0 & 0 \\
    b^{x'} \partial_{x'} & 0 & 0 & 0 & a^{x'} & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
    p^t \\
    v_y^{t-\frac{1}{2}} \\
    v_x^{t-\frac{1}{2}} \\
    \psi_y^{t-1} \\
    \psi_x^{t-1} \\
    f_y^t \\
    f_x^t
    \end{pmatrix}
    \label{acoustic_matrix_v}

**2. Pressure Update**

This step updates the pressure and the divergence memory variables.

.. math::

    \begin{pmatrix}
    p^{t+1} \\
    \phi_y^{t+\frac{1}{2}} \\
    \phi_x^{t+\frac{1}{2}}
    \end{pmatrix} =
    \begin{pmatrix}
    -h_t K^{yx} (1+b^y) \partial_y & -h_t K^{yx} (1+b^x) \partial_x & 1 & -h_t K^{yx} a^y & -h_t K^{yx} a^x & h_t K^{yx} \\
    b^y \partial_y & 0 & 0 & a^y & 0 & 0 \\
    0 & b^x \partial_x & 0 & 0 & a^x & 0
    \end{pmatrix}
    \begin{pmatrix}
    v_y^{t+\frac{1}{2}} \\
    v_x^{t+\frac{1}{2}} \\
    p^t \\
    \phi_y^{t-\frac{1}{2}} \\
    \phi_x^{t-\frac{1}{2}} \\
    s^{t+\frac{1}{2}}
    \end{pmatrix}
    \label{acoustic_matrix_p}

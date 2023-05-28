Elastic wave equation
=====================

Derivation
^^^^^^^^^^

As in the derivation of the :doc:`scalar <scalar>` wave equation, we use Newton's second law.

.. math::
    :label: newtons_2nd_law_elastic

    \nabla\cdot\left(\mathbf{\sigma} + \mathbf{M}\right) + \mathbf{f} = \rho \ddot{\mathbf{u}},

where :math:`\mathbf{\sigma}` is the stress tensor, :math:`\mathbf{M}` is the moment tensor, :math:`\mathbf{f}` is the body force, :math:`\rho` is the density, and :math:`\mathbf{u}` is the particle displacement.

The strain can be approximated as

.. math::
    :label: strain

    \epsilon_{ij} = \frac{1}{2}\left(\partial_iu_j + \partial_ju_i\right),

and, for an isotropic medium, the stress simplifies to

.. math::
    :label: stress

    \sigma_{ij} = \lambda\delta_{ij}\epsilon_{kk} + 2\mu\epsilon_{ij},

where :math:`\lambda` and :math:`\mu` are the Lamé parameters.

Implementation
^^^^^^^^^^^^^^

As moment tensor sources can be incorporated into body forces (but may need to be split over multiple adjacent grid cells), we will only use body sources from now on. For example, an explosion, for which the moment tensor is diagonal, can be approximated by negative body force terms in the x and y cells immediately before the source and positive terms in the x and y cells immediately after the source. 

Deepwave uses the velocity-stress formulation of the elastic wave equation on a staggered grid. To obtain the velocity-stress formulation, we rewrite :eq:`newtons_2nd_law_elastic` as

.. math::
    :label: newtons_2nd_law_vel

    \nabla\cdot\left(\mathbf{\sigma}\right) + \mathbf{f} = \rho \dot{\mathbf{v}},

where :math:`v` is the particle velocity. We also rewrite :eq:`strain` as

.. math::
    :label: strain_vel

    \dot\epsilon_{ij} = \frac{1}{2}\left(\partial_iv_j + \partial_jv_i\right),

and :eq:`stress` as

.. math::
    :label: stress_vel

    \dot\sigma_{ij} = \lambda\delta_{ij}\dot\epsilon_{kk} + 2\mu\dot\epsilon_{ij}.

The staggered grid has :math:`v_x` positioned on :math:`(y, x)` grid cells, :math:`v_y` at :math:`(y+\frac{1}{2}, x+\frac{1}{2})`, :math:`\sigma_{xx}` and :math:`\sigma_{yy}` at :math:`(y, x+\frac{1}{2})`, and :math:`\sigma_{xy}` at :math:`(y+\frac{1}{2}, x)`. Staggering also occurs in time, with the stress (:math:`\sigma`) components defined at time steps :math:`t` while velocity components are at times :math:`t+\frac{1}{2}`. Staggering the components in time and space, we obtain

.. math::

    \begin{align}
    v_y^{t+\frac{1}{2}} &= v_y^{t-\frac{1}{2}} + B^{y+\frac{1}{2}, x+\frac{1}{2}}h_t\left(\partial_y\sigma_{yy} + \partial_x\sigma_{xy} + f_y^t\right) \\
    v_x^{t+\frac{1}{2}} &= v_x^{t-\frac{1}{2}} + B^{y, x}h_t\left(\partial_x\sigma_{xx} + \partial_y\sigma_{xy} + f_x^t\right) \\
    \sigma_{yy}^{t} &= \sigma_{yy}^{t-1} + h_t\left(\left(\lambda^{y, x+\frac{1}{2}}+2\mu^{y, x+\frac{1}{2}}\right)\partial_yv_{y} + \lambda^{y, x+\frac{1}{2}}\partial_xv_{x}\right)\\
    \sigma_{xx}^{t} &= \sigma_{xx}^{t-1} + h_t\left(\left(\lambda^{y, x+\frac{1}{2}}+2\mu^{y, x+\frac{1}{2}}\right)\partial_xv_{x} + \lambda^{y, x+\frac{1}{2}}\partial_yv_{y}\right)\\
    \sigma_{xy}^{t} &= \sigma_{xy}^{t-1} + h_t\left(\mu^{y+\frac{1}{2}, x}\left(\partial_xv_{y} + \partial_yv_{x}\right)\right), \\
    \end{align}

where :math:`B` is the buoyancy (reciprocal of the density), and :math:`h_t` is the finite difference time step interval.

To use the C-PML method to implement an absorbing boundary, we replace spatial derivatives by

.. math::

    \frac{\partial}{\partial \tilde{x}} = \frac{\partial}{\partial x} + \psi.

For example,

.. math::

    \partial_{\tilde{y}}\sigma_{yy} = \partial_y\sigma_{yy} + \partial_y\sigma_{yy}^m,

where :math:`\partial_y\sigma_{yy,m}` is a "memory" variable, an auxiliary wavefield that is needed for the calculation. We require one of these memory variables for each spatial derivative, resulting in the following eight auxiliary wavefields that depend on the PML profiles :math:`a` and :math:`b`.

.. math::

    \begin{align}
    \partial_y\sigma_{yy,m} &= a^{y+\frac{1}{2}}\partial_y\sigma_{yy,m} + b^{y+\frac{1}{2}} \partial_y\sigma_{yy} \\
    \partial_x\sigma_{xx,m} &= a^{x}\partial_x\sigma_{xx,m} + b^{x} \partial_x\sigma_{xx} \\
    \partial_y\sigma_{xy,m} &= a^{y}\partial_y\sigma_{xy,m} + b^{y} \partial_y\sigma_{xy} \\
    \partial_x\sigma_{xy,m} &= a^{x+\frac{1}{2}}\partial_x\sigma_{xy,m} + b^{x+\frac{1}{2}} \partial_x\sigma_{xy} \\
    \partial_yv_{y,m} &= a^{y}\partial_yv_{y,m} + b^{y} \partial_yv_{y} \\
    \partial_xv_{y,m} &= a^{x}\partial_xv_{y,m} + b^{x} \partial_xv_{y} \\
    \partial_yv_{x,m} &= a^{y+\frac{1}{2}}\partial_yv_{x,m} + b^{y+\frac{1}{2}} \partial_yv_{x} \\
    \partial_xv_{x,m} &= a^{x+\frac{1}{2}}\partial_xv_{x,m} + b^{x+\frac{1}{2}} \partial_xv_{x} \\
    \end{align}

Combining these, we can express a time step in matrix form as

.. math::

    \begin{pmatrix}
    v_y^{t+\frac{1}{2}} \\
    v_x^{t+\frac{1}{2}} \\
    \partial_y\sigma_{yy, m}^{t} \\
    \partial_x\sigma_{xx, m}^{t} \\
    \partial_y\sigma_{xy, m}^{t} \\
    \partial_x\sigma_{xy, m}^{t} \\
    r_{v_y}^{t-\frac{1}{2}} \\
    r_{v_x}^{t-\frac{1}{2}} \\
    \end{pmatrix} = 
    \begin{pmatrix}
    B^{y'x'}h_t(1+b^{y'})\partial_{y'}&0&B^{y'x'}h_t(1+b^{x'}) \partial_{x'}&1&0&B^{y'x'}h_ta^{y'}&0&0&B^{y'x'}h_ta^{x'} \\
    0&B^{yx}h_t(1+b^x)\partial_x&B^{yx}h_t(1+b^y)\partial_y&0&1&0&B^{yx}h_ta^x&B^{yx}h_ta^y&0 \\
    b^{y'}\partial_{y'}&0&0&0&0&a^{y'}&0&0&0 \\
    0&b^x\partial_x&0&0&0&0&a^x&0&0 \\
    0&0&b^y\partial_y&0&0&0&0&a^y&0 \\
    0&0&b^{x'} \partial_{x'}&0&0&0&0&0&a^{x'} \\
    0&0&0&\delta_{r_y}&0&0&0&0&0 \\
    0&0&0&0&\delta_{r_x}&0&0&0&0 \\
    \end{pmatrix}
    \begin{pmatrix}
    \sigma_{yy}^t \\
    \sigma_{xx}^t \\
    \sigma_{xy}^t \\
    v_y^{t-\frac{1}{2}} \\
    v_x^{t-\frac{1}{2}} \\
    \partial_y\sigma_{yy, m}^{t-1} \\
    \partial_x\sigma_{xx, m}^{t-1} \\
    \partial_y\sigma_{xy, m}^{t-1} \\
    \partial_x\sigma_{xy, m}^{t-1} \\
    \end{pmatrix}
    \label{elastic_matrix_v}

and

.. math::

    \begin{pmatrix}
    \sigma_{yy}^{t+1} \\
    \sigma_{xx}^{t+1} \\
    \sigma_{xy}^{t+1} \\
    \partial_yv_{y, m}^{t+\frac{1}{2}} \\
    \partial_xv_{y, m}^{t+\frac{1}{2}} \\
    \partial_yv_{x, m}^{t+\frac{1}{2}} \\
    \partial_xv_{x, m}^{t+\frac{1}{2}} \\
    r_p^t \\
    \end{pmatrix} = 
    \begin{pmatrix}
    \left(\lambda^{yx'}+2\mu^{yx'}\right)h_t(1+b^y)\partial_y&\lambda^{yx'}h_t(1+b^{x'})\partial_{x'}&1&0&0&\left(\lambda^{yx'}+2\mu^{yx'}\right)h_ta^y&0&0&\lambda^{yx'}h_ta^{x'} \\
    \lambda^{yx'}h_t(1+b^y)\partial_y&\left(\lambda^{yx'}+2\mu^{yx'}\right)h_t(1+b^{x'})\partial_{x'}&0&1&0&\lambda^{yx'}h_ta^y&0&0&\left(\lambda^{yx'}+2\mu^{yx'}\right)h_ta^{x'} \\
    \mu^{y'x}h_t(1+b^x)\partial_x&\mu^{y'x}h_t(1+b^{y'})\partial_{y'}&0&0&1&0&\mu^{y'x}h_ta^x&\mu^{y'x}h_ta^{y'}&0 \\
    b^y\partial_y&0&0&0&0&a^y&0&0&0 \\
    b^x\partial_x&0&0&0&0&0&a^x&0&0 \\
    0&b^{y'}\partial_{y'}&0&0&0&0&0&a^{y'}&0 \\
    0&b^{x'}\partial_{x'}&0&0&0&0&0&0&a^{x'} \\
    0&0&-\delta_{r_p}&-\delta_{r_p}&0&0&0&0&0 \\
    \end{pmatrix}
    \begin{pmatrix}
    v_y^{t+\frac{1}{2}} \\
    v_x^{t+\frac{1}{2}} \\
    \sigma_{yy}^{t} \\
    \sigma_{xx}^{t} \\
    \sigma_{xy}^{t} \\
    \partial_yv_{y, m}^{t-\frac{1}{2}} \\
    \partial_xv_{y, m}^{t-\frac{1}{2}} \\
    \partial_yv_{x, m}^{t-\frac{1}{2}} \\
    \partial_xv_{x, m}^{t-\frac{1}{2}} \\
    \end{pmatrix}
    \label{elastic_matrix_sigma}

where, for conciseness, half grid cell shifts are represented by a prime, for example :math:`B^{y'x'}` is the buoyancy at locations :math:`(y+\frac{1}{2}, x+\frac{1}{2})`. :math:`\delta_{r_y}`, :math:`\delta_{r_x}`, and :math:`\delta_{r_p}`, are the locations of :math:`v_y`, :math:`v_x`, and pressure receivers, respectively. In order for the velocity receiver data to cover the same time range as the input sources, the recordings are shifted by half a time step before being returned to the user.

A "free surface" refers to a surface where the traction is zero. For example, if the :math:`y` dimension is depth, then for the top surface to be a free surface we need :math:`\sigma_{yy}=0` and :math:`\sigma_{xy}=0` there. Different methods have been proposed to implement this. Deepwave currently uses the `W-AFDA <https://doi.org/10.1023/A:1019866422821>`_ approach, and applies it to all four edges so that setting the PML width to zero on any of them will result in a free surface. W-AFDA uses non-symmetric finite difference stencils near the edges so that no values beyond the free surface are needed for calculations, and imposes the constraint that the traction is zero by setting the relevant stresses to zero in the calculations.

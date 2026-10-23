Elastic wave equation
=====================

This page derives the isotropic elastic wave equation that
:func:`deepwave.elastic` solves and describes the staggered velocity--stress
scheme used to discretise it. It is reference material for readers who need
the underlying physics and numerics; the example below, and :doc:`quickstart`'s
"Elastic in ~10 lines" section, are enough to get a first result running.

Minimal example
^^^^^^^^^^^^^^^^

:func:`deepwave.elastic` takes the Lamé parameters and buoyancy (built here
from the more familiar :math:`v_p`/:math:`v_s`/:math:`\rho` with
:func:`deepwave.vpvsrho_to_lambmubuoyancy`), a grid spacing, a time step, and
a source, and returns the final wavefields, one opaque ``auxiliary_state``
holding the absorbing boundary's memory variables, and the recorded receiver
data. Passing the whole output back in as ``initial_state`` continues the
propagation in a later call (see :doc:`usage`):

.. code-block:: python

    import torch
    import deepwave

    vp = 1500 * torch.ones(40, 40)
    vs = 1000 * torch.ones(40, 40)
    rho = 2200 * torch.ones(40, 40)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

    out = deepwave.elastic(
        lamb, mu, buoyancy, grid_spacing=5, dt=0.001,
        source_amplitudes_y=deepwave.wavelets.ricker(25, 200, 0.001, 0.05).reshape(1, 1, -1),
        source_locations_y=torch.tensor([[[5, 20]]]),
        receiver_locations_y=torch.tensor([[[35, 20]]]),
        pml_width=10,
        pml_freq=25,
    )
    out.receiver_amplitudes_y  # the vertical particle velocity, shape [1, 1, 200]

Derivation
^^^^^^^^^^

As in the derivation of the :doc:`scalar <scalar>` wave equation, we use Newton's second law.

.. math::
    :label: newtons_2nd_law_elastic

    \nabla\cdot\left(\mathbf{\sigma} - \mathbf{M}\right) + \mathbf{f} = \rho \ddot{\mathbf{u}},

where :math:`\mathbf{\sigma}` is the stress tensor given by the constitutive relation below, :math:`\mathbf{M}` is the moment density tensor of any source inside the medium (a moment per unit volume, so it has the units of stress), :math:`\mathbf{f}` is the body force, :math:`\rho` is the density, and :math:`\mathbf{u}` is the particle displacement. The moment density is the source's stress glut, so the true stress is :math:`\mathbf{\sigma}-\mathbf{M}` and the moment density is equivalent to the body force :math:`f_i = -\partial_jM_{ij}` (Aki and Richards, *Quantitative Seismology: Theory and Methods*, W. H. Freeman, 1980). The minus sign matters: with the opposite sign, an explosion would radiate as an implosion.

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

Moment tensor sources can be incorporated into body forces using :math:`f_i = -\partial_jM_{ij}`, though this may need splitting the source over multiple adjacent grid cells; so we will only use body sources from now on. For example, an explosion has a diagonal moment tensor with positive entries, and approximating that derivative by a difference turns it into negative body force terms in the x and y cells immediately before the source, and positive terms in the x and y cells immediately after the source, which push outwards in both directions as an explosion should. The :doc:`moment tensor example <example_moment_tensor>` does this for a general moment tensor and checks the result against Deepwave's own explosive source.

Deepwave uses the velocity-stress formulation of the elastic wave equation on a staggered grid. To obtain the velocity-stress formulation, with any moment density source now expressed as part of the body force :math:`\mathbf{f}`, we rewrite :eq:`newtons_2nd_law_elastic` as

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

.. _Staggered grid:

The staggered grid has :math:`\sigma_{yy}` and :math:`\sigma_{xx}` positioned on :math:`(y, x)` grid cells, :math:`v_y` at :math:`(y+\frac{1}{2}, x)`, :math:`v_x` at :math:`(y, x+\frac{1}{2})`, and :math:`\sigma_{xy}` at :math:`(y+\frac{1}{2}, x+\frac{1}{2})`. The :math:`v_y` and :math:`\sigma_{xy}` components in the last row of the model, and the :math:`v_x` and :math:`\sigma_{xy}` components in the last column, are not used. This is depicted in the diagram below, where lowercase means that the component is not used. In this diagram, equals symbols and square brackets are used to denote the edges of the computational domain (so, if the PML width is zero, this will be the surface), while pipes and hyphens correspond to grid cell boundaries. The model parameters (lambda, mu, and buoyancy) are at the same locations as :math:`\sigma_{yy}` and :math:`\sigma_{xx}`::

    o--------->x
    | SII=VX==|=SII=VX==|=SII vx 
    | VY  SXY | VY  SXY | VY  sxy
    v [-------------------]------
    y SII VX  | SII VX  | SII vx 
      VY  SXY | VY  SXY | VY  sxy
      [-------------------]------
      SII=VX==|=SII=VX==|=SII vx 
      vy  sxy | vy  sxy | vy  sxy

Staggering also occurs in time, with the stress (:math:`\sigma`) components defined at time steps :math:`t` while velocity components are at times :math:`t+\frac{1}{2}`. Staggering the components in time and space, we obtain

.. math::

    \begin{align}
    v_y^{t+\frac{1}{2}} &= v_y^{t-\frac{1}{2}} + B^{y+\frac{1}{2}, x}h_t\left(\partial_y\sigma_{yy} + \partial_x\sigma_{xy} + f_y^t\right) \\
    v_x^{t+\frac{1}{2}} &= v_x^{t-\frac{1}{2}} + B^{y, x+\frac{1}{2}}h_t\left(\partial_x\sigma_{xx} + \partial_y\sigma_{xy} + f_x^t\right) \\
    \sigma_{yy}^{t} &= \sigma_{yy}^{t-1} + h_t\left(\left(\lambda^{y, x}+2\mu^{y, x}\right)\partial_yv_{y} + \lambda^{y, x}\partial_xv_{x}\right)\\
    \sigma_{xx}^{t} &= \sigma_{xx}^{t-1} + h_t\left(\left(\lambda^{y, x}+2\mu^{y, x}\right)\partial_xv_{x} + \lambda^{y, x}\partial_yv_{y}\right)\\
    \sigma_{xy}^{t} &= \sigma_{xy}^{t-1} + h_t\left(\mu^{y+\frac{1}{2}, x+\frac{1}{2}}\left(\partial_xv_{y} + \partial_yv_{x}\right)\right), \\
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
    \partial_x\sigma_{xx,m} &= a^{x+\frac{1}{2}}\partial_x\sigma_{xx,m} + b^{x+\frac{1}{2}} \partial_x\sigma_{xx} \\
    \partial_y\sigma_{xy,m} &= a^{y}\partial_y\sigma_{xy,m} + b^{y} \partial_y\sigma_{xy} \\
    \partial_x\sigma_{xy,m} &= a^{x}\partial_x\sigma_{xy,m} + b^{x} \partial_x\sigma_{xy} \\
    \partial_yv_{y,m} &= a^{y}\partial_yv_{y,m} + b^{y} \partial_yv_{y} \\
    \partial_xv_{y,m} &= a^{x+\frac{1}{2}}\partial_xv_{y,m} + b^{x+\frac{1}{2}} \partial_xv_{y} \\
    \partial_yv_{x,m} &= a^{y+\frac{1}{2}}\partial_yv_{x,m} + b^{y+\frac{1}{2}} \partial_yv_{x} \\
    \partial_xv_{x,m} &= a^{x}\partial_xv_{x,m} + b^{x} \partial_xv_{x} \\
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
    B^{y'x}h_t(1+b^{y'})\partial_{y'}&0&B^{y'x}h_t(1+b^{x}) \partial_{x}&1&0&B^{y'x}h_ta^{y'}&0&0&B^{y'x}h_ta^{x} \\
    0&B^{yx'}h_t(1+b^{x'})\partial_{x'}&B^{yx'}h_t(1+b^y)\partial_y&0&1&0&B^{yx'}h_ta^{x'}&B^{yx'}h_ta^y&0 \\
    b^{y'}\partial_{y'}&0&0&0&0&a^{y'}&0&0&0 \\
    0&b^{x'}\partial_{x'}&0&0&0&0&a^{x'}&0&0 \\
    0&0&b^y\partial_y&0&0&0&0&a^y&0 \\
    0&0&b^x \partial_x&0&0&0&0&0&a^x \\
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
    \left(\lambda^{yx}+2\mu^{yx}\right)h_t(1+b^y)\partial_y&\lambda^{yx}h_t(1+b^x)\partial_x&1&0&0&\left(\lambda^{yx}+2\mu^{yx}\right)h_ta^y&0&0&\lambda^{yx}h_ta^x \\
    \lambda^{yx}h_t(1+b^y)\partial_y&\left(\lambda^{yx}+2\mu^{yx}\right)h_t(1+b^x)\partial_x&0&1&0&\lambda^{yx}h_ta^y&0&0&\left(\lambda^{yx}+2\mu^{yx}\right)h_ta^x \\
    \mu^{y'x'}h_t(1+b^{x'})\partial_{x'}&\mu^{y'x'}h_t(1+b^{y'})\partial_{y'}&0&0&1&0&\mu^{y'x'}h_ta^{x'}&\mu^{y'x'}h_ta^{y'}&0 \\
    b^y\partial_y&0&0&0&0&a^y&0&0&0 \\
    b^{x'}\partial_{x'}&0&0&0&0&0&a^{x'}&0&0 \\
    0&b^{y'}\partial_{y'}&0&0&0&0&0&a^{y'}&0 \\
    0&b^x\partial_x&0&0&0&0&0&0&a^x \\
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

where, for conciseness, half grid cell shifts are represented by a prime, for example :math:`B^{y'x'}` is the buoyancy at locations :math:`(y+\frac{1}{2}, x+\frac{1}{2})`. :math:`\delta_{r_y}`, :math:`\delta_{r_x}`, and :math:`\delta_{r_p}`, are the locations of :math:`v_y`, :math:`v_x`, and pressure receivers, respectively. The receiver data cover the same time range as the corresponding input sources, so the velocity receiver recordings are from -half a time step earlier than the pressure recordings.

A "free surface" refers to a surface where the traction is zero. For example, if the :math:`y` dimension is depth, then for the top surface to be a free surface we need :math:`\sigma_{yy}=0` and :math:`\sigma_{xy}=0` there. Different methods have been proposed to implement this. Deepwave uses the `improved vacuum method <https://doi.org/10.1190/geo2011-0067.1>`_ approach, so setting the property models to zero at any point will result in a free surface there.

Free-surface method
^^^^^^^^^^^^^^^^^^^^^

The improved vacuum method (IVM) above is the default (``surface_method="ivm"``). It is simple: the interior update kernel runs everywhere, and the stress-free condition emerges from the zeroed vacuum parameters. But its accuracy degrades as Poisson's ratio increases, because it places the effective free surface half a grid cell above the intended position and does not adapt the surface constitutive relation.

Passing ``surface_method="pm"`` selects the parameter-modified method of `Cao and Chen (2018) <https://doi.org/10.1190/geo2018-0098.1>`_ instead. This option is provisional (see :doc:`stability`): its accuracy characteristics are still being mapped, so a second spelling may be added alongside it, while ``surface_method="ivm"`` (the default) is stable. The value ``"pm"`` and its meaning are frozen like everything else. As with IVM, the free surface (and arbitrary staircase topography) is still specified purely through the vacuum cells (``vp = vs = rho = 0`` above the surface -- there is no separate geometry argument). In addition, the medium parameters at the solid cells lying *on* the discrete surface are modified once, before time stepping, so that the stress-free condition is satisfied adaptively for any Poisson's ratio:

- The per-component normal-stress stiffnesses are replaced by their surface-reduced values. Writing the modified P-modulus as :math:`\alpha = 2\mu(\lambda+\mu)/(\lambda+2\mu)`, in 2-D a horizontal-surface cell keeps only :math:`\sigma_{xx}=\alpha\,\partial_x v_x` (and freezes :math:`\sigma_{yy}`), a vertical-surface cell keeps only :math:`\sigma_{yy}=\alpha\,\partial_y v_y` (and freezes :math:`\sigma_{xx}`), and an outer-corner/transition cell freezes both. In 3-D the same rule applies per surface-normal axis, with the two surviving normal-stress components additionally coupled by the modified cross-modulus :math:`\beta = \mu\lambda/(\lambda+2\mu)`, and both the corner and edge cells (where vacuum is adjacent along more than one axis) freeze every component. Each surface cell is classified from the vacuum geometry alone.
- The velocity nodes lying in the discrete surface plane see half the solid density (their buoyancy is doubled). In 3-D, the shear-stress nodes lying in the discrete surface plane likewise take half the interior shear modulus (in 2-D the single shear node never lies in a surface plane, so no shear modification arises).

The interior update kernel is otherwise unchanged, so PM costs nothing extra per time step -- only the one-time parameter setup differs. On rugged topography and at large Poisson's ratio PM is markedly more accurate than IVM (Cao and Chen report about 99% average accuracy in 2-D versus a spectral-element reference, against tens of percent of misfit for IVM). About 15 grid points per minimum (shear) wavelength near the surface is recommended to control staircase diffraction. ``surface_method="pm"`` requires at least one vacuum cell (there must be a surface to modify, otherwise it raises ``ValueError``), and rejects models containing fluid regions (``mu = 0`` with ``lamb != 0``, e.g. a water layer) with ``ValueError``: the vacuum-based classifier cannot distinguish fluid from air, so a fluid-solid interface would wrongly be treated as a free surface (use the default ``"ivm"`` for marine models). It is implemented in every dimensionality (1-D, 2-D and 3-D) on the eager backend (``backend="eager"``); in 1-D the modification reduces to freezing the normal stress at the discrete surface cell (the exact stress-free condition), with no density or shear modification arising. The compiled C/CUDA backend (which ``backend="auto"``, the default, resolves to for ``elastic``; ``backend="compiled"`` asserts it explicitly) currently implements PM in 2-D only (other dimensionalities raise ``NotImplementedError``). It is not available for the ``"torch_compile"`` Python mode, or, on the compiled backend, in combination with first-class boundary faces (``boundary=``), multiaxial PML (``mpml_fraction > 0``), or second-order differentiation (``create_graph=True``); those cases raise ``NotImplementedError`` (use ``backend="eager"``, which supports the model-parameter combinations). The default ``surface_method="ivm"`` is unchanged on every backend.

Specifying sources and receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel-keyed mapping form is the canonical way to specify a
survey::

    out = deepwave.elastic(
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


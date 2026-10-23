Material models and physical scope
===================================

This page states the constitutive assumptions behind the propagators. A tensor
with the right shape and finite entries is not necessarily a physical material.
Deepwave validates the family conditions below when ordinary model values are
readable, before propagation or planning, and never projects a model into an
admissible class. Under ``vmap`` or compilation, a tensor's values may be
unavailable to these checks; validity remains the caller's precondition while
static shape/device checks still apply. A user-supplied ``max_vel`` does not
validate stiffnesses or density. See :doc:`units` for sources, receivers, and
their time and spatial normalisation, and :doc:`discretisation` for the
numerical scheme.

.. _material-dimensions:

Variables and dimensions
------------------------

Use one consistent length, time, and mass system. In SI:

.. list-table:: Material and field quantities
   :header-rows: 1
   :widths: 24 49 27

   * - Symbol / argument
     - Meaning
     - SI unit
   * - :math:`c`, ``v``; :math:`v_P,v_S`
     - Wave speed; elastic P and S wave speeds
     - :math:`\mathrm{m/s}`
   * - :math:`\mathbf v`
     - Particle velocity, distinct from the argument ``v`` for wave speed
     - :math:`\mathrm{m/s}`
   * - :math:`\rho`, ``rho``
     - Mass density
     - :math:`\mathrm{kg/m^3}`
   * - :math:`b`, ``buoyancy``
     - Inverse density :math:`1/\rho` in material; zero is a separate vacuum sentinel
     - :math:`\mathrm{m^3/kg}`
   * - :math:`K,\lambda,\mu,c_{IJ}`
     - Bulk modulus, Lamé parameters, and Voigt stiffness components
     - :math:`\mathrm{Pa}`
   * - :math:`p,\sigma_{ij}`
     - Compression-positive pressure and tension-positive stress
     - :math:`\mathrm{Pa}`
   * - :math:`\epsilon_{ij}`
     - Infinitesimal strain
     - Dimensionless
   * - :math:`Q,\epsilon,\delta,\gamma` (Thomsen parameters)
     - Attenuation quality factor and anisotropy ratios, respectively
     - Dimensionless
   * - :math:`\tau_\sigma,\tau_\varepsilon`
     - Stress and strain relaxation times
     - :math:`\mathrm{s}`
   * - ``q_freq``, endpoints of ``q_band``
     - Cyclic frequency; angular frequency is :math:`\omega=2\pi f`
     - :math:`\mathrm{s}^{-1}` (Hz)

The scalar wavefield :math:`u` has the units chosen for its interpretation;
when interpreted as pressure it has units Pa and its PDE source has units
:math:`\mathrm{Pa/m^2}`. Born perturbations have the units of their named
background parameter: ``dv`` is velocity, ``drho`` density, ``dlamb``/``dmu``
pressure, and ``dbuoyancy`` inverse density. A Born perturbation need not itself
be a positive material: it is a direction about a valid background.

Scalar and variable-density acoustic media
------------------------------------------

``scalar`` solves :math:`\nabla^2u-u_{tt}/c^2=f`. Positive :math:`c` gives
the ordinary wave equation; spatial variation in :math:`c` alone does not
implement variable-density acoustics. ``acoustic`` instead solves

.. math::

   \rho\,\partial_t\mathbf v=-\nabla p+\mathbf f,
   \qquad
   \partial_t p=-K\nabla\cdot\mathbf v+K s,
   \qquad K=\rho c^2,

with positive density and bulk modulus in material. Here :math:`\mathbf f`
is force density (:math:`\mathrm{N/m^3}`), and :math:`s` is volume injection
rate per volume (:math:`\mathrm{s}^{-1}`). Zero-density/stiffness masks used
for a surface are separate degeneracies; they are not ordinary material with
an infinite or negative inverse density.

Isotropic elastic solids, fluids, and dimension reduction
----------------------------------------------------------

The constitutive law is
:math:`\sigma=\lambda\operatorname{tr}(\epsilon)I+2\mu\epsilon`.
Writing :math:`\epsilon=\epsilon_{\rm dev}+\operatorname{tr}(\epsilon)I/3`
gives the 3D strain energy density

.. math::

   W=\tfrac12\epsilon:\sigma
    =\mu\,\epsilon_{\rm dev}:\epsilon_{\rm dev}
     +\tfrac12 K_3[\operatorname{tr}(\epsilon)]^2,
   \qquad K_3=\lambda+\frac{2\mu}{3}.

A positive-energy 3D solid therefore has :math:`\rho>0`, :math:`\mu>0`,
and :math:`K_3>0`. **The Lamé parameter** :math:`\lambda` **may be negative**.
A fluid has :math:`\mu=0`, :math:`K_3>0`; its zero shear energy is deliberate.
An elastic vacuum cell is represented by zero stiffnesses and zero buoyancy,
not by taking :math:`\rho\to\infty`. The improved-vacuum treatment follows
`Zeng et al. (2012) <https://doi.org/10.1190/GEO2011-0067.1>`_; its surface
averaging is a distinct part of the scheme, with the restrictions described
in :doc:`elastic` and :doc:`boundaries`.

In 2D, Deepwave keeps the in-plane P-SV displacement/velocity and stress
components: the omitted displacement and its spatial derivatives vanish.
This is a plane-strain restriction, not a plane-stress reduction of the Lamé
parameters. The retained strain energy is positive for :math:`\mu>0` and
:math:`\lambda+\mu>0`. These weaker conditions suffice for the restricted
2D system; a model intended as a physical 3D material should satisfy the full
3D conditions above. In 1D only the longitudinal modulus
:math:`M=\lambda+2\mu>0` enters. The elastic ``p`` receiver is minus the mean
of the retained normal stresses; in 2D that is not the 3D thermodynamic
pressure. The source-conversion derivation in :doc:`units` keeps this distinction.

.. _vti-material-domain:

VTI material domains
--------------------

The symmetry axis is dimension 0: ``y`` in 2D and ``z`` in 3D. The complete
stiffness matrix and exact Thomsen conversion are given in :doc:`elastic_vti`;
the parameter definitions originate in
`Thomsen (1986) <https://doi.org/10.1190/1.1442051>`_. Although the conventional
parameter names come from a weak-anisotropy paper, the implemented stiffness
conversion uses the exact defining algebra, not its first-order expansion.

For 2D elastic VTI, the retained normal-stiffness block is
:math:`\left[\begin{smallmatrix}c_{11}&c_{13}\\c_{13}&c_{33}\end{smallmatrix}\right]`
and the shear stiffness is :math:`c_{55}`. Positive definiteness is equivalent
to

.. math::

   b>0,\quad c_{55}>0,\quad c_{11}>0,\quad c_{33}>0,
   \quad c_{11}c_{33}>c_{13}^2.

In 3D, :math:`c_{12}=c_{11}-2c_{66}`. In an orthonormal basis containing
the horizontal difference strain, the normal block splits into the scalar
:math:`2c_{66}` and

.. math::

   \begin{pmatrix}
     2(c_{11}-c_{66}) & \sqrt{2}c_{13}\\
     \sqrt{2}c_{13} & c_{33}
   \end{pmatrix}.

Thus a positive-energy 3D VTI solid has

.. math::

   b>0,\quad c_{55}>0,\quad c_{66}>0,\quad c_{33}>0,
   \quad (c_{11}-c_{66})c_{33}>c_{13}^2.

These are positive-energy material conditions. They should not be confused
with a complete classification of strong ellipticity or a fully discrete CFL
condition. Fluid and vacuum limits are semidefinite degeneracies, not strictly
positive-definite solids. As in isotropic elasticity, the 2D conditions only
constrain the represented in-plane system.

``acoustic_vti`` sets the shear stiffnesses to zero and keeps two stress-like
fields. Its physical domain has :math:`b>0`, :math:`c_{11},c_{33}>0`, and
:math:`c_{11}c_{33}\ge c_{13}^2`; equality includes elliptical anisotropy.
The propagator checks these conditions when values can be inspected, using a
scaled normal-block comparison that avoids squaring large stiffnesses.
This is the pseudo-acoustic approximation of
`Duveneck et al. (2008) <https://doi.org/10.1190/1.3059320>`_, not a general
anisotropic fluid: its extra pseudo-shear mode and the model-changing
``source_taper`` option are described in :doc:`acoustic_vti`.

.. _aligned-grid-energy:

What the interior energy argument establishes
---------------------------------------------

The aligned velocity/stress layout follows the staggered-grid approach of
`Virieux (1986) <https://doi.org/10.1190/1.1442147>`_; a separate, inspectable
implementation of this layout is the
`PETSc DMStag elastic-wave example <https://petsc.org/release/src/dm/impls/stag/tutorials/ex6.c.html>`_.
Deepwave's relevant implementation choices are:

* Normal stresses, normal strains, and their stiffness block are collocated.
  The supplied normal stiffnesses are used there without averaging.
* A shear modulus at a staggered shear node is the harmonic mean of the four
  surrounding cell values, :math:`4/\sum_{j=1}^{4}(1/\mu_j)`. Elastic VTI
  uses :math:`c_{55}` or :math:`c_{66}` according to the shear plane. The
  positive-material branch has positive means; zero/vacuum masks use a
  separate zero branch. Values near numerical zero are guarded in the code.
* Velocity-node density is the arithmetic mean of its two adjacent cell
  densities, so its buoyancy is :math:`2/(\rho_i+\rho_{i+1})`, not the
  arithmetic mean of their buoyancies. Elastic inputs are converted from
  buoyancy before this averaging.

These statements correspond to ``elastic.prepare_parameters``,
``elastic_vti.prepare_parameters``, and
``_staggered_grid.stagger_density_arithmetic_mean``. Harmonic modulus and
arithmetic density averaging are discussed in
`Moczo et al. (2002) <https://doi.org/10.1785/0120010167>`_; this citation
provides context, not a claim that Deepwave implements every averaging detail
of that paper.

For the lossless interior on an infinite uniform grid (or with vanishing
boundary terms), let :math:`D` form normal strain rates and **engineering**
shear strain rates from velocity. Its entries have units inverse length;
engineering shear rate is :math:`\partial_i v_j+\partial_j v_i`. The paired
integer/half-grid stencils in ``staggered_grid.h`` and ``_staggered_grid.py``
obey summation by parts: their reverse shifted difference is the negative
transpose. This follows term by term from the antisymmetric difference
weights at each supported spatial order, before applying boundary fills.

Let :math:`R_\rho` be the diagonal matrix of positive staggered densities,
and :math:`C` the block-diagonal matrix of collocated normal stiffnesses and
staggered shear stiffnesses. With sources and PML omitted, the semidiscrete
equations and energy are

.. math::

   R_\rho\dot{\mathbf v}=-D^T\boldsymbol\sigma,\qquad
   \dot{\boldsymbol\sigma}=CD\mathbf v,
   \qquad
   E_h=\frac{\Delta V}{2}\left(
      \mathbf v^T R_\rho\mathbf v+
      \boldsymbol\sigma^T C^{-1}\boldsymbol\sigma\right).

All terms inside the parentheses have energy-density units Pa;
:math:`E_h` is energy in 3D and energy per omitted-axis length in 2D.
For positive-definite :math:`C`,

.. math::

   \dot E_h=\Delta V\left(
     -\mathbf v^TD^T\boldsymbol\sigma+
      \boldsymbol\sigma^TD\mathbf v\right)=0.

Spatially varying positive material coefficients do not break this
cancellation. This is an interior, continuous-time discrete-space result.
It does **not** prove stability of every finite timestep, PML, ghost-cell
closure, or staircase vacuum interface. The inverse-stiffness expression also
does not apply directly to fluid/vacuum degeneracies. Finite boundaries must
supply the appropriate energy flux or a compatible closure; :doc:`boundaries`
states the accepted combinations and the tests of their reflection and energy
behaviour. CPML/MPML adds memory and damping terms, requiring its own analysis.
In particular this derivation does not guarantee arbitrary anisotropic PML
stability or a particular interface-reflection error.

Attenuating media
-----------------

``viscoacoustic`` uses :math:`K=\rho c^2` as the relaxed, zero-frequency bulk
modulus. For one SLS, finite positive :math:`Q` and positive reference frequency
give positive relaxation times with :math:`\tau_\varepsilon\ge\tau_\sigma`.
The source convention and exact sourced equations are stated in :doc:`units`;
the homogeneous constitutive response and memory update are derived in
:doc:`viscoacoustic`.

With several mechanisms, :math:`\tau_l=w_l/Q` is a weak-attenuation band fit.
Relaxation-memory approaches to approximate constant Q are established in
`Blanch, Robertsson and Symes (1995) <https://doi.org/10.1190/1.1443744>`_.
The following formula specifies Deepwave's particular fit, rather than
claiming identity with every parameter-fitting method in that literature.
For :math:`x_l=\omega\tau_{\sigma,l}`, its normalised complex modulus is

.. math::

   \frac{M(\omega)}{K}
     =1+\sum_l\tau_l\frac{i x_l}{1+i x_l},\qquad
   Q_{\rm actual}(\omega)
     =\frac{1+\sum_l\tau_l x_l^2/(1+x_l^2)}
            {\sum_l\tau_l x_l/(1+x_l^2)}.

Here the harmonic convention is :math:`e^{i\omega t}`; positive imaginary
modulus denotes loss. The fit approximates the denominator by :math:`1/Q`
and treats the numerator as approximately one. It therefore does not enforce
the supplied target Q exactly at low Q. Individual weights may be signed:
aggregate loss, rather than the sign of one weight alone, determines whether
the response is dissipative at a frequency. Large cancellation can impair
numerical accuracy, particularly for many mechanisms in a narrow band. Neither
the fit nor the elastic interior-energy argument proves passivity for every
possible fitted configuration; the documented approximation should be checked
against the attenuation required for the scientific application.

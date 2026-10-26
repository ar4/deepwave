Glossary
========

Deepwave's documentation assumes familiarity with PyTorch and with gradient-based
optimisation, but not with seismic/wave-equation modelling or geophysics. This
page collects the wave-propagation and inversion terms used elsewhere in these
docs, defined the way Deepwave itself uses them (not as an exhaustive academic
treatment), with links to where each is covered in full depth.

.. glossary::
   :sorted:

   Adjoint
      The technique Deepwave's backward pass uses to obtain gradients cheaply.
      Rather than perturbing each model cell separately, a second wavefield
      (the *adjoint wavefield*) is propagated backward in time from the
      receivers, and combined with the stored forward wavefield to produce the
      gradient. This is exactly what PyTorch's ``backward()`` computes when you
      call it on a Deepwave propagator's output -- "adjoint" (or "adjoint-state
      method") is simply the name this specific case of automatic
      differentiation has in the wave-propagation and inversion literature.
      See :doc:`what_deepwave_calculates` for exactly what Deepwave's backward
      pass does and does not guarantee.

   Dimension order
      The rule that fixes which spatial axis of a model, wavefield or
      location tensor is which. Deepwave orders them slowest-varying to
      fastest-varying, so dimension 0 is the slowest axis, and the component
      suffixes count from the *fastest*: ``_x`` is the last dimension, ``_y``
      the second-last, ``_z`` the third-last. Deepwave attaches no physical
      meaning to any axis; its own documentation and examples put depth in
      dimension 0. See :ref:`axis-convention`.

   Born
      The approximation that the wavefield in a heterogeneous medium is the
      sum of a background (unscattered) wavefield and a small scattered
      correction due to a perturbation in the medium. Deepwave's ``_born``
      propagators (:func:`deepwave.scalar_born`, :func:`deepwave.acoustic_born`,
      :func:`deepwave.elastic_born`) propagate the background wavefield through
      a smooth model and a separate scattered wavefield driven by a scattering
      potential, producing the linearised response to that perturbation (for
      :func:`deepwave.scalar`, this is exactly the Jacobian-vector product
      computed by ``torch.func.jvp``; see :doc:`example_torch_func`). See
      :doc:`scalar_born` for the model, and :doc:`example_lsrtm`/
      :doc:`example_rtm` for how it is used for imaging.

   CFL
      Short for the **Courant-Friedrichs-Lewy condition**: a stability
      requirement linking the time step ``dt``, the grid spacing, and the
      fastest wave speed in the model. A finite-difference solver diverges if
      ``dt`` is too large for a wave to be resolved as it crosses a grid cell.
      Deepwave enforces this automatically, internally using a smaller time
      step than ``dt`` if necessary; see :doc:`accuracy` for the precise
      condition and how to control it.

   Free surface
      A boundary condition representing a true material interface, such as the
      ground or sea surface, at which stress (rather than the wavefield
      itself) must vanish. This is different from the :term:`PML`, which
      absorbs outgoing waves rather than reflecting them.

      It is also different from simply setting the absorbing region's width
      to zero, which reflects but does *not* let you choose what it reflects
      as: **which** condition a bare zero-width face produces depends on the
      propagator and on the field component. It is a free surface for the
      wavefield in the scalar family, and a rigid surface for pressure in
      the staggered (acoustic and elastic) families. See
      :ref:`bare-zero-width-face` for the measured coefficient of each, and
      prefer stating the condition with ``boundary`` (or
      :func:`deepwave.boundaries`), which also uses the exact image rather
      than the approximate one. See :doc:`boundaries` for the full
      explanation, including how the elastic propagator implements a true
      free surface.

   FWI
      Short for **Full-Waveform Inversion**: iteratively adjusting a model
      (e.g. the wavespeed) so that forward-modelled data matches observed data
      as closely as possible, using the gradient of a misfit function with
      respect to the model. This is exactly the pattern used in the
      :doc:`quickstart` example, typically with more realistic data and
      models; see :doc:`example_fwi` for a complete worked example.

   GSLS
      Short for **Generalised Standard Linear Solid**: the relaxation-mechanism
      model behind Deepwave's viscoacoustic propagator
      (:func:`deepwave.viscoacoustic`), which
      give the medium a finite quality factor :math:`Q` -- realistic,
      frequency-dependent amplitude loss and the physical velocity dispersion
      that necessarily accompanies it, instead of the lossless idealisation of
      the plain acoustic/elastic equations. A single relaxation mechanism
      (Standard Linear Solid, SLS) is only approximately constant with
      frequency; combining several fitted to a target band (a *Generalised*
      SLS) approximates a constant :math:`Q` ("constant-Q") more closely
      across that band. See :doc:`viscoacoustic` for the mechanism and the
      ``Q``-to-relaxation-time fitting.

   LSRTM
      Short for **Least-Squares Reverse-Time Migration**: iteratively
      refining an :term:`RTM` image by optimising the scattering potential
      (rather than taking a single gradient step) so that Born-modelled
      scattered data matches observed scattered data -- the least-squares
      counterpart of plain RTM, in the same way :term:`FWI` is the iterative
      counterpart of a single gradient step. See :doc:`example_lsrtm` for a
      worked example.

   PML
      Short for **Perfectly Matched Layer**: the absorbing boundary Deepwave
      places around the edges of the model by default, so that outgoing waves
      are attenuated instead of reflecting back into the simulation, mimicking
      an infinite medium. See :doc:`boundaries` for how Deepwave implements it
      and how to control its width or replace it with a reflecting or
      periodic edge. A face given zero width does not absorb: it becomes a
      reflecting boundary whose type depends on the propagator (see
      :ref:`bare-zero-width-face`), so name the condition you want with
      ``boundary`` rather than implying one with a zero width.

   RTM
      Short for **Reverse-Time Migration**: forming an image of subsurface
      reflectivity by propagating recorded data backward in time and
      correlating it with the forward-propagated source wavefield. In
      Deepwave's autograd framework, this correlation forms the gradient
      with respect to a scattering potential that starts at zero, computed
      with a :term:`Born` propagator. Its numerical exactness depends on the
      history-storage and sampling settings (see :doc:`what_deepwave_calculates`).
      See :doc:`example_rtm` for a worked example.

   Staggered grid
      A finite-difference layout in which different physical quantities (e.g.
      velocity and pressure/stress) are stored at grid points offset from each
      other by half a cell, rather than all at the same points. Deepwave's
      velocity-stress propagators (acoustic, elastic, and their variants) use
      a staggered grid because it gives more accurate derivatives for the
      first-order form of the wave equations they solve. See
      :doc:`discretisation` for the details.

   Thomsen parameters
      The :math:`(v_{P0}, v_{S0}, \rho, \epsilon, \delta, \gamma)`
      parameterisation (Thomsen, 1986) commonly used to describe transversely
      isotropic (VTI) media in exploration seismology, instead of specifying
      the stiffness tensor directly. :func:`deepwave.anisotropy.thomsen_to_cij` converts
      them to the stiffnesses that the ``acoustic_vti``/``elastic_vti``
      propagators take as input; see :doc:`elastic_vti` for the exact
      relations.

   gradcheck
      Numerically verifying an analytic gradient by comparing it against a
      finite-difference approximation. Deepwave's own test suite relies
      heavily on PyTorch's
      :func:`torch.autograd.gradcheck.gradcheck` this way to verify backward
      passes are correct, and you can use it the same way to sanity-check your
      own use of Deepwave (e.g. a custom loss function or model
      parameterisation) before trusting an optimisation result. See
      :doc:`what_deepwave_calculates` for what Deepwave's own gradients are
      checked against, and what is/isn't guaranteed to machine precision.

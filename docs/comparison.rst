Choosing a wave-propagation package
====================================

Deepwave is one of several open-source tools for simulating and inverting wave
propagation. They optimise for different things — differentiability, physics
breadth, HPC scale, a specific application domain — and the best choice depends
on what your project actually needs. This page is an honest comparison,
written by Deepwave's maintainer, including the cases where another package is
the better choice. Claims about other projects are current as of mid-2026;
check each project's own documentation for the latest state, since this is a
fast-moving area.

PyTorch-native differentiable propagators
------------------------------------------

**ADFWI** (`github.com/liufeng2317/ADFWI <https://github.com/liufeng2317/ADFWI>`_,
described in a 2025 JGR: Machine Learning and Computation paper) is, like
Deepwave, a PyTorch finite-difference propagator that differentiates through
wave propagation for full-waveform inversion. It supports acoustic and elastic
modelling including VTI/HTI anisotropy, and ships a rich library of misfit
functions (optimal-transport/Wasserstein, soft-DTW, envelope, and more) plus
deep-image-prior reparameterisation and dropout-based uncertainty
quantification as built-in features. It differentiates by taping the forward
stencil with PyTorch autograd rather than using hand-derived adjoints, which
is simple and flexible but memory-hungry for long simulations. **Prefer ADFWI
over Deepwave** if you need TTI/HTI (tilted or horizontal transverse isotropy)
anisotropy today -- Deepwave's :mod:`deepwave.elastic_vti` covers VTI only --
or want its built-in exotic misfit functions and uncertainty-quantification
tooling without writing them yourself.

**SWEEP** (`github.com/DeepWave-KAUST/sweep <https://github.com/DeepWave-KAUST/sweep>`_,
`arXiv:2604.14189 <https://arxiv.org/abs/2604.14189>`_) is a KAUST project
covering 20+ wave-equation variants (including VTI/TTI, viscoelasticity, and
DAS strain-rate modelling) with both PyTorch and JAX backends and compiled CUDA
kernels; it absorbed the earlier SeisTorch project, which is now in
maintenance mode. As of mid-2026 SWEEP is pre-release and still maturing.
**Prefer SWEEP over Deepwave** if you need 3-D TTI anisotropy today, and can
tolerate a younger, less-tested codebase, or if you want a JAX backend
option. Deepwave supports VTI anisotropy (:mod:`deepwave.elastic_vti`), DAS
receivers (:mod:`deepwave.das`), and viscoacoustic attenuation
(:mod:`deepwave.viscoacoustic`, constant-Q single-SLS). TTI and viscoelastic
propagation are not currently supported.

.. note::

   The GitHub organisation ``DeepWave-KAUST``, which hosts SWEEP, is **not**
   affiliated with this project. This package (`ar4/deepwave
   <https://github.com/ar4/deepwave>`_) predates that organisation's use of
   the name; the similarity is coincidental.

Symbolic code generation: Devito
---------------------------------

`Devito <https://www.devitoproject.org/>`_ generates optimised C code from a
symbolic description of a PDE stencil, giving it enormous flexibility to
express essentially arbitrary finite-difference physics, and it scales to
multi-node clusters via MPI domain decomposition — something Deepwave does not
attempt. It has real industry adoption and a commercial offering
(DevitoPRO) for production seismic workloads. Its adjoints are derived at the
equation level by its symbolic engine rather than obtained through a general
autodiff framework, and it JIT-compiles kernels at runtime rather than shipping
prebuilt wheels; it also does not support Windows. **Prefer Devito over
Deepwave** if you need multi-node distributed-memory 3-D propagation, need to
express a stencil physics that Deepwave doesn't implement, or are already
integrated with its ecosystem.

JAX: j-Wave
-----------

`j-Wave <https://github.com/ucl-bug/jwave>`_ implements acoustic and Helmholtz
wave propagation in JAX, giving you the full JAX transform stack (``grad``,
``vmap``, ``jit``) over the simulation. Like ADFWI, it differentiates by
taping the forward computation rather than using hand-derived adjoints, so
long simulations can be memory-hungry. Its release cadence has slowed (no
release since September 2024 as of mid-2026), and it does not support Windows
natively. **Prefer j-Wave over Deepwave** if your project is built around JAX
and you specifically need its Helmholtz (frequency-domain) solver, which
Deepwave does not offer.

Ultrasound k-space: k-Wave
--------------------------

`k-Wave <https://github.com/ucl-bug/k-wave>`_ (and its Python bindings,
`k-wave-python <https://github.com/waltsims/k-wave-python>`_) is the
established standard for medical-ultrasound and photoacoustic simulation. It
uses a k-space pseudospectral method rather than finite differences, natively
supports power-law frequency-dependent absorption (the standard model for
soft-tissue attenuation), and provides ``kWaveArray`` for modelling realistic
transducer element geometries. **k-Wave is not differentiable** — it has no
gradient or inversion capability. Deepwave has a constant-Q, single-mechanism
SLS attenuation propagator (:mod:`deepwave.viscoacoustic`) and transducer-array
helpers (:mod:`deepwave.transducers`), but not k-Wave's power-law
frequency-dependent absorption model (a multi-SLS fit to a measured power law
is the honest bridge, and remains future work). **Prefer k-Wave over Deepwave**
if you need power-law tissue absorption specifically, or its more mature
k-space numerics for a given points-per-wavelength budget, and do not need
gradients.

Spectral elements: SPECFEM / SPECFEM++, Salvus
------------------------------------------------

The SPECFEM family (`SPECFEM3D <https://specfem.org/>`_ and successors) and
the commercial `Salvus <https://mondaic.com/>`_ use the spectral-element
method on unstructured, deformable meshes. This lets the mesh conform to
real topography and complex geological interfaces rather than approximating
them with a staircased regular grid, and both scale to large HPC clusters.
This is the right tool when geometric fidelity or continental/global-scale
seismology is the point. **Prefer SPECFEM/SPECFEM++ or Salvus over Deepwave**
if you need mesh conformance to topography or curved interfaces, or
global/regional-scale seismology on HPC infrastructure; Deepwave's regular
Cartesian grid only staircases such features.

None of the above are a good fit if you specifically need a MATLAB workflow;
most of these tools, Deepwave included, are Python-first (k-Wave has the
longest-standing MATLAB heritage alongside its newer Python bindings).

When Deepwave is the right choice
----------------------------------

Deepwave is a good fit when you want:

- Machine-precision gradients (and, uniquely among these packages, true
  second-order Hessian-vector products through compiled kernels) without
  paying autograd's memory cost for taping every time step.
- Bitwise-reproducible gradients: on a GPU the same call gives the same
  gradients every time (no atomic adds, in the CUDA kernels or in the
  PyTorch operations around them), and on a CPU the compiled backend gives
  the same forward outputs and gradients at any thread count. Both matter
  for debugging optimisation runs and reproducing published results
  exactly.
- A dependency-light compiled backend: prebuilt wheels for Linux, Windows,
  and macOS (Apple Silicon), with a C/C++/CUDA core that does not link against
  ``libtorch`` and keeps working across supported PyTorch version upgrades
  without rebuilding.
- Ordinary PyTorch idioms throughout: the propagators are standard
  ``autograd.Function``/``nn.Module`` objects, so they compose with
  optimisers, ``DistributedDataParallel`` over shots, ``torch.func``
  transforms, and arbitrary torch losses or neural-network
  reparameterisations, with no special integration layer -- see
  :doc:`example_parameterisation` for a worked impedance/velocity and
  slowness/velocity FWI recipe, built from plain PyTorch with no dedicated
  "impedance equation" needed.
- A tested, single-workstation-to-single-GPU workflow for constant/variable-
  density acoustic, elastic and viscoacoustic modelling, elastic VTI and
  pseudo-acoustic VTI (:mod:`deepwave.acoustic_vti`, Duveneck et al. 2008)
  wave equations (including Born/linearised modelling), rather than a
  multi-node HPC deployment.

Deepwave is currently **not** the right choice if you need, today:

- TTI or HTI (tilted or horizontal transverse isotropy) anisotropy, or
  viscoelastic propagation;
- k-Wave-style power-law frequency-dependent tissue absorption (Deepwave's
  :mod:`deepwave.viscoacoustic` is a
  constant-Q GSLS models, not a power-law fit);
- multi-node distributed-memory 3-D propagation (Deepwave parallelises
  over shots on a single machine, not via MPI domain decomposition); or
- mesh conformance to topography or curved interfaces, or unstructured
  meshes (Deepwave is a regular-grid finite-difference code, though
  :mod:`deepwave.elastic`'s parameter-modified free surface handles
  staircase topography).

In each of those cases, one of the packages above is likely a better fit
today.

Summary table
-------------

"Differentiable" means gradients with respect to model parameters are
available. "Compiled kernels" means the core time-stepping loop runs as
compiled C/C++/CUDA rather than being expressed as traced/interpreted
tensor ops. "Physics breadth" is a rough indication of equation variants
supported (anisotropy, attenuation, etc.), not modelling quality.
"Second-order autograd" means Hessian-vector products are available through
the same compiled/traced machinery as the gradient, not merely by external
finite differences of the gradient.

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 20 12 14 16

   * - Package
     - Differentiable
     - Compiled kernels
     - Physics breadth
     - Multi-node
     - Windows wheels
     - Second-order autograd
   * - Deepwave
     - Yes
     - Yes (hand-derived adjoints)
     - Acoustic/elastic (const. + variable density), VTI, viscoacoustic, DAS
     - No (shot-parallel, single machine)
     - Yes
     - Yes (scalar, acoustic, elastic)
   * - ADFWI
     - Yes
     - No (autograd-taped)
     - Acoustic/elastic + VTI/HTI
     - No
     - Unclear/likely (pure PyTorch)
     - No
   * - SWEEP
     - Yes
     - Yes (compiled CUDA)
     - 20+ variants incl. VTI/TTI, viscoelastic, DAS
     - Unclear (pre-release)
     - Unclear (pre-release)
     - No
   * - Devito
     - Adjoint at equation level (not autodiff)
     - Yes (JIT-generated C)
     - Very broad (user-defined stencils)
     - Yes (MPI)
     - No
     - No
   * - j-Wave
     - Yes
     - No (JAX-traced)
     - Acoustic + Helmholtz
     - No
     - No
     - Via JAX (not kernel-level)
   * - k-Wave
     - No
     - Yes (k-space pseudospectral)
     - Acoustic, power-law absorption
     - No
     - Yes
     - No
   * - SPECFEM/SPECFEM++
     - No (forward simulation focus)
     - Yes
     - Very broad, HPC-scale
     - Yes (MPI)
     - No
     - No
   * - Salvus
     - Adjoint-based (commercial)
     - Yes
     - Very broad
     - Yes
     - Unclear
     - No

This table reflects each project's public documentation as of mid-2026 and may
be out of date by the time you read it; verify against the linked project
before relying on it.

.. _backends:

Backends
========

Every one of the nine propagators takes a ``backend`` argument that selects
which implementation of the time loop runs the call. The argument is a
string, and the same four values mean the same thing on all nine:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - What it selects

   * - ``"auto"``
     - The default. Native C/CUDA execution, with the documented eager
       fallback for elastic-Born M-PML.

   * - ``"compiled"``
     - The compiled C/CUDA kernel, strictly. Raises rather than resolving
       around a gap.

   * - ``"eager"``
     - The pure-PyTorch reference time loop, stepped under native autograd.

   * - ``"torch_compile"``
     - The same pure-PyTorch step functions under ``torch.compile`` with
       PyTorch's defaults.

Any other value raises :exc:`deepwave.DeepwaveValueError` listing the four
accepted ones. The values are matched exactly, so ``"Eager"`` is not
accepted. ``out.info.backend`` (see :class:`deepwave.RunInfo`) reports which
backend actually ran, and is never ``"auto"``.

What each value does
--------------------

``"auto"``
^^^^^^^^^^

``"auto"`` selects the compiled C/CUDA backend except for
:func:`deepwave.elastic_born` with ``mpml_fraction > 0``, which selects eager.
It does not search all backends for a working combination or retry a failed
call. For example, native VTI JVPs and elastic PM surfaces outside the native
supported subset require an explicit ``backend="eager"``. Higher derivatives
requested after the forward call cannot change the selected backend.

The fallback is silent: it is reported in ``out.info.backend`` rather than
in a warning, so that it is checkable in code inside a training loop or a
distributed worker. Unsupported combinations raise with the relevant
restriction; consult :doc:`capabilities` and the propagator's documentation
before choosing a backend for transforms or advanced boundary options.

For ``elastic_born(mpml_fraction>0)``, that eager fallback retains a PyTorch
autograd graph instead of using native history offload or compression, so its
memory cost can be substantially higher. Its model derivative follows every
internal step; unlike the native compiled gradient, it does not use
``model_gradient_sampling_interval`` to sample the correlation. Check
``out.info.backend`` before relying on native memory or sampled-gradient
behaviour. Combining this fallback with ``storage_mode="cpu"``/``"disk"`` or
``storage_compression=True`` raises rather than pretending those settings ran.

``"compiled"``
^^^^^^^^^^^^^^

The compiled C/CUDA kernel and nothing else. Unsupported native option
combinations raise :exc:`deepwave.DeepwaveNotImplementedError` with the
relevant restriction instead of resolving around it. The installed library
must also contain kernels for the requested device: a CPU-only build cannot
run CUDA tensors, even if PyTorch can see a GPU. Use ``"compiled"`` to assert
that the fast path is really being taken.

``"eager"``
^^^^^^^^^^^

The pure-PyTorch reference time loop, stepped under native autograd. It is
slower, but it composes with ``torch.func`` transforms: pass it explicitly
when you wrap a call in ``torch.func.vmap``/``grad``/``jacrev``, which
Deepwave cannot detect from inside the call through any public PyTorch API.

``"torch_compile"``
^^^^^^^^^^^^^^^^^^^

The eager step functions under ``torch.compile`` with its defaults. The first
call in a process pays the compilation cost, which can be several seconds;
later calls reuse the same PyTorch wrapper. PyTorch decides when a change in
shape, dtype or accuracy needs recompilation and applies its normal cache
limits. After PyTorch's recompilation limit, PyTorch may run the step eagerly
and emit a warning. Deepwave does not override the process's compiler settings.

Speed
-----

The documented ``"auto"`` fallback from the compiled kernel to eager
PyTorch can be tens of times slower (see
:doc:`capabilities`). Because the fallback is reported rather than warned
about, a call that is unexpectedly slow is worth checking against
``out.info.backend``; ``backend="compiled"`` turns the assumption that the
fast path ran into an assertion.

Device support
--------------

The compiled backend provides CPU and CUDA kernels; it is selected per call
from the propagator, number of dimensions, finite-difference accuracy,
dtype and device, in ``float32`` and ``float64`` only. The ``"eager"`` and
``"torch_compile"`` backends are ordinary PyTorch, so they run wherever the
tensor operations they use run.

Numerical agreement between backends
------------------------------------

Backend agreement is conditional. With lossless storage, effective
model-gradient sampling stride one, fixed inferred parameters and no callback
mutation, compiled and eager model gradients are checked against each other
to the applicable :ref:`numerical-standard`; individual propagators document
any narrower approximation. When the effective stride is greater than one,
compiled kernels sample the model-gradient correlation, while
``"eager"``/``"torch_compile"`` differentiate every internal step and ignore
``model_gradient_sampling_interval`` for that derivative. Their model
gradients can then differ materially for the same call. Source and
initial-state gradients follow every internal step. A backward callback that
modifies an accumulator deliberately changes the returned gradient, so it is
not covered by an unmodified-backend agreement claim. See
:ref:`gradient-guarantees`. **Bitwise equality between two backends is never
promised**, any more than it is between two devices or two builds;
the reproducibility promise that is made is about repeated runs of one
build, on one device, at one configuration, and is stated in
:doc:`what_deepwave_calculates`.

Some propagators note a narrower agreement in their own reference page -- for
example :func:`deepwave.acoustic_vti` states where its compiled model
gradients match the eager ones to near machine precision and where they are
instead a bounded approximation. Those per-propagator facts live with the
propagator.

Transforms and second-order differentiation
-------------------------------------------

``torch.func`` transforms and double-backward are not supported uniformly
across backends or across propagators; the
:doc:`capability matrix <capabilities>` has the current per-propagator,
per-backend answer, probed from the code itself. The general rules are:

* Choose ``backend="eager"`` explicitly for transforms that the native
  implementation does not support. Native JVPs are available for scalar,
  acoustic and elastic; the other families need eager. ``"auto"`` does not
  negotiate transform support. See :ref:`torch-func`.
* Second-order differentiation (``create_graph=True``, Hessian-vector
  products) is available on the compiled backend only for the propagators
  whose adjoint has an analytic second-order path. The others raise a
  :exc:`deepwave.DeepwaveNotImplementedError` on the compiled backend that
  names ``backend="eager"`` as the alternative, because the eager time loop
  differentiates through native autograd.
  For ``scalar``, ``acoustic`` and ``elastic``, compiled model double backward
  requires an effective model-gradient sampling stride of one. With a larger
  stride the first gradient is sampled and its Jacobian need not be symmetric;
  the compiled second-order path rejects the call. Use ``backend="eager"`` or
  a CFL-stable time step with ``model_gradient_sampling_interval=1``.
* ``"torch_compile"`` uses PyTorch's default graph-break and recompilation
  behaviour. ``out.info.backend`` records the requested wrapper, while PyTorch
  may run a step eagerly after its recompile limit is reached. Select
  ``backend="compiled"`` to require the native C/CUDA kernel.

Stability
---------

``"auto"``, ``"compiled"`` and ``"eager"`` are stable. The set of *Python
execution modes* -- which Python-level modes exist beyond ``"eager"``, and
what each supports -- is provisional, because it follows PyTorch's own
moving parts; see :doc:`stability`.

``python_backend``
------------------

``python_backend`` is a deprecated alias for ``backend`` on ``scalar``, ``scalar_born``, ``acoustic``, and ``elastic``, **removed in Deepwave 0.4.0**. Passing it emits a
:exc:`deepwave.DeepwaveFutureWarning`, and passing both it and a non-default
``backend`` raises :exc:`deepwave.DeepwaveValueError`. Its values map as
follows:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``python_backend``
     - ``backend``

   * - ``"eager"``
     - ``"eager"``

   * - ``"jit"``
     - ``"eager"`` (the ``torch.jit.script`` mode no longer exists; these are
       the same step functions, so results differ only in floating-point
       rounding)

   * - ``"compile"``
     - ``"torch_compile"``

   * - ``True``
     - ``"torch_compile"`` if the installed compiled library was built with
       OpenMP support, ``"eager"`` otherwise

   * - ``False``
     - ``"auto"``

Note that ``python_backend=True`` therefore triggers a ``torch.compile`` on
the first call on most installations.

Capability matrix
==================

Deepwave's nine propagators share a common calling convention but do not all
support the same options: which dimensionalities, which ``backend``
values, which boundary faces, ``mpml_fraction``, ``surface_method``, batching
(``torch.func.vmap``), second-order differentiation (``create_graph``),
forward-mode AD (``torch.func.jvp``), and the callback hooks. The tables
below answer "does propagator X support Y?" in one place. Use this page once
you already know which propagator you want, to check whether a specific
option is available on it before relying on the combination.

**Every shared option is accepted on every propagator.** ``mpml_fraction``,
``surface_method``, ``q_freq``, ``q_n_mechanisms`` and ``q_band`` appear in
all nine signatures, at the same default, so that a wrapper can forward
one uniform set of keyword arguments to whichever propagator it was handed
without first asking which options that propagator implements. The one
exception is that ``viscoacoustic`` **requires** ``q_freq``
(unless ``q_n_mechanisms > 1``, which requires ``q_band`` and rejects
``q_freq``): a quality factor without the frequency it was specified at is
not an attenuation model, so there is no default for a wrapper to forward. What the
columns below report is therefore whether a **non-default value** is
*acted on*: a value the propagator cannot act on raises a
``DeepwaveValueError`` naming the option, the propagator, and the
propagators that do implement it. That is the distinction the "yes"/"no"
cells make.

The cells are per-option answers, **not an unrestricted product of rows and
columns**. The backend selected for one option can rule out another. For
example, ``elastic_born(mpml_fraction>0)`` uses eager under ``backend="auto"``;
adding ``storage_mode="disk"``, lossy compression or a backward callback is
then rejected because eager does not implement those facilities. The runtime
guards decide the full call and raise rather than silently dropping an option.

**How this page stays accurate.** The generated table reads declared support
from ``PropagatorSpec`` and the finite set of family contracts in
``deepwave._capabilities``. Building documentation does not run a propagation
or infer support from a caught exception. ``tests/test_capabilities.py``
compares runtime probes with those declarations and independently checks
selected option intersections. A promised feature that raises unexpectedly
therefore fails validation instead of disappearing from this table. Each
individual cell describes a supported contract; the runtime guards still
decide combinations of options.

A few columns need more explanation than a table cell can hold:

Status
    The propagator's stability tier, as defined by the :doc:`stability`
    policy. It is a documentation label: ``experimental`` propagators are
    exported, documented and tested exactly like the others, and the
    interface promise Deepwave makes is identical on all
    nine -- no call emits a warning about its own maturity, and nothing on
    any of the nine is renamed, removed or reordered after ``0.1.0``. What
    the label means is that the propagator has never appeared in a release,
    so ``0.1.0`` is the first time real use meets its
    design. The five propagators that have never appeared
    in a release -- ``acoustic_born``, ``acoustic_vti``, ``viscoacoustic``,
    ``elastic_born`` and ``elastic_vti`` -- are ``experimental``; the four released in
    ``v0.0.27`` (``scalar``, ``scalar_born``, ``acoustic``, ``elastic``) are
    ``stable``. The :mod:`deepwave.transducers` module and the
    ``elastic``/``Elastic`` ``surface_method="pm"`` option are
    ``provisional`` -- their shape may still be corrected, and never
    without a release's notice -- but are not rows in this table
    (``transducers`` is not a propagator; ``surface_method="pm"`` is one
    option of the stable ``elastic`` propagator, not the whole propagator)
    -- see :doc:`stability` for the full per-name list.

mpml_fraction
    Implemented only by the propagators whose physics can actually provoke
    the CFS-C-PML corner instability M-PML cures: the elastic family and
    ``acoustic_vti`` (the pseudo-acoustic reduction of ``elastic_vti``). The
    isotropic scalar/acoustic propagators -- including ``acoustic_born`` and
    ``viscoacoustic`` -- have no shear waves and no free-surface waveguide
    mode (see :doc:`boundaries`), so there is nothing for it to cure there:
    they accept ``mpml_fraction=0.0`` and raise ``DeepwaveValueError`` on anything
    else. Where it is implemented it is not
    always backend-uniform: ``elastic_born`` accepts the keyword but its
    compiled M-PML *forward* kernel is not implemented yet, so ``mpml_fraction >
    0`` there runs on the eager backend. ``backend="auto"`` selects it for
    you (``out.info.backend`` then reads ``"eager"``); only
    ``backend="compiled"``, which asserts that the compiled kernel ran,
    raises. Every other ``mpml_fraction``-supporting propagator accepts it on every
    backend.

Storage modes
    ``storage_mode`` controls where wavefield snapshots needed for the
    backward pass are kept (``"device"``, ``"cpu"``, ``"disk"``, or
    ``"none"`` to keep no history, which rules out model gradients). The compiled backend's
    ``StorageManager`` supports all four for every propagator that has a
    compiled backend at all; the pure-Python ``eager``/``torch_compile``
    backends run their own native-autograd time loop with
    no ``StorageManager`` and so accept ``"device"`` only, for every
    propagator. Use ``backend="compiled"`` to require offload or compression;
    ``"auto"`` also selects it where all requested options have compiled
    implementations. The two Born families additionally reject every
    differentiable input under ``storage_mode="none"``.

Backends
    ``backend="auto"`` (the default) selects native C/CUDA execution with
    the explicit elastic-Born M-PML fallback described below. It does not
    search backends for arbitrary option combinations or autodiff transforms.
    Request ``backend="eager"`` explicitly for a supported eager-only case.

    Two things make that choice visible rather than hidden.
    ``out.info.backend`` reports the backend that actually ran (never
    ``"auto"``), which is machine-checkable inside a training loop or a
    distributed worker, where a warning would not be seen. And
    ``backend="compiled"`` turns the assumption into an assertion.

    ``elastic_born`` with
    ``mpml_fraction > 0`` is the automatic fallback implemented today, so ``"auto"`` resolves
    to ``"eager"`` for that call and to the compiled backend for every other
    ``elastic_born`` call. The selected backend's restrictions still apply;
    for example, eager M-PML cannot be combined with native disk storage.

    ``backend="compiled"`` raises
    ``deepwave.DeepwaveNotImplementedError``, naming the propagator, dtype,
    ndim, device and the option at fault, wherever the compiled kernel is
    unavailable, rather than resolving around it the way ``"auto"`` does.
    ``"torch_compile"`` (``torch.compile`` over the same eager
    reference implementation) is available for every propagator, since it
    only needs the eager reference kernels themselves to be traceable.
    Batching (``vmap``) and
    forward-mode AD (``jvp``) behave differently again on the compiled
    backend -- see those columns below.

Boundary faces accepted
    Which ``boundary`` face configurations the propagator **accepts and
    runs**. Each face type is called on its own (``boundary=["free"] * 4`` and
    so on, with ``pml_width=0`` on every face for the non-absorbing types),
    and then one mixed per-face combination is called as well, since accepting
    each type individually is not by itself evidence that they can be
    combined. ``free`` and ``rigid`` are each probed twice, on every face and
    on the low faces alone (the beginning of each axis, with the high faces
    left as zero-width ``pml``), because that is the distinction Deepwave
    makes: a guard-ring fill that conserves a discrete energy on the low side
    of an axis need not conserve on the high side. Every one of those calls is
    made on both the default and the ``eager`` backend, and the cell names the
    backends that ran it. See :doc:`boundaries` for what each face type means
    and for its measured reflection coefficient.

    **This column is probed at the default** ``accuracy`` **of 4.** The
    low-face restriction on ``free`` (acoustic family) and on ``rigid``
    (elastic family) applies from ``accuracy=4`` upwards only; at
    ``accuracy=2`` those propagators accept the face on either side of an
    axis. :ref:`boundary-faces-accepted` states the whole rule, and is the
    page to read before relying on a face.

    **"Accepted" is not an accuracy guarantee.** The probe behind this column
    runs a handful of time steps on a 10-cell grid: enough to establish that a
    configuration is implemented and does not raise, and nothing more. It is,
    however, now a statement about long-run stability as well, because a face
    Deepwave accepts is one whose fill was measured to conserve a discrete
    energy: a face that does not conserve raises
    :class:`deepwave.DeepwaveNotImplementedError` during argument validation
    instead of running and growing. The measurement is
    ``benchmarks/boundary_energy.py``.

Batching (vmap)
    ``torch.func.vmap`` over a propagator's property models (velocity, Lamé
    parameters, stiffnesses) works on the eager backend for every propagator
    except ``acoustic_born`` and ``elastic_born``, which currently reject
    ``vmap`` over any of their model tensors outright. This is a real gap
    relative to their non-Born siblings and to ``scalar_born``, which does
    support it (see ``deepwave._common.check_inputs_not_vmapped``'s call
    sites). The compiled backend never supports ``vmap`` -- the ``ctypes`` C/CUDA call
    cannot operate on the batched tensors a transform produces -- and raises
    an actionable error pointing at the eager backend instead.

    What that covers is one model argument at a time: every model argument of
    every propagator has been measured, one batched and the rest ordinary
    tensors, including the ones that reach only part of the time step -- such
    as ``elastic_vti``'s ``c55``, which enters the shear stresses alone and no
    normal stress. ``tests/test_vmap_models.py`` runs
    that whole matrix and checks each batched result against the equivalent
    loop over the same models, so the claim above is not an assumption made
    from the first model of each propagator.

    What is **not** supported on any propagator, on any backend, is ``vmap``
    over an input that is not a property model: source amplitudes and initial
    wavefields raise ``NotImplementedError``, and batched locations fail
    inside PyTorch's own transform machinery. Batch those over the shot
    dimension the propagators already take instead.

    Batching is also what makes ``max_vel`` necessary. ``max_vel`` has to be
    given when a model that the maximum wave speed is computed from is batched
    by ``torch.func.vmap``: a batched tensor stands for one model per batch
    entry, so it has no single maximum, while ``max_vel`` is one number for the
    whole call. Omitting it then raises an error naming ``max_vel=``. The table
    below is that set, per propagator; batching any other model does not
    require ``max_vel``, and a call that runs today keeps running. Under every
    other ``torch.func`` transform
    (``grad``, ``jacrev``, ``jvp``) the values stay readable, so nothing has to
    be supplied.

    .. list-table:: Models the maximum wave speed is read from
       :header-rows: 1
       :widths: 30 70

       * - Propagator
         - Models
       * - ``scalar``
         - ``v``
       * - ``scalar_born``
         - ``v``
       * - ``acoustic``
         - ``v``
       * - ``acoustic_born``
         - ``v``
       * - ``viscoacoustic``
         - ``v``
       * - ``acoustic_vti``
         - ``c11``, ``c33``, ``buoyancy``
       * - ``elastic``
         - ``lamb``, ``mu``, ``buoyancy``
       * - ``elastic_born``
         - ``lamb``, ``mu``, ``buoyancy``
       * - ``elastic_vti``
         - ``c11``, ``c33``, ``c55``, ``buoyancy``, and ``c66`` in 3-D

    The anisotropic rows read fewer models than they take: ``c13`` does not
    raise the bound these propagators use. The two Born
    propagators in the table reject ``vmap`` over any model outright, as above,
    so their rows say which models the maximum comes from without ``max_vel``
    ever being the thing that stops such a call.

Second-order (create_graph)
    Double-backward: differentiating a gradient again, which is what
    ``torch.autograd.grad(..., create_graph=True)`` asks for. Hessian-vector
    products, Gauss-Newton and truncated-Newton inversion, and second-order
    optimisers all need it. It works on the ``eager`` backend for every
    propagator, because PyTorch's own autograd builds the second-order graph
    from the eager reference implementation's operations generically, with
    nothing for Deepwave to derive.

    The compiled backend has to be told how. Its backward pass is a
    hand-derived adjoint that enters C/CUDA through ``ctypes``, not a graph
    autograd can differentiate again, so double-backward works there only
    where a second-order rule has been derived by hand: ``scalar``,
    ``acoustic`` and ``elastic``. Every other propagator raises on the
    compiled backend rather than returning the zeros that differentiating
    through an opaque call would otherwise produce -- for ``scalar_born``,
    for example (wrapped here; the real message is one line)::

        RuntimeError: Second-order differentiation (double-backward /
        Hessian-vector products) is not implemented for the scalar_born
        propagator. Its hand-derived adjoint has no analytic second-order
        path, so attempting to differentiate through its backward pass (for
        example, calling torch.autograd.grad(..., create_graph=True) and then
        differentiating the resulting gradient) would silently return
        incorrect (typically zero) results rather than the true second-order
        derivative. Deepwave therefore raises this error instead of returning
        a wrong answer. The scalar, acoustic, and elastic propagators do
        support correct second-order differentiation (create_graph=True) on
        the compiled backend. For scalar_born, compute Hessian-vector
        products via finite differences of first-order gradients (perturb the
        model, recompute the gradient, and take the difference). Native
        second-order support may be added in a future release.

    ``backend="eager"`` is the other way out, at the eager backend's usual
    cost in speed. Two of ``elastic``'s options are exceptions within a
    supported propagator: ``mpml_fraction > 0`` and ``surface_method="pm"`` each
    raise a ``NotImplementedError`` for double-backward on the compiled
    backend, both naming ``backend="eager"`` as the alternative.

jvp (forward-mode AD)
    Works generically on every propagator's eager backend (plain
    differentiable PyTorch operations support forward-mode AD without any
    special-casing). On the compiled backend (what ``"auto"``, the default,
    resolves to), only ``scalar``,
    ``acoustic``, and ``elastic`` have a hand-derived Born-based ``jvp``
    rule; the other six compiled propagators raise an actionable
    ``RuntimeError`` pointing at ``backend="eager"`` instead of
    PyTorch's generic, unhelpful "implement the jvp function" error.

Callbacks
    ``forward_callback`` is supported by every propagator (it has no
    dependency on the shared step-based engine that ``backward_callback``
    needs). ``backward_callback`` works on every propagator that has a
    compiled backend -- but, on the *eager* backend specifically, only
    ``scalar`` supports it; every other propagator's eager backend rejects a
    ``backward_callback`` outright. ``get_wavefield(role="forward")`` (the
    reconstructed-forward-wavefield lookup inside a ``backward_callback``,
    for building custom imaging conditions) is narrower still: it is a
    ``scalar``-only capability, on both of ``scalar``'s backends. Every
    other propagator's ``backward_callback`` can only see the adjoint
    wavefield, not the forward one; use a ``forward_callback`` to record the
    forward field instead. The argument that opts into it,
    ``backward_callback_needs_forward_wavefield``, is *accepted* on all
    nine so that a wrapper can forward one uniform keyword set, but only
    at its default of ``False`` where the capability is missing: passing
    ``True`` there raises :exc:`deepwave.DeepwaveNotImplementedError`.
    ``deepwave.spec(propagator).has_forward_wavefield`` reports which
    propagators implement it, without making a call. A ``backward_callback``
    fires on its usual schedule on every backend even when the output
    being differentiated turns out to be unused and autograd has only an
    undefined (identically zero) gradient to propagate: the adjoint
    wavefield it is given is then a tensor of zeros, never ``None``. See
    :class:`deepwave.CallbackState`.

Not supported: outer ``torch.compile``, ``torch.export`` and CUDA graphs
------------------------------------------------------------------------

**Outer** ``torch.compile``, ``torch.export`` **and CUDA graphs are not
supported.** Deepwave's compiled backend enters C/CUDA through ``ctypes``,
which is opaque to Dynamo's tracer, to ``torch.export``'s graph capture and to
CUDA-graph capture. Wrapping a propagator in ``torch.compile`` will graph-break
at the call (at best) and is untested; ``torch.export`` cannot serialise it
(it fails while tracing, on a data-dependent guard); capturing a propagator
call into a CUDA graph is not supported. The ``backend="torch_compile"``
**option** is different and is supported: it compiles Deepwave's own eager
reference implementation (see the Backends section above). If you need a
traceable propagator, use ``backend="eager"`` and expect it to be
substantially slower.

.. include:: _generated/capabilities_table.rst

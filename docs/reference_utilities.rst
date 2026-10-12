Utilities reference
===================

The shared machinery every propagator uses or returns: the output and
callback types, the type aliases, checkpointing, storage, planning and
diagnostics, the discretisation helpers, off-grid source and receiver
locations, wavelets, transducer arrays, misfit functions,
wavefield-space tools, and the parameter converters. :doc:`usage`
explains when to reach for each of them.

.. contents:: :local:

Anisotropy parameter converters
-------------------------------

.. py:module:: deepwave.anisotropy

Each anisotropic propagator takes the parameterisation it actually solves in,
which is not the one models are usually stored or inverted in.
:func:`deepwave.elastic_vti` takes the VTI stiffnesses ``c11``, ``c13``,
``c33``, ``c55`` (and ``c66`` in 3-D) together
with the **buoyancy** :math:`1/\rho`; :func:`deepwave.acoustic_vti` takes the
same stiffnesses less the shear ones (``c11``, ``c13``, ``c33``) and the
buoyancy. Practitioners, by contrast, reason in Thomsen's
:math:`(v_{P0}, v_{S0}, \rho, \epsilon, \delta, \gamma)` parameters, or -- when
the model came from surface P-wave velocity analysis, which cannot resolve
:math:`\epsilon` and :math:`\delta` separately -- in Alkhalifah and Tsvankin's
:math:`(v_{P0}, v_{nmo}, \eta)`. The :mod:`deepwave.anisotropy` module holds
the exact (not weak-anisotropy linearised) conversions between them, plus the
near-source taper :func:`deepwave.acoustic_vti` sometimes needs. Every one of
them is ordinary differentiable tensor arithmetic, so inverting in Thomsen or
Alkhalifah-Tsvankin space is just a matter of building the propagator's inputs
inside the loss and letting autograd chain through (see
:ref:`model-parameterisations` and :doc:`example_vti`).

``thomsen_to_cij`` returns the buoyancy alongside the stiffnesses, so its whole
return value unpacks straight into a propagator call, and a vacuum cell
(:math:`\rho = 0`) becomes ``buoyancy = 0``, the sentinel the propagators
expect, rather than the infinity a plain ``1 / rho`` would give.

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Converter
     - From
     - To
   * - :func:`deepwave.anisotropy.thomsen_to_cij`
     - ``vp``, ``vs``, ``rho``, ``epsilon``, ``delta``, ``gamma``
     - ``c11``, ``c13``, ``c33``, ``c55``, ``c66``, ``buoyancy`` -- the inputs
       of ``elastic_vti`` (and, dropping ``c55``/``c66``, of
       ``acoustic_vti``, for which ``vs = 0``)
   * - :func:`deepwave.anisotropy.cij_to_thomsen`
     - ``c11``, ``c13``, ``c33``, ``c55``, ``c66``, ``buoyancy``
     - ``vp``, ``vs``, ``rho``, ``epsilon``, ``delta``, ``gamma`` -- the exact
       inverse, taking and returning the tuples in the orders that let the two
       compose directly
   * - :func:`deepwave.anisotropy.alkhalifah_to_thomsen`
     - ``vp``, ``vnmo``, ``eta``
     - ``epsilon``, ``delta`` -- feed to ``thomsen_to_cij`` with a chosen
       ``vs``, since P-wave data constrains no shear information
   * - :func:`deepwave.anisotropy.thomsen_to_alkhalifah`
     - ``vp``, ``epsilon``, ``delta``
     - ``vnmo``, ``eta`` -- the inverse, for comparing with or initialising
       from a conventional NMO-velocity model
   * - :func:`deepwave.anisotropy.source_taper`
     - ``c11``, ``c13``, ``c33``, ``source_locations``, ``radius``
     - a modified ``c13`` -- not a parameterisation change, but the
       near-source ellipticity taper that suppresses ``acoustic_vti``'s
       pseudo-shear artefact

The isotropic pair :func:`deepwave.vpvsrho_to_lambmubuoyancy` /
:func:`deepwave.lambmubuoyancy_to_vpvsrho` stays at the package root: it is
not anisotropic, and it shipped at that spelling.

.. autofunction:: deepwave.anisotropy.thomsen_to_cij

.. autofunction:: deepwave.anisotropy.cij_to_thomsen

.. autofunction:: deepwave.anisotropy.alkhalifah_to_thomsen

.. autofunction:: deepwave.anisotropy.thomsen_to_alkhalifah

.. autofunction:: deepwave.anisotropy.source_taper

PropagatorOutput, AuxiliaryState and RunInfo
--------------------------------------------

.. autoclass:: deepwave.PropagatorOutput
    :members:

.. autoclass:: deepwave.AuxiliaryState
    :members:

Every :class:`~deepwave.PropagatorOutput`'s ``.info`` attribute is a
:class:`deepwave.RunInfo` carrying non-differentiable metadata
about how the call actually ran: the user ``dt`` and ``grid_spacing``, the
``accuracy``, the per-face ``boundary``, ``pml_width``, ``pml_freq``,
``mpml_fraction`` and ``surface_method``, the attenuation model's
``q_freq``/``q_n_mechanisms``/``q_band``, the resolved internal
``step_ratio``, the internal ``dt``, the targeted Courant number, ``nt``,
``max_vel``, and the survey ``origin``/``extent``. Together these are the
numerical configuration of the run, so ``out.info`` is enough to set up an
identical one. ``pml_freq`` is the
frequency the run actually used -- the argument you gave, or, if you gave
none, the value inferred from your source amplitudes (see :ref:`pml-freq`);
pass it back in as ``pml_freq=`` to pin a later run to the same absorbing
boundary. It is plain attribute metadata, not a
tuple element, so it never affects iteration, unpacking, ``len()``, or
indexing.

.. autoclass:: deepwave.RunInfo
    :members:

Attenuation response
--------------------

.. py:module:: deepwave.attenuation

.. autofunction:: deepwave.attenuation.response

deepwave.checkpointing
----------------------

.. py:module:: deepwave.checkpointing

.. autofunction:: deepwave.checkpointing.run_segments

.. autofunction:: deepwave.checkpointing.snapshots

.. autoclass:: deepwave.checkpointing.SnapshotResult

CallbackState
-------------

Every accessor below takes an *internal* name -- the kernel's own spelling of
a model or wavefield, not the argument you passed the propagator.
:doc:`callback_names` lists them all, per propagator and per dimensionality,
and distinguishes stable physical fields from provisional internal diagnostics.

.. autoclass:: deepwave.CallbackState
   :members:

MemoryEstimate
--------------

.. autoclass:: deepwave.MemoryEstimate
   :members:

Type aliases
------------

The closed sets of values Deepwave's arguments accept, named so that they can
be used in your own annotations. ``boundary`` entries are deliberately plain
``str`` rather than a ``Literal``; see :doc:`stability` for why.

.. autodata:: deepwave.StorageMode
   :annotation:

.. autodata:: deepwave.WavefieldView
   :annotation:

.. autodata:: deepwave.CallbackRole
   :annotation:

.. autodata:: deepwave.CallbackMode
   :annotation:

.. autodata:: deepwave.CallbackKind
   :annotation:

.. autodata:: deepwave.CallbackDirection
   :annotation:

.. autodata:: deepwave.LocationInterp
   :annotation:

.. autodata:: deepwave.Reduction
   :annotation:

.. autodata:: deepwave.Accuracy
   :annotation:

.. autodata:: deepwave.InitialState
   :annotation:

.. autodata:: deepwave.ON_NODE_TOLERANCE
   :annotation:

boundaries Function
-------------------

.. autofunction:: deepwave.boundaries

Callback
--------

.. autoclass:: deepwave.Callback
   :members: __call__

cosine_taper_end Function
-------------------------

.. autofunction:: deepwave.discretisation.cosine_taper_end

upsample Function
-----------------

.. autofunction:: deepwave.discretisation.upsample

downsample Function
-------------------

.. autofunction:: deepwave.discretisation.downsample

cfl_condition Function
----------------------

.. autofunction:: deepwave.discretisation.cfl_condition

cfl_condition_n Function
------------------------

.. autofunction:: deepwave.discretisation.cfl_condition_n

critical_courant_number Function
---------------------------------

.. autofunction:: deepwave.discretisation.critical_courant_number

suggest_discretisation Function
--------------------------------

.. autofunction:: deepwave.suggest_discretisation

Discretisation
---------------

.. autoclass:: deepwave.Discretisation

DEFAULT_COURANT_NUMBER
----------------------

.. autodata:: deepwave.DEFAULT_COURANT_NUMBER
   :annotation:

check_finite Function
----------------------

.. autofunction:: deepwave.check_finite

estimate_memory Function
-------------------------

.. autofunction:: deepwave.estimate_memory

Storage
-------

Where and how the wavefield history that the model gradient is computed from
is kept. Every propagator accepts the policy as one ``storage=`` object, or as
the individual ``storage_mode``, ``storage_path`` and
``storage_compression`` arguments, which remain supported and are not
deprecated. Passing both spellings in one call raises,
because there would be no honest way to merge them.
``storage_compression=True`` (``Storage.lossy_compression``) is **lossy**: it
changes the model gradient, by about 0.2-4% in 2-D and 0.2-8% in 3-D in
relative L2 error, with the largest errors in the model's outermost cells
(measured on ``scalar_born``'s ``dv`` gradient, normalised by its own
maximum: 0.03-0.07 in 1-D, about 1e-4 in 2-D and about 1e-5 in 3-D at the
edge, against below 1e-3, 2e-5 and 1e-5 in the interior), because an edge
cell's gradient sums large, largely cancelling contributions from the
absorbing layer. Without compression the stored history is lossless; exact
model gradients also require internal-step sampling and fixed numerical
parameters. The
keyword's name does not say any of this, which is why its documentation
does::

    out = deepwave.scalar(
        v, dx, dt, source_amplitudes=s, source_locations=sl,
        storage=deepwave.Storage(mode="disk", path="/scratch/history"),
    )

.. autoclass:: deepwave.Storage

plan Function
-------------

:func:`deepwave.plan` answers "what will this run do, and what will it cost"
without running it. It is not a model of a propagator call: it *is* the call,
stopped before propagation and gradient-history allocation, so
``plan(...).info`` is the very :class:`deepwave.RunInfo` the run would return
and cannot drift from it::

    p = deepwave.plan(
        deepwave.scalar, v, dx, dt,
        source_amplitudes=s, source_locations=sl, receiver_locations=rl,
    )
    print(p.summary())

``Plan.warnings`` contains only warnings emitted by Deepwave itself. It does
not capture or suppress warnings from PyTorch or other dependencies.

.. autofunction:: deepwave.plan

.. autoclass:: deepwave.Plan
   :members: summary

deepwave.diagnostics
--------------------

.. autofunction:: deepwave.diagnostics.report

.. autoclass:: deepwave.diagnostics.Report
   :members:

IGNORE_LOCATION
---------------

.. autodata:: deepwave.IGNORE_LOCATION
   :annotation:

Type aliases
------------

.. autodata:: deepwave.LocationsLike
   :annotation:

.. autodata:: deepwave.SourceSpec
   :annotation:

.. autodata:: deepwave.ReceiverSpec
   :annotation:

Introspection
-------------

The machine-readable description of what a propagator offers, for code that
must drive any of the nine without hard-coding one of them. Each function
accepts the functional (``deepwave.elastic``), the Module class
(``deepwave.Elastic``) or the name (``"elastic"``).

.. autofunction:: deepwave.spec

.. autoclass:: deepwave.PropagatorSpec
   :members:

.. autofunction:: deepwave.wavefield_names

.. autofunction:: deepwave.receiver_names

vpvsrho_to_lambmubuoyancy Function
----------------------------------

.. autofunction:: deepwave.vpvsrho_to_lambmubuoyancy

lambmubuoyancy_to_vpvsrho Function
----------------------------------

.. autofunction:: deepwave.lambmubuoyancy_to_vpvsrho

to_cells Function
-----------------

.. autofunction:: deepwave.to_cells

Hicks Class
-----------

.. autoclass:: deepwave.Hicks
    :members:

ricker Function
---------------

.. py:module:: deepwave.wavelets

.. autofunction:: deepwave.wavelets.ricker

tone_burst Function
-------------------

.. autofunction:: deepwave.wavelets.tone_burst

gaussian_pulse Function
-----------------------

.. autofunction:: deepwave.wavelets.gaussian_pulse

ormsby Function
---------------

.. autofunction:: deepwave.wavelets.ormsby

Transducer array helpers
------------------------

.. py:module:: deepwave.transducers

Stability: provisional (see :doc:`stability`) -- this module is expected to
gain element types, functions and constructor keywords. The unit convention has
settled: positions in grid cells, speeds in cells per unit time, delays and
``dt`` in one time unit of your choosing. What it has today is not respelled
or withdrawn without a release's notice under
:class:`deepwave.DeepwaveFutureWarning`.

The :mod:`deepwave.transducers` module builds extended transducer elements
(lines, arcs, discs, focused bowls) as tessellated point sets with apodisation
weights and focusing delay laws, for ultrasound and NDT modelling. See
:doc:`ultrasound` for an overview and worked usage.

.. autofunction:: deepwave.transducers.line_element

.. autofunction:: deepwave.transducers.arc_element

.. autofunction:: deepwave.transducers.disc_element

.. autofunction:: deepwave.transducers.bowl_element

.. autofunction:: deepwave.transducers.window_weights

.. autofunction:: deepwave.transducers.delays_to_focus

.. autofunction:: deepwave.transducers.apply_delays

.. autofunction:: deepwave.transducers.element_source

.. autofunction:: deepwave.transducers.element_receiver_trace

Misfit functions
-----------------

.. py:module:: deepwave.misfits

The :mod:`deepwave.misfits` module provides three differentiable FWI
data-misfit (loss) functions: :func:`deepwave.misfits.l2` (the conventional
sum of squared differences), and two that are less prone to cycle-skipping
than it, :func:`deepwave.misfits.envelope` (mean squared error between
Hilbert envelopes) and
:func:`deepwave.misfits.normalised_cross_correlation` (one minus the
per-trace normalised cross-correlation, which is invariant to amplitude
scaling). All three share one shape contract -- the time axis is reduced
away to one value per trace, and ``reduction`` (``"mean"``, ``"sum"`` or
``"none"``) then says what to do with that per-trace tensor -- so swapping
one for another really is a one-line change. They do **not** share a scale,
though: :func:`deepwave.misfits.l2` sums the squared differences over time
while :func:`deepwave.misfits.envelope` averages them, and
:func:`deepwave.misfits.normalised_cross_correlation` is bounded in
``[0, 2]`` per trace whatever the amplitudes, so expect to retune the step
length after a swap. See :doc:`example_gsot` for when and why to swap the
misfit function during an inversion.

Two of these functions are commonly reached for as a replacement for
:func:`torch.nn.functional.mse_loss`, and the correspondence is exact in one
reduction only. ``l2(pred, target, reduction="sum")`` equals
``mse_loss(pred, target, reduction="sum")``, but
``l2(pred, target, reduction="mean")`` is ``nt`` times
``mse_loss(pred, target, reduction="mean")`` -- :func:`deepwave.misfits.l2`
sums over time and averages over traces, whereas ``mse_loss`` averages over
every element. Nothing raises, so a swap made on the default reduction
scales the loss and its gradient by ``nt`` (often 1e3 to 1e4); divide the
learning rate by ``nt``, or use ``reduction="sum"`` on both sides.

.. autofunction:: deepwave.misfits.l2

.. autofunction:: deepwave.misfits.envelope

.. autofunction:: deepwave.misfits.normalised_cross_correlation

Wavefield-space tools
---------------------

.. py:currentmodule:: deepwave.wavefields

The :mod:`deepwave.wavefields` module widens Deepwave from a differentiable
*shot-data* operator (point sources in, receiver traces out) to a
differentiable *wavefield* operator. Reach for it when the thing you want to
force, reduce, or observe is the field itself rather than a trace:
:func:`~deepwave.wavefields.source_field_to_sources` and
:func:`~deepwave.wavefields.merge_point_sources` turn a dense forcing field
into ordinary point sources, for wavefield-reconstruction inversion, source
encoding and adjoint-source experiments;
:func:`~deepwave.wavefields.measure` with a
:class:`~deepwave.wavefields.RunningDFT` accumulates a frequency-domain
reduction as a run proceeds, so a frequency-domain objective costs
``O(n_freqs)`` storage instead of ``O(nt)`` and still carries gradients; and
:func:`~deepwave.wavefields.record_boundary` /
:func:`~deepwave.wavefields.reinject_boundary` record traces on a
rectangular surface and replay them as source amplitudes at its cells.
Everything here is pure Python built on the public propagator interface, and
works on any propagator. :doc:`wavefields` covers each in depth, including
the limits of trace replay.

.. autofunction:: deepwave.wavefields.source_field_to_sources
   :no-index:

.. autofunction:: deepwave.wavefields.merge_point_sources
   :no-index:

.. autoclass:: deepwave.wavefields.RunningDFT
   :no-index:

.. autofunction:: deepwave.wavefields.measure
   :no-index:

.. autoclass:: deepwave.wavefields.MeasureResult
   :no-index:

.. autofunction:: deepwave.wavefields.boundary_surface_locations
   :no-index:

.. autofunction:: deepwave.wavefields.record_boundary
   :no-index:

.. autoclass:: deepwave.wavefields.BoundaryRecording
   :no-index:

.. autofunction:: deepwave.wavefields.reinject_boundary
   :no-index:

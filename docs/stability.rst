Stability policy
================

.. note::

   Looking for *numerical* stability -- the CFL condition, a diverging
   simulation, ``NaN`` wavefields? That is :doc:`accuracy` (the CFL condition
   and grid design) and :doc:`troubleshooting` (what to do when a run
   diverges). This page is about interface stability: which names you may
   rely on, and when they may change.

This page says which parts of Deepwave's public interface you may rely on,
and what a later release is allowed to change. It exists because "everything
that is not documented as changeable is frozen by silence" is a promise no
project can keep, and because the alternative -- finding out from a broken
script -- is worse for you than a policy you can read in ten minutes.

``0.1.0`` labels public interfaces **stable** or **provisional**. The core
propagators, their ``nn.Module`` forms, outputs, continuation and established
helpers are stable. Planning, costing, measuring and newer helper interfaces
are provisional. The tables under :ref:`tier-list` give individual labels.
Both tiers are supported and tested. The labels describe how a later release
may change an interface; they do not grade its scientific correctness.

Which half a name is in is stated by its label, in the tables under
:ref:`tier-list`. The rest of this page says what each promise is worth.

.. _the-freeze:

The stable interface
--------------------

Patch releases preserve stable calls and results, apart from necessary
correctness or security fixes, which receive explicit migration notes.
Stable names may gain optional arguments or named output attributes. An
existing stable name may be deprecated in a minor release only with a working
replacement, a warning and a migration example. It remains available through
at least two **subsequent** minor releases; after ``1.0`` removal also
requires a major release. A deprecated name is removed only in a release
announced in its warning and release notes. The aliases deprecated in
``0.1.0`` therefore work in ``0.2`` and ``0.3`` and are removed no earlier
than ``0.4``; see :ref:`deprecation-rule`.

For a fixed propagator and number of dimensions, the current tuple fields,
their order and the opaque auxiliary entry are part of the stable contract.
Future metadata is added as attributes, not tuple entries. Correctness fixes
may still require a documented change of meaning or layout. Read receiver
fields by name and continue with ``initial_state=out``.

Two consequences are worth stating separately, because they are the ones that
would otherwise look like exceptions:

* **An exception may become a more specific subclass** of what it already
  raises. A caller that catches the builtin type, or
  :exc:`deepwave.DeepwaveError`, is unaffected; see
  :ref:`errors-and-warnings`.
* **The warning categories listed in** :ref:`errors-and-warnings` **keep
  their meaning and their builtin base**, and none of them is removed or
  moved to a different cause. A genuinely new cause may add a further
  subclass of :class:`deepwave.DeepwaveWarning`, which every filter on that
  base -- or on ``UserWarning`` -- already catches. Message *text* is not
  part of the interface: it is written for a human and is reworded whenever a
  clearer wording is found.
* A ``NamedTuple`` **result's fields are tuple elements**, so where Deepwave
  returns one -- :class:`deepwave.Discretisation`,
  :class:`deepwave.checkpointing.SnapshotResult`,
  :class:`deepwave.wavefields.MeasureResult`,
  :class:`deepwave.wavefields.BoundaryRecording` -- its field set and their
  order follow the stable compatibility policy. Future metadata is a property,
  not a field, so that unpacking it by position keeps working.

Deepwave is pre-1.0 and follows `Semantic Versioning
<https://semver.org/>`_'s guidance for initial development. These stability
commitments are narrower than the changes that versioning scheme permits.

.. _numerical-standard:

The numerical standard
----------------------

Numerical changes are checked on matching calls, devices and pinned thread
counts, including receivers, final states and gradients. The v0.0.27 parity
harness uses a **separate float64 reference for each version**. For a
``float32`` quantity its comparison threshold is twice the larger of the two
versions' measured float32-to-float64 differences. Comparing both versions
against one common reference would make this bound vacuous by the triangle
inequality. Float64 comparisons use a scoped relative threshold and measured
run-to-run spread. The harness also rejects non-finite data and restricts
documented intentional changes to their affected quantities and measured
magnitude.

These thresholds are validation criteria for the tested configurations, not
a universal error bound for ill-conditioned calls. Agreement across versions
or precisions alone does not establish correct physics. Independent analytic,
convergence, adjoint and reciprocity checks remain necessary. The conditions
under which a *gradient of a particular call* is exact are described in
:doc:`what_deepwave_calculates`.

**Bitwise equality of results between two Deepwave versions is never
promised**, and neither is bitwise equality between two devices, two builds
or two backends. Bitwise *reproducibility of repeated runs of one build*, on
one device, at one configuration, is a separate and narrower promise, stated
with its exact conditions in :doc:`what_deepwave_calculates`.

.. _maturity-labels:

The maturity labels
-------------------

Every public name carries one of four labels. ``stable`` and
``experimental`` make the same compatibility commitment; the latter also
signals that the physics or design has less use outside the project.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Label
     - What it tells you
   * - **stable**
     - The interface follows :ref:`the-freeze`.
   * - **experimental**
     - The same interface commitment as ``stable``; the label records its
       shorter history of external use. Supported cases still require
       scientific validation.
   * - **provisional**
     - Supported and tested, but its interface may evolve in a minor
       release with explicit notes and a deprecation path where feasible.
       Scientific meaning never changes silently.
   * - **deprecated**
     - It remains available through the stated transition period. Calling
       it warns, naming its replacement and the earliest removal release.

The ``stable``, ``experimental`` and ``provisional`` labels do not themselves
emit warnings. In particular, no propagator warns merely because it is
experimental. **Deprecated calls do warn**, naming their replacement and
earliest removal release. If a provisional interface later changes, its
transition follows :ref:`the-provisional-rule`; stable and experimental
interfaces follow :ref:`deprecation-rule`.

.. _the-provisional-rule:

The provisional rule
--------------------

A provisional interface may change in a minor release. Its changelog entry
must describe the old and new forms and how to migrate. Keep the old form
working with a :class:`deepwave.DeepwaveFutureWarning` and a named
replacement where feasible. If preserving it is impractical, explain why in
the release notes before removal. A provisional label never permits an
unannounced change of scientific meaning.

A name's label is stated in its own documentation: every propagator's
docstring carries a one-line ``Stability:`` entry, and the
:doc:`capability matrix <capabilities>` has a ``Status`` column read from the
one declaration that records it. The column is generated from that
declaration, and a test checks every propagator's and every ``nn.Module``
form's docstring line against it, so the two cannot disagree.

.. _deprecation-rule:

The deprecation rule
--------------------

Stable deprecations require a replacement, a warning, a migration example
and at least two subsequent minor releases retaining the old form. Removal
after ``1.0`` also requires a major release. The names deprecated in
``0.1.0`` remain in ``0.2`` and ``0.3``; ``0.4.0`` is their earliest
removal release. A later release may extend the period, with the warning and
release notes updated before the previously announced date. Provisional
deprecations follow :ref:`the-provisional-rule`.

Every deprecation:

* keeps working, with its old behaviour, for the whole deprecation period;
* emits exactly one :class:`deepwave.DeepwaveFutureWarning` per call per
  deprecated name, naming its replacement;
* states its earliest removal version in its docstring and warning.

There is one category for every deprecation, so one line hides (or, in CI,
escalates) all of them::

    warnings.filterwarnings("error", category=deepwave.DeepwaveFutureWarning)

:class:`deepwave.DeepwaveFutureWarning` subclasses ``FutureWarning`` rather
than ``DeprecationWarning`` deliberately. Python's default filters hide
``DeprecationWarning`` outside ``__main__``, and several of Deepwave's
deprecations are emitted from code that something *else* calls -- PyTorch's
``load_state_dict`` machinery, or a ``Module.forward`` reached through
``torch.nn.Module.__call__`` -- so there is no ``stacklevel`` that attributes
them to your own call on every path, and a ``DeprecationWarning`` attributed
to ``torch`` or to Deepwave itself would never be seen. ``FutureWarning`` is
shown whatever module it ends up attributed to. A filter that already targets
``FutureWarning`` covers Deepwave's deprecations too.

The table below lists deprecations present in ``0.1.0``. New stable
deprecations must meet the same requirements; they are not excluded by this
list. The migration warnings for differences from ``v0.0.27`` are listed in
:doc:`upgrading`; their removal is announced separately from alias removal.

Deprecated today
----------------

Every name in the ``deprecated`` tier, with its earliest removal release:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Name
     - Earliest removal
   * - ``scalar_born``'s ``scatter=`` argument
     - ``0.4.0``
   * - ``ScalarBorn``'s ``scatter=`` and ``scatter_requires_grad=``
       constructor arguments
     - ``0.4.0``
   * - ``ScalarBorn.scatter`` property
     - ``0.4.0``
   * - The ``scatter`` key in a ``ScalarBorn`` ``state_dict`` (remapped to
       ``dv`` on load)
     - ``0.4.0``
   * - ``CallbackState.get_model("scatter")`` and
       ``CallbackState.get_gradient("scatter")``
     - ``0.4.0``
   * - ``wavelets.ricker``'s ``length=`` argument
     - ``0.4.0``
   * - ``python_backend=`` (the four propagators shipped in ``v0.0.27``)
     - ``0.4.0``
   * - ``"p"`` as a *source* channel name on ``acoustic`` and ``elastic``:
       the ``sources=`` key and ``source_amplitudes_p`` /
       ``source_locations_p`` keywords. Use ``"volume_rate"`` on
       ``acoustic`` and ``"pressure_rate"`` on ``elastic``. The ``"p"``
       receiver channel remains valid
     - ``0.4.0``
   * - ``scalar_born``'s bare ``receiver_locations=`` argument, the spelling
       ``v0.0.27`` shipped. Use ``sc_receiver_locations=``, which matches
       the order of the returned tuple and the other two Born propagators
     - ``0.4.0``
   * - The five module spellings ``v0.0.27`` exposed that are now private:
       ``deepwave.common``, ``deepwave.location_interpolation``,
       ``deepwave.backend_utils``, ``deepwave.regular_grid`` and
       ``deepwave.staggered_grid``. Each is a shim whose every attribute
       access warns and names the canonical spelling
     - ``0.4.0``

Each of these keeps working, unchanged, through ``0.3`` and emits
a warning naming its replacement (``dv`` for ``scatter``,
``dv_requires_grad`` for ``scatter_requires_grad``, ``backend`` for
``python_backend``, ``nt`` for ``length``, and the private module for each
shim).

Removed in ``0.1.0``
--------------------

Names that only ever existed on the development branch were removed in
``0.1.0`` rather than deprecated: no released code can be using them, so
there is nothing to give a deprecation period to. Each one **raises** rather
than being silently ignored:

* ``backend="torch_jit"``, ``backend="jit"`` and ``backend="compile"``.
  Deepwave no longer calls ``torch.jit`` at all. ``v0.0.27``'s
  ``python_backend="jit"`` runs the eager backend instead, as does
  ``python_backend=True`` on a build where ``v0.0.27`` would have chosen
  ``torch.jit``; both keep working, under the existing ``python_backend``
  deprecation warning, and the difference in numbers is within the standard
  above.
* ``role="backward"`` on :class:`deepwave.CallbackState`, whose spelling is
  ``role="adjoint"``.
* ``dt_internal`` on the returned object, which is ``out.info.internal_dt``.
* the PML attributes on :class:`deepwave.PropagatorOutput`, which are
  ``out.info.pml_width`` and ``out.info.pml_freq``.
* ``staggered=`` in ``deepwave.discretisation``.
* the absorbing-boundary and attenuation-memory field names as
  ``initial_state`` keys; pass ``initial_state=out`` instead.
* ``scalar_born``'s bare ``receiver_mask=`` and ``receiver_amplitudes=``,
  whose spellings are ``bg_receiver_mask=`` / ``sc_receiver_mask=`` and the
  returned ``bg_receiver_amplitudes`` / ``sc_receiver_amplitudes``. Only the
  bare ``receiver_locations=`` shipped, and that one is deprecated rather
  than removed.
* the ``""`` channel key, which was a second spelling of ``"u"``.

:doc:`upgrading` lists them with the exact text each one prints.

.. _tier-list:

Every public name, and what may change
--------------------------------------

The label of every name exported from the ``deepwave`` namespace. Stable
and experimental names follow :ref:`the-freeze`; provisional names follow
:ref:`the-provisional-rule`. Descriptions of likely additions do not rule
out the documented deprecation or correctness processes above.

The propagators, both the function and its ``nn.Module`` form:

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Name
     - Label
     - What a later release may change
   * - ``scalar``, ``scalar_born``, ``acoustic``, ``elastic``, and their
       ``nn.Module`` forms ``Scalar``, ``ScalarBorn``, ``Acoustic``,
       ``Elastic``
     - stable
     - Released in ``v0.0.27``. May gain optional keywords and named output
       attributes and, on the ``nn.Module`` form, further methods and
       properties. Signature, channel names, output field names and output
       tuple layout follow the stable compatibility policy.
   * - ``acoustic_born``, ``acoustic_vti``, ``viscoacoustic``,
       ``elastic_born``, ``elastic_vti``,
       and their ``nn.Module`` forms ``AcousticBorn``, ``AcousticVti``,
       ``Viscoacoustic``, ``ElasticBorn``, ``ElasticVti``
     - experimental
     - Never released before ``0.1.0``. The same as the row above: the label
       records that ``0.1.0`` is their first exposure to real use, not a
       weaker promise. Each is expected to be relabelled ``stable`` in a
       later release, which changes nothing about how it behaves.

Their arguments and returns, which are shared:

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Name
     - Label
     - What a later release may change
   * - :class:`deepwave.PropagatorOutput`
     - stable
     - May gain named attributes. Its tuple layout is fixed on every
       propagator (see :ref:`output-layout`), and its existing field names
       and their meanings are fixed.
   * - :class:`deepwave.AuxiliaryState`
     - stable
     - Continuing a run by handing this object back is supported. It may gain
       methods, recorded run parameters and further tensor names.
       Its internal tensor keys are provisional diagnostic names, listed in
       :doc:`callback_names`; they do not freeze an absorbing-boundary
       implementation. Pass the bundle back whole. A change of layout or
       numerical interpretation increments ``format_version``; see
       :ref:`auxiliary-state`.
   * - The field set of :class:`deepwave.RunInfo`
     - stable
     - May gain fields as more resolved configuration becomes worth
       reporting. Existing fields follow the stable compatibility policy.
   * - :class:`deepwave.Storage`
     - provisional
     - May gain fields. The individual ``storage_mode`` / ``storage_path`` /
       ``storage_compression`` arguments it is an alternative spelling
       for are **stable** and are not deprecated, and so is ``storage=``
       itself when what you pass it is one of the plain mode strings. It is
       the object, and its field set, that is provisional.
   * - ``checkpoint_segments=`` on the propagators
     - provisional
     - The argument's name and its division of labour with
       :func:`deepwave.checkpointing.run_segments` have had no external use
       yet, so a second spelling may be added alongside it. What it computes
       is not provisional at all: see :ref:`segmented-runs`.
   * - ``backward_callback_needs_forward_wavefield=`` on the propagators
     - provisional
     - Exposed only by ``scalar`` today. A later release may implement it on
       other propagators. The name is provisional
       because a backend that could reconstruct the forward wavefield
       without a second stored history would leave nothing to opt into.
       Branch on :attr:`deepwave.CallbackState.has_forward_wavefield`
       rather than on the argument.
   * - ``elastic(surface_method="pm")``
     - provisional
     - The parameter-modified free-surface method is new and its accuracy
       characteristics are still being mapped. ``surface_method="ivm"`` (the
       default) is stable. The value ``"pm"`` keeps working and keeps its
       meaning.
   * - The Python execution modes of ``backend`` (``"eager"``,
       ``"torch_compile"``)
     - provisional
     - Which Python execution modes exist, and what each one supports, may
       grow, because it follows PyTorch's own moving parts. The compiled
       C/CUDA backend is the production path and is stable.
       ``backend="auto"`` selects it except for the documented elastic-Born
       M-PML fallback; it does not search for an implementation of every
       option combination. ``backend="compiled"`` requires native execution.
       ``out.info.backend`` reports which backend ran; see :doc:`backends`.

The helper modules and functions:

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Name
     - Label
     - What a later release may change
   * - :func:`deepwave.plan` and the :class:`deepwave.Plan` it returns
     - provisional
     - ``Plan`` may gain fields and ``Plan.summary()`` may be reworded, as
       more of a run's resolved configuration becomes worth reporting. That
       a call can be resolved and costed without running it, and the
       existing fields (``info``, ``memory``, ``resampling``, ``warnings``,
       ``checkpoint_segments``), do not change.
   * - :func:`deepwave.estimate_memory` and the
       :class:`deepwave.MemoryEstimate` it returns
     - provisional
     - The estimate may gain keys, and the definition of a key may be made
       more precise as a defect fix. That the function exists and reports a
       total does not change.
   * - :func:`deepwave.suggest_discretisation`, the
       :class:`deepwave.Discretisation` it returns, and
       ``deepwave.discretisation.critical_courant_number``
     - provisional
     - The function may gain optional keywords; ``Discretisation`` is a
       ``NamedTuple``, so its fields are fixed. The numbers it returns are
       advice built from an explicit heuristic -- a relative cost-per-cell
       weight picks the accuracy order, and a points-per-wavelength floor is
       applied over the dispersion table -- so read them as a starting point
       and check them against your own problem. Like any other result they
       change only as a defect fix. That a grid spacing and time step can be
       suggested from a velocity and a frequency does not change.
   * - :mod:`deepwave.checkpointing` (``run_segments``, ``snapshots``,
       ``SnapshotResult``)
     - provisional
     - May gain functions and optional keywords. ``SnapshotResult``'s fields
       are fixed, as a ``NamedTuple``'s are. The exactness promise in
       :ref:`segmented-runs` is not provisional.
   * - :mod:`deepwave.wavefields` (``measure``, ``record_boundary``,
       ``reinject_boundary``, ``boundary_surface_locations``,
       ``merge_point_sources``, ``source_field_to_sources``,
       ``RunningDFT``, ``MeasureResult``, ``BoundaryRecording``)
     - provisional
     - The same: a new module whose helper set is expected to grow. The
       fields of ``MeasureResult`` and ``BoundaryRecording`` are fixed, as a
       ``NamedTuple``'s are.
   * - :mod:`deepwave.attenuation`
     - provisional
     - Frequency-domain constitutive response helpers. Their functions may
       gain optional arguments; the documented storage/loss sign convention
       and quality-factor definition remain fixed.
   * - :mod:`deepwave.transducers`
     - provisional
     - May gain element types and functions, and the element constructors may
       gain keywords. The unit convention has
       settled: positions in grid cells, speeds in cells per unit time,
       delays and ``dt`` in one time unit of your choosing.
   * - ``deepwave.diagnostics``
     - provisional
     - :func:`deepwave.diagnostics.report` reports build and CPU detail
       whose vocabulary follows the compiled backend, so the ``Report`` it
       returns may gain fields. That a report exists and can be pasted into
       a bug report does not change.
   * - :func:`deepwave.spec`, :class:`deepwave.PropagatorSpec` and the
       ``deepwave.introspection`` module they live in
     - provisional
     - The spec may gain fields and methods as more of a propagator's
       interface becomes worth publishing. Existing fields and methods, and
       what they return for a given propagator, follow the provisional rule.
       :meth:`deepwave.PropagatorSpec.channel_aliases`
       is reserved and empty on all nine propagators: every channel has
       exactly one spelling today, and any entry added later would be
       documented.
   * - ``Hicks(differentiable=True)``
     - provisional
     - Differentiable source and receiver *positions* are new; the keyword
       and its meaning are fixed, and the numbers it produces are covered by
       :ref:`numerical-standard`. The rest of :class:`deepwave.Hicks` is
       stable.
   * - :class:`deepwave.Hicks`, :func:`deepwave.to_cells`,
       :data:`deepwave.IGNORE_LOCATION`,
       :data:`deepwave.ON_NODE_TOLERANCE`
     - stable
     - May gain optional keywords.
   * - :mod:`deepwave.wavelets`, ``deepwave.discretisation``'s
       ``v0.0.27``-era arithmetic -- ``cfl_condition``, ``cfl_condition_n``,
       ``upsample``, ``downsample``, ``cosine_taper_end`` and
       :data:`deepwave.DEFAULT_COURANT_NUMBER` --
       :func:`deepwave.vpvsrho_to_lambmubuoyancy`,
       :func:`deepwave.lambmubuoyancy_to_vpvsrho`,
       :func:`deepwave.receiver_names`, :func:`deepwave.wavefield_names`
     - stable
     - May gain functions and optional keywords. These are the helpers a
       ``v0.0.27`` script already calls, or that name a propagator's own
       outputs, so they follow the stable compatibility policy.
   * - :mod:`deepwave.misfits`, :mod:`deepwave.anisotropy`,
       :mod:`deepwave.das`, :mod:`deepwave.time_dispersion`,
       :func:`deepwave.boundaries`, :func:`deepwave.check_finite`
     - provisional
     - May gain functions and optional keywords. Each arrived with
       ``0.1.0``: a set of misfit functions, of anisotropy conversions, of
       DAS helpers, a dispersion correction, a boundary-condition builder
       and a finiteness check, none of which has met a user yet. What they
       compute is not in question; which functions there are, and how each
       spells its arguments, is the part ``0.1.0`` is not willing to freeze
       sight unseen.
   * - :class:`deepwave.Callback`, :class:`deepwave.CallbackState` and the
       callback names
     - stable
     - May gain methods, and may accept further names. The public argument
       and state names accepted by ``get_model``, ``get_wavefield`` and
       ``get_gradient``, the per-propagator internal names they also accept,
       and which of those are writable, are fixed; see
       :doc:`callback_names`.
   * - :exc:`deepwave.DeepwaveError` and its subclasses
       :exc:`deepwave.DeepwaveValueError`,
       :exc:`deepwave.DeepwaveTypeError`,
       :exc:`deepwave.DeepwaveRuntimeError`,
       :exc:`deepwave.DeepwaveNotImplementedError`,
       :exc:`deepwave.DeepwaveFloatingPointError` and
       :exc:`deepwave.DeepwaveOSError`; :class:`deepwave.DeepwaveWarning` and
       its subclasses :class:`deepwave.ZeroWidthFaceWarning`,
       :class:`deepwave.MigrationWarning` and
       :class:`deepwave.PmlFreqInferenceWarning`;
       :class:`deepwave.DeepwaveFutureWarning`
     - stable
     - See :ref:`errors-and-warnings`: each of these categories keeps its
       meaning and its builtin base, a new cause may add a further subclass
       of the base a filter already catches, an exception may become a more
       specific subclass of the same builtin base, and message text is not
       part of the interface.
   * - The closed-set type aliases (:data:`deepwave.StorageMode`,
       :data:`deepwave.WavefieldView`, :data:`deepwave.CallbackRole`,
       :data:`deepwave.CallbackDirection`, :data:`deepwave.CallbackKind`,
       :data:`deepwave.CallbackMode`, :data:`deepwave.LocationInterp`,
       :data:`deepwave.Reduction`, :data:`deepwave.Accuracy`)
     - stable
     - May be *widened* -- gaining an accepted value -- which is additive.
       Never narrowed. See :ref:`type-annotations`.
   * - The structural type aliases (:data:`deepwave.LocationsLike`,
       ``deepwave.StorageLike``, ``deepwave.StoragePath``,
       :data:`deepwave.InitialState`, :data:`deepwave.SourceSpec`,
       :data:`deepwave.ReceiverSpec`)
     - stable
     - May gain a member of the union, which is additive: an argument
       annotated with one keeps accepting everything it accepts today.
   * - ``deepwave.__version__``
     - stable
     - Its value, obviously; its existence and its format do not change.

Supported aliases
-----------------

Some arguments and fields have two supported spellings. Neither spelling is
deprecated in ``0.1.0``. Any later stable deprecation must follow
:ref:`deprecation-rule`.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Spelling
     - Aliases
   * - ``storage=deepwave.Storage(mode=..., path=...,
       lossy_compression=...)``, the canonical spelling of the storage
       policy
     - The individual ``storage_mode=`` / ``storage_path=`` /
       ``storage_compression=`` keywords, the spellings ``v0.0.27`` shipped,
       which remain supported. Passing ``storage=`` together with any of them
       raises, because there is no honest way to merge two policies. Note
       that ``storage_compression=True`` (``Storage.lossy_compression``) is
       **lossy**: it changes the model gradient by about 0.2-4% in 2-D and
       0.2-8% in 3-D, which the short keyword's name does not say, so its
       documentation does.
   * - The flat ``source_amplitudes_*`` / ``source_locations_*`` /
       ``receiver_locations_*`` keywords
     - The channel-keyed ``sources=`` / ``receivers=`` /
       ``bg_receivers=`` mappings, which are the canonical way to specify a
       survey. Giving one channel in both forms raises. Both forms are
       supported, and a propagator added later has both: one rule generates
       every flat keyword from the propagator's own channel and state names
       (see :ref:`flat-keywords`), so there is nothing per-propagator to
       learn. On a Born propagator ``sources=`` names the channels of the one
       source set the call takes, and ``receivers=`` fills the **scattered**
       receiver outputs (``sc_receiver_amplitudes``), leaving the background
       ones empty unless ``bg_receivers=`` asks for them too.

.. _flat-keywords:

The flat keyword rule
---------------------

Every flat keyword on every propagator is generated by one rule. With
``<ch>`` a channel name and ``<s>`` a state field name, a propagator takes:

* ``source_amplitudes_<ch>``, ``source_locations_<ch>``,
  ``source_mask_<ch>``;
* ``receiver_locations_<ch>``, ``receiver_mask_<ch>``;
* ``<s>_0``, and ``<s>_m1`` where the scheme has a previous step.

The ``_<ch>`` suffix is dropped for the scalar channel ``u``, which is why
``scalar`` and ``scalar_born`` spell theirs ``source_amplitudes=``. On a Born
propagator the receiver keywords are ``bg_receiver_*`` and ``sc_receiver_*``,
and the scattered state adds ``<s>_sc_0`` (and ``<s>_sc_m1``).
:func:`deepwave.spec` reports a propagator's channel and state names, so the
keywords can be derived rather than looked up.

There are exactly two departures from the rule, both deprecated spellings
rather than exemptions, and both listed under "Deprecated today" above:
``source_{amplitudes,locations}_p`` on the two released propagators whose scalar
source channel is ``"volume_rate"`` or ``"pressure_rate"``, and
``scalar_born``'s bare ``receiver_locations``. The table of departures is
closed. A test asserts that every flat keyword on all nine propagators is
either derived by the rule or one of those two.

.. _output-layout:

The returned tuple
------------------

Every propagator returns a :class:`deepwave.PropagatorOutput`: a real
``tuple`` subclass whose entries are also reachable by name.

For a fixed propagator and dimensionality, the current tuple fields and
their order follow the stable compatibility policy. Future metadata is added
as a named attribute, not a new tuple element. For example::

    *_, receiver_amplitudes = scalar(...)
    out[-1]                       # current last receiver field for scalar
    out.receiver_amplitudes       # the supported way to read a result

**Accessing outputs by name is the recommended habit.** It is the form that
is correct for both 2-D and 3-D, because the
number of state and receiver elements depends on the number of spatial
dimensions, so a fixed positive index means different things in 2-D and 3-D
even though neither layout ever changes. And the elements in the middle of
the tuple are not the ones a reader would guess: on ``scalar``, ``out[-2]``
is ``out.auxiliary_state``, not a receiver field, whereas on ``scalar_born``
it is the background data and on ``acoustic``/``elastic`` in 2-D it is the
y-component receiver amplitudes. Index by name and the question does not
arise.

``out.names`` lists the field names of the instance you hold, and
:func:`deepwave.spec` reports them for a propagator and a number of
dimensions without running anything.

Pytrees
~~~~~~~

:class:`deepwave.PropagatorOutput` and :class:`deepwave.AuxiliaryState` are
registered as ``torch.utils._pytree`` nodes, which is what lets
``torch.func`` transforms, ``torch.compile`` and ``tree_map`` see the tensors
inside them instead of one opaque object:

* an output's **children** are its tuple elements, in return order, with the
  auxiliary state expanding into the tensors it holds, so the leaves of an
  output are exactly its tensors and the flattened order is the return order
  above. Its **context** is the field names together with ``info``: the names
  belong to the instance rather than to the class, so an output cannot be
  rebuilt without them.
* an auxiliary state's children are its tensors, and its context is their
  names and every recorded run parameter, including ``format_version``, so a
  state that has been through a transform still continues a run.

A ``treespec`` containing either type **is not serialisable** to JSON:
neither type registers a serialised name or a JSON format for its run
metadata. Keyed pytree registration is supported by the declared minimum
PyTorch version (2.4). Object serialisation through ``pickle`` and
``torch.save``/``torch.load`` is separate from tree-specification serialisation.
:doc:`design/propagator_output` records why the context is shaped this way.

.. _auxiliary-state:

Continuing a run: the auxiliary state
-------------------------------------

Every propagator returns a :class:`deepwave.AuxiliaryState` as
``out.auxiliary_state``. What is promised:

* that it exists on all nine, as one element of the returned tuple;
* that it is a read-only mapping from name to tensor (``names``, ``keys``,
  ``values``, ``items``, ``as_dict``) with the documented helpers (``to``,
  ``detach``, ``compatible_with``) and the recorded run parameters it
  checks a continuation against;
* that handing it back unchanged, as part of ``initial_state=out``,
  continues the simulation;
* that it carries an integer ``format_version`` describing its layout and
  numerical interpretation, and an incompatible version is rejected with an
  error naming both versions;
* that its diagnostic keys are documented in :doc:`callback_names` for the
  installed version. These internal keys follow the provisional policy, not
  the stable public wavefield-name contract.

A saved auxiliary state is a restart for a compatible implementation, not a
long-term interchange format. Keep the package version and run configuration
with saved states. A necessary numerical correction may require a new state
format, including in a patch release.

What is **not** promised is the bundle as a whole: it is **opaque as an output
element**. Do not index it positionally, and do not build on how many tensors
it holds, on their order or on their shapes -- those follow the absorbing
boundary's and the attenuation model's own implementation, which is state
rather than physics. Pass the object back whole.

.. _segmented-runs:

Segmented runs
--------------

Running a call as several time segments -- ``checkpoint_segments=`` on a
propagator, or :func:`deepwave.checkpointing.run_segments` -- is a memory
lever, not a different computation:

* **forward values agree with the equivalent unsegmented run to rounding**,
  with the same resolved setup, backend and storage policy. Even when final
  wavefields match bitwise, receiver resampling can round differently because
  a concatenated trace and the direct call's trace have different memory
  layouts;
* **gradients agree to rounding** under the same conditions and sampling
  phase, because segmentation changes the accumulation order. This preserves
  the unsegmented call's gradient approximation, including any time sampling
  or lossy storage; it does not make that approximation an exact derivative.

These equivalence promises are not provisional. An unconditional bitwise
guarantee is not made. :doc:`design/checkpointing` explains whole-record
resampling and what a caller driving its own outer loop has to do instead.
Segmentation accepts the supported point-source mappings and flat
keywords. Continue a state with ``initial_state=out``; individual initial-field
keywords and propagator callbacks are not supported by
``checkpoint_segments=``. Use ``run_segments``'s
``segment_callback`` for observations at segment boundaries. Unsupported
combinations raise before propagation.

A checkpointed segmented run uses re-entrant ``torch.utils.checkpoint`` by
default, which retains less native wavefield history. Pass
``use_reentrant=False`` to :func:`deepwave.checkpointing.run_segments` when
``torch.autograd.grad`` is needed; that mode can retain a history buffer per
segment and use substantially more memory.

If you wrap a propagator in ``torch.utils.checkpoint`` yourself, pass
``use_reentrant=False``. Under the re-entrant mode the return degrades to a
plain tuple -- losing ``out.info`` and named access -- and the tensors in
``out.auxiliary_state`` come back detached; that is a property of
``torch.utils.checkpoint`` rather than of Deepwave, and :doc:`usage` covers it
in full. Deepwave's own segmented runs return a complete
:class:`deepwave.PropagatorOutput`, ``out.info`` included, either way.

.. _torch-func:

``torch.func`` and ``torch.compile``
------------------------------------

Backend agreement for gradients depends on the model-gradient sampling stride,
storage and callbacks. At effective stride one with lossless storage, fixed
inferred parameters and no callback mutation, compiled and eager gradients are
checked against :ref:`numerical-standard`, subject to any narrower
per-propagator approximation. At a larger effective stride, compiled model
gradients use sampled correlations while eager/``torch_compile`` differentiate
every internal step, so their model gradients can differ materially. Source
and initial-state gradients still use every internal step; see
:ref:`gradient-guarantees` and :doc:`backends`.

``backend="auto"`` selects native C/CUDA execution except for the documented
elastic-Born M-PML fallback to eager. It does not negotiate transform support
or retry failed calls. Select ``backend="eager"`` explicitly when the
native implementation lacks the transform or boundary combination you need;
see :doc:`backends` and :doc:`capabilities`. ``out.info.backend`` records
which backend ran. ``backend="compiled"`` requires native execution.

``max_vel`` is the one argument a transform can make mandatory. Deepwave
infers it, detached, from the model values, which are unreadable when
``torch.func.vmap`` batches a model: a batched tensor stands for one model
per batch entry and has no single maximum. So:

* if no model argument the maximum wave speed is computed from is batched at
  any ``vmap`` level, ``max_vel`` is inferred as usual and nothing has to be
  passed;
* if such a model is batched, the call raises, naming ``max_vel=``. Pass the
  maximum absolute wave speed over every model the call stands for.

The :doc:`capability matrix <capabilities>` lists, per propagator, the models
the maximum wave speed is read from. Batching any other model does not require
``max_vel``, and a call that runs today keeps running.

What a propagator added later looks like
----------------------------------------

A propagator added in a future release is not free-form: it follows the same
layouts every existing one does, so that what you learn from one propagator
transfers to the next.

**Its output tuple** holds, in order:

1. the physical state fields;
2. ``auxiliary_state``;
3. the Born scattered state fields, on a Born propagator;
4. the receiver amplitudes -- background before scattered, on a Born
   propagator.

Channels are declared in one order throughout, both for the state fields and
for the receivers: the scalar channel (``p`` or ``u``) first, then ``z``,
``y``, ``x``, then any others in state-field order.

**Its models** are one named tensor each, one per entry of the propagator's
model-name list:

* the required models that are not perturbations come first, as ordinary
  positional-or-keyword arguments, followed by ``grid_spacing`` and ``dt``;
* Born perturbations, and models that are optional or that exist only in
  some numbers of dimensions, are keyword-only;
* no argument takes several models stacked into one tensor.

The anisotropic propagators have enough model arguments that position is easy
to get wrong, so the documentation and the examples call them by keyword;
both forms work.

**Its flat keywords** come from the rule above, with no new exemption.

Boundary conditions
-------------------

``boundary=`` names each face's condition. A single string applies to all
faces; a sequence gives one code per face, in the same order as
``pml_width``. The accepted codes today are ``"pml"``, ``"free"``,
``"rigid"`` and ``"periodic"``; an unrecognised one raises, listing the
accepted names.

**In** ``0.1.0`` **the promise attached to a face code is the same on every
propagator and at every accuracy order: the face either meets Deepwave's
energy and spectral checks for that condition, or the combination raises**
:exc:`deepwave.DeepwaveNotImplementedError` **there**, rather than returning a
plausible wrong answer. A later release that makes a raising combination work
is an additive change. ``boundary=`` has never appeared in a release, so none
of this changes a ``v0.0.27`` result. :doc:`boundaries` documents each
condition, its measured reflection coefficient and its accuracy caveats, which
are separate from those checks.

What that leaves accepted in ``0.1.0``:

* ``"pml"`` on every face of every propagator, at every accuracy, at any
  width including zero;
* every code on ``scalar`` and ``scalar_born``;
* ``"rigid"`` on every face of ``acoustic``, ``acoustic_born``,
  ``viscoacoustic`` and ``acoustic_vti``;
* ``"free"`` on those three isotropic acoustic propagators at the beginning
  of an axis at every accuracy, and on either face at ``accuracy=2``;
* ``"rigid"`` on ``elastic``, ``elastic_vti`` and ``elastic_born`` at the beginning of an axis at every
  accuracy, and on either face at ``accuracy=2``.

Everything else -- ``"periodic"`` anywhere but the two regular-grid
propagators, ``"free"`` on the elastic family or ``acoustic_vti``, and a
reflecting face at the end of an axis at ``accuracy`` 4 or more where the
list above does not allow it -- raises. Model an elastic or ``acoustic_vti``
free surface with a vacuum layer instead, and put a reflecting face at the
beginning of its axis; :ref:`boundary-faces-accepted` has the table and the
alternatives.

A zero-width ``"pml"`` face is the one case that needs a word here, because
``v0.0.27`` had it and said nothing. It behaves identically whether
``boundary=`` was omitted, given as the single string ``"pml"`` or given per
face -- it is ``v0.0.27``'s ``pml_width=0`` idiom -- and it never raises,
because ``"pml"`` is the spelled-out default. It is exactly a Dirichlet
(zero-exterior) truncation, which on ``scalar`` acts as a free surface and on
``elastic`` as a frictionless wall. On all nine it warns
:class:`deepwave.ZeroWidthFaceWarning`, which is also the way to silence it;
where a ``boundary=`` code means the same condition, the warning names that
code, and naming the face explicitly is silent -- and, at accuracy above 2,
uses the exact image rather than this approximation of it.

.. _errors-and-warnings:

Errors and warnings
-------------------

Deepwave raises its own exception types (:class:`deepwave.DeepwaveError` and
the mixed-in subclasses documented in :doc:`usage`). Each one also inherits
the builtin exception type that the same failure raised before, so
``except ValueError`` keeps catching what it caught; the addition is that
``except deepwave.DeepwaveError`` now catches everything Deepwave raises
deliberately, and nothing else. The *builtin* base a given failure raises is
frozen: changing it is a break and would need a major version. Which
``Deepwave*Error`` subclass it is may become more specific within that base at
any time, which is not a break for a caller that catches either the builtin
type or :exc:`deepwave.DeepwaveError`. A location error names the propagator
and the argument you passed, so that a message about coordinates says which
of your tensors they came from.

Warning *categories* are part of the interface. There is one per *cause*, and
each of the ones below keeps its meaning and its builtin base: none is
removed, renamed or moved to a different cause.
:class:`deepwave.DeepwaveWarning` (a ``UserWarning``) is the base of the
categories that describe what a call will compute:

* :class:`deepwave.ZeroWidthFaceWarning`
* :class:`deepwave.MigrationWarning`, with its
  :class:`deepwave.PmlFreqInferenceWarning` subclass

:class:`deepwave.DeepwaveFutureWarning` (a ``FutureWarning``) is separate,
because a deprecated *spelling* is a message about the code you wrote rather
than about the physics you asked for; filtering one does not silence the
other. You may equally filter on ``UserWarning`` or ``FutureWarning``, the
two builtin bases. Every warning in these two families names its own category
in its own text, as the way to filter it. A cause Deepwave has not met yet may
add a further subclass of :class:`deepwave.DeepwaveWarning`, which a filter on
that base -- or on ``UserWarning`` -- already catches; the categories above
are not reshuffled to make room for one.

Deepwave also still emits some plain ``UserWarning``\ s that predate these
categories and name no category of their own -- the one about a ``max_vel``
below the model's own maximum, and the grid-cells-per-wavelength advice, for
example. Filter those on ``UserWarning``. No Deepwave deprecation uses
``DeprecationWarning``: every one of them is a
:class:`deepwave.DeepwaveFutureWarning`, so there is a single category to
filter rather than a choice that could change between releases.

Warning *message text* is not part of the interface: it is written for a human
to read and is reworded whenever a clearer wording is found. Do not match on
it.

.. _type-annotations:

Type annotations
----------------

The closed sets of values an argument accepts are published as type aliases
(:data:`deepwave.StorageMode`, :data:`deepwave.WavefieldView`,
:data:`deepwave.CallbackRole`, :data:`deepwave.CallbackDirection`,
:data:`deepwave.CallbackKind`, :data:`deepwave.CallbackMode`,
:data:`deepwave.LocationInterp`, :data:`deepwave.Reduction`,
:data:`deepwave.Accuracy`), so that a type checker can reject a typo before
the call runs. Each one mirrors a runtime check, and a test asserts that the
two agree, so an alias cannot quietly drift from what the code accepts.

Widening one of these aliases -- adding a value -- is *not* a breaking
change: an argument that accepts more than it did still accepts everything it
did. Narrowing one follows the stable deprecation or correctness-fix policy
above.

``accuracy`` is the one alias likely to be noticed: it is
``Literal[2, 4, 6, 8]``, so a computed ``int`` is a type error even though it
still runs. Silence it with a cast::

    from typing import cast
    out = deepwave.scalar(v, dx, dt, accuracy=cast(deepwave.Accuracy, order))

``boundary`` entries remain plain ``str`` so configuration strings can be
passed without a cast. The runtime check reports accepted names and rejects
unsupported combinations. This trades static typo checking for convenience;
widening a ``Literal`` argument would not itself break callers using its
previous values. No new boundary name is accepted until it is implemented
and documented.

Reproducibility
---------------

Two promises are made about a single build, and :doc:`what_deepwave_calculates`
states both with their exact scope and the reasons they stop where they do.
That page is the scope: nothing wider is promised here.

* Repeated runs of one build of the compiled backend, on one device and at
  one call configuration, give bitwise-reproducible gradients.
* Compiled-backend CPU results -- forward outputs and gradients alike -- are
  invariant to ``torch.get_num_threads()``, on the nine propagators that have
  a compiled CPU backend.

Neither says anything about two devices, two builds, two backends or two
releases: they are properties of one build rather than relationships between
releases, which is what :ref:`numerical-standard` covers.

Pre-announced changes
---------------------

Changes that are not deprecations -- no name is going away -- but that would
alter what an existing call returns in a future release. They would be
announced here a release ahead, so that the change is never the first you hear
of it.

**There are none at present.** A future correction with a user-visible
effect belongs in the changelog and migration notes; this section can
announce it earlier when useful.

In particular, ``mode="write"`` is the current stable default of
:meth:`deepwave.CallbackState.get_wavefield`,
:meth:`deepwave.CallbackState.get_model` and
:meth:`deepwave.CallbackState.get_gradient`. All three keep returning the
tensor Deepwave itself holds, with the autograd edge it carries, and writing
into an internal one in place is a supported feature rather than a tolerated
accident. Forward callbacks must not mutate models or wavefields during a
differentiable run: that would invalidate the discrete adjoint, so Deepwave
rejects it. A backward callback that edits an internal gradient accumulator
deliberately changes the returned gradient; that value is a user-defined
backward result, not necessarily the derivative of the unchanged forward
loss. :meth:`deepwave.CallbackState.writable` is how to ask whether a
given name may be written where you are: with ``kind="model"`` it is ``True``
for an internal model name when mutation is supported. It is ``False`` during
a differentiable forward callback and for a name merely
derived from the internal state, such as ``acoustic``'s ``v`` and ``rho``,
where there is nothing behind the name to write into.

Reserved names
--------------

Names Deepwave has set aside for possible later use. They are listed so that a
release that starts using one is not a surprise, and so that you do not build
something of your own on the assumption that they will stay free:

* ``region=``, on the propagators, for a future spatial-subsetting argument.
  It is not accepted today. The existing ``survey_pad`` / ``origin``
  arguments are stable and are how you restrict propagation to the part of
  the model your survey occupies (see :doc:`usage`).
* ``Grid``, ``Model``, ``State``, ``Output``, ``propagate`` and
  ``Propagator``, in the ``deepwave`` namespace. Deepwave defines none of
  them today.

``0.1.0`` publishes the nine propagator functions and their ``nn.Module``
forms, and nothing above them. Should a later release add a container type or
a single ``propagate()`` entry point, it would be an addition beside those
eighteen, which follow the deprecation policy above, and it would arrive with its own
documentation. No naming grammar for such a thing is published here, because
committing to one would be committing to the design.

Supporting both ``v0.0.27`` and ``0.1.0``
-----------------------------------------

If your code must run against both the last release and this one, detect
features rather than parsing behaviour out of the version string:

.. code-block:: python

    import deepwave

    # Preferred: ask whether the thing you need exists.
    if hasattr(deepwave, "PropagatorOutput"):
        receiver_amplitudes = deepwave.scalar(...).receiver_amplitudes
    else:  # v0.0.27 returned a plain tuple
        receiver_amplitudes = deepwave.scalar(...)[-1]

Where you genuinely need a version comparison (for example to decide whether
to pass an argument that did not exist before), read the version defensively
and compare *parsed* versions rather than strings. Two traps:

* ``deepwave.__version__`` **does not exist in v0.0.27.** It is new in
  ``0.1.0``, so reading it directly is itself an ``AttributeError`` on the
  version you are trying to support. Use ``getattr`` with a fallback, or ask
  the installed distribution's metadata, which works on both.
* **String comparison is not version comparison.** ``"0.0.27" < "0.1.0"``
  happens to be true, but ``"0.0.9" < "0.0.27"`` is false.

.. code-block:: python

    from importlib.metadata import version as _dist_version
    from packaging.version import parse

    import deepwave

    installed = parse(getattr(deepwave, "__version__", None) or
                      _dist_version("deepwave"))
    if installed >= parse("0.1.0"):
        ...

``importlib.metadata.version("deepwave")`` reports what was installed, which
is the more reliable of the two when both are available. In this build,
``deepwave.__version__`` is |version|.

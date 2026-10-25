Upgrading
=========

Deepwave 0.1.0 is the first release since ``v0.0.27``, and it changes a lot.
This page records interface and numerical changes from ``v0.0.27`` and the
available migration for each.

**The two-minute version.** Read receiver results by name and continue a run
with ``initial_state=out``. Most changed spellings raise or warn. Some tuple
indices remain valid Python but now refer to a different field, and an
unrequested receiver tensor has a new empty shape. Audit positional output
uses even if the script runs without a message. Released deprecated aliases
remain available through ``0.3``; see :doc:`stability`.

**The exception**, stated here because it is the only one: the 42
absorbing-boundary auxiliary-field **input keyword arguments**
(``psiy_m1=``, ``zetax_m1=``, ``psiy_sc_m1=``, ``phi_y_0=``, ``psi_x_0=``,
``m_vzz_0=``, ``m_sigmaxxx_0=`` and the rest) are gone outright, with no
alias. Passing one now raises ``TypeError``. The fields themselves are still
there -- carried opaquely in ``out.auxiliary_state`` -- but the whole set now
occupies that one element of the returned tuple instead of one element each,
so the tuple is shorter (:ref:`upgrade-tuple-length`) and code that read a
boundary field by index has to change. The individual names are neither
attributes of the output nor ``initial_state`` keys, and asking for one by
either spelling raises; they remain the keys of ``auxiliary_state``. See
:ref:`upgrade-auxiliary-state-input` for the list and the replacement. So
when your script raises or warns, search this page for the message text.

Five numerical changes from ``v0.0.27`` emit
:class:`deepwave.MigrationWarning`:
:ref:`upgrade-pml-freq` (an omitted ``pml_freq`` is now inferred),
:ref:`upgrade-off-node` (a barely-off-node coordinate is interpolated rather
than truncated), :ref:`upgrade-hicks-stencil` (an off-node interpolation
window is placed correctly), :ref:`upgrade-hicks-dtype` (``Hicks`` weights
follow the default dtype) and :ref:`upgrade-born-edge-scatter`
(``scalar_born`` extends its scattering model into the absorbing layer).
Four of the five name the one line that restores the old numbers; the fifth
has no such line, and says so, naming instead the cells the difference is
confined to. Each is emitted once per cause per process, and all five are
removed in Deepwave 0.4.0 -- they exist to explain a difference against an
older run, not to be lived with. Silence all five at once with
``warnings.filterwarnings("ignore", category=deepwave.MigrationWarning)``.

:doc:`stability` is the authoritative statement of what may change in a
future release, which names are in which stability tier, and how long a
deprecated name survives. This page does not repeat it.

PML damping also has a numerical correction: each absorbing face now uses its
own physical width, rather than the longest width anywhere on the grid. Runs
with unequal face widths or unequal grid spacing may change, even with an
explicit ``pml_freq``. See :doc:`boundaries`; there is no legacy global-width
mode.

Fourier resampling also corrects two edge cases: a positive frequency taper
that rounds to zero bins still removes the even-length Nyquist bin, and
``downsample`` rejects nonempty records whose length is not divisible by its
rate. Pad or trim such records explicitly. These correctness fixes and the
PML-width correction are documented here and in the changelog; they do not
emit one of the five migration warnings above.

.. contents:: On this page
   :local:
   :depth: 2

Every break, in one table
-------------------------

Ordered roughly by how likely you are to meet it. "Loud" means it raises;
"warned" means it runs and tells you; "silent" means you have to read the
row.

Interpreter-generated ``TypeError`` wording can vary with Python version;
for example, Python 3.9 prints ``__init__`` where newer versions include
the class name. The argument restriction and migration are the same.

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - What changed
     - How it shows
     - The one-line fix
   * - :ref:`Every argument after dt is keyword-only <upgrade-keyword-only>`
     - loud
     - Name the arguments: ``scalar(v, dx, dt, source_amplitudes=..., ...)``
   * - :ref:`Module constructors take keywords after grid_spacing
       <upgrade-module-keyword-only>`
     - loud
     - ``Scalar(v, dx, v_requires_grad=True)``
   * - :ref:`The returned tuple is shorter <upgrade-tuple-length>`
     - loud
     - ``out.receiver_amplitudes``, or ``*_, rec = out``
   * - :ref:`The PML fields are no longer input keyword arguments
       <upgrade-auxiliary-state-input>`
     - loud
     - ``initial_state=out`` to continue a simulation
   * - :ref:`The PML fields are not names on the output
       <upgrade-auxiliary-state>`
     - loud
     - ``initial_state=out`` to continue a simulation
   * - :ref:`scalar_born's second positional argument is grid_spacing
       <upgrade-scalar-born-dv>`
     - loud
     - ``scalar_born(v, dx, dt, dv=dv, ...)``
   * - :ref:`pml_freq is inferred when omitted <upgrade-pml-freq>`
     - warned
     - ``pml_freq=25.0`` restores the old PML frequency setting
   * - :ref:`scalar_born extends dv into the absorbing layer
       <upgrade-born-edge-scatter>`
     - warned
     - No one-line fix; only a ``dv`` non-zero on the model's edge is
       affected
   * - :ref:`python_backend= is replaced by backend= <upgrade-backend>`
     - warned
     - ``backend="auto"`` (or ``"eager"``, ``"compiled"``)
   * - :ref:`scatter is renamed dv on scalar_born and ScalarBorn
       <upgrade-scatter-dv>`
     - warned
     - ``dv=``, ``.dv``, ``"dv"``
   * - :ref:`storage_path defaults to the temporary directory
       <upgrade-storage-path>`
     - warned
     - ``storage_path="."`` restores the old location
   * - :ref:`storage_mode="none" with a model gradient raises
       <upgrade-storage-none>`
     - loud
     - ``storage_mode="device"`` (or ``"cpu"``/``"disk"``)
   * - :ref:`model_gradient_sampling_interval must divide nt
       <upgrade-mgsi>`
     - loud
     - Use an ``nt`` that is a multiple of the interval
   * - :ref:`A non-finite, non-Tensor or empty model is rejected
       <upgrade-model-validation>`
     - loud
     - Pass a finite ``float32``/``float64`` ``torch.Tensor``
   * - :ref:`A diverged run raises instead of returning NaNs
       <upgrade-check-output-finite>`
     - loud
     - ``check_output_finite=False`` restores the old behaviour
   * - :ref:`An unsupported model dtype raises DeepwaveTypeError
       <upgrade-model-dtype>`
     - loud
     - ``v = v.float()``; catch ``TypeError``, not ``RuntimeError``
   * - :ref:`Some failures raise ValueError where v0.0.27 raised RuntimeError
       <upgrade-exception-types>`
     - loud
     - Catch ``deepwave.DeepwaveError``, or the builtin base
   * - :ref:`Options a propagator cannot act on raise <upgrade-dead-options>`
     - loud
     - Drop the option, or pass its default
   * - :ref:`deepwave.location_interpolation is a deprecated shim
       <upgrade-location-interpolation>`
     - warned
     - ``deepwave.Hicks``
   * - :ref:`deepwave.common is a deprecated shim <upgrade-common>`
     - warned
     - ``deepwave.discretisation.*`` and the top level
   * - :ref:`cfl_condition_n's max_abs_vel and c_max are renamed
       <upgrade-cfl-condition-n>`
     - loud
     - ``max_vel=``, ``courant_number=``
   * - :ref:`backend_utils, regular_grid and staggered_grid are deprecated
       shims <upgrade-backend-utils>`
     - warned
     - Nothing: none was ever a supported interface
   * - :ref:`CallbackState requires direction and n_steps
       <upgrade-callback-state>`
     - loud
     - ``direction="forward"``, ``n_steps=nt``
   * - :ref:`Hicks takes keywords after halfwidth <upgrade-hicks>`
     - loud
     - Name the arguments
   * - :ref:`Hicks(dtype=) follows the default dtype <upgrade-hicks-dtype>`
     - warned
     - ``dtype=torch.float32`` restores ``v0.0.27``
   * - :ref:`ricker's nt and dt are required again <upgrade-ricker>`
     - loud
     - ``ricker(freq, nt, dt)``
   * - :ref:`Off-node coordinates are interpolated, not truncated
       <upgrade-off-node>`
     - warned
     - ``locations.floor().long()`` for the old behaviour
   * - :ref:`An off-node Hicks stencil is placed correctly
       <upgrade-hicks-stencil>`
     - warned
     - Nothing; the old cells were a bug
   * - :ref:`scalar_born's scattered data is sc_receiver_amplitudes
       <upgrade-sc-receiver>`
     - warned
     - ``out.sc_receiver_amplitudes``, ``sc_receiver_locations=``
   * - :ref:`p as a source channel names its quantity <upgrade-source-p>`
     - warned
     - ``volume_rate`` or ``pressure_rate``, whichever the propagator
       injects
   * - :ref:`PropagatorOutput is not a plain tuple <upgrade-dataparallel>`
     - silent
     - ``isinstance(x, tuple)``, not ``type(x) is tuple``
   * - :ref:`An unrequested receiver channel is shaped like the data
       <upgrade-empty-receivers>`
     - silent
     - Nothing; both shapes hold no elements
   * - :ref:`The compiled CUDA backend requires the default stream
       <upgrade-cuda-stream>`
     - loud
     - Run outside ``torch.cuda.stream()``
   * - :ref:`The compiled CUDA kernels ship a curated architecture list
       <upgrade-cuda-arch>`
     - silent
     - Check the toolkit/device floor and driver PTX support
   * - :ref:`The step_ratio warning fires earlier <upgrade-step-ratio>`
     - warned
     - Reduce ``dt``, or filter the message
   * - :ref:`max_vel is required under vmap, not under every transform
       <upgrade-max-vel-torch-func>`
     - loud
     - ``max_vel=`` when a model the maximum is read from is batched by
       ``vmap``; nothing otherwise

.. _upgrade-five:

The five you are most likely to hit
-----------------------------------

**1. Keyword-only arguments.** Every argument after ``dt`` (after
``grid_spacing`` on the ``Module`` constructors) must be named.

.. code-block:: python

    # v0.0.27
    out = deepwave.scalar(v, dx, dt, source_amplitudes, source_locations,
                          receiver_locations)
    # 0.1.0
    out = deepwave.scalar(v, dx, dt,
                          source_amplitudes=source_amplitudes,
                          source_locations=source_locations,
                          receiver_locations=receiver_locations)

**2. ``scalar_born``'s second positional argument.** It used to be the
scattering model; it is now ``grid_spacing``.

.. code-block:: python

    # v0.0.27
    out = deepwave.scalar_born(v, scatter, dx, dt, ...)
    # 0.1.0
    out = deepwave.scalar_born(v, dx, dt, dv=dv, ...)   # scatter= still works, deprecated

**3. Unpacking the return.** The returned tuple is shorter, because the
absorbing boundary's auxiliary fields are now one opaque element.

.. code-block:: python

    # v0.0.27
    wavefield_0, wavefield_m1, psiy, psix, zetay, zetax, rec = deepwave.scalar(...)
    # 0.1.0
    out = deepwave.scalar(...)
    rec = out.receiver_amplitudes          # preferred
    *_, rec = out                          # also fine, and independent of ndim

**4. Restarting a simulation.** One bundle replaces the individual
wavefield arguments.

.. code-block:: python

    # v0.0.27
    out = deepwave.scalar(..., wavefield_0=out[0], wavefield_m1=out[1],
                               psiy_m1=out[2], psix_m1=out[3],
                               zetay_m1=out[4], zetax_m1=out[5])
    # 0.1.0
    out = deepwave.scalar(..., initial_state=out)

**5. ``Module`` construction.**

.. code-block:: python

    # v0.0.27
    prop = deepwave.Scalar(v, dx, True)
    # 0.1.0
    prop = deepwave.Scalar(v, dx, v_requires_grad=True)

Breaks in detail
----------------

Each entry quotes what Deepwave actually prints. The quoted text is checked
against the live message by ``tests/test_upgrading_docs.py``, so it cannot
drift from the code.

.. _upgrade-keyword-only:

Every argument after ``dt`` is keyword-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: keyword_only_arguments

On all nine supported propagators and their ``Module.forward`` methods, only the
models, ``grid_spacing`` and ``dt`` may be given positionally. This is a
loud, rare break with a one-line fix, and it is what makes it possible to add
and reorder arguments in later releases without silently rebinding anyone's
call.

.. code-block:: text

   TypeError: scalar() takes 3 positional arguments but 6 were given

The count differs per propagator, because each has a different number of
models: ``acoustic() takes 4``, ``elastic() takes 5``, and
``Scalar.forward() takes 2``. On Python 3.9 the message has no class-name
prefix on the ``forward`` variants.

Fix: name the arguments, as in :ref:`upgrade-five` item 1.

.. _upgrade-module-keyword-only:

``Module`` constructors take keywords after ``grid_spacing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: module_constructor_keyword_only

``grid_spacing`` is the last required parameter of every one of the nine
supported constructors.

.. code-block:: text

   TypeError: Scalar.__init__() takes 3 positional arguments but 4 were given

The other two released ``Module`` classes that had a positional flag give
``Acoustic.__init__() takes 4 positional arguments but 5 were given`` and
``Elastic.__init__() takes 5 positional arguments but 6 were given``. On
Python 3.9 the message lacks the ``Scalar.`` prefix.

``ScalarBorn`` gives the same message as ``Scalar``
(``ScalarBorn.__init__() takes 3 positional arguments but 4 were given``) but
for a different reason: its second positional argument used to be the
*scattering model*, and is now ``grid_spacing``. Fix:
``ScalarBorn(v, dx, dv=dv)`` -- the same rename
:ref:`the function form <upgrade-scalar-born-dv>` documents.

Fix: ``Scalar(v, dx, v_requires_grad=True)``. There is also a new
``requires_grad=`` keyword that takes ``True``, ``False`` or a sequence of
model names, and covers every model the module owns.

.. _upgrade-tuple-length:

The returned tuple is shorter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: tuple_index_out_of_range

On ``scalar``, ``scalar_born``, ``acoustic`` and ``elastic``, the absorbing
boundary's per-field auxiliary state occupies **one** element
(``auxiliary_state``) instead of one element each. An index or an unpacking
that reaches into the middle of the tuple now finds a different element, or
raises. A changed positive index can therefore fail **silently**; prefer
named fields:

.. code-block:: text

   IndexError: tuple index out of range

.. upgrade-break: not_enough_values_to_unpack

and an unpacking of the whole tuple raises:

.. code-block:: text

   ValueError: not enough values to unpack (expected 7, got 4)

Old and new lengths:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Propagator
     - 2-D (was → is)
     - 3-D (was → is)
   * - ``scalar``
     - 7 → 4
     - 9 → 4
   * - ``scalar_born``
     - 14 → 7
     - 18 → 7
   * - ``acoustic``
     - 10 → 7
     - 14 → 9
   * - ``elastic``
     - 16 → 9
     - 31 → 14

The two leading wavefields and the trailing receiver fields keep their
positions, so ``*_, rec = out``, ``out[-1]``, and ``out[-2]`` where that was
already a receiver field are all unaffected. Mostly, only code that indexes or
unpacks the *middle* of the tuple breaks.

The exception is ``scalar_born``, whose *scattered* wavefields are physical
fields that moved: ``v0.0.27`` returned them at indices 6 and 7, after the
background wavefields' four PML fields; the branch returns them as
``wavefield_sc_0`` and ``wavefield_sc_m1`` at indices **3 and 4**, because the
four elements between them folded into one ``auxiliary_state``. ``out[6]`` no
longer gives a scattered wavefield: it gives
``bg_receiver_amplitudes``. Read them by name.

Fix: use the field names (``out.receiver_amplitudes``,
``out.auxiliary_state``), or ``initial_state=out`` to continue a simulation
without naming any field. ``print(out)`` names every field it holds, and
``out.names`` returns them as a tuple.

.. _upgrade-auxiliary-state:

The PML fields are not names on the output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: auxiliary_state_attribute

``v0.0.27`` returned a plain tuple, so ``out.psiy_m1`` was never a name a
released Deepwave answered to; the individual absorbing-boundary and
attenuation-memory names (``psiy_m1``, ``zetax_m1``, ``phi_y_0``,
``psi_x_0``, ``m_vzz_0``, ``m_sigmaxzx_0``, ``psiy_sc_m1``, and the rest) are
carried opaquely in ``out.auxiliary_state`` instead. Asking for one by name
says so, on all nine:

.. code-block:: text

   AttributeError: 'PropagatorOutput' has no field 'psiy_m1': it is an
   absorbing-boundary or attenuation-memory auxiliary field, which is carried
   opaquely in out.auxiliary_state rather than as an output attribute. Pass
   initial_state=out (or initial_state=out.final_state) to continue a
   simulation.

.. upgrade-break: auxiliary_state_initial_state_key

``initial_state`` did not exist in ``v0.0.27`` either, so the same names are
not keys of it:

.. code-block:: text

   DeepwaveValueError: scalar: initial_state key 'psiy_m1' names
   absorbing-boundary or attenuation-memory auxiliary fields, which are not
   accepted as initial_state keys. To continue a run, pass the previous output
   as initial_state=out (or initial_state=out.final_state, or its
   auxiliary_state).

Neither spelling ever shipped, which is why both raise rather than warning: a
plain tuple has no attribute names, and ``initial_state`` did not exist. The
fields themselves are unchanged, but they are no longer one tuple element
each -- the whole set is the single ``auxiliary_state`` element -- so the
returned tuple is shorter than ``v0.0.27``'s and its middle holds different
elements. Code that read a boundary field by *index* has to change as well;
:ref:`upgrade-tuple-length` gives the old and new lengths.

Fix: ``initial_state=out`` (or ``initial_state=out.final_state``).

.. _upgrade-auxiliary-state-input:

The PML fields are no longer input keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: auxiliary_state_input_keyword

This is the one place where Deepwave 0.1.0 removes something ``v0.0.27``
shipped, rather than deprecating it. ``v0.0.27`` took the absorbing
boundary's auxiliary fields *back in* as ordinary keyword arguments, which is
how a simulation was continued. Those parameters no longer exist, on any
propagator, and there is no alias:

.. code-block:: text

   TypeError: scalar() got an unexpected keyword argument 'psiy_m1'

The name in the message is whichever one you passed, and the propagator name
is whichever you called. In full, the 42 removed parameters are:

* ``scalar`` (6): ``psiz_m1``, ``psiy_m1``, ``psix_m1``, ``zetaz_m1``,
  ``zetay_m1``, ``zetax_m1``.
* ``scalar_born`` (12): the same six, plus ``psiz_sc_m1``, ``psiy_sc_m1``,
  ``psix_sc_m1``, ``zetaz_sc_m1``, ``zetay_sc_m1``, ``zetax_sc_m1``.
* ``acoustic`` (6): ``phi_z_0``, ``phi_y_0``, ``phi_x_0``, ``psi_z_0``,
  ``psi_y_0``, ``psi_x_0``.
* ``elastic`` (18): ``m_vzz_0``, ``m_vzy_0``, ``m_vzx_0``, ``m_vyz_0``,
  ``m_vxz_0``, ``m_vyy_0``, ``m_vyx_0``, ``m_vxy_0``, ``m_vxx_0``,
  ``m_sigmazzz_0``, ``m_sigmayzy_0``, ``m_sigmaxzx_0``, ``m_sigmayzz_0``,
  ``m_sigmaxzz_0``, ``m_sigmayyy_0``, ``m_sigmaxyy_0``, ``m_sigmaxyx_0``,
  ``m_sigmaxxx_0``.

The *physical* initial wavefields keep their own keyword arguments
(``wavefield_0``, ``wavefield_m1``, ``pressure_0``, ``vz_0``,
``sigmazz_0`` and so on): only the boundary's memory variables were
withdrawn. They were withdrawn rather than deprecated because they are no
longer separable -- the boundary's state is carried as one opaque
:class:`deepwave.AuxiliaryState`, whose contents depend on ``accuracy``,
``pml_width`` and the boundary condition on every face, and a call that
supplied six of them and left the rest to their defaults would be seeding an
inconsistent boundary rather than continuing a simulation. An alias that
accepted them would have to reconstruct a state it cannot check, so it would
be wrong quietly; the ``TypeError`` is wrong loudly.

Fix: pass the previous call's whole output. ``initial_state`` replaces every
one of these arguments at once, and is what
:ref:`upgrade-auxiliary-state` describes reading back.

.. code-block:: python

    # v0.0.27
    out = deepwave.scalar(v, dx, dt, ..., wavefield_0=out[0], wavefield_m1=out[1],
                          psiy_m1=out[2], psix_m1=out[3],
                          zetay_m1=out[4], zetax_m1=out[5])
    # 0.1.0
    out = deepwave.scalar(v, dx, dt, ..., initial_state=out)

``initial_state`` also accepts a mapping, and a mapping *may* still be keyed
by these names -- that form is deprecated rather than removed, and is the
``FutureWarning`` quoted in :ref:`upgrade-auxiliary-state`. It is only the
top-level keyword arguments that are gone.

.. _upgrade-scalar-born-dv:

``scalar_born``'s second positional argument is ``grid_spacing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: model_as_grid_spacing

``scalar_born(v, scatter, dx, dt, ...)`` gives
``TypeError: scalar_born() takes 3 positional arguments but N were given``,
like every other call with too many positional arguments, where ``N`` is
however many positional arguments the call passed -- 4 for the spelling above,
7 for the fully-positional ``v0.0.27`` form
``scalar_born(v, scatter, dx, dt, wav, src, rec)``. The spelling that
would otherwise pass a *model* to ``grid_spacing`` -- ``dt`` given by keyword,
so that only three arguments are positional -- gets a message of its own:

.. code-block:: text

   DeepwaveTypeError: grid_spacing must be a float or a sequence of floats,
   but a torch.Tensor of shape (12, 12) was passed. A grid spacing for a
   2-dimensional model is one value, or one per dimension (2); a Tensor with
   more entries than that is a model, not a spacing. In Deepwave 0.0.27
   scalar_born's second positional argument was the scattering model; it is
   now grid_spacing, and the scattering model is the keyword argument dv
   (formerly scatter). Fix: scalar_born(v, grid_spacing, dt, dv=dv, ...).
   See the Upgrading page of the documentation.

The same check runs on every propagator (a ``Tensor`` with more entries than
there are spatial dimensions is never a valid grid spacing), naming that
propagator instead. A ``grid_spacing`` that genuinely is a ``Tensor`` -- a
0-dimensional one, or one holding a value per dimension -- is unaffected.

Fix: ``scalar_born(v, grid_spacing, dt, dv=dv, ...)``.

.. _upgrade-pml-freq:

``pml_freq`` is inferred when omitted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: pml_freq_inference

``v0.0.27`` used ``25.0`` Hz for every call that omitted ``pml_freq``, whatever
the data. Deepwave now takes it from the peak of the source amplitudes'
spectrum. **This changes numbers** for any call that omitted the argument and
whose dominant frequency is not exactly 25 Hz, which is why it warns:

.. code-block:: text

   PmlFreqInferenceWarning: pml_freq was not specified, so it was resolved to

The rest of the message names the resolved value and how it was inferred (the
peak of the source amplitudes' energy spectrum, or a fifth of the Nyquist
frequency -- saying whether that is because the call has no sources at all or
because its sources held no dominant frequency), says that ``v0.0.27``
used ``25.0`` here, gives both remedies, and names the category to filter on.
It is reported against your own call, whichever form it took: a propagator
function, a ``Module``, ``checkpoint_segments=`` or
:func:`deepwave.checkpointing.run_segments`.
``deepwave.PmlFreqInferenceWarning`` is a ``deepwave.MigrationWarning``,
which is a ``deepwave.DeepwaveWarning``, which is a ``UserWarning``, so a
filter at any of those four levels silences it.

**There is no quiet band.** The warning fires whenever the inferred value is
anything other than exactly ``25.0``, because any other value builds a
different absorbing boundary. A nearby value is not a harmless one: measured
on a 60x60 model, ``dx=4``, ``dt=0.004``, ``nt=400``, a 15 Hz Ricker source
(inferred ``pml_freq`` 15.0) against the same call with ``pml_freq=25.0``, the
receiver amplitudes move by 0.3% in relative L2 and 0.04% at the peak, and the
**returned wavefields move by 75% in relative L2** -- because a returned
wavefield covers the padded domain, and most of that difference is inside the
PML region itself, where the two calls built different profiles. A regression
test that compares receiver gathers may still pass on such a call; one that
compares returned wavefields will not.

The classic 25 Hz Ricker call infers ``25.0``, so the frequency-setting
migration warning stays silent. This establishes the PML frequency only;
other numerical changes can still affect a result.

Fix: ``pml_freq=25.0`` restores ``v0.0.27``'s PML frequency setting;
``pml_freq=<your dominant frequency>`` says what you meant. Either silences
the warning, which is a migration aid and is removed in ``0.4.0``. **Pin it
explicitly on every call of a continued simulation** -- see below.

``pml_freq`` inference and ``initial_state``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inference is per call, from *that call's* source amplitudes. A simulation run
in segments -- the ``initial_state=out`` recipe in
:ref:`upgrade-auxiliary-state-input` -- would therefore have a hazard
``v0.0.27``, with its fixed 25.0, did not: a later segment whose sources are
quieter, differently shaped, or absent would infer a different ``pml_freq``
and continue against an absorbing boundary that its inherited memory
variables were not built for.

Nothing is inferred on such a call. The state records the ``pml_freq`` and
the ``max_vel`` the run that produced it resolved, and a continuation that
omits either takes the recorded value rather than deriving a new one, so a
segmented run builds one absorbing boundary throughout without being told to.
Passing either explicitly still overrides that, and a value that disagrees
with the recorded one warns::

   UserWarning: This auxiliary_state was produced with pml_freq=13.6 but this
   call resolved pml_freq=25.0. The absorbing boundary's profiles therefore
   differ from the ones its memory variables were built with. Pass
   pml_freq=13.6 to continue with the same boundary.

Like the other resolution warnings this fires at most once per process, so a
loop over shots or segments reports the first mismatch only.

With the internal time step the same in both halves, a two-segment run
reproduces the forward values of a single unsegmented run of the same length
exactly; gradients agree to rounding. ``accuracy`` and ``pml_width`` are the
two settings a continuation must still repeat, because their defaults (``4``
and ``20``) cannot be told apart from values passed by hand: a continuation
that leaves out an ``accuracy`` or a ``pml_width`` the first call gave is
refused, naming both values, rather than continuing on a different grid.

.. _upgrade-born-edge-scatter:

``scalar_born`` extends ``dv`` into the absorbing layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: born_edge_scatter

Every model a propagator is given is extended outwards to fill the absorbing
layer and the finite-difference halo around the region being simulated: the
outermost cell of the region is repeated across the padding. ``v0.0.27``
made one exception, ``scalar_born``'s scattering model, which it filled with
zeros instead. The scattered wavefield was then not the linearisation of
``scalar`` for a ``dv`` that was non-zero on the edge of the region: the
perturbed and unperturbed velocity models differ inside the padding, and
nothing in the scattered field accounted for it. ``dv`` is now extended like
everything else, which is what :func:`deepwave.acoustic_born` and
:func:`deepwave.elastic_born` always did with their own perturbations.

**This changes numbers** only for a call whose ``dv`` is non-zero in the
outermost cells of the region, and only through those cells' extension into
the padding. Such a call says so:

.. code-block:: text

   MigrationWarning: dv is non-zero in the outermost cells of the model
   region, and those values are now extended outwards into the padding
   around it

The rest of the message says what ``v0.0.27`` did instead, that a ``dv``
which is zero in those cells computes what it always did, and names the
category to filter on. A ``dv`` that is zero there -- which it is in a
typical migration image, where the edge of the model is not where the
reflectivity lives -- is unaffected and says nothing.

Fix: there is no argument that restores the old numbers, because the old
behaviour was the scattering model disagreeing with the background one. To
reproduce a ``v0.0.27`` result, enlarge the model (or the region
``survey_pad`` extracts) so that ``dv`` is zero in its outermost cells; the
two then agree. The warning is a migration aid and is removed in ``0.4.0``.

.. _upgrade-backend:

``python_backend=`` is replaced by ``backend=``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: python_backend

.. code-block:: text

   DeepwaveFutureWarning: python_backend is deprecated and will be removed in
   Deepwave 0.4.0; use backend. Silence this with
   warnings.filterwarnings('ignore',
   category=deepwave.DeepwaveFutureWarning).

``backend`` is a ``str`` taking ``"auto"`` (the default), ``"compiled"``,
``"eager"`` or ``"torch_compile"``. ``python_backend`` survives on the four
propagators shipped in ``v0.0.27`` and is removed in 0.4.0. New propagators
accept only ``backend``.

It is mapped to what ``v0.0.27`` *did*, not to what the argument's name
suggests: ``python_backend=True`` selected ``torch.compile`` when the
installed compiled library was built with OpenMP and ``torch.jit.script``
otherwise, and ``False`` selects the compiled backend. ``v0.0.27`` never ran
the plain eager loop for ``python_backend=True``.

Deepwave no longer calls ``torch.jit`` at all, so the two values that chose
TorchScript -- ``python_backend="jit"``, and ``python_backend=True`` on a
build without OpenMP -- now run the **eager** backend. Both still work and
still warn. The step functions are the same arithmetic either way, so the
only change is floating-point rounding: results move at the level of the last
bits, as they do between any two backends (see
:doc:`what_deepwave_calculates`). There is no ``backend`` value for
TorchScript: use ``"eager"`` or ``"torch_compile"``.

Fix: replace ``python_backend=`` with ``backend=``, mapping ``None``/``False``
to ``"auto"``, ``"jit"`` to ``"eager"``. ``out.info.backend`` reports what
``"auto"`` resolved to, which
-- unlike a warning -- works inside a training loop and in a distributed
worker.

.. _upgrade-scatter-dv:

``scatter`` is renamed ``dv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: scatter_keyword

The scattering potential is ``dv`` everywhere: ``scalar_born``'s keyword,
``ScalarBorn``'s constructor argument and attribute, and the
``CallbackState`` model/gradient key. All three released spellings survive as
deprecated aliases:

.. code-block:: text

   DeepwaveFutureWarning: scatter is deprecated and will be removed in Deepwave
   0.4.0; use dv. Silence this with warnings.filterwarnings('ignore',
   category=deepwave.DeepwaveFutureWarning).

.. upgrade-break: scatter_state_dict

The ``state_dict`` key changes with them. An old checkpoint loads, remapped:

.. code-block:: text

   DeepwaveFutureWarning: The state_dict key 'scatter' is deprecated and will
   be removed in Deepwave 0.4.0; use 'dv'. It has been loaded as 'dv'; re-save
   the checkpoint to silence this.

(:class:`deepwave.DeepwaveFutureWarning` subclasses ``FutureWarning``, which
matters here: PyTorch calls the loading hook, so a ``DeprecationWarning``
would be attributed to ``torch`` and hidden by Python's default filters.)

Fix: ``dv=``/``dv_requires_grad=``, read ``.dv``, and use ``"dv"`` as the
callback model and gradient name. Re-save any checkpoint.

.. _upgrade-storage-path:

``storage_path`` defaults to the temporary directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: storage_path_default

``v0.0.27``'s ``storage_path`` defaulted to the literal string ``"."``, so a
``storage_mode="disk"`` run wrote its wavefield history into the current
working directory. The default is now ``None``, meaning the system temporary
directory, resolved **when you call** rather than at import -- so ``TMPDIR``
set after ``import deepwave`` is honoured. The parameter also accepts
:class:`os.PathLike` (a :class:`pathlib.Path`, say) as well as ``str``, which
``v0.0.27`` did not.

On many systems the temporary directory is a RAM-backed ``tmpfs`` far smaller
than the disk, so a disk run using the default now warns once:

.. code-block:: text

   UserWarning: storage_mode="disk" is writing to the default directory

The rest of the message names the resolved directory, its free space and the
size of the history the run is about to write there.

Fix: ``storage_path="."`` if you relied on files landing in the working
directory, or ``storage_path=`` a real filesystem for large runs. Reading
``module.storage_path`` on a ``Module`` that was not given one now returns
``None`` ("resolved at call time") rather than ``v0.0.27``'s ``"."``.

.. _upgrade-storage-none:

``storage_mode="none"`` with a model gradient raises
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: storage_mode_none_gradient

.. code-block:: text

   RuntimeError: scalar storage_mode="none" retains no forward wavefield
   history, so it cannot compute a model gradient; it is a forward-only
   (no-grad) low-memory mode.

That mode keeps no forward wavefield history, so the imaging correlation that
forms the model gradient cannot be computed. ``v0.0.27``, and this branch
until now, returned a plausible-looking but entirely wrong model gradient
instead -- a spike at the source locations, 92% to about 1300% off depending
on the case.

Fix: use ``storage_mode="device"``, ``"cpu"`` or ``"disk"`` for a model
gradient. Forward-only ``"none"`` is unaffected, and gradients with respect to
the source amplitudes and the initial wavefields -- which never needed the
forward history, and were always correct -- remain available in ``"none"``
mode.

.. _upgrade-mgsi:

``model_gradient_sampling_interval`` must divide ``nt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: mgsi_divides_nt

.. code-block:: text

   DeepwaveValueError: model_gradient_sampling_interval (3) must divide the
   number of time steps (8), but leaves a remainder of 2.

``v0.0.27`` silently shortened the propagation by ``nt % interval`` steps
instead: the trailing receiver samples came back as exactly ``0.0`` and the
returned wavefields were the state at an earlier time. The message names both
nearest valid ``nt`` values.

A run under :class:`torch.no_grad` is not affected even if its model is an
``nn.Parameter`` with ``requires_grad=True``: no gradient is recorded, so
there is nothing for the interval to sample.

When **no** model requires a gradient, the interval is reset to ``1``
rather than truncating a forward-only run for a gradient that is never
computed, so such runs return the same numbers for every interval. Callback
observations now use user time steps regardless of whether model gradients
are active.

Fix: use an ``nt`` that is a multiple of the interval, or an interval that
divides ``nt``.

.. _upgrade-model-validation:

A non-finite, non-``Tensor`` or empty model is rejected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: non_finite_model

.. code-block:: text

   DeepwaveValueError: v contains 144 non-finite values (NaN or Inf) out of
   144 elements; scalar: material requires finite v.

.. upgrade-break: list_model

.. code-block:: text

   TypeError: scalar: v must be a torch.Tensor, but got list. Convert a
   Python list with torch.tensor(...) or a NumPy array with
   torch.from_numpy(...) first.

.. upgrade-break: empty_model

.. code-block:: text

   ValueError: scalar: v must not be empty, but has shape (0, 0).

These are new, more informative rejections of inputs that were already
invalid or already produced a wrong answer later in the call. A
``source_amplitudes`` with a transposed shape now names the expected
``[shot, source, time]`` shape rather than a misleading batch/time-sample
mismatch. An explicit ``max_vel`` below the model's conservative speed bound
now raises before stepping; previously the comparison was unreachable for an
explicit value and an unstable run could start. The staggered families now
include density-weighted stencil or constitutive bounds, because high-order
stencils can couple stiff cells to distant low-density velocity nodes. This
can increase automatic subdivision even at constant physical wavespeed.
The inferred bound also sets PML damping, so automatic elastic results can
change even when the internal time step remains unchanged. To compare the
same discretisation across versions, pass the same conservative ``max_vel``
and explicit ``pml_freq`` to both. This pins their numerical settings; it
does not reproduce the older automatically inferred bound. The v0.0.27
parity harness pins a shared bound for its elastic comparisons and therefore
checks matched settings rather than equality of these automatic defaults.

Fix: pass a real ``torch.Tensor``, with no ``NaN``/``Inf``, and a ``max_vel``
that covers the material/operator bound. Inspect ``deepwave.plan`` with
automatic inference to obtain that bound before pinning a numerical setup.

.. _upgrade-check-output-finite:

A diverged run raises instead of returning ``NaN``\ s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: non_finite_output

.. code-block:: text

   DeepwaveFloatingPointError: deepwave.scalar produced non-finite output:
   'wavefield_0' contains 2704 non-finite values (nt=60, dt=0.001 for the
   propagation that produced it, which under checkpoint_segments is one
   segment rather than the whole run). This usually means the simulation
   diverged. Check the CFL condition
   (deepwave.discretisation.cfl_condition), that the model contains no zero,
   negative or non-finite values, and that any initial wavefield and the
   source amplitudes are finite. Pass check_output_finite=False to disable
   this check.

Every propagator now checks its final wavefields and receiver amplitudes
before returning them. In ``v0.0.27`` a diverged simulation ran to
completion and handed back ``NaN``\ s, which then flowed into the loss, the
gradients and the next FWI iteration, so the first symptom was usually an
all-``NaN`` loss many iterations later, far from its cause. The check costs
one device synchronisation per call and nothing per element; see
:doc:`troubleshooting` for the measured overhead.

Fix: nothing, if your runs were finite -- the numbers are bit-identical with
the check on and off. If you were relying on getting the ``NaN``\ s back (to
inspect them, or because a later stage of your own code detects them), pass
``check_output_finite=False``.

.. _upgrade-model-dtype:

An unsupported model dtype raises ``DeepwaveTypeError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: model_dtype

.. code-block:: text

   DeepwaveTypeError: Model tensor 0 (of 1) has dtype torch.int64, but
   Deepwave's propagators require torch.float32 or torch.float64

In ``v0.0.27`` an integer model instead reached the source-amplitude dtype
check and raised ``RuntimeError: Inconsistent dtype: Expected source
amplitudes to have datatype torch.int64, ...``, which blamed the wrong
tensor. :class:`deepwave.DeepwaveTypeError` is a ``TypeError``, not a
``RuntimeError``, so ``except RuntimeError`` around a propagator call stops
catching this case.

Fix: cast the model to ``float32``/``float64`` before the call, and catch
:class:`deepwave.DeepwaveError` (or ``TypeError``) if you relied on the old
exception type. ``torch.autocast`` is not supported and is a no-op for
Deepwave; the message says so.

.. _upgrade-exception-types:

Some failures raise ``ValueError`` where ``v0.0.27`` raised ``RuntimeError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A negative ``nt``, and an empty or non-positive ``model_shape``, now raise
``ValueError``. More generally, Deepwave 0.1.0 raises its own exception types
(:class:`deepwave.DeepwaveError` and its subclasses); each also inherits the
builtin type that the same failure raised *before*, wherever that was
sensible, so ``except ValueError`` and ``except TypeError`` keep working. The
one place the builtin base genuinely changed is
:ref:`the model dtype check <upgrade-model-dtype>`.

Fix: catch :class:`deepwave.DeepwaveError` to catch everything Deepwave
raises deliberately, or keep catching the builtin base.

.. _upgrade-dead-options:

Options a propagator cannot act on are absent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: meaningless_option

``mpml_fraction``, ``surface_method``, ``q_freq``, ``q_n_mechanisms`` and
``q_band`` appear only on propagators that implement them. Passing one to a
different propagator raises ``TypeError``, even at its default value:

.. code-block:: text

   TypeError: viscoacoustic() got an unexpected keyword argument 'mpml_fraction'

Fix: pass the option only to a propagator that implements it. For a generic
wrapper, check ``deepwave.spec(name).supported_options`` before forwarding
physics settings.

.. _upgrade-location-interpolation:

``deepwave.location_interpolation`` is a deprecated shim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: location_interpolation_shim

``v0.0.27`` exposed this module, and ``Hicks`` and ``DEFAULT_EPS`` through it.
The implementation is now private, because ``location_interpolation`` is also
the name of the propagators' own keyword argument (a string mode) and the two
must not be confused. The module spelling keeps working until ``0.4.0``:

.. code-block:: text

   DeepwaveFutureWarning: deepwave.location_interpolation.Hicks is deprecated
   and will be removed in Deepwave 0.4.0; use deepwave.Hicks.

Fix: :class:`deepwave.Hicks`, and :data:`deepwave.ON_NODE_TOLERANCE` for
``DEFAULT_EPS``.

.. _upgrade-common:

``deepwave.common`` is a deprecated shim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: common_shim

Ten names were reachable as ``deepwave.common.*`` in ``v0.0.27``
(``cfl_condition``, ``cfl_condition_n``, ``cosine_taper_end``, ``upsample``,
``downsample``, ``Callback``, ``CallbackState``,
``vpvsrho_to_lambmubuoyancy``, ``lambmubuoyancy_to_vpvsrho`` and
``IGNORE_LOCATION``). Each still resolves, and warns:

.. code-block:: text

   DeepwaveFutureWarning: deepwave.common.cfl_condition is deprecated and will
   be removed in Deepwave 0.4.0; use deepwave.discretisation.cfl_condition.

``v0.0.27``'s module had no ``__all__`` and 68 names that do not start with an
underscore; the shim is an explicit *practical subset*, not a reproduction of
everything that module happened to expose. One name is deliberately absent:
``deepwave.common.StorageMode`` shipped as an ``IntEnum``, and
:data:`deepwave.StorageMode` is now a string ``Literal`` -- a different kind of
object, so an alias would resolve to something that is not a drop-in. The
``AttributeError`` routes you to it and says what changed.

Fix: import from ``deepwave`` or ``deepwave.discretisation``, as the warning
says.

.. _upgrade-cfl-condition-n:

``cfl_condition_n``'s ``max_abs_vel`` and ``c_max`` are renamed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: cfl_condition_n_signature

The shim above carries the *name* ``cfl_condition_n`` unchanged, but two of
its keyword arguments were renamed, so a call that named them fails whether
it goes through ``deepwave.common`` or through
:func:`deepwave.discretisation.cfl_condition_n`:

.. code-block:: text

   TypeError: cfl_condition_n() got an unexpected keyword argument 'max_abs_vel'

``max_abs_vel`` is now ``max_vel``, matching the propagators' own
``max_vel``; and ``c_max`` is now ``courant_number``, matching
:func:`deepwave.discretisation.critical_courant_number`. ``courant_number``
defaults to ``None``, which resolves to the same ``0.6`` that ``c_max``
defaulted to, so a call that used the defaults is unaffected -- and one that
named either argument raises rather than silently doing something else.
Positional calls are unaffected: the order of the parameters did not change.

.. code-block:: python

    # v0.0.27
    dt_inner, step_ratio = cfl_condition_n(grid_spacing, dt, max_abs_vel=v.max(),
                                           c_max=0.6)
    # 0.1.0
    dt_inner, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel=v.max(),
                                           courant_number=0.6)

Fix: rename the two keywords, or drop them and pass the velocity
positionally.

.. _upgrade-callback-state:

``CallbackState`` requires ``direction`` and ``n_steps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: callback_state_required_arguments

:class:`deepwave.CallbackState` is what a ``forward_callback`` or
``backward_callback`` is handed; Deepwave constructs it, and almost nobody
else needs to. Code that *does* construct one -- a test double, say -- must
now supply two more keyword-only arguments:

.. code-block:: text

   TypeError: CallbackState.__init__() missing 2 required keyword-only
   arguments: 'direction' and 'n_steps'

That is the message when the construction already supplies the seven
``v0.0.27`` positional arguments. A construction missing those as well names
them instead: ``TypeError: CallbackState.__init__() missing 7 required
positional arguments: 'dt', 'step', 'wavefields', 'models', 'gradients',
'fd_pad', and 'pml_width'``.

``direction`` is ``"forward"`` or ``"backward"`` and says which pass the
callback is in; ``n_steps`` is the total number of steps that pass will take,
so that a callback can report progress or decide how often to act without
having to be told separately. Both are required rather than defaulted,
because there is no value for either that is right more often than it is
wrong. The class is also reachable as ``deepwave.common.CallbackState``,
through the shim above.

Fix: pass ``direction=`` and ``n_steps=``.

.. _upgrade-backend-utils:

``backend_utils``, ``regular_grid`` and ``staggered_grid`` are deprecated shims
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``v0.0.27``'s top level held fifteen names, six of which were modules
(``backend_utils``, ``common``, ``location_interpolation``, ``regular_grid``,
``staggered_grid`` and ``wavelets``). Three of the six are Deepwave's own
internals rather than anything a user was meant to
call: ``deepwave.backend_utils`` is the ``ctypes`` plumbing that finds the
compiled library and builds argument-type lists for its entry points, and
``deepwave.regular_grid`` and ``deepwave.staggered_grid`` hold the eager
backend's finite-difference stencils and PML profile builders. All three were
importable, though, and Deepwave 0.1.0 removes no *module* that ``v0.0.27``
shipped, so each is kept until ``0.4.0`` as a shim over its private
spelling.

.. upgrade-break: backend_utils_shim

.. code-block:: text

   DeepwaveFutureWarning: deepwave.backend_utils.dll is deprecated and will be
   removed in Deepwave 0.4.0; use deepwave._backend_utils.dll.
   deepwave.backend_utils was never a supported interface -- it is the compiled
   backend's ctypes plumbing -- so the private module's contents may change in
   any release.

.. upgrade-break: regular_grid_shim

.. code-block:: text

   DeepwaveFutureWarning: deepwave.regular_grid.set_pml_profiles is deprecated
   and will be removed in Deepwave 0.4.0; use
   deepwave._regular_grid.set_pml_profiles.

.. upgrade-break: staggered_grid_shim

.. code-block:: text

   DeepwaveFutureWarning: deepwave.staggered_grid.diff1 is deprecated and will
   be removed in Deepwave 0.4.0; use deepwave._staggered_grid.diff1.

Each shim carries the names its ``v0.0.27`` module defined itself -- nineteen
for ``backend_utils``, nine each for the two grid modules -- with the same
signatures they had then. The rest of what those modules exposed were imported
modules, ``typing`` imports and ``ctypes`` scalar types that their import
statements happened to re-export; they have their own canonical homes and are
not carried.

Fix: there is none, and none is needed -- nothing in these three modules was,
or is, a supported way to use Deepwave. Use
:func:`deepwave.diagnostics.report` for what the compiled backend on this
machine supports, and ``out.info.backend`` for what a call actually used.

.. _upgrade-hicks:

``Hicks`` takes keywords after ``halfwidth``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: hicks_keyword_only

:class:`deepwave.Hicks` takes the required ``locations`` and eleven optional
parameters. Only ``halfwidth``, the first of the eleven, may still be given
positionally; the other ten are now keyword-only.

.. code-block:: text

   TypeError: Hicks.__init__() takes from 2 to 3 positional arguments but 4
   were given

Fix: name them -- ``Hicks(locations, 4, free_surfaces=...)``.

.. _upgrade-hicks-dtype:

``Hicks(dtype=)`` defaults to the *default* dtype, not to ``float32``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: hicks_default_dtype

``v0.0.27`` built the interpolation weights in ``torch.float32`` whatever the
global default; the default is now ``None``, meaning
:func:`torch.get_default_dtype`. Under the stock ``float32`` default the two
are identical, nothing changes, and nothing is said.

They differ for anyone who has called ``torch.set_default_dtype``. After
``torch.set_default_dtype(torch.float64)``, the weights are now built in
``float64`` where ``v0.0.27`` would have built them in ``float32`` and then
promoted them. That is the intended behaviour -- a float64 model deserves
float64 weights, and the old promotion capped the interpolation's accuracy at
about ``1e-7`` relative -- but it is a numerical difference in the seventh
significant figure, so it is said out loud:

.. code-block:: text

   MigrationWarning: Hicks was built with no dtype= and the default dtype is
   torch.float64, so its interpolation weights are built in torch.float64.
   Deepwave 0.0.27 and earlier always built them in torch.float32 and promoted
   them afterwards, so this call's weights, and any receiver amplitude or
   gradient computed through them, are rounded differently from what that
   release produced: the same window of cells, with the weights on them
   computed in a different dtype.

This warning is about the rounding of the weights, which is the only thing the
dtype decides. The window -- the integer cells those weights are applied to --
is built in exact integer arithmetic whatever the dtype, and where it differs
from ``v0.0.27``'s :ref:`upgrade-hicks-stencil` says so separately. That
comparison is made in the dtype ``v0.0.27`` would have used (``float32`` when
``dtype=`` is omitted, whatever the default dtype is now), so a changed window
is reported under a ``float64`` default exactly as it is under the stock one.

Fix, if you want the old weights exactly: ``Hicks(..., dtype=torch.float32)``.
Passing the default dtype explicitly says that the new weights are what you
want, and silences it too.

.. _upgrade-ricker:

``ricker``'s ``nt`` and ``dt`` are required again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: ricker_required_arguments

.. code-block:: text

   DeepwaveTypeError: ricker() missing required argument: 'nt'

The signature is back to ``(freq, nt, dt, peak_time=None, dtype=None)``, so
the canonical ``v0.0.27`` call ``ricker(freq, nt, dt, peak_time)`` keeps
working positionally. On this branch (never in a release) ``nt`` and ``dt``
briefly had ``None`` defaults, which deferred the failure to a ``ValueError``
further in.

.. upgrade-break: ricker_length

The ``length`` alias for ``nt`` survives as a keyword-only argument:

.. code-block:: text

   DeepwaveFutureWarning: length is deprecated and will be removed in Deepwave
   0.4.0; use nt.

Fix: ``ricker(freq, nt, dt)``.

.. _upgrade-off-node:

Off-node coordinates are interpolated, not truncated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A float source or receiver coordinate that is not on a grid node is no longer
silently moved onto one. ``v0.0.27`` accepted any coordinate whose fractional
part was at most ``0.1`` cells and **truncated** it to the node below --
``5.05`` became cell ``5`` -- and rejected anything further off-node. What
happens now depends on ``location_interpolation``.

``"auto"`` (the default) **interpolates** it, using :class:`deepwave.Hicks`. A
coordinate 0.05 cells off-node is modelled where you put it rather than moved
back to the node below, and nothing raises. The tolerance for deciding "on
node", and so for taking the exact path instead, is
:data:`deepwave.ON_NODE_TOLERANCE`, ``1e-5`` cells. Because such a coordinate
is one ``v0.0.27`` accepted, the change of meaning is said out loud, once:

.. upgrade-break: truncated_coordinate

.. code-block:: text

   MigrationWarning: a float source or receiver coordinate lies more than
   1e-05 cells off its grid node but no more than 0.1 cells past the node
   below it. Deepwave 0.0.27 and earlier accepted such a coordinate and
   truncated it to that node; it is now interpolated onto the grid
   (deepwave.Hicks), so it is modelled where you put it and this call's
   output differs from what that release produced. Pass
   locations.floor().long() to reproduce the old placement exactly

A coordinate *further* than ``0.1`` cells off-node is one ``v0.0.27`` would
have rejected outright, so there is nothing to migrate and nothing is said.
Naming ``location_interpolation`` yourself is silent too: neither ``"hicks"``
nor ``"none"`` existed in ``v0.0.27``, so a call that passes one has already
been written for this release.

``"none"`` **raises**:

.. upgrade-break: off_node_none

.. code-block:: text

   ValueError: location_interpolation='none' requires source/receiver
   coordinates that lie exactly on grid nodes (within 1e-05 cells), but a
   float location Tensor has one or more off-node coordinates.

Fix: for the old placement, truncate explicitly
(``locations.floor().long()``); otherwise no change is needed, and your
off-node coordinates are now placed accurately.

.. _upgrade-hicks-stencil:

An off-node ``Hicks`` stencil is placed correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**This changes numbers for off-node coordinates, and it is a bug fix.**
``v0.0.27`` computed a Hicks stencil's integer cell indices by truncating a
floating-point sum, as ``(location + x).long()``. Where that sum landed a
fraction of a ULP below an integer, truncation moved one point of the stencil
a whole cell: a coordinate at 5.6 with the default ``halfwidth=4`` gave cells
``[1, 3, 4, 5, 6, 7, 8, 9]`` rather than ``[2, 3, 4, 5, 6, 7, 8, 9]``, so one
interpolation weight was applied a cell away from where it belonged and the
cell it belonged in got nothing. Truncation also rounds towards zero, so a
coordinate just below a cell boundary near the origin repeated a cell instead
of reaching the one below it. Sweeping coordinates on a hundredth-of-a-cell
grid, 6.2% of off-node coordinates in ``[1, 50]`` hit this (``halfwidth=4``,
monopole; measured by
``tests/test_v0027_hicks_stencil.py``). The indices are now built in integer
arithmetic (``floor(location) + arange``), which is exact; the interpolation
*weights* are bit-for-bit unchanged.

Deepwave says so when it happens, once, at the point the stencil is built:

.. upgrade-break: hicks_stencil

.. code-block:: text

   MigrationWarning: the Hicks interpolation window for a coordinate at
   5.599999904632568 cells covers cells [2, 3, 4, 5, 6, 7, 8, 9], where
   Deepwave 0.0.27 and earlier would have used [1, 3, 4, 5, 6, 7, 8, 9]. That
   release built the cell indices by truncating a floating-point sum, which put
   one cell of the window a whole cell away from where it belonged; they are now
   built in exact integer arithmetic.

There is nothing to do about it, and no way to ask for the old cells back: they
were wrong. It is warned about because a ``v0.0.27`` regression test on a
simulation with off-node float source or receiver coordinates, or with an
explicit ``deepwave.Hicks``, may fail on values rather than on an error, and
this is the reason. Integer and on-node coordinates are unaffected.

The comparison is made on the window as it is built, before a free or rigid
surface folds any part of it back into the model, so it may also warn when a
free surface makes the delivered cells identical. It is the one direction worth
erring in: a spurious message costs a line of reading, a missing one costs the
explanation of a changed number.

.. _upgrade-sc-receiver:

``scalar_born``'s scattered data is ``sc_receiver_amplitudes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattered receiver output field is named ``sc_receiver_amplitudes``,
matching :func:`deepwave.acoustic_born` and :func:`deepwave.elastic_born` and
the order of the returned tuple (background data first, scattered second).
``v0.0.27`` returned a plain tuple with no named fields, so
``out.receiver_amplitudes`` was never a name a released Deepwave answered
to: it is simply not a field.

``receiver_locations=`` *was* the spelling ``v0.0.27`` shipped. It keeps
working, with a :class:`deepwave.DeepwaveFutureWarning` naming
``sc_receiver_locations=``, and is removed in ``0.4.0``. The bare
``receiver_mask=`` never shipped either -- ``v0.0.27`` had no ``*_mask``
argument at all -- and is gone.

Fix: read the scattered data from ``out.sc_receiver_amplitudes``, and pass
``sc_receiver_locations=``.

.. _upgrade-source-p:

``p`` as a source channel names its quantity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``v0.0.27``, ``source_amplitudes_p`` meant injected volume rate on
:func:`deepwave.acoustic` and pressure rate on :func:`deepwave.elastic`.
These quantities differ by roughly the bulk modulus, so a source wavelet
cannot be moved between the two solvers unchanged.

The canonical channel names are ``volume_rate`` on the acoustic family and
``pressure_rate`` on the elastic family and ``acoustic_vti``. The released
``acoustic`` and ``elastic`` APIs continue to accept ``p`` as a deprecated
source alias until ``0.4.0``. New propagators accept only the quantity name.

This is a *source*-side change only. ``p`` is unchanged as a receiver
channel name: there it has always meant a pressure, on every propagator that
records one.

.. code-block:: python

    # v0.0.27, and 0.1.0 with a warning
    out = deepwave.acoustic(v, rho, dx, dt,
                            source_amplitudes_p=amplitudes,
                            source_locations_p=locations,
                            receiver_locations_p=receivers)
    # 0.1.0
    out = deepwave.acoustic(v, rho, dx, dt,
                            source_amplitudes_volume_rate=amplitudes,
                            source_locations_volume_rate=locations,
                            receiver_locations_p=receivers)

Fix: rename the source keyword, or the ``sources=`` key, to the quantity
name your propagator injects. The error message for the wrong family's name
says which one that is.

.. _upgrade-dataparallel:

``PropagatorOutput`` is not a plain ``tuple``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every propagator returns a :class:`deepwave.PropagatorOutput`: a ``tuple``
subclass that is also attribute-accessible. Indexing, slicing and unpacking
still work, but the tuple layout changed: absorbing-boundary tensors are now
grouped in one ``auxiliary_state`` entry. Update positional indexes and
unpacking to the new layout, or use named fields. An **exact**-type check such
as ``type(result) is tuple`` must become ``isinstance(result, tuple)``.

That matters in one specific place: ``torch.nn.DataParallel.gather``
*performs exactly that check* inside PyTorch. A ``forward()`` used with
``DataParallel`` on more than one GPU must therefore return a tensor or a
plain tuple rather than the whole ``PropagatorOutput``.

.. code-block:: python

    class Wrapped(torch.nn.Module):
        def forward(self, *args, **kwargs):
            out = deepwave.scalar(*args, **kwargs)
            return out.receiver_amplitudes      # a Tensor: DataParallel is happy

(An earlier version of this note claimed that ``v, r =
deepwave.scalar(...)`` was among the unpackings that still work. It never
worked, in any version: ``scalar`` returns four values today and returned
seven in 2-D and nine in 3-D before, never two.)

Fix: ``isinstance(result, tuple)``; return a tensor or plain tuple from a
``DataParallel``-wrapped ``forward``.

.. _upgrade-empty-receivers:

An unrequested receiver channel is shaped like the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A propagator returns every one of its receiver channels, whether or not the
call asked for it, because an omitted channel could not be told apart from a
misspelled one. ``v0.0.27`` returned a one-dimensional ``(0,)`` Tensor for a
channel with no receiver locations; ``0.1.0`` returns a three-dimensional
``[n_shots, 0, nt]`` one -- the shape a requested channel has, with zero
receivers. Both hold no elements, so anything that sums, plots or saves the
array is unaffected; only code that reads ``amplitudes.ndim`` or
``amplitudes.shape`` on an unrequested channel sees the difference, and it
now sees the same rank there as everywhere else. In exchange,
``torch.cat(list(out.receivers.values()), dim=1)`` builds a multichannel
gather without first filtering out the channels the call did not request.

Fix: none. If you tested for an unrequested channel with
``amplitudes.shape == (0,)``, test ``amplitudes.numel() == 0`` instead.

.. _upgrade-cuda-stream:

The compiled CUDA backend requires the default stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: cuda_non_default_stream

.. code-block:: text

   RuntimeError: Deepwave's compiled CUDA backend requires the default CUDA
   stream (kernel configuration uses device globals that are not stream-safe).
   Run outside torch.cuda.stream(), or use backend="eager".

The compiled kernels configure themselves through ``__constant__`` device
globals written by a synchronous ``cudaMemcpyToSymbol``, which is not
stream-safe: a call from a non-default stream could previously race with
another stream's launch and produce silently wrong results.

``torch.nn.DataParallel`` and ``DistributedDataParallel`` are unaffected --
they use each device's own default stream.

Fix: run on the default stream, or pass
``backend="eager"``/``backend="torch_compile"``, both of which are
stream-safe.

.. _upgrade-cuda-arch:

The compiled CUDA kernels ship a curated architecture list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiled kernels embed Ampere (sm_86) and Ada (sm_89) machine code,
plus PTX for the oldest architecture in the build's supported toolkit bucket.
This keeps wheel size within the release budget. Release wheels use CUDA 12.4
and compute_50 (Maxwell) as the PTX floor to support a wider range of drivers
and GPU generations. CUDA 13 source builds use compute_75 (Turing) as their
PTX floor because that is the oldest architecture accepted by CUDA 13.

A GPU at or above that floor without a matching machine-code entry can use
PTX JIT, provided its driver supports the emitted PTX version and the installed
PyTorch build supports that device. First use then incurs a compilation cost.
PTX is not a fallback for devices below the floor, and does not promise support
for every future device or driver.

Fix: check :func:`deepwave.diagnostics.report` and the installed PyTorch build.
For a GPU below the floor of the toolkit used for a source build, use a
compatible older CUDA toolkit and PyTorch, or run on the CPU.

.. _upgrade-step-ratio:

The ``step_ratio`` warning fires earlier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: step_ratio_warning

``cfl_condition_n``, and every propagator that calls it, now warns at
``step_ratio >= 8``; ``v0.0.27`` only warned at ``step_ratio >= 20``.

.. code-block:: text

   UserWarning: dt~0.02 exceeds the stable internal time step; Deepwave is
   taking step_ratio=18 internal steps per output sample

The two warnings never both fire for the same call, both now fire at most
once per process (previously on every call), and any model-derived value in
the message is rounded to three significant figures rather than printed at
full precision.

Fix: if you run with ``warnings.filterwarnings("error")``, either supply
``courant_number``/reduce ``dt`` to bring ``step_ratio`` down, or filter this
message explicitly.

.. _upgrade-max-vel-torch-func:

``max_vel`` is required under ``vmap``, not under every transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. upgrade-break: max_vel_batched_model

``v0.0.27`` treated every ``torch.func``-wrapped model as unreadable, so a
``torch.func.grad``, ``jacrev`` or ``jvp`` with respect to a model raised
``RuntimeError: If using BatchedTensor inputs, you must specify max_vel``
even though such a model's values are perfectly readable. Only
:func:`torch.func.vmap` actually hides them. ``max_vel`` is therefore now
inferred from the model under every other transform -- resolving the same
value, and so the same internal time step, as the call makes without the
transform -- and has to be given when a model that the maximum wave speed is
computed from is batched:

.. code-block:: text

   DeepwaveRuntimeError: scalar: max_vel was not specified and the model's
   values could not be read at this point, so the maximum wavespeed could not
   be inferred from them. torch.func.vmap does this to a model it maps over:
   a batched tensor stands for one model per batch entry, so it has no single
   maximum.

The message names the propagator, so all nine say the same thing in their
own name, and it names the other way a model's values can be unreadable (a
tensor that holds none: ``meta``, fake, nested). ``DeepwaveRuntimeError`` is a
``RuntimeError``, so an ``except RuntimeError`` still catches it; two of the
propagators (``acoustic_born`` and ``elastic_born``) refuse a batched model
outright, for their own reason, and say so instead. :doc:`capabilities` lists,
per propagator, the models the maximum is read from; batching any other model
does not require ``max_vel``.

One combination is outside this: under ``torch.compile`` of a ``vmap``\ ped
function, ``max_vel=`` must be given explicitly, and omitting it produces
PyTorch's own "``.item()`` on a Tensor ... with vmap" error rather than the
message above, because the readability probe is skipped while compiling.

Fix: nothing, unless you are mapping over a model with ``vmap``, in which case
pass ``max_vel=`` -- the maximum absolute wavespeed over the whole batch. It is
worth passing whenever you want the internal time step pinned across an
inversion, transform or not.

Renamed, removed, and deprecated symbols
----------------------------------------

**Everything in this first table shipped in ``v0.0.27``.** Every row but the
last is an alias that still works. The last row is the one thing ``v0.0.27``
shipped that Deepwave 0.1.0 removes outright.

.. list-table::
   :header-rows: 1
   :widths: 34 34 20 12

   * - Old spelling
     - New spelling
     - Status
     - Earliest removal
   * - ``python_backend=`` (``scalar``, ``scalar_born``, ``acoustic``, ``elastic``)
     - ``backend=``
     - deprecated alias, faithful ``v0.0.27`` mapping
     - ``0.4.0``
   * - ``scalar_born(scatter=)``
     - ``dv=``
     - deprecated alias
     - ``0.4.0``
   * - ``ScalarBorn(scatter=, scatter_requires_grad=)``
     - ``dv=``, ``dv_requires_grad=``
     - deprecated aliases
     - ``0.4.0``
   * - ``ScalarBorn.scatter``
     - ``.dv``
     - deprecated property
     - ``0.4.0``
   * - The ``scatter`` key in a ``ScalarBorn`` ``state_dict``
     - ``dv``
     - remapped on load, with a warning
     - ``0.4.0``
   * - ``CallbackState.get_model("scatter")``, ``get_gradient("scatter")``
     - ``"dv"``
     - deprecated
     - ``0.4.0``
   * - ``wavelets.ricker(length=)``
     - ``nt=``
     - deprecated, keyword-only
     - ``0.4.0``
   * - ``deepwave.common.<name>``
     - ``deepwave.*``, ``deepwave.discretisation.*``
     - deprecated shim
     - ``0.4.0``
   * - ``deepwave.location_interpolation``
     - ``deepwave.Hicks``
     - deprecated shim
     - ``0.4.0``
   * - ``deepwave.location_interpolation.DEFAULT_EPS``
     - ``deepwave.ON_NODE_TOLERANCE`` (``1e-5``)
     - deprecated shim
     - ``0.4.0``
   * - ``deepwave.backend_utils``
     - ``deepwave._backend_utils`` (private)
     - deprecated shim
     - ``0.4.0``
   * - ``deepwave.regular_grid``
     - ``deepwave._regular_grid`` (private)
     - deprecated shim
     - ``0.4.0``
   * - ``deepwave.staggered_grid``
     - ``deepwave._staggered_grid`` (private)
     - deprecated shim
     - ``0.4.0``
   * - ``scalar_born(receiver_locations=)``
     - ``sc_receiver_locations=``
     - deprecated (:ref:`upgrade-sc-receiver`)
     - ``0.4.0``
   * - ``source_amplitudes_p=`` / ``source_locations_p=`` and the
       ``sources={"p": ...}`` key on ``acoustic`` and ``elastic``
     - ``source_amplitudes_volume_rate=`` on ``acoustic`` and
       ``source_amplitudes_pressure_rate=`` on ``elastic``
     - deprecated
     - ``0.4.0``
   * - ``psiy_m1=``, ``phi_y_0=``, ``m_vzz_0=`` and the other 42
       auxiliary-field *input* keyword arguments
     - ``initial_state=out``
     - **removed**, no alias; raises ``TypeError``
       (:ref:`upgrade-auxiliary-state-input`)
     - ``0.1.0``

One name in that table is worth reading twice: ``storage_compression`` is a
**permanent** alias. It emits no warning and will not be removed.

**Nothing in this second table shipped in ``v0.0.27``.** These are names that
existed only on the development branch, renamed outright with no alias. If
you have only ever used a released Deepwave, none of them can affect you.

.. list-table::
   :header-rows: 1
   :widths: 46 46 8

   * - Old spelling (branch only)
     - New spelling
     - Note
   * - ``RunInfo.dt_internal``
     - ``RunInfo.internal_dt``
     - removed; raises ``AttributeError`` naming ``internal_dt``
   * - ``CallbackState.get_wavefield(role="backward")``
     - ``role="adjoint"``
     - removed; raises ``DeepwaveValueError`` naming ``"adjoint"``
   * - ``critical_courant_number(staggered=)``
     - ``propagator=``
     - removed; ``propagator=`` knows the answer, so the argument had nothing
       to add. Passing it raises ``TypeError``
   * - ``out.psiy_m1`` and the other auxiliary field names, as **output
       attributes** and as ``initial_state`` **keys**
     - ``out.auxiliary_state``, ``initial_state=out``
     - removed; ``v0.0.27`` returned a plain tuple and had no
       ``initial_state``, so neither spelling ever shipped
       (:ref:`upgrade-auxiliary-state`)
   * - ``backend="torch_jit"`` and its short spelling ``backend="jit"``
     - ``backend="eager"`` or ``backend="torch_compile"``
     - removed; Deepwave no longer calls ``torch.jit`` at all. The
       ``v0.0.27`` spelling ``python_backend="jit"`` still works and runs the
       eager backend (:ref:`upgrade-backend`)
   * - ``misfits.normalized_cross_correlation``
     - ``misfits.normalised_cross_correlation``
     - renamed outright
   * - ``estimate_memory(step_ratio=)``
     - --
     - deleted
   * - ``estimate_memory(total_bytes=)``
     - ``total_bytes_lower_bound``
     - renamed
   * - ``scalar(source_field=, source_field_origin=, source_field_mask=)``
     - ``wavefields.source_field_to_sources``
     - removed
   * - ``acoustic_born``/``elastic_born`` ``receiver_*``
     - ``sc_receiver_*``
     - renamed
   * - ``scalar_born(receiver_mask=)`` and ``scalar_born``'s
       ``out.receiver_amplitudes``
     - ``sc_receiver_mask=``, ``out.sc_receiver_amplitudes``
     - removed; ``v0.0.27`` had no ``*_mask`` argument and no named output
       fields, so neither spelling ever shipped
   * - ``acoustic_born``/``elastic_born`` ``*_0_sc``
     - ``*_sc_0``
     - renamed
   * - ``cpu_features()``
     - ``deepwave.diagnostics.report()``
     - moved
   * - ``acoustic_vti_source_taper``
     - ``deepwave.anisotropy.source_taper``
     - moved
   * - ``thomsen_to_cij``, ``cij_to_thomsen``, ``alkhalifah_to_thomsen``
     - ``deepwave.anisotropy.*``
     - moved
   * - ``run_segments(checkpoint_segments=<bool>)``
     - ``checkpoint=``
     - renamed; ``checkpoint_segments`` is now a segment *count*
   * - ``run_segments(callback=)``
     - ``segment_callback=``
     - renamed
   * - ``run_segments(checkpoint_use_reentrant=, checkpoint_slab_pml=)``
     - --
     - removed
   * - ``viscoacoustic`` ``r_0``, ...
     - ``auxiliary_state``
     - the relaxation memory variables are no longer output fields or
       keywords of their own; they are carried in the opaque
       ``auxiliary_state``, and ``initial_state`` passes them back
   * - ``transducers``: ``apodization=``, ``apodization_kind=``
     - ``window=``
     - renamed
   * - ``transducers.line_element(origin=)``
     - ``centre=``
     - renamed outright; the segment's midpoint
   * - ``transducers.arc_element(origin=)``
     - ``centre_of_curvature=``
     - renamed outright; not a point on the arc
   * - ``transducers.disc_element(origin=)``
     - ``centre=``
     - renamed outright; the disc centre
   * - ``transducers.bowl_element(origin=)``
     - ``apex=``
     - renamed outright; the pole, not the centre of curvature (``focus=``)
   * - ``transducers.window()``
     - ``transducers.window_weights()``
     - renamed outright; the builders' ``window=`` parameter is unchanged
   * - ``transducers.delays_to_focus(grid_spacing=)``
     - --
     - removed; ``c`` is now in cells per unit time
   * - ``deepwave.models``
     - --
     - removed; every function raised unconditionally
   * - ``das.helical_fiber_response(wrap_angle_deg=)``
     - ``wrap_angle=`` (radians)
     - renamed outright
   * - ``das.strain_rate_from_velocities`` velocity sequences
     - channel-keyed mappings
     - changed outright

Clarifications (nothing changed)
--------------------------------

These are places where the documentation was wrong, or silent, and the code
was not. No behaviour changed; only the description of it.

``time_taper``, ``freq_taper_frac`` and ``time_pad_frac``
    All three are ignored when no resampling occurs
    (``RunInfo.step_ratio == 1``) and are applied identically when it does.
    This was true in ``v0.0.27`` too; only the documentation now says so.

``region=`` is reserved
    The name is reserved on the callback accessors for a future release and
    is not accepted today. Nothing that used to work stops working.

The on-node tolerance
    Two earlier statements about this on the development branch were wrong,
    and are corrected in :ref:`upgrade-off-node`: the default mode
    interpolates rather than raising, and the constant is
    :data:`deepwave.ON_NODE_TOLERANCE` = ``1e-5``, not ``1e-6`` (the ``1e-6``
    belonged to a validation helper the propagator path does not reach).

``v0.0.27`` truncated an off-node coordinate; it did not round it
    An earlier statement on the development branch said that ``v0.0.27``
    "rounded any coordinate within ``0.1`` cells of an integer", and offered
    ``locations.round().long()`` as the way back to it. It *truncated*: it
    accepted a coordinate whose fractional part was at most ``0.1`` and used
    the node below, so ``locations.floor().long()`` is the expression that
    reproduces it and ``locations.round().long()`` is not. Corrected in
    :ref:`upgrade-off-node`, which now also warns when a call meets one of
    those coordinates.

``deepwave.location_interpolation`` did ship in ``v0.0.27``
    An earlier statement on the development branch said Deepwave had always
    pointed at :class:`deepwave.Hicks` as the module's only supported entry
    point. It had not: ``v0.0.27`` had no ``deepwave.Hicks`` at all, and
    ``deepwave.location_interpolation.Hicks`` was the supported spelling.
    That is why the module is a deprecated shim
    (:ref:`upgrade-location-interpolation`) rather than simply gone.

Changes that affect the development branch only
-----------------------------------------------

Deepwave has released nothing since ``v0.0.27``, so none of the following can
affect code written against a release. They are recorded for anyone who has
been tracking the development branch directly.

``callback_frequency`` counts user time steps
    In 0.1.0, callbacks observe user indices ``0, f, 2*f, ... < nt`` with
    ``callback_frequency=f``. The backward pass visits the same indices in
    reverse. ``CallbackState.step``, ``n_steps`` and ``dt`` are now in user
    units; ``internal_dt`` names the CFL step and ``time=step*dt``. This
    changes schedules that previously depended on model-gradient sampling.
    Index a snapshot array by ``step // f`` for either direction. The final
    state at ``nt`` remains in the propagator output, without a callback.

Every Module takes a ``<model>_requires_grad`` flag per model
    ``v0.0.27`` shipped individual flags on four Modules only (``Scalar``,
    ``ScalarBorn``, ``Acoustic``, ``Elastic``); the development branch had
    the other five take the single ``requires_grad=`` argument instead. All
    nine now take both, with one flag per public model name, so a wrapper
    that forwards ``v_requires_grad=`` or ``c11_requires_grad=`` no longer
    has to know which propagator it is driving. The flag set is checked
    against ``deepwave.spec(propagator).model_names(ndim)``, so it cannot
    drift from the models a propagator actually has. Combining
    ``requires_grad=`` with an individual flag still raises, as it always
    did, and a flag naming an *optional* model that was not supplied now
    raises rather than being silently dropped.

The Born propagators have one spelling for a scattered name
    ``acoustic_born`` and ``elastic_born`` spelled their scattered state
    fields with the ``sc`` tag after the time index internally
    (``pressure_0_sc``, ``m_vyy_0_sc``) while publishing the physical ones
    with it before (``pressure_sc_0``). There is now one spelling
    everywhere, the published one, including for the absorbing-boundary
    memory variables that :class:`deepwave.AuxiliaryState` carries. The same
    goes for the models: ``elastic_born``'s internal perturbations are
    ``dlamb``, ``dmu``, ``dmu_yx``, ``dbuoyancy_y``, ``dbuoyancy_x``, not
    ``lamb_sc`` and friends, matching ``acoustic_born``'s ``dK``/``dB*`` and
    the propagator's own ``dlamb``/``dmu``/``dbuoyancy`` arguments.

The forward wavefield is no longer retained for every ``backward_callback``
    All nine supported propagators gain
    ``backward_callback_needs_forward_wavefield=False``, which
    :func:`deepwave.scalar` implements and the others accept only at that
    default (``True`` raises :exc:`deepwave.DeepwaveNotImplementedError`
    there, and ``deepwave.spec(propagator).has_forward_wavefield`` says
    which is which without a call). Supplying any
    ``backward_callback`` used to make the forward pass retain a second full
    wavefield history so that
    ``CallbackState.get_wavefield(..., role="forward")`` could work, roughly
    doubling the compiled backend's backward-pass storage -- and costing more
    than that on ``backend="eager"`` -- even for a callback that never asked
    for it. Now you ask: pass
    ``backward_callback_needs_forward_wavefield=True``. Without it,
    ``role="forward"`` raises a ``NotImplementedError`` naming the argument,
    and ``CallbackState.has_forward_wavefield`` reports which case you are in.
    ``v0.0.27``'s ``CallbackState.get_wavefield`` had no ``role`` parameter at
    all. Note that under ``storage_compression=True`` the field
    ``role="forward"`` returns is dequantised from the lossy 8-bit history, so
    it approximates the propagated field rather than reproducing it.

``storage_path`` is resolved at call time, not at import
    For part of this branch's history ``storage_path`` defaulted to the
    *expression* ``tempfile.gettempdir()``, which Python evaluates once, when
    the module is imported. ``TMPDIR`` set after ``import deepwave`` therefore
    had no effect at all, and the rendered signature in the documentation
    showed whichever temporary directory the machine that built the
    documentation happened to have. The default is now ``None`` and the
    directory is resolved on each call. ``v0.0.27`` did not have this
    defect -- its default was the literal ``"."`` -- so this affects only the
    development branch. See :ref:`upgrade-storage-path` for the change that
    does affect a ``v0.0.27`` user.

Mutating a callback's wavefields and models
    The promise has been narrowed, but no behaviour changed.
    ``CallbackState.get_gradient`` still returns a live view whose in-place
    mutation is supported. ``get_wavefield`` and ``get_model`` still return
    views today, but mutating them is now documented as undefined: use
    :meth:`deepwave.CallbackState.set_wavefield`, and
    :meth:`deepwave.CallbackState.writable` to ask what is supported.

An ``origin`` that disagrees with the state you continue from now raises
    Passing ``initial_state=out`` together with an explicit ``origin`` that
    differs from the region ``out`` was produced in raises a ``ValueError``
    naming both regions. It previously continued silently over a different
    part of the model, displacing every field. The same applies to a
    ``survey_pad`` around sources that have moved -- the case that a
    same-shaped region elsewhere in the model made undetectable by shape
    alone. Omit ``origin`` to continue in the stored region.

A mapping-valued ``initial_state`` with no ``origin`` now warns
    A mapping of bare tensors records no region, unlike a
    ``PropagatorOutput`` or an ``AuxiliaryState``, so the survey origin is
    assumed to be zero. If the wavefields came from a call that used
    ``survey_pad`` or ``origin``, pass that call's output
    (``initial_state=out``) or the same ``origin=``.

``acoustic_vti``'s default backend now resolves to the compiled kernel
    One now exists. Pass ``backend="eager"`` to keep the pure-Python path.

``run_segments``' ``storage_mode`` default
    Changed from ``"none"`` to ``"device"``, matching every propagator. The
    previous mismatch silently lost differentiability for a segmented run
    that did not pass ``storage_mode`` explicitly.

``thomsen_to_cij`` returns a sixth value, the buoyancy
    :func:`deepwave.anisotropy.thomsen_to_cij` now returns
    ``(c11, c13, c33, c55, c66, buoyancy)``. A five-element unpacking breaks
    loudly (``ValueError: too many values to unpack (expected 5)``), which is
    the point. Add the trailing name::

        # before
        c11, c13, c33, c55, c66 = thomsen_to_cij(vp, vs, rho, eps, delta, gamma)
        buoyancy = 1 / rho

        # after
        c11, c13, c33, c55, c66, buoyancy = thomsen_to_cij(
            vp, vs, rho, eps, delta, gamma
        )

    and delete your own ``1 / rho``. In 2-D, where ``c66`` does not enter the
    equations, that is ``c11, c13, c33, c55, _, buoyancy = ...``. The reason
    for the change is that the propagators take ``buoyancy``, not ``rho``, so
    every caller was inverting the density by hand -- and a hand-written
    ``1 / rho`` gives ``inf`` at a vacuum cell, where the propagators' sentinel
    is ``buoyancy = 0``. The returned buoyancy maps ``rho == 0`` to exactly
    ``0``. The five stiffnesses are bit-for-bit what they were.

``cij_to_thomsen`` takes ``buoyancy`` instead of ``rho``, after ``c66``
    :func:`deepwave.anisotropy.cij_to_thomsen` is now
    ``cij_to_thomsen(c11, c13, c33, c55, c66, buoyancy)``: the density
    argument became the buoyancy, and moved to the end, so that the function
    takes exactly what :func:`deepwave.anisotropy.thomsen_to_cij` returns.
    ``c66`` no longer defaults to ``None``; pass it explicitly (``None`` in
    2-D)::

        # before
        vp, vs, rho, eps, delta, gamma = cij_to_thomsen(c11, c13, c33, c55, rho, c66)

        # after
        vp, vs, rho, eps, delta, gamma = cij_to_thomsen(
            c11, c13, c33, c55, c66, buoyancy
        )

    A positional call that is not updated passes your ``rho`` where
    ``buoyancy`` is expected and silently returns nonsense scaled by
    :math:`\rho^2`, so check every call site rather than relying on an
    exception. A vacuum cell (``buoyancy == 0``) now returns zeros for every
    Thomsen parameter rather than ``nan``/``inf``: nothing is recoverable
    there, and zero is the vacuum input that maps back to the same all-zero
    stiffnesses.

``deepwave.misfits``' ``reduction`` is keyword-only
    ``l2``, ``envelope`` and ``normalised_cross_correlation`` all take
    ``reduction`` after a ``*``, so ``envelope(pred, target, "sum")`` raises
    ``TypeError: envelope() takes 2 positional arguments but 3 were given``.
    Pass ``reduction="sum"``. This makes swapping one misfit for another safe
    in either calling style, which is the module's whole promise.

``transducers`` is unit-agnostic, and ``delays_to_focus`` lost ``grid_spacing``
    Every quantity in :mod:`deepwave.transducers` is now in *your* units.
    Positions are in grid cells, as they always were; ``c`` is a speed in
    **cells per unit time**; and delays and ``dt`` are in the same unit of
    time as each other. The module previously mixed three systems -- cells,
    a physical ``c`` "e.g. m/s", and delays "in seconds" -- with
    ``delays_to_focus(grid_spacing=)`` reconciling them, defaulting to ``1``
    and so silently asserting that one cell is one metre for anyone who
    omitted it. That argument is gone: the new ``c`` absorbs it. Fix:
    ``c_cells = c_physical / grid_spacing``, so
    ``delays_to_focus(loc, focus, c=1500.0, grid_spacing=1e-4)`` becomes
    ``delays_to_focus(loc, focus, c=1500.0 / 1e-4)`` and the delays come back
    in seconds as before. In the mm/us system :doc:`ultrasound` recommends,
    the same medium is ``c=1.5 / 0.1`` and the delays come back in
    microseconds, matching that system's ``dt``. The removal is loud: a
    keyword caller gets ``TypeError: delays_to_focus() got an unexpected
    keyword argument 'grid_spacing'``, and
    ``delays_to_focus(loc, focus, c, dx)`` gets ``TypeError:
    delays_to_focus() takes 3 positional arguments but 4 were given``.

Driving an element no longer needs ``* weights[:, None]``
    :func:`deepwave.transducers.element_source` returns the
    ``(source_amplitudes, source_locations)`` pair a propagator takes,
    applying the apodisation weights and any delay law for you::

        loc, weights = transducers.line_element(
            centre=[6.0, 45.0], direction=[0.0, 1.0], length=60.0)
        delays = transducers.delays_to_focus(
            loc, focus=[55.0, 45.0], c=1500.0 / 1e-4)
        amps, locs = transducers.element_source(
            wavelet, loc, weights, delays=delays, dt=dt)
        out = scalar(v, dx, dt,
                     source_amplitudes=amps, source_locations=locs)

    The raw ``(locations, weights)`` form still works and is documented as
    the low-level path; nothing that used it breaks.

``das.helical_fiber_response``'s wrap angle is in radians
    The argument is now ``wrap_angle``, in radians, and keyword-only;
    ``wrap_angle_deg``, in degrees, is gone with no alias. Every other angle
    in Deepwave is in radians (``transducers.arc_element(angles=)``), and an
    alias that quietly converted degrees to
    radians is exactly the silent unit change the rename exists to remove --
    hence keyword-only, so the old spelling fails loudly rather than
    reinterpreting your number. ``wrap_angle_deg=40.0`` raises ``TypeError:
    helical_fiber_response() got an unexpected keyword argument
    'wrap_angle_deg'``, and a positional ``40.0`` raises ``TypeError:
    helical_fiber_response() takes 2 positional arguments but 3 were
    given``. To convert, wrap the value you were passing in
    ``math.radians``: ``wrap_angle_deg=40.0`` becomes
    ``wrap_angle=math.radians(40.0)``, ``wrap_angle_deg=90`` becomes
    ``wrap_angle=math.pi / 2``, and the magic angle
    ``wrap_angle_deg=54.7356`` becomes
    ``wrap_angle=math.acos(1 / math.sqrt(3))`` (0.9553 radians).
    ``wrap_angle_deg=0`` is the one value that needs no conversion.

``das.strain_rate_from_velocities`` takes channel-keyed mappings
    ``v_receivers_plus`` and ``v_receivers_minus`` are now mappings from
    velocity channel name to receiver amplitudes -- ``{"y": vy, "x": vx}``
    in 2-D, ``{"z": vz, "y": vy, "x": vx}`` in 3-D -- using the same channel
    names as the propagators' ``receivers=`` mappings and their
    ``receiver_amplitudes_y``-style outputs. They used to be sequences whose
    order had to match the columns of ``tangents``, checked only for length,
    so passing the components in the wrong order returned a plausible wrong
    answer instead of raising. The keys are matched to the tangent columns
    by the library's slowest-to-fastest axis order, so no ordering is
    asserted anywhere. Migration is mechanical:
    ``das.strain_rate_from_velocities([vy_p, vx_p], [vy_m, vx_m], tangents,
    gauge_length)`` becomes
    ``das.strain_rate_from_velocities({"y": vy_p, "x": vx_p}, {"y": vy_m,
    "x": vx_m}, tangents, gauge_length)``. A sequence now raises
    ``DeepwaveTypeError`` naming the mapping form, and a missing or
    unexpected key raises ``DeepwaveValueError`` naming both the keys given
    and the keys expected for that ``tangents.shape[-1]``.

``wavefields.reinject_boundary``'s ``scale`` is now ``amplitude_scale``
    A rename only: the argument was already keyword-only, its default is
    still ``1.0``, and no number changes. There is no alias, so the old
    spelling fails loudly with ``TypeError: reinject_boundary() got an
    unexpected keyword argument 'scale'``; ``scale=k`` becomes
    ``amplitude_scale=k``. The new name says what the number does -- it
    multiplies the injected amplitudes -- and matches the amplitude
    discussion the function's docstring and :doc:`wavefields` now carry.
    The helper replays supplied traces as source amplitudes at surface cells.
    It does not convert recorded-field units to source units or generally
    reconstruct the original field.

``tone_burst``'s default ``peak_time`` is now record-centred
    It was ``n_cycles / freq / 2``, which centred the burst *window* and so
    placed the start of the burst at ``t = 0``. It is now ``nt * dt / 2``, the
    record-centring convention :func:`deepwave.wavelets.ricker`,
    :func:`deepwave.wavelets.gaussian_pulse` and
    :func:`deepwave.wavelets.ormsby` already used, so that swapping one
    wavelet builder for another no longer moves the source in time. This
    changes numbers for any call that omits ``peak_time``: the source moves
    later by ``nt * dt / 2 - n_cycles / freq / 2``, which is up to half a
    record. ``tone_burst(3.0, 5, nt=400, dt=0.01)`` moves its peak from
    ``0.833333`` to ``2.0`` (116.667 samples, 29.17% of the record);
    ``tone_burst(1e6, 5, nt=4000, dt=2e-9)`` moves its peak from ``2.5e-6`` s
    to ``4e-6`` s (750 samples, 18.75% of the record). Where the shift exceeds
    the burst duration the old and new bursts do not overlap at all, so the
    two waveforms differ everywhere the burst is nonzero. To restore the old
    waveform exactly, pass the old default explicitly::

        tone_burst(freq, n_cycles, nt=nt, dt=dt,
                   peak_time=n_cycles / freq / 2)

``viscoacoustic`` requires ``q_freq``
    ``q_freq`` used to default to ``pml_freq``. It no longer has a default: a
    default single-mechanism call (``q_n_mechanisms == 1``) that omits it
    raises ``DeepwaveValueError``, and a multi-mechanism call
    (``q_n_mechanisms > 1``) that passes it raises too, mirroring ``q_band``,
    which is required there and rejected here. ``q_freq`` is the frequency at
    which you specified ``Q``, a property of the medium you are
    describing; ``pml_freq`` tunes the absorbing boundary and is itself
    inferred from the source amplitudes when omitted, so the old default made
    the attenuation model a function of the wavelet, silently. Migration is one
    keyword: ``q_freq=<the frequency your Q was measured or chosen at>``. To
    reproduce a development-branch run that relied on the default, pass the
    ``pml_freq`` that run resolved (its ``out.info.pml_freq``) as ``q_freq``.
    The value must be a single positive, finite number: zero, negative,
    infinite and ``NaN`` values raise ``DeepwaveValueError`` rather than
    producing a ``NaN`` wavefield thousands of time steps later, and a value
    that is not a number at all -- a string, a ``bool``, a list, a
    multi-element tensor -- raises ``DeepwaveTypeError``. A one-element tensor
    or a NumPy scalar is accepted.

``to_cells``'s ``origin`` is now ``grid_origin``
    A rename only, of the third parameter of :func:`deepwave.to_cells`; it is
    still positional, so ``to_cells(coords, dx, origin_value)`` is unaffected
    and only ``origin=`` keyword calls need changing. The propagators' own
    ``origin`` is a *cell index* -- where an extracted model region begins --
    while this one is a *physical coordinate*, that of cell 0. One name for
    two quantities in the same units-conversion step was the ambiguity worth
    removing.

``backend="compile"``, the ``torch.jit`` modes and the ``""`` channel key are gone
    ``backend`` takes ``"torch_compile"``; ``"compile"`` is the internal
    execution mode's name and ``v0.0.27``'s ``python_backend`` value, and
    ``python_backend="compile"`` still works (deprecated). The
    ``backend="torch_jit"`` mode and its ``"jit"`` spelling are gone too:
    Deepwave no longer calls ``torch.jit``, and ``python_backend="jit"``
    runs the eager backend (see :ref:`upgrade-backend`). The two scalar
    propagators'
    single channel is spelled ``"u"`` in ``sources=``/``receivers=`` mappings;
    the empty string is no longer accepted as a second spelling of it, so an
    empty key is reported as the mistake it usually is. None of these
    spellings shipped in ``v0.0.27``.

Detecting the version in code
-----------------------------

``deepwave.__version__`` is new in 0.1.0 -- reading it is itself an
``AttributeError`` on the version you are trying to support -- and string
comparison is not version comparison. :doc:`stability` gives the two recipes
(feature detection, and a defensive parsed-version comparison) and explains
both traps; use those rather than inventing a third.

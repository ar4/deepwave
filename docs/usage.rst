Usage
=====

This page introduces what every propagator has in common: the axis
convention, how a survey's sources and receivers are specified, what a call
returns and how a run is continued, and the model, checkpointing and
off-grid options that apply across the whole interface. The API reference --
every propagator class and function, and the shared utilities alongside them
-- is split over the per-family pages listed under :ref:`api-reference`
below.

Not every propagator supports every option (dimensionality, backend, boundary
faces, batching, second-order differentiation, callbacks, ...); see
:doc:`capabilities` for the full, generated matrix before assuming a
combination works.

Every name in Deepwave's interface has a stability tier -- ``stable``,
``provisional``, ``experimental``, or ``deprecated``. :doc:`stability` defines
what each tier promises and lists which names are in which tier today. In
short, ``0.1.0`` freezes the supported propagator interface; newer helper
modules are ``provisional`` and may change under the process described in
:doc:`stability`.

.. contents:: :local:

.. _api-reference:

API reference
-------------

* :doc:`reference_scalar` -- the ``scalar`` and ``scalar_born`` propagators.
* :doc:`reference_acoustic` -- the ``acoustic``, ``acoustic_born``,
  ``viscoacoustic`` and ``acoustic_vti`` propagators.
* :doc:`reference_elastic` -- the ``elastic``, ``elastic_born`` and
  ``elastic_vti`` propagators.
* :doc:`reference_utilities` -- everything the propagators share: the output
  and callback types, the type aliases, checkpointing, storage, planning and
  diagnostics, the discretisation helpers, off-grid locations, wavelets,
  transducer arrays, misfit functions, wavefield-space tools, and the
  parameter converters.
* :doc:`reference_errors` -- the exception and warning classes Deepwave
  raises.

.. _axis-convention:

Dimension order and the axis convention
---------------------------------------

**Axis convention.** Deepwave orders spatial dimensions from slowest-varying
to fastest-varying: dimension 0 is the slowest axis of every model, wavefield
and location tensor. **Deepwave attaches no physical meaning to any axis** --
dimension 0 is depth only if you decide it is, and the propagators behave
identically under any relabelling, provided ``grid_spacing``, the
source/receiver location columns and the component suffixes all agree. The
component suffixes count from the **fastest** axis: ``_x`` is the last
dimension, ``_y`` the second-last, ``_z`` the third-last. So a 2-D run has
``_y`` and ``_x``, and a 3-D run has ``_z``, ``_y`` and ``_x``. **Deepwave's
own documentation and examples put depth in dimension 0** throughout, so the
vertical component is ``_y`` in 2-D and ``_z`` in 3-D. The one exception is
the anisotropic VTI family (:func:`deepwave.acoustic_vti`,
:func:`deepwave.elastic_vti`): its symmetry
axis **is** dimension 0 by construction, so for those propagators the axis
order is a physics constraint, not a convention.

.. note::

   **Promoting a 2-D script to 3-D.** The suffixes count from the fastest
   axis, so adding a dimension renames the ones you already have. Appending a
   new fastest axis -- ``(depth, x)`` becomes ``(depth, y, x)`` -- renames
   ``_y`` to ``_z`` and ``_x`` to ``_y``, and the new axis becomes ``_x``:

   * ``source_amplitudes_y`` becomes ``source_amplitudes_z`` (the vertical
     source);
   * ``source_amplitudes_x`` becomes ``source_amplitudes_y``;
   * add a new ``source_amplitudes_x`` for the new axis;
   * ``grid_spacing`` and every location tensor's last dimension grow from 2
     to 3.

   **A 2-D script that is promoted without renaming still runs**, and injects
   a horizontal force where you meant a vertical one. There is no error,
   because every suffix is valid in 3-D -- only the meaning changed.

   If instead you insert the new axis *before* the last one -- ``(depth, x)``
   becomes ``(depth, new, x)`` -- the old depth axis changes from ``_y`` to
   ``_z``, the existing ``_x`` stays ``_x``, and the inserted axis is ``_y``.
   Decide which of the two you mean before you start.

Data types
----------

All propagators require model tensors (e.g. ``v``, or ``lamb``/``mu``/
``buoyancy``) to be ``torch.float32`` or ``torch.float64``: the compiled
finite-difference backend does not implement half precision, so a
``float16``/``bfloat16`` model raises a ``TypeError`` immediately, before
propagation starts. ``torch.autocast``/``torch.cuda.amp`` are not supported
and are a no-op for Deepwave -- if a model ends up in a low-precision dtype
because of an enclosing autocast context, cast it back explicitly (e.g.
``v = v.float()``) before calling the propagator. ``source_amplitudes`` and
any initial wavefields must match the model's dtype exactly.

.. _model-parameterisations:

Model parameterisations
------------------------

Every propagator takes physical model tensors directly (wavespeed, density,
Lamé parameters, stiffnesses, ...), but nothing requires you to *optimise*
those exact tensors. Propagation is an ordinary differentiable PyTorch
operation, so any invertible reparameterisation built from plain tensor
arithmetic composes with autograd for free. Build the propagator's native
inputs from your own leaf variables inside the loss, call the propagator,
and ``backward()`` chains the gradient through the reparameterisation with
no propagator-side support required. :doc:`example_parameterisation` works
through this in full (velocity/impedance and slowness reparameterisations);
this section summarises the pattern and the two situations that most often
prompt "does Deepwave support parameterisation X?" questions.

**Converters between physical parameterisations.** Where a standard,
non-trivial (not just elementwise-invertible) mapping exists between two
established physical parameterisations, Deepwave provides it as a plain,
differentiable function rather than leaving every user to re-derive it:
:func:`deepwave.vpvsrho_to_lambmubuoyancy`/
:func:`deepwave.lambmubuoyancy_to_vpvsrho` convert between elastic
wavespeed/density and Lamé/buoyancy parameters, and
:func:`deepwave.anisotropy.thomsen_to_cij`/:func:`deepwave.anisotropy.cij_to_thomsen`/
:func:`deepwave.anisotropy.alkhalifah_to_thomsen`/
:func:`deepwave.anisotropy.thomsen_to_alkhalifah` convert between VTI stiffnesses and
the Thomsen/Alkhalifah-Tsvankin anisotropy parameterisations that
practitioners more commonly reason and optimise in. Each is an ordinary
differentiable function of tensors, so it composes with the same
build-inputs-then-call-the-propagator pattern as any other
reparameterisation -- there is nothing propagator-specific about them.

**Optimising only a known subset of the model (masked-subregion FWI).** A
common request is to update the model only inside a known region -- for
example, freezing a well-constrained overburden and inverting only a target
zone, or excluding a known-correct block from gradient updates entirely.
This does not need any propagator-level masking feature. (`GitHub PR #114
<https://github.com/ar4/deepwave/pull/114>`_ proposed one and it was
declined on cost/benefit grounds: the model gradient is a single
model-sized accumulator, so masking it saves only a small fraction of the
adjoint's peak memory, which is dominated by wavefield snapshot storage.
Adding a mask argument would also mean threading it through every compiled
kernel signature.) The supported route is the same chain-rule pattern as any
other reparameterisation, applied with a fixed background and a masked
update::

    v_fixed = v_init.clone()                 # the known-correct background
    update = torch.zeros_like(v_init, requires_grad=True)  # the only leaf
    mask = torch.zeros_like(v_init, dtype=torch.bool)
    mask[40:60, 20:80] = True                # the target zone to invert

    optimiser = torch.optim.Adam([update], lr=1.0)
    for _epoch in range(n_epochs):
        optimiser.zero_grad()
        v = v_fixed + torch.where(mask, update, torch.zeros_like(update))
        loss = torch.nn.functional.mse_loss(run(v), observed)
        loss.backward()
        optimiser.step()

``torch.where(mask, update, torch.zeros_like(update))`` zeroes ``update``'s
contribution (and, by the ordinary chain rule, its incoming gradient)
outside the masked region before it reaches ``v``. Cells outside ``mask``
therefore never move from ``v_fixed`` and never accumulate a gradient,
without changing what the propagator itself computes or requiring any
kernel-level support. The same pattern composes with the converters above
(e.g. mask only the ``vp`` term of an impedance reparameterisation) and with
a mask that varies per iteration, since ``mask`` is an ordinary tensor, not
a propagator argument.

.. _batched-models:

Batched models
--------------

Every propagator accepts a **batched model**: a model tensor carrying a
leading shot dimension, so that each shot in the call propagates through its
own model. Pass ``v`` of shape ``[n_shots, *model_shape]`` instead of
``[*model_shape]``, and everything else stays the same.

**Deepwave never infers dimensionality from your model.** It works out
``ndim`` from the survey -- from an initial wavefield's rank, or from the
number of coordinates per point in ``source_locations``/``receiver_locations``
-- and only then looks at the models. A model is then required to have either
``ndim`` dimensions (unbatched) or ``ndim + 1`` (batched); anything else
raises, naming both numbers. ``grid_spacing`` is checked against the resolved
``ndim`` and never contributes to deciding it.

That rule is what removes the apparent ambiguity of a shape like
``[2, 100, 200]``. The same tensor is:

* a **2-shot batch of 2-D** ``[100, 200]`` models, if the locations have two
  coordinates per point; and
* a **single 3-D** ``[2, 100, 200]`` model, if they have three.

Nothing about the model tensor, the propagator, or ``grid_spacing`` enters
into it, so there is no heuristic to be surprised by -- but it does mean that
a call with no sources, receivers or initial wavefield cannot determine
``ndim`` at all, and says so rather than guessing.

The leading dimension must be exactly ``1`` or exactly ``n_shots``. A size-1
batch dimension is accepted and broadcast (it is the same as passing an
unbatched model); a size that is neither raises. General broadcasting is
deliberately not supported -- 3 models against 2 shots is a mistake, not a
convenience. The choice is per model tensor, so a batched ``v`` may be
combined with an unbatched ``rho``.

Gradients follow the shape you passed: a batched model receives one gradient
per batch entry, an unbatched model receives a single gradient accumulated
over every shot. Memory scales with the model you pass, so a batched model
costs ``n_shots`` times an unbatched one; batch models when the shots really
do see different media (a moving survey, or per-shot perturbations), not
merely to run more shots at once.

:doc:`example_batched_models` is a worked example. Note that the "Batching"
column of :doc:`capabilities` is about something else -- it reports whether
``torch.func.vmap`` over the primary model works, which is a separate feature
with its own backend restrictions -- so do not read it as a statement about
the batched models described here.

Named returns and continuing a simulation
-----------------------------------------

Every propagator returns a :class:`deepwave.PropagatorOutput`, which exposes
every output by name (e.g. ``out.receiver_amplitudes``). It also remains an
``tuple`` whose final entry is receiver data and whose earlier entries
include final wavefields and auxiliary state. Positional access remains
available, but the layout changed from v0.0.27; audit old indices and unpacking
against :doc:`upgrading`. Named access is recommended because
the index of the receiver data in the positional return shifts with the
number of spatial dimensions::

    import torch
    import deepwave
    from deepwave.wavelets import ricker

    v = 1500 * torch.ones(60, 60)
    source = ricker(15, 200, 0.001, 0.05).reshape(1, 1, -1)
    receiver_locations = torch.tensor([[[40, 40]]])
    out = deepwave.scalar(
        v, 5.0, 0.001,
        source_amplitudes=source,
        source_locations=torch.tensor([[[30, 30]]]),
        receiver_locations=receiver_locations,
        max_vel=1500.0, pml_freq=15.0, origin=[0, 0],
    )
    data = out.receiver_amplitudes        # by name, independent of ndim

Every receiver output is ``[n_shots, n_receivers_per_shot, nt]``, and a
propagator returns **all** of its receiver channels, whether or not the call
asked for them: omitting an unrequested channel would make it
indistinguishable from a misspelled one. A channel with no receiver locations
comes back with ``n_receivers_per_shot == 0`` and the run's own ``n_shots``
and ``nt``, so it holds no elements while still having the rank and the axis
order of the data. ``out.receivers`` is the same data keyed by channel name,
which makes a multichannel gather one line::

    gather = torch.cat(list(out.receivers.values()), dim=1)

The returned **wavefields** cover the *padded* domain, not the region of the
model you gave: their spatial shape is the extracted model region
(``out.info.extent``) plus ``out.info.pml_width`` cells on each face for the
absorbing layer, plus the finite-difference halo the stencil needs (which
grows with ``accuracy``). Index ``0`` along a dimension is therefore the
outermost halo cell, not model cell ``out.info.origin``; the model region
starts ``halo + pml_width`` cells in. This is deliberate -- an output has to
be re-feedable as an ``initial_state``, and stripping the padding would throw
away exactly the state a continuation needs. To look at the field over your
own model region, slice that padding off, or use a ``forward_callback``,
whose accessors take a ``view=`` argument (:data:`deepwave.WavefieldView`)
and default to ``"inner"``, the model region alone.

Each output wavefield is named identically to the matching initial-wavefield
argument (``wavefield_0``, ``wavefield_m1``, ...), so an output can be fed
straight back in through the ``initial_state`` argument to continue a
simulation -- for example to propagate further in time, the basis of manual
checkpointing and domain decomposition. ``wavefield_0`` (and every other
``*_0`` field) holds the wavefield at the run's *final* time step, not the
initial one: the same name is used for both the initial-wavefield argument
and the matching output field, so that an output can be fed straight back in,
but on the output side it always means "the last step reached". Its receiver
fields are ignored, so a whole output can be passed directly::

    more = deepwave.scalar(
        v, 5.0, 0.001, nt=100,
        initial_state=out,
        receiver_locations=receiver_locations,
        max_vel=1500.0, pml_freq=15.0, origin=[0, 0],
    )

Alongside the physical wavefields, ``out.auxiliary_state`` carries the
absorbing boundary's memory variables and any other auxiliary state as one
opaque :class:`deepwave.AuxiliaryState` object, rather than as separate named
fields: continuing a simulation exactly requires carrying this state across
the join, but its individual entries are implementation detail of the
absorbing-boundary scheme, not physics, so the output carries them as one
value rather than as fields of its own. The entries are named -- the mapping's
keys are the internal auxiliary-field names :doc:`/callback_names` lists, and
they follow the provisional diagnostic-name policy (see :doc:`/stability`) --
but the bundle is meant to be
passed back whole rather than taken apart. ``initial_state=out`` (or
``initial_state=out.final_state``) carries it forward automatically; you do
not need to touch it yourself. ``initial_state`` also accepts a plain
``{name: tensor}`` mapping with a ``"auxiliary_state"`` key holding a
``AuxiliaryState`` (exactly the shape of ``out.final_state``). Supplying a
wavefield both through ``initial_state`` and as an explicit argument raises an
error. Supplying only some of an ``AuxiliaryState``'s fields also raises: it is
one bundle, and the absent fields would otherwise silently start from rest.

For the continuation to exactly reproduce a single run over the combined time
range, the same internal time step **and the same absorbing boundary** must be
used throughout. The first holds when the user ``dt`` already satisfies the
CFL condition (no internal sub-stepping). The second means pinning
``pml_freq``: an inferred ``pml_freq`` comes from each call's own portion of
the source record, and half a wavelet has a different spectrum from the whole
of it, so segments of an otherwise identical propagation resolve different
absorbing boundaries (measured on this: ~1e-7 relative on the scalar and
acoustic families, 6e-4 to 1.5e-3 on the elastic family and ``acoustic_vti``).
Pin ``max_vel`` too if the model differs between calls -- it scales the PML
profiles as well as choosing the internal time step, so a model whose maximum
wavespeed lies outside one call's extracted region is enough to change them.
Deepwave checks all three: a continuation whose internal time step differs
raises, and one whose ``pml_freq`` or ``max_vel`` differs by more than 1%
warns.

``out.info`` is a :class:`deepwave.RunInfo`: everything the call resolved,
so that a run's numerical configuration can be read back off its own output
rather than reconstructed. Its fields are ``dt``, ``grid_spacing``,
``accuracy``, ``boundary``, ``pml_width``, ``pml_freq``, ``mpml_fraction``,
``surface_method``, ``q_freq``, ``q_n_mechanisms``, ``q_band``,
``step_ratio``, ``internal_dt``, ``courant_number``, ``nt``, ``max_vel``,
``origin``, ``extent``, ``ndim``, ``dtype``, ``device``, ``model_batched``,
``wavefield_names``, ``receiver_names``, ``location_interpolation_used``,
``backend``, ``backward_backend``, ``storage`` and
``lossy_compression_used``. Each is the value the run used, not the value you
passed: ``pml_width`` is expanded to one entry per face, ``boundary`` names
every face including the ones a zero ``pml_width`` turned into something
else, ``pml_freq`` and ``max_vel`` are the inferred values when you gave
none, and ``storage`` is the resolved policy including the directory
``mode="disk"`` chose. :class:`deepwave.RunInfo`'s own reference entry
describes each one.

The three classes named here -- :class:`deepwave.PropagatorOutput`,
:class:`deepwave.AuxiliaryState` and :class:`deepwave.RunInfo` -- are
documented in :doc:`reference_utilities`.

.. _auxiliary-state-checkpoint:

``auxiliary_state`` and ``torch.utils.checkpoint``
--------------------------------------------------

``PropagatorOutput`` is a genuine ``tuple`` subclass so that it survives
``torch.utils.checkpoint`` the way a plain tensor tuple does. That survival is
only complete with the non-reentrant implementation. Under
``torch.utils.checkpoint.checkpoint(..., use_reentrant=True)`` (currently
torch's default), the wrapped call runs under ``torch.no_grad()`` and only the
top-level ``Tensor`` elements of its return are re-attached to the graph
afterwards: the return degrades to a plain ``tuple`` (losing ``out.<name>``
attribute access and ``out.info``), and every tensor inside
``out.auxiliary_state`` comes back **detached**, so continuing a simulation
from it gives no gradient through the absorbing boundary. There are two ways
out of that, and they are not equivalent.

The easy one is to **pass ``use_reentrant=False``** (which PyTorch also
recommends generally) whenever a checkpointed propagator call's output -- and
in particular its ``auxiliary_state`` -- is used after the checkpoint
boundary. The non-reentrant implementation re-runs the function under a
saved-tensor hook rather than under ``torch.no_grad()``, so it does not have
to inspect the return value at all: the ``PropagatorOutput`` subclass and
every tensor inside ``auxiliary_state`` survive with their graph intact.

The robust one is to **not return a container in the first place**, and that
is what :func:`deepwave.checkpointing.run_segments` does. The function it
hands to ``torch.utils.checkpoint`` does not return the
:class:`~deepwave.PropagatorOutput`; it flattens the segment's state and
receiver fields into a plain tuple of tensors, and rebuilds the
``PropagatorOutput`` and its ``AuxiliaryState`` on the other side of the
boundary from that tuple. A reentrant checkpoint re-attaches exactly the
top-level ``Tensor`` elements of a returned tuple, so a flat tuple loses
nothing: there is no non-``Tensor`` element for a gradient to be silently
dropped inside. That is why a segmented run keeps its auxiliary state's
gradients whichever implementation torch is using, rather than depending on
a flag -- and it is the pattern to copy if you write your own segment loop.
Threading a container -- a dataclass, or the ``PropagatorOutput`` with its
``AuxiliaryState`` inside it -- across a reentrant boundary does not fail
loudly: the run completes, the receiver amplitudes are right, and the only
symptom is that the gradient contribution flowing back through the absorbing
boundary at that join is missing.

Deliberately truncating backpropagation through a long simulation -- running
many segments but backpropagating through only the most recent ones --
requires cutting the graph at a segment join on purpose, which is
indistinguishable, from inside a propagator, from having lost the gradient by
accident (the ``use_reentrant=True`` case above). Call
:meth:`~deepwave.PropagatorOutput.detach` on the output and continue from
that::

    out = scalar(v, dx, dt, source_amplitudes=chunk, initial_state=state)
    loss(out.receiver_amplitudes).backward()
    state = out.detach()          # the whole join, cut on purpose

Every field of the join has to be cut for the graph to actually be cut. The
physical wavefields (``wavefield_0``, ``wavefield_m1`` and their equivalents)
are separate fields from ``auxiliary_state``, so detaching the auxiliary state
alone leaves them attached: the join still carries the whole history, nothing
is freed, and the gradient is the untruncated one.
``PropagatorOutput.detach()`` detaches all of them, and detaches the boundary
state through :meth:`~deepwave.AuxiliaryState.detach` so that the cut is
recorded as deliberate. A state detached this way is accepted silently, while
an ``auxiliary_state`` that arrives detached without that marker raises a
warning, since it is usually a sign that gradient history was lost by mistake
somewhere upstream. ``AuxiliaryState.detach()`` remains available for a state
assembled field by field, where the rest of the join is being detached by
hand anyway.

.. _gradient-checkpointing:

Gradient checkpointing
-----------------------

For long propagations the backward pass is dominated by the stored forward
wavefield snapshots (one per time step). The standard way to bound this is
``checkpoint_segments=`` on the propagator itself: it splits the propagation
into that many contiguous time segments, threads each segment's output state
(wavefields and PML auxiliaries) into the next via ``initial_state``, and
wraps every segment except the last in
``torch.utils.checkpoint.checkpoint``, so a segment's forward activations are
recomputed on demand during the backward pass instead of being kept alive for
the whole run. Peak backward memory becomes roughly one segment's history plus
the auxiliary states, at the cost of one extra forward pass per checkpointed
segment.

``checkpoint_segments=k`` is shorthand for
:func:`deepwave.checkpointing.run_segments` with ``n_segments=k`` and
``checkpoint=True``, and returns the same combined output. Call the driver
directly for anything the shorthand cannot say: ``seg_lengths=`` for uneven
segments, ``segment_callback=`` to observe each segment as it completes, or
``checkpoint_device`` to park the auxiliary states on another device (e.g.
``"cpu"``) between segments.

The states carried across a segment boundary are shrunk automatically.
A PML memory variable is exactly zero away from the PML, so only its two
per-axis boundary slabs -- ``pml_width + accuracy`` cells at each end -- are
carried across, and the zero interior is rebuilt inside the next segment.
This changes no value and no gradient, and shrinks the auxiliary-state term of
peak memory by roughly two to three times, more at larger grid sizes. The
"exactly zero" part is a property of the compiled kernels' standard PML
formulation, not a general truth about absorbing boundaries, so the
optimisation checks that it applies and silently threads full-size states
when it does not: with ``mpml_fraction > 0`` (multi-axial PML, whose memory variables
are legitimately non-zero wherever *any* axis is inside its PML, not just the
variable's own), with a non-default
``boundary``, on a backend other than the compiled one, for a propagator that
does not declare which axes its memory variables are slab-shaped on, and for
grids too small for a slab to be smaller than the whole field. In every one of
those cases the numbers are identical; only the memory is larger.

Internal sub-stepping is handled exactly: at ``step_ratio > 1`` the whole source trace is
upsampled once, the segments run at the internal time step, and the assembled
receiver amplitudes are downsampled once at the end, so the result matches an
unsegmented call -- pass ``max_vel`` in that case. See :doc:`the checkpointing
example <example_checkpointing>` for a worked inversion.

If you need custom per-segment logic, you can drive the segments in your own
loop instead, calling a propagator (or ``run_segments``) once per segment and
wrapping each call in ``torch.utils.checkpoint.checkpoint`` yourself. Then the
resampling is yours to do, because no single segment's call can see the rest of
the trace (the resampling FFTs are global in time): upsample the whole source
before chunking it, run the segments at the internal time step, and downsample
the assembled receiver amplitudes at the end::

    inner_dt, step_ratio = deepwave.discretisation.cfl_condition_n(
        grid_spacing_list, dt, max_vel
    )
    source_amplitudes = deepwave.discretisation.upsample(source_amplitudes, step_ratio)
    # drive segments at inner_dt with model_gradient_sampling_interval=step_ratio,
    # then deepwave.discretisation.downsample(receiver_amplitudes, step_ratio) at the end

Note that ``cfl_condition_n`` must be given the grid spacing **per dimension**
(a bare float is treated as a one-dimensional grid, giving a larger stable time
step than a 2-D or 3-D grid actually allows).

:func:`deepwave.checkpointing.run_segments` is documented in
:doc:`reference_utilities`.

``run_segments`` uses reentrant checkpointing by default. This retains less
native simulation history and is the lower-memory choice, but PyTorch's
reentrant checkpoint implementation does not support calling
``torch.autograd.grad`` through the checkpoint. Pass ``use_reentrant=False``
to ``run_segments`` when you need ``torch.autograd.grad``; the non-reentrant
implementation supports it at the cost of retaining more native history.
Setting ``use_reentrant=False`` is a compatibility choice for that autograd
interface, not a requirement for ordinary ``loss.backward()`` calls.

.. _differentiable-snapshots:

Differentiable intermediate snapshots
-------------------------------------

A normal propagator call exposes the wavefield only as a *detached*, read-only
observation (through a ``forward_callback``) and returns only the *final*
wavefield state inside the autograd graph. :func:`deepwave.checkpointing.snapshots`
returns the wavefield state at chosen intermediate outer-step indices **inside
the autograd graph**, so you can build a loss on a mid-propagation wavefield and
differentiate it w.r.t. the model and source. It is a thin layer over
``run_segments`` (it breaks the run at the requested steps and threads the state
through ``initial_state``), so it inherits that function's handling of internal
sub-stepping: snapshots work at any ``step_ratio``, with ``max_vel`` required
when the user ``dt`` needs sub-stepping::

    from deepwave.checkpointing import snapshots

    final, snaps = snapshots(
        deepwave.scalar,
        (v,),                                  # model tensors, positional order
        dx, dt,
        {"source_amplitudes": source_amplitudes},
        [100, 200, 300],                       # outer-step indices to snapshot
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=max_vel, pml_freq=freq,
    )
    # A loss on an intermediate wavefield, differentiable w.r.t. v and the source:
    (snaps[200]["wavefield_0"] ** 2).sum().backward()

``final`` is the whole propagation's output (all receiver amplitudes and the
final state), exactly as a direct call would return. ``snaps`` maps each
requested step to a ``{wavefield_name: tensor}`` dict of that step's wavefield
state. The snapshot at step ``s`` is the wavefield *after* ``s`` outer steps --
bit-identical to what a ``forward_callback`` observes at ``step == s`` (in the
"pml" view). This surfaces wavefield-space objectives such as wavefield matching,
field regularisation, and WRI-style formulations.

:func:`deepwave.checkpointing.snapshots` and its result type are
documented in :doc:`reference_utilities`.


Skipping points: the sentinel and the masks
-------------------------------------------

A point whose coordinates are *all* :data:`deepwave.IGNORE_LOCATION` is
ignored: it injects nothing and records zeros. A point with only *some*
coordinates set to the sentinel raises. Compare against
:data:`deepwave.IGNORE_LOCATION`; do not hard-code its value.

Every location argument has a matching boolean ``*_mask`` argument that says
the same thing more legibly. It has shape ``[shot, point]``, dtype
``torch.bool``, and **``True`` means active**::

    # These two calls produce bit-identical output.
    locations_with_sentinel = locations.clone()
    locations_with_sentinel[0, 1, :] = deepwave.IGNORE_LOCATION
    a = deepwave.scalar(v, dx, dt, source_amplitudes=amplitudes,
                        source_locations=locations_with_sentinel)

    mask = torch.tensor([[True, False, True]])
    b = deepwave.scalar(v, dx, dt, source_amplitudes=amplitudes,
                        source_locations=locations, source_mask=mask)

They are one mechanism, not two: a mask is converted to the sentinel
internally, at one place, before any bounds checking. A ``False`` row's
coordinates are therefore ignored entirely -- they need not be valid, in
range, or unique. In the mapping form the mask is the optional trailing
element of the value: ``sources={"u": (amplitudes, locations, mask)}``,
``receivers={"u": (locations, mask)}``.

Supplying both a mask and a sentinel-containing locations tensor for the
same component raises, rather than silently combining two statements of
intent that may disagree. A mask cannot be combined with a
:class:`deepwave.Hicks` object either; disable points inside a ``Hicks`` by
putting the sentinel in the locations it is built from.

Specifying sources and receivers
--------------------------------

Every propagator takes its survey as **channel-keyed mappings**, which is the
canonical form::

    out = deepwave.elastic(
        lamb, mu, buoyancy, dx, dt,
        sources={"y": (amplitudes, source_locations)},
        receivers={"y": receiver_locations, "x": receiver_locations},
    )

A *channel* is the physical quantity a source injects or a receiver records.
Each propagator's channels are:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Propagator
     - Source channels
     - Receiver channels
   * - ``scalar``, ``scalar_born``
     - ``"u"``
     - ``"u"``
   * - ``acoustic``, ``acoustic_born``
     - ``"volume_rate"``, ``"z"`` (3-D), ``"y"``, ``"x"``
     - ``"p"``, ``"z"`` (3-D), ``"y"``, ``"x"``
   * - ``viscoacoustic``
     - ``"pressure_rate"``, ``"z"`` (3-D), ``"y"``, ``"x"``
     - ``"p"``, ``"z"`` (3-D), ``"y"``, ``"x"``
   * - ``elastic``, ``elastic_vti``, ``elastic_born``
     - ``"pressure_rate"``, ``"z"`` (3-D), ``"y"``, ``"x"``
     - ``"p"``, ``"z"`` (3-D), ``"y"``, ``"x"``
   * - ``acoustic_vti``
     - ``"pressure_rate"``, ``"z"`` (3-D), ``"y"``, ``"x"``,
       ``"sigma_v"``, ``"sigma_h"``
     - ``"p"``, ``"z"`` (3-D), ``"y"``, ``"x"``, ``"sigma_v"``,
       ``"sigma_h"``

A source channel is named for the quantity it injects. ``"volume_rate"`` has
units of 1/time and ``"pressure_rate"`` units of pressure/time -- roughly a
factor of the bulk modulus apart -- and the letter ``"p"`` used to name both,
depending on the propagator. It is still accepted as a source name, with a
:class:`deepwave.DeepwaveFutureWarning` naming the replacement, and is
removed in ``0.4.0``. See :doc:`units`.

``"p"`` is unchanged as a *receiver* channel: there it has always meant a
pressure, on every propagator that records one. ``"volume_rate"`` and
``"pressure_rate"`` are source names only.

A ``sources`` value is ``(amplitudes, locations)``, or
``(amplitudes, locations, mask)`` to add an activity mask (see below). A
``receivers`` value is a locations object, or ``(locations, mask)``. The
Born propagators additionally take ``bg_receivers=`` for the receivers that
record the background wavefield; their ``receivers=`` is the **scattered**
data, matching :attr:`deepwave.PropagatorOutput.receivers` and the ``sc_``
flat spelling.

The recorded data comes back keyed by the same channel names::

    out.receivers["y"]        # the scattered data on a Born propagator
    out.bg_receivers["y"]     # the background data (Born only)

Every channel of the propagator is present in those mappings; a channel with
no receiver locations holds an empty tensor.

**The flat keywords are permanent.** ``source_amplitudes_y=``,
``source_locations_y=``, ``receiver_locations_y=`` and their siblings are
sugar for exactly the same thing and will not be removed. One rule generates
every one of them from the propagator's own channel and state names (see
:ref:`flat-keywords`), so a propagator added later has the same set, derived
the same way. Giving one channel in *both* forms raises, since there would
be no way to tell which was meant.

Off-grid sources and receivers
------------------------------

Source and receiver locations are given in **grid cells** relative to the model
origin. An integer (``torch.long``) location is an exact array index and is
injected/recorded exactly (bit-for-bit identical across versions). A
floating-point location instead places the point off-grid: **all nine**
propagators interpolate it onto the grid using the method of Hicks (a
Kaiser-windowed sinc), on every source and receiver channel they take. On
the seven staggered-grid propagators (everything except ``scalar`` and
``scalar_born``) the interpolation is about the nodes of the channel's own
field component, so a coordinate that names a point in space means the same
point whichever component records it: the half-cell offset of a velocity
component is applied for you, and you never add it yourself. This is
controlled by the
``location_interpolation`` argument -- ``"auto"`` (the default) takes the
exact path for an integer-valued float on a node and interpolates otherwise;
``"hicks"`` forces interpolation; ``"none"`` selects the exact integer path
only and **raises** ``ValueError`` if a float coordinate is off-node -- and by
``location_interpolation_halfwidth``.
:func:`deepwave.to_cells` converts physical-unit coordinates (metres, say)
into the float grid-cell coordinates these arguments expect, given
``grid_spacing`` and the physical coordinate of cell 0 (``grid_origin``)::

    src = torch.tensor([[[10.3, 12.7]]])       # off-grid, in cells
    out = deepwave.scalar(
        v, 5.0, 0.001,
        source_amplitudes=source, source_locations=src,
        receiver_locations=torch.tensor([[[20.4, 11.1]]]),
    )

The tolerance for "on a node" is :data:`deepwave.ON_NODE_TOLERANCE`
(``1e-5`` cells). It is the one value that decides all three of the routing
above, ``"none"``'s rejection, and :class:`deepwave.Hicks`' default ``eps``.

That matters when your geometry is *arithmetically* on-node but does not
come back that way: ``175.0 / 12.5`` is exactly 14, but the same division in
float32 over a whole survey can land a few ulp away, which routes an exactly
on-node survey through the interpolated path and changes the numbers. Pass
``snap=True`` to clean that up::

    coords = torch.tensor([[[175.0, 250.0]]])          # metres
    cells = deepwave.to_cells(coords, 12.5, snap=True)  # exactly [[14., 20.]]

``snap`` removes floating-point error, it does not round: a coordinate that
is genuinely 0.4 cells off a node comes back unchanged and is still
interpolated. Pass a float instead of ``True`` for a different tolerance.
The default, ``snap=False``, is the plain division, bit for bit.

The staggered propagators (``acoustic``, ``elastic``, ``viscoacoustic``,
``elastic_vti``) are handled per-component: pressure / normal-stress points
live on the integer grid, while a
velocity component is staggered half a cell in its own dimension. The binding
convention is that **integer index N and float N+offset denote the same physical
point**, so a ``vx`` source at integer ``3`` is the same as one at float ``3.5``.
You can therefore move it continuously with fractional coordinates and it
stays consistent with the integer placement. The spread (source) and gather (receiver)
are differentiable, so gradients flow to the source amplitudes and model.

For full control — dipole sources/receivers, a custom component ``offset``, or
free surfaces — pass a :class:`deepwave.Hicks` object as
the location argument instead of a tensor; the propagator then performs the
spread/gather for you and uses the object's ``offset`` as given (it does not
override it with the component's default).

Differentiable positions
^^^^^^^^^^^^^^^^^^^^^^^^

The source and receiver **positions themselves** can also be differentiable:
the Kaiser-windowed-sinc weights are a smooth, analytic function of the
sub-cell offset, so the gradient of a loss with respect to a point's
coordinates is well defined. This enables gradient-based survey design
(optimising receiver placement) and source localisation (descending on a
mislocated source's coordinates until its predicted traces match observed
ones). Set ``requires_grad_(True)`` on the float location tensor and
gradients flow back to it through a propagator call::

    src = torch.tensor([[[17.0, 23.0]]], dtype=torch.double).requires_grad_(True)
    out = deepwave.scalar(
        v, dx, dt,
        source_amplitudes=source, source_locations=src,
        receiver_locations=rec,
    )
    loss = ((out.receiver_amplitudes - observed) ** 2).sum()
    loss.backward()   # src.grad is d(loss)/d(position), in units of 1/cells

A float location tensor with ``requires_grad=True`` is always routed through
interpolation in ``"auto"`` mode (even when it happens to lie exactly on a
node), because only the interpolated path can carry position gradients. For
manual control, :class:`~deepwave.Hicks` accepts
``differentiable=True``, and its ``source``/``receiver`` methods then
backpropagate to the positions::

    hicks = Hicks(src, dtype=torch.double, differentiable=True)
    out = deepwave.scalar(
        v, dx, dt,
        source_amplitudes=hicks.source(amp),
        source_locations=hicks.get_locations(),
        ...
    )

Notes:

- **Cell crossings.** The integer support cells are piecewise-constant in
  position (and detached), so the gradient is exact within a cell. At a node
  crossing, monopole interpolation remains continuous but its position
  gradient can jump. Dipole interpolation can jump in value. Keep position
  optimisation within a cell when a continuous derivative is required.
- **Precision.** ``float64`` positions (and propagation) are recommended for
  position-gradient work, as the gradients can be small relative to the loss.
- **Restrictions.** Free/rigid surfaces raise ``NotImplementedError`` with
  ``differentiable=True`` (the image-folding branch is not written
  differentiably); dipoles and custom ``offset`` are supported. Points whose
  coordinates are the ``IGNORE_LOCATION`` sentinel receive exactly zero
  position gradient. The default (``differentiable=False``) path is
  unchanged, bit-for-bit.
- **One backward per instance.** A ``differentiable=True``
  :class:`~deepwave.Hicks` instance holds its
  interpolation weights in the autograd graph, which ``backward()`` frees. In an
  optimisation loop, construct a **fresh** ``Hicks`` instance each iteration
  (build it from the current positions inside the loop — the pattern shown
  above), or pass ``retain_graph=True`` if you deliberately reuse one instance
  across ``backward()`` calls; otherwise the second backward raises the usual
  "backward through the graph a second time" error.

DAS strain-rate receivers
-------------------------

The :mod:`deepwave.das` module models distributed acoustic sensing (DAS)
fibre measurements as differentiable receiver-side post-processing of the
elastic (or acoustic) propagator's particle-velocity receivers:
:func:`deepwave.das.fiber_gauge_locations` turns an ordered polyline of
fibre points into per-channel gauge-endpoint pairs and unit tangents, and
:func:`deepwave.das.strain_rate_from_velocities` combines the velocities
recorded at those endpoints into gauge-averaged axial strain rate. The two
velocity arguments are mappings keyed by the same channel names the
propagators use -- ``{"y": receiver_amplitudes_y, "x":
receiver_amplitudes_x}`` in 2-D, with ``"z"`` first in 3-D -- so which
component is which is stated rather than inferred from the order of a
sequence. :func:`deepwave.das.helical_fiber_response` combines the
cable-frame normal strain rates of a helically wound fibre at a
keyword-only ``wrap_angle`` in **radians**, like every other angle in
Deepwave. See :doc:`das` for the measurement physics, a worked recipe,
units, and the API reference.

Time-dispersion correction
--------------------------

Explicit second-order leapfrog time stepping introduces temporal dispersion:
each temporal frequency propagates as if it were the slightly lower
:math:`(2/\Delta t)\sin(\omega\Delta t/2)`. Raising the spatial ``accuracy``
order does nothing to it — only a smaller ``dt`` (a larger Courant safety
margin) reduces it. The optional :mod:`deepwave.time_dispersion` module applies
the Koene–Robertsson frequency remappings as an approximate correction:
:func:`deepwave.time_dispersion.correct_source` pre-transforms the source
wavelet before modelling and
:func:`deepwave.time_dispersion.correct_receivers` post-transforms the recorded
data afterwards. Both are differentiable, batched, and take the propagator's
**internal** time step ``internal_dt = dt / step_ratio`` (the step that governs
the dispersion), which :func:`deepwave.time_dispersion.cfl_internal_dt` derives
from the same inputs the propagator uses. See :doc:`time_dispersion` for the
derivation, the transform directions, and the API reference, and
:doc:`example_time_dispersion` for a runnable example.

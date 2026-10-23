Segmented propagation and whole-record resampling
=================================================

:mod:`deepwave.checkpointing` carries the complete integration state between
segments. For a stationary experiment, it also preserves the sampling phase
and applies source and receiver resampling to the whole record. Under matching
numerical settings, this makes segmented results comparable with a single
combined call; floating-point reduction order, backend, setup, callbacks and
the selected sampling/compression policy still determine the attainable
agreement. It is not an unconditional bitwise guarantee. This page explains
why resampling has to live in the segment driver.

Where a naive split loses exactness
-----------------------------------

Splitting a propagation into segments and threading the full returned state
(wavefields and constitutive/PML auxiliaries) back in through ``initial_state``
preserves the integration boundary. At ``step_ratio == 1`` there is no
source/receiver rate conversion, so this can closely reproduce a combined
run if all other settings and material values agree.

When ``step_ratio > 1`` a naive split does not. Deepwave upsamples the
*source* amplitudes from the user time step to the internal time step with a
global FFT (:func:`deepwave.discretisation.upsample`); upsampling a *slice*
of the source is not the same as slicing the upsampled source, so the
injected source -- and hence the whole propagation -- differs across a
segment boundary.

Even without sources, separately resampling receiver records can differ from
resampling their concatenation: the FFT window, taper and edge handling see a
different record. Thus source-free state continuation and equality of final
user-rate receiver records are distinct claims.

How ``run_segments`` restores it
--------------------------------

:func:`deepwave.checkpointing.run_segments` resolves ``step_ratio`` itself
and, for a run of two or more segments at ``step_ratio > 1``, performs the
resampling once for the whole trace: it upsamples every source tensor a
single time, drives every segment at the internal time step (so each
segment's own ``step_ratio`` is 1), and downsamples the time-concatenated
receiver output once at the end. This avoids segment-local FFT edge effects.
Agreement with a combined call also requires the same resolved setup, native
history-sampling phase, model values, backend and storage fidelity. Gradients
can differ through accumulation order even when forward fields agree closely.

An omitted ``max_vel`` is resolved through the propagator's own preparation
before splitting the record, including heterogeneous and attenuation bounds.
The resampling therefore uses the same speed as the direct call. :func:`deepwave.wavefields.measure` still
requires ``step_ratio == 1`` for a segmented measurement, because its
streaming DFT is defined on user-rate samples.
:func:`deepwave.checkpointing.snapshots` inherits the whole-trace resampling
from ``run_segments``.

Why it cannot be automated inside one segment
---------------------------------------------

A caller driving segments in its own outer loop -- calling ``run_segments``
or a propagator once per segment -- must still pre-upsample the source
itself and downsample the concatenated receiver output at the end. That
cannot be automated from inside a single segment's call: correct upsampling
of one chunk needs the samples of every other chunk, because the FFT is
global, and correct receiver downsampling needs the full concatenated
internal-rate trace, which does not exist until after the last segment. The
resampling must bracket the whole loop, which only code that owns the loop
can do.

That is why the automation lives in ``run_segments``, which is handed the
whole trace, and why ``checkpoint=True`` exists: it moves the per-segment
``torch.utils.checkpoint`` loop inside ``run_segments``, so checkpointed runs
get the same automation.

Rejected: a source/receiver wrapper type
----------------------------------------

A source/receiver "wrapper type" carrying the whole trace through a user's
own loop was considered and rejected. ``torch.utils.checkpoint`` tracks
gradients only through plain ``Tensor`` inputs and ``Tensor``/tuple-of-\
``Tensor`` outputs -- the reason :class:`deepwave.PropagatorOutput`
subclasses ``tuple`` -- so a wrapper crossing a checkpoint boundary is stored
as an opaque non-tensor whose embedded whole-trace tensor keeps the upstream
graph alive across every segment: exactly what checkpointing exists to avoid.

A wrapper unwrapped *before* the boundary avoids that, but then the upsample
has already happened outside the loop, which is the three-line manual pattern
with extra machinery around it. Neither form pays for itself, so the manual
pattern is documented instead and the automation is offered only where it can
be done correctly, inside ``run_segments``.

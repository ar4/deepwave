Wavefield-space tools
=====================

.. py:module:: deepwave.wavefields

Deepwave is, at its core, a differentiable *shot-data* operator: you inject point
sources and read receiver traces, and gradients flow back to the model, the
source, and the initial state. The :mod:`deepwave.wavefields` module widens that
surface toward a differentiable *wavefield* operator, for research workflows that
need to force, reduce, or observe the field itself. Everything here is
additive, pure Python, built on the public propagator interface (and, for the
segment drivers, on :func:`deepwave.checkpointing.run_segments`); no
compiled-backend or ABI change is involved, and the existing propagators are
unaffected.

There are three capabilities: volumetric source fields, on-the-fly reductions,
and boundary record/reinject. The first is wired into :func:`deepwave.scalar` as
the ``source_field`` keyword; the others are drivers you call on any propagator.

.. contents::
   :local:


Volumetric / extended source fields
------------------------------------

A point source injects forcing at a single grid cell. A *volumetric* (extended)
source injects a forcing field ``f(shot, t, *cell)`` at every cell of a region on
every time step -- the building block of wavefield-reconstruction inversion (WRI),
source encoding, and adjoint-source experiments. Convert the field (shape
``[shot, nt, *spatial]``) into ordinary point sources with
:func:`~deepwave.wavefields.source_field_to_sources`, optionally merge it
with point sources of your own with
:func:`~deepwave.wavefields.merge_point_sources`, and pass the result to any
propagator::

    import torch
    import deepwave
    from deepwave.wavefields import merge_point_sources, source_field_to_sources

    # A dense forcing over the whole model, differentiable w.r.t. the field.
    source_field = torch.zeros(n_shots, nt, ny, nx, requires_grad=True)
    amplitudes, locations = source_field_to_sources(source_field)
    # Optional: add explicit point sources on cells the field does not use.
    amplitudes, locations = merge_point_sources(
        amplitudes, locations, source_amplitudes, source_locations
    )
    out = deepwave.scalar(
        v, dx, dt,
        sources={"u": (amplitudes, locations)},
        receivers={"u": receiver_locations},
        pml_freq=freq,
    )
    out.receiver_amplitudes.pow(2).sum().backward()   # d/d source_field

The field is implemented by mapping each (active) cell to one point source at that
integer grid node. On a regular grid an on-node source is an *exact* single-cell
forcing -- the Hicks windowed-sinc interpolation collapses to a Kronecker delta at
an integer location -- so the mapping introduces no smoothing and is bit-exact
against building the per-cell sources by hand. Because each cell is used exactly
once, the field's sources are automatically unique: the unique-cell constraint
that governs hand-placed point sources never applies to field forcing. If you also
pass explicit ``source_amplitudes``/``source_locations``, the two sets are
concatenated and their cells must be disjoint (fold any coincident point source
into the field).

Two options restrict the field to a sub-region, keeping the source count (and
cost) down:

* ``origin`` -- the model cell index of the field's first cell, so a
  field of shape ``[shot, nt, *region_shape]`` covers a rectangular sub-box.
* ``mask`` -- a boolean tensor of shape ``region_shape`` selecting
  which cells are active (a non-rectangular, region-masked source).

Both helpers are propagator-agnostic, so acoustic and elastic users build the
same per-cell sources and pass them on whichever channel they want.

.. autofunction:: deepwave.wavefields.source_field_to_sources

.. autofunction:: deepwave.wavefields.merge_point_sources


Differentiable on-the-fly reductions
------------------------------------

Frequency-domain FWI and related objectives need only a handful of Fourier
components of the receiver data, not the whole time series. :func:`deepwave.wavefields.measure`
runs a propagator segment-by-segment and accumulates a declarative reduction --
a :class:`~deepwave.wavefields.RunningDFT` at chosen frequencies -- as the segments
complete, returning the result **inside the autograd graph**. The full trace is
never retained across segments. DFT accumulators cost ``O(n_freqs)`` per
receiver; peak memory also includes a segment's receiver trace and propagation
history. The accumulated transform equals an offline FFT/DFT of the
assembled trace to floating-point round-off, in both value and gradient::

    from deepwave.wavefields import measure, RunningDFT

    result = measure(
        deepwave.scalar,
        (v,), dx, dt,
        {"source_amplitudes": source_amplitudes},
        [RunningDFT(frequencies=[5.0, 9.0, 13.0], on="u")],
        n_segments=8,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=max_vel, pml_freq=freq,
    )
    dft = result.measurements[0]     # [shot, receiver, n_freq], complex, in-graph
    (dft.abs() ** 2).sum().backward()

``measure`` returns a :class:`~deepwave.wavefields.MeasureResult`: a named
``(measurements, output)`` pair, so ``result.measurements`` and
``measurements, output = measure(...)`` both read naturally.
``measurements`` holds one complex tensor per measurement, in the order of
the ``measurements`` argument. ``output`` is ``None`` unless you ask for it
with ``return_output=True``, which populates it with the run's final
:class:`~deepwave.PropagatorOutput` -- the receiver amplitudes and final
state a direct call would have returned, alongside the reduction::

    result = measure(
        deepwave.scalar,
        (v,), dx, dt,
        {"source_amplitudes": source_amplitudes},
        [RunningDFT(frequencies=[5.0, 9.0, 13.0], on="u")],
        n_segments=8,
        return_output=True,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=max_vel, pml_freq=freq,
    )
    dft = result.measurements[0]
    print(result.output.info.step_ratio)          # the run's own metadata
    final_wavefield = result.output.wavefield_0   # in the autograd graph

The shape of the return value does not depend on ``return_output``: the pair
is always a pair, so unpacking code does not have to change when you turn the
output on. Keeping the output does cost the final state's memory, which is
why it is off by default.

``on`` is required, and names either a **channel** of the propagator
(``"u"`` on :func:`deepwave.scalar`, ``"p"`` on :func:`deepwave.elastic`) or
a full output field name (``"receiver_amplitudes_p"``). Naming a channel is
the better habit: it is stable, and it is unambiguous on a Born propagator,
where "the first receiver output" is the *background* data.

Receiver-amplitude measurements work at any ``n_segments`` (the segment-wise
partial DFT sums are exact regardless of where the boundaries fall). A measurement
whose ``on`` names a *wavefield* (e.g. ``RunningDFT([...], on="wavefield_0")``)
needs the field at every outer step, so it forces per-step segmentation
(``n_segments = nt``, reported by a warning): it is exact but has Python-per-step
overhead, so prefer receiver measurements where you can.

``on`` may also name one of the keys of the propagator's ``auxiliary_state``
(e.g. ``RunningDFT([...], on="psiy_m1")``), which is treated the same way as a
wavefield target and forces the same per-step segmentation. Those keys are the
internal auxiliary-field names :doc:`/callback_names` lists, and they keep
working (see :doc:`/stability`), but what they hold is the absorbing
boundary's own state rather than physics: a measurement built this way is a
debugging tool rather than a stable part of an objective function, because the
numbers in it follow the absorbing-boundary implementation.

.. note::

   **Exactness requires** ``step_ratio == 1`` **whenever the run is segmented**
   (the resolved ``n_segments > 1``, including the per-step wavefield path).
   ``measure`` raises a ``ValueError`` otherwise (reduce ``dt``): its running
   DFT is accumulated on user-rate samples, so, unlike
   :func:`deepwave.checkpointing.run_segments`, it cannot simply run its
   segments at the internal time step when the user ``dt`` needs internal
   sub-stepping. A single-segment (``n_segments == 1``) receiver measurement
   runs at any ``step_ratio``.

.. autoclass:: deepwave.wavefields.RunningDFT
   :members:

.. autofunction:: deepwave.wavefields.measure

.. autoclass:: deepwave.wavefields.MeasureResult


Boundary recording and trace replay
-----------------------------------

Use dense receivers to record traces at the cells of a rectangular surface,
then replay those traces as source amplitudes at the same cells::

    from deepwave.wavefields import record_boundary, reinject_boundary

    box = [(30, 55), (30, 55)]
    recording, surface = record_boundary(
        deepwave.scalar, (v,), dx, dt,
        source_amplitudes, source_locations, box,
        pml_freq=freq,
    )
    replay = reinject_boundary(
        deepwave.scalar, (v,), dx, dt,
        recording, surface,
        receiver_locations=interior_locations, pml_freq=freq,
    )

``record_boundary`` selects its receiver quantity with ``channel=`` and its
excitation independently with ``source_channel=``. ``reinject_boundary`` uses
``channel=`` to select the source channel. Names resolve through
:func:`deepwave.spec`::

    recording, surface = record_boundary(
        deepwave.elastic, (lamb, mu, buoyancy), dx, dt,
        source_amplitudes, source_locations, box,
        source_channel="pressure_rate", channel="p", pml_freq=freq,
    )
    replay = reinject_boundary(
        deepwave.elastic, (lamb, mu, buoyancy), dx, dt,
        recording, surface, channel="pressure_rate", pml_freq=freq,
    )

For recording, ``channel=None`` selects the first receiver channel and
``source_channel=None`` selects the first source channel for the box dimension.
For replay, ``channel=None`` selects the first source channel for the surface
dimension. A Born source can therefore be selected separately from a scattered
receiver, for example ``source_channel="volume_rate", channel="p"`` on
``acoustic_born`` (where ``channel="p"`` records scattered pressure).

:func:`~deepwave.wavefields.record_boundary` returns a
:class:`~deepwave.wavefields.BoundaryRecording`, a named
``(recording, surface_locations)`` pair, so the unpacking above and
``rec.recording``/``rec.surface_locations`` are equivalent.

This helper multiplies the supplied traces by ``amplitude_scale`` and uses
them as source amplitudes at the supplied surface cells. It applies no
boundary-integral weights, normal derivatives, dipole terms, or automatic
conversion from recorded-field units to source units. It does not in general
reconstruct the original field or preserve its phase or amplitude.

The default ``amplitude_scale=1.0`` injects the traces unchanged. A different
value scales the injected amplitudes linearly; it is not a general field
reconstruction calibration. Use the same explicit ``pml_freq`` in recording
and replay runs when comparing them, so the absorbing boundaries are
configured consistently.

.. autofunction:: deepwave.wavefields.boundary_surface_locations

.. autofunction:: deepwave.wavefields.record_boundary

.. autoclass:: deepwave.wavefields.BoundaryRecording

.. autofunction:: deepwave.wavefields.reinject_boundary

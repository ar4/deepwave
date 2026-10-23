Scaling 3-D FWI past a single GPU's memory
===========================================

:doc:`troubleshooting`'s "Running out of memory" section lists the individual
memory levers (shot batching, ``model_gradient_sampling_interval``,
``storage_mode``, ``storage_compression``) and :doc:`example_distributed`
covers splitting shots across GPUs with ``DataParallel``/
``DistributedDataParallel``. This page shows what happens when you need
**all** of those at once -- a 3-D problem too large for one GPU even after
applying any single lever on its own -- with a measured decision table
instead of arithmetic projections, and points at
:doc:`example_flagship_3d` as a complete, working composition.

.. contents:: :local:

The four levers, composed
--------------------------

- **DDP shot-splitting.** :class:`torch.nn.parallel.DistributedDataParallel`
  divides the shots between the two GPUs of the machine measured below,
  halving every per-GPU term
  below.
- **Shot batching.** Within a rank's shots, propagate a smaller group at a
  time and accumulate the gradient over groups (as in :doc:`example_rtm`),
  scaling the per-call cost down further.
- **``storage_mode``/``storage_compression``.** Offload the sampled
  wavefield history kept for the model gradient to host memory
  (``storage_mode="cpu"``) and/or compress it. Compression is lossy in the
  model gradient; see :func:`deepwave.scalar` for what it costs.
- **``checkpoint_segments``.** Every propagator takes this argument: it
  splits the time axis into that many segments and checkpoints all but the
  last, bounding backward-pass memory to roughly one segment's history plus
  the segment auxiliary states instead of the whole trace's history, at the
  cost of one extra forward pass per checkpointed segment. It is shorthand
  for a :func:`deepwave.checkpointing.run_segments` call, which spells the
  same thing as ``n_segments=`` plus a separate boolean ``checkpoint=True``.
  The two names are not interchangeable: ``checkpoint_segments`` is a segment
  *count* on a propagator, ``checkpoint`` is a *boolean* on the driver. This
  page says ``checkpoint_segments`` throughout, and ``n_segments`` only where
  it means the driver's argument.

Each lever is independently useful, but they compose: DDP and shot batching
shrink the *shot* dimension a call sees, ``checkpoint_segments`` shrinks the
*time* dimension, and ``storage_mode``/``storage_compression`` shrink
the *bytes per stored step*. :doc:`example_flagship_3d` uses all four
together.

Measured decision table
-------------------------

Scalar propagator, accuracy 4, float32, forward + backward, one
homogeneous-ish model (values do not depend on the model's content, only its
shape and the call parameters). Measured on a two-GPU workstation with
31 GB of host RAM and no swap -- ``cuda:0`` (RTX 4000
Ada, 20 GB) and ``cuda:1`` (RTX A4000, 16 GB) gave identical GPU-peak figures
to the precision shown wherever both were checked (rows 4 and 7 below),
consistent with :doc:`troubleshooting`'s "pure allocation-size effect"
finding for this propagator -- so most rows below were only measured once,
on ``cuda:0``, which has more headroom for the rows that are meant to fail.
"Shots/call" is the shot count passed to a single propagator call (after DDP
splitting and, where used, shot batching); "Segs" is that call's
``checkpoint_segments`` (1, or an omitted argument, means no checkpointing).
The two OOM rows report the CUDA allocator's attempted
allocation size at the point of failure (the run never completed, so there
is no peak to report); every other row is ``torch.cuda.max_memory_allocated``
after a real, completed forward+backward pass, plus host RSS
(``resource.getrusage().ru_maxrss``) for ``storage_mode="cpu"`` rows.

.. list-table::
   :header-rows: 1
   :widths: 8 8 10 8 10 10 12 12 22

   * - Grid
     - nt
     - Shots/call
     - Segs
     - storage_mode
     - compression
     - GPU peak
     - Host RSS
     - Result
   * - 40\ :sup:`3`
     - 200
     - 8 (no split)
     - 1
     - device
     - --
     - 4.39 GB
     - --
     - Fits with no levers at all.
   * - 100\ :sup:`3`
     - 600
     - 8 (no split, single GPU)
     - 1
     - device
     - --
     - OOM (needed ~57.3 GB)
     - --
     - Neither card has 57 GB; this is the "hurts" baseline.
   * - 100\ :sup:`3`
     - 600
     - 4 (DDP split only, single GPU)
     - 1
     - device
     - --
     - OOM (needed ~28.7 GB)
     - --
     - DDP alone is not enough at this size.
   * - 100\ :sup:`3`
     - 600
     - 4 (DDP split)
     - 5
     - device
     - --
     - 14.33 GB
     - 1.48 GB
     - Fits cuda:0 (72%); only ~14% headroom on cuda:1 (16 GB) -- checkpointing alone is not comfortable on the smaller card.
   * - 100\ :sup:`3`
     - 600
     - 4 (DDP split)
     - 5
     - cpu
     - False
     - 3.05 GB
     - 17.86 GB
     - GPU now trivial, but host RSS alone is more than half of that machine's 31 GB *per rank* -- two DDP ranks would risk the no-swap ceiling.
   * - 100\ :sup:`3`
     - 600
     - 4 (DDP split)
     - 5
     - cpu
     - True
     - 3.08 GB
     - 5.57 GB
     - Compression cuts this rank's host RSS ~3.2x (the offloaded history itself by more, since about 1.5 GB of that RSS is the process rather than the buffer); comfortable on both axes. It is lossy in the model gradient -- see :func:`deepwave.scalar`.
   * - 100\ :sup:`3`
     - 600
     - 2 (DDP split + batch)
     - 5
     - cpu
     - True
     - 1.55 GB
     - 3.53 GB
     - :doc:`example_flagship_3d`'s actual per-call cost; ample margin on either card.
   * - 100\ :sup:`3`
     - 600
     - 2 per rank, both ranks concurrently
     - 5
     - cpu
     - True
     - ~1.7 GB each card
     - well under 31 GB combined
     - The real, literal :doc:`example_flagship_3d` run: both GPUs at once, 2 epochs, loss decreased, finished in about a minute.

Reading the table
-------------------

The baseline (row 2) needs roughly **37x** the memory the fully-composed
recipe (row 7) uses -- that gap is what "a size that hurts" means here: not
the largest grid that can be coaxed onto one card, but a size that
unambiguously does not fit unmodified and comfortably does once the levers
are composed (each card sits under 10% utilisation in the final row, well
inside the 80-85% margin worth leaving for a real optimiser run, which
touches more of the graph than this table's fixed-velocity measurement
pass).

Two findings worth designing around, not just reading past:

* **Checkpointing alone is not enough headroom on the smaller card.** Row 4
  (14.33 GB) is 72% of cuda:0's 20 GB but 86% of cuda:1's 16 GB --
  above the comfortable range, and :doc:`troubleshooting` already reports
  that real runs measure higher than any static estimate, so this is not a
  configuration to ship without also offloading storage.
* **``storage_mode="cpu"`` without compression can cost more host RAM than
  the whole point of offloading was meant to save.** Row 5's 17.86 GB host
  RSS, doubled across two concurrent DDP ranks, would use more than that
  machine's entire 31 GB with no swap -- offload and compression are a package deal at
  this scale, not two independently optional levers.

``estimate_memory`` under checkpointing: still a lower bound
-------------------------------------------------------------------

:doc:`troubleshooting`'s "Measured accuracy of ``estimate_memory``" table
already documents that the estimate under-predicts real peak memory
everywhere it was checked (device/uncompressed: -4% to -20%; compression,
high sampling intervals, or offload: up to -70%). That table does not
exercise checkpointing.

:func:`~deepwave.estimate_memory` does have a model of it:
``n_segments=k`` divides the stored history by ``k`` and adds a
``auxiliary_state_bytes`` term for the segment auxiliary states held alive --
``k`` of them where a model gradient is being computed, one where none is.
Without that argument the estimate assumes the whole trace is stored
at once and over-predicts a checkpointed run enormously (29.10 GB against a
measured 14.33 GB for the first row below). With it, the estimate becomes a
much closer -- but still low -- lower bound:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 20 18

   * - Row (levers)
     - Estimated, ``n_segments=5``
     - Measured (GPU + host)
     - Relative error
     - Error without the term
   * - checkpoint only (device)
     - 7.05 GB
     - 14.33 GB
     - -51%
     - -57%
   * - checkpoint + cpu offload
     - 7.07 GB
     - 20.91 GB (3.05 + 17.86)
     - -66%
     - -70%
   * - checkpoint + cpu + compression
     - 2.79 GB
     - 8.65 GB (3.08 + 5.57)
     - -68%
     - -78%
   * - full recipe (+ batching)
     - 1.39 GB
     - 5.08 GB (1.55 + 3.53)
     - -73%
     - -81%

The measured column is the same measurement as the decision table above. The
"error without the term" column is what the estimate gave before it had one,
when the only way to account for checkpointing was to hand it a per-segment
``nt`` of ``600 / 5 = 120`` and ignore the auxiliary states. An independent
single-process re-run of the same four configurations (one GPU, no DDP)
measured 14.62, 22.07, 9.21 and 5.39 GB, i.e. within 7% of the figures above,
so the relative errors are not an artefact of one run.

**Read the result honestly: the term helps, and it is not enough.** It
recovers between 4 and 10 percentage points of a gap that was -57% to -81%,
and leaves one of -51% to -73%. What it does not model accounts for the rest:
the fixed per-call overhead that :doc:`troubleshooting`'s own table
identifies (allocator fragmentation, the compiled backend's scratch, the
host-side double buffer for offload, and roughly 1.5 GB of process baseline
in each host-RSS figure here) is unmodelled and grows in relative terms
exactly as the levers shrink the modelled ones -- which is why the row with
the fewest levers has the smallest relative error (-51%, device only) and the
row with the most has the largest (-73%, the full recipe).

The auxiliary-state term itself is a floor on the auxiliary states, like every
other term here rather than an exception to them. It sizes each state field
by field: the finite-difference halo is not carried across a boundary, and
the PML memory variables are stored as the two end slabs of the axis they
vary along, not as full grids, so both reductions are priced in. Where a run
cannot use the slab form -- a propagator with no compiled backend, an
explicitly requested Python backend, a non-default ``boundary``, M-PML -- the
real states are larger than the term says, which leaves it a floor there too.
And the count is the number of states genuinely held at once: one per segment
while a model gradient is being computed, because each is an input to a
checkpoint node that autograd keeps until that segment is recomputed, but
exactly **one** when no model requires a gradient, because then there is
nothing to checkpoint and the driver threads a single state forward and drops
the previous one. Scaling that case by the segment count was how this
estimate used to exceed the true peak by more than fourteen times at sixty
segments; it does not now.

So ``total_bytes_lower_bound`` means what it says, at every segment count.
Treat ``estimate_memory`` as a first-pass sizing tool -- use it to rule out
configurations that are obviously too large, and pass ``n_segments=`` so it
is not wrong by a factor when you are checkpointing -- then confirm your
actual target size with a real measurement, as this page's table and
:doc:`example_flagship_3d` both do, before committing to it.
:func:`deepwave.plan` gives the same estimate from the real resolved call
(``deepwave.plan(scalar, ..., checkpoint_segments=5).memory``), which removes
the risk of describing your configuration to the estimator incorrectly, but
does not make it any less of a lower bound.

See also
----------

- :doc:`example_flagship_3d` -- the full, working composition.
- :doc:`troubleshooting` -- each lever explained individually, plus
  ``estimate_memory``'s own measured accuracy table.
- :doc:`example_distributed` -- ``DataParallel``/``DistributedDataParallel``
  shot-splitting on its own.
- :doc:`example_storage` and :doc:`example_checkpointing` -- ``storage_mode``/
  ``storage_compression`` and ``checkpoint_segments`` each in isolation.

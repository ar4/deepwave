Troubleshooting
===============

This page collects fixes for problems that come up often enough to be worth a
dedicated recipe: a diverged (``NaN``) simulation, running out of memory,
progress feedback for long runs, confusing location/units errors, and pointers
to the pages that explain *why* certain symptoms (large edge gradients,
cycle-skipping, numerical dispersion) happen.

.. contents:: :local:

Detecting divergence (NaN) early
---------------------------------

A simulation can diverge (produce ``NaN``/``Inf`` values) if, for example,
``dt``/``grid_spacing`` violate the CFL condition (an excessively large time
step for the grid spacing and velocity), or, during FWI, if an optimiser step
pushes the velocity model to an unstable value.

Every propagator checks for this in its own outputs, and does so by default.
Before returning, it tests every final wavefield and receiver-amplitude
tensor and raises :class:`deepwave.DeepwaveFloatingPointError` if any is
non-finite:

.. code-block:: text

   DeepwaveFloatingPointError: deepwave.scalar produced non-finite output:
   'wavefield_0' contains 2704 non-finite values (nt=60, dt=0.01 for the
   propagation that produced it, which under checkpoint_segments is one
   segment rather than the whole run). This usually means the simulation
   diverged. Check the CFL condition
   (deepwave.discretisation.cfl_condition), that the model contains no zero,
   negative or non-finite values, and that any initial wavefield and the
   source amplitudes are finite. Pass check_output_finite=False to disable
   this check.

That is one check per call, not one per time step: Deepwave still does not
check every internal step, because that would force a device synchronisation
every step, which is prohibitively slow. The output check forces exactly one,
by reducing each output on the device, stacking the per-output flags and
reading a single value back.

**What it costs.** One synchronisation drains the pipeline of asynchronously
launched work, so the cost is a fixed amount per *call* rather than an amount
per element -- which means its relative size is largest for the smallest
calls. Measured with loops of 100 independent :func:`deepwave.scalar` calls
(median of five timings):

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Device
     - 40x40, ``nt=60``
     - 100x100, ``nt=300``
     - 300x300, ``nt=1000``
   * - RTX 4000 Ada
     - 5.93 -> 6.35 ms (+7.1%)
     - 12.25 -> 12.41 ms (+1.3%)
     - 33.72 -> 33.89 ms (+0.5%)
   * - RTX A4000
     - 5.85 -> 6.10 ms (+4.3%)
     - 12.25 -> 12.40 ms (+1.2%)
     - 37.93 -> 38.64 ms (+1.9%)
   * - CPU
     - 6.26 -> 6.42 ms (+2.5%)
     - 19.35 -> 19.52 ms (+0.9%)
     - --

The check also allocates a transient boolean mask, one byte per element of
the output it is testing, freed as soon as the reduction is done. That is
small beside the output itself (one byte against four or eight), but if a run
is already at the edge of its memory it is one more thing to turn off.

Be honest about the first column: a 40x40 grid for 60 steps is a unit test,
not a simulation, and there the check is a real 4-7% of the call. For
anything of a size you would actually invert, it is around or below 1%. Pass
``check_output_finite=False`` if you are running a tight loop of very small
propagations and have measured that it matters, or if the outputs feed
something that detects a ``NaN`` itself. Turning it off changes no numbers:
a healthy run returns bit-identical results either way.

The check covers the tensors a propagator returns. For anything else -- a
gradient, a loss, a model an optimiser has just updated -- use
:func:`deepwave.check_finite`, which raises the same error type:

.. code-block:: python

    out = deepwave.scalar(v, grid_spacing, dt, ...)
    loss.backward()
    deepwave.check_finite(v.grad, "v.grad")

For long runs, even checking at the end of the call still wastes the compute
already spent on a run that diverged early. To catch it at the *first* bad
time step instead, use a ``forward_callback`` (see
:class:`deepwave.CallbackState` and :doc:`example_callback_animation` for the
full callback API) that checks the wavefield every ``callback_frequency``
steps and raises with the step number:

.. code-block:: python

    def forward_callback(state):
        wavefield = state.get_wavefield("wavefield_0")
        if not torch.isfinite(wavefield).all():
            raise FloatingPointError(
                f"Non-finite wavefield at forward step {state.step} "
                f"(t = {state.time:.6g} s). The simulation has "
                "diverged -- check the CFL condition (dt/grid_spacing/"
                "max_vel) and, if this is an FWI iteration, whether the "
                "last optimiser step produced an extreme velocity value."
            )

    out = deepwave.scalar(
        v, grid_spacing, dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=forward_callback,
        callback_frequency=10,  # check every 10 user time steps
    )

Use ``state.get_wavefield(name)`` for whichever of the propagator's wavefield
components you want to check (``"wavefield_0"`` for scalar/scalar_born,
``"pressure_0"``/``"vy_0"``/... for acoustic/elastic -- see each propagator's
API page for the full list), and an analogous ``backward_callback`` if you
need to catch divergence introduced during backpropagation (e.g. from
:meth:`~deepwave.CallbackState.get_gradient`). Re-entering Python every
``callback_frequency`` steps has a real cost (each call is a synchronisation
point and, on GPU, a device-to-host transfer for the ``isfinite`` check).
Prefer a larger ``callback_frequency`` (e.g. 10-50) over checking every
single step once you trust a configuration, and remove the callback entirely
for production runs where divergence is not a live risk.

Running out of memory
----------------------

Before running a large 3-D or many-shot problem, ask the call itself what it
will do. :func:`deepwave.plan` takes exactly the arguments you were about to
pass to the propagator, resolves them the way the run would, and stops before
propagation and gradient-history allocation. Source preprocessing can still
allocate tensors proportional to the time axis:

.. code-block:: python

    p = deepwave.plan(
        deepwave.acoustic, v, rho, dx, dt,
        source_amplitudes_volume_rate=s, source_locations_volume_rate=sl,
        receiver_locations_p=rl,
    )
    print(p.summary())

Because it is the run's own resolution rather than a second opinion about it,
``p.info`` is the very :class:`deepwave.RunInfo` the run would return -- the
internal time step, the step ratio, the inferred PML frequency, the survey
extent, the backend and the resolved storage policy -- and ``p.warnings``
holds Deepwave's own warnings the call would emit, as text, without spending
their once-per-process budget. Warnings from PyTorch or other dependencies
are neither captured nor suppressed. ``p.memory`` is the estimate below, but read off the
resolved run, so its padded shape, sampling interval, storage-slot count and
wavefield count are exact rather than derived from what you typed. Add
``checkpoint_segments=k`` to see what segmenting would cost.

:func:`deepwave.estimate_memory` answers the same question *without* the
tensors, which is what you want when you are sizing a machine for a problem
you have not built yet. It mirrors the two dominant sources of backward-pass
memory (the sampled wavefield history kept for the model gradient, and the
per-thread/per-shot gradient-accumulation scratch buffers) plus a rough
figure for the forward time-loop wavefields:

.. code-block:: python

    est = deepwave.estimate_memory(
        "acoustic",
        model_shape=v.shape, nt=nt, n_shots=n_shots,
        accuracy=accuracy, pml_width=pml_width,
        model_gradient_sampling_interval=1,
        models_requiring_grad=["v", "rho"],
    )
    print(est["total_bytes_lower_bound"] / 1e9, "GB (at least)")

``propagator`` is required -- every term depends on it. Pass
``models_requiring_grad`` too whenever you can: it is what makes the largest
term accurate. ``propagator`` fills in the number
of forward time-loop wavefield buffers (equation-specific), and
``models_requiring_grad`` names the models you will ask for gradients, which
fixes the number of wavefield histories the backward pass stores. That is
**not** one history per model: ``acoustic`` stores one for the bulk modulus
plus one per axis for buoyancy, so a 3-D ``["v", "rho"]`` run stores four,
and ``elastic`` stores one per saved spatial derivative -- nine in 3-D with
``["lamb", "mu", "buoyancy"]``. Naming only the count
(``n_models_requiring_grad``) gives the worst case over any choice of that
many models, an upper bound; naming neither leaves the old
one-history-per-model assumption, which under-predicts those runs several-fold.
Read the docstring for the remaining approximations. If the estimate (or an
actual out-of-memory error) is too large, the levers to reduce it, in roughly
increasing order of effort, are:

* **Shot batching.** Propagate fewer shots at once (smaller ``n_shots`` per
  call, looping over batches) -- this scales every term in the estimate above
  roughly linearly.
* **model_gradient_sampling_interval.** Sampling the wavefield less often for
  the model gradient (e.g. every 2nd or 4th step instead of every step) scales
  ``backward_storage_bytes`` down by the same factor, at the cost of a
  strided-quadrature approximation to the model gradient (the source/receiver/
  wavefield gradients remain exact); see :doc:`accuracy` for the corresponding
  time-discretisation accuracy trade-off.
* **storage_mode.** ``"device"`` (the default) keeps the full sampled
  wavefield history on the compute device; ``"cpu"`` keeps it in pinned host
  memory instead (trading device memory for a slower, host-transfer-bound
  backward pass -- measured on RTX 4000 Ada 20 GB and RTX A4000 16 GB cards,
  near-identical on both, 120^3 grid, nt=200, scalar: device peak 2.81 GB
  -> 0.44 GB, at about 4.1-4.8x the wall-time cost); ``"disk"`` spills it to
  a temporary directory (``storage_path``); ``"none"`` disables
  model-gradient storage entirely (only usable when you do not need that
  gradient). See the propagator API pages (:doc:`reference_scalar`,
  :doc:`reference_acoustic`, :doc:`reference_elastic`) for the full
  ``storage_mode``/``storage_path``/``storage_compression`` options.
* **checkpoint_segments.** ``checkpoint_segments=k`` runs the time axis as
  ``k`` checkpointed segments, so only one segment's history is alive during
  the backward pass; the price is running each segment's forward pass a
  second time. It divides ``backward_storage_bytes`` by ``k`` and adds a
  ``auxiliary_state_bytes`` term for the states held at the segment
  boundaries, which is what ``estimate_memory(n_segments=k)`` and
  ``plan(..., checkpoint_segments=k)`` report. That term counts ``k`` states
  while a model gradient is being computed and **one** where none is (a
  forward-only segmented run threads a single state forward), and it sizes a
  state field by field rather than as whole padded grids, so it stays a floor
  like every other term. Unlike the levers above it
  costs no accuracy at all: the result matches an unsegmented run to
  summation order.
* **storage_compression.** Compresses the sampled wavefield history,
  regardless of ``storage_mode``. It shrinks the offloaded host buffer by
  about 3.8x in 2-D and 4.0x in 3-D for float32 (about 7.1x and 7.9x for
  float64) and, composed with ``storage_mode="cpu"``, runs roughly 2 to 3
  times faster than uncompressed offload, because it reduces the PCIe
  traffic that offload is bottlenecked on. **This is lossy and it changes
  the model gradient.** Measured relative L2 error against an uncompressed
  run: about **0.2-4% for 2-D problems and 0.2-8% for 3-D**. For a matched
  geometry 3-D is two to five times the 2-D error -- a 3-D block shares one
  scale across 512 cells against 64 in 2-D -- but the spread across
  geometries and loss functions is wide enough that the two ranges overlap.
  The error does *not* grow with the size of the problem: with the survey and
  the model anomaly held at fixed cell indices it is constant to five
  significant figures from 32x32 to 400x400 and from 32^3 to 120^3. It does
  depend on the survey geometry and on the loss function, so treat these as
  ranges rather than a number to budget against; a record too short for the
  wavefield to reach the receivers at all can give tens of percent, because
  there the gradient itself is nearly empty. The relative maximum error,
  defined as ``max|g_comp - g_exact| / max|g_exact|``, is comparable to the
  relative L2 error rather than a fixed multiple of it -- typically 0.3 to 3
  times it, and in 3-D more often below it than above. The returned
  wavefields, auxiliary state, receiver amplitudes and the gradients with
  respect to the source amplitudes and the initial wavefields are unaffected,
  bit for bit. **Do not use it for ``gradcheck``, adjoint or dot-product
  tests, or publication-quality gradients.** ``storage_compression`` is a
  permanent, non-deprecated alias of the same argument.

Measured accuracy of ``estimate_memory``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below compares ``estimate_memory``'s prediction against real
measured peak memory for the scalar propagator, which has a single model and
so stores a single wavefield history (see the multi-parameter propagators
below for the case where that is not true) (accuracy 4, float32,
forward + backward, one homogeneous model, a handful of receivers),
measured on RTX 4000 Ada 20 GB / RTX A4000 16 GB cards (the two
gave identical results to 4 significant figures at every point, as
expected for a pure allocation-size effect). "Measured" is
``torch.cuda.max_memory_allocated()`` for ``storage_mode="device"``; for
``storage_mode="cpu"`` it is that GPU figure plus the host RSS high-water
mark, since ``estimate_memory``'s ``total_bytes_lower_bound`` for
``"cpu"``/``"disk"``
modes is deliberately a *combined* host+device figure (see its docstring),
not a device-only one.

.. list-table::
   :header-rows: 1

   * - Grid
     - nt
     - Shots
     - storage_mode
     - compression
     - Measured
     - Estimated
     - Relative error
   * - 300x300 (2-D)
     - 400
     - 8
     - device
     - False
     - 1.43 GB
     - 1.37 GB
     - -4.3%
   * - 300x300 (2-D)
     - 400
     - 8
     - device
     - True
     - 0.45 GB
     - 0.39 GB
     - -13.6%
   * - 64\ :sup:`3`
     - 200
     - 2
     - device
     - False
     - 1.10 GB
     - 0.99 GB
     - -10.2%
   * - 64\ :sup:`3`
     - 200
     - 2
     - device
     - True
     - 0.40 GB
     - 0.29 GB
     - -28.0%
   * - 120\ :sup:`3`
     - 200
     - 2
     - device
     - False
     - 5.57 GB
     - 4.99 GB
     - -10.4%
   * - 120\ :sup:`3`
     - 200
     - 2
     - device
     - True
     - 2.04 GB
     - 1.47 GB
     - -28.2%
   * - 180\ :sup:`3`
     - 100
     - 1
     - device
     - False
     - 4.56 GB
     - 3.67 GB
     - -19.5%
   * - 180\ :sup:`3`
     - 100
     - 1
     - device
     - True
     - 2.09 GB
     - 1.20 GB
     - -42.7%
   * - 300x300 (2-D)
     - 400
     - 8
     - cpu
     - False
     - 2.19 GB
     - 1.37 GB
     - -37.2%
   * - 300x300 (2-D)
     - 400
     - 8
     - cpu
     - True
     - 0.62 GB
     - 0.39 GB
     - -37.1%
   * - 64\ :sup:`3`
     - 200
     - 2
     - cpu
     - False
     - 1.21 GB
     - 1.00 GB
     - -17.6%
   * - 64\ :sup:`3`
     - 200
     - 2
     - cpu
     - True
     - 0.43 GB
     - 0.29 GB
     - -32.0%
   * - 120\ :sup:`3`
     - 200
     - 2
     - cpu
     - False
     - 9.95 GB
     - 5.04 GB
     - -49.3%
   * - 120\ :sup:`3`
     - 200
     - 2
     - cpu
     - True
     - 2.94 GB
     - 1.47 GB
     - -50.0%
   * - 180\ :sup:`3`
     - 100
     - 1
     - cpu
     - False
     - 5.42 GB
     - 3.74 GB
     - -31.1%
   * - 180\ :sup:`3`
     - 100
     - 1
     - cpu
     - True
     - 2.29 GB
     - 1.20 GB
     - -47.4%

Reading this honestly: ``estimate_memory`` **under**-predicts at every
measured point here, never over-predicts, so leave headroom rather than
treating the number as a ceiling. For the default ``storage_mode="device"``,
uncompressed estimates were 4-20% below measured, growing with grid size;
``storage_compression=True`` estimates were characteristically worse
(14-43% below) at every grid size tested -- not because compression itself
is mismodelled, but because it correctly shrinks the sampled-history term
the formula gets right, which leaves a roughly constant, currently-unmodelled
per-call overhead (extra buffers beyond the documented forward-wavefield
count) making up a larger share of the true total. That same effect shows
up directly when reducing the sampled-history term another way: at a fixed
120^3/nt=200/2-shot/uncompressed/device configuration, the estimate was
-10.4% at ``model_gradient_sampling_interval=1``, widening to -18.1% at 2,
-29.1% at 4, -45.5% at 10, and -70.7% at 200, as the (correctly modelled)
stored-history term shrinks toward zero and that fixed overhead dominates.
``n_shots`` behaves better: at a fixed 64^3/nt=200/device/uncompressed
configuration the relative error stayed close to -10% across 1, 2, 4 and 8
shots (-10.9%, -10.2%, -9.9%, -9.7%), so more shots does not by itself
degrade the estimate. For ``storage_mode="cpu"``/``"disk"``, even the
combined host+device measured total exceeded the estimate by 17-50% in
every tested configuration. The key is named ``total_bytes_lower_bound``
for exactly this reason: it is a floor, not a ceiling, in every mode. A run
that exceeds it is normal -- every configuration measured did -- so size
with substantial headroom.

Multi-parameter propagators (``acoustic``, ``elastic``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table above is scalar-only, where one model means one stored wavefield
history. For ``acoustic`` and ``elastic`` the estimate is only accurate if
you tell it *which* models require gradients, with ``models_requiring_grad``
(the rows below are measured on an RTX 4000 Ada, float32, accuracy 4, 1 shot,
``storage_mode="device"``, uncompressed; the last column is the same call
given only ``n_models_requiring_grad``, whose worst-case slot count is an
upper bound rather than an estimate of that particular run):

.. list-table::
   :header-rows: 1

   * - Configuration
     - Histories stored
     - Measured
     - Estimated (named)
     - Estimated (count only)
   * - ``acoustic`` 96\ :sup:`3`, nt=150, ``v``
     - 1
     - 1.90 GiB
     - 1.64 GiB
     - 6.24 GiB
   * - ``acoustic`` 96\ :sup:`3`, nt=150, ``v``\ +\ ``rho``
     - 4
     - 6.49 GiB
     - 6.24 GiB
     - 6.24 GiB
   * - ``acoustic`` 300x1000, nt=1000, ``v``\ +\ ``rho``
     - 3
     - 4.05 GiB
     - 4.02 GiB
     - 4.02 GiB
   * - ``elastic`` 96\ :sup:`3`, nt=40, ``lamb``
     - 3
     - 2.19 GiB
     - 1.50 GiB
     - 2.73 GiB
   * - ``elastic`` 96\ :sup:`3`, nt=40, all three
     - 9
     - 4.88 GiB
     - 3.96 GiB
     - 3.96 GiB
   * - ``elastic`` 300x1000, nt=600, all three
     - 5
     - 4.09 GiB
     - 4.03 GiB
     - 4.03 GiB

Named, the stored-history term is now exact in every row, and the remaining
gap is the same fixed per-call overhead the scalar table already describes:
the backward pass's own padded wavefield set, the wavefields returned to the
caller (which cover the padded domain, PML included), the padded/staggered internal model copies, and one
gradient buffer per internal model. That overhead does not grow with ``nt``,
so it is negligible for the long 2-D runs (1.01x) and worst for a 3-D
elastic run with few time steps (1.23-1.46x at ``nt=40``, where it is about
0.9 GiB of a 4.9 GiB peak).

Progress feedback for long runs
---------------------------------

For a single long forward/backward pass, a ``forward_callback`` (see above)
can drive a `tqdm <https://tqdm.github.io/>`_ progress bar by advancing it
``callback_frequency`` steps each time it is invoked:

.. code-block:: python

    from tqdm import tqdm

    nt = source_amplitudes.shape[-1]
    callback_frequency = 10
    with tqdm(total=nt) as pbar:
        def forward_callback(state):
            pbar.update(callback_frequency)

        out = deepwave.scalar(
            v, grid_spacing, dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            forward_callback=forward_callback,
            callback_frequency=callback_frequency,
        )

As above, remember that the callback is a real Python re-entry point every
``callback_frequency`` steps, so pick a value that keeps overhead small
relative to the work done between callbacks (a few tens of steps is usually
enough for a smooth-looking bar without materially slowing propagation).

For an FWI epoch loop (see :doc:`example_fwi`), wrap the outer iteration
count instead, which has negligible overhead since it only updates once per
epoch, not once per internal time step:

.. code-block:: python

    for epoch in tqdm(range(n_epochs)):
        optimiser.zero_grad()
        out = deepwave.scalar(v, grid_spacing, dt, ...)
        loss = loss_fn(out.receiver_amplitudes, observed_data)
        loss.backward()
        optimiser.step()

Location errors ("... is out of bounds")
------------------------------------------

If a ``source_locations``/``receiver_locations`` (or, for elastic, a
``source_locations_z``/``_y``/``_x``/``_p`` or ``receiver_locations_*``)
Tensor contains a coordinate outside the model, Deepwave raises a
``RuntimeError`` naming the propagator, the argument, the shot and point
index, the dimension, the offending value, and the valid range, for
example::

    scalar: source_locations is out of bounds: at shot 0, point 0,
    dimension 1 (of 2), the coordinate is 15, but the model has only 10
    cells (valid indices 0 to 9). Locations are grid-cell indices, not physical
    distances -- if you have coordinates in metres (or another physical
    unit), divide by grid_spacing first. Negative indices are not
    supported, except the sentinel deepwave.IGNORE_LOCATION ...

Two common causes:

* **Coordinates in physical units, not grid cells.** Locations are always
  integer (or, for off-grid placement, floating-point) grid-cell indices, not
  metres/millimetres/etc; divide a physical coordinate by ``grid_spacing``
  first. See :doc:`units` (and :doc:`ultrasound` if you are working at
  ultrasound/NDT scales) for the full unit conventions.
* **An accidental negative index.** Negative indices are never valid *except*
  the sentinel :data:`deepwave.IGNORE_LOCATION`, which marks an unused
  source/receiver slot (e.g. to give shots in a batch differing numbers of
  active receivers) -- a stray ``-1`` from, for instance, subtracting one too
  many from a computed index, is almost always a bug rather than an intended
  sentinel.

The ``pml_freq`` / physical-units mismatch
--------------------------------------------

``pml_freq`` tunes both the PML's absorption and the
grid-cells-per-wavelength check, and it is in whatever frequency unit your
``dt`` implies. Deepwave 0.0.27 and earlier defaulted it to ``25.0`` -- a
*seismic* dominant frequency in Hz -- whenever it was omitted, which at a
different physical scale (most commonly ultrasound/NDT, where source
frequencies are more like ``1e5``-``1e7`` Hz, or single-digit MHz in a
millimetre--microsecond system) was wrong by orders of magnitude and typically
showed up as reflections off model edges that should have been absorbed.

Deepwave now infers ``pml_freq`` from the peak of your source amplitudes'
spectrum instead, so it comes out at the right scale on its own; if this is
the bug you are chasing on 0.0.27, upgrading fixes it. Two things to check
anyway:

* If your run emits the migration warning that says ``pml_freq`` was
  inferred and is not ``25.0``, that is Deepwave telling you the numbers now
  differ from what 0.0.27 produced for this call -- usually because 0.0.27 was
  wrong for it. ``pml_freq=25.0`` restores the old behaviour if you need to
  reproduce an old result. That warning is removed in 0.4.0.
* Pass ``pml_freq`` explicitly for anything you publish or invert, set to the
  dominant/peak frequency of your source in whatever unit system your
  ``dt``/``grid_spacing`` use: an inferred value ties the absorbing boundary
  to the wavelet, so changing the wavelet changes the boundary. See
  :ref:`pml-freq`, :doc:`units` for the general conventions, and
  :doc:`ultrasound` for a worked ultrasound-scale example.

Large gradients at the edge of the model
-------------------------------------------

If the gradient with respect to a velocity/model tensor is much larger at the
edge cells than elsewhere, this is very likely the (correct, but often
undesirable) contribution of the PML absorbing boundary rather than a bug --
see :doc:`example_large_edge_gradient` for why it happens and how to avoid it
(typically gradient clipping, or making the affected edge reflective instead
of absorbing; see :doc:`boundaries`).

Cycle-skipping in FWI
------------------------

If an FWI inversion converges to an implausible model (e.g. missing
low-wavenumber/background structure) despite a reasonable starting model and
low data misfit, this is a classic symptom of cycle-skipping. This is the
optimiser matching the wrong cycle of an oscillatory arrival. See
:doc:`example_fwi` for the frequency-continuation mitigation.
:doc:`example_gsot` describes an
alternative loss function, Graph Space Optimal Transport, that is more
robust to cycle-skipping.

Numerical dispersion
------------------------

If your wavefield or synthetic data looks "ringy" or shows spurious
high-frequency oscillations trailing the main wavefronts, check whether your
grid has enough points per wavelength for the chosen finite-difference
``accuracy`` order; see :doc:`accuracy` for the points-per-wavelength
guidance and the CFL condition.

Oscillations near a fluid-solid (water-layer) interface
--------------------------------------------------------

If an ``elastic`` simulation with a water layer (``vs = 0``) over a solid shows
oscillations or ringing in traces recorded near the fluid-solid interface (the
symptom reported in Deepwave issue #116), this is almost always a near-interface
numerical/physical effect -- Scholte interface waves and grid dispersion -- and
**not** a divergence or a bug in the fluid-solid handling. A water layer over a
solid (with a free surface on the top of the water) is long-run stable in
Deepwave; the fluid-solid physics (zero shear traction at the interface, correct
reflection/transmission and P-to-S mode conversion) is validated in the test
suite. To reduce the oscillations:

* increase the points per wavelength near the interface -- use a finer
  ``grid_spacing`` or a higher finite-difference ``accuracy`` -- since the
  interface sharpens the wavefield locally;
* do not place receivers exactly *on* the interface row; move them a cell or two
  into the medium you want to record;
* if you also have a free surface, keep the standard vacuum free-surface setup
  (see :doc:`boundaries`).

See :doc:`ultrasound` for the fluid-solid modelling recipe (transcranial
ultrasound and ocean-bottom seismic) and :doc:`boundaries` for the distinct,
genuinely-unstable thin-slab-between-two-free-surfaces case and its ``mpml_fraction``
mitigation.

"Illegal instruction (core dumped)"
---------------------------------------

If Deepwave reports an unsupported CPU instruction set on import, its wheel
contains instructions that your processor does not support. Older builds can
instead abort with ``Illegal instruction (core dumped)`` on the first
propagator call. This most often affects older x86 CPUs that lack AVX2 or
virtual machines that expose a restricted instruction set.

Current x86 Deepwave wheels require AVX2 for propagation performance. On an
older x86 CPU or a virtual machine that hides AVX2, build Deepwave from source
with its portable default CPU settings:

.. code-block:: bash

    python -m pip install --force-reinstall --no-deps --no-binary deepwave deepwave

You can inspect what your CPU supports and what the installed binary expects
with:

.. code-block:: python

    import deepwave
    print(deepwave.diagnostics.report())

If ``required_baseline`` is non-empty and the corresponding feature is
``False``, the installed binary needs an instruction set your CPU lacks;
the default source build above resolves it. Do not set
``DW_FIXED_AVX2_OPT_IN=ON`` for that build. Deepwave raises a
clear ``RuntimeError`` pointing here rather than letting the process crash
with the illegal-instruction message.

Will a Deepwave wheel keep working after I upgrade PyTorch (or Python)?
--------------------------------------------------------------------------

Yes. Deepwave's compiled backend (``libdeepwave_C``) is a single shared
library, built from C, a small C++ synchronisation helper, and CUDA, and
loaded from Python with ``ctypes``. It uses the platform C/C++ and OpenMP
runtimes but never links against ``libtorch`` or the CPython ABI, and never
uses ``pybind11``/``nanobind``/Cython. Within the supported platform and
Python range, a PyTorch or Python upgrade alone does not require rebuilding
the wheel. This is unlike
extensions built with ``torch.utils.cpp_extension``/``pybind11``, which are
tied to the libtorch ABI and CPython ABI they were compiled against.

"Compiled backend failed"
--------------------------

A ``RuntimeError`` reading ``Compiled backend failed: the '<propagator>'
propagator's compiled <forward/backward> pass returned non-zero status code
<code>`` means the C/CUDA backend itself reported failure, rather than
Deepwave's own (Python-level) input validation catching a problem before the
call. This is rare, since most invalid inputs are already rejected earlier
with a more specific error. Things to try:

* re-run the same call with ``backend="eager"``: if the eager
  backend also fails, the problem is in the model/inputs, not the compiled
  backend specifically, and the eager traceback usually pinpoints it more
  precisely;
* if ``storage_mode`` is ``"disk"``, check that ``storage_path`` is writable
  and that the disk has enough free space for the run's snapshots (see
  :func:`deepwave.estimate_memory`, above);
* check the process's memory usage; a backend that is killed or aborts
  under memory pressure can surface here rather than as a Python
  ``MemoryError``.

If none of these explain it, please open an issue with the full script,
the reported status code, and your platform/CUDA version.

Running inside ``torch.cuda.stream()``
------------------------------------------

The compiled CUDA backend (which ``backend="auto"``, the default, resolves
to on a CUDA propagator that has one; ``backend="compiled"`` asserts it
explicitly) configures its kernels through ``__constant__``
device-global memory, written by a synchronous ``cudaMemcpyToSymbol`` before
each launch. That mechanism is not stream-safe: if the call runs on a
non-default CUDA stream (for example inside a ``with
torch.cuda.stream(my_stream):`` block), a concurrent call on another stream
can overwrite the configuration before the kernel that needs it has launched,
silently producing wrong results. To prevent this, Deepwave raises immediately
with a ``RuntimeError`` when the compiled backend is invoked outside the
default stream, rather than letting it run and risk a wrong answer::

    RuntimeError: Deepwave's compiled CUDA backend requires the default CUDA
    stream (kernel configuration uses device globals that are not
    stream-safe). Run outside torch.cuda.stream(), or use
    backend="eager".

If you need to run Deepwave on a non-default stream, use
``backend="eager"`` or ``backend="torch_compile"`` instead of the compiled
backend -- both are ordinary PyTorch
operations with no shared device-global configuration state, so they are
stream-safe.

Wrapping a propagator in ``torch.compile`` or ``torch.export``
----------------------------------------------------------------

An **outer** ``torch.compile``, ``torch.export`` or CUDA-graph capture around
a propagator call is not supported: the compiled backend enters C/CUDA through
``ctypes``, which Dynamo cannot trace, so ``torch.compile`` graph-breaks at the
call (at best) and ``torch.export`` fails while tracing. The
``backend="torch_compile"`` *option* is a different thing and is supported --
see :doc:`capabilities`.

"None of the inputs have requires_grad=True" from a segmented run
--------------------------------------------------------------------

This is PyTorch's own warning, emitted once per segment by
``torch.utils.checkpoint``. Deepwave no longer triggers it: a segmented run
with nothing to differentiate skips the checkpoint wrapper and calls each
segment directly, which produces bitwise-identical results, so
``checkpoint_segments=`` and
:func:`deepwave.checkpointing.run_segments`'s ``checkpoint=True`` can be left
set on a forward-only call. If you still see the warning, the wrapper is
being applied by your own code rather than by Deepwave.

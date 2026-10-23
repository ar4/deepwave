3D FWI with storage options to reduce memory usage
==================================================

This example shows how to reduce the memory used by 3D FWI, useful whenever
a model is large enough that gradient calculation runs out of memory. 3D
wave propagation naturally requires significantly more memory than 2D
propagation. For large 3D models, or even large 2D models, the memory
required to store the intermediate wavefields needed for gradient
calculation (backpropagation) can exceed what is available on the compute
device (e.g. GPU).

To address this, Deepwave allows you to offload the storage of these intermediate wavefields to CPU memory or to disk, and optionally compress them. This can drastically reduce the peak memory usage on the compute device, enabling the processing of much larger models.

The storage policy has three parts. You can set them one at a time, as
individual keyword arguments to the propagator:

    - ``storage_mode``: Where to store the intermediate data. Options are ``"device"`` (default), ``"cpu"``, ``"disk"``, or ``"none"``.
    - ``storage_path``: The directory to use for disk storage. The default is
      the system temporary directory, chosen when you call rather than when
      Deepwave is imported, so setting ``TMPDIR`` at any point beforehand is
      honoured. A ``pathlib.Path`` is accepted. On many systems the temporary
      directory is a RAM-backed ``tmpfs`` much smaller than the disk, so using
      the default with ``storage_mode="disk"`` emits a one-time warning naming
      the directory it picked.
    - ``storage_compression``: Whether to apply a **lossy** compression to
      the stored wavefield history (default ``False``). It changes the model
      gradient, by about 0.2-4% in 2-D and 0.2-8% in 3-D in relative L2
      error; the keyword's name does not say so, which is why this does.

or as one :class:`deepwave.Storage` object through ``storage=``, which is
what this example does::

    policy = deepwave.Storage(mode="cpu", lossy_compression=True)

The two spellings are equivalent, and neither is going away. The object is
worth reaching for when the policy is a thing in its own right -- shared
between the several calls of an inversion, swept over with
``dataclasses.replace(policy, mode="disk")``, or read back off a finished run
as ``out.info.storage``, which reports the *resolved* policy including the
directory that ``"disk"`` actually chose. Passing both spellings in the same
call raises, because there is no honest way to merge them: a ``storage=``
object is the whole policy, so a stray ``storage_mode=`` alongside it could
only be silently ignored or silently override.

If you would rather see the bill than be refused, :func:`deepwave.plan`
answers the same question without running anything. It resolves the call
using the propagator's own front end and stops before propagation and
gradient-history allocation; source preprocessing can still scale with time::

    cost = deepwave.plan(
        scalar,
        v,
        [dz, dy, dx],
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=pml_width,
        storage=policy,
    )
    print(cost.summary())

which for this example's sizes, on a machine with a CUDA device, prints::

    extent 10 x 30 x 30 (origin (0, 0, 0)), padded to 34 x 74 x 74
    nt 150 at dt 0.004 s -> 300 internal steps at 0.002 s (step_ratio 2, courant 0.6)
    pml_width (0, 20, 20, 20, 20, 20) at 25 Hz, max_vel 1500.0
    backend 'compiled' (backward 'compiled') on cuda:0, torch.float32
    storage Storage(mode='cpu', path=None (resolved at call time), lossy_compression=True), lossy_compression_used True
    memory at least 146.4 MB = 23.8 MB wavefields + 119.6 MB history + 3.0 MB gradient scratch + 0 B auxiliary states
    no warnings

``cost.memory`` is the same breakdown as a dict, so
``cost.memory["backward_storage_bytes"]`` is what the history alone will
cost, and ``cost.info`` is the very :class:`deepwave.RunInfo` the run would
return. ``plan`` also takes ``checkpoint_segments=``, so you can compare
the estimates for segmented and unsegmented runs before executing either.

``"none"`` is a **forward-only** mode: it stores nothing, so the imaging
correlation between the forward and adjoint wavefields — which *is* the model
gradient — cannot be formed. Combining it with a model that requires a
gradient raises a ``RuntimeError`` rather than returning a wrong number.
Gradients with respect to the source amplitudes and the initial wavefields do
not need the stored forward history and remain available in this mode, as does
any forward-only run. Two propagators are stricter: ``acoustic_born`` and
``elastic_born`` reject **every** gradient under ``storage_mode="none"``,
source amplitudes and initial wavefields included, so for those two the mode is
forward-only without exception.

``storage_compression=True`` is lossy, and the loss shows up in the
model gradient. It is an 8-bit block max-abs quantisation of the stored
wavefield history: 8 cells per spatial dimension per block, one
full-precision scale per block (``BLOCK_SIZE=8`` in
``src/deepwave/simple_compress.c``). Measured relative L2 error against an
uncompressed run is about **0.2-4% for 2-D problems and 0.2-8% for 3-D**.
For a matched geometry 3-D is two to five times the 2-D error -- a 3-D block
shares one scale across 512 cells against 64 in 2-D -- but the spread across
geometries and loss functions is wide enough that the two ranges overlap. The
error stayed constant to five significant figures in the measured experiments
with the survey and model anomaly held at fixed cell indices from 32x32 to
400x400 and from 32\ :sup:`3` to 120\ :sup:`3`, and stayed constant in ``nt``
once the record was long enough for the wavefield to reach the receivers. This is a scoped observation, not a guarantee that
compression error is independent of problem size or record length. It depends
on the survey geometry and on the loss
function, so treat these as ranges rather than a number to budget against. A
record too short for the wavefield to reach the receivers at all is the
exception: it can give tens of percent, because there the gradient itself is
nearly empty. The relative maximum error, defined as
``max|g_comp - g_exact| / max|g_exact|``, is comparable to the relative L2
error rather than a fixed multiple of it -- typically 0.3 to 3 times it, and
in 3-D more often below it than above. The worst-affected values are the
model's outermost cells, whose gradient sums large, largely cancelling
contributions from the absorbing layer, so the quantisation error does not
cancel with them: measured on the gradient of ``scalar_born``'s ``dv``,
normalised by the gradient's own maximum, the largest edge-cell error is
0.03-0.07 in 1-D, about 1e-4 in 2-D and about 1e-5 in 3-D, against below
1e-3, 2e-5 and 1e-5 respectively in the interior. Without compression, the
stored history has no compression error; the model gradient can still differ
from the full finite-discrete derivative when its correlation is sampled at
an effective stride greater than one (see :doc:`what_deepwave_calculates`).

Compression can amplify second-derivative error. In the small scalar storage
regressions (1D/2D/3D, float32, one/two shots, effective history stride one),
model-gradient errors are 1.1--3.0%, while the second derivative of the squared
model-gradient norm differs by 0.5--9.7%. Device, CPU and disk storage give the
same compressed results. These are measurements for those calls, not error
bounds for other objectives. Keep compression off when validating Hessians.

**Unaffected in a first-order call, bit for bit:** the returned wavefields and auxiliary state, the
receiver amplitudes, and the gradients with respect to the source amplitudes
and the initial wavefields. Compression touches only the stored history that
forms the model gradient. That is normally fine for a preconditioner-quality
FWI gradient, but it is far above ``gradcheck`` tolerances: validate
gradients, and produce publication-quality ones, with compression off. In
exchange it shrinks the offloaded history several-fold -- about 3.8x in 2-D
and 4.0x in 3-D for float32, and about 7.1x and 7.9x for float64, a ratio
fixed by the dtype and the number of spatial dimensions rather than by the
size of the problem -- and, composed with ``storage_mode="cpu"``, runs
roughly two to three times faster than uncompressed offload, because it
reduces the PCIe traffic that offload is bottlenecked on.
:doc:`example_flagship_3d` measures the error on its own problem rather than
assuming it, which is the habit to copy.

Note that if none of the input property models (such as ``v``) require gradients (``requires_grad=True``), Deepwave will not store any intermediate data regardless of the ``storage_mode`` setting. Thus, for pure forward modelling with a fixed model, the storage settings will not affect memory usage.

In this example, we demonstrate 3D FWI on a small model. We use ``mode="cpu"`` and ``lossy_compression=True`` to minimise the memory footprint on the device. While this small example would fit in GPU memory without these options, the same approach applies to large-scale 3D problems. Combining these storage options with the batched shots approach described in :doc:`the previous example <example_rtm>` is a good idea, but it is not used in this example for simplicity.

We define a simple 3D velocity model with a gradient background and a block anomaly::

    # Define the 3D model
    nx, ny, nz = 30, 30, 10
    # ... (setup code) ...
    v_background = torch.full((nz, ny, nx), 1500.0, device=device)
    v_true = v_background.clone()
    # Add anomaly
    v_true[4:8, 13:18, 13:18] = 1600

The interface for 3D propagation is consistent with 2D, but input tensors and coordinates now have an additional dimension. `pml_width` can be specified as a list of 6 integers (start/end for each of the 3 dimensions)::

    # Inversion loop
    optimizer = torch.optim.SGD([v], lr=1e8)
    loss_fn = torch.nn.MSELoss()

    for _epoch in range(1):
        optimizer.zero_grad()

        out = scalar(
            v,
            [dz, dy, dx],
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=pml_width,
            storage=policy,
        )

        loss = loss_fn(out.receiver_amplitudes, observed_data)
        loss.backward()
        optimizer.step()

Using ``mode="disk"`` works similarly, but writes data to temporary files in the policy's ``path``. This is the most memory-efficient option but may be slower due to disk I/O. Leaving ``path`` unset means the system temporary directory, which on many machines is a RAM-backed ``tmpfs`` smaller than the disk -- hence the one-time warning naming the directory it chose. ``out.info.storage.path`` reports which directory was actually used.

The files themselves live in a subdirectory named
``deepwave_tmp_<pid>_<random hex>``, created inside ``path`` when the call
allocates its wavefield history -- after the inputs have been validated and
before time stepping starts -- and holding one file per shot (one file in
total when the run is on CUDA). Deepwave removes that whole subdirectory when
the object owning it is garbage-collected, which is when nothing refers to
the run's stored history any more: after ``backward()`` for a run whose
gradient you take, and when the output and its autograd graph go out of scope
for one you do not. Nothing is left behind between runs in the normal course
of events; a process killed outright (``SIGKILL``, a hard crash) is the
exception, since no destructor runs then, and its directory has to be removed
by hand. The name carries the process id to make that easy to identify.

Dividing the data into :doc:`batches containing several shots <example_rtm>`, combined with compression and possibly transferring to CPU or disk for storage, is usually a good approach.

`Full example code <https://github.com/ar4/deepwave/blob/master/docs/example_storage.py>`_

Flagship 3-D FWI at a size that does not fit on one GPU
=========================================================

The other memory-lever examples (:doc:`example_storage`,
:doc:`example_checkpointing`, :doc:`example_distributed`) each demonstrate
one lever in isolation. This example composes all four at once, for a
synthetic 3-D FWI problem sized so that it genuinely does not fit on either
of a two-GPU machine's cards unmodified, but does fit -- comfortably -- once
the levers are composed. See :doc:`scaling` for the measured numbers behind
that claim and a decision table for other sizes.

The velocity model is a synthetic layered, faulted 3-D block (not the flat
anomaly used by the smaller storage example): a depth-increasing background,
a fault that vertically offsets the layering for half of the model, and a
localised high-velocity lens, giving the inversion genuine 3-D structure to
recover::

    def make_true_model(nz, ny, nx, device):
        z = torch.arange(nz, device=device, dtype=torch.float64)
        depth_trend = 1500.0 + 1500.0 * z / (nz - 1)
        v = depth_trend.view(nz, 1, 1).expand(nz, ny, nx).clone()

        fault_x = nx // 2
        throw = max(1, nz // 12)
        v[:, :, fault_x:] = torch.roll(v[:, :, fault_x:], shifts=throw, dims=0)
        # ... plus a localised high-velocity lens; see the full script.
        return v.clamp(1450.0, 3600.0).float()

The four levers are composed in two places. ``DistributedDataParallel``
splits the shots between the two GPUs, exactly as in
:doc:`example_distributed`'s fuller examples; within each rank, shots are
further split into small batches and the gradient is accumulated over them
(as in :doc:`example_rtm`)::

    import contextlib

    n_shots_per_rank = observed.shape[0]
    n_batches = (n_shots_per_rank + shot_batch - 1) // shot_batch
    for _epoch in range(n_epochs):
        optimiser.zero_grad()
        for batch in range(n_batches):
            b0 = batch * shot_batch
            b1 = min(b0 + shot_batch, n_shots_per_rank)
            s = slice(b0, b1)
            sync = contextlib.nullcontext() if batch == n_batches - 1 else prop.no_sync()
            with sync:
                out = prop(source_amplitudes[s], source_locations[s], receiver_locations[s])
                loss = loss_fn(out, observed[s]) * world_size * (b1 - b0) / n_shots_total
                loss.backward()
        optimiser.step()

The final partial batch is included. Weighting by the shot count compensates
for DDP's rank averaging, so the accumulated gradient represents the
full-survey mean loss. Only the final batch synchronises gradients on each
rank; otherwise unequal numbers of local batches would issue different
numbers of collective operations. The script uses ``torch.tensor_split``
to distribute all shots, including when the shot count is not divisible by
the number of GPUs.

Each batch's call goes through a ``Prop`` module that adds the remaining two
levers to an ordinary propagator call: ``checkpoint_segments`` (bounding
backward-pass memory to one time segment instead of the whole trace) and
``storage_mode="cpu"``/``storage_compression=True`` (offloading and
compressing the sampled wavefield history that segment keeps)::

    def forward(self, source_amplitudes, source_locations, receiver_locations):
        v = self.model()
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=self.freq,
            max_vel=self.max_vel,
            checkpoint_segments=self.n_segments,
            storage_mode="cpu",
            storage_compression=True,
        ).receiver_amplitudes

Nothing about that call is special: it is the same ``scalar`` call any other
example makes, with three extra keyword arguments. ``checkpoint_segments``
builds the :func:`deepwave.checkpointing.run_segments` call described in
:doc:`example_checkpointing` and returns the same combined output.

At the size this example uses by default (a 100\ :sup:`3` grid, 600 time
steps, 8 shots split across two GPUs), none of these levers alone is enough
-- see :doc:`scaling` for the measured progression from "out of memory" down
to comfortably fitting once all four are composed.

``DEEPWAVE_SMOKE=1`` shrinks the grid, ``nt``, shot count, and epochs so the
script runs in seconds as an automated check, using the same code path,
including the comparison below.

What the compression lever costs
---------------------------------

``storage_compression=True`` is the one lever here that changes an
answer. It stores the sampled wavefield history as 8-bit block max-abs
quantisation, and the loss lands entirely in the model gradient: the receiver
amplitudes, the returned wavefields and the gradients with respect to the
source amplitudes are bit-identical to an uncompressed run. So rather than
assert that the cost is acceptable, the script measures it, on one shot batch
at the model the inversion reached, by running the same propagation twice::

    gradients = []
    receivers = []
    for lossy in (False, True):
        v = v_value.detach().clone().requires_grad_()
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            checkpoint_segments=n_segments,
            storage_mode="cpu",
            storage_compression=lossy,
        )
        loss_fn(out.receiver_amplitudes, observed).backward()
        gradients.append(v.grad.detach())
        receivers.append(out.receiver_amplitudes.detach())

and printing the relative L2 and maximum differences between the two
gradients. On the reference two-GPU machine, at the default 100\ :sup:`3`
size, that comparison reports a **3.6% relative L2 and 4.4% relative maximum**
difference, with the two runs' receiver amplitudes bit-identical. The relative
maximum here is ``max|g_comp - g_exact| / max|g_exact|`` (see
``compression_cost`` in the script), i.e. 1.2 times the relative L2 on this
problem -- comparable to it rather than a fixed multiple of it.

Under ``DEEPWAVE_SMOKE=1`` the same comparison reports 26.0% and 58.6%. Read
that as a property of the smoke size rather than as a trend with problem
size: a 12\ :sup:`3` grid propagated for 40 time steps is far below the range
over which the error has been measured to be flat (32\ :sup:`3` upwards), and
it exists so that continuous integration exercises this code path at all, not
to characterise the lever.

What the error is mostly set by is the number of spatial dimensions -- a 3-D
block shares one quantisation scale across 512 cells against 64 in 2-D --
rather than the size of the grid. It also depends on the survey geometry and
on the loss function: on one fixed 3-D model and survey, changing nothing but
the objective, the measured relative L2 moved from 1.9% for a data residual to
3.4% for a plain sum of squared receiver amplitudes.
:func:`deepwave.scalar`'s ``storage_compression`` entry gives the ranges
measured across problems (about 0.2-4% in 2-D and 0.2-8% in 3-D). This
example's 3.6% sits inside the 3-D range; the smoke size's 26.0% sits outside
it, because a 12\ :sup:`3` grid at ``nt=40`` is too short a record for the
wavefield to reach the receivers at all, so the gradient it compresses is
nearly empty. Treat them as ranges, and measure your own case the way this
script does.

The practical rule: this lever is for making a problem run at all, not for
producing a number you will publish. Use it while you are sizing, debugging
or running a preconditioner-quality gradient; switch it off -- accepting the
larger host buffer and the slower offload -- for a final inversion, and never
use it for ``gradcheck``, adjoint or dot-product tests. In exchange it shrinks
the offloaded history several-fold and runs faster than uncompressed offload,
because the offload is bottlenecked on PCIe traffic:
:doc:`scaling`'s table measures this example's own configuration at 17.86 GB
of host memory uncompressed against 5.57 GB compressed, which is the
difference between fitting two ranks on a 31 GB machine and not.

`Full example code <https://github.com/ar4/deepwave/blob/master/docs/example_flagship_3d.py>`_

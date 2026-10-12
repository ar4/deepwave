Distributed (multi-GPU) execution
=================================

PyTorch can spread work over the multiple GPUs of a single machine, with
some flexibility in how this is done to suit different situations. Because
Deepwave is a PyTorch module like any other, it shares these abilities: in
practice, Deepwave is used shot-parallel across the GPUs of one node, not
across multiple nodes (see :doc:`comparison` for how this compares to
distributed-memory codes such as Devito, which do scale across nodes). The
most obvious way to use multiple GPUs with Deepwave is to divide the shots
between them: with two GPUs and one hundred shots to propagate, fifty could
run on each. PyTorch takes care of combining gradients correctly during
backpropagation, so both GPUs end up with the same model parameters for the
optimiser's next step.

The easiest way to divide shots over multiple GPUs is with `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`_, which you simply have to apply to the model you wish to use::

    class ReceiverModel(torch.nn.Module):
        def __init__(self, v_init, dx, dt):
            super().__init__()
            self.propagator = deepwave.Scalar(v_init, dx, requires_grad=True)
            self.dt = dt

        def forward(self, amplitudes, sources, receivers):
            return self.propagator(
                self.dt, source_amplitudes=amplitudes,
                source_locations=sources, receiver_locations=receivers,
            ).receiver_amplitudes

    propagator = torch.nn.DataParallel(ReceiverModel(v_init, dx, dt).cuda())
    receiver_amplitudes = propagator(
        source_amplitudes, source_locations, receiver_locations,
    )

Set ``requires_grad=True`` on the Deepwave module when optimising its model.
The module defaults to a fixed model, even if ``v_init`` requires a gradient;
that default avoids retaining model-gradient history during forward modelling.

No change is needed to the input, and the shot (batch) dimension is split
over the available GPUs to do the computation, but there is one wrinkle in
the output: with more than one GPU, ``DataParallel`` combines the per-GPU
results itself (``DataParallel.gather``) rather than letting Deepwave do it,
and it only knows how to do this for a plain ``torch.Tensor`` or a real
`namedtuple <https://docs.python.org/3/library/collections.html#collections.namedtuple>`_
whose fields are fixed by its class. Deepwave's return value,
:class:`deepwave.PropagatorOutput`, looks like a namedtuple (it supports
``out.receiver_amplitudes``, etc.) but is not one internally -- its field
names vary per call (they depend on the number of dimensions and, for
viscoacoustic, the number of relaxation mechanisms) rather than
being fixed by the class, which ``DataParallel``'s generic reconstruction
cannot handle. If your ``forward()`` returns the full ``PropagatorOutput``
directly (instead of pulling out just the tensor you need, as above) on more
than one GPU, ``DataParallel`` raises::

    deepwave.DeepwaveTypeError: PropagatorOutput cannot be reconstructed from
    a single iterable of values, which is what torch.nn.DataParallel.gather
    does when a module's forward() returns one. PropagatorOutput's field
    names vary per instance (they depend on ndim and, for
    viscoacoustic, on q_n_mechanisms), so they cannot be
    recovered from the class alone. Return plain tensors from forward()
    instead, e.g. `return out.receiver_amplitudes`. See
    docs/example_distributed.rst.

The fix is always the same: have ``forward()`` return the
individual tensor(s) you need (``out.receiver_amplitudes``, or a small tuple
of plain tensors if you need more than one field for e.g. a custom loss) --
see the fuller DataParallel example linked below.
(``DistributedDataParallel``, covered next, does not gather this way and has
no such restriction.)

This is convenient, but the PyTorch documentation instead recommends the
more complicated `DistributedDataParallel
<https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_.
It requires larger code changes, since more of the setup must be done
manually, but it can give better performance::

    import os

    def run_rank(rank, world_size, v_init, dx, dt,
                 source_amplitudes, source_locations, receiver_locations):
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        source_amplitudes = \
            torch.tensor_split(source_amplitudes, world_size)[rank].to(rank)
        source_locations = \
            torch.tensor_split(source_locations, world_size)[rank].to(rank)
        receiver_locations = \
            torch.tensor_split(receiver_locations, world_size)[rank].to(rank)
        propagator = torch.nn.parallel.DistributedDataParallel(
            deepwave.Scalar(v_init, dx, requires_grad=True).to(rank),
            device_ids=[rank]
        )
        out = propagator(dt, source_amplitudes=source_amplitudes,
                         source_locations=source_locations,
                         receiver_locations=receiver_locations)
        torch.distributed.destroy_process_group()


    if __name__ == "__main__":
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        n_gpus = min(torch.cuda.device_count(), source_amplitudes.shape[0])
        if n_gpus == 0:
            raise RuntimeError("This example requires CUDA and at least one shot.")
        torch.multiprocessing.spawn(run_rank,
                                    args=(n_gpus, v_init, dx, dt,
                                          source_amplitudes, source_locations,
                                          receiver_locations),
                                    nprocs=n_gpus)

Fuller examples of both of these approaches (with thanks to Vladimir Kazei for testing them):

- `DataParallel example <https://github.com/ar4/deepwave/blob/master/docs/example_distributed_dp.py>`_
- `DistributedDataParallel example <https://github.com/ar4/deepwave/blob/master/docs/example_distributed_ddp.py>`_

DDP averages gradients across ranks; it does not combine the loss values
returned by an optimiser closure. The full DDP example weights each rank's
mean loss by ``world_size * local_shots / total_shots`` before backpropagation
and averages those weighted losses across ranks before returning from the
LBFGS closure. This makes both the gradient and the returned loss correspond
to the full-survey mean, even for unequal shot partitions. A shared loss is
essential for LBFGS: rank-local stopping decisions can lead to different
numbers of closure calls and hang the next collective operation.

Both of the fuller examples above load the Marmousi model, like the
examples under :doc:`examples`'s "Using the Marmousi model" section --
run ``python docs/get_marmousi.py`` first; see :doc:`example_forward_model`
for what that does and why.

Shot-splitting is also the first of four memory levers (alongside shot
batching, ``storage_mode``/``storage_compression``, and
``checkpoint_segments``, the argument every propagator takes to run its time
axis as that many checkpointed segments) that compose to fit a 3-D problem
too large for one GPU across several -- see :doc:`example_flagship_3d` for a
complete, working composition and :doc:`scaling` for the measured numbers
behind it.

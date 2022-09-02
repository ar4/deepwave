Distributed (multi-GPU) execution
=================================

PyTorch provides the ability to spread work over multiple GPUs, both on the same node and across nodes. There is quite a bit of flexibility on how to do this, to suit different situations. As it is a PyTorch module like any other, Deepwave shares these abilities. The most obvious way to use multiple GPUs with Deepwave is to divide the shots between the GPUs. If you have two GPUs and want to propagate one hundred shots, you could run fifty of them on each GPU. PyTorch takes care of ensuring that the gradients are handled correctly, so if you perform backpropagation on the shots propagated by these two GPUs, the gradient contributions from each will be combined so that both GPUs will have the same model parameters for the next step of the optimiser.

The easiest way to divide shots over multiple GPUs is with `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`_, which you simply have to call on the model you wish to apply it to::

    propagator = torch.nn.DataParallel(deepwave.Scalar(v_init, dx))
    out = propagator(dt, source_amplitudes=source_amplitudes,
                     source_locations=source_locations)

You don't need to make any changes to your input, and the output should be the same as in the single GPU case, but the shot (batch) dimension will have been split over available GPUs to do the computation.

This is nice, but the PyTorch documentation instead recommends using the more complicated `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_. Larger code changes are required in this case as you need to do more of the setup manually, but it can apparently provide better performance::

    def run_rank(rank, world_size):
        torch.distributed.init_process_group("nccl")
        source_amplitudes = \
            torch.chunk(source_amplitudes, world_size)[rank].to(rank)
        source_locations = \
            torch.chunk(source_locations, world_size)[rank].to(rank)
        propagator = torch.nn.parallel.DistributedDataParallel(
            deepwave.Scalar(v_init, dx).to(rank),
            device_idxs=[rank]
        )
        out = propagator(dt, source_amplitudes=source_amplitudes,
                         source_locations=source_locations)
        torch.distributed.destroy_process_group()


    if __name__ == "__main__":
        n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(run_rank,
                                    args=(n_gpus,),
                                    nprocs=n_gpus)

Fuller examples of both of these approaches (with thanks to Vladimir Kazei for testing them):

- `DataParallel example <https://github.com/ar4/deepwave/blob/master/docs/example_distributed_dp.py>`_
- `DistributedDataParallel example <https://github.com/ar4/deepwave/blob/master/docs/example_distributed_ddp.py>`_


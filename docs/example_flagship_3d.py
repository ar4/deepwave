"""Flagship 3-D FWI at a size that does not fit on one GPU unmodified.

Composes four independent memory levers so a 3-D FWI problem too big for
either of the two GPUs individually fits comfortably on both at once:

- DDP shot-splitting (:class:`torch.nn.parallel.DistributedDataParallel`)
  divides the shots between the two GPUs.
- Shot batching further divides each GPU's shots into smaller groups,
  accumulating the gradient over them (as in :doc:`example_rtm`).
- ``storage_mode="cpu"`` and ``storage_compression=True`` offload and
  compress the sampled wavefield history kept for the model gradient.
  Compression is lossy in the model gradient, so the script also measures
  what it costs, against an otherwise identical uncompressed run.
- ``checkpoint_segments`` bounds the backward pass to one time segment's
  history at a time instead of the whole trace's.

See :doc:`/scaling` for the measured effect of each of these levers, on
their own and composed, at the sizes used here.
"""

import contextlib
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.ndimage import gaussian_filter

import deepwave
from deepwave import scalar


def setup(rank, world_size):
    """Sets up the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()


def make_true_model(nz, ny, nx, device):
    """Builds a synthetic layered, faulted 3-D velocity model.

    A depth-increasing background (1500 -> 3000 m/s), a fault that
    vertically offsets the layering for half of the x-extent, and a
    localised high-velocity lens (e.g. a carbonate build-up), so the
    inversion target has real 3-D structure rather than a flat block.
    """
    z = torch.arange(nz, device=device, dtype=torch.float64)
    depth_trend = 1500.0 + 1500.0 * z / (nz - 1)
    v = depth_trend.view(nz, 1, 1).expand(nz, ny, nx).clone()

    fault_x = nx // 2
    throw = max(1, nz // 12)
    v[:, :, fault_x:] = torch.roll(v[:, :, fault_x:], shifts=throw, dims=0)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    lens_r2 = (
        ((zz - 0.6 * nz) / (0.12 * nz + 1)) ** 2
        + ((yy - 0.5 * ny) / (0.18 * ny + 1)) ** 2
        + ((xx - 0.3 * nx) / (0.18 * nx + 1)) ** 2
    )
    v = v + 700.0 * torch.exp(-lens_r2)
    return v.clamp(1450.0, 3600.0).float()


class Model(torch.nn.Module):
    """A PyTorch module that represents a bounded velocity model."""

    def __init__(self, initial, min_vel, max_vel):
        """Initialises the Model.

        Args:
            initial (torch.Tensor): The initial velocity model.
            min_vel (float): The minimum allowed velocity.
            max_vel (float): The maximum allowed velocity.
        """
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel)),
        )

    def forward(self):
        """Maps the unconstrained parameter back to a velocity in range."""
        return (
            torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
            + self.min_vel
        )


class Prop(torch.nn.Module):
    """Wraps a checkpointed, storage-offloaded propagation for DDP."""

    def __init__(self, model, dx, dt, freq, max_vel, n_segments):
        """Initialises the Prop module.

        Args:
            model (torch.nn.Module): The velocity model.
            dx (float): The grid spacing.
            dt (float): The time step.
            freq (float): The dominant frequency of the source wavelet.
            max_vel (float): An upper bound on the wave speed.
            n_segments (int): The number of checkpointed time segments.
        """
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq
        self.max_vel = max_vel
        self.n_segments = n_segments

    def forward(
        self, source_amplitudes, source_locations, receiver_locations
    ):
        """Runs one checkpointed, storage-offloaded propagation."""
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


def compression_cost(
    v_value,
    dx,
    dt,
    freq,
    max_vel,
    n_segments,
    source_amplitudes,
    source_locations,
    receiver_locations,
    observed,
    loss_fn,
):
    """Relative error that lossy compression puts into the model gradient.

    Runs the same checkpointed, host-offloaded propagation twice -- once
    with ``storage_compression=True`` and once without -- and compares
    the two model gradients. Everything else about the two calls, including
    the receiver amplitudes they return, is identical: the compression
    touches only the stored wavefield history from which the model gradient
    is formed.

    Returns:
        tuple: the relative L2 error, the relative maximum error, and
        whether the two runs' receiver amplitudes are bit-identical.
    """
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
    exact, compressed = gradients
    difference = compressed - exact
    return (
        float(difference.norm() / exact.norm()),
        float(difference.abs().max() / exact.abs().max()),
        bool(torch.equal(receivers[0], receivers[1])),
    )


def run_rank(rank, world_size):
    """Runs the flagship 3-D FWI for one DDP rank."""
    setup(rank, world_size)

    # ``DEEPWAVE_SMOKE=1`` shrinks every size below so the script runs in
    # well under a minute as an automated check; the "else" branch is the
    # size that does not fit on either GPU without the four composed levers
    # (see docs/scaling.rst for the measurements backing that claim).
    smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
    grid = 12 if smoke else 100
    nz = ny = nx = grid
    dx = 10.0
    nt = 40 if smoke else 600
    # A larger dt (looser than the CFL condition, so step_ratio > 1) keeps
    # the smoke run's total record length well past the source wavelet's
    # onset without needing many more time steps; a segmented run resamples
    # this automatically (see deepwave.checkpointing.run_segments).
    dt = 0.004 if smoke else 0.0008
    freq = 15.0
    max_vel = 3700.0
    n_segments = 2 if smoke else 5
    n_shots_total = world_size if smoke else 8
    if world_size > n_shots_total:
        raise ValueError("Use at most one GPU per shot.")
    shot_batch = 1 if smoke else 2
    # At least 2 epochs even in smoke mode, since the convergence check
    # below needs more than one loss value to compare.
    n_epochs = 2
    n_rec_side = 2 if smoke else 8

    v_true = make_true_model(nz, ny, nx, device=rank)
    v_init = torch.tensor(
        gaussian_filter(
            v_true.cpu().numpy(), sigma=6 if not smoke else 1.5
        ),
        device=rank,
    )

    source_x = (
        torch.linspace(nx * 0.1, nx * 0.9, n_shots_total).round().long()
    )
    source_locations = torch.zeros(
        n_shots_total, 1, 3, dtype=torch.long, device=rank
    )
    source_locations[:, 0, 0] = max(1, nz // 20)
    source_locations[:, 0, 1] = ny // 2
    source_locations[:, 0, 2] = source_x

    rec_pos = (
        torch.linspace(nx * 0.05, nx * 0.95, n_rec_side).round().long()
    )
    rec_zz, rec_xx = torch.meshgrid(rec_pos, rec_pos, indexing="ij")
    n_receivers = n_rec_side * n_rec_side
    receiver_locations = torch.zeros(
        n_shots_total, n_receivers, 3, dtype=torch.long, device=rank
    )
    receiver_locations[:, :, 0] = max(1, nz // 30)
    receiver_locations[:, :, 1] = rec_zz.reshape(-1)
    receiver_locations[:, :, 2] = rec_xx.reshape(-1)

    peak_time = 1.5 / freq
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .reshape(1, 1, -1)
        .repeat(n_shots_total, 1, 1)
        .to(rank)
    )

    # Generate "observed" data from the true model directly (no segmentation
    # needed: v_true never requires grad, so Deepwave stores no wavefield
    # history regardless of storage_mode).
    with torch.no_grad():
        observed_full = scalar(
            v_true,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
        ).receiver_amplitudes

    # DDP shot-splitting: each rank gets its own contiguous slice of shots.
    observed = torch.tensor_split(observed_full, world_size)[rank]
    source_amplitudes = torch.tensor_split(source_amplitudes, world_size)[
        rank
    ]
    source_locations = torch.tensor_split(source_locations, world_size)[
        rank
    ]
    receiver_locations = torch.tensor_split(receiver_locations, world_size)[
        rank
    ]
    n_shots_per_rank = observed.shape[0]

    model = Model(v_init, 1400.0, 3650.0).to(rank)
    prop = Prop(model, dx, dt, freq, max_vel, n_segments).to(rank)
    prop = torch.nn.parallel.DistributedDataParallel(
        prop, device_ids=[rank]
    )

    optimiser = torch.optim.Adam(prop.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    n_batches = (n_shots_per_rank + shot_batch - 1) // shot_batch
    loss_history: list[float] = []
    for _epoch in range(n_epochs):
        optimiser.zero_grad()
        epoch_loss = 0.0
        for batch in range(n_batches):
            b0 = batch * shot_batch
            b1 = min(b0 + shot_batch, n_shots_per_rank)
            if b1 <= b0:
                continue
            s = slice(b0, b1)
            # Synchronise once per epoch, even when ranks have different
            # numbers of batches. no_sync must cover forward and backward.
            sync = (
                contextlib.nullcontext()
                if batch == n_batches - 1
                else prop.no_sync()
            )
            with sync:
                out = prop(
                    source_amplitudes[s],
                    source_locations[s],
                    receiver_locations[s],
                )
                loss = (
                    loss_fn(out, observed[s])
                    * world_size
                    * (b1 - b0)
                    / n_shots_total
                )
                loss.backward()
            epoch_loss += float(loss.item())
        optimiser.step()
        global_loss = torch.tensor(
            epoch_loss, dtype=torch.float64, device=rank
        )
        dist.all_reduce(global_loss)
        loss_history.append(float(global_loss.item()) / world_size)

    # What the lossy compression lever actually costs, measured rather than
    # asserted: the same gradient with and without it, on one shot batch.
    # A published inversion should be run with it switched off; it is here
    # because this problem does not otherwise fit.
    if rank == 0:
        rel_l2, rel_max, receivers_identical = compression_cost(
            model(),
            dx,
            dt,
            freq,
            max_vel,
            n_segments,
            source_amplitudes[:shot_batch],
            source_locations[:shot_batch],
            receiver_locations[:shot_batch],
            observed[:shot_batch],
            loss_fn,
        )
        print(
            "lossy compression cost: "
            f"relative L2 {rel_l2:.1%}, relative max {rel_max:.1%}, "
            f"receiver amplitudes identical: {receivers_identical}"
        )

    if rank == 0:
        v_out = model().detach()
        y_mid = ny // 2
        vmin = float(v_true.min())
        vmax = float(v_true.max())
        _, ax = plt.subplots(3, figsize=(8, 9), sharex=True, sharey=True)
        ax[0].imshow(
            v_init[:, y_mid, :].cpu(),
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax[0].set_title("Initial")
        ax[1].imshow(
            v_out[:, y_mid, :].cpu(),
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax[1].set_title("Out")
        ax[2].imshow(
            v_true[:, y_mid, :].cpu(),
            aspect="auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax[2].set_title("True")
        plt.tight_layout()
        plt.savefig("example_flagship_3d.jpg")

        v_out.cpu().numpy().tofile("flagship_3d_v_inv.bin")

        assert all(math.isfinite(x) for x in loss_history), (
            "loss history contains non-finite values"
        )
        assert loss_history[-1] < loss_history[0], "loss did not decrease"
        assert receivers_identical, (
            "compression must not change the receiver amplitudes"
        )
        assert 0.0 < rel_l2 < 0.5, (
            f"compressed gradient error {rel_l2} is outside the range this "
            "example expects"
        )
    cleanup()


def run(world_size):
    """Spawns processes for distributed training."""
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run(n_gpus)

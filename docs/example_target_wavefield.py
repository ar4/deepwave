"""Demonstrates optimizing source amplitudes to match a target wavefield.

This script optimises source amplitudes to minimise the difference between
the final wavefield and a target wavefield. This shows that optimization
can involve any of the inputs and outputs, not just the usual task of
adjusting the velocity model to match the receiver data.
"""

import torch
import torchvision

import deepwave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny = 200
nx = 200
dx = 5.0
dt = 0.004
nt = 100
n_shots = 1
n_sources_per_shot = 400
freq = 25
peak_time = 1.5 / freq
target = (
    torchvision.io.read_image(
        "target.jpg", torchvision.io.ImageReadMode.GRAY
    ).float()
    / 255
)
target = (target[0] - target.mean()).to(device)

v = 2000 * torch.ones(ny, nx, device=device)

# source_locations
source_locations = torch.zeros(
    n_shots,
    n_sources_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
torch.manual_seed(1)
grid_cells = torch.cartesian_prod(torch.arange(ny), torch.arange(nx))
source_cell_idxs = torch.randperm(len(grid_cells))[:n_sources_per_shot]
source_locations = (
    grid_cells[source_cell_idxs]
    .reshape(1, n_sources_per_shot, 2)
    .long()
    .to(device)
)

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)
source_amplitudes.requires_grad_()

optimiser = torch.optim.LBFGS([source_amplitudes])
loss_fn = torch.nn.MSELoss()


def closure():
    """Closure function for the LBFGS optimiser."""
    optimiser.zero_grad()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        pml_width=0,
    )
    y = out[0][0]
    loss = loss_fn(y, target) + 1e-3 * source_amplitudes.norm()
    loss.backward()
    return loss.item()


for _i in range(50):
    optimiser.step(closure)

(source_amplitudes.detach().cpu().numpy().tofile("source_amplitudes.bin"))

target_abs_max = target.abs().max().item()


def forward_callback(state):
    """A function called during the forward pass."""
    # Scale the wavefield to be between 0 and 1 and save as an image
    val = (
        state.get_wavefield("wavefield_0")[0].cpu() / target_abs_max / 2
        + 0.5
    )
    torchvision.utils.save_image(val, f"wavefield_{state.step:03d}.jpg")


deepwave.scalar(
    v,
    dx,
    dt,
    source_amplitudes=source_amplitudes.detach(),
    source_locations=source_locations,
    pml_width=0,
    forward_callback=forward_callback,
    callback_frequency=1,
)

# Alternative method of saving snapshots
# We want to save every time step, so we will create a loop over time steps.
# When we call the wave propagator, we only want it to advance by
# one time step.
# We can achieve this by calling the propagator with each time sample of the
# source amplitudes. As we discussed in the checkpointing example, however,
# that might not give us exactly the result that we want due to
# upscaling within Deepwave to obey the CFL condition. We therefore perform
# the upscaling ourselves and then call the propagator with chunks of the
# upscaled source amplitudes that correspond to one pre-upsampling time
# step.
#
# dt, step_ratio = deepwave.common.cfl_condition(dx, dx, dt, 2000)
# source_amplitudes = deepwave.common.upsample(source_amplitudes.detach(),
# step_ratio)
#
# for i in range(nt):
#    chunk = source_amplitudes[..., i * step_ratio : (i + 1) * step_ratio]
#    if i == 0:
#        out = deepwave.scalar(
#            v,
#            dx,
#            dt,
#            source_amplitudes=chunk,
#            source_locations=source_locations,
#            pml_width=0,
#        )
#    else:
#        out = deepwave.scalar(
#            v,
#            dx,
#            dt,
#            source_amplitudes=chunk,
#            source_locations=source_locations,
#            pml_width=0,
#            wavefield_0=out[0],
#            wavefield_m1=out[1],
#            psiy_m1=out[2],
#            psix_m1=out[3],
#            zetay_m1=out[4],
#            zetax_m1=out[5],
#        )
#    val = out[0][0] / target_abs_max / 2 + 0.5
#    torchvision.utils.save_image(val, f"wavefield_{i:03d}.jpg")

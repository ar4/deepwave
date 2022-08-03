import torch
import torchvision
import deepwave

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ny = 200
nx = 200
dx = 5
dt = 0.004
nt = 100
n_shots = 1
n_sources_per_shot = 400
freq = 25
peak_time = 1.5 / freq
target = torchvision.io.read_image(
    'target.jpg',
    torchvision.io.ImageReadMode.GRAY
).float()/255
target = (target[0] - target.mean()).to(device)

v = 2000*torch.ones(ny, nx, device=device)

# source_locations
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)
torch.manual_seed(1)
grid_cells = torch.cartesian_prod(torch.arange(ny), torch.arange(nx))
source_cell_idxs = torch.randperm(len(grid_cells))[:n_sources_per_shot]
source_locations = (grid_cells[source_cell_idxs]
                    .reshape(1, n_sources_per_shot, 2)).long().to(device)

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)
source_amplitudes.requires_grad_()

optimiser = torch.optim.LBFGS([source_amplitudes])
loss_fn = torch.nn.MSELoss()


def closure():
    optimiser.zero_grad()
    out = deepwave.scalar(v, dx, dt,
                          source_amplitudes=source_amplitudes,
                          source_locations=source_locations,
                          pml_width=0)
    y = out[0][0]
    loss = loss_fn(y, target) + 1e-4*source_amplitudes.norm()
    loss.backward()
    return loss


for i in range(50):
    optimiser.step(closure)

source_amplitudes.detach().cpu().numpy().tofile('source_amplitudes.bin')

dt, step_ratio = deepwave.common.cfl_condition(dx, dx, dt, 2000)
source_amplitudes = deepwave.common.upsample(source_amplitudes, step_ratio)

target_abs_max = target.abs().max()
for i in range(nt):
    chunk = source_amplitudes[..., i*step_ratio:(i+1)*step_ratio]
    if i == 0:
        out = deepwave.scalar(v, dx, dt,
                              source_amplitudes=chunk,
                              source_locations=source_locations,
                              pml_width=0)
    else:
        out = deepwave.scalar(v, dx, dt,
                              source_amplitudes=chunk,
                              source_locations=source_locations,
                              pml_width=0,
                              wavefield_0=out[0],
                              wavefield_m1=out[1],
                              psiy_m1=out[2],
                              psix_m1=out[3],
                              zetay_m1=out[4],
                              zetax_m1=out[5])
    val = out[0][0] / target_abs_max / 2 + 0.5
    torchvision.utils.save_image(val, f'wavefield_{i:03d}.jpg')

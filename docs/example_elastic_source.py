import matplotlib.pyplot as plt
import torch

import deepwave
from deepwave import elastic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny = 71
nx = 71
dx = 4.0

vp = torch.ones(ny, nx, device=device) * 1500
vs = torch.ones(ny, nx, device=device) * 1000
rho = torch.ones(ny, nx, device=device) * 2200
lamb, mu, buoyancy = deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

freq = 15
nt = 50
dt = 0.004
peak_time = 1.5 / freq

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time)).reshape(1, 1, -1).to(device)
)

# single body forces in y and x dimensions
source_locations = torch.tensor([[[35, 35]]]).to(device)
# body force in y located at (35.5, 35.5)
out = elastic(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    source_amplitudes_y=source_amplitudes,
    source_locations_y=source_locations,
    pml_freq=freq,
)
sy_y = out[0][0]
sy_x = out[1][0]
sy_p = -(out[2][0] + out[4][0])
# body force in x located at (35, 35)
out = elastic(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    source_amplitudes_x=source_amplitudes,
    source_locations_x=source_locations,
    pml_freq=freq,
)
sx_y = out[0][0]
sx_x = out[1][0]
sx_p = -(out[2][0] + out[4][0])

# explosive source located at (35, 35.5)
source_locations_y = torch.tensor([[[34, 35], [35, 35]]]).to(device)
source_locations_x = torch.tensor([[[35, 35], [35, 36]]]).to(device)
source_amplitudes_y = source_amplitudes.repeat(1, 2, 1)
source_amplitudes_y[:, 0] *= -1
source_amplitudes_x = source_amplitudes.repeat(1, 2, 1)
source_amplitudes_x[:, 0] *= -1
out = elastic(
    lamb,
    mu,
    buoyancy,
    dx,
    dt,
    source_amplitudes_y=source_amplitudes_y,
    source_locations_y=source_locations_y,
    source_amplitudes_x=source_amplitudes_x,
    source_locations_x=source_locations_x,
    pml_freq=freq,
)
sp_y = out[0][0]
sp_x = out[1][0]
sp_p = -(out[2][0] + out[4][0])

# Plot
_, ax = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
ax[0, 0].imshow(sy_y.cpu(), aspect="auto", cmap="gray")
ax[0, 0].set_title("$f_y$ $v_y$")
ax[0, 1].imshow(sx_y.cpu(), aspect="auto", cmap="gray")
ax[0, 1].set_title("$f_x$ $v_y$")
ax[0, 2].imshow(sp_y.cpu(), aspect="auto", cmap="gray")
ax[0, 2].set_title("$f_p$ $v_y$")
ax[1, 0].imshow(sy_x.cpu(), aspect="auto", cmap="gray")
ax[1, 0].set_title("$f_y$ $v_x$")
ax[1, 1].imshow(sx_x.cpu(), aspect="auto", cmap="gray")
ax[1, 1].set_title("$f_x$ $v_x$")
ax[1, 2].imshow(sp_x.cpu(), aspect="auto", cmap="gray")
ax[1, 2].set_title("$f_p$ $v_x$")
ax[2, 0].imshow(sy_p.cpu(), aspect="auto", cmap="gray")
ax[2, 0].set_title("$f_y$ $p$")
ax[2, 1].imshow(sx_p.cpu(), aspect="auto", cmap="gray")
ax[2, 1].set_title("$f_x$ $p$")
ax[2, 2].imshow(sp_p.cpu(), aspect="auto", cmap="gray")
ax[2, 2].set_title("$f_p$ $p$")
plt.tight_layout()
plt.savefig("example_elastic_source.jpg")

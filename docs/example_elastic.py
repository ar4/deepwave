import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import elastic

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
ny = 30
nx = 100
dx = 4.0

vp_background = torch.ones(ny, nx, device=device) * 1500
vs_background = torch.ones(ny, nx, device=device) * 1000
rho_background = torch.ones(ny, nx, device=device) * 2200

vp_true = vp_background.clone()
vp_true[10:20, 30:40] = 1600
vs_true = vs_background.clone()
vs_true[10:20, 45:55] = 1100
rho_true = rho_background.clone()
rho_true[10:20, 60:70] = 2300

n_shots = 8

n_sources_per_shot = 1
d_source = 12
first_source = 8
source_depth = 2

n_receivers_per_shot = nx-1
d_receiver = 1
first_receiver = 0
receiver_depth = 2

freq = 15
nt = 200
dt = 0.004
peak_time = 1.5 / freq

# source_locations
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)
source_locations[..., 0] = source_depth
source_locations[:, 0, 1] = (torch.arange(n_shots) * d_source +
                             first_source)

# receiver_locations
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                 dtype=torch.long, device=device)
receiver_locations[..., 0] = receiver_depth
receiver_locations[:, :, 1] = (
    (torch.arange(n_receivers_per_shot) * d_receiver +
     first_receiver)
    .repeat(n_shots, 1)
)

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)

# Create observed data using true models
observed_data = elastic(
    *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_true, vs_true,
                                               rho_true),
    dx, dt,
    source_amplitudes_y=source_amplitudes,
    source_locations_y=source_locations,
    receiver_locations_y=receiver_locations,
    pml_freq=freq,
)[-2]

# Setup optimiser to perform inversion
vp = vp_background.clone().requires_grad_()
vs = vs_background.clone().requires_grad_()
rho = rho_background.clone().requires_grad_()
optimiser = torch.optim.LBFGS([vp, vs, rho])
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 20

for epoch in range(n_epochs):
    def closure():
        optimiser.zero_grad()
        out = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            dx, dt,
            source_amplitudes_y=source_amplitudes,
            source_locations_y=source_locations,
            receiver_locations_y=receiver_locations,
            pml_freq=freq,
        )[-2]
        loss = 1e20*loss_fn(out, observed_data)
        loss.backward()
        return loss

    optimiser.step(closure)

# Plot
vpmin = vp_true.min()
vpmax = vp_true.max()
vsmin = vs_true.min()
vsmax = vs_true.max()
rhomin = rho_true.min()
rhomax = rho_true.max()
_, ax = plt.subplots(2, 3, figsize=(10.5, 5.25), sharex=True,
                     sharey=True)
ax[0, 0].imshow(vp_true.cpu(), aspect='auto', cmap='gray',
                vmin=vpmin, vmax=vpmax)
ax[0, 0].set_title("True vp")
ax[0, 1].imshow(vs_true.cpu(), aspect='auto', cmap='gray',
                vmin=vsmin, vmax=vsmax)
ax[0, 1].set_title("True vs")
ax[0, 2].imshow(rho_true.cpu(), aspect='auto', cmap='gray',
                vmin=rhomin, vmax=rhomax)
ax[0, 2].set_title("True rho")
ax[1, 0].imshow(vp.detach().cpu(), aspect='auto', cmap='gray',
                vmin=vpmin, vmax=vpmax)
ax[1, 0].set_title("Out vp")
ax[1, 1].imshow(vs.detach().cpu(), aspect='auto', cmap='gray',
                vmin=vsmin, vmax=vsmax)
ax[1, 1].set_title("Out vs")
ax[1, 2].imshow(rho.detach().cpu(), aspect='auto', cmap='gray',
                vmin=rhomin, vmax=rhomax)
ax[1, 2].set_title("Out rho")
plt.tight_layout()
plt.savefig('example_elastic.jpg')

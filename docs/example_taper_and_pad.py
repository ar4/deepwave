
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ny = 20
nx = 100
dx = 4.0
v = torch.ones(ny, nx, device=device) * 1500

freq = 25
nt = 40
dt = 0.004
peak_time = 1.5 / freq

# receiver_locations
receiver_locations = torch.zeros(1, nx, 2,
                                 dtype=torch.long, device=device)
receiver_locations[:, :, 1] = torch.arange(nx)

# source_locations
source_locations = torch.tensor([0, nx//2]).long().to(device).reshape(1, 1, 2)

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .to(device)
)

# Propagate
out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
             source_locations=source_locations,
             receiver_locations=receiver_locations,
             pml_freq=freq)
receiver_amplitudes_1 = out[-1][0].detach().clone()

# Propagate with taper and pad
out = scalar(v, dx, dt, source_amplitudes=source_amplitudes,
             source_locations=source_locations,
             receiver_locations=receiver_locations,
             pml_freq=freq,
             freq_taper_frac=0.2,
             time_pad_frac=0.2)
receiver_amplitudes_2 = out[-1][0].detach().clone()

# Plot receiver amplitudes
vmin, vmax = torch.quantile(receiver_amplitudes_1,
                            torch.tensor([0.05, 0.95]).to(device))
_, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharex=True, sharey=True)
ax[0].imshow(receiver_amplitudes_1.cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Time Sample')
ax[0].set_title('No taper or pad')
ax[1].imshow(receiver_amplitudes_2.cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[1].set_title('0.2 taper and pad')
plt.tight_layout()
plt.savefig('example_taper_and_pad.jpg')

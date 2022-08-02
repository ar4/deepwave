import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ny = 100
nx = 300
dx = 4.0
v = torch.ones(ny, nx, device=device) * 1500

freq = 25
nt = 40
dt = 0.004
peak_time = 1.5 / freq

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(1, 3, 1)
    .to(device)
)

# Location interpolation
source_locations = torch.zeros(1, 3, 2, device=device)
source_locations[..., 0] = ny / 2 + 0.5
source_locations[..., 1] = torch.arange(3) * nx / 3 + nx / 6 + 0.5
source_monopole = torch.zeros(1, 3, dtype=torch.bool, device=device)
source_monopole[0, 0] = True
source_monopole[0, 1] = False
source_monopole[0, 2] = False
source_dipole_dim = torch.zeros(1, 3, dtype=torch.int, device=device)
source_dipole_dim[0, 2] = 1
hicks_source = deepwave.location_interpolation.Hicks(
    source_locations,
    monopole=source_monopole,
    dipole_dim=source_dipole_dim
)
hicks_source_locations = hicks_source.get_locations()
hicks_source_amplitudes = hicks_source.source(source_amplitudes)

receiver_locations = torch.zeros(1, 3, 2, device=device)
receiver_locations[..., 0] = source_locations[0, 0, 0] - ny / 8
receiver_locations[..., 1] = source_locations[0, 0, 1]
receiver_monopole = torch.zeros(1, 3, dtype=torch.bool, device=device)
receiver_monopole[0, 0] = True
receiver_monopole[0, 1] = False
receiver_monopole[0, 2] = False
receiver_dipole_dim = torch.zeros(1, 3, dtype=torch.int, device=device)
receiver_dipole_dim[0, 2] = 1
hicks_receiver = deepwave.location_interpolation.Hicks(
    receiver_locations,
    monopole=receiver_monopole,
    dipole_dim=receiver_dipole_dim
)
hicks_receiver_locations = hicks_receiver.get_locations()

# Propagate
out = scalar(v, dx, dt, source_amplitudes=hicks_source_amplitudes,
             source_locations=hicks_source_locations,
             receiver_locations=hicks_receiver_locations,
             pml_freq=freq)

# Plot wavefield
wavefield = out[0]
plt.figure(figsize=(10.5, 7))
plt.imshow(wavefield[0].cpu(), cmap='gray')
plt.savefig('example_location_interpolation_wavefield.jpg')

# Plot receiver amplitudes
receiver_hicks_amplitudes = out[-1]
receiver_amplitudes = hicks_receiver.receiver(receiver_hicks_amplitudes)
plt.figure(figsize=(10.5, 7))
plt.plot(receiver_amplitudes[0, 0].cpu(), label='Monopole')
plt.plot(receiver_amplitudes[0, 1].cpu(), label='Dipole 0')
plt.plot(receiver_amplitudes[0, 2].cpu(), label='Dipole 1')
plt.legend()
plt.xlabel('Time sample')
plt.ylabel('Receiver amplitude')
plt.savefig('example_location_interpolation_receiver_data.jpg')

# Plot comparison with grid-centred points
out2 = scalar(v, dx, dt,
              source_amplitudes=source_amplitudes,
              source_locations=source_locations.long(),
              receiver_locations=receiver_locations.long(),
              pml_freq=freq)

v = torch.ones(2 * ny, 2 * nx, device=device) * 1500
out3 = scalar(v, dx / 2, dt,
              source_amplitudes=source_amplitudes*4,
              source_locations=(source_locations*2).long(),
              receiver_locations=(receiver_locations*2).long(),
              pml_freq=freq)

receiver_amplitudes2 = out2[-1]
receiver_amplitudes3 = out3[-1]
plt.figure(figsize=(10.5, 7))
plt.plot(receiver_amplitudes[0, 0].cpu(), label='Hicks', linewidth=6)
plt.plot(receiver_amplitudes2[0, 0].cpu(), 'g', label='Without Hicks')
plt.plot(receiver_amplitudes3[0, 0].cpu(), 'r', label='Denser grid')
plt.legend()
plt.xlabel('Time sample')
plt.ylabel('Receiver amplitude')
plt.savefig('example_location_interpolation_comparison.jpg')

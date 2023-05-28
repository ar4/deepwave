import math
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
ny = 300
nx = 300
dx = 4.0
v = torch.ones(ny, nx, device=device) * 1500

freq = 25
nt = 50
dt = 0.004
peak_time = 1.5 / freq

cy, cx = (ny/2, nx/2)  # center of circle
radius = ny/3  # radius of circle
n_shots = 1
n_source_locations = 16
# multiply the number of sources per shot by two because we pass two
# at each location to the Hicks function. One of these two will be
# oriented in the y dimension and the other in the x dimension. By
# adjusting the relative amplitude of these two sources we can
# simulate an oriented source.
n_sources_per_shot = n_source_locations * 2

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)

for l in range(n_source_locations):
    angle = l / n_source_locations * 2 * math.pi
    source_locations[0, l*2:(l+1)*2, 0] = (
        math.sin(angle) * radius + cy
    )
    source_locations[0, l*2:(l+1)*2, 1] = (
        math.cos(angle) * radius + cx
    )
    source_amplitudes[0, l*2] *= math.sin(angle)
    source_amplitudes[0, l*2+1] *= math.cos(angle)

source_monopole = torch.zeros(n_shots, n_sources_per_shot,
                              dtype=torch.bool, device=device)
source_dipole_dim = torch.zeros(n_shots, n_sources_per_shot,
                                dtype=torch.int, device=device)

source_dipole_dim[0, 1::2] = 1  # every second source is x-oriented
hicks_source = deepwave.location_interpolation.Hicks(
    source_locations,
    monopole=source_monopole,
    dipole_dim=source_dipole_dim,
)

hicks_source_locations = hicks_source.get_locations()
hicks_source_amplitudes = hicks_source.source(source_amplitudes)

out = scalar(v, dx, dt, source_amplitudes=hicks_source_amplitudes,
             source_locations=hicks_source_locations,
             pml_freq=freq)

# Plot wavefield
wavefield = out[0]
plt.figure(figsize=(8, 8))
plt.imshow(wavefield[0].cpu(), cmap='gray')
plt.savefig('example_location_interpolation_circle.jpg')

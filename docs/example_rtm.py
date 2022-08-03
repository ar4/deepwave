import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar_born

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ny = 2301
nx = 751
dx = 4.0
v = torch.from_file('marmousi_vp.bin',
                    size=ny*nx).reshape(ny, nx)

# Smooth to use as migration model
v_mig = torch.tensor(1/gaussian_filter(1/v.numpy(), 40))
v_mig = v_mig[::2, ::2].to(device)
ny = v_mig.shape[0]
nx = v_mig.shape[1]
dx *= 2

n_shots = 115

n_sources_per_shot = 1
d_source = 10  # 10 * 8m = 80m
first_source = 5  # 5 * 8m = 40m
source_depth = 1  # 1 * 8m = 8m

n_receivers_per_shot = 384
d_receiver = 3  # 3 * 8m = 24m
first_receiver = 0  # 0 * 8m = 0m
receiver_depth = 1  # 1 * 8m = 8m

freq = 25
nt = 750
dt = 0.004
peak_time = 1.5 / freq

# source_locations
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source

# receiver_locations
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                 dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = (
    (torch.arange(n_receivers_per_shot) * d_receiver + first_receiver)
    .repeat(n_shots, 1)
)

# source_amplitudes
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, n_sources_per_shot, 1)
    .to(device)
)

# Load observed data
observed_data = (
    torch.from_file('marmousi_data.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
    .to(device)
)

# Create mask to attenuate direct arrival
mask = torch.ones_like(observed_data)
flat_len = 100
taper_len = 200
taper = torch.cos(torch.arange(taper_len)/taper_len * torch.pi/2)
mute_len = flat_len + 2*taper_len
mute = torch.zeros(mute_len, device=device)
mute[:taper_len] = taper
mute[-taper_len:] = taper.flip(0)
v_direct = 1700
for shot_idx in range(n_shots):
    sx = (shot_idx * d_source + first_source) * dx
    for receiver_idx in range(n_receivers_per_shot):
        rx = (receiver_idx * d_receiver + first_receiver) * dx
        dist = abs(sx - rx)
        arrival_time = dist / v_direct / dt
        mute_start = int(arrival_time) - mute_len//2
        mute_end = mute_start + mute_len
        if (mute_start > nt):
            continue
        actual_mute_start = max(mute_start, 0)
        actual_mute_end = min(mute_end, nt)
        mask[shot_idx, receiver_idx, actual_mute_start:actual_mute_end] = \
            mute[actual_mute_start-mute_start:actual_mute_end-mute_start]
observed_scatter_masked = observed_data * mask

# Create scattering amplitude that we will invert for
scatter = torch.zeros_like(v_mig)
scatter.requires_grad_()

# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([scatter], lr=1)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 1
n_batch = 46
n_shots_per_batch = (n_shots + n_batch - 1) // n_batch
for epoch in range(n_epochs):
    def closure():
        epoch_loss = 0
        optimiser.zero_grad()
        for batch in range(n_batch):
            batch_start = batch * n_shots_per_batch
            batch_end = min(batch_start + n_shots_per_batch, n_shots)
            if batch_end <= batch_start:
                continue
            out = scalar_born(
                v_mig, scatter, dx, dt,
                source_amplitudes=source_amplitudes[batch_start:batch_end],
                source_locations=source_locations[batch_start:batch_end],
                receiver_locations=receiver_locations[batch_start:batch_end],
                pml_freq=freq
            )
            loss = (
                1e9 * loss_fn(out[-1] * mask[batch_start:batch_end],
                              observed_scatter_masked[batch_start:batch_end])
            )
            epoch_loss += loss.item()
            loss.backward()
        print(epoch_loss)
        return epoch_loss
    optimiser.step(closure)

# Plot
vmin, vmax = torch.quantile(scatter.detach(),
                            torch.tensor([0.05, 0.95]).to(device))
plt.figure(figsize=(10.5, 3.5))
plt.imshow(scatter.detach().cpu().T, aspect='auto', cmap='gray',
           vmin=vmin, vmax=vmax)
plt.savefig('example_rtm.jpg')

scatter.detach().cpu().numpy().tofile('marmousi_scatter.bin')

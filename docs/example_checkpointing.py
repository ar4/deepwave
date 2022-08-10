import torch
import torch.utils.checkpoint
from torchaudio.functional import biquad
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ny = 2301
nx = 751
dx = 4.0
v_true = torch.from_file('marmousi_vp.bin',
                         size=ny*nx).reshape(ny, nx)

# Select portion of model for inversion
ny = 600
nx = 250
v_true = v_true[:ny, :nx]

# Smooth to use as starting model
v_init = torch.tensor(1/gaussian_filter(1/v_true.numpy(), 40))

n_shots = 115

n_sources_per_shot = 1
d_source = 20  # 20 * 4m = 80m
first_source = 10  # 10 * 4m = 40m
source_depth = 2  # 2 * 4m = 8m

n_receivers_per_shot = 384
d_receiver = 6  # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 2  # 2 * 4m = 8m

freq = 25
nt = 750
dt = 0.004
peak_time = 1.5 / freq

observed_data = (
    torch.from_file('marmousi_data.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
)


# Select portion of data for inversion
n_shots = 20
n_receivers_per_shot = 100
nt = 300
observed_data = (
    observed_data[:n_shots, :n_receivers_per_shot, :nt].to(device)
)

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
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)

# Upsample source_amplitudes
max_vel = 2500
dt, step_ratio = deepwave.common.cfl_condition(dx, dx, dt, max_vel)
source_amplitudes = deepwave.common.upsample(source_amplitudes, step_ratio)
observed_data = deepwave.common.upsample(observed_data, step_ratio)
nt = source_amplitudes.shape[-1]


# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(torch.logit((initial - min_vel) /
                                                    (max_vel - min_vel)))

    def forward(self):
        return (torch.sigmoid(self.model) * (self.max_vel - self.min_vel) +
                self.min_vel)


model = Model(v_init, 1000, 2500).to(device)

# Setup optimiser to perform inversion
optimiser = torch.optim.LBFGS(model.parameters(),
                              line_search_fn='strong_wolfe')
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 2
n_segments = 5
cutoff_freq = 10

sos = butter(6, cutoff_freq, fs=1/dt, output='sos')
sos = [torch.tensor(sosi).to(observed_data.dtype).to(device)
       for sosi in sos]


def taper(x):
    return deepwave.common.cosine_taper_end(x, 600)


def filt(x):
    return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])


def wrap(v, chunk, wavefield_0, wavefield_m1, psiy_m1, psix_m1,
         zetay_m1, zetax_m1):
    return scalar(
                  v, dx, dt,
                  source_amplitudes=chunk,
                  source_locations=source_locations,
                  receiver_locations=receiver_locations,
                  pml_freq=freq,
                  wavefield_0=wavefield_0,
                  wavefield_m1=wavefield_m1,
                  psiy_m1=psiy_m1,
                  psix_m1=psix_m1,
                  zetay_m1=zetay_m1,
                  zetax_m1=zetax_m1,
                  max_vel=max_vel,
                  model_gradient_sampling_interval=step_ratio
              )


observed_data = filt(taper(observed_data))

for epoch in range(n_epochs):
    def closure():
        pml_width = 20
        wavefield_size = [n_shots, ny + 2 * pml_width, nx + 2 * pml_width]
        wavefield_0 = torch.zeros(*wavefield_size, device=device)
        wavefield_m1 = torch.zeros(*wavefield_size, device=device)
        psiy_m1 = torch.zeros(*wavefield_size, device=device)
        psix_m1 = torch.zeros(*wavefield_size, device=device)
        zetay_m1 = torch.zeros(*wavefield_size, device=device)
        zetax_m1 = torch.zeros(*wavefield_size, device=device)
        optimiser.zero_grad()
        receiver_amplitudes = torch.zeros(n_shots, n_receivers_per_shot, nt,
                                          device=device)
        v = model()
        k = 0
        for i, chunk in enumerate(torch.chunk(source_amplitudes, n_segments,
                                              dim=-1)):
            if i == n_segments - 1:
                (wavefield_0, wavefield_m1, psiy_m1, psix_m1,
                 zetay_m1, zetax_m1, receiver_amplitudes_chunk) = \
                    wrap(v, chunk, wavefield_0,
                         wavefield_m1,
                         psiy_m1, psix_m1,
                         zetay_m1, zetax_m1)
            else:
                (wavefield_0, wavefield_m1, psiy_m1, psix_m1,
                 zetay_m1, zetax_m1, receiver_amplitudes_chunk) = \
                 torch.utils.checkpoint.checkpoint(wrap, v, chunk, wavefield_0,
                                                   wavefield_m1,
                                                   psiy_m1, psix_m1,
                                                   zetay_m1, zetax_m1)
            receiver_amplitudes[..., k:k+chunk.shape[-1]] = \
                receiver_amplitudes_chunk
            k += chunk.shape[-1]
        receiver_amplitudes = filt(taper(receiver_amplitudes))
        loss = 1e6 * loss_fn(receiver_amplitudes, observed_data)
        loss.backward()
        return loss

    optimiser.step(closure)

# Plot
v = model()
vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
ax[0].imshow(v_init.cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.detach().cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu().T, aspect='auto', cmap='gray',
             vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig('example_checkpointing.jpg')

v.detach().cpu().numpy().tofile('marmousi_v_inv.bin')

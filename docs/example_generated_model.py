"""This script demonstrates how to use a custom model generation function
within Deepwave's optimization framework to invert for layered velocities.
"""

import matplotlib.pyplot as plt
import torch

import deepwave


class Model(torch.nn.Module):
    """A layered model generator.

    Inputs:
        layer_thickness:
            The number of cells thick each layer is in the first
            dimension
        nx:
            The number of cells of the model in the second dimension
    """

    def __init__(self, layer_thickness, nx):
        super().__init__()
        self.layer_thickness = layer_thickness
        self.nx = nx

    def forward(self, x):
        """Generate model.

        Inputs:
            x:
                A Tensor of layer velocities.

        Returns:
            A layered model with the specified velocities.

        """
        return (
            x.reshape(-1, 1)
            .repeat(1, self.layer_thickness)
            .reshape(-1, 1)
            .repeat(1, self.nx)
        )


layer_thickness = 5
nx_model = 100
m = Model(layer_thickness, nx_model)

x_true = torch.tensor([1500.0, 1600.0, 1750.0, 1920.0])
v_true = m(x_true)

n_shots = 20
freq = 10
nt = 200
dx = 4.0
dt = 0.004
peak_time = 1.5 / freq
source_locations = torch.zeros(n_shots, 1, 2)
source_locations[:, 0, 1] = torch.arange(n_shots) * 5 + 2
receiver_locations = torch.zeros(n_shots, nx_model - 1, 2)
receiver_locations[..., 1] = torch.arange(nx_model - 1).repeat(n_shots, 1)
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_shots, 1, 1)
)

# Generate "true" observed data
observed_data = deepwave.scalar(
    v_true,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_freq=freq,
)[-1]

# Start from initial guess (all layers 1750) and try to invert
x_init = 1750.0 * torch.ones(4)
x = x_init.clone().requires_grad_()
opt = torch.optim.Adam([x], lr=10.0)
loss_fn = torch.nn.MSELoss()


def closure():
    opt.zero_grad()
    v = m(x)
    y = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
    )[-1]
    loss = loss_fn(y, observed_data)
    loss.backward()
    return loss


for i in range(100):
    opt.step(closure)

print(x.detach())

_, ax = plt.subplots(3, 1, figsize=(10.5, 10.5))
vmin = 1500
vmax = 2000
v_init = m(x_init)
v = m(x.detach())
ax[0].imshow(v_init, aspect="auto", vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v, aspect="auto", vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true, aspect="auto", vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig("example_generated_model.jpg")

# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# These examples use the Deepwave 0.1 interface and require PyTorch 2.4
# or later, as declared by Deepwave. Install plotting support too.
# %pip install "deepwave>=0.1.0" matplotlib

# %% [markdown]
# # Generated model
#
# Because Deepwave provides wave propagation as a PyTorch module, we can
# use it as part of a chain of operations, with PyTorch doing forward
# modelling and backpropagation end-to-end. We have already seen examples
# of applying operations to the output of wave propagation (loss
# functions) and before wave propagation (a function that generates the
# velocity model, constrained to only have velocities within a specified
# range). This example shows another instance of using a function to
# generate the velocity model.
#
# We use a function to generate a layered velocity model. Given a list of
# the velocities of the different layers, it expands those into the model
# that we provide to Deepwave.

# %%
"""Demonstrates custom model generation for layered velocity inversion."""

import math
import os

import matplotlib.pyplot as plt
import torch

import deepwave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"


# %%
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
        """Initialises the Model.

        Args:
            layer_thickness (int): The number of cells thick each layer
                                   is in the first dimension.
            nx (int): The number of cells of the model
                      in the second dimension.
        """
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


# %% [markdown]
# Let's pick some velocities to use in our "true" model, and set up the
# survey (source/receiver geometry) and the source wavelet.

# %%
layer_thickness = 3 if smoke else 5
nx_model = 40 if smoke else 100
m = Model(layer_thickness, nx_model)

x_true = torch.tensor([1500.0, 1600.0, 1750.0, 1920.0], device=device)
v_true = m(x_true)

n_shots = 6 if smoke else 20
freq = 10
nt = 80 if smoke else 200
dx = 4.0
dt = 0.004
peak_time = 1.5 / freq
source_locations = torch.zeros(n_shots, 1, 2, device=device)
source_locations[:, 0, 1] = torch.arange(n_shots) * 5 + 2
receiver_locations = torch.zeros(n_shots, nx_model - 1, 2, device=device)
receiver_locations[..., 1] = torch.arange(nx_model - 1).repeat(n_shots, 1)
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .reshape(1, 1, -1)
    .repeat(n_shots, 1, 1)
    .to(device)
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
).receiver_amplitudes

# %% [markdown]
# Now we invert for the four layer velocities, starting from an initial
# guess that all layers have the same (wrong) velocity.

# %%
# Start from initial guess (all layers 1750) and try to invert
x_init = 1750.0 * torch.ones(4, device=device)
x = x_init.clone().requires_grad_()
opt = torch.optim.Adam([x], lr=10.0)
loss_fn = torch.nn.MSELoss()


loss_history: list[float] = []


def closure():
    """Closure function for the Adam optimiser."""
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
    ).receiver_amplitudes
    loss = loss_fn(y, observed_data)
    loss.backward()
    loss_history.append(float(loss.detach()))
    return loss


for _i in range(10 if smoke else 100):
    opt.step(closure)

print(x.detach())

# %% [markdown]
# Finally, we plot the initial, recovered, and true models.

# %%
_, ax = plt.subplots(3, 1, figsize=(10.5, 10.5))
vmin = 1500
vmax = 2000
v_init = m(x_init)
v = m(x.detach())
ax[0].imshow(v_init.cpu(), aspect="auto", vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.cpu(), aspect="auto", vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu(), aspect="auto", vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig("example_generated_model.jpg")
plt.show()

# %%
assert all(math.isfinite(loss) for loss in loss_history), (
    "loss history contains non-finite values"
)
assert loss_history[-1] < loss_history[0], "loss did not decrease"

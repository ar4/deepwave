"""Demonstrates Hessian matrix calculation in Deepwave.

This script shows how to compute the Hessian and use it in a
Newton-Raphson optimization scheme.
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.sparse.linalg import eigsh

import deepwave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

ny = 20
nx = 30
freq = 25
nt = 80
dt = 0.004
dx = 4

v_true = 1500 * torch.ones(ny, nx, dtype=dtype, device=device)
v_true[10:] += 200
v_init = torchvision.transforms.functional.gaussian_blur(
    v_true[None],
    [31, 31],
).squeeze()
v = v_init.clone().requires_grad_()

source_locations = torch.tensor(
    [[[1, 15]]], dtype=torch.long, device=device
)
source_amplitudes = (
    deepwave.wavelets.ricker(
        freq,
        nt,
        dt,
        1.3 / freq,
        dtype=dtype,
    )
    .reshape(1, 1, -1)
    .to(device)
)
receiver_locations = torch.ones(
    1, nx - 20, 2, dtype=torch.long, device=device
)
receiver_locations[0, :, 1] = torch.arange(10, nx - 10)

d_true = deepwave.scalar(
    v_true,
    grid_spacing=dx,
    dt=dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=[0, 20, 20, 20],
    max_vel=2000,
)[-1]

loss_fn = torch.nn.MSELoss()


def wrap(v):
    """Wraps the scalar propagation function for Hessian calculation."""
    d = deepwave.scalar(
        v,
        grid_spacing=dx,
        dt=dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=[0, 20, 20, 20],
        max_vel=2000,
    )[-1]
    return loss_fn(d, d_true)


hess = torch.autograd.functional.hessian(wrap, v).detach()
wrap(v).backward()
grad = v.grad.detach()

_, ax = plt.subplots(1, 2, figsize=(10.5, 7))
vmax = torch.quantile(hess, 0.95).item()
ax[0].imshow(
    hess.reshape(v.numel(), v.numel()).cpu(),
    aspect="auto",
    cmap="gray",
    vmin=-vmax,
    vmax=vmax,
)
ax[0].set_title("Hessian")
ax[1].imshow(
    hess[10, 10].cpu(), aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax
)
ax[1].set_title("Hessian at (10, 10)")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")
plt.tight_layout()
plt.savefig("example_hessian.jpg")

eig0 = eigsh(
    hess.cpu().numpy().reshape(v.numel(), v.numel()), k=1, which="SA"
)[0].item()
tau = max(-1.5 * eig0, 0)
L = torch.linalg.cholesky(
    hess.reshape(v.numel(), v.numel())
    + tau * torch.eye(v.numel(), dtype=dtype, device=device),
)
h = torch.cholesky_solve(grad.reshape(-1, 1).neg(), L).reshape(ny, nx)

_, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharex=True, sharey=True)
ax[0].imshow(grad.cpu(), aspect="auto", cmap="gray")
ax[0].set_title("Gradient")
ax[1].imshow(h.cpu(), aspect="auto", cmap="gray")
ax[1].set_title("Inverse Hessian times gradient")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
plt.tight_layout()
plt.savefig("example_hessian_vs_gradient.jpg")

for _epoch in range(3):
    hess = torch.autograd.functional.hessian(wrap, v).detach()
    wrap(v).backward()
    grad = v.grad.detach()
    eig0 = eigsh(
        hess.cpu().numpy().reshape(v.numel(), v.numel()), k=1, which="SA"
    )[0].item()
    tau = max(-1.5 * eig0, 0)
    L = torch.linalg.cholesky(
        hess.reshape(v.numel(), v.numel())
        + tau * torch.eye(v.numel(), dtype=v.dtype, device=v.device),
    )
    h = torch.cholesky_solve(grad.reshape(-1, 1).neg(), L).reshape(ny, nx)
    v = (v.detach() + h).requires_grad_()

_, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
vmin = 1400
vmax = 1800
ax[0].imshow(v_init.cpu(), vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.detach().cpu(), vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu(), vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig("example_hessian_result.jpg")

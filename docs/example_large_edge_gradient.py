import torch
import torchvision
import deepwave
import matplotlib.pyplot as plt

source_amplitudes = deepwave.wavelets.ricker(25, 200, 0.004,
                                             0.06).reshape(1, 1, -1)
source_locations = torch.tensor([[[0, 10]]])
receiver_locations = torch.tensor([[[0, 90]]])

v_true = 1500 * torch.ones(50, 100)
v = 1600 * torch.ones(50, 100)
v.requires_grad_()


# With large gradient at edges

out_true = deepwave.scalar(
    v_true, grid_spacing=4, dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations
)[-1]

out = deepwave.scalar(
    v, grid_spacing=4, dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations
)[-1]

torch.nn.MSELoss()(out, out_true).backward()

plt.figure(figsize=(10.5, 5))
plt.imshow(v.grad.detach())
plt.title("Gradient, with large edge values")
plt.savefig("example_large_edge_gradient1.jpg")


# With free surface, reducing gradient at top edge

v.grad = None

out_true = deepwave.scalar(
    v_true, grid_spacing=4, dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=[0, 20, 20, 20]  # <--
)[-1]

out = deepwave.scalar(
    v, grid_spacing=4, dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=[0, 20, 20, 20]  # <--
)[-1]

torch.nn.MSELoss()(out, out_true).backward()

plt.figure(figsize=(10.5, 5))
plt.imshow(v.grad.detach())
plt.title("Gradient, with free surface")
plt.savefig("example_large_edge_gradient2.jpg")


# Model smoothed before being passed to Deepwave,
# resulting in smooth gradient

v.grad = None

out_true = deepwave.scalar(
    v_true, grid_spacing=4, dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations
)[-1]

# Smooth model
v_smooth = (
    torchvision.transforms.functional.gaussian_blur(
        v[None], [11, 11]
    ).squeeze()
)

out = deepwave.scalar(
    v_smooth,  # <-- Pass smoothed model to Deepwave
    grid_spacing=4, dt=0.004,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations
)[-1]

torch.nn.MSELoss()(out, out_true).backward()

plt.figure(figsize=(10.5, 5))
plt.imshow(v.grad.detach())
plt.title("Gradient, using smoothed model")
plt.savefig("example_large_edge_gradient3.jpg")

# Apply gradient and forward through smoothing
optimiser = torch.optim.SGD([v], lr=1e9)
optimiser.step()
v_smooth = (
    torchvision.transforms.functional.gaussian_blur(
        v[None], [11, 11]
    ).squeeze()
)

plt.figure(figsize=(10.5, 5))
plt.imshow(v_smooth.detach())
plt.title("Smooth model at the next iteration")
plt.savefig("example_large_edge_gradient4.jpg")

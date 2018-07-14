# Deepwave
[![Build Status](https://travis-ci.org/ar4/deepwave.svg?branch=master)](https://travis-ci.org/ar4/deepwave)[![Codacy Badge](https://api.codacy.com/project/badge/Grade/52d27677ef0a439195d574964a6b4be4)](https://www.codacy.com/app/ar4/deepwave?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ar4/deepwave&amp;utm_campaign=Badge_Grade)

Deepwave provides wave propagation modules for PyTorch (currently only for the constant density acoustic / scalar wave equation). It is primarily aimed at seismic imaging/inversion. Potential applications include:

* Seismic Full-Waveform Inversion (FWI) using the PyTorch framework
* Integrating wave propagation into PyTorch deep networks
* Using the wave propagators (written in pure C) and ignoring all of the Python parts

Wave propagation and FWI [can be implemented using deep neural networks](https://arxiv.org/abs/1801.07232). Deep learning tools such as TensorFlow and PyTorch are currently too slow and memory inefficient to do this with realistic seismic datasets, however. Deepwave extends PyTorch with high performance wave propagation modules so that you can benefit from the conveniences of PyTorch without sacrificing performance.

## Installation
Deepwave needs NumPy, so [installing Anaconda](https://www.anaconda.com/download) first is highly recommended. It also needs [PyTorch](https://pytorch.org/), so it is also best to install that (using something like `conda install pytorch-cpu -c pytorch`) before installing Deepwave. You can then install Deepwave using

`pip install deepwave`


## Usage
Deepwave can do two things: forward modeling and backpropagation.

### Forward modeling
We first need to create a wave propagation module. If you are familiar with training neural networks, this is the same as defining your network (or a portion of it) and initializing the weights.

```python
prop = deepwave.scalar.Propagator(model, dx)
```

The two required arguments are the wave speed model (`model`), and the cell size used in the model (`dx`). The model should be provided as a PyTorch Float Tensor of shape `[1, nz, (ny, (nx))]` (the first dimension is used to store different properties, but for the scalar wave propagator the only property is wave speed). 1D, 2D, and 3D propagators are available, with the model shape used to choose between them. If you want 1D wave propagation, then the model shape would be `[1, nz]`, for example. The cell size should also be provided as a Float Tensor containing the cell size in each spatial dimension; `[dz, dy, dx]` in 3D, `[dz]` in 1D.

Now we can call our propagator.

```python
receiver_amplitudes = prop.forward(source_amplitudes, x_s, x_r, dt)
```

The required arguments are the source amplitudes, the source coordinates (`x_s`), the receiver coordinates (`x_r`), and the time interval between source samples (`dt`). Receiver amplitudes are returned. The source amplitudes should be a Float Tensor of shape `[nt, num_shots, num_sources_per_shot]`, where `nt` is the number of time samples. Note that you can have more than one source per shot, although in active source seismic experiments there is normally only one. The source coordinates array is a Float Tensor of shape `[num_shots, num_sources_per_shot, num_dimensions]`, where `num_dimensions` corresponds to the number of dimensions in the model. Similarly, the receiver coordinates array is a Float Tensor of shape `[num_shots, num_receivers_per_shot, num_dimensions]`. The coordinates are specified in the same units as `dx` and are with respect to the origin of the model. The time step size `dt` should be a regular float. The returned Float Tensor of shape `[nt, num_shots, num_receivers_per_shot]` contains receiver amplitudes.

You can run forward propagation multiple times with different inputs, for example to forward model each shot in a dataset. You may have noticed that the propagator is also designed to forward model multiple separate shots simultaneously (the `num_shots` dimension in the inputs). Doing so may give better performance than propagating each shot separately, but you might also run out of memory.

Here is a full example:

```python
import math
import torch
import deepwave

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = torch.arange(length) * dt - peak_time
    y = (1 - 2 * math.pi**2 * freq**2 * t**2) \
            * torch.exp(-math.pi**2 * freq**2 * t**2)
    return y

dx = torch.Tensor([5.0, 5.0])
dt = 0.004 # 4ms
nz = 20
ny = 40
nt = int(1 / dt) # 1s

# constant 1500m/s model
model = torch.ones(1, nz, ny) * 1500

# one source and receiver at the same location
x_s = torch.Tensor([[[0, 20 * dx[1]]]])
x_r = x_s.clone()

source_amplitudes = ricker(4, nt, dt, 0.3).reshape(-1, 1, 1)

prop = deepwave.scalar.Propagator(model, dx)
receiver_amplitudes = prop.forward(source_amplitudes, x_s, x_r, dt)
```

### Backpropagation/Inversion (FWI)
FWI attempts to match the predicted and true receiver amplitudes by modifying the model. This is similar to modifying the weights in a neural network during training. To achieve this with Deepwave, we first specify that we will need to calculate gradients with respect to our model.

```python
model.requires_grad = True
```

We also need to specify which optimization algorithm and loss/cost/objective function we wish to use.

```python
optimizer = torch.optim.Adam([model], lr=1)
criterion = torch.nn.MSELoss()
```

I have chosen to use the Adam optimizer with a learning rate of 1, and mean squared error (MSE) as the cost function.

With our propagator initialized as in the forward modeling section above, we can then iteratively optimize the model.

```python
for iter in range(num_iter):
    optimizer.zero_grad()
    receiver_amplitudes_pred = prop.forward(source_amplitudes, x_s, x_r, dt)
    loss = criterion(receiver_amplitudes_pred, receiver_amplitudes_true)
    loss.backward()
    optimizer.step()
```

In each iteration, we zero the gradients, calculate the predicted receiver amplitudes from the current model, compare these with the true data using our cost function, backpropagate to calculate the gradient of the loss with respect to the model, and then update the model.

The flexibility of PyTorch means that it is easy to modify this to suit your own needs.

For a full example of using Deepwave for FWI, see [this notebook](https://colab.research.google.com/drive/1PMO1rFAaibRjwjhBuyH3dLQ1sW5wfec_).

## Contributing
Your help to improve Deepwave would be greatly appreciated. If you encounter any difficulty using Deepwave, please report it as a Github Issue so that it can be fixed. If you have feature suggestions or other ideas to make Deepwave better, please also report those as Github Issues. If you want to help with the coding, that would be especially wonderful. The Github Issues contain a list of things that need work. If you see one that you would like to attempt, please ask for it to be assigned to you.

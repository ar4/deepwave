# Deepwave
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/52d27677ef0a439195d574964a6b4be4)](https://www.codacy.com/app/ar4/deepwave?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ar4/deepwave&amp;utm_campaign=Badge_Grade)

Deepwave provides wave propagation modules for PyTorch (currently only for the constant density acoustic / scalar wave equation). It is primarily aimed at seismic imaging/inversion. One interesting advantage of this is that it allows you to chain operations together. You could, for example, use Deepwave to perform FWI using a custom cost function (as long as PyTorch is able to automatically differentiate it), or add some other operations before and after wave propagation and PyTorch will backpropagate through all of them.

Wave propagation and FWI [can be implemented using deep neural networks](https://arxiv.org/abs/1801.07232). Deep learning tools such as TensorFlow and PyTorch are currently too slow and memory inefficient to do this with realistic seismic datasets, however. Deepwave extends PyTorch with higher performance wave propagation modules (written in C and CUDA) so that you can benefit from the conveniences of PyTorch without sacrificing performance.

Deepwave runs on CPUs and NVIDIA GPUs. It should automatically detect compatible GPUs during installation and compile the GPU code if any are found. To run on the GPU you must transfer the model, source amplitude, and source and receiver location Tensors to the GPU in the standard PyTorch way (using `.cuda()` or `.to(device)`).

## Installation
Deepwave needs NumPy, so [installing Anaconda](https://www.anaconda.com/download) first is highly recommended. It also needs [PyTorch](https://pytorch.org/), so it is also best to install that (using something like `conda install pytorch-cpu -c pytorch`) before installing Deepwave. You can then install the latest release of Deepwave using

`pip install deepwave`

or the latest development version using

`pip install git+https://github.com/ar4/deepwave.git`


## Usage
Deepwave can do two things: forward modeling and backpropagation.

### Forward modeling
We first need to create a wave propagation module. If you are familiar with training neural networks, this is the same as defining your network (or a portion of it) and initializing the weights.

```python
prop = deepwave.scalar.Propagator({'vp': model}, dx)
```

The two required arguments are the wave speed model (`model`), and the cell size used in the model (`dx`). The model should be provided as a PyTorch Float Tensor of shape `[nz, (ny, (nx))]`. 1D, 2D, and 3D propagators are available, with the model shape used to choose between them. If you want 1D wave propagation, then the model shape would be `[nz]`, for example. The cell size should also be provided as a single float (if it is the same in every dimension) or a Float Tensor containing the cell size in each spatial dimension; `[dz, dy, dx]` in 3D, `[dz]` in 1D.

Now we can call our propagator.

```python
receiver_amplitudes = prop(source_amplitudes, x_s, x_r, dt)
```

The required arguments are the source amplitudes, the source coordinates (`x_s`), the receiver coordinates (`x_r`), and the time interval between source samples (`dt`). Receiver amplitudes are returned. The source amplitudes should be a Float Tensor of shape `[nt, num_shots, num_sources_per_shot]`, where `nt` is the number of time samples. Note that you can have more than one source per shot, although in active source seismic experiments there is normally only one. The source coordinates array is a Float Tensor of shape `[num_shots, num_sources_per_shot, num_dimensions]`, where `num_dimensions` corresponds to the number of dimensions in the model. Similarly, the receiver coordinates array is a Float Tensor of shape `[num_shots, num_receivers_per_shot, num_dimensions]`. The coordinates are specified in the same units as `dx` and are with respect to the origin of the model. The time step size `dt` should be a regular float. The returned Float Tensor of shape `[nt, num_shots, num_receivers_per_shot]` contains receiver amplitudes.

You can run forward propagation multiple times with different inputs, for example to forward model each shot in a dataset. You may have noticed that the propagator is also designed to forward model multiple separate shots simultaneously (the `num_shots` dimension in the inputs). Doing so may give better performance than propagating each shot separately, but you might also run out of memory.

Here is a full example:

```python
import math
import torch
import deepwave

dx = 5.0 # 5m in each dimension
dt = 0.004 # 4ms
nz = 20
ny = 40
nt = int(1 / dt) # 1s
peak_freq = 4
peak_source_time = 1/peak_freq

# constant 1500m/s model
model = torch.ones(nz, ny) * 1500

# one source and receiver at the same location
x_s = torch.Tensor([[[0, 20 * dx]]])
x_r = x_s.clone()

source_amplitudes = deepwave.wavelets.ricker(peak_freq, nt, dt,
                                             peak_source_time).reshape(-1, 1, 1)

prop = deepwave.scalar.Propagator({'vp': model}, dx)
receiver_amplitudes = prop(source_amplitudes, x_s, x_r, dt)
```

### Backpropagation/Inversion (FWI)
FWI attempts to match the predicted and true receiver amplitudes by modifying the model. This is similar to modifying the weights in a neural network during training. To achieve this with Deepwave, we first specify that we will need to calculate gradients with respect to our model.

```python
model.requires_grad = True
```

We also need to specify which optimization algorithm and loss/cost/objective function we wish to use.

```python
optimizer = torch.optim.Adam([model], lr=10)
criterion = torch.nn.MSELoss()
```

I have chosen to use the Adam optimizer with a learning rate of 10, and mean squared error (MSE) as the cost function. The MSE cost function is provided for us by PyTorch, but I could also have made-up my own cost function and used that instead (as long as PyTorch can automatically differentiate it).

With our propagator initialized as in the forward modeling section above, we can then iteratively optimize the model.

```python
for iter in range(num_iter):
    optimizer.zero_grad()
    receiver_amplitudes_pred = prop(source_amplitudes, x_s, x_r, dt)
    loss = criterion(receiver_amplitudes_pred, receiver_amplitudes_true)
    loss.backward()
    optimizer.step()
```

In each iteration, we zero the gradients, calculate the predicted receiver amplitudes from the current model, compare these with the true data using our cost function, backpropagate to calculate the gradient of the loss with respect to the model, and then update the model.

The flexibility of PyTorch means that it is easy to modify this to suit your own needs.

For a full example of using Deepwave for FWI, see [this notebook](https://colab.research.google.com/drive/1PMO1rFAaibRjwjhBuyH3dLQ1sW5wfec_).

It is also possible to perform source inversion with Deepwave. Simply set `requires_grad` to `True` on the source amplitude Tensor.

### Born forward modeling (for demigration and LSRTM)
To perform Born forward modeling, use `BornPropagator`. In addition to a background wave speed model, you will also need a scattering model ("image"). As an example, add the following line to the regular forward modeling example above:

```python
scatter = torch.zeros_like(model)
scatter[10, 20] = 100
```

This creates a scattering model with a point scatterer of amplitude 100 m/s. Then, replace the line that creates the propagator with one that uses the Born propagator:

```python
prop = deepwave.scalar.BornPropagator({'vp': model, 'scatter': scatter}, dx)
```

The resulting data should only contain recordings from the scattered wavefield, so there will not be any direct arrival.

To perform LSRTM, optimize the variable `scatter`. Optimizing the source amplitude or the wave speed model is not currently supported with the Born propagator.

For an example of using Deepwave for LSRTM, see [this notebook](https://colab.research.google.com/drive/1BgQM5VGgyFp7Q--pAJX-vGb2bW9mcbM8).

## Notes
* For a reflective free surface, set the PML width to zero at the surface. For example, in 3D and when the PML width on the other sides is 10 cells, provide the argument `pml_width=[0,10,10,10,10,10]` when creating the propagator if the free surface is at the beginning of the first dimension. The format is [z1, z2, y1, y2, x1, x2], where z1, y1, and x1 are the PML widths at the beginning of the z, y, and x dimensions, and z2, y2, and x2 are the PML widths at the ends of those dimensions.
* To limit wave simulation to the region covered by the survey (the sources and receivers), provide the `survey_pad` keyword argument when creating the propagator. For example, to use the whole z dimension, but only up to 100m from the first and last source/receiver in the y and x dimensions, use `survey_pad=[None, None, 100, 100, 100, 100]`, where `None` means continue to the edge of the model, and the format is similar to that used for `pml_width`. The default is `None` for every entry.
* [@LukasMosser](https://github.com/LukasMosser) discovered that GCC 4.9 or above is necessary ([#18](https://github.com/ar4/deepwave/issues/18)).
* Distributed parallelization over shots is supported, but not within a shot; each shot must run within one node.
* Currently, the forward source wavefield is saved in memory for use during backpropagation. This means that realistic 3D surveys will probably require more memory than is available. This will be fixed in a future release.

## Contributing
Your help to improve Deepwave would be greatly appreciated. If you encounter any difficulty using Deepwave, please report it as a Github Issue so that it can be fixed. If you have feature suggestions or other ideas to make Deepwave better, please also report those as Github Issues. If you want to help with the coding, that would be especially wonderful. The Github Issues contain a list of things that need work. If you see one that you would like to attempt, please ask for it to be assigned to you.

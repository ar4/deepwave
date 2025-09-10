PyTorch
=======

Deepwave leverages PyTorch to provide wave propagation as a differentiable operation. A solid understanding of PyTorch is key to maximising Deepwave's utility. While PyTorch offers extensive features and flexibility, and its `website <https://pytorch.org/tutorials>`_ provides comprehensive tutorials, this section offers a concise overview of its most crucial aspects.

PyTorch, much like the popular Python package `NumPy <https://numpy.org/>`_, offers tools for numerical computation in Python. Many NumPy functions have direct PyTorch equivalents. However, two key differences set PyTorch apart: Tensors and automatic differentiation (autograd).

**Tensors**

NumPy stores data in multi-dimensional arrays called ndarrays. PyTorch uses Tensors, which are similar but offer significant advantages. Unlike ndarrays, which are restricted to CPUs, Tensors can reside on either a CPU or a GPU (if available), enabling substantially better performance. You can create, manipulate, and move Tensors between devices:

.. code-block:: python

    import torch

    a = torch.arange(3)  # [0, 1, 2] on CPU
    b = torch.ones(3, device=torch.device('cuda'))  # [1, 1, 1] on GPU
    c = a.to(b.device)**2 + 2 * b  # [2, 3, 6] on GPU
    d = c.cpu()  # d = [2, 3, 6] on CPU, c is still on GPU

**Automatic Differentiation (Autograd)**

The second, and arguably more crucial, difference is PyTorch's autograd system. It allows you to automatically compute gradients through complex chains of operations, enabling inversion and optimisation tasks by simply defining the forward pass of your calculations.

Let's illustrate with a simple example:

.. code-block:: python

    import torch

    a = torch.arange(3.0, requires_grad=True)
    b = (2 * a).sum()
    print(b)
    # Expected output:
    # tensor(6., grad_fn=<SumBackward0>)
    b.backward()
    print(a.grad)
    # Expected output:
    # tensor([2., 2., 2.])

Here, we create a Tensor `a` with `requires_grad=True`, indicating that we need to compute gradients with respect to it. The operation `(2 * a).sum()` computes a scalar `b`. When `b.backward()` is called, PyTorch automatically computes the gradient of `b` with respect to `a` and stores it in `a.grad`. In this case, if we change any element of `a` by :math:`\delta`, `b` changes by :math:`2\delta`, so the gradient for each element of `a` is 2.

While this example is simple, the power of PyTorch's autograd becomes evident with more complex calculations. You only need to define the forward pass, and PyTorch handles the gradient computation.

Consider a slightly more complex chain of operations:

.. code-block:: python

    import torch

    a = torch.arange(3.0, requires_grad=True)
    b = (2 * a).sum()
    c = torch.ones(3, requires_grad=True)
    d = (a * c).sum() + b
    d.backward()
    print(a.grad)
    # Expected output:
    # tensor([5., 5., 5.])
    print(c.grad)
    # Expected output:
    # tensor([0., 1., 2.])

Even with a longer and more intricate operation chain, PyTorch effortlessly calculates the gradients. Notice that `d` depends on `a` both directly (through `a * c`) and indirectly (through `b`). Also, the reported gradient for `a` is `[5., 5., 5.]`, not `[3., 3., 3.]`. This is because PyTorch accumulates gradients. If `a.grad` was not explicitly cleared from a previous computation, the new gradients are added to the existing ones. This accumulation is particularly useful when processing large datasets in smaller batches, allowing you to accumulate gradients across batches before an optimiser step.

**Optimisation and Inversion**

With the ability to compute gradients, we can perform optimisation and inversion. PyTorch offers various optimisers, or you can implement your own. Here's an example using the Stochastic Gradient Descent (SGD) method:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim

    def f(x):
        return 3 * (torch.sin(5 * x) + 2 * torch.exp(x))

    x_true = torch.tensor([0.123, 0.321])
    y_true = f(x_true)

    x = torch.zeros(2, requires_grad=True)
    opt = optim.SGD([x], lr=1e-4)
    loss_fn = nn.MSELoss()

    for i in range(2000):
        opt.zero_grad()  # Clear gradients from previous iteration
        y = f(x)
        loss = loss_fn(y, y_true)
        loss.backward()  # Compute gradients
        opt.step()  # Update x based on gradients

    print(x)
    # Expected output (approximately):
    # tensor([0.1230, 0.3210], requires_grad=True)

In this example, `f` represents our forward model. We define `x_true` and its corresponding `y_true`. Starting with an initial guess for `x` (zeros), we iteratively: clear previous gradients, compute the model output `y`, calculate the Mean Squared Error (MSE) loss between `y` and `y_true`, backpropagate to get the gradient of the loss with respect to `x`, and then update `x` using the optimiser. After 2000 iterations, `x` converges to the correct solution. It's crucial to call `opt.zero_grad()` at the beginning of each iteration to prevent gradient accumulation from previous steps.

**Deepwave's Role**

Deepwave integrates seamlessly into this PyTorch ecosystem by providing wave propagation as a PyTorch operation. Just as you would use basic operations like multiplication or sine, Deepwave allows you to apply wave propagation and then backpropagate through it to compute gradients for inversion. Deepwave propagators accept various inputs as Tensors, such as wavespeeds, source amplitudes, and initial wavefields. By setting `requires_grad=True` on these input Tensors, you can calculate gradients with respect to them.

Furthermore, you can chain multiple operations together. For instance, you could use a neural network to transform a latent vector into a wavespeed model, feed this model into Deepwave's propagator, and then apply further operations before computing a loss function. PyTorch's autograd will then perform end-to-end backpropagation through this entire computational graph, calculating gradients for your loss function with respect to the latent vector and any other `requires_grad=True` variables. This flexibility significantly accelerates the process of testing new ideas. I hope that you find it useful.

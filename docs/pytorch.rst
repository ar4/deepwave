PyTorch
=======

Deepwave uses PyTorch to provide wave propagation as a differentiable
operation. Understanding PyTorch well makes Deepwave easier to use. Its
`website <https://pytorch.org/tutorials>`_ has full tutorials; this page
gives a short overview of the parts of PyTorch that matter most for Deepwave.

PyTorch, like the Python package `NumPy <https://numpy.org/>`_, provides
tools for numerical computation. Many NumPy functions have direct PyTorch
equivalents. Two things set PyTorch apart: Tensors, and automatic
differentiation (autograd).

**Tensors**

NumPy stores data in multi-dimensional arrays called ndarrays. PyTorch uses
Tensors, which are similar but can do more. An ndarray is restricted to the
CPU; a Tensor can reside on either a CPU or a GPU (if one is available),
which allows better performance. You can create, manipulate, and move
Tensors between devices:

.. code-block:: python

    import torch

    a = torch.arange(3)  # [0, 1, 2] on CPU
    b = torch.ones(3, device=torch.device('cuda'))  # [1, 1, 1] on GPU
    c = a.to(b.device)**2 + 2 * b  # [2, 3, 6] on GPU
    d = c.cpu()  # d = [2, 3, 6] on CPU, c is still on GPU

**Automatic Differentiation (Autograd)**

The second difference is PyTorch's autograd system. It computes gradients
through chains of operations automatically: you define the forward pass of
your calculation, and PyTorch works out how to backpropagate through it.

The following example illustrates this:

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

This creates a Tensor `a` with `requires_grad=True`, which tells PyTorch to
track gradients with respect to it. The operation `(2 * a).sum()` computes a
scalar `b`. When `b.backward()` is called, PyTorch computes the gradient of
`b` with respect to `a` and stores it in `a.grad`. Here, changing any element
of `a` by :math:`\delta` changes `b` by :math:`2\delta`, so the gradient for
each element of `a` is 2.

This example is simple, but the same mechanism scales to much longer chains
of operations: you still only need to define the forward pass, and PyTorch
computes the gradients.

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
    # tensor([3., 3., 3.])
    print(c.grad)
    # Expected output:
    # tensor([0., 1., 2.])

PyTorch computes the gradients for this longer, more intricate operation
chain in the same way. `d` depends on each element of `a` along two paths:
directly through `a * c`, which contributes `c_i` (1, for every element),
and indirectly through `b`, which contributes 2. Autograd sums the
contributions of every path, so `a.grad` is `[3., 3., 3.]`. `d`'s dependence
on `c` only involves the direct path, `a * c`, so `c.grad` is simply `a`
itself, `[0., 1., 2.]`.

Gradients accumulate across separate calls to `backward()`, though: if you
do not clear `a.grad` before calling `backward()` again, the new gradients
are added to whatever is already stored there.

.. code-block:: python

    import torch

    a = torch.arange(3.0, requires_grad=True)
    b = (2 * a).sum()
    b.backward()
    print(a.grad)
    # Expected output:
    # tensor([2., 2., 2.])
    b2 = (3 * a).sum()
    b2.backward()
    print(a.grad)
    # Expected output:
    # tensor([5., 5., 5.])
    a.grad = None  # equivalent to optimiser.zero_grad()
    print(a.grad)
    # Expected output:
    # None

The second `backward()` call added the gradient of `b2` with respect to `a`,
`[3., 3., 3.]`, to the `[2., 2., 2.]` already stored from the first call.
Setting `a.grad = None` (or calling `optimiser.zero_grad()`, as in the next
example) clears it. This accumulation is useful when processing large
datasets in smaller batches, since it lets you sum gradients across batches
before taking an optimiser step.

**Optimisation and Inversion**

The ability to compute gradients enables optimisation and inversion. PyTorch
provides several optimisers, and you can also implement your own. Here is an
example using the Stochastic Gradient Descent (SGD) method:

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

Here, `f` represents the forward model. `x_true` and its corresponding
`y_true` are the target values. Starting from an initial guess for `x`
(zeros), each iteration:

1. clears gradients from the previous iteration (``opt.zero_grad()``);
2. computes the model output `y`;
3. calculates the Mean Squared Error (MSE) loss between `y` and `y_true`;
4. backpropagates to get the gradient of the loss with respect to `x`
   (``loss.backward()``);
5. updates `x` using the optimiser (``opt.step()``).

After 2000 iterations, `x` converges to the correct solution. Calling
``opt.zero_grad()`` at the start of each iteration is required to prevent
gradient accumulation from previous steps.

**Deepwave's Role**

Deepwave fits into this PyTorch ecosystem by providing wave propagation as a
PyTorch operation. Just as you would use a basic operation like
multiplication or sine, Deepwave lets you apply wave propagation and then
backpropagate through it to compute gradients for inversion. Deepwave
propagators accept wavespeeds, source amplitudes, initial wavefields, and
similar inputs as Tensors. Setting `requires_grad=True` on these input
Tensors lets you calculate gradients with respect to them.

You can also chain multiple operations together. For example, a neural
network could transform a latent vector into a wavespeed model; that model
could be fed into Deepwave's propagator; further operations could then
compute a loss function from the propagator's output. PyTorch's autograd
performs end-to-end backpropagation through this entire computational graph,
calculating gradients for the loss function with respect to the latent
vector and any other `requires_grad=True` variables. This is what makes it
straightforward to test new modelling and inversion ideas.

A composition like that must use the *functional* interface
(`deepwave.scalar(v, ...)`), not the Module (`deepwave.Scalar(v, ...)`). A
Module wraps the model you give it in a `torch.nn.Parameter` that it owns,
and a Parameter is a leaf tensor: it cannot carry the autograd graph that
produced it, so wrapping a network-produced wavespeed in a Module severs the
graph and no gradient ever reaches the network. The functional interface
takes the model as an ordinary Tensor argument and backpropagates through it.

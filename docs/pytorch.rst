PyTorch
=======

Deepwave provides wave propagation as a differentiable operation in PyTorch. Having a good understanding of PyTorch will therefore help you to get the greatest benefit from Deepwave. PyTorch has a lot of features and is very flexible. The tutorials on the `PyTorch website <https://pytorch.org/tutorials>`_ are a good place to learn about these, but I include a quick overview of some of the most important features here.

PyTorch is similar to the popular Python package `NumPy <https://numpy.org/>`_, providing the tools to do numerical work in Python. Many NumPy functions have an equivalent PyTorch version. There are two important differences, however: Tensors and backpropagation.

NumPy stores data in a multi-dimensional array known as an ndarray. PyTorch calls them Tensors. Unlike ndarrays, which are restricted to CPUs, a Tensor can be on a CPU or a GPU (if you have an appropriate one), providing the opportunity for substantially better performance. You can create a Tensor, move it around, and apply operations to it, like this::
    
    a = torch.arange(3)  # [0, 1, 2] on CPU
    b = torch.ones(3, device=torch.device('cuda'))  # [1, 1, 1] on GPU
    c = a.cuda()**2 + 2 * b  # [2, 3, 6] on GPU
    d = c.cpu()  # d = [2, 3, 6] on CPU, c is still on GPU

The second change, backpropagation, is even more important. It allows you to backpropagate gradients through chains of operations, using automatic differentiation, enabling you to perform inversion/optimisation for complicated calculations after just coding the forward pass.

Let's demonstrate this by starting with a simple example:

>>> import torch
>>> a = torch.arange(3.0, requires_grad=True)
>>> b = (2 * a).sum()
>>> print(b)
tensor(6., grad_fn=<SumBackward0>)
>>> b.backward()
>>> print(a.grad)
tensor([2., 2., 2.])

Here we created a Tensor, `a`, containing `[0, 1, 2]`, and indicated that we will require gradients with respect to it (`requires_grad=True`). Multiplying each element by 2 and summing gives a Tensor containing the number 6, which we store in `b`. But `b` also contains a reference to the adjoint of the operation that generated it (`SumBackward0`), which is needed when we call `backward` on `b` to calculate gradients with respect to it. This backpropagates gradients through the computational graph until they reach all of the Tensors in the calculation that had `requires_grad` set to `True`. Calling `backward` on `b` thus backpropagated to calculate the gradient of `b` with respect to `a`. The result is stored in `a.grad`. We expect that if we change the value of any of the elements in `a` by :math:`\delta`, the effect on `b` will be :math:`2\delta` (since we obtained `b` by multiplying `a` by two and summing), so the gradient with respect to every element of `a` should be 2, which we see it is.

We were able to easily work out what that gradient was ourselves. The power of PyTorch becomes more evident when calculations become more complicated. In those cases it makes life much easier when we only have to write the "forward" part of the calculation, and leave working out the gradient to PyTorch's automatic differentiation.

>>> b = (2 * a).sum()
>>> c = torch.ones(3, requires_grad=True)
>>> d = (a * c).sum() + b
>>> d.backward()
>>> print(a.grad)
tensor([5., 5., 5.])
>>> print(c.grad)
tensor([0., 1., 2.])

Here we have made the chain of operations a bit longer and more complicated, and we now want to calculate the gradient of the final answer with respect to two input Tensors. PyTorch still has no problem calculating the gradients for us, though. Note that `d` depends on `a` both directly through the `a * c` operation, and also through `b`. Note also that the reported gradient with respect to `a` is `[5, 5, 5]` and not `[3, 3, 3]` as you might have expected. That is because PyTorch allows you to accumulate gradients. We didn't clear the gradient stored in `a` after the previous example, so the gradient with respect to `d` got added to it. This can be useful when you want to calculate gradients for many data samples but do not have the memory to process all of them simultaneously, so this enables you to process in smaller batches and accumulate the gradients. For Deepwave this happens when you want to calculate the gradient for many shots.

Now that we know how to calculate gradients, we can use these to perform optimisation/inversion. PyTorch provides several optimisers, or you can use the gradients to create your own. As an example, I will use the simple Stochastic Gradient Descent method.

>>> def f(x):
>>>     return 3 * (torch.sin(5 * x) + 2 * torch.exp(x))
>>> 
>>> x_true = torch.tensor([0.123, 0.321])
>>> y_true = f(x_true)
>>> 
>>> x = torch.zeros(2, requires_grad=True)
>>> opt = torch.optim.SGD([x], lr=1e-4)
>>> loss_fn = torch.nn.MSELoss()
>>> 
>>> for i in range(2000):
>>>     opt.zero_grad()
>>>     y = f(x)
>>>     loss = loss_fn(y, y_true)
>>>     loss.backward()
>>>     opt.step()
>>> 
>>> print(x)
tensor([0.1230, 0.3210], requires_grad=True)

This time the forward model is in a function `f`. We know what the true value of `x` should be, so we can calculate what the true output of `f` should be, which we store in `y_true`. We then start from an initial guess of `x` (zeros), calculate the output we get from `f` with that value of `x`, compare it to the desired output using a Mean Squared Error loss, backpropagate to get the gradient of that loss with respect to `x` and then update `x` by taking a step with our optimiser to reduce the loss. Running this for 2000 iterations is enough to converge on the correct solution. We zero the gradient stored in `x.grad` at the beginning of each iteration as it wouldn't make sense to accumulate it.

You can now hopefully see why Deepwave is useful. It provides wave propagation as a PyTorch operation. In the above examples we used operations like multiplications and sines. Deepwave enables us to apply wave propagation as an operation just as easily, and then backpropagate through it to calculate gradients so that we can use it in inversion. There are several inputs to the Deepwave propagators, such as wavespeeds, source amplitudes, and initial wavefields, which you provide as Tensors. You can set `requires_grad=True` on some or all of these, to calculate gradients with respect to them. Just as we did with other operations above, you can chain multiple operations together, such as using a neural network to convert a latent vector into a wavespeed model, which is then used as input into wave propagation, and the output of that goes through multiple other operations before being used to calculate a loss function. As above, PyTorch can then do end-to-end backpropagation through this whole computational graph to calculate the gradient of your loss function with respect to that latent vector and any other variables involved in the calculation that had `requires_grad=True`. This makes it easy to quickly try out new ideas, which I hope will help you to make great progress in wave equation inverse problems.

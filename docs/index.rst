Deepwave
========

Deepwave implements wave propagators as PyTorch modules. This allows you to perform forward modelling, and backpropagation to calculate the gradient of the outputs with respect to the inputs (and thus perform inversion/optimisation). It currently includes regular and Born modelling of the scalar wave equation, and runs on both CPUs and GPUs.

To install it, I recommend first installing PyTorch using the instructions on the `PyTorch website <https://pytorch.org>`_. Deepwave can then be installed using::

    pip install deepwave

The first time you import Deepwave may take some time as the C++ and (if you have a suitable GPU) CUDA components of Deepwave will then be compiled.

If you encounter any problems, it might be because compilation failed due to your compiler not being compatible with PyTorch. If you are able to install one that is compatible, you might need to set the `CXX` environment variable to the path to the new compiler before launching Python. If you are using Windows, PyTorch's `Windows FAQ <https://pytorch.org/docs/stable/notes/windows.html#cpp-extension>`_ might help. On MacOS you might need to first install OpenMP. If you are still stuck, please `file an issue <https://github.com/ar4/deepwave/issues>`_ or `send me an email <mailto:alan@ausargeo.com>`_.

As Deepwave is a module for PyTorch, learning how to use PyTorch will help you a lot with using Deepwave. The `PyTorch website <https://pytorch.org>`_ has many tutorials.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   usage

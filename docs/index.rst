Deepwave
========

Deepwave offers PyTorch-integrated wave propagators, enabling efficient forward modeling and backpropagation for gradient calculation in inversion and optimization tasks. It currently supports 2D standard and Born modeling of the scalar wave equation, alongside propagation with the elastic wave equation, optimized for both CPUs and GPUs.

Installation
------------

It is recommended to first install PyTorch by following the instructions on the `PyTorch website <https://pytorch.org>`_. Once PyTorch is installed, Deepwave can be installed using pip:

.. code-block:: bash

    pip install deepwave

This method should work for most users. However, if you have an unusual system configuration, you might need to build Deepwave from source. Instructions for building from source are provided below.

As Deepwave is built on PyTorch, a good understanding of PyTorch will significantly enhance your experience. The `PyTorch website <https://pytorch.org>`_ offers many helpful tutorials.

Building from source
--------------------

Deepwave uses `cibuildwheel` and `scikit-build-core` for its build process. When you run `pip install deepwave`, pip will first attempt to download a precompiled wheel if one is available for your system. If a suitable precompiled wheel is not found, pip will automatically download the source distribution and attempt to compile it on your system. You will need to have a C compiler and CMake available, and a CUDA compiler if you wish to have CUDA support.

If you wish to force building from source rather than using a wheel (such as because your system does not have AVX2 support, which the wheels assume is available), you can use the following command:

.. code-block:: bash

    pip install deepwave --no-binary deepwave

This command forces pip to download and compile the source distribution on your system, bypassing any precompiled wheels.

Support
-------

If you encounter any issues (e.g., typos, broken links, unexpected behavior), or have suggestions for improvement, please `file an issue on GitHub <https://github.com/ar4/deepwave/issues>`_ or `send me an email <mailto:alan@ausargeo.com>`_. Your feedback is highly appreciated.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pytorch
   what_deepwave_calculates
   examples
   usage

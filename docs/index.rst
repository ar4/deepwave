Deepwave
========

Deepwave implements wave propagators as PyTorch modules. This allows you to perform forward modelling, and backpropagation to calculate the gradient of the outputs with respect to the inputs (and thus perform inversion/optimisation). It currently includes 2D regular and Born modelling of the scalar wave equation, propagation with the elastic wave equation, and runs on both CPUs and GPUs.

To install it, I recommend first installing PyTorch using the instructions on the `PyTorch website <https://pytorch.org>`_. Deepwave can then be installed using::

    pip install deepwave

This should work for most people, but if you have an unusual system then you might need to instead build it from the source. Instructions for doing that are below.

As Deepwave is a module for PyTorch, learning how to use PyTorch will help you a lot with using Deepwave. The `PyTorch website <https://pytorch.org>`_ has many tutorials.

If you find any mistakes (even a typo or broken link), anything that doesn't behave the way you expected it to, something that you think could be clearer, have a feature suggestion, or just want to say hello, please `write to me <mailto:alan@ausargeo.com>`_.

Building from source
^^^^^^^^^^^^^^^^^^^^

Deepwave uses `cibuildwheel` and `scikit-build-core` to manage its build process. This means that when you run `pip install deepwave`, `pip` will first attempt to download a precompiled wheel if one is available for your system. If a suitable precompiled wheel is not found, `pip` will automatically download the source distribution and attempt to compile it on your system.

The precompiled wheels still make certain assumptions, such as on the `glibc` version and AVX2 support. If the installed package fails to run (possibly with an error about an illegal operation or instruction) due to your system not conforming to these assumptions, you should install from the source distribution instead. To do this, you can run:

.. code-block:: bash

    pip install --no-binary :all: deepwave

This command forces `pip` to download the source distribution and compile it on your system, bypassing any precompiled wheels. You may need to ensure your build environment is correctly set up for `scikit-build-core` to compile from source.

(Assumptions for the precompiled version include: Linux (glibc >= 2.17), recent MacOS or Windows, and an x86-64 processor with AVX2 support (Linux and Windows) or x86-64/ARM64 (MacOS).)

If you encounter any issues during installation or compilation, please `file an issue <https://github.com/ar4/deepwave/issues>`_ or `send me an email <mailto:alan@ausargeo.com>`_. I am happy to assist.

Other issues
^^^^^^^^^^^^

If you encounter any problems, please `file an issue <https://github.com/ar4/deepwave/issues>`_ or `send me an email <mailto:alan@ausargeo.com>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pytorch
   what_deepwave_calculates
   examples
   usage

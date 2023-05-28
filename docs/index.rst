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

I have tried to make it so that the precompiled version of Deepwave (what you get if you just run `pip install deepwave`) will work for common setups, but it does make some assumptions and so the installed package may fail to run (possibly with an error about an illegal operation or instruction) if your system does not conform to those assumptions. In that case you will need to compile the code yourself. First, `download the code (.zip or .tar.gz) of the latest release tag <https://github.com/ar4/deepwave/tags>`_ and extract it. In the extracted result, descend to the `src/deepwave` directory. Here, you will find `build_[linux, macos, windows].sh` text files that contain the compiler commands that are used to build the precompiled releases. Use the one that is closest to your setup, after making any necessary modifications, to compile the code for your system. You should then return to the base extracted directory and run `pip install .` to install the package. I am very happy to help, so please `let me know <mailto:alan@ausargeo.com>`_ if you need assistance with any of this. Please also tell me if you think your system should be included in those targeted by the precompiled releases.

(The assumptions made by the precompiled version include that you are running either Linux (with a glibc version of at least 2.17), or a recent version of MacOS or Windows, and using an x86-64 processor that supports AVX2 (Linux and Windows) or an x86-64 or ARM64 (MacOS).)

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

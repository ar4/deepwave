Deepwave
========

Deepwave implements wave propagators as PyTorch modules. This allows you to perform forward modelling, and backpropagation to calculate the gradient of the outputs with respect to the inputs (and thus perform inversion/optimisation). It currently includes regular and Born modelling of the scalar wave equation, propagation with the elastic wave equation, and runs on both CPUs and GPUs.

To install it, I recommend first installing PyTorch using the instructions on the `PyTorch website <https://pytorch.org>`_. Deepwave can then be installed using::

    pip install deepwave

The first time you import Deepwave may take some time (several minutes) as the C++ and (if you have a suitable GPU) CUDA components of Deepwave will then be compiled.

As Deepwave is a module for PyTorch, learning how to use PyTorch will help you a lot with using Deepwave. The `PyTorch website <https://pytorch.org>`_ has many tutorials.

If you find any mistakes, anything that doesn't behave the way you expected it to, something that you think could be clearer, have a feature suggestion, or just want to say hello, please `write to me <mailto:alan@ausargeo.com>`_.

Troubleshooting
---------------

Compiling exhausts memory
^^^^^^^^^^^^^^^^^^^^^^^^^

Compiling Deepwave, which happens the first time you import it, requires a lot of RAM. If you have less than about 16GB RAM, you will probably need to reduce the number of parallel compilation jobs to avoid running out of memory during this compile. You can do this using `MAX_JOBS=2 python -c "import deepwave"` after installing (limiting compilation to two parallel jobs and so reducing the amount of RAM required).

Import hangs
^^^^^^^^^^^^

When compiling an extension like Deepwave, PyTorch creates a lock file. This lock file gets removed when compiling completes. If compiling failed (such as because you ran out of memory), then the lock file might not be removed and so when you try to import Deepwave it will hang idly while it waits for the lock file to be removed. In this case you will need to delete the lock file manually. You can find where this lock file is located by running `python -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension._get_build_directory('deepwave', False))"`. Delete that directory and then try importing Deepwave again (with the `MAX_JOBS` environment variable, discussed above, if necessary).

`pip` unable to access the internet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The only dependencies of Deepwave are PyTorch and Ninja. If you are unable to use `pip` to install from the internet (perhaps because you are behind a firewall), you can manually install these two dependencies, `download the Deepwave code <https://github.com/ar4/deepwave/tags>`_, extract it, and run `pip install .` in the resulting directory. If you are in this situation and you are also unable to install Ninja, please `contact me <mailto:alan@ausargeo.com>`_ for instructions.

Undefined symbol errors
^^^^^^^^^^^^^^^^^^^^^^^

If you update to a new version of PyTorch, you will probably need to reinstall Deepwave (`pip install --force-reinstall deepwave`, and you may also want to include `--upgrade` to check for updates) or you will get errors such as complaints about undefined symbols.

Other issues
^^^^^^^^^^^^

If you encounter any problems, it might be because compilation failed due to your compiler not being compatible with PyTorch. If you are able to install one that is compatible, you might need to set the `CXX` environment variable to the path to the new compiler before launching Python. If you are using Windows, PyTorch's `Windows FAQ <https://pytorch.org/docs/stable/notes/windows.html#cpp-extension>`_ might help. On MacOS you might need to first install OpenMP. If you are still stuck, please `file an issue <https://github.com/ar4/deepwave/issues>`_ or `send me an email <mailto:alan@ausargeo.com>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pytorch
   examples
   usage

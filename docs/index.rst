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

Compiler error "cannot find -lcudart"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch is hardcoded to expect the CUDA runtime library to be in the "lib64" directory, but sometimes it is instead in the "lib" directory, cuasing the compilation (during the first import) to fail because it cannot find the library. A solution is to create a soft link from the expected path to the actual path so that the library can be found. First you need to find where it is looking. In the compiler output (usually just before the error) you will see something like `-L/home/alanr/anaconda3/lib64`. That is where the compiler is being told to search the lib64 directory for libraries. Look in the "lib" directory (in the example case, this would be `/home/alanr/anaconda3/lib`) to make sure the `libcudart.so` file that the compiler wants is there. If it is, you can add a soft link. In the example case, this would be done with `ln -s /home/alanr/anaconda3/lib/libcudart.so /home/alanr/anaconda3/lib64/libcudart.so`.

Windows
^^^^^^^

I successfully installed Deepwave on Windows using the following steps (but I am not very familiar with Windows, so there might be a better way). I installed `Visual Studio <https://visualstudio.microsoft.com/>`_ with the "Desktop Development with C++" workload to install the MSVC C++ compiler. I also installed the `Anaconda <https://www.anaconda.com/>`_ Python distribution. I opened the Anaconda command prompt and ran `echo %PATH%` and copied the resulting list of paths. I closed this command prompt and opened the "x86_64 Cross Tools Command Prompt". I then ran `set PATH=<paste>;%PATH%`, where in `<paste>` I pasted the list of paths copied from the Anaconda environment. I then ran `set LIB=%LIB%;<path to Python library>`. On my computer the `<path to Python library>` was `C:\ProgramData\Anaconda3` (the same location as the `python` executable), but it might be something different on yours. Finally I ran `pip install deepwave`, `set MAX_JOBS=1` (to limit the number of parallel compiler jobs to one to avoid running out of memory during the compile stage, but if you have more memory than I do then you might not need to do this), and `python -c "import deepwave"` to do an initial import of Deepwave that causes it to be compiled.

MacOS
^^^^^

The main problem on MacOS is that the default compiler does not support OpenMP. One user who successfully installed Deepwave on MacOS posted the steps they took `here <https://github.com/ar4/deepwave/issues/57#issuecomment-1531786477>`_.

Other issues
^^^^^^^^^^^^

If you encounter any problems, it might be because compilation failed due to your compiler not being compatible with PyTorch. If you are able to install one that is compatible, you might need to set the `CXX` environment variable to the path to the new compiler before launching Python. If you are still stuck, please `file an issue <https://github.com/ar4/deepwave/issues>`_ or `send me an email <mailto:alan@ausargeo.com>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pytorch
   examples
   usage

# Deepwave Specification

This document provides a technical specification of the Deepwave project. It is intended for developers who wish to understand the internal workings of the package, its design patterns, and how to contribute.

Deepwave provides a PyTorch-compatible implementation of wave propagation. It has a wide range of users, many of whom are graduate students and researchers who use it to test new ideas for applications such as seismic inversion and non-destructive testing. As it will often be used in unexpected ways to test unusual ideas, it is important that it has a logical structure, carefully implements the mathematically correct behaviour even in non-physical situations, and is as stable and robust as possible. Many of the users are not experienced programmers, and many will use it from a Notebook environment for quickly testing ideas. This means that ease-of-use, comprehensive input checking to quickly detect mistakes, and helpful error messages, are important.

Everyone who looks at the code should be impressed by how clean, clear, and Pythonic it is. It should be an exemplar of best practices. It is, however, a high performance code implementing computational intensive operations that will be limited by available computational power and memory. Complicated C and CUDA code implementing the inner loops of these calculations might therefore need to be tolerated to ensure maximum performance.

## 1. High-Level Architecture

Deepwave uses a hybrid architecture to balance high performance with a user-friendly Python API. The core wave propagation logic is implemented in C (for CPUs) and CUDA (for GPUs). A Python layer provides the user interface and integrates with PyTorch for automatic differentiation.

The key components are:

1.  **Python Layer**: Provides the public API (`deepwave.scalar`, `deepwave.elastic`, etc.). It handles input validation, setup, and wrapping the core logic in `torch.autograd.Function` to enable backpropagation.
2.  **Ctypes Interface**: A thin layer in `src/deepwave/__init__.py` that uses Python's `ctypes` library to load the compiled C/CUDA shared library and define the function signatures. This avoids a direct dependency on the Python C API, making the build process simpler and more robust against Python version changes.
3.  **C/CUDA Layer**: Contains the high-performance implementations of the wave propagators. This code is compiled into a single shared library (`libdeepwave_C.so` on Linux) that the Python layer calls into.

## 2. Code Organization

-   `docs/`: Project documentation.
-   `src/deepwave/`
    -   `__init__.py`: The crucial link between the Python and C/CUDA layers. It loads the compiled shared library and defines the `ctypes` interfaces for all the C functions.
    -   `common.py`: Contains shared Python helper functions for all propagators, including input validation, survey extraction, PML setup, and CFL condition calculations.
    -   `scalar.py`, `elastic.py`, etc.: The public-facing Python modules for each propagator. They define the `torch.nn.Module` and functional interfaces.
    -   `*.c`, `*.cu`: The low-level C and CUDA implementations of the propagators.
    -   `*.h`: C header files, primarily for defining shared macros and function prototypes.
-   `tests/`: Unit and integration tests.

## 3. The Build System

The C/CUDA code is compiled into a shared library using `scikit-build-core` and CMake. This process is designed for performance and portability.

-   **CMake Configuration**: The `CMakeLists.txt` file orchestrates the compilation. It defines how C and CUDA source files are compiled and linked.
-   **Compile-Time Permutations**: To avoid runtime conditionals in the performance-critical loops, the C/CUDA source files are compiled multiple times, each with a different set of preprocessor macros. These macros, such as `DW_ACCURACY`, `DW_DTYPE`, and `DW_DEVICE`, are passed by CMake to the compiler, creating specialized versions of each function for different numerical accuracies (e.g., 2, 4, 6, 8), data types (`float`, `double`), and devices (CPU, CUDA).
-   **Function Naming**: CMake controls the function names in the compiled library, creating a unique name for each permutation. For example, the forward scalar propagator for 4th-order accuracy and `float` data type on the CPU is named `scalar_iso_4_float_forward_cpu`. For CUDA, the functions are suffixed with `_cuda` (e.g., `scalar_iso_4_float_forward_cuda`).
-   **Output**: The build system produces a single shared library (e.g., `libdeepwave_C.so` on Linux) containing all the compiled function permutations.

## 4. The Python/C Interface (`__init__.py`)

This file is the heart of the interface layer. Its primary responsibilities are:

1.  **Loading the Library**: It locates and loads the `libdeepwave.so` shared library using `ctypes.CDLL`.
2.  **Defining Function Signatures**: For each function in the C/CUDA library, it defines the argument types (`argtypes`) and return type (`restype`). This is critical for passing data correctly and safely between Python and C. For example, it maps PyTorch Tensors to `ctypes` pointers and Python floats to `c_float`.
3.  **Function Dispatch**: It contains a dictionary or similar structure that maps a set of options (propagator type, accuracy, dtype, device) to the corresponding compiled function. The high-level Python code uses this to select and call the correct C/CUDA function at runtime.

## 5. Anatomy of a Propagator

Each propagator (e.g., scalar) consists of several connected parts:

1.  **Python Module (`scalar.py`)**: 
    -   Provides the user-friendly `Scalar` class and `scalar` function.
    -   It calls `common.setup_propagator` to prepare all necessary inputs.
    -   It calls the `apply` method of a custom `torch.autograd.Function` (e.g., `ScalarForwardFunc.apply`).

2.  **Autograd Function (`ScalarForwardFunc` in `scalar.py`)**:
    -   The `forward` method is the direct bridge to the compiled code. It retrieves the correct function pointer from the dispatcher in `__init__.py` and calls it, passing pointers to the data of the input Tensors.
    -   It saves the necessary Tensors for the backward pass using `ctx.save_for_backward`.
    -   The `backward` method is responsible for computing the gradient. It calls the corresponding `backward` function from the C/CUDA library.

3.  **C/CUDA Implementation (`scalar.c`, `scalar.cu`)**:
    -   Contains the main entry-point functions (e.g., `scalar_iso_4_float_forward_cpu`) that are called from Python.
    -   These functions typically contain the main time-stepping loop.
    -   Inside the loop, they call a `_kernel` function (e.g., `forward_kernel`) that computes one time step of the wave equation.
    -   The kernels are heavily optimized and use macros to handle different boundary conditions (PML regions) without `if` statements in the inner loops.
    -   The C code uses OpenMP for multi-threading across shots on the CPU.
    -   The CUDA code uses a 3D grid of thread blocks, where the dimensions correspond to the spatial dimensions and the shot index, enabling parallel processing on the GPU.

## 6. Key Concepts & Conventions

-   **Staggered Grids**: The elastic propagator uses a staggered grid where velocity and stress components are located at different points in the grid cell. This improves the accuracy of the finite-difference approximations. The grid layout is documented in `docs/elastic.rst` and in the comments of `src/deepwave/elastic.c`.
-   **PML Implementation**: The PML absorbing boundaries are implemented using a Convolutional PML (C-PML) scheme, which requires auxiliary wavefields (`psi` and `zeta`). To optimize performance, the PML calculations are only performed in the outer regions of the grid. The auxiliary variables are explicitly zeroed outside the PML region to ensure correctness.
-   **Internal Time-Stepping (CFL)**: The propagators automatically choose an internal time step `dt` that is smaller than the user's `dt` to satisfy the CFL stability condition. Source wavelets are upsampled to this finer `dt` and receiver data is downsampled back to the user's `dt`. This process is handled transparently by `setup_propagator` and `downsample_and_movedim` in `common.py`.
-   **Backpropagation**: The gradient with respect to the model parameters is calculated by cross-correlating the forward-propagating wavefield with the backward-propagating adjoint wavefield. To reduce memory usage, the forward wavefield is not stored at every time step. Instead, only the necessary components for the gradient calculation are stored, and only at a sampling interval (`model_gradient_sampling_interval`) that respects the Nyquist frequency of the source, not the (potentially much higher) frequency of the internal time stepping.

## 7. Developer's Guide: Adding a New Propagator

To add a new propagator (e.g., `new_prop`), follow these steps:

1.  **Create the Python Module (`src/deepwave/new_prop.py`)**:
    -   Create the main user-facing function and/or `torch.nn.Module` class.
    -   Implement a `torch.autograd.Function` for the forward and backward passes. This function will call the compiled C/CUDA code.

2.  **Implement the C/CUDA Kernels (`src/deepwave/new_prop.c`, `src/deepwave/new_prop.cu`)**:
    -   Write the core numerical logic for the forward and backward propagation. Follow the existing pattern of a main function containing the time loop, which in turn calls a kernel function for the core calculations.
    -   Use the `DW_ACCURACY` and `DW_DTYPE` macros to support compile-time permutations.

3.  **Update the Build Scripts (`CMakeLists.txt`)**:
    -   Add your new `.c` and `.cu` files to the list of source files to be compiled.

4.  **Update the `ctypes` Interface (`src/deepwave/__init__.py`)**:
    -   In the section that loads functions, add logic to load all permutations of your new propagator's functions (e.g., `new_prop_iso_2_float_forward_cpu`, etc.).
    -   For each loaded function, define its `argtypes` and `restype` to ensure correct data marshalling.
    -   Add the new propagator to the dispatch mechanism so the Python layer can find and call your compiled functions.

5.  **Integrate with `common.py`**:
    -   Add a new `prop_type` (e.g., `'new_prop'`) to the `if/elif` block in `setup_propagator` to handle any specific setup requirements for your propagator (e.g., number of models, padding modes).

6.  **Add Tests and Documentation**: Create new test files in the `tests/` directory to validate the correctness of your propagator (including checking gradients). Add a new documentation page in `docs/` explaining the physics and usage.

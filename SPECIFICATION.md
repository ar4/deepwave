# Deepwave Specification

Deepwave provides a PyTorch-compatible implementation of wave propagation. It is widely used by graduate students and researchers for testing new ideas in applications like seismic inversion and non-destructive testing. As it is often used in unexpected ways to test unusual ideas, it must have a logical structure, carefully implement mathematically correct behavior even in non-physical situations, and be as stable and robust as possible. Many users are not experienced programmers and will use it from a Notebook environment for rapid prototyping. Therefore, ease-of-use, comprehensive input validation to quickly detect mistakes, and helpful error messages are crucial.

Everyone who looks at the code should be impressed by how clean, clear, and Pythonic it is. It should be an exemplar of best practices. However, as a high-performance code implementing computationally intensive operations, it will be limited by available computational power and memory. Therefore, complex C and CUDA code implementing the inner loops of these calculations may be necessary to ensure maximum performance.

All code must be comprehensively tested; thus, the test suite is a crucial component of the package. Like the main code, it should be clear, well-structured, and follow best practices. It should also be extremely thorough, verifying code correctness (including extensive physics-based testing), checking behavior for edge cases, and ensuring invalid arguments are caught with helpful error messages. Internal components should also be unit tested so that developers can more confidently make changes.

To help users and developers avoid mistakes, the code should use type hinting. Since many users will call the code from a Notebook environment, type hints alone may not suffice to prevent incorrect inputs. Therefore, exhaustive input validation with helpful error messages is also necessary to quickly catch mistakes. As a scientific code used by researchers to test ideas, it should not impose unnecessary input restrictions. For example, if a parameter requires a list of floating-point numbers, users should be able to pass a Python `list` or `tuple`, a PyTorch `Tensor`, a NumPy `ndarray` with the appropriate number of elements, or any other type that can reasonably and unambiguously provide the necessary information. Even though floating-point numbers are required, passing integers is acceptable as they can be reasonably and unambiguously converted. However, complex numbers should generally not be accepted for such inputs, as their conversion is not straightforward and may indicate a user error. In some cases, a warning may be more appropriate than an error, such as when an input is valid but unusual, potentially not what the user intended. The goal is to help users quickly identify and correct mistakes.

Consistent style across all files is important. Deepwave uses the Google style guides, and they should be rigorously enforced.

## 1. High-Level Architecture

Deepwave uses a hybrid architecture to balance high performance with a user-friendly Python API. The core wave propagation logic is implemented in C (for CPUs) and CUDA (for GPUs). A Python layer provides the user interface and integrates with PyTorch for automatic differentiation.

The key components are:

1.  **Python Layer**: Provides the public API (`deepwave.scalar`, `deepwave.elastic`, etc.). It handles input validation, setup, and wrapping the core logic in `torch.autograd.Function` to enable backpropagation.
2.  **Ctypes Interface**: A thin layer in `src/deepwave/backend_utils.py` that uses Python's `ctypes` library to load the compiled C/CUDA shared library and define the function signatures. This avoids a direct dependency on the Python C API, making the build process simpler and more robust against Python version changes.
3.  **C/CUDA Layer**: Contains the high-performance implementations of the wave propagators. This code is compiled into a single shared library (`libdeepwave_C.so` on Linux) that the Python layer calls into.

## 2. Code Organization

-   `docs/`: Project documentation.
-   `src/deepwave/`
    -   `backend_utils.py`: The crucial link between the Python and C/CUDA layers, loading the compiled shared library and defining `ctypes` interfaces for all C functions.
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

## 4. The Python/C Interface (`backend_utils.py`, imported by `__init__.py`)

This file is the heart of the interface layer. Its primary responsibilities are:

1.  **Loading the Library**: It locates and loads the `libdeepwave_C.so` shared library using `ctypes.CDLL`.
2.  **Defining Function Signatures**: For each function in the C/CUDA library, it defines the argument types (`argtypes`) and return type (`restype`). This is critical for passing data correctly and safely between Python and C. For example, it maps PyTorch Tensors to `ctypes` pointers and Python floats to `c_float`.
3.  **Function Dispatch**: It contains a dictionary or similar structure that maps a set of options (propagator type, accuracy, dtype, device) to the corresponding compiled function. The high-level Python code uses this to select and call the correct C/CUDA function at runtime.

## 5. Anatomy of a Propagator

Each propagator (e.g., scalar) consists of several connected parts:

1.  **Python Module (`scalar.py`)**: 
    -   Provides the user-friendly `Scalar` class and `scalar` function.
    -   It calls `common.setup_propagator` to prepare all necessary inputs.
    -   It calls the `apply` method of a custom `torch.autograd.Function` (e.g., `ScalarForwardFunc.apply`).

2.  **Autograd Function (`ScalarForwardFunc` in `scalar.py`)**:
    -   The `forward` method is the direct bridge to the compiled code. It retrieves the correct function pointer from the dispatcher in `backend_utils.py` and calls it, passing pointers to the data of the input Tensors.
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

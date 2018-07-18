import os
import setuptools
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)
from torch import cuda

with open("README.md", "r") as fh:
    long_description = fh.read()

scalar_dir = os.path.join('deepwave', 'scalar')
scalar_cpp_file = os.path.join(scalar_dir, 'scalar.cpp')
scalar_cpu_file = os.path.join(scalar_dir, 'scalar_cpu.cpp')
scalar_gpu_file = os.path.join(scalar_dir, 'scalar_gpu.cu')
scalar_wrapper_file = os.path.join(scalar_dir, 'scalar_wrapper.cpp')

if cuda.is_available():
    cuda_extensions = [
        CUDAExtension('scalar1d_gpu_iso_4',
                      [scalar_gpu_file, scalar_cpp_file, scalar_wrapper_file],
                      define_macros=[('DIM', '1')],
                      include_dirs=[scalar_dir],
                      extra_compile_args={'nvcc': ['--restrict', '-O3',
                                                   '--use_fast_math'],
                                          'cxx': ['-Ofast', '-march=native']}),
        CUDAExtension('scalar2d_gpu_iso_4',
                      [scalar_gpu_file, scalar_cpp_file, scalar_wrapper_file],
                      define_macros=[('DIM', '2')],
                      include_dirs=[scalar_dir],
                      extra_compile_args={'nvcc': ['--restrict', '-O3',
                                                   '--use_fast_math'],
                                          'cxx': ['-Ofast', '-march=native']}),
        CUDAExtension('scalar3d_gpu_iso_4',
                      [scalar_gpu_file, scalar_cpp_file, scalar_wrapper_file],
                      define_macros=[('DIM', '3')],
                      include_dirs=[scalar_dir],
                      extra_compile_args={'nvcc': ['--restrict', '-O3',
                                                   '--use_fast_math'],
                                          'cxx': ['-Ofast', '-march=native']})]
else:
    cuda_extensions = []

setuptools.setup(
    name="deepwave",
    version="0.0.1",
    author="Alan Richardson",
    author_email="alan@ausargeo.com",
    description="Wave propagation modules for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ar4/deepwave",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=["numpy",
                      "torch>=0.4.0"],
    setup_requires=["torch>=0.4.0"],
    extras_require={"testing": ["pytest",
                                "scipy"]},
    ext_modules=[
        CppExtension('scalar1d_cpu_iso_4',
                     [scalar_cpu_file, scalar_cpp_file, scalar_wrapper_file],
                     define_macros=[('DIM', '1')],
                     include_dirs=[scalar_dir],
                     extra_compile_args=['-Ofast', '-march=native',
                                         '-fopenmp'],
                     extra_link_args=['-fopenmp']),
        CppExtension('scalar2d_cpu_iso_4',
                     [scalar_cpu_file, scalar_cpp_file, scalar_wrapper_file],
                     define_macros=[('DIM', '2')],
                     include_dirs=[scalar_dir],
                     extra_compile_args=['-Ofast', '-march=native',
                                         '-fopenmp'],
                     extra_link_args=['-fopenmp']),
        CppExtension('scalar3d_cpu_iso_4',
                     [scalar_cpu_file, scalar_cpp_file, scalar_wrapper_file],
                     define_macros=[('DIM', '3')],
                     include_dirs=[scalar_dir],
                     extra_compile_args=['-Ofast', '-march=native',
                                         '-fopenmp'],
                     extra_link_args=['-fopenmp']),
    ] + cuda_extensions,
    cmdclass={
        'build_ext': BuildExtension}
)

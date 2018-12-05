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


def _make_cpp_extension(dim, dtype):
    return CppExtension('scalar{}d_cpu_iso_4_{}'.format(dim, dtype),
                        [scalar_cpu_file, scalar_cpp_file, scalar_wrapper_file],
                        define_macros=[('DIM', dim), ('TYPE', dtype)],
                        include_dirs=[scalar_dir],
                        extra_compile_args=['-Ofast', '-march=native',
                                            '-fopenmp'],
                        extra_link_args=['-fopenmp'])


def _make_cuda_extension(dim, dtype):
    return CUDAExtension('scalar{}d_gpu_iso_4_{}'.format(dim, dtype),
                         [scalar_gpu_file, scalar_cpp_file, scalar_wrapper_file],
                         define_macros=[('DIM', dim), ('TYPE', dtype)],
                         include_dirs=[scalar_dir],
                         extra_compile_args={'nvcc': ['--restrict', '-O3',
                                                      '--use_fast_math'],
                                             'cxx': ['-Ofast', '-march=native']})


cpp_extensions = [_make_cpp_extension(dim, dtype)
                  for dim in ['1', '2', '3']
                  for dtype in ['float', 'double']]

if cuda.is_available():
    cuda_extensions = [_make_cuda_extension(dim, dtype)
                       for dim in ['1', '2', '3']
                       for dtype in ['float']]
else:
    cuda_extensions = []

setuptools.setup(
    name="deepwave",
    version="0.0.5",
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
                      "scipy",
                      "torch>=0.4.1"],
    setup_requires=["torch>=0.4.1"],
    extras_require={"testing": ["pytest"]},
    ext_modules=cpp_extensions + cuda_extensions,
    cmdclass={'build_ext': BuildExtension}
)

import os
import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

scalar_dir = os.path.join('deepwave', 'scalar')
scalar_cpp_file = os.path.join(scalar_dir, 'scalar.cpp')
scalar_wrapper_file = os.path.join(scalar_dir, 'scalar_wrapper.cpp')

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
                     [scalar_cpp_file, scalar_wrapper_file],
                     define_macros=[('DIM', '1')],
                     include_dirs=[scalar_dir],
                     extra_compile_args=['-Ofast', '-march=native',
                                         '-fopenmp'],
                     extra_link_args=['-fopenmp']),
        CppExtension('scalar2d_cpu_iso_4',
                     [scalar_cpp_file, scalar_wrapper_file],
                     define_macros=[('DIM', '2')],
                     include_dirs=[scalar_dir],
                     extra_compile_args=['-Ofast', '-march=native',
                                         '-fopenmp'],
                     extra_link_args=['-fopenmp']),
        CppExtension('scalar3d_cpu_iso_4',
                     [scalar_cpp_file, scalar_wrapper_file],
                     define_macros=[('DIM', '3')],
                     include_dirs=[scalar_dir],
                     extra_compile_args=['-Ofast', '-march=native',
                                         '-fopenmp'],
                     extra_link_args=['-fopenmp']),
        ],
    cmdclass={
        'build_ext': BuildExtension}
                     
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    install_requires=["cffi>=1.0.0",
                      "numpy",
                      "torch>=0.4.0"],
    setup_requires=["cffi>=1.0.0"],
    extras_require={"testing": ["pytest",
                                "scipy"]},
    cffi_modules=["build.py:scalar1d",
                  "build.py:scalar2d",
                  "build.py:scalar3d"],
)

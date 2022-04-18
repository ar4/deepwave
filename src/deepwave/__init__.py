"""Wave propagation modules for PyTorch.

Provides forward modelling and Born forward modelling of the scalar wave
equation, and backpropagation for both. It can thus be used to generate
synthetic data, and to perform FWI and LSRTM.

Module and functional interfaces are provided, and allow the propagators
to be easily integrated into a larger computational graph, if desired.

For better computational performance, the code is written in C++ and
CUDA. It is compiled when Deepwave is loaded, if necessary.
"""

import pathlib
import torch.utils.cpp_extension
import torch.cuda
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born
import deepwave.wavelets
source_dir = pathlib.Path(__file__).parent.resolve()
sources = [source_dir / 'scalar.cpp',
           source_dir / 'scalar_born.cpp']
if torch.cuda.is_available():
    sources += [source_dir / 'scalar.cu',
                source_dir / 'scalar_born.cu']
torch.utils.cpp_extension.load(
    name="deepwave",
    sources=sources,
    is_python_module=False,
    extra_cflags=['-march=native', '-Ofast', '-fopenmp'],
    extra_cuda_cflags=['--restrict', '-O3', '--use_fast_math'],
    extra_ldflags=['-march=native', '-Ofast', '-fopenmp']
)

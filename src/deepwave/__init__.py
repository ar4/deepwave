"""Wave propagation modules for PyTorch.

Provides forward modelling and backpropagation of the scalar wave
equation (regular and Born), and elastic wave equation.
It can thus be used to generate synthetic data, and to perform
FWI and LSRTM.

Module and functional interfaces are provided, and allow the propagators
to be easily integrated into a larger computational graph, if desired.

For better computational performance, the code is written in C++ and
CUDA.
"""

__all__ = [
    "Scalar",
    "scalar",
    "ScalarBorn",
    "scalar_born",
    "Elastic",
    "elastic",
    "wavelets",
    "location_interpolation",
    "_version",
    "backend_utils",
]

# These imports are for exposing the public API
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born
from deepwave.elastic import Elastic, elastic
from . import wavelets
from . import location_interpolation
from . import _version

# Import backend utilities to ensure DLL is loaded and functions are assigned
from . import backend_utils

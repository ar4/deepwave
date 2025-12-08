"""Wave propagation modules for PyTorch.

Provides forward modelling and backpropagation of the scalar wave
equation (regular and Born), variable-density acoustic wave equation,
and elastic wave equation.
It can thus be used to generate synthetic data, and to perform
FWI and LSRTM.

Module and functional interfaces are provided, and allow the propagators
to be easily integrated into a larger computational graph, if desired.

For better computational performance, the code is written in C++ and
CUDA.
"""

__all__ = [
    "IGNORE_LOCATION",
    "Acoustic",
    "Elastic",
    "Scalar",
    "ScalarBorn",
    "_version",
    "acoustic",
    "backend_utils",
    "elastic",
    "location_interpolation",
    "scalar",
    "scalar_born",
    "wavelets",
]

# These imports are for exposing the public API
from deepwave.acoustic import Acoustic, acoustic
from deepwave.elastic import Elastic, elastic
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born

# Import backend utilities to ensure DLL is loaded and functions are assigned
from . import _version, backend_utils, location_interpolation, wavelets
from .common import IGNORE_LOCATION

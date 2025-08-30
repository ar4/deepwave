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

# These imports are for exposing the public API
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born
from deepwave.elastic import Elastic, elastic
import deepwave.wavelets
import deepwave.location_interpolation
from ._version import __version__

# Import backend utilities to ensure DLL is loaded and functions are assigned
from . import backend_utils

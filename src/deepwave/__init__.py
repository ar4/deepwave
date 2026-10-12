"""Wave propagation modules for PyTorch.

Provides forward modelling and backpropagation of the scalar wave
equation (regular and Born), variable-density acoustic wave equation
(regular and Born), elastic wave equation (regular and Born), viscoacoustic
(SLS attenuation) wave equation, elastic VTI (anisotropic) wave equation,
and pseudo-acoustic VTI (Duveneck
2008) wave equation. It also
provides distributed acoustic sensing (DAS) strain-rate receivers
(:mod:`deepwave.das`), transducer-array helpers for ultrasound/NDT
(:mod:`deepwave.transducers`), and cycle-skipping-resistant FWI misfit
functions (:mod:`deepwave.misfits`).
It can thus be used to generate synthetic data, and to perform
FWI and LSRTM.

Module and functional interfaces are provided, and allow the propagators
to be easily integrated into a larger computational graph, if desired.

For better computational performance, the code is written in C and
CUDA.
"""

__all__ = [
    "DEFAULT_COURANT_NUMBER",
    "IGNORE_LOCATION",
    "ON_NODE_TOLERANCE",
    "Accuracy",
    "Acoustic",
    "AcousticBorn",
    "AcousticVti",
    "AuxiliaryState",
    "Callback",
    "CallbackDirection",
    "CallbackKind",
    "CallbackMode",
    "CallbackRole",
    "CallbackState",
    "DeepwaveError",
    "DeepwaveFloatingPointError",
    "DeepwaveFutureWarning",
    "DeepwaveNotImplementedError",
    "DeepwaveOSError",
    "DeepwaveRuntimeError",
    "DeepwaveTypeError",
    "DeepwaveValueError",
    "DeepwaveWarning",
    "Discretisation",
    "Elastic",
    "ElasticBorn",
    "ElasticVti",
    "Hicks",
    "InitialState",
    "LocationInterp",
    "LocationsLike",
    "MemoryEstimate",
    "MigrationWarning",
    "Plan",
    "PmlFreqInferenceWarning",
    "PropagatorOutput",
    "PropagatorSpec",
    "ReceiverSpec",
    "Reduction",
    "RunInfo",
    "Scalar",
    "ScalarBorn",
    "SourceSpec",
    "Storage",
    "StorageLike",
    "StorageMode",
    "StoragePath",
    "Viscoacoustic",
    "WavefieldView",
    "ZeroWidthFaceWarning",
    "__version__",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "anisotropy",
    "attenuation",
    "boundaries",
    "check_finite",
    "checkpointing",
    "das",
    "diagnostics",
    "discretisation",
    "elastic",
    "elastic_born",
    "elastic_vti",
    "estimate_memory",
    "introspection",
    "lambmubuoyancy_to_vpvsrho",
    "misfits",
    "plan",
    "receiver_names",
    "scalar",
    "scalar_born",
    "spec",
    "suggest_discretisation",
    "time_dispersion",
    "to_cells",
    "transducers",
    "viscoacoustic",
    "vpvsrho_to_lambmubuoyancy",
    "wavefield_names",
    "wavefields",
    "wavelets",
]

# These imports are for exposing the public API
from deepwave._plan import Plan as Plan
from deepwave._plan import plan as plan
from deepwave._storage import Storage as Storage
from deepwave.acoustic import Acoustic, acoustic
from deepwave.acoustic_born import AcousticBorn, acoustic_born
from deepwave.acoustic_vti import AcousticVti, acoustic_vti
from deepwave.discretisation import (
    DEFAULT_COURANT_NUMBER as _DEFAULT_COURANT_NUMBER,
)
from deepwave.discretisation import Discretisation, suggest_discretisation
from deepwave.elastic import Elastic, elastic
from deepwave.elastic_born import ElasticBorn, elastic_born
from deepwave.elastic_vti import ElasticVti, elastic_vti
from deepwave.introspection import (
    PropagatorSpec,
    receiver_names,
    spec,
    wavefield_names,
)
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born
from deepwave.types import ON_NODE_TOLERANCE as _ON_NODE_TOLERANCE
from deepwave.types import (
    Accuracy,
    Callback,
    CallbackDirection,
    CallbackKind,
    CallbackMode,
    CallbackRole,
    CallbackState,
    InitialState,
    LocationInterp,
    LocationsLike,
    MemoryEstimate,
    PropagatorOutput,
    ReceiverSpec,
    Reduction,
    RunInfo,
    SourceSpec,
    StorageLike,
    StorageMode,
    StoragePath,
    WavefieldView,
)
from deepwave.viscoacoustic import Viscoacoustic, viscoacoustic

# The public submodules, imported here so that they are documented by autodoc
# and reachable as ``deepwave.<name>`` after a bare ``import deepwave``.
# Five of them are *not* in ``__all__``: ``backend_utils``, ``common``,
# ``location_interpolation``, ``regular_grid`` and ``staggered_grid`` are
# deprecated compatibility shims for module spellings that shipped in v0.0.27
# (see their module docstrings). They are imported here so that a bare
# ``import deepwave`` still makes them reachable as ``deepwave.<name>``, which
# is how v0.0.27 exposed them -- importable, but not advertised.
from . import (
    anisotropy,
    attenuation,
    checkpointing,
    das,
    diagnostics,
    discretisation,
    introspection,
    misfits,
    time_dispersion,
    transducers,
    wavefields,
    wavelets,
)
from . import backend_utils as backend_utils
from . import common as common
from . import location_interpolation as location_interpolation
from . import regular_grid as regular_grid
from . import staggered_grid as staggered_grid

# Re-export the IGNORE_LOCATION sentinel and the ergonomics helpers at the
# package root so users can refer to them as ``deepwave.IGNORE_LOCATION`` /
# ``deepwave.check_finite`` / ``deepwave.estimate_memory`` without reaching
# into ``deepwave.common`` (which is now only a compatibility shim).
#
# ``IGNORE_LOCATION`` is re-exported as a plain assignment at the end of this
# file (rather than the ``from X import Y as Y`` pattern used for the other
# names here) so that its ``#:`` docstring is discoverable by Sphinx autodoc
# at the ``deepwave.IGNORE_LOCATION`` qualname used in ``docs/usage.rst``:
# autodoc resolves a data attribute's docstring by scanning its *defining*
# module for an assignment, and a plain re-import is not an assignment.
# ``Callback`` no longer needs that treatment -- it is a ``Protocol`` class,
# documented with ``autoclass``.
#
# ``check_finite`` is a cheap opt-in NaN/Inf check for propagator outputs, and
# ``estimate_memory`` gives a rough pre-flight byte-size estimate for a
# proposed model shape/nt/n_shots/storage_mode. ``vpvsrho_to_lambmubuoyancy``/
# ``lambmubuoyancy_to_vpvsrho`` convert between the elastic propagators'
# (lambda, mu, buoyancy) and (vp, vs, rho) parameterisations, and ``to_cells``
# converts physical-unit coordinates to the float grid-cell coordinates the
# propagators' float location arguments expect.
from ._common import IGNORE_LOCATION as _IGNORE_LOCATION
from ._common import (
    boundaries as boundaries,
)
from ._common import (
    check_finite as check_finite,
)
from ._common import (
    estimate_memory as estimate_memory,
)
from ._common import (
    lambmubuoyancy_to_vpvsrho as lambmubuoyancy_to_vpvsrho,
)
from ._common import (
    to_cells as to_cells,
)
from ._common import (
    vpvsrho_to_lambmubuoyancy as vpvsrho_to_lambmubuoyancy,
)
from ._engine import (
    AuxiliaryState as AuxiliaryState,
)

# Deepwave's own warning and exception types. They are defined in a private
# module but documented and raised under their public ``deepwave.<Name>``
# spelling (each class sets ``__module__ = "deepwave"``); see
# :doc:`/stability` for the deprecation policy ``DeepwaveFutureWarning``
# announces. ``DeepwaveWarning`` is the base of the cause-named warning
# categories; every warning raised under one of them names its own category
# in its text, so the filter line is always in the message being read. The
# plain ``UserWarning``s that predate these categories are not classified
# under them and name none.
from ._exceptions import (
    DeepwaveError as DeepwaveError,
)
from ._exceptions import (
    DeepwaveFloatingPointError as DeepwaveFloatingPointError,
)
from ._exceptions import (
    DeepwaveFutureWarning as DeepwaveFutureWarning,
)
from ._exceptions import (
    DeepwaveNotImplementedError as DeepwaveNotImplementedError,
)
from ._exceptions import (
    DeepwaveOSError as DeepwaveOSError,
)
from ._exceptions import (
    DeepwaveRuntimeError as DeepwaveRuntimeError,
)
from ._exceptions import (
    DeepwaveTypeError as DeepwaveTypeError,
)
from ._exceptions import (
    DeepwaveValueError as DeepwaveValueError,
)
from ._exceptions import (
    DeepwaveWarning as DeepwaveWarning,
)
from ._exceptions import (
    MigrationWarning as MigrationWarning,
)
from ._exceptions import (
    PmlFreqInferenceWarning as PmlFreqInferenceWarning,
)
from ._exceptions import (
    ZeroWidthFaceWarning as ZeroWidthFaceWarning,
)

# ``Hicks`` is the only public surface of the (private) location-interpolation
# module; the module itself is private because ``location_interpolation`` is
# also the name of the propagators' own keyword argument (a string mode), and
# the two must not be confused.
from ._location_interpolation import Hicks as Hicks

# ``__version__`` comes from the (build-time generated) ``_version`` module.
from ._version import __version__ as __version__

#: Sentinel value to place in a source or receiver location tensor to mark
#: that point as unused, allowing a single rectangular
#: ``[shot, point, ndim]`` locations tensor to describe shots that have
#: differing numbers of active sources or receivers.
#:
#: A point whose coordinates are *all* ``IGNORE_LOCATION`` is ignored: a
#: source injects nothing, a receiver records zeros. A point with only
#: *some* coordinates set to the sentinel raises, because that is far more
#: likely to be a mistake than an intention.
#:
#: **Its numeric value is unspecified and may change.** Always compare
#: against, and assign from, ``deepwave.IGNORE_LOCATION`` itself; never
#: hard-code the number it happens to hold.
#:
#: The ``*_mask`` arguments are the modern spelling of the same mechanism --
#: a ``[shot, point]`` boolean tensor, ``True`` = active -- and produce
#: bit-identical results. Supplying both for one component raises.
IGNORE_LOCATION = _IGNORE_LOCATION

#: How close to a grid node a coordinate must be, **in cells**, for Deepwave
#: to treat it as being *on* that node.
#:
#: This one number decides three things:
#:
#: * ``location_interpolation="auto"`` builds a :class:`deepwave.Hicks`
#:   interpolator only if some coordinate is further than this from its
#:   component's node (but see that argument's documentation for the
#:   ``requires_grad`` exception, which always interpolates);
#: * ``location_interpolation="none"`` rejects a locations Tensor that has
#:   any coordinate further than this off-node;
#: * :func:`deepwave.to_cells` with ``snap=True`` cleans up floating-point
#:   error out to this distance.
#:
#: It is the single source for all three: ``Hicks``' default ``eps`` is this
#: value, and so is the propagators' internal node test. Changing it changes
#: which calls are interpolated, and therefore changes numbers.
#:
#: Stability: stable (see :doc:`/stability`).
ON_NODE_TOLERANCE = _ON_NODE_TOLERANCE

#: The Courant number Deepwave targets when ``courant_number`` is not given.
#:
#: Every propagator, and
#: :func:`deepwave.discretisation.cfl_condition_n`, uses this value to choose
#: the internal time step when the caller leaves ``courant_number=None``. It
#: is deliberately conservative -- the same number for every ``accuracy``
#: order and every dimensionality, well below the critical Courant number
#: :func:`deepwave.discretisation.critical_courant_number` reports for any of
#: them -- because it is the value Deepwave has always used, and a run that
#: does not set ``courant_number`` is reproduced bit-for-bit by passing this
#: constant explicitly.
#:
#: It is exported so that the default can be named rather than hard-coded:
#: ``courant_number=1.2 * deepwave.DEFAULT_COURANT_NUMBER`` says what it
#: means, and ``RunInfo.courant_number`` can be compared against
#: it to tell a defaulted run from an explicit one. Raising
#: ``courant_number`` above it permits a larger internal time step (fewer
#: steps, faster) at the cost of stability margin; the value is validated
#: against the critical Courant number and rejected if it would be unstable.
#:
#: Stability: stable (see :doc:`/stability`).
DEFAULT_COURANT_NUMBER = _DEFAULT_COURANT_NUMBER

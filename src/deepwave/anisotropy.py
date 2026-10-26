"""Anisotropic-model helpers: Thomsen/Alkhalifah conversions and a taper.

Stability: provisional (see :doc:`/stability`) -- this module is expected to
gain conversions, so its shape may still be corrected.

The home for the conversions between the parameterisations the anisotropic
propagators accept, and for :func:`source_taper`, the near-source stability
taper :func:`deepwave.acoustic_vti` needs.

These names are re-exported from the propagator modules that define them, so
:func:`deepwave.anisotropy.thomsen_to_cij` and
``deepwave.elastic_vti.thomsen_to_cij`` are the same object; this module is
the documented home and the one to import from.

The isotropic converters :func:`deepwave.vpvsrho_to_lambmubuoyancy` and
:func:`deepwave.lambmubuoyancy_to_vpvsrho` deliberately stay at the top
level: they are not anisotropic, and they shipped at that spelling.
"""

from deepwave.acoustic_vti import acoustic_vti_source_taper as source_taper
from deepwave.elastic_vti import (
    alkhalifah_to_thomsen as alkhalifah_to_thomsen,
)
from deepwave.elastic_vti import (
    cij_to_thomsen as cij_to_thomsen,
)
from deepwave.elastic_vti import (
    thomsen_to_alkhalifah as thomsen_to_alkhalifah,
)
from deepwave.elastic_vti import (
    thomsen_to_cij as thomsen_to_cij,
)

__all__ = [
    "alkhalifah_to_thomsen",
    "cij_to_thomsen",
    "source_taper",
    "thomsen_to_alkhalifah",
    "thomsen_to_cij",
]

# The five are defined in the propagator modules but *documented and
# supported* here, so they report this module -- ``repr``, ``help()``, the
# generated API extract and Sphinx autodoc all name the canonical home
# rather than the file the definition happens to sit in. ``source_taper``
# additionally reports its new, shorter name: the old
# ``acoustic_vti_source_taper`` spelled the propagator into the function
# name, which this module's name now carries.
source_taper.__name__ = "source_taper"
source_taper.__qualname__ = "source_taper"
for _fn in (
    source_taper,
    thomsen_to_cij,
    cij_to_thomsen,
    alkhalifah_to_thomsen,
    thomsen_to_alkhalifah,
):
    _fn.__module__ = "deepwave.anisotropy"
del _fn

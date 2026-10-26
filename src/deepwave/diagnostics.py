"""Environment diagnostics, for bug reports.

:func:`report` collects the facts that are asked for first whenever
something behaves differently on one machine than another -- Deepwave's and
PyTorch's versions, the CPU's SIMD support against the compiled backend's
instruction-set baseline, and the visible CUDA devices -- and formats them
for pasting into an issue.

Stability: provisional (see :doc:`/stability`). :class:`Report` may gain
*fields* as the useful set of facts changes, and the ones it has are not
renamed or withdrawn without a release's notice under
:class:`deepwave.DeepwaveFutureWarning`; the ``str(report())`` text is meant
to be read, not parsed.
"""

import platform
from dataclasses import dataclass
from typing import Optional

import torch

from deepwave._backend_utils import (
    _CPU_FEATURE_NAMES,
    _cpu_supports,
    _required_baseline,
)
from deepwave._version import __version__

__all__ = ["Report", "report"]


@dataclass(frozen=True)
class Report:
    """A snapshot of the environment Deepwave is running in.

    Returned by :func:`report`. Print it (or ``str()`` it) to get the text
    to paste into a bug report; read the individual fields if you need one
    fact programmatically.

    Attributes:
        deepwave_version: Deepwave's version string.
        torch_version: PyTorch's version string.
        python_version: The running Python's version string.
        machine: The machine architecture, from ``platform.machine()``.
        required_baseline: The instruction-set baseline the compiled
            backend was built to require. An empty string reports no fixed
            ISA requirement; ``None`` means the backend did not report it.
        simd_features: Per-feature CPU SIMD support. ``True``/``False``
            where the feature is meaningful and could be probed, ``None``
            where it is not applicable (any non-x86 machine).
        cuda_available: Whether ``torch.cuda.is_available()``.
        cuda_devices: The name of each visible CUDA device, in device
            order. Empty when CUDA is unavailable.

    Stability: provisional (see :doc:`/stability`).
    """

    deepwave_version: str
    torch_version: str
    python_version: str
    machine: str
    required_baseline: Optional[str]
    simd_features: dict[str, Optional[bool]]
    cuda_available: bool
    cuda_devices: tuple[str, ...]

    def __str__(self) -> str:
        """A multi-line report, formatted for pasting into a bug report."""
        simd = ", ".join(
            f"{name}={'yes' if value else 'no' if value is False else 'n/a'}"
            for name, value in self.simd_features.items()
        )
        if self.cuda_available:
            devices = ", ".join(self.cuda_devices) or "none visible"
        else:
            devices = "not available"
        baseline = (
            "portable (no fixed ISA requirement)"
            if self.required_baseline == ""
            else self.required_baseline or "not reported"
        )
        return "\n".join(
            (
                f"Deepwave {self.deepwave_version}",
                f"PyTorch  {self.torch_version}",
                f"Python   {self.python_version}",
                f"Machine  {self.machine}",
                f"Backend instruction-set baseline: {baseline}",
                f"CPU SIMD: {simd}",
                f"CUDA: {devices}",
            )
        )


Report.__module__ = "deepwave.diagnostics"


def report() -> Report:
    """Collects the environment facts a Deepwave bug report should carry.

    Print the result::

        >>> import deepwave
        >>> print(deepwave.diagnostics.report())  # doctest: +SKIP
        Deepwave 0.1.0
        PyTorch  2.5.1
        ...

    Returns:
        A :class:`Report`. It is a dataclass with named fields rather than
        a plain ``dict`` so that the shape of the answer is documented and
        so that nothing is tempted to parse the printed text.

    Stability: provisional (see :doc:`/stability`).
    """
    features: dict[str, Optional[bool]] = {}
    for name in _CPU_FEATURE_NAMES:
        supported = _cpu_supports(name)
        features[name] = None if supported is None or supported < 0 else bool(supported)
    cuda_available = torch.cuda.is_available()
    devices: tuple[str, ...] = ()
    if cuda_available:
        devices = tuple(
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        )
    return Report(
        deepwave_version=__version__,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        machine=platform.machine(),
        required_baseline=_required_baseline(),
        simd_features=features,
        cuda_available=cuda_available,
        cuda_devices=devices,
    )

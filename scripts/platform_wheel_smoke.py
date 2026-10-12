"""Installed-wheel identity and native ABI smoke for every retained family.

Run with the candidate wheel installed in a clean interpreter, from a working
directory that is not a source-tree ``src``. This deliberately fails when an
editable/source Python package loads a different installed native library.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def check_backend_symbols(
    library: Any, registry: Mapping[str, Any], *, require_cuda: bool = False
) -> bool:
    """Require complete compiled variants without executing any GPU code.

    CPU symbols are mandatory. Any CUDA symbol means the entire CUDA set
    must be present; release builders can also require CUDA explicitly so
    a silently CPU-only Linux/Windows wheel cannot pass qualification.
    """
    stems = [
        f"{spec.symbol_family}_{ndim}d_{accuracy}_{dtype}_"
        f"{variant.direction}{variant.suffix}"
        for spec in registry.values()
        for variant in spec.variants
        for ndim in sorted(spec.ndim_range & variant.ndim_range)
        for accuracy in (2, 4, 6, 8)
        for dtype in ("float", "double")
    ]
    if not stems:
        raise AssertionError("no expected backend symbols")
    has_cuda = any(hasattr(library, f"{stem}_cuda") for stem in stems)
    devices = ("cpu", "cuda") if require_cuda or has_cuda else ("cpu",)
    missing = [
        f"{stem}_{device}"
        for device in devices
        for stem in stems
        if not hasattr(library, f"{stem}_{device}")
    ]
    if missing:
        raise AssertionError(
            f"native library is missing {len(missing)} expected symbols: "
            + ", ".join(missing[:10])
        )
    return has_cuda


def main() -> None:
    """Check installed identity, then exercise native forward and adjoint ABIs."""
    import torch

    import deepwave
    from deepwave._backend_utils import PROPAGATOR_REGISTRY, dll
    from deepwave._capabilities import PROPAGATOR_NAMES, _models, _receiver

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require CUDA symbols (no GPU needed).",
    )
    args = parser.parse_args()
    package_dir = Path(deepwave.__file__).resolve().parent
    library_path = Path(dll._name).resolve()  # noqa: SLF001 - ctypes library identity
    if not library_path.is_file() or library_path.parent != package_dir:
        raise AssertionError(
            f"Python package {package_dir} loaded native library {library_path}"
        )
    required_notices = []
    if (package_dir / "libiomp5md.dll").is_file():
        required_notices.append("INTEL_OPENMP_LICENSE.txt")
    if any((package_dir.parent / "deepwave.libs").glob("libgomp*")):
        required_notices.extend(("GNU_GPL_3.txt", "GCC_RUNTIME_EXCEPTION.txt"))
    has_cuda = check_backend_symbols(
        dll, PROPAGATOR_REGISTRY, require_cuda=args.require_cuda
    )
    if has_cuda:
        cuda_notices = list((package_dir / "licenses").glob("NVIDIA_CUDA_EULA.*"))
        if not cuda_notices or any(path.stat().st_size < 100 for path in cuda_notices):
            raise AssertionError(
                "CUDA-enabled wheel is missing toolkit redistribution notices"
            )
    for filename in required_notices:
        notice = package_dir / "licenses" / filename
        if not notice.is_file() or notice.stat().st_size < 100:
            raise AssertionError(
                f"bundled runtime is missing its redistribution notice: {filename}"
            )
    for name in PROPAGATOR_NAMES:
        for ndim in deepwave.spec(name).ndims:
            primary = _models(name, ndim)[0].clone().requires_grad_(True)
            trace = _receiver(name, ndim, "compiled", primary)
            if not trace.numel() or not bool(torch.isfinite(trace).all()):
                raise AssertionError(f"{name}/{ndim}D returned empty/nonfinite data")
            if not bool(torch.count_nonzero(trace)):
                raise AssertionError(f"{name}/{ndim}D produced zero receiver data")
            (gradient,) = torch.autograd.grad(trace.square().sum(), primary)
            if not bool(torch.isfinite(gradient).all()) or not bool(
                torch.count_nonzero(gradient)
            ):
                raise AssertionError(f"{name}/{ndim}D produced invalid model gradient")
            print(f"{name}/{ndim}D native forward+adjoint OK")
    print(f"installed package={package_dir}; native library={library_path}")


if __name__ == "__main__":
    main()

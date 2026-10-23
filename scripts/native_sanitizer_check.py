"""Bounded native sanitizer paths; scientific accuracy is qualified separately."""

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

# Reuse the deterministic survey fixture used by the checkpoint benchmarks.
# ruff: noqa: SLF001

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

import deepwave
from benchmarks import checkpoint_reentrancy as cp


def main() -> None:
    """Run the bounded native-path matrix and report its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    installed = Path(deepwave.__file__).parent
    native = installed / "libdeepwave_C.so"
    if not native.is_file():
        raise RuntimeError(f"Native library missing: {native}")
    identity = {
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "native": str(native),
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
    }
    torch.set_num_threads(1)
    cp._DT = 0.0002
    cases = []
    for family in cp.PROPAGATORS:
        for ndim in (2, 3):
            for dtype in ("float32", "float64"):
                for accuracy in (2, 4, 6, 8):
                    modes = ("device", "cpu", "disk") if accuracy == 8 else ("device",)
                    for storage in modes:
                        cases.append(
                            {
                                "family": family,
                                "ndim": ndim,
                                "dtype": dtype,
                                "accuracy": accuracy,
                                "storage": storage,
                                "compression": False,
                                "mpml": 0,
                                "mask": "all",
                                "shots": 1,
                            }
                        )
                        if family in ("scalar", "elastic"):
                            cases.append({**cases[-1], "compression": True})
                    if family == "elastic_vti":
                        common = {
                            "family": family,
                            "ndim": ndim,
                            "dtype": dtype,
                            "accuracy": accuracy,
                            "storage": "device",
                            "compression": False,
                        }
                        cases.append({**common, "mpml": 0.2, "mask": "all", "shots": 1})
                        cases.extend(
                            {
                                **common,
                                "mpml": mpml,
                                "mask": "buoyancy_only",
                                "shots": 2,
                            }
                            for mpml in (0, 0.2)
                        )
    if len(cases) != 312 or len({json.dumps(c, sort_keys=True) for c in cases}) != 312:
        raise RuntimeError("Native path selection is incomplete or contains duplicates")
    completed = []
    start = time.monotonic()
    for case in cases:
        cp._DTYPE = getattr(torch, case["dtype"])
        family, ndim, shots = case["family"], case["ndim"], case["shots"]
        with tempfile.TemporaryDirectory() as storage_path:
            models, sources, kwargs = cp._problem(
                family, ndim, 8, 16, shots, torch.device("cpu")
            )
            perturbations = tuple(cp._perturbations(family, (8,) * ndim))
            if shots > 1:
                models = [x.unsqueeze(0).repeat(shots, *([1] * ndim)) for x in models]
                for name in (*perturbations, *(("c66",) if "c66" in kwargs else ())):
                    kwargs[name] = (
                        kwargs[name].unsqueeze(0).repeat(shots, *([1] * ndim))
                    )
            leaves = cp._leaves(models, sources, kwargs, perturbations)
            if "c66" in kwargs:
                kwargs["c66"].requires_grad_()
                leaves.append(kwargs["c66"])
            if case["mask"] == "buoyancy_only":
                for leaf in leaves:
                    leaf.requires_grad_(False)
                models[-1].requires_grad_()
                leaves = [models[-1]]
            kwargs.update(
                accuracy=case["accuracy"],
                backend="compiled",
                storage_mode=case["storage"],
                storage_compression=case["compression"],
            )
            if family == "elastic_vti":
                kwargs["mpml_fraction"] = case["mpml"]
            if case["storage"] == "disk":
                kwargs["storage_path"] = storage_path
            tick = time.monotonic()
            out = getattr(deepwave, family)(
                *models, cp._DX, cp._DT, **sources, **kwargs
            )
            loss = cp._loss(out) * 1e6
            if not torch.isfinite(loss) or loss == 0:
                raise RuntimeError(f"Non-finite or zero loss: {case}")
            loss.backward()
            for leaf in leaves:
                if (
                    leaf.grad is None
                    or not torch.isfinite(leaf.grad).all()
                    or not torch.count_nonzero(leaf.grad)
                ):
                    raise RuntimeError(f"Missing, non-finite or zero gradient: {case}")
            row = {
                **case,
                "seconds": time.monotonic() - tick,
                "loss": float(loss),
                "gradient_norms": [float(x.grad.double().norm()) for x in leaves],
            }
            completed.append(row)

    result = {
        "valid": True,
        "cases": len(cases),
        "completed": completed,
        "seconds": time.monotonic() - start,
        "instrumented_identity": identity,
        "installed": str(installed),
        "n": 8,
        "nt": 16,
        "pml_width": 3,
        "purpose": "ASan/UBSan native path coverage",
        "independent_scientific_evidence": "Normal and physics workflows",
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid": True,
                "cases": len(cases),
                "seconds": result["seconds"],
                "identity": identity,
            }
        )
    )


if __name__ == "__main__":
    main()

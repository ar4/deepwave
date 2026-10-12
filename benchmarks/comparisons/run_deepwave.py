"""Run the deepwave side of the tool-comparison benchmark.

Reuses `benchmarks.run.run_one_case` (the existing benchmark harness -- one
dedicated subprocess per case, `OMP_NUM_THREADS` pinned before that
process's own `import torch`, peak RSS read from a clean process) rather
than re-implementing timing/measurement. See `params.py` for why this
`Case` is exactly deepwave's own micro-tier `scalar` 2-D case.

Usage (from an environment with deepwave installed):
    python benchmarks/comparisons/run_deepwave.py

Writes `benchmarks/comparisons/results_deepwave.json` and prints it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import params  # noqa: E402

from benchmarks.cases import Case  # noqa: E402 -- path insert must come first
from benchmarks.run import run_one_case  # noqa: E402

OUT_PATH = _THIS_DIR / "results_deepwave.json"


def main() -> int:
    """Run the matched deepwave case and write/print its result JSON."""
    case = Case(
        propagator="scalar",
        ndim=2,
        accuracy=params.ACCURACY,
        dtype="float32",
        mode="forward_backward",
        tier="micro",
    )
    # Sanity check: this case's grid/nt must equal params.py's matched
    # workload, since params.py documents that they were chosen to match.
    if case.size_per_dim != params.GRID_SIZE or case.nt != params.NT:
        raise ValueError(
            f"case grid/nt {case.size_per_dim, case.nt} does not match "
            f"params.py {params.GRID_SIZE, params.NT}"
        )

    result = run_one_case(
        case,
        reps=params.REPS,
        warmup=params.WARMUP,
        threads=params.THREADS,
        timeout=120.0,
    )
    text = json.dumps(result, indent=2)
    print(text)
    OUT_PATH.write_text(text)
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())

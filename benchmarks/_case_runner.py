"""Runs exactly one benchmark case in its own process.

Invoked by `run.py` as a subprocess (`python -m benchmarks._case_runner`)
so that:
  - `OMP_NUM_THREADS` (set by the parent in the subprocess environment,
    i.e. before this interpreter -- and therefore before `torch` --
    starts) pins the C backend's OpenMP thread count for this case only.
  - `/proc/self/status` VmHWM (read in `harness.run_case`) is a per-case
    peak RSS rather than a running-maximum across the whole suite.

Protocol: a JSON object is read from stdin:
    {"case": {...Case fields...}, "reps": int, "warmup": int,
     "threads": int, "storage_scratch_dir": str | null}
and exactly one JSON object (the result of `harness.run_case`) is written
to stdout. Nothing else should be written to stdout; diagnostics go to
stderr.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.loads(sys.stdin.read())

    threads = int(payload["threads"])
    import torch

    torch.set_num_threads(threads)

    from .cases import Case
    from .harness import run_case

    case = Case(**payload["case"])
    result = run_case(
        case,
        reps=int(payload["reps"]),
        warmup=int(payload["warmup"]),
        threads=threads,
        storage_scratch_dir=payload.get("storage_scratch_dir"),
    )
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

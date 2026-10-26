"""Run the Devito side of the tool-comparison benchmark.

Uses Devito's own `examples.seismic.acoustic` tutorial helpers (shipped
inside the `devito` pip package) to build the constant-density acoustic
FWI-gradient recipe: `solver.forward(save=True)` (forward, saving the full
wavefield the adjoint pass needs) followed by `solver.jacobian_adjoint`
(the adjoint-state gradient -- Devito's adjoint is derived at the equation
level by its symbolic engine, not by autodiff, per `docs/comparison.rst`).
Grid size, spacing, number of time steps, dt, background velocity, PML/nbl
width, source wavelet and source/receiver layout are all taken from
`params.py`/`wavelet.py` so the workload matches deepwave's and j-Wave's.

The first call to `solver.forward`/`solver.jacobian_adjoint` triggers
Devito's runtime C-code generation + gcc JIT compile; `params.WARMUP`
untimed reps absorb that cost, matching the other two runners'
warmup convention.

Must be run with the interpreter of a separate virtual environment holding
`devito` (this script is NOT importable from the environment deepwave is
installed in):

    /path/to/comparison-venv/bin/python benchmarks/comparisons/run_devito.py

Writes `benchmarks/comparisons/results_devito.json` and prints it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("DEVITO_LANGUAGE", "openmp")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("DEVITO_LOGGING", "ERROR")

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import params  # noqa: E402
from memtools import read_vmhwm_bytes  # noqa: E402
from wavelet import ricker  # noqa: E402

OUT_PATH = _THIS_DIR / "results_devito.json"


def main() -> int:
    """Build the matched Devito acoustic case, time forward + gradient, write JSON."""
    import devito
    import numpy as np
    from devito import Function, configuration
    from examples.seismic import AcquisitionGeometry, Model
    from examples.seismic.acoustic import AcousticWaveSolver

    result: dict = {
        "tool": "Devito",
        "version": devito.__version__,
        "language": configuration["language"],
        "platform": str(configuration["platform"]),
        "error": None,
    }

    try:
        shape = (params.GRID_SIZE, params.GRID_SIZE)
        spacing = (params.GRID_SPACING, params.GRID_SPACING)
        origin = (0.0, 0.0)
        vp = np.full(shape, params.BACKGROUND_V, dtype=np.float32)

        model = Model(
            vp=vp,
            origin=origin,
            shape=shape,
            spacing=spacing,
            space_order=params.ACCURACY,
            nbl=params.PML_WIDTH,
            dtype=np.float32,
            dt=params.DT,  # force the exact matched dt (see params.py)
        )

        src_coords = np.array(
            [[p * params.GRID_SPACING for p in params.SOURCE_POINT]],
            dtype=np.float32,
        )
        rec_coords = np.array(
            [[p * params.GRID_SPACING for p in pt] for pt in params.RECEIVER_POINTS],
            dtype=np.float32,
        )

        t0 = 0.0
        tn = (params.NT - 1) * params.DT
        geometry = AcquisitionGeometry(
            model, rec_coords, src_coords, t0, tn, src_type=None
        )
        if geometry.nt != params.NT:
            raise ValueError(f"geometry.nt {geometry.nt} != params.NT {params.NT}")
        if geometry.dt != params.DT:
            raise ValueError(f"geometry.dt {geometry.dt} != params.DT {params.DT}")

        solver = AcousticWaveSolver(model, geometry, space_order=params.ACCURACY)

        src = geometry.new_src()
        wavelet = ricker(
            params.SOURCE_FREQ, params.NT, params.DT, params.SOURCE_PEAK_TIME
        ).astype(np.float32)
        src.data[:, 0] = wavelet

        def one_forward() -> tuple[Any, Any]:
            rec, u, _summary = solver.forward(save=True, src=src)
            return rec, u

        def one_grad(rec: Any, u: Any) -> Any:
            grad = Function(name="grad", grid=model.grid)
            grad_out, _summary = solver.jacobian_adjoint(rec=rec, u=u, grad=grad)
            return grad_out

        # Warmup: triggers Devito's C-code generation + gcc JIT compile.
        rec = u = None
        for _ in range(params.WARMUP):
            rec, u = one_forward()
            one_grad(rec, u)

        fwd_times = []
        for _ in range(params.REPS):
            t0_ = time.perf_counter()
            rec, u = one_forward()
            fwd_times.append(time.perf_counter() - t0_)

        grad_times = []
        for _ in range(params.REPS):
            t0_ = time.perf_counter()
            grad_out = one_grad(rec, u)
            grad_times.append(time.perf_counter() - t0_)

        fwd_times.sort()
        grad_times.sort()
        result["forward_s_reps"] = fwd_times
        result["grad_s_reps"] = grad_times
        result["forward_median_s"] = fwd_times[len(fwd_times) // 2]
        result["grad_median_s"] = grad_times[len(grad_times) // 2]
        result["rec_abs_max"] = float(np.abs(rec.data).max())
        result["grad_abs_max"] = float(np.abs(grad_out.data).max())

    except Exception as exc:  # noqa: BLE001 -- reported, not raised, by design
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["vmhwm_bytes"] = read_vmhwm_bytes()

    text = json.dumps(result, indent=2)
    print(text)
    OUT_PATH.write_text(text)
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run the j-Wave side of the tool-comparison benchmark.

j-Wave (https://github.com/ucl-bug/jwave) implements a k-space/PSTD
(pseudospectral, spectral-derivative) time-domain acoustic solver in JAX,
not a finite-difference one -- there is no "accuracy order" knob to match
`params.ACCURACY`, so the two methods are compared on the same *problem*
rather than on the same discretisation. Grid size, spacing,
number of time steps, dt, background velocity, and the source wavelet are
all taken from `params.py`/`wavelet.py` so the *problem* matches even
though the numerical method differs.

Must be run with the interpreter of a separate virtual environment holding
`jwave` (this script is NOT importable from the environment deepwave is
installed in -- it does not import deepwave or torch at all):

    /path/to/comparison-venv/bin/python benchmarks/comparisons/run_jwave.py

Writes `benchmarks/comparisons/results_jwave.json` and prints it.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import params  # noqa: E402
from memtools import read_vmhwm_bytes  # noqa: E402
from wavelet import ricker  # noqa: E402

OUT_PATH = _THIS_DIR / "results_jwave.json"


def main() -> int:
    """Build the matched acoustic case in j-Wave, time forward + grad, write JSON."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_platforms", "cpu")

    from jwave.geometry import Domain, Medium, Sensors, Sources, TimeAxis

    result: dict = {
        "tool": "j-Wave",
        "version": importlib.metadata.version("jwave"),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "error": None,
    }

    try:
        n = (params.GRID_SIZE, params.GRID_SIZE)
        dx = (params.GRID_SPACING, params.GRID_SPACING)
        domain = Domain(n, dx)

        sound_speed0 = jnp.ones(n, dtype=jnp.float32) * params.BACKGROUND_V

        signal = jnp.asarray(
            ricker(params.SOURCE_FREQ, params.NT, params.DT, params.SOURCE_PEAK_TIME),
            dtype=jnp.float32,
        )
        src_x = (params.SOURCE_POINT[0],)
        src_y = (params.SOURCE_POINT[1],)
        sources = Sources(
            positions=(src_x, src_y),
            signals=jnp.stack([signal]),
            dt=params.DT,
            domain=domain,
        )

        rec_x = tuple(p[0] for p in params.RECEIVER_POINTS)
        rec_y = tuple(p[1] for p in params.RECEIVER_POINTS)
        sensors = Sensors(positions=(rec_x, rec_y))

        time_axis = TimeAxis(dt=params.DT, t_end=params.NT * params.DT)
        if int(time_axis.Nt) != params.NT:
            raise ValueError(f"time_axis.Nt {time_axis.Nt} != params.NT {params.NT}")

        from jwave.acoustics import simulate_wave_propagation

        def forward(sound_speed: jax.Array) -> jax.Array:
            medium = Medium(
                domain=domain,
                sound_speed=sound_speed,
                density=1.0,
                pml_size=float(params.PML_WIDTH),
            )
            return simulate_wave_propagation(
                medium, time_axis, sources=sources, sensors=sensors
            )

        def loss(sound_speed: jax.Array) -> jax.Array:
            p = forward(sound_speed)
            return jnp.sum(p)

        forward_jit = jax.jit(forward)
        grad_jit = jax.jit(jax.grad(loss))

        # Warmup: triggers JAX tracing/XLA compilation, discarded from timing.
        for _ in range(params.WARMUP):
            out = forward_jit(sound_speed0)
            jax.block_until_ready(out)
            g = grad_jit(sound_speed0)
            jax.block_until_ready(g)

        fwd_times = []
        for _ in range(params.REPS):
            t0 = time.perf_counter()
            out = forward_jit(sound_speed0)
            jax.block_until_ready(out)
            fwd_times.append(time.perf_counter() - t0)

        grad_times = []
        for _ in range(params.REPS):
            t0 = time.perf_counter()
            g = grad_jit(sound_speed0)
            jax.block_until_ready(g)
            grad_times.append(time.perf_counter() - t0)

        fwd_times.sort()
        grad_times.sort()
        result["forward_s_reps"] = fwd_times
        result["grad_s_reps"] = grad_times
        result["forward_median_s"] = fwd_times[len(fwd_times) // 2]
        result["grad_median_s"] = grad_times[len(grad_times) // 2]

    except Exception as exc:  # noqa: BLE001 -- reported, not raised, by design
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["vmhwm_bytes"] = read_vmhwm_bytes()

    text = json.dumps(result, indent=2)
    print(text)
    OUT_PATH.write_text(text)
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())

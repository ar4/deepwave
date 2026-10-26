"""Shared parameters for the deepwave-vs-third-party comparison benchmark.

Every runner script in this directory (deepwave, j-Wave, Devito) is built
from these exact numbers so the workload is genuinely matched: same grid
size, same number of time steps, same time step, same background velocity,
same source wavelet, same source/receiver layout. This module is stdlib
only (no torch/jax/deepwave imports) so it can be imported unmodified both
from the environment deepwave is installed in (the deepwave runner) and
from the separate comparison environment (the j-Wave/Devito runners) with
no extra installs.

These constants were deliberately chosen to exactly equal deepwave's own
benchmark suite case `Case(propagator="scalar", ndim=2, accuracy=4,
dtype="float32", mode="forward_backward", tier="micro")` (see
`benchmarks/cases.py` and `benchmarks/models.py`) -- so the deepwave number
in this comparison is a direct read of the existing benchmark suite, not a
bespoke workload. `deepwave.scalar` (constant-density acoustic) is used
rather than `deepwave.acoustic` (variable-density) because it is the
physics every other tool in this comparison also implements.

DT was computed once via `deepwave.discretisation.cfl_condition_n` for this exact
grid/velocity/accuracy (`c_max=0.5`, matching `benchmarks/models.py`'s
`_dt`) and is hardcoded here so every tool uses the identical value rather
than each tool picking its own CFL-safe step.
"""

from __future__ import annotations

GRID_SIZE = 128  # cells per spatial dimension (2-D: 128 x 128 grid)
GRID_SPACING = 5.0  # metres
NT = 200  # number of (recorded) time steps
DT = 0.0006349206349206349  # seconds (CFL-safe for accuracy=4, c_max=0.5)
ACCURACY = 4  # finite-difference spatial order (deepwave/Devito; j-Wave is
# spectral (k-space/PSTD) and has no analogous "order" knob, so it is
# matched on the problem rather than on the discretisation)
BACKGROUND_V = 1500.0  # m/s, homogeneous background velocity
SOURCE_FREQ = 15.0  # Hz, Ricker wavelet centre frequency
SOURCE_PEAK_TIME = 1.5 / SOURCE_FREQ  # seconds, matches benchmarks/models.py
PML_WIDTH = 10  # cells, absorbing-boundary width (deepwave PML; mirrored as
# nbl for Devito / pml_size for j-Wave)

SOURCE_POINT = [64, 64]  # [dim0, dim1] grid indices, matches models.py
RECEIVER_POINTS = [[32, 53], [53, 74], [74, 95], [95, 117]]  # grid indices

REPS = 3  # timed repetitions per tool (median reported)
WARMUP = 1  # untimed warmup reps (discards first-call/JIT-compile cost)

THREADS = 4  # OMP_NUM_THREADS / equivalent CPU thread count pin

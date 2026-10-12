"""Benchmark case definitions.

A `Case` is one point in the (propagator, ndim, accuracy, dtype, mode, tier)
axis space. `cases.py` only defines *what* to run; `models.py` turns a `Case`
into actual tensors/kwargs, and `harness.py` times it.

Axes:
    propagator: all nine retained families
    ndim:       1, 2, 3
    accuracy:   2, 4, 8   (finite-difference order)
    dtype:      "float32", "float64"
    mode:       "forward", "forward_backward"
    backend:    "compiled" (default; the hand-written C/CUDA backend),
                "eager", "torch_compile" (PyTorch's `torch.compile` path)
                -- see deepwave.scalar's `backend=` kwarg.
    device:     "cpu", "cuda"
    tier:       "micro", "meso", "macro" -- controls problem size, see
                TIER_SIZES below.

Not every combination is exercised by default (the full cross product is
300 propagator/ndim/accuracy/dtype/mode points before even considering
tiers or storage_mode). `DEFAULT_CASES` is a small, fast, representative
subset; `--all` requests the full cross product.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

PROPAGATORS = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "elastic",
    "elastic_born",
    "elastic_vti",
    "acoustic_vti",
    "viscoacoustic",
)
NDIMS = (1, 2, 3)
ACCURACIES = (2, 4, 8)
DTYPES = ("float32", "float64")
MODES = ("forward", "forward_backward")
STORAGE_MODES = ("device", "cpu", "disk", "none")
# Mirrors the `Literal["compiled", "eager", "torch_compile"]` accepted by
# every propagator's own `backend=` kwarg (e.g. `deepwave.scalar`).
BACKENDS = ("compiled", "eager", "torch_compile")
DEVICES = ("cpu", "cuda")

# Per-tier, per-ndim (grid_size_per_dim, nt). The 2-D/3-D meso and the 3-D
# macro values are a deliberately chosen benchmark-suite sizing target. The
# 1-D and 3-D-micro values are this suite's own choice, picked to have a
# roughly comparable total interior-cell count to the 2-D value at the same
# tier so that tiers stay meaningfully ordered across ndim.
TIER_SIZES = {
    "micro": {1: (4_096, 200), 2: (128, 200), 3: (32, 100)},
    "meso": {1: (65_536, 500), 2: (512, 1000), 3: (128, 500)},
    "macro": {1: (1_048_576, 2000), 2: (2048, 1000), 3: (256, 1000)},
}

# Per-(propagator, tier, ndim) size/nt overrides for propagators whose
# forward_backward footprint is far larger than the tier's nominal case. The
# elastic propagator stores ~9 wavefield components per adjoint snapshot vs the
# scalar propagator's ~1, so at a matched size its forward_backward peak RSS is
# ~7x scalar's (measured). At the nominal 3-D meso size (128^3, nt=500) the
# elastic forward_backward case needs ~65 GB -- it OOM-kills any host with less
# RAM. At 96^3/nt=200 it peaks at ~14.6 GB (measured on a 23 GiB box), which
# fits a typical workstation. The override applies to elastic 3-D meso only, so
# scalar/acoustic/scalar_born 3-D meso stay at 128^3/nt=500 and remain directly
# comparable to previously recorded runs. Every result JSON records the actual
# size_per_dim/nt, so a run is always self-describing regardless of overrides.
SIZE_OVERRIDES: dict[tuple[str, str, int], tuple[int, int]] = {
    ("elastic", "meso", 3): (96, 200),
    **{
        (name, "meso", 3): (64, 100)
        for name in (
            "acoustic_born",
            "elastic_born",
            "elastic_vti",
            "acoustic_vti",
            "viscoacoustic",
        )
    },
}

# Case names excluded from the `--all` cross product by default because their
# peak host RSS exceeds the 16 GB RAM available on GitHub-hosted
# `ubuntu-latest` CI runners (measured via `vmhwm_bytes` in a recorded
# meso-tier baseline run): acoustic 3-D meso (accuracy 2 and 4, float32,
# forward_backward) at ~25.8/26.8 GB, and elastic 3-D meso (accuracy 2 and 4,
# float64, forward_backward, already at the SIZE_OVERRIDES-reduced 96^3/nt=200
# size above) at ~23.6/24.8 GB. These are excluded for CI budget reasons only,
# not because they are uninteresting -- a maintainer with more RAM can still
# run them by passing an explicit `--propagator`/`--filter`, either of which
# bypasses this default exclusion (see `get_cases`).
CI_MEMORY_EXCLUDED_CASES: frozenset[str] = frozenset(
    {
        "acoustic_3d_acc2_f32_fwdbwd_meso",
        "acoustic_3d_acc4_f32_fwdbwd_meso",
        "elastic_3d_acc2_f64_fwdbwd_meso",
        "elastic_3d_acc4_f64_fwdbwd_meso",
    }
)

TIERS = tuple(TIER_SIZES.keys())


@dataclass(frozen=True)
class Case:
    """One benchmark case (an axis point + tier + a couple of knobs).

    ``shape``/``nt_override`` optionally replace the square tier-derived
    problem size with an explicit (possibly non-square) grid shape and
    time-step count -- used by the multi-shot small-model sweep, whose
    defining regime (many shots on a laterally elongated ~200x500 model)
    the square tiers cannot express. When ``shape`` is set the recorded
    ``tier`` is ignored for sizing and the case name carries the explicit
    shape/nt instead of the tier suffix.

    ``backend`` selects which of the propagator's own backends runs the
    case (see ``BACKENDS``); default ``"compiled"`` is the hand-written
    C/CUDA backend that every other case in this suite implicitly measures.
    """

    propagator: str
    ndim: int
    accuracy: int
    dtype: str
    mode: str
    backend: str = "compiled"
    tier: str = "micro"
    storage_mode: str = "device"
    storage_compression: bool = False
    n_shots: int = 1
    device: str = "cpu"
    shape: tuple[int, ...] | None = None
    nt_override: int | None = None

    def __post_init__(self) -> None:
        """Validate that every field is one of the recognised axis values."""
        if self.shape is not None:
            # JSON round-trips (run.py -> _case_runner.py) deliver lists.
            object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        if self.propagator not in PROPAGATORS:
            raise ValueError(f"Unknown propagator {self.propagator!r}")
        if self.ndim not in NDIMS:
            raise ValueError(f"Unknown ndim {self.ndim!r}")
        if self.accuracy not in ACCURACIES:
            raise ValueError(f"Unknown accuracy {self.accuracy!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r}")
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend {self.backend!r}")
        if self.tier not in TIER_SIZES:
            raise ValueError(f"Unknown tier {self.tier!r}")
        if self.storage_mode not in STORAGE_MODES:
            raise ValueError(f"Unknown storage_mode {self.storage_mode!r}")
        if not isinstance(self.storage_compression, bool):
            raise TypeError(
                f"storage_compression must be a bool, got {self.storage_compression!r}"
            )
        if self.device not in DEVICES:
            raise ValueError(f"Unknown device {self.device!r}")
        if self.propagator in ("elastic_vti", "acoustic_vti") and self.ndim == 1:
            raise ValueError(f"{self.propagator} requires ndim >= 2")
        if self.n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {self.n_shots!r}")
        if self.shape is not None and len(self.shape) != self.ndim:
            raise ValueError(
                f"shape {self.shape!r} has {len(self.shape)} dims, "
                f"expected ndim={self.ndim}"
            )

    @property
    def name(self) -> str:
        """A stable, unique, filesystem/CLI-friendly identifier for this case."""
        dtype_short = "f32" if self.dtype == "float32" else "f64"
        mode_short = "fwd" if self.mode == "forward" else "fwdbwd"
        if self.shape is not None:
            size_part = "x".join(str(s) for s in self.shape) + f"_nt{self.nt}"
        else:
            size_part = self.tier
        base = (
            f"{self.propagator}_{self.ndim}d_acc{self.accuracy}"
            f"_{dtype_short}_{mode_short}_{size_part}"
        )
        if self.n_shots != 1:
            base += f"_shots{self.n_shots}"
        if self.storage_mode != "device":
            base += f"_sm-{self.storage_mode}"
        if self.storage_compression:
            base += "_comp"
        if self.device != "cpu":
            base += f"_{self.device}"
        if self.backend != "compiled":
            base += f"_backend-{self.backend}"
        return base

    def _tier_size(self) -> tuple[int, int]:
        """(grid_size_per_dim, nt) for this case, honouring SIZE_OVERRIDES.

        A ``(propagator, tier, ndim)`` entry in ``SIZE_OVERRIDES`` takes
        precedence over the propagator-independent ``TIER_SIZES``; used to
        shrink memory-heavy propagators (elastic 3-D meso) so their
        forward_backward case fits host RAM. See ``SIZE_OVERRIDES``.
        """
        override = SIZE_OVERRIDES.get((self.propagator, self.tier, self.ndim))
        return override if override is not None else TIER_SIZES[self.tier][self.ndim]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """The full grid shape: explicit ``shape`` if set, else the square tier size."""
        if self.shape is not None:
            return self.shape
        return (self._tier_size()[0],) * self.ndim

    @property
    def size_per_dim(self) -> int:
        """Grid cells per spatial dimension (smallest dim if ``shape`` is set)."""
        if self.shape is not None:
            return min(self.shape)
        return self._tier_size()[0]

    @property
    def nt(self) -> int:
        """Number of (user-facing) time steps for this case's tier/ndim."""
        if self.nt_override is not None:
            return self.nt_override
        return self._tier_size()[1]


def _default_axis_cases(tier: str) -> list[Case]:
    """16-case default subset: scalar+elastic, 2-D+3-D, acc 4+8, fp32, both modes.

    This is exactly the cross product `{scalar, elastic} x {2, 3} x {4, 8}
    x {forward, forward_backward}` at `float32` (2*2*2*2 = 16).
    `acoustic`/`scalar_born`, 1-D, `float64`, and other accuracies are all
    reachable via `--all`, `--propagator`, or `--filter` rather than baked
    into the default so that a plain `run.py` invocation stays small and
    fast.
    """
    return [
        Case(
            propagator=propagator,
            ndim=ndim,
            accuracy=accuracy,
            dtype="float32",
            mode=mode,
            tier=tier,
        )
        for propagator in ("scalar", "elastic")
        for ndim in (2, 3)
        for accuracy in (4, 8)
        for mode in MODES
    ]


def _all_axis_cases(tier: str) -> list[Case]:
    return [
        Case(
            propagator=propagator,
            ndim=ndim,
            accuracy=accuracy,
            dtype=dtype,
            mode=mode,
            tier=tier,
        )
        for propagator in PROPAGATORS
        for ndim in NDIMS
        if not (propagator in ("elastic_vti", "acoustic_vti") and ndim == 1)
        for accuracy in ACCURACIES
        for dtype in DTYPES
        for mode in MODES
    ]


def macro_storage_cases() -> list[Case]:
    """The macro-tier storage_mode sweep: 3-D 256^3, nt=1000, forward_backward."""
    return [
        Case(
            propagator="scalar",
            ndim=3,
            accuracy=4,
            dtype="float32",
            mode="forward_backward",
            tier="macro",
            storage_mode=storage_mode,
        )
        for storage_mode in STORAGE_MODES
    ]


def get_cases(
    tier: str = "micro",
    all_cases: bool = False,
    propagators: Sequence[str] | None = None,
    name_filter: str | None = None,
    device: str = "cpu",
) -> list[Case]:
    """Resolve CLI selection flags into a concrete list of Cases.

    Args:
        tier: size tier to use ("micro", "meso", "macro"). "macro" is only
            reached if the caller explicitly asks for it (opt-in).
        all_cases: if True, use the full propagator/ndim/accuracy/dtype/mode
            cross product instead of the small default subset.
        propagators: if given, keep only cases for these propagator names.
            Also bypasses CI_MEMORY_EXCLUDED_CASES (an explicit propagator
            request overrides the default CI-memory exclusion).
        name_filter: if given, keep only cases whose `.name` contains this
            substring. Also bypasses CI_MEMORY_EXCLUDED_CASES.
        device: "cpu" (default) or "cuda"; applied to every generated case.
    """
    if tier == "macro" and all_cases:
        cases = _all_axis_cases(tier)
    elif tier == "macro":
        # Macro's "default" content is the storage-mode sweep, since a
        # macro-tier forward-only cross product would be extremely heavy
        # and isn't what the tier was designed for: macro is meant to
        # exercise the storage-mode cost tradeoff at 3-D 256^3 scale,
        # forward_backward per storage_mode, not a full axis sweep.
        cases = macro_storage_cases()
    else:
        cases = _all_axis_cases(tier) if all_cases else _default_axis_cases(tier)

    if not propagators and not name_filter:
        cases = [c for c in cases if c.name not in CI_MEMORY_EXCLUDED_CASES]
    if propagators:
        propagators_set = set(propagators)
        cases = [c for c in cases if c.propagator in propagators_set]
    if name_filter:
        cases = [c for c in cases if name_filter in c.name]
    if device != "cpu":
        cases = [replace(c, device=device) for c in cases]
    return cases


def with_storage_mode(case: Case, storage_mode: str) -> Case:
    """Return a copy of `case` with a different `storage_mode`."""
    return replace(case, storage_mode=storage_mode)


def with_storage_compression(case: Case, storage_compression: bool) -> Case:
    """Return a copy of `case` with a different `storage_compression`."""
    return replace(case, storage_compression=storage_compression)


def with_backend(case: Case, backend: str) -> Case:
    """Return a copy of `case` with a different `backend`."""
    return replace(case, backend=backend)


def shots_sweep_cases(n_shots_list: Sequence[int] = (1, 20, 40)) -> list[Case]:
    """The many-shots/small-model sweep: scalar 2-D, 200x500 grid, nt=1000.

    The spatial-OpenMP kernels parallelise each time step across the grid,
    so a small model gives each thread little work per fork/join while many
    concurrent shots would previously have been the parallelism source.
    This sweep exists to measure exactly that regime (all host threads
    occupied via shots, small per-shot grids) -- the square tiers never
    exercise it. Memory stays modest by design (scalar only, 2-D only):
    the heaviest case (forward_backward, 40 shots) stores
    ~nt * n_shots * grid * 4 B ~= 19 GB of forward snapshots; run.py's
    --mem-cap still guards it.
    """
    return [
        Case(
            propagator="scalar",
            ndim=2,
            accuracy=4,
            dtype="float32",
            mode=mode,
            tier="meso",
            n_shots=n_shots,
            shape=(200, 500),
            nt_override=1000,
        )
        for n_shots in n_shots_list
        for mode in MODES
    ]

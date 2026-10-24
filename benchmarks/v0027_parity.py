#!/usr/bin/env python
"""Numerical parity gate: this Deepwave against the released v0.0.27.

This gate checks whether selected calls written against v0.0.27 still agree
within the scoped numerical criteria below, or differ by a documented break
or intentional change. It runs the same v0.0.27-style calls under two Python
interpreters -- this one, and one named by ``--reference-python`` that has
``deepwave`` v0.0.27 installed -- and classifies every difference.

Elastic comparisons pin one shared conservative ``max_vel``, resolved by
the current version over both encoded input precisions before either worker
runs. This pins the speed-dependent timestep and PML settings; intentional
``pml_freq`` migration cases retain their documented frequency differences.
It does not
claim parity of automatic defaults: the strengthened heterogeneous stability
bound can change both subdivision and PML damping (see docs/upgrading.rst).

Usage
-----
::

    v0027_parity.py run --reference-python PATH [--out OUT.json] [--md OUT.md]
                        [--only REGEX] [--no-cuda] [--repeats N]
                        [--cuda-device cuda:0] [--quick] [--keep-tensors]
    v0027_parity.py list [--only REGEX] [--coverage]
    v0027_parity.py report IN.json [--md OUT.md] [--coverage]

``run`` captures and classifies in one go, writing a JSON record and a
Markdown report; ``report`` re-renders and re-classifies a saved JSON without
re-running anything; ``list`` prints the design without running it.

Exit status of ``run``/``report``: ``0`` if every case was classified as
expected; ``1`` if every difference was expected but at least one of them is
not yet reported at call time; ``2`` if any case differs in a way this gate
does not explain, or an expected difference failed to appear -- which would
mean the tolerance is too loose to see it.

The standard
------------
Each case records the forward outputs -- the receiver amplitudes and the
physical final-state fields -- and the gradients of every model tensor, of
the source amplitudes and of every initial wavefield, and compares them
quantity by quantity. Let ``R`` be this version's float64 result for the same
call on the same device, and ``R027`` v0.0.27's.

(i) A rounding-level match for the tested configuration:

    * float32: ``max|new-v027| <= 2*max(max|new-R|, max|v027-R027|)``, that
      is, twice the larger of the two versions' own rounding errors. Each
      version is measured against its own float64 result, because measuring
      both against one version's float64 result would let the bound satisfy
      itself: a real difference between the versions would show up in that
      term as well, doubling the tolerance it is being tested against. Both
      quantities are recorded, so either reading can be read off the JSON.
    * float64: ``max|new-v027| <= 1e-10*max|R|``, or ten times v0.0.27's own
      measured run-to-run spread if that is larger.

(ii) Otherwise the call hits a documented break: something v0.0.27 accepted
     that this version refuses. ``docs/upgrading.rst`` lists them all.

(iii) Otherwise it is a documented change that alters v0.0.27's numbers on
      purpose, and it is reported at call time.

Anything else breaks the promise. :data:`CHANGES` and :data:`BREAK_PROBES`
name every documented change and break this design triggers, and
:func:`expected_clause` says which case each one excuses. The excuse is
bounded in three ways, because an unbounded one would let a real regression
hide behind it: the difference must actually appear (a case expected to
differ that does not fails, since the tolerance has then grown too loose to
see it), it must be of the size the documented change produces (each entry
carries a relative band, and a quantity outside it fails), and it must be in
a quantity the change can reach. A refusal, likewise, only counts as a
documented break if it is the documented refusal. A case that differs with
no entry at all fails outright.

Both interpreters run with their thread count pinned to
:data:`_PINNED_THREADS`: some kernels' summation order, and so their
rounding, depends on the thread count, and a comparison made at two
different thread counts would not be a comparison of the two versions.

The reduced design
------------------
The axes are the propagator, the number of dimensions, the finite-difference
accuracy, the dtype, the device, the location kind, the call form, the
storage location, whether ``pml_freq`` is given, and the boundary geometry.
Their full cross product is several hundred thousand calls. The design is
reduced instead, in blocks with separate justifications.

**The kernel block is a full factorial, by design.** The compiled kernels are
instantiated once per propagator, per number of dimensions, per accuracy, per
dtype and per device: five axes that each select a different compiled
function, sharing no arithmetic with their neighbours. Nothing about one
point of that grid can be inferred from another, so every supported
``(propagator, ndim, accuracy, dtype, device)`` point is run -- at integer
locations, in function form, with ``pml_freq`` given, device storage and a
uniform PML.

**The call-shape block is a covering design.** The remaining axes are decided
in Python, above the kernel: each either rewrites the call's arguments before
any kernel sees them (location kind, ``pml_freq``, call form) or changes only
where a tensor is kept between the forward and backward passes (storage).
None of them selects a different kernel, so crossing them with the kernel
block would re-run the same arithmetic. Each is crossed instead with the one
axis it can interact with, and its cases are spread over distinct kernel
points so that no case duplicates another's ``(propagator, ndim, dtype,
device)``:

* **location kind** x **propagator**: crossed. Each propagator injects and
  records through its own channels, so the interpolated amplitudes reach
  different kernels. Spread over 1-D, 2-D and 3-D, both dtypes and both
  devices. (v0.0.27's ``Hicks`` is 2-D only, so the Hicks cases are 2-D.)
* **call form** (Module) x **propagator**: crossed; the Module form is a
  per-propagator class holding the models as parameters, so the gradient
  path differs per propagator.
* **storage** (``cpu``, ``disk``) x **propagator**: covered, not crossed.
  Storage moves the checkpointed forward wavefield off the device and back;
  it can interact with a propagator's state size but not with its
  arithmetic.
* **omitted pml_freq** x **source frequency** (10, 15, 25 and 40 Hz):
  crossed, because the value this version infers is a function of the
  source spectrum alone. Spread over propagators, dtypes and devices.
* **boundary geometry**: the zero-width-face idiom, and a vacuum layer and
  an air layer over an elastic model, each at the four ``pml_width``
  spellings that open or close the vertical axis.

A last block of four probes covers clause (ii): calls v0.0.27 accepted whose
spelling this version refuses.

Every level of every axis appears at least once; ``list --coverage`` prints
the tally that proves it, computed from the design rather than asserted.

Self-containment
----------------
The two interpreters run two different versions of ``deepwave``, so this
script builds its own problems and imports nothing from the rest of the
``benchmarks`` package: that code is written against this version's API and
would not run under v0.0.27. The only surfaces used here are ones both
versions share -- the propagator functions and classes,
``deepwave.wavelets.ricker``, ``deepwave.location_interpolation.Hicks`` and
the returned tuple.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# --------------------------------------------------------------------------
# Thread pinning -- must happen before `import torch`, since torch reads
# these environment variables when it first initialises its thread pools.
# --------------------------------------------------------------------------

#: Both interpreters run at this thread count. Some kernels' summation order,
#: and hence their rounding, depends on it, so the two versions are only
#: comparable if both are pinned to the same one. If the caller has already
#: set one of these variables to something else, refuse to run rather than
#: silently overriding their choice.
_PINNED_THREADS = 8
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _pin_threads() -> None:
    """Pin every thread-count environment variable, or refuse to run."""
    for var in _THREAD_ENV_VARS:
        existing = os.environ.get(var)
        if existing is None:
            os.environ[var] = str(_PINNED_THREADS)
        elif existing != str(_PINNED_THREADS):
            raise SystemExit(
                f"{var}={existing!r} is already set in the environment, but "
                f"this gate pins every thread-count environment variable to "
                f"{_PINNED_THREADS} before torch starts, so that the two "
                f"versions are compared at an equal thread count. Unset "
                f"{var}, or set it to {_PINNED_THREADS}, before running."
            )


if __name__ == "__main__":
    _pin_threads()

import torch  # noqa: E402 - must follow the thread-pinning block above

if __name__ == "__main__":
    torch.set_num_threads(_PINNED_THREADS)


# --------------------------------------------------------------------------
# Problem constants
# --------------------------------------------------------------------------

#: The propagators v0.0.27 shipped. They all support 1, 2 and 3 dimensions.
PROPAGATORS: tuple[str, ...] = ("scalar", "scalar_born", "acoustic", "elastic")

#: Interior grid shape, time steps and PML width per number of dimensions.
#: Small enough for the whole design to run in minutes, large enough that the
#: wave crosses the grid and reaches the absorbing boundary, which is where
#: two versions' arithmetic would diverge if it were going to.
_SHAPE: dict[int, tuple[int, ...]] = {1: (320,), 2: (48, 60), 3: (24, 24, 28)}
_NT: dict[int, int] = {1: 300, 2: 300, 3: 150}
_PML: dict[int, int] = {1: 10, 2: 10, 3: 6}

_DX = 5.0
_DT = 0.0005
_FREQ = 25.0
_N_SHOTS = 2
_N_RECEIVERS = 6

#: Model values, built in float64 from a fixed seed and cast down, so that a
#: float32 case and its float64 reference see the same numbers.
_V_MIN, _V_SPAN = 1500.0, 500.0
_RHO_MIN, _RHO_SPAN = 1000.0, 500.0
_DV_SCALE = 40.0
_SEED = 20270027

#: Initial wavefields are given a small smooth non-zero field so that their
#: gradients are not compared all-zero against all-zero.
_STATE_SCALE = 1e-3

#: The boundary-geometry block: a uniform elastic model under a three-row
#: vacuum or air layer, the configuration in which a C-PML waveguide can
#: fail to decay. 15 Hz source six cells down, PML width 10.
_GEOM_SHAPE = (60, 80)
_GEOM_NT = 400
_GEOM_DT = 0.0005
_GEOM_DX = 5.0
_GEOM_FREQ = 15.0
_GEOM_VP, _GEOM_VS, _GEOM_RHO = 2000.0, 1000.0, 2000.0
_GEOM_LAYER_ROWS = 3
_GEOM_SOURCE_DEPTH = 6
_AIR_VP, _AIR_RHO = 340.0, 1.2
#: The four ``pml_width`` spellings, with and without a PML over the layer
#: and under the model.
_GEOM_PML_WIDTHS: tuple[tuple[int, ...], ...] = (
    (0, 0, 10, 10),
    (10, 0, 10, 10),
    (0, 10, 10, 10),
    (10, 10, 10, 10),
)

#: The ``pml_freq`` block reuses the setup in which the inferred value's
#: effect was measured: a 60x60 random model, dx 5 m, dt 2 ms, 400 steps.
_PMLFREQ_SHAPE = (60, 60)
_PMLFREQ_NT = 400
_PMLFREQ_DT = 0.002
_PMLFREQ_FREQS: tuple[float, ...] = (10.0, 15.0, 25.0, 40.0)

#: v0.0.27's default ``Hicks`` half-width.
_HICKS_HALFWIDTH = 4

#: An off-node coordinate inside the band v0.0.27 accepted: it took any
#: coordinate whose fractional part was at most 0.1 and truncated it, and
#: raised on larger fractions. This version interpolates it instead.
_OFF_NODE_FRACTION = 0.05

#: A coordinate whose ``Hicks`` stencil v0.0.27 placed one cell wide,
#: because it built the stencil's integer cell indices in float32 as
#: ``(location + x).long()`` and the sum landed a fraction of a ULP below an
#: integer. Verified against the reference interpreter by the gate itself:
#: the ``hicks_stencil`` case fails if the two versions' stencils agree here.
_HICKS_SHIFTED_COORD = 5.6

#: The tolerance constants of clause (i).
_F32_ROUNDING_FACTOR = 2.0
_F64_RELATIVE = 1e-10
_F64_SPREAD_FACTOR = 10.0

#: Repeats used to measure a version's own run-to-run spread.
_DEFAULT_REPEATS = 3


# --------------------------------------------------------------------------
# Output layout
# --------------------------------------------------------------------------
#
# v0.0.27 returns a plain tuple: every wavefield, then every receiver
# channel. This version returns the same quantities under the same names but
# folds the absorbing boundary's auxiliary fields into one opaque element, so
# the tuples have different lengths and the comparison goes by name: this
# version's output is read by name, and v0.0.27's by the positions below.
#
# The positions were derived by calling every (propagator, ndim) under both
# interpreters with a different receiver count per channel, so that each
# output tensor is identifiable by shape, and cross-checked against the tuple
# length each version returns. Every case asserts that the tuple it got has
# the length recorded here, so a layout that ever moves fails loudly instead
# of silently comparing the wrong pair of tensors.
#
# Only the physical fields are compared. The absorbing boundary's auxiliary
# fields are an implementation detail of the boundary rather than part of the
# state a caller is given -- this version keeps them in one opaque object --
# so they are neither named nor compared here.


def _dims(ndim: int) -> tuple[str, ...]:
    """The axis letters of an ``ndim``-dimensional problem, slowest first."""
    return ("z", "y", "x")[3 - ndim :]


def _sigma_names(ndim: int) -> tuple[str, ...]:
    """Elastic stress fields, in the order v0.0.27 returns them."""
    dims = _dims(ndim)
    return tuple(f"sigma{i}{j}_0" for k, j in enumerate(dims) for i in dims[k:])


def _v027_wavefield_count(prop: str, ndim: int) -> int:
    """How many wavefields v0.0.27 returns before the receiver block."""
    if prop == "scalar":
        return 2 + 2 * ndim
    if prop == "scalar_born":
        return 2 * (2 + 2 * ndim)
    if prop == "acoustic":
        return 1 + 3 * ndim
    # elastic: velocities, stresses, then one PML memory field per velocity
    # derivative and per stress derivative.
    return ndim + ndim * (ndim + 1) // 2 + 2 * ndim * ndim


def physical_state_names(prop: str, ndim: int) -> tuple[str, ...]:
    """The physical final-state fields this gate compares."""
    dims = _dims(ndim)
    if prop == "scalar":
        return ("wavefield_0", "wavefield_m1")
    if prop == "scalar_born":
        return (
            "wavefield_0",
            "wavefield_m1",
            "wavefield_sc_0",
            "wavefield_sc_m1",
        )
    if prop == "acoustic":
        return ("pressure_0", *(f"v{d}_0" for d in dims))
    return (*(f"v{d}_0" for d in dims), *_sigma_names(ndim))


def _v027_state_positions(prop: str, ndim: int) -> dict[str, int]:
    """Where each physical field sits in v0.0.27's returned tuple."""
    dims = _dims(ndim)
    if prop == "scalar":
        return {"wavefield_0": 0, "wavefield_m1": 1}
    if prop == "scalar_born":
        half = 2 + 2 * ndim
        return {
            "wavefield_0": 0,
            "wavefield_m1": 1,
            "wavefield_sc_0": half,
            "wavefield_sc_m1": half + 1,
        }
    if prop == "acoustic":
        return {"pressure_0": 0, **{f"v{d}_0": 1 + i for i, d in enumerate(dims)}}
    positions = {f"v{d}_0": i for i, d in enumerate(dims)}
    positions.update({name: ndim + i for i, name in enumerate(_sigma_names(ndim))})
    return positions


def receiver_names(prop: str, ndim: int) -> tuple[str, ...]:
    """The receiver channels each propagator returns, in tuple order."""
    if prop == "scalar":
        return ("receiver_amplitudes",)
    if prop == "scalar_born":
        return ("bg_receiver_amplitudes", "sc_receiver_amplitudes")
    return ("receiver_amplitudes_p", *(f"receiver_amplitudes_{d}" for d in _dims(ndim)))


def source_channels(prop: str, ndim: int) -> tuple[str, ...]:
    """The source channels this gate drives, one amplitude tensor each.

    ``p`` rather than ``volume_rate``/``pressure_rate`` deliberately: the
    same keywords are passed to both interpreters, so they have to be the
    ones ``v0.0.27`` accepts, and the point of this gate is that a call
    written against ``v0.0.27`` still gets ``v0.0.27``'s numbers. The
    deprecation warning it now raises is not a result, so it does not affect
    the comparison.
    """
    if prop in ("scalar", "scalar_born"):
        return ("",)
    return ("p", *_dims(ndim))


# --------------------------------------------------------------------------
# The design
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One v0.0.27-style call, run under both interpreters."""

    name: str
    block: str
    prop: str
    ndim: int
    accuracy: int
    dtype: str  # float32 | float64
    device: str  # cpu | cuda
    locations: str = "integer"  # integer|off_node|hicks|hicks_stencil|hicks_dtype
    form: str = "function"  # function | module
    storage: str = "device"  # device | cpu | disk
    pml_freq: str = "explicit"  # explicit | omitted
    source_freq: float = _FREQ
    geometry: str = "pml"  # pml | zero_width | vacuum_layer | air_layer
    pml_width: tuple[int, ...] | None = None
    probe: str = ""  # non-empty for the documented-break probes
    repeats: int = 1
    max_vel: float | None = None  # shared elastic discretisation, resolved once

    def key(self) -> str:
        """Stable identifier, used as the JSON key."""
        return self.name


def _case_name(case: Case) -> str:
    parts = [case.block, case.prop, f"{case.ndim}d", f"a{case.accuracy}"]
    parts.append("f32" if case.dtype == "float32" else "f64")
    parts.append(case.device)
    if case.locations != "integer":
        parts.append(case.locations)
    if case.form != "function":
        parts.append(case.form)
    if case.storage != "device":
        parts.append(f"storage_{case.storage}")
    if case.pml_freq != "explicit":
        parts.append(f"infer{case.source_freq:g}hz")
    if case.geometry != "pml":
        parts.append(case.geometry)
    if case.pml_width is not None:
        parts.append("pw" + "".join(str(w) for w in case.pml_width))
    if case.probe:
        parts.append(case.probe)
    return "_".join(parts)


def _named(case: Case) -> Case:
    return replace(case, name=_case_name(case))


def _repeats_for(device: str, accuracy: int, repeats: int) -> int:
    """Which cases get repeated to measure each version's own spread.

    Every CUDA case, because gradients on CUDA are known to differ from run
    to run in both versions, and clause (i)'s float64 tolerance is stated in
    terms of that spread. On the CPU, the accuracy-4 cases -- one per
    propagator, dimension, dtype and device -- are repeated too, so that the
    "repeats itself exactly" claim for CPU runs is measured here rather than
    assumed.
    """
    if device == "cuda" or accuracy == 4:
        return repeats
    return 1


def _kernel_block(repeats: int, accuracies: Sequence[int]) -> list[Case]:
    """Full factorial over the five axes that select a compiled kernel."""
    grid = itertools.product(
        PROPAGATORS, (1, 2, 3), accuracies, ("float32", "float64"), ("cpu", "cuda")
    )
    return [
        _named(
            Case(
                name="",
                block="kernel",
                prop=prop,
                ndim=ndim,
                accuracy=accuracy,
                dtype=dtype,
                device=device,
                repeats=_repeats_for(device, accuracy, repeats),
            )
        )
        for prop, ndim, accuracy, dtype, device in grid
    ]


def _rotate(values: Sequence[Any], index: int) -> Any:
    return values[index % len(values)]


def _locations_block(repeats: int) -> list[Case]:
    """Off-node and Hicks locations, crossed with the propagator.

    Off-node coordinates are spread over 1-D, 2-D and 3-D and over both
    dtypes and devices; the Hicks cases are 2-D because v0.0.27's ``Hicks``
    is 2-D only. Two further Hicks cases isolate the two documented
    changes in the interpolation itself: a stencil whose integer indices
    v0.0.27 computed in float32, and weights whose dtype used to be pinned
    to float32 whatever the default dtype.
    """
    cases: list[Case] = []
    ndims = (2, 3, 1, 2)
    for i, prop in enumerate(PROPAGATORS):
        cases.append(
            _named(
                Case(
                    name="",
                    block="locations",
                    prop=prop,
                    ndim=ndims[i],
                    accuracy=_rotate((4, 8, 2, 4), i),
                    dtype=_rotate(("float64", "float32"), i),
                    device=_rotate(("cpu", "cuda"), i),
                    locations="off_node",
                    repeats=_repeats_for("cpu", 4, repeats),
                )
            )
        )
    for i, prop in enumerate(PROPAGATORS):
        cases.append(
            _named(
                Case(
                    name="",
                    block="locations",
                    prop=prop,
                    ndim=2,
                    accuracy=_rotate((8, 2, 4, 8), i),
                    dtype=_rotate(("float32", "float64"), i),
                    device=_rotate(("cuda", "cpu"), i),
                    locations="hicks",
                    repeats=1,
                )
            )
        )
    for kind, dtype in (("hicks_stencil", "float64"), ("hicks_dtype", "float64")):
        cases.append(
            _named(
                Case(
                    name="",
                    block="locations",
                    prop="scalar",
                    ndim=2,
                    accuracy=4,
                    dtype=dtype,
                    device="cpu",
                    locations=kind,
                )
            )
        )
    return cases


def _form_block() -> list[Case]:
    """The Module form, once per propagator."""
    return [
        _named(
            Case(
                name="",
                block="form",
                prop=prop,
                ndim=_rotate((3, 1, 2, 2), i),
                accuracy=_rotate((2, 4, 8, 2), i),
                dtype=_rotate(("float32", "float64"), i + 1),
                device=_rotate(("cuda", "cpu"), i + 1),
                form="module",
            )
        )
        for i, prop in enumerate(PROPAGATORS)
    ]


def _storage_block() -> list[Case]:
    """``cpu`` and ``disk`` storage of the checkpointed forward wavefield.

    Both ``cpu`` cases run on CUDA on purpose: storing the forward wavefield
    on the host only moves anything when the run is not on the host already,
    so a ``cpu`` storage case on a CPU run would exercise nothing. ``disk``
    is meaningful on either, and runs once on each.
    """
    return [
        _named(
            Case(
                name="",
                block="storage",
                prop=prop,
                ndim=2,
                accuracy=4,
                dtype=dtype,
                device=device,
                storage=storage,
            )
        )
        for prop, storage, device, dtype in (
            ("scalar", "cpu", "cuda", "float32"),
            ("acoustic", "cpu", "cuda", "float64"),
            ("elastic", "disk", "cuda", "float32"),
            ("scalar_born", "disk", "cpu", "float64"),
        )
    ]


def _pml_freq_block(repeats: int) -> list[Case]:
    """``pml_freq`` omitted, at four source frequencies.

    v0.0.27 always used 25 Hz here and said so; this version infers the
    value from the source spectrum. The first four cases reuse the setup in
    which that difference was measured, so their magnitudes are comparable
    with the recorded ones; the second four spread the same axis over the
    other propagators, dtypes and devices.
    """
    cases = [
        _named(
            Case(
                name="",
                block="pmlfreq",
                prop="scalar",
                ndim=2,
                accuracy=4,
                dtype="float64",
                device="cpu",
                pml_freq="omitted",
                source_freq=freq,
                repeats=repeats,
            )
        )
        for freq in _PMLFREQ_FREQS
    ]
    spread = (
        ("scalar_born", "float32", "cpu", 15.0, 8),
        ("acoustic", "float32", "cuda", 40.0, 2),
        ("elastic", "float64", "cuda", 10.0, 4),
        ("scalar", "float32", "cuda", 25.0, 8),
    )
    for prop, dtype, device, freq, accuracy in spread:
        cases.append(
            _named(
                Case(
                    name="",
                    block="pmlfreq",
                    prop=prop,
                    ndim=2,
                    accuracy=accuracy,
                    dtype=dtype,
                    device=device,
                    pml_freq="omitted",
                    source_freq=freq,
                    repeats=_repeats_for(device, accuracy, repeats),
                )
            )
        )
    return cases


def _geometry_block() -> list[Case]:
    """Zero-width faces, and the vacuum and air layers over an elastic model.

    ``pml_width=0`` on a face is v0.0.27's idiom for closing it. The vacuum
    and air layers are the configurations in which a C-PML waveguide can
    stop decaying; each runs at the four ``pml_width`` spellings that open
    or close the vertical axis, with the default boundary, which is the only
    way v0.0.27 could express them.
    """
    cases = [
        _named(
            Case(
                name="",
                block="geometry",
                prop="scalar",
                ndim=2,
                accuracy=4,
                dtype="float64",
                device="cpu",
                geometry="zero_width",
                pml_width=(0, 0, 0, 0),
            )
        ),
        _named(
            Case(
                name="",
                block="geometry",
                prop="elastic",
                ndim=2,
                accuracy=4,
                dtype="float32",
                device="cuda",
                geometry="zero_width",
                pml_width=(0, 10, 10, 10),
            )
        ),
    ]
    cases.extend(
        _named(
            Case(
                name="",
                block="geometry",
                prop="elastic",
                ndim=2,
                accuracy=4,
                dtype="float64",
                device="cpu",
                source_freq=_GEOM_FREQ,
                geometry=layer,
                pml_width=pml_width,
            )
        )
        for layer in ("vacuum_layer", "air_layer")
        for pml_width in _GEOM_PML_WIDTHS
    )
    return cases


@dataclass(frozen=True)
class Break:
    """A spelling v0.0.27 accepted that this version refuses."""

    #: What changed, as ``docs/upgrading.rst`` describes it.
    description: str
    #: A pattern the refusal must match. Without it, the probe would accept
    #: *any* failure as the documented break -- including a typo in the probe
    #: itself, or an unrelated error that happened to be raised first.
    error_pattern: str


#: Clause (ii) probes, each naming the break ``docs/upgrading.rst`` documents.
BREAK_PROBES: dict[str, Break] = {
    "positional_scatter": Break(
        "scalar_born's perturbation is a keyword argument",
        r"takes 3 positional arguments",
    ),
    "positional_survey": Break(
        "arguments after dt are keyword-only",
        r"takes 3 positional arguments",
    ),
    "tuple_length": Break(
        "the returned tuple is shorter",
        r"values to unpack",
    ),
    "pml_initial_state": Break(
        "the absorbing boundary's initial values move into initial_state",
        r"unexpected keyword argument '(psi|zeta|phi|m_)",
    ),
}


def _break_block() -> list[Case]:
    return [
        _named(
            Case(
                name="",
                block="break",
                prop="scalar_born" if probe == "positional_scatter" else "scalar",
                ndim=2,
                accuracy=4,
                dtype="float64",
                device="cpu",
                probe=probe,
            )
        )
        for probe in BREAK_PROBES
    ]


def build_design(repeats: int = _DEFAULT_REPEATS, quick: bool = False) -> list[Case]:
    """The whole reduced design, in run order."""
    accuracies: tuple[int, ...] = (4,) if quick else (2, 4, 6, 8)
    cases = _kernel_block(repeats, accuracies)
    cases += _locations_block(repeats)
    cases += _form_block()
    cases += _storage_block()
    cases += _pml_freq_block(repeats)
    cases += _geometry_block()
    cases += _break_block()
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"duplicate case names in the design: {duplicates}")
    return cases


#: The axis levels the design must cover, for ``--coverage``.
COVERAGE_AXES: dict[str, tuple[Any, ...]] = {
    "prop": PROPAGATORS,
    "ndim": (1, 2, 3),
    "accuracy": (2, 4, 6, 8),
    "dtype": ("float32", "float64"),
    "device": ("cpu", "cuda"),
    "locations": ("integer", "off_node", "hicks", "hicks_stencil", "hicks_dtype"),
    "form": ("function", "module"),
    "storage": ("device", "cpu", "disk"),
    "pml_freq": ("explicit", "omitted"),
    "source_freq": _PMLFREQ_FREQS,
    "geometry": ("pml", "zero_width", "vacuum_layer", "air_layer"),
}


def coverage(cases: Sequence[Case]) -> dict[str, dict[str, int]]:
    """How many cases carry each level of each axis."""
    tally: dict[str, dict[str, int]] = {}
    for axis, levels in COVERAGE_AXES.items():
        counts: dict[str, int] = {}
        for level in levels:
            counts[str(level)] = sum(
                1 for c in cases if getattr(c, axis) == level and not c.probe
            )
        tally[axis] = counts
    tally["pml_width"] = {
        str(list(width)): sum(1 for c in cases if c.pml_width == width)
        for width in _GEOM_PML_WIDTHS
    }
    tally["break_probe"] = {
        probe: sum(1 for c in cases if c.probe == probe) for probe in BREAK_PROBES
    }
    return tally


# --------------------------------------------------------------------------
# Expected differences
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """A documented change that alters a v0.0.27 call's numbers on purpose.

    A change is an excuse for a difference, so it has to be a *bounded*
    excuse. Without one, a case on this list would report success no matter
    how far the two versions had drifted, which is exactly the regression
    this gate exists to catch. Each change therefore carries the quantities
    it is allowed to explain and the size the difference is allowed to be.
    """

    #: What changed, and where ``docs/upgrading.rst`` says so.
    description: str
    #: A pattern the call-time report must match, since clause (iii) is only
    #: satisfied by a difference the caller is told about.
    warning_pattern: str
    #: A pattern for the quantity keys this change may explain. A quantity
    #: outside clause (i) whose key does not match is not excused.
    affects: str
    #: Why ``affects`` is drawn where it is.
    affects_reason: str
    #: The relative difference ``max|new-v027| / max|reference|`` a quantity
    #: this change explains is allowed to show. Outside it, the case fails:
    #: too large means something beyond the documented change moved, too
    #: small means the difference has nearly vanished and the excuse no
    #: longer describes what is happening.
    band: tuple[float, float]
    #: Where the band comes from.
    band_basis: str


#: Source placement and PML profile changes can affect each physical state,
#: receiver and gradient that this harness measures. The classifier still
#: checks the exact expected key set and a magnitude band per quantity.
_WHOLE_SIMULATION = (
    "the change alters the wavefield itself, so every receiver, state field "
    "and gradient follows from it"
)
_DYNAMICAL_QUANTITIES = r"^(?:state:|recv:|grad:model:|grad:source:|grad:state:)"
#: Changing only scalar_born's dv padding cannot change its background
#: forward state or background receiver data. Their gradients may still
#: change because the measured loss includes scattered data.
_BORN_EDGE_QUANTITIES = (
    r"^(?:state:wavefield_sc_|recv:sc_receiver_amplitudes|"
    r"grad:model:|grad:source:|grad:state:)"
)

#: The documented changes this design is built to trigger.
CHANGES: dict[str, Change] = {
    "off_node": Change(
        "an off-node coordinate is interpolated, where v0.0.27 truncated it "
        "to the nearest node (docs/upgrading.rst, 'Off-node coordinates are "
        "interpolated, not rounded')",
        r"off[- ]node|interpolat",
        _DYNAMICAL_QUANTITIES,
        _WHOLE_SIMULATION,
        (1e-7, 3.0),
        "a source moved by a fraction of a cell: recorded at 0.09% and 2% of "
        "the receiver amplitudes for two coordinates, and measured here from "
        "2.3e-5 on a receiver to 0.87 on a source gradient, over four "
        "propagators and both devices; the band allows a factor of about 30 "
        "beyond each end",
    ),
    "hicks_stencil": Change(
        "a Hicks stencil's integer cell indices are built in integer "
        "arithmetic, where v0.0.27 built them in float32 and could place one "
        "point a whole cell away (docs/upgrading.rst, 'An off-node Hicks "
        "stencil is placed correctly')",
        # The message calls the stencil an interpolation window, which is
        # what docs/upgrading.rst quotes; either word satisfies clause (iii).
        r"stencil|interpolation window",
        _DYNAMICAL_QUANTITIES,
        _WHOLE_SIMULATION,
        (1e-5, 1.0),
        "one point of an eight-point stencil placed a cell away, so the "
        "effect is a fraction of one interpolation weight; measured here "
        "from 1.5e-3 to 0.16",
    ),
    "hicks_dtype": Change(
        "Hicks weights follow the default dtype, where v0.0.27 always built "
        "them in float32 (docs/upgrading.rst, 'Hicks(dtype=) defaults to the "
        "default dtype, not to float32')",
        r"[Hh]icks.*dtype|dtype.*[Hh]icks",
        _DYNAMICAL_QUANTITIES,
        _WHOLE_SIMULATION,
        (1e-9, 1e-5),
        "float32 weights promoted to float64 cap the interpolation at about "
        "1e-7 relative, which docs/upgrading.rst calls a difference in the "
        "seventh significant figure; measured here from 8.7e-8 to 2.8e-7",
    ),
    "born_edge_scatter": Change(
        "scalar_born extends its scattering model into the absorbing layer "
        "and the finite-difference halo, where v0.0.27 filled them with "
        "zeros (docs/upgrading.rst, 'scalar_born extends dv into the "
        "absorbing layer')",
        r"outermost cells",
        _BORN_EDGE_QUANTITIES,
        "only the scattered forward fields change when dv padding changes; "
        "the background forward fields and receivers are unaffected",
        (1e-8, 3.0),
        "the scattering model's edge cells now fill the padding instead of "
        "zeros, so the difference is the part of the scattered field that "
        "the padding produces; measured here from 1.3e-6 on a model "
        "gradient to 0.73 on a scattered final-state field, which is mostly "
        "inside the padding, over four accuracies, three dimensionalities, "
        "both dtypes and both devices; the band allows a factor of about 100 "
        "below and 4 above",
    ),
    "pml_freq_inference": Change(
        "an omitted pml_freq is inferred from the source spectrum, where "
        "v0.0.27 always used 25 Hz",
        r"pml_freq",
        _DYNAMICAL_QUANTITIES,
        _WHOLE_SIMULATION,
        (1e-6, 10.0),
        "a different absorbing profile: recorded at 0.013%, 0.93% and 1.95% "
        "of the receiver amplitudes for 40, 15 and 10 Hz sources, and "
        "measured here from 2.3e-5 to 1.8, the largest on a final-state "
        "field that the profile change reshapes almost entirely",
    ),
}


def expected_clause(case: Case) -> tuple[str, str]:
    """The clause this case is expected to land in, and the reason's key.

    Returns ``("i", "")`` for a case that must match at rounding level.
    """
    if case.probe:
        return "ii", case.probe
    if case.locations in ("off_node", "hicks_stencil", "hicks_dtype"):
        return "iii", case.locations
    if case.pml_freq == "omitted" and case.source_freq != 25.0:
        return "iii", "pml_freq_inference"
    if case.prop == "scalar_born":
        # Every scalar_born case here has a scattering model that is non-zero
        # everywhere, its outermost cells included, which is exactly the
        # condition that now fills the padding differently.
        return "iii", "born_edge_scatter"
    return "i", ""


def reason_text(clause: str, reason: str) -> str:
    """Human-readable form of an expected clause's reason."""
    if not reason:
        return ""
    if clause == "ii":
        return BREAK_PROBES[reason].description
    return CHANGES[reason].description


# --------------------------------------------------------------------------
# Building one call
# --------------------------------------------------------------------------

#: The models each propagator takes, in positional order.
MODEL_ORDER: dict[str, tuple[str, ...]] = {
    "scalar": ("v",),
    "scalar_born": ("v", "scatter"),
    "acoustic": ("v", "rho"),
    "elastic": ("lamb", "mu", "buoyancy"),
}

#: The Module class of each propagator.
MODULE_NAME: dict[str, str] = {
    "scalar": "Scalar",
    "scalar_born": "ScalarBorn",
    "acoustic": "Acoustic",
    "elastic": "Elastic",
}

#: A model's attribute name on the Module, newest spelling first.
MODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "v": ("v",),
    "scatter": ("dv", "scatter"),
    "rho": ("rho",),
    "lamb": ("lamb",),
    "mu": ("mu",),
    "buoyancy": ("buoyancy",),
}


def _shape_for(case: Case) -> tuple[int, ...]:
    if case.geometry in ("vacuum_layer", "air_layer"):
        return _GEOM_SHAPE
    if case.block == "pmlfreq":
        return _PMLFREQ_SHAPE
    return _SHAPE[case.ndim]


def _timing(case: Case) -> tuple[int, float, float]:
    """``(nt, dt, grid_spacing)`` for this case."""
    if case.geometry in ("vacuum_layer", "air_layer"):
        return _GEOM_NT, _GEOM_DT, _GEOM_DX
    if case.block == "pmlfreq":
        return _PMLFREQ_NT, _PMLFREQ_DT, _DX
    return _NT[case.ndim], _DT, _DX


def _pml_width(case: Case) -> int | list[int]:
    if case.pml_width is not None:
        return list(case.pml_width)
    return _PML[case.ndim]


def _padded_shape(case: Case) -> tuple[int, ...]:
    """The shape of a wavefield: the model, plus the PML on each face."""
    shape = _shape_for(case)
    width = _pml_width(case)
    per_face = [width] * (2 * case.ndim) if isinstance(width, int) else list(width)
    return tuple(
        shape[d] + per_face[2 * d] + per_face[2 * d + 1] for d in range(case.ndim)
    )


def _survey_points(case: Case) -> tuple[list[int], list[int], int, int]:
    """``(source x, receiver x, source depth, receiver depth)`` in cells.

    The last axis carries the spread; the first axis is depth and any middle
    axis sits at the centre. Hicks cases space their receivers by more than
    a stencil width, because the interpolated locations of two receivers
    closer than that would collide, which v0.0.27 rejects.
    """
    shape = _shape_for(case)
    last = shape[-1]
    if case.block == "pmlfreq":
        return [30, 25], [40 + 3 * i for i in range(_N_RECEIVERS)], 30, 30
    if case.geometry in ("vacuum_layer", "air_layer"):
        return (
            [last // 3, last // 2],
            [last // 4 + 6 * i for i in range(_N_RECEIVERS)],
            _GEOM_SOURCE_DEPTH,
            _GEOM_SOURCE_DEPTH + 4,
        )
    step = (
        2 * _HICKS_HALFWIDTH + 1
        if case.locations.startswith("hicks")
        else max(2, (last - 12) // _N_RECEIVERS)
    )
    return (
        [round(0.35 * last), round(0.6 * last)],
        [6 + step * i for i in range(_N_RECEIVERS)],
        6,
        12,
    )


def _points(case: Case, xs: Sequence[int], depth: int) -> list[list[int]]:
    """Grid points at ``depth`` on the first axis and ``xs`` on the last."""
    shape = _shape_for(case)
    if case.ndim == 1:
        return [[int(x)] for x in xs]
    middle = [s // 2 for s in shape[1:-1]]
    return [[depth, *middle, int(x)] for x in xs]


def _locations_tensors(
    case: Case, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source and receiver locations, one source and N receivers per shot."""
    src_x, rec_x, src_depth, rec_depth = _survey_points(case)
    src = torch.tensor([[p] for p in _points(case, src_x, src_depth)], dtype=torch.long)
    rec = torch.tensor(
        [_points(case, rec_x, rec_depth) for _ in range(_N_SHOTS)],
        dtype=torch.long,
    )
    if case.locations == "off_node":
        # A fractional part inside the band v0.0.27 accepted, on the first
        # coordinate, as in the measurement this reproduces.
        src_f = src.to(dtype)
        src_f[..., 0] += _OFF_NODE_FRACTION
        return src_f.to(device), rec.to(device)
    return src.to(device), rec.to(device)


def _model_tensors(case: Case, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The models, built in float64 from a fixed seed and cast down."""
    shape = _shape_for(case)
    if case.geometry in ("vacuum_layer", "air_layer"):
        vp = torch.full(shape, _GEOM_VP, dtype=torch.float64)
        vs = torch.full(shape, _GEOM_VS, dtype=torch.float64)
        rho = torch.full(shape, _GEOM_RHO, dtype=torch.float64)
        lamb = rho * (vp**2 - 2 * vs**2)
        mu = rho * vs**2
        buoyancy = 1.0 / rho
        rows = _GEOM_LAYER_ROWS
        if case.geometry == "vacuum_layer":
            lamb[:rows] = 0.0
            mu[:rows] = 0.0
            buoyancy[:rows] = 0.0
        else:
            lamb[:rows] = _AIR_RHO * _AIR_VP**2
            mu[:rows] = 0.0
            buoyancy[:rows] = 1.0 / _AIR_RHO
        return {
            "lamb": lamb.to(dtype),
            "mu": mu.to(dtype),
            "buoyancy": buoyancy.to(dtype),
        }
    gen = torch.Generator().manual_seed(_SEED)
    unit_v = torch.rand(shape, generator=gen, dtype=torch.float64)
    unit_rho = torch.rand(shape, generator=gen, dtype=torch.float64)
    unit_dv = torch.randn(shape, generator=gen, dtype=torch.float64)
    v = _V_MIN + _V_SPAN * unit_v
    rho = _RHO_MIN + _RHO_SPAN * unit_rho
    if case.prop == "scalar":
        return {"v": v.to(dtype)}
    if case.prop == "scalar_born":
        return {"v": v.to(dtype), "scatter": (_DV_SCALE * unit_dv).to(dtype)}
    if case.prop == "acoustic":
        return {"v": v.to(dtype), "rho": rho.to(dtype)}
    vs = v / 2
    return {
        "lamb": (rho * (v**2 - 2 * vs**2)).to(dtype),
        "mu": (rho * vs**2).to(dtype),
        "buoyancy": (1.0 / rho).to(dtype),
    }


def _initial_state(case: Case, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """A small smooth non-zero initial wavefield for every physical field.

    Zero initial wavefields would make their gradients an all-zero-against-
    all-zero comparison, which passes without testing anything.
    """
    shape = (_N_SHOTS, *_padded_shape(case))
    total = 1
    for size in shape:
        total *= size
    ramp = torch.arange(total, dtype=torch.float64).reshape(shape)
    fields: dict[str, torch.Tensor] = {}
    for i, name in enumerate(physical_state_names(case.prop, case.ndim)):
        value = _STATE_SCALE * torch.sin(ramp / (7.0 + i) + i)
        fields[name] = value.to(dtype)
    return fields


def _hicks_class() -> Any:
    """``Hicks``, under whichever name this version exposes it."""
    import deepwave

    hicks = getattr(deepwave, "Hicks", None)
    if hicks is not None:
        return hicks
    return deepwave.location_interpolation.Hicks


def _ricker(freq: float, nt: int, dt: float, dtype: torch.dtype) -> torch.Tensor:
    """A Ricker wavelet, called the way both versions accept positionally."""
    import deepwave

    return deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(dtype)


@dataclass
class _Call:
    """Everything one run needs, and everything its outputs are read with."""

    kwargs: dict[str, Any]
    models: dict[str, torch.Tensor]
    amplitudes: dict[str, torch.Tensor]
    states: dict[str, torch.Tensor]
    receiver_hicks: Any = None
    warnings_seen: list[str] = field(default_factory=list)


def _build_call(
    case: Case,
    dtype: torch.dtype,
    device: str,
    scratch: Path,
    run_dtype: torch.dtype | None = None,
) -> _Call:
    """Build one v0.0.27-style call: every argument, with gradients enabled.

    ``dtype`` is the case's own dtype and ``run_dtype`` the dtype the call is
    made in. They differ only for the float64 reference of a float32 case,
    which is the same call -- the same numbers, rounded to float32 -- carried
    out in float64, so that the reference measures the arithmetic's rounding
    rather than the inputs'.
    """
    run = dtype if run_dtype is None else run_dtype
    nt, dt, _grid_spacing = _timing(case)
    models = {
        name: tensor.to(run).to(device).requires_grad_(True)
        for name, tensor in _model_tensors(case, dtype).items()
    }
    states = {
        name: tensor.to(run).to(device).requires_grad_(True)
        for name, tensor in _initial_state(case, dtype).items()
    }
    src_locs, rec_locs = _locations_tensors(case, device, dtype)
    if src_locs.is_floating_point():
        src_locs = src_locs.to(run)
    wavelet = _ricker(case.source_freq, nt, dt, dtype).to(run).to(device)
    receiver_hicks = None

    if case.locations.startswith("hicks"):
        # A dyadic offset, exactly representable in float32, so that both
        # versions' stencils cover the same cells -- except in the
        # `hicks_stencil` case, whose coordinate is chosen so they do not.
        src_f = (src_locs.to(dtype) + 0.25).to(run)
        rec_f = (rec_locs.to(dtype) + 0.25).to(run)
        if case.locations == "hicks_stencil":
            src_f[..., -1] = _HICKS_SHIFTED_COORD
        hicks_cls = _hicks_class()
        if case.locations == "hicks_dtype":
            # v0.0.27 always built the weights in float32; this version
            # follows the default dtype. Everything else is built before the
            # default changes, so the weights are the only difference.
            previous = torch.get_default_dtype()
            torch.set_default_dtype(run)
            try:
                source_hicks = hicks_cls(src_f, _HICKS_HALFWIDTH)
                receiver_hicks = hicks_cls(rec_f, _HICKS_HALFWIDTH)
            finally:
                torch.set_default_dtype(previous)
        else:
            source_hicks = hicks_cls(src_f, _HICKS_HALFWIDTH)
            receiver_hicks = hicks_cls(rec_f, _HICKS_HALFWIDTH)
        src_locs = source_hicks.get_locations()
        rec_locs = receiver_hicks.get_locations()
    else:
        source_hicks = None

    amplitudes: dict[str, torch.Tensor] = {}
    kwargs: dict[str, Any] = {
        "accuracy": case.accuracy,
        "pml_width": _pml_width(case),
    }
    if case.max_vel is not None:
        kwargs["max_vel"] = case.max_vel
    if case.pml_freq == "explicit":
        kwargs["pml_freq"] = _FREQ
    for i, channel in enumerate(source_channels(case.prop, case.ndim)):
        shots = torch.stack(
            [wavelet * (1.0 - 0.2 * s) * (1.0 + 0.1 * i) for s in range(_N_SHOTS)]
        ).unsqueeze(1)
        if source_hicks is not None:
            shots = source_hicks.source(shots)
        shots = shots.detach().requires_grad_(True)
        amplitudes[channel or "u"] = shots
        suffix = f"_{channel}" if channel else ""
        kwargs[f"source_amplitudes{suffix}"] = shots
        kwargs[f"source_locations{suffix}"] = src_locs
    if case.prop == "scalar_born":
        kwargs["bg_receiver_locations"] = rec_locs
        kwargs["receiver_locations"] = rec_locs
    elif case.prop in ("acoustic", "elastic"):
        for channel in ("p", *_dims(case.ndim)):
            kwargs[f"receiver_locations_{channel}"] = rec_locs
    else:
        kwargs["receiver_locations"] = rec_locs
    kwargs.update(states)
    if case.storage != "device":
        kwargs["storage_mode"] = case.storage
        if case.storage == "disk":
            path = scratch / "storage"
            path.mkdir(parents=True, exist_ok=True)
            kwargs["storage_path"] = str(path)
    return _Call(kwargs, models, amplitudes, states, receiver_hicks)


def _resolve_elastic_bounds(cases: Sequence[Case], scratch: Path) -> list[Case]:
    """Pin one bound across versions and each version's own float64 reference.

    Resolve before dispatch, rather than independently in each interpreter:
    otherwise dtype rounding can alter the PML even within an own-reference
    comparison. Planning executes setup without launching native propagation.
    The upward margin covers the encoded input precisions, not a numerical
    comparison tolerance; all existing output/gradient criteria are retained.
    """
    import deepwave

    resolved = []
    for case in cases:
        if case.prop != "elastic" or case.probe or case.max_vel is not None:
            resolved.append(case)
            continue
        bounds = []
        for dtype in (torch.float32, torch.float64):
            call = _build_call(case, dtype, "cpu", scratch)
            _, dt, grid_spacing = _timing(case)
            info = deepwave.plan(
                deepwave.elastic,
                *(call.models[name] for name in MODEL_ORDER["elastic"]),
                grid_spacing,
                dt,
                **call.kwargs,
            ).info
            if info.max_vel is None or not math.isfinite(info.max_vel):
                raise RuntimeError(f"{case.name}: missing finite elastic bound")
            bounds.append(info.max_vel)
        resolved.append(replace(case, max_vel=max(bounds) * 1.000001))
    return resolved


def _storage_kwargs(call: _Call) -> dict[str, Any]:
    """The storage arguments, which the Module form takes at construction."""
    return {
        key: call.kwargs.pop(key)
        for key in ("storage_mode", "storage_path")
        if key in call.kwargs
    }


def _invoke(case: Case, call: _Call) -> tuple[Any, dict[str, torch.Tensor]]:
    """Make the call, in function or Module form; return it and the models.

    The returned model mapping is what the gradients are read from: in
    Module form the models are the Module's own parameters, not the tensors
    passed in.
    """
    import deepwave

    _nt, dt, grid_spacing = _timing(case)
    models = call.models
    if case.form == "function":
        fn = getattr(deepwave, case.prop)
        if case.prop == "scalar_born":
            if "dv" in inspect.signature(fn).parameters:
                return (
                    fn(
                        models["v"],
                        grid_spacing,
                        dt,
                        dv=models["scatter"],
                        **call.kwargs,
                    ),
                    models,
                )
            return (
                fn(
                    models["v"],
                    models["scatter"],
                    grid_spacing,
                    dt,
                    **call.kwargs,
                ),
                models,
            )
        positional = [models[name] for name in MODEL_ORDER[case.prop]]
        return fn(*positional, grid_spacing, dt, **call.kwargs), models

    cls = getattr(deepwave, MODULE_NAME[case.prop])
    storage = _storage_kwargs(call)
    flags = {f"{name}_requires_grad": True for name in MODEL_ORDER[case.prop]}
    detached = {name: tensor.detach() for name, tensor in models.items()}
    if case.prop == "scalar_born":
        if "dv" in inspect.signature(cls.__init__).parameters:
            flags = {"v_requires_grad": True, "dv_requires_grad": True}
            module = cls(
                detached["v"],
                grid_spacing,
                dv=detached["scatter"],
                **flags,
                **storage,
            )
        else:
            module = cls(
                detached["v"],
                detached["scatter"],
                grid_spacing,
                **flags,
                **storage,
            )
    else:
        module = cls(
            *(detached[name] for name in MODEL_ORDER[case.prop]),
            grid_spacing,
            **flags,
            **storage,
        )
    parameters: dict[str, torch.Tensor] = {}
    for name in MODEL_ORDER[case.prop]:
        for attribute in MODULE_ATTRS[name]:
            found = getattr(module, attribute, None)
            if isinstance(found, torch.Tensor):
                parameters[name] = found
                break
        else:
            raise RuntimeError(
                f"{MODULE_NAME[case.prop]} exposes none of "
                f"{MODULE_ATTRS[name]} as a parameter for model {name!r}"
            )
    return module(dt, **call.kwargs), parameters


def _read_outputs(case: Case, out: Any) -> dict[str, torch.Tensor]:
    """The receiver and physical final-state tensors, by name.

    This version's output is read by name; v0.0.27's plain tuple is read by
    position. Both paths refuse to guess: the tuple's length is checked
    against the recorded layout, and a name this version no longer returns
    is an error rather than something to look up by position instead. A
    renamed channel found by counting from the end of the tuple would
    compare the wrong pair of tensors and pass.
    """
    receivers = receiver_names(case.prop, case.ndim)
    as_dict = getattr(out, "as_dict", None)
    if callable(as_dict):
        named = as_dict()
        found: dict[str, torch.Tensor] = {}
        wanted = (
            *physical_state_names(case.prop, case.ndim),
            *receivers,
        )
        for name in wanted:
            if name not in named:
                raise RuntimeError(
                    f"{case.prop} {case.ndim}-D returned no field named "
                    f"{name!r}; it returned {sorted(named)}"
                )
            prefix = "recv" if name in receivers else "state"
            found[f"{prefix}:{name}"] = named[name]
        return found
    expected_len = _v027_wavefield_count(case.prop, case.ndim) + len(receivers)
    if len(out) != expected_len:
        raise RuntimeError(
            f"{case.prop} {case.ndim}-D returned {len(out)} elements, but "
            f"the recorded v0.0.27 layout has {expected_len}"
        )
    positions = _v027_state_positions(case.prop, case.ndim)
    found = {f"state:{name}": out[i] for name, i in positions.items()}
    for i, name in enumerate(receivers):
        found[f"recv:{name}"] = out[len(out) - len(receivers) + i]
    return found


def _recombine(call: _Call, outputs: dict[str, torch.Tensor]) -> None:
    """Apply the receiver side of the Hicks interpolation, in place."""
    if call.receiver_hicks is None:
        return
    for key, tensor in list(outputs.items()):
        if key.startswith("recv:") and tensor.numel() > 0:
            outputs[key] = call.receiver_hicks.receiver(tensor)


# --------------------------------------------------------------------------
# Running one job
# --------------------------------------------------------------------------

#: Which version, and at which precision, one run is made in. Each version
#: has its own float64 reference run, which is what its own rounding error is
#: measured against.
ROLES: tuple[str, ...] = ("current", "reference", "v027", "v027_reference")

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    """A small summary, so the JSON alone shows what was compared."""
    flat = tensor.detach().double().flatten().cpu()
    return {
        "shape": list(tensor.shape),
        "maxabs": float(flat.abs().max()) if flat.numel() else 0.0,
        "sum": float(flat.sum()) if flat.numel() else 0.0,
    }


def _deepwave_version() -> str:
    """The installed version, under whichever name this one exposes it."""
    import deepwave

    version = getattr(deepwave, "__version__", None)
    if version is None:
        version = deepwave._version.__version__  # noqa: SLF001
    return str(version)


def _run_probe(case: Case, scratch: Path) -> dict[str, Any]:
    """One documented-break probe: does the v0.0.27 spelling still work?"""
    import deepwave

    dtype = _DTYPES[case.dtype]
    call = _build_call(case, dtype, case.device, scratch)
    _nt, dt, grid_spacing = _timing(case)
    models = call.models
    kwargs = dict(call.kwargs)
    try:
        # Each probe is deliberately spelled the way v0.0.27 allowed and
        # this version does not, so the type checker is right to object:
        # the probe exists to find out what happens at run time.
        positional: tuple[Any, ...]
        if case.probe == "positional_scatter":
            positional = (models["v"], models["scatter"], grid_spacing, dt)
            deepwave.scalar_born(*positional, **kwargs)  # type: ignore[call-arg,arg-type]
        elif case.probe == "positional_survey":
            positional = (
                models["v"],
                grid_spacing,
                dt,
                kwargs.pop("source_amplitudes"),
                kwargs.pop("source_locations"),
                kwargs.pop("receiver_locations"),
                kwargs["accuracy"],
            )
            deepwave.scalar(*positional)  # type: ignore[call-arg]
        elif case.probe == "tuple_length":
            out = deepwave.scalar(models["v"], grid_spacing, dt, **kwargs)
            (_a, _b, _c, _d, _e, _f, _g) = out
        elif case.probe == "pml_initial_state":
            padded = (_N_SHOTS, *_padded_shape(case))
            kwargs["psiy_m1"] = torch.zeros(padded, dtype=dtype)
            deepwave.scalar(models["v"], grid_spacing, dt, **kwargs)
        else:
            raise RuntimeError(f"unknown probe {case.probe!r}")
    except Exception as exc:  # noqa: BLE001 - the probe's whole purpose
        message = str(exc)[:400]
        # A refusal only counts as the documented break if it is the right
        # refusal. Any other failure -- a mistake in the probe itself, or an
        # unrelated error raised first -- would otherwise look identical.
        return {
            "ok": True,
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": message,
            "expected_error": bool(
                re.search(BREAK_PROBES[case.probe].error_pattern, message)
            ),
        }
    return {"ok": True, "accepted": True, "expected_error": False}


def _rearm_once_per_process_warnings() -> None:
    """Forget which migration warnings this interpreter has already emitted.

    Several of them are deliberately emitted once per cause per process, so
    that a loop does not repeat itself. A worker runs many cases in one
    process, and clause (iii) asks of *each* case whether its own call
    reported the difference, so the record is cleared before every case. Done
    by name and guarded, because the reference interpreter is an older
    Deepwave that has neither set.
    """
    import deepwave

    for module_name in ("_common", "_engine"):
        module = getattr(deepwave, module_name, None)
        causes = getattr(module, "_warned_once_causes", None)
        if causes is not None:
            causes.clear()


def run_job(
    case: Case, role: str, rep: int, scratch: Path, cuda_device: str
) -> dict[str, Any]:
    """Run one case once, save its tensors, and return its record."""
    started = time.perf_counter()
    _rearm_once_per_process_warnings()
    if case.probe:
        result = _run_probe(case, scratch)
        result["seconds"] = time.perf_counter() - started
        result["deepwave_version"] = _deepwave_version()
        return result
    device = cuda_device if case.device == "cuda" else "cpu"
    dtype = _DTYPES[case.dtype]
    run_dtype = torch.float64 if role.endswith("reference") else dtype
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Inside the record, not before it: a Hicks object the caller builds
        # for its own survey is part of the call, and clause (iii) asks
        # whether the caller was told -- not whether the propagator was the
        # one that told them.
        call = _build_call(case, dtype, device, scratch, run_dtype=run_dtype)
        out, models = _invoke(case, call)
        outputs = _read_outputs(case, out)
        _recombine(call, outputs)
        losses = [
            (tensor**2).sum()
            for key, tensor in outputs.items()
            if key.startswith("recv:") and tensor.numel() > 0
        ]
        if not losses:
            raise RuntimeError(f"{case.name}: no receiver output to differentiate")
        total = losses[0]
        for term in losses[1:]:
            total = total + term
        total.backward()  # type: ignore[no-untyped-call]
    if device != "cpu":
        torch.cuda.synchronize(device)
    tensors = {key: value.detach() for key, value in outputs.items()}
    for name, tensor in models.items():
        tensors[f"grad:model:{name}"] = _require_grad(case, f"model {name}", tensor)
    for name, tensor in call.amplitudes.items():
        tensors[f"grad:source:{name}"] = _require_grad(
            case, f"source amplitudes {name}", tensor
        )
    for name, tensor in call.states.items():
        tensors[f"grad:state:{name}"] = _require_grad(
            case, f"initial wavefield {name}", tensor
        )
    saved = {key: value.double().cpu() for key, value in tensors.items()}
    _check_finite(case, role, saved)
    path = scratch / f"{case.name}__{role}__{rep}.pt"
    torch.save(saved, str(path))
    return {
        "ok": True,
        "file": path.name,
        "seconds": time.perf_counter() - started,
        "deepwave_version": _deepwave_version(),
        "torch_version": torch.__version__,
        "warnings": sorted(
            {f"{type(w.message).__name__}: {w.message}"[:300] for w in caught}
        ),
        "fingerprints": {key: _fingerprint(value) for key, value in tensors.items()},
    }


def _check_finite(case: Case, role: str, tensors: dict[str, torch.Tensor]) -> None:
    """Refuse to record a run that produced a NaN or an infinity.

    Failing here, at the source, is the only place where a non-finite value
    is unambiguous. Downstream it becomes a NaN difference, and every
    comparison against a NaN is false -- including the one that decides
    whether a case failed -- so an unguarded NaN would be reported as a
    perfect match.
    """
    bad = sorted(
        key for key, tensor in tensors.items() if not bool(torch.isfinite(tensor).all())
    )
    if bad:
        raise RuntimeError(
            f"{case.name} ({role}): {len(bad)} quantities are not finite: {bad[:6]}"
        )


def _require_grad(case: Case, what: str, tensor: torch.Tensor) -> torch.Tensor:
    """The gradient of a tensor that must have one."""
    if tensor.grad is None:
        raise RuntimeError(f"{case.name}: no gradient reached {what}")
    return tensor.grad.detach()


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a batch of jobs in this interpreter; one JSON object on stdout."""
    scratch = Path(payload["scratch"])
    cuda_device = payload["cuda_device"]
    results: dict[str, Any] = {}
    for job in payload["jobs"]:
        case = Case(**job["case"])
        role = job["role"]
        key = f"{case.name}|{role}|{job['rep']}"
        try:
            record = run_job(case, role, job["rep"], scratch, cuda_device)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            record = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:600],
            }
        results[key] = record
    return {"results": results, "deepwave_version": _deepwave_version()}


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

_BATCH = 12


def _interpreter_facts(python: str) -> dict[str, Any]:
    """Version and CUDA availability of an interpreter, or raise."""
    code = (
        "import json, torch, deepwave\n"
        "v = getattr(deepwave, '__version__', None)\n"
        "if v is None: v = deepwave._version.__version__\n"
        "print(json.dumps({'deepwave': str(v), 'torch': torch.__version__,\n"
        " 'file': deepwave.__file__, 'cuda': torch.cuda.is_available()}))\n"
    )
    proc = subprocess.run(  # noqa: S603 - fixed argv, caller-named interpreter
        [python, "-c", code],
        capture_output=True,
        text=True,
        env=_child_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"could not import deepwave under {python!r}:\n"
            f"{(proc.stderr or '').strip()[-800:]}"
        )
    return dict(json.loads(proc.stdout.strip().splitlines()[-1]))


def _child_env() -> dict[str, str]:
    """The environment every child runs in, with the thread count pinned."""
    env = dict(os.environ)
    for var in _THREAD_ENV_VARS:
        env[var] = str(_PINNED_THREADS)
    return env


def _dispatch(
    python: str,
    jobs: list[dict[str, Any]],
    scratch: Path,
    cuda_device: str,
    label: str,
) -> dict[str, Any]:
    """Run jobs under one interpreter, in batches, isolating any failure."""
    results: dict[str, Any] = {}
    done = 0
    for start in range(0, len(jobs), _BATCH):
        batch = jobs[start : start + _BATCH]
        got = _run_batch(python, batch, scratch, cuda_device)
        if got is None:
            # The batch died as a whole (a crash, not an exception). Re-run
            # its jobs one at a time so that only the guilty one is lost.
            for job in batch:
                single = _run_batch(python, [job], scratch, cuda_device)
                key = f"{job['case']['name']}|{job['role']}|{job['rep']}"
                results[key] = (
                    single[key]
                    if single and key in single
                    else {
                        "ok": False,
                        "error_type": "Crash",
                        "error": "interpreter exited",
                    }
                )
        else:
            results.update(got)
        done += len(batch)
        print(f"  [{label}] {done}/{len(jobs)} runs", flush=True)
    return results


def _run_batch(
    python: str, jobs: list[dict[str, Any]], scratch: Path, cuda_device: str
) -> dict[str, Any] | None:
    payload = json.dumps(
        {"jobs": jobs, "scratch": str(scratch), "cuda_device": cuda_device}
    )
    proc = subprocess.run(  # noqa: S603 - fixed argv, this script's own module
        [python, str(Path(__file__).resolve()), "_worker"],
        input=payload,
        capture_output=True,
        text=True,
        env=_child_env(),
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        sys.stderr.write((proc.stderr or "")[-1500:] + "\n")
        return None
    parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    return dict(parsed["results"])


def _jobs_for(cases: Sequence[Case]) -> dict[str, list[dict[str, Any]]]:
    """The runs each interpreter has to make, by role."""
    jobs: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for case in cases:
        payload = asdict(case)
        reps = 1 if case.probe else case.repeats
        for rep in range(reps):
            jobs["current"].append({"case": payload, "role": "current", "rep": rep})
            jobs["v027"].append({"case": payload, "role": "v027", "rep": rep})
        if not case.probe and case.dtype == "float32":
            # One float64 reference run per version: each version's own
            # rounding error is the difference between its float32 result
            # and its own float64 result for the same call.
            for role in ("reference", "v027_reference"):
                jobs[role].append({"case": payload, "role": role, "rep": 0})
    return jobs


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------


def _load_tensors(scratch: Path, record: dict[str, Any]) -> dict[str, torch.Tensor]:
    return dict(torch.load(str(scratch / record["file"]), weights_only=True))


def _own_spread(scratch: Path, records: list[dict[str, Any]]) -> dict[str, float]:
    """The largest difference between two runs of the same version."""
    loaded = [_load_tensors(scratch, r) for r in records if r.get("file")]
    if len(loaded) < 2:
        return {}
    spread: dict[str, float] = {}
    for key in loaded[0]:
        worst = 0.0
        for i, left in enumerate(loaded):
            for right in loaded[i + 1 :]:
                worst = max(worst, float((left[key] - right[key]).abs().max()))
        spread[key] = worst
    return spread


def _measure(
    case: Case, scratch: Path, runs: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Every number clause (i) needs, per quantity, for one case."""
    current = _load_tensors(scratch, runs["current"][0])
    v027 = _load_tensors(scratch, runs["v027"][0])
    float32 = case.dtype == "float32"
    reference = _load_tensors(scratch, runs["reference"][0]) if float32 else current
    v027_reference = (
        _load_tensors(scratch, runs["v027_reference"][0]) if float32 else v027
    )
    spread_v027 = _own_spread(scratch, runs["v027"])
    spread_current = _own_spread(scratch, runs["current"])
    missing = set(current) ^ set(v027)
    if missing:
        raise RuntimeError(
            f"{case.name}: quantities on one side only: {sorted(missing)}"
        )
    expected = _expected_quantity_keys(case)
    if set(current) != expected:
        raise RuntimeError(
            f"{case.name}: missing {sorted(expected - set(current))}; "
            f"unexpected {sorted(set(current) - expected)} quantities"
        )
    for role, tensors in (("reference", reference), ("v027_reference", v027_reference)):
        if set(tensors) != expected:
            raise RuntimeError(
                f"{case.name}: {role} has missing {sorted(expected - set(tensors))} "
                f"and unexpected {sorted(set(tensors) - expected)} quantities"
            )
    quantities: dict[str, Any] = {}
    for key in sorted(current):
        diff = float((current[key] - v027[key]).abs().max())
        reference_max = float(reference[key].abs().max())
        quantities[key] = {
            "diff": diff,
            # Each version's own rounding error: its float32 result against
            # its own float64 result for the same call.
            "current_rounding": float((current[key] - reference[key]).abs().max()),
            "v027_rounding": float((v027[key] - v027_reference[key]).abs().max()),
            # v0.0.27 against *this* version's float64 result. Recorded for
            # completeness; see `_tolerance` for why it is not the bound.
            "v027_vs_current_reference": float(
                (v027[key] - reference[key]).abs().max()
            ),
            "reference_maxabs": reference_max,
            "v027_spread": spread_v027.get(key),
            "current_spread": spread_current.get(key),
        }
    return quantities


def _tolerance(case_dtype: str, quantity: dict[str, Any]) -> tuple[float, str]:
    """Clause (i)'s tolerance for one quantity, and how it was formed."""
    if case_dtype == "float32":
        # Each version against its own float64 result. Measuring both
        # against one version's float64 result instead would make the bound
        # satisfy itself: a genuine difference would appear in that term too,
        # doubling the tolerance it is being tested against.
        rounding = max(quantity["current_rounding"], quantity["v027_rounding"])
        return _F32_ROUNDING_FACTOR * rounding, "2x each version's own rounding"
    relative = _F64_RELATIVE * quantity["reference_maxabs"]
    spread = quantity.get("v027_spread")
    if spread is not None and _F64_SPREAD_FACTOR * spread > relative:
        return _F64_SPREAD_FACTOR * spread, "10x v0.0.27's own spread"
    return relative, "1e-10 x max|reference|"


#: The numbers every verdict rests on. Any of them arriving non-finite means
#: the comparison cannot be trusted, because every comparison against a NaN
#: is false -- including ``diff > tolerance``, which decides whether a case
#: failed. Runs are already refused at the source if they produce one
#: (:func:`_check_finite`); this is the second line of the same defence, and
#: it also covers a record read back from an older JSON.
_MUST_BE_FINITE = (
    "diff",
    "current_rounding",
    "v027_rounding",
    "reference_maxabs",
)


def _non_finite(quantity: dict[str, Any]) -> list[str]:
    """Which of a quantity's numbers are NaN or infinite."""
    return [
        field
        for field in _MUST_BE_FINITE
        if field in quantity and not math.isfinite(float(quantity[field]))
    ]


def _expected_quantity_keys(case: Case) -> set[str]:
    """Independent expectations for every measured output and gradient."""
    states = physical_state_names(case.prop, case.ndim)
    return {
        *(f"state:{name}" for name in states),
        *(f"recv:{name}" for name in receiver_names(case.prop, case.ndim)),
        *(f"grad:model:{name}" for name in MODEL_ORDER[case.prop]),
        *(
            f"grad:source:{channel or 'u'}"
            for channel in source_channels(case.prop, case.ndim)
        ),
        *(f"grad:state:{name}" for name in states),
    }


def _classify_probe(record: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    """Clause (ii): v0.0.27 accepted the call and this version refuses it."""
    accepted_now = record["accepted_current"]
    accepted_then = record["accepted_v027"]
    if not accepted_then or accepted_now:
        verdict.update(
            clause="i" if accepted_now else "error",
            status="unexpected",
            detail=(
                f"v0.0.27 accepted={accepted_then}, this version "
                f"accepted={accepted_now}"
            ),
        )
        return verdict
    if not record.get("expected_error_current"):
        verdict.update(
            clause="error",
            status="unexpected",
            detail=(
                "this version refused the call, but not with the documented "
                f"refusal: {record.get('error_current', '')[:200]}"
            ),
        )
        return verdict
    verdict.update(clause="ii", status="ok", detail=record.get("error_current", ""))
    return verdict


def _check_band(
    change: Change, record: dict[str, Any], failures: list[str]
) -> tuple[list[str], list[str]]:
    """Which failing quantities this change may not excuse.

    Returns the ones it is not allowed to touch at all, and the ones whose
    size falls outside the band. Both make the case fail: a documented
    change explains a difference of a particular kind and a particular size,
    and anything else hiding behind it is what this gate exists to find.
    """
    unrelated: list[str] = []
    out_of_band: list[str] = []
    low, high = change.band
    for key in failures:
        if not re.search(change.affects, key):
            unrelated.append(key)
            continue
        quantity = record["quantities"][key]
        maximum = quantity["reference_maxabs"]
        relative = quantity["diff"] / maximum if maximum > 0 else float("inf")
        quantity["relative"] = relative
        if not math.isfinite(relative) or not low <= relative <= high:
            out_of_band.append(f"{key} ({relative:.3e})")
    return unrelated, out_of_band


def classify(case: Case, record: dict[str, Any]) -> dict[str, Any]:
    """Give one case a clause, a status and the numbers behind them."""
    expected, reason = expected_clause(case)
    verdict: dict[str, Any] = {
        "expected_clause": expected,
        "expected_reason": reason,
        "expected_explanation": reason_text(expected, reason),
    }
    if record.get("error"):
        verdict.update(clause="error", status="error", detail=record["error"])
        return verdict
    if case.probe:
        return _classify_probe(record, verdict)
    quantities = record.get("quantities")
    expected_keys = _expected_quantity_keys(case)
    if not isinstance(quantities, dict) or set(quantities) != expected_keys:
        found = set(quantities) if isinstance(quantities, dict) else set()
        verdict.update(
            clause="error",
            status="error",
            detail=(
                f"missing quantities {sorted(expected_keys - found)}; "
                f"unexpected quantities {sorted(found - expected_keys)}"
            ),
        )
        return verdict
    worst_key, worst_ratio = "", 0.0
    failures: list[str] = []
    unfinite: list[str] = []
    compared = 0
    for key, quantity in quantities.items():
        required = set(_MUST_BE_FINITE)
        if not isinstance(quantity, dict) or not required <= set(quantity):
            verdict.update(
                clause="error",
                status="error",
                detail=f"{key}: malformed comparison quantity",
            )
            return verdict
        tolerance, basis = _tolerance(case.dtype, quantity)
        quantity["tolerance"] = tolerance
        quantity["tolerance_basis"] = basis
        broken = _non_finite(quantity)
        if broken:
            unfinite.append(f"{key} ({', '.join(broken)})")
            continue
        if quantity["reference_maxabs"] == 0.0 and quantity["diff"] == 0.0:
            quantity["vacuous"] = True
            continue
        compared += 1
        ratio = (
            quantity["diff"] / tolerance
            if tolerance > 0
            else (0.0 if quantity["diff"] == 0.0 else float("inf"))
        )
        if ratio > worst_ratio:
            worst_key, worst_ratio = key, ratio
        if quantity["diff"] > tolerance:
            failures.append(key)
    verdict["quantities_compared"] = compared
    verdict["worst_quantity"] = worst_key
    verdict["worst_ratio"] = worst_ratio
    if unfinite:
        verdict.update(
            clause="error",
            status="non-finite",
            detail=(
                f"{len(unfinite)} quantities have a NaN or infinite "
                f"comparison, which no tolerance can judge: {unfinite[:4]}"
            ),
        )
        return verdict
    if compared == 0:
        verdict.update(
            clause="error",
            status="error",
            detail="every quantity was identically zero, so nothing was compared",
        )
        return verdict
    if not failures:
        verdict["clause"] = "i"
        verdict["status"] = "ok" if expected == "i" else "missing-expected"
        if expected != "i":
            verdict["detail"] = (
                "this difference was expected but did not appear, so the "
                "tolerance is too loose to see it"
            )
        return verdict
    verdict["clause"] = "iii"
    verdict["failing_quantities"] = failures
    if expected != "iii":
        verdict["status"] = "unexplained"
        verdict["detail"] = f"{len(failures)} quantities outside clause (i)"
        return verdict
    change = CHANGES[reason]
    unrelated, out_of_band = _check_band(change, record, failures)
    verdict["band"] = list(change.band)
    if unrelated or out_of_band:
        verdict["status"] = "out-of-band"
        parts = []
        if unrelated:
            parts.append(f"{len(unrelated)} quantities this change cannot touch")
        if out_of_band:
            parts.append(
                f"{len(out_of_band)} outside its {change.band[0]:.0e} to "
                f"{change.band[1]:.0e} relative band: {out_of_band[:3]}"
            )
        verdict["detail"] = (
            "the documented change does not account for this difference -- "
            + "; ".join(parts)
        )
        verdict["unrelated_quantities"] = unrelated
        verdict["out_of_band_quantities"] = out_of_band
        return verdict
    pattern = change.warning_pattern
    reported = [w for w in record.get("warnings_current", []) if re.search(pattern, w)]
    verdict["reported_at_call_time"] = bool(reported)
    verdict["reports"] = reported
    verdict["status"] = "ok" if reported else "pending-warning"
    if not reported:
        verdict["detail"] = (
            "clause (iii) also requires the difference to be reported at "
            "call time, and no such report was made"
        )
    return verdict


#: What each status does to the exit code. ``skipped`` is a failure and
#: ``skipped-by-request`` is not, because they mean different things: the
#: first is coverage this run silently lost -- a CUDA-less machine reporting
#: success while a whole device axis went unrun -- and the second is the
#: caller having said, with ``--no-cuda``, that they did not want it.
_STATUS_EXIT: dict[str, int] = {
    "ok": 0,
    "skipped-by-request": 0,
    "pending-warning": 1,
    "missing-expected": 2,
    "unexplained": 2,
    "unexpected": 2,
    "out-of-band": 2,
    "non-finite": 2,
    "error": 2,
    "skipped": 2,
}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _format_coverage(tally: dict[str, dict[str, int]]) -> list[str]:
    lines = ["| axis | level | cases |", "|---|---|---|"]
    for axis, counts in tally.items():
        for level, count in counts.items():
            flag = "" if count else "  <- NOT COVERED"
            lines.append(f"| {axis} | {level} | {count}{flag} |")
    return lines


def _spread_section(cases: dict[str, Any]) -> list[str]:
    """How much each version differs from itself, run to run.

    Clause (i)'s float64 tolerance is the larger of a relative bound and ten
    times v0.0.27's own spread, so the spread has to be measured rather than
    assumed. A case is only measured if it was run more than once.
    """
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name, entry in cases.items():
        case = entry.get("case", {})
        for key, quantity in entry.get("quantities", {}).items():
            for version in ("v027", "current"):
                spread = quantity.get(f"{version}_spread")
                if spread is None:
                    continue
                kind = "gradients" if key.startswith("grad:") else "outputs"
                group = groups.setdefault(
                    (case["device"], case["dtype"], kind),
                    {
                        "measured": 0,
                        "moving": {"v027": 0, "current": 0},
                        "worst": 0.0,
                        "where": "",
                    },
                )
                if version == "v027":
                    group["measured"] += 1
                if spread <= 0.0:
                    continue
                group["moving"][version] += 1
                reference = quantity["reference_maxabs"]
                relative = spread / reference if reference else float("inf")
                if version == "v027" and relative > group["worst"]:
                    group["worst"] = relative
                    group["where"] = f"{name} / {key}"
    lines = [
        "",
        "## Each version's own run-to-run spread",
        "",
        "Measured by repeating the run and taking the largest difference "
        "between two repeats. Only cases run more than once appear here.",
        "",
        "| device | dtype | kind | quantities measured | v0.0.27 moved "
        "| this version moved | worst v0.0.27 spread, relative | where |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (device, dtype, kind), group in sorted(groups.items()):
        lines.append(
            f"| {device} | {dtype} | {kind} | {group['measured']} | "
            f"{group['moving']['v027']} | {group['moving']['current']} | "
            f"{group['worst']:.2e} | {group['where'] or '-'} |"
        )
    return lines


def _rounding_balance_section(cases: dict[str, Any], limit: int = 10) -> list[str]:
    """Where the two versions' float32 rounding errors are least alike.

    Clause (i)'s float32 bound is twice the larger of the two, so it is
    satisfied by construction whenever the versions' float64 results agree:
    it asks whether each float32 result is close to its own float64 result,
    and whether those two float64 results are the same. It cannot, on its
    own, notice one version being systematically less accurate in float32
    than the other, so the two rounding errors are reported side by side
    here and the most lopsided quantities are listed.
    """
    rows: list[tuple[float, str, str, float, float]] = []
    for name, entry in cases.items():
        if entry.get("case", {}).get("dtype") != "float32":
            continue
        for key, quantity in entry.get("quantities", {}).items():
            mine, theirs = quantity["current_rounding"], quantity["v027_rounding"]
            if min(mine, theirs) <= 0.0:
                continue
            rows.append(
                (max(mine, theirs) / min(mine, theirs), name, key, mine, theirs)
            )
    rows.sort(reverse=True)
    lines = [
        "",
        "## Float32 rounding, the two versions side by side",
        "",
        f"The {limit} quantities whose two rounding errors are least alike. A "
        "ratio near 1 means the versions are equally accurate in float32; a "
        "large one means one of them is losing more, which clause (i) does "
        "not itself test for.",
        "",
        "| ratio | case | quantity | this version | v0.0.27 |",
        "|---|---|---|---|---|",
    ]
    for ratio, name, key, mine, theirs in rows[:limit]:
        lines.append(f"| {ratio:.1f} | `{name}` | {key} | {mine:.3e} | {theirs:.3e} |")
    return lines


def write_report(data: dict[str, Any], path: Path, with_coverage: bool) -> None:
    """Write the Markdown report."""
    meta = data["meta"]
    cases = data["cases"]
    counts: dict[str, int] = {}
    for entry in cases.values():
        counts[entry["verdict"]["status"]] = (
            counts.get(entry["verdict"]["status"], 0) + 1
        )
    lines = [
        "# v0.0.27 parity",
        "",
        f"- this version: `{meta['current_version']}` "
        f"({meta['current_file']}), torch {meta['torch_version']}",
        f"- reference: `{meta['reference_version']}` via `{meta['reference_python']}`",
        f"- threads pinned to {meta['threads']}; CUDA device "
        f"`{meta['cuda_device']}`{'' if meta['cuda'] else ' (unavailable)'}",
        f"- {len(cases)} cases, {meta['runs']} runs, "
        f"{meta['seconds']:.0f} s wall clock",
        "",
        "## Verdicts",
        "",
        "| status | cases | meaning |",
        "|---|---|---|",
    ]
    if meta.get("elastic_settings"):
        lines[9:9] = [
            "Elastic parity uses shared explicit conservative `max_vel` bounds. "
            "It does not establish equality of automatic defaults. Documented "
            "`pml_freq` migration cases retain their intended frequency differences.",
            "",
        ]
    meanings = {
        "ok": "classified as expected",
        "pending-warning": "an expected difference, not yet reported at call time",
        "missing-expected": "an expected difference did not appear",
        "unexplained": "a difference outside clause (i) with no documented cause",
        "unexpected": "a documented break behaved unexpectedly",
        "out-of-band": "an expected difference, but not of the expected size",
        "non-finite": "a NaN or infinity, which no tolerance can judge",
        "error": "the case did not run",
        "skipped": "not run: no CUDA, and --no-cuda was not passed",
        "skipped-by-request": "not run: --no-cuda",
    }
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} | {meanings.get(status, '')} |")
    lines += [
        "",
        "## Cases that differ, or should",
        "",
        "| case | expected | found | worst quantity | max abs diff | tolerance "
        "| reported |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, entry in sorted(cases.items()):
        verdict = entry["verdict"]
        if verdict["expected_clause"] == "i" and verdict.get("clause") == "i":
            continue
        worst = verdict.get("worst_quantity", "")
        quantity = entry.get("quantities", {}).get(worst)
        detail = verdict.get("detail", "")[:90]
        numbers = (
            f"{quantity['diff']:.3e} | {quantity['tolerance']:.3e}"
            if quantity
            else "- | -"
        )
        lines.append(
            f"| `{name}` | ({verdict['expected_clause']}) | "
            f"({verdict.get('clause', '?')}) | {worst or detail} | {numbers} "
            f"| {verdict.get('reported_at_call_time', '-')} |"
        )
    lines += ["", "### Why each is expected", ""]
    seen: set[str] = set()
    for entry in sorted(cases.values(), key=lambda e: e["case"]["name"]):
        verdict = entry["verdict"]
        reason = verdict["expected_reason"]
        if reason and reason not in seen:
            seen.add(reason)
            lines.append(f"- **{reason}**: {verdict['expected_explanation']}")
    lines += _spread_section(cases)
    lines += _rounding_balance_section(cases)
    lines += [
        "",
        "## Every case",
        "",
        "| case | status | clause | worst ratio |",
        "|---|---|---|---|",
    ]
    for name, entry in sorted(cases.items()):
        verdict = entry["verdict"]
        ratio = verdict.get("worst_ratio")
        lines.append(
            f"| `{name}` | {verdict['status']} | ({verdict.get('clause', '-')}) | "
            f"{'-' if ratio is None else f'{ratio:.2f}'} |"
        )
    if with_coverage:
        lines += ["", "## Axis coverage", "", *_format_coverage(data["coverage"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _filter(cases: list[Case], pattern: str | None) -> list[Case]:
    if not pattern:
        return cases
    matcher = re.compile(pattern)
    return [c for c in cases if matcher.search(c.name)]


def _collect(
    case: Case, results: dict[str, dict[str, Any]], scratch: Path
) -> dict[str, Any]:
    """Turn one case's runs into the record :func:`classify` reads."""
    runs: dict[str, list[dict[str, Any]]] = {}
    reps = 1 if case.probe else case.repeats
    for role in ROLES:
        keys = (f"{case.name}|{role}|{rep}" for rep in range(reps))
        runs[role] = [results[role][k] for k in keys if k in results[role]]
    entry: dict[str, Any] = {"case": asdict(case)}
    needed = ["current", "v027"] + (
        ["reference", "v027_reference"]
        if case.dtype == "float32" and not case.probe
        else []
    )
    for role in needed:
        if not runs[role] or not runs[role][0].get("ok"):
            failure = runs[role][0] if runs[role] else {"error": "not run"}
            entry["error"] = f"{role}: {failure.get('error', 'missing')}"
            return entry
    entry["seconds"] = sum(r.get("seconds", 0.0) for role in ROLES for r in runs[role])
    entry["warnings_current"] = runs["current"][0].get("warnings", [])
    entry["warnings_v027"] = runs["v027"][0].get("warnings", [])
    if case.probe:
        entry["accepted_current"] = runs["current"][0]["accepted"]
        entry["expected_error_current"] = runs["current"][0].get(
            "expected_error", False
        )
        entry["accepted_v027"] = runs["v027"][0]["accepted"]
        entry["error_current"] = runs["current"][0].get("error", "")
        entry["error_v027"] = runs["v027"][0].get("error", "")
        return entry
    try:
        entry["quantities"] = _measure(case, scratch, runs)
    except Exception as exc:  # noqa: BLE001 - reported as a failed case
        entry["error"] = f"comparison: {exc}"
    return entry


def cmd_run(args: argparse.Namespace) -> int:
    """Capture both versions, classify every case, write the record."""
    import deepwave

    started = time.time()
    reference_python = shutil.which(args.reference_python) or args.reference_python
    if not Path(reference_python).exists():
        raise SystemExit(
            f"--reference-python {args.reference_python!r} is not an "
            "executable. Point it at the interpreter of an environment with "
            "deepwave v0.0.27 installed."
        )
    reference_facts = _interpreter_facts(reference_python)
    if reference_facts["deepwave"] != "0.0.27":
        raise SystemExit(
            f"--reference-python {reference_python!r} has deepwave "
            f"{reference_facts['deepwave']}, but this gate compares against "
            "v0.0.27. Point it at an environment with deepwave 0.0.27 "
            "installed."
        )
    cases = _filter(build_design(args.repeats, args.quick), args.only)
    if not cases:
        raise SystemExit("no cases selected")
    cuda = (
        torch.cuda.is_available() and bool(reference_facts["cuda"]) and not args.no_cuda
    )
    active = [c for c in cases if cuda or c.device != "cuda"]
    skipped = [c for c in cases if c not in active]
    if skipped:
        why = "--no-cuda" if args.no_cuda else "no CUDA"
        print(f"skipped: {why} ({len(skipped)} CUDA cases)", flush=True)
    scratch = (
        Path(args.scratch)
        if args.scratch
        else Path(tempfile.mkdtemp(prefix="v0027_parity_"))
    )
    scratch.mkdir(parents=True, exist_ok=True)
    active = _resolve_elastic_bounds(active, scratch)
    jobs = _jobs_for(active)
    total_runs = sum(len(j) for j in jobs.values())
    print(
        f"{len(active)} cases, {total_runs} runs, scratch {scratch}",
        flush=True,
    )
    results: dict[str, dict[str, Any]] = {}
    clean = True
    try:
        data: dict[str, Any] = {
            "meta": {
                "current_version": _deepwave_version(),
                "current_file": deepwave.__file__,
                "torch_version": torch.__version__,
                "reference_python": reference_python,
                "reference_version": reference_facts["deepwave"],
                "reference_file": reference_facts["file"],
                "threads": _PINNED_THREADS,
                "cuda": cuda,
                "cuda_device": args.cuda_device,
                "repeats": args.repeats,
                "quick": args.quick,
                "runs": total_runs,
                "elastic_settings": "shared explicit conservative max_vel",
                "capture_group_size": 8,
            },
            # Computed from what actually ran, not from what was asked for:
            # a coverage table that counts cases this run never made would
            # report a covered axis level that nothing tested.
            "coverage": coverage(active),
            "cases": {},
        }
        for offset in range(0, len(active), 8):
            group = active[offset : offset + 8]
            folder = scratch / f"group-{offset:03d}"
            folder.mkdir()  # Refuse to overwrite a previous run's evidence.
            group_jobs = _jobs_for(group)
            for role in ROLES:
                python = reference_python if role.startswith("v027") else sys.executable
                results[role] = _dispatch(
                    python, group_jobs[role], folder, args.cuda_device, role
                )
            group_clean = True
            for case in group:
                entry = _collect(case, results, folder)
                entry["verdict"] = classify(case, entry)
                data["cases"][case.name] = entry
                group_clean &= _STATUS_EXIT.get(entry["verdict"]["status"], 2) == 0
            clean &= group_clean
            progress = Path(args.out).with_suffix(".progress.json")
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps(data, indent=1), encoding="utf-8")
            if group_clean and not args.keep_tensors:
                shutil.rmtree(folder)
            else:
                print(f"Retained full-tensor captures: {folder}", flush=True)
            print(
                f"PARITY_GROUP_COMPLETE {min(offset + 8, len(active))}/{len(active)}",
                flush=True,
            )
    except BaseException:
        clean = False
        raise
    finally:
        if clean and not args.keep_tensors and not args.scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    skipped_status = "skipped-by-request" if args.no_cuda else "skipped"
    for case in skipped:
        data["cases"][case.name] = {
            "case": asdict(case),
            "verdict": {
                "status": skipped_status,
                "expected_clause": expected_clause(case)[0],
                "expected_reason": expected_clause(case)[1],
                "expected_explanation": reason_text(*expected_clause(case)),
                "detail": (
                    "skipped: no CUDA, so this run does not cover the CUDA "
                    "cases; pass --no-cuda to say that was intended"
                    if skipped_status == "skipped"
                    else "skipped: no CUDA, by request"
                ),
            },
        }
    data["meta"]["seconds"] = time.time() - started
    return _finish(data, args)


def _finish(data: dict[str, Any], args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    markdown = Path(args.md) if args.md else out.with_suffix(".md")
    write_report(data, markdown, args.coverage)
    status_counts: dict[str, int] = {}
    for entry in data["cases"].values():
        status = entry["verdict"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print("", flush=True)
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    for name, entry in sorted(data["cases"].items()):
        verdict = entry["verdict"]
        if verdict["status"] not in ("ok", "skipped"):
            print(f"  {verdict['status']}: {name} -- {verdict.get('detail', '')[:140]}")
    print(f"\nwrote {out} and {markdown}", flush=True)
    return max(
        (_STATUS_EXIT.get(status, 2) for status in status_counts),
        default=0,
    )


def cmd_report(args: argparse.Namespace) -> int:
    """Re-classify and re-render a saved record."""
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    for entry in data["cases"].values():
        if entry["verdict"]["status"] == "skipped":
            continue
        entry["verdict"] = classify(Case(**entry["case"]), entry)
    args.out = args.out or args.input
    return _finish(data, args)


def cmd_list(args: argparse.Namespace) -> int:
    """Print the design, without running anything."""
    cases = _filter(build_design(args.repeats, args.quick), args.only)
    for case in cases:
        clause, reason = expected_clause(case)
        note = f"  expects ({clause}) {reason}" if clause != "i" else ""
        print(f"{case.name}  [{case.block}] repeats={case.repeats}{note}")
    print(f"\n{len(cases)} cases")
    if args.coverage:
        print("\n".join(_format_coverage(coverage(cases))))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the requested subcommand."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare this Deepwave's numbers against the released v0.0.27, "
            "and classify every difference."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_design_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--only", help="run only cases whose name matches this regex")
        sub.add_argument(
            "--repeats",
            type=int,
            default=_DEFAULT_REPEATS,
            help="runs per case where a version's own spread is measured",
        )
        sub.add_argument(
            "--quick",
            action="store_true",
            help=(
                "run the kernel block at accuracy 4 only (48 of its 192 "
                "cases) and leave the call-shape blocks alone; a fast check "
                "of the harness, not a gate run"
            ),
        )
        sub.add_argument(
            "--coverage", action="store_true", help="also print the axis coverage"
        )

    run = subparsers.add_parser("run", help="capture both versions and classify")
    run.add_argument(
        "--reference-python",
        required=True,
        help="interpreter of an environment with deepwave v0.0.27 installed",
    )
    run.add_argument(
        "--out",
        default="benchmarks/results/v0027_parity.json",
        help="where to write the record",
    )
    run.add_argument("--md", help="where to write the report (default: OUT.md)")
    run.add_argument("--no-cuda", action="store_true", help="skip the CUDA cases")
    run.add_argument(
        "--cuda-device", default="cuda:0", help="which CUDA device to run on"
    )
    run.add_argument(
        "--scratch", help="directory for the captured tensors (kept if given)"
    )
    run.add_argument(
        "--keep-tensors",
        action="store_true",
        help="do not delete the captured tensors afterwards",
    )
    add_design_flags(run)

    report = subparsers.add_parser("report", help="re-render a saved record")
    report.add_argument("input")
    report.add_argument("--out")
    report.add_argument("--md")
    report.add_argument("--coverage", action="store_true")

    listing = subparsers.add_parser("list", help="print the design")
    add_design_flags(listing)

    subparsers.add_parser("_worker", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    if args.command == "_worker":
        print(json.dumps(_worker(json.loads(sys.stdin.read()))))
        return 0
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    return cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Measure the absorbing-boundary reflection cost of ``mpml_fraction``.

The multiaxial PML is not perfectly matched: adding a fraction of each axis's
damping profile to the other axes buys stability at the price of a small
reflection off the absorbing layer. This script measures that price directly,
as a reflection coefficient against angle of incidence, so that the numbers
quoted in the ``mpml_fraction`` documentation can be reproduced.

Method
------
A homogeneous elastic half-space is excited by a point force a fixed distance
``d`` from one absorbing face; receivers sit on the same line as the source at
increasing offsets, so the ray reflected off that face strikes it at incidence
``theta = arctan(offset / (2 d))``. The same interior geometry is then run in a
domain extended far enough on every side that none of *its* boundaries can
contribute within the recorded window. The difference between the two runs is
therefore exactly the artefact radiated by the face under test, and

    R(theta) = max_t |u_test(t) - u_reference(t)| / max_t |u_reference(t)|

is the reflection coefficient at that incidence.

Run with no arguments; it prints one table per ``mpml_fraction`` value.
"""

from __future__ import annotations

import argparse
import math

import torch

from deepwave import elastic, vpvsrho_to_lambmubuoyancy
from deepwave.wavelets import ricker

# Geometry (cells), medium and recording parameters.
DX = 5.0
DT = 0.0008
FREQ = 15.0
VP, VS, RHO = 2000.0, 1200.0, 2200.0
PML_WIDTH = 20
SOURCE_DEPTH = 30  # cells from the face under test
INTERIOR_NY, INTERIOR_NX = 140, 400
SOURCE_X = 200
EXTENSION = 180  # cells added on every side for the reference run
NT = 625  # 0.5 s: only the face under test can contribute by then
ANGLES = (0.0, 10.0, 20.0, 30.0, 40.0, 49.0)


def _offsets() -> list[int]:
    return [round(2 * SOURCE_DEPTH * math.tan(math.radians(a))) for a in ANGLES]


def _run(mpml_fraction: float, extension: int) -> torch.Tensor:
    """Record ``vy`` and ``vx`` for the geometry, padded by ``extension``."""
    ny = INTERIOR_NY + extension * 2
    nx = INTERIOR_NX + extension * 2
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(
        torch.tensor(VP), torch.tensor(VS), torch.tensor(RHO)
    )
    shape = (ny, nx)
    lamb = torch.full(shape, float(lamb), dtype=torch.float64)
    mu = torch.full(shape, float(mu), dtype=torch.float64)
    buoyancy = torch.full(shape, float(buoyancy), dtype=torch.float64)
    src_y = SOURCE_DEPTH + extension
    src_x = SOURCE_X + extension
    source_locations = torch.tensor([[[src_y, src_x]]])
    receiver_locations = torch.tensor(
        [[[src_y, src_x + offset] for offset in _offsets()]]
    )
    amplitudes = (ricker(FREQ, NT, DT, 1.5 / FREQ).to(torch.float64) * 1e6).reshape(
        1, 1, NT
    )
    out = elastic(
        lamb,
        mu,
        buoyancy,
        DX,
        DT,
        source_amplitudes_y=amplitudes,
        source_locations_y=source_locations,
        receiver_locations_y=receiver_locations,
        receiver_locations_x=receiver_locations,
        pml_width=PML_WIDTH,
        pml_freq=FREQ,
        mpml_fraction=mpml_fraction,
        max_vel=VP * 1.01,
    )
    return torch.stack([out.receivers["y"][0], out.receivers["x"][0]])


def reflection_coefficients(mpml_fraction: float) -> list[float]:
    """Reflection coefficient at each angle in :data:`ANGLES`."""
    test = _run(mpml_fraction, 0)
    reference = _run(mpml_fraction, EXTENSION)
    residual = (test - reference).abs().amax(dim=(0, 2))
    direct = reference.abs().amax(dim=(0, 2))
    return (residual / direct).tolist()


def main() -> None:
    """Print reflection coefficients for the requested M-PML fractions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpml-fraction",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.2],
        help="values of mpml_fraction to measure",
    )
    args = parser.parse_args()
    header = "  ".join(f"{a:5.0f}deg" for a in ANGLES)
    print(f"{'p':>6}  {header}")
    for p in args.mpml_fraction:
        row = reflection_coefficients(p)
        print(f"{p:6.2f}  " + "  ".join(f"{r:8.2e}" for r in row))


if __name__ == "__main__":
    main()

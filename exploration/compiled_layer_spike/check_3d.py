"""Bonus 3-D existence check for the scalar family.

3-D was out of the spike's required scope; this only asks whether the same
generator, with no changes, produces a 3-D scalar kernel that builds, passes
the doubly-derived adjoint check, and agrees with the incumbent.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

import deepwave

from dwgen import build as buildmod
from dwgen.specs import scalar as scalar_spec
from spike import frontend, ops


def rel(a, b):
    return float((a - b).abs().max() / a.abs().max())


def main():
    ops.configure(
        [buildmod.Variant(scalar_spec.make(3), 4, "f64")],
        verify=True,
        cuda=torch.cuda.is_available(),
    )
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    for device in devices:
        torch.manual_seed(5)
        n, nt = 32, 60
        v = 1500 + 100 * torch.rand(n, n, n, dtype=torch.float64, device=device)
        v = v.requires_grad_()
        w = deepwave.wavelets.ricker(15.0, nt, 0.002, 0.05).double().to(device)
        amp = w.reshape(1, 1, -1).repeat(1, 1, 1).requires_grad_()
        sl = torch.tensor([[[6, 16, 16]]], device=device)
        rl = torch.tensor([[[24, 16, 8], [24, 16, 24], [16, 8, 16]]], device=device)
        kw = dict(accuracy=4, pml_width=6, pml_freq=15.0)
        r0 = deepwave.scalar(
            v,
            10.0,
            0.002,
            source_amplitudes=amp,
            source_locations=sl,
            receiver_locations=rl,
            **kw,
        )[-1]
        r1 = frontend.scalar(v, 10.0, 0.002, amp, sl, rl, ndim=3, **kw)
        g0 = torch.autograd.grad(r0.square().sum(), [v, amp], retain_graph=True)
        g1 = torch.autograd.grad(r1.square().sum(), [v, amp])
        print(
            f"{device:8s} 3-D scalar acc4 f64: forward {rel(r0, r1):.2e}  "
            f"grad_v {rel(g0[0], g1[0]):.2e}  grad_f {rel(g0[1], g1[1]):.2e}  "
            f"finite={bool(torch.isfinite(r1).all())}"
        )


if __name__ == "__main__":
    main()

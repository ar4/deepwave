"""Run-to-run bit-reproducibility of CUDA model gradients.

The incumbent accumulates model gradients with ``atomicAdd`` on CUDA, whose
summation order varies between runs, so its gradients are not bit-reproducible
(recorded as open finding NF9).  The generated kernels give every shot a
private gradient buffer and let exactly one thread own each cell, with the
cross-shot reduction done afterwards by torch; there is no atomic anywhere in
the gradient path.  This script runs each backward twice on the same inputs
and reports whether the two gradients are bitwise equal.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

import deepwave

from dwgen import build as buildmod
from dwgen.specs import elastic as elastic_spec
from dwgen.specs import scalar as scalar_spec
from spike import frontend, ops

N, NT, NSHOTS = 256, 300, 4


def scalar_grads(fn, device):
    torch.manual_seed(1)
    v = (1500 + 200 * torch.rand(N, N, device=device)).requires_grad_()
    w = deepwave.wavelets.ricker(15.0, NT, 0.001, 0.05).to(device)
    amp = w.repeat(NSHOTS, 2, 1)
    sl = torch.tensor([[[10, 40], [10, 200]]] * NSHOTS, device=device)
    rl = torch.tensor(
        [[[200, i] for i in range(20, N - 20, 4)]] * NSHOTS, device=device
    )
    r = fn(v, amp, sl, rl)
    (r.square().sum()).backward()
    return v.grad.clone()


def elastic_grads(incumbent, device, kw):
    torch.manual_seed(2)
    vp = 2000 + 200 * torch.rand(N, N, device=device)
    vs = 1100 + 100 * torch.rand(N, N, device=device)
    rho = 2000 + 100 * torch.rand(N, N, device=device)
    lamb, mu, buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb = lamb.detach().requires_grad_()
    mu = mu.detach().requires_grad_()
    buoy = buoy.detach().requires_grad_()
    w = deepwave.wavelets.ricker(12.0, NT, 0.001, 0.06).to(device)
    amp = w.repeat(NSHOTS, 2, 1)
    sl = torch.tensor([[[10, 40], [10, 200]]] * NSHOTS, device=device)
    rl = torch.tensor(
        [[[200, i] for i in range(20, N - 20, 4)]] * NSHOTS, device=device
    )
    io = dict(source_amplitudes_y=amp, source_locations_y=sl, receiver_locations_p=rl)
    if incumbent:
        r = deepwave.elastic(lamb, mu, buoy, 4.0, 0.001, **io, **kw)[-3]
    else:
        r = frontend.elastic(lamb, mu, buoy, 4.0, 0.001, **io, **kw)[0]
    r.square().sum().backward()
    return [lamb.grad.clone(), mu.grad.clone(), buoy.grad.clone()]


def main():
    if not torch.cuda.is_available():
        print("no CUDA available")
        return
    variants = [
        buildmod.Variant(s.make(2), a, d)
        for s in (scalar_spec, elastic_spec)
        for a in (2, 4)
        for d in ("f32", "f64")
    ]
    ops.configure(variants, verify=False, cuda=True)
    kw = dict(accuracy=4, pml_width=20, pml_freq=15.0)
    for dev in [f"cuda:{i}" for i in range(torch.cuda.device_count())]:

        def inc(v, amp, sl, rl):
            return deepwave.scalar(
                v,
                4.0,
                0.001,
                source_amplitudes=amp,
                source_locations=sl,
                receiver_locations=rl,
                **kw,
            )[-1]

        def spk(v, amp, sl, rl):
            return frontend.scalar(v, 4.0, 0.001, amp, sl, rl, **kw)

        for name, fn in (("incumbent", inc), ("spike", spk)):
            gs = [scalar_grads(fn, dev) for _ in range(3)]
            same = all(torch.equal(gs[0], g) for g in gs[1:])
            worst = max((gs[0] - g).abs().max().item() for g in gs[1:])
            print(
                f"{dev} {name:9s} scalar  model gradient, 3 runs: "
                f"{'BITWISE IDENTICAL' if same else 'DIFFERS'} "
                f"(max abs difference {worst:.3e})"
            )
        for name, fn in (("incumbent", True), ("spike", False)):
            gs = [elastic_grads(fn, dev, kw) for _ in range(3)]
            same = all(all(torch.equal(a, b) for a, b in zip(gs[0], g)) for g in gs[1:])
            worst = max(
                max((a - b).abs().max().item() for a, b in zip(gs[0], g))
                for g in gs[1:]
            )
            print(
                f"{dev} {name:9s} elastic model gradients, 3 runs: "
                f"{'BITWISE IDENTICAL' if same else 'DIFFERS'} "
                f"(max abs difference {worst:.3e})"
            )


if __name__ == "__main__":
    main()

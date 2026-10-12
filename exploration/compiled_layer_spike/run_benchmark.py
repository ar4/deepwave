"""Wall-time benchmark: generated kernels vs. the incumbent Deepwave.

One representative 2-D size (the project's own "meso" tier grid, 512^2, with
nt halved to 500 to keep the elastic gradient snapshots inside the smaller
card's memory), accuracy 4, float32, one shot, on CPU and on both GPUs.
Forward-only and forward+backward are timed separately.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

import deepwave

from dwgen import build as buildmod
from dwgen.specs import elastic as elastic_spec
from dwgen.specs import scalar as scalar_spec
from spike import frontend, ops

N = 512
NT = 500
ACC = 4
DX = 4.0
DT = 0.0005


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def timeit(fn, device, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
        sync(device)
    ts = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def scalar_case(device, dtype, grad):
    v = torch.full((N, N), 1500.0, dtype=dtype, device=device)
    if grad:
        v = v.requires_grad_()
    w = deepwave.wavelets.ricker(15.0, NT, DT, 0.05).to(dtype).to(device)
    amp = w.reshape(1, 1, -1)
    sl = torch.tensor([[[N // 2, N // 2]]], device=device)
    rl = torch.tensor([[[N // 4, i] for i in range(20, N - 20, 8)]], device=device)
    kw = dict(accuracy=ACC, pml_width=20, pml_freq=15.0)
    return v, amp, sl, rl, kw


def elastic_case(device, dtype, grad):
    vp = torch.full((N, N), 2000.0, dtype=dtype, device=device)
    vs = torch.full((N, N), 1100.0, dtype=dtype, device=device)
    rho = torch.full((N, N), 2000.0, dtype=dtype, device=device)
    lamb, mu, buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb, mu, buoy = lamb.detach(), mu.detach(), buoy.detach()
    if grad:
        lamb.requires_grad_()
        mu.requires_grad_()
        buoy.requires_grad_()
    w = deepwave.wavelets.ricker(12.0, NT, DT, 0.06).to(dtype).to(device)
    amp = w.reshape(1, 1, -1)
    sl = torch.tensor([[[N // 2, N // 2]]], device=device)
    rl = torch.tensor([[[N // 4, i] for i in range(20, N - 20, 8)]], device=device)
    kw = dict(accuracy=ACC, pml_width=20, pml_freq=12.0)
    return lamb, mu, buoy, amp, sl, rl, kw


def run(device, dtype, n_threads, results):
    tag = "f32" if dtype is torch.float32 else "f64"
    # ---- scalar ----
    for grad in (False, True):
        v, amp, sl, rl, kw = scalar_case(device, dtype, grad)

        def inc():
            r = deepwave.scalar(
                v,
                DX,
                DT,
                source_amplitudes=amp,
                source_locations=sl,
                receiver_locations=rl,
                **kw,
            )[-1]
            if grad:
                r.square().sum().backward()

        def spk():
            r = frontend.scalar(v, DX, DT, amp, sl, rl, n_threads=n_threads, **kw)
            if grad:
                r.square().sum().backward()

        mi, _ = timeit(inc, device)
        ms, _ = timeit(spk, device)
        mode = "fwd+bwd" if grad else "fwd"
        results.append(("scalar", device, tag, mode, mi * 1e3, ms * 1e3))
        print(
            f"scalar {device:8s} {tag} {mode:8s} incumbent {mi * 1e3:8.1f} ms  "
            f"spike {ms * 1e3:8.1f} ms  ratio {ms / mi:.2f}x"
        )
        del v, amp
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- elastic ----
    for grad in (False, True):
        lamb, mu, buoy, amp, sl, rl, kw = elastic_case(device, dtype, grad)
        io = dict(
            source_amplitudes_y=amp,
            source_locations_y=sl,
            receiver_locations_p=rl,
        )

        def inc():
            r = deepwave.elastic(lamb, mu, buoy, DX, DT, **io, **kw)[-3]
            if grad:
                r.square().sum().backward()

        def spk():
            r = frontend.elastic(
                lamb, mu, buoy, DX, DT, n_threads=n_threads, **io, **kw
            )[0]
            if grad:
                r.square().sum().backward()

        mi, _ = timeit(inc, device)
        ms, _ = timeit(spk, device)
        mode = "fwd+bwd" if grad else "fwd"
        results.append(("elastic", device, tag, mode, mi * 1e3, ms * 1e3))
        print(
            f"elastic {device:7s} {tag} {mode:8s} incumbent {mi * 1e3:8.1f} ms  "
            f"spike {ms * 1e3:8.1f} ms  ratio {ms / mi:.2f}x"
        )
        del lamb, mu, buoy, amp
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    variants = [
        buildmod.Variant(s.make(2), a, d)
        for s in (scalar_spec, elastic_spec)
        for a in (2, 4)
        for d in ("f32", "f64")
    ]
    ops.configure(variants, verify=False, cuda=torch.cuda.is_available())
    devices = (
        args.devices.split(",")
        if args.devices
        else ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    )
    print(f"grid {N}^2, nt={NT}, accuracy {ACC}, 1 shot, CPU threads={args.threads}")
    results = []
    for dev in devices:
        run(dev, torch.float32, args.threads, results)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

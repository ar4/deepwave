"""Forward and gradient parity of the generated kernels vs. incumbent Deepwave.

For every (family, accuracy, dtype, device) in scope the generated kernel is
compared against the real ``deepwave.scalar`` / ``deepwave.elastic``
propagator, driven through the same Python front-end, on the same inputs.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import deepwave  # noqa: E402

from dwgen import build as buildmod  # noqa: E402
from dwgen.specs import elastic as elastic_spec  # noqa: E402
from dwgen.specs import scalar as scalar_spec  # noqa: E402
from spike import frontend, ops  # noqa: E402

ACCS = (2, 4)
DTYPES = {"f32": torch.float32, "f64": torch.float64}
# Forward tolerance, and a looser gradient tolerance.  Both are calibrated
# against the *incumbent's own* float32-vs-float64 self-discrepancy on this
# survey, measured at 8e-7..1.7e-5 (forward) and 1.6e-5..2.2e-5 (model
# gradient): a float32 disagreement of that order between two correct
# implementations is the arithmetic noise floor, not a defect.  Both sides are
# built with -ffast-math / --use_fast_math, so reassociation differs.
TOL = {"f32": 2e-5, "f64": 1e-11}
GTOL = {"f32": 1e-4, "f64": 1e-11}

_devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _configure():
    variants = [
        buildmod.Variant(s.make(2), a, d)
        for s in (scalar_spec, elastic_spec)
        for a in ACCS
        for d in DTYPES
    ]
    ops.configure(variants, verify=True, cuda=torch.cuda.is_available())


_CONFIGURED = False


def setup_module(module):
    global _CONFIGURED
    if not _CONFIGURED:
        _configure()
        _CONFIGURED = True


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a - b).abs().max().item()
    s = a.abs().max().item()
    return d / s if s > 0 else d


def _survey(dtype, device, n_shots=2, nt=160):
    torch.manual_seed(7)
    ny, nx = 50, 55
    v = (1500 + 300 * torch.rand(ny, nx, dtype=dtype, device=device)).requires_grad_()
    f = deepwave.wavelets.ricker(15.0, nt, 0.002, 0.05).to(dtype).to(device)
    amp = (
        f.repeat(n_shots, 3, 1)
        * torch.tensor([1.0, 0.7, 1.3], dtype=dtype, device=device)[None, :, None]
    ).requires_grad_()
    sl = torch.tensor([[[8, 10], [8, 25], [8, 40]]] * n_shots, device=device)
    rl = torch.tensor(
        [[[40, 10], [40, 20], [40, 30], [40, 45], [25, 47]]] * n_shots, device=device
    )
    return v, amp, sl, rl


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("dtag", list(DTYPES))
@pytest.mark.parametrize("acc", ACCS)
def test_scalar_2d(device, dtag, acc):
    if device != "cpu":
        pytest.importorskip("torch.cuda")
    dtype = DTYPES[dtag]
    v, amp, sl, rl = _survey(dtype, device)
    kw = dict(accuracy=acc, pml_width=10, pml_freq=15.0)
    r0 = deepwave.scalar(
        v,
        4.0,
        0.002,
        source_amplitudes=amp,
        source_locations=sl,
        receiver_locations=rl,
        **kw,
    )[-1]
    r1 = frontend.scalar(v, 4.0, 0.002, amp, sl, rl, **kw)
    assert r0.shape == r1.shape
    assert r0.abs().max() > 0
    fe = _rel(r0, r1)
    g0 = torch.autograd.grad(r0.square().sum(), [v, amp], retain_graph=True)
    g1 = torch.autograd.grad(r1.square().sum(), [v, amp])
    ge = [_rel(a, b) for a, b in zip(g0, g1)]
    assert fe < TOL[dtag], f"forward {fe}"
    assert ge[0] < GTOL[dtag], f"grad_v {ge[0]}"
    assert ge[1] < GTOL[dtag], f"grad_f {ge[1]}"


def _elastic_survey(dtype, device, n_shots=2, nt=140):
    torch.manual_seed(3)
    ny, nx = 50, 55
    vp = 2000 + 200 * torch.rand(ny, nx, dtype=dtype, device=device)
    vs = 1100 + 100 * torch.rand(ny, nx, dtype=dtype, device=device)
    rho = 2000 + 100 * torch.rand(ny, nx, dtype=dtype, device=device)
    lamb, mu, buoy = deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb = lamb.detach().requires_grad_()
    mu = mu.detach().requires_grad_()
    buoy = buoy.detach().requires_grad_()
    w = deepwave.wavelets.ricker(12.0, nt, 0.001, 0.06).to(dtype).to(device)
    amp = w.repeat(n_shots, 2, 1).detach().requires_grad_()
    sl = torch.tensor([[[10, 12], [10, 30]]] * n_shots, device=device)
    rl = torch.tensor(
        [[[35, 12], [35, 25], [35, 40], [20, 45]]] * n_shots, device=device
    )
    return lamb, mu, buoy, amp, sl, rl


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("dtag", list(DTYPES))
@pytest.mark.parametrize("acc", ACCS)
def test_elastic_2d(device, dtag, acc):
    dtype = DTYPES[dtag]
    lamb, mu, buoy, amp, sl, rl = _elastic_survey(dtype, device)
    kw = dict(accuracy=acc, pml_width=10, pml_freq=12.0)
    io = dict(
        source_amplitudes_y=amp,
        source_locations_y=sl,
        receiver_locations_p=rl,
        receiver_locations_y=rl,
        receiver_locations_x=rl,
    )
    o0 = deepwave.elastic(lamb, mu, buoy, 4.0, 0.001, **io, **kw)
    r0 = (o0[-3], o0[-2], o0[-1])
    r1 = frontend.elastic(lamb, mu, buoy, 4.0, 0.001, **io, **kw)
    for a, b in zip(r0, r1):
        assert a.shape == b.shape
        assert a.abs().max() > 0
        assert _rel(a, b) < TOL[dtag], f"forward {_rel(a, b)}"
    leaves = [lamb, mu, buoy, amp]
    L0 = sum(t.square().sum() for t in r0)
    L1 = sum(t.square().sum() for t in r1)
    g0 = torch.autograd.grad(L0, leaves, retain_graph=True)
    g1 = torch.autograd.grad(L1, leaves)
    for nm, a, b in zip(["lamb", "mu", "buoyancy", "src"], g0, g1):
        e = _rel(a, b)
        assert e < GTOL[dtag], f"grad {nm} {e}"

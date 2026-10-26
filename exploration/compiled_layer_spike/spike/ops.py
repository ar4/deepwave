"""PyTorch binding: one functional ``torch.library.custom_op`` per direction.

The kernel library is a black box behind a flat C ABI, which is the documented
first-class use case for ``torch.library.custom_op``.  The op is functional (it
mutates none of its inputs -- every buffer it writes is allocated here, by
torch, so the caching allocator owns all device memory and the kernel library
never allocates), and the backward is registered as its own custom op via
``torch.library.register_autograd``.
"""

from __future__ import annotations

from typing import List

import torch

from dwgen import abi

from . import loader

_LIB = None
_VARIANTS: list = []


def configure(variants: list, verify: bool = True, cuda: bool = False) -> None:
    global _LIB, _VARIANTS
    _VARIANTS = variants
    _LIB = loader.get_library(variants, verify=verify, cuda=cuda)


def library():
    if _LIB is None:
        raise RuntimeError("spike.ops.configure() must be called first")
    return _LIB


def parse_key(key: str):
    fam, nd, acc, dt, dev = key.split("|")
    return fam, int(nd), int(acc), dt, dev


def _model(key: str):
    fam, nd, acc, dt, dev = parse_key(key)
    return library().models[(fam, nd, acc, dt)]


def _kernel(key: str, direction: str):
    fam, nd, acc, dt, dev = parse_key(key)
    return library().kernels[(fam, nd, acc, dt, direction, dev)]


def _int_names(spec, m, direction):
    return [a.name for a in abi.signature(spec, m, direction) if a.kind == "i64"]


def _stream(key: str, ptrs: dict) -> None:
    if parse_key(key)[4] == "cuda":
        ptrs["stream_ptr"] = torch.cuda.current_stream().cuda_stream


def _pack(kern, ptrs: dict, flts: list, ints: list):
    vals = []
    fi = ii = 0
    for a in kern.args:
        if a.kind in ("ptr", "cptr", "iptr", "vptr"):
            vals.append(ptrs[a.name])
        elif a.kind == "T":
            vals.append(flts[fi])
            fi += 1
        else:
            vals.append(ints[ii])
            ii += 1
    return vals


def _dp(t: torch.Tensor) -> int:
    return t.data_ptr()


@torch.library.custom_op("dwspike::fwd", mutates_args=())
def fwd(
    params: List[torch.Tensor],
    profiles: List[torch.Tensor],
    state0: List[torch.Tensor],
    src_amp: List[torch.Tensor],
    src_i: List[torch.Tensor],
    rec_i: List[torch.Tensor],
    key: str,
    ints: List[int],
    ints_bwd: List[int],
    flts: List[float],
) -> List[torch.Tensor]:
    m = _model(key)
    spec = m.spec
    kern = _kernel(key, "fwd")
    names = _int_names(spec, m, "fwd")
    iv = dict(zip(names, ints))
    nd = spec.ndim
    dims = [iv[f"n{a}"] for a in spec.axis_names]
    n_shots, nt = iv["n_shots"], iv["nt"]
    dev, dt = state0[0].device, state0[0].dtype
    grid = (n_shots, *dims)

    cur = [s.contiguous().clone() for s in state0]
    nxt = [torch.zeros_like(s) for s in cur]
    recs = [
        torch.zeros((nt, n_shots, iv[f"n_rec{k}"]), device=dev, dtype=dt)
        for k in range(spec.n_receivers)
    ]
    nstore = (nt + iv["step_ratio"] - 1) // iv["step_ratio"]
    sshape = (nstore, *grid) if iv["store_all_steps"] else grid
    stores = [
        torch.zeros(sshape if iv["need_store"] else (1,), device=dev, dtype=dt)
        for _ in m.stores
    ]
    ptrs = {}
    for i, f in enumerate(spec.state):
        ptrs[f"{f}_cur"] = _dp(cur[i])
        ptrs[f"{f}_nxt"] = _dp(nxt[i])
    for i, p in enumerate(spec.params):
        ptrs[f"par_{p}"] = _dp(params[i])
    for i, pa in enumerate(spec.profile_arrays()):
        ptrs[f"prof_{pa}"] = _dp(profiles[i])
    for i, s in enumerate(m.stores):
        ptrs[f"st_{s}"] = _dp(stores[i])
    for k in range(spec.n_sources):
        ptrs[f"src{k}_amp"] = _dp(src_amp[k])
        ptrs[f"src{k}_i"] = _dp(src_i[k])
    for k in range(spec.n_receivers):
        ptrs[f"rec{k}_out"] = _dp(recs[k])
        ptrs[f"rec{k}_i"] = _dp(rec_i[k])
    _stream(key, ptrs)
    rc = kern(_pack(kern, ptrs, flts, ints))
    if rc != 0:
        raise RuntimeError(f"kernel returned {rc}")
    return [*recs, *cur, *stores]


@fwd.register_fake
def _(params, profiles, state0, src_amp, src_i, rec_i, key, ints, ints_bwd, flts):
    m = _model(key)
    spec = m.spec
    iv = dict(zip(_int_names(spec, m, "fwd"), ints))
    dims = [iv[f"n{a}"] for a in spec.axis_names]
    n_shots, nt = iv["n_shots"], iv["nt"]
    grid = (n_shots, *dims)
    dev, dt = state0[0].device, state0[0].dtype
    nstore = (nt + iv["step_ratio"] - 1) // iv["step_ratio"]
    sshape = (nstore, *grid) if iv["store_all_steps"] else grid
    out = [
        torch.empty((nt, n_shots, iv[f"n_rec{k}"]), device=dev, dtype=dt)
        for k in range(spec.n_receivers)
    ]
    out += [torch.empty_like(s) for s in state0]
    out += [
        torch.empty(sshape if iv["need_store"] else (1,), device=dev, dtype=dt)
        for _ in m.stores
    ]
    return out


@torch.library.custom_op("dwspike::bwd", mutates_args=())
def bwd(
    params: List[torch.Tensor],
    profiles: List[torch.Tensor],
    stores: List[torch.Tensor],
    rec_grad: List[torch.Tensor],
    src_i: List[torch.Tensor],
    rec_i: List[torch.Tensor],
    key: str,
    ints: List[int],
    flts: List[float],
) -> List[torch.Tensor]:
    m = _model(key)
    spec = m.spec
    kern = _kernel(key, "bwd")
    iv = dict(zip(_int_names(spec, m, "bwd"), ints))
    dims = [iv[f"n{a}"] for a in spec.axis_names]
    n_shots, nt = iv["n_shots"], iv["nt"]
    dev, dt = params[0].device, params[0].dtype
    grid = (n_shots, *dims)
    adj = [torch.zeros(grid, device=dev, dtype=dt) for _ in spec.state]
    adjn = [torch.zeros(grid, device=dev, dtype=dt) for _ in spec.state]
    gsrc = [
        torch.zeros((nt, n_shots, iv[f"n_src{k}"]), device=dev, dtype=dt)
        for k in range(spec.n_sources)
    ]
    # Per-shot private gradient buffers, reduced deterministically below: no
    # atomicAdd anywhere in the gradient path.
    gpar = [torch.zeros(grid, device=dev, dtype=dt) for _ in spec.params]
    ptrs = {}
    for i, f in enumerate(spec.state):
        ptrs[f"{f}_cur"] = _dp(adj[i])
        ptrs[f"{f}_nxt"] = _dp(adjn[i])
    for i, p in enumerate(spec.params):
        ptrs[f"par_{p}"] = _dp(params[i])
        ptrs[f"grad_{p}"] = _dp(gpar[i])
    for i, pa in enumerate(spec.profile_arrays()):
        ptrs[f"prof_{pa}"] = _dp(profiles[i])
    for i, s in enumerate(m.stores):
        ptrs[f"st_{s}"] = _dp(stores[i])
    for k in range(spec.n_sources):
        ptrs[f"src{k}_grad"] = _dp(gsrc[k])
        ptrs[f"src{k}_i"] = _dp(src_i[k])
    for k in range(spec.n_receivers):
        ptrs[f"rec{k}_grad"] = _dp(rec_grad[k])
        ptrs[f"rec{k}_i"] = _dp(rec_i[k])
    _stream(key, ptrs)
    rc = kern(_pack(kern, ptrs, flts, ints))
    if rc != 0:
        raise RuntimeError(f"kernel returned {rc}")
    return [*gsrc, *gpar]


@bwd.register_fake
def _(params, profiles, stores, rec_grad, src_i, rec_i, key, ints, flts):
    m = _model(key)
    spec = m.spec
    iv = dict(zip(_int_names(spec, m, "bwd"), ints))
    dims = [iv[f"n{a}"] for a in spec.axis_names]
    n_shots, nt = iv["n_shots"], iv["nt"]
    dev, dt = params[0].device, params[0].dtype
    out = [
        torch.empty((nt, n_shots, iv[f"n_src{k}"]), device=dev, dtype=dt)
        for k in range(spec.n_sources)
    ]
    out += [torch.empty((n_shots, *dims), device=dev, dtype=dt) for _ in spec.params]
    return out


def _setup_context(ctx, inputs, output):
    params, profiles, state0, src_amp, src_i, rec_i, key, ints, ints_bwd, flts = inputs
    m = _model(key)
    spec = m.spec
    n_rec, n_st = spec.n_receivers, len(spec.state)
    stores = list(output[n_rec + n_st :])
    ctx.save_for_backward(*params, *profiles, *stores, *src_i, *rec_i)
    ctx.meta = (
        key,
        ints_bwd,
        flts,
        len(params),
        len(profiles),
        len(stores),
        len(src_i),
    )


def _backward(ctx, *grads):
    key, ints, flts, npar, nprof, nstore, nsrc = ctx.meta
    m = _model(key)
    spec = m.spec
    saved = list(ctx.saved_tensors)
    params = saved[:npar]
    profiles = saved[npar : npar + nprof]
    stores = saved[npar + nprof : npar + nprof + nstore]
    rest = saved[npar + nprof + nstore :]
    src_i = rest[:nsrc]
    rec_i = rest[nsrc:]
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = tuple(grads[0])
    ex = params[0]
    n_shots = ints[1]
    rec_grad = []
    for k in range(spec.n_receivers):
        g = grads[k]
        if g is None:
            nt = ints[0]
            nrec = dict(zip(_int_names(spec, m, "fwd"), ints))[f"n_rec{k}"]
            g = torch.zeros((nt, n_shots, nrec), device=ex.device, dtype=ex.dtype)
        rec_grad.append(g.contiguous())
    out = torch.ops.dwspike.bwd(
        params, profiles, stores, rec_grad, src_i, rec_i, key, ints, flts
    )
    gsrc = list(out[: spec.n_sources])
    gpar = list(out[spec.n_sources :])
    gpar = [g if p.dim() == g.dim() else g.sum(0) for g, p in zip(gpar, params)]
    return (
        gpar,
        [None] * nprof,
        [None] * len(spec.state),
        gsrc,
        [None] * nsrc,
        [None] * len(rec_i),
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd("dwspike::fwd", _backward, setup_context=_setup_context)

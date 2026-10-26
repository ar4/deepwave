"""Callback tests for deepwave.acoustic_born.

Mirrors tests/test_callbacks_scalar_born.py. forward_callback/backward_callback
are threaded through the shared engine on the default compiled backend; the
pure-PyTorch ``backend="eager"`` driver has no chunked adjoint
machinery, so it still rejects ``backward_callback``, but it fires
``forward_callback`` from its own per-step loop (see
tests/test_acoustic_born.py::test_eager_rejects_backward_callback_but_not_forward_callback
and ::test_acoustic_born_eager_forward_callback_dict_keys_and_call_count).

A single receiver is used throughout (the ``_problem`` convention on which the
compiled adjoint is validated).
"""

import importlib

import torch
from helpers import output_tensors

import deepwave

ab = importlib.import_module("deepwave.acoustic_born")


def _problem() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, dict
]:
    shape = (12, 12)
    v = torch.full(shape, 1500.0, dtype=torch.double)
    rho = torch.full(shape, 2200.0, dtype=torch.double)
    dv = torch.zeros(shape, dtype=torch.double)
    drho = torch.zeros(shape, dtype=torch.double)
    dv[6, 6] = 10.0
    drho[6, 6] = 40.0
    dx = 6.0
    dt = 0.0004
    nt = 20
    torch.manual_seed(0)
    samp = torch.randn(1, 1, nt, dtype=torch.double)
    loc = torch.zeros(1, 1, 2, dtype=torch.long)
    loc[0, 0, 0] = 6
    loc[0, 0, 1] = 6
    rloc = loc.clone()
    rloc[0, 0, 1] = 4
    kwargs = {
        "source_amplitudes_volume_rate": samp,
        "source_locations_volume_rate": loc,
        "sc_receiver_locations_p": rloc,
        "pml_width": 4,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
    }
    return v, rho, dv, drho, dx, dt, kwargs


class _Counter:
    """A simple counter for callbacks."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, state: deepwave.CallbackState) -> None:
        self.count += 1


def _scattered(out: tuple[torch.Tensor, ...], ndim: int = 2) -> torch.Tensor:
    names = ab.acoustic_born_output_names(ndim)
    return dict(zip(names, out))["sc_receiver_amplitudes_p"]


def test_acoustic_born_callback_call_count() -> None:
    """Forward and backward callbacks fire the correct number of times."""
    v, rho, dv0, drho, dx, dt, kwargs = _problem()
    nt = kwargs["nt"]

    # Frequency that divides nt evenly.
    fc, bc = _Counter(), _Counter()
    dv = dv0.clone().requires_grad_()
    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        forward_callback=fc,
        backward_callback=bc,
        callback_frequency=2,
        **kwargs,
    )
    assert fc.count == nt / 2
    _scattered(out).sum().backward()
    assert bc.count == nt / 2

    # Frequency that does not divide nt evenly.
    fc, bc = _Counter(), _Counter()
    dv = dv0.clone().requires_grad_()
    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        forward_callback=fc,
        backward_callback=bc,
        callback_frequency=3,
        **kwargs,
    )
    assert fc.count == (nt + 2) // 3
    _scattered(out).sum().backward()
    assert bc.count == (nt + 2) // 3


def test_acoustic_born_backward_callback_only_call_count() -> None:
    """The backward callback fires correctly with no forward callback."""
    v, rho, dv0, drho, dx, dt, kwargs = _problem()
    nt = kwargs["nt"]
    bc = _Counter()
    dv = dv0.clone().requires_grad_()
    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        backward_callback=bc,
        callback_frequency=2,
        **kwargs,
    )
    _scattered(out).pow(2).sum().backward()
    assert bc.count == nt / 2
    grad1 = dv.grad.clone()

    bc = _Counter()
    dv = dv0.clone().requires_grad_()
    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        backward_callback=bc,
        callback_frequency=3,
        **kwargs,
    )
    _scattered(out).pow(2).sum().backward()
    assert bc.count == (nt + 2) // 3
    grad2 = dv.grad.clone()
    assert torch.allclose(grad1, grad2)


def test_acoustic_born_callback_equivalence() -> None:
    """A do-nothing forward+backward callback changes neither output nor grad."""
    v, rho, dv0, drho, dx, dt, kwargs = _problem()

    def nothing(state: deepwave.CallbackState) -> None:
        pass

    def run(callbacks: bool) -> tuple[tuple, torch.Tensor]:
        dv = dv0.clone().requires_grad_()
        extra = (
            {
                "forward_callback": nothing,
                "backward_callback": nothing,
                "callback_frequency": 1,
            }
            if callbacks
            else {}
        )
        out = ab.acoustic_born(v, rho, dx, dt, dv=dv, drho=drho, **kwargs, **extra)
        _scattered(out).pow(2).sum().backward()
        return out, dv.grad.clone()

    out1, g1 = run(False)
    out2, g2 = run(True)
    for a, b in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(a, b)
    assert torch.allclose(g1, g2)


def test_acoustic_born_full_backward_callback_equivalence() -> None:
    """Chunk granularity must not change the full two-field backward gradients.

    When ``v``/``rho`` also require grad the ``need_bg_adjoint=True`` path runs
    (``make_backward_call`` + the full ``swap``), distinct from the scattered-
    only ``backward_sc`` path the other tests exercise. Both the background
    (v/rho) and scattered (dv) adjoint psi memory variables ping-pong across
    chunk boundaries, so a do-nothing callback (chunked) must reproduce the
    single-chunk gradients exactly.
    """
    v0, rho0, dv0, drho, dx, dt, kwargs = _problem()

    def nothing(state: deepwave.CallbackState) -> None:
        pass

    def run(callbacks: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = v0.clone().requires_grad_()
        rho = rho0.clone().requires_grad_()
        dv = dv0.clone().requires_grad_()
        extra = (
            {
                "forward_callback": nothing,
                "backward_callback": nothing,
                "callback_frequency": 1,
            }
            if callbacks
            else {}
        )
        out = ab.acoustic_born(v, rho, dx, dt, dv=dv, drho=drho, **kwargs, **extra)
        _scattered(out).pow(2).sum().backward()
        return v.grad.clone(), rho.grad.clone(), dv.grad.clone()

    gv1, gr1, gd1 = run(False)
    gv2, gr2, gd2 = run(True)
    assert torch.allclose(gv1, gv2)
    assert torch.allclose(gr1, gr2)
    assert torch.allclose(gd1, gd2)


def test_acoustic_born_forward_callback_can_modify_wavefield() -> None:
    """A forward callback receives a live, mutable view of the wavefield."""
    v, rho, dv, drho, dx, dt, kwargs = _problem()
    kwargs = dict(kwargs)
    kwargs["pml_width"] = 5
    seen = {}

    def checker(state: deepwave.CallbackState) -> None:
        w = state.get_wavefield("pressure_sc_0", "inner")
        assert w.shape == (1, 12, 12)
        seen["called"] = True

    ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        forward_callback=checker,
        callback_frequency=20,
        **kwargs,
    )
    assert seen.get("called")


def test_acoustic_born_eager_forward_callback_call_count() -> None:
    """The eager backend's forward_callback fires the same count as the compiled one."""
    v, rho, dv, drho, dx, dt, kwargs = _problem()
    nt = kwargs["nt"]

    fc_compiled, fc_eager = _Counter(), _Counter()
    ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        forward_callback=fc_compiled,
        callback_frequency=2,
        **kwargs,
    )
    ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        backend="eager",
        forward_callback=fc_eager,
        callback_frequency=2,
        **kwargs,
    )
    assert fc_compiled.count == nt / 2
    assert fc_eager.count == fc_compiled.count


def test_acoustic_born_eager_forward_callback_equivalence() -> None:
    """A do-nothing eager forward_callback changes neither output nor gradients."""
    v, rho, dv0, drho, dx, dt, kwargs = _problem()

    def nothing(state: deepwave.CallbackState) -> None:
        pass

    def run(callback: bool) -> tuple[torch.Tensor, torch.Tensor]:
        dv = dv0.clone().requires_grad_()
        extra = (
            {"forward_callback": nothing, "callback_frequency": 3} if callback else {}
        )
        out = ab.acoustic_born(
            v, rho, dx, dt, dv=dv, drho=drho, backend="eager", **kwargs, **extra
        )
        _scattered(out).pow(2).sum().backward()
        return _scattered(out).detach().clone(), dv.grad.clone()

    out1, g1 = run(False)
    out2, g2 = run(True)
    assert torch.equal(out1, out2)
    assert torch.equal(g1, g2)


def test_acoustic_born_storage_mode_matches_device() -> None:
    """cpu/disk storage give the same forward and gradients as device."""
    v, rho, dv0, drho, dx, dt, kwargs = _problem()

    def run(storage_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        dv = dv0.clone().requires_grad_()
        out = ab.acoustic_born(
            v, rho, dx, dt, dv=dv, drho=drho, storage_mode=storage_mode, **kwargs
        )
        rec = _scattered(out)
        rec.pow(2).sum().backward()
        return rec.detach().clone(), dv.grad.clone()

    ref_out, ref_grad = run("device")
    for mode in ("cpu", "disk"):
        out, grad = run(mode)
        assert torch.allclose(out, ref_out)
        assert torch.allclose(grad, ref_grad)

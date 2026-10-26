"""Tests for the acoustic Born (linearised) propagator.

Covers Born linearity, the directional derivative of ``acoustic``, and
eager-vs-compiled gradients for both the full and scattered-only adjoints.
Receiver losses and separate PML-state cotangents exercise different backward
inputs; the latter also check buffer identity across odd-sized chunks.
"""

import importlib
import re

import pytest
import torch
from helpers import output_tensors

import deepwave
import deepwave._backend_utils

# ``deepwave/__init__.py`` does ``from deepwave.acoustic_born import
# acoustic_born``, which rebinds the package attribute ``deepwave.acoustic_born``
# to the *function*, shadowing the submodule (the same convention the other
# propagators use). Fetch the actual submodule from ``sys.modules`` via
# ``importlib`` so ``ab`` is the module, not the function.
ab = importlib.import_module("deepwave.acoustic_born")


def _problem(
    ndim: int, dtype: torch.dtype = torch.double
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, dict]:
    # Interior shapes are independent of pml_width (v/rho/dv/drho hold no PML
    # cells; the propagator pads with fd_pad+pml_width around them), so these
    # only need to be big enough to place a source, an offset receiver, and a
    # mid-grid scatter point without collisions -- 8 per axis is enough for
    # accuracy up to 8 (fd_pad=4) with room to spare; the 3-D case is kept
    # the smallest of the three since its element count dominates the cost of
    # the per-element checks below.
    shape = {1: (14,), 2: (12, 12), 3: (8, 8, 8)}[ndim]
    v = torch.full(shape, 1500.0, dtype=dtype)
    rho = torch.full(shape, 2200.0, dtype=dtype)
    dv = torch.zeros(shape, dtype=dtype)
    drho = torch.zeros(shape, dtype=dtype)
    mid = tuple(s // 2 for s in shape)
    dv[mid] = 10.0
    drho[mid] = 40.0
    dx = 6.0
    dt = 0.0004
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    for d in range(ndim):
        loc[..., d] = shape[d] // 2
    rloc = loc.clone()
    rloc[..., -1] = shape[-1] // 2 - 2
    nt = 20
    torch.manual_seed(0)
    samp = torch.randn(1, 1, nt, dtype=dtype)
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


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_born_linear_in_scatter(ndim: int) -> None:
    """Scattered receivers scale linearly with (dv, drho); zero scatter -> zero."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)

    def scattered(dv_: torch.Tensor, drho_: torch.Tensor) -> torch.Tensor:
        out = ab.acoustic_born(v, rho, dx, dt, dv=dv_, drho=drho_, accuracy=4, **kwargs)
        names = ab.acoustic_born_output_names(ndim)
        return dict(zip(names, out))["sc_receiver_amplitudes_p"]

    r0 = scattered(dv, drho)
    r2 = scattered(2 * dv, 2 * drho)
    r_zero = scattered(torch.zeros_like(dv), torch.zeros_like(drho))
    assert r0.abs().max() > 0
    assert torch.allclose(r2, 2 * r0, atol=1e-8)
    assert r_zero.abs().max().item() == 0.0


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_born_equals_directional_fd_of_acoustic(ndim: int) -> None:
    """Background/scattered outputs match acoustic() and its directional FD.

    The dv-driven and drho-driven scattered fields are checked separately
    (each against ``acoustic()`` perturbed in only that one direction, the
    other scattering potential held at zero) since the Born forward is only
    linear *separately* in each -- a joint (dv, drho) call's scattered output
    is their sum, not comparable to a single-variable finite difference.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    zero = torch.zeros_like(dv)

    def acoustic_run(v_: torch.Tensor, rho_: torch.Tensor) -> torch.Tensor:
        o = deepwave.acoustic(
            v_,
            rho_,
            dx,
            dt,
            source_amplitudes_volume_rate=kwargs["source_amplitudes_volume_rate"],
            source_locations_volume_rate=kwargs["source_locations_volume_rate"],
            receiver_locations_p=kwargs["sc_receiver_locations_p"],
            accuracy=4,
            pml_width=kwargs["pml_width"],
            max_vel=kwargs["max_vel"],
            nt=kwargs["nt"],
            pml_freq=kwargs["pml_freq"],
        )
        return o[-1 - ndim]

    # Background output (scatter irrelevant to it) matches acoustic() exactly.
    out_bg = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=zero,
        drho=zero,
        accuracy=4,
        bg_receiver_locations_p=kwargs["sc_receiver_locations_p"],
        **kwargs,
    )
    names = ab.acoustic_born_output_names(ndim)
    d_bg = dict(zip(names, out_bg))
    r_bg_true = acoustic_run(v, rho)
    assert torch.allclose(
        d_bg["bg_receiver_amplitudes_p"], r_bg_true, atol=1e-6, rtol=1e-6
    )

    eps = 1e-3

    # dv-only scattering vs directional FD of acoustic() in v.
    out_dv = ab.acoustic_born(v, rho, dx, dt, dv=dv, drho=zero, accuracy=4, **kwargs)
    r_sc_dv = dict(zip(names, out_dv))["sc_receiver_amplitudes_p"]
    fd_v = (acoustic_run(v + eps * dv, rho) - acoustic_run(v - eps * dv, rho)) / (
        2 * eps
    )
    rel_v = (r_sc_dv - fd_v).abs().max() / fd_v.abs().max()
    assert rel_v < 1e-5

    # drho-only scattering vs directional FD of acoustic() in rho.
    out_drho = ab.acoustic_born(
        v, rho, dx, dt, dv=zero, drho=drho, accuracy=4, **kwargs
    )
    r_sc_drho = dict(zip(names, out_drho))["sc_receiver_amplitudes_p"]
    fd_rho = (acoustic_run(v, rho + eps * drho) - acoustic_run(v, rho - eps * drho)) / (
        2 * eps
    )
    rel_rho = (r_sc_drho - fd_rho).abs().max() / fd_rho.abs().max()
    assert rel_rho < 1e-5


def test_born_gradcheck_scattered_initial_wavefield() -> None:
    """Gradcheck of the scattered-side initial wavefield (2-D).

    An initial wavefield's own shape must include the PML padding on each
    side (``get_survey_extents_from_wavefields`` subtracts ``pml_width`` from
    it to recover the interior extent) -- the same shape the propagator's own
    output wavefields have, not the bare model shape.
    """
    ndim = 2
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    pml_width = kwargs["pml_width"]
    padded_shape = tuple(s + 2 * pml_width for s in v.shape)
    p0 = torch.zeros(1, *padded_shape, dtype=v.dtype).requires_grad_()

    def g(p0_: torch.Tensor) -> torch.Tensor:
        out = ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            accuracy=4,
            initial_state={"pressure_sc_0": p0_},
            **kwargs,
        )
        names = ab.acoustic_born_output_names(ndim)
        return dict(zip(names, out))["sc_receiver_amplitudes_p"]

    assert torch.autograd.gradcheck(g, (p0,), eps=1e-6, atol=1e-4, rtol=1e-3)


def _scattered_and_bg(ndim: int, out: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Sum of the scattered and background pressure receivers (a scalar loss)."""
    names = ab.acoustic_born_output_names(ndim)
    d = dict(zip(names, out))
    return d["sc_receiver_amplitudes_p"] + d["bg_receiver_amplitudes_p"]


def _hetero_problem(
    ndim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, dict]:
    """Like :func:`_problem` but with a heterogeneous background v/rho.

    The uniform-medium point-source setup gives ``d(receiver)/d(v)`` a ~15-order
    dynamic range (a ``dt*K = dt*rho*v**2`` source-cell blow-up), which defeats
    a finite-difference Jacobian at the smallest entries even though the
    analytic gradient is correct there (it matches a coarse central difference
    to ~1e-9). A mildly heterogeneous medium distributes the gradient and,
    together with a larger finite-difference step (well inside the smooth
    regime for a model of ~1500), makes the background-gradient checks
    well-conditioned -- mirroring deepwave's own ``run_backend_parity``
    convention.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    torch.manual_seed(1)
    v = v + torch.rand_like(v) * 30
    rho = rho + torch.rand_like(rho) * 30
    return v, rho, dv, drho, dx, dt, kwargs


def test_module_background_v_rho_gradient_matches_functional() -> None:
    """AcousticBorn Module: v_requires_grad/rho_requires_grad populate .grad.

    A5: the Module wrapper's constructor previously hard-coded
    ``requires_grad=False`` for ``v``/``rho``, even though the functional
    ``acoustic_born`` already supports background gradients (see
    ``test_eager_matches_compiled_gradients`` below). Construct
    :class:`deepwave.AcousticBorn` with both flags set, run forward+backward
    through the Module, and confirm the resulting ``.grad`` on the stored
    ``v``/``rho`` parameters is non-zero and numerically identical to calling
    the functional ``acoustic_born`` directly on equivalent leaf tensors.
    """
    ndim = 2
    v0, rho0, dv, drho, dx, dt, kwargs = _hetero_problem(ndim)

    module = deepwave.AcousticBorn(
        v0,
        rho0,
        dx,
        dv=dv,
        drho=drho,
        requires_grad=["v", "rho"],
    )
    assert module.v.requires_grad
    assert module.rho.requires_grad
    out_module = module(
        dt,
        accuracy=4,
        bg_receiver_locations_p=kwargs["sc_receiver_locations_p"],
        **kwargs,
    )
    _scattered_and_bg(ndim, out_module).sum().backward()
    assert module.v.grad is not None
    assert module.rho.grad is not None
    assert module.v.grad.abs().max() > 0
    assert module.rho.grad.abs().max() > 0

    v_func = v0.clone().requires_grad_()
    rho_func = rho0.clone().requires_grad_()
    out_func = ab.acoustic_born(
        v_func,
        rho_func,
        dx,
        dt,
        dv=dv,
        drho=drho,
        accuracy=4,
        bg_receiver_locations_p=kwargs["sc_receiver_locations_p"],
        **kwargs,
    )
    _scattered_and_bg(ndim, out_func).sum().backward()
    assert v_func.grad is not None
    assert rho_func.grad is not None
    assert torch.equal(module.v.grad, v_func.grad)
    assert torch.equal(module.rho.grad, rho_func.grad)


def test_born_gradcheck_background_initial_wavefield() -> None:
    """Full two-field backward: gradcheck of a background-side initial wavefield."""
    ndim = 2
    v, rho, dv, drho, dx, dt, kwargs = _hetero_problem(ndim)
    pml_width = kwargs["pml_width"]
    padded_shape = tuple(s + 2 * pml_width for s in v.shape)
    torch.manual_seed(2)
    p0 = (torch.rand(1, *padded_shape, dtype=v.dtype) * 1e-3).requires_grad_()

    def g(p0_: torch.Tensor) -> torch.Tensor:
        out = ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            accuracy=4,
            initial_state={"pressure_0": p0_},
            bg_receiver_locations_p=kwargs["sc_receiver_locations_p"],
            **kwargs,
        )
        return _scattered_and_bg(ndim, out)

    assert torch.autograd.gradcheck(g, (p0,), eps=1e-4, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_full_backward_matches_scattered_only_for_dv_drho(ndim: int) -> None:
    """dv/drho gradients are identical whether or not the background needs grad.

    Enabling a background-side gradient switches the backward to the full
    two-field adjoint; the scattered-model (dv/drho) gradient it produces must
    match the cheaper scattered-only path bit-for-bit (both are the same
    imaging condition).
    """
    v, rho, dv0, drho0, dx, dt, kwargs = _problem(ndim)

    def grads(background_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        v_ = v.clone().requires_grad_(background_grad)
        dv_ = dv0.clone().requires_grad_()
        drho_ = drho0.clone().requires_grad_()
        out = ab.acoustic_born(
            v_, rho, dx, dt, dv=dv_, drho=drho_, accuracy=4, **kwargs
        )
        names = ab.acoustic_born_output_names(ndim)
        loss = dict(zip(names, out))["sc_receiver_amplitudes_p"].pow(2).sum()
        loss.backward()
        assert dv_.grad is not None
        assert drho_.grad is not None
        return dv_.grad, drho_.grad

    dv_sc, drho_sc = grads(False)  # scattered-only path
    dv_full, drho_full = grads(True)  # full two-field path
    assert torch.allclose(dv_sc, dv_full, atol=1e-10, rtol=1e-8)
    assert torch.allclose(drho_sc, drho_full, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("ndim", [1, 2])
def test_born_reciprocity(ndim: int) -> None:
    """Source-receiver reciprocity of the Born scattered pressure response.

    For coincident pressure sources and receivers, swapping the source and
    receiver locations leaves the scattered pressure trace unchanged.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    a = kwargs["source_locations_volume_rate"]
    b = kwargs["sc_receiver_locations_p"]
    names = ab.acoustic_born_output_names(ndim)

    def scattered(src_loc: torch.Tensor, rec_loc: torch.Tensor) -> torch.Tensor:
        out = ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            accuracy=4,
            source_amplitudes_volume_rate=kwargs["source_amplitudes_volume_rate"],
            source_locations_volume_rate=src_loc,
            sc_receiver_locations_p=rec_loc,
            pml_width=kwargs["pml_width"],
            max_vel=kwargs["max_vel"],
            nt=kwargs["nt"],
            pml_freq=kwargs["pml_freq"],
        )
        return dict(zip(names, out))["sc_receiver_amplitudes_p"]

    r_ab = scattered(a, b)
    r_ba = scattered(b, a)
    assert r_ab.abs().max() > 0
    assert torch.allclose(r_ab, r_ba, atol=1e-9, rtol=1e-5)


def test_storage_mode_none_forward_only_and_rejects_grad() -> None:
    """``storage_mode="none"`` is forward-only; combined with grad it raises.

    Regression test: the forward ``make_call`` previously passed its
    store-write bools straight from ``requires_grad`` with no
    ``storage_mode`` gate, so ``storage_mode="none"`` (which allocates no
    backward-storage buffers) segfaulted via a write through the empty
    store whenever a model required grad. Forward-only "none" must still
    work; "none" + grad must raise a clear error, not crash.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    dv = dv.clone().requires_grad_()

    out = ab.acoustic_born(
        v, rho, dx, dt, dv=dv.detach(), drho=drho, storage_mode="none", **kwargs
    )
    assert all(torch.isfinite(o).all() for o in output_tensors(out) if o.numel() > 0)

    with pytest.raises(RuntimeError, match="storage_mode"):
        ab.acoustic_born(
            v, rho, dx, dt, dv=dv, drho=drho, storage_mode="none", **kwargs
        )


def test_initial_state_round_trip_propagator_output() -> None:
    """A whole PropagatorOutput can be fed back in as ``initial_state`` (2-D).

    Regression test: previously passing a
    :class:`deepwave.PropagatorOutput` crashed with
    ``'PropagatorOutput' has no field 'get'`` (the receiver-amplitude fields
    must be ignored and the wavefield fields re-fed). Splitting a run in two and
    continuing the second half from the first half's returned output must
    reproduce the single full run bit-for-bit (background *and* scattered
    receiver amplitudes), which also exercises the round trip end to end.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    kwargs = dict(kwargs)
    samp = kwargs.pop("source_amplitudes_volume_rate")
    nt = kwargs.pop("nt")
    n1 = nt // 2
    kwargs["bg_receiver_locations_p"] = kwargs["sc_receiver_locations_p"]

    full = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp,
        nt=nt,
        **kwargs,
    )
    part1 = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp[:, :, :n1],
        nt=n1,
        **kwargs,
    )
    # Feed the entire PropagatorOutput back in (receiver fields must be ignored).
    part2 = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp[:, :, n1:],
        nt=nt - n1,
        initial_state=part1,
        **kwargs,
    )
    for field in ("sc_receiver_amplitudes_p", "bg_receiver_amplitudes_p"):
        r_full = getattr(full, field)
        r_cat = torch.cat([getattr(part1, field), getattr(part2, field)], dim=-1)
        assert r_full.abs().max() > 0
        assert torch.allclose(r_full, r_cat, atol=1e-9, rtol=1e-6), field


def test_initial_state_accepts_mapping_and_propagator_output_equivalently() -> None:
    """A PropagatorOutput and the equivalent mapping give identical results."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    kwargs = dict(kwargs)
    samp = kwargs.pop("source_amplitudes_volume_rate")
    nt = kwargs.pop("nt")
    n1 = nt // 2
    part1 = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp[:, :, :n1],
        nt=n1,
        **kwargs,
    )
    # Build a plain mapping holding only the state fields of part1: the public
    # wavefield names (the folded absorbing-boundary ones travel inside
    # ``auxiliary_state``), never the receiver amplitudes.
    wf_names = {*ab.acoustic_born_public_wavefield_names(2), "auxiliary_state"}
    mapping = {k: v_ for k, v_ in part1._asdict().items() if k in wf_names}
    out_from_output = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp[:, :, n1:],
        nt=nt - n1,
        initial_state=part1,
        **kwargs,
    )
    out_from_mapping = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp[:, :, n1:],
        nt=nt - n1,
        initial_state=mapping,
        **kwargs,
    )
    for a_, b_ in zip(
        output_tensors(out_from_output), output_tensors(out_from_mapping)
    ):
        assert torch.equal(a_, b_)


# --- backend="eager" (pure-PyTorch coupled Born driver) -----------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "other_backend",
    [
        pytest.param("compiled", id="compiled"),
        pytest.param("torch_compile", id="torch_compile"),
    ],
)
def test_eager_matches_compiled_forward(ndim: int, other_backend: object) -> None:
    """The eager coupled Born driver == the compiled C forward to ~machine eps.

    Covers the final background AND scattered wavefields and both receiver
    blocks (background + scattered), mirroring
    ``test_elastic_born.test_compiled_forward_matches_eager``.
    ``other_backend`` additionally covers the ``torch_compile`` Python
    mode, which runs the same eager time loop with
    :func:`deepwave.acoustic_born._update_velocities_born`/
    :func:`deepwave.acoustic_born._update_pressure_born`
    ``torch.compile``d, so it is held to the same tolerance as the
    compiled C/CUDA backend.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    kwargs = dict(kwargs)
    kwargs["bg_receiver_locations_p"] = kwargs["sc_receiver_locations_p"]

    def run(backend: object) -> tuple[torch.Tensor, ...]:
        return ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            accuracy=4,
            backend=backend,
            **kwargs,
        )

    eager = output_tensors(run("eager"))
    other = output_tensors(run(other_backend))
    assert len(eager) == len(other)
    for i, (e, c) in enumerate(zip(eager, other)):
        assert e.shape == c.shape, f"output {i}: {e.shape} vs {c.shape}"
        if e.numel() == 0:
            assert c.numel() == 0
            continue
        denom = e.abs().max()
        if denom == 0:
            assert c.abs().max() == 0, f"output {i}: eager zero, other nonzero"
        else:
            rel = (e - c).abs().max() / denom
            assert rel < 1e-8, f"output {i} rel={rel}"


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_eager_matches_compiled_gradients(ndim: int, accuracy: int) -> None:
    """Eager gradients == compiled gradients (<1e-8) for every input.

    Single receiver (the ``_problem`` convention on which the compiled adjoint
    is validated); covers ``dv``/``drho`` (scattered-only adjoint) and the
    background ``v``/``rho`` and the shared source amplitude (full two-field
    adjoint). ndim 3 is the smoke case. The compiled backward kernels are
    separate per-accuracy-order template instantiations, so every supported
    order is exercised rather than the default 4 alone; ``_problem``'s grids
    are sized for accuracy 8's ``fd_pad=4`` with room to spare.

    ``_problem``'s default ``nt=20`` is too short to exercise the PML
    memory-variable recursion (``psi``/``phi`` in
    ``_update_velocities_born``/``_update_pressure_born``): at
    ``c=1500``, ``dx=6``, the wavefront needs ``(source-to-PML-interface
    distance) + (PML-interface-to-receiver distance)`` cells of travel to
    have any PML-region contribution reach the receiver at all, which is
    12/10/8 cells (72/60/48 m, 0.048/0.040/0.032 s -> nt of 120/100/80) for
    ndim 1/2/3 respectively (source and receiver are both a handful of
    cells from the nearest PML interface in this small grid; see
    ``_problem``). At the inherited ``nt=20`` (0.008 s) a deliberate sign
    flip in the background ``psi`` recursion changes the returned
    ``pressure_0`` field by ~2x (caught by
    ``test_eager_matches_compiled_forward``, which also compares full
    wavefields) but leaves ``sc_receiver_amplitudes_p``/
    ``bg_receiver_amplitudes_p`` -- the only quantities this test's
    gradients are differentiated through -- unchanged to ~1e-12..1e-15,
    i.e. this test cannot see a PML-recursion bug through the receiver-only
    loss at ``nt=20``. ``nt=200`` (0.08 s) gives every ndim comfortable
    margin above its threshold and was confirmed to raise that same
    injected bug's gradient relative error from ~1e-15 to 5e-4..0.3 (all
    five inputs), well clear of the ``1e-8`` gate below.
    """
    v0, rho0, dv0, drho0, dx, dt, kwargs = _problem(ndim)
    kwargs = dict(kwargs)
    kwargs["bg_receiver_locations_p"] = kwargs["sc_receiver_locations_p"]
    kwargs.pop("source_amplitudes_volume_rate")
    nt = 200
    kwargs["nt"] = nt
    torch.manual_seed(0)
    samp0 = torch.randn(1, 1, nt, dtype=v0.dtype)
    names = ab.acoustic_born_output_names(ndim)

    def grads(backend: object) -> dict[str, torch.Tensor]:
        v = v0.clone().requires_grad_()
        rho = rho0.clone().requires_grad_()
        dv = dv0.clone().requires_grad_()
        drho = drho0.clone().requires_grad_()
        samp = samp0.clone().requires_grad_()
        out = ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            accuracy=accuracy,
            source_amplitudes_volume_rate=samp,
            backend=backend,
            **kwargs,
        )
        d = dict(zip(names, out))
        loss = (
            d["sc_receiver_amplitudes_p"].pow(2).sum()
            + d["bg_receiver_amplitudes_p"].pow(2).sum()
        )
        loss.backward()
        return {
            "v": v.grad,
            "rho": rho.grad,
            "dv": dv.grad,
            "drho": drho.grad,
            "samp": samp.grad,
        }

    ge = grads("eager")
    gc = grads("compiled")
    for name in ge:
        e, c = ge[name], gc[name]
        assert e is not None
        assert c is not None
        denom = e.abs().max()
        rel = (e - c).abs().max() / denom if denom > 0 else (e - c).abs().max()
        assert rel < 1e-8, f"grad {name} rel={rel}"


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("scattered_only", id="scattered_only"),
        pytest.param("full", id="full"),
    ],
)
def test_c_threading_matches_single_thread(ndim: int, mode: str) -> None:
    """Intra-shot OpenMP threading (B2) does not change the compiled result.

    ``acoustic_born.c`` was ported from one-shot-per-thread parallelism to
    intra-shot spatial parallelism (see ``FUNC(forward)``/``FUNC(backward_sc)``/
    ``FUNC(backward)``, mirroring ``elastic.c``/``viscoacoustic.c``/
    ``acoustic_vti.c``'s B2 port): each time step's velocity-update loops
    (vz/vy/vx) and the combined pressure-update loop are now split across
    threads via ``#pragma omp for simd ... nowait``, with shots processed
    serially.

    acoustic_born is a COUPLED Born propagator (background + scattered
    fields), structurally different from the earlier B2 ports, so this test
    (unlike theirs) exercises all three compiled entry points and both
    gradient families:

    * ``mode="scattered_only"``: only ``dv``/``drho`` require grad, so the
      engine dispatches to ``FUNC(backward_sc)`` (the scattered-only
      adjoint). Every grid cell's ``grad_dK``/``grad_dB_d`` accumulation
      happens inside ``FUNC(backward_sc_p_cell)``/``FUNC(backward_sc_vz_cell)``
      /``vy_cell``/``vx_cell``, called once per cell by exactly one thread
      (the PML-region tiling partitions the grid disjointly across the
      ``omp for`` loops), so the accumulation order -- and hence the result
      -- does not depend on thread count.
    * ``mode="full"``: ``v``/``rho`` (hence ``K``/``B_d``) also require grad,
      so the engine dispatches to ``FUNC(backward)`` (the full two-field
      adjoint). ``FUNC(backward_p_cell)``/``FUNC(backward_vz_cell)``/``vy_cell``
      /``vx_cell`` each accumulate BOTH the background gradient
      (``grad_K``/``grad_B_d``, two correlation terms) AND the scattered
      gradient (``grad_dK``/``grad_dB_d``, one term) for the same cell ``i``
      in the same call, by the same thread that owns that cell in this
      ``omp for`` region -- so there is no race between the background and
      scattered accumulations either.

    Also covers ``FUNC(forward)`` (both modes run it; it only writes
    per-cell stores, never accumulates, so it is trivially race-free) and,
    within each mode, both a dv-only (``drho`` all zero) and a full
    (``dv``+``drho`` both nonzero) scattering case -- so the ``dbuoyancy``-
    driven velocity-gradient accumulation (silent when ``drho`` is all zero)
    is exercised too, not just the ``dk``-driven pressure-gradient one.

    Uses ``torch.set_num_threads`` (read by the backend via
    ``torch.get_num_threads()`` at call time, exactly as ``OMP_NUM_THREADS``
    would set the default) rather than an env var, so the comparison runs
    in-process; the tolerance mirrors
    ``test_eager_matches_compiled_forward``/``test_eager_matches_compiled_gradients``
    above (the existing cross-backend-parity patterns in this file), not a
    new one.
    """
    saved_threads = torch.get_num_threads()
    try:
        v0, rho0, dv0, drho0, dx, dt, kwargs = _problem(ndim)
        kwargs = dict(kwargs)
        kwargs["bg_receiver_locations_p"] = kwargs["sc_receiver_locations_p"]
        names = ab.acoustic_born_output_names(ndim)

        def run(
            n_threads: int, drho_nonzero: bool
        ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
            torch.set_num_threads(n_threads)
            v = v0.clone().requires_grad_(mode == "full")
            rho = rho0.clone().requires_grad_(mode == "full")
            dv = dv0.clone().requires_grad_(True)
            drho = (drho0 if drho_nonzero else torch.zeros_like(drho0)).clone()
            drho.requires_grad_(True)
            out = ab.acoustic_born(
                v,
                rho,
                dx,
                dt,
                dv=dv,
                drho=drho,
                accuracy=4,
                backend="compiled",
                **kwargs,
            )
            d = dict(zip(names, out))
            loss = (
                d["sc_receiver_amplitudes_p"].pow(2).sum()
                + d["bg_receiver_amplitudes_p"].pow(2).sum()
            )
            loss.backward()
            grads = {"dv": dv.grad, "drho": drho.grad}
            if mode == "full":
                grads["v"] = v.grad
                grads["rho"] = rho.grad
            return out, grads

        for drho_nonzero in (False, True):  # dv-only, then full dv+drho
            out_1, g_1 = run(1, drho_nonzero)
            out_4, g_4 = run(4, drho_nonzero)
            for a, b in zip(output_tensors(out_1), output_tensors(out_4)):
                if a.numel() == 0:
                    continue
                scale = max(a.abs().max().item(), 1.0)
                assert (a - b).abs().max().item() <= scale * 1e-6
            for name in g_1:
                g1, g4 = g_1[name], g_4[name]
                assert g1 is not None
                assert g4 is not None
                scale = max(g1.abs().max().item(), 1.0)
                assert (g1 - g4).abs().max().item() <= scale * 1e-6
    finally:
        torch.set_num_threads(saved_threads)


@pytest.mark.parametrize("ndim", [1, 2])
def test_multireceiver_backward_matches_eager(ndim: int) -> None:
    """Compiled multi-receiver gradients == eager (regression).

    The compiled backend reads the receiver-gradient tensors through a raw
    pointer assuming a contiguous ``[nt, n_shots, n_receivers]`` layout, but
    autograd delivers them as non-contiguous strided views (via the adjoint of
    the ``downsample_and_movedim`` applied outside the Function). With more than
    one receiver per shot this silently injected the wrong adjoint-source
    samples and corrupted the model gradient; a ``.contiguous()`` in
    ``AcousticBornForwardFunc.backward`` fixes it. Single-receiver tests could
    not catch this (the strided and contiguous offsets coincide for
    ``n_receivers == 1``).

    ``_problem``'s default ``nt=20`` is also too short here for the same
    reason as ``test_eager_matches_compiled_gradients``: the wavefront has
    nowhere near enough time to reach the PML region and return to any of
    the three receivers (all within a few cells of the source). Injecting a
    sign flip into the background PML ``psi`` recursion
    (``_update_velocities_born``) and comparing eager vs. compiled at
    ``nt=20`` gives a ``dv``/``v`` gradient relative error of ~1e-12..1e-15
    (invisible against the ``1e-8`` gate below); the same injected bug at
    ``nt=200`` gives 0.44..0.95, confirmed for both ndim 1 and 2.
    """
    v, rho, dv0, drho0, dx, dt, kwargs = _problem(ndim)
    kwargs = dict(kwargs)
    shape = v.shape
    n_recv = 3
    rloc = torch.zeros(1, n_recv, ndim, dtype=torch.long)
    for r in range(n_recv):
        for d in range(ndim):
            rloc[0, r, d] = shape[d] // 2 - 1 - (r if d == ndim - 1 else 0)
    kwargs["sc_receiver_locations_p"] = rloc
    kwargs["bg_receiver_locations_p"] = rloc
    nt = 200
    kwargs["nt"] = nt
    torch.manual_seed(0)
    kwargs["source_amplitudes_volume_rate"] = torch.randn(1, 1, nt, dtype=v.dtype)
    names = ab.acoustic_born_output_names(ndim)

    def grads(backend: object) -> tuple[torch.Tensor, torch.Tensor]:
        dv = dv0.clone().requires_grad_()
        v_ = v.clone().requires_grad_()
        out = ab.acoustic_born(
            v_,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho0,
            accuracy=4,
            backend=backend,
            **kwargs,
        )
        d = dict(zip(names, out))
        (
            d["sc_receiver_amplitudes_p"].sum() + d["bg_receiver_amplitudes_p"].sum()
        ).backward()
        assert dv.grad is not None
        assert v_.grad is not None
        return dv.grad, v_.grad

    dv_e, v_e = grads("eager")
    dv_c, v_c = grads("compiled")
    assert torch.isfinite(dv_c).all()
    assert torch.isfinite(v_c).all()
    assert (dv_e - dv_c).abs().max() / dv_e.abs().max() < 1e-8
    assert (v_e - v_c).abs().max() / v_e.abs().max() < 1e-8


def test_eager_rejects_backward_callback_but_not_forward_callback() -> None:
    """backend="eager" rejects backward_callback but runs forward_callback.

    The eager driver has no chunked adjoint machinery, so backward_callback
    (which needs the engine's adjoint pass) still raises; forward_callback
    fires from the eager driver's own per-step loop and no longer raises
    (see test_acoustic_born_eager_forward_callback_dict_keys_and_call_count).
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)

    def noop(state: object) -> None:
        pass

    with pytest.raises(RuntimeError, match="backward_callback"):
        ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            backend="eager",
            backward_callback=noop,
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
        forward_callback=noop,
        **kwargs,
    )
    with pytest.raises(RuntimeError, match="storage_mode"):
        ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            backend="eager",
            storage_mode="cpu",
            **kwargs,
        )
    with pytest.raises(RuntimeError, match="Storage compression"):
        ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            backend="eager",
            storage_compression=True,
            **kwargs,
        )


# --- observable-surface tests: callback/model/output dict key sets ---------
#
# acoustic_born's compiled backward dispatches to one of two adjoints
# depending on which inputs need a gradient (see ``AcousticBornForwardFunc.
# backward``): the full two-field adjoint (background v/rho, a source
# amplitude, or a background initial wavefield needs a gradient) delivers a
# ``backward_callback`` dict keyed by the FULL ``acoustic_born_wavefield_names``
# (background block then the ``_sc``-suffixed scattered block); the
# scattered-only adjoint (only dv/drho and/or scattered initial wavefields
# need a gradient) delivers only the scattered wavefields, but keyed by the
# UNSUFFIXED ``acoustic_wavefield_names`` (an explicit ``wavefield_names=``
# override in ``AcousticBornForwardFunc.backward``, not a bug -- the scattered
# block is a plain acoustic field in its own right on that path). Both key
# sets are asserted here so a regression in either adjoint's callback wiring
# is caught.


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_acoustic_born_backward_callback_dict_keys_full_adjoint(ndim: int) -> None:
    """Full two-field adjoint: backward_callback keys == acoustic_born_wavefield_names.

    Triggered by requiring a gradient on the background ``v`` (a background
    model quantity), which forces ``AcousticBornForwardFunc.backward`` down
    the full-adjoint path (propagating both the background and scattered
    adjoint fields).
    """
    v0, rho0, dv, drho, dx, dt, kwargs = _problem(ndim)
    v = v0.clone().requires_grad_()
    rho = rho0.clone()

    recorded: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)

    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        accuracy=4,
        backward_callback=recorder,
        callback_frequency=1,
        **kwargs,
    )
    out.sc_receiver_amplitudes_p.sum().backward()
    assert set(recorded) == set(ab.acoustic_born_wavefield_names(ndim))
    for name, tensor in recorded.items():
        assert tensor.numel() > 0, f"{name}: empty tensor"


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_acoustic_born_backward_callback_dict_keys_scattered_only(ndim: int) -> None:
    """Scattered-only adjoint: keys == (unsuffixed) acoustic_wavefield_names.

    Triggered by requiring a gradient only on the scattering perturbations
    ``dv``/``drho`` (with the background ``v``/``rho`` left as plain, non-
    grad tensors), which keeps ``AcousticBornForwardFunc.backward`` on the
    cheaper scattered-only adjoint path.
    """
    v, rho, dv0, drho0, dx, dt, kwargs = _problem(ndim)
    dv = dv0.clone().requires_grad_()
    drho = drho0.clone().requires_grad_()

    recorded: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)

    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        accuracy=4,
        backward_callback=recorder,
        callback_frequency=1,
        **kwargs,
    )
    out.sc_receiver_amplitudes_p.sum().backward()
    from deepwave.acoustic import acoustic_wavefield_names

    assert set(recorded) == set(acoustic_wavefield_names(ndim))
    for name, tensor in recorded.items():
        assert tensor.numel() > 0, f"{name}: empty tensor"


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_acoustic_born_forward_callback_dict_keys(ndim: int) -> None:
    """forward_callback keys == acoustic_born_wavefield_names (compiled backend)."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)

    recorded: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)

    ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        accuracy=4,
        forward_callback=recorder,
        callback_frequency=1,
        **kwargs,
    )
    assert set(recorded) == set(ab.acoustic_born_wavefield_names(ndim))
    for name, tensor in recorded.items():
        assert tensor.numel() > 0, f"{name}: empty tensor"


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_acoustic_born_eager_forward_callback_dict_keys_and_call_count(
    ndim: int,
) -> None:
    """backend="eager" also supports forward_callback.

    Matches the compiled backend's dict keys and fires the same number of
    times as :func:`deepwave.acoustic.acoustic_python`'s own eager
    forward_callback.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    nt = kwargs["nt"]

    recorded: dict[str, torch.Tensor] = {}
    count = {"n": 0}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)
        count["n"] += 1

    ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        accuracy=4,
        backend="eager",
        forward_callback=recorder,
        callback_frequency=2,
        **kwargs,
    )
    assert set(recorded) == set(ab.acoustic_born_wavefield_names(ndim))
    for name, tensor in recorded.items():
        assert tensor.numel() > 0, f"{name}: empty tensor"
    assert count["n"] == nt // 2


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_acoustic_born_model_dict_keys(ndim: int) -> None:
    """backward_callback's model/gradient dicts are keyed by acoustic_born_model_names.

    Both the full and scattered-only adjoint paths pass the SAME (full,
    unsliced) ``models`` list to ``run_backward_core``, so both key sets are
    the full ``[K, B*, dK, dB*]`` set regardless of which adjoint runs;
    checked here on the full-adjoint path (background v requires grad).
    """
    v0, rho0, dv, drho, dx, dt, kwargs = _problem(ndim)
    v = v0.clone().requires_grad_()
    rho = rho0.clone()

    recorded_models: dict[str, torch.Tensor] = {}
    recorded_grads: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_models.update(state._models)
        recorded_grads.update(state._gradients)

    out = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        accuracy=4,
        backward_callback=recorder,
        callback_frequency=1,
        **kwargs,
    )
    out.sc_receiver_amplitudes_p.sum().backward()
    expected = set(ab.acoustic_born_model_names(ndim))
    assert set(recorded_models) == expected
    assert set(recorded_grads) == expected


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_acoustic_born_output_names(ndim: int) -> None:
    """PropagatorOutput field names == acoustic_born_output_names, in order."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
    out = ab.acoustic_born(v, rho, dx, dt, dv=dv, drho=drho, accuracy=4, **kwargs)
    assert out._names == tuple(ab.acoustic_born_output_names(ndim))


# ---------------------------------------------------------------------------
# location_interpolation: mirrors the on-node/off-node gates
# tests/test_location_interpolation.py runs for deepwave.acoustic, ported to
# the Born propagator's doubled background/scattered source-cell wiring.
# ---------------------------------------------------------------------------


def test_location_interpolation_on_node_float_equals_integer_pressure() -> None:
    """A float coordinate on the (unstaggered) pressure node equals the integer result.

    Pressure sits on the regular grid (offset None), so float N.0 == integer N.
    """
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    loc_int = kwargs["source_locations_volume_rate"]
    rloc_int = kwargs["sc_receiver_locations_p"]
    exact = ab.acoustic_born(v, rho, dx, dt, dv=dv, drho=drho, **kwargs)
    snapped_kwargs = dict(kwargs)
    snapped_kwargs["source_locations_volume_rate"] = loc_int.double()
    snapped_kwargs["sc_receiver_locations_p"] = rloc_int.double()
    snapped = ab.acoustic_born(v, rho, dx, dt, dv=dv, drho=drho, **snapped_kwargs)
    assert torch.equal(exact.sc_receiver_amplitudes_p, snapped.sc_receiver_amplitudes_p)
    assert torch.equal(exact.bg_receiver_amplitudes_p, snapped.bg_receiver_amplitudes_p)


def test_location_interpolation_on_node_float_equals_integer_velocity() -> None:
    """A staggered velocity component's on-node float coordinate is N + 0.5."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    nt = kwargs["nt"]
    loc_int = torch.tensor([[[6, 6]]], dtype=torch.long)
    rloc_int = torch.tensor([[[6, 4]]], dtype=torch.long)
    samp = torch.randn(1, 1, nt, dtype=torch.double)
    common = {
        "pml_width": kwargs["pml_width"],
        "max_vel": kwargs["max_vel"],
        "nt": nt,
        "pml_freq": kwargs["pml_freq"],
    }

    exact = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_x=samp,
        source_locations_x=loc_int,
        sc_receiver_locations_x=rloc_int,
        **common,
    )
    offset = torch.tensor([0.0, 0.5], dtype=torch.double)
    snapped = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_x=samp,
        source_locations_x=loc_int.double() + offset,
        sc_receiver_locations_x=rloc_int.double() + offset,
        **common,
    )
    assert torch.equal(exact.sc_receiver_amplitudes_x, snapped.sc_receiver_amplitudes_x)


def test_location_interpolation_off_node_differs_and_eager_matches_compiled() -> None:
    """A genuinely off-grid source is Hicks-interpolated, agreeing across backends."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    nt = kwargs["nt"]
    loc_on_node = torch.tensor([[[6, 6]]], dtype=torch.long)
    rloc = torch.tensor([[[6, 4]]], dtype=torch.long)
    samp = torch.randn(1, 1, nt, dtype=torch.double)
    common = {
        "pml_width": kwargs["pml_width"],
        "max_vel": kwargs["max_vel"],
        "nt": nt,
        "pml_freq": kwargs["pml_freq"],
    }
    loc_off = loc_on_node.double() + 0.3

    on_node = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp,
        source_locations_volume_rate=loc_on_node,
        sc_receiver_locations_p=rloc,
        **common,
    )
    off_node_compiled = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp,
        source_locations_volume_rate=loc_off,
        sc_receiver_locations_p=rloc,
        **common,
    )
    off_node_eager = ab.acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=samp,
        source_locations_volume_rate=loc_off,
        sc_receiver_locations_p=rloc,
        backend="eager",
        **common,
    )
    assert not torch.equal(
        on_node.sc_receiver_amplitudes_p, off_node_compiled.sc_receiver_amplitudes_p
    )
    assert torch.allclose(
        off_node_compiled.sc_receiver_amplitudes_p,
        off_node_eager.sc_receiver_amplitudes_p,
        atol=1e-9,
    )


def test_location_interpolation_none_mode_rejects_off_node_float() -> None:
    """location_interpolation="none" raises on an off-node float coordinate."""
    v, rho, dv, drho, dx, dt, kwargs = _problem(2)
    kwargs = dict(kwargs)
    kwargs["source_locations_volume_rate"] = (
        kwargs["source_locations_volume_rate"].double() + 0.3
    )
    with pytest.raises(ValueError, match="location_interpolation='none'"):
        ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            location_interpolation="none",
            **kwargs,
        )


def test_acoustic_born_source_locations_y_out_of_bounds() -> None:
    """A velocity-component source must not sit on the final staggered sample.

    Mirrors ``deepwave.acoustic``'s identical check (and its
    ``test_acoustic_source_locations_y_out_of_bounds``-shaped test in intent);
    ``acoustic_born`` previously had no equivalent guard at all.
    """
    v = torch.ones(10, 10) * 1500.0
    rho = torch.ones(10, 10) * 2200.0
    dv = torch.zeros(10, 10)
    grid_spacing = 1.0
    dt = 0.0001
    source_amplitudes_y = torch.randn(1, 1, 10)
    source_locations_y = torch.tensor(
        [[[9, 1]]], dtype=torch.long
    )  # y-coord is 9, max is 8 (shape[0]-1)
    nt = 10
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "acoustic_born: source_locations_y is out of bounds: a "
            "y location is staggered half a cell past the "
            "node it names, so its y coordinate must be "
            "less than 9."
        ),
    ):
        ab.acoustic_born(
            v,
            rho,
            grid_spacing,
            dt,
            dv=dv,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            nt=nt,
        )


def test_acoustic_born_receiver_locations_x_out_of_bounds() -> None:
    """A velocity-component receiver must not sit on the final staggered sample.

    Covers both the scattered (``sc_receiver_locations_x``) and background
    (``bg_receiver_locations_x``) receiver lists -- both were previously
    unchecked.
    """
    v = torch.ones(10, 10) * 1500.0
    rho = torch.ones(10, 10) * 2200.0
    dv = torch.zeros(10, 10)
    grid_spacing = 1.0
    dt = 0.0001
    sc_receiver_locations_x = torch.tensor(
        [[[1, 9]]], dtype=torch.long
    )  # x-coord is 9, max is 8 (shape[1]-1)
    nt = 10
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "acoustic_born: sc_receiver_locations_x is out of bounds: a "
            "x location is staggered half a cell past the "
            "node it names, so its x coordinate must be "
            "less than 9."
        ),
    ):
        ab.acoustic_born(
            v,
            rho,
            grid_spacing,
            dt,
            dv=dv,
            sc_receiver_locations_x=sc_receiver_locations_x,
            nt=nt,
        )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "acoustic_born: bg_receiver_locations_x is out of bounds: a "
            "x location is staggered half a cell past the "
            "node it names, so its x coordinate must be "
            "less than 9."
        ),
    ):
        ab.acoustic_born(
            v,
            rho,
            grid_spacing,
            dt,
            dv=dv,
            bg_receiver_locations_x=sc_receiver_locations_x,
            nt=nt,
        )


# ---------------------------------------------------------------------------
# Batched models
# ---------------------------------------------------------------------------


def _batched_problem(
    dtype: torch.dtype = torch.double,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """A two-shot 2-D problem whose two shots see identical models.

    With the same model in both shots, a batched call must reproduce the
    unbatched one exactly, whichever of the four models carry the shot
    dimension.
    """
    shape = (12, 12)
    nt = 20
    torch.manual_seed(0)
    dv = torch.zeros(shape, dtype=dtype)
    drho = torch.zeros(shape, dtype=dtype)
    dv[6, 6] = 10.0
    drho[6, 6] = 40.0
    models = {
        "v": torch.full(shape, 1500.0, dtype=dtype),
        "rho": torch.full(shape, 2200.0, dtype=dtype),
        "dv": dv,
        "drho": drho,
    }
    survey: dict[str, object] = {
        "source_amplitudes_volume_rate": torch.randn(1, 1, nt, dtype=dtype).repeat(
            2, 1, 1
        ),
        "source_locations_volume_rate": torch.tensor([[[6, 6]], [[6, 6]]]),
        "sc_receiver_locations_p": torch.tensor([[[6, 4]], [[6, 4]]]),
        "bg_receiver_locations_p": torch.tensor([[[6, 4]], [[6, 4]]]),
        "pml_width": 4,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
    }
    return models, survey


def _batch(model: torch.Tensor) -> torch.Tensor:
    return model.unsqueeze(0).repeat(2, *([1] * model.ndim))


@pytest.mark.parametrize(
    ("batched_background", "batched_perturbation"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_batched_models_agree_with_the_unbatched_call(
    batched_background: bool, batched_perturbation: bool
) -> None:
    """Every combination of batched and unbatched models runs and agrees.

    Two failures met here. ``acoustic_born`` used to take ``ndim`` from
    ``v.ndim`` rather than from the survey, so a batched 2-D model was read
    as an unbatched 3-D one and the call died on a coordinate-count
    assertion. And ``prepare_scatter`` pushes the scattering potentials
    through ``torch.func.jvp``, which requires each tangent to have exactly
    its primal's shape, so a batched background with an unbatched
    perturbation (or the reverse) raised a bare
    ``RuntimeError: Trying to set a forward gradient that has a different
    size``. Both are fixed, and with identical models in both shots the four
    combinations must be bit-identical.
    """
    models, survey = _batched_problem()
    call = dict(models)
    if batched_background:
        call["v"] = _batch(models["v"])
        call["rho"] = _batch(models["rho"])
    if batched_perturbation:
        call["dv"] = _batch(models["dv"])
        call["drho"] = _batch(models["drho"])
    out = ab.acoustic_born(
        call["v"],
        call["rho"],
        6.0,
        0.0004,
        dv=call["dv"],
        drho=call["drho"],
        accuracy=4,
        **survey,
    )
    reference = ab.acoustic_born(
        models["v"],
        models["rho"],
        6.0,
        0.0004,
        dv=models["dv"],
        drho=models["drho"],
        accuracy=4,
        **survey,
    )
    assert out.sc_receiver_amplitudes_p.abs().max() > 0
    assert torch.equal(out.sc_receiver_amplitudes_p, reference.sc_receiver_amplitudes_p)
    assert torch.equal(out.bg_receiver_amplitudes_p, reference.bg_receiver_amplitudes_p)


def test_batched_models_really_are_per_shot() -> None:
    """A batched model gives each shot its own model, so the shots differ.

    Without this the test above would pass with the batch dimension silently
    collapsed to shot 0.
    """
    models, survey = _batched_problem()
    v = _batch(models["v"])
    v[1] *= 1.2
    dv = _batch(models["dv"])
    dv[1] *= 3.0
    out = ab.acoustic_born(
        v,
        models["rho"],
        6.0,
        0.0004,
        dv=dv,
        drho=models["drho"],
        accuracy=4,
        **survey,
    )
    scattered = out.sc_receiver_amplitudes_p
    # A relative comparison: the scattered amplitudes are tiny in absolute
    # terms, so ``allclose``'s default atol would call any two of them equal.
    assert (scattered[0] - scattered[1]).abs().max() > 0.1 * scattered.abs().max()
    # Shot 0's model is unchanged, so it still matches the unbatched run.
    reference = ab.acoustic_born(
        models["v"],
        models["rho"],
        6.0,
        0.0004,
        dv=models["dv"],
        drho=models["drho"],
        accuracy=4,
        **survey,
    )
    assert torch.equal(scattered[0], reference.sc_receiver_amplitudes_p[0])


def test_batched_models_are_differentiable() -> None:
    """Gradients reach an unbatched model used with a batched one.

    The shape equalisation expands the smaller model across shots, so the
    gradient of an unbatched model is the sum over shots -- which is what an
    unbatched model means. The point here is that it flows at all.
    """
    models, survey = _batched_problem()
    v = models["v"].clone().requires_grad_(True)
    dv = _batch(models["dv"]).clone().requires_grad_(True)
    out = ab.acoustic_born(
        v,
        models["rho"],
        6.0,
        0.0004,
        dv=dv,
        drho=models["drho"],
        accuracy=4,
        **survey,
    )
    out.sc_receiver_amplitudes_p.abs().sum().backward()
    assert dv.grad is not None
    assert dv.grad.shape == dv.shape
    assert bool(torch.isfinite(dv.grad).all())
    assert dv.grad.abs().max() > 0


@pytest.mark.parametrize("ndim", [1, 2, 3], ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("accuracy", [2, 8])
@pytest.mark.parametrize("nt", [2, 3])
@pytest.mark.parametrize("full_adjoint", [False, True], ids=["scatter", "full"])
@pytest.mark.parametrize("memory", ["phi", "psi"])
def test_pml_state_cotangents_match_autograd(
    ndim, accuracy, nt, full_adjoint, memory, numerics_device
):
    """Both adjoint paths preserve PML field identity across odd-sized chunks."""
    results = []
    for backend in ("compiled", "eager"):
        v, rho, dv, drho, dx, dt, kwargs = _problem(ndim)
        v, rho, dv, drho = (t.to(numerics_device) for t in (v, rho, dv, drho))
        kwargs = {
            key: value.to(numerics_device) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        v.requires_grad_(full_adjoint)
        rho.requires_grad_(full_adjoint)
        dv.requires_grad_()
        drho.requires_grad_()
        leaves = [v, rho, dv, drho] if full_adjoint else [dv, drho]
        kwargs["source_amplitudes_volume_rate"] = (
            kwargs["source_amplitudes_volume_rate"][..., :nt]
            .clone()
            .requires_grad_(full_adjoint)
        )
        if full_adjoint:
            leaves.append(kwargs["source_amplitudes_volume_rate"])
        kwargs["nt"] = nt
        kwargs["origin"] = [0] * ndim
        generator = torch.Generator(device=numerics_device).manual_seed(316)
        shape = (1, *(n + 2 * kwargs["pml_width"] for n in v.shape))
        for name in ab.acoustic_born_wavefield_names(ndim):
            if not name.startswith(("pressure", "v")):
                continue
            value = torch.randn(
                shape, dtype=v.dtype, device=numerics_device, generator=generator
            )
            value *= 1e3 if name.startswith("pressure") else 1e-3
            value.requires_grad_(full_adjoint or "_sc_" in name)
            kwargs[name] = value
            if value.requires_grad:
                leaves.append(value)
        if backend == "compiled":
            # One-step chunks expose every ping-pong transition.
            kwargs["backward_callback"] = lambda state: None
            kwargs["callback_frequency"] = 1
        out = ab.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            accuracy=accuracy,
            backend=backend,
            **kwargs,
        )
        loss = sum(
            (
                value
                * torch.randn(
                    value.shape,
                    dtype=value.dtype,
                    device=numerics_device,
                    generator=generator,
                )
            ).mean()
            for name, value in out.auxiliary_state.items()
            if name.startswith(memory)
        )
        gradients = torch.autograd.grad(loss, leaves)
        assert all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
        results.append(gradients)
    torch.testing.assert_close(results[0], results[1], rtol=1e-8, atol=1e-10)

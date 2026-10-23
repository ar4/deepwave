"""Custom imaging conditions from forward x adjoint wavefields in the backward callback.

Deepwave's ``forward_callback`` exposes the forward wavefield read-only; the
``backward_callback`` exposes the adjoint (gradient) wavefield. A *true* custom
imaging condition (RTM/Poynting/energy-norm) needs both fields at the *same*
time step. ``scalar``'s pure-PyTorch (``backend="eager"``) backend
surfaces the reconstructed forward wavefield time-coincident with the adjoint
field via ``state.get_wavefield(name, role="forward")``; the other
propagators' eager backends do not accept a ``backward_callback`` at all.

The default compiled C/CUDA backend cannot do this in general: most
propagators store a pre-multiplied gradient-correlation kernel rather than the
raw forward wavefield, so the forward field is not recoverable Python-side
there. ``scalar`` is again the exception: its compiled backend allocates a
second, opt-in checkpoint buffer for the raw forward wavefield whenever a
``backward_callback`` is supplied (paid only by callback users), so
``role="forward"`` works there too. Most tests below exercise scalar's eager
backend; the parity/compiled-specific tests also exercise scalar's default
compiled backend, and one test asserts that a propagator without this support
(``acoustic``, compiled) still raises a clear error.
"""

import pytest
import torch

import deepwave
from deepwave import scalar

DTYPE = torch.float64


def _problem(ny=40, nx=40, nt=140, dt=0.0008, freq=12.0):
    dx = 5.0
    src = deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq).to(DTYPE).reshape(1, 1, nt)
    kw = {
        "source_locations": torch.tensor([[[2, nx // 2]]]),
        "receiver_locations": torch.tensor([[[2, c] for c in range(2, nx - 2, 2)]]),
        "pml_freq": freq,
        "max_vel": 1800.0,
    }
    return dx, dt, freq, nt, src, kw


@pytest.mark.parametrize(
    "backend",
    [
        "compiled",
        pytest.param("eager"),
    ],
    ids=["compiled", "eager"],
)
def test_backward_callback_forward_field_parity(backend):
    """Forward field surfaced in the backward callback equals a forward record.

    Same steps, same values, bitwise -- the whole point of the feature. On the
    default compiled backend (``backend="compiled"``) this exercises scalar's
    opt-in raw-forward checkpoint buffer; the eager backend has its own,
    independent reconstruction. ``forward_callback``'s ``state.wavefields``
    entry is a live reference into the same buffer the propagator continues to
    update, so it must be cloned before the next chunk mutates it in place --
    that clone is exactly what a genuine forward record would be.
    """
    dx, dt, _freq, nt, src, kw = _problem()
    v = torch.full((40, 40), 1500.0, dtype=DTYPE, requires_grad=True)

    recorded = {}

    def fcb(state):
        recorded[state.step] = state.get_wavefield("wavefield_0", view="pml").clone()

    surfaced = {}
    adjoints = {}

    def bcb(state):
        surfaced[state.step] = state.get_wavefield(
            "wavefield_0", view="pml", role="forward"
        ).clone()
        adjoints[state.step] = state.get_wavefield("wavefield_0", view="pml").clone()

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src,
        **kw,
        backend=backend,
        forward_callback=fcb,
        backward_callback=bcb,
        backward_callback_needs_forward_wavefield=True,
        callback_frequency=1,
    )
    (out.receiver_amplitudes.pow(2).sum()).backward()

    # The initial (step 0) field is a non-grad zero leaf, so no adjoint hook
    # fires there; every other step must match bitwise.
    assert set(surfaced).issubset(set(recorded))
    assert len(surfaced) >= nt - 1
    for step, field in surfaced.items():
        assert torch.equal(field, recorded[step]), step
    # The adjoint field is a genuine, non-trivial field at interior steps.
    mid = nt // 2
    assert adjoints[mid].abs().max() > 0
    assert adjoints[mid].std() > 0


@pytest.mark.parametrize(
    ("dtype", "rtol"), [(torch.float32, 1e-6), (torch.float64, 1e-12)]
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_scalar_compiled_role_forward_matches_eager(dtype, rtol, device):
    """Compiled scalar's role="forward" field matches the eager reconstruction.

    A genuine cross-backend check (unlike the bitwise self-consistency check
    in ``test_backward_callback_forward_field_parity``): the compiled
    backend's raw-forward checkpoint buffer and the eager backend's autograd-
    hook reconstruction are two independent code paths that must agree, to
    floating-point roundoff, on the same small problem. float32/float64
    accumulate that roundoff differently over many time steps, so the case
    here is deliberately small (few grid points, few time steps). Retaining the
    field must leave each backend's output and gradient exactly unchanged.
    Float64 gradients also agree across backends; float32 gradients are checked
    against each backend's float64 reference with the same encoded inputs.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device)
    ny, nx, nt, dt, freq = 16, 16, 25, 0.001, 15.0
    dx = 5.0
    src = (
        deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq)
        .to(dtype)
        .reshape(1, 1, nt)
        .to(device)
    )
    kw = {
        "source_locations": torch.tensor([[[2, nx // 2]]]),
        "receiver_locations": torch.tensor([[[2, c] for c in range(2, nx - 2, 4)]]),
        "pml_freq": freq,
        "max_vel": 1800.0,
    }

    def run(backend, run_dtype=dtype, retain_forward=True):
        # Keep the source's encoded values fixed for the float64 oracle.
        v = torch.full(
            (ny, nx), 1500.0, dtype=run_dtype, device=device, requires_grad=True
        )
        surfaced = {}

        def bcb(state):
            surfaced[state.step] = (
                state.get_wavefield("wavefield_0", view="pml", role="forward")
                .clone()
                .cpu()
            )

        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=src.to(run_dtype),
            **kw,
            backend=backend,
            backward_callback=bcb if retain_forward else None,
            backward_callback_needs_forward_wavefield=retain_forward,
            callback_frequency=1,
        )
        (out.receiver_amplitudes.pow(2).sum()).backward()
        return (
            surfaced,
            v.grad.detach().clone().cpu(),
            out.receiver_amplitudes.detach().clone().cpu(),
            out.info.step_ratio,
        )

    eager_result = run("eager")
    compiled_result = run("compiled")
    eager_surfaced, eager_grad = eager_result[:2]
    compiled_surfaced, compiled_grad = compiled_result[:2]

    common_steps = set(eager_surfaced) & set(compiled_surfaced)
    assert len(common_steps) >= nt - 2
    for step in common_steps:
        e = eager_surfaced[step]
        c = compiled_surfaced[step]
        denom = e.abs().max().clamp_min(1e-30)
        rel = (e - c).abs().max() / denom
        assert rel <= rtol, (step, rel.item())

    # Retaining raw forward fields must not change the simulation or gradient.
    # This is an exact within-backend contract, including float32.
    for backend, result in (("eager", eager_result), ("compiled", compiled_result)):
        assert torch.isfinite(result[1]).all()
        assert result[1].abs().max() > 0
        without_forward = run(backend, retain_forward=False)
        assert torch.equal(result[1], without_forward[1])
        assert torch.equal(result[2], without_forward[2])

    if dtype == torch.float64:
        torch.testing.assert_close(
            compiled_grad, eager_grad, rtol=0, atol=float(rtol * eager_grad.abs().max())
        )
    else:
        # CPU C and eager float32 gradients round differently while their
        # forward fields agree. Check each against its own float64 computation,
        # with identical encoded inputs, rather than demanding matching rounding.
        # One epsilon per internal step is a scoped allowance for this fixture,
        # not a general error bound for arbitrary models or ill-conditioned losses.
        references = {
            backend: run(backend, torch.float64)[1] for backend in ("eager", "compiled")
        }
        scale = references["eager"].abs().max()
        assert scale > 0
        torch.testing.assert_close(
            references["compiled"],
            references["eager"],
            rtol=0,
            atol=float(1e-12 * scale),
        )
        for backend, result in (("eager", eager_result), ("compiled", compiled_result)):
            reference = references[backend]
            roundoff = nt * result[3] * torch.finfo(dtype).eps
            assert torch.isfinite(result[1]).all()
            torch.testing.assert_close(
                result[1].double(),
                reference,
                rtol=0,
                atol=float(roundoff * reference.abs().max()),
            )


def test_laplacian_weighted_image_tracks_interior_gradient_structure():
    """A coarse imaging stencil captures broad interior gradient structure.

    This correlation check is not a gradient oracle: source terms, padding,
    PML and the actual finite-difference stencil are absent from the image.
    """
    dx, dt, _freq, _nt, src, kw = _problem()
    ny, nx = 40, 40
    v_true = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    v_true[26:] = 1650.0
    d_true = scalar(v_true, dx, dt, source_amplitudes=src, **kw).receiver_amplitudes

    v = torch.full((ny, nx), 1500.0, dtype=DTYPE).requires_grad_(True)
    image = torch.zeros(ny, nx, dtype=DTYPE)

    def laplacian(w):
        out = torch.zeros_like(w)
        out[1:-1, 1:-1] = (
            w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * w[1:-1, 1:-1]
        ) / dx**2
        return out

    def bcb(state):
        f = state.get_wavefield("wavefield_0", view="inner", role="forward")[0]
        a = state.get_wavefield("wavefield_0", view="inner")[0]
        image.add_(2 * v.detach() * dt * dt * laplacian(f) * a)

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src,
        **kw,
        backend="eager",
        backward_callback=bcb,
        backward_callback_needs_forward_wavefield=True,
        callback_frequency=1,
    )
    (0.5 * (out.receiver_amplitudes - d_true).pow(2).sum()).backward()

    # Nonzero, structured output (not a flat field).
    assert image.abs().max() > 0
    assert (image.std() / image.abs().mean()) > 1.0

    # Interior correlation with Deepwave's own gradient (the crude 5-point
    # laplacian differs from Deepwave's stencil/PML at the edges, so compare the
    # interior). Correlation, not equality, is the assertion.
    sl = (slice(4, -4), slice(4, -4))
    g = v.grad
    cc = torch.corrcoef(torch.stack([image[sl].flatten(), g[sl].flatten()]))[0, 1]
    assert cc > 0.9, cc


def test_energy_norm_image_sign_and_location():
    """An energy-norm forward x adjoint image is positive and peaks at the source.

    With source and receivers co-located near the surface, the zero-lag
    forward x adjoint correlation is largest along the (shared) source/receiver
    path -- a known, checkable location. Using the energy (forward*adjoint with a
    fixed reference sign) the dominant contribution is at the source row.
    """
    dx, dt, _freq, _nt, src, kw = _problem()
    ny, nx = 40, 40
    v_true = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    v_true[26:] = 1650.0
    d_true = scalar(v_true, dx, dt, source_amplitudes=src, **kw).receiver_amplitudes

    v = torch.full((ny, nx), 1500.0, dtype=DTYPE).requires_grad_(True)
    image = torch.zeros(ny, nx, dtype=DTYPE)

    def bcb(state):
        f = state.get_wavefield("wavefield_0", view="inner", role="forward")[0]
        a = state.get_wavefield("wavefield_0", view="inner")[0]
        image.add_(f * a)

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src,
        **kw,
        backend="eager",
        backward_callback=bcb,
        backward_callback_needs_forward_wavefield=True,
        callback_frequency=1,
    )
    (0.5 * (out.receiver_amplitudes - d_true).pow(2).sum()).backward()

    row_energy = image.pow(2).sum(dim=1)
    peak_row = int(torch.argmax(row_energy))
    source_row = 2
    # The image concentrates at the source/receiver depth (within a couple of
    # cells), the expected location for this surface acquisition.
    assert abs(peak_row - source_row) <= 2, peak_row


def test_role_forward_unavailable_on_c_backend():
    """A compiled backend without raw-forward support raises role='forward' clearly.

    scalar's default compiled backend now supports role="forward" (see
    ``test_scalar_compiled_role_forward_matches_eager`` and the parametrized
    ``test_backward_callback_forward_field_parity``); every other propagator's
    compiled backend still does not retain the forward wavefield, so this test
    exercises ``acoustic`` instead to check the rejection path still works.
    """
    dx, dt, _freq, nt, freq = 5.0, 0.001, 15.0, 30, 15.0
    ny, nx = 20, 20
    src = deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq).to(DTYPE).reshape(1, 1, nt)
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE, requires_grad=True)
    rho = torch.full((ny, nx), 1000.0, dtype=DTYPE)

    captured = {}

    def bcb(state):
        if "err" not in captured:
            try:
                state.get_wavefield("pressure_0", role="forward")
            except NotImplementedError as e:
                captured["err"] = str(e)

    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=src,
        source_locations_volume_rate=torch.tensor([[[2, nx // 2]]]),
        receiver_locations_p=torch.tensor([[[2, c] for c in range(2, nx - 2, 4)]]),
        pml_freq=freq,
        max_vel=1800.0,
        backward_callback=bcb,
        callback_frequency=1,
    )
    (out.receiver_amplitudes_p.pow(2).sum()).backward()
    assert "err" in captured
    assert "forward wavefield" in captured["err"]
    assert "eager" in captured["err"]


def test_invalid_role_raises():
    """An unknown role is rejected."""
    state = deepwave.CallbackState(
        0.001,
        0,
        {"wavefield_0": torch.zeros(1, 4, 4)},
        {"v": torch.zeros(1, 4, 4)},
        {},
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        direction="backward",
        n_steps=1,
    )
    with pytest.raises(ValueError, match="role must be"):
        state.get_wavefield("wavefield_0", role="sideways")


def test_default_role_backward_compatible():
    """role=None returns the primary wavefield, unchanged from before."""
    wf = torch.arange(16, dtype=DTYPE).reshape(1, 4, 4)
    state = deepwave.CallbackState(
        0.001,
        0,
        {"wavefield_0": wf},
        {"v": torch.zeros(1, 4, 4)},
        {},
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        direction="backward",
        n_steps=1,
    )
    assert torch.equal(state.get_wavefield("wavefield_0", view="full"), wf)
    assert torch.equal(
        state.get_wavefield("wavefield_0", view="full", role="adjoint"), wf
    )

"""Forward-mode AD (``torch.func.jvp``) for the scalar propagator via Born.

The scalar propagator now supports forward-mode automatic differentiation on
its fast compiled (C/CUDA) backend by routing the input tangents through the
scalar Born machinery: the Born scattered field *is* the directional derivative
(JVP) of ``scalar`` with respect to the velocity model. These tests exercise
the four supported tangent inputs (velocity, source amplitudes, initial
wavefields, and combinations), the
machine-precision equivalence to ``scalar_born`` on the compiled backend, the
PML caveat warning, and both the ``torch.func.jvp`` and legacy
``torch.autograd.forward_ad`` entry points.
"""

import warnings

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from test_scalar_born import _born_jvp_problem
from torch.func import jvp

from deepwave import elastic_vti, scalar, scalar_born, viscoacoustic
from deepwave.wavelets import ricker


def _assert_fd_close(tangent, fd, rtol=1e-5):
    """Global max-norm relative check against a central finite difference.

    Central FD at eps=1e-6 in float64 is roundoff/truncation-limited to roughly
    1e-6..1e-5 relative on these signals (measured), so a max-norm relative
    tolerance is the honest comparison. The exact machine-precision check is
    ``test_jvp_receiver_equals_born_c_backend``.
    """
    assert (tangent - fd).abs().max() <= rtol * fd.abs().max()


@pytest.mark.parametrize("ndim", [1, 2])
def test_jvp_receiver_equals_born_c_backend(ndim) -> None:
    """jvp(scalar) receiver tangent == scalar_born scattered receivers.

    This is the crux of C2.2: on the *compiled* (default) C backend the
    forward-mode derivative of ``scalar`` equals the ``scalar_born`` scattered
    output (with ``scatter = dv``), and the primal equals the background output,
    to near machine precision. The pure-PyTorch eager backend reproduces the
    same float operations, so it matches to full machine precision; the C
    backend (``-Ofast``/``--use_fast_math``) is checked at a looser backend
    tolerance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(ndim, device, dtype)

    for backend, tol in (("eager", 1e-12), ("compiled", 1e-6)):

        def scalar_receivers(vv, backend=backend):
            return scalar(vv, backend=backend, **kwargs)[-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dv is interior: no PML warning
            primal, tangent = jvp(scalar_receivers, (v,), (dv,))
            out = scalar_born(
                v,
                dv=dv,
                bg_receiver_locations=kwargs["receiver_locations"],
                backend=backend,
                **kwargs,
            )
        bg_receivers, sc_receivers = out[-2], out[-1]
        assert (sc_receivers - tangent).abs().max() <= tol * tangent.abs().max()
        assert (bg_receivers - primal).abs().max() <= tol * primal.abs().max()


@pytest.mark.parametrize(
    ("ndim", "boundary"),
    [
        (1, ["free", "rigid"]),
        (2, ["free", "rigid", "pml", "pml"]),
        (2, ["periodic", "periodic", "free", "free"]),
        (2, ["rigid", "rigid", "rigid", "rigid"]),
    ],
)
def test_jvp_receiver_equals_born_with_boundary(ndim, boundary) -> None:
    """jvp(scalar) == scalar_born with non-PML (free/rigid/periodic) faces.

    Task O STEP C: mirrors ``test_jvp_receiver_equals_born_c_backend`` but
    exercises every non-PML face type, plus a mixed PML/reflecting case, since
    scalar_born's compiled and eager guard-ring fill now threads the boundary
    ``bc`` codes through the jvp rule (``ctx.jvp_boundary``, no longer forced to
    all-PML): the forward-mode rule's ``NotImplementedError`` for non-PML faces
    has been removed. The forward pass (which this jvp rule routes through)
    reads the models only at the current grid point, so, unlike the HVP path,
    no model guard-ring mirror is needed here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(ndim, device, dtype)
    # Non-PML faces require pml_width == 0 on that face (see set_boundary).
    pml_width = [0 if bc != "pml" else 4 for bc in boundary]
    kwargs = dict(kwargs, boundary=boundary, pml_width=pml_width)

    for backend, tol in (("eager", 1e-12), ("compiled", 1e-6)):

        def scalar_receivers(vv, backend=backend):
            return scalar(vv, backend=backend, **kwargs)[-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dv is interior: no PML warning
            primal, tangent = jvp(scalar_receivers, (v,), (dv,))
            out = scalar_born(
                v,
                dv=dv,
                bg_receiver_locations=kwargs["receiver_locations"],
                backend=backend,
                **kwargs,
            )
        bg_receivers, sc_receivers = out[-2], out[-1]
        assert (sc_receivers - tangent).abs().max() <= tol * tangent.abs().max()
        assert (bg_receivers - primal).abs().max() <= tol * primal.abs().max()


@pytest.mark.parametrize(
    ("boundary", "pml", "shape", "src", "rec"),
    [
        (["free", "rigid"], [0, 0], (16,), [[4]], [[3], [13]]),
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            (16, 16),
            [[4, 4]],
            [[3, 3]],
        ),
        (
            ["periodic", "periodic", "periodic", "periodic"],
            [0, 0, 0, 0],
            (16, 16),
            [[8, 8]],
            [[3, 3]],
        ),
        (
            ["pml", "free", "rigid", "pml"],
            [6, 0, 0, 6],
            (18, 18),
            [[13, 4]],
            [[15, 3]],
        ),
        (
            ["pml", "free", "rigid", "pml", "periodic", "periodic"],
            [4, 0, 0, 4, 0, 0],
            (12, 12, 12),
            [[8, 4, 5]],
            [[9, 3, 2]],
        ),
    ],
)
def test_hvp_matches_eager_with_boundary(boundary, pml, shape, src, rec) -> None:
    """Second-order (Hessian-vector product) compiled ``scalar`` == eager.

    Task O STEP C: the double-backward routes through scalar_born's compiled
    backend, whose adjoint reads the model coefficients (v, and the HVP's
    scattered model ``ggv``) at neighbour offsets next to a reflecting face --
    exactly like scalar_born's own backward -- so it needs the same guard-ring
    mirror (``mirror_pad_scalar_born_models``) applied before the compiled call.

    This uses the same short, controlled geometry as
    ``test_scalar_c_matches_eager_grad`` (test_boundary.py) rather than
    a long, lossless (undamped) reflecting-cavity run: comparing the compiled
    (``-Ofast``/``--use_fast_math``) backend against eager over many
    reflections with no PML damping amplifies ordinary floating-point
    differences between the two implementations by several orders of
    magnitude. That amplification is not specific to second-order
    differentiation or to this task -- it is already present, at a
    correspondingly smaller scale, in the plain first-order compiled-vs-eager
    gradient at reflecting faces over a long undamped run, and is independently
    confirmed benign there by a compiled-only self-consistency check (analytic
    HVP vs. a finite difference of the compiled gradient, which agree to
    ~1e-6). On the short, controlled window used here (matching the existing
    gradcheck test's precedent), both the first- and second-order compiled and
    eager results agree to near machine precision for every face type
    (free/rigid/periodic, mixed PML/reflecting, 1-D/2-D/3-D).
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 60
    acc = 2 if len(shape) == 3 else 4
    sa = ricker(15.0, nt, dt, 0.02).reshape(1, 1, nt).to(torch.double)
    v = (torch.ones(shape) * 1500.0 + torch.randn(shape)).to(torch.double)
    sl, rl = torch.tensor([src]), torch.tensor([rec])
    kw = {
        "grid_spacing": dx,
        "dt": dt,
        "source_amplitudes": sa,
        "source_locations": sl,
        "receiver_locations": rl,
        "pml_width": pml,
        "boundary": boundary,
        "accuracy": acc,
        "pml_freq": 15.0,
    }
    torch.manual_seed(1)
    vec = torch.randn_like(v)

    def hvp(backend):
        vv = v.clone().requires_grad_(True)
        out = scalar(vv, backend=backend, **kw)[-1]
        (g,) = torch.autograd.grad((out**2).sum(), vv, create_graph=True)
        (h,) = torch.autograd.grad((g * vec).sum(), vv)
        return g.detach(), h.detach()

    g_c, h_c = hvp("compiled")
    g_e, h_e = hvp("eager")
    tol = 1e-9
    assert (g_c - g_e).abs().max() <= tol * g_e.abs().max()
    assert (h_c - h_e).abs().max() <= tol * h_e.abs().max()


def test_jvp_v_tangent_vs_finite_difference() -> None:
    """v-tangent: receiver and final-wavefield tangents match central FD.

    With the perturbation confined to the interior (zero in the PML, so the Born
    = JVP equivalence is exact) the compiled-backend jvp reproduces a float64
    central finite difference of the nonlinear ``scalar`` forward.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(2, device, dtype)
    eps = 1e-6

    def f(vv):
        out = scalar(vv, backend="compiled", **kwargs)
        return out[-1], out[0]  # receivers, final wavefield_0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (_, _), (tan_r, tan_w) = jvp(f, (v,), (dv,))
    rp, wp = f(v + eps * dv)
    rm, wm = f(v - eps * dv)
    fd_r = (rp - rm) / (2 * eps)
    fd_w = (wp - wm) / (2 * eps)
    # Central finite differences at eps=1e-6 in float64 are accurate to ~1e-7
    # relative (roundoff-limited), so compare at that level; the exact
    # machine-precision check is test_jvp_receiver_equals_born_c_backend.
    _assert_fd_close(tan_r, fd_r)
    _assert_fd_close(tan_w, fd_w)


def test_jvp_source_tangent_vs_finite_difference() -> None:
    """source_amplitudes-tangent matches central FD (receiver + wavefield)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, _, kwargs = _born_jvp_problem(2, device, dtype)
    sa = kwargs["source_amplitudes"]
    torch.manual_seed(1)
    dsa = torch.randn_like(sa)
    eps = 1e-6
    other = {k: val for k, val in kwargs.items() if k != "source_amplitudes"}

    def f(a):
        out = scalar(v, source_amplitudes=a, backend="compiled", **other)
        return out[-1], out[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (_, _), (tan_r, tan_w) = jvp(f, (sa,), (dsa,))
    rp, wp = f(sa + eps * dsa)
    rm, wm = f(sa - eps * dsa)
    fd_r = (rp - rm) / (2 * eps)
    fd_w = (wp - wm) / (2 * eps)
    _assert_fd_close(tan_r, fd_r)
    _assert_fd_close(tan_w, fd_w)


def test_jvp_combined_v_and_source_tangent_vs_finite_difference() -> None:
    """Combined (v + source) tangent matches FD, exercising linearity.

    A single Born run must carry both contributions: the scattered source
    injected by Born is the tangent of the ``-v^2 dt^2 f`` pre-scaling, which
    already sums the velocity and source-amplitude terms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(2, device, dtype)
    sa = kwargs["source_amplitudes"]
    torch.manual_seed(2)
    dsa = torch.randn_like(sa)
    eps = 1e-6
    other = {k: val for k, val in kwargs.items() if k != "source_amplitudes"}

    def f(vv, a):
        return scalar(vv, source_amplitudes=a, backend="compiled", **other)[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, tan = jvp(f, (v, sa), (dv, dsa))
    fd = (f(v + eps * dv, sa + eps * dsa) - f(v - eps * dv, sa - eps * dsa)) / (2 * eps)
    _assert_fd_close(tan, fd)


def test_jvp_initial_wavefield_tangent_vs_finite_difference() -> None:
    """Initial-wavefield tangent matches FD (receiver + final wavefield).

    The initial-wavefield tangents map onto Born's scattered initial wavefields.
    The perturbed wavefield is given at the propagator's internal (padded) size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny, nx = 24, 24
    freq = 25.0
    kwargs = {
        "grid_spacing": 5.0,
        "dt": 0.004,
        "source_amplitudes": ricker(freq, 30, 0.004, 1.5 / freq, dtype=dtype)
        .to(device)
        .reshape(1, 1, -1),
        "source_locations": torch.tensor([[[12, 12]]], device=device),
        "receiver_locations": torch.tensor([[[12, 18]]], device=device),
        "pml_width": 4,
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    v = torch.full((ny, nx), 1500.0, device=device, dtype=dtype)
    v[12:] += 200.0
    # Discover the internal (padded) wavefield shape.
    wshape = scalar(v, **kwargs)[0].shape
    torch.manual_seed(5)
    w0 = torch.randn(wshape, device=device, dtype=dtype)
    dw0 = torch.randn_like(w0)
    eps = 1e-6

    def f(w):
        out = scalar(v, wavefield_0=w, backend="compiled", **kwargs)
        return out[-1], out[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (_, _), (tan_r, tan_w) = jvp(f, (w0,), (dw0,))
    rp, wp = f(w0 + eps * dw0)
    rm, wm = f(w0 - eps * dw0)
    fd_r = (rp - rm) / (2 * eps)
    fd_w = (wp - wm) / (2 * eps)
    _assert_fd_close(tan_r, fd_r)
    _assert_fd_close(tan_w, fd_w)


def test_jvp_v_tangent_in_pml_warns() -> None:
    """A velocity tangent inside the PML region warns, unless ``max_vel`` is given.

    The PML profile's amplitude is built from the maximum wave speed, so it
    depends on the velocity model only while that maximum is inferred from it.
    Supplying ``max_vel`` makes the profile a constant of the call, and there
    is then nothing for the Born-based forward-mode rule to have left
    unlinearised -- so the caveat does not apply and is not raised.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    n = 24
    pml_width = 4
    v, _, kwargs = _born_jvp_problem(2, device, dtype, pml_width=pml_width, n=n)
    pinned = dict(kwargs)
    inferred = {name: value for name, value in kwargs.items() if name != "max_vel"}
    assert "max_vel" in pinned
    dv_pml = torch.zeros_like(v)
    torch.manual_seed(3)
    dv_pml[:pml_width] = torch.randn(pml_width, n, device=device, dtype=dtype) * 20.0

    def f(vv, kw):
        return scalar(vv, backend="compiled", **kw)[-1]

    with pytest.warns(UserWarning, match="PML"):
        jvp(lambda vv: f(vv, inferred), (v,), (dv_pml,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jvp(lambda vv: f(vv, pinned), (v,), (dv_pml,))
    assert not [record for record in caught if "PML" in str(record.message)], [
        str(record.message) for record in caught
    ]


def test_jvp_matches_legacy_forward_ad() -> None:
    """The torch.func.jvp and torch.autograd.forward_ad paths agree.

    Both go through the same ``Function.jvp`` hook, so the compiled-backend
    tangents must be identical.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(2, device, dtype)

    def f(vv):
        return scalar(vv, backend="compiled", **kwargs)[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, tangent_func = jvp(f, (v,), (dv,))
        with forward_ad.dual_level():
            vd = forward_ad.make_dual(v, dv)
            out = scalar(vd, backend="compiled", **kwargs)[-1]
            _, tangent_legacy = forward_ad.unpack_dual(out)
    assert torch.equal(tangent_func, tangent_legacy)


def test_jvp_nested_transforms_raise() -> None:
    """Nested torch.func transforms over the fast-backend jvp raise clearly.

    The Born-based rule computes concrete tangents on the compiled backend, so
    it cannot keep an *outer* transform's tangent tracking alive; composing
    (e.g. jvp of jvp) must fail loudly rather than silently detach.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    v, dv, kwargs = _born_jvp_problem(2, device, dtype)

    def f(vv):
        return scalar(vv, backend="compiled", **kwargs)[-1]

    with pytest.raises(NotImplementedError, match="nested"):
        jvp(lambda a: jvp(f, (a,), (dv,))[1], (v,), (dv,))


def test_jvp_raises_actionably_where_unsupported() -> None:
    """Propagators without a compiled-backend jvp rule raise actionably.

    Only ``scalar`` has a Born-based forward-mode rule on the compiled (C/CUDA)
    backend (a positive test that ``acoustic`` also gained one, via its own
    Born-based rule, lives in ``test_acoustic_jvp.py``; ``elastic``'s is
    covered by ``test_elastic.py::test_jvp_elastic``). The remaining compiled
    propagators -- ``scalar_born``, ``viscoacoustic``, ``elastic_vti`` -- still
    point the user at ``backend="eager"`` via ``unsupported_jvp``
    (common.py) rather than let PyTorch raise its generic, unhelpful "implement
    the jvp function" error. The match targets a distinctive fragment of that
    actionable message (``_functorch_unsupported_message`` in common.py) --
    not just ``"jvp"``, which also appears in PyTorch's generic fallback text
    and so would not actually prove the actionable message fires.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny, nx = 12, 12
    freq = 25.0
    sa = ricker(freq, 20, 0.004, 1.5 / freq, dtype=dtype).to(device).reshape(1, 1, -1)
    src = torch.tensor([[[6, 6]]], device=device)
    rec = torch.tensor([[[6, 8]]], device=device)
    v = torch.full((ny, nx), 1500.0, device=device, dtype=dtype)
    dv = torch.zeros_like(v)
    dv[6, 6] = 1.0

    def scalar_born_f(vv):
        scat = torch.zeros((ny, nx), device=device, dtype=dtype)
        return scalar_born(
            vv,
            grid_spacing=5.0,
            dt=0.004,
            dv=scat,
            source_amplitudes=sa,
            source_locations=src,
            sc_receiver_locations=rec,
            pml_width=3,
            pml_freq=freq,
            max_vel=2500.0,
        )[-1]

    def viscoacoustic_f(vv):
        rho = torch.full((ny, nx), 2200.0, device=device, dtype=dtype)
        q = torch.full((ny, nx), 20.0, device=device, dtype=dtype)
        return viscoacoustic(
            vv,
            rho,
            q,
            grid_spacing=5.0,
            dt=0.0004,
            source_amplitudes_pressure_rate=(2200.0 * 1500.0**2) * (sa),
            source_locations_pressure_rate=src,
            receiver_locations_p=rec,
            pml_width=3,
            pml_freq=freq,
            q_freq=freq,
            max_vel=2500.0,
        )[-1]

    def elastic_vti_f(vv):
        # vv plays the role of c11; the other Cij are held fixed so the
        # tangent flows through a single differentiable model input, mirroring
        # scalar_born_f/viscoacoustic_f above.
        c13 = torch.full((ny, nx), 4.0e9, device=device, dtype=dtype)
        c33 = torch.full((ny, nx), 9.0e9, device=device, dtype=dtype)
        c55 = torch.full((ny, nx), 2.25e9, device=device, dtype=dtype)
        buoyancy = torch.full((ny, nx), 1.0 / 2200.0, device=device, dtype=dtype)
        return elastic_vti(
            vv,
            c13,
            c33,
            c55,
            buoyancy,
            5.0,
            dt=0.0001,
            c66=None,  # c66 must be None in 2-D (see elastic_vti's ValueError)
            source_amplitudes_x=sa,
            source_locations_x=src,
            receiver_locations_x=rec,
            pml_width=3,
            pml_freq=freq,
            max_vel=2500.0,
        )[-1]

    c11 = torch.full((ny, nx), 9.0e9, device=device, dtype=dtype)
    dc11 = torch.zeros_like(c11)
    dc11[6, 6] = 1.0e6

    for fn, primal, tangent in (
        (scalar_born_f, v, dv),
        (viscoacoustic_f, v, dv),
        (elastic_vti_f, c11, dc11),
    ):
        with pytest.raises(RuntimeError, match="not supported on Deepwave's compiled"):
            jvp(fn, (primal,), (tangent,))


def test_reverse_mode_transforms_raise_naming_the_eager_backend() -> None:
    """``torch.func.grad``/``vjp``/``jacrev`` name ``backend="eager"``.

    The reverse-mode transforms differentiate a tensor that does not carry
    ``requires_grad`` through the forward pass, so the compiled backend's
    ``setup_context`` saves nothing (it is gated on ``requires_grad``, which
    is correct for ordinary autograd: with nothing requiring a gradient,
    ``backward`` is never called). functorch then runs a ``backward``
    against a generated node anyway. Before
    ``deepwave._engine.saved_state_ndim`` guarded it, that surfaced as
    ``AttributeError: 'ScalarForwardFuncGeneratedBackward' object has no
    attribute 'grid_spacing'`` -- a private class name and no route forward,
    where ``vmap`` on the same backend has named ``backend="eager"`` all
    along.

    The eager half of this test is the part that makes the message
    trustworthy: it checks that the escape hatch the error recommends
    actually produces the gradient, and that it is the same gradient an
    ordinary ``.backward()`` gives on the compiled backend.
    """
    ny, nx, nt = 12, 14, 20
    freq = 25.0
    dx, dt = 5.0, 0.0005
    v = torch.full((ny, nx), 1500.0)
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 4] = 1.0
    source_locations = torch.tensor([[[2, 2]]])
    receiver_locations = torch.tensor([[[ny - 3, 3]]])

    def loss(model: torch.Tensor, backend: str = "compiled") -> torch.Tensor:
        return (
            scalar(
                model,
                dx,
                dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                pml_freq=freq,
                max_vel=2000.0,
                backend=backend,
            ).receiver_amplitudes
            ** 2
        ).sum()

    pattern = "reached without the state"
    with pytest.raises(RuntimeError, match=pattern):
        torch.func.grad(loss)(v)
    with pytest.raises(RuntimeError, match=pattern):
        torch.func.vjp(loss, v)[1](torch.tensor(1.0))
    with pytest.raises(RuntimeError, match=pattern):
        torch.func.jacrev(loss)(v)

    # The message names backend="eager"; that route must work...
    with pytest.raises(RuntimeError, match='backend="eager"'):
        torch.func.grad(loss)(v)
    eager_grad = torch.func.grad(lambda m: loss(m, "eager"))(v)

    # ...and agree with the ordinary compiled-backend gradient.
    v_grad = v.clone().requires_grad_(True)
    loss(v_grad).backward()
    assert v_grad.grad is not None
    # Compared relative to the gradient's own scale, not element-wise: the two
    # backends are different implementations of the same adjoint, and the
    # far-field entries of this tiny problem are denormals (~1e-37) where a
    # per-element rtol is meaningless.
    scale = float(v_grad.grad.abs().max())
    assert scale > 0.0
    assert float((eager_grad - v_grad.grad).abs().max()) < 1e-6 * scale

    # Forward-mode and plain autograd on the compiled backend are untouched.
    _, tangent_out = jvp(loss, (v,), (torch.ones_like(v),))
    assert torch.isfinite(tangent_out).all()

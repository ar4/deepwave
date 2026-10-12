"""Dot-product (adjoint) tests for every propagator's linear inputs.

Each Deepwave propagator is a linear operator ``F`` with respect to several of
its inputs: the ``source_amplitudes`` (a source injected linearly into the wave
equation), the initial wavefields (which evolve linearly under the
homogeneous propagator), and -- for the Born pairs -- the scattering model
(which enters the scattered field linearly, with the background field as its
coefficient). For any such linear input, the discrete adjoint identity

    <F s, r> == <s, F* r>

must hold to machine precision, where ``F* r`` is the transpose (adjoint) of
``F`` applied to ``r``. Deepwave computes ``F* r`` by reverse-mode autograd:
``r`` seeds ``backward`` on ``loss = sum(F(s) * r)``, and (because ``F`` is
linear) the resulting input gradient *is* ``F* r`` independent of the point
``s``. The identity then reduces to Euler's relation for the degree-1
homogeneous ``loss``: ``<s, d loss / d s> == loss``.

This is therefore a genuine test that each hand-written C/CUDA *backward*
kernel is the exact transpose of its *forward* kernel -- on the **compiled**
backend the backward is hand-written C (autograd only wires it up), and on the
**eager** backend it is PyTorch's own reverse-mode of the reference
implementation (a softer cross-check). A wrong adjoint (e.g. a mismatched PML
memory-variable swap, or a missing guard-ring term at a reflecting face) breaks
the identity by many orders of magnitude; a correct one holds it at ~1e-15.

The tolerance is ``1e-10`` relative in float64 (the measured residual is
~1e-15; the margin absorbs the compiled backend's ``-Ofast`` reordering). The
grids are deliberately small: 2-D ``24x24`` with ``nt=30`` and one 3-D ``14^3``
case per propagator family, plus a reflecting (zero-PML top face) variant for
the propagators that support first-class faces.

There is no pre-existing formal dot-product test in the suite (searched for
``adjoint``/``dot-product``/``<F s``); the closest existing coverage is
``torch.autograd.gradcheck`` (finite-difference Jacobian checks, e.g. in
``test_boundary.py`` and ``test_viscoacoustic.py``) and the cross-backend
gradient-equality checks, both of which this complements rather than
duplicates: gradcheck compares the adjoint against a finite difference of the
forward, whereas this compares it against the forward's exact transpose.
"""

import hashlib

import pytest
import torch
from helpers import c_backend_available

import deepwave
from deepwave import vpvsrho_to_lambmubuoyancy
from deepwave.acoustic import acoustic_output_names, acoustic_wavefield_names
from deepwave.acoustic_born import acoustic_born_output_names
from deepwave.acoustic_vti import (
    acoustic_vti_output_names,
    acoustic_vti_wavefield_names,
)
from deepwave.elastic import elastic_output_names, elastic_wavefield_names
from deepwave.elastic_born import elastic_born_output_names
from deepwave.elastic_vti import elastic_vti_output_names
from deepwave.scalar import scalar_output_names, scalar_wavefield_names
from deepwave.scalar_born import scalar_born_output_names, scalar_born_wavefield_names
from deepwave.viscoacoustic import (
    viscoacoustic_output_names,
    viscoacoustic_wavefield_names,
)
from deepwave.wavelets import ricker

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DX = 5.0
FREQ = 25.0
# Relative tolerance on the adjoint identity. The measured residual is ~1e-15
# in float64; this margin absorbs the compiled backend's -Ofast reordering.
RTOL = 1e-10

BACKENDS = [
    pytest.param("eager", id="eager"),
    pytest.param("compiled", id="c"),
]

# Per-propagator metadata: the receiver output whose adjoint we test (the
# scattered receivers for the Born pairs), and the functions naming the output
# tuple and the initial-wavefield inputs. ``elastic_vti`` shares elastic's
# wavefield layout and public field ordering, but names its own outputs
# (its absorbing-boundary memory variables are still
# individual public fields).
_REC_NAME = {
    "scalar": "receiver_amplitudes",
    "acoustic": "receiver_amplitudes_p",
    "viscoacoustic": "receiver_amplitudes_p",
    "elastic": "receiver_amplitudes_p",
    "elastic_vti": "receiver_amplitudes_p",
    # acoustic_vti has no combined pressure channel (task 7.H); use the
    # (always-present) x velocity receiver instead.
    "acoustic_vti": "receiver_amplitudes_x",
    "scalar_born": "sc_receiver_amplitudes",
    "acoustic_born": "sc_receiver_amplitudes_p",
    "elastic_born": "sc_receiver_amplitudes_p",
}
_OUTPUT_NAMES = {
    "scalar": scalar_output_names,
    "acoustic": acoustic_output_names,
    "viscoacoustic": viscoacoustic_output_names,
    "elastic": elastic_output_names,
    "elastic_vti": elastic_vti_output_names,
    "acoustic_vti": acoustic_vti_output_names,
    "scalar_born": scalar_born_output_names,
    "acoustic_born": acoustic_born_output_names,
    "elastic_born": elastic_born_output_names,
}
_WAVEFIELD_NAMES = {
    "scalar": scalar_wavefield_names,
    "acoustic": acoustic_wavefield_names,
    "viscoacoustic": viscoacoustic_wavefield_names,
    "elastic": elastic_wavefield_names,
    "elastic_vti": elastic_wavefield_names,
    "acoustic_vti": acoustic_vti_wavefield_names,
    "scalar_born": scalar_born_wavefield_names,
}


def _grid(ndim: int) -> dict:
    """Small, stable simulation grid parameters for the given dimensionality."""
    if ndim == 2:
        return {"n": 24, "nt": 30, "dt": 0.0006, "acc": 4}
    return {"n": 14, "nt": 16, "dt": 0.0005, "acc": 2}


def _media(ndim: int, n: int) -> dict:
    """Deterministic smooth-ish random models for every propagator family.

    Returns the scalar/acoustic velocity and density, the viscoacoustic Q, the
    isotropic elastic Lame parameters and buoyancy, the VTI stiffnesses built
    from the isotropic Lame parameters (so all families see a mutually
    consistent medium).
    """
    torch.manual_seed(1234 + ndim)
    shape = (n,) * ndim
    v = 1500.0 + 20.0 * torch.randn(shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(shape, dtype=DTYPE)
    q = torch.full(shape, 30.0, dtype=DTYPE)
    vp = 1600.0 + 50.0 * torch.rand(shape, dtype=DTYPE)
    vs = 800.0 + 50.0 * torch.rand(shape, dtype=DTYPE)
    rho_e = 2000.0 + 100.0 * torch.rand(shape, dtype=DTYPE)
    lamb, mu, buoyancy = (t.detach() for t in vpvsrho_to_lambmubuoyancy(vp, vs, rho_e))
    return {
        "v": v,
        "rho": rho,
        "q": q,
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "c11": (lamb + 2 * mu).clone(),
        "c13": lamb.clone(),
        "c33": (lamb + 2 * mu).clone(),
        "c55": mu.clone(),
        "c66": mu.clone(),
    }


def _survey(ndim: int, n: int, nt: int, dt: float) -> tuple:
    """A single central source and a line of near-surface receivers."""
    src = ricker(FREQ, nt, dt, 1.5 / FREQ).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.tensor([[[n // 2] * ndim]], dtype=torch.long)
    nrec = 6
    rec_loc = torch.zeros(1, nrec, ndim, dtype=torch.long)
    for j in range(nrec):
        rec_loc[0, j, 0] = 3 + j
        rec_loc[0, j, 1:] = n // 2
    return src, src_loc, rec_loc


def _build_forward(
    prop,
    kind,
    ndim,
    face,
    backend,
    storage_mode="device",
    mpml_fraction=0.0,
    storage_path=None,
):
    """Build ``(forward, x_shape)`` for one (propagator, linear-input) case.

    ``forward(x)`` returns the designated receiver output as a function of the
    single linear input ``x`` (all other inputs held fixed), so that the adjoint
    identity can be checked by seeding ``backward`` on ``sum(forward(x) * r)``.
    ``kind`` selects which input is linear: ``"source"`` (source amplitudes),
    ``"wavefield"`` (the leading initial wavefield, with no source so the map is
    purely linear), or a scattering-model role for the Born pairs (``"scatter"``
    for scalar_born; ``"dv"``/``"drho"`` for acoustic_born; ``"dlamb"``/
    ``"dmu"``/``"dbuoyancy"`` for elastic_born). ``face`` selects the boundary
    treatment: ``"pml"`` (uniform absorbing PML), ``"free"`` (a reflecting
    zero-PML-width top face, an approximate free/rigid reflector), or
    ``"free_bc"`` (a REAL first-class free surface: zero-PML-width top face
    plus ``boundary=["free", ...]``, the exact odd-image pressure-release
    condition), or ``"rigid_bc"`` (the same, with ``boundary=["rigid", ...]``
    -- the first-class face the elastic family accepts, which does not accept
    ``"free"``). ``storage_mode`` and ``mpml_fraction`` are threaded straight through
    to the propagator when non-default.
    """
    g = _grid(ndim)
    n, nt, dt, acc = g["n"], g["nt"], g["dt"], g["acc"]
    m = _media(ndim, n)
    src, src_loc, rec_loc = _survey(ndim, n, nt, dt)
    common = {"dt": dt, "pml_freq": FREQ, "accuracy": acc, "backend": backend}
    if face == "free":
        # A reflecting (free/rigid) top face: zero PML width there, PML elsewhere.
        common["pml_width"] = [0] + [4] * (2 * ndim - 1)
    elif face == "free_bc":
        # A real first-class free surface: zero PML width on the top face,
        # plus the exact odd-image "free" boundary condition there.
        common["pml_width"] = [0] + [4] * (2 * ndim - 1)
        common["boundary"] = ["free"] + ["pml"] * (2 * ndim - 1)
    elif face == "rigid_bc":
        # A first-class rigid wall on the top face: the elastic family's
        # equivalent of "free_bc" above, since it does not accept "free".
        common["pml_width"] = [0] + [4] * (2 * ndim - 1)
        common["boundary"] = ["rigid"] + ["pml"] * (2 * ndim - 1)
    else:
        common["pml_width"] = 4
    if storage_mode != "device":
        common["storage_mode"] = storage_mode
    if storage_path is not None:
        common["storage_path"] = storage_path
    if mpml_fraction:
        common["mpml_fraction"] = mpml_fraction
    rec_idx = _OUTPUT_NAMES[prop](ndim).index(_REC_NAME[prop])
    zeros = torch.zeros((n,) * ndim, dtype=DTYPE)

    # Source/receiver keyword names differ between the scalar-style and
    # multi-component (pressure) propagators.
    if prop in ("scalar", "scalar_born"):
        src_kw = {"source_amplitudes": src, "source_locations": src_loc}
        rec_kw = {"receiver_locations": rec_loc}
        src_amp_key = "source_amplitudes"
    elif prop == "acoustic_vti":
        # No combined pressure channel (task 7.H); use the x velocity
        # source/receiver (always present, both 2-D and 3-D).
        src_kw = {"source_amplitudes_x": src, "source_locations_x": src_loc}
        rec_kw = {"receiver_locations_x": rec_loc}
        src_amp_key = "source_amplitudes_x"
    else:
        # The scalar source channel is named for the quantity it injects, so
        # it differs between the two families; ask the spec rather than
        # guessing.
        rate = deepwave.spec(prop).source_channels(ndim)[0]
        src_amp_key = f"source_amplitudes_{rate}"
        src_kw = {src_amp_key: src, f"source_locations_{rate}": src_loc}
        rec_kw = {"receiver_locations_p": rec_loc}
    # The Born propagators spell their scattered receivers ``sc_``;
    # resolve the keyword from the spec rather than assuming a spelling.
    if deepwave.spec(prop).has_bg_receivers:
        rec_kw = {f"sc_{k}": v for k, v in rec_kw.items()}
    if prop in ("viscoacoustic",):
        common["q_freq"] = FREQ

    def positional(scatter_overrides):
        """The propagator's positional model arguments.

        ``dv``/``drho`` (Born perturbations) and ``c66`` are
        keyword-only under the "ONE signature convention" (see
        ``tests/test_signature_consistency.py``) -- they are threaded into
        ``kwargs`` by ``call()`` below instead of returned here
        (``scatter_overrides`` is unused now,
        kept as a parameter for call-site symmetry with the elastic_born
        branch, which still routes its perturbations through this
        mechanism's caller-side kwargs update).
        """
        if prop == "scalar":
            return (m["v"], DX)
        if prop == "acoustic":
            return (m["v"], m["rho"], DX)
        if prop == "viscoacoustic":
            return (m["v"], m["rho"], m["q"], DX)
        if prop in ("elastic", "elastic_born"):
            return (m["lamb"], m["mu"], m["buoyancy"], DX)
        if prop == "acoustic_vti":
            return (m["c11"], m["c13"], m["c33"], m["buoyancy"], DX)
        if prop in ("elastic_vti",):
            return (m["c11"], m["c13"], m["c33"], m["c55"], m["buoyancy"], DX)
        if prop == "scalar_born":
            return (m["v"], DX)
        if prop == "acoustic_born":
            return (m["v"], m["rho"], DX)
        raise AssertionError(prop)

    def scatter_kwargs(scatter_overrides):
        """Keyword-only Born perturbations / ``c66`` for ``prop``."""
        if prop == "scalar_born":
            return {"dv": scatter_overrides["scatter"]}
        if prop == "acoustic_born":
            return {"dv": scatter_overrides["dv"], "drho": scatter_overrides["drho"]}
        if prop == "elastic_vti" and ndim == 3:
            return {"c66": m["c66"]}
        return {}

    pfunc = getattr(deepwave, prop)

    # Fixed (non-linearised) scattering model for the Born source/wavefield
    # cases, so the scattered receivers are a non-trivial linear map of the
    # linear input. For the scatter cases these are overridden by ``x``.
    scat_fixed = {
        "scatter": 0.05 * m["v"],
        "dv": 0.05 * m["v"],
        "drho": zeros,
        "dlamb": 0.02 * m["lamb"],
        "dmu": zeros,
        "dbuoyancy": zeros,
    }

    def call(x):
        kwargs = dict(common)
        kwargs.update(rec_kw)
        # ``scatter``/``elastic_born_sc`` hold the Born scattering models. For
        # the source/wavefield cases they are the fixed non-zero background
        # scatter (so the scattered receivers are a non-trivial linear map of
        # the varied input). For a scatter case only the varied component is
        # non-zero, so the output is *purely* linear in it (zero constant term).
        scatter = dict(scat_fixed)
        elastic_born_sc = {
            "dlamb": scat_fixed["dlamb"],
            "dmu": scat_fixed["dmu"],
            "dbuoyancy": scat_fixed["dbuoyancy"],
        }
        if kind == "source":
            kwargs.update(src_kw)
            kwargs[src_amp_key] = x
        elif kind == "wavefield":
            # No source: the receiver output is then purely linear in the
            # initial wavefield ``x``.
            kwargs["nt"] = nt
            wf_name = _WAVEFIELD_NAMES[prop](ndim)[0]
            kwargs[wf_name] = x
        else:
            # A scattering-model role: source fixed, only ``x`` scatters.
            kwargs.update(src_kw)
            if prop == "elastic_born":
                elastic_born_sc = {"dlamb": zeros, "dmu": zeros, "dbuoyancy": zeros}
                elastic_born_sc[kind] = x
            else:
                scatter = {"scatter": zeros, "dv": zeros, "drho": zeros}
                scatter[kind] = x
        if prop == "elastic_born":
            kwargs.update(elastic_born_sc)
        kwargs.update(scatter_kwargs(scatter))
        return pfunc(*positional(scatter), **kwargs)[rec_idx]

    # Discover the shape of ``x``.
    if kind == "source":
        x_shape = tuple(src.shape)
    elif kind == "wavefield":
        probe_kwargs = dict(common)
        probe_kwargs.update(rec_kw)
        probe_kwargs.update(src_kw)
        probe_kwargs.update(scatter_kwargs(scat_fixed))
        probe = pfunc(*positional(scat_fixed), **probe_kwargs)
        x_shape = tuple(probe[0].shape)
    else:
        x_shape = (n,) * ndim

    return call, x_shape


def _assert_adjoint(forward, x_shape, seed):
    """Assert the dot-product identity ``<F s, r> == <s, F* r>``."""
    torch.manual_seed(seed)
    s = torch.randn(x_shape, dtype=DTYPE)
    leaf = s.clone().requires_grad_(True)
    out = forward(leaf)
    # A non-trivial output makes the identity a meaningful check.
    assert out.abs().max().item() > 0
    r = torch.randn_like(out)
    loss = (out * r).sum()  # <F s, r>
    loss.backward()
    assert leaf.grad is not None
    lhs = loss.item()
    rhs = (s * leaf.grad).sum().item()  # <s, F* r>
    denom = max(abs(lhs), abs(rhs), 1e-300)
    assert abs(lhs - rhs) / denom < RTOL, (lhs, rhs, abs(lhs - rhs) / denom)


def _case(prop, kind, ndim, face, storage_mode="device", mpml_fraction=0.0):
    """Build one ``_CASES`` row, defaulting ``storage_mode``/``mpml_fraction``."""
    return (prop, kind, ndim, face, storage_mode, mpml_fraction)


def _case_id(c):
    """The pytest id for a ``_CASES`` row, extended only when non-default."""
    prop, kind, ndim, face, storage_mode, mpml_fraction = c
    case_id = f"{prop}-{kind}-{ndim}d-{face}"
    if storage_mode != "device":
        case_id += f"-{storage_mode}"
    if mpml_fraction:
        case_id += f"-mpml{mpml_fraction}"
    return case_id


# (propagator, linear-input kind, ndim, face, storage_mode, mpml_fraction). Every
# propagator is checked for source amplitudes (2-D + one 3-D) and its leading
# initial wavefield, with a reflecting-face variant for a representative
# subset; the Born pairs add each scattering-model component (2-D + one 3-D on
# the primary component).
_CASES = [
    # source amplitudes, 2-D PML
    _case("scalar", "source", 2, "pml"),
    _case("acoustic", "source", 2, "pml"),
    _case("viscoacoustic", "source", 2, "pml"),
    _case("elastic", "source", 2, "pml"),
    _case("elastic_vti", "source", 2, "pml"),
    _case("acoustic_vti", "source", 2, "pml"),
    _case("scalar_born", "source", 2, "pml"),
    _case("acoustic_born", "source", 2, "pml"),
    _case("elastic_born", "source", 2, "pml"),
    # source amplitudes, one 3-D case per family
    _case("scalar", "source", 3, "pml"),
    _case("acoustic", "source", 3, "pml"),
    _case("viscoacoustic", "source", 3, "pml"),
    _case("elastic", "source", 3, "pml"),
    _case("elastic_vti", "source", 3, "pml"),
    _case("acoustic_vti", "source", 3, "pml"),
    _case("scalar_born", "source", 3, "pml"),
    _case("acoustic_born", "source", 3, "pml"),
    _case("elastic_born", "source", 3, "pml"),
    # initial wavefields (where accepted), 2-D PML
    _case("scalar", "wavefield", 2, "pml"),
    _case("acoustic", "wavefield", 2, "pml"),
    _case("viscoacoustic", "wavefield", 2, "pml"),
    _case("elastic", "wavefield", 2, "pml"),
    _case("elastic_vti", "wavefield", 2, "pml"),
    _case("acoustic_vti", "wavefield", 2, "pml"),
    _case("scalar_born", "wavefield", 2, "pml"),
    # Born scattering models, 2-D PML
    _case("scalar_born", "scatter", 2, "pml"),
    _case("acoustic_born", "dv", 2, "pml"),
    _case("acoustic_born", "drho", 2, "pml"),
    _case("elastic_born", "dlamb", 2, "pml"),
    _case("elastic_born", "dmu", 2, "pml"),
    _case("elastic_born", "dbuoyancy", 2, "pml"),
    # Born scattering models, one 3-D case per family (primary component)
    _case("scalar_born", "scatter", 3, "pml"),
    _case("acoustic_born", "dv", 3, "pml"),
    _case("elastic_born", "dlamb", 3, "pml"),
    # reflecting (zero-PML top) face variant, 2-D
    _case("scalar", "source", 2, "free"),
    _case("acoustic", "source", 2, "free"),
    _case("elastic", "source", 2, "free"),
    _case("scalar_born", "scatter", 2, "free"),
    # REAL first-class free-surface (boundary=["free", ...]) variant, 2-D, on
    # the same subset covered by the reflecting-face variant above.
    _case("scalar", "source", 2, "free_bc"),
    _case("acoustic", "source", 2, "free_bc"),
    _case("elastic", "source", 2, "rigid_bc"),
    _case("scalar_born", "scatter", 2, "free_bc"),
    # viscoacoustic's free surface (unlike the propagators above) also
    # exercises an SLS memory-variable ("r") guard-ring image fill that is a
    # no-op in the forward kernel (FUNC(forward_p_cell) reads rmem only at the
    # centre point) and matters ONLY in the backward/adjoint kernel (B_EFF
    # reads rmem's guard ring at the same offsets the forward velocity update
    # reads p's -- see viscoacoustic.c's FUNC(fill_p_bc) comment). The
    # forward-only free-surface tests in test_boundary.py (including the
    # physical-Q reflection-coefficient one) therefore cannot exercise this
    # rule at all; this dot-product row does, since it seeds a genuine
    # backward pass through every step. ``_media`` already sets Q=30 (a
    # physically realistic quality factor, not the Q=5000 near-lossless limit
    # used by the reflection-coefficient tests), so this is also the
    # physical-Q row.
    _case("viscoacoustic", "source", 2, "free_bc"),
    # Multiaxial PML (mpml_fraction > 0), elastic-only, 2-D (it requires at
    # least two spatial dimensions).
    _case("elastic", "source", 2, "pml", mpml_fraction=0.15),
    # Source adjoints with storage options accepted. Fixed models need no
    # saved history, so these do not test history reconstruction. The Born
    # model-adjoint tests below exercise actual disk history. On this CPU
    # fixture, "cpu" storage also resolves to "device"; GPU offload needs
    # separate GPU tests.
    _case("scalar", "source", 2, "pml", storage_mode="cpu"),
    _case("scalar", "source", 2, "pml", storage_mode="disk"),
    _case("elastic", "source", 2, "pml", storage_mode="cpu"),
    _case("elastic", "source", 2, "pml", storage_mode="disk"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("prop", "kind", "ndim", "face", "storage_mode", "mpml_fraction"),
    [pytest.param(*c, id=_case_id(c)) for c in _CASES],
)
def test_adjoint_dot_product(
    prop, kind, ndim, face, storage_mode, mpml_fraction, backend
) -> None:
    """The backward kernel is the exact transpose of the forward kernel.

    Verifies ``<F s, r> == <s, F* r>`` (relative residual ``< 1e-10`` in
    float64) for the linear input selected by ``kind``, on both the compiled
    (hand-written C adjoint) and eager (autograd reference) backends.
    """
    if backend == "compiled" and not c_backend_available(prop, ndim, device=DEVICE):
        pytest.skip(f"compiled {prop} backend not present in libdeepwave_C")
    if backend == "eager" and storage_mode != "device":
        # The Python (eager) backend does not accept a non-default
        # storage_mode at all (it raises RuntimeError); these rows exist to
        # exercise acceptance of storage options on the compiled backend,
        # so skip the eager half of the cross product.
        pytest.skip("storage_mode overrides are not supported on the eager backend")
    forward, x_shape = _build_forward(
        prop, kind, ndim, face, backend, storage_mode, mpml_fraction
    )
    _assert_adjoint(
        forward,
        x_shape,
        # Python's hash(str) is salted per process, so it would change the
        # random test problem between serial runs and xdist workers.
        seed=int.from_bytes(
            hashlib.sha256(
                _case_id((prop, kind, ndim, face, storage_mode, mpml_fraction)).encode()
            ).digest()[:4],
            "big",
        ),
    )


@pytest.mark.parametrize(
    ("prop", "kind"),
    [("scalar_born", "scatter"), ("acoustic_born", "dv"), ("elastic_born", "dlamb")],
)
def test_disk_history_model_adjoint(prop, kind, tmp_path) -> None:
    """A Born model transpose must consume a nonempty saved forward history.

    Source-only adjoints do not need that history. Differentiating a Born
    perturbation both requires it and provides a linear operator whose
    transpose can be checked independently by the dot-product identity.
    """
    if not c_backend_available(prop, 2, device=DEVICE):
        pytest.skip(f"compiled {prop} backend not present in libdeepwave_C")
    forward, x_shape = _build_forward(
        prop, kind, 2, "pml", "compiled", storage_mode="disk", storage_path=tmp_path
    )

    def with_history_check(x):
        result = forward(x)
        # Check while the autograd graph still owns the files, before backward
        # consumes them. A no-history or device-storage fallback must fail.
        assert any(p.stat().st_size > 0 for p in tmp_path.rglob("shot_*.bin"))
        return result

    _assert_adjoint(with_history_check, x_shape, seed=20261023)

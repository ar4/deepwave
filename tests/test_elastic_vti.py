"""Tests for the elastic VTI propagator (deepwave.elastic_vti).

The validation strategy is:

1. **Isotropic limit** (the most important test): with ``c11 = c33 =
   lamb + 2 mu``, ``c13 = lamb``, ``c55 = c66 = mu``, the VTI propagator must
   reproduce the isotropic elastic propagator to machine precision, forward
   and gradients, in 2-D and 3-D, on both the eager and compiled C backends.
2. **Analytic phase velocities**: plane-wave qP and qSV phase speeds in a
   homogeneous, strongly anisotropic medium are measured with a
   slowness-decomposition method (delayed line source with along-line
   slowness p; the transverse slowness q is measured from the delay between
   two receivers) and compared against the exact Christoffel solution.
3. **Reciprocity** and **gradient checks** (cross-backend eager-vs-C
   agreement, plus absolute finite-difference ``torch.autograd.gradcheck``
   where nothing else covers the path: the multiaxial-PML gradients, the
   Thomsen/Alkhalifah converters themselves, and one chained through
   ``thomsen_to_cij`` into the propagator).
"""

import math
import re
import sys

import pytest
import torch
from helpers import c_backend_available, output_tensors

import deepwave
import deepwave.common
from deepwave import _common

DEVICE = torch.device("cpu")
DTYPE = torch.float64

C_BACKEND = pytest.mark.skipif(
    not c_backend_available("elastic_vti", device=DEVICE),
    reason="compiled elastic_vti backend not present in libdeepwave_C",
)

# Backends to run correctness tests on: the eager Python reference and, when
# built, the compiled C backend (backend="compiled").
BACKENDS = [
    pytest.param("eager", id="eager"),
    pytest.param("compiled", id="c", marks=C_BACKEND),
]


def _iso_cij(lamb, mu, ndim):
    """Cij tensors for the isotropic special case (differentiable)."""
    c11 = lamb + 2 * mu
    c13 = lamb * 1.0
    c33 = lamb + 2 * mu
    c55 = mu * 1.0
    c66 = mu * 1.0 if ndim == 3 else None
    return c11, c13, c33, c55, c66


def _make_iso_media(ndim, n=12, batch=False):
    torch.manual_seed(1)
    shape = (n,) * ndim
    if batch:
        shape = (2, *shape)
    vp = 1500 + 300 * torch.rand(*shape, dtype=DTYPE)
    vs = 700 + 200 * torch.rand(*shape, dtype=DTYPE)
    rho = 2000 + 300 * torch.rand(*shape, dtype=DTYPE)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    return lamb.detach(), mu.detach(), buoyancy.detach()


def _survey_kwargs(ndim, nt, n=12, n_shots=1, backend="eager", dt=0.001):
    """Sources on every component + pressure, receivers likewise."""
    torch.manual_seed(2)
    freq = 15.0
    peak_time = 1.5 / freq
    amp = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .to(DTYPE)
        .reshape(1, 1, -1)
        .repeat(n_shots, 1, 1)
    )
    src_loc = torch.tensor([[[n // 2] * ndim]], dtype=torch.long).repeat(n_shots, 1, 1)
    rec_loc = torch.tensor([[[n // 2 - 2] * ndim]], dtype=torch.long).repeat(
        n_shots, 1, 1
    )
    kwargs = {
        "dt": dt,
        "source_amplitudes_x": amp,
        "source_locations_x": src_loc,
        "source_amplitudes_pressure_rate": amp,
        "source_locations_pressure_rate": src_loc,
        "receiver_locations_x": rec_loc,
        "receiver_locations_p": rec_loc,
        "pml_width": 3,
        "accuracy": 2,
        "max_vel": 2500.0,
        "backend": backend,
        "pml_freq": freq,
    }
    if ndim >= 2:
        kwargs["source_amplitudes_y"] = amp
        kwargs["source_locations_y"] = src_loc
        kwargs["receiver_locations_y"] = rec_loc
    if ndim >= 3:
        kwargs["source_amplitudes_z"] = amp
        kwargs["source_locations_z"] = src_loc
        kwargs["receiver_locations_z"] = rec_loc
    return kwargs


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("backend", BACKENDS)
def test_isotropic_limit(ndim, backend):
    """VTI with isotropic Cij == elastic, forward and gradients.

    This is the single most important correctness test for the VTI
    propagator: with c11 = c33 = lamb + 2 mu, c13 = lamb, c55 = c66 = mu it
    must match the isotropic elastic propagator to machine precision
    (float64), including all model gradients (chained back to lamb, mu,
    buoyancy through the Cij construction).
    """
    lamb, mu, buoyancy = _make_iso_media(ndim)
    nt = 30
    kwargs = _survey_kwargs(ndim, nt, backend=backend)

    lamb_e = lamb.clone().requires_grad_(True)
    mu_e = mu.clone().requires_grad_(True)
    b_e = buoyancy.clone().requires_grad_(True)
    out_e = deepwave.elastic(lamb_e, mu_e, b_e, 5.0, **kwargs)

    lamb_v = lamb.clone().requires_grad_(True)
    mu_v = mu.clone().requires_grad_(True)
    b_v = buoyancy.clone().requires_grad_(True)
    c11, c13, c33, c55, c66 = _iso_cij(lamb_v, mu_v, ndim)
    out_v = deepwave.elastic_vti(c11, c13, c33, c55, b_v, 5.0, c66=c66, **kwargs)

    assert len(out_e) == len(out_v)
    for a, b in zip(output_tensors(out_e), output_tensors(out_v)):
        assert a.shape == b.shape
        if a.numel() == 0:
            continue
        scale = a.abs().max().item()
        assert (a - b).abs().max().item() <= max(scale, 1.0) * 1e-12

    sum(o.pow(2).sum() for o in output_tensors(out_e) if o.numel() > 0).backward()
    sum(o.pow(2).sum() for o in output_tensors(out_v) if o.numel() > 0).backward()
    for ge, gv in [
        (lamb_e.grad, lamb_v.grad),
        (mu_e.grad, mu_v.grad),
        (b_e.grad, b_v.grad),
    ]:
        scale = ge.abs().max().item()
        assert (ge - gv).abs().max().item() <= max(scale, 1.0) * 1e-11


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mpml_fraction", [0.0, 0.15])
@C_BACKEND
def test_isotropic_limit_c_equals_eager_anisotropic(ndim, mpml_fraction):
    """Unsampled C adjoints agree with autograd in anisotropic PML and M-PML."""
    lamb, mu, buoyancy = _make_iso_media(ndim)
    # Make it anisotropic: c11 != c33, c13 != lamb, c66 != c55.
    c11 = (lamb + 2 * mu) * 1.3
    c13 = lamb * 1.1
    c33 = lamb + 2 * mu
    c55 = mu
    c66 = mu * 1.2 if ndim == 3 else None
    nt = 30
    outs = {}
    grads = {}
    for backend in ["eager", "compiled"]:
        kwargs = _survey_kwargs(ndim, nt, backend=backend, dt=0.0003)
        kwargs["mpml_fraction"] = mpml_fraction
        models = [
            m.clone().requires_grad_(True)
            for m in [c11, c13, c33, c55, buoyancy] + ([c66] if c66 is not None else [])
        ]
        if ndim == 3:
            c11_r, c13_r, c33_r, c55_r, b_r, c66_r = models
        else:
            c11_r, c13_r, c33_r, c55_r, b_r = models
            c66_r = None
        out = deepwave.elastic_vti(
            c11_r, c13_r, c33_r, c55_r, b_r, 5.0, c66=c66_r, **kwargs
        )
        # Exact adjoint comparison requires unsampled model gradients. The
        # strided approximation is tested in test_gradient_sampling_characterisation.
        assert out.info.step_ratio == out.info.model_gradient_stride == 1
        sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0).backward()
        outs[backend] = out
        grads[backend] = [m.grad for m in models]
    for a, b in zip(output_tensors(outs["eager"]), output_tensors(outs["compiled"])):
        if a.numel() == 0:
            continue
        scale = a.abs().max().item()
        torch.testing.assert_close(b, a, rtol=0, atol=scale * 5e-12)
    for ga, gb in zip(grads["eager"], grads["compiled"]):
        scale = ga.abs().max().item()
        assert scale > 0, "the reference must exercise every model gradient"
        torch.testing.assert_close(gb, ga, rtol=0, atol=scale * 1e-8)


def test_ndim_1_raises():
    """1-D inputs are rejected (VTI degenerates to isotropic in 1-D)."""
    c = torch.ones(10, dtype=DTYPE)
    amp = torch.zeros(1, 1, 10, dtype=DTYPE)
    loc = torch.tensor([[[5]]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"degenerates|at least 2"):
        deepwave.elastic_vti(
            c,
            c,
            c,
            c,
            c,
            5.0,
            0.001,
            c66=None,
            source_amplitudes_x=amp,
            source_locations_x=loc,
        )


def test_c66_must_be_none_in_2d():
    """Passing c66 in 2-D raises (it does not enter the P-SV equations)."""
    c = torch.ones(10, 10, dtype=DTYPE)
    amp = torch.zeros(1, 1, 10, dtype=DTYPE)
    loc = torch.tensor([[[5, 5]]], dtype=torch.long)
    with pytest.raises(ValueError, match="c66 must be None in 2-D"):
        deepwave.elastic_vti(
            c,
            0.5 * c,
            c,
            c,
            c,
            5.0,
            0.001,
            c66=0.25 * c,
            source_amplitudes_x=amp,
            source_locations_x=loc,
        )


def test_c66_required_in_3d():
    """Omitting c66 in 3-D raises."""
    c = torch.ones(10, 10, 10, dtype=DTYPE)
    amp = torch.zeros(1, 1, 10, dtype=DTYPE)
    loc = torch.tensor([[[5, 5, 5]]], dtype=torch.long)
    with pytest.raises(ValueError, match="c66 is required in 3-D"):
        deepwave.elastic_vti(
            c,
            c,
            c,
            c,
            c,
            5.0,
            0.001,
            c66=None,
            source_amplitudes_x=amp,
            source_locations_x=loc,
        )


def test_elastic_vti_model_names_rejects_bad_ndim():
    """elastic_vti_model_names raises for ndim outside {2, 3} (1-D degenerates)."""
    mod = sys.modules["deepwave.elastic_vti"]
    with pytest.raises(ValueError, match="only 2 or 3 spatial dimensions"):
        mod.elastic_vti_model_names(1)
    with pytest.raises(ValueError, match="only 2 or 3 spatial dimensions"):
        mod.elastic_vti_model_names(4)


def test_prepare_parameters_requires_c66_in_3d():
    """The public prepare_parameters helper rejects c66=None in 3-D directly.

    ``prepare_parameters`` infers spatial ndim as ``c55.ndim - 1`` (the
    leading axis is the shot dimension of an already-padded model, as passed
    internally by ``elastic_vti()``), so a 3-D case needs a 4-D tensor here.
    """
    c55 = torch.ones(1, 6, 6, 6, dtype=DTYPE)
    buoyancy = torch.ones(1, 6, 6, 6, dtype=DTYPE)
    mod = sys.modules["deepwave.elastic_vti"]
    with pytest.raises(ValueError, match="c66 is required in 3-D"):
        mod.prepare_parameters(c55, None, buoyancy)


def test_module_matches_functional():
    """The ElasticVti module (2-D, c66=None) reproduces the functional interface."""
    ndim = 2
    lamb, mu, buoyancy = _make_iso_media(ndim, n=10)
    c11, c13, c33, c55, c66 = _iso_cij(lamb, mu, ndim)
    kwargs = _survey_kwargs(ndim, nt=12, n=10, backend="eager")
    dt = kwargs.pop("dt")
    prop = deepwave.ElasticVti(c11, c13, c33, c55, buoyancy, 5.0, c66=c66)
    out_m = prop(dt, **kwargs)
    out_f = deepwave.elastic_vti(
        c11, c13, c33, c55, buoyancy, 5.0, dt, c66=c66, **kwargs
    )
    for name in ("receiver_amplitudes_x", "vy_0", "vx_0"):
        assert torch.equal(getattr(out_m, name), getattr(out_f, name))


@pytest.mark.slow
def test_compile_backend_matches_eager():
    """backend='torch_compile' reproduces the eager backend (iso limit)."""
    ndim = 2
    lamb, mu, buoyancy = _make_iso_media(ndim, n=10)
    c11, c13, c33, c55, c66 = _iso_cij(lamb, mu, ndim)
    out_eager = deepwave.elastic_vti(
        c11, c13, c33, c55, buoyancy, 5.0, c66=c66, **_survey_kwargs(ndim, nt=12, n=10)
    )
    for backend in ("torch_compile",):
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            5.0,
            c66=c66,
            **_survey_kwargs(ndim, nt=12, n=10, backend=backend),
        )
        assert torch.allclose(
            out.receiver_amplitudes_x, out_eager.receiver_amplitudes_x, atol=1e-9
        )


def test_scope_and_validation_errors():
    """z-locations in <3-D, mpml_fraction range, bad location, vmap/max_vel."""
    ndim = 2
    lamb, mu, buoyancy = _make_iso_media(ndim, n=10)
    c11, c13, c33, c55, c66 = _iso_cij(lamb, mu, ndim)
    amp = torch.zeros(1, 1, 5, dtype=DTYPE)
    loc = torch.tensor([[[5, 5]]], dtype=torch.long)
    with pytest.raises(ValueError, match="z dimension should not be provided"):
        deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            5.0,
            0.001,
            c66=c66,
            source_amplitudes_x=amp,
            source_locations_x=loc,
            source_locations_z=torch.tensor([[[0, 0]]], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="mpml_fraction"):
        deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            5.0,
            0.001,
            c66=c66,
            source_amplitudes_x=amp,
            source_locations_x=loc,
            mpml_fraction=1.0,
        )
    with pytest.raises(RuntimeError, match="must be less"):
        # source_locations_x is checked against the x (last) dimension's
        # extent, so it is the location's *second* (x) coordinate that must
        # be out of bounds, not the first (y).
        deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            5.0,
            0.001,
            c66=c66,
            source_amplitudes_x=amp,
            source_locations_x=torch.tensor([[[0, 9]]], dtype=torch.long),
        )
    with pytest.raises(RuntimeError, match="max_vel"):
        torch.func.vmap(
            lambda c11_b: (
                deepwave.elastic_vti(
                    c11_b,
                    c13,
                    c33,
                    c55,
                    buoyancy,
                    5.0,
                    0.001,
                    c66=c66,
                    source_amplitudes_x=amp,
                    source_locations_x=loc,
                ).receiver_amplitudes_x
            )
        )(c11.unsqueeze(0))
    vmap_out = torch.func.vmap(
        lambda c11_b: (
            deepwave.elastic_vti(
                c11_b,
                c13,
                c33,
                c55,
                buoyancy,
                5.0,
                0.001,
                c66=c66,
                source_amplitudes_x=amp,
                source_locations_x=loc,
                max_vel=3000.0,
                backend="eager",
            ).receiver_amplitudes_x
        )
    )(c11.unsqueeze(0))
    assert torch.isfinite(vmap_out).all()


class TestThomsen:
    """Tests of the exact Thomsen-to-Cij conversion helper."""

    def test_exact_relations(self):
        """The produced Cij satisfy Thomsen's exact definitions."""
        vp = torch.tensor([3000.0], dtype=DTYPE)
        vs = torch.tensor([1500.0], dtype=DTYPE)
        rho = torch.tensor([2000.0], dtype=DTYPE)
        eps = torch.tensor([0.24], dtype=DTYPE)
        delta = torch.tensor([0.1], dtype=DTYPE)
        gamma = torch.tensor([0.15], dtype=DTYPE)
        c11, c13, c33, c55, c66, buoyancy = deepwave.anisotropy.thomsen_to_cij(
            vp, vs, rho, eps, delta, gamma
        )
        assert torch.allclose(buoyancy, 1 / rho)
        assert torch.allclose(c33, rho * vp**2)
        assert torch.allclose(c55, rho * vs**2)
        assert torch.allclose((c11 - c33) / (2 * c33), eps)
        assert torch.allclose((c66 - c55) / (2 * c55), gamma)
        delta_back = ((c13 + c55) ** 2 - (c33 - c55) ** 2) / (2 * c33 * (c33 - c55))
        assert torch.allclose(delta_back, delta)
        assert bool((c13 + c55 > 0).all())

    def test_isotropic_gives_lame(self):
        """Eps = delta = gamma = 0 reduces to the isotropic Lame moduli."""
        vp = torch.tensor([3000.0], dtype=DTYPE)
        vs = torch.tensor([1500.0], dtype=DTYPE)
        rho = torch.tensor([2000.0], dtype=DTYPE)
        zero = torch.zeros(1, dtype=DTYPE)
        c11, c13, c33, c55, c66, buoyancy = deepwave.anisotropy.thomsen_to_cij(
            vp, vs, rho, zero, zero, zero
        )
        lamb, mu, buoyancy_iso = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
        # The buoyancy is the same guarded reciprocal as the isotropic
        # converter's, bit for bit.
        assert torch.equal(buoyancy, buoyancy_iso)
        assert torch.allclose(c11, lamb + 2 * mu)
        assert torch.allclose(c33, lamb + 2 * mu)
        assert torch.allclose(c13, lamb)
        assert torch.allclose(c55, mu)
        assert torch.allclose(c66, mu)

    def test_vacuum_cells_and_gradients(self):
        """Vacuum (vp=vs=rho=0) cells give zero Cij and no NaN gradients."""
        vp = torch.tensor([3000.0, 0.0], dtype=DTYPE, requires_grad=True)
        vs = torch.tensor([1500.0, 0.0], dtype=DTYPE, requires_grad=True)
        rho = torch.tensor([2000.0, 0.0], dtype=DTYPE, requires_grad=True)
        eps = torch.tensor([0.2, 0.0], dtype=DTYPE, requires_grad=True)
        delta = torch.tensor([0.1, 0.0], dtype=DTYPE, requires_grad=True)
        c11, c13, c33, c55, _, buoyancy = deepwave.anisotropy.thomsen_to_cij(
            vp, vs, rho, eps, delta
        )
        for c in (c11, c13, c33, c55):
            assert c[1].item() == 0.0
        # rho == 0 maps to buoyancy == 0 exactly (the propagators' vacuum
        # sentinel), not to inf.
        assert buoyancy[1].item() == 0.0
        assert torch.isfinite(buoyancy).all()
        (c11.sum() + c13.sum() + c33.sum() + c55.sum() + buoyancy.sum()).backward()
        for t in (vp, vs, rho, eps, delta):
            assert torch.isfinite(t.grad).all()

    def test_first_five_outputs_are_bit_identical_to_the_five_tuple(self):
        """Adding buoyancy did not perturb the stiffnesses by a single ulp.

        The stiffnesses are propagator inputs covered by a bit-exactness
        fingerprint, so they are computed here exactly as the pre-buoyancy
        implementation did and compared with torch.equal, not allclose.
        """
        vp = torch.tensor([3000.0, 2500.0, 0.0], dtype=DTYPE)
        vs = torch.tensor([1500.0, 1200.0, 0.0], dtype=DTYPE)
        rho = torch.tensor([2000.0, 2200.0, 0.0], dtype=DTYPE)
        eps = torch.tensor([0.24, 0.1, 0.0], dtype=DTYPE)
        delta = torch.tensor([0.1, -0.05, 0.0], dtype=DTYPE)
        gamma = torch.tensor([0.15, 0.2, 0.0], dtype=DTYPE)

        # The old implementation, verbatim.
        c33_old = rho * vp**2
        c55_old = rho * vs**2
        c11_old = c33_old * (1 + 2 * eps)
        dc = c33_old - c55_old
        arg = 2 * delta * c33_old * dc + dc**2
        positive = arg > 0
        arg_safe = torch.where(positive, arg, torch.ones_like(arg))
        c13_old = (
            torch.where(positive, torch.sqrt(arg_safe), torch.zeros_like(arg)) - c55_old
        )
        c66_old = c55_old * (1 + 2 * gamma)

        c11, c13, c33, c55, c66, buoyancy = deepwave.anisotropy.thomsen_to_cij(
            vp, vs, rho, eps, delta, gamma
        )
        assert torch.equal(c11, c11_old)
        assert torch.equal(c13, c13_old)
        assert torch.equal(c33, c33_old)
        assert torch.equal(c55, c55_old)
        assert c66 is not None
        assert torch.equal(c66, c66_old)
        assert buoyancy[-1].item() == 0.0

    def test_validation_errors(self):
        """Invalid inputs raise clear errors."""
        one = torch.ones(1, dtype=DTYPE)
        with pytest.raises(ValueError, match="c33 >= c55"):
            deepwave.anisotropy.thomsen_to_cij(one, 2 * one, one, one * 0, one * 0)
        with pytest.raises(ValueError, match="non-negative"):
            deepwave.anisotropy.thomsen_to_cij(one, one, -one, one * 0, one * 0)
        # delta below the branch bound: needs delta >= -(c33-c55)/(2 c33).
        vp = 2 * one
        vs = one
        # c33 = 4 rho, c55 = rho -> bound = -3/8.
        with pytest.raises(ValueError, match="square root"):
            deepwave.anisotropy.thomsen_to_cij(vp, vs, one, one * 0, -0.5 * one)

    def test_cij_to_thomsen_round_trips_and_validates(self):
        """cij_to_thomsen inverts thomsen_to_cij and rejects negative buoyancy."""
        vp = torch.tensor([3000.0], dtype=DTYPE)
        vs = torch.tensor([1500.0], dtype=DTYPE)
        rho = torch.tensor([2000.0], dtype=DTYPE)
        eps = torch.tensor([0.24], dtype=DTYPE)
        delta = torch.tensor([0.1], dtype=DTYPE)
        gamma = torch.tensor([0.15], dtype=DTYPE)
        cij = deepwave.anisotropy.thomsen_to_cij(vp, vs, rho, eps, delta, gamma)
        c11, c13, c33, c55, c66, buoyancy = cij
        # The two functions compose directly: cij_to_thomsen takes exactly
        # what thomsen_to_cij returns, in the same order.
        vp2, vs2, rho2, eps2, delta2, gamma2 = deepwave.anisotropy.cij_to_thomsen(*cij)
        assert torch.allclose(vp2, vp)
        assert torch.allclose(vs2, vs)
        assert torch.allclose(rho2, rho)
        assert torch.allclose(eps2, eps)
        assert torch.allclose(delta2, delta)
        assert gamma2 is not None
        assert torch.allclose(gamma2, gamma)
        # ...and back again, returning the Thomsen arguments in the order
        # thomsen_to_cij takes them.
        round_tripped = deepwave.anisotropy.thomsen_to_cij(
            vp2, vs2, rho2, eps2, delta2, gamma2
        )
        # An explicit length check rather than ``zip(..., strict=True)``:
        # ``strict`` is Python 3.10+ and Deepwave supports 3.9.
        assert len(round_tripped) == len(cij)
        for produced, expected in zip(round_tripped, cij):
            assert torch.allclose(produced, expected)
        # gamma is None when c66 is None.
        *_rest, gamma3 = deepwave.anisotropy.cij_to_thomsen(
            c11, c13, c33, c55, None, buoyancy
        )
        assert gamma3 is None
        with pytest.raises(ValueError, match="non-negative"):
            deepwave.anisotropy.cij_to_thomsen(c11, c13, c33, c55, c66, -buoyancy)

    def test_cij_to_thomsen_vacuum_cells(self):
        """A vacuum cell round-trips as all-zero Thomsen parameters."""
        vp = torch.tensor([3000.0, 0.0], dtype=DTYPE, requires_grad=True)
        vs = torch.tensor([1500.0, 0.0], dtype=DTYPE, requires_grad=True)
        rho = torch.tensor([2000.0, 0.0], dtype=DTYPE, requires_grad=True)
        eps = torch.tensor([0.24, 0.0], dtype=DTYPE, requires_grad=True)
        delta = torch.tensor([0.1, 0.0], dtype=DTYPE, requires_grad=True)
        gamma = torch.tensor([0.15, 0.0], dtype=DTYPE, requires_grad=True)
        cij = deepwave.anisotropy.thomsen_to_cij(vp, vs, rho, eps, delta, gamma)
        back = deepwave.anisotropy.cij_to_thomsen(*cij)
        # The solid cell round-trips; the vacuum cell becomes all zeros
        # (nothing is recoverable there), with no NaN or inf anywhere.
        thomsen = (vp, vs, rho, eps, delta, gamma)
        assert len(back) == len(thomsen)
        for produced, expected in zip(back, thomsen):
            assert produced is not None
            assert torch.isfinite(produced).all()
            assert produced[1].item() == 0.0
            assert torch.allclose(produced[0], expected[0])
        # Those zeros are the canonical vacuum representative: feeding them
        # back reproduces the same all-zero stiffnesses and buoyancy.
        re_fed = deepwave.anisotropy.thomsen_to_cij(*back)
        assert len(re_fed) == len(cij)
        for produced, expected in zip(re_fed, cij):
            assert produced[1].item() == 0.0
            assert torch.allclose(produced, expected)
        # Gradients through the masked vacuum cell stay finite.
        sum(t.sum() for t in back if t is not None).backward()
        for t in (vp, vs, rho, eps, delta, gamma):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()

    def test_gradcheck_through_thomsen(self):
        """FD gradcheck of thomsen_to_cij itself."""
        vp = torch.tensor([3000.0], dtype=DTYPE, requires_grad=True)
        vs = torch.tensor([1500.0], dtype=DTYPE, requires_grad=True)
        rho = torch.tensor([2000.0], dtype=DTYPE, requires_grad=True)
        eps = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
        delta = torch.tensor([0.1], dtype=DTYPE, requires_grad=True)
        gamma = torch.tensor([0.15], dtype=DTYPE, requires_grad=True)
        assert torch.autograd.gradcheck(
            deepwave.anisotropy.thomsen_to_cij, (vp, vs, rho, eps, delta, gamma)
        )


class TestAlkhalifahToThomsen:
    """Tests of the Alkhalifah-Tsvankin-to-Thomsen conversion helper."""

    def test_hand_computed_values(self):
        """Matches hand-computed epsilon/delta for several (vp, vnmo, eta)."""
        # Each case is (vp, vnmo, eta, expected_delta, expected_epsilon), with
        # delta/epsilon from delta = (vnmo**2/vp**2 - 1) / 2 and
        # epsilon = eta*(1 + 2*delta) + delta.
        cases = [
            (3000.0, 3000.0, 0.0, 0.0, 0.0),  # elliptical, vnmo == vp.
            (3000.0, 3200.0, 0.2, 0.0688888888889, 0.2964444444444),
            (2000.0, 2100.0, 0.1, 0.05125, 0.1615),
            (4000.0, 3800.0, -0.05, -0.04875, -0.093875),
        ]
        for vp_v, vnmo_v, eta_v, expected_delta, expected_epsilon in cases:
            vp = torch.tensor([vp_v], dtype=DTYPE)
            vnmo = torch.tensor([vnmo_v], dtype=DTYPE)
            eta = torch.tensor([eta_v], dtype=DTYPE)
            epsilon, delta = deepwave.anisotropy.alkhalifah_to_thomsen(vp, vnmo, eta)
            assert torch.allclose(
                delta, torch.tensor([expected_delta], dtype=DTYPE), atol=1e-10
            )
            assert torch.allclose(
                epsilon, torch.tensor([expected_epsilon], dtype=DTYPE), atol=1e-10
            )

    def test_round_trips_through_alkhalifah_tsvankin_definitions(self):
        """delta/epsilon feed back into vnmo/eta via their defining relations."""
        vp = torch.tensor([2500.0, 3100.0], dtype=DTYPE)
        vnmo = torch.tensor([2600.0, 3400.0], dtype=DTYPE)
        eta = torch.tensor([0.15, 0.05], dtype=DTYPE)
        epsilon, delta = deepwave.anisotropy.alkhalifah_to_thomsen(vp, vnmo, eta)
        vnmo_back = vp * torch.sqrt(1 + 2 * delta)
        eta_back = (epsilon - delta) / (1 + 2 * delta)
        assert torch.allclose(vnmo_back, vnmo)
        assert torch.allclose(eta_back, eta)

    def test_gradcheck(self):
        """FD gradcheck of alkhalifah_to_thomsen itself."""
        vp = torch.tensor([3000.0], dtype=DTYPE, requires_grad=True)
        vnmo = torch.tensor([3200.0], dtype=DTYPE, requires_grad=True)
        eta = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
        assert torch.autograd.gradcheck(
            deepwave.anisotropy.alkhalifah_to_thomsen, (vp, vnmo, eta)
        )

    def test_thomsen_to_alkhalifah_round_trips_both_ways(self):
        """thomsen_to_alkhalifah is the exact inverse of alkhalifah_to_thomsen."""
        vp = torch.tensor([2500.0, 3100.0, 4000.0], dtype=DTYPE)
        vnmo = torch.tensor([2600.0, 3400.0, 3800.0], dtype=DTYPE)
        eta = torch.tensor([0.15, 0.05, -0.05], dtype=DTYPE)
        epsilon, delta = deepwave.anisotropy.alkhalifah_to_thomsen(vp, vnmo, eta)
        vnmo_back, eta_back = deepwave.anisotropy.thomsen_to_alkhalifah(
            vp, epsilon, delta
        )
        assert torch.allclose(vnmo_back, vnmo)
        assert torch.allclose(eta_back, eta)
        epsilon_back, delta_back = deepwave.anisotropy.alkhalifah_to_thomsen(
            vp, vnmo_back, eta_back
        )
        assert torch.allclose(epsilon_back, epsilon)
        assert torch.allclose(delta_back, delta)

    def test_thomsen_to_alkhalifah_matches_the_defining_relations(self):
        """The returned vnmo/eta are Alkhalifah and Tsvankin's definitions."""
        vp = torch.tensor([2500.0, 3100.0], dtype=DTYPE)
        epsilon = torch.tensor([0.2, 0.05], dtype=DTYPE)
        delta = torch.tensor([0.1, -0.02], dtype=DTYPE)
        vnmo, eta = deepwave.anisotropy.thomsen_to_alkhalifah(vp, epsilon, delta)
        assert torch.allclose(vnmo, vp * torch.sqrt(1 + 2 * delta))
        assert torch.allclose(eta, (epsilon - delta) / (1 + 2 * delta))

    def test_thomsen_to_alkhalifah_rejects_delta_at_or_below_minus_half(self):
        """A delta of -0.5 or less would make vnmo imaginary, and is rejected."""
        vp = torch.tensor([2500.0], dtype=DTYPE)
        epsilon = torch.tensor([0.2], dtype=DTYPE)
        with pytest.raises(ValueError, match=re.escape("delta > -0.5")):
            deepwave.anisotropy.thomsen_to_alkhalifah(
                vp, epsilon, torch.tensor([-0.5], dtype=DTYPE)
            )

    def test_thomsen_to_alkhalifah_gradcheck(self):
        """FD gradcheck of thomsen_to_alkhalifah itself."""
        vp = torch.tensor([3000.0], dtype=DTYPE, requires_grad=True)
        epsilon = torch.tensor([0.3], dtype=DTYPE, requires_grad=True)
        delta = torch.tensor([0.1], dtype=DTYPE, requires_grad=True)
        assert torch.autograd.gradcheck(
            deepwave.anisotropy.thomsen_to_alkhalifah, (vp, epsilon, delta)
        )

    def test_gradcheck_chained_through_thomsen_to_cij(self):
        """FD gradcheck of the full alkhalifah_to_thomsen -> thomsen_to_cij chain.

        Velocities/density are rescaled to O(1) leaves (and the Cij outputs
        rescaled back down) for the same reason as the propagator tiny
        gradchecks below: ``gradcheck``'s ``eps`` is an ABSOLUTE
        perturbation, so O(1e3) leaves and O(1e10) outputs would make some
        finite-difference entries pure float64 rounding noise (this is
        sharpened here because ``c11`` is, by an exact identity of this
        parameterisation, independent of ``vp`` -- ``c11 = rho * vnmo**2 *
        (1 + 2*eta)`` -- so its analytical ``d(c11)/d(vp)`` is exactly zero
        and has no signal to compete with that noise at the unscaled
        magnitude).
        """
        vp = torch.tensor([3.0], dtype=DTYPE, requires_grad=True)  # x1000 m/s
        vnmo = torch.tensor([3.2], dtype=DTYPE, requires_grad=True)  # x1000 m/s
        eta = torch.tensor([0.2], dtype=DTYPE, requires_grad=True)
        vs = torch.tensor([1.5], dtype=DTYPE, requires_grad=True)  # x1000 m/s
        rho = torch.tensor([2.0], dtype=DTYPE, requires_grad=True)  # x1000 kg/m^3

        def chain(
            vp: torch.Tensor,
            vnmo: torch.Tensor,
            eta: torch.Tensor,
            vs: torch.Tensor,
            rho: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            epsilon, delta = deepwave.anisotropy.alkhalifah_to_thomsen(
                vp * 1000, vnmo * 1000, eta
            )
            c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
                vp * 1000, vs * 1000, rho * 1000, epsilon, delta
            )
            scale = 1e9
            return c11 / scale, c13 / scale, c33 / scale, c55 / scale

        assert torch.autograd.gradcheck(chain, (vp, vnmo, eta, vs, rho))


# ---------------------------------------------------------------------------
# Analytic phase-velocity validation (exact 2-D VTI Christoffel solution)
# ---------------------------------------------------------------------------

VP0, VS0, RHO0 = 3000.0, 1500.0, 2000.0
EPS_T, DELTA_T = 0.3, 0.1


def _christoffel_constants():
    one = torch.ones(1, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, EPS_T * one, DELTA_T * one
    )
    return float(c11), float(c13), float(c33), float(c55)


def _christoffel_speeds(theta, c11, c13, c33, c55):
    """Exact 2-D VTI phase speeds (qP, qSV) at angle theta from the axis."""
    s2 = math.sin(theta) ** 2
    c2 = math.cos(theta) ** 2
    g11 = c11 * s2 + c55 * c2
    g33 = c55 * s2 + c33 * c2
    g13 = (c13 + c55) * math.sin(theta) * math.cos(theta)
    tr = g11 + g33
    det = g11 * g33 - g13 * g13
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    return (
        math.sqrt((tr / 2 + disc) / RHO0),
        math.sqrt((tr / 2 - disc) / RHO0),
    )


def _q_of_p(p, wave, orient, c11, c13, c33, c55):
    """Transverse slowness from the dispersion relation (quadratic in q^2)."""
    if orient == "h":
        a = c55 * c33
        b = (
            c55 * (c55 * p * p - RHO0)
            + c33 * (c11 * p * p - RHO0)
            - (c13 + c55) ** 2 * p * p
        )
        c = (c11 * p * p - RHO0) * (c55 * p * p - RHO0)
    else:
        a = c11 * c55
        b = (
            c11 * (c33 * p * p - RHO0)
            + c55 * (c55 * p * p - RHO0)
            - (c13 + c55) ** 2 * p * p
        )
        c = (c55 * p * p - RHO0) * (c33 * p * p - RHO0)
    disc = math.sqrt(max(b * b - 4 * a * c, 0.0))
    roots = sorted(((-b - disc) / (2 * a), (-b + disc) / (2 * a)))
    u = roots[0] if wave == "qP" else roots[1]
    if u < 0:
        raise ValueError(f"{wave} evanescent at p={p}")
    return math.sqrt(u)


def _window_at(tr, dt, freq, t_center):
    period = 1.0 / freq
    nt = tr.shape[0]
    t = torch.arange(nt, dtype=DTYPE) * dt
    w = torch.zeros(nt, dtype=DTYPE)
    inside = (t >= t_center - 1.5 * period) & (t <= t_center + 1.5 * period)
    w[inside] = torch.hann_window(int(inside.sum()), periodic=False, dtype=DTYPE)
    return tr * w


def _measure_delay(x1, x2, dt, freq):
    """Delay of x2 relative to x1: coarse xcorr peak + residual phase fit."""
    nt = x1.shape[0]
    x1_f = torch.fft.rfft(x1)
    x2_f = torch.fft.rfft(x2)
    freqs = torch.fft.rfftfreq(nt, dt)
    cross = x2_f * x1_f.conj()
    corr = torch.fft.irfft(cross, n=nt)
    lag = int(corr.argmax().item())
    if lag > nt // 2:
        lag -= nt
    coarse = lag * dt
    omega_all = 2 * math.pi * freqs
    cross_res = cross * torch.exp(1j * omega_all * coarse)
    band = (freqs > 0.5 * freq) & (freqs < 1.5 * freq)
    w = cross_res.abs()[band]
    omega = omega_all[band]
    phase = torch.angle(cross_res[band])
    residual = -(w * phase * omega).sum() / (w * omega * omega).sum()
    return coarse + float(residual)


def _run_plane_wave_case(p_slow, wave, orient, wide, backend):
    """Measure (theta, v) of a plane wave with along-line slowness p_slow.

    A line of sources fired with delay gradient p launches plane waves whose
    along-line slowness is exactly p; the transverse slowness q is measured
    from the arrival-time difference between two receivers separated
    transversely on the line's perpendicular bisector. Then
    v = 1/sqrt(p^2+q^2) and theta follows from atan2. The source line is
    apodised (edge diffraction otherwise outruns slow plane waves) and the
    traces are windowed around the predicted arrival, so a wrong kernel
    moves the arrival out of the window and fails loudly.
    """
    c11c, c13c, c33c, c55c = _christoffel_constants()
    across = 200
    along = 200 + 300 * wide
    ny, nx = (across, along) if orient == "h" else (along, across)
    dx = 5.0
    dt = 0.0005
    freq = 14.0

    line_extent = (along // 2 - 30) * dx
    peak_time = 1.5 / freq + p_slow * line_extent
    q_exp = _q_of_p(p_slow, wave, orient, c11c, c13c, c33c, c55c)
    t_far = peak_time + q_exp * (140 - 10) * dx + 3.0 / freq
    nt = math.ceil(t_far / dt)

    shape = (ny, nx)
    one = torch.ones(shape, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, EPS_T * one, DELTA_T * one
    )
    buoyancy = one / RHO0

    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time).to(DTYPE)
    line = torch.arange(30, along - 30, dtype=torch.long)
    src_loc = torch.zeros(1, line.shape[0], 2, dtype=torch.long)
    center = along // 2
    if orient == "h":
        src_loc[0, :, 0] = 10
        src_loc[0, :, 1] = line
        rec_loc = torch.tensor([[[100, center], [140, center]]], dtype=torch.long)
    else:
        src_loc[0, :, 0] = line
        src_loc[0, :, 1] = 10
        rec_loc = torch.tensor([[[center, 100], [center, 140]]], dtype=torch.long)
    delays = p_slow * (line.to(DTYPE) - float(center)) * dx
    amps = deepwave.transducers.apply_delays(wavelet, delays, dt)
    n_line = line.shape[0]
    n_tap = n_line // 4
    taper = torch.ones(n_line, dtype=DTYPE)
    ramp = torch.hann_window(2 * n_tap, periodic=False, dtype=DTYPE)
    taper[:n_tap] = ramp[:n_tap]
    taper[-n_tap:] = ramp[n_tap:]
    amps = (amps * taper.unsqueeze(-1)).unsqueeze(0)

    kwargs = {
        "pml_width": 20,
        "accuracy": 8,
        "backend": backend,
        "pml_freq": freq,
    }
    if wave == "qP":
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            c66=None,
            source_amplitudes_pressure_rate=amps,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            **kwargs,
        )
        rec = out[-3]
    else:
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            c66=None,
            source_amplitudes_x=amps,
            source_locations_x=src_loc,
            receiver_locations_x=rec_loc,
            **kwargs,
        )
        rec = out[-1]

    tr1 = _window_at(rec[0, 0, :], dt, freq, peak_time + q_exp * (100 - 10) * dx)
    tr2 = _window_at(rec[0, 1, :], dt, freq, peak_time + q_exp * (140 - 10) * dx)
    dy = (140 - 100) * dx
    q = _measure_delay(tr1, tr2, dt, freq) / dy
    v_meas = 1.0 / math.sqrt(p_slow**2 + q**2)
    if orient == "h":
        theta = math.atan2(p_slow, q)
    else:
        theta = math.pi / 2 - math.atan2(p_slow, q)
    v_qp, v_qsv = _christoffel_speeds(theta, c11c, c13c, c33c, c55c)
    v_ref = v_qp if wave == "qP" else v_qsv
    return theta, v_meas, v_ref


# (p, wave, orient, wide, nominal angle). The qSV case at p=4.12e-4 lies
# beyond the maximum horizontal qP slowness, so only qSV propagates there.
PLANE_WAVE_CASES = [
    pytest.param(0.0, "qP", "h", 1, 0.0, id="qP-0deg"),
    pytest.param(1.5e-4, "qP", "h", 2, 28.0, id="qP-28deg"),
    pytest.param(2.13e-4, "qP", "h", 2, 45.0, id="qP-45deg"),
    pytest.param(1.0e-4, "qP", "v", 1, 68.5, id="qP-68deg"),
    pytest.param(0.0, "qP", "v", 1, 90.0, id="qP-90deg"),
    pytest.param(0.0, "qSV", "h", 1, 0.0, id="qSV-0deg"),
    pytest.param(4.12e-4, "qSV", "h", 1, 45.0, id="qSV-45deg"),
]


@pytest.mark.parametrize(
    ("p_slow", "wave", "orient", "wide", "angle"), PLANE_WAVE_CASES
)
@C_BACKEND
def test_phase_velocity_christoffel(p_slow, wave, orient, wide, angle):
    """Measured plane-wave phase speeds match the exact Christoffel solution.

    Homogeneous strongly-anisotropic medium (Thomsen eps=0.3, delta=0.1,
    vp0/vs0 = 2). Tolerance 2% (grid-dispersion + measurement budget; the
    observed errors are <= ~0.2% at most angles). This is the test that
    detects axis-convention or constitutive-coupling errors, which the
    isotropic limit cannot (c11 = c33 there).
    """
    theta, v_meas, v_ref = _run_plane_wave_case(
        p_slow, wave, orient, wide, backend="compiled"
    )
    assert abs(math.degrees(theta) - angle) < 6.0
    assert abs(v_meas - v_ref) / v_ref < 0.02


@pytest.mark.parametrize(
    ("p_slow", "wave", "orient", "wide", "angle"),
    [pytest.param(0.0, "qP", "h", 1, 0.0, id="qP-0deg")],
)
def test_phase_velocity_christoffel_eager(p_slow, wave, orient, wide, angle):
    """One (cheap) eager-backend phase-velocity case.

    The full angle sweep runs on the C backend; the eager backend is tied to
    it by the cross-backend tests, but this direct case keeps the eager
    physics covered even when the C backend is unavailable.
    """
    theta, v_meas, v_ref = _run_plane_wave_case(
        p_slow, wave, orient, wide, backend="eager"
    )
    assert abs(math.degrees(theta) - angle) < 6.0
    assert abs(v_meas - v_ref) / v_ref < 0.02


# ---------------------------------------------------------------------------
# Reciprocity
# ---------------------------------------------------------------------------


@C_BACKEND
def test_reciprocity_2d():
    """Source/receiver reciprocity for same-component pairs (vx-vx).

    For force sources and velocity receivers of the same component,
    swapping source and receiver locations must give the same trace in an
    anisotropic (VTI) medium (elastodynamic reciprocity).
    """
    torch.manual_seed(3)
    ny, nx = 60, 70
    dx = 5.0
    dt = 0.001
    nt = 200
    freq = 15.0
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        3000 * one, 1500 * one, 2000 * one, 0.25 * one, 0.1 * one
    )
    buoyancy = one / 2000.0
    amp = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    loc_a = torch.tensor([[[25, 20]]], dtype=torch.long)
    loc_b = torch.tensor([[[35, 50]]], dtype=torch.long)

    def run(src, rec):
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            c66=None,
            source_amplitudes_x=amp,
            source_locations_x=src,
            receiver_locations_x=rec,
            pml_width=20,
            accuracy=4,
            pml_freq=freq,
        )
        return out[-1]

    d_ab = run(loc_a, loc_b)
    d_ba = run(loc_b, loc_a)
    scale = d_ab.abs().max().item()
    assert (d_ab - d_ba).abs().max().item() < 5e-3 * scale


# ---------------------------------------------------------------------------
# Gradchecks
# ---------------------------------------------------------------------------


def _tiny_problem(ndim, backend, requires_grad_models=True, mpml_fraction=0.0):
    """A tiny anisotropic problem returning (inputs, wrapper) for gradcheck.

    Two conditioning requirements (both mirror
    ``tests/test_acoustic_vti.py``'s ``_tiny_problem``, where each is
    explained at length):

    * ``dt`` is deliberately courant-safe so ``step_ratio == 1`` at
      ``max_vel=2000`` in BOTH 2-D and 3-D (the 3-D CFL bound is the
      tighter one -- see ``_COURANT_SAFE_DT``'s module comment in
      ``tests/test_acoustic_vti.py`` on the 2-D-vs-3-D CFL trap). The
      compiled backend's model gradients are the exact discrete adjoint
      only at ``step_ratio == 1``; at ``step_ratio > 1`` they are, by
      documented design, a strided rectangle-rule quadrature of the
      adjoint time integral (``docs/what_deepwave_calculates.rst``), which
      an FD gradcheck of the forward must not be compared against. The
      previous ``dt=0.001`` gave ``step_ratio == 2`` in 3-D only, making
      the 3-D compiled case fail on exactly that (documented, bounded)
      quadrature difference while 2-D and eager passed.
    * ``gradcheck``'s ``eps`` (default 1e-6) is an ABSOLUTE perturbation.
      The Pa-scale stiffnesses (~1e9) are rescaled to O(1-10) leaf
      tensors (below float64's local ULP, the unscaled perturbation makes
      the numerical stiffness Jacobian exactly zero, vacuously matching
      near-zero analytical entries), and the buoyancy (~5e-4) to O(1)
      (unscaled, eps is a 0.2% relative perturbation whose FD truncation
      error exceeds eager's rtol). ``wrapper`` multiplies back, so the
      physical medium is unchanged.
    """
    torch.manual_seed(4)
    n = 8
    nt = 16
    dt = 0.0005
    assert _common.cfl_condition_n([5.0] * ndim, dt, 2000.0)[1] == 1
    scale = 1e9
    bscale = 5e-4
    shape = (n,) * ndim
    base = 1e9 * (1 + 0.1 * torch.rand(*shape, dtype=DTYPE))
    c33 = 4 * base
    c11 = 5 * base
    c55 = 1 * base
    c13 = 2 * base
    c66 = 1.2 * base if ndim == 3 else None
    buoyancy = 1 / (2000 * (1 + 0.1 * torch.rand(*shape, dtype=DTYPE)))
    amp = torch.randn(1, 1, nt, dtype=DTYPE)
    src_loc = torch.tensor([[[n // 2] * ndim]], dtype=torch.long)
    rec_loc = torch.tensor([[[n // 2 - 1] * ndim]], dtype=torch.long)

    models = [c11, c13, c33, c55] + ([c66] if c66 is not None else [])
    models = [(m / scale).requires_grad_(requires_grad_models) for m in models]
    buoyancy = (buoyancy / bscale).requires_grad_(requires_grad_models)
    amp = amp.clone().requires_grad_(True)

    def wrapper(*tensors):
        if ndim == 3:
            c11_t, c13_t, c33_t, c55_t, c66_t, b_t, amp_t = tensors
            c66_t = c66_t * scale
        else:
            c11_t, c13_t, c33_t, c55_t, b_t, amp_t = tensors
            c66_t = None
        out = deepwave.elastic_vti(
            c11_t * scale,
            c13_t * scale,
            c33_t * scale,
            c55_t * scale,
            b_t * bscale,
            5.0,
            dt,
            c66=c66_t,
            source_amplitudes_x=amp_t,
            source_locations_x=src_loc,
            source_amplitudes_pressure_rate=amp_t,
            source_locations_pressure_rate=src_loc,
            receiver_locations_x=rec_loc,
            receiver_locations_p=rec_loc,
            pml_width=2,
            accuracy=2,
            max_vel=2000.0,
            backend=backend,
            pml_freq=15.0,
            mpml_fraction=mpml_fraction,
        )
        # Receiver outputs are the trailing (ndim + 1) slots, pressure first
        # then velocity components; only x and p receivers are set above (y/z
        # are unrequested and so return empty tensors), so pick x (last) and
        # p (first of the (ndim + 1) receiver slots) explicitly rather than
        # by a fixed negative offset that silently degenerates to an empty
        # tensor if the trailing-slot layout changes.
        return out[-1], out[-(ndim + 1)]

    return (*models, buoyancy, amp), wrapper


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradcheck_fd_mpml(backend):
    """FD gradcheck with multiaxial PML enabled (2-D).

    Uses a tiny float64 problem so torch.autograd.gradcheck is affordable;
    see ``_tiny_problem``'s docstring for the two conditioning requirements
    (``step_ratio == 1`` and leaf rescaling) that make this an exact-adjoint
    gate rather than a comparison against the documented ``step_ratio > 1``
    gradient quadrature. The finite-difference tolerances below do not define
    the accuracy required of the compiled adjoint: the routine
    ``test_isotropic_limit_c_equals_eager_anisotropic`` comparison checks its
    M-PML gradients directly against autograd with a much tighter tolerance.
    """
    inputs, wrapper = _tiny_problem(2, backend, mpml_fraction=0.15)
    atol, rtol = (1e-8, 1e-4) if backend == "eager" else (1e-5, 1e-3)
    assert torch.autograd.gradcheck(
        wrapper, inputs, atol=atol, rtol=rtol, nondet_tol=1e-10
    )


def test_gradcheck_fd_through_thomsen():
    """FD gradcheck chained through thomsen_to_cij (Thomsen-space FWI)."""
    torch.manual_seed(5)
    n = 8
    nt = 8
    dt = 0.001
    shape = (n,) * 2
    vp = (3000 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))).requires_grad_(True)
    vs = (1500 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))).requires_grad_(True)
    rho = (2000 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))).requires_grad_(True)
    eps = (0.2 * torch.ones(*shape, dtype=DTYPE)).requires_grad_(True)
    delta = (0.1 * torch.ones(*shape, dtype=DTYPE)).requires_grad_(True)
    amp = torch.randn(1, 1, nt, dtype=DTYPE)
    src_loc = torch.tensor([[[n // 2] * 2]], dtype=torch.long)
    rec_loc = torch.tensor([[[n // 2 - 1] * 2]], dtype=torch.long)

    def wrapper(vp_t, vs_t, rho_t, eps_t, delta_t):
        c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
            vp_t, vs_t, rho_t, eps_t, delta_t
        )
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            1 / rho_t,
            5.0,
            dt,
            c66=None,
            source_amplitudes_pressure_rate=amp,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_width=2,
            accuracy=2,
            max_vel=4500.0,
            backend="eager",
            pml_freq=15.0,
        )
        return out[-3]

    assert torch.autograd.gradcheck(
        wrapper, (vp, vs, rho, eps, delta), atol=1e-8, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# M-PML and miscellaneous behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
@C_BACKEND
def test_mpml_c_equals_eager(ndim):
    """C and eager backends agree with multiaxial PML enabled."""
    lamb, mu, buoyancy = _make_iso_media(ndim)
    c11 = (lamb + 2 * mu) * 1.3
    c13 = lamb
    c33 = lamb + 2 * mu
    c55 = mu
    c66 = mu * 1.2 if ndim == 3 else None
    nt = 30
    outs = {}
    for backend in ["eager", "compiled"]:
        kwargs = _survey_kwargs(ndim, nt, backend=backend)
        kwargs["mpml_fraction"] = 0.2
        outs[backend] = deepwave.elastic_vti(
            c11, c13, c33, c55, buoyancy, 5.0, c66=c66, **kwargs
        )
    for a, b in zip(output_tensors(outs["eager"]), output_tensors(outs["compiled"])):
        if a.numel() == 0:
            continue
        scale = a.abs().max().item()
        assert (a - b).abs().max().item() <= max(scale, 1.0) * 5e-12


@C_BACKEND
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mpml", [False, True], ids=["default", "mpml"])
def test_c_threading_matches_single_thread(ndim, mpml):
    """Intra-shot OpenMP threading (B2) does not change the compiled result.

    ``elastic_vti.c`` was ported from one-shot-per-thread parallelism to
    intra-shot spatial parallelism (see ``elastic_vti.c``'s ``FUNC(forward)``/
    ``FUNC(backward)``, mirroring ``elastic.c``, ``viscoacoustic.c``, and
    ``acoustic_vti.c``'s B2 ports): each time step's velocity-update loops
    (vz/vy/vx) and the four stress-update loops (the combined sigmaii normal
    stress, plus the independent sigmaxy/sigmaxz/sigmayz shear stresses) are
    now split across threads via ``#pragma omp for simd ... nowait``, with
    shots processed serially rather than one-shot-per-thread. Every grid cell
    is written by exactly one thread per parallel region (the PML-region
    tiling partitions the grid disjointly): the combined (grad_c11, grad_c12,
    grad_c13, grad_c33) accumulation happens together, at the same cell
    index, inside the single sigmaii parallel region (see
    ``elastic_vti_kernels.h``'s ``FUNC(backward_sigmaii_cell)``), and each
    independent shear gradient (grad_c55_zy/grad_c55_zx/grad_c_xy) is
    accumulated only inside its own shear component's region/sub-loop. Time
    steps and shots remain barrier-separated (one implicit barrier per omp
    parallel region, one serial shot loop), so the arithmetic performed for
    any given cell does not depend on how many threads are used -- forward
    and gradients should be identical (to measurement precision) whether the
    compiled backend runs with 1 or 4 threads.

    Parametrized over the M-PML build variant (``mpml_fraction``, see
    ``test_mpml_c_equals_eager`` above) as well as the default PML, since
    ``DW_MPML`` only changes the ``DW_PML_SKIP`` macro and the P_A/P_B
    full-grid profile accessors in ``elastic_vti_kernels.h`` -- not the loop
    nesting, PML-region tiling, or omp pragma placement that this port
    touches -- so the same race-freedom argument applies to both compiled
    variants, and both are exercised here rather than assumed.

    This is a single-shot (``n_shots=1``, the default from
    ``_survey_kwargs``) problem, so under the old one-shot-per-thread scheme
    extra threads would have sat idle; here they should be put to work within
    the single shot instead (this is what ``spatial_parallel=True`` in
    ``elastic_vti.py``'s ``SPEC`` enables).

    Uses ``torch.set_num_threads`` (read by the backend via
    ``torch.get_num_threads()`` at call time, exactly as ``OMP_NUM_THREADS``
    would set the default) rather than an env var, so the comparison runs
    in-process; the tolerance mirrors ``test_mpml_c_equals_eager``'s cross-
    backend-parity pattern above (scaled to at least ``1.0``), not a new one.
    """
    saved_threads = torch.get_num_threads()
    try:
        lamb, mu, buoyancy = _make_iso_media(ndim)
        c11 = (lamb + 2 * mu) * 1.3
        c13 = lamb * 1.0
        c33 = lamb + 2 * mu
        c55 = mu * 1.0
        c66 = mu * 1.2 if ndim == 3 else None
        nt = 24
        kwargs = _survey_kwargs(ndim, nt, backend="compiled")
        if mpml:
            kwargs["mpml_fraction"] = 0.2

        def run(n_threads):
            torch.set_num_threads(n_threads)
            c11_ = c11.clone().requires_grad_(True)
            c13_ = c13.clone().requires_grad_(True)
            c33_ = c33.clone().requires_grad_(True)
            c55_ = c55.clone().requires_grad_(True)
            b_ = buoyancy.clone().requires_grad_(True)
            c66_ = c66.clone().requires_grad_(True) if c66 is not None else None
            out = deepwave.elastic_vti(
                c11_, c13_, c33_, c55_, b_, 5.0, c66=c66_, **kwargs
            )
            loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
            loss.backward()
            grads = [c11_.grad, c13_.grad, c33_.grad, c55_.grad, b_.grad]
            if c66_ is not None:
                grads.append(c66_.grad)
            return out, grads

        out_1, g_1 = run(1)
        out_4, g_4 = run(4)
        for a, b in zip(output_tensors(out_1), output_tensors(out_4)):
            if a.numel() == 0:
                continue
            scale = max(a.abs().max().item(), 1.0)
            assert (a - b).abs().max().item() <= scale * 1e-6
        for g1, g4 in zip(g_1, g_4):
            scale = max(g1.abs().max().item(), 1.0)
            assert (g1 - g4).abs().max().item() <= scale * 1e-6
    finally:
        torch.set_num_threads(saved_threads)


@pytest.mark.parametrize("backend", BACKENDS)
def test_named_output_and_initial_state_roundtrip(backend):
    """PropagatorOutput field names match, and outputs feed back as initial_state."""
    lamb, mu, buoyancy = _make_iso_media(2)
    c11, c13, c33, c55, _ = _iso_cij(lamb, mu, 2)
    kwargs = _survey_kwargs(2, 20, backend=backend)
    out = deepwave.elastic_vti(c11, c13, c33, c55, buoyancy, 5.0, c66=None, **kwargs)
    assert hasattr(out, "vy_0")
    assert hasattr(out, "receiver_amplitudes_p")
    # Continue the simulation from the returned state.
    out2 = deepwave.elastic_vti(
        c11, c13, c33, c55, buoyancy, 5.0, c66=None, initial_state=out, **kwargs
    )
    assert out2[-3].shape == out.receiver_amplitudes_p.shape


def test_batched_models_eager():
    """Per-shot batched models run and differ across shots (eager)."""
    lamb, mu, buoyancy = _make_iso_media(2, batch=True)
    # Make the second shot's medium anisotropic so outputs differ.
    c11, c13, c33, c55, _ = _iso_cij(lamb, mu, 2)
    c11 = c11.clone()
    c11[1] *= 1.5
    kwargs = _survey_kwargs(2, 20, n_shots=2, backend="eager")
    out = deepwave.elastic_vti(c11, c13, c33, c55, buoyancy, 5.0, c66=None, **kwargs)
    rec = out[-3]
    assert rec.shape[0] == 2
    assert not torch.allclose(rec[0], rec[1])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ndim", [2, 3])
def test_forward_and_gradient_cpu_gpu_match(ndim):
    """Compiled backend: CPU and GPU forward output and gradients match.

    Mirrors ``test_elastic.py``'s ``test_forward_cpu_gpu_match``, extended to
    also compare gradients: elastic_vti (unlike elastic_born) has a full
    compiled CUDA adjoint, so its gradients -- not just its forward output --
    should agree between CPU and GPU. Runs only on the maintainer's GPU
    (skipped here, no CUDA on this box); its value while skipped is making the
    presence of GPU coverage for elastic_vti visible.
    """
    lamb, mu, buoyancy = _make_iso_media(ndim, n=10)
    c11, c13, c33, c55, c66 = _iso_cij(lamb, mu, ndim)
    kwargs = _survey_kwargs(ndim, nt=20, n=10, backend="compiled")

    def run(device):
        c11d = c11.to(device).clone().requires_grad_(True)
        c13d = c13.to(device).clone().requires_grad_(True)
        c33d = c33.to(device).clone().requires_grad_(True)
        c55d = c55.to(device).clone().requires_grad_(True)
        c66d = c66.to(device).clone().requires_grad_(True) if c66 is not None else None
        bd = buoyancy.to(device).clone().requires_grad_(True)
        dkwargs = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        out = deepwave.elastic_vti(c11d, c13d, c33d, c55d, bd, 5.0, c66=c66d, **dkwargs)
        loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
        loss.backward()
        leaves = [c11d, c13d, c33d, c55d, bd]
        if c66d is not None:
            leaves.append(c66d)
        grads = [leaf.grad.detach().cpu() for leaf in leaves]
        outputs = [o.detach().cpu() for o in output_tensors(out)]
        return outputs, grads

    out_cpu, grads_cpu = run(torch.device("cpu"))
    out_gpu, grads_gpu = run(torch.device("cuda"))
    for a, b in zip(out_cpu, out_gpu):
        if a.numel() == 0:
            continue
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(grads_cpu, grads_gpu):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


# --- compiled backward boundary-face adjoint fix --------------------------
# Regression tests for a bug in which elastic_vti.c/elastic_vti.cu's compiled
# backward re-applied the forward boundary guard-ring fill to the adjoint
# fields instead of that fill's true transpose (the same bug class fixed for
# elastic.c/elastic.cu -- see that file's own equivalent tests). elastic_vti's
# multi-field fill (unlike acoustic's single-field pressure/velocity fill) is
# not self-adjoint, so this produced model gradients wrong whenever any
# ``boundary=`` face was non-PML. Uses a long enough ``nt`` for several
# source-to-face-to-receiver round trips, so a wrong adjoint fill is not
# masked by numerical dust (short-``nt`` gradchecks elsewhere in this file
# never exercise a non-PML face at all).


def _isotropic_vti_media(
    shape: tuple[int, ...], dtype: torch.dtype = torch.double
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VTI stiffnesses reducing exactly to isotropic elastic.

    rho/c/cs match test_elastic.py's ``_isotropic_elastic_media`` constants:
    ``c11 = c33 = lamb + 2*mu``, ``c13 = lamb``, ``c55 = mu``.
    """
    rho, c, cs = 1000.0, 1500.0, 900.0
    lamb = rho * (c**2 - 2 * cs**2)
    mu = rho * cs**2
    c11 = torch.full(shape, lamb + 2 * mu, dtype=dtype)
    c13 = torch.full(shape, lamb, dtype=dtype)
    c33 = torch.full(shape, lamb + 2 * mu, dtype=dtype)
    c55 = torch.full(shape, mu, dtype=dtype)
    buoyancy = torch.full(shape, 1.0 / rho, dtype=dtype)
    return c11, c13, c33, c55, buoyancy


_ELASTIC_VTI_C_BACKEND = pytest.mark.skipif(
    not c_backend_available("elastic_vti", device=DEVICE),
    reason="compiled elastic_vti backend not present in libdeepwave_C",
)


@_ELASTIC_VTI_C_BACKEND
@pytest.mark.parametrize(
    ("boundary", "pml_width"),
    [
        (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
        (["rigid", "pml", "pml", "pml"], [0, 0, 10, 10]),
        (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
        (["rigid", "pml", "rigid", "pml"], [0, 0, 0, 0]),
    ],
    ids=[
        "rigid_lo_only",
        "rigid_lo_and_bare_hi",
        "rigid_lo_transverse",
        "rigid_lo_both_axes",
    ],
)
def test_elastic_vti_compiled_backward_boundary_adjoint_parity(
    boundary: list[str], pml_width: list[int]
) -> None:
    """Compiled backward matches eager for c11/c13/c33/c55/buoyancy/source grads.

    Negative control: run against elastic_vti.c/.cu reverted to the pre-fix
    (committed 3254816) state and rebuilt, the mixed free/rigid/pml
    configuration of this test fails with c11 grad relative error ~0.26 and
    c55 grad relative error ~0.09, matching the pre-fix baseline documented
    in this repository's B5-fix-elastic task notes for the isotropic-elastic
    analogue of this bug.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    c11_0, c13_0, c33_0, c55_0, buoyancy0 = _isotropic_vti_media((ny, nx))
    # Perturb c55 off the exact isotropic reduction so a bug specific to the
    # VTI shear/normal-stress coupling (not present in isotropic elastic)
    # would also show up here.
    torch.manual_seed(0)
    c55_0 = c55_0 * (1.0 + 0.08 * torch.rand(ny, nx, dtype=torch.double))
    sa0 = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw: dict[str, object] = {
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": pml_width,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def run(backend: object) -> tuple[torch.Tensor, ...]:
        c11 = c11_0.clone().requires_grad_(True)
        c13 = c13_0.clone().requires_grad_(True)
        c33 = c33_0.clone().requires_grad_(True)
        c55 = c55_0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        sa = sa0.clone().requires_grad_(True)
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            backend=backend,
            **kw,
        ).receiver_amplitudes_p
        return torch.autograd.grad(out.sum(), [c11, c13, c33, c55, buoyancy, sa])

    g_c = run("compiled")
    g_e = run("eager")
    for name, gc, ge in zip(
        ["c11", "c13", "c33", "c55", "buoyancy", "source_amplitudes_pressure_rate"],
        g_c,
        g_e,
    ):
        denom = ge.abs().max()
        rel = (gc - ge).abs().max() / denom
        assert rel < 1e-9, f"{name} grad rel err {rel.item()} boundary={boundary}"


@_ELASTIC_VTI_C_BACKEND
def test_elastic_vti_compiled_backward_boundary_adjoint_parity_3d() -> None:
    """3-D counterpart of the 2-D adjoint-parity test above, mixed faces."""
    nz = ny = nx = 10
    dx, dt, nt = 6.0, 0.00012, 400
    torch.manual_seed(1)

    def _media() -> tuple[torch.Tensor, ...]:
        vp = 1500 + 100 * torch.rand(nz, ny, nx, dtype=torch.double)
        vs = 800 + 60 * torch.rand(nz, ny, nx, dtype=torch.double)
        rho = 2200 + 50 * torch.rand(nz, ny, nx, dtype=torch.double)
        c33 = rho * vp**2
        c11 = c33 * 1.1
        c55 = rho * vs**2
        c13 = c33 - 2 * c55
        c66 = c55 * 1.05
        buoyancy = 1 / rho
        return c11, c13, c33, c55, c66, buoyancy

    c11_0, c13_0, c33_0, c55_0, c66_0, buoyancy0 = _media()
    sa0 = deepwave.wavelets.ricker(20.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw: dict[str, object] = {
        "source_amplitudes_pressure_rate": sa0,
        "source_locations_pressure_rate": torch.tensor([[[5, 5, 5]]]),
        "receiver_locations_p": torch.tensor([[[3, 3, 3]]]),
        "pml_width": [0, 0, 0, 0, 5, 5],
        "boundary": ["rigid", "pml", "rigid", "pml", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 20.0,
    }

    def run(backend: object) -> tuple[torch.Tensor, ...]:
        c11 = c11_0.clone().requires_grad_(True)
        c13 = c13_0.clone().requires_grad_(True)
        c33 = c33_0.clone().requires_grad_(True)
        c55 = c55_0.clone().requires_grad_(True)
        c66 = c66_0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = deepwave.elastic_vti(
            c11, c13, c33, c55, buoyancy, dx, dt, c66=c66, backend=backend, **kw
        ).receiver_amplitudes_p
        return torch.autograd.grad(out.sum(), [c11, c13, c33, c55, c66, buoyancy])

    g_c = run("compiled")
    g_e = run("eager")
    for name, gc, ge in zip(["c11", "c13", "c33", "c55", "c66", "buoyancy"], g_c, g_e):
        rel = (gc - ge).abs().max() / ge.abs().max()
        assert rel < 1e-9, f"{name} grad rel err {rel.item()} (3-D)"


@_ELASTIC_VTI_C_BACKEND
def test_elastic_vti_compiled_backward_boundary_fd_arbitration() -> None:
    """Central-FD arbitration at the max-|g| cell (independent of eager).

    Rather than trusting eager as ground truth, perturb the model directly
    at the cell with the largest-magnitude eager gradient and compare the
    compiled analytic gradient there against a central finite difference of
    the compiled forward loss.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    c11_0, c13_0, c33_0, c55_0, buoyancy0 = _isotropic_vti_media((ny, nx))
    sa0 = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw: dict[str, object] = {
        "source_amplitudes_pressure_rate": sa0,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": [0, 0, 10, 10],
        "boundary": ["rigid", "pml", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def loss(c11: torch.Tensor, backend: object) -> torch.Tensor:
        out = deepwave.elastic_vti(
            c11, c13_0, c33_0, c55_0, buoyancy0, dx, dt, backend=backend, **kw
        )
        return out.receiver_amplitudes_p.sum()

    c11_e = c11_0.clone().requires_grad_(True)
    g_e = torch.autograd.grad(loss(c11_e, "eager"), c11_e)[0]
    idx = tuple(int(i) for i in (g_e.abs() == g_e.abs().max()).nonzero()[0])

    c11_c = c11_0.clone().requires_grad_(True)
    g_c = torch.autograd.grad(loss(c11_c, "compiled"), c11_c)[0]

    eps = float(c11_0[idx].item()) * 1e-6
    c11_plus = c11_0.clone()
    c11_plus[idx] += eps
    c11_minus = c11_0.clone()
    c11_minus[idx] -= eps
    fd = (loss(c11_plus, "compiled") - loss(c11_minus, "compiled")) / (2 * eps)

    rel = abs(g_c[idx].item() - fd.item()) / abs(fd.item())
    assert rel < 2e-3, f"compiled grad vs FD rel err {rel} at {idx}"


@_ELASTIC_VTI_C_BACKEND
@pytest.mark.parametrize(
    ("boundary", "pml_width"),
    [
        (["rigid", "pml", "pml", "pml"], [0, 0, 10, 10]),
        (["rigid", "pml", "rigid", "pml"], [0, 0, 0, 0]),
    ],
    ids=["rigid_lo_and_bare_hi", "rigid_lo_both_axes"],
)
def test_elastic_vti_compiled_forward_final_state_matches_eager(
    boundary: list[str], pml_width: list[int]
) -> None:
    """Compiled forward's returned final wavefields are boundary-consistent.

    Regression test for a secondary bug: the compiled forward returned the
    post-update, unfilled final stress/velocity state (the last timestep's
    field update runs after the last in-loop boundary fill), while eager
    applies a final guard-ring/on-surface fill after its time loop
    (``_elastic_vti_eager_core``). Without the fix, e.g. the free-surface
    stress nodes were nonzero in the compiled output where eager (and
    physics) requires them to be exactly zero.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 400
    c11, c13, c33, c55, buoyancy = _isotropic_vti_media((ny, nx))
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw: dict[str, object] = {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": pml_width,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 25.0,
    }
    out_c = deepwave.elastic_vti(
        c11, c13, c33, c55, buoyancy, dx, dt, backend="compiled", **kw
    )
    out_e = deepwave.elastic_vti(
        c11, c13, c33, c55, buoyancy, dx, dt, backend="eager", **kw
    )
    for field in ("vy_0", "vx_0", "sigmayy_0", "sigmaxy_0", "sigmaxx_0"):
        field_c = getattr(out_c, field)
        field_e = getattr(out_e, field)
        denom = field_e.abs().max()
        if denom == 0:
            assert field_c.abs().max() == 0, f"{field}: compiled nonzero, eager zero"
            continue
        rel = (field_c - field_e).abs().max() / denom
        assert rel < 1e-12, f"{field} final-state rel err {rel.item()}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_vti_backward_callback_dict_keys(ndim) -> None:
    """The backward-callback dict's key set matches elastic_wavefield_names exactly.

    ``elastic_vti`` shares ``_engine.run_backward`` (and, on the eager
    backend, ``elastic._elastic_eager_core``'s wavefield bookkeeping) with
    isotropic ``elastic``; previously ``callback_drop_last=1`` dropped the
    last wavefield name from the dict a user's ``backward_callback``
    receives (see PR #121).
    """
    from deepwave.elastic import elastic_wavefield_names

    n = 8
    shape = (n,) * ndim
    base = torch.full(shape, 1e9)
    c33, c11, c55, c13 = (
        (4 * base).requires_grad_(),
        (5 * base).requires_grad_(),
        (1 * base).requires_grad_(),
        (2 * base).requires_grad_(),
    )
    c66 = (1.2 * base).requires_grad_() if ndim == 3 else None
    buoyancy = torch.full(shape, 1 / 2000.0).requires_grad_()
    amp = torch.zeros(1, 1, 8)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_keys: set[str] = set()

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_keys.update(state._wavefields.keys())

    out = deepwave.elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoyancy,
        5.0,
        0.001,
        c66=c66,
        source_amplitudes_x=amp,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_x.sum().backward()
    assert recorded_keys == set(elastic_wavefield_names(ndim))


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    n = 8
    base = torch.full((n, n), 1e9)
    c33, c11, c55, c13 = 4 * base, 5 * base, 1 * base, 2 * base
    buoyancy = torch.full((n, n), 1 / 2000.0)
    amp = torch.zeros(1, 1, 8)
    loc = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_x=amp,
            source_locations_x=loc,
            receiver_locations_x=loc,
            backend="eager",
            storage_compression=True,
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_vti_forward_callback_shear_fields_match_compiled(ndim) -> None:
    """Eager forward-callback wavefield/model dicts match the compiled backend.

    ``elastic_vti``'s eager backend reuses ``elastic._elastic_eager_core``
    (see the module-level import in ``deepwave/elastic_vti.py``), so this
    exercises the same shear/memory-field zip fix as elastic's own
    ``test_elastic_forward_callback_shear_fields_match_compiled`` (see
    ``tests/test_elastic.py``, added by 837b3cb) under elastic_vti's
    different stiffness parametrization, and additionally checks the model
    dict's key set, which neither propagator's forward-callback test
    checked before.
    """
    from deepwave.elastic import elastic_wavefield_names
    from deepwave.elastic_vti import elastic_vti_model_names

    n = 8
    shape = (n,) * ndim
    base = torch.full(shape, 1e9)
    c33, c11, c55, c13 = 4 * base, 5 * base, 1 * base, 2 * base
    c66 = 1.2 * base if ndim == 3 else None
    buoyancy = torch.full(shape, 1 / 2000.0)
    amp = torch.zeros(1, 1, 8)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_wf: dict[str, dict[str, torch.Tensor]] = {}
    recorded_models: dict[str, dict[str, torch.Tensor]] = {}

    def make_recorder(key: str):
        def recorder(state: deepwave.CallbackState) -> None:
            recorded_wf[key] = {
                name: t.clone() for name, t in state._wavefields.items()
            }
            recorded_models[key] = dict(state._models)

        return recorder

    common_kwargs: dict[str, object] = {
        "c66": c66,
        "source_amplitudes_x": amp,
        "source_locations_x": loc,
        "pml_width": 3,
        "callback_frequency": 1,
    }
    deepwave.elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoyancy,
        5.0,
        0.001,
        backend="eager",
        forward_callback=make_recorder("eager"),
        **common_kwargs,
    )
    deepwave.elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoyancy,
        5.0,
        0.001,
        backend="compiled",
        forward_callback=make_recorder("compiled"),
        **common_kwargs,
    )
    eager_wf, compiled_wf = recorded_wf["eager"], recorded_wf["compiled"]
    eager_models, compiled_models = (
        recorded_models["eager"],
        recorded_models["compiled"],
    )

    assert set(eager_wf) == set(compiled_wf) == set(elastic_wavefield_names(ndim))
    assert (
        set(eager_models) == set(compiled_models) == set(elastic_vti_model_names(ndim))
    )
    for name, tensor in eager_wf.items():
        assert tensor.numel() > 0, f"{name}: eager tensor is empty"
    for name in eager_wf:
        e, c = eager_wf[name], compiled_wf[name]
        assert e.shape == c.shape, f"{name}: shape mismatch {e.shape} vs {c.shape}"
        denom = max(c.abs().max().item(), 1e-30)
        relerr = (e - c).abs().max().item() / denom
        assert relerr < 1e-3, f"{name}: eager/compiled mismatch, rel err {relerr:.3e}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_vti_backward_callback_model_and_gradient_dict_keys(ndim) -> None:
    """The backward-callback model/gradient dicts' keys match elastic_vti_model_names.

    ``CallbackState._models``/``_gradients`` are built by
    ``deepwave._engine.run_backward_core`` from ``SPEC.model_names(ndim)``,
    the same mechanism 837b3cb fixed for the wavefield dict (see
    ``test_elastic_vti_backward_callback_dict_keys`` above); the
    model/gradient dicts' key sets were never separately asserted.
    """
    from deepwave.elastic_vti import elastic_vti_model_names

    n = 8
    shape = (n,) * ndim
    base = torch.full(shape, 1e9)
    c33, c11, c55, c13 = (
        (4 * base).requires_grad_(),
        (5 * base).requires_grad_(),
        (1 * base).requires_grad_(),
        (2 * base).requires_grad_(),
    )
    c66 = (1.2 * base).requires_grad_() if ndim == 3 else None
    buoyancy = torch.full(shape, 1 / 2000.0).requires_grad_()
    amp = torch.zeros(1, 1, 8)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_models: dict[str, torch.Tensor] = {}
    recorded_grads: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_models.update(state._models)
        recorded_grads.update(state._gradients)

    out = deepwave.elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoyancy,
        5.0,
        0.001,
        c66=c66,
        source_amplitudes_x=amp,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_x.sum().backward()
    expected = set(elastic_vti_model_names(ndim))
    assert set(recorded_models) == expected
    assert set(recorded_grads) == expected
    for name, t in recorded_models.items():
        assert t.numel() > 0, f"{name}: zero numel in model dict"
    for name, t in recorded_grads.items():
        assert t.numel() > 0, f"{name}: zero numel in gradient dict"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_vti_output_names(ndim) -> None:
    """PropagatorOutput field names match elastic_output_names exactly."""
    from deepwave.elastic import elastic_output_names

    n = 8
    shape = (n,) * ndim
    base = torch.full(shape, 1e9)
    c33, c11, c55, c13 = 4 * base, 5 * base, 1 * base, 2 * base
    c66 = 1.2 * base if ndim == 3 else None
    buoyancy = torch.full(shape, 1 / 2000.0)
    amp = torch.zeros(1, 1, 8)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    out = deepwave.elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoyancy,
        5.0,
        0.001,
        c66=c66,
        source_amplitudes_x=amp,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
    )
    assert out._names == tuple(elastic_output_names(ndim))

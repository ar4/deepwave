"""Tests for the opt-in multiaxial-PML (M-PML) elastic stability option.

M-PML (``mpml_fraction > 0``) adds a fraction ``p`` of each axis's PML damping to the
other axes (Meza-Fajardo & Papageorgiou, 2008), curing the late-time C-PML
instability that afflicts high-Poisson elastic media with facing free surfaces.
It is opt-in and elastic-only; ``mpml_fraction = 0`` (the default) is byte-identical to
the standard CFS-C-PML.

These tests cover the validation gates:

1. ``mpml_fraction = 0`` is bit-identical to the standard propagator (forward and
   gradients, C and eager backends).
2. The archived instability reproduction: unstable at ``p = 0``, stable at
   ``p = 0.15`` (marked slow).
3. Interior accuracy is unchanged at ``p = 0.2`` before boundary reflections
   arrive.
4. The absorption cost of ``p = 0.2`` is modest.
5. Cross-backend gradient parity holds at ``p = 0.15`` (the source-amplitude
   half of that gate is the ``mpml0.15`` row of
   ``tests/test_adjoint_dot_product.py``, which checks the exact adjoint
   identity on both backends).
6. The C and eager backends agree at ``p > 0``.
"""

import math

import pytest
import torch

from deepwave import elastic, vpvsrho_to_lambmubuoyancy
from deepwave.wavelets import ricker

torch.manual_seed(0)


def _homogeneous(ny, nx, vp, vs, rho, dtype=torch.float64, device="cpu"):
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(
        torch.tensor(vp), torch.tensor(vs), torch.tensor(rho)
    )
    lamb = torch.full((ny, nx), float(lamb), dtype=dtype, device=device)
    mu = torch.full((ny, nx), float(mu), dtype=dtype, device=device)
    buoyancy = torch.full((ny, nx), float(buoyancy), dtype=dtype, device=device)
    return lamb, mu, buoyancy


# ---------------------------------------------------------------------------
# Gate 1: p = 0 is bit-identical to the standard propagator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_p0_bit_identical(backend):
    """mpml_fraction=0 reproduces the standard propagator bit-for-bit (fwd + grad).

    Compares an explicit ``mpml_fraction=0`` call against a call that omits the
    argument entirely; both the forward outputs and all model gradients must be
    exactly equal (the p=0 path is the untouched standard propagator).
    """
    ny, nx = 26, 32
    dtype = torch.float64
    lamb0, mu0, buoyancy0 = _homogeneous(ny, nx, 2000.0, 1200.0, 2200.0, dtype)
    nt = 40
    src = (ricker(15.0, nt, 0.001, 1.0 / 15.0).to(dtype) * 1e6).reshape(1, 1, nt)
    sl = torch.tensor([[[13, 16]]])
    rl = torch.tensor([[[8, 8]]])

    def run(kwargs):
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_y=src,
            source_locations_y=sl,
            receiver_locations_y=rl,
            pml_width=8,
            pml_freq=15.0,
            nt=nt,
            backend=backend,
            **kwargs,
        )
        loss = out.receiver_amplitudes_y.pow(2).sum() + out.vy_0.pow(2).sum()
        grads = torch.autograd.grad(loss, [lamb, mu, buoyancy])
        return out.receiver_amplitudes_y.detach(), out.vy_0.detach(), grads

    r_def, w_def, g_def = run({})
    r_p0, w_p0, g_p0 = run({"mpml_fraction": 0.0})
    assert torch.equal(r_def, r_p0)
    assert torch.equal(w_def, w_p0)
    for a, b in zip(g_def, g_p0):
        assert torch.equal(a, b)


def test_p0_matches_default_call():
    """Passing mpml_fraction=0 equals omitting it (default), byte-identical."""
    ny, nx = 24, 28
    dtype = torch.float64
    lamb, mu, buoyancy = _homogeneous(ny, nx, 2000.0, 1100.0, 2200.0, dtype)
    nt = 35
    src = (ricker(15.0, nt, 0.001, 1.0 / 15.0).to(dtype) * 1e6).reshape(1, 1, nt)
    sl = torch.tensor([[[12, 14]]])
    rl = torch.tensor([[[6, 6]]])
    common = {
        "source_amplitudes_y": src,
        "source_locations_y": sl,
        "receiver_locations_y": rl,
        "pml_width": 7,
        "pml_freq": 15.0,
        "nt": nt,
    }
    default = elastic(lamb, mu, buoyancy, 5.0, 0.001, **common)
    explicit = elastic(lamb, mu, buoyancy, 5.0, 0.001, mpml_fraction=0.0, **common)
    assert torch.equal(default.receiver_amplitudes_y, explicit.receiver_amplitudes_y)
    assert torch.equal(default.vy_0, explicit.vy_0)


# ---------------------------------------------------------------------------
# Gate 6: C and eager backends agree at p > 0
# ---------------------------------------------------------------------------


def test_c_eager_parity_mpml():
    """The C and eager backends produce matching output for mpml_fraction > 0."""
    ny, nx = 30, 34
    dtype = torch.float64
    lamb, mu, buoyancy = _homogeneous(ny, nx, 2000.0, 300.0, 2200.0, dtype)
    nt = 60
    src = (ricker(15.0, nt, 0.001, 1.0 / 15.0).to(dtype) * 1e6).reshape(1, 1, nt)
    sl = torch.tensor([[[15, 17]]])
    rl = torch.tensor([[[10, 10]], [[20, 24]]]).reshape(1, 2, 2)

    def run(backend):
        out = elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.0008,
            source_amplitudes_y=src,
            source_locations_y=sl,
            receiver_locations_y=rl,
            pml_width=10,
            pml_freq=15.0,
            nt=nt,
            mpml_fraction=0.2,
            backend=backend,
        )
        return out.receiver_amplitudes_y, out.vy_0, out.vx_0

    rc, vyc, vxc = run("compiled")
    re, vye, vxe = run("eager")
    assert torch.allclose(rc, re, atol=1e-10, rtol=1e-6)
    assert torch.allclose(vyc, vye, atol=1e-10, rtol=1e-6)
    assert torch.allclose(vxc, vxe, atol=1e-10, rtol=1e-6)


# ---------------------------------------------------------------------------
# Gate 3: interior accuracy unchanged before reflections arrive
# ---------------------------------------------------------------------------


def test_interior_accuracy_unchanged():
    """p=0.2 leaves the interior field unchanged before boundaries matter.

    The cross-damping lives only inside the PML strips, so a wave that has not
    yet reached them must be identical to the p=0 case.
    """
    npml = 20
    ny = nx = 120 + 2 * npml
    dtype = torch.float64
    lamb, mu, buoyancy = _homogeneous(ny, nx, 2000.0, 1200.0, 2200.0, dtype)
    nt = 200  # direct wave still mid-interior; reflections not yet returned
    src = (ricker(15.0, nt, 0.0008, 1.0 / 15.0).to(dtype) * 1e6).reshape(1, 1, nt)
    cy = cx = ny // 2
    sl = torch.tensor([[[cy, cx]]])

    def run(p):
        out = elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.0008,
            source_amplitudes_y=src,
            source_locations_y=sl,
            pml_width=npml,
            pml_freq=15.0,
            nt=nt,
            mpml_fraction=p,
        )
        return out.vy_0[0, npml : npml + 120, npml : npml + 120].clone()

    v0 = run(0.0)
    v2 = run(0.2)
    rel = (v0 - v2).norm() / v0.norm()
    assert rel < 1e-6, f"interior differs by {rel:.2e}"


# ---------------------------------------------------------------------------
# Gate 4: absorption cost of p=0.2 is modest
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_absorption_cost_modest():
    """M-PML's non-matched reflection at p=0.2 stays modest (not a sponge)."""
    ny_int, nx_int = 120, 70
    npml = 20
    ny = ny_int + 2 * npml
    nx = nx_int + 2 * npml
    dtype = torch.float64
    lamb, mu, buoyancy = _homogeneous(ny, nx, 2000.0, 1200.0, 2200.0, dtype)
    seg, nseg = 50, 40  # 2000 steps
    src = (ricker(15.0, seg, 0.0008, 1.0 / 15.0).to(dtype) * 1e6).reshape(1, 1, seg)
    zero = torch.zeros_like(src)
    # Source near a corner to stress grazing incidence (the M-PML cost).
    sl = torch.tensor([[[npml + 8, npml + 8]]])

    def residual_over_peak(p):
        state = None
        energies = []
        for c in range(nseg):
            out = elastic(
                lamb,
                mu,
                buoyancy,
                5.0,
                0.0008,
                source_amplitudes_y=(src if c == 0 else zero),
                source_locations_y=sl,
                pml_width=npml,
                pml_freq=15.0,
                nt=seg,
                mpml_fraction=p,
                initial_state=state,
            )
            state = out
            vy = out.vy_0[0, npml : npml + ny_int, npml : npml + nx_int]
            vx = out.vx_0[0, npml : npml + ny_int, npml : npml + nx_int]
            energies.append(float((vy**2 + vx**2).sum()))
        e = torch.tensor(energies)
        late = e[int(0.75 * nseg) :].mean()
        return float(late / e.max())

    r0 = residual_over_peak(0.0)
    r2 = residual_over_peak(0.2)
    # The key claim: the M-PML absorption cost relative to the standard PML is
    # modest (measured ~1.002x for this configuration, i.e. negligible).
    assert r2 < 2 * r0, f"p=0.2 {r2:.2e} vs p=0 {r0:.2e}"
    # Absolute sanity: the boundary still absorbs well (the late window keeps
    # some decaying reverberation, so this is a loose cap, not a PML spec).
    assert r2 < 2e-2, f"p=0.2 residual/peak too large: {r2:.2e}"


# ---------------------------------------------------------------------------
# Gate 2: the instability reproduction
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_instability_repro():
    """Reproduce the C-PML blow-up: unstable at p=0, stable at p=0.15.

    Thin high-Poisson slab between two facing vacuum free surfaces with x-only
    PML -- the archived cheap reproduction of the late-time elastic C-PML
    instability. At p=0 the energy grows exponentially; M-PML at p=0.15 keeps it
    bounded.
    """
    ny, nx, n_vac = 12, 70, 2
    nt_seg, n_chunks = 1000, 18  # 18000 steps
    dt, dx = 9e-4, 4.0
    dtype = torch.float64
    vp, vs, rho = 2000.0, 300.0, 2200.0
    mu_val = rho * vs * vs
    lamb_val = rho * (vp * vp - 2 * vs * vs)

    lamb = torch.full((ny, nx), lamb_val, dtype=dtype)
    mu = torch.full((ny, nx), mu_val, dtype=dtype)
    buoyancy = torch.full((ny, nx), 1.0 / rho, dtype=dtype)
    for r in list(range(n_vac)) + list(range(ny - n_vac, ny)):
        lamb[r, :] = 0.0
        mu[r, :] = 0.0
        buoyancy[r, :] = 0.0
    top_solid = n_vac
    pml_width = [0, 0, 30, 30]
    src = ricker(15.0, nt_seg, dt, 1.0 / 15.0).to(dtype).reshape(1, 1, nt_seg)
    zero = torch.zeros_like(src)
    sl = torch.tensor([[[top_solid, nx // 2]]])

    def final_energy(p):
        state = None
        peak = 0.0
        for c in range(n_chunks):
            out = elastic(
                lamb,
                mu,
                buoyancy,
                dx,
                dt,
                source_amplitudes_y=(src if c == 0 else zero),
                source_locations_y=sl,
                accuracy=4,
                pml_width=pml_width,
                pml_freq=15.0,
                nt=nt_seg,
                mpml_fraction=p,
                initial_state=state,
            )
            state = out
            energy = float((out.vy_0**2 + out.vx_0**2).sum())
            peak = max(peak, energy)
            if not math.isfinite(energy) or energy > 1e12:
                return energy, peak
        return energy, peak

    e0, _peak0 = final_energy(0.0)
    e15, _peak15 = final_energy(0.15)

    # p=0: unmistakable exponential blow-up.
    assert (not math.isfinite(e0)) or e0 > 1e6, (
        f"expected C-PML blow-up at p=0, got final energy {e0:.3e}"
    )
    # p=0.15: bounded (the late energy has decayed well below the peak).
    assert math.isfinite(e15)
    assert e15 < 1e-3, f"p=0.15 did not remain bounded: final energy {e15:.3e}"


# ---------------------------------------------------------------------------
# Gate 5: cross-backend gradient parity at p = 0.15
# ---------------------------------------------------------------------------


def _tiny_problem(dtype=torch.float64):
    ny, nx = 12, 14
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1600.0, 900.0, 2000.0, dtype)
    nt = 8
    src = (ricker(20.0, nt, 0.0006, 1.0 / 20.0).to(dtype) * 1e3).reshape(1, 1, nt)
    sl = torch.tensor([[[6, 7]]])
    rl = torch.tensor([[[4, 4]]])
    return lamb, mu, buoyancy, src, sl, rl, nt


def test_backend_parity_mpml_c_vs_eager():
    """The C and eager backends give matching gradients at p=0.15."""
    lamb, mu, buoyancy, src, sl, rl, nt = _tiny_problem()

    def grads(backend):
        lamb_ = lamb.clone().requires_grad_(True)
        mu_ = mu.clone().requires_grad_(True)
        buoyancy_ = buoyancy.clone().requires_grad_(True)
        out = elastic(
            lamb_,
            mu_,
            buoyancy_,
            6.0,
            0.0006,
            source_amplitudes_y=src,
            source_locations_y=sl,
            receiver_locations_y=rl,
            pml_width=3,
            pml_freq=20.0,
            nt=nt,
            mpml_fraction=0.15,
            backend=backend,
        )
        loss = out.receiver_amplitudes_y.pow(2).sum()
        return torch.autograd.grad(loss, [lamb_, mu_, buoyancy_])

    gc = grads("compiled")
    ge = grads("eager")
    for a, b in zip(gc, ge):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-5)


# ---------------------------------------------------------------------------
# Validation and symbol presence
# ---------------------------------------------------------------------------


def test_mpml_p_validation():
    """mpml_fraction outside [0, 1) is rejected with a clear error."""
    lamb, mu, buoyancy, src, sl, _rl, nt = _tiny_problem()
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="mpml_fraction"):
            elastic(
                lamb,
                mu,
                buoyancy,
                6.0,
                0.0006,
                source_amplitudes_y=src,
                source_locations_y=sl,
                pml_width=3,
                pml_freq=20.0,
                nt=nt,
                mpml_fraction=bad,
            )


def test_mpml_requires_2d():
    """mpml_fraction > 0 is rejected for 1D propagation."""
    nx = 20
    dtype = torch.float64
    lamb = torch.full((nx,), 4e9, dtype=dtype)
    mu = torch.full((nx,), 1e9, dtype=dtype)
    buoyancy = torch.full((nx,), 1 / 2000, dtype=dtype)
    nt = 8
    src = (ricker(20.0, nt, 0.0006, 1 / 20.0).to(dtype)).reshape(1, 1, nt)
    sl = torch.tensor([[[10]]])
    with pytest.raises(ValueError, match="requires at least two spatial"):
        elastic(
            lamb,
            mu,
            buoyancy,
            6.0,
            0.0006,
            source_amplitudes_x=src,
            source_locations_x=sl,
            pml_width=3,
            pml_freq=20.0,
            nt=nt,
            mpml_fraction=0.2,
        )


def test_mpml_symbols_present():
    """The M-PML backend symbols exist for 2D and 3D (all accuracies/dtypes)."""
    from deepwave import _backend_utils as backend_utils

    for ndim in (2, 3):
        for acc in (2, 4, 6, 8):
            for dtype in ("float", "double"):
                for direction in ("forward", "backward"):
                    name = f"elastic_iso_{ndim}d_{acc}_{dtype}_{direction}_mpml_cpu"
                    assert getattr(backend_utils.dll, name, None) is not None, name
    # And are absent for 1D (M-PML has no meaning there).
    assert (
        getattr(backend_utils.dll, "elastic_iso_1d_4_double_forward_mpml_cpu", None)
        is None
    )

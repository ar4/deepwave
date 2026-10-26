"""Physics-correctness tests for the elastic Born update rules.

These tests validate the core claim of ``docs/elastic_born_derivation.md``
S2 directly: that :func:`deepwave.elastic_born.update_velocities_born` /
:func:`deepwave.elastic_born.update_stresses_born`, run for N substeps
starting from a zero scattered field, produce exactly the directional
derivative (to O(eps^2)) of running the existing, tested nonlinear
``update_velocities`` / ``update_stresses`` with the models
perturbed by +-eps*scatter -- i.e. Born == FD of the nonlinear per-step
update -- plus Born linearity in the scattering models. No PML (pml_width=0,
trivial profiles) is used so the test isolates the coupling algebra from the
CPML memory-variable machinery (which is architecturally identical between
the background and scattered passes, per the derivation).
"""

import importlib
import re

import pytest
import torch
from helpers import element_tensors, output_tensors

import deepwave.wavelets

# ``deepwave/__init__.py`` does ``from deepwave.elastic import ... elastic``,
# which rebinds the package attribute ``deepwave.elastic`` to the *function*
# ``elastic``, shadowing the submodule (see the identical note in
# ``elastic_born.py``). Import the needed names directly from the submodule.
from deepwave.elastic import (
    elastic_wavefield_names,
    update_stresses,
    update_velocities,
)

# ``deepwave/__init__.py`` also does ``from deepwave.elastic_born import
# elastic_born``, which rebinds the package attribute ``deepwave.elastic_born``
# to the *function*, shadowing the submodule. Fetch the actual submodule via
# ``importlib`` so ``_eb`` is the module (needed both for its helpers and for
# monkeypatching its globals).
_eb = importlib.import_module("deepwave.elastic_born")


def _sc_boundary_tensors(out):
    """The scattered absorbing-boundary memory variables of an output.

    They are carried inside ``auxiliary_state`` under their internal names,
    which are the background ones with an ``_sc`` suffix.
    """
    state = out.auxiliary_state
    return [state[name] for name in state.names if name.endswith("_sc")]


def _split_captured_func_args(args):
    """Split args captured from a spy on ``elastic_born_func`` into core layout.

    Returns ``(leading8, boundary, mpml, tensor_args)``: the 8 shared scalar
    leading args (``pml_profiles, grid_spacing, dt, nt, step_ratio, accuracy,
    pml_width, n_shots``), the per-face ``boundary`` ``bc`` code list (NOT
    part of ``leading8`` -- ``elastic_born_compiled_core``/
    ``elastic_born_eager_core``/``_elastic_born_backward_*_compiled`` all
    expect it spliced in between ``pml_width`` and ``n_shots``, i.e. between
    ``leading8[6]`` and ``leading8[7]`` -- see each call site below), the
    ``mpml`` bool (the second-to-last leading arg of ``elastic_born_func``),
    and the differentiable tensor tail. This drops the compiled-only
    callback/storage leading params ``elastic_born_func`` also accepts
    (indices 8..13).
    Located via ``_N_LEAD`` so it survives any further leading-count change.
    """
    n_lead = _eb._N_LEAD
    return (
        tuple(args[:8]),
        args[n_lead - 1],
        args[n_lead - 2],
        list(args[n_lead:]),
    )


def _make_state(shape: list[int], device: torch.device, dtype: torch.dtype, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*s: int) -> torch.Tensor:
        return torch.randn(*s, generator=gen).to(device=device, dtype=dtype)

    ndim = len(shape)
    full_shape = [1, *shape]  # batch dim of 1

    velocities = [rnd(*full_shape) * 0.01 for _ in range(ndim)]
    normal_stresses = [rnd(*full_shape) * 0.01 for _ in range(ndim)]
    shear_stresses = [[torch.empty(0) for _ in range(ndim)] for _ in range(ndim)]
    for d1 in range(ndim):
        for d2 in range(d1 + 1, ndim):
            s = rnd(*full_shape) * 0.01
            shear_stresses[d1][d2] = s
            shear_stresses[d2][d1] = s

    s_mem_vars_normal = [
        torch.zeros(*full_shape, device=device, dtype=dtype) for _ in range(ndim)
    ]
    s_mem_vars_shear = [
        [torch.zeros(*full_shape, device=device, dtype=dtype) for _ in range(ndim)]
        for _ in range(ndim)
    ]
    v_mem_vars_normal = [
        torch.zeros(*full_shape, device=device, dtype=dtype) for _ in range(ndim)
    ]
    v_mem_vars_shear = [
        [torch.zeros(*full_shape, device=device, dtype=dtype) for _ in range(ndim)]
        for _ in range(ndim)
    ]

    return (
        velocities,
        normal_stresses,
        shear_stresses,
        s_mem_vars_normal,
        s_mem_vars_shear,
        v_mem_vars_normal,
        v_mem_vars_shear,
    )


def _clone_nested(x):
    if isinstance(x, (list, tuple)):
        return [_clone_nested(v) for v in x]
    return x.clone()


def _make_models(
    ndim: int, shape: list[int], device, dtype, seed: int, scale: float = 1.0
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    full_shape = [1, *shape]

    def pos(*s):
        return (1.0 + 0.1 * torch.rand(*s, generator=gen)).to(
            device=device, dtype=dtype
        ) * scale

    lamb = pos(*full_shape)
    mu = pos(*full_shape)
    models = [lamb, mu]
    if ndim == 3:
        models += [pos(*full_shape) for _ in range(3)]  # mu_zy, mu_zx, mu_yx
    elif ndim == 2:
        models += [pos(*full_shape)]  # mu_yx
    models += [pos(*full_shape) for _ in range(ndim)]  # buoyancy_z/y/x
    return models


def _make_scatter(
    ndim: int, shape: list[int], device, dtype, seed: int, amp: float = 1e-3
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    full_shape = [1, *shape]

    def small(*s):
        return (torch.randn(*s, generator=gen)).to(device=device, dtype=dtype) * amp

    n_shear = {1: 0, 2: 1, 3: 3}[ndim]
    dlamb = small(*full_shape)
    dmu = small(*full_shape)
    dshear = [small(*full_shape) for _ in range(n_shear)]
    dbuoy = [small(*full_shape) for _ in range(ndim)]
    return [dlamb, dmu, *dshear, *dbuoy]


def _run_nonlinear(
    ndim, models, grid_spacing, dt, accuracy, pml_profiles, nsteps, state
):
    (
        velocities,
        normal_stresses,
        shear_stresses,
        s_mem_vars_normal,
        s_mem_vars_shear,
        v_mem_vars_normal,
        v_mem_vars_shear,
    ) = state
    for _ in range(nsteps):
        velocities, s_mem_vars_normal, s_mem_vars_shear = update_velocities(
            models,
            normal_stresses,
            shear_stresses,
            velocities,
            s_mem_vars_normal,
            s_mem_vars_shear,
            pml_profiles,
            grid_spacing,
            dt,
            accuracy,
        )
        normal_stresses, shear_stresses, v_mem_vars_normal, v_mem_vars_shear = (
            update_stresses(
                models,
                velocities,
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )
        )
    return velocities, normal_stresses, shear_stresses


def _run_born(
    ndim,
    bg_models,
    sc_models,
    grid_spacing,
    dt,
    accuracy,
    pml_profiles,
    nsteps,
    bg_state,
    sc_state,
):
    (
        velocities,
        normal_stresses,
        shear_stresses,
        s_mem_vars_normal,
        s_mem_vars_shear,
        v_mem_vars_normal,
        v_mem_vars_shear,
    ) = bg_state
    (
        velocities_sc,
        normal_stresses_sc,
        shear_stresses_sc,
        s_mem_vars_normal_sc,
        s_mem_vars_shear_sc,
        v_mem_vars_normal_sc,
        v_mem_vars_shear_sc,
    ) = sc_state

    for _ in range(nsteps):
        (
            velocities,
            velocities_sc,
            s_mem_vars_normal,
            s_mem_vars_shear,
            s_mem_vars_normal_sc,
            s_mem_vars_shear_sc,
        ) = _eb.update_velocities_born(
            bg_models,
            sc_models,
            normal_stresses,
            shear_stresses,
            normal_stresses_sc,
            shear_stresses_sc,
            velocities,
            velocities_sc,
            s_mem_vars_normal,
            s_mem_vars_shear,
            s_mem_vars_normal_sc,
            s_mem_vars_shear_sc,
            pml_profiles,
            grid_spacing,
            dt,
            accuracy,
        )
        (
            normal_stresses,
            shear_stresses,
            normal_stresses_sc,
            shear_stresses_sc,
            v_mem_vars_normal,
            v_mem_vars_shear,
            v_mem_vars_normal_sc,
            v_mem_vars_shear_sc,
        ) = _eb.update_stresses_born(
            bg_models,
            sc_models,
            velocities,
            velocities_sc,
            normal_stresses,
            shear_stresses,
            normal_stresses_sc,
            shear_stresses_sc,
            v_mem_vars_normal,
            v_mem_vars_shear,
            v_mem_vars_normal_sc,
            v_mem_vars_shear_sc,
            pml_profiles,
            grid_spacing,
            dt,
            accuracy,
        )
    return velocities_sc, normal_stresses_sc, shear_stresses_sc


def _zero_pml_profiles(ndim, shape, device, dtype):
    # a=1, b=0 (and ah=1, bh=0) everywhere reproduces the no-PML recursion
    # exactly (memory variable stays zero, derivative passes through
    # unmodified): mem = 1*mem + 0*d = mem (stays 0); d_corrected = d + mem = d.
    full_shape = [1, *shape]
    profiles = []
    for _ in range(ndim):
        a = torch.ones(*full_shape, device=device, dtype=dtype)
        b = torch.zeros(*full_shape, device=device, dtype=dtype)
        ah = torch.ones(*full_shape, device=device, dtype=dtype)
        bh = torch.zeros(*full_shape, device=device, dtype=dtype)
        profiles += [a, b, ah, bh]
    return profiles


ACCURACY = 4
GRID_SHAPE = {1: [24], 2: [20, 20], 3: [14, 14, 14]}
NSTEPS = 5
DT = 1e-3
DX = 1.0


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_born_matches_fd_of_nonlinear(ndim):
    """Born scattered output == directional FD of the nonlinear per-step update."""
    device = torch.device("cpu")
    dtype = torch.float64
    shape = GRID_SHAPE[ndim]
    grid_spacing = [DX] * ndim
    pml_profiles = _zero_pml_profiles(ndim, shape, device, dtype)

    bg_models = _make_models(ndim, shape, device, dtype, seed=1)
    scatter = _make_scatter(ndim, shape, device, dtype, seed=2, amp=1e-4)

    # scatter amplitude already small; a scale check for linearity is done
    # separately (see test_born_linearity).

    state0 = _make_state(shape, device, dtype, seed=3)

    # Nonlinear FD: models +- eps*scatter, central difference.
    models_plus = [m + s for m, s in zip(bg_models, scatter)]
    models_minus = [m - s for m, s in zip(bg_models, scatter)]

    v_plus, s_plus, sh_plus = _run_nonlinear(
        ndim,
        models_plus,
        grid_spacing,
        DT,
        ACCURACY,
        pml_profiles,
        NSTEPS,
        _clone_nested(state0),
    )
    v_minus, s_minus, sh_minus = _run_nonlinear(
        ndim,
        models_minus,
        grid_spacing,
        DT,
        ACCURACY,
        pml_profiles,
        NSTEPS,
        _clone_nested(state0),
    )
    fd_v = [(vp - vm) / 2 for vp, vm in zip(v_plus, v_minus)]
    fd_s = [(sp - sm) / 2 for sp, sm in zip(s_plus, s_minus)]
    fd_sh = [
        [
            (sh_plus[i][j] - sh_minus[i][j]) / 2
            if sh_plus[i][j].numel() > 0
            else torch.empty(0)
            for j in range(ndim)
        ]
        for i in range(ndim)
    ]

    # Born: background state0 unperturbed, scattered state starts at zero.
    sc_state0 = (
        [torch.zeros_like(v) for v in state0[0]],
        [torch.zeros_like(s) for s in state0[1]],
        [
            [torch.zeros_like(t) if t.numel() > 0 else torch.empty(0) for t in row]
            for row in state0[2]
        ],
        [torch.zeros_like(m) for m in state0[3]],
        [[torch.zeros_like(t) for t in row] for row in state0[4]],
        [torch.zeros_like(m) for m in state0[5]],
        [[torch.zeros_like(t) for t in row] for row in state0[6]],
    )

    v_sc, s_sc, sh_sc = _run_born(
        ndim,
        bg_models,
        scatter,
        grid_spacing,
        DT,
        ACCURACY,
        pml_profiles,
        NSTEPS,
        _clone_nested(state0),
        sc_state0,
    )

    for vb, vfd in zip(v_sc, fd_v):
        torch.testing.assert_close(vb, vfd, atol=1e-9, rtol=1e-6)
    for sb, sfd in zip(s_sc, fd_s):
        torch.testing.assert_close(sb, sfd, atol=1e-9, rtol=1e-6)
    for i in range(ndim):
        for j in range(ndim):
            if sh_sc[i][j].numel() > 0:
                torch.testing.assert_close(
                    sh_sc[i][j], fd_sh[i][j], atol=1e-9, rtol=1e-6
                )


@pytest.mark.parametrize("ndim", [2, 3])
def test_born_linearity(ndim):
    """Scaling the scatter by alpha scales the scattered output by alpha."""
    device = torch.device("cpu")
    dtype = torch.float64
    shape = GRID_SHAPE[ndim]
    grid_spacing = [DX] * ndim
    pml_profiles = _zero_pml_profiles(ndim, shape, device, dtype)

    bg_models = _make_models(ndim, shape, device, dtype, seed=10)
    scatter = _make_scatter(ndim, shape, device, dtype, seed=11, amp=1e-3)
    state0 = _make_state(shape, device, dtype, seed=12)

    def zero_sc_state(state0):
        return (
            [torch.zeros_like(v) for v in state0[0]],
            [torch.zeros_like(s) for s in state0[1]],
            [
                [torch.zeros_like(t) if t.numel() > 0 else torch.empty(0) for t in row]
                for row in state0[2]
            ],
            [torch.zeros_like(m) for m in state0[3]],
            [[torch.zeros_like(t) for t in row] for row in state0[4]],
            [torch.zeros_like(m) for m in state0[5]],
            [[torch.zeros_like(t) for t in row] for row in state0[6]],
        )

    alpha = 3.7
    scatter_scaled = [alpha * s for s in scatter]

    v1, s1, sh1 = _run_born(
        ndim,
        bg_models,
        scatter,
        grid_spacing,
        DT,
        ACCURACY,
        pml_profiles,
        NSTEPS,
        _clone_nested(state0),
        zero_sc_state(state0),
    )
    v2, s2, sh2 = _run_born(
        ndim,
        bg_models,
        scatter_scaled,
        grid_spacing,
        DT,
        ACCURACY,
        pml_profiles,
        NSTEPS,
        _clone_nested(state0),
        zero_sc_state(state0),
    )

    for a, b in zip(v1, v2):
        torch.testing.assert_close(b, alpha * a, atol=1e-10, rtol=1e-8)
    for a, b in zip(s1, s2):
        torch.testing.assert_close(b, alpha * a, atol=1e-10, rtol=1e-8)
    for i in range(ndim):
        for j in range(ndim):
            if sh1[i][j].numel() > 0:
                torch.testing.assert_close(
                    sh2[i][j], alpha * sh1[i][j], atol=1e-10, rtol=1e-8
                )


@pytest.mark.parametrize("ndim", [2, 3])
def test_zero_scatter_gives_zero_scattered_output(ndim):
    """Zero scattering models give an identically zero scattered field."""
    device = torch.device("cpu")
    dtype = torch.float64
    shape = GRID_SHAPE[ndim]
    grid_spacing = [DX] * ndim
    pml_profiles = _zero_pml_profiles(ndim, shape, device, dtype)

    bg_models = _make_models(ndim, shape, device, dtype, seed=20)
    n_models = len(bg_models)
    scatter_zero = [torch.zeros_like(m) for m in bg_models][:n_models]
    state0 = _make_state(shape, device, dtype, seed=21)
    sc_state0 = (
        [torch.zeros_like(v) for v in state0[0]],
        [torch.zeros_like(s) for s in state0[1]],
        [
            [torch.zeros_like(t) if t.numel() > 0 else torch.empty(0) for t in row]
            for row in state0[2]
        ],
        [torch.zeros_like(m) for m in state0[3]],
        [[torch.zeros_like(t) for t in row] for row in state0[4]],
        [torch.zeros_like(m) for m in state0[5]],
        [[torch.zeros_like(t) for t in row] for row in state0[6]],
    )

    v_sc, s_sc, sh_sc = _run_born(
        ndim,
        bg_models,
        scatter_zero,
        grid_spacing,
        DT,
        ACCURACY,
        pml_profiles,
        NSTEPS,
        _clone_nested(state0),
        sc_state0,
    )
    for v in v_sc:
        assert torch.all(v == 0)
    for s in s_sc:
        assert torch.all(s == 0)
    for i in range(ndim):
        for j in range(ndim):
            if sh_sc[i][j].numel() > 0:
                assert torch.all(sh_sc[i][j] == 0)


def test_prepare_scatter_parameters_matches_jvp_of_prepare_parameters():
    """prepare_scatter_parameters is exactly torch.func.jvp of prepare_parameters."""
    from deepwave.elastic import prepare_parameters as _pp

    dtype = torch.float64
    shape = [1, 16, 16]  # leading batch/shot dim: prepare_parameters infers
    # ndim = mu.ndim - 1, so a bare (16, 16) tensor would be treated as 1-D.
    gen = torch.Generator().manual_seed(42)
    mu = (1.0 + 0.1 * torch.rand(*shape, generator=gen)).to(dtype)
    buoyancy = (1.0 + 0.1 * torch.rand(*shape, generator=gen)).to(dtype)
    dmu = torch.randn(*shape, generator=gen).to(dtype) * 1e-3
    dbuoyancy = torch.randn(*shape, generator=gen).to(dtype) * 1e-3

    result = _eb.prepare_scatter_parameters(mu, buoyancy, dmu, dbuoyancy)

    _, expected = torch.func.jvp(_pp, (mu, buoyancy), (dmu, dbuoyancy))
    for r, e in zip(result, expected):
        torch.testing.assert_close(r, e)


def test_prepare_scatter_parameters_vacuum_masking():
    """A scatterer entirely inside a vacuum region produces zero prepared scatter.

    (IVM free-surface interaction, derivation SS7.1.)
    """
    dtype = torch.float64
    shape = [1, 16, 16]
    mu = torch.ones(*shape, dtype=dtype)
    buoyancy = torch.ones(*shape, dtype=dtype)
    # Make a vacuum block (mu=buoyancy=0) in the corner.
    mu[:, :4, :4] = 0.0
    buoyancy[:, :4, :4] = 0.0

    dmu = torch.zeros(*shape, dtype=dtype)
    dbuoyancy = torch.zeros(*shape, dtype=dtype)
    dmu[:, :3, :3] = 0.05  # perturbation strictly inside the vacuum block
    dbuoyancy[:, :3, :3] = 0.05

    dmu_yx, dbuoyancy_y, dbuoyancy_x = _eb.prepare_scatter_parameters(
        mu, buoyancy, dmu, dbuoyancy
    )
    # All shear/velocity nodes whose stencil lies within the vacuum block
    # must have zero scattering image (vacuum stays vacuum under the mask).
    assert torch.all(dmu_yx[:, :3, :3] == 0)
    assert torch.all(dbuoyancy_y[:, :3, :3] == 0)
    assert torch.all(dbuoyancy_x[:, :3, :3] == 0)


# ---------------------------------------------------------------------------
# Public eager `elastic_born(...)` driver + API gates (derivation S10).
# ---------------------------------------------------------------------------


def _born_setup(ndim, seed=3):
    """Small float64 background + interior scatter for the public-API gates."""
    from deepwave import vpvsrho_to_lambmubuoyancy

    dtype = torch.float64
    dx, dt, pml = 5.0, 4e-4, 3
    nt = 12
    g = torch.Generator().manual_seed(seed)
    if ndim == 2:
        shape = (13, 15)
        sloc = torch.tensor([[[6, 7]]])
        rloc = torch.tensor([[[3, 4], [9, 10]]])
    else:
        shape = (15, 15, 15)
        sloc = torch.tensor([[[7, 7, 7]]])
        rloc = torch.tensor([[[3, 4, 5], [10, 11, 12]]])
    vp = 2000 + 50 * torch.rand(shape, generator=g, dtype=dtype)
    vs = 1100 + 30 * torch.rand(shape, generator=g, dtype=dtype)
    rho = 2200 + 40 * torch.rand(shape, generator=g, dtype=dtype)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    src = torch.randn(1, 1, nt, generator=g, dtype=dtype)
    mask = torch.zeros(shape, dtype=dtype)
    m = pml + 2
    mask[tuple(slice(m, s - m) for s in shape)] = 1.0
    dlamb = (
        torch.randn(shape, generator=g, dtype=dtype) * lamb.abs().mean() * 0.1 * mask
    )
    dmu = torch.randn(shape, generator=g, dtype=dtype) * mu.abs().mean() * 0.1 * mask
    dbuoyancy = (
        torch.randn(shape, generator=g, dtype=dtype)
        * buoyancy.abs().mean()
        * 0.1
        * mask
    )
    comp = "y" if ndim == 2 else "z"
    common = {
        "pml_width": pml,
        "backend": "eager",
        "pml_freq": 15.0,
        "max_vel": 2100.0,
    }
    sk = {f"source_amplitudes_{comp}": src, f"source_locations_{comp}": sloc}
    # The Born propagator spells its scattered receivers ``sc_``; the
    # plain elastic propagator this is compared against keeps the bare
    # spelling, so the two need separate keyword sets.
    rk = {f"sc_receiver_locations_{comp}": rloc, "sc_receiver_locations_p": rloc}
    rk_plain = {f"receiver_locations_{comp}": rloc, "receiver_locations_p": rloc}
    return {
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "dlamb": dlamb,
        "dmu": dmu,
        "dbuoyancy": dbuoyancy,
        "dx": dx,
        "dt": dt,
        "rloc": rloc,
        "sk": sk,
        "rk": rk,
        "rk_plain": rk_plain,
        "common": common,
        "comp": comp,
    }


@pytest.mark.parametrize("ndim", [2, 3])
def test_public_born_matches_fd_of_elastic(ndim):
    """Scattered receivers equal the directional FD of nonlinear elastic."""
    from deepwave.elastic import elastic

    s = _born_setup(ndim)
    n_src = ndim + 1
    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["rk"],
        **s["sk"],
        **s["common"],
    )
    sc_p = out[-n_src]
    sc_v = out[-n_src + 1]
    eps = 1e-4

    def el(sign):
        o = elastic(
            s["lamb"] + sign * eps * s["dlamb"],
            s["mu"] + sign * eps * s["dmu"],
            s["buoyancy"] + sign * eps * s["dbuoyancy"],
            s["dx"],
            s["dt"],
            **s["rk_plain"],
            **s["sk"],
            **s["common"],
        )
        return o[-n_src], o[-n_src + 1]

    pp, vp_ = el(+1)
    pm, vm_ = el(-1)
    fd_p = (pp - pm) / (2 * eps)
    fd_v = (vp_ - vm_) / (2 * eps)
    assert (sc_p - fd_p).norm() / fd_p.norm() < 1e-6
    assert (sc_v - fd_v).norm() / fd_v.norm() < 1e-6


@pytest.mark.parametrize("ndim", [2, 3])
def test_public_born_linearity_and_zero_scatter(ndim):
    """Scattered output is linear; zero scatter -> zero scattered + bg==elastic."""
    from deepwave.elastic import elastic

    s = _born_setup(ndim)
    n_src = ndim + 1
    bg_rk = {
        f"bg_receiver_locations_{s['comp']}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }

    def run(scale):
        return _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=scale * s["dlamb"],
            dmu=scale * s["dmu"],
            dbuoyancy=scale * s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **bg_rk,
            **s["common"],
        )

    sc1 = run(1.0)[-n_src]
    sc2 = run(2.7)[-n_src]
    assert (sc2 - 2.7 * sc1).norm() / (2.7 * sc1).norm() < 1e-10

    out0 = run(0.0)
    for i in range(n_src):
        sc = out0[-n_src + i]
        if sc.numel() > 0:
            assert sc.abs().max() == 0
    o = elastic(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        **s["rk_plain"],
        **s["sk"],
        **s["common"],
    )
    for i in range(n_src):
        assert torch.allclose(out0[-2 * n_src + i], o[-n_src + i], atol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_public_jvp_matches_born(ndim):
    """torch.func.jvp of elastic equals the elastic_born scattered output."""
    from deepwave.elastic import elastic

    s = _born_setup(ndim)
    n_src = ndim + 1
    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["rk"],
        **s["sk"],
        **s["common"],
    )
    sc_p = out[-n_src]

    def f(lamb, mu, buoyancy):
        o = elastic(
            lamb,
            mu,
            buoyancy,
            s["dx"],
            s["dt"],
            **s["rk_plain"],
            **s["sk"],
            **s["common"],
        )
        return o[-n_src]

    _, tangent = torch.func.jvp(
        f,
        (s["lamb"], s["mu"], s["buoyancy"]),
        (s["dlamb"], s["dmu"], s["dbuoyancy"]),
    )
    assert (tangent - sc_p).norm() / sc_p.norm() < 1e-8


@pytest.mark.parametrize("ndim", [2, 3])
def test_jvp_compiled_backend_matches_born(ndim):
    """torch.func.jvp of the *compiled* elastic backend equals Born (S6/gate 3).

    ``test_public_jvp_matches_born`` exercises the eager backend (native
    forward-mode AD). This gate covers the fast (C) backend, whose
    ``ElasticForwardFunc.jvp`` routes the tangents through the Born relationship
    (``docs/elastic_born_derivation.md`` S6): the returned tangent must equal the
    ``elastic_born`` scattered output, and the primal must equal the compiled
    forward. The tangent now comes from the *compiled* Born forward core (the
    same kernel the ``elastic_born`` reference uses), not the eager bridge, so
    the two agree well inside the ~1e-6 fast-math band the derivation anticipated
    (here to near machine precision, sharing the kernel).
    """
    from deepwave.elastic import elastic

    s = _born_setup(ndim)
    n_src = ndim + 1
    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["rk"],
        **s["sk"],
        **s["common"],
    )
    sc_p = out[-n_src]
    sc_v = out[-n_src + 1]

    common_c = dict(s["common"])
    common_c["backend"] = "compiled"  # compiled C backend

    def f(lamb, mu, buoyancy):
        o = elastic(
            lamb, mu, buoyancy, s["dx"], s["dt"], **s["rk_plain"], **s["sk"], **common_c
        )
        return o[-n_src], o[-n_src + 1]

    (prim_p, _), (tan_p, tan_v) = torch.func.jvp(
        f,
        (s["lamb"], s["mu"], s["buoyancy"]),
        (s["dlamb"], s["dmu"], s["dbuoyancy"]),
    )
    assert (tan_p - sc_p).norm() / sc_p.norm() < 1e-8
    assert (tan_v - sc_v).norm() / sc_v.norm() < 1e-8
    # The primal is just the compiled elastic forward.
    o_c = elastic(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        **s["rk_plain"],
        **s["sk"],
        **common_c,
    )
    assert torch.allclose(prim_p, o_c[-n_src])


def test_jvp_compiled_backend_guards():
    """The compiled-backend jvp warns on a PML-band tangent and rejects nesting.

    The warning is about a model tangent that the PML profiles depend on: a
    perturbation inside the absorbing layer changes ``max_vel``, and hence the
    profiles, which the forward-mode derivative does not carry. Pinning
    ``max_vel`` removes that dependence, so the warning is expected only when
    ``max_vel`` is left to be inferred, and pinning it is the documented way
    to silence it. Both halves are checked here.
    """
    import warnings

    from deepwave.elastic import elastic

    s = _born_setup(2)
    n_src = 3
    common_c = dict(s["common"])
    common_c["backend"] = "compiled"
    common_inferred = {k: v for k, v in common_c.items() if k != "max_vel"}

    def make_f(common):
        def f(lamb, mu, buoyancy):
            o = elastic(
                lamb,
                mu,
                buoyancy,
                s["dx"],
                s["dt"],
                **s["rk_plain"],
                **s["sk"],
                **common,
            )
            return o[-n_src]

        return f

    f = make_f(common_c)

    # A model tangent that is non-zero inside the PML/FD halo must warn when
    # max_vel is inferred from the (perturbed) model.
    dlamb_pml = torch.zeros_like(s["lamb"])
    dlamb_pml[0, 0] = 1.0
    tangents = (
        dlamb_pml,
        torch.zeros_like(s["mu"]),
        torch.zeros_like(s["buoyancy"]),
    )
    primals = (s["lamb"], s["mu"], s["buoyancy"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.func.jvp(make_f(common_inferred), primals, tangents)
    assert any("PML" in str(x.message) for x in w), [str(x.message) for x in w]

    # ... and pinning max_vel silences it, because the profiles no longer
    # depend on the perturbed model.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.func.jvp(f, primals, tangents)
    assert not any("PML" in str(x.message) for x in w), [str(x.message) for x in w]

    # Nested/composed transforms are not supported on the fast backend.
    def g(lamb):
        def h(mu):
            o = elastic(
                lamb,
                mu,
                s["buoyancy"],
                s["dx"],
                s["dt"],
                **s["rk_plain"],
                **s["sk"],
                **common_c,
            )
            return o[-n_src]

        _, t = torch.func.jvp(h, (s["mu"],), (s["dmu"],))
        return t

    with pytest.raises(NotImplementedError):
        torch.func.jvp(g, (s["lamb"],), (s["dlamb"],))


@pytest.mark.slow
def test_public_born_gradcheck():
    """gradcheck/gradgradcheck of elastic_born (2D) w.r.t. bg and scatter models."""
    from deepwave import vpvsrho_to_lambmubuoyancy

    dtype = torch.float64
    shape = (10, 11)
    g = torch.Generator().manual_seed(5)
    vp = 2000 + 30 * torch.rand(shape, generator=g, dtype=dtype)
    vs = 1100 + 20 * torch.rand(shape, generator=g, dtype=dtype)
    rho = 2200 + 30 * torch.rand(shape, generator=g, dtype=dtype)
    lamb, mu, bu = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    src = torch.randn(1, 1, 6, generator=g, dtype=dtype) * 1e3
    sloc = torch.tensor([[[5, 6]]])
    rloc = torch.tensor([[[3, 4], [7, 8]]])
    common = {
        "pml_width": 2,
        "backend": "eager",
        "pml_freq": 15.0,
        "max_vel": 2200.0,
    }
    n_src = 3

    def f(lamb, mu, bu, dl, dm, db):
        out = _eb.elastic_born(
            lamb,
            mu,
            bu,
            5.0,
            4e-4,
            dlamb=dl,
            dmu=dm,
            dbuoyancy=db,
            source_amplitudes_y=src,
            source_locations_y=sloc,
            sc_receiver_locations_y=rloc,
            sc_receiver_locations_p=rloc,
            **common,
        )
        return out[-n_src], out[-n_src + 1]

    args = (
        lamb.clone().requires_grad_(True),
        mu.clone().requires_grad_(True),
        bu.clone().requires_grad_(True),
        (0.05 * lamb * torch.rand(shape, generator=g, dtype=dtype)).requires_grad_(
            True
        ),
        (0.05 * mu * torch.rand(shape, generator=g, dtype=dtype)).requires_grad_(True),
        (0.05 * bu * torch.rand(shape, generator=g, dtype=dtype)).requires_grad_(True),
    )
    assert torch.autograd.gradcheck(f, args, atol=1e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(f, args, atol=1e-4, rtol=1e-2)


# ---------------------------------------------------------------------------
# Compiled (C) forward backend == eager driver (phase-4 forward-only gate).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "other_backend",
    [
        pytest.param("compiled", id="compiled"),
        pytest.param("torch_compile", id="torch_compile"),
    ],
)
def test_compiled_forward_matches_eager(ndim, other_backend):
    """Compiled C Born forward == eager Born driver to near machine precision.

    Covers the background AND scattered final wavefields and both receiver
    blocks (background + scattered), the phase-4 forward-only gate of
    ``docs/elastic_born_derivation.md``. ``other_backend`` additionally covers
    the ``torch_compile`` Python mode, which runs the same eager time
    loop with :func:`deepwave.elastic_born.update_velocities_born`/
    :func:`deepwave.elastic_born.update_stresses_born`
    ``torch.compile``d, so it is held to the same tolerance as the
    compiled C/CUDA backend.
    """
    s = _born_setup(ndim)
    bg_rk = {
        f"bg_receiver_locations_{s['comp']}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }
    # Also drive a pressure source (exercises the bg pressure injection and the
    # scattered-pressure-absent path -- the elastic pressure asymmetry).
    g = torch.Generator().manual_seed(99)
    src_p = torch.randn(1, 1, 12, generator=g, dtype=torch.float64)
    sloc = s["sk"][f"source_locations_{s['comp']}"]
    pk = {
        "source_amplitudes_pressure_rate": src_p,
        "source_locations_pressure_rate": sloc,
    }

    def run(backend):
        common = dict(s["common"])
        common["backend"] = backend
        return _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **pk,
            **bg_rk,
            **common,
        )

    eager = output_tensors(run("eager"))
    other = output_tensors(run(other_backend))
    assert len(eager) == len(other)
    for i, (e, c) in enumerate(zip(eager, other)):
        assert e.shape == c.shape, f"output {i} shape {e.shape} vs {c.shape}"
        if e.numel() == 0:
            assert c.numel() == 0
            continue
        denom = e.abs().max()
        if denom == 0:
            assert c.abs().max() == 0, f"output {i}: eager zero, other nonzero"
        else:
            rel = (e - c).abs().max() / denom
            assert rel < 1e-10, f"output {i} mismatch: rel={rel}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_compiled_zero_scatter(ndim):
    """Compiled path: zero scatter -> zero scattered output, bg == elastic."""
    from deepwave.elastic import elastic

    s = _born_setup(ndim)
    n_src = ndim + 1
    common = dict(s["common"])
    common["backend"] = "compiled"
    bg_rk = {
        f"bg_receiver_locations_{s['comp']}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }
    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=torch.zeros_like(s["lamb"]),
        dmu=torch.zeros_like(s["mu"]),
        dbuoyancy=torch.zeros_like(s["buoyancy"]),
        **s["rk"],
        **s["sk"],
        **bg_rk,
        **common,
    )
    # All scattered wavefields and receivers are identically zero. The
    # scattered physical fields are the public names carrying the `sc` infix;
    # the scattered absorbing-boundary variables are inside auxiliary_state.
    scattered = [v for k, v in out.as_dict().items() if k.endswith("_sc_0")]
    assert scattered
    for w in [*scattered, *_sc_boundary_tensors(out)]:
        assert torch.all(w == 0)
    for i in range(n_src):
        sc = out[-n_src + i]
        if sc.numel() > 0:
            assert sc.abs().max() == 0
    # Background receivers match the plain compiled elastic forward.
    o = elastic(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        **s["rk_plain"],
        **s["sk"],
        **common,
    )
    for i in range(n_src):
        assert torch.allclose(out[-2 * n_src + i], o[-n_src + i], atol=1e-10)


@pytest.mark.parametrize("ndim", [2])
def test_compiled_backward_matches_eager(ndim):
    """The compiled path's eager-recompute backward matches the eager path.

    The compiled default path keeps gradients via an eager-recompute backward
    (``ElasticBornForwardFunc``); its model gradients must match the pure eager
    backend to near machine precision.
    """
    s = _born_setup(ndim)
    n_src = ndim + 1

    def grads(backend):
        lamb = s["lamb"].clone().requires_grad_(True)
        dlamb = s["dlamb"].clone().requires_grad_(True)
        dmu = s["dmu"].clone().requires_grad_(True)
        common = dict(s["common"])
        common["backend"] = backend
        out = _eb.elastic_born(
            lamb,
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **common,
        )
        loss = (out[-n_src] ** 2).sum()
        loss.backward()
        return lamb.grad, dlamb.grad, dmu.grad

    ge = grads("eager")
    gc = grads("compiled")
    for a, b in zip(ge, gc):
        denom = a.abs().max()
        assert denom > 0
        assert (a - b).abs().max() / denom < 1e-7


def test_second_order_raises():
    """Double-backward through the COMPILED elastic_born raises a clear error.

    The compiled elastic_born backward (``ElasticBornForwardFunc.backward``,
    whether the compiled adjoint kernels or the eager-recompute bridge) has no
    analytic second-order path, so a Hessian-vector product would silently
    return an incorrect result. :func:`deepwave._engine.no_second_order` makes it
    raise instead (mirrors ``scalar_born``/``acoustic_born``). The guard fires on
    the ``create_graph=True`` backward itself, before a second ``.grad`` is
    needed. The eager backend (``backend="eager"``) supports second-order
    natively and is unaffected (see ``test_public_born_gradcheck``).
    """
    s = _born_setup(2)
    n_src = 2 + 1
    lamb = s["lamb"].clone().requires_grad_(True)
    dlamb = s["dlamb"].clone().requires_grad_(True)
    common = dict(s["common"])
    common["backend"] = "compiled"  # compiled path
    out = _eb.elastic_born(
        lamb,
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=dlamb,
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["rk"],
        **s["sk"],
        **common,
    )
    loss = (out[-n_src] ** 2).sum()
    with pytest.raises(RuntimeError, match="Second-order differentiation"):
        torch.autograd.grad(loss, dlamb, create_graph=True)


def test_first_order_unaffected_by_second_order_guard():
    """Plain first-order compiled elastic_born gradients are unchanged by the guard."""
    s = _born_setup(2)
    n_src = 2 + 1

    def grad_dlamb():
        dlamb = s["dlamb"].clone().requires_grad_(True)
        common = dict(s["common"])
        common["backend"] = "compiled"  # compiled path
        out = _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **common,
        )
        return torch.autograd.grad((out[-n_src] ** 2).sum(), dlamb)[0]

    g0 = grad_dlamb()
    g1 = grad_dlamb()
    assert torch.isfinite(g0).all()
    assert g0.abs().sum() > 0
    assert torch.equal(g0, g1)


def _mgsi_grads(ndim, mgsi):
    """Compiled lamb/dlamb/dmu gradients at a given mgsi (M5: step_ratio > 1)."""
    s = _born_setup(ndim)
    n_src = ndim + 1
    lamb = s["lamb"].clone().requires_grad_(True)
    dlamb = s["dlamb"].clone().requires_grad_(True)
    dmu = s["dmu"].clone().requires_grad_(True)
    common = dict(s["common"])
    common["backend"] = "compiled"  # compiled path
    out = _eb.elastic_born(
        lamb,
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=dlamb,
        dmu=dmu,
        dbuoyancy=s["dbuoyancy"],
        **s["rk"],
        **s["sk"],
        model_gradient_sampling_interval=mgsi,
        **common,
    )
    # Seed the scattered receiver block AND background receiver block so both the
    # scattered-model and background-model imaging conditions are exercised.
    loss = (out[-2 * n_src] ** 2).sum() + (out[-n_src] ** 2).sum()
    loss.backward()
    return lamb.grad, dlamb.grad, dmu.grad


@pytest.mark.parametrize("ndim", [2, 3])
def test_compiled_mgsi_gt1_approximates_mgsi1(ndim):
    """model_gradient_sampling_interval > 1 runs end-to-end and approximates mgsi=1.

    The public ``model_gradient_sampling_interval`` is folded into the
    engine's ``step_ratio`` (``step_ratio *= mgsi``), so
    ``mgsi > 1`` genuinely exercises step_ratio > 1 on BOTH sides: the forward
    only writes a gradient store every ``step_ratio`` internal steps
    (``store && (t % step_ratio) == 0``) and the backward reads those strided
    snapshots and scales the accumulated gradient by ``step_ratio``. The model
    gradient is a strided rectangle-rule quadrature of a time integral, exact
    only at stride 1, so ``mgsi=2`` is a deliberate APPROXIMATION of the
    ``mgsi=1`` gradient (not an exact match -- cf.
    ``test_model_gradient_quadrature_error`` for the plain scalar propagator).
    This asserts it is finite, nonzero, points the same way, and stays within a
    loose relative tolerance of the ``mgsi=1`` gradient (i.e. the strided store
    read/write/scale is wired correctly, not returning garbage or zeros).
    """
    lamb1, dlamb1, dmu1 = _mgsi_grads(ndim, 1)
    lamb2, dlamb2, dmu2 = _mgsi_grads(ndim, 2)
    for name, g1, g2 in [
        ("lamb", lamb1, lamb2),
        ("dlamb", dlamb1, dlamb2),
        ("dmu", dmu1, dmu2),
    ]:
        assert torch.isfinite(g2).all(), f"{name}: mgsi=2 gradient not finite"
        denom = g1.norm()
        assert denom > 0, f"{name}: mgsi=1 gradient is zero (test not exercised)"
        # Same direction (subsampling coarsens but does not reorient the imaging
        # condition) and same ballpark magnitude.
        cos = torch.dot(g1.flatten(), g2.flatten()) / (g1.norm() * g2.norm())
        assert cos > 0.85, f"{name}: mgsi=2 vs mgsi=1 cosine={cos.item()}"
        rel = (g1 - g2).norm() / denom
        assert rel < 0.6, f"{name}: mgsi=2 vs mgsi=1 rel L2={rel.item()}"
        # Magnitude band: this is the specific regression guard for the
        # elastic_born.c step_ratio gradient-accumulation bug fixed in
        # commit 2d049fc, which the loose cosine/rel-L2 checks above do NOT
        # reliably catch
        # (both the buggy and the fixed kernel give cos > 0.99 and rel < 0.31
        # here -- the bug does not reorient or grossly distort the gradient,
        # it rescales it). With the correct ``t % step_ratio == 0`` gate on
        # the gradient-accumulation flags AND the matching ``* step_ratio``
        # accumulation factor, this ratio empirically lands in [1.08, 1.43]
        # across {ndim in (2, 3)} x {lamb, dlamb, dmu} (measured on this
        # exact test setup). With the bug (flags ungated, no scale factor --
        # verified by temporarily reverting the fix and rebuilding), the same
        # sweep gives [0.70, 0.74]: a clean, non-overlapping separation. The
        # band below sits with margin on both sides of that boundary.
        ratio = g2.norm() / denom
        assert 0.85 < ratio < 1.6, (
            f"{name}: mgsi=2/mgsi=1 gradient-norm ratio={ratio.item()} outside "
            "[0.85, 1.6]; step_ratio store gate/scale likely wrong"
        )


@pytest.mark.parametrize("storage_mode", ["cpu", "disk"])
@pytest.mark.parametrize("mgsi", [1, 2])
def test_public_storage_modes_match_device(storage_mode, mgsi, tmp_path):
    """M5: storage_mode is threaded through and every mode gives identical results.

    The storage mode only changes WHERE the forward's gradient checkpoints live
    (device RAM / host RAM / disk), never the numbers, so both the forward
    outputs and the backward gradients must match the default ``"device"`` mode
    to machine precision. Exercises the public parameter that M5 exposed
    (previously ``ElasticBornForwardFunc.forward`` hard-coded ``"device"``).
    ``"none"`` is covered separately (it retains no history for gradients).

    The ``mgsi=2`` (``step_ratio=2``) parametrization is a REGRESSION GUARD for
    the elastic_born.c step_ratio gradient-accumulation bug fixed in commit
    2d049fc: the compiled ``backward``/``backward_sc`` gradient-accumulation
    flags were not gated on ``t % step_ratio == 0``, so at ``step_ratio > 1``
    they accumulated
    on EVERY internal step. In ``"device"`` mode the store pointer still tracked
    ``t // step_ratio`` (masquerading as a plausible different quadrature), but
    in ``"cpu"``/``"disk"`` mode the single scratch snapshot buffer is only
    reloaded on the gate steps, so the ungated per-step accumulation correlated
    the adjoint field against a STALE snapshot from the wrong step_ratio group --
    a structurally different (wrong) number. The bug therefore breaks the
    ``cpu/disk == device`` equality specifically at ``mgsi > 1``; with the gate
    (and the matching ``* step_ratio`` scale) restored, every mode reads the same
    snapshot on the same steps and the gradients are bit-identical again.
    """
    ndim = 2
    s = _born_setup(ndim)
    n_src = ndim + 1

    def run(mode):
        lamb = s["lamb"].clone().requires_grad_(True)
        dlamb = s["dlamb"].clone().requires_grad_(True)
        common = dict(s["common"])
        common["backend"] = "compiled"
        kw = {"storage_mode": mode, "model_gradient_sampling_interval": mgsi}
        if mode == "disk":
            kw["storage_path"] = str(tmp_path)
        out = _eb.elastic_born(
            lamb,
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **kw,
            **common,
        )
        fwd = [o.detach().clone() for o in output_tensors(out)]
        (out[-n_src] ** 2).sum().backward()
        return fwd, lamb.grad, dlamb.grad

    fwd_ref, lamb_ref, dlamb_ref = run("device")
    fwd, lamb_g, dlamb_g = run(storage_mode)
    for i, (a, b) in enumerate(zip(fwd_ref, fwd)):
        assert torch.equal(a, b), f"output {i} differs for storage_mode={storage_mode}"
    assert (lamb_ref - lamb_g).abs().max() == 0
    assert (dlamb_ref - dlamb_g).abs().max() == 0


def test_public_storage_mode_none_forward_only_and_rejects_grad():
    """M5: storage_mode="none" is a forward-only low-memory mode.

    With ``"none"`` the forward retains no wavefield history, so its outputs
    still match the default ``"device"`` forward exactly (storage never affects
    the forward), but the model-gradient imaging correlation cannot be formed.
    elastic_born therefore raises a clear error when ``"none"`` is combined with
    a differentiable input, rather than silently returning a wrong (zero) model
    gradient or segfaulting on an unallocated store read (the analogous
    acoustic_born bug is the regression
    ``tests/test_acoustic_born.py::test_storage_mode_none_forward_only_and_rejects_grad``
    guards against).
    """
    ndim = 2
    s = _born_setup(ndim)
    common = dict(s["common"])
    common["backend"] = "compiled"

    def call(mode, grad):
        lamb = s["lamb"].clone().requires_grad_(grad)
        return _eb.elastic_born(
            lamb,
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            storage_mode=mode,
            **common,
        )

    # Forward-only (no grad): "none" runs and matches the device forward.
    out_dev = call("device", grad=False)
    out_none = call("none", grad=False)
    for i, (a, b) in enumerate(zip(output_tensors(out_dev), output_tensors(out_none))):
        assert torch.equal(a, b), f"output {i} differs for storage_mode=none"

    # With a differentiable input, "none" raises rather than crashing.
    with pytest.raises(RuntimeError, match=r"storage_mode.*none.*no forward"):
        call("none", grad=True)


def test_public_callbacks_fire_and_preserve_gradient():
    """M5: forward/backward callbacks fire and do not change the result.

    ``callback_frequency`` chunks the forward and backward time loops; the
    callbacks are invoked once per chunk. Exercising them must not perturb the
    outputs or gradients versus a single-chunk (no-callback) run. This is the
    public-API counterpart of the internal multi-chunk equivalence tests.
    """
    ndim = 2
    s = _born_setup(ndim)
    n_src = ndim + 1

    def run(with_callbacks):
        lamb = s["lamb"].clone().requires_grad_(True)
        dlamb = s["dlamb"].clone().requires_grad_(True)
        common = dict(s["common"])
        common["backend"] = "compiled"
        counts = {"fwd": 0, "bwd": 0}
        kw = {}
        if with_callbacks:
            kw = {
                "forward_callback": lambda _st: counts.__setitem__(
                    "fwd", counts["fwd"] + 1
                ),
                "backward_callback": lambda _st: counts.__setitem__(
                    "bwd", counts["bwd"] + 1
                ),
                "callback_frequency": 3,
            }
        out = _eb.elastic_born(
            lamb,
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **kw,
            **common,
        )
        fwd = [o.detach().clone() for o in output_tensors(out)]
        (out[-n_src] ** 2).sum().backward()
        return fwd, dlamb.grad, counts

    fwd_ref, grad_ref, _ = run(False)
    fwd_cb, grad_cb, counts = run(True)
    assert counts["fwd"] > 0, "forward_callback never fired"
    assert counts["bwd"] > 0, "backward_callback never fired"
    for i, (a, b) in enumerate(zip(fwd_ref, fwd_cb)):
        assert torch.equal(a, b), f"output {i} perturbed by callbacks"
    assert torch.equal(grad_ref, grad_cb)


def test_module_callback_smoke() -> None:
    """A5: ElasticBorn Module wrapper threads the callback kwargs through.

    ``ElasticBorn.forward`` accepts ``forward_callback``/``backward_callback``/
    ``callback_frequency`` (mirroring the functional ``elastic_born``, see
    ``test_public_callbacks_fire_and_preserve_gradient`` above). Constructs
    the Module directly (rather than calling the functional) and confirms a
    forward/backward callback pair actually fires the expected number of
    times, ``ceil(nt / callback_frequency)``.
    """
    import deepwave

    ndim = 2
    s = _born_setup(ndim)
    nt = 12  # matches _born_setup's fixed nt
    callback_frequency = 3

    module = deepwave.ElasticBorn(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        requires_grad=["lamb", "dlamb"],
    )

    counts = {"fwd": 0, "bwd": 0}

    def fwd_cb(_state: object) -> None:
        counts["fwd"] += 1

    def bwd_cb(_state: object) -> None:
        counts["bwd"] += 1

    common = dict(s["common"])
    common["backend"] = "compiled"  # callbacks require the compiled backend
    out = module(
        s["dt"],
        **s["rk"],
        **s["sk"],
        forward_callback=fwd_cb,
        backward_callback=bwd_cb,
        callback_frequency=callback_frequency,
        **common,
    )
    expected_count = (nt + callback_frequency - 1) // callback_frequency
    assert counts["fwd"] == expected_count, "forward_callback never fired as expected"

    names = _eb.elastic_born_output_names(ndim)
    d = dict(zip(names, out))
    (d["sc_receiver_amplitudes_p"] ** 2).sum().backward()
    assert counts["bwd"] == expected_count, "backward_callback never fired as expected"
    assert module.dlamb is not None
    assert module.dlamb.grad is not None
    assert module.dlamb.grad.abs().max() > 0


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("pml", [0, 3])
def test_compiled_backward_full_matches_eager(ndim, pml):
    """Full two-field compiled ``backward`` == eager autograd (SS4.2/SS4.3).

    Exercises the complete two-field adjoint: background AND scattered adjoint
    fields, the SS4.2 coupling, and the SS4.3 two-term background gradients. A
    scalar loss is seeded on EVERY forward output (both final wavefields and both
    receiver blocks) so both adjoint fields are directly seeded, and gradients
    are requested for every differentiable input class: background models
    (``lamb``/``mu``/``buoyancy``), scattering models (``dlamb``/``dmu``/
    ``dbuoyancy``), and the background source amplitudes. ``pml=0`` isolates the
    PLAIN adjoint sites; ``pml=3`` adds the ``_PML`` sites and the memory-variable
    recursions. Compiled gradients must match the eager driver's autograd to near
    float64 machine precision.
    """
    s = _born_setup(ndim)
    comp = s["comp"]
    bg_rk = {
        f"bg_receiver_locations_{comp}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }

    def grads(backend):
        lamb = s["lamb"].clone().requires_grad_(True)
        mu = s["mu"].clone().requires_grad_(True)
        buoyancy = s["buoyancy"].clone().requires_grad_(True)
        dlamb = s["dlamb"].clone().requires_grad_(True)
        dmu = s["dmu"].clone().requires_grad_(True)
        dbuoyancy = s["dbuoyancy"].clone().requires_grad_(True)
        src = s["sk"][f"source_amplitudes_{comp}"].clone().requires_grad_(True)
        common = dict(s["common"])
        common["backend"] = backend
        common["pml_width"] = pml
        out = _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            **{
                f"source_amplitudes_{comp}": src,
                f"source_locations_{comp}": s["sk"][f"source_locations_{comp}"],
            },
            **s["rk"],
            **bg_rk,
            **common,
        )
        loss = sum((o**2).sum() for o in output_tensors(out) if o.numel() > 0)
        loss.backward()
        return [
            lamb.grad,
            mu.grad,
            buoyancy.grad,
            dlamb.grad,
            dmu.grad,
            dbuoyancy.grad,
            src.grad,
        ]

    ge = grads("eager")
    gc = grads("compiled")
    names = ["lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy", "src"]
    for name, a, b in zip(names, ge, gc):
        assert a is not None, name
        assert b is not None, name
        denom = a.abs().max()
        assert denom > 0, name
        rel = (a - b).abs().max() / denom
        assert rel < 1e-6, f"{name}: rel={rel}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_compiled_backward_sc_dispatch(ndim):
    """Scattered-only public gradients route through the compiled backward_sc.

    When only the scattering models (``dlamb``/``dmu``/``dbuoyancy``) need a
    gradient -- the LSRTM/imaging case -- the compiled default backend
    (``ElasticBornForwardFunc``) must dispatch to the compiled fixed-background
    adjoint (``ElasticBornForwardFunc._backward_sc_dispatch``, which since M2
    drives the ``backward_sc`` kernel via ``run_backward_core`` and
    ``ctx.storage_manager`` directly, no forward re-run), not the eager-recompute
    bridge, and the resulting gradients must still match the pure-eager backend
    to the established tolerance. Also checks the negative case: if a background
    model needs a gradient, the compiled backward_sc must *not* be used.
    """
    s = _born_setup(ndim)
    n_src = ndim + 1

    def grads(backend, bg_grad):
        lamb = s["lamb"].clone().requires_grad_(bg_grad)
        dlamb = s["dlamb"].clone().requires_grad_(True)
        dmu = s["dmu"].clone().requires_grad_(True)
        dbuoyancy = s["dbuoyancy"].clone().requires_grad_(True)
        common = dict(s["common"])
        common["backend"] = backend
        # ElasticBornForwardFunc._backward_sc_dispatch no longer calls
        # elastic_born.py's _elastic_born_backward_sc_compiled -- it calls
        # deepwave._engine.run_backward_core directly (reading
        # ctx.storage_manager instead of re-running the forward). Spy on the
        # dispatch staticmethod itself instead -- that's the thing this test
        # wants to confirm fires exactly once.
        real = _eb.ElasticBornForwardFunc._backward_sc_dispatch
        calls = {"n": 0}

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        _eb.ElasticBornForwardFunc._backward_sc_dispatch = spy
        try:
            out = _eb.elastic_born(
                lamb,
                s["mu"],
                s["buoyancy"],
                s["dx"],
                s["dt"],
                dlamb=dlamb,
                dmu=dmu,
                dbuoyancy=dbuoyancy,
                **s["rk"],
                **s["sk"],
                **common,
            )
            # Seed the loss on BOTH the scattered receivers and a scattered final
            # wavefield so the receiver-residual and initial-wavefield adjoint
            # seed paths are both exercised.
            first_sc_wavefield = next(
                v for k, v in out.as_dict().items() if k.endswith("_sc_0")
            )
            loss = (out[-n_src] ** 2).sum() + (first_sc_wavefield**2).sum()
            loss.backward()
        finally:
            _eb.ElasticBornForwardFunc._backward_sc_dispatch = real
        return (dlamb.grad, dmu.grad, dbuoyancy.grad), calls["n"]

    # Scattered-only compiled path: backward_sc must fire exactly once.
    gc, n_calls_compiled = grads("compiled", bg_grad=False)
    assert n_calls_compiled == 1, "compiled scattered-only path did not use backward_sc"

    # Eager backend must never touch the compiled backward_sc.
    ge, n_calls_eager = grads("eager", bg_grad=False)
    assert n_calls_eager == 0, "eager backend used the compiled backward_sc"

    for a, b in zip(ge, gc):
        denom = a.abs().max()
        assert denom > 0
        assert (a - b).abs().max() / denom < 1e-6

    # Negative: a background model needing a gradient must NOT use backward_sc.
    _gc_bg, n_calls_bg = grads("compiled", bg_grad=True)
    assert n_calls_bg == 0, "background-grad path incorrectly used backward_sc"


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("scattered_only", id="scattered_only"),
        pytest.param("full", id="full"),
    ],
)
def test_c_threading_matches_single_thread(ndim, mode):
    """Intra-shot OpenMP threading (B2) does not change the compiled result.

    ``elastic_born.c`` was ported from one-shot-per-thread parallelism to
    intra-shot spatial parallelism (see ``FUNC(forward)``/``FUNC(backward_sc)``/
    ``FUNC(backward)``, mirroring ``elastic.c``/``viscoacoustic.c``/
    ``acoustic_vti.c``/``acoustic_born.c``'s B2 ports): each time step's
    velocity-update loops (vz/vy/vx) and stress-update loops (sigmaii/sigmaxy/
    sigmaxz/sigmayz) are now split across threads via ``#pragma omp for simd
    ... nowait``, with shots processed serially.

    elastic_born is the largest/most complex propagator (a coupled Born
    version of full velocity-stress elastic: 3 velocity + 4 stress field
    groups, 7 groups x 3 entry points = 21 shared kernel functions), so this
    test exercises all three compiled entry points and both gradient
    families across both normal-stress and shear-stress components:

    * ``mode="scattered_only"``: only ``dlamb``/``dmu``/``dbuoyancy`` require
      grad, so the engine dispatches to ``FUNC(backward_sc)`` (the
      scattered-only adjoint, confirmed by ``test_compiled_backward_sc_dispatch``
      above). Every grid cell's ``grad_dlamb``/``grad_dmu``/``grad_dbuoyancy_*``
      accumulation happens inside ``FUNC(backward_sc_sigmaii_cell)``/
      ``sigmaxy_cell``/``sigmaxz_cell``/``sigmayz_cell``/``vz_cell``/``vy_cell``/
      ``vx_cell``, called once per cell by exactly one thread (the PML-region
      tiling partitions the grid disjointly across the ``omp for`` loops), so
      the accumulation order -- and hence the result -- does not depend on
      thread count.
    * ``mode="full"``: ``lamb``/``mu``/``buoyancy`` also require grad, so the
      engine dispatches to ``FUNC(backward)`` (the full two-field adjoint).
      ``FUNC(backward_sigmaii_cell)``/``sigmaxy_cell``/``sigmaxz_cell``/
      ``sigmayz_cell``/``vz_cell``/``vy_cell``/``vx_cell`` each accumulate BOTH
      the background gradient (``grad_lamb``/``grad_mu``/``grad_buoyancy_*``)
      AND the scattered gradient (``grad_dlamb``/``grad_dmu``/
      ``grad_dbuoyancy_*``) for the same cell in the same call, by the same
      thread that owns that cell in this ``omp for`` region -- so there is no
      race between the background and scattered accumulations either.

    Also covers ``FUNC(forward)`` (both modes run it; it only writes per-cell
    stores, never accumulates, so it is trivially race-free).

    Uses ``torch.set_num_threads`` (read by the backend via
    ``torch.get_num_threads()`` at call time, exactly as ``OMP_NUM_THREADS``
    would set the default) rather than an env var, so the comparison runs
    in-process; the tolerance mirrors ``test_compiled_backward_full_matches_eager``/
    ``test_compiled_backward_sc_dispatch`` above (the existing cross-backend
    parity patterns in this file), not a new one.
    """
    s = _born_setup(ndim)
    comp = s["comp"]
    bg_rk = {
        f"bg_receiver_locations_{comp}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }

    def run(n_threads):
        saved = torch.get_num_threads()
        torch.set_num_threads(n_threads)
        try:
            lamb = s["lamb"].clone().requires_grad_(mode == "full")
            mu = s["mu"].clone().requires_grad_(mode == "full")
            buoyancy = s["buoyancy"].clone().requires_grad_(mode == "full")
            dlamb = s["dlamb"].clone().requires_grad_(True)
            dmu = s["dmu"].clone().requires_grad_(True)
            dbuoyancy = s["dbuoyancy"].clone().requires_grad_(True)
            common = dict(s["common"])
            common["backend"] = "compiled"
            kwargs = dict(s["rk"])
            if mode == "full":
                kwargs.update(bg_rk)
            out = _eb.elastic_born(
                lamb,
                mu,
                buoyancy,
                s["dx"],
                s["dt"],
                dlamb=dlamb,
                dmu=dmu,
                dbuoyancy=dbuoyancy,
                **kwargs,
                **s["sk"],
                **common,
            )
            loss = sum((o**2).sum() for o in output_tensors(out) if o.numel() > 0)
            loss.backward()
            grads = {"dlamb": dlamb.grad, "dmu": dmu.grad, "dbuoyancy": dbuoyancy.grad}
            if mode == "full":
                grads["lamb"] = lamb.grad
                grads["mu"] = mu.grad
                grads["buoyancy"] = buoyancy.grad
            return out, grads
        finally:
            torch.set_num_threads(saved)

    out_1, g_1 = run(1)
    out_4, g_4 = run(4)
    for a, b in zip(output_tensors(out_1), output_tensors(out_4)):
        if a.numel() == 0:
            continue
        scale = max(a.abs().max().item(), 1.0)
        assert (a - b).abs().max().item() <= scale * 1e-6
    for name in g_1:
        g1, g4 = g_1[name], g_4[name]
        assert g1 is not None, name
        assert g4 is not None, name
        scale = max(g1.abs().max().item(), 1.0)
        assert (g1 - g4).abs().max().item() <= scale * 1e-6, name


# ---------------------------------------------------------------------------
# Forward background gradient-store checkpointing (phase 5 prerequisite).
#
# The compiled forward optionally checkpoints the background finite-difference
# terms the Born adjoint (backward_sc / backward, phase 6) will consume, into a
# single packed ``bg_store`` buffer. These tests validate the two properties
# this session can establish end-to-end:
#   1. store_bg=False is byte-identical to the pre-existing forward-only path
#      (no regression), and store_bg=True does not perturb any forward output;
#   2. the packed store has the correct shape/dtype and is populated (nonzero in
#      the interior) once the background field is nontrivial.
# Store *value* correctness (the exact imaging quantities) is validated by the
# phase-6 backward_sc gradient check that consumes them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
def test_forward_bg_store_does_not_perturb_and_is_populated(ndim):
    """store_bg=True leaves every forward output unchanged and fills bg_store."""
    # ElasticBornForwardFunc.forward no longer calls
    # elastic_born_compiled_core on its default (fast) path -- it calls
    # run_forward_engine directly -- so spying on elastic_born_compiled_core no
    # longer observes anything. elastic_born(...)'s ``core_args`` are still
    # assembled in EXACTLY the layout elastic_born_compiled_core expects and
    # are handed, unchanged, to elastic_born_func (by bare module-level name,
    # so monkeypatching the module attribute still intercepts it); spy there
    # instead. elastic_born_compiled_core itself is unchanged (now a thin
    # StorageManager-stacking bridge, still used by the compiled backward
    # re-run and by elastic.py's jvp/HVP) and is still the right tool to
    # replay the captured args with store_bg=True/False below.
    real_core = _eb.elastic_born_compiled_core
    real_func = _eb.elastic_born_func
    captured = {}

    def spy(*args, **kwargs):
        # Record the exact positional args the public front-end assembled.
        captured["args"] = args
        return real_func(*args, **kwargs)

    s = _born_setup(ndim)
    common = dict(s["common"])
    common["backend"] = "compiled"  # compiled path
    _eb.elastic_born_func = spy
    try:
        _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **common,
        )
    finally:
        _eb.elastic_born_func = real_func

    args = captured["args"]
    lead8, boundary, mpml, t_args = _split_captured_func_args(args)
    core_args = (*lead8[:7], boundary, lead8[7], mpml, *t_args)
    out_no = real_core(*core_args, store_bg=False)
    out_yes = real_core(*core_args, store_bg=True)
    # The store is appended only when store_bg=True.
    assert len(out_yes) == len(out_no) + 1
    store = out_yes[-1]
    for i, (a, b) in enumerate(zip(out_no, out_yes[:-1])):
        assert a.shape == b.shape
        if a.numel() == 0:
            assert b.numel() == 0
            continue
        # Near-bit-identical: the store writes must not touch the forward
        # state's *values*, but exact bitwise equality (``torch.equal``) is not
        # a valid gate once *any* sibling conditional code exists in the same
        # ``-Ofast``-compiled SIMD loop (here, the phase-7 ``store_sc`` writes
        # alongside these ``store_bg`` writes): ``-Ofast``/``-ffast-math``
        # explicitly permit floating-point reassociation, and the compiler's
        # vectorisation/scheduling choices for a loop can shift when unrelated
        # (even always-false-at-runtime) branches are added to its body,
        # changing summation order at the ULP level. That is a property of the
        # compiled code, not a correctness bug: verified separately (see
        # ``test_forward_sc_store_does_not_perturb_and_matches_fd``) that the
        # store values themselves match an independent finite-difference
        # reference to ~1e-10 relative, far tighter than this tolerance. A
        # relative error at the level below therefore cannot be hiding a real
        # aliasing/overflow bug (which would show up at order 1, not 1e-9).
        denom = a.abs().max()
        if denom > 0:
            rel = (a - b).abs().max() / denom
            assert rel < 1e-9, f"output {i} perturbed by store_bg: rel={rel}"
        else:
            assert b.abs().max() == 0, f"output {i} perturbed by store_bg"

    n_bg_store = {2: 5, 3: 9}[ndim]
    assert store.shape[0] == n_bg_store
    assert store.dtype == s["lamb"].dtype
    # The background field is driven by a source, so at least one store slot is
    # populated in the interior once a few steps have elapsed.
    assert store.abs().max() > 0


@pytest.mark.parametrize("ndim", [2, 3])
def test_forward_sc_store_does_not_perturb_and_matches_fd(ndim):
    """store_sc=True leaves the forward unchanged; sc_store == FD of bg_store.

    The scattered stores hold ``dt *`` the *scattered*-field finite-difference
    terms -- the directional derivative, along the prepared scatter, of the
    background stores (derivation SS2.5/SS4.3). Since the compiled Born forward
    computes that directional derivative exactly (Born == analytic d/deps of the
    discrete update), the scattered store must equal the central finite
    difference of the background store when the *physical* background models
    are perturbed by ``+-eps * (physical scatter)`` -- **through the public
    front end**, so that everything derived from the models (including the
    prepared staggered models AND the pre-scaled force source amplitudes, which
    bake in a buoyancy-at-the-source-location factor, per ``elastic_born``'s
    ``source_amplitudes_out[i] = base * gathered(buoyancy) * dt``) is
    consistently re-derived for the perturbed model. Perturbing only the
    kernel-facing prepared model tensors while leaving a buoyancy-baked source
    amplitude frozen is INCONSISTENT and produces a large, spurious mismatch
    localised at the source cell -- caught during this test's development by
    reconstructing the scattered velocity from the stores by hand and finding
    the one-point discrepancy sat exactly at the (flattened) source index; the
    fix is to always perturb by re-invoking the public API. This is a
    first-principles check of the store values, independent of the
    (not-yet-landed) full backward.
    """
    # See the retargeting note in
    # test_forward_bg_store_does_not_perturb_and_is_populated -- spy on
    # elastic_born_func (still called by bare name from elastic_born(...),
    # still handed core_args in elastic_born_compiled_core's exact expected
    # layout) rather than elastic_born_compiled_core, which the default
    # forward path no longer calls.
    real_core = _eb.elastic_born_compiled_core
    real_func = _eb.elastic_born_func
    captured: dict = {}

    def spy(*args, **kwargs):
        captured.setdefault("list", []).append(args)
        return real_func(*args, **kwargs)

    s = _born_setup(ndim)
    common = dict(s["common"])
    common["backend"] = "compiled"

    def run_public(lamb, mu, buoyancy, dlamb, dmu, dbuoyancy):
        _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            **s["rk"],
            **s["sk"],
            **common,
        )
        return captured["list"].pop()

    def bg_store_at(eps):
        a = run_public(
            s["lamb"] + eps * s["dlamb"],
            s["mu"] + eps * s["dmu"],
            s["buoyancy"] + eps * s["dbuoyancy"],
            zero,
            zero,
            zero,
        )
        lead8, boundary, mpml, t_args = _split_captured_func_args(a)
        return real_core(*lead8[:7], boundary, lead8[7], mpml, *t_args, store_bg=True)[
            -1
        ]

    zero = torch.zeros_like(s["dlamb"])
    _eb.elastic_born_func = spy
    try:
        args0 = run_public(
            s["lamb"], s["mu"], s["buoyancy"], s["dlamb"], s["dmu"], s["dbuoyancy"]
        )
        lead8, boundary_bg, mpml_bg, t_args_bg = _split_captured_func_args(args0)
        leading = (*lead8[:7], boundary_bg, lead8[7], mpml_bg)
        tensor_args = t_args_bg

        # Near-bit-identity: store_sc must not perturb any forward output's
        # *value* (see the near-equality rationale in
        # test_forward_bg_store_does_not_perturb_and_is_populated -- the same
        # -Ofast/sibling-branch codegen caveat applies here).
        out_no = real_core(*leading, *tensor_args)
        out_sc = real_core(*leading, *tensor_args, store_sc=True)
        assert len(out_sc) == len(out_no) + 1
        sc_store = out_sc[-1]
        for i, (a, b) in enumerate(zip(out_no, out_sc[:-1])):
            if a.numel() == 0:
                assert b.numel() == 0
                continue
            denom = a.abs().max()
            if denom > 0:
                rel = (a - b).abs().max() / denom
                assert rel < 1e-9, f"output {i} perturbed by store_sc: rel={rel}"
            else:
                assert b.abs().max() == 0, f"output {i} perturbed by store_sc"

        # store_bg and store_sc together: (bg_store, sc_store) both appended.
        # Also near- (not bit-) identical to the store_sc-only sc_store: two
        # separate calls taking an extra sibling branch (store_bg's write) can
        # still differ at the ULP floor for the same -Ofast reason as above
        # (empirically ~1e-16 relative here, i.e. a single ULP -- far below
        # anything a real bug would produce).
        out_both = real_core(*leading, *tensor_args, store_bg=True, store_sc=True)
        assert len(out_both) == len(out_no) + 2
        sc_both = out_both[-1]
        rel_sc = (sc_both - sc_store).abs().max() / sc_store.abs().max()
        assert rel_sc < 1e-9, f"sc_store differs with store_bg also set: {rel_sc}"

        # --- sc_store == central FD of bg_store, perturbing through the
        #     public API (so the buoyancy-baked force-source amplitudes are
        #     consistently re-derived for the perturbed model too) ---
        eps = 1e-4
        fd = (bg_store_at(eps) - bg_store_at(-eps)) / (2 * eps)
    finally:
        _eb.elastic_born_func = real_func

    denom = fd.abs().max()
    assert denom > 0
    rel = (fd - sc_store).abs().max() / denom
    assert rel < 1e-6, f"sc_store vs FD(bg_store) rel={rel}"


# ---------------------------------------------------------------------------
# Compiled scattered-only adjoint (backward_sc, derivation SS4.1).
#
# The compiled ``elastic_born_iso_*_backward_sc_*`` kernel runs the plain
# elastic adjoint operator on a single scattered adjoint field, seeded by the
# scattered receiver residuals, and correlates it with the forward's packed
# ``bg_store`` to accumulate the scattering-model gradients. This validates the
# hand-written transpose against the eager driver's autograd gradient (the
# scattered-only imaging condition) to near machine precision.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("pml", [0, 3])
def test_compiled_backward_sc_matches_eager_imaging(ndim, pml):
    """backward_sc scatter-model gradients == eager autograd (SS4.1 imaging).

    Drives an explosive (pressure) source in addition to the force source so the
    background develops significant dilatation/strain at the scatterers, giving
    the ``lamb``/``mu`` imaging conditions headroom above the float64 roundoff
    floor (they are otherwise small in absolute terms only because ``lamb`` is
    ~1e10). Seeds the adjoint from BOTH the scattered final wavefields (the
    ``sc_seeds`` initial-adjoint path) and the scattered receiver residuals (the
    ``grad_r_sc`` LSRTM path), covering every scattering model's imaging
    condition and both memory-variable (pml>0) and no-PML regimes.
    """
    from deepwave.elastic import elastic_model_names

    # See the retargeting note in
    # test_forward_bg_store_does_not_perturb_and_is_populated -- spy on
    # elastic_born_func instead of elastic_born_compiled_core, which the
    # default FORWARD path no longer calls (this test's own compiled
    # backward_sc dispatch below, _elastic_born_backward_sc_compiled, is
    # unaffected -- it is called directly, not spied on).
    real_func = _eb.elastic_born_func
    captured = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        return real_func(*args, **kwargs)

    s = _born_setup(ndim)
    g = torch.Generator().manual_seed(1)
    src_p = torch.randn(1, 1, 12, generator=g, dtype=torch.float64) * 1e6
    sloc = s["sk"][f"source_locations_{s['comp']}"]
    pk = {
        "source_amplitudes_pressure_rate": src_p,
        "source_locations_pressure_rate": sloc,
    }
    bg_rk = {
        f"bg_receiver_locations_{s['comp']}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }
    common = dict(s["common"])
    common["backend"] = "compiled"
    common["pml_width"] = pml
    _eb.elastic_born_func = spy
    try:
        _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **pk,
            **bg_rk,
            **common,
        )
    finally:
        _eb.elastic_born_func = real_func

    args = captured["args"]
    (
        (
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_shots,
        ),
        boundary,
        mpml,
        tensor_args,
    ) = _split_captured_func_args(args)
    n_models = len(elastic_model_names(ndim))
    n_wf = len(elastic_wavefield_names(ndim))
    n_src = ndim + 1

    # --- eager reference: autograd gradient wrt the prepared scattering models
    #     of a loss seeded on the scattered final wavefields AND scattered
    #     receivers (= L^T applied to the corresponding residuals) ---
    eager_args = list(tensor_args)
    sc_leaves = []
    for i in range(n_models, 2 * n_models):
        leaf = eager_args[i].detach().clone().requires_grad_(True)
        eager_args[i] = leaf
        sc_leaves.append(leaf)
    outs = _eb.elastic_born_eager_core(
        pml_profiles,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        boundary,
        n_shots,
        mpml,
        *eager_args,
    )
    sc_final = list(outs[n_wf : 2 * n_wf])
    sc_amps = list(outs[2 * n_wf + n_src : 2 * n_wf + 2 * n_src])
    wgen = torch.Generator().manual_seed(7)
    w_wf = [torch.randn(a.shape, generator=wgen, dtype=a.dtype) for a in sc_final]
    w_r = [
        torch.randn(a.shape, generator=wgen, dtype=a.dtype)
        if a.numel() > 0
        else torch.empty(0, dtype=a.dtype)
        for a in sc_amps
    ]
    loss = sum((w * a).sum() for w, a in zip(w_wf, sc_final)) + sum(
        (w * a).sum() for w, a in zip(w_r, sc_amps) if a.numel() > 0
    )
    eager_grads = torch.autograd.grad(loss, sc_leaves)

    # --- compiled backward_sc: seeds are the loss gradients wrt the scattered
    #     final wavefields (sc_seeds) and scattered receivers (grad_r_sc) ---
    grad_sc, _grad_f, _adj = _eb._elastic_born_backward_sc_compiled(
        pml_profiles,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        boundary,
        n_shots,
        mpml,
        tuple(tensor_args),
        w_r,
        sc_seeds=w_wf,
    )

    assert len(grad_sc) == n_models
    max_rel = 0.0
    for name, ge, gc in zip(elastic_model_names(ndim), eager_grads, grad_sc):
        denom = ge.abs().max()
        assert denom > 0, f"{name}: eager gradient is zero (test not exercised)"
        rel = (ge - gc).abs().max() / denom
        max_rel = max(max_rel, float(rel))
        assert rel < 1e-5, f"{name}: rel={rel}"
    assert max_rel > 0  # sanity: the gradients are nontrivial


@pytest.mark.parametrize("ndim", [2, 3])
def test_compiled_backward_sc_multi_chunk_matches_single_chunk(ndim):
    """Chunked backward_sc (small callback_frequency) matches the single call.

    backward_sc's ``m_sigma*`` memory-variable aux buffers now ping-pong
    across chunk boundaries
    (``deepwave.elastic._elastic_swap``), replacing the old per-call
    calloc/free that made every call implicitly a single full-``nt`` chunk. A
    single-chunk run cannot exercise the swap at all (it only fires between
    chunks), so this forces ``deepwave._engine.run_backward_core`` to run with
    ``callback_frequency=3`` (and a no-op ``backward_callback``) via a
    monkeypatch wrapping the existing ``_elastic_born_backward_sc_compiled``
    bridge, and asserts the resulting scattering-model, source-amplitude, and
    initial-wavefield gradients match an ordinary (single-chunk,
    ``callback_frequency=1``) call to machine precision. Getting the
    ``m_sigma*`` swap parity wrong corrupts ONLY PML-region adjoints
    (derivation SS4.2/5.3) -- interior-only checks would pass regardless --
    so this runs at ``pml=3``, where every memory-variable recursion is live.
    """
    import deepwave._common
    from deepwave.elastic import elastic_model_names

    s = _born_setup(ndim)
    real_func = _eb.elastic_born_func
    captured = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        return real_func(*args, **kwargs)

    common = dict(s["common"])
    common["backend"] = "compiled"
    common["pml_width"] = 3
    _eb.elastic_born_func = spy
    try:
        _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["rk"],
            **s["sk"],
            **common,
        )
    finally:
        _eb.elastic_born_func = real_func

    args = captured["args"]
    (
        (
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_shots,
        ),
        boundary,
        mpml,
        tensor_args,
    ) = _split_captured_func_args(args)
    n_wf = len(elastic_wavefield_names(ndim))
    n_src = ndim + 1

    outs = _eb.elastic_born_eager_core(
        pml_profiles,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        boundary,
        n_shots,
        mpml,
        *tensor_args,
    )
    sc_final = list(outs[n_wf : 2 * n_wf])
    sc_amps = list(outs[2 * n_wf + n_src : 2 * n_wf + 2 * n_src])
    wgen = torch.Generator().manual_seed(13)
    sc_seeds = [torch.randn(a.shape, generator=wgen, dtype=a.dtype) for a in sc_final]
    grad_r_sc = [
        torch.randn(a.shape, generator=wgen, dtype=a.dtype)
        if a.numel() > 0
        else torch.empty(0, dtype=a.dtype)
        for a in sc_amps
    ]

    real_run_backward_core = deepwave._engine.run_backward_core
    forced_frequency = {"value": None}

    def patched_run_backward_core(*a, **k):
        if forced_frequency["value"] is not None:
            k = dict(k)
            k["callback_frequency"] = forced_frequency["value"]
            k["backward_callback"] = lambda _state: None
        return real_run_backward_core(*a, **k)

    deepwave._engine.run_backward_core = patched_run_backward_core
    try:
        forced_frequency["value"] = None  # single chunk (the existing default)
        grad_sc_1, grad_f_1, adj_1 = _eb._elastic_born_backward_sc_compiled(
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            boundary,
            n_shots,
            mpml,
            tuple(tensor_args),
            grad_r_sc,
            sc_seeds=sc_seeds,
        )
        forced_frequency["value"] = 3  # multiple chunks: exercises the aux swap
        grad_sc_3, grad_f_3, adj_3 = _eb._elastic_born_backward_sc_compiled(
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            boundary,
            n_shots,
            mpml,
            tuple(tensor_args),
            grad_r_sc,
            sc_seeds=sc_seeds,
        )
    finally:
        deepwave._engine.run_backward_core = real_run_backward_core

    # Near-bit-identity, not exact equality: the chunked run takes a genuinely
    # different code path (per-chunk kernel calls with intermediate
    # SETUP_STORE_LOAD reads and an aux-buffer swap between chunks, vs one
    # monolithic loop), so -Ofast/-ffast-math floating-point reassociation
    # differences accumulate beyond the ~1e-13 single-call ULP floor observed
    # elsewhere (see the store_bg/store_sc near-equality rationale above);
    # empirically ~1e-9 here (measured up to 1.05e-9 for lamb). This is still
    # far tighter than anything a real swap-parity bug would produce (a wrong
    # swap corrupts PML-region values at order 1, not 1e-9); 1e-8 gives
    # headroom above the measured noise floor while remaining many orders of
    # magnitude below any real-bug scale.
    tol = 1e-8
    checked = 0
    for name, g1, g3 in zip(elastic_model_names(ndim), grad_sc_1, grad_sc_3):
        denom = g1.abs().max()
        if denom == 0:
            assert g3.abs().max() == 0, f"{name}: zero vs nonzero across chunking"
            continue
        rel = (g1 - g3).abs().max() / denom
        assert rel < tol, f"{name}: chunked vs single-chunk grad rel={rel}"
        checked += 1
    for i, (f1, f3) in enumerate(zip(grad_f_1, grad_f_3)):
        if f1.numel() == 0:
            assert f3.numel() == 0
            continue
        denom = f1.abs().max()
        if denom == 0:
            assert f3.abs().max() == 0, f"source {i}: zero vs nonzero across chunking"
            continue
        rel = (f1 - f3).abs().max() / denom
        assert rel < tol, f"source {i}: chunked vs single-chunk grad rel={rel}"
    for i, (a1, a3) in enumerate(zip(adj_1, adj_3)):
        denom = a1.abs().max()
        if denom == 0:
            assert a3.abs().max() == 0, f"wavefield {i}: zero vs nonzero, chunking"
            continue
        rel = (a1 - a3).abs().max() / denom
        assert rel < tol, f"wavefield {i}: chunked vs single-chunk grad rel={rel}"
    assert checked > 0  # sanity: the scattering-model gradients are nontrivial


@pytest.mark.parametrize("ndim", [2, 3])
def test_compiled_backward_full_multi_chunk_matches_single_chunk(ndim):
    """Chunked full two-field backward (small callback_frequency) matches single call.

    The full ``backward`` kernel's ``m_sigma*`` memory-variable aux buffers
    -- BOTH the background-field and the scattered-field families
    independently -- now ping-pong across chunk
    boundaries (:func:`deepwave.elastic_born._elastic_born_swap`), replacing
    the old per-call calloc/free that made every call implicitly a single
    full-``nt`` chunk. Mirrors
    ``test_compiled_backward_sc_multi_chunk_matches_single_chunk`` exactly
    (see its docstring for the full rationale), doubled: this forces
    ``deepwave._engine.run_backward_core`` to run with
    ``callback_frequency=3`` via a monkeypatch wrapping the
    ``elastic_born_backward_compiled`` bridge, and asserts the resulting
    background-model, scattering-model, background/scattered source-
    amplitude, and background/scattered initial-wavefield gradients match an
    ordinary (single-chunk) call to near machine precision. Getting either
    aux family's ordering wrong (independently) corrupts ONLY PML-region
    adjoints -- interior-only checks would pass regardless -- so this runs at
    ``pml=3``, where every memory-variable recursion is live.
    """
    import deepwave._common

    s = _born_setup(ndim)
    real_func = _eb.elastic_born_func
    captured = {}

    def spy(*args, **kwargs):
        captured["args"] = args
        return real_func(*args, **kwargs)

    common = dict(s["common"])
    common["backend"] = "compiled"
    common["pml_width"] = 3
    # elastic_born_backward_compiled's owner-family requires_grad flags (see
    # its docstring) are read off the ORIGINAL model tensors' .requires_grad,
    # not forced unconditionally (unlike the backward_sc bridge) -- so every
    # model family must actually request grad here for grad_bg/grad_sc to be
    # nontrivial (checked below).
    lamb = s["lamb"].clone().requires_grad_(True)
    mu = s["mu"].clone().requires_grad_(True)
    buoyancy = s["buoyancy"].clone().requires_grad_(True)
    dlamb = s["dlamb"].clone().requires_grad_(True)
    dmu = s["dmu"].clone().requires_grad_(True)
    dbuoyancy = s["dbuoyancy"].clone().requires_grad_(True)
    _eb.elastic_born_func = spy
    try:
        _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            **s["rk"],
            **s["sk"],
            **common,
        )
    finally:
        _eb.elastic_born_func = real_func

    args = captured["args"]
    (
        (
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_shots,
        ),
        boundary,
        mpml,
        tensor_args,
    ) = _split_captured_func_args(args)
    n_wf = len(elastic_wavefield_names(ndim))
    n_src = ndim + 1

    outs = _eb.elastic_born_eager_core(
        pml_profiles,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        boundary,
        n_shots,
        mpml,
        *tensor_args,
    )
    bg_final = list(outs[:n_wf])
    sc_final = list(outs[n_wf : 2 * n_wf])
    bg_amps = list(outs[2 * n_wf : 2 * n_wf + n_src])
    sc_amps = list(outs[2 * n_wf + n_src : 2 * n_wf + 2 * n_src])
    wgen = torch.Generator().manual_seed(17)
    bg_seeds = [torch.randn(a.shape, generator=wgen, dtype=a.dtype) for a in bg_final]
    sc_seeds = [torch.randn(a.shape, generator=wgen, dtype=a.dtype) for a in sc_final]

    def make_grad_r(amps: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.randn(a.shape, generator=wgen, dtype=a.dtype)
            if a.numel() > 0
            else torch.empty(0, dtype=a.dtype)
            for a in amps
        ]

    grad_r_bg = make_grad_r(bg_amps)
    grad_r_sc = make_grad_r(sc_amps)

    real_run_backward_core = deepwave._engine.run_backward_core
    forced_frequency = {"value": None}

    def patched_run_backward_core(*a, **k):
        if forced_frequency["value"] is not None:
            k = dict(k)
            k["callback_frequency"] = forced_frequency["value"]
            k["backward_callback"] = lambda _state: None
        return real_run_backward_core(*a, **k)

    deepwave._engine.run_backward_core = patched_run_backward_core
    try:
        forced_frequency["value"] = None  # single chunk (the existing default)
        (
            grad_bg_1,
            grad_sc_1,
            grad_f_bg_1,
            grad_f_sc_1,
            bg_adj_1,
            sc_adj_1,
        ) = _eb.elastic_born_backward_compiled(
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            boundary,
            n_shots,
            mpml,
            tuple(tensor_args),
            grad_r_bg,
            grad_r_sc,
            bg_seeds,
            sc_seeds,
        )
        forced_frequency["value"] = 3  # multiple chunks: exercises the aux swap
        (
            grad_bg_3,
            grad_sc_3,
            grad_f_bg_3,
            grad_f_sc_3,
            bg_adj_3,
            sc_adj_3,
        ) = _eb.elastic_born_backward_compiled(
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            boundary,
            n_shots,
            mpml,
            tuple(tensor_args),
            grad_r_bg,
            grad_r_sc,
            bg_seeds,
            sc_seeds,
        )
    finally:
        deepwave._engine.run_backward_core = real_run_backward_core

    # Near-bit-identity, not exact equality: see
    # test_compiled_backward_sc_multi_chunk_matches_single_chunk's identical
    # rationale (chunked vs. monolithic code paths reassociate floating-point
    # sums differently under -Ofast/-ffast-math). Same tolerance choice.
    tol = 1e-8

    def compare(label: str, g1: list[torch.Tensor], g3: list[torch.Tensor]) -> int:
        checked = 0
        for i, (a, b) in enumerate(zip(g1, g3)):
            if a.numel() == 0:
                assert b.numel() == 0, f"{label}[{i}]: numel mismatch across chunking"
                continue
            denom = a.abs().max()
            if denom == 0:
                assert b.abs().max() == 0, f"{label}[{i}]: zero vs nonzero, chunking"
                continue
            rel = (a - b).abs().max() / denom
            assert rel < tol, f"{label}[{i}]: chunked vs single-chunk rel={rel}"
            checked += 1
        return checked

    checked = 0
    checked += compare("grad_bg", grad_bg_1, grad_bg_3)
    checked += compare("grad_sc", grad_sc_1, grad_sc_3)
    compare("grad_f_bg", grad_f_bg_1, grad_f_bg_3)
    compare("grad_f_sc", grad_f_sc_1, grad_f_sc_3)
    compare("bg_adjoint_final", bg_adj_1, bg_adj_3)
    compare("sc_adjoint_final", sc_adj_1, sc_adj_3)
    assert checked > 0  # sanity: at least some model gradients are nontrivial


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ndim", [2, 3])
def test_public_born_forward_cpu_gpu_match(ndim):
    """Compiled-backend forward output matches between CPU and GPU.

    elastic_born has no compiled CUDA adjoint kernel yet, so only the
    forward pass is compared here --
    mirroring ``test_elastic.py``'s ``test_forward_cpu_gpu_match`` but
    forward-only. Runs only on the maintainer's GPU (skipped here, no CUDA on
    this box); its value while skipped is making the GPU-adjoint gap visible.
    """
    s = _born_setup(ndim)
    common = dict(s["common"])
    common["backend"] = "compiled"  # compiled backend

    def run(device):
        lamb = s["lamb"].to(device)
        mu = s["mu"].to(device)
        buoyancy = s["buoyancy"].to(device)
        dlamb = s["dlamb"].to(device)
        dmu = s["dmu"].to(device)
        dbuoyancy = s["dbuoyancy"].to(device)
        sk = {k: v.to(device) for k, v in s["sk"].items()}
        rk = {k: v.to(device) for k, v in s["rk"].items()}
        out = _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            **rk,
            **sk,
            **common,
        )
        return [o.detach().cpu() for o in output_tensors(out)]

    out_cpu = run(torch.device("cpu"))
    out_gpu = run(torch.device("cuda"))
    for a, b in zip(out_cpu, out_gpu):
        if a.numel() == 0:
            continue
        scale = max(a.abs().max().item(), 1.0)
        assert (a - b).abs().max().item() <= scale * 1e-6


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_backward_on_cuda_uses_eager_fallback():
    """On CUDA, the compiled backend's gradients are correct (whichever route).

    As of task N, elastic_born's compiled backend has CUDA adjoint kernels
    (``backward``/``backward_sc``) in ABI lockstep with the CPU twins, so
    ``_use_compiled_adjoint`` now prefers them on CUDA (see that function and
    ``ElasticBornForwardFunc``'s docstring); only M-PML still forces the eager,
    ``torch.enable_grad()``-recomputed VJP bridge on every device. This test
    predates that change and does not assert which route was taken -- it only
    checks that the CUDA backward produces a correct, non-zero gradient
    (matching the CPU compiled-backend gradient) rather than raising or
    silently returning zeros, so it remains a valid regression guard either
    way. Runs only on the maintainer's GPU (skipped here, no CUDA on this box).
    """
    s = _born_setup(2)
    common = dict(s["common"])
    common["backend"] = "compiled"  # compiled backend

    def run(device):
        lamb = s["lamb"].to(device).clone().requires_grad_(True)
        mu = s["mu"].to(device).clone().requires_grad_(True)
        buoyancy = s["buoyancy"].to(device).clone().requires_grad_(True)
        dlamb = s["dlamb"].to(device)
        dmu = s["dmu"].to(device)
        dbuoyancy = s["dbuoyancy"].to(device)
        sk = {k: v.to(device) for k, v in s["sk"].items()}
        rk = {k: v.to(device) for k, v in s["rk"].items()}
        out = _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            s["dx"],
            s["dt"],
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            **rk,
            **sk,
            **common,
        )
        loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
        loss.backward()
        return (
            lamb.grad.detach().cpu(),
            mu.grad.detach().cpu(),
            buoyancy.grad.detach().cpu(),
        )

    g_cpu = run(torch.device("cpu"))
    g_cuda = run(torch.device("cuda"))
    for gc, gg in zip(g_cpu, g_cuda):
        assert gg.abs().max() > 0
        scale = max(gc.abs().max().item(), 1.0)
        assert (gc - gg).abs().max().item() <= scale * 1e-4


def test_use_compiled_adjoint_dispatch_logic():
    """The compiled-adjoint dispatch rule is correct without needing a GPU.

    The compiled scattered-only and full adjoint kernels exist on CPU and (as of
    task N) on CUDA, both only without M-PML; under M-PML the backward routes to
    the eager bridge on either device. The compiled GPU path is additionally
    gated on a live CUDA runtime, so on this CPU-only box the CUDA case still
    resolves to the eager bridge. This pins that pure decision.
    """
    import torch

    from deepwave.elastic_born import _use_compiled_adjoint

    # CPU, no M-PML -> compiled adjoints usable.
    assert _use_compiled_adjoint(mpml=False, is_cuda=False) is True
    # CUDA, no M-PML -> compiled adjoints usable iff a CUDA runtime is present
    # (the GPU kernels landed in task N; gated on torch.cuda.is_available()).
    assert _use_compiled_adjoint(mpml=False, is_cuda=True) is torch.cuda.is_available()
    # M-PML (no compiled M-PML adjoint) -> eager bridge, on either device.
    assert _use_compiled_adjoint(mpml=True, is_cuda=False) is False
    assert _use_compiled_adjoint(mpml=True, is_cuda=True) is False


def test_initial_state_round_trip_propagator_output() -> None:
    """A whole PropagatorOutput can be fed back in as ``initial_state`` (2-D).

    Regression test: previously passing a
    :class:`deepwave.PropagatorOutput` crashed (it has no ``.get`` /
    ``.values`` a mapping has, and its receiver-amplitude fields are not
    re-feedable state). Splitting a run in two and continuing the second half
    from the first half's returned output must reproduce the single full run
    (background *and* scattered receiver amplitudes), exercising the round trip.
    """
    s = _born_setup(2)
    comp = s["comp"]
    src = s["sk"][f"source_amplitudes_{comp}"]
    sloc = s["sk"][f"source_locations_{comp}"]
    nt = src.shape[-1]
    n1 = nt // 2
    bg_rk = {
        f"bg_receiver_locations_{comp}": s["rloc"],
        "bg_receiver_locations_p": s["rloc"],
    }

    def run(samp, n, **extra):
        return _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **{f"source_amplitudes_{comp}": samp, f"source_locations_{comp}": sloc},
            **s["rk"],
            **bg_rk,
            nt=n,
            **s["common"],
            **extra,
        )

    full = run(src, nt)
    part1 = run(src[:, :, :n1], n1)
    part2 = run(src[:, :, n1:], nt - n1, initial_state=part1)

    for field in (
        f"sc_receiver_amplitudes_{comp}",
        "sc_receiver_amplitudes_p",
        f"bg_receiver_amplitudes_{comp}",
        "bg_receiver_amplitudes_p",
    ):
        r_full = getattr(full, field)
        r_cat = torch.cat([getattr(part1, field), getattr(part2, field)], dim=-1)
        assert r_full.abs().max() > 0, field
        assert torch.allclose(r_full, r_cat, atol=1e-9, rtol=1e-6), field


def test_compiled_backward_boundary_adjoint_parity() -> None:
    """Compiled CPU backward matches eager to ~1e-9 at reflecting faces.

    Reproduces the well-conditioned harness used to root-cause and fix the
    compiled backward's reflecting-``boundary=`` adjoint bug (the compiled
    backward re-applied the forward guard-ring fill to the adjoint fields
    verbatim; elastic_born's multi-field fill, like elastic's, is not
    self-adjoint -- see ``fill_elastic_stress_bc_adjoint``/
    ``fill_elastic_velocity_bc_adjoint`` in elastic_born.c). ``nt=1200`` is
    long enough that waves genuinely reflect off the non-PML faces many
    times, so a wrong near-face adjoint contribution is not lost in
    numerical dust. Checks background (lamb/mu/buoyancy) AND scattering
    (dlamb/dmu/dbuoyancy) gradients together, since elastic_born's full
    two-field compiled adjoint folds both fields' guard rings in one pass.
    """
    torch.manual_seed(0)
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    rho, c, cs = 2200.0, 1500.0, 900.0
    lamb = torch.full((ny, nx), rho * (c**2 - 2 * cs**2), dtype=torch.double)
    mu = torch.full((ny, nx), rho * cs**2, dtype=torch.double)
    buoyancy = torch.full((ny, nx), 1.0 / rho, dtype=torch.double)
    dlamb = (lamb * 0.02 * torch.randn(ny, nx, dtype=torch.double)).detach()
    dmu = (mu * 0.02 * torch.randn(ny, nx, dtype=torch.double)).detach()
    dbuoyancy = (buoyancy * 0.02 * torch.randn(ny, nx, dtype=torch.double)).detach()
    kw = {
        "source_amplitudes_pressure_rate": sa.double(),
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "sc_receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": [0, 0, 10, 10],
        "boundary": ["rigid", "pml", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def grad_of(backend, name):
        params = {
            "lamb": lamb,
            "mu": mu,
            "buoyancy": buoyancy,
            "dlamb": dlamb,
            "dmu": dmu,
            "dbuoyancy": dbuoyancy,
        }
        tens = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        out = _eb.elastic_born(
            tens["lamb"],
            tens["mu"],
            tens["buoyancy"],
            dx,
            dt,
            dlamb=tens["dlamb"],
            dmu=tens["dmu"],
            dbuoyancy=tens["dbuoyancy"],
            backend=backend,
            **kw,
        )
        loss = out.sc_receiver_amplitudes_p.sum()
        return torch.autograd.grad(loss, tens[name])[0]

    for name in ("lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy"):
        gc = grad_of("compiled", name)
        ge = grad_of("eager", name)
        denom = ge.abs().max().item()
        assert denom > 0, name
        relerr = (gc - ge).abs().max().item() / denom
        assert relerr < 1e-9, f"{name}: rel err {relerr:.3e}"


def test_compiled_forward_final_state_matches_eager_boundary() -> None:
    """Compiled forward's returned wavefields are boundary-consistent (2-D).

    Eager applies a final stress/velocity boundary fill after its time loop
    (see the end of ``elastic_born_eager_core``), so eager's returned
    on-surface normal stress is exactly zero on a free face and its guard
    rings are exactly the mirror images the fill imposes. The compiled
    forward's last timestep update runs after its last in-loop fill, so its
    raw returned wavefields would be boundary-inconsistent without a
    matching post-loop fill. Checks both the background and scattered field
    sets, at low-to-machine precision (the compiled/eager forward paths use
    the identical stencil arithmetic once boundary-consistent, so the
    residual gap is FD-associativity noise, not model error).
    """
    torch.manual_seed(1)
    ny, nx = 14, 14
    dx, dt, nt = 5.0, 0.0001, 60
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    rho, c, cs = 2200.0, 1500.0, 900.0
    lamb = torch.full((ny, nx), rho * (c**2 - 2 * cs**2), dtype=torch.double)
    mu = torch.full((ny, nx), rho * cs**2, dtype=torch.double)
    buoyancy = torch.full((ny, nx), 1.0 / rho, dtype=torch.double)
    dlamb = (lamb * 0.02 * torch.randn(ny, nx, dtype=torch.double)).detach()
    dmu = (mu * 0.02 * torch.randn(ny, nx, dtype=torch.double)).detach()
    dbuoyancy = (buoyancy * 0.02 * torch.randn(ny, nx, dtype=torch.double)).detach()
    kw = {
        "source_amplitudes_pressure_rate": sa.double(),
        "source_locations_pressure_rate": torch.tensor([[[7, 7]]]),
        "sc_receiver_locations_p": torch.tensor([[[4, 4]]]),
        "pml_width": [0, 0, 10, 10],
        "boundary": ["rigid", "pml", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 25.0,
    }
    out_c = _eb.elastic_born(
        lamb, mu, buoyancy, dx, dt, dlamb=dlamb, dmu=dmu, dbuoyancy=dbuoyancy, **kw
    )
    out_e = _eb.elastic_born(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        dlamb=dlamb,
        dmu=dmu,
        dbuoyancy=dbuoyancy,
        backend="eager",
        **kw,
    )
    checked = 0
    for name in _eb.elastic_born_output_names(2):
        if name.startswith(("sc_receiver_amplitudes", "bg_receiver_amplitudes")):
            continue
        # Every field is one tensor except auxiliary_state, which holds the
        # folded absorbing-boundary memory variables -- compare all of them.
        for fc, fe in zip(
            element_tensors(getattr(out_c, name)),
            element_tensors(getattr(out_e, name)),
        ):
            denom = max(fe.abs().max().item(), 1e-30)
            relerr = (fc - fe).abs().max().item() / denom
            assert relerr < 1e-9, f"{name}: rel err {relerr:.3e}"
            checked += 1
    assert checked >= 10, "sanity: expected many wavefield outputs to be checked"


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    s = _born_setup(2)
    with pytest.raises(RuntimeError, match="Storage compression"):
        _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            storage_compression=True,
            **s["rk"],
            **s["sk"],
            **s["common"],
        )


def _born_callback_setup(ndim):
    """Requires-grad background+scattering setup for callback-dict tests.

    Mirrors ``_born_setup``'s validated shapes/locations/timestep (proven
    CFL-safe for elastic_born) but marks every model requires_grad=True, so a
    ``backward_callback`` run dispatches through the full two-field adjoint
    (``ElasticBornForwardFunc._backward_full_dispatch``), exercising both the
    background and scattered wavefield/model blocks in one call. Drops
    ``backend`` from ``common`` so the default (``"compiled"``)
    backend is used, since only the compiled backend supports
    ``backward_callback`` (see
    ``test_elastic_born_eager_backend_rejects_backward_callback``).
    """
    s = _born_setup(ndim)
    lamb = s["lamb"].detach().clone().requires_grad_()
    mu = s["mu"].detach().clone().requires_grad_()
    buoyancy = s["buoyancy"].detach().clone().requires_grad_()
    dlamb = s["dlamb"].detach().clone().requires_grad_()
    dmu = s["dmu"].detach().clone().requires_grad_()
    dbuoyancy = s["dbuoyancy"].detach().clone().requires_grad_()
    common = dict(s["common"])
    del common["backend"]
    return {
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "dlamb": dlamb,
        "dmu": dmu,
        "dbuoyancy": dbuoyancy,
        "dx": s["dx"],
        "dt": s["dt"],
        "sk": s["sk"],
        "rk": s["rk"],
        "common": common,
    }


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_born_backward_callback_dict_keys(ndim) -> None:
    """The backward-callback dict's key set matches elastic_born_wavefield_names.

    elastic_born doubles the wavefield set (background block then scattered
    block, each in ``elastic.elastic_wavefield_names`` order, see
    ``elastic_born_wavefield_names``). Its compiled backend shares
    ``deepwave._engine.run_backward_core`` with elastic/elastic_vti, which
    previously dropped the last wavefield name from every propagator's
    backward-callback dict via a ``callback_drop_last=1`` parameter (fixed in
    837b3cb, which added the equivalent test for elastic/elastic_vti/
    acoustic_vti but not elastic_born -- this is that missing coverage).
    Both background and scattering models require grad here so the backward
    pass dispatches through the full two-field adjoint, populating both
    field blocks in the recorded dict in a single run.
    """
    s = _born_callback_setup(ndim)
    recorded: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)

    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        backward_callback=recorder,
        callback_frequency=1,
        **s["sk"],
        **s["rk"],
        **s["common"],
    )
    total = sum(t.sum() for t in output_tensors(out))
    total.backward()
    assert set(recorded) == set(_eb.elastic_born_wavefield_names(ndim))
    for name, t in recorded.items():
        assert t.numel() > 0, f"{name}: zero numel in backward-callback dict"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_born_forward_callback_dict_keys(ndim) -> None:
    """The forward-callback wavefield/model dicts' key sets match the doubled sets.

    Exercises the compiled backend's forward callback: the wavefield dict
    against ``elastic_born_wavefield_names`` (background+scattered) and the
    model dict against ``elastic_born_model_names`` (background+scattering
    models). The eager backend's own forward_callback dict keys are checked
    separately (see ``test_elastic_born_eager_forward_callback_dict_keys``);
    only ``backward_callback`` is compiled-backend only (see
    ``test_elastic_born_eager_backend_rejects_backward_callback``).
    """
    s = _born_callback_setup(ndim)
    recorded_wf: dict[str, torch.Tensor] = {}
    recorded_models: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_wf.update(state._wavefields)
        recorded_models.update(state._models)

    _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        forward_callback=recorder,
        callback_frequency=1,
        **s["sk"],
        **s["rk"],
        **s["common"],
    )
    assert set(recorded_wf) == set(_eb.elastic_born_wavefield_names(ndim))
    assert set(recorded_models) == set(_eb.elastic_born_model_names(ndim))
    for name, t in recorded_wf.items():
        assert t.numel() > 0, f"{name}: zero numel in forward-callback wavefield dict"
    for name, t in recorded_models.items():
        assert t.numel() > 0, f"{name}: zero numel in forward-callback model dict"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_born_backward_callback_model_and_gradient_dict_keys(ndim) -> None:
    """The backward-callback model/gradient dicts cover background+scattering models.

    ``elastic_born_model_names`` doubles the model set (background lamb/mu/
    buoyancy family, then the ``_sc``-suffixed scattering perturbations); the
    compiled backend's model/gradient dicts (``CallbackState._models``/
    ``_gradients``) are built by ``deepwave._engine.run_backward_core`` from
    ``SPEC.model_names(ndim)`` and were never asserted against by name.
    """
    s = _born_callback_setup(ndim)
    recorded_models: dict[str, torch.Tensor] = {}
    recorded_grads: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_models.update(state._models)
        recorded_grads.update(state._gradients)

    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        backward_callback=recorder,
        callback_frequency=1,
        **s["sk"],
        **s["rk"],
        **s["common"],
    )
    total = sum(t.sum() for t in output_tensors(out))
    total.backward()
    expected = set(_eb.elastic_born_model_names(ndim))
    assert set(recorded_models) == expected
    assert set(recorded_grads) == expected
    for name, t in recorded_models.items():
        assert t.numel() > 0, f"{name}: zero numel in model dict"
    for name, t in recorded_grads.items():
        assert t.numel() > 0, f"{name}: zero numel in gradient dict"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_born_output_names(ndim) -> None:
    """PropagatorOutput field names match elastic_born_output_names exactly."""
    s = _born_setup(ndim)
    out = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["sk"],
        **s["rk"],
        **s["common"],
    )
    assert out._names == tuple(_eb.elastic_born_output_names(ndim))


def test_elastic_born_eager_backend_rejects_backward_callback() -> None:
    """backend="eager" rejects backward_callback.

    elastic_born's eager time loop runs native PyTorch autograd with no
    chunked-adjoint machinery at all, so it rejects backward_callback outright
    rather than silently ignoring it. forward_callback has no such dependency
    (the eager driver fires it from its own per-step loop) and is no longer
    rejected -- see ``test_elastic_born_eager_forward_callback_dict_keys``.
    """
    s = _born_setup(2)
    with pytest.raises(RuntimeError, match="not supported by"):
        _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            backward_callback=lambda _state: None,
            **s["rk"],
            **s["sk"],
            **s["common"],
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_born_eager_forward_callback_dict_keys(ndim) -> None:
    """The eager forward_callback's wavefield/model dicts match the compiled backend's.

    Mirrors ``test_elastic_born_forward_callback_dict_keys`` but on
    ``backend="eager"``.
    """
    s = _born_callback_setup(ndim)
    common = dict(s["common"])
    common["backend"] = "eager"
    recorded_wf: dict[str, torch.Tensor] = {}
    recorded_models: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_wf.update(state._wavefields)
        recorded_models.update(state._models)

    _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        forward_callback=recorder,
        callback_frequency=1,
        **s["sk"],
        **s["rk"],
        **common,
    )
    assert set(recorded_wf) == set(_eb.elastic_born_wavefield_names(ndim))
    assert set(recorded_models) == set(_eb.elastic_born_model_names(ndim))
    for name, t in recorded_wf.items():
        assert t.numel() > 0, f"{name}: zero numel in forward-callback wavefield dict"
    for name, t in recorded_models.items():
        assert t.numel() > 0, f"{name}: zero numel in forward-callback model dict"


def test_elastic_born_eager_forward_callback_call_count_and_equivalence() -> None:
    """The eager forward_callback fires the documented count without altering output."""
    s = _born_setup(2)
    common = dict(s["common"])
    common["backend"] = "eager"
    nt = s["sk"][f"source_amplitudes_{s['comp']}"].shape[-1]

    count = {"n": 0}

    def counter(state: deepwave.CallbackState) -> None:
        count["n"] += 1

    def nothing(state: deepwave.CallbackState) -> None:
        pass

    def run(callback) -> tuple[torch.Tensor, ...]:
        extra = (
            {"forward_callback": callback, "callback_frequency": 2}
            if callback is not None
            else {}
        )
        return _eb.elastic_born(
            s["lamb"],
            s["mu"],
            s["buoyancy"],
            s["dx"],
            s["dt"],
            dlamb=s["dlamb"],
            dmu=s["dmu"],
            dbuoyancy=s["dbuoyancy"],
            **s["sk"],
            **s["rk"],
            **common,
            **extra,
        )

    run(counter)
    assert count["n"] == nt // 2

    out_none = run(None)
    out_nothing = run(nothing)
    for a, b in zip(output_tensors(out_none), output_tensors(out_nothing)):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# location_interpolation: mirrors the on-node/off-node gates
# tests/test_location_interpolation.py runs for deepwave.elastic, ported to
# the Born propagator's doubled background/scattered source-cell wiring.
# ---------------------------------------------------------------------------


def test_location_interpolation_on_node_float_equals_integer_velocity() -> None:
    """A staggered velocity component's on-node float coordinate is N + 0.5."""
    s = _born_setup(2)
    comp = s["comp"]
    offset = torch.tensor(
        [0.5, 0.0] if comp == "y" else [0.0, 0.5], dtype=torch.float64
    )
    loc_key = f"source_locations_{comp}"
    rloc_key = f"sc_receiver_locations_{comp}"
    loc_int = s["sk"][loc_key]
    rloc_int = s["rk"][rloc_key]

    exact = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["sk"],
        **s["rk"],
        **s["common"],
    )
    snapped_sk = dict(s["sk"])
    snapped_sk[loc_key] = loc_int.double() + offset
    snapped_rk = dict(s["rk"])
    snapped_rk[rloc_key] = rloc_int.double() + offset
    snapped = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **snapped_sk,
        **snapped_rk,
        **s["common"],
    )
    rec_name = f"sc_receiver_amplitudes_{comp}"
    assert torch.equal(getattr(exact, rec_name), getattr(snapped, rec_name))


def test_location_interpolation_off_node_differs_and_eager_matches_compiled() -> None:
    """A genuinely off-grid source is Hicks-interpolated, agreeing across backends."""
    s = _born_setup(2)
    comp = s["comp"]
    loc_key = f"source_locations_{comp}"
    loc_int = s["sk"][loc_key]
    off_sk = dict(s["sk"])
    off_sk[loc_key] = loc_int.double() + 0.3
    common_eager = dict(s["common"])
    common_eager["backend"] = "eager"
    common_compiled = dict(s["common"])
    common_compiled["backend"] = "compiled"

    on_node = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **s["sk"],
        **s["rk"],
        **common_compiled,
    )
    off_node_compiled = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **off_sk,
        **s["rk"],
        **common_compiled,
    )
    off_node_eager = _eb.elastic_born(
        s["lamb"],
        s["mu"],
        s["buoyancy"],
        s["dx"],
        s["dt"],
        dlamb=s["dlamb"],
        dmu=s["dmu"],
        dbuoyancy=s["dbuoyancy"],
        **off_sk,
        **s["rk"],
        **common_eager,
    )
    rec_name = f"sc_receiver_amplitudes_{comp}"
    assert not torch.equal(
        getattr(on_node, rec_name), getattr(off_node_compiled, rec_name)
    )
    assert torch.allclose(
        getattr(off_node_compiled, rec_name),
        getattr(off_node_eager, rec_name),
        atol=1e-8,
    )


def test_elastic_born_source_locations_y_out_of_bounds() -> None:
    """A velocity-component source must not sit on the final staggered sample.

    Mirrors ``deepwave.elastic``'s identical check (see
    ``test_elastic_source_locations_y_out_of_bounds``); ``elastic_born`` had
    no equivalent guard at all.
    """
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    dlamb = torch.zeros(10, 10)
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
            "elastic_born: source_locations_y is out of bounds: a "
            "y location is staggered half a cell past the "
            "node it names, so its y coordinate must be "
            "less than 9."
        ),
    ):
        _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            dlamb=dlamb,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            nt=nt,
        )


def test_elastic_born_receiver_locations_x_out_of_bounds() -> None:
    """A velocity-component receiver must not sit on the final staggered sample.

    Covers both the scattered (``sc_receiver_locations_x``) and background
    (``bg_receiver_locations_x``) receiver lists -- both were previously
    unchecked.
    """
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    dlamb = torch.zeros(10, 10)
    grid_spacing = 1.0
    dt = 0.0001
    sc_receiver_locations_x = torch.tensor(
        [[[1, 9]]], dtype=torch.long
    )  # x-coord is 9, max is 8 (shape[1]-1)
    nt = 10
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "elastic_born: sc_receiver_locations_x is out of bounds: a "
            "x location is staggered half a cell past the "
            "node it names, so its x coordinate must be "
            "less than 9."
        ),
    ):
        _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            dlamb=dlamb,
            sc_receiver_locations_x=sc_receiver_locations_x,
            nt=nt,
        )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "elastic_born: bg_receiver_locations_x is out of bounds: a "
            "x location is staggered half a cell past the "
            "node it names, so its x coordinate must be "
            "less than 9."
        ),
    ):
        _eb.elastic_born(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            dlamb=dlamb,
            bg_receiver_locations_x=sc_receiver_locations_x,
            nt=nt,
        )


# ---------------------------------------------------------------------------
# Batched models
# ---------------------------------------------------------------------------


def _batched_born_problem(
    dtype: torch.dtype = torch.double,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """A two-shot 2-D problem whose two shots see identical models."""
    shape = (14, 14)
    nt = 20
    torch.manual_seed(0)
    vp = torch.full(shape, 1500.0, dtype=dtype)
    vs = torch.full(shape, 800.0, dtype=dtype)
    rho = torch.full(shape, 2200.0, dtype=dtype)
    lamb = rho * (vp**2 - 2 * vs**2)
    mu = rho * vs**2
    buoyancy = 1 / rho
    dlamb = torch.zeros(shape, dtype=dtype)
    dmu = torch.zeros(shape, dtype=dtype)
    dbuoyancy = torch.zeros(shape, dtype=dtype)
    dlamb[7, 7] = 1e7
    dmu[7, 7] = 1e7
    dbuoyancy[7, 7] = 1e-6
    models = {
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "dlamb": dlamb,
        "dmu": dmu,
        "dbuoyancy": dbuoyancy,
    }
    survey: dict[str, object] = {
        "source_amplitudes_y": torch.randn(1, 1, nt, dtype=dtype).repeat(2, 1, 1),
        "source_locations_y": torch.tensor([[[7, 7]], [[7, 7]]]),
        "sc_receiver_locations_y": torch.tensor([[[7, 4]], [[7, 4]]]),
        "bg_receiver_locations_y": torch.tensor([[[7, 4]], [[7, 4]]]),
        "pml_width": 4,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
    }
    return models, survey


def _batch_model(model: torch.Tensor) -> torch.Tensor:
    return model.unsqueeze(0).repeat(2, *([1] * model.ndim))


def _run_batched_born(models, survey, **overrides):
    call = dict(models)
    call.update(overrides)
    return deepwave.elastic_born(
        call["lamb"],
        call["mu"],
        call["buoyancy"],
        6.0,
        0.0004,
        dlamb=call["dlamb"],
        dmu=call["dmu"],
        dbuoyancy=call["dbuoyancy"],
        accuracy=4,
        **survey,
    )


@pytest.mark.parametrize(
    ("batched_background", "batched_perturbation"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_batched_models_agree_with_the_unbatched_call(
    batched_background: bool, batched_perturbation: bool
) -> None:
    """Every combination of batched and unbatched models runs and agrees.

    ``prepare_scatter_parameters`` pushes ``(dmu, dbuoyancy)`` through
    ``torch.func.jvp``, which requires each tangent to have exactly its
    primal's shape. A batched background model with an unbatched
    perturbation (or the reverse) therefore used to raise a bare
    ``RuntimeError: Trying to set a forward gradient that has a different
    size than that of the original Tensor``, with nothing to tell the caller
    that the batch status of the two was the problem. The models are now
    equalised on their leading dimension first, so with identical models in
    both shots all four combinations are bit-identical.
    """
    models, survey = _batched_born_problem()
    overrides: dict[str, torch.Tensor] = {}
    if batched_background:
        for name in ("lamb", "mu", "buoyancy"):
            overrides[name] = _batch_model(models[name])
    if batched_perturbation:
        for name in ("dlamb", "dmu", "dbuoyancy"):
            overrides[name] = _batch_model(models[name])
    out = _run_batched_born(models, survey, **overrides)
    reference = _run_batched_born(models, survey)
    assert out.sc_receiver_amplitudes_y.abs().max() > 0
    assert torch.equal(out.sc_receiver_amplitudes_y, reference.sc_receiver_amplitudes_y)
    assert torch.equal(out.bg_receiver_amplitudes_y, reference.bg_receiver_amplitudes_y)


def test_batched_models_really_are_per_shot() -> None:
    """A batched model gives each shot its own model, so the shots differ."""
    models, survey = _batched_born_problem()
    mu = _batch_model(models["mu"])
    mu[1] *= 1.3
    dmu = _batch_model(models["dmu"])
    dmu[1] *= 4.0
    out = _run_batched_born(models, survey, mu=mu, dmu=dmu)
    scattered = out.sc_receiver_amplitudes_y
    # A relative comparison: the scattered amplitudes are tiny in absolute
    # terms, so ``allclose``'s default atol would call any two of them equal.
    assert (scattered[0] - scattered[1]).abs().max() > 0.1 * scattered.abs().max()
    reference = _run_batched_born(models, survey)
    assert torch.equal(scattered[0], reference.sc_receiver_amplitudes_y[0])


def test_batched_models_are_differentiable() -> None:
    """Gradients reach an unbatched model used alongside a batched one."""
    models, survey = _batched_born_problem()
    dmu = _batch_model(models["dmu"]).clone().requires_grad_(True)
    out = _run_batched_born(models, survey, dmu=dmu)
    out.sc_receiver_amplitudes_y.abs().sum().backward()
    assert dmu.grad is not None
    assert dmu.grad.shape == dmu.shape
    assert bool(torch.isfinite(dmu.grad).all())
    assert dmu.grad.abs().max() > 0

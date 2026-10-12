"""SymPy re-derivation of hand-written kernel adjoints (exact discrete adjoint).

This module independently re-derives, *symbolically*, the per-step reverse-mode
updates that the hand-written C backward kernels implement, and pins them as
tests. It targets a specific bug class repeated adjoint-review audits have
found in the analytic adjoints: wrong-block swaps (a gradient correlating the wrong pair
of fields), a missing ``step_ratio`` scaling on a model-gradient accumulation,
and a wrong ``requires_grad``/store gating.

Method (per audited scheme). On a tiny symbolic grid (PML-free interior, so the
CPML memory variables and their profile coefficients drop out), the *forward*
discrete update ``u_next = STEP(u_now, u_prev, model)`` is transcribed exactly
as the C kernel computes it -- every transcribed expression quotes the ``.c``/
``.h`` ``file:line`` it comes from in a comment. We then form the scalar
``L = sum_i lambda_i * (u_next)_i`` with independent adjoint symbols ``lambda``
(the seed of reverse mode), let SymPy compute ``d L / d(input)`` exactly, and
assert -- after ``simplify`` -- that it equals the expression the C *backward*
kernel implements (also transcribed with ``file:line``). Because the forward
step is linear in the wavefields and affine/linear in each model parameter, the
per-step reverse update *is* this derivative, so ``simplify(derivative - kernel)
== 0`` is an exact, backend-independent check of the transcribed adjoint.

What is covered:
  * scalar.c regular-grid interior step: the wavefield adjoint (the FD Laplacian
    transpose acting on ``v**2 dt**2`` times the adjoint field, plus the leapfrog
    ``+2 u_now`` / ``-u_prev`` terms) and the ``v`` gradient accumulation,
    including its ``step_ratio`` scaling. Checked in 1-D and 2-D, accuracy 2 and 4.
  * elastic.c 2-D isotropic constitutive step: the lambda, mu (on-node and the
    ``mu_yx`` half-cell average), and buoyancy (``buoyancy_x``/``buoyancy_y``)
    model-gradient accumulations, each with its ``step_ratio`` scaling.
  * elastic_born.c 2-D scattered-field coupling (a site prone to exactly this
    bug class): the scattering-model gradients ``d(buoyancy)`` (velocity
    source-injection coupling -- the representative term), ``d(lamb)``,
    ``d(mu)``, and ``d(mu_yx)`` (stress coupling). These verify that the
    scattered adjoint field is correlated with the *background* strain-rate
    store (not the scattered one) and that the ``step_ratio`` scaling is
    present. A negative control asserts the check would fail if the
    *scattered* store were used instead.

What is NOT covered (do not mistake this for full adjoint coverage):
  * The PML region: the CPML memory-variable (psi/zeta, m_*) recurrences and
    their ``a``/``b``/``dbdx`` profile coefficients are dropped by working in the
    PML-free interior. The whole-operator PML transpose is exercised numerically
    by ``tests/test_adjoint_dot_product.py`` instead.
  * Reflecting/rigid/periodic boundary faces and their guard-ring image fills.
  * The staggered-grid *wavefield* FD transpose for elastic/elastic_born (the
    velocity<->stress spatial coupling). Only the *model-parameter* gradients are
    re-derived here (these are diagonal in space, so the FD strain-rate operands
    are treated as independent symbols); the wavefield transpose is covered
    numerically by the dot-product test.
  * Storage/decimation plumbing (compression, disk vs device snapshots): only
    the arithmetic of what is stored and how it is scaled on read-back is
    checked, not the I/O path.
  * The actual choice of which time steps satisfy ``t % step_ratio == 0``: what
    is checked is that each per-step model-gradient increment carries the
    ``step_ratio`` factor, matching the every-``step_ratio`` decimation of the
    stored forward snapshots.

The transcription itself is the main drift risk (the C source could be misread),
which is why every transcribed line is quoted verbatim next to its Python form.
If any assertion here fails it is EITHER a transcription error in this file OR a
real kernel bug -- per the task, such a failure must be reported, not patched.
"""

import itertools

import pytest

sympy = pytest.importorskip("sympy")


# --- Finite-difference stencil coefficients, transcribed from regular_grid.h ---
# Second-derivative central-difference weights, keyed by accuracy then by |offset|
# (the C macros write the on-node term c0*a(0) plus symmetric pairs
# c_k*(a(k)+a(-k))). Transcribed verbatim from src/deepwave/regular_grid.h:
#   accuracy 2 -> DIFF*2 at :16-18   accuracy 4 -> :35-49
#   accuracy 6 -> :69-86             accuracy 8 -> :109-129
_SECOND_DERIV_COEFFS = {
    2: {0: sympy.Rational(-2), 1: sympy.Rational(1)},
    4: {0: sympy.Rational(-5, 2), 1: sympy.Rational(4, 3), 2: sympy.Rational(-1, 12)},
    6: {
        0: sympy.Rational(-49, 18),
        1: sympy.Rational(3, 2),
        2: sympy.Rational(-3, 20),
        3: sympy.Rational(1, 90),
    },
    8: {
        0: sympy.Rational(-205, 72),
        1: sympy.Rational(8, 5),
        2: sympy.Rational(-1, 5),
        3: sympy.Rational(8, 315),
        4: sympy.Rational(-1, 560),
    },
}


def _field(name, coords):
    """Build a dict of distinct SymPy symbols for a wavefield/model on ``coords``.

    ``coords`` is an iterable of integer coordinate tuples (length = ndim); the
    returned mapping sends each coordinate to a uniquely named symbol.
    """
    out = {}
    for c in coords:
        suffix = "_".join(f"m{-k}" if k < 0 else str(k) for k in c)
        out[c] = sympy.Symbol(f"{name}_{suffix}")
    return out


def _diff2(field, base, axis, acc, rd2):
    """Second-derivative FD stencil along ``axis`` at ``base`` (matches DIFF?2).

    Mirrors the ``DIFFZ2``/``DIFFY2``/``DIFFX2`` macros in regular_grid.h: a
    weighted sum of ``field`` over neighbours ``base +/- k`` along ``axis``,
    scaled by the reciprocal-squared grid spacing ``rd2``.
    """
    coeffs = _SECOND_DERIV_COEFFS[acc]
    half = acc // 2
    total = sympy.Integer(0)
    for k in range(-half, half + 1):
        nbr = tuple(base[d] + (k if d == axis else 0) for d in range(len(base)))
        total += coeffs[abs(k)] * field[nbr]
    return total * rd2


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("acc", [2, 4])
def test_scalar_interior_adjoint(ndim, acc):
    """scalar.c interior: wavefield transpose and ``v``-gradient are exact.

    Re-derives the reverse-mode update of the PML-free scalar step and checks it
    against ``FUNC(backward)`` in scalar.c, including the ``step_ratio`` scaling
    of the ``v`` gradient.
    """
    pad = acc // 2
    # A grid wide enough that every output cell within ``pad`` of the centre has
    # its full stencil in-bounds: coordinates span [-2*pad, 2*pad] per axis.
    coords = list(itertools.product(range(-2 * pad, 2 * pad + 1), repeat=ndim))
    u_now = _field("unow", coords)
    u_prev = _field("uprev", coords)
    lam = _field("lam", coords)  # adjoint seed: d L / d u_next
    v = _field("v", coords)
    dt2 = sympy.Symbol("dt2")
    rd2 = [sympy.Symbol(f"rd2_{a}") for a in range(ndim)]
    center = (0,) * ndim
    step_ratio = sympy.Symbol("step_ratio", positive=True)

    # Forward interior update, transcribed from scalar.c. In the central region
    # (pml_* == 1) w_sum is the plain FD Laplacian (scalar.c:530,542,553:
    #   "w_sum += DIFFZ2(WFC);" / DIFFY2 / DIFFX2), and (scalar.c:562-563)
    #   "wfp_t[i] = v_shot[i] * v_shot[i] * dt2 * w_sum + 2 * wfc_t[i] - wfp_t[i];"
    def u_next(j):
        w_sum = sum(_diff2(u_now, j, a, acc, rd2[a]) for a in range(ndim))
        return v[j] ** 2 * dt2 * w_sum + 2 * u_now[j] - u_prev[j]

    out_coords = list(itertools.product(range(-pad, pad + 1), repeat=ndim))
    loss = sum(lam[j] * u_next(j) for j in out_coords)

    # Backward wavefield update, transcribed from scalar.c. V2DT2_WFC(dz,dy,dx)
    # is v2dt2_shot[...] * WFC(...) (scalar.c:108-109), where v2dt2 = v**2 * dt2
    # and WFC here is the adjoint field (== lam). The interior w_sum is the FD
    # Laplacian of V2DT2_WFC (scalar.c:868,880,887: "w_sum += DIFFZ2(V2DT2_WFC);"
    # etc.), and (scalar.c:895) "wfp_t[i] = w_sum + 2 * wfc_t[i] - wfp_t[i];".
    v2dt2_lam = {c: v[c] ** 2 * dt2 * lam[c] for c in coords}
    w_sum_bwd = sum(_diff2(v2dt2_lam, center, a, acc, rd2[a]) for a in range(ndim))

    # The backward "wfp_t = w_sum + 2*wfc_t - wfp_t" leapfrog realises two
    # reverse-mode contributions: the terms carried by wfc_t (== lam) are the
    # transpose w.r.t. u_now, and the "-wfp_t" term is the transpose w.r.t.
    # u_prev. Check each against SymPy's exact derivative.
    d_du_now = w_sum_bwd + 2 * lam[center]
    assert sympy.simplify(sympy.diff(loss, u_now[center]) - d_du_now) == 0
    assert sympy.simplify(sympy.diff(loss, u_prev[center]) - (-lam[center])) == 0

    # v gradient. Forward stores (scalar.c:565) "w_store_1_t[i] = 2 * v_shot[i]
    # * dt2 * w_sum;" and backward accumulates (scalar.c:898-899)
    #   "grad_v_shot[i] += wfc_t[i] * w_store_1_t[i] * (DW_DTYPE)step_ratio;"
    # with wfc_t == lam. w_sum here is the *forward* Laplacian of u_now.
    w_sum_fwd = sum(_diff2(u_now, center, a, acc, rd2[a]) for a in range(ndim))
    w_store_1 = 2 * v[center] * dt2 * w_sum_fwd
    grad_v_increment = lam[center] * w_store_1 * step_ratio
    assert (
        sympy.simplify(grad_v_increment - step_ratio * sympy.diff(loss, v[center])) == 0
    )


def test_elastic_2d_model_gradients():
    """elastic.c 2-D isotropic: lambda/mu/buoyancy gradient accumulations are exact.

    The model parameters are diagonal in space, so the FD strain-rate operands
    (dvxdx, dvydy, ...) enter each cell's update only as coefficients and are
    treated as independent symbols. Re-derives each model gradient from the
    forward constitutive update and checks it against ``FUNC(backward)`` in
    elastic.c, including the ``step_ratio`` scaling.
    """
    dt = sympy.Symbol("dt")
    step_ratio = sympy.Symbol("step_ratio", positive=True)
    # Model parameters (mu is the on-node modulus; mu_yx is the separate
    # half-cell-averaged array used only by sigmaxy -- elastic.c:216,229).
    lamb, mu, mu_yx = sympy.symbols("lamb mu mu_yx")
    buoyancy_y, buoyancy_x = sympy.symbols("buoyancy_y buoyancy_x")
    # Strain rates and velocity divergences (opaque FD operands).
    dvxdx, dvydy, dvxdy, dvydx = sympy.symbols("dvxdx dvydy dvxdy dvydx")
    w_sum_vy, w_sum_vx = sympy.symbols("w_sum_vy w_sum_vx")
    # Old field values.
    sxx0, syy0, sxy0, vy0, vx0 = sympy.symbols("sxx0 syy0 sxy0 vy0 vx0")
    # Adjoint (reverse-time) fields that seed the gradients.
    bar_sxx, bar_syy, bar_sxy = sympy.symbols("bar_sxx bar_syy bar_sxy")
    bar_vy, bar_vx = sympy.symbols("bar_vy bar_vx")

    # Forward velocity update, transcribed from elastic.c:1006,1063
    #   "vy_shot[i] += buoyancy_y_shot[i] * dt * w_sum;"
    #   "vx_shot[i] += buoyancy_x_shot[i] * dt * w_sum;"
    vy_new = vy0 + buoyancy_y * dt * w_sum_vy
    vx_new = vx0 + buoyancy_x * dt * w_sum_vx
    # Forward stress update, transcribed from elastic.c. w_sum for sigmaii is
    # lamb*(dvydy + dvxdx) (elastic.c:1160,1174), and (elastic.c:1183,1185)
    #   "sigmayy_shot[i] += dt * (w_sum + 2 * mu_shot[i] * dvydy);"
    #   "sigmaxx_shot[i] += dt * (w_sum + 2 * mu_shot[i] * dvxdx);"
    strain = dvxdx + dvydy
    syy_new = syy0 + dt * (lamb * strain + 2 * mu * dvydy)
    sxx_new = sxx0 + dt * (lamb * strain + 2 * mu * dvxdx)
    # sigmaxy (elastic.c:1229) "sigmaxy_shot[i] += dt * mu_yx_shot[i] * w_sum;"
    # with w_sum = dvxdy + dvydx (elastic.c:1215,1224).
    sxy_new = sxy0 + dt * mu_yx * (dvxdy + dvydx)

    loss = (
        bar_sxx * sxx_new
        + bar_syy * syy_new
        + bar_sxy * sxy_new
        + bar_vy * vy_new
        + bar_vx * vx_new
    )

    # Forward stores (elastic.c:1162,1176,1227,1004,1061):
    #   dvydy_store = dt*dvydy ; dvxdx_store = dt*dvxdx
    #   dvydx_plus_dvxdy_store = dt*(dvxdy+dvydx)
    #   dvydbuoyancy_store = dt*w_sum_vy ; dvxdbuoyancy_store = dt*w_sum_vx
    dvydy_store = dt * dvydy
    dvxdx_store = dt * dvxdx
    dvydx_plus_dvxdy_store = dt * (dvxdy + dvydx)
    dvydbuoyancy_store = dt * w_sum_vy
    dvxdbuoyancy_store = dt * w_sum_vx

    # Backward model-gradient accumulations, transcribed from elastic.c:
    #  (2116-2118) grad_lamb += (sigmaxx+sigmayy)*(dvxdx_store+dvydy_store)*step_ratio
    #  (2132-2135) grad_mu   += 2*(sigmaxx*dvxdx_store+sigmayy*dvydy_store)*step_ratio
    #  (2210-2212) grad_mu_yx+= sigmaxy*dvydx_plus_dvxdy_store*step_ratio
    #  (1972-1974) grad_buoyancy_y += vy*dvydbuoyancy_store*step_ratio
    #  (2031-2033) grad_buoyancy_x += vx*dvxdbuoyancy_store*step_ratio
    grad_lamb = (bar_sxx + bar_syy) * (dvxdx_store + dvydy_store) * step_ratio
    grad_mu = 2 * (bar_sxx * dvxdx_store + bar_syy * dvydy_store) * step_ratio
    grad_mu_yx = bar_sxy * dvydx_plus_dvxdy_store * step_ratio
    grad_buoyancy_y = bar_vy * dvydbuoyancy_store * step_ratio
    grad_buoyancy_x = bar_vx * dvxdbuoyancy_store * step_ratio

    checks = [
        (grad_lamb, lamb),
        (grad_mu, mu),
        (grad_mu_yx, mu_yx),
        (grad_buoyancy_y, buoyancy_y),
        (grad_buoyancy_x, buoyancy_x),
    ]
    for kernel_increment, param in checks:
        exact = step_ratio * sympy.diff(loss, param)
        assert sympy.simplify(kernel_increment - exact) == 0, param


def test_elastic_born_2d_scattered_coupling():
    """elastic_born.c 2-D: scattering-model gradients (background-store) are exact.

    Re-derives the scattered-field forward coupling and its scattered-only
    adjoint (``FUNC(backward_sc)``), verifying each scattering-model gradient
    correlates the scattered adjoint field with the *background* strain-rate
    store (not the scattered one) and carries the ``step_ratio`` scaling.
    """
    dt = sympy.Symbol("dt")
    step_ratio = sympy.Symbol("step_ratio", positive=True)
    # Background model, on-node and half-cell-averaged shear modulus.
    lamb, mu, mu_yx = sympy.symbols("lamb mu mu_yx")
    buoyancy_y = sympy.Symbol("buoyancy_y")
    # Scattering perturbations (the parameters we differentiate w.r.t.).
    dlamb, dmu, dmu_yx, dbuoyancy_y = sympy.symbols("dlamb dmu dmu_yx dbuoyancy_y")
    # Background strain rates / divergences (what the store holds).
    dvxdx, dvydy = sympy.symbols("dvxdx dvydy")
    w_sum_vy_bg, w_sum_xy_bg = sympy.symbols("w_sum_vy_bg w_sum_xy_bg")
    # Scattered strain rates / divergences.
    dvxdx_sc, dvydy_sc = sympy.symbols("dvxdx_sc dvydy_sc")
    w_sum_vy_sc, w_sum_xy_sc = sympy.symbols("w_sum_vy_sc w_sum_xy_sc")
    # Old scattered field values and their reverse-time adjoints.
    vy_sc0, sxx_sc0, syy_sc0, sxy_sc0 = sympy.symbols("vy_sc0 sxx_sc0 syy_sc0 sxy_sc0")
    bar_vy_sc, bar_sxx_sc, bar_syy_sc, bar_sxy_sc = sympy.symbols(
        "bar_vy_sc bar_sxx_sc bar_syy_sc bar_sxy_sc"
    )

    strain = dvxdx + dvydy  # background volumetric strain rate
    strain_sc = dvxdx_sc + dvydy_sc  # scattered volumetric strain rate

    # Scattered forward updates, transcribed from elastic_born.c. The second
    # term of each is the scattered-field SOURCE INJECTION: a perturbation model
    # times the BACKGROUND field FD term.
    #  vy_sc (1230-1231):
    #    "vy_sc_shot[i] += buoyancy_y_shot[i] * dt * w_sum_sc
    #                      + dbuoyancy_y_shot[i] * dt * w_sum;"  (w_sum == background)
    vy_sc_new = vy_sc0 + buoyancy_y * dt * w_sum_vy_sc + dbuoyancy_y * dt * w_sum_vy_bg
    #  sigmayy_sc / sigmaxx_sc (1466-1468, 1471-1473):
    #    "... += dt*(lamb*strain_sc + 2*mu*dvydy_sc)
    #            + dt*(dlamb*strain + 2*dmu*dvydy);"  (strain, dvydy == background)
    syy_sc_new = (
        syy_sc0
        + dt * (lamb * strain_sc + 2 * mu * dvydy_sc)
        + dt * (dlamb * strain + 2 * dmu * dvydy)
    )
    sxx_sc_new = (
        sxx_sc0
        + dt * (lamb * strain_sc + 2 * mu * dvxdx_sc)
        + dt * (dlamb * strain + 2 * dmu * dvxdx)
    )
    #  sigmaxy_sc (1523-1524):
    #    "sigmaxy_sc_shot[i] += dt*mu_yx_shot[i]*w_sum_sc + dt*dmu_yx_shot[i]*w_sum;"
    sxy_sc_new = sxy_sc0 + dt * mu_yx * w_sum_xy_sc + dt * dmu_yx * w_sum_xy_bg

    loss_sc = (
        bar_vy_sc * vy_sc_new
        + bar_sxx_sc * sxx_sc_new
        + bar_syy_sc * syy_sc_new
        + bar_sxy_sc * sxy_sc_new
    )

    # Background stores (gated on store_bg_t), transcribed from elastic_born.c:
    #  (1227) dvydbuoyancy_store = dt * w_sum          (background velocity div)
    #  (1444,1446) dvydy_store = dt*dvydy ; dvxdx_store = dt*dvxdx
    #  (1519) dvydx_plus_dvxdy_store = dt * w_sum       (background shear div)
    dvydbuoyancy_store = dt * w_sum_vy_bg
    dvydy_store = dt * dvydy
    dvxdx_store = dt * dvxdx
    dvydx_plus_dvxdy_store = dt * w_sum_xy_bg

    # Scattered-only adjoint (backward_sc) accumulations, transcribed from
    # elastic_born.c. The adjoint fields (sigma*_shot, v*_shot) here are the
    # SCATTERED adjoint wavefields.
    #  (2618-2619) grad_dbuoyancy_y += vy * dvydbuoyancy_store * step_ratio
    #  (2732-2734) grad_dlamb += (sigmaxx+sigmayy)*(dvxdx_store+dvydy_store)*step_ratio
    #  (2747-2749) grad_dmu   += 2*(sigmaxx*dvxdx_store+sigmayy*dvydy_store)*step_ratio
    #  (2815-2816) grad_dmu_yx+= sigmaxy*dvydx_plus_dvxdy_store*step_ratio
    grad_dbuoyancy_y = bar_vy_sc * dvydbuoyancy_store * step_ratio
    grad_dlamb = (bar_sxx_sc + bar_syy_sc) * (dvxdx_store + dvydy_store) * step_ratio
    grad_dmu = 2 * (bar_sxx_sc * dvxdx_store + bar_syy_sc * dvydy_store) * step_ratio
    grad_dmu_yx = bar_sxy_sc * dvydx_plus_dvxdy_store * step_ratio

    checks = [
        (grad_dbuoyancy_y, dbuoyancy_y),
        (grad_dlamb, dlamb),
        (grad_dmu, dmu),
        (grad_dmu_yx, dmu_yx),
    ]
    for kernel_increment, param in checks:
        exact = step_ratio * sympy.diff(loss_sc, param)
        assert sympy.simplify(kernel_increment - exact) == 0, param

    # Negative control (the wrong-block-swap bug class). The d(buoyancy) gradient must
    # correlate with the BACKGROUND divergence store dt*w_sum_vy_bg. Had the
    # kernel used the SCATTERED store dt*w_sum_vy_sc instead, the check would no
    # longer hold -- confirm the audit actually discriminates the two blocks.
    wrong_increment = bar_vy_sc * (dt * w_sum_vy_sc) * step_ratio
    assert (
        sympy.simplify(wrong_increment - step_ratio * sympy.diff(loss_sc, dbuoyancy_y))
        != 0
    )

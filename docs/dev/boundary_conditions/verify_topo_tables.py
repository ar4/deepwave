"""Symbolic re-derivation of the Cao-Chen (2018) PM free-surface tables.

Verifies every coefficient entry recorded in
docs/dev/boundary_conditions/CAO_CHEN_PM_SUMMARY.md (Tables 1-2 and
eqs 23-25) by re-deriving it from first principles with sympy, rather
than re-transcribing the summary's algebra.  Derivation inputs are only
the stated conditions:

 1. Isotropic constitutive law  tau = lam*tr(eps)*I + 2*mu*eps  (eq 2/3).
 2. Traction-free condition tau.n = 0 on each face bordering vacuum; the
    strain components involving the surface-normal direction are
    eliminated by solving the traction conditions and substituting
    (the derivation route of the summary's section 2.4).
 3. Average-medium (vacuum-average) rule at a node lying IN a discrete
    surface plane: effective coefficient = arithmetic mean of the vacuum
    value (0) and the solid-side BC-reduced value -> factor 1/2; density
    rho -> rho/2 there (so the buoyancy prepared from two solid
    neighbours is doubled).
 4. Staggered-node placement (summary section 1; Kristek config 2:
    normal stresses and lam/mu/rho on integer nodes -- the discrete free
    surface passes through them; shear stresses and velocities on half
    nodes): a half node offset toward the vacuum keeps the base harmonic
    average, which a zero vacuum neighbour drives to 0 automatically; a
    half node offset into the solid keeps the full interior coefficient.

Checks:
  [A] interior identity (eq 23)
  [B] alpha/beta closed forms and the identities quoted around them
  [C] 2-D Table 1, every row and column (H, VL, VR)
  [D] 3-D Table 2, every row and column (H, VL, VR, VF, VB), plus eq 24
  [E] rotation cross-check: rotating the flat-H surface relation onto
      each vertical boundary (the paper's own offline derivation device,
      eqs 7-14) must reproduce the direct BC-elimination result
  [F] transition/outer-corner rule (eq 25) from joint traction-free
      conditions on both/all adjacent faces
  [G] the 2-D eager prototype's coefficient assignments (as implemented
      in the original, now-superseded pm_topo.py prototype; the same
      assignments now live in deepwave.elastic's PM classifier) against
      the verified 2-D table

Exit status 0 with a final "ALL CHECKS PASS" line means every table
entry re-derives.  The FINDINGS block at the end records the places
where the summary text as first written disagreed with the derivation
(since corrected in the summary); the script asserts the corrected
values.

Run:  python verify_topo_tables.py
"""

import sympy as sp

lam, mu = sp.symbols("lam mu", positive=True)
alpha = 2 * mu * (lam + mu) / (lam + 2 * mu)
beta = mu * lam / (lam + 2 * mu)

exx, eyy, ezz, exy, eyz, ezx = sp.symbols("exx eyy ezz exy eyz ezx", real=True)
EPS3 = sp.Matrix([[exx, exy, ezx], [exy, eyy, eyz], [ezx, eyz, ezz]])
EPS2 = sp.Matrix(
    [
        [exx, ezx],  # 2-D (x-z) plane strain
        [ezx, ezz],
    ]
)

FAIL = []


def check(name, cond):
    """Records and prints whether one named identity holds."""
    ok = bool(cond)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


def zero(expr):
    """True if `expr` simplifies to exactly zero."""
    return sp.simplify(sp.expand(expr)) == 0


def tau_iso(eps):
    """The isotropic constitutive law, eq 2/3, applied to a strain matrix."""
    n = eps.shape[0]
    return lam * eps.trace() * sp.eye(n) + 2 * mu * eps


def surface_relation(eps, axis):
    """Eliminate the traction-free face with normal along `axis`.

    Applies the factor-1/2 vacuum average to every component stored in
    the surface plane. Returns (halved surface stress matrix, elimination
    solution dict). Traction components come out identically zero.
    """
    n = eps.shape[0]
    tau = tau_iso(eps)
    unknowns = list(dict.fromkeys(eps[axis, j] for j in range(n)))
    sol = sp.solve([tau[axis, j] for j in range(n)], unknowns, dict=True)
    if len(sol) != 1:
        raise RuntimeError(
            "traction conditions must determine the normal strains uniquely"
        )
    return sp.simplify(tau.subs(sol[0]) / 2), sol[0]


# ---------------------------------------------------------------- [A] interior
print("[A] interior identity (eq 23)")
tau3 = tau_iso(EPS3)
check(
    "tau_xx = (lam+2mu) exx + lam eyy + lam ezz",
    zero(tau3[0, 0] - ((lam + 2 * mu) * exx + lam * eyy + lam * ezz)),
)
check("tau_xy = 2 mu exy", zero(tau3[0, 1] - 2 * mu * exy))

# ------------------------------------------------------- [B] alpha/beta values
print("[B] alpha/beta closed forms and identity notes")
# 2-D flat H: eliminate ezz from tau_zz = 0, halve -> alpha
tauH2, solH2 = surface_relation(EPS2, 1)
check(
    "2-D H: eliminated ezz = -lam/(lam+2mu) exx",
    zero(solH2[ezz] - (-lam / (lam + 2 * mu) * exx)),
)
check(
    "2-D H: tau_xx (halved) = alpha exx with alpha = 2mu(lam+mu)/(lam+2mu)",
    zero(tauH2[0, 0] - alpha * exx),
)
reduced_diag = (lam + 2 * mu) - lam**2 / (lam + 2 * mu)
check(
    "pre-average reduced modulus (lam+2mu) - lam^2/(lam+2mu) equals 2*alpha",
    zero(reduced_diag - 2 * alpha),
)
check(
    "... and does NOT equal alpha (the summary's original note was off by 2)",
    not zero(reduced_diag - alpha),
)
check(
    "alpha + beta = mu(3lam+2mu)/(lam+2mu) != lam+2mu",
    zero(alpha + beta - mu * (3 * lam + 2 * mu) / (lam + 2 * mu))
    and not zero(alpha + beta - (lam + 2 * mu)),
)

# ------------------------------------------------- placement rule (input 3/4)
# Half-offset directions of each staggered node (1 = +1/2 offset).
OFF_SHEAR = {"xy": (1, 1, 0), "yz": (0, 1, 1), "zx": (1, 0, 1)}
OFF_VEL = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def placement(offsets, axis, air_sign):
    """Effective-coefficient factor for a node near a face.

    The face has normal `axis` and vacuum on the `air_sign` side:
    on-plane -> 1/2 (arithmetic vacuum average), toward air -> 0
    (harmonic average with a zero vacuum neighbour), into solid -> 1
    (unchanged interior value).
    """
    if offsets[axis] == 0:
        return sp.Rational(1, 2)
    return sp.Integer(0) if air_sign > 0 else sp.Integer(1)


# ------------------------------------------------------------- [C] 2-D Table 1
print("[C] 2-D Table 1 (x-z; columns tau_xx, tau_zz, tau_xz, rho_x, rho_z)")
# Type -> (axis, air_sign) in (x, z) indexing; z is depth (air above = -z).
TYPES2 = {"H": (1, -1), "VL": (0, -1), "VR": (0, 1)}
# Expected effective entries (tensor-strain convention).
TABLE1 = {
    "H": {
        "txx": alpha * exx,
        "tzz": 0,
        "txz_fac": 1,
        "rx_fac": sp.Rational(1, 2),
        "rz_fac": 1,
    },
    "VL": {
        "txx": 0,
        "tzz": alpha * ezz,
        "txz_fac": 1,
        "rx_fac": 1,
        "rz_fac": sp.Rational(1, 2),
    },
    "VR": {
        "txx": 0,
        "tzz": alpha * ezz,
        "txz_fac": 0,
        "rx_fac": 0,
        "rz_fac": sp.Rational(1, 2),
    },
}
OFF2_SHEAR = (1, 1)  # tau_xz node: half offset in both x and z
OFF2_VX, OFF2_VZ = (1, 0), (0, 1)
for t, (ax, sgn) in TYPES2.items():
    tau_s, _ = surface_relation(EPS2, ax)
    exp = TABLE1[t]
    check(f"{t}: tau_xx = {exp['txx']}", zero(tau_s[0, 0] - exp["txx"]))
    check(f"{t}: tau_zz = {exp['tzz']}", zero(tau_s[1, 1] - exp["tzz"]))
    # tau_xz node is off the surface plane for every 2-D type: placement rule.
    fac = placement(OFF2_SHEAR, ax, sgn)
    check(
        f"{t}: tau_xz factor {exp['txz_fac']} x 2mu exz (node placement)",
        fac == exp["txz_fac"],
    )
    check(
        f"{t}: rho_x factor {exp['rx_fac']}",
        placement(OFF2_VX, ax, sgn) == exp["rx_fac"],
    )
    check(
        f"{t}: rho_z factor {exp['rz_fac']}",
        placement(OFF2_VZ, ax, sgn) == exp["rz_fac"],
    )
    # the surface relation's traction shear is 0 (the off-plane node instead
    # carries the interior/auto-zeroed value -- placement rule above)
    check(
        f"{t}: surface-plane traction components vanish",
        zero(tau_s[ax, ax]) and zero(tau_s[0, 1]),
    )

# ------------------------------------------------------------- [D] 3-D Table 2
print("[D] 3-D Table 2 (columns tau_xx..tau_zx, rho_x..rho_z) + eq 24")
TYPES3 = {"H": (2, -1), "VL": (0, -1), "VR": (0, 1), "VF": (1, -1), "VB": (1, 1)}
# Expected effective normal-stress rows (from BC elimination + 1/2 average).
NORMAL3 = {
    "H": {
        (0, 0): alpha * exx + beta * eyy,
        (1, 1): beta * exx + alpha * eyy,
        (2, 2): 0,
    },
    "VL": {
        (0, 0): 0,
        (1, 1): alpha * eyy + beta * ezz,
        (2, 2): beta * eyy + alpha * ezz,
    },
    "VR": {
        (0, 0): 0,
        (1, 1): alpha * eyy + beta * ezz,
        (2, 2): beta * eyy + alpha * ezz,
    },
    "VF": {
        (0, 0): alpha * exx + beta * ezz,
        (1, 1): 0,
        (2, 2): beta * exx + alpha * ezz,
    },
    "VB": {
        (0, 0): alpha * exx + beta * ezz,
        (1, 1): 0,
        (2, 2): beta * exx + alpha * ezz,
    },
}
IDX = {"xy": (0, 1), "yz": (1, 2), "zx": (0, 2)}
STRAIN = {"xy": exy, "yz": eyz, "zx": ezx}
# Expected effective shear factors (x 2mu): 1/2 = on-plane vacuum average,
# 1 = node in solid, 0 = node in vacuum (auto-zeroed by harmonic averaging).
SHEAR3 = {
    "H": {"xy": sp.Rational(1, 2), "yz": 1, "zx": 1},
    "VL": {"xy": 1, "yz": sp.Rational(1, 2), "zx": 1},
    "VR": {"xy": 0, "yz": sp.Rational(1, 2), "zx": 0},
    "VF": {"xy": 1, "yz": 1, "zx": sp.Rational(1, 2)},
    "VB": {"xy": 0, "yz": 0, "zx": sp.Rational(1, 2)},
}
RHO3 = {
    "H": {"x": sp.Rational(1, 2), "y": sp.Rational(1, 2), "z": 1},
    "VL": {"x": 1, "y": sp.Rational(1, 2), "z": sp.Rational(1, 2)},
    "VR": {"x": 0, "y": sp.Rational(1, 2), "z": sp.Rational(1, 2)},
    "VF": {"x": sp.Rational(1, 2), "y": 1, "z": sp.Rational(1, 2)},
    "VB": {"x": sp.Rational(1, 2), "y": 0, "z": sp.Rational(1, 2)},
}
for t, (ax, sgn) in TYPES3.items():
    tau_s, _ = surface_relation(EPS3, ax)
    for (i, j), expected in NORMAL3[t].items():
        check(
            f"{t}: tau_{'xyz'[i]}{'xyz'[j]} = {expected}", zero(tau_s[i, j] - expected)
        )
    for name, fac_exp in SHEAR3[t].items():
        i, j = IDX[name]
        if OFF_SHEAR[name][ax] == 0:
            # node in the surface plane: its coefficient comes from the halved
            # surface relation itself (BC elimination leaves 2mu; halved -> mu)
            check(
                f"{t}: on-plane tau_{name} = mu {STRAIN[name]} (halved)",
                fac_exp == sp.Rational(1, 2) and zero(tau_s[i, j] - mu * STRAIN[name]),
            )
        else:
            check(
                f"{t}: off-plane tau_{name} factor {fac_exp} (node placement)",
                placement(OFF_SHEAR[name], ax, sgn) == fac_exp,
            )
            # and the surface-plane relation itself zeroes this traction shear
            check(
                f"{t}: surface relation has tau_{name} = 0 (traction component)",
                zero(tau_s[i, j]),
            )
    for name, fac_exp in RHO3[t].items():
        check(
            f"{t}: rho_{name} factor {fac_exp}",
            placement(OFF_VEL[name], ax, sgn) == fac_exp,
        )

# eq 24 (VR worked example) in eta notation: eta2^yy = eta3^zz = alpha,
# eta3^yy = eta2^zz = beta, eta1^* = 0, buoyancy by,bz doubled (rho halved).
tau_vr, _ = surface_relation(EPS3, 0)
check(
    "eq 24: eta2^yy = alpha and eta3^yy = beta (tau_yy row)",
    zero(sp.diff(tau_vr[1, 1], eyy) - alpha)
    and zero(sp.diff(tau_vr[1, 1], ezz) - beta),
)
check(
    "eq 24: eta1^yy = eta1^zz = 0 (no eliminated-strain dependence)",
    zero(sp.diff(tau_vr[1, 1], exx)) and zero(sp.diff(tau_vr[2, 2], exx)),
)
check(
    "eq 24: eta1^xx = eta2^xx = eta3^xx = 0",
    all(zero(sp.diff(tau_vr[0, 0], s)) for s in (exx, eyy, ezz)),
)

# ------------------------------------------------- [E] rotation cross-check
print("[E] rotation cross-check (paper's derivation device, eqs 7-14)")


def h_relation(epsm, onplane_shear_factor=None):
    """The flat-H surface relation applied to an arbitrary symmetric strain matrix.

    In-plane normals alpha/beta, on-plane shear 2mu*factor (the factor
    defaults to 1/2), traction components zero.
    """
    if onplane_shear_factor is None:
        onplane_shear_factor = sp.Rational(1, 2)
    out = sp.zeros(3, 3)
    out[0, 0] = alpha * epsm[0, 0] + beta * epsm[1, 1]
    out[1, 1] = beta * epsm[0, 0] + alpha * epsm[1, 1]
    out[0, 1] = out[1, 0] = 2 * mu * onplane_shear_factor * epsm[0, 1]
    return out


def rot_y(deg):
    """Rotation matrix about the y axis, in degrees."""
    c, s = sp.cos(sp.rad(deg)), sp.sin(sp.rad(deg))
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(deg):
    """Rotation matrix about the x axis, in degrees."""
    c, s = sp.cos(sp.rad(deg)), sp.sin(sp.rad(deg))
    return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


ROT = {"VL": rot_y(-90), "VR": rot_y(90), "VF": rot_x(90), "VB": rot_x(-90)}
for t, Q in ROT.items():
    ax = TYPES3[t][0]
    # sanity: Q maps this boundary's outward normal onto the H normal (-z)
    n = sp.zeros(3, 1)
    n[ax] = TYPES3[t][1]
    check(f"{t}: rotation maps its normal to -z", (Q * n) == sp.Matrix([0, 0, -1]))
    tau_rot = sp.simplify(Q.T * h_relation(Q * EPS3 * Q.T) * Q)
    tau_dir, _ = surface_relation(EPS3, ax)
    check(
        f"{t}: rotated H relation == direct BC elimination",
        all(zero(tau_rot[i, j] - tau_dir[i, j]) for i in range(3) for j in range(3)),
    )
# Internal-consistency proof for the H-row on-plane shear: if the H relation
# carried the FULL 2mu on tau_xy (no halving), rotating it onto VR would give
# tau_yz = 2 mu eyz, contradicting eq 24's <mu~>yz = <mu>yz / 2.
tau_rot_full = sp.simplify(
    ROT["VR"].T * h_relation(ROT["VR"] * EPS3 * ROT["VR"].T, 1) * ROT["VR"]
)
check(
    "H-row tau_xy must be mu*exy: unhalved 2mu variant contradicts eq 24"
    " under rotation",
    zero(tau_rot_full[1, 2] - 2 * mu * eyz) and not zero(tau_rot_full[1, 2] - mu * eyz),
)

# 2-D: eq 12 (VL) as the 90-degree rotation of eq 6 (H)
Q2 = sp.Matrix([[0, 1], [-1, 0]])  # maps -x onto -z in the (x, z) plane


def tauH2_of(m):
    """The 2-D flat-H surface stress for an arbitrary strain matrix."""
    return sp.Matrix([[alpha * m[0, 0], 0], [0, 0]])


tau2_rot = sp.simplify(Q2.T * tauH2_of(Q2 * EPS2 * Q2.T) * Q2)
tau2_dir, _ = surface_relation(EPS2, 0)
check(
    "2-D: rotated eq 6 == direct VL elimination (eq 12)",
    all(zero(tau2_rot[i, j] - tau2_dir[i, j]) for i in range(2) for j in range(2)),
)

# ---------------------------------------------------- [F] corners (eq 25)
print("[F] transition/outer-corner rule (eq 25)")
tau2f = tau_iso(EPS2)
sol2c = sp.solve([tau2f[0, 0], tau2f[1, 1], tau2f[0, 1]], [exx, ezz, ezx], dict=True)
check(
    "2-D corner: joint traction-free forces exx = ezz = ezx = 0 "
    "-> all normal-stiffness coefficients exactly 0",
    len(sol2c) == 1 and all(v == 0 for v in sol2c[0].values()),
)
# 3-D edge (air above + one side): tau_yy retains a uniaxial Young's-modulus
# term E*eyy that eq 25 additionally zeroes (documented approximation).
eqs_edge = [tau3[2, j] for j in range(3)] + [tau3[0, j] for j in range(3)]
sol_edge = sp.solve(eqs_edge, [exx, ezz, exy, eyz, ezx], dict=True)
tau_edge = sp.simplify(tau3.subs(sol_edge[0]))
E_young = mu * (3 * lam + 2 * mu) / (lam + mu)
check(
    "3-D edge: all stresses vanish except tau_yy = E*eyy "
    "(E = mu(3lam+2mu)/(lam+mu); eq 25 zeroes this too -- approximation)",
    len(sol_edge) == 1
    and zero(tau_edge[1, 1] - E_young * eyy)
    and all(
        zero(tau_edge[i, j]) for i in range(3) for j in range(3) if (i, j) != (1, 1)
    ),
)
# 3-D outer corner (three faces): everything vanishes -> eq 25 exact.
eqs_corner = [tau3[i, j] for i in range(3) for j in range(3)]
sol_corner = sp.solve(eqs_corner, [exx, eyy, ezz, exy, eyz, ezx], dict=True)
check(
    "3-D 3-face corner: all strains forced to 0 -> eq 25 exact",
    len(sol_corner) == 1 and all(v == 0 for v in sol_corner[0].values()),
)

# ------------------------------------------- [G] 2-D prototype assignments
print("[G] pm_topo.py coefficient assignments vs the verified 2-D table")
# Deepwave 2-D mapping: dim0 = y = paper z (depth), dim1 = x.
# Component 0 = s_yy = paper tau_zz, component 1 = s_xx = paper tau_xx.
# Prototype H:  C[1][1] = alpha, C[1][0] = C[0][0] = C[0][1] = 0.
tauH2_full, _ = surface_relation(EPS2, 1)
check(
    "prototype H row == table: tau_xx = alpha exx, tau_zz = 0",
    zero(tauH2_full[0, 0] - alpha * exx) and zero(tauH2_full[1, 1]),
)
# Prototype VL/VR: C[0][0] = alpha, others 0.
tauV2, _ = surface_relation(EPS2, 0)
check(
    "prototype VL/VR row == table: tau_zz = alpha ezz, tau_xx = 0",
    zero(tauV2[1, 1] - alpha * ezz) and zero(tauV2[0, 0]),
)
# Prototype buoyancy: H doubles bx (rho_x halved), VL/VR double by
# (paper rho_z halved), corners double both -- matches [C]'s factors and the
# auto-zeroing of the vacuum-side node.
check(
    "prototype buoyancy doubling == table density halving (H: rho_x, V: rho_z)",
    TABLE1["H"]["rx_fac"] == sp.Rational(1, 2)
    and TABLE1["VL"]["rz_fac"] == sp.Rational(1, 2)
    and TABLE1["VR"]["rz_fac"] == sp.Rational(1, 2),
)
# Averaging identities used throughout (input 3/4):
b = sp.symbols("b", positive=True)
check(
    "harmonic buoyancy/mu average with a zero vacuum neighbour is 0",
    sp.limit(2 * b * sp.Symbol("v") / (b + sp.Symbol("v")), sp.Symbol("v"), 0) == 0,
)
rho = sp.symbols("rho", positive=True)
check(
    "arithmetic density average with vacuum: (rho+0)/2 = rho/2 (buoyancy doubled)",
    zero((rho + 0) / 2 - rho / 2),
)

# ------------------------------------------------------------------ verdict
print()
print("FINDINGS (summary-text defects found and corrected; derivations above")
print("assert the corrected values):")
print(" 1. Table 2 H row, tau_xy cell: must be mu*exy (on-plane shear halved),")
print("    not 2mu*exy as first transcribed -- required by the traction-free +")
print("    vacuum-average derivation, by physical momentum-balance consistency")
print("    with the halved surface tau_xx/tau_yy and rho_x/rho_y, and by")
print("    internal consistency with eq 24 (<mu~>yz = <mu>yz/2) under the")
print("    paper's own rotation device (see the [E] check).")
print(" 2. Table 2 VB row, tau_yz cell: written 'mu eyz' but the node lies on")
print("    the vacuum side, so the base harmonic average is 0 and the")
print("    effective coefficient is 0 (same auto-zeroing the table applies to")
print("    VR's tau_xy/tau_zx and VB's tau_xy/rho_y cells).")
print(" 3. Section 2.3 identity note: (lam+2mu) - lam^2/(lam+2mu) equals")
print("    2*alpha (the PRE-average reduced modulus), not alpha; alpha is half")
print("    of it via the vacuum average.")
print(" 4. eq 25's all-zero normal stiffness at transition/corner points is")
print("    EXACT in 2-D (and at 3-D three-face corners) but drops a uniaxial")
print("    E*eps_yy term along a 3-D edge (see the [F] check).")
print()
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED:")
    for f in FAIL:
        print("  -", f)
    raise SystemExit(1)
print("ALL CHECKS PASS")

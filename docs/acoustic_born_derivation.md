# Derivation: variable-density acoustic Born (linearised) propagator

*Developer/reference documentation: the discrete derivation behind
`deepwave/acoustic_born` and its adjoint. See {doc}`acoustic_born` for the
user-facing propagator page.*

This document derives the linearised (Born) variable-density acoustic system, its
adjoint, and the second-order (Hessian-vector product) structure, mirroring the
existing `scalar` / `scalar_born` machinery. It also fixes the planned public API,
kernel filenames, and validation gates so the follow-up implementation task can
proceed without re-deriving anything.

Throughout, `v` (or `c`) is wavespeed, `ρ` density, `K = ρ v²` the bulk modulus,
`B = 1/ρ` the buoyancy, `p` pressure, `v = (v_z, v_y, v_x)` the particle-velocity
vector. A background quantity carries a bar (`p̄`, `v̄`, `K`, `B`); a first-order
perturbation carries a `δ` (`δp`, `δv`, `δK`, `δB`).

---

## 1. The forward system (what `acoustic` solves)

### 1.1 Continuous equations

The variable-density first-order (velocity–pressure) acoustic system is

```
∂v/∂t = −B ∇p           (+ B f)      momentum / velocity equation
∂p/∂t = −K ∇·v          (+ K s)      constitutive / pressure equation
```

with `B = 1/ρ`, `K = ρ v²`, `s` a volume-injection-rate (pressure) source, and `f`
a body-force (velocity) source. This is exactly the system in the `acoustic.py`
module docstring (`ρ ∂v/∂t = −∇p`, `(1/K) ∂p/∂t = −∇·v + s`), divided through by
`ρ` and `1/K` respectively.

### 1.2 Discrete staggered leapfrog (the reference backend)

On the staggered grid, `p` and `K` live at integer nodes; `v_d` and `B_d` live half
a cell from `p` in dimension `d`. Writing `G_d` for the half-step gradient
(`staggered_grid.diff1h`, integer→half node) and `D_d` for the integer-step
divergence operator (`staggered_grid.diff1`, half→integer node), one internal step
is (from `acoustic.py::update_velocities`/`update_pressure` and `acoustic.c`
`FUNC(forward)`):

```
velocity:  ψ_d      ← a_d^h ψ_d + b_d^h (G_d p̄)              (CPML memory, per dim)
           term_d   = G_d p̄ + ψ_d                             (checkpointed: b*_grad_store_1)
           v̄_d      ← v̄_d − dt · B_d · term_d

pressure:  φ_d      ← a_d φ_d + b_d (D_d v̄_d)                 (CPML memory, per dim)
           div      = Σ_d ( D_d v̄_d + φ_d )                   (checkpointed: k_grad_store_1)
           p̄        ← p̄ − dt · K · div
```

Sources: a pressure source is added to `p̄` scaled by `dt·K` at its cell; a velocity
source is added to `v̄_d` scaled by `dt·B_d` (the `source_amplitudes_out[i] =
src_amp * gather_model_at_sources(model) * dt` scaling in `acoustic.py`, with
`model = K` for pressure and `model = B_d` for the velocity components).

**Key discrete fact (used by the adjoint):** on the staggered grid the divergence
is the negative transpose of the gradient, `D_d = −G_dᵀ` (the exact-adjoint property,
verified symbolically by `tests/test_adjoint_symbolic.py`). This is why the acoustic adjoint kernel is a clean
transpose (§4).

---

## 2. Born linearisation

Perturb the constitutive fields `K → K + δK`, `B_d → B_d + δB_d`, and split the
fields `p = p̄ + δp`, `v = v̄ + δv`. Substitute into §1.1 and keep first order.

### 2.1 Velocity equation

```
∂(v̄+δv)/∂t = −(B+δB) ∇(p̄+δp)
```

Zeroth order (background):  `∂v̄/∂t = −B ∇p̄`.
First order:

```
∂δv/∂t = −B ∇δp  −  δB ∇p̄
```

### 2.2 Pressure equation

```
∂(p̄+δp)/∂t = −(K+δK) ∇·(v̄+δv)
```

Zeroth order:  `∂p̄/∂t = −K ∇·v̄  (+ K s)`.
First order:

```
∂δp/∂t = −K ∇·δv  −  δK ∇·v̄
```

### 2.3 The scattered field obeys the same operator, with virtual sources

Collecting §2.1–§2.2: **the scattered field `(δp, δv)` satisfies the identical
variable-density acoustic system with the background parameters `(K, B)`, driven by
two virtual sources:**

```
velocity equation:   virtual force   = − δB_d ∇_d p̄     (one per velocity component, at the v_d node)
pressure equation:   virtual injection = − δK ∇·v̄        (at the p node)
```

This is exactly the structure the task specifies: `δK ∇·v` drives the pressure
equation and `δbuoyancy ∇p` drives the velocity equations. Note the **node
alignment is automatic**: `δK` lives at the pressure node (same as `K`) and `∇·v̄`
is evaluated there; `δB_d` lives at the `v_d` node (same as `B_d`) and `∇_d p̄` is
evaluated there. No interpolation of the scattering models is needed — this is the
main structural payoff of parameterising in constitutive space (§3).

### 2.4 Discrete Born forward (mirrors `scalar_born.c`'s single coupled pass)

The Born forward runs the background and scattered fields side by side in one pass,
exactly like `scalar_born.c` (which updates `wfp` and `wfpsc` together with the
`2·v·scatter·dt²·w_sum` coupling term). The scattered updates reuse the *stored
background terms* the forward kernel already checkpoints:

```
scattered velocity:  δv_d ← δv_d − dt·B_d·(G_d δp + ψ_d^{sc})  −  dt · δB_d · term̄_d
scattered pressure:  δp   ← δp   − dt·K·(Σ_d D_d δv_d + φ^{sc}) −  dt · δK · div̄
```

where `term̄_d` (= `G_d p̄ + ψ_d`) and `div̄` (= `Σ_d(D_d v̄_d + φ_d)`) are the
**same quantities** the forward already stores as `b*_grad_store_1` and
`k_grad_store_1` for its gradient pass. So the Born virtual sources are
`−dt·δB_d·(b*_grad_store_1)` and `−dt·δK·(k_grad_store_1)` — a direct analogue of
scalar_born's `+2·v·scatter·dt²·w_store_1` coupling. This is a strong implementation
signpost: the acoustic forward kernel already computes and (optionally) stores the
exact fields the Born coupling needs.

**Scattered stores, required for the full (two-field) backward.** When the
*background* models `K`/`B_d` require grad (the HVP path, §5, and any case where
the background is itself differentiable), the full backward also needs the
**scattered** analogues of the checkpointed terms — `div_sc` (= `Σ_d(D_d δv_d +
φ^{sc}_d)`) and `term_sc_d` (= `G_d δp + ψ^{sc}_d`) — at every stored step. This is
the acoustic counterpart of `scalar_born.c` checkpointing both `w_store_1` (the
background Laplacian term, saved for `grad_scatter`) *and* `wsc_store_1` (the
scattered Laplacian term, saved at line 528 and consumed at lines 967–969 for
`grad_v`'s cross-term). Concretely: the forward must checkpoint `div̄`, `term̄_d`
*and* `div_sc`, `term_sc_d` whenever `K.requires_grad or any(B_d.requires_grad)`,
doubling the per-step checkpoint storage relative to the scattered-only
(`backward_sc`) path (which needs only `div̄`/`term̄_d`). See §4.3 for how
`div_sc`/`term_sc_d` enter the background gradient formula.

### 2.5 Source-cell subtlety (Born = jvp; no extra source needed)

Because the forward pre-scales the pressure source by `dt·K` and each velocity
source by `dt·B_d`, differentiating that pre-scaling in the perturbation direction
yields a *scattered* point source automatically:

```
pressure source tangent:  dt · (δK · s + K · δs)     injected into δp
velocity source tangent:   dt · (δB_d · f + B_d · δf) injected into δv_d
```

This is the exact acoustic analogue of scalar's `−(2 v dv dt² f + v² dt² df)`
scattered source (`_scalar_forward_jvp`, lines 1112–1117). It means a velocity/
density-model tangent that overlaps a source cell is handled with no separate
source term — the Born scattered source supplies it. `acoustic_born` therefore
shares the source *locations* between the background and scattered fields (as
`scalar_born` shares them), and the jvp rule (§6) feeds these pre-scaled tangents in
as the scattered source amplitudes.

---

## 3. Parameterisation choice: kernel scatters in `(δK, δB)`

**Decision: the kernel-level scattering variables are the constitutive
perturbations `(δK, δB_z, δB_y, δB_x)`. The physical `(δv, δρ)` chain rule lives in
differentiable Python.**

### 3.1 The chain rule (done in Python, autograd-composed)

```
δK   = 2 ρ v δv + v² δρ          (= 2 (K/v) δv + (K/ρ) δρ)
δB_d = − δρ̄_d / ρ̄_d²             (ρ̄_d = staggered arithmetic mean of ρ, per prepare_models)
```

The `δB_d` averaging is inherited automatically: `prepare_models(v, ρ)` already maps
`(v, ρ) → (K, B_d)` with `B_d = 2/(ρ_i + ρ_{i+1})` differentiably, so a
`prepare_scatter(v, ρ, δv, δρ)` (or, equivalently, a forward-mode derivative of
`prepare_models`) produces `(δK, δB_d)` with the correct discrete averaging, and the
adjoint (VJP) turns kernel gradients `(grad_δK, grad_δB_d)` back into
`(grad_δv, grad_δρ)` with no hand-derivation.

### 3.2 Why constitutive space (not `(δv, δρ)`) at the kernel boundary

1. **`(δK, δB)` are the perturbations that appear *linearly* in the wave operator**
   (§2). `(δv, δρ) → (δK, δB)` is a pointwise smooth reparameterisation containing no
   wave physics — precisely the kind of composition `thomsen_to_cij` performs for
   VTI (`(vp, vs, ρ, ε, δ, γ) → Cij`) and `prepare_models` performs for acoustic
   (`(v, ρ) → (K, B)`). Keeping that map in Python keeps the kernel
   parameterisation-agnostic: a user wanting impedance/velocity, `K/ρ`, or slowness
   perturbations composes a different differentiable Python chain and reuses the same
   kernel.
2. **It matches how `acoustic` already works.** The nonlinear C kernel never sees
   `v` or `ρ` — `acoustic.py` calls `prepare_models` and the kernel consumes only
   `K` and the `B_d`. The Born kernel's scattering models are then *literally* `δ`
   of the same `K, B_d` the forward kernel consumes, so forward and Born share model
   layout, node alignment, and PML profiles.
3. **The adjoint is clean and reuses the acoustic backward.** Because Born is linear
   in `(δK, δB)`, the kernel produces `grad_δK`, `grad_δB_d` with the *same imaging
   condition* as the nonlinear acoustic model gradient (§4); the physical-parameter
   gradient is the Python VJP. This is exactly the `scalar_born` pattern (kernel
   scatters a raw model perturbation; any chain rule to physical parameters is
   Python).

### 3.3 Public vs kernel parameterisation

The **public** `acoustic_born` API takes the *physical* scattering potentials
`dv`, `drho` (mirroring `scalar_born(v, scatter, …)`, whose public scatterer is a
velocity-space perturbation). The Python front-end composes `(δK, δB_d)` and hands
those to the autograd `Function`/kernel. Gradients w.r.t. `dv`/`drho` flow back
through the composition automatically. (An escape hatch to pass `(δK, δB)` directly
can be added later if a user needs raw constitutive LSRTM; not required for v1.)

---

## 4. Adjoint of the linearised system

The Born forward is a linear operator `L : (δK, δB) ↦ δd` (scattered receiver data),
for fixed background `(K, B)` and fixed acquisition. We need `Lᵀ` (data residual →
model-perturbation gradient) for LSRTM, for the VJP `backward`, and as a building
block of the HVP (§5).

### 4.1 Imaging conditions

Run the acoustic **adjoint** (already implemented in `acoustic.c` `FUNC(backward)`)
on a *scattered adjoint field* `(q_p, q_{v,d})` sourced by the data residuals
`grad_δd` at the receiver locations (pressure residual at pressure receivers,
velocity-component residuals at velocity receivers), time-reversed. Then correlate
with the stored forward background terms:

```
grad_δK    = − Σ_t dt · q_p^{n}       · div̄^{n}      · step_ratio
grad_δB_d  = − Σ_t dt · q_{v,d}^{n+½}  · term̄_d^{n+½} · step_ratio
```

with `div̄ = k_grad_store_1` and `term̄_d = b*_grad_store_1` the checkpointed forward
background quantities. These are **identical in form to the nonlinear acoustic model
gradients** `grad_k` / `grad_b` (`acoustic.c` backward, lines 1175–1177 and
1097–1099: `grad -= dt · (adjoint field) · (stored forward term) · step_ratio`) —
because `L` is linear in `(δK, δB)`, its adjoint w.r.t. the scattering models is the
same correlation, just with the adjoint field sourced by scattered-data residuals.
Note `grad_δK`, `grad_δB_d` are **independent of `(δK, δB)`** (linearity), exactly as
`scalar_born`'s `grad_scatter = wfcsc · 2 v dt² · w_store_1` is independent of
`scatter`.

### 4.2 Transpose structure (already in the acoustic backward)

The discrete adjoint swaps the roles of the two updates (because `D_d = −G_dᵀ`): the
adjoint *velocity* update differentiates `K·p` and the adjoint *pressure* update
differentiates `B·v` (`acoustic.c`: `VX_ADJOINT ~ −dt·K·p`, `P_ADJOINT_X ~
−dt·B·v`), with the CPML `a/b` transpose form. The Born adjoint reuses this exact
machinery, adding the coupling terms — the direct analogue of `scalar_born.c`
backward embedding the coupling `2·VDT2·SCATTER·WFCSC` into the background adjoint
via the `V2DT2_WFC` macro.

**Explicit coupling injection (the easiest place to drop a term).** In the full
(two-field) backward, the acoustic backward's per-dimension adjoint macros gain a
second, coupling term reading the *scattered* adjoint field and the *scattering*
model, exactly mirroring how `V2DT2_WFC` embeds `2·VDT2·SCATTER·WFCSC` into
scalar_born's background adjoint update:

```
VX_ADJOINT      = −dt · ( K   · P_adj      +  δK   · P_adj_sc )
P_ADJOINT_X     = −dt · ( B_d · V_adj_d    +  δB_d · V_adj_sc_d )
```

for every dimension `d` (`VZ_ADJOINT`/`VY_ADJOINT` analogously), where `P_adj`,
`V_adj_d` are the **background** adjoint fields and `P_adj_sc`, `V_adj_sc_d` are the
**scattered** adjoint fields. This must be applied **including the PML/CPML
memory-variable variants** of these macros (the `(1+b)`/`ZETA`/`PSI`-bearing forms
in `acoustic.c`, analogous to how `scalar_born.c` embeds `V2DT2_WFC` — itself
already containing the `2·VDT2·SCATTER·WFCSC` coupling — inside the `UT_TERMZ1`/
`UT_TERMZ2`/`PSIZ_TERM` PML macros at lines 111–176, and reads it again inside the
main accumulation at lines 956–958): i.e. every occurrence of `K·P_adj` inside a PML
term must become `K·P_adj + δK·P_adj_sc`, and every occurrence of `B_d·V_adj_d`
must become `B_d·V_adj_d + δB_d·V_adj_sc_d`, in both the plain and the
PML/memory-variable forms of the macro. Missing the PML-variant substitution would
silently zero out the coupling exactly inside the absorbing layer while looking
correct in the interior — the failure mode this note exists to prevent.

### 4.3 Full adjoint (background + scattered) and the `_sc` variant

For **LSRTM / `acoustic_born` public modelling** with a *fixed* background, only
`grad_δK`, `grad_δB_d` (and source/receiver/initial-wavefield gradients) are needed;
this is served by a scattered-only backward `FUNC(backward_sc)`, mirroring
`scalar_born.c`'s `backward_sc` (propagate only the scattered adjoint, accumulate
only `grad_scatter`). For the **HVP** (§5) and for gradients through the background
(`grad_K`, `grad_B` when the background is also an input), the full `FUNC(backward)`
propagates both fields and accumulates both background and scattered model
gradients — the two-field adjoint pass, exactly as `scalar_born.c` `backward`
produces both `grad_v` (background) and `grad_scatter` (scattered).

**Full-backward background gradient formula.** Using the scattered stores
introduced in §2.4 (`div_sc`, `term_sc_d`), the background model gradients in the
full backward are

```
grad_K   = − Σ_t dt · ( q_p^bg   · div̄     +  q_p^sc   · div_sc   ) · step_ratio
grad_B_d = − Σ_t dt · ( q_{v,d}^bg · term̄_d +  q_{v,d}^sc · term_sc_d ) · step_ratio
```

i.e. the background adjoint field correlated with the background store *plus* the
scattered adjoint field correlated with the scattered store. **Unlike
`scalar_born`, there is no analogue of its middle cross-term** (the
`2·dt²·scatter·w_store` term folded into `V2DT2_WFC`, `scalar_born.c` line 968):
in scalar_born the coupling `2·v·scatter·dt²·L(u)` has explicit dependence on the
background model `v` (through the leading factor `v`), so differentiating the
coupling itself w.r.t. `v` produces that extra term. Here the acoustic coupling
`−dt·δK·div̄` (resp. `−dt·δB_d·term̄_d`) has **no explicit dependence on `K`
(resp. `B_d`)** — the only place `K`/`B_d` enters is through `div̄`/`term̄_d`
themselves (the background *state*, not the coupling coefficient). That
state-dependence is exactly what the `q_p^sc · div_sc` (resp.
`q_{v,d}^sc · term_sc_d`) cross-term above already captures: `div̄` is the
divergence of the background velocity field, whose own dependence on `K, B_d`
is handled by the ordinary background-field adjoint recursion (the same
recursion the plain nonlinear `acoustic` backward already runs), not by a
separate coupling-coefficient term. So `grad_K`/`grad_B_d` above have exactly
two terms each, not three.

---

## 5. Second-order (HVP) structure, mirroring `ScalarBackwardFunc`

**Goal:** for the nonlinear `acoustic` propagator with a scalar loss `J(m)`,
`m = (K, B)` (or, in physical space, `(v, ρ)`), compute the Hessian-vector product
`H·h` without forming `H`, where `h` is a model-space perturbation.

### 5.1 Why Born powers the HVP

The first-order backward computes the adjoint-state gradient `g(m) = ∂J/∂m`. The
HVP is the directional derivative of `g` in direction `h`:

```
H·h = (∂g/∂m) · h = d/dα |_{α=0} g(m + α h).
```

`g` is a correlation of the forward wavefield with the adjoint wavefield.
Differentiating it in direction `h` produces two contributions — the perturbation of
the *forward* wavefield due to `h`, and the perturbation of the *adjoint* wavefield
due to `h` — and both perturbations are **Born (linearised) propagations with
`scatter = h`**. Hence one Born forward + one Born backward per HVP. This is the
same reason `scalar` uses the scalar-Born backend as its HVP engine.

### 5.2 The two-pass mechanism (acoustic analogue of `ScalarBackwardFunc.backward`)

Following `ScalarBackwardFunc.backward` (scalar.py lines 1503–1935; docstring lines
1521–1531), with the role assignment:

- **Step 1 — Born forward.** Run `acoustic_born` forward with background `= m` and
  `scatter = h` (i.e. `(δK, δB) =` chain-rule image of `h`). The **background**
  fields recompute the forward wavefield; the **scattered** fields give the effect of
  `h` on the forward wavefield. Record the scattered field at the original receivers
  (needed for the term of `H·h` that flows through the adjoint source / data
  residual). No background receivers are needed (empty), matching scalar's
  `fwd_receivers_i = empty`, `fwd_ggreceivers_i = receivers_i` (lines 1611–1616).

- **Step 2 — Born backward.** Run `acoustic_born` backward with the original adjoint
  source `grad_r` (the data residuals for `p` and each `v_d`) as the adjoint source
  and `scatter = h`. Here the **scattered** adjoint field is ordinary
  backpropagation while the **background** (adjoint) field carries the effect of `h`
  on the adjoint wavefield; their combination accumulates the second-order model
  gradient `H·h`.

  **Correction (no negation in acoustic, unlike scalar):** `scalar`/`scalar_born`'s
  two-time-level `wfc`/`wfp` scheme needs a `grad_wfp = −grad_wfp` negation before
  and after the adjoint loop, applied via the `pre_loop`/`post_loop` pointer-swap
  hooks (`negate`/`negate_sc` in `scalar_born.py`). `acoustic.py`'s first-order
  backward has no such scheme and no such negation: `AcousticForwardFunc.backward`
  seeds the adjoint `(p, v_d)` fields directly from the output-wavefield gradients
  (`grad_wavefields`), and the only ping-pong is the `psi`/`psin` auxiliary-buffer
  `swap` (`acoustic.py` lines 1013–1018, `run_backward(..., swap=swap)`). The
  acoustic-Born backward reuses that same direct seeding and `swap` unmodified —
  no negate hook is needed or added for either the background or the scattered
  adjoint field.

### 5.3 Correspondence table (scalar → acoustic)

| scalar (`ScalarBackwardFunc`)          | acoustic (`AcousticBackwardFunc`)                     |
|----------------------------------------|-------------------------------------------------------|
| single model `v`                       | models `(K, B_z, B_y, B_x)`                           |
| perturbation `ggv` → Born `scatter`    | `h` → `(δK, δB_d)` (chain-rule image)                 |
| coupling `2 v · scatter · dt² · w_sum` | `−dt·δK·div̄`, `−dt·δB_d·term̄_d`                       |
| adjoint source `grad_r`                | `grad_r` = residuals for `p, v_z, v_y, v_x`          |
| `v2dt2` stored coupling                | `dt·K`, `dt·B_d` couplings + `k/b*_grad_store_1`     |
| Born fwd + Born bwd = `H·h`            | acoustic-Born fwd + acoustic-Born bwd = `H·h`        |

### 5.4 Wiring change that replaced the `no_second_order` stopgap

`AcousticForwardFunc.backward` no longer is decorated `@no_second_order` and no
longer calls the C backend directly (which used to make a `create_graph=True`
backward *raise*). To enable the true HVP, `AcousticForwardFunc.backward`
instead builds a nested differentiable `Function` — `AcousticBackwardFunc` —
exactly as `ScalarForwardFunc.backward` builds `ScalarBackwardFunc.apply(...)`
(scalar.py line 1011). `AcousticBackwardFunc.forward` runs the first-order
adjoint (calling the acoustic backward backend); its `backward` is the
double-backward of §5.2 (`@once_differentiable`, calling the acoustic-Born
forward + backward backends). The `no_second_order` guard has been removed
from acoustic, whose HVP path is real.

---

## 6. Forward-mode (`jvp`) via Born, mirroring `_scalar_forward_jvp`

Register an `AcousticForwardFunc.jvp` that routes tangents through the Born backend
(replacing the current `unsupported_jvp`). With the primal inputs saved via
`save_for_forward`, the tangent map is (analogue of `_scalar_forward_jvp`,
lines 1090–1251):

```
(v, ρ) tangent      → (δK, δB) via forward-mode derivative of prepare_models → Born scatter
source tangents     → Born scattered sources  (pre-scaling tangent: dt·(δK·s+K·δs), dt·(δB_d·f+B_d·δf))
initial-wf tangents → Born scattered initial wavefields
```

One `AcousticBornForwardFunc.forward` call (invoked directly on concrete `no_grad`
tensors at base dispatch level, not via `.apply`, so it builds no graph and is not
re-intercepted by the active transform) yields primal `= acoustic(v,ρ)` (Born
background outputs) and tangent `= acoustic_born` scattered outputs together. So:

```
jvp(acoustic, (v,ρ); (δv,δρ)).tangent = acoustic_born(dv=δv, drho=δρ).scattered_output
jvp(acoustic, …).primal               = acoustic(v,ρ)              = Born background output
```

**PML caveat (identical to scalar).** Born does not linearise the PML profile's
dependence on `v` (through `max_vel`/absorber design), so the jvp is exact only
where the tangent is zero in the PML region (or `pml_width = 0`); a warning is
emitted otherwise, matching `_scalar_forward_jvp`'s `_v_tangent_in_pml` warning.

---

## 7. Planned public API

Mirrors `scalar_born` (background + scattered sources/receivers, both fields'
initial wavefields) and `acoustic` (staggered pressure + per-component velocity
sources/receivers).

```
def acoustic_born(
    v, rho, grid_spacing, dt,
    dv=None, drho=None,                       # scattering potentials (physical space)
    # background + scattered share source *locations* (see §2.5); amplitudes given once:
    source_amplitudes_volume_rate=None, source_locations_volume_rate=None,
    source_amplitudes_z=None, source_locations_z=None,
    source_amplitudes_y=None, source_locations_y=None,
    source_amplitudes_x=None, source_locations_x=None,
    # scattered-field receivers (the Born data) and, optionally, background receivers:
    sc_receiver_locations_p=None, sc_receiver_locations_z=None,
    sc_receiver_locations_y=None, sc_receiver_locations_x=None,
    bg_receiver_locations_p=None, bg_receiver_locations_z=None,
    bg_receiver_locations_y=None, bg_receiver_locations_x=None,
    accuracy=4, pml_width=20, pml_freq=None, max_vel=None, courant_number=None,
    survey_pad=None,
    # initial wavefields: background (pressure_0, vz_0, vy_0, vx_0) AND
    # scattered (pressure_sc_0, vz_sc_0, vy_sc_0, vx_sc_0) -- physical fields
    # only; the absorbing boundary's memory variables for both fields are
    # carried opaquely in initial_state/auxiliary_state, not as keywords.
    pressure_0=None, vz_0=None, vy_0=None, vx_0=None,
    pressure_sc_0=None, vz_sc_0=None, vy_sc_0=None, vx_sc_0=None,
    initial_state=None, origin=None, nt=None,
    model_gradient_sampling_interval=1,
    freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    forward_callback=None, backward_callback=None, callback_frequency=1,
    backend="compiled",
    storage_mode="device", storage_path=".", storage_compression=False,
    location_interpolation="auto", location_interpolation_halfwidth=4,
):
    ...
```

Plus an `AcousticBorn(torch.nn.Module)` wrapper storing `v, rho, grid_spacing`
(and `dv, drho` if desired), matching `ScalarBorn`/`Acoustic`.

**Outputs** (tuple, matching the `scalar_born` "background data before scattered
data" convention): final background wavefields (`pressure_0, vz_0, vy_0, vx_0`),
one opaque `auxiliary_state` holding both fields' absorbing-boundary memory
variables, final scattered wavefields (`pressure_sc_0, vz_sc_0, vy_sc_0,
vx_sc_0`), then background receiver amplitudes (`p, z, y, x`), then scattered
receiver amplitudes (`p, z, y, x`), always returned as a named
`PropagatorOutput`.

**`PropagatorSpec`** (declarative, like `scalar_born`'s):
- `name = "acoustic_born"`, `staggered = True`
- `model_names(ndim) = ["K", "B_z"/"B_y"/"B_x"…, "dK", "dB_z"/…]` (background then
  scattered constitutive models — or keep `(v, rho, dv, drho)` at the front-end and
  compose inside, so the `Function` sees background + scattered `(K, B_d)`)
- `wavefield_names` = acoustic background wavefields followed by scattered ones
- `n_source_types = ndim + 1`, `n_receiver_types = 2·(ndim + 1)` (background +
  scattered), or model background/scattered receiver sets like `scalar_born` does
- `output_names` accordingly.

Differentiability: outputs differentiable w.r.t. `v, rho, dv, drho`, source
amplitudes, and initial wavefields. The `dv/drho → δK/δB` chain rule and the
`v/rho → K/B` background chain rule are differentiable Python (§3), so first-order
gradients need no new hand-derivation beyond the kernel's `grad_δK/grad_δB` (§4).

---

## 8. Planned kernel filenames and symbols

- **`src/deepwave/acoustic_born.c`** and **`src/deepwave/acoustic_born.cu`** (kept in
  lockstep, `.cu` GPU-tested by the maintainer before release).
- Symbol convention (mirroring `acoustic_iso_*` and `scalar_born_iso_*`):
  `acoustic_born_iso_<ndim>d_<accuracy>_<dtype>_<name>_<device>`, with
  `name ∈ {forward, backward, backward_sc}` — e.g.
  `acoustic_born_iso_2d_4_float_forward_cpu`.
  - `forward`: single coupled pass, background + scattered fields (§2.4).
  - `backward`: full two-field adjoint, accumulates background *and* scattered model
    gradients (used by the HVP and background gradients) (§4.3).
  - `backward_sc`: scattered-only adjoint, accumulates only `grad_δK`/`grad_δB_d`
    (fixed-background LSRTM), no background adjoint field (§4.3).
- Registered under backend name `"acoustic_born"` in `backend_utils`
  (`get_backend_function("acoustic_born", ndim, "forward"/"backward", …)`), mirroring
  `"scalar_born"`.
- **`CMakeLists.txt`** gains the `acoustic_born` translation units in the same
  `ndim × accuracy × dtype × device` specialisation matrix (with multiversion
  flags consistent with `acoustic`; there is no `DW_MPML` variant, since
  `acoustic_born` — like `acoustic` — has no M-PML support at all).

---

## 9. Planned validation gates

The follow-up implementation is complete only when all of these pass (CPU C backend
+ eager backend; `.cu` compiled, GPU-run by the maintainer). Names mirror the
existing `tests/test_scalar_born.py` / `tests/test_scalar_jvp.py` gates.

1. **Born linearity** — scattered output is exactly linear in `(dv, drho)`: scaling
   the scatterer by `α` scales the scattered receivers by `α`; `(dv,drho)=0` gives
   zero scattered output and background output byte-identical to `acoustic(v,ρ)`.
   *(cf. `test_born_linear_in_scatter`.)*
2. **Born == directional FD of acoustic** — scattered receivers equal
   `(acoustic(v+ε dv, ρ+ε dρ) − acoustic(v−ε dv, ρ−ε dρ)) / (2ε)`, and background
   receivers equal `acoustic(v,ρ)`, for interior perturbations. *(cf.
   `test_born_scatter`, `test_born_is_jvp_of_scalar`.)*
3. **jvp == Born** — `torch.func.jvp(acoustic, (v,ρ), (dv,dρ))` tangent equals the
   `acoustic_born` scattered output (eager to ~1e-12; compiled C to ~1e-6 fast-math
   tol); legacy `forward_ad` path agrees; the PML-boundary caveat warns and the
   nested-transform case raises. *(cf. `test_born_is_jvp_of_scalar`,
   `test_jvp_*`, `test_jvp_v_tangent_in_pml_warns`, `test_jvp_nested_transforms_raise`.)*
4. **HVP == FD-of-gradient** — the second-order path gives `H·h` equal to
   `(∇J(m+ε h) − ∇J(m−ε h)) / (2ε)` for `J` a scattered/background data misfit and
   `h` an interior model perturbation, for `(v,ρ)` and cross terms. This replaced
   the former `test_second_order_raises`; the acoustic `no_second_order` raise test
   has been retired for acoustic (kept for elastic until its Born lands).
5. **gradcheck + gradgradcheck** — `torch.autograd.gradcheck` on `acoustic_born`
   w.r.t. `v, rho, dv, drho`, source amplitudes, and initial wavefields (float64,
   accuracies 2/4/6/8, `step_ratio` 1 and >1, batched `v`/`rho`, `pml_width=0` and
   asymmetric); `torch.autograd.gradgradcheck` on `acoustic` exercising the HVP path.
   Plus at least one **absolute FD gradcheck** on a tiny problem as ground truth, and
   cross-backend (C vs eager) parity on forward, gradient, and HVP.
6. **Reciprocity** — source↔receiver reciprocity of the Born scattered response
   (accounting for the pressure/velocity source–receiver duality on the staggered
   grid). *(cf. `test_reciprocity`.)*

Supporting checks: byte-identity of background outputs vs plain `acoustic` when the
scatterer is zero; `IGNORE_LOCATION` handling; storage-mode (`device/cpu/disk/none`)
and checkpointing parity; callback parity; ABI freeze/drift tests extended with the
additive `acoustic_born_*` symbols.

---

## 10. Draft header comment for `acoustic_born.c`

```c
/* Variable-density acoustic Born (linearised) propagator.
 *
 * Propagates a background velocity-pressure field (p, v_d) and a scattered field
 * (dp, dv_d) simultaneously on a staggered grid, for perturbations of the
 * constitutive models: bulk modulus dK (at the pressure node) and buoyancy dB_d
 * (at the v_d node). The scattered field satisfies the SAME acoustic system with
 * the background (K, B_d), driven by two virtual sources (first-order Born):
 *
 *     scattered velocity:  dv_d -= dt * B_d * (grad_d(dp) + psi_sc)  -  dt * dB_d * term_bg_d
 *     scattered pressure:  dp   -= dt * K   * (div(dv)   + phi_sc)   -  dt * dK   * div_bg
 *
 * where term_bg_d = grad_d(p) + psi  and  div_bg = sum_d(div_d(v) + phi) are the
 * background quantities the forward already checkpoints (b*_grad_store_1,
 * k_grad_store_1). This is the acoustic analogue of scalar_born.c's
 * 2*v*scatter*dt^2*w_sum coupling.
 *
 * The v/rho -> K,B and dv,drho -> dK,dB chain rules live in differentiable Python
 * (see docs/acoustic_born_derivation.md); this kernel scatters only in (dK, dB).
 *
 * When K/B_d require grad (full two-field backward / HVP path), the forward ALSO
 * checkpoints the scattered stores div_sc = sum_d(div_d(dv) + phi_sc_d) and
 * term_sc_d = grad_d(dp) + psi_sc_d (doubling per-step checkpoint storage vs the
 * scattered-only path), the direct analogue of scalar_born.c checkpointing both
 * w_store_1 (background) and wsc_store_1 (scattered, saved line 528 / consumed
 * lines 967-969).
 *
 * backward: full two-field adjoint (transpose: adjoint velocity differentiates K*p,
 * adjoint pressure differentiates B*v; div = -grad^T on the staggered grid). The
 * per-dimension adjoint macros carry an explicit coupling term reading the
 * SCATTERED adjoint field and the SCATTERING model -- this must be applied in BOTH
 * the plain and the PML/memory-variable (phi/psi) forms of the macro, e.g.:
 *
 *     VX_ADJOINT  = -dt * ( K   * P_adj     + dK   * P_adj_sc )
 *     P_ADJOINT_X = -dt * ( B_d * V_adj_d   + dB_d * V_adj_sc_d )
 *
 * (analogue of V2DT2_WFC = V2DT2*WFC + 2*VDT2*SCATTER*WFCSC in scalar_born.c,
 * itself embedded inside the PML macros UT_TERMZ1/UT_TERMZ2/PSIZ_TERM at lines
 * 111-176 and read again in the main accumulation at lines 956-958). Background
 * gradients (only needed when K/B_d require grad):
 *
 *     grad_K   -= dt * ( q_p_bg   * div_bg     + q_p_sc   * div_sc   ) * step_ratio
 *     grad_B_d -= dt * ( q_v_bg_d * term_bg_d  + q_v_sc_d * term_sc_d) * step_ratio
 *
 * (two terms each -- NO third "coupling-coefficient" cross-term as in scalar_born,
 * because the coupling -dt*dK*div_bg has no explicit K-dependence; K enters only
 * through div_bg itself, handled by the ordinary background-field adjoint
 * recursion). Also accumulates grad_dK, grad_dB_d (imaging = scattered adjoint
 * field correlated with the stored background term; independent of dK/dB_d by
 * linearity).
 * backward_sc: scattered-only adjoint (fixed background) accumulating grad_dK/grad_dB
 * only, no coupling term, no scattered stores needed. The full backward additionally
 * powers the acoustic Hessian-vector product (one Born forward + one Born backward
 * per HVP), mirroring scalar's ScalarBackwardFunc; unlike scalar_born's backward,
 * NO grad_wfp/adjoint-field negation is used here (acoustic's first-order backward
 * seeds (p, v_d) directly from the output-wavefield gradients; only the psi/psin
 * auxiliary swap is ping-ponged).
 *
 * Symbols: acoustic_born_iso_<ndim>d_<accuracy>_<dtype>_<forward|backward|
 * backward_sc>_<device>.
 */
```

---

## 11. Summary of decisions (for the reviewer)

- **Forward system:** `∂v/∂t = −B∇p`, `∂p/∂t = −K∇·v (+Ks)`, `K=ρv²`, `B=1/ρ`,
  staggered leapfrog with CPML (`acoustic.c`).
- **Born:** scattered `(δp, δv)` obeys the same operator with background `(K,B)`,
  virtual sources `−δK ∇·v̄` (pressure) and `−δB_d ∇_d p̄` (velocity); discrete
  coupling reuses the forward's stored `k/b*_grad_store_1`.
- **Constitutive perturbations:** `δK = 2ρvδv + v²δρ`, `δB = −δρ/ρ²`.
- **Parameterisation:** kernel scatters in `(δK, δB)`; `(δv, δρ)` chain rule in
  differentiable Python (like `thomsen_to_cij`/`prepare_models`). Public API is
  physical `(dv, drho)`.
- **Adjoint:** same imaging condition as the nonlinear acoustic gradient, adjoint
  field sourced by scattered-data residuals; `backward` (two-field) + `backward_sc`
  (scattered-only), reusing `acoustic.c`'s transpose adjoint.
- **HVP:** one Born forward + one Born backward per `H·h`, mirroring
  `ScalarBackwardFunc`; `AcousticForwardFunc.backward` builds a nested
  differentiable `AcousticBackwardFunc`, and the `no_second_order` guard has
  been removed.
- **jvp:** Born-based `AcousticForwardFunc.jvp`, mirroring `_scalar_forward_jvp`,
  with the same PML caveat.
- **Files:** `acoustic_born.c`/`.cu`; symbols `acoustic_born_iso_*`; backend name
  `"acoustic_born"`.
- **Gates:** Born linearity; Born == directional FD; jvp == Born; HVP == FD-of-grad;
  gradcheck + gradgradcheck (+ absolute FD); reciprocity; plus parity/ABI checks.
```

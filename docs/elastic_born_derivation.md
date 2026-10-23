# Derivation: isotropic elastic Born (linearised) propagator

*Developer/reference documentation: the discrete derivation behind
`deepwave/elastic_born` and its adjoint. See {doc}`elastic_born` for the
user-facing propagator page.*

**Implementation status (updated).** The physics core
(`prepare_scatter_parameters`, `update_velocities_born`, `update_stresses_born`),
the full **eager** driver (`elastic_born_eager_core`), and the public
`deepwave.elastic_born(...)` / `ElasticBorn` API (§8) are landed and validated
on the eager backend against every physics gate of §10 that does not require the
compiled backend: Born linearity (gate 1), Born == directional FD w.r.t.
`lamb`/`mu`/`buoyancy` in 2D and 3D (gate 2), `jvp == Born` for the eager path
(gate 3, machine-precision), FD-of-gradient / gradcheck + gradgradcheck (gates
4/5 — the eager backend gives correct second-order results through ordinary
autograd), free-surface interaction (gate 7), and zero-scatter background
byte-identity. The **`torch.func.jvp` forward-mode rule on the compiled
`elastic` propagator (§6) is now landed**: `ElasticForwardFunc.jvp`
(`_elastic_forward_jvp`) replaces the `unsupported_jvp` guard and routes the
tangents through the Born relationship — the primal comes from the fast compiled
forward, and the tangent from the **compiled** Born forward core
(`elastic_born_compiled_core`; the eager Born driver is
used only as the M-PML fallback) — so `torch.func.jvp(elastic, …)` and
`torch.autograd.forward_ad` run entirely on the fast backend and match the
`elastic_born` scattered output to near machine precision in 2D and 3D (gate 3,
compiled path, well inside the ~1e-6 fast-math band), with the PML-band-tangent
warning and the single-transform-level restriction. The **second-order (HVP,
§5) path is also landed**: `ElasticForwardFunc.backward` drops
`@no_second_order` for a nested `ElasticBackwardFunc` whose double-backward
drives the compiled Born backend, so `gradgradcheck`/reverse-over-reverse
Hessian-vector products work on the compiled backend (M-PML falls back to
`backend="eager"`). **The
compiled `elastic_born` FORWARD kernel is now landed** (`elastic_born.c`/`.cu`,
symbols `elastic_born_iso_*_forward_*`; §9 ABI/CMake/backend_utils wiring done):
a single coupled pass over the doubled 26/54-wavefield set (§2.5), forward-only
(no gradient stores). `elastic_born(..., backend="compiled")`, and the
default `backend="auto"`,
dispatches to it, validated to reproduce the eager Born driver to near machine
precision on CPU (background AND scattered wavefields + both receiver blocks,
2D/3D, including the zero-scatter and pressure-source paths); the `.cu`
nvcc-compiles in lockstep (GPU-untested on the CPU-only dev host — maintainer
GPU-tests). It is forward-only: gradients on the compiled path are supplied by
re-running the validated eager Born driver in the autograd `Function`'s
backward (`ElasticBornForwardFunc`, a checkpoint-style bridge), so the default
path stays differentiable. M-PML on the compiled path is deferred (compiled
`mpml_fraction > 0` raises, directing to `backend="eager"`).

**The compiled forward's background gradient-store checkpointing is now
landed** (phase-5 prerequisite): when `store_bg` is true (a scatter/background
model needs grad) the forward writes the eight/five/two background
finite-difference terms the adjoint will correlate against —
`dv{i}dbuoyancy` (= `dt·W̄_i`), `dv{k}d{k}` (= `dt·Dv̄_k`),
`dv{j}d{i}_plus_dv{i}d{j}` (= `dt·S̄_ij`) — into a single packed `bg_store`
buffer of shape `[n_store, nt, n_shots, *grid]` (a bespoke full-history buffer,
not the shared step-based storage engine, matching the self-contained Born
forward). The grad-free fast path (`store_bg=false`) is byte-identical (test
`test_forward_bg_store_does_not_perturb_and_is_populated`). ABI grows additively
by one pointer + one bool; the CPU kernel implements the writes (the CUDA symbol
accepts the ABI, GPU store implementation deferred to the GPU adjoint pass).

**Phase 6 landed the compiled `backward_sc` (scattered-only adjoint /
imaging condition, §4.1).** `elastic_born.c`/`.cu` now export
`elastic_born_iso_*_backward_sc_*`: a self-contained reverse loop propagating
a single scattered adjoint field with the plain `elastic` adjoint operator
(fixed background models `λ, μ, μ_ij, b_i`), seeded by the scattered receiver
residuals and/or scattered-final-wavefield gradients, reading the checkpointed
`bg_store` in place of the shared-engine stores exactly as this document
anticipated, and accumulating `grad_δλ/grad_δμ/grad_δμ_ij/grad_δb_i` with the
same imaging forms as `elastic.c`'s nonlinear model gradients. Validated
(`tests/test_elastic_born.py::test_compiled_backward_sc_matches_eager_imaging`)
against the eager Born driver's autograd gradient of a loss seeded on both the
scattered final wavefields and the scattered receivers, in 2D and 3D, with and
without PML (relative error ~1e-6, matching the fast-math §10 gate-3 band);
`abi_reference.json` extended additively (+6 entries: `elastic_born|backward_sc|
{1,2,3}|{float,double}`), no drift to existing symbols.

**Phase 7 wired `backward_sc` into the public dispatch and extended `bg_store`
with the scattered stores.** `ElasticBornForwardFunc.backward` now dispatches
to the compiled `backward_sc` (via `_elastic_born_backward_sc_compiled`)
whenever no background-side quantity needs a gradient — no background model
(`lamb`/`mu`/`buoyancy`), no background source amplitude, no background
initial wavefield, and M-PML off (mirrors `acoustic_born.py`'s
`need_bg_adjoint` condition) — instead of always falling back to the
eager-recompute bridge; this is the first real, user-facing speedup on the
compiled path (`test_compiled_backward_sc_dispatch`, which spies on the
compiled core to prove it fires exactly once on the scattered-only path and
never on the eager backend or a background-grad request, gradients matching
the eager backend to <1e-6). The eager bridge remains the fallback for any
case needing background-side gradients, until the full two-field `backward`
below lands. Also extended the forward's checkpointing with `sc_store`
(`store_sc`-gated), the scattered-field analogue of `bg_store`
(§2.5/§4.3 — `W_sc_i`/`Dv_sc_k`/`S_sc_ij`, written at the same 7 sites bg_store
already writes, from the already-computed `w_sum_sc`/`dv*_sc` terms), the
prerequisite the full two-field `backward` needs for its two-term background
gradients. Validated (`test_forward_sc_store_does_not_perturb_and_matches_fd`)
against a central finite difference of `bg_store`, taken through the *public*
`elastic_born()` API (perturbing the physical `lamb`/`mu`/`buoyancy`) to ~1e-6
relative. A genuine, if surprising, finding surfaced while developing that
test: comparing `store_bg`/`store_sc` on/off via strict `torch.equal` is not a
valid gate once *any* sibling conditional exists in the same `-Ofast`-compiled
SIMD loop, because `-ffast-math` permits floating-point reassociation and the
compiler's vectorisation can shift when new (even always-false-at-runtime)
branches are added nearby — measured at ~2e-13 relative after 12 leapfrog
steps, ULP-level and physically meaningless, confirmed by the independent FD
check agreeing far more tightly (~1e-6, limited by truncation). Both
store-perturbation tests now use a tight relative tolerance (<1e-9) instead of
bitwise equality, documented inline in `tests/test_elastic_born.py`.
`backward_sc` itself, the full two-field `backward` (§4.2's 36-site coupling
checklist), the HVP, and the `jvp` re-pointing remain future work, as detailed
in the phase-6 plan below (still accurate for what remains — items 1 and the
"extend `bg_store`" prerequisite of item 2 are now done; the 36-site coupling
substitution itself is unstarted).

**Phase 8 landed the full two-field `backward` (§4.2's complete 36-site
coupling checklist, §4.3's two-term background gradients).** `elastic_born.c`
now exports `elastic_born_iso_*_backward_*`: a second (background) adjoint
field runs alongside the scattered adjoint field of `backward_sc`, using
`elastic.c`'s exact transpose macros with the `+ Dmodel·Q_sc` coupling injected
at every one of the 36 sites — **all 9 velocity-adjoint macros (A), all 9
stress-adjoint macros (B), all 9 velocity memory-variable recursions (C), and
all 9 stress memory-variable recursions (D), each in BOTH its PLAIN and `_PML`
form** (`_BG`-suffixed macros in the kernel: `VZ_Z_BG`/`VZ_Z_BG_PML`, …,
`SIGMAZZ_Z_BG`/`SIGMAZZ_Z_BG_PML`, …, and the coupled `m_v*_bg`/`m_sigma*_bg`
recursions) — no sub-family deferred. Per §5.3, only the `m_sigma*` family is
double-buffered/ping-ponged (for both fields); `m_v*` updates in place for both
fields. Background-model gradients (`grad_lamb/grad_mu/grad_mu_ij/
grad_buoyancy_i`) accumulate the exact two-term form of §4.3 (background
adjoint × `bg_store` PLUS scattered adjoint × `sc_store`); scattering-model
gradients (`grad_dlamb/…`) accumulate from the scattered adjoint alone, as in
`backward_sc`. Wiring: `get_elastic_born_backward_template` in
`backend_utils.py` (additive ABI, +1 pointer block for scattering models/
background residuals/background adjoint wavefields/`sc_store`/background
grad buffers/background source-amplitude grads over `backward_sc`'s
signature), `elastic_born_backward_compiled` in `elastic_born.py` (re-runs
the forward once with `store_bg=store_sc=True` to regenerate both packed
stores, then drives the kernel), and `ElasticBornForwardFunc.backward`'s
dispatch now routes to this compiled kernel whenever a background-side
quantity needs a gradient (the case `backward_sc` doesn't cover) UNLESS M-PML
is active, which still falls back to the eager bridge. Validated
(`test_compiled_backward_full_matches_eager`, 2D+3D × pml∈{0,3} — `pml=0`
isolates the PLAIN sites, `pml=3` exercises every `_PML` site and memory
recursion) against the eager driver's autograd gradient of a loss seeded on
**every** forward output (both final wavefield blocks and both receiver
blocks, so both adjoint fields are directly driven), requesting gradients for
`lamb`/`mu`/`buoyancy` (background models), `dlamb`/`dmu`/`dbuoyancy`
(scattering models), and the background source amplitude — relative error
~1e-7, all 4 cases green. `abi_reference.json` extended additively (+6
entries: `elastic_born|backward|{1,2,3}|{float,double}`); a real ABI-wiring
bug was caught during verification (not a false green): `backend_utils.py`'s
module-init loop that calls `_assign_argtypes` for every propagator/ndim/
accuracy/dtype combination was missing the plain `elastic_born` `backward`
entry (only `forward` and `backward_sc` were registered), so the new
kernel's `argtypes` were never assigned and the first call raised
`ctypes.ArgumentError`; fixed by adding the missing `_assign_argtypes` call.
CUDA: `elastic_born.cu` still only exports `forward`; the full `backward`
(like `backward_sc`) has no CUDA implementation yet (deferred, matching every
prior phase's compiled-adjoint scope — the maintainer GPU-tests before
release). HVP (§5) and `jvp` re-pointing (§6) were NOT attempted this phase —
budget went entirely to the 36-site kernel and its verification, per the
task brief's priority.

**Phase 9 landed the HVP (§5) and re-pointed the `jvp` (§6) — pure Python, no
kernel change.** `ElasticForwardFunc.backward`'s `@no_second_order` guard is
removed and replaced by a nested differentiable `ElasticBackwardFunc.apply(...)`
(mirroring `AcousticForwardFunc`/`AcousticBackwardFunc`): its `forward` runs the
ordinary first-order compiled adjoint (via `run_backward_core` with the elastic
`SPEC`, the `m_sigma*`-only aux swap of §5.3, `callback_drop_last=1`,
`cpu_tmp_batch`, `backend_extra="_mpml"` when M-PML), and its
`@once_differentiable` `backward` is the two-pass HVP of §5.2. **The HVP drives
the now-landed COMPILED Born backend, not the eager driver** the phase-6 note (3)
anticipated: Step 1 calls `elastic_born_compiled_core` (background = `m`,
scatter = `h` in prepared-model space) for the scattered forward wavefields /
scattered receivers (the grads w.r.t. `grad_wavefields`/`grad_receivers`), and
Step 2 calls `elastic_born_backward_compiled` with the HVP seeding (background
adjoint seeded at zero with empty background residual; scattered adjoint seeded
by the original `grad_wavefields` and sourced by the original `grad_receivers`;
`sc_models = h`, background models `requires_grad`), reading `grad_bg` as `H·h`,
`grad_f_bg` as the source HVP, and the background adjoint final as the
initial-wavefield HVP. `model_gradient_sampling_interval` must give an effective
stride of one for compiled double backward; otherwise the sampled first gradient
is not an exact gradient field and the Born directional operation is not its
reverse Jacobian. `h` lives entirely in prepared-model space — the chain
rule through `prepare_parameters` is handled by `elastic`'s differentiable
front-end, exactly as acoustic's HVP works in `(K, B_d)` space (so
`prepare_scatter_parameters` is *not* called by the HVP). To let the
double-backward recompute the forward field, `ElasticForwardFunc.forward` now
passes `save_wavefields=True` (initial wavefields saved for backward). The `jvp`
(`_elastic_forward_jvp`) now routes its tangent through
`elastic_born_compiled_core` (the same fast kernel the `elastic_born` default
uses) instead of the eager bridge, so `torch.func.jvp`/`torch.autograd.forward_ad`
run entirely on the compiled engine (tangent and reference share the kernel, so
they agree to near machine precision, well inside the ~1e-6 fast-math gate-3
band). **M-PML fallback:** the compiled Born backend has no `_mpml` variants, so
the HVP `backward` raises `NotImplementedError` directing to
`backend="eager"` under M-PML, and the `jvp` falls back to the eager Born
driver under M-PML (first-order M-PML gradients are unaffected — the first-order
adjoint keeps its `_mpml` dispatch). **§5.3 aux-swap finding (confirmed):** the
memory-variable aux swap applies to elastic's *default* (standard-PML) path, not
only under M-PML — the `m_sigma*` family exists whenever `pml_width > 0`, and the
first-order swap ping-pongs exactly that family (2D: wavefield indices `6:9` and
`12`; the `m_v*` family at `3:5`, `11` is never swapped). Validation: compiled
HVP == `backend="eager"` exact second-order to ~1e-11 (2D) / ~1e-12 (3D)
for `lamb`/`mu`/`buoyancy`/source and cross-terms; `gradgradcheck` on compiled
`elastic` succeeds (2D+3D) for `lamb`/`mu`/source (the `buoyancy` block's
finite-difference-of-forward numerical Hessian is unreliable *even on the exact
eager autograd path*, so it is covered by the exact eager-comparison gate
instead); HVP == FD-of-gradient ~2e-7. No `.c`/`.cu`/ABI change (Python-only, as
the phase-8 assessment predicted).

**Phase-6 plan (the remaining compiled fast path), with de-risking findings.**
(1) **`backward_sc` is `elastic.c`'s `FUNC(backward)` almost verbatim** — the
scattered adjoint field obeys the plain elastic adjoint operator with the
*background* models, seeded by the scattered data residuals; its imaging
conditions (`grad_δλ/δμ/δμ_ij/δb_i`, §4.1) are *identical in form* to elastic's
`grad_lamb/grad_mu/…`, reading the `bg_store` terms in place of the shared-engine
stores. So `backward_sc` is a self-contained reverse loop over `bg_store` with
one adjoint field and the memory-variable double-buffer swap on the `m_sigma*`
family only (§5.3). (2) **The full two-field `backward`** adds a second
(background) adjoint field whose update is that same elastic adjoint *plus* the
`+ Dmodel·Q_sc` coupling at the §4.2 sites, and the two-term background gradients
(§4.3), which additionally need the *scattered* stores (extend `bg_store` to also
pack `W_sc/Dv_sc/S_sc`, written from the already-computed `w_sum_sc`/`dv*_sc`
terms — the exact analogues of the bg store sites). (3) **The HVP does not
require the compiled Born backward:** `ElasticBackwardFunc.forward` keeps the
fast compiled first-order adjoint (first-order performance unchanged) and its
`@once_differentiable` `backward` may compute `H·h` via the *eager* Born driver
(one eager Born forward + one eager Born backward, §5.2);
`torch.is_grad_enabled()` (the `no_second_order` discriminator) already isolates
the second-order path. Re-pointing §6 `jvp` to the compiled backend is then a
one-line swap (eager→compiled core), tightening gate 3 to the ~1e-6 fast-math
band. The eager backend already covers all differentiability
(first- and second-order) in the meantime. This document adapts the
variable-density acoustic Born work (`docs/acoustic_born_derivation.md`) to the
isotropic elastic velocity–stress staggered-grid propagator (`elastic.c`/`.py`).
It derives the linearised (Born) system, its adjoint (imaging conditions + the
full two-field background gradients), the second-order (Hessian-vector product)
structure, and the forward-mode (`jvp`) rule; it fixes the planned public API,
kernel filenames, symbols, and validation gates. Per the task's hard requirement,
NO `.c`/`.cu`/CMake code is written in this pass — the kernel implementation is a
separate follow-up after this derivation is reviewed. The guards this work will
eventually replace are `elastic.py` lines 1629–1633
(`ElasticForwardFunc.jvp = unsupported_jvp` and the `@no_second_order` decorator
on `ElasticForwardFunc.backward`).*

Throughout, the isotropic elastic constitutive fields are the first Lamé
parameter `λ` (`lamb`), the shear modulus `μ` (`mu`), and the buoyancy
`b = 1/ρ` (`buoyancy`). The kernel additionally consumes the *staggered* derived
models `μ_zy, μ_zx, μ_yx` (shear moduli harmonically averaged onto the shear-stress
nodes) and `b_z, b_y, b_x` (buoyancy arithmetically averaged onto the velocity
nodes), all produced by `elastic.py::prepare_parameters(mu, buoyancy)`. The state is
the particle velocity `v = (v_z, v_y, v_x)` and the symmetric stress tensor
`σ = (σ_zz, σ_yy, σ_xx, σ_yz, σ_xz, σ_xy)`. A background quantity carries a bar
(`v̄`, `σ̄`, `λ`, `μ`, `b`); a first-order perturbation carries a `δ`
(`δv`, `δσ`, `δλ`, `δμ`, `δb`).

---

## 1. The forward system (what `elastic` solves)

### 1.1 Continuous equations

The isotropic elastic first-order velocity–stress system is

```
velocity:   ρ ∂v_i/∂t = Σ_j ∂_j σ_ij   (+ f_i)      ⇔   ∂v_i/∂t = b (Σ_j ∂_j σ_ij) (+ b f_i)
stress:     ∂σ_ij/∂t = λ δ_ij (Σ_k ∂_k v_k) + μ (∂_i v_j + ∂_j v_i)   (+ source)
```

i.e. Newton's law (buoyancy times the divergence of the stress row `i`) for each
velocity component, and the isotropic Hooke's law rate form for each stress
component: the normal stresses `σ_ii` couple to the dilatation `Σ_k ∂_k v_k` through
`λ` plus `2μ ∂_i v_i`; the shear stresses `σ_ij` (`i≠j`) couple to the engineering
shear strain rate `∂_i v_j + ∂_j v_i` through `μ`.

### 1.2 Discrete staggered leapfrog (the reference backend)

On the staggered grid, `λ`, `μ`, and the normal stresses `σ_ii` live at the integer
("normal-stress") nodes; each velocity `v_i` and its buoyancy `b_i` live half a cell
from `σ_ii` in dimension `i`; each shear stress `σ_ij` and its shear modulus `μ_ij`
live half a cell in *both* `i` and `j`. One internal step is (from `elastic.c`
`FUNC(forward)`; `∂_j` denotes the appropriate half/integer staggered difference,
optionally augmented by a CPML/M-PML memory variable `m_*`):

```
velocity:   W_i   = Σ_j ∂_j σ̄_ij  (+ PML memory)               (checkpointed: dv{i}dbuoyancy_store_1 = dt·W_i)
            v̄_i  ← v̄_i + b_i · dt · W_i

normal σ:   Dv̄_k = ∂_k v̄_k       (+ PML memory)               (checkpointed: dv{k}d{k}_store_1 = dt·Dv̄_k)
            σ̄_ii ← σ̄_ii + dt·( λ · Σ_k Dv̄_k  +  2μ · Dv̄_i )

shear σ:    S̄_ij = ∂_i v̄_j + ∂_j v̄_i  (+ PML memory)          (checkpointed: dv{j}d{i}_plus_dv{i}d{j}_store_1 = dt·S̄_ij)
            σ̄_ij ← σ̄_ij + dt · μ_ij · S̄_ij
```

**The eight checkpointed stores** (3D; 5 in 2D, 2 in 1D — `elastic.c` lines given for
reference) are exactly the quantities the Born virtual sources need. Each store holds
`dt ·` its finite-difference quantity (the `dt` is baked in), and each is written only
when the corresponding model requires grad:

| store name (`elastic.c`)                 | holds `dt ·`                 | node        | multiplies (fwd)        | grad guard                    |
|------------------------------------------|------------------------------|-------------|-------------------------|-------------------------------|
| `dvzdbuoyancy_store_1` (l.994)           | `W_z = Σ_j ∂_j σ̄_zj`         | `v_z`       | `buoyancy_z`            | `buoyancy_requires_grad`      |
| `dvydbuoyancy_store_1` (l.1045)          | `W_y = Σ_j ∂_j σ̄_yj`         | `v_y`       | `buoyancy_y`            | `buoyancy_requires_grad`      |
| `dvxdbuoyancy_store_1` (l.1098)          | `W_x = Σ_j ∂_j σ̄_xj`         | `v_x`       | `buoyancy_x`            | `buoyancy_requires_grad`      |
| `dvzdz_store_1` (l.1168)                 | `Dv̄_z = ∂_z v̄_z`            | `σ_ii`      | `λ` (all σ_ii), `2μ` (σzz) | `lamb_or_mu_requires_grad` |
| `dvydy_store_1` (l.1183)                 | `Dv̄_y = ∂_y v̄_y`            | `σ_ii`      | `λ` (all σ_ii), `2μ` (σyy) | `lamb_or_mu_requires_grad` |
| `dvxdx_store_1` (l.1197)                 | `Dv̄_x = ∂_x v̄_x`            | `σ_ii`      | `λ` (all σ_ii), `2μ` (σxx) | `lamb_or_mu_requires_grad` |
| `dvydx_plus_dvxdy_store_1` (l.1244)      | `S̄_xy = ∂_y v̄_x + ∂_x v̄_y` | `σ_xy`      | `mu_yx`                 | `mu_requires_grad`            |
| `dvzdx_plus_dvxdz_store_1` (l.1283)      | `S̄_xz = ∂_z v̄_x + ∂_x v̄_z` | `σ_xz`      | `mu_zx`                 | `mu_requires_grad`            |
| `dvzdy_plus_dvydz_store_1` (l.1321)      | `S̄_yz = ∂_z v̄_y + ∂_y v̄_z` | `σ_yz`      | `mu_zy`                 | `mu_requires_grad`            |

**Key discrete fact (used by the adjoint):** on the staggered grid the difference
operators satisfy the exact-transpose property (the same one that gives `D_d =
−G_dᵀ` for acoustic), so the elastic adjoint kernel `FUNC(backward)` is a clean
transpose: the adjoint *velocity* update differentiates the stress equation (carries
`λ/μ/μ_ij · σ_adj` products) and the adjoint *stress* update differentiates the
velocity equation (carries `b_i · v_adj` products). This is the machinery the Born
adjoint reuses (§4).

Sources (`elastic.py` lines 1296–1328): each velocity/force source is pre-scaled by
`dt · b_i` (buoyancy gathered at the source cell) and added into `v̄_i`; the
"pressure" source is pre-scaled by `−dt` **only, with no model factor**, and added
equally into all normal stresses `σ_ii`. (The pressure *receiver* reads
`−(Σ_i σ_ii)/ndim`.) This asymmetry — velocity source carries a model factor,
pressure source does not — drives the source-cell tangent difference in §5.

---

## 2. Born linearisation

Perturb the constitutive fields `λ→λ+δλ`, `μ→μ+δμ`, `b_i→b_i+δb_i`, and (as derived
kernel models) `μ_ij→μ_ij+δμ_ij`. Split the state `v = v̄+δv`, `σ = σ̄+δσ`.
Substitute into §1.1 and keep first order.

### 2.1 Velocity equation

```
∂(v̄_i+δv_i)/∂t = (b_i+δb_i) Σ_j ∂_j (σ̄_ij + δσ_ij)
```

Zeroth order (background): `∂v̄_i/∂t = b_i Σ_j ∂_j σ̄_ij`. First order:

```
∂δv_i/∂t = b_i Σ_j ∂_j δσ_ij   +   δb_i Σ_j ∂_j σ̄_ij
```

### 2.2 Normal-stress equation

```
∂(σ̄_ii+δσ_ii)/∂t = (λ+δλ) Σ_k ∂_k(v̄_k+δv_k) + 2(μ+δμ) ∂_i(v̄_i+δv_i)
```

First order:

```
∂δσ_ii/∂t = λ Σ_k ∂_k δv_k + 2μ ∂_i δv_i   +   δλ Σ_k ∂_k v̄_k + 2δμ ∂_i v̄_i
```

### 2.3 Shear-stress equation

```
∂(σ̄_ij+δσ_ij)/∂t = (μ_ij+δμ_ij)(∂_i(v̄_j+δv_j) + ∂_j(v̄_i+δv_i))
```

First order:

```
∂δσ_ij/∂t = μ_ij(∂_i δv_j + ∂_j δv_i)   +   δμ_ij(∂_i v̄_j + ∂_j v̄_i)
```

### 2.4 The scattered field obeys the same operator, with virtual sources

Collecting §2.1–§2.3: **the scattered field `(δv, δσ)` satisfies the identical
isotropic elastic system with the background parameters `(λ, μ, μ_ij, b_i)`, driven
by three families of virtual sources** — precisely the quantities the forward already
checkpoints (§1.2):

```
velocity eq  (at v_i node):    virtual force      = δb_i · W̄_i           = δb_i · (dv{i}dbuoyancy_store_1 / dt)
normal-σ eq  (at σ_ii node):   virtual injection  = δλ · Σ_k Dv̄_k + 2δμ · Dv̄_i
shear-σ eq   (at σ_ij node):   virtual injection  = δμ_ij · S̄_ij         = δμ_ij · (dv{j}d{i}_plus_dv{i}d{j}_store_1 / dt)
```

with `W̄_i = dv{i}dbuoyancy_store_1/dt`, `Dv̄_k = dv{k}d{k}_store_1/dt`,
`S̄_ij = dv{j}d{i}_plus_dv{i}d{j}_store_1/dt`. This is the exact elastic analogue of
acoustic §2.3 (`−δK ∇·v̄`, `−δB_d ∇_d p̄`) and of `scalar_born`'s
`2·v·scatter·dt²·w_sum` coupling. **Node alignment is automatic**: `δλ`/`δμ` live at
the normal-stress node (where `Σ_k Dv̄_k`/`Dv̄_i` are evaluated), `δμ_ij` at the shear
node (where `S̄_ij` lives), `δb_i` at the `v_i` node (where `W̄_i` lives) — because the
scattering models are `δ` of the *same* prepared model set the forward consumes (§3).

### 2.5 Discrete Born forward (one coupled pass)

The Born forward runs the background and scattered fields side by side in one pass,
exactly like `acoustic_born.c` and `scalar_born.c`. The scattered updates reuse the
stored background terms:

```
scattered velocity:   δv_i  ← δv_i  + b_i·dt·(Σ_j ∂_j δσ_ij + PML_sc)  +  δb_i · (dv{i}dbuoyancy_store_1)
scattered normal σ:   δσ_ii ← δσ_ii + dt·( λ·Σ_k ∂_k δv_k + 2μ·∂_i δv_i + PML_sc )
                                     +  δλ·(Σ_k dv{k}d{k}_store_1) + 2δμ·(dv{i}d{i}_store_1)
scattered shear σ:    δσ_ij ← δσ_ij + dt·μ_ij·(∂_i δv_j + ∂_j δv_i + PML_sc)
                                     +  δμ_ij·(dv{j}d{i}_plus_dv{i}d{j}_store_1)
```

So the Born virtual sources are literally the stored background terms multiplied by
the scattering models — no new spatial derivatives of the background are needed in the
Born forward.

**Scattered stores, required for the full (two-field) backward (§4.3).** When a
*background* model (`λ`, `μ`, or any `b_i`) requires grad — the HVP path (§5) and any
differentiable-background case — the full backward also needs the **scattered**
analogues of the eight stores, evaluated on the *scattered* field:

```
W_sc_i  = Σ_j ∂_j δσ_ij   (+ PML_sc)          [scattered velocity-buoyancy store]
Dv_sc_k = ∂_k δv_k        (+ PML_sc)          [scattered normal store]
S_sc_ij = ∂_i δv_j + ∂_j δv_i  (+ PML_sc)     [scattered shear store]
```

This is the elastic counterpart of `scalar_born.c`/`acoustic_born.c` checkpointing
*both* the background stores *and* the scattered stores (`wsc_store_1` / `div_sc`,
`term_sc_d`). Concretely: the forward must checkpoint all eight background stores AND
all eight scattered stores whenever `lamb.requires_grad or mu.requires_grad or
any(buoyancy.requires_grad)`, roughly doubling per-step checkpoint storage relative to
the scattered-only (`backward_sc`) path (which needs only the background stores). See
§4.3 for how these scattered stores enter the background gradients.

---

## 3. Parameterisation choice: kernel scatters in the prepared model set

**Decision: the kernel-level scattering variables are the perturbations of the exact
model set `elastic.c` consumes — `(δλ, δμ, δμ_zy, δμ_zx, δμ_yx, δb_z, δb_y, δb_x)`.
The chain rule from the public scattering potentials `(dlamb, dmu, dbuoyancy)` lives
in differentiable Python, as the forward-mode derivative of `prepare_parameters`.**

### 3.1 `prepare_scatter_parameters` — forward-mode derivative of `prepare_parameters`

`elastic.py::prepare_parameters(mu, buoyancy)` maps `(μ, b)` to the six staggered
derived models: `μ_zy/μ_zx/μ_yx` are 4-cell **harmonic** averages of `μ` in the two
named dimensions' plane, `b_z/b_y/b_x` are the reciprocal of the 2-cell **arithmetic**
average of `ρ = 1/b`, all guarded by `torch.where` masks that zero any node whose
stencil touches a vacuum cell (the IVM free surface, §7.1). `λ` and `μ` pass through
untouched (they already live at the normal-stress node).

Mirroring `acoustic_born.py::prepare_scatter` (which is *just* `torch.func.jvp` over
`prepare_models`), define:

```python
def prepare_scatter_parameters(mu, buoyancy, dmu, dbuoyancy):
    # forward-mode derivative of prepare_parameters at (mu, buoyancy)
    _, tangents = torch.func.jvp(prepare_parameters, (mu, buoyancy), (dmu, dbuoyancy))
    return list(tangents)           # = [δμ_zy, δμ_zx, δμ_yx, δb_z, δb_y, δb_x]  (3D order)
```

with `δλ = dlamb` and `δμ = dmu` passing straight through (λ, μ are consumed at the
normal-stress node without averaging). This yields, for the record (what the jvp
reproduces automatically):

```
δμ_ij  = Σ_{four plane neighbours k} ( μ_ij_val² / (4 μ_k²) ) · δμ_k     on the mask; 0 where masked
         (derivative of the 4-cell harmonic mean 4/Σ_k 1/μ_k)
δb_i   = ½ · b_i_val² · ( ρ_a² δb_a + ρ_c² δb_c )                        on the mask; 0 where masked
         (from ρ = 1/b, ρ̄_i = (ρ_a+ρ_c)/2, b_i = 1/ρ̄_i, chain-ruled)
```

but **the code never hand-derives these** — `torch.func.jvp` differentiates the exact
masked `torch.where` expressions, so we inherit node alignment, the correct averaging,
AND the IVM vacuum masking "for free" (§7.1). Its **VJP** (the reverse mode used by
`backward` to chain `grad_δμ_*`/`grad_δb_*` back to `grad_dmu`/`grad_dbuoyancy`) is
handled automatically by autograd, because `prepare_scatter_parameters` is plain
differentiable torch ops sitting in front of the kernel `Function` — there is no
hand-written backward for the scatter step. (`torch.func.jvp` composes and
differentiates fine under autograd, exactly as in `acoustic_born`.)

### 3.2 Why the prepared-model space (not physical space) at the kernel boundary

1. **These are the perturbations that appear *linearly* in the wave operator** (§2).
   The map `(dmu, dbuoyancy) → (δμ_ij, δb_i)` is a pointwise smooth reparameterisation
   containing no wave physics — the same kind of composition `prepare_parameters`
   (and, one level up, `vpvsrho_to_lambmubuoyancy`) already performs. Keeping it in
   Python keeps the kernel parameterisation-agnostic.
2. **It matches how `elastic` already works.** The nonlinear C kernel never sees
   `vp/vs/ρ` or even raw `μ/b` at shear/velocity nodes — `elastic.py` calls
   `prepare_parameters` and the kernel consumes `lamb, mu, mu_ij, buoyancy_i`. The Born
   kernel's scattering models are then *literally* `δ` of the same models the forward
   consumes: forward and Born share model layout, node alignment, PML profiles, and IVM
   masking.
3. **The adjoint is clean and reuses the elastic backward.** Because Born is linear in
   the prepared scattering models, the kernel produces `grad_δλ`, `grad_δμ`,
   `grad_δμ_ij`, `grad_δb_i` with the *same imaging condition* as the nonlinear elastic
   model gradients (§4); the physical-parameter gradient is the Python VJP with no
   hand-derivation.

### 3.3 Public vs kernel parameterisation

The **public** `elastic_born` API takes the scattering potentials `dlamb`, `dmu`,
`dbuoyancy` (Lamé/buoyancy space — the *same* space as `elastic`'s own public models
`lamb, mu, buoyancy`). The Python front-end composes the eight prepared scatter models
and hands them to the autograd `Function`/kernel; gradients w.r.t. `dlamb/dmu/dbuoyancy`
flow back through the composition automatically. A user wanting physical `(dvp, dvs,
drho)` scatter composes `torch.func.jvp(vpvsrho_to_lambmubuoyancy, …)` in front (its VJP
is again autograd's job) — no kernel change. (`vpvsrho_to_lambmubuoyancy` is
`common.py` lines 3329–3356: `λ=(vp²−2vs²)ρ`, `μ=vs²ρ`, `b=ρ/(ρ²+ε)`.)

---

## 4. Adjoint of the linearised system

The Born forward is a linear operator `L : (δλ, δμ, δμ_ij, δb_i) ↦ δd` (scattered
receiver data) for fixed background and acquisition. We need `Lᵀ` for LSRTM, for the
VJP `backward`, and as a building block of the HVP (§5).

### 4.1 Imaging conditions (scattered-only backward, `backward_sc`)

Run the elastic **adjoint** (`elastic.c` `FUNC(backward)`, the transpose machinery) on
a *scattered adjoint field* `(q^sc_v, q^sc_σ)` sourced by the data residuals `grad_δd`
at the receiver locations (velocity-component residuals at velocity receivers, pressure
residual `−(Σ q^sc_σii)/ndim` at pressure receivers), time-reversed. Then correlate
with the stored forward **background** terms, using **exactly the same forms as the
nonlinear elastic model gradients** (`elastic.c` grad accumulations; `dt` is already in
each store so only `step_ratio` multiplies):

```
grad_δλ    = Σ_t ( Σ_i q^sc_σii ) · ( Σ_k Dv̄_k )        · step_ratio      (cf. nonlinear grad_lamb, l.2076–2091)
grad_δμ    = Σ_t 2·( Σ_i q^sc_σii · Dv̄_i )              · step_ratio      (cf. grad_mu,   l.2092–2108)
grad_δμ_yx = Σ_t q^sc_σxy · S̄_xy                        · step_ratio      (cf. grad_mu_yx, l.2173–2177)
grad_δμ_zx = Σ_t q^sc_σxz · S̄_xz                        · step_ratio      (cf. grad_mu_zx, l.2211–2215)
grad_δμ_zy = Σ_t q^sc_σyz · S̄_yz                        · step_ratio      (cf. grad_mu_zy, l.2248–2252)
grad_δb_z  = Σ_t q^sc_vz · W̄_z                          · step_ratio      (cf. grad_buoyancy_z, l.1907–1911)
grad_δb_y  = Σ_t q^sc_vy · W̄_y                          · step_ratio      (cf. grad_buoyancy_y, l.1959–1963)
grad_δb_x  = Σ_t q^sc_vx · W̄_x                          · step_ratio      (cf. grad_buoyancy_x, l.2014–2018)
```

where `Dv̄_k = dv{k}d{k}_store_1`, `S̄_ij = dv{j}d{i}_plus_dv{i}d{j}_store_1`,
`W̄_i = dv{i}dbuoyancy_store_1` are the checkpointed forward background terms. These are
**independent of the scatter** (`L` linear), exactly as acoustic's `grad_δK` is
independent of `δK`. `backward_sc` propagates only the scattered adjoint field and
accumulates only these eight gradients — the cheap fixed-background LSRTM path,
mirroring `scalar_born.c`/`acoustic_born.c` `backward_sc`.

### 4.2 The adjoint coupling substitution checklist (full two-field `backward`)

The full backward propagates BOTH a background adjoint field `(q^bg_v, q^bg_σ)` and a
scattered adjoint field `(q^sc_v, q^sc_σ)`. It reuses `elastic.c`'s transpose adjoint
unchanged, EXCEPT that **every product of the form `MODEL · Q` (background model times a
background adjoint field) gains a second coupling term `+ Dmodel · Q_sc` (the scattering
model times the *scattered* adjoint field)** — the direct analogue of `scalar_born.c`
embedding `2·VDT2·SCATTER·WFCSC` into `V2DT2_WFC`, and `acoustic_born.c`'s
`K·P_adj + δK·P_adj_sc` / `B_d·V_adj_d + δB_d·V_adj_sc_d`.

Symbolically, in every site below:

```
λ · Qσ      →  λ · Qσ      + δλ · Qσ_sc
2μ · Qσ     →  2μ · Qσ     + 2δμ · Qσ_sc            (normal-stress diagonal only)
μ_ij · Qσij →  μ_ij · Qσij + δμ_ij · Qσij_sc
b_i · Qv    →  b_i · Qv    + δb_i · Qv_sc
```

**This substitution MUST be applied at ALL of the following sites — the PLAIN form,
the `_PML` form, AND the memory-variable recursion — because missing the PML/memory
variant would silently zero the coupling inside the absorbing layer while looking
correct in the interior.** The exhaustive site list (from `elastic.c`; line numbers for
the reviewer):

**(A) Velocity-adjoint macros — `λ/μ/μ_ij · σ_adj` (transpose of the stress update).**
Each has a PLAIN and a `_PML` form; the `_PML` form also reads a `m_v*` memory variable.

| macro (PLAIN / `_PML`)                | products to augment with `+ Dmodel·Qσ_sc`                       |
|---------------------------------------|-----------------------------------------------------------------|
| `VZ_Z` (l.274 / l.280)                | `λ·σxx + λ·σyy + (λ+2μ)·σzz`  (PML adds `P_BZ·m_vzz`)            |
| `VZ_Y` (l.278 / l.286)                | `μ_zy·σyz`  (PML `P_BYH·m_vzy`)                                  |
| `VZ_X` (l.279 / l.289)                | `μ_zx·σxz`  (PML `P_BXH·m_vzx`)                                  |
| `VY_Z` (l.296 / l.310)  [3D]          | `μ_zy·σyz`  (PML `P_BZH·m_vyz`)                                  |
| `VY_Y` (l.299/303 / l.315/321)        | `λ·σxx + (λ+2μ)·σyy [+ λ·σzz]`  (PML `P_BY·m_vyy`)              |
| `VY_X` (l.308 / l.328)                | `μ_yx·σxy`  (PML `P_BXH·m_vyx`)                                  |
| `VX_Z` (l.334 / l.348)  [3D]          | `μ_zx·σxz`  (PML `P_BZH·m_vxz`)                                  |
| `VX_Y` (l.337 / l.353)  [2D+]         | `μ_yx·σxy`  (PML `P_BYH·m_vxy`)                                  |
| `VX_X` (l.340/342 / l.358/362)        | `(λ+2μ)·σxx [+ λ·σyy] [+ λ·σzz]`  (PML `P_BX·m_vxx`)           |

**(B) Stress-adjoint macros — `b_i · v_adj` (transpose of the velocity update).**
Each has a PLAIN and a `_PML` form (the `_PML` form reads a `m_sigma*` memory variable).

| macro (PLAIN / `_PML`)                | product to augment with `+ δb_i·Qv_sc`     |
|---------------------------------------|--------------------------------------------|
| `SIGMAZZ_Z` (l.374 / l.371)           | `b_z·v_z`  (PML `P_BZH·m_sigmazzz`)         |
| `SIGMAYY_Y` (l.380 / l.377)           | `b_y·v_y`  (PML `P_BYH·m_sigmayyy`)         |
| `SIGMAXX_X` (l.385 / l.382)           | `b_x·v_x`  (PML `P_BXH·m_sigmaxxx`)         |
| `SIGMAXY_Y` (l.391 / l.388)           | `b_x·v_x`  (PML `P_BY·m_sigmaxyy`)          |
| `SIGMAXY_X` (l.395 / l.392)           | `b_y·v_y`  (PML `P_BX·m_sigmaxyx`)          |
| `SIGMAXZ_Z` (l.402 / l.399)  [3D]     | `b_x·v_x`  (PML `P_BZ·m_sigmaxzz`)          |
| `SIGMAXZ_X` (l.406 / l.403)  [3D]     | `b_z·v_z`  (PML `P_BX·m_sigmaxzx`)          |
| `SIGMAYZ_Z` (l.411 / l.408)  [3D]     | `b_y·v_y`  (PML `P_BZ·m_sigmayzz`)          |
| `SIGMAYZ_Y` (l.415 / l.412)  [3D]     | `b_z·v_z`  (PML `P_BY·m_sigmayzy`)          |

**(C) Velocity-block memory-variable recursions — `b_i · v_adj` (the `m_sigma*n`
updates, written when `!DW_PML_SKIP`).** Each is of the form
`b_i·dt·P_A?·v_shot + P_A?·m_…`; augment the `b_i·…·v_shot` term with `+ δb_i·…·v_sc_shot`:

```
vz block:  m_sigmazzzn (l.1892–1894, b_z·P_AZH·vz),  m_sigmayzyn (l.1897–1899, b_z·P_AY·vz),  m_sigmaxzxn (l.1902–1904, b_z·P_AX·vz)
vy block:  m_sigmayzzn (l.1943–1945, b_y·P_AZ·vy),  m_sigmayyyn (l.1949–1951, b_y·P_AYH·vy), m_sigmaxyxn (l.1954–1956, b_y·P_AX·vy)
vx block:  m_sigmaxzzn (l.1996–1998, b_x·P_AZ·vx),  m_sigmaxyyn (l.2003–2005, b_x·P_AY·vx),  m_sigmaxxxn (l.2009–2011, b_x·P_AXH·vx)
```

**(D) Stress-block memory-variable recursions — `λ/(λ+2μ)/μ_ij · σ_adj` (the `m_v*`
updates, written when `!DW_PML_SKIP`).** Augment every `MODEL·…·σ_shot` term with
`+ Dmodel·…·σ_sc_shot`:

```
m_vzz (l.2111–2117): (λ+2μ)·P_AZ·σzz + λ·P_AZ·σxx + λ·P_AZ·σyy   (+ P_AZ·m_vzz)
m_vyy (l.2121–2129): (λ+2μ)·P_AY·σyy + λ·P_AY·σxx [+ λ·P_AY·σzz]  (+ P_AY·m_vyy)
m_vxx (l.2132–2142): (λ+2μ)·P_AX·σxx [+ λ·P_AX·σyy] [+ λ·P_AX·σzz] (+ P_AX·m_vxx)
m_vxy (l.2180–2182): μ_yx·P_AYH·σxy   (+ P_AYH·m_vxy)
m_vyx (l.2185–2187): μ_yx·P_AXH·σxy   (+ P_AXH·m_vyx)
m_vxz (l.2218–2220): μ_zx·P_AZH·σxz   (+ …)          [3D]
m_vzx (l.2223–2225): μ_zx·P_AXH·σxz   (+ …)          [3D]
m_vyz (l.2255–2257): μ_zy·P_AZH·σyz   (+ …)          [3D]
m_vzy (l.2260–2262): μ_zy·P_AYH·σyz   (+ …)          [3D]
```

That is the complete enumeration: **9 velocity-adjoint macros (A) + 9 stress-adjoint
macros (B) + 9 velocity memory recursions (C) + 9 stress memory recursions (D)**, each
in its PLAIN and (for A,B) `_PML` variant, gains the `+ Dmodel · Q_sc` coupling. Note
the diagonal normal-stress coupling carries the factor 2 on `δμ` (from `2μ`), and
`(λ+2μ)` couplings split into a `δλ` part and a `2δμ` part.

### 4.3 Full-backward background gradients (two terms each, no third term)

Using the scattered stores of §2.5, the background model gradients in the full
backward are the background-adjoint × background-store term PLUS the
scattered-adjoint × scattered-store term:

```
grad_λ    = Σ_t ( (Σ_i q^bg_σii)·(Σ_k Dv̄_k) + (Σ_i q^sc_σii)·(Σ_k Dv_sc_k) ) · step_ratio
grad_μ    = Σ_t 2·( (Σ_i q^bg_σii·Dv̄_i) + (Σ_i q^sc_σii·Dv_sc_i) )           · step_ratio
grad_μ_ij = Σ_t ( q^bg_σij·S̄_ij + q^sc_σij·S_sc_ij )                          · step_ratio
grad_b_i  = Σ_t ( q^bg_vi·W̄_i  + q^sc_vi·W_sc_i )                             · step_ratio
```

**There is NO third "coupling-coefficient" cross-term (unlike `scalar_born`).** Same
argument as acoustic §4.3: in `scalar_born` the coupling `2·v·scatter·dt²·L(u)` has
explicit dependence on the background `v` (the leading factor), so differentiating the
coupling itself w.r.t. `v` yields an extra term. Here every elastic Born coupling —
`δb_i·W̄_i`, `δλ·ΣDv̄_k + 2δμ·Dv̄_i`, `δμ_ij·S̄_ij` — has **no explicit dependence on the
background models** `λ/μ/μ_ij/b_i`: the background enters ONLY through the stored terms
`W̄_i`/`Dv̄_k`/`S̄_ij` (the background *state*, not the coupling coefficient). That
state-dependence is exactly what the `q^sc · (scattered store)` cross-term already
captures, via the ordinary background-field adjoint recursion the plain nonlinear
`elastic` backward already runs. So each `grad_*` above has **exactly two terms**.

Consequently the full `backward` requires the forward to also checkpoint the SCATTERED
stores (`W_sc_i`, `Dv_sc_k`, `S_sc_ij`) whenever any background model needs grad
(§2.5). `backward_sc` (fixed background) needs neither the coupling substitution of
§4.2 nor the scattered stores — it propagates only `q^sc` and accumulates only the
`grad_δ*` of §4.1.

---

## 5. Second-order (HVP) structure, mirroring `AcousticBackwardFunc`

**Goal:** for the nonlinear `elastic` propagator with a scalar loss `J(m)`,
`m = (λ, μ, b)` (or physical `(vp, vs, ρ)`), compute `H·h` without forming `H`, where
`h` is a model-space perturbation.

### 5.1 Why Born powers the HVP

The first-order backward computes `g(m) = ∂J/∂m` as a correlation of the forward
wavefield with the adjoint wavefield. `H·h = d/dα|₀ g(m+αh)` has two contributions —
the perturbation of the *forward* field by `h` and the perturbation of the *adjoint*
field by `h` — and both are **Born propagations with `scatter = h`**. Hence one Born
forward + one Born backward per HVP, exactly as `scalar`/`acoustic` use their Born
backend as the HVP engine.

### 5.2 The two-pass mechanism (elastic analogue of `AcousticBackwardFunc.backward`)

Mirroring `AcousticForwardFunc.backward` (`acoustic.py` l.1259–1324), replace the
`@no_second_order` C-backend call in `ElasticForwardFunc.backward` with a nested
differentiable `Function`, `ElasticBackwardFunc`:

- **`ElasticBackwardFunc.forward`** runs the ordinary first-order elastic adjoint
  (via the shared `run_backward` / `run_backward_core` with the elastic `SPEC`), seeding
  the adjoint velocity/stress fields **directly** from the output-wavefield gradients
  `grad_wavefields` (no negation), with the elastic memory-variable aux swap (§5.3) as
  the only ping-pong. Returns the first-order `grad_models`, `grad_sources`, and final
  adjoint wavefields.

- **`ElasticBackwardFunc.backward`** (`@once_differentiable`) is the two-pass HVP:

  - **Step 1 — Born forward.** Run `ElasticBornForwardFunc.forward` (called directly,
    not `.apply`, so no graph is built) with background `= m` (models
    `.detach().requires_grad_(True)` to *force* both background and scattered store
    checkpointing) and `scatter = h` (i.e. `(δλ, δμ, δμ_ij, δb_i) =` chain-rule image of
    `h` via `prepare_scatter_parameters`). The background fields recompute the forward
    wavefield; the scattered fields give the effect of `h` on the forward wavefield.
    Record the scattered field at the original receivers (feeds the adjoint-source term
    of `H·h`); background receivers empty. `h` is zero-filled where a model/source/
    wavefield direction is absent.

  - **Step 2 — Born full backward.** Run `ElasticBornForwardFunc`'s full `backward`
    (via `run_backward_core` with the Born `SPEC`, reusing Step-1's storage manager)
    with the ORIGINAL adjoint source `grad_r` (velocity/stress data residuals) as the
    *scattered* adjoint source and the ORIGINAL `grad_wavefields` seeding the *scattered*
    adjoint field, while the *background* adjoint field is seeded at zero and carries
    `h`'s effect through the Born coupling (§4.2). Their combination accumulates the
    second-order model gradient `H·h` (`state.grad_models[:n]`), plus HVP source and
    initial-wavefield terms.

  Field/seed assignment mirrors acoustic exactly: scattered adjoint = ordinary
  backprop (sourced by `grad_r`, seeded by `grad_wavefields`); background adjoint =
  zero-seeded, coupling-driven. Return: `None` for the leading inputs, then grad wrt
  `grad_receivers` (= the Step-1 scattered receiver output), `H·h` for the models, HVP
  sources, grad wrt `grad_wavefields` (= Step-1 scattered forward wavefields), and HVP
  initial wavefields.

### 5.3 No negate hook — only the memory-variable aux swap

**Verified from `ElasticForwardFunc.backward`: there is no `grad_wfp`-style negation
hook** (that scheme is specific to scalar/scalar_born's two-time-level `wfc`/`wfp`
storage). Elastic's first-order backward seeds the adjoint `(v_i, σ_ij)` fields
directly from `grad_wavefields`, and the only ping-pong is the PML **memory-variable
aux swap** — but only on ONE of the two memory-variable families. Per the §4.2(C)/(D)
site enumeration: the `m_sigma*` recursions (§4.2(C), the *velocity*-adjoint block's
memory variables — `m_sigmazzzn`, `m_sigmayzyn`, …) are the ones double-buffered
`m_*`/`m_*n` and swapped by `run_backward` each odd-length chunk. The `m_v*`
recursions (§4.2(D), the *stress*-adjoint block's memory variables — `m_vzz`, `m_vyy`,
…, `elastic.c` lines 2112–2262) are updated **in place**, with no swap. **A future
kernel implementer must not swap `m_v*` in the Born two-field HVP pass** — only the
`m_sigma*` blocks of both the background and scattered fields get the aux-swap
treatment (mirroring `_acoustic_born_swap`, which swaps both fields' psi blocks; here
"psi" corresponds to the `m_sigma*` family only). No negate hook is added for either
field.

### 5.4 Correspondence table (acoustic → elastic)

| acoustic (`AcousticBackwardFunc`)        | elastic (`ElasticBackwardFunc`)                                  |
|------------------------------------------|------------------------------------------------------------------|
| models `(K, B_z, B_y, B_x)`              | models `(λ, μ, μ_zy, μ_zx, μ_yx, b_z, b_y, b_x)`                 |
| `h → (δK, δB_d)` via jvp(`prepare_models`)| `h → (δλ, δμ, δμ_ij, δb_i)` via jvp(`prepare_parameters`) + λ/μ passthrough |
| couplings `−dt·δK·div̄`, `−dt·δB_d·term̄` | `δλ·ΣDv̄+2δμ·Dv̄_i`, `δμ_ij·S̄_ij`, `δb_i·W̄_i`                    |
| adjoint source = residuals for `p, v_d`  | adjoint source = residuals for `v_z,v_y,v_x` + pressure          |
| aux swap = psi block; NO negate          | aux swap = `m_v*`/`m_sigma*` blocks; NO negate                    |
| Born fwd + Born bwd = `H·h`              | Born fwd + Born bwd = `H·h`                                       |

### 5.5 Wiring change required (replaces the `no_second_order` guard) — DONE

`ElasticForwardFunc.backward` no longer uses `@no_second_order`; it builds
`ElasticBackwardFunc.apply(...)` (as `AcousticForwardFunc.backward` builds
`AcousticBackwardFunc`), whose `@once_differentiable` `backward` is the §5.2
double-backward. The `no_second_order` guard is removed from elastic. The
double-backward drives the compiled Born backend
(`elastic_born_compiled_core` + `elastic_born_backward_compiled`) rather than
the eager driver, and raises `NotImplementedError` (→ `backend="eager"`)
under M-PML, where the compiled Born backend has no `_mpml` variant.

---

## 6. Forward-mode (`jvp`) via Born, mirroring `AcousticForwardFunc.jvp`

Replace `ElasticForwardFunc.jvp = unsupported_jvp` (l.1630) with a Born-routed rule
mirroring `_acoustic_forward_jvp`. With the primal inputs saved via `save_for_forward`
in `setup_context`, the tangent map is:

```
(dlamb, dmu, dbuoyancy) tangent → (δλ, δμ, δμ_ij, δb_i) via prepare_scatter_parameters → Born scatter
source tangents                → Born scattered sources (pre-scaling tangents per §5-src below)
initial-wf tangents            → Born scattered initial wavefields (all 13/27 components)
```

One `ElasticBornForwardFunc.forward` call — invoked directly on `no_grad` tensors at
base dispatch level (`_base_level_dispatch()` clearing the functorch stack, so the
ctypes backend sees real storage and builds no graph), `storage_mode_str="none"` —
yields primal `= elastic(...)` (Born background outputs) and tangent `= elastic_born`
scattered outputs together:

```
jvp(elastic, (λ,μ,b); (δλ,δμ,δb)).tangent = elastic_born(dlamb=δλ, dmu=δμ, dbuoyancy=δb).scattered_output
jvp(elastic, …).primal                     = elastic(λ,μ,b) = Born background output
```

Guards (mirroring acoustic): raise `NotImplementedError` on nested functorch depth > 1;
warn (`_model_tangent_in_pml`) when a model tangent is nonzero inside the PML band.

**Non-PML boundary-face guard — not applicable yet.** Acoustic's jvp additionally
raises on any non-PML boundary face (`boundary="free"/"rigid"/"periodic"`, Task G).
`elastic` has **no `boundary` argument** at all today (Task G's first-class boundary
faces are a scalar/acoustic-only feature, not yet extended to elastic), so this guard
is currently vacuous for elastic and is **dropped from the initial `jvp`
implementation**. Note it here only as future-proofing: if/when `elastic` gains a
`boundary` argument, `ElasticForwardFunc.jvp` must add the matching
`NotImplementedError` guard at that time (Born does not linearise a non-PML boundary
fill any more than it linearises the PML profile).

**Source-cell tangents (§5-src) — the elastic asymmetry.** Because the forward
pre-scales sources (§1.2), differentiating the pre-scaling gives the scattered source:

```
velocity/force source tangent:  dt · ( δb_i · f_i  +  b_i · δf_i )      injected into δv_i
pressure source tangent:         − dt · δf_p                             injected into all δσ_ii
```

The velocity tangent carries the buoyancy model factor (exactly like acoustic's
`dt·(δB_d·f + B_d·δf)`); the **pressure tangent has NO model factor** (`−dt · δf_p`
only) because the elastic pressure source is pre-scaled by `−dt` alone — unlike
acoustic's `dt·(δK·s + K·δs)`. As in acoustic, these pre-scaled tangents are produced
automatically by autograd through `elastic`'s differentiable source preprocessing, so
they inject directly with no double counting.

**PML caveat (identical to scalar/acoustic).** Born does not linearise the PML
profile's dependence on the model (through `max_vel`/absorber design), so the jvp is
exact only where the tangent is zero in the PML region (or `pml_width = 0`); a warning
is emitted otherwise.

---

## 7. Elastic-specific complications

### 7.1 IVM free surface — scatterers automatically vanish in vacuum

The Immersed Vacuum Method encodes the free surface **entirely in the models**:
`lamb = mu = buoyancy = 0` in vacuum cells, with `pml_width = 0` on that face
(`elastic` docstring l.561–572). `prepare_parameters` turns those zeros into correct
coefficients via `torch.where` masks (elastic.py l.187–268): a shear node whose 4-cell
harmonic stencil touches any vacuum cell is masked to `μ_ij = 0`; a velocity node whose
2-cell density average is (near) zero is masked to `b_i = 0`, while a node straddling
one solid + one vacuum cell gets `b_i = 2/(ρ_solid + 0) = 2·b_solid` (the arithmetic
density average naturally halving the surface density).

**Born inherits this for free.** Because `prepare_scatter_parameters` is the
forward-mode derivative of *this same masked composition*, the scattering models
`δμ_ij`, `δb_i` are **automatically zero wherever the corresponding prepared model is
masked to zero** (the mask is a piecewise-constant `torch.where` selector, so its jvp
propagates the same indicator): a scatterer in a vacuum cell produces no scattered
source. The "masked-average JVP" also correctly linearises the vacuum-*adjacent* nodes
(the solid-side node straddling the surface, `b_i = 2·b_solid`), which is exactly where
a naive fixed-stencil δ would get the surface physics wrong. So the free surface needs
NO special Born handling beyond routing the scatter through
`prepare_scatter_parameters`. **Validity:** Born is a first-order interior
perturbation — perturbations must keep vacuum cells vacuum (i.e. `dlamb/dmu/dbuoyancy`
are zero in vacuum); a perturbation that *moves* the free surface (creates/removes
vacuum) is outside the linearisation (the mask indicator is not differentiated).

### 7.2 M-PML (CFS-M-PML) — keep the `_mpml` machinery; compile `_mpml` variants of all three entry points

The elastic kernel's M-PML is selected by `DW_MPML`, which makes `ELASTIC_VARIANT(name)`
append `_mpml` to the symbol and switches the `P_*` profile accessors to full-grid
arrays and the `DW_PML_SKIP` region test to interior-only (elastic.c l.71–184).
`elastic_born.c` must keep this machinery verbatim and **compile `_mpml` variants of all
three Born entry points** (`forward`, `backward`, `backward_sc`): build each once with
`DW_MPML` undefined and once defined; the `_mpml` suffix, full-grid `P_*` loads, and
union-of-strips `DW_PML_SKIP` follow automatically from the macros. The coupling
substitution of §4.2 applies inside the `_PML` macro forms regardless of the `_mpml`
variant (the coupling reads the same adjoint/scatter fields; only the `P_*` accessor
implementation differs). Dispatch is via `backend_extra="_mpml"` through
`run_forward_engine`/`run_backward` (as elastic already does; `mpml_fraction > 0` selects it).
**Validity:** as with the free surface, Born is valid for interior perturbations only;
a perturbation inside the M-PML band is not linearised (§6 PML caveat).

### 7.3 Doubled wavefields (26 in 2D, 54 in 3D) and the `_sc` initial-wavefield API

The elastic wavefield set is 4 (1D) / 13 (2D) / 27 (3D) components:
velocities `v_i`, normal stresses `σ_ii`, shear stresses `σ_ij`, velocity-PML memory
vars `m_v*` (9 in 3D), and stress-PML memory vars `m_sigma*` (9 in 3D)
(`elastic_wavefield_names`). Born **doubles** this — a background field and a scattered
field of every component — giving **26 (2D) / 54 (3D)** wavefields, exactly as
`acoustic_born` doubles the acoustic set. Only the *physical* velocity and stress
components of each field are exposed as public initial-wavefield keyword
arguments -- BOTH a background set (`vz_0`, `sigmazz_0`, …) AND a scattered set
(`vz_sc_0`, `sigmazz_sc_0`, …); the `m_v*`/`m_sigma*` PML memory variables of
both fields are not keyword arguments at all, and are reachable only through
the `initial_state` bundle (as part of the folded `auxiliary_state`). Outputs
return the final background wavefields, one opaque `auxiliary_state` holding
both fields' PML memory variables, then the final scattered wavefields, then
background receiver amplitudes, then scattered receiver amplitudes (the
"background data before scattered data" convention). The eager core's
`wavefields_flat = args_list[-27:]`/`[-13:]`/`[-4:]` slicing must become the doubled
slice for the Born spec.

---

## 8. Planned public API

Mirrors `acoustic_born` (background + scattered sources/receivers, both fields'
initial wavefields) and `elastic` (per-component velocity force sources + a pressure
source, `mpml_fraction`).

```
def elastic_born(
    lamb, mu, buoyancy, grid_spacing, dt,
    dlamb=None, dmu=None, dbuoyancy=None,            # scattering potentials (Lamé/buoyancy space)
    # background + scattered share source *locations* (§5-src); amplitudes given once:
    source_amplitudes_z=None, source_locations_z=None,
    source_amplitudes_y=None, source_locations_y=None,
    source_amplitudes_x=None, source_locations_x=None,
    source_amplitudes_pressure_rate=None, source_locations_pressure_rate=None,
    # scattered-field receivers (the Born data) and, optionally, background receivers:
    sc_receiver_locations_z=None, sc_receiver_locations_y=None,
    sc_receiver_locations_x=None, sc_receiver_locations_p=None,
    bg_receiver_locations_z=None, bg_receiver_locations_y=None,
    bg_receiver_locations_x=None, bg_receiver_locations_p=None,
    accuracy=4, pml_width=20, pml_freq=None, mpml_fraction=0.0,
    max_vel=None, courant_number=None, survey_pad=None,
    # background initial wavefields (physical velocity/stress components only):
    vz_0=None, vy_0=None, vx_0=None, sigmazz_0=None, …, sigmaxx_0=None,
    # scattered initial wavefields (physical components only): vz_sc_0, …, sigmaxx_sc_0
    vz_sc_0=None, …, sigmaxx_sc_0=None,
    # the m_v*/m_sigma* PML memory variables of both fields are not keyword
    # arguments; they are reachable only through initial_state/auxiliary_state.
    initial_state=None, origin=None, nt=None,
    model_gradient_sampling_interval=1,
    freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False,
    forward_callback=None, backward_callback=None, callback_frequency=1,
    backend="compiled",
    storage_mode="device", storage_path=".", storage_compression=False,
    location_interpolation="auto", location_interpolation_halfwidth=4,
) -> PropagatorOutput:
    ...
```

Plus an `ElasticBorn(torch.nn.Module)` wrapper storing `lamb, mu, buoyancy,
grid_spacing` (and `dlamb, dmu, dbuoyancy` if desired), matching `Elastic`/`AcousticBorn`.

**Front-end composition** (mirroring `acoustic_born`):
1. `setup_propagator` pads/survey-crops `[lamb, mu, buoyancy, dlamb, dmu, dbuoyancy]`.
2. `bg_models = [lamb_p, mu_p] + prepare_parameters(mu_p, buoyancy_p)` = `[λ, μ, μ_zy,
   μ_zx, μ_yx, b_z, b_y, b_x]`.
3. `sc_models = [dlamb_p, dmu_p] + prepare_scatter_parameters(mu_p, buoyancy_p, dmu_p,
   dbuoyancy_p)` = `[δλ, δμ, δμ_zy, δμ_zx, δμ_yx, δb_z, δb_y, δb_x]`.
4. Shared-source scaling: background velocity amp `= amp · gather(b_i) · dt`, scattered
   virtual "source-cell" amp `= amp · gather(δb_i) · dt`, both at the shared
   `sources_i`; pressure background `= −amp·dt`, scattered `= −amp_tangent·dt` (no model
   factor). Where an amplitude is absent the scattered slot mirrors the background slot.
5. Call `ElasticBornForwardFunc.apply(...)`; downsample receiver outputs.

**`PropagatorSpec`** (`name="elastic_born"`, `staggered=True`):
- `model_names(ndim)` = background prepared models then scattered prepared models
  (`[λ, μ, μ_ij…, b_i…, δλ, δμ, δμ_ij…, δb_i…]`).
- `wavefield_names(ndim)` = elastic background wavefields (§7.3) followed by scattered
  ones (doubled: 26/54).
- `n_source_types = ndim + 1`; `n_receiver_types = 2·(ndim + 1)` (background +
  scattered) — or model background/scattered receiver sets like `acoustic_born`.
- `output_names` = doubled reordered wavefields + background then scattered receiver
  amplitudes `{z,y,x,p}`.
- Scatter-model flag indices analogous to elastic's `c_model_flag_indices=[0,1,-1]`
  (the independently-differentiable owners: `lamb`, `mu`, raw `buoyancy` on the
  background side and `dlamb`, `dmu`, raw `dbuoyancy` on the scattered side).

Differentiability: outputs differentiable w.r.t. `lamb, mu, buoyancy, dlamb, dmu,
dbuoyancy`, source amplitudes, and initial wavefields — all first-order gradients need
no new hand-derivation beyond the kernel's `grad_δλ/grad_δμ/grad_δμ_ij/grad_δb_i` (§4),
because the `prepare_scatter_parameters` and `prepare_parameters` chain rules are
differentiable Python (§3).

**Note — this deliberately EXCEEDS the `acoustic_born` precedent.** `acoustic_born`'s
public API does not make background (`v`, `rho`) gradients a first-class public
differentiability promise: background gradients there are produced only as a
Function-level *internal* capability of the full two-field `backward` (consumed by
`AcousticBackwardFunc`'s HVP, §5, and by the general autograd machinery when the
background happens to require grad), not as a documented/tested public guarantee of
`acoustic_born(...)`. This document's claim that `elastic_born` differentiability
w.r.t. `lamb, mu, buoyancy` is public and first-class is thus an **intentional
extension** beyond that precedent, not an oversight — flagged explicitly so the
reviewer can weigh in. When implementing, `ElasticBornForwardFunc.backward` must
mirror `AcousticBornForwardFunc`'s dispatch logic (`acoustic_born.py` lines ~843–857:
dispatch to the full `backward` vs. the cheaper `backward_sc` based on whether any
background-side input — model, source amplitude, or initial wavefield — needs grad)
and its conditional scattered-store checkpointing flags (lines ~793–806: background
stores saved whenever any model needs grad; scattered stores saved only when a
*background* model needs grad), for consistency with the rest of the Born family.

---

## 9. Planned kernel filenames and symbols

- **`src/deepwave/elastic_born.c`** and **`src/deepwave/elastic_born.cu`** (kept in
  lockstep; `.cu` GPU-tested by the maintainer before release).
- Symbol convention (mirroring `elastic_iso_*` and `acoustic_born_iso_*`):
  `elastic_born_iso_<ndim>d_<accuracy>_<dtype>_<name>[_mpml]_<device>`, with
  `name ∈ {forward, backward, backward_sc}` — e.g.
  `elastic_born_iso_2d_4_float_forward_cpu` and its `…_forward_mpml_cpu` M-PML variant.
  The `_mpml` infix is inserted by `ELASTIC_VARIANT` before the ndim/accuracy suffix,
  exactly as in `elastic.c`.
  - `forward`: single coupled pass, background + scattered fields (§2.5); conditionally
    checkpoints the scattered stores when a background model needs grad (§2.5/§4.3).
  - `backward`: full two-field adjoint, coupling substitution at every §4.2 site,
    accumulating background *and* scattered model gradients (§4.3) — powers HVP and
    background gradients.
  - `backward_sc`: scattered-only adjoint, accumulates only `grad_δλ/grad_δμ/
    grad_δμ_ij/grad_δb_i` (§4.1), no coupling, no scattered stores.
- Registered under backend name `"elastic_born"` in `backend_utils`
  (`get_backend_function("elastic_born", ndim, name, …)`), mirroring `"acoustic_born"`.
- **`CMakeLists.txt`** gains the `elastic_born` translation units in the same
  `ndim × accuracy × dtype × device` specialisation matrix as `elastic`. *(Done
  for the default (non-M-PML) variant; there is no compiled M-PML Born
  variant, which falls back to `backend="eager"`.)*

---

## 10. Planned validation gates

Complete only when all pass (CPU C backend + eager backend; `.cu` compiled, GPU-run by
the maintainer). Names mirror `tests/test_acoustic_born.py` / `tests/test_elastic.py`.

1. **Born linearity (per parameter)** — the scattered output is exactly linear in each
   of `(dlamb, dmu, dbuoyancy)` separately and jointly: scaling a scatterer by `α`
   scales the scattered receivers by `α`; all-zero scatter gives zero scattered output
   and background output byte-identical to `elastic(λ,μ,b)`.
2. **Born == directional FD w.r.t. lamb / mu / dbuoyancy** — scattered receivers equal
   `(elastic(m+ε h) − elastic(m−ε h)) / (2ε)` for interior perturbations `h`, for
   `dlamb`, `dmu`, `dbuoyancy` independently and mixed; background receivers equal
   `elastic(λ,μ,b)`.
3. **jvp == Born** — `torch.func.jvp(elastic, (λ,μ,b), (δλ,δμ,δb))` tangent equals the
   `elastic_born` scattered output (eager ~1e-12; compiled C ~1e-6 fast-math tol);
   legacy `forward_ad` path agrees; the PML-boundary caveat warns; nested-transform and
   non-PML-boundary cases raise. **Status: DONE** for both the eager path and the
   compiled `elastic` propagator's `ElasticForwardFunc.jvp` (which currently
   computes the tangent through the eager Born driver): the two new
   `tests/test_elastic_born.py::test_jvp_compiled_backend_matches_born[2,3]` and
   `::test_jvp_compiled_backend_guards` verify jvp == Born to machine precision on
   the compiled backend, the PML-band-tangent warning, and the nested-transform
   `NotImplementedError`. (When the compiled `elastic_born` backend lands the
   tolerance moves to the ~1e-6 fast-math band.)
4. **HVP == FD-of-gradient** — the second-order path gives `H·h` equal to
   `(∇J(m+ε h) − ∇J(m−ε h)) / (2ε)` for `J` a scattered/background data misfit and `h`
   an interior perturbation, for `(λ,μ,b)` and cross terms. Replaces the current
   elastic `test_second_order_raises`.
5. **gradcheck + gradgradcheck** — `torch.autograd.gradcheck` on `elastic_born` w.r.t.
   `lamb, mu, buoyancy, dlamb, dmu, dbuoyancy`, source amplitudes, and (background +
   scattered) initial wavefields (float64, accuracies 2/4/6/8, `step_ratio` 1 and >1,
   batched models, `pml_width=0` and asymmetric); `torch.autograd.gradgradcheck` on
   `elastic` exercising the HVP path. Plus ≥1 absolute FD gradcheck on a tiny problem as
   ground truth; cross-backend (C vs eager) parity on forward, gradient, and HVP.
6. **Reciprocity** — source↔receiver reciprocity of the Born scattered response
   (accounting for velocity/stress/pressure source–receiver duality on the staggered
   grid).
7. **Free-surface interaction** — Born linearity + Born == FD hold with an IVM free
   surface present (vacuum cells + `pml_width=0` on that face), for interior
   perturbations; scatterers in vacuum cells produce zero scattered output (§7.1).
8. **M-PML interaction** — Born linearity + Born == FD hold with `mpml_fraction > 0`; the
   `_mpml` Born variants match the default variants where the M-PML is inactive; the
   union-of-strips region behaviour is preserved (interior perturbation only).
9. **ABI additive** — `abi_reference.json` extended with the additive `elastic_born_*`
   symbols; no drift to existing `elastic_*` symbols.
10. **CUDA lockstep** — `elastic_born.cu` compiles clean under nvcc across all
    `ndim × accuracy × dtype × {default, mpml}` specialisations; GPU-run parity with the
    CPU C backend on forward, gradient, and HVP (maintainer-run).

Supporting checks: byte-identity of background outputs vs plain `elastic` when the
scatter is zero; `IGNORE_LOCATION` handling; storage-mode (`device/cpu/disk/none`) and
checkpointing parity (including the conditional scattered-store checkpointing);
callback parity.

---

## 11. Summary of decisions (for the reviewer)

- **Forward system:** `∂v_i/∂t = b Σ_j ∂_j σ_ij (+bf_i)`, `∂σ_ii/∂t = λΣ_k∂_k v_k +
  2μ∂_i v_i`, `∂σ_ij/∂t = μ_ij(∂_i v_j + ∂_j v_i)`; staggered leapfrog with CPML/M-PML
  (`elastic.c`). IVM free surface encoded entirely in the models.
- **Born:** scattered `(δv, δσ)` obeys the same operator with background
  `(λ, μ, μ_ij, b_i)`, virtual sources `δb_i·W̄_i` (velocity), `δλ·ΣDv̄_k + 2δμ·Dv̄_i`
  (normal σ), `δμ_ij·S̄_ij` (shear σ), where `W̄/Dv̄/S̄` are exactly the eight forward
  stores `dv{i}dbuoyancy_store_1` / `dv{k}d{k}_store_1` /
  `dv{j}d{i}_plus_dv{i}d{j}_store_1`.
- **Parameterisation (and why):** the kernel scatters in `δ` of the *prepared* model
  set `(δλ, δμ, δμ_ij, δb_i)`; the public API is Lamé/buoyancy space
  `(dlamb, dmu, dbuoyancy)`, and `prepare_scatter_parameters =
  torch.func.jvp(prepare_parameters, …)` (+ λ/μ passthrough) does the chain rule in
  differentiable Python. Chosen because (1) these are the perturbations linear in the
  operator, (2) it matches how `elastic` already prepares its models (same layout, node
  alignment, PML profiles, IVM masking), (3) the adjoint is the same imaging condition
  as the nonlinear model gradient, and the physical/VJP chain rule is autograd's job —
  it also gives node alignment AND IVM vacuum masking for free.
- **Adjoint coupling checklist:** every `λ·Qσ`, `2μ·Qσ`, `μ_ij·Qσij`, `b_i·Qv` product
  gains `+Dmodel·Q_sc`, at all 36 sites: 9 velocity-adjoint macros (A) + 9 stress-
  adjoint macros (B), each in PLAIN and `_PML` forms, + 9 velocity memory recursions
  (C) + 9 stress memory recursions (D). Full-backward background gradients have exactly
  TWO terms each (bg adjoint × bg store + sc adjoint × sc store); NO third coupling-
  coefficient term (the couplings have no explicit λ/μ/b dependence).
- **Elastic-specific complications:** (1) IVM free surface — scatterers vanish in
  vacuum and the masked-average JVP handles vacuum-adjacent nodes automatically, because
  the scatter is the jvp of the masked `prepare_parameters`; Born valid for interior
  perturbations that keep vacuum vacuum. (2) M-PML — keep `ELASTIC_VARIANT`/`P_*`/
  `DW_PML_SKIP`, compile `_mpml` variants of all three entry points; Born valid for
  interior perturbations only. (3) Doubled wavefields 26 (2D)/54 (3D) and a `_sc`
  initial-wavefield API for every component. (4) Pressure source tangent is `−dt·δf_p`
  ONLY (no model factor) — unlike acoustic — while velocity source tangent is
  `dt·(δb_i·f + b_i·δf)`.
- **HVP:** one Born forward + one Born backward per `H·h`, mirroring
  `AcousticBackwardFunc`; requires `ElasticForwardFunc.backward` to build a nested
  differentiable `ElasticBackwardFunc` and removal of the `no_second_order` guard; NO
  negate hook (only the `m_v*`/`m_sigma*` memory-variable aux swap, applied to both
  fields).
- **jvp:** Born-based `ElasticForwardFunc.jvp` replacing `unsupported_jvp`, mirroring
  `_acoustic_forward_jvp`, with the same PML caveat and the pressure/velocity source-
  tangent asymmetry above.
- **Files:** `elastic_born.c`/`.cu`; symbols
  `elastic_born_iso_<ndim>d_<accuracy>_<dtype>_<forward|backward|backward_sc>[_mpml]_<device>`;
  backend name `"elastic_born"`.
- **Gates:** Born linearity per parameter; Born == directional FD w.r.t.
  lamb/mu/dbuoyancy; jvp == Born; HVP == FD-of-gradient; gradcheck + gradgradcheck;
  reciprocity; free-surface interaction; M-PML interaction; ABI additive; CUDA lockstep.

**File path:** `docs/elastic_born_derivation.md` (this document).

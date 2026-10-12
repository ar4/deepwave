# Compiled-layer spike — measured report

Campaign 7, task C3. Spike of the synthesised design from the C1/C2 blind
three-panel comparison (`MAINTAINER_ACTIONS.md` item 32). Evaluation only:
nothing here is proposed for adoption, and nothing under `src/deepwave/` was
modified.

Everything below is measured on Box C (24-core x86-64, 31 GB RAM,
`cuda:0` = RTX A4000 16 GB, `cuda:1` = RTX 4000 Ada 20 GB — note this is the
opposite ordering to the one recorded in `ORCHESTRATION7.md` §0.2), torch
2.13.0+cu130, gcc 14, nvcc 13.3, box otherwise idle.

## 1. What was built

```
exploration/compiled_layer_spike/
  dwgen/                  the generator
    ir.py                 symbolic IR (fields, FD operators, PML profiles)
    spec.py               PhysicsSpec: a time step as an ordered action list
    adjoint.py            adjoint by structural transposition
    model.py              buffering/ordering plan, interior-vs-PML split
    symcheck.py           doubly-derived adjoint check (SymPy)
    abi.py                the flat extern "C" ABI, shared by emitters + binding
    emit_c.py             plain-C emitter (CPU)
    emit_cuda.py          CUDA device code + driver-API host driver
    build.py              generate -> verify -> compile
    specs/scalar.py       106 lines
    specs/elastic.py      106 lines
  runtime/
    dw_pool.h             pthread thread pool (no OpenMP)
    dw_cuda.h/.c          libcuda.so.1 driver-API shim (dlopen, no libcudart)
  spike/
    loader.py             ctypes binding, signature derived from the spec
    ops.py                torch.library.custom_op + register_autograd
    frontend.py           reuses Deepwave's Python front-end
  tests/test_parity.py    24 parity cases
  build_all.py measure.py run_benchmark.py check_determinism.py check_3d.py
  demo_injected_bug.py
```

Run: `python build_all.py`, `pytest tests/test_parity.py`,
`python run_benchmark.py`, `python measure.py`, `python demo_injected_bug.py`,
`python check_determinism.py`, `python check_3d.py`.

### Which parts of the synthesised design are actually present

| design element | status |
|---|---|
| code generation from a small symbolic per-physics spec | **yes** — 106 lines of spec per family, no hand-written kernel source |
| flat `extern "C"` ABI, zero libtorch/ATen/`Python.h` | **yes** — `readelf -d` shows `DT_NEEDED: [libc.so.6]` and nothing else |
| bound via `torch.library.custom_op` + `register_autograd` | **yes**, and `torch.compile` traces through it (§4) |
| CUDA via the driver API + fatbin, not libcudart | **yes** — `dlopen("libcuda.so.1")`, `cuModuleLoadData`, `cuLaunchKernel` |
| doubly-derived adjoint check, automatic, whole matrix | **yes** — §3 |
| hard interior/PML split, PML structurally absent from the fast path | **yes** — separate functions/kernels; PML profile pointers are not even in scope in the interior kernel |
| kernel library never allocates | **yes** — every buffer is a torch tensor passed as `data_ptr()` |
| deterministic gradient reduction, no `atomicAdd` | **yes** structurally (per-shot private buffers, one thread owns each cell, cross-shot reduction by torch); see §6 for what could and could not be demonstrated |
| no OpenMP | **yes** — hand-written pthread pool, measured scaling in §5 |
| GPU tap loop **rolled** by default (panel B tie-breaker) | **NO — not implemented.** See §7.1 |
| CPU ISA multiversioning (AVX2/AVX-512 + CPUID dispatch) | **NO — not implemented.** Single `-march=native` build. See §7.1 |

### Scope actually covered

`scalar` and `elastic` (isotropic, velocity-stress), **2-D**, accuracy 2 and 4,
float32 and float64, CPU + both GPUs, standard CPML only. Differentiable
inputs: `v` for scalar; `lamb`, `mu`, `buoyancy` for elastic; source amplitudes
for both. A bonus 3-D `scalar` check is in §8.

**Cut explicitly** (each would be real work, none was attempted): M-PML, free
surface, reflecting/rigid/periodic faces, Born/scattering variants,
initial-wavefield differentiability, double backward, `vmap`, per-parameter
batching flags (the ABI has one shared `par_batched`), `storage_mode` /
compression / disk snapshots / checkpointing callbacks, Hicks interpolation
(the reused front-end supports it but it is untested here), accuracy 6 and 8
(the mechanism is a generator parameter and generates fine; only 2 and 4 were
*verified*).

## 2. Correctness: parity against the incumbent

`tests/test_parity.py` drives the real `deepwave.scalar` / `deepwave.elastic`
and the generated kernels through the *same* Python front-end on the same
inputs, and compares receiver traces and `torch.autograd.grad` outputs.

**24 / 24 pass** = 2 families × 2 accuracies (2, 4) × 2 dtypes × 3 devices
(cpu, cuda:0, cuda:1).

Representative float64 relative errors (max-norm, 2 shots, nt=140–480,
50×55 grid, `pml_width=10`):

| family | forward | grad (model) | grad (sources) |
|---|---|---|---|
| scalar 2-D acc 2/4 | 6e-15 | 3e-14 | 1e-14 |
| elastic 2-D acc 2/4 | 1e-15 | 1e-15 (lamb, mu), 5e-15 (buoyancy) | 3e-15 |

Bitwise equality is **not** achievable and was not attempted: both sides are
built with `-ffast-math` / `--use_fast_math` (the incumbent's own default,
`DW_FAST_MATH=ON`), so floating-point reassociation legitimately differs.

Float32 tolerances were calibrated against the **incumbent's own float32 vs
float64 self-discrepancy** on the same survey, measured at 8.0e-7–1.7e-5
(forward) and 1.6e-5–2.2e-5 (model gradient). The spike's float32-vs-incumbent
disagreement is 3.4e-6–4.6e-6 (forward) and 2.6e-5–3.2e-5 (model gradient),
i.e. the same order — the float32 arithmetic noise floor for this problem, not
a defect. Tolerances in the test are set at 2e-5 (forward) / 1e-4 (gradients)
with that reasoning written into the file.

## 3. The doubly-derived adjoint check

Two genuinely independent derivations of the same object:

* **Path A (what is emitted).** `adjoint.py` transposes the forward time step
  *structurally*: each output is flattened into terms `c(i) * f(i+off)`, and
  the transpose is obtained by translating each term by `-off`. No
  differentiation is involved, only index algebra. The whole time step is an
  ordered list of actions (stages, receiver records, source injections), so
  the adjoint step is the reversed list of transposed actions.
* **Path B (the check).** `symcheck.py` instantiates the forward update on a
  small symbolic grid — a SymPy symbol per (field, cell), per (parameter,
  cell), per PML-profile entry, per runtime scalar — and differentiates it
  with `sympy.diff` to get the Jacobian of the time-step map explicitly. The
  emitted adjoint must be that Jacobian's transpose entry by entry, and the
  emitted model-gradient accumulation must equal
  `sum_i adj_g[i] * d F_g[i] / d p[j]`. Everything is exact rational
  arithmetic over symbols.

It runs at generation time for every (family, ndim, accuracy) emitted, on both
the full-PML and the interior variants, and costs **25.8 s** for the 4
(family, accuracy) pairs in scope (2.5–5.0 s each, `active_size=3`).

### Did it catch anything?

**Yes — twice for real, and 4/4 on deliberately injected bugs.**

Two real bugs were found by it or by the parity tests during development, both
in the generator rather than the physics:

1. **Group scheduling inverted.** The in-place scheduler ordered field groups
   *writer-before-reader* instead of reader-before-writer, so the elastic
   adjoint read already-overwritten PML memory variables. Caught by the
   elastic gradient parity test (grad_lamb/grad_mu wrong by ~2e-4–5e-4,
   grad_buoyancy wrong by 5000×); the symbolic check does not see it because
   it verifies the *expressions*, not the sweep order.
2. **Backward pass clobbering the forward's gradient snapshots.** The emitter
   wrote the stored forward quantities in *both* directions, so the backward's
   own group bodies overwrote them mid-sweep. This is a silent-wrong-gradient
   bug that survived the scalar family entirely (there the clobbering happened
   after the accumulation) and only appeared on elastic, where two stages mean
   one stage's snapshots are destroyed before they are read. Caught by the
   elastic buoyancy gradient parity test, arbitrated by finite differences.

Neither is a physics error, which is the point: the symbolic check makes the
*math* essentially unfalsifiable, so the residual bug surface moves to
scheduling and buffering — which numerical parity tests do catch.

`demo_injected_bug.py` injects four plausible hand-adjoint mistakes into the
derived model (the objects the emitters consume) and re-runs the check:

| injected bug | result |
|---|---|
| sign flipped on one off-centre term of the transposed wavefield update | **CAUGHT** (18 symbolic mismatches) |
| `dt^2` factor dropped from the stored model-gradient quantity | **CAUGHT** (9 mismatches) |
| model gradient correlated against adjoint `p` instead of adjoint `u` | **CAUGHT** (9 mismatches) |
| adjoint PML `b`-profile sampled at the cell instead of the shifted tap | **CAUGHT** (180 mismatches) |

Baseline (no injection): 0 mismatches.

The second and third are the interesting ones: a missing constant scale and a
wrong field pairing are both invisible to a forward-only test and produce a
gradient that still points roughly downhill, which is exactly the class of bug
that survives in an optimiser for a long time.

## 4. PyTorch integration

The op is functional (`mutates_args=()`), takes `list[Tensor]` inputs, and
allocates every buffer it writes through torch, so the caching allocator owns
all device memory. A fake/meta kernel is registered. Measured:
`torch.compile(f)` over a full propagator call traces through the custom op —
forward output **bitwise identical** to eager, gradient relative difference
1.2e-18.

## 5. Performance

`run_benchmark.py`: 512² grid (the project's own "meso" 2-D tier), nt=500
(halved from the tier's 1000 so the elastic gradient snapshots fit the smaller
card), accuracy 4, float32, 1 shot, `pml_width=20`, 12 CPU threads for both
sides, min of 3 timed runs after a warm-up.

| family | device | mode | incumbent | spike | spike/incumbent |
|---|---|---|---|---|---|
| scalar | CPU (12 thr) | forward | 31.5 ms | 119.9 ms | **3.80×** |
| scalar | CPU (12 thr) | fwd+bwd | 246.9 ms | 517.0 ms | **2.09×** |
| elastic | CPU (12 thr) | forward | 125.6 ms | 375.7 ms | **2.99×** |
| elastic | CPU (12 thr) | fwd+bwd | 2705.3 ms | 2305.4 ms | **0.85×** (spike faster) |
| scalar | cuda:0 A4000 | forward | 8.2 ms | 9.6 ms | **1.17×** |
| scalar | cuda:0 A4000 | fwd+bwd | 24.8 ms | 31.1 ms | **1.25×** |
| elastic | cuda:0 A4000 | forward | 18.0 ms | 24.4 ms | **1.36×** |
| elastic | cuda:0 A4000 | fwd+bwd | 61.2 ms | 100.3 ms | **1.64×** |
| scalar | cuda:1 4000 Ada | forward | 13.1 ms | 18.0 ms | **1.38×** |
| scalar | cuda:1 4000 Ada | fwd+bwd | 38.3 ms | 45.4 ms | **1.19×** |
| elastic | cuda:1 4000 Ada | forward | 47.1 ms | 54.5 ms | **1.16×** |
| elastic | cuda:1 4000 Ada | fwd+bwd | 139.1 ms | 198.0 ms | **1.42×** |

(The A4000 beating the newer 4000 Ada on both implementations alike is
consistent across runs and not investigated; it does not affect the ratios.)

CPU thread scaling of the generated scalar forward (512², nt=500, acc 4, f32),
which is the hand-written pthread pool doing the work OpenMP does for the
incumbent:

| threads | 1 | 2 | 4 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|
| spike | 559 ms | 296 | 239 | 138 | 120 | 139 |
| incumbent | 214 ms | — | 62 | — | 31 | 33 |

So the pool scales comparably (4.7× at 12 threads vs the incumbent's 6.9×) and
the gap is mostly single-thread throughput (2.6×).

### What the CPU gap is made of, and one bug worth recording

A **4× CPU speed-up was recovered mid-spike** by finding that GCC was silently
refusing to vectorise the interior stencil. The generated kernels load their
`__restrict` pointers out of a context struct, which GCC honours only via
runtime alias versioning, and its default cap
(`--param=vect-max-version-for-alias-checks=10`) is exceeded by a stencil with
this many distinct arrays — so it emitted scalar code with no diagnostic
beyond `-fopt-info-vec-missed`. Raising the parameter to 200 took the scalar
forward from 2224 ms to 559 ms single-threaded. **This is a portability
landmine for any generated-kernel design**: the same source compiles, passes
every test, and runs 4× slow. A production version should pass the pointers as
`restrict` *function parameters* rather than relying on the versioning cap.

The remaining ~2.6× single-thread gap is understood and mostly structural, not
fundamental:

* **SSA/ping-pong buffering.** The generator models a time step as a linear
  map on the state, so `p' = u` is emitted as a real full-grid copy, where the
  incumbent gets it free by swapping `wfc`/`wfp` pointers. That is roughly two
  extra array streams per step for scalar. Recognising pure-copy updates and
  turning them into a buffer-role rotation is a known, unimplemented
  optimisation.
* **No ISA multiversioning.** One `-march=native` build against the
  incumbent's `DW_TARGET_CLONES` AVX2 clones.
* **Runtime alias checks** left in by the workaround above.
* **Elastic backward is already faster than the incumbent** (0.85×), which
  suggests the generated adjoint is not intrinsically worse — the forward gap
  is where the buffering overhead lives.

GPU is within 1.16×–1.64× — with unrolled kernels (see §7.1), a `pml` kernel
launched over the whole active region with an early-out rather than over
per-face slabs, and no occupancy tuning at all.

## 6. Deterministic gradients (NF9)

The generated gradient path contains no atomic of any kind: each shot writes a
private full-grid gradient buffer, exactly one thread (or one GPU thread) owns
each cell, and the cross-shot reduction is a `torch` `sum(0)`. That is a
structural property of the emitted code, verifiable by reading it.

**However, NF9 did not reproduce in this test.** `check_determinism.py` runs
each backward three times on identical inputs (256², nt=300, 4 shots, shared
model) on both cards: the incumbent's scalar and elastic model gradients came
out **bitwise identical** across runs, as did the spike's. So this spike
provides *no* evidence that the design fixes NF9 — only that it does not
introduce non-determinism, and that NF9 needs a different configuration
(different propagator, batching, or grid) to reproduce than the one used here.
Anyone picking up D4 should not assume the recorded configuration is arbitrary.

## 7. Measured size, build time and lines of code

All figures from `measure.py` (build directory cleared first, `-j4` for gcc,
`-j2` for nvcc).

### Hand-maintained lines

| | lines |
|---|---|
| generator (`dwgen/`, incl. both physics specs) | 2,432 |
| — of which the two physics specs | **212** (106 each) |
| hand-written runtime (`runtime/`: thread pool + CUDA shim) | 331 |
| torch binding (`spike/`: loader, custom_op, front-end) | 739 |
| tests + harness | 742 |
| **total hand-maintained** | **4,244** |

Incumbent, for the same two families: `scalar.c` + `scalar.cu` +
`scalar_kernels.h` + `regular_grid.h` = **2,630 lines**; `elastic.c` +
`elastic.cu` + `elastic_kernels.h` + `staggered_grid.h` + `staggered_bc.h` +
`elastic_line_bc*.h` = **11,277 lines**. Combined **13,907 lines** of
hand-maintained C/CUDA for the two families — against **212 lines of physics
spec** plus a 2,432-line generator shared by all families. Whole-tree
incumbent: 70,796 lines of `*.c`/`*.cu`/`*.h`.

That comparison flatters the spike and should be read with §1's cut list in
hand: the incumbent's 13,907 lines also implement M-PML, free surfaces,
reflecting/periodic faces, 1-D and 3-D, four accuracy orders, disk and
compressed snapshot storage, and callback checkpointing, none of which the
spike has. A fairer reading is that the *stencil-and-adjoint* part — the part
that is 73–91% duplicated between sibling families today — collapses from
thousands of hand-maintained lines per family to about a hundred.

### Generated volume (8 variants: 2 families × 2 accuracies × 2 dtypes, 2-D)

| | |
|---|---|
| generated C (CPU kernels + CUDA host drivers) | 28,736 lines |
| generated CUDA device code | 13,724 lines |
| generated headers | 344 lines |
| **generated total** | **42,804 lines** |

### Build

| step | time |
|---|---|
| generation (all 8 variants) | 0.32 s |
| doubly-derived adjoint check (4 family×accuracy pairs) | 25.82 s |
| gcc `-j4` (CPU kernels + CUDA host drivers) | 17.07 s |
| nvcc `-j2` (`sm_86` + `sm_89` + PTX) | 7.73 s |
| **total cold build** | **~51 s** |

### Binary

| | bytes |
|---|---|
| `libdwspike.so` (CPU kernels + CUDA host drivers + runtime) | 1,943,064 |
| fatbins (8 files, sm_86 + sm_89 + PTX) | 1,948,168 |
| **total shipped** | **3,891,232 (3.7 MB)** |

`readelf -d libdwspike.so` gives `DT_NEEDED: [libc.so.6]`, and nothing else: no
libtorch, no libcudart, no libgomp, no libstdc++. `libcuda.so.1` is `dlopen`ed
at first GPU use.

### Against the incumbent's recorded figures

The incumbent's real numbers (`MAINTAINER_ACTIONS.md` items 31–32) are
**335.1 MB installed / 91.9 MB compressed wheel**, **587 objects / 46m54s
wall, 185.4 CPU-minutes** cold at `-j4`, for **11 propagators × 1/2/3-D × 4
accuracies × 2 dtypes × CPU+CUDA**. The spike's 3.7 MB / 51 s covers **2
propagators × 2-D × 2 accuracies × 2 dtypes**, i.e. roughly 8 of a
~350-variant matrix.

Scaling the spike's per-variant cost naively (486 kB and 6.4 s per variant)
to the incumbent's matrix gives ~170 MB and ~37 minutes — the same order as
the incumbent, **not** an order-of-magnitude win. Two caveats in opposite
directions: the spike emits fully-unrolled GPU kernels where the panels'
design rolls them (which is precisely the binary-size lever, §7.1), and the
spike's variants are missing the boundary-condition and storage machinery that
inflates the incumbent's per-variant code. **The honest conclusion is that this
spike does not demonstrate a binary-size or build-time win, and did not test
the mechanism the panels proposed for getting one.**

## 7.1 Deviations from the synthesised design, stated plainly

* **The GPU tap loop is UNROLLED, not rolled.** The C2 decision named panel
  B's "roll by default on GPU, let the regression gate choose unrolled
  specialisations" as the tie-breaker for the one three-way disagreement, and
  this spike did **not** implement it — the generator fully expands every
  stencil tap into straight-line code on both CPU and GPU. Consequences:
  (a) the GPU timings in §5 are an *upper bound* on a rolled design's speed
  (rolling costs up to 36% per this campaign's own measurement), so the
  spike's GPU numbers flatter the design; (b) the binary-size argument that
  motivates rolling is untested, which is why §7 cannot claim a size win. The
  reason is time budget, not a technical obstacle: the generator's tap
  expansion is one function, and a rolled variant would emit a coefficient
  table plus a loop.
* **No CPU ISA multiversioning.** One `-march=native` build; the incumbent
  builds AVX2 function clones with runtime dispatch. `-march=native` is also
  not shippable, so the CPU numbers assume a build tuned to the host.
* **The Python front-end is Deepwave's own**, imported unchanged. The spike
  replaces the compiled kernel layer and its binding only. This is deliberate
  (it isolates the kernel in the parity comparison) but it means the spike
  says nothing about whether `_engine.py`'s marshalling glue would shrink.
* **Elastic stores 6 gradient snapshot fields where the incumbent stores 5.**
  The generator deduplicates structurally identical stored quantities (the
  `lamb` snapshot is shared by both normal stresses) but does not factor the
  `2*dt*dvydy` / `2*dt*dvxdx` pair the way a human did. That is ~20% more
  backward-pass snapshot memory for elastic.
* **One `par_batched` flag for all model parameters**, where the incumbent
  tracks batching per parameter.

## 8. Bonus: 3-D scalar

3-D was explicitly out of scope. Since the generator is dimension-generic, it
was tried anyway with no code changes beyond `scalar.make(3)`:

```
cpu      3-D scalar acc4 f64: forward 7.30e-16  grad_v 3.43e-15  grad_f 2.09e-15
cuda:0   3-D scalar acc4 f64: forward 3.92e-16  grad_v 2.34e-15  grad_f 3.13e-15
cuda:1   3-D scalar acc4 f64: forward 3.92e-16  grad_v 2.34e-15  grad_f 3.13e-15
```

The doubly-derived adjoint check passes for 3-D scalar as well. This is an
existence check on a 32³ grid, not a performance claim; the panels' 3-D
register-pressure worry concerns `elastic_tti`/`viscoelastic`, not `scalar`.

## 9. The riskiest-assumption question

All three panels converged on the same open question: **can one generator
produce competitively performing code for the hardest families without
family-specific escape hatches?** What this spike observed going from the easy
family to the harder one:

**Evidence for.** Not one line of family-specific code was needed. `elastic`
is 106 lines of spec — the same length as `scalar` — and everything that
differs between them was expressed in the *shared* generator as data:
staggering (a per-field tuple of half-offsets), a staggered vs regular pad
rule, multi-stage time steps, staggered difference operators, three model
parameters instead of one, three source and three receiver channels, and
in-place vs ping-pong buffering. The generator *derived* the elastic adjoint
and the elastic model-gradient snapshots correctly on the first symbolic-check
pass, and independently rediscovered the incumbent's own sweep decomposition
(the same field groupings, and the same "adjoint velocities before the
memory-variable writes" ordering that the incumbent's `elastic.c` implements by
hand and documents as a hazard). It also derived the elastic adjoint *without*
the incumbent's ping-pong memory-variable buffers, by scheduling groups
instead — a small, real simplification that fell out of the general machinery.

**Evidence against / unresolved.** Three things the spike genuinely surfaced:

1. **The physics generalised; the *scheduling* did not, at first.** Both real
   bugs found during the spike were in the parts of the generator that decide
   *when* things run and *which buffer* they touch — group ordering and
   snapshot lifetime. Those are exactly the concerns that grow with family
   complexity (more stages, more fields, more in-place hazards), and the
   symbolic check does not cover them. A 27-field 3-D `elastic_tti` would
   stress that machinery far harder than 13-field 2-D elastic did, and there
   is no cheap symbolic oracle for it. **The residual risk moved from the math
   to the schedule**, and the spike did not build a checker for the schedule.
2. **Performance generality is unproven in the direction that matters.** The
   spike is 2–4× off on CPU forward and 1.2–1.6× off on GPU, and the CPU gap
   is dominated by a *generic* modelling choice (SSA state map, hence real
   copies where a human swaps pointers) rather than anything family-specific.
   Fixing it means teaching the generator a buffer-role-rotation optimisation
   — i.e. the generator has to grow the reasoning a human does by hand. That
   is not an escape hatch, but it is the beginning of the same complexity.
3. **The one design element that most needed measuring was the one skipped.**
   The GPU rolled-vs-unrolled question is where all three panels put their
   disagreement, and it is the lever for the binary-size argument. Not
   implementing it means the spike's most-quoted numbers (GPU speed, binary
   size) are the two numbers the rolled design would move.

**Verdict.** On the evidence here, one generator *can* cover a genuinely
harder family with no escape hatch, at the cost of a factor of two to four on
CPU and under two on GPU before any tuning — and the correctness story is
strictly stronger than the incumbent's (whole-matrix, build-time, exact, and
demonstrably catching four classes of injected adjoint bug that the current
`test_adjoint_symbolic.py`'s narrower scope — 1-D/2-D, accuracy 2/4,
model-parameter gradients only, PML explicitly not covered — would not reach
in the PML or on elastic). What is *not* demonstrated: any binary-size or
build-time win, any NF9 improvement, and any evidence about the 27-field
anisotropic families where the panels themselves located the risk. A follow-up
spike wanting to close the question should do `elastic_tti` or `viscoelastic`
in 3-D and implement the rolled GPU variant; going from `scalar` to `elastic`
turned out to be the easy half of the extrapolation.

## 10. Git state

Isolated worktree `.claude/worktrees/agent-a74ed998ebe6757c8`, branch
`worktree-agent-a74ed998ebe6757c8`, based on `refactor/propagator-engine` at
`f45464d`. Everything added is under `exploration/compiled_layer_spike/`;
nothing under `src/deepwave/` was touched.

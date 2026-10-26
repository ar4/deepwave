# Deepwave benchmark suite

A small, dependency-light (torch + deepwave only, no pytest-benchmark)
performance harness for deepwave's propagators. It exists to (a) give
honest, reproducible numbers for public comparisons and (b) gate future
kernel changes: any performance-motivated kernel rewrite (e.g. sharing `.c`/
`.cu` kernel bodies to remove hand-duplication) is contingent on this suite
showing no regression.

**Numbers from this suite are only comparable on the same machine, the same
deepwave commit, and the same thread/dtype/tier configuration.** Wall-clock
timings are not portable across machines; always compare a baseline and a
candidate run captured back-to-back on the same box.

## Quick start

```bash
python benchmarks/run.py --tier micro --reps 5 --label baseline
python benchmarks/run.py --tier micro --reps 5 --label candidate
python benchmarks/run.py --compare benchmarks/results/baseline.json benchmarks/results/candidate.json
```

A smoke test (fast, tiny, 1 rep -- do not use for regression comparisons):

```bash
python benchmarks/run.py --tier micro --reps 1 --label smoke
```

## What gets run

`benchmarks/cases.py` defines the axis space:

- `propagator`: all nine retained families (`scalar`, `scalar_born`,
  `acoustic`, `acoustic_born`, `elastic`, `elastic_born`, `elastic_vti`,
  `acoustic_vti`, `viscoacoustic`)
- `ndim`: 1, 2, 3 (the two VTI families require ndim >= 2)
- `accuracy`: 2, 4, 8 (finite-difference order)
- `dtype`: `float32`, `float64`
- `mode`: `forward`, `forward_backward`
- `tier`: `micro`, `meso`, `macro` (problem size; macro is opt-in and heavy)

The full supported cross product is 300 propagator/ndim/accuracy/dtype/mode points
before even considering tiers, which is too much to run by default. With no
flags, `run.py` uses a **default subset of 16 cases**: the cross product of
`{scalar, elastic} x {2-D, 3-D} x {accuracy 4, 8} x {forward,
forward_backward}` at `float32`. `acoustic`, `scalar_born`, 1-D, `float64`, and
accuracy 2 are not in the default but are one flag away: use `--all` for
the full cross product, or `--propagator`/`--filter` to add just what you
need (e.g. `--propagator acoustic --propagator scalar_born`), and `--tier`
to change problem size.

Size tiers (`grid_size_per_dim`, `nt`), from `cases.TIER_SIZES`:

| tier  | 1-D             | 2-D          | 3-D          |
|-------|-----------------|--------------|--------------|
| micro | 4096, nt=200    | 128, nt=200  | 32, nt=100   |
| meso  | 65536, nt=500   | 512, nt=1000 | 128, nt=500  |
| macro | 1048576, nt=2000| 2048, nt=1000| 256, nt=1000 |

The 2-D/3-D `meso` values and the 3-D `macro` value are a deliberately
chosen benchmark-suite sizing target; the 1-D and 3-D-micro values are this
suite's own choice (picked for a roughly comparable interior-cell count to
the 2-D value at the same tier).

**Per-propagator size overrides.** A few `(propagator, tier, ndim)` points are
too memory-heavy at the nominal tier size to run on a normal host and get a
smaller size via `cases.SIZE_OVERRIDES`. **Elastic 3-D `meso`**
runs at `96^3, nt=200` instead of `128^3, nt=500`: the elastic propagator
stores ~9 wavefield components per adjoint snapshot (vs scalar's ~1), so at the
nominal size its `forward_backward` peak RSS is ~65 GB (measured) and OOM-kills
any host with less RAM; at the override size it peaks at ~14.6 GB. The five
additional families use `64^3, nt=100` for 3-D meso. Scalar/acoustic/scalar_born 3-D `meso` stay at `128^3`
and remain comparable to previously recorded runs. Every result JSON records the
actual `size_per_dim`/`nt`, so a run is always self-describing.

**CI-memory exclusions.** `--all` normally requests the full cross product,
but four meso-tier cases (`acoustic` 3-D accuracy 2/4 float32, `elastic`
3-D accuracy 2/4 float64, all `forward_backward`) exceed the 16 GB RAM of a
GitHub-hosted `ubuntu-latest` runner even after the size override above, so
`get_cases()` drops them by default (`cases.CI_MEMORY_EXCLUDED_CASES`). This
only applies when neither `--propagator` nor `--filter` is given; either
one overrides the exclusion, so a maintainer with more RAM can still reach
these cases explicitly.

**Per-case memory cap (`--mem-cap`, default `auto`).** Each case's subprocess is
wrapped in `systemd-run --user --scope -p MemoryMax=...` so a case that would
exceed host RAM is killed inside its own cgroup and recorded as a per-case error
(`out of memory: exceeded --mem-cap ...`) instead of triggering a global OOM —
which, under systemd's `OOMPolicy=stop`, would otherwise tear down the whole
tmux/session scope (parent process included), not just the case. `auto` = 85% of
host RAM; pass a systemd size like `--mem-cap 18G`, or `--mem-cap off` to
disable. Where `systemd-run --user` is unavailable (some CI/containers) the cap
is skipped with a warning and cases run uncapped, as before. `--tier macro`'s *default case list* is not
the full cross product at 256^3 (that would be extremely heavy) but the
`storage_mode` sweep described below, since that's what macro was
specifically requested for; pass `--all --tier macro` if you really want the
full macro cross product.

The five additional families use `64^3, nt=100` for 3-D meso cases to bound
history memory. VTI cases use distinct anisotropic stiffnesses; viscoacoustic
uses three GSLS mechanisms. Born cases exercise both background and
perturbation gradients. All builders pin `max_vel` and `pml_freq`, and each
result records its timestep and sampling/attenuation settings. Preserve
these settings when comparing revisions or storage choices.

Backward timing uses the sum of requested receiver traces as its objective in
both the candidate and released comparator. Auxiliary-memory tuple layout
therefore does not change the differentiated experiment. Results record the
objective, nonzero material-gradient norms, and the actual current
`RunInfo.step_ratio`. The wavelet peaks at `min(1.5/freq, nt*dt/2)` so short
records include its peak. Re-run both versions with this harness when comparing
timings; older result files used different objectives and source placement.

Tier cases use a single shot (`n_shots=1`), a homogeneous background
model, one source, and a handful of receivers. The CPU kernels
parallelise each time step across the spatial grid, so a single shot
already saturates all threads on large grids; the tier cases therefore
measure the spatial kernel directly. The complementary regime -- many
concurrent shots on a small model, where per-step spatial parallelism
gives each thread little work and the shot batch is the natural
parallelism source -- is covered by the dedicated `--shots-sweep`
report (see below), not by the tiers.

## Methodology

- **1 warmup + N timed repetitions** (`--reps`, default 5) per case, run
  in-process within a dedicated subprocess (see below). The warmup run is
  discarded; timings are **median** and **MAD** (median absolute deviation)
  of the timed reps, which is robust to occasional scheduler noise without
  needing many reps.
- **Subprocess-per-case isolation.** `run.py` spawns one
  `python -m benchmarks._case_runner` subprocess per case. Two things
  require this:
    - `OMP_NUM_THREADS` (and `MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`) must
      be set in the *environment before the Python interpreter starts* to
      reliably pin the C backend's OpenMP thread count -- setting it after
      `import torch` from the same process is unreliable, since torch may
      have already initialised its thread pools. `run.py` sets these in the
      subprocess's `env=...` before spawning it, i.e. before that process's
      own `import torch` happens (`torch.set_num_threads` is additionally
      called inside the subprocess as a second line of defence for torch's
      own intra-op threads).
    - Peak RSS (`VmHWM` in `/proc/self/status`) is a **whole-process,
      monotonically increasing high-water mark** -- it never decreases
      within a process's lifetime. Read from a single long-lived process
      running many cases in a row, it would report the running maximum
      across all of them, not each case's own peak. Running each case in a
      fresh subprocess is the simplest way to get a clean per-case number.
      Caveat: it still includes Python/torch/deepwave's own baseline import
      footprint, not just the propagator call.
- **Environment recorded per run**: CPU model (from `/proc/cpuinfo`),
  `platform.machine()`, `torch.__version__`, the deepwave commit
  (`git rev-parse HEAD`), thread count, and per-case dtype -- all written
  into the result JSON so a number can never be read without its context.

## Metrics per case

- Wall time: median and MAD across `--reps` repetitions.
- `Mcells/s`: `interior_cells * nt_effective / median_time / 1e6`, where
  `interior_cells` excludes the PML region and `nt_effective = nt *
  step_ratio` (the internal CFL time-step subsampling factor, usually 1 --
  cases pick `dt` to satisfy the CFL condition directly where possible).
- Peak RSS (`vmhwm_bytes`): see the subprocess-isolation caveat above.
- Backward/forward time ratio (`mode=forward_backward` only).
- Numerical setup, source peak time, objective and material-gradient norms.
- Disk bytes used (`storage_mode="disk"` only): deepwave's disk storage
  creates and deletes a temporary directory *within* the propagator call, so
  there is no clean post-hoc way to measure it. Instead a background thread
  polls the storage directory's total size every 10 ms while the case runs
  and reports the maximum observed (`disk_bytes`). This is a best-effort,
  possibly-under-sampled number for very fast cases. Other storage modes
  report `disk_bytes: "N/A"` (not applicable/not measurable this way).

## CLI reference

```
python benchmarks/run.py [--tier {micro,meso,macro}] [--reps N] [--warmup N]
                          [--label NAME] [--all] [--propagator P ...]
                          [--filter SUBSTRING] [--threads N] [--out-dir DIR]
                          [--timeout SECONDS] [--smoke]

python benchmarks/run.py --compare BASELINE.json CANDIDATE.json
                          [--regression-threshold FRACTION]
                          [--memory-regression-threshold FRACTION]

python benchmarks/run.py --thread-scaling [--threads-list N ...] [--reps N]
python benchmarks/run.py --storage-modes [--threads N] [--reps N]
```

- `--all`: full axis cross product instead of the ~15-case default subset.
- `--propagator`: repeatable, restricts to given propagator(s).
- `--filter`: substring match on the generated case name (e.g.
  `--filter acc8` for all accuracy-8 cases).
- `--compare A B`: prints a per-case wall-time AND peak-host-RSS
  (`vmhwm_bytes`) delta table, and exits with status 2 if any case
  regressed on either metric by more than its threshold
  (`--regression-threshold` for time, `--memory-regression-threshold` for
  memory; both default 10%). Cases present in only one of the two files
  are listed and skipped. GPU cases' memory column reflects host RSS
  only, which does not track the CUDA caching allocator, so it is not
  yet a meaningful comparison for GPU peak memory. `.github/workflows/
  benchmarks.yml` (a scheduled/on-demand run, plus an opt-in
  `benchmark`-label PR variant) does **not** use `--compare`: it runs the
  suite and publishes the numbers (a results artifact + a job summary
  table) with no pass/fail gate, since a committed baseline captured on a
  different machine is too noisy for an automatic gate on shared CI
  runners. `--compare` remains the intended tool for a manual
  regression check between two runs captured back-to-back on the same
  box -- e.g. before/after a performance-motivated kernel change such as
  a possible single-sourced `.c`/`.cu` kernel-body investigation.
- `--thread-scaling`: runs a single-shot `scalar` 2-D `meso` forward case at
  1, 2, and 4 threads and prints the speedup. This report was written to
  track the (since-closed) gap where a single shot pinned one core; with
  the spatial-OpenMP kernels it now measures real per-step spatial
  scaling.
- `--shots-sweep`: the many-shots/small-model report -- `scalar` 2-D on a
  200x500 grid (nt=1000), `n_shots` 1/20/40, forward and
  forward_backward. Small per-shot grids give each thread little work per
  time step under spatial OpenMP, so this is the regime where kernel-flag
  or parallelisation-strategy changes that look free at meso tier can
  cost real throughput. Pair with `--threads <all host cores>`. Memory
  stays modest by design (scalar 2-D only; the heaviest case stores
  ~19 GB of forward snapshots and is still guarded by `--mem-cap`).
- `--storage-modes`: runs a small 3-D `scalar` `forward_backward` case
  across all four `storage_mode` values (`device`, `cpu`, `disk`, `none`)
  and reports time + peak RSS (+ disk bytes for `disk`) for each -- the
  checkpointing memory/time tradeoff curve.
- `--backend-compare`: runs four fixed `scalar` cases (CPU 512^2 forward,
  CUDA 512^2 forward_backward, CUDA 1024^2 forward, CUDA 3-D 32^3 forward)
  at both `backend="compiled"` (the hand-written C/CUDA backend, default
  for every other case in this suite) and `backend="torch_compile"`
  (`torch.compile`), and prints the speed ratio between them. This is the
  rerunnable measurement behind any "the compiled backend is N times faster
  than `backend='torch_compile'`" claim -- each (case, backend) pair runs in
  its own subprocess (see Methodology above), which also isolates any
  `torch.compile`/Inductor crash to that one subprocess.

## Adding a case

Cases are generated, not hand-written. To add a new point to the default
subset or the tier sizing:

1. To change *what* combinations run by default, edit
   `_default_axis_cases()` in `benchmarks/cases.py`.
2. To change *problem size*, edit `TIER_SIZES` in the same file (or
   `SIZE_OVERRIDES` for a single memory-heavy `(propagator, tier, ndim)`).
3. If a propagator needs different model construction (e.g. a new
   propagator is added to deepwave), add a `_build_<propagator>()`
   function in `benchmarks/models.py` following the existing ones
   (`_build_scalar`, `_build_acoustic`, `_build_elastic`) and wire it into
   `build_inputs()`.

No changes to `run.py` or `harness.py` are normally needed to add a case --
they operate generically on `Case` objects.

## Regression-compare workflow

1. On the commit you consider "known good", run e.g.
   `python benchmarks/run.py --tier meso --reps 7 --label baseline_<commit>`.
2. Make your change.
3. Run the same command with a new label:
   `python benchmarks/run.py --tier meso --reps 7 --label candidate_<change>`.
4. `python benchmarks/run.py --compare benchmarks/results/baseline_<commit>.json benchmarks/results/candidate_<change>.json`.
5. A non-zero exit status (2) means at least one case regressed beyond the
   threshold -- inspect the printed per-case table before deciding whether
   it's real (rerun with more `--reps` on a quiet machine; a busy/shared
   machine like a CI runner or a loaded dev box will show noise well above
   10% on tight cases).

Result JSON files live in `benchmarks/results/` and are git-ignored (they
are machine- and run-specific, not build artifacts to version).

## v0.0.27 parity gate

`benchmarks/v0027_parity.py` is a different kind of instrument from the rest
of this directory: it measures **numbers, not time**. It runs the same
v0.0.27-style calls under two Python interpreters -- the current one, and one
named by `--reference-python` that has `deepwave` v0.0.27 installed -- and
classifies every difference it finds.

```bash
python benchmarks/v0027_parity.py run --reference-python /path/to/v0027/bin/python
python benchmarks/v0027_parity.py list --coverage        # the design, without running it
python benchmarks/v0027_parity.py report benchmarks/results/v0027_parity.json
```

`run` writes `benchmarks/results/v0027_parity.json` (every number) and
`.md` (the report) by default; `--out` and `--md` move them. `report`
re-classifies and re-renders a saved JSON without re-running anything, so a
result can be re-read long after the run. `--only REGEX` restricts the
design; `--quick` runs the kernel block at accuracy 4 only (48 of its 192
cases) and leaves the call-shape blocks alone, which is a fast check of the
harness rather than a gate run; `--repeats N` sets how many times a case is
repeated where each version's own run-to-run spread is being measured.
Full tensors are compared in groups of eight cases; successful captures are
retired after comparison, while failed groups are retained at the printed
scratch path. `--keep-tensors` retains successful captures as well. Use
`--scratch` in a RAM-backed filesystem to avoid bulk writes to persistent disk.

`--no-cuda` skips the CUDA cases. They are also skipped if either
interpreter has no CUDA -- but only `--no-cuda` makes that a success:
without it, a run that lost the CUDA half of the design exits non-zero and
says so, because a gate that passes while a whole device axis went unrun is
worse than one that fails. The coverage table is likewise computed from the
cases that actually ran, so it never claims a level nothing tested.

The reference interpreter is checked before anything runs: if it cannot
import `deepwave`, or the version it imports is not `0.0.27`, the gate says
so and stops. Both interpreters are pinned to 8 threads, before their
`torch` import, because some kernels' summation order -- and so their
rounding -- depends on the thread count.

Elastic cases use the same explicit conservative `max_vel` in both versions
and their own float64 references. The current interpreter resolves this bound
once over both encoded input precisions, before dispatch; each case records
the value. This pins the speed-dependent timestep and PML settings; documented
`pml_freq` migration cases retain their intended frequency differences.
Automatic elastic
defaults can differ because the current heterogeneous stability bound is more
conservative and also controls PML damping, even without timestep subdivision.
See `docs/upgrading.rst`; the gate does not assert equality of these defaults
or classify their differences under an unrelated migration warning.

### What a verdict means

Each case's forward outputs (receiver amplitudes and the physical
final-state fields) and gradients (of every model tensor, of the source
amplitudes and of every initial wavefield) are compared quantity by
quantity, and the case takes the worst one's verdict:

- **(i)** a rounding-level match, which is the promise. In float32 the bound
  is twice the larger of the two versions' own rounding errors, each
  measured against *its own* float64 result for the same call; in float64 it
  is `1e-10 * max|reference|`, or ten times v0.0.27's own measured
  run-to-run spread if that is larger.
- **(ii)** the call hits a documented break: a spelling v0.0.27 accepted and
  this version refuses. `docs/upgrading.rst` lists them.
- **(iii)** a documented change that alters v0.0.27's numbers on purpose,
  reported to the caller at call time.

Anything else breaks the promise. The script carries a table of every case
it *expects* to land in (ii) or (iii), with the documented change that
explains it. Being on that table is a **bounded** excuse, not a blanket one,
because an unbounded excuse is exactly where a future regression would hide:

- the difference has to appear at all -- an expected difference that has
  gone away fails, since the tolerance has then grown too loose to see
  something known to be there;
- it has to be the size the documented change produces. Each entry carries
  a relative band, set from the magnitudes recorded for that change and the
  range this design measures, with its reasoning written next to it. A
  quantity outside the band fails, so a case cannot drift arbitrarily far
  and still report success;
- it has to be in a quantity the change can reach. All four changes in the
  table today alter the wavefield itself -- a moved source, a moved stencil
  point, different interpolation weights, a different absorbing profile --
  so every receiver, state field and gradient follows from them and the
  reach is everything; the field is carried per change so that a future
  change confined to one output cannot excuse the others.

A refusal is bounded the same way: a clause (ii) probe only passes if the
call is refused with the *documented* refusal, not merely refused.

Non-finite results fail at the source. A run that produces a NaN or an
infinity is refused before it is recorded, and any non-finite number
reaching a comparison fails the case: every comparison against a NaN is
false, including the one that decides whether a case failed, so an unguarded
NaN would otherwise be reported as a perfect match.

Exit status: `0` everything was classified as expected; `1` every difference
was expected, but at least one is not yet reported at call time; `2`
something differs that the gate cannot explain, or an expected difference
failed to appear.

### The reduced design

The axes are the propagator, the number of dimensions, the accuracy, the
dtype, the device, the location kind, the call form, the storage location,
whether `pml_freq` is given, and the boundary geometry. Crossing all of them
would be several hundred thousand calls. The design instead has two blocks,
justified separately, and the module docstring gives the full reasoning.

- **The kernel block is a full factorial** over propagator x ndim x accuracy
  x dtype x device. Those five axes each select a *different compiled
  function* -- the C and CUDA kernels are instantiated once per combination
  -- so no point of that grid can be inferred from another. 144 cases.
- **The call-shape block is a covering design.** The remaining axes are
  decided in Python, above the kernel: each either rewrites the call's
  arguments before any kernel sees them, or changes only where a tensor is
  kept between the forward and backward passes. Crossing them with the
  kernel block would re-run the same arithmetic, so each is crossed only
  with the axis it can interact with, and its cases are spread over distinct
  kernel points. 40 cases, including four probes for clause (ii).

`list --coverage` prints the level-by-level tally, computed from the design,
which is how "every level of every axis appears at least once" is checked
rather than asserted. The whole design is 184 cases and just under a
thousand runs; it takes a few minutes on a machine with a GPU.

## Checkpoint reentrancy instrument

`benchmarks/checkpoint_reentrancy.py` compares the two implementations of
`torch.utils.checkpoint.checkpoint` -- reentrant and non-reentrant -- as
`deepwave.checkpointing.run_segments(checkpoint=True)` uses them, on the two
questions that decide which of them should be the default: whether they
compute the same gradients, and what the non-reentrant one costs.

```bash
python benchmarks/checkpoint_reentrancy.py gradients
python benchmarks/checkpoint_reentrancy.py cost --reps 5
```

`gradients` sweeps every propagator, both devices, `storage_mode`
`device`/`cpu`/`disk`, one shot and several, and both ways of reaching the
machinery (`run_segments` directly, and a propagator's own
`checkpoint_segments=`), and prints the largest relative difference between
the two modes' gradients against a tolerance (`1e-12` by default). Exit
status is non-zero if any case exceeds it. A combination a propagator refuses
outright (the two with no compiled backend take `device` storage only) is
printed as `n/a`.

`cost` reports wall-clock time and peak memory in both modes for one
scalar-family and one elastic-family propagator, at two segment counts, on
each available device, with the median and the min/max over `--reps`
repetitions. Each cell runs in its own subprocess, and each memory
repetition in a subprocess of its own: the CPU has no allocator peak counter,
so peak memory there is the growth in `VmHWM`, which never falls within a
process. Time numbers need an idle machine, as everywhere else in this
directory.

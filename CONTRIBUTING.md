# Contributing to Deepwave

Thanks for your interest in improving Deepwave. This document covers dev setup,
testing, linting, the architecture constraints every contribution must respect, how
to add a new propagator, and what reviewers look for in a PR.

## Development setup

Deepwave ships a compiled C/CUDA backend (`libdeepwave_C.so`) built with CMake via
[scikit-build-core](https://scikit-build-core.readthedocs.io/) (see
`[build-system]` in `pyproject.toml` and `CMakeLists.txt`). An editable install
rebuilds and reloads that backend as you change `.c`/`.cu`/`CMakeLists.txt`.

1. Install PyTorch first (CPU or CUDA build, per the
   [PyTorch install instructions](https://pytorch.org/get-started/locally/)).
2. Clone the repo and install Deepwave in editable mode:

   ```bash
   git clone https://github.com/ar4/deepwave.git
   cd deepwave
   pip install -e . -Cbuild-dir=build
   ```

   The normal build isolation installs the declared
   `scikit-build-core>=1.0` backend without depending on a backend installed in
   your environment. `-Cbuild-dir=build` keeps a persistent CMake build directory so
   incremental rebuilds (e.g. after editing a `.c` file) are fast — re-run the
   same `pip install -e .` command to rebuild.
3. Install the test dependencies and the lint-tool versions used by CI:

   ```bash
   pip install pytest pytest-xdist pytest-cov hypothesis scipy sympy matplotlib ruff==0.15.6 mypy==1.19.1 clang-format==23.1.3 clang-tidy==23.1.3
   ```

   (`sympy` is used only by `tests/test_adjoint_symbolic.py`, which re-derives
   the hand-written kernel adjoints symbolically; that file skips itself when
   `sympy` is not installed.)

You need a working C compiler (and, optionally, the CUDA toolkit + `nvcc` if you
want to exercise the CUDA build locally — see the "CUDA cannot be tested here"
note below). CMake auto-detects `CUDAToolkit`; if it isn't found, the build
proceeds CPU-only with a warning.

Source builds use a portable CPU instruction baseline by default. If every CPU
that will run a custom build supports AVX2, you can request fixed AVX2 kernels
with `-Ccmake.define.DW_FIXED_AVX2_OPT_IN=ON`. On supported x86 compilers,
`-Ccmake.define.DW_X86_MULTIVERSION_OPT_IN=ON` instead builds runtime-dispatched
baseline, AVX2 and AVX-512 variants; that increases build time and binary size.
Use only one option. For example:

```bash
pip install -e . -Cbuild-dir=build-avx2 \
  -Ccmake.define.DW_FIXED_AVX2_OPT_IN=ON
```

The released PyPI wheels are built by GitHub Actions; their x86 variants opt
into AVX2. A local source build uses the portable baseline unless explicitly
configured, including on systems that cannot run an x86 release wheel.

## Building the documentation

From the repository root, install this checkout and its documentation dependencies
in the Python environment you will use for Sphinx:

```bash
python -m pip install -e '.[docs]' -Cbuild-dir=build
python -m sphinx -M html docs docs/_build -W --keep-going
```

Alternatively, run `make -C docs html SPHINXOPTS="-W --keep-going"` with that
environment activated. Open `docs/_build/html/index.html` to view the result.
The `docs` extra declares Sphinx and MyST-Parser; the HTML theme ships with
Sphinx. Example dependencies are not needed to render the documentation.

The build imports Deepwave for API documentation and generated capability tables,
so it requires a working installation of **this checkout**, including its compiled
backend. An older installed release or an editable install of another checkout
can fail or document the wrong code. Use `python -c "import deepwave; print(deepwave.__file__)"`
to check which installation your interpreter uses. Installing the extra with
`-e .` above selects this checkout. To force a CPU-only build, add
`-Ccmake.define.DW_SKIP_CUDA=ON` to the install command.

## Running the tests

For local release validation, use the public runner with the Python
interpreter that has the candidate wheel installed:

```bash
python scripts/release_check.py --python /path/to/python \
  --wheel /path/to/candidate.whl --stage all \
  --out /tmp/deepwave-release-results
```

It rejects editable/source-tree imports and requires the loaded Python and
native bytes to match the supplied wheel. It prints the selected interpreter,
records source, wheel and native hashes, and writes a JSON summary with
selected, passed, failed,
skipped and deselected counts per stage. `--stage fast` is useful during
development. `--stage anchors` explicitly runs the finite-difference
anchors that ordinary collection excludes. `--stage gpu` requires CUDA;
repeat `--cuda-device INDEX` to cover each available GPU architecture.
An unavailable CUDA device is an unverified release stage, not a pass.
`--stage self-test` runs a tiny failing pytest child followed by a successful
one and checks that the aggregate still fails.

For direct pytest use, a serial run is simple and keeps resource use bounded:

```bash
python -m pytest tests/ -m "not slow"
```

Pytest fixtures restore the default torch dtype, thread count and anomaly
mode after each test. `test_elastic.py` also scopes its anomaly setting to
that module. The release runner defaults to three workers for the fast stage
(`--fast-workers` controls this limit); its other stages run serially.

Notes:

- `tests/conftest.py` sets each xdist worker's Torch thread count to 1 and
  defaults the OpenMP/BLAS/MKL/NumExpr/TorchInductor environment limits to 1.
  Explicit environment limits are preserved. Serial runs honour their
  configured thread limits; some native kernels parallelise the stencil
  within a shot.
- Tests whose node id contains `gradcheck` are auto-marked `slow` by
  `conftest.py`. Run the fast subset during iterative development by adding
  `-m "not slow"` to any of the invocations above, and run the full suite
  (including `slow`) before opening a PR that touches forward/backward
  numerics.
- `tests/test_gradcheck_real.py` is gated further, behind the
  `manual_gradcheck_anchor` marker, and `conftest.py` deselects it from every
  automatic selection (no default run and no CI job collects it). Those
  finite-difference anchors independently check derivatives of small calls.
  They complement the routine analytic, adjoint dot-product, Taylor, symbolic
  adjoint and compiled-versus-eager checks. Run the anchors with
  `pytest -m manual_gradcheck_anchor`.
- `gpu`-marked tests are skipped (not silently no-op'd) when no CUDA device is
  available — expect them to be skipped in most local/dev environments and in
  this project's CPU-only CI; they only really run on a maintainer's GPU box or
  future GPU CI.
- GPU and heavy differentiation tests should run in bounded batches. A
  failure in a combined run needs diagnosis and a rerun of the same failing
  node, but it is not reclassified as infrastructure merely because an
  isolated rerun passes.

### Coverage and generated numerical tests

The CPU test matrix collects Python statement and branch coverage across all
its test batches and uploads the data, JSON and HTML reports. The existing 92%
statement requirement is enforced by `scripts/check_line_coverage.py`; branch
coverage is reported separately until a full-suite baseline has been reviewed.
Python coverage does not measure execution inside the C/CUDA kernels. A covered
line or branch also does not establish that its result was checked correctly.

`tests/test_numerical_properties.py` uses Hypothesis to generate bounded cases
with exact rational CFL expectations and PML unit-change identities. These
complement the independent Decimal PML oracle. Each generated profile has at
most 24 elements. The explicit deterministic settings reproduce the examples
for the same test, Python and Hypothesis versions; changing those versions can
change the examples. Hypothesis shrinks a failure to a small counterexample,
which should then become an explicit regression test. There is no timing
deadline for numerical assertions: use benchmarks to test performance.

```bash
python -m pytest tests/test_numerical_properties.py tests/test_pml_numerics.py
```

The lint workflow also runs actionlint with ShellCheck on every workflow and its
embedded shell scripts. Keep their versions in sync with `.github/workflows/lint.yml`
when reproducing diagnostics locally.

## Sanitizer builds

CMake's `DW_SANITIZE` option builds the CPU object libraries with
AddressSanitizer + UndefinedBehaviorSanitizer instead of the normal release
flags (`-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1`). Clang's
3D objects use `-O0` to bound compiler memory; all objects retain both
sanitizers and debug information. It
forces `DW_SKIP_CUDA=ON`: ASan/UBSan instrumentation of CPU code and CUDA
device code don't mix, so a sanitizer build is always CPU-only. GCC or Clang
only.

```bash
# GCC
pip install -e . -Cbuild-dir=build-asan-gcc \
  -Ccmake.define.DW_SANITIZE=ON \
  -Ccmake.define.CMAKE_C_COMPILER=gcc -Ccmake.define.CMAKE_CXX_COMPILER=g++

# Clang
pip install -e . -Cbuild-dir=build-asan-clang \
  -Ccmake.define.DW_SANITIZE=ON \
  -Ccmake.define.CMAKE_C_COMPILER=clang -Ccmake.define.CMAKE_CXX_COMPILER=clang++
```

Use a separate `-Cbuild-dir` from your normal build so switching back and
forth doesn't force a full rebuild each time.

**Running tests against a sanitizer build needs `LD_PRELOAD`, every time** —
the instrumented `libdeepwave_C.so` is `dlopen`'d into a `python3` that was
not itself linked against the sanitizer runtime:

```bash
# GCC build
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libasan.so.8 /lib/x86_64-linux-gnu/libstdc++.so.6" \
  CUDA_VISIBLE_DEVICES="" ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
  python scripts/native_sanitizer_check.py --out /tmp/deepwave-sanitizer.json

# Clang build (path depends on your clang version, e.g. llvm-19)
LD_PRELOAD="$(clang -print-resource-dir)/lib/linux/libclang_rt.asan-x86_64.so /lib/x86_64-linux-gnu/libstdc++.so.6" \
  CUDA_VISIBLE_DEVICES="" ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
  python scripts/native_sanitizer_check.py --out /tmp/deepwave-sanitizer.json
```

Runtime settings:

- **`libstdc++.so.6` must be preloaded alongside the sanitizer runtime, not
  just the sanitizer runtime by itself.** Without it, the *first* C++
  exception thrown anywhere in the process — including ordinary internal
  PyTorch control flow, not just deepwave's own code — crashes with
  `AddressSanitizer: CHECK failed ... real___cxa_throw != 0`. The sanitizer
  runtime's constructor resolves the "real" `__cxa_throw` via
  `dlsym(RTLD_NEXT, ...)` at load time, before `libstdc++` has been pulled
  into the process by anything else; preloading it alongside forces it to
  already be present at that point.
- For CPU-only sanitizer jobs, set `CUDA_VISIBLE_DEVICES=""` before
  starting Python so automatic device selection stays on CPU. Preload both
  runtimes as shown above when CUDA is hidden.
- The sanitizer workflow uses 312 small forward/backward cases across all nine
  families, 2-D/3-D, both precisions and all four accuracies, with storage,
  compression and elastic VTI multiaxial-PML/optional-gradient paths. It also
  runs `tests/test_abi_drift.py` and `tests/test_native_storage_boundaries.py`
  under the same preloads. These diagnose native memory and undefined behaviour;
  the normal and physics workflows independently validate scientific accuracy.
  Long 3-D physics runs are especially expensive under Clang's unoptimized
  sanitizer build. A full instrumented suite is optional extended coverage,
  rather than a duplicate mandatory CI gate.

## GPU sanitizer checks (compute-sanitizer)

`compute-sanitizer` (memcheck + racecheck) runs against the normal CUDA
build — it instruments the GPU kernels directly, so `DW_SANITIZE` is neither
needed nor applicable here.

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest tests/test_boundary.py -k test_scalar_c_boundary_cpu_gpu_match
compute-sanitizer --tool racecheck --error-exitcode 1 python -m pytest tests/test_boundary.py -k test_scalar_c_boundary_cpu_gpu_match
```

A full-suite `compute-sanitizer` run is impractically slow; instead run one
small forward+backward invocation per propagator family that actually
dispatches to the compiled CUDA kernel (`python_backend=None` with a CUDA
device tensor) — the various `*_c_boundary_cpu_gpu_match` and
`*_cpu_gpu_match` tests across `tests/test_boundary.py`,
`tests/test_elastic.py`, `tests/test_elastic_vti.py`, and
`tests/test_viscoacoustic.py` are good candidates, since they already compare
against a CPU reference. `racecheck` in particular is worth running
alongside `memcheck`: it detects data-access hazards in shared memory that
`memcheck` does not diagnose. It does not establish freedom from global-memory
races; see the [NVIDIA tool documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool).

## Lint and type checking

```bash
ruff format src/deepwave/ tests/ docs/ && ruff check --fix src/deepwave/ tests/ docs/ && mypy --strict src/
```

Run this before pushing; CI (`.github/workflows/lint.yml`) runs the
non-mutating equivalents (`ruff format --check`, `ruff check`, `mypy --strict`).
All three lint steps are fully blocking: CI fails the build if any of them
reports a problem. `ruff.toml` and `mypy` configuration apply project-wide;
`tests/` has a relaxed Ruff rule set (see `[lint.per-file-ignores]`
in `ruff.toml`) and `docs/` has its own `docs/ruff.toml` (a narrower line
length, so example code fits in a rendered code block, plus the ignores an
example script needs), but neither is exempt from CI -- both `ruff` steps run
against `tests/` and `docs/` as well as `src/deepwave/` and must pass. `mypy
--strict` is checked against `src/` only. `src/deepwave/`, `tests/`, `docs/`,
`ruff format --check`, `ruff check`, and `mypy --strict src/` are all
genuinely clean today, so any regression you introduce will show up as a red,
build-failing CI annotation.

The C/CUDA formatter and CPU static analyser use clang-format and clang-tidy
23.1.3, matching CI. Check formatting with:

```bash
clang-format --dry-run --Werror src/deepwave/*.c src/deepwave/*.cu src/deepwave/*.h
```

Configure a CPU build directory, then run the same clang-tidy command as
`.github/workflows/lint.yml`, covering all nine propagator families and the
storage, compression, and CPU feature helpers. When using the pip-distributed tool
with a GCC build database, pass `--extra-arg=-idirafter` and
`--extra-arg="$(gcc -print-file-name=include)"` to find OpenMP headers after
Clang's own intrinsic headers.

## Example notebooks (jupytext)

Four of the examples under `docs/` (`example_generated_model.py`,
`example_elastic.py`, `example_photoacoustic.py`, `example_fwi_recipes.py`)
are paired with a committed `.ipynb` via
[jupytext](https://jupytext.readthedocs.io/) (`percent` cell format,
`# %%`/`# %% [markdown]` markers), so that Colab users get a real, runnable
notebook (see the "Open in Colab" badges in `README.md` and
`docs/examples.rst`) while the `.py` stays the single source of truth that
Sphinx/`ruff`/normal `python foo.py` execution use. The `.py`'s first cell
runs `%pip install "deepwave>=0.1.0" matplotlib`; its jupytext YAML header declares
`formats: py:percent,ipynb`.

To regenerate the paired `.ipynb` after editing one of these `.py` files
(you must have `jupytext` installed -- `pip install jupytext`):

```bash
jupytext --sync docs/example_generated_model.py
```

(repeat per file, or pass all four at once). `--sync` updates whichever of
the pair is older to match the other; since the `.py` is what you edit, in
practice this regenerates the `.ipynb`'s cells from the `.py` (existing
cell outputs for unchanged cells are preserved). To also verify the
notebook actually executes end-to-end (not just that it parses), re-run it
headlessly and bake in fresh outputs:

```bash
jupyter nbconvert --to notebook --execute --output example_generated_model.ipynb \
    docs/example_generated_model.ipynb
```

`example_elastic.py` and `example_fwi_recipes.py` are physically expensive
(elastic FWI / many FWI-recipe optimisation loops) and take several minutes
to run at full size on a CPU-only machine; both respect an opt-in
`DEEPWAVE_EXAMPLE_FAST=1` environment variable that shrinks iteration counts
for a quick (non-representative) smoke test -- set it before the
`nbconvert --execute` call above if you just want to confirm the code path
runs, e.g. `DEEPWAVE_EXAMPLE_FAST=1 jupyter nbconvert --to notebook --execute ...`.
Leave it unset to reproduce the full results shown in the committed
notebook and the docs page.

## Examples run as automated checks

Every `docs/example_*.py` is executed as an automated check in CI
(`.github/workflows/docs.yml`), not merely checked for syntax. A new or
edited example must actually run.

Each example honours the `DEEPWAVE_SMOKE=1` environment variable by
shrinking its problem size (fewer shots, iterations, grid cells, and so on)
so it runs quickly as an automated check, while still exercising the same
code path a reader would see at full size. Follow this convention when
adding an example: with the other imports, and immediately before the first
size constant it affects, add

```python
# ``DEEPWAVE_SMOKE=1`` shrinks this example so that it can be run quickly as
# an automated check; the default sizes are the ones the text describes.
smoke = os.environ.get("DEEPWAVE_SMOKE") == "1"
```

then write each size that needs shrinking as a conditional expression, with
the pedagogical default kept in the `else` position so a reader still sees
the real number:

```python
n_shots = 4 if smoke else 115
```

Change numbers only — never which functions are called, which branch runs,
or the dtype/device — so the example still demonstrates everything it is
meant to demonstrate, just faster. Finish the example with a final
assertion, running unconditionally, that would catch it silently producing
nothing: a strictly decreasing, finite loss history for an
inversion/optimisation example; a finite, non-zero-maximum output for one
that produces receiver data or a wavefield; or an existing, non-empty output
file otherwise. That assertion must hold in both smoke mode and the default
size.

Every `docs/example_*.py` file must also have an entry in
`scripts/examples_manifest.json`; `scripts/run_examples_smoke.py` fails
loudly, before running anything, if the manifest and the `docs/` directory
disagree. Each entry has a `category`, one of:

- `cpu` — runs on CPU in smoke mode; CI runs it.
- `gpu` — needs a CUDA device to run at all (an example that merely uses one
  when available, falling back to CPU otherwise, is `cpu`, not `gpu`).
- `multi_gpu` — needs two or more CUDA devices.
- `external_data` — cannot run anywhere without a data file that is not in
  this repository and that this repository provides no way to obtain.

Run the sweep locally with

```bash
python scripts/run_examples_smoke.py
```

which runs every `cpu`-category example. Pass `--only <path>` to run a
single example, and `--include external_data` (or `--include gpu`,
`--include multi_gpu`) to also run examples of that category if you happen
to have the data or hardware available.

## Architecture constraints you must respect

These are load-bearing design decisions, not style preferences. Please read
before touching anything under `src/deepwave/*.c`, `*.cu`, `*.h`, or the ctypes
plumbing in `backend_utils.py`.

1. **One compiled artifact, minimal link surface.** The entire backend is a
   single shared library, `libdeepwave_C.so` (`libdeepwave_C.dll` on Windows),
   built from C, a small C++ synchronisation helper, and CUDA mirror `.cu`
   files if `nvcc` is available, then loaded from Python via `ctypes`. It
   uses platform C/C++ and OpenMP runtimes, but **never** links against
   `libtorch`, **never** via `pybind11`/
   `nanobind`/Cython, and it exposes **no CPython ABI dependency at all**. This
   means a supported Deepwave wheel does not need rebuilding solely because
   PyTorch or CPython bumps its ABI. Do not introduce a dependency
   that breaks this (no `#include <torch/...>`, no `Python.h`, no pybind
   bindings) — this constraint is treated as sacrosanct.
2. **`.c` and `.cu` are kept in lockstep.** Every kernel exists as both a CPU
   (`foo.c`) and CUDA (`foo.cu`) translation unit with matching signatures,
   macro-specialised over number of dimensions × finite-difference accuracy ×
   dtype (see the `NDIMS`/`ACCURACIES`/`DTYPES` loops in `CMakeLists.txt`).
   **The CUDA kernels cannot be executed or validated in most contributor/CI
   environments** (no GPU) — they only compile-check. If you change a `.c`
   kernel's math or signature, you must make the matching change in the `.cu`
   file in the same PR, even though you can't run it locally; a maintainer (or
   future GPU CI) validates CUDA execution before release. Say so explicitly in
   your PR description (see "PR expectations" below).
   Both backends retain signed 64-bit tensor indices, offsets and sizes.
   Widen operands before multiplication; assigning a wrapped 32-bit product
   to `int64_t` cannot recover it. Control values use the receiving API's
   type: validate the ABI's 64-bit thread count or device ordinal before
   converting it to `int`. CUDA launch dimensions likewise require checked
   conversion after division of the full tensor size. `native_types.h`
   centralises these operations, with C and C++ boundary tests that require
   no large allocations. A narrower internal index needs a demonstrated
   range bound and measured benefit; CPU and GPU instruction costs can differ.
3. **ABI changes must update the ABI reference tests.** `deepwave._backend_utils`
   assigns `ctypes.argtypes` to every backend entry point; those must exactly
   mirror the C function signatures, or calls silently corrupt memory.
   `tests/test_abi_stability.py` freezes the assigned argtypes against a
   checked-in reference (`tests/abi_reference.json`, keyed by
   `propagator|direction|ndim|dtype`, values are a compact encoding of the
   `ctypes.argtypes` list — see `_CTYPE_CODE`/`_encode` in that test file);
   `tests/test_abi_drift.py` cross-checks that the actual positional arguments
   passed at each call site agree with those argtypes. If you deliberately
   change a backend function's signature, update `abi_reference.json` in the
   same PR and explain why. With the candidate library installed,
   `python tests/test_abi_stability.py --regenerate` produces the reference
   from its assigned argtypes. Review every changed encoding against the
   native signatures and call sites: regeneration alone does not prove the
   ABI is correct, and a reference change should never be incidental.
4. **Backwards compatibility of the public Python API.** All nine
   supported propagators (`scalar`, `scalar_born`, `acoustic`,
   `acoustic_born`, `acoustic_vti`, `viscoacoustic`, `elastic`,
   `elastic_born`, `elastic_vti` — see
   `docs/capabilities.rst`), their `nn.Module` forms, plus the helper modules
   (`wavelets`, `location_interpolation`, `checkpointing`, `wavefields`,
   `das`, `transducers`, `boundaries`) and the `IGNORE_LOCATION` sentinel are
   the public contract. Additive changes (new optional kwargs, new opt-in
   behaviour) are welcome; changing the behaviour of an existing call with
   its current arguments is not, unless it is fixing a genuine correctness
   bug (and even then, flag it loudly in the PR and the changelog).
5. **Keep user-rate model-gradient sampling.** The compiled default
   `model_gradient_sampling_interval=1` retains and correlates histories every
   `step_ratio` CFL steps. It is an intentional data-rate method: for two
   strictly bandlimited correlation factors, shifted-sinc orthogonality gives
   the exact continuous inner product from user-rate samples. The finite DFT
   alias identity in `docs/operator_contract.rst` states its periodic version
   and error term; the adjoint's bandwidth and staggered time phase both
   matter. Finite records can have quadrature error. Do not change the default
   to every-CFL-step history to pass a tiny full-derivative test: at fixed
   duration that multiplies history capacity and traffic by roughly
   `step_ratio`, while live stepping is unchanged. Test exact finite-call
   derivatives at effective stride one, and test sampled correlations with an
   independent sampled oracle at larger strides. Record measured finite-record
   error separately from compression or setup changes.

## Adding a new propagator

Deepwave's Python engine is spec-driven: each propagator declares only its
unique physics, and shares `run_forward`/`run_backward`/`run_backward_core`,
`extract_survey`, ctypes call plumbing, and the gradient-parity/reciprocity
test harness. This is how Deepwave keeps new physics cheap to add. Concretely,
a new propagator is:

1. A `PropagatorSpec` instance (the class lives in `common.py`) describing its
   fields, model parameters, and output names — see the `SPEC =
   deepwave.common.PropagatorSpec(...)` declaration near the top of
   `scalar.py`, `acoustic.py`, `elastic.py`, or `scalar_born.py` for the
   pattern to follow.
2. A thin front-end module (`newprop.py`) that assembles inputs, validates
   them, and calls the shared forward/backward engine (`common.run_forward`/
   `common.run_backward`/`common.run_backward_core`) — following the shape of
   `scalar.py`/`acoustic.py`/`elastic.py`.
3. A CPU kernel (`newprop.c`) and its CUDA mirror (`newprop.cu`), plus the
   corresponding ctypes argtype template entry in `backend_utils.py` (kept in
   lockstep per the constraint above).
4. Two `CMakeLists.txt` additions: an `add_deepwave_cpu_object_library(newprop
   ...)` / `add_deepwave_cuda_object_library(newprop ...)` pair inside the
   existing `NDIMS`/`ACCURACIES`/`DTYPES` loops.
5. Tests that reuse the shared harness (analytic Green's-function checks,
   reciprocity, cross-backend gradient parity) rather than writing bespoke
   infrastructure — follow the existing `tests/test_scalar.py` /
   `tests/test_acoustic.py` / `tests/test_elastic.py` structure.

The genuinely new work per propagator is the constitutive update, the model
preparation, the hand-derived analytic adjoint, and its CUDA mirror — the FD
stencil machinery itself lives in `regular_grid.h`/`staggered_grid.h` and is
reused as-is.

## Docstring cross-reference convention

Most propagator functions share a long tail of generic keyword-only
arguments (PML/CFL controls, storage, callbacks, backend selection, ...)
that are unrelated to the propagator's specific physics. To avoid ~11-way
duplicated (or silently drifting, thinly-paraphrased) prose, each such
parameter has exactly one canonical home — the propagator whose docstring
carries the full description — and every other propagator's entry for that
parameter is `See :func:\`deepwave.<home>\`.` followed by any genuinely
propagator-specific caveat sentence (a different default, a backend
restriction, a physical bound) kept inline rather than deleted.

Canonical home, current mapping:

| Parameter(s) | Canonical home | Why |
| --- | --- | --- |
| `survey_pad`, `origin` | `scalar` | Describe extracting/placing a plain rectangular grid region; the concept and its illustrated example are grid-family-independent, and `scalar`'s prose is the most complete. |
| `accuracy`, `pml_width`, `pml_freq`, `max_vel`, `courant_number`, `nt`, `model_gradient_sampling_interval`, `freq_taper_frac`, `time_pad_frac`, `time_taper`, `forward_callback`, `backward_callback`, `callback_frequency`, `python_backend`, `storage_mode`, `storage_path`, `storage_compression`, `initial_state`, `location_interpolation_halfwidth` | `elastic` | `elastic`'s docstring carries the generic prose for these options; none of the text is regular-vs-staggered-grid-specific enough to warrant a second, `scalar`-flavoured copy. |

Deliberately **excluded** from this convention (each documents a genuine,
non-duplicated per-propagator technical difference, not boilerplate):
`boundary` (per-propagator image-method physics), `mpml_fraction` (M-PML
applicability differs by propagator's instability physics), and the
full-length form of `location_interpolation` (the regular-grid
`N.0` vs staggered-grid `N + 0.5` coordinate convention genuinely differs;
only the grid-independent `location_interpolation_halfwidth` is generic).

When adding a new propagator, or a new generic tail keyword shared by more
than one propagator, follow the same pattern: write the full prose once (on
`scalar` or `elastic`, whichever family it belongs to, or elsewhere with a
one-line justification if neither fits), then cross-reference plus caveat
everywhere else.

Apply the same convention to the Module class and constructor docstrings:
reference the canonical functional description for shared storage arguments,
then add any Module-specific restrictions. Document model tensors and
`*_requires_grad` arguments directly, since those describe the constructor's
own inputs.

Sphinx uses `autoclass_content = "both"`, so constructor and class docstrings
both appear in the rendered API documentation. Keep their descriptions
complementary to avoid documenting the same arguments twice. Public `forward`
methods are rendered as usual.

## PR expectations

- **Tests for behaviour changes.** Any change that alters forward or backward
  output must come with a test that would have caught the regression (unit
  test, analytic check, or an extension of the existing cross-backend
  gradient-parity suite).
- **Full suite for gradient-path changes.** If your change touches anything on
  the path from a propagator's inputs to its gradients (kernel math, the
  autograd `Function`, checkpointing/storage logic, boundary handling), run
  the full (not `-m "not slow"`) test suite locally, plus
  `pytest -m manual_gradcheck_anchor`, before opening the PR, and say in the
  PR description that you did.
- **`.c`/`.cu` lockstep confirmation.** If you touched a kernel, explicitly
  confirm in the PR description that the `.c` and `.cu` files were changed
  together and describe how you reasoned about CUDA correctness given you
  likely could not execute it (mirrored math, matching signature, etc.).
- Keep changes additive per the backwards-compatibility constraint above;
  call out explicitly anywhere you believe an exception is justified.
- Run `ruff format src/deepwave/ && ruff check --fix src/deepwave/ && mypy
  --strict src/` before opening the PR (CI enforces the non-mutating equivalent).

## Release checklist (maintainer)

Follow [the release procedure](docs/dev/releasing.md). Required builds, tests,
docs, lint and sanitizer workflows run on pull requests and manual dispatch.
A standalone branch push is insufficient; unchanged master merges and tag
pushes do not repeat qualification. The benchmark schedule/manual workflow is
independent.

Update the single version source in `src/deepwave/_version.py`, the changelog
and `CITATION.cff` when preparing a new version. Review all required checks
against the actual checked-out Git tree and retain the sole sdist, all four
platform wheel sets, source hashes and run/attempt identities. Squash merge
with an edited maintainer message, verify the merged/tagged tree equals the
accepted tree, and push only the intended release tag.

Publication is a separate manual `publish.yml` dispatch from trusted master,
using PyPI Trusted Publishing and the approved `pypi` environment. It validates
all mandatory workflow evidence and promotes the exact accepted artifacts
without rebuilding. Account setup and the equivalent manual upload fallback
are documented in the linked procedure. Retain local GPU evidence separately;
Actions CPU tests do not claim GPU execution of the rebuilt wheels.

"""Frozen-ABI safety net for the ctypes backend boundary.

The compiled library ``libdeepwave_C.so`` exposes a flat C ABI that Python calls
via ctypes (see ``deepwave._backend_utils``). The ctypes ``argtypes`` assigned in
Python must mirror those C signatures exactly; a mismatch corrupts memory and is
only otherwise caught by the (slow) gradient checks.

These tests freeze that contract:

* ``test_argtypes_match_reference`` asserts that the ``argtypes`` actually
  assigned to every backend function equal a checked-in reference
  (``abi_reference.json``). This makes any change to how ``argtypes`` are built
  (e.g. refactoring ``backend_utils``) prove it did not alter the ABI for any
  combination -- including the CUDA entry points, which share the same argtypes
  as their CPU twins and so are validated even where CUDA cannot be compiled.
* ``test_cpu_cuda_argtypes_identical`` asserts the CPU and CUDA functions of a
  given signature get identical argtypes (only relevant when the library was
  built with CUDA).

The reference depends only on (propagator, direction, ndim, dtype); the
finite-difference accuracy does not change the signature.

To regenerate the reference after an intentional, additive ABI change, run
``python tests/test_abi_stability.py --regenerate`` with the freshly rebuilt
library installed (an editable ``pip install -e .``; a stale ``.so`` would be
read back into the reference). Then verify the diff to
``tests/abi_reference.json`` is purely additive: only new keys, or new tokens
inserted into existing values at a single position, never a changed or
removed token for an existing key.

Symbol completeness is checked even without a GPU: every CPU variant must
exist, and a CUDA-enabled library must contain every CUDA variant. Additional
checks on GPU hosts verify that execution-availability markers agree with the
installed library and visible hardware.
"""

import ctypes
import json
import pathlib

import pytest
import torch
from helpers import c_backend_available

from deepwave import _backend_utils as backend_utils

_CTYPE_CODE = {
    ctypes.c_void_p: "P",
    ctypes.c_int64: "i64",
    ctypes.c_size_t: "sz",
    ctypes.c_bool: "b",
    ctypes.c_float: "f32",
    ctypes.c_double: "f64",
    ctypes.c_int: "i32",
}

# The set of (propagator, direction+suffix) combinations to check -- e.g.
# "backward_sc" is scalar_born/acoustic_born/elastic_born's scattered-only
# adjoint entry point -- and which ndim
# each is built for (VTI has no 1-D form, M-PML has no 1-D build, DW_PM is
# 2-D only) is the ground truth this test suite checks against, and is
# authoritatively defined once in ``deepwave._backend_utils.PROPAGATOR_REGISTRY``
# (see that module for the per-propagator rationale comments this table used
# to duplicate by hand). ``_all_combos`` below derives every valid
# combination directly from it.
_ACCURACIES = (2, 4, 6, 8)
_DTYPES = ("float", "double")

_REFERENCE = json.loads(
    (pathlib.Path(__file__).parent / "abi_reference.json").read_text()
)


def _encode(argtypes):
    return ",".join(_CTYPE_CODE.get(t, "?") for t in argtypes)


def _func_name(prop, ndim, acc, dtype, direction, device):
    # ``direction`` here is already the full ``f"{direction}{suffix}"`` key
    # (e.g. "backward_sc") -- see _all_combos -- so it is used verbatim; the
    # symbol family (the ``_iso`` infix for the historical propagators, or
    # the bare name for the anisotropic elastic_vti/acoustic_vti families)
    # comes straight from the registry entry backend_utils itself assigned
    # argtypes from.
    family = backend_utils.PROPAGATOR_REGISTRY[prop].symbol_family
    return f"{family}_{ndim}d_{acc}_{dtype}_{direction}_{device}"


def _all_combos():
    """Yield every (propagator, ndim, accuracy, dtype, direction) combination.

    Derived directly from ``PROPAGATOR_REGISTRY``: each variant's own
    ``ndim_range`` (intersected with its propagator's) already encodes every
    ndim restriction (VTI's no-1-D form, M-PML's ndim >= 2 build, DW_PM's
    2-D-only build), so there is no exclusion logic to hand-maintain here.
    """
    for prop, spec in backend_utils.PROPAGATOR_REGISTRY.items():
        for variant in spec.variants:
            direction = f"{variant.direction}{variant.suffix}"
            for ndim in sorted(spec.ndim_range & variant.ndim_range):
                for acc in _ACCURACIES:
                    for dtype in _DTYPES:
                        yield prop, ndim, acc, dtype, direction


def test_argtypes_match_reference():
    """Every assigned argtype list matches the frozen reference for its class."""
    checked = 0
    for prop, ndim, acc, dtype, direction in _all_combos():
        key = f"{prop}|{direction}|{ndim}|{dtype}"
        assert key in _REFERENCE, f"no reference for {key}"
        for device in ("cpu", "cuda"):
            func = getattr(
                backend_utils.dll,
                _func_name(prop, ndim, acc, dtype, direction, device),
                None,
            )
            if func is None and device == "cuda":
                continue  # CUDA is optional; completeness is checked separately.
            assert func is not None, f"missing CPU backend for {key}"
            assert func.argtypes is not None, f"untyped backend for {key} ({device})"
            assert _encode(func.argtypes) == _REFERENCE[key], (
                f"ABI drift for {key} ({device})"
            )
            checked += 1
    assert checked > 0, "no backend functions had argtypes assigned"


def test_cpu_cuda_argtypes_identical():
    """CPU and CUDA twins of a signature share identical argtypes."""
    saw_cuda = False
    for prop, ndim, acc, dtype, direction in _all_combos():
        cpu = getattr(
            backend_utils.dll,
            _func_name(prop, ndim, acc, dtype, direction, "cpu"),
            None,
        )
        cuda = getattr(
            backend_utils.dll,
            _func_name(prop, ndim, acc, dtype, direction, "cuda"),
            None,
        )
        if cpu is None or cuda is None:
            continue
        if cuda.argtypes is None:
            continue
        saw_cuda = True
        assert _encode(cpu.argtypes) == _encode(cuda.argtypes)
    if not saw_cuda:
        pytest.skip("library built without CUDA")


_PROPAGATORS = tuple(backend_utils.PROPAGATOR_REGISTRY.keys())


def _cuda_build_capable() -> bool:
    """Whether the loaded library exports CUDA entry points at all.

    CMakeLists.txt (``DW_SKIP_CUDA`` / ``CUDAToolkit_FOUND``) builds the CUDA
    object libraries for every propagator in a single pass, so either all of
    them are present or none are; this is confirmed exhaustively for all nine
    propagators by ``test_all_expected_symbols_resolve_without_a_gpu`` below. A
    single representative signature is enough to classify the build.
    """
    return c_backend_available("scalar", device=torch.device("cuda"))


# Full capability means the runtime can actually use a CUDA device *and* the
# loaded library was built with CUDA support. Either alone is not enough: a
# CUDA-built library on a box with no visible GPU still reports
# ``torch.cuda.is_available() is False``, and a CPU-only build run on a
# CUDA-capable box still has no CUDA symbols to call.
_FULL_CAPABILITY = torch.cuda.is_available() and _cuda_build_capable()


def test_full_capability_backends_available():
    """Every propagator's compiled backend resolves on both cpu and cuda.

    On a box that reports full capability, a propagator silently missing its
    compiled backend would otherwise only surface as scattered ``pytest``
    skips (the ``C_BACKEND`` / ``_ELASTIC_C_BACKEND`` /
    ``_ELASTIC_VTI_C_BACKEND`` markers in test_elastic.py, test_elastic_vti.py,
    test_viscoacoustic.py, and test_acoustic_vti.py, plus the CUDA-only tests
    gated on ``torch.cuda.is_available()``). This makes that a hard failure.
    """
    if not _FULL_CAPABILITY:
        pytest.skip("full CUDA capability (runtime + CUDA-built library) absent")
    for prop in _PROPAGATORS:
        for device in ("cpu", "cuda"):
            assert c_backend_available(prop, device=torch.device(device)), (
                f"c_backend_available({prop!r}, device={device!r}) is False "
                "despite the environment reporting full CUDA capability"
            )


def test_all_expected_symbols_resolve_without_a_gpu():
    """Every expected backend symbol resolves in the loaded ``ctypes.CDLL``.

    Use the release wheel check, which enumerates the same registry as the
    ABI-reference checks, including optional variants and higher accuracies.
    """
    # Symbol resolution is a property of the library, not of GPU availability.
    # The release wheel check additionally requires CUDA on Linux/Windows.
    from scripts.platform_wheel_smoke import check_backend_symbols

    check_backend_symbols(backend_utils.dll, backend_utils.PROPAGATOR_REGISTRY)


def test_full_capability_no_known_capability_skips():
    """No known CUDA/backend-presence skip marker fires on this box.

    A fully generic whole-suite skip count would need to drive a nested
    ``pytest --collect-only`` from inside a test, which is fragile (working
    directory, plugin, and marker-expression state would all need to be
    reconstructed). Instead this enumerates the specific capability-gated
    ``pytest.mark.skipif`` conditions that exist in the suite today (found by
    grepping ``tests/`` for ``skipif.*cuda`` and ``skipif.*backend_available``)
    and asserts each evaluates to "run, don't skip" in this environment:

    * every ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` marker
      (test_acoustic.py, test_scalar.py, test_scalar_born.py, test_elastic.py,
      test_elastic_born.py, test_elastic_vti.py, test_viscoacoustic.py,
      test_wavefields.py, and five in test_boundary.py) shares the identical
      condition, so checking ``torch.cuda.is_available()`` once covers all of
      them;
    * every ``@pytest.mark.skipif(not c_backend_available(prop, device=...), ...)``
      marker (the ``C_BACKEND`` / ``_ELASTIC_C_BACKEND`` /
      ``_ELASTIC_VTI_C_BACKEND`` markers in test_elastic.py,
      test_elastic_vti.py, test_viscoacoustic.py, and test_acoustic_vti.py)
      reduces to one of the (propagator, device) pairs enumerated below.

    The runtime ``pytest.skip(...)`` guards inside test_convergence.py,
    test_equivalence_matrix.py, and test_adjoint_dot_product.py are not
    re-enumerated here: they only ever call ``c_backend_available`` with
    ``device=torch.device("cpu")`` and one of the nine propagator names, so
    ``test_full_capability_backends_available`` (which checks all nine on cpu)
    already proves those guards cannot fire either.
    """
    if not _FULL_CAPABILITY:
        pytest.skip("full CUDA capability (runtime + CUDA-built library) absent")
    assert torch.cuda.is_available(), (
        "a torch.cuda.is_available()-gated test would skip in this environment"
    )
    backend_gated_markers = (
        ("elastic", "cpu"),
        ("elastic_vti", "cpu"),
        ("viscoacoustic", "cpu"),
        ("acoustic_vti", "cpu"),
    )
    for prop, device in backend_gated_markers:
        assert c_backend_available(prop, device=torch.device(device)), (
            f"a c_backend_available({prop!r}, device={device!r})-gated test "
            "would skip in this environment"
        )


def _regenerate_reference() -> "dict[str, str]":
    """Rebuild the reference mapping from the installed library's argtypes.

    The reference key is ``"{prop}|{direction}|{ndim}|{dtype}"``; accuracy
    does not change a signature, and CPU and CUDA twins share argtypes
    (``test_cpu_cuda_argtypes_identical``), so the first device found with
    assigned argtypes for each key supplies its value.
    """
    reference: dict[str, str] = {}
    expected = set()
    for prop, ndim, acc, dtype, direction in _all_combos():
        key = f"{prop}|{direction}|{ndim}|{dtype}"
        expected.add(key)
        if key in reference:
            continue
        for device in ("cpu", "cuda"):
            func = getattr(
                backend_utils.dll,
                _func_name(prop, ndim, acc, dtype, direction, device),
                None,
            )
            if func is not None and func.argtypes is not None:
                reference[key] = _encode(func.argtypes)
                break
    missing = expected - set(reference)
    if missing:
        raise SystemExit(
            "cannot regenerate: no argtypes found for "
            f"{len(missing)} combination(s), e.g. {sorted(missing)[:5]} -- "
            "is the installed libdeepwave_C.so current and complete?"
        )
    return reference


def test_capability_reporting_is_consistent_without_full_capability():
    """Backend-availability reporting is honest on a CPU-only build.

    Symmetric complement of the tests above: on a library built WITHOUT
    CUDA support (``_cuda_build_capable()`` is False -- no CUDA symbols
    exist in the DLL at all), ``c_backend_available`` must consistently
    report CUDA as unavailable for every propagator -- never a false
    "available" claim for a symbol that was never compiled -- while still
    reporting the cpu backend as available for every propagator (CUDA
    absence must not degrade the cpu build).

    Deliberately guarded on ``_cuda_build_capable()`` alone, not the
    combined ``_FULL_CAPABILITY`` (which also folds in
    ``torch.cuda.is_available()``): ``c_backend_available`` only checks
    DLL symbol presence (see its docstring), a build-time property, so it
    correctly stays True for a CUDA-capable build even when
    ``torch.cuda.is_available()`` is transiently False (e.g. a wedged
    driver, or simply no GPU visible right now) -- exactly the case
    ``_FULL_CAPABILITY``'s own comment already documents as one of the two
    independent ways to lose full capability. Asserting the CUDA branch
    False in that case would be asserting against this helper's own
    documented contract, not testing a real defect. This test's actual
    subject -- a CPU-only build -- is not exercised on this box, which has
    a CUDA-capable build; it runs for real (and must pass) on a build
    where ``DW_SKIP_CUDA=ON`` or no CUDA toolkit was found.
    """
    if _cuda_build_capable():
        pytest.skip("only meaningful for a CPU-only build (no CUDA symbols in the DLL)")
    for prop in _PROPAGATORS:
        assert not c_backend_available(prop, device=torch.device("cuda")), (
            f"c_backend_available({prop!r}, device='cuda') is True despite "
            "the library being built without CUDA support"
        )
        assert c_backend_available(prop, device=torch.device("cpu")), (
            f"c_backend_available({prop!r}, device='cpu') is False; the cpu "
            "backend must remain available regardless of CUDA status"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ABI reference maintenance for tests/abi_reference.json"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "rewrite tests/abi_reference.json from the installed library's "
            "argtypes (see the module docstring for the required additive-"
            "diff verification afterwards)"
        ),
    )
    args = parser.parse_args()
    if not args.regenerate:
        parser.error("nothing to do; pass --regenerate")
    _reference_path = pathlib.Path(__file__).parent / "abi_reference.json"
    _reference_path.write_text(
        json.dumps(_regenerate_reference(), indent=4, sort_keys=True) + "\n"
    )
    print(f"wrote {_reference_path}")

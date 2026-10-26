"""Wheel qualification rejects incomplete libraries even without a GPU."""

from types import SimpleNamespace

import pytest

from scripts.platform_wheel_smoke import check_backend_symbols


@pytest.fixture
def registry():
    """A base kernel and an optional two-dimensional kernel variant."""
    return {
        "wave": SimpleNamespace(
            symbol_family="wave_iso",
            ndim_range=frozenset({1, 2}),
            variants=(
                SimpleNamespace(
                    direction="forward", suffix="", ndim_range=frozenset({1, 2})
                ),
                SimpleNamespace(
                    direction="backward", suffix="_extra", ndim_range=frozenset({2})
                ),
            ),
        )
    }


def library(*devices):
    """Model exported symbols, with no torch, native library or CUDA runtime."""
    names = {
        f"wave_iso_{ndim}d_{accuracy}_{dtype}_{direction}_{device}": object()
        for ndim, direction in ((1, "forward"), (2, "forward"), (2, "backward_extra"))
        for accuracy in (2, 4, 6, 8)
        for dtype in ("float", "double")
        for device in devices
    }
    return SimpleNamespace(**names)


def test_cpu_only_library_is_valid_when_cuda_is_optional(registry):
    """A contributor's CPU-only build still supports every CPU variant."""
    assert not check_backend_symbols(library("cpu"), registry)


def test_cuda_release_cannot_silently_be_cpu_only(registry):
    """The release requirement is independent of GPU hardware detection."""
    with pytest.raises(AssertionError, match=r"expected symbols: .*_cuda"):
        check_backend_symbols(library("cpu"), registry, require_cuda=True)


def test_complete_cuda_library_needs_no_gpu(registry):
    """Checking both symbol sets never initialises or executes CUDA."""
    assert check_backend_symbols(library("cpu", "cuda"), registry, require_cuda=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_missing_high_order_double_variant_fails(registry, device):
    """A representative forward float32 kernel cannot hide missing variants."""
    candidate = library("cpu", "cuda")
    missing = f"wave_iso_2d_8_double_backward_extra_{device}"
    delattr(candidate, missing)
    with pytest.raises(AssertionError, match=missing):
        check_backend_symbols(candidate, registry)


def test_partial_cuda_library_cannot_be_classified_as_cpu_only(registry):
    """Even a non-representative CUDA symbol makes CUDA completeness mandatory."""
    candidate = library("cpu")
    candidate.wave_iso_2d_8_double_backward_extra_cuda = object()
    with pytest.raises(AssertionError, match=r"expected symbols: .*_cuda"):
        check_backend_symbols(candidate, registry)


def test_empty_registry_is_not_a_success():
    """Qualification must inspect at least one expected kernel."""
    with pytest.raises(AssertionError, match="no expected backend symbols"):
        check_backend_symbols(SimpleNamespace(), {})

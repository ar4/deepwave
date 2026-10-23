"""Tests for deepwave.misfits: l2, envelope and normalised_cross_correlation."""

import pytest
import torch

from deepwave.misfits import envelope, l2, normalised_cross_correlation

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------


def test_envelope_gradcheck_odd_nt() -> None:
    """float64 gradcheck for envelope() with an odd nt (no Nyquist bin)."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 7, dtype=DTYPE, requires_grad=True)
    target = torch.randn(2, 3, 7, dtype=DTYPE, requires_grad=True)

    def f(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return envelope(pred, target, reduction="none")

    assert torch.autograd.gradcheck(
        f, (pred, target), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_envelope_gradcheck_even_nt() -> None:
    """float64 gradcheck for envelope() with an even nt (has a Nyquist bin)."""
    torch.manual_seed(1)
    pred = torch.randn(2, 8, dtype=DTYPE, requires_grad=True)
    target = torch.randn(2, 8, dtype=DTYPE, requires_grad=True)

    def f(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return envelope(pred, target, reduction="mean")

    assert torch.autograd.gradcheck(
        f, (pred, target), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_ncc_gradcheck() -> None:
    """float64 gradcheck for normalised_cross_correlation()."""
    torch.manual_seed(2)
    pred = torch.randn(3, 4, 9, dtype=DTYPE, requires_grad=True)
    target = torch.randn(3, 4, 9, dtype=DTYPE, requires_grad=True)

    def f(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return normalised_cross_correlation(pred, target, reduction="none")

    assert torch.autograd.gradcheck(
        f, (pred, target), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


# ---------------------------------------------------------------------------
# envelope identity and zero gradient
# ---------------------------------------------------------------------------


def test_envelope_identity_is_zero() -> None:
    """envelope(x, x) is exactly zero for both odd and even nt."""
    torch.manual_seed(3)
    for nt in (7, 8):
        x = torch.randn(2, 3, nt, dtype=DTYPE)
        loss = envelope(x, x, reduction="none")
        assert torch.equal(loss, torch.zeros_like(loss))


def test_envelope_identity_gradient_is_zero() -> None:
    """The gradient of envelope(x, x) w.r.t. x is exactly zero (no NaN)."""
    torch.manual_seed(4)
    for nt in (7, 8):
        x = torch.randn(2, nt, dtype=DTYPE, requires_grad=True)
        loss = envelope(x, x, reduction="mean")
        (grad,) = torch.autograd.grad(loss, x)
        assert not torch.isnan(grad).any()
        assert torch.equal(grad, torch.zeros_like(grad))


# ---------------------------------------------------------------------------
# NCC: scale invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1e-6, 1e-3, 0.5, 1.0, 5.0, 1e3, 1e6])
def test_ncc_scale_invariance(a: float) -> None:
    """loss(a * x, x) ~= 0 for a > 0, including large and small a."""
    torch.manual_seed(5)
    x = torch.randn(2, 3, 11, dtype=DTYPE)
    loss = normalised_cross_correlation(a * x, x, reduction="none")
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-10)


# Regression test: an earlier implementation used a fixed absolute epsilon
# (1e-8) to guard the zero-norm case in normalised_cross_correlation. That
# guard fired for any trace whose norms multiplied to below 1e-8 -- ordinary
# seismic amplitudes, not a degenerate edge case (e.g. this repository's own
# docs/example_elastic_groundroll.py plots receiver data with limits of
# +/-1e-8) -- silently collapsing both the identity property (loss(x, x) == 0)
# and scale invariance at small-but-nonzero amplitudes. The fix normalises
# each trace by its own max absolute value before computing the norm, which
# has no scale threshold at all. These amplitude scales span 1e+3 down to
# 1e-30 in float64 (and the float32 sweep checks down to the float32-normal
# range), well past where the old fixed-epsilon guard broke.
_AMPLITUDE_SCALES_FLOAT64 = [1e3, 1e0, 1e-3, 1e-6, 1e-8, 1e-15, 1e-22, 1e-30]
_AMPLITUDE_SCALES_FLOAT32 = [1e3, 1e0, 1e-3, 1e-6, 1e-8, 1e-15, 1e-20]


@pytest.mark.parametrize("scale", _AMPLITUDE_SCALES_FLOAT64)
def test_ncc_identity_is_zero_across_amplitude_scales_float64(scale: float) -> None:
    """normalised_cross_correlation(x, x) == 0 to round-off from 1e3 to 1e-30."""
    torch.manual_seed(10)
    x = torch.randn(2, 3, 11, dtype=DTYPE) * scale
    loss = normalised_cross_correlation(x, x, reduction="none")
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-10)


@pytest.mark.parametrize("scale", _AMPLITUDE_SCALES_FLOAT64)
def test_ncc_scale_invariance_across_amplitude_scales_float64(scale: float) -> None:
    """normalised_cross_correlation(2x, x) == 0 to round-off from 1e3 to 1e-30."""
    torch.manual_seed(11)
    x = torch.randn(2, 3, 11, dtype=DTYPE) * scale
    loss = normalised_cross_correlation(2 * x, x, reduction="none")
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-10)


@pytest.mark.parametrize("scale", _AMPLITUDE_SCALES_FLOAT32)
def test_ncc_identity_is_zero_across_amplitude_scales_float32(scale: float) -> None:
    """normalised_cross_correlation(x, x) == 0 (float32 tolerance) across scales."""
    torch.manual_seed(12)
    x = torch.randn(2, 3, 11, dtype=torch.float32) * scale
    loss = normalised_cross_correlation(x, x, reduction="none")
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)


@pytest.mark.parametrize("scale", _AMPLITUDE_SCALES_FLOAT64)
def test_envelope_identity_is_zero_across_amplitude_scales(scale: float) -> None:
    """envelope(x, x) == 0 exactly across amplitude scales (no epsilon involved)."""
    torch.manual_seed(13)
    x = torch.randn(2, 3, 11, dtype=DTYPE) * scale
    loss = envelope(x, x, reduction="none")
    assert torch.equal(loss, torch.zeros_like(loss))


# ---------------------------------------------------------------------------
# NCC: zero-trace behaviour
# ---------------------------------------------------------------------------


def test_ncc_zero_pred_trace_gives_documented_loss_and_finite_grad() -> None:
    """An all-zero pred trace gives NCC = 0, loss = 1, with a finite gradient."""
    torch.manual_seed(6)
    pred = torch.zeros(1, 5, dtype=DTYPE, requires_grad=True)
    target = torch.randn(1, 5, dtype=DTYPE, requires_grad=True)
    loss = normalised_cross_correlation(pred, target, reduction="none")
    assert torch.allclose(loss, torch.ones_like(loss))
    loss.sum().backward()
    assert pred.grad is not None
    assert target.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(target.grad).all()


def test_ncc_both_traces_zero_gives_documented_loss_and_finite_grad() -> None:
    """Both traces all-zero also gives NCC = 0, loss = 1, with a finite gradient."""
    pred = torch.zeros(1, 5, dtype=DTYPE, requires_grad=True)
    target = torch.zeros(1, 5, dtype=DTYPE, requires_grad=True)
    loss = normalised_cross_correlation(pred, target, reduction="none")
    assert torch.allclose(loss, torch.ones_like(loss))
    loss.sum().backward()
    assert pred.grad is not None
    assert target.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(target.grad).all()


# ---------------------------------------------------------------------------
# reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_reduction_none_shape(func) -> None:
    """reduction='none' returns per-trace values with the time axis removed."""
    torch.manual_seed(7)
    pred = torch.randn(2, 3, 6, dtype=DTYPE)
    target = torch.randn(2, 3, 6, dtype=DTYPE)
    out = func(pred, target, reduction="none")
    assert out.shape == (2, 3)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_reduction_sum_equals_sum_of_none(func) -> None:
    """reduction='sum' equals the sum of the per-trace ('none') values."""
    torch.manual_seed(8)
    pred = torch.randn(2, 3, 6, dtype=DTYPE)
    target = torch.randn(2, 3, 6, dtype=DTYPE)
    per_trace = func(pred, target, reduction="none")
    total = func(pred, target, reduction="sum")
    assert total.shape == ()
    assert torch.allclose(total, per_trace.sum())


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_reduction_mean_equals_mean_of_none(func) -> None:
    """reduction='mean' (the default) equals the mean of the per-trace values."""
    torch.manual_seed(9)
    pred = torch.randn(2, 3, 6, dtype=DTYPE)
    target = torch.randn(2, 3, 6, dtype=DTYPE)
    per_trace = func(pred, target, reduction="none")
    mean_explicit = func(pred, target, reduction="mean")
    mean_default = func(pred, target)
    assert mean_explicit.shape == ()
    assert torch.allclose(mean_explicit, per_trace.mean())
    assert torch.allclose(mean_default, mean_explicit)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_shape_mismatch_raises(func) -> None:
    """Mismatched pred/target shapes raise ValueError naming the shapes."""
    pred = torch.randn(2, 5, dtype=DTYPE)
    target = torch.randn(3, 5, dtype=DTYPE)
    with pytest.raises(ValueError, match=r"pred and target must have the same shape"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_zero_dim_raises(func) -> None:
    """A 0-dimensional pred/target (no time axis) raises ValueError."""
    pred = torch.tensor(1.0, dtype=DTYPE)
    target = torch.tensor(1.0, dtype=DTYPE)
    with pytest.raises(ValueError, match=r"at least one dimension"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_empty_time_axis_raises_consistently(func) -> None:
    """All misfits reject records with no time samples before reducing."""
    pred = torch.empty(2, 0, dtype=DTYPE)
    target = torch.empty_like(pred)
    with pytest.raises(ValueError, match="at least one time sample"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_empty_leading_trace_dimensions_raise_consistently(func) -> None:
    """All misfits reject empty batches before reductions or FFT operations."""
    pred = torch.empty(0, 5, dtype=DTYPE)
    target = torch.empty_like(pred)
    with pytest.raises(ValueError, match="at least one trace"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_non_floating_dtype_raises(func) -> None:
    """A non-floating-point pred dtype raises TypeError naming the dtype."""
    pred = torch.randint(0, 10, (2, 5))
    target = torch.randn(2, 5)
    with pytest.raises(TypeError, match=r"pred must be a floating-point tensor"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_non_floating_target_dtype_raises(func) -> None:
    """A non-floating-point target dtype raises TypeError naming the dtype."""
    pred = torch.randn(2, 5)
    target = torch.randint(0, 10, (2, 5))
    with pytest.raises(TypeError, match=r"target must be a floating-point tensor"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_dtype_mismatch_raises(func) -> None:
    """Mismatched pred/target dtypes raise TypeError naming both dtypes."""
    pred = torch.randn(2, 5, dtype=torch.float32)
    target = torch.randn(2, 5, dtype=torch.float64)
    with pytest.raises(TypeError, match=r"pred and target must have the same dtype"):
        func(pred, target)


@pytest.mark.parametrize("func", [l2, envelope, normalised_cross_correlation])
def test_unknown_reduction_raises(func) -> None:
    """An unrecognised reduction raises ValueError naming the bad value."""
    pred = torch.randn(2, 5, dtype=DTYPE)
    target = torch.randn(2, 5, dtype=DTYPE)
    with pytest.raises(ValueError, match=r"reduction must be one of .*bogus"):
        func(pred, target, reduction="bogus")


# ---------------------------------------------------------------------------
# __all__ / top-level exports
# ---------------------------------------------------------------------------


def test_exported_from_module_all() -> None:
    """The three misfit functions are exactly deepwave.misfits.__all__."""
    import deepwave.misfits as misfits_module

    assert set(misfits_module.__all__) == {
        "envelope",
        "l2",
        "normalised_cross_correlation",
    }


def test_deepwave_misfits_is_importable_and_reachable() -> None:
    """deepwave.misfits is reachable as a deepwave attribute, like wavelets."""
    import deepwave

    assert deepwave.misfits.envelope is envelope
    assert deepwave.misfits.l2 is l2
    assert deepwave.misfits.normalised_cross_correlation is normalised_cross_correlation


# ---------------------------------------------------------------------------
# l2
# ---------------------------------------------------------------------------


def test_l2_is_the_per_trace_sum_of_squared_differences() -> None:
    """l2 sums the squared difference over time, one value per trace."""
    torch.manual_seed(20)
    pred = torch.randn(2, 3, 6, dtype=DTYPE)
    target = torch.randn(2, 3, 6, dtype=DTYPE)
    expected = ((pred - target) ** 2).sum(-1)
    out = l2(pred, target, reduction="none")
    assert out.shape == (2, 3)
    assert torch.equal(out, expected)


def test_l2_sum_equals_mse_loss_sum() -> None:
    """l2's 'sum' reduction is torch's mse_loss(reduction='sum')."""
    torch.manual_seed(21)
    pred = torch.randn(2, 3, 6, dtype=DTYPE)
    target = torch.randn(2, 3, 6, dtype=DTYPE)
    assert torch.allclose(
        l2(pred, target, reduction="sum"),
        torch.nn.functional.mse_loss(pred, target, reduction="sum"),
    )


def test_l2_mean_is_per_trace_not_per_sample() -> None:
    """l2's 'mean' averages over traces, not over samples (unlike mse_loss).

    This is the shape contract shared with the other misfits: the time axis is
    summed away first, so 'mean' is nt times mse_loss's element-wise mean.
    """
    torch.manual_seed(22)
    nt = 6
    pred = torch.randn(2, 3, nt, dtype=DTYPE)
    target = torch.randn(2, 3, nt, dtype=DTYPE)
    assert torch.allclose(
        l2(pred, target, reduction="mean"),
        nt * torch.nn.functional.mse_loss(pred, target, reduction="mean"),
    )


def test_l2_of_identical_data_is_exactly_zero_with_zero_gradient() -> None:
    """l2(x, x) is exactly 0 and its gradient is exactly 0 (no NaN)."""
    torch.manual_seed(25)
    x = torch.randn(2, 5, dtype=DTYPE, requires_grad=True)
    loss = l2(x, x, reduction="mean")
    assert loss.item() == 0.0
    loss.backward()
    assert x.grad is not None
    assert bool((x.grad == 0).all())


def test_l2_gradient_is_twice_the_residual() -> None:
    """d/d(pred) of the per-trace l2 is exactly 2 * (pred - target)."""
    torch.manual_seed(23)
    pred = torch.randn(2, 4, dtype=DTYPE, requires_grad=True)
    target = torch.randn(2, 4, dtype=DTYPE)
    l2(pred, target, reduction="sum").backward()
    assert pred.grad is not None
    assert torch.allclose(pred.grad, 2 * (pred.detach() - target))


def test_l2_gradcheck() -> None:
    """float64 gradcheck for l2()."""
    torch.manual_seed(24)
    pred = torch.randn(2, 3, 7, dtype=DTYPE, requires_grad=True)
    target = torch.randn(2, 3, 7, dtype=DTYPE, requires_grad=True)

    def f(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return l2(pred, target, reduction="none")

    assert torch.autograd.gradcheck(
        f, (pred, target), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


# ---------------------------------------------------------------------------
# the Reduction annotation
# ---------------------------------------------------------------------------


def test_reduction_is_annotated_with_the_reduction_literal() -> None:
    """Every misfit annotates reduction as Reduction, keyword-only."""
    import inspect
    import typing

    import deepwave

    for func in (l2, envelope, normalised_cross_correlation):
        hints = typing.get_type_hints(func)
        assert hints["reduction"] == deepwave.Reduction, func.__name__
        parameter = inspect.signature(func).parameters["reduction"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, func.__name__
        assert parameter.default == "mean", func.__name__


def test_runtime_reductions_are_single_sourced_from_the_literal() -> None:
    """_REDUCTIONS is derived from the Reduction Literal, not written twice."""
    import typing

    import deepwave
    import deepwave.misfits as misfits_module

    assert typing.get_args(deepwave.Reduction) == misfits_module._REDUCTIONS

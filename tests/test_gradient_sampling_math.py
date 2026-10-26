"""Independent bandlimited inner-product checks for sampled model gradients.

These checks exercise the mathematical sampling contract, not a Deepwave
backend. Finite-call adjoints have endpoints and need not be bandlimited;
their separate source/native tests check the actual correlation weight and
staggering.
"""

import numpy as np
import pytest


def test_shifted_sinc_inner_product_integrates_from_samples() -> None:
    """Orthogonal sinc translates give the continuous correlation integral."""
    delta = 0.2
    shift = 0.31 * delta
    a_coeff = {-2: 1 + 0.2j, 0: -0.3 + 0.5j, 3: 0.8 - 0.1j}
    b_coeff = {-1: 0.2 - 0.4j, 0: 1 + 0.3j, 2: -0.6 + 0.2j}
    # u=(t-shift)/delta: a shifted sinc basis remains orthogonal. The wide
    # integration interval leaves a directly measurable truncation error.
    du = 0.01
    t = shift + np.arange(-100000, 100001, dtype=np.float64) * delta * du
    u = (t - shift) / delta
    a = sum(value * np.sinc(u - index) for index, value in a_coeff.items())
    b = sum(value * np.sinc(u - index) for index, value in b_coeff.items())
    product = a * np.conj(b)
    numerical_integral = delta * du * np.sum((product[1:] + product[:-1]) / 2)
    sample_sum = delta * sum(
        value * np.conj(b_coeff[index])
        for index, value in a_coeff.items()
        if index in b_coeff
    )
    assert abs(numerical_integral - sample_sum) < 5e-5


@pytest.mark.parametrize(("stride", "coarse_count"), [(2, 64), (4, 32)])
def test_finite_dft_alias_identity_and_bandlimited_exactness(
    stride: int, coarse_count: int
) -> None:
    """Coarse and fine correlations differ by the DFT alias bins exactly."""
    fine_count = stride * coarse_count
    delta = 0.2
    j = np.arange(fine_count)

    def check(a: np.ndarray, b: np.ndarray) -> complex:
        correlation = a * np.conj(b)
        spectrum = np.fft.fft(correlation) / fine_count
        full = delta / stride * correlation.sum()
        sampled = delta * correlation[::stride].sum()
        aliases = coarse_count * delta * spectrum[coarse_count::coarse_count].sum()
        np.testing.assert_allclose(sampled - full, aliases, rtol=0, atol=1e-12)
        return sampled - full

    # Every factor's Fourier support is strictly inside the coarse Nyquist
    # band; their product can occupy more than that band without aliasing the
    # zero-frequency correlation integral.
    phase = 0.37
    a = np.cos(2 * np.pi * 3 * j / fine_count + phase)
    b = np.sin(2 * np.pi * 7 * j / fine_count - 0.2)
    assert abs(check(a, b)) < 1e-12

    # One factor crosses the coarse Nyquist band. A product component lands
    # at a coarse-grid alias bin and produces a real sampling discrepancy.
    high = coarse_count // 2 + 4
    low = coarse_count - high
    a = np.cos(2 * np.pi * high * j / fine_count)
    b = np.cos(2 * np.pi * low * j / fine_count)
    assert abs(check(a, b)) > 0.1

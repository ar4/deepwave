"""Tests for deepwave.wavelets."""

import inspect
import math
import typing
import warnings

import pytest
import torch

import deepwave
from deepwave.wavelets import gaussian_pulse, ormsby, ricker, tone_burst


def test_ricker_basic_functionality() -> None:
    """Test the basic functionality of the ricker wavelet generation."""
    freq = 25
    length = 100
    dt = 0.001
    peak_time = 0.05
    wavelet = ricker(freq, length, dt, peak_time)

    assert isinstance(wavelet, torch.Tensor)
    assert wavelet.shape == (length,)
    assert wavelet.dtype == torch.float32  # Default dtype

    # Check peak value (should be 1.0 at peak_time)
    # The peak time is 0.05, which corresponds to index 50 (0.05 / 0.001)
    assert torch.isclose(wavelet[50], torch.tensor(1.0))

    # Check values at start and end (should be close to zero)
    assert torch.isclose(wavelet[0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(wavelet[-1], torch.tensor(0.0), atol=1e-4)  # Increased atol


def test_ricker_different_freq() -> None:
    """Test ricker wavelet generation with different frequencies."""
    length = 100
    dt = 0.001
    peak_time = 0.05

    wavelet_low_freq = ricker(10, length, dt, peak_time)
    wavelet_high_freq = ricker(50, length, dt, peak_time)

    # Higher frequency wavelet should be narrower
    assert torch.argmax(wavelet_low_freq) == torch.argmax(
        wavelet_high_freq,
    )  # Peak time should be the same
    # Check that values away from the peak decay faster for higher freq
    assert abs(wavelet_high_freq[40]) < abs(wavelet_low_freq[40])


def test_ricker_different_length() -> None:
    """Test ricker wavelet generation with different lengths."""
    freq = 25
    dt = 0.001

    wavelet_short = ricker(freq, 50, dt, 0.02)  # Adjusted peak_time to be within bounds
    wavelet_long = ricker(freq, 200, dt, 0.05)

    assert wavelet_short.shape == (50,)
    assert wavelet_long.shape == (200,)
    assert torch.isclose(wavelet_short[20], torch.tensor(1.0))  # 0.02 / 0.001 = 20
    assert torch.isclose(wavelet_long[50], torch.tensor(1.0))


def test_ricker_different_dt() -> None:
    """Test ricker wavelet generation with different time steps (dt)."""
    freq = 25
    length = 100

    wavelet_small_dt = ricker(freq, length, 0.0005, 0.04)  # Adjusted peak_time
    wavelet_large_dt = ricker(freq, length, 0.002, 0.05)

    # Peak index should change with dt (peak_time / dt)
    assert torch.isclose(wavelet_small_dt[80], torch.tensor(1.0))  # 0.04 / 0.0005 = 80
    assert torch.isclose(wavelet_large_dt[25], torch.tensor(1.0))  # 0.05 / 0.002 = 25


def test_ricker_different_peak_time() -> None:
    """Test ricker wavelet generation with different peak times."""
    freq = 25
    length = 100
    dt = 0.001

    wavelet_early_peak = ricker(freq, length, dt, 0.02)
    wavelet_late_peak = ricker(freq, length, dt, 0.08)

    assert torch.isclose(wavelet_early_peak[20], torch.tensor(1.0))  # 0.02 / 0.001 = 20
    assert torch.isclose(wavelet_late_peak[80], torch.tensor(1.0))  # 0.08 / 0.001 = 80


def test_ricker_default_peak_time_centres_wavelet() -> None:
    """Omitting peak_time auto-centres the ricker wavelet in the record."""
    freq = 25
    length = 100
    dt = 0.001

    auto = ricker(freq, length, dt)  # peak_time omitted
    # The peak of the (symmetric) wavelet sits at the record centre.
    assert int(torch.argmax(auto)) == length // 2
    # Equivalent to the explicit centred value, bit-for-bit.
    explicit = ricker(freq, length, dt, length * dt / 2)
    assert torch.equal(auto, explicit)


def test_ricker_explicit_peak_time_unchanged() -> None:
    """An explicit peak_time is honoured exactly (back-compat preserved)."""
    freq = 25
    length = 100
    dt = 0.001
    w = ricker(freq, length, dt, 0.02)
    assert torch.isclose(w[20], torch.tensor(1.0))


def test_ricker_different_dtype() -> None:
    """Test ricker wavelet generation with different dtypes."""
    freq = 25
    length = 100
    dt = 0.001
    peak_time = 0.05

    wavelet_float32 = ricker(freq, length, dt, peak_time, dtype=torch.float32)
    wavelet_float64 = ricker(freq, length, dt, peak_time, dtype=torch.float64)

    assert wavelet_float32.dtype == torch.float32
    assert wavelet_float64.dtype == torch.float64
    assert torch.isclose(wavelet_float32[50], torch.tensor(1.0, dtype=torch.float32))
    assert torch.isclose(wavelet_float64[50], torch.tensor(1.0, dtype=torch.float64))


def test_ricker_length_one() -> None:
    """Test ricker wavelet generation with a length of one."""
    freq = 25
    length = 1
    dt = 0.001
    peak_time = 0.05
    wavelet = ricker(freq, length, dt, peak_time)

    assert wavelet.shape == (1,)
    # The single sample should be the value of the Ricker wavelet at t = -0.05
    t = torch.tensor([-0.05])
    expected_val = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
        -(math.pi**2) * freq**2 * t**2,
    )
    assert torch.isclose(wavelet[0], expected_val[0])


def test_ricker_freq_zero_raises_error() -> None:
    """Test that ricker raises an error when frequency is not positive.

    Previously freq=0 was accepted (the formula degenerates to a constant 1,
    since ``exp(0) == 1``), but ricker now validates ``freq > 0`` (matching
    tone_burst's existing validation), so this is now a ValueError.
    """
    freq = 0
    length = 10
    dt = 0.001
    peak_time = 0.005
    with pytest.raises(ValueError, match="freq must be positive"):
        ricker(freq, length, dt, peak_time)


def test_ricker_dt_zero_raises_error() -> None:
    """Test that ricker raises an error when dt is zero."""
    freq = 25
    length = 10
    dt = 0  # dt cannot be zero
    peak_time = 0.005
    with pytest.raises(ValueError, match="dt cannot be zero"):  # Expect ValueError
        ricker(freq, length, dt, peak_time)


# ---------------------------------------------------------------------------
# ricker: nt/length §0.7 deprecation shim
# ---------------------------------------------------------------------------


def test_ricker_nt_keyword() -> None:
    """nt= is the non-deprecated way to specify the number of time samples."""
    freq = 25
    dt = 0.001
    peak_time = 0.05
    w_nt = ricker(freq, dt=dt, peak_time=peak_time, nt=100)
    w_length = ricker(freq, 100, dt, peak_time)
    assert torch.equal(w_nt, w_length)


def test_ricker_length_deprecation_warning() -> None:
    """Passing length= warns ``DeepwaveFutureWarning`` and maps to nt."""
    freq = 25
    dt = 0.001
    peak_time = 0.05
    with pytest.warns(deepwave.DeepwaveFutureWarning, match="length is deprecated"):
        w_length = ricker(freq, length=100, dt=dt, peak_time=peak_time)
    w_nt = ricker(freq, dt=dt, peak_time=peak_time, nt=100)
    assert torch.equal(w_length, w_nt)


def test_ricker_both_length_and_nt_raises_error() -> None:
    """Passing both length and nt raises a ValueError."""
    freq = 25
    dt = 0.001
    peak_time = 0.05
    with pytest.raises(
        ValueError, match="Pass only nt, not both nt and the deprecated length"
    ):
        ricker(freq, length=100, dt=dt, peak_time=peak_time, nt=100)


def test_ricker_missing_nt_raises_type_error() -> None:
    """Omitting a required argument is a ``TypeError``, as for any function.

    ``nt`` and ``dt`` carry a sentinel default only so that the deprecated
    ``length`` alias can still supply ``nt``; omitting them must be
    indistinguishable from omitting an ordinary required argument, so
    ``ricker(25.0)`` is a ``TypeError`` again as it was in ``v0.0.27``.
    """
    freq = 25.0
    dt = 0.001
    peak_time = 0.05
    with pytest.raises(TypeError, match="missing required argument: 'nt'"):
        ricker(freq, dt=dt, peak_time=peak_time)
    with pytest.raises(TypeError, match="missing required argument: 'nt'"):
        ricker(freq)
    with pytest.raises(TypeError, match="missing required argument: 'dt'"):
        ricker(freq, 100)


def test_ricker_freq_not_positive_raises_error() -> None:
    """Ricker validates freq > 0, matching tone_burst's validation style."""
    length = 10
    dt = 0.001
    peak_time = 0.005
    with pytest.raises(ValueError, match="freq must be positive"):
        ricker(-5, length, dt, peak_time)


@pytest.mark.parametrize("freq", [8.0, 15.0, 25.0, 40.0])
def test_ricker_amplitude_spectrum(freq) -> None:
    """The Ricker amplitude spectrum peaks at the central frequency.

    The existing tests check the wavelet's time-domain shape (peak value,
    timing, decay), but not its defining *spectral* property: a Ricker wavelet
    with central frequency ``freq`` has the analytic amplitude spectrum
    ``|A(f)| proportional to f**2 * exp(-f**2 / freq**2)``, which peaks exactly
    at ``f = freq``. This validates that the ``freq`` argument really is the
    central (peak) frequency, that the full spectral shape matches the closed
    form, and that the wavelet is zero-mean (zero DC component).
    """
    dt = 0.0005
    length = 4000  # long record => fine frequency resolution, little truncation
    df = 1.0 / (length * dt)
    peak_time = length * dt / 2  # centred, so the record is not truncated
    w = ricker(freq, length, dt, peak_time, dtype=torch.float64)

    spectrum = torch.fft.rfft(w).abs()
    f = torch.fft.rfftfreq(length, dt, dtype=torch.float64)

    # 1. The amplitude spectrum peaks at the central frequency (to within the
    #    discrete frequency resolution).
    peak_f = f[spectrum.argmax()].item()
    assert abs(peak_f - freq) <= df

    # 2. A Ricker is zero-mean, so its DC (zero-frequency) component vanishes.
    assert (spectrum[0] / spectrum.max()).item() < 1e-6

    # 3. The full spectral shape matches the analytic f**2 exp(-f**2/freq**2)
    #    over the energy-carrying band (measured rel. error ~5e-8).
    analytic = f**2 * torch.exp(-(f**2) / freq**2)
    band = f <= 4 * freq
    measured = spectrum[band] / spectrum[band].max()
    reference = analytic[band] / analytic[band].max()
    rel_error = (measured - reference).norm() / reference.norm()
    assert rel_error < 1e-4


# ---------------------------------------------------------------------------
# tone_burst
# ---------------------------------------------------------------------------


def test_tone_burst_shape_dtype() -> None:
    """tone_burst returns a [nt] tensor with the requested dtype."""
    w32 = tone_burst(1e6, 5, nt=500, dt=1e-8)
    assert w32.shape == (500,)
    assert w32.dtype == torch.float32
    w64 = tone_burst(1e6, 5, nt=500, dt=1e-8, dtype=torch.float64)
    assert w64.dtype == torch.float64


@pytest.mark.parametrize("freq", [0.5e6, 1e6, 2.5e6])
def test_tone_burst_peak_frequency(freq) -> None:
    """The tone-burst amplitude spectrum peaks at the centre frequency."""
    dt = 2e-9
    nt = 8000  # long record => fine frequency resolution
    n_cycles = 8  # enough cycles for a well-peaked spectrum
    df = 1.0 / (nt * dt)
    w = tone_burst(freq, n_cycles, nt=nt, dt=dt, dtype=torch.float64)
    spectrum = torch.fft.rfft(w).abs()
    f = torch.fft.rfftfreq(nt, dt, dtype=torch.float64)
    peak_f = f[spectrum.argmax()].item()
    assert abs(peak_f - freq) <= 2 * df


def test_tone_burst_duration() -> None:
    """The nonzero envelope spans exactly n_cycles / freq seconds."""
    freq = 1e6
    n_cycles = 4
    dt = 1e-8
    nt = 400
    peak_time = nt * dt / 2  # centred, fully contained
    w = tone_burst(
        freq, n_cycles, nt=nt, dt=dt, peak_time=peak_time, dtype=torch.float64
    )
    nonzero = torch.nonzero(w).view(-1)
    # Support width in seconds (add one sample for the inclusive endpoints).
    support = (nonzero[-1] - nonzero[0] + 1).item() * dt
    expected = n_cycles / freq
    assert abs(support - expected) <= 2 * dt


def test_tone_burst_default_peak_time_centres_record() -> None:
    """Omitting peak_time centres the tone burst in the record.

    ``tone_burst`` follows the same record-centring convention as
    :func:`ricker`, :func:`gaussian_pulse` and :func:`ormsby`:
    ``peak_time = nt * dt / 2``.
    """
    freq = 1e6
    n_cycles = 5
    dt = 2e-9
    nt = 4000

    auto = tone_burst(freq, n_cycles, nt=nt, dt=dt, dtype=torch.float64)
    explicit = tone_burst(
        freq, n_cycles, nt=nt, dt=dt, peak_time=nt * dt / 2, dtype=torch.float64
    )
    assert torch.equal(auto, explicit)

    # The envelope's centre of mass sits at the record centre, to within one
    # sample. (Use the envelope, i.e. |signal|, not the signal: the carrier is
    # an odd function about the peak, so the signal's own centre of mass is
    # dominated by cancellation rather than by the burst's position.)
    t = torch.arange(nt, dtype=torch.float64) * dt
    envelope = auto.abs()
    centre_of_mass = (t * envelope).sum() / envelope.sum()
    assert abs(centre_of_mass.item() - nt * dt / 2) <= dt


def test_tone_burst_window_centred_peak_time_still_available() -> None:
    """The window-centred burst is still reachable as an explicit peak_time.

    ``tone_burst``'s default ``peak_time`` centres the burst in the record
    (``nt * dt / 2``). Passing ``peak_time=n_cycles / freq / 2`` instead
    centres the burst *window*, so that the burst begins at ``t = 0``; this
    test pins that waveform sample-for-sample against future drift.
    """
    freq = 1e6
    n_cycles = 5
    dt = 2e-9
    nt = 4000
    duration = n_cycles / freq

    window_centred = tone_burst(
        freq, n_cycles, nt=nt, dt=dt, peak_time=duration / 2, dtype=torch.float64
    )

    # Reconstruct the waveform independently of wavelets.py: a Hann-tapered
    # sinusoid whose window spans exactly [0, duration].
    t = torch.arange(nt, dtype=torch.float64) * dt - duration / 2
    tau = t / duration
    expected = torch.where(
        tau.abs() <= 0.5,
        0.5 * (1 + torch.cos(2 * math.pi * tau)) * torch.sin(2 * math.pi * freq * t),
        torch.zeros_like(t),
    )
    assert torch.allclose(window_centred, expected, atol=1e-15)

    # It really is the burst-begins-at-zero placement: the support starts at
    # the first sample and is one burst duration wide.
    nonzero = torch.nonzero(window_centred).view(-1)
    assert int(nonzero[0]) <= 1
    assert abs((int(nonzero[-1]) + 1) * dt - duration) <= 2 * dt

    # ...and it is not what the default now gives.
    default = tone_burst(freq, n_cycles, nt=nt, dt=dt, dtype=torch.float64)
    assert not torch.equal(default, window_centred)
    shift_samples = round((nt * dt / 2 - duration / 2) / dt)
    assert shift_samples == 750
    assert int(default.abs().argmax()) - int(window_centred.abs().argmax()) == (
        shift_samples
    )


def test_tone_burst_windows_and_errors() -> None:
    """Rect window is untapered; unknown window/invalid args raise."""
    dt = 1e-8
    # Long enough for the burst plus its trailing zero, so that the
    # truncation warning (tested separately) does not fire here: at nt = 400
    # the record is exactly the burst duration and the rectangular window
    # stops 6% of the way up its last cycle.
    nt = 500
    rect = tone_burst(1e6, 4, nt=nt, dt=dt, window="rect", dtype=torch.float64)
    hann = tone_burst(1e6, 4, nt=nt, dt=dt, window="hann", dtype=torch.float64)
    # The rectangular burst reaches unit amplitude; the Hann-tapered one is
    # strictly smaller in peak envelope.
    assert rect.abs().max() >= hann.abs().max()
    with pytest.raises(ValueError, match="Unknown window"):
        tone_burst(1e6, 4, nt=nt, dt=dt, window="nope")
    with pytest.raises(ValueError, match="dt cannot be zero"):
        tone_burst(1e6, 4, nt=nt, dt=0.0)
    with pytest.raises(ValueError, match="freq must be positive"):
        tone_burst(0.0, 4, nt=nt, dt=dt)


def test_tone_burst_warns_when_the_record_is_too_short() -> None:
    """A record shorter than the burst truncates it, and says so.

    The record-centred default ``peak_time`` places the middle of the burst
    at ``nt * dt / 2``, so a record shorter than the burst cuts off its
    leading half: the wavelet steps from zero to a large value between the
    (absent) sample before the record and its first sample. That step is
    broadband, which defeats the point of choosing ``freq`` and
    ``n_cycles``, and nothing else in the call is an error.
    """
    with pytest.warns(UserWarning, match="tone_burst is truncated"):
        truncated = tone_burst(3.0, 3, nt=50, dt=0.01, dtype=torch.float64)
    # The artefact the warning is about: a half-amplitude step at t = 0.
    assert abs(float(truncated[0])) > 0.1


def test_tone_burst_truncation_warning_names_both_ends() -> None:
    """The message says which end was cut, and suggests a workable ``nt``."""
    with pytest.warns(UserWarning, match="tone_burst is truncated") as caught:
        tone_burst(3.0, 3, nt=50, dt=0.01, dtype=torch.float64)
    message = str(caught[0].message)
    assert "start" in message
    assert "end" in message
    # nt >= ceil(duration / dt) + 2 = 102 holds the burst and its two zeros.
    assert "nt >= 102" in message
    with pytest.warns(UserWarning, match="tone_burst is truncated") as caught:
        # The window-centred peak time puts the leading edge exactly at
        # t = 0, so only the trailing end is cut.
        tone_burst(3.0, 3, nt=50, dt=0.01, peak_time=0.5, dtype=torch.float64)
    message = str(caught[0].message)
    assert "start" not in message
    assert "end" in message


@pytest.mark.parametrize(
    ("freq", "n_cycles", "nt", "dt"),
    [
        (3.0, 5, 400, 0.01),
        (1e6, 5, 4000, 2e-9),
        (2.5e6, 3, 1200, 4e-9),
        (1e6, 4, 500, 1e-8),
    ],
)
def test_tone_burst_does_not_warn_when_the_burst_fits(
    freq: float, n_cycles: float, nt: int, dt: float
) -> None:
    """A record that holds the burst is silent, and starts and ends at zero."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wavelet = tone_burst(freq, n_cycles, nt=nt, dt=dt, dtype=torch.float64)
    truncation = [
        entry for entry in caught if "tone_burst is truncated" in str(entry.message)
    ]
    assert not truncation, [str(entry.message) for entry in truncation]
    assert float(wavelet[0]) == 0.0
    assert float(wavelet[-1]) == 0.0


# ---------------------------------------------------------------------------
# gaussian_pulse
# ---------------------------------------------------------------------------


def test_gaussian_pulse_shape_dtype() -> None:
    """gaussian_pulse returns a [nt] tensor with the requested dtype."""
    w = gaussian_pulse(1e6, 0.5, nt=500, dt=1e-8)
    assert w.shape == (500,)
    assert w.dtype == torch.float32
    w64 = gaussian_pulse(1e6, 0.5, nt=500, dt=1e-8, dtype=torch.float64)
    assert w64.dtype == torch.float64


@pytest.mark.parametrize("freq", [0.5e6, 1e6, 2e6])
def test_gaussian_pulse_peak_frequency(freq) -> None:
    """The Gaussian-pulse amplitude spectrum peaks at the centre frequency."""
    dt = 2e-9
    nt = 8000
    df = 1.0 / (nt * dt)
    w = gaussian_pulse(freq, 0.6, nt=nt, dt=dt, dtype=torch.float64)
    spectrum = torch.fft.rfft(w).abs()
    f = torch.fft.rfftfreq(nt, dt, dtype=torch.float64)
    peak_f = f[spectrum.argmax()].item()
    assert abs(peak_f - freq) <= 2 * df


@pytest.mark.parametrize("bw", [0.3, 0.5, 0.8])
def test_gaussian_pulse_bandwidth(bw) -> None:
    """The measured -6 dB fractional bandwidth matches bandwidth_frac.

    The magnitude spectrum should fall to half its peak (-6 dB) at
    ``freq * (1 +/- bw/2)``; the measured fractional bandwidth
    ``(f_hi - f_lo) / freq`` must therefore match ``bw``.
    """
    freq = 1e6
    dt = 2e-9
    nt = 16000  # fine frequency resolution for an accurate bandwidth read
    w = gaussian_pulse(freq, bw, nt=nt, dt=dt, dtype=torch.float64)
    spectrum = torch.fft.rfft(w).abs()
    f = torch.fft.rfftfreq(nt, dt, dtype=torch.float64)
    spectrum = spectrum / spectrum.max()
    # -6 dB = half amplitude. Find the crossing frequencies around the peak,
    # linearly interpolating between bins to beat the frequency resolution.
    peak_idx = int(spectrum.argmax())
    half = 0.5

    def _cross(idx_lo: int, idx_hi: int) -> float:
        s_lo, s_hi = spectrum[idx_lo].item(), spectrum[idx_hi].item()
        frac = (half - s_lo) / (s_hi - s_lo)
        return (f[idx_lo] + frac * (f[idx_hi] - f[idx_lo])).item()

    # Lower crossing: last index below the peak where spectrum rises through 0.5.
    below = (spectrum[:peak_idx] <= half).nonzero().view(-1)
    lo_i = int(below.max())
    f_lo = _cross(lo_i, lo_i + 1)
    # Upper crossing: first index after the peak where spectrum falls below 0.5.
    above = (spectrum[peak_idx:] <= half).nonzero().view(-1)
    hi_i = peak_idx + int(above.min())
    f_hi = _cross(hi_i, hi_i - 1)
    measured_bw = (f_hi - f_lo) / freq
    assert abs(measured_bw - bw) < 0.02


def test_gaussian_pulse_errors() -> None:
    """Invalid arguments raise."""
    with pytest.raises(ValueError, match="dt cannot be zero"):
        gaussian_pulse(1e6, 0.5, nt=400, dt=0.0)
    with pytest.raises(ValueError, match="bandwidth_frac must be positive"):
        gaussian_pulse(1e6, 0.0, nt=400, dt=1e-8)


# ---------------------------------------------------------------------------
# ormsby
# ---------------------------------------------------------------------------


def test_ormsby_shape_dtype() -> None:
    """Ormsby returns a [nt] tensor with the requested dtype."""
    f1, f2, f3, f4 = 5.0, 10.0, 40.0, 60.0
    dt = 0.001
    nt = 500
    peak_time = nt * dt / 2
    w32 = ormsby(f1, f2, f3, f4, nt=nt, dt=dt, peak_time=peak_time)
    assert w32.shape == (nt,)
    assert w32.dtype == torch.float32
    w64 = ormsby(f1, f2, f3, f4, nt=nt, dt=dt, peak_time=peak_time, dtype=torch.float64)
    assert w64.dtype == torch.float64


def test_ormsby_peak_amplitude_and_time() -> None:
    """Peak absolute amplitude is 1.0, at (within a sample of) peak_time."""
    f1, f2, f3, f4 = 5.0, 10.0, 40.0, 60.0
    dt = 0.001
    nt = 8000
    peak_time = nt * dt / 2
    w = ormsby(f1, f2, f3, f4, nt=nt, dt=dt, peak_time=peak_time, dtype=torch.float64)
    assert torch.isclose(w.abs().max(), torch.tensor(1.0, dtype=torch.float64))
    peak_idx = int(torch.argmax(w.abs()))
    expected_idx = round(peak_time / dt)
    assert abs(peak_idx - expected_idx) <= 1


@pytest.mark.parametrize(
    ("f1", "f2", "f3", "f4"),
    [(5.0, 10.0, 40.0, 60.0), (2.0, 8.0, 15.0, 20.0), (1.0, 3.0, 5.0, 7.0)],
)
def test_ormsby_amplitude_spectrum(f1, f2, f3, f4) -> None:
    """The amplitude spectrum is flat in [f2, f3] and ~zero outside [f1, f4].

    Verified numerically via FFT: the Ormsby wavelet's defining property is a
    trapezoidal amplitude spectrum (zero below f1, a linear ramp up to f2,
    flat between f2 and f3, a linear ramp down to f4, zero above f4). This
    checks that :func:`ormsby`'s time-domain formula (built from
    ``torch.sinc``) really does produce that frequency-domain shape, rather
    than just trusting the formula transcription.
    """
    dt = 1.0 / (8 * f4)  # comfortably satisfies Nyquist for f4
    nt = 16000  # long record => fine frequency resolution, little leakage
    peak_time = nt * dt / 2
    w = ormsby(f1, f2, f3, f4, nt=nt, dt=dt, peak_time=peak_time, dtype=torch.float64)

    spectrum = torch.fft.rfft(w).abs()
    f = torch.fft.rfftfreq(nt, dt, dtype=torch.float64)
    spectrum = spectrum / spectrum.max()

    # Flat (unity) passband between f2 and f3.
    passband = (f >= f2) & (f <= f3)
    assert torch.allclose(
        spectrum[passband], torch.ones_like(spectrum[passband]), atol=0.02
    )

    # Comfortably below f1 and comfortably above f4: ~zero.
    margin = 0.5 * (f2 - f1)
    below = f <= max(f1 - margin, 0.0)
    above = f >= f4 + margin
    assert spectrum[below].max().item() < 0.02
    assert spectrum[above].max().item() < 0.02


def test_ormsby_corner_frequency_validation() -> None:
    """0 < f1 < f2 < f3 < f4 is required, else ValueError."""
    nt = 100
    dt = 0.001
    peak_time = nt * dt / 2
    with pytest.raises(ValueError, match="0 < f1 < f2 < f3 < f4"):
        ormsby(10.0, 5.0, 40.0, 60.0, nt=nt, dt=dt, peak_time=peak_time)  # f1 > f2
    with pytest.raises(ValueError, match="0 < f1 < f2 < f3 < f4"):
        ormsby(0.0, 10.0, 40.0, 60.0, nt=nt, dt=dt, peak_time=peak_time)  # f1 == 0
    with pytest.raises(ValueError, match="0 < f1 < f2 < f3 < f4"):
        ormsby(5.0, 10.0, 60.0, 40.0, nt=nt, dt=dt, peak_time=peak_time)  # f3 > f4


def test_ormsby_dt_zero_raises_error() -> None:
    """Test that ormsby raises an error when dt is zero."""
    with pytest.raises(ValueError, match="dt cannot be zero"):
        ormsby(5.0, 10.0, 40.0, 60.0, nt=100, dt=0.0, peak_time=0.05)


def test_ormsby_default_peak_time_centres_wavelet() -> None:
    """Omitting peak_time auto-centres ormsby, matching ricker's convention."""
    f1, f2, f3, f4 = 5.0, 10.0, 40.0, 60.0
    nt = 100
    dt = 0.001

    auto = ormsby(f1, f2, f3, f4, nt=nt, dt=dt)  # peak_time omitted
    explicit = ormsby(f1, f2, f3, f4, nt=nt, dt=dt, peak_time=nt * dt / 2)
    assert torch.equal(auto, explicit)


# ---------------------------------------------------------------------------
# Canonical signature order: (freq..., nt, dt, peak_time=None, dtype=None)
# ---------------------------------------------------------------------------


_CANONICAL_TAIL = ["nt", "dt", "peak_time", "dtype"]


@pytest.mark.parametrize(
    ("func", "n_freq_params", "tail_is_positional", "extra_kwonly"),
    [
        (ricker, 1, True, {"length", "shape"}),
        (tone_burst, 2, False, {"window", "shape"}),
        (gaussian_pulse, 2, False, {"shape"}),
        (ormsby, 4, False, {"shape"}),
    ],
)
def test_wavelet_canonical_signature_order(
    func, n_freq_params, tail_is_positional, extra_kwonly
) -> None:
    """All four wavelets declare (freq..., nt, dt, peak_time, dtype) in order.

    The order is the same everywhere; only the *kind* differs. ``ricker``
    shipped in ``v0.0.27`` with a positional tail, so
    ``ricker(freq, nt, dt, peak_time)`` -- the call in every tutorial -- keeps
    working. The three wavelets that only ever existed on this branch make
    the tail keyword-only, which is why ``nt`` being parameter 3, 3 and 5
    across them can no longer be confused.

    Non-canonical extras (``ricker``'s deprecated ``length``,
    ``tone_burst``'s ``window``, and the ``shape`` argument that all four
    share) are excluded from the order check but confirmed to still exist as
    keyword-only parameters.
    """
    params = list(inspect.signature(func).parameters.values())
    freq_params = params[:n_freq_params]
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in freq_params
    ), "the frequency parameters stay positional on every wavelet"

    tail = [p.name for p in params[n_freq_params:] if p.name in _CANONICAL_TAIL]
    assert tail == _CANONICAL_TAIL

    expected_kind = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
        if tail_is_positional
        else inspect.Parameter.KEYWORD_ONLY
    )
    kinds = {p.name: p.kind for p in params}
    for name in _CANONICAL_TAIL:
        assert kinds[name] is expected_kind, name

    kwonly = {p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY}
    assert extra_kwonly <= kwonly


@pytest.mark.parametrize(
    ("func", "positional_args"),
    [
        (tone_burst, (1e6, 5)),
        (gaussian_pulse, (1e6, 0.5)),
        (ormsby, (5.0, 10.0, 40.0, 60.0)),
    ],
)
def test_branch_only_wavelets_are_keyword_only(func, positional_args) -> None:
    """``tone_burst``/``gaussian_pulse``/``ormsby`` reject positional nt/dt.

    None of the three shipped in ``v0.0.27``, so no released code can be
    passing them positionally.
    """
    with pytest.raises(TypeError):
        func(*positional_args, 400, 1e-8)
    # The keyword form is what works.
    assert func(*positional_args, nt=400, dt=1e-8).shape == (400,)


def test_ricker_positional_v0027_form_still_works() -> None:
    """``ricker(freq, nt, dt, peak_time)`` -- the v0.0.27 call -- is unchanged."""
    freq, nt, dt, peak_time = 25.0, 100, 0.004, 0.2
    positional = ricker(freq, nt, dt, peak_time)
    keyword = ricker(freq, nt=nt, dt=dt, peak_time=peak_time)
    assert torch.equal(positional, keyword)


def test_ricker_nt_dt_annotated_non_optional() -> None:
    """``nt``/``dt`` annotate as ``int``/``float``, not ``Optional[...]``.

    The sentinel default exists so that the annotation can stay honest while
    the deprecated ``length`` alias still supplies ``nt``.
    """
    hints = typing.get_type_hints(ricker)
    assert hints["nt"] is int
    assert hints["dt"] is float


# ---------------------------------------------------------------------------
# The keyword-only shape= argument, shared by all four builders
# ---------------------------------------------------------------------------

_NT = 200

#: ``(builder, positional args)`` for each wavelet, all sharing ``nt``/``dt``
#: so that ``shape=`` can be exercised identically on every one. ``ricker``
#: takes ``nt``/``dt`` positionally; the other three take them as keywords,
#: so both are passed by keyword here (which ``ricker`` also accepts).
_BUILDERS = [
    (ricker, (25.0,)),
    (tone_burst, (25.0, 5)),
    (gaussian_pulse, (25.0, 0.5)),
    (ormsby, (5.0, 10.0, 40.0, 60.0)),
]

_BUILDER_IDS = ["ricker", "tone_burst", "gaussian_pulse", "ormsby"]


@pytest.mark.parametrize(("func", "args"), _BUILDERS, ids=_BUILDER_IDS)
@pytest.mark.parametrize("shape", [(), 3, (2,), (2, 3), (1, 1), (4, 1, 2)])
def test_wavelet_shape_equals_expand_of_default(func, args, shape) -> None:
    """``shape=`` returns exactly ``.expand`` of the ``shape=None`` wavelet.

    ``nt`` is always the last dimension, an ``int`` ``n`` is shorthand for
    ``(n,)``, and ``()`` is the same as ``None``.
    """
    dt = 0.004
    base = func(*args, nt=_NT, dt=dt, dtype=torch.float64)
    assert base.shape == (_NT,)

    got = func(*args, nt=_NT, dt=dt, dtype=torch.float64, shape=shape)
    dims = (shape,) if isinstance(shape, int) else tuple(shape)
    assert got.shape == (*dims, _NT)
    assert torch.equal(got, base.expand(*dims, _NT))


@pytest.mark.parametrize(("func", "args"), _BUILDERS, ids=_BUILDER_IDS)
def test_wavelet_shape_is_a_view_not_a_copy(func, args) -> None:
    """The expanded wavelet is a stride-zero view sharing one copy of the data."""
    w = func(*args, nt=_NT, dt=0.004, dtype=torch.float64, shape=(7, 5))
    assert w.shape == (7, 5, _NT)
    # ``expand`` gives the added dimensions stride zero, so the whole tensor
    # is backed by one ``nt``-element buffer rather than 35 copies of it.
    assert w.stride()[:2] == (0, 0)
    assert w.untyped_storage().size() == _NT * w.element_size()
    assert not w.is_contiguous()


@pytest.mark.parametrize(("func", "args"), _BUILDERS, ids=_BUILDER_IDS)
def test_wavelet_shape_gradients_flow(func, args) -> None:
    """Gradients flow back through the expanded form to a scaling parameter.

    ``expand`` is differentiable (its backward sums over the broadcast
    dimensions), so an expanded wavelet can be used inside an optimisation
    just as the ``[nt]`` form can.
    """
    scale = torch.ones((), dtype=torch.float64, requires_grad=True)
    n_shots, n_sources = 2, 3
    w = func(*args, nt=_NT, dt=0.004, dtype=torch.float64, shape=(n_shots, n_sources))
    # ``abs`` so that the expected gradient is a healthy positive number
    # rather than the near-cancelling sum of a zero-mean wavelet.
    (w.abs() * scale).sum().backward()
    assert scale.grad is not None
    base = func(*args, nt=_NT, dt=0.004, dtype=torch.float64)
    # Every one of the n_shots * n_sources copies contributed.
    assert scale.grad.item() > 0
    assert torch.isclose(scale.grad, base.abs().sum() * n_shots * n_sources)


@pytest.mark.parametrize(("func", "args"), _BUILDERS, ids=_BUILDER_IDS)
def test_wavelet_shape_invalid_raises(func, args) -> None:
    """A non-positive or non-integer shape raises, naming the shape it saw."""
    kwargs = {"nt": _NT, "dt": 0.004}
    with pytest.raises(ValueError, match=r"shape must be positive, got shape=\(0,\)"):
        func(*args, **kwargs, shape=0)
    with pytest.raises(ValueError, match=r"shape must be positive, got shape=\(2, 0\)"):
        func(*args, **kwargs, shape=(2, 0))
    with pytest.raises(ValueError, match=r"shape must be positive, got shape=\(-1,\)"):
        func(*args, **kwargs, shape=(-1,))
    with pytest.raises(TypeError, match=r"shape must be an int, got shape=\(2\.5,\)"):
        func(*args, **kwargs, shape=(2.5,))
    with pytest.raises(TypeError, match=r"shape must be None, a positive int"):
        func(*args, **kwargs, shape=2.5)
    with pytest.raises(TypeError, match=r"shape must be None, a positive int"):
        func(*args, **kwargs, shape=True)


@pytest.mark.parametrize(("func", "args"), _BUILDERS, ids=_BUILDER_IDS)
def test_wavelet_shape_errors_are_deepwave_errors(func, args) -> None:
    """The shape errors are Deepwave's own exception types."""
    with pytest.raises(deepwave.DeepwaveValueError):
        func(*args, nt=_NT, dt=0.004, shape=0)
    with pytest.raises(deepwave.DeepwaveTypeError):
        func(*args, nt=_NT, dt=0.004, shape=2.5)


@pytest.mark.parametrize(("func", "args"), _BUILDERS, ids=_BUILDER_IDS)
def test_wavelet_shape_default_none_is_unchanged(func, args) -> None:
    """``shape=None`` is exactly the pre-existing ``[nt]`` behaviour."""
    explicit_none = func(*args, nt=_NT, dt=0.004, dtype=torch.float64, shape=None)
    omitted = func(*args, nt=_NT, dt=0.004, dtype=torch.float64)
    assert explicit_none.shape == (_NT,)
    assert torch.equal(explicit_none, omitted)


def test_ricker_shape_with_positional_nt_dt() -> None:
    """``shape=`` composes with ``ricker``'s positional ``v0.0.27`` call."""
    w = ricker(25.0, 200, 0.004, 0.06, shape=(2, 3))
    assert w.shape == (2, 3, 200)
    assert torch.equal(w, ricker(25.0, 200, 0.004, 0.06).expand(2, 3, 200))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("parameter", ["freq", "dt", "peak_time"])
@pytest.mark.parametrize("wavelet", [ricker, tone_burst, gaussian_pulse])
def test_wavelet_rejects_nonfinite_parameters(wavelet, parameter, value):
    """Reject nonfinite time and frequency inputs before wavelet arithmetic."""
    kwargs = {"freq": 25.0, "nt": 100, "dt": 0.001, parameter: value}
    if wavelet is tone_burst:
        kwargs["n_cycles"] = 3.0
    elif wavelet is gaussian_pulse:
        kwargs["bandwidth_frac"] = 0.5
    with pytest.raises(deepwave.DeepwaveValueError, match="finite"):
        wavelet(**kwargs)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_ormsby_rejects_nonfinite_corner(value):
    """Reject an infinite upper corner as well as nonordered frequencies."""
    with pytest.raises(deepwave.DeepwaveValueError, match="Corner frequencies"):
        ormsby(5.0, 10.0, 20.0, value, nt=100, dt=0.001)

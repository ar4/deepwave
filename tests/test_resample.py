import torch
from deepwave.common import upsample, downsample


def test_spike_upsample(n=128, step_ratio=2, dtype=torch.double,
                        device=None):
    """Test that padding mitigates wraparound."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(n, dtype=dtype, device=device)
    x[0] = 1
    y = upsample(x, step_ratio, freq_taper_frac=0.25, time_pad_frac=1.0)
    assert y[len(y)//2:].abs().max() < 1e-4


def test_spike_downsample(n=128, step_ratio=2, dtype=torch.double,
                          device=None):
    """Test that padding mitigates wraparound."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(n, dtype=dtype, device=device)
    x[1] = 1
    y = downsample(x, step_ratio, freq_taper_frac=0.25, time_pad_frac=1.0)
    assert y[len(y)//2:].abs().max() < 2e-4

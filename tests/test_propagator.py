import torch
import pytest
import deepwave

NZ = 5
NY = 3
DX = 5.0


@pytest.fixture
def prop():
    """Return a propagator."""
    model = torch.ones(NZ, NY) * 1500
    dx = DX
    return deepwave.Scalar(model, dx)


def test_passes(prop):
    """Check that the test passes when everything is correct."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_wrong_num_dims(prop):
    """Check error when Tensors don't have right num dims."""
    source_amplitudes = torch.zeros(1, 1)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_shape_mismatch1(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_shape_mismatch2(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 2, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_shape_mismatch3(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 3, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_shape_mismatch4(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(2, 1, 1)
    source_locations = torch.zeros(1, 1, 3, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_source_outside_model(prop):
    """Check error when source location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = -1.0
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_receiver_outside_model(prop):
    """Check error when receiver location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 1] = NY
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_passes_wavefield(prop):
    """Check passes when wavefields have the right size."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(2, NZ + 2 * 22, NY + 2 * 22)
    prop(dt, source_amplitudes, source_locations, receiver_locations,
         wavefield_0=wavefield_0)


def test_wrong_wavefield_batch_size1(prop):
    """Check error when wavefield batch size too big."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(3, NZ + 2 * 22, NY + 2 * 22)
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations,
             wavefield_0=wavefield_0)


def test_wrong_wavefield_batch_size2(prop):
    """Check error when wavefield batch size too small."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(1, NZ + 2 * 22, NY + 2 * 22)
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations,
             wavefield_0=wavefield_0)


def test_wavefield_too_big1(prop):
    """Check error when wavefield is bigger than model."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(2, NZ + 2 * 22 + 1, NY + 2 * 22)
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations,
             wavefield_0=wavefield_0)


def test_wavefield_too_big2(prop):
    """Check error when wavefield is bigger than model."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(2, NZ + 2 * 22, NY + 2 * 22 + 1)
    with pytest.raises(RuntimeError):
        prop(dt, source_amplitudes, source_locations, receiver_locations,
             wavefield_0=wavefield_0)

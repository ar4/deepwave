import torch
import pytest
import deepwave.base.propagator

NZ = 5
NY = 3
DX = 5.0


class PropagatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                source_amplitudes,
                source_locations,
                receiver_locations,
                dt, model, property_names, vp):
        return vp


@pytest.fixture
def prop():
    """Return a propagator."""
    propfunc = PropagatorFunction
    model = torch.ones(NZ, NY) * 1500
    dx = DX
    fd_width = 2
    return deepwave.base.propagator.Propagator(propfunc, {'vp': model},
                                               dx, fd_width)


def test_passes(prop):
    """Check that the test passes when everything is correct."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.004
    prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_int_dt(prop):
    """Check error when dt is an int."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 1
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_zero_dt(prop):
    """Check error when dt is 0."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.0
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_negative_dt(prop):
    """Check error when dt is negative."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = -0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_different_device_to_model(prop):
    """Check error when Tensors are not all on same device."""
    if torch.cuda.is_available():
        source_amplitudes = torch.zeros(1, 1, 1).cuda()
        source_locations = torch.zeros(1, 1, 2)
        receiver_locations = torch.zeros(1, 1, 2)
        dt = 0.004
        with pytest.raises(RuntimeError):
            prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_wrong_num_dims(prop):
    """Check error when Tensors don't have right num dims."""
    source_amplitudes = torch.zeros(1, 1)
    source_locations = torch.zeros(1, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_shape_mismatch1(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(2, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_shape_mismatch2(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 2, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_shape_mismatch3(prop):
    """Check error when Tensors not right shape."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 3)
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_source_outside_model(prop):
    """Check error when source location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 2)
    source_locations[0, 0, 0] = -1.0
    receiver_locations = torch.zeros(1, 1, 2)
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)


def test_receiver_outside_model(prop):
    """Check error when receiver location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 1)
    source_locations = torch.zeros(1, 1, 2)
    receiver_locations = torch.zeros(1, 1, 2)
    receiver_locations[0, 0, 1] = NY * DX
    dt = 0.004
    with pytest.raises(RuntimeError):
        prop(source_amplitudes, source_locations, receiver_locations, dt)

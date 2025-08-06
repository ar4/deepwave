import pytest
import torch
from test_utils import direct_2d_approx

import deepwave
from deepwave.location_interpolation import Hicks, _get_hicks_for_one_location_dim


def test_hicks_init_invalid_locations_type():
    with pytest.raises(TypeError, match="locations must be a torch.Tensor."):
        Hicks([1, 2, 3])


def test_hicks_init_invalid_locations_ndim():
    locations = torch.randn(10, 2)  # Should be 3D
    with pytest.raises(RuntimeError, match="locations must have three dimensions."):
        Hicks(locations)


def test_hicks_init_zero_batch_size():
    locations = torch.randn(0, 10, 2)
    hicks = Hicks(locations)
    assert hicks.locations.shape[0] == 0


def test_hicks_init_zero_sources_per_shot():
    locations = torch.randn(10, 0, 2)
    hicks = Hicks(locations)
    assert hicks.locations.shape[1] == 0


def test_hicks_init_invalid_dtype():
    locations = torch.randn(10, 10, 2)
    with pytest.raises(TypeError, match="dtype must be a torch.dtype."):
        Hicks(locations, dtype="float")


def test_monopole(
    c=1500,
    freq=25,
    dx=(5, 5),
    dt=0.001,
    nx=(50, 50),
    nt=200,
    device=None,
    dtype=torch.double,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peak_time = 1.5 / freq
    model = torch.ones(*nx, dtype=dtype, device=device) * c
    dx = torch.tensor(dx)
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time, dtype=dtype)
        .reshape(1, 1, -1)
        .to(device)
        .repeat(20, 1, 1)
    )
    source_locations = torch.zeros(20, 1, 2, device=device)
    source_locations[..., 0] = (
        torch.linspace(20.0, 20.9, 10, device=device).reshape(10, 1).repeat(2, 1)
    )
    source_locations[:10, :, 1] = 25.0
    source_locations[10:, :, 1] = 25.25
    receiver_locations = torch.zeros(20, 20, 2, device=device)
    receiver_locations[..., 0] = (
        torch.linspace(30.0, 30.9, 10, device=device).reshape(1, -1).repeat(20, 2)
    )
    receiver_locations[:10, :, 1] = 25.0
    receiver_locations[10:, :, 1] = 25.25
    hicks_source = Hicks(source_locations, dtype=dtype, model_shape=list(nx))
    hicks_source_locations = hicks_source.hicks_locations
    hicks_amplitudes = hicks_source.source(source_amplitudes)
    hicks_receiver = Hicks(receiver_locations, dtype=dtype, model_shape=list(nx))
    hicks_receiver_locations = hicks_receiver.hicks_locations
    o = deepwave.scalar(
        model,
        dx.tolist(),
        dt,
        source_amplitudes=hicks_amplitudes,
        source_locations=hicks_source_locations,
        receiver_locations=hicks_receiver_locations,
    )
    receiver_amplitudes = hicks_receiver.receiver(o[-1])
    for i, source_location in enumerate(source_locations):
        for j, receiver_location in enumerate(receiver_locations[i]):
            e = direct_2d_approx(
                receiver_location.float().cpu(),
                source_location[0].float().cpu(),
                dx,
                dt,
                c,
                -source_amplitudes[i].flatten().cpu(),
            )
            assert torch.allclose(receiver_amplitudes[i, j].cpu(), e, atol=0.025)


def test_shot_idxs(n_shots=10, n_per_shot=3, nt=5, device=None, dtype=torch.double):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    locations = torch.rand(n_shots, n_per_shot, 2, dtype=dtype, device=device) * 10
    amplitudes = torch.randn(n_shots, n_per_shot, nt, dtype=dtype, device=device)
    hicks = Hicks(locations, dtype=dtype, model_shape=[100, 100])
    hicks_amplitudes_fwd = hicks.source(amplitudes)
    shot_idxs = torch.tensor([n_shots // 2])
    hicks_amplitudes_half = hicks.source(amplitudes[shot_idxs], shot_idxs)
    assert torch.allclose(hicks_amplitudes_fwd[shot_idxs], hicks_amplitudes_half)
    shot_idxs = torch.arange(n_shots).flip(0)
    hicks_amplitudes_bwd = hicks.source(amplitudes.flip(0), shot_idxs)
    assert torch.allclose(hicks_amplitudes_fwd.flip(0), hicks_amplitudes_bwd)


def test_hicks_source_invalid_amplitudes_type():
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    with pytest.raises(TypeError, match="amplitudes must be a torch.Tensor."):
        hicks.source([1, 2, 3])


def test_hicks_source_invalid_amplitudes_ndim():
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    amplitudes = torch.randn(10)  # Should be 3D
    with pytest.raises(RuntimeError, match="amplitudes must have three dimensions."):
        hicks.source(amplitudes)


def test_hicks_receiver_invalid_amplitudes_type():
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    with pytest.raises(TypeError, match="amplitudes must be a torch.Tensor."):
        hicks.receiver([1, 2, 3])


def test_hicks_receiver_invalid_amplitudes_ndim():
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    amplitudes = torch.randn(10)  # Should be 3D
    with pytest.raises(RuntimeError, match="amplitudes must have three dimensions."):
        hicks.receiver(amplitudes)


def test_free_surface_left(halfwidth=8):
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        0.5,
        halfwidth,
        beta,
        [False, False],
        [-0.5, 19.5],
        20,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        0.5,
        halfwidth,
        beta,
        [True, False],
        [-0.5, 19.5],
        20,
        True,
    )
    assert len(l0) == 2 * halfwidth
    assert l0[0] == -halfwidth + 1
    assert len(l1) == halfwidth + 1
    assert l1[0] == 0
    assert w0[-1] == w1[-1]
    assert w0[-2] == w1[-2]
    for i in range(halfwidth - 1):
        assert w1[i] == pytest.approx(w0[halfwidth - 1 + i] - w0[halfwidth - 2 - i])


def test_free_surface_right(halfwidth=8):
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        18.5,
        halfwidth,
        beta,
        [False, False],
        [-0.5, 19.5],
        20,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        18.5,
        halfwidth,
        beta,
        [False, True],
        [-0.5, 19.5],
        20,
        True,
    )
    assert len(l0) == 2 * halfwidth
    assert l0[-1] == 19 + halfwidth - 1
    assert len(l1) == halfwidth + 1
    assert l1[-1] == 19
    assert w0[0] == w1[0]
    assert w0[1] == w1[1]
    for i in range(halfwidth - 1):
        assert w1[len(w1) - 1 - i] == pytest.approx(
            w0[len(w0) - 1 - (halfwidth - 1 + i)]
            - w0[len(w0) - 1 - (halfwidth - 2 - i)],
        )


def test_free_surface_shift_left(halfwidth=8):
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        1.5,
        halfwidth,
        beta,
        [False, False],
        [1.0, 18.0],
        20,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        1.5,
        halfwidth,
        beta,
        [True, False],
        [1.0, 18.0],
        20,
        True,
    )
    assert len(l0) == 2 * halfwidth
    assert l0[0] == 1 - halfwidth + 1
    assert len(l1) == halfwidth + 1
    assert l1[0] == 1
    assert w0[-1] == w1[-1]
    for i in range(halfwidth):
        assert w1[i] == pytest.approx(w0[halfwidth - 1 + i] - w0[halfwidth - 1 - i])


def test_free_surface_shift_right(halfwidth=8):
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        17.5,
        halfwidth,
        beta,
        [False, False],
        [1.0, 18.0],
        20,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        17.5,
        halfwidth,
        beta,
        [False, True],
        [1.0, 18.0],
        20,
        True,
    )
    assert len(l0) == 2 * halfwidth
    assert l0[-1] == 18 + halfwidth - 1
    assert len(l1) == halfwidth + 1
    assert l1[-1] == 18
    assert w0[0] == w1[0]
    for i in range(halfwidth):
        assert w1[len(w1) - 1 - i] == pytest.approx(
            w0[len(w0) - 1 - (halfwidth - 1 + i)]
            - w0[len(w0) - 1 - (halfwidth - 1 - i)],
        )


def test_get_hicks_for_one_location_dim_invalid_halfwidth():
    with pytest.raises(RuntimeError, match=r"halfwidth must be in \[1, 10\]"):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            0,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5],
            20,
        )
    with pytest.raises(RuntimeError, match=r"halfwidth must be in \[1, 10\]"):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            11,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5],
            20,
        )


def test_get_hicks_for_one_location_dim_invalid_beta():
    with pytest.raises(RuntimeError, match="beta must be non-negative."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(-1.0),
            [False, False],
            [-0.5, 19.5],
            20,
        )


def test_get_hicks_for_one_location_dim_invalid_extent():
    with pytest.raises(RuntimeError, match="extent must be a list of two floats."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            [-0.5],
            20,
        )
    with pytest.raises(RuntimeError, match="extent must be a list of two floats."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5, 20.0],
            20,
        )
    with pytest.raises(RuntimeError, match="extent must be a list of two floats."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            ["a", 19.5],
            20,
        )


def test_get_hicks_for_one_location_dim_invalid_n_grid_points():
    with pytest.raises(RuntimeError, match="n_grid_points must be positive."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5],
            0,
        )
    with pytest.raises(RuntimeError, match="n_grid_points must be positive."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5],
            -1,
        )

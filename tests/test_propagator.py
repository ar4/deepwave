import torch
import pytest
import re
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


@pytest.mark.parametrize(
    "arg_name, arg_value, expected_error_match, expected_error_type",
    [
        ("accuracy", 3, "accuracy must be 2, 4, 6, or 8", ValueError),
        ("accuracy", 4.0, "accuracy must be an int.", TypeError),
        ("pml_width", -1, "pml_width must be non-negative.", ValueError),
        (
            "pml_width",
            [1, 2, 3],
            "Expected pml_width to be of length 1 or 4, got 3.",
            RuntimeError,
        ),
        ("pml_freq", -1.0, "pml_freq must be non-negative.", ValueError),
        (
            "max_vel",
            -1.0,
            "max_vel is less than the actual maximum velocity.",
            UserWarning,
        ),  # This will be a warning, not an error
        (
            "model_gradient_sampling_interval",
            -1,
            "model_gradient_sampling_interval must be >= 0",
            ValueError,
        ),
        ("freq_taper_frac", -0.1, "freq_taper_frac must be in \[0, 1\]", ValueError),
        ("time_pad_frac", 1.1, "time_pad_frac must be in \[0, 1\]", ValueError),
        ("time_taper", "invalid", "time_taper must be a bool.", TypeError),
    ],
)
def test_scalar_forward_invalid_optional_args(
    prop, arg_name, arg_value, expected_error_match, expected_error_type
):
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    kwargs = {arg_name: arg_value}
    if expected_error_type == UserWarning:
        with pytest.warns(expected_error_type, match=expected_error_match):
            prop(dt, source_amplitudes, source_locations, receiver_locations, **kwargs)
    else:
        with pytest.raises(expected_error_type, match=expected_error_match):
            prop(dt, source_amplitudes, source_locations, receiver_locations, **kwargs)


@pytest.mark.parametrize(
    "source_amplitudes_shape, source_locations_shape, receiver_locations_shape, expected_error_match",
    [
        (
            (1, 1),
            (1, 1, 2),
            (1, 1, 2),
            "source amplitudes Tensors should have 3 dimensions",
        ),  # wrong num dims
        (
            (1, 1, 2),
            (2, 1, 2),
            (1, 1, 2),
            "Expected source amplitudes to have size 2 in the batch dimension",
        ),  # shape mismatch 1
        (
            (1, 1, 2),
            (1, 2, 2),
            (1, 1, 2),
            "Expected source amplitudes and locations to be the same size in the n_sources_per_shot dimension",
        ),  # shape mismatch 2
        (
            (1, 1, 2),
            (1, 1, 3),
            (1, 1, 2),
            "Source locations must have 2 dimensional coordinates, but found one with 3.",
        ),  # shape mismatch 3
        (
            (2, 1, 1),
            (1, 1, 3),
            (1, 1, 2),
            "Expected source amplitudes to have size 1 in the batch dimension",
        ),  # shape mismatch 4
    ],
)
def test_input_shape_mismatch(
    prop,
    source_amplitudes_shape,
    source_locations_shape,
    receiver_locations_shape,
    expected_error_match,
):
    source_amplitudes = torch.zeros(source_amplitudes_shape)
    source_locations = torch.zeros(source_locations_shape, dtype=torch.long)
    receiver_locations = torch.zeros(receiver_locations_shape, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError, match=re.escape(expected_error_match)):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_source_outside_model(prop):
    """Check error when source location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = -1.0
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError, match="Locations must be >= 0."):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


def test_receiver_outside_model(prop):
    """Check error when receiver location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 1] = NY
    dt = 0.004
    with pytest.raises(RuntimeError, match="Locations must be within model."):
        prop(dt, source_amplitudes, source_locations, receiver_locations)


@pytest.mark.parametrize(
    "wavefield_shape, expected_error_match",
    [
        (
            (3, NZ + 2 * 20, NY + 2 * 20),
            "Inconsistent batch size",
        ),  # batch size too big
        (
            (1, NZ + 2 * 20, NY + 2 * 20),
            "Inconsistent batch size",
        ),  # batch size too small
        (
            (2, NZ + 2 * 20 + 1, NY + 2 * 20),
            "Survey extents are larger than the model.",
        ),  # too big in Z
        (
            (2, NZ + 2 * 20, NY + 2 * 20 + 1),
            "Survey extents are larger than the model.",
        ),  # too big in Y
    ],
)
def test_wavefield_shape_mismatch(prop, wavefield_shape, expected_error_match):
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(wavefield_shape)
    with pytest.raises(RuntimeError, match=re.escape(expected_error_match)):
        prop(
            dt,
            source_amplitudes,
            source_locations,
            receiver_locations,
            wavefield_0=wavefield_0,
        )


def test_pml_width_list_too_long(prop):
    """Check error when pml_width list is too long."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(
        RuntimeError, match="Expected pml_width to be of length 1 or 4, got 5."
    ):
        prop(
            dt,
            source_amplitudes,
            source_locations,
            receiver_locations,
            pml_width=[1, 20, 20, 20, 20],
        )

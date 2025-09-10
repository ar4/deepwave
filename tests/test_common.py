"""Tests for deepwave.common."""

import re

import pytest
import torch

from deepwave.common import (
    cfl_condition_n,
    check_points_per_wavelength,
    check_source_amplitudes_locations_match,
    cosine_taper_end,
    downsample,
    extract_survey,
    get_n_batch,
    set_accuracy,
    set_freq_taper_frac,
    set_grid_spacing,
    set_max_vel,
    set_model_gradient_sampling_interval,
    set_nt,
    set_pml_freq,
    set_pml_width,
    set_source_amplitudes,
    set_time_pad_frac,
    upsample,
    zero_last_element_of_final_dimension,
)


# Tests for get_n_batch
def test_get_n_batch_source_locations() -> None:
    """Test get_n_batch with source_locations provided."""
    source_locations = [torch.zeros(5, 1, 2), None]
    wavefields = [None, None]
    assert get_n_batch(source_locations, wavefields) == 5


def test_get_n_batch_wavefields() -> None:
    """Test get_n_batch with wavefields provided."""
    source_locations = [None, None]
    wavefields = [torch.zeros(3, 10, 10), None]
    assert get_n_batch(source_locations, wavefields) == 3


def test_get_n_batch_multiple_sources_and_wavefields() -> None:
    """Test get_n_batch with both source_locations and wavefields provided."""
    source_locations = [None, torch.zeros(7, 1, 2)]
    wavefields = [torch.zeros(7, 10, 10), None]
    assert get_n_batch(source_locations, wavefields) == 7


def test_get_n_batch_all_none() -> None:
    """Test get_n_batch when all inputs are None."""
    source_locations = [None, None]
    wavefields = [None, None]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "At least one input source_locations or wavefield must be non-None.",
        ),
    ):
        get_n_batch(source_locations, wavefields)


def test_get_n_batch_inconsistent_batch_size() -> None:
    """Test get_n_batch with inconsistent batch sizes (should pick first)."""
    source_locations = [torch.zeros(5, 1, 2), None]
    wavefields = [torch.zeros(3, 10, 10), None]
    # The function takes the first non-None tensor's batch size. This test
    # ensures it picks the first one.
    assert get_n_batch(source_locations, wavefields) == 5


def test_get_n_batch_empty_tensor_list() -> None:
    """Test get_n_batch with empty tensor lists."""
    source_locations = []
    wavefields = []
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "At least one input source_locations or wavefield must be non-None.",
        ),
    ):
        get_n_batch(source_locations, wavefields)


def test_get_n_batch_zero_batch_size() -> None:
    """Test get_n_batch with zero batch size in source_locations."""
    source_locations = [torch.zeros(0, 1, 2), None]
    wavefields = [None, None]
    assert get_n_batch(source_locations, wavefields) == 0


def test_get_n_batch_zero_batch_size_wavefields() -> None:
    """Test get_n_batch with zero batch size in wavefields."""
    source_locations = [None, None]
    wavefields = [torch.zeros(0, 10, 10), None]
    assert get_n_batch(source_locations, wavefields) == 0


# Tests for set_grid_spacing
def test_set_grid_spacing_single_value() -> None:
    """Test set_grid_spacing with a single float value."""
    assert set_grid_spacing(10.0, 2) == [10.0, 10.0]
    assert set_grid_spacing(5, 3) == [5.0, 5.0, 5.0]


def test_set_grid_spacing_list_correct_length() -> None:
    """Test set_grid_spacing with a list of correct length."""
    assert set_grid_spacing([10.0, 11.0], 2) == [10.0, 11.0]
    assert set_grid_spacing([5, 6, 7], 3) == [5.0, 6.0, 7.0]


def test_set_grid_spacing_tensor_scalar() -> None:
    """Test set_grid_spacing with a scalar tensor."""
    assert set_grid_spacing(torch.tensor(10.0), 2) == [10.0, 10.0]


def test_set_grid_spacing_tensor_1d_correct_length() -> None:
    """Test set_grid_spacing with a 1D tensor of correct length."""
    assert set_grid_spacing(torch.tensor([10.0, 11.0]), 2) == [10.0, 11.0]


def test_set_grid_spacing_list_incorrect_length() -> None:
    """Test set_grid_spacing with a list of incorrect length."""
    with pytest.raises(
        RuntimeError,
        match=re.escape("grid_spacing must have 1 or 2 elements, got 3."),
    ):
        set_grid_spacing([10.0, 11.0, 12.0], 2)
    with pytest.raises(
        RuntimeError,
        match=re.escape("grid_spacing must have 1 or 3 elements, got 2."),
    ):
        set_grid_spacing([5, 6], 3)


def test_set_grid_spacing_invalid_type() -> None:
    """Test set_grid_spacing with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("grid_spacing must be a float or sequence of floats."),
    ):
        set_grid_spacing("invalid", 2)


def test_set_grid_spacing_negative_value() -> None:
    """Test set_grid_spacing with a negative value."""
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        set_grid_spacing(-10.0, 2)


def test_set_grid_spacing_zero_value() -> None:
    """Test set_grid_spacing with a zero value."""
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        set_grid_spacing(0.0, 2)


def test_set_grid_spacing_list_negative_element() -> None:
    """Test set_grid_spacing with a list containing a negative element."""
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        set_grid_spacing([10.0, -1.0], 2)


def test_set_grid_spacing_list_zero_element() -> None:
    """Test set_grid_spacing with a list containing a zero element."""
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        set_grid_spacing([10.0, 0.0], 2)


# Tests for set_accuracy
def test_set_accuracy_valid_values() -> None:
    """Test set_accuracy with valid values."""
    assert set_accuracy(2) == 2
    assert set_accuracy(4) == 4
    assert set_accuracy(6) == 6
    assert set_accuracy(8) == 8


def test_set_accuracy_invalid_value() -> None:
    """Test set_accuracy with invalid values."""
    with pytest.raises(
        ValueError,
        match=re.escape("accuracy must be 2, 4, 6, or 8, got 3"),
    ):
        set_accuracy(3)
    with pytest.raises(
        ValueError,
        match=re.escape("accuracy must be 2, 4, 6, or 8, got 0"),
    ):
        set_accuracy(0)
    with pytest.raises(
        ValueError,
        match=re.escape("accuracy must be 2, 4, 6, or 8, got 10"),
    ):
        set_accuracy(10)


def test_set_accuracy_invalid_type() -> None:
    """Test set_accuracy with invalid types."""
    with pytest.raises(TypeError, match=re.escape("accuracy must be an int.")):
        set_accuracy(4.0)
    with pytest.raises(TypeError, match=re.escape("accuracy must be an int.")):
        set_accuracy("4")


# Tests for set_pml_width
def test_set_pml_width_single_value() -> None:
    """Test set_pml_width with a single integer value."""
    assert set_pml_width(20, 2) == [20, 20, 20, 20]
    assert set_pml_width(10, 3) == [10, 10, 10, 10, 10, 10]


def test_set_pml_width_list_correct_length() -> None:
    """Test set_pml_width with a list of correct length."""
    assert set_pml_width([10, 11, 12, 13], 2) == [10, 11, 12, 13]
    assert set_pml_width([1, 2, 3, 4, 5, 6], 3) == [1, 2, 3, 4, 5, 6]


def test_set_pml_width_tensor_scalar() -> None:
    """Test set_pml_width with a scalar tensor."""
    assert set_pml_width(torch.tensor(20), 2) == [20, 20, 20, 20]


def test_set_pml_width_tensor_1d_correct_length() -> None:
    """Test set_pml_width with a 1D tensor of correct length."""
    assert set_pml_width(torch.tensor([10, 11, 12, 13]), 2) == [10, 11, 12, 13]


def test_set_pml_width_list_incorrect_length() -> None:
    """Test set_pml_width with a list of incorrect length."""
    with pytest.raises(
        RuntimeError,
        match=re.escape("Expected pml_width to be of length 1 or 4, got 3."),
    ):
        set_pml_width([10, 11, 12], 2)
    with pytest.raises(
        RuntimeError,
        match=re.escape("Expected pml_width to be of length 1 or 6, got 5."),
    ):
        set_pml_width([1, 2, 3, 4, 5], 3)


def test_set_pml_width_invalid_type() -> None:
    """Test set_pml_width with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_width must be an int or sequence of ints."),
    ):
        set_pml_width("invalid", 2)


def test_set_pml_width_list_invalid_element_type() -> None:
    """Test set_pml_width with a list containing an invalid element type."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_width must be an int or sequence of ints."),
    ):
        set_pml_width([1, "invalid"], 2)


def test_set_pml_width_negative_value() -> None:
    """Test set_pml_width with a negative value."""
    with pytest.raises(ValueError, match=re.escape("pml_width must be non-negative.")):
        set_pml_width(-10, 2)


def test_set_pml_width_zero_value() -> None:
    """Test set_pml_width with a zero value."""
    assert set_pml_width(0, 2) == [0, 0, 0, 0]


def test_set_pml_width_list_negative_element() -> None:
    """Test set_pml_width with a list containing a negative element."""
    with pytest.raises(ValueError, match=re.escape("pml_width must be non-negative.")):
        set_pml_width([10, -1, 12, 13], 2)


def test_set_pml_width_list_zero_element() -> None:
    """Test set_pml_width with a list containing a zero element."""
    assert set_pml_width([10, 0, 12, 13], 2) == [10, 0, 12, 13]


# Tests for set_pml_freq
def test_set_pml_freq_none() -> None:
    """Test set_pml_freq with None value (should default and warn)."""
    # Default value is 25.0
    with pytest.warns(
        UserWarning,
        match=re.escape("pml_freq was not set, so defaulting to 25.0."),
    ):
        assert set_pml_freq(None, 0.004) == 25.0


def test_set_pml_freq_valid_value() -> None:
    """Test set_pml_freq with valid values."""
    assert set_pml_freq(30.0, 0.004) == 30.0
    assert set_pml_freq(10, 0.004) == 10.0


def test_set_pml_freq_negative_value() -> None:
    """Test set_pml_freq with a negative value."""
    with pytest.raises(ValueError, match=re.escape("pml_freq must be non-negative.")):
        set_pml_freq(-10.0, 0.004)


def test_set_pml_freq_above_nyquist() -> None:
    """Test set_pml_freq with a value above Nyquist frequency (should warn)."""
    # Nyquist for dt=0.004 is 0.5 / 0.004 = 125.0
    with pytest.warns(
        UserWarning,
        match=re.escape("pml_freq 150.0 is greater than the Nyquist frequency 125.0."),
    ):
        assert set_pml_freq(150.0, 0.004) == 150.0


def test_set_pml_freq_invalid_type() -> None:
    """Test set_pml_freq with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_freq must be None or convertible to a float."),
    ):
        set_pml_freq("invalid", 0.004)


def test_set_pml_freq_zero_dt() -> None:
    """Test set_pml_freq with zero dt (should raise error)."""
    with pytest.raises(
        ValueError,
        match=re.escape("dt is too small."),
    ):
        set_pml_freq(30.0, 0.0)


# Tests for set_max_vel
def test_set_max_vel_none() -> None:
    """Test set_max_vel with None value (should default to actual_max_vel)."""
    assert set_max_vel(None, 1500.0) == 1500.0


def test_set_max_vel_valid_value() -> None:
    """Test set_max_vel with valid values."""
    assert set_max_vel(2000.0, 1500.0) == 2000.0
    assert set_max_vel(1000, 1500.0) == 1000.0


def test_set_max_vel_less_than_actual() -> None:
    """Test set_max_vel when it's less than actual_max_vel (should warn)."""
    with pytest.warns(
        UserWarning,
        match=re.escape("max_vel is less than the actual maximum velocity."),
    ):
        assert set_max_vel(1000.0, 1500.0) == 1000.0


def test_set_max_vel_negative_value() -> None:
    """Test set_max_vel with a negative value (should return absolute value)."""
    assert set_max_vel(-2000.0, 1500.0) == 2000.0


def test_set_max_vel_invalid_type() -> None:
    """Test set_max_vel with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("max_vel must be None or convertible to a float."),
    ):
        set_max_vel("invalid", 1500.0)


def test_set_max_vel_zero_actual_max_vel() -> None:
    """Test set_max_vel with zero actual_max_vel (should raise error)."""
    with pytest.raises(
        ValueError,
        match=re.escape("maximum absolute velocity must be greater than zero."),
    ):
        set_max_vel(0.0, 0.0)


# Tests for set_nt
def test_set_nt_none_with_source_amplitudes() -> None:
    """Test set_nt when nt is None and source_amplitudes are provided."""
    source_amplitudes = [torch.zeros(1, 1, 100)]
    step_ratio = 1
    assert set_nt(None, source_amplitudes, step_ratio) == 100


def test_set_nt_none_without_source_amplitudes() -> None:
    """Test set_nt when nt is None and no source_amplitudes are provided."""
    source_amplitudes = [None]
    step_ratio = 1
    with pytest.raises(
        RuntimeError,
        match=re.escape("nt or source amplitudes must be specified"),
    ):
        set_nt(None, source_amplitudes, step_ratio)


def test_set_nt_with_nt_only() -> None:
    """Test set_nt when only nt is specified."""
    source_amplitudes = [None]
    step_ratio = 2
    assert set_nt(50, source_amplitudes, step_ratio) == 100


def test_set_nt_with_both_consistent() -> None:
    """Test set_nt when both nt and source_amplitudes are consistent."""
    source_amplitudes = [torch.zeros(1, 1, 100)]
    step_ratio = 1
    assert set_nt(100, source_amplitudes, step_ratio) == 100


def test_set_nt_with_both_inconsistent() -> None:
    """Test set_nt when both nt and source_amplitudes are inconsistent."""
    source_amplitudes = [torch.zeros(1, 1, 100)]
    step_ratio = 1
    with pytest.raises(
        RuntimeError,
        match=re.escape("Only one of nt or source amplitudes should be specified"),
    ):
        set_nt(50, source_amplitudes, step_ratio)


def test_set_nt_negative_nt() -> None:
    """Test set_nt with a negative nt value."""
    source_amplitudes = [None]
    step_ratio = 1
    with pytest.raises(RuntimeError, match=re.escape("nt must be >= 0")):
        set_nt(-10, source_amplitudes, step_ratio)


def test_set_nt_invalid_type() -> None:
    """Test set_nt with an invalid type for nt."""
    source_amplitudes = [None]
    step_ratio = 1
    with pytest.raises(TypeError, match=re.escape("nt must be an int or None.")):
        set_nt(10.0, source_amplitudes, step_ratio)
    with pytest.raises(TypeError, match=re.escape("nt must be an int or None.")):
        set_nt("invalid", source_amplitudes, step_ratio)


def test_set_nt_zero_step_ratio() -> None:
    """Test set_nt with a zero step_ratio."""
    source_amplitudes = [None]
    step_ratio = 0
    with pytest.raises(ValueError, match=re.escape("step_ratio must be >= 1")):
        set_nt(100, source_amplitudes, step_ratio)


def test_set_nt_negative_step_ratio() -> None:
    """Test set_nt with a negative step_ratio."""
    source_amplitudes = [None]
    step_ratio = -1
    with pytest.raises(ValueError, match=re.escape("step_ratio must be >= 1")):
        set_nt(100, source_amplitudes, step_ratio)


# Tests for set_model_gradient_sampling_interval
def test_set_model_gradient_sampling_interval_valid_value() -> None:
    """Test set_model_gradient_sampling_interval with valid values."""
    assert set_model_gradient_sampling_interval(1) == 1
    assert set_model_gradient_sampling_interval(10) == 10


def test_set_model_gradient_sampling_interval_negative_value() -> None:
    """Test set_model_gradient_sampling_interval with a negative value."""
    with pytest.raises(
        ValueError,
        match=re.escape("model_gradient_sampling_interval must be >= 0"),
    ):
        set_model_gradient_sampling_interval(-1)


def test_set_model_gradient_sampling_interval_invalid_type() -> None:
    """Test set_model_gradient_sampling_interval with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("model_gradient_sampling_interval must be an int."),
    ):
        set_model_gradient_sampling_interval(1.0)
    with pytest.raises(
        TypeError,
        match=re.escape("model_gradient_sampling_interval must be an int."),
    ):
        set_model_gradient_sampling_interval("invalid")


def test_set_model_gradient_sampling_interval_zero_value() -> None:
    """Test set_model_gradient_sampling_interval with a zero value."""
    assert set_model_gradient_sampling_interval(0) == 0


# Tests for set_freq_taper_frac
def test_set_freq_taper_frac_valid_value() -> None:
    """Test set_freq_taper_frac with valid values."""
    assert set_freq_taper_frac(0.0) == 0.0
    assert set_freq_taper_frac(0.5) == 0.5
    assert set_freq_taper_frac(1.0) == 1.0


def test_set_freq_taper_frac_out_of_range() -> None:
    """Test set_freq_taper_frac with values out of range."""
    with pytest.raises(
        ValueError,
        match=re.escape("freq_taper_frac must be in [0, 1], got -0.1"),
    ):
        set_freq_taper_frac(-0.1)
    with pytest.raises(
        ValueError,
        match=re.escape("freq_taper_frac must be in [0, 1], got 1.1"),
    ):
        set_freq_taper_frac(1.1)


def test_set_freq_taper_frac_invalid_type() -> None:
    """Test set_freq_taper_frac with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("freq_taper_frac must be convertible to a float."),
    ):
        set_freq_taper_frac("invalid")


# Tests for set_time_pad_frac
def test_set_time_pad_frac_valid_value() -> None:
    """Test set_time_pad_frac with valid values."""
    assert set_time_pad_frac(0.0) == 0.0
    assert set_time_pad_frac(0.5) == 0.5
    assert set_time_pad_frac(1.0) == 1.0


def test_set_time_pad_frac_out_of_range() -> None:
    """Test set_time_pad_frac with values out of range."""
    with pytest.raises(
        ValueError,
        match=re.escape("time_pad_frac must be in [0, 1], got -0.1"),
    ):
        set_time_pad_frac(-0.1)
    with pytest.raises(
        ValueError,
        match=re.escape("time_pad_frac must be in [0, 1], got 1.1"),
    ):
        set_time_pad_frac(1.1)


def test_set_time_pad_frac_invalid_type() -> None:
    """Test set_time_pad_frac with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("time_pad_frac must be convertible to a float."),
    ):
        set_time_pad_frac("invalid")


# Tests for check_source_amplitudes_locations_match
def test_check_source_amplitudes_locations_match_valid() -> None:
    """Test check_source_amplitudes_locations_match with valid inputs."""
    source_amplitudes = [torch.zeros(1, 1, 10), torch.zeros(1, 2, 10)]
    source_locations = [torch.zeros(1, 1, 2), torch.zeros(1, 2, 2)]
    check_source_amplitudes_locations_match(source_amplitudes, source_locations)


def test_check_source_amplitudes_locations_match_length_mismatch() -> None:
    """Test check_source_amplitudes_locations_match with length mismatch."""
    source_amplitudes = [torch.zeros(1, 1, 10)]
    source_locations = [torch.zeros(1, 1, 2), torch.zeros(1, 2, 2)]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The same number of source_amplitudes (1) and source_locations (2) "
            "must be provided.",
        ),
    ):
        check_source_amplitudes_locations_match(source_amplitudes, source_locations)


def test_check_source_amplitudes_locations_match_none_mismatch() -> None:
    """Test check_source_amplitudes_locations_match with None mismatch."""
    source_amplitudes = [torch.zeros(1, 1, 10), None]
    source_locations = [torch.zeros(1, 1, 2), torch.zeros(1, 2, 2)]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Each pair of source locations and amplitudes must both be None or "
            "both be non-None.",
        ),
    ):
        check_source_amplitudes_locations_match(source_amplitudes, source_locations)


def test_check_source_amplitudes_locations_match_n_sources_mismatch() -> None:
    """Test check_source_amplitudes_locations_match with n_sources mismatch."""
    source_amplitudes = [torch.zeros(1, 1, 10)]
    source_locations = [torch.zeros(1, 2, 2)]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Expected source amplitudes and locations to be the same size in the "
            "n_sources_per_shot dimension, got 1 and 2.",
        ),
    ):
        check_source_amplitudes_locations_match(source_amplitudes, source_locations)


def test_check_source_amplitudes_locations_match_empty_lists() -> None:
    """Test check_source_amplitudes_locations_match with empty lists."""
    source_amplitudes = []
    source_locations = []
    check_source_amplitudes_locations_match(source_amplitudes, source_locations)


def test_check_source_amplitudes_locations_match_zero_n_sources_per_shot() -> None:
    """Test check_source_amplitudes_locations_match with zero n_sources_per_shot."""
    source_amplitudes = [torch.zeros(1, 0, 10)]
    source_locations = [torch.zeros(1, 0, 2)]
    check_source_amplitudes_locations_match(source_amplitudes, source_locations)


# Tests for set_source_amplitudes
def test_set_source_amplitudes_none() -> None:
    """Test set_source_amplitudes when source_amplitudes is None."""
    source_amplitudes = [None]
    n_batch = 2
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    result = set_source_amplitudes(
        source_amplitudes,
        n_batch,
        nt,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    )
    assert len(result) == 1
    assert result[0].shape == (nt, n_batch, 0)
    assert result[0].device == device
    assert result[0].dtype == dtype


def test_set_source_amplitudes_valid() -> None:
    """Test set_source_amplitudes with valid inputs."""
    source_amplitudes = [torch.randn(2, 3, 50)]
    n_batch = 2
    nt = 100
    step_ratio = 2
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    result = set_source_amplitudes(
        source_amplitudes,
        n_batch,
        nt,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    )
    assert len(result) == 1
    assert result[0].shape == (nt, n_batch, 3)
    assert result[0].device == device
    assert result[0].dtype == dtype


def test_set_source_amplitudes_invalid_ndim() -> None:
    """Test set_source_amplitudes with invalid number of dimensions."""
    source_amplitudes = [torch.randn(2, 50)]  # Should be 3D
    n_batch = 2
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "source amplitudes Tensors should have 3 dimensions, but found one with 2.",
        ),
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


def test_set_source_amplitudes_inconsistent_device() -> None:
    """Test set_source_amplitudes with inconsistent device."""
    source_amplitudes = [
        torch.randn(2, 3, 100, device="cuda:0")
        if torch.cuda.is_available()
        else torch.randn(2, 3, 100),
    ]
    n_batch = 2
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Inconsistent device: Expected all Tensors be on device cpu, "
                "but found a source amplitudes torch.Tensor on device cuda:0.",
            ),
        ):
            set_source_amplitudes(
                source_amplitudes,
                n_batch,
                nt,
                step_ratio,
                freq_taper_frac,
                time_pad_frac,
                time_taper,
                device,
                dtype,
            )
    else:
        # If CUDA is not available, the tensor will be on CPU, so no error
        # should be raised.
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


def test_set_source_amplitudes_inconsistent_dtype() -> None:
    """Test set_source_amplitudes with inconsistent dtype."""
    source_amplitudes = [torch.randn(2, 3, 50, dtype=torch.float64)]
    n_batch = 2
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"Inconsistent dtype: Expected source amplitudes to have datatype "
            f"{dtype}, but found one with dtype {torch.float64}.",
        ),
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


def test_set_source_amplitudes_inconsistent_batch_size() -> None:
    """Test set_source_amplitudes with inconsistent batch size."""
    source_amplitudes = [torch.randn(3, 3, 50)]  # n_batch is 2, but tensor is 3
    n_batch = 2
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Expected source amplitudes to have size 2 in the batch dimension, "
            "but found one with size 3.",
        ),
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


def test_set_source_amplitudes_inconsistent_nt() -> None:
    """Test set_source_amplitudes with inconsistent number of time samples."""
    source_amplitudes = [
        torch.randn(2, 3, 40),
    ]  # nt is 100, step_ratio is 1, so expected time samples is 100
    n_batch = 2
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"Inconsistent number of time samples: Expected source amplitudes "
            f"to have {nt // step_ratio} time samples, but found one with 40.",
        ),
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


def test_set_source_amplitudes_upsampling() -> None:
    """Test set_source_amplitudes with upsampling."""
    source_amplitudes = [torch.randn(2, 3, 50)]
    n_batch = 2
    nt = 100
    step_ratio = 2
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    result = set_source_amplitudes(
        source_amplitudes,
        n_batch,
        nt,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    )
    assert result[0].shape == (nt, n_batch, 3)
    # Further checks for upsampling correctness would require comparing with a
    # known good upsampling, which is complex. For now, shape check is
    # sufficient as upsample function has its own tests.


def test_set_source_amplitudes_zero_n_batch() -> None:
    """Test set_source_amplitudes with zero n_batch."""
    source_amplitudes = [torch.randn(2, 3, 50)]
    n_batch = 0
    nt = 100
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Expected source amplitudes to have size 0 in the batch dimension, "
            "but found one with size 2.",
        ),
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


def test_set_source_amplitudes_zero_nt() -> None:
    """Test set_source_amplitudes with zero nt."""
    source_amplitudes = [torch.randn(2, 3, 50)]
    n_batch = 2
    nt = 0
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Inconsistent number of time samples: Expected source amplitudes "
            "to have 0 time samples, but found one with 50.",
        ),
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


# Tests for check_points_per_wavelength
def test_check_points_per_wavelength_valid() -> None:
    """Test check_points_per_wavelength with valid inputs."""
    min_nonzero_vel = 1500.0
    pml_freq = 25.0
    grid_spacing = [5.0, 5.0]
    # Should not warn
    check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_warns() -> None:
    """Test that a warning is issued when points per wavelength is too low.

    min_wavelength = 100 / 25 = 4
    max_spacing = 5
    cells_per_wavelength = 4 / 5 = 0.8 (less than 6)
    """
    min_nonzero_vel = 100.0
    pml_freq = 25.0
    grid_spacing = [5.0, 5.0]
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "At least six grid cells per wavelength is recommended, but at a "
            "frequency of 25.0, a minimum non-zero velocity of 100.0, and a "
            "grid cell spacing of 5.0, there are only 0.80.",
        ),
    ):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_zero_pml_freq() -> None:
    """Test check_points_per_wavelength with zero pml_freq."""
    min_nonzero_vel = 1500.0
    pml_freq = 0.0  # This will cause division by zero if not handled
    grid_spacing = [5.0, 5.0]
    # Should not warn, as min_wavelength will be inf, so cells_per_wavelength
    # will be inf
    check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_zero_min_nonzero_vel() -> None:
    """Test check_points_per_wavelength with zero min_nonzero_vel."""
    min_nonzero_vel = 0.0
    pml_freq = 25.0
    grid_spacing = [5.0, 5.0]
    # Should not warn, as min_wavelength will be 0, so cells_per_wavelength
    # will be 0
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "At least six grid cells per wavelength is recommended, but at a "
            "frequency of 25.0, a minimum non-zero velocity of 0.0, and a "
            "grid cell spacing of 5.0, there are only 0.00.",
        ),
    ):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_negative_min_nonzero_vel() -> None:
    """Test check_points_per_wavelength with negative min_nonzero_vel."""
    min_nonzero_vel = -100.0
    pml_freq = 25.0
    grid_spacing = [5.0, 5.0]
    with pytest.raises(
        ValueError,
        match=re.escape("min_nonzero_vel must be non-negative."),
    ):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_negative_pml_freq() -> None:
    """Test check_points_per_wavelength with negative pml_freq."""
    min_nonzero_vel = 1500.0
    pml_freq = -25.0
    grid_spacing = [5.0, 5.0]
    with pytest.raises(ValueError, match=re.escape("pml_freq must be non-negative.")):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_zero_grid_spacing_element() -> None:
    """Test check_points_per_wavelength with a zero grid_spacing element."""
    min_nonzero_vel = 1500.0
    pml_freq = 25.0
    grid_spacing = [5.0, 0.0]
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_negative_grid_spacing_element() -> None:
    """Test check_points_per_wavelength with a negative grid_spacing element."""
    min_nonzero_vel = 1500.0
    pml_freq = 25.0
    grid_spacing = [5.0, -1.0]
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


# Tests for cosine_taper_end
def test_cosine_taper_end_basic() -> None:
    """Test cosine_taper_end with basic functionality."""
    signal = torch.ones(10)
    n_taper = 5
    expected = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.9045, 0.6545, 0.3455, 0.0955, 0.0],
    )
    result = cosine_taper_end(signal, n_taper)
    assert torch.allclose(result, expected, atol=1e-4)


def test_cosine_taper_end_full_taper() -> None:
    """Test cosine_taper_end with full taper (n_taper equals signal length)."""
    signal = torch.ones(5)
    n_taper = 5
    expected = torch.tensor([0.9045, 0.6545, 0.3455, 0.0955, 0.0])
    result = cosine_taper_end(signal, n_taper)
    assert torch.allclose(result, expected, atol=1e-4)


def test_cosine_taper_end_no_taper() -> None:
    """Test cosine_taper_end with no taper (n_taper is zero)."""
    signal = torch.ones(10)
    n_taper = 0
    expected = torch.ones(10)
    result = cosine_taper_end(signal, n_taper)
    assert torch.allclose(result, expected)


def test_cosine_taper_end_n_taper_greater_than_signal_length() -> None:
    """Test cosine_taper_end when n_taper is greater than signal length."""
    signal = torch.ones(5)
    n_taper = 10
    expected = torch.tensor([0.9045, 0.6545, 0.3455, 0.0955, 0.0])
    result = cosine_taper_end(signal, n_taper)
    assert torch.allclose(result, expected, atol=1e-4)


def test_cosine_taper_end_zero_signal() -> None:
    """Test cosine_taper_end with a zero signal."""
    signal = torch.zeros(10)
    n_taper = 5
    expected = torch.zeros(10)
    result = cosine_taper_end(signal, n_taper)
    assert torch.allclose(result, expected)


def test_cosine_taper_end_empty_signal() -> None:
    """Test cosine_taper_end with an empty signal."""
    signal = torch.tensor([])
    n_taper = 5
    expected = torch.tensor([])
    result = cosine_taper_end(signal, n_taper)
    assert torch.allclose(result, expected)


def test_cosine_taper_end_negative_n_taper() -> None:
    """Test cosine_taper_end with a negative n_taper."""
    signal = torch.ones(10)
    n_taper = -5
    with pytest.raises(ValueError, match=re.escape("n_taper must be non-negative.")):
        cosine_taper_end(signal, n_taper)


# Tests for zero_last_element_of_final_dimension
def test_zero_last_element_of_final_dimension_basic() -> None:
    """Test zero_last_element_of_final_dimension with a basic 1D tensor."""
    signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0])
    result = zero_last_element_of_final_dimension(signal)
    assert torch.allclose(result, expected)


def test_zero_last_element_of_final_dimension_multi_dim() -> None:
    """Test zero_last_element_of_final_dimension with a multi-dimensional tensor."""
    signal = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])
    result = zero_last_element_of_final_dimension(signal)
    assert torch.allclose(result, expected)


def test_zero_last_element_of_final_dimension_single_element() -> None:
    """Test zero_last_element_of_final_dimension with a single-element tensor."""
    signal = torch.tensor([1.0])
    expected = torch.tensor([0.0])
    result = zero_last_element_of_final_dimension(signal)
    assert torch.allclose(result, expected)


def test_zero_last_element_of_final_dimension_empty_tensor() -> None:
    """Test zero_last_element_of_final_dimension with an empty tensor."""
    signal = torch.tensor([])
    expected = torch.tensor([])
    result = zero_last_element_of_final_dimension(signal)
    assert torch.allclose(result, expected)


# Tests for cfl_condition_n
def test_cfl_condition_n_basic() -> None:
    """Test cfl_condition_n with basic valid inputs."""
    grid_spacing = [5.0, 5.0]
    dt = 0.001
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 1
    assert inner_dt == pytest.approx(0.001)


def test_cfl_condition_larger_dt() -> None:
    """Test cfl_condition_n with a larger dt requiring upsampling."""
    grid_spacing = [5.0, 5.0]
    dt = 0.002
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 2
    assert inner_dt == pytest.approx(0.001)


def test_cfl_condition_larger_max_vel() -> None:
    """Test cfl_condition_n with a larger max_vel requiring more upsampling."""
    grid_spacing = [5.0, 5.0]
    dt = 0.002
    max_vel = 4000.0  # Higher velocity, so CFL will require smaller dt
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 4
    assert inner_dt == pytest.approx(0.0005)


def test_cfl_condition_n_zero_max_vel() -> None:
    """Test cfl_condition_n with zero max_vel (should raise error)."""
    grid_spacing = [5.0, 5.0]
    dt = 0.004
    max_vel = 0.0
    with pytest.raises(
        RuntimeError,
        match=re.escape("max_abs_vel must be greater than zero."),
    ):
        cfl_condition_n(grid_spacing, dt, max_vel)


def test_cfl_condition_n_negative_dt() -> None:
    """Test cfl_condition_n with a negative dt."""
    grid_spacing = [5.0, 5.0]
    dt = -0.001
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 1
    assert inner_dt == pytest.approx(-0.001)


def test_cfl_condition_n_different_grid_spacing() -> None:
    """Test cfl_condition_n with different grid spacings in dimensions."""
    grid_spacing = [5.0, 10.0]
    dt = 0.002
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 2
    assert inner_dt == pytest.approx(0.001)


def test_cfl_condition_n_invalid_type_grid_spacing() -> None:
    """Test cfl_condition_n with invalid type for grid_spacing."""
    with pytest.raises(
        TypeError,
        match=re.escape("grid_spacing must be a list of floats."),
    ):
        cfl_condition_n("invalid", 0.004, 1500.0)


def test_cfl_condition_n_invalid_type_dt() -> None:
    """Test cfl_condition_n with invalid type for dt."""
    with pytest.raises(TypeError, match=re.escape("dt must be a float.")):
        cfl_condition_n([5.0, 5.0], "invalid", 1500.0)


def test_cfl_condition_n_invalid_type_max_vel() -> None:
    """Test cfl_condition_n with invalid type for max_vel."""
    with pytest.raises(TypeError, match=re.escape("max_abs_vel must be a float.")):
        cfl_condition_n([5.0, 5.0], 0.004, "invalid")


def test_cfl_condition_n_negative_grid_spacing_element() -> None:
    """Test cfl_condition_n with a negative grid_spacing element."""
    grid_spacing = [5.0, -1.0]
    dt = 0.002
    max_vel = 1500.0
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        cfl_condition_n(grid_spacing, dt, max_vel)


def test_cfl_condition_n_zero_grid_spacing_element() -> None:
    """Test cfl_condition_n with a zero grid_spacing element."""
    grid_spacing = [5.0, 0.0]
    dt = 0.002
    max_vel = 1500.0
    with pytest.raises(
        ValueError,
        match=re.escape("grid_spacing elements must be positive."),
    ):
        cfl_condition_n(grid_spacing, dt, max_vel)


def test_cfl_condition_n_empty_grid_spacing() -> None:
    """Test cfl_condition_n with an empty grid_spacing list."""
    grid_spacing = []
    dt = 0.002
    max_vel = 1500.0
    with pytest.raises(ValueError, match=re.escape("grid_spacing must not be empty.")):
        cfl_condition_n(grid_spacing, dt, max_vel)


# Additional tests for set_grid_spacing
def test_set_grid_spacing_tuple_correct_length() -> None:
    """Test set_grid_spacing with a tuple of correct length."""
    assert set_grid_spacing((10.0, 11.0), 2) == [10.0, 11.0]
    assert set_grid_spacing((5, 6, 7), 3) == [5.0, 6.0, 7.0]


def test_set_grid_spacing_complex_value() -> None:
    """Test set_grid_spacing with a complex value."""
    with pytest.raises(TypeError):
        set_grid_spacing(10.0 + 1j, 2)


def test_set_grid_spacing_list_complex_element() -> None:
    """Test set_grid_spacing with a list containing a complex element."""
    with pytest.raises(TypeError):
        set_grid_spacing([10.0, 1j], 2)


# Additional tests for set_pml_width
def test_set_pml_width_tuple_correct_length() -> None:
    """Test set_pml_width with a tuple of correct length."""
    assert set_pml_width((10, 11, 12, 13), 2) == [10, 11, 12, 13]
    assert set_pml_width((1, 2, 3, 4, 5, 6), 3) == [1, 2, 3, 4, 5, 6]


def test_set_pml_width_float_value() -> None:
    """Test set_pml_width with a float value."""
    assert set_pml_width(20.0, 2) == [20, 20, 20, 20]
    with pytest.raises(
        TypeError,
        match=re.escape("pml_width must be an int or sequence of ints."),
    ):
        set_pml_width(20.5, 2)


def test_set_pml_width_list_float_element() -> None:
    """Test set_pml_width with a list containing a float element."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_width must be an int or sequence of ints."),
    ):
        set_pml_width([10, 11.5], 1)


def test_set_pml_width_complex_value() -> None:
    """Test set_pml_width with a complex value."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_width must be an int or sequence of ints."),
    ):
        set_pml_width(10 + 1j, 2)


def test_set_pml_width_list_complex_element() -> None:
    """Test set_pml_width with a list containing a complex element."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_width must be an int or sequence of ints."),
    ):
        set_pml_width([10, 1j], 2)


# Additional tests for set_pml_freq
def test_set_pml_freq_complex_value() -> None:
    """Test set_pml_freq with a complex value."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_freq must be None or convertible to a float."),
    ):
        set_pml_freq(25.0 + 1j, 0.004)


# Additional tests for set_max_vel
def test_set_max_vel_complex_value() -> None:
    """Test set_max_vel with a complex value."""
    with pytest.raises(
        TypeError,
        match=re.escape("max_vel must be None or convertible to a float."),
    ):
        set_max_vel(1500.0 + 1j, 1500.0)


def test_set_max_vel_complex_max_model_vel() -> None:
    """Test set_max_vel with a complex max_model_vel."""
    with pytest.raises(
        TypeError,
        match=re.escape("max_abs_model_vel must be convertible to a float."),
    ):
        set_max_vel(2000.0, 1500.0 + 1j)


# Additional tests for set_nt
def test_set_nt_complex_value() -> None:
    """Test set_nt with a complex value."""
    source_amplitudes = [None]
    step_ratio = 1
    with pytest.raises(
        TypeError,
        match=re.escape("nt must be an int or None."),
    ):
        set_nt(100 + 1j, source_amplitudes, step_ratio)


# Additional tests for set_model_gradient_sampling_interval
def test_set_model_gradient_sampling_interval_complex_value() -> None:
    """Test set_model_gradient_sampling_interval with a complex value."""
    with pytest.raises(
        TypeError,
        match=re.escape("model_gradient_sampling_interval must be an int."),
    ):
        set_model_gradient_sampling_interval(10 + 1j)


# Additional tests for set_freq_taper_frac
def test_set_freq_taper_frac_complex_value() -> None:
    """Test set_freq_taper_frac with a complex value."""
    with pytest.raises(
        TypeError,
        match=".*real number.*",
    ):
        set_freq_taper_frac(0.5 + 1j)


# Additional tests for set_time_pad_frac
def test_set_time_pad_frac_complex_value() -> None:
    """Test set_time_pad_frac with a complex value."""
    with pytest.raises(
        TypeError,
        match=".*real number.*",
    ):
        set_time_pad_frac(0.5 + 1j)


# Additional tests for check_source_amplitudes_locations_match
def test_check_source_amplitudes_locations_match_non_tensor_amplitudes() -> None:
    """Test check_source_amplitudes_locations_match with non-tensor amplitudes."""
    source_amplitudes = [[1, 2, 3]]
    source_locations = [torch.zeros(1, 1, 2)]
    with pytest.raises(
        TypeError, match=re.escape("source_amplitudes must be a torch.Tensor.")
    ):
        check_source_amplitudes_locations_match(source_amplitudes, source_locations)


def test_check_source_amplitudes_locations_match_non_tensor_locations() -> None:
    """Test check_source_amplitudes_locations_match with non-tensor locations."""
    source_amplitudes = [torch.zeros(1, 1, 10)]
    source_locations = [[1, 2, 3]]
    with pytest.raises(
        TypeError, match=re.escape("source_locations must be a torch.Tensor.")
    ):
        check_source_amplitudes_locations_match(source_amplitudes, source_locations)


# Additional tests for set_source_amplitudes
def test_set_source_amplitudes_non_tensor_input() -> None:
    """Test set_source_amplitudes with non-tensor input."""
    source_amplitudes = [[1, 2, 3]]
    n_batch = 1
    nt = 10
    step_ratio = 1
    freq_taper_frac = 0.0
    time_pad_frac = 0.0
    time_taper = False
    device = torch.device("cpu")
    dtype = torch.float32
    with pytest.raises(
        TypeError, match=re.escape("source_amplitudes must be a torch.Tensor.")
    ):
        set_source_amplitudes(
            source_amplitudes,
            n_batch,
            nt,
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            device,
            dtype,
        )


# Additional tests for check_points_per_wavelength
def test_check_points_per_wavelength_complex_min_nonzero_vel() -> None:
    """Test check_points_per_wavelength with complex min_nonzero_vel."""
    with pytest.raises(
        TypeError,
    ):
        check_points_per_wavelength(1500.0 + 1j, 25.0, [5.0, 5.0])


def test_check_points_per_wavelength_complex_pml_freq() -> None:
    """Test check_points_per_wavelength with complex pml_freq."""
    with pytest.raises(
        TypeError,
    ):
        check_points_per_wavelength(1500.0, 25.0 + 1j, [5.0, 5.0])


def test_check_points_per_wavelength_list_complex_grid_spacing_element() -> None:
    """Test check_points_per_wavelength with complex grid_spacing element in list."""
    with pytest.raises(
        TypeError,
    ):
        check_points_per_wavelength(1500.0, 25.0, [5.0, 1j])


# Additional tests for cosine_taper_end
def test_cosine_taper_end_non_tensor_signal() -> None:
    """Test cosine_taper_end with non-tensor signal."""
    signal = [1.0, 2.0, 3.0]
    n_taper = 1
    with pytest.raises(TypeError, match=re.escape("signal must be a torch.Tensor.")):
        cosine_taper_end(signal, n_taper)


def test_cosine_taper_end_float_n_taper() -> None:
    """Test cosine_taper_end with float n_taper."""
    signal = torch.ones(10)
    n_taper = 5.0
    with pytest.raises(TypeError, match=re.escape("n_taper must be an int.")):
        cosine_taper_end(signal, n_taper)


def test_cosine_taper_end_complex_n_taper() -> None:
    """Test cosine_taper_end with complex n_taper."""
    signal = torch.ones(10)
    n_taper = 5 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("n_taper must be an int."),
    ):
        cosine_taper_end(signal, n_taper)


# Additional tests for zero_last_element_of_final_dimension
def test_zero_last_element_of_final_dimension_non_tensor_signal() -> None:
    """Test zero_last_element_of_final_dimension with non-tensor signal."""
    signal = [1.0, 2.0, 3.0]
    with pytest.raises(TypeError, match=re.escape("signal must be a torch.Tensor.")):
        zero_last_element_of_final_dimension(signal)


# Additional tests for upsample
def test_upsample_non_tensor_signal() -> None:
    """Test upsample with non-tensor signal."""
    signal = [1.0, 2.0, 3.0]
    step_ratio = 2
    with pytest.raises(TypeError, match=re.escape("signal must be a torch.Tensor.")):
        upsample(signal, step_ratio)


def test_upsample_float_step_ratio() -> None:
    """Test upsample with float step_ratio."""
    signal = torch.ones(10)
    step_ratio = 2.0
    with pytest.raises(TypeError, match=re.escape("step_ratio must be an int.")):
        upsample(signal, step_ratio)


def test_upsample_complex_step_ratio() -> None:
    """Test upsample with complex step_ratio."""
    signal = torch.ones(10)
    step_ratio = 2 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("step_ratio must be an int."),
    ):
        upsample(signal, step_ratio)


def test_upsample_complex_freq_taper_frac() -> None:
    """Test upsample with complex freq_taper_frac."""
    signal = torch.ones(10)
    step_ratio = 2
    freq_taper_frac = 0.5 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("freq_taper_frac must be a float."),
    ):
        upsample(signal, step_ratio, freq_taper_frac=freq_taper_frac)


def test_upsample_complex_time_pad_frac() -> None:
    """Test upsample with complex time_pad_frac."""
    signal = torch.ones(10)
    step_ratio = 2
    time_pad_frac = 0.5 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("time_pad_frac must be a float."),
    ):
        upsample(signal, step_ratio, time_pad_frac=time_pad_frac)


def test_upsample_non_bool_time_taper() -> None:
    """Test upsample with non-boolean time_taper."""
    signal = torch.ones(10)
    step_ratio = 2
    time_taper = 1  # int instead of bool
    with pytest.raises(TypeError, match=re.escape("time_taper must be a bool.")):
        upsample(signal, step_ratio, time_taper=time_taper)


# Additional tests for downsample
def test_downsample_non_tensor_signal() -> None:
    """Test downsample with non-tensor signal."""
    signal = [1.0, 2.0, 3.0]
    step_ratio = 2
    with pytest.raises(TypeError, match=re.escape("signal must be a torch.Tensor.")):
        downsample(signal, step_ratio)


def test_downsample_float_step_ratio() -> None:
    """Test downsample with float step_ratio."""
    signal = torch.ones(10)
    step_ratio = 2.0
    with pytest.raises(TypeError, match=re.escape("step_ratio must be an int.")):
        downsample(signal, step_ratio)


def test_downsample_complex_step_ratio() -> None:
    """Test downsample with complex step_ratio."""
    signal = torch.ones(10)
    step_ratio = 2 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("step_ratio must be an int."),
    ):
        downsample(signal, step_ratio)


def test_downsample_complex_freq_taper_frac() -> None:
    """Test downsample with complex freq_taper_frac."""
    signal = torch.ones(10)
    step_ratio = 2
    freq_taper_frac = 0.5 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("freq_taper_frac must be a float."),
    ):
        downsample(signal, step_ratio, freq_taper_frac=freq_taper_frac)


def test_downsample_complex_time_pad_frac() -> None:
    """Test downsample with complex time_pad_frac."""
    signal = torch.ones(10)
    step_ratio = 2
    time_pad_frac = 0.5 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("time_pad_frac must be a float."),
    ):
        downsample(signal, step_ratio, time_pad_frac=time_pad_frac)


def test_downsample_non_bool_time_taper() -> None:
    """Test downsample with non-boolean time_taper."""
    signal = torch.ones(10)
    step_ratio = 2
    time_taper = 1  # int instead of bool
    with pytest.raises(TypeError, match=re.escape("time_taper must be a bool.")):
        downsample(signal, step_ratio, time_taper=time_taper)


def test_downsample_complex_shift() -> None:
    """Test downsample with complex shift."""
    signal = torch.ones(10)
    step_ratio = 1
    shift = 0.5 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("shift must be a float."),
    ):
        downsample(signal, step_ratio, shift=shift)


# Additional tests for cfl_condition_n
def test_cfl_condition_n_tuple_grid_spacing() -> None:
    """Test cfl_condition_n with tuple grid_spacing."""
    grid_spacing = (5.0, 5.0)
    dt = 0.001
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 1
    assert inner_dt == pytest.approx(0.001)


def test_cfl_condition_n_complex_dt() -> None:
    """Test cfl_condition_n with complex dt."""
    grid_spacing = [5.0, 5.0]
    dt = 0.001 + 1j
    max_vel = 1500.0
    with pytest.raises(
        TypeError,
        match=re.escape("dt must be a float."),
    ):
        cfl_condition_n(grid_spacing, dt, max_vel)


def test_cfl_condition_n_complex_max_abs_vel() -> None:
    """Test cfl_condition_n with complex max_abs_vel."""
    grid_spacing = [5.0, 5.0]
    dt = 0.001
    max_vel = 1500.0 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("max_abs_vel must be a float."),
    ):
        cfl_condition_n(grid_spacing, dt, max_vel)


def test_cfl_condition_n_list_complex_grid_spacing_element() -> None:
    """Test cfl_condition_n with a list containing a complex grid_spacing element."""
    grid_spacing = [5.0, 1j]
    dt = 0.001
    max_vel = 1500.0
    with pytest.raises(
        TypeError,
    ):
        cfl_condition_n(grid_spacing, dt, max_vel)


def test_locations_float_raises_error() -> None:
    """Test that extract_survey raises an error for float locations."""
    models = [torch.ones(10, 10)]
    source_locations = [torch.tensor([[[1.5, 2.0]]], dtype=torch.float32)]
    receiver_locations = [None]
    wavefields = [None, None, None, None, None, None]
    survey_pad = None
    origin = None
    fd_pad = [0, 0, 0, 0]
    pml_width = [0, 0, 0, 0]
    model_pad_modes = ["replicate"]
    n_batch = 1
    n_dims = 2
    device = torch.device("cpu")
    dtype = torch.float32

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Locations should be specified as integer numbers of cells. "
            "If you wish to have a source or receiver that is not centred on a "
            "cell, please consider using the Hick's method, which is "
            "implemented in deepwave.location_interpolation.",
        ),
    ):
        extract_survey(
            models,
            source_locations,
            receiver_locations,
            wavefields,
            survey_pad,
            origin,
            fd_pad,
            pml_width,
            model_pad_modes,
            n_batch,
            n_dims,
            device,
            dtype,
        )

"""Tests for deepwave._common."""

import importlib
import inspect
import re
import warnings

import pytest
import torch

import deepwave
import deepwave._common
from deepwave._common import (
    IGNORE_LOCATION,
    _critical_courant_number,
    cfl_condition_n,
    check_locations_are_within_model,
    check_locations_within_extents,
    check_model_dtype,
    check_models,
    check_points_per_wavelength,
    check_source_amplitudes_locations_match,
    cosine_taper_end,
    default_pml_freq,
    downsample,
    extract_survey,
    get_n_batch,
    infer_pml_freq,
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
    setup_pml,
    to_cells,
    upsample,
    velocity_extremes,
    zero_last_element_of_final_dimension,
)


def test_compiler_predicate_supports_torch_without_compiler_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyTorch 2.0 lacks ``torch.compiler`` but has Dynamo's predicate."""
    dynamo = importlib.import_module("torch._dynamo")
    monkeypatch.setattr(dynamo, "is_compiling", lambda: True)
    monkeypatch.delattr(torch, "compiler", raising=False)
    assert deepwave._common._torch_is_compiling() is True
    monkeypatch.setattr(torch, "compiler", object(), raising=False)
    assert deepwave._common._torch_is_compiling() is True


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
        match=re.escape("grid_spacing must have 2 elements, got 3."),
    ):
        set_grid_spacing([10.0, 11.0, 12.0], 2)
    with pytest.raises(
        RuntimeError,
        match=re.escape("grid_spacing must have 3 elements, got 2."),
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
    """With no sources to infer from, set_pml_freq falls back to 0.5/dt/5.

    At the classic seismic dt=0.004 that is exactly the 25.0 that Deepwave
    <= 0.0.27 always used, so this case is unchanged and must not warn.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert set_pml_freq(None, 0.004) == 25.0


def test_set_pml_freq_valid_value() -> None:
    """Test set_pml_freq with valid values."""
    assert set_pml_freq(30.0, 0.004) == 30.0
    assert set_pml_freq(10, 0.004) == 10.0


def test_set_pml_freq_negative_value() -> None:
    """Test set_pml_freq with a negative value."""
    with pytest.raises(ValueError, match=re.escape("pml_freq must be non-negative")):
        set_pml_freq(-10.0, 0.004)


def test_set_pml_freq_above_nyquist() -> None:
    """Test set_pml_freq with a value above Nyquist frequency (should warn)."""
    # Nyquist for dt=0.004 is 0.5 / 0.004 = 125.0
    with pytest.warns(
        UserWarning,
        match=re.escape("pml_freq 150.0 is greater than the Nyquist frequency 125.0"),
    ):
        assert set_pml_freq(150.0, 0.004) == 150.0


def test_set_pml_freq_invalid_type() -> None:
    """Test set_pml_freq with an invalid type."""
    with pytest.raises(
        TypeError,
        match=re.escape("pml_freq must be None or convertible to a float"),
    ):
        set_pml_freq("invalid", 0.004)


def test_set_pml_freq_zero_dt() -> None:
    """A zero time step does not impose a finite Nyquist bound."""
    assert set_pml_freq(30.0, 0.0) == 30.0


# Tests for set_max_vel
def test_set_max_vel_none() -> None:
    """Test set_max_vel with None value (should default to actual_max_vel)."""
    assert set_max_vel(None, 1500.0) == 1500.0


def test_set_max_vel_valid_value() -> None:
    """Test set_max_vel with valid values."""
    assert set_max_vel(2000.0, 1500.0) == 2000.0


def test_set_max_vel_less_than_actual() -> None:
    """An explicit speed below the model's conservative bound is refused."""
    with pytest.raises(deepwave.DeepwaveValueError, match="conservative maximum"):
        set_max_vel(1000.0, 1500.0)


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
    """Zero is an exact speed bound for a stationary model."""
    assert set_max_vel(0.0, 0.0) == 0.0


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
    """``nt`` wins over the source amplitudes' own length."""
    source_amplitudes = [torch.zeros(1, 1, 100)]
    step_ratio = 1
    assert set_nt(50, source_amplitudes, step_ratio) == 50
    assert set_nt(200, source_amplitudes, step_ratio) == 200


def test_set_nt_negative_nt() -> None:
    """Test set_nt with a negative nt value."""
    source_amplitudes = [None]
    step_ratio = 1
    with pytest.raises(ValueError, match=re.escape("nt must be >= 0")):
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
        match=re.escape("model_gradient_sampling_interval must be >= 1"),
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
    """Zero is rejected: it is used as a divisor of the stored step count."""
    with pytest.raises(
        ValueError,
        match=re.escape("model_gradient_sampling_interval must be >= 1"),
    ):
        set_model_gradient_sampling_interval(0)


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
            "the same number of source_amplitudes (1) and source_locations (2) "
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
            "source_locations[1] was provided but "
            "source_amplitudes[1] was not -- the two must be given "
            "together, or neither.",
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
            "source_amplitudes has shape (3, 3, 50), but Deepwave expects "
            "[shot, source, time] = [2, *, 100]",
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
            "source_amplitudes has shape (2, 3, 40), but Deepwave expects "
            f"[shot, source, time] = [2, *, {nt // step_ratio}]",
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
            "source_amplitudes has shape (2, 3, 50), but Deepwave expects "
            "[shot, source, time] = [0, *, 100]",
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
            "source_amplitudes has shape (2, 3, 50), but Deepwave expects "
            "[shot, source, time] = [2, *, 0]",
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

    The warning is guarded to fire at most once per process (so that an
    evolving FWI model does not spam it every iteration), so the guard is
    cleared first to make this test independent of test order.
    """
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    min_nonzero_vel = 100.0
    pml_freq = 25.0
    grid_spacing = [5.0, 5.0]
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "At least six grid cells per wavelength is recommended, but "
            "there are only 0.80 (frequency ~25, minimum non-zero velocity "
            "~100, grid spacing ~5). The velocity is the smallest non-zero "
            "value in the model, so a single slow cell, such as a thin fluid "
            "or near-vacuum layer, can set it. Use a finer grid_spacing; "
            "pass pml_freq=<your dominant frequency> if this is not the "
            "frequency you need to resolve; or raise accuracy, which lowers "
            "dispersion at the same sampling, though this six-cell figure is "
            "calibrated for the default accuracy=4 and does not move with "
            "it. deepwave.suggest_discretisation returns a grid spacing and "
            "time step that satisfy both this check and the CFL condition.",
        ),
    ):
        check_points_per_wavelength(min_nonzero_vel, pml_freq, grid_spacing)


def test_check_points_per_wavelength_warns_once_per_process() -> None:
    """The low-points-per-wavelength warning fires at most once per process.

    A second call with a different (still-too-low) velocity -- as happens
    across an FWI loop's evolving model -- must not re-fire the warning,
    since it is guarded by cause rather than by Python's default
    per-message-text de-duplication.
    """
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    grid_spacing = [5.0, 5.0]
    with pytest.warns(UserWarning, match=re.escape("grid cells per wavelength")):
        check_points_per_wavelength(100.0, 25.0, grid_spacing)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check_points_per_wavelength(101.0, 25.0, grid_spacing)
    assert len(record) == 0


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
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The test for the number of grid cells per wavelength is being skipped",
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
    with pytest.raises(ValueError, match=re.escape("pml_freq must be non-negative")):
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
def test_cfl_tiny_speed_does_not_create_artificial_substeps() -> None:
    """A nearly stationary model needs one step, independently of legacy eps."""
    for eps in (0.0, 1e-15, 1.0):
        assert cfl_condition_n([1.0], 1.0, 1e-20, eps=eps) == (1.0, 1)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-30, 1.0, 1e30, 1e200, 1e300])
@pytest.mark.parametrize("sign", [-1, 1])
def test_cfl_is_invariant_to_extreme_length_units(scale: float, sign: int) -> None:
    """Changing the common distance unit preserves the dimensionless bound."""
    dt = sign * 0.005
    actual = cfl_condition_n([5 * scale, 10 * scale, 15 * scale], dt, 1500 * scale)
    assert actual == (dt / 3, 3)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1.0, 1e200, 1e300])
def test_cfl_is_invariant_to_extreme_time_units(scale: float) -> None:
    """Reciprocal time/speed rescaling does not overflow the speed squared."""
    dt = 0.005 * scale
    assert cfl_condition_n([5.0, 10.0, 15.0], dt, 1500 / scale) == (dt / 3, 3)


@pytest.mark.parametrize(
    ("spacing", "dt", "speed", "courant", "ratio"),
    [
        ([1e-320], 0.1, 1e-320, 0.6, 1),
        ([5e-324] * 3, 1.0, 5e-324, 0.6, 3),
        ([5.0], 1e-300, 1e300, 0.6, 1),
        ([5.0], 1e300, 1e-300, 0.6, 1),
        ([2.0], 5e-324, 1.0, 5e-324, 1),
    ],
)
def test_cfl_extreme_finite_inputs(spacing, dt, speed, courant, ratio) -> None:
    """Representable CFL results survive tiny and huge intermediate scales."""
    assert cfl_condition_n(spacing, dt, speed, courant_number=courant) == (
        dt / ratio,
        ratio,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cfl_rejects_nonfinite_spacing_and_courant(value: float) -> None:
    """Undefined bounds fail explicitly before the exponent calculation."""
    with pytest.raises(ValueError, match="grid_spacing elements must be finite"):
        cfl_condition_n([value], 0.1, 1.0)
    with pytest.raises(ValueError, match="courant_number must be finite"):
        cfl_condition_n([1.0], 0.1, 1.0, courant_number=value)


def test_cfl_rejects_unrepresentable_time_discretisation() -> None:
    """An infinite ratio or a time step rounded to zero cannot drive a solver."""
    with pytest.raises(ValueError, match="CFL step ratio is too large"):
        cfl_condition_n([1.0], 1e308, 1e308)
    with pytest.raises(ValueError, match="CFL-stable time step is too small"):
        cfl_condition_n([1e-323], 5e-324, 4.0)


def test_cfl_condition_n_basic() -> None:
    """Test cfl_condition_n with basic valid inputs."""
    grid_spacing = [5.0, 5.0]
    dt = 0.001
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 1
    assert inner_dt == pytest.approx(0.001)


def test_cfl_condition_larger_dt() -> None:
    """A dt requiring upsampling below the step_ratio warning threshold (8) is silent.

    step_ratio=2 is below the threshold (raised from 2 to 8 so that, e.g.,
    the README's own Quick Example -- which lands at step_ratio=5 -- does
    not warn).
    """
    grid_spacing = [5.0, 5.0]
    dt = 0.002
    max_vel = 1500.0
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 2
    assert inner_dt == pytest.approx(0.001)
    assert len(record) == 0


def test_cfl_condition_n_step_ratio_mild_warning() -> None:
    """A step_ratio in [8, 20) warns once with the milder message (Item 5).

    The message names the (3-significant-figure-rounded) dt and the actual
    step_ratio value and approximate compute multiplier, and points at
    courant_number/max_vel. Guarded by the same once-per-cause helper as the
    extreme tier, so the guard is cleared first to make this test independent
    of test order.
    """
    deepwave._common._warned_once_causes.discard("cfl_step_ratio_mild")
    grid_spacing = [5.0, 5.0]
    dt = 0.011
    max_vel = 1500.0
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "dt~0.011 exceeds the stable internal time step; Deepwave is "
            "taking step_ratio=8 internal steps per output sample (~8x "
            "compute). Reduce dt, or see courant_number/max_vel.",
        ),
    ) as record:
        _, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 8
    assert len(record) == 1


def test_cfl_condition_n_step_ratio_extreme_keeps_stronger_wording() -> None:
    """A step_ratio >= 20 keeps the existing stronger wording, not the mild one.

    The two warnings must not both fire for the same call (they describe the
    same condition at different severities). The extreme-tier warning is
    guarded to fire at most once per process, so the guard is cleared first
    to make this test independent of test order.
    """
    deepwave._common._warned_once_causes.discard("cfl_step_ratio_extreme")
    grid_spacing = [5.0, 5.0]
    dt = 0.03
    max_vel = 1500.0
    with pytest.warns(
        UserWarning,
        match=re.escape("greatly exceeds the CFL-stable internal time step"),
    ) as record:
        _, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio >= 20
    assert len(record) == 1
    assert "exceeds the stable internal time step;" not in str(record[0].message)


def test_cfl_condition_n_step_ratio_extreme_warns_once_per_process() -> None:
    """The extreme-tier (step_ratio >= 20) warning fires at most once per process.

    A second call with different values -- as happens across an FWI loop's
    evolving model -- must not re-fire the warning, since it is guarded by
    cause rather than by Python's default per-message-text de-duplication.
    """
    deepwave._common._warned_once_causes.discard("cfl_step_ratio_extreme")
    grid_spacing = [5.0, 5.0]
    with pytest.warns(UserWarning, match=re.escape("greatly exceeds the CFL-stable")):
        cfl_condition_n(grid_spacing, 0.03, 1500.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cfl_condition_n(grid_spacing, 0.05, 1500.0)
    assert len(record) == 0


def test_cfl_condition_larger_max_vel() -> None:
    """A larger max_vel requiring more upsampling, still below the warning threshold."""
    grid_spacing = [5.0, 5.0]
    dt = 0.002
    max_vel = 4000.0  # Higher velocity, so CFL will require smaller dt
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 4
    assert inner_dt == pytest.approx(0.0005)
    assert len(record) == 0


def test_cfl_condition_n_zero_max_vel() -> None:
    """A stationary model needs no CFL substeps."""
    assert cfl_condition_n([5.0, 5.0], 0.004, 0.0) == (0.004, 1)


def test_cfl_condition_n_negative_dt() -> None:
    """Test cfl_condition_n with a negative dt."""
    grid_spacing = [5.0, 5.0]
    dt = -0.001
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 1
    assert inner_dt == pytest.approx(-0.001)


def test_cfl_condition_n_zero_dt() -> None:
    """The zero-step limit is represented without dividing by a zero step ratio."""
    assert cfl_condition_n([5.0, 5.0], 0.0, 1500.0) == (0.0, 1)


def test_cfl_condition_n_different_grid_spacing() -> None:
    """Different grid spacings in dimensions, still below the warning threshold."""
    grid_spacing = [5.0, 10.0]
    dt = 0.002
    max_vel = 1500.0
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 2
    assert inner_dt == pytest.approx(0.001)
    assert len(record) == 0


def test_cfl_condition_n_invalid_type_grid_spacing() -> None:
    """Test cfl_condition_n with invalid type for grid_spacing."""
    with pytest.raises(
        TypeError,
        match=re.escape("grid_spacing must be a float or sequence of floats."),
    ):
        cfl_condition_n("invalid", 0.004, 1500.0)


def test_cfl_condition_n_invalid_type_dt() -> None:
    """Test cfl_condition_n with invalid type for dt."""
    with pytest.raises(TypeError, match=re.escape("dt must be a float.")):
        cfl_condition_n([5.0, 5.0], "invalid", 1500.0)


def test_cfl_condition_n_invalid_type_max_vel() -> None:
    """Test cfl_condition_n with invalid type for max_vel."""
    with pytest.raises(TypeError, match=re.escape("max_vel must be a float.")):
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
        match=re.escape("pml_freq must be None or convertible to a float"),
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
@pytest.mark.parametrize(
    "value", [0.5 + 1j, None, 10**1000], ids=["complex", "none", "overflow"]
)
def test_set_freq_taper_frac_invalid_conversion(value: object) -> None:
    """Failed float conversions use the documented Deepwave error."""
    with pytest.raises(
        deepwave.DeepwaveTypeError,
        match=re.escape("freq_taper_frac must be convertible to a float."),
    ):
        set_freq_taper_frac(value)  # type: ignore[arg-type]


# Additional tests for set_time_pad_frac
@pytest.mark.parametrize(
    "value", [0.5 + 1j, None, 10**1000], ids=["complex", "none", "overflow"]
)
def test_set_time_pad_frac_invalid_conversion(value: object) -> None:
    """Failed float conversions use the documented Deepwave error."""
    with pytest.raises(
        deepwave.DeepwaveTypeError,
        match=re.escape("time_pad_frac must be convertible to a float."),
    ):
        set_time_pad_frac(value)  # type: ignore[arg-type]


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


def test_upsample_string_freq_taper_frac_is_converted() -> None:
    """A numeric-string freq_taper_frac/time_pad_frac is used as a float.

    Regression test: _validate_sample_args used to convert-and-discard the
    value, so a valid numeric string still crashed downstream with a raw
    TypeError from comparing a str to a float, and an out-of-range numeric
    string raised the wrong exception type (TypeError instead of ValueError).
    """
    signal = torch.ones(10)
    out_str = upsample(signal, 2, freq_taper_frac="0.5", time_pad_frac="0.1")
    out_float = upsample(signal, 2, freq_taper_frac=0.5, time_pad_frac=0.1)
    assert torch.equal(out_str, out_float)
    with pytest.raises(
        ValueError,
        match=re.escape("freq_taper_frac must be in [0, 1], got 5.0."),
    ):
        upsample(signal, 2, freq_taper_frac="5.0")


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


def test_downsample_string_freq_taper_frac_and_shift_are_converted() -> None:
    """A numeric-string freq_taper_frac/time_pad_frac/shift is used as a float.

    Regression test: _validate_sample_args used to convert-and-discard the
    value, so a valid numeric string still crashed downstream with a raw
    TypeError from comparing/using a str where a float was expected.
    """
    signal = torch.ones(10)
    out_str = downsample(
        signal, 2, freq_taper_frac="0.2", time_pad_frac="0.1", shift="0.5"
    )
    out_float = downsample(signal, 2, freq_taper_frac=0.2, time_pad_frac=0.1, shift=0.5)
    assert torch.equal(out_str, out_float)
    with pytest.raises(
        ValueError,
        match=re.escape("time_pad_frac must be in [0, 1], got 2.0."),
    ):
        downsample(signal, 2, time_pad_frac="2.0")


# Additional tests for cfl_condition_n
def test_cfl_condition_n_tuple_grid_spacing() -> None:
    """Test cfl_condition_n with tuple grid_spacing."""
    grid_spacing = (5.0, 5.0)
    dt = 0.001
    max_vel = 1500.0
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    assert step_ratio == 1
    assert inner_dt == pytest.approx(0.001)


def test_cfl_condition_n_scalar_grid_spacing() -> None:
    """A bare float grid_spacing is equivalent to a one-element sequence."""
    dt = 0.001
    max_vel = 1500.0
    assert cfl_condition_n(5.0, dt, max_vel) == cfl_condition_n([5.0], dt, max_vel)


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


def test_cfl_condition_n_complex_max_vel() -> None:
    """Test cfl_condition_n with complex max_vel."""
    grid_spacing = [5.0, 5.0]
    dt = 0.001
    max_vel = 1500.0 + 1j
    with pytest.raises(
        TypeError,
        match=re.escape("max_vel must be a float."),
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


# Tests for the opt-in accuracy-aware Courant number
def test_critical_courant_number_values() -> None:
    """Critical Courant numbers match the values derived from the stencils.

    Regular grid: C_crit = 2/sqrt(S_M); staggered: C_crit = 2/K_M. See the
    appendix of docs/discretisation.rst.
    """
    assert _critical_courant_number(2, staggered=False) == pytest.approx(1.0)
    assert _critical_courant_number(4, staggered=False) == pytest.approx(
        0.8660254037844386
    )
    assert _critical_courant_number(6, staggered=False) == pytest.approx(
        0.8134892168, abs=1e-9
    )
    assert _critical_courant_number(8, staggered=False) == pytest.approx(
        0.7843687749, abs=1e-9
    )
    assert _critical_courant_number(2, staggered=True) == pytest.approx(1.0)
    assert _critical_courant_number(4, staggered=True) == pytest.approx(6.0 / 7.0)
    assert _critical_courant_number(6, staggered=True) == pytest.approx(120.0 / 149.0)
    assert _critical_courant_number(8, staggered=True) == pytest.approx(
        0.7774178621, abs=1e-9
    )


def test_critical_courant_number_invalid_accuracy() -> None:
    """critical_courant_number rejects unsupported accuracy orders."""
    with pytest.raises(ValueError, match="accuracy must be one of"):
        _critical_courant_number(3, staggered=False)


def test_cfl_condition_n_courant_none_is_default() -> None:
    """courant_number=None reproduces the default (0.6) exactly (bit-identical)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # every case here has step_ratio >= 2
        for grid_spacing, dt, max_vel in (
            ([5.0, 5.0], 0.004, 1500.0),
            ([5.0, 10.0], 0.002, 4000.0),
            ([4.0, 5.0, 6.0], 0.003, 2500.0),
        ):
            default = cfl_condition_n(grid_spacing, dt, max_vel)
            explicit_none = cfl_condition_n(
                grid_spacing, dt, max_vel, courant_number=None
            )
            assert default == explicit_none


def test_cfl_condition_n_courant_larger_dt() -> None:
    """A larger (valid) Courant number gives a larger inner_dt / fewer steps.

    At the default 0.6 this configuration needs step_ratio=4; at 0.9 (still
    below the order-2 critical value of 1.0) it needs only 3.
    """
    grid_spacing = [5.0, 5.0]
    dt = 0.002
    max_vel = 4000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # step_ratio >= 2 in both calls below
        _, default_sr = cfl_condition_n(grid_spacing, dt, max_vel)
        inner_dt, sr = cfl_condition_n(
            grid_spacing,
            dt,
            max_vel,
            courant_number=0.9,
        )
    assert default_sr == 4
    assert sr == 3
    assert inner_dt == pytest.approx(dt / 3)


def test_cfl_condition_n_courant_nonpositive_raises() -> None:
    """A non-positive Courant number raises an error."""
    with pytest.raises(ValueError, match="courant_number must be greater than zero"):
        cfl_condition_n([5.0, 5.0], 0.004, 1500.0, courant_number=0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("argument", ["dt", "max_vel"])
def test_cfl_nonfinite_inputs_are_rejected_before_zero_shortcut(argument, value):
    """An inert step or model must not hide an invalid input on the other side."""
    kwargs = {"dt": 0.0, "max_vel": 0.0, argument: value}
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        cfl_condition_n([5.0], **kwargs)


def test_cfl_courant_requires_real_scalar():
    """Report a non-real Courant argument consistently with other scalar inputs."""
    with pytest.raises(TypeError, match="courant_number must be a float"):
        cfl_condition_n([5.0], 0.001, 1500.0, courant_number=1j)


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
            "Locations should be specified as integer numbers of cells "
            "(within 1e-06 of an integer). If you wish to have a source or "
            "receiver that is not centred on a cell, please consider using "
            "the Hick's method, which is implemented in "
            "deepwave.Hicks.",
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


# Tests for velocity_extremes
def test_velocity_extremes_max_vel_none_matches_actual() -> None:
    """max_vel not given: both extremes are read from the tensor (unchanged)."""
    v = torch.tensor([0.0, 1000.0, -1500.0, 500.0])
    min_nonzero, max_abs = velocity_extremes(v, None, "scalar")
    assert min_nonzero == 500.0
    assert max_abs == 1500.0


def test_velocity_extremes_all_zero_max_vel_none() -> None:
    """max_vel not given, all-zero model: min_nonzero falls back to 0."""
    v = torch.zeros(4)
    min_nonzero, max_abs = velocity_extremes(v, None, "scalar")
    assert min_nonzero == 0.0
    assert max_abs == 0.0


def test_velocity_extremes_max_vel_given_still_reads_actual_max() -> None:
    """max_vel given, not compiling: the actual tensor maximum is still read.

    The returned ``max_abs`` reflects the model's real maximum (not
    ``max_vel`` echoed back), so that ``set_max_vel`` can compare the two and
    warn when ``max_vel`` underestimates the model. This function itself
    never warns -- it only supplies the value ``set_max_vel`` warns from --
    so no warning is expected here even though ``max_vel`` (1.0) is far below
    the model's actual max (1500.0).
    """
    v = torch.tensor([0.0, 1000.0, -1500.0, 500.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        min_nonzero, max_abs = velocity_extremes(v, 1.0, "scalar")
    assert min_nonzero == 500.0
    assert max_abs == 1500.0


def test_velocity_extremes_compiling_skips_both_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both data-dependent reads are skipped while torch.compiler.is_compiling().

    This holds regardless of whether max_vel was given.
    """
    v = torch.tensor([0.0, 1000.0, -1500.0, 500.0])
    monkeypatch.setattr(deepwave._common, "_torch_is_compiling", lambda: True)

    # max_vel given: max is passed through, min falls back to 0.
    min_nonzero, max_abs = velocity_extremes(v, 42.0, "scalar")
    assert min_nonzero == 0.0
    assert max_abs == 42.0

    # max_vel not given: max must still be read (needed for the CFL
    # condition, not just a warning), but min is still skipped.
    min_nonzero, max_abs = velocity_extremes(v, None, "scalar")
    assert min_nonzero == 0.0
    assert max_abs == 1500.0


def test_velocity_extremes_reads_a_transform_wrapped_model() -> None:
    """A transform other than ``vmap`` leaves the model readable.

    A ``grad``/``jacrev``/``jvp`` wrapper leaves the primal readable, and
    reading it gives the same extremes as the untransformed call, so the run
    is discretised the same way inside and outside the transform. Before this,
    every functorch-wrapped model was refused.
    """
    v = torch.tensor([0.0, 1000.0, -1500.0, 500.0])
    seen: list[tuple[float, float]] = []
    torch.func.grad(
        lambda t: (seen.append(velocity_extremes(t, None, "scalar")), t.sum())[1]
    )(v)
    torch.func.jvp(
        lambda t: (seen.append(velocity_extremes(t, None, "scalar")), t.sum())[1],
        (v,),
        (torch.ones_like(v),),
    )
    assert seen == [(500.0, 1500.0), (500.0, 1500.0)]


def test_velocity_extremes_requires_max_vel_for_a_batched_model() -> None:
    """A batched model raises naming ``max_vel=``; with one given it reports it."""
    v = torch.tensor([[0.0, 1000.0, -1500.0, 500.0]])
    with pytest.raises(deepwave.DeepwaveRuntimeError, match="max_vel="):
        torch.func.vmap(lambda t: torch.tensor(velocity_extremes(t, None, "scalar")))(v)
    extremes = torch.func.vmap(
        lambda t: torch.tensor(velocity_extremes(t, 42.0, "scalar"))
    )(v)
    assert extremes.tolist() == [[0.0, 42.0]]


def test_velocity_extremes_compiling_zero_min_triggers_ppw_skip_warning() -> None:
    """The min=0.0 returned while compiling flows into an existing warning.

    It reaches check_points_per_wavelength's existing "test is being
    skipped" path, so the graph-break fix reuses an existing, documented
    warning rather than introducing a new one.
    """
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The test for the number of grid cells per wavelength is being skipped",
        ),
    ):
        check_points_per_wavelength(0.0, 25.0, [5.0, 5.0])


def _reset_pml_freq_warnings() -> None:
    """Discards only this feature's warn-once keys, not the whole cache.

    The migration warning is keyed by the rounded value, so there is no single
    key to name; everything it can produce is prefixed ``pml_freq``. Clearing
    the entire ``_warned_once_causes`` set instead would silently re-arm every
    other guarded warning in the library and make unrelated once-per-process
    tests order-dependent.
    """
    causes = deepwave._common._warned_once_causes
    for cause in [c for c in causes if c.startswith("pml_freq")]:
        causes.discard(cause)
    causes.discard("points_per_wavelength")


# Tests for the pml_freq migration warning: an inferred value far from the
# 25.0 that Deepwave <= 0.0.27 always used must say so, and say how to get
# the old behaviour back.
def test_pml_freq_migration_warning_names_the_old_default_and_the_fix() -> None:
    """The migration warning names both values and the exact line to restore."""
    _reset_pml_freq_warnings()
    with pytest.warns(UserWarning, match="pml_freq was not specified") as record:
        set_pml_freq(None, 1e-8)  # no sources: the 0.5/dt/5 fallback
    message = str(record[0].message)
    assert "pml_freq=25.0" in message
    assert "0.0.27" in message
    assert "1e+07" in message  # the inferred value, 0.5 / 1e-8 / 5


# Tests for check_locations_are_within_model (Item 2: error localisation).
def test_check_locations_are_within_model_reports_name_shot_point_dim() -> None:
    """The error names the argument, shot, point, dimension, value, range."""
    model_shape = [5, 7]
    locations = torch.zeros(2, 3, 2, dtype=torch.long)
    locations[1, 2, 1] = 7  # out of bounds: valid range is [0, 6]
    with pytest.raises(RuntimeError) as excinfo:
        check_locations_are_within_model(model_shape, [locations], ["my_locations"])
    message = str(excinfo.value)
    assert "my_locations" in message
    assert "shot 1" in message
    assert "point 2" in message
    assert "dimension 1" in message
    assert "7" in message
    assert "0 to 6" in message or "0, 6" in message


def test_check_locations_are_within_model_negative_hint() -> None:
    """A negative (non-sentinel) index gets the grid-cells-not-metres hint."""
    model_shape = [5, 7]
    locations = torch.zeros(1, 1, 2, dtype=torch.long)
    locations[0, 0, 0] = -3
    with pytest.raises(RuntimeError) as excinfo:
        check_locations_are_within_model(model_shape, [locations], ["source_locations"])
    message = str(excinfo.value)
    assert "source_locations" in message
    assert "negative" in message
    assert "grid-cell indices" in message
    assert "IGNORE_LOCATION" in message


def test_check_locations_are_within_model_ignore_location_is_valid() -> None:
    """IGNORE_LOCATION entries are always accepted, regardless of model size."""
    model_shape = [5, 7]
    locations = torch.full((1, 1, 2), IGNORE_LOCATION, dtype=torch.long)
    check_locations_are_within_model(model_shape, [locations], ["source_locations"])


def test_check_locations_are_within_model_default_names() -> None:
    """Without explicit names, a generic but still-informative name is used."""
    model_shape = [5, 7]
    locations = torch.zeros(1, 1, 2, dtype=torch.long)
    locations[0, 0, 0] = 100
    with pytest.raises(RuntimeError, match=r"locations\[0\] is out of bounds"):
        check_locations_are_within_model(model_shape, [locations])


def test_check_locations_are_within_model_valid_unchanged() -> None:
    """Valid locations (including whole IGNORE_LOCATION points) do not raise.

    The ignored point has the sentinel in *every* coordinate. This test
    previously set it in only one of the two, which is a partially-sentinel
    row -- silently mis-mapped downstream, and now rejected outright (see
    ``test_check_locations_are_within_model_rejects_partial_ignore``).
    """
    model_shape = [5, 7]
    locations = torch.zeros(2, 3, 2, dtype=torch.long)
    locations[0, 0, :] = IGNORE_LOCATION
    locations[1, 2, 1] = 6  # last valid index
    check_locations_are_within_model(model_shape, [locations], ["source_locations"])


def test_check_locations_are_within_model_rejects_partial_ignore() -> None:
    """A point with IGNORE_LOCATION in only some coordinates is an error.

    IGNORE_LOCATION marks an entire unused source/receiver slot, so it is
    all-or-none across a point's coordinates. A partially-sentinel row used
    to pass validation (each coordinate is checked independently, and the
    sentinel is legal in any single one of them) and was then silently
    mis-mapped by ``_convert_to_1d_and_check_uniqueness``: with dimension 0
    real, the sentinel coordinate contributes 0 and the point is really
    injected/recorded at the *edge* of the padded grid instead of being
    skipped; with dimension 0 the sentinel, the flattened index becomes
    ``IGNORE_LOCATION + stride * coord``, which is not the sentinel any more.
    """
    model_shape = [30, 30]
    for coords in ([15, IGNORE_LOCATION], [IGNORE_LOCATION, 5]):
        locations = torch.tensor([[coords]])
        with pytest.raises(RuntimeError, match=r"mixes deepwave\.IGNORE_LOCATION"):
            check_locations_are_within_model(
                model_shape, [locations], ["source_locations"]
            )

    # A 3-D point with the sentinel in two of three coordinates is equally bad.
    locations = torch.tensor([[[IGNORE_LOCATION, 5, IGNORE_LOCATION]]])
    with pytest.raises(RuntimeError, match=r"mixes deepwave\.IGNORE_LOCATION"):
        check_locations_are_within_model(
            [30, 30, 30], [locations], ["receiver_locations"]
        )


def test_check_locations_are_within_model_column_count_mismatch_is_an_invariant() -> (
    None
):
    """A location with the wrong number of coordinate columns is an AssertionError.

    Not reachable via the public API (get_ndim, called before this function's
    only caller, already enforces that every location's last dimension
    matches the number of spatial dimensions), so this is a defensive
    internal-invariant check, not a user-facing input error.
    """
    model_shape = [5, 7]
    locations = torch.zeros(2, 3, 3, dtype=torch.long)  # 3 columns, 2-D model
    with pytest.raises(AssertionError, match="coordinate columns"):
        check_locations_are_within_model(model_shape, [locations], ["source_locations"])


# Tests for check_locations_within_extents (empty/all-ignore should not crash).
def test_check_locations_within_extents_empty_tensor_does_not_crash() -> None:
    """An empty locations tensor has nothing to validate and must not crash.

    Previously ``.min()``/``.max()`` were called unconditionally on the
    (post-IGNORE_LOCATION-filtered) per-dimension tensor, which raises for an
    empty tensor (see PR #121).
    """
    empty_locations = torch.empty(0, 1, dtype=torch.long)
    check_locations_within_extents([(0, 10)], [empty_locations])


def test_check_locations_within_extents_all_ignore_does_not_crash() -> None:
    """A locations tensor that is entirely IGNORE_LOCATION does not crash."""
    all_ignore = torch.full((1, 3, 1), IGNORE_LOCATION, dtype=torch.long)
    check_locations_within_extents([(0, 10)], [all_ignore])


def test_check_locations_within_extents_none_entry_is_skipped() -> None:
    """A None entry in locations is skipped, as before."""
    check_locations_within_extents([(0, 10)], [None])


def test_check_locations_within_extents_still_rejects_out_of_range() -> None:
    """A genuinely out-of-range location still raises, unaffected by the fix."""
    out_of_range = torch.tensor([[[15]]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="not within"):
        check_locations_within_extents([(0, 10)], [out_of_range])


def test_check_locations_within_extents_accepts_in_range_with_ignored_mixed_in() -> (
    None
):
    """A mix of a valid, in-range location and IGNORE_LOCATION does not raise."""
    mixed = torch.tensor([[[5], [IGNORE_LOCATION]]], dtype=torch.long)
    check_locations_within_extents([(0, 10)], [mixed])


# Tests for check_model_dtype (Item 3: dtype guard).
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_check_model_dtype_rejects_half_precision(dtype: torch.dtype) -> None:
    """float16/bfloat16 models raise a clear, actionable TypeError."""
    v = torch.ones(4, 4, dtype=dtype)
    with pytest.raises(TypeError) as excinfo:
        check_model_dtype([v])
    message = str(excinfo.value)
    assert str(dtype) in message
    assert "float32" in message
    assert "float64" in message
    assert "autocast" in message


def test_check_model_dtype_reports_offending_index() -> None:
    """When several models are given, the offending one's index is reported."""
    v = torch.ones(4, 4)
    rho = torch.ones(4, 4, dtype=torch.float16)
    with pytest.raises(TypeError, match=r"Model tensor 1 \(of 2\)"):
        check_model_dtype([v, rho])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_check_model_dtype_accepts_supported_dtypes(dtype: torch.dtype) -> None:
    """float32/float64 models pass without raising."""
    v = torch.ones(4, 4, dtype=dtype)
    check_model_dtype([v])


def test_scalar_bfloat16_model_raises_dtype_error_not_source_amplitudes() -> None:
    """End-to-end: a bf16 model is rejected before it can mislabel the error.

    Before the dtype guard, this raised a misleading error blaming
    ``source_amplitudes`` (since ``models[0].dtype`` became the "expected"
    dtype for every other input); now it is rejected immediately, naming the
    model tensor and float32/float64.
    """
    import deepwave

    v = (1500 * torch.ones(10, 10)).to(torch.bfloat16)
    with pytest.raises(TypeError) as excinfo:
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=torch.zeros(1, 1, 5),
            source_locations=torch.tensor([[[0, 0]]]),
            receiver_locations=torch.tensor([[[0, 5]]]),
            pml_freq=25.0,
        )
    message = str(excinfo.value)
    assert "float32" in message
    assert "source_amplitudes" not in message


# Tests for check_models (non-Tensor / empty model guards).
def test_check_models_rejects_non_tensor() -> None:
    """A non-Tensor model raises a TypeError naming the argument."""
    with pytest.raises(TypeError, match=re.escape("v must be a torch.Tensor")):
        check_models([[1.0, 2.0], [3.0, 4.0]], ["v"])


def test_check_models_rejects_empty() -> None:
    """A 0-element model Tensor raises a ValueError naming the argument."""
    with pytest.raises(ValueError, match=re.escape("v must not be empty")):
        check_models([torch.empty(0, 4)], ["v"])


def test_check_models_reports_offending_name() -> None:
    """With several models, the offending one's name (not index) is used."""
    v = torch.ones(4, 4)
    with pytest.raises(TypeError, match=re.escape("rho must be a torch.Tensor")):
        check_models([v, [1.0, 2.0]], ["v", "rho"])


def test_check_models_accepts_valid_tensors() -> None:
    """Non-empty Tensor models pass without raising."""
    check_models([torch.ones(4, 4), torch.ones(4, 4)], ["v", "rho"])


# Tests for the "Compiled backend failed" message enrichment (Item 8): the
# error must name the propagator, the direction, and the actual non-zero
# status code, and point at the troubleshooting page.
def test_backend_failure_message_names_propagator_direction_and_status() -> None:
    """The message helper reports the propagator, direction, and status code."""
    import deepwave._engine

    message = deepwave._engine.backend_failure_message("scalar", "forward", 7)
    assert "scalar" in message
    assert "forward" in message
    assert "7" in message
    assert "docs/troubleshooting.rst" in message


def test_run_backend_loop_raises_enriched_message_on_backend_failure() -> None:
    """``run_backend_loop`` raises the enriched message when ``call`` fails.

    A non-zero status code from the chunked call must produce a
    ``RuntimeError`` naming the propagator, direction, and the actual status
    code returned -- not the old bare "Compiled backend failed.".
    """
    import deepwave._engine

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Compiled backend failed: the 'my_propagator' propagator's "
            "compiled forward pass returned non-zero status code 3.",
        ),
    ):
        deepwave._engine.run_backend_loop(
            propagator="my_propagator",
            direction="forward",
            active=True,
            n_outer=1,
            step_ratio=1,
            callback=None,
            callback_frequency=1,
            call=lambda step, step_nt: 3,
            build_state=lambda step: None,  # type: ignore[arg-type,return-value]
        )


def test_run_backend_loop_succeeds_when_call_returns_zero() -> None:
    """A zero status code from ``call`` does not raise."""
    import deepwave._engine

    deepwave._engine.run_backend_loop(
        propagator="my_propagator",
        direction="forward",
        active=True,
        n_outer=1,
        step_ratio=1,
        callback=None,
        callback_frequency=1,
        call=lambda step, step_nt: 0,
        build_state=lambda step: None,  # type: ignore[arg-type,return-value]
    )


# ---------------------------------------------------------------------------
# setup_pml: the pml_freq=0 division-by-zero guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e12, 1e100])
@pytest.mark.parametrize("sign", [-1, 1])
def test_setup_pml_is_invariant_to_time_units(scale: float, sign: int) -> None:
    """Dimensionless damping must not depend on the unit used for time."""
    profiles = [
        setup_pml(
            [20, 20],
            [20.0, 39.0],
            5.0,
            sign * 0.001 * time_scale,
            60,
            1500.0 / time_scale,
            torch.float64,
            torch.device("cpu"),
            25.0 / time_scale,
        )
        for time_scale in (1.0, scale)
    ]
    torch.testing.assert_close(profiles[0], profiles[1], rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pml_width", [[0, 0], [1, 1], [10, 0], [0, 10], [20, 20]])
def test_setup_pml_zero_pml_freq_is_finite(dtype, pml_width) -> None:
    """``pml_freq=0`` must not produce a NaN ``b`` profile.

    ``alpha0 = pi * pml_freq`` is zero at ``pml_freq=0``, and ``sigma`` is zero
    throughout the non-PML interior, so ``sigmaalpha`` is exactly zero there
    and the old ``b = sigma / sigmaalpha * (a - 1)`` was ``0/0``. Unlike ``a``
    (which the following line overwrites in the interior), ``b`` was never
    repaired, so it reached the kernels as NaN and every output went NaN once
    the wave arrived. ``pml_freq=0`` is an explicitly accepted value:
    ``set_pml_freq`` rejects only negatives, and ``check_points_per_wavelength``
    has a dedicated zero branch.
    """
    n = 60
    pml_start = [float(pml_width[0]), float(n - 1 - pml_width[1])]
    a, b = setup_pml(
        pml_width,
        pml_start,
        5.0,
        0.001,
        n,
        1500.0,
        dtype,
        torch.device("cpu"),
        0.0,
    )
    assert bool(torch.isfinite(a).all()), "pml_freq=0 produced a non-finite a profile"
    assert bool(torch.isfinite(b).all()), "pml_freq=0 produced a non-finite b profile"


def test_scalar_propagation_with_zero_pml_freq_is_finite() -> None:
    """End-to-end: a real propagation at ``pml_freq=0`` returns finite data.

    The profile tests above pin ``setup_pml``; this pins the property the user
    actually sees, which is what the NaN ``b`` destroyed. ``nt`` is long enough
    for the wave to cross the model and reach the receiver -- a short run would
    pass even with a NaN profile, since the NaN only spreads once the wavefield
    is non-zero in the interior.
    """
    ny, nx, nt = 40, 40, 400
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(25.0, nt, 0.0005, 0.06)
        .to(torch.float64)
        .reshape(1, 1, nt)
    )
    sources = torch.full((1, 1, 2), 10, dtype=torch.long)
    receivers = torch.full((1, 1, 2), 25, dtype=torch.long)
    out = deepwave.scalar(
        v,
        5.0,
        0.0005,
        source_amplitudes=amplitudes,
        source_locations=sources,
        receiver_locations=receivers,
        pml_freq=0.0,
    )
    receiver_amplitudes = out[-1]
    assert bool(torch.isfinite(receiver_amplitudes).all()), (
        "pml_freq=0 produced non-finite receiver amplitudes"
    )
    assert receiver_amplitudes.abs().max() > 0, "no energy reached the receiver"


# ---------------------------------------------------------------------------
# to_cells: integer coordinates must not truncate a fractional spacing/origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_spacing", [4.5, [4.5, 2.5]])
@pytest.mark.parametrize("grid_origin", [None, [2.5, 0.5]])
def test_to_cells_integer_coordinates_match_float(grid_spacing, grid_origin) -> None:
    """An integer ``coordinates`` Tensor must give the same cells as a float one.

    ``grid_spacing`` and ``grid_origin`` used to be cast to ``coordinates``'
    dtype, so an integer coordinate Tensor -- the natural way to write survey
    geometry in whole metres -- truncated a fractional spacing (``4.5`` ->
    ``4``) or origin (``2.5`` -> ``2``). The subsequent division promotes to
    float, so the result looked entirely reasonable while being up to 12%
    wrong.
    """
    values = [[100, 250], [0, 0], [7, 3], [1000, 12]]
    integer = torch.tensor(values, dtype=torch.long)
    floating = torch.tensor(values, dtype=torch.get_default_dtype())
    from_integer = to_cells(integer, grid_spacing, grid_origin=grid_origin)
    from_float = to_cells(floating, grid_spacing, grid_origin=grid_origin)
    assert torch.equal(from_integer, from_float), (
        f"grid_spacing={grid_spacing}, grid_origin={grid_origin}: integer "
        f"coordinates give {from_integer.tolist()} but float coordinates give "
        f"{from_float.tolist()}"
    )
    assert from_integer.is_floating_point()
    # Non-vacuity: this configuration really does depend on the fractional
    # part, so a truncating implementation would disagree.
    truncating = torch.tensor(values, dtype=torch.long)
    spacing = grid_spacing if isinstance(grid_spacing, list) else [grid_spacing] * 2
    origin_seq = grid_origin if grid_origin is not None else [0.0, 0.0]
    truncated = (
        truncating - torch.tensor([int(o) for o in origin_seq])
    ) / torch.tensor([int(s) for s in spacing])
    assert not torch.allclose(from_integer, truncated)


def test_to_cells_physical_origin_is_spelled_grid_origin() -> None:
    """The physical origin is ``grid_origin``, positionally the third argument.

    ``origin`` on a propagator is a *cell index* -- where an extracted model
    region begins -- while this one is the physical coordinate of cell 0. One
    name for two quantities in the same conversion step was worth removing;
    the keyword spelling is therefore ``grid_origin`` only.
    """
    coordinates = torch.tensor([[10.0, 20.0]])
    by_keyword = to_cells(coordinates, 2.0, grid_origin=[2.0, 4.0])
    assert torch.equal(by_keyword, to_cells(coordinates, 2.0, [2.0, 4.0]))
    assert torch.equal(by_keyword, torch.tensor([[4.0, 8.0]]))
    with pytest.raises(TypeError, match="origin"):
        to_cells(coordinates, 2.0, origin=[2.0, 4.0])  # type: ignore[call-arg]


def test_to_cells_float_coordinates_are_unchanged() -> None:
    """Float input keeps its own dtype and the exact previous values."""
    for dtype in (torch.float32, torch.float64):
        coordinates = torch.tensor([[100.0, 250.0], [7.0, 3.0]], dtype=dtype)
        cells = to_cells(coordinates, 4.5, grid_origin=[2.5, 2.5])
        assert cells.dtype == dtype
        expected = (coordinates - torch.tensor([2.5, 2.5], dtype=dtype)) / torch.tensor(
            [4.5, 4.5], dtype=dtype
        )
        assert torch.equal(cells, expected)


# ---------------------------------------------------------------------------
# model_gradient_sampling_interval: the divisibility guard is grad-mode aware
# ---------------------------------------------------------------------------


def _scalar_run(model, nt, model_gradient_sampling_interval):
    generator = torch.Generator().manual_seed(0)
    amplitudes = torch.randn(1, 1, nt, dtype=torch.float64, generator=generator)
    locations = torch.full((1, 1, 2), 8, dtype=torch.long)
    return deepwave.scalar(
        model,
        5.0,
        0.0005,
        source_amplitudes=amplitudes,
        source_locations=locations,
        receiver_locations=torch.full((1, 1, 2), 14, dtype=torch.long),
        pml_freq=25.0,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
    )[-1]


def test_mgsi_indivisible_nt_raises_when_a_gradient_will_be_computed() -> None:
    """The guard still fires for the case it exists for.

    The history layout still requires complete sampling blocks after the
    scheduler split. Earlier engines also used sampling as their outer time
    step, which silently dropped an incomplete block's final time steps.
    """
    model = torch.nn.Parameter(torch.full((20, 20), 1500.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="must divide the number of time steps"):
        _scalar_run(model, 42, 4)


def test_mgsi_indivisible_nt_is_allowed_under_no_grad() -> None:
    """Under ``torch.no_grad()`` there is no gradient to sample, so no error.

    A model that is an ``nn.Parameter`` keeps ``requires_grad=True`` during an
    inference run, so a guard that looked only at ``requires_grad`` refused a
    forward-only call for a gradient it was never going to compute. The
    ``storage_mode="none"`` guard in the engine does not have this problem --
    it inspects the *internal* models, which autograd has already detached
    under ``no_grad`` -- and this one now matches it.

    The interval is reset to 1, so the numbers must equal a ``mgsi=1`` run.
    """
    model = torch.nn.Parameter(torch.full((20, 20), 1500.0, dtype=torch.float64))
    with torch.no_grad():
        indivisible = _scalar_run(model, 42, 4)
        reference = _scalar_run(model, 42, 1)
    assert torch.equal(indivisible, reference), (
        "a forward-only run under no_grad is not independent of "
        "model_gradient_sampling_interval"
    )
    assert reference.abs().max() > 0, "no energy reached the receiver; test is vacuous"


def test_mgsi_indivisible_nt_is_allowed_when_no_model_requires_grad() -> None:
    """The pre-existing no-gradient case: still allowed, still reset to 1."""
    model = torch.full((20, 20), 1500.0, dtype=torch.float64)
    indivisible = _scalar_run(model, 42, 4)
    reference = _scalar_run(model, 42, 1)
    assert torch.equal(indivisible, reference)


# --- pml_freq inference -----------------------------------------------------


def _ricker(freq: float, nt: int, dt: float) -> torch.Tensor:
    """A Ricker wavelet as a one-shot, one-source amplitude tensor."""
    return deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq).reshape(1, 1, -1)


@pytest.mark.parametrize("freq", [2.0, 5.0, 15.0, 25.0, 40.0])
@pytest.mark.parametrize(("nt", "dt"), [(750, 0.004), (300, 0.004), (1000, 0.002)])
def test_infer_pml_freq_recovers_a_ricker_centre_frequency(
    freq: float, nt: int, dt: float
) -> None:
    """A Ricker's inferred pml_freq is its own centre frequency, within 2%.

    This is the property the whole scheme rests on: the classic seismic call
    (a Ricker at ``freq``, ``pml_freq`` omitted) must resolve to ``freq``, so
    that the 25 Hz case reproduces Deepwave <= 0.0.27 exactly and no other
    case is silently mis-tuned.
    """
    inferred = infer_pml_freq([_ricker(freq, nt, dt)], dt)
    assert inferred == pytest.approx(freq, rel=0.02)


def test_infer_pml_freq_docs_style_call_is_25_to_within_a_tenth_of_a_hz() -> None:
    """The v0.0.27 docs-style call resolves to 25.0, so nothing changes for it."""
    inferred = infer_pml_freq([_ricker(25.0, 750, 0.004)], 0.004)
    assert abs(inferred - 25.0) < 0.1


def test_infer_pml_freq_is_rounded_to_three_significant_figures() -> None:
    """The inferred value is quantised, so small record changes do not move it.

    Trimming, padding or extending a record perturbs the raw peak estimate in
    its fourth significant digit; rounding keeps every such run on the same
    PML (and makes a 25 Hz Ricker infer exactly 25.0).
    """
    assert infer_pml_freq([_ricker(25.0, 750, 0.004)], 0.004) == 25.0
    values = {
        infer_pml_freq([_ricker(25.0, nt, 0.004)], 0.004)
        for nt in (300, 512, 750, 1000, 1234)
    }
    assert values == {25.0}


def test_infer_pml_freq_is_independent_of_zero_padding_the_wavelet() -> None:
    """Padding a record with silence must not move the inferred frequency."""
    wavelet = _ricker(15.0, 200, 0.004)
    padded = torch.cat([wavelet, torch.zeros(1, 1, 800)], dim=-1)
    assert infer_pml_freq([padded], 0.004) == pytest.approx(
        infer_pml_freq([wavelet], 0.004), rel=1e-3
    )


def test_infer_pml_freq_is_independent_of_source_delays_and_polarity() -> None:
    """Sources firing at different times, with different signs, agree.

    The spectra are combined as a sum of squared magnitudes, so neither a
    per-source time shift nor a sign flip changes the answer -- unlike the
    spectrum of the summed traces, where the sources would interfere.
    """
    wavelet = _ricker(15.0, 600, 0.004)[0, 0]
    shifted = torch.stack(
        [
            torch.roll(wavelet, 0),
            -torch.roll(wavelet, 137),
            torch.roll(wavelet, 401),
        ]
    ).reshape(1, 3, -1)
    assert infer_pml_freq([shifted], 0.004) == pytest.approx(
        infer_pml_freq([wavelet.reshape(1, 1, -1)], 0.004), rel=1e-3
    )


def test_infer_pml_freq_weights_channels_equally_despite_unit_scales() -> None:
    """A huge-amplitude channel does not drown out a small-amplitude one.

    Channels (e.g. a pressure and a body force) are in different units, so
    each channel's spectrum is normalised before they are summed. Two
    channels at 10 Hz and 30 Hz, one scaled by 1e6, must give the same answer
    as the same two channels unscaled.
    """
    low = _ricker(10.0, 600, 0.004)
    high = _ricker(30.0, 600, 0.004)
    unscaled = infer_pml_freq([low, high], 0.004)
    scaled = infer_pml_freq([low * 1e6, high], 0.004)
    assert scaled == pytest.approx(unscaled, rel=1e-6)


def test_infer_pml_freq_falls_back_without_usable_sources() -> None:
    """No sources, empty sources, silence, or NaNs all give 0.5/dt/5."""
    assert default_pml_freq(0.004) == 25.0
    assert infer_pml_freq([], 0.004) == 25.0
    assert infer_pml_freq([None], 0.004) == 25.0
    assert infer_pml_freq([torch.zeros(1, 1, 0)], 0.004) == 25.0
    assert infer_pml_freq([torch.zeros(1, 1, 500)], 0.004) == 25.0
    assert infer_pml_freq([torch.full((1, 1, 500), float("nan"))], 0.004) == 25.0
    # A DC-only source has no dominant frequency to find.
    assert infer_pml_freq([torch.ones(1, 1, 500)], 0.004) == 25.0


def test_infer_pml_freq_fallback_is_exactly_a_fifth_of_nyquist() -> None:
    """The source-free default is 0.5/dt/5 for every dt, not a fixed 25."""
    for dt in (0.004, 0.001, 1e-8, 0.008):
        assert infer_pml_freq([], dt) == 0.5 / dt / 5
        assert set_pml_freq(None, dt) == 0.5 / dt / 5


def test_set_pml_freq_never_overrides_an_explicit_value() -> None:
    """An explicit pml_freq is used as given, whatever the sources contain."""
    sources = [_ricker(3.0, 600, 0.004)]
    assert set_pml_freq(30.0, 0.004, sources) == 30.0
    assert set_pml_freq(0.0, 0.004, sources) == 0.0


def test_set_pml_freq_infers_at_the_user_dt_not_the_internal_one() -> None:
    """Inference uses the sample interval of the amplitudes, not the CFL step.

    ``setup_propagator`` calls ``set_pml_freq`` after the CFL condition has
    already shrunk ``dt``, but the source amplitudes are still at the user's
    sample interval; using the internal step would scale the answer by the
    step ratio.
    """
    sources = [_ricker(15.0, 600, 0.004)]
    assert set_pml_freq(None, 0.001, sources, 0.004) == pytest.approx(15.0, rel=0.02)


def test_pml_freq_migration_warning_is_silent_only_at_exactly_25() -> None:
    """The one silent case is the one that reproduces v0.0.27 exactly."""
    _reset_pml_freq_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert set_pml_freq(None, 0.004, [_ricker(25.0, 3000, 0.004)]) == 25.0
    assert [str(w.message) for w in record if "pml_freq" in str(w.message)] == []


@pytest.mark.parametrize(
    ("freq", "warns"),
    [(25.0, False), (25.1, True), (24.9, True), (12.5, True), (50.0, True)],
)
def test_pml_freq_migration_warning_fires_unless_the_value_is_25(
    freq: float, warns: bool
) -> None:
    """Any inferred value but 25.0 changes the boundary, so any but 25.0 warns.

    The old rule was a factor-of-two band around 25.0, which left a call that
    inferred 12.5, 15.0 or 50.0 silently different from what v0.0.27 computed.
    """
    _reset_pml_freq_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        deepwave._common.warn_pml_freq_migration(
            freq,
            stacklevel=2,
            inferred_from=deepwave._common._PML_FREQ_FROM_SPECTRUM,
        )
    assert bool(record) is warns


@pytest.mark.parametrize("freq", [6.0, 15.0, 40.0, 60.0])
def test_pml_freq_migration_warning_fires_in_the_old_quiet_band_too(
    freq: float,
) -> None:
    """A real call that infers 15 or 40 Hz -- once silent -- now warns."""
    _reset_pml_freq_warnings()
    with pytest.warns(UserWarning, match=re.escape("pml_freq=25.0")):
        set_pml_freq(None, 0.004, [_ricker(freq, 3000, 0.004)])


def test_pml_freq_migration_warning_says_which_route_gave_the_value() -> None:
    """The message states how this call's value was reached, not both ways.

    The three routes are the source spectrum, a call with no sources at all,
    and sources with no usable dominant frequency; naming the wrong one, or
    naming several, leaves the reader unable to tell whether their sources
    were used at all -- or sends them to inspect amplitudes they never passed.
    """
    _reset_pml_freq_warnings()
    with pytest.warns(deepwave.PmlFreqInferenceWarning) as spectrum:
        set_pml_freq(None, 0.004, [_ricker(15.0, 3000, 0.004)])
    message = str(spectrum[0].message)
    assert "peak of the source amplitudes' energy spectrum" in message
    assert "fifth of the Nyquist frequency" not in message

    _reset_pml_freq_warnings()
    # No sources at all, and a dt whose fifth-of-Nyquist default is not 25.0.
    with pytest.warns(deepwave.PmlFreqInferenceWarning) as fallback:
        set_pml_freq(None, 0.002)
    message = str(fallback[0].message)
    assert "fifth of the Nyquist frequency" in message
    assert "no source amplitudes to infer a dominant frequency from" in message
    assert "peak of the source amplitudes' energy spectrum" not in message

    _reset_pml_freq_warnings()
    # Sources that exist but hold no dominant frequency: a constant record,
    # whose spectrum peaks at DC.
    with pytest.warns(deepwave.PmlFreqInferenceWarning) as no_peak:
        set_pml_freq(None, 0.002, [torch.ones(1, 1, 40, dtype=torch.float64)])
    message = str(no_peak[0].message)
    assert "fifth of the Nyquist frequency" in message
    assert "no dominant frequency could be inferred from the source" in message
    assert "has no source amplitudes" not in message


def test_pml_freq_migration_warning_fires_once_per_process() -> None:
    """A repeated identical call must not spam an FWI loop with the warning."""
    _reset_pml_freq_warnings()
    sources = [_ricker(3.0, 3000, 0.004)]
    with pytest.warns(UserWarning, match="pml_freq was not specified"):
        set_pml_freq(None, 0.004, sources)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            set_pml_freq(None, 0.004, sources)
    assert [str(w.message) for w in record if "pml_freq" in str(w.message)] == []


def test_check_points_per_wavelength_says_when_the_frequency_was_inferred() -> None:
    """The resolution warning reports where its frequency came from."""
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    with pytest.warns(UserWarning, match="inferred from the source amplitudes"):
        check_points_per_wavelength(100.0, 25.0, [5.0, 5.0], inferred=True)


def _pml_freq_scalar_run(
    freq: float,
    nt: int = 300,
    dt: float = 0.004,
    pml_freq: object = "omit",
) -> "deepwave.PropagatorOutput":
    """A small 2-D scalar run whose wavelet has centre frequency ``freq``."""
    v = torch.full((60, 60), 1500.0, dtype=torch.float64)
    kwargs = {} if pml_freq == "omit" else {"pml_freq": pml_freq}
    return deepwave.scalar(
        v,
        10.0,
        dt,
        source_amplitudes=_ricker(freq, nt, dt).to(torch.float64),
        source_locations=torch.full((1, 1, 2), 30, dtype=torch.long),
        receiver_locations=torch.full((1, 1, 2), 15, dtype=torch.long),
        pml_width=10,
        **kwargs,
    )


def test_run_info_reports_the_resolved_pml_freq() -> None:
    """RunInfo.pml_freq holds the value used, explicit or inferred or default."""
    _reset_pml_freq_warnings()
    explicit = _pml_freq_scalar_run(15.0, pml_freq=33.0)
    assert explicit.info is not None
    assert explicit.info.pml_freq == 33.0

    inferred = _pml_freq_scalar_run(15.0)
    assert inferred.info is not None
    assert inferred.info.pml_freq == pytest.approx(15.0, rel=0.02)

    # No sources at all: the 0.5/dt/5 fallback.
    v = torch.full((60, 60), 1500.0, dtype=torch.float64)
    no_sources = deepwave.scalar(
        v,
        10.0,
        0.004,
        wavefield_0=torch.zeros(1, 60, 60, dtype=torch.float64),
        origin=[0, 0],
        nt=10,
        pml_width=10,
    )
    assert no_sources.info is not None
    assert no_sources.info.pml_freq == 25.0


def test_inferred_pml_freq_reproduces_the_v0_0_27_default_for_a_25hz_ricker() -> None:
    """A 25 Hz Ricker with pml_freq omitted equals the old fixed pml_freq=25.0.

    This is what makes the inference a non-break for the common seismic call:
    the two runs must agree bit-for-bit, not merely closely.
    """
    _reset_pml_freq_warnings()
    omitted = _pml_freq_scalar_run(25.0)
    legacy = _pml_freq_scalar_run(25.0, pml_freq=25.0)
    assert torch.equal(omitted.receiver_amplitudes, legacy.receiver_amplitudes)
    assert legacy.receiver_amplitudes.abs().max() > 0


def test_explicit_pml_freq_is_unaffected_by_the_source_spectrum() -> None:
    """With pml_freq given, changing the wavelet cannot change the boundary."""
    _reset_pml_freq_warnings()
    low = _pml_freq_scalar_run(5.0, pml_freq=25.0)
    assert low.info is not None
    assert low.info.pml_freq == 25.0
    high = _pml_freq_scalar_run(40.0, pml_freq=25.0)
    assert high.info is not None
    assert high.info.pml_freq == 25.0


def test_ultrasound_scale_call_infers_its_own_frequency_without_ppw_warning() -> None:
    """A 3 MHz ultrasound call infers ~3 MHz, not the old seismic 25 Hz.

    The old fixed 25.0 made ``check_points_per_wavelength`` compare the grid
    against a 60 km wavelength, so it never fired however coarse the grid
    was. With the frequency inferred, the same call is judged against its
    real wavelength -- and this (well-resolved) grid passes.
    """
    _reset_pml_freq_warnings()
    dt = 1e-8
    freq = 3e6
    # 1500 m/s / 3 MHz = 0.5 mm wavelength; 50 um cells is 10 per wavelength.
    v = torch.full((60, 60), 1500.0, dtype=torch.float64)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = deepwave.scalar(
            v,
            5e-5,
            dt,
            source_amplitudes=_ricker(freq, 400, dt).to(torch.float64),
            source_locations=torch.full((1, 1, 2), 30, dtype=torch.long),
            receiver_locations=torch.full((1, 1, 2), 15, dtype=torch.long),
            pml_width=10,
        )
    assert out.info is not None
    assert out.info.pml_freq == pytest.approx(freq, rel=0.02)
    assert [
        str(w.message) for w in record if "grid cells per wavelength" in str(w.message)
    ] == []


def test_infer_pml_freq_warns_and_falls_back_when_values_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under torch.compile/vmap the amplitudes' values cannot be read.

    Inference then falls back to 0.5/dt/5 and says so, rather than silently
    giving the compiled run a different PML from the eager one.
    """
    _reset_pml_freq_warnings()
    monkeypatch.setattr(deepwave._common, "_torch_is_compiling", lambda: True)
    with pytest.warns(UserWarning, match="could not be inferred"):
        assert infer_pml_freq([_ricker(15.0, 600, 0.004)], 0.004) == 25.0


def test_infer_pml_freq_reads_a_forward_mode_dual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``torch.func.jvp`` dual is inferred from, not fallen back on.

    ``jvp`` wraps its inputs in a functorch tensor, but a forward-mode dual's
    primal is perfectly readable -- unlike a ``vmap`` ``BatchedTensor``.
    Rejecting it would give the jvp primal a different PML from the same call
    made without the transform, so the two would silently disagree.
    """
    _reset_pml_freq_warnings()
    dt, nt, freq = 0.002, 400, 30.0
    v = torch.full((60, 60), 1500.0)
    src = _ricker(freq, nt, dt)
    kwargs = {
        "source_locations": torch.full((1, 1, 2), 30, dtype=torch.long),
        "receiver_locations": torch.full((1, 1, 2), 12, dtype=torch.long),
        "pml_width": 10,
    }
    seen: list[float] = []

    def run(amplitudes: torch.Tensor) -> torch.Tensor:
        out = deepwave.scalar(v, 10.0, dt, source_amplitudes=amplitudes, **kwargs)
        assert out.info is not None
        seen.append(out.info.pml_freq)
        return out.receiver_amplitudes

    plain = run(src)
    primal, _tangent = torch.func.jvp(run, (src,), (torch.ones_like(src),))
    assert seen == [freq, freq]
    assert torch.equal(plain, primal)
    assert plain.abs().max() > 0


def test_infer_pml_freq_never_exceeds_the_nyquist_frequency() -> None:
    """A near-Nyquist peak must not round to above Nyquist.

    The rounding (and the parabolic offset) can push a near-Nyquist estimate
    just past the limit, which would then trip ``set_pml_freq``'s "pass a
    smaller pml_freq" warning about a value the user never passed.
    """
    _reset_pml_freq_warnings()
    dt = 0.003
    nyquist = 0.5 / dt
    time = torch.arange(4000) * dt
    for freq in (166.0, 166.5, 166.6, 166.65):
        tone = (
            torch.sin(2 * torch.pi * freq * time) * torch.hann_window(4000)
        ).reshape(1, 1, -1)
        assert infer_pml_freq([tone], dt) <= nyquist
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            set_pml_freq(None, dt, [tone])
        # (The migration warning may fire -- 166 Hz is far from 25 -- but the
        # "greater than the Nyquist frequency" one must not.)
        assert [w for w in record if "greater than the Nyquist" in str(w.message)] == []


@pytest.mark.parametrize("scale", [1e20, 1e-25])
def test_infer_pml_freq_is_scale_invariant_in_float32(scale: float) -> None:
    """Amplitudes in awkward units must not overflow or underflow the spectrum.

    Squaring a float32 magnitude overflows above ~1.8e19 and flushes to zero
    below ~1e-19, so a float32 record in the "wrong" units would otherwise lose
    its whole spectrum and silently fall back to 0.5/dt/5.
    """
    _reset_pml_freq_warnings()
    wavelet = _ricker(25.0, 750, 0.004).float() * scale
    assert wavelet.dtype == torch.float32
    assert infer_pml_freq([wavelet], 0.004) == 25.0


def test_pml_freq_inference_runs_after_source_amplitude_validation() -> None:
    """A bad source dtype gets Deepwave's message, not a raw error from the FFT."""
    _reset_pml_freq_warnings()
    v = torch.full((40, 40), 1500.0)
    with pytest.raises(RuntimeError, match="Expected source amplitudes to have"):
        deepwave.scalar(
            v,
            10.0,
            0.004,
            source_amplitudes=_ricker(25.0, 100, 0.004).half(),
            source_locations=torch.full((1, 1, 2), 20, dtype=torch.long),
            receiver_locations=torch.full((1, 1, 2), 10, dtype=torch.long),
            pml_width=10,
        )


# ---------------------------------------------------------------------------
# Initial wavefield shape resolution: the simulation domain (the extracted
# model region plus the PML) or the extracted model region itself.
# ---------------------------------------------------------------------------

_SHAPE_NY = 24
_SHAPE_NX = 26
_SHAPE_PML = 5
_SHAPE_DX = 5.0
_SHAPE_DT = 0.001
_SHAPE_NT = 10
# Locations that fit inside the region the *padded* reading of a
# whole-model-shaped wavefield selects, [0, ny - 2 * pml) x [0, nx - 2 * pml).
_SHAPE_SOURCES_INSIDE = torch.tensor([[[8, 8]]])
_SHAPE_RECEIVERS_INSIDE = torch.tensor([[[3, 3], [12, 14]]])
# A receiver outside that region, so the padded reading cannot be used.
_SHAPE_RECEIVERS_OUTSIDE = torch.tensor([[[3, 3], [20, 22]]])


def _shape_models(name: str, ny: int, nx: int) -> list[torch.Tensor]:
    """Heterogeneous models, so that extracting a different region shows."""
    ramp = (
        torch.arange(ny).reshape(-1, 1) + torch.arange(nx).reshape(1, -1)
    ).float() / (ny + nx)
    if name == "scalar":
        return [1500.0 + 400.0 * ramp]
    if name == "acoustic":
        return [1500.0 + 400.0 * ramp, 2200.0 + 200.0 * ramp]
    if name == "elastic":
        return [
            2.25e9 * (1 + 0.1 * ramp),
            1.0e9 * (1 + 0.1 * ramp),
            (1 / 2200) * (1 + 0.05 * ramp),
        ]
    raise AssertionError(f"no shape-test problem for {name}")


def _shape_kwargs(name: str, sources: torch.Tensor, receivers: torch.Tensor) -> dict:
    """Source, receiver and stability arguments of the shape-test problem.

    ``max_vel`` and ``pml_freq`` are given explicitly because some of these
    tests compare a run on the whole model against a run on a model that has
    already been cropped to the region the whole-model run extracts: both are
    otherwise derived from the *uncropped* model, which would make the two
    runs differ for a reason that has nothing to do with wavefield shapes.
    """
    wavelet = _ricker(15.0, _SHAPE_NT, _SHAPE_DT)
    shared = {"pml_width": _SHAPE_PML, "pml_freq": 15.0, "max_vel": 2000.0}
    suffix = {"scalar": "", "acoustic": "_p", "elastic": "_y"}[name]
    return {
        f"source_amplitudes{suffix}": wavelet,
        f"source_locations{suffix}": sources,
        f"receiver_locations{suffix}": receivers,
        **shared,
    }


_SHAPE_FIELD = {"scalar": "wavefield_0", "acoustic": "pressure_0", "elastic": "vy_0"}


def _shape_run(
    name: str,
    models: list[torch.Tensor],
    wavefield: torch.Tensor,
    sources: torch.Tensor = _SHAPE_SOURCES_INSIDE,
    receivers: torch.Tensor = _SHAPE_RECEIVERS_INSIDE,
    **extra,
) -> "deepwave.PropagatorOutput":
    """One call of ``name`` with ``wavefield`` as its first initial field."""
    kwargs = _shape_kwargs(name, sources, receivers)
    kwargs[_SHAPE_FIELD[name]] = wavefield
    kwargs.update(extra)
    return getattr(deepwave, name)(*models, _SHAPE_DX, _SHAPE_DT, **kwargs)


def _shape_state(out: "deepwave.PropagatorOutput") -> dict:
    """Every Tensor an output carries, including the auxiliary state's."""
    tensors: dict = {}
    for name, value in out.as_dict().items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value
        elif isinstance(value, deepwave.AuxiliaryState):
            for key, tensor in value.items():
                tensors[f"auxiliary_state.{key}"] = tensor
    return tensors


def _shape_outputs_equal(
    a: "deepwave.PropagatorOutput",
    b: "deepwave.PropagatorOutput",
) -> bool:
    """Whether two outputs are bit-identical in every Tensor they carry."""
    left = _shape_state(a)
    right = _shape_state(b)
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def _pad_to_simulation_domain(wavefield: torch.Tensor) -> torch.Tensor:
    """Zero-pad a model-shaped wavefield by the PML width on each face."""
    return torch.nn.functional.pad(wavefield, (_SHAPE_PML,) * 4)


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_model_shaped_initial_wavefield_padded_internally(name: str) -> None:
    """A model-shaped initial wavefield equals the hand-padded one exactly.

    A receiver outside the region that the padded reading of a whole-model
    shaped wavefield would select means that reading cannot be used, so the
    wavefield is taken to be the extracted model region and zero-padded
    internally. That must give bit-identical results to doing the padding by
    hand, in every output Tensor, not just the receiver data.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    torch.manual_seed(0)
    model_shaped = torch.rand(1, _SHAPE_NY, _SHAPE_NX)
    hand_padded = _pad_to_simulation_domain(model_shaped)
    assert list(hand_padded.shape[1:]) == [
        _SHAPE_NY + 2 * _SHAPE_PML,
        _SHAPE_NX + 2 * _SHAPE_PML,
    ]
    common = {
        "receivers": _SHAPE_RECEIVERS_OUTSIDE,
        "origin": [0, 0],
    }
    out_model = _shape_run(name, models, model_shaped, **common)
    out_padded = _shape_run(name, models, hand_padded, **common)
    assert _shape_outputs_equal(out_model, out_padded)
    # The run is not vacuous: something actually propagated.
    assert any(tensor.abs().sum() > 0 for tensor in _shape_state(out_model).values())
    # The returned state is at the simulation-domain shape either way.
    assert list(out_model.as_dict()[_SHAPE_FIELD[name]].shape[1:]) == [
        _SHAPE_NY + 2 * _SHAPE_PML,
        _SHAPE_NX + 2 * _SHAPE_PML,
    ]


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_model_shaped_initial_wavefield_is_differentiable(name: str) -> None:
    """A gradient reaches a model-shaped initial wavefield, at its own shape."""
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    torch.manual_seed(0)
    model_shaped = torch.rand(1, _SHAPE_NY, _SHAPE_NX).requires_grad_()
    out = _shape_run(
        name,
        models,
        model_shaped,
        receivers=_SHAPE_RECEIVERS_OUTSIDE,
        origin=[0, 0],
    )
    sum(tensor.sum() for tensor in _shape_state(out).values()).backward()
    assert model_shaped.grad is not None
    assert list(model_shaped.grad.shape) == [1, _SHAPE_NY, _SHAPE_NX]
    assert model_shaped.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_padded_shape_still_accepted(name: str) -> None:
    """A simulation-domain-shaped wavefield keeps the meaning it always had.

    With all sources and receivers inside it, the padded reading of a
    whole-model-shaped wavefield is valid, so the extracted region is
    ``model - 2 * pml``. Running on a model that has already been cropped to
    that region must give bit-identical results, and running with the
    wavefield zero-padded to the whole simulation domain (the model-shaped
    reading) must not -- otherwise this test could not tell the two apart.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    inner_ny = _SHAPE_NY - 2 * _SHAPE_PML
    inner_nx = _SHAPE_NX - 2 * _SHAPE_PML
    cropped = [model[:inner_ny, :inner_nx].contiguous() for model in models]
    torch.manual_seed(0)
    whole_model_shaped = torch.rand(1, _SHAPE_NY, _SHAPE_NX)
    out = _shape_run(name, models, whole_model_shaped, origin=[0, 0])
    reference = _shape_run(name, cropped, whole_model_shaped, origin=[0, 0])
    assert _shape_outputs_equal(out, reference)
    other_reading = _shape_run(
        name,
        models,
        _pad_to_simulation_domain(whole_model_shaped),
        origin=[0, 0],
    )
    assert not _shape_outputs_equal(out, other_reading)


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_ambiguous_shape_warns_and_keeps_old_meaning(name: str) -> None:
    """The whole-model shape with no origin warns but keeps the old meaning.

    With no ``origin`` and no ``survey_pad``, a wavefield whose spatial shape
    is the whole model could be either reading. The padded one is kept,
    because that is the meaning the call has always had, and the extents
    warning says so. A true whole-model restart (``model + 2 * pml``) is not
    ambiguous and must not get the extra text.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    torch.manual_seed(0)
    whole_model_shaped = torch.rand(1, _SHAPE_NY, _SHAPE_NX)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = _shape_run(name, models, whole_model_shaped)
    messages = [
        str(w.message)
        for w in record
        if "Survey extents were inferred" in str(w.message)
    ]
    assert len(messages) == 1
    assert "whole model unpadded" in messages[0]
    # It names both one-line fixes.
    assert "origin=[0, 0]" in messages[0]
    assert (
        f"spatial shape [{_SHAPE_NY + 2 * _SHAPE_PML}, {_SHAPE_NX + 2 * _SHAPE_PML}]"
        in messages[0]
    )
    # ... and the meaning is still the padded one: model - 2 * pml.
    inner_ny = _SHAPE_NY - 2 * _SHAPE_PML
    inner_nx = _SHAPE_NX - 2 * _SHAPE_PML
    assert f"[(0, {inner_ny}), (0, {inner_nx})]" in messages[0]
    cropped = [model[:inner_ny, :inner_nx].contiguous() for model in models]
    reference = _shape_run(name, cropped, whole_model_shaped, origin=[0, 0])
    assert _shape_outputs_equal(out, reference)
    other_reading = _shape_run(
        name,
        models,
        _pad_to_simulation_domain(whole_model_shaped),
        origin=[0, 0],
    )
    assert not _shape_outputs_equal(out, other_reading)

    # A true whole-model restart is not ambiguous: it still warns that the
    # extents were inferred, but without the extra text.
    restart_shaped = torch.rand(
        1,
        _SHAPE_NY + 2 * _SHAPE_PML,
        _SHAPE_NX + 2 * _SHAPE_PML,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _shape_run(name, models, restart_shaped)
    restart_messages = [
        str(w.message)
        for w in record
        if "Survey extents were inferred" in str(w.message)
    ]
    assert len(restart_messages) == 1
    assert "whole model unpadded" not in restart_messages[0]
    assert f"[(0, {_SHAPE_NY}), (0, {_SHAPE_NX})]" in restart_messages[0]


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_ambiguous_shape_warns_even_with_an_explicit_origin(name: str) -> None:
    """An explicit ``origin`` does not make an ambiguous shape unambiguous.

    ``origin`` says where the extracted region begins; it says nothing about
    whether the supplied shape includes the PML. So whenever both readings
    of a shape are usable and they select different regions, the ambiguity
    warning must fire whether or not ``origin`` was given -- otherwise the
    contents of a model-shaped wavefield are silently displaced by the
    low-face PML width relative to the model coordinates the user placed
    them at, with no diagnostic of any kind.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    torch.manual_seed(0)
    whole_model_shaped = torch.rand(1, _SHAPE_NY, _SHAPE_NX)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = _shape_run(name, models, whole_model_shaped, origin=[0, 0])
    messages = [str(w.message) for w in record if "is ambiguous" in str(w.message)]
    assert len(messages) == 1
    # Nothing was inferred: origin was given, so only the ambiguity fires.
    assert "Survey extents were inferred" not in messages[0]
    # It names both readings and the extents each selects.
    inner_ny = _SHAPE_NY - 2 * _SHAPE_PML
    inner_nx = _SHAPE_NX - 2 * _SHAPE_PML
    assert f"[(0, {inner_ny}), (0, {inner_nx})]" in messages[0]
    assert f"[(0, {_SHAPE_NY}), (0, {_SHAPE_NX})]" in messages[0]
    assert "origin" in messages[0]
    # ... and the numbers did not change: the padded reading still wins.
    cropped = [model[:inner_ny, :inner_nx].contiguous() for model in models]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = _shape_run(name, cropped, whole_model_shaped, origin=[0, 0])
    assert _shape_outputs_equal(out, reference)


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_unambiguous_shapes_do_not_warn_about_ambiguity(name: str) -> None:
    """Only a genuinely ambiguous shape gets the ambiguity warning.

    A simulation-domain-shaped wavefield whose model-shaped reading would
    run off the end of the model has only one usable reading, and a call
    with no PML has two readings that select the same region. Neither is
    ambiguous, so neither may be warned about -- a warning that fires on
    unambiguous calls would be noise that trains users to ignore it.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    torch.manual_seed(0)
    # One usable reading: model + 2 * pml cannot itself be a model region.
    restart_shaped = torch.rand(
        1,
        _SHAPE_NY + 2 * _SHAPE_PML,
        _SHAPE_NX + 2 * _SHAPE_PML,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _shape_run(name, models, restart_shaped, origin=[0, 0])
    assert [w for w in record if "is ambiguous" in str(w.message)] == []

    # Two readings that agree: with no PML they select the same region.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _shape_run(
            name,
            models,
            torch.rand(1, _SHAPE_NY, _SHAPE_NX),
            origin=[0, 0],
            pml_width=0,
        )
    assert [w for w in record if "is ambiguous" in str(w.message)] == []

    # survey_pad picks the region itself, so the shape is only checked
    # against it: it raises rather than choosing between readings.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(deepwave.DeepwaveValueError):
            _shape_run(
                name,
                models,
                torch.rand(1, _SHAPE_NY, _SHAPE_NX),
                survey_pad=0,
            )
    assert [w for w in record if "is ambiguous" in str(w.message)] == []


def test_ambiguous_model_shaped_wavefield_is_displaced_and_says_so() -> None:
    """The displacement that makes this ambiguity matter is warned about.

    An impulse written at index ``i`` of a model-shaped ``wavefield_0`` is
    re-read as a simulation domain, so it acts at model index
    ``i - pml_width[low face]``. That is a silent wrong number if nothing
    says the shape was read the other way, so this pins both halves: the
    displacement is real, and the warning fires for every width that
    produces one, with an explicit ``origin`` given throughout.
    """
    ny, nx = 400, 20
    v = torch.full((ny, nx), 1500.0)
    receiver = torch.tensor([[[100, 10]]])
    peaks = []
    for pml in (0, 20, 40):
        wavefield = torch.zeros(1, ny, nx)
        wavefield[0, 200, 10] = 1.0
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = deepwave.scalar(
                v,
                2.0,
                0.0005,
                source_amplitudes=torch.zeros(1, 1, 600),
                source_locations=torch.tensor([[[10, 10]]]),
                receiver_locations=receiver,
                wavefield_0=wavefield,
                origin=[0, 0],
                pml_width=[pml, 40, 0, 0],
                accuracy=4,
                pml_freq=25.0,
            )
        ambiguity = [str(w.message) for w in record if "is ambiguous" in str(w.message)]
        assert len(ambiguity) == 1, f"no ambiguity warning at pml_width[0]={pml}"
        assert f"[(0, {ny - pml - 40}), (0, {nx})]" in ambiguity[0]
        assert f"[(0, {ny}), (0, {nx})]" in ambiguity[0]
        peaks.append(int(out.receiver_amplitudes[0, 0].abs().argmax()))
    # The arrival moves earlier as the low-face PML grows, which is the
    # displacement the warning now announces.
    assert peaks[0] > peaks[1] > peaks[2]


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_unrecognised_shape_error_names_both_shapes(name: str) -> None:
    """A shape that is neither accepted shape is refused, naming both.

    The two shapes the message names are then used, and both are accepted --
    so the message cannot drift away from what the code does. The message
    also names the *region* this call selected (origin and extent), which is
    what tells the caller whether they meant to continue somewhere else at
    all, rather than only that their tensor is the wrong size.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    wrong = torch.zeros(1, 7, 7)
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _shape_run(name, models, wrong, survey_pad=0)
    message = str(excinfo.value)
    assert "has spatial shape [7, 7]" in message
    assert "Pass one of those, or pass initial_state=out" in message
    region = re.search(r"origin=\[(\d+), (\d+)\], extent=\[(\d+), (\d+)\]", message)
    assert region is not None, message
    # The reported shape, then the region's origin and extent, then the
    # simulation domain, then the model region.
    shapes = re.findall(r"\[(\d+), (\d+)\]", message)
    assert len(shapes) == 5
    padded_shape = [int(shapes[3][0]), int(shapes[3][1])]
    model_shape = [int(shapes[4][0]), int(shapes[4][1])]
    # The region's extent is the model-region shape the message accepts.
    assert [int(region.group(3)), int(region.group(4))] == model_shape
    assert padded_shape == [
        model_shape[0] + 2 * _SHAPE_PML,
        model_shape[1] + 2 * _SHAPE_PML,
    ]
    for accepted in (padded_shape, model_shape):
        _shape_run(name, models, torch.zeros(1, *accepted), survey_pad=0)

    # The same message comes from a second initial wavefield whose shape does
    # not match the region the first one selected.
    if name == "scalar":
        with pytest.raises(deepwave.DeepwaveValueError, match=r"accepted shapes are"):
            _shape_run(
                name,
                models,
                torch.zeros(1, _SHAPE_NY, _SHAPE_NX),
                receivers=_SHAPE_RECEIVERS_OUTSIDE,
                origin=[0, 0],
                wavefield_m1=torch.zeros(1, 7, 7),
            )


def test_unrecognised_shape_error_uses_the_propagator_field_name() -> None:
    """Callers that know the field's name get it in the message."""
    extents = [(0, 20), (0, 30)]
    pml_width = [1, 2, 3, 4]
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        deepwave._common.prepare_wavefields(
            [torch.zeros(1, 5, 6)],
            extents,
            pml_width,
            1,
            torch.device("cpu"),
            torch.float32,
            wavefield_names=["pressure_0"],
        )
    message = str(excinfo.value)
    assert message.startswith("pressure_0 has spatial shape [5, 6].")
    assert "[23, 37]" in message
    assert "[20, 30]" in message


def test_model_shaped_wavefield_is_zero_padded_by_prepare_wavefields() -> None:
    """``prepare_wavefields`` pads a model-shaped wavefield, and only that one."""
    extents = [(0, 4), (0, 5)]
    pml_width = [1, 2, 3, 4]
    model_shaped = torch.arange(20).reshape(1, 4, 5).float() + 1
    already_padded = torch.ones(1, 7, 12)
    prepared = deepwave._common.prepare_wavefields(
        [model_shaped, already_padded, None],
        extents,
        pml_width,
        1,
        torch.device("cpu"),
        torch.float32,
    )
    assert list(prepared[0].shape) == [1, 7, 12]
    assert torch.equal(prepared[0][:, 1:5, 3:8], model_shaped)
    assert prepared[0].sum() == model_shaped.sum()
    assert prepared[1] is already_padded
    assert prepared[2].numel() == 0


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
def test_initial_state_restart_is_never_shape_ambiguous(name: str) -> None:
    """Continuing with ``initial_state`` cannot hit the ambiguous shape case.

    This constructs the ambiguity deliberately: the first call extracts
    ``model - 2 * pml``, so the state it returns has the spatial shape of the
    whole model, which is exactly the shape that is ambiguous when it is
    passed as a bare wavefield with no origin. Passing the output back as
    ``initial_state`` is unambiguous anyway, because the auxiliary state's
    fingerprint pins the origin, so no extents are inferred at all.
    """
    models = _shape_models(name, _SHAPE_NY, _SHAPE_NX)
    torch.manual_seed(0)
    whole_model_shaped = torch.rand(1, _SHAPE_NY, _SHAPE_NX)
    first = _shape_run(name, models, whole_model_shaped, origin=[0, 0])
    # The returned state really does have the ambiguous shape.
    assert list(first.as_dict()[_SHAPE_FIELD[name]].shape[1:]) == [
        _SHAPE_NY,
        _SHAPE_NX,
    ]
    kwargs = _shape_kwargs(name, _SHAPE_SOURCES_INSIDE, _SHAPE_RECEIVERS_INSIDE)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        restarted = getattr(deepwave, name)(
            *models,
            _SHAPE_DX,
            _SHAPE_DT,
            initial_state=first,
            **kwargs,
        )
    assert [w for w in record if "Survey extents were inferred" in str(w.message)] == []
    explicit = getattr(deepwave, name)(
        *models,
        _SHAPE_DX,
        _SHAPE_DT,
        initial_state=first,
        origin=[0, 0],
        **kwargs,
    )
    assert _shape_outputs_equal(restarted, explicit)


# ---------------------------------------------------------------------------
# to_cells(snap=) and deepwave.ON_NODE_TOLERANCE
# ---------------------------------------------------------------------------


def test_on_node_tolerance_is_single_sourced() -> None:
    """One number decides "on node" everywhere it is decided.

    The propagators' routing test, ``Hicks``' default ``eps`` and
    ``to_cells(snap=True)`` must all use the same tolerance, or a
    coordinate could be on-node for one and off-node for another.
    """
    import inspect

    from deepwave import _location_interpolation as li

    assert deepwave.ON_NODE_TOLERANCE == 1e-5
    assert deepwave.ON_NODE_TOLERANCE is li._DEFAULT_EPS
    hicks_eps = inspect.signature(deepwave.Hicks.__init__).parameters["eps"].default
    assert hicks_eps == deepwave.ON_NODE_TOLERANCE


def test_to_cells_snap_false_is_bit_identical() -> None:
    """The default is the plain division, unchanged, bit for bit.

    ``snap`` is opt-in precisely so that no existing call moves; this pins
    that against the arithmetic written out by hand.
    """
    coordinates = torch.tensor(
        [[0.0, 1.0], [123.456, -7.5], [1e-9, 4321.0]], dtype=torch.float64
    )
    expected = (coordinates - torch.tensor([2.5, -1.0], dtype=torch.float64)) / (
        torch.tensor([4.5, 7.0], dtype=torch.float64)
    )
    got = deepwave.to_cells(coordinates, [4.5, 7.0], [2.5, -1.0])
    assert torch.equal(got, expected)
    assert torch.equal(
        deepwave.to_cells(coordinates, [4.5, 7.0], [2.5, -1.0], snap=False), expected
    )


def test_to_cells_snap_true_snaps_only_within_tolerance() -> None:
    """``snap`` cleans up floating-point error; it does not round.

    A coordinate that is genuinely off-node stays off-node however it is
    snapped -- otherwise ``snap=True`` would silently move a source onto a
    grid node and change the physics.
    """
    tol = deepwave.ON_NODE_TOLERANCE
    cells = torch.tensor(
        [
            [7.0 - 0.5 * tol, 7.0 + 0.5 * tol],  # within: snaps to 7
            [7.0 - 10 * tol, 7.0 + 10 * tol],  # outside: unchanged
            [7.4, -3.6],  # genuinely off-node: unchanged
        ],
        dtype=torch.float64,
    )
    snapped = deepwave.to_cells(cells, 1.0, snap=True)
    assert snapped[0, 0] == 7.0
    assert snapped[0, 1] == 7.0
    assert torch.equal(snapped[1], cells[1])
    assert torch.equal(snapped[2], cells[2])
    # Shape, dtype and device are untouched.
    assert snapped.shape == cells.shape
    assert snapped.dtype == cells.dtype
    assert snapped.device == cells.device


def test_to_cells_snap_accepts_a_custom_tolerance() -> None:
    """A float ``snap`` is the tolerance, in cells."""
    tol = deepwave.ON_NODE_TOLERANCE
    cells = torch.tensor([[7.0 + 10 * tol]], dtype=torch.float64)
    assert deepwave.to_cells(cells, 1.0, snap=True)[0, 0] != 7.0
    assert deepwave.to_cells(cells, 1.0, snap=100 * tol)[0, 0] == 7.0


def test_to_cells_snap_rejects_a_negative_tolerance() -> None:
    """A negative tolerance is a mistake, not a no-op."""
    cells = torch.tensor([[1.0]], dtype=torch.float64)
    with pytest.raises(deepwave.DeepwaveValueError, match="non-negative"):
        deepwave.to_cells(cells, 1.0, snap=-1e-3)


def test_to_cells_snap_is_keyword_only() -> None:
    """``snap`` cannot be passed positionally.

    The three existing parameters are positional in released code and in the
    documentation, so a fourth positional would be a trap.
    """
    cells = torch.tensor([[1.0]], dtype=torch.float64)
    with pytest.raises(TypeError):
        deepwave.to_cells(cells, 1.0, None, True)  # type: ignore[misc]


def test_to_cells_snap_puts_a_survey_back_on_node() -> None:
    """The motivating case: exact metres that divide exactly, in float32.

    Without snapping these land a few ulp off an integer, which routes an
    on-node survey through the interpolated path and changes the numbers.
    """
    spacing = 12.5
    coordinates = torch.tensor(
        [[i * spacing for i in range(1, 40)]], dtype=torch.float32
    ).unsqueeze(-1)
    raw = deepwave.to_cells(coordinates, spacing)
    snapped = deepwave.to_cells(coordinates, spacing, snap=True)
    assert torch.equal(snapped, snapped.round())
    # And it really was a cleanup, not a move: nothing shifted by more than
    # the tolerance.
    assert (raw - snapped).abs().max() <= deepwave.ON_NODE_TOLERANCE


# ---------------------------------------------------------------------------
# critical_courant_number(propagator=)
# ---------------------------------------------------------------------------


_ALL_PROPAGATOR_NAMES = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)


@pytest.mark.parametrize("name", _ALL_PROPAGATOR_NAMES)
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_critical_courant_number_propagator_matches_staggered(
    name: str, accuracy: int
) -> None:
    """Naming the propagator gives the same answer as knowing its grid."""
    from deepwave.introspection import _resolve_spec

    staggered = _resolve_spec(name).staggered
    assert deepwave.discretisation.critical_courant_number(
        accuracy, propagator=name
    ) == _critical_courant_number(accuracy, staggered)


def test_critical_courant_number_accepts_all_three_spellings() -> None:
    """The functional, the Module class and the name all resolve."""
    ccn = deepwave.discretisation.critical_courant_number
    expected = _critical_courant_number(4, True)
    assert ccn(4, propagator=deepwave.elastic) == expected
    assert ccn(4, propagator=deepwave.Elastic) == expected
    assert ccn(4, propagator="elastic") == expected


def test_critical_courant_number_has_no_staggered_argument() -> None:
    """``staggered=`` never shipped, and is not a parameter.

    ``propagator=`` knows which grid each propagator uses, so the argument had
    nothing to add and two ways to disagree with itself. It was only ever on
    the development branch, so it is gone rather than deprecated.
    """
    assert (
        "staggered"
        not in inspect.signature(
            deepwave.discretisation.critical_courant_number
        ).parameters
    )
    with pytest.raises(TypeError, match="staggered"):
        deepwave.discretisation.critical_courant_number(4, staggered=True)  # type: ignore[call-arg]


def test_critical_courant_number_requires_one_of_them() -> None:
    """Neither given raises, naming ``propagator=`` and the propagators."""
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        deepwave.discretisation.critical_courant_number(4)
    message = str(excinfo.value)
    assert "propagator=" in message
    for name in _ALL_PROPAGATOR_NAMES:
        assert name in message


def test_critical_courant_number_rejects_an_unknown_propagator() -> None:
    """An unknown name lists the ones that exist."""
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        deepwave.discretisation.critical_courant_number(4, propagator="acoustic_tti")
    for name in _ALL_PROPAGATOR_NAMES:
        assert name in str(excinfo.value)


def test_critical_courant_number_internal_use_does_not_warn() -> None:
    """A propagator call must not emit the ``staggered=`` deprecation.

    Deepwave's own stability check needs the table, and calls the private
    helper for exactly this reason; if it ever called the public function
    with ``staggered=``, every propagator call would warn.
    """
    v = torch.full((30, 30), 1500.0)
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=source_amplitudes,
            source_locations=torch.tensor([[[15, 15]]]),
            pml_freq=25.0,
        )
    assert not [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "critical_courant_number" in str(w.message)
    ]


# ---------------------------------------------------------------------------
# A location error names the caller's own argument
# ---------------------------------------------------------------------------


_ALL_ELEVEN_PROPAGATORS = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)


@pytest.mark.parametrize("name", _ALL_ELEVEN_PROPAGATORS)
@pytest.mark.parametrize("kind", ["source", "receiver"])
def test_location_group_names_are_the_callers_arguments(name: str, kind: str) -> None:
    """The shared setup code names arguments, not the slots it holds them in.

    The propagators gather their location arguments into one list in
    whatever order suits them -- the acoustic family puts the pressure
    channel first, the elastic family last, a Born propagator hands the
    source list twice -- so a slot index is neither something the caller
    wrote nor the same thing on two propagators. This checks the names
    against the real signature, one channel at a time, rather than trusting
    the layout rule the helper encodes.
    """
    from deepwave._common import _location_group_names
    from deepwave.introspection import _resolve_spec

    spec = _resolve_spec(name)
    ndims = 2
    if kind == "source":
        expected = [
            spec.channel_kwargs("source", channel)[1]
            for channel in spec.source_channels(ndims)
            if channel not in spec.synthetic_source_channels
        ]
        if spec.has_bg_receivers():
            expected = expected * 2
    else:
        kinds = (
            ("bg_receiver", "sc_receiver") if spec.has_bg_receivers() else ("receiver",)
        )
        expected = [
            spec.channel_kwargs(k, channel)[0]
            for k in kinds
            for channel in spec.receiver_channels(ndims)
            if channel not in spec.synthetic_receiver_channels
        ]
    got = _location_group_names(f"{kind}_locations", len(expected), name, ndims, kind)
    if len(expected) <= 1:
        # A single-group propagator's argument is the unsuffixed stem, which
        # is what the helper returns without consulting anything.
        assert got == [f"{kind}_locations"]
        return
    assert set(got) == set(expected), (name, kind)
    assert not any("[" in n for n in got), (name, kind, got)


def test_location_group_names_fall_back_without_a_propagator() -> None:
    """A caller that cannot name the propagator still gets a usable label."""
    from deepwave._common import _location_group_names

    assert _location_group_names("source_locations", 3) == [
        "source_locations[0]",
        "source_locations[1]",
        "source_locations[2]",
    ]


def test_out_of_bounds_location_names_the_argument_that_was_passed() -> None:
    """End to end: the message quotes a string the caller can search for."""
    lamb = torch.full((30, 30), 2.25e9, dtype=torch.float64)
    mu = torch.full((30, 30), 1.0e9, dtype=torch.float64)
    buoyancy = torch.full((30, 30), 1.0 / 2200.0, dtype=torch.float64)
    amplitudes = torch.zeros(1, 1, 10, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="source_locations_y is out of bounds"):
        deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            4.0,
            0.0005,
            source_amplitudes_y=amplitudes,
            source_locations_y=torch.tensor([[[8, 9999]]]),
            pml_width=5,
            pml_freq=15.0,
        )


def test_models_of_different_shapes_give_one_shared_message() -> None:
    """The cross-model shape check runs before any model arithmetic.

    The elastic and stiffness families multiply two models together to find
    the maximum wave speed before the shared setup code sees them, so a
    mismatched pair used to surface as torch's own broadcast error, which
    names neither the models nor the propagator.
    """
    good = torch.full((30, 30), 1.0e9, dtype=torch.float64)
    bad = torch.full((29, 30), 1.0e9, dtype=torch.float64)
    amplitudes = torch.zeros(1, 1, 10, dtype=torch.float64)
    with pytest.raises(ValueError, match="same spatial shape") as excinfo:
        deepwave.elastic(
            good,
            bad,
            torch.full((30, 30), 1.0 / 2200.0, dtype=torch.float64),
            4.0,
            0.0005,
            source_amplitudes_y=amplitudes,
            source_locations_y=torch.tensor([[[8, 8]]]),
            pml_width=5,
            pml_freq=15.0,
        )
    assert "lamb" in str(excinfo.value)
    assert "mu" in str(excinfo.value)


def test_batched_and_unbatched_models_are_still_a_valid_pair() -> None:
    """The shape check compares trailing dimensions only."""
    from deepwave._common import check_model_shapes_match

    check_model_shapes_match([torch.zeros(3, 10, 10), torch.zeros(10, 10)], ["a", "b"])


def test_grid_spacing_accepts_builtin_conversion_protocols() -> None:
    """Type-checking helpers must preserve float's buffer and sequence inputs."""

    class IndexedSpacing:
        def __getitem__(self, index: int) -> float:
            return (4.0, 5.0)[index]

    assert set_grid_spacing(memoryview(b"4"), 2) == [4.0, 4.0]
    assert set_grid_spacing(IndexedSpacing(), 2) == [4.0, 5.0]
    assert set_pml_width(memoryview(b"2"), 2) == [2, 2, 2, 2]

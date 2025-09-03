import re

import pytest
import torch

import deepwave
import deepwave.common


def test_set_survey_pad():
    """Check set_survey_pad."""
    survey_pad = deepwave.common.set_survey_pad(1, 2)
    assert survey_pad == [1, 1, 1, 1]
    survey_pad = deepwave.common.set_survey_pad(None, 2)
    assert survey_pad == [-1, -1, -1, -1]
    survey_pad = deepwave.common.set_survey_pad([1, 2, 3, 4], 2)
    assert survey_pad == [1, 2, 3, 4]
    survey_pad = deepwave.common.set_survey_pad([1, None, None, 4], 2)
    assert survey_pad == [1, -1, -1, 4]
    with pytest.raises(
        RuntimeError,
        match=r"survey_pad must have length 2 \* dims in model, but got 1.",
    ):
        deepwave.common.set_survey_pad([1], 2)
    with pytest.raises(
        RuntimeError,
        match=r"survey_pad must have length 2 \* dims in model, but got 2.",
    ):
        deepwave.common.set_survey_pad([1, 2], 2)
    with pytest.raises(
        RuntimeError,
        match=r"survey_pad must have length 2 \* dims in model, but got 3.",
    ):
        deepwave.common.set_survey_pad([1, 2, 3], 2)
    with pytest.raises(
        RuntimeError,
        match=r"survey_pad must have length 2 \* dims in model, but got 5.",
    ):
        deepwave.common.set_survey_pad([1, 2, 3, 4, 5], 2)
    with pytest.raises(
        RuntimeError,
        match=r"survey_pad must have length 2 \* dims in model, but got 6.",
    ):
        deepwave.common.set_survey_pad([1, 2, 3, 4, 5, 6], 2)
    with pytest.raises(
        RuntimeError,
        match="survey_pad entries must be None or non-negative ints.",
    ):
        deepwave.common.set_survey_pad([-1, 2, 3, 4], 2)
    with pytest.raises(
        RuntimeError,
        match="survey_pad entries must be None or non-negative ints.",
    ):
        deepwave.common.set_survey_pad([1, 2, 3, -4], 2)
    with pytest.raises(RuntimeError, match="survey_pad must be non-negative."):
        deepwave.common.set_survey_pad(-1, 2)
    with pytest.raises(RuntimeError, match="ndim must be positive."):
        deepwave.common.set_survey_pad(1, 0)
    with pytest.raises(RuntimeError, match="ndim must be positive."):
        deepwave.common.set_survey_pad(1, -1)
    with pytest.raises(
        RuntimeError,
        match=re.escape("survey_pad entries must be None or non-negative ints."),
    ):
        deepwave.common.set_survey_pad([1, "a", 3, 4], 2)


def test_check_locations_are_within_model():
    model_shape = [2, 3]
    locations = [None, None]
    deepwave.common.check_locations_are_within_model(model_shape, locations)
    locations = [None, torch.ones(2, 3, 2)]
    deepwave.common.check_locations_are_within_model(model_shape, locations)
    locations = [None, torch.ones(2, 3, 2)]
    deepwave.common.check_locations_are_within_model(model_shape, locations)
    locations = [None, torch.ones(2, 3, 2)]
    locations[1][1, 1, 1] = -1
    with pytest.raises(RuntimeError):
        deepwave.common.check_locations_are_within_model(model_shape, locations)
    locations = [None, torch.ones(2, 3, 2)]
    locations[1][1, 1, 1] = 3
    with pytest.raises(RuntimeError):
        deepwave.common.check_locations_are_within_model(model_shape, locations)


def test_check_locations_are_within_model_empty_model_shape():
    model_shape = []
    locations = [torch.zeros(1, 1, 0)]
    with pytest.raises(RuntimeError, match="model_shape must not be empty."):
        deepwave.common.check_locations_are_within_model(model_shape, locations)


def test_check_locations_are_within_model_non_positive_model_shape():
    model_shape = [2, 0]
    locations = [torch.zeros(1, 1, 2)]
    with pytest.raises(RuntimeError, match="model_shape elements must be positive."):
        deepwave.common.check_locations_are_within_model(model_shape, locations)


def test_check_locations_are_within_model_negative_model_shape():
    model_shape = [2, -1]
    locations = [torch.zeros(1, 1, 2)]
    with pytest.raises(RuntimeError, match="model_shape elements must be positive."):
        deepwave.common.check_locations_are_within_model(model_shape, locations)


def test_get_extents_from_locations():
    model_shape = [8, 9]
    locations = [None, None]
    survey_pad = None
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(0, 8), (0, 9)]
    survey_pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(0, 8), (0, 9)]
    locations = [None, 3 * torch.ones(2, 3, 2)]
    survey_pad = None
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(0, 8), (0, 9)]
    survey_pad = [None, 2, None, 4]
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(0, 6), (0, 8)]
    survey_pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(2, 6), (0, 8)]
    locations = [3 * torch.ones(3, 4, 2), 3 * torch.ones(2, 3, 2)]
    locations[0][1, 1, 0] = 2
    locations[1][1, 2, 1] = 8
    survey_pad = [None, 2, None, 4]
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(0, 6), (0, 9)]
    survey_pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_locations(
        model_shape,
        locations,
        survey_pad,
    ) == [(1, 6), (0, 9)]


def test_get_extents_from_locations_empty_model_shape():
    model_shape = []
    locations = [None]
    survey_pad = None
    with pytest.raises(RuntimeError, match="model_shape must not be empty."):
        deepwave.common.get_survey_extents_from_locations(
            model_shape,
            locations,
            survey_pad,
        )


def test_get_extents_from_locations_non_positive_model_shape():
    model_shape = [8, 0]
    locations = [None]
    survey_pad = None
    with pytest.raises(RuntimeError, match="model_shape elements must be positive."):
        deepwave.common.get_survey_extents_from_locations(
            model_shape,
            locations,
            survey_pad,
        )


def test_get_extents_from_locations_negative_model_shape():
    model_shape = [8, -1]
    locations = [None]
    survey_pad = None
    with pytest.raises(RuntimeError, match="model_shape elements must be positive."):
        deepwave.common.get_survey_extents_from_locations(
            model_shape,
            locations,
            survey_pad,
        )


def test_get_extents_from_wavefields():
    wavefields = [None, torch.zeros(2, 8, 9)]
    origin = None
    pml_width = [0, 0, 0, 0]
    assert deepwave.common.get_survey_extents_from_wavefields(
        wavefields,
        origin,
        pml_width,
    ) == [(0, 8), (0, 9)]
    pml_width = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_wavefields(
        wavefields,
        origin,
        pml_width,
    ) == [(0, 5), (0, 2)]
    origin = [1, 2]
    assert deepwave.common.get_survey_extents_from_wavefields(
        wavefields,
        origin,
        pml_width,
    ) == [(1, 6), (2, 4)]
    origin = [1, -2]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )
    origin = [1, 2, 3]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )
    origin = [1, None]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )
    origin = [1, 2]
    pml_width = [1, 2]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )
    pml_width = [1, -2, 1, 1]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )
    pml_width = None
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )


def test_get_extents_from_wavefields_empty_wavefields():
    wavefields = []
    origin = [1, 2]
    pml_width = [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="At least one wavefield must be non-None."):
        deepwave.common.get_survey_extents_from_wavefields(
            wavefields,
            origin,
            pml_width,
        )


def test_extract_survey():
    nx = (8, 9)
    models = [
        torch.arange(nx[0] * nx[1]).reshape(nx).float(),
        123 + torch.arange(nx[0] * nx[1]).reshape(nx).float(),
    ]
    source_locations = [3 * torch.ones(3, 4, 2).long()]
    receiver_locations = [3 * torch.ones(3, 3, 2).long()]
    source_locations[0][..., 1] = 3 + torch.arange(4).reshape(1, -1).repeat(3, 1)
    receiver_locations[0][..., 1] = 3 + torch.arange(3).reshape(1, -1).repeat(3, 1)
    source_locations[0][1, 1, 0] = 2
    receiver_locations[0][1, 2, 1] = 8
    survey_pad = [1, 2, 3, 4]
    wavefields = [None, None]
    origin = None
    fd_pad = [1, 1, 1, 1]
    pml_width = [3, 2, 1, 0]
    model_pad_modes = ["replicate"] * 2
    n_batch = 3
    n_dims = 2
    device = torch.device("cpu")
    dtype = torch.float32
    out = deepwave.common.extract_survey(
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
    # locations cover ranges y in [2, 3], x in [3, 8]
    # with survey_pad, extent should be y in [1, 5], x in [0, 12]
    # but since the original model extent is only [0, 8] in x,
    # the extent should be y in [1, 5], x in [0, 8]. The models are
    # then padded with fd_pad + pml_width
    assert torch.allclose(
        out[0][0],
        torch.nn.functional.pad(
            models[0][1:6, :].unsqueeze(0),
            (2, 1, 4, 3),
            mode="replicate",
        ),
    )
    assert torch.allclose(
        out[0][1],
        torch.nn.functional.pad(
            models[1][1:6, :].unsqueeze(0),
            (2, 1, 4, 3),
            mode="replicate",
        ),
    )
    # The beginning of the extent in each dimension (1 for y, 0 for x)
    # will be subtracted from the locations, and the padding for the
    # beginning of each dimension will be added (4 for y, 2 for x).
    # The locations will then be converted to 1D indices. The nx used
    # to convert to 1D will be the length of the extent in x (9) plus the padding (2 + 1),
    # so 12.
    locations_2d = (
        source_locations[0] - torch.Tensor([1, 0]).long() + torch.Tensor([4, 2]).long()
    )
    assert torch.allclose(out[1][0], locations_2d[..., 0] * 12 + locations_2d[..., 1])
    locations_2d = (
        receiver_locations[0]
        - torch.Tensor([1, 0]).long()
        + torch.Tensor([4, 2]).long()
    )
    assert torch.allclose(out[2][0], locations_2d[..., 0] * 12 + locations_2d[..., 1])

    wavefields = [torch.zeros(3, 4 + 5, 6 + 1)]
    origin = [2, 3]
    # The extent from the origin and wavefields should be y in [2, 2+9-3-2=6), x in [3, 3+7-1=9),
    # but from the locations and survey_pad it should be y in [1, 5], x in [0, 12]. This
    # sort of conflict is why specifying both survey_pad and origin is not allowed and
    # should raise an error.
    with pytest.raises(RuntimeError):
        out = deepwave.common.extract_survey(
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
    survey_pad = None
    # As survey_pad is no longer specified, it should now run.
    out = deepwave.common.extract_survey(
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
    assert torch.allclose(
        out[0][0],
        torch.nn.functional.pad(
            models[0][2:6, 3:9].unsqueeze(0),
            (2, 1, 4, 3),
            mode="replicate",
        ),
    )
    assert torch.allclose(
        out[0][1],
        torch.nn.functional.pad(
            models[1][2:6, 3:9].unsqueeze(0),
            (2, 1, 4, 3),
            mode="replicate",
        ),
    )
    # The nx for the 1D index conversion should be (given x range from 3 to 8, padding of 3) 9.
    locations_2d = (
        source_locations[0] - torch.Tensor([2, 3]).long() + torch.Tensor([4, 2]).long()
    )
    assert torch.allclose(out[1][0], locations_2d[..., 0] * 9 + locations_2d[..., 1])
    locations_2d = (
        receiver_locations[0]
        - torch.Tensor([2, 3]).long()
        + torch.Tensor([4, 2]).long()
    )
    assert torch.allclose(out[2][0], locations_2d[..., 0] * 9 + locations_2d[..., 1])


def test_extract_survey_zero_n_batch():
    nx = (8, 9)
    models = [
        torch.arange(nx[0] * nx[1]).reshape(nx).float(),
        123 + torch.arange(nx[0] * nx[1]).reshape(nx).float(),
    ]
    source_locations = [3 * torch.ones(0, 4, 2).long()]
    receiver_locations = [3 * torch.ones(0, 3, 2).long()]
    survey_pad = None
    wavefields = [None, None]
    origin = None
    fd_pad = [1, 1, 1, 1]
    pml_width = [3, 2, 1, 0]
    model_pad_modes = ["replicate"] * 2
    n_batch = 0
    n_dims = 2
    device = torch.device("cpu")
    dtype = torch.float32
    out = deepwave.common.extract_survey(
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
    assert out[0][0].shape[0] == 0
    assert out[0][1].shape[0] == 0
    assert out[1][0].shape[0] == 0
    assert out[2][0].shape[0] == 0
    assert out[3][0].shape[0] == 0


def test_check_extents_match_wavefields_shape():
    survey_extents = [(2, 5), (6, 9)]
    pad = [0, 1, 2, 3]
    # Wavefields are None, so all extents match
    wavefields = [None, None]
    deepwave.common.check_extents_match_wavefields_shape(
        survey_extents,
        wavefields,
        pad,
    )
    # Spatial size of wavefield is (4, 8), with padding of (1, 5),
    # so the size of the extents should be (4-1, 8-5) = (3, 3),
    # which it is.
    wavefields = [None, torch.zeros(2, 4, 8)]
    deepwave.common.check_extents_match_wavefields_shape(
        survey_extents,
        wavefields,
        pad,
    )
    # When the wavefield is smaller, it will not match the extents.
    wavefields = [None, torch.zeros(2, 3, 8)]
    with pytest.raises(RuntimeError):
        deepwave.common.check_extents_match_wavefields_shape(
            survey_extents,
            wavefields,
            pad,
        )
    # Similarly, when it is bigger, it will not match.
    wavefields = [None, torch.zeros(2, 4, 9)]
    with pytest.raises(RuntimeError):
        deepwave.common.check_extents_match_wavefields_shape(
            survey_extents,
            wavefields,
            pad,
        )


def test_check_extents_match_wavefields_shape_incorrect_extents_length():
    survey_extents = [(2, 5)]  # Should be 2 elements for 2D
    pad = [0, 1, 2, 3]
    wavefields = [None, torch.zeros(2, 4, 8)]
    with pytest.raises(AssertionError):
        deepwave.common.check_extents_match_wavefields_shape(
            survey_extents,
            wavefields,
            pad,
        )


def test_check_extents_match_wavefields_shape_incorrect_pad_length():
    survey_extents = [(2, 5), (6, 9)]
    pad = [0, 1, 2]  # Should be 4 elements for 2D
    wavefields = [None, torch.zeros(2, 4, 8)]
    with pytest.raises(AssertionError):
        deepwave.common.check_extents_match_wavefields_shape(
            survey_extents,
            wavefields,
            pad,
        )


def test_check_extents_match_wavefields_shape_empty_wavefields():
    survey_extents = [(2, 5), (6, 9)]
    pad = [0, 1, 2, 3]
    wavefields = []
    deepwave.common.check_extents_match_wavefields_shape(
        survey_extents,
        wavefields,
        pad,
    )

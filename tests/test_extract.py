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
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([1], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([1, 2], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([1, 2, 3], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([1, 2, 3, 4, 5], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([1, 2, 3, 4, 5, 6], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([-1, 2, 3, 4], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad([1, 2, 3, -4], 2)
    with pytest.raises(RuntimeError):
        deepwave.common.set_survey_pad(-1, 2)


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
        deepwave.common.check_locations_are_within_model(model_shape,
                                                         locations)
    locations = [None, torch.ones(2, 3, 2)]
    locations[1][1, 1, 1] = 3
    with pytest.raises(RuntimeError):
        deepwave.common.check_locations_are_within_model(model_shape,
                                                         locations)


def test_get_extents_from_locations():
    model_shape = [8, 9]
    locations = [None, None]
    survey_pad = None
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [0, 8, 0, 9]
    survey_pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [0, 8, 0, 9]
    locations = [None, 3*torch.ones(2, 3, 2)]
    survey_pad = None
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [0, 8, 0, 9]
    survey_pad = [None, 2, None, 4]
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [0, 6, 0, 8]
    survey_pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [2, 6, 0, 8]
    locations = [3*torch.ones(3, 4, 2), 3*torch.ones(2, 3, 2)]
    locations[0][1, 1, 0] = 2
    locations[1][1, 2, 1] = 8
    survey_pad = [None, 2, None, 4]
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [0, 6, 0, 9]
    survey_pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_locations(model_shape,
                                                             locations,
                                                             survey_pad) == \
        [1, 6, 0, 9]


def test_get_extents_from_wavefields():
    wavefields = [None, torch.zeros(2, 8, 9)]
    origin = None
    pad = [0, 0, 0, 0]
    assert deepwave.common.get_survey_extents_from_wavefields(wavefields,
                                                              origin, pad) == \
        [0, 8, 0, 9]
    pad = [1, 2, 3, 4]
    assert deepwave.common.get_survey_extents_from_wavefields(wavefields,
                                                              origin, pad) == \
        [0, 5, 0, 2]
    origin = [1, 2]
    assert deepwave.common.get_survey_extents_from_wavefields(wavefields,
                                                              origin, pad) == \
        [1, 6, 2, 4]
    origin = [1, -2]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(wavefields, origin,
                                                           pad)
    origin = [1, 2, 3]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(wavefields, origin,
                                                           pad)
    origin = [1, None]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(wavefields, origin,
                                                           pad)
    origin = [1, 2]
    pad = [1, 2]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(wavefields, origin,
                                                           pad)
    pad = [1, -2]
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(wavefields, origin,
                                                           pad)
    pad = None
    with pytest.raises(RuntimeError):
        deepwave.common.get_survey_extents_from_wavefields(wavefields, origin,
                                                           pad)


def test_extract_survey():
    nx = (8, 9)
    models = [torch.arange(nx[0]*nx[1]).reshape(nx),
              123 + torch.arange(nx[0]*nx[1]).reshape(nx)]
    locations = [3*torch.ones(3, 4, 2).long(),
                 3*torch.ones(2, 3, 2).long()]
    locations[0][1, 1, 0] = 2
    locations[1][1, 2, 1] = 8
    survey_pad = [1, 2, 3, 4]
    wavefields = [None, None]
    origin = None
    pad = [4, 3, 2, 1]
    out = deepwave.common.extract_survey(models, locations, survey_pad,
                                         wavefields, origin, pad)
    assert torch.allclose(out[0][0], models[0][1:6])
    assert torch.allclose(out[0][1], models[1][1:6])
    assert torch.allclose(out[1][0],
                          locations[0] - torch.Tensor([1, 0]).long())
    assert torch.allclose(out[1][1],
                          locations[1] - torch.Tensor([1, 0]).long())
    wavefields = [torch.zeros(2, 4+7, 6+3)]
    origin = [2, 3]
    out = deepwave.common.extract_survey(models, locations, survey_pad,
                                         wavefields, origin, pad)
    assert torch.allclose(out[0][0], models[0][2:6, 3:9])
    assert torch.allclose(out[0][1], models[1][2:6, 3:9])
    assert torch.allclose(out[1][0],
                          locations[0] - torch.Tensor([2, 3]).long())
    assert torch.allclose(out[1][1],
                          locations[1] - torch.Tensor([2, 3]).long())

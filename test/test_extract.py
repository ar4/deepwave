import torch
import deepwave.base.model
import deepwave.base.extract


def test_set_survey_pad():
    """Check conversion of float to list."""
    survey_pad = deepwave.base.extract._set_survey_pad(1.0, 2)
    assert survey_pad == [1.0, 1.0, 1.0, 1.0]


def test_survey_extents1():
    """Two shots, padded survey within model."""
    dx = [5.0, 5.0]
    nx = (5, 5)
    properties = {'a': torch.ones(nx),
                  'b': torch.zeros(nx)}
    model = deepwave.base.model.Model(properties, dx)
    survey_pad = [5.0] * 4
    num_shots = 2
    num_sources_per_shot = 2
    num_receivers_per_shot = 2
    # sources and receivers located in center of model
    source_locs = torch.ones(num_shots, num_sources_per_shot, 2) * 2 * 5.0
    receiver_locs = torch.ones(num_shots, num_receivers_per_shot, 2) * 2 * 5.0
    expected_extents = [slice(1, 4), slice(1, 4)]
    survey_extents = \
        deepwave.base.extract._get_survey_extents(model.shape, model.dx,
                                                  survey_pad,
                                                  source_locs, receiver_locs)
    assert survey_extents == expected_extents


def test_survey_pad2():
    """Two shots, padded survey exceeds model."""
    dx = torch.Tensor([5.0, 4.0, 3.0])
    nx = (5, 5, 5)
    properties = {'a': torch.ones(nx),
                  'b': torch.zeros(nx)}
    model = deepwave.base.model.Model(properties, dx)
    survey_pad = [5.0] * 6
    num_shots = 2
    num_sources_per_shot = 2
    num_receivers_per_shot = 2
    # sources and receivers located in center of model
    source_locs = torch.ones(num_shots, num_sources_per_shot, 3) * 2 * dx
    receiver_locs = torch.ones(num_shots, num_receivers_per_shot, 3) * 2 * dx
    # except for these ones that cause padding to go outside the model
    source_locs[0, 0, 1] = 1 * dx[1]
    receiver_locs[-1, -1, 2] = 4 * dx[2]
    expected_extents = [slice(None)] * 3
    expected_extents[0] = slice(1, 4)
    survey_extents = \
        deepwave.base.extract._get_survey_extents(model.shape, model.dx,
                                                  survey_pad, source_locs,
                                                  receiver_locs)
    assert survey_extents == expected_extents


def test_survey_pad3():
    """Two shots, uses varying list of pad values."""
    dx = torch.Tensor([5.0, 4.0, 3.0])
    nx = (5, 5, 5)
    properties = {'a': torch.ones(nx),
                  'b': torch.zeros(nx)}
    model = deepwave.base.model.Model(properties, dx)
    survey_pad = [2.0, 6.0, 0.0, 5.0, None, 1.0]
    num_shots = 2
    num_sources_per_shot = 2
    num_receivers_per_shot = 2
    # sources and receivers located in center of model
    source_locs = torch.ones(num_shots, num_sources_per_shot, 3) * 2 * dx
    receiver_locs = torch.ones(num_shots, num_receivers_per_shot, 3) * 2 * dx
    # except for these ones that cause padding to go outside the model
    source_locs[0, 0, 1] = 1 * dx[1]
    receiver_locs[-1, -1, 2] = 4 * dx[2]
    expected_extents = [slice(1, None), slice(1, None), slice(None)]
    survey_extents = \
        deepwave.base.extract._get_survey_extents(model.shape, model.dx,
                                                  survey_pad, source_locs,
                                                  receiver_locs)
    assert survey_extents == expected_extents

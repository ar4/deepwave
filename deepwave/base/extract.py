"""Extract Module to extract model around survey."""
import math
import torch


class Extract(torch.nn.Module):
    """Extract a portion of the model containing the sources and receivers.

    Args:
        survey_pad: Float or None or list of such of length
            2 * dimensionality of model, specifying padding around sources and
            receivers in units of dx.
    """

    def __init__(self, survey_pad):
        super(Extract, self).__init__()
        self.survey_pad = survey_pad

    def forward(self, model, source_locations, receiver_locations):
        """Perform extraction.

        Args:
            model: A Model object
            source_locations: A Tensor containing source locations in units
                of dx
            receiver_locations: A Tensor containing receiver locations in
                units of dx

        Returns:
            Extracted model as a Model object
        """
        survey_pad = _set_survey_pad(self.survey_pad, model.ndim)
        survey_extents = _get_survey_extents(
            model.shape,
            model.dx,
            survey_pad,
            source_locations,
            receiver_locations)
        extracted_model = _extract_model(model, survey_extents)
        return extracted_model


def _set_survey_pad(survey_pad, ndim):
    """Check survey_pad, and convert to a list if it is a scalar."""
    # Expand to list
    if isinstance(survey_pad, (float, type(None))):
        survey_pad = [survey_pad] * 2 * ndim

    # Check is non-negative or None
    if not all((pad is None) or (pad >= 0) for pad in survey_pad):
        raise RuntimeError('survey_pad must be non-negative or None, '
                           'but got {}'.format(survey_pad))

    # Check has correct size
    if len(survey_pad) != 2 * ndim:
        raise RuntimeError('survey_pad must have length 2 * dims in model, '
                           'but got {}'.format(len(survey_pad)))

    return survey_pad


def _get_survey_extents(model_shape, dx, survey_pad, source_locations,
                        receiver_locations):
    """Calculate the extents of the model to use for the survey.

    Args:
        model_shape: A tuple containing the shape of the full model
        dx: A Tensor containing the cell spacing in each dimension
        survey_pad: A list with two entries for
            each dimension, specifying the padding to add
            around the sources and receivers included in all of the
            shots being propagated. If None, the padding continues
            to the edge of the model
        source_locations: A Tensor containing source locations
        receiver_locations: A Tensor containing receiver locations

    Returns:
        A list of slices of the same length as the model shape,
            specifying the extents of the model that will be
            used for wave propagation
    """
    ndims = len(dx)
    extents = []
    for dim in range(ndims):
        left_pad = survey_pad[dim * 2]
        left_extent = \
            _get_survey_extents_one_side(left_pad, 'left',
                                         source_locations[..., dim],
                                         receiver_locations[..., dim],
                                         model_shape[dim],
                                         dx[dim].item())

        right_pad = survey_pad[dim * 2 + 1]
        right_extent = \
            _get_survey_extents_one_side(right_pad, 'right',
                                         source_locations[..., dim],
                                         receiver_locations[..., dim],
                                         model_shape[dim],
                                         dx[dim].item())

        extents.append(slice(left_extent, right_extent))

    return extents


def _get_survey_extents_one_side(pad, side, source_locations,
                                 receiver_locations, shape, dx):
    """Get the survey extent for the left or right side of one dimension.

    Args:
        pad: Positive float specifying padding for the side
        side: 'left' or 'right'
        source/receiver_locations: Tensor with coordinates for the current
            dimension
        shape: Int specifying length of full model in current dimension
        dx: Float specifying cell spacing in current dimension

    Returns:
        Min/max index as int or None
    """
    if pad is None:
        return None
    if side == 'left':
        pad = -pad
        op = torch.min
        nearest = math.floor
    else:
        pad = +pad
        op = torch.max
        nearest = math.ceil
    extreme_source = op(source_locations + pad)
    extreme_receiver = op(receiver_locations + pad)
    extreme_cell = nearest(op(extreme_source, extreme_receiver).item() / dx)
    if side == 'right':
        extreme_cell += 1
    if (extreme_cell <= 0) or (extreme_cell >= shape):
        extreme_cell = None
    return extreme_cell


def _extract_model(model, extents):
    """Extract the specified portion of the model.

    Args:
        model: A Model object
        extents: A list of slices specifying the portion of the model to
            extract

    Returns:
        A Model containing the desired portion of the model
    """

    return model[extents]

"""Base class for wave propagators."""
import torch
import deepwave.base.model
import deepwave.base.pad
import deepwave.base.extract


class Propagator(torch.nn.Module):
    """PyTorch Module to setup and call a wave propagator.

    Args:
        propfunc: Propagator Function to use
        model: A dictionary of [nz, (ny, (nx))] shape Float Tensors containing
            model properties, where nz, ny, and nz are the numbers of cells in
            the z, y, and x directions, respectively.
            For 1D propagation, each Tensor will therefore
            have shape [nz], while for 2D it will be [nz, ny], and
            [nz, ny, nx] in 3D.
        dx: A float or list of floats containing cell spacing in each
            dimension. In 3D, it contains [dz, dy, dx], while in 1D it is [dz].
        fd_width: Padding added around the edge of the model to make
            finite difference calculation code cleaner (no edge cases)
        pml_width: An int or list of ints specifying number of cells to use
            for the PML. This will be added to the beginning and end of each
            propagating dimension. If provided as a list, it should be of
            length 6, with each sequential group of two integer elements
            referring to the beginning and end PML width for a dimension.
            For dimensions less than 3, the elements for the remaining
            dimensions should be 0. Optional, default 10.
        survey_pad: A float or None, or list of such with 2 elements for each
            dimension, specifying the padding (in units of dx) to add.
            In each dimension, the survey (wave propagation) area for each
            batch of shots will be from the left-most source/receiver minus
            the left survey_pad, to the right-most source/receiver plus the
            right survey pad, over all shots in the batch, or to the edges of
            the model, whichever comes first. If a list, it specifies the
            left and right survey_pad in each dimension. If None, the survey
            area will continue to the edges of the model. If a float, that
            value will be used on the left and right of each dimension.
            Optional, default None.
    """

    def __init__(self, propfunc, model, dx, fd_width, pml_width=None,
                 survey_pad=None):
        super(Propagator, self).__init__()
        if not isinstance(model, dict):
            raise RuntimeError("Model must be a dict, e.g. {'vp': vp}, where "
                               "vp is a Tensor, but got a " + str(type(model)))

        self.propfunc = propfunc
        self.model = deepwave.base.model.Model(model, dx)
        self.extract = deepwave.base.extract.Extract(survey_pad)
        if pml_width is None:
            pml_width = 10
        self.pad = deepwave.base.pad.Pad(fd_width, pml_width)

    def forward(self, source_amplitudes, source_locations, receiver_locations,
                dt):
        """Forward modeling of sources to create synthetic data.

        Args:
            source_amplitudes: A [nt, num_shots, num_sources_per_shot] Float
                Tensor containing source amplitudes.
            source_locations: A [num_shots, num_sources_per_shot, num_dim]
                Float Tensor containing coordinates of the sources relative
                to the origin of the model.
            receiver_locations: A [num_shots, num_receivers_per_shot, num_dim]
                Float Tensor containing coordinates of the receivers relative
                to the origin of the model.
            dt: A float specifying the time interval between source samples.

        Returns:
            receiver_amplitudes: A [nt, num_shots, num_receivers_per_shot]
                Float Tensor containing synthetic receiver data.
        """
        # Check dt
        if not isinstance(dt, float):
            raise RuntimeError('dt must be a float, but has type {}'
                               .format(type(dt)))
        if dt <= 0.0:
            raise RuntimeError('dt must be > 0, but is {}'.format(dt))

        # Check same device as model
        if not (self.model.device == source_amplitudes.device ==
                source_locations.device == receiver_locations.device):
            raise RuntimeError('model, source amplitudes, source_locations, '
                               'and receiver_locations must all have the same '
                               'device, but got {} {} {} {}'
                               .format(self.model.device,
                                       source_amplitudes.device,
                                       source_locations.device,
                                       receiver_locations.device))

        # Check shapes
        if source_amplitudes.dim() != 3:
            raise RuntimeError('source_amplitude must have shape '
                               '[nt, num_shots, num_sources_per_shot]')

        if source_locations.dim() != 3:
            raise RuntimeError('source_locations must have shape '
                               '[num_shots, num_sources_per_shot, num_dims]')

        if receiver_locations.dim() != 3:
            raise RuntimeError('receiver_locations must have shape '
                               '[num_shots, num_receivers_per_shot, num_dims]')

        if not (source_amplitudes.shape[1] == source_locations.shape[0] ==
                receiver_locations.shape[0]):
            raise RuntimeError('Shape mismatch, expected '
                               'source_amplitudes.shape[1] '
                               '== source_locations.shape[0] '
                               '== receiver_locations.shape[0], but got '
                               '{} {} {}'.format(source_amplitudes.shape[1],
                                                 source_locations.shape[0],
                                                 receiver_locations.shape[0]))

        if not (source_amplitudes.shape[2] == source_locations.shape[1]):
            raise RuntimeError('Shape mismatch, expected '
                               'source_amplitudes.shape[2] '
                               '== source_locations.shape[1], but got '
                               '{} {}'.format(source_amplitudes.shape[2],
                                              source_locations.shape[1]))

        if not (self.model.ndim == source_locations.shape[2] ==
                receiver_locations.shape[2]):
            raise RuntimeError('Shape mismatch, expected '
                               'model num dims == source_locations.shape[2] '
                               '== receiver_locations.shape[2], but got '
                               '{} {} {}'.format(self.model.ndim,
                                                 source_locations.shape[2],
                                                 receiver_locations.shape[2]))

        # Check src/rec locations within model
        _check_locations_with_model(self.model, source_locations, 'source')
        _check_locations_with_model(self.model, receiver_locations, 'receiver')

        # Extract a region of the model around the sources/receivers
        model = self.extract(self.model, source_locations, receiver_locations)

        # Apply padding for the spatial finite difference and for the PML
        model = self.pad(model)

        # Run the propagator
        return self.propfunc.apply(source_amplitudes, source_locations,
                                   receiver_locations, dt, model,
                                   list(model.properties.keys()),
                                   *model.properties.values())


def _check_locations_with_model(model, locations, name):
    for dim in range(model.ndim):
        model_min = model.origin[dim].item()
        model_max = (model_min
                     + ((model.shape[dim] - 1).float() * model.dx[dim]).item())
        if locations[..., dim].min().item() < model_min:
            raise RuntimeError('{} locations not within model: {} < {}'
                               .format(name, locations[..., dim].min().item(),
                                       model_min))
        if locations[..., dim].max().item() > model_max:
            raise RuntimeError('{} locations not within model: {} > {}'
                               .format(name, locations[..., dim].max().item(),
                                       model_max))

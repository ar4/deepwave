"""Store a possibly padded model containing property Tensors."""
import torch


class Model(object):
    """A class for models.

    Args:
        properties: Dict containing padded model Tensors. All must be the
            same shape and type.
        dx: Float or list, tuple, or Tensor of length equal to model
            dimension, containing cell spacing.
        pad_width: Int or list of length 6 containing padding already added
            around the edge of the model. Optional, default 0.
        origin: Float or list of length equal to model dimension, containing
            spatial location of first unpadded cell in model. Optional,
            default 0.
        extra_info: Dict to store extra information. Optional, default None.
    """

    def __init__(self, properties, dx, pad_width=None, origin=None,
                 extra_info=None):
        _check_properties(properties)
        self.properties = properties
        self.device = _get_model_device(properties)
        self.ndim = _get_model_dim(properties)
        self.shape = _get_model_shape(properties)
        self.dx = _set_dx(dx, self.ndim)
        self.pad_width = _set_pad_width(pad_width, self.ndim)
        self.origin = _set_origin(origin, self.ndim)
        self.interior = _set_interior(self.shape, self.pad_width, self.ndim)
        self.dtype = _set_dtype(properties)
        self.extra_info = _set_extra_info(extra_info)

    def __getitem__(self, key):
        """Return the specified portion of the model.

        The key (specified portion) is applied to the interior (non-padded
        area) of the model, so model[1:] would remove the first element of
        the unpadded region, and then the result is repadded.
        """

        properties = {}
        for k, v in self.properties.items():
            properties[k] = v[self.interior][key]
        origin = self._new_origin(key)
        model_nopad = Model(properties, self.dx, origin=origin,
                            extra_info=self.extra_info)
        return model_nopad.pad(self.pad_width)

    def to(self, device):
        """Return a copy of the model moved to the specified device."""
        properties = {}
        for k, v in self.properties.items():
            properties[k] = v[k].to(device)
        return Model(properties, self.dx, self.pad_width, self.origin,
                     extra_info=self.extra_info)

    def _new_origin(self, key):
        """Return the new origin after extracting key.

        Args:
            key: A tuple of ints or slices indexing into model
        """
        offset = torch.tensor([0] * self.ndim).float()
        for dim, k in enumerate(key):
            if isinstance(k, slice):
                if k.start is None:
                    offset[dim] = 0
                else:
                    offset[dim] = k.start
            else:
                offset[dim] = k
        return self.origin + offset * self.dx

    def pad(self, *pad_widths_in):
        """Return a copy of the model with the specified pad_width.

        Padding is applied to the interior (non-padded) region, not to the
        existing padding. This is a no-op if the requested pad_width is
        the same as the current pad_width.

        pad_widths_in may be multiple pad_widths, which will be added
        """
        pad_widths = []
        for pad_width_in in pad_widths_in:
            pad_widths.append(_set_pad_width(pad_width_in, self.ndim))
        pad_width = sum(pad_widths)
        if (pad_width == self.pad_width).all():
            return self
        padded_properties = {}
        for key, value in self.properties.items():
            # Flip order of dimensions (as required by pad), but not order
            # within dimension, so [1, 0, ...] -> [..., 1, 0]
            padding = pad_width[:2 * self.ndim].reshape(-1, 2).flip(0)
            padding = padding.flatten().tolist()
            value_for_pad = value.reshape(1, 1, *self.shape[:self.ndim])
            value_for_pad = value_for_pad[[...] + self.interior]
            padded_properties[key] = torch.nn.functional.pad(
                value_for_pad,
                padding,
                mode='replicate')[0, 0]
        return Model(padded_properties, self.dx, pad_width=pad_width,
                     origin=self.origin, extra_info=self.extra_info)

    def add_properties(self, properties):
        """Store model property Tensors.

        Args:
                properties: A dict of Tensors. Each Tensor should be of the
                    same shape as the base model"""
        for key, value in properties.items():
            self.properties[key] = value
        _check_properties(self.properties)
        self.dtype = _set_dtype(self.properties)

    def allocate_wavefield(self, num_steps, num_shots):
        """Allocate a multiple of the shape of the model.

        Used for allocating wavefield Tensors.

        Args:
            num_steps: Int specifying number of time samples in source/receiver
                (and thus number of wavefields to be stored)
            num_shots: Int specifying number of shots to be propagated
                simultaneously

        Returns:
            Float Tensor of shape [num_steps, num_shots, model shape]
                on the PyTorch device and filled with zeros
        """

        return torch.zeros(num_steps, num_shots, *self.shape,
                           device=self.device, dtype=self.dtype)

    def get_locations(self, real_locations):
        """Convert spatial coordinates into model cell coordinates.

        E.g. [5.0, 10.0] -> [1, 2] if [dz, dy] == [5, 5]

        Args:
            real_locations: Tensor of coordinates in units of dx

        Returns:
            Tensor of coordinates in units of cells from origin of model
        """
        device = real_locations.device
        if real_locations.shape[-1] != self.ndim:
            raise RuntimeError('locations must have same dimension as model, '
                               'but {} != {}'
                               .format(real_locations.shape[-1].item(),
                                       self.ndim))
        return (((real_locations - self.origin.to(device)) /
                 self.dx.to(device)).long() +
                self.pad_width[:2 * self.ndim:2].to(device))


def _check_properties(model):
    """Check that the model property dictionary is appropriate."""
    shape = None
    tensor_type = None
    for key, value in model.items():
        # Check is a Tensor
        if not isinstance(value, torch.Tensor):
            raise TypeError('model properties must be Tensors, but {} '
                            'is of type {}'.format(key, type(value)))

        # Check is same type of Tensor
        if tensor_type is not None:
            if value.type() != tensor_type:
                raise RuntimeError('model properties must have same type, '
                                   'expected {}, but {} is of type {}'
                                   .format(tensor_type, key, value.type()))
        tensor_type = value.type()

        # Check is same size
        if shape is not None:
            if value.shape != shape:
                raise RuntimeError('model properties should have same shape, '
                                   'expected {}, but {} has shape {}'
                                   .format(shape, key, value.shape))
        shape = value.shape


def _get_model_device(model):
    """Return the device of a random model property."""
    return list(model.values())[0].device


def _get_model_dim(model):
    """Return the dimensionality of a random model property."""
    return list(model.values())[0].dim()


def _get_model_shape(model):
    """Return the shape of a random model property as a Tensor.

    Shape Tensor always contains 3 elements. When propagating in fewer
    than three dimensions, extra dimensions are 1.
    """
    ndim = _get_model_dim(model)
    shape = torch.ones(3, dtype=torch.long)
    property_shape = list(model.values())[0].shape
    shape[:ndim] = torch.tensor(property_shape)
    return shape


def _set_dx(dx, ndim):
    """Check dx is appropriate and convert to Tensor."""
    if isinstance(dx, int):
        dx = float(dx)
    dx = _set_tensor(dx, 'dx', ndim, float, torch.float)
    if (dx <= 0).any() or not torch.isfinite(dx).all():
        raise RuntimeError('All entries of dx must be positive, but got {}'
                           .format(dx))
    return dx


def _set_pad_width(pad_width, ndim):
    """Initialize, check, and convert pad_width to Tensor."""
    if pad_width is None:
        pad_width = 0
    pad_width = _set_tensor(pad_width, 'pad_width', 6, int, torch.long)
    pad_width[2*ndim:] = 0
    if (pad_width < 0).any():
        raise RuntimeError('All entries of pad_width must be non-negative, '
                           'but got {}'.format(pad_width))
    return pad_width


def _set_origin(origin, ndim):
    """Initialize, check, and convert origin to Tensor."""
    if origin is None:
        origin = 0.0
    if isinstance(origin, int) and origin == 0:
        origin = 0.0
    origin = _set_tensor(origin, 'origin', ndim, float, torch.float)
    if not torch.isfinite(origin).all():
        raise RuntimeError('All entries of origin must be finite, '
                           'but got {}'.format(origin))
    return origin


def _set_tensor(var, name, length, dtype, dtype_torch):
    """Set and check a Tensor.

    Args:
        var: list or scalar of type dtype
        name: String specifying the name to use in error messages
        length: List specifying the desired length
        dtype: The Python type that var must have
        dtype_torch: The PyTorch type that the Tensor must have
    """
    # Convert to list if a scalar
    if isinstance(var, dtype):
        var = [var] * length

    # Check is a list, tuple, or Tensor
    if not isinstance(var, (list, tuple, torch.Tensor)):
        raise TypeError('Expected {} to be a list, tuple, Tensor, or {}, '
                        'but got {}'.format(name, dtype, type(var)))

    # Check has correct length
    if len(var) != length:
        raise RuntimeError('{} must have length {}, but got {}'
                           .format(name, length, len(var)))

    ## If list/tuple, check every entry correct type
    #if isinstance(var, [list, tuple])
    #if not all(isinstance(entry, dtype) for entry in var):
    #    raise TypeError('Not every entry in {} is of type {}'
    #                    .format(name, dtype))

    # Convert to Tensor
    var = torch.tensor(var).to(dtype_torch)

    return var


def _set_interior(shape, pad_width, ndim):
    """Return a list of slices that will extract the interior region.

    Apply to a property tensor to extract its interior (non-pad) region.
    """
    return [slice(pad_width[2 * i].item(),
                  (shape[i] - pad_width[2 * i + 1]).item())
            for i in range(ndim)]


def _set_dtype(properties):
    """Return the dtype of a random model property."""
    return list(properties.values())[0].dtype


def _set_extra_info(extra_info):
    """Check input and return extra_info dictionary."""
    if extra_info is None:
        return {}
    if isinstance(extra_info, dict):
        return extra_info
    raise RuntimeError('extra_info must be None or a dict, but got {}'
                       .format(extra_info))

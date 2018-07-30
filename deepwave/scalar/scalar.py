"""PyTorch Module and Function for scalar wave propagator."""
import math
import torch
import numpy as np


class Propagator(torch.nn.Module):
    """PyTorch Module for scalar wave propagator.

    Args:
        model: A [1, nz, (ny, (nx))] shape Float Tensor containing wave speed,
            where nz, ny, and nz are the numbers of cells in the z, y, and x
            directions, respectively. For 1D propagation, it will therefore
            have shape [1, nz], while for 2D it will be [1, nz, ny], and
            [1, nz, ny, nx] in 3D.
        dx: A Float Tensor containing cell spacing in each dimension. In 3D, it
            contains [dz, dy, dx], while in 1D it is [dz].
        pml_width: Optional int or Tensor specifying number of cells to use
            for the PML. This will be added to the beginning and end of each
            propagating dimension. If provided as a Tensor, it should be of
            length 6, with each sequential group of two integer elements
            referring to the beginning and end PML width for a dimension.
            For dimensions less than 3, the elements for the remaining
            dimensions should be 0. Default 10.
        survey_pad: A float, or list with 2 elements for each dimension,
            specifying the padding (in units of dx) to add.
            In each dimension, the survey (wave propagation) area for each
            batch of shots will be from the left-most source/receiver minus
            the left survey_pad, to the right-most source/receiver plus the
            right survey pad, over all shots in the batch, or to the edges of
            the model, whichever comes first. If a Tensor, it specifies the
            left and right survey_pad in each dimension. If None, the survey
            area will continue to the edges of the model. If a float, that
            value will be used on the left and right of each dimension.
            Optional, default None.
    """

    def __init__(self, model, dx, pml_width=None, survey_pad=None):
        super(Propagator, self).__init__()
        self.model = model
        self.dx = dx.cpu()
        self.pml_width = pml_width
        self.survey_pad = survey_pad

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
        return PropagatorFunction.apply(self.model,
                                        source_amplitudes,
                                        source_locations,
                                        receiver_locations,
                                        self.dx,
                                        dt,
                                        self.pml_width,
                                        self.survey_pad)


class PropagatorFunction(torch.autograd.Function):
    """Forward modeling and backpropagation functions. Not called by users."""
    @staticmethod
    def forward(ctx,
                model_tensor,
                source_amplitudes,
                source_locations,
                receiver_locations,
                dx,
                dt,
                pml_width,
                survey_pad):
        """Forward modeling. Not called by users - see Propagator instead.

        Args:
            ctx: Context provided by PyTorch and used to store Tensors for
                backpropagation.
            See Propagator for descriptions of the other arguments.

        Returns:
            receiver amplitudes
        """

        pad_width = 2
        if pml_width is None:
            pml_width = 10
        device = model_tensor.device
        num_steps, num_shots, num_sources_per_shot = source_amplitudes.shape
        num_receivers_per_shot = receiver_locations.shape[1]
        survey_extents = _get_survey_extents(
            model_tensor.shape,
            dx,
            survey_pad,
            source_locations,
            receiver_locations)
        extracted_model_tensor = _extract_model(model_tensor, survey_extents)
        survey_extents_tensor = _slice_to_tensor(
            survey_extents, model_tensor.shape)
        model = Model(extracted_model_tensor, dx, pad_width, pml_width,
                      survey_extents_tensor)
        pml = Pml(model, num_shots)
        timestep = Timestep(model, dt)
        model.add_padded_properties({'vp2dt2': model.vp**2 *
                                     timestep.inner_dt**2})
        source_model_locations = model.get_locations(source_locations)
        receiver_model_locations = model.get_locations(receiver_locations)
        shape = torch.tensor(model.padded_shape)
        scalar_wrapper = _select_propagator(model.ndim, model_tensor.is_cuda)
        wavefield_save_strategy = \
            _set_wavefield_save_strategy(model_tensor.requires_grad, dt,
                                         timestep.inner_dt, scalar_wrapper)
        fd1, fd2 = _set_finite_diff_coeffs(model.ndim, dx, device)
        wavefield, saved_wavefields = \
            _allocate_wavefields(wavefield_save_strategy, scalar_wrapper,
                                 model, num_steps, num_shots)
        receiver_amplitudes = torch.zeros(
            num_steps, num_shots, num_receivers_per_shot, device=device)

        # Call compiled C code to do forward modeling
        scalar_wrapper.forward(
            wavefield.float().contiguous(),
            pml.aux.float().contiguous(),
            receiver_amplitudes.float().contiguous(),
            saved_wavefields.float().contiguous(),
            pml.sigma.float().contiguous(),
            model.padded_properties['vp2dt2'].float().contiguous(),
            fd1.float().contiguous(),
            fd2.float().contiguous(),
            source_amplitudes.float().contiguous(),
            source_model_locations.long().contiguous(),
            receiver_model_locations.long().contiguous(),
            shape.long().contiguous(),
            model.pml_width.long().contiguous(),
            num_steps,
            timestep.step_ratio,
            num_shots,
            num_sources_per_shot,
            num_receivers_per_shot,
            timestep.inner_dt,
            wavefield_save_strategy)

        if wavefield_save_strategy == scalar_wrapper.STRATEGY_INPLACE:
            # compensate for save beginning at step 2
            saved_wavefields = saved_wavefields[2:]

        # Allocate gradients that will be calculated during backpropagation
        model_gradient = _allocate_grad(model_tensor)
        if model_tensor.shape != extracted_model_tensor.shape:
            extracted_model_gradient = _allocate_grad(extracted_model_tensor)
        else:
            extracted_model_gradient = model_gradient
        source_gradient = _allocate_grad(source_amplitudes)

        ctx.save_for_backward(saved_wavefields, pml.aux, pml.sigma,
                              model.padded_properties['vp2dt2'],
                              model.properties['scaling'],
                              model.pml_width,
                              source_model_locations,
                              receiver_model_locations,
                              torch.tensor(timestep.step_ratio),
                              torch.tensor(timestep.inner_dt), fd1, fd2,
                              model_gradient, extracted_model_gradient,
                              survey_extents_tensor, source_gradient)

        return receiver_amplitudes

    @staticmethod
    def backward(ctx,
                 grad_receiver_amplitudes):
        """Performs backpropagation of gradient. Not directly called by users.

        Args:
            ctx: Context provided by PyTorch and used to load Tensors saved
                in forward modeling.
            grad_receiver_amplitudes: A [nt, num_shots, num_receivers_per_shot]
                Float Tensor containing gradients of a loss function with
                respect to the forward modeled receiver amplitudes

        Returns:
            model_gradient: A Float Tensor of the same shape as the input
                model, containing the gradient of the loss function with
                respect to the model.
            None for the other inputs to forward modeling.
        """

        (adjoint_wavefield, aux, sigma, vp2dt2, scaling,
         pml_width, source_model_locations, receiver_model_locations,
         step_ratio, inner_dt, fd1, fd2,
         model_gradient, extracted_model_gradient, survey_extents_tensor,
         source_gradient) = ctx.saved_variables
        step_ratio = step_ratio.item()
        inner_dt = inner_dt.item()
        survey_extents = _tensor_to_slice(survey_extents_tensor)

        ndim = receiver_model_locations.shape[-1]
        scalar_wrapper = _select_propagator(ndim, vp2dt2.is_cuda)

        num_steps, num_shots, num_receivers_per_shot = \
            grad_receiver_amplitudes.shape
        num_sources_per_shot = source_model_locations.shape[1]
        shape = torch.tensor(vp2dt2.shape)

        aux.fill_(0)
        wavefield = torch.zeros_like(aux[:2])

        # Call compiled C code to do backpropagation
        scalar_wrapper.backward(
            wavefield.float().contiguous(),
            aux.float().contiguous(),
            extracted_model_gradient.float().contiguous(),
            source_gradient.float().contiguous(),
            adjoint_wavefield.float().contiguous(),
            scaling.float().contiguous(),
            sigma.float().contiguous(),
            vp2dt2.float().contiguous(),
            fd1.float().contiguous(),
            fd2.float().contiguous(),
            grad_receiver_amplitudes.float().contiguous(),
            source_model_locations.long().contiguous(),
            receiver_model_locations.long().contiguous(),
            shape.long().contiguous(),
            pml_width.long().contiguous(),
            num_steps,
            step_ratio,
            num_shots,
            num_sources_per_shot,
            num_receivers_per_shot,
            inner_dt)

        _insert_model_gradient(extracted_model_gradient, survey_extents,
                               model_gradient)

        return (model_gradient, source_gradient, None, None, None, None, None,
                None)


def _select_propagator(ndim, cuda):
    """Returns the appropriate propagator based on the number of dimensions.

    Args:
        ndim: Int specifying number of dimensions
        cuda: Bool specifying whether will be running on GPU

    Returns:
        scalar_wrapper: module wrapping the compiled propagators
    """
    if cuda:
        if ndim == 1:
            import scalar1d_gpu_iso_4 as scalar_wrapper
        elif ndim == 2:
            import scalar2d_gpu_iso_4 as scalar_wrapper
        elif ndim == 3:
            import scalar3d_gpu_iso_4 as scalar_wrapper
        else:
            raise ValueError('unsupported number of dimensions')
    else:
        if ndim == 1:
            import scalar1d_cpu_iso_4 as scalar_wrapper
        elif ndim == 2:
            import scalar2d_cpu_iso_4 as scalar_wrapper
        elif ndim == 3:
            import scalar3d_cpu_iso_4 as scalar_wrapper
        else:
            raise ValueError('unsupported number of dimensions')

    return scalar_wrapper


def _set_wavefield_save_strategy(requires_grad, dt, inner_dt, scalar_wrapper):
    """Decides which of the source wavefield saving strategies to use.

    The source wavefield must be saved for backpropagation if model gradients
    required. The C code provides multiple ways of doing this, which are
    applicable in different situations.

    Args:
        requires_grad: Boolean specifying whether model gradients are required
        dt: The time interval between source samples
        inner_dt: The time interval between time steps of the wave propagator
        scalar_wrapper: The object that contains enum values for the strategies

    Returns:
        An enum value specifying which strategy to use
    """
    if requires_grad:
        if inner_dt == dt:
            wavefield_save_strategy = scalar_wrapper.STRATEGY_INPLACE
        else:
            wavefield_save_strategy = scalar_wrapper.STRATEGY_COPY
    else:
        wavefield_save_strategy = scalar_wrapper.STRATEGY_NONE

    return wavefield_save_strategy


def _set_finite_diff_coeffs(ndim, dx, device):
    """Calculates coefficients for finite difference derivatives.

    Currently only supports 4th order accurate derivatives.

    Args:
        ndim: Int specifying number of dimensions (1, 2, or 3)
        dx: Float Tensor containing cell spacing in each dimension
        device: PyTorch device to create coefficient Tensors on

    Returns:
        Float Tensors containing the coefficients for 1st and 2nd
            derivatives.
        fd1: Contains 2 coefficients for each dimension
        fd2: Contains 1 coefficient for the central element, followed by
            2 coefficients for each dimension
    """

    fd1 = torch.zeros(ndim, 2, device=device)
    fd2 = torch.zeros(ndim * 2 + 1, device=device)
    dx = dx.to(device)
    for dim in range(ndim):
        fd1[dim] = torch.tensor([8 / 12, -1 / 12], device=device) / dx[dim]
        fd2[0] += -5 / 2 / dx[dim]**2
        fd2[1 + dim * 2: 1 + (dim + 1) * 2] = \
            torch.tensor([4 / 3, -1 / 12], device=device) / dx[dim]**2

    return fd1, fd2


def _allocate_wavefields(wavefield_save_strategy, scalar_wrapper, model,
                         num_steps, num_shots):
    """Allocate wavefield Tensors.

    These will be used for propagation and to store wavefields for
    backpropagation.

    Args:
        wavefield_save_strategy: Enum specifying which strategy to use to
            save wavefields for backpropagation
        scalar_wrapper: An object containing the enum values
        model: A Model object that contains a method to allocate Tensors of
            the appropriate size
        num_steps: An int specifying the number of time samples in the input
            source
        num_shots: An int specifying the number of shots that will be
            propagated simultaneously

    Returns:
        wavefield: A Tensor that will be used by the propagator
        saved_wavefields: A Tensor that will store wavefields for
            backpropagation
    """

    if wavefield_save_strategy == scalar_wrapper.STRATEGY_NONE:
        wavefield = model.allocate_wavefield(2, num_shots)
        saved_wavefields = wavefield
    elif wavefield_save_strategy == scalar_wrapper.STRATEGY_INPLACE:
        wavefield = model.allocate_wavefield(num_steps + 2, num_shots)
        saved_wavefields = wavefield
    elif wavefield_save_strategy == scalar_wrapper.STRATEGY_COPY:
        wavefield = model.allocate_wavefield(2, num_shots)
        saved_wavefields = model.allocate_wavefield(num_steps, num_shots)

    return wavefield, saved_wavefields


def _allocate_grad(tensor):
    """Allocate a Tensor to store a gradient.

    If the provided Tensor requires a gradient, then a Tensor of the
    appropriate size will be allocated to store it. If it does not,
    then an empty Tensor will be created (so that it can be passed to
    save_for_backward either way without causing an error).

    Args:
        tensor: A Tensor that may or may not require a gradient

    Returns:
        Either a Tensor to store the gradient, or an empty Tensor.
    """

    if tensor.requires_grad:
        grad = torch.zeros_like(tensor)
    else:
        grad = torch.empty(0)

    return grad


def _get_survey_extents(model_shape, dx, survey_pad, source_locations,
                        receiver_locations):
    """Calculate the extents of the model to use for the survey.

    Args:
        model_shape: A tuple containing the shape of the full model
        dx: A Tensor containing the cell spacing in each dimension
        survey_pad: Either a float or a Tensor with two entries for
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

    ndims = len(model_shape)
    if survey_pad is None:
        survey_pad = [None] * 2 * (ndims - 1)
    if isinstance(survey_pad, (int, float)):
        survey_pad = [survey_pad] * 2 * (ndims - 1)
    if len(survey_pad) != 2 * (ndims - 1):
        raise ValueError('survey_pad has incorrect length: {} != {}'
                         .format(len(survey_pad), 2 * (ndims - 1)))
    extents = [slice(None)]  # property dim
    for dim in range(ndims - 1):  # ndims - 1 as no property dim
        left_pad = survey_pad[dim * 2]
        if left_pad is None:
            left_extent = None
        else:
            left_source = (source_locations[..., dim] - left_pad).min()
            left_receiver = (receiver_locations[..., dim] - left_pad).min()
            left_sourec_cell = \
                    math.floor((min(left_source, left_receiver).cpu() /
                                dx[dim]).item())
            left_extent = max(0, left_sourec_cell)
            if left_extent == 0:
                left_extent = None

        right_pad = survey_pad[dim * 2 + 1]
        if right_pad is None:
            right_extent = None
        else:
            right_source = (source_locations[..., dim] + right_pad).max()
            right_receiver = (receiver_locations[..., dim] + right_pad).max()
            right_sourec_cell = \
                    math.ceil((max(right_source, right_receiver).cpu() /
                               dx[dim]).item())
            right_extent = min(model_shape[dim + 1], right_sourec_cell) + 1
            if right_extent >= model_shape[dim + 1]:
                right_extent = None

        extents.append(slice(left_extent, right_extent))

    return extents


def _extract_model(model_tensor, extents):
    """Extract the specified portion of the model.

    Args:
        model_tensor: A Tensor containing the model
        extents: A list of slices specifying the portion of the model to
            extract

    Returns:
        A Tensor containing the desired portion of the model
    """

    return model_tensor[extents]


def _slice_to_tensor(extents, model_shape):
    """Convert a list of slices to a Tensor.

    This is needed to convert the list of extents into a Tensor so that it
    can be passed to save_for_backward. It is undone by _tensor_to_slice.

    Args:
        extents: A list of slices created by _get_survey_extents
        model_shape: The shape of the full model

    Returns:
        A Tensor containing 2 entries for each dimension of the model
        (the beginning and end of the slice).
    """
    ndims = len(extents)
    extents_tensor = torch.zeros(ndims, 2)
    for dim in range(ndims):
        if extents[dim].start is None:
            extents_tensor[dim, 0] = 0
        else:
            extents_tensor[dim, 0] = extents[dim].start
        if extents[dim].stop is None:
            extents_tensor[dim, 1] = model_shape[dim]
        else:
            extents_tensor[dim, 1] = extents[dim].stop

    return extents_tensor.long()


def _tensor_to_slice(extents_tensor):
    """Convert a Tensor to a list of slices.

    Needed because save_for_backward requires Tensors. This undoes
    _slice_to_tensor, and is called in backward to convert back to
    a list of slices.

    Args:
        extents_tensor: The Tensor created by _slice_to_tensor

    Returns:
        A list of slices
    """

    ndims = extents_tensor.shape[0]
    extents = []
    for dim in range(ndims):
        extents.append(slice(extents_tensor[dim, 0], extents_tensor[dim, 1]))

    return extents


def _insert_model_gradient(extracted_model_grad, extents, model_grad):
    """Insert the calculated gradient into the full model.

    The gradient might have been calculated on only a portion of the model
    (if survey_pad was not None), but the returned gradient is expected
    to be of the same size as the full model, so this inserts the
    calculated gradient into the full model (the areas not calculated
    will be zero).

    Args:
        extracted_model_grad: A Tensor containing the calculated model grad
        extents: A list of slices specifying the extents covered by
            the calculated model gradient
        model_grad: A Tensor of the same size as the full model, initialized
            to zero, which the model gradient will be inserted into.
    """
    if model_grad.numel() > 0:
        model_grad[extents] = extracted_model_grad


class Model(object):
    """A class for models.

    Args:
        pad_width: Padding added around the edge of the model to make
            finite difference calculation code cleaner (no edge cases)
        See Propagator for descriptions of the other arguments.
    """

    def __init__(self, model_tensor, dx, pad_width, pml_width, survey_extents):
        self.properties = {}
        self.padded_properties = {}
        self.dx = dx
        self.device = model_tensor.device
        self.ndim = len(model_tensor.shape[1:])
        # Shape Tensor always contains 3 elements. When propagating in fewer
        # than three dimensions, extra dimensions are 1.
        self.shape = torch.ones(3, dtype=torch.long)
        self.shape[:self.ndim] = torch.tensor(model_tensor.shape[1:])
        # pml_width and pad_width Tensors always contain 6 elements each:
        # padding at the beginning and end of each dimension. When propagating
        # in fewer than three dimensions, extra dimensions are 0.
        if isinstance(pml_width, torch.Tensor):
            self.pml_width = pml_width.cpu().long()
        else:
            self.pml_width = torch.zeros(6, dtype=torch.long)
            self.pml_width[:2 * self.ndim] = pml_width
        self.pad_width = torch.zeros(6, dtype=torch.long)
        self.pad_width[:2 * self.ndim] = pad_width
        self.total_pad = self.pad_width + self.pml_width
        self.padded_shape = [(self.shape[i] + self.total_pad[2 * i] +
                              self.total_pad[2 * i + 1]).item()
                             for i in range(3)]
        self.vp = model_tensor[0]
        self.max_vel = self.vp.max().item()
        self.add_properties({'scaling': 2 / self.vp**3})  # for backpropagation
        self.survey_extents = survey_extents

    def add_properties(self, properties):
        """Store an unpadded property."""
        for key, value in properties.items():
            self.properties[key] = value

    def add_padded_properties(self, properties):
        """Add padding to a property and store it."""
        for key, value in properties.items():
            padding = self.total_pad[:2 * self.ndim].tolist()[::-1]
            self.padded_properties[key] = \
                torch.nn.functional.pad(value.reshape(1, 1,
                                                      *self.shape[:self.ndim]),
                                        padding,
                                        mode='replicate')\
                .reshape(*self.padded_shape)

    def allocate_wavefield(self, num_steps, num_shots):
        """Allocate a multiple of the shape of the padded model.

        Used for allocating wavefield Tensors.

        Args:
            num_steps: Int specifying number of time samples in source/receiver
                (and thus number of wavefields to be stored)
            num_shots: Int specifying number of shots to be propagated
                simultaneously

        Returns:
            Float Tensor of shape [num_steps, num_shots, padded model shape]
                on the PyTorch device and filled with zeros
        """

        return torch.zeros(num_steps, num_shots, *self.padded_shape,
                           device=self.device)

    def get_locations(self, real_locations):
        """Convert spatial coordinates into model cell coordinates.

        E.g. [5.0, 10.0] -> [1, 2] if [dz, dy] == [5, 5]

        Args:
            real_locations: Tensor of coordinates in units of dx

        Returns:
            Tensor of coordinates in units of cells from origin of model
        """
        device = real_locations.device
        return ((real_locations / self.dx.to(device)).long() +
                self.total_pad[:2 * self.ndim:2].to(device) -
                self.survey_extents[1:, 0].view(-1).to(device))


class Pml(object):
    """Perfectly Matched Layer to absorb waves reaching model boundaries

    Args:
        model: Model object
        num_shots: Int specifying number of shots to be propagated
            simultaneously
    """

    def __init__(self, model, num_shots):

        def _set_sigma(model, profile, dim):
            """Create the sigma vector needed for the PML for one dimension.

            Makes a vector of the appropriate length, and copies the PML
            profile to the beginning and end, with zero elsewhere.

            Args:
                model: Model object
                profile: List of NumPy vectors of the PML profiles for the
                    beginning and end. Both are increasing, so the profile for
                    the beginning will be reversed (so it increases away from
                    the propagation domain)
                dim: Int specifying the dimension to use

            Returns:
                A Float Tensor of the same length as the padded model in the
                    specified dimension.
            """

            total_pad = model.total_pad[2 * dim:2 * dim + 2]
            pad_width = model.pad_width[2 * dim:2 * dim + 2]
            sigma = np.zeros(model.padded_shape[dim], np.float32)
            sigma[total_pad[0] - 1:pad_width[0] - 1:-1] = profile[0]
            sigma[-total_pad[1]:-pad_width[1]] = profile[1]
            sigma[:pad_width[0]] = sigma[pad_width[0]]
            sigma[-pad_width[1]:] = sigma[-pad_width[1] - 1]
            return torch.tensor(sigma).to(model.device)

        sigma = []
        for dim in range(model.ndim):
            pml_widths = model.pml_width[2 * dim:2 * dim + 2]
            profile = [((np.arange(w) / w)**2 *
                        3 * model.max_vel * np.log(1000) /
                        (2 * model.dx[dim].numpy() * w))
                       for w in pml_widths.numpy()]
            sigma.append(_set_sigma(model, profile, dim))
        self.sigma = torch.cat(sigma)

        # The number of auxiliary wavefields needed for the PML depends on
        # the dimension of the model
        if model.ndim == 1:
            self.aux = model.allocate_wavefield(2, num_shots)
        elif model.ndim == 2:
            self.aux = model.allocate_wavefield(2 * 2, num_shots)
        elif model.ndim == 3:
            self.aux = model.allocate_wavefield(2 * 4, num_shots)


class Timestep(object):
    """The timestep used during wave propagation.

    The dt specified by the user during forward modeling is the time interval
    between source samples, and is also used for the time interval between
    output receiver samples and between the wavefields saved for
    backpropagation. The wave propagator may need a smaller time step size for
    numerical stability, which will be called inner_dt. It depends on the
    cell size and the maximum wave speed. This inner_dt will be chosen so
    that dt is a multiple of it. The factor will be called step_ratio.

    Args:
        model: Model object
        dt: A float specifying the time interval between source samples
    """

    def __init__(self, model, dt):
        max_dt = 0.6 / np.linalg.norm(1 / model.dx.numpy()) / model.max_vel
        self.step_ratio = int(np.ceil(dt / max_dt))
        self.inner_dt = dt / self.step_ratio

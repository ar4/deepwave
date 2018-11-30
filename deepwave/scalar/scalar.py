"""PyTorch Module and Function for scalar wave propagator."""
import torch
import numpy as np
import scipy.signal
import deepwave.base.propagator

class Propagator(deepwave.base.propagator.Propagator):
    """PyTorch Module for scalar wave propagator.

    See deepwave.base.propagator.Propagator for description.
    """

    def __init__(self, model, dx, pml_width=None, survey_pad=None, vpmax=None):
        if list(model.keys()) != ['vp']:
            raise RuntimeError('Model must only contain vp, but contains {}'
                               .format(list(model.keys())))
        super(Propagator, self).__init__(PropagatorFunction, model, dx,
                                         fd_width=2,  # also in Pml
                                         pml_width=pml_width,
                                         survey_pad=survey_pad)
        self.model.extra_info['vpmax'] = vpmax
        if model['vp'].min() <= 0.0:
            raise RuntimeError('vp must be > 0, but min is {}'
                               .format(model['vp'].min()))


class PropagatorFunction(torch.autograd.Function):
    """Forward modeling and backpropagation functions. Not called by users."""
    @staticmethod
    def forward(ctx,
                source_amplitudes,
                source_locations,
                receiver_locations,
                dt, model, property_names, vp):
        """Forward modeling. Not called by users - see Propagator instead.

        Args:
            ctx: Context provided by PyTorch and used to store Tensors for
                backpropagation.
            property_names: The list ['vp']
            vp: P wave speed
            See Propagator for descriptions of the other arguments.

        Returns:
            receiver amplitudes
        """
        if property_names != ['vp']:
            raise RuntimeError('Model must only contain vp, but contains {}'
                               .format(property_names))
        if vp.min() <= 0.0:
            raise RuntimeError('vp must be > 0, but min is {}'
                               .format(vp.min()))
        device = model.device
        dtype = model.dtype
        num_steps, num_shots, num_sources_per_shot = source_amplitudes.shape
        num_receivers_per_shot = receiver_locations.shape[1]

        if model.extra_info['vpmax'] is None:
            max_vel = vp.max().item()
        else:
            max_vel = model.extra_info['vpmax']
        timestep = Timestep(dt, model.dx, max_vel)
        model.add_properties({'vp2dt2': vp**2 * timestep.inner_dt**2,
                              'scaling': 2 / vp**3})
        source_model_locations = model.get_locations(source_locations)
        receiver_model_locations = model.get_locations(receiver_locations)
        scalar_wrapper = _select_propagator(model.ndim, vp.dtype, vp.is_cuda)
        wavefield_save_strategy = \
            _set_wavefield_save_strategy(ctx.needs_input_grad[6], dt,
                                         timestep.inner_dt, scalar_wrapper)
        fd1, fd2 = _set_finite_diff_coeffs(model.ndim, model.dx, device, dtype)
        wavefield, saved_wavefields = \
            _allocate_wavefields(wavefield_save_strategy, scalar_wrapper,
                                 model, num_steps, num_shots)
        receiver_amplitudes = torch.zeros(
            num_steps, num_shots, num_receivers_per_shot, device=device,
            dtype=dtype)
        inner_dt = torch.tensor([timestep.inner_dt]).to(dtype)
        pml = Pml(model, num_shots, max_vel)
        source_amplitudes_resampled = \
                scipy.signal.resample(source_amplitudes.detach().cpu().numpy(),
                                      num_steps * timestep.step_ratio)
        source_amplitudes_resampled = \
                torch.tensor(source_amplitudes_resampled)\
                    .to(dtype).to(source_amplitudes.device)
        source_amplitudes_resampled.requires_grad = \
                source_amplitudes.requires_grad

        # Call compiled C code to do forward modeling
        scalar_wrapper.forward(
            wavefield.to(dtype).contiguous(),
            pml.aux.to(dtype).contiguous(),
            receiver_amplitudes.to(dtype).contiguous(),
            saved_wavefields.to(dtype).contiguous(),
            pml.sigma.to(dtype).contiguous(),
            model.properties['vp2dt2'].to(dtype).contiguous(),
            fd1.to(dtype).contiguous(),
            fd2.to(dtype).contiguous(),
            source_amplitudes_resampled.to(dtype).contiguous(),
            source_model_locations.long().contiguous(),
            receiver_model_locations.long().contiguous(),
            model.shape.contiguous(),
            pml.pml_width.long().contiguous(),
            inner_dt,
            num_steps,
            timestep.step_ratio,
            num_shots,
            num_sources_per_shot,
            num_receivers_per_shot,
            wavefield_save_strategy)

        # Allocate gradients that will be calculated during backpropagation
        model_gradient = _allocate_grad(vp, ctx.needs_input_grad[6])
        source_gradient = _allocate_grad(source_amplitudes,
                                         ctx.needs_input_grad[0])

        ctx.save_for_backward(saved_wavefields, pml.aux, pml.sigma,
                              model.properties['vp2dt2'],
                              model.properties['scaling'],
                              pml.pml_width,
                              source_model_locations,
                              receiver_model_locations,
                              torch.tensor(timestep.step_ratio),
                              inner_dt, fd1, fd2,
                              model_gradient, source_gradient)

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

        (saved_wavefields, aux, sigma, vp2dt2, scaling,
         pml_width, source_model_locations, receiver_model_locations,
         step_ratio, inner_dt, fd1, fd2,
         model_gradient, source_gradient) = ctx.saved_tensors
        step_ratio = step_ratio.item()

        dtype = vp2dt2.dtype
        ndim = receiver_model_locations.shape[-1]
        scalar_wrapper = _select_propagator(ndim, vp2dt2.dtype, vp2dt2.is_cuda)

        num_steps, num_shots, num_receivers_per_shot = \
            grad_receiver_amplitudes.shape
        num_sources_per_shot = source_model_locations.shape[1]
        shape = torch.ones(3).long()
        shape[:ndim] = torch.tensor(vp2dt2.shape)

        aux.fill_(0)
        model_gradient.fill_(0)
        source_gradient.fill_(0)
        wavefield = torch.zeros(3, *aux[0].shape, device=aux.device)

        grad_receiver_amplitudes_resampled = \
                scipy.signal.resample(grad_receiver_amplitudes
                                      .detach().cpu().numpy(),
                                      num_steps * step_ratio)
        grad_receiver_amplitudes_resampled = \
                torch.tensor(grad_receiver_amplitudes_resampled)\
                    .to(dtype).to(grad_receiver_amplitudes.device)
        grad_receiver_amplitudes_resampled.requires_grad = \
                grad_receiver_amplitudes.requires_grad

        # Ensure that gradient Tensors are of the right type and contiguous
        model_gradient = model_gradient.to(dtype).contiguous()
        source_gradient = source_gradient.to(dtype).contiguous()

        # Call compiled C code to do backpropagation
        scalar_wrapper.backward(
            wavefield.to(dtype).contiguous(),
            aux.to(dtype).contiguous(),
            model_gradient,
            source_gradient,
            saved_wavefields.to(dtype).contiguous(),
            scaling.to(dtype).contiguous(),
            sigma.to(dtype).contiguous(),
            vp2dt2.to(dtype).contiguous(),
            fd1.to(dtype).contiguous(),
            fd2.to(dtype).contiguous(),
            grad_receiver_amplitudes_resampled.to(dtype).contiguous(),
            source_model_locations.long().contiguous(),
            receiver_model_locations.long().contiguous(),
            shape.long().contiguous(),
            pml_width.long().contiguous(),
            inner_dt,
            num_steps,
            step_ratio,
            num_shots,
            num_sources_per_shot,
            num_receivers_per_shot)

        if not ctx.needs_input_grad[0]:
            source_gradient = None

        if not ctx.needs_input_grad[6]:
            model_gradient = None

        return (source_gradient, None, None, None, None, None, model_gradient)


def _select_propagator(ndim, dtype, cuda):
    """Returns the appropriate propagator based on the number of dimensions.

    Args:
        ndim: Int specifying number of dimensions
        dtype: Datatype, either torch.float or torch.double
        cuda: Bool specifying whether will be running on GPU

    Returns:
        scalar_wrapper: module wrapping the compiled propagators
    """
    if ndim not in [1, 2, 3]:
        raise RuntimeError('unsupported number of dimensions: {}'.format(ndim))
    if dtype not in [torch.float, torch.double]:
        raise RuntimeError('unsupported datatype: {}'.format(dtype))

    if cuda:
        if dtype == torch.float:
            if ndim == 1:
                import scalar1d_gpu_iso_4_float as scalar_wrapper
            elif ndim == 2:
                import scalar2d_gpu_iso_4_float as scalar_wrapper
            elif ndim == 3:
                import scalar3d_gpu_iso_4_float as scalar_wrapper
        elif dtype == torch.double:
            raise NotImplementedError('To enable double-precision GPU '
                                      'propagators (on supported hardware) '
                                      'add "double" to the list of dtypes in '
                                      'the CUDA portion of setup.py')
            #if ndim == 1:
            #    import scalar1d_gpu_iso_4_double as scalar_wrapper
            #elif ndim == 2:
            #    import scalar2d_gpu_iso_4_double as scalar_wrapper
            #elif ndim == 3:
            #    import scalar3d_gpu_iso_4_double as scalar_wrapper
    else:
        if dtype == torch.float:
            if ndim == 1:
                import scalar1d_cpu_iso_4_float as scalar_wrapper
            elif ndim == 2:
                import scalar2d_cpu_iso_4_float as scalar_wrapper
            elif ndim == 3:
                import scalar3d_cpu_iso_4_float as scalar_wrapper
        elif dtype == torch.double:
            if ndim == 1:
                import scalar1d_cpu_iso_4_double as scalar_wrapper
            elif ndim == 2:
                import scalar2d_cpu_iso_4_double as scalar_wrapper
            elif ndim == 3:
                import scalar3d_cpu_iso_4_double as scalar_wrapper

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
        wavefield_save_strategy = scalar_wrapper.STRATEGY_COPY
    else:
        wavefield_save_strategy = scalar_wrapper.STRATEGY_NONE

    return wavefield_save_strategy


def _set_finite_diff_coeffs(ndim, dx, device, dtype):
    """Calculates coefficients for finite difference derivatives.

    Currently only supports 4th order accurate derivatives.

    Args:
        ndim: Int specifying number of dimensions (1, 2, or 3)
        dx: Float Tensor containing cell spacing in each dimension
        device: PyTorch device to create coefficient Tensors on
        dtype: PyTorch datatype to use

    Returns:
        Float Tensors containing the coefficients for 1st and 2nd
            derivatives.
        fd1: Contains 2 coefficients for each dimension
        fd2: Contains 1 coefficient for the central element, followed by
            2 coefficients for each dimension
    """

    fd1 = torch.zeros(ndim, 2, device=device, dtype=dtype)
    fd2 = torch.zeros(ndim * 2 + 1, device=device, dtype=dtype)
    dx = dx.to(device).to(dtype)
    for dim in range(ndim):
        fd1[dim] = (torch.tensor([8 / 12, -1 / 12], device=device, dtype=dtype)
                    / dx[dim])
        fd2[0] += -5 / 2 / dx[dim]**2
        fd2[1 + dim * 2: 1 + (dim + 1) * 2] = \
            (torch.tensor([4 / 3, -1 / 12], device=device, dtype=dtype)
             / dx[dim]**2)

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
        wavefield = model.allocate_wavefield(3, num_shots)
        saved_wavefields = wavefield
    elif wavefield_save_strategy == scalar_wrapper.STRATEGY_COPY:
        wavefield = model.allocate_wavefield(3, num_shots)
        saved_wavefields = model.allocate_wavefield(num_steps, 3 * num_shots)

    return wavefield, saved_wavefields


def _allocate_grad(tensor, requires_grad):
    """Allocate a Tensor to store a gradient.

    If the provided Tensor requires a gradient, then a Tensor of the
    appropriate size will be allocated to store it. If it does not,
    then an empty Tensor will be created (so that it can be passed to
    save_for_backward either way without causing an error).

    Args:
        tensor: A Tensor that may or may not require a gradient
        requires_grad: Bool specifying whether tensor requires gradient

    Returns:
        Either a Tensor to store the gradient, or an empty Tensor.
    """

    if requires_grad:
        grad = torch.zeros_like(tensor)
    else:
        grad = torch.empty(0)

    return grad


class Pml(object):
    """Perfectly Matched Layer to absorb waves reaching model boundaries

    Args:
        model: Model object
        num_shots: Int specifying number of shots to be propagated
            simultaneously
    """

    def __init__(self, model, num_shots, max_vel):

        def _set_sigma(model, profile, dim, fd_pad):
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
                fd_pad: Int specifying spatial finite difference padding

            Returns:
                A Float Tensor of the same length as the padded model in the
                    specified dimension.
            """

            pad_width = model.pad_width[2 * dim:2 * dim + 2]
            sigma = np.zeros(model.shape[dim])
            sigma[pad_width[0] - 1:fd_pad - 1:-1] = profile[0]
            sigma[-pad_width[1]:-fd_pad] = profile[1]
            sigma[:fd_pad] = sigma[fd_pad]
            sigma[-fd_pad:] = sigma[-fd_pad - 1]
            return torch.tensor(sigma).to(model.dtype).to(model.device)

        fd_pad = 2
        sigma = []
        self.pml_width = model.pad_width - fd_pad
        for dim in range(model.ndim):
            pml_widths = self.pml_width[2 * dim:2 * dim + 2]
            profile = [((np.arange(w) / w)**2 *
                        3 * max_vel * np.log(1000) /
                        (2 * model.dx[dim].numpy() * w))
                       for w in pml_widths.numpy()]
            sigma.append(_set_sigma(model, profile, dim, fd_pad))
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

    def __init__(self, dt, dx, max_vel):
        max_dt = 0.6 / np.linalg.norm(1 / dx.numpy()) / max_vel
        self.step_ratio = int(np.ceil(dt / max_dt))
        self.inner_dt = dt / self.step_ratio

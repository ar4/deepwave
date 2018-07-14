"""PyTorch Module and Function for scalar wave propagator."""
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
        pml_width: Optional int specifying number of cells to use for the PML.
            This will be added to the beginning and end of each propagating
            dimension. Default 10.
    """

    def __init__(self, model, dx, pml_width=None):
        super(Propagator, self).__init__()
        self.model = model
        self.dx = dx
        self.pml_width = pml_width

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
                                        self.pml_width)


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
                pml_width):
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
        model = Model(model_tensor, dx, pad_width, pml_width)
        pml = Pml(model, num_shots)
        timestep = Timestep(model, dt)
        model.add_padded_properties({'vp2dt2': model.vp**2 *
                                     timestep.inner_dt**2})
        source_model_locations = model.get_locations(source_locations)
        receiver_model_locations = model.get_locations(receiver_locations)
        shape = torch.tensor(model.padded_shape)
        ffi, lib = _select_propagator(model.ndim)
        wavefield_save_strategy = \
            _set_wavefield_save_strategy(model_tensor.requires_grad, dt,
                                         timestep.inner_dt, lib)
        fd1, fd2 = _set_finite_diff_coeffs(model.ndim, dx, device)
        wavefield, saved_wavefields = \
            _allocate_wavefields(wavefield_save_strategy, lib, model,
                                 num_steps, num_shots)
        receiver_amplitudes = torch.zeros(
            num_steps, num_shots, num_receivers_per_shot, device=device)

        # Call compiled C code to do forward modeling
        lib.forward(
            ffi.cast("float *", wavefield.float().contiguous().data_ptr()),
            ffi.cast("float *", pml.aux.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     receiver_amplitudes.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     saved_wavefields.float().contiguous().data_ptr()),
            ffi.cast("float *", pml.sigma.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     model.padded_properties['vp2dt2'].float().contiguous()
                                                      .data_ptr()),
            ffi.cast("float *", fd1.float().contiguous().data_ptr()),
            ffi.cast("float *", fd2.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     source_amplitudes.float().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *",
                     source_model_locations.long().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *",
                     receiver_model_locations.long().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *", shape.long().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *",
                     model.pml_width.long().contiguous().data_ptr()),
            num_steps,
            timestep.step_ratio,
            num_shots,
            num_sources_per_shot,
            num_receivers_per_shot,
            timestep.inner_dt,
            wavefield_save_strategy)

        if wavefield_save_strategy == lib.STRATEGY_INPLACE:
            # compensate for save beginning at step 2
            saved_wavefields = saved_wavefields[2:]

        ctx.save_for_backward(saved_wavefields, pml.aux, pml.sigma,
                              model.padded_properties['vp2dt2'],
                              model.properties['scaling'],
                              model.pml_width,
                              receiver_model_locations,
                              torch.tensor(timestep.step_ratio),
                              torch.tensor(timestep.inner_dt), fd1, fd2)

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
         pml_width, receiver_model_locations,
         step_ratio, inner_dt, fd1, fd2) = ctx.saved_variables
        step_ratio = step_ratio.item()
        inner_dt = inner_dt.item()

        ndim = receiver_model_locations.shape[-1]
        ffi, lib = _select_propagator(ndim)

        num_steps, num_shots, num_sources_per_shot = \
            grad_receiver_amplitudes.shape
        shape = torch.tensor(vp2dt2.shape)

        aux.fill_(0)
        wavefield = torch.zeros_like(aux[:2])
        model_gradient = torch.zeros_like(scaling.view(1, *scaling.shape))

        # Call compiled C code to do backpropagation
        lib.backward(
            ffi.cast("float *", wavefield.float().contiguous().data_ptr()),
            ffi.cast("float *", aux.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     model_gradient.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     adjoint_wavefield.float().contiguous().data_ptr()),
            ffi.cast("float *", scaling.float().contiguous().data_ptr()),
            ffi.cast("float *", sigma.float().contiguous().data_ptr()),
            ffi.cast("float *", vp2dt2.float().contiguous().data_ptr()),
            ffi.cast("float *", fd1.float().contiguous().data_ptr()),
            ffi.cast("float *", fd2.float().contiguous().data_ptr()),
            ffi.cast("float *",
                     grad_receiver_amplitudes.float().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *",
                     receiver_model_locations.long().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *", shape.long().contiguous().data_ptr()),
            ffi.cast("ptrdiff_t *", pml_width.long().contiguous().data_ptr()),
            num_steps,
            step_ratio,
            num_shots,
            num_sources_per_shot,
            inner_dt)

        return model_gradient, None, None, None, None, None, None


def _select_propagator(ndim):
    """Returns the appropriate propagator based on the number of dimensions.

    Args:
        ndim: Int specifying number of dimensions

    Returns:
        ffi, lib: CFFI objects to interact with the compiled propagators
    """
    if ndim == 1:
        from deepwave.scalar1d import ffi, lib
    elif ndim == 2:
        from deepwave.scalar2d import ffi, lib
    elif ndim == 3:
        from deepwave.scalar3d import ffi, lib
    else:
        raise ValueError('unsupported number of dimensions')

    return ffi, lib


def _set_wavefield_save_strategy(requires_grad, dt, inner_dt, lib):
    """Decides which of the source wavefield saving strategies to use.

    The source wavefield must be saved for backpropagation if model gradients
    required. The C code provides multiple ways of doing this, which are
    applicable in different situations.

    Args:
        requires_grad: Boolean specifying whether model gradients are required
        dt: The time interval between source samples
        inner_dt: The time interval between time steps of the wave propagator
        lib: The CFFI object that contains enum values for the strategies

    Returns:
        An enum value specifying which strategy to use
    """
    if requires_grad:
        if inner_dt == dt:
            wavefield_save_strategy = lib.STRATEGY_INPLACE
        else:
            wavefield_save_strategy = lib.STRATEGY_COPY
    else:
        wavefield_save_strategy = lib.STRATEGY_NONE

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
    for dim in range(ndim):
        fd1[dim] = torch.tensor([8 / 12, -1 / 12], device=device) / dx[dim]
        fd2[0] += -5 / 2 / dx[dim]**2
        fd2[1 + dim * 2: 1 + (dim + 1) * 2] = \
            torch.tensor([4 / 3, -1 / 12], device=device) / dx[dim]**2

    return fd1, fd2


def _allocate_wavefields(wavefield_save_strategy, lib, model, num_steps,
                         num_shots):
    """Allocate wavefield Tensors.

    These will be used for propagation and to store wavefields for
    backpropagation.

    Args:
        wavefield_save_strategy: Enum specifying which strategy to use to
            save wavefields for backpropagation
        lib: CFFI object containing the enum values
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

    if wavefield_save_strategy == lib.STRATEGY_NONE:
        wavefield = model.allocate_wavefield(2, num_shots)
        saved_wavefields = wavefield
    elif wavefield_save_strategy == lib.STRATEGY_INPLACE:
        wavefield = model.allocate_wavefield(num_steps + 2, num_shots)
        saved_wavefields = wavefield
    elif wavefield_save_strategy == lib.STRATEGY_COPY:
        wavefield = model.allocate_wavefield(2, num_shots)
        saved_wavefields = model.allocate_wavefield(num_steps, num_shots)

    return wavefield, saved_wavefields


class Model(object):
    """A class for models.

    Args:
        pad_width: Padding added around the edge of the model to make
            finite difference calculation code cleaner (no edge cases)
        See Propagator for descriptions of the other arguments.
    """

    def __init__(self, model_tensor, dx, pad_width, pml_width):
        self.properties = {}
        self.padded_properties = {}
        self.dx = dx
        self.device = model_tensor.device
        self.ndim = len(model_tensor.shape[1:])
        # Shape Tensor always contains 3 elements. When propagating in fewer
        # than three dimensions, extra dimensions are 1.
        self.shape = torch.ones(3, device=self.device, dtype=torch.long)
        self.shape[:self.ndim] = torch.tensor(
            model_tensor.shape[1:]).to(self.device)
        # pml_width and pad_width Tensors always contain 6 elements each:
        # padding at the beginning and end of each dimension. When propagating
        # in fewer than three dimensions, extra dimensions are 0.
        self.pml_width = torch.zeros(6, device=self.device, dtype=torch.long)
        self.pml_width[:2 * self.ndim] = pml_width
        self.pad_width = torch.zeros(6, device=self.device, dtype=torch.long)
        self.pad_width[:2 * self.ndim] = pad_width
        self.total_pad = self.pad_width + self.pml_width
        self.padded_shape = [(self.shape[i] + self.total_pad[2 * i] +
                              self.total_pad[2 * i + 1]).item()
                             for i in range(3)]
        self.vp = model_tensor[0]
        self.max_vel = self.vp.max().item()
        self.add_properties({'scaling': 2 / self.vp**3})  # for backpropagation

    def add_properties(self, properties):
        """Store an unpadded property."""
        for key, value in properties.items():
            self.properties[key] = value

    def add_padded_properties(self, properties):
        """Add padding to a property and store it."""
        for key, value in properties.items():
            self.padded_properties[key] = \
                torch.nn.functional.pad(value.reshape(1, 1,
                                                      *self.shape[:self.ndim]),
                                        self.total_pad[:2 * self.ndim].tolist(),
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
        return ((real_locations / self.dx).long() +
                self.total_pad[:2 * self.ndim:2])


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

"""Scalar wave propagation module for Deepwave.

Implements scalar wave equation propagation using finite differences in time
(2nd order) and space (user-selectable order: 2, 4, 6, or 8). Supports PML
boundaries and adjoint mode for gradient computation. For PML details, see

    Pasalic, Damir, and Ray McGarry. "Convolutional perfectly matched
        layer for isotropic and anisotropic acoustic wave equations."
        SEG Technical Program Expanded Abstracts 2010.
        Society of Exploration Geophysicists, 2010. 2925-2929.


    Required inputs: wavespeed model (`v`), grid cell size (`dx`), time step (`dt`),
and either source term (`source_amplitudes` and `source_locations`) or number
of time steps (`nt`).
Outputs: final wavefields (including PML) and receiver amplitudes (empty if no
receivers).
All outputs are differentiable with respect to float torch.Tensor inputs.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.regular_grid


class Scalar(torch.nn.Module):
    """Convenience nn.Module wrapper for scalar wave propagation.

    Stores `v` and `grid_spacing`. Gradients do not propagate to the
    provided wavespeed. Use the module's `v` attribute to access the wavespeed.
    """

    def __init__(
        self,
        v: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        v_requires_grad: bool = False,
    ) -> None:
        """Initializes the Scalar propagator module.

        Args:
            v: A torch.Tensor containing the wavespeed model.
            grid_spacing: The spatial grid cell size. It can be a single number
                that will be used for all dimensions, or a number for each
                dimension.
            v_requires_grad: A bool specifying whether gradients will be
                computed for `v`. Defaults to False.

        """
        super().__init__()
        if not isinstance(v_requires_grad, bool):
            raise TypeError(
                f"v_requires_grad must be bool, got {type(v_requires_grad).__name__}",
            )
        if not isinstance(v, torch.Tensor):
            raise TypeError("v must be a torch.Tensor.")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitudes: Optional[torch.Tensor] = None,
        source_locations: Optional[torch.Tensor] = None,
        receiver_locations: Optional[torch.Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        wavefield_0: Optional[torch.Tensor] = None,
        wavefield_m1: Optional[torch.Tensor] = None,
        psiy_m1: Optional[torch.Tensor] = None,
        psix_m1: Optional[torch.Tensor] = None,
        zetay_m1: Optional[torch.Tensor] = None,
        zetax_m1: Optional[torch.Tensor] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        forward_callback: Optional[deepwave.common.Callback] = None,
        backward_callback: Optional[deepwave.common.Callback] = None,
        callback_frequency: int = 1,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Performs forward propagation/modelling.

        See :func:`scalar` for details.
        """
        return scalar(
            self.v,
            self.grid_spacing,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=accuracy,
            pml_width=pml_width,
            pml_freq=pml_freq,
            max_vel=max_vel,
            survey_pad=survey_pad,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            psiy_m1=psiy_m1,
            psix_m1=psix_m1,
            zetay_m1=zetay_m1,
            zetax_m1=zetax_m1,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            forward_callback=forward_callback,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
        )


def scalar(
    v: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    psiy_m1: Optional[torch.Tensor] = None,
    psix_m1: Optional[torch.Tensor] = None,
    zetay_m1: Optional[torch.Tensor] = None,
    zetax_m1: Optional[torch.Tensor] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    forward_callback: Optional[deepwave.common.Callback] = None,
    backward_callback: Optional[deepwave.common.Callback] = None,
    callback_frequency: int = 1,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Scalar wave propagation (functional interface).

    This function performs forward modelling with the scalar wave equation.
    The outputs are differentiable with respect to the wavespeed, the
    source amplitudes, and the initial wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    The function returns the final wavefields and the data recorded at
    the specified receiver locations. All returned torch.Tensors are
    differentiable with respect to the float torch.Tensors in the input.

    Args:
        v: A torch.Tensor containing the wavespeed model.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dt: A float specifying the time step interval of the input and
            output (internally a smaller interval may be used in
            propagation to obey the CFL condition for stability).
        source_amplitudes: A torch.Tensor with dimensions [shot, source, time].
            For example, if two shots are propagated simultaneously, each
            containing three sources of one hundred time samples, the shape
            would be [2, 3, 100]. The time dimension length determines the
            number of time steps in the simulation. Optional. If
            provided, `source_locations` must also be specified. If not
            provided, `nt` must be specified.
        source_locations: A torch.Tensor with dimensions [shot, source, 2],
            containing the index in the two spatial dimensions of the cell
            that each source is located in, relative to the origin of the
            model. Optional. Must be provided if `source_amplitudes` is. It
            should have torch.long (int64) datatype. The location of each
            source must be unique within the same shot (you cannot have two
            sources in the same shot that both have location [1, 2], for
            example). Setting both coordinates to
            deepwave.IGNORE_LOCATION will result in the source being
            ignored.
        receiver_locations: A torch.Tensor with dimensions
            [shot, receiver, 2], containing the coordinates of the cell
            containing each receiver. Optional. It should have torch.long
            (int64) datatype. If not provided, the output
            `receiver_amplitudes` torch.Tensor will be empty. If
            backpropagation will be performed, the location of each
            receiver must be unique within the same shot.
            Setting both coordinates to deepwave.IGNORE_LOCATION will
            result in the receiver being ignored.
        accuracy: An int specifying the finite difference order of accuracy.
            Possible values are 2, 4, 6, and 8, with larger numbers resulting
            in more accurate results but greater computational cost. Optional,
            with a default of 4.
        pml_width: A single number, or two numbers for each dimension,
            specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model.
            If a single value is provided, it will be used for all edges.
            If a sequence is provided, it should contain values for the
            edges in the following order:

            - beginning of first dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension

            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective or "free" surface, set the
            value for that edge to be zero. For example, if your model is
            oriented so that the surface that you want to be reflective is the
            beginning of the second dimension, then you could specify
            `pml_width=[20, 20, 0, 20]`. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
            model. Optional, default 20.
        pml_freq: A float specifying the frequency that you wish to use when
            constructing the PML. This is usually the dominant frequency
            of the source wavelet. Choosing this value poorly will
            result in the edges becoming more reflective. Optional, default
            25 Hz (assuming `dt` is in seconds).
        max_vel: A float specifying the maximum velocity, which is used when
            applying the CFL condition and when constructing the PML. If
            not specified, the actual maximum absolute wavespeed in the
            model (or portion of it that is used) will be used. The option
            to specify this is provided to allow you to ensure consistency
            between runs even if there are changes in the wavespeed.
            Optional, default None.
        survey_pad: A single value or list of four values, all of which are
            either an int or None, specifying whether the simulation domain
            should be restricted to a region surrounding the sources
            and receivers. If you have a large model, but the sources
            and receivers of individual shots only cover a small portion
            of it (such as in a towed streamer survey), then it may be
            wasteful to simulate wave propagation over the whole model. This
            parameter allows you to specify what distance around sources
            and receivers you would like to extract from the model to use
            for the simulation. A value of None means that no restriction
            of the model should be performed in that direction, while an
            integer value specifies the minimum number of cells that
            the edge of the simulation domain should be from any source
            or receiver in that direction (if possible). If a single value
            is provided, it applies to all directions, so specifying `None`
            will use the whole model for every simulation, while specifying
            `10` will cause the simulation domain to contain at least 10
            cells of padding in each direction around sources and receivers
            if possible. The padding will end if the edge of the model is
            encountered. Specifying a list, in the following order, allows
            the padding in each direction to be controlled:

            - beginning of first dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension

            Ints and `None` may be mixed, so a `survey_pad` of
            [5, None, None, 10] means that there should be at least 5 cells
            of padding towards the beginning of the first dimension, no
            restriction of the simulation domain towards the end of the
            first dimension or beginning of the second, and 10 cells of
            padding towards the end of the second dimension. If the
            simulation contains one source at [20, 15], one receiver at
            [10, 15], and the model is of shape [40, 50], then the
            extracted simulation domain with this value of `survey_pad`
            would cover the region [5:40, 0:25]. The same simulation
            domain will be used for all shots propagated simultaneously, so
            if this option is used then it is advisable to propagate shots
            that cover similar regions of the model simultaneously so that
            it provides a benefit (if the shots cover opposite ends of a
            dimension, then that whole dimension will be used regardless of
            what `survey_pad` value is used). Optional, default None.
            Cannot be specified if origin is also specified.

        wavefield_0: A torch.Tensor specifying the initial wavefield at time
            step 0. It should have three dimensions, with the first dimension
            being shot and the subsequent two corresponding to the two spatial
            dimensions. The spatial shape should be equal to the simulation
            domain, which is the extracted model plus the PML. If two shots
            are being propagated simultaneously in a region of size [20, 30]
            extracted from the model for the simulation, and
            `pml_width=[1, 2, 3, 4]`, then `wavefield_0` should be of shape
            [2, 23, 37]. Optional, default all zeros.
        wavefield_m1: A torch.Tensor specifying the initial wavefield at time
            step -1 (using Deepwave's internal time step interval, which may
            be smaller than the user provided one to obey the CFL condition).
            See the entry for `wavefield_0` for more details.
        psiy_m1: PML-related wavefield at time step -1.
        psix_m1: PML-related wavefield at time step -1.
        zetay_m1: PML-related wavefield at time step -1.
        zetax_m1: PML-related wavefield at time step -1.
        origin: A list of ints specifying the origin of the provided initial
            wavefields relative to the origin of the model. Only relevant
            if initial wavefields are provided. The origin of a wavefield
            is the cell where the extracted model used for the simulation
            domain began. It does not include the PML. So if a simulation
            is performed using the model region [10:20, 15:30], the origin
            is [10, 15]. Optional, default [0, 0]. Cannot be specified if
            survey_pad is also specified.
        nt: If the source amplitudes are not provided then you must instead
            specify the number of time steps to run the simulation for by
            providing an integer for `nt`. You cannot specify both the
            source amplitudes and `nt`.
        model_gradient_sampling_interval: An int specifying the number of time
            steps between contributions to the model gradient. The gradient
            with respect to the model is an integral over the backpropagation
            time steps. The time sampling frequency of this integral should be
            at least the Nyquist frequency of the source or data (whichever
            is greater). If this Nyquist frequency is substantially less
            than 1/dt, you may wish to reduce the sampling frequency of
            the integral to reduce computational (especially memory) costs.
            Optional, default 1 (integral is sampled every time step
            interval `dt`).
        freq_taper_frac: A float specifying the fraction of the end of the
            source and receiver amplitudes (if present) in the frequency
            domain to cosine taper if they are resampled due to the CFL
            condition. This might be useful to reduce ringing. A value of 0.1
            means that the top 10% of frequencies will be tapered.
            Default 0.0 (no tapering).
        time_pad_frac: A float specifying the amount of zero padding that will
            be added to the source and receiver amplitudes (if present) before
            resampling and removed afterwards, if they are resampled due to
            the CFL condition, as a fraction of their length. This might be
            useful to reduce wraparound artifacts. A value of 0.1 means that
            zero padding of 10% of the number of time samples will be used.
            Default 0.0.
        time_taper: A bool specifying whether to apply a Hann window in time to
            source and receiver amplitudes (if present). This is useful
            during correctness tests of the propagators as it ensures that
            signals taper to zero at their edges in time, avoiding the
            possibility of high frequencies being introduced.
        forward_callback: A function that will be called during the forward
            pass. See :class:`deepwave.common.CallbackState` for the
            state that will be provided to the function.
        backward_callback: A function that will be called during the backward
            pass. See :class:`deepwave.common.CallbackState` for the
            state that will be provided to the function.
        callback_frequency: The number of time steps between calls to the
            callback.

    Returns:
        Tuple:

            - wavefield_nt: The wavefield at the final time step.
            - wavefield_ntm1: The wavefield at the second-to-last time step.
            - psiy_ntm1: PML-related wavefield.
            - psix_ntm1: PML-related wavefield.
            - zetay_ntm1: PML-related wavefield.
            - zetax_ntm1: PML-related wavefield.
            - receiver_amplitudes: The receiver amplitudes. Empty if no
              receivers were specified.

    """
    v_nonzero = v[v != 0]
    if v_nonzero.numel() > 0:
        min_nonzero_model_vel = v_nonzero.abs().min().item()
    else:
        min_nonzero_model_vel = 0.0
    max_model_vel = v.abs().max().item()
    fd_pad = [accuracy // 2] * 4
    (
        models,
        source_amplitudes_out,
        wavefields,
        sources_i,
        receivers_i,
        grid_spacing,
        dt,
        nt,
        n_shots,
        step_ratio,
        model_gradient_sampling_interval,
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    ) = deepwave.common.setup_propagator(
        [v],
        ["replicate"],
        grid_spacing,
        dt,
        [source_amplitudes],
        [source_locations],
        [receiver_locations],
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
        survey_pad,
        [
            wavefield_0,
            wavefield_m1,
            psiy_m1,
            psix_m1,
            zetay_m1,
            zetax_m1,
        ],
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        2,
    )

    # In the finite difference implementation, the source amplitudes we
    # add to the wavefield each time step are multiplied by
    # -v^2 * dt^2 (where v is the velocity model, stored in models[0]).
    # For simplicity, we multiply the amplitudes by that here.
    # We use 0 as the location of the sources that are to be ignored to
    # avoid out-of-bounds accesses to the model. Since these sources will not be
    # used, their amplitudes are not important.
    ny, nx = models[0].shape[-2:]
    mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[0].clone()
    sources_i_masked[mask] = 0
    source_amplitudes_out[0] = (
        -source_amplitudes_out[0]
        * (models[0].view(-1, ny * nx).expand(n_shots, -1).gather(1, sources_i_masked))
        ** 2
        * dt**2
    )

    pml_profiles = deepwave.regular_grid.set_pml_profiles(
        pml_width,
        accuracy,
        fd_pad,
        dt,
        grid_spacing,
        max_vel,
        dtype,
        device,
        pml_freq,
        ny,
        nx,
    )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    (wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes) = scalar_func(
        *models,
        *source_amplitudes_out,
        *wavefields,
        *pml_profiles,
        *sources_i,
        *receivers_i,
        *grid_spacing,
        dt,
        nt,
        step_ratio * model_gradient_sampling_interval,
        accuracy,
        pml_width,
        n_shots,
        forward_callback,
        backward_callback,
        callback_frequency,
    )

    receiver_amplitudes = deepwave.common.downsample_and_movedim(
        receiver_amplitudes,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    return wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes


class ScalarForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of scalar wave propagation.

    This class defines the forward and backward passes for the scalar wave
    equation, allowing PyTorch to compute gradients through the wave propagation
    operation. It interfaces directly with the C/CUDA backend.
    """

    @staticmethod
    def forward(
        ctx: Any,
        v: torch.Tensor,
        source_amplitudes: torch.Tensor,
        wfc: torch.Tensor,
        wfp: torch.Tensor,
        psiy: torch.Tensor,
        psix: torch.Tensor,
        zetay: torch.Tensor,
        zetax: torch.Tensor,
        ay: torch.Tensor,
        ax: torch.Tensor,
        by: torch.Tensor,
        bx: torch.Tensor,
        dbydy: torch.Tensor,
        dbxdx: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        dy: float,
        dx: float,
        dt: float,
        nt: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        n_shots: int,
        forward_callback: Optional[deepwave.common.Callback],
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Performs the forward propagation of the scalar wave equation.

        This method is called by PyTorch during the forward pass. It prepares
        the input tensors, calls the appropriate C/CUDA function for wave
        propagation, and saves necessary tensors for the backward pass.

        Args:
            ctx: A context object that can be used to save information for
                the backward pass.
            v: The wavespeed model tensor.
            source_amplitudes: The source amplitudes tensor.
            wfc: Wavefield at current time step.
            wfp: Wavefield at previous time step.
            psiy: PML auxiliary variable for y-dimension (current time step).
            psix: PML auxiliary variable for x-dimension (current time step).
            zetay: PML auxiliary variable for y-dimension (previous time step).
            zetax: PML auxiliary variable for x-dimension (previous time step).
            ay: PML absorption profile for y-dimension (a-coefficient).
            ax: PML absorption profile for x-dimension (a-coefficient).
            by: PML absorption profile for y-dimension (b-coefficient).
            bx: PML absorption profile for x-dimension (b-coefficient).
            dbydy: Derivative of PML b-coefficient with respect to y.
            dbxdx: Derivative of PML b-coefficient with respect to x.
            sources_i: 1D indices of source locations.
            receivers_i: 1D indices of receiver locations.
            dy: Grid spacing in y-dimension.
            dx: Grid spacing in x-dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            n_shots: Number of shots in the batch.
            forward_callback: The forward callback.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.

        Returns:
            A tuple containing the output wavefields and receiver amplitudes.

        """
        if (
            v.requires_grad
            or source_amplitudes.requires_grad
            or wfc.requires_grad
            or wfp.requires_grad
            or psiy.requires_grad
            or psix.requires_grad
            or zetay.requires_grad
            or zetax.requires_grad
        ):
            ctx.save_for_backward(
                v,
                ay,
                ax,
                by,
                bx,
                dbydy,
                dbxdx,
                sources_i,
                receivers_i,
                source_amplitudes,
                wfc,
                wfp,
                psiy,
                psix,
                zetay,
                zetax,
            )
            ctx.dy = dy
            ctx.dx = dx
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = source_amplitudes.requires_grad
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency

        v = v.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()

        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape[-2:])
        wfc = deepwave.common.create_or_pad(
            wfc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfp = deepwave.common.create_or_pad(
            wfp,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.create_or_pad(
            psiy,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psix = deepwave.common.create_or_pad(
            psix,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetay = deepwave.common.create_or_pad(
            zetay,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetax = deepwave.common.create_or_pad(
            zetax,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.zero_interior(psiy, fd_pad, pml_width, True)
        psix = deepwave.common.zero_interior(psix, fd_pad, pml_width, False)
        zetay = deepwave.common.zero_interior(zetay, fd_pad, pml_width, True)
        zetax = deepwave.common.zero_interior(zetax, fd_pad, pml_width, False)

        device = v.device
        dtype = v.dtype
        ny = v.shape[-2]
        nx = v.shape[-1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        dwdv = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        v_batched = v.ndim == 3 and v.shape[0] > 1

        if v.requires_grad:
            dwdv.resize_(nt // step_ratio, *wfc.shape)
            dwdv.fill_(0)
        if receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)

        if v.is_cuda:
            aux = v.get_device()
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1
        forward = deepwave.backend_utils.get_backend_function(
            "scalar",
            "forward",
            accuracy,
            dtype,
            v.device,
        )

        if forward_callback is None:
            callback_frequency = nt // step_ratio

        if wfc.numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                if forward_callback is not None:
                    state = deepwave.common.CallbackState(
                        dt,
                        step,
                        {
                            "wavefield_0": wfc,
                            "wavefield_m1": wfp,
                            "psiy_m1": psiy,
                            "psix_m1": psix,
                            "zetay_m1": zetay,
                            "zetax_m1": zetax,
                        },
                        {"v": v},
                        {},
                        [fd_pad] * 4,
                        pml_width,
                    )
                    forward_callback(state)
                step_nt = min(nt // step_ratio - step, callback_frequency)
                forward(
                    v.data_ptr(),
                    source_amplitudes.data_ptr(),
                    wfc.data_ptr(),
                    wfp.data_ptr(),
                    psiy.data_ptr(),
                    psix.data_ptr(),
                    psiyn.data_ptr(),
                    psixn.data_ptr(),
                    zetay.data_ptr(),
                    zetax.data_ptr(),
                    dwdv.data_ptr(),
                    receiver_amplitudes.data_ptr(),
                    ay.data_ptr(),
                    ax.data_ptr(),
                    by.data_ptr(),
                    bx.data_ptr(),
                    dbydy.data_ptr(),
                    dbxdx.data_ptr(),
                    sources_i.data_ptr(),
                    receivers_i.data_ptr(),
                    1 / dy,
                    1 / dx,
                    1 / dy**2,
                    1 / dx**2,
                    dt**2,
                    step_nt * step_ratio,
                    n_shots,
                    ny,
                    nx,
                    n_sources_per_shot,
                    n_receivers_per_shot,
                    step_ratio,
                    v.requires_grad,
                    v_batched,
                    step * step_ratio,
                    pml_y0,
                    pml_y1,
                    pml_x0,
                    pml_x1,
                    aux,
                )
                if (step_nt * step_ratio) % 2 != 0:
                    wfc, wfp, psiy, psix, psiyn, psixn = (
                        wfp,
                        wfc,
                        psiyn,
                        psixn,
                        psiy,
                        psix,
                    )

        ctx.dwdv = dwdv

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        return (
            wfc[s],
            wfp[s],
            psiy[s],
            psix[s],
            zetay[s],
            zetax[s],
            receiver_amplitudes,
        )

    @staticmethod
    def backward(
        ctx: Any,
        gwfc: torch.Tensor,
        gwfp: torch.Tensor,
        gpsiy: torch.Tensor,
        gpsix: torch.Tensor,
        gzetay: torch.Tensor,
        gzetax: torch.Tensor,
        grad_r: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object that contains information saved during the
                forward pass.
            gwfc: Gradient of the loss with respect to `wfc`.
            gwfp: Gradient of the loss with respect to `wfp`.
            gpsiy: Gradient of the loss with respect to `psiy`.
            gpsix: Gradient of the loss with respect to `psix`.
            gzetay: Gradient of the loss with respect to `zetay`.
            gzetax: Gradient of the loss with respect to `zetax`.
            grad_r: Gradient of the loss with respect to `receiver_amplitudes`.

        Returns:
            Gradients with respect to the inputs of the forward pass.

        """
        (
            v,
            ay,
            ax,
            by,
            bx,
            dbydy,
            dbxdx,
            sources_i,
            receivers_i,
            source_amplitudes,
            wfc,
            wfp,
            psiy,
            psix,
            zetay,
            zetax,
        ) = ctx.saved_tensors
        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        dwdv = ctx.dwdv

        result = ScalarBackwardFunc.apply(  # type: ignore[no-untyped-call]
            gwfc,
            gwfp,
            gpsiy,
            gpsix,
            gzetay,
            gzetax,
            grad_r,
            v,
            ay,
            ax,
            by,
            bx,
            dbydy,
            dbxdx,
            sources_i,
            receivers_i,
            dwdv,
            source_amplitudes,
            wfc,
            wfp,
            psiy,
            psix,
            zetay,
            zetax,
            dy,
            dx,
            dt,
            nt,
            n_shots,
            step_ratio,
            accuracy,
            pml_width,
            source_amplitudes_requires_grad,
            ctx.backward_callback,
            ctx.callback_frequency,
        )
        return (
            cast(
                "Tuple[Optional[torch.Tensor], Optional[torch.Tensor], "
                "Optional[torch.Tensor], Optional[torch.Tensor], "
                "Optional[torch.Tensor], Optional[torch.Tensor], "
                "Optional[torch.Tensor], Optional[torch.Tensor]]",
                result,
            )
            + (None,) * 19
        )


class ScalarBackwardFunc(torch.autograd.Function):
    """Autograd function for the backward pass of scalar wave propagation.

    This class defines the forward and backward passes for computing gradients
    of the scalar wave equation, interfacing directly with the C/CUDA backend.
    It is typically called by `ScalarForwardFunc.backward`.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gwfc: torch.Tensor,
        gwfp: torch.Tensor,
        gpsiy: torch.Tensor,
        gpsix: torch.Tensor,
        gzetay: torch.Tensor,
        gzetax: torch.Tensor,
        grad_r: torch.Tensor,
        v: torch.Tensor,
        ay: torch.Tensor,
        ax: torch.Tensor,
        by: torch.Tensor,
        bx: torch.Tensor,
        dbydy: torch.Tensor,
        dbxdx: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        dwdv: torch.Tensor,
        source_amplitudes: torch.Tensor,
        wfc: torch.Tensor,
        wfp: torch.Tensor,
        psiy: torch.Tensor,
        psix: torch.Tensor,
        zetay: torch.Tensor,
        zetax: torch.Tensor,
        dy: float,
        dx: float,
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        source_amplitudes_requires_grad: bool,
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Performs the backward propagation of the scalar wave equation.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object for saving information for the backward pass.
            gwfc: Gradient of the loss with respect to `wfc`.
            gwfp: Gradient of the loss with respect to `wfp`.
            gpsiy: Gradient of the loss with respect to `psiy`.
            gpsix: Gradient of the loss with respect to `psix`.
            gzetay: Gradient of the loss with respect to `zetay`.
            gzetax: Gradient of the loss with respect to `zetax`.
            grad_r: Gradient of the loss with respect to `receiver_amplitudes`.
            v: The wavespeed model tensor.
            ay: PML absorption profile for y-dimension (a-coefficient).
            ax: PML absorption profile for x-dimension (a-coefficient).
            by: PML absorption profile for y-dimension (b-coefficient).
            bx: PML absorption profile for x-dimension (b-coefficient).
            dbydy: Derivative of PML b-coefficient with respect to y.
            dbxdx: Derivative of PML b-coefficient with respect to x.
            sources_i: 1D indices of source locations.
            receivers_i: 1D indices of receiver locations.
            dwdv: Wavefield for model gradient calculation.
            source_amplitudes: The source amplitudes tensor.
            wfc: Wavefield at current time step (from forward pass).
            wfp: Wavefield at previous time step (from forward pass).
            psiy: PML y-dim auxiliary variable (current, from forward pass).
            psix: PML x-dim auxiliary variable (current, from forward pass).
            zetay: PML y-dim auxiliary variable (previous, from forward pass).
            zetax: PML x-dim auxiliary variable (previous, from forward pass).
            dy: Grid spacing in y-dimension.
            dx: Grid spacing in x-dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            n_shots: Number of shots in the batch.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            source_amplitudes_requires_grad: If source amplitudes need grads.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.

        Returns:
            A tuple containing the gradients of the inputs.

        """
        ctx.save_for_backward(
            gwfc,
            gwfp,
            gpsiy,
            gpsix,
            gzetay,
            gzetax,
            grad_r,
            v,
            ay,
            ax,
            by,
            bx,
            dbydy,
            dbxdx,
            sources_i,
            receivers_i,
            source_amplitudes,
            wfc,
            wfp,
            psiy,
            psix,
            zetay,
            zetax,
        )
        ctx.dy = dy
        ctx.dx = dx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.pml_width = pml_width
        ctx.source_amplitudes_requires_grad = source_amplitudes_requires_grad

        v = v.contiguous()
        grad_r = grad_r.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        dwdv = dwdv.contiguous()

        device = v.device
        dtype = v.dtype
        ny = v.shape[-2]
        nx = v.shape[-1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        fd_pad = accuracy // 2

        size_with_batch = (n_shots, *v.shape[-2:])
        gwfc = deepwave.common.create_or_pad(
            gwfc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gwfp = deepwave.common.create_or_pad(
            gwfp,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gpsiy = deepwave.common.create_or_pad(
            gpsiy,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gpsix = deepwave.common.create_or_pad(
            gpsix,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gzetay = deepwave.common.create_or_pad(
            gzetay,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gzetax = deepwave.common.create_or_pad(
            gzetax,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gpsiy = deepwave.common.zero_interior(gpsiy, fd_pad, pml_width, True)
        gpsix = deepwave.common.zero_interior(gpsix, fd_pad, pml_width, False)
        gzetay = deepwave.common.zero_interior(gzetay, fd_pad, pml_width, True)
        gzetax = deepwave.common.zero_interior(gzetax, fd_pad, pml_width, False)

        gpsiyn = torch.zeros_like(gpsiy)
        gpsixn = torch.zeros_like(gpsix)
        gzetayn = torch.zeros_like(gzetay)
        gzetaxn = torch.zeros_like(gzetax)
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        if v.requires_grad:
            grad_v.resize_(*v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 3 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 3 * fd_pad)
        pml_x0 = min(pml_width[2] + 3 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 3 * fd_pad)

        v_batched = v.ndim == 3 and v.shape[0] > 1

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad and not v_batched and n_shots > 1:
                grad_v_tmp.resize_(n_shots, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
        else:
            if deepwave.backend_utils.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if (
                v.requires_grad
                and not v_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_v_tmp.resize_(aux, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
        backward = deepwave.backend_utils.get_backend_function(
            "scalar",
            "backward",
            accuracy,
            dtype,
            v.device,
        )

        v2dt2 = v**2 * dt**2
        gwfp = -gwfp

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        if gwfc.numel() > 0 and nt > 0:
            for step in range(nt // step_ratio, 0, -callback_frequency):
                step_nt = min(step, callback_frequency)
                backward(
                    v2dt2.data_ptr(),
                    grad_r.data_ptr(),
                    gwfc.data_ptr(),
                    gwfp.data_ptr(),
                    gpsiy.data_ptr(),
                    gpsix.data_ptr(),
                    gpsiyn.data_ptr(),
                    gpsixn.data_ptr(),
                    gzetay.data_ptr(),
                    gzetax.data_ptr(),
                    gzetayn.data_ptr(),
                    gzetaxn.data_ptr(),
                    dwdv.data_ptr(),
                    grad_f.data_ptr(),
                    grad_v.data_ptr(),
                    grad_v_tmp_ptr,
                    ay.data_ptr(),
                    ax.data_ptr(),
                    by.data_ptr(),
                    bx.data_ptr(),
                    dbydy.data_ptr(),
                    dbxdx.data_ptr(),
                    sources_i.data_ptr(),
                    receivers_i.data_ptr(),
                    1 / dy,
                    1 / dx,
                    1 / dy**2,
                    1 / dx**2,
                    step_nt * step_ratio,
                    n_shots,
                    ny,
                    nx,
                    n_sources_per_shot * source_amplitudes_requires_grad,
                    n_receivers_per_shot,
                    step_ratio,
                    v.requires_grad,
                    v_batched,
                    step * step_ratio,
                    pml_y0,
                    pml_y1,
                    pml_x0,
                    pml_x1,
                    aux,
                )
                if (step_nt * step_ratio) % 2 != 0:
                    (
                        gwfc,
                        gwfp,
                        gpsiy,
                        gpsix,
                        gpsiyn,
                        gpsixn,
                        gzetay,
                        gzetax,
                        gzetayn,
                        gzetaxn,
                    ) = (
                        gwfp,
                        gwfc,
                        gpsiyn,
                        gpsixn,
                        gpsiy,
                        gpsix,
                        gzetayn,
                        gzetaxn,
                        gzetay,
                        gzetax,
                    )
                if backward_callback is not None:
                    state = deepwave.common.CallbackState(
                        dt,
                        step - 1,
                        {
                            "wavefield_0": gwfc,
                            "wavefield_m1": gwfp,
                            "psiy_m1": gpsiy,
                            "psix_m1": gpsix,
                            "zetay_m1": gzetay,
                            "zetax_m1": gzetax,
                        },
                        {"v": v},
                        {"v": grad_v},
                        [fd_pad] * 4,
                        pml_width,
                    )
                    backward_callback(state)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        return (
            grad_v,
            grad_f,
            gwfc[s],
            -gwfp[s],
            gpsiy[s],
            gpsix[s],
            gzetay[s],
            gzetax[s],
        )

    @staticmethod
    @torch.autograd.function.once_differentiable  # type: ignore[misc]
    def backward(
        ctx: Any,
        ggv: torch.Tensor,
        ggf: torch.Tensor,
        ggwfc: torch.Tensor,
        ggwfp: torch.Tensor,
        ggpsiy: torch.Tensor,
        ggpsix: torch.Tensor,
        ggzetay: torch.Tensor,
        ggzetax: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients of the backward pass (second-order gradients).

        This method is called by PyTorch during the backward pass of the backward
        pass to compute second-order gradients.

        Args:
            ctx: A context object that contains information saved during the
                forward pass of the backward function.
            ggv: Gradient of the loss with respect to the gradient of `v`.
            ggf: Gradient of the loss with respect to the gradient of
                `source_amplitudes`.
            ggwfc: Gradient of the loss with respect to the gradient of `wfc`.
            ggwfp: Gradient of the loss with respect to the gradient of `wfp`.
            ggpsiy: Gradient of the loss with respect to the gradient of `psiy`.
            ggpsix: Gradient of the loss with respect to the gradient of `psix`.
            ggzetay: Gradient of the loss with respect to the gradient of
                `zetay`.
            ggzetax: Gradient of the loss with respect to the gradient of
                `zetax`.

        Returns:
            Second-order gradients with respect to the inputs of the
            `ScalarBackwardFunc.forward` method.

        """
        (
            gwfc,
            gwfp,
            gpsiy,
            gpsix,
            gzetay,
            gzetax,
            grad_r,
            v,
            ay,
            ax,
            by,
            bx,
            dbydy,
            dbxdx,
            sources_i,
            receivers_i,
            source_amplitudes,
            wfc,
            wfp,
            psiy,
            psix,
            zetay,
            zetax,
        ) = ctx.saved_tensors
        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad

        if ggv.numel() == 0:
            ggv = torch.zeros_like(v.detach())
        if ggf.numel() == 0:
            ggf = torch.zeros_like(source_amplitudes.detach())

        v = v.contiguous()
        ggv = ggv.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        ggsource_amplitudes = ggf.contiguous()
        grad_r = grad_r.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        fwd_sources_i = sources_i.contiguous()
        fwd_receivers_i = torch.empty(0)
        fwd_ggreceivers_i = receivers_i.contiguous()

        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape[-2:])
        wfc = deepwave.common.create_or_pad(
            wfc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfp = deepwave.common.create_or_pad(
            wfp,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.create_or_pad(
            psiy,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psix = deepwave.common.create_or_pad(
            psix,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetay = deepwave.common.create_or_pad(
            zetay,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetax = deepwave.common.create_or_pad(
            zetax,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        ggwfc = deepwave.common.create_or_pad(
            ggwfc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        ggwfp = deepwave.common.create_or_pad(
            ggwfp,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        ggpsiy = deepwave.common.create_or_pad(
            ggpsiy,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        ggpsix = deepwave.common.create_or_pad(
            ggpsix,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        ggzetay = deepwave.common.create_or_pad(
            ggzetay,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        ggzetax = deepwave.common.create_or_pad(
            ggzetax,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.zero_interior(psiy, fd_pad, pml_width, True)
        psix = deepwave.common.zero_interior(psix, fd_pad, pml_width, False)
        zetay = deepwave.common.zero_interior(zetay, fd_pad, pml_width, True)
        zetax = deepwave.common.zero_interior(zetax, fd_pad, pml_width, False)
        ggpsiy = deepwave.common.zero_interior(ggpsiy, fd_pad, pml_width, True)
        ggpsix = deepwave.common.zero_interior(ggpsix, fd_pad, pml_width, False)
        ggzetay = deepwave.common.zero_interior(ggzetay, fd_pad, pml_width, True)
        ggzetax = deepwave.common.zero_interior(ggzetax, fd_pad, pml_width, False)

        device = v.device
        dtype = v.dtype
        ny = v.shape[-2]
        nx = v.shape[-1]
        n_sources_per_shot = fwd_sources_i.numel() // n_shots
        n_receivers_per_shot = fwd_receivers_i.numel() // n_shots
        n_ggreceivers_per_shot = fwd_ggreceivers_i.numel() // n_shots
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        ggpsiyn = torch.zeros_like(ggpsiy)
        ggpsixn = torch.zeros_like(ggpsix)
        w_store = torch.empty(0, device=device, dtype=dtype)
        ggw_store = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        ggreceiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        v_batched = v.ndim == 3 and v.shape[0] > 1
        ggv_batched = ggv.ndim == 3 and ggv.shape[0] > 1

        if v.requires_grad:
            w_store.resize_(nt // step_ratio, *wfc.shape)
            w_store.fill_(0)
            ggw_store.resize_(nt // step_ratio, *wfc.shape)
            ggw_store.fill_(0)
        if fwd_receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)
        if fwd_ggreceivers_i.numel() > 0:
            ggreceiver_amplitudes.resize_(nt, n_shots, n_ggreceivers_per_shot)
            ggreceiver_amplitudes.fill_(0)

        if v.is_cuda:
            aux = v.get_device()
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1
        forward = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            "forward",
            accuracy,
            dtype,
            v.device,
        )

        if wfc.numel() > 0 and nt > 0:
            forward(
                v.data_ptr(),
                ggv.data_ptr(),
                source_amplitudes.data_ptr(),
                ggsource_amplitudes.data_ptr(),
                wfc.data_ptr(),
                wfp.data_ptr(),
                psiy.data_ptr(),
                psix.data_ptr(),
                psiyn.data_ptr(),
                psixn.data_ptr(),
                zetay.data_ptr(),
                zetax.data_ptr(),
                ggwfc.data_ptr(),
                ggwfp.data_ptr(),
                ggpsiy.data_ptr(),
                ggpsix.data_ptr(),
                ggpsiyn.data_ptr(),
                ggpsixn.data_ptr(),
                ggzetay.data_ptr(),
                ggzetax.data_ptr(),
                w_store.data_ptr(),
                ggw_store.data_ptr(),
                receiver_amplitudes.data_ptr(),
                ggreceiver_amplitudes.data_ptr(),
                ay.data_ptr(),
                ax.data_ptr(),
                by.data_ptr(),
                bx.data_ptr(),
                dbydy.data_ptr(),
                dbxdx.data_ptr(),
                fwd_sources_i.data_ptr(),
                fwd_receivers_i.data_ptr(),
                fwd_ggreceivers_i.data_ptr(),
                1 / dy,
                1 / dx,
                1 / dy**2,
                1 / dx**2,
                dt**2,
                nt,
                n_shots,
                ny,
                nx,
                n_sources_per_shot,
                n_receivers_per_shot,
                n_ggreceivers_per_shot,
                step_ratio,
                v.requires_grad,
                False,
                v_batched,
                ggv_batched,
                0,
                pml_y0,
                pml_y1,
                pml_x0,
                pml_x1,
                aux,
            )

        bwd_sources_i = sources_i.contiguous()
        bwd_receivers_i = torch.empty(0)
        bwd_greceivers_i = receivers_i.contiguous()

        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        n_sources_per_shot = bwd_sources_i.numel() // n_shots
        n_receivers_per_shot = bwd_receivers_i.numel() // n_shots
        n_greceivers_per_shot = bwd_greceivers_i.numel() // n_shots

        wfc = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfp = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psix = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetay = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetax = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.zero_interior(psiy, fd_pad, pml_width, True)
        psix = deepwave.common.zero_interior(psix, fd_pad, pml_width, False)
        zetay = deepwave.common.zero_interior(zetay, fd_pad, pml_width, True)
        zetax = deepwave.common.zero_interior(zetax, fd_pad, pml_width, False)
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        zetayn = torch.zeros_like(zetay)
        zetaxn = torch.zeros_like(zetax)
        gwfc = deepwave.common.create_or_pad(
            gwfc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gwfp = deepwave.common.create_or_pad(
            gwfp,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gpsiy = deepwave.common.create_or_pad(
            gpsiy,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gpsix = deepwave.common.create_or_pad(
            gpsix,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gzetay = deepwave.common.create_or_pad(
            gzetay,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gzetax = deepwave.common.create_or_pad(
            gzetax,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        gpsiy = deepwave.common.zero_interior(gpsiy, fd_pad, pml_width, True)
        gpsix = deepwave.common.zero_interior(gpsix, fd_pad, pml_width, False)
        gzetay = deepwave.common.zero_interior(gzetay, fd_pad, pml_width, True)
        gzetax = deepwave.common.zero_interior(gzetax, fd_pad, pml_width, False)
        gpsiyn = torch.zeros_like(gpsiy)
        gpsixn = torch.zeros_like(gpsix)
        gzetayn = torch.zeros_like(gzetay)
        gzetaxn = torch.zeros_like(gzetax)
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        if v.requires_grad:
            grad_v.resize_(v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        grad_gf = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 3 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 3 * fd_pad)
        pml_x0 = min(pml_width[2] + 3 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 3 * fd_pad)

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad and not v_batched and n_shots > 1:
                grad_v_tmp.resize_(n_shots, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
        else:
            if deepwave.backend_utils.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if (
                v.requires_grad
                and not v_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_v_tmp.resize_(aux, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
        backward = deepwave.backend_utils.get_backend_function(
            "scalar_born", "backward", accuracy, dtype, v.device
        )

        gwfp = -gwfp

        if wfc.numel() > 0 and nt > 0 and v.requires_grad:
            backward(
                v.data_ptr(),
                ggv.data_ptr(),
                torch.empty(0).data_ptr(),
                grad_r.data_ptr(),
                wfc.data_ptr(),
                wfp.data_ptr(),
                psiy.data_ptr(),
                psix.data_ptr(),
                psiyn.data_ptr(),
                psixn.data_ptr(),
                zetay.data_ptr(),
                zetax.data_ptr(),
                zetayn.data_ptr(),
                zetaxn.data_ptr(),
                gwfc.data_ptr(),
                gwfp.data_ptr(),
                gpsiy.data_ptr(),
                gpsix.data_ptr(),
                gpsiyn.data_ptr(),
                gpsixn.data_ptr(),
                gzetay.data_ptr(),
                gzetax.data_ptr(),
                gzetayn.data_ptr(),
                gzetaxn.data_ptr(),
                w_store.data_ptr(),
                ggw_store.data_ptr(),
                grad_f.data_ptr(),
                grad_gf.data_ptr(),
                grad_v.data_ptr(),
                torch.empty(0).data_ptr(),
                grad_v_tmp_ptr,
                torch.empty(0).data_ptr(),
                ay.data_ptr(),
                ax.data_ptr(),
                by.data_ptr(),
                bx.data_ptr(),
                dbydy.data_ptr(),
                dbxdx.data_ptr(),
                bwd_sources_i.data_ptr(),
                bwd_receivers_i.data_ptr(),
                bwd_greceivers_i.data_ptr(),
                1 / dy,
                1 / dx,
                1 / dy**2,
                1 / dx**2,
                dt**2,
                nt,
                n_shots,
                ny,
                nx,
                n_sources_per_shot * source_amplitudes_requires_grad,
                0,
                n_receivers_per_shot,
                n_greceivers_per_shot,
                step_ratio,
                v.requires_grad,
                False,
                v_batched,
                ggv_batched,
                nt,
                pml_y0,
                pml_y1,
                pml_x0,
                pml_x1,
                aux,
            )

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return (
                ggwfc[s],
                ggwfp[s],
                ggpsiy[s],
                ggpsix[s],
                ggzetay[s],
                ggzetax[s],
                ggreceiver_amplitudes,
                grad_v,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                grad_f,
                wfc[s],
                -wfp[s],
                psiy[s],
                psix[s],
                zetay[s],
                zetax[s],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return (
            ggwfp[s],
            ggwfc[s],
            ggpsiyn[s],
            ggpsixn[s],
            ggzetay[s],
            ggzetax[s],
            ggreceiver_amplitudes,
            grad_v,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_f,
            wfp[s],
            -wfc[s],
            psiyn[s],
            psixn[s],
            zetayn[s],
            zetaxn[s],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def scalar_func(*args: Any) -> Tuple[torch.Tensor, ...]:
    """Helper function to apply the ScalarForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ScalarForwardFunc`, which is the entry point for the autograd graph
    for scalar wave propagation.

    Args:
        *args: Variable length argument list to be passed directly to
            `ScalarForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ScalarForwardFunc.apply`.

    """
    return cast(
        "Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor]",
        ScalarForwardFunc.apply(*args),  # type: ignore[no-untyped-call]
    )

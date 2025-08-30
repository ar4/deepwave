"""
Scalar wave propagation module for Deepwave.

Implements scalar wave equation propagation using finite differences in time (2nd order)
and space (user-selectable order: 2, 4, 6, or 8). Supports PML boundaries and adjoint mode
for gradient computation. See Pasalic & McGarry (SEG 2010) for PML details.

Required inputs: wavespeed model (`v`), grid cell size (`dx`), time step (`dt`), and either
source term (`source_amplitudes` and `source_locations`) or number of time steps (`nt`).
Outputs: final wavefields (including PML) and receiver amplitudes (empty if no receivers).
All outputs are differentiable with respect to float Tensor inputs.
"""

from typing import Optional, Union, List, Tuple, Sequence, Any
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from deepwave.backend_utils import dll, USE_OPENMP
from deepwave.common import (
    setup_propagator,
    downsample_and_movedim,
    zero_interior,
    create_or_pad,
    IGNORE_LOCATION,
    PMLConfig,
    SurveyConfig,
    _as_list,
)
from deepwave.regular_grid import set_pml_profiles


class Scalar(torch.nn.Module):
    """
    Convenience nn.Module wrapper for scalar wave propagation.

    Stores `v` and `grid_spacing` for repeated use. Gradients do not propagate to the
    initial guess wavespeed. Use the module's `v` attribute to access the wavespeed.
    """

    def __init__(
        self,
        v: Tensor,
        grid_spacing: Union[int, float, torch.Tensor, Sequence[Union[int, float]]],
        v_requires_grad: bool = False,
    ) -> None:
        """
        Initialize the Scalar propagator module.

        Args:
            v (Tensor): Wavespeed model.
            grid_spacing (int, float, torch.Tensor, or sequence of them): Grid cell size(s).
            v_requires_grad (bool, optional): If True, gradients will be computed for v. Default: False.
        """
        super().__init__()
        if not isinstance(v_requires_grad, bool):
            raise TypeError(
                f"v_requires_grad must be bool, got {type(v_requires_grad).__name__}"
            )
        if not isinstance(v, Tensor):
            raise RuntimeError("model must be a torch.Tensor.")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: Union[int, float],
        source_amplitudes: Optional[Tensor] = None,
        source_locations: Optional[Tensor] = None,
        receiver_locations: Optional[Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, float, torch.Tensor, Sequence[Union[int, float]]] = 20,
        pml_freq: Optional[Union[int, float]] = None,
        max_vel: Optional[Union[int, float]] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        wavefield_0: Optional[Tensor] = None,
        wavefield_m1: Optional[Tensor] = None,
        psiy_m1: Optional[Tensor] = None,
        psix_m1: Optional[Tensor] = None,
        zetay_m1: Optional[Tensor] = None,
        zetax_m1: Optional[Tensor] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform forward propagation/modelling. See `scalar` for details.
        """
        pml_config = PMLConfig(_as_list(pml_width, "pml_width", int), pml_freq)
        survey_config = SurveyConfig(
            source_locations=[source_locations],
            receiver_locations=[receiver_locations],
            source_amplitudes=[source_amplitudes],
            wavefields=[
                wavefield_0,
                wavefield_m1,
                psiy_m1,
                psix_m1,
                zetay_m1,
                zetax_m1,
            ],
            survey_pad=survey_pad,
            origin=origin,
        )
        return scalar(
            self.v,
            self.grid_spacing,
            dt,
            accuracy=accuracy,
            pml_config=pml_config,
            max_vel=max_vel,
            survey_config=survey_config,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )


def scalar(
    v: Tensor,
    grid_spacing: Union[int, float, torch.Tensor, Sequence[Union[int, float]]],
    dt: Union[int, float],
    source_amplitudes: Optional[Tensor] = None,
    source_locations: Optional[Tensor] = None,
    receiver_locations: Optional[Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, float, torch.Tensor, Sequence[Union[int, float]]] = 20,
    pml_freq: Optional[Union[int, float]] = None,
    max_vel: Optional[Union[int, float]] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    wavefield_0: Optional[Tensor] = None,
    wavefield_m1: Optional[Tensor] = None,
    psiy_m1: Optional[Tensor] = None,
    psix_m1: Optional[Tensor] = None,
    zetay_m1: Optional[Tensor] = None,
    zetax_m1: Optional[Tensor] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    pml_config: Optional[PMLConfig] = None,
    survey_config: Optional[SurveyConfig] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Scalar wave propagation (functional interface).

    This function performs forward modelling with the scalar wave equation.
    The outputs are differentiable with respect to the wavespeed, the
    source amplitudes, and the initial wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    The scalar wave equation is: `d^2u/dt^2 = v^2 * laplacian(u) + v^2 * f`,
    where `u` is the wavefield, `t` is time, `v` is the wavespeed, and `f`
    is the source. The Laplacian is applied to the spatial dimensions.

    The function returns the final wavefields and the data recorded at
    the specified receiver locations. All returned Tensors are
    differentiable with respect to the float Tensors in the input.

    Args:
        v:
            A Tensor containing the wavespeed model. Unlike the module
            interface (:class:`Scalar`), in this functional interface a copy is
            not made of the model, so gradients will propagate back into
            the provided Tensor.
        grid_spacing:
            The spatial grid cell size. It can be a single number (int or
            float), a torch.Tensor (scalar or with two elements), or a
            sequence (list or tuple) of two numbers.
        dt:
            A number (int or float) specifying the time step interval of
            the input and output (internally a smaller interval may be
            used in propagation to obey the CFL condition for stability).
        source_amplitudes:
            A Tensor with dimensions [shot, source, time]. If two shots
            are being propagated simultaneously, each containing three
            sources of one hundred time samples, it would have shape
            [2, 3, 100]. The time length will be the number of time steps
            in the simulation. Optional. If provided, `source_locations` must
            also be specified. If not provided, `nt` must be specified.
        source_locations:
            A Tensor with dimensions [shot, source, 2], containing the
            index in the two spatial dimensions of the cell that each
            source is located in, relative to the origin of the model.
            Optional. Must be provided if `source_amplitudes` is. It should
            have torch.long (int64) datatype. The location of each source
            must be unique within the same shot (you cannot have two
            sources in the same shot that both have location [1, 2], for
            example).
        receiver_locations:
            A Tensor with dimensions [shot, receiver, 2], containing
            the coordinates of the cell containing each receiver. Optional.
            It should have torch.long (int64) datatype. If not provided,
            the output `receiver_amplitudes` Tensor will be empty. If
            backpropagation will be performed, the location of each
            receiver must be unique within the same shot.
        accuracy:
            An int specifying the finite difference order of accuracy. Possible
            values are 2, 4, 6, and 8, with larger numbers resulting in more
            accurate results but greater computational cost. Optional, with
            a default of 4.
        pml_width:
            A number (int or float), a torch.Tensor, or a sequence of
            numbers specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model. Floats
            will be truncated to integers. If a single value is provided,
            it will be used for all edges. If a sequence is provided, it
            should contain the values for the edges in the following order:
            [the beginning of the first dimension,
            the end of the first dimension,
            the beginning of the second dimension,
            the end of the second dimension].
            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective or "free" surface, set the
            value for that edge to be zero. For example, if your model is
            oriented so that the surface that you want to be reflective is the
            beginning of the second dimension, then you could specify
            `pml_width=[20, 20, 0, 20]`. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
            model. Optional, default 20.
        pml_freq:
            A number (int or float) specifying the frequency that you wish
            to use when constructing the PML. This is usually the dominant
            frequency of the source wavelet. Choosing this value poorly
            will result in the edges becoming more reflective. Optional,
            default 25 Hz (assuming `dt` is in seconds).
        max_vel:
            A number (int or float) specifying the maximum velocity, which
            is used when applying the CFL condition and when constructing
            the PML. If not specified, the actual maximum absolute
            wavespeed in the model (or portion of it that is used) will be
            used. The option to specify this is provided to allow you to
            ensure consistency between runs even if there are changes in
            the wavespeed. Optional, default None.
        survey_pad:
            A single value or list of four values, all of which are either
            an int or None, specifying whether the simulation domain
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
            encountered. Specifying a list, in the following order:
            [towards the beginning of the first dimension,
            towards the end of the first dimension,
            towards the beginning of the second dimension,
            towards the end of the second dimension]
            allows the padding in each direction to be controlled. Ints and
            `None` may be mixed, so a `survey_pad` of [5, None, None, 10]
            means that there should be at least 5 cells of padding towards
            the beginning of the first dimension, no restriction of the
            simulation domain towards the end of the first dimension or
            beginning of the second, and 10 cells of padding towards the
            end of the second dimension. If the simulation contains
            one source at [20, 15], one receiver at [10, 15], and the
            model is of shape [40, 50], then the extracted simulation
            domain with this value of `survey_pad` would cover the region
            [5:40, 0:25]. The same simulation domain will be used for all
            shots propagated simultaneously, so if this option is used then
            it is advisable to propagate shots that cover similar regions
            of the model simultaneously so that it provides a benefit (if
            the shots cover opposite ends of a dimension, then that
            whole dimension will be used regardless of what `survey_pad`
            value is used). Optional, default None. Cannot be specified
            if origin is also specified.
        wavefield_0:
            A Tensor specifying the initial wavefield at time step 0. It
            should have three dimensions, with the first dimension being shot
            and the subsequent two corresponding to the two spatial
            dimensions. The spatial shape should be equal to the simulation
            domain, which is the extracted model plus the PML. If two shots
            are being propagated simultaneously in a region of size [20, 30]
            extracted from the model for the simulation, and
            `pml_width=[1, 2, 3, 4]`, then `wavefield_0` should be of shape
            [2, 23, 37]. Optional, default all zeros.
        wavefield_m1:
            A Tensor specifying the initial wavefield at time step -1 (using
            Deepwave's internal time step interval, which may be smaller than
            the user provided one to obey the CFL condition). See the entry for
            `wavefield_0` for more details.
        psiy_m1, psix_m1, zetay_m1, zetax_m1:
            Tensor specifying the initial value for this PML-related
            wavefield at time step -1. See the entry for `wavefield_0`
            for more details.
        origin:
            A list of ints specifying the origin of the provided initial
            wavefields relative to the origin of the model. Only relevant
            if initial wavefields are provided. The origin of a wavefield
            is the cell where the extracted model used for the simulation
            domain began. It does not include the PML. So if a simulation
            is performed using the model region [10:20, 15:30], the origin
            is [10, 15]. Optional, default [0, 0]. Cannot be specified if
            survey_pad is also specified.
        nt:
            If the source amplitudes are not provided then you must instead
            specify the number of time steps to run the simulation for by
            providing an integer for `nt`. You cannot specify both the
            source amplitudes and `nt`.
        model_gradient_sampling_interval:
            An int specifying the number of time steps between
            contributions to the model gradient. The gradient with respect
            to the model is an integral over the backpropagation time steps.
            The time sampling frequency of this integral should be at
            least the Nyquist frequency of the source or data (whichever
            is greater). If this Nyquist frequency is substantially less
            than 1/dt, you may wish to reduce the sampling frequency of
            the integral to reduce computational (especially memory) costs.
            Optional, default 1 (integral is sampled every time step
            interval `dt`).
        freq_taper_frac:
            A float specifying the fraction of the end of the source and
            receiver amplitudes (if present) in the frequency domain to
            cosine taper if they are resampled due to the CFL condition.
            This might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered.
            Default 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the source and receiver amplitudes (if present) before
            resampling and removed afterwards, if they are resampled due to
            the CFL condition, as a fraction of their length. This might be
            useful to reduce wraparound artifacts. A value of 0.1 means that
            zero padding of 10% of the number of time samples will be used.
            Default 0.0.
        time_taper:
            A bool specifying whether to apply a Hann window in time to
            source and receiver amplitudes (if present). This is useful
            during correctness tests of the propagators as it ensures that
            signals taper to zero at their edges in time, avoiding the
            possibility of high frequencies being introduced.

    Returns:
        Tuple[Tensor]:

            wavefield_nt:
                A Tensor containing the wavefield at the final time step.
            wavefield_ntm1:
                A Tensor containing the wavefield at the second-to-last
                time step (using Deepwave's internal time interval, which may
                be smaller than the user provided one to obey the CFL
                condition).
            psiy_ntm1, psix_ntm1, zetay_ntm1, zetax_ntm1:
                Tensor containing the wavefield related to the PML at the
                second-to-last time step.
            receiver_amplitudes:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations. If no receiver locations were specified then
                this Tensor will be empty.

    """
    if pml_config is None:
        pml_config = PMLConfig(_as_list(pml_width, "pml_width", int), pml_freq)
    if survey_config is None:
        survey_config = SurveyConfig(
            source_locations=[source_locations],
            receiver_locations=[receiver_locations],
            source_amplitudes=[source_amplitudes],
            wavefields=[
                wavefield_0,
                wavefield_m1,
                psiy_m1,
                psix_m1,
                zetay_m1,
                zetax_m1,
            ],
            survey_pad=survey_pad,
            origin=origin,
        )
    # Convert Sequence arguments to List for compatibility with
    # setup_propagator
    try:
        min_nonzero_model_vel = v[v != 0].abs().min().item()
    except RuntimeError:
        min_nonzero_model_vel = 0.0
    max_model_vel = v.abs().max().item()
    fd_pad = [accuracy // 2] * 4
    (
        models,
        source_amplitudes_l,
        wavefields,
        sources_i_l,
        receivers_i_l,
        grid_spacing,
        dt,
        nt,
        n_shots,
        step_ratio,
        model_gradient_sampling_interval,
        accuracy,
        pml_width_l,
        pml_freq,
        max_vel,
        resample_config,
        device,
        dtype,
    ) = setup_propagator(
        [v],
        ["replicate"],
        _as_list(grid_spacing, "grid_spacing", float),
        dt,
        survey_config,
        accuracy,
        fd_pad,
        pml_config,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
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
    mask = sources_i_l[0] == IGNORE_LOCATION
    sources_i_masked = sources_i_l[0].clone()
    sources_i_masked[mask] = 0
    # Multiply source amplitudes by -v^2 * dt^2 at source locations
    source_amplitudes_l[0] = (
        -source_amplitudes_l[0]
        * (models[0].view(-1, ny * nx).expand(n_shots, -1).gather(1, sources_i_masked))
        ** 2
        * dt**2
    )

    pml_profiles = set_pml_profiles(
        pml_width_l,
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

    (wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes) = scalar_func(
        *models,
        *source_amplitudes_l,
        *wavefields,
        *pml_profiles,
        *sources_i_l,
        *receivers_i_l,
        *grid_spacing,
        dt,
        nt,
        step_ratio * model_gradient_sampling_interval,
        accuracy,
        pml_width_l,
        n_shots,
    )

    receiver_amplitudes = downsample_and_movedim(
        receiver_amplitudes,
        resample_config.step_ratio,
        resample_config.freq_taper_frac,
        resample_config.time_pad_frac,
        resample_config.time_taper,
    )
    return wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes


class ScalarForwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        v: Tensor,
        source_amplitudes: Tensor,
        wfc: Tensor,
        wfp: Tensor,
        psiy: Tensor,
        psix: Tensor,
        zetay: Tensor,
        zetax: Tensor,
        ay: Tensor,
        ax: Tensor,
        by: Tensor,
        bx: Tensor,
        dbydy: Tensor,
        dbxdx: Tensor,
        sources_i: Tensor,
        receivers_i: Tensor,
        dy: float,
        dx: float,
        dt: float,
        nt: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        n_shots: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        forward = None # Initialize to None to prevent E0601
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
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype, size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = zero_interior(psiy, fd_pad, pml_width, True)
        psix = zero_interior(psix, fd_pad, pml_width, False)
        zetay = zero_interior(zetay, fd_pad, pml_width, True)
        zetax = zero_interior(zetax, fd_pad, pml_width, False)

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
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = dll.scalar_iso_2_float_forward_cuda
                elif accuracy == 4:
                    forward = dll.scalar_iso_4_float_forward_cuda
                elif accuracy == 6:
                    forward = dll.scalar_iso_6_float_forward_cuda
                else:
                    forward = dll.scalar_iso_8_float_forward_cuda
            else:
                if accuracy == 2:
                    forward = dll.scalar_iso_2_double_forward_cuda
                elif accuracy == 4:
                    forward = dll.scalar_iso_4_double_forward_cuda
                elif accuracy == 6:
                    forward = dll.scalar_iso_6_double_forward_cuda
                else:
                    forward = dll.scalar_iso_8_double_forward_cuda
        else:
            if USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = dll.scalar_iso_2_float_forward_cpu
                elif accuracy == 4:
                    forward = dll.scalar_iso_4_float_forward_cpu
                elif accuracy == 6:
                    forward = dll.scalar_iso_6_float_forward_cpu
                else:
                    forward = dll.scalar_iso_8_float_forward_cpu
            else:
                if accuracy == 2:
                    forward = dll.scalar_iso_2_double_forward_cpu
                elif accuracy == 4:
                    forward = dll.scalar_iso_4_double_forward_cpu
                elif accuracy == 6:
                    forward = dll.scalar_iso_6_double_forward_cpu
                else:
                    forward = dll.scalar_iso_8_double_forward_cpu

        if wfc.numel() > 0 and nt > 0:
            start_t = 0
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
                nt,
                n_shots,
                ny,
                nx,
                n_sources_per_shot,
                n_receivers_per_shot,
                step_ratio,
                v.requires_grad,
                v_batched,
                start_t,
                pml_y0,
                pml_y1,
                pml_x0,
                pml_x1,
                aux,
            )

        ctx.dwdv = dwdv

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return (
                wfc[s],
                wfp[s],
                psiy[s],
                psix[s],
                zetay[s],
                zetax[s],
                receiver_amplitudes,
            )
        return (
            wfp[s],
            wfc[s],
            psiyn[s],
            psixn[s],
            zetay[s],
            zetax[s],
            receiver_amplitudes,
        )

    @staticmethod
    def backward(
        ctx: Any,
        gwfc: Tensor,
        gwfp: Tensor,
        gpsiy: Tensor,
        gpsix: Tensor,
        gzetay: Tensor,
        gzetax: Tensor,
        grad_r: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
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

        return (
            ScalarBackwardFunc.apply(
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
            )
            + (None,) * 16
        )


class ScalarBackwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        gwfc: Tensor,
        gwfp: Tensor,
        gpsiy: Tensor,
        gpsix: Tensor,
        gzetay: Tensor,
        gzetax: Tensor,
        grad_r: Tensor,
        v: Tensor,
        ay: Tensor,
        ax: Tensor,
        by: Tensor,
        bx: Tensor,
        dbydy: Tensor,
        dbxdx: Tensor,
        sources_i: Tensor,
        receivers_i: Tensor,
        dwdv: Tensor,
        source_amplitudes: Tensor,
        wfc: Tensor,
        wfp: Tensor,
        psiy: Tensor,
        psix: Tensor,
        zetay: Tensor,
        zetax: Tensor,
        dy: float,
        dx: float,
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        source_amplitudes_requires_grad: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        backward = None # Initialize to None to prevent E0601
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
        gwfc = create_or_pad(gwfc, fd_pad, v.device, v.dtype, size_with_batch)
        gwfp = create_or_pad(gwfp, fd_pad, v.device, v.dtype, size_with_batch)
        gpsiy = create_or_pad(gpsiy, fd_pad, v.device, v.dtype, size_with_batch)
        gpsix = create_or_pad(gpsix, fd_pad, v.device, v.dtype, size_with_batch)
        gzetay = create_or_pad(gzetay, fd_pad, v.device, v.dtype, size_with_batch)
        gzetax = create_or_pad(gzetax, fd_pad, v.device, v.dtype, size_with_batch)
        gpsiy = zero_interior(gpsiy, fd_pad, pml_width, True)
        gpsix = zero_interior(gpsix, fd_pad, pml_width, False)
        gzetay = zero_interior(gzetay, fd_pad, pml_width, True)
        gzetax = zero_interior(gzetax, fd_pad, pml_width, False)

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
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = dll.scalar_iso_2_float_backward_cuda
                elif accuracy == 4:
                    backward = dll.scalar_iso_4_float_backward_cuda
                elif accuracy == 6:
                    backward = dll.scalar_iso_6_float_backward_cuda
                else:
                    backward = dll.scalar_iso_8_float_backward_cuda
            else:
                if accuracy == 2:
                    backward = dll.scalar_iso_2_double_backward_cuda
                elif accuracy == 4:
                    backward = dll.scalar_iso_4_double_backward_cuda
                elif accuracy == 6:
                    backward = dll.scalar_iso_6_double_backward_cuda
                else:
                    backward = dll.scalar_iso_8_double_backward_cuda
        else:
            if USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad and not v_batched and aux > 1 and USE_OPENMP:
                grad_v_tmp.resize_(aux, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = dll.scalar_iso_2_float_backward_cpu
                elif accuracy == 4:
                    backward = dll.scalar_iso_4_float_backward_cpu
                elif accuracy == 6:
                    backward = dll.scalar_iso_6_float_backward_cpu
                else:
                    backward = dll.scalar_iso_8_float_backward_cpu
            else:
                if accuracy == 2:
                    backward = dll.scalar_iso_2_double_backward_cpu
                elif accuracy == 4:
                    backward = dll.scalar_iso_4_double_backward_cpu
                elif accuracy == 6:
                    backward = dll.scalar_iso_6_double_backward_cpu
                else:
                    backward = dll.scalar_iso_8_double_backward_cpu

        v2dt2 = v**2 * dt**2
        gwfp = -gwfp

        if gwfc.numel() > 0 and nt > 0:
            start_t = 0
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
                nt,
                n_shots,
                ny,
                nx,
                n_sources_per_shot * source_amplitudes_requires_grad,
                n_receivers_per_shot,
                step_ratio,
                v.requires_grad,
                v_batched,
                start_t,
                pml_y0,
                pml_y1,
                pml_x0,
                pml_x1,
                aux,
            )

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
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
        return (
            grad_v,
            grad_f,
            gwfp[s],
            -gwfc[s],
            gpsiyn[s],
            gpsixn[s],
            gzetayn[s],
            gzetaxn[s],
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        ggv: Tensor,
        ggf: Tensor,
        ggwfc: Tensor,
        ggwfp: Tensor,
        ggpsiy: Tensor,
        ggpsix: Tensor,
        ggzetay: Tensor,
        ggzetax: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
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
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype, size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype, size_with_batch)
        ggwfc = create_or_pad(ggwfc, fd_pad, v.device, v.dtype, size_with_batch)
        ggwfp = create_or_pad(ggwfp, fd_pad, v.device, v.dtype, size_with_batch)
        ggpsiy = create_or_pad(ggpsiy, fd_pad, v.device, v.dtype, size_with_batch)
        ggpsix = create_or_pad(ggpsix, fd_pad, v.device, v.dtype, size_with_batch)
        ggzetay = create_or_pad(ggzetay, fd_pad, v.device, v.dtype, size_with_batch)
        ggzetax = create_or_pad(ggzetax, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = zero_interior(psiy, fd_pad, pml_width, True)
        psix = zero_interior(psix, fd_pad, pml_width, False)
        zetay = zero_interior(zetay, fd_pad, pml_width, True)
        zetax = zero_interior(zetax, fd_pad, pml_width, False)
        ggpsiy = zero_interior(ggpsiy, fd_pad, pml_width, True)
        ggpsix = zero_interior(ggpsix, fd_pad, pml_width, False)
        ggzetay = zero_interior(ggzetay, fd_pad, pml_width, True)
        ggzetax = zero_interior(ggzetax, fd_pad, pml_width, False)

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
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = dll.scalar_born_iso_2_float_forward_cuda
                elif accuracy == 4:
                    forward = dll.scalar_born_iso_4_float_forward_cuda
                elif accuracy == 6:
                    forward = dll.scalar_born_iso_6_float_forward_cuda
                else:
                    forward = dll.scalar_born_iso_8_float_forward_cuda
            else:
                if accuracy == 2:
                    forward = dll.scalar_born_iso_2_double_forward_cuda
                elif accuracy == 4:
                    forward = dll.scalar_born_iso_4_double_forward_cuda
                elif accuracy == 6:
                    forward = dll.scalar_born_iso_6_double_forward_cuda
                else:
                    forward = dll.scalar_born_iso_8_double_forward_cuda
        else:
            if USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = dll.scalar_born_iso_2_float_forward_cpu
                elif accuracy == 4:
                    forward = dll.scalar_born_iso_4_float_forward_cpu
                elif accuracy == 6:
                    forward = dll.scalar_born_iso_6_float_forward_cpu
                else:
                    forward = dll.scalar_born_iso_8_float_forward_cpu
            else:
                if accuracy == 2:
                    forward = dll.scalar_born_iso_2_double_forward_cpu
                elif accuracy == 4:
                    forward = dll.scalar_born_iso_4_double_forward_cpu
                elif accuracy == 6:
                    forward = dll.scalar_born_iso_6_double_forward_cpu
                else:
                    forward = dll.scalar_born_iso_8_double_forward_cpu

        if wfc.numel() > 0 and nt > 0:
            start_t = 0
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
                start_t,
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

        wfc = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(
            torch.empty(0), fd_pad, v.device, v.dtype, size_with_batch
        )
        zetax = create_or_pad(
            torch.empty(0), fd_pad, v.device, v.dtype, size_with_batch
        )
        psiy = zero_interior(psiy, fd_pad, pml_width, True)
        psix = zero_interior(psix, fd_pad, pml_width, False)
        zetay = zero_interior(zetay, fd_pad, pml_width, True)
        zetax = zero_interior(zetax, fd_pad, pml_width, False)
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        zetayn = torch.zeros_like(zetay)
        zetaxn = torch.zeros_like(zetax)
        gwfc = create_or_pad(gwfc, fd_pad, v.device, v.dtype, size_with_batch)
        gwfp = create_or_pad(gwfp, fd_pad, v.device, v.dtype, size_with_batch)
        gpsiy = create_or_pad(gpsiy, fd_pad, v.device, v.dtype, size_with_batch)
        gpsix = create_or_pad(gpsix, fd_pad, v.device, v.dtype, size_with_batch)
        gzetay = create_or_pad(gzetay, fd_pad, v.device, v.dtype, size_with_batch)
        gzetax = create_or_pad(gzetax, fd_pad, v.device, v.dtype, size_with_batch)
        gpsiy = zero_interior(gpsiy, fd_pad, pml_width, True)
        gpsix = zero_interior(gpsix, fd_pad, pml_width, False)
        gzetay = zero_interior(gzetay, fd_pad, pml_width, True)
        gzetax = zero_interior(gzetax, fd_pad, pml_width, False)
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
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = dll.scalar_born_iso_2_float_backward_cuda
                elif accuracy == 4:
                    backward = dll.scalar_born_iso_4_float_backward_cuda
                elif accuracy == 6:
                    backward = dll.scalar_born_iso_6_float_backward_cuda
                else:
                    backward = dll.scalar_born_iso_8_float_backward_cuda
            else:
                if accuracy == 2:
                    backward = dll.scalar_born_iso_2_double_backward_cuda
                elif accuracy == 4:
                    backward = dll.scalar_born_iso_4_double_backward_cuda
                elif accuracy == 6:
                    backward = dll.scalar_born_iso_6_double_backward_cuda
                else:
                    backward = dll.scalar_born_iso_8_double_backward_cuda
        else:
            if USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad and not v_batched and aux > 1 and USE_OPENMP:
                grad_v_tmp.resize_(aux, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = dll.scalar_born_iso_2_float_backward_cpu
                elif accuracy == 4:
                    backward = dll.scalar_born_iso_4_float_backward_cpu
                elif accuracy == 6:
                    backward = dll.scalar_born_iso_6_float_backward_cpu
                else:
                    backward = dll.scalar_born_iso_8_float_backward_cpu
            else:
                if accuracy == 2:
                    backward = dll.scalar_born_iso_2_double_backward_cpu
                elif accuracy == 4:
                    backward = dll.scalar_born_iso_4_double_backward_cpu
                elif accuracy == 6:
                    backward = dll.scalar_born_iso_6_double_backward_cpu
                else:
                    backward = dll.scalar_born_iso_8_double_backward_cpu

        gwfp = -gwfp

        if wfc.numel() > 0 and nt > 0 and v.requires_grad:
            start_t = 0
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
                start_t,
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
        )


def scalar_func(*args: Any) -> Tuple[Tensor, ...]:
    return ScalarForwardFunc.apply(*args)

"""This script demonstrates two methods for implementing custom imaging conditions
in Deepwave: one using PyTorch's backward hooks and another using a lower-level
internal interface. It compares their results and performance.
"""

import time

import matplotlib.pyplot as plt
import torch
from torch.autograd.function import once_differentiable

import deepwave
from deepwave.common import (
    create_or_pad,
    diff,
    downsample_and_movedim,
    setup_propagator,
    zero_interior,
)

ny = 200
nx = 200
dx = 5.0
nt = 400
dt = 0.004
freq = 25
peak_time = 1.3 / freq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_shots = 1
n_receivers_per_shot = nx
acquisition_depth = 10

v_init = 1550 * torch.ones(ny, nx, device=device)
v_true = v_init.clone()
v_true[-20:] = 1650
max_vel = 1650

source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[..., 0] = acquisition_depth
source_locations[:, 0, 1] = torch.linspace(
    0,
    nx - 1,
    n_shots,
    dtype=torch.long,
    device=device,
)
receiver_locations = torch.zeros(
    n_shots,
    n_receivers_per_shot,
    2,
    dtype=torch.long,
    device=device,
)
receiver_locations[..., 0] = acquisition_depth
receiver_locations[0, :, 1] = torch.linspace(
    0,
    nx - 1,
    n_receivers_per_shot,
    dtype=torch.long,
    device=device,
)
receiver_locations[..., 1] = receiver_locations[:1, :, 1].repeat(n_shots, 1)

source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, 1, 1).to(device)
)

# Make data with true data
d_true = deepwave.scalar(
    v_true,
    dx,
    dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    max_vel=max_vel,
)[-1]


def method_1():
    # Upsample source so that we can call propagator on chunks of it
    n_segments = source_amplitudes.shape[-1]
    dt_up, step_ratio = deepwave.common.cfl_condition(dx, dx, dt, max_vel)
    source_amplitudes_up = deepwave.common.upsample(source_amplitudes, step_ratio)

    loss_fn = torch.nn.MSELoss()
    v = v_init.clone()

    def wrap(chunk, wfc, wfp, psiy, psix, zetay, zetax):
        return deepwave.scalar(
            v,
            dx,
            dt_up,
            source_amplitudes=chunk,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wfc,
            wavefield_m1=wfp,
            psiy_m1=psiy,
            psix_m1=psix,
            zetay_m1=zetay,
            zetax_m1=zetax,
            max_vel=max_vel,
        )

    # Create zero-filled initial wavefields to pass to propagator
    pml_width = 20
    wavefield_size = [n_shots, ny + 2 * pml_width, nx + 2 * pml_width]
    # We set requires_grad so that we will be able to call backward
    wavefield_0 = torch.zeros(*wavefield_size, device=device, requires_grad=True)
    wavefield_m1 = torch.zeros(*wavefield_size, device=device)
    psiy_m1 = torch.zeros(*wavefield_size, device=device)
    psix_m1 = torch.zeros(*wavefield_size, device=device)
    zetay_m1 = torch.zeros(*wavefield_size, device=device)
    zetax_m1 = torch.zeros(*wavefield_size, device=device)

    # Storage for second time derivative of forward wavefield
    source_wavefields = torch.zeros(n_segments, n_shots, ny, nx, device=device)
    receiver_amplitudes = torch.zeros(
        n_shots,
        n_receivers_per_shot,
        n_segments * step_ratio,
        device=device,
    )
    # Storage for gradient with respect to model
    out = [torch.zeros(ny, nx, device=device)]
    k = 0
    loss = 0
    # Loop over chunks of time, probably more than one timestep in each as
    # we don't need to apply the imaging condition more frequently than the
    # original (before upsampling) source is sampled.
    # We propagate twice, first up to the last timestep, and then the last
    # timestep. This is because we need the wavefield at three timesteps
    # in order to calculate its second time derivative, but the propagator
    # only stores two timesteps, so we record the wavefield before the
    # final timestep propagation, and then the two wavefields after it, to
    # get the three wavefields needed for the second time derivative.
    for i, chunk in enumerate(torch.chunk(source_amplitudes_up, n_segments, dim=-1)):
        # We first propagate up to the last timestep in the chunk
        if chunk.shape[-1] > 1:
            # Propagate forward
            (
                wavefield_0,
                wavefield_m1,
                psiy_m1,
                psix_m1,
                zetay_m1,
                zetax_m1,
                receiver_amplitudes_chunk,
            ) = wrap(
                chunk[..., :-1],
                wavefield_0,
                wavefield_m1,
                psiy_m1,
                psix_m1,
                zetay_m1,
                zetax_m1,
            )
            receiver_amplitudes[..., k : k + chunk.shape[-1] - 1] = (
                receiver_amplitudes_chunk
            )

        # We then save the source (forward) wavefield (part 1)
        source_wavefields[i] = wavefield_m1.detach()[
            :,
            pml_width:-pml_width,
            pml_width:-pml_width,
        ]

        # Then propagate forward for the last timestep in the chunk
        (
            wavefield_0,
            wavefield_m1,
            psiy_m1,
            psix_m1,
            zetay_m1,
            zetax_m1,
            receiver_amplitudes_chunk,
        ) = wrap(
            chunk[..., -1:],
            wavefield_0,
            wavefield_m1,
            psiy_m1,
            psix_m1,
            zetay_m1,
            zetax_m1,
        )
        receiver_amplitudes[..., k + chunk.shape[-1] - 1 : k + chunk.shape[-1]] = (
            receiver_amplitudes_chunk
        )

        # And then combine with part 1 of saving the source wavefield to
        # actually calculate its second time derivative
        source_wavefields[i] += (
            wavefield_0.detach()[:, pml_width:-pml_width, pml_width:-pml_width]
            - 2 * wavefield_m1.detach()[:, pml_width:-pml_width, pml_width:-pml_width]
        )

        # Create a hook to apply the imaging condition during backpropagation
        def hook_closure():
            bi = i

            def hook(x):
                if x is not None:
                    # Apply the imaging condition
                    out[0] += (
                        2
                        / v.detach()
                        * step_ratio
                        * (
                            source_wavefields[bi]
                            * x.detach()[:, pml_width:-pml_width, pml_width:-pml_width]
                        ).sum(dim=0)
                    )

            return hook

        wavefield_0.register_hook(hook_closure())
        k += chunk.shape[-1]

    loss = 1e6 * loss_fn(
        deepwave.common.downsample(receiver_amplitudes, step_ratio),
        d_true,
    )
    loss.backward()

    return out[0]


# Method 2 is broken into parts. First we do some setup of the propagator and then
# call the Autograd Function that is defined further down
def method2_scalar(
    v,
    grid_spacing,
    dt,
    source_amplitudes=None,
    source_locations=None,
    receiver_locations=None,
    accuracy=4,
    pml_width=20,
    pml_freq=None,
    max_vel=None,
    survey_pad=None,
    wavefield_0=None,
    wavefield_m1=None,
    psiy_m1=None,
    psix_m1=None,
    zetay_m1=None,
    zetax_m1=None,
    origin=None,
    nt=None,
    model_gradient_sampling_interval=1,
    freq_taper_frac=0.0,
    time_pad_frac=0.0,
):
    (
        models,
        source_amplitudes_l,
        wavefields,
        pml_profiles,
        sources_i_l,
        receivers_i_l,
        dy,
        dx,
        dt,
        nt,
        n_shots,
        step_ratio,
        model_gradient_sampling_interval,
        accuracy,
        pml_width_list,
    ) = setup_propagator(
        [v],
        "scalar",
        grid_spacing,
        dt,
        [wavefield_0, wavefield_m1, psiy_m1, psix_m1, zetay_m1, zetax_m1],
        [source_amplitudes],
        [source_locations],
        [receiver_locations],
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        survey_pad,
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
    )
    v = models[0]
    wfc, wfp, psiy, psix, zetay, zetax = wavefields
    source_amplitudes = source_amplitudes_l[0]
    sources_i = sources_i_l[0]
    receivers_i = receivers_i_l[0]
    ay, ax, by, bx = pml_profiles
    dbydy = diff(by, accuracy, dy)
    dbxdx = diff(bx, accuracy, dx)

    (wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes) = method2_func(
        v,
        source_amplitudes,
        wfc,
        wfp,
        psiy,
        psix,
        zetay,
        zetax,
        ay,
        ax,
        by,
        bx,
        dbydy,
        dbxdx,
        sources_i,
        receivers_i,
        dy,
        dx,
        dt,
        nt,
        step_ratio * model_gradient_sampling_interval,
        accuracy,
        pml_width_list,
        n_shots,
    )

    receiver_amplitudes = downsample_and_movedim(
        receiver_amplitudes,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
    )

    return wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes


# Using an Autograd Function enables us to define our own forward and backward
class Method2ForwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v,
        source_amplitudes,
        wfc,
        wfp,
        psiy,
        psix,
        zetay,
        zetax,
        ay,
        ax,
        by,
        bx,
        dbydy,
        dbxdx,
        sources_i,
        receivers_i,
        dy,
        dx,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        n_shots,
    ):
        # Ensure that the Tensors are contiguous in memory as the C/CUDA code
        # assumes that they are
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

        # Create the wavefields, or add padding (for the finite difference
        # stencil) if they already exist
        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape)
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype, size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype, size_with_batch)
        # Zero the interior of the PML-related wavefields
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)

        # Define some needed values
        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots

        # Allocate some temporary and output Tensors. dwdv stores snapshots
        # from forward propagation for use during backpropagation.
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        dwdv = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        # Set the coordinates of the edges of where the PML calculations
        # need to be performed.
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        # Define quantities that depend on whether we are running on a CPU or GPU
        if v.is_cuda:
            aux = v.get_device()  # the CUDA device number
            if v.requires_grad:
                dwdv.resize_(nt // step_ratio, n_shots, *v.shape)
                dwdv.fill_(0)
            if receivers_i is not None:
                receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            # Select the CUDA function that will be used to propagate
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_iso_6_float_forward
                else:
                    forward = deepwave.dll_cuda.scalar_iso_8_float_forward
            elif accuracy == 2:
                forward = deepwave.dll_cuda.scalar_iso_2_double_forward
            elif accuracy == 4:
                forward = deepwave.dll_cuda.scalar_iso_4_double_forward
            elif accuracy == 6:
                forward = deepwave.dll_cuda.scalar_iso_6_double_forward
            else:
                forward = deepwave.dll_cuda.scalar_iso_8_double_forward
        else:  # Running on CPU
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad:
                dwdv.resize_(n_shots, nt // step_ratio, *v.shape)
                dwdv.fill_(0)
            if receivers_i is not None:
                receiver_amplitudes.resize_(n_shots, nt, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_iso_6_float_forward
                else:
                    forward = deepwave.dll_cpu.scalar_iso_8_float_forward
            elif accuracy == 2:
                forward = deepwave.dll_cpu.scalar_iso_2_double_forward
            elif accuracy == 4:
                forward = deepwave.dll_cpu.scalar_iso_4_double_forward
            elif accuracy == 6:
                forward = deepwave.dll_cpu.scalar_iso_6_double_forward
            else:
                forward = deepwave.dll_cpu.scalar_iso_8_double_forward

        # Call the C/CUDA function to propagate forward
        if wfc.numel() > 0 and nt > 0:
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
                pml_y0,
                pml_y1,
                pml_x0,
                pml_x1,
                aux,
            )

        # Save data needed for backpropagation
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
                dwdv,
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

        # Remove the padding added for the finite difference stencil
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
    @once_differentiable
    def backward(ctx, wfc, wfp, psiy, psix, zetay, zetax, grad_r):
        (v, ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i, dwdv) = (
            ctx.saved_tensors
        )

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

        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width

        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_receivers_per_shot = receivers_i.numel() // n_shots
        fd_pad = accuracy // 2

        size_with_batch = (n_shots, *v.shape)
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype, size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype, size_with_batch)
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)

        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        zetayn = torch.zeros_like(zetay)
        zetaxn = torch.zeros_like(zetax)
        grad_v = torch.zeros_like(v)
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_iso_6_float_backward
                else:
                    backward = deepwave.dll_cuda.scalar_iso_8_float_backward
            elif accuracy == 2:
                backward = deepwave.dll_cuda.scalar_iso_2_double_backward
            elif accuracy == 4:
                backward = deepwave.dll_cuda.scalar_iso_4_double_backward
            elif accuracy == 6:
                backward = deepwave.dll_cuda.scalar_iso_6_double_backward
            else:
                backward = deepwave.dll_cuda.scalar_iso_8_double_backward
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_iso_6_float_backward
                else:
                    backward = deepwave.dll_cpu.scalar_iso_8_float_backward
            elif accuracy == 2:
                backward = deepwave.dll_cpu.scalar_iso_2_double_backward
            elif accuracy == 4:
                backward = deepwave.dll_cpu.scalar_iso_4_double_backward
            elif accuracy == 6:
                backward = deepwave.dll_cpu.scalar_iso_6_double_backward
            else:
                backward = deepwave.dll_cpu.scalar_iso_8_double_backward

        v2dt2 = v**2 * dt**2
        wfp = -wfp
        # Backward loop over timesteps. As in method 1, we only need to
        # apply the imaging condition at the same frequency as the
        # sampling of the input source, so the propagator will actually
        # run "step_ratio" inner timesteps each time it is called.
        for i in range(nt // step_ratio - 1, -1, -1):
            # Apply the imaging condition and extract the adjoint sources
            # for this time chunk. These depend on whether we are running
            # on a CPU or GPU as the order of dimensions are different.
            # On a GPU the dwdv Tensor has dimensions
            # [nt, n_shots, ny, nx], while on a CPU it is
            # [n_shots, nt, ny, nx]. Similarly, on a GPU the receiver
            # data will have dimensions
            # [nt, n_shots, n_receivers_per_shot], while on a CPU it is
            # [n_shots, nt, n_receivers_per_shot].
            # The dwdv Tensor stores the second time derivative of the
            # forward wavefield at snapshots in time.
            if v.is_cuda:
                grad_v += (wfc * dwdv[i]).sum(dim=0) * step_ratio
                chunk = grad_r[i * step_ratio : (i + 1) * step_ratio].contiguous()
            else:
                grad_v += (wfc * dwdv[:, i]).sum(dim=0) * step_ratio
                chunk = grad_r[:, i * step_ratio : (i + 1) * step_ratio].contiguous()
            backward(
                v2dt2.data_ptr(),
                chunk.data_ptr(),
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
                torch.empty(0).data_ptr(),
                torch.empty(0).data_ptr(),
                torch.empty(0).data_ptr(),
                torch.empty(0).data_ptr(),
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
                step_ratio,
                n_shots,
                ny,
                nx,
                0,
                n_receivers_per_shot,
                step_ratio,
                False,
                pml_y0,
                pml_y1,
                pml_x0,
                pml_x1,
                aux,
            )
            # We need to swap some Tensors so that we can flip-flop
            # between memory locations at each timestep
            if step_ratio % 2 != 0:
                wfc, wfp = wfp, wfc
                psiy, psiyn = psiyn, psiy
                psix, psixn = psixn, psix
                zetax, zetaxn = zetaxn, zetax
                zetay, zetayn = zetayn, zetay

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        return (
            grad_v,
            None,
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
            None,
            None,
            None,
            None,
            None,
        )


def method2_func(*args):
    return Method2ForwardFunc.apply(*args)


def method_2():
    loss_fn = torch.nn.MSELoss()
    v = v_init.clone().requires_grad_()

    receiver_amplitudes = method2_scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=max_vel,
    )[-1]
    loss = 1e6 * loss_fn(receiver_amplitudes, d_true)
    loss.backward()

    return v.grad.detach()


def regular_deepwave():
    loss_fn = torch.nn.MSELoss()
    v = v_init.clone().requires_grad_()

    receiver_amplitudes = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        max_vel=max_vel,
    )[-1]
    loss = 1e6 * loss_fn(receiver_amplitudes, d_true)
    loss.backward()

    return v.grad.detach()


t0 = time.time()
method_1_grad = method_1()
print("Method 1:", time.time() - t0)
t0 = time.time()
method_2_grad = method_2()
print("Method 2:", time.time() - t0)
t0 = time.time()
regular_deepwave_grad = regular_deepwave()
print("Regular Deepwave:", time.time() - t0)

# We ignore the edges of the models as Method 2 and regular Deepwave
# include the effect of the PML in the gradient, but Method 1 does not.
_, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
vmax = 0.0003
im = ax[0].imshow(
    method_1_grad.cpu()[1:-1, 1:-1],
    aspect="auto",
    cmap="gray",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, ax=ax[0])
ax[0].set_title("Method 1")
im = ax[1].imshow(
    method_2_grad.cpu()[1:-1, 1:-1],
    aspect="auto",
    cmap="gray",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, ax=ax[1])
ax[1].set_title("Method 2")
im = ax[2].imshow(
    regular_deepwave_grad.cpu()[1:-1, 1:-1],
    aspect="auto",
    cmap="gray",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, ax=ax[2])
ax[2].set_title("Regular Deepwave")
plt.tight_layout()
plt.savefig("example_custom_imaging_condition.jpg")

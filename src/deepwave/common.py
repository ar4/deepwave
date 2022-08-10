import math
import warnings
from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor


def setup_propagator(v: Tensor,
                     grid_spacing: Union[int, float,
                                         List[float], Tensor],
                     dt: float,
                     other_models: List[Tensor],
                     other_model_modes: List[str],
                     wavefields: List[Optional[Tensor]],
                     source_amplitudes: Optional[Tensor] = None,
                     source_locations: Optional[Tensor] = None,
                     receiver_locations: Optional[Tensor] = None,
                     accuracy: int = 4,
                     pml_width: Union[int, List[int]] = 20,
                     pml_freq: Optional[float] = None,
                     max_vel: Optional[float] = None,
                     survey_pad: Optional[Union[int,
                                          List[Optional[int]]]] = None,
                     origin: Optional[List[int]] = None,
                     nt: Optional[int] = None,
                     model_gradient_sampling_interval: int = 1,
                     freq_taper_frac: float = 0.0,
                     time_pad_frac: float = 0.0) -> Tuple[
                        Tensor, List[Tensor], Tensor, List[Tensor],
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        float, float, float, int, int, int, int, int,
                        List[int]]:
    # Check inputs
    check_inputs(source_amplitudes, source_locations, receiver_locations,
                 wavefields, accuracy, nt, v)

    if nt is None:
        nt = 0
        if source_amplitudes is not None:
            nt = source_amplitudes.shape[-1]
    device = v.device
    dtype = v.dtype
    dy, dx = set_dx(grid_spacing)
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(4)]
    fd_pad = accuracy // 2
    pad = [fd_pad + width for width in pml_width]
    models, locations = extract_survey(
        [v] + list(other_models),
        [source_locations, receiver_locations],
        survey_pad, wavefields, origin, pml_width
    )
    v = models[0]
    other_models = models[1:]
    source_locations, receiver_locations = locations
    if max_vel is None:
        max_vel = v.abs().max().item()
    else:
        max_vel = abs(max_vel)
        if max_vel < v.abs().max().item():
            warnings.warn("max_vel is less than the actual maximum velocity")
    check_points_per_wavelength(v, pml_freq, dy, dx, dt)
    v = pad_model(v, pad)
    for i in range(len(other_models)):
        other_models[i] = pad_model(other_models[i], pad,
                                    mode=other_model_modes[i])
    dt, step_ratio = cfl_condition(dy, dx, dt, max_vel)
    if source_amplitudes is not None and source_locations is not None:
        source_locations = pad_locations(source_locations, pad)
        sources_i = location_to_index(source_locations, v.shape[1])
        source_amplitudes = (
            -source_amplitudes * v[source_locations[..., 0],
                                   source_locations[..., 1],
                                   None] ** 2 * dt**2
        )
        source_amplitudes = upsample(source_amplitudes, step_ratio,
                                     freq_taper_frac=freq_taper_frac,
                                     time_pad_frac=time_pad_frac)
    else:
        sources_i = None
    if receiver_locations is not None:
        receiver_locations = pad_locations(receiver_locations, pad)
        receivers_i = location_to_index(receiver_locations, v.shape[1])
    else:
        receivers_i = None
    n_batch = get_n_batch([source_locations, receiver_locations] +
                          list(wavefields))
    ay, ax, by, bx = \
        setup_pml(pml_width, fd_pad, dt, v, max_vel, pml_freq)
    nt *= step_ratio

    if source_amplitudes is not None:
        if source_amplitudes.device == torch.device('cpu'):
            source_amplitudes = torch.movedim(source_amplitudes, -1, 1)
        else:
            source_amplitudes = torch.movedim(source_amplitudes, -1, 0)

    v = convert_to_contiguous(v)
    for i in range(len(other_models)):
        other_models[i] = convert_to_contiguous(other_models[i])
    source_amplitudes = (convert_to_contiguous(source_amplitudes)
                         .to(dtype).to(device))
    contiguous_wavefields = []
    for wavefield in wavefields:
        contiguous_wavefields.append(convert_to_contiguous(wavefield))
    sources_i = convert_to_contiguous(sources_i)
    receivers_i = convert_to_contiguous(receivers_i)
    ay = convert_to_contiguous(ay)
    ax = convert_to_contiguous(ax)
    by = convert_to_contiguous(by)
    bx = convert_to_contiguous(bx)
    return (v, other_models, source_amplitudes, contiguous_wavefields,
            ay, ax, by, bx, sources_i.long(), receivers_i.long(),
            dy, dx, dt, nt, n_batch,
            step_ratio, model_gradient_sampling_interval,
            accuracy, pml_width)


def downsample_and_movedim(receiver_amplitudes: Tensor,
                           step_ratio: int, freq_taper_frac: float = 0.0,
                           time_pad_frac: float = 0.0) -> Tensor:
    if receiver_amplitudes.numel() > 0:
        if receiver_amplitudes.device == torch.device('cpu'):
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 1, -1)
        else:
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 0, -1)
        receiver_amplitudes = downsample(receiver_amplitudes, step_ratio,
                                         freq_taper_frac=freq_taper_frac,
                                         time_pad_frac=time_pad_frac)
    return receiver_amplitudes


def set_dx(dx: Union[int, float, List[float],
                     Tensor]) -> Tuple[float, float]:
    if isinstance(dx, int):
        return float(dx), float(dx)
    if isinstance(dx, float):
        return dx, dx
    if (isinstance(dx, list) and len(dx) == 2):
        return float(dx[0]), float(dx[1])
    if isinstance(dx, torch.Tensor) and dx.shape == (2,):
        return float(dx[0].item()), float(dx[1].item())
    raise RuntimeError("Expected dx to be a real number or a list of "
                       "two real numbers")


def check_inputs(source_amplitudes: Optional[Tensor],
                 source_locations: Optional[Tensor],
                 receiver_locations: Optional[Tensor],
                 wavefields: List[Optional[Tensor]],
                 accuracy: int, nt: Optional[int], model: Tensor) -> None:
    if source_amplitudes is not None or source_locations is not None:
        if source_amplitudes is None or source_locations is None:
            raise RuntimeError("source_locations and source_amplitudes must "
                               "both be None or non-None")
    if source_amplitudes is not None and source_locations is not None:
        if source_amplitudes.ndim != 3:
            raise RuntimeError("source_amplitudes should have dimensions "
                               "[shot, source, time]")
        if source_locations.ndim != 3:
            raise RuntimeError("source_locations should have dimensions "
                               "[shot, source, dimension]")
        if source_amplitudes.shape[0] != source_locations.shape[0]:
            raise RuntimeError("Expected source_amplitudes.shape[0] == "
                               "source_locations.shape[0]")
        if source_amplitudes.shape[1] != source_locations.shape[1]:
            raise RuntimeError("Expected source_amplitudes.shape[1] == "
                               "source_locations.shape[1]")
        if source_locations.shape[2] != 2:
            raise RuntimeError("Expected source_locations.shape[2] == 2")
        if source_amplitudes.device != model.device:
            raise RuntimeError("Expected source_amplitudes and the model "
                               "to be on the same device")
        if source_locations.device != model.device:
            raise RuntimeError("Expected source_locations and the model "
                               "to be on the same device")
        if source_amplitudes.dtype != model.dtype:
            raise RuntimeError("Expected source_amplitudes and the model "
                               "to have the same dtype")
        if source_amplitudes.numel() == 0:
            raise RuntimeError("source_amplitudes has zero elements. Please "
                               "instead set it to None if you do not wish to "
                               "use a source.")
    if receiver_locations is not None:
        if receiver_locations.ndim != 3:
            raise RuntimeError("receiver_locations should have dimensions "
                               "[shot, receiver, dimension]")
        if receiver_locations.shape[2] != 2:
            raise RuntimeError("Expected receiver_locations.shape[2] == 2")
        if receiver_locations.device != model.device:
            raise RuntimeError("Expected receiver_locations and the model "
                               "to be on the same device")
        if receiver_locations.numel() == 0:
            raise RuntimeError("receiver_locations has zero elements. Please "
                               "instead set it to None if you do not wish to "
                               "have receivers.")
    if source_locations is not None and receiver_locations is not None:
        if source_locations.shape[0] != receiver_locations.shape[0]:
            raise RuntimeError("Expected source_locations.shape[0] == "
                               "receiver_locations.shape[0]")
    wavefield_shape = [0, 0, 0]
    for wavefield in wavefields:
        if wavefield is not None:
            if wavefield.ndim != 3:
                raise RuntimeError("All wavefields must have three "
                                   "dimensions")
            if wavefield_shape == [0, 0, 0]:
                wavefield_shape = [wavefield.shape[0], wavefield.shape[1],
                                   wavefield.shape[2]]
            else:
                if (wavefield_shape[0] != wavefield.shape[0] or
                        wavefield_shape[1] != wavefield.shape[1] or
                        wavefield_shape[2] != wavefield.shape[2]):
                    raise RuntimeError("All wavefields must have the same "
                                       "shape")
            if wavefield.device != model.device:
                raise RuntimeError("All wavefields must be on the same "
                                   "device as the model")
            if wavefield.dtype != model.dtype:
                raise RuntimeError("All wavefields must have the same "
                                   "dtype as the model")

    if accuracy not in [2, 4, 6, 8]:
        raise RuntimeError("Only accuracy values of 2, 4, 6, and 8 are "
                           "implemented")
    if source_amplitudes is not None and nt is not None:
        raise RuntimeError("Specify source_amplitudes or nt, not both")
    if source_amplitudes is None and nt is None:
        raise RuntimeError("Specify source_amplitudes or nt")


def check_points_per_wavelength(v: Tensor, pml_freq: Optional[float],
                                dy: float, dx: float, dt: float) -> None:
    if pml_freq is not None:
        min_vel = v.abs().min()
        min_wavelength = min_vel / pml_freq
        max_spacing = max(dy, dx)
        if min_wavelength / max_spacing < 6:
            warnings.warn("At least six grid cells per wavelength is "
                          "recommended, but at a frequency of {}, a "
                          "minimum velocity of {}, and a grid cell "
                          "spacing of {}, there are only {}."
                          .format(pml_freq, min_vel, max_spacing,
                                  min_wavelength / max_spacing))
        if 1/2/dt < pml_freq:
            warnings.warn("The provided pml_freq is greater than the "
                          "Nyquist frequency of the data.")


def get_n_batch(args: List[Optional[Tensor]]) -> int:
    "Get the size of the batch dimension."
    for arg in args:
        if arg is not None:
            return arg.shape[0]
    raise RuntimeError("One of the inputs to get_n_batch must be non-None")


def pad_model(model: Tensor, pad: List[int],
              mode: str = "replicate") -> Tensor:
    return torch.nn.functional.pad(model[None, None],
                                   (pad[2], pad[3], pad[0], pad[1]),
                                   mode=mode)[0, 0]


def pad_locations(locations: Tensor, pad: List[int]) -> Tensor:
    pad_starts = pad[::2]
    pad_starts_tensor = torch.tensor(pad_starts).to(locations[0].device)
    return locations + pad_starts_tensor


def location_to_index(locations: Tensor, nx: int) -> Tensor:
    return locations[..., 0] * nx + locations[..., 1]


def cosine_taper_end(signal: Tensor, n_taper: int) -> Tensor:
    """Tapers the end of the final dimension of a Tensor using a cosine.

    A half period, shifted and scaled to taper from 1 to 0, is used.

    Args:
        signal:
            The Tensor that will have its final dimension tapered.
        n_taper:
            The length of the cosine taper, in number of samples.

    Returns:
        The tapered signal.
    """
    taper = torch.ones(signal.shape[-1], dtype=signal.real.dtype,
                       device=signal.device)
    taper[len(taper)-n_taper:] = (
        torch.cos(torch.arange(n_taper, dtype=signal.real.dtype,
                               device=signal.device) /
                  (n_taper - 1) * torch.pi) + 1
    ) / 2
    return signal * taper


def zero_last_element_of_final_dimension(signal: Tensor) -> Tensor:
    zeroer = torch.ones(signal.shape[-1], dtype=signal.real.dtype,
                        device=signal.device)
    zeroer[-1] = 0
    return signal * zeroer


def upsample(signal: Tensor, step_ratio: int, freq_taper_frac: float = 0.0,
             time_pad_frac: float = 0.0) -> Tensor:
    """Upsamples the final dimension of a Tensor by a factor.

    Low-pass upsampling is used to produce an upsampled signal without
    introducing higher frequencies than were present in the input. The
    Nyquist frequency of the input will be zeroed.

    Args:
        signal:
            The Tensor that will have its final dimension upsampled.
        step_ratio:
            The integer factor by which the signal will be upsampled.
            The input signal is returned if this is 1 (freq_taper_frac
            and time_pad_frac will be ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered before
            upsampling. Default 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before upsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the input signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the signal will be used. Default 0.0.

    Returns:
        The signal after upsampling.
    """
    if step_ratio == 1:
        return signal
    if time_pad_frac > 0.0:
        n_time_pad = int(time_pad_frac * signal.shape[-1])
        signal = torch.nn.functional.pad(signal, (0, n_time_pad))
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    signal_f = torch.fft.rfft(signal)
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    else:
        # Set Nyquist frequency to zero
        signal_f = zero_last_element_of_final_dimension(signal_f)
    signal_f = torch.nn.functional.pad(signal_f,
                                       (0, up_nt//2+1 - nt//2+1))
    signal = torch.fft.irfft(signal_f, n=up_nt)
    if time_pad_frac > 0.0:
        signal = signal[..., :signal.shape[-1] - n_time_pad * step_ratio]
    return signal


def downsample(signal: Tensor, step_ratio: int, freq_taper_frac: float = 0.0,
               time_pad_frac: float = 0.0) -> Tensor:
    """Downsamples the final dimension of a Tensor by a factor.

    Frequencies higher than or equal to the Nyquist frequency of the
    downsampled signal will be zeroed before downsampling.

    Args:
        signal:
            The Tensor that will have its final dimension downsampled.
        step_ratio:
            The integer factor by which the signal will be downsampled.
            The input signal is returned if this is 1 (freq_taper_frac
            and time_pad_frac will be ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered after
            downsampling. Default 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before downsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the output signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the output signal will be used.
            Default 0.0.

    Returns:
        The signal after downsampling.
    """
    if step_ratio == 1:
        return signal
    if time_pad_frac > 0.0:
        n_time_pad = int(time_pad_frac * (signal.shape[-1] // step_ratio))
        signal = torch.nn.functional.pad(signal, (0, n_time_pad * step_ratio))
    nt = signal.shape[-1]
    down_nt = nt // step_ratio
    signal_f = torch.fft.rfft(signal)[..., :down_nt//2+1]
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    else:
        # Set Nyquist frequency to zero
        signal_f = zero_last_element_of_final_dimension(signal_f)
    signal = torch.fft.irfft(signal_f, n=down_nt)
    if time_pad_frac > 0.0:
        signal = signal[..., :signal.shape[-1] - n_time_pad]
    return signal


def convert_to_contiguous(tensor: Optional[Tensor]) -> Tensor:
    if tensor is None:
        return torch.empty(0)
    return tensor.contiguous()


def extract_survey(models: List[Tensor],
                   locations: List[Optional[Tensor]],
                   survey_pad: Optional[Union[int,
                                              List[Optional[int]]]],
                   wavefields: List[Optional[Tensor]],
                   origin: Optional[List[int]],
                   pad: List[int]) -> Tuple[List[Tensor],
                                            List[Optional[Tensor]]]:
    check_locations_are_within_model(models[0].shape, locations)
    if any([wavefield is not None for wavefield in wavefields]):
        survey_extents = get_survey_extents_from_wavefields(
            wavefields, origin, pad
        )
        check_extents_within_model(survey_extents, models[0].shape)
        check_locations_within_extents(survey_extents, locations)
    else:
        survey_extents = get_survey_extents_from_locations(
            models[0].shape, locations, survey_pad
        )
    return extract_models(models, survey_extents), extract_locations(
        locations, survey_extents
    )


def check_locations_are_within_model(
            model_shape: List[int],
            locations: List[Optional[Tensor]]
        ) -> None:
    for location in locations:
        if location is not None:
            if location.min() < 0:
                raise RuntimeError("Locations must be >= 0")
            for dim, model_dim_shape in enumerate(model_shape):
                if location[..., dim].max() >= model_dim_shape:
                    raise RuntimeError("Locations must be within model")


def set_survey_pad(survey_pad: Optional[Union[int, List[Optional[int]]]],
                   ndim: int) -> List[int]:
    """Check survey_pad, and convert to a list if it is a scalar."""
    # Expand to list
    survey_pad_list: List[int] = []
    if survey_pad is None:
        survey_pad_list = [-1] * 2 * ndim
    elif isinstance(survey_pad, int):
        if survey_pad < 0:
            raise RuntimeError("survey_pad must be non-negative")
        survey_pad_list = [survey_pad] * 2 * ndim
    elif isinstance(survey_pad, list):
        for pad in survey_pad:
            if pad is None:
                survey_pad_list.append(-1)
            elif pad >= 0:
                survey_pad_list.append(pad)
            else:
                raise RuntimeError("survey_pad must be non-negative")
    else:
        raise RuntimeError("survey_pad must be None, an int, or a list")

    # Check has correct size
    if len(survey_pad_list) != 2 * ndim:
        raise RuntimeError(
            "survey_pad must have length 2 * dims in model, "
            "but got {}".format(len(survey_pad_list))
        )

    return survey_pad_list


def get_survey_extents_from_locations(
            model_shape: List[int],
            locations: List[Optional[Tensor]],
            survey_pad: Optional[Union[int, List[Optional[int]]]]
        ) -> List[int]:
    """Calculate the extents of the model to use for the survey.

    Args:
        model_shape:
            A list containing the shape of the full model.
        locations:
            A list of Tensors containing source/receiver locations.
        survey_pad:
            A list with two entries for each dimension, specifying the padding
            to add around the sources and receivers included in all of the
            shots being propagated. If None, the padding continues to the edge
            of the model.

    Returns:
        A list of length twice the number of dimensions in the model,
        specifying the extents of the model that will be
        used for wave propagation.
    """
    ndims = 2
    extents: List[int] = []
    survey_pad_list = set_survey_pad(survey_pad, ndims)
    for dim in range(ndims):
        left_pad = survey_pad_list[dim * 2]
        left_extent = get_survey_extents_one_side(
            left_pad, "left", dim, locations, model_shape[dim]
        )

        right_pad = survey_pad_list[dim * 2 + 1]
        right_extent = get_survey_extents_one_side(
            right_pad, "right", dim, locations, model_shape[dim]
        )

        extents.append(left_extent)
        extents.append(right_extent)
    return extents


def get_survey_extents_one_side(pad: int, side: str, dim: int,
                                locations: List[Optional[Tensor]],
                                shape: int) -> int:
    """Get the survey extent for the left or right side of one dimension.

    Args:
        pad:
            Positive int specifying padding for the side, or negative if
            the extracted model should continue to the edge of the model.
        side:
            'left' or 'right'.
        dim:
            Integer specifying the dimension index.
        locations:
            List of Tensors with coordinates for the current dimension.
        shape:
            Int specifying length of full model in current dimension.

    Returns:
        Min/max index as int.
    """
    if side == "left":
        if pad < 0 or not any([location is not None
                               for location in locations]):
            return 0
        extreme_location = shape
    else:
        if pad < 0 or not any([location is not None
                               for location in locations]):
            return shape
        extreme_location = 0

    for location in locations:
        if location is not None:
            if side == "left":
                extreme_location = \
                    min(extreme_location,
                        int(location[..., dim].min().item()) - pad)
            else:
                extreme_location = \
                    max(extreme_location,
                        int(location[..., dim].max().item()) + pad)
    if side == "right":
        extreme_location += 1
    extreme_location = max(extreme_location, 0)
    extreme_location = min(extreme_location, shape)
    return extreme_location


def get_survey_extents_from_wavefields(wavefields: List[Optional[Tensor]],
                                       origin: Optional[List[int]],
                                       pad: List[int]) -> List[int]:
    """Determine the extent of the model to extract from the wavefields.

    Args:
        wavefields:
            A list of initial wavefields (potentially None).
        origin:
            A list containing the coordinates of the origin of the wavefields.
            Optional, default [0, 0].
        pad:
            A list containing the PML width of each side.

    Returns:
        A list of length twice the number of dimensions in the model,
        specifying the extents of the model that will be
        used for wave propagation.
    """

    if origin is not None:
        if any([not isinstance(dim_origin, int) for dim_origin in origin]):
            raise RuntimeError("origin must be list of ints")
        if any([dim_origin < 0 for dim_origin in origin]):
            raise RuntimeError("origin coordinates must be non-negative")
    if not isinstance(pad, list):
        raise RuntimeError("pad must be a list of ints")
    if any([not isinstance(dim_pad, int) for dim_pad in pad]):
        raise RuntimeError("pad must be a list of ints")
    if any([dim_pad < 0 for dim_pad in pad]):
        raise RuntimeError("pad must be positive")
    ndims = len(pad) // 2
    extents: List[int] = []
    for wavefield in wavefields:
        if wavefield is not None:
            if wavefield.ndim != ndims + 1:
                raise RuntimeError("wavefields must have one more dimension "
                                   "(batch) than pad")
            for dim in range(ndims):
                if origin is not None:
                    if len(origin) != ndims:
                        raise RuntimeError("origin has incorrect number of "
                                           "dimensions")
                    extents.append(origin[dim])
                    extents.append(origin[dim]
                                   + wavefield.shape[1+dim]
                                   - (pad[dim*2] + pad[dim*2+1]))
                else:
                    extents.append(0)
                    extents.append(wavefield.shape[1+dim]
                                   - (pad[dim*2] + pad[dim*2+1]))
            return extents
    raise RuntimeError("At least one wavefield must be non-None")


def check_extents_within_model(extents: List[int],
                               model_shape: List[int]) -> None:
    for (extent, model_dim_shape) in zip(extents[::2], model_shape):
        if extent > model_dim_shape:
            raise RuntimeError("Wavefields must be within the model")


def check_locations_within_extents(
            extents: List[int],
            locations: List[Optional[Tensor]]
        ) -> None:
    for location in locations:
        if location is not None:
            for dim in range(location.shape[-1]):
                if (location is not None and
                    (location[..., dim].min() < extents[dim * 2] or
                     location[..., dim].max() >= extents[dim * 2 + 1])):
                    raise RuntimeError("Locations must be within "
                                       "wavefields")


def extract_models(models: List[Tensor],
                   extents: List[int]) -> List[Tensor]:
    """Extract the specified portion of the model.

    Args:
        models:
            A list of Tensors to extract from.
        extents:
            A list specifying the portion of the model to extract.

    Returns:
        A list containing the desired portion of the models.
    """
    return [model[extents[0]:extents[1], extents[2]:extents[3]]
            for model in models]


def extract_locations(locations: List[Optional[Tensor]],
                      extents: List[int]) -> List[Optional[Tensor]]:
    """Set locations relative to extracted model."""
    origin = extents[::2]
    extracted_locations: List[Optional[Tensor]] = []
    for location in locations:
        if location is not None:
            extracted_locations.append(
                location
                - torch.tensor(origin).to(location.device)
            )
        else:
            extracted_locations.append(location)
    return extracted_locations


def cfl_condition(dy: float, dx: float, dt: float, max_vel: float,
                  eps: float = 1e-15) -> Tuple[float, int]:
    """Calculates the time step interval to obey the CFL condition.

    The output time step will be a factor of the input time step.

    Args:
        dy:
            The grid spacing in the first dimension.
        dx:
            The grid spacing in the second dimension.
        dt:
            The time step interval.
        max_vel:
            The maximum wave speed in the model.
        eps:
            A small quantity to prevent division by zero. Default 1e-15.

    Returns:
        Tuple[float, int]:

            inner_dt:
                A time step interval that obeys the CFL condition.
            step_ratio:
                The integer dt / inner_dt.
    """
    max_dt = (0.6 / math.sqrt(1 / dy**2 + 1 / dx**2) /
              (max_vel**2 + eps)) * max_vel
    step_ratio = int(math.ceil((abs(dt) / (max_dt**2 + eps)) * max_dt))
    inner_dt = dt / step_ratio
    return inner_dt, step_ratio


def setup_pml(pml_width: List[int], fd_pad: int, dt: float, v: Tensor,
              max_vel: float, pml_freq: Optional[float],
              eps: float = 1e-9) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    R = 0.001
    if pml_freq is None:
        pml_freq = 25.0
    device = v.device
    dtype = v.dtype
    a = []
    b = []
    for width in pml_width:
        if width == 0:
            a.append(torch.zeros(0, device=device, dtype=dtype))
            b.append(torch.zeros(0, device=device, dtype=dtype))
            continue
        sigma = (
            3
            * max_vel
            * math.log((1 + R) / 2 * width)
            * (torch.arange(width, device=device, dtype=dtype)
               / width) ** 2
        )
        alpha = (
            2 * math.pi * pml_freq
            * (1 - torch.arange(width, device=device, dtype=dtype)
               / width)
        )
        sigmaalpha = sigma + alpha
        a.append(torch.exp(-sigmaalpha * abs(dt)))
        b.append(sigma / (sigmaalpha**2 + eps) * sigmaalpha * (a[-1] - 1))
    a0 = math.exp(-2 * math.pi * pml_freq * abs(dt))
    ay = a0 * torch.ones(v.shape[0], device=device, dtype=dtype)
    ax = a0 * torch.ones(v.shape[1], device=device, dtype=dtype)
    by = torch.zeros(v.shape[0], device=device, dtype=dtype)
    bx = torch.zeros(v.shape[1], device=device, dtype=dtype)
    ay[fd_pad:fd_pad + pml_width[0]] = a[0].flip(0)
    ay[:fd_pad] = ay[fd_pad]
    ay[len(ay) - pml_width[1] - fd_pad:-fd_pad] = a[1]
    ay[-fd_pad:] = ay[-fd_pad-1]
    ax[fd_pad:fd_pad + pml_width[2]] = a[2].flip(0)
    ax[:fd_pad] = ax[fd_pad]
    ax[len(ax) - pml_width[3] - fd_pad:-fd_pad] = a[3]
    ax[-fd_pad:] = ax[-fd_pad-1]
    by[fd_pad:fd_pad + pml_width[0]] = b[0].flip(0)
    by[:fd_pad] = by[fd_pad]
    by[len(by) - pml_width[1] - fd_pad:-fd_pad] = b[1]
    by[-fd_pad:] = by[-fd_pad-1]
    bx[fd_pad:fd_pad + pml_width[2]] = b[2].flip(0)
    bx[:fd_pad] = bx[fd_pad]
    bx[len(bx) - pml_width[3] - fd_pad:-fd_pad] = b[3]
    bx[-fd_pad:] = bx[-fd_pad-1]
    return ay, ax, by, bx

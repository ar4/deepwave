import math
from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor


def set_dx(dx: Union[int, float, List[Union[int, float]],
           Tensor]) -> Tuple[float, float]:
    if isinstance(dx, int):
        return float(dx), float(dx)
    if isinstance(dx, float):
        return dx, dx
    if (isinstance(dx, list) and len(dx) == 2 and
            isinstance(dx[0], (int, float)) and
            isinstance(dx[1], (int, float))):
        return float(dx[0]), float(dx[1])
    if (isinstance(dx, torch.Tensor) and dx.shape == (2,)):
        return float(dx[0].item()), float(dx[1].item())
    raise RuntimeError("Expected dx to be a real number or a list of "
                       "two real numbers")


def check_inputs(source_amplitudes: Optional[Tensor],
                 source_locations: Optional[Tensor],
                 receiver_locations: Optional[Tensor],
                 wavefields: List[Optional[Tensor]],
                 accuracy: int, nt: Optional[int], model: Tensor):
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


def get_n_batch(args: List[Optional[Tensor]]) -> int:
    "Get the size of the batch dimension."
    for arg in args:
        if arg is not None:
            return arg.shape[0]
    raise RuntimeError("One of the inputs to get_n_batch must be non-None")


def pad_model(model: Tensor, pad: List[int], mode: str = "replicate"):
    return torch.nn.functional.pad(model[None],
                                   (pad[2], pad[3], pad[0], pad[1]),
                                   mode=mode)[0]


def pad_locations(locations: Tensor, pad: List[int]):
    pad_starts = pad[::2]
    pad_starts = torch.tensor(pad_starts).to(locations[0].device)
    return locations + pad_starts


def location_to_index(locations: Tensor, ny: int):
    return locations[..., 0] * ny + locations[..., 1]


def upsample(signal: Tensor, step_ratio: int):
    if step_ratio == 1:
        return signal
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    return torch.fft.irfft(torch.nn.functional.pad(torch.fft.rfft(signal),
                                                   (0, up_nt//2+1 - nt//2+1)),
                           n=up_nt)


def downsample(signal: Tensor, step_ratio: int):
    if step_ratio == 1:
        return signal
    nt = signal.shape[-1]
    down_nt = nt // step_ratio
    return torch.fft.irfft(torch.fft.rfft(signal)[..., :down_nt//2+1],
                           n=down_nt)


def convert_to_contiguous(tensor: Optional[Tensor]):
    if tensor is None:
        return torch.empty(0)
    return tensor.contiguous()


def extract_survey(models: List[Tensor], locations: List[Optional[Tensor]],
                   survey_pad: Optional[Union[int, List[Optional[int]]]],
                   wavefields: List[Optional[Tensor]],
                   origin: Optional[List[int]],
                   pad: List[int]) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
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


def check_locations_are_within_model(model_shape: List[int],
                                     locations: List[Optional[Tensor]]):
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
    if survey_pad is None:
        survey_pad_list = [-1] * 2 * ndim
    elif isinstance(survey_pad, int):
        if survey_pad < 0:
            raise RuntimeError("survey_pad must be non-negative")
        survey_pad_list = [survey_pad] * 2 * ndim
    elif isinstance(survey_pad, list):
        survey_pad_list: List[int] = []
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
    survey_pad = set_survey_pad(survey_pad, ndims)
    for dim in range(ndims):
        left_pad = survey_pad[dim * 2]
        left_extent = get_survey_extents_one_side(
            left_pad, "left", dim, locations, model_shape[dim]
        )

        right_pad = survey_pad[dim * 2 + 1]
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


def check_extents_within_model(extents: List[int], model_shape: List[int]):
    for (extent, model_dim_shape) in zip(extents[::2], model_shape):
        if extent > model_dim_shape:
            raise RuntimeError("Wavefields must be within the model")


def check_locations_within_extents(extents: List[int],
                                   locations: List[Optional[Tensor]]):
    for location in locations:
        if location is not None:
            for dim in range(location.shape[-1]):
                if (location is not None and
                    (location[..., dim].min() < extents[dim * 2] or
                     location[..., dim].max() >= extents[dim * 2 + 1])):
                    raise RuntimeError("Locations must be within "
                                       "wavefields")


def extract_models(models: List[Tensor], extents: List[int]) -> List[Tensor]:
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


def cfl_condition(dx: float, dy: float, dt: float, max_vel: float,
                  eps: float = 1e-15) -> Tuple[float, int]:
    max_dt = (0.6 / math.sqrt(1 / dx**2 + 1 / dy**2) /
              (max_vel**2 + eps)) * max_vel
    step_ratio = int(math.ceil((abs(dt) / (max_dt**2 + eps)) * max_dt))
    inner_dt = dt / step_ratio
    return inner_dt, step_ratio


def setup_pml(pml_width: List[int], fd_pad: int, dt: float, v: Tensor,
              max_vel: float, pml_freq: Optional[float], eps: float = 1e-9):
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
    ax = a0 * torch.ones(v.shape[0], device=device, dtype=dtype)
    ay = a0 * torch.ones(v.shape[1], device=device, dtype=dtype)
    bx = torch.zeros(v.shape[0], device=device, dtype=dtype)
    by = torch.zeros(v.shape[1], device=device, dtype=dtype)
    ax[fd_pad:fd_pad + pml_width[0]] = a[0].flip(0)
    ax[len(ax) - pml_width[1] - fd_pad:-fd_pad] = a[1]
    ay[fd_pad:fd_pad + pml_width[2]] = a[2].flip(0)
    ay[len(ay) - pml_width[3] - fd_pad:-fd_pad] = a[3]
    bx[fd_pad:fd_pad + pml_width[0]] = b[0].flip(0)
    bx[len(bx) - pml_width[1] - fd_pad:-fd_pad] = b[1]
    by[fd_pad:fd_pad + pml_width[2]] = b[2].flip(0)
    by[len(by) - pml_width[3] - fd_pad:-fd_pad] = b[3]
    ax = ax[None, :, None]
    ay = ay[None, None, :]
    bx = bx[None, :, None]
    by = by[None, None, :]
    return ax, ay, bx, by

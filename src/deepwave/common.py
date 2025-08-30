"""Common helper functions for Deepwave propagators."""

import math
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

from dataclasses import dataclass
import torch
import torch.fft
from torch import Tensor


@dataclass
class ResampleConfig:
    step_ratio: int = 1
    freq_taper_frac: float = 0.0
    time_pad_frac: float = 0.0
    time_taper: bool = False


@dataclass
class PMLConfig:
    pml_width: Union[int, Sequence[int]]
    pml_freq: Optional[float]


@dataclass
class SurveyConfig:
    source_locations: Sequence[Optional[Tensor]]
    receiver_locations: Sequence[Optional[Tensor]]
    source_amplitudes: Sequence[Optional[Tensor]]
    wavefields: Sequence[Optional[Tensor]]
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]]
    origin: Optional[Sequence[int]]


IGNORE_LOCATION = -1 << 31


def _as_list(
    value: Union[int, float, torch.Tensor, Sequence[Any]], name: str, target_type: type
) -> List[Any]:
    """Convert a value to a list of a target type."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return [target_type(value.item())]
        return [target_type(v) for v in value.flatten().tolist()]
    if isinstance(value, (int, float)):
        return [target_type(value)]
    if isinstance(value, Sequence) and not isinstance(value, str):
        try:
            return [target_type(v) for v in value]
        except ValueError as e:
            raise TypeError(
                f"Elements in {name} could not be converted to {target_type.__name__}. "
                f"Original error: {e}"
            ) from e
    raise TypeError(
        f"{name} must be a float, int, torch.Tensor, or a sequence of "
        f"floats/ints, got {type(value)}."
    )


def _validate_propagator_inputs(
    models: Sequence[Tensor],
    model_pad_modes: Sequence[str],
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    survey_config: SurveyConfig,
    accuracy: int,
    fd_pad: Sequence[int],
    pml_config: PMLConfig,
    max_vel: Optional[float],
    min_nonzero_model_vel: float,
    max_model_vel: float,
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    n_dims: int,
) -> Tuple[float, float, float]:
    """Perform runtime type checking for propagator inputs."""
    if not (
        isinstance(models, Sequence) and all(isinstance(m, Tensor) for m in models)
    ):
        raise TypeError("models must be a sequence of torch.Tensor objects.")
    if not (
        isinstance(model_pad_modes, Sequence)
        and all(isinstance(m, str) for m in model_pad_modes)
    ):
        raise TypeError("model_pad_modes must be a sequence of str.")
    # grid_spacing is checked in set_grid_spacing
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a float or an int.")

    if not (
        isinstance(survey_config.source_amplitudes, Sequence)
        and all(
            (a is None or isinstance(a, Tensor))
            for a in survey_config.source_amplitudes
        )
    ):
        raise TypeError("source_amplitudes must be a sequence of torch.Tensor or None.")
    if not (
        isinstance(survey_config.source_locations, Sequence)
        and all(
            (l is None or isinstance(l, Tensor)) for l in survey_config.source_locations
        )
    ):
        raise TypeError("source_locations must be a sequence of torch.Tensor or None.")
    if not (
        isinstance(survey_config.receiver_locations, Sequence)
        and all(
            (l is None or isinstance(l, Tensor))
            for l in survey_config.receiver_locations
        )
    ):
        raise TypeError(
            "receiver_locations must be a sequence of torch.Tensor or None."
        )
    if not isinstance(accuracy, int):
        raise TypeError("accuracy must be an int.")
    if not (isinstance(fd_pad, Sequence) and all(isinstance(f, int) for f in fd_pad)):
        raise TypeError("fd_pad must be a sequence of int.")
    # pml_width is checked in set_pml_width
    if pml_config.pml_freq is not None and not isinstance(
        pml_config.pml_freq, (int, float)
    ):
        raise TypeError("pml_freq must be a float, int, or None.")
    if max_vel is not None and not isinstance(max_vel, (int, float)):
        raise TypeError("max_vel must be a float, int, or None.")
    if not isinstance(min_nonzero_model_vel, float):
        raise TypeError("min_nonzero_model_vel must be a float.")
    if not isinstance(max_model_vel, float):
        raise TypeError("max_model_vel must be a float.")
    if survey_config.survey_pad is not None and not (
        isinstance(survey_config.survey_pad, int)
        or (
            isinstance(survey_config.survey_pad, Sequence)
            and all((p is None or isinstance(p, int)) for p in survey_config.survey_pad)
        )
    ):
        raise TypeError(
            "survey_pad must be an int, a sequence of int or None, or a sequence of None and int."
        )
    if not (
        isinstance(survey_config.wavefields, Sequence)
        and all((w is None or isinstance(w, Tensor)) for w in survey_config.wavefields)
    ):
        raise TypeError("wavefields must be a sequence of torch.Tensor or None.")
    if survey_config.origin is not None and not (
        isinstance(survey_config.origin, Sequence)
        and all(isinstance(o, int) for o in survey_config.origin)
    ):
        raise TypeError("origin must be a sequence of int or None.")
    if nt is not None and not isinstance(nt, int):
        raise TypeError("nt must be an int or None.")
    if not isinstance(model_gradient_sampling_interval, int):
        raise TypeError("model_gradient_sampling_interval must be an int.")
    if not isinstance(freq_taper_frac, (int, float)):
        raise TypeError("freq_taper_frac must be a float or an int.")
    if not isinstance(time_pad_frac, (int, float)):
        raise TypeError("time_pad_frac must be a float or an int.")
    if not isinstance(time_taper, bool):
        raise TypeError("time_taper must be a bool.")
    if not isinstance(n_dims, int):
        raise TypeError("n_dims must be an int.")

    return float(dt), float(freq_taper_frac), float(time_pad_frac)


def setup_propagator(
    models: Sequence[Tensor],
    model_pad_modes: Sequence[str],
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    survey_config: SurveyConfig,
    accuracy: int,
    fd_pad: Sequence[int],
    pml_config: PMLConfig,
    max_vel: Optional[float],
    min_nonzero_model_vel: float,
    max_model_vel: float,
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    n_dims: int,
) -> Tuple[
    List[Tensor],
    List[Tensor],
    List[Tensor],
    List[Tensor],
    List[Tensor],
    List[float],
    float,
    int,
    int,
    int,
    int,
    int,
    List[int],
    float,
    float,
    ResampleConfig,
    torch.device,
    torch.dtype,
]:
    """
    Common setup for all propagators.

    Performs input validation, calculates internal time step, sets up PML, and prepares source/receiver tensors.
    """
    dt, freq_taper_frac, time_pad_frac = _validate_propagator_inputs(
        models,
        model_pad_modes,
        grid_spacing,
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
        n_dims,
    )

    n_batch = get_n_batch(survey_config.source_locations, survey_config.wavefields)
    device = models[0].device
    dtype = models[0].dtype
    grid_spacing = set_grid_spacing(grid_spacing, n_dims)
    accuracy = set_accuracy(accuracy)
    pml_width = set_pml_width(pml_config.pml_width, n_dims)
    pml_freq = set_pml_freq(pml_config.pml_freq, dt)
    check_points_per_wavelength(min_nonzero_model_vel, pml_freq, grid_spacing)
    max_vel = set_max_vel(max_vel, max_model_vel)
    dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    nt = set_nt(nt, survey_config.source_amplitudes, step_ratio)
    model_gradient_sampling_interval = set_model_gradient_sampling_interval(
        model_gradient_sampling_interval, step_ratio
    )
    freq_taper_frac = set_freq_taper_frac(freq_taper_frac)
    time_pad_frac = set_time_pad_frac(time_pad_frac)
    resample_config = ResampleConfig(
        step_ratio=step_ratio,
        freq_taper_frac=freq_taper_frac,
        time_pad_frac=time_pad_frac,
        time_taper=time_taper,
    )
    check_source_amplitudes_locations_match(
        survey_config.source_amplitudes, survey_config.source_locations
    )
    source_amplitudes_out = set_source_amplitudes(
        survey_config.source_amplitudes, n_batch, nt, resample_config, device, dtype
    )
    models_out, source_locations_out, receiver_locations_out, wavefields_out = (
        extract_survey(
            models,
            survey_config.source_locations,
            survey_config.receiver_locations,
            list(survey_config.wavefields),
            survey_config.survey_pad,
            survey_config.origin,
            fd_pad,
            pml_width,
            model_pad_modes,
            n_batch,
            n_dims,
            device,
            dtype,
        )
    )
    return (
        models_out,
        source_amplitudes_out,
        wavefields_out,
        source_locations_out,
        receiver_locations_out,
        grid_spacing,
        dt,
        nt,
        n_batch,
        step_ratio,
        model_gradient_sampling_interval,
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        resample_config,
        device,
        dtype,
    )


def get_n_batch(
    source_locations: Sequence[Optional[Tensor]], wavefields: Sequence[Optional[Tensor]]
) -> int:
    """
    Get the batch size from source_locations or wavefields.

    Args:
        source_locations: Sequence of source location tensors (or None).
        wavefields: Sequence of wavefield tensors (or None).

    Returns:
        int: Batch size (first dimension of any non-None tensor).

    Raises:
        RuntimeError: If all tensors are None.
    """
    tensors = list(source_locations) + list(wavefields)
    for tensor in tensors:
        if tensor is not None:
            return tensor.shape[0]
    raise RuntimeError(
        "At least one input source_locations or wavefield must be non-None."
    )


def downsample_and_movedim(
    receiver_amplitudes: Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> Tensor:
    """
    Downsample receiver data and move the time dimension to the last axis.

    This is a convenience function that combines the downsampling of
    receiver data (if `step_ratio` > 1) with moving the time dimension
    to be the last dimension, which is the format expected by the user.

    Args:
        receiver_amplitudes: A Tensor containing the receiver amplitudes.
        step_ratio: The integer factor by which to downsample.
        freq_taper_frac: The fraction of the end of the frequency spectrum
                         to taper.
        time_pad_frac: The fraction of the time axis to pad with zeros.
        time_taper: Whether to apply a Hann window in time.
        shift: The amount to shift the data in time before downsampling.

    Returns:
        The processed receiver amplitudes.
    """
    if receiver_amplitudes.numel() > 0:
        receiver_amplitudes = torch.movedim(receiver_amplitudes, 0, -1)
        receiver_amplitudes = downsample(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            shift=shift,
        )
    return receiver_amplitudes


def set_grid_spacing(
    grid_spacing: Union[float, Sequence[float]], n_dims: int
) -> List[float]:
    """
    Ensure grid_spacing is a sequence of length n_dims.
    """
    processed_grid_spacing = _as_list(grid_spacing, "grid_spacing", float)

    for g in processed_grid_spacing:
        if g <= 0:
            raise ValueError("grid_spacing elements must be positive.")

    if len(processed_grid_spacing) == 1:
        return processed_grid_spacing * n_dims
    if len(processed_grid_spacing) != n_dims:
        raise RuntimeError(
            f"grid_spacing must have 1 or {n_dims} elements, got {len(processed_grid_spacing)}."
        )
    return processed_grid_spacing


def set_accuracy(accuracy: int) -> int:
    """
    Validate finite difference accuracy.
    """
    if not isinstance(accuracy, int):
        raise TypeError("accuracy must be an int.")
    if accuracy not in (2, 4, 6, 8):
        raise ValueError(f"accuracy must be 2, 4, 6, or 8, got {accuracy}")
    return accuracy


def set_pml_width(pml_width: Union[int, Sequence[int]], n_dims: int) -> List[int]:
    """
    Ensure pml_width is a sequence of length 2 * n_dims.
    """
    processed_pml_width = _as_list(pml_width, "pml_width", int)

    for w in processed_pml_width:
        if w < 0:
            raise ValueError("pml_width must be non-negative.")

    if len(processed_pml_width) == 1:
        return processed_pml_width * (2 * n_dims)
    if len(processed_pml_width) != 2 * n_dims:
        raise RuntimeError(
            f"Expected pml_width to be of length 1 or {2 * n_dims}, got {len(processed_pml_width)}."
        )
    return processed_pml_width


def set_pml_freq(pml_freq: Optional[float], dt: float) -> float:
    """
    Set or validate PML frequency. Defaults to 25.0 Hz if not set.
    Warns if out of range.
    """
    if pml_freq is not None and not isinstance(pml_freq, (int, float)):
        raise TypeError("pml_freq must be a float, int, or None.")
    if dt <= 0:
        raise ValueError("dt must be greater than zero to calculate Nyquist frequency.")
    nyquist = 0.5 / abs(dt)
    if pml_freq is None:
        pml_freq = 25.0
        warnings.warn(f"pml_freq was not set, so defaulting to {pml_freq}.")
    if pml_freq < 0:
        raise ValueError("pml_freq must be non-negative.")
    if pml_freq > nyquist:
        warnings.warn(
            f"pml_freq {pml_freq} is greater than the Nyquist frequency {nyquist}."
        )
    return float(pml_freq)


def set_max_vel(max_vel: Optional[float], max_model_vel: float) -> float:
    """
    Set or validate maximum velocity for CFL condition.
    """
    if max_vel is not None and not isinstance(max_vel, (int, float)):
        raise TypeError("max_vel must be a float, int, or None.")
    if max_model_vel <= 0:
        raise ValueError("max_model_vel must be greater than zero.")
    if max_vel is None:
        return max_model_vel
    max_vel = abs(max_vel)
    if max_vel < max_model_vel:
        warnings.warn("max_vel is less than the actual maximum velocity.")
    return max_vel


def set_nt(
    nt: Optional[int], source_amplitudes: Sequence[Optional[Tensor]], step_ratio: int
) -> int:
    """
    Set or validate number of time steps.
    """
    if nt is not None and not isinstance(nt, int):
        raise TypeError("nt must be an int or None.")
    if step_ratio < 1:
        raise ValueError("step_ratio must be >= 1")
    source_amplitudes_nt = next(
        (a.shape[-1] for a in source_amplitudes if a is not None), None
    )
    if nt is None:
        if source_amplitudes_nt is None:
            raise RuntimeError("nt or source amplitudes must be specified")
        nt = source_amplitudes_nt
    elif source_amplitudes_nt is not None and nt != source_amplitudes_nt:
        raise RuntimeError("Only one of nt or source amplitudes should be specified")
    if nt < 0:
        raise RuntimeError("nt must be >= 0")
    return nt * step_ratio


def set_model_gradient_sampling_interval(
    model_gradient_sampling_interval: int, step_ratio: int
) -> int:
    """
    Validate model gradient sampling interval.
    """
    if not isinstance(model_gradient_sampling_interval, int):
        raise TypeError("model_gradient_sampling_interval must be an int.")
    if model_gradient_sampling_interval < 0:
        raise ValueError("model_gradient_sampling_interval must be >= 0")
    return model_gradient_sampling_interval


def set_freq_taper_frac(freq_taper_frac: float) -> float:
    """
    Validate frequency taper fraction.
    """
    if not isinstance(freq_taper_frac, (int, float)):
        raise TypeError("freq_taper_frac must be a float or an int.")
    if not (0.0 <= freq_taper_frac <= 1.0):
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}.")
    return freq_taper_frac


def set_time_pad_frac(time_pad_frac: float) -> float:
    """
    Validate time padding fraction.
    """
    if not isinstance(time_pad_frac, (int, float)):
        raise TypeError("time_pad_frac must be a float or an int.")
    if not (0.0 <= time_pad_frac <= 1.0):
        raise ValueError(f"time_pad_frac must be in [0, 1], got {time_pad_frac}.")
    return time_pad_frac


def check_source_amplitudes_locations_match(
    source_amplitudes: Sequence[Optional[Tensor]],
    source_locations: Sequence[Optional[Tensor]],
) -> None:
    """
    Ensure source_amplitudes and source_locations match in length and structure.
    """
    if len(source_amplitudes) != len(source_locations):
        raise RuntimeError(
            f"The same number of source_amplitudes ({len(source_amplitudes)}) and source_locations ({len(source_locations)}) must be provided."
        )
    for amplitudes, locations in zip(source_amplitudes, source_locations):
        if (amplitudes is None) != (locations is None):
            raise RuntimeError(
                "Each pair of source locations and amplitudes must both be None or both be non-None."
            )
        if amplitudes is not None and locations is not None:
            if amplitudes.shape[1] != locations.shape[1]:
                raise RuntimeError(
                    f"Expected source amplitudes and locations to be the same size in the n_sources_per_shot dimension, got {amplitudes.shape[1]} and {locations.shape[1]}."
                )


def set_source_amplitudes(
    source_amplitudes: Sequence[Optional[Tensor]],
    n_batch: int,
    nt: int,
    resample_config: ResampleConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    result: List[Tensor] = []
    for amplitudes in source_amplitudes:
        if amplitudes is None:
            result.append(torch.empty(nt, n_batch, 0, device=device, dtype=dtype))
            continue
        if amplitudes.ndim != 3:
            raise RuntimeError(
                f"source amplitudes Tensors should have 3 dimensions, but found one with {amplitudes.ndim}."
            )
        if amplitudes.device != device:
            raise RuntimeError(
                f"Inconsistent device: Expected all Tensors be on device {device}, but found a source amplitudes Tensor on device {amplitudes.device}."
            )
        if amplitudes.dtype != dtype:
            raise RuntimeError(
                f"Inconsistent dtype: Expected source amplitudes to have datatype {dtype}, but found one with dtype {amplitudes.dtype}."
            )
        if amplitudes.shape[0] != n_batch:
            raise RuntimeError(
                f"Expected source amplitudes to have size {n_batch} in the batch dimension, but found one with size {amplitudes.shape[0]}."
            )
        if amplitudes.shape[2] * resample_config.step_ratio != nt:
            raise RuntimeError(
                f"Inconsistent number of time samples: Expected source amplitudes to have {nt // resample_config.step_ratio} time samples, but found one with {amplitudes.shape[2]}."
            )
        result.append(
            torch.movedim(
                upsample(
                    amplitudes,
                    step_ratio=resample_config.step_ratio,
                    freq_taper_frac=resample_config.freq_taper_frac,
                    time_pad_frac=resample_config.time_pad_frac,
                    time_taper=resample_config.time_taper,
                ),
                -1,
                0,
            ).contiguous()
        )
    return result


def check_points_per_wavelength(
    min_nonzero_vel: float, pml_freq: float, grid_spacing: Sequence[float]
) -> None:
    if min_nonzero_vel < 0:
        raise ValueError("min_nonzero_vel must be non-negative.")
    if pml_freq < 0:
        raise ValueError("pml_freq must be non-negative.")
    for g in grid_spacing:
        if g <= 0:
            raise ValueError("grid_spacing elements must be positive.")

    if pml_freq == 0:
        min_wavelength = float("inf")
    else:
        min_wavelength = abs(min_nonzero_vel / pml_freq)
    max_spacing = max(abs(dim_spacing) for dim_spacing in grid_spacing)
    cells_per_wavelength = min_wavelength / max_spacing
    if cells_per_wavelength < 6:
        warnings.warn(
            f"At least six grid cells per wavelength is recommended, but at a frequency of {pml_freq}, a minimum non-zero velocity of {min_nonzero_vel}, and a grid cell spacing of {max_spacing}, there are only {cells_per_wavelength:.2f}."
        )


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
    if n_taper < 0:
        raise ValueError("n_taper must be non-negative.")
    taper = torch.ones(signal.shape[-1], dtype=signal.dtype, device=signal.device)
    n_taper = min(n_taper, signal.shape[-1])
    if n_taper == 0:
        return signal * taper
    taper[len(taper) - n_taper :] = (
        torch.cos(
            torch.arange(1, n_taper + 1, device=signal.device) / n_taper * math.pi
        )
        + 1
    ).to(signal.dtype) / 2
    return signal * taper


def zero_last_element_of_final_dimension(signal: Tensor) -> Tensor:
    if signal.numel() == 0:
        return signal
    zeroer = torch.ones(signal.shape[-1], dtype=signal.dtype, device=signal.device)
    zeroer[-1] = 0
    return signal * zeroer


def upsample(
    signal: Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
) -> Tensor:
    """Upsamples the final dimension of a Tensor by a factor.

    Low-pass upsampling is used to produce an upsampled signal without
    introducing higher frequencies than were present in the input. The
    Nyquist frequency of the input will be zeroed.

    Args:
        signal:
            The Tensor that will have its final dimension upsampled.
        step_ratio:
            The integer factor by which the signal will be upsampled.
            The input signal is returned if this is 1 (freq_taper_frac,
            time_pad_frac, and time_taper will be ignored).
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
        time_taper:
            A bool specifying whether to apply a Hann window in time.
            This is useful during correctness tests of the propagators
            as it ensures that signals taper to zero at their edges in
            time, avoiding the possibility of high frequencies being
            introduced.

    Returns:
        The signal after upsampling.
    """
    if not isinstance(signal, Tensor):
        raise TypeError("signal must be a torch.Tensor.")
    if signal.numel() == 0:
        return signal

    if not isinstance(step_ratio, int):
        raise TypeError("step_ratio must be an int.")
    if step_ratio <= 0:
        raise ValueError("step_ratio must be positive.")
    if not isinstance(freq_taper_frac, (int, float)):
        raise TypeError("freq_taper_frac must be a float or an int.")
    if not (0.0 <= freq_taper_frac <= 1.0):
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}.")
    if not isinstance(time_pad_frac, (int, float)):
        raise TypeError("time_pad_frac must be a float or an int.")
    if not (0.0 <= time_pad_frac <= 1.0):
        raise ValueError(f"time_pad_frac must be in [0, 1], got {time_pad_frac}.")
    if not isinstance(time_taper, bool):
        raise TypeError("time_taper must be a bool.")

    if step_ratio == 1:
        return signal
    n_time_pad = int(time_pad_frac * signal.shape[-1]) if time_pad_frac > 0.0 else 0
    if n_time_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_time_pad))
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    signal_f = torch.fft.rfft(signal, norm="ortho") * math.sqrt(step_ratio)  # pylint: disable=not-callable
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)
    pad_len = up_nt // 2 + 1 - signal_f.shape[-1]
    if pad_len > 0:
        signal_f = torch.nn.functional.pad(signal_f, (0, pad_len))
    signal = torch.fft.irfft(signal_f, n=up_nt, norm="ortho")  # pylint: disable=not-callable
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad * step_ratio]
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1], periodic=False, device=signal.device
        )
    return signal


def downsample(
    signal: Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> Tensor:
    """Downsamples the final dimension of a Tensor by a factor.

    Frequencies higher than or equal to the Nyquist frequency of the
    downsampled signal will be zeroed before downsampling.

    Args:
        signal:
            The Tensor that will have its final dimension downsampled.
        step_ratio:
            The integer factor by which the signal will be downsampled.
            The input signal is returned if this is 1 and shift is 0
            (freq_taper_frac, time_pad_frac, and time_taper will be
            ignored).
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
        time_taper:
            A bool specifying whether to apply a Hann window in time.
            This is useful during correctness tests of the propagators
            as it ensures that signals taper to zero at their edges in
            time, avoiding the possibility of high frequencies being
            introduced.
        shift:
            Amount (in units of time samples) to shift the data in time
            before downsampling.
            Default 0.0.

    Returns:
        The signal after downsampling.
    """
    if not isinstance(signal, Tensor):
        raise TypeError("signal must be a torch.Tensor.")
    if signal.numel() == 0:
        return signal

    if not isinstance(step_ratio, int):
        raise TypeError("step_ratio must be an int.")
    if step_ratio <= 0:
        raise ValueError("step_ratio must be positive.")
    if not isinstance(freq_taper_frac, (int, float)):
        raise TypeError("freq_taper_frac must be a float or an int.")
    if not (0.0 <= freq_taper_frac <= 1.0):
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}.")
    if not isinstance(time_pad_frac, (int, float)):
        raise TypeError("time_pad_frac must be a float or an int.")
    if not (0.0 <= time_pad_frac <= 1.0):
        raise ValueError(f"time_pad_frac must be in [0, 1], got {time_pad_frac}.")
    if not isinstance(time_taper, bool):
        raise TypeError("time_taper must be a bool.")
    if not isinstance(shift, (int, float)):
        raise TypeError("shift must be a float or an int.")

    if step_ratio == 1 and shift == 0.0:
        return signal
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1], periodic=False, device=signal.device
        )
    n_time_pad = (
        int(time_pad_frac * (signal.shape[-1] // step_ratio))
        if time_pad_frac > 0.0
        else 0
    )
    if n_time_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_time_pad * step_ratio))
    nt = signal.shape[-1]
    down_nt = nt // step_ratio
    signal_f = torch.fft.rfft(signal, norm="ortho")[..., : down_nt // 2 + 1]  # pylint: disable=not-callable
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)
    if shift != 0.0:
        freqs = torch.fft.rfftfreq(signal.shape[-1], device=signal.device)[  # pylint: disable=not-callable
            : down_nt // 2 + 1
        ]
        signal_f *= torch.exp(-1j * 2 * math.pi * freqs * shift)
    signal = torch.fft.irfft(signal_f, n=down_nt, norm="ortho") / math.sqrt(step_ratio)  # pylint: disable=not-callable
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad]
    return signal


def extract_survey(
    models: Sequence[Tensor],
    source_locations: Sequence[Optional[Tensor]],
    receiver_locations: Sequence[Optional[Tensor]],
    wavefields: List[Optional[Tensor]],  # Changed from Sequence to List
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]],
    origin: Optional[Sequence[int]],
    fd_pad: Sequence[int],
    pml_width: Sequence[int],
    model_pad_modes: Sequence[str],
    n_batch: int,
    n_dims: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Extract a region of the model to reduce simulation computations, and pad.

    The region can either be chosen based on `survey_pad` and the
    source and receiver locations, or using `origin` and initial
    wavefields.

    This function extracts the appropriate region of the model and
    adjusts source and receiver locations to be relative to the
    new origin.

    This function does any necessary processing to prepare the models and locations.
    Padding is added to the models, and the locations are adjusted appropriately.
    A batch dimension is added to the models if they don't already have one,
    the device and dtype of all models and locations are checked and they are
    made contiguous, and locations that are None are changed to be empty Tensors.
    """
    if survey_pad is not None and origin is not None:
        raise RuntimeError("survey_pad and origin cannot both be specified.")
    locations: List[Optional[Tensor]] = list(source_locations) + list(
        receiver_locations
    )  # Explicitly type as List
    model_spatial_shape = list(models[0].shape[-n_dims:])
    check_locations_are_within_model(model_spatial_shape, locations)
    pad = [fd + pml for fd, pml in zip(fd_pad, pml_width)]
    if survey_pad is None and any(w is not None for w in wavefields):
        survey_extents = get_survey_extents_from_wavefields(
            wavefields, origin, pml_width
        )
        check_extents_within_model(survey_extents, model_spatial_shape)
        check_locations_within_extents(survey_extents, locations)
    else:
        survey_extents = get_survey_extents_from_locations(
            model_spatial_shape, locations, survey_pad
        )
        check_extents_match_wavefields_shape(survey_extents, wavefields, pml_width)
    return (
        list(
            extract_models(
                models, survey_extents, pad, model_pad_modes, n_batch, device, dtype
            )
        ),
        list(
            extract_locations(
                "Source", source_locations, survey_extents, pad, n_batch, device
            )
        ),
        list(
            extract_locations(
                "Receiver", receiver_locations, survey_extents, pad, n_batch, device
            )
        ),
        list(
            prepare_wavefields(
                wavefields, survey_extents, pml_width, n_batch, device, dtype
            )
        ),
    )


def check_locations_are_within_model(
    model_shape: Sequence[int], locations: Sequence[Optional[Tensor]]
) -> None:
    if not model_shape:
        raise RuntimeError("model_shape must not be empty.")
    for dim_shape in model_shape:
        if dim_shape <= 0:
            raise RuntimeError("model_shape elements must be positive.")

    for location in locations:
        if location is not None:
            for dim, model_dim_shape in enumerate(model_shape):
                dim_location = location[..., dim]
                dim_location = dim_location[dim_location != IGNORE_LOCATION]
                if dim_location.numel() == 0:
                    continue
                if dim_location.min() < 0:
                    raise RuntimeError("Locations must be >= 0.")
                if dim_location.max() >= model_dim_shape:
                    raise RuntimeError("Locations must be within model.")


def set_survey_pad(
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]], ndim: int
) -> List[int]:
    """Check survey_pad, and convert to a list if it is a scalar."""
    if ndim <= 0:
        raise RuntimeError("ndim must be positive.")

    # Expand to list
    survey_pad_list: List[int] = []
    if survey_pad is None:
        survey_pad_list = [-1] * 2 * ndim
    elif isinstance(survey_pad, int):
        if survey_pad < 0:
            raise RuntimeError("survey_pad must be non-negative.")
        survey_pad_list = [survey_pad] * 2 * ndim
    else:
        for pad in survey_pad:
            if pad is None:
                survey_pad_list.append(-1)
            elif isinstance(pad, int):
                if pad >= 0:
                    survey_pad_list.append(pad)
                else:
                    raise RuntimeError(
                        "survey_pad entries must be None or non-negative ints."
                    )
            else:
                raise RuntimeError(
                    "survey_pad entries must be None or non-negative ints."
                )
    if len(survey_pad_list) != 2 * ndim:
        raise RuntimeError(
            f"survey_pad must have length 2 * dims in model, but got {len(survey_pad_list)}."
        )
    return survey_pad_list


def get_survey_extents_from_locations(
    model_shape: Sequence[int],
    locations: Sequence[Optional[Tensor]],
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]],
) -> List[Tuple[int, int]]:
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
        A list of length equal to the number of dimensions in the model,
        specifying the extents of the model, as a tuple [beginning, end)
        for each dimension, that will be used for wave propagation.
    """
    if not model_shape:
        raise RuntimeError("model_shape must not be empty.")
    for dim_shape in model_shape:
        if dim_shape <= 0:
            raise RuntimeError("model_shape elements must be positive.")

    ndims = len(model_shape)
    extents: List[Tuple[int, int]] = []
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
        extents.append((left_extent, right_extent))
    return extents


def get_survey_extents_one_side(
    pad: int, side: str, dim: int, locations: Sequence[Optional[Tensor]], shape: int
) -> int:  # Added return type
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
    # if pad < 0, return the edge of the model in that direction (0 if left,
    # the model shape if right). Otherwise, initialise extreme_location to
    # the other side of the model so that we can find the min (left) or max (right)
    # location below.
    if side == "left":
        if pad < 0 or not any(location is not None for location in locations):
            return 0
        extreme_location = shape
    else:
        if pad < 0 or not any(location is not None for location in locations):
            return shape
        extreme_location = 0

    # Find the min (left) or max (right) extent in order for all locations to
    # be at least `pad` distant from the edge.
    for location in locations:
        if location is not None:
            dim_location = location[..., dim]
            dim_location = dim_location[dim_location != IGNORE_LOCATION]
            if dim_location.numel() == 0:
                continue
            # Check if the location is within the model bounds. This check is duplicated from check_locations_are_within_model
            # but is necessary here to ensure that the extents are calculated
            # correctly.
            if dim_location.min() < 0:
                raise RuntimeError("Locations must be >= 0.")
            if dim_location.max() >= shape:
                raise RuntimeError(
                    f"Locations must be within model bounds (0 to {shape - 1})."
                )

            if side == "left":
                extreme_location = min(
                    extreme_location, int(dim_location.min().item()) - pad
                )
            else:
                extreme_location = max(
                    extreme_location, int(dim_location.max().item()) + pad
                )
    # The right side of the extent is not included - it is one past the end. It
    # currently holds the largest value that should be included, though, so we add
    # 1.
    if side == "right":
        extreme_location += 1
    # The extents should not be outside the model, so we clip them to the extents
    # of the model.
    extreme_location = max(extreme_location, 0)
    extreme_location = min(extreme_location, shape)
    return extreme_location


def get_survey_extents_from_wavefields(
    wavefields: Sequence[Optional[Tensor]],
    origin: Optional[Sequence[int]],
    pml_width: Sequence[int],
) -> List[Tuple[int, int]]:
    """Determine the extent of the model to extract from the wavefields.

    Args:
        wavefields:
            A list of initial wavefields (potentially None).
        origin:
            A list containing the coordinates of the origin of the wavefields.
            Optional, default all zero.
        pml_width:
            A list containing the padding to add on each side for the PML, using the specified pad mode.

    Returns:
        A list of length equal to the number of dimensions in the model,
        specifying the extents of the model, as a tuple [beginning, end)
        for each dimension, that will be used for wave propagation.
    """

    if not isinstance(pml_width, Sequence):
        raise RuntimeError("pml_width must be a Sequence")
    ndims = len(pml_width) // 2
    if origin is not None:
        if len(origin) != ndims:
            raise RuntimeError(f"origin must be a list of length {ndims}.")
        if any(not isinstance(dim_origin, int) for dim_origin in origin):
            raise RuntimeError("origin must be a list of integers.")
        if any(dim_origin < 0 for dim_origin in origin):
            raise RuntimeError("origin coordinates must be non-negative.")
    if any(dim_pml_width < 0 for dim_pml_width in pml_width):
        raise RuntimeError("pml_width must be non-negative.")
    extents: List[Tuple[int, int]] = []
    for wavefield in wavefields:
        if wavefield is not None:
            if wavefield.ndim != ndims + 1:
                raise RuntimeError(
                    f"wavefields must have {ndims + 1} dimensions (batch + spatial)."
                )
            for dim in range(ndims):
                dim_origin = 0 if origin is None else origin[dim]
                extent_size = (
                    wavefield.shape[1 + dim]
                    - pml_width[dim * 2]
                    - pml_width[dim * 2 + 1]
                )
                extents.append((dim_origin, dim_origin + extent_size))
            if origin is None:
                warnings.warn(
                    f"Survey extents were inferred from the wavefield shape to be {extents} because origin was not provided."
                )
            return extents
    raise RuntimeError("At least one wavefield must be non-None.")


def check_extents_within_model(
    extents: Sequence[Tuple[int, int]], model_shape: Sequence[int]
) -> None:
    for extent, model_dim_shape in zip(extents, model_shape):
        if extent[0] < 0 or model_dim_shape < extent[1]:
            raise RuntimeError(
                "Survey extents are larger than the model. "
                + "This probably occurred because you provided "
                + "an initial wavefield of the wrong shape."
            )


def check_locations_within_extents(
    extents: Sequence[Tuple[int, int]], locations: Sequence[Optional[Tensor]]
) -> None:
    for location in locations:
        if location is not None:
            for dim in range(location.shape[-1]):
                dim_location = location[..., dim]
                dim_location = dim_location[dim_location != IGNORE_LOCATION]
                if (
                    dim_location.min() < extents[dim][0]
                    or extents[dim][1] <= dim_location.max()
                ):
                    raise RuntimeError(
                        "Locations are not within "
                        "survey extents. This probably occurred "
                        + "because you provided an "
                        + "initial wavefield that does not cover all "
                        + "source and receiver locations, given the "
                        + "specified origin."
                    )


def check_extents_match_wavefields_shape(
    extents: Sequence[Tuple[int, int]],
    wavefields: Sequence[Optional[Tensor]],
    pad: Sequence[int],
) -> None:
    n_dims = len(extents)
    assert len(pad) == 2 * n_dims
    for wavefield in wavefields:
        if wavefield is not None:
            assert len(wavefield.shape) - 1 == n_dims
            for dim in range(n_dims):
                # The spatial dimensions should have size equal to the extent
                # plus the padding
                expected = (
                    extents[dim][1] - extents[dim][0] + pad[2 * dim + 1] + pad[2 * dim]
                )
                if wavefield.shape[1 + dim] != expected:
                    raise RuntimeError(
                        "The shape of the provided wavefield, "
                        + "in dimension "
                        + str(dim)
                        + ", "
                        + str(wavefield.shape[1 + dim])
                        + ", does not match "
                        "the extent determined by the specified "
                        "survey_pad, " + str(expected) + "."
                    )


def reverse_pad(pad: Sequence[int]) -> List[int]:
    n_dims = len(pad) // 2
    reversed_pad: List[int] = []
    for dim in range(n_dims - 1, -1, -1):
        reversed_pad.extend([pad[2 * dim], pad[2 * dim + 1]])
    return reversed_pad


def extract_models(
    models: Sequence[Tensor],
    extents: Sequence[Tuple[int, int]],
    pad: Sequence[int],
    pad_modes: Sequence[str],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    """Extract the specified portion of the models and prepares them.

    Args:
        models:
            A list of Tensors to extract from.
        extents:
            A list of tuples specifying the portion of the model to extract.
        pad:
            An list of integers specifying the padding to add on each side
        pad_modes:
            A list of strings specifying the pad mode to use for the PML region of each model.
        n_batch:
            An integer specifying the size of the batch dimension.
        device:
            The device that the models should be located on.
        dtype:
            The datatype that the models should have.

    Returns:
        A list containing the desired portion of the models, with padding applied
        and a batch dimension of size either 1 or n_batch.
    """
    n_dims = len(pad) // 2
    assert len(models) == len(pad_modes)
    assert len(extents) == n_dims

    # Check all models have correct dimensions and consistent attributes
    spatial_shape = models[0].shape[-n_dims:]
    for model in models:
        if model.ndim not in (n_dims, n_dims + 1):
            raise RuntimeError(f"Models must have {n_dims} or {n_dims + 1} dimensions.")
        if model.device != device:
            raise RuntimeError(f"All models must be on device {device}.")
        if model.dtype != dtype:
            raise RuntimeError(f"All models must have dtype {dtype}.")
        if model.shape[-n_dims:] != spatial_shape:
            raise RuntimeError("All models must have the same spatial shape.")

    models = [m.unsqueeze(0) if m.ndim == n_dims else m for m in models]
    for m in models:
        if m.shape[0] not in (1, n_batch):
            raise RuntimeError(
                f"The batch dimension of models must be of size 1 or {n_batch}."
            )

    region = (slice(None),) + tuple(slice(begin, end) for begin, end in extents)
    reversed_pad = reverse_pad(pad)

    # New logic to handle n_batch = 0
    if n_batch == 0:
        return [
            torch.empty(0, *model[region].shape[1:], device=device, dtype=dtype)
            for model in models
        ]
    else:
        return [
            torch.nn.functional.pad(
                model[region], pad=reversed_pad, mode=pad_mode
            ).contiguous()
            for model, pad_mode in zip(models, pad_modes)
        ]


def extract_locations(
    name: str,
    locations: Sequence[Optional[Tensor]],
    extents: Sequence[Tuple[int, int]],
    pad: Sequence[int],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    eps: float = 0.1,
) -> List[Tensor]:
    """Set locations relative to extracted model and prepare them.

    Locations are returned as 1D indices into each batch.
    """
    n_dims = len(extents)
    origin: List[int] = []  # origin of extracted survey, including padding
    shape: List[int] = []  # shape of extracted survey, including padding
    stride: List[int] = [
        1
    ] * n_dims  # stride of each dimension in the extracted survey, including padding
    for dim in range(n_dims):
        origin.append(extents[dim][0] - pad[2 * dim])
        shape.append(
            extents[dim][1] - extents[dim][0] + pad[2 * dim] + pad[2 * dim + 1]
        )
    for dim in range(n_dims - 2, -1, -1):
        stride[dim] = stride[dim + 1] * shape[dim + 1]

    extracted_locations: List[Tensor] = []
    for location in locations:
        if location is not None:
            # if location.device != device:
            #    raise RuntimeError("Inconsistent device: Expected all Tensors be on device " + str(device) + ", but found a locations Tensor on device " + str(location.device) + ".")

            # if location.dtype != dtype:
            #    raise RuntimeError("Locations must have dtype " + str(dtype) + ", but found one with dtype " + str(location.dtype) + ".")
            if location.numel() > 0 and eps < (location - location.long()).abs().max():
                warnings.warn(
                    "Locations should be specified as integer numbers "
                    "of cells. If you wish to have a source or receiver "
                    "that is not centred on a cell, please consider "
                    "using the Hick's method, which is implemented "
                    "in deepwave.location_interpolation."
                )

            if location.ndim != 3:
                raise RuntimeError(
                    name + " location Tensors must have three dimensions"
                )

            if location.shape[0] != n_batch:
                raise RuntimeError(
                    "Inconsistent batch size: Expected all Tensors to have a batch size of "
                    + str(n_batch)
                    + ", but found a "
                    + name.lower()
                    + " locations Tensor with a batch size of "
                    + str(location.shape[0])
                    + "."
                )

            if location.shape[-1] != n_dims:
                raise RuntimeError(
                    name
                    + " locations must have "
                    + str(n_dims)
                    + " dimensional coordinates, but found one with "
                    + str(location.shape[-1])
                    + "."
                )

            # Shift locations to be relative to new origin in extracted (and padded)
            # model
            shifted_location = location.clone().long().to(device)
            for dim in range(n_dims):
                shifted_location[..., dim] = torch.where(
                    shifted_location[..., dim] != IGNORE_LOCATION,
                    shifted_location[..., dim] - origin[dim],
                    IGNORE_LOCATION,
                )

            # Convert locations to 1d coordinate
            location_1d = torch.where(
                shifted_location[..., 0] != IGNORE_LOCATION, 0, IGNORE_LOCATION
            )
            for dim in range(n_dims):
                location_1d += torch.where(
                    shifted_location[..., dim] != IGNORE_LOCATION,
                    shifted_location[..., dim] * stride[dim],
                    0,
                )

            # Check that locations are unique within each shot (as they may not
            # be added atomically)
            for batch_idx in range(n_batch):
                shot_locations = location_1d[batch_idx]
                shot_locations = shot_locations[shot_locations != IGNORE_LOCATION]
                if len(shot_locations) != len(shot_locations.unique()):  # type: ignore[no-untyped-call]
                    raise RuntimeError(
                        name
                        + " locations must be unique within each shot. You cannot have two in the same cell, but in shot "
                        + str(batch_idx)
                        + " there is/are "
                        + str(len(shot_locations))
                        + " active "
                        + name.lower()
                        + " locations while only "
                        + str(len(shot_locations.unique()))
                        + " is/are unique."
                    )  # type: ignore[no-untyped-call]

            extracted_locations.append(location_1d.contiguous())
        else:
            extracted_locations.append(
                torch.empty(n_batch, 0, device=device, dtype=dtype)
            )
    return extracted_locations


def prepare_wavefields(
    wavefields: List[Optional[Tensor]],  # Changed from Sequence to List
    extents: Sequence[Tuple[int, int]],
    pad: Sequence[int],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:  # Return type is List[Tensor] as Nones are replaced
    """Check wavefields and prepare them."""
    n_dims = len(extents)
    spatial_shape = [
        extents[dim][1] - extents[dim][0] + pad[2 * dim] + pad[2 * dim + 1]
        for dim in range(n_dims)
    ]
    for i, wavefield in enumerate(wavefields):
        if wavefield is not None:
            if wavefield.device != device:
                raise RuntimeError(
                    f"Inconsistent device: Expected all Tensors be on device {device}, but found a wavefield Tensor on device {wavefield.device}."
                )
            if wavefield.dtype != dtype:
                raise RuntimeError(
                    f"Inconsistent dtype: Expected wavefields to have datatype {dtype}, but found one with dtype {wavefield.dtype}."
                )
            if wavefield.ndim != n_dims + 1:
                raise RuntimeError(
                    f"Wavefields must have {n_dims + 1} dimensions, but found one with {wavefield.ndim}."
                )
            if wavefield.shape[0] != n_batch:
                raise RuntimeError(
                    f"Inconsistent batch size: Expected all Tensors to have a batch size of {n_batch}, but found a wavefield with a batch size of {wavefield.shape[0]}."
                )
            if list(wavefield.shape[1:]) != spatial_shape:
                raise RuntimeError(
                    f"Inconsistent spatial shape: Expected wavefield to have spatial shape {spatial_shape} but found one with spatial shape {list(wavefield.shape[1:])}."
                )
        else:
            wavefields[i] = torch.zeros(
                n_batch, *spatial_shape, device=device, dtype=dtype
            )
    return cast(List[Tensor], wavefields)


def cfl_condition_n(
    grid_spacing: Sequence[float],
    dt: float,
    max_abs_vel: float,
    eps: float = 1e-15,
    c_max: float = 0.6,
) -> Tuple[float, int]:
    """Calculates the time step interval to obey the CFL condition.

    The output time step will be a factor of the input time step.

    We use the maximum dt to calculate the integer factor
    (step_ratio) we need to divide the provided dt by to ensure it is
    less than or equal to this value.

    Args:
        grid_spacing:
            A List specifying the grid spacing in each spatial dimension.
        dt:
            The time step interval.
        max_abs_vel:
            The maximum absolute wavespeed in the model.
        eps:
            A small quantity to prevent division by zero. Default 1e-15.
        c_max:
            The maximum allowable Courant number.

    Returns:
        Tuple[float, int]:

            inner_dt:
                A time step interval that obeys the CFL condition.
            step_ratio:
                The integer dt / inner_dt.
    """
    if not isinstance(grid_spacing, Sequence) or not all(
        isinstance(g, (int, float)) for g in grid_spacing
    ):
        raise TypeError("grid_spacing must be a sequence of floats or ints.")
    if not grid_spacing:
        raise ValueError("grid_spacing must not be empty.")
    for g in grid_spacing:
        if g <= 0:
            raise ValueError("grid_spacing elements must be positive.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a float or an int.")
    if not isinstance(max_abs_vel, (int, float)):
        raise TypeError("max_abs_vel must be a float or an int.")
    if max_abs_vel <= 0:
        raise RuntimeError("max_abs_vel must be greater than zero.")

    max_dt = (
        c_max
        / math.sqrt(sum(1 / dim_spacing**2 for dim_spacing in grid_spacing))
        / (max_abs_vel**2 + eps)
    ) * max_abs_vel
    step_ratio = int(math.ceil(abs(dt) / max_dt))
    inner_dt = dt / step_ratio
    return inner_dt, step_ratio


def cfl_condition(
    dy: float, dx: float, *args: Any, **kwargs: Any
) -> Tuple[float, int]:  # Added type hints for *args and **kwargs
    """Calculates the time step interval to obey the CFL condition for 2D models."""
    return cfl_condition_n([dy, dx], *args, **kwargs)


def vpvsrho_to_lambmubuoyancy(
    vp: Tensor, vs: Tensor, rho: Tensor, eps: float = 1e-15
) -> Tuple[Tensor, Tensor, Tensor]:
    """Converts vp, vs, rho to lambda, mu, buoyancy.

    All input Tensors must have the same shape.

    Args:
        vp:
            A Tensor containing the p wavespeed.
        vs:
            A Tensor containing the s wavespeed.
        rho:
            A Tensor containing the density.
        eps:
            An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:

            lambda:
                A Tensor containing the first Lamé parameter.
            mu:
                A Tensor containing the second Lamé parameter.
            buoyancy:
                A Tensor containing the reciprocal of density.
    """

    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = 1 / (rho**2 + eps) * rho
    return lamb, mu, buoyancy


def lambmubuoyancy_to_vpvsrho(
    lamb: Tensor, mu: Tensor, buoyancy: Tensor, eps: float = 1e-15
) -> Tuple[Tensor, Tensor, Tensor]:
    """Converts lambda, mu, buoyancy to vp, vs, rho.

    All input Tensors must have the same shape.

    Args:
        lamb:
            A Tensor containing the first Lamé parameter.
        mu:
            A Tensor containing the second Lamé parameter.
        buoyancy:
            A Tensor containing the reciprocal of density.
        eps:
            An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:

            vp:
                A Tensor containing the p wavespeed.
            vs:
                A Tensor containing the s wavespeed.
            rho:
                A Tensor containing the density.
    """
    vs = (mu * buoyancy).sqrt()
    vp = (lamb * buoyancy + 2 * vs**2).sqrt()
    rho = 1 / (buoyancy**2 + eps) * buoyancy
    return vp, vs, rho


def setup_pml(
    pml_width: Sequence[int],
    pml_start: Sequence[float],
    max_pml: float,
    dt: float,
    dx: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float = 0.0,
    eps: float = 1e-9,
) -> Tuple[Tensor, Tensor]:
    """Creates a and b profiles for C-PML

    Args:
        pml_width: List of two integers specifying the width of the PML
                   region.
        pml_start: List of two floats specifying the coordinates (in grid cells) of the
                   start of the PML regions.
        max_pml: Float specifying the length (in distance units) of the longest
                 PML over all sides and dimensions.
        dt: Time step interval
        dx: Grid spacing
        n: Integer specifying desired profile length, including fd_pad and
           pml_width
        max_vel: Maximum wave speed
        dtype: PyTorch datatype to use
        device: PyTorch device to use
        pml_freq: The frequency value to use for the profile, usually the
                  dominant frequency in the wavefield.
        start: Float specifying the coordinate (in grid cells) of the first element.
                    Optional, default 0.
        eps: A small number to prevent division by zero. Optional, default 1e-9.

    Only the first fd_pad[0]+pml_width[0] and last fd_pad[1]+pml_width[1]
    elements of the profiles will be non-zero.

    """
    r_val = 0.001
    n_power = 2
    alpha0 = math.pi * pml_freq
    if max_pml == 0:
        a = torch.zeros(n, device=device, dtype=dtype)
        b = torch.zeros(n, device=device, dtype=dtype)
        return a, b
    sigma0 = -(1 + n_power) * max_vel * math.log(r_val) / (2 * max_pml)
    x = torch.arange(start, start + n, device=device, dtype=dtype)
    if pml_width[0] == 0:
        pml_frac0 = torch.zeros_like(x)
    else:
        pml_frac0 = (pml_start[0] - x) / pml_width[0]
    if pml_width[1] == 0:
        pml_frac1 = torch.zeros_like(x)
    else:
        pml_frac1 = (x - pml_start[1]) / pml_width[1]
    pml_frac = torch.clamp(torch.maximum(pml_frac0, pml_frac1), min=0, max=1)
    sigma = sigma0 * pml_frac**n_power
    alpha = alpha0 * (1 - pml_frac)
    sigmaalpha = sigma + alpha
    a = torch.exp(-sigmaalpha * abs(dt))
    b = sigma / sigmaalpha * (a - 1)
    a[pml_frac == 0] = 0
    return a, b


def create_or_pad(
    tensor: Tensor,
    fd_pad: Union[int, Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
    size: Sequence[int],
) -> Tensor:
    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * len(size) * 2
    if tensor.numel() == 0:
        return torch.zeros(size[0], size[1], size[2], device=device, dtype=dtype)
    if max(fd_pad) == 0:
        return tensor.clone()
    return (
        torch.nn.functional.pad(
            tensor, (fd_pad[2], fd_pad[3], fd_pad[0], fd_pad[1])
        )
    ).requires_grad_(tensor.requires_grad)


def zero_interior(
    tensor: Tensor, fd_pad: Union[int, Sequence[int]], pml_width: Sequence[int], y: bool
) -> Tensor:
    ny = tensor.shape[1]
    nx = tensor.shape[2]
    tensor = tensor.clone()
    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * 4
    if y:
        tensor[:, fd_pad[0] + pml_width[0] : ny - pml_width[1] - fd_pad[1]].fill_(0)
    else:
        tensor[:, :, fd_pad[2] + pml_width[2] : nx - pml_width[3] - fd_pad[3]].fill_(0)
    return tensor


def diff(a: Tensor, accuracy: int, grid_spacing: float) -> Tensor:
    if accuracy == 2:
        a = torch.nn.functional.pad(a, (1, 1))
        return (1 / 2 * (a[2:] - a[:-2])) / grid_spacing
    if accuracy == 4:
        a = torch.nn.functional.pad(a, (2, 2))
        return (
            8 / 12 * (a[3:-1] - a[1:-3]) + -1 / 12 * (a[4:] - a[:-4])
        ) / grid_spacing
    if accuracy == 6:
        a = torch.nn.functional.pad(a, (3, 3))
        return (
            3 / 4 * (a[4:-2] - a[2:-4])
            + -3 / 20 * (a[5:-1] - a[1:-5])
            + 1 / 60 * (a[6:] - a[:-6])
        ) / grid_spacing
    a = torch.nn.functional.pad(a, (4, 4))
    return (
        4 / 5 * (a[5:-3] - a[3:-5])
        + -1 / 5 * (a[6:-2] - a[2:-6])
        + 4 / 105 * (a[7:-1] - a[1:-7])
        + -1 / 280 * (a[8:] - a[:-8])
    ) / grid_spacing

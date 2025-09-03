"""Common helper functions for Deepwave propagators.

This module provides a collection of utility functions used across various
Deepwave propagators. These functions handle tasks such as input validation,
PML setup, and data preparation for wave propagation simulations.
"""

import math
import warnings
from collections import abc
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    SupportsInt,
    Tuple,
    Union,
    cast,
)

import torch
import torch.fft

IGNORE_LOCATION = -1 << 31


def setup_propagator(
    models: Sequence[torch.Tensor],
    model_pad_modes: Sequence[str],
    grid_spacing: Union[float, Iterable[float]],
    dt: float,
    source_amplitudes: Sequence[Optional[torch.Tensor]],
    source_locations: Sequence[Optional[torch.Tensor]],
    receiver_locations: Sequence[Optional[torch.Tensor]],
    accuracy: int,
    fd_pad: Sequence[int],
    pml_width: Union[int, Iterable[int]],
    pml_freq: Optional[float],
    max_vel: Optional[float],
    min_nonzero_model_vel: float,
    max_model_vel: float,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]],
    wavefields: Sequence[Optional[torch.Tensor]],
    origin: Optional[Sequence[int]],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    n_dims: int,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
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
    float,
    float,
    bool,
    torch.device,
    torch.dtype,
]:
    """Performs common setup for all propagators.

    This includes input validation, calculation of the internal time step,
    PML setup, and preparation of source and receiver tensors.

    Args:
        models:
            A sequence of PyTorch Tensors representing the model
            parameters (e.g., velocity, density).
        model_pad_modes:
            A sequence of strings specifying the padding mode
            for each model (e.g., 'constant', 'reflect', 'replicate', 'circular').
        grid_spacing:
            The spacing between grid points in each dimension.
            Can be a single float (for isotropic spacing) or a sequence of
            floats (for anisotropic spacing).
        dt:
            The desired time step interval for the simulation.
        source_amplitudes:
            A sequence of optional Tensors, where each torch.Tensor
            contains the amplitudes of the sources for a given shot.
        source_locations:
            A sequence of optional Tensors, where each torch.Tensor
            contains the locations of the sources for a given shot.
        receiver_locations:
            A sequence of optional Tensors, where each torch.Tensor
            contains the locations of the receivers for a given shot.
        accuracy:
            The finite-difference accuracy order (e.g., 2, 4, 6, 8).
        fd_pad:
            A sequence of integers specifying the padding for finite
            difference stencils in each dimension.
        pml_width:
            The width of the PML (Perfectly Matched Layer) in grid
            cells. Can be a single integer or a sequence of integers for
            each side of each dimension.
        pml_freq:
            The dominant frequency for PML absorption. If None, defaults
            to 25.0 Hz.
        max_vel:
            The maximum velocity in the model. Used for CFL condition
            calculation. If None, inferred from `max_model_vel`.
        min_nonzero_model_vel:
            The minimum non-zero velocity in the model.
            Used for warning about points per wavelength.
        max_model_vel:
            The maximum velocity in the model. Used for CFL
            condition calculation.
        survey_pad:
            Optional padding around the survey area. Can be an int
            or a sequence of optional ints.
        wavefields:
            A sequence of optional Tensors representing initial
            wavefields.
        origin:
            Optional sequence of integers specifying the origin of the
            survey within the full model.
        nt:
            Optional integer specifying the number of time steps. If None,
            inferred from `source_amplitudes`.
        model_gradient_sampling_interval:
            The interval at which to sample
            the wavefield for model gradient calculation.
        freq_taper_frac:
            The fraction of the frequency spectrum to taper.
        time_pad_frac:
            The fraction of the time axis to pad with zeros.
        time_taper:
            Whether to apply a Hann window in time.
        n_dims:
            The number of spatial dimensions of the model.

    Returns:
        Tuple:

            - models_out: Processed model Tensors.
            - source_amplitudes_out: Processed source amplitude Tensors.
            - wavefields_out: Processed wavefield Tensors.
            - source_locations_out: Processed source location Tensors.
            - receiver_locations_out: Processed receiver location Tensors.
            - grid_spacing: Processed grid spacing.
            - dt: Internal time step.
            - nt: Number of time steps.
            - n_batch: Batch size.
            - step_ratio: Ratio between user dt and internal dt.
            - model_gradient_sampling_interval: Model gradient sampling interval.
            - accuracy: Finite difference accuracy.
            - pml_width: PML width.
            - pml_freq: PML frequency.
            - max_vel: Maximum velocity.
            - freq_taper_frac: Frequency taper fraction.
            - time_pad_frac: Time padding fraction.
            - time_taper: Time taper flag.
            - device: PyTorch device.
            - dtype: PyTorch data type.

    Raises:
        TypeError: If any input is of an incorrect type.
        ValueError: If any input has an invalid value.
        RuntimeError: If there are inconsistencies between inputs.

    """
    if not all(isinstance(m, torch.Tensor) for m in models):
        raise TypeError("models must be torch.Tensor objects.")
    device = models[0].device
    dtype = models[0].dtype
    n_batch = get_n_batch(source_locations, wavefields)
    grid_spacing = set_grid_spacing(grid_spacing, n_dims)
    max_vel = set_max_vel(max_vel, max_model_vel)
    dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)
    accuracy = set_accuracy(accuracy)
    pml_width = set_pml_width(pml_width, n_dims)
    pml_freq = set_pml_freq(pml_freq, dt)
    check_points_per_wavelength(min_nonzero_model_vel, pml_freq, grid_spacing)
    nt = set_nt(nt, source_amplitudes, step_ratio)
    model_gradient_sampling_interval = set_model_gradient_sampling_interval(
        model_gradient_sampling_interval,
    )
    freq_taper_frac = set_freq_taper_frac(freq_taper_frac)
    time_pad_frac = set_time_pad_frac(time_pad_frac)
    check_source_amplitudes_locations_match(source_amplitudes, source_locations)
    source_amplitudes_out = set_source_amplitudes(
        source_amplitudes,
        n_batch,
        nt,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    )
    models_out, source_locations_out, receiver_locations_out, wavefields_out = (
        extract_survey(
            models,
            source_locations,
            receiver_locations,
            wavefields,
            survey_pad,
            origin,
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
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    )


def get_n_batch(
    source_locations: Sequence[Optional[torch.Tensor]],
    wavefields: Sequence[Optional[torch.Tensor]],
) -> int:
    """Get the batch size from source_locations or wavefields.

    Args:
        source_locations: Sequence of source location tensors (or None).
        wavefields: Sequence of wavefield tensors (or None).

    Returns:
        Batch size (first dimension of any non-None tensor).

    Raises:
        RuntimeError: If all tensors are None.

    """
    tensors = list(source_locations) + list(wavefields)
    for tensor in tensors:
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Expected a torch.Tensor, but got {type(tensor).__name__}",
                )
            return tensor.shape[0]
    raise RuntimeError(
        "At least one input source_locations or wavefield must be non-None.",
    )


def downsample_and_movedim(
    receiver_amplitudes: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> torch.Tensor:
    """Downsample receiver data and move the time dimension to the last axis.

    This is a convenience function that combines the downsampling of
    receiver data (if `step_ratio` > 1) with moving the time dimension
    to be the last dimension, which is the format expected by the user.

    Args:
        receiver_amplitudes: A torch.Tensor containing the receiver amplitudes.
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
    grid_spacing: Union[float, Iterable[float]], n_dims: int,
) -> List[float]:
    """Ensures grid_spacing is a sequence of length n_dims.

    Args:
        grid_spacing: The spacing between grid points in each dimension.
            Can be a single float (for isotropic spacing) or a sequence of
            floats (for anisotropic spacing).
        n_dims: The number of spatial dimensions.

    Returns:
        A list of floats representing the grid spacing for each dimension.

    Raises:
        TypeError: If `grid_spacing` is not a float or sequence of floats.
        ValueError: If any element of `grid_spacing` is not positive.
        RuntimeError: If the length of `grid_spacing` is not 1 or `n_dims`.

    """
    if (
        isinstance(grid_spacing, abc.Iterable)
        and not isinstance(grid_spacing, (str, bytes))
        and not (hasattr(grid_spacing, "ndim") and grid_spacing.ndim == 0)
    ):
        try:
            processed_grid_spacing = [float(spacing) for spacing in grid_spacing]
        except (TypeError, ValueError):
            raise TypeError("grid_spacing must be a float or sequence of floats.")
    else:
        try:
            scalar_grid_spacing = cast("SupportsFloat", grid_spacing)
            processed_grid_spacing = [float(scalar_grid_spacing)] * n_dims
        except (TypeError, ValueError):
            raise TypeError("grid_spacing must be a float or sequence of floats.")

    if any(spacing <= 0 for spacing in processed_grid_spacing):
        raise ValueError("grid_spacing elements must be positive.")

    if len(processed_grid_spacing) != n_dims:
        raise RuntimeError(
            f"grid_spacing must have 1 or {n_dims} elements, "
            f"got {len(processed_grid_spacing)}.",
        )
    return processed_grid_spacing


def set_accuracy(accuracy: int) -> int:
    """Validates finite difference accuracy.

    Args:
        accuracy: The finite-difference accuracy order (e.g., 2, 4, 6, 8).

    Returns:
        The validated accuracy.

    Raises:
        TypeError: If `accuracy` is not an integer.
        ValueError: If `accuracy` is not one of 2, 4, 6, or 8.

    """
    if not isinstance(accuracy, int):
        raise TypeError("accuracy must be an int.")
    if accuracy not in (2, 4, 6, 8):
        raise ValueError(f"accuracy must be 2, 4, 6, or 8, got {accuracy}")
    return accuracy


def set_pml_width(pml_width: Union[int, Iterable[int]], n_dims: int) -> List[int]:
    """Ensures pml_width is a sequence of length 2 * n_dims.

    Args:
        pml_width: The width of the PML (Perfectly Matched Layer) in grid
            cells. Can be a single integer or a sequence of integers for
            each side of each dimension.
        n_dims: The number of spatial dimensions.

    Returns:
        A list of integers representing the PML width for each side of each
        dimension.

    Raises:
        TypeError: If `pml_width` is not an int or sequence of ints.
        ValueError: If any element of `pml_width` is negative.
        RuntimeError: If the length of `pml_width` is not 1 or `2 * n_dims`.

    """
    if (
        isinstance(pml_width, abc.Iterable)
        and not isinstance(pml_width, (str, bytes))
        and not (hasattr(pml_width, "ndim") and pml_width.ndim == 0)
    ):
        try:
            processed_pml_width = [int(width) for width in pml_width]
        except (TypeError, ValueError):
            raise TypeError("pml_width must be an int or sequence of ints.")
    else:
        try:
            scalar_pml_width = cast("SupportsInt", pml_width)
            processed_pml_width = [int(scalar_pml_width)] * 2 * n_dims
        except (TypeError, ValueError):
            raise TypeError("pml_width must be an int or sequence of ints.")

    if any(width < 0 for width in processed_pml_width):
        raise ValueError("pml_width must be non-negative.")

    if len(processed_pml_width) != 2 * n_dims:
        raise RuntimeError(
            f"Expected pml_width to be of length 1 or {2 * n_dims}, "
            f"got {len(processed_pml_width)}.",
        )
    return processed_pml_width


def set_pml_freq(pml_freq: Optional[float], dt: float) -> float:
    """Sets or validates the PML frequency.

    Defaults to 25.0 Hz if not set. Warns if the frequency is out of range.

    Args:
        pml_freq: The dominant frequency for PML absorption. If None, defaults
            to 25.0 Hz.
        dt: The time step interval. Used to calculate the Nyquist frequency.

    Returns:
        The validated PML frequency.

    Raises:
        TypeError: If `pml_freq` is not None or convertible to a float.
        ValueError: If `dt` is not positive or `pml_freq` is negative.

    """
    if pml_freq is not None:
        try:
            pml_freq = float(pml_freq)
        except (TypeError, ValueError):
            raise TypeError("pml_freq must be None or convertible to a float.")
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
            f"pml_freq {pml_freq} is greater than the Nyquist frequency {nyquist}.",
        )
    return pml_freq


def set_max_vel(max_vel: Optional[float], max_model_vel: float) -> float:
    """Sets or validates the maximum velocity for the CFL condition.

    Args:
        max_vel: The maximum velocity in the model. If None, `max_model_vel`
            is used.
        max_model_vel: The maximum velocity present in the model.

    Returns:
        The validated maximum velocity.

    Raises:
        TypeError: If `max_vel` or `max_model_vel` are not convertible to a float.
        ValueError: If `max_model_vel` is not positive.

    """
    if max_vel is not None:
        try:
            max_vel = float(max_vel)
        except (TypeError, ValueError):
            raise TypeError("max_vel must be None or convertible to a float.")
    try:
        max_model_vel = float(max_model_vel)
    except (TypeError, ValueError):
        raise TypeError("max_model_vel must be convertible to a float.")
    if max_model_vel <= 0:
        raise ValueError("max_model_vel must be greater than zero.")
    if max_vel is None:
        return max_model_vel
    max_vel = abs(max_vel)
    if max_vel < max_model_vel:
        warnings.warn("max_vel is less than the actual maximum velocity.")
    return max_vel


def set_nt(
    nt: Optional[int],
    source_amplitudes: Sequence[Optional[torch.Tensor]],
    step_ratio: int,
) -> int:
    """Sets or validates the number of time steps.

    Args:
        nt: The desired number of time steps. If None, it is inferred from
            `source_amplitudes`.
        source_amplitudes: A sequence of optional Tensors, where each torch.Tensor
            contains the amplitudes of the sources for a given shot.
        step_ratio: The ratio between the user-specified time step and the
            internal time step.

    Returns:
        The total number of time steps, adjusted by `step_ratio`.

    Raises:
        TypeError: If `nt` is not an int or None, or if `source_amplitudes`
            contains non-torch.Tensor elements.
        ValueError: If `step_ratio` is less than 1.
        RuntimeError: If `nt` and `source_amplitudes` are both None, or if
            they are inconsistent, or if `nt` is negative.

    """
    if nt is not None and not isinstance(nt, int):
        raise TypeError("nt must be an int or None.")
    if step_ratio < 1:
        raise ValueError("step_ratio must be >= 1")
    source_amplitudes_not_none = next(
        (a for a in source_amplitudes if a is not None), None,
    )
    source_amplitudes_nt = None
    if source_amplitudes_not_none is not None:
        if not isinstance(source_amplitudes_not_none, torch.Tensor):
            raise TypeError("source_amplitudes must be a torch.Tensor or None.")
        source_amplitudes_nt = source_amplitudes_not_none.shape[-1]
    if nt is None:
        if source_amplitudes_nt is None:
            raise RuntimeError("nt or source amplitudes must be specified")
        nt = source_amplitudes_nt
    elif source_amplitudes_nt is not None and nt != source_amplitudes_nt:
        raise RuntimeError("Only one of nt or source amplitudes should be specified")
    if nt < 0:
        raise RuntimeError("nt must be >= 0")
    return nt * step_ratio


def set_model_gradient_sampling_interval(model_gradient_sampling_interval: int) -> int:
    """Validates the model gradient sampling interval.

    Args:
        model_gradient_sampling_interval: The interval at which to sample
            the wavefield for model gradient calculation.

    Returns:
        The validated sampling interval.

    Raises:
        TypeError: If `model_gradient_sampling_interval` is not an integer.
        ValueError: If `model_gradient_sampling_interval` is negative.

    """
    if not isinstance(model_gradient_sampling_interval, int):
        raise TypeError("model_gradient_sampling_interval must be an int.")
    if model_gradient_sampling_interval < 0:
        raise ValueError("model_gradient_sampling_interval must be >= 0")
    return model_gradient_sampling_interval


def set_freq_taper_frac(freq_taper_frac: float) -> float:
    """Validates the frequency taper fraction.

    Args:
        freq_taper_frac: The fraction of the frequency spectrum to taper.

    Returns:
        The validated frequency taper fraction.

    Raises:
        TypeError: If `freq_taper_frac` is not convertible to a float.
        ValueError: If `freq_taper_frac` is not within the range [0, 1].

    """
    try:
        freq_taper_frac = float(freq_taper_frac)
    except ValueError:
        raise TypeError("freq_taper_frac must be convertible to a float.")
    if not 0.0 <= freq_taper_frac <= 1.0:
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}.")
    return freq_taper_frac


def set_time_pad_frac(time_pad_frac: float) -> float:
    """Validates the time padding fraction.

    Args:
        time_pad_frac: The fraction of the time axis to pad with zeros.

    Returns:
        The validated time padding fraction.

    Raises:
        TypeError: If `time_pad_frac` is not convertible to a float.
        ValueError: If `time_pad_frac` is not within the range [0, 1].

    """
    try:
        time_pad_frac = float(time_pad_frac)
    except ValueError:
        raise TypeError("time_pad_frac must be convertible to a float.")
    if not 0.0 <= time_pad_frac <= 1.0:
        raise ValueError(f"time_pad_frac must be in [0, 1], got {time_pad_frac}.")
    return time_pad_frac


def check_source_amplitudes_locations_match(
    source_amplitudes: Sequence[Optional[torch.Tensor]],
    source_locations: Sequence[Optional[torch.Tensor]],
) -> None:
    """Ensures source_amplitudes and source_locations match.

    This function verifies that the number of source amplitude tensors matches
    the number of source location tensors, and that for each pair, they are
    either both None or both non-None. If non-None, it checks that they are
    `torch.Tensor` objects and have matching sizes in their
    `n_sources_per_shot` dimension.

    Args:
        source_amplitudes: A sequence of optional Tensors, where each
            torch.Tensor contains the amplitudes of the sources for a given shot.
        source_locations: A sequence of optional Tensors, where each
            torch.Tensor contains the locations of the sources for a given shot.

    Raises:
        RuntimeError: If the lengths of `source_amplitudes` and
            `source_locations` do not match, or if a pair of
            amplitude/location is inconsistent (one is None and the other is
            not), or if their `n_sources_per_shot` dimensions do not match.
        TypeError: If `source_amplitudes` or `source_locations` are not
            `torch.Tensor` when expected to be.

    """
    if len(source_amplitudes) != len(source_locations):
        raise RuntimeError(
            f"The same number of source_amplitudes ({len(source_amplitudes)}) "
            f"and source_locations ({len(source_locations)}) must be provided.",
        )
    for amplitudes, locations in zip(source_amplitudes, source_locations):
        if (amplitudes is None) != (locations is None):
            raise RuntimeError(
                "Each pair of source locations and amplitudes must both be "
                "None or both be non-None.",
            )
        if amplitudes is not None and locations is not None:
            if not isinstance(amplitudes, torch.Tensor):
                raise TypeError("source_amplitudes must be a torch.Tensor.")
            if not isinstance(locations, torch.Tensor):
                raise TypeError("source_locations must be a torch.Tensor.")
            if amplitudes.shape[1] != locations.shape[1]:
                raise RuntimeError(
                    "Expected source amplitudes and locations to be the same "
                    "size in the n_sources_per_shot dimension, got "
                    f"{amplitudes.shape[1]} and {locations.shape[1]}.",
                )


def set_source_amplitudes(
    source_amplitudes: Sequence[Optional[torch.Tensor]],
    n_batch: int,
    nt: int,
    step_ratio: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Prepares source amplitudes for propagation.

    This function validates the input source amplitudes, ensures consistency
    in device, dtype, and shape, and upsamples them if necessary.

    Args:
        source_amplitudes: A sequence of optional Tensors, where each
            torch.Tensor contains the amplitudes of the sources for a given shot.
        n_batch: The batch size.
        nt: The total number of time steps.
        step_ratio: The ratio between the user-specified time step and the
            internal time step.
        freq_taper_frac: The fraction of the frequency spectrum to taper.
        time_pad_frac: The fraction of the time axis to pad with zeros.
        time_taper: Whether to apply a Hann window in time.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.

    Returns:
        List[torch.Tensor]: A list of processed source amplitude Tensors.

    Raises:
        TypeError: If `source_amplitudes` contains elements that are not
            `torch.Tensor` or `None`.
        RuntimeError: If there are inconsistencies in the dimensions, device,
            dtype, or batch size of the source amplitudes.

    """
    result: List[torch.Tensor] = []
    for amplitudes in source_amplitudes:
        if amplitudes is None:
            result.append(torch.empty(nt, n_batch, 0, device=device, dtype=dtype))
            continue
        if not isinstance(amplitudes, torch.Tensor):
            raise TypeError("source_amplitudes must be a torch.Tensor.")
        if amplitudes.ndim != 3:
            raise RuntimeError(
                "source amplitudes Tensors should have 3 dimensions, but "
                f"found one with {amplitudes.ndim}.",
            )
        if amplitudes.device != device:
            raise RuntimeError(
                "Inconsistent device: Expected all Tensors be on device "
                f"{device}, but found a source amplitudes torch.Tensor on "
                f"device {amplitudes.device}.",
            )
        if amplitudes.dtype != dtype:
            raise RuntimeError(
                "Inconsistent dtype: Expected source amplitudes to have "
                f"datatype {dtype}, but found one with dtype "
                f"{amplitudes.dtype}.",
            )
        if amplitudes.shape[0] != n_batch:
            raise RuntimeError(
                f"Expected source amplitudes to have size {n_batch} in the "
                "batch dimension, but found one with size "
                f"{amplitudes.shape[0]}.",
            )
        if amplitudes.shape[2] * step_ratio != nt:
            raise RuntimeError(
                "Inconsistent number of time samples: Expected source "
                f"amplitudes to have {nt // step_ratio} time samples, but "
                f"found one with {amplitudes.shape[2]}.",
            )
        result.append(
            torch.movedim(
                upsample(
                    amplitudes,
                    step_ratio=step_ratio,
                    freq_taper_frac=freq_taper_frac,
                    time_pad_frac=time_pad_frac,
                    time_taper=time_taper,
                ),
                -1,
                0,
            ).contiguous(),
        )
    return result


def check_points_per_wavelength(
    min_nonzero_vel: float, pml_freq: float, grid_spacing: Sequence[float],
) -> None:
    """Checks if there are enough grid points per wavelength.

    Warns the user if the number of grid cells per wavelength falls below
    a recommended threshold (6 cells per wavelength).

    Args:
        min_nonzero_vel: The minimum non-zero velocity in the model.
        pml_freq: The dominant frequency for PML absorption.
        grid_spacing: The spacing between grid points in each spatial dimension.

    Raises:
        ValueError: If `min_nonzero_vel`, `pml_freq`, or any element of
            `grid_spacing` is negative or zero.

    """
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
            "At least six grid cells per wavelength is recommended, but at a "
            f"frequency of {pml_freq}, a minimum non-zero velocity of "
            f"{min_nonzero_vel}, and a grid cell spacing of {max_spacing}, "
            f"there are only {cells_per_wavelength:.2f}.",
        )


def cosine_taper_end(signal: torch.Tensor, n_taper: int) -> torch.Tensor:
    """Tapers the end of the final dimension of a torch.Tensor using a cosine.

    A half period, shifted and scaled to taper from 1 to 0, is used.

    Args:
        signal:
            The torch.Tensor that will have its final dimension tapered.
        n_taper:
            The length of the cosine taper, in number of samples.

    Returns:
        The tapered signal.

    """
    if not isinstance(signal, torch.Tensor):
        raise TypeError("signal must be a torch.Tensor.")
    if not isinstance(n_taper, int):
        raise TypeError("n_taper must be an int.")
    if n_taper < 0:
        raise ValueError("n_taper must be non-negative.")
    taper = torch.ones(signal.shape[-1], dtype=signal.dtype, device=signal.device)
    n_taper = min(n_taper, signal.shape[-1])
    if n_taper == 0:
        return signal * taper
    taper[len(taper) - n_taper :] = (
        torch.cos(
            torch.arange(1, n_taper + 1, device=signal.device) / n_taper * math.pi,
        )
        + 1
    ).to(signal.dtype) / 2
    return signal * taper


def zero_last_element_of_final_dimension(signal: torch.Tensor) -> torch.Tensor:
    """Sets the last element of the final dimension of a torch.Tensor to zero.

    Args:
        signal: The torch.Tensor to modify.

    Returns:
        The modified torch.Tensor with the last element of its final dimension
        set to zero.

    Raises:
        TypeError: If `signal` is not a `torch.Tensor`.

    """
    if not isinstance(signal, torch.Tensor):
        raise TypeError("signal must be a torch.Tensor.")
    if signal.numel() == 0:
        return signal
    zeroer = torch.ones(signal.shape[-1], dtype=signal.dtype, device=signal.device)
    zeroer[-1] = 0
    return signal * zeroer


def upsample(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
) -> torch.Tensor:
    """Upsamples the final dimension of a torch.Tensor by a factor.

    Low-pass upsampling is used to produce an upsampled signal without
    introducing higher frequencies than were present in the input. The
    Nyquist frequency of the input will be zeroed.

    Args:
        signal:
            The torch.Tensor that will have its final dimension upsampled.
        step_ratio:
            The integer factor by which the signal will be upsampled.
            The input signal is returned if this is 1 (freq_taper_frac,
            time_pad_frac, and time_taper will be ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered before
            upsampling. Defaults to 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before upsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the input signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the signal will be used. Defaults to 0.0.
        time_taper:
            A bool specifying whether to apply a Hann window in time.
            This is useful during correctness tests of the propagators
            as it ensures that signals taper to zero at their edges in
            time, avoiding the possibility of high frequencies being
            introduced.

    Returns:
        The signal after upsampling.

    """
    if not isinstance(signal, torch.Tensor):
        raise TypeError("signal must be a torch.Tensor.")
    if signal.numel() == 0:
        return signal

    if not isinstance(step_ratio, int):
        raise TypeError("step_ratio must be an int.")
    if step_ratio <= 0:
        raise ValueError("step_ratio must be positive.")
    try:
        freq_taper_frac = float(freq_taper_frac)
    except (TypeError, ValueError):
        raise TypeError("freq_taper_frac must be a float.")
    if not 0.0 <= freq_taper_frac <= 1.0:
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}.")
    try:
        time_pad_frac = float(time_pad_frac)
    except (TypeError, ValueError):
        raise TypeError("time_pad_frac must be a float.")
    if not 0.0 <= time_pad_frac <= 1.0:
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
    signal_f = torch.fft.rfft(signal, norm="ortho") * math.sqrt(step_ratio)
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)
    pad_len = up_nt // 2 + 1 - signal_f.shape[-1]
    if pad_len > 0:
        signal_f = torch.nn.functional.pad(signal_f, (0, pad_len))
    signal = torch.fft.irfft(signal_f, n=up_nt, norm="ortho")
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad * step_ratio]
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1], periodic=False, device=signal.device,
        )
    return signal


def downsample(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> torch.Tensor:
    """Downsamples the final dimension of a torch.Tensor by a factor.

    Frequencies higher than or equal to the Nyquist frequency of the
    downsampled signal will be zeroed before downsampling.

    Args:
        signal:
            The torch.Tensor that will have its final dimension downsampled.
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
            downsampling. Defaults to 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before downsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the output signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the output signal will be used.
            Defaults to 0.0.
        time_taper:
            A bool specifying whether to apply a Hann window in time.
            This is useful during correctness tests of the propagators
            as it ensures that signals taper to zero at their edges in
            time, avoiding the possibility of high frequencies being
            introduced.
        shift:
            Amount (in units of time samples) to shift the data in time
            before downsampling.
            Defaults to 0.0.

    Returns:
        The signal after downsampling.

    """
    if not isinstance(signal, torch.Tensor):
        raise TypeError("signal must be a torch.Tensor.")
    if signal.numel() == 0:
        return signal

    if not isinstance(step_ratio, int):
        raise TypeError("step_ratio must be an int.")
    if step_ratio <= 0:
        raise ValueError("step_ratio must be positive.")
    try:
        freq_taper_frac = float(freq_taper_frac)
    except (TypeError, ValueError):
        raise TypeError("freq_taper_frac must be a float.")
    if not 0.0 <= freq_taper_frac <= 1.0:
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}.")
    try:
        time_pad_frac = float(time_pad_frac)
    except (TypeError, ValueError):
        raise TypeError("time_pad_frac must be a float.")
    if not 0.0 <= time_pad_frac <= 1.0:
        raise ValueError(f"time_pad_frac must be in [0, 1], got {time_pad_frac}.")
    if not isinstance(time_taper, bool):
        raise TypeError("time_taper must be a bool.")
    try:
        shift = float(shift)
    except (TypeError, ValueError):
        raise TypeError("shift must be a float.")

    if step_ratio == 1 and shift == 0.0:
        return signal
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1], periodic=False, device=signal.device,
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
    signal_f = torch.fft.rfft(signal, norm="ortho")[..., : down_nt // 2 + 1]
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)
    if shift != 0.0:
        freqs = torch.fft.rfftfreq(signal.shape[-1], device=signal.device)[
            : down_nt // 2 + 1
        ]
        signal_f *= torch.exp(-1j * 2 * math.pi * freqs * shift)
    signal = torch.fft.irfft(signal_f, n=down_nt, norm="ortho") / math.sqrt(step_ratio)
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad]
    return signal


def extract_survey(
    models: Sequence[torch.Tensor],
    source_locations: Sequence[Optional[torch.Tensor]],
    receiver_locations: Sequence[Optional[torch.Tensor]],
    wavefields: Sequence[Optional[torch.Tensor]],
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]],
    origin: Optional[Sequence[int]],
    fd_pad: Sequence[int],
    pml_width: Sequence[int],
    model_pad_modes: Sequence[str],
    n_batch: int,
    n_dims: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
]:
    """Extracts a region of the model, pads it and prepares related tensors.

    The region can either be chosen based on `survey_pad` and the
    source and receiver locations, or using `origin` and initial
    wavefields.

    This function extracts the appropriate region of the model and
    adjusts source and receiver locations to be relative to the
    new origin.

    This function performs necessary processing to prepare the models and
    locations. Padding is added to the models, and the locations are adjusted
    appropriately. A batch dimension is added to the models if they don't
    already have one. The device and dtype of all models and locations are
    checked and they are made contiguous. Locations that are None are changed
    to be empty Tensors.

    Args:
        models: A sequence of PyTorch Tensors representing the model
            parameters (e.g., velocity, density).
        source_locations: A sequence of optional Tensors, where each
            torch.Tensor contains the locations of the sources for a given shot.
        receiver_locations: A sequence of optional Tensors, where each
            torch.Tensor contains the locations of the receivers for a given shot.
        wavefields: A sequence of optional Tensors representing initial
            wavefields.
        survey_pad: Optional padding around the survey area. Can be an int
            or a sequence of optional ints.
        origin: Optional sequence of integers specifying the origin of the
            survey within the full model.
        fd_pad: A sequence of integers specifying the padding for finite
            difference stencils in each dimension.
        pml_width: The width of the PML (Perfectly Matched Layer) in grid
            cells.
        model_pad_modes: A sequence of strings specifying the padding mode
            for each model.
        n_batch: The batch size.
        n_dims: The number of spatial dimensions of the model.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.

    Returns:
        Tuple:

            - models_out: Processed model Tensors.
            - source_locations_out: Processed source location Tensors.
            - receiver_locations_out: Processed receiver location Tensors.
            - wavefields_out: Processed wavefield Tensors.

    """
    if any(not isinstance(m, torch.Tensor) for m in models):
        raise TypeError("Models must be a torch.Tensor.")
    if survey_pad is not None and origin is not None:
        raise RuntimeError("survey_pad and origin cannot both be specified.")
    locations: List[Optional[torch.Tensor]] = list(source_locations) + list(
        receiver_locations,
    )
    model_spatial_shape = list(models[0].shape[-n_dims:])
    check_locations_are_within_model(model_spatial_shape, locations)
    pad = [fd + pml for fd, pml in zip(fd_pad, pml_width)]
    if survey_pad is None and any(w is not None for w in wavefields):
        survey_extents = get_survey_extents_from_wavefields(
            wavefields, origin, pml_width,
        )
        check_extents_within_model(survey_extents, model_spatial_shape)
        check_locations_within_extents(survey_extents, locations)
    else:
        survey_extents = get_survey_extents_from_locations(
            model_spatial_shape, locations, survey_pad,
        )
        check_extents_match_wavefields_shape(survey_extents, wavefields, pml_width)
    return (
        list(
            extract_models(
                models, survey_extents, pad, model_pad_modes, n_batch, device, dtype,
            ),
        ),
        list(
            extract_locations(
                "Source", source_locations, survey_extents, pad, n_batch, device,
            ),
        ),
        list(
            extract_locations(
                "Receiver", receiver_locations, survey_extents, pad, n_batch, device,
            ),
        ),
        list(
            prepare_wavefields(
                wavefields, survey_extents, pml_width, n_batch, device, dtype,
            ),
        ),
    )


def check_locations_are_within_model(
    model_shape: Sequence[int],
    locations: Sequence[Optional[torch.Tensor]],
) -> None:
    """Checks if all locations are within the bounds of the model.

    Args:
        model_shape: A sequence of integers representing the spatial shape
            of the model.
        locations: A sequence of optional Tensors, where each torch.Tensor
            contains the locations to check.

    Raises:
        RuntimeError: If `model_shape` is empty, contains non-positive
            elements, or if any location is outside the model bounds.

    """
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
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]], ndim: int,
) -> List[int]:
    """Checks `survey_pad` and converts it to a list if it is a scalar.

    Args:
        survey_pad: Optional padding around the survey area. Can be an int
            or a sequence of optional ints.
        ndim: The number of spatial dimensions.

    Returns:
        A list of integers representing the padding for each side of each
        dimension. Negative values indicate padding to the model edge.

    Raises:
        RuntimeError: If `ndim` is not positive, or if `survey_pad` contains
            invalid values or has an incorrect length.

    """
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
                        "survey_pad entries must be None or non-negative ints.",
                    )
            else:
                raise RuntimeError(
                    "survey_pad entries must be None or non-negative ints.",
                )
    if len(survey_pad_list) != 2 * ndim:
        raise RuntimeError(
            f"survey_pad must have length 2 * dims in model, but got "
            f"{len(survey_pad_list)}.",
        )
    return survey_pad_list


def get_survey_extents_from_locations(
    model_shape: Sequence[int],
    locations: Sequence[Optional[torch.Tensor]],
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
            left_pad, "left", dim, locations, model_shape[dim],
        )
        right_pad = survey_pad_list[dim * 2 + 1]
        right_extent = get_survey_extents_one_side(
            right_pad, "right", dim, locations, model_shape[dim],
        )
        extents.append((left_extent, right_extent))
    return extents


def get_survey_extents_one_side(
    pad: int,
    side: str,
    dim: int,
    locations: Sequence[Optional[torch.Tensor]],
    shape: int,
) -> int:
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
    # the other side of the model so that we can find the min (left) or
    # max (right) location below.
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
            # Check if the location is within the model bounds. This check is
            # duplicated from check_locations_are_within_model but is
            # necessary here to ensure that the extents are calculated
            # correctly.
            if dim_location.min() < 0:
                raise RuntimeError("Locations must be >= 0.")
            if dim_location.max() >= shape:
                raise RuntimeError(
                    f"Locations must be within model bounds (0 to {shape - 1}).",
                )

            if side == "left":
                extreme_location = min(
                    extreme_location, int(dim_location.min().item()) - pad,
                )
            else:
                extreme_location = max(
                    extreme_location, int(dim_location.max().item()) + pad,
                )
    # The right side of the extent is not included - it is one past the end. It
    # currently holds the largest value that should be included, though, so we
    # add 1.
    if side == "right":
        extreme_location += 1
    # The extents should not be outside the model, so we clip them to the
    # extents of the model.
    extreme_location = max(extreme_location, 0)
    extreme_location = min(extreme_location, shape)
    return extreme_location


def get_survey_extents_from_wavefields(
    wavefields: Sequence[Optional[torch.Tensor]],
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
            A list containing the padding to add on each side for the PML,
            using the specified pad mode.

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
                    f"wavefields must have {ndims + 1} dimensions (batch + spatial).",
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
                    "Survey extents were inferred from the wavefield shape to "
                    f"be {extents} because origin was not provided.",
                )
            return extents
    raise RuntimeError("At least one wavefield must be non-None.")


def check_extents_within_model(
    extents: Sequence[Tuple[int, int]], model_shape: Sequence[int],
) -> None:
    """Checks if the survey extents are within the bounds of the model.

    Args:
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        model_shape: A sequence of integers representing the spatial shape
            of the model.

    Raises:
        RuntimeError: If any survey extent is outside the model bounds.

    """
    for extent, model_dim_shape in zip(extents, model_shape):
        if extent[0] < 0 or model_dim_shape < extent[1]:
            raise RuntimeError(
                "Survey extents are larger than the model. "
                 "This probably occurred because you provided "
                 "an initial wavefield of the wrong shape.",
            )


def check_locations_within_extents(
    extents: Sequence[Tuple[int, int]],
    locations: Sequence[Optional[torch.Tensor]],
) -> None:
    """Checks if all locations are within the specified survey extents.

    Args:
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        locations: A sequence of optional Tensors, where each torch.Tensor
            contains the locations to check.

    Raises:
        RuntimeError: If any location is outside the specified survey extents.

    """
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
                        "because you provided an "
                        "initial wavefield that does not cover all "
                        "source and receiver locations, given the "
                        "specified origin.",
                    )


def check_extents_match_wavefields_shape(
    extents: Sequence[Tuple[int, int]],
    wavefields: Sequence[Optional[torch.Tensor]],
    pad: Sequence[int],
) -> None:
    """Checks if wavefield shapes match the calculated extents and padding.

    Args:
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        wavefields: A sequence of optional Tensors representing initial
            wavefields.
        pad: A sequence of integers specifying the padding for finite
            difference stencils and PML in each dimension.

    Raises:
        RuntimeError: If the shape of any provided wavefield does not match
            the expected shape based on the extents and padding.

    """
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
                        f"in dimension {dim}, {wavefield.shape[1 + dim]}, "
                        "does not match the extent determined by the "
                        f"specified survey_pad, {expected}.",
                    )


def reverse_pad(pad: Sequence[int]) -> List[int]:
    """Reverses the order of padding for each dimension.

    Given a sequence of padding values in the order
    [dim0_start, dim0_end, dim1_start, dim1_end, ...], this function returns
    a new list with the padding for each dimension reversed.

    Args:
        pad: A sequence of integers representing padding values for each
            dimension. Expected format:
            [dim0_start, dim0_end, dim1_start, dim1_end, ...]

    Returns:
        List[int]: A new list with the dimension order reversed. Format:
            [dim_N_start, dim_N_end, ..., dim1_start, dim1_end,
             dim0_start, dim0_end].

    """
    n_dims = len(pad) // 2
    reversed_pad: List[int] = []
    for dim in range(n_dims - 1, -1, -1):
        reversed_pad.extend([pad[2 * dim], pad[2 * dim + 1]])
    return reversed_pad


def extract_models(
    models: Sequence[torch.Tensor],
    extents: Sequence[Tuple[int, int]],
    pad: Sequence[int],
    pad_modes: Sequence[str],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Extract the specified portion of the models and prepares them.

    Args:
        models:
            A list of Tensors to extract from.
        extents:
            A list of tuples specifying the portion of the model to extract.
        pad:
            An list of integers specifying the padding to add on each side
        pad_modes:
            A list of strings specifying the pad mode to use for the PML
            region of each model.
        n_batch:
            An integer specifying the size of the batch dimension.
        device:
            The device that the models should be located on.
        dtype:
            The datatype that the models should have.

    Returns:
        A list containing the desired portion of the models, with padding
        applied and a batch dimension of size either 1 or n_batch.

    """
    n_dims = len(pad) // 2
    assert len(models) == len(pad_modes)
    assert len(extents) == n_dims

    # Check all models have correct dimensions and consistent attributes
    spatial_shape = models[0].shape[-n_dims:]
    for model in models:
        if not isinstance(model, torch.Tensor):
            raise TypeError("Models must be a torch.Tensor.")
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
                f"The batch dimension of models must be of size 1 or {n_batch}.",
            )

    region = (slice(None),) + tuple(slice(begin, end) for begin, end in extents)
    reversed_pad = reverse_pad(pad)

    if n_batch == 0:
        return [
            torch.empty(0, *model[region].shape[1:], device=device, dtype=dtype)
            for model in models
        ]
    return [
        torch.nn.functional.pad(
            model[region], pad=reversed_pad, mode=pad_mode,
        ).contiguous()
        for model, pad_mode in zip(models, pad_modes)
    ]


def extract_locations(
    name: str,
    locations: Sequence[Optional[torch.Tensor]],
    extents: Sequence[Tuple[int, int]],
    pad: Sequence[int],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    eps: float = 0.1,
) -> List[torch.Tensor]:
    """Sets locations relative to the extracted model and prepares them.

    Locations are returned as 1D indices into each batch.

    Args:
        name: A string indicating the type of locations
            (e.g., "Source", "Receiver").
        locations: A sequence of optional Tensors, where each torch.Tensor
            contains the locations to process.
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        pad: A sequence of integers specifying the padding for finite
            difference stencils and PML in each dimension.
        n_batch: The batch size.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.
            Defaults to `torch.long`.
        eps: A small float value used for checking if locations are
            integer-like. Defaults to 0.1.

    Returns:
        List[torch.Tensor]: A list of processed location Tensors, where each
            location is converted to a 1D index relative to the extracted
            and padded model.

    Raises:
        TypeError: If `locations` contains elements that are not
            `torch.Tensor` or `None`.
        RuntimeError: If there are inconsistencies in the dimensions, batch
            size, or if locations are not unique within a shot.

    """
    n_dims = len(extents)
    origin: List[int] = []  # origin of extracted survey, including padding
    shape: List[int] = []  # shape of extracted survey, including padding
    stride: List[int] = [
        1,
    ] * n_dims  # stride of each dim in extracted survey, including padding
    for dim in range(n_dims):
        origin.append(extents[dim][0] - pad[2 * dim])
        shape.append(
            extents[dim][1] - extents[dim][0] + pad[2 * dim] + pad[2 * dim + 1],
        )
    for dim in range(n_dims - 2, -1, -1):
        stride[dim] = stride[dim + 1] * shape[dim + 1]

    extracted_locations: List[torch.Tensor] = []
    for location in locations:
        if location is not None:
            if not isinstance(location, torch.Tensor):
                raise TypeError("locations must be a torch.Tensor.")
            if location.numel() > 0 and eps < (location - location.long()).abs().max():
                warnings.warn(
                    "Locations should be specified as integer numbers "
                    "of cells. If you wish to have a source or receiver "
                    "that is not centred on a cell, please consider "
                    "using the Hick's method, which is implemented "
                    "in deepwave.location_interpolation.",
                )

            if location.ndim != 3:
                raise RuntimeError(
                    name + " location Tensors must have three dimensions",
                )

            if location.shape[0] != n_batch:
                raise RuntimeError(
                    "Inconsistent batch size: Expected all Tensors to have a "
                    f"batch size of {n_batch}, but found a {name.lower()} "
                    "locations torch.Tensor with a batch size of "
                    f"{location.shape[0]}.",
                )

            if location.shape[-1] != n_dims:
                raise RuntimeError(
                    f"{name} locations must have {n_dims} dimensional "
                    f"coordinates, but found one with {location.shape[-1]}.",
                )

            # Shift locations to be relative to new origin in extracted
            # (and padded) model
            shifted_location = location.clone().long().to(device)
            for dim in range(n_dims):
                shifted_location[..., dim] = torch.where(
                    shifted_location[..., dim] != IGNORE_LOCATION,
                    shifted_location[..., dim] - origin[dim],
                    IGNORE_LOCATION,
                )

            # Convert locations to 1d coordinate
            location_1d = torch.where(
                shifted_location[..., 0] != IGNORE_LOCATION, 0, IGNORE_LOCATION,
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
                if len(shot_locations) != len(torch.unique(shot_locations)):
                    raise RuntimeError(
                        f"{name} locations must be unique within each shot. "
                        "You cannot have two in the same cell, but in shot "
                        f"{batch_idx} there is/are {len(shot_locations)} "
                        f"active {name.lower()} locations while only "
                        f"{len(torch.unique(shot_locations))} is/are unique.",
                    )

            extracted_locations.append(location_1d.contiguous())
        else:
            extracted_locations.append(
                torch.empty(n_batch, 0, device=device, dtype=dtype),
            )
    return extracted_locations


def prepare_wavefields(
    wavefields: Sequence[Optional[torch.Tensor]],
    extents: Sequence[Tuple[int, int]],
    pad: Sequence[int],
    n_batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Checks and prepares initial wavefields.

    This function validates the input wavefields, ensures consistency
    in device, dtype, and shape, and replaces `None` wavefields with
    zero-filled tensors of the appropriate shape.

    Args:
        wavefields: A sequence of optional Tensors representing initial
            wavefields.
        extents: A sequence of tuples, where each tuple represents the
            (start, end) coordinates of the survey extent for a dimension.
        pad: A sequence of integers specifying the padding for finite
            difference stencils and PML in each dimension.
        n_batch: The batch size.
        device: The PyTorch device to which the tensors should be moved.
        dtype: The PyTorch data type to which the tensors should be cast.

    Returns:
        A list of processed wavefield Tensors, with `None` values
        replaced by zero tensors.

    Raises:
        TypeError: If `wavefields` contains elements that are not
            `torch.Tensor` or `None`.
        RuntimeError: If there are inconsistencies in the dimensions, device,
            dtype, batch size, or spatial shape of the wavefields.

    """
    n_dims = len(extents)
    spatial_shape = [
        extents[dim][1] - extents[dim][0] + pad[2 * dim] + pad[2 * dim + 1]
        for dim in range(n_dims)
    ]
    prepared_wavefields: List[torch.Tensor] = []
    for wavefield in wavefields:
        if wavefield is not None:
            if not isinstance(wavefield, torch.Tensor):
                raise TypeError("Wavefields must be a torch.Tensor.")
            if wavefield.device != device:
                raise RuntimeError(
                    "Inconsistent device: Expected all Tensors be on device "
                    f"{device}, but found a wavefield torch.Tensor on device "
                    f"{wavefield.device}.",
                )
            if wavefield.dtype != dtype:
                raise RuntimeError(
                    "Inconsistent dtype: Expected wavefields to have "
                    f"datatype {dtype}, but found one with dtype "
                    f"{wavefield.dtype}.",
                )
            if wavefield.ndim != n_dims + 1:
                raise RuntimeError(
                    f"Wavefields must have {n_dims + 1} dimensions, but found "
                    f"one with {wavefield.ndim}.",
                )
            if wavefield.shape[0] != n_batch:
                raise RuntimeError(
                    "Inconsistent batch size: Expected all Tensors to have a "
                    f"batch size of {n_batch}, but found a wavefield with a "
                    f"batch size of {wavefield.shape[0]}.",
                )
            if list(wavefield.shape[1:]) != spatial_shape:
                raise RuntimeError(
                    "Inconsistent spatial shape: Expected wavefield to have "
                    f"spatial shape {spatial_shape} but found one with "
                    f"spatial shape {list(wavefield.shape[1:])}.",
                )
            prepared_wavefields.append(wavefield)
        else:
            prepared_wavefields.append(
                torch.zeros(n_batch, *spatial_shape, device=device, dtype=dtype),
            )
    return prepared_wavefields


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
        grid_spacing: A List specifying the grid spacing in each spatial
            dimension.
        dt: The time step interval.
        max_abs_vel: The maximum absolute wavespeed in the model.
        eps: A small quantity to prevent division by zero. Default 1e-15.
        c_max: The maximum allowable Courant number.

    Returns:
        Tuple:

            - inner_dt: A time step interval that obeys the CFL
              condition.
            - step_ratio: The integer dt / inner_dt.

    """
    try:
        grid_spacing = list(grid_spacing)
    except TypeError:
        raise TypeError("grid_spacing must be a list of floats.")
    for i, spacing in enumerate(grid_spacing):
        try:
            grid_spacing[i] = float(spacing)
        except ValueError:
            raise TypeError("grid_spacing must be a list of floats.")
    if not grid_spacing:
        raise ValueError("grid_spacing must not be empty.")
    for g in grid_spacing:
        if g <= 0:
            raise ValueError("grid_spacing elements must be positive.")
    try:
        dt = float(dt)
    except (TypeError, ValueError):
        raise TypeError("dt must be a float.")
    try:
        max_abs_vel = float(max_abs_vel)
    except (TypeError, ValueError):
        raise TypeError("max_abs_vel must be a float.")
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


def cfl_condition(dy: float, dx: float, *args: Any, **kwargs: Any) -> Tuple[float, int]:
    """Calculates the time step interval for 2D models.

    This is a convenience wrapper around `cfl_condition_n` for 2D models.

    Args:
        dy: The grid spacing in the y-dimension.
        dx: The grid spacing in the x-dimension.
        *args: Additional positional arguments to pass to `cfl_condition_n`.
        **kwargs: Additional keyword arguments to pass to `cfl_condition_n`.

    Returns:
        Tuple:

            - inner_dt: A time step interval that obeys the CFL
              condition.
            - step_ratio: The integer dt / inner_dt.

    """
    return cfl_condition_n([dy, dx], *args, **kwargs)


def vpvsrho_to_lambmubuoyancy(
    vp: torch.Tensor,
    vs: torch.Tensor,
    rho: torch.Tensor,
    eps: float = 1e-15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts vp, vs, rho to lambda, mu, buoyancy.

    All input Tensors must have the same shape.

    Args:
        vp: A torch.Tensor containing the p wavespeed.
        vs: A torch.Tensor containing the s wavespeed.
        rho: A torch.Tensor containing the density.
        eps: An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple:

            - lambda: A torch.Tensor containing the first Lamé parameter.
            - mu: A torch.Tensor containing the second Lamé parameter.
            - buoyancy: A torch.Tensor containing the reciprocal of density.

    """
    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = 1 / (rho**2 + eps) * rho
    return lamb, mu, buoyancy


def lambmubuoyancy_to_vpvsrho(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    eps: float = 1e-15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts lambda, mu, buoyancy to vp, vs, rho.

    All input Tensors must have the same shape.

    Args:
        lamb: A torch.Tensor containing the first Lamé parameter.
        mu: A torch.Tensor containing the second Lamé parameter.
        buoyancy: A torch.Tensor containing the reciprocal of density.
        eps: An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple:

            - vp: A torch.Tensor containing the p wavespeed.
            - vs: A torch.Tensor containing the s wavespeed.
            - rho: A torch.Tensor containing the density.

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
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float = 0.0,
    eps: float = 1e-9,
    r_val: float = 0.001,
    n_power: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates a and b profiles for C-PML.

    Only the first fd_pad[0]+pml_width[0] and last fd_pad[1]+pml_width[1]
    elements of the profiles will be non-zero.

    Args:
        pml_width: List of two integers specifying the width of the PML
            region.
        pml_start: List of two floats specifying the coordinates (in grid
            cells) of the start of the PML regions.
        max_pml: Float specifying the length (in distance units) of the
            longest PML over all sides and dimensions.
        dt: Time step interval.
        n: Integer specifying desired profile length, including fd_pad and
           pml_width.
        max_vel: Maximum wave speed.
        dtype: PyTorch datatype to use.
        device: PyTorch device to use.
        pml_freq: The frequency value to use for the profile, usually the
            dominant frequency in the wavefield.
        start: Float specifying the coordinate (in grid cells) of the first
            element. Optional, default 0.
        eps: A small number to prevent division by zero. Optional,
            default 1e-9.
        r_val: The reflection coefficient. Optional, default 0.001.
        n_power: The power for the profile. Optional, default 2.

    Returns:
        A tuple containing the a and b profiles as Tensors.

    """
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
    tensor: torch.Tensor,
    fd_pad: Union[int, Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
    size: Sequence[int],
) -> torch.Tensor:
    """Creates a zero tensor of a specified size or pads an existing tensor.

    If the input `tensor` is empty (numel == 0), a new zero tensor with the
    given `size` is created. Otherwise, the `tensor` is padded according to
    `fd_pad`.

    Args:
        tensor: The input torch.Tensor to be created or padded.
        fd_pad: The padding to apply. Can be an integer (for uniform padding)
            or a sequence of integers (for per-dimension padding).
        device: The PyTorch device for the tensor.
        dtype: The PyTorch data type for the tensor.
        size: The desired size of the tensor if it needs to be created.

    Returns:
        torch.Tensor: The created or padded torch.Tensor.

    """
    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * len(size) * 2
    if tensor.numel() == 0:
        return torch.zeros(size[0], size[1], size[2], device=device, dtype=dtype)
    if max(fd_pad) == 0:
        return tensor.clone()
    return (
        torch.nn.functional.pad(tensor, (fd_pad[2], fd_pad[3], fd_pad[0], fd_pad[1]))
    ).requires_grad_(tensor.requires_grad)


def zero_interior(
    tensor: torch.Tensor,
    fd_pad: Union[int, Sequence[int]],
    pml_width: Sequence[int],
    y: bool,
) -> torch.Tensor:
    """Zeros out the interior region of a 2D tensor.

    This function is typically used for debugging or visualization purposes to
    inspect the effects of padding and PML.

    Args:
        tensor: The input 2D torch.Tensor.
        fd_pad: The finite-difference padding. Can be an integer or a sequence
            of integers [top, bottom, left, right].
        pml_width: The width of the PML regions for each side
            [top, bottom, left, right].
        y: A boolean indicating whether to zero along the y-dimension (True)
            or x-dimension (False).

    Returns:
        torch.Tensor: A new torch.Tensor with the interior region zeroed out.

    """
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


def diff(a: torch.Tensor, accuracy: int, grid_spacing: float) -> torch.Tensor:
    """Calculates the spatial derivative of a 1D tensor.

    Args:
        a: The input 1D torch.Tensor.
        accuracy: The finite-difference accuracy order (2, 4, 6, or 8).
        grid_spacing: The spacing between grid points.

    Returns:
        The spatial derivative of the input tensor.

    """
    if accuracy == 2:
        # Coefficients from Wikipedia
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient
        coeffs = [-1 / 2, 1 / 2]
        stencil = [-1, 1]
    elif accuracy == 4:
        coeffs = [1 / 12, -2 / 3, 2 / 3, -1 / 12]
        stencil = [-2, -1, 1, 2]
    elif accuracy == 6:
        coeffs = [-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60]
        stencil = [-3, -2, -1, 1, 2, 3]
    else:
        coeffs = [1 / 280, -4 / 105, 1 / 5, -4 / 5, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
        stencil = [-4, -3, -2, -1, 1, 2, 3, 4]
    return (
        sum(
            (c * torch.roll(a, -s, dims=-1) for c, s in zip(coeffs, stencil)),
            start=torch.zeros_like(a, memory_format=torch.contiguous_format),
        ) / grid_spacing
    )

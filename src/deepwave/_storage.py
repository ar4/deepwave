"""Wavefield storage abstraction for Deepwave propagators.

This module holds the storage-mode enum, the public :class:`Storage` policy
object, the single funnel that resolves the several spellings of the storage
arguments into one answer, and the classes that manage intermediate wavefield
storage (device, CPU, or disk) used by the backward pass of every propagator.

It is deliberately a leaf: it imports nothing from Deepwave except
``_exceptions`` (which itself imports nothing), so that every other module is
free to import it. The two places that need Deepwave's warn-once registry
import it inside the function body for the same reason.
"""

import contextlib
import ctypes
import dataclasses
import math
import os
import shutil
import tempfile
import threading
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast
from uuid import uuid4

import torch

from deepwave._exceptions import DeepwaveOSError, DeepwaveValueError

if TYPE_CHECKING:  # pragma: no cover -- typing only; keeps this module a leaf
    from deepwave.types import StorageMode as _StorageModeStr
    from deepwave.types import StoragePath

#: The four values ``mode`` accepts, in the order the error message lists
#: them. Single-sourced here so that :class:`Storage`, the engine's
#: ``resolve_storage_mode`` and :func:`deepwave.estimate_memory` cannot drift
#: apart in either the accepted set or the wording of the rejection.
STORAGE_MODES: tuple[str, ...] = ("device", "cpu", "disk", "none")


def storage_mode_error_message(value: object) -> str:
    """The one wording for an unrecognised ``storage_mode``/``mode``.

    The text is the one the engine has always used, character for character,
    because it is the wording users have seen and tests pin. The duplicate
    that had grown up alongside it interpolated ``repr(value)`` and ended in a
    full stop; that copy is gone and this is now the only source.

    Args:
        value: The rejected value, interpolated into the message.

    Returns:
        The message text, identical whichever of the three call paths
        (propagator, :class:`Storage`, :func:`deepwave.estimate_memory`)
        rejected the value.

    """
    return f"storage_mode must be 'device', 'cpu', 'disk', or 'none', but got {value}"


class StorageMode(IntEnum):
    """Storage modes for intermediate wavefields."""

    DEVICE = 0
    CPU = 1
    DISK = 2
    NONE = 3


def storage_mode_enum(value: object) -> StorageMode:
    """Maps a ``storage_mode`` string to the backend's internal enum.

    The one place the mapping and its rejection message live. Three copies of
    both had grown up (the engine's ``resolve_storage_mode``, the engine's
    python-backend option check, and ``estimate_memory``) and two of them had
    already drifted apart in user-visible text.

    Args:
        value: The ``storage_mode`` string to map.

    Returns:
        The corresponding :class:`StorageMode`.

    Raises:
        DeepwaveValueError: If ``value`` is not one of the four modes.

    """
    modes = {
        "device": StorageMode.DEVICE,
        "cpu": StorageMode.CPU,
        "disk": StorageMode.DISK,
        "none": StorageMode.NONE,
    }
    if not isinstance(value, str) or value not in modes:
        raise DeepwaveValueError(storage_mode_error_message(value))
    return modes[value]


#: ``None`` means no explicit policy; a Module forward inherits its constructor
#: policy, while a functional call uses :class:`Storage` defaults.
STORAGE_DEFAULT = None


def format_bytes(n: Union[int, float]) -> str:
    """Renders a byte count in the largest unit that keeps it readable.

    Args:
        n: A number of bytes.

    Returns:
        A short string such as ``"8.0 GB"`` or ``"512.0 MB"``. Powers of 1000
        (not 1024), matching how disks and ``shutil.disk_usage`` are labelled.

    """
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1000.0
    raise AssertionError  # pragma: no cover - the loop always returns


@dataclasses.dataclass(frozen=True)
class Storage:
    """Where and how the wavefield history for the model gradient is kept.

    One object carrying the whole storage policy, accepted as ``storage=`` by
    every propagator (functional and :class:`~torch.nn.Module`, on both the
    constructor and ``forward``), by
    :func:`deepwave.checkpointing.run_segments`, by
    :func:`deepwave.estimate_memory`. It is an
    alternative spelling, never a requirement: the individual
    ``storage_mode``/``storage_path``/``storage_compression`` arguments are
    permanent and are not deprecated. Passing both spellings in one call
    raises, because there would be no honest way to merge them.

    It is plain data -- it carries no device, no time-step count and no
    shapes, so one object can be reused across a sweep. Derive a variant with
    ``dataclasses.replace(storage, mode="disk")``.

    Stability: provisional (see :doc:`/stability`) -- fields may be added as
    more of a storage policy becomes worth recording. The fields it has are
    not renamed or withdrawn without a release's notice under
    :class:`deepwave.DeepwaveFutureWarning`, and the individual
    ``storage_mode`` / ``storage_path`` /
    ``storage_compression`` arguments it is an alternative spelling for
    are stable, as is passing a plain mode string to ``storage=``.

    Args:
        mode: One of ``"device"`` (keep the history on the compute device,
            the default and the fastest), ``"cpu"`` (offload it to host
            memory), ``"disk"`` (write it to files under ``path``), or
            ``"none"`` (keep nothing, which makes the call forward-only with
            respect to the model gradient). On a CPU device ``"cpu"`` and
            ``"device"`` are the same thing and the engine resolves the former
            to the latter; :attr:`deepwave.RunInfo.storage` records which
            was used.
        path: The directory ``mode="disk"`` writes into. ``None`` (the
            default) means the system temporary directory, resolved **when you
            call**, so ``TMPDIR`` set after ``import deepwave`` is honoured.
            Accepts ``str`` or any ``os.PathLike`` (a ``pathlib.Path``).
            Ignored, with a warning, by every other mode. Each run makes its
            own ``deepwave_tmp_<pid>_<random hex>`` subdirectory here when it
            allocates its wavefield history, and removes the whole
            subdirectory once nothing refers to that history any more (after
            ``backward()``, or when the output and its graph go out of
            scope). Only a process killed outright leaves one behind, since
            no destructor runs then.
        lossy_compression: If True, store the history as 8-bit block max-abs
            quantisation. **This is lossy and it changes the model gradient**
            -- see :func:`deepwave.scalar`'s ``storage_compression``
            entry for the measured error and what is unaffected. Defaults to
            False.

    """

    mode: "_StorageModeStr" = "device"
    path: Optional["StoragePath"] = None
    lossy_compression: bool = False

    def __post_init__(self) -> None:
        """Validates the policy fields, raising or warning as each deserves."""
        if self.mode not in STORAGE_MODES:
            raise DeepwaveValueError(storage_mode_error_message(self.mode))
        # A path that no mode will read, and compression that no mode will
        # apply, are both harmless but always mistakes worth naming: the user
        # believes they configured something. Warn rather than raise, because
        # sweeping one Storage across several modes is a legitimate pattern
        # and stripping the field at each step would be worse than the noise.
        from deepwave._common import _warn_once_per_cause

        if self.path is not None and self.mode != "disk":
            _warn_once_per_cause(
                "storage_path_ignored",
                f"storage path {os.fspath(self.path)!r} is ignored when "
                f"mode={self.mode!r}; it is used only by mode='disk'.",
                UserWarning,
                stacklevel=4,
            )
        if self.lossy_compression and self.mode == "none":
            _warn_once_per_cause(
                "storage_compression_ignored",
                "lossy_compression=True is ignored when mode='none', which "
                "stores no wavefield history to compress.",
                UserWarning,
                stacklevel=4,
            )

    def __repr__(self) -> str:
        """One short line that never invents a path it was not given."""
        path = (
            "None (resolved at call time)"
            if self.path is None
            else repr(os.fspath(self.path))
        )
        return (
            f"Storage(mode={self.mode!r}, path={path}, "
            f"lossy_compression={self.lossy_compression})"
        )

    __str__ = __repr__


Storage.__module__ = "deepwave"


def _describe(name: str, value: object) -> str:
    """Renders one ``name=value`` pair for a conflict message."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return f"{name}={value!r}"


def resolve_storage(
    storage: Optional[Union[str, Storage]] = STORAGE_DEFAULT,
    storage_mode: Optional["_StorageModeStr"] = None,
    storage_path: Optional["StoragePath"] = None,
    storage_compression: Optional[bool] = None,
) -> Optional[Storage]:
    """Collapses the four spellings of the storage policy into one answer.

    This is the single funnel. Every entry point that accepts storage
    arguments -- the nine functionals, the nine Modules' ``__init__`` and
    ``forward``, :func:`deepwave.checkpointing.run_segments`,
    :func:`deepwave.estimate_memory` and :func:`deepwave.plan` -- calls it, so
    the conflict rules and the alias mapping cannot differ between them.

    Args:
        storage: The combined spelling: a mode string or a :class:`Storage`.
            ``None`` means the caller did not provide a policy.
        storage_mode: The individual mode, or None if not passed.
        storage_path: The individual path, or None if not passed.
        storage_compression: The individual compression flag, or None if
            not passed.

    Returns:
        A :class:`Storage` carrying the resolved policy, or ``None`` when the
        caller passed none of the four -- which lets a ``Module.forward`` tell
        "no override" from "override with the defaults" and fall back to the
        value its constructor was given.

    Raises:
        DeepwaveValueError: If ``storage=`` is combined with any individual
            argument.

    """
    individual = [
        ("storage_mode", storage_mode),
        ("storage_path", storage_path),
        ("storage_compression", storage_compression),
    ]
    given = [(name, value) for name, value in individual if value is not None]
    storage_given = storage is not None

    if storage_given and given:
        combined = ", ".join(_describe(name, value) for name, value in given)
        raise DeepwaveValueError(
            "Pass either storage= or the individual storage_mode/"
            "storage_path/storage_compression arguments, not both (got "
            f"{_describe('storage', storage)} and {combined}). The individual "
            "arguments are permanent aliases for the fields of Storage: "
            "storage=deepwave.Storage(mode='disk', path='/scratch') is the "
            "combined form."
        )

    if storage_given:
        if isinstance(storage, Storage):
            return storage
        if isinstance(storage, str):
            # Validation (and the shared message) happens in __post_init__,
            # which is why the cast is safe: a value outside the Literal
            # raises there rather than reaching a caller.
            return Storage(mode=cast("_StorageModeStr", storage))
        raise DeepwaveValueError(
            "storage must be a mode string ('device', 'cpu', 'disk' or "
            f"'none') or a deepwave.Storage, but got {storage!r} of type "
            f"{type(storage).__name__}."
        )

    if not given:
        return None

    return Storage(
        mode="device" if storage_mode is None else storage_mode,
        path=storage_path,
        lossy_compression=bool(storage_compression),
    )


@dataclasses.dataclass(frozen=True)
class ResolvedStorage:
    """A :class:`Storage` with everything a run needs actually filled in.

    The difference from :class:`Storage` is that ``path`` is a real directory
    string rather than ``None``, and ``used_default_path`` records whether the
    user chose it -- which is what the tmpfs warning keys on.
    """

    mode: "_StorageModeStr"
    path: str
    lossy_compression: bool
    used_default_path: bool

    def as_storage(self) -> Storage:
        """The public view recorded in :attr:`deepwave.RunInfo.storage`.

        ``path`` is reported only for ``mode="disk"``, the one mode that reads
        it; for the others it stays ``None`` rather than advertising a
        temporary directory the run never touched.
        """
        return Storage(
            mode=self.mode,
            path=self.path if self.mode == "disk" else None,
            lossy_compression=self.lossy_compression,
        )


def downgraded_storage_mode(
    mode: "_StorageModeStr", device: object
) -> "_StorageModeStr":
    """Applies the one device-dependent adjustment to a storage mode.

    On a CPU device ``"cpu"`` and ``"device"`` name the same memory, so the
    former is transparently downgraded to the latter. Single-sourced here
    because two places need the answer and they must not disagree: the engine,
    which acts on it, and :attr:`deepwave.RunInfo.storage`, which reports it.

    Args:
        mode: The requested mode.
        device: The device the propagation will run on.

    Returns:
        The mode that will actually be used.

    """
    if str(device) == "cpu" and mode == "cpu":
        return "device"
    return mode


def resolved_storage_for_device(storage: ResolvedStorage, device: object) -> Storage:
    """The public view of a resolved policy, as the run will really use it.

    Args:
        storage: The resolved policy.
        device: The device the propagation will run on.

    Returns:
        The :class:`Storage` recorded in :attr:`deepwave.RunInfo.storage`.

    """
    return dataclasses.replace(
        storage, mode=downgraded_storage_mode(storage.mode, device)
    ).as_storage()


def default_storage_directory() -> str:
    """The directory ``storage_path=None`` means, resolved now.

    Not simply ``tempfile.gettempdir()``: that function computes its answer
    once and **caches it** in the module-global ``tempfile.tempdir``, so the
    first call anywhere in the process -- very often made by an unrelated
    library long before Deepwave is asked to store anything -- freezes the
    value. ``TMPDIR`` set after that point would be silently ignored, which
    is precisely the defect that moving this resolution out of the
    default-argument expressions was meant to fix. Consulting the environment
    first, in the same order ``tempfile`` itself does, makes the answer
    genuinely current, and ``gettempdir`` remains the fallback so that the
    platform-specific search (and its writability check) still applies when
    no variable is set.

    Returns:
        The directory, as a string.

    """
    for variable in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(variable)
        if value:
            return value
    return tempfile.gettempdir()


def materialise_storage(policy: Optional[Storage]) -> ResolvedStorage:
    """Fills in the parts of a storage policy that depend on the call.

    Resolving the path here, rather than in a default-argument expression, is
    the whole point of resolving it at call time: ``tempfile.gettempdir()`` is read
    rather than when Deepwave was imported, so ``TMPDIR`` set at any point
    before the call is honoured, the rendered signature does not hard-code one
    machine's temporary directory, and ``pathlib.Path`` is accepted.

    Args:
        policy: The resolved policy, or ``None`` for "nothing was passed",
            which means the defaults.

    Returns:
        The policy with ``path`` materialised.

    """
    if policy is None:
        policy = Storage()
    used_default_path = policy.path is None
    path = (
        default_storage_directory() if policy.path is None else os.fspath(policy.path)
    )
    return ResolvedStorage(
        mode=policy.mode,
        path=path,
        lossy_compression=policy.lossy_compression,
        used_default_path=used_default_path,
    )


def _free_bytes(path: str) -> Optional[int]:
    """The free space at ``path``, or None if the question cannot be asked."""
    try:
        return int(shutil.disk_usage(path).free)
    except OSError:
        return None


def warn_if_storage_path_too_small(
    path: str, history_bytes: Optional[int], stacklevel: int = 3
) -> None:
    """Warns when the chosen disk-storage directory looks too small.

    A warning rather than an error on purpose: ``history_bytes`` is a
    documented lower bound and free space can change between this check and
    the writes, so refusing here would sometimes refuse a run that would have
    succeeded.

    Args:
        path: The directory that will be written to.
        history_bytes: The estimated bytes to be written, or None to skip.
        stacklevel: Passed through so the warning points at the user's call.

    """
    if history_bytes is None:
        return
    free = _free_bytes(path)
    if free is None or free >= history_bytes:
        return
    from deepwave._common import _warn_once_per_cause

    _warn_once_per_cause(
        "storage_path_too_small",
        f"storage_path={path!r} has {format_bytes(free)} free but this run's "
        f"wavefield history is about {format_bytes(history_bytes)}. The run "
        "will probably fail partway through with a disk write error. Pass "
        "storage_path= a larger filesystem, or use storage_mode='cpu'.",
        UserWarning,
        stacklevel=stacklevel,
    )


def warn_if_default_disk_path(
    storage: ResolvedStorage,
    history_bytes: Optional[int],
    stacklevel: int,
) -> None:
    """Warns that ``mode="disk"`` fell back to the system temporary directory.

    On many systems that directory is a RAM-backed ``tmpfs`` far smaller than
    the disk, so the mode chosen to *avoid* exhausting memory can exhaust it
    instead. Emitted once per process, and only when the user did not choose
    the directory themselves.

    Args:
        storage: The resolved storage policy.
        history_bytes: The estimated size of the wavefield history, or None if
            it could not be computed (the clause is then omitted).
        stacklevel: Passed through so the warning points at the user's call.

    """
    if storage.mode != "disk" or not storage.used_default_path:
        return
    from deepwave._common import _warn_once_per_cause

    size_clause = (
        ""
        if history_bytes is None
        else (f" This run's wavefield history is about {format_bytes(history_bytes)}.")
    )
    free = _free_bytes(storage.path)
    free_clause = "" if free is None else f" (this one has {format_bytes(free)} free)"
    _warn_once_per_cause(
        "storage_default_disk_path",
        f'storage_mode="disk" is writing to the default directory '
        f"{storage.path}, which is where tempfile.gettempdir() points on "
        f"this machine.{size_clause} On many systems that directory is a "
        f"RAM-backed tmpfs{free_clause}, so a large run can exhaust memory "
        'rather than disk. Pass storage_path="/path/on/a/real/disk" (or '
        'storage=deepwave.Storage(mode="disk", path=...)) to choose the '
        "directory, or set TMPDIR before the call.",
        UserWarning,
        stacklevel=stacklevel,
    )


class TemporaryStorage:
    """Manages temporary files for disk storage.

    Creates a unique subdirectory for each instantiation to prevent collisions.
    """

    def __init__(
        self,
        base_path: str,
        num_files: int,
        history_bytes: Optional[int] = None,
    ) -> None:
        """Initialise the temporary storage.

        Args:
            base_path: The base path for the temporary directory.
            num_files: The number of files to create.
            history_bytes: An estimate of the bytes this run will write here,
                used only to make the failure messages concrete. ``None``
                omits the size clauses.
        """
        self.base_dir = Path(base_path) / f"deepwave_tmp_{os.getpid()}_{uuid4().hex}"
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Without this the user sees a bare PermissionError/FileNotFound
            # naming a generated subdirectory they never chose, with nothing
            # to connect it to the argument that chose its parent.
            needs = (
                ""
                if history_bytes is None
                else f" This run needs about {format_bytes(history_bytes)} there."
            )
            raise DeepwaveOSError(
                f"Could not create the disk-storage directory under "
                f"storage_path={base_path!r} (mode='disk'): {e.strerror or e}."
                f"{needs} Pass storage_path= (or "
                'storage=deepwave.Storage(mode="disk", path=...)) to a '
                "writable directory with enough free space, or use "
                'storage_mode="cpu" to keep the history in host memory '
                "instead."
            ) from e
        warn_if_storage_path_too_small(str(self.base_dir), history_bytes)
        self.filenames: list[bytes] = []

        for i in range(num_files):
            self.filenames.append(str(self.base_dir / f"shot_{i}.bin").encode("utf-8"))

        self.filenames_ptr = (ctypes.c_char_p * len(self.filenames))(*self.filenames)

    def get_filenames_ptr(self) -> ctypes.Array[ctypes.c_char_p]:
        """Return a pointer to the array of filenames."""
        return self.filenames_ptr

    def close(self) -> None:
        """Close the storage and remove the temporary directory."""
        if self.base_dir.exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(self.base_dir)

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()


class IntermediateStorage:
    """Stores pointers to wavefield storage for a single allocation."""

    def __init__(
        self,
        store_1a: torch.Tensor,
        store_1b: torch.Tensor,
        store_2: torch.Tensor,
        store_3: torch.Tensor,
        temporary_storage: Optional[TemporaryStorage],
    ) -> None:
        """Store the intermediate storage information."""
        self.store_1a = store_1a
        self.store_1b = store_1b
        self.store_2 = store_2
        self.store_3 = store_3
        self.temporary_storage = temporary_storage

    def get_ptrs(
        self,
    ) -> tuple[int, int, int, int, ctypes.Array[ctypes.c_char_p]]:
        """Return pointers to the storage and filenames array."""
        return (
            self.store_1a.data_ptr(),
            self.store_1b.data_ptr(),
            self.store_2.data_ptr(),
            self.store_3.data_ptr(),
            self.temporary_storage.get_filenames_ptr()
            if self.temporary_storage is not None
            else (ctypes.c_char_p * 0)(),
        )


class StorageManager:
    """Manages temporary storage for wave propagation."""

    def __init__(
        self,
        shot_shape: tuple[int, ...],
        dtype: torch.dtype,
        n_shots: int,
        nt: int,
        step_ratio: int,
        storage_mode: "StorageMode",
        storage_compression: bool,
        storage_path: str,
        device: torch.device,
        is_cuda: bool,
    ) -> None:
        """Initialise the StorageManager.

        Args:
            shot_shape: The spatial shape of a single shot's wavefield.
            dtype: The data type of the stored wavefields.
            n_shots: The number of shots being simulated.
            nt: The total number of time steps.
            step_ratio: The ratio between the user time step and the internal time step.
            storage_mode: The active :class:`StorageMode` (device, CPU, disk, or
                none).
            storage_compression: Whether to apply compression when storing to disk.
            storage_path: The path to the directory for disk storage.
            device: The PyTorch device for computation.
            is_cuda: Whether the device is a CUDA device.
        """
        self.shot_shape = shot_shape
        self.dtype = dtype
        self.n_shots = n_shots
        self.nt = nt
        self.step_ratio = step_ratio
        self.storage_mode = storage_mode
        self.storage_compression = storage_compression
        self.storage_path = storage_path
        self.device = device
        self.is_cuda = is_cuda

        self.dtype_size = 4 if dtype == torch.float32 else 8
        # `shot_shape` is a plain tuple of Python ints (a spatial shape), not
        # a tensor of model *values* -- there is nothing data-dependent here.
        # The previous `int(torch.prod(torch.tensor(shot_shape)).item())`
        # nonetheless round-tripped through a Tensor, and `Tensor.item()` is
        # opaque to Dynamo: it forced two graph breaks (this constructor
        # runs twice per `scalar()`/`elastic()` call, once for each
        # StorageManager). `math.prod` computes the same product as plain
        # Python ints, so this is unconditionally equivalent (eager and
        # compiled) and removes the tensor round-trip entirely.
        self.num_elements_per_shot = math.prod(shot_shape)
        self.shot_bytes_uncomp = self.num_elements_per_shot * self.dtype_size
        self.shot_bytes_comp = 0
        if storage_compression and storage_mode != StorageMode.NONE:
            n_blocks = 1
            for dim in self.shot_shape:
                n_blocks *= (dim + 7) // 8
            self.shot_bytes_comp = (
                (
                    self.num_elements_per_shot
                    + n_blocks * self.dtype_size
                    + self.dtype_size
                    - 1
                )
                // self.dtype_size
            ) * self.dtype_size

        self.storages: list[IntermediateStorage] = []
        self.storage_ptrs: list[Union[int, ctypes.Array[ctypes.c_char_p]]] = []
        # Native snapshot loading reuses staging buffers. A retained graph may
        # receive overlapping VJPs, so only one backward may use them at once.
        self.backward_lock = threading.RLock()

    def allocate(self, requires_grad: bool) -> None:
        """Allocates storage tensors and handles disk storage if needed."""
        store_1a = torch.empty(0)
        store_1b = torch.empty(0)
        store_2 = torch.empty(0)
        store_3 = torch.empty(0)
        temporary_storage: Optional[TemporaryStorage] = None

        if requires_grad and self.storage_mode != StorageMode.NONE:
            num_steps_stored = self.nt // self.step_ratio

            if self.storage_mode == StorageMode.DEVICE:  # Device
                if self.storage_compression:
                    store_1a = torch.zeros(
                        self.n_shots,
                        *self.shot_shape,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    store_1b = torch.zeros(
                        self.n_shots,
                        *self.shot_shape,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    store_2 = torch.zeros(
                        num_steps_stored,
                        self.n_shots,
                        self.shot_bytes_comp // self.dtype_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    store_1a = torch.zeros(
                        num_steps_stored,
                        self.n_shots,
                        *self.shot_shape,
                        device=self.device,
                        dtype=self.dtype,
                    )

            elif self.storage_mode == StorageMode.CPU:  # CPU
                store_1a = torch.zeros(
                    self.n_shots,
                    *self.shot_shape,
                    device=self.device,
                    dtype=self.dtype,
                )
                store_1b = torch.zeros(
                    self.n_shots,
                    *self.shot_shape,
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.storage_compression:
                    store_2 = torch.zeros(
                        self.n_shots,
                        self.shot_bytes_comp // self.dtype_size,
                        device=self.device,
                        dtype=self.dtype,
                    )

                size_3 = (
                    self.shot_bytes_comp
                    if self.storage_compression
                    else self.shot_bytes_uncomp
                ) // self.dtype_size
                store_3 = torch.zeros(
                    num_steps_stored,
                    self.n_shots,
                    size_3,
                    device="cpu",
                    pin_memory=True,
                    dtype=self.dtype,
                )

            elif self.storage_mode == StorageMode.DISK:  # Disk
                temporary_storage = TemporaryStorage(
                    self.storage_path, 1 if self.is_cuda else self.n_shots
                )

                store_1a = torch.zeros(
                    self.n_shots,
                    *self.shot_shape,
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.storage_compression:
                    store_2 = torch.zeros(
                        self.n_shots,
                        self.shot_bytes_comp // self.dtype_size,
                        device=self.device,
                        dtype=self.dtype,
                    )

                if self.is_cuda:
                    size_3 = (
                        self.shot_bytes_comp
                        if self.storage_compression
                        else self.shot_bytes_uncomp
                    ) // self.dtype_size
                    store_3 = torch.zeros(
                        self.n_shots,
                        size_3,
                        device="cpu",
                        pin_memory=True,
                        dtype=self.dtype,
                    )

        storage = IntermediateStorage(
            store_1a, store_1b, store_2, store_3, temporary_storage
        )
        self.storages.append(storage)
        self.storage_ptrs.extend(storage.get_ptrs())


def resolve_storage_arguments(
    storage: Optional[Union[str, Storage]],
    storage_mode: Optional["_StorageModeStr"],
    storage_path: Optional["StoragePath"],
    storage_compression: Optional[bool],
) -> ResolvedStorage:
    """The one call every entry point makes to settle its storage policy.

    Composes :func:`resolve_storage` (which spelling did the caller use, and
    do the spellings conflict?) with :func:`materialise_storage` (what does
    that mean for *this* call, right now?).

    It deliberately emits no warning about the resolved directory. That
    warning names the size of the history the run is about to write, which is
    not known until the timing and the padded shape are resolved, so it is
    emitted from the shared setup instead -- which also keeps a call that only
    *estimates* a run from warning about a directory nothing will be written
    to.

    Args:
        storage: The combined spelling, or ``None`` when unspecified.
        storage_mode: The individual mode, or None.
        storage_path: The individual path, or None.
        storage_compression: The individual compression flag, or None.

    Returns:
        The fully resolved policy, with ``path`` a real directory.

    """
    return materialise_storage(
        resolve_storage(
            storage,
            storage_mode,
            storage_path,
            storage_compression,
        )
    )

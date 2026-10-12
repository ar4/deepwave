"""Machine-readable description of what each propagator offers.

Deepwave has four reflection APIs that used to work by matching strings:
:func:`deepwave.wavefields.record_boundary`,
:class:`deepwave.wavefields.RunningDFT`, :func:`deepwave.estimate_memory` and
:func:`deepwave.checkpointing.run_segments` all had to know, or guess, how a
propagator spells its receiver keyword or which of its outputs is a receiver.
Guessing from a positional slot is how a Born propagator's "first receiver
output" silently means the *background* data. This module publishes the
vocabulary those APIs -- and any user-written generic wrapper -- should
reflect over instead:

* :func:`deepwave.spec` returns a frozen :class:`PropagatorSpec` describing
  one propagator: its channels, its supported options, and the names of
  everything it returns;
* :func:`deepwave.wavefield_names` and :func:`deepwave.receiver_names` are
  the two questions that are asked often enough to deserve their own
  functions.

All three accept a propagator in any of the three spellings that name one:
the functional (``deepwave.elastic``), the Module class
(``deepwave.Elastic``) or the plain name (``"elastic"``).

The class published here is a *projection*. Deepwave's internal
``PropagatorSpec`` additionally carries a dozen fields that describe the
compiled kernels rather than the interface -- which models own a gradient
slot, which PML memory variables are slab-localised, what the pure-Python
backends accept -- and those are implementation detail that moves between
releases. Only the interface half is projected here, in one function, so the
public class cannot drift from the private one and a new private field is
invisible until someone deliberately publishes it.

Stability: provisional (see :doc:`/stability`).
"""

import dataclasses
import functools
import inspect
from collections import abc
from types import MappingProxyType
from typing import Any, Callable, Union

from deepwave import _engine
from deepwave._exceptions import DeepwaveTypeError as _DeepwaveTypeError
from deepwave._exceptions import DeepwaveValueError as _DeepwaveValueError

__all__ = ["PropagatorSpec", "receiver_names", "spec", "wavefield_names"]


#: A propagator, in any of the three spellings that name one: the functional,
#: the ``torch.nn.Module`` subclass, or the plain name.
_PropagatorLike = Union[Callable[..., Any], type, str]


def _resolve_module_name(propagator: _PropagatorLike) -> str:
    """The propagator name that ``propagator`` refers to."""
    if isinstance(propagator, str):
        name = propagator
        if name not in _engine.PROPAGATOR_NAMES:
            raise _DeepwaveValueError(
                f"{name!r} is not the name of a Deepwave propagator. The "
                f"propagators are: {', '.join(_engine.PROPAGATOR_NAMES)}."
            )
        return name
    module = getattr(propagator, "__module__", None)
    if isinstance(module, str) and module.startswith("deepwave."):
        candidate = module[len("deepwave.") :]
        if candidate in _engine.PROPAGATOR_NAMES:
            return candidate
    raise _DeepwaveTypeError(
        f"{propagator!r} does not look like a Deepwave propagator. Pass the "
        "functional (deepwave.elastic), the Module class (deepwave.Elastic) "
        "or the name ('elastic'). The propagators are: "
        f"{', '.join(_engine.PROPAGATOR_NAMES)}."
    )


def _resolve_spec(propagator: _PropagatorLike) -> _engine.PropagatorSpec:
    """The *private* spec of ``propagator``.

    Deepwave-internal: the single resolver shared by
    :mod:`deepwave.checkpointing`, :mod:`deepwave.wavefields`,
    ``deepwave/_capabilities.py`` and :func:`spec`, so that all of them accept
    the same three spellings and produce the same error for anything else.
    """
    return _engine.propagator_spec(_resolve_module_name(propagator))


def _call_names(
    fn: Callable[..., list[str]], ndim: int, options: abc.Mapping[str, Any]
) -> list[str]:
    """Calls one of a spec's name functions, forwarding ``n_mechanisms``.

    The viscous propagators' name functions take an ``n_mechanisms`` second
    argument; every other propagator's takes only ``ndim``. The spec declares
    them all as functions of ``ndim`` alone, so the extra argument is passed
    only where the callable actually accepts it.
    """
    n_mechanisms = options.get("q_n_mechanisms")
    if n_mechanisms is None:
        return fn(ndim)
    parameters = inspect.signature(fn).parameters
    if "n_mechanisms" not in parameters:
        return fn(ndim)
    return fn(ndim, n_mechanisms=n_mechanisms)


def _accepted_options(private: _engine.PropagatorSpec) -> tuple[str, ...]:
    """The ``**options`` keys that change this propagator's name lists."""
    if "q_n_mechanisms" in private.supported_options:
        return ("q_n_mechanisms",)
    return ()


def _check_options(
    private: _engine.PropagatorSpec, options: abc.Mapping[str, Any]
) -> None:
    """Raises on an option key that cannot change this propagator's names."""
    accepted = _accepted_options(private)
    unknown = [key for key in options if key not in accepted]
    if not unknown:
        return
    spelled = ", ".join(repr(key) for key in sorted(unknown))
    allowed = (
        ", ".join(repr(key) for key in accepted)
        if accepted
        else "none (this propagator's names depend only on ndim)"
    )
    raise _DeepwaveValueError(
        f"{private.name}: unknown option {spelled}. The options that change "
        f"this propagator's name lists are: {allowed}."
    )


def _check_ndim(private: _engine.PropagatorSpec, ndim: int) -> None:
    """Raises if ``ndim`` is not one this propagator supports."""
    if ndim in private.ndims:
        return
    supported = ", ".join(str(n) for n in private.ndims)
    raise _DeepwaveValueError(
        f"{private.name} does not support ndim={ndim}; it supports {supported}."
    )


@dataclasses.dataclass(frozen=True)
class PropagatorSpec:
    """What one propagator offers, as data.

    Returned by :func:`deepwave.spec`; not constructed directly. Every field
    is plain data and every method is a pure function of the number of
    spatial dimensions, so instances compare equal, ``repr`` usefully, and
    pickle.

    Stability: provisional (see :doc:`/stability`). Fields and methods may be
    added as more of a propagator's interface becomes worth publishing, and
    what a field reports changes when the thing it describes changes. The
    fields and methods it has are not renamed or withdrawn without a
    release's notice under :class:`deepwave.DeepwaveFutureWarning`.

    Attributes:
        name: The propagator's name, e.g. ``"elastic"``.
        status: Its stability tier, one of ``"stable"``,
            ``"provisional"``, ``"experimental"``, ``"deprecated"``.
        ndims: The numbers of spatial dimensions it supports.
        supported_options: The shared options it implements
            (``"mpml_fraction"``, ``"surface_method"``, ``"q_freq"``,
            ``"q_n_mechanisms"``, ``"q_band"``). Every propagator *accepts*
            every shared option at its default value; a non-default value of
            one that is not listed here raises.
        synthetic_source_channels: Source channels the propagator offers that
            have no backend slot of their own because they are expanded into
            other channels. ``{"pressure_rate"}`` on ``acoustic_vti``, empty
            everywhere else.
        synthetic_receiver_channels: Receiver channels the propagator offers
            that have no backend slot of their own because it computes them
            from other channels' recordings. ``{"p"}`` on ``acoustic_vti``,
            whose pressure is ``-(sigma_v + (ndim - 1) sigma_h) / ndim``,
            empty everywhere else. Such a channel still occupies its own
            output tuple slot.
        has_bg_receivers: Whether the propagator takes a ``bg_receivers=``
            mapping (equivalently, whether it is a Born propagator).
        has_forward_wavefield: Whether the propagator implements
            ``backward_callback_needs_forward_wavefield=True``, and so can
            serve :meth:`deepwave.CallbackState.get_wavefield`'s
            ``role="forward"`` inside a ``backward_callback``. ``True`` on
            ``scalar`` alone today. All nine *accept* the argument at its
            default of ``False``; passing ``True`` where this is ``False``
            raises :exc:`deepwave.DeepwaveNotImplementedError`.

    """

    name: str
    status: str
    ndims: tuple[int, ...]
    supported_options: frozenset[str]
    synthetic_source_channels: frozenset[str]
    synthetic_receiver_channels: frozenset[str]
    has_bg_receivers: bool
    has_forward_wavefield: bool

    @property
    def _private(self) -> _engine.PropagatorSpec:
        """The internal spec this is a projection of.

        Looked up by name rather than stored as a field, so that every field
        of this class is plain data. The internal spec holds the
        propagator's name functions, several of which are ``lambda``s, and a
        stored reference to it would make this class unpicklable and give it
        an unreadable ``repr``. Resolution is a :data:`sys.modules` lookup.
        """
        return _engine.propagator_spec(self.name)

    def model_names(self, ndim: int) -> tuple[str, ...]:
        """The propagator's model tensors, in the order it takes them.

        The *user-facing* models -- the arguments the propagator's public
        function takes, in signature order -- not the internal prepared
        models the kernel consumes: ``acoustic`` reports ``("v", "rho")``,
        the two tensors a caller passes, rather than the bulk modulus and
        per-component buoyancies built from them. A Born propagator's
        perturbation models are included (``("v", "rho", "dv", "drho")`` for
        ``acoustic_born``), as is any optional model tensor the propagator
        accepts at this ``ndim`` (``elastic_vti``'s ``c66``, which is a model
        in 3-D only).

        These are the names :func:`deepwave.estimate_memory`'s
        ``models_requiring_grad`` takes, and the names a generic wrapper
        needs in order to assemble a call.

        Args:
            ndim: The number of spatial dimensions.

        Returns:
            The model argument names.

        Raises:
            deepwave.DeepwaveValueError: On an unsupported ``ndim``.

        """
        from deepwave import _common

        _check_ndim(self._private, ndim)
        return _common._propagator_user_model_names(self.name, ndim)  # noqa: SLF001

    def source_channels(self, ndim: int) -> tuple[str, ...]:
        """The source channels, in canonical order."""
        _check_ndim(self._private, ndim)
        return tuple(self._private.source_channels(ndim))

    def receiver_channels(self, ndim: int) -> tuple[str, ...]:
        """The receiver channels, in the order of the returned tuple."""
        _check_ndim(self._private, ndim)
        return tuple(self._private.receiver_channels(ndim))

    def bg_receiver_channels(self, ndim: int) -> tuple[str, ...]:
        """The background-receiver channels; empty on a non-Born propagator."""
        _check_ndim(self._private, ndim)
        return tuple(self._private.bg_receiver_channels(ndim))

    def channel_aliases(self) -> abc.Mapping[str, str]:
        """The permanent alternative spellings of this propagator's channels.

        Maps each accepted alias to the channel it names, on both the source
        and the receiver side. Empty on all nine propagators: every channel
        has exactly one spelling. Read-only; any entry would be permanent. A
        spelling that is going away is in
        :meth:`deprecated_source_channels` instead.
        """
        return MappingProxyType(dict(self._private.channel_aliases))

    def deprecated_source_channels(self) -> abc.Mapping[str, str]:
        """The deprecated *source* channel spellings, and what replaces each.

        ``{"p": "volume_rate"}`` on the acoustic family and
        ``{"p": "pressure_rate"}`` on the elastic family and
        ``acoustic_vti``; empty on the two scalar propagators. The old
        spelling keeps working, warning
        :class:`deepwave.DeepwaveFutureWarning`, until it is removed. It is a
        source-side deprecation only: ``"p"`` is the canonical *receiver*
        channel name on every propagator that records a pressure.
        """
        return MappingProxyType(dict(self._private.deprecated_source_channels))

    def wavefield_names(self, ndim: int, **options: Any) -> tuple[str, ...]:
        """The state fields the propagator returns, in return order.

        Exactly the keys of :attr:`deepwave.PropagatorOutput.final_state`:
        the public wavefields plus ``"auxiliary_state"``, and no receiver
        fields. These are the names ``initial_state=`` consumes, so they are
        what a wrapper needs in order to continue a simulation.

        Args:
            ndim: The number of spatial dimensions.
            **options: Options that change the list.
                ``q_n_mechanisms`` on viscoacoustic is the only
                one today; any other key raises rather than being ignored.

        Returns:
            The state field names.

        Raises:
            deepwave.DeepwaveValueError: On an unsupported ``ndim`` or an
                option key this propagator's names do not depend on.

        """
        receivers = set(self.receiver_names(ndim))
        return tuple(
            name for name in self.output_names(ndim, **options) if name not in receivers
        )

    def output_names(self, ndim: int, **options: Any) -> tuple[str, ...]:
        """Every field of the returned tuple, in order.

        The state fields of :meth:`wavefield_names` followed by the receiver
        fields of :meth:`receiver_names`; the same tuple as
        :attr:`deepwave.PropagatorOutput.names`.
        """
        _check_ndim(self._private, ndim)
        _check_options(self._private, options)
        return tuple(_call_names(self._private.output_names, ndim, options))

    def receiver_names(self, ndim: int) -> tuple[str, ...]:
        """The receiver-amplitude output fields, in return order.

        On a Born propagator the background block comes first, then the
        scattered block, matching the returned tuple.
        """
        _check_ndim(self._private, ndim)
        receivers = _engine.receiver_output_names(self._private)
        return tuple(
            name for name in self._private.output_names(ndim) if name in receivers
        )

    def source_amplitudes_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's source amplitudes."""
        return self._kwarg("source", channel, 0)

    def source_locations_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's source locations."""
        return self._kwarg("source", channel, 1)

    def source_mask_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's source mask."""
        return self._kwarg("source", channel, 2)

    def receiver_locations_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's receiver locations.

        On a Born propagator this is the *scattered* receivers' keyword,
        spelled ``sc_receiver_locations*``; the background ones are
        :meth:`bg_receiver_locations_kwarg`.
        """
        kind = "sc_receiver" if self.has_bg_receivers else "receiver"
        return self._kwarg(kind, channel, 0)

    def receiver_mask_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's receiver mask."""
        kind = "sc_receiver" if self.has_bg_receivers else "receiver"
        return self._kwarg(kind, channel, 1)

    def bg_receiver_locations_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's background locations."""
        return self._kwarg("bg_receiver", channel, 0)

    def bg_receiver_mask_kwarg(self, channel: str) -> str:
        """The flat keyword carrying this channel's background mask."""
        return self._kwarg("bg_receiver", channel, 1)

    def receiver_output_name(self, channel: str) -> str:
        """The output field holding this channel's recorded data.

        The inverse of :meth:`receiver_channels`: the name under which
        ``channel``'s data appears in the returned tuple and in
        :attr:`deepwave.PropagatorOutput.receivers`.
        """
        canonical = self._canonical(channel, "receiver")
        prefix = (
            "sc_receiver_amplitudes"
            if self.has_bg_receivers
            else ("receiver_amplitudes")
        )
        if not self._private.channel_suffixes:
            return prefix
        return f"{prefix}_{canonical}"

    def bg_receiver_output_name(self, channel: str) -> str:
        """The output field holding this channel's background data."""
        if not self.has_bg_receivers:
            raise _DeepwaveValueError(
                f"{self.name} has no background receivers; only the Born "
                "propagators do."
            )
        canonical = self._canonical(channel, "bg_receiver")
        if not self._private.channel_suffixes:
            return "bg_receiver_amplitudes"
        return f"bg_receiver_amplitudes_{canonical}"

    def _canonical(self, channel: str, kind: str) -> str:
        """Resolves a spelling and checks that the channel exists.

        Resolution is silent even for a deprecated spelling: this is a query
        about the interface, not a call that uses it, so it reports what the
        propagator would do rather than warning about it.
        """
        canonical = self._private.canonical_channel(channel, kind)
        if canonical not in self._private.all_channels(kind):
            raise _DeepwaveValueError(
                f"{self.name}: unknown channel {channel!r}. Valid "
                f"{kind.replace('_', ' ')} channels are "
                f"{self._private.describe_channels(kind)}."
            )
        return canonical

    def _kwarg(self, kind: str, channel: str, index: int) -> str:
        """One flat keyword name of one channel of one kind."""
        canonical = self._canonical(channel, kind)
        return _engine.channel_kwargs(self._private, kind, canonical)[index]


@functools.cache
def _spec_by_name(name: str) -> PropagatorSpec:
    """The cached projection of one propagator's private spec."""
    private = _engine.propagator_spec(name)
    return PropagatorSpec(
        name=private.name,
        status=private.status,
        ndims=private.ndims,
        supported_options=frozenset(private.supported_options),
        synthetic_source_channels=frozenset(private.synthetic_source_channels),
        synthetic_receiver_channels=frozenset(private.synthetic_receiver_channels),
        has_bg_receivers=private.has_bg_receivers(),
        has_forward_wavefield=private.has_forward_wavefield,
    )


def spec(propagator: _PropagatorLike) -> PropagatorSpec:
    """What a propagator offers, as data.

    The machine-readable description a generic wrapper needs in order to
    drive any of the nine propagators without hard-coding one of them: the
    channels it has, the options it implements, and the names of everything
    it returns. See :class:`deepwave.PropagatorSpec`.

    Args:
        propagator: The functional (``deepwave.elastic``), the Module class
            (``deepwave.Elastic``) or the name (``"elastic"``).

    Returns:
        The propagator's frozen specification. Cached, so the same
        propagator always yields the same object.

    Raises:
        deepwave.DeepwaveTypeError: If the argument does not name a
            propagator.
        deepwave.DeepwaveValueError: If a string is not a propagator's name.

    Examples:
        >>> import deepwave
        >>> deepwave.spec(deepwave.elastic).source_channels(2)
        ('pressure_rate', 'y', 'x')
        >>> deepwave.spec("elastic").source_amplitudes_kwarg("pressure_rate")
        'source_amplitudes_pressure_rate'

    """
    return _spec_by_name(_resolve_module_name(propagator))


def wavefield_names(
    propagator: _PropagatorLike, ndim: int, **options: Any
) -> tuple[str, ...]:
    """The state fields a propagator returns, in return order.

    Exactly the keys of :attr:`deepwave.PropagatorOutput.final_state` -- the
    public wavefields plus ``"auxiliary_state"``, and no receiver fields --
    so this is the list an ``initial_state=`` consumer needs.

    Args:
        propagator: The functional, the Module class or the name.
        ndim: The number of spatial dimensions.
        **options: Options that change the list; ``q_n_mechanisms`` on the
            viscoacoustic propagator is the only one today, and any other key
            raises rather than being silently ignored.

    Returns:
        The state field names.

    Raises:
        deepwave.DeepwaveValueError: On an unsupported ``ndim``, or an option
            key this propagator's names do not depend on.

    Examples:
        >>> import deepwave
        >>> deepwave.wavefield_names(deepwave.scalar, 2)
        ('wavefield_0', 'wavefield_m1', 'auxiliary_state')

    """
    return spec(propagator).wavefield_names(ndim, **options)


def receiver_names(propagator: _PropagatorLike, ndim: int) -> tuple[str, ...]:
    """The receiver-amplitude fields a propagator returns, in return order.

    On a Born propagator the background block comes first and the scattered
    block second, matching the returned tuple.

    Args:
        propagator: The functional, the Module class or the name.
        ndim: The number of spatial dimensions.

    Returns:
        The receiver output field names.

    Raises:
        deepwave.DeepwaveValueError: If ``ndim`` is one the propagator does
            not support.

    Examples:
        >>> import deepwave
        >>> deepwave.receiver_names(deepwave.acoustic, 2)
        ('receiver_amplitudes_p', 'receiver_amplitudes_y', 'receiver_amplitudes_x')

    """
    return spec(propagator).receiver_names(ndim)

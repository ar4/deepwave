# Public API conventions: the propagator keyword-order contract

Deepwave's nine supported propagator functionals (`scalar`, `scalar_born`, `acoustic`,
`acoustic_born`, `viscoacoustic`, `elastic`, `elastic_born`, `elastic_vti`,
`acoustic_vti`) and their `nn.Module` wrappers
share one ordering convention; each signature exposes only relevant options. This page is the canonical description of that
convention; `tests/test_signature_consistency.py` enforces it automatically
and fails if any propagator's signature drifts from it.

## Positional block, then keyword-only

Every functional's signature is:

```
<propagator>(<required model tensors>, grid_spacing, dt, *, <keyword-only tail>)
```

`grid_spacing` and `dt` are always the last two positional parameters.
Everything else is keyword-only (enforced after `dt`).

## Canonical keyword-only order

Within the keyword-only tail, parameters appear in exactly this relative
order (a given propagator omits whichever rows it does not have):

```
<optional model tensors: dv, drho, dlamb, dmu, dbuoyancy, c66>
sources, receivers, bg_receivers    # the channel-keyed mapping form; the
                                    # canonical way to specify a survey.
                                    # bg_receivers is Born-only.
source_amplitudes[_<ch>] ...        # source channel order: the quantity
source_locations[_<ch>] ...         # name (volume_rate/pressure_rate), then
                                    # the released `p` alias where applicable,
                                    # then z, y, x; acoustic_vti puts sigma_v,
                                    # sigma_h after the quantity name
source_mask[_<ch>] ...              # one per locations argument
bg_receiver_locations[_<ch>] ...    # Born only, before the scattered receivers
bg_receiver_mask[_<ch>] ...
receiver_locations[_<ch>] ...       # Born: sc_receiver_locations[_<ch>]
receiver_mask[_<ch>] ...            # Born: sc_receiver_mask[_<ch>]
<physical initial-field kwargs>      # see "Physical initial-field kwargs" below
initial_state
accuracy, pml_width, boundary, pml_freq, mpml_fraction,
surface_method,                      # only on elastic
q_freq, q_n_mechanisms, q_band,      # only on viscoacoustic
max_vel, courant_number, survey_pad, origin, nt,
model_gradient_sampling_interval,
checkpoint_segments,                  # run the time axis as this many
                                     # checkpointed segments
freq_taper_frac, time_pad_frac, time_taper, check_output_finite,
forward_callback, backward_callback, callback_frequency,
backward_callback_needs_forward_wavefield,   # scalar only
backend,
python_backend,                      # released alias on scalar, scalar_born,
                                     # acoustic, and elastic only
storage, storage_mode, storage_path,
storage_compression,
location_interpolation, location_interpolation_halfwidth,
```

A keyword shared by two or more propagators has the same default value
everywhere it appears, with no exceptions.

## Channel order

Multi-channel groups (`source_amplitudes_*`, `source_locations_*`,
`bg_receiver_locations_*`, `receiver_locations_*`) lead with the scalar
channel, then `z, y, x`, then anything else in state-field order. The scalar
channel is `volume_rate` or `pressure_rate` on the source side (named for the
quantity injected) and `p` on the receiver side (named for the field
recorded); `acoustic_vti` appends `sigma_v, sigma_h` to both. On `acoustic` and `elastic`, the released deprecated `_p` source spelling
sits immediately after the quantity name it replaces. New propagators use
only the quantity name.

## Physical initial-field kwargs

A wavefield name is *physical* (and therefore exposed as a public kwarg) iff
it does not start with `psi`, `zeta`, or `m_` (those are absorbing-boundary
auxiliary fields, carried opaquely in `auxiliary_state`, not physical state).
Concretely: `scalar` exposes `wavefield_0`/`wavefield_m1`; `scalar_born` adds
`wavefield_sc_0`/`wavefield_sc_m1`; `acoustic`/`acoustic_born` expose
`pressure_0` plus the velocity `_0` fields (`acoustic_born` also exposes the
`_sc` variants); `elastic`/`elastic_born`/`elastic_vti` expose
the `v*_0` and `sigma*_0` fields (`elastic_born` also the `_sc` variants).
`viscoacoustic`'s relaxation-mechanism variables
(`r_mech0`, `r_mech1`, ...) are not public output fields: how
many of them there are follows a run's `q_n_mechanisms`, which is an
attenuation setting rather than a physical quantity the caller asked for,
so they are
carried opaquely in `auxiliary_state` alongside the absorbing boundary's
own memory. Viscoacoustic does not expose them as initial-value **keyword
arguments** either: there is no `r_0=` parameter. `initial_state` is the
way to supply a starting value for a mechanism variable, from mechanism 0
onward. Each propagator's authoritative list of physical field names is its
`<prop>_wavefield_names(ndim)` function, and the subset of it that is
folded away is `<prop>_auxiliary_state_names(ndim, n_mechanisms)`.

## `nn.Module` split

Each Module's `__init__` holds: the required and optional model tensors,
`requires_grad` (a bool or sequence of model names) and individual
`<model>_requires_grad` flags, the
`storage` policy and `storage_mode`/`storage_path`/`storage_compression`
arguments, and `grid_spacing`.
`forward` takes `dt` followed by everything else, keyword-only after `dt`,
mirroring the functional's keyword tail name-for-name and default-for-default
(minus the model parameters and constructor-only gradient flags). Storage
options can also be overridden per forward call; `None` inherits the
constructor policy. Omitted gradient flags resolve to false; `None` in their
signatures distinguishes omission from an explicit false when checking
conflicts with `requires_grad`.

## Standard deprecation shim

Deprecated aliases are accepted for backward compatibility and removed in the
release `DEPRECATION_REMOVAL_VERSION` names. A released alias remains visible
in the signature as a keyword parameter typed
`Optional[...]`/`object` and coerced at the top of the function with the one
shared emitter:

```python
from deepwave._exceptions import warn_deprecated_name

warn_deprecated_name("<old>", "<new>", stacklevel=2)
```

That is the *only* place `warnings.warn` is called with a deprecation
category, which is what makes every deprecation say the same three things:
the replacement, the removal release, and the one line that filters
`deepwave.DeepwaveFutureWarning`. A test enforces it, so a new deprecation
written with its own `warnings.warn` call fails rather than shipping a second
wording. Never use `DeprecationWarning`: Python's default filters hide it
outside `__main__`, which is where a library's deprecation is least likely to
be read.

If both the old and new names are supplied, raise
`ValueError("Pass only <new>, not both <new> and the deprecated <old>.")`.
Every deprecated name is a row of `tests/test_deprecation_removal_versions.py`'s
table, which asserts the category, the wording, the value mapping and -- for a
name that shipped in `v0.0.27` -- that it really did.

New deprecations require a reviewed migration path and a documented removal
release. Development-only names can be removed before their first release.

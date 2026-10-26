``PropagatorOutput``: ``DataParallel`` and pytrees
==================================================

:class:`deepwave.PropagatorOutput` is deliberately not usable as a full
``torch.nn.DataParallel`` return value on more than one GPU. The user-facing
consequence is documented on the class itself (return a tensor or a plain
tuple of tensors from a ``forward()`` that will be wrapped in
``DataParallel``, and see :doc:`/example_distributed`); this page records why
supporting it was rejected.

One fact decides both halves of this page: the field names belong to the
*instance*, not to the class. That is what makes ``DataParallel.gather``
impossible and what the pytree registration has to work around.

Why gathering cannot work
-------------------------

``DataParallel.gather`` reconstructs a module's return value via
``type(out)(map(gather_map, zip(*outputs)))`` -- the same convention it uses
for a plain ``namedtuple``. ``PropagatorOutput`` looks like a namedtuple from
the outside (``out.name``, ``_asdict()``) but cannot support this
reconstruction path: unlike a real namedtuple, whose field names are fixed by
its *class*, ``PropagatorOutput``'s names vary per *instance* (they depend on
``ndim`` and, for viscoacoustic, ``q_n_mechanisms``) and are
therefore not recoverable from ``type(out)`` alone -- there is no instance to
read ``_names`` from at reconstruction time.

Because ``gather`` never reaches the point of building an output, ``info``
is unreachable under it too, in the same way that it is unreachable under
``torch.utils.checkpoint(..., use_reentrant=True)``.

The rejected workaround
-----------------------

Making ``__new__`` accept a single positional iterable of values to paper
over this would only ever produce an unnamed tuple (all of the safety this
class exists to provide, silently gone) while also making
``PropagatorOutput(some_iterable)`` -- currently a loud, immediate error --
degrade into a silent, wrong construction whenever a caller passes ``names``
without ``values`` by mistake.

Neither trade-off is worth taking for a ``DataParallel``-only workaround, so
this is unsupported by design. ``__new__`` instead raises a
``DeepwaveTypeError`` naming ``DataParallel.gather`` as the likely cause and
pointing at the fix, so the failure is immediate and self-explaining rather
than a silently unnamed tuple.

The supported pattern is for a ``forward()`` wrapping a propagator to return
a single tensor (e.g. ``out.receiver_amplitudes``) or a small tuple of plain
tensors instead of a full ``PropagatorOutput`` when used with more than one
GPU under ``DataParallel``. ``DistributedDataParallel`` does not gather this
way and is unaffected.

Why the pytree context carries the names, not just ``info``
-----------------------------------------------------------

The same fact that stops ``DataParallel.gather`` working decides the shape of
the pytree registration. A pytree node splits an object into *children* (the
parts a transform maps over) and a *context* (everything needed to put it back
together). The children are the obvious choice -- the tuple elements, in return
order, with ``auxiliary_state`` expanding into the tensors it holds, so the
leaves of an output are exactly its tensors and the flattened order is the
documented positional return order.

The context has to hold the field names as well as ``info``. Rebuilding an
output means calling ``PropagatorOutput(names, values, info=info)``, and the
names are not recoverable from the class: they depend on ``ndim`` and, on the
viscous propagators, on the number of relaxation mechanisms. A registration
that carried only ``info`` could flatten an output but never unflatten one, so
every transform that rebuilds its result -- ``vmap``, a ``has_aux`` gradient, a
``tree_map`` -- would fail at the point of return.

``info`` travels with them rather than being dropped or recomputed because a
transform does not change how the run was discretised: the internal time step,
the resolved PML and the survey region of a ``vmap``\ ped call are the ones the
mapped call resolved. The two per-instance caches (``final_state`` and the
receiver views) are deliberately *not* carried: they are derived from the
children and the names, so a rebuilt output regenerates them on demand and
cannot carry a stale one.

:class:`deepwave.AuxiliaryState` is registered on the same principle, and for
it the context matters even more: it holds the tensor names and every recorded
run parameter, including ``format_version``. Those are what a continuation
checks itself against, so a state that lost them in a round trip would be
accepted by a call it does not belong to, or -- worse -- have a bundle written
to an older layout read as this release's.

Why a ``treespec`` holding one is not serialisable
--------------------------------------------------

The same context is what puts JSON serialisation of a *tree structure* out of
reach. ``torch.utils._pytree.treespec_dumps`` writes a ``treespec`` -- the
shape of a tree, with each node's context embedded -- as JSON, which
``torch.export`` uses. An output's context holds a :class:`deepwave.RunInfo`,
and an auxiliary state's holds its recorded run parameters; neither is JSON,
and inventing a JSON encoding for them would be inventing a serialisation
format for the state of a run, with the compatibility promise that implies, in
order to serve a tree *shape*. So neither type registers a serialised type
name, and ``treespec_dumps`` on a structure containing one raises PyTorch's
``NotImplementedError`` saying that no serialisation name is registered for it.

Nothing else is affected: flattening, unflattening, key paths, every
``torch.func`` transform, ``torch.compile``, ``tree_map``, and
``torch.save``/``torch.load`` or ``pickle`` of the *objects* (which use the
ordinary pickle protocol, not a ``treespec``) work as usual. Saving a state to
a file is what :class:`deepwave.AuxiliaryState`'s ``format_version`` exists
for, and that path is untouched.

What the registration needs from PyTorch
----------------------------------------

The registration is an enhancement rather than a requirement: a propagator
returns and accepts its output either way, and ``import deepwave`` must not
depend on it. ``deepwave._pytree`` therefore probes for each capability it uses
and degrades instead of failing:

* ``torch.utils._pytree.register_pytree_node``. Without it nothing is
  registered and both types stay pytree *leaves* -- an opaque whole, which is
  what they were before they were registered at all. Transforms that need to
  see inside them then fail at the point of use, as they did before.
* the ``flatten_with_keys_fn`` keyword and the ``GetAttrKey``/``MappingKey``
  classes, which arrived later than the function. Without them both types are
  still registered and every transform still works; only the *names* of the
  leaves are lost, so ``tree_flatten_with_path`` cannot key them.

A registration already in place -- another library that registered the same
type, or a module reload that runs the code twice -- is left alone rather than
overwritten, because objects flattened under it may still be held. None of
these cases warns: an older PyTorch losing an optional convenience is not
something a user of it can act on.

Why the class still subclasses ``tuple``
----------------------------------------

That decision is load-bearing for gradient propagation through
:class:`torch.autograd.Function` and :mod:`torch.utils.checkpoint`, and its
rationale stays in :class:`deepwave.PropagatorOutput`'s own docstring rather
than moving here, because anyone tempted to change the base class needs to
read it at the point of change.

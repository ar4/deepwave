"""Register Deepwave containers with the pytree API available in PyTorch 2.4+.

Keyed flattening lets torch.func and torch.compile see the tensors in outputs
and auxiliary state. A tree specification containing these types has no JSON
serialisation contract: its context includes run metadata.
"""

from typing import Any, Callable, Optional

import torch.utils._pytree as _torch_pytree


def attribute_key(name: str) -> Any:
    """Return the key for a named output field."""
    return _torch_pytree.GetAttrKey(name)


def mapping_key(name: str) -> Any:
    """Return the key for a named auxiliary-state tensor."""
    return _torch_pytree.MappingKey(name)


def register(
    cls: type[Any],
    flatten_fn: Callable[[Any], tuple[list[Any], Any]],
    unflatten_fn: Callable[[Any, Any], Any],
    flatten_with_keys_fn: Optional[Callable[[Any], tuple[list[Any], Any]]] = None,
) -> bool:
    """Register a container, preserving an existing registration on reload."""
    if cls not in _torch_pytree.SUPPORTED_NODES:
        _torch_pytree.register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            flatten_with_keys_fn=flatten_with_keys_fn,
        )
    return True

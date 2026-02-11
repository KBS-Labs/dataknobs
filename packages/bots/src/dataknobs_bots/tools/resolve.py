"""Utility for resolving dotted import paths to callables.

Used by ``from_config()`` classmethods on config tools to resolve
callable references (e.g., ``builder_factory``, ``on_save``) specified
as strings in YAML configuration.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def resolve_callable(ref: str) -> Callable[..., Any]:
    """Resolve a dotted import path to a callable.

    Supports two formats:
    - ``"module.path:attr_name"`` (setuptools entry_point style, preferred)
    - ``"module.path.attr_name"`` (dot-separated, fallback)

    Args:
        ref: Dotted import path string.

    Returns:
        The resolved callable.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute is not found on the module.
        ValueError: If the resolved attribute is not callable.
    """
    if ":" in ref:
        module_path, attr_name = ref.rsplit(":", 1)
    else:
        module_path, attr_name = ref.rsplit(".", 1)

    module = importlib.import_module(module_path)
    attr = getattr(module, attr_name)

    if not callable(attr):
        raise ValueError(
            f"Resolved attribute {ref!r} is not callable: {type(attr)}"
        )

    return attr  # type: ignore[return-value]

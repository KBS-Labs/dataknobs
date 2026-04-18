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


def resolve_optional_callable(
    ref: Any,
    *,
    field_name: str,
    owner: str,
) -> Callable[..., Any] | None:
    """Resolve an optional dotted-import reference to a callable.

    Thin wrapper over :func:`resolve_callable` that (a) returns
    ``None`` when ``ref`` is ``None`` (config field omitted), and
    (b) wraps the underlying import/lookup errors in a ``ValueError``
    that includes the owning object's name and the config field name.
    This is the standard lift consumers reach for when a config block
    accepts callable references under several optional keys — identity
    callables on a source, transform hooks on a bot, on-save handlers
    on a tool, etc.

    Args:
        ref: Dotted-import string (``"module.path:attr"`` or
            ``"module.path.attr"``), or ``None``.
        field_name: Name of the config field (for error messages).
        owner: Name of the owning object being built — typically a
            source name, bot name, tool name, etc. Included in the
            error message so misconfigured refs point back to their
            config site.

    Returns:
        The resolved callable, or ``None`` if ``ref`` is ``None``.

    Raises:
        ValueError: If ``ref`` is not ``None`` and cannot be resolved
            to a callable.
    """
    if ref is None:
        return None
    try:
        return resolve_callable(ref)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(
            f"{owner!r}: failed to resolve {field_name}={ref!r}: {e}"
        ) from e

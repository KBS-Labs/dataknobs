"""Chunker and transform registries backed by PluginRegistry."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from dataknobs_common.registry import PluginRegistry

from dataknobs_xization.chunking.base import ChunkTransform, Chunker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunker registry
# ---------------------------------------------------------------------------

def _register_builtins(registry: PluginRegistry[Chunker]) -> None:
    """Lazily register built-in chunkers on first access."""
    from dataknobs_xization.chunking.markdown import MarkdownTreeChunker

    registry.register("markdown_tree", MarkdownTreeChunker)


chunker_registry: PluginRegistry[Chunker] = PluginRegistry(
    "chunkers",
    canonicalize_keys=True,
    config_key="chunker",
    config_key_default="markdown_tree",
    strip_config_key=True,
    on_first_access=_register_builtins,
    validate_type=Chunker,
)


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

def _register_builtin_transforms(registry: PluginRegistry[ChunkTransform]) -> None:
    """Lazily register built-in transforms on first access."""
    from dataknobs_xization.chunking.transforms import (
        MergeSmallChunks,
        QualityFilterTransform,
        SplitLargeChunks,
    )

    registry.register("merge_small", MergeSmallChunks)
    registry.register("split_large", SplitLargeChunks)
    registry.register("quality_filter", QualityFilterTransform)


transform_registry: PluginRegistry[ChunkTransform] = PluginRegistry(
    "chunk_transforms",
    canonicalize_keys=True,
    config_key="transform",
    strip_config_key=True,
    on_first_access=_register_builtin_transforms,
    validate_type=ChunkTransform,
)


# ---------------------------------------------------------------------------
# Dotted import resolution
# ---------------------------------------------------------------------------

def _resolve_dotted_import(dotted_path: str, base_type: type) -> type:
    """Import a class from a dotted path and validate it.

    Accepts either ``module.ClassName`` or ``package.module.ClassName``.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
        TypeError: If the resolved object is not a subclass of *base_type*.
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Invalid dotted path '{dotted_path}': expected 'module.ClassName'"
        )

    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(
            f"'{class_name}' not found in module '{module_path}' "
            f"(from dotted path '{dotted_path}')"
        ) from None

    if not (isinstance(cls, type) and issubclass(cls, base_type)):
        raise TypeError(
            f"'{dotted_path}' resolved to {cls!r} which is not a "
            f"{base_type.__name__} subclass"
        )

    return cls


def _ensure_registered(
    registry: PluginRegistry[Any],
    key: str,
    base_type: type,
) -> None:
    """Register a dotted-path key if not already registered."""
    if "." in key and not registry.is_registered(key):
        cls = _resolve_dotted_import(key, base_type)
        registry.register(key, cls, override=True)
        logger.info("Registered custom %s '%s'", base_type.__name__, key)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_chunker(config: dict[str, Any] | None = None) -> Chunker:
    """Create a chunker from configuration.

    The ``chunker`` key in *config* selects the implementation:

    - **Registry key** (e.g. ``"markdown_tree"``): looks up a
      pre-registered factory.
    - **Dotted import path** (e.g. ``"my_project.chunkers.RFCChunker"``):
      dynamically imports the class, registers it, and creates an
      instance.

    When ``chunker`` is absent the default ``"markdown_tree"`` is used,
    preserving backward compatibility.

    If a ``transforms`` key is present, the resolved chunker is wrapped
    in a :class:`~dataknobs_xization.chunking.composite.CompositeChunker`
    that applies the transforms in order.  Each entry in ``transforms``
    is a dict with a single key (the transform registry key or dotted
    path) mapping to the transform's config dict::

        {
            "chunker": "markdown_tree",
            "max_chunk_size": 800,
            "transforms": [
                {"merge_small": {"min_size": 200}},
                {"my_project.transforms.Custom": {"flag": True}},
            ],
        }

    When ``transforms`` is absent or empty, no wrapper is created.

    Args:
        config: Chunking configuration dict.

    Returns:
        Configured chunker instance.
    """
    config = dict(config) if config else {}

    # Extract transforms before passing config to chunker
    transforms_config = config.pop("transforms", None)

    chunker_key = config.get("chunker", "markdown_tree")
    _ensure_registered(chunker_registry, chunker_key, Chunker)

    chunker = chunker_registry.create(config=config)

    # Wrap in CompositeChunker if transforms are configured
    if transforms_config:
        transforms = _build_transforms(transforms_config)
        if transforms:
            from dataknobs_xization.chunking.composite import CompositeChunker

            chunker = CompositeChunker(inner=chunker, transforms=transforms)

    return chunker


def _build_transforms(
    transforms_config: list[dict[str, Any]],
) -> list[ChunkTransform]:
    """Build transform instances from a config list.

    Each entry is a dict with a single key (registry key or dotted
    path) mapping to the transform's config dict.
    """
    transforms: list[ChunkTransform] = []
    for entry in transforms_config:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                f"Each transform entry must be a dict with exactly one key, "
                f"got: {entry!r}"
            )
        key, transform_config = next(iter(entry.items()))
        if not isinstance(transform_config, dict):
            transform_config = {}

        _ensure_registered(transform_registry, key, ChunkTransform)
        transform = transform_registry.create(
            config={"transform": key, **transform_config}
        )
        transforms.append(transform)
    return transforms


def register_chunker(
    key: str,
    factory: type[Chunker],
    override: bool = False,
) -> None:
    """Register a custom chunker implementation.

    Args:
        key: Short name for the chunker (e.g. ``"plaintext"``).
        factory: Chunker subclass or callable factory.
        override: Allow replacing an existing registration.
    """
    chunker_registry.register(key, factory, override=override)


def register_transform(
    key: str,
    factory: type[ChunkTransform],
    override: bool = False,
) -> None:
    """Register a custom chunk transform implementation.

    Args:
        key: Short name for the transform (e.g. ``"custom_merge"``).
        factory: ChunkTransform subclass or callable factory.
        override: Allow replacing an existing registration.
    """
    transform_registry.register(key, factory, override=override)

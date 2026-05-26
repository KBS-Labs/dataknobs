"""Typed configuration dataclasses for the memory backends.

Each concrete memory backend documents a small set of configuration keys.
These :class:`~dataknobs_common.structured_config.StructuredConfig`
subclasses capture those keys with defaults so the backend factories can
project a raw config dict onto a typed, validated, round-trippable object
before constructing the concrete memory instance.

The composite backend keeps its child specs as raw dicts — each child is
dispatched element-wise by its own ``type`` discriminator through the
backend registry, so discriminated selection stays in the registry layer
rather than being baked into the static field graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class BufferMemoryConfig(StructuredConfig):
    """Configuration for :class:`~dataknobs_bots.memory.buffer.BufferMemory`.

    Attributes:
        max_messages: Maximum number of messages retained in the FIFO buffer.
    """

    max_messages: int = 10


@dataclass(frozen=True)
class SummaryMemoryConfig(StructuredConfig):
    """Configuration for :class:`~dataknobs_bots.memory.summary.SummaryMemory`.

    Attributes:
        recent_window: Number of recent messages kept verbatim before the
            oldest are compressed into the running summary.
        summary_prompt: Optional custom summarization prompt template.
        llm: Optional dedicated LLM-provider config. When present, a
            dedicated provider is built and its lifecycle owned by the
            memory; when absent, the injected fallback provider (the bot's
            main LLM) is used. Kept as a raw mapping — LLM-provider config
            is owned by ``dataknobs-llm``.
    """

    recent_window: int = 10
    summary_prompt: str | None = None
    llm: dict[str, Any] | None = None

    # The ``llm`` section is dispatched by ``provider`` in the LLM provider
    # registry; binding it here lets ``validate()`` dry-run-build the
    # ``LLMConfig`` and catch an unknown provider / bad field at config-lint
    # time. The binding name is a string — ``dataknobs-llm`` registers the
    # ``llm`` resolver eagerly on import, so no LLM config type is imported.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = {"llm": "llm"}


@dataclass(frozen=True)
class VectorMemoryConfig(StructuredConfig):
    """Configuration for :class:`~dataknobs_bots.memory.vector.VectorMemory`.

    Attributes:
        backend: Vector-store backend key (``memory``, ``faiss``, …).
        dimension: Vector-store dimension (singular). Distinct from
            ``dimensions`` below, which is forwarded to the embedder.
        collection: Optional collection/index name for the store.
        persist_path: Optional persistence path for the store.
        store_params: Extra keyword arguments merged into the
            vector-store factory call.
        embedding: Nested embedding-provider config (preferred). Kept as
            a raw mapping — embedder config is owned by ``dataknobs-llm``.
        embedding_provider: Legacy flat embedding-provider key.
        embedding_model: Legacy flat embedding-model key.
        dimensions: Embedder dimension (plural). Forwarded to the
            embedding provider, NOT the vector store.
        api_base: Custom embedder endpoint, forwarded to the embedding
            provider as a legacy flat passthrough.
        api_key: Embedder credential, forwarded to the embedding provider
            as a legacy flat passthrough. Redacted from ``repr``.
        max_results: Maximum number of similar messages returned.
        similarity_threshold: Minimum similarity score (0-1) for results.
        default_metadata: Metadata merged into every ``add_message``.
        default_filter: Filter merged into every ``get_context`` search.
        immutable_metadata_keys: Keys whose ``default_metadata`` values
            cannot be overridden by caller-supplied metadata.
    """

    backend: str = "memory"
    dimension: int = 1536
    collection: str | None = None
    persist_path: str | None = None
    store_params: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    dimensions: int | None = None
    api_base: str | None = None
    api_key: str | None = None
    max_results: int = 5
    similarity_threshold: float = 0.7
    default_metadata: dict[str, Any] | None = None
    default_filter: dict[str, Any] | None = None
    immutable_metadata_keys: list[str] | None = None

    # Redacted from ``repr`` by the StructuredConfig base. A secret nested
    # inside the raw ``embedding`` mapping (its ``api_key``) is also masked:
    # the base's repr descends into raw ``Mapping``/``list`` fields and masks
    # interior keys in its default sensitive-key set unioned with this set.
    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})


@dataclass(frozen=True)
class CompositeMemoryConfig(StructuredConfig):
    """Configuration for :class:`~dataknobs_bots.memory.composite.CompositeMemory`.

    Attributes:
        strategies: Raw child memory specs. Each is dispatched element-wise
            by its own ``type`` discriminator through the backend registry,
            so this stays a list of raw dicts rather than typed per-strategy
            configs.
        primary_index: Index of the primary strategy in ``strategies``. The
            documented ``primary`` config key is accepted as an alias.
    """

    # Each ``strategies`` element is a raw memory spec dispatched by its own
    # ``type`` discriminator, so the ``"memory"`` resolver validates them
    # element-wise (the base's list-valued-section handling) under
    # ``CompositeMemoryConfig.from_dict(raw).validate()``.
    _polymorphic_fields: ClassVar[Mapping[str, str]] = {"strategies": "memory"}

    strategies: list[dict[str, Any]] = field(default_factory=list)
    primary_index: int = 0

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Accept the documented ``primary`` key as a ``primary_index`` alias."""
        if "primary" in raw and "primary_index" not in raw:
            raw["primary_index"] = raw.pop("primary")
        return raw

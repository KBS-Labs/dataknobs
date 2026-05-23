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

from dataclasses import dataclass, field
from typing import Any

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

    strategies: list[dict[str, Any]] = field(default_factory=list)
    primary_index: int = 0

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Accept the documented ``primary`` key as a ``primary_index`` alias."""
        if "primary" in raw and "primary_index" not in raw:
            raw["primary_index"] = raw.pop("primary")
        return raw

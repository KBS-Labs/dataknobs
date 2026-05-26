"""Configuration dataclass for the simple reasoning strategy."""

from __future__ import annotations

from dataclasses import dataclass

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class SimpleReasoningConfig(StructuredConfig):
    """Configuration for :class:`SimpleReasoning`.

    The simple strategy makes a direct LLM call with no extra reasoning
    steps, so its only configurable surface is the optional greeting
    template shared by every strategy.

    Attributes:
        greeting_template: Optional Jinja2 template for bot-initiated
            greetings (same semantics as other strategies).
    """

    greeting_template: str | None = None

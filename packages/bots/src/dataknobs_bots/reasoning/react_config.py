"""Configuration dataclass for the ReAct reasoning strategy."""

from __future__ import annotations

from dataclasses import dataclass

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class ReActReasoningConfig(StructuredConfig):
    """Configuration for :class:`ReActReasoning`.

    Captures the config-derived scalars that ``ReActReasoning.from_config``
    reads from its raw dict today.  Injected collaborators (artifact
    registry, review executor, context builder, prompt refresher) are *not*
    config — they travel through the constructor's keyword arguments and are
    deliberately excluded here.

    Attributes:
        max_iterations: Maximum reasoning/action iterations.
        verbose: Enable debug-level logging for reasoning steps.
        store_trace: Store the reasoning trace in conversation metadata.
        greeting_template: Optional Jinja2 template for bot-initiated
            greetings (same semantics as other strategies).
    """

    max_iterations: int = 5
    verbose: bool = False
    store_trace: bool = False
    greeting_template: str | None = None

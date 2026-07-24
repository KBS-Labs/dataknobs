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
        truncation_retry_max_tokens: When set, a tool-call turn the provider
            truncated at the token budget (``LLMResponse.truncated``) is
            retried **once** at this ``max_tokens`` before being abandoned.
            ``None`` (default) keeps the terminal behavior: a truncated tool
            call is abandoned and the turn is synthesized without retry. The
            requested budget is clamped to the model's output ceiling by the
            provider, so a generous value is safe. A still-truncated retry
            falls back to the terminal synthesis path — one attempt, no loop.
    """

    max_iterations: int = 5
    verbose: bool = False
    store_trace: bool = False
    greeting_template: str | None = None
    truncation_retry_max_tokens: int | None = None

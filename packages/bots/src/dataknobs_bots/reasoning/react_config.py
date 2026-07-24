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
            retried **once per truncated tool-call iteration** at this
            ``max_tokens`` before being abandoned. ``None`` (default) keeps the
            terminal behavior: a truncated tool call is abandoned and the turn
            is synthesized without retry. Must be a positive integer when set —
            ``0`` or a negative value is rejected at construction (a
            non-positive budget could never widen the response and would
            re-truncate). Set it comfortably above the configured ``max_tokens``
            so the retry actually has room. When the model advertises an output
            ceiling (e.g. the Claude family) the provider clamps the request to
            it, so an oversized value is safe; providers without a known ceiling
            (the project-default Ollama, HuggingFace, Echo) pass the budget
            through unclamped. Loop-safety does not depend on the clamp: the
            retry is structurally single-shot, so a still-truncated retry simply
            falls back to the terminal synthesis path — one attempt, no loop.
    """

    max_iterations: int = 5
    verbose: bool = False
    store_trace: bool = False
    greeting_template: str | None = None
    truncation_retry_max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Reject a non-positive truncation-retry budget at construction.

        A ``truncation_retry_max_tokens`` of ``0`` or negative is a
        misconfiguration, not "enabled": the runtime guard is
        ``budget is None``, so ``0`` would enable the retry with an impossible
        budget — a hard provider error (Anthropic requires ``max_tokens >= 1``)
        or, for a value below the configured ``max_tokens``, a guaranteed
        re-truncation that wastes a completion and then abandons. Validating
        here fails loud at config-construction for every consumer (including
        ``from_dict`` loading), so the guard lives once on the config rather
        than at each call site — mirroring ``RetryConfig.__post_init__``.
        """
        if (
            self.truncation_retry_max_tokens is not None
            and self.truncation_retry_max_tokens <= 0
        ):
            raise ValueError(
                "truncation_retry_max_tokens must be a positive integer when "
                f"set, got {self.truncation_retry_max_tokens}"
            )

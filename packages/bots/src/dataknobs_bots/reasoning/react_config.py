"""Configuration dataclass for the ReAct reasoning strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataknobs_common.structured_config import StructuredConfig

from dataknobs_bots.reasoning.config_base import ReasoningConfig

#: The compaction strategies ``HistoryCompactionConfig.strategy`` accepts.
_COMPACTION_STRATEGIES: frozenset[str] = frozenset({"window", "summarize"})


@dataclass(frozen=True)
class HistoryCompactionConfig(StructuredConfig):
    """Opt-in in-loop history compaction for a long ReAct tool loop.

    Bounds the conversation history a tool-using turn accumulates so it never
    trips a vendor input-context overflow. Disabled by default: an unset /
    ``enabled=False`` block is byte-identical to no compaction (no token
    estimation, no compaction call).

    Scope: bounds the **current** tool loop (from the last user message
    forward). The head — system prompt and all prior turns — is retained
    verbatim, so this does not bound cross-turn accumulation over a long
    multi-turn conversation.

    Attributes:
        enabled: Master switch. ``False`` (default) → no estimation, no
            compaction.
        budget_fraction: The proactive threshold as a fraction of the provider's
            resolved input ceiling (``ModelConstraints.max_input_tokens``). When
            the estimated history exceeds ``max_input_tokens * budget_fraction``
            the loop compacts before the next completion. The common path — used
            whenever the provider resolves an input ceiling (the Claude family
            via the live Models API / bundled fallback). Must be in ``(0, 1]``.
        history_token_budget: Absolute-token budget. When no input ceiling
            resolves (non-Anthropic providers, or an unknown model) it is the
            sole proactive threshold; when a ceiling *does* resolve it also
            **caps** ``max_input_tokens * budget_fraction`` — set it to your
            effective per-request window when that is smaller than the model's
            advertised maximum context (the published ceiling is the maximum
            attainable window, which a consumer may not actually have enabled).
            When both a resolved ceiling and this are unavailable, proactive
            compaction is disabled and only the reactive backstop (a caught
            context-overflow error) applies. Must be a positive integer when
            set.
        keep_recent_iterations: The number of most-recent tool iterations to
            retain verbatim on compaction. Must be ``>= 0``.
        strategy: ``"window"`` (default — drop the oldest iterations, LLM-free)
            or ``"summarize"`` (fold them into one summary node via an LLM call).
        summary_llm: Optional provider config (an ``LLMConfig``-shaped mapping)
            for the ``"summarize"`` strategy. ``None`` → reuse the bot's main
            provider as the summarizer. Ignored for ``"window"``.
    """

    enabled: bool = False
    budget_fraction: float = 0.75
    history_token_budget: int | None = None
    keep_recent_iterations: int = 3
    strategy: str = "window"
    summary_llm: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate the compaction knobs at construction (fail-loud)."""
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError(f"budget_fraction must be in (0, 1], got {self.budget_fraction}")
        if self.keep_recent_iterations < 0:
            raise ValueError(
                f"keep_recent_iterations must be >= 0, got {self.keep_recent_iterations}"
            )
        if self.history_token_budget is not None and self.history_token_budget <= 0:
            raise ValueError(
                "history_token_budget must be a positive integer when set, "
                f"got {self.history_token_budget}"
            )
        if self.strategy not in _COMPACTION_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_COMPACTION_STRATEGIES)}, got {self.strategy!r}"
            )


@dataclass(frozen=True)
class ReActReasoningConfig(ReasoningConfig):
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
        greeting_template: Declared once for the whole family on
            :class:`~dataknobs_bots.reasoning.config_base.ReasoningConfig`;
            inherited here.
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
    truncation_retry_max_tokens: int | None = None
    #: Opt-in in-loop history compaction. ``None`` / disabled → byte-identical
    #: to today (no estimation, no compaction). See
    #: :class:`HistoryCompactionConfig`.
    history_compaction: HistoryCompactionConfig | None = None

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
        if self.truncation_retry_max_tokens is not None and self.truncation_retry_max_tokens <= 0:
            raise ValueError(
                "truncation_retry_max_tokens must be a positive integer when "
                f"set, got {self.truncation_retry_max_tokens}"
            )

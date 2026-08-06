"""Per-turn pipeline state for DynaBot."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_llm import LLMResponse, LLMStreamResponse
    from dataknobs_llm.llm.model_profile import ModelPricing

    from .context import BotContext

logger = logging.getLogger(__name__)


class TurnMode(Enum):
    """How a turn was initiated."""

    CHAT = "chat"
    STREAM = "stream"
    GREET = "greet"


@dataclass
class ToolExecution:
    """Record of a single tool execution within a turn."""

    tool_name: str
    parameters: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None


@dataclass
class TurnState:
    """Carries all state through a single bot turn pipeline.

    Created at the start of ``chat()`` / ``stream_chat()`` / ``greet()``,
    threaded through each pipeline stage, and available to middleware via
    the ``after_turn`` hook.

    This is distinct from ``reasoning.wizard.TurnContext`` which carries
    per-turn FSM state for wizard transforms.
    """

    # --- Immutable inputs (set at creation) ---
    mode: TurnMode
    message: str  # "" for greet
    context: BotContext
    rag_query: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    llm_config_overrides: dict[str, Any] | None = None
    initial_context: dict[str, Any] | None = None  # greet only

    # --- Pipeline state (set during execution) ---
    manager: Any = None  # ConversationManager, set by _prepare_turn
    # True once this turn has pinned its conversation in the bounded manager
    # cache (set by ``_prepare_turn`` right after ``_get_or_create_conversation``
    # takes the pin).  The pin is released exactly once, in the turn driver's
    # ``finally`` via ``_call_finally_turn_middleware``, and ONLY when this flag
    # is set — so a turn that reached the ``finally`` without ever pinning (the
    # greet no-strategy early-exit, or an exception in ``_prepare_turn`` before
    # the pin point) does not decrement a pin it never took.  Pins are a global
    # per-key refcount; guarding the release per-turn is what makes the
    # "concurrent turns on the same id each hold their own pin" contract true
    # rather than relying on refcount underflow being harmless.
    pinned_conversation: bool = False
    response: LLMResponse | None = None  # set after generation (chat/greet)
    response_content: str = ""  # extracted text content
    stream_chunks: list[str] = field(default_factory=list)  # stream path only

    # True when the DynaBot tool loop terminated (break/cap) while a tool
    # call was still pending — i.e. an unexecuted assistant ``tool_use`` was
    # persisted to history and must be paired before the next turn replays it.
    # Set by the buffered/streaming monolithic-loop tails; gates the Layer-A
    # orphan-pairing history read in ``DynaBot._finalize_turn`` so the
    # already-paired majority (happy-path completion, no-tools, and phased
    # strategies that pair their own orphans) skips it.
    #
    # CONTRACT for future turn paths: this gate makes Layer-A pairing
    # opt-in, not automatic.  Any NEW path that can persist an unexecuted
    # ``tool_use`` to LLM history MUST either pair the orphan itself before
    # it returns (as phased strategies do — ReAct's "Layer B") OR set this
    # flag True so ``_finalize_turn`` pairs it (Layer A).  A path that does
    # neither leaves a dangling ``tool_use`` and the next turn's replay 400s
    # on Anthropic — the exact defect the finalize read exists to close.  If
    # you add a break/cap route to a tool loop, or a strategy that drives
    # tool calls without going through these loops, cover it in one of those
    # two ways and add a reproduce-first test alongside
    # ``tests/unit/test_finalize_orphan_pairing.py``'s T1-T8.
    tool_loop_left_pending_call: bool = False

    # --- Usage / observability ---
    usage: dict[str, int] | None = None  # token usage from response
    model: str | None = None  # model that generated the response
    # Canonical provider *family* key (e.g. "openai") — the value to key
    # rate tables, metrics labels, and log fields on.
    provider_name: str | None = None
    # Concrete provider *class* (e.g. "CachingEmbedProvider") — diagnostic
    # only.  Carried separately because ``provider_name`` deliberately no
    # longer reports it, and ``TurnState`` discards the provider object after
    # reading a name, so nothing downstream could recover it otherwise.
    provider_impl: str | None = None
    # Per-model USD pricing the provider resolved for this turn's model, or
    # ``None`` when it sources none.  Captured for the same reason as
    # ``provider_impl``: the provider object is in hand here and gone by the
    # time any consumer wants it, so a rate not taken now has to be guessed
    # from a second table later — which is how the middleware's hand-written
    # duplicate came to exist and to drift.
    pricing: ModelPricing | None = None

    # --- Tool tracking ---
    tool_executions: list[ToolExecution] = field(default_factory=list)

    # --- Plugin data (cross-middleware communication) ---
    plugin_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_streaming(self) -> bool:
        """Whether this turn uses the streaming path."""
        return self.mode == TurnMode.STREAM

    @property
    def is_greet(self) -> bool:
        """Whether this turn is a bot-initiated greeting."""
        return self.mode == TurnMode.GREET

    def middleware_kwargs(self) -> dict[str, Any]:
        """Build backward-compatible kwargs dict for legacy middleware hooks.

        Provides ``tokens_used``, ``model``, ``provider`` from the turn's
        response data, matching the format that ``after_message`` consumers
        expect.
        """
        kwargs: dict[str, Any] = {}
        if self.usage:
            kwargs["tokens_used"] = self.usage
        if self.model:
            kwargs["model"] = self.model
        if self.provider_name:
            kwargs["provider"] = self.provider_name
        return kwargs

    def populate_from_response(self, response: Any, provider: Any) -> None:
        """Extract usage/model/provider info from an LLM response."""
        if hasattr(response, "usage") and response.usage:
            self.usage = response.usage
        if hasattr(response, "model") and response.model:
            self.model = response.model
        self._extract_provider_name(provider)
        self._capture_pricing(provider)

    def populate_from_final_stream_chunk(
        self, chunk: LLMStreamResponse, provider: Any
    ) -> None:
        """Extract usage from the final streaming chunk."""
        if chunk.usage:
            self.usage = chunk.usage
        if chunk.model:
            self.model = chunk.model
        self._extract_provider_name(provider)
        self._capture_pricing(provider)

    def accumulate_usage(self, response: Any) -> None:
        """Add usage from an intermediate LLM response to the running total.

        Called during the tool execution loop to capture token counts
        from re-generation calls that would otherwise be discarded when
        ``populate_from_response`` overwrites ``self.usage`` with the
        final call's data.
        """
        resp_usage = getattr(response, "usage", None)
        if not resp_usage:
            return
        self._add_usage(resp_usage)

    def accumulate_usage_from_stream(self) -> None:
        """Snapshot current streaming usage before a re-stream round.

        In the streaming tool loop, ``populate_from_final_stream_chunk``
        overwrites ``self.usage`` each round.  Call this before each
        re-stream to fold the current round's usage into the running
        total.
        """
        if self.usage:
            # Stash current usage — populate_from_final_stream_chunk will
            # overwrite self.usage with the next round's data.
            stashed = dict(self.usage)
            self.usage = None
            self._add_usage(stashed)

    def _add_usage(self, new_usage: dict[str, int]) -> None:
        """Merge token counts into the running total."""
        if self.usage is None:
            self.usage = {}
        for key in ("input", "output", "prompt_tokens", "completion_tokens",
                     "total_tokens"):
            if key in new_usage:
                self.usage[key] = self.usage.get(key, 0) + new_usage[key]

    def _extract_provider_name(self, provider: Any) -> None:
        """Capture both provider axes from a provider instance.

        The two are captured independently so an explicit family key does not
        suppress the diagnostic one.

        ``provider_name`` is the canonical family key and is left ``None``
        when the object cannot supply one. A class name is deliberately *not*
        substituted: this field keys rate tables, metrics labels, and log
        fields, and a class name in it is precisely the defect the accessor
        pair exists to close — writing one here would re-open it one layer
        down, and consumers read ``"unknown"`` plus a warning instead, which
        is the truth. The class remains available on ``provider_impl``.
        """
        if provider is None:
            return
        name = getattr(provider, "provider_name", None)
        if name:
            self.provider_name = name
        # ``impl_name`` before the class name: re-deriving it inline ignores a
        # provider that has declared what it wants to be called, which is the
        # same reconstruct-it-yourself pattern this contract replaced.
        self.provider_impl = (
            getattr(provider, "impl_name", None) or type(provider).__name__
        )

    def _capture_pricing(self, provider: Any) -> None:
        """Resolve this turn's per-model pricing from the provider.

        ``get_pricing`` reads an already-resolved model profile — no I/O, no
        network — so this is safe on every turn.

        Failures are swallowed deliberately. Pricing is observability: a
        provider whose profile resolution raises must degrade to "unpriced"
        (the consumer's own rate table then applies, and the miss warning
        fires if it has nothing either) rather than fail the conversation it
        is only measuring.
        """
        get_pricing = getattr(provider, "get_pricing", None)
        if get_pricing is None:
            return
        try:
            self.pricing = get_pricing(self.model) if self.model else get_pricing()
        # Broad by intent: any failure resolving a rate must degrade to
        # "unpriced" rather than propagate into the turn being measured.
        except Exception:
            logger.debug(
                "Could not resolve pricing for %s/%s",
                self.provider_name,
                self.model,
                exc_info=True,
            )

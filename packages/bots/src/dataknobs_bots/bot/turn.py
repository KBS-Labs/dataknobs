"""Per-turn pipeline state for DynaBot."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_llm import LLMResponse, LLMStreamResponse

    from .context import BotContext


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
    response: LLMResponse | None = None  # set after generation (chat/greet)
    response_content: str = ""  # extracted text content
    stream_chunks: list[str] = field(default_factory=list)  # stream path only

    # --- Usage / observability ---
    usage: dict[str, int] | None = None  # token usage from response
    model: str | None = None  # model that generated the response
    provider_name: str | None = None  # provider name

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

    def populate_from_final_stream_chunk(
        self, chunk: LLMStreamResponse, provider: Any
    ) -> None:
        """Extract usage from the final streaming chunk."""
        if chunk.usage:
            self.usage = chunk.usage
        if chunk.model:
            self.model = chunk.model
        self._extract_provider_name(provider)

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
        """Set provider_name from a provider instance."""
        if provider is None:
            return
        name = getattr(provider, "provider_name", None)
        if name:
            self.provider_name = name
        elif hasattr(provider, "__class__"):
            self.provider_name = type(provider).__name__

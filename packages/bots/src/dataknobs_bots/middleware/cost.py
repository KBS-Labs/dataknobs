"""Cost tracking middleware for monitoring LLM usage."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from .base import Middleware

if TYPE_CHECKING:
    from dataknobs_bots.bot.context import BotContext
    from dataknobs_bots.bot.turn import TurnState

logger = logging.getLogger(__name__)


class CostTrackingMiddleware(Middleware):
    """Middleware for tracking LLM API costs and usage.

    Monitors token usage across different providers (Ollama, OpenAI, Anthropic, etc.)
    to help optimize costs and track budgets.

    Attributes:
        track_tokens: Whether to track token usage
        cost_rates: Token cost rates per provider/model
        usage_stats: Accumulated usage statistics by client_id

    Example:
        ```python
        # Create middleware with default rates
        middleware = CostTrackingMiddleware()

        # Or with custom rates
        middleware = CostTrackingMiddleware(
            cost_rates={
                "openai": {
                    "gpt-4o": {"input": 0.0025, "output": 0.01},
                },
            }
        )

        # Get stats
        stats = middleware.get_client_stats("my-client")
        total = middleware.get_total_cost()

        # Export to JSON
        json_data = middleware.export_stats_json()
        ```
    """

    # Default cost rates (USD per 1K tokens) - Updated Dec 2024
    DEFAULT_RATES: dict[str, Any] = {
        "ollama": {"input": 0.0, "output": 0.0},  # Free (infrastructure cost only)
        "openai": {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "o1": {"input": 0.015, "output": 0.06},
            "o1-mini": {"input": 0.003, "output": 0.012},
        },
        "anthropic": {
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        },
        "google": {
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        },
    }

    def __init__(
        self,
        track_tokens: bool = True,
        cost_rates: dict[str, Any] | None = None,
    ):
        """Initialize cost tracking middleware.

        Args:
            track_tokens: Enable token tracking
            cost_rates: Optional custom cost rates (merged with defaults)
        """
        self.track_tokens = track_tokens
        # Merge custom rates with defaults
        self.cost_rates = self.DEFAULT_RATES.copy()
        if cost_rates:
            for provider, rates in cost_rates.items():
                if provider in self.cost_rates:
                    if isinstance(rates, dict) and isinstance(
                        self.cost_rates[provider], dict
                    ):
                        self.cost_rates[provider].update(rates)
                    else:
                        self.cost_rates[provider] = rates
                else:
                    self.cost_rates[provider] = rates

        self._usage_stats: dict[str, dict[str, Any]] = {}
        self._logger = logging.getLogger(f"{__name__}.CostTracker")

    @staticmethod
    def _new_client_stats(client_id: str) -> dict[str, Any]:
        """Create a fresh stats dict for a client."""
        return {
            "client_id": client_id,
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "after_message_calls": 0,
            "post_stream_calls": 0,
            "on_error_calls": 0,
            "on_hook_error_calls": 0,
            "by_provider": {},
        }

    def _record_usage(
        self,
        client_id: str,
        hook_counter: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage and cost for a request.

        Called by ``after_turn`` for consistent stats tracking
        (totals, by-provider, by-model).

        Args:
            client_id: Client identifier
            hook_counter: Stats key to increment (e.g. "after_message_calls")
            provider: LLM provider name
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Calculated cost in USD.
        """
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)

        if client_id not in self._usage_stats:
            self._usage_stats[client_id] = self._new_client_stats(client_id)

        stats = self._usage_stats[client_id]
        stats["total_requests"] += 1
        stats[hook_counter] += 1
        stats["total_input_tokens"] += input_tokens
        stats["total_output_tokens"] += output_tokens
        stats["total_cost_usd"] += cost

        # Track by provider
        if provider not in stats["by_provider"]:
            stats["by_provider"][provider] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "by_model": {},
            }

        provider_stats = stats["by_provider"][provider]
        provider_stats["requests"] += 1
        provider_stats["input_tokens"] += input_tokens
        provider_stats["output_tokens"] += output_tokens
        provider_stats["cost_usd"] += cost

        # Track by model within provider
        if model not in provider_stats["by_model"]:
            provider_stats["by_model"][model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        model_stats = provider_stats["by_model"][model]
        model_stats["requests"] += 1
        model_stats["input_tokens"] += input_tokens
        model_stats["output_tokens"] += output_tokens
        model_stats["cost_usd"] += cost

        return cost

    async def on_turn_start(self, turn: TurnState) -> str | None:
        """Log estimated input tokens at the start of a turn.

        Args:
            turn: Turn state at the start of the pipeline.

        Returns:
            None (no message transform).
        """
        # Estimate input tokens (rough approximation: ~4 chars per token)
        estimated_tokens = len(turn.message) // 4
        self._logger.debug("Estimated input tokens: %d", estimated_tokens)
        return None

    async def after_turn(self, turn: TurnState) -> None:
        """Track costs after turn completion using TurnState data.

        Uses real token usage when the provider reports it, otherwise
        estimates from message/response text length (~4 chars per token).

        Args:
            turn: Completed turn state with usage and response data.
        """
        if not self.track_tokens:
            return

        client_id = turn.context.client_id
        provider = turn.provider_name or "unknown"
        model = turn.model or "unknown"

        if turn.usage:
            input_tokens = int(
                turn.usage.get(
                    "input",
                    turn.usage.get("prompt_tokens", 0),
                )
            )
            output_tokens = int(
                turn.usage.get(
                    "output",
                    turn.usage.get("completion_tokens", 0),
                )
            )
            estimated = False
        else:
            # Estimate from text length (~4 chars per token).
            # Note: turn.message is the user's message before KB/memory
            # augmentation.  The actual LLM input includes system prompt,
            # KB chunks, and memory context, so this underestimates real
            # input tokens.  When real usage data is available (above
            # branch), this fallback is not reached.
            input_tokens = len(turn.message) // 4
            output_tokens = len(turn.response_content) // 4
            estimated = True

        hook_counter = (
            "post_stream_calls" if turn.is_streaming else "after_message_calls"
        )
        cost = self._record_usage(
            client_id, hook_counter,
            provider, model, input_tokens, output_tokens,
        )

        total = self._usage_stats[client_id]["total_cost_usd"]
        mode_label = turn.mode.value
        est_marker = " (estimated)" if estimated else ""
        self._logger.info(
            "Turn complete (%s) - Client %s: %s/%s - "
            "%d in + %d out tokens%s, cost: $%.6f, total: $%.6f",
            mode_label, client_id, provider, model,
            input_tokens, output_tokens, est_marker, cost, total,
        )

    async def on_error(
        self, error: Exception, message: str, context: BotContext
    ) -> None:
        """Log errors but don't track costs for failed requests.

        Args:
            error: The exception that occurred
            message: User message that caused the error
            context: Bot context
        """
        client_id = context.client_id
        if client_id not in self._usage_stats:
            self._usage_stats[client_id] = self._new_client_stats(client_id)
        self._usage_stats[client_id]["on_error_calls"] += 1

        self._logger.warning(
            "Error during request for client %s: %s", client_id, error,
        )

    async def on_hook_error(
        self, hook_name: str, error: Exception, context: BotContext
    ) -> None:
        """Track middleware hook failures.

        Args:
            hook_name: Name of the hook that failed
            error: The exception raised by the middleware hook
            context: Bot context
        """
        client_id = context.client_id
        if client_id not in self._usage_stats:
            self._usage_stats[client_id] = self._new_client_stats(client_id)
        self._usage_stats[client_id]["on_hook_error_calls"] += 1

        self._logger.warning(
            "Middleware hook %s failed for client %s: %s",
            hook_name, client_id, error,
        )

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for token usage.

        Args:
            provider: LLM provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get rates for provider/model
        if provider in self.cost_rates:
            provider_rates = self.cost_rates[provider]

            if isinstance(provider_rates, dict):
                # Check if model-specific rates exist
                if model in provider_rates:
                    rates = provider_rates[model]
                elif "input" in provider_rates:
                    # Use generic rates for provider (e.g., ollama)
                    rates = provider_rates
                else:
                    # Try partial model name match
                    for model_key in provider_rates:
                        if model_key in model or model in model_key:
                            rates = provider_rates[model_key]
                            break
                    else:
                        return 0.0
            else:
                return 0.0

            # Calculate cost (rates are per 1K tokens)
            input_cost = (input_tokens / 1000) * float(rates.get("input", 0.0))
            output_cost = (output_tokens / 1000) * float(rates.get("output", 0.0))
            return float(input_cost + output_cost)

        return 0.0

    def get_client_stats(self, client_id: str) -> dict[str, Any] | None:
        """Get usage statistics for a client.

        Args:
            client_id: Client identifier

        Returns:
            Usage statistics or None if not found
        """
        return self._usage_stats.get(client_id)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get all usage statistics.

        Returns:
            Dictionary mapping client_id to statistics
        """
        return self._usage_stats.copy()

    def get_total_cost(self) -> float:
        """Get total cost across all clients.

        Returns:
            Total cost in USD
        """
        return float(
            sum(stats["total_cost_usd"] for stats in self._usage_stats.values())
        )

    def get_total_tokens(self) -> dict[str, int]:
        """Get total tokens across all clients.

        Returns:
            Dictionary with 'input', 'output', and 'total' token counts
        """
        input_tokens = sum(
            stats["total_input_tokens"] for stats in self._usage_stats.values()
        )
        output_tokens = sum(
            stats["total_output_tokens"] for stats in self._usage_stats.values()
        )
        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

    def clear_stats(self, client_id: str | None = None) -> None:
        """Clear usage statistics.

        Args:
            client_id: If provided, clear only this client. Otherwise clear all.
        """
        if client_id:
            if client_id in self._usage_stats:
                del self._usage_stats[client_id]
        else:
            self._usage_stats.clear()

    def export_stats_json(self, indent: int = 2) -> str:
        """Export all statistics as JSON.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string of all statistics
        """
        return json.dumps(self._usage_stats, indent=indent)

    def export_stats_csv(self) -> str:
        """Export statistics as CSV (one row per client).

        Returns:
            CSV string with headers
        """
        lines = [
            "client_id,total_requests,total_input_tokens,total_output_tokens,total_cost_usd"
        ]
        for client_id, stats in self._usage_stats.items():
            lines.append(
                f"{client_id},{stats['total_requests']},"
                f"{stats['total_input_tokens']},{stats['total_output_tokens']},"
                f"{stats['total_cost_usd']:.6f}"
            )
        return "\n".join(lines)

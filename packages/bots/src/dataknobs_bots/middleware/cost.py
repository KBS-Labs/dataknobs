"""Cost tracking middleware for monitoring LLM usage."""

import json
import logging
from typing import Any

from dataknobs_bots.bot.context import BotContext

from .base import Middleware

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

    async def before_message(self, message: str, context: BotContext) -> None:
        """Track message before processing (mainly for logging).

        Args:
            message: User's input message
            context: Bot context
        """
        # Estimate input tokens (rough approximation: ~4 chars per token)
        estimated_tokens = len(message) // 4
        self._logger.debug(f"Estimated input tokens: {estimated_tokens}")

    async def after_message(
        self, response: str, context: BotContext, **kwargs: Any
    ) -> None:
        """Track costs after bot response.

        Args:
            response: Bot's generated response
            context: Bot context
            **kwargs: Should contain 'tokens_used', 'provider', 'model' if available
        """
        if not self.track_tokens:
            return

        client_id = context.client_id

        # Extract provider and model info
        provider = kwargs.get("provider", "unknown")
        model = kwargs.get("model", "unknown")

        # Get token counts
        tokens_used = kwargs.get("tokens_used", {})
        if isinstance(tokens_used, int):
            # If single number, assume it's total and estimate split
            input_tokens = len(context.session_metadata.get("last_message", "")) // 4
            output_tokens = tokens_used - input_tokens
        else:
            input_tokens = int(
                tokens_used.get(
                    "input",
                    tokens_used.get(
                        "prompt_tokens",
                        len(context.session_metadata.get("last_message", "")) // 4,
                    ),
                )
            )
            output_tokens = int(
                tokens_used.get(
                    "output",
                    tokens_used.get("completion_tokens", len(response) // 4),
                )
            )

        # Calculate cost
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)

        # Update stats
        if client_id not in self._usage_stats:
            self._usage_stats[client_id] = {
                "client_id": client_id,
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "by_provider": {},
            }

        stats = self._usage_stats[client_id]
        stats["total_requests"] += 1
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

        self._logger.info(
            f"Client {client_id}: {provider}/{model} - "
            f"{input_tokens} in + {output_tokens} out tokens, "
            f"cost: ${cost:.6f}, total: ${stats['total_cost_usd']:.6f}"
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
        self._logger.warning(
            f"Error during request for client {context.client_id}: {error}"
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

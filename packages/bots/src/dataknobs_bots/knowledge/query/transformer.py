"""Query transformation using LLM for improved retrieval.

This module provides LLM-based query transformation to generate
optimized search queries from user input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for query transformation.

    Attributes:
        enabled: Whether transformation is enabled
        llm_provider: LLM provider name (e.g., "ollama", "openai")
        llm_model: Model to use for transformation
        num_queries: Number of alternative queries to generate
        domain_context: Domain-specific context for better queries
        suppress_thinking: Whether to pass ``think: false`` to the LLM
            provider via ``config_overrides``.  Prevents thinking models
            from spending their full token budget on reasoning before
            producing short query strings.  Recommended when query
            generation is a utility call (e.g., grounded strategy),
            not a reasoning task.
    """

    enabled: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    num_queries: int = 3
    domain_context: str = ""
    suppress_thinking: bool = False


def parse_query_response(response_text: str, fallback: str) -> list[str]:
    """Parse an LLM response into a list of search queries.

    Strips numbering, bullets, and quotes from each line.  Returns
    ``[fallback]`` if no valid queries are extracted.

    Args:
        response_text: Raw LLM response text.
        fallback: Fallback query when parsing produces nothing.

    Returns:
        List of parsed queries.
    """
    lines = response_text.strip().split("\n")
    queries: list[str] = []

    for line in lines:
        cleaned = line.strip()
        cleaned = cleaned.lstrip("0123456789.-) ")
        cleaned = cleaned.strip("\"'")

        if cleaned and len(cleaned) > 2:
            queries.append(cleaned)

    return queries if queries else [fallback]


class QueryTransformer:
    """LLM-based query transformation for improved RAG retrieval.

    Transforms user input into optimized search queries by using an LLM
    to extract key concepts and generate alternative phrasings.

    Supports two provider management modes:

    **External provider** (preferred for composition)::

        transformer = QueryTransformer(config, provider=my_provider)
        # or: transformer.set_provider(my_provider)
        queries = await transformer.transform("user question")

    **Self-managed provider** (standalone use)::

        transformer = QueryTransformer(config)
        await transformer.initialize()  # creates provider from config
        queries = await transformer.transform("user question")
        await transformer.close()  # releases the provider

    When an external provider is supplied, the transformer does **not**
    own it — ``close()`` will not release it.

    Example:
        ```python
        config = TransformerConfig(
            enabled=True,
            domain_context="prompt engineering",
        )
        transformer = QueryTransformer(config, provider=echo_provider)

        queries = await transformer.transform(
            "Analyze this: Write a poem about cats"
        )
        # Returns: ["prompt analysis techniques", "evaluating prompt quality", ...]
        ```
    """

    def __init__(
        self,
        config: TransformerConfig | None = None,
        provider: Any | None = None,
    ) -> None:
        """Initialize the query transformer.

        Args:
            config: Transformer configuration, uses defaults if not provided.
            provider: Optional pre-built LLM provider.  When given, the
                transformer is immediately ready (no ``initialize()`` call
                needed) and ``close()`` will not release the provider.
        """
        self.config = config or TransformerConfig()
        self._llm: Any | None = None
        self._initialized = False
        self._owns_provider = False

        if provider is not None:
            self._llm = provider
            self._initialized = True
            self._owns_provider = False

    def set_provider(self, provider: Any) -> None:
        """Set or replace the LLM provider (external injection).

        The transformer does **not** own this provider and will not
        close it.  This is the preferred injection point when the
        transformer is composed inside a larger pipeline (e.g.
        :class:`GroundedReasoning`).

        Args:
            provider: An LLM provider with an async ``complete()`` method.
        """
        self._llm = provider
        self._initialized = True
        self._owns_provider = False

    async def initialize(self) -> None:
        """Initialize a self-managed LLM provider from config.

        Must be called before ``transform()`` when no external provider
        was supplied.  Skipped when ``config.enabled`` is ``False``.
        """
        if not self.config.enabled:
            return

        from dataknobs_llm.llm import LLMProviderFactory

        factory = LLMProviderFactory(is_async=True)
        self._llm = factory.create({
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
        })
        await self._llm.initialize()
        self._initialized = True
        self._owns_provider = True

    async def close(self) -> None:
        """Close the LLM provider and release resources.

        Only closes the provider if it was created by ``initialize()``
        (self-managed).  Externally-injected providers are not closed.
        """
        if self._owns_provider and self._llm and hasattr(self._llm, "close"):
            await self._llm.close()
        self._initialized = False

    async def transform(
        self,
        user_input: str,
        num_queries: int | None = None,
    ) -> list[str]:
        """Transform user input into optimized search queries.

        Args:
            user_input: The user's message or question
            num_queries: Number of queries to generate (overrides config)

        Returns:
            List of optimized search queries

        Raises:
            RuntimeError: If transformer is enabled but not initialized
        """
        # If disabled, return the original input as a single query
        if not self.config.enabled:
            return [user_input]

        if not self._initialized:
            raise RuntimeError(
                "QueryTransformer not initialized. Call initialize() first."
            )

        num = num_queries or self.config.num_queries
        prompt = self._build_prompt(user_input, num)

        response_text = await self._call_llm(prompt)
        queries = self._parse_response(response_text, user_input)

        return queries[:num]

    async def transform_with_context(
        self,
        user_input: str,
        conversation_context: str,
        num_queries: int | None = None,
    ) -> list[str]:
        """Transform with additional conversation context.

        Args:
            user_input: The user's message
            conversation_context: Recent conversation history
            num_queries: Number of queries to generate

        Returns:
            List of optimized search queries
        """
        if not self.config.enabled:
            return [user_input]

        if not self._initialized:
            raise RuntimeError(
                "QueryTransformer not initialized. Call initialize() first."
            )

        num = num_queries or self.config.num_queries
        prompt = self._build_contextual_prompt(
            user_input, conversation_context, num,
        )

        response_text = await self._call_llm(prompt)
        queries = self._parse_response(response_text, user_input)

        return queries[:num]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, user_input: str, num_queries: int) -> str:
        """Build the transformation prompt.

        Args:
            user_input: User's message
            num_queries: Number of queries to generate

        Returns:
            Prompt string for LLM
        """
        domain_context = ""
        if self.config.domain_context:
            domain_context = f" in the context of {self.config.domain_context}"

        return (
            f"Generate {num_queries} search queries to find relevant "
            f"knowledge base content for the following user "
            f"message{domain_context}.\n\n"
            f'User message: "{user_input}"\n\n'
            "Focus on:\n"
            "- Key concepts and techniques being discussed\n"
            "- The underlying intent, not the literal text\n"
            "- Related topics that would provide useful context\n\n"
            "Return ONLY the search queries, one per line, without "
            "numbering or explanation.\n"
            "Keep each query concise (2-6 words)."
        )

    def _build_contextual_prompt(
        self,
        user_input: str,
        conversation_context: str,
        num_queries: int,
    ) -> str:
        """Build prompt with conversation context.

        Args:
            user_input: User's message
            conversation_context: Recent conversation
            num_queries: Number of queries to generate

        Returns:
            Prompt string for LLM
        """
        domain_context = ""
        if self.config.domain_context:
            domain_context = f" in the context of {self.config.domain_context}"

        return (
            f"Generate {num_queries} search queries to find relevant "
            f"knowledge base content for the user's "
            f"message{domain_context}.\n\n"
            f"Recent conversation context:\n{conversation_context}\n\n"
            f'Current user message: "{user_input}"\n\n'
            "Focus on:\n"
            "- Key concepts relevant to what the user is asking\n"
            "- Context from the conversation that clarifies the query\n"
            "- Related topics that would provide useful information\n\n"
            "Return ONLY the search queries, one per line, without "
            "numbering or explanation.\n"
            "Keep each query concise (2-6 words)."
        )

    # ------------------------------------------------------------------
    # LLM call + response parsing
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM provider and return the response text.

        When ``suppress_thinking`` is enabled, passes
        ``options: {"think": false}`` via ``config_overrides`` to prevent
        thinking models from spending their token budget on reasoning for
        a utility extraction call.  The ``think`` key is provider-specific
        (Ollama/qwen3) and goes through the ``options`` override path.
        """
        config_overrides: dict[str, Any] | None = None
        if self.config.suppress_thinking:
            config_overrides = {"options": {"think": False}}

        response = await self._llm.complete(
            prompt, config_overrides=config_overrides,
        )
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _parse_response(self, response: str, fallback: str) -> list[str]:
        """Parse LLM response into list of queries.

        Delegates to the module-level :func:`parse_query_response`.

        Args:
            response: Raw LLM response
            fallback: Fallback query if parsing fails

        Returns:
            List of parsed queries
        """
        return parse_query_response(response, fallback)


async def create_transformer(config: dict[str, Any]) -> QueryTransformer:
    """Create and initialize a QueryTransformer from config dict.

    Convenience function for creating transformer from configuration.

    Args:
        config: Configuration dictionary with TransformerConfig fields

    Returns:
        Initialized QueryTransformer

    Example:
        ```python
        transformer = await create_transformer({
            "enabled": True,
            "llm_provider": "ollama",
            "llm_model": "llama3.2",
            "domain_context": "prompt engineering"
        })
        ```
    """
    transformer_config = TransformerConfig(**config)
    transformer = QueryTransformer(transformer_config)
    await transformer.initialize()
    return transformer

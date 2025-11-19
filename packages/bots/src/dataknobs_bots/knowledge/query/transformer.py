"""Query transformation using LLM for improved retrieval.

This module provides LLM-based query transformation to generate
optimized search queries from user input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TransformerConfig:
    """Configuration for query transformation.

    Attributes:
        enabled: Whether transformation is enabled
        llm_provider: LLM provider name (e.g., "ollama", "openai")
        llm_model: Model to use for transformation
        num_queries: Number of alternative queries to generate
        domain_context: Domain-specific context for better queries
    """

    enabled: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    num_queries: int = 3
    domain_context: str = ""


class QueryTransformer:
    """LLM-based query transformation for improved RAG retrieval.

    Transforms user input into optimized search queries by using an LLM
    to extract key concepts and generate alternative phrasings.

    This is particularly useful when:
    - User input contains literal text to analyze (not queries)
    - User asks vague questions that need expansion
    - Domain-specific terminology needs translation

    Example:
        ```python
        config = TransformerConfig(
            enabled=True,
            llm_provider="ollama",
            llm_model="llama3.2",
            domain_context="prompt engineering"
        )
        transformer = QueryTransformer(config)
        await transformer.initialize()

        # Transform user input to search queries
        queries = await transformer.transform(
            "Analyze this: Write a poem about cats"
        )
        # Returns: ["prompt analysis techniques", "evaluating prompt quality", ...]
        ```
    """

    def __init__(self, config: TransformerConfig | None = None):
        """Initialize the query transformer.

        Args:
            config: Transformer configuration, uses defaults if not provided
        """
        self.config = config or TransformerConfig()
        self._llm = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LLM provider.

        Must be called before using transform() if enabled.
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

    async def close(self) -> None:
        """Close the LLM provider and release resources."""
        if self._llm and hasattr(self._llm, "close"):
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

        # Build the transformation prompt
        prompt = self._build_prompt(user_input, num)

        # Generate queries using LLM
        response = await self._llm.generate(prompt)

        # Parse the response into individual queries
        queries = self._parse_response(response, user_input)

        return queries[:num]

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

        return f"""Generate {num_queries} search queries to find relevant knowledge base content for the following user message{domain_context}.

User message: "{user_input}"

Focus on:
- Key concepts and techniques being discussed
- The underlying intent, not the literal text
- Related topics that would provide useful context

Return ONLY the search queries, one per line, without numbering or explanation.
Keep each query concise (2-6 words).
"""

    def _parse_response(self, response: str, fallback: str) -> list[str]:
        """Parse LLM response into list of queries.

        Args:
            response: Raw LLM response
            fallback: Fallback query if parsing fails

        Returns:
            List of parsed queries
        """
        # Split by newlines and clean up
        lines = response.strip().split("\n")
        queries = []

        for line in lines:
            # Remove common prefixes (numbering, bullets, etc.)
            cleaned = line.strip()
            cleaned = cleaned.lstrip("0123456789.-) ")
            cleaned = cleaned.strip('"\'')

            if cleaned and len(cleaned) > 2:
                queries.append(cleaned)

        # Ensure we have at least one query
        if not queries:
            queries = [fallback]

        return queries

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

        # Build enhanced prompt with context
        prompt = self._build_contextual_prompt(
            user_input, conversation_context, num
        )

        response = await self._llm.generate(prompt)
        queries = self._parse_response(response, user_input)

        return queries[:num]

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

        return f"""Generate {num_queries} search queries to find relevant knowledge base content for the user's message{domain_context}.

Recent conversation context:
{conversation_context}

Current user message: "{user_input}"

Focus on:
- Key concepts relevant to what the user is asking
- Context from the conversation that clarifies the query
- Related topics that would provide useful information

Return ONLY the search queries, one per line, without numbering or explanation.
Keep each query concise (2-6 words).
"""


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

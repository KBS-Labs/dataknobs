"""Contextual query expansion using conversation history.

This module provides query expansion without requiring LLM calls,
using recent conversation context to enrich ambiguous queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Message:
    """A conversation message.

    Attributes:
        role: Message role ("user", "assistant", "system")
        content: Message content
    """

    role: str
    content: str


class ContextualExpander:
    """Expands queries using conversation context.

    This expander enriches ambiguous or context-dependent queries
    by incorporating information from recent conversation turns.
    Unlike QueryTransformer, it doesn't require LLM calls.

    Example:
        ```python
        expander = ContextualExpander(max_context_turns=3)

        # User asks: "Show me an example"
        # Recent context: discussing chain-of-thought prompting
        expanded = expander.expand(
            "Show me an example",
            conversation_history
        )
        # Returns: "chain-of-thought prompting examples Show me an example"
        ```
    """

    def __init__(
        self,
        max_context_turns: int = 3,
        include_assistant: bool = False,
        keyword_weight: int = 2,
    ):
        """Initialize the contextual expander.

        Args:
            max_context_turns: Maximum conversation turns to consider
            include_assistant: Whether to include assistant messages
            keyword_weight: How many times to repeat extracted keywords
        """
        self.max_context_turns = max_context_turns
        self.include_assistant = include_assistant
        self.keyword_weight = keyword_weight

        # Common words to filter out
        self._stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
            "who", "when", "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "also", "now",
            "here", "there", "about", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "for", "yet", "because", "as", "until",
            "while", "of", "at", "by", "with", "without", "between", "me", "my",
            "your", "his", "her", "its", "our", "their", "please", "help", "want",
            "need", "like", "show", "tell", "give", "make", "let", "get", "see",
        }

    def expand(
        self,
        user_input: str,
        conversation_history: list[Message] | list[dict[str, Any]],
    ) -> str:
        """Expand query with conversation context.

        Args:
            user_input: The user's current message
            conversation_history: Recent conversation messages

        Returns:
            Expanded query string
        """
        # Normalize conversation history to Message objects
        messages = self._normalize_messages(conversation_history)

        # Get recent context
        recent = self._get_recent_context(messages)

        # Extract keywords from context
        keywords = self._extract_keywords(recent)

        # Build expanded query
        if keywords:
            keyword_str = " ".join(keywords)
            return f"{keyword_str} {user_input}"

        return user_input

    def _normalize_messages(
        self,
        history: list[Message] | list[dict[str, Any]],
    ) -> list[Message]:
        """Normalize history to Message objects.

        Args:
            history: Conversation history in various formats

        Returns:
            List of Message objects
        """
        messages = []
        for item in history:
            if isinstance(item, Message):
                messages.append(item)
            elif isinstance(item, dict):
                messages.append(Message(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                ))
        return messages

    def _get_recent_context(self, messages: list[Message]) -> list[str]:
        """Get recent relevant context from conversation.

        Args:
            messages: Conversation messages

        Returns:
            List of context strings
        """
        context = []
        count = 0

        # Walk backwards through messages
        for msg in reversed(messages):
            if count >= self.max_context_turns:
                break

            if msg.role == "user" or (msg.role == "assistant" and self.include_assistant):
                context.insert(0, msg.content)
                count += 1

        return context

    def _extract_keywords(self, context: list[str]) -> list[str]:
        """Extract meaningful keywords from context.

        Args:
            context: List of context strings

        Returns:
            List of extracted keywords
        """
        # Combine all context
        combined = " ".join(context)

        # Tokenize and filter
        words = combined.lower().split()
        keywords = []

        for word in words:
            # Clean punctuation
            cleaned = word.strip(".,!?\"'()[]{}:;")

            # Skip short words, stop words, and numbers
            if (
                len(cleaned) < 3
                or cleaned in self._stop_words
                or cleaned.isdigit()
            ):
                continue

            # Add keyword if not already present
            if cleaned not in keywords:
                keywords.append(cleaned)

        # Return top keywords (most recent first gives natural weighting)
        return keywords[:5]

    def expand_with_topics(
        self,
        user_input: str,
        conversation_history: list[Message] | list[dict[str, Any]],
        topic_extractor: Callable[[str], list[str]] | None = None,
    ) -> str:
        """Expand query with extracted topics.

        Enhanced expansion that uses a custom topic extractor.

        Args:
            user_input: The user's current message
            conversation_history: Recent conversation messages
            topic_extractor: Optional function to extract topics from text

        Returns:
            Expanded query string
        """
        messages = self._normalize_messages(conversation_history)
        recent = self._get_recent_context(messages)

        if topic_extractor:
            # Use custom topic extraction
            topics = []
            for text in recent:
                topics.extend(topic_extractor(text))
            topic_str = " ".join(topics[:5])
        else:
            # Fall back to keyword extraction
            keywords = self._extract_keywords(recent)
            topic_str = " ".join(keywords)

        if topic_str:
            return f"{topic_str} {user_input}"

        return user_input


def is_ambiguous_query(query: str) -> bool:
    """Check if a query is likely ambiguous and needs expansion.

    Args:
        query: The query to check

    Returns:
        True if query appears ambiguous

    Example:
        ```python
        is_ambiguous_query("Show me an example")  # True
        is_ambiguous_query("How do I configure OAuth?")  # False
        ```
    """
    # Short queries are often ambiguous
    words = query.split()
    if len(words) < 4:
        return True

    # Queries with demonstratives are often context-dependent
    ambiguous_patterns = [
        "this", "that", "these", "those", "it", "them",
        "example", "more", "another", "same", "similar",
    ]

    query_lower = query.lower()
    for pattern in ambiguous_patterns:
        if pattern in query_lower:
            return True

    return False

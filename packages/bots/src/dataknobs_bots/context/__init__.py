"""Context management for conversational workflows.

This package provides infrastructure for accumulating and managing
conversation context, including:

- ConversationContext: Unified context from wizard state, artifacts, assumptions
- ContextBuilder: Builds context from conversation manager
- ContextPersister: Persists context to conversation metadata
- Assumption: Tracked assumptions about user intent
- ContextSection: Prioritized sections for prompt injection

Example:
    >>> from dataknobs_bots.context import (
    ...     ConversationContext,
    ...     ContextBuilder,
    ...     ContextPersister,
    ... )
    >>>
    >>> # Build context from manager
    >>> builder = ContextBuilder(artifact_registry=registry)
    >>> context = builder.build(manager)
    >>>
    >>> # Add an assumption
    >>> context.add_assumption(
    ...     content="User wants a math tutor",
    ...     source="inferred",
    ...     confidence=0.7,
    ... )
    >>>
    >>> # Generate prompt injection
    >>> prompt_context = context.to_prompt_injection(max_tokens=2000)
    >>>
    >>> # Persist changes
    >>> persister = ContextPersister()
    >>> persister.persist(context, manager)
"""

from .accumulator import (
    Assumption,
    AssumptionSource,
    ContextSection,
    ConversationContext,
)
from .builder import ContextBuilder, ContextPersister

__all__ = [
    # Accumulator
    "ConversationContext",
    "Assumption",
    "AssumptionSource",
    "ContextSection",
    # Builder
    "ContextBuilder",
    "ContextPersister",
]

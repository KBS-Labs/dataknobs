"""Context builder for constructing conversation context.

This module provides:
- ContextBuilder: Builds ConversationContext from manager and registries
- ContextPersister: Persists context back to conversation metadata

Example:
    >>> builder = ContextBuilder(artifact_registry=registry)
    >>> context = builder.build(manager)
    >>> prompt_context = context.to_prompt_injection(max_tokens=2000)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .accumulator import Assumption, ConversationContext

if TYPE_CHECKING:
    from ..artifacts.registry import ArtifactRegistry

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds ConversationContext from conversation manager and related sources.

    The builder gathers context from:
    - Conversation manager metadata
    - Artifact registry
    - Tool execution records
    - Wizard state

    Example:
        >>> builder = ContextBuilder(
        ...     artifact_registry=registry,
        ...     tool_registry=tool_registry,
        ... )
        >>> context = builder.build(manager)
        >>> prompt_context = context.to_prompt_injection(max_tokens=2000)
    """

    def __init__(
        self,
        artifact_registry: ArtifactRegistry | None = None,
        tool_registry: Any | None = None,
    ) -> None:
        """Initialize builder.

        Args:
            artifact_registry: Optional ArtifactRegistry for artifact data
            tool_registry: Optional ToolRegistry for execution history
        """
        self._artifact_registry = artifact_registry
        self._tool_registry = tool_registry

    def build(self, manager: Any) -> ConversationContext:
        """Build context from conversation manager.

        Args:
            manager: ConversationManager instance

        Returns:
            Complete ConversationContext
        """
        metadata = getattr(manager, "metadata", {}) or {}

        context = ConversationContext(
            conversation_id=getattr(manager, "conversation_id", None),
        )

        # Extract wizard state
        self._extract_wizard_state(context, metadata)

        # Extract artifacts
        self._extract_artifacts(context, metadata)

        # Extract assumptions
        self._extract_assumptions(context, metadata)

        # Extract tool history
        self._extract_tool_history(context, metadata)

        # Extract transitions
        self._extract_transitions(context, metadata)

        return context

    def build_from_metadata(
        self,
        metadata: dict[str, Any],
        conversation_id: str | None = None,
    ) -> ConversationContext:
        """Build context directly from metadata dict.

        Useful when you have metadata but not a full manager.

        Args:
            metadata: Conversation metadata dictionary
            conversation_id: Optional conversation ID

        Returns:
            ConversationContext
        """
        context = ConversationContext(conversation_id=conversation_id)

        self._extract_wizard_state(context, metadata)
        self._extract_artifacts(context, metadata)
        self._extract_assumptions(context, metadata)
        self._extract_tool_history(context, metadata)
        self._extract_transitions(context, metadata)

        return context

    def _extract_wizard_state(
        self,
        context: ConversationContext,
        metadata: dict[str, Any],
    ) -> None:
        """Extract wizard state from metadata.

        Args:
            context: Context to populate
            metadata: Conversation metadata
        """
        wizard_meta = metadata.get("wizard", {})
        fsm_state = wizard_meta.get("fsm_state", {})

        context.wizard_stage = fsm_state.get("current_stage")
        context.wizard_data = fsm_state.get("data", {})
        context.wizard_progress = wizard_meta.get("progress", 0.0)

        # Extract tasks from task tracking
        tasks_data = fsm_state.get("tasks", {})
        context.wizard_tasks = tasks_data.get("tasks", [])

    def _extract_artifacts(
        self,
        context: ConversationContext,
        metadata: dict[str, Any],
    ) -> None:
        """Extract artifacts from registry or metadata.

        Args:
            context: Context to populate
            metadata: Conversation metadata
        """
        # Try registry first
        if self._artifact_registry:
            context.artifacts = [
                a.to_dict() for a in self._artifact_registry._artifacts.values()
            ]
        else:
            # Fall back to metadata
            context.artifacts = metadata.get("artifacts", [])

    def _extract_assumptions(
        self,
        context: ConversationContext,
        metadata: dict[str, Any],
    ) -> None:
        """Extract assumptions from metadata.

        Args:
            context: Context to populate
            metadata: Conversation metadata
        """
        context_data = metadata.get("context", {})
        assumptions_data = context_data.get("assumptions", [])
        for a_data in assumptions_data:
            context.assumptions.append(Assumption.from_dict(a_data))

    def _extract_tool_history(
        self,
        context: ConversationContext,
        metadata: dict[str, Any],
    ) -> None:
        """Extract tool execution history.

        Args:
            context: Context to populate
            metadata: Conversation metadata
        """
        # Try registry first
        if self._tool_registry and hasattr(self._tool_registry, "get_execution_history"):
            try:
                history = self._tool_registry.get_execution_history(
                    context_id=context.conversation_id,
                )
                context.tool_history = [
                    {
                        "tool_name": r.tool_name,
                        "timestamp": r.timestamp,
                        "success": r.success,
                        "duration_ms": r.duration_ms,
                    }
                    for r in history
                ]
            except Exception as e:
                logger.warning("Failed to get tool history from registry: %s", e)
                context.tool_history = metadata.get("tool_history", [])
        else:
            # Fall back to metadata
            context.tool_history = metadata.get("tool_history", [])

    def _extract_transitions(
        self,
        context: ConversationContext,
        metadata: dict[str, Any],
    ) -> None:
        """Extract wizard transitions.

        Args:
            context: Context to populate
            metadata: Conversation metadata
        """
        wizard_meta = metadata.get("wizard", {})
        fsm_state = wizard_meta.get("fsm_state", {})
        context.transitions = fsm_state.get("transitions", [])


class ContextPersister:
    """Persists ConversationContext to conversation metadata.

    Used to save context changes back to the conversation manager.

    Example:
        >>> persister = ContextPersister()
        >>> persister.persist(context, manager)
    """

    def persist(
        self,
        context: ConversationContext,
        manager: Any,
    ) -> None:
        """Persist context to conversation manager metadata.

        Args:
            context: Context to persist
            manager: ConversationManager instance
        """
        metadata = getattr(manager, "metadata", {}) or {}

        # Update context section in metadata
        metadata["context"] = {
            "assumptions": [a.to_dict() for a in context.assumptions],
            "sections": [
                {
                    "name": s.name,
                    "content": s.content,
                    "priority": s.priority,
                }
                for s in context.sections
            ],
            "updated_at": context.updated_at,
        }

        # Set metadata back on manager
        manager.metadata = metadata

        logger.debug(
            "Persisted context",
            extra={
                "conversation_id": context.conversation_id,
                "assumptions_count": len(context.assumptions),
                "sections_count": len(context.sections),
            },
        )

    def persist_to_dict(self, context: ConversationContext) -> dict[str, Any]:
        """Convert context to metadata dict format.

        Useful when you need to update metadata without a manager.

        Args:
            context: Context to convert

        Returns:
            Dictionary suitable for conversation metadata
        """
        return {
            "context": {
                "assumptions": [a.to_dict() for a in context.assumptions],
                "sections": [
                    {
                        "name": s.name,
                        "content": s.content,
                        "priority": s.priority,
                    }
                    for s in context.sections
                ],
                "updated_at": context.updated_at,
            }
        }

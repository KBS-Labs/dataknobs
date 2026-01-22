"""Tool execution context for passing runtime information to tools.

This module provides context classes that allow tools to receive
information about the execution environment, conversation state,
and wizard progress without tight coupling to specific implementations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WizardStateSnapshot:
    """Snapshot of wizard state for tool context.

    Provides tools with read-only access to wizard state information
    without coupling them to the full WizardReasoning implementation.

    Attributes:
        current_stage: Name of the current wizard stage
        collected_data: Data collected across all stages
        history: List of visited stage names
        completed: Whether the wizard has finished
        stage_metadata: Metadata for the current stage (prompt, schema, etc.)
    """

    current_stage: str | None = None
    collected_data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    completed: bool = False
    stage_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manager_metadata(cls, metadata: dict[str, Any]) -> "WizardStateSnapshot":
        """Create snapshot from conversation manager metadata.

        Args:
            metadata: The manager.metadata dict containing wizard state

        Returns:
            WizardStateSnapshot populated from metadata
        """
        wizard_data = metadata.get("wizard", {})
        fsm_state = wizard_data.get("fsm_state", {})

        return cls(
            current_stage=fsm_state.get("current_stage"),
            collected_data=fsm_state.get("data", {}),
            history=fsm_state.get("history", []),
            completed=fsm_state.get("completed", False),
            stage_metadata={},  # Stage metadata not stored in fsm_state
        )


@dataclass
class ToolExecutionContext:
    """Context available to tools during execution.

    This context is passed to ContextAwareTool implementations,
    providing access to conversation state, user information,
    and wizard progress without requiring tools to have direct
    dependencies on conversation managers or bot infrastructure.

    Attributes:
        conversation_id: Unique identifier for the conversation
        user_id: Optional user identifier
        client_id: Optional client/session identifier
        conversation_metadata: Full conversation metadata dict
        wizard_state: Optional wizard state snapshot
        request_metadata: Per-request metadata (headers, etc.)
        extra: Additional context for custom use cases

    Example:
        ```python
        class MyTool(ContextAwareTool):
            async def execute_with_context(
                self,
                context: ToolExecutionContext,
                query: str,
                **kwargs
            ) -> dict:
                # Access wizard data if available
                if context.wizard_state:
                    domain_id = context.wizard_state.collected_data.get("domain_id")

                # Access user info
                user_id = context.user_id

                return {"result": f"Processed for user {user_id}"}
        ```
    """

    conversation_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    conversation_metadata: dict[str, Any] = field(default_factory=dict)
    wizard_state: WizardStateSnapshot | None = None
    request_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "ToolExecutionContext":
        """Create an empty context for tools that don't need context.

        Returns:
            Empty ToolExecutionContext instance
        """
        return cls()

    @classmethod
    def from_manager(
        cls,
        manager: Any,
        request_metadata: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "ToolExecutionContext":
        """Build context from a ConversationManager instance.

        This is the primary factory method for creating context
        during tool execution in reasoning strategies.

        Args:
            manager: ConversationManager instance
            request_metadata: Optional per-request metadata
            extra: Optional additional context

        Returns:
            ToolExecutionContext populated from manager

        Example:
            ```python
            # In reasoning strategy
            context = ToolExecutionContext.from_manager(manager)
            result = await tool.execute(**params, _context=context)
            ```
        """
        # Extract conversation ID
        conversation_id = getattr(manager, "conversation_id", None)

        # Extract metadata
        metadata = getattr(manager, "metadata", {}) or {}

        # Build wizard state if present
        wizard_state = None
        if "wizard" in metadata:
            wizard_state = WizardStateSnapshot.from_manager_metadata(metadata)

        return cls(
            conversation_id=conversation_id,
            conversation_metadata=metadata,
            wizard_state=wizard_state,
            request_metadata=request_metadata or {},
            extra=extra or {},
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from extra context.

        Provides dict-like access to extra context values.

        Args:
            key: Key to look up
            default: Default value if key not found

        Returns:
            Value from extra or default
        """
        return self.extra.get(key, default)

    def with_extra(self, **kwargs: Any) -> "ToolExecutionContext":
        """Create a new context with additional extra values.

        Does not modify the original context.

        Args:
            **kwargs: Additional key-value pairs to add

        Returns:
            New ToolExecutionContext with merged extra values
        """
        new_extra = {**self.extra, **kwargs}
        return ToolExecutionContext(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            client_id=self.client_id,
            conversation_metadata=self.conversation_metadata,
            wizard_state=self.wizard_state,
            request_metadata=self.request_metadata,
            extra=new_extra,
        )

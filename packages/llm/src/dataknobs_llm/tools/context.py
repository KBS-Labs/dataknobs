"""Tool execution context for passing runtime information to tools.

This module provides context classes that allow tools to receive
information about the execution environment, conversation state,
and wizard progress without tight coupling to specific implementations.

Wizard state reaches a tool by one of two routes, and which one ran
decides whether the tool is looking at live state or at the last save;
:class:`ToolWizardState` documents the difference and
:meth:`ToolExecutionContext.wizard_data` is the supported way to read it.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolWizardState:
    """The wizard state a tool is allowed to see.

    Two suppliers build this object, and which one ran decides what a
    tool's reads and writes actually do:

    * **Published (live).** A reasoning strategy may publish an instance
      on ``ConversationState.live_wizard_state`` for the duration of a
      turn. ``collected_data`` is then the strategy's own dict, held by
      reference: a tool's writes land in wizard state, and its reads see
      values extracted earlier in the *same* turn.
    * **Metadata fallback (the last save).** When no strategy published,
      :meth:`from_manager_metadata` reads the *persisted* wizard
      metadata. ``collected_data`` is then the persisted dict itself, so
      a tool's writes are visible to the rest of the turn — but the
      component that owns the wizard rewrites that dict from its own
      state when the turn is saved, so they do not survive it. Reads are
      as old as the last save for the same reason.

    Prefer :meth:`ToolExecutionContext.wizard_data` over reaching into
    ``collected_data`` directly: it returns ``None`` when there is no
    wizard state at all, so a tool cannot write into a throwaway dict and
    report success.

    Attributes:
        current_stage: Name of the current wizard stage
        collected_data: Data collected across all stages
        history: List of visited stage names
        completed: Whether the wizard has finished
        stage_metadata: Metadata for the current stage (prompt, schema,
            etc.). Only the publisher can supply this; the metadata
            fallback leaves it empty.
    """

    current_stage: str | None = None
    collected_data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    completed: bool = False
    stage_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manager_metadata(cls, metadata: dict[str, Any]) -> "ToolWizardState":
        """Create state from persisted conversation manager metadata.

        This is the fallback route described in the class docstring: it
        reads the wizard's *saved* FSM state, so the result is as old as
        the last save rather than current.

        Args:
            metadata: The manager.metadata dict containing wizard state

        Returns:
            ToolWizardState populated from metadata
        """
        wizard_data = metadata.get("wizard", {})
        fsm_state = wizard_data.get("fsm_state", {})

        return cls(
            current_stage=fsm_state.get("current_stage"),
            collected_data=fsm_state.get("data", {}),
            history=fsm_state.get("history", []),
            completed=fsm_state.get("completed", False),
            # Stage metadata is not part of the persisted fsm_state, so it
            # is unavailable on this route. A publisher holds it and fills
            # it in; see the class docstring.
            stage_metadata={},
        )


# Deprecated alias.
#
# ``WizardStateSnapshot`` is also the name of an unrelated and much larger
# observability dataclass in ``dataknobs_bots.reasoning.observability``.
# The two are routinely confused — shipped prose already documents a field
# of the ``bots`` class under an import of this one — so the tool-facing
# class is now spelled ``ToolWizardState``.
#
# .. deprecated::
#    Use :class:`ToolWizardState`. This alias resolves for one minor
#    version and is then removed.
WizardStateSnapshot = ToolWizardState


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
        wizard_state: Optional wizard state — live or a copy, see
            :class:`ToolWizardState`
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
                # Access wizard data if available. ``None`` means there is
                # no wizard state to read or write — not an empty wizard.
                wizard_data = context.wizard_data()
                if wizard_data is None:
                    return {"error": "requires a wizard conversation"}
                domain_id = wizard_data.get("domain_id")

                # Access user info
                user_id = context.user_id

                return {"result": f"Processed for user {user_id}"}
        ```
    """

    conversation_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    conversation_metadata: dict[str, Any] = field(default_factory=dict)
    wizard_state: ToolWizardState | None = None
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
    def from_wizard_data(cls, wizard_data: dict[str, Any], **kwargs: Any) -> "ToolExecutionContext":
        """Create context from wizard collected data for standalone tool use.

        Convenience factory for calling wizard-aware tools outside the
        DynaBot framework (tests, scripts, direct invocation). The dict is
        held by reference, so a tool's writes are visible to the caller.

        Args:
            wizard_data: The wizard's collected data dictionary.
            **kwargs: Additional ToolExecutionContext fields
                (conversation_id, user_id, etc.).

        Returns:
            ToolExecutionContext with wizard state populated.
        """
        return cls(
            wizard_state=ToolWizardState(collected_data=wizard_data),
            **kwargs,
        )

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

        Wizard state is taken from the live view a reasoning strategy
        published for this turn when there is one, and rebuilt from the
        persisted metadata when there is not. See :class:`ToolWizardState`
        for what that difference means to a tool.

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

        state = getattr(manager, "state", None)

        # Prefer the live view the reasoning strategy published for this
        # turn; fall back to the persisted metadata when it published none.
        wizard_state: ToolWizardState | None = None
        if state is not None:
            wizard_state = getattr(state, "live_wizard_state", None)
        if wizard_state is None and "wizard" in metadata:
            wizard_state = ToolWizardState.from_manager_metadata(metadata)

        # Bridge turn_data from ConversationState into extra so tools
        # can read/write per-turn plugin data set by the bot layer.
        merged_extra = dict(extra or {})
        if state is not None:
            turn_data = getattr(state, "turn_data", None)
            if turn_data:
                merged_extra["turn_data"] = turn_data

        return cls(
            conversation_id=conversation_id,
            conversation_metadata=metadata,
            wizard_state=wizard_state,
            request_metadata=request_metadata or {},
            extra=merged_extra,
        )

    def wizard_data(self) -> dict[str, Any] | None:
        """The wizard's collected data, or ``None`` when there is none.

        This is the supported way for a tool to reach wizard data. The
        dict is returned by reference, so writes to it are writes to
        whatever the context is backed by — live wizard state when a
        strategy published one, the persisted metadata otherwise, which
        is not the same thing (see :class:`ToolWizardState`).

        Returns ``None``, deliberately, rather than an empty dict: a tool
        called outside a wizard would otherwise write into a throwaway
        and report success, which is indistinguishable from working. A
        tool that needs wizard data should treat ``None`` as an error
        condition and say so in its result.

        Returns:
            The collected-data dict, or ``None`` if the context carries no
            wizard state.
        """
        if self.wizard_state is None:
            return None
        return self.wizard_state.collected_data

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

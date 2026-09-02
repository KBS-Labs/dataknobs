"""Conversation flow definitions using FSM backend.

This module provides high-level abstractions for defining and executing
conversation flows using the FSM engine.

``ConversationFlow``, ``FlowState``, ``TransitionCondition`` and the condition
types need no FSM engine and are imported eagerly. ``ConversationFlowAdapter``
and ``FlowExecutionState`` are the two names that do, and they are resolved
lazily — see :func:`__getattr__`.
"""

from typing import TYPE_CHECKING, Any

from .flow import (
    ConversationFlow,
    FlowState,
    TransitionCondition,
)

from .conditions import (
    AlwaysCondition,
    KeywordCondition,
    RegexCondition,
    LLMClassifierCondition,
    ContextCondition,
    CompositeCondition,
    SentimentCondition,
    # Factory functions
    keyword_condition,
    regex_condition,
    always,
    context_condition,
)

if TYPE_CHECKING:  # pragma: no cover - annotation-only, no runtime import
    from .adapter import ConversationFlowAdapter, FlowExecutionState

__all__ = [
    # Core classes
    "ConversationFlow",
    "FlowState",
    "TransitionCondition",
    # Adapter
    "ConversationFlowAdapter",
    "FlowExecutionState",
    # Conditions
    "AlwaysCondition",
    "KeywordCondition",
    "RegexCondition",
    "LLMClassifierCondition",
    "ContextCondition",
    "CompositeCondition",
    "SentimentCondition",
    # Factory functions
    "keyword_condition",
    "regex_condition",
    "always",
    "context_condition",
]

_LAZY_ADAPTER_NAMES = frozenset({"ConversationFlowAdapter", "FlowExecutionState"})


def __getattr__(name: str) -> Any:
    """Lazily expose the two FSM-backed names (PEP 562).

    ``.adapter`` imports ``dataknobs_fsm`` at module scope, so re-exporting it
    eagerly pulled the optional ``dataknobs-fsm`` dependency into every import
    of this package. That reached further than it looks: ``manager.py`` imports
    the FSM-free leaf ``.flow``, and importing a submodule runs this
    ``__init__`` first — so the eager re-export made
    ``import dataknobs_llm.conversations`` (and with it ``ConversationManager``)
    fail on any install without ``dataknobs-fsm``.

    Deferring to first access keeps both names importable from this package
    exactly as before, for the callers that do want the adapter, while the
    engine stays out of the base install. Install ``dataknobs-llm[fsm]`` to
    use them.
    """
    if name in _LAZY_ADAPTER_NAMES:
        from . import adapter

        return getattr(adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

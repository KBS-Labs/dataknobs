"""Conversation flow definitions using FSM backend.

This module provides high-level abstractions for defining and executing
conversation flows using the FSM engine.
"""

from .flow import (
    ConversationFlow,
    FlowState,
    TransitionCondition,
)

from .adapter import (
    ConversationFlowAdapter,
    FlowExecutionState,
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

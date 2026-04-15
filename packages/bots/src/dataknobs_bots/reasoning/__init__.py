"""Reasoning strategies for DynaBot."""

from typing import Any

from .base import (
    PhasedReasoningProtocol,
    ProcessResult,
    ReasoningManagerProtocol,
    ReasoningStrategy,
    StreamStageContext,
    StreamingPhasedProtocol,
    StrategyCapabilities,
    ToolCallSpec,
    TurnHandle,
)
from .focus_guard import FocusContext, FocusEvaluation, FocusGuard
from .grounded import GroundedReasoning, SynthesisPlan
from .grounded_config import (
    GroundedIntentConfig,
    GroundedReasoningConfig,
    GroundedRetrievalConfig,
    GroundedSourceConfig,
    GroundedSynthesisConfig,
)
from .hybrid import HybridReasoning
from .hybrid_config import HybridReasoningConfig
from .observability import (
    # Task tracking types
    TaskCompletionTrigger,
    TaskStatus,
    WizardTask,
    WizardTaskList,
    # Wizard-specific types
    TransitionHistoryQuery,
    TransitionRecord,
    TransitionStats,
    TransitionTracker,
    WizardStateSnapshot,
    create_transition_record,
    # Conversion utilities
    execution_record_to_transition_record,
    transition_record_to_execution_record,
    transition_stats_to_execution_stats,
    # Re-exported FSM types
    ExecutionHistoryQuery,
    ExecutionRecord,
    ExecutionStats,
    ExecutionTracker,
)
from .react import ReActReasoning, ReActTurnHandle
from .simple import SimpleReasoning
from .task_injection import (
    TaskInjectionContext,
    TaskInjectionResult,
    TaskInjector,
)
from .wizard import WizardAdvanceResult, WizardReasoning, WizardStageContext, WizardState
from .wizard_response import StageResponseResult
from .wizard_types import RecoveryResult, ToolResultMappingEntry, WizardTurnHandle
from .wizard_derivations import DerivationRule, FieldTransform
from .wizard_fsm import WizardFSM
from .wizard_hooks import WizardHooks
from .wizard_loader import WizardConfigLoader, load_wizard_config
from .registry import (
    StrategyFactory,
    get_registry,
    get_strategy_factory,
    is_strategy_registered,
    list_strategies,
    register_strategy,
)

__all__ = [
    "PhasedReasoningProtocol",
    "ProcessResult",
    "ReasoningManagerProtocol",
    "ReasoningStrategy",
    "RecoveryResult",
    "StreamStageContext",
    "StreamingPhasedProtocol",
    "ToolResultMappingEntry",
    "StrategyCapabilities",
    "StrategyFactory",
    "ToolCallSpec",
    "TurnHandle",
    "WizardTurnHandle",
    "register_strategy",
    "is_strategy_registered",
    "list_strategies",
    "get_strategy_factory",
    "get_registry",
    "SimpleReasoning",
    "ReActReasoning",
    "ReActTurnHandle",
    "GroundedReasoning",
    "GroundedReasoningConfig",
    "GroundedIntentConfig",
    "GroundedRetrievalConfig",
    "GroundedSynthesisConfig",
    "GroundedSourceConfig",
    "SynthesisPlan",
    "HybridReasoning",
    "HybridReasoningConfig",
    "WizardAdvanceResult",
    "WizardReasoning",
    "WizardStageContext",
    "WizardState",
    "StageResponseResult",
    "WizardFSM",
    "WizardHooks",
    "WizardConfigLoader",
    "load_wizard_config",
    "create_reasoning_from_config",
    # Task tracking
    "WizardTask",
    "WizardTaskList",
    "TaskStatus",
    "TaskCompletionTrigger",
    # Task injection
    "TaskInjector",
    "TaskInjectionContext",
    "TaskInjectionResult",
    # Focus guard
    "FocusGuard",
    "FocusContext",
    "FocusEvaluation",
    # Wizard observability
    "TransitionRecord",
    "TransitionHistoryQuery",
    "TransitionStats",
    "TransitionTracker",
    "WizardStateSnapshot",
    "create_transition_record",
    # Conversion utilities
    "transition_record_to_execution_record",
    "execution_record_to_transition_record",
    "transition_stats_to_execution_stats",
    # Field derivation
    "DerivationRule",
    "FieldTransform",
    # FSM observability (re-exported)
    "ExecutionRecord",
    "ExecutionHistoryQuery",
    "ExecutionStats",
    "ExecutionTracker",
]


def create_reasoning_from_config(
    config: dict[str, Any],
    *,
    knowledge_base: Any | None = None,
    prompt_resolver: Any | None = None,
) -> ReasoningStrategy:
    """Create reasoning strategy from configuration.

    Delegates to the strategy :class:`~dataknobs_common.registry.PluginRegistry`
    singleton.  Built-in
    strategies (simple, react, wizard, grounded, hybrid) are registered
    automatically; 3rd-party strategies can be added via
    :func:`register_strategy`.

    See each strategy class's ``from_config()`` for available config
    keys (e.g. ``ReActReasoning.from_config``,
    ``WizardReasoning.from_config``).

    Args:
        config: Reasoning configuration dict.  The ``strategy`` key
            selects the strategy type (default ``"simple"``).  All
            other keys are forwarded to the strategy's
            ``from_config()`` classmethod.
        knowledge_base: Optional knowledge base instance forwarded
            as a kwarg to the strategy factory.

    Returns:
        Configured reasoning strategy instance.

    Raises:
        NotFoundError: If strategy type is not registered.
        OperationError: If strategy factory raises an exception.

    Example:
        ```python
        # Simple reasoning
        config = {"strategy": "simple"}
        strategy = create_reasoning_from_config(config)

        # Grounded reasoning (deterministic KB retrieval)
        config = {
            "strategy": "grounded",
            "intent": {"mode": "extract", "num_queries": 3},
            "retrieval": {"top_k": 5},
        }
        strategy = create_reasoning_from_config(config, knowledge_base=kb)
        ```
    """
    return get_registry().create(
        config=config,
        knowledge_base=knowledge_base,
        prompt_resolver=prompt_resolver,
    )

"""Tool system for LLM function calling.

This module provides abstractions for creating and managing tools that can be
called by LLMs during generation.

Classes:
    Tool: Base class for LLM-callable tools
    ContextAwareTool: Base class for tools that need execution context
    ContextEnhancedTool: Wrapper to add context awareness to existing tools
    ToolRegistry: Registry for managing available tools
    ToolExecutionContext: Context passed to context-aware tools
    ToolWizardState: The wizard state a tool is allowed to see
    WizardStateSnapshot: Deprecated alias for ToolWizardState; warns on use

Observability:
    ToolExecutionRecord: Record of a single tool execution
    ExecutionHistoryQuery: Query parameters for filtering history
    ExecutionStats: Aggregated statistics for tool executions
    ExecutionTracker: Standalone tracker for tool executions
"""

import warnings
from typing import TYPE_CHECKING

from dataknobs_llm.tools.base import Tool
from dataknobs_llm.tools.context import (
    ToolExecutionContext,
    ToolWizardState,
)
from dataknobs_llm.tools.context_aware import (
    ContextAwareTool,
    ContextEnhancedTool,
    default_wizard_data_injector,
)
from dataknobs_llm.tools.observability import (
    ExecutionHistoryQuery,
    ExecutionStats,
    ExecutionTracker,
    ToolExecutionRecord,
)
from dataknobs_llm.tools.registry import ToolRegistry

if TYPE_CHECKING:
    # Resolved at runtime by ``__getattr__`` below, which warns.
    from dataknobs_llm.tools.context import WizardStateSnapshot

__all__ = [
    # Core tool classes
    "Tool",
    "ToolRegistry",
    "ContextAwareTool",
    "ContextEnhancedTool",
    "ToolExecutionContext",
    "ToolWizardState",
    # Deprecated alias for ToolWizardState since 0.8.0; warns on access,
    # and is removed at 1.0.0.
    "WizardStateSnapshot",
    "default_wizard_data_injector",
    # Observability
    "ToolExecutionRecord",
    "ExecutionHistoryQuery",
    "ExecutionStats",
    "ExecutionTracker",
]


def __getattr__(name: str) -> type:
    """Resolve the deprecated ``WizardStateSnapshot`` name, warning on access.

    The name is re-exported from this package for import-site stability,
    but it is temporary rather than permanent, so it warns here as well as
    on :mod:`dataknobs_llm.tools.context`. Importing it eagerly above
    would fire that warning on every import of this package and name our
    own file rather than the caller's.
    """
    if name == "WizardStateSnapshot":
        warnings.warn(
            "WizardStateSnapshot is deprecated since 0.8.0; use "
            "ToolWizardState instead. The alias resolves until 1.0.0, when "
            "it is removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ToolWizardState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

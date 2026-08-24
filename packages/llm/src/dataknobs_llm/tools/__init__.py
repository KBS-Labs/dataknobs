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
    WizardStateSnapshot: Deprecated alias for ToolWizardState

Observability:
    ToolExecutionRecord: Record of a single tool execution
    ExecutionHistoryQuery: Query parameters for filtering history
    ExecutionStats: Aggregated statistics for tool executions
    ExecutionTracker: Standalone tracker for tool executions
"""

from dataknobs_llm.tools.base import Tool
from dataknobs_llm.tools.context import (
    ToolExecutionContext,
    ToolWizardState,
    WizardStateSnapshot,
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

__all__ = [
    # Core tool classes
    "Tool",
    "ToolRegistry",
    "ContextAwareTool",
    "ContextEnhancedTool",
    "ToolExecutionContext",
    "ToolWizardState",
    # Deprecated alias for ToolWizardState; removed after one minor version.
    "WizardStateSnapshot",
    "default_wizard_data_injector",
    # Observability
    "ToolExecutionRecord",
    "ExecutionHistoryQuery",
    "ExecutionStats",
    "ExecutionTracker",
]

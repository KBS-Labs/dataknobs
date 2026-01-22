"""Tool system for LLM function calling.

This module provides abstractions for creating and managing tools that can be
called by LLMs during generation.

Classes:
    Tool: Base class for LLM-callable tools
    ContextAwareTool: Base class for tools that need execution context
    ContextEnhancedTool: Wrapper to add context awareness to existing tools
    ToolRegistry: Registry for managing available tools
    ToolExecutionContext: Context passed to context-aware tools
    WizardStateSnapshot: Snapshot of wizard state for tool context
"""

from dataknobs_llm.tools.base import Tool
from dataknobs_llm.tools.context import ToolExecutionContext, WizardStateSnapshot
from dataknobs_llm.tools.context_aware import (
    ContextAwareTool,
    ContextEnhancedTool,
    default_wizard_data_injector,
)
from dataknobs_llm.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolRegistry",
    "ContextAwareTool",
    "ContextEnhancedTool",
    "ToolExecutionContext",
    "WizardStateSnapshot",
    "default_wizard_data_injector",
]

"""DataKnobs Finite State Machine framework.

A flexible FSM framework with data modes, resource management, and streaming support
for building complex data processing workflows.
"""

__version__ = "0.1.4"

# Core FSM components
from .core.fsm import FSM
from .core.state import StateDefinition, StateInstance
from .core.arc import ArcDefinition
from .core.data_modes import DataHandlingMode

# Simple API
from .api.simple import SimpleFSM
from .api.async_simple import AsyncSimpleFSM

# Advanced API
from .api.advanced import (
    AdvancedFSM,
    ExecutionMode,
    ExecutionHook,
    StepResult,
    FSMDebugger,
    create_advanced_fsm
)

# Execution context
from .execution.context import ExecutionContext

# Configuration
from .config.loader import ConfigLoader
from .config.builder import FSMBuilder

__all__ = [
    "__version__",
    # Core
    "FSM",
    "StateDefinition",
    "StateInstance",
    "ArcDefinition",
    "DataHandlingMode",
    # Simple API
    "SimpleFSM",
    "AsyncSimpleFSM",
    # Advanced API
    "AdvancedFSM",
    "ExecutionMode",
    "ExecutionHook",
    "StepResult",
    "FSMDebugger",
    "create_advanced_fsm",
    # Execution
    "ExecutionContext",
    # Config
    "ConfigLoader",
    "FSMBuilder",
]

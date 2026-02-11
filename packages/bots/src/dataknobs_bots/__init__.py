"""DataKnobs Bots - Configuration-driven AI agents."""

from .bot import BotContext, BotManager, BotRegistry, DynaBot
from .config import (
    ConfigDraftManager,
    ConfigTemplate,
    ConfigTemplateRegistry,
    ConfigValidator,
    DraftMetadata,
    DynaBotConfigBuilder,
    DynaBotConfigSchema,
    TemplateVariable,
    ValidationResult,
)
from .knowledge import RAGKnowledgeBase, create_knowledge_base_from_config
from .memory import BufferMemory, Memory, VectorMemory, create_memory_from_config
from .middleware import CostTrackingMiddleware, LoggingMiddleware, Middleware
from .reasoning import (
    ReActReasoning,
    ReasoningStrategy,
    SimpleReasoning,
    create_reasoning_from_config,
)
from .tools import (
    GetTemplateDetailsTool,
    KnowledgeSearchTool,
    ListTemplatesTool,
    PreviewConfigTool,
    SaveConfigTool,
    ValidateConfigTool,
)

__version__ = "0.4.3"

__all__ = [
    # Bot
    "DynaBot",
    "BotContext",
    "BotManager",
    "BotRegistry",
    # Memory
    "Memory",
    "BufferMemory",
    "VectorMemory",
    "create_memory_from_config",
    # Knowledge
    "RAGKnowledgeBase",
    "create_knowledge_base_from_config",
    # Tools
    "KnowledgeSearchTool",
    # Config Toolkit
    "DynaBotConfigSchema",
    "ConfigValidator",
    "ValidationResult",
    "DynaBotConfigBuilder",
    "ConfigTemplate",
    "TemplateVariable",
    "ConfigTemplateRegistry",
    "ConfigDraftManager",
    "DraftMetadata",
    # Config Tools
    "ListTemplatesTool",
    "GetTemplateDetailsTool",
    "PreviewConfigTool",
    "ValidateConfigTool",
    "SaveConfigTool",
    # Reasoning
    "ReasoningStrategy",
    "SimpleReasoning",
    "ReActReasoning",
    "create_reasoning_from_config",
    # Middleware
    "Middleware",
    "CostTrackingMiddleware",
    "LoggingMiddleware",
]

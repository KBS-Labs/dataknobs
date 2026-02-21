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
    ToolCatalog,
    ToolEntry,
    ValidationResult,
    create_default_catalog,
    default_catalog,
)
from .knowledge import RAGKnowledgeBase, create_knowledge_base_from_config
from .memory import BufferMemory, Memory, SummaryMemory, VectorMemory, create_memory_from_config
from .middleware import CostTrackingMiddleware, LoggingMiddleware, Middleware
from .reasoning import (
    ReActReasoning,
    ReasoningStrategy,
    SimpleReasoning,
    create_reasoning_from_config,
)
from .tools import (
    AddKBResourceTool,
    CheckKnowledgeSourceTool,
    GetTemplateDetailsTool,
    IngestKnowledgeBaseTool,
    KnowledgeSearchTool,
    ListAvailableToolsTool,
    ListKBResourcesTool,
    ListTemplatesTool,
    PreviewConfigTool,
    RemoveKBResourceTool,
    SaveConfigTool,
    ValidateConfigTool,
)

__version__ = "0.4.8"

__all__ = [
    # Bot
    "DynaBot",
    "BotContext",
    "BotManager",
    "BotRegistry",
    # Memory
    "Memory",
    "BufferMemory",
    "SummaryMemory",
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
    # Tool catalog
    "ToolCatalog",
    "ToolEntry",
    "create_default_catalog",
    "default_catalog",
    # Config Tools
    "ListTemplatesTool",
    "GetTemplateDetailsTool",
    "PreviewConfigTool",
    "ValidateConfigTool",
    "SaveConfigTool",
    "ListAvailableToolsTool",
    # KB Tools
    "CheckKnowledgeSourceTool",
    "ListKBResourcesTool",
    "AddKBResourceTool",
    "RemoveKBResourceTool",
    "IngestKnowledgeBaseTool",
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

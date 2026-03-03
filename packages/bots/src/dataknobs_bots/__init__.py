"""DataKnobs Bots - Configuration-driven AI agents."""

from .bot import BotContext, BotManager, BotRegistry, DynaBot, normalize_wizard_state
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
from .testing import CaptureReplay, inject_providers
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

__version__ = "0.6.0"

__all__ = [
    "AddKBResourceTool",
    "BotContext",
    "BotManager",
    "BotRegistry",
    "BufferMemory",
    "CaptureReplay",
    "CheckKnowledgeSourceTool",
    "ConfigDraftManager",
    "ConfigTemplate",
    "ConfigTemplateRegistry",
    "ConfigValidator",
    "CostTrackingMiddleware",
    "DraftMetadata",
    "DynaBot",
    "DynaBotConfigBuilder",
    "DynaBotConfigSchema",
    "GetTemplateDetailsTool",
    "IngestKnowledgeBaseTool",
    "KnowledgeSearchTool",
    "ListAvailableToolsTool",
    "ListKBResourcesTool",
    "ListTemplatesTool",
    "LoggingMiddleware",
    "Memory",
    "Middleware",
    "PreviewConfigTool",
    "RAGKnowledgeBase",
    "ReActReasoning",
    "ReasoningStrategy",
    "RemoveKBResourceTool",
    "SaveConfigTool",
    "SimpleReasoning",
    "SummaryMemory",
    "TemplateVariable",
    "ToolCatalog",
    "ToolEntry",
    "ValidationResult",
    "VectorMemory",
    "create_default_catalog",
    "create_knowledge_base_from_config",
    "create_memory_from_config",
    "create_reasoning_from_config",
    "default_catalog",
    "inject_providers",
    "normalize_wizard_state",
]

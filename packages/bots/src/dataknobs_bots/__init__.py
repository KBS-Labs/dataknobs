"""DataKnobs Bots - Configuration-driven AI agents."""

from .bot import BotContext, BotManager, BotRegistry, DynaBot
from .knowledge import RAGKnowledgeBase, create_knowledge_base_from_config
from .memory import BufferMemory, Memory, VectorMemory, create_memory_from_config
from .middleware import CostTrackingMiddleware, LoggingMiddleware, Middleware
from .reasoning import (
    ReActReasoning,
    ReasoningStrategy,
    SimpleReasoning,
    create_reasoning_from_config,
)
from .tools import KnowledgeSearchTool

__version__ = "0.1.0"

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

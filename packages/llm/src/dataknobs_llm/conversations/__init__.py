"""Conversation management system for dataknobs-llm.

This module provides tools for managing multi-turn conversations with:
- Tree-based message history supporting branching
- Persistent storage across sessions
- Middleware for request/response processing

Main Components:
- ConversationNode: Data stored in each tree node
- ConversationState: Tree-based conversation state
- ConversationStorage: Abstract storage interface
- DataknobsConversationStorage: Storage adapter for dataknobs backends
- ConversationManager: Orchestrates multi-turn conversations (coming soon)
- ConversationMiddleware: Base class for middleware (coming soon)

Example:
    >>> from dataknobs_llm.conversations import (
    ...     ConversationState,
    ...     ConversationNode,
    ...     DataknobsConversationStorage
    ... )
    >>> from dataknobs_llm.llm.base import LLMMessage
    >>> from dataknobs_structures.tree import Tree
    >>> from dataknobs_data.backends import AsyncMemoryDatabase
    >>>
    >>> # Create conversation with system message
    >>> root_node = ConversationNode(
    ...     message=LLMMessage(role="system", content="You are helpful"),
    ...     node_id=""
    ... )
    >>> tree = Tree(root_node)
    >>> state = ConversationState(
    ...     conversation_id="conv-123",
    ...     message_tree=tree,
    ...     current_node_id=""
    ... )
    >>>
    >>> # Store conversation
    >>> storage = DataknobsConversationStorage(AsyncMemoryDatabase())
    >>> await storage.save_conversation(state)
"""

from dataknobs_llm.conversations.storage import (
    ConversationNode,
    ConversationState,
    ConversationStorage,
    DataknobsConversationStorage,
    StorageError,
    SchemaVersionError,
    SCHEMA_VERSION,
    calculate_node_id,
    get_node_by_id,
    get_messages_for_llm,
)
from dataknobs_llm.conversations.manager import ConversationManager
from dataknobs_llm.conversations.middleware import (
    ConversationMiddleware,
    LoggingMiddleware,
    ContentFilterMiddleware,
    ValidationMiddleware,
    MetadataMiddleware,
    RateLimitMiddleware,
    RateLimitError,
)

__all__ = [
    # Data structures
    "ConversationNode",
    "ConversationState",

    # Storage interfaces
    "ConversationStorage",
    "DataknobsConversationStorage",
    "StorageError",
    "SchemaVersionError",
    "SCHEMA_VERSION",

    # Manager
    "ConversationManager",

    # Middleware
    "ConversationMiddleware",
    "LoggingMiddleware",
    "ContentFilterMiddleware",
    "ValidationMiddleware",
    "MetadataMiddleware",
    "RateLimitMiddleware",
    "RateLimitError",

    # Helper functions
    "calculate_node_id",
    "get_node_by_id",
    "get_messages_for_llm",
]

"""Conversation storage with tree-based branching support.

This module provides:
- ConversationNode: Data stored in each tree node
- ConversationState: Tree-based conversation state
- ConversationStorage: Abstract storage interface
- DataknobsConversationStorage: Storage adapter for dataknobs backends
- Helper functions for node ID management and tree navigation

Schema Versioning:
    The storage format uses semantic versioning (MAJOR.MINOR.PATCH):
    - MAJOR: Incompatible changes requiring migration
    - MINOR: Backward-compatible additions
    - PATCH: Bug fixes, no schema changes

    Current schema version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import logging

from dataknobs_structures.tree import Tree
from dataknobs_llm.llm.base import LLMMessage

# Current schema version - increment when making schema changes
SCHEMA_VERSION = "1.0.0"

logger = logging.getLogger(__name__)


@dataclass
class ConversationNode:
    """Data stored in each conversation tree node.

    Each node represents a single message (system, user, or assistant) in the
    conversation. The tree structure allows for branching conversations where
    multiple alternative messages can be explored.

    Attributes:
        message: The LLM message (role + content)
        node_id: Dot-delimited child positions from root (e.g., "0.1.2")
        timestamp: When this message was created
        prompt_name: Optional name of prompt template used to generate this
        branch_name: Optional human-readable label for this branch
        metadata: Additional metadata (usage stats, model info, etc.)

    Example:
        >>> node = ConversationNode(
        ...     message=LLMMessage(role="user", content="Hello"),
        ...     node_id="0.1",
        ...     timestamp=datetime.now(),
        ...     prompt_name="greeting",
        ...     branch_name="polite-variant"
        ... )
    """
    message: LLMMessage
    node_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    prompt_name: str | None = None
    branch_name: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage."""
        return {
            "message": {
                "role": self.message.role,
                "content": self.message.content,
                "name": self.message.name,
                "metadata": self.message.metadata or {}
            },
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "prompt_name": self.prompt_name,
            "branch_name": self.branch_name,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationNode":
        """Create node from dictionary."""
        msg_data = data["message"]
        return cls(
            message=LLMMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                name=msg_data.get("name"),
                metadata=msg_data.get("metadata", {})
            ),
            node_id=data["node_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            prompt_name=data.get("prompt_name"),
            branch_name=data.get("branch_name"),
            metadata=data.get("metadata", {})
        )


def calculate_node_id(node: Tree) -> str:
    """Calculate dot-delimited node ID by walking up to root.

    The node ID represents the path from root to this node as a series of
    child indexes. For example, "0.1.2" means: root's child 0, then that
    node's child 1, then that node's child 2.

    Args:
        node: Tree node to calculate ID for

    Returns:
        Dot-delimited node ID (e.g., "0", "0.1", "0.1.2")

    Example:
        >>> root = Tree(data)
        >>> child = root.add_child(data2)
        >>> grandchild = child.add_child(data3)
        >>> calculate_node_id(grandchild)
        '0.0'
    """
    if node.parent is None:
        # Root node has no parent, so it's just "0" or we could use ""
        # Let's use "" for root to make child IDs cleaner
        return ""

    # Walk up to root, collecting child indexes
    indexes = []
    current = node
    while current.parent is not None:
        indexes.append(str(current.sibnum))
        current = current.parent

    # Reverse to get root-to-node order
    indexes.reverse()

    return ".".join(indexes) if indexes else "0"


def get_node_by_id(tree: Tree, node_id: str) -> Tree | None:
    """Retrieve tree node by its dot-delimited ID.

    Args:
        tree: Root of the tree
        node_id: Dot-delimited node ID (e.g., "0.1.2")

    Returns:
        Tree node with that ID, or None if not found

    Example:
        >>> node = get_node_by_id(tree, "0.1.2")
        >>> # Equivalent to: tree.children[0].children[1].children[2]
    """
    if not node_id or node_id == "":
        return tree  # Root node

    # Split into child indexes
    try:
        indexes = [int(i) for i in node_id.split(".")]
    except ValueError:
        return None  # Invalid node_id format

    # Navigate down the tree
    current = tree
    for idx in indexes:
        if not current.children or idx >= len(current.children):
            return None  # Invalid path
        current = current.children[idx]

    return current


def get_messages_for_llm(tree: Tree, node_id: str) -> List[LLMMessage]:
    """Get linear message sequence from root to specified node.

    This is what gets sent to the LLM - the path through the tree from
    root to current position.

    Args:
        tree: Root of conversation tree
        node_id: ID of current position

    Returns:
        List of messages from root to current node

    Example:
        >>> messages = get_messages_for_llm(tree, "0.1.2")
        >>> # Returns: [root_msg, child_0_msg, child_1_msg, child_2_msg]
    """
    node = get_node_by_id(tree, node_id)
    if node is None:
        return []

    # Get path from root to node
    path = node.get_path()

    # Extract messages from each node's data
    messages = []
    for tree_node in path:
        if isinstance(tree_node.data, ConversationNode):
            messages.append(tree_node.data.message)

    return messages


@dataclass
class ConversationState:
    """State of a conversation with tree-based branching support.

    This replaces the linear message history with a tree structure that
    supports multiple branches (alternative conversation paths).

    Attributes:
        conversation_id: Unique conversation identifier
        message_tree: Root of conversation tree (Tree[ConversationNode])
        current_node_id: ID of current position in tree (dot-delimited)
        metadata: Additional conversation metadata
        created_at: Conversation creation timestamp
        updated_at: Last update timestamp
        schema_version: Version of the storage schema used

    Example:
        >>> # Create conversation with system message
        >>> root_node = ConversationNode(
        ...     message=LLMMessage(role="system", content="You are helpful"),
        ...     node_id=""
        ... )
        >>> tree = Tree(root_node)
        >>> state = ConversationState(
        ...     conversation_id="conv-123",
        ...     message_tree=tree,
        ...     current_node_id="",
        ...     metadata={"user_id": "alice"}
        ... )
        >>>
        >>> # Add user message
        >>> user_node = ConversationNode(
        ...     message=LLMMessage(role="user", content="Hello"),
        ...     node_id="0"
        ... )
        >>> tree.add_child(Tree(user_node))
        >>> state.current_node_id = "0"
    """
    conversation_id: str
    message_tree: Tree  # Tree[ConversationNode]
    current_node_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    schema_version: str = SCHEMA_VERSION

    def get_current_node(self) -> Tree | None:
        """Get the current tree node."""
        return get_node_by_id(self.message_tree, self.current_node_id)

    def get_current_messages(self) -> List[LLMMessage]:
        """Get messages from root to current position (for LLM)."""
        return get_messages_for_llm(self.message_tree, self.current_node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for storage.

        The tree is serialized as a list of edges (parent_id, child_id, node_data).
        Includes schema_version for backward compatibility.
        """
        # Collect all nodes and their data
        nodes = []
        edges = []

        all_nodes = self.message_tree.find_nodes(lambda n: True, traversal="bfs")
        for tree_node in all_nodes:
            if isinstance(tree_node.data, ConversationNode):
                nodes.append(tree_node.data.to_dict())

                # Add edge to parent (if not root)
                if tree_node.parent is not None:
                    parent_id = calculate_node_id(tree_node.parent)
                    child_id = tree_node.data.node_id
                    edges.append([parent_id, child_id])

        return {
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "nodes": nodes,
            "edges": edges,
            "current_node_id": self.current_node_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create state from dictionary.

        Reconstructs the tree from nodes and edges.
        Handles schema version migration if needed.
        """
        # Check schema version
        stored_version = data.get("schema_version", "0.0.0")  # Default to 0.0.0 if missing

        # Apply migrations if needed
        if stored_version != SCHEMA_VERSION:
            logger.info(
                f"Migrating conversation {data['conversation_id']} "
                f"from schema {stored_version} to {SCHEMA_VERSION}"
            )
            data = cls._migrate_schema(data, stored_version, SCHEMA_VERSION)

        # Create nodes indexed by ID
        nodes_by_id: Dict[str, ConversationNode] = {}
        for node_data in data["nodes"]:
            node = ConversationNode.from_dict(node_data)
            nodes_by_id[node.node_id] = node

        # Find root (node with empty ID)
        root_node = nodes_by_id.get("")
        if root_node is None:
            # Try to find node with no parent in edges
            child_ids = {edge[1] for edge in data["edges"]}
            parent_ids = {edge[0] for edge in data["edges"]}
            root_ids = parent_ids - child_ids
            if root_ids:
                root_node = nodes_by_id[root_ids.pop()]
            else:
                # Fallback: first node
                root_node = list(nodes_by_id.values())[0]

        tree = Tree(root_node)
        tree_nodes_by_id = {"": tree}  # Map node_id -> Tree node

        # Build tree by adding edges
        for parent_id, child_id in data["edges"]:
            if parent_id in tree_nodes_by_id:
                parent_tree_node = tree_nodes_by_id[parent_id]
                child_node = nodes_by_id[child_id]
                child_tree_node = parent_tree_node.add_child(Tree(child_node))
                tree_nodes_by_id[child_id] = child_tree_node

        return cls(
            conversation_id=data["conversation_id"],
            message_tree=tree,
            current_node_id=data["current_node_id"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            schema_version=SCHEMA_VERSION  # Always use current version after migration
        )

    @staticmethod
    def _migrate_schema(
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Migrate data from one schema version to another.

        This method applies migrations sequentially to transform data from
        an older schema version to the current version.

        Args:
            data: Data in old schema format
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            Data in new schema format

        Raises:
            SchemaVersionError: If migration path is not supported
        """
        # Parse version strings
        from_major, from_minor, from_patch = map(int, from_version.split("."))
        to_major, to_minor, to_patch = map(int, to_version.split("."))

        # No migration needed if versions match
        if from_version == to_version:
            return data

        # Apply migrations based on version transitions
        # Future migrations will be added here as needed

        # Example migration patterns:
        # if from_version == "0.0.0" and to_version >= "1.0.0":
        #     data = cls._migrate_0_to_1(data)
        # if from_version < "1.1.0" and to_version >= "1.1.0":
        #     data = cls._migrate_1_0_to_1_1(data)

        # For now, version 0.0.0 (no version field) to 1.0.0 is a no-op
        # because the schema didn't change, we just added versioning
        if from_version == "0.0.0":
            logger.debug("Migrating from unversioned schema to 1.0.0 (no changes needed)")
            data["schema_version"] = "1.0.0"
            return data

        # If we get here and versions still don't match, it's unsupported
        if from_major > to_major:
            raise SchemaVersionError(
                f"Cannot downgrade from schema {from_version} to {to_version}"
            )

        logger.warning(
            f"No migration path defined from {from_version} to {to_version}. "
            "Using data as-is."
        )
        data["schema_version"] = to_version
        return data

    # Future migration methods will be added here as needed:
    # @staticmethod
    # def _migrate_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
    #     """Migrate from schema 1.0 to 1.1."""
    #     # Add new field with default value
    #     data["new_field"] = "default_value"
    #     return data


class ConversationStorage(ABC):
    """Abstract storage interface for conversations.

    This interface defines the contract for persisting conversation state.
    Implementations can use any backend (SQL, NoSQL, file, etc.).
    """

    @abstractmethod
    async def save_conversation(self, state: ConversationState) -> None:
        """Save conversation state (upsert).

        Args:
            state: Conversation state to save
        """
        pass

    @abstractmethod
    async def load_conversation(
        self,
        conversation_id: str
    ) -> ConversationState | None:
        """Load conversation state.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation state or None if not found
        """
        pass

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_conversations(
        self,
        filter_metadata: Dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ConversationState]:
        """List conversations with optional filtering.

        Args:
            filter_metadata: Optional metadata filters
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of conversation states
        """
        pass


class DataknobsConversationStorage(ConversationStorage):
    """Conversation storage using dataknobs_data backends.

    Stores conversations as Records with the tree serialized as nodes + edges.
    Works with any dataknobs backend (Memory, File, S3, Postgres, etc.).

    Example:
        >>> from dataknobs_data.backends import AsyncMemoryDatabase
        >>> storage = DataknobsConversationStorage(AsyncMemoryDatabase())
        >>> await storage.save_conversation(state)
    """

    def __init__(self, backend: Any):
        """Initialize storage with dataknobs backend.

        Args:
            backend: Dataknobs async database backend (AsyncMemoryDatabase, etc.)
        """
        self.backend = backend

    def _state_to_record(self, state: ConversationState) -> Any:
        """Convert ConversationState to Record.

        Args:
            state: Conversation state to convert

        Returns:
            Record object for storage
        """
        # Import here to avoid circular dependency
        try:
            from dataknobs_data.records import Record
        except ImportError:
            raise StorageError(
                "dataknobs_data package not available. "
                "Install it to use DataknobsConversationStorage."
            )

        # Convert state to dict
        data = state.to_dict()

        # Create Record with conversation_id as storage_id
        return Record(
            data=data,
            storage_id=state.conversation_id
        )

    def _record_to_state(self, record: Any) -> ConversationState:
        """Convert Record to ConversationState.

        Args:
            record: Record object from storage

        Returns:
            Conversation state
        """
        # Extract data from record
        data = {}
        for field_name, field in record.fields.items():
            data[field_name] = field.value

        # Reconstruct conversation state
        return ConversationState.from_dict(data)

    async def save_conversation(self, state: ConversationState) -> None:
        """Save conversation to backend."""
        try:
            record = self._state_to_record(state)
            # Use upsert to insert or update
            await self.backend.upsert(state.conversation_id, record)
        except Exception as e:
            raise StorageError(f"Failed to save conversation: {e}") from e

    async def load_conversation(
        self,
        conversation_id: str
    ) -> ConversationState | None:
        """Load conversation from backend."""
        try:
            # Read record by ID
            record = await self.backend.read(conversation_id)
            if record is None:
                return None

            return self._record_to_state(record)

        except Exception as e:
            raise StorageError(f"Failed to load conversation: {e}") from e

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation from backend."""
        try:
            return await self.backend.delete(conversation_id)
        except Exception as e:
            raise StorageError(f"Failed to delete conversation: {e}") from e

    async def list_conversations(
        self,
        filter_metadata: Dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ConversationState]:
        """List conversations from backend."""
        try:
            # Import Query for filtering
            try:
                from dataknobs_data.query import Query
            except ImportError:
                raise StorageError(
                    "dataknobs_data package not available. "
                    "Install it to use DataknobsConversationStorage."
                )

            # Build query with metadata filters using fluent interface
            query = Query()
            query.limit(limit).offset(offset)

            if filter_metadata:
                for key, value in filter_metadata.items():
                    # Add filter for metadata.key = value
                    query.filter(f"metadata.{key}", "=", value)

            # Search with query
            results = await self.backend.search(query)

            # Convert records to conversation states
            return [self._record_to_state(record) for record in results]

        except Exception as e:
            raise StorageError(f"Failed to list conversations: {e}") from e


class StorageError(Exception):
    """Exception raised for storage operation errors."""
    pass


class SchemaVersionError(Exception):
    """Exception raised for schema version incompatibilities."""
    pass

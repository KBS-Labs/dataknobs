"""Utility functions for working with Large Language Model outputs.

Provides functions for LLM-related operations including response parsing,
prompt formatting, and structured output extraction.
"""

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from dataknobs_structures.tree import Tree


def get_value_by_key(
    d: Dict[str, Any] | None,
    pathkey: str,
    default_value: Any = None,
) -> Any:
    """Get a nested value from a dictionary using dot-delimited path.

    Navigates through nested dictionaries using a path key like "foo.bar.baz"
    to retrieve deeply nested values.

    Args:
        d: Possibly nested dictionary to search.
        pathkey: Dot-delimited path to the value (e.g., "foo.bar").
        default_value: Value to return if path doesn't exist. Defaults to None.

    Returns:
        Any: Value at the path, or default_value if path doesn't exist.

    Examples:
        >>> d = {"foo": {"bar": "baz"}}
        >>> get_value_by_key(d, "foo.bar")
        'baz'
    """
    path = pathkey.split(".")
    if d is None:
        return default_value

    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default_value
        d = d[key]

    return d


class PromptMessage:
    """Structured prompt message with role, content, and optional metadata.

    Represents a single message in an LLM conversation with format:
        {"role": <role>, "content": <content>}

    Where role is typically "system", "user", or "assistant" and content
    contains the prompt or LLM-generated text.

    Optional metadata can include:
        {
          "generation_args": {...},
          "execution_data": {"model_name": ..., "starttime": ..., "endtime": ...},
          "user_comments": [{"user": ..., "comment": ...}, ...],
        }

    Attributes:
        role: Message role (e.g., "system", "user", "assistant").
        content: Message text content.
        metadata: Optional metadata dictionary.
    """

    def __init__(self, role: str, content: str, metadata: Dict[str, Any] | None = None):
        """Initialize prompt message with role and content.

        Args:
            role: Message role (e.g., "system", "user", "assistant").
            content: Prompt or LLM-generated text.
            metadata: Optional metadata containing generation args, execution
                data, and user comments. Defaults to None.
        """
        self.role = role
        self.content = content
        self.metadata = metadata
        self._dict: Dict[str, str] | None = None  # The dict without metadata

    def __repr__(self) -> str:
        """Get message as JSON string without metadata.

        Returns:
            str: JSON string representation of message.
        """
        return self.to_json(with_metadata=False)

    def to_json(self, with_metadata: bool = True) -> str:
        """Serialize message to JSON string.

        Args:
            with_metadata: If True, includes metadata. Defaults to True.

        Returns:
            str: JSON string representation of message.

        Raises:
            TypeError: If metadata is not JSON serializable.
        """
        return json.dumps(self.get_message(with_metadata=with_metadata))

    def get_message(self, with_metadata: Union[bool, str] = False) -> Dict[str, Any]:
        """Get message as a dictionary.

        Args:
            with_metadata: If True, includes metadata. If a string, uses that
                as the metadata key instead of "metadata". Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary with "role" and "content", optionally
                including metadata.
        """
        if self._dict is None:
            self._dict = {
                "role": self.role,
                "content": self.content,
            }
        retval: Dict[str, Any]
        if with_metadata and self.metadata is not None:
            retval = dict(self._dict)  # Convert to Dict[str, Any]
            attr = with_metadata if isinstance(with_metadata, str) else "metadata"
            retval[attr] = self.metadata
        else:
            retval = dict(self._dict)
        return retval

    @staticmethod
    def build_instance(message_dict: Dict[str, Any]) -> "PromptMessage":
        """Reconstruct a PromptMessage from its dictionary representation.

        Args:
            message_dict: Dictionary with "role", "content", and optionally
                "metadata" keys.

        Returns:
            PromptMessage: Reconstructed message instance.
        """
        return PromptMessage(
            message_dict.get("role", "unknown"),
            message_dict.get("content", ""),
            metadata=message_dict.get("metadata"),
        )


class PromptTree:
    """Tree structure for managing branching LLM conversation histories.

    Each instance represents a node in a tree where the root holds the initial
    prompt and descendants hold follow-on prompts. The tree structure allows
    exploring different conversation branches from any point.

    Tree navigation uses the "node" property. The PromptTree instance can be
    retrieved from a node via node.data.

    Metadata is inherited hierarchically - descendants can override ancestor
    metadata or access inherited values.

    Attributes:
        message: The PromptMessage for this node.
        node: Tree structure node containing this PromptTree as data.
    """

    def __init__(
        self,
        message: PromptMessage | None = None,
        role: str | None = None,
        content: str | None = None,
        metadata: Dict[str, Any] | None = None,
        parent: Optional["PromptTree"] = None,
    ):
        """Initialize a prompt tree node.

        Can construct from an existing message or create a new one. If both
        a message and override parameters are provided, creates a new message
        with overridden values.

        Args:
            message: Existing message or template to use. Defaults to None.
            role: Role override or role for new message. Defaults to None.
            content: Content override or content for new message. Defaults to None.
            metadata: Metadata override or metadata for new message. Defaults to None.
            parent: Parent PromptTree node. If None, creates a root node.
                Defaults to None.
        """
        # NOTE: _tree_id is the ID factory global to this full tree
        self._tree_id: PromptTree.IdTracker = (
            self.IdTracker() if parent is None else parent._tree_id
        )
        self._node_id = self._tree_id.next_id()
        self.node: Tree | None = None
        the_message: PromptMessage | None = message
        # If there's an override AND a message
        if (
            role is not None or content is not None or metadata is not None
        ) and message is not None:
            the_message = None
            if role is None:
                # Keep the original message's role, else use the override
                role = message.role
            if content is None:
                # Keep the original message's content, else use the override
                content = message.content
            if metadata is None:
                # Keep the original message's metadata, else use the override
                metadata = message.metadata
        self.message = the_message or PromptMessage(
            role or "user", content or "", metadata=metadata
        )
        if parent is not None and parent.node is not None:
            self.node = parent.node.add_child(self)
        else:
            self.node = Tree(self)

    def __repr__(self) -> str:
        """Get string representation showing node ID, role, and content length.

        Returns:
            str: String in format "node_id:role(content_length)".
        """
        return f"{self.node_id}:{self.message.role}({len(self.message.content)})"

    def _inc_tree_id(self) -> int:
        """Store a global (to all nodes in this tree) ID as a value in a single
        element list. This shared value across all PromptTree instances will
        always hold the total number of nodes in the full tree.
        """
        return 0  # This appears to be an incomplete method

    @property
    def node_count(self) -> int:
        """Get total number of nodes in the entire tree.

        Returns:
            int: Total node count across all branches.
        """
        return self._tree_id.id_count

    @property
    def node_id(self) -> int:
        """Get this node's unique identifier.

        Returns:
            int: Node ID.
        """
        return self._node_id

    def add_message(
        self,
        message: PromptMessage | None = None,
        role: str | None = None,
        content: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> "PromptTree":
        """Add a child message to this node.

        Args:
            message: Child prompt message or template. Defaults to None.
            role: Role override or role for new message. Defaults to None.
            content: Content override or content for new message. Defaults to None.
            metadata: Metadata override or metadata for new message. Defaults to None.

        Returns:
            PromptTree: New child node.
        """
        return PromptTree(
            message=message, role=role, content=content, metadata=metadata, parent=self
        )

    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree.

        Returns:
            int: Depth (0 for root, increasing for descendants).
        """
        return self.node.depth if self.node is not None else 0

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata for this node, inheriting from ancestors if needed.

        Searches up the tree to find the nearest ancestor with metadata.

        Returns:
            Dict[str, Any]: Metadata dictionary (empty dict if none found).
        """
        metadata = self.message.metadata
        node = self.node
        while metadata is None and node is not None and node.parent is not None:
            node = node.parent
            if hasattr(node.data, "metadata"):
                metadata = node.data.metadata
        return metadata if metadata is not None else {}

    def get_metadata_value(self, pathkey: str, default_value: Any = None) -> Any:
        """Get a metadata value by path, inheriting from ancestors if needed.

        Searches up the tree to find the nearest ancestor with the requested
        metadata value.

        Args:
            pathkey: Dot-delimited path to metadata value (e.g., "execution_data.model_name").
            default_value: Value to return if path doesn't exist. Defaults to None.

        Returns:
            Any: Metadata value at path, or default_value if not found.
        """
        value = None
        node = self.node
        while value is None and node is not None:
            metadata = node.data.message.metadata
            if metadata is not None:
                value = get_value_by_key(metadata, pathkey, default_value=default_value)
            node = node.parent
        return value

    def apply(
        self,
        ptree_fn: Callable[["PromptTree"], None],
        level_offset: int = -1,
    ) -> None:
        """Apply a function to nodes from a starting level up to this node.

        Traverses from this node up to the specified starting node, calling
        the function on each node along the path.

        Args:
            ptree_fn: Function to apply to each PromptTree node, called in
                order from this node upward.
            level_offset: Starting level relative to this node:
                - 0, 1, 2, ... = current, parent, grandparent, etc.
                - -1, -2, ... = root, 1-under-root, etc.
                Defaults to -1 (start from root).
        """
        node = self.node  # start from this node
        level_node = self.get_level_node(level_offset)
        end_node = level_node.node if level_node is not None else None
        while True:
            if node is None:
                break
            ptree_fn(node.data)  # call the function
            if node == end_node:  # time to stop moving up
                break
            node = node.parent  # move up

    def get_messages(
        self, level_offset: int = -1, with_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Get conversation messages from a starting level to this node.

        Collects messages suitable for passing as context to an LLM, ordered
        from earliest to latest in the conversation path.

        Args:
            level_offset: Starting level relative to this node:
                - 0, 1, 2, ... = current, parent, grandparent, etc.
                - -1, -2, ... = root, 1-under-root, etc.
                Defaults to -1 (include all from root).
            with_metadata: If True, includes metadata in messages. Defaults to False.

        Returns:
            List[Dict[str, Any]]: Ordered list of message dictionaries.
        """
        node = self.node
        level_node = self.get_level_node(level_offset)
        end_node = level_node.node if level_node is not None else None
        messages: List[Dict[str, Any]] = []
        while True:
            if node is None:
                break
            messages.insert(0, node.data.message.get_message(with_metadata=with_metadata))
            if node == end_node:
                break
            node = node.parent
        return messages

    def get_duration(self, level_offset: int = 0) -> int:
        """Get total execution duration in seconds for a range of nodes.

        Sums execution durations from metadata["execution_data"]["endtime"] -
        metadata["execution_data"]["starttime"] for nodes from starting level
        to this node.

        Args:
            level_offset: Starting level relative to this node:
                - 0, 1, 2, ... = current, parent, grandparent, etc.
                - -1, -2, ... = root, 1-under-root, etc.
                Defaults to 0 (this node only).

        Returns:
            int: Total duration in seconds.
        """
        node = self.node
        level_node = self.get_level_node(level_offset)
        end_node = level_node.node if level_node is not None else None
        duration = 0
        while True:
            if node is None:
                break
            duration += node.data.get_self_duration()
            if node == end_node:
                break
            node = node.parent
        return duration

    def get_self_duration(self) -> int:
        """Get execution duration in seconds for this node only.

        Calculates duration from metadata["execution_data"] using ISO format
        timestamps (endtime - starttime).

        Returns:
            int: Duration in seconds, or 0 if execution data not available.
        """
        duration = 0
        metadata = self.message.metadata  # NOT self.metadata! don't inherit.
        if metadata is not None:
            exec_data = metadata.get("execution_data", None)
            if exec_data is not None:
                starttime = exec_data.get("starttime", None)
                endtime = exec_data.get("endtime", None)
                if starttime and endtime:
                    duration = int(
                        (
                            datetime.fromisoformat(endtime) - datetime.fromisoformat(starttime)
                        ).total_seconds()
                    )
        return duration

    def get_level_node(self, level_offset: int) -> Optional["PromptTree"]:
        """Get the node at a specified level relative to this node.

        Args:
            level_offset: Level relative to this node:
                - 0 = this node
                - 1, 2, ... = parent, grandparent, etc.
                - -1, -2, ... = root, 1-under-root, etc.

        Returns:
            PromptTree | None: Node at the specified level, or None if not found.
        """
        if level_offset == 0:
            # This is the node
            return self
        elif level_offset > 0:
            # The parent node, stopping at the root
            node = self.node
            if node is None:
                return None
            for _ in range(level_offset):
                if node.parent is None:
                    return node.data if hasattr(node, "data") else None
                else:
                    node = node.parent
            return node.data if node is not None and hasattr(node, "data") else None
        else:  # level_offset < 0
            # Offset from root, stopping at this node
            node = self.node
            if node is None:
                return None
            nodes: List[Tree] = []
            while node is not None:
                nodes.insert(0, node)
                node = node.parent
            selected_node = nodes[min(abs(level_offset + 1), len(nodes) - 1)]
            return selected_node.data if hasattr(selected_node, "data") else None

    def find_node_by_id(self, node_id: int) -> Optional["PromptTree"]:
        """Find a node in the tree by its unique ID.

        Searches the entire tree from the root for the specified node ID.

        Args:
            node_id: Unique node identifier to find.

        Returns:
            PromptTree | None: Node with matching ID, or None if not found.
        """
        if self.node is None:
            return None
        found = self.node.root.find_nodes(
            lambda node: hasattr(node.data, "node_id") and node.data.node_id == node_id,
            only_first=True,
        )
        return found[0].data if len(found) > 0 else None

    def find_nodes(
        self,
        match_fn: Callable[[Tree], bool] | None = None,
    ) -> List["PromptTree"]:
        """Find all nodes in the tree matching given criteria.

        Searches the entire tree from the root for nodes satisfying the
        match function.

        Args:
            match_fn: Function taking a Tree node (where tree.data is a
                PromptTree) and returning True to select it. If None, returns
                empty list. Defaults to None.

        Returns:
            List[PromptTree]: List of matching PromptTree nodes (may be empty).
        """
        if self.node is None:
            return []
        found = self.node.root.find_nodes(match_fn if match_fn is not None else lambda _: False)
        return [node.data for node in found]

    def serialize_tree(
        self, full: bool = False, with_metadata: bool = True
    ) -> Union[Dict[Any, Any], List[Any]]:
        """Serialize tree structure to nested lists for storage/transmission.

        Single nodes serialize to message dictionaries. Multi-node trees
        serialize to nested lists where each element represents a node and
        its children.

        Args:
            full: If True, serializes from root; if False, from this node.
                Defaults to False.
            with_metadata: If True, includes metadata in serialization.
                Defaults to True.

        Returns:
            Dict[Any, Any] | List[Any]: Serialized tree structure that can be
                reconstructed using build_instance().

        Note:
            The complete tree can be reconstructed from the serialized output.
        """
        if self.node is None:
            return {}
        return self._do_serialize(
            self.node.root if full else self.node, with_metadata=with_metadata
        )

    def _do_serialize(
        self,
        node: Tree,
        with_metadata: bool,
    ) -> Union[Dict[Any, Any], List[Any]]:
        """Recursively serialize tree nodes.

        Args:
            node: Current Tree node to serialize.
            with_metadata: If True, includes metadata in serialization.

        Returns:
            Dict[Any, Any] | List[Any]: Serialized node and children.
        """
        retval: Union[Dict[Any, Any], List[Any]] = node.data.message.get_message(
            with_metadata=with_metadata
        )
        if node.has_children():
            retval = [
                retval,
                [
                    self._do_serialize(
                        child,
                        with_metadata=with_metadata,
                    )
                    for child in (node.children if node.children is not None else [])
                ],
            ]
        return retval

    @staticmethod
    def build_instance(
        data: Union[Dict[str, Any], List[Any]], parent: Optional["PromptTree"] = None
    ) -> Optional["PromptTree"]:
        """Reconstruct a PromptTree from serialized data.

        Deserializes output from serialize_tree() back into a PromptTree structure.

        Args:
            data: Serialized tree data (from serialize_tree()).
            parent: Parent PromptTree to attach to. If None, creates a root node.
                Defaults to None.

        Returns:
            PromptTree | None: Last (deepest) node in the deserialized tree,
                suitable for continuing a conversation. Returns None if data
                is invalid.
        """
        pt = parent
        if isinstance(data, list) and len(data) > 0:
            # data[0] is a parent
            # data[1] is a list of children
            cur_parent = PromptTree.build_instance(data[0], parent=pt)
            for item in data[1]:
                pt = PromptTree.build_instance(item, parent=cur_parent)
        elif isinstance(data, dict):
            # data is a single (terminal) node
            pt = PromptTree(message=PromptMessage.build_instance(data), parent=parent)
        else:
            pt = None
        return pt

    class IdTracker:
        """Simple counter for generating unique node IDs.

        Maintains a shared counter across all nodes in a PromptTree to ensure
        each node has a unique identifier.
        """

        def __init__(self) -> None:
            self._next_id = 0

        def next_id(self) -> int:
            """Generate and return the next unique ID.

            Returns:
                int: Next available ID.
            """
            retval = self._next_id
            self._next_id += 1
            return retval

        @property
        def id_count(self) -> int:
            """Get the total number of IDs assigned.

            Returns:
                int: Count of IDs generated so far.
            """
            return self._next_id


class MessageCollector:
    """Example callable for use with PromptTree.apply() to collect messages.

    Demonstrates how to use PromptTree.apply() by collecting messages from
    a branch. Produces identical results to PromptTree.get_messages().

    Examples:
        >>> from dataknobs_utils.llm_utils import MessageCollector
        >>>
        >>> # Assuming you have a PromptTree instance cur_prompt_tree
        >>> # Using MessageCollector:
        >>> mc = MessageCollector(with_metadata=False)
        >>> cur_prompt_tree.apply(mc, level_offset=-1)
        >>> messages = mc.messages
        >>>
        >>> # Equivalent using get_messages:
        >>> messages = cur_prompt_tree.get_messages(level_offset=-1, with_metadata=False)

    Note:
        This pattern can be adapted for other tree traversal operations.

    Attributes:
        with_metadata: Whether to include metadata in collected messages.
        messages: Collected message dictionaries.
    """

    def __init__(self, with_metadata: bool = True):
        """Initialize message collector.

        Args:
            with_metadata: If True, includes metadata in collected messages.
                Defaults to True.
        """
        self.with_metadata = with_metadata
        self.messages: List[Dict[str, Any]] = []

    def __call__(self, ptree: PromptTree) -> None:
        """Collect message from a PromptTree node.

        Called by PromptTree.apply() for each node in the traversal.

        Args:
            ptree: PromptTree node to collect message from.
        """
        self.messages.insert(0, ptree.message.get_message(with_metadata=self.with_metadata))

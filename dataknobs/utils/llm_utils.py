import json
from dataknobs.structures.tree import Tree
from datetime import datetime
from typing import Any, Callable, Dict, List, Union


def get_value_by_key(
        d: Dict[str, Any],
        pathkey: str,
        default_value: Any = None,
) -> Any:
    '''
    Get the "deep" value from the dictionary according to the dot-delimited
    path key.
    :param d: The (possibly nested) dictionary.
    :param pathkey: The path key
    :param default_value: The value to return when the path key doesn't exist
    :return: The retrieved value from the dictionary or the default_value

    For example:
       Given d = {"foo": {"bar": "baz"}} and pathkey="foo.bar", the value="baz"
    '''
    path = pathkey.split(".")
    if d is not None:
        failed = False
        for key in path:
            if d is None or not isinstance(d, dict):
                failed = True
                break
            if key not in d:
                failed = True
                break
            d = d[key]
    return d if not failed else default_value


class PromptMessage:
    '''
    Wrapper for a prompt message of the form:
      {"role": <role>, "content": <content>}
    Where <role> is e.g., "system", "user", "assistant" and content is the
    prompt or LLM-generated text.

    Optionally, "metadata" can be added to any message with a form like
        {
          "generation_args": {...},
          "execution_data": {...},  # model_name, starttime, endtime
          "user_comments": [{"user": user, "comment": comment},...],
        }
    '''

    def __init__(self, role: str, content: str, metadata: Dict[str, Any] = None):
        '''
        Initialize with the given role and content.
        :param role: The role (e.g., "system", "user", or "assistant")
        :param content: The associated prompt or LLM-generated text
        :param metadata: The metadata for this message

        Where metadata holds:
          {
            "generation_args": {...},
            "execution_data": {...},  # model_name, starttime, endtime
            "user_comments": ["",...],
          }
        '''
        self.role = role
        self.content = content
        self.metadata = metadata
        self._dict = None  # The dict without metadata

    def __repr__(self) -> str:
        ''' Get this message as a json string without metadata '''
        return self.to_json(with_metadata=False)

    def to_json(self, with_metadata: bool = True) -> str:
        '''
        Get this node's message as a json str. Note that an error will be
        thrown if the metadata is not json serializable.
        '''
        return json.dumps(self.get_message(with_metadata=with_metadata))

    def get_message(self, with_metadata: Union[bool, str] = False) -> Dict[str, str]:
        '''
        Get the role and content for this node as a dict of the form:
          {"role": <role>, "content": <content>}
        :param with_metadata: If true, and there is metadata, then include the
            metadata in the returned dictionary. If a string, then also use
            this string as the attribute for the metadata object in the
            dictionary.
        :return: The message dictionary
        '''
        retval = None
        if self._dict is None:
            self._dict = {
                "role": self.role,
                "content": self.content,
            }
        if with_metadata and self.metadata is not None:
            retval = self._dict.copy()
            attr = with_metadata if isinstance(with_metadata, str) else "metadata"
            retval[attr] = self.metadata
        else:
            retval = self._dict
        return retval

    @staticmethod
    def build_instance(message_dict: Dict[str, Any]) -> 'PromptMessage':
        '''
        (Re)build a PromptMessage instance from its dictionary form.
        :param message_dict: A message dictionary of the form {
          "role": <str>, "content": <str>, "metadata": <dict>
        }
        '''
        return PromptMessage(
            message_dict.get("role", "unknown"),
            message_dict.get("content", ""),
            metadata = message_dict.get("metadata", None)
        )


class PromptTree:
    '''
    Data structure to preserve a tree of prompt messages.

    Where each instance represents a node in a tree where the root node holds
    the information for the initial prompt, and each descendant holds
    information for successive follow-on prompts.

    The structure is a tree so that we can follow different paths of
    prompts at any point (from any node.)

    The tree structure is accessed through the "node" property of an instance.
    The PromptTree instance can be retrieved from the node through the node's
    data property.

    Because of the hierarchical nature of the tree, prompt message metadata
    is inherited from ancestors or redefined by descendants
    '''

    def __init__(
            self,
            message: PromptMessage = None,
            role: str = None,
            content: str = None,
            metadata: Dict[str, Any] = None,
            parent: 'PromptTree' = None,
    ):
        '''
        Construct either with the given message or create a message
        from the supplied role and content. If a role and/or content are
        supplied with a message, then a new message will be created with
        the role and/or content overridden by the parameters; otherwise
        the message object is used as-is.

        :param message: The message (or message template) for this node
        :param role: The role (override) for this node
        :param content: The content (override) for this node
        :param metadata: The metadata (override) for this node
        :param parent: The parent PromptTree to this new instance
        '''
        # NOTE: _tree_id is the ID factory global to this full tree
        self._tree_id = self.IdTracker() if parent is None else parent._tree_id
        self._node_id = self._tree_id.next_id()
        self.node = None
        the_message = message
        # If there's an override AND a message
        if (
                role is not None or
                content is not None or
                metadata is not None
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
        self.message = the_message or PromptMessage(role, content, metadata=metadata)
        if parent is not None:
            self.node = parent.node.add_child(self)
        else:
            self.node = Tree(self)

    def __repr__(self) -> str:
        '''
        Get this PromptTree (node) as a string of the form:
            <node_depth>.<sibling_id>:<role>(content_length)
        '''
        return f"{self.node_id}:{self.message.role}({len(self.message.content)})"

    def _inc_tree_id(self) -> int:
        '''
        Store a global (to all nodes in this tree) ID as a value in a single
        element list. This shared value across all PromptTree instances will
        always hold the total number of nodes in the full tree.
        '''

    @property
    def node_count(self) -> int:
        '''
        Get the total number of nodes in this prompt tree.
        '''
        return self._tree_id.id_count

    @property
    def node_id(self) -> int:
        '''
        Get this node's ID.
        '''
        return self._node_id

    def add_message(
            self,
            message: PromptMessage = None,
            role: str = None,
            content: str = None,
            metadata: Dict[str, Any] = None,
    ) -> 'PromptTree':
        '''
        Add the message as a child of this node, returning the new tree node.
        :param message: The "child" prompt message
        :param role: The role (override)
        :param content: The content (override)
        :param metadata: The metadata (override)
        '''
        return PromptTree(
            message=message,
            role=role,
            content=content,
            metadata=metadata, parent=self
        )
        

    @property
    def depth(self) -> int:
        ''' Get the depth of this node in its tree '''
        return self.node.depth

    @property
    def metadata(self) -> Dict[str, Any]:
        '''
        Get this node's metadata, inheriting from an ancestor if needed.
        :return: This node's metadata
        '''
        metadata = self.message.metadata
        node = self.node
        while metadata is None and node.parent is not None:
            node = node.parent
            metadata = node.data.metadata
        return metadata

    def get_metadata_value(self, pathkey: str, default_value: Any = None) -> Any:
        '''
        Get the (inherited if necessary) metadata value by path key.
        :param pathkey: A dot-delimited path to the data to fetch.
        :param default_value: The value to return when the path key doesn't exist
        :return: The (possibly inherited) metadata value.
        '''
        value = None
        node = self.node
        while value is None and node is not None:
            metadata = node.data.message.metadata
            if metadata is not None:
                value = get_value_by_key(
                    metadata, pathkey, default_value=default_value
                )
            node = node.parent
        return value

    def apply(
            self,
            ptree_fn: Callable[['PromptTree'], None],
            level_offset: int = -1,
    ):
        '''
        Apply a function to each of the PromptTree nodes from the identified
        level through this node.
        :param ptree_fn: The fn(ptree_node) to apply to each node in order
            from this, the deepest, node up to the identified node.
        :param level_offset: The level offset specifying the nodes
            relative to this from which to collect messages.
            Where, level_offset identifies the start node as:
              0, 1, ... ==> current, parent, grandparent, etc.
              -1, -2, ... ==> root, 1-under root, etc.
            and the end node is this node.
            Default is to get the duration from this node through the root
        '''
        node = self.node  # start from this node
        end_node = self.get_level_node(level_offset).node  # need .node!
        while True:
            if node is None:
                break
            ptree_fn(node.data)  # call the function
            if node == end_node:  # time to stop moving up
                break
            node = node.parent  # move up


    def get_messages(
            self,
            level_offset: int = -1,
            with_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        '''
        Get the prompt messages suitable for passing as context into an LLM.
        :param level_offset: The level offset specifying the nodes
            relative to this from which to collect messages.
            Where, level_offset identifies the start node as:
              0, 1, ... ==> current, parent, grandparent, etc.
              -1, -2, ... ==> root, 1-under root, etc.
            and the end node is this node.
            Default is to get the duration from this node through the root
        :param with_metadata: True to include metadata
        :return: The list of message dictionaries.
        '''
        node = self.node
        end_node = self.get_level_node(level_offset).node  # need .node!
        messages = list()
        while True:
            if node is None:
                break
            messages.insert(
                0, node.data.message.get_message(with_metadata=with_metadata)
            )
            if node == end_node:
                break
            node = node.parent
        return messages

    def get_duration(self, level_offset: int = 0) -> int:
        '''
        Get the total number of seconds from metadata["execution_data"]'s
        "endtime" - "starttime" for the specified prompt levels.
        :param level_offset: The level offset specifying the nodes
            relative to this for which to get the duration.
            Where, level_offset identifies the start node as:
              0, 1, ... ==> current, parent, grandparent, etc.
              -1, -2, ... ==> root, 1-under root, etc.
            and the end node is this node.
            Default is to get the duration for this node only.
        :return: The total number of seconds
        '''
        node = self.node
        end_node = self.get_level_node(level_offset).node  # need .node!
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
        '''
        Get the duration in seconds for this node from
        metadata["execution_data"]'s endtime - starttime, assuming
        endtime and starttime are time strings in iso format
        (e.g., s = d.isoformat() for d=starttime or endtime)
        :return: The duration in seconds (or 0)
        '''
        duration = 0
        metadata = self.message.metadata  # NOT self.metadata! don't inherit.
        if metadata is not None:
            exec_data = metadata.get("execution_data", None)
            if exec_data is not None:
                starttime = exec_data.get("starttime", None)
                endtime = exec_data.get("endtime", None)
                if starttime and endtime:
                    duration = (
                        datetime.fromisoformat(endtime)
                        - datetime.fromisoformat(starttime)
                    ).total_seconds()
        return duration

    def get_level_node(self, level_offset: int) -> 'PromptTree':
        '''
        Get the prompt tree node at the specified level offset relative to
        this node.
        :param level_offset: The level offset specifying the nodes
            relative to this for which to get the duration.
            Where, level_offset identifies the start node as:
              0, 1, ... ==> current, parent, grandparent, etc.
              -1, -2, ... ==> root, 1-under root, etc.
            and the end node is this node.
        :return: The specified PromptTree
        '''
        if level_offset == 0:
            # This is the node
            return self
        elif level_offset > 0:
            # The parent node, stopping at the root
            node = self.node
            for _ in range(level_offset):
                if node.parent is None:
                    return node.data
                else:
                    node = node.parent
            return node.data
        elif level_offset < 0:
            # Offset from root, stopping at this node
            node = self.node
            nodes = []
            while node is not None:
                nodes.insert(0, node)
                node = node.parent
            node = nodes[min(abs(level_offset + 1), len(nodes) - 1)]
            return node.data

    def find_node_by_id(self, node_id: int) -> "PromptTree":
        '''
        Find the PromptTree Node with the given node ID.

        :param node_id: The node ID to find
        :return: The node or None
        '''
        found = self.node.root.find_nodes(
            lambda node: node.data.node_id == node_id,
            only_first=True
        )
        return found[0].data if len(found) > 0 else None

    def find_nodes(
            self,
            match_fn: Callable[[Tree], bool] = None,
    ) -> List["PromptTree"]:
        '''
        Find prompt tree nodes that match the given criteria.

        :param match_fn: fn(<tree>) where tree.data is a PromptTree instance
            and the function returns True to select the instance
        :return: The (possibly empty) list of all matches.
        '''
        found = self.node.root.find_nodes(match_fn)
        return [node.data for node in found]

    def serialize_tree(
            self,
            full: bool = False,
            with_metadata: bool = True
    ) -> Union[Dict, List]:
        '''
        Serialize this tree's messages as nested lists representing the tree
        structure.

        If the tree to be serialized is a single node, then the result will be
        the serialization of the node's message; otherwise, the result will be
        a list of lists representing the tree structure, where list elements
        are the serialized message for each node.

        :param full: When True, serialize from the root; otherwise only
            serialize from this node.
        :param with_metadata: True to include metadata in the serialization
        :return: A list of nested strings.

        Note that the full prompt tree can be reconstructed from this output.
        '''
        return self._do_serialize(
            self.node.root if full else self.node,
            with_metadata=with_metadata
        )

    def _do_serialize(
            self,
            node: Tree,
            with_metadata: bool,
    ) -> Union[Dict, List]:
        '''
        Do the recursive serialization of the tree nodes.
        :param node: The current node to serialize
        :param with_metadata: True to include metadata in the serialization
        '''
        retval = node.data.message.get_message(with_metadata=with_metadata)
        if node.has_children():
            retval = [
                retval,
                [
                    self._do_serialize(
                        child,
                        with_metadata=with_metadata,
                    )
                    for child in node.children
                ]
            ]
        return retval

    @staticmethod
    def build_instance(
            data: Union[Dict, List],
            parent: "PromptTree" = None
    ) -> "PromptTree":
        '''
        (Re)build a PromptTree from its serialized data, e.g., output
        from prompt_tree.serialize_tree()

        :param data: The list of data
        :param parent: The parent prompt tree into which to deserialize
        :return: The last, deepest prompt tree (suitable for continuing
            a conversation in the last branch where it left off)
        '''
        pt = parent
        if isinstance(data, list) and len(data) > 0:
            # data[0] is a parent
            # data[1] is a list of children
            cur_parent = PromptTree.build_instance(data[0], parent=pt)
            for item in data[1]:
                pt = PromptTree.build_instance(item, parent=cur_parent)
        else:
            # data is a single (terminal) node
            pt = PromptTree(message=PromptMessage.build_instance(data), parent=parent)
        return pt

    class IdTracker:
        ''' Simple ID tracker for the connected PromptTreeNodes '''
        def __init__(self):
            self._next_id = 0
    
        def next_id(self) -> int:
            ''' Get the next ID '''
            retval = self._next_id
            self._next_id += 1
            return retval

        @property
        def id_count(self) -> int:
            '''
            Get the number of assigned IDs
            '''
            return self._next_id


class MessageCollector:
    '''
    An example of a class to be used by PromptTree.apply to collect all
    messages in a branch with or without metadata to illustrate how to use the
    PromptTree.apply method. The results from using this should be identical
    to those from using the PromptTree.get_messages method.

    Usage:
        > mc = MessageCollector(with_metadata=False)
        > cur_prompt_tree.apply(mc, level_offset=-1)
        > messages = mc.messages

    Where this alternate path would achieve the same results:
        > messages = cur_prompt_tree.get_messages(level_offset=-1, with_metadata=False)

    The intent is that you can use this as a pattern for performing other
    types of operations with the prompt tree.
    '''

    def __init__(self, with_metadata: bool = True):
        '''
        :param with_metadata: True to include metadata with the message dicts.
        '''
        self.with_metadata = with_metadata
        self.messages = []

    def __call__(self, ptree: PromptTree):
        '''
        :param ptree: The prompt tree (node) on which to apply this function.
        '''
        self.messages.insert(
            0, ptree.message.get_message(with_metadata=self.with_metadata)
        )

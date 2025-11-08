"""Tree data structure with parent-child relationships and traversal methods.

This module provides a flexible Tree implementation where each node contains
arbitrary data and maintains bidirectional links between parents and children
for efficient traversal in both directions.

The Tree class supports various operations including:
- Adding and removing nodes
- Depth-first and breadth-first search
- Finding common ancestors
- Collecting terminal nodes
- Building visual representations with Graphviz

Typical usage example:

    ```python
    from dataknobs_structures import Tree

    # Create a tree structure
    root = Tree("root")
    child1 = root.add_child("child1")
    child2 = root.add_child("child2")
    grandchild = child1.add_child("grandchild")

    # Find nodes
    found = root.find_nodes(lambda n: "child" in str(n.data))

    # Traverse the tree
    for parent, child in root.get_edges():
        print(f"{parent} -> {child}")
    ```
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any, Deque, List, Tuple, Union

import graphviz
from pyparsing import OneOrMore, nestedExpr


class Tree:
    """A tree node with arbitrary data and parent-child relationships.

    Each Tree instance represents a node in a tree structure that maintains:
    - Arbitrary data of any type
    - An ordered list of child nodes
    - A reference to its parent node (None for root)
    - Bidirectional links between parents and children for efficient traversal

    The tree structure supports both depth-first and breadth-first traversal,
    node searching, ancestor finding, and various tree manipulation operations.
    All nodes are doubly-linked, allowing traversal both up (to parents) and
    down (to children).

    Attributes:
        data: The data contained in this node (any type).
        children: List of child Tree nodes, or None if no children.
        parent: Parent Tree node, or None if this is a root node.
        root: The root node of this tree (traverses up to find it).
        depth: Number of hops from the root to this node (root has depth 0).

    Example:
        ```python
        # Create a tree structure
        root = Tree("root")
        child1 = Tree("child1", parent=root)
        child2 = Tree("child2", parent=root)

        # Add grandchildren
        grandchild = Tree("grandchild", parent=child1)

        # Navigate the tree
        print(grandchild.root.data)  # "root"
        print(grandchild.depth)      # 2
        print(child1.num_children)   # 1
        ```

    Note:
        Tree nodes can be created standalone and connected later, or created
        with a parent reference for immediate attachment. When a node is added
        as a child, it's automatically removed from any previous parent.
    """

    def __init__(
        self,
        data: Any,
        parent: Union[Tree, Any] = None,
        child_pos: int | None = None,
    ):
        """Initialize a tree node with optional parent attachment.

        Creates a new tree node containing arbitrary data. If a parent is provided,
        the new node is automatically added as a child of that parent. If the parent
        is not already a Tree instance, it will be wrapped in one.

        Args:
            data: The data to be contained within this node. Can be any type.
            parent: Optional parent node to attach to. Can be a Tree instance or
                data that will be wrapped in a Tree. If None, creates a root node.
            child_pos: Optional 0-based position among parent's children. If None,
                appends to the end of the parent's children list.

        Example:
            ```python
            # Create root node
            root = Tree("root")

            # Add children
            child1 = Tree("child1", parent=root)
            child2 = Tree("child2", parent=root, child_pos=0)  # Insert at start

            # Create subtree
            grandchild = Tree("grandchild", parent=child1)
            ```
        """
        self._data = data
        self._children: List[Tree] | None = None
        self._parent: Tree | None = None
        if parent is not None:
            if not isinstance(parent, Tree):
                parent = Tree(parent)
            parent.add_child(self, child_pos)

    def __repr__(self) -> str:
        """Return string representation of this tree.

        Returns:
            Multi-line string representation using as_string() method.
        """
        return self.as_string(delim="  ", multiline=True)

    @property
    def data(self) -> Any:
        """The data contained in this node.

        Returns:
            The arbitrary data stored in this node.
        """
        return self._data

    @data.setter
    def data(self, data: Any) -> None:
        """Set this node's data.

        Args:
            data: The new data value for this node.
        """
        self._data = data

    @property
    def children(self) -> List[Tree] | None:
        """This node's children as an ordered list.

        Returns:
            List of child Tree nodes, or None if this node has no children.
        """
        return self._children

    @property
    def parent(self) -> Tree | None:
        """This node's parent.

        Returns:
            Parent Tree node, or None if this is a root node.
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Tree | None) -> None:
        """Set this node's parent.

        Args:
            parent: The new parent node, or None to make this a root node.
        """
        self._parent = parent

    @property
    def root(self) -> Tree:
        """The root node of this tree.

        Traverses up the parent chain to find the root node.

        Returns:
            The root Tree node (the node with no parent).
        """
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    @property
    def sibnum(self) -> int:
        """This node's position among its siblings.

        Returns:
            0-based index among parent's children, or 0 if this is a root node.
        """
        return (
            self._parent.children.index(self)
            if self._parent is not None and self._parent.children is not None
            else 0
        )

    @property
    def num_siblings(self) -> int:
        """Total number of siblings including this node.

        Returns:
            Number of children the parent has (including this node), or 1 if root.
        """
        return self._parent.num_children if self._parent is not None else 1

    @property
    def next_sibling(self) -> Tree | None:
        """The next sibling of this node.

        Returns:
            Next sibling Tree node in parent's children list, or None if this is
            the last child or a root node.
        """
        result = None
        if self._parent and self._parent.children:
            sibs = self._parent.children
            nextsib = sibs.index(self) + 1
            if nextsib < len(sibs):
                result = sibs[nextsib]
        return result

    @property
    def prev_sibling(self) -> Tree | None:
        """The previous sibling of this node.

        Returns:
            Previous sibling Tree node in parent's children list, or None if this
            is the first child or a root node.
        """
        result = None
        if self._parent and self._parent.children:
            sibs = self._parent.children
            prevsib = sibs.index(self) - 1
            if prevsib >= 0:
                result = sibs[prevsib]
        return result

    def has_children(self) -> bool:
        """Check if this node has any children.

        Returns:
            True if this node has at least one child, False otherwise.
        """
        return self._children is not None and len(self._children) > 0

    @property
    def num_children(self) -> int:
        """Number of children under this node.

        Returns:
            Count of direct children (0 if no children).
        """
        return len(self._children) if self._children is not None else 0

    def has_parent(self) -> bool:
        """Check if this node has a parent.

        Returns:
            True if this node has a parent, False if this is a root node.

        Note:
            The root of a tree has no parent.
        """
        return self._parent is not None

    @property
    def depth(self) -> int:
        """Depth of this node in the tree.

        The depth is the number of edges from the root to this node.
        The root node has depth 0, its children have depth 1, and so on.

        Returns:
            Number of hops from the root to this node (0 for root).

        Example:
            ```python
            root = Tree("root")
            child = Tree("child", parent=root)
            grandchild = Tree("grandchild", parent=child)

            print(root.depth)        # 0
            print(child.depth)       # 1
            print(grandchild.depth)  # 2
            ```
        """
        result = 0
        curp = self.parent
        while curp is not None:
            curp = curp.parent
            result += 1
        return result

    def add_child(self, node_or_data: Union[Tree, Any], child_pos: int | None = None) -> Tree:
        """Add a child node to this node.

        If the child is already part of another tree, it will be pruned from that
        tree first. If node_or_data is not a Tree instance, it will be wrapped in
        a new Tree node. The child's parent reference is automatically updated.

        Args:
            node_or_data: The child to add. Can be a Tree instance or any data that
                will be wrapped in a Tree node.
            child_pos: Optional 0-based position to insert the child. If None,
                appends to the end. If provided, inserts at the specified position,
                shifting existing children.

        Returns:
            The child Tree node (either the provided node or a newly created one).

        Example:
            ```python
            # Create parent and add children
            parent = Tree("parent")
            child1 = parent.add_child("child1")
            child2 = parent.add_child("child2")

            # Insert at specific position
            child0 = parent.add_child("child0", child_pos=0)

            # Method chaining
            root = Tree("root").add_child("a").add_child("b")
            ```

        Note:
            The child's parent attribute is automatically updated to reference
            this node, maintaining the tree's structural integrity.
        """
        if self._children is None:
            self._children = []
        if isinstance(node_or_data, Tree):
            child = node_or_data
            child.prune()
            if child_pos is not None and child_pos < len(self._children) and child_pos >= 0:
                self._children.insert(child_pos, child)
            else:
                self._children.append(child)
        else:
            child = Tree(node_or_data, self, child_pos=child_pos)
        child.parent = self
        return child

    def add_edge(
        self,
        parent_node_or_data: Union[Tree, Any],
        child_node_or_data: Union[Tree, Any],
    ) -> Tuple[Tree, Tree]:
        """Add a parent-child edge to this tree.

        Creates or reuses nodes to establish a parent-child relationship. If the
        parent or child already exist in the tree (matched by data equality), they
        are reused. If the child exists elsewhere, it's moved to be a child of the
        parent. If neither exist, the parent is added as a child of this node.

        Args:
            parent_node_or_data: The parent node (Tree instance) or data to match/create.
            child_node_or_data: The child node (Tree instance) or data to match/create.

        Returns:
            Tuple of (parent_node, child_node) Tree instances.

        Example:
            ```python
            # Build tree by adding edges
            root = Tree("root")
            root.add_edge("parent1", "child1")
            root.add_edge("parent1", "child2")  # Reuses existing parent1
            root.add_edge("child1", "grandchild")

            # Result:
            # root
            #   parent1
            #     child1
            #       grandchild
            #     child2
            ```

        Note:
            Nodes are matched by data equality. If a node with matching data
            already exists in the tree, it will be reused rather than creating
            a duplicate.
        """
        parent = None
        child = None

        if isinstance(parent_node_or_data, Tree):
            parent = parent_node_or_data
            # if it is not in this tree ...
            if (
                len(
                    self.find_nodes(lambda node: node == parent, include_self=True, only_first=True)
                )
                == 0
            ):
                # ...then add it as a child of self
                self.add_child(parent)
        else:
            # can we find the data in this tree ...
            found = self.find_nodes(
                lambda node: node.data == parent_node_or_data, include_self=True, only_first=True
            )
            if len(found) > 0:
                parent = found[0]
            else:
                parent = self.add_child(parent_node_or_data)

        if isinstance(child_node_or_data, Tree):
            child = parent.add_child(child_node_or_data)
        else:
            # can we find the data in this tree ...
            found = self.find_nodes(
                lambda node: node.data == child_node_or_data, include_self=True, only_first=True
            )
            if len(found) > 0:
                child = parent.add_child(found[0])
            else:
                child = parent.add_child(child_node_or_data)

        return (parent, child)

    def prune(self) -> Tree | None:
        """Remove this node from its parent.

        Detaches this node from its parent's children list and sets this node's
        parent to None. The node and its subtree remain intact.

        Returns:
            This node's former parent, or None if this was already a root node.

        Example:
            ```python
            root = Tree("root")
            child = root.add_child("child")
            grandchild = child.add_child("grandchild")

            # Prune child (and its subtree) from root
            former_parent = child.prune()
            print(former_parent.data)  # "root"
            print(child.parent)        # None
            print(child.children)      # [grandchild] - subtree intact
            ```
        """
        result = self._parent
        if self._parent is not None:
            if self._parent.children is not None:
                self._parent.children.remove(self)
            self._parent = None
        return result

    def find_nodes(
        self,
        accept_node_fn: Callable[[Tree], bool],
        traversal: str = "dfs",
        include_self: bool = True,
        only_first: bool = False,
        highest_only: bool = False,
    ) -> List[Tree]:
        """Find nodes matching a condition using depth-first or breadth-first search.

        Searches the tree using the specified traversal strategy and returns all
        nodes for which the accept function returns True.

        Args:
            accept_node_fn: Function that takes a Tree node and returns True to
                include it in results, False to skip it.
            traversal: Search strategy - either 'dfs' (depth-first) or 'bfs'
                (breadth-first). Defaults to 'dfs'.
            include_self: If True, considers this node in the search. If False,
                starts with this node's children. Defaults to True.
            only_first: If True, stops after finding the first match. Defaults
                to False.
            highest_only: If True, doesn't search below matched nodes (stops
                descending when a match is found). Defaults to False.

        Returns:
            List of Tree nodes that matched the accept function.

        Example:
            ```python
            root = Tree("root")
            root.add_child("apple")
            root.add_child("banana")
            child = root.add_child("parent")
            child.add_child("apricot")

            # Find all nodes containing 'a'
            found = root.find_nodes(lambda n: 'a' in str(n.data))
            # Returns: [root, apple, banana, parent, apricot]

            # Find first node containing 'a'
            first = root.find_nodes(
                lambda n: 'a' in str(n.data),
                only_first=True
            )
            # Returns: [root]

            # Breadth-first search
            bfs = root.find_nodes(lambda n: 'a' in str(n.data), traversal='bfs')
            ```
        """
        queue: Deque[Tree] = deque()
        found: List[Tree] = []
        if include_self:
            queue.append(self)
        elif self.children:
            queue.extend(self.children)
        while bool(queue):  # true while length(queue) > 0
            item = queue.popleft()
            if accept_node_fn(item):
                found.append(item)
                if only_first:
                    break
                elif highest_only:
                    continue
            if item.children:
                if traversal == "dfs":
                    queue.extendleft(reversed(item.children))
                elif traversal == "bfs":
                    queue.extend(item.children)
        return found

    def collect_terminal_nodes(
        self, accept_node_fn: Callable[[Tree], bool] | None = None, _found: List[Tree] | None = None
    ) -> List[Tree]:
        """Collect all leaf nodes (nodes with no children) in this tree.

        Recursively traverses the tree and collects nodes that have no children,
        optionally filtering them with an accept function.

        Args:
            accept_node_fn: Optional function to filter which terminal nodes to
                include. If None, includes all terminal nodes.
            _found: Internal parameter for recursion. Do not use.

        Returns:
            List of terminal (leaf) Tree nodes.

        Example:
            ```python
            root = Tree("root")
            child1 = root.add_child("child1")
            child1.add_child("leaf1")
            child1.add_child("leaf2")
            root.add_child("leaf3")

            # Collect all terminal nodes
            leaves = root.collect_terminal_nodes()
            # Returns: [leaf1, leaf2, leaf3]

            # Collect only specific leaves
            filtered = root.collect_terminal_nodes(
                lambda n: "1" in str(n.data)
            )
            # Returns: [leaf1]
            ```
        """
        if _found is None:
            _found = []
        if not self._children:
            if accept_node_fn is None or accept_node_fn(self):
                _found.append(self)
        else:
            for child in self._children:
                child.collect_terminal_nodes(accept_node_fn=accept_node_fn, _found=_found)
        return _found

    def get_edges(
        self,
        traversal: str = "bfs",
        include_self: bool = True,
        as_data: bool = True,
    ) -> List[Tuple[Union[Tree, Any], Union[Tree, Any]]]:
        """Get all parent-child edges in this tree.

        Returns a list of (parent, child) tuples representing all edges in the tree,
        using either depth-first or breadth-first traversal.

        Args:
            traversal: Search strategy - either 'dfs' (depth-first) or 'bfs'
                (breadth-first). Defaults to 'bfs'.
            include_self: If True, includes edges from this node to its children.
                If False, starts with this node's children. Defaults to True.
            as_data: If True, returns data values in tuples. If False, returns
                Tree node instances. Defaults to True.

        Returns:
            List of (parent, child) tuples, either as data values or Tree nodes.

        Example:
            ```python
            root = Tree("root")
            child1 = root.add_child("child1")
            child2 = root.add_child("child2")
            child1.add_child("grandchild")

            # Get edges as data
            edges = root.get_edges()
            # Returns: [("root", "child1"), ("root", "child2"),
            #           ("child1", "grandchild")]

            # Get edges as Tree nodes
            edges_nodes = root.get_edges(as_data=False)
            # Returns: [(root_node, child1_node), ...]
            ```
        """
        queue: Deque[Tree] = deque()
        result: List[Tuple[Union[Tree, Any], Union[Tree, Any]]] = []
        if self.children:
            queue.extend(self.children)
        while bool(queue):  # true while length(queue) > 0
            item = queue.popleft()
            if item.parent:
                if item.parent != self or include_self:
                    result.append((item.parent.data, item.data) if as_data else (item.parent, item))
            if item.children:
                if traversal == "dfs":
                    queue.extendleft(reversed(item.children))
                elif traversal == "bfs":
                    queue.extend(item.children)
        return result

    def get_path(self) -> List[Tree]:
        """Get the path from the root to this node.

        Returns:
            Ordered list of Tree nodes from root to this node (inclusive).

        Example:
            ```python
            root = Tree("root")
            child = Tree("child", parent=root)
            grandchild = Tree("grandchild", parent=child)

            path = grandchild.get_path()
            print([n.data for n in path])  # ["root", "child", "grandchild"]
            ```
        """
        path: Deque[Tree] = deque()
        node: Tree | None = self
        while node is not None:
            path.appendleft(node)
            node = node.parent
        return list(path)

    def is_ancestor(self, other: Tree, self_is_ancestor: bool = False) -> bool:
        """Check if this node is an ancestor of another node.

        An ancestor is any node on the path from another node to the root.

        Args:
            other: The potential descendant node to check.
            self_is_ancestor: If True, considers a node to be its own ancestor.
                Defaults to False.

        Returns:
            True if this node is an ancestor of the other node, False otherwise.

        Example:
            ```python
            root = Tree("root")
            child = Tree("child", parent=root)
            grandchild = Tree("grandchild", parent=child)

            print(root.is_ancestor(grandchild))         # True
            print(child.is_ancestor(grandchild))        # True
            print(grandchild.is_ancestor(root))         # False
            print(grandchild.is_ancestor(grandchild))   # False
            print(grandchild.is_ancestor(grandchild, self_is_ancestor=True))  # True
            ```
        """
        result = False
        parent = other if self_is_ancestor else other.parent
        while parent is not None:
            if parent == self:
                result = True
                break
            parent = parent.parent
        return result

    def find_deepest_common_ancestor(self, other: Tree | None) -> Tree | None:
        """Find the deepest (closest) common ancestor of this node and another.

        The deepest common ancestor is the lowest node that is an ancestor of
        both nodes. This is also known as the lowest common ancestor (LCA).

        Args:
            other: The other node to find a common ancestor with. Can be None.

        Returns:
            The deepest common ancestor Tree node, or None if other is None or
            if the nodes are not in the same tree.

        Example:
            ```python
            root = Tree("root")
            left = root.add_child("left")
            right = root.add_child("right")
            left_child = left.add_child("left_child")
            right_child = right.add_child("right_child")

            # Find common ancestor
            ancestor = left_child.find_deepest_common_ancestor(right_child)
            print(ancestor.data)  # "root"

            # Nodes in same branch
            ancestor = left_child.find_deepest_common_ancestor(left)
            print(ancestor.data)  # "left"

            # Same node
            ancestor = left_child.find_deepest_common_ancestor(left_child)
            print(ancestor.data)  # "left_child"
            ```
        """
        if other is None:
            return None
        if self == other:
            return self
        result: Tree | None = None
        mypath, otherpath = self.get_path(), other.get_path()
        mypathlen, otherpathlen = len(mypath), len(otherpath)
        mypathidx, otherpathidx = 0, 0
        while mypathidx < mypathlen and otherpathidx < otherpathlen:
            mynode, othernode = mypath[mypathidx], otherpath[otherpathidx]
            mypathidx += 1
            otherpathidx += 1
            if mynode != othernode:
                break  # diverged
            else:
                result = mynode
        return result

    def as_string(self, delim: str = " ", multiline: bool = False) -> str:
        """Get a string representation of this tree.

        Creates a parenthesized string representation of the tree structure,
        with optional multiline formatting and custom indentation.

        Args:
            delim: The delimiter/indentation to use between levels. Defaults to
                a single space.
            multiline: If True, includes newlines for readability. If False,
                creates a compact single-line representation. Defaults to False.

        Returns:
            String representation of this tree and its descendants.

        Example:
            ```python
            root = Tree("root")
            child1 = root.add_child("child1")
            child1.add_child("leaf1")
            root.add_child("child2")

            # Compact representation
            print(root.as_string())
            # Output: (root child1 leaf1 child2)

            # Multiline with indentation
            print(root.as_string(delim="  ", multiline=True))
            # Output:
            # (root
            #   child1
            #     leaf1
            #   child2)
            ```
        """
        result = ""
        if self._children:
            btwn = "\n" if multiline else ""
            result = "(" + str(self.data)
            for child in self._children:
                d = (child.depth if multiline else 1) * delim
                result += btwn + d + child.as_string(delim=delim, multiline=multiline)
            result += ")"
        else:
            result = str(self.data)
        return result

    def get_deepest_left(self) -> Tree:
        """Get the leftmost terminal descendant of this node.

        Follows the leftmost (first) child at each level until reaching a leaf node.

        Returns:
            The terminal (leaf) node reached by always taking the first child.

        Example:
            ```python
            root = Tree("root")
            left = root.add_child("left")
            left.add_child("left_left")
            root.add_child("right")

            leftmost = root.get_deepest_left()
            print(leftmost.data)  # "left_left"
            ```
        """
        node = self
        while node.has_children() and node.children is not None:
            node = node.children[0]
        return node

    def get_deepest_right(self) -> Tree:
        """Get the rightmost terminal descendant of this node.

        Follows the rightmost (last) child at each level until reaching a leaf node.

        Returns:
            The terminal (leaf) node reached by always taking the last child.

        Example:
            ```python
            root = Tree("root")
            root.add_child("left")
            right = root.add_child("right")
            right.add_child("right_right")

            rightmost = root.get_deepest_right()
            print(rightmost.data)  # "right_right"
            ```
        """
        node = self
        while node.has_children() and node.children is not None:
            node = node.children[-1]
        return node

    def build_dot(
        self, node_name_fn: Callable[[Tree], str] | None = None, **kwargs: Any
    ) -> graphviz.graphs.Digraph:
        """Build a Graphviz Digraph for visualizing this tree.

        Creates a directed graph representation of the tree structure that can be
        rendered to various formats (PNG, PDF, SVG, etc.) using Graphviz.

        Args:
            node_name_fn: Optional function to generate node labels. Takes a Tree
                node and returns a string. If None, uses str(node.data).
            **kwargs: Additional keyword arguments passed to graphviz.Digraph
                constructor (e.g., name, format, node_attr, edge_attr).

        Returns:
            A graphviz.Digraph object representing this tree.

        Example:
            ```python
            root = Tree("root")
            child1 = root.add_child("child1")
            child2 = root.add_child("child2")

            # Basic usage
            dot = root.build_dot(name='MyTree', format='png')
            print(dot.source)  # View the DOT source

            # Render to file
            dot.render('/tmp/tree', format='png')

            # Custom node labels
            dot = root.build_dot(
                node_name_fn=lambda n: f"[{n.data}]",
                node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue'}
            )

            # Display in Jupyter
            from IPython.display import Image
            Image(filename=dot.render('/tmp/tree'))
            ```

        Note:
            Requires the graphviz package and Graphviz system installation.
        """
        if node_name_fn is None:
            def node_name_fn(n):
                return str(n.data)
        dot = graphviz.Digraph(**kwargs)
        ids = {}  # ids[node] -> id
        for idx, node in enumerate(self.root.find_nodes(lambda _n: True, traversal="bfs")):
            ids[node] = idx
            dot.node(f"N_{idx:03}", node_name_fn(node))
        for node1, node2 in self.get_edges(as_data=False):
            idx1 = ids[node1]
            idx2 = ids[node2]
            dot.edge(f"N_{idx1:03}", f"N_{idx2:03}")
        return dot


def build_tree_from_string(from_string: str) -> Tree:
    """Build a Tree from a parenthesized string representation.

    Parses a string representation of a tree (such as produced by Tree.as_string())
    and reconstructs the tree structure. The string format uses parentheses to
    indicate parent-child relationships.

    Args:
        from_string: The tree string in parenthesized format, e.g., "(root child1 child2)".

    Returns:
        The reconstructed Tree with all nodes and structure preserved.

    Example:
        ```python
        # Build from string
        tree_str = "(root (child1 leaf1 leaf2) child2)"
        tree = build_tree_from_string(tree_str)

        print(tree.data)              # "root"
        print(tree.num_children)      # 2
        print(tree.children[0].data)  # "child1"

        # Roundtrip: tree -> string -> tree
        original = Tree("root")
        original.add_child("child")
        tree_str = original.as_string()
        reconstructed = build_tree_from_string(tree_str)
        ```
    """
    if not from_string.strip().startswith("("):
        return Tree(from_string)
    data = OneOrMore(nestedExpr()).parseString(from_string)
    return build_tree_from_list(data.as_list())


def build_tree_from_list(data: Union[Any, List]) -> Tree:
    """Build a Tree from nested list representation.

    Recursively constructs a tree from a nested list structure where the first
    element is the parent data and subsequent elements are children (which can
    themselves be nested lists).

    Args:
        data: The tree data as nested lists. Format: [parent, child1, child2, ...]
            where children can be single values or nested lists.

    Returns:
        The root Tree node with all descendants constructed.

    Example:
        ```python
        # Build from nested list
        data = ["root", ["child1", "leaf1", "leaf2"], "child2"]
        tree = build_tree_from_list(data)

        print(tree.data)                    # "root"
        print(tree.children[0].data)        # "child1"
        print(tree.children[0].children[0].data)  # "leaf1"
        ```

    Note:
        This is primarily used internally by build_tree_from_string() but can
        be called directly if you have tree data in nested list format.
    """
    node = None
    if isinstance(data, list) and len(data) > 0:
        node = build_tree_from_list(data[0])
        for cdata in data[1:]:
            node.add_child(build_tree_from_list(cdata))
    else:  # e.g. if isinstance(data, str):
        node = Tree(data)
    return node

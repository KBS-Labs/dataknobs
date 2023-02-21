'''
Implementation of a simple tree data structure.
'''

from __future__ import annotations
from collections import deque
from pyparsing import OneOrMore, nestedExpr
from typing import Any, Callable, List, Tuple, Union


class Tree:
    '''
    Implementation of a simple tree data structure.

    Where the tree is represented as a node containing:
      * (arbitrary) data
      * a list of (ordered) child nodes
      * a single (optional) parent node

    And each tree node is doubly linked from parent to child(ren) and from
    child to parent for efficient traversal both up (to parent) and down
    (to children) the tree.
    '''

    def __init__(
            self,
            data: Any,
            parent: Union[Tree, Any] = None,
            child_pos: int = None,
    ):
        '''
        Initialize a tree (node), optionally adding it to the given parent
        at an optional child position.
        
        :param data: The data to be contained within the node.
        :param parent: The parent node to this node.
        '''
        self._data = data
        self._children = None
        self._parent = None
        if parent is not None:
            if not isinstance(parent, Tree):
                parent = Tree(parent)
            parent.add_child(self, child_pos)

    def __repr__(self) -> str:
        '''
        :return: The string representation of this tree.
        '''
        return self.as_string(delim='  ', multiline=True)

    @property
    def data(self) -> Any:
        '''
        :return: This node's data.
        '''
        return self._data

    @data.setter
    def data(self, data: Any):
        '''
        :return: Set this node's data.
        '''
        self._data = data

    @property
    def children(self) -> List[Tree]:
        '''
        :return: This node's children -- list of child nodes.
        '''
        return self._children

    @property
    def parent(self) -> Tree:
        '''
        :return: This node's parent.
        '''
        return self._parent

    @parent.setter
    def parent(self, parent: Tree):
        '''
        :return: Set this node's parent.
        '''
        self._parent = parent

    @property
    def root(self) -> Tree:
        '''
        :return: The root of this node's tree.
        '''
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    @property
    def sibnum(self) -> int:
        '''
        :return: This node's sibling number (0-based) among its parent's children
        '''
        return self._parent.children.index(self) if self._parent is not None else 0

    @property
    def num_siblings(self) -> int:
        '''
        :return: Get the number of siblings (including self) of this node
        '''
        return self._parent.num_children if self._parent is not None else 1

    @property
    def next_sibling(self) -> Tree:
        '''
        :return: This node's next sibling (or None)
        '''
        result = None
        if self._parent:
            sibs = self._parent.children
            nextsib = sibs.index(self) + 1
            if nextsib < len(sibs):
                result = sibs[nextsib]
        return result

    @property
    def prev_sibling(self) -> Tree:
        '''
        :return: This node's previous sibling (or None)
        '''
        result = None
        if self._parent:
            sibs = self._parent.children
            prevsib = sibs.index(self) - 1
            if prevsib >= 0:
                result = sibs[prevsib]
        return result

    def has_children(self) -> bool:
        '''
        :return: Whether this node has children.
        '''
        return self._children is not None and len(self._children) > 0

    @property
    def num_children(self) -> int:
        '''
        :return: The number of children under this node.
        '''
        return len(self._children) if self._children is not None else 0

    def has_parent(self) -> bool:
        '''
        :return: Whether this not has a parent.

        Note that the "root" of a tree has no parent.
        '''
        return self._parent is not None

    @property
    def depth(self) -> int:
        '''
        :return: The depth of this node in its tree.

        Where the depth is measured as the number of "hops" from the root,
        whose depth is 0, to children until this node is reached.
        '''
        result = 0
        curp = self.parent
        while curp is not None:
            curp = curp.parent
            result += 1
        return result

    def add_child(
            self,
            node_or_data: Union[Tree, Any],
            child_pos: int = None
    ) -> Tree:
        '''
        Add a child node to this node, pruning the child from any other tree.

        :param node_or_data: The node (or data for a new node) to add
        :param child_pos: The (optional) position at which to insert the node.
        :return: the (passed or new) child_node
        '''
        if self._children is None:
            self._children = []
        if isinstance(node_or_data, Tree):
            child = node_or_data
            child.prune()
            if (
                    child_pos is not None and
                    child_pos < len(self._children) and
                    child_pos >= 0
            ):
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
        '''
        Add the child to the parent, using an existing (matching) child or parent.
        If the parent and child already exist, but not as parent and child, the
        child node will be moved to be a child of the parent.

        If neither the parent nor child nodes exist in this tree, the parent
        will be added as a child of this (self) node.

        :param parent_node_or_data: The parent node (or its data)
        :param child_node_or_data: The child node (or its data)
        :return: The (parent-node, child-node) tuple
        '''
        parent = None
        child = None

        if isinstance(parent_node_or_data, Tree):
            parent = parent_node_or_data
            # if it is not in this tree ...
            if len(self.find_nodes(
                lambda node: node == parent,
                include_self = True,
                only_first = True
            )) == 0:
                # ...then add it as a child of self
                self.add_child(parent)
        else:
            # can we find the data in this tree ...
            found = self.find_nodes(
                lambda node: node.data == parent_node_or_data,
                include_self = True,
                only_first = True
            )
            if len(found) > 0:
                parent = found[0]
            else:
                parent = self.add_child(parent_node_or_data)

        if isinstance(child_node_or_data, Tree):
            child = parent.add_chlld(child_node_or_data)
        else:
            # can we find the data in this tree ...
            found = self.find_nodes(
                lambda node: node.data == child_node_or_data,
                include_self = True,
                only_first = True
            )
            if len(found) > 0:
                child = parent.add_child(found[0])
            else:
                child = parent.add_child(child_node_or_data)

        return (parent, child)

    def prune(self) -> Tree:
        '''
        Prune this node from its tree.
        :return: this node's (former) parent.
        '''
        result = self._parent
        if self._parent is not None:
            if self._parent.children is not None:
                self._parent.children.remove(self)
            self._parent = None
        return result

    def find_nodes(
            self,
            accept_node_fn: Callable[[Tree], bool],
            traversal: str = 'dfs',
            include_self: bool = True,
            only_first: bool = False,
            highest_only: bool = False
    ) -> List[Tree]:
        '''
        Find nodes where accept_node_fn(tree) is True,
        using a traversal of:
          'dfs' -- depth first search
          'bfs' -- breadth first search

        :param accept_node_fn: A function returning a boolean for any Tree
            argument; True to select the node or False to skip it
        :param traversal: Either 'dfs' or 'bfs' for depth- or breadth-first
        :param include_self: True to consider this node, False to start with 
            its children
        :param only_first: True to stop after finding the first match
        :param highest_only: True to not collect any nodes under a selected node
        :return: The list of matching/accepted nodes
        '''
        queue, found = deque(), []
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
                if traversal == 'dfs':
                    queue.extendleft(reversed(item.children))
                elif traversal == 'bfs':
                    queue.extend(item.children)
        return found

    def collect_terminal_nodes(
            self,
            accept_node_fn: Callable[[Tree], bool] = None,
            _found: List[Tree] = None
    ):
        '''
        Collect this tree's terminal nodes.

        :param accept_node_fn: Optional function to select which terminal nodes
            to include in the result
        :param _found: The (optional) list to which to add results
        :return: The list of collected nodes
        '''
        if _found is None:
            _found = list()
        if not self._children:
            if accept_node_fn is None or accept_node_fn(self):
                _found.append(self)
        else:
            for child in self._children:
                child.collect_terminal_nodes(accept_node_fn=accept_node_fn, _found=_found)
        return _found

    def get_edges(
            self,
            traversal: str = 'bfs',
            include_self: bool = True,
            as_data: bool = True,
    ) -> List[Tuple[Union[Tree, Any], Union[Tree, Any]]]:
        '''
        Get the edges of this tree, either as Tree nodes or data.

        :param traversal: Either 'dfs' or 'bfs' for depth- or breadth-first
        :param include_self: True to include this node, False to start with
            its children
        :param as_data: If True, then collect node data instead of Tree nodes
        :return: A list of (parent, child) tuples of edge nodes or data
        '''
        queue, result = deque(), []
        if self.children:
            queue.extend(self.children)
        while bool(queue):  # true while length(queue) > 0
            item = queue.popleft()
            if item.parent:
                if item.parent != self or include_self:
                    result.append(
                        (
                            item.parent.data, item.data
                        ) if as_data else (
                            item.parent, item
                        )
                    )
            if item.children:
                if traversal == 'dfs':
                    queue.extendleft(reversed(item.children))
                elif traversal == 'bfs':
                    queue.extend(item.children)
        return result

    def get_path(self) -> List[Tree]:
        '''
        Get the nodes from the root to this node (inclusive).
        '''
        path = deque()
        node = self
        while node is not None:
            path.appendleft(node)
            node = node.parent
        return list(path)

    def is_ancestor(self, other: Tree, self_is_ancestor: bool = False) -> bool:
        '''
        Determine whether this node is an ancestor to the other.
        :param other: The potential descendant of this node
        :param self_is_ancestor: True if this node could be considered to
            be its own ancestor
        :return: True if this node is an ancestor of the other
        '''
        result = False
        parent = other if self_is_ancestor else other.parent
        while parent is not None:
            if parent == self:
                result = True
                break
            parent = parent.parent
        return result

    def find_deepest_common_ancestor(self, other: Tree) -> Tree:
        '''
        Find the deepest common ancestor to self and other.
        :param other: The other node whose shared ancestor with self to find
        :return: The deepest common ancestor to self and other, or None
        '''
        if other is None:
            return None
        if self == other:
            return self
        result = None
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

    def as_string(self, delim: str = ' ', multiline: bool = False) -> str:
        '''
        Get a string representing this tree.
        :param delim: The (indentation) delimiter to use between node data
        :param multiline: True to include newlines in the result
        :param: A string representation of this tree and its descendants
        '''
        result = ''
        if self._children:
            btwn = '\n' if multiline else ''
            result = '(' + str(self.data)
            for child in self._children:
                d = (child.depth if multiline else 1) * delim
                result += btwn + d + child.as_string(delim=delim, multiline=multiline)
            result += ')'
        else:
            result = str(self.data)
        return result

    def get_deepest_left(self) -> Tree:
        '''
        :return: The terminal descendent following the left-most branches
            of this node.
        '''
        node = self
        while node is not None and node.has_children():
            node = node.children[0]
        return node

    def get_deepest_right(self) -> Tree:
        '''
        :return: The terminal descendent following the right-most branches
            of this node.
        '''
        node = self
        while node is not None and node.has_children():
            node = node.children[-1]
        return node


def build_tree_from_string(from_string: str) -> Tree:
    '''
    Build a tree object from the given tree string, e.g., output from
    the "Tree.as_string" method.
    :param from_string: The tree string
    :return: The built Tree
    '''
    if not from_string.strip().startswith('('):
        return Tree(from_string)
    data = OneOrMore(nestedExpr()).parseString(from_string)
    return build_tree_from_list(data.as_list())


def build_tree_from_list(data: Union[Any, List]) -> Tree:
    '''
    Auxiliary to build_tree for recursively building nodes from a list of
    lists.
    :param data: The tree data as a list of lists.
    :return: The root tree node
    '''
    node = None
    if isinstance(data, list) and len(data) > 0:
        node = build_tree_from_list(data[0])
        for cdata in data[1:]:
            node.add_child(build_tree_from_list(cdata))
    else:  # e.g. if isinstance(data, str):
        node = Tree(data)
    return node

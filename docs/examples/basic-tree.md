# Basic Tree Examples

Worked examples for `Tree`, the node-based hierarchy in
[`dataknobs-structures`](../packages/structures/index.md). The
[Tree reference](../packages/structures/tree.md) documents every member; this
page builds a few trees and asks them questions.

## A node is the whole tree

There is no separate container object to create and no root to register.
`Tree` **is** the node, and any node is a complete tree from where it stands —
the root is simply the node whose `parent` is `None`.

```python
from dataknobs_structures import Tree

root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2")
leaf = child1.add_child("leaf")

print(root.as_string())     # (root (child1 leaf) child2)
print(leaf.root.data)       # root
print(leaf.depth)           # 2
```

`add_child` takes either data to wrap or an existing node, and returns the
child, so building downward is a chain of calls on whatever you last got back.

The same structure can be written from the child's side — the constructor
attaches to a parent as it builds:

```python
root = Tree("root")
child1 = Tree("child1", parent=root)
child2 = Tree("child2", parent=root)
leaf = Tree("leaf", parent=child1)

print(root.as_string())     # (root (child1 leaf) child2)
```

Either form takes `child_pos` to insert rather than append:

```python
root = Tree("root")
root.add_child("second")
root.add_child("first", child_pos=0)

print(root.as_string())     # (root first second)
```

## Building from a string

`as_string()` and `build_tree_from_string()` are inverses, which makes a
parenthesized literal the shortest way to write a fixture:

```python
from dataknobs_structures import build_tree_from_string

tree = build_tree_from_string("(root (child1 leaf1 leaf2) child2)")

print(tree.data)                  # root
print(tree.num_children)          # 2
print(tree.children[0].data)      # child1
print(tree.as_string())           # (root (child1 leaf1 leaf2) child2)
```

Round-tripping a tree you built in code goes the same way:

```python
original = Tree("a")
original.add_child("b").add_child("c")

restored = build_tree_from_string(original.as_string())
print(restored.as_string() == original.as_string())   # True
```

Nodes are rebuilt from their *string* form, so a tree carrying non-string data
does not survive the trip unchanged.

## Reading the shape

```python
tree = build_tree_from_string("(root (child1 leaf1 leaf2) child2)")
child1 = tree.children[0]

print(tree.num_children)        # 2
print(tree.has_children())      # True
print(child1.parent.data)       # root
print(child1.depth)             # 1
print(tree.parent)              # None
```

`children` answers with a **tuple**, and its empty answer has two spellings:
`None` if the node has never held a child, and `()` once its children have been
removed. Test `has_children()` — or `num_children`, which counts `0` for both —
rather than comparing against `None`:

```python
leaf = child1.children[0]

print(leaf.children)            # None
print(leaf.has_children())      # False
print(leaf.num_children)        # 0

emptied = Tree("emptied")
emptied.add_child("only").prune()
print(emptied.children)         # ()
print(emptied.has_children())   # False

print([child.data for child in tree.children or ()])   # ['child1', 'child2']
```

Siblings are reachable in both directions, and a node knows its own position:

```python
leaf1, leaf2 = child1.children

print(leaf1.sibnum)                 # 0
print(leaf1.next_sibling.data)      # leaf2
print(leaf2.prev_sibling.data)      # leaf1
print(leaf2.next_sibling)           # None
```

`num_siblings` counts the sibling group, **this node included** — it is the
parent's `num_children`, not the number of other nodes beside it, and it
answers `1` for a root:

```python
print(leaf1.num_siblings)           # 2
print(tree.num_siblings)            # 1
```

## Walking the tree

`find_nodes` takes a predicate and walks depth-first by default:

```python
tree = build_tree_from_string("(root (child1 leaf1 leaf2) child2)")

matched = tree.find_nodes(lambda node: "leaf" in str(node.data))
print([node.data for node in matched])          # ['leaf1', 'leaf2']
```

Four keywords steer it, and they compose:

```python
# Breadth-first instead of depth-first
order = tree.find_nodes(lambda node: True, traversal="bfs")
print([node.data for node in order])
# ['root', 'child1', 'child2', 'leaf1', 'leaf2']

# Skip the node you asked from
below = tree.find_nodes(lambda node: True, include_self=False)
print([node.data for node in below])
# ['child1', 'leaf1', 'leaf2', 'child2']

# Stop at the first hit
first = tree.find_nodes(lambda node: node.data.startswith("child"), only_first=True)
print([node.data for node in first])            # ['child1']

# Do not descend below a match
tops = tree.find_nodes(lambda node: node.has_children(), highest_only=True)
print([node.data for node in tops])             # ['root']
```

`find_nodes(lambda node: True)` is how you enumerate every node — there is no
separate `traverse()` method.

Leaves have a dedicated collector, so finding them does not mean filtering
every node:

```python
print([node.data for node in tree.collect_terminal_nodes()])
# ['leaf1', 'leaf2', 'child2']

print([node.data for node in tree.collect_terminal_nodes(lambda n: "1" in str(n.data))])
# ['leaf1']
```

`get_edges` gives the structure as parent-child pairs, breadth-first by
default, as data or as nodes:

```python
print(tree.get_edges())
# [('root', 'child1'), ('root', 'child2'), ('child1', 'leaf1'), ('child1', 'leaf2')]

pairs = tree.get_edges(as_data=False)
print([(p.data, c.data) for p, c in pairs][:1])
# [('root', 'child1')]
```

## Asking about relationships

```python
tree = build_tree_from_string("(root (left left_child) (right right_child))")
left, right = tree.children
left_child = left.children[0]
right_child = right.children[0]

print([node.data for node in left_child.get_path()])
# ['root', 'left', 'left_child']

print(tree.is_ancestor(left_child))             # True
print(left.is_ancestor(right_child))            # False
print(left.is_ancestor(left))                   # False
print(left.is_ancestor(left, self_is_ancestor=True))    # True

print(left_child.find_deepest_common_ancestor(right_child).data)   # root
print(left_child.find_deepest_common_ancestor(left).data)          # left
```

Two nodes in unrelated trees have no common ancestor, and the answer is `None`
rather than an exception.

## Moving and removing nodes

`prune()` detaches a node from its parent and returns the former parent. The
subtree below it stays intact, so pruning is also how you lift a branch out to
use on its own:

```python
tree = build_tree_from_string("(root (child1 leaf1 leaf2) child2)")
child1 = tree.children[0]

former = child1.prune()
print(former.data)              # root
print(child1.parent)            # None
print(child1.as_string())       # (child1 leaf1 leaf2)
print(tree.as_string())         # (root child2)
```

Re-attaching is just `add_child` — it prunes the node from wherever it was
first, so a move is one call rather than a detach and an add:

```python
child2 = tree.children[0]
child2.add_child(child1)

print(tree.as_string())         # (root (child2 (child1 leaf1 leaf2)))
```

Assigning `parent` does the same thing from the other side, and `None` detaches:

```python
child1.parent = tree
print(tree.as_string())         # (root child2 (child1 leaf1 leaf2))

child1.parent = None
print(child1.parent)            # None
print(tree.as_string())         # (root child2)
```

Both halves of the link are always updated together — there is no way to set
one side and leave the other stale.

## What the writers refuse

A `Tree` is doubly linked, so a node placed under one of its own descendants
becomes reachable from itself, and a walk that follows the links never reaches
an end. Every writer therefore refuses that attachment up front, rather than
accepting it and leaving the traversals to spin:

```python
from dataknobs_common.exceptions import ValidationError

root = Tree("root")
a = root.add_child("a")
b = a.add_child("b")

try:
    b.add_child(root)           # root is b's ancestor
except ValidationError as error:
    print(error)
    # a node cannot be added under itself or under one of its own descendants:
    # that would make it its own ancestor
```

The same refusal covers `node.add_child(node)`, the `parent` setter, and
`add_edge` asked for an edge that inverts one already present. A refused write
changes nothing — the tree is exactly as it was:

```python
print(root.as_string())         # (root (a b))
print(b.children)               # None
```

Moving a descendant *upward* is not a cycle and stays allowed, which is the
common case this guard has to leave alone:

```python
root.add_child(b)               # b was under a; now it is under root
print(root.as_string())         # (root a b)
```

Because the invariant is the writers' to keep, `children` hands out a snapshot
rather than the live list. Mutating what you get back is refused instead of
being silently discarded:

```python
try:
    root.children.append(Tree("smuggled"))
except AttributeError as error:
    print(error)                # 'tuple' object has no attribute 'append'
```

## A worked example: paths as a tree

Putting it together — index a set of filesystem paths, then ask the tree the
questions a caller actually has. Note that the same segment name (`user`,
`log`) can appear under different parents, so lookups are scoped to a node's
own children rather than searched tree-wide.

```python
from dataknobs_structures import Tree


class PathTree:
    """Index slash-separated paths as a tree of segments."""

    def __init__(self, root_label: str = "/"):
        self.root = Tree(root_label)

    def add(self, path: str) -> Tree:
        node = self.root
        for segment in path.strip("/").split("/"):
            node = self._child_named(node, segment) or node.add_child(segment)
        return node

    @staticmethod
    def _child_named(node: Tree, name: str) -> Tree | None:
        for child in node.children or ():
            if child.data == name:
                return child
        return None

    def path_to(self, name: str) -> str | None:
        hits = self.root.find_nodes(lambda n: n.data == name, only_first=True)
        if not hits:
            return None
        return "/".join(str(n.data) for n in hits[0].get_path()[1:])


paths = PathTree()
for path in (
    "/home/user/documents/file.txt",
    "/home/user/documents/report.pdf",
    "/home/user/downloads/image.png",
    "/var/log/system.log",
):
    paths.add(path)

# Every file is a leaf.
print(sorted(str(n.data) for n in paths.root.collect_terminal_nodes()))
# ['file.txt', 'image.png', 'report.pdf', 'system.log']

# Where does one of them live?
print(paths.path_to("report.pdf"))
# home/user/documents/report.pdf

# Which directory holds both documents?
first, second = (
    paths.root.find_nodes(lambda n: n.data == name, only_first=True)[0]
    for name in ("file.txt", "report.pdf")
)
print(first.find_deepest_common_ancestor(second).data)
# documents

# The whole shape, indented.
print(paths.root.as_string(delim="  ", multiline=True))
```

The last call prints:

```text
(/
  (home
    (user
      (documents
        file.txt
        report.pdf)
      (downloads
        image.png)))
  (var
    (log
      system.log)))
```

## See also

- [Tree API reference](../packages/structures/tree.md) — every member, with
  signatures and raises
- [Structures overview](../packages/structures/index.md) — the other structures
  in the package

# Tree API Documentation

The `Tree` class provides a flexible hierarchical data structure for organizing data in tree form.

## Overview

The Tree class implements a doubly-linked tree structure where each node contains:

- Arbitrary data
- A list of ordered child nodes
- A single optional parent node
- Efficient traversal methods

## Class Definition

```python
from dataknobs_structures import Tree
```

## Constructor

```python
Tree(data, parent=None, child_pos=None)
```

Creates a new tree node.

**Parameters:**
- `data` (Any): The data to be contained within the node
- `parent` (Tree | Any, optional): The parent node or data for a new parent node
- `child_pos` (int, optional): Position to insert this node in parent's children

**Example:**
```python
# Create root node
root = Tree("root")

# Create child with parent
child = Tree("child", parent=root)

# Create child at specific position
child2 = Tree("child2", parent=root, child_pos=0)
```

## Properties

### data
```python
@property
def data(self) -> Any
```
Gets or sets the node's data.

**Example:**
```python
node = Tree("initial")
print(node.data)  # "initial"
node.data = "updated"
print(node.data)  # "updated"
```

### children
```python
@property
def children(self) -> Optional[List[Tree]]
```
Returns the list of child nodes, or None if no children.

**Example:**
```python
root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2")
print(len(root.children))  # 2
```

### parent
```python
@property
def parent(self) -> Optional[Tree]
```
Gets or sets the parent node.

**Example:**
```python
child = Tree("child")
parent = Tree("parent")
child.parent = parent
print(child.parent.data)  # "parent"
```

### root
```python
@property
def root(self) -> Tree
```
Returns the root node of the tree.

**Example:**
```python
root = Tree("root")
child = root.add_child("child")
grandchild = child.add_child("grandchild")
print(grandchild.root.data)  # "root"
```

### depth
```python
@property
def depth(self) -> int
```
Returns the depth of the node (0 for root).

**Example:**
```python
root = Tree("root")
child = root.add_child("child")
grandchild = child.add_child("grandchild")
print(root.depth)      # 0
print(child.depth)     # 1
print(grandchild.depth) # 2
```

### num_children
```python
@property
def num_children(self) -> int
```
Returns the number of child nodes.

**Example:**
```python
node = Tree("node")
node.add_child("child1")
node.add_child("child2")
print(node.num_children)  # 2
```

## Tree Navigation Properties

### sibnum
```python
@property
def sibnum(self) -> int
```
Returns this node's position among siblings (0-based).

### num_siblings
```python
@property
def num_siblings(self) -> int
```
Returns the total number of siblings including self.

### next_sibling
```python
@property
def next_sibling(self) -> Optional[Tree]
```
Returns the next sibling node, or None.

### prev_sibling
```python
@property
def prev_sibling(self) -> Optional[Tree]
```
Returns the previous sibling node, or None.

**Example:**
```python
root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2") 
child3 = root.add_child("child3")

print(child2.sibnum)        # 1
print(child2.num_siblings)  # 3
print(child2.prev_sibling.data)  # "child1"
print(child2.next_sibling.data)  # "child3"
```

## Methods

### add_child()
```python
def add_child(self, node_or_data, child_pos=None) -> Tree
```
Adds a child node, pruning it from any other tree first.

**Parameters:**
- `node_or_data` (Tree | Any): Node to add or data for new node
- `child_pos` (int, optional): Position to insert the child

**Returns:** The child node

**Example:**
```python
parent = Tree("parent")
child = parent.add_child("child_data")
print(child.data)  # "child_data"

# Add at specific position
sibling = parent.add_child("sibling", child_pos=0)
print(parent.children[0].data)  # "sibling"
```

### add_edge()
```python
def add_edge(self, parent_node_or_data, child_node_or_data) -> Tuple[Tree, Tree]
```
Adds a parent-child relationship, creating nodes if needed.

**Parameters:**
- `parent_node_or_data` (Tree | Any): Parent node or data
- `child_node_or_data` (Tree | Any): Child node or data

**Returns:** Tuple of (parent_node, child_node)

**Example:**
```python
root = Tree("root")
parent_node, child_node = root.add_edge("parent", "child")
print(parent_node.data)  # "parent"
print(child_node.data)   # "child"
```

### prune()
```python
def prune(self) -> Optional[Tree]
```
Removes this node from its tree, returning the former parent.

**Example:**
```python
root = Tree("root")
child = root.add_child("child")
former_parent = child.prune()
print(former_parent.data)  # "root"
print(root.num_children)   # 0
```

## Tree Traversal and Search

### find_nodes()
```python
def find_nodes(self, accept_node_fn, traversal="dfs", 
               include_self=True, only_first=False, 
               highest_only=False) -> List[Tree]
```
Finds nodes matching the acceptance function.

**Parameters:**
- `accept_node_fn` (Callable[[Tree], bool]): Function that returns True to select node
- `traversal` (str): "dfs" (depth-first) or "bfs" (breadth-first)
- `include_self` (bool): Whether to consider this node
- `only_first` (bool): Stop after finding first match
- `highest_only` (bool): Don't collect nodes under selected nodes

**Example:**
```python
root = Tree({"type": "root", "value": 10})
child1 = root.add_child({"type": "child", "value": 5})
child2 = root.add_child({"type": "child", "value": 15})

# Find nodes with value > 10
high_value_nodes = root.find_nodes(
    lambda node: node.data.get("value", 0) > 10
)
print(len(high_value_nodes))  # 1 (child2)
```

### collect_terminal_nodes()
```python
def collect_terminal_nodes(self, accept_node_fn=None) -> List[Tree]
```
Collects all leaf nodes (nodes without children).

**Example:**
```python
root = Tree("root")
branch1 = root.add_child("branch1")
leaf1 = branch1.add_child("leaf1")
leaf2 = root.add_child("leaf2")

leaves = root.collect_terminal_nodes()
print([node.data for node in leaves])  # ["leaf1", "leaf2"]
```

### get_edges()
```python
def get_edges(self, traversal="bfs", include_self=True, 
              as_data=True) -> List[Tuple[Union[Tree, Any], Union[Tree, Any]]]
```
Gets parent-child edge relationships.

**Parameters:**
- `traversal` (str): "dfs" or "bfs"
- `include_self` (bool): Include edges from this node
- `as_data` (bool): Return data instead of Tree objects

**Example:**
```python
root = Tree("A")
child1 = root.add_child("B")
child2 = root.add_child("C")

edges = root.get_edges(as_data=True)
print(edges)  # [("A", "B"), ("A", "C")]
```

### get_path()
```python
def get_path(self) -> List[Tree]
```
Returns path from root to this node (inclusive).

**Example:**
```python
root = Tree("root")
child = root.add_child("child")
grandchild = child.add_child("grandchild")

path = grandchild.get_path()
path_data = [node.data for node in path]
print(path_data)  # ["root", "child", "grandchild"]
```

## Relationship Methods

### is_ancestor()
```python
def is_ancestor(self, other, self_is_ancestor=False) -> bool
```
Determines if this node is an ancestor of another.

**Example:**
```python
root = Tree("root")
child = root.add_child("child")
grandchild = child.add_child("grandchild")

print(root.is_ancestor(grandchild))  # True
print(child.is_ancestor(root))       # False
```

### find_deepest_common_ancestor()
```python
def find_deepest_common_ancestor(self, other) -> Optional[Tree]
```
Finds the deepest common ancestor with another node.

**Example:**
```python
root = Tree("root")
branch1 = root.add_child("branch1")
branch2 = root.add_child("branch2")
leaf1 = branch1.add_child("leaf1")
leaf2 = branch2.add_child("leaf2")

common = leaf1.find_deepest_common_ancestor(leaf2)
print(common.data)  # "root"
```

## Tree Utilities

### has_children()
```python
def has_children(self) -> bool
```
Returns True if the node has children.

### has_parent()
```python
def has_parent(self) -> bool
```
Returns True if the node has a parent.

### get_deepest_left()
```python
def get_deepest_left(self) -> Tree
```
Returns the leftmost terminal descendant.

### get_deepest_right()
```python
def get_deepest_right(self) -> Tree
```
Returns the rightmost terminal descendant.

## String Representation

### as_string()
```python
def as_string(self, delim=" ", multiline=False) -> str
```
Returns a string representation of the tree.

**Parameters:**
- `delim` (str): Delimiter for indentation
- `multiline` (bool): Whether to use newlines

**Example:**
```python
root = Tree("A")
child1 = root.add_child("B")
child2 = root.add_child("C")
child1.add_child("D")

print(root.as_string(multiline=True))
# (A
#  B
#   D
#  C)
```

## Visualization

### build_dot()
```python
def build_dot(self, node_name_fn=None, **kwargs) -> graphviz.Digraph
```
Creates a Graphviz visualization of the tree.

**Parameters:**
- `node_name_fn` (Callable[[Tree], str], optional): Function to generate node labels
- `**kwargs`: Additional arguments for graphviz.Digraph

**Example:**
```python
root = Tree({"name": "Root", "id": 1})
child = root.add_child({"name": "Child", "id": 2})

# Create visualization
dot = root.build_dot(
    node_name_fn=lambda n: n.data["name"],
    format='png'
)

# Save as image
dot.render('/tmp/tree', format='png')
```

## Utility Functions

### build_tree_from_string()
```python
def build_tree_from_string(from_string: str) -> Tree
```
Reconstructs a tree from its string representation.

**Example:**
```python
tree_str = "(A B (C D) E)"
tree = build_tree_from_string(tree_str)
print(tree.data)  # "A"
print(len(tree.children))  # 3
```

## Error Handling

Common error scenarios and how to handle them:

```python
# Handling empty trees
tree = Tree(None)
if tree.data is None:
    print("Tree has no data")

# Checking for children before accessing
if tree.has_children():
    first_child = tree.children[0]
else:
    print("No children")

# Safe parent access
if tree.has_parent():
    parent_data = tree.parent.data
else:
    print("This is the root node")
```

## Performance Considerations

- Tree traversal methods are optimized for different use cases
- Use `only_first=True` when you only need one match
- Consider `highest_only=True` to avoid redundant subtree searches
- Large trees may benefit from breadth-first search for shallow targets

## Integration Examples

### With JSON Processing
```python
from dataknobs_utils import json_utils

# Build tree from JSON structure
data = {"name": "root", "children": [{"name": "child1"}, {"name": "child2"}]}
root = Tree(data["name"])
for child_data in data["children"]:
    root.add_child(child_data["name"])
```

### With Text Processing
```python
from dataknobs_xization import normalize

# Normalize tree data
def normalize_tree_data(node):
    if isinstance(node.data, str):
        node.data = normalize.basic_normalization_fn(node.data)

root = Tree("Hello WORLD!")
root.find_nodes(lambda n: True)  # Apply to all nodes
for node in root.find_nodes(lambda n: True):
    normalize_tree_data(node)
```
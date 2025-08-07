# Basic Tree Examples

Examples of using the Tree data structure from dataknobs-structures.

## Creating Trees

### From String

```python
from dataknobs_structures import build_tree_from_string

# Simple tree
tree_str = """root -> child1, child2
child1 -> leaf1, leaf2
child2 -> leaf3"""

tree = build_tree_from_string(tree_str)
```

### Manual Construction

```python
from dataknobs_structures import Tree

tree = Tree()
root = tree.add_root("root")
child1 = tree.add_child(root, "child1")
child2 = tree.add_child(root, "child2")
leaf = tree.add_child(child1, "leaf")
```

## Tree Traversal

```python
# Depth-first traversal
for node in tree.traverse():
    print(f"{' ' * node.level}{node.value}")

# Breadth-first traversal
from collections import deque

queue = deque([tree.root])
while queue:
    node = queue.popleft()
    print(node.value)
    queue.extend(node.children)
```

## Tree Search

```python
# Find node by value
def find_node(tree, value):
    for node in tree.traverse():
        if node.value == value:
            return node
    return None

# Find all leaves
def get_leaves(tree):
    return [node for node in tree.traverse() if not node.children]
```

## Tree Manipulation

```python
# Prune tree at depth
def prune_at_depth(tree, max_depth):
    for node in tree.traverse():
        if node.level == max_depth:
            node.children.clear()

# Copy subtree
def copy_subtree(node):
    new_tree = Tree()
    new_root = new_tree.add_root(node.value)
    
    def copy_children(src, dst):
        for child in src.children:
            new_child = new_tree.add_child(dst, child.value)
            copy_children(child, new_child)
    
    copy_children(node, new_root)
    return new_tree
```

## Complete Example

```python
from dataknobs_structures import Tree, build_tree_from_string

class FileSystemTree:
    """Represent filesystem as a tree."""
    
    def __init__(self):
        self.tree = Tree()
        self.root = self.tree.add_root("/")
    
    def add_path(self, path):
        parts = path.strip("/").split("/")
        current = self.root
        
        for part in parts:
            # Find or create child
            child = self.find_child(current, part)
            if not child:
                child = self.tree.add_child(current, part)
            current = child
    
    def find_child(self, node, name):
        for child in node.children:
            if child.value == name:
                return child
        return None
    
    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.root
        
        print(" " * indent + node.value)
        for child in sorted(node.children, key=lambda n: n.value):
            self.print_tree(child, indent + 2)

# Usage
fs = FileSystemTree()
fs.add_path("/home/user/documents/file.txt")
fs.add_path("/home/user/downloads/image.png")
fs.add_path("/home/user/documents/report.pdf")
fs.add_path("/var/log/system.log")

fs.print_tree()
# Output:
# /
#   home
#     user
#       documents
#         file.txt
#         report.pdf
#       downloads
#         image.png
#   var
#     log
#       system.log
```

import dataknobs_structures.tree as dk_tree


def test_basics():
    tree = make_simple_tree()
    assert tree.data == "a"
    tree.data = "z"
    assert tree.data == "z"

    assert tree.has_children() == True
    assert tree.num_children == 2
    assert [node.data for node in tree.children] == ["b", "c"]
    assert ["z"] * tree.num_children == [node.parent.data for node in tree.children]

    assert tree.has_parent() == False
    assert tree.depth == 0
    assert tree.__repr__() is not None

    for idx in range(tree.num_children):
        assert tree.children[idx].sibnum == idx
        assert tree.children[idx].depth == 1

    assert tree.children[0].prev_sibling is None
    assert tree.children[0].next_sibling == tree.children[1]
    assert tree.children[1].prev_sibling == tree.children[0]
    assert tree.children[1].next_sibling is None


def test_find_nodes():
    tree = make_simple_tree()
    contents = []

    def accept_fn(node):
        contents.append(node.data)
        return True

    # bfs w/include_self=True
    nodes1 = tree.find_nodes(accept_fn, traversal="bfs", include_self=True)
    assert len(nodes1) == 9
    assert contents == ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    # bfs w/include_self=False
    contents = []
    nodes2 = tree.find_nodes(accept_fn, traversal="bfs", include_self=False)
    assert len(nodes2) == 8
    assert contents == ["b", "c", "d", "e", "f", "g", "h", "i"]

    # dfs w/include_self=True
    contents = []
    nodes3 = tree.find_nodes(accept_fn, traversal="dfs", include_self=True)
    assert len(nodes3) == 9
    assert contents == ["a", "b", "d", "e", "h", "c", "f", "g", "i"]

    # dfs w/include_self=False, only_first=True
    contents = []
    nodes3 = tree.find_nodes(accept_fn, traversal="dfs", include_self=False, only_first=True)
    assert len(nodes3) == 1
    assert contents == ["b"]


def test_collect_all_terminal_nodes():
    tree = make_simple_tree()
    all_leaf_nodes = tree.collect_terminal_nodes()
    assert len(all_leaf_nodes) == 4
    assert [node.data for node in all_leaf_nodes] == ["d", "h", "f", "i"]
    assert [node.depth for node in all_leaf_nodes] == [2, 3, 2, 3]
    assert all_leaf_nodes[0].root == tree


def test_collect_some_terminal_nodes():
    tree = make_simple_tree()
    some_leaf_nodes = tree.collect_terminal_nodes(accept_node_fn=lambda x: x.data == "h")
    print(f"collected: {[node.data for node in some_leaf_nodes]}")
    assert len(some_leaf_nodes) == 1
    assert [node.data for node in some_leaf_nodes] == ["h"]


def test_path_and_ancestor():
    tree = make_simple_tree()
    node = tree.find_nodes(lambda x: x.data == "h", only_first=True)[0]
    assert [node.data for node in node.get_path()] == ["a", "b", "e", "h"]
    assert tree.is_ancestor(node) is True
    assert node.is_ancestor(node, self_is_ancestor=False) is False
    assert node.is_ancestor(node, self_is_ancestor=True) is True
    assert node.parent.is_ancestor(node)


def test_deepest_common_ancestor():
    tree = make_simple_tree()
    h = tree.find_nodes(lambda x: x.data == "h")[0]
    i = tree.find_nodes(lambda x: x.data == "i")[0]
    d = tree.find_nodes(lambda x: x.data == "d")[0]
    b = tree.find_nodes(lambda x: x.data == "b")[0]
    assert h.find_deepest_common_ancestor(None) is None
    assert h.find_deepest_common_ancestor(i) == tree
    assert i.find_deepest_common_ancestor(h) == tree
    assert h.find_deepest_common_ancestor(d) == b
    assert d.find_deepest_common_ancestor(h) == b
    assert h.find_deepest_common_ancestor(b) == b
    assert b.find_deepest_common_ancestor(h) == b
    assert h.find_deepest_common_ancestor(h) == h


def test_prune():
    tree = make_simple_tree()
    b = tree.find_nodes(lambda x: x.data == "b")[0]
    c = tree.find_nodes(lambda x: x.data == "c")[0]
    assert b.sibnum == 0
    assert c.sibnum == 1
    assert b.next_sibling == c
    assert c.prev_sibling == b
    assert b.root == tree
    assert b.prune() == tree
    assert b.root == b
    # No longer find b in tree
    b2 = tree.find_nodes(lambda x: x.data == "b")
    assert len(b2) == 0
    # No longer find h (descendant of b) in tree
    h2 = tree.find_nodes(lambda x: x.data == "h")
    assert len(h2) == 0
    # still find i in pruned tree
    i = tree.find_nodes(lambda x: x.data == "i")[0]
    assert i.data == "i"
    # Still find h in pruned b
    h = b.find_nodes(lambda x: x.data == "h")[0]
    assert h.data == "h"
    # pruning root has no effect
    assert tree.prune() is None
    # c's sibnum is now 0
    assert b.next_sibling is None
    assert c.prev_sibling is None
    assert c.sibnum == 0


def test_insert():
    tree = make_simple_tree()
    j = dk_tree.Tree("j")
    tree.add_child(j, 1)
    assert [node.data for node in tree.children] == ["b", "j", "c"]
    tree.add_child("k", 0)
    assert [node.data for node in tree.children] == ["k", "b", "j", "c"]


def test_get_deepest_left_and_right():
    tree = make_simple_tree()
    assert tree.get_deepest_left().data == "d"
    assert tree.get_deepest_right().data == "i"


def make_simple_tree():
    # (a (b d (e h)) (c f (g i)))
    a = dk_tree.Tree("a")
    b = a.add_child("b")
    c = a.add_child("c")
    d = b.add_child("d")
    e = b.add_child("e")
    f = c.add_child("f")
    g = c.add_child("g")
    h = e.add_child("h")
    i = g.add_child("i")
    return a


def test_edges():
    # (root (a (b d e) (c f)))
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("b", "e"), ("c", "f")]
    x = dk_tree.Tree("root")
    for parent, child in edges:
        x.add_edge(parent, child)
    assert x.as_string() == "(root (a (b d e) (c f)))"
    assert x.get_edges() == [("root", "a")] + edges
    assert x.get_edges(include_self=False) == edges
    assert x.get_edges(include_self=False, traversal="dfs") == [
        ("a", "b"),
        ("b", "d"),
        ("b", "e"),
        ("a", "c"),
        ("c", "f"),
    ]
    assert [
        (parent.data, child.data)
        for (parent, child) in x.get_edges(include_self=False, traversal="bfs", as_data=False)
    ] == edges


def test_build_tree1():
    x = dk_tree.build_tree_from_string("(root (a (b d e) (c f)))")
    assert x.as_string() == "(root (a (b d e) (c f)))"


def test_build_tree2():
    x = dk_tree.Tree(0)
    assert dk_tree.build_tree_from_string(x.as_string()).as_string() == x.as_string()

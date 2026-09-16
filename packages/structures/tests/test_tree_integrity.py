"""The write surface cannot put a tree into a state its own readers hang on.

Every traversal this package ships assumes one thing about the structure it is
given: that following ``children`` down, or ``parent`` up, terminates. Twelve
of them make that assumption and none of them check it, so a cyclic tree does
not produce a wrong answer -- it produces no answer, and on ten of the twelve
it does so by spinning rather than raising.

The assumption is cheap to keep and impossible to verify at the point of use,
so it is kept where the structure is written instead. These tests are about
the four writers, not the twelve readers: after them there is no public call
that builds the input the readers cannot survive.

**Why none of these drives a traversal over a cycle.** The natural test here --
assert that ``find_nodes`` terminates -- cannot be written against the unfixed
code, because against the unfixed code it *is* the hang. The red half of each
pair below is a refusal that either happens or does not, which fails in
milliseconds either way.
"""

import pytest
from dataknobs_common.exceptions import ValidationError

import dataknobs_structures.tree as dk_tree


def a_chain():
    """``root -> a -> b``, the shortest tree with a grandchild."""
    root = dk_tree.Tree("root")
    a = root.add_child("a")
    b = a.add_child("b")
    return root, a, b


# --------------------------------------------------------------- the refusals


def test_a_node_cannot_be_added_under_its_own_descendant():
    root, _a, b = a_chain()

    with pytest.raises(ValidationError):
        b.add_child(root)


def test_a_node_cannot_be_added_under_itself():
    _root, a, _b = a_chain()

    with pytest.raises(ValidationError):
        a.add_child(a)


def test_the_parent_setter_refuses_the_same_cycle():
    """One assignment, no method call -- and it used to be the cheapest way in."""
    root, _a, b = a_chain()

    with pytest.raises(ValidationError):
        root.parent = b


def test_add_edge_refuses_to_invert_an_existing_edge():
    """``add_edge`` reuses nodes it finds, so it can re-parent an ancestor.

    Before the guard this both built the cycle and emptied the tree: ``a`` was
    pruned from ``root`` on its way under ``b``, leaving ``root`` childless and
    ``a`` and ``b`` pointing at each other.
    """
    root = dk_tree.Tree("root")
    root.add_edge("a", "b")

    with pytest.raises(ValidationError):
        root.add_edge("b", "a")

    assert root.as_string() == "(root (a b))"


def test_children_cannot_be_mutated_in_place():
    """The list used to be live, so ``append`` bypassed every writer's guard.

    A tuple refuses at the call site rather than accepting a write nothing
    honours, which is the difference between a break a caller sees and a
    corruption they do not.
    """
    root, _a, _b = a_chain()
    orphan = dk_tree.Tree("orphan")

    with pytest.raises(AttributeError):
        root.children.append(orphan)


def test_a_refused_add_leaves_the_tree_exactly_as_it_was():
    """The guard runs before the prune, so a refusal is not half a move."""
    root, a, b = a_chain()
    before = root.as_string()

    with pytest.raises(ValidationError):
        b.add_child(root)

    assert root.as_string() == before
    assert a.parent is root
    assert b.parent is a


def test_a_tree_already_made_cyclic_is_reported_rather_than_walked():
    """The guard's own walk terminates on the input it exists to refuse.

    Reachable only through the private attributes now, but an object pickled
    before this guard shipped is the case that matters: the walk must answer,
    and ``is_ancestor`` -- the obvious predicate -- would not.
    """
    root, _a, b = a_chain()
    root._parent = b  # what the parent setter used to do

    with pytest.raises(ValidationError):
        b.add_child(dk_tree.Tree("anything"))


# ------------------------------------------------- what stays possible


def test_a_descendant_may_still_be_moved_up():
    """Hoisting a subtree is a legitimate move and the one consumers make.

    A conversation compaction re-parents the retained tail directly under the
    branch point. That child is a *descendant* of its new parent, not an
    ancestor of it, so the guard has nothing to say about it.
    """
    root = dk_tree.Tree("root")
    user = root.add_child("user")
    x = user.add_child("x")
    y = x.add_child("y")

    user.add_child(y)

    assert root.as_string() == "(root (user x y))"
    assert y.parent is user


def test_a_detached_subtree_may_be_re_rooted():
    """Inverting a tree is permitted -- it just has to be detached first."""
    root, a, _b = a_chain()

    a.prune()
    a.add_child(root)

    assert a.parent is None
    assert root.parent is a
    assert a.as_string() == "(a b root)"


def test_the_parent_setter_makes_the_two_halves_agree():
    """The published example, which used to produce a one-sided link.

    ``docs/packages/structures/tree.md`` teaches this assignment and asserts
    only the half that already worked; the parent did not gain the child.
    """
    child = dk_tree.Tree("child")
    parent = dk_tree.Tree("parent")

    child.parent = parent

    assert child.parent.data == "parent"
    assert parent.children == (child,)


def test_setting_a_parent_to_none_detaches_both_halves():
    root, a, _b = a_chain()

    a.parent = None

    assert a.parent is None
    assert root.children == ()
    assert a.as_string() == "(a b)"

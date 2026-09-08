"""Minting ids for a tree nobody keyed: the walk, the slug, and the refusal.

Two doors read the same nested shape -- ``kind: nested`` in an ontology
config, and :meth:`~dataknobs_common.hierarchy.MappingHierarchy.from_nested`
over a tree held in memory -- and they must mint the **same** id for the same
node. A consumer who loads one document through the first and holds the same
tree through the second would otherwise get two sets of keys for one tree,
which is the drift the minting rule exists to prevent, arriving through the
door it was not watching.

So the traversal, the slug and the collision refusal are written once, here,
and both doors call them. A private sibling rather than something inside
``hierarchy.py``, for the reason :mod:`dataknobs_common._walk_core` gives: a
private name reached out of a *public* module is a boundary crossed, and the
same name reached out of a private core is two consumers of one core.

The direction is legal and adds nothing. ``ontology.loader`` already reaches
``hierarchy`` transitively -- loader to values to taxonomy to hierarchy -- and
the standing rule is one-way the other way: the general module imports nothing
from ``ontology/``, never the reverse. A private core below both is below that
rule entirely.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from dataknobs_common.exceptions import ValidationError

#: Everything a slug collapses. ``/`` is absent deliberately -- see :func:`_slug`.
_SLUG_SEPARATORS = re.compile(r"[^a-z0-9/]+")


def _slug(text: str) -> str:
    """A stable id from a path: lower-cased, with separators collapsed.

    ``/`` survives because it is the path separator and carries the structure;
    everything else non-alphanumeric becomes a single ``-``.
    """
    collapsed = _SLUG_SEPARATORS.sub("-", text.lower()).strip("-")
    return "/".join(part.strip("-") for part in collapsed.split("/"))


def _walk_tree(
    tree: Any, child_key: str, name_key: str
) -> list[tuple[Mapping[str, Any], list[str]]]:
    """Every node of a nested tree, paired with its path from the root.

    Depth-first and iterative. A tree a person maintains is not deep, but a
    recursive walk would turn a malformed self-referential document into a
    stack overflow rather than a refusal.

    A single mapping and a list of them are both accepted: a forest is an
    ordinary shape for a hand-maintained file, not a malformed one. A node that
    is not a mapping is skipped rather than refused, which keeps the traversal
    usable by a caller that only wants to *read* a tree -- the refusals belong
    to whoever is minting from it.
    """
    if tree is None:
        return []
    roots = tree if isinstance(tree, list) else [tree]
    found: list[tuple[Mapping[str, Any], list[str]]] = []
    stack: list[tuple[Any, list[str]]] = [(node, []) for node in reversed(roots)]
    while stack:
        node, prefix = stack.pop()
        if not isinstance(node, Mapping):
            continue
        path = [*prefix, str(node.get(name_key, node.get("id", "")))]
        found.append((node, path))
        children = node.get(child_key) or []
        stack.extend((child, path) for child in reversed(children))
    return found


@dataclass(frozen=True)
class _MintedNode:
    """One node of a nested tree, with the id and edge the path minted."""

    #: The node as written, for whatever else the caller reads off it.
    node: Mapping[str, Any]
    #: The path, joined. This is the *name*: it changes when the file is edited.
    name: str
    #: The slug of :attr:`name`. This is the id, and it does not change.
    id: str
    #: The slug of the parent's path, or ``None`` for a root.
    parent_id: str | None


def _mint_tree(
    tree: Any,
    *,
    child_key: str = "children",
    name_key: str = "name",
    source: str | None = None,
    claimed: MutableMapping[str, str] | None = None,
) -> list[_MintedNode]:
    """Mint an id per node of a nested tree, refusing a collision.

    A hand-maintained tree has no id field anywhere: a node *is* its path. But
    a path is a name, and a name renames itself when someone edits the file --
    which would re-key every descendant of a renamed interior node at once. So
    the id is a slug of the path taken at first load and the path becomes the
    name, which means renaming a node changes what it is called and not what it
    is.

    Args:
        tree: A mapping, or a list of them for a forest. ``None`` mints nothing.
        child_key: Where a node keeps its children.
        name_key: Where a node keeps its name. A node with neither that key nor
            ``id`` contributes an empty path segment rather than being dropped,
            so the collision below is what reports it.
        source: Named in the refusal, when the tree came from somewhere with a
            name. ``None`` reads as *this tree*, which is what an in-memory
            caller has.
        claimed: Ids already minted, mapped to the name that minted each. Pass
            one across several calls to refuse a collision *between* trees as
            well as within one; omit it and each call starts clean. **Mutated**,
            which is the point of passing it.

    Returns:
        One :class:`_MintedNode` per node, in depth-first order.

    Raises:
        ValidationError: On two paths that slug to one id, naming both paths and
            the id they share.
    """
    where = f"source {source!r}" if source is not None else "this tree"
    seen: MutableMapping[str, str] = {} if claimed is None else claimed
    minted: list[_MintedNode] = []
    for node, path in _walk_tree(tree, child_key, name_key):
        name = "/".join(path)
        node_id = _slug(name)
        if node_id in seen:
            raise ValidationError(
                f"tree node {name!r} in {where} mints id {node_id!r}, which "
                f"{seen[node_id]!r} already minted. The slug collapses "
                f"punctuation and case, so two nodes a person reads as "
                f"different can name one entity",
                context={"source_id": source, "id": node_id, "path": name},
            )
        seen[node_id] = name
        minted.append(
            _MintedNode(
                node=node,
                name=name,
                id=node_id,
                parent_id=_slug("/".join(path[:-1])) if len(path) > 1 else None,
            )
        )
    return minted

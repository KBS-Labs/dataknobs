# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The keys an indexed row carries to say what it is about.

Four string constants, and nothing else. A vector row written from a
vocabulary needs to say which vocabulary, which axis, which node, and which
surface forms it stood for --- and every one of those is a key spelled into
somebody else's corpus, so it is published here rather than written as a
literal at each end. A key spelled at each end is a reader reaching for the
wrong one, and a key nothing wrote reads as absent, which every reader treats
as *unknown, assume current*.

**The family is split across two modules, and this docstring is half of what
pays for that.** The fifth key of the same row --- ``MODEL_NAME_KEY``, the
model that produced the vector --- lives in
``dataknobs_data.vector.content``, because it is about an **embedder** rather
than about identity, it already ships, and five files import it. Moving it
would be a migration. So the two halves name each other, in both directions:
``vector/content.py``'s module docstring names this one, and this one names
it. A reader reaching for one module finds the other.

**The values are namespaced and underscore-separated**, because they are
written into a store we do not own and a family spelled two ways is the defect
a published family exists to prevent. Each value is its constant's name less
``_KEY``, lower-cased.

Reading these back --- turning a row's mapping into tags, and refusing a
half-written one --- is not here. This module is the keys; the read is its own
construct against this module, and a leg that needs to write a row needs only
what is below.
"""

from __future__ import annotations

__all__ = [
    "ALIAS_FORMS_KEY",
    "NODE_ID_KEY",
    "ONTOLOGY_ID_KEY",
    "TAXONOMY_ID_KEY",
]

#: Which vocabulary this row's id belongs to.
#:
#: The one key every row written from a vocabulary carries, whatever else it
#: does: an id without its namespace is a local id, and a reader that cannot
#: tell the two apart resolves a hit to nothing while reporting *not found*.
ONTOLOGY_ID_KEY = "dk_ontology_id"

#: Which structure axis of that vocabulary this row is placed on.
TAXONOMY_ID_KEY = "dk_taxonomy_id"

#: Which node or nodes of that axis this row is about.
#:
#: **List-valued**, because a row may be about several nodes at once. A reader
#: takes a bare string as one node rather than as its characters.
NODE_ID_KEY = "dk_node_id"

#: The surface forms this row's entity is known by.
#:
#: **List-valued.** Written by a source that yields entities, and read by the
#: decorator that turns one entity into one indexed row per form --- one key,
#: two ends, which is why it is a constant rather than a literal at either.
ALIAS_FORMS_KEY = "dk_alias_forms"

"""The worked call site from the content-tags guide, executed as written.

This module is not collected by pytest. ``test_worked_content_tags_call_site.py``
runs it with :func:`runpy.run_path`, in a directory holding the vocabulary the
guide publishes beside it, and asserts on what it leaves bound.

**Everything below the blank line after this docstring is the guide's fence,
character for character.** The test asserts that, in both directions, so
editing either copy alone turns the suite red rather than letting the page and
the code drift.

So this file is the one place where a formatter, a linter and a type checker
read the published page. That is the point of executing a copy rather than
transcribing one, and it is not free: the fence is written the way
``ruff format`` writes it, because a formatting finding here is a finding
against a documentation page that no ``# noqa`` can answer -- the directive
would render on the page. The one bare-attribute rule that has no such
spelling, ``B018``, is waived for this file in the root config with its reason.

Do not add to it. An assertion that is not in the fence is an assertion the
reader of the guide never sees; the one ``assert`` line that *is* in the fence
is there because ``entity()`` returns ``Entity | None`` and the page should say
so rather than let a reader assume otherwise.
"""

from pathlib import Path
from typing import Any

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    load_ontology,
    qualify,
    read_node_tags,
    read_node_tags_many,
)

# What the consumer's own retrieval handed back. This page wrote none of it and
# names no corpus: four rows out of somebody else's store, read back.
hits: list[dict[str, Any]] = [
    {
        "text": "Spay recovery in retrievers is usually uneventful.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "species",
            NODE_ID_KEY: ["golden_retriever", "beagle"],
            "model_name": "all-MiniLM-L6-v2",
        },
    },
    {
        "text": "Invoice 2291 paid in full.",
        "metadata": {"invoice_id": "2291"},
    },
    {
        "text": "Double coats need more grooming in spring.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "coat",  # an axis this vocabulary does not declare
            NODE_ID_KEY: ["double_coat"],
        },
    },
    {
        "text": "Vaccination schedule, revised.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",  # a writer got this one wrong:
            NODE_ID_KEY: ["beagle"],  # no dk_taxonomy_id
        },
    },
]

# (1) what is this row about? -- a pure read of what the row declares. No store,
#     no embedder, no event loop, and no vocabulary either.
tags = read_node_tags(hits[0]["metadata"])
[(t.ontology_id, t.taxonomy_id, t.node_id) for t in tags]
# [("mammals", "species", "golden_retriever"), ("mammals", "species", "beagle")]

read_node_tags(hits[1]["metadata"])  # () -- a row that touched no vocabulary.
# NOT an error, and NOT the same answer as a row whose tag is half written.

# (2) the whole hit set at once, because one row a writer got wrong must not
#     take the readable ones with it
reading = read_node_tags_many([hit["metadata"] for hit in hits])
len(reading.tags)  # 4 -- one entry per position, in the order asked
[len(row) for row in reading.tags]  # [2, 0, 1, 0] -- () for BOTH kinds of nothing
[bad.row for bad in reading.malformed]  # [3] -- which of them was refused, and:
reading.malformed[0].reason
# "a content row carrying 'dk_ontology_id', 'dk_node_id' declares no
#  'dk_taxonomy_id': a row carrying any of the three tag keys carries all three"

# The refusal is a member rather than the stance -- ask for it when you want it.
try:
    reading.require_readable()
except ValidationError as refusal:
    refused = str(refusal)  # names EVERY malformed position, once

# (3) name the vocabulary, and keep the tags it is about. The filter is yours: a
#     name off a row is not a name you typed.
onto = load_ontology(Path("mammals.yaml"))
mine = [
    t
    for row in reading.tags
    for t in row
    if t.ontology_id == onto.id and t.taxonomy_id in onto.taxonomies
]
[t.node_id for t in mine]  # ["golden_retriever", "beagle"] -- the coat row named
# an axis this vocabulary does not declare, and dropping it is a branch you took
# rather than a refusal you caught.

# (4) the tie-back-in: the id the row carries, judged by the vocabulary
axis = onto.taxonomy(mine[0].taxonomy_id)
here = axis.at(onto.localize(mine[0].qualified_id))
here.node  # "golden_retriever"
here.exists()  # True -- the tag names a node this vocabulary actually carries
placed = here.entity()  # -> Entity | None
assert placed is not None  # an id that misses is a typo, not a result
placed.name  # "Golden Retriever"
here.ancestors()  # retriever, dog, mammal

# (5) and the refusal, which is why the PAIR travels rather than a bare id
try:
    onto.localize(qualify("procedures", "spay"))
except ValidationError as refusal:
    foreign = str(refusal)  # names both ontologies

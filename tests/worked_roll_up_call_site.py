"""The worked call site from the roll-up guide, executed as written.

This module is not collected by pytest. ``test_worked_roll_up_call_site.py``
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
reader of the guide never sees, and this page's fence carries none: every line
of it is a call whose value a reader is being shown.
"""

from pathlib import Path
from typing import Any

from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    Granularity,
    load_ontology,
    ontology_support,
    read_node_tags_many,
    roll_up,
)

# What the consumer's own retrieval handed back. This page wrote none of it and
# names no corpus: five rows out of somebody else's store, read back. A roll-up
# over one row is an accessor -- the operation only exists at three or more.
hits: list[dict[str, Any]] = [
    {
        "text": "Spay recovery in retrievers is usually uneventful.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "species",
            NODE_ID_KEY: ["golden_retriever", "beagle"],
        },
    },
    {
        "text": "Invoice 2291 paid in full.",
        "metadata": {"invoice_id": "2291"},  # touched no vocabulary at all
    },
    {
        "text": "Most dogs tolerate the anaesthetic well.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "species",
            NODE_ID_KEY: ["dog"],
        },
    },
    {
        "text": "Goldies are prone to hip dysplasia.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "species",
            NODE_ID_KEY: ["golden_retriever"],
        },
    },
    {
        "text": "Wolfhound intake, 14 March.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "species",
            NODE_ID_KEY: ["wolfhound"],  # a node this vocabulary does not carry
        },
    },
]

# (1) read the rows. One entry per position, in the order asked, and no
#     vocabulary is loaded or held to do it.
reading = read_node_tags_many([hit["metadata"] for hit in hits])
[len(row) for row in reading.tags]  # [2, 0, 1, 1, 1]

# (2) which vocabularies is this page of results even about? Ranked by how many
#     rows named each, so the answer is also a measure of how much of it.
[(s.ontology_id, s.rows) for s in ontology_support(reading.tags)]
# [("mammals", (0, 2, 3, 4))] -- row 1 named none

# (3) load the one they named, and roll the rows up onto one of its axes
onto = load_ontology(Path("mammals.yaml"))
axis = onto.taxonomy("species")
answer = roll_up(onto, "species", reading.tags)

[(s.node_id, s.rows, s.above) for s in answer.supported]
# [("golden_retriever", (0, 3), ("dog",)),   <- what the rows said, in the order
#  ("beagle",           (0,),   ("dog",)),      they said it. `above` is the
#  ("dog",              (2,),   ())]            nodes OF THIS SET standing over
#                                               each -- not ancestors().

[(s.node_id, s.rows) for s in answer.unplaced]
# [("wolfhound", (4,))] -- a tag this vocabulary does not carry. REPORTED, and
# the call does not raise: a corpus tagged before a vocabulary was reorganised
# is ordinary, and the person who could fix it is not here.

answer.unsupported_rows  # (1, 4) -- rows that supported nothing: the untagged
# one, and the one whose only tag was unplaced.

# (4) the projection. Membership is the policy's; order is the evidence's --
#     row count descending, ties in the order the rows named them.
answer.prune()  # Granularity.MOST_SPECIFIC, the default
# (NodeSupport("golden_retriever", rows=(0, 3), above=("dog",)),
#  NodeSupport("beagle",           rows=(0,),   above=("dog",)))
# `dog` is gone: two nodes below it were named. It is still in `supported`, so
# row 2 stays reachable -- pruning changes what is PRESENTED.

answer.prune(Granularity.MOST_GENERAL)  # (NodeSupport("dog", ...),)
answer.prune(Granularity.ALL)  # all three, ranked

# (5) and the tie back in: each kept node, judged by the vocabulary that placed
#     it, with the rows that are its evidence.
for support in answer.prune():
    here = axis.at(support.node_id)  # no localize() here: the roll-up already
    placed = here.entity()  #          answers in the axis's own key space
    named = placed.name if placed is not None else support.node_id
    context = [above.node for above in here.ancestors()]
    evidence = [hits[i]["text"] for i in support.rows]

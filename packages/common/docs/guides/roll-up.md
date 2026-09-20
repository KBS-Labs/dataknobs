# Roll-Up

A page of results is not one row. Each row already says which entities it is
about --- that is what its tags are --- and `roll_up` asks the next question:
**given all of them, which entities do these rows support, and on what
evidence?**

```python
answer = roll_up(onto, "species", tagged)
answer.prune()  # the entities to put in front of a person, ranked
```

**Not `deepest_common_ancestor`.** The two are one word apart and go in
opposite directions, which is the mistake this page exists to prevent. That
walk is pairwise and *generalises*: folded over the entities a row set supports
it returns something standing above every one of them, frequently an entity
your rows never named and have no evidence for. This *selects*: of the entities
the rows did name, it keeps the ones nothing more specific was named for, and
each one carries the rows that are its evidence. [Both are run below](#the-walk-one-word-away).

**It holds no vocabulary until you give it one, and opens nothing itself.** The
count over tags needs no ontology at all; the roll-up needs one because only a
vocabulary can say whether `beagle` is a node it carries. Neither reaches a
store or an embedder.

**The roll-up comes in two flavours** --- `roll_up` over an `Ontology` and
`async_roll_up` over an `AsyncOntology` --- because asking a vocabulary whether
it carries a node, and what stands above one, are reads, and on the
asynchronous flavour both are awaited. If your vocabulary's structure is backed
by rows rather than authored in a file, that is the flavour you hold, and
`async_roll_up` is the one to call. `ontology_support` has no twin, because it
reads the tags' own `ontology_id` and never asks a vocabulary anything.

## Where the names live

On the ontology door, beside the vocabulary they answer about:

```python
from dataknobs_common.ontology import (
    Granularity,
    NodeSupport,
    OntologySupport,
    SupportSet,
    async_roll_up,
    ontology_support,
    roll_up,
)
```

## The whole input

The same hand-edited vocabulary the [anchored view](anchored-view.md) and
[content tags](content-tags.md) use: two entity types, five entities, the edges
between them, and a taxonomy naming the relation those edges are made of.

<!-- worked-input -->

```yaml
# mammals.yaml
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string}
        - {name: lifespan_years, type: number, field_type: float}
    - id: Breed
      isa: Species                      # the TYPE lattice
      attributes:
        - {name: akc_group, type: string}

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal,
       description: "Warm-blooded, milk-producing vertebrates."}
    - {id: dog, type: Species, name: Dog, aliases: [Canine, "Domestic dog"],
       description: "A domesticated carnivoran."}
    - {id: retriever, type: Breed, name: Retriever}
    - {id: golden_retriever, type: Breed, name: Golden Retriever, aliases: [Goldie]}
    - {id: beagle, type: Breed, name: Beagle, aliases: [Beagles],
       source: {source_id: clinic_db, table: species, key: "sp-2291"}}

  assertions:
    - {subject: dog, relation: isa, object: mammal}
    - {subject: retriever, relation: isa, object: dog}
    - {subject: golden_retriever, relation: isa, object: retriever}
    - {subject: beagle, relation: isa, object: dog}

  taxonomies:
    - {id: species, name: Species, relation: isa}
```

## The worked call site

Five hits, of which one is about two nodes, one touched no vocabulary at all,
and one names a node this vocabulary does not carry. That is not a contrived
set --- it is what a page of results off a real corpus looks like, and a roll-up
whose worked case contains neither of the last two is a roll-up whose residue
was never designed.

<!-- worked-call-site -->

```python
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
```

That block is executed as written by a workspace test, and the test asserts it
is character-identical to the fence above. If this page and the code ever
disagree, the suite goes red rather than the page going quietly wrong.

## Three answers, and only one of them is a presentation

| | What it is | Who reads it |
|---|---|---|
| `supported` | every node the rows named that this axis carries, **in the order the rows named them** | the record. Nothing is dropped, so every row stays reachable |
| `unplaced` | nodes this vocabulary's tags named and this axis does **not** carry | a maintainer looking for the tagger --- it carries the rows, not just the ids |
| `prune(...)` | the entries to put in front of a person, ranked | a reader |

**`supported` is first-seen order and `prune()` is ranked, and the difference is
deliberate.** One is the record and the other is the presentation, so they are
two members rather than one answer that had to choose. `dog` disappears from
`prune()` because two nodes below it were named --- and it is still in
`supported`, so the row that named it is still yours. Pruning changes what is
*presented*, never what is *reachable*.

## What the ranking is by

Row count descending, ties in the order the rows named them. The word most
people arrive with is *ranked by specificity*, and after `MOST_SPECIFIC` that
cannot be the ranking: the survivors are mutually incomparable by construction,
because any pair where one stood above the other would have had the upper one
pruned. **The policy decides membership; the evidence decides order.**

There is no scoring anywhere here. The number being ordered is a count of rows,
and a count is comparable to another count whatever produced it --- which is why
the ranking survives every corpus rather than only the ones something scored.

## `above` is not `ancestors()`

```python
NodeSupport("golden_retriever", rows=(0, 3), above=("dog",))
```

`golden_retriever`'s ancestors in this vocabulary are `retriever`, `dog` and
`mammal`. Its `above` is `("dog",)` alone, because `above` is restricted to
**the nodes of this same set** --- some row named `dog`, and no row named
`retriever` or `mammal`.

It is a field rather than a member for the reason `prune` has no walk in it:
deriving it means asking the hierarchy, and a value that walks is a value that
performs I/O when you read it. Computing it once, in the call that already has
the vocabulary open, is what lets the projection be a pure reader afterwards.

## Rows are positions

```python
[hits[i] for i in support.rows]  # the evidence, not an approximation of it
```

Ascending, without duplicates, and indexes into the sequence **you** passed. The
row is recoverable from its position and the position is never recoverable from
the row, because two rows with the same content have one content and two
places. Your corpus is yours, so there is no row id this package is entitled to
invent --- which is also why the argument must be a `Sequence`. A generator has
no positions to report.

## Two names off a row, disposed of differently

A tag carries an ontology id and an axis id, and so does your call. They are
treated as opposites, and which one is which is the whole rule:

```python
roll_up(onto, "coat", tagged)  # NotFoundError, naming the axes that DO exist
```

**The axis you pass is refused**, because a name this vocabulary does not
declare is a name somebody typed, and they are one edit from the right one.

**A name off a row is filtered**, silently, into `unsupported_rows`. Nobody
typed it: it is an ordinary drifted corpus, written when the vocabulary had an
axis it has since dropped, and there is no one to tell. So `localize` is never
even asked about a foreign tag --- the filter runs first, which keeps that
door's refusal for the caller who does hand it an id directly.

The same split governs `unplaced`. A node id the axis does not carry is
*reported* rather than refused, because a corpus tagged before a vocabulary was
reorganised is the ordinary case rather than a bug.

## Which vocabularies are even in play

```python
[(s.ontology_id, s.rows) for s in ontology_support(reading.tags)]
# [("mammals", (0, 2, 3, 4))]
```

`ontology_support` is the same counting one level up: same corpus, same
evidence, same measure, grouped by *vocabulary* rather than by node and ranked
the same way. It is what you call when you have a page of results and do not yet
know what is on it --- and it holds no vocabulary, so it answers over ones you
have never loaded and ones nobody holds.

If you hold a registry of vocabularies and want this narrowed to the ones it
carries, `OntologyRegistry.ontologies_in_play` in `dataknobs-data` is that
narrowing, and it filters the tuple rather than counting again. A filter over a
ranked tuple preserves both the ranking and the tie-break, so one
implementation of the measure serves both heights --- which is why
`OntologySupport` is published from here rather than from there. What the
narrowing drops, a vocabulary the rows name and that registry does not hold, is
recovered by calling `ontology_support` yourself: the same call, over the same
tags, with nothing filtered. The registry guide is where that member is taught:
<https://kbs-labs.github.io/dataknobs/packages/data/guides/ontology-registry/#which-vocabularies-a-page-of-results-is-about>.

## The walk one word away

```python
from dataknobs_common.hierarchy import deepest_common_ancestor

kept = answer.prune()  # golden_retriever, beagle
deepest_common_ancestor(axis.structure, kept[0].node_id, kept[1].node_id)  # "dog"
answer.prune(Granularity.MOST_GENERAL)  # (NodeSupport("dog", ...),)
```

Over this vocabulary, **the walk whose name says *deepest* returns the same
answer as the policy called `MOST_GENERAL`** --- two opposite members of one
enum, over one corpus.

And the answer it hands back is one the caller has no evidence for. After
`prune()` you are holding `golden_retriever` and `beagle`, whose rows are 0 and
3; neither of those rows named `dog`. In an operation whose whole contract is
each entry carrying the rows that are its evidence, presenting `dog` there means
presenting a result with an empty evidence set. That `dog` happens to be in
`supported` on the strength of a different row is what makes the point precise:
the evidence exists, and it is not evidence for what was asked.

## Projecting onto a type is refused

```python
answer.prune(Granularity.AT_TYPE)  # ValidationError, naming what it would need
```

`AT_TYPE` is a member of the enum and its only behaviour is this refusal, which
is deliberate rather than unfinished. Reading what a node *is* means reading the
vocabulary's entity source, and a `SupportSet` holds node ids and row positions
and no source at all. So the refusal names the source it would need, and a
caller who wants the types asks the ontology for the entity of each node.

## What this is not

**Not a reader of rows.** It takes tags somebody already read ---
`read_node_tags_many(...).tags` is exactly the type it wants --- so the stance
you took about a row a writer got wrong stays visible at your call site rather
than becoming a policy buried in here.

**Not a resolver.** Nothing here matches text to an entity. The rows arrived
already placed, by whatever built the index; this asks what a set of placements
adds up to.

**No entity reads.** Neither call asks what a node *is*. `roll_up` asks whether
an axis carries a node and what stands above it; the entity behind a node is a
source read, and it is why `Granularity.AT_TYPE` is refused rather than
answered. If you want the types, ask the vocabulary for the entity of each node
the projection kept.

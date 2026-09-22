# The Anchored View

`Taxonomy.at(node_id)` returns a cursor: one axis, one node, and every question
asked from there — *am I here, am I a root, am I a leaf, what is one step above
me, what is one step below me, what is the whole way above me, what is
everything below me, how did I get here, what am I,* and *what was written on
the edge I just walked*. `HierarchyView` is the same cursor over a bare
`Hierarchy`, for a structure with no vocabulary behind it.

A view holds the structure, not a copy of it. It owns nothing, caches nothing,
and two views over one axis are equal exactly when they name the same node — so
a view taken before the axis changes reads the axis after, and there is nothing
in it to go stale.

## Where the names live

On the package door, with the rest of this family. All four are complete now —
`HierarchyView` carries its eleven members and `TaxonomyView` its fourteen, and
the three module walks that deliberately have no member are named in [What is
not on it](#what-is-not-on-it) — and they were published before they were, because adding a member to a class breaks
nobody while withholding the name costs a consumer something real. For the
taxonomy cursors that cost was an annotation: `at()` hands you one, and the
import only lets you write down what you hold. For `HierarchyView` it was the
whole capability — nothing published returns or constructs one, so over a
`Hierarchy` of your own the import is the only door in.

```python
from dataknobs_common import (
    AsyncHierarchyView,
    AsyncTaxonomyView,
    HierarchyView,
    TaxonomyView,
)
```

## The whole input

Hand-edited, no tooling, no second file. Two entity types where one is declared
`isa:` the other, five entities, the edges between them, and a taxonomy naming
the relation those edges are made of:

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

Two `isa` lattices are in that file and they are not the same set of edges. The
`isa:` on `Breed` is a **type** declaration — `Breed` inherits `Species`'
attributes. The `isa` assertions are between **entities** — `beagle` is a kind
of `dog`. They share a relation id and nothing else, which is why the call site
below asks about each one of a different object.

## The worked call site

Five things a cursor is for, in the order someone meets them. Every line runs
against the file above, exactly as written:

<!-- worked-call-site -->

```python
from pathlib import Path

from dataknobs_common.ontology import build_resolver, load_ontology

onto = load_ontology(Path("mammals.yaml"))
axis = onto.taxonomy("species")  # no store, no embedder, no event loop

# (1) the triage flow places a message
resolver = build_resolver(Path("mammals.yaml"), onto)
hit = resolver.resolve("golden retriever", k=5).ranked()[0]
hit.entity_id  # "golden_retriever"

# (2) anchor a cursor there, and widen to the context above it
here = axis.at(hit.entity_id)  # the anchored view
here.node  # "golden_retriever"
here.parents()  # (view("retriever"),) -- PLURAL, always
here.ancestors()  # retriever, dog, mammal

for above in here.ancestors():
    entity = above.entity()  # -> Entity | None
    assert entity is not None  # an id that misses is a typo, not a result
    entity.name  # "Retriever", "Dog", "Mammal"
    entity.description  # what folds into the prompt
    onto.assertions.find(subject=above.node)  # what is TRUE of it -- and it is
    # NOT on the view: a taxonomy holds no assertion axis

# (3) what does this type inherit? -- the OTHER isa lattice, on the axis
placed = here.entity()
assert placed is not None
axis.inherited_attributes(placed.type)  # akc_group, latin_name, lifespan_years

# (4) see what is still unspecified
there = here.at("dog")  # re-anchor: the message stopped here
there.is_leaf()  # False -- the CONSUMER concludes
there.children()  # retriever, beagle -- ask which

# (5) leave with keys, in your own id space
axis.subtree_keys(there.node)  # ["dog", "retriever", "golden_retriever", "beagle"]
```

That block is executed as written by a workspace test, and the test asserts it
is character-identical to the fence above. If this page and the code ever
disagree, the suite goes red rather than the page going quietly wrong.

Every example below runs against that same `mammals.yaml` and the `axis` it
yields. The sections take one question at a time rather than one step at a
time — what the door is and what it refuses, what a walked edge carries, what
the cursor deliberately does not hold — and two of them (`twice`,
`contradiction`) load a deliberately malformed vocabulary of their own, which
is said where they do it.

## The door, and the move

The door is on the axis. It takes the node and nothing else, because everything
a cursor needs is a field the taxonomy already holds:

```python
from pathlib import Path

from dataknobs_common.ontology import load_ontology

onto = load_ontology(Path("mammals.yaml"))   # the file above, unchanged
axis = onto.taxonomy("species")           # no store, no embedder, no loop

here = axis.at("golden_retriever")        # the door: from an axis
assert here.node == "golden_retriever"
assert [above.node for above in here.parents()] == ["retriever"]   # plural, always

there = here.at("dog")                    # the move: from a view you already hold
assert [below.node for below in there.children()] == ["retriever", "beagle"]
assert there.at("beagle") == axis.at("beagle")
```

`parents()` returns a tuple however many parents there are. An open-world
relation yields a DAG, and a cursor that silently picked one parent out of two
would be a lie the caller cannot detect — so there is no `parent`, and no `up()`.

**The door does not check that the node is there**, and that is deliberate
rather than lax. The check lives on the view, where the caller asks it; on the
asynchronous twin it would also drag a plain accessor into the event loop for a
question nobody has asked yet. `walk(from_id=…)` on the same axis *does* refuse
an unknown anchor, and the two are consistent: a walk includes its anchor, so
an unknown one would be emitted as a term of the axis, where a cursor answers
for itself.

## A node that is not here is neither a root nor a leaf

```python
gone = axis.at("no_such_node")

assert gone.exists() is False
assert gone.is_leaf() is False
assert gone.is_root() is False
assert gone.children() == ()
```

The guard is the whole reason `is_leaf()` and `is_root()` are not one line each.
Without it, `is_leaf()` answers *fully specified* about a node the axis has
never heard of — an absent node has an empty `children()` too — and a consumer
deciding whether a placement is specific enough to act on would decline to ask
a narrowing question on exactly the terms it knows nothing about. Measured on a
public vocabulary whose `isa` axis named 92 of its 170 terms, that was 46% of
it.

The cost, accepted: after the guard, `False` alone no longer separates *absent*
from *present with both parents and children*. `exists()` is what does, and it
is safe to call alone:

```python
assert axis.at("beagle").is_leaf() is True       # present, nothing below
assert axis.at("dog").is_leaf() is False         # present, two below
assert axis.at("mammal").is_root() is True       # present, nothing above
assert axis.at("dog").exists() and not axis.at("dog").is_root()
```

## What is written on the edge you walked

A hierarchy answers with ids. A taxonomy also carries the assertions its edges
were made of, so its cursor can report the **edge** and not only the endpoint —
the neighbour cursor and the `Assertion` together, one pair per assertion, so
one call gives both the annotation and a position to keep walking from:

```python
from dataknobs_common.ontology import EntityRef

((neighbour, placed_by),) = axis.at("beagle").parent_edges()
assert neighbour == axis.at("dog")
assert (placed_by.subject, placed_by.object) == ("beagle", EntityRef("dog"))

down = axis.at("dog").child_edges()
assert [neighbour.node for neighbour, _ in down] == ["retriever", "beagle"]
```

**Keyed per assertion, not per parent.** Two parents with one assertion each
give two pairs; one parent annotated twice also gives two, and the pairs share
their first element:

```python
twice = load_ontology(
    {
        "id": "twice",
        "assertions": [
            {"subject": "dog", "relation": "isa", "object": "mammal"},
            {"id": "by-the-kennel-club", "subject": "beagle", "relation": "isa",
             "object": "dog", "metadata": {"source": "akc"}},
            {"id": "by-the-federation", "subject": "beagle", "relation": "isa",
             "object": "dog", "metadata": {"source": "fci"}},
        ],
        "taxonomies": [{"id": "species", "relation": "isa"}],
    }
).taxonomy("species")

up = twice.at("beagle").parent_edges()
assert [neighbour.node for neighbour, _ in up] == ["dog", "dog"]
assert [placed_by.metadata["source"] for _, placed_by in up] == ["akc", "fci"]
```

**`()` means *nothing is written on any edge here*.** That is true of a root,
of an absent node, of a taxonomy whose `assertions` is `None` — a hierarchy
built from a `parent_id` column has rows and no assertions at all — and of a
parent nothing annotates. The three are told apart by other members rather
than by a wider return type that would make iterating the result unsafe:

```python
from dataclasses import replace

assert axis.at("mammal").parent_edges() == () and axis.at("mammal").is_root()
assert axis.at("no_such_node").parent_edges() == () and not axis.at("no_such_node").exists()

unannotated = replace(axis, assertions=None)
assert unannotated.at("beagle").parent_edges() == ()
assert not unannotated.has_edge_annotations()               # the question to ask
assert [above.node for above in unannotated.at("beagle").parents()] == ["dog"]

assert axis.has_edge_annotations()                          # and the other answer
```

!!! warning "Ask `has_edge_annotations()`, not `assertions is None`"

    `assertions is None` was the documented question and **it does not
    survive a live binding.** `OntologyRegistry` constructs an assertion
    source for every vocabulary it loads, whether or not any axis of that
    vocabulary is made of assertions — so a `kind: column` taxonomy over a
    live table has a source that is *empty* rather than *absent*.

    Measured, asked of both flavours of one five-row tree: `assertions is
    None` answers `False` **both times**, while the edge read answers `0` and
    `1`. So the consumer who follows the old instruction gets the **wrong**
    answer rather than no answer — `False` reads as *this axis does carry
    annotations*, which makes `()` read as *nothing is written on this edge*
    when the truth is *nothing can be*.

    `has_edge_annotations()` asks whether an **asserted** edge under this
    axis's relation lands on an edge *of this axis* — which is exactly what
    `parent_edges()` reads, so the two cannot disagree. It is named for what
    it measures rather than for provenance on purpose, and provenance is the
    wrong question in *both* directions: a `materialization.structure:
    materialized` copy of an assertion axis is *bound* and its edges are
    still assertions, and a `kind: column` axis is not one where nothing can
    be written — its edges come from the table, but an assertion landing on
    one of them annotates it.

    Narrowing by relation alone is not enough either, and it is the narrower
    mistake. Two axes may share a relation — a live column axis beside a
    legacy assertion axis over the same edge name is what a migration looks
    like — and an assertion belonging to the other axis would answer for
    this one. A stated **negation** would count too, which is the failure
    `edge_criteria` exists to prevent.

    `assertions` itself stays what it is — the source, or `None` for an axis
    built with none — and is still the thing the edge members read.

The structure decides which parents there are; the edge members only read what
is written on the edges to them. So an axis whose structure was copied at load
while its assertions stay live disagrees, where it disagrees, in the structure's
favour.

## A stated negation is not a pair

A vocabulary is open-world, so a document that wants to say *this edge does not
hold* says it, as an assertion with `polarity: negated`. An axis is made of
**asserted** edges: the structure does not walk the negation, and the edge read
does not hand it back either — even when a document states both polarities on
one edge:

```python
from dataknobs_common.ontology import Polarity

contradiction = load_ontology(
    {
        "id": "contradiction",
        "assertions": [
            {"subject": "dog", "relation": "isa", "object": "mammal"},
            {"id": "yes", "subject": "beagle", "relation": "isa", "object": "dog"},
            {"id": "no", "subject": "beagle", "relation": "isa", "object": "dog",
             "polarity": "negated"},
        ],
        "taxonomies": [{"id": "species", "relation": "isa"}],
    }
).taxonomy("species")

((_, placed_by),) = contradiction.at("beagle").parent_edges()
assert placed_by.id == "yes" and placed_by.polarity is Polarity.ASSERTED
```

That narrowing has one home, `edge_criteria`, and the cursor reads through the
same function the assertion-backed hierarchy does. It is a function returning
criteria rather than one performing a read, so it has no flavour and unpacks
into either `find`. Use it yourself wherever you read an axis's edges straight
from the assertion source, rather than writing the polarity by hand:

```python
from dataknobs_common import edge_criteria

edges = contradiction.assertions.find(subject="beagle", **edge_criteria("isa"))
assert [assertion.id for assertion in edges] == ["yes"]
```

## Why the taxonomy cursor is thin

Every structural member on `TaxonomyView` invokes the `HierarchyView` member of
the same name over `taxonomy.structure` and re-wraps the answer. It does not
re-walk, and it does not call the protocol itself. That is exactly the shape a
maintainer reimplements without noticing the forward is there — one protocol
call, correct today, drifting the first time either side is edited — so a test
patches each `HierarchyView` member and asserts the `TaxonomyView` member moves.
`at()` is the one exception: it constructs a `TaxonomyView`, which no
`HierarchyView` member can return, so it forwards to nothing.

The two edge members are the reason the taxonomy cursor exists at all. They
need the assertions the edges were made of, and a bare `Hierarchy` has none.

## The same cursor over a bare hierarchy

A structure with no vocabulary behind it — a `parent_id` column, an object
tree, a mapping — gets the same six questions from `HierarchyView`, generic in
the key the way the protocol is, with `str` defaulted:

```python
from dataknobs_common import HierarchyView, MappingHierarchy

billing = MappingHierarchy({"late-fees": ("billing",), "refunds": ("billing",), "billing": ()})
view = HierarchyView(billing, "billing")

assert view.is_root() and not view.is_leaf()
assert [below.node for below in view.children()] == ["late-fees", "refunds"]
assert view.at("late-fees").is_leaf()
assert HierarchyView(MappingHierarchy({3: (2,), 2: (1,), 1: ()}), 3).parents()[0].node == 2
```

Both structural cursors hash, so a walk can key a `seen` set on them the way
the taxonomy cursor's section below does — **and they hash exactly as far as
what they hold does**. The cursor reports that capability rather than requiring
it: `HierarchyView` is frozen, so its hash is its **field tuple**, which is the
`Hierarchy` *and* the key, and either can withhold it.

Everything shipped here gives it, and each for its own reason. The two
mapping twins are compared by identity, so their `dict` fields are never
reached — forced rather than chosen, since they hold the mapping *you* passed:
a hash derived from it would move when you mutate it, while a cursor over it
sits in the `seen` set the hash exists to serve. The assertion-backed pair
holds a source that hashes the way any object does over a relation it
canonicalises to an id, so naming a relation by its `RelationType` rather than
its id builds the same axis rather than a second one. And a `str` key hashes.
A structure of your own will not, if it is a frozen dataclass over a `dict` —
and neither will a **key** of your own of that shape, which is the half worth
saying out loud, because the `Hashable` bound on the key parameter does not
catch it: such a type satisfies the bound and raises, so the bound documents
the requirement rather than enforcing it. Either way it fails in the same
shape, answering `isinstance(view, Hashable)` with `True` and raising at
`hash(view)`. Declare such a type `eq=False`, or hold its contents in something
hashable.

The cost is the axis's cost, one level down: two mappings holding the same
edges are no longer equal to each other. Ask `parent_edges()` on both when that
is the question.

The asynchronous twins — `AsyncHierarchyView`, `AsyncTaxonomyView`, and `at()`
on `AsyncTaxonomy` — have the same members, every one `async def` except
`at()`, which constructs rather than reads and so awaits nothing.

## The whole way up, and everything below

Five members walk, and each is one line over the module-level walk of the same
name — so what each one does, and what it refuses, is that function's contract
rather than a second one written here:

```python
here = axis.at("golden_retriever")

assert [up.node for up in here.ancestors()] == ["retriever", "dog", "mammal"]
assert [down.node for down in axis.at("dog").descendants()] == [
    "retriever",
    "golden_retriever",
    "beagle",
]
assert here.paths_to_root() == (("golden_retriever", "retriever", "dog", "mammal"),)
```

`ancestors()` and `descendants()` come back as **cursors**, so a walk composes:
each answer is a place to keep asking from. `paths_to_root()` comes back as
**keys**, because what it answers with is routes and a route's meaning is its
order — ask `at()` for a cursor on a node you found in one.

**`descendants()` and `descendants_to_depth()` are two members, not one member
with a `depth=`.** They differ in both of the things a walk can differ in, and
folding them would make a keyword select between two contracts:

```python
from dataknobs_common.exceptions import NotFoundError

dog = axis.at("dog")

assert [n.node for n in dog.descendants()] == ["retriever", "golden_retriever", "beagle"]
assert [n.node for n in dog.descendants_to_depth(1)] == ["dog", "retriever", "beagle"]

assert axis.at("no_such_node").descendants() == ()      # excludes its anchor: answers
try:                                                     # includes it: refuses
    axis.at("no_such_node").descendants_to_depth(1)
except NotFoundError as refusal:
    assert refusal.context["anchor"] == "no_such_node"
```

**`children_at_depth()` is the third of that family and answers one level
rather than a span** — *exactly this far down*, where `descendants_to_depth()`
answers *everything down to here*. A caller who wants the one from the other
has to subtract two results, which is why both are members:

```python
assert [n.node for n in dog.descendants_to_depth(2)] == [
    "dog",
    "retriever",
    "golden_retriever",
    "beagle",
]
assert [n.node for n in dog.children_at_depth(2)] == ["golden_retriever"]
assert dog.children_at_depth(9) == ()          # nothing is that deep: an answer
```

Its anchor is *where you are* — `depth=0` is the node itself — which is what
makes it a member here at all, and it emits that anchor, so it refuses one the
axis does not contain exactly as `descendants_to_depth()` does. A depth the axis
does not reach is a different answer from an anchor it does not know, and the
two stay apart: `()` says *nothing is that deep*, the refusal says *no such
node*.

Each carries the keywords its module function carries — `cache=` on all five,
`max_paths=` on `paths_to_root()`, and `max_concurrency=` on every asynchronous
twin. A cache spent across two walks is the caller's, and
`MappingHierarchy.snapshot` takes one too — so a snapshot and a later walk over
the same backing can share the replies instead of each paying for them:

```python
from dataknobs_common import HierarchyView, MappingHierarchy

class CountingParents:
    # A backing that reports how often it was asked to descend.
    def __init__(self, parents):
        self._parents, self.calls = parents, 0

    def parents(self, node_id):
        return self._parents.get(node_id, ())

    def children(self, node_id):
        self.calls += 1
        return tuple(n for n, ps in self._parents.items() if node_id in ps)

    def roots(self):
        return tuple(n for n, ps in self._parents.items() if not ps)

    def contains(self, node_id):
        return node_id in self._parents

edges = {"mammal": (), "dog": ("mammal",), "retriever": ("dog",),
         "golden_retriever": ("retriever",), "beagle": ("dog",)}

cold = CountingParents(edges)
MappingHierarchy.snapshot(cold)
answer = [d.node for d in HierarchyView(cold, "dog").descendants()]
assert cold.calls == 9                      # five for the snapshot, four to descend

warm = CountingParents(edges)
memo: dict = {}
MappingHierarchy.snapshot(warm, cache=memo)
assert [d.node for d in HierarchyView(warm, "dog").descendants(cache=memo)] == answer
assert warm.calls == 5                      # the descent asks nothing new
```

**The memo reaches the descending branch only.** An axis that publishes
`parent_edges()` — `MappingHierarchy` does, and so does the `AssertionHierarchy`
behind the `axis.structure` above — is asked for its edges in one call and never
descends, so there is no reply for a memo to hold. Passing one there is not an
error and saves nothing.

## What a node is

`TaxonomyView.entity()` reads the other axis — the content one — and answers
what a node **is** rather than where it sits:

```python
assert here.entity().name == "Golden Retriever"
assert [up.entity().name for up in here.ancestors()] == ["Retriever", "Dog", "Mammal"]
```

**`None` from it is a state, not an error, and it is not what `exists()`
answers.** `exists()` asks the structure; this asks the content, and a node the
structure knows with nothing written about it is ordinary — an axis built from
a parent-id column knows ids the entity store has never been given. Ask
`exists()` for *is this node here*, and read `None` here as *nothing is written
about it*.

What is **asserted about** an ancestor is the ontology's, not the cursor's:

```python
facts = onto.assertions.find(subject="dog")
assert [(a.relation, a.object.entity_id) for a in facts] == [("isa", "mammal")]
```

That boundary is deliberate. The cursor reports the **edge it walked** and
nothing else; the assertion axis stays on the vocabulary, because a taxonomy
over a `parent_id` column has no assertions anywhere and a cursor that promised
them could not keep the promise.

## What is not on it

Three module walks have no cursor member, and the reasons differ:

`deepest_common_ancestor` will never have one — a cursor names one node and that
walk takes two anchors. Call the module function with both.

`flatten` and `leaves` each take a single anchor and would fit a cursor, and are
left off because the anchor means something different on them: on a cursor an
anchor is *where you are*, and on those two it is a **filter over the whole
axis** — `flatten(from_id=…)` and `leaves(under=…)` take an *optional* anchor
that defaults to every root and narrows from there. A member that read the
cursor's node as that argument would be answering a different question from the
one the same name answers a line above. Call the module function with the axis
and the node:

```python
from dataknobs_common import leaves

assert leaves(axis.structure, under="dog") == ("golden_retriever", "beagle")
```

A walk you write yourself may key its `seen` set on the cursors rather than on
bare ids — they hash, and two cursors over one axis collide exactly when they
name the same node, which is the property such a set needs:

```python
from dataknobs_common.ontology import TaxonomyView

seen: set[TaxonomyView] = set()
frontier = [here]
while frontier:
    node = frontier.pop()
    if node in seen:
        continue
    seen.add(node)
    frontier.extend(node.children())
```

An axis is compared by the identity of what it holds and the value of what it
names. A `Taxonomy` holds a definition and three handles, so identity is the
whole of it: cursors over two *separately built* taxonomies never compare
equal, even when both came from the same ontology and name the same node. Hold
the axis you walk from — not because a second build always compares unequal
(an assertion-backed axis over one source and one relation does compare equal,
by that same rule) but because `Ontology.taxonomy()` builds on each call,
which `at()` already encourages by being the only door. A
`set[HierarchyView[str]]` works the same way over a bare structure, on the one
condition [the section above](#the-same-cursor-over-a-bare-hierarchy) states.

Nothing on either cursor is named `granularity`: whether a placement is specific
enough is the consumer's conclusion, drawn from `is_leaf()` and `children()`,
and the word belongs to a projection policy over a match set.

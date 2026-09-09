# The Anchored View

`Taxonomy.at(node_id)` returns a cursor: one axis, one node, and six questions
asked from there — *am I here, am I a root, am I a leaf, what is above me, what
is below me,* and *what was written on the edge I just walked*. `HierarchyView`
is the same cursor over a bare `Hierarchy`, for a structure with no vocabulary
behind it.

A view holds the structure, not a copy of it. It owns nothing, caches nothing,
and two views over one axis are equal exactly when they name the same node — so
a view taken before the axis changes reads the axis after, and there is nothing
in it to go stale.

## Where the names live

Imported by module path, like the rest of this family, and on no package door
yet: `HierarchyView` carries six of its nine planned members and `TaxonomyView`
eight of twelve, and a name goes on a door once. A consumer rarely imports the
cursors at all — `at()` constructs them — so these lines are for annotations.

```python
from dataknobs_common.hierarchy import AsyncHierarchyView, HierarchyView
from dataknobs_common.ontology.taxonomy import AsyncTaxonomyView, TaxonomyView
```

## The door, and the move

The door is on the axis. It takes the node and nothing else, because everything
a cursor needs is a field the taxonomy already holds:

```python
from dataknobs_common.ontology import load_ontology

onto = load_ontology(
    {
        "id": "mammals",
        "entity_types": [{"id": "Species"}, {"id": "Breed", "isa": "Species"}],
        "relation_types": [{"id": "isa", "transitive": True}],
        "entities": [
            {"id": "mammal", "type": "Species", "name": "Mammal"},
            {"id": "dog", "type": "Species", "name": "Dog"},
            {"id": "retriever", "type": "Breed", "name": "Retriever"},
            {"id": "golden_retriever", "type": "Breed", "name": "Golden Retriever"},
            {"id": "beagle", "type": "Breed", "name": "Beagle"},
        ],
        "assertions": [
            {"subject": "dog", "relation": "isa", "object": "mammal"},
            {"subject": "retriever", "relation": "isa", "object": "dog"},
            {"subject": "golden_retriever", "relation": "isa", "object": "retriever"},
            {"subject": "beagle", "relation": "isa", "object": "dog"},
        ],
        "taxonomies": [{"id": "species", "relation": "isa"}],
    }
)
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
assert unannotated.assertions is None                       # the question to ask
assert [above.node for above in unannotated.at("beagle").parents()] == ["dog"]
```

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
from dataknobs_common.ontology.hierarchy import edge_criteria

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
from dataknobs_common.hierarchy import HierarchyView, MappingHierarchy

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

## What is not on it yet

The three walks — `ancestors()`, `descendants()`, `paths_to_root()` — and
`TaxonomyView.entity()` are declared and arrive in a later release. Until then,
walk from a view's node with the module-level walks:

```python
from dataknobs_common.hierarchy import ancestors

assert ancestors(axis.structure, here.node) == ("retriever", "dog", "mammal")
```

A walk you write yourself may key its `seen` set on the cursors rather than on
bare ids — they hash, and two cursors over one axis collide exactly when they
name the same node, which is the property such a set needs:

```python
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

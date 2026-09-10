# Hierarchies and Taxonomies

`dataknobs_common.hierarchy` states what a *structure* is — what a node's
parents and children are — and the traversals over it.
`dataknobs_common.ontology.taxonomy` reifies one such structure as a walkable
axis of a vocabulary, and `dataknobs_common.ontology.hierarchy` backs one with
the assertions of a single relation.

The three are separable on purpose: the protocols know nothing about
ontologies, so a hierarchy over a `parent_id` column or an in-memory object
tree satisfies them and inherits every walk without importing a vocabulary.

## Overview

- **Two protocols, one synchronous and one asynchronous** — `Hierarchy` and
  `AsyncHierarchy`, four members each, both `@runtime_checkable`.
- **Read-only, key-addressed, multi-parent-tolerant** — `parents()` returns a
  sequence and never a single node, because an open-world relation yields a DAG.
- **Each walk is written once.** A traversal is a flavour-free generator; the
  only twinned code is the driver pair, which is a fixed cost — it does not
  grow when a walk is added. One public traversal ships today, so this is a
  bet on the second, not a saving already banked.
- **A backing that can answer a whole level says so.** `BulkHierarchy` is an
  optional protocol adding `children_many` / `parents_many`; the drivers use
  them when they are there and fan out when they are not.
- **The key type is a parameter defaulting to `str`**, so a bare `Hierarchy`
  means `Hierarchy[str]` and an object tree with no ids at all can bind `K` to
  its own node type.
- **`Taxonomy` holds no copy.** The sources are the authority, so a rebuild
  beneath one is visible on the next read.

## Where the names live

On the package door. Everything in the three groups below is importable from
`dataknobs_common` directly, and from the module that defines it — the same
names either way, so a line already spelling the module path keeps working.

**The protocols, and what implements them.** `Hierarchy` and `AsyncHierarchy`
are the four-member core — `roots`, `parents`, `children`, `contains`.
`BulkHierarchy` and `AsyncBulkHierarchy` add a batched frontier read, bounded
by `DEFAULT_FRONTIER_CONCURRENCY`; `EnumerableHierarchy` and
`AsyncEnumerableHierarchy` add enumeration for a backing that can afford it.
Both pairs are optional: a walk uses one when the structure it was handed
implements it and takes the singular path when it does not. `MappingHierarchy`
and `AsyncMappingHierarchy` are the concretes over edges already in memory, and
`AssertionHierarchy` and `AsyncAssertionHierarchy` are the ones over a
vocabulary's assertions.

**The walks, and the core they are written against.** `ancestors` and
`async_ancestors` are module-level functions generic over `Hierarchy`, not
methods on it — implementing the protocol earns every walk and overrides none.
`drive` and `async_drive` are that core exposed, which is what you write a walk
of your own against; `Ask`, `Member` and `Walk` are the three types its
protocol is spelled in.

**Still gaining members, and on no door.** `Taxonomy` and `AsyncTaxonomy` carry
two methods today, `walk()` and `at()`, out of the nine they are planned to
hold; the rest arrive in later releases. What ships now will not change shape.
A name on a door is a promise, so these stay reachable by module path until
they are finished — and nothing needs the import to use them, since
`onto.taxonomy("species")` hands one back. `HierarchyView` and
`AsyncHierarchyView` are held back for the same reason; the cursor `at()`
returns has a page of its own — [The Anchored View](anchored-view.md).

```python
from dataknobs_common import (
    DEFAULT_FRONTIER_CONCURRENCY,
    AssertionHierarchy,
    AsyncAssertionHierarchy,
    AsyncBulkHierarchy,
    AsyncHierarchy,
    BulkHierarchy,
    Hierarchy,
    ancestors,
    async_ancestors,
    async_drive,
    drive,
)
from dataknobs_common.hierarchy import AsyncHierarchyView, HierarchyView
from dataknobs_common.ontology.taxonomy import AsyncTaxonomy, Taxonomy
```

## Quick start — an axis of a vocabulary

A vocabulary declares an axis by naming the relation its edges are made of:

```yaml
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
    - id: Breed
      isa: Species

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal}
    - {id: dog, type: Species, name: Dog}
    - {id: retriever, type: Breed, name: Retriever}
    - {id: beagle, type: Breed, name: Beagle}

  assertions:
    - {subject: dog, relation: isa, object: mammal}
    - {subject: retriever, relation: isa, object: dog}
    - {subject: beagle, relation: isa, object: dog}
    # an attribute value is an assertion whose object is a literal
    - {subject: dog, relation: lifespan_years, object: 12}

  taxonomies:
    - {id: species, name: Species, relation: isa}
```

`Ontology.taxonomy(name)` reaches it:

```python
from pathlib import Path

from dataknobs_common.ontology import load_ontology

onto = load_ontology(Path("mammals.yaml"))
species = onto.taxonomy("species")

assert tuple(species.walk()) == ("mammal", "dog", "retriever", "beagle")
```

It takes the name and nothing else. Every collaborator the axis needs — the
structure, the entity source, the assertions — is a field the ontology already
holds, which is what makes this an accessor rather than a factory: no store to
bind, no event loop to be inside, no collaborator to construct first.

The asynchronous twin is reached the same way and is deliberately *not*
awaitable, because it constructs over fields already in hand:

```python
from dataknobs_common.ontology import async_load_ontology

async_onto = await async_load_ontology(Path("mammals.yaml"))
async_species = async_onto.taxonomy("species")    # no await — it builds, it does not fetch

assert [node async for node in async_species.walk()] == [
    "mammal", "dog", "retriever", "beagle",
]
```

## The structure protocol

Four members, and each exists to keep a specific pair of answers apart:

| Member | Returns | |
|---|---|---|
| `roots()` | `Sequence[K]` | the nodes this axis leaves unplaced |
| `parents(node_id)` | `Sequence[K]` | plural, always |
| `children(node_id)` | `Sequence[K]` | |
| `contains(node_id)` | `bool` | whether the axis knows the node at all |

**`roots()` is not an extent.** It answers *which nodes this relation places
under nothing*, which coincides with a type's membership only by accident. For
the extent — every entity of a type, placed or not — ask the entity source
(`EntitySource.by_type`).

**`contains()` is why an empty `children()` is unambiguous.** *Nothing below
this node* and *this node is not here* are opposite answers that an empty
sequence alone cannot tell apart, so the question has its own member.

**A literal object contributes no edge.** `dog lifespan_years 12` is a value,
not a place in a structure, so an assertion whose object is not an entity is
invisible to every member above.

### Answering a whole level at once

Every walk here asks about a **frontier**, not a node — that is what earns
`async_drive` its one round of concurrency per depth. But `children(node_id)`
is singular, so a backing that could answer a level in one query was being
asked once per node with no way to say otherwise.

`BulkHierarchy` and `AsyncBulkHierarchy` are how it says otherwise. They are
**optional** and deliberately separate protocols: the singular pair stays
sufficient, and an implementation that has only those four members is driven
exactly as before.

```python
class RowBackedAxis:
    def children_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        rows = self.db.query(
            "select parent, id from nodes where parent = any(%s)", (list(node_ids),)
        )                                        # one query for the level
        by_parent: dict[str, list[str]] = {n: [] for n in node_ids}
        for parent, node_id in rows:
            by_parent[parent].append(node_id)
        return tuple(by_parent[n] for n in node_ids)   # positional, one per node
```

Two things the drivers rely on:

- **The reply is positional** — one sequence per node asked about, in the order
  asked. A node with no answer contributes an empty sequence rather than being
  dropped, because the walk pairs replies with the frontier it sent.
- **Bulk and singular must agree.** The driver prefers the bulk member without
  asking, so the two forms answering differently is a bug the caller cannot
  see.

`AssertionHierarchy` implements both, over `AssertionSource.find_many`, which
is where the capability already existed. So does `MappingHierarchy` below, where
a level is one dict lookup per node — which makes it the cheapest possible
implementation of the pair, and the reason declining to offer it would be the
decision needing an argument. Concurrency does not substitute for either:
`gather` runs one round trip per node at the same time, while `children_many` is
one query for the level.

### `@runtime_checkable`, and what it does not reach

Both protocols are runtime-checkable, so `isinstance(x, Hierarchy)` works —
against the **bare** name, since `isinstance(x, Hierarchy[str])` raises
`TypeError` as it does for every subscripted generic.

The check compares member *names* and nothing else. A synchronous
implementation therefore satisfies `AsyncHierarchy` at runtime; only the static
check separates the flavours. Use `isinstance` to reject an object that is not
a hierarchy at all, not to decide which flavour you are holding.

## Walking

`ancestors()` returns every node above one, nearest first, excluding the node
itself:

```python
from dataknobs_common.hierarchy import ancestors, async_ancestors

assert ancestors(species.structure, "beagle") == ("dog", "mammal")
assert await async_ancestors(async_species.structure, "beagle") == ("dog", "mammal")
```

Two guarantees worth relying on: the visited set is unconditional, so a walk
terminates on cyclic data whatever an acyclicity constraint claims; and results
are deduplicated in walk order, so a DAG node reachable by several paths is
still one entry.

An empty result means **either a root or a node the axis does not have**, and
`ancestors` does not refuse the second — unlike `Taxonomy.walk` below, which
refuses an anchor its axis does not contain. The difference is recoverability:
`walk` includes its anchor, so an unknown one would be emitted as a term of the
axis and the caller could not detect it, where `ancestors` excludes its anchor
and returns nothing false. The answer is ambiguous rather than wrong, and
`contains()` resolves it in one call:

```python
assert ancestors(species.structure, "mammal") == ()      # a root
assert ancestors(species.structure, "marmoset") == ()    # not in the axis
assert species.structure.contains("mammal")
assert not species.structure.contains("marmoset")
```

`Taxonomy.walk()` is the axis's own traversal — every node at or under a point,
breadth first, **including** the anchor:

```python
tuple(species.walk())                              # from the roots
tuple(species.walk(from_id="dog"))                 # ("dog", "retriever", "beagle")
tuple(species.walk(from_id="dog", max_depth=0))    # ("dog",) — the anchor alone
```

It is a generator, so a caller that stops early does no work for the levels it
never reached.

An anchor the axis does not contain is **refused**, not walked:

```python
from dataknobs_common.exceptions import NotFoundError

try:
    tuple(species.walk(from_id="marmoset"))
except NotFoundError as refusal:
    assert refusal.context == {"taxonomy": "species", "anchor": "marmoset"}
```

Because the anchor is included in the output, seeding a walk with it unchecked
would emit an id the axis does not contain as though it were a term of the
axis, and the caller could not tell — a walk is exactly what they asked for.
Yielding nothing instead would be worse: it collapses *nothing below this node*
into *this node is not here*, which are the two answers `contains()` exists to
keep apart. This is the walk that asks it.

### Writing a walk of your own

`drive()` and `async_drive()` are the pair that makes a traversal
flavour-agnostic, and they are usable directly. A walk is a generator that
*yields a request* — a member name and the nodes to ask it about — and receives
one sequence per node asked about:

```python
from dataknobs_common.hierarchy import async_drive, drive

def _leaves():
    """Every node with no children, from the roots."""
    (roots,) = yield ("roots", ())
    frontier, found = tuple(roots), []
    seen = set(frontier)
    while frontier:
        replies = yield ("children", frontier)
        fresh = []
        for node_id, children in zip(frontier, replies, strict=True):
            if not children:
                found.append(node_id)
            fresh.extend(c for c in children if c not in seen)
        seen.update(fresh)
        frontier = tuple(fresh)
    return tuple(found)

assert drive(species.structure, _leaves()) == ("retriever", "beagle")
assert await async_drive(async_species.structure, _leaves()) == ("retriever", "beagle")
```

The generator contains no `await` and no knowledge of which flavour is driving
it: `_leaves` is one definition and both drivers run it.

**Build a fresh walk per drive.** Note the two `_leaves()` calls above — a walk
is a generator and therefore single-use, and handing a spent one back to a
driver raises `RuntimeError` rather than answering:

```python
walk = _leaves()
drive(species.structure, walk)
drive(species.structure, walk)      # RuntimeError: this walk has already been driven
```

The refusal exists because the alternative was silent. A driver learns a walk's
result from the `StopIteration` the generator raises when it returns, so a walk
that was *already* exhausted raised the same exception with `value=None` — and
the driver handed that `None` back typed as the walk's declared result, to fail
somewhere else entirely.

A walk left *suspended* part-way through is refused too, and for a worse reason
than exhaustion: resuming one sends `None` where the reply to its outstanding
question belongs, so it would answer out of a frontier it never read rather
than fail.

**Write a walk that wraps another as a generator function, not as a class.**
The freshness the drivers read lives on the generator object, so a walk that
satisfies `Generator` structurally — a class forwarding `send` and `throw` — is
refused with `TypeError` instead. `yield from` keeps the wrapper a generator,
and hands back the inner walk's result to do as you like with:

```python
def _leaves_sorted():
    """`_leaves`, ordered — a generator function, and so still a walk."""
    return tuple(sorted((yield from _leaves())))

assert drive(species.structure, _leaves_sorted()) == ("beagle", "retriever")
```

### How wide a frontier read gets

The asynchronous driver gathers a whole frontier at once, so concurrency is per
*depth* rather than per node — a property a hand-twinned walk gets only if
someone remembers to write it into both copies.

That fan-out is bounded, and the bound is yours to set:

```python
from dataknobs_common.hierarchy import DEFAULT_FRONTIER_CONCURRENCY

assert DEFAULT_FRONTIER_CONCURRENCY == 8            # what you get for saying nothing

await async_ancestors(async_species.structure, "beagle", max_concurrency=4)
await async_drive(async_species.structure, _leaves(), max_concurrency=4)

async for node_id in async_species.walk(max_concurrency=4):
    ...
```

Without a bound the width of the fan-out is the width of the *level*, which is
a property of the data rather than of anything anyone configured: a node with
ten thousand children issues ten thousand concurrent calls into whatever the
backing is.

`DEFAULT_FRONTIER_CONCURRENCY` is deliberately small, sized against what
dataknobs itself ships — the asynchronous Postgres pool in `dataknobs-data`
defaults to five connections and the pgvector store to ten — on the reasoning
that a walk should not be the thing that saturates a pool it does not own. Set
it yourself when you know your backing; that is what the keyword is for.

A backing that implements the bulk members never had the problem — it gets one
call per level — which is why the bound matters most for the plain
`parents`/`children` pair, the one a hand-written hierarchy starts with.

## The assertion backing

`AssertionHierarchy` binds an assertion source to one relation. It opens
nothing and caches nothing:

```python
from dataknobs_common.ontology.hierarchy import AssertionHierarchy

structure = AssertionHierarchy(onto.assertions, "isa")

assert structure.parents("beagle") == ("dog",)
assert structure.children("dog") == ("retriever", "beagle")
assert structure.contains("beagle") and not structure.contains("marmoset")
```

`parents(x)` is `find(subject=x, …)` read for its entity objects; `children(x)`
is the mirrored query, `find(object=x, …)` read for its subjects. Because the
source is the authority rather than a snapshot, a rebuild beneath the structure
is visible immediately — which is what lets a long-lived axis hold the structure
rather than a copy of it.

Neither call names the relation itself. Both go through one private query
helper per flavour, which is where the axis says which relation it is and that
a negated edge is not part of it — so a member added to the class later cannot
omit either by writing a `find(…)` that looks complete.

### A stated negation is not an edge

A vocabulary is open-world: an assertion nobody wrote is *unknown*, not false.
So a document that wants to say *a whale is not a fish* says it, as an
assertion with a polarity of its own:

```python
from dataknobs_common.ontology import load_ontology

negations = load_ontology(
    {
        "id": "sea",
        "assertions": [
            {"subject": "orca", "relation": "isa", "object": "whale"},
            {"subject": "whale", "relation": "isa", "object": "fish",
             "polarity": "negated"},
        ],
    }
)
sea = AssertionHierarchy(negations.assertions, "isa")
```

An axis is made of **asserted** edges, so it does not walk that one:

```python
assert sea.parents("whale") == ()          # not ("fish",)
assert sea.children("fish") == ()
assert sea.roots() == ("whale",)           # no asserted parent, so unplaced
assert not sea.contains("fish")            # named by nothing but the negation
assert "fish" not in sea.parent_edges()    # and so absent from a snapshot
```

All five follow from that one sentence. `roots()` is *the nodes this relation
leaves unplaced*, and reporting `whale` as placed would report a placement no
edge makes; `contains` is what keeps *nothing below this node* apart from *this
node is not here*, and a negation places nothing — the same way a literal
object does not.

The negation is still in the vocabulary, and a query that does not narrow finds
it. That is the point of stating one:

```python
stated = negations.assertions.find(subject="whale", relation="isa")
assert [assertion.polarity.value for assertion in stated] == ["negated"]
```

The narrowing an axis applies has one home, `edge_criteria`: *this relation,
asserted*, as criteria you unpack into either `find` rather than a polarity
you write by hand. Both flavours of the assertion backing read through it, the
taxonomy cursor's edge members read through it, and a consumer reading an
axis's edges straight from the source can too:

```python
from dataknobs_common.ontology.hierarchy import edge_criteria

assert negations.assertions.find(subject="whale", **edge_criteria("isa")) == []
assert [a.subject for a in negations.assertions.find(**edge_criteria("isa"))] == ["orca"]
```

It lives under `ontology/` rather than beside the protocol it satisfies:
reading an edge means asking at runtime whether an assertion's object is an
entity or a literal, and the general module may not import the vocabulary
package to find out. A concrete goes where its dependency is — the same rule
that puts a database-backed hierarchy in `dataknobs-data`.

## The mapping backing

`AssertionHierarchy` reads its edges. `MappingHierarchy` *holds* them — a
node's parents, as a mapping — which is the axis for a vocabulary somebody
typed, or holds in memory, or has just finished walking. It opens nothing, so
it lives beside the protocols rather than with a backing package:

```python
from dataknobs_common.hierarchy import (
    AsyncMappingHierarchy,
    MappingHierarchy,
    ancestors,
)

species = MappingHierarchy({"dog": ("mammal",), "beagle": ("dog",)})

assert species.roots() == ("mammal",)
assert ancestors(species, "beagle") == ("dog", "mammal")
assert species.contains("mammal")           # a value under a key is in the axis
```

The argument is positional, and the field behind it is called `parent_map`
rather than `parents`: a dataclass field and a protocol member may not share a
name, and `parents` is a member.

**`roots()` is every node with no parents, and `contains()` is every node the
walks can reach** — a key of the mapping, *or* a value under any key. `mammal`
above is never a key, and both members answer for it. That is not a nicety:
`Taxonomy.walk(from_id=...)` refuses an unknown anchor by asking `contains`, so
a backing answering on keys alone would refuse a node that is genuinely there
and name the caller's anchor for it.

The inversion behind `children()` is computed once, at construction, rather than
per call — a downward walk over a parent mapping is the case that would make an
inversion-per-call quadratic.

`AsyncMappingHierarchy` is the same mapping in an `AsyncHierarchy`-shaped slot.
It awaits nothing, and that is the point rather than an oversight: a consumer
typed against the asynchronous protocol — because the *rest* of their vocabulary
is by-reference — still needs something hand-built to put in the slot.

### From a tree somebody maintains by hand

A nested file has no id field anywhere: a node *is* its path.
`from_nested` mints one id per node, as a slug of the path:

```python
areas = MappingHierarchy.from_nested(
    {
        "name": "Billing",
        "children": [
            {"name": "Invoices", "children": [{"name": "Late Fees"}]},
            {"name": "Refunds"},
        ],
    }
)

assert areas.roots() == ("billing",)
assert areas.parents("billing/invoices/late-fees") == ("billing/invoices",)
```

The id is a slug of the path *taken once* and the path becomes the name, so
renaming a node changes what it is called and not what it is — which is what
keeps a rename from re-keying every descendant of a renamed interior node.

**These are the same ids an ontology mints for the same tree** under a
`kind: nested` source. The traversal, the slug and the collision refusal are one
implementation that both doors call, so a consumer who loads a document one way
and holds the same tree the other is holding one vocabulary rather than two.
A path is not a key, though, so two paths can slug to one id — `Late Fees` and
`late-fees` are two nodes to the person editing the file and one to the slug —
and that is refused, naming both.

`tree` takes a list for a forest, and `child_key` / `name_key` take a file's own
spelling:

```python
MappingHierarchy.from_nested([{"name": "One"}, {"name": "Two"}])
MappingHierarchy.from_nested(doc, child_key="sub", name_key="title")
```

### From a live axis, walked once

`snapshot` copies an axis into a mapping: ids and edges, not content.

```python
live = AssertionHierarchy(onto.assertions, "isa")
copied = MappingHierarchy.snapshot(live)

assert copied.parents("beagle") == live.parents("beagle")
```

Walking `copied` afterwards asks the mapping rather than the backing. What that
buys is the one thing a live axis cannot do — say what has changed since it was
taken — and what it costs is freshness: a rebuild beneath `live` is invisible to
`copied`, deliberately.

The asynchronous twin is `async`, because this one really does read the axis:

```python
copied = await AsyncMappingHierarchy.snapshot(
    async_species.structure, max_concurrency=4
)
```

`max_concurrency` means exactly what it means on every other asynchronous entry
point here, and a snapshot walks the *whole* axis rather than one branch of it —
when it has to walk at all.

### How complete the copy is, is a property of the axis

`roots()` is not an extent, so the four members of `Hierarchy` offer exactly one
way to enumerate an axis: start at the roots and descend. A cyclic component
with nothing above it has no root, so a descent never reaches it and it is
absent from the copy.

That absence is not only a gap in the mapping. `Taxonomy.walk` refuses an anchor
its axis does not contain, so a taxonomy over such a copy **refuses a walk the
same taxonomy over the live axis performs** — a hole, not a shrink, and a
*semantic* difference rather than only a freshness one.

An axis that can say what it holds is not subject to that, and says so by
implementing `EnumerableHierarchy`:

```python
class EnumerableHierarchy(Hierarchy[K], Protocol):
    def parent_edges(self) -> Mapping[K, Sequence[K]]: ...
```

One entry per node, including a node that only ever appears as somebody's
parent, whose entry is empty. `snapshot` asks for it when it is there and
descends when it is not, so the same call gives the complete copy from a backing
that can enumerate and the root-reachable one from a backing that cannot:

```python
edges = {"dog": ("mammal",), "a": ("b",), "b": ("a",)}   # 'a' and 'b' are a
                                                         # cycle with no root

# a MappingHierarchy is holding its edges, so snapshot asks it for them
MappingHierarchy.snapshot(MappingHierarchy(edges)).contains("a")   # True

# a backing with only the four members — RowBackedAxis above — is descended
MappingHierarchy.snapshot(row_backed_axis).contains("a")           # False
```

`MappingHierarchy`, `AsyncMappingHierarchy`, `AssertionHierarchy` and
`AsyncAssertionHierarchy` all implement it — the first two hold their edges and
the second two fetch them in one query, which is the query `roots()` already
makes. Like `BulkHierarchy`, it is optional and separate for a reason: an axis
behind a paged API may genuinely have no way to answer, and a hand-written
hierarchy with only the four members stays a valid one.

## What a taxonomy carries, and what it refuses

`Taxonomy` has four fields: the `definition` it was declared by, the
`structure`, the `entities` source, and the `assertions` the edges were made of.

The last is optional, and `assertions is None` is a question worth asking: it
distinguishes *this edge carries no annotation* from *this axis has no
annotations to give*. A hierarchy built from a `parent_id` column has rows and
no assertions at all.

A definition may state a `materialization` per axis. Both defaults are the live
read, which is also what a file gets for declaring nothing:

```yaml
taxonomies:
  - id: species
    relation: isa
    materialization:
      structure: on_demand        # reads the edges through the assertion source
      content: on_demand          # reads the entities through the entity source
```

**`structure: materialized` is honoured; `content: materialized` is refused.**
The structure copy is ids and edges — cheap, and a loader door takes it for you:

```yaml
taxonomies:
  - id: species
    relation: isa
    materialization:
      structure: materialized
```

```python
species = onto.taxonomy("species")
type(species.structure)      # MappingHierarchy — the copy, not the live read
species.structure is onto.taxonomy("species").structure   # True
```

The copy is taken **once, at load**, and that location is forced rather than
chosen. `taxonomy()` is a plain `def` on both flavours, so there is nowhere in
it to await `AsyncMappingHierarchy.snapshot`; and it builds afresh on every
call, so a copy taken there would be a new copy with a new build time each time
you asked — the one property a copy exists not to have.

Because the axis being copied can enumerate its edges, the copy is the whole
axis rather than the part a descent reaches: a mutual `isa` survives being
materialized. That is the [`EnumerableHierarchy`](#how-complete-the-copy-is-is-a-property-of-the-axis)
route above, arriving where it matters.

`content: materialized` is a copy of every entity the axis covers, and it needs
somewhere to live — a door loading a hand-edited file binds no such store. It is
refused naming the axis, at the call that asked for it, rather than at the first
read:

```python
onto.taxonomy("species")
# ValidationError: taxonomy 'species' declares `materialization.content:
# materialized`, which is a copy of every entity on the axis and needs a store
# to hold it; this ontology binds none. Use `content: on_demand`, which reads
# through the entity source
```

An `Ontology` built by hand rather than by a door carries no copies, so a
definition declaring `structure: materialized` there is refused too — pass the
axis in `structures={...}`, or declare `on_demand`. Substituting the live read
would hand back a different object under the name the config used.

An undeclared name is refused too, listing what *is* declared — the useful
answer to a typo being the set it was nearly one of:

```python
onto.taxonomy("speceis")
# NotFoundError: no taxonomy 'speceis' in this ontology. Declared: ['species']
```

Both refusals carry a `context` mapping (`taxonomy`, and `axis`/`declared`) for
a caller handling them programmatically.

## Keys that are not strings

The walks never inspect a node id — they only hash one — so `K` is a type
parameter bounded by `Hashable`, with `str` as its default:

```python
from dataknobs_common.hierarchy import Hierarchy, ancestors

class IntTree:
    """A hierarchy whose nodes are integers: n is the parent of 2n and 2n+1."""

    def roots(self) -> tuple[int, ...]:
        return (1,)

    def parents(self, node_id: int) -> tuple[int, ...]:
        return () if node_id <= 1 else (node_id // 2,)

    def children(self, node_id: int) -> tuple[int, ...]:
        return (2 * node_id, 2 * node_id + 1) if node_id < 8 else ()

    def contains(self, node_id: int) -> bool:
        return node_id >= 1

tree: Hierarchy[int] = IntTree()
above: tuple[int, ...] = ancestors(tree, 13)    # (6, 3, 1) — inferred from the hierarchy
```

`MappingHierarchy` is generic in `K` too, so a hierarchy over integers need not
be written out at all:

```python
from dataknobs_common.hierarchy import MappingHierarchy

powers: Hierarchy[int] = MappingHierarchy({2: (1,), 3: (1,), 4: (2,), 5: (2,)})
assert ancestors(powers, 4) == (2, 1)
```

`from_nested` is the exception, and it is keyed by `str` whatever it is called
on: a minted id is a slug, and a slug is a string.

Because the default is declared, a bare `Hierarchy` annotation still means
`Hierarchy[str]`, so existing annotations read as they always did rather than
silently widening to `Hierarchy[Any]`.

**A `Taxonomy` pins the key to `str`, and this is a boundary rather than an
oversight.** The parameter serves the walks and the module-level drivers, which
only ever hash a node id. A taxonomy is both axes at once, and its content axis
is an `EntitySource` addressed by `str` because an `Entity` has a `str` id — so
a generic structure axis inside one would let you hold an integer-keyed
hierarchy beside an entity lookup that cannot be asked about an integer. Widen
it and the content side has to move first, or explicitly stay behind.

!!! note "Python 3.12"

    Type-parameter defaults ([PEP 696](https://peps.python.org/pep-0696/)) are
    in `typing` only from 3.13, so on 3.12 this needs `typing-extensions` at
    runtime. `dataknobs-common` declares it under a
    `python_full_version < '3.13'` marker: a 3.13 install resolves to nothing,
    and the import disappears when the supported floor rises.

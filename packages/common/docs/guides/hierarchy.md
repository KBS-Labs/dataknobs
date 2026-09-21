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

Other packages implement the same four members over structures of their own,
which is the point of the protocol being four members wide.
`dataknobs_data`'s `TopicNodeHierarchy` is one: a `Hierarchy` over a heading or
cluster tree, keyed by a node's position because a `TopicNode` carries no id.
It is worth reading as a worked adapter — it shows what a structure has to
supply (four members and a key that is a value) and what it gets back (every
walk on this page, including the ones its own type never had).

**The walks, and the core they are written against.** `ancestors` and
`async_ancestors` are module-level functions generic over `Hierarchy`, not
methods on it — implementing the protocol earns every walk and overrides none.
`drive` and `async_drive` are that core exposed, which is what you write a walk
of your own against; `Ask`, `Member` and `Walk` are the three types its
protocol is spelled in.

**Still gaining members, and on the door anyway.** `Taxonomy` and
`AsyncTaxonomy` carry two methods today, `walk()` and `at()`, out of the nine
they are planned to hold; the rest arrive in later releases. What ships now
will not change shape, and adding a member to a class breaks nobody — so the
promise a door makes is one the missing members do not put at risk. You rarely
need the taxonomy import at all, since `onto.taxonomy("species")` hands one
back; it is there so you can write the type down. `HierarchyView` and
`AsyncHierarchyView` are on the door for a stronger reason: nothing published
returns or constructs one, so the import is the only way to put a cursor over a
`Hierarchy` of your own. The cursor `at()` returns has a page of its own —
[The Anchored View](anchored-view.md).

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
from dataknobs_common import (
    AsyncHierarchyView,
    AsyncTaxonomy,
    HierarchyView,
    Taxonomy,
)
```

## Quick start — an axis of a vocabulary

A vocabulary declares an axis by naming the relation its edges are made of:

<!-- worked-input -->

```yaml
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
      attributes:
        - {name: lifespan_years, type: integer}
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

<!-- worked-call-site -->

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
`async_drive` its one round of concurrency per depth, and it holds for all
eight because all eight are readings of one descent. But `children(node_id)`
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
from dataknobs_common import ancestors, async_ancestors

assert ancestors(species.structure, "beagle") == ("dog", "mammal")
assert await async_ancestors(async_species.structure, "beagle") == ("dog", "mammal")
```

Two guarantees worth relying on: the visited set is unconditional, so a walk
terminates on cyclic data whatever an acyclicity constraint claims; and results
are deduplicated in walk order, so a DAG node reachable by several paths is
still one entry.

An empty result means **either a root or a node the axis does not have**, and
`ancestors` does not refuse the second — unlike every walk that *includes* its
anchor, which does. The difference is recoverability: an including walk emits
an unknown anchor as a term of the axis and the caller cannot detect it, where
`ancestors` excludes its anchor and returns nothing false. The answer is
ambiguous rather than wrong, and `contains()` resolves it in one call:

```python
assert ancestors(species.structure, "mammal") == ()      # a root
assert ancestors(species.structure, "marmoset") == ()    # not in the axis
assert species.structure.contains("mammal")
assert not species.structure.contains("marmoset")
```

### Paths, and the nearest node above two

`ancestors()` answers *what is above me* and deduplicates, so a node reachable
two ways appears once. `paths_to_root()` answers *how did I get here*, which is
a different question with a different shape — one tuple per route, the anchor
at index 0:

```python
from dataknobs_common import (
    MappingHierarchy, ancestors, deepest_common_ancestor, paths_to_root,
)

diamond = MappingHierarchy(
    {"x": ("a", "b"), "a": ("root",), "b": ("root",), "root": ()}
)

assert paths_to_root(diamond, "x") == (("x", "a", "root"), ("x", "b", "root"))
assert ancestors(diamond, "x") == ("a", "b", "root")     # one entry for `root`
```

**Its cycle guard is scoped to the path**, which is why it dedups differently
from every other walk here. Reachability is a property of a route: a node on two
routes is two answers, and a walk-scoped visited set would return one path where
two exist. A path ends where it cannot be extended — at a root, or at a node
whose every parent is already on that path — so **cyclic data returns paths that
reach no root**, and nothing is raised. Paths come back in parent order,
outermost first: every route through a node's first parent precedes every route
through its second.

The two endings look the same in the result, because the result is routes and
not a verdict about the axis. A caller counting *routes that reach a root* over
possibly-cyclic data asks: `parents(path[-1])` is empty for a root and
non-empty for a path that closed a cycle.

The cost is the ascent — one request per node above the anchor, one frontier
per level, like every other walk here — while the *answer* is the question's
own size: maximal routes double per stacked branch point, so a sixty-one node
axis can carry a million ways up.

**`max_paths` is the ceiling for that, and it refuses rather than truncating:**

```python
from dataknobs_common.exceptions import OperationError

branchy = MappingHierarchy({            # three stacked diamonds, eight routes
    "n0": (), "a0": ("n0",), "b0": ("n0",), "n1": ("a0", "b0"),
    "a1": ("n1",), "b1": ("n1",), "n2": ("a1", "b1"),
    "a2": ("n2",), "b2": ("n2",), "n3": ("a2", "b2"),
})

assert len(paths_to_root(branchy, "n3")) == 8
assert paths_to_root(branchy, "n3", max_paths=8) == paths_to_root(branchy, "n3")

try:
    paths_to_root(branchy, "n3", max_paths=4)
except OperationError as refused:
    assert refused.context == {"anchor": "n3", "max_paths": 4}
```

Returning the first four routes was the other candidate and is the worse one:
it collapses *there were exactly four ways up* into *there were at least four*,
which is the pair of answers `contains()` and the unknown-anchor refusal exist
to keep apart. So what comes back is every maximal route or nothing, and a
ceiling the axis stays under changes nothing at all.

It bounds the **routes** and not the depth, because depth is not what explodes:
an axis three levels deep whose every node has ten parents carries a thousand
routes. A bound on the *descent* would miss that, and would also truncate each
route at a node the walk never asked about — which is not *unextendable* but
*unknown*, so the result could no longer be read as the ways up from here.

A caller who cannot afford even a bounded enumeration wants membership rather
than routes, which `ancestors` answers over the same ascent for the size of the
axis.

`deepest_common_ancestor()` is the nearest node standing above two others, or
`None`:

```python
assert deepest_common_ancestor(diamond, "a", "b") == "root"
assert deepest_common_ancestor(diamond, "x", "a") == "a"   # either may be the answer
assert deepest_common_ancestor(diamond, "a", "x") == "a"
```

Each has an `async_` twin taking the same arguments, `max_concurrency`
included, and it binds in both: each walk reads a frontier per level.

**Deepest is not nearest, once the axis has a shortcut edge.** The answer is a
common ancestor with no *other* common ancestor standing below it — which
distance from the first argument cannot decide:

```python
shortcut = MappingHierarchy({
    "beagle": ("mammal", "hound"),   # asserted under both a broad and a narrow term
    "hound": ("dog",), "dog": ("canine",), "canine": ("mammal",),
    "puppy": ("dog",), "mammal": (),
})

assert deepest_common_ancestor(shortcut, "beagle", "puppy") == "dog"
assert ancestors(shortcut, "beagle")[0] == "mammal"   # nearest, and the wrong answer
```

`mammal` is **one** hop from `beagle` and `dog` is three, so *the nearest common
ancestor* answers `mammal` — while `dog` is a common ancestor of both arguments
standing strictly below it. A consumer reaching for the most specific shared
category would have been handed the least specific one.

The shape is ordinary rather than pathological: a term asserted under both a
narrow and a broad category is what a `broader` relation collects.

**The tie-break is asymmetric and is part of the definition.** Over a DAG
several common ancestors can be minimal and pairwise incomparable, with nothing
to choose between them on depth; the answer is then the first in the *first*
argument's own ancestry, nearest first, so swapping the arguments can swap the
answer. Over a tree there is one candidate and the asymmetry is invisible.

**Both of its arguments are anchors, and an unknown one is refused** — like
every walk here that can emit its anchor, which this one does in either
position. `None` means the two nodes have no common ancestor and nothing else.
`paths_to_root` refuses for the same reason: `(("x",),)` is exactly the shape a
root gives.

### Descending

Five walks go the other way, and the two guarantees above hold for all of them:
the visited set is unconditional and results are deduplicated in walk order.

```python
from dataknobs_common import (
    children_at_depth, descendants, descendants_to_depth, flatten, leaves,
)

assert descendants(species.structure, "dog") == ("retriever", "beagle")
assert flatten(species.structure, from_id="dog") == ("dog", "retriever", "beagle")
assert flatten(species.structure) == ("mammal", "dog", "retriever", "beagle")
assert descendants_to_depth(species.structure, "mammal", 1) == ("mammal", "dog")
assert children_at_depth(species.structure, "mammal", 2) == ("retriever", "beagle")
assert leaves(species.structure) == ("retriever", "beagle")
```

Each has an `async_` twin taking the same arguments. `flatten` and `leaves`
descend from every root when their anchor is omitted; the other three require
one.

#### What each one emits

**`descendants`, `flatten`, `descendants_to_depth` and `leaves` emit pre-order
by discovery**: each node, then everything first reached through it, then the
next. One rule across the bound — `max_depth` decides how far the walk goes and
never which order it comes back in, so a bounded answer is the unbounded one
cut short rather than a differently sorted one.

```python
from dataknobs_common import MappingHierarchy

tree = MappingHierarchy({
    "root": (), "a": ("root",), "b": ("root",),
    "a1": ("a",), "a2": ("a",), "b1": ("b",),
})

assert flatten(tree) == ("root", "a", "a1", "a2", "b", "b1")
#                                    ^^^^^^^^^^^^  a's subtree, before b
```

**`ancestors` is the exception, and it is not a preference.** It publishes
*nearest first*, which is a claim about distance: over a node with two parents
where a chain sits above only one of them, level order gives distances
1, 1, 2, 3 and pre-order gives 1, 2, 3, 1. The two promises conflict, so that
walk keeps level order.

**`children_at_depth` has no emission order to choose.** It returns one level,
and a level is a set of equals — what orders it is the order the backing
answered in. A depth the axis does not reach returns `()`, which is an answer
rather than a failure: nothing is that far below the anchor.

#### Pre-order by discovery is not depth-first

Over a tree the two coincide. Over a DAG they do not, and the name says which
one you get: a node reachable by several paths is emitted under whichever
*discovered it first*, and a level-at-a-time descent discovers it along the
shortest path rather than the leftmost one.

```python
dag = MappingHierarchy({
    "root": (), "a": ("root",), "b": ("root",), "y": ("a",), "x": ("b", "y"),
})

assert flatten(dag) == ("root", "a", "y", "b", "x")
#   x is under b, which reached it at depth 2 — not under y, which is depth 3.
#   A depth-first walk would return ("root", "a", "y", "x", "b").
```

The same distinction is why `leaves` reads the **reply** rather than the
discovery edges. A node that discovered nothing is either a leaf or a node
whose every child had already been reached along another path — and which one
it is is in the reply the descent already received:

```python
assert dag.children("y") == ("x",)     # y has a child
assert leaves(dag) == ("x",)           # ...and is not reported as a leaf
```

### Where the anchor is

Every walk that takes one, ascending and descending together, because the
answers do not follow from each other and do not follow from the direction:

| Walk | The anchor is |
|---|---|
| `ancestors`, `descendants` | **excluded** — they walk *away* from it |
| `flatten`, `descendants_to_depth` | **included** — the axis from a point, not the strict descendants of it |
| `children_at_depth` | **included at `depth=0`**, which is the anchor alone |
| `leaves` | **included if it is one** — a childless node is its own only leaf |
| `paths_to_root` | **included at the near end of every path** — a route from a node that omits the node is not a route from it |
| `deepest_common_ancestor` | **either argument may be the answer** — it returns one key rather than a sequence, and a node above the other is returned rather than passed over |

The last two rows are why the refusal is phrased over walks that *emit* their
anchor rather than over walks that return a sequence containing it.

`Taxonomy.subtree_keys()` below gives the **included** answer outside this
table: it includes its root deliberately, because *this and everything under
it* is the question a subtree filter is built to ask.

**A walk that includes its anchor refuses one the axis does not contain.** It
has to: a one-element result is exactly what a childless node returns, so an
unchecked anchor comes back as a term of the axis with nothing to distinguish
it from a real leaf.

```python
from dataknobs_common.exceptions import NotFoundError

for walk in (
    lambda: flatten(species.structure, from_id="marmoset"),
    lambda: descendants_to_depth(species.structure, "marmoset", 3),
    lambda: children_at_depth(species.structure, "marmoset", 0),
    lambda: leaves(species.structure, under="marmoset"),
    lambda: paths_to_root(species.structure, "marmoset"),
    lambda: deepest_common_ancestor(species.structure, "dog", "marmoset"),
):
    try:
        walk()
    except NotFoundError as refusal:
        assert refusal.context == {"anchor": "marmoset"}

assert descendants(species.structure, "marmoset") == ()   # excluding: ambiguous, not refused
assert ancestors(species.structure, "marmoset") == ()     # likewise
```

`deepest_common_ancestor` is on that list because **both** of its arguments are
anchors: it returns one key rather than a sequence and may return either
argument, so an unknown one would come back as a term of the axis. Answering
`None` looks like the excluding behaviour and is not — it is true of two
*different* unknown nodes and false of the same one twice, which comes back as
itself.

`leaves` is on that list by the longest route: an unknown anchor discovers
nothing, is confirmed childless because a backing has no children for a node it
has never heard of, and would be returned as a leaf of the axis. Omitting the
anchor refuses nothing — there is none to check, and an axis with no roots at
all is walked and returns `()`.

### Spending edge replies across walks

**No walk here asks its backing about a node twice**, and that is structural
rather than a tally. Every walk is a *reading* of one level-synchronous
descent: the descent visits a node once, keeps the reply it received, and the
reading issues no request of its own. A walk wanting an edge the descent
discarded is the one shape that breaks the rule — which is why the descent
keeps the replies rather than the spanning subset of them it used to, and why
the two readings that want routes and comparability find them in hand.

Two of the readings arrived as separate algorithms and were re-read as
projections, which is what the rule is worth stating for:

| Reading | What it wants that a spanning record cannot give |
|---|---|
| `leaves` | whether the reply was **empty** — over a DAG a node can discover nothing and still have children |
| `paths_to_root` | **every** edge — a node reached by two parents is two routes where it is one member |
| `deepest_common_ancestor` | the **induced subgraph** — *is this common ancestor above that one* is not answerable from a spanning tree |

So a caller's cache is for spending replies across **other** walks, in every
case and with no exception: a second walk over the same axis, a streaming walk
warmed by a collecting one, or the two walks `Taxonomy.subtree_keys()`
delegates to.

The descent's own promise is what the first row measures:

```python
class CountingAxis:
    def __init__(self, inner):
        self._inner, self.asked = inner, []

    def roots(self): return self._inner.roots()
    def parents(self, node_id): return self._inner.parents(node_id)
    def contains(self, node_id): return self._inner.contains(node_id)

    def children(self, node_id):
        self.asked.append(node_id)
        return self._inner.children(node_id)


axis = CountingAxis(tree)
leaves(axis)
assert sorted(axis.asked) == ["a", "a1", "a2", "b", "b1", "root"]   # each node once
```

Nothing is cached by default: for a walk that asks each node once, a per-walk
memo would hold a second copy of every reply the walk will never read back — a
20,000-node snapshot held one at 107% of what the walk returns — and the walk
that does re-ask keeps what it needs in its own frame. A caller who wants
replies to outlive one walk supplies their own:

```python
from collections.abc import Sequence
from typing import Any

from dataknobs_common import WalkCache, WalkCacheKey   # a Protocol: `get` and `__setitem__`
from dataknobs_common.bounded_cache import BoundedLRUCache

shared: BoundedLRUCache[WalkCacheKey, Sequence[Any]] = BoundedLRUCache(max_size=4096)
seam: WalkCache = shared        # it satisfies the seam without inheriting anything

flatten(axis, cache=shared)
flatten(axis, cache=shared)     # answered from `shared`, not from the backing
```

The value type is the reply type, and `Sequence[Any]` rather than `object` is
load-bearing: `get` is read covariantly, so a cache whose values are `object`
does not satisfy the seam it is about to be handed to. `hierarchy.py` carries
that assignment as a type-checked proof rather than as a claim.

`cache=` is a keyword on **every** walk here — `drive()`, `async_drive()`,
`ancestors`, the five descending walks, `paths_to_root`,
`deepest_common_ancestor`, both flavours of each, and `Taxonomy.walk()` and
`Taxonomy.subtree_keys()` below — because only the caller knows how fast their
data moves, and staleness is therefore theirs to decide.
Two members rather than `MutableMapping` so that a cache implementing the whole
mapping interface without inheriting the ABC — `BoundedLRUCache` is one — still
fits. It may also be **bounded**: nothing a walk does depends on a hit, so a
cache too small to hold a frontier costs re-fetches and never an answer.

A cache outliving one walk is scoped to **one axis**. The key is
`(member, node_id)` and names no hierarchy, so a cache spent on a second axis
answers it from the first's edges. There is no discriminator to put there: a
`Hierarchy` is arbitrary consumer code and need not be hashable — a plain
`@dataclass` backing has `__hash__` of `None` — while `id()` is reused after a
collection and would answer a new axis from a dead one's entries. One axis, one
cache.

`roots()` is deliberately **not** memoised: one walk asks it, once, and
remembering it would cost a caller their only chance to notice the axis grew a
root.

### The axis's own traversal

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
from dataknobs_common import NotFoundError

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

### Spending the axis on a query

A taxonomy earns its place when a query covers a *subtree*. If the answer is an
equality on the node the user named, a synonym list would have matched the
phrase equally well.

`subtree_keys()` is the member that makes the join expressible — the node and
everything under it, as keys:

```python
from dataknobs_data.query import Filter, Operator

assert species.subtree_keys("dog") == ["dog", "retriever", "beagle"]
assert species.subtree_keys("dog", depth=0) == ["dog"]

subtree_filter = Filter("species_id", Operator.IN, species.subtree_keys("dog"))
```

Five things to rely on:

* **the root is included.** Naming an interior node means *this and everything
  under it*, and an off-by-one here under-counts silently while the count is
  the whole answer;
* **deduplicated, in walk order** — a DAG node reachable by several paths is
  one key, not several, so a length a caller reports stays right;
* **`depth` is the bound and it is optional** — unbounded by default, because
  the ordinary question is *everything under this*;
* **an unknown root is refused**, for the reason `walk()` refuses one: this
  answer includes its anchor, so an unknown one would come back as a
  one-element list indistinguishable from a leaf;
* **the keys are the axis's own** — the filter above is right exactly when the
  axis and `species_id` are keyed alike, which is a property of how you *bound*
  the axis rather than of this call. `subtree_keys()` has never been told which
  table you are about to filter, so it does not translate; where the two spaces
  differ you hold both, and `split_qualified` is on the package door for it.
  It is also what lets a returned key go straight back in — as a `root_id`, to
  `at()`, or to `contains()` — which a translated one could not.

It is two delegations rather than an algorithm — `flatten` unbounded,
`descendants_to_depth` bounded — so it emits their pre-order at both ends of
the bound.

**So this package answers *everything at or under X* in two orders, and the
difference is not a preference.** `Taxonomy.walk()` streams and is breadth
first; `subtree_keys()` collects and is pre-order by discovery. A stream cannot
be pre-ordered with any bound on its lag: a node's place depends on everything
under its earlier siblings, so over a root whose first child leads a chain the
second child waits for the whole chain. Streaming it with a bounded lag means
one request per node instead of one per level — the round trip the shared
descent exists not to make. Pick by what you are doing rather than by order:
`subtree_keys()` builds a filter, where order does not survive the `IN` clause
anyway, and `walk()` is for consuming nodes as they arrive.

```python
assert species.subtree_keys("dog") == ["dog", "retriever", "beagle"]
assert tuple(species.walk(from_id="dog")) == ("dog", "retriever", "beagle")
#   the same here, because this axis branches in only one place; they differ
#   wherever a branch has a branch under it.
```

### Writing a walk of your own

`drive()` and `async_drive()` are the pair that makes a traversal
flavour-agnostic, and they are usable directly. A walk is a generator that
*yields a request* — a member name and the nodes to ask it about — and receives
one sequence per node asked about:

```python
from dataknobs_common import async_drive, drive

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
from dataknobs_common import DEFAULT_FRONTIER_CONCURRENCY

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
from dataknobs_common import AssertionHierarchy

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
from dataknobs_common import edge_criteria

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
from dataknobs_common import (
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

**Both flavours take `cache=` as well, and it is the same memo the walks take.**
A snapshot that descends is a walk, so a snapshot and a later walk over one
backing can share the replies instead of each paying for them:

```python
from dataknobs_common import MappingHierarchy, leaves

EDGES = {"mammal": (), "dog": ("mammal",), "beagle": ("dog",)}


class CountingAxis:
    # The four members and nothing else, so snapshot has to descend.
    def __init__(self):
        self.calls = 0

    def roots(self):
        return tuple(n for n, ps in EDGES.items() if not ps)

    def parents(self, node_id):
        return EDGES.get(node_id, ())

    def children(self, node_id):
        self.calls += 1
        return tuple(n for n, ps in EDGES.items() if node_id in ps)

    def contains(self, node_id):
        return node_id in EDGES


cold = CountingAxis()
MappingHierarchy.snapshot(cold)
leaves(cold)
assert cold.calls == 6                # three nodes, asked for children twice

warm, memo = CountingAxis(), {}
MappingHierarchy.snapshot(warm, cache=memo)
leaves(warm, cache=memo)
assert warm.calls == 3                # the second descent spent the memo
```

The memo reaches the descending branch only — a backing that can enumerate its
edges is asked for them in one call and has nothing to memoise, which is why
the axis above is `CountingParents` and not a `MappingHierarchy`. It is the
same keyword every walk carries, and the same one an anchored view's walk
members take: see [The Anchored View](anchored-view.md).

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

`Taxonomy` has five fields: the `definition` it was declared by, the
`structure`, the `entities` source, the `assertions` the edges were made of, and
the `entity_types` store the *type* lattice lives in — a different lattice from
the one `structure` walks, and what `inherited_attributes` reads.

`assertions` is optional, and the two readings of an empty `parent_edges()` —
*this edge carries no annotation* versus *this axis has no annotations to
give* — are separated by `has_edge_annotations()`. A hierarchy built from a
`parent_id` column has rows and no assertions at all.

Ask the member rather than `assertions is None`: a registry constructs an
assertion source for every vocabulary it loads, so a live axis has one that is
*empty* rather than *absent* and the field answers `False` for both kinds.
`has_edge_annotations()` asks what the vocabulary holds under the axis's
relation, which is what decides it.

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
from dataknobs_common import Hierarchy, ancestors

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
from dataknobs_common import MappingHierarchy

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

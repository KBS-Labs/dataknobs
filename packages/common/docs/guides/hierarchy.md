# Hierarchies and Taxonomies

`dataknobs_common.hierarchy` states what a *structure* is — what a node's
parents and children are — and the traversals over it.
`dataknobs_common.taxonomy` reifies one such structure as a walkable axis of a
vocabulary, and `dataknobs_common.ontology.hierarchy` backs one with the
assertions of a single relation.

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

Everything below is imported by module path. None of it is re-exported from
`dataknobs_common` or `dataknobs_common.ontology` yet — a name on a package
door is a name consumers hold, so it goes there once and deliberately. The
reason each name is still waiting is not the same reason, which is what you
need if you are deciding what to build on.

**Settled, waiting only on the door.** `Hierarchy`, `AsyncHierarchy`,
`BulkHierarchy`, `AsyncBulkHierarchy`, `ancestors`, `async_ancestors`,
`DEFAULT_FRONTIER_CONCURRENCY`, `AssertionHierarchy` and
`AsyncAssertionHierarchy` are complete, and their signatures are not expected
to move. They are absent from the door because the
release that opens it has not happened — not because anything about them is
unsettled.

**Complete, with the shape still open.** `drive` and `async_drive` do what this
page documents, and writing a walk of your own against them is what they are
for. A wider core — one that also drives a streaming pair — has been
prototyped and neither adopted nor rejected. Publishing the narrow form and
widening it later would change a signature consumers had already written
against, so it waits for that question to settle.

**Still gaining members.** `Taxonomy` and `AsyncTaxonomy` carry one method
today, `walk()`, out of the nine they are planned to hold; the rest arrive in
later releases. What ships now will not change shape.

```python
from dataknobs_common.hierarchy import (
    DEFAULT_FRONTIER_CONCURRENCY,
    AsyncBulkHierarchy,
    AsyncHierarchy,
    BulkHierarchy,
    Hierarchy,
    ancestors,
    async_ancestors,
    async_drive,
    drive,
)
from dataknobs_common.ontology.hierarchy import AssertionHierarchy, AsyncAssertionHierarchy
from dataknobs_common.taxonomy import AsyncTaxonomy, Taxonomy
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
is where the capability already existed. Concurrency does not substitute for
this: `gather` runs one round trip per node at the same time, while
`children_many` is one query for the level.

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

`parents(x)` is `find(subject=x, relation=…)` read for its entity objects;
`children(x)` is the mirrored query, `find(object=x, relation=…)` read for its
subjects. Because the source is the authority rather than a snapshot, a rebuild
beneath the structure is visible immediately — which is what lets a long-lived
axis hold the structure rather than a copy of it.

It lives under `ontology/` rather than beside the protocol it satisfies:
reading an edge means asking at runtime whether an assertion's object is an
entity or a literal, and the general module may not import the vocabulary
package to find out. A concrete goes where its dependency is — the same rule
that puts a database-backed hierarchy in `dataknobs-data`.

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

**Both snapshots are refused, for different reasons.** `content: materialized`
is a copy of every entity the axis covers and it needs somewhere to live — an
ontology loaded from a file binds no such store. `structure: materialized` is
the cheap copy, ids and edges, and what it lacks is not a store but an
implementation: the axis `taxonomy()` returns is an `AssertionHierarchy`, which
opens nothing and caches nothing, so it **is** the live read. Handing that back
to a definition that asked for a snapshot would be answering under the wrong
name, and a snapshot is wanted for the one thing a live axis cannot do — say
what has changed since it was taken.

So the structure refusal is the temporary one: it lifts when something builds a
snapshot. Either is refused naming the axis, at the call that asked for it,
rather than at the first walk:

```python
onto.taxonomy("species")
# ValidationError: taxonomy 'species' declares `materialization.content:
# materialized`, which is a copy of every entity on the axis and needs a store
# to hold it; this ontology binds none. Use `content: on_demand`, which reads
# through the entity source
```

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

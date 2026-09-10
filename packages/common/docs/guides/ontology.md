# The Authored Vocabulary

An ontology here is a file someone typed. It declares what kinds of thing
exist, what they are called, and what is true of them — and loading it gives
you a value you can read, walk and resolve against.

Nothing opens a connection, reads a second file, or awaits. A vocabulary
someone authored is dictionary lookups all the way down, which is why the
synchronous flavour is the native one and the asynchronous flavour is the
accommodation for a source that genuinely needs it.

## Where the names live

On the package door. Everything in this guide is importable from
`dataknobs_common` directly, and from this family's own door
`dataknobs_common.ontology` — the same names, so a consumer already spelling
the subpackage path does not have to change.

```python
from dataknobs_common.ontology import (
    AssertionHierarchy,
    Entity,
    Ontology,
    build_resolver,
    load_ontology,
)
```

Nine constructs are deliberately *not* on either door; see
[What is not on the door](#what-is-not-on-the-door) for which, and why.

## The whole input

Hand-edited, no tooling, no second file:

<!-- worked-input -->

```yaml
# mammals.yaml
ontology:
  id: mammals
  version: "1.0"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string, required: true}
    - id: Breed
      isa: Species

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal}
    - {id: dog,    type: Species, name: Dog, aliases: [Canine, "Domestic dog"]}
    - {id: beagle, type: Breed,   name: Beagle, aliases: [Beagles],
       source: {source_id: clinic_db, table: species, key: "sp-2291"}}

  assertions:
    - {subject: dog,    relation: isa, object: mammal}
    - {subject: beagle, relation: isa, object: dog}
```

## The worked call site

Five things a vocabulary is for, in the order someone meets them. Every line
below runs against the file above, exactly as written.

<!-- worked-call-site -->

```python
from pathlib import Path

from dataknobs_common.hierarchy import ancestors
from dataknobs_common.ontology import (
    AssertionHierarchy,
    build_resolver,
    load_ontology,
)

onto = load_ontology(Path("mammals.yaml"))  # -> Ontology
# No database. No embedder. No event loop.

# (1) the reader typed "beagles"
onto.entities.by_surface_form("beagles")  # -> frozenset({"beagle"})
beagle = onto.entities.get("beagle")  # -> Entity | None
assert beagle is not None  # an id that misses is a typo, not a result
beagle.name  # "Beagle"

# (2) what is it a kind of?
structure = AssertionHierarchy(onto.assertions, "isa")  # a Hierarchy over the isa edges
ancestors(structure, "beagle")  # ("dog", "mammal") -- module-level, over any Hierarchy

# (3) what does the vocabulary say about it?
onto.assertions.find(subject="beagle", relation="isa")  # -> [Assertion(...)]

# (4) leave with something spendable on your own data
beagle.source  # SourceRef(clinic_db, ...)
onto.entities.describe().capabilities  # frozenset() -- no ORIGIN_FETCH, so the
# reference is yours to spend and not ours to dereference

# (5) the same placement, ranked and with its reasons
resolver = build_resolver(Path("mammals.yaml"), onto)
result = resolver.resolve("beagles", k=5)
result.candidates[0].entity_id  # "beagle"
result.candidates[0].evidence[0].kind  # EvidenceKind.DECLARED
```

That block is executed as written by a workspace test, and the test asserts it
is character-identical to the fence above. If this page and the code ever
disagree, the suite goes red rather than the page going quietly wrong.

## Loading one

`load_ontology` takes a path or an already-parsed mapping and returns a pure
`Ontology`. `async_load_ontology` is its twin and returns an `AsyncOntology`.

```python
from pathlib import Path

from dataknobs_common.ontology import async_load_ontology, load_ontology

onto = load_ontology(Path("mammals.yaml"))
same = load_ontology({"id": "mammals", "entity_types": [{"id": "Species"}]})
```

Both accept a `normalizer=` keyword — a `Callable[[str], str]` folded over
surface forms before they are compared. `default_normalizer` is what they use
when you pass nothing.

Underneath, the document is validated into an `OntologyConfig` and assembled by
`build_ontology`, which returns an `OntologyParts` — the declared material,
before any source is constructed over it. Reach for those two when you want the
parse without the assembly; `load_ontology` is the whole path and is what most
callers want.

## What one holds

Ten fields, and the ones you read most are sources rather than containers:

| Field | What it is |
|---|---|
| `id`, `version` | the vocabulary's own identity |
| `entity_types`, `relation_types` | the declared kinds, as `EntityType` and `RelationType` |
| `entities` | an `EntitySource` — `get`, `get_many`, `by_surface_form`, `by_type`, `fetch_origin`, `describe` |
| `assertions` | an `AssertionSource` — `get`, `find`, `find_many` |
| `taxonomies` | `TaxonomyDefinition` per declared axis, keyed by its own id |
| `structures` | a materialized structure per axis, where one was asked for |
| `describes`, `imports` | what this vocabulary claims to describe, and what it pulls in |

The value types are ordinary frozen or field-wise dataclasses: `Entity`,
`EntityRef`, `Assertion`, `AttributeDef`, `Term` (an `EntityRef` or a
`Literal`). An `Assertion` carries a `Polarity` — `ASSERTED` or `NEGATED` — so
a vocabulary can state that something is *not* the case, and a walk can honour
it.

`Ontology` and `AsyncOntology` are frozen and compared by identity: they hold
the sources a vocabulary was built over, and two loads of one document are two
vocabularies rather than one repeated.

## Ids are local at the door

Inside a vocabulary an id is what the author typed — `beagle`, not
`mammals:beagle`. Qualification happens when two vocabularies meet, and the
helpers are explicit rather than automatic:

```python
from dataknobs_common.ontology import QualifiedId, qualify, split_qualified

qualify("mammals", "beagle")                # "mammals:beagle"
qualify("mammals", "sp-2291", "clinic_db")  # "mammals:clinic_db:sp-2291"
split_qualified("mammals:beagle")           # QualifiedId(ontology_id=..., source_id=..., local_id=...)
```

`RESERVED_ONTOLOGY_ID` is `dk`, the prefix this package keeps for itself:
`DK_ENTITY_TYPE` and `DK_RELATION_TYPE` are the two built-in types every
vocabulary has without declaring them. `ENTITY_TYPE_ISA_KEY` is the key an
entity type's own parent is declared under, and `DEFAULT_NESTED_RELATION` is
the relation a nested declaration means when it names none.

A relation can be written as a bare string or as a `RelationType`;
`relation_id` reduces either to the string, which is what every holder stores.

## Sources, and why they are protocols

`EntitySource` and `AssertionSource` are what an `Ontology` holds, and each has
an asynchronous twin. `MappingEntitySource` and `MappingAssertionSource` are
the concretes over material already in memory — what `load_ontology` builds —
and their async twins are `AsyncMappingEntitySource` and
`AsyncMappingAssertionSource`.

The point of the protocol is that a vocabulary loaded from a file and one
backed by your own table are the same value to everything downstream. What
differs is what a source can *tell you about itself*:

```python
from dataknobs_common.ontology import load_ontology

onto = load_ontology({"id": "mammals", "entity_types": [{"id": "Species"}]})
onto.entities.describe()   # SourceDescription(source_id='authored', backend='authored', ...)
```

A `SourceDescription` carries the backend, the table where there is one, the
projection, the `capabilities` the source has, and the entity types it
`declares`. `AUTHORED_SOURCE_ID` (`"authored"`) is the source id a hand-written
vocabulary gets, and `AUTHORED_SOURCE_KINDS` is the set of declaration shapes
that count as authored.

That is what makes step (4) of the call site honest. `beagle.source` is a
`SourceRef` naming `clinic_db`, and this vocabulary has no `clinic_db` behind
it — so the question worth asking is not *what does `fetch_origin` return?* but
*can this source reach an origin at all?* `Capability.ORIGIN_FETCH` in
`describe().capabilities` answers it before the call, which matters because
`fetch_origin` returning `None` cannot be told apart from a row that is simply
not there. An authored vocabulary carrying references into your production
table is the ordinary case, not a broken one: the reference is yours to spend
and not ours to dereference.

A `Provenance` records the other direction: where an assertion came from, who
asserted it and when.

### Optional protocols

Three capabilities a source may have and need not:

| Protocol | What it adds |
|---|---|
| `AliasFormSource` / `AsyncAliasFormSource` | `by_alias_form` — reverse lookup from a normalized form |
| `MembershipOracle` | `memberships` and `axes` — which scopes an entity belongs to |

A source that implements one is used through it; a source that does not is
used without it, and nothing degrades silently.

## Taxonomies and how a tree is projected

A `TaxonomyDefinition` names the relation its edges are made of, and carries
how that axis should be projected and whether it is written down:

- `TreeProjection` — `choice` (a `ParentChoice`, which picks one parent for a
  node declaring several), `on_cycle` (a `CyclePolicy`: `REPORT` or
  `BREAK_AT_REVISIT`), `order` (a `SiblingOrder`: `BY_NAME`, `BY_INSERTION`,
  `BY_METADATA` or `DECLARED`) and `order_key`.
- `Materialization` — `structure` and `content`, each an `InferenceMode`
  (`MATERIALIZED` or `ON_DEMAND`, and `ON_DEMAND` is the default for both).
- `ProjectionContext` — what a `ParentChoice` is handed when it is asked to
  decide: the taxonomy id, the relation, the roots, the depths and the types.

`EdgeCriteria` and `edge_criteria` are the narrowing a structural read applies:
a relation plus a polarity, so a walk follows asserted edges and not negated
ones. The walks themselves are in
[Walking a Structure](hierarchy.md), and the cursor `taxonomy().at()` returns
has its own page — [The Anchored View](anchored-view.md).

## What a resolution leaves behind

Placing a surface form is [its own guide](entity-resolution.md); three of its
value types live here because a vocabulary stores them:

- `ResolutionRef` — a resolution recorded rather than performed: the query, the
  entity it landed on, the score, the `Scoring` mode that produced it
  (`DECLARED`, `RANK_FUSED`, `NORMALIZED`, `NATIVE` or `DECAYED`), the corpus
  and the signals.
- `CompatibilityVerdict` — `COMPATIBLE`, `INCOMPATIBLE`, `UNVERIFIABLE` or
  `UNKNOWN`, which is how a resolution says whether the thing it matched was
  even the right kind of thing.

`build_resolver` is the bridge: hand it the same document and the loaded
vocabulary and it returns an `EntityResolver` composed from the document's own
`resolver:` section. A document that declares none gets the sensible default —
exact then alias, over the vocabulary's own entities — which is what makes step
(5) of the call site run against a file that never mentions resolution.

## What is not on the door

Nine constructs are reachable by module path and are on neither door. A module
path is not a claim of public API, which is the point: each of these still
gains members, and a name on a door is a promise.

| Construct | Where it is | Why it waits |
|---|---|---|
| `Taxonomy`, `AsyncTaxonomy` | `dataknobs_common.ontology.taxonomy` | still gaining members |
| `TaxonomyView`, `AsyncTaxonomyView` | `dataknobs_common.ontology.taxonomy` | still gaining members |
| `HierarchyView`, `AsyncHierarchyView` | `dataknobs_common.hierarchy` | still gaining members |
| `CascadeState`, `merge_rung`, `finish` | `dataknobs_common.entity_resolution.cascade` | internals, not the extension point |

None of them needs importing to be used. `onto.taxonomy("species")` returns a
`Taxonomy` and `.at(...)` returns a view; you only need the import to write an
annotation. The cascade internals are a different case — a consumer extends the
cascade by implementing `MatchSignal` and registering it, never by calling
`merge_rung`.

## Related

- [Walking a Structure](hierarchy.md) — the structural protocols and the walks
- [The Anchored View](anchored-view.md) — the cursor an axis hands back
- [Placing a Phrase Against a Vocabulary](entity-resolution.md) — the cascade

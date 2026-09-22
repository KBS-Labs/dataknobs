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

Three constructs are deliberately on neither of those two doors, and sit on the
resolution family's instead; see
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

Reading a `.yaml` path needs PyYAML, which `dataknobs-common` does not install
by default — `pip install dataknobs-common[yaml]`. Nothing else on this page
does: `load_ontology` also takes a `.json` path or a plain mapping, and either
runs on the base install.

## The worked call site

Five things a vocabulary is for, in the order someone meets them. Every line
below runs against the file above, exactly as written.

<!-- worked-call-site -->

```python
from pathlib import Path

from dataknobs_common import Capability, ancestors
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
capabilities = onto.entities.describe().capabilities
Capability.ORIGIN_FETCH in capabilities  # False -- the reference is yours to
# spend and not ours to dereference
Capability.SURFACE_FORM_LOOKUP in capabilities  # True -- step (1) is this one

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

**This is where a spelling convention is answered, and it is the only place.**
If your vocabulary and your queries spell the same name by different rules —
`gear_pump` against `gear pump` against `GearPump`, `C.D.C.` against `cdc` —
the fold has to apply to *both sides*, and the loader is what applies it to
both. No rung can do this: a rung probes the index as it is, and the index is
keyed on folded declared forms, so rewriting the query alone reaches nothing
when the form's own spelling needs rewriting too.

```python
import re

from dataknobs_common.entity_resolution import ExactNormalizedSignal
from dataknobs_common.ontology import load_ontology
from dataknobs_common.text import default_normalizer

parts = {
    "id": "parts",
    "entity_types": [{"id": "Part"}],
    "entities": [
        {"id": "gear_pump", "type": "Part", "name": "Gear Pump"},
        {"id": "gate_valve", "type": "Part", "name": "Gate Valve"},
    ],
}


def squash(form: str) -> str:
    """Delete every character that is not a letter or a digit, then casefold."""
    return re.sub(r"[^0-9a-z]", "", form.casefold())


for label, normalizer in (("default", default_normalizer), ("squash", squash)):
    rung = ExactNormalizedSignal(load_ontology(parts, normalizer=normalizer).entities)
    for query in ("Gear Pump", "gear-pump", "GearPump"):
        hit = rung.candidates(query, k=1)
        found = f"{hit[0].entity_id} ({hit[0].evidence[0].kind.value})" if hit else "-"
        print(f"{label:8} {query!r:13} {found}")
```

```
default  'Gear Pump'   gear_pump (declared)
default  'gear-pump'   -
default  'GearPump'    -
squash   'Gear Pump'   gear_pump (declared)
squash   'gear-pump'   gear_pump (declared)
squash   'GearPump'    gear_pump (declared)
```

The default already folds case and whitespace, which is why the first row
answers without anything being passed. The other two are the convention:
one function, applied to the declared forms at load and to the query at
probe time, and the hit that comes back is an ordinary **declared** one —
score `1.0`, `kind` `DECLARED`. That is the difference from
[`LexicalSignal`](entity-resolution.md#when-the-query-does-not-spell-the-form),
which measures how *near* two spellings are and answers `INFERRED`: the right
instrument for a typo, and the wrong one for a rule.

**The fold must be deterministic and total.** It is applied to every declared
form once at load and to every query at probe time, so a function that is not
a pure mapping from string to string makes the two sides disagree in a way
nothing reports.

Underneath, the document is validated into an `OntologyConfig` and assembled by
`build_ontology`, which returns an `OntologyParts` — the declared material,
before any source is constructed over it. Reach for those two when you want the
parse without the assembly; `load_ontology` is the whole path and is what most
callers want.

### Writing a door of your own

A door is `build_ontology` for the parse, sources of your own over the result,
and `assemble_ontology` (or `assemble_async_ontology`) to turn the two back
into a vocabulary:

```python
from dataknobs_common.ontology import assemble_async_ontology, build_ontology

parts = build_ontology(config)
entities = MySource(parts.declared_entities)
onto = await assemble_async_ontology(
    parts,
    entities=entities,
    assertions=MyAssertions(parts.declared_assertions),
    describes=(entities.describe(),),
)
```

The assembler is published because the third door in this workspace is in
another distribution — `OntologyRegistry`, in `dataknobs-data`, which binds
live sources and owns their lifecycle. Everything a vocabulary carries that is
not a bound source is the same for every door, so it is written once: a field
added to `Ontology` reaches all three without anyone threading it three times.

## What one holds

Ten fields, and the ones you read most are sources rather than containers:

| Field | What it is |
|---|---|
| `id`, `version` | the vocabulary's own identity |
| `entity_types`, `relation_types` | the declared kinds, as `EntityType` and `RelationType` |
| `entities` | an `EntitySource` — `get`, `get_many`, `by_surface_form`, `by_type`, `fetch_origin`, `fetch_origins`, `describe` |
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
vocabulary has without declaring them, and `DEFAULT_NESTED_RELATION` is the
relation a nested declaration means when it names none. An entity type's own
parent is `EntityType.isa` — a declared field, like `RelationType.inverse_of`,
rather than a key in the open `metadata` dict.

A relation can be written as a bare string or as a `RelationType`;
`relation_id` reduces either to the string, which is what every holder stores.

## What an entity id may be

**Whatever your records are keyed by.** An id is a `str` unless you say
otherwise, and every vocabulary loaded from a document is `str`-keyed, because
an author types strings. What changed is that the type is a parameter rather
than a pin, so a source over your own key works with the same axes, cursors and
walks:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Sku:
    plant: str
    line: int
```

Binding it costs one thing, and the type checker will not let you skip it: a
**`KeyCodec`**, which says how the key is written down when it leaves and read
back when it arrives.

```python
from dataknobs_common.ontology import KeyCodec, StrCodec, load_ontology


class SkuCodec:
    def to_id(self, key: Sku, /) -> str:
        return f"{key.plant}/{key.line}"

    def from_id(self, rendered: str, /) -> Sku:
        plant, _, line = rendered.rpartition("/")
        return Sku(plant=plant, line=int(line))


assert isinstance(SkuCodec(), KeyCodec)
```

**It has no default, and that is the point rather than an omission.** It cannot
be inferred: *a type that has a string representation* describes every type in
Python, so no bound can single out the ones that mean it. And it cannot be
`repr()`: a key is addressed by **equality**, while a default `repr` is a
function of **identity** — so two equal keys would render to two different
strings, one node would address two entities, and nothing would report it.

```python
class Opaque:
    def __init__(self, part): self.part = part
    def __eq__(self, other): return isinstance(other, Opaque) and other.part == self.part
    def __hash__(self): return hash(self.part)


first, second = Opaque("a-1"), Opaque("a-1")

assert first == second and hash(first) == hash(second)   # the structure axis is fine
assert repr(first) != repr(second)                       # the rendering is not
```

So the codec is a field an `Ontology` requires. `StrCodec` is the identity and
what `load_ontology` supplies, so **nothing you already wrote changes**: a
vocabulary from a document keeps its string ids, `onto.entity("beagle")` takes
the same argument it always took, and `qualify` / `localize` answer what they
always answered.

```python
vocabulary = load_ontology(
    {"id": "mammals", "entities": [{"id": "beagle", "type": "Species"}],
     "entity_types": [{"id": "Species"}]}
)

assert isinstance(vocabulary.codec, StrCodec)
assert vocabulary.qualify("beagle") == "mammals:beagle"
assert vocabulary.localize("mammals:beagle") == "beagle"
assert vocabulary.entity("beagle").name == "beagle"
```

The rendering happens only where an id **leaves** — `qualify` out, `localize`
back. Everything between simply carries the key: `Entity.id`, `Assertion.subject`,
every `entity_id` on a resolution. So a value type never reaches for a codec,
and a key that is not a `str` never becomes one by accident.

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

`fetch_origins` asks the same question for a sequence of refs, and answers
**positionally**: one slot per ref, in the order they were passed, with
`len(result) == len(refs)`. A ref that reached no row is a `None` in its own
slot.

The obvious signature — `dict[SourceRef, Record]` — is not one any
implementation can satisfy. `SourceRef` is compared field-wise, so that two
references naming one row are one reference, and its `locator` is a mapping;
a type that answers `Hashable` and then raises at the call is not a shape this
package ships. A positional answer loses nothing by comparison: the caller
already holds `refs`, so it can build any pairing it wants, while the misses a
mapping would have dropped are in the result where they happened.

```python
origins = await onto.entities.fetch_origins([first.source, second.source])
# [Record(...), None]  -- the second ref reached no row
```

The member exists for the round trips, not the rows: a live source answers N
refs in one read where a loop over `fetch_origin` pays N. A source that cannot
reach an origin at all answers all-`None` and withholds
`Capability.ORIGIN_FETCH`, which is the same thing `fetch_origin` says one ref
at a time.

A `Provenance` records the other direction: where an assertion came from, who
asserted it and when.

### Optional protocols

Two capabilities a source may have and need not, across three protocols:

| Protocol | What it adds |
|---|---|
| `AliasFormSource` / `AsyncAliasFormSource` | `by_alias_form` — reverse lookup from a normalized form |
| `MembershipOracle` | `memberships` and `axes` — which scopes an entity belongs to |

A source that implements one is used through it; a source that does not is
used without it, and nothing degrades silently.

### The one member that is partial

`by_surface_form` is on the protocol rather than beside it, and is the one
member a conforming source may decline. The fold is the **source's** —
a caller hands the query as it was typed — and a source reading a live table
holds the form as it was written, with no engine primitive folding at query
time the way `str.casefold` does. Such a source withholds
`Capability.SURFACE_FORM_LOOKUP` from `describe()` and raises
`CapabilityNotSupportedError` when asked anyway, because `frozenset()` already
means *ran and matched nothing* and a cascade falls through to a guessing rung
on exactly that reading.

An authored vocabulary declares the capability: its index folds every entity's
id, name and aliases when it is built, which is what the member answers over.

A rung reading the member declares `reads_surface_forms`, and refuses **at
construction** over a source that withholds it rather than carrying a call that
can only fail. `exact`, `scan` and `lexical` do; `alias` does not, because it
reads `by_alias_form` and a vocabulary genuinely may declare no aliases.

Both sources here answer the **capability contract**, so either question
reaches the same answer — the probe on the description, and the guard the
capability surface tells you to call:

```python
from dataknobs_common.capabilities import Capability, require_capability

require_capability(onto.entities, Capability.SURFACE_FORM_LOOKUP)
onto.entities.supported_capabilities()   # what this kind of source can do
onto.entities.instance_capabilities()    # what this one does — describe()'s set
```

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

## Two `isa` lattices, and the one that carries the schema

A vocabulary writes `isa` twice, and they are different stores:

| | Where it is written | What it relates |
|---|---|---|
| **instance** | `assertions: - {subject: beagle, relation: isa, object: dog}` | entities, and it is what a taxonomy's `structure` walks |
| **type** | `entity_types: - {id: Breed, isa: Species}` | *declarations*, and it is what attribute inheritance runs on |

**The second must not leak into the first.** If it did, walking up from
`beagle` would return `Species` beside `dog` — a schema node in a walk over
instances, which a consumer folding ancestors into a prompt has no way to spot.
It does not: an axis is built from the assertion store, and the type store is
read only by the member below.

`inherited_attributes(type_id)` is that member, and it is on **both** the
vocabulary and the axis. It walks the **type** lattice and returns what a type
may be asked for — its own declarations first, then each ancestor's:

```python
catalogue = load_ontology(
    {
        "id": "catalogue",
        "entity_types": [
            {
                "id": "Item",
                "attributes": [{"name": "sku", "type": "string"}, {"name": "weight", "type": "number"}],
            },
            {"id": "Product", "isa": "Item", "attributes": [{"name": "warranty", "type": "string"}]},
        ],
        "entities": [{"id": "widget", "type": "Product"}],
        "relation_types": [{"id": "isa"}],
        "taxonomies": [{"id": "kinds", "relation": "isa"}],
    }
)
kinds = catalogue.taxonomy("kinds")

assert [a.name for a in kinds.inherited_attributes("Product")] == ["warranty", "sku", "weight"]
```

**Ask the vocabulary directly when you are not already holding an axis.**
`Ontology.inherited_attributes` is the same walk over the same store, and it is
the surface to reach for first — the answer is a function of `entity_types` and
nothing else, so building an axis to ask would mean choosing a relation the
answer does not depend on:

```python
assert catalogue.inherited_attributes("Product") == kinds.inherited_attributes("Product")
```

Every taxonomy of one vocabulary therefore answers this identically, and
neither surface is a second implementation: both are one line over the shared
walk.

**A nearer declaration shadows a farther one of the same name**, because a
subtype redeclaring `sku` is specialising it rather than adding a second field.

**An undeclared type is refused rather than answered with `[]`.** A type
declared with nothing legitimately inherits nothing, so an empty list for a
type the store has never heard of would report a caller's typo as a fact about
their vocabulary:

```python
from dataknobs_common.exceptions import NotFoundError

try:
    kinds.inherited_attributes("NoSuchType")
except NotFoundError as refusal:
    assert refusal.context["entity_type"] == "NoSuchType"
```

The store is `Taxonomy.entity_types`, a plain mapping rather than a source —
a vocabulary's instances may be millions behind a backing, and its types are
tens, authored in the document. `onto.taxonomy()` fills it. An axis you build
by hand may leave it out, and then this member refuses every call, which is the
answer rather than a gap in it.

**It is a plain `def` on both asynchronous twins too**, because a mapping
awaits nothing.

## What a resolution leaves behind

Placing a surface form is [its own guide](entity-resolution.md); two of its
value types live here because a vocabulary stores them, and whatever those two
carry comes with them — `Scoring`, `EvidenceKind`, `MatchEvidence` and
`RunnerUp`:

- `ResolutionRef` — a resolution recorded rather than performed: the query, the
  entity it landed on, the score, the `Scoring` mode that produced it
  (`DECLARED`, `RANK_FUSED`, `NORMALIZED`, `NATIVE` or `DECAYED`), the rung of
  record and whether its match was `DECLARED` or `INFERRED`, where in the query
  it sat, the corpus, and the alternatives it ranked below.
- `CompatibilityVerdict` — `COMPATIBLE`, `INCOMPATIBLE`, `UNVERIFIABLE` or
  `UNKNOWN`, which is how a resolution says whether the thing it matched was
  even the right kind of thing.

A stored resolution is only worth storing if it can still be **judged**, which
is what `kind` is for: without it a declared alias and a vector neighbour are
the same row. Each entry in `runners_up` is a `RunnerUp` carrying its own
`MatchEvidence` for the same reason — a bare id and a number said which
alternatives existed and nothing about what any of them meant.

`build_resolver` is the bridge: hand it the same document and the loaded
vocabulary and it returns an `EntityResolver` composed from the document's own
`resolver:` section. A document that declares none gets the sensible default —
exact, then alias, then a scan that locates declared forms *inside* a longer
string, all over the vocabulary's own entities — which is what makes step (5)
of the call site run against a file that never mentions resolution.

## What is not on the door

`CascadeState`, `merge_rung` and `finish` are on the resolution family's own
door, `dataknobs_common.entity_resolution`, and not on this package's. That is
a placement rather than a promise deferred: they are the cascade's internals,
and being able to import one is not an invitation to build against it.

A consumer extends the cascade by implementing `MatchSignal` and registering it
in `signal_backends` — both exported here — never by calling `merge_rung`. That
advice holds whichever door the three sit behind, which is why it is the half
worth keeping.

**The taxonomy and cursor types used to be here and are now on the door.**
`Taxonomy`, `AsyncTaxonomy`, `TaxonomyView` and `AsyncTaxonomyView` are on this
guide's door and the package's; `HierarchyView` and `AsyncHierarchyView` are on
the package's. They still gain members, and publishing them anyway is the
deliberate part: adding a member to a class breaks nobody, while withholding
the name cost two different things. For the taxonomy types it cost an
annotation — `onto.taxonomy("species")` already hands you one, so the import
was only ever needed to write its type down. For the cursors it cost more than
that: nothing published returns or constructs a `HierarchyView`, so without the
import there is no way to get a cursor over a `Hierarchy` of your own at all.

## Related

- [Walking a Structure](hierarchy.md) — the structural protocols and the walks
- [The Anchored View](anchored-view.md) — the cursor an axis hands back
- [Placing a Phrase Against a Vocabulary](entity-resolution.md) — the cascade
